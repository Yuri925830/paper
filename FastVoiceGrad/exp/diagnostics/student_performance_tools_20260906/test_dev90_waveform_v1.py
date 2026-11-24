"""仅在租用机CPU运行：用可核算信号检查波形描述，不给真实音频重评分。"""

import copy
import contextlib
import csv
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import inspect_dev90_waveform_v1 as audit


class NumericalTests(unittest.TestCase):
    def test_periodic_hann_parseval(self):
        # rFFT总能量应为1024乘以加周期窗后的时域平方和。
        x = np.random.RandomState(123).normal(size=(3, 1024))
        band, fractions, total = audit.spectrum(x)
        w = .5-.5*np.cos(2*np.pi*np.arange(1024)/1024)
        expected = 1024*np.sum((x*w)**2)
        self.assertAlmostEqual(total/expected, 1., places=14)
        self.assertAlmostEqual(sum(fractions), 1., places=14)
        symmetric = 1024*np.sum((x*np.hanning(1024))**2)
        self.assertGreater(abs(total-symmetric), 1.)

    def test_known_sine_each_band(self):
        for index, bin_number in enumerate((20, 100, 240, 450)):
            x = np.sin(2*np.pi*bin_number*np.arange(1024)/1024)[None, :]
            energy, fractions, total = audit.spectrum(x)
            self.assertAlmostEqual(fractions[index], 1., places=13)
            self.assertAlmostEqual(total, 1024**2*3/16, places=8)

    def test_dc_and_nyquist_not_doubled(self):
        for x in (np.ones((1, 1024)), (-1.)**np.arange(1024)[None, :]):
            self.assertAlmostEqual(audit.spectrum(x)[2], 1024**2*3/8, places=8)

    def test_exact_frequency_boundaries(self):
        freq = np.array([0, 999, 1000, 3999, 4000, 7999, 8000, 11025])
        masks = audit.band_masks(freq)
        np.testing.assert_array_equal(np.argmax(masks, axis=0), [0,0,1,1,2,2,3,3])
        with self.assertRaises(RuntimeError): audit.band_masks(np.array([11026]))

    def test_frames_no_padding_or_shift(self):
        x = np.arange(2049, dtype=float)
        frames = audit.frames_of(x)
        self.assertEqual(frames.shape, (5, 1024))
        np.testing.assert_array_equal(frames[:, 0], [0,256,512,768,1024])
        self.assertEqual(frames[-1, -1], 2047)
        with self.assertRaises(RuntimeError): audit.frames_of(np.ones(1023))
        with self.assertRaises(RuntimeError): audit.frames_of(np.full(1024, np.nan))

    def test_source_mask_strict_threshold(self):
        # 直接给出已知帧，确保恰在阈值处的帧不会被选入。
        frames = np.array([[1.]*1024, [.01]*1024, [.009]*1024, [0.]*1024])
        with mock.patch.object(audit, 'frames_of', return_value=frames):
            mask = audit.source_mask(np.ones(1024))
        self.assertEqual(mask['low_energy_mask'], [False, False, True, True])

    def test_common_mask_and_amplitude_scaling(self):
        source = np.r_[np.zeros(2048), np.ones(2048)]
        mask = audit.source_mask(source)
        frozen = copy.deepcopy(mask)
        x = np.r_[np.full(2048, .02), np.full(2048, .2)]
        first = audit.waveform_features(x, mask)
        second = audit.waveform_features(x*2, mask)
        self.assertEqual(mask, frozen)
        for key in ('peak', 'rms', 'low_frame_rms', 'other_frame_rms'):
            self.assertAlmostEqual(second['features'][key], 2*first['features'][key])
        self.assertAlmostEqual(first['features']['low_to_other_db'], second['features']['low_to_other_db'])
        self.assertAlmostEqual(second['features']['spectral_energy_total'], 4*first['features']['spectral_energy_total'])
        for band in audit.BANDS:
            self.assertAlmostEqual(first['features']['spectral_fraction_'+band], second['features']['spectral_fraction_'+band])

    def test_zero_source_undefined(self):
        x = np.zeros(2048)
        mask = audit.source_mask(x)
        result = audit.waveform_features(x, mask)
        self.assertIsNone(mask['low_energy_mask'])
        self.assertIsNone(result['features']['low_frame_rms'])
        self.assertIsNone(result['features']['spectral_fraction_0_1000'])
        self.assertEqual(result['spectral_fraction_undefined_reason'], 'all_spectral_energy_zero')

    def test_empty_or_zero_groups(self):
        x = np.ones(2048)
        mask = audit.source_mask(x)
        result = audit.waveform_features(x, mask)
        self.assertIsNone(result['features']['low_frame_rms'])
        self.assertEqual(result['features']['other_frame_rms'], 1.)
        mask['low_energy_mask'] = [True]*5
        result = audit.waveform_features(x, mask)
        self.assertIsNone(result['features']['other_frame_rms'])
        mask['low_energy_mask'] = [True, False, False, False, False]
        result = audit.waveform_features(np.zeros(2048), mask)
        self.assertEqual(result['features']['low_frame_rms'], 0.)
        self.assertIsNone(result['features']['low_to_other_db'])

    def test_peak_boundary_and_tail(self):
        x = np.zeros(2049); x[0] = -.999; x[1] = .9991
        result = audit.waveform_features(x, audit.source_mask(x))['features']
        self.assertEqual(result['near_boundary_count'], 2)
        self.assertEqual(result['peak'], .9991)
        self.assertEqual(result['uncovered_tail_samples'], 1)
        self.assertAlmostEqual(result['rms'], math.sqrt((.999**2+.9991**2)/2049))

    def test_missing_values_not_zero(self):
        d = audit.describe([None, -2, 0, 4])
        self.assertEqual((d['valid'], d['missing'], d['positive'], d['negative'], d['zero']), (3,1,1,1,1))
        self.assertAlmostEqual(d['mean'], 2/3)
        self.assertIsNone(audit.describe([None])['mean'])
        with self.assertRaises(RuntimeError): audit.describe([float('inf')])
        with self.assertRaises(ValueError): audit.encode({'x': float('nan')})

    def test_all90_paired_grouping_and_corpus_cer(self):
        rows = []
        for source in range(10):
            for target in range(10):
                if source == target: continue
                i = len(rows)
                metric = dict.fromkeys(audit.METRIC_FIELDS, .2)
                metric.update(char_errors=source+1, reference_chars=20*(source+1), cer_percent=5.)
                rows.append(dict(pair_id=i, source_utt=str(source), reference_speaker=str(target),
                                 features={'x':float(i), 'null':None}, frame_mean_square=[1.,2.], metrics=metric))
        other = copy.deepcopy(rows)
        for row in other:
            row['features']['x'] -= 1
            row['metrics']['char_errors'] -= 1
        delta = [audit.differences(a,b) for a,b in zip(rows,other)]
        for row, original in zip(delta,rows): row['reference_chars'] = original['metrics']['reference_chars']
        grouped = audit.grouped(delta, True)
        self.assertEqual(grouped['all90']['n_pairs'], 90)
        self.assertEqual(grouped['all90']['features']['x']['positive'], 90)
        self.assertEqual(grouped['all90']['features']['null']['missing'], 90)
        self.assertEqual(len(grouped['source_utt']), 10)
        self.assertEqual(len(grouped['reference_speaker']), 10)
        self.assertTrue(all(v['n_pairs'] == 9 for v in grouped['source_utt'].values()))
        self.assertAlmostEqual(grouped['all90']['CER']['corpus_percent_difference'], 100*90/9900)
        self.assertEqual(audit.grouped(rows)['all90']['CER']['corpus_percent'], 5.)
        with self.assertRaises(RuntimeError): audit.differences(rows[0],other[1])


class IdentityAndSafetyTests(unittest.TestCase):
    def test_header_and_length(self):
        h = types.SimpleNamespace(samplerate=22050, channels=1, subtype='PCM_16', format='WAV', frames=2048)
        self.assertEqual(audit.check_header(h, 2048), 0)
        h.frames += 255
        self.assertEqual(audit.check_header(h, 2048, True), 255)
        with self.assertRaises(RuntimeError): audit.check_header(h, 2048)
        h.frames += 1
        with self.assertRaises(RuntimeError): audit.check_header(h, 2048, True)
        h.frames = 2048; h.samplerate = 16000
        with self.assertRaises(RuntimeError): audit.check_header(h, 2048)

    def test_csv_order_and_pair_identity(self):
        with tempfile.TemporaryDirectory(prefix='dev90_cpu_test_') as root:
            path = Path(root)/'pairs.csv'
            def write(order):
                with path.open('w', newline='') as f:
                    writer=csv.writer(f); writer.writerow(['pair_id','source']); writer.writerows((i,'s') for i in order)
            write(range(90)); rows = audit.rows90(path)
            altered = copy.deepcopy(rows); altered[0]['source'] = 'different'
            with self.assertRaises(RuntimeError): audit.pair_equal(altered, rows)
            altered = copy.deepcopy(rows); altered[0]['output_wav'] = 'different'
            audit.pair_equal(altered, rows)
            write(list(range(89))+[88])
            with self.assertRaises(RuntimeError): audit.rows90(path)

    def test_sha_and_conflicting_binding(self):
        with tempfile.TemporaryDirectory(prefix='dev90_cpu_test_') as root:
            path=Path(root)/'x'; path.write_bytes(b'fixture')
            files={}; audit.bind(files,path,audit.sha(path)); audit.verify(files)
            with self.assertRaises(RuntimeError): audit.bind(files,path,'0'*64)
            path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError): audit.verify(files)
            link=Path(root)/'link'; link.symlink_to(path)
            with self.assertRaises(RuntimeError): audit.sha(link)

    def test_illegal_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='dev90_cpu_test_') as root:
            parent=Path(root); out=parent/'new'
            audit.output_path(out,parent)
            with self.assertRaises(RuntimeError): audit.output_path(out/'nested',parent)
            Path(str(out)+'.log').touch()
            with self.assertRaises(RuntimeError): audit.output_path(out,parent)
            other=parent/'second'; other.mkdir()
            with self.assertRaises(RuntimeError): audit.output_path(other,parent)
            b=audit.Budget(other)
            with mock.patch.object(b,'check'):
                with self.assertRaises(RuntimeError): b.write('../bad.json', {})
                b.write('once.json', {'ok':True})
                with self.assertRaises(RuntimeError): b.write('once.json', {})

    def test_budget_and_deadline(self):
        b=audit.Budget(audit.OUTPUT)
        with mock.patch.object(audit.os,'statvfs',return_value=types.SimpleNamespace(f_bavail=1, f_frsize=1)):
            with self.assertRaises(RuntimeError): b.check()
        c=audit.Clock(seconds=-1)
        with self.assertRaises(audit.Interrupted): c.check()
        c=audit.Clock(); c.reason='SIGTERM'
        with self.assertRaises(audit.Interrupted): c.check()

    def test_actual_registered_plan_metadata_and_hash_only(self):
        # 此项只读原元数据并验SHA；禁止触发真实WAV解码或特征入口。
        with mock.patch.object(audit.sf,'read',side_effect=AssertionError('No waveform decode in CPU tests')):
            plan=audit.load_plan()
        self.assertEqual(len(plan['pairs']),90)
        self.assertEqual(len(plan['sources']),10)
        self.assertEqual(len(plan['systems']),3)
        self.assertTrue(all(len(s['metrics']) == 90 for s in plan['systems'].values()))
        self.assertEqual(audit.environment()['numpy'],'1.23.5')
        self.assertNotIn('torch',sys.modules)


class MainFlowTests(unittest.TestCase):
    def run_toy(self, fault=None):
        # 用临时文本身份文件和内存波形走完整270行流程，不读取真实音频样本。
        with tempfile.TemporaryDirectory(prefix='dev90_cpu_main_') as directory:
            root=Path(directory); output=root/'audit'; fixture=root/'input'; fixture.write_bytes(b'fixed')
            locks=root/'exp/locks'; locks.mkdir(parents=True)
            for name in audit.LOCK_NAMES: (locks/name).touch()
            digest=audit.sha(fixture)
            pairs=[]; sources={}
            for source in range(10):
                sources[str(source)]={'source_utt':str(source), 'source_speaker':str(source),
                    'source_wav':str(fixture), 'source_wav_sha256':digest, 'expected_num_samples':'4096'}
                for target in range(10):
                    if source == target: continue
                    pairs.append({'pair_id':str(len(pairs)), 'source_utt':str(source),
                                  'reference_speaker':str(target), 'output_wav':str(fixture), 'expected_num_samples':'4096'})
            systems={}
            for system in audit.SYSTEMS:
                metrics=[]
                for i in range(90):
                    metric=dict.fromkeys(audit.METRIC_FIELDS, .5)
                    metric.update(pair_id=i, char_errors=1, reference_chars=50, cer_percent=2.)
                    metrics.append(metric)
                systems[system]={'rows':pairs,'metrics':metrics,'inference':{}}
            plan={'files':{str(fixture):digest}, 'pairs':pairs, 'sources':sources, 'systems':systems}
            counts={'generated':0, 'source':0}
            wave=np.r_[np.zeros(2048),np.sin(2*np.pi*100*np.arange(2048)/1024)*.2]
            def read_wave(path,expected,source=False):
                key='source' if source else 'generated'; counts[key]+=1
                if fault == 'midway' and not source and counts[key] == 5: raise RuntimeError('Injected decode failure')
                if not source and counts[key] == 270:
                    if fault == 'identity': (output/'run_identity.json').write_text('{}')
                    if fault == 'dependency': fixture.write_bytes(b'changed')
                return wave.copy(),0
            original_close=audit.Budget.close
            original_output_path=audit.output_path
            def failing_close(budget):
                original_close(budget)
                raise OSError('Injected close failure')
            with contextlib.ExitStack() as stack:
                for name,value in [('ROOT',root),('OUTPUT',output)]: stack.enter_context(mock.patch.object(audit,name,value))
                stack.enter_context(mock.patch.object(audit,'environment',return_value={'synthetic':True}))
                stack.enter_context(mock.patch.object(audit,'watchdog',return_value={'seconds':300}))
                stack.enter_context(mock.patch.object(audit,'output_path',side_effect=lambda path:original_output_path(path,root)))
                stack.enter_context(mock.patch.object(audit,'load_plan',return_value=plan))
                stack.enter_context(mock.patch.object(audit,'read_wave',side_effect=read_wave))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if fault == 'close': stack.enter_context(mock.patch.object(audit.Budget,'close',failing_close))
                if fault:
                    with self.assertRaises((RuntimeError,OSError)): audit.main()
                else: audit.main()
            if fault:
                self.assertFalse((output/'results.json').exists())
                failure=json.loads((output/'failure.json').read_text())
                self.assertEqual(failure['status'],'failed')
                if fault == 'close':
                    self.assertTrue((output/'results.invalidated.json').exists())
                    self.assertEqual(failure['cleanup_errors'][0]['action'],'log_close')
                if fault == 'midway': self.assertEqual(failure['generated_rows_completed'],4)
            else:
                result=json.loads((output/'results.json').read_text())
                self.assertEqual(result['status'],'dev90_waveform_readonly_complete')
                self.assertEqual(result['run_identity_file_sha256'],audit.sha(output/'run_identity.json'))
                rows=[json.loads(line) for line in (output/'waveforms.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),270)
                self.assertEqual(counts,{'generated':270,'source':10})
                self.assertEqual(len(json.loads((output/'source_frames.json').read_text())),10)
                for contrast in result['candidate_minus_references'].values():
                    self.assertEqual(len(contrast['all90']),90)
                    self.assertEqual(contrast['summaries']['all90']['features']['rms']['zero'],90)
            # finally路径后，全部真实临时文件锁均能重新独占取得。
            for name in audit.LOCK_NAMES:
                with (locks/name).open('r+') as f:
                    audit.fcntl.flock(f.fileno(),audit.fcntl.LOCK_EX|audit.fcntl.LOCK_NB)

    def test_complete_toy270(self): self.run_toy()
    def test_midway_failure_no_complete(self): self.run_toy('midway')
    def test_final_dependency_mutation_no_complete(self): self.run_toy('dependency')
    def test_identity_mutation_no_complete(self): self.run_toy('identity')
    def test_close_failure_invalidates_completion(self): self.run_toy('close')

    def test_cleanup_attempts_every_action(self):
        budget=mock.Mock(); budget.close.side_effect=OSError('log')
        first=mock.Mock(); first.close.side_effect=OSError('lock')
        second=mock.Mock()
        with mock.patch.object(audit.signal,'signal') as restore:
            errors=audit.cleanup(budget,[first,second],{2:None,15:None})
        self.assertEqual(len(errors),2)
        self.assertEqual(restore.call_count,2)
        first.close.assert_called_once(); second.close.assert_called_once()


if __name__ == '__main__':
    if not sys.platform.startswith('linux') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run tests on rented Matpool CPU with CUDA_VISIBLE_DEVICES empty')
    unittest.main(verbosity=2)
