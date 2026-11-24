"""在租用CPU验收固定3327正式推理身份、容量和批次恢复；不启动真实推理。"""

import ast
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on Matpool with CUDA_VISIBLE_DEVICES=''")
import numpy as np
import torch
import infer_formal_student3327_v1 as f


class Formal3327Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # 只核文件/清单；400项资产SHA、模型SHA及真实checkpoint字段均不跑网络。
        cls.plan=f.load_plan(True)
        cls.payload=f.load_torch_checkpoint(f.STUDENT)

    def fixture(self,root):
        output=root/'owned_output';identity={'output_dir':str(output),'system_id':f.SYSTEM,'student_global_step':3327}
        rows=[{'pair_id':str(i),'output_wav':str(output/'wav'/('pair_%04d.wav'%i))} for i in range(2)]
        samples={str(i):80+i*16 for i in range(2)}
        chunks=[[SimpleNamespace(pair_id=i) for i in range(2)]]
        f.prepare_output(output,identity,b'pair_id,output_wav\n')
        identity_sha=f.digest(output/'run_identity.json')
        prepared={'type':'prepared','chunk_head':0,'run_identity_sha256':identity_sha,'pairs':[]}
        for i,row in enumerate(rows):
            key=row['pair_id'];staged=output/'staging'/Path(row['output_wav']).name
            f.atomic_write_wav(staged,np.linspace(-.3,.3,samples[key],dtype=np.float32),22050)
            prepared['pairs'].append({'pair_id':key,'output_wav':row['output_wav'],'num_samples':samples[key],
                'wav_sha256':f.digest(staged),'seed':1234,'chunk_index':i,'chunk_size':2,
                'source_noise_sha256':hashlib.sha256(('noise'+key).encode()).hexdigest(),
                'x950_sha256':hashlib.sha256(('q950'+key).encode()).hexdigest()})
        return output,identity,rows,samples,chunks,identity_sha,prepared

    def test_actual_plan_all400_exact8100_and_byte_budget(self):
        plan=self.plan
        self.assertEqual(len(plan['test_input_file_sha256']),400)
        self.assertEqual(len(plan['rows']),8100)
        self.assertEqual(len(plan['chunks']),2100)
        self.assertEqual(sum(plan['expected_wav_bytes'].values()),1972847664)
        self.assertEqual(sum(plan['expected_allocations'].values()),1991651328)
        self.assertEqual(plan['allocation_unit'],4096)
        self.assertEqual(plan['actual_max_wav_allocation'],421888)
        self.assertEqual(plan['max_temporary_wav_allocation'],425984)
        self.assertEqual(f.required_free(1991651328),5382123520)
        self.assertEqual(f.TEACHER_PAIRS_SHA,f.digest(f.TEACHER_PAIRS))
        f.match_teacher_pairs(plan['rows'],f.csv_rows(f.TEACHER_PAIRS.read_bytes()))
        self.assertEqual([len(c) for c in plan['chunks']].count(1),100)

    def test_true3327_epoch0_contract_and_wrong_models_rejected(self):
        config=f.validate_payload(self.payload,self.plan['contract'],self.plan['completion'])
        self.assertEqual((config.student_steps,config.student_start_step),(1,950))
        for key,value in [('epoch',100),('global_step',3328),('run_contract_sha256','0'*64),('kind','epoch100'),('score_semantics','legacy')]:
            changed=dict(self.payload);changed[key]=value
            with self.assertRaises(RuntimeError):f.validate_payload(changed,self.plan['contract'],self.plan['completion'])
        changed=dict(self.payload);changed['effective_optimizer_lrs']={'student':.0002,'discriminator':.0002}
        with self.assertRaises(RuntimeError):f.validate_payload(changed,self.plan['contract'],self.plan['completion'])
        bad=copy.deepcopy(self.plan['completion']);bad['partial_batch_may_have_updated_D']=True
        with self.assertRaises(RuntimeError):f.validate_payload(self.payload,self.plan['contract'],bad)

    def test_teacher_pairs_cannot_be_reordered_cloned_or_retexted(self):
        old=f.csv_rows(f.TEACHER_PAIRS.read_bytes())
        rows=copy.deepcopy(self.plan['rows']);rows[0]['source_text']='an easier sentence'
        with self.assertRaises(RuntimeError):f.match_teacher_pairs(rows,old)
        rows=copy.deepcopy(self.plan['rows']);rows[0]['reference_wav']=rows[1]['reference_wav']
        with self.assertRaises(RuntimeError):f.match_teacher_pairs(rows,old)
        rows=copy.deepcopy(self.plan['rows']);rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(RuntimeError):f.match_teacher_pairs(rows,old)
        with self.assertRaises(RuntimeError):f.match_teacher_pairs(self.plan['rows'][:-1],old)

    def test_cli_only_explicit_fixed_candidate_not_epoch_noise_or_other_output(self):
        self.assertEqual(f.parse_args(['--run-approved']).max_seconds,10800)
        for args in ([],['--run-approved','--max-seconds','10801'],['--run-approved','--max-seconds','0'],
                     ['--run-approved','--output-dir','/tmp/cloned_formal'],['--run-approved','--checkpoint','epoch_0100.pt'],
                     ['--run-approved','--seed','4321'],['--run-approved','--limit','90']):
            with self.assertRaises(SystemExit):f.parse_args(args)

    def test_output_identity_clone_mismatch_and_partial_initialization_stop(self):
        with tempfile.TemporaryDirectory(prefix='formal3327_cpu_') as directory:
            root=Path(directory);out=root/'out';identity={'output_dir':str(out),'student_global_step':3327}
            f.prepare_output(out,identity,b'fixed pairs\n')
            before=(out/'run_identity.json').stat().st_mtime_ns
            f.prepare_output(out,identity,b'fixed pairs\n')
            self.assertEqual((out/'run_identity.json').stat().st_mtime_ns,before)
            with self.assertRaises(RuntimeError):f.prepare_output(root/'clone',identity,b'fixed pairs\n')
            changed=dict(identity,student_global_step=123000)
            with self.assertRaises(RuntimeError):f.prepare_output(out,changed,b'fixed pairs\n')
            partial=root/'partial';partial.mkdir();(partial/'unknown.tmp').write_bytes(b'broken')
            with self.assertRaises(RuntimeError):f.prepare_output(partial,{'output_dir':str(partial)},b'fixed')

    def test_orphan_staging_before_prepared_is_not_silently_deleted_or_recomputed(self):
        with tempfile.TemporaryDirectory(prefix='formal3327_cpu_') as directory:
            out,identity,rows,samples,chunks,sha,prepared=self.fixture(Path(directory))
            paths=list((out/'staging').iterdir());before={str(p):f.digest(p) for p in paths}
            with self.assertRaisesRegex(RuntimeError,'Orphan'):f.scan_journal(out,chunks,rows,samples,sha)
            self.assertEqual({str(p):f.digest(p) for p in paths},before)

    def test_prepared_and_partial_rename_recover_without_network_or_overwriting_valid_WAV(self):
        for renamed in (0,1,2):
            with tempfile.TemporaryDirectory(prefix='formal3327_cpu_') as directory:
                out,identity,rows,samples,chunks,sha,prepared=self.fixture(Path(directory))
                f.append_record(out/'wav_progress.jsonl',prepared)
                for record in prepared['pairs'][:renamed]:
                    final=Path(record['output_wav']);(out/'staging'/final.name).rename(final)
                valid_times={r['output_wav']:Path(r['output_wav']).stat().st_mtime_ns for r in prepared['pairs'][:renamed]}
                complete,pending,position=f.scan_journal(out,chunks,rows,samples,sha)
                self.assertEqual((complete,position),([],0));self.assertEqual(pending,prepared)
                with mock.patch.object(f.formal,'student_one_step',side_effect=AssertionError('No repeated inference')):
                    f.finish_prepared(out,pending)
                complete,pending,position=f.scan_journal(out,chunks,rows,samples,sha)
                self.assertEqual((len(complete),pending,position),(2,None,1))
                self.assertEqual({path:Path(path).stat().st_mtime_ns for path in valid_times},valid_times)
                self.assertEqual(list((out/'staging').iterdir()),[])

    def test_torn_or_wrong_commit_preserves_journal_for_review(self):
        for corruption in ('torn','wrongsha','missingpair','duplicatefile'):
            with tempfile.TemporaryDirectory(prefix='formal3327_cpu_') as directory:
                out,identity,rows,samples,chunks,sha,prepared=self.fixture(Path(directory))
                if corruption=='missingpair':prepared['pairs']=prepared['pairs'][:1]
                f.append_record(out/'wav_progress.jsonl',prepared)
                if corruption=='torn':
                    with (out/'wav_progress.jsonl').open('ab') as h:h.write(b'{"type": "comm')
                elif corruption=='wrongsha':f.append_record(out/'wav_progress.jsonl',{'type':'committed','prepared_sha256':'0'*64})
                elif corruption=='duplicatefile':
                    record=prepared['pairs'][0];source=out/'staging'/Path(record['output_wav']).name
                    Path(record['output_wav']).write_bytes(source.read_bytes())
                before=(out/'wav_progress.jsonl').read_bytes()
                with self.assertRaises(RuntimeError):f.scan_journal(out,chunks,rows,samples,sha)
                self.assertEqual((out/'wav_progress.jsonl').read_bytes(),before)

    def test_changed_noise_order_source_identity_WAV_or_missing_staged_file_is_rejected(self):
        for corruption in ('noise','seed','length','order','wav','missing'):
            with tempfile.TemporaryDirectory(prefix='formal3327_cpu_') as directory:
                out,identity,rows,samples,chunks,sha,prepared=self.fixture(Path(directory))
                if corruption=='noise':prepared['pairs'][0]['source_noise_sha256']='bad'
                elif corruption=='seed':prepared['pairs'][0]['seed']=55
                elif corruption=='length':prepared['pairs'][0]['num_samples']=99
                elif corruption=='order':prepared['pairs'].reverse()
                elif corruption=='wav':
                    p=out/'staging'/Path(rows[0]['output_wav']).name;p.write_bytes(b'not WAV')
                else:(out/'staging'/Path(rows[0]['output_wav']).name).unlink()
                f.append_record(out/'wav_progress.jsonl',prepared)
                with self.assertRaises((RuntimeError,ValueError)):f.scan_journal(out,chunks,rows,samples,sha)

    def test_exact_remaining_disk_budget_and_metadata_caps(self):
        total=f.required_free(f.EXPECTED_WAV_ALLOCATED)
        self.assertEqual(total,f.EXPECTED_START_FREE)
        written=425984;metadata=4096;directory=4096
        self.assertEqual(f.required_free(f.EXPECTED_WAV_ALLOCATED-written,metadata,directory),total-written-metadata-directory)
        with self.assertRaises(RuntimeError):f.required_free(0,f.METADATA_BUDGET+1)
        with self.assertRaises(RuntimeError):f.required_free(0,0,f.DIRECTORY_BUDGET+1)
        with tempfile.TemporaryDirectory(prefix='formal3327_cpu_') as directory:
            out=Path(directory)/'out'
            with mock.patch.object(f,'metadata_usage',return_value=(0,0)),mock.patch.object(f.os,'statvfs',return_value=SimpleNamespace(f_bavail=total-1,f_frsize=1)):
                with self.assertRaisesRegex(RuntimeError,'Insufficient disk'):f.capacity(out,f.EXPECTED_WAV_ALLOCATED,425984)
            with mock.patch.object(f,'metadata_usage',return_value=(0,0)),mock.patch.object(f.os,'statvfs',return_value=SimpleNamespace(f_bavail=total,f_frsize=1)):
                self.assertEqual(f.capacity(out,f.EXPECTED_WAV_ALLOCATED,425984)['required_bytes'],total)

    def test_frozen_source_hash_conflicts_and_changed_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='formal3327_cpu_') as directory:
            path=Path(directory)/'only_test_input';path.write_bytes(b'original feature bytes')
            files={str(path):f.digest(path)};f.verify_files(files)
            with self.assertRaises(RuntimeError):f.merge_hashes(files,{str(path):'0'*64})
            path.write_bytes(b'changed feature bytes')
            with self.assertRaises(RuntimeError):f.verify_files(files)

    def test_complete_readonly_API_binds_journal_WAV_epoch_and_collection_hash(self):
        # 这里只用两份极小WAV模拟发布状态；实际8100/400输入另有完整真实只读测试。
        with tempfile.TemporaryDirectory(prefix='formal3327_cpu_') as directory:
            out,identity,rows,samples,chunks,sha,prepared=self.fixture(Path(directory))
            pair_bytes=(out/'pairs.csv').read_bytes()
            identity.update(pairs_csv_sha256=hashlib.sha256(pair_bytes).hexdigest(),inference_driver_sha256=f.digest(Path(f.__file__)),
                file_sha256={},runtime=f.EXPECTED_RUNTIME,seed_rule='fixed test recipe',limitations=[])
            (out/'run_identity.json').write_bytes(f.json_bytes(identity));sha=f.digest(out/'run_identity.json')
            prepared['run_identity_sha256']=sha
            f.append_record(out/'wav_progress.jsonl',prepared);f.finish_prepared(out,prepared)
            plan={'chunks':chunks,'rows':rows,'pair_bytes':pair_bytes,'expected_samples':samples}
            with mock.patch.object(f,'OUTPUT',out),mock.patch.object(f,'PAIR_COUNT',2),mock.patch.object(f,'load_plan',return_value=plan),mock.patch.object(f,'make_identity',return_value=identity):
                report=f.readout_report(plan,identity,prepared['pairs'],.5,2,0)
                (out/'inference_report.json').write_bytes(f.json_bytes(report))
                bundle=f.validate_complete_output(out)
                self.assertEqual(bundle['report']['student_epoch'],0)
                self.assertEqual(bundle['report']['generated_wav_collection_sha256'],f.training.canonical_json_sha256(report['generated_wav_sha256']))
                before=(out/'inference_report.json').read_bytes()
                changed=dict(report,student_epoch=100);(out/'inference_report.json').write_bytes(f.json_bytes(changed))
                with self.assertRaises(RuntimeError):f.validate_complete_output(out)
                (out/'inference_report.json').write_bytes(before)
                # 合法header也不能替代实际WAV SHA身份。
                f.atomic_write_wav(Path(rows[0]['output_wav']),np.zeros(samples['0'],dtype=np.float32),22050)
                with self.assertRaises(RuntimeError):f.validate_complete_output(out)

    def test_identity_honestly_declares3327_epoch0_training_deviations_and_all_inputs(self):
        identity=f.make_identity(self.plan)
        self.assertEqual(identity['student_global_step'],3327)
        self.assertEqual(identity['student_epoch'],0)
        self.assertFalse(identity['completed_paper_training'])
        self.assertEqual(identity['effective_optimizer_lrs']['student'],.00005)
        self.assertEqual(identity['runtime'],f.EXPECTED_RUNTIME)
        self.assertEqual(identity['system_id'],'fastvoicegrad_student3327')
        self.assertEqual(len(identity['expected_samples_per_pair']),8100)
        self.assertEqual(identity['file_sha256'][str(f.FULL)],f.FULL_SHA)
        self.assertEqual(len(identity['test_input_file_sha256']),400)
        self.assertEqual(identity['batch_size'],4)

    def test_original_fixed_chunk_noise_recipe_matches_direct_seeded_tensor(self):
        # 小CPU张量检查每源尾批与噪声顺序；不声称CPU/GPU噪声字节相同。
        for chunk in (self.plan['chunks'][0],self.plan['chunks'][20],self.plan['chunks'][21]):
            seed=1234+int(chunk[0].pair_id);shape=(len(chunk),80,5)
            actual=torch.randn(shape,generator=torch.Generator(device='cpu').manual_seed(seed))
            repeated=torch.randn(shape,generator=torch.Generator(device='cpu').manual_seed(seed))
            self.assertTrue(torch.equal(actual,repeated))
            self.assertEqual(len({p.source_utt for p in chunk}),1)
        self.assertEqual(len(self.plan['chunks'][20]),1)
        self.assertEqual(self.plan['chunks'][21][0].pair_id,81)

    def test_source_preserves_frozen_helpers_and_has_no_train_or_model_save(self):
        source=Path(f.__file__).read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('save','save_checkpoint','train_batch','train_epoch','backward','step','load_student_checkpoint','resolve_inference_system'))
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('formal','teacher_infer','training','PAPER_STUDENT_CONFIG') for t in targets))
        self.assertIn('formal.student_one_step(model,diffusion,x,speaker,condition,noise)',source)
        self.assertIn('atomic_write_wav(path,np.clip(',source)
        self.assertIn('finish_prepared(OUTPUT,pending)',source)
        self.assertLess(source.index('finish_prepared(OUTPUT,pending)'),source.index("model=build_model('legacy_decimate')"))
        self.assertIn("for chunk in plan['chunks'][position:]:",source)


if __name__=='__main__':
    unittest.main()
