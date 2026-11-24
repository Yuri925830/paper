"""租用服务器CUDA隐藏的小型合成测试；不读取真实模型或训练张量。"""
import importlib.util
import copy
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('CPU toy tests only on the registered rental with CUDA hidden')
import torch

HERE = Path(__file__).resolve().parent
def module(name):
    spec = importlib.util.spec_from_file_location('_toy_' + name, HERE / (name + '.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
p = module('inspect_wavlm_anchor_gradients4287_v1')
base = module('inspect_teacher30_anchor_train16_v1')


class SmallTests(unittest.TestCase):
    def fixture(self):
        rows = [{'speaker': 'p%03d' % (i % 8), 'utt_id': 'u%d' % i, 'split': 'train'} for i in range(84)]
        dev = {'retained_train_indices': list(range(32)), 'excluded_train_indices': list(range(32, 84))}
        return rows, list(range(32)), dev

    def test_first8_and_real_other_speaker(self):
        rows, indices, dev = self.fixture(); state = random.getstate()
        selected, perm = p.choose(rows, indices, dev, [{'speaker': 'test'}], p.shuffle_function())
        self.assertEqual([x['source_index'] for x in selected], list(range(8)))
        self.assertEqual(sorted(perm), list(range(32)))
        self.assertTrue(all(x['source_speaker'] != x['reference_speaker'] for x in selected))
        self.assertEqual(random.getstate(), state)
        self.assertEqual(perm, p.shuffle_function()([r['speaker'] for r in rows[:32]], 88007000))

    def test_excluded_source_rejected(self):
        rows, indices, dev = self.fixture(); indices[0] = 33
        with self.assertRaises(RuntimeError): p.choose(rows, indices, dev, [], p.shuffle_function())

    def test_test_speaker_reference_rejected(self):
        rows, indices, dev = self.fixture()
        with self.assertRaises(RuntimeError): p.choose(rows, indices, dev, [{'speaker': rows[31]['speaker']}], p.shuffle_function())

    def test_duplicate_and_bad_permutation_rejected(self):
        rows, indices, dev = self.fixture()
        with self.assertRaises(RuntimeError): p.choose(rows, indices, dev, [], lambda s, seed: list(range(32)))
        indices[1] = indices[0]
        with self.assertRaises(RuntimeError): p.choose(rows, indices, dev, [], p.shuffle_function())

    def test_unsatisfiable_mapping_has_no_fallback(self):
        with self.assertRaises(RuntimeError): p.shuffle_function()(['one'] * 32, 88007000)

    def test_full32_second_draw_private_rng_exact(self):
        core = p.load(p.MC, p.MC_SHA)
        full = torch.zeros(32, 80, 128); diffusion = types.SimpleNamespace(n_levels=1000)
        state = torch.get_rng_state().clone(); original_range = (core.START, core.STOP)
        noise, times = p.second32(core, full, diffusion)
        generator = torch.Generator().manual_seed(74004288)
        self.assertTrue(torch.equal(noise, torch.randn(full.shape, generator=generator)))
        self.assertTrue(torch.equal(times, torch.randint(1, 1001, (32,), generator=generator)))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(original_range, (core.START, core.STOP))
        again = p.second32(core, full, diffusion)
        self.assertTrue(torch.equal(noise[:8], again[0][:8])); self.assertTrue(torch.equal(times[:8], again[1][:8]))
        with self.assertRaises(RuntimeError): p.second32(core, full[:8], diffusion)

    def test_teacher30_exact_sameq_and_private_random(self):
        class Teacher(torch.nn.Module):
            def forward(self, x, t, speaker, bnf): return x + .01
        class Diffusion:
            def sample(self, model, q, speaker, bnf, *, num_steps, start_step, generator, diffuse_source):
                self.call = (num_steps, start_step, diffuse_source)
                result = q
                for i in range(num_steps): result = model(result, torch.full((len(q),), start_step - i), speaker, bnf)
                return result + torch.randn(q.shape, generator=generator) * .001
        teacher, diffusion = Teacher(), Diffusion(); q = torch.zeros(8, 80, 128)
        ledger = base.Ledger(lambda: None, {'teacher': 30}); ledger.add(teacher, 'teacher')
        state = torch.get_rng_state().clone()
        target, receipt = p.teacher_endpoint(torch, teacher, diffusion, q, torch.zeros(8, 256), torch.zeros(8, 144, 128), 88008000, base.tensor_id)
        self.assertEqual(diffusion.call, (30, 950, False)); self.assertEqual(ledger.counts, {'teacher': 30})
        self.assertEqual(receipt['first_input'][0], base.tensor_id(q)); self.assertFalse(target.requires_grad)
        self.assertTrue(torch.equal(state, torch.get_rng_state())); ledger.close()
        self.assertFalse(teacher._forward_pre_hooks)

    def test_teacher_wrong_initial_input_rejected_and_hook_removed(self):
        class Teacher(torch.nn.Module):
            def forward(self, x, t, speaker, bnf): return x
        class Bad:
            def sample(self, model, q, speaker, bnf, **kw): return model(q + 1, torch.full((len(q),), 950), speaker, bnf)
        teacher = Teacher()
        with self.assertRaises(RuntimeError): p.teacher_endpoint(torch, teacher, Bad(), torch.zeros(4, 80, 128), torch.zeros(4, 256), torch.zeros(4, 144, 128), 1, base.tensor_id)
        self.assertFalse(teacher._forward_pre_hooks)

    def test_calibration_direct_double_ratio_negative_cosine_allowed(self):
        s = {'norms': {'45MC2_conversion': 7.39201739284721, 'LMOS': 3.1251699829833},
             'cosines': {'LMOS': {'45MC2_conversion': -.4}}}
        result = p.calibration(s)
        self.assertEqual(result['anchor_weight'], s['norms']['45MC2_conversion'] / s['norms']['LMOS'])
        self.assertFalse(result['automatic_training_approval']); self.assertEqual(result['cosine'], -.4)

    def test_zero_nonfinite_calibration_rejected(self):
        for number in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(RuntimeError): p.calibration({'norms': {'45MC2_conversion': 2., 'LMOS': number}})

    def test_calibration_overflow_rejected(self):
        with self.assertRaises(RuntimeError):
            p.calibration({'norms': {'45MC2_conversion': 1e308, 'LMOS': 1e-308}})

    def test_official_asset_receipt_identity(self):
        receipt = {'status': 'completed_verified_official_conv_extraction',
            'source_revision': 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c',
            'source_sha256': 'fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d',
            'source_size_bytes': 1261990257, 'tensor_count': 21, 'parameter_count': 4206592,
            'cuda_initialized': False, 'forward_calls': 0, 'model_initializations': 0,
            'source_config_sha256': 'a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d',
            'output': {'path': str(p.ASSET), 'size_bytes': 16832166, 'sha256': p.ASSET_SHA, 'state_format': 'plain_conv_only_state_dict'}}
        p.validate_asset_receipt(receipt, p.ASSET_SHA)
        for key, changed in [('source_sha256', '0' * 64), ('tensor_count', 20), ('cuda_initialized', True), ('forward_calls', 1)]:
            bad = copy.deepcopy(receipt); bad[key] = changed
            with self.assertRaises(RuntimeError): p.validate_asset_receipt(bad, p.ASSET_SHA)

    def test_two_shared_graph_gradients_71_no_updates(self):
        parameters = {str(i): torch.nn.Parameter(torch.tensor(1. + i / 100)) for i in range(71)}
        output = sum(parameters.values()) * torch.ones(8, 2, 3)
        losses = {'LMOS': (output - 1).square().mean(), '45MC2_conversion': 45 * (output - 2).abs().mean()}
        before = {n: v.detach().clone() for n, v in parameters.items()}; banks = {}
        for label, loss in losses.items():
            values = torch.autograd.grad(loss, tuple(parameters.values()) + (output,), retain_graph=label == 'LMOS', allow_unused=True)
            banks[label] = {n: g.detach().cpu() for n, g in zip(parameters, values[:-1])}
            self.assertEqual(values[-1].shape, output.shape)
        stat = p.load(p.STATS, p.STATS_SHA).blockwise_statistics(torch=torch, banks=banks)
        self.assertEqual(stat['parameter_count'], 71); self.assertGreater(p.calibration(stat)['anchor_weight'], 0)
        self.assertTrue(all(v.grad is None and torch.equal(v, before[n]) for n, v in parameters.items()))

    def test_ledger_denies_excess_before_actual_forward(self):
        model = torch.nn.Linear(1, 1); ledger = base.Ledger(lambda: None, {'s': 1}); ledger.add(model, 's')
        model(torch.ones(1, 1))
        with self.assertRaises(RuntimeError): model(torch.ones(1, 1))
        self.assertEqual(ledger.counts, {'s': 1}); ledger.close()

    def test_budget_paths_quota_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            budget = p.Budget(root)
            with patch.object(p.os, 'statvfs', return_value=types.SimpleNamespace(f_bavail=4 * 1024**3, f_frsize=1)):
                budget.write('small.json', {'ok': True})
                with self.assertRaises(RuntimeError): budget.write('small.json', {'new': True})
                with self.assertRaises(RuntimeError): budget.write('../escape.json', {})
                with self.assertRaises(RuntimeError): budget.write('wave.wav', {})
                with self.assertRaises(RuntimeError): budget.write('large.json', {'x': 'a' * p.CAP})
            with patch.object(p.os, 'statvfs', return_value=types.SimpleNamespace(f_bavail=p.RESERVE, f_frsize=1)):
                with self.assertRaises(RuntimeError): budget.check()

    def test_finish_lock_failure_invalidates_completion(self):
        class BadLock:
            def close(self): raise OSError('toy lock close failure')
        with tempfile.TemporaryDirectory() as root, patch.object(p, 'OUTPUT', Path(root)):
            budget = p.Budget(root)
            clock = types.SimpleNamespace(started=p.time.monotonic(), check=lambda: None, close=lambda: None)
            ledger = types.SimpleNamespace(counts={'teacher': 62}, close=lambda: None)
            with patch.object(p.os, 'statvfs', return_value=types.SimpleNamespace(f_bavail=4 * 1024**3, f_frsize=1)):
                budget.write('results.json', {}); budget.write('run_contract.json', {})
                error = p.finish(base, budget, clock, ledger, [BadLock()], True, {'counts': ledger.counts}, {}, None)
            self.assertIsNotNone(error); self.assertFalse((Path(root) / 'completion.json').exists())
            self.assertTrue((Path(root) / 'completion_invalidated.json').exists()); self.assertTrue((Path(root) / 'failure.json').exists())

    def test_finish_existing_error_has_no_completion(self):
        with tempfile.TemporaryDirectory() as root, patch.object(p, 'OUTPUT', Path(root)):
            budget = p.Budget(root)
            clock = types.SimpleNamespace(started=p.time.monotonic(), check=lambda: None, close=lambda: None)
            ledger = types.SimpleNamespace(counts={}, close=lambda: None)
            with patch.object(p.os, 'statvfs', return_value=types.SimpleNamespace(f_bavail=4 * 1024**3, f_frsize=1)):
                error = p.finish(base, budget, clock, ledger, [], True, None, {}, RuntimeError('toy forward error'))
            self.assertIsNotNone(error); self.assertFalse((Path(root) / 'completion.json').exists())
            self.assertTrue((Path(root) / 'failure.json').exists())

    def test_no_cuda_initialized(self): self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__': unittest.main()
