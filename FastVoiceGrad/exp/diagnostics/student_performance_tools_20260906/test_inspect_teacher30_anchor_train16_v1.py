"""仅在租用服务器、CUDA隐藏环境运行；使用小张量，不读取真实模型或音频。"""
import ast
import copy
import importlib.util
import inspect
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('train16_probe_under_test', HERE / 'inspect_teacher30_anchor_train16_v1.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def selection_fixture():
    rows = [{'split': 'train', 'speaker': 'p%d' % (i % 4), 'utt_id': 'u%d' % i, 'text': 'supplied text'} for i in range(84)]
    return rows, list(range(32)), {'retained_train_indices': list(range(32)), 'excluded_train_indices': list(range(32, 84))}, []


def parent_fixture():
    contract = {'configuration': {'lambda_feature_matching': 2.0, 'student_learning_rate': 5e-5}}
    identity = probe.canonical(contract)
    done = {'status': 'completed360', 'cleanup_verified': True, 'contract_sha256': identity,
        'state': {'completed_global_step': 4287, 'partial_batch_may_have_updated_D': False, 'execution_error': None}}
    payload = {'kind': 'MC2_FM2_independent360_full', 'global_step': 4287, 'epoch': 0, 'resumable': True,
        'run_contract_sha256': identity, 'execution_error': None, 'configuration': contract['configuration'], 'student': {'toy': 1}}
    return contract, done, payload


class StructureTests(unittest.TestCase):
    def test_approval_and_sha(self):
        common = ['--driver-sha256', 'a' * 64, '--tests-sha256', 'b' * 64, '--protocol-sha256', 'c' * 64]
        self.assertEqual(probe.arguments(['--phase', 'preflight'] + common).phase, 'preflight')
        with self.assertRaises(RuntimeError): probe.arguments(['--phase', 'infer'] + common)
        self.assertTrue(probe.arguments(['--phase', 'infer', '--run-approved', '--plan-sha256', 'd' * 64] + common).run_approved)
        with self.assertRaises(RuntimeError): probe.arguments(['--phase', 'preflight'] + common[:-1] + ['C' * 64])

    def test_selection_is_fixed_train_only_and_cross_speaker(self):
        fixture = selection_fixture(); chosen = probe.select_pairs(*fixture)
        self.assertEqual(chosen, probe.select_pairs(*fixture))
        self.assertEqual([r['source_index'] for r in chosen], list(range(16)))
        self.assertEqual([r['seed'] for r in chosen], list(range(probe.SEED_BASE, probe.SEED_BASE + 16)))
        for row in chosen:
            self.assertNotEqual(row['source']['speaker'], row['reference']['speaker'])
            self.assertLess(row['reference_index'], 32)

    def test_selection_rejects_invalid_partition_and_proof(self):
        rows, indices, dev, tests = selection_fixture()
        for wrong in (indices[:-1], [0] * 32, indices[:-1] + [32]):
            with self.assertRaises(RuntimeError): probe.select_pairs(rows, wrong, dev, tests)
        dev['excluded_train_indices'] = list(range(33, 84))
        with self.assertRaises(RuntimeError): probe.select_pairs(rows, indices, dev, tests)

    def test_selection_rejects_formal_overlap_and_insufficient_text(self):
        rows, indices, dev, _ = selection_fixture()
        with self.assertRaises(RuntimeError): probe.select_pairs(rows, indices, dev, [{'speaker': 'p0', 'utt_id': 'elsewhere'}])
        for row in rows[15:32]: row['text'] = ' '
        with self.assertRaises(RuntimeError): probe.select_pairs(rows, indices, dev, [])

    def test_numpy_headers_and_feature_axes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'header.npy'
            def header(dtype, shape):
                value = repr({'descr': dtype, 'fortran_order': False, 'shape': shape}).encode() + b'\n'
                path.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(value)) + value)
            header('<f4', (80, 7)); self.assertEqual(probe.npy_shape(path), (80, 7))
            header('|O', (80, 7))
            with self.assertRaises(RuntimeError): probe.npy_shape(path)
            header('<f4', (80, 0))
            with self.assertRaises(RuntimeError): probe.npy_shape(path)
        self.assertEqual(probe.frame_count((80, 7), 80), 7)
        self.assertEqual(probe.frame_count((7, 80), 80), 7)
        with self.assertRaises(RuntimeError): probe.frame_count((7, 8), 80)

    def test_real_parent_field_names_and_rejections(self):
        contract, done, payload = parent_fixture()
        with mock.patch.object(probe, 'PARENT_CANONICAL', probe.canonical(contract)):
            probe.validate_parent(contract, done, payload)
            for key, wrong in [('global_step', 4047), ('configuration', {}), ('kind', 'old'), ('resumable', False)]:
                altered = dict(payload); altered[key] = wrong
                with self.assertRaises(RuntimeError): probe.validate_parent(contract, done, altered)
            for key, wrong in [('completed_global_step', 4286), ('partial_batch_may_have_updated_D', True), ('execution_error', 'failure')]:
                altered = copy.deepcopy(done); altered['state'][key] = wrong
                with self.assertRaises(RuntimeError): probe.validate_parent(contract, altered, payload)

    def test_no_update_or_save_calls_and_correct_metric_identity_kind(self):
        tree = ast.parse(inspect.getsource(probe))
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('backward', 'step', 'save'): bad.append(node.func.attr)
        self.assertEqual(bad, [])
        self.assertEqual(probe.COUNTS, {'teacher': 600, 'student': 16, 'vocoder': 32})
        self.assertNotEqual(probe.METRIC_ID_FILE_SHA, probe.METRIC_ID_CANONICAL)
        self.assertEqual(probe.FIXED[str(probe.METRIC_ID_SOURCE)], probe.METRIC_ID_FILE_SHA)

    def test_budget_terminal_no_postcheck_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = probe.Budget(folder); path = Path(folder) / 'done.json'
            with mock.patch.object(budget, 'check', return_value={}) as check:
                budget.write(path, {'ok': True}, terminal=True)
                self.assertEqual(check.call_count, 1)
                with self.assertRaises(RuntimeError): budget.write(path, {}, terminal=True)
            with self.assertRaises(RuntimeError): budget.write(Path(folder).parent / 'outside.json', {})

    def test_no_checkpoint_output_and_wav_quota(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); budget = probe.Budget(folder)
            (path / 'illegal.pt').write_bytes(b'x')
            with self.assertRaises(RuntimeError): budget.usage()
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'small.wav').write_bytes(b'12')
            with mock.patch.object(probe, 'WAV_LIMIT', 1):
                with self.assertRaises(RuntimeError): probe.Budget(folder).check()

    def terminal_case(self, release_error=False, deadline_error=False):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        output = Path(folder.name); events = []
        class FakeBudget:
            def write(self, path, value, terminal=False):
                events.append(path.name)
                path.write_text(probe.json.dumps(value), encoding='utf-8')
        def close_lock():
            events.append('lock_close')
            if release_error: raise OSError('toy release failure')
        def check_clock():
            if deadline_error: raise TimeoutError('toy deadline')
        clock = SimpleNamespace(close=lambda: events.append('clock_close'), check=check_clock, started=time.monotonic())
        ledger = SimpleNamespace(close=lambda: events.append('ledger_close'), counts={'teacher': 600})
        args = SimpleNamespace(phase='infer', plan_sha256='a' * 64)
        with mock.patch.object(probe, 'OUTPUT', output):
            error = probe.finish_phase(args, FakeBudget(), clock, ledger, [SimpleNamespace(close=close_lock)], True, {}, None)
        return output, events, error

    def test_terminal_is_published_while_lock_owned(self):
        output, events, error = self.terminal_case()
        self.assertIsNone(error)
        self.assertLess(events.index('infer_completion.json'), events.index('lock_close'))
        self.assertTrue((output / 'infer_completion.json').exists())

    def test_release_failure_invalidates_only_new_completion(self):
        output, events, error = self.terminal_case(release_error=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse((output / 'infer_completion.json').exists())
        self.assertTrue((output / 'infer_completion_invalidated.json').exists())
        self.assertTrue((output / 'infer_failure.json').exists())

    def test_deadline_publication_failure_leaves_failure(self):
        output, events, error = self.terminal_case(deadline_error=True)
        self.assertIsInstance(error, TimeoutError)
        self.assertFalse((output / 'infer_completion.json').exists())
        self.assertTrue((output / 'infer_failure.json').exists())
        self.assertIn('lock_close', events)


class CPUModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not sys.platform.startswith('linux') or Path.cwd().resolve() != probe.ROOT:
            raise RuntimeError('Run these CPU workloads only in the rented project')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('Explicitly hide CUDA for these tests')
        sys.path.insert(0, str(probe.ROOT))
        import torch
        import numpy as np
        from fastvoicegrad import train as training
        from voicegrad.diffusion import VoiceGradDiffusion
        cls.torch, cls.np, cls.training, cls.Diffusion = torch, np, training, VoiceGradDiffusion
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.evaluator = probe.load_private('evaluate_development_candidates.py', probe.EVALUATOR_SHA)

    @classmethod
    def tearDownClass(cls): cls.torch.set_num_threads(cls.old_threads)

    def test_actual_capture_rng_schema_and_full_numpy_state(self):
        state = self.training.capture_rng_state()
        self.assertEqual(set(state), {'python', 'numpy', 'torch', 'cuda_all'})
        self.assertTrue(probe.same_rng(self.torch, self.np, state, copy.deepcopy(state)))
        changed = copy.deepcopy(state); values = list(changed['numpy']); values[2] = (values[2] + 1) % 624; changed['numpy'] = tuple(values)
        self.assertFalse(probe.same_rng(self.torch, self.np, state, changed))
        changed = copy.deepcopy(state); values = list(changed['numpy']); values[4] += 1; changed['numpy'] = tuple(values)
        self.assertFalse(probe.same_rng(self.torch, self.np, state, changed))
        changed = copy.deepcopy(state); changed['torch'][0] ^= 1
        self.assertFalse(probe.same_rng(self.torch, self.np, state, changed))

    def test_snapshot_detects_state_mode_requires_grad_and_grad(self):
        torch = self.torch; model = torch.nn.Linear(2, 2); model.register_buffer('toy_buffer', torch.ones(1))
        original = probe.snapshot(model)
        model.eval(); self.assertNotEqual(probe.snapshot(model), original); model.train()
        model.weight.requires_grad_(False); self.assertNotEqual(probe.snapshot(model), original); model.weight.requires_grad_(True)
        model.weight.grad = torch.zeros_like(model.weight); self.assertNotEqual(probe.snapshot(model), original); model.weight.grad = None
        model.toy_buffer.add_(1); self.assertNotEqual(probe.snapshot(model), original)
        model.toy_buffer.fill_(float('nan'))
        with self.assertRaises(RuntimeError): probe.snapshot(model)

    def test_forward_deadline_and_limit_precede_model_call(self):
        torch = self.torch; model = torch.nn.Identity(); entered = []
        model.register_forward_hook(lambda *unused: entered.append(True))
        def deadline(): raise TimeoutError('before forward')
        ledger = probe.Ledger(deadline, {'student': 1}); ledger.add(model, 'student')
        with self.assertRaises(TimeoutError): model(torch.ones(1))
        self.assertEqual(entered, []); self.assertEqual(ledger.counts['student'], 0)
        ledger.check = lambda: None; model(torch.ones(1))
        with self.assertRaises(RuntimeError): model(torch.ones(1))
        self.assertEqual(len(entered), 1); ledger.close(); model(torch.ones(1)); self.assertEqual(len(entered), 2)

    def pair(self, repeat=False, corrupt=False):
        torch = self.torch
        class TinyScore(torch.nn.Module):
            def forward(self, x, t, speaker, bnf): return x * 0.01
        class TinyVocoder(torch.nn.Module):
            def forward(self, mel): return mel.mean(dim=1, keepdim=True).repeat_interleave(256, dim=-1)
        teacher, student, vocoder = TinyScore(), TinyScore(), TinyVocoder()
        diffusion = self.Diffusion(1000, offset=0.008)
        source = torch.zeros(1, 80, 8); speaker = torch.zeros(1, 256); bnf = torch.zeros(1, 144, 8)
        mean, std = torch.zeros(1, 80, 1), torch.ones(1, 80, 1)
        counts = {'teacher': 60 if repeat else 30, 'student': 1, 'vocoder': 2}
        ledger = probe.Ledger(lambda: None, counts)
        for name, model in [('teacher', teacher), ('student', student), ('vocoder', vocoder)]: ledger.add(model, name)
        predict = self.evaluator.predict_candidate
        if corrupt:
            def predict(*args):
                prediction, q = self.evaluator.predict_candidate(*args)
                return prediction, q + 1
        before = self.training.capture_rng_state()
        try:
            result = probe.one_pair(torch, teacher, student, vocoder, diffusion, source, speaker, bnf,
                17, mean, std, predict, repeat_seed=1000017 if repeat else None)
            self.assertEqual(ledger.counts, counts)
            self.assertTrue(probe.same_rng(torch, self.np, before, self.training.capture_rng_state()))
            self.assertTrue(torch.equal(source, torch.zeros_like(source)))
            self.assertTrue(all(p.grad is None for model in (teacher, student, vocoder) for p in model.parameters()))
            return result
        finally: ledger.close()

    def test_actual_sampler_primary_30_and_student_share_q950(self):
        waves, record = self.pair()
        self.assertEqual(len(waves), 2)
        self.assertEqual([len(w) for w in waves], [2048, 2048])
        self.assertNotIn('teacher_repeat', record)
        self.assertTrue(self.np.isfinite(waves).all())

    def test_extra_teacher_repeat_has_no_extra_wave_or_primary_replacement(self):
        waves, record = self.pair(repeat=True)
        primary_waves, primary_record = self.pair()
        self.assertEqual(record['teacher_endpoint'], primary_record['teacher_endpoint'])
        self.assertTrue(self.np.array_equal(waves, primary_waves))
        self.assertEqual(record['teacher_repeat']['initial_q950'], record['initial_q950'])
        for key in ('primary_teacher_replaced', 'waveform_generated', 'metric_computed'):
            self.assertFalse(record['teacher_repeat'][key])

    def test_actual_q950_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'initial state'): self.pair(corrupt=True)


if __name__ == '__main__': unittest.main()
