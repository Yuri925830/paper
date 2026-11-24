"""仅在租用服务器CPU上测试新四格薄层；不加载真实模型，不执行GPU主入口。"""

import ast
import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

TOOLS = Path('/root/FsatVoiceGrad/exp/diagnostics/student_performance_tools_20260906')
FILE = TOOLS / 'inspect_acdd_mc2_fm_cross_3447_3927_v1.py'
spec = importlib.util.spec_from_file_location('new_fm_cross_probe', FILE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ToyBranch(torch.nn.Module):
    # 每支只做小张量乘法，保留真实接口和6层，但不冒充真实模型性能。
    def __init__(self, index):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1. + .05 * index))
        self.register_buffer('fixed_window', torch.ones(1), persistent=False)
        self.period = (2, 3, 5, 7, 11)[index] if index < 5 else None
        self.resolution = SimpleNamespace(fft_size=512, hop_size=50, win_length=240)
        self.lrelu_slope = .2

    def forward(self, value):
        if torch.is_grad_enabled() or self.training or self.weight.requires_grad:
            raise RuntimeError('Toy must use the declared fully read-only path')
        features = [value * self.weight * (index + 1) / 10 for index in range(6)]
        return features[-1].flatten(1), features


class ToyD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = torch.nn.ModuleList(ToyBranch(index) for index in range(8))
        self.eval().requires_grad_(False)

    def loss_branches(self): return iter(self.branches)
    def forward(self, value): raise AssertionError('Top-level discriminator is forbidden')


class FourCellThinLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_available(): raise RuntimeError('Run these tests with CUDA_VISIBLE_DEVICES empty')

    def toy_states(self, model):
        first = {key: value.clone() for key, value in model.state_dict().items()}
        return {3447: {'discriminator': first},
                3927: {'discriminator': {key: value * 1.25 for key, value in first.items()}}}

    def receipt(self):
        return {'models_unchanged': False, 'rng_restored': False, 'flags_unchanged_and_restored': False}

    def test_complete_four_cells_192_layers_exact_48_D_and_scale(self):
        model = ToyD(); states = self.toy_states(model); check = Mock()
        ledger = probe.Ledger(check); ledger.add(model, 'D_top')
        for branch in model.loss_branches(): ledger.add(branch, 'D_branch')
        real = torch.linspace(.1, .9, 32 * 8).reshape(32, 1, 8)
        remembered, progress = {}, []
        def remember(name, model, fingerprint, label): remembered[label] = fingerprint
        try:
            matrix, summary = probe.crossed_forward(model, states, real, {3447: real * .8, 3927: real * .9},
                                                   check, remember, progress.append)
            self.assertEqual(ledger.counts, {'student': 0, 'vocoder': 0, 'D_branch': 48, 'D_top': 0})
            self.assertEqual(len(progress), 32)
            self.assertEqual(len(matrix), 4)
            self.assertEqual(sum(len(row['layers']) for cell in matrix.values() for row in cell['branches']), 192)
            self.assertEqual(len(summary['layers']), 48)
            self.assertEqual(len(summary['real_activation_scale_ratios']), 48)
            self.assertEqual(set(remembered), {'D3447', 'D3927'})
            for cell in matrix.values():
                self.assertAlmostEqual(cell['fm_raw'], cell['fm_hidden_raw'] + cell['fm_score_raw'])
                self.assertAlmostEqual(cell['fm_weighted'], 2 * cell['fm_raw'])
                self.assertAlmostEqual(cell['fm_raw'], sum(value['fm_raw'] for value in cell['families'].values()))
                for branch in cell['branches']:
                    self.assertEqual(branch['family'], 'MPD' if branch['branch_index'] < 5 else 'MRD')
                    for layer in branch['layers']:
                        self.assertEqual(len(layer['per_sample_fm_raw']), 32)
                        self.assertEqual((layer['numel_total'], layer['numel_per_sample']), (256, 8))
            for ratio in summary['real_activation_scale_ratios']:
                self.assertAlmostEqual(ratio['mean_abs_ratio'], 1.25, places=5)
                self.assertAlmostEqual(ratio['rms_ratio'], 1.25, places=5)
            for scope in [summary['total']] + list(summary['families'].values()):
                for item in scope.values():
                    self.assertAlmostEqual(item['average_student_effect'] + item['average_D_effect'], item['diagonal_change'])
            self.assertLess(summary['total']['fm_raw']['student_change_at_D3447'], 0)
            self.assertGreater(summary['total']['fm_raw']['D_change_at_S3447'], 0)
            broken = copy.deepcopy(matrix)
            broken['S3927_D3447']['branches'][0]['layers'][0]['real']['mean_abs'] += .1
            with self.assertRaisesRegex(RuntimeError, 'cached real'): probe.summarize_matrix(broken)
            broken = copy.deepcopy(matrix); broken['S3927_D3447']['branches'].pop()
            with self.assertRaisesRegex(RuntimeError, 'eight branches'): probe.summarize_matrix(broken)
        finally: ledger.close()

    def test_four_cell_effect_formula_and_missing_nonfinite_rejection(self):
        values = {'S3447_D3447': 1., 'S3927_D3447': 3., 'S3447_D3927': 6., 'S3927_D3927': 10.}
        result = probe.effects(values)
        self.assertEqual(result['interaction'], 2.)
        self.assertEqual(result['average_student_effect'], 3.)
        self.assertEqual(result['average_D_effect'], 6.)
        self.assertEqual(result['diagonal_change'], 9.)
        with self.assertRaises(RuntimeError): probe.effects(dict(values, S3927_D3927=float('nan')))
        del values['S3927_D3927']
        with self.assertRaises(RuntimeError): probe.effects(values)

    def test_near_zero_kept_with_raw_rms_relative_and_batch_shape(self):
        real = [torch.zeros(32, 1, 2) for _ in range(6)]
        fake = [torch.ones(32, 1, 2) for _ in range(6)]
        result = probe.branch_row(real[-1].flatten(1), real, fake[-1].flatten(1), fake, {'branch_index': 0})
        self.assertEqual(result['fm_raw'], 6.)
        for layer in result['layers']:
            self.assertTrue(layer['real_mean_abs_near_zero']); self.assertTrue(layer['real_rms_near_zero'])
            self.assertEqual(layer['rms_relative_l1_descriptive_only'], 1e8)
        with self.assertRaises(RuntimeError):
            probe.branch_row(real[-1].flatten(1), real[:-1], fake[-1].flatten(1), fake, {})
        small = [x[:1] for x in real]
        with self.assertRaisesRegex(RuntimeError, 'original32'):
            probe.branch_row(small[-1].flatten(1), small, small[-1].flatten(1), small, {})

    def test_forward_guard_runs_before_computation_and_counts_failed_attempt(self):
        model = ToyBranch(0).eval().requires_grad_(False)
        gate = Mock(side_effect=RuntimeError('capacity deadline'))
        ledger = probe.Ledger(gate); ledger.add(model, 'D_branch')
        with self.assertRaisesRegex(RuntimeError, 'capacity deadline'): model(torch.ones(32, 1, 2))
        self.assertEqual(ledger.counts['D_branch'], 0)
        gate.side_effect = None
        with patch.object(model, 'forward', side_effect=RuntimeError('fake OOM')):
            with self.assertRaisesRegex(RuntimeError, 'fake OOM'): model(torch.ones(32, 1, 2))
        self.assertEqual(ledger.counts['D_branch'], 1)
        ledger.counts['D_branch'] = 48
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'): model(torch.ones(32, 1, 2))
        ledger.close(); self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_progress_write_error_stops_without_retry_or_extra_forward(self):
        model = ToyD(); ledger = probe.Ledger(lambda: None)
        for branch in model.loss_branches(): ledger.add(branch, 'D_branch')
        snapshots = {}
        def remember(name, model, fingerprint, label): snapshots[name] = (model, fingerprint)
        try:
            with self.assertRaisesRegex(OSError, 'log write'):
                probe.crossed_forward(model, self.toy_states(model), torch.ones(32, 1, 2),
                    {3447: torch.zeros(32, 1, 2), 3927: torch.zeros(32, 1, 2)}, lambda: None, remember,
                    Mock(side_effect=OSError('log write')))
            self.assertEqual(ledger.counts['D_branch'], 2)
            for loaded, fingerprint in snapshots.values(): probe.old.assert_unchanged(loaded, fingerprint)
        finally: ledger.close()

    def test_rng_uses_whole_numpy_array_and_real_cpu_restore(self):
        original = probe.training.capture_rng_state()
        try:
            identity = probe.rng_fingerprint(original)
            changed = copy.deepcopy(original); changed['numpy'][1][321] ^= 1
            self.assertNotEqual(identity, probe.rng_fingerprint(changed))
            random.random(); np.random.random(3); torch.rand(4)
            report = self.receipt()
            probe.restore_readonly({}, original, probe.flags(), torch.device('cpu'), None, report)
            self.assertTrue(all(report.values()))
            self.assertEqual(identity, probe.rng_fingerprint(probe.training.capture_rng_state()))
        finally: probe.training.restore_rng_state(original, torch.device('cpu'))

    def test_readonly_error_still_restores_rng_flags_and_rejects_changed_buffer(self):
        model = ToyD(); before = probe.old.fingerprint_model(model)
        original, original_flags = probe.training.capture_rng_state(), probe.flags()
        try:
            model.branches[0].fixed_window.add_(.01)
            torch.rand(2)
            torch.backends.cudnn.benchmark = not original_flags['benchmark']
            report = self.receipt()
            with self.assertRaisesRegex(RuntimeError, 'Read-only restoration failed'):
                probe.restore_readonly({'D': (model, before)}, original, original_flags, torch.device('cpu'), None, report)
            self.assertFalse(report['models_unchanged']); self.assertTrue(report['rng_restored'])
            self.assertFalse(report['flags_unchanged_and_restored'])
            self.assertEqual(probe.flags(), original_flags)
            self.assertEqual(probe.rng_fingerprint(probe.training.capture_rng_state()), probe.rng_fingerprint(original))
        finally:
            probe.training.restore_rng_state(original, torch.device('cpu')); probe.restore_flags(original_flags)

    def test_partial_strict_load_failure_leaves_detectable_failure_not_success(self):
        model = ToyD(); before = probe.old.fingerprint_model(model)
        broken = self.toy_states(model)[3927]['discriminator']; del broken['branches.7.weight']
        with self.assertRaises(RuntimeError): probe.old.load_exact_weights(model, broken)
        with self.assertRaises(RuntimeError): probe.old.assert_unchanged(model, before)

    def test_cleanup_tries_all_actions_even_after_log_lock_restore_failures(self):
        called = []
        def action(label, fail=False):
            def execute():
                called.append(label)
                if fail: raise OSError(label)
            return execute
        errors = probe.close_envelope(SimpleNamespace(close=action('hooks')),
            SimpleNamespace(close=action('clock')), SimpleNamespace(flush=action('flush', True), close=action('log')),
            [SimpleNamespace(close=action('lock0', True)), SimpleNamespace(close=action('lock1'))], action('restore', True))
        self.assertEqual(called, ['restore', 'hooks', 'clock', 'flush', 'log', 'lock1', 'lock0'])
        self.assertEqual(len(errors), 3)

    def test_budget_exact_initial_gate_output_bound_and_exclusive_log(self):
        with tempfile.TemporaryDirectory(prefix='fm_thin_cpu_') as directory:
            output = Path(directory) / 'probe'; budget = probe.Budget(output)
            with patch.object(probe.os, 'statvfs', return_value=SimpleNamespace(f_bavail=3373268992, f_frsize=1)):
                self.assertEqual(budget.require()['required_bytes'], 3373268992)
            with patch.object(probe.os, 'statvfs', return_value=SimpleNamespace(f_bavail=3373268991, f_frsize=1)):
                with self.assertRaises(RuntimeError): budget.require()
            with patch.object(probe.os, 'statvfs', return_value=SimpleNamespace(f_bavail=10**10, f_frsize=1)):
                log = probe.LimitedLog(output, budget)
                try:
                    log.write('中文日志')
                    self.assertEqual(budget.used(), len('中文日志'.encode('utf8')))
                    with self.assertRaises(RuntimeError): probe.LimitedLog(output, budget)
                    with patch.object(budget, 'used', return_value=probe.META_LIMIT - probe.FAILURE_RESERVE):
                        with self.assertRaises(RuntimeError): budget.require(1)
                finally: log.close()

    def test_terminal_no_postrename_check_and_regular_results_have_postcheck(self):
        with tempfile.TemporaryDirectory(prefix='fm_thin_terminal_') as directory:
            budget = probe.Budget(Path(directory)); value = {'status': 'done'}
            gate = Mock(side_effect=[None, RuntimeError('late capacity')])
            with patch.object(budget, 'require', gate):
                budget.write('completion.json', value, terminal=True)
            self.assertEqual(gate.call_count, 1)
            self.assertEqual(json.loads((Path(directory) / 'completion.json').read_text()), value)
            with patch.object(budget, 'require', side_effect=[None, RuntimeError('postcheck')]):
                with self.assertRaisesRegex(RuntimeError, 'postcheck'): budget.write('results.json', value)
            with self.assertRaises(RuntimeError): budget.write('completion.json', value)

    def test_failed_terminal_gate_creates_no_completion_and_cleanup_failure_has_failure(self):
        with tempfile.TemporaryDirectory(prefix='fm_thin_failure_') as directory:
            budget = probe.Budget(Path(directory)); clock = SimpleNamespace(active=Mock())
            good = {key: True for key in self.receipt()}
            with patch.object(budget, 'require', side_effect=RuntimeError('capacity')):
                with self.assertRaisesRegex(RuntimeError, 'capacity'):
                    probe.publish_terminal(budget, clock, {}, good, [], None)
            self.assertFalse((budget.output / 'completion.json').exists())
            with patch.object(budget, 'require'):
                probe.publish_terminal(budget, clock, {}, good, [{'part': 'log_close', 'error': 'fake'}], None)
            self.assertTrue((budget.output / 'failure.json').exists())
            self.assertTrue((budget.output / 'cleanup_failure_or_abort.json').exists())
            self.assertFalse((budget.output / 'completion.json').exists())

    def test_clock_stops_at_300_and_cleanup_only_30(self):
        with patch.object(probe.time, 'monotonic', return_value=10): clock = probe.Clock()
        with patch.object(probe.time, 'monotonic', return_value=309.9): clock.active()
        with patch.object(probe.time, 'monotonic', return_value=310):
            with self.assertRaises(TimeoutError): clock.active()
        with patch.object(probe.time, 'monotonic', return_value=340): clock.cleanup()
        with patch.object(probe.time, 'monotonic', return_value=340.1):
            with self.assertRaises(TimeoutError): clock.cleanup()

    def test_replay_tolerance_original_float32_and_nonfinite_reject(self):
        reference = {'mel_l1': .18723352253437042, 'rms': .9591109752655029}
        probe.close_numbers(dict(reference), reference)
        probe.close_numbers(dict(reference, mel_l1=reference['mel_l1'] + 1e-7), reference)
        for value in (reference['mel_l1'] + 1e-5, float('nan'), float('inf')):
            with self.assertRaises(RuntimeError): probe.close_numbers(dict(reference, mel_l1=value), reference)

    def test_payload_headers_and_no_adam_rng_selection(self):
        contract = {'test_contract': True}; config = SimpleNamespace(as_dict=lambda: {'fixed': True})
        receipt = {'last_monitor': {'development': {'L1': .2}}, 'partial_batch_may_have_updated_D': False}
        payload = {'global_step': 3927, 'epoch': 0, 'mc_samples': 2, 'resumable': True,
            'run_contract_sha256': probe.training.canonical_json_sha256(contract), 'configuration': config.as_dict(),
            'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'execution_error': None,
            'development': receipt['last_monitor']['development'], 'lineage': {'parent_checkpoint_sha256': probe.CHECKPOINT_SHAS[3447]},
            'student': {'w': torch.ones(2)}, 'discriminator': {'w': torch.ones(3)},
            'student_optimizer': object(), 'discriminator_optimizer': object(), 'rng_state': object()}
        chosen = probe.select_payload(payload, 3927, config, contract, receipt)
        self.assertEqual(set(chosen), {'student', 'discriminator'})
        for key, wrong in [('global_step', 3447), ('epoch', 1), ('mc_samples', 1), ('resumable', False),
                           ('configuration', {}), ('score_semantics', 'noise'), ('execution_error', 'broken')]:
            with self.subTest(field=key), self.assertRaises(RuntimeError):
                probe.select_payload(dict(payload, **{key: wrong}), 3927, config, contract, receipt)

    def test_real_endpoint_metadata_and_scope_only_no_checkpoint_load(self):
        # 只读取已存在的JSON，核对真实字段名/终态；不读两full、模型或旧特征文件。
        verified = []
        with patch.object(probe, 'verify_files', side_effect=lambda files, check: verified.append(dict(files))):
            files, contracts, receipts, config = probe.validated_context(
                SimpleNamespace(driver_sha256='a' * 64, tests_sha256='b' * 64), SimpleNamespace(active=lambda: None))
        self.assertEqual(len(verified), 2)
        self.assertEqual(set(receipts), {3447, 3927})
        self.assertEqual(config.num_workers, 0)
        pt_paths = {path for path in files if path.endswith('.pt')}
        self.assertEqual(pt_paths, {str(probe.CHECKPOINTS[3447]), str(probe.CHECKPOINTS[3927]), str(probe.INPUT),
                                   str(probe.baseline.VOCODER_DIR / 'generator_step_2500000.pt')})
        self.assertLess(len(files), 100)
        for path, digest in probe.NEW_G_ASSET_SHAS.items():
            self.assertEqual(files[path], digest)
            for contract in contracts.values(): self.assertNotIn(path, contract['file_sha256'])
        for suffix in ('fastvoicegrad/train.py', 'fastvoicegrad/components.py', 'fastvoicegrad/objectives.py',
                       'voicegrad/train.py', 'hifigan/custom/mrd.py', 'hifigan/official/models.py'):
            self.assertIn(str(probe.ROOT / suffix), files)

    def test_no_training_noise_scoring_or_legacy_main_and_explicit_sha_gate(self):
        tree = ast.parse(FILE.read_text())
        forbidden = {'backward', 'step', 'Adam', 'AdamW', 'train_batch', 'train_batch_mc2', 'manual_seed',
                     'randn', 'randn_like', 'randint', 'noise_inputs', 'load_teacher_and_student', 'load_network_states',
                     'verify_context', 'matrix_effects', 'save', 'autograd', 'inference_main'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                self.assertNotIn(name, forbidden)
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    self.assertFalse(node.func.value.id == 'old' and node.func.attr == 'main')
        with self.assertRaises(RuntimeError): probe.launch_arguments([])
        with self.assertRaises(RuntimeError): probe.launch_arguments(['--probe-approved'])
        args = probe.launch_arguments(['--probe-approved', '--driver-sha256', 'a' * 64, '--tests-sha256', 'b' * 64])
        self.assertTrue(args.probe_approved)


if __name__ == '__main__': unittest.main(verbosity=2)
