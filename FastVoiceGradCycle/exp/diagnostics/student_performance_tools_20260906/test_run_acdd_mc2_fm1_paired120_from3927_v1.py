"""租用CPU验证FM1候选薄层；不调用真实训练main或读取真实模型。"""

import ast
import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Run these tests only on rental CPU with CUDA hidden')
import torch
import run_acdd_mc2_fm1_paired120_from3927_v1 as f
import evaluate_development_candidates as evaluator


def isolated(function, **replacements):
    namespace = dict(function.__globals__); namespace.update(replacements)
    return types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def raw(fm_weight=1., fm=3.):
    return {'student_total': 2. + fm_weight * fm + 4.5, 'student_adv': 2.,
            'fm_raw': fm, 'fm_weighted': fm_weight * fm, 'dist_raw': .1,
            'dist_weighted': 4.5, 'discriminator': 4.}


def monitor(step, mel=.19, rms=1.):
    return {'global_step': step, 'development': {'mel_l1': mel, 'prediction_rms': rms},
            'uniform64': {'GT_l1': .191}}


def update(model, optimizer):
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)


def toy(step=3927):
    student, discriminator = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5, betas=(.5, .9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .9))
    for model, opt in ((student, optimizer), (discriminator, d_optimizer)):
        update(model, opt)
        for state in opt.state.values(): state['step'].fill_(step)
    state = f.d.new_state(f.d.ARMS[0], 'toy')
    state.update(arm=f.ARM, restore_completed=True, completed_global_step=step,
                 new_updates=step - 3927, last_monitor=monitor(step), reason='maximum120_updates_completed')
    return state, student, discriminator, optimizer, d_optimizer


class FM1DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def cli(self):
        return ['--run-approved'] + [value for name in ('driver', 'tests', 'helper', 'helper-tests', 'protocol')
                                      for value in ('--' + name + '-sha256', 'a' * 64)]

    def test_explicit_candidate_approval_five_full_source_hashes(self):
        self.assertTrue(f.arguments(self.cli()).run_approved)
        with self.assertRaises(RuntimeError): f.arguments(self.cli()[1:])
        for index in (2, 4, 6, 8, 10):
            for bad in ('a' * 63, 'g' * 64, 'A' * 64):
                args = self.cli(); args[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(RuntimeError): f.arguments(args)

    def test_effective_config_changes_only_FM_and_preserves_original_paper_config(self):
        parent = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        before = parent.as_dict(); actual = f.candidate_config(parent)
        self.assertEqual(parent.as_dict(), before)
        self.assertEqual(actual.as_dict(), dict(before, lambda_fm=1.))
        with self.assertRaises(RuntimeError): f.candidate_config(actual)
        with self.assertRaises(RuntimeError): f.candidate_config(f.base.replace(parent, lambda_dist=44))

    def fixture(self):
        parts = toy(); state, s, D, a, b = parts
        config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        payload = {'global_step': 3927, 'epoch': 0, 'resumable': True, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
            'run_contract_sha256': f.d.PARENT_CANONICAL, 'effective_optimizer_lrs': dict(f.LRS),
            'configuration': config.as_dict(), 'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar',
            'stop_reason': 'maximum480_updates_completed', 'execution_error': None, 'development_monitor_global_step': 3927,
            'development': monitor(3927)['development'], 'student': copy.deepcopy(s.state_dict()),
            'discriminator': copy.deepcopy(D.state_dict()), 'student_optimizer': copy.deepcopy(a.state_dict()),
            'discriminator_optimizer': copy.deepcopy(b.state_dict()), 'rng_state': f.training.capture_rng_state()}
        # 为toy实际RNG构造独立函数globals，不更改冻结Dlr模块的常量。
        restore = isolated(f.d.restore_parent, PARENT_RNG_SHA=f.probe.tree_sha(payload['rng_state']))
        shim = SimpleNamespace(**dict(f.d.__dict__, restore_parent=restore))
        return parts, payload, config, {'configuration': config.as_dict()}, {'last_monitor': monitor(3927)}, shim

    def test_complete_S_D_twoAdam_RNG_restore_then_only_real_config_delta(self):
        parts, payload, config, contract, receipt, shim = self.fixture()
        state, s, D, a, b = parts; before = f.previous.content_state(s, D, a, b)
        payload_sha = f.probe.tree_sha(payload)
        update(s, a); update(D, b); torch.rand(3)
        with patch.object(f, 'd', shim):
            effective, evidence = f.restore_parent(payload, s, D, a, b, config, contract, receipt, torch.device('cpu'))
        self.assertEqual(f.previous.content_state(s, D, a, b), before)
        self.assertEqual(evidence['parent_restore'], before)
        self.assertTrue(evidence['all_models_optimizers_rng_restored_without_change'])
        self.assertEqual(effective.lambda_fm, 1.); self.assertEqual(config.lambda_fm, 2.)
        self.assertEqual((a.param_groups[0]['lr'], b.param_groups[0]['lr']), (5e-5, 2e-4))
        self.assertEqual(f.probe.tree_sha(payload), payload_sha)
        self.assertEqual(f.probe.tree_sha(f.training.capture_rng_state()), f.probe.tree_sha(payload['rng_state']))
        for name, opt in (('student', a), ('discriminator', b)):
            for index, value in opt.state_dict()['state'].items():
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    self.assertNotEqual(value[key].data_ptr(), payload[name + '_optimizer']['state'][index][key].data_ptr())

    def test_parent_wrong_config_step_or_Dlr_refused_not_overwritten(self):
        for key, value in [('configuration', {'lambda_fm': 1.}), ('global_step', 4047),
                           ('effective_optimizer_lrs', {'student': 5e-5, 'discriminator': 1e-4}), ('mc_samples', 1)]:
            parts, payload, config, contract, receipt, shim = self.fixture(); payload[key] = value
            with patch.object(f, 'd', shim), self.subTest(key=key), self.assertRaises(RuntimeError):
                f.restore_parent(payload, *parts[1:], config, contract, receipt, torch.device('cpu'))

    def test_raw_fields_preserved_with_two_totals_and_bad_weighted_fields_rejected(self):
        old = raw(2.); new = raw(1.); old_before = dict(old); new_before = dict(new)
        control, candidate = f.loss_columns(old, 2.), f.loss_columns(new, 1.)
        self.assertEqual(candidate['actual_total'], 9.5)
        self.assertEqual(candidate['paper_equivalent_total'], 12.5)
        self.assertEqual(control['paper_equivalent_total'], candidate['paper_equivalent_total'])
        self.assertEqual(candidate['student_total'], new['student_total'])
        self.assertEqual(old, old_before); self.assertEqual(new, new_before)
        for key, value in [('student_total', 0.), ('fm_weighted', 6.), ('dist_weighted', 9.), ('fm_raw', float('nan'))]:
            changed = dict(new, **{key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError): f.loss_columns(changed, 1.)

    def paired_row(self, coefficient=1., step=3928):
        row = f.loss_columns(raw(coefficient), coefficient)
        row.update(global_step=step, indices=[1], crop_seed=101234 + step, noise_seed=201234 + step,
            input_identity={'toy': 'same'}, mc={key: key for key in ('source_noise', 'score_noise1', 't1',
            't1_values', 'score_noise2', 't2', 't2_values', 'private_seed')})
        row['mc'].update(dist1=.1, dist2=.1)
        return row

    def test_first_raw_adv_FM_D_dist_same_but_two_totals_have_expected_difference(self):
        control, candidate = self.paired_row(2.), self.paired_row(1.)
        f.paired_input_check(candidate, control)
        for key in ('student_adv', 'fm_raw', 'discriminator', 'dist_raw', 'dist_weighted', 'paper_equivalent_total'):
            changed = copy.deepcopy(candidate); changed[key] += .01
            with self.subTest(key=key), self.assertRaises(RuntimeError): f.paired_input_check(changed, control)
        control, candidate = self.paired_row(2., 3929), self.paired_row(1., 3929)
        candidate['fm_raw'] = 5.; candidate['student_adv'] = 9.
        f.paired_input_check(candidate, control)
        candidate['mc']['score_noise2'] = 'changed'
        with self.assertRaises(RuntimeError): f.paired_input_check(candidate, control)

    def simulate(self, mel=None, fail_after=None):
        state = toy()[0]; state['last_monitor'] = monitor(3927); called, written = [], []
        future = [{'global_step': step} for step in range(3928, 4048)]
        def step(planned):
            number = planned['global_step']; state.update(completed_global_step=number, new_updates=number - 3927)
            return dict(f.loss_columns(raw(1., 3. + .01 * (number - 3927)), 1.), global_step=number, mc={'dist1': .1, 'dist2': .1})
        def evaluate(number): called.append(number); return monitor(number, (mel or {}).get(number, .19))
        def publish(name, value, append=True): written.append((name, copy.deepcopy(value)))
        def check():
            if fail_after is not None and state['new_updates'] >= fail_after: raise RuntimeError('time bound')
        return state, future, step, evaluate, publish, check, monitor(3927), called, written

    def test_all120_rows_two_full_windows_no_stop_from_rising_actual_or_paper_total(self):
        parts = self.simulate(); result = f.run_updates(*parts[:7])
        self.assertEqual(len(result['rows']), 120); self.assertEqual(len(result['windows']), 2)
        self.assertEqual(parts[7], [3987, 4047]); self.assertEqual(len(result['monitors']), 3)
        for key in f.LOSS_KEYS: self.assertIn(key, result['windows'][0]['means'])
        self.assertGreater(result['windows'][1]['means']['actual_total'], result['windows'][0]['means']['actual_total'])
        self.assertGreater(result['windows'][1]['means']['paper_equivalent_total'], result['windows'][0]['means']['paper_equivalent_total'])
        self.assertEqual(result['reason'], 'maximum120_updates_completed')
        self.assertFalse(any(w['used_as_stop_gate'] for w in result['windows']))

    def test_original_guards_only_new60_120_and_partial_loss_log_survives(self):
        parts = self.simulate({3987: .203, 4047: .204}); result = f.run_updates(*parts[:7])
        self.assertEqual(parts[0]['new_updates'], 120)
        self.assertEqual(result['reason'], 'two_consecutive_DEV32_above3327_by5percent')
        parts = self.simulate({3987: .22}); result = f.run_updates(*parts[:7])
        self.assertEqual(parts[0]['new_updates'], 60); self.assertEqual(result['reason'], 'original_S0_extreme_guard')
        parts = self.simulate(fail_after=11)
        with self.assertRaisesRegex(RuntimeError, 'time bound'): f.run_updates(*parts[:7])
        self.assertEqual(len([row for row in parts[8] if row[0] == 'step_losses.jsonl']), 11)
        self.assertEqual(parts[7], [])

    def test_real_Adam_returns_complete_and_half_step_keeps_last_complete_cursor(self):
        state, s, D, a, b = toy()
        def complete(): update(D, b); update(s, a); return raw()
        f.d.complete_update(state, 3928, a, b, complete)
        self.assertEqual(state['completed_global_step'], 3928); self.assertFalse(state['partial_batch_may_have_updated_D'])
        state, s, D, a, b = toy()
        def half(): update(D, b); raise RuntimeError('half')
        with self.assertRaisesRegex(RuntimeError, 'half'): f.d.complete_update(state, 3928, a, b, half)
        self.assertEqual(state['completed_global_step'], 3927); self.assertTrue(state['partial_batch_may_have_updated_D'])

    def test_exact_full_only_budget_and_terminal_publish_no_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'trial'; output.mkdir(); (output / f.ARM).mkdir()
            budget = f.Budget(output)
            fake_pair = SimpleNamespace(available_bytes=lambda path: 20 * 1024**3, InsufficientDisk=f.pair.InsufficientDisk)
            require = isolated(f.d.Budget.require, pair=fake_pair)
            self.assertEqual(require(budget)['required_bytes'], 4832505856)
            self.assertEqual(set(budget.pending), {'full', 'later_DEV90'})
            fail_pair = SimpleNamespace(available_bytes=lambda path: 4832505855, InsufficientDisk=f.pair.InsufficientDisk)
            with self.assertRaises(f.pair.InsufficientDisk): isolated(f.d.Budget.require, pair=fail_pair)(budget)
            budget.require = Mock(return_value={})
            path = output / f.ARM / 'completion.json'
            budget.publish_success(path, {'status': 'completed120'})
            self.assertEqual(budget.require.call_count, 1)
            self.assertEqual(json.loads(path.read_text())['status'], 'completed120')
            with self.assertRaises(RuntimeError): budget.publish_success(path, {})

    def test_full_toy_save_has_real_FM1_S_D_Adam_rng_and_evaluator_accepts_contract(self):
        state, s, D, a, b = toy(4047)
        config = f.candidate_config(f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0))
        contract = {'configuration': config.as_dict(), 'candidate_deviation': {'lambda_fm': 1.}}
        state['contract_sha256'] = f.training.canonical_json_sha256(contract)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); stage = output / f.ARM; stage.mkdir(); budget = f.Budget(output)
            budget.require = Mock(return_value={})
            f.save_endpoint(state, config, s, D, a, b, budget, SimpleNamespace(active=lambda: None))
            path = stage / 'final_full.pt'; payload = torch.load(path, map_location='cpu')
            self.assertTrue(payload['resumable']); self.assertEqual(payload['global_step'], 4047)
            self.assertEqual(payload['configuration']['lambda_fm'], 1.)
            self.assertEqual(payload['paper_reference_coefficients']['lambda_fm'], 2.)
            self.assertEqual(payload['effective_optimizer_lrs'], {'student': 5e-5, 'discriminator': 2e-4})
            for key in ('student', 'discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state'):
                self.assertIn(key, payload)
            self.assertEqual(evaluator.validate_payload(payload, {'epoch': 0, 'global_step': 4047, 'state_key': 'student'}, contract), state['contract_sha256'])
            self.assertNotIn('full', budget.pending)
            self.assertEqual([p.name for p in stage.glob('*.pt')], ['final_full.pt'])
            self.assertEqual(path.stat().st_mode & 0o777, 0o444)

    def test_no_save_at_zero_or_partial_failed_or_wrong_coefficient(self):
        for mode in ('zero', 'partial', 'failure', 'coefficient', 'wrong_monitor'):
            state, s, D, a, b = toy(3927 if mode == 'zero' else 4047)
            config = f.candidate_config(f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0))
            if mode == 'partial': state['partial_batch_may_have_updated_D'] = True
            if mode == 'failure': state['execution_error'] = 'failed'
            if mode == 'coefficient': config = f.base.replace(config, lambda_fm=2.)
            if mode == 'wrong_monitor': state['last_monitor'] = monitor(3987)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                f.save_endpoint(state, config, s, D, a, b, None, None)

    def test_static_no_delete_control_launch_old_save_or_unbounded_forward_counts(self):
        source = Path(f.__file__).read_text(); tree = ast.parse(source)
        self.assertEqual(f.FORWARD_LIMITS, {'teacher': 303, 'student': 252, 'vocoder': 366, 'active_D_branches': 3840, 'fixed_D_branches': 48})
        self.assertEqual(f.LRS, {'student': 5e-5, 'discriminator': 2e-4})
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call): continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None)
            self.assertNotIn(name, ('unlink', 'rmdir', 'rmtree', 'remove', 'infer_candidate', 'evaluate_metrics'))
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                self.assertNotEqual((node.func.value.id, node.func.attr), ('d', 'save_endpoint'))
                self.assertNotEqual((node.func.value.id, node.func.attr), ('d', 'main'))
        self.assertIn("'post_registered_reused_historical_control'", source)
        self.assertIn("'control_retrained_or_rescored': False", source)
        self.assertIn("d.candidate_gate(SimpleNamespace(control_archive_sha256=CONTROL_ARCHIVE_SHA), joint, check)", source)
        self.assertIn("parent_config, device, step)", source)
        self.assertIn("'paper_reference_coefficients': {'lambda_fm': 2.", source)


if __name__ == '__main__': unittest.main()
