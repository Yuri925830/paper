"""只在租用机CPU检验调度/真实Adam游标/小模型原子保存；不读真实checkpoint或特征。"""

import ast
import copy
import inspect
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only Matpool CPU tests with CUDA_VISIBLE_DEVICES empty are allowed')

import torch
from torch import nn
import run_acdd_mc2_paired120_v1 as driver


def losses(value=1.):
    return {key: float(value) for key in driver.pair.LOSS_KEYS}


def entry(step, mel=.19254307448863983, rms=1.):
    result = {'global_step': step, 'development': {'mel_l1': mel, 'prediction_rms': rms}}
    if step - driver.START in driver.UNIFORM64_UPDATES:
        result['uniform64'] = {'uniform_weighted_score': .04}
    return result


def toy_adams(arm=driver.ARMS[1]):
    student, discriminator = nn.Linear(2, 1), nn.Linear(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5, betas=(.5, .9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .9))
    for model, opt in ((student, optimizer), (discriminator, d_optimizer)):
        for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
        opt.step(); opt.zero_grad(set_to_none=True)
        for state in opt.state.values(): state['step'].fill_(driver.START)
    state = driver.new_arm_state(arm, 'tiny_contract')
    state['restore_completed'] = True
    return state, student, discriminator, optimizer, d_optimizer


def update(model, opt):
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    opt.step(); opt.zero_grad(set_to_none=True)


def complete_toy(args):
    state, student, discriminator, optimizer, d_optimizer = args
    def call():
        update(discriminator, d_optimizer); update(student, optimizer)
        return losses()
    return driver.complete_update(state, driver.START + 1, optimizer, d_optimizer, call)


class MC2DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_entry_and_unbound_review_digests_reject_before_runtime_actions(self):
        with mock.patch.object(driver.training, 'acquire_runtime_locks', side_effect=AssertionError('runtime touched')):
            with self.assertRaisesRegex(RuntimeError, 'explicit approval'):
                driver.main([])
        with mock.patch.object(driver, 'PROTOCOL_SHA', None), mock.patch.object(driver, 'FIRST_ANCHOR_SHA', None):
            with self.assertRaisesRegex(RuntimeError, 'final review'):
                driver.launch_arguments(['--run-approved', '--parent-checkpoint-sha256', driver.PARENT_SHA])
        with mock.patch.object(driver, 'PROTOCOL_SHA', 'a' * 64), mock.patch.object(driver, 'FIRST_ANCHOR_SHA', 'b' * 64), \
             mock.patch.object(driver, 'TEST_SHAS', {'tiny_test.py': 'c' * 64}):
            self.assertEqual(driver.launch_arguments(['--run-approved', '--parent-checkpoint-sha256', driver.PARENT_SHA]).max_seconds, 1800)
            with self.assertRaises(RuntimeError):
                driver.launch_arguments(['--run-approved', '--parent-checkpoint-sha256', driver.PARENT_SHA, '--max-seconds', '1200'])

    def test_no_old_stop_winner_retiming_or_metric_entry_is_called(self):
        tree = ast.parse(Path(driver.__file__).read_text())
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for forbidden in ('stop_rules', 'trend_state', 'candidate_gate', 'permits_candidate', 'stratified_timesteps',
                          'sample_timesteps', 'save_primary', 'train_batch_with_timesteps'):
            self.assertNotIn(forbidden, calls)
        for forbidden in ('run_conversion_score_development', 'evaluate_development_candidates', 'inference'):
            self.assertNotIn(forbidden, [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)])
        self.assertEqual(driver.ARMS, ('control_mc1', 'candidate_mc2'))
        self.assertEqual(driver.STOP - driver.START, 120)

    def test_budget_exact_initial_math_future_commits_and_preserved_reserves(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(driver.pair, 'available_bytes', return_value=20 * 1024**3):
            output = Path(name) / 'trial'; budget = driver.Budget(output)
            initial = budget.require()
            self.assertEqual(initial['required_bytes'], 5109268480)
            output.mkdir(); path = output / 'tiny.pt'; path.write_bytes(b'toy only')
            budget.committed('student', path)
            self.assertEqual(budget.require()['required_bytes'], initial['required_bytes'] - driver.STUDENT_BYTES - 1024**2)
            with mock.patch.object(driver.pair, 'available_bytes', return_value=initial['required_bytes'] - 1):
                fresh = driver.Budget(Path(name) / 'new_trial')
                with self.assertRaises(driver.pair.InsufficientDisk): fresh.require()
            self.assertEqual(driver.previous.RESERVE, 3 * 1024**3 + 128 * 1024**2)
            self.assertEqual(driver.previous.META_LIMIT, 16 * 1024**2)

    def test_metadata_path_links_overflow_and_exclusive_log(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(driver.pair, 'available_bytes', return_value=20 * 1024**3):
            output = Path(name) / 'trial'; budget = driver.Budget(output)
            log = driver.replication.open_exclusive_log(output, budget)
            log.write('only this new log\n'); log.close()
            with self.assertRaisesRegex(RuntimeError, 'must be new'):
                driver.replication.open_exclusive_log(output, budget)
            output.mkdir(); budget.write(output / 'status.json', {'status': 'toy'})
            outside = Path(name) / 'outside'; outside.mkdir()
            (output / 'link').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(RuntimeError): budget.write(output / 'link' / 'status.json', {})
            (output / 'link').unlink()
            with mock.patch.object(budget, 'metadata_used', return_value=16 * 1024**2):
                with self.assertRaises(driver.pair.InsufficientDisk): budget.require(1)

    def test_guards_exact_S0_boundaries_and_only_two_consecutive5percent(self):
        baseline = {'mel_l1': .19254307448863983}
        upper = driver.S0_ANCHOR['mel_l1'] * 1.1
        lower = driver.S0_ANCHOR['prediction_rms'] * .8
        self.assertEqual(upper, .2193548917770386)
        self.assertEqual(lower, .7995790004730225)
        self.assertFalse(driver.guard_reasons([entry(3327, upper, lower)], baseline))
        self.assertIn('original_S0_extreme_guard', driver.guard_reasons([entry(3357, upper + 1e-9)], baseline))
        self.assertIn('original_S0_extreme_guard', driver.guard_reasons([entry(3357, rms=lower - 1e-9)], baseline))
        threshold = baseline['mel_l1'] * 1.05
        start = entry(3327)
        self.assertFalse(driver.guard_reasons([start, entry(3357, threshold + 1e-8)], baseline))
        self.assertFalse(driver.guard_reasons([start, entry(3357, threshold + 1e-8), entry(3387, threshold)], baseline))
        found = driver.guard_reasons([start, entry(3357, threshold + 1e-8), entry(3387, threshold + 1e-8)], baseline)
        self.assertEqual(found, ['two_consecutive_DEV32_above3327_by5percent'])
        self.assertFalse(driver.guard_reasons([entry(3327, .19), entry(3357, .1901), entry(3387, .1902)], baseline))
        with self.assertRaises(RuntimeError): driver.guard_reasons([start, entry(3357, float('nan'))], baseline)
        with self.assertRaises(RuntimeError): driver.guard_reasons([start, start], baseline)

    def simulate_arm(self, mel_by_update=None, abort_after=None):
        state = driver.new_arm_state(driver.ARMS[0], 'test')
        state['restore_completed'] = True
        initial = entry(driver.START); state['last_monitor'] = initial
        future = [{'global_step': step} for step in range(driver.START + 1, driver.STOP + 1)]
        calls, publications = [], []
        def step(planned):
            number = planned['global_step']; state['completed_global_step'] = number
            state['new_updates'] = number - driver.START
            return dict(losses(number), global_step=number)  # 损失持续升高仍不得引入旧窗口门。
        def evaluate(number):
            calls.append(number - driver.START)
            return entry(number, (mel_by_update or {}).get(number - driver.START, .19254307448863983))
        def publish(name, value, append=True): publications.append((name, copy.deepcopy(value)))
        def check():
            if abort_after is not None and state['new_updates'] >= abort_after:
                raise driver.BoundaryStop('time_budget')
        return state, initial, future, step, evaluate, publish, check, calls, publications

    def test_full120_schedule_exact_windows_and_no_three_rising_loss_gate(self):
        state, initial, future, step, evaluate, publish, check, calls, saved = self.simulate_arm()
        result = driver.run_updates(state, future, step, evaluate, publish, check, initial['development'], initial)
        self.assertEqual(calls, [30, 60, 90, 120])
        self.assertEqual([m['global_step'] - driver.START for m in result['monitors'] if 'uniform64' in m], [0, 60, 120])
        self.assertEqual([w['first_global_step'] for w in result['windows']], [3328, 3388])
        self.assertEqual([w['global_step'] for w in result['windows']], [3387, 3447])
        self.assertEqual([w['count'] for w in result['windows']], [60, 60])
        self.assertEqual(result['reason'], 'maximum120_updates_completed')
        self.assertEqual(len([x for x in saved if x[0] == 'step_losses.jsonl']), 120)
        self.assertEqual(state['complete_loss_records'], 120)

    def test_guard_stops_only_this_arm_at_second_check_without_extra_monitor(self):
        setup = self.simulate_arm({30: .203, 60: .204})
        state, initial, future, step, evaluate, publish, check, calls, saved = setup
        result = driver.run_updates(state, future, step, evaluate, publish, check, initial['development'], initial)
        self.assertEqual(state['new_updates'], 60)
        self.assertEqual(calls, [30, 60])
        self.assertEqual(result['reason'], 'two_consecutive_DEV32_above3327_by5percent')
        self.assertEqual(len(result['windows']), 1)
        # 一个新臂不继承前臂的监测序列或保护失败，仍完整执行已登记120步。
        fresh = self.simulate_arm()
        other = driver.run_updates(fresh[0], fresh[2], fresh[3], fresh[4], fresh[5], fresh[6], fresh[1]['development'], fresh[1])
        self.assertEqual(len(other['rows']), 120)

    def test_timeout_does_not_backfill_unregistered_endpoint_monitor(self):
        setup = self.simulate_arm(abort_after=17)
        state, initial, future, step, evaluate, publish, check, calls, saved = setup
        with self.assertRaisesRegex(driver.BoundaryStop, 'time_budget'):
            driver.run_updates(state, future, step, evaluate, publish, check, initial['development'], initial)
        self.assertEqual(state['new_updates'], 17)
        self.assertEqual(calls, [])
        self.assertEqual(state['last_monitor']['global_step'], driver.START)
        with self.assertRaises(RuntimeError):
            driver.run_updates(state, future[:-1], step, evaluate, publish, check, initial['development'], initial)

    def test_complete_update_checks_actual_two_Adam_steps_and_restores_step_attributes(self):
        args = toy_adams(); state, student, discriminator, optimizer, d_optimizer = args
        self.assertNotIn('step', optimizer.__dict__)
        actual = complete_toy(args)
        self.assertEqual(actual, losses())
        self.assertEqual(state['completed_global_step'], 3328)
        self.assertFalse(state['partial_batch_may_have_updated_D'])
        self.assertEqual(state['adam_step_returns'], {'student': 1, 'discriminator': 1})
        self.assertNotIn('step', optimizer.__dict__); self.assertNotIn('step', d_optimizer.__dict__)
        self.assertTrue(all(v['uniform_step'] == 3328 for v in state['optimizer_cursors'].values()))

    def test_D_only_exception_cannot_advance_cursor_or_save(self):
        args = toy_adams(); state, student, discriminator, optimizer, d_optimizer = args
        def half():
            update(discriminator, d_optimizer)
            raise RuntimeError('half update')
        with self.assertRaisesRegex(RuntimeError, 'half update'):
            driver.complete_update(state, 3328, optimizer, d_optimizer, half)
        self.assertEqual(state['completed_global_step'], 3327)
        self.assertTrue(state['partial_batch_may_have_updated_D'])
        self.assertEqual(state['optimizer_cursors']['discriminator']['uniform_step'], 3328)
        with self.assertRaisesRegex(RuntimeError, 'half or uncertain'):
            driver.assert_saveable(*args)

    def test_error_after_both_step_returns_reports_new_complete_step(self):
        args = toy_adams(); state, student, discriminator, optimizer, d_optimizer = args
        def after():
            update(discriminator, d_optimizer); update(student, optimizer)
            raise RuntimeError('post-update receipt failure')
        with self.assertRaisesRegex(RuntimeError, 'post-update receipt failure'):
            driver.complete_update(state, 3328, optimizer, d_optimizer, after)
        self.assertEqual(state['completed_global_step'], 3328)
        self.assertFalse(state['partial_batch_may_have_updated_D'])
        self.assertTrue(state['exception_after_both_adam_steps'])
        driver.assert_saveable(*args)

    def test_all_counters_new_but_Adam_step_did_not_return_is_still_uncertain(self):
        args = toy_adams(); state, student, discriminator, optimizer, d_optimizer = args
        original = optimizer.step
        def broken(*positional, **keywords):
            original(*positional, **keywords)
            raise RuntimeError('inside Adam before return')
        optimizer.step = broken
        def half():
            update(discriminator, d_optimizer); update(student, optimizer)
            return losses()
        with self.assertRaisesRegex(RuntimeError, 'inside Adam'):
            driver.complete_update(state, 3328, optimizer, d_optimizer, half)
        self.assertEqual(state['completed_global_step'], 3327)
        self.assertTrue(state['partial_batch_may_have_updated_D'])
        self.assertTrue(all(value['uniform_step'] == 3328 for value in state['optimizer_cursors'].values()))
        self.assertIs(optimizer.step, broken)

    def test_nonuniform_parameter_counters_are_not_a_complete_update(self):
        args = toy_adams(); state, student, discriminator, optimizer, d_optimizer = args
        next(iter(optimizer.state.values()))['step'].fill_(3326)
        called = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'previous complete step'):
            driver.complete_update(state, 3328, optimizer, d_optimizer, called)
        called.assert_not_called()
        self.assertIsNone(driver.optimizer_cursor(optimizer)['uniform_step'])

    def save_fixture(self, name, arm):
        args = toy_adams(arm); complete_toy(args)
        args[0]['reason'] = 'toy_complete'; output = Path(name) / arm; output.mkdir()
        budget = driver.Budget(output); directory = output / arm; directory.mkdir()
        common = {'global_step': 3328, 'kind': 'tiny_CPU_test'}
        clock = SimpleNamespace(saving=lambda: None)
        return args, budget, directory, common, clock

    def test_only_control_S_and_candidate_full_saved_no_duplicate_candidate_S(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(driver.pair, 'available_bytes', return_value=20 * 1024**3):
            for arm in driver.ARMS:
                args, budget, directory, common, clock = self.save_fixture(name, arm)
                driver.save_endpoint(directory, args[0], common, *args[1:], budget, clock)
                models = list(directory.glob('*.pt'))
                self.assertEqual(len(models), 1)
                tiny = torch.load(models[0], map_location='cpu')  # 只读取本测试刚创建的几KB小模型。
                if arm == driver.ARMS[0]:
                    self.assertEqual(models[0].name, 'student_step_003328.pt')
                    self.assertNotIn('student_optimizer', tiny)
                    self.assertNotIn('discriminator', tiny)
                    self.assertFalse(tiny['resumable'])
                    self.assertEqual(args[0]['last_full_saved_global_step'], 3327)
                    self.assertEqual(args[0]['last_student_saved_global_step'], 3328)
                else:
                    self.assertEqual(models[0].name, 'final_full.pt')
                    self.assertTrue(tiny['resumable'])
                    for key in ('student', 'discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state'):
                        self.assertIn(key, tiny)
                    self.assertEqual(args[0]['last_full_saved_global_step'], 3328)
                    self.assertIsNone(args[0]['student_snapshot'])
                with self.assertRaisesRegex(RuntimeError, 'must both be new'):
                    driver.save_endpoint(directory, args[0], common, *args[1:], budget, clock)

    def test_sha_failure_after_atomic_commit_keeps_true_saved_cursor(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(driver.pair, 'available_bytes', return_value=20 * 1024**3):
            args, budget, directory, common, clock = self.save_fixture(name, driver.ARMS[1])
            with mock.patch.object(driver.base, 'sha256_file', side_effect=RuntimeError('hash failed')):
                with self.assertRaisesRegex(RuntimeError, 'hash failed'):
                    driver.save_endpoint(directory, args[0], common, *args[1:], budget, clock)
            self.assertTrue((directory / 'final_full.pt').is_file())
            self.assertEqual(args[0]['last_full_saved_global_step'], 3328)
            self.assertTrue(args[0]['full_snapshot']['committed'])
            self.assertIsNone(args[0]['full_snapshot']['sha256'])
            self.assertNotIn('full', budget.pending)

    def test_commit_accounting_error_still_cannot_deny_the_renamed_model(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(driver.pair, 'available_bytes', return_value=20 * 1024**3):
            args, budget, directory, common, clock = self.save_fixture(name, driver.ARMS[0])
            with mock.patch.object(budget, 'committed', side_effect=RuntimeError('accounting failed')):
                with self.assertRaisesRegex(RuntimeError, 'accounting failed'):
                    driver.save_endpoint(directory, args[0], common, *args[1:], budget, clock)
            self.assertTrue((directory / 'student_step_003328.pt').exists())
            self.assertEqual(args[0]['last_student_saved_global_step'], 3328)
            self.assertTrue(args[0]['student_snapshot']['committed'])

    def test_serialization_failure_keeps_parent_cursor_and_preserves_owned_tmp(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(driver.pair, 'available_bytes', return_value=20 * 1024**3):
            args, budget, directory, common, clock = self.save_fixture(name, driver.ARMS[1])
            with mock.patch.object(driver.torch, 'save', side_effect=RuntimeError('serialization failed')):
                with self.assertRaisesRegex(RuntimeError, 'serialization failed'):
                    driver.save_endpoint(directory, args[0], common, *args[1:], budget, clock)
            self.assertFalse((directory / 'final_full.pt').exists())
            self.assertTrue((directory / 'final_full.pt.tmp').exists())
            self.assertEqual(args[0]['last_full_saved_global_step'], 3327)
            self.assertIsNone(args[0]['full_snapshot'])

    def test_nonfinite_model_or_moment_cannot_be_serialized_as_resumable(self):
        for corrupt_model in (True, False):
            args = toy_adams(); complete_toy(args)
            with torch.no_grad():
                if corrupt_model: next(args[1].parameters()).fill_(float('nan'))
                else: next(iter(args[3].state.values()))['exp_avg'].fill_(float('inf'))
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                driver.assert_saveable(*args)

    def test_anchor_rejects_retimed_first_loss_and_pairing_compares_only_first_dist(self):
        identity = {'batch_tensors': {}, 'source_noise': {}, 'score_noise1': {}, 't1': {}, 't1_values': [1] * 32}
        anchor = {'global_step': 3328, 'parent_checkpoint_sha256': driver.PARENT_SHA,
                  'production_train_sha256': driver.mc.PRODUCTION_SHA, 'uses_original_uniform_t': True,
                  'input_identity': identity, 'expected_losses': losses(), 'source_sha256': {'/tiny-source': 'a' * 64}}
        driver.validate_first_anchor(anchor); driver.first_input_check(identity, anchor)
        with self.assertRaises(RuntimeError): driver.validate_first_anchor(dict(anchor, uses_original_uniform_t=False))
        candidate = losses(); candidate['dist_raw'] = 9.; candidate['student_total'] = 99.
        self.assertTrue(driver.first_loss_check(driver.ARMS[1], candidate, {'dist1': 1.}, anchor))
        with self.assertRaises(RuntimeError): driver.first_loss_check(driver.ARMS[0], candidate, {'dist1': 1.}, anchor)
        row = dict(losses(), global_step=3328, indices=[1], input_identity=identity, mc={'dist1': 1.})
        driver.paired_input_check(row, copy.deepcopy(row))
        bad = copy.deepcopy(row); bad['input_identity']['t1_values'][0] = 2
        with self.assertRaises(RuntimeError): driver.paired_input_check(row, bad)

    def test_frozen_real_first_anchor_metadata_has_original_uniform_inputs(self):
        for name, expected in driver.TEST_SHAS.items():
            self.assertEqual(hashlib.sha256((driver.TOOLS / name).read_bytes()).hexdigest(), expected)
        raw = driver.FIRST_ANCHOR.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), driver.FIRST_ANCHOR_SHA)
        anchor = driver.validate_first_anchor(json.loads(raw))
        self.assertEqual(anchor['expected_losses']['dist_raw'], .054745227098464966)
        self.assertEqual(anchor['input_identity']['t1_values'][0], 445)
        self.assertEqual(len(anchor['input_identity']['t1_values']), 32)
        self.assertEqual(anchor['batch_plan']['crop_seed'], 104562)
        self.assertEqual(anchor['batch_plan']['noise_seed'], 204562)
        self.assertTrue(anchor['verification']['control_original_and_used_timesteps_identical'])
        self.assertEqual(anchor['verification']['GPU_forwards_or_new_CUDA_draws'], 0)
        planned = dict(anchor['batch_plan'], global_step=3328)
        driver.first_plan_check(planned, anchor)
        for key, bad_value in (('global_step', 3329), ('indices', []), ('crop_seed', 1234), ('noise_seed', 1234)):
            with self.assertRaisesRegex(RuntimeError, key):
                driver.first_plan_check(dict(planned, **{key: bad_value}), anchor)

    def test_cleanup_attempts_every_close_even_when_flush_clock_and_one_lock_fail(self):
        events = []
        def action(name, fail=False):
            def close():
                events.append(name)
                if fail: raise RuntimeError('injected ' + name)
            return close
        ledger = SimpleNamespace(close=action('hooks'))
        clock = SimpleNamespace(close=action('clock', True))
        log = SimpleNamespace(flush=action('flush', True), close=action('log_close'))
        locks = [SimpleNamespace(close=action('lock0')), SimpleNamespace(close=action('lock1', True)),
                 SimpleNamespace(close=action('lock2'))]
        errors = driver.cleanup_resources(ledger, clock, log, locks)
        self.assertEqual(events, ['hooks', 'clock', 'flush', 'log_close', 'lock2', 'lock1', 'lock0'])
        self.assertEqual(len(errors), 3)

    def test_completion_stays_pending_until_successful_resource_cleanup(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(driver.pair, 'available_bytes', return_value=20 * 1024**3):
            output = Path(name) / 'trial'; output.mkdir(); budget = driver.Budget(output)
            pending = {'status': 'finalization_pending', 'cleanup_verified': False, 'intended_status': 'both120_completed'}
            budget.write(output / 'completion.json', pending)
            self.assertEqual(json.loads((output / 'completion.json').read_text())['status'], 'finalization_pending')
            driver.publish_completed_after_cleanup(output, pending, budget)
            actual = json.loads((output / 'completion.json').read_text())
            self.assertEqual(actual['status'], 'both120_completed')
            self.assertTrue(actual['cleanup_verified'])
            self.assertNotIn('intended_status', actual)

    def test_cleanup_failure_preserves_the_original_main_exception(self):
        errors = [{'resource': 'log_close', 'error': 'injected close failure'}]
        with mock.patch.object(driver, 'cleanup_resources', return_value=errors), \
             mock.patch.object(driver, 'publish_completed_after_cleanup', side_effect=AssertionError('false completion')), \
             mock.patch.object(driver.training, 'acquire_runtime_locks', side_effect=AssertionError('runtime touched')), \
             mock.patch.object(driver.sys, 'stderr', io.StringIO()) as stderr:
            with self.assertRaisesRegex(RuntimeError, 'explicit approval'):
                driver.main([])
            self.assertIn('MC2 resource cleanup failed', stderr.getvalue())

    def test_actual_helper_and_cursor_observers_together_preserve_two_complete_updates(self):
        import test_conversion_score_step as fixture
        for arm, function in ((driver.ARMS[0], driver.mc.train_batch_mc1), (driver.ARMS[1], driver.mc.train_batch_mc2)):
            args = fixture.make_fixture(torch.float32)
            args[6].param_groups[0]['lr'] = 5e-5
            # 仅给四参数小学生/小D建立模拟3327 Adam状态，不读真实权重。
            for model, optimizer in ((args[2], args[6]), (args[4], args[7])):
                update(model, optimizer)
                for value in optimizer.state.values(): value['step'].fill_(3327)
            state = driver.new_arm_state(arm, 'toy'); state['restore_completed'] = True
            for step in (3328, 3329):
                fixture.reset_rng(201234 + step)
                receipt = {}
                result = driver.complete_update(state, step, args[6], args[7],
                    lambda: function(*args, global_step=step, mc_receipt=receipt))
                self.assertEqual(result['dist_raw'], receipt['dist_mixed'])
                self.assertEqual(state['completed_global_step'], step)
                self.assertEqual(state['adam_step_returns'], {'student': 1, 'discriminator': 1})
                self.assertFalse(state['partial_batch_may_have_updated_D'])
                self.assertNotIn('step', args[6].__dict__)
                self.assertNotIn('step', args[7].__dict__)

    def test_forward_ledger_counts_real_attempts_and_enforces_limits_before_call(self):
        ledger = driver.ForwardLedger()
        model = nn.Linear(2, 1)
        handle = model.register_forward_pre_hook(ledger.before('student'))
        try:
            model(torch.ones(1, 2))
            self.assertEqual(ledger.counts['student'], 1)
            ledger.counts['student'] = driver.FORWARD_LIMITS['student']
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                model(torch.ones(1, 2))
            self.assertEqual(ledger.counts['student'], driver.FORWARD_LIMITS['student'])
        finally: handle.remove()
        self.assertEqual(driver.FORWARD_LIMITS['teacher'], 360 + 2 * (5 + 3 * 20))
        self.assertEqual(driver.FORWARD_LIMITS['student'], 240 * 2 + 2 * (5 * 2 + 3 * 2))
        self.assertEqual(sum(driver.FORWARD_LIMITS[k] for k in ('active_D_branches', 'fixed_D_branches')), 7840)

    def test_failure_receipt_retains_saved_cursor_and_invalidates_own_late_completion(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(driver.pair, 'available_bytes', return_value=20 * 1024**3):
            output = Path(name) / 'trial'; output.mkdir(); budget = driver.Budget(output)
            budget.write(output / 'completion.json', {'status': 'old success'})
            state = driver.new_arm_state(driver.ARMS[1], 'tiny')
            state.update(last_full_saved_global_step=3328, full_snapshot={'committed': True, 'global_step': 3328})
            budget.cancel_future_on_fatal_abort('time_budget')
            result = driver.failure_record(output, state, {}, RuntimeError('late stop'), budget, driver.ForwardLedger())
            self.assertEqual(result['active_arm']['last_full_saved_global_step'], 3328)
            self.assertEqual(json.loads((output / 'completion.json').read_text())['status'], 'execution_aborted')
            self.assertFalse(budget.pending)
            self.assertEqual(set(budget.cancelled), {'full', 'student'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
