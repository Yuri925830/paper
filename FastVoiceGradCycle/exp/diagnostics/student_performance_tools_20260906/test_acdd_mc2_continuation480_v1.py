"""仅租用机CPU测试新480步调度、3447父状态和保存边界；不读真实大权重。"""

import ast
import copy
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only Matpool CPU tests with CUDA_VISIBLE_DEVICES empty')
import torch
from torch import nn
import run_acdd_mc2_continuation480_v1 as d


def losses(value=1.): return {key: float(value) for key in d.pair.LOSS_KEYS}


def entry(step, mel=.19065389037132263, rms=1.):
    result = {'global_step': step, 'development': {'mel_l1': mel, 'prediction_rms': rms}}
    if step - d.START in d.UNIFORM64_UPDATES: result['uniform64'] = {'uniform_weighted_score': .047756376620964144}
    return result


def update(model, optimizer):
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)


def toy():
    student, discriminator = nn.Linear(2, 1), nn.Linear(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5, betas=(.5, .9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .9))
    for model, opt in ((student, optimizer), (discriminator, d_optimizer)):
        update(model, opt)
        for state in opt.state.values(): state['step'].fill_(d.START)
    state = d.new_state('tiny-contract'); state['restore_completed'] = True
    return state, student, discriminator, optimizer, d_optimizer


def toy_call(parts):
    state, student, discriminator, optimizer, d_optimizer = parts
    def call(): update(discriminator, d_optimizer); update(student, optimizer); return losses()
    return call


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_default_refuses_before_locks_or_gpu_and_requires_all_four_review_shas(self):
        with mock.patch.object(d.training, 'acquire_runtime_locks', side_effect=AssertionError('runtime touched')):
            with self.assertRaisesRegex(RuntimeError, 'explicit approval'): d.main([])
        args = ['--run-approved', '--parent-checkpoint-sha256', d.PARENT_SHA]
        with self.assertRaisesRegex(RuntimeError, 'reviewed SHA'): d.launch_arguments(args)
        for key in ('protocol', 'helper', 'driver', 'tests'): args += ['--' + key + '-sha256', 'a' * 64]
        self.assertEqual(d.launch_arguments(args).max_seconds, 1800)
        with self.assertRaises(RuntimeError): d.launch_arguments(args + ['--max-seconds', '3600'])

    def test_no_frozen_globals_mutated_and_old120_still_bounded(self):
        self.assertEqual((d.old.START, d.old.STOP, d.old.UPDATES), (3327, 3447, 120))
        self.assertEqual(d.old.previous.META_LIMIT, 16 * 1024**2)
        self.assertEqual(d.old.FORWARD_LIMITS['teacher'], 490)
        tree = ast.parse(Path(d.__file__).read_text())
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for name in ('stop_rules', 'candidate_gate', 'permits_candidate', 'stratified_timesteps', 'validate_uniform_initial', 'validate_parent'):
            self.assertNotIn(name, calls)
        self.assertNotIn('train_batch_mc1', calls)
        self.assertEqual(d.WINDOW_STARTS, (3448, 3568, 3688, 3808))

    def test_budget_one_full_exact_32MiB_no_student_reserve(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            output = Path(name) / 'new'; b = d.Budget(output)
            self.assertEqual(b.require()['required_bytes'], 4815728640)
            self.assertEqual(set(b.pending), {'full'})
            with mock.patch.object(d.pair, 'available_bytes', return_value=4815728639):
                with self.assertRaises(d.pair.InsufficientDisk): b.require()
            with mock.patch.object(b, 'metadata_used', return_value=d.META_LIMIT):
                with self.assertRaises(d.pair.InsufficientDisk): b.require(1)
            output.mkdir(); (output / 'full.pt').write_bytes(b'toy')
            b.committed('full', output / 'full.pt')
            self.assertFalse(b.pending)
            self.assertEqual(b.require()['required_bytes'], 4815728640 - d.FULL_BYTES - d.previous.MODEL_SLACK)

    def test_metadata_links_and_existing_exclusive_log_refuse(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            output = Path(name) / 'new'; budget = d.Budget(output)
            log = d.old.replication.open_exclusive_log(output, budget); log.close()
            with self.assertRaises(RuntimeError): d.old.replication.open_exclusive_log(output, budget)
            output.mkdir(); outside = Path(name) / 'outside'; outside.mkdir()
            (output / 'link').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(RuntimeError): budget.write(output / 'link/x.json', {})

    def simulate(self, monitor_values=None, stop_at=None):
        state = d.new_state('tiny'); initial = entry(d.START); state['last_monitor'] = initial
        future = [{'global_step': step} for step in range(d.START + 1, d.STOP + 1)]
        calls, saved = [], []
        def step(row):
            state['completed_global_step'] = row['global_step']; state['new_updates'] = row['global_step'] - d.START
            return dict(losses(1. + state['new_updates'] * 1e-7), global_step=row['global_step'])
        def evaluate(number):
            calls.append(number - d.START)
            return entry(number, (monitor_values or {}).get(number - d.START, .19065389037132263))
        def publish(name, value, append=True): saved.append((name, copy.deepcopy(value)))
        def check():
            if stop_at is not None and state['new_updates'] >= stop_at: raise d.BoundaryStop('time_budget')
        return state, future, step, evaluate, publish, check, initial, calls, saved

    def test_complete480_exact17_pair9_uniform_four120windows(self):
        setup = self.simulate()
        with contextlib.redirect_stdout(io.StringIO()): result = d.run_updates(*setup[:7])
        self.assertEqual(setup[7], list(range(30, 481, 30)))
        self.assertEqual(len(result['monitors']), 17)
        self.assertEqual(len([row for row in result['monitors'] if 'uniform64' in row]), 9)
        self.assertEqual([row['first_global_step'] for row in result['windows']], list(d.WINDOW_STARTS))
        self.assertEqual([row['global_step'] for row in result['windows']], [3567, 3687, 3807, 3927])
        self.assertEqual(result['reason'], 'maximum480_updates_completed')
        self.assertEqual(len([row for row in setup[8] if row[0] == 'step_losses.jsonl']), 480)
        self.assertTrue(all(not row['used_as_stop_gate'] for row in result['windows']))
        self.assertGreater(result['windows'][-1]['means']['student_total'], result['windows'][0]['means']['student_total'])
        trend = d.primary_loss_trend(result['rows'])
        self.assertEqual(trend['first_global_steps'], [3448, 3687]); self.assertEqual(trend['last_global_steps'], [3688, 3927])
        self.assertGreater(trend['student_total_relative_change'], 0.)
        self.assertFalse(d.primary_loss_trend(result['rows'][:-1])['available'])

    def test_guard_original3327_and_no_double_count_initial3447(self):
        setup = self.simulate({30: .201, 60: .201})
        with contextlib.redirect_stdout(io.StringIO()): result = d.run_updates(*setup[:7])
        self.assertEqual(result['reason'], 'maximum480_updates_completed')  # 高于3447的5%，仍低于原3327的5%。
        setup = self.simulate({30: .203, 60: .204})
        result = d.run_updates(*setup[:7])
        self.assertEqual(setup[0]['new_updates'], 60)
        self.assertEqual(result['reason'], 'two_consecutive_DEV32_above3327_by5percent')
        self.assertEqual(setup[7], [30, 60])
        with self.assertRaises(RuntimeError): d.guard_reasons([entry(3447), entry(3447)], d.BASELINE3327)
        self.assertEqual(d.BASELINE3327['mel_l1'] * 1.05, .20217022821307185)

    def test_original_S0_finite_and_exact_boundaries(self):
        upper, lower = .2193548917770386, .7995790004730225
        self.assertFalse(d.guard_reasons([entry(3447, upper, lower)], d.BASELINE3327))
        self.assertIn('original_S0_extreme_guard', d.guard_reasons([entry(3447, upper + 1e-9)], d.BASELINE3327))
        with self.assertRaises(RuntimeError): d.guard_reasons([entry(3447, float('nan'))], d.BASELINE3327)

    def test_timeout_no_extra_monitor_no_omitted_rows_or_fabricated_window(self):
        setup = self.simulate(stop_at=17)
        with self.assertRaisesRegex(d.BoundaryStop, 'time_budget'): d.run_updates(*setup[:7])
        self.assertEqual(setup[0]['new_updates'], 17); self.assertEqual(setup[7], [])
        self.assertEqual(len([x for x in setup[8] if x[0] == 'step_losses.jsonl']), 17)
        setup = self.simulate(); setup[1].pop()
        with self.assertRaises(RuntimeError): d.run_updates(*setup[:7])

    def test_actual_global_batch_plan_does_not_reset_at_pass3688(self):
        development = json.loads((d.DEV / 'development_contract.json').read_text())
        rows = d.future_plan(development); plan = d.resume.GlobalBatchPlan(development['retained_train_indices'])
        self.assertEqual(len(rows), 480); self.assertEqual(rows[0]['crop_seed'], 104682)
        self.assertEqual(rows[0]['noise_seed'], 204682)
        self.assertEqual(rows[240]['global_step'], 3688)
        self.assertEqual(rows[240]['indices'], plan.for_step(3688))
        self.assertNotEqual(rows[240]['indices'], plan.for_step(1))
        self.assertFalse(any(set(row['indices']) & set(development['excluded_train_indices']) for row in rows))

    def test_complete_actual_two_adam_updates_and_3447_boundary(self):
        parts = toy(); state, student, discriminator, optimizer, d_optimizer = parts
        result = d.complete_update(state, 3448, optimizer, d_optimizer, toy_call(parts))
        self.assertEqual(result, losses()); self.assertEqual(state['completed_global_step'], 3448)
        self.assertEqual(state['new_updates'], 1); self.assertFalse(state['partial_batch_may_have_updated_D'])
        self.assertEqual(state['adam_step_returns'], {'student': 1, 'discriminator': 1})
        self.assertNotIn('step', optimizer.__dict__)
        with self.assertRaises(RuntimeError): d.complete_update(state, 3448, optimizer, d_optimizer, toy_call(parts))
        state['completed_global_step'] = d.STOP
        with self.assertRaises(RuntimeError): d.complete_update(state, d.STOP + 1, optimizer, d_optimizer, toy_call(parts))

    def test_half_D_error_does_not_advance_and_cannot_save(self):
        parts = toy(); state, student, discriminator, optimizer, d_optimizer = parts
        def half(): update(discriminator, d_optimizer); raise RuntimeError('half')
        with self.assertRaisesRegex(RuntimeError, 'half'): d.complete_update(state, 3448, optimizer, d_optimizer, half)
        self.assertEqual(state['completed_global_step'], 3447); self.assertTrue(state['partial_batch_may_have_updated_D'])
        with self.assertRaises(RuntimeError): d.assert_saveable(*parts)

    def test_Adam_raised_after_counter_increment_is_uncertain(self):
        parts = toy(); state, student, discriminator, optimizer, d_optimizer = parts
        original = optimizer.step
        def fail(*args, **kwargs): original(*args, **kwargs); raise RuntimeError('Adam before return')
        optimizer.step = fail
        with self.assertRaisesRegex(RuntimeError, 'Adam before return'): d.complete_update(state, 3448, optimizer, d_optimizer, toy_call(parts))
        self.assertEqual(state['completed_global_step'], 3447)
        self.assertTrue(state['partial_batch_may_have_updated_D']); self.assertIs(optimizer.step, fail)

    def test_post_two_step_exception_keeps_true_complete_step(self):
        parts = toy(); state, student, discriminator, optimizer, d_optimizer = parts
        def post(): toy_call(parts)(); raise RuntimeError('receipt failed')
        with self.assertRaisesRegex(RuntimeError, 'receipt failed'): d.complete_update(state, 3448, optimizer, d_optimizer, post)
        self.assertEqual(state['completed_global_step'], 3448); self.assertFalse(state['partial_batch_may_have_updated_D'])
        d.assert_saveable(*parts)

    def save_fixture(self, name):
        parts = toy(); state = parts[0]
        d.complete_update(state, 3448, parts[3], parts[4], toy_call(parts))
        state.update(reason='tiny_complete', last_monitor=entry(3447))
        output = Path(name) / 'new'; output.mkdir(); budget = d.Budget(output)
        return parts, budget, SimpleNamespace(saving=lambda: None), SimpleNamespace(as_dict=lambda: {'toy': True})

    def test_one_full_no_S_only_no_overwrite(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            parts, budget, clock, config = self.save_fixture(name)
            d.save_endpoint(parts[0], config, *parts[1:], budget, clock)
            self.assertEqual([path.name for path in budget.output.glob('*.pt')], ['final_full.pt'])
            payload = torch.load(budget.output / 'final_full.pt', map_location='cpu')
            self.assertEqual(payload['global_step'], 3448); self.assertTrue(payload['resumable'])
            self.assertEqual(payload['mc_samples'], 2); self.assertIsNone(payload['development'])
            self.assertEqual(payload['development_monitor_global_step'], 3447)
            for name in ('student', 'discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state'): self.assertIn(name, payload)
            with self.assertRaises(RuntimeError): d.save_endpoint(parts[0], config, *parts[1:], budget, clock)

    def test_no_updates_no_duplicate_parent_save(self):
        with tempfile.TemporaryDirectory() as name:
            parts = toy(); budget = d.Budget(Path(name) / 'new')
            d.save_endpoint(parts[0], None, *parts[1:], budget, None)
            self.assertFalse(budget.output.exists()); self.assertEqual(parts[0]['last_full_saved_global_step'], 3447)

    def test_hash_failure_after_commit_still_records_saved3448(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            parts, budget, clock, config = self.save_fixture(name)
            with mock.patch.object(d.base, 'sha256_file', side_effect=RuntimeError('hash failure')):
                with self.assertRaisesRegex(RuntimeError, 'hash failure'): d.save_endpoint(parts[0], config, *parts[1:], budget, clock)
            self.assertEqual(parts[0]['last_full_saved_global_step'], 3448)
            self.assertTrue(parts[0]['full_snapshot']['committed']); self.assertFalse(budget.pending)

    def test_serialization_failure_retains_tmp_and_parent_cursor(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            parts, budget, clock, config = self.save_fixture(name)
            with mock.patch.object(d.torch, 'save', side_effect=RuntimeError('serialization failed')):
                with self.assertRaisesRegex(RuntimeError, 'serialization failed'): d.save_endpoint(parts[0], config, *parts[1:], budget, clock)
            self.assertEqual(parts[0]['last_full_saved_global_step'], 3447)
            self.assertTrue((budget.output / 'final_full.pt.tmp').exists())

    def test_nonfinite_model_or_moment_cannot_be_marked_resumable(self):
        for role in ('model', 'moment'):
            parts = toy()
            with torch.no_grad():
                if role == 'model': next(parts[1].parameters()).fill_(float('nan'))
                else: next(iter(parts[3].state.values()))['exp_avg'].fill_(float('inf'))
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'): d.assert_saveable(*parts)

    def test_actual3447_metadata_and_no_archived_control_dependency(self):
        contract = json.loads((d.PARENT_RUN / 'run_contract.json').read_text())
        completion = json.loads((d.PARENT_RUN / 'completion.json').read_text())
        receipt = d.validate_parent_metadata(contract, completion)
        self.assertEqual(receipt['completed_global_step'], 3447)
        for corrupt in ('cleanup_verified', 'status'):
            invalid = copy.deepcopy(completion); invalid[corrupt] = False
            with self.assertRaises(RuntimeError): d.validate_parent_metadata(contract, invalid)
        self.assertNotIn(str(d.PARENT_RUN / 'control_mc1/student_step_003447.pt'), contract['file_sha256'])

    def test_initial_replay_checks_whole_uniform_not_only_mean(self):
        reference = json.loads((d.PARENT_RUN / 'completion.json').read_text())['arms']['candidate_mc2']['last_monitor']
        d.validate_replay(copy.deepcopy(reference), reference)
        invalid = copy.deepcopy(reference); invalid['uniform64']['per_utterance_GT_l1'][0] += 1e-6
        with self.assertRaisesRegex(RuntimeError, 'uniform64'): d.validate_replay(invalid, reference)

    def parent_fixture(self):
        parts = toy(); state, student, discriminator, optimizer, d_optimizer = parts
        config = SimpleNamespace(as_dict=lambda: {'tiny_cpu_test': True})
        payload = {'global_step': 3447, 'run_contract_sha256': d.PARENT_CANONICAL, 'effective_optimizer_lrs': dict(d.LRS),
            'mc_samples': 2, 'mc_private_seed_base': 74000000, 'resumable': True, 'epoch': 0,
            'configuration': config.as_dict(), 'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar',
            'development': entry(3447)['development'], 'development_monitor_global_step': 3447,
            'stop_reason': 'maximum120_updates_completed', 'execution_error': None,
            'rng_state': d.training.capture_rng_state(), 'student': copy.deepcopy(student.state_dict()),
            'discriminator': copy.deepcopy(discriminator.state_dict()),
            'student_optimizer': copy.deepcopy(optimizer.state_dict()), 'discriminator_optimizer': copy.deepcopy(d_optimizer.state_dict())}
        receipt = {'last_monitor': entry(3447)}; contract = {'configuration': config.as_dict()}
        return parts, payload, config, contract, receipt

    def test_real_toy3447_parent_restores_models_twoAdam_RNG_without_payload_aliases(self):
        parts, payload, config, contract, receipt = self.parent_fixture()
        state, student, discriminator, optimizer, d_optimizer = parts
        expected = d.previous.content_state(student, discriminator, optimizer, d_optimizer)
        before_payload = d.probe.tree_sha(payload)
        update(student, optimizer); update(discriminator, d_optimizer); d.training.seed_everything(7)
        with mock.patch.object(d, 'PARENT_RNG_SHA', d.probe.tree_sha(payload['rng_state'])):
            actual = d.restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, contract, receipt, torch.device('cpu'))
        self.assertEqual(actual, expected)
        self.assertEqual(d.actual_cursors(optimizer, d_optimizer)['student']['uniform_step'], 3447)
        self.assertEqual(d.probe.tree_sha(payload), before_payload)
        for role, model, opt in (('student', student, optimizer), ('discriminator', discriminator, d_optimizer)):
            for key, tensor in model.state_dict().items(): self.assertNotEqual(tensor.data_ptr(), payload[role][key].data_ptr())
            for identifier, values in opt.state_dict()['state'].items():
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    self.assertNotEqual(values[key].data_ptr(), payload[role + '_optimizer']['state'][identifier][key].data_ptr())

    def test_toy_parent_rejects_wrong_header_config_RNG_and_moment(self):
        for key, value in (('global_step', 3327), ('mc_samples', 1), ('resumable', False), ('configuration', {}), ('execution_error', 'old failure')):
            parts, payload, config, contract, receipt = self.parent_fixture(); payload[key] = value
            with self.assertRaises(RuntimeError): d.restore_parent(payload, *parts[1:], config, contract, receipt, torch.device('cpu'))
        parts, payload, config, contract, receipt = self.parent_fixture()
        with mock.patch.object(d, 'PARENT_RNG_SHA', '0' * 64):
            with self.assertRaisesRegex(RuntimeError, 'RNG'): d.restore_parent(payload, *parts[1:], config, contract, receipt, torch.device('cpu'))
        for corrupt in ('step', 'moment'):
            parts, payload, config, contract, receipt = self.parent_fixture()
            saved = next(iter(payload['student_optimizer']['state'].values()))
            if corrupt == 'step': saved['step'].fill_(3446)
            else: saved['exp_avg'].fill_(float('nan'))
            with mock.patch.object(d, 'PARENT_RNG_SHA', d.probe.tree_sha(payload['rng_state'])):
                with self.assertRaises(RuntimeError): d.restore_parent(payload, *parts[1:], config, contract, receipt, torch.device('cpu'))

    def test_new_actual_mc2_helper_with_real_toy_Adam_observers_two_steps(self):
        import acdd_mc2_train_batch_3448_3927_v1 as mc
        import test_conversion_score_step as fixture
        args = fixture.make_fixture(torch.float32); args[6].param_groups[0]['lr'] = 5e-5
        for model, optimizer in ((args[2], args[6]), (args[4], args[7])):
            update(model, optimizer)
            for value in optimizer.state.values(): value['step'].fill_(3447)
        state = d.new_state('toy'); state['restore_completed'] = True
        for step in (3448, 3449):
            fixture.reset_rng(201234 + step); receipt = {}
            result = d.complete_update(state, step, args[6], args[7], lambda: mc.train_batch_mc2(*args, global_step=step, mc_receipt=receipt))
            self.assertEqual(result['dist_raw'], receipt['dist_mixed'])
            self.assertEqual(state['completed_global_step'], step)
            self.assertEqual(state['adam_step_returns'], {'student': 1, 'discriminator': 1})
            self.assertFalse(state['partial_batch_may_have_updated_D'])

    def test_forward_counts_exact_and_limit_enforced(self):
        self.assertEqual(d.FORWARD_LIMITS['teacher'], 480 * 2 + 17 + 9 * 20)
        self.assertEqual(d.FORWARD_LIMITS['student'], 480 * 2 + 17 * 2 + 9 * 2)
        self.assertEqual(d.FORWARD_LIMITS['vocoder'], 480 * 3 + 17 * 2)
        self.assertEqual(d.FORWARD_LIMITS['active_D_branches'], 480 * 32)
        self.assertEqual(d.FORWARD_LIMITS['fixed_D_branches'], 17 * 16)
        ledger = d.ForwardLedger(); model = nn.Linear(2, 1)
        handle = model.register_forward_pre_hook(ledger.before('student'))
        try:
            ledger.counts['student'] = d.FORWARD_LIMITS['student']
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'): model(torch.ones(1, 2))
        finally: handle.remove()

    def test_cleanup_shared_implementation_and_terminal_requires_cleanup(self):
        self.assertIs(d.cleanup_resources, d.old.cleanup_resources)
        self.assertIs(d.publish_completed_after_cleanup, d.old.publish_completed_after_cleanup)
        with tempfile.TemporaryDirectory() as name, mock.patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            output = Path(name) / 'new'; output.mkdir(); budget = d.Budget(output)
            completion = {'status': 'finalization_pending', 'cleanup_verified': False, 'intended_status': 'completed480'}
            d.publish_completed_after_cleanup(output, completion, budget)
            self.assertEqual(json.loads((output / 'completion.json').read_text())['status'], 'completed480')


if __name__ == '__main__': unittest.main(verbosity=2)
