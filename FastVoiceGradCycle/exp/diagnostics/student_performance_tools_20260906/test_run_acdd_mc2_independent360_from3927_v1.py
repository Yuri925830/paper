"""租用CPU的独立360薄层测试；不读真实checkpoint，不调用训练main或GPU。"""

import ast
import copy
import inspect
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Run only on rental CPU with CUDA hidden')
import torch
import run_acdd_mc2_independent360_from3927_v1 as f
import evaluate_development_candidates as evaluator


def isolated(function, **replacements):
    function = inspect.unwrap(function)
    namespace = dict(function.__globals__); namespace.update(replacements)
    copied = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def raw(fm=3.):
    return {'student_total': 2. + 2 * fm + 4.5, 'student_adv': 2., 'fm_raw': fm, 'fm_weighted': 2 * fm,
        'dist_raw': .1, 'dist_weighted': 4.5, 'discriminator': 4.}


def monitor(step, mel=.19, rms=1.):
    return {'global_step': step, 'development': {'mel_l1': mel, 'prediction_rms': rms},
        'fixed_train': {'student_total': 6.}, 'uniform64': {'GT_l1': .191, 'computed_SHA': 'a'}}


def update(model, optimizer):
    for parameter in model.parameters(): parameter.grad = torch.full_like(parameter, .25)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)


def toy(step=3927):
    torch.manual_seed(713)
    student, discriminator = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5, betas=(.5, .9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .9))
    for model, opt in ((student, optimizer), (discriminator, d_optimizer)):
        update(model, opt)
        for value in opt.state.values(): value['step'].fill_(step)
    count = step - f.START; state = f.new_state('a' * 64)
    state.update(restore_completed=True, completed_global_step=step, actual_updates_this_process=count, new_updates=count,
        overlapping_historical_step_numbers=min(count, 120), beyond_historical_step_numbers=max(0, count - 120),
        historical_input_checks=min(count, 120), complete_loss_records=count, last_monitor=monitor(step),
        monitor_global_steps=list(range(f.START, step + 1, 60)), reason='maximum360_updates_completed')
    return state, student, discriminator, optimizer, d_optimizer


def historical_row(step=3928):
    return dict(raw(), global_step=step, indices=[1, 2], crop_seed=101234 + step, noise_seed=201234 + step,
        input_identity={'batch_tensors': {'mel': {'sha256': 'a' * 64}}}, actual_forward_counts=dict(f.TRAIN_COUNTS),
        mc={'mc_samples': 2, 'global_step': step, 'source_noise': 'source', 'score_noise1': 'n1', 'score_noise2': 'n2',
            't1': 't1', 't2': 't2', 't1_values': [1, 1000], 't2_values': [3, 800], 'private_seed': 74000000 + step,
            'dist1': .1, 'dist2': .1, 'first': {'teacher_target': {'sha256': 'b' * 64}, 'per_utterance_l1': [.19]},
            'second': {'teacher_target': {'sha256': 'c' * 64}, 'per_utterance_l1': [.18]}})


class Independent360Tests(unittest.TestCase):
    def cli(self):
        return ['--run-approved', '--repeatability-reviewed'] + [item
            for name in ('driver', 'tests', 'protocol', 'repeatability-results', 'repeatability-completion')
            for item in ('--' + name + '-sha256', 'a' * 64)]

    def test_explicit_run_and_AA_review_with_five_full_SHA_values(self):
        self.assertTrue(f.arguments(self.cli()).repeatability_reviewed)
        for flag in ('--run-approved', '--repeatability-reviewed'):
            args = self.cli(); args.remove(flag)
            with self.assertRaises(RuntimeError): f.arguments(args)
        for index in (3, 5, 7, 9, 11):
            for value in ('a' * 63, 'g' * 64, 'A' * 64):
                args = self.cli(); args[index] = value
                with self.subTest(index=index), self.assertRaises(RuntimeError): f.arguments(args)

    def test_fresh_CPU_runtime_records_real_threads_without_forcing_them(self):
        code = '''import json
import torch
before = torch.get_num_threads()
import run_acdd_mc2_independent360_from3927_v1 as f
record = f.runtime()
print(json.dumps({'before': before, 'after': torch.get_num_threads(), 'record': record,
    'again': f.runtime(), 'cuda_initialized': torch.cuda.is_initialized()}))
'''
        env = dict(os.environ, **f.THREAD_ENV, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=str(f.TOOLS) + os.pathsep + str(f.ROOT))
        run = subprocess.run([f.sys.executable, '-c', code], env=env, cwd=f.ROOT, capture_output=True, text=True, timeout=30, check=True)
        result = json.loads(run.stdout)
        self.assertGreater(result['before'], 0); self.assertEqual(result['before'], result['after'])
        self.assertEqual(result['record']['torch_threads'], result['before']); self.assertEqual(result['record'], result['again'])
        self.assertEqual(result['record']['thread_environment'], f.THREAD_ENV); self.assertFalse(result['cuda_initialized'])

    def test_frozen_restore_plan_storage_reused_without_old_main_or_math_changes(self):
        self.assertIs(f.restore_parent, f.prior.restore_parent)
        self.assertIs(f.future_plan, f.prior.future_plan); self.assertIs(f.Budget, f.prior.Budget)
        development = {'retained_train_indices': list(range(39335)), 'excluded_train_indices': list(range(39335, 39387))}
        actual = f.future_plan(development)
        self.assertEqual(len(actual), 360); self.assertEqual(actual[:120], f.d.future_plan(development))
        self.assertEqual((actual[0]['global_step'], actual[0]['crop_seed'], actual[0]['noise_seed']), (3928, 105162, 205162))
        self.assertEqual((actual[-1]['global_step'], actual[-1]['crop_seed'], actual[-1]['noise_seed']), (4287, 105521, 205521))

    def test_historical_inputs_strict_computed_loss_target_and_per_sentence_values_descriptive(self):
        control = historical_row(); row = copy.deepcopy(control)
        row.update(raw(33.)); row['mc']['first']['teacher_target']['sha256'] = 'd' * 64
        row['mc']['first']['per_utterance_l1'] = [.809]; row['mc']['dist2'] = 8.1
        comparison = f.historical_input_check(row, control)
        self.assertTrue(comparison['inputs_match']); self.assertFalse(comparison['teacher_target_SHA_equal']['first'])
        self.assertGreater(comparison['raw_loss_delta_from_historical']['student_total'], 0)
        self.assertFalse(comparison['historical_state_reconstruction_claimed'])
        for key in ('global_step', 'indices', 'crop_seed', 'noise_seed', 'input_identity'):
            bad = copy.deepcopy(control); bad[key] = [] if key == 'indices' else 'changed'
            with self.subTest(key=key), self.assertRaises((RuntimeError, TypeError)): f.historical_input_check(bad, control)
        for key in ('source_noise', 'score_noise1', 'score_noise2', 't1', 't2', 't1_values', 't2_values', 'private_seed', 'mc_samples', 'global_step'):
            bad = copy.deepcopy(control); bad['mc'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(RuntimeError): f.historical_input_check(bad, control)
        bad = copy.deepcopy(control); bad['actual_forward_counts']['teacher'] = 3
        with self.assertRaises(RuntimeError): f.historical_input_check(bad, control)
        with self.assertRaises(RuntimeError): f.historical_input_check(historical_row(4048), historical_row(4048))

    def test_deadline_wrapper_has_exact_real_Adam_model_moments_RNG_equivalence(self):
        state, s, D, a, b = toy(); before = f.previous.content_state(s, D, a, b); check = Mock()
        with f.deadline_steps(a, b, check): update(D, b); update(s, a)
        actual = f.previous.content_state(s, D, a, b)
        _, s2, D2, a2, b2 = toy(); self.assertEqual(f.previous.content_state(s2, D2, a2, b2), before)
        update(D2, b2); update(s2, a2)
        self.assertEqual(actual, f.previous.content_state(s2, D2, a2, b2)); self.assertEqual(check.call_count, 2)
        self.assertNotIn('step', a.__dict__); self.assertNotIn('step', b.__dict__)

    def test_true_complete_cursor_and_half_or_poststep_failure_are_preserved(self):
        for mode in ('normal', 'D_only_deadline', 'after_both'):
            state, s, D, a, b = toy()
            def check():
                if mode == 'D_only_deadline' and f.old.optimizer_cursor(b)['uniform_step'] == 3928: raise RuntimeError('deadline')
            def call():
                with f.deadline_steps(a, b, check):
                    update(D, b); update(s, a)
                    if mode == 'after_both': raise RuntimeError('after_both')
                    return raw()
            if mode == 'normal': f.complete_update(state, 3928, a, b, call)
            else:
                with self.subTest(mode=mode), self.assertRaises(RuntimeError): f.complete_update(state, 3928, a, b, call)
            expected = 3927 if mode == 'D_only_deadline' else 3928
            self.assertEqual(state['completed_global_step'], expected)
            self.assertEqual(state['partial_batch_may_have_updated_D'], mode == 'D_only_deadline')
            if mode == 'after_both': self.assertTrue(state['exception_after_both_adam_steps'])
            self.assertNotIn('replay_updates', state); self.assertNotIn('replay_verified_steps', state)
            self.assertNotIn('step', a.__dict__); self.assertNotIn('step', b.__dict__)

    def test_actual_counts_cross4047_without_historical_state_reconstruction(self):
        state, s, D, a, b = toy(4047)
        def call(): update(D, b); update(s, a); return raw()
        f.complete_update(state, 4048, a, b, call)
        self.assertEqual((state['actual_updates_this_process'], state['overlapping_historical_step_numbers'], state['beyond_historical_step_numbers']), (121, 120, 1))
        for step in (3928, 4048, 4288):
            with self.assertRaises(RuntimeError): f.complete_update(state, step, a, b, call)

    def simulate(self, mel=None, initial_mel=.19, fail_after=None):
        state = toy()[0]; written, called = [], []
        future = [{'global_step': step} for step in range(3928, 4288)]
        def step_call(planned):
            number = planned['global_step']; count = number - f.START
            state.update(completed_global_step=number, new_updates=count, actual_updates_this_process=count,
                overlapping_historical_step_numbers=min(count, 120), beyond_historical_step_numbers=max(0, count - 120), historical_input_checks=min(count, 120))
            return dict(raw(3. + count * .01), global_step=number, mc={'dist1': .1, 'dist2': .1})
        def evaluate(number): called.append(number); return monitor(number, (mel or {}).get(number, .19))
        def publish(name, value, append=True): written.append((name, copy.deepcopy(value)))
        def check():
            if fail_after is not None and state['new_updates'] >= fail_after: raise RuntimeError('bounded time')
        return state, future, step_call, evaluate, publish, check, monitor(3927, initial_mel), written, called

    def test_360_all_actual_updates_three_windows_seven_monitors_no_float_replay_gate(self):
        parts = self.simulate(initial_mel=.20); result = f.run_updates(*parts[:7])
        self.assertEqual(len(result['rows']), 360); self.assertEqual(len(result['windows']), 3); self.assertEqual(len(result['monitors']), 7)
        self.assertEqual(parts[8], [3987, 4047, 4107, 4167, 4227, 4287]); self.assertEqual(result['reason'], 'maximum360_updates_completed')
        self.assertEqual(parts[0]['actual_updates_this_process'], 360)
        trend = f.loss_trend(result['rows']); self.assertGreater(trend['primary_third_minus_second_window']['student_total'], 0)
        self.assertTrue(trend['all360_are_actual_updates_of_this_independent_trajectory']); self.assertFalse(trend['used_as_stop_gate'])
        self.assertTrue(trend['all_three_windows'][0]['overlaps_historical_step_numbers'])
        self.assertTrue(all(w['phase'] == 'independent_trajectory' for w in result['windows']))
        self.assertFalse(f.loss_trend(result['rows'][:240])['available'])

    def test_original_protection_uses_only_actual_distinct_monitors_zero_guard_no_updates(self):
        parts = self.simulate({3987: .203, 4047: .204}); result = f.run_updates(*parts[:7])
        self.assertEqual(parts[0]['completed_global_step'], 4047); self.assertEqual(result['reason'], 'two_consecutive_DEV32_above3327_by5percent')
        guards = [value for name, value in parts[7] if name == 'guard_checks.jsonl']
        self.assertEqual(guards[-1]['distinct_global_steps_used'], [3987, 4047]); self.assertFalse(guards[-1]['historical_points_pooled'])
        parts = self.simulate({3987: .22}); result = f.run_updates(*parts[:7])
        self.assertEqual(parts[0]['completed_global_step'], 3987); self.assertEqual(result['reason'], 'original_S0_extreme_guard')
        parts = self.simulate(initial_mel=.22); result = f.run_updates(*parts[:7])
        self.assertEqual(result['rows'], []); self.assertEqual(parts[0]['new_updates'], 0); self.assertEqual(parts[8], [])
        parts = self.simulate(fail_after=123)
        with self.assertRaisesRegex(RuntimeError, 'bounded time'): f.run_updates(*parts[:7])
        self.assertEqual(len([row for row in parts[7] if row[0] == 'step_losses.jsonl']), 123)

    def test_forward_checks_deadline_first_and_keeps_original_exact_caps(self):
        self.assertEqual(f.FORWARD_LIMITS, {k: 360 * f.TRAIN_COUNTS[k] + 7 * f.MONITOR_COUNTS[k] for k in f.TRAIN_COUNTS})
        ledger = f.ForwardLedger(); ledger.counts['student'] = f.FORWARD_LIMITS['student'] - 1; ledger.check = Mock()
        hook = ledger.before('student'); hook(None, None)
        with self.assertRaises(RuntimeError): hook(None, None)
        self.assertEqual(ledger.check.call_count, 2); self.assertEqual(ledger.counts['student'], 748)
        ledger.check = Mock(side_effect=RuntimeError('deadline'))
        with self.assertRaisesRegex(RuntimeError, 'deadline'): ledger.before('teacher')(None, None)
        self.assertEqual(ledger.counts['teacher'], 0)

    def test_full_save_actual_configuration_Adam_RNG_newcanonical_and_frozen_evaluator(self):
        state, s, D, a, b = toy(4287); config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        contract = {'kind': 'independent360_toy', 'configuration': config.as_dict(), 'historical_state_reconstruction_claimed': False}
        state['contract_sha256'] = f.training.canonical_json_sha256(contract)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / f.ARM).mkdir(); budget = f.Budget(output); budget.require = Mock(return_value={})
            f.save_endpoint(state, config, s, D, a, b, budget, SimpleNamespace(active=lambda: None))
            path = output / f.ARM / 'final_full.pt'; payload = torch.load(path, map_location='cpu')
            self.assertEqual(payload['global_step'], 4287); self.assertEqual(payload['actual_updates_this_process'], 360)
            self.assertEqual(payload['configuration'], config.as_dict()); self.assertEqual(payload['effective_optimizer_lrs'], f.LRS)
            self.assertFalse(payload['historical_state_reconstruction_claimed']); self.assertTrue(payload['resumable'])
            self.assertEqual(path.stat().st_mode & 0o777, 0o444); self.assertNotIn('full', budget.pending)
            self.assertEqual([p.name for p in path.parent.glob('*.pt')], ['final_full.pt'])
            for key in ('student', 'discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state'): self.assertIn(key, payload)
            self.assertEqual(evaluator.validate_payload(payload, {'epoch': 0, 'global_step': 4287, 'state_key': 'student'}, contract), state['contract_sha256'])
            self.assertEqual(f.probe.tree_sha(payload['rng_state']), f.probe.tree_sha(f.training.capture_rng_state()))

    def test_save_rejects_zero_half_technical_wrong_cursor_config_or_unfinished_endpoint(self):
        for case in ('zero', 'half', 'error', 'afterstep', 'counts', 'input_checks', 'monitor', 'normal4047', 'config', 'cursor'):
            state, s, D, a, b = toy(3927 if case == 'zero' else 4047 if case == 'normal4047' else 4287)
            config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
            if case == 'half': state['partial_batch_may_have_updated_D'] = True
            if case == 'error': state['execution_error'] = 'failure'
            if case == 'afterstep': state['exception_after_both_adam_steps'] = True
            if case == 'counts': state['actual_updates_this_process'] -= 1
            if case == 'input_checks': state['historical_input_checks'] = 119
            if case == 'monitor': state['monitor_global_steps'] = [3927, 4287]
            if case == 'config': config = f.base.replace(config, lambda_fm=1.)
            if case == 'cursor': next(iter(a.state.values()))['step'].fill_(4286)
            with self.subTest(case=case), self.assertRaises(RuntimeError): f.save_endpoint(state, config, s, D, a, b, None, None)

    def test_protective3987_is_its_actual_new_state_and_has_no_automatic_evaluation(self):
        state, s, D, a, b = toy(3987); state['reason'] = 'original_S0_extreme_guard'
        config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / f.ARM).mkdir(); budget = f.Budget(output); budget.require = Mock(return_value={})
            f.save_endpoint(state, config, s, D, a, b, budget, SimpleNamespace(active=lambda: None))
            payload = torch.load(output / f.ARM / 'final_full.pt', map_location='cpu')
            self.assertEqual(payload['global_step'], 3987); self.assertEqual(payload['actual_updates_this_process'], 60)
            self.assertEqual(payload['stop_reason'], 'original_S0_extreme_guard')
        self.assertEqual(f.EVALUATION['normal_endpoint_global_step'], 4287)
        self.assertTrue(f.EVALUATION['protective_endpoint_requires_separate_review']); self.assertFalse(f.EVALUATION['executed_by_this_driver'])

    def test_AA_completed_finite_difference_accepted_but_failure_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {'status': 'two_independent_one_step_diagnostics_completed', 'complete_updates': 2, 'actual_forward_counts': f.AA_COUNTS,
                'all_compared_values_identical': False, 'child_exits': [{'child': 'child_a', 'returncode': 0}, {'child': 'child_b', 'returncode': 0}]}
            result_path = root / 'results.json'; result_path.write_text(json.dumps(result))
            args = SimpleNamespace(repeatability_results_sha256=f.base.sha256_file(result_path), repeatability_reviewed=True)
            done = {'status': result['status'], 'complete_updates': 2, 'actual_forward_counts': f.AA_COUNTS,
                'cleanup_verified': True, 'results_sha256': args.repeatability_results_sha256}
            done_path = root / 'completion.json'; done_path.write_text(json.dumps(done)); args.repeatability_completion_sha256 = f.base.sha256_file(done_path)
            read = isolated(f.repeatability_evidence, AA_ROOT=root)
            files = {}; evidence = read(args, files, lambda: None)
            self.assertFalse(evidence['all_compared_values_identical']); self.assertTrue(evidence['reviewed_before_launch']); self.assertEqual(len(files), 2)
            (root / 'failure.json').write_text('{}')
            with self.assertRaises(RuntimeError): read(args, {}, lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'results.json').write_text(json.dumps(result))
            done['cleanup_verified'] = False; (root / 'completion.json').write_text(json.dumps(done))
            args.repeatability_completion_sha256 = f.base.sha256_file(root / 'completion.json')
            with self.assertRaises(RuntimeError): isolated(f.repeatability_evidence, AA_ROOT=root)(args, {}, lambda: None)

    def test_metadata_budget_one_full_future_DEV90_and_terminal_no_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / f.ARM).mkdir(); budget = f.Budget(output)
            fake_pair = SimpleNamespace(available_bytes=lambda p: 20 * 1024**3, InsufficientDisk=f.pair.InsufficientDisk)
            require = isolated(f.Budget.require, pair=fake_pair)
            self.assertEqual(require(budget)['required_bytes'], 4849283072)
            self.assertEqual(set(budget.pending), {'full', 'later_DEV90'}); self.assertEqual(f.META_LIMIT, 32 * 1024**2)
            (output / 'kept.json').write_text('x' * 100)
            self.assertEqual(require(budget)['metadata_bytes'], 100)
            budget.require = Mock(return_value={}); budget.publish_success(output / f.ARM / 'completion.json', {'done': True})
            self.assertEqual(budget.require.call_count, 1)
            with self.assertRaises(RuntimeError): budget.publish_success(output / f.ARM / 'completion.json', {})

    def test_nonfinite_failure_evidence_is_explicit_and_strict_JSON_serializable(self):
        value = {'loss': float('nan'), 'other': [float('inf'), -float('inf'), .2], 'reason': 'nonfinite'}
        converted = f.failure_record(value)
        self.assertEqual(converted['loss'], {'nonfinite_float': 'nan'})
        self.assertEqual(converted['other'], [{'nonfinite_float': 'inf'}, {'nonfinite_float': '-inf'}, .2])
        json.dumps(converted, allow_nan=False)
        self.assertNotEqual(value['loss'], value['loss']); self.assertEqual(f.failure_record(raw()), raw())
        state, s, D, a, b = toy()
        def call(): update(D, b); update(s, a); return dict(raw(), student_total=float('nan'))
        with self.assertRaises(RuntimeError): f.complete_update(state, 3928, a, b, call)
        self.assertEqual(state['completed_global_step'], 3928)
        self.assertIn('unpublished_returned_losses', state); json.dumps(f.failure_record(state), allow_nan=False)

    def test_current_source_closure_does_not_require_archived_parent_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'active.py'; source.write_text('x=1')
            sys_copy = SimpleNamespace(modules={'active': SimpleNamespace(__file__=str(source))})
            select = isolated(f.d.imported_source_files, ROOT=root, sys=sys_copy)
            pins = {str(source): 'a' * 64, str(root / 'retired.pt'): 'b' * 64}
            self.assertEqual(select(pins, {}), {str(source): 'a' * 64})
            with self.assertRaises(RuntimeError): select({}, {})

    def test_static_no_old_replay_gate_oldsave_globalpatch_metrics_or_deletion(self):
        source = Path(f.__file__).read_text(); tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = [n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, 'attr', None) for n in calls]
        for name in ('replay_check', 'replay_monitor_check', 'same_tree', 'unlink', 'rmdir', 'rmtree', 'remove', 'infer_candidate', 'evaluate_metrics',
                     'set_num_threads', 'use_deterministic_algorithms', 'backward'):
            self.assertNotIn(name, names)
        self.assertEqual(names.count('train_batch_mc2'), 1)
        for call in calls:
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                self.assertNotIn((call.func.value.id, call.func.attr), (('prior', 'main'), ('prior', 'save_endpoint'), ('d', 'main')))
        self.assertNotIn("state['replay_verified_steps']", source)
        self.assertIn('self.check(); original(module, arguments)', source)
        self.assertIn('check(); return _original(*args, **kwargs)', source)
        self.assertIn("'phase': 'terminal_publication'", source)
        self.assertIn('original_error is None and not errors and completion is not None', source)
        self.assertEqual(f.EVALUATION['metrics_only'], ['cer', 'dnsmos', 'utmos'])
        self.assertIn("'DNSMOS_P808_strictly_greater_than': 3.77", source)
        self.assertIn("'SVA_gate': False, 'OVRL_gate': False", source)


if __name__ == '__main__': unittest.main()
