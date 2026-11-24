"""租用CPU的小模型/元数据测试；不调用真实训练main、不读取真实checkpoint。"""

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
    raise RuntimeError('Run only on rental CPU with CUDA hidden')
import torch
import run_acdd_mc2_continuation360_from3927_v1 as f
import evaluate_development_candidates as evaluator


def isolated(function, **replacements):
    namespace = dict(function.__globals__); namespace.update(replacements)
    return types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def raw(fm=3.):
    return {'student_total': 2. + 2 * fm + 4.5, 'student_adv': 2., 'fm_raw': fm, 'fm_weighted': 2 * fm,
            'dist_raw': .1, 'dist_weighted': 4.5, 'discriminator': 4.}


def monitor(step, mel=.19, rms=1.):
    return {'global_step': step, 'fixed_train': {'mel': .18},
            'development': {'mel_l1': mel, 'prediction_rms': rms}, 'uniform64': {'GT_l1': .191, 'identity': 'fixed'}}


def update(model, optimizer):
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)


def toy(step=3927):
    student, discriminator = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5, betas=(.5, .9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .9))
    for model, opt in ((student, optimizer), (discriminator, d_optimizer)):
        update(model, opt)
        for value in opt.state.values(): value['step'].fill_(step)
    state = f.new_state('toy')
    state.update(restore_completed=True, completed_global_step=step, new_updates=step - f.START,
        actual_updates_this_process=step - f.START, replay_updates=min(120, step - f.START),
        novel_updates=max(0, step - f.REPLAY_STOP), replay_verified_steps=min(120, step - f.START),
        replay_monitor_verified_steps=[3927, 3987, 4047] if step >= 4047 else [3927],
        last_monitor=monitor(step), reason='maximum360_updates_completed')
    return state, student, discriminator, optimizer, d_optimizer


def replay_row(step=3928):
    return dict(raw(), global_step=step, indices=[1, 2], crop_seed=101234 + step, noise_seed=201234 + step,
        input_identity={'batch': {'sha256': 'a' * 64}}, actual_optimizer_cursors={'S': {'uniform_step': step}},
        actual_forward_counts=dict(f.TRAIN_COUNTS), mc={'source_noise': 'x', 'score_noise1': 'n1', 'score_noise2': 'n2',
            't1_values': [1, 1000], 't2_values': [22, 333], 'private_seed': 74000000 + step,
            'dist1': .1, 'dist2': .12, 'first': {'teacher_target': {'sha256': 'b' * 64}, 'weights': [.1, .2]},
            'second': {'teacher_target': {'sha256': 'c' * 64}, 'per_utterance_l1': [.2, .3]}})


class Continuation360Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def cli(self):
        return ['--run-approved'] + [item for name in ('driver', 'tests', 'helper', 'helper-tests', 'protocol')
                                      for item in ('--' + name + '-sha256', 'a' * 64)]

    def test_explicit_approval_and_all_five_full_SHA_values(self):
        self.assertTrue(f.arguments(self.cli()).run_approved)
        with self.assertRaises(RuntimeError): f.arguments(self.cli()[1:])
        for index in (2, 4, 6, 8, 10):
            for bad in ('a' * 63, 'g' * 64, 'A' * 64):
                args = self.cli(); args[index] = bad
                with self.subTest(index=index), self.assertRaises(RuntimeError): f.arguments(args)

    def fixture(self):
        parts = toy(); _, s, D, a, b = parts
        config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        payload = {'global_step': 3927, 'epoch': 0, 'resumable': True, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
            'run_contract_sha256': f.d.PARENT_CANONICAL, 'effective_optimizer_lrs': dict(f.LRS),
            'configuration': config.as_dict(), 'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar',
            'stop_reason': 'maximum480_updates_completed', 'execution_error': None, 'development_monitor_global_step': 3927,
            'development': monitor(3927)['development'], 'student': copy.deepcopy(s.state_dict()),
            'discriminator': copy.deepcopy(D.state_dict()), 'student_optimizer': copy.deepcopy(a.state_dict()),
            'discriminator_optimizer': copy.deepcopy(b.state_dict()), 'rng_state': f.training.capture_rng_state()}
        restore = isolated(f.d.restore_parent, PARENT_RNG_SHA=f.probe.tree_sha(payload['rng_state']))
        shim = SimpleNamespace(**dict(f.d.__dict__, restore_parent=restore))
        return parts, payload, config, {'configuration': config.as_dict()}, {'last_monitor': monitor(3927)}, shim

    def test_complete_parent_restore_all_models_dualAdam_RNG_unchanged_and_no_alias(self):
        parts, payload, config, contract, receipt, shim = self.fixture()
        _, s, D, a, b = parts; before = f.previous.content_state(s, D, a, b); saved = f.probe.tree_sha(payload)
        update(s, a); update(D, b); torch.rand(3)
        with patch.object(f, 'd', shim): result = f.restore_parent(payload, s, D, a, b, config, contract, receipt, torch.device('cpu'))
        self.assertEqual(f.previous.content_state(s, D, a, b), before)
        self.assertEqual(result['parent_restore'], before); self.assertFalse(result['configuration_changed'])
        self.assertEqual(f.probe.tree_sha(payload), saved)
        self.assertEqual(f.probe.tree_sha(f.training.capture_rng_state()), f.probe.tree_sha(payload['rng_state']))
        for role, opt in (('student', a), ('discriminator', b)):
            for index, value in opt.state_dict()['state'].items():
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    self.assertNotEqual(value[key].data_ptr(), payload[role + '_optimizer']['state'][index][key].data_ptr())

    def test_parent_wrong_step_config_half_Adam_or_RNG_rejected(self):
        for case in ('step', 'config', 'lr', 'half', 'rng'):
            parts, payload, config, contract, receipt, shim = self.fixture()
            if case == 'step': payload['global_step'] = 4047
            if case == 'config': payload['configuration']['lambda_fm'] = 1.
            if case == 'lr': payload['effective_optimizer_lrs']['discriminator'] = 1e-4
            if case == 'half': next(iter(payload['student_optimizer']['state'].values()))['step'].fill_(3926)
            if case == 'rng': payload['rng_state']['torch'][0] ^= 1
            with patch.object(f, 'd', shim), self.subTest(case=case), self.assertRaises(RuntimeError):
                f.restore_parent(payload, *parts[1:], config, contract, receipt, torch.device('cpu'))

    def test_actual_absolute_plan_all360_and_original120_prefix(self):
        development = {'retained_train_indices': list(range(39335)), 'excluded_train_indices': list(range(39335, 39387))}
        actual = f.future_plan(development); prior = f.d.future_plan(development)
        self.assertEqual(actual[:120], prior); self.assertEqual(len(actual), 360)
        self.assertEqual([r['global_step'] for r in actual], list(range(3928, 4288)))
        self.assertEqual((actual[0]['crop_seed'], actual[0]['noise_seed']), (105162, 205162))
        self.assertEqual((actual[-1]['crop_seed'], actual[-1]['noise_seed']), (105521, 205521))
        development['excluded_train_indices'] = [actual[155]['indices'][0]]
        with self.assertRaises(RuntimeError): f.future_plan(development)

    def test_every_replay_step_allraw_teacher_targets_noise_and_cursor_checked(self):
        for step in (3928, 3988, 4047):
            control = replay_row(step); actual = copy.deepcopy(control); f.replay_check(actual, control)
            actual['student_total'] += 1e-7; f.replay_check(actual, control)
            for key in f.pair.LOSS_KEYS:
                bad = copy.deepcopy(control); bad[key] += .01
                with self.subTest(step=step, key=key), self.assertRaises(RuntimeError): f.replay_check(bad, control)
            for key in ('source_noise', 'score_noise1', 'score_noise2', 'private_seed'):
                bad = copy.deepcopy(control); bad['mc'][key] = 'changed'
                with self.assertRaises(RuntimeError): f.replay_check(bad, control)
            bad = copy.deepcopy(control); bad['mc']['second']['teacher_target']['sha256'] = 'd' * 64
            with self.assertRaises(RuntimeError): f.replay_check(bad, control)
            bad = copy.deepcopy(control); bad['actual_optimizer_cursors']['S']['uniform_step'] -= 1
            with self.assertRaises(RuntimeError): f.replay_check(bad, control)
        with self.assertRaises(RuntimeError): f.replay_check(replay_row(4048), replay_row(4048))

    def test_monitor_replay_original_numeric_tolerance_and_exact_identity(self):
        for step in (3927, 3987, 4047):
            expected = monitor(step); actual = copy.deepcopy(expected)
            actual['uniform64']['GT_l1'] += 1e-8; f.replay_monitor_check(actual, expected)
            actual['uniform64']['GT_l1'] += .01
            with self.assertRaises(RuntimeError): f.replay_monitor_check(actual, expected)
            actual = copy.deepcopy(expected); actual['uniform64']['identity'] = 'changed'
            with self.assertRaises(RuntimeError): f.replay_monitor_check(actual, expected)
        with self.assertRaises(RuntimeError): f.replay_monitor_check(monitor(4047), monitor(3987))

    def simulate(self, mel=None, fail_after=None):
        state = toy()[0]; written, called = [], []
        future = [{'global_step': step} for step in range(3928, 4288)]
        controls = {step: monitor(step) for step in (3927, 3987, 4047)}
        def step(planned):
            number = planned['global_step']; count = number - f.START
            state.update(completed_global_step=number, new_updates=count, actual_updates_this_process=count,
                replay_updates=min(120, count), novel_updates=max(0, count - 120), replay_verified_steps=min(120, count))
            return dict(raw(3. + count * .01), global_step=number, mc={'dist1': .1, 'dist2': .1})
        def evaluate(number): called.append(number); return monitor(number, (mel or {}).get(number, .19))
        def publish(name, value, append=True): written.append((name, copy.deepcopy(value)))
        def check():
            if fail_after is not None and state['new_updates'] >= fail_after: raise RuntimeError('bounded time')
        return state, future, step, evaluate, publish, check, monitor(3927), controls, written, called

    def test_360_records_3_full_windows_7_monitors_no_shortloss_gate(self):
        parts = self.simulate(); result = f.run_updates(*parts[:8])
        self.assertEqual(len(result['rows']), 360); self.assertEqual(len(result['windows']), 3)
        self.assertEqual(parts[9], [3987, 4047, 4107, 4167, 4227, 4287]); self.assertEqual(len(result['monitors']), 7)
        self.assertEqual(parts[0]['replay_monitor_verified_steps'], [3927, 3987, 4047])
        self.assertEqual((parts[0]['replay_updates'], parts[0]['novel_updates']), (120, 240))
        self.assertEqual(result['reason'], 'maximum360_updates_completed')
        trend = f.novel_trend(result['rows'])
        self.assertGreater(trend['second_new_minus_first_new']['student_total'], 0)
        self.assertEqual(trend['replay_window']['phase'], 'historical_reconstruction_replay')
        self.assertFalse(trend['used_as_stop_gate'])
        self.assertFalse(f.novel_trend(result['rows'][:240])['available'])

    def test_original_distinct_guards_at_new_endpoints_and_replay_failure_kept(self):
        parts = self.simulate({4107: .203, 4167: .204}); result = f.run_updates(*parts[:8])
        self.assertEqual(parts[0]['completed_global_step'], 4167)
        self.assertEqual(result['reason'], 'two_consecutive_DEV32_above3327_by5percent')
        self.assertEqual(len(result['windows']), 2)
        parts = self.simulate({4107: .22}); result = f.run_updates(*parts[:8])
        self.assertEqual(parts[0]['completed_global_step'], 4107); self.assertEqual(result['reason'], 'original_S0_extreme_guard')
        parts = self.simulate({3987: .20})
        with self.assertRaisesRegex(RuntimeError, 'Replay numeric mismatch'): f.run_updates(*parts[:8])
        self.assertEqual(len([r for r in parts[8] if r[0] == 'step_losses.jsonl']), 60)
        self.assertEqual(len([r for r in parts[8] if r[0] == 'monitor.jsonl']), 1)
        parts = self.simulate(fail_after=123)
        with self.assertRaisesRegex(RuntimeError, 'bounded time'): f.run_updates(*parts[:8])
        self.assertEqual(len([r for r in parts[8] if r[0] == 'step_losses.jsonl']), 123)

    def test_actual_Adam_complete_novel_boundary_and_half_or_poststep_failure(self):
        state, s, D, a, b = toy(4047)
        def complete(): update(D, b); update(s, a); return raw()
        f.complete_update(state, 4048, a, b, complete)
        self.assertEqual(state['completed_global_step'], 4048)
        self.assertEqual((state['replay_updates'], state['novel_updates'], state['actual_updates_this_process']), (120, 1, 121))
        self.assertFalse(state['partial_batch_may_have_updated_D'])
        for fail in ('half', 'after_both'):
            state, s, D, a, b = toy(4047)
            def failed():
                update(D, b)
                if fail == 'after_both': update(s, a)
                raise RuntimeError(fail)
            with self.assertRaisesRegex(RuntimeError, fail): f.complete_update(state, 4048, a, b, failed)
            if fail == 'half':
                self.assertEqual(state['completed_global_step'], 4047); self.assertTrue(state['partial_batch_may_have_updated_D'])
            else:
                self.assertEqual(state['completed_global_step'], 4048); self.assertTrue(state['exception_after_both_adam_steps'])

    def test_budget_32MiB_single_full_future_eval_and_terminal_no_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / f.ARM).mkdir(); budget = f.Budget(output)
            fake = SimpleNamespace(available_bytes=lambda p: 20 * 1024**3, InsufficientDisk=f.pair.InsufficientDisk)
            require = isolated(f.Budget.require, pair=fake)
            self.assertEqual(require(budget)['required_bytes'], f.INITIAL_REQUIRED_BYTES)
            self.assertEqual(f.INITIAL_REQUIRED_BYTES, 4849283072)
            (output / 'kept.json').write_text('x' * 1024)
            self.assertEqual(require(budget)['metadata_bytes'], 1024)
            self.assertEqual(require(budget)['required_bytes'], f.INITIAL_REQUIRED_BYTES - 1024)
            with self.assertRaises(f.pair.InsufficientDisk): require(budget, f.META_LIMIT)
            budget.require = Mock(return_value={}); path = output / f.ARM / 'completion.json'
            budget.publish_success(path, {'status': 'completed360'})
            self.assertEqual(budget.require.call_count, 1)
            with self.assertRaises(RuntimeError): budget.publish_success(path, {})

    def test_real_toy_full_save_config_Adam_RNG_and_evaluator_compatibility(self):
        state, s, D, a, b = toy(4287); config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        contract = {'configuration': config.as_dict(), 'kind': 'toy360'}
        state['contract_sha256'] = f.training.canonical_json_sha256(contract)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); stage = output / f.ARM; stage.mkdir(); budget = f.Budget(output)
            budget.require = Mock(return_value={})
            f.save_endpoint(state, config, s, D, a, b, budget, SimpleNamespace(active=lambda: None))
            path = stage / 'final_full.pt'; payload = torch.load(path, map_location='cpu')
            self.assertEqual(payload['global_step'], 4287); self.assertEqual(payload['configuration']['lambda_fm'], 2.)
            self.assertEqual(payload['effective_optimizer_lrs'], f.LRS)
            self.assertEqual((payload['replay_updates'], payload['novel_updates']), (120, 240))
            self.assertTrue(payload['resumable']); self.assertEqual(path.stat().st_mode & 0o777, 0o444)
            self.assertEqual([p.name for p in stage.glob('*.pt')], ['final_full.pt']); self.assertNotIn('full', budget.pending)
            for key in ('student', 'discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state'): self.assertIn(key, payload)
            self.assertEqual(evaluator.validate_payload(payload, {'epoch': 0, 'global_step': 4287, 'state_key': 'student'}, contract), state['contract_sha256'])

    def test_never_save_zero_replay_partial_failed_unverified_wrongconfig_endpoints(self):
        for case in ('zero', 'replay60', 'replay120', 'half', 'failed', 'poststep_error', 'unverified', 'wrongconfig', 'wrongmonitor', 'wrongcounts'):
            step = {'zero': 3927, 'replay60': 3987, 'replay120': 4047}.get(case, 4287)
            state, s, D, a, b = toy(step); config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
            if case == 'half': state['partial_batch_may_have_updated_D'] = True
            if case == 'failed': state['execution_error'] = 'error'
            if case == 'poststep_error': state['exception_after_both_adam_steps'] = True
            if case == 'unverified': state['replay_verified_steps'] = 119
            if case == 'wrongconfig': config = f.base.replace(config, lambda_fm=1.)
            if case == 'wrongmonitor': state['last_monitor'] = monitor(4227)
            if case == 'wrongcounts': state['novel_updates'] = 360
            with self.subTest(case=case), self.assertRaises(RuntimeError): f.save_endpoint(state, config, s, D, a, b, None, None)

    def test_protective_stop_saves_only_its_actual_new_complete_endpoint(self):
        state, s, D, a, b = toy(4107); config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        with self.assertRaisesRegex(RuntimeError, 'Endpoint reason'):
            f.save_endpoint(state, config, s, D, a, b, None, None)
        state['reason'] = 'original_S0_extreme_guard'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / f.ARM).mkdir(); budget = f.Budget(output); budget.require = Mock(return_value={})
            f.save_endpoint(state, config, s, D, a, b, budget, SimpleNamespace(active=lambda: None))
            payload = torch.load(output / f.ARM / 'final_full.pt', map_location='cpu')
            self.assertEqual(payload['global_step'], 4107); self.assertEqual(payload['novel_updates'], 60)
            self.assertEqual(payload['stop_reason'], 'original_S0_extreme_guard')

    def test_forward_limits_and_cleanup_errors_do_not_skip_other_resources(self):
        self.assertEqual(f.FORWARD_LIMITS, {k: 360 * f.TRAIN_COUNTS[k] + 7 * f.MONITOR_COUNTS[k] for k in f.TRAIN_COUNTS})
        ledger = f.ForwardLedger(); hook = ledger.before('student'); ledger.counts['student'] = 747
        hook(None, None)
        with self.assertRaises(RuntimeError): hook(None, None)
        calls = []
        def action(name, fail=False):
            def call():
                calls.append(name)
                if fail: raise RuntimeError(name)
            return call
        errors = f.old.cleanup_resources(SimpleNamespace(close=action('hooks', True)), SimpleNamespace(close=action('clock')),
            SimpleNamespace(flush=action('flush', True), close=action('log_close')), [SimpleNamespace(close=action('lock'))])
        self.assertEqual(calls, ['hooks', 'clock', 'flush', 'log_close', 'lock']); self.assertEqual(len(errors), 2)

    def test_import_closure_unknown_source_rejected_not_parent_assets_recursed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'current.py'; source.write_text('value = 1')
            history = str(root / 'archived_full.pt')
            fake_sys = SimpleNamespace(modules={'current': SimpleNamespace(__file__=str(source))})
            select = isolated(f.d.imported_source_files, ROOT=root, sys=fake_sys)
            pins = {str(source): 'a' * 64, history: 'b' * 64}
            self.assertEqual(select(pins, {}), {str(source): 'a' * 64})
            with self.assertRaisesRegex(RuntimeError, 'lacks reviewed SHA'): select({history: 'b' * 64}, {})

    def test_retained_control_JSON_layout_windows_and_no_archived_asset_read(self):
        # 使用旧driver真实字段布局；只构造小JSON，不创建旧S、WAV或任何模型文件。
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); folder = output / f.d.ARMS[0]; folder.mkdir()
            config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0).as_dict()
            rows = [replay_row(step) for step in range(3928, 4048)]
            future = [{k: r[k] for k in ('global_step', 'indices', 'crop_seed', 'noise_seed')} for r in rows]
            future += [{'global_step': step} for step in range(4048, 4288)]
            windows = [f.d.window60(rows[:60]), f.d.window60(rows[60:])]
            monitors = [monitor(step) for step in (3927, 3987, 4047)]
            joint = {'parent_checkpoint_sha256': f.d.PARENT_SHA, 'configuration': config,
                'future_batches': future[:120], 'effective_optimizer_lrs_by_arm': {f.d.ARMS[0]: f.LRS}}
            canonical = f.training.canonical_json_sha256(joint)
            state = {'completed_global_step': 4047, 'partial_batch_may_have_updated_D': False,
                'restore_completed': True, 'new_updates': 120, 'loss_windows': windows, 'last_monitor': monitors[-1]}
            completion = {'status': 'completed120', 'cleanup_verified': True, 'contract_sha256': canonical,
                'state': state, 'actual_forward_counts': f.d.FORWARD_LIMITS}
            joint_path, done_path = output / 'run_contract.json', folder / 'completion.json'
            joint_path.write_text(json.dumps(joint)); done_path.write_text(json.dumps(completion))
            archive = {'joint_contract_file_sha256': f.base.sha256_file(joint_path),
                'control_completion_file_sha256': f.base.sha256_file(done_path)}
            archive_path = output / f.d.ARCHIVE_NAME; archive_path.write_text(json.dumps(archive))
            for name, values in (('step_losses.jsonl', rows), ('loss_windows.jsonl', windows), ('monitor.jsonl', monitors)):
                (folder / name).write_text('\n'.join(json.dumps(r) for r in values))
            shim = SimpleNamespace(**dict(f.d.__dict__, OUTPUT=output))
            read = isolated(f.control_evidence, d=shim, CONTROL_CANONICAL=canonical,
                CONTROL_CONTRACT_SHA=f.base.sha256_file(joint_path), CONTROL_COMPLETION_SHA=f.base.sha256_file(done_path),
                CONTROL_ARCHIVE_SHA=f.base.sha256_file(archive_path))
            files = {}; actual, actual_monitors, proof = read({'configuration': config}, future, files, lambda: None)
            self.assertEqual(len(actual), 120); self.assertEqual(set(actual_monitors), {3927, 3987, 4047})
            self.assertEqual(len(files), 6); self.assertTrue(all(Path(p).suffix in ('.json', '.jsonl') for p in files))
            self.assertTrue(proof['no_archived_model_or_wav_is_a_runtime_dependency'])
            rows[85]['fm_raw'] += .1
            (folder / 'step_losses.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(RuntimeError): read({'configuration': config}, future, {}, lambda: None)

    def test_static_no_archived_model_score_delete_or_old_range_mutation(self):
        source = Path(f.__file__).read_text(); tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call): continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None)
            self.assertNotIn(name, ('unlink', 'rmdir', 'rmtree', 'remove', 'infer_candidate', 'evaluate_metrics', 'candidate_gate'))
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                self.assertNotIn((node.func.value.id, node.func.attr), (('d', 'main'), ('d', 'save_endpoint'), ('d', 'complete_update')))
        self.assertIn("'future_metrics_only': ['cer', 'dnsmos', 'utmos']", source)
        self.assertIn("'DNSMOS_P808_strictly_greater_than': 3.77", source)
        self.assertNotIn('c54e1e4f06d5ae5d88f2170ccec63788d3a17170181efd28e5474a680b7b2c95', source)
        self.assertIn('old.StopClock(1500, started)', source)
        self.assertIn('d.imported_source_files(files, files)', source)
        self.assertIn('original_error is None and not errors and completion is not None', source)


if __name__ == '__main__': unittest.main()
