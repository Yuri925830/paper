"""仅租用CPU的小测试：使用toy权重，不读真实full、不调用main、不初始化CUDA。"""
import ast
import copy
import inspect
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only the rented CPU with CUDA hidden may run these tests')
import torch
import run_acdd_mc2_perceptual240_from4287_v1 as f
import evaluate_development_candidates as evaluator


def isolated(function, **replacements):
    # 单独复制测试函数的globals，冻结生产模块的函数和常量完全不动。
    function = inspect.unwrap(function)
    namespace = dict(function.__globals__); namespace.update(replacements)
    result = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def raw(fm=3., auxiliary=.5, weight=.25):
    total = 2. + 2 * fm + 4.5
    return {'student_total': total, 'student_adv': 2., 'fm_raw': fm, 'fm_weighted': 2 * fm,
        'dist_raw': .1, 'dist_weighted': 4.5, 'discriminator': 4.,
        'conversion_perceptual_raw': auxiliary, 'conversion_perceptual_weighted': auxiliary * weight,
        'conversion_perceptual_weight': weight, 'student_joint_total': total + auxiliary * weight}


def monitor(step, mel=.19, rms=1.):
    return {'global_step': step, 'development': {'mel_l1': mel, 'prediction_rms': rms},
        'fixed_train': {'student_total': 6.}, 'uniform64': {'GT_l1': .191}}


def update(model, optimizer):
    for parameter in model.parameters(): parameter.grad = torch.full_like(parameter, .25)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)


def toy(step=4287):
    torch.manual_seed(713)
    student, discriminator = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5, betas=(.5, .9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .9))
    for model, opt in ((student, optimizer), (discriminator, d_optimizer)):
        update(model, opt)
        for item in opt.state.values(): item['step'].fill_(step)
    count = step - f.START; state = f.new_state('a' * 64)
    state.update(restore_completed=True, completed_global_step=step, actual_updates_this_process=count,
        new_updates=count, complete_loss_records=count, last_monitor=monitor(step),
        monitor_global_steps=list(range(f.START, step + 1, 60)), reason='maximum240_updates_completed',
        auxiliary_configuration={'anchor_weight': .25, 'method': 'toy_provenance'})
    return state, student, discriminator, optimizer, d_optimizer


class Perceptual240Tests(unittest.TestCase):
    def cli(self):
        names = ('driver', 'tests', 'helper', 'helper-tests', 'anchor-helper', 'loss-helper', 'loss-helper-tests',
                 'protocol', 'wavlm', 'wavlm-receipt', 'gradient-results', 'gradient-completion')
        return ['--run-approved', '--anchor-weight', '.25'] + [x for name in names for x in ('--' + name + '-sha256', 'a' * 64)]

    def test_explicit_run_calibrated_weight_and_every_full_SHA(self):
        self.assertEqual(f.arguments(self.cli()).anchor_weight, .25)
        with self.assertRaises(RuntimeError): f.arguments(self.cli()[1:])
        for bad in ('nan', 'inf', '0', '-1'):
            args = self.cli(); args[2] = bad
            with self.subTest(weight=bad), self.assertRaises(RuntimeError): f.arguments(args)
        for index in range(4, len(self.cli()), 2):
            args = self.cli(); args[index] = 'g' * 64
            with self.subTest(index=index), self.assertRaises(RuntimeError): f.arguments(args)

    def parent_fixture(self):
        parts = toy(); _, s, D, a, b = parts
        config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        contract = {'configuration': config.as_dict(), 'parent_checkpoint_sha256': 'b' * 64, 'parent_contract_sha256': 'c' * 64}
        payload = {'kind': 'MC2_FM2_independent360_full', 'arm': f.PARENT_ARM, 'global_step': 4287, 'epoch': 0,
            'resumable': True, 'run_contract_sha256': f.PARENT_CANONICAL, 'effective_optimizer_lrs': dict(f.LRS),
            'mc_samples': 2, 'mc_private_seed_base': 74000000, 'actual_updates_this_process': 360,
            'historical_state_reconstruction_claimed': False, 'configuration': config.as_dict(),
            'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'stop_reason': 'maximum360_updates_completed',
            'execution_error': None, 'development_monitor_global_step': 4287, 'development': monitor(4287)['development'],
            'lineage': {'parent_checkpoint_sha256': 'b' * 64, 'parent_contract_sha256': 'c' * 64, 'start_after_global_step': 3927},
            'student': copy.deepcopy(s.state_dict()), 'discriminator': copy.deepcopy(D.state_dict()),
            'student_optimizer': copy.deepcopy(a.state_dict()), 'discriminator_optimizer': copy.deepcopy(b.state_dict()),
            'rng_state': f.training.capture_rng_state()}
        restore = isolated(f.restore_parent, PARENT_RNG_SHA=f.probe.tree_sha(payload['rng_state']))
        return parts, config, contract, payload, restore

    def test_real_toy_full4287_restores_both_models_twoAdam_RNG_without_alias(self):
        parts, config, contract, payload, restore = self.parent_fixture(); _, s, D, a, b = parts
        before, payload_sha = f.previous.content_state(s, D, a, b), f.probe.tree_sha(payload)
        update(s, a); update(D, b); torch.rand(4)
        evidence = restore(payload, s, D, a, b, config, contract, {'last_monitor': monitor(4287)}, torch.device('cpu'))
        self.assertEqual(f.previous.content_state(s, D, a, b), before); self.assertEqual(evidence['parent_restore'], before)
        self.assertTrue(evidence['all_models_optimizers_rng_restored_without_change']); self.assertFalse(evidence['configuration_changed'])
        self.assertEqual(f.probe.tree_sha(payload), payload_sha)
        self.assertEqual(f.probe.tree_sha(f.training.capture_rng_state()), f.probe.tree_sha(payload['rng_state']))
        for name, opt in (('student', a), ('discriminator', b)):
            for index, item in opt.state_dict()['state'].items():
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    self.assertNotEqual(item[key].data_ptr(), payload[name + '_optimizer']['state'][index][key].data_ptr())

    def test_parent_wrong_kind_step_config_lr_momenta_rng_or_half_cursor_rejected(self):
        for case in ('kind', 'step', 'config', 'lr', 'moment', 'rng', 'cursor', 'failure'):
            parts, config, contract, payload, restore = self.parent_fixture()
            if case == 'kind': payload['kind'] = 'old3927'
            if case == 'step': payload['global_step'] = 3927
            if case == 'config': payload['configuration']['lambda_fm'] = 1.
            if case == 'lr': payload['effective_optimizer_lrs']['discriminator'] = 1e-4
            if case == 'moment': next(iter(payload['student_optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
            if case == 'rng': payload['rng_state']['torch'] = torch.zeros_like(payload['rng_state']['torch'])
            if case == 'cursor': next(iter(payload['discriminator_optimizer']['state'].values()))['step'].fill_(4286)
            if case == 'failure': payload['execution_error'] = 'old failure'
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                restore(payload, *parts[1:], config, contract, {'last_monitor': monitor(4287)}, torch.device('cpu'))

    def test_parent_metadata_real_complete360_schema_cannot_accept_half_or_missing_monitor(self):
        contract = {'baseline3327_development': f.d.cont.BASELINE3327, 'original_step0_guard_anchor': f.d.cont.S0_ANCHOR}
        canonical = f.training.canonical_json_sha256(contract)
        state = {'completed_global_step': 4287, 'last_full_saved_global_step': 4287, 'new_updates': 360,
            'actual_updates_this_process': 360, 'partial_batch_may_have_updated_D': False, 'restore_completed': True,
            'complete_loss_records': 360, 'monitor_global_steps': list(range(3927, 4288, 60)),
            'last_monitor': monitor(4287), 'effective_optimizer_lrs': dict(f.LRS),
            'full_snapshot': {'path': str(f.PARENT), 'sha256': f.PARENT_SHA, 'global_step': 4287,
                'committed': True, 'resumable': True, 'frozen_readonly': True}}
        done = {'status': 'completed360', 'cleanup_verified': True, 'contract_sha256': canonical,
            'state': state, 'actual_forward_counts': dict(f.prior.FORWARD_LIMITS)}
        validate = isolated(f.validate_parent_metadata, PARENT_CANONICAL=canonical)
        self.assertIs(validate(contract, done), state)
        for key, value in (('partial_batch_may_have_updated_D', True), ('new_updates', 359),
                           ('monitor_global_steps', [3927, 4287]), ('exception_after_both_adam_steps', True)):
            changed = copy.deepcopy(done); changed['state'][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): validate(contract, changed)

    def test_true_return_cursors_keep_complete_half_and_poststep_failure_distinct(self):
        for mode in ('normal', 'half', 'afterboth'):
            state, s, D, a, b = toy()
            def call():
                update(D, b)
                if mode == 'half': raise RuntimeError('between D/S')
                update(s, a)
                if mode == 'afterboth': raise RuntimeError('after both')
                return raw()
            if mode == 'normal': f.complete_update(state, 4288, a, b, call)
            else:
                with self.assertRaises(RuntimeError): f.complete_update(state, 4288, a, b, call)
            self.assertEqual(state['completed_global_step'], 4287 if mode == 'half' else 4288)
            self.assertEqual(state['partial_batch_may_have_updated_D'], mode == 'half')
            if mode == 'afterboth': self.assertTrue(state['exception_after_both_adam_steps'])
            self.assertNotIn('step', a.__dict__); self.assertNotIn('step', b.__dict__)

    def test_absolute_future240_plan_52_exclusions_seeds_no_replay(self):
        development = {'retained_train_indices': list(range(39335)), 'excluded_train_indices': list(range(39335, 39387))}
        rows = f.future_plan(development); planner = f.resume.GlobalBatchPlan(development['retained_train_indices'])
        self.assertEqual(len(rows), 240)
        for row in rows:
            step = row['global_step']; self.assertEqual(row['indices'], planner.for_step(step))
            self.assertEqual((row['crop_seed'], row['noise_seed'], row['shuffle_seed'], row['reverse_seed']),
                             (101234 + step, 201234 + step, 90000000 + step, 91000000 + step))
        self.assertEqual((rows[0]['global_step'], rows[-1]['global_step']), (4288, 4527))
        development['excluded_train_indices'] = [rows[0]['indices'][0]]
        with self.assertRaises(RuntimeError): f.future_plan(development)

    def test_original_and_auxiliary_totals_finite_and_separate(self):
        original = raw(); f.finite_losses(original)
        self.assertEqual(original['student_total'], 12.5); self.assertEqual(original['student_joint_total'], 12.625)
        for key in f.AUX_KEYS:
            with self.subTest(key=key), self.assertRaises(FloatingPointError): f.finite_losses(dict(original, **{key: float('nan')}))
        with self.assertRaises(RuntimeError): f.finite_losses(dict(original, student_joint_total=12.5))

    def simulate(self, values=None, initial_mel=.19):
        state = toy()[0]; written = []
        future = [{'global_step': step} for step in range(4288, 4528)]
        def step_call(row):
            step = row['global_step']; count = step - f.START
            state.update(completed_global_step=step, new_updates=count, actual_updates_this_process=count)
            return dict(raw(3. + count * .01), global_step=step, mc={'dist1': .1, 'dist2': .1})
        def publish(name, value, append=True): written.append((name, copy.deepcopy(value)))
        result = f.run_updates(state, future, step_call, lambda step: monitor(step, (values or {}).get(step, .19)),
                               publish, lambda: None, monitor(4287, initial_mel))
        return state, written, result

    def test_all240_two_complete_windows_five_monitors_rising_scalar_not_new_gate(self):
        state, written, result = self.simulate()
        self.assertEqual(len(result['rows']), 240); self.assertEqual(len(result['windows']), 2); self.assertEqual(len(result['monitors']), 5)
        self.assertEqual(result['reason'], 'maximum240_updates_completed'); self.assertEqual(state['new_updates'], 240)
        self.assertEqual([m['global_step'] for m in result['monitors']], [4287, 4347, 4407, 4467, 4527])
        trend = f.loss_trend(result['rows']); self.assertTrue(trend['available']); self.assertFalse(trend['used_as_stop_gate'])
        self.assertGreater(trend['second_minus_first_window']['student_joint_total'], 0)
        self.assertTrue(trend['original_reconstruction_total_and_new_joint_total_reported_separately'])
        self.assertFalse(f.loss_trend(result['rows'][:120])['available'])

    def test_original_guard_actual_distinct_points_and_zero_update_guard(self):
        state, written, result = self.simulate({4347: .203, 4407: .204})
        self.assertEqual(state['completed_global_step'], 4407); self.assertEqual(result['reason'], 'two_consecutive_DEV32_above3327_by5percent')
        checks = [value for name, value in written if name == 'guard_checks.jsonl']
        self.assertEqual(checks[-1]['distinct_global_steps_used'], [4347, 4407])
        state, _, result = self.simulate({4347: .22}); self.assertEqual(result['reason'], 'original_S0_extreme_guard')
        self.assertEqual(state['new_updates'], 60)
        state, _, result = self.simulate(initial_mel=.22); self.assertEqual(state['new_updates'], 0); self.assertEqual(result['rows'], [])

    def test_exact_full_save_actual_new_config_twoAdam_RNG_and_frozen_evaluator(self):
        state, s, D, a, b = toy(4527); config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        contract = {'kind': 'perceptual240_toy', 'configuration': config.as_dict(), 'auxiliary_configuration': state['auxiliary_configuration']}
        state['contract_sha256'] = f.training.canonical_json_sha256(contract)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / f.ARM).mkdir(); budget = f.Budget(output); budget.require = Mock(return_value={})
            f.save_endpoint(state, config, s, D, a, b, budget, SimpleNamespace(active=lambda: None))
            path = output / f.ARM / 'final_full.pt'; payload = torch.load(path, map_location='cpu')
            self.assertEqual(payload['kind'], 'MC2_FINALLY_inspired_perceptual240_from4287_full')
            self.assertEqual((payload['global_step'], payload['actual_updates_this_process']), (4527, 240))
            self.assertEqual(payload['configuration'], config.as_dict()); self.assertEqual(payload['auxiliary_configuration'], state['auxiliary_configuration'])
            self.assertEqual(payload['effective_optimizer_lrs'], f.LRS); self.assertTrue(payload['resumable'])
            self.assertEqual(path.stat().st_mode & 0o777, 0o444); self.assertNotIn('full', budget.pending)
            self.assertEqual([p.name for p in path.parent.glob('*.pt')], ['final_full.pt'])
            self.assertEqual(evaluator.validate_payload(payload, {'epoch': 0, 'global_step': 4527, 'state_key': 'student'}, contract), state['contract_sha256'])
            self.assertEqual(f.probe.tree_sha(payload['rng_state']), f.probe.tree_sha(f.training.capture_rng_state()))
            for key in ('student', 'discriminator', 'student_optimizer', 'discriminator_optimizer'): self.assertTrue(payload[key])

    def test_save_rejects_zero_half_technical_wrong_counts_config_or_cursor(self):
        for case in ('zero', 'half', 'error', 'afterstep', 'counts', 'monitor', 'earlynormal', 'config', 'cursor'):
            state, s, D, a, b = toy(4287 if case == 'zero' else 4347 if case == 'earlynormal' else 4527)
            config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
            if case == 'half': state['partial_batch_may_have_updated_D'] = True
            if case == 'error': state['execution_error'] = 'failed'
            if case == 'afterstep': state['exception_after_both_adam_steps'] = True
            if case == 'counts': state['actual_updates_this_process'] -= 1
            if case == 'monitor': state['monitor_global_steps'] = [4287, 4527]
            if case == 'config': config = f.base.replace(config, lambda_fm=1.)
            if case == 'cursor': next(iter(a.state.values()))['step'].fill_(4526)
            with self.subTest(case=case), self.assertRaises(RuntimeError): f.save_endpoint(state, config, s, D, a, b, None, None)

    def test_actual_protection4347_full_has_no_automatic_score_or_S_only(self):
        state, s, D, a, b = toy(4347); state['reason'] = 'original_S0_extreme_guard'
        config = f.base.replace(f.base.PAPER_STUDENT_CONFIG, num_workers=0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / f.ARM).mkdir(); budget = f.Budget(output); budget.require = Mock(return_value={})
            f.save_endpoint(state, config, s, D, a, b, budget, SimpleNamespace(active=lambda: None))
            payload = torch.load(output / f.ARM / 'final_full.pt', map_location='cpu')
            self.assertEqual((payload['global_step'], payload['actual_updates_this_process']), (4347, 60))
        self.assertEqual(f.EVALUATION['normal_endpoint_global_step'], 4527); self.assertFalse(f.EVALUATION['executed_by_this_driver'])

    def gradient_fixture(self, directory, variable=False):
        root = Path(directory); lengths = [32768] * 8
        if variable: lengths[7] = 32000
        counts = dict(f.GRADIENT_FIXED_COUNTS, wavlm=18 if variable else 4)
        args = SimpleNamespace(anchor_weight=.25, wavlm_sha256='b' * 64, loss_helper_sha256='c' * 64)
        contract = {'configuration': {'lambda_fm': 2.}, 'pairs': [{'valid_samples': x} for x in lengths], 'forward_limits': counts,
            'files': {str(f.PARENT): f.PARENT_SHA, str(f.WAVLM): args.wavlm_sha256, str(f.TOOLS / (f.LOSS_HELPER + '.py')): args.loss_helper_sha256}}
        result = {'counts': counts, 'gradient_statistics': {'norms': {'45MC2_conversion': 2., 'LMOS': 8.}},
            'calibration': {'anchor_weight': .25}, 'optimizer_updates': 0, 'checkpoint_saves': 0}
        for name, value in (('run_contract.json', contract), ('results.json', result)): (root / name).write_text(json.dumps(value))
        args.gradient_results_sha256 = f.base.sha256_file(root / 'results.json')
        done = {'status': 'completed_readonly_TRAIN8_gradient_diagnostic', 'cleanup_verified': True, 'counts': counts,
            'results_sha256': args.gradient_results_sha256, 'contract_sha256': f.base.sha256_file(root / 'run_contract.json'),
            'cleanup': dict.fromkeys(('rng_unchanged', 'model_snapshots_unchanged', 'flags_unchanged', 'restored_rng_and_flags'), True)}
        (root / 'completion.json').write_text(json.dumps(done)); args.gradient_completion_sha256 = f.base.sha256_file(root / 'completion.json')
        return args, contract, result, done, isolated(f.gradient_provenance, GRADIENT_ROOT=root)

    def test_actual_gradient_schema_exact_ratio_dynamic_counts_and_three_JSON_pins(self):
        for variable in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                args, contract, _, _, read = self.gradient_fixture(directory, variable); files = {}
                provenance = read(args, contract, files, lambda: None)
                self.assertEqual(provenance['anchor_weight'], .25); self.assertEqual(len(files), 3)
                self.assertTrue(provenance['not_GradNorm']); self.assertTrue(provenance['no_performance_claim_from_calibration'])
                args.anchor_weight = .26
                with self.assertRaises(RuntimeError): read(args, contract, {}, lambda: None)

    def test_gradient_failure_bad_cleanup_counts_or_asset_prevents_training(self):
        for case in ('failure', 'cleanup', 'counts', 'asset'):
            with tempfile.TemporaryDirectory() as directory:
                args, contract, result, done, read = self.gradient_fixture(directory); root = Path(directory)
                if case == 'failure': (root / 'failure.json').write_text('{}')
                if case == 'cleanup': done['cleanup']['rng_unchanged'] = False
                if case == 'counts': done['counts']['teacher'] = 61
                if case == 'asset': args.wavlm_sha256 = 'd' * 64
                (root / 'completion.json').write_text(json.dumps(done)); args.gradient_completion_sha256 = f.base.sha256_file(root / 'completion.json')
                with self.subTest(case=case), self.assertRaises(RuntimeError): read(args, contract, {}, lambda: None)

    def test_new_allocation_budget_exact_metadata_and_terminal_no_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / f.ARM).mkdir(); budget = f.Budget(output)
            pair = SimpleNamespace(available_bytes=lambda p: f.INITIAL_REQUIRED_BYTES, InsufficientDisk=f.pair.InsufficientDisk)
            require = isolated(f.Budget.require, pair=pair)
            self.assertEqual(f.INITIAL_REQUIRED_BYTES, 2669008128); self.assertEqual(require(budget)['required_bytes'], 2669008128)
            pair.available_bytes = lambda p: f.INITIAL_REQUIRED_BYTES - 1
            with self.assertRaises(f.pair.InsufficientDisk): require(budget)
            pair.available_bytes = lambda p: f.INITIAL_REQUIRED_BYTES
            (output / 'kept.json').write_text('x' * 100); self.assertEqual(require(budget)['metadata_bytes'], 100)
            budget.require = Mock(return_value={}); budget.publish_success(output / f.ARM / 'completion.json', {'done': True})
            self.assertEqual(budget.require.call_count, 1)
            with self.assertRaises(RuntimeError): budget.publish_success(output / f.ARM / 'completion.json', {})

    def test_own3300_timeout_and_forward_Adam_deadlines_preserve_normal_math(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory); (parent / 'exe').symlink_to('/usr/bin/timeout')
            # 二进制分隔与数字分别拼接，避免八进制转义误把3300变成控制字符。
            (parent / 'cmdline').write_bytes(b'\0'.join([b'timeout', b'--signal=TERM', b'--kill-after=90s', b'3300s']) + b'\0')
            self.assertEqual(f.watchdog(parent)['compute_seconds'], 3300)
            (parent / 'cmdline').write_bytes(b'\0'.join([b'timeout', b'--signal=TERM', b'--kill-after=90s', b'1500s']))
            with self.assertRaises(RuntimeError): f.watchdog(parent)
        _, s, D, a, b = toy(); _, s2, D2, a2, b2 = toy(); check = Mock()
        with f.deadline_steps(a, b, check): update(D, b); update(s, a)
        update(D2, b2); update(s2, a2)
        self.assertEqual(f.previous.content_state(s, D, a, b), f.previous.content_state(s2, D2, a2, b2)); self.assertEqual(check.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            with f.deadline_steps(a, b, Mock(side_effect=RuntimeError('deadline'))): update(D, b)

    def test_forward_limits_and_wavlm_hooks_count_before_any_compute(self):
        self.assertEqual(f.FORWARD_LIMITS, {'teacher': 7785, 'student': 740, 'vocoder': 1210, 'active_D_branches': 7680, 'fixed_D_branches': 80})
        ledger = f.ForwardLedger(); ledger.check = Mock(side_effect=RuntimeError('deadline'))
        with self.assertRaises(RuntimeError): ledger.before('teacher')(None, None)
        self.assertEqual(ledger.counts['teacher'], 0)
        encoder = torch.nn.Identity(); ledger.install_wavlm(encoder)
        with self.assertRaises(RuntimeError): encoder(torch.ones(1))
        self.assertEqual(ledger.wavlm_calls, 0); ledger.check = Mock(); encoder(torch.ones(1)); self.assertEqual(ledger.wavlm_calls, 1)
        ledger.wavlm_calls = 64 * 240
        with self.assertRaises(RuntimeError): encoder(torch.ones(1))
        ledger.close()

    def test_source_closure_only_actual_files_no_archived_ancestor_or_scoring_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'active.py'; source.write_text('x=1')
            select = isolated(f.d.imported_source_files, ROOT=root, sys=SimpleNamespace(modules={'active': SimpleNamespace(__file__=str(source))}))
            self.assertEqual(select({str(source): 'a' * 64, str(root / 'archived.pt'): 'b' * 64}, {}), {str(source): 'a' * 64})
        source = Path(f.__file__).read_text(); tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = [node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None) for node in calls]
        for name in ('replay_check', 'replay_monitor_check', 'infer_candidate', 'evaluate_metrics', 'unlink', 'rmtree', 'backward', 'set_num_threads'):
            self.assertNotIn(name, names)
        self.assertEqual(names.count('train_batch_mc2_perceptual_anchor'), 1)
        for node in calls:
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                self.assertNotIn((node.func.value.id, node.func.attr), (('prior', 'main'), ('prior', 'restore_parent'), ('prior', 'save_endpoint')))
        self.assertIn("'auxiliary_configuration': state['auxiliary_configuration']", source)
        self.assertIn("'phase': 'terminal_publication'", source)
        self.assertEqual(f.EVALUATION['metrics_only'], ['cer', 'dnsmos', 'utmos'])


if __name__ == '__main__': unittest.main()
