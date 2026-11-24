"""租用机CPU测试Dlr分阶段薄层；不调用GPU、训练主入口、评分或删除项目文件。"""

import copy
import ast
import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('Rental CPU tests require CUDA_VISIBLE_DEVICES empty')
import torch
import run_acdd_mc2_dlr_paired120_from3927_v1 as d


def losses(value=1.): return {key: float(value) for key in d.pair.LOSS_KEYS}
def entry(step, mel=.18963544070720673, rms=1.0055525302886963):
    return {'global_step': step, 'development': {'mel_l1': mel, 'prediction_rms': rms}, 'uniform64': {'GT_l1': .19156639277935028}}
def update(model, opt):
    for p in model.parameters(): p.grad = torch.ones_like(p)
    opt.step(); opt.zero_grad(set_to_none=True)
def toy(arm=d.ARMS[0], step=d.START):
    s, D = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    a, b = (torch.optim.Adam(model.parameters(), lr=lr, betas=(.5, .9)) for model, lr in ((s, 5e-5), (D, 2e-4)))
    for model, opt in ((s, a), (D, b)):
        update(model, opt)
        for value in opt.state.values(): value['step'].fill_(step)
    state = d.new_state(arm, 'toy'); state.update(restore_completed=True, completed_global_step=step, optimizer_cursors=d.old.actual_cursors(a, b))
    return state, s, D, a, b
def toy_call(parts):
    def call(): update(parts[2], parts[4]); update(parts[1], parts[3]); return losses()
    return call


class DlrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_default_stage_approval_SHA_gate_before_gpu_locks(self):
        with patch.object(d.training, 'acquire_runtime_locks', side_effect=AssertionError('lock touched')):
            with self.assertRaises(RuntimeError): d.main([])
        for arm in d.ARMS:
            argv = ['--run-approved', '--arm', arm, '--driver-sha256', 'a' * 64, '--tests-sha256', 'b' * 64]
            if arm == d.ARMS[1]:
                with self.assertRaises(RuntimeError): d.launch_arguments(argv)
                argv += ['--control-archive-sha256', 'c' * 64]
            self.assertEqual(d.launch_arguments(argv).arm, arm)

    def test_actual3927_parent_JSON_fields_and_fixed_guard(self):
        parent = d.cross.read_json(d.PARENT_RUN / 'run_contract.json'); done = d.cross.read_json(d.PARENT_RUN / 'completion.json')
        receipt = d.validate_parent_metadata(parent, done)
        self.assertEqual(receipt['completed_global_step'], 3927)
        for key, value in [('cleanup_verified', False), ('status', 'running')]:
            with self.assertRaises(RuntimeError): d.validate_parent_metadata(parent, dict(done, **{key: value}))
        self.assertEqual(d.cont.BASELINE3327['mel_l1'] * 1.05, .20217022821307185)

    def fixture(self, arm):
        parts = toy(arm); state, s, D, a, b = parts; config = SimpleNamespace(as_dict=lambda: {'toy': True})
        payload = {'global_step': 3927, 'run_contract_sha256': d.PARENT_CANONICAL, 'effective_optimizer_lrs': dict(d.PARENT_LRS),
            'mc_samples': 2, 'mc_private_seed_base': 74000000, 'resumable': True, 'epoch': 0, 'configuration': config.as_dict(),
            'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'stop_reason': 'maximum480_updates_completed',
            'execution_error': None, 'development_monitor_global_step': 3927, 'development': entry(3927)['development'],
            'student': copy.deepcopy(s.state_dict()), 'discriminator': copy.deepcopy(D.state_dict()),
            'student_optimizer': copy.deepcopy(a.state_dict()), 'discriminator_optimizer': copy.deepcopy(b.state_dict()),
            'rng_state': d.training.capture_rng_state()}
        return parts, payload, config, {'configuration': config.as_dict()}, {'last_monitor': entry(3927)}

    def test_both_roles_restore_exact_models_Adam_RNG_then_only_candidate_D_lr(self):
        for arm in d.ARMS:
            parts, payload, config, contract, receipt = self.fixture(arm); state, s, D, a, b = parts
            before = d.previous.content_state(s, D, a, b); payload_sha = d.probe.tree_sha(payload)
            update(s, a); update(D, b); torch.rand(3)
            with patch.object(d, 'PARENT_RNG_SHA', d.probe.tree_sha(payload['rng_state'])):
                result = d.restore_parent(payload, s, D, a, b, config, contract, receipt, torch.device('cpu'), arm)
            self.assertEqual(result['parent_restore'], before)
            self.assertEqual(b.param_groups[0]['lr'], d.ROLE_LRS[arm]['discriminator'])
            self.assertEqual(d.probe.tree_sha(payload), payload_sha)
            self.assertEqual(d.old.actual_cursors(a, b)['student']['uniform_step'], 3927)
            for role, opt in (('student', a), ('discriminator', b)):
                for index, values in opt.state_dict()['state'].items():
                    for key in ('step', 'exp_avg', 'exp_avg_sq'):
                        self.assertNotEqual(values[key].data_ptr(), payload[role + '_optimizer']['state'][index][key].data_ptr())

    def test_parent_wrong_header_RNG_or_Adam_nonfinite_refused(self):
        for key, value in [('global_step', 3447), ('effective_optimizer_lrs', d.ROLE_LRS[d.ARMS[1]]), ('mc_samples', 1),
                           ('configuration', {}), ('execution_error', 'bad'), ('score_semantics', 'noise')]:
            parts, payload, config, contract, receipt = self.fixture(d.ARMS[0]); payload[key] = value
            with self.assertRaises(RuntimeError): d.restore_parent(payload, *parts[1:], config, contract, receipt, torch.device('cpu'), d.ARMS[0])
        parts, payload, config, contract, receipt = self.fixture(d.ARMS[1])
        next(iter(payload['discriminator_optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
        with patch.object(d, 'PARENT_RNG_SHA', d.probe.tree_sha(payload['rng_state'])), self.assertRaises(RuntimeError):
            d.restore_parent(payload, *parts[1:], config, contract, receipt, torch.device('cpu'), d.ARMS[1])

    def test_original_absolute120_plan_seeds_and_exclusions(self):
        development = d.cross.read_json(d.DEV / 'development_contract.json'); rows = d.future_plan(development)
        self.assertEqual(len(rows), 120); self.assertEqual(rows[0]['global_step'], 3928); self.assertEqual(rows[-1]['global_step'], 4047)
        self.assertEqual((rows[0]['crop_seed'], rows[0]['noise_seed']), (105162, 205162))
        self.assertEqual(rows[0]['indices'], d.resume.GlobalBatchPlan(development['retained_train_indices']).for_step(3928))
        self.assertFalse(any(set(r['indices']) & set(development['excluded_train_indices']) for r in rows))

    def test_actual_Adam_returns_complete_and_half_or_late_failure(self):
        parts = toy(); d.complete_update(parts[0], 3928, parts[3], parts[4], toy_call(parts))
        self.assertEqual(parts[0]['completed_global_step'], 3928); self.assertFalse(parts[0]['partial_batch_may_have_updated_D'])
        for mode in ('half', 'after_return', 'raise_inside_S_step'):
            parts = toy(); state, s, D, a, b = parts
            if mode == 'raise_inside_S_step':
                saved = a.step
                def bad(): saved(); raise RuntimeError('inside')
                a.step = bad
            def call():
                update(D, b)
                if mode == 'half': raise RuntimeError('half')
                update(s, a)
                raise RuntimeError('after')
            with self.assertRaises(RuntimeError): d.complete_update(state, 3928, a, b, call)
            self.assertEqual(state['completed_global_step'], 3928 if mode == 'after_return' else 3927)
            self.assertEqual(state['partial_batch_may_have_updated_D'], mode != 'after_return')

    def test_pair_first_preD_dist_same_but_advFM_allowed_to_differ(self):
        row = dict(losses(), global_step=3928, indices=[1], crop_seed=105162, noise_seed=205162, input_identity={'a': 'x'},
            mc={key: key for key in ('source_noise', 'score_noise1', 't1', 't1_values', 'score_noise2', 't2', 't2_values', 'private_seed')})
        row['mc'].update(dist1=.1, dist2=.2); candidate = copy.deepcopy(row)
        candidate.update(student_adv=7., fm_raw=9., fm_weighted=18., student_total=30.)
        d.paired_input_check(candidate, row)
        for field in ('discriminator', 'dist_raw'):
            invalid = copy.deepcopy(candidate); invalid[field] += .01
            with self.assertRaises(RuntimeError): d.paired_input_check(invalid, row)
        invalid = copy.deepcopy(candidate); invalid['mc']['dist2'] += .01
        with self.assertRaises(RuntimeError): d.paired_input_check(invalid, row)
        candidate['global_step'] = row['global_step'] = 3929; candidate['dist_raw'] = 7.; candidate['mc']['dist2'] = 8.
        d.paired_input_check(candidate, row)
        candidate['mc']['t2'] = 'changed'
        with self.assertRaises(RuntimeError): d.paired_input_check(candidate, row)

    def simulate(self, values=None, fail_at=None):
        state = d.new_state(d.ARMS[0], 'toy'); initial = entry(3927); state['last_monitor'] = initial
        future = [{'global_step': i} for i in range(3928, 4048)]; calls, written = [], []
        def step(plan):
            n = plan['global_step']; state.update(completed_global_step=n, new_updates=n - 3927)
            return dict(losses(1. + (n - 3927) * .01), global_step=n, mc={'dist1': 1., 'dist2': 2.})
        def evaluate(n): calls.append(n); return entry(n, (values or {}).get(n, .19))
        def publish(name, value, append=True): written.append((name, copy.deepcopy(value)))
        def check():
            if fail_at and state['new_updates'] >= fail_at: raise d.old.BoundaryStop('time')
        return state, future, step, evaluate, publish, check, initial, calls, written

    def test_120_complete_raw_rows_two60_windows_only_monitors_0_60_120(self):
        parts = self.simulate(); result = d.run_updates(*parts[:7])
        self.assertEqual(parts[7], [3987, 4047]); self.assertEqual(len(result['monitors']), 3)
        self.assertEqual(len(result['rows']), 120); self.assertEqual(len(result['windows']), 2)
        self.assertEqual(result['reason'], 'maximum120_updates_completed')
        self.assertGreater(result['windows'][1]['means']['student_total'], result['windows'][0]['means']['student_total'])
        self.assertTrue(all(not x['used_as_stop_gate'] for x in result['windows']))

    def test_guard_earliest120_for_two_bad_new_points_and_extreme60(self):
        parts = self.simulate({3987: .203, 4047: .204}); result = d.run_updates(*parts[:7])
        self.assertEqual(parts[0]['new_updates'], 120); self.assertEqual(result['reason'], 'two_consecutive_DEV32_above3327_by5percent')
        parts = self.simulate({3987: .22}); result = d.run_updates(*parts[:7])
        self.assertEqual(parts[0]['new_updates'], 60); self.assertEqual(result['reason'], 'original_S0_extreme_guard')
        parts = self.simulate(fail_at=11)
        with self.assertRaises(d.old.BoundaryStop): d.run_updates(*parts[:7])
        self.assertEqual(len([r for r in parts[8] if r[0] == 'step_losses.jsonl']), 11); self.assertEqual(parts[7], [])

    def test_replay_recursive_tolerance_but_identity_and_every_utterance_strict(self):
        ref = d.cross.read_json(d.PARENT_RUN / 'completion.json')['state']['last_monitor']['uniform64']
        d.same_tree(copy.deepcopy(ref), ref)
        changed = copy.deepcopy(ref); changed['per_utterance_GT_l1'][0] += 1e-8; d.same_tree(changed, ref)
        changed['per_utterance_GT_l1'][0] += 1e-3
        with self.assertRaises(RuntimeError): d.same_tree(changed, ref)
        changed = copy.deepcopy(ref); changed['prediction_identities'][0]['sha256'] = '0' * 64
        with self.assertRaises(RuntimeError): d.same_tree(changed, ref)

    def test_stage_budget_exact_models_plus_future_eval_and_joint_metadata(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            root = Path(temp) / 'trial'; root.mkdir()
            for arm, expected, role in ((d.ARMS[0], 3717140480, 'student'), (d.ARMS[1], 4832505856, 'full')):
                budget = d.Budget(root, arm); self.assertEqual(budget.require()['required_bytes'], expected)
                self.assertEqual(set(budget.pending), {role, 'later_DEV90'})
                with patch.object(d.pair, 'available_bytes', return_value=expected - 1):
                    with self.assertRaises(d.pair.InsufficientDisk): budget.require()
                with patch.object(budget, 'metadata_used', return_value=d.META_LIMIT):
                    with self.assertRaises(d.pair.InsufficientDisk): budget.require(1)

    def test_real_joint_metadata_scan_counts_both_arms_logs_and_excludes_only_models(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            root = Path(temp)
            for arm in d.ARMS:
                (root / arm).mkdir(); (root / (arm + '.log')).write_text('stage log')
                (root / arm / 'monitor.jsonl').write_text('observations')
                (root / arm / 'test.pt').write_bytes(b'not a real model')
            budget = d.Budget(root, d.ARMS[1]); prior = budget.metadata_used()
            self.assertEqual(prior, 2 * (len('stage log') + len('observations')))
            budget.write(root / d.ARMS[1] / 'step_losses.jsonl', {'step': 3928}, append=True)
            self.assertEqual(budget.metadata_used(), prior + (root / d.ARMS[1] / 'step_losses.jsonl').stat().st_size)

    def save_fixture(self, directory, arm):
        parts = toy(arm, 3987); state = parts[0]; parts[4].param_groups[0]['lr'] = d.ROLE_LRS[arm]['discriminator']
        state.update(last_monitor=entry(3987), reason='original_S0_extreme_guard', new_updates=60)
        root = Path(directory) / 'run'; (root / arm).mkdir(parents=True)
        return parts, d.Budget(root, arm), SimpleNamespace(active=lambda: None), SimpleNamespace(as_dict=lambda: {'toy': True})

    def test_save_controlS_only_candidatefull_and_payload_evaluator_fields(self):
        for arm in d.ARMS:
            with tempfile.TemporaryDirectory() as temp, patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
                parts, budget, clock, config = self.save_fixture(temp, arm)
                d.save_endpoint(parts[0], config, *parts[1:], budget, clock)
                state = parts[0]; row = state['student_snapshot' if arm == d.ARMS[0] else 'full_snapshot']
                payload = torch.load(row['path'], map_location='cpu')
                self.assertEqual(payload['global_step'], 3987); self.assertEqual(payload['epoch'], 0)
                self.assertEqual(payload['effective_optimizer_lrs'], d.ROLE_LRS[arm]); self.assertEqual(payload['mc_samples'], 2)
                self.assertEqual(payload['resumable'], arm == d.ARMS[1])
                self.assertEqual('student_optimizer' in payload, arm == d.ARMS[1]); self.assertIn('student', payload)
                # 直接调用冻结评价入口的真实payload验证函数，绝不运行其main或模型。
                evaluator = importlib.import_module('evaluate_development_candidates')
                original_config = d.cross.read_json(d.PARENT_RUN / 'run_contract.json')['configuration']
                joint = {'configuration': original_config}; payload['configuration'] = original_config
                payload['run_contract_sha256'] = d.training.canonical_json_sha256(joint)
                spec = {'epoch': 0, 'global_step': 3987, 'state_key': 'student'}
                self.assertEqual(evaluator.validate_payload(payload, spec, joint), payload['run_contract_sha256'])
                self.assertNotIn('student' if arm == d.ARMS[0] else 'full', budget.pending)
                self.assertEqual(len(list((budget.output / arm).glob('*.pt'))), 1)
                with self.assertRaises(RuntimeError): d.save_endpoint(state, config, *parts[1:], budget, clock)

    def test_half_or_technical_failure_cannot_save_and_commit_cursor_survives_hash_error(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(d.pair, 'available_bytes', return_value=20 * 1024**3):
            parts, budget, clock, config = self.save_fixture(temp, d.ARMS[1]); state = parts[0]
            state['partial_batch_may_have_updated_D'] = True
            with self.assertRaises(RuntimeError): d.save_endpoint(state, config, *parts[1:], budget, clock)
            state['partial_batch_may_have_updated_D'] = False; state['execution_error'] = 'bad'
            with self.assertRaises(RuntimeError): d.save_endpoint(state, config, *parts[1:], budget, clock)
            del state['execution_error']
            with patch.object(d.base, 'sha256_file', side_effect=OSError('hash error')):
                with self.assertRaises(OSError): d.save_endpoint(state, config, *parts[1:], budget, clock)
            self.assertEqual(state['last_full_saved_global_step'], 3987); self.assertTrue(state['full_snapshot']['committed'])
            self.assertNotIn('full', budget.pending)

    def test_terminal_rename_has_no_postcheck_and_cleanup_attempts_all(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / d.ARMS[0]).mkdir(); budget = d.Budget(root, d.ARMS[0])
            gate = Mock(side_effect=[None, RuntimeError('late')])
            with patch.object(budget, 'require', gate): budget.publish_success(root / d.ARMS[0] / 'completion.json', {'cleanup_verified': True})
            self.assertEqual(gate.call_count, 1)
        actions = []
        def action(name, bad=False):
            def f():
                actions.append(name)
                if bad: raise RuntimeError(name)
            return f
        errors = d.old.cleanup_resources(SimpleNamespace(close=action('hooks')), SimpleNamespace(close=action('clock')),
            SimpleNamespace(flush=action('flush', True), close=action('log')), [SimpleNamespace(close=action('lock'))])
        self.assertEqual(actions, ['hooks', 'clock', 'flush', 'log', 'lock']); self.assertEqual(len(errors), 1)

    def archive_fixture(self):
        joint = {'fixed_candidate': d.ROLE_LRS, 'configuration': {}}; canonical = d.training.canonical_json_sha256(joint)
        path = str(d.OUTPUT / d.ARMS[0] / 'student_step_004047.pt')
        state = {'student_snapshot': {'path': path, 'sha256': 'a' * 64, 'global_step': 4047, 'committed': True, 'resumable': False, 'frozen_readonly': True},
                 'completed_global_step': 4047, 'partial_batch_may_have_updated_D': False, 'arm': d.ARMS[0], 'restore_completed': True,
                 'last_student_saved_global_step': 4047}
        control = {'status': 'completed120', 'cleanup_verified': True, 'contract_sha256': canonical, 'state': state}
        infer = {'spec': {'checkpoint': path, 'sha256': 'a' * 64, 'global_step': 4047}, 'saved_training_contract_sha256': canonical,
                 'wav_sha256': {str(i): 'b' * 64 for i in range(90)}}
        summary = {'status': 'development_only', 'evaluated_pairs': 90, 'metric_identity_sha256': d.METRIC_IDENTITY_SHA, 'metrics': {'CER': 100.}}
        folder = Path(d.EVALUATION[d.ARMS[0]]['output_root']) / d.EVALUATION[d.ARMS[0]]['label'] / 'wavs'
        originals = [(path, 'a' * 64)] + [(str(folder / ('pair_%04d.wav' % i)), 'b' * 64) for i in range(90)]
        record = {'schema_version': 1, 'status': 'completed_private_archive_and_remote_removal', 'joint_contract_file_sha256': 'f' * 64,
                  'joint_contract_canonical_sha256': canonical, 'control_global_step': 4047,
                  'records': [{'original_remote_path': p, 'sha256': sha, 'size_bytes': 100,
                    'private_backup_absolute_path': 'C:/Users/rich_/AppData/Local/VoiceGradPrivateBackups/test/' + str(i),
                    'backup_sha256': sha, 'backup_size_bytes': 100, 'remote_removed': True} for i, (p, sha) in enumerate(originals)]}
        return record, joint, 'f' * 64, control, infer, summary

    def test_control_archive_exact91_absent_and_score_cannot_select_candidate(self):
        parts = self.archive_fixture()
        d.validate_control_archive(*parts, exists=lambda p: False)
        parts[-1]['metrics']['CER'] = 0.
        d.validate_control_archive(*parts, exists=lambda p: False)
        self.assertEqual(parts[1]['fixed_candidate'], d.ROLE_LRS)
        with self.assertRaises(RuntimeError): d.validate_control_archive(*parts, exists=lambda p: True)
        for change in ('extra', 'missing', 'duplicate', 'backup_sha', 'not_removed', 'bad_path'):
            current = copy.deepcopy(parts); records = current[0]['records']
            if change == 'extra': records.append(dict(records[0], original_remote_path='/other'))
            elif change == 'missing': records.pop()
            elif change == 'duplicate': records[0] = records[1]
            elif change == 'backup_sha': records[0]['backup_sha256'] = 'c' * 64
            elif change == 'not_removed': records[0]['remote_removed'] = False
            else: records[0]['private_backup_absolute_path'] = 'relative.pt'
            with self.assertRaises(RuntimeError): d.validate_control_archive(*current, exists=lambda p: False)

    def test_candidate_gate_actual3927_evaluator_JSON_CSV_adapted_to_control_archive(self):
        # 使用真实3927已完成的三份JSON和五份CSV结构；仅替换新角色、步数、路径与测试文件SHA。
        # 临时metadata中的4047是假设兼容样本，不声称产生了新模型、音频或性能结果。
        original = d.DEV / 'evaluation_mc2_continuation480_endpoint_v1'
        original_system = original / 'student_mc2_continuation480_endpoint_v1'
        actual_identity = d.cross.read_json(original / 'run_identity.json')
        actual_infer = d.cross.read_json(original_system / 'inference_complete.json')
        actual_summary = d.cross.read_json(original_system / 'metrics/development_summary_cer_dnsmos_sva_utmos.json')
        self.assertEqual(d.base.sha256_file(original / 'run_identity.json'), actual_infer['run_identity_sha256'])
        self.assertEqual(d.base.sha256_file(original_system / 'inference_complete.json'), actual_summary['inference_complete_sha256'])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'training'; evaluation = Path(temp) / 'evaluation'
            label = d.EVALUATION[d.ARMS[0]]['label']; table = dict(d.EVALUATION)
            table[d.ARMS[0]] = {'output_root': str(evaluation), 'label': label}
            with patch.object(d, 'OUTPUT', output), patch.object(d, 'EVALUATION', table):
                stage = output / d.ARMS[0]; system = evaluation / label
                stage.mkdir(parents=True); (system / 'metrics').mkdir(parents=True)
                record, joint, unused, control, unused_infer, unused_summary = self.archive_fixture()
                joint['configuration'] = d.cross.read_json(d.PARENT_RUN / 'run_contract.json')['configuration']
                canonical = d.training.canonical_json_sha256(joint)
                def write(path, value):
                    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf8')
                    return d.base.sha256_file(Path(path))
                contract_sha = write(output / 'run_contract.json', joint)
                control['contract_sha256'] = canonical
                record.update(joint_contract_file_sha256=contract_sha, joint_contract_canonical_sha256=canonical,
                              control_completion_file_sha256=write(stage / 'completion.json', control))
                infer = copy.deepcopy(actual_infer); identity = copy.deepcopy(actual_identity); summary = copy.deepcopy(actual_summary)
                infer.update(system=label, student_updates=4047, saved_training_contract_sha256=canonical)
                infer['spec'].update(checkpoint=control['state']['student_snapshot']['path'], sha256='a' * 64, global_step=4047)
                identity.update(systems={label: infer['spec']}, candidate_training_contract=str(output / 'run_contract.json'))
                del identity['inputs'][str(d.PARENT)]; del identity['inputs'][str(d.PARENT_RUN / 'run_contract.json')]
                identity['inputs'].update({infer['spec']['checkpoint']: 'a' * 64, str(output / 'run_contract.json'): contract_sha})
                infer['run_identity_sha256'] = write(evaluation / 'run_identity.json', identity)
                pairs = d.csv_rows(original_system / 'pairs.csv')
                self.assertEqual(d.base.sha256_file(original_system / 'pairs.csv'), actual_infer['pairs_csv_sha256'])
                for row in pairs: row['output_wav'] = str(system / 'wavs' / ('pair_%04d.wav' % int(row['pair_id'])))
                with (system / 'pairs.csv').open('w', newline='', encoding='utf8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
                infer['pairs_csv_sha256'] = d.base.sha256_file(system / 'pairs.csv')
                for row in record['records'][1:]:
                    pair_id = str(int(Path(row['original_remote_path']).stem.split('_')[1]))
                    row['sha256'] = row['backup_sha256'] = infer['wav_sha256'][pair_id]
                record['infer_completion_file_sha256'] = write(system / 'inference_complete.json', infer)
                summary.update(system=label, inference_complete_sha256=record['infer_completion_file_sha256'], run_identity_sha256=infer['run_identity_sha256'])
                for name, digest in summary['metric_csv_sha256'].items():
                    source = original_system / 'metrics' / (name + '.csv')
                    self.assertEqual(d.base.sha256_file(source), digest)
                    (system / 'metrics' / source.name).write_bytes(source.read_bytes())
                record['evaluation_summary_file_sha256'] = write(system / 'metrics/development_summary_cer_dnsmos_sva_utmos.json', summary)
                (stage / 'step_losses.jsonl').write_text('\n'.join(json.dumps({'global_step': i}) for i in range(3928, 4048)))
                args = SimpleNamespace(control_archive_sha256=write(output / d.ARCHIVE_NAME, record))
                rows, proof = d.candidate_gate(args, joint)
                self.assertEqual(len(rows), 120); self.assertEqual(proof['archive_receipt_file_sha256'], args.control_archive_sha256)
                # 评分CSV变化即拒绝，且不需要再次加载控制S或任何已归档WAV。
                (system / 'metrics' / 'cer.csv').write_text('changed retained CSV')
                with self.assertRaisesRegex(RuntimeError, 'Retained control CSV changed'): d.candidate_gate(args, joint)

    def test_import_closure_unknown_source_fails_instead_of_allowing_test_imports(self):
        fake = SimpleNamespace(__file__=str(d.ROOT / 'unreviewed_new_module.py'))
        with patch.dict(sys.modules, {'_unreviewed_probe': fake}):
            sources = {str(Path(getattr(m, '__file__')).resolve()): 'a' * 64 for m in tuple(sys.modules.values())
                       if getattr(m, '__file__', None) and Path(getattr(m, '__file__')).resolve().suffix == '.py'
                       and d.ROOT in Path(getattr(m, '__file__')).resolve().parents}
            del sources[fake.__file__]
            with self.assertRaisesRegex(RuntimeError, 'unreviewed_new_module'): d.imported_source_files(sources, {})

    def test_source_scope_excludes_old_metadata_loaders_and_archived3447(self):
        tree = ast.parse(Path(d.__file__).read_text())
        forbidden = {'load_metadata', 'validate_effective_adam', 'choose_plan', 'validate_uniform_initial', 'train_batch_mc1',
                     'unlink', 'rmtree', 'remove', 'candidate_gate', 'window20'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute): self.assertNotIn(node.func.attr, forbidden)
        self.assertNotIn(str(d.cont.PARENT), d.FIXED_PINS)
        self.assertIn(str(d.pair.PARENT / 'latest_full.pt'), d.FIXED_PINS)
        d.validate_supplemental_source_pins()

    def test_forward_limits_per_arm_exact_production_budget(self):
        self.assertEqual(d.FORWARD_LIMITS, {'teacher': 303, 'student': 252, 'vocoder': 366, 'active_D_branches': 3840, 'fixed_D_branches': 48})
        ledger = d.ForwardLedger(); hook = ledger.before('student')
        for _ in range(252): hook(None, ())
        with self.assertRaises(RuntimeError): hook(None, ())
        self.assertEqual(ledger.counts['student'], 252)


if __name__ == '__main__': unittest.main(verbosity=2)
