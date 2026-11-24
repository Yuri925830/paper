"""从完整3927开始一条独立360步轨迹；不重建旧4047，不自动评估或长训。"""

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
PRIOR_FILE = TOOLS / 'run_acdd_mc2_continuation360_from3927_v1.py'
PRIOR_SHA = 'c7ecf8b32ae531ee9039dbc80a9b60974d98f6c58b3b59cb53bf2b7099da674c'
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Run only on the registered Matpool rental')
if hashlib.sha256(PRIOR_FILE.read_bytes()).hexdigest() != PRIOR_SHA:
    raise RuntimeError('Frozen360 restoration/planning/storage primitives changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_mc2_continuation360_from3927_v1 as prior

d, old, previous, cross, pair, base = prior.d, prior.old, prior.previous, prior.cross, prior.pair, prior.base
training, resume, probe, DEV = prior.training, prior.resume, prior.probe, prior.DEV
START, STOP, UPDATES, OVERLAP_STOP = 3927, 4287, 360, 4047
OUTPUT = DEV / 'acdd_mc2_independent360_from3927_v1'
ARM = 'candidate_mc2'
PROTOCOL = DEV / 'ACDD3927_MC2_INDEPENDENT360_PROTOCOL_CN.md'
HELPER = 'acdd_mc2_train_batch_3928_4287_v1'
HELPER_SHA = 'c99091c3d666621682814a8a697701fca9beb99b0ccefd90d67f24b62cbef6d8'
HELPER_TEST_SHA = '19c851159fdcbb5fee7dff01d8cb6de6b4b2a31d5caebc56162f3c9807f71636'
AA_ROOT = DEV / 'acdd_mc2_one_step_repeatability_v1'
AA_COUNTS = {'teacher': 50, 'student': 14, 'vocoder': 10, 'active_D_branches': 64, 'fixed_D_branches': 32}
LRS = dict(prior.LRS)
META_LIMIT, INITIAL_REQUIRED_BYTES = prior.META_LIMIT, prior.INITIAL_REQUIRED_BYTES
FORWARD_LIMITS, TRAIN_COUNTS, MONITOR_COUNTS = prior.FORWARD_LIMITS, prior.TRAIN_COUNTS, prior.MONITOR_COUNTS
MONITOR_UPDATES = tuple(range(0, UPDATES + 1, 60))
THREAD_ENV = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
Budget, future_plan, restore_parent = prior.Budget, prior.future_plan, prior.restore_parent
EVALUATION = {'output_root': str(DEV / 'evaluation_mc2_independent360_endpoint_v1'),
    'label': 'student_mc2_independent360_endpoint_v1', 'normal_endpoint_global_step': STOP,
    'metrics_only': ['cer', 'dnsmos', 'utmos'], 'executed_by_this_driver': False,
    'protective_endpoint_requires_separate_review': True}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    parser.add_argument('--repeatability-reviewed', action='store_true')
    for name in ('driver', 'tests', 'protocol', 'repeatability-results', 'repeatability-completion'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args(argv)
    if not args.run_approved or not args.repeatability_reviewed:
        raise RuntimeError('Require bounded360 approval and reviewed actual A/A diagnostic')
    for name in ('driver', 'tests', 'protocol', 'repeatability_results', 'repeatability_completion'):
        value = getattr(args, name + '_sha256')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value): raise RuntimeError('Require reviewed full SHA: ' + name)
    return args


def runtime():
    actual = {name: os.environ.get(name) for name in THREAD_ENV}; threads = torch.get_num_threads()
    if actual != THREAD_ENV or type(threads) is not int or threads < 1:
        raise RuntimeError('Require explicit environment and positive measured thread count; never force it')
    if ((platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6')
            or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve()
            or cross.flags() != cross.ORIGINAL_FLAGS): raise RuntimeError('Original runtime and precision changed')
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'flags': cross.flags(), 'thread_environment': actual,
        'torch_threads': threads, 'torch_interop_threads': torch.get_num_interop_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'historical_CONTROL71128_thread_environment': 'not_recorded_cannot_claim_same'}


class ForwardLedger(prior.ForwardLedger):
    def __init__(self): super().__init__(); self.check = lambda: None
    def before(self, name):
        original = super().before(name)
        def before_call(module, arguments): self.check(); original(module, arguments)
        return before_call


@contextlib.contextmanager
def deadline_steps(optimizer, d_optimizer, check):
    # 只在当前两个实例的step调用前查截止点，不加梯度摘要或改任何更新运算。
    installed = []; missing = object()
    try:
        for opt in (d_optimizer, optimizer):
            original, local = opt.step, opt.__dict__.get('step', missing)
            def step(*args, _original=original, **kwargs): check(); return _original(*args, **kwargs)
            opt.step = step; installed.append((opt, local))
        yield
    finally:
        for opt, local in reversed(installed):
            if local is missing: del opt.step
            else: opt.step = local


def new_state(contract_sha):
    state = d.new_state(d.ARMS[0], contract_sha)
    state.update(arm=ARM, effective_optimizer_lrs=dict(LRS), actual_updates_this_process=0,
        overlapping_historical_step_numbers=0, beyond_historical_step_numbers=0, historical_input_checks=0)
    return state


def complete_update(state, step, optimizer, d_optimizer, call):
    # 保留实际D/S正常返回和全参数Adam游标的证明；计数只描述本次独立轨迹。
    if step != state['completed_global_step'] + 1 or not START < step <= STOP:
        raise RuntimeError('Require exactly the next3928..4287 absolute step')
    if any(x['uniform_step'] != step - 1 for x in old.actual_cursors(optimizer, d_optimizer).values()):
        raise RuntimeError('Entering Adam cursor differs')
    state.update(attempted_global_step=step, partial_batch_may_have_updated_D=True, phase='D_S_update'); returned = False
    try:
        with old.observe_adam_step_returns(state, optimizer, d_optimizer): losses = call()
        returned = True
    finally:
        try:
            cursors = old.actual_cursors(optimizer, d_optimizer); state['optimizer_cursors'] = cursors
            if all(x['uniform_step'] == step for x in cursors.values()) and state['adam_step_returns'] == {'student': 1, 'discriminator': 1}:
                count = step - START
                state.update(completed_global_step=step, new_updates=count, actual_updates_this_process=count,
                    overlapping_historical_step_numbers=min(count, 120), beyond_historical_step_numbers=max(0, count - 120),
                    partial_batch_may_have_updated_D=False, exception_after_both_adam_steps=not returned)
        except BaseException as error: state['cursor_inspection_error'] = repr(error)
    if state['partial_batch_may_have_updated_D']: raise RuntimeError('Missing complete D then S update proof')
    state['unpublished_returned_losses'] = {'global_step': step, 'losses': losses}
    pair.finite_losses(losses); return losses


def historical_input_check(row, control):
    # 只对真正的输入/随机抽样身份作硬核对；教师目标和损失是模型计算结果。
    if not START < row['global_step'] <= OVERLAP_STOP: raise RuntimeError('Historical inputs only cover3928..4047')
    for key in ('global_step', 'indices', 'crop_seed', 'noise_seed', 'input_identity'):
        if row[key] != control[key]: raise RuntimeError('Historical input mismatch: ' + key)
    for key in ('source_noise', 'score_noise1', 't1', 't1_values', 'score_noise2', 't2', 't2_values', 'private_seed', 'mc_samples', 'global_step'):
        if row['mc'][key] != control['mc'][key]: raise RuntimeError('Historical MC draw mismatch: ' + key)
    if row['actual_forward_counts'] != TRAIN_COUNTS: raise RuntimeError('Wrong unchanged training calls')
    # 两条轨迹的全部raw记录分别保留，离线可以比较任意分项，无须扩大现场差异树。
    return {'inputs_match': True, 'computed_values_compared_as': 'descriptive_only',
        'raw_loss_delta_from_historical': {k: row[k] - control[k] for k in pair.LOSS_KEYS},
        'teacher_target_SHA_equal': {k: row['mc'][k]['teacher_target']['sha256'] == control['mc'][k]['teacher_target']['sha256'] for k in ('first', 'second')},
        'historical_state_reconstruction_claimed': False}


def window120(rows):
    if len(rows) != 120 or rows[0]['global_step'] not in (3928, 4048, 4168): raise RuntimeError('Only three complete120-step windows')
    first = rows[0]['global_step']
    if [r['global_step'] for r in rows] != list(range(first, first + 120)): raise RuntimeError('Nonconsecutive raw window')
    for row in rows: pair.finite_losses(row)
    return {'first_global_step': first, 'global_step': rows[-1]['global_step'], 'count': 120,
        'phase': 'independent_trajectory', 'overlaps_historical_step_numbers': first == 3928,
        'means': {k: math.fsum(r[k] for r in rows) / 120 for k in pair.LOSS_KEYS},
        'MC2_dist_means': {k: math.fsum(r['mc'][k] for r in rows) / 120 for k in ('dist1', 'dist2')},
        'first_last_difference': {k: rows[-1][k] - rows[0][k] for k in pair.LOSS_KEYS}, 'used_as_stop_gate': False}


def loss_trend(rows):
    if len(rows) != UPDATES: return {'available': False, 'completed_records': len(rows), 'reason': 'requires_all360_steps'}
    windows = [window120(rows[offset:offset + 120]) for offset in (0, 120, 240)]
    return {'available': True, 'all_three_windows': windows,
        'primary_third_minus_second_window': {k: windows[2]['means'][k] - windows[1]['means'][k] for k in pair.LOSS_KEYS},
        'used_as_stop_gate': False, 'all360_are_actual_updates_of_this_independent_trajectory': True}


def failure_record(value):
    # 仅失败收据把非有限值明确标注，否则严格JSON会连失败原因一起拒绝保存。
    if isinstance(value, float) and not math.isfinite(value): return {'nonfinite_float': repr(value)}
    if isinstance(value, dict): return {key: failure_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [failure_record(item) for item in value]
    return value


def run_updates(state, future, step_call, evaluate, publish, check, initial):
    if [r['global_step'] for r in future] != list(range(START + 1, STOP + 1)) or initial['global_step'] != START:
        raise RuntimeError('Wrong prospective independent360 plan')
    rows, windows, monitors = [], [], [initial]
    initial_guards = old.guard_reasons([initial], d.cont.BASELINE3327)
    if initial_guards: return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': initial_guards[0]}
    reason = 'maximum360_updates_completed'
    for planned in future:
        check(); row = step_call(planned); state['unpublished_step_record'] = row
        publish('step_losses.jsonl', row); state.pop('unpublished_step_record', None); rows.append(row)
        state['complete_loss_records'] = len(rows)
        if len(rows) % 120 == 0:
            window = window120(rows[-120:]); windows.append(window); publish('loss_windows.jsonl', window)
        if len(rows) % 60 == 0:
            entry = evaluate(planned['global_step']); state['last_monitor'] = entry
            monitors.append(entry); publish('monitor.jsonl', entry)
            # 只使用本轨迹实际不同step的监测；不并入历史控制或重复的3927观察。
            guards = old.guard_reasons(monitors[1:], d.cont.BASELINE3327); state['guard_reasons'] = guards
            publish('guard_checks.jsonl', {'global_step': entry['global_step'], 'stop_reasons': guards,
                'distinct_global_steps_used': [r['global_step'] for r in monitors[1:]], 'historical_points_pooled': False})
            if guards: reason = guards[0]; break
        publish('status.json', state, False)
    return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': reason}


def repeatability_evidence(args, files, check):
    paths = {str(AA_ROOT / 'results.json'): args.repeatability_results_sha256,
             str(AA_ROOT / 'completion.json'): args.repeatability_completion_sha256}
    d.check_files(paths, check); result = cross.read_json(AA_ROOT / 'results.json'); done = cross.read_json(AA_ROOT / 'completion.json')
    expected = ('two_independent_one_step_diagnostics_completed', True, 2, AA_COUNTS, args.repeatability_results_sha256)
    if tuple(done[k] for k in ('status', 'cleanup_verified', 'complete_updates', 'actual_forward_counts', 'results_sha256')) != expected:
        raise RuntimeError('Require actual two-child A/A completion and cleanup')
    if result['status'] != expected[0] or result['complete_updates'] != 2 or result['actual_forward_counts'] != AA_COUNTS:
        raise RuntimeError('A/A result identity differs')
    if [(r['child'], r['returncode']) for r in result['child_exits']] != [('child_a', 0), ('child_b', 0)]:
        raise RuntimeError('Both real child processes must have exited successfully')
    if any((AA_ROOT / name).exists() for name in ('failure.json', 'cleanup_failure.json')): raise RuntimeError('A/A contains technical failure evidence')
    cross.merge_hashes(files, paths)
    return {'files_sha256': paths, 'reviewed_before_launch': args.repeatability_reviewed,
        'all_compared_values_identical': result['all_compared_values_identical'],
        'finite_differences_are_not_a_performance_or_reproducibility_gate': True}


def prepare_context(args, check):
    files = {p: sha for p, sha in d.FIXED_PINS.items() if not p.endswith('/acdd_mc2_fm_cross_3447_3927_v1/results.json')}
    files.update(d.NEW_SOURCE_PINS)
    files.update({str(PRIOR_FILE): PRIOR_SHA, str(TOOLS / 'run_acdd_mc2_dlr_paired120_from3927_v1.py'): prior.DLR_SHA,
        str(TOOLS / 'acdd_mc2_train_batch_v1.py'): 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16',
        str(Path(__file__).resolve()): args.driver_sha256, str(TOOLS / 'test_run_acdd_mc2_independent360_from3927_v1.py'): args.tests_sha256,
        str(TOOLS / (HELPER + '.py')): HELPER_SHA, str(TOOLS / ('test_' + HELPER + '.py')): HELPER_TEST_SHA,
        str(PROTOCOL): args.protocol_sha256})
    d.check_files(files, check); d.validate_supplemental_source_pins()
    aa = repeatability_evidence(args, files, check)
    mc = importlib.import_module(HELPER)
    if (mc.START, mc.STOP, mc.MC2_SEED) != (START, STOP, 74000000): raise RuntimeError('Wrong unchanged MC2 range')
    parent = cross.read_json(d.PARENT_RUN / 'run_contract.json')
    receipt = d.validate_parent_metadata(parent, cross.read_json(d.PARENT_RUN / 'completion.json'))
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if parent['configuration'] != config.as_dict() or (config.lambda_fm, config.lambda_dist) != (2., 45.): raise RuntimeError('Keep originalFM2/dist45 configuration')
    development, dependencies = resume.load_development(DEV / 'development_contract.json')
    if (parent['training_indices'], parent['excluded_train_indices']) != (development['retained_train_indices'], development['excluded_train_indices']):
        raise RuntimeError('Original training data and52 exclusions changed')
    cross.merge_hashes(files, dependencies); cross.merge_hashes(files, d.imported_source_files(parent['file_sha256'], files))
    for relative, digest in training.collect_source_fingerprints(ROOT).items():
        path = str(ROOT / relative) if not Path(relative).is_absolute() else relative
        if parent['file_sha256'].get(path) != digest: raise RuntimeError('Production source changed')
        files[path] = digest
    for asset in (base.TEACHER, base.VOCODER_DIR / 'generator_step_2500000.pt', ROOT / 'data/stats/mel_mean.npy', ROOT / 'data/stats/mel_std.npy'):
        files[str(asset)] = parent['file_sha256'][str(asset)]
    plan = parent['monitor64_plan']
    if plan != cross.read_json(previous.CROSS_IDENTITY)['plan']: raise RuntimeError('Original64 monitor plan changed')
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387: raise RuntimeError('Dataset indexing changed')
    future = future_plan(development)
    future_files = previous.register_future_files(dataset, future, files, check)
    monitor_files = previous.register_future_files(dataset, [{'indices': plan['selected_indices']}], files, check)
    if len(monitor_files) != 192 or any(parent['file_sha256'].get(p) != sha for p, sha in monitor_files.items()): raise RuntimeError('Fixed64 features changed')
    controls, control_monitors, historical = prior.control_evidence(parent, future, files, check)
    historical = {k: v for k, v in historical.items() if k not in ('role', 'control_replayed_updates')}
    historical.update(role='historical_comparison_only_not_state_reconstruction', overlapping_absolute_step_numbers=120)
    d.check_files(files, check)
    return mc, parent, receipt, config, development, dataset, future, future_files, monitor_files, files, controls, control_monitors, historical, aa


def make_contract(args, parent, receipt, config, development, future, future_files, monitor_files, files, historical, aa, environment):
    return {'kind': 'MC2_FM2_independent360_from3927', 'protocol_sha256': args.protocol_sha256,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS, 'runtime': environment,
        'paper_reference_coefficients': {'lambda_fm': 2., 'lambda_dist': 45.},
        'disclosed_extensions': {'student_optimizer_lr': .00005, 'MC_samples': 2,
            'score_target_interpretation': 'clean_x0_sqrt_alpha_bar', 'not_claimed_exact_paper_reproduction': True},
        'parent_checkpoint': str(d.PARENT), 'parent_checkpoint_sha256': d.PARENT_SHA,
        'parent_contract_sha256': d.PARENT_CANONICAL, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'start_after_global_step': START, 'stop_global_step': STOP, 'maximum_actual_updates': UPDATES,
        'historical_state_reconstruction_claimed': False, 'overlapping_historical_step_numbers': [3928, 4047],
        'future_batches': future, 'future_training_feature_sha256': future_files, 'fixed64_feature_sha256': monitor_files,
        'training_indices': development['retained_train_indices'], 'excluded_train_indices': development['excluded_train_indices'],
        'input_identities': parent['input_identities'], 'monitor64_plan': parent['monitor64_plan'],
        'baseline3927_monitor': receipt['last_monitor'], 'baseline3327_development': d.cont.BASELINE3327,
        'original_step0_guard_anchor': d.cont.S0_ANCHOR, 'fixed_frozen_models_sha256': parent['fixed_frozen_models_sha256'],
        'monitor_updates': list(MONITOR_UPDATES), 'loss_windows': [[3928, 4047], [4048, 4167], [4168, 4287]],
        'loss_keys': list(pair.LOSS_KEYS), 'primary_trend': 'third120_minus_second120; all3 complete windows shown; no scalar loss gate',
        'historical_control': historical, 'reviewed_repeatability_diagnostic': aa, 'runtime_files': files,
        'historical_input_identity_strict': True, 'historical_computed_loss_and_target_SHA_comparison': 'descriptive_only',
        'forward_limits': FORWARD_LIMITS, 'storage': {'initial_required_bytes': INITIAL_REQUIRED_BYTES,
            'full_upper_bytes': d.FULL_BYTES, 'save_slack': previous.MODEL_SLACK, 'system_reserve': previous.RESERVE,
            'directory_reserve': previous.DIR_RESERVE, 'metadata_limit': META_LIMIT, 'later_DEV90_bytes': d.EVAL_BYTES},
        'evaluation': EVALUATION, 'formal_gate_only': {'UTMOS_min': 3.9812097230664008, 'CER_percent_max': 2.109892438933927,
            'DNSMOS_P808_strictly_greater_than': 3.77, 'SVA_gate': False, 'OVRL_gate': False},
        'compute_seconds': 1500, 'cleanup_only_seconds': 90, 'original_flags': cross.ORIGINAL_FLAGS,
        'no_automatic_evaluation_training_retry_archive_or_deletion': True, 'formal3327_remains_stopped': True}


def save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock):
    if (not state['restore_completed'] or state['partial_batch_may_have_updated_D'] or state.get('execution_error')
            or state.get('exception_after_both_adam_steps') or state.get('cursor_inspection_error')):
        raise RuntimeError('Never save a technical failure or half update')
    step = state['completed_global_step']; count = step - START; last = state['last_monitor']
    if step not in range(3987, STOP + 1, 60) or last['global_step'] != step or state['arm'] != ARM:
        raise RuntimeError('Only an actual complete monitored new-trajectory endpoint; never zero steps')
    if config.as_dict() != base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0).as_dict(): raise RuntimeError('Save actual unchangedFM2 configuration')
    if (state['actual_updates_this_process'], state['new_updates'], state['overlapping_historical_step_numbers'],
        state['beyond_historical_step_numbers'], state['historical_input_checks'], state['complete_loss_records']) != (
        count, count, min(count, 120), max(0, count - 120), min(count, 120), count): raise RuntimeError('Actual independent trajectory accounting differs')
    if state['monitor_global_steps'] != list(range(START, step + 1, 60)): raise RuntimeError('Missing actual distinct monitors')
    protective = ('original_S0_extreme_guard', 'two_consecutive_DEV32_above3327_by5percent')
    if state['reason'] not in protective and not (step == STOP and state['reason'] == 'maximum360_updates_completed'):
        raise RuntimeError('Endpoint must be normal4287 or an actual original protective stop')
    d.validate_adams(student, discriminator, optimizer, d_optimizer, step, LRS)
    state['optimizer_cursors'] = old.actual_cursors(optimizer, d_optimizer)
    payload = {'kind': 'MC2_FM2_independent360_full', 'arm': ARM, 'global_step': step, 'epoch': 0, 'resumable': True,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'run_contract_sha256': state['contract_sha256'],
        'development': last['development'], 'development_monitor_global_step': step,
        'lineage': {'parent_checkpoint_sha256': d.PARENT_SHA, 'parent_contract_sha256': d.PARENT_CANONICAL, 'start_after_global_step': START},
        'actual_updates_this_process': count, 'historical_state_reconstruction_claimed': False,
        'stop_reason': state['reason'], 'execution_error': None, 'actual_optimizer_cursors': state['optimizer_cursors'],
        'student': student.state_dict(), 'discriminator': discriminator.state_dict(),
        'student_optimizer': optimizer.state_dict(), 'discriminator_optimizer': d_optimizer.state_dict(), 'rng_state': training.capture_rng_state()}
    path = budget.output / ARM / 'final_full.pt'; clock.active(); budget.require()
    def committed():
        state['full_snapshot'] = {'path': str(path), 'global_step': step, 'sha256': None, 'committed': True, 'resumable': True}
        state['last_full_saved_global_step'] = step; budget.committed('full', path)
    old.atomic_checkpoint_commit(payload, path, committed)
    state['full_snapshot']['sha256'] = base.sha256_file(path); path.chmod(0o444); state['full_snapshot']['frozen_readonly'] = True
    clock.active(); budget.write(path.parent / 'checkpoint_receipt.json', state)


def main(argv=None):
    started = time.monotonic(); clock = budget = log = None; locks = []; owned = False
    state, completion, stage = {}, None, OUTPUT / ARM; ledger = ForwardLedger()
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        args = arguments(argv); clock = old.StopClock(1500, started); clock.install(); previous.watchdog(1500)
        training.assert_rented_server(ROOT); cross.regular(DEV, directory=True)
        locks = training.acquire_runtime_locks(ROOT)
        if len(locks) != 3: raise RuntimeError('Require three workflow locks')
        lock_path = ROOT / 'exp/locks/text_unseen_fastvoicegrad_student_inference.lock'
        if lock_path.is_symlink(): raise RuntimeError('Inference lock symlink')
        handle = lock_path.open('a+')
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException: handle.close(); raise
        locks.append(handle)
        if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('Independent run root must be new; never retry in place')
        budget = Budget(OUTPUT); disk = budget.require(); OUTPUT.mkdir()
        log = d.StageLog(stage, budget); sys.stdout = sys.stderr = log; stage.mkdir(); owned = True
        def check(): clock.active(); budget.require()
        ledger.check = check; environment = runtime()
        if training.git_head(ROOT) != cross.HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Require unchanged clean frozen production source')
        context = prepare_context(args, check)
        mc, parent, receipt, config, development, dataset, future, future_files, monitor_files, files, controls, control_monitors, historical, aa = context
        contract = make_contract(args, parent, receipt, config, development, future, future_files, monitor_files, files, historical, aa, environment)
        contract_sha = training.canonical_json_sha256(contract); state = new_state(contract_sha)
        budget.write(OUTPUT / 'run_contract.json', contract)
        gpu = cross.gpu_gate(); device = torch.device('cuda'); check()
        diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
        inputs, batches, target, input_identity = d.prepare_inputs(parent, dataset, diffusion, device)
        check(); payload = resume.load_torch_checkpoint(d.PARENT)
        teacher, student, unused = base.load_teacher_and_student(base.TEACHER, config, device); del unused
        vocoder, vconfig, unused = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
            base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device); del unused
        discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed_d = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed = resume.load_torch_checkpoint(pair.PARENT / 'latest_full.pt')
        if fixed['global_step'] != 3107 or fixed['run_contract_sha256'] != pair.PARENT_CONTRACT: raise RuntimeError('Wrong fixedD3107')
        fixed_d.load_state_dict(fixed['discriminator'], strict=True); del fixed
        fixed_d.eval(); training.set_requires_grad(fixed_d, False)
        mean, std = base.load_mel_statistics(ROOT / 'data', device); d.imported_source_files(files, files)
        frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
        frozen_sha = {name: pair.model_sha(model) for name, model in frozen.items()}
        if frozen_sha != parent['fixed_frozen_models_sha256']: raise RuntimeError('Frozen assets changed')
        models = dict(frozen, student=student, active_D=discriminator, diffusion=diffusion)
        optimizer = torch.optim.Adam(student.parameters(), lr=LRS['student'], betas=(.5, .9))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LRS['discriminator'], betas=(.5, .9))
        restored = restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, parent, receipt, device); del payload
        state.update(restore_completed=True, restore_identity=restored, optimizer_cursors=old.actual_cursors(optimizer, d_optimizer),
            status='running', initial_disk=disk, initial_gpu=gpu)
        ledger.install(student, teacher, vocoder, discriminator, fixed_d)
        def unchanged():
            pair.assert_frozen(frozen, frozen_sha)
            if input_identity() != parent['input_identities'] or cross.flags() != cross.ORIGINAL_FLAGS: raise RuntimeError('Fixed inputs or flags changed')
        def evaluate(step):
            if step - START not in MONITOR_UPDATES: raise RuntimeError('Undeclared monitoring cadence')
            check(); before = dict(ledger.counts)
            with previous.monitoring_state(models, device):
                entry = pair.monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, step)
                entry['updates'] = entry['trial_updates'] = step - START
                entry['uniform64'] = previous.uniform64_monitor(student, teacher, diffusion, batches, parent['monitor64_plan'], config, device, check)
            if ledger.delta(before) != MONITOR_COUNTS: raise RuntimeError('Wrong unchanged monitor calls')
            entry['actual_forward_counts'] = ledger.delta(before); entry['observation_role'] = 'current_independent_trajectory'
            entry['historical_same_step_monitor_available'] = step in control_monitors
            entry['historical_computed_comparison'] = 'descriptive_only_not_a_gate'
            unchanged(); return entry
        initial = evaluate(START); state['last_monitor'] = initial
        # 该监测输出完整保留；不要求它与历史浮点或targetSHA相同。
        budget.write(stage / 'monitor.jsonl', initial, append=True)
        def step_call(planned):
            step = planned['global_step']; batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
            training.seed_everything(planned['noise_seed']); identity = mc.original_input_identity(batch, device, config); mc_receipt = {}
            check(); before = dict(ledger.counts)
            def call():
                with deadline_steps(optimizer, d_optimizer, check):
                    return mc.train_batch_mc2(batch, teacher, student, vocoder, discriminator, diffusion, optimizer, d_optimizer,
                        mean, std, device, config, global_step=step, mc_receipt=mc_receipt)
            losses = complete_update(state, step, optimizer, d_optimizer, call)
            row = dict(losses, global_step=step, updates=step - START, indices=planned['indices'], crop_seed=planned['crop_seed'],
                noise_seed=planned['noise_seed'], input_identity=identity, mc=mc_receipt,
                actual_optimizer_cursors=state['optimizer_cursors'], actual_forward_counts=ledger.delta(before), observation_role='independent_trajectory')
            state['unpublished_step_record'] = row; state.pop('unpublished_returned_losses', None)
            for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
                if mc_receipt[key] != identity[key]: raise RuntimeError('Original RNG preview mismatch')
            if row['actual_forward_counts'] != TRAIN_COUNTS: raise RuntimeError('Wrong complete train calls')
            previous.adam_counters(optimizer, d_optimizer, step)
            if [optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr']] != list(LRS.values()): raise RuntimeError('Learning-rate drift')
            if step <= OVERLAP_STOP:
                row['historical_input_comparison'] = historical_input_check(row, controls[step]); state['historical_input_checks'] = step - START
            return row
        result = run_updates(state, future, step_call, evaluate,
            lambda name, value, append=True: budget.write(stage / name, value, append), check, initial)
        state.update(reason=result['reason'], loss_windows=result['windows'], monitor_global_steps=[r['global_step'] for r in result['monitors']],
            status='completed360' if result['reason'] == 'maximum360_updates_completed' else 'protective_stop', phase='saving')
        budget.write(stage / 'loss_trend.json', loss_trend(result['rows']))
        unchanged()
        if state['new_updates'] > 0: save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock)
        else: state['no_checkpoint_reason'] = 'initial_guard_before_any_new_update'
        d.imported_source_files(files, files); d.check_files(files, check); unchanged()
        if runtime() != environment: raise RuntimeError('Measured runtime changed inside the trajectory')
        if training.canonical_json_sha256(cross.read_json(OUTPUT / 'run_contract.json')) != contract_sha: raise RuntimeError('New contract changed')
        actual_updates = state['new_updates']; monitor_count = len(result['monitors'])
        expected_counts = {k: actual_updates * TRAIN_COUNTS[k] + monitor_count * MONITOR_COUNTS[k] for k in FORWARD_LIMITS}
        if ledger.counts != expected_counts: raise RuntimeError('Actual full or protective endpoint call counts differ')
        completion = {'status': state['status'], 'cleanup_verified': True, 'state': dict(state), 'contract_sha256': contract_sha,
            'actual_forward_counts': dict(ledger.counts), 'elapsed_seconds': time.monotonic() - started,
            'actual_updates_this_process': actual_updates, 'historical_state_reconstruction_claimed': False,
            'no_evaluation_or_archive_executed': True, 'automatic_next_stage_authorized': False}
        budget.write(stage / 'trial_completion.json', dict(completion, cleanup_verified=False, status='pending_cleanup')); check()
    except BaseException as error:
        if clock: clock.request('technical_failure')
        if budget: budget.cancel_future_on_fatal_abort(repr(error))
        state.update(status='technical_failure', execution_error=repr(error))
        if owned:
            try: budget.write(stage / 'failure.json', failure_record({'status': 'technical_failure', 'state': state, 'actual_forward_counts': ledger.counts, 'error': repr(error)}))
            except BaseException as failed: saved_stderr.write('Failure receipt unavailable: %r; original=%r\n' % (failed, error))
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = old.cleanup_resources(ledger, clock, log, locks)
        if errors and budget: budget.cancel_future_on_fatal_abort(repr(errors))
        if errors and owned:
            try: budget.write(stage / 'cleanup_failure.json', failure_record({'status': 'technical_failure', 'errors': errors, 'state': state}))
            except BaseException as error: saved_stderr.write('Cleanup receipt unavailable: %r\n' % error)
        if errors and original_error is None: raise RuntimeError('Independent360 cleanup failed: ' + repr(errors))
        if original_error is None and not errors and completion is not None:
            try: clock.active(); budget.publish_success(stage / 'completion.json', completion)
            except BaseException as error:
                budget.cancel_future_on_fatal_abort(repr(error))
                try: budget.write(stage / 'failure.json', failure_record({'status': 'technical_failure', 'state': state, 'error': repr(error), 'phase': 'terminal_publication'}))
                except BaseException as failed: saved_stderr.write('Terminal failure receipt unavailable: %r\n' % failed)
                raise


if __name__ == '__main__': main()
