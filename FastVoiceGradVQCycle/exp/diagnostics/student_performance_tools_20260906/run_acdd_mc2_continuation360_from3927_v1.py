"""恢复3927，重放已有120步，再观察新增240步；不自动评估、长训或清理。"""

import argparse
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
DLR_SHA = 'e2eb8d53f18999874dab447d1b124b6c34bd0a0d056d0323df59b320e14ff821'
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Run only on the registered Matpool rental')
if hashlib.sha256((TOOLS / 'run_acdd_mc2_dlr_paired120_from3927_v1.py').read_bytes()).hexdigest() != DLR_SHA:
    raise RuntimeError('Frozen3927 restoration primitives changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_mc2_dlr_paired120_from3927_v1 as d

old, previous, cross, pair, base = d.old, d.previous, d.cross, d.pair, d.base
training, resume, probe = d.training, d.resume, d.probe
DEV, START, REPLAY_STOP, STOP, UPDATES = d.DEV, 3927, 4047, 4287, 360
OUTPUT = DEV / 'acdd_mc2_continuation360_from3927_v1'
ARM = 'candidate_mc2'
PROTOCOL = DEV / 'ACDD3927_MC2_CONTINUATION360_PROTOCOL_CN.md'
HELPER = 'acdd_mc2_train_batch_3928_4287_v1.py'
HELPER_TESTS = 'test_acdd_mc2_train_batch_3928_4287_v1.py'
LRS = {'student': .00005, 'discriminator': .0002}
META_LIMIT = 32 * 1024**2
INITIAL_REQUIRED_BYTES = 4849283072
MONITOR_UPDATES = tuple(range(0, UPDATES + 1, 60))
WINDOW_STARTS = (3928, 4048, 4168)
FORWARD_LIMITS = {'teacher': 867, 'student': 748, 'vocoder': 1094,
                  'active_D_branches': 11520, 'fixed_D_branches': 112}
TRAIN_COUNTS = {'teacher': 2, 'student': 2, 'vocoder': 3, 'active_D_branches': 32, 'fixed_D_branches': 0}
MONITOR_COUNTS = {'teacher': 21, 'student': 4, 'vocoder': 2, 'active_D_branches': 0, 'fixed_D_branches': 16}
CONTROL_CONTRACT_SHA = 'c7b403ff9d84257bea2f80a9f8523d90bc94a30edbf1811de3c9f0fcaa5d8eb4'
CONTROL_CANONICAL = '9aba349cdcf14b39cc224abf02fb1a95107fb8ff6a9fad8bb0d7da13137ce551'
CONTROL_ARCHIVE_SHA = '1f3f69658dd2498a2b577295ec2925ae98c05f66d8fbfc2d261a11edc883afc9'
CONTROL_COMPLETION_SHA = '221f9aa42d24c2ff3e94a753dba920042e38034e5b0a0bd21fe43f11b41f376c'
EVALUATION = {'output_root': str(DEV / 'evaluation_mc2_continuation360_endpoint_v1'),
              'label': 'student_mc2_continuation360_endpoint_v1',
              'future_metrics_only': ['cer', 'dnsmos', 'utmos'], 'executed_by_this_driver': False}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    for name in ('driver', 'tests', 'helper', 'helper-tests', 'protocol'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args(argv)
    if not args.run_approved: raise RuntimeError('Explicit bounded360 approval required')
    for name in ('driver', 'tests', 'helper', 'helper_tests', 'protocol'):
        value = getattr(args, name + '_sha256')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Require all five reviewed full SHA256 values')
    return args


class Budget(d.Budget):
    def __init__(self, output):
        self.output, self.arm, self.cancelled = Path(output), ARM, {}
        self.pending = {'full': d.FULL_BYTES + previous.MODEL_SLACK, 'later_DEV90': d.EVAL_BYTES}

    def require(self, extra_metadata=0):
        # 全部360条原始记录、契约、窗口和日志共用32MiB，不截断观察记录。
        used = self.metadata_used()
        if used + extra_metadata > META_LIMIT: raise pair.InsufficientDisk('Complete metadata exceeds32MiB')
        need = previous.RESERVE + previous.DIR_RESERVE + sum(self.pending.values()) + META_LIMIT - used
        free = pair.available_bytes(self.output.parent)
        if free < need: raise pair.InsufficientDisk('available=%d required=%d' % (free, need))
        return {'available_bytes': free, 'required_bytes': need, 'pending_bytes': dict(self.pending), 'metadata_bytes': used}


class ForwardLedger(old.ForwardLedger):
    def __init__(self): self.counts = dict.fromkeys(FORWARD_LIMITS, 0); self.handles = []
    def before(self, name):
        def hook(module, arguments):
            if self.counts[name] >= FORWARD_LIMITS[name]: raise RuntimeError('Bounded360 forward limit reached: ' + name)
            self.counts[name] += 1
        return hook


def new_state(contract_sha):
    state = d.new_state(d.ARMS[0], contract_sha)
    state.update(arm=ARM, effective_optimizer_lrs=dict(LRS), replay_updates=0, novel_updates=0,
                 replay_verified_steps=0, replay_monitor_verified_steps=[], actual_updates_this_process=0)
    return state


def restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, contract, receipt, device):
    # 原control角色不改lr；恢复后每个模型、Adam、RNG内容都应原样保留。
    restored = d.restore_parent(payload, student, discriminator, optimizer, d_optimizer,
                                config, contract, receipt, device, d.ARMS[0])
    if restored['parent_restore'] != restored['role_lr_applied']:
        raise RuntimeError('Continuation may not change any restored model, optimizer, RNG or flags')
    return {'parent_restore': restored['parent_restore'], 'configuration_changed': False,
            'all_models_optimizers_rng_restored_without_change': True}


def complete_update(state, step, optimizer, d_optimizer, call):
    # 只扩大准入范围；仍以两个真实Adam.step的正常返回和所有参数游标证明完整更新。
    if step != state['completed_global_step'] + 1 or not START < step <= STOP:
        raise RuntimeError('Require exactly the next3928..4287 absolute step')
    if any(x['uniform_step'] != step - 1 for x in old.actual_cursors(optimizer, d_optimizer).values()):
        raise RuntimeError('Entering Adam counters differ from the last complete step')
    state.update(attempted_global_step=step, partial_batch_may_have_updated_D=True, phase='D_S_update')
    returned = False
    try:
        with old.observe_adam_step_returns(state, optimizer, d_optimizer): losses = call()
        returned = True
    finally:
        try:
            cursors = old.actual_cursors(optimizer, d_optimizer); state['optimizer_cursors'] = cursors
            if all(x['uniform_step'] == step for x in cursors.values()) and state['adam_step_returns'] == {'student': 1, 'discriminator': 1}:
                count = step - START
                state.update(completed_global_step=step, new_updates=count, actual_updates_this_process=count,
                    replay_updates=min(count, 120), novel_updates=max(0, count - 120),
                    partial_batch_may_have_updated_D=False, exception_after_both_adam_steps=not returned)
        except BaseException as error: state['cursor_inspection_error'] = repr(error)
    if state['partial_batch_may_have_updated_D']: raise RuntimeError('Missing complete D then S update proof')
    state['unpublished_returned_losses'] = {'global_step': step, 'losses': losses}
    pair.finite_losses(losses)
    return losses


def future_plan(development):
    planner = resume.GlobalBatchPlan(development['retained_train_indices'])
    if len(development['retained_train_indices']) != 39335 or planner.batches_per_pass != 1229:
        raise RuntimeError('Original retained absolute batch plan changed')
    rows = [{'global_step': step, 'indices': planner.for_step(step), 'crop_seed': 101234 + step,
             'noise_seed': 201234 + step} for step in range(START + 1, STOP + 1)]
    if any(set(row['indices']) & set(development['excluded_train_indices']) for row in rows):
        raise RuntimeError('Original52 development exclusions leaked into training')
    return rows


def replay_check(row, control):
    if not START < row['global_step'] <= REPLAY_STOP: raise RuntimeError('Replay proof is limited to the original120 steps')
    for key in ('global_step', 'indices', 'crop_seed', 'noise_seed', 'input_identity'):
        if row[key] != control[key]: raise RuntimeError('Replay input identity differs: ' + key)
    # 全MC收据含两份teacher_target SHA、逐句权重/L1、两次噪声和t。
    # 身份严格一致；浮点沿用原1e-7+1e-6*abs(expected)，不为重放临时放宽。
    d.same_tree(row['mc'], control['mc'])
    d.same_tree({k: row[k] for k in pair.LOSS_KEYS}, {k: control[k] for k in pair.LOSS_KEYS})
    for key in ('actual_optimizer_cursors', 'actual_forward_counts'):
        if row[key] != control[key]: raise RuntimeError('Replay actual cursor or forward count differs')


def replay_monitor_check(actual, expected):
    if actual['global_step'] != expected['global_step'] or actual['global_step'] not in (3927, 3987, 4047):
        raise RuntimeError('Only the three original monitor points may be replayed')
    for key in ('fixed_train', 'development', 'uniform64'): d.same_tree(actual[key], expected[key])


def window120(rows):
    if len(rows) != 120 or rows[0]['global_step'] not in WINDOW_STARTS:
        raise RuntimeError('Only three predeclared complete120-step windows')
    first = rows[0]['global_step']
    if [r['global_step'] for r in rows] != list(range(first, first + 120)): raise RuntimeError('Nonconsecutive window')
    for row in rows: pair.finite_losses(row)
    return {'first_global_step': first, 'global_step': rows[-1]['global_step'], 'count': 120,
        'phase': 'historical_reconstruction_replay' if first == 3928 else 'novel_training',
        'means': {k: math.fsum(r[k] for r in rows) / 120 for k in pair.LOSS_KEYS},
        'MC2_dist_means': {k: math.fsum(r['mc'][k] for r in rows) / 120 for k in ('dist1', 'dist2')},
        'first_last_difference': {k: rows[-1][k] - rows[0][k] for k in pair.LOSS_KEYS}, 'used_as_stop_gate': False}


def novel_trend(rows):
    # 主要新趋势只比较后240步中的两个完整120窗，前面的重复观察另列。
    if len(rows) != UPDATES: return {'available': False, 'completed_records': len(rows), 'reason': 'requires_all360_steps'}
    windows = [window120(rows[offset:offset + 120]) for offset in (0, 120, 240)]
    return {'available': True, 'replay_window': windows[0], 'first_new_window': windows[1], 'second_new_window': windows[2],
        'second_new_minus_first_new': {k: windows[2]['means'][k] - windows[1]['means'][k] for k in pair.LOSS_KEYS},
        'used_as_stop_gate': False, 'replay_is_not_new_independent_evidence': True}


def run_updates(state, future, step_call, evaluate, publish, check, initial, control_monitors):
    if [r['global_step'] for r in future] != list(range(START + 1, STOP + 1)) or initial['global_step'] != START:
        raise RuntimeError('Wrong prospective360 plan')
    replay_monitor_check(initial, control_monitors[START])
    if old.guard_reasons([initial], d.cont.BASELINE3327): raise RuntimeError('Original3927 guard mismatch')
    state['replay_monitor_verified_steps'] = [START]
    rows, windows, monitors, reason = [], [], [initial], 'maximum360_updates_completed'
    for planned in future:
        check(); row = step_call(planned); state['unpublished_step_record'] = row
        publish('step_losses.jsonl', row); state.pop('unpublished_step_record', None); rows.append(row)
        state['complete_loss_records'] = len(rows)
        if len(rows) % 120 == 0:
            window = window120(rows[-120:]); windows.append(window); publish('loss_windows.jsonl', window)
        if len(rows) % 60 == 0:
            entry = evaluate(planned['global_step']); state['last_monitor'] = entry
            # 即使身份核对失败也先保存本次真实监测，不能丢弃失配证据。
            monitors.append(entry); publish('monitor.jsonl', entry)
            if entry['global_step'] <= REPLAY_STOP:
                replay_monitor_check(entry, control_monitors[entry['global_step']])
                state['replay_monitor_verified_steps'].append(entry['global_step'])
            guards = old.guard_reasons(monitors[1:], d.cont.BASELINE3327)
            state['guard_reasons'] = guards
            publish('guard_checks.jsonl', {'global_step': entry['global_step'], 'stop_reasons': guards,
                'distinct_global_steps_used': [r['global_step'] for r in monitors[1:]],
                'replayed_points_are_not_new_independent_observations': True})
            if guards:
                if entry['global_step'] <= REPLAY_STOP: raise RuntimeError('Completed historical control now violates its original guard')
                reason = guards[0]; break
        publish('status.json', state, False)
    return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': reason}


def control_evidence(parent, future, files, check):
    folder = d.OUTPUT / d.ARMS[0]
    paths = {str(d.OUTPUT / 'run_contract.json'): CONTROL_CONTRACT_SHA,
             str(d.OUTPUT / d.ARCHIVE_NAME): CONTROL_ARCHIVE_SHA,
             str(folder / 'completion.json'): CONTROL_COMPLETION_SHA}
    d.check_files(paths, check)
    joint = cross.read_json(d.OUTPUT / 'run_contract.json'); done = cross.read_json(folder / 'completion.json')
    archive = cross.read_json(d.OUTPUT / d.ARCHIVE_NAME)
    if (training.canonical_json_sha256(joint), joint['parent_checkpoint_sha256'], joint['configuration'],
        joint['future_batches'], joint['effective_optimizer_lrs_by_arm'][d.ARMS[0]]) != (
        CONTROL_CANONICAL, d.PARENT_SHA, parent['configuration'], future[:120], LRS):
        raise RuntimeError('Original control is not the same3927/config/lr/120 absolute batches')
    state = done['state']
    if (done['status'], done['cleanup_verified'], done['contract_sha256'], state['completed_global_step'],
        state['partial_batch_may_have_updated_D'], state['restore_completed'], state['new_updates']) != (
        'completed120', True, CONTROL_CANONICAL, REPLAY_STOP, False, True, 120):
        raise RuntimeError('Historical control lacks complete120 and cleanup proof')
    if done['actual_forward_counts'] != d.FORWARD_LIMITS: raise RuntimeError('Original control forward counts differ')
    if any((folder / name).exists() for name in ('failure.json', 'cleanup_failure.json')):
        raise RuntimeError('Historical control contains failure evidence')
    if (archive['joint_contract_file_sha256'], archive['control_completion_file_sha256']) != (CONTROL_CONTRACT_SHA, CONTROL_COMPLETION_SHA):
        raise RuntimeError('Retained archive receipt belongs to another control')
    # 只验已存在的小型归档收据；不读/重验已归档的S或90WAV，不递归旧runtime_files。
    for name in ('step_losses.jsonl', 'loss_windows.jsonl', 'monitor.jsonl'):
        path = folder / name; check(); cross.regular(path); paths[str(path)] = base.sha256_file(path)
    rows = [json.loads(line) for line in (folder / 'step_losses.jsonl').read_text().splitlines()]
    monitors = [json.loads(line) for line in (folder / 'monitor.jsonl').read_text().splitlines()]
    windows = [json.loads(line) for line in (folder / 'loss_windows.jsonl').read_text().splitlines()]
    if [r['global_step'] for r in rows] != list(range(3928, 4048)) or [r['global_step'] for r in monitors] != [3927, 3987, 4047]:
        raise RuntimeError('Historical raw rows or monitor sequence incomplete')
    for planned, row in zip(future[:120], rows):
        for key in ('global_step', 'indices', 'crop_seed', 'noise_seed'):
            if row[key] != planned[key]: raise RuntimeError('Historical raw plan differs')
        pair.finite_losses(row)
    if len(windows) != 2: raise RuntimeError('Unexpected historical windows')
    for index in (0, 1): d.same_tree(d.window60(rows[index * 60:(index + 1) * 60]), windows[index])
    d.same_tree(windows, state['loss_windows']); d.same_tree(monitors[-1], state['last_monitor'])
    cross.merge_hashes(files, paths)
    return {r['global_step']: r for r in rows}, {r['global_step']: r for r in monitors}, {
        'role': 'post_registered_reconstruction_of_already_observed_control', 'files_sha256': paths,
        'original_contract_canonical_sha256': CONTROL_CANONICAL, 'control_replayed_updates': 120,
        'control_rescored': False, 'no_archived_model_or_wav_is_a_runtime_dependency': True,
        'raw_log_SHA_binding': 'new_preexecution_binding_checked_against_retained_completion_and_windows'}


def prepare_context(args, check):
    omit = {str(DEV / 'acdd_mc2_fm_cross_3447_3927_v1/results.json')}
    files = {p: sha for p, sha in d.FIXED_PINS.items() if p not in omit}
    files.update(d.NEW_SOURCE_PINS)
    files.update({str(TOOLS / 'run_acdd_mc2_dlr_paired120_from3927_v1.py'): DLR_SHA,
        str(TOOLS / 'acdd_mc2_train_batch_v1.py'): 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16',
        str(Path(__file__).resolve()): args.driver_sha256,
        str(TOOLS / 'test_run_acdd_mc2_continuation360_from3927_v1.py'): args.tests_sha256,
        str(TOOLS / HELPER): args.helper_sha256, str(TOOLS / HELPER_TESTS): args.helper_tests_sha256,
        str(PROTOCOL): args.protocol_sha256})
    d.check_files(files, check); d.validate_supplemental_source_pins()
    mc = importlib.import_module(HELPER[:-3])
    if (mc.START, mc.STOP, mc.MC2_SEED) != (START, STOP, 74000000): raise RuntimeError('Wrong newMC2 range')
    parent = cross.read_json(d.PARENT_RUN / 'run_contract.json')
    receipt = d.validate_parent_metadata(parent, cross.read_json(d.PARENT_RUN / 'completion.json'))
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if parent['configuration'] != config.as_dict() or (config.lambda_fm, config.lambda_dist) != (2., 45.):
        raise RuntimeError('Keep the actual originalFM2/dist45 configuration')
    development, dependencies = resume.load_development(DEV / 'development_contract.json')
    if parent['training_indices'] != development['retained_train_indices'] or parent['excluded_train_indices'] != development['excluded_train_indices']:
        raise RuntimeError('Original training data and52 exclusions changed')
    cross.merge_hashes(files, dependencies)
    cross.merge_hashes(files, d.imported_source_files(parent['file_sha256'], files))
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
    if len(monitor_files) != 192 or any(parent['file_sha256'].get(p) != sha for p, sha in monitor_files.items()):
        raise RuntimeError('Original192 fixed64 features changed')
    controls, control_monitors, evidence = control_evidence(parent, future, files, check)
    d.check_files(files, check)
    return mc, parent, receipt, config, development, dataset, future, future_files, monitor_files, files, controls, control_monitors, evidence


def make_contract(args, parent, receipt, config, development, future, future_files, monitor_files, files, evidence):
    return {'kind': 'MC2_FM2_bounded360_replay120_then_novel240', 'protocol_sha256': args.protocol_sha256,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS,
        'paper_reference_coefficients': {'lambda_fm': 2., 'lambda_dist': 45.},
        'disclosed_extensions': {'student_optimizer_lr': .00005, 'MC_samples': 2,
            'score_target_interpretation': 'clean_x0_sqrt_alpha_bar', 'not_claimed_exact_paper_reproduction': True},
        'parent_checkpoint': str(d.PARENT), 'parent_checkpoint_sha256': d.PARENT_SHA,
        'parent_contract_sha256': d.PARENT_CANONICAL, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'start_after_global_step': START, 'replay_stop_global_step': REPLAY_STOP, 'stop_global_step': STOP,
        'maximum_updates': UPDATES, 'maximum_replayed_updates': 120, 'maximum_novel_updates': 240,
        'future_batches': future, 'future_training_feature_sha256': future_files, 'fixed64_feature_sha256': monitor_files,
        'training_indices': development['retained_train_indices'], 'excluded_train_indices': development['excluded_train_indices'],
        'input_identities': parent['input_identities'], 'monitor64_plan': parent['monitor64_plan'],
        'baseline3927_monitor': receipt['last_monitor'], 'baseline3327_development': d.cont.BASELINE3327,
        'original_step0_guard_anchor': d.cont.S0_ANCHOR, 'fixed_frozen_models_sha256': parent['fixed_frozen_models_sha256'],
        'monitor_updates': list(MONITOR_UPDATES), 'replay_monitor_global_steps': [3927, 3987, 4047],
        'loss_windows': [[3928, 4047], [4048, 4167], [4168, 4287]], 'loss_keys': list(pair.LOSS_KEYS),
        'primary_new_trend': 'second_new120_minus_first_new120; replay window reported separately; no scalar loss gate',
        'historical_control': evidence, 'runtime_files': files, 'forward_limits': FORWARD_LIMITS,
        'storage': {'initial_required_bytes': INITIAL_REQUIRED_BYTES, 'full_upper_bytes': d.FULL_BYTES,
            'save_slack': previous.MODEL_SLACK, 'system_reserve': previous.RESERVE, 'directory_reserve': previous.DIR_RESERVE,
            'metadata_limit': META_LIMIT, 'later_DEV90_bytes': d.EVAL_BYTES},
        'evaluation': EVALUATION, 'formal_gate_only': {'UTMOS_min': 3.9812097230664008, 'CER_percent_max': 2.109892438933927,
            'DNSMOS_P808_strictly_greater_than': 3.77, 'SVA_gate': False, 'OVRL_gate': False},
        'compute_seconds': 1500, 'cleanup_only_seconds': 90, 'original_flags': cross.ORIGINAL_FLAGS,
        'no_automatic_evaluation_training_retry_archive_or_deletion': True, 'formal3327_remains_stopped': True}


def save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock):
    if (not state['restore_completed'] or state['partial_batch_may_have_updated_D'] or state.get('execution_error')
            or state.get('exception_after_both_adam_steps') or state.get('cursor_inspection_error')):
        raise RuntimeError('Never save a technical failure or incomplete update as an endpoint')
    step = state['completed_global_step']; last = state['last_monitor']
    if step not in (4107, 4167, 4227, 4287) or last['global_step'] != step or state['arm'] != ARM:
        raise RuntimeError('Only a complete monitored NEW endpoint may be saved; no3927/3987/4047 copy')
    if config.as_dict() != base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0).as_dict():
        raise RuntimeError('Save only the real unchangedFM2 configuration')
    if state['replay_verified_steps'] != 120 or state['replay_monitor_verified_steps'] != [3927, 3987, 4047]:
        raise RuntimeError('Do not publish a descendant without complete replay proof')
    if (state['actual_updates_this_process'], state['replay_updates'], state['novel_updates']) != (step - START, 120, step - REPLAY_STOP):
        raise RuntimeError('Actual replay/new update accounting differs from the endpoint')
    protective_reasons = ('original_S0_extreme_guard', 'two_consecutive_DEV32_above3327_by5percent')
    if state['reason'] not in protective_reasons and not (step == STOP and state['reason'] == 'maximum360_updates_completed'):
        raise RuntimeError('Endpoint reason must identify the real protection stop or complete360 limit')
    d.validate_adams(student, discriminator, optimizer, d_optimizer, step, LRS)
    state['optimizer_cursors'] = old.actual_cursors(optimizer, d_optimizer)
    payload = {'kind': 'MC2_FM2_continuation360_full', 'arm': ARM, 'global_step': step, 'epoch': 0, 'resumable': True,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'run_contract_sha256': state['contract_sha256'],
        'development': last['development'], 'development_monitor_global_step': step,
        'lineage': {'parent_checkpoint_sha256': d.PARENT_SHA, 'parent_contract_sha256': d.PARENT_CANONICAL, 'start_after_global_step': START},
        'replay_updates': state['replay_updates'], 'novel_updates': state['novel_updates'],
        'stop_reason': state['reason'], 'execution_error': None, 'actual_optimizer_cursors': state['optimizer_cursors'],
        'student': student.state_dict(), 'discriminator': discriminator.state_dict(),
        'student_optimizer': optimizer.state_dict(), 'discriminator_optimizer': d_optimizer.state_dict(),
        'rng_state': training.capture_rng_state()}
    path = budget.output / ARM / 'final_full.pt'; clock.active(); budget.require()
    def committed():
        state['full_snapshot'] = {'path': str(path), 'global_step': step, 'sha256': None, 'committed': True, 'resumable': True}
        state['last_full_saved_global_step'] = step; budget.committed('full', path)
    old.atomic_checkpoint_commit(payload, path, committed)
    state['full_snapshot']['sha256'] = base.sha256_file(path); path.chmod(0o444)
    state['full_snapshot']['frozen_readonly'] = True
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
        if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('Continuation root must be entirely new')
        budget = Budget(OUTPUT); disk = budget.require(); OUTPUT.mkdir()
        log = d.StageLog(stage, budget); sys.stdout = sys.stderr = log; stage.mkdir(); owned = True
        def check(): clock.active(); budget.require()
        if training.git_head(ROOT) != cross.HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Require unchanged clean frozen production source')
        if ((platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6')
            or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or cross.flags() != cross.ORIGINAL_FLAGS):
            raise RuntimeError('Keep the actual runtime and original precision')
        context = prepare_context(args, check)
        mc, parent, receipt, config, development, dataset, future, future_files, monitor_files, files, controls, control_monitors, evidence = context
        contract = make_contract(args, parent, receipt, config, development, future, future_files, monitor_files, files, evidence)
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
        if fixed['global_step'] != 3107 or fixed['run_contract_sha256'] != pair.PARENT_CONTRACT: raise RuntimeError('Wrong original fixedD3107')
        fixed_d.load_state_dict(fixed['discriminator'], strict=True); del fixed
        fixed_d.eval(); training.set_requires_grad(fixed_d, False)
        mean, std = base.load_mel_statistics(ROOT / 'data', device)
        d.imported_source_files(files, files)  # 后加载的网络源码也须事先pin，不能现场放宽清单。
        frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
        frozen_sha = {name: pair.model_sha(model) for name, model in frozen.items()}
        if frozen_sha != parent['fixed_frozen_models_sha256']: raise RuntimeError('Original frozen assets changed')
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
            if ledger.delta(before) != MONITOR_COUNTS: raise RuntimeError('Wrong monitor forward count')
            entry['actual_forward_counts'] = ledger.delta(before)
            entry['observation_role'] = 'historical_replay' if step <= REPLAY_STOP else 'new_observation'
            unchanged(); return entry
        initial = evaluate(START)
        for key in ('fixed_train', 'development', 'uniform64'): d.same_tree(initial[key], receipt['last_monitor'][key])
        state['last_monitor'] = initial; budget.write(stage / 'monitor.jsonl', initial, append=True)
        def step_call(planned):
            step = planned['global_step']; batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
            training.seed_everything(planned['noise_seed']); identity = mc.original_input_identity(batch, device, config); mc_receipt = {}
            check(); before = dict(ledger.counts)
            losses = complete_update(state, step, optimizer, d_optimizer, lambda: mc.train_batch_mc2(
                batch, teacher, student, vocoder, discriminator, diffusion, optimizer, d_optimizer, mean, std, device, config,
                global_step=step, mc_receipt=mc_receipt))
            row = dict(losses, global_step=step, updates=step - START, indices=planned['indices'], crop_seed=planned['crop_seed'],
                noise_seed=planned['noise_seed'], input_identity=identity, mc=mc_receipt,
                actual_optimizer_cursors=state['optimizer_cursors'], actual_forward_counts=ledger.delta(before),
                observation_role='historical_reconstruction_replay' if step <= REPLAY_STOP else 'novel_training')
            state['unpublished_step_record'] = row; state.pop('unpublished_returned_losses', None)
            for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
                if mc_receipt[key] != identity[key]: raise RuntimeError('Original RNG preview mismatch')
            if row['actual_forward_counts'] != TRAIN_COUNTS: raise RuntimeError('Wrong complete train forward count')
            previous.adam_counters(optimizer, d_optimizer, step)
            if [optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr']] != list(LRS.values()): raise RuntimeError('Learning-rate drift')
            if step <= REPLAY_STOP:
                replay_check(row, controls[step]); state['replay_verified_steps'] = step - START
                row['historical_replay_verified'] = True
            return row
        result = run_updates(state, future, step_call, evaluate,
            lambda name, value, append=True: budget.write(stage / name, value, append), check, initial, control_monitors)
        state.update(reason=result['reason'], loss_windows=result['windows'], monitor_global_steps=[r['global_step'] for r in result['monitors']],
            status='completed360' if result['reason'] == 'maximum360_updates_completed' else 'protective_stop', phase='saving')
        budget.write(stage / 'novel_loss_trend.json', novel_trend(result['rows']))
        unchanged(); save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock)
        d.imported_source_files(files, files); d.check_files(files, check); unchanged()
        if training.canonical_json_sha256(cross.read_json(OUTPUT / 'run_contract.json')) != contract_sha: raise RuntimeError('New contract changed')
        if state['new_updates'] == UPDATES and ledger.counts != FORWARD_LIMITS: raise RuntimeError('Full360 actual counts differ')
        completion = {'status': state['status'], 'cleanup_verified': True, 'state': dict(state), 'contract_sha256': contract_sha,
            'actual_forward_counts': dict(ledger.counts), 'elapsed_seconds': time.monotonic() - started,
            'actual_replay_steps': state['replay_updates'], 'actual_novel_steps': state['novel_updates'],
            'replay_is_not_new_independent_evidence': True, 'no_evaluation_or_archive_executed': True,
            'automatic_next_stage_authorized': False}
        budget.write(stage / 'trial_completion.json', dict(completion, cleanup_verified=False, status='pending_cleanup')); check()
    except BaseException as error:
        if clock: clock.request('technical_failure')
        if budget: budget.cancel_future_on_fatal_abort(repr(error))
        state.update(status='technical_failure', execution_error=repr(error))
        if owned:
            try: budget.write(stage / 'failure.json', {'status': 'technical_failure', 'state': state, 'actual_forward_counts': ledger.counts, 'error': repr(error)})
            except BaseException as failed: saved_stderr.write('Failure receipt unavailable: %r; original=%r\n' % (failed, error))
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = old.cleanup_resources(ledger, clock, log, locks)
        if errors and budget: budget.cancel_future_on_fatal_abort(repr(errors))
        if errors and owned:
            try: budget.write(stage / 'cleanup_failure.json', {'status': 'technical_failure', 'errors': errors, 'state': state})
            except BaseException as error: saved_stderr.write('Cleanup receipt unavailable: %r\n' % error)
        if errors and original_error is None: raise RuntimeError('Continuation cleanup failed: ' + repr(errors))
        if original_error is None and not errors and completion is not None:
            try: clock.active(); budget.publish_success(stage / 'completion.json', completion)
            except BaseException as error:
                budget.cancel_future_on_fatal_abort(repr(error))
                try: budget.write(stage / 'failure.json', {'status': 'technical_failure', 'state': state, 'error': repr(error)})
                except BaseException: pass
                raise


if __name__ == '__main__': main()
