"""同一完整3927的D学习率配对，分阶段各至多120步；默认拒绝训练，不负责评估或删除。"""

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
import subprocess
from pathlib import Path, PureWindowsPath

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Only the registered Matpool rental may execute this module')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
CONT_NAME = 'run_acdd_mc2_continuation480_v1.py'
CONT_SHA = '8208e396b035b79c39d5094d68e965dd319d34d31d1aa673b80da4d03be04c67'
if hashlib.sha256((TOOLS / CONT_NAME).read_bytes()).hexdigest() != CONT_SHA: raise RuntimeError('Frozen continuation primitive changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_mc2_continuation480_v1 as cont

old, previous, cross, pair, base = cont.old, cont.previous, cont.cross, cont.pair, cont.base
training, resume, probe = cont.training, cont.resume, cont.probe
DEV, START, STOP, UPDATES = cont.DEV, 3927, 4047, 120
OUTPUT = DEV / 'acdd_mc2_dlr_paired120_from3927_v1'
PROTOCOL = DEV / 'ACDD3927_MC2_DLR_PAIRED120_PLAN_CN.md'
PROTOCOL_SHA = '24ec87e4ec1c03aa7afa984b5c857edf7db36124284a972fb8ddc3b05f98fce0'
PARENT_RUN = DEV / 'acdd_mc2_continuation480_from3447_v1'
PARENT = PARENT_RUN / 'final_full.pt'
PARENT_SHA = '6a4f5beabcd3264e7aa016b3236f98da174af396b2ce7fb0c0b5b083f8aa2630'
PARENT_CONTRACT_SHA = '4ba9228b4eb06f5315baa76778b63b20efb5bf309ac95ea0951d3694ad6c774e'
PARENT_CANONICAL = 'd3e6c7a7f98e58d02985ac5328ae86bcb143ec3db7469b73b79b7c6d93377d96'
PARENT_COMPLETION_SHA = '22c29b860351604275fae3f9784ca158e535790443806f022e3b639bb2c800e0'
PARENT_RNG_SHA = 'b2d639cf9be2a6194137d7b4d48fd5c854d7e89ad4083628a6545511846e3d7a'
HELPER_NAME = 'acdd_mc2_train_batch_3928_4047_v1.py'
HELPER_SHA = '6f04abb3838f574419986829c68196eddc72b3fe545d98a3e6a9f4162beffec3'
HELPER_TEST_NAME = 'test_acdd_mc2_train_batch_3928_4047_v1.py'
HELPER_TEST_SHA = 'fcfec04b1feb42e2b820ac047e1ee2e9af85f3f91ea634d6f461ee54763b6a8c'
ARMS = ('control_d2e4', 'candidate_d1e4')
ROLE_LRS = {ARMS[0]: {'student': .00005, 'discriminator': .0002}, ARMS[1]: {'student': .00005, 'discriminator': .0001}}
PARENT_LRS = dict(ROLE_LRS[ARMS[0]])
FORWARD_LIMITS = {'teacher': 303, 'student': 252, 'vocoder': 366, 'active_D_branches': 3840, 'fixed_D_branches': 48}
MONITOR_UPDATES = (0, 60, 120)
FULL_BYTES, STUDENT_BYTES, META_LIMIT = 1424633856, 309268480, 16 * 1024**2
EVAL_BYTES = 32 * 1024**2
METRIC_IDENTITY_SHA = 'c0ff0e76a25b5656f1b2c110a2a6664342b03d8cf244971a8590074c75a37d86'
ARCHIVE_NAME = 'control_archive_receipt.json'
EVALUATION = {arm: {'output_root': str(DEV / ('evaluation_mc2_dlr_%s_paired120_endpoint_v1' % suffix)),
                   'label': 'student_mc2_dlr_%s_paired120_endpoint_v1' % suffix}
              for arm, suffix in zip(ARMS, ('d2e4', 'd1e4'))}
NEW_SOURCE_PINS = {
    str(ROOT / 'voicegrad/__init__.py'): '59f7bb0e691e70e16afedf1afe031238b95f57172a310d0edf1517307fba40cb',
    str(ROOT / 'hifigan/__init__.py'): '21f72bdb5a5e8b938f00eb8540a0a222f3a15963cabffa5cbec020dc92bc7de7',
    str(ROOT / 'hifigan/custom/__init__.py'): '6dd3c541f5994e73a601ae5265d00d9183b99dfbc8e8acabbbcbba61f0c74d82',
    str(TOOLS / 'inspect_acdd_timestep_cross_3327_v1.py'): 'fb2b94ba1aedf1beb29a333ac629c4095286e86adc70e89fdd551a1efa015db9',
    str(TOOLS / 'inspect_acdd_teacher_target_cross_v1.py'): 'e2373af110a3e2a828a128bb76621036883b19963a7aa6b27722ef7439bce668',
    str(TOOLS / 'inspect_acdd_adam_3227_v1.py'): 'c193ad8b8a96eda8058856c4946bf207631bb8d6b6ee5940770740a1c7a8212e',
    str(TOOLS / 'inspect_acdd_d_update_3327_v1.py'): 'c14d0997173bb734ae9d461d95d1637d2b7503725bbcc474febdb04e12c2cd49',
}
FIXED_PINS = {
    str(PARENT): PARENT_SHA, str(PARENT_RUN / 'run_contract.json'): PARENT_CONTRACT_SHA,
    str(PARENT_RUN / 'completion.json'): PARENT_COMPLETION_SHA,
    str(DEV / 'MC2_CONTINUATION480_ENDPOINT_AUDIT_V1.json'): '4e197dfff54d3c4f24277306193eb84bb9d84708a0fe357d2796e777d9b096f7',
    str(cross.INPUT_FILE): '296a7b4a7c289f0d19409947cd2c4dcb54754ae5636a58af43e658bc12b17da8',
    str(cross.TARGET_FILE): '2aedd82fb269ed07c98b206c41e071c73f64555c57ed4ce3faf1eb8db12144bb',
    str(pair.PARENT / 'latest_full.pt'): 'f007dd4375bbf23a78372549e6e79b87c06e94fd38c07b5e09eb6624ae50ce44',
    str(previous.CROSS_IDENTITY): 'b9183198b8d671b24c3c6a0af0384353b11a51874a1f6357f8e77acb77031491',
    str(DEV / 'acdd_mc2_fm_cross_3447_3927_v1/results.json'): '4f442b3fd8168fe5f7eb549b20d0d3159d64958937540d5c805f9c416a655571',
    str(base.VOCODER_DIR / 'config.json'): '4b570fee140ffbda92bb217218792dca069b2d73627bc5e7bf7b1669869dd05a',
    str(base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json'): '2112690afbe6fc2f7a5325d106c8dcfdd4503d7cfb09fee7675712d34b46205e',
    str(TOOLS / 'evaluate_development_candidates.py'): '7e3b5676fd57e4ffa83bd9d705edcc690979af1ce55688b59bd69a39538797d8',
    str(TOOLS / 'evaluate_development_baselines.py'): '14142b4909583f9a79424e9a23c188a00605c8cbfd9797daa7eff04848849d21',
    str(TOOLS / 'ddim_teacher.py'): '0e3100d14e421a6f7eb1a06b1dc79d195091c2f20a35f4c3e4266e77dbd3c6cc',
    str(ROOT / 'evaluation/evaluate_metrics.py'): '9797b724c7296517a1e0b31a74dbe05e6f06454669902b351e42caea2cba42ab',
}


def check_files(files, check=lambda: None):
    # 只遍历显式运行清单；从不递归验证父合同列出的祖先或历史WAV。
    for path, expected in files.items():
        check(); cross.regular(Path(path))
        if base.sha256_file(Path(path)) != expected: raise RuntimeError('Runtime file changed: ' + path)


def same_tree(actual, expected, path='$'):
    # 数值使用方案原容差；结构、整数、身份SHA、字符串均严格一致。
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual) or abs(actual - expected) > 1e-7 + 1e-6 * abs(expected):
            raise RuntimeError('Replay numeric mismatch at ' + path)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise RuntimeError('Replay mapping mismatch at ' + path)
        for key in expected: same_tree(actual[key], expected[key], path + '.' + str(key))
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected): raise RuntimeError('Replay list mismatch at ' + path)
        for i, (a, b) in enumerate(zip(actual, expected)): same_tree(a, b, path + '[%d]' % i)
    elif type(actual) is not type(expected) or actual != expected: raise RuntimeError('Replay identity mismatch at ' + path)


class Budget(old.Budget):
    # M覆盖两臂合计，不因启动第二进程而获得第二份16MiB。
    def __init__(self, output, arm):
        if arm not in ARMS: raise RuntimeError('Unknown role')
        self.output, self.arm, self.cancelled = Path(output), arm, {}
        role, size = ('student', STUDENT_BYTES) if arm == ARMS[0] else ('full', FULL_BYTES)
        self.pending = {role: size + previous.MODEL_SLACK, 'later_DEV90': EVAL_BYTES}

    def require(self, extra_metadata=0):
        used = self.metadata_used()
        if used + extra_metadata > META_LIMIT: raise pair.InsufficientDisk('Joint16MiB exhausted; keep every observation')
        need = previous.RESERVE + previous.DIR_RESERVE + sum(self.pending.values()) + META_LIMIT - used
        free = pair.available_bytes(self.output.parent)
        if free < need: raise pair.InsufficientDisk('available=%d required=%d' % (free, need))
        return {'available_bytes': free, 'required_bytes': need, 'pending_bytes': dict(self.pending), 'metadata_bytes': used}

    def publish_success(self, path, value):
        # 所有可失败的检查在最后rename之前完成，不留下“发布成功后又退出失败”的收据。
        path = Path(path); temporary = path.with_suffix('.json.tmp')
        if path.parent != self.output / self.arm or path.name != 'completion.json': raise RuntimeError('Wrong terminal path')
        if any(p.exists() or p.is_symlink() for p in (path, temporary)): raise RuntimeError('Never overwrite terminal evidence')
        data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        self.require(len(data)); cross.regular(path.parent, directory=True)
        with temporary.open('xb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.rename(temporary, path)


class ForwardLedger(old.ForwardLedger):
    def __init__(self): self.counts = dict.fromkeys(FORWARD_LIMITS, 0); self.handles = []
    def before(self, name):
        def hook(module, arguments):
            if self.counts[name] >= FORWARD_LIMITS[name]: raise RuntimeError('Per-arm forward limit reached: ' + name)
            self.counts[name] += 1
        return hook


class StageLog:
    def __init__(self, directory, budget):
        path = directory.with_suffix('.log')
        if directory.exists() or path.exists() or directory.is_symlink() or path.is_symlink(): raise RuntimeError('Stage and log must be new')
        self.budget = budget; self.handle = path.open('x', encoding='utf8')
    def write(self, value):
        self.budget.require(len(value.encode())); result = self.handle.write(value); self.handle.flush(); return result
    def flush(self): self.handle.flush()
    def close(self): self.handle.close()


def new_state(arm, contract_sha):
    if arm not in ARMS: raise RuntimeError('Unknown role')
    state = old.new_arm_state('candidate_mc2', contract_sha)
    state.update(arm=arm, completed_global_step=START, last_full_saved_global_step=START,
                 parent_checkpoint_sha256=PARENT_SHA, effective_optimizer_lrs=dict(ROLE_LRS[arm]))
    return state


def validate_parent_metadata(contract, completion):
    if training.canonical_json_sha256(contract) != PARENT_CANONICAL: raise RuntimeError('Wrong3927 parent contract')
    if (completion['status'], completion['cleanup_verified'], completion['contract_sha256']) != ('completed480', True, PARENT_CANONICAL):
        raise RuntimeError('Require normally completed3927 parent')
    state = completion['state']
    if (state['completed_global_step'], state['last_full_saved_global_step'], state['partial_batch_may_have_updated_D'], state['restore_completed']) != (START, START, False, True):
        raise RuntimeError('Invalid3927 actual complete cursors')
    if (state['full_snapshot']['path'], state['full_snapshot']['sha256'], state['full_snapshot']['resumable']) != (str(PARENT), PARENT_SHA, True):
        raise RuntimeError('Wrong3927 sole parent snapshot')
    if state['last_monitor']['global_step'] != START or 'uniform64' not in state['last_monitor']: raise RuntimeError('Missing3927 endpoint monitor')
    if contract['baseline3327_development']['mel_l1'] != cont.BASELINE3327['mel_l1']: raise RuntimeError('Original3327 protection baseline changed')
    return state


def validate_adams(student, discriminator, optimizer, d_optimizer, step, lrs):
    if {id(p) for p in student.parameters()} & {id(p) for p in discriminator.parameters()}: raise RuntimeError('Overlapping S/D ownership')
    for role, model, opt in (('student', student, optimizer), ('discriminator', discriminator, d_optimizer)):
        previous.prior.check_owner(opt, model)
        previous.dstate.old.validate_saved_adam(opt.state_dict(), model, step, lrs[role])
        if any(not bool(torch.isfinite(value).all()) for value in list(model.parameters()) + list(model.buffers()) if value.is_floating_point()):
            raise RuntimeError('Nonfinite model cannot be resumed or saved')


def restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, contract, receipt, device, arm):
    expected = (START, PARENT_CANONICAL, PARENT_LRS, 2, 74000000, True, 0)
    actual = tuple(payload[key] for key in ('global_step', 'run_contract_sha256', 'effective_optimizer_lrs', 'mc_samples', 'mc_private_seed_base', 'resumable', 'epoch'))
    if actual != expected: raise RuntimeError('Wrong complete3927 MC2 header')
    if payload['configuration'] != config.as_dict() or contract['configuration'] != config.as_dict(): raise RuntimeError('Parent config changed')
    if (payload['score_semantics'], payload['stop_reason'], payload['execution_error'], payload['development_monitor_global_step']) != ('reconstruction_clean_x0_sqrt_alpha_bar', 'maximum480_updates_completed', None, START):
        raise RuntimeError('Wrong3927 score semantics or terminal state')
    if payload['development'] != receipt['last_monitor']['development'] or probe.tree_sha(payload['rng_state']) != PARENT_RNG_SHA:
        raise RuntimeError('3927 monitor or RNG differs')
    for role, model, opt in (('student', student, optimizer), ('discriminator', discriminator, d_optimizer)):
        previous.prior.check_owner(opt, model)
        previous.dstate.old.validate_saved_adam(payload[role + '_optimizer'], model, START, PARENT_LRS[role])
        model.load_state_dict(payload[role], strict=True)
        previous.dstate.restore_optimizer(opt, payload[role + '_optimizer'])
        model.train(); training.set_requires_grad(model, True)
        for parameter in model.parameters(): parameter.grad = None
    training.restore_rng_state(payload['rng_state'], device)
    expected_rng = payload['rng_state'] if device.type == 'cuda' else dict(payload['rng_state'], cuda_all=[])
    if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(expected_rng): raise RuntimeError('3927 RNG did not restore')
    validate_adams(student, discriminator, optimizer, d_optimizer, START, PARENT_LRS)
    before = previous.content_state(student, discriminator, optimizer, d_optimizer)
    # 先确认父D lr2e-4完整恢复，随后才更改单个角色的lr；不用新lr冒充归档原值。
    saved_D = d_optimizer.state_dict(); expected_D = dict(saved_D, param_groups=[dict(g, lr=ROLE_LRS[arm]['discriminator']) for g in saved_D['param_groups']])
    expected_D_sha = probe.tree_sha(expected_D)
    for group in d_optimizer.param_groups: group['lr'] = ROLE_LRS[arm]['discriminator']
    after = previous.content_state(student, discriminator, optimizer, d_optimizer)
    if any(before[k] != after[k] for k in before if k != 'discriminator_optimizer') or after['discriminator_optimizer'] != expected_D_sha:
        raise RuntimeError('Role change touched more than D lr')
    validate_adams(student, discriminator, optimizer, d_optimizer, START, ROLE_LRS[arm])
    return {'parent_restore': before, 'role_lr_applied': after, 'only_declared_D_lr_may_differ': True}


def complete_update(state, step, optimizer, d_optimizer, call):
    if step != state['completed_global_step'] + 1 or not START < step <= STOP: raise RuntimeError('Only the next3928..4047 step is permitted')
    if any(x['uniform_step'] != step - 1 for x in old.actual_cursors(optimizer, d_optimizer).values()): raise RuntimeError('Bad entering Adam counters')
    state.update(attempted_global_step=step, partial_batch_may_have_updated_D=True, phase='D_S_update'); returned = False
    try:
        with old.observe_adam_step_returns(state, optimizer, d_optimizer): losses = call()
        returned = True
    finally:
        try:
            cursors = old.actual_cursors(optimizer, d_optimizer); state['optimizer_cursors'] = cursors
            if all(x['uniform_step'] == step for x in cursors.values()) and state['adam_step_returns'] == {'student': 1, 'discriminator': 1}:
                state.update(completed_global_step=step, new_updates=step - START, partial_batch_may_have_updated_D=False,
                             exception_after_both_adam_steps=not returned)
        except BaseException as error: state['cursor_inspection_error'] = repr(error)
    if state['partial_batch_may_have_updated_D']: raise RuntimeError('No proof of one complete D then S update')
    pair.finite_losses(losses); return losses


def paired_input_check(row, control):
    for key in ('global_step', 'indices', 'crop_seed', 'noise_seed', 'input_identity'):
        if row[key] != control[key]: raise RuntimeError('Actual paired inputs differ: ' + key)
    for key in ('source_noise', 'score_noise1', 't1', 't1_values', 'score_noise2', 't2', 't2_values', 'private_seed'):
        if row['mc'][key] != control['mc'][key]: raise RuntimeError('Actual paired MC2 draws differ: ' + key)
    if row['global_step'] == START + 1:
        same_tree({k: row[k] for k in ('discriminator', 'dist_raw')}, {k: control[k] for k in ('discriminator', 'dist_raw')})
        same_tree({k: row['mc'][k] for k in ('dist1', 'dist2')}, {k: control['mc'][k] for k in ('dist1', 'dist2')})
    # 首步adv/FM已经用了不同D更新，故不能在此要求相等。


def future_plan(development):
    planner = resume.GlobalBatchPlan(development['retained_train_indices'])
    if len(development['retained_train_indices']) != 39335 or planner.batches_per_pass != 1229: raise RuntimeError('Original retained global plan changed')
    rows = [{'global_step': step, 'indices': planner.for_step(step), 'crop_seed': 101234 + step, 'noise_seed': 201234 + step} for step in range(START + 1, STOP + 1)]
    if any(set(row['indices']) & set(development['excluded_train_indices']) for row in rows): raise RuntimeError('Development data leaked into training')
    return rows


def window60(rows):
    if len(rows) != 60 or [r['global_step'] for r in rows] not in [list(range(3928, 3988)), list(range(3988, 4048))]:
        raise RuntimeError('Only the two registered60-step windows')
    return {'first_global_step': rows[0]['global_step'], 'global_step': rows[-1]['global_step'], 'count': 60,
        'means': {k: math.fsum(r[k] for r in rows) / 60 for k in pair.LOSS_KEYS},
        'MC2_dist_means': {k: math.fsum(r['mc'][k] for r in rows) / 60 for k in ('dist1', 'dist2')},
        'first_last_difference': {k: rows[-1][k] - rows[0][k] for k in pair.LOSS_KEYS}, 'used_as_stop_gate': False}


def run_updates(state, future, step_call, evaluate, publish, check, initial):
    if [r['global_step'] for r in future] != list(range(3928, 4048)) or initial['global_step'] != START: raise RuntimeError('Wrong prospective plan')
    rows, windows, monitors = [], [], [initial]; reason = 'maximum120_updates_completed'
    initial_reasons = old.guard_reasons(monitors, cont.BASELINE3327)
    if initial_reasons: raise RuntimeError('Validated3927 replay unexpectedly violates its original guard')
    for planned in future:
        check(); row = step_call(planned); state['unpublished_step_record'] = row
        publish('step_losses.jsonl', row); state.pop('unpublished_step_record', None); rows.append(row)
        state['complete_loss_records'] = len(rows)
        if len(rows) % 60 == 0:
            window = window60(rows[-60:]); windows.append(window); publish('loss_windows.jsonl', window)
            entry = evaluate(planned['global_step']); monitors.append(entry); state['last_monitor'] = entry; publish('monitor.jsonl', entry)
            guards = old.guard_reasons(monitors[1:], cont.BASELINE3327)
            # 初始3927是身份复现；连续2次门只统计新60/120两个不同监测点。
            state['guard_reasons'] = guards; publish('guard_checks.jsonl', {'global_step': planned['global_step'], 'stop_reasons': guards})
            if guards: reason = guards[0]; break
        publish('status.json', state, False)
    return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': reason}


def checkpoint_common(state, config):
    last, step = state['last_monitor'], state['completed_global_step']
    return {'kind': 'acdd_mc2_Dlr_paired120_checkpoint', 'arm': state['arm'], 'global_step': step, 'epoch': 0,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': dict(ROLE_LRS[state['arm']]),
        'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'development': last['development'] if last and last['global_step'] == step else None,
        'development_monitor_global_step': last['global_step'] if last else None, 'run_contract_sha256': state['contract_sha256'],
        'lineage': {'parent_checkpoint_sha256': PARENT_SHA, 'parent_contract_sha256': PARENT_CANONICAL, 'start_after_global_step': START},
        'original_step0_guard_anchor': cont.S0_ANCHOR, 'stop_reason': state['reason'], 'execution_error': state.get('execution_error'),
        'actual_optimizer_cursors': state['optimizer_cursors']}


def save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock):
    if not state['restore_completed'] or state['partial_batch_may_have_updated_D'] or state.get('execution_error'):
        raise RuntimeError('Do not save technical failure or uncertain update as complete endpoint')
    step, arm = state['completed_global_step'], state['arm']
    if not START < step <= STOP or step not in (3987, 4047): raise RuntimeError('Endpoint must be a real complete monitored step')
    cursors = old.actual_cursors(optimizer, d_optimizer); state['optimizer_cursors'] = cursors
    if any(x['uniform_step'] != step for x in cursors.values()): raise RuntimeError('Saved cursor differs from actualAdam')
    validate_adams(student, discriminator, optimizer, d_optimizer, step, ROLE_LRS[arm])
    candidate = arm == ARMS[1]; role, key = ('full', 'full_snapshot') if candidate else ('student', 'student_snapshot')
    path = budget.output / arm / ('final_full.pt' if candidate else 'student_step_%06d.pt' % step)
    payload = dict(checkpoint_common(state, config), student=student.state_dict(), resumable=candidate)
    if candidate: payload.update(discriminator=discriminator.state_dict(), student_optimizer=optimizer.state_dict(),
                                discriminator_optimizer=d_optimizer.state_dict(), rng_state=training.capture_rng_state())
    clock.active(); budget.require()
    def committed():
        state[key] = {'path': str(path), 'global_step': step, 'sha256': None, 'committed': True, 'resumable': candidate}
        state['last_full_saved_global_step' if candidate else 'last_student_saved_global_step'] = step
        budget.committed(role, path)
    old.atomic_checkpoint_commit(payload, path, committed)
    state[key]['sha256'] = base.sha256_file(path); path.chmod(0o444); state[key]['frozen_readonly'] = True
    clock.active(); budget.write(path.parent / 'checkpoint_receipt.json', state)


def imported_source_files(parent_files, explicit):
    # 按真实import到的本项目源码找旧pin；只选源码，绝不整体继承父资产清单。
    selected = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name: continue
        path = Path(name).resolve()
        if path.suffix != '.py' or ROOT not in path.parents: continue
        expected = explicit.get(str(path), parent_files.get(str(path)))
        if expected is None: raise RuntimeError('Imported project source lacks reviewed SHA: ' + str(path))
        selected[str(path)] = expected
    return selected


def validate_supplemental_source_pins():
    for path, digest in NEW_SOURCE_PINS.items():
        p = Path(path)
        if p.name == '__init__.py':
            raw = subprocess.check_output(['git', 'show', cross.HEAD + ':' + str(p.relative_to(ROOT))], cwd=ROOT)
            if hashlib.sha256(raw).hexdigest() != digest: raise RuntimeError('Package init differs from frozen Git HEAD')
    declared = {str(TOOLS / name): value for name, value in previous.MODULES.items()}
    declared[str(TOOLS / 'inspect_acdd_teacher_target_cross_v1.py')] = previous.dstate.old.CROSS_SHA
    declared[str(TOOLS / 'inspect_acdd_adam_3227_v1.py')] = previous.dstate.HELPER_SHA
    for path, value in NEW_SOURCE_PINS.items():
        if TOOLS in Path(path).parents and declared.get(path) != value: raise RuntimeError('Supplemental helper differs from frozen importer declaration')


def prepare_context(args, mc, check):
    explicit = dict(FIXED_PINS, **{str(PROTOCOL): PROTOCOL_SHA, str(TOOLS / CONT_NAME): CONT_SHA,
        str(TOOLS / HELPER_NAME): HELPER_SHA, str(TOOLS / HELPER_TEST_NAME): HELPER_TEST_SHA,
        str(Path(__file__).resolve()): args.driver_sha256,
        str(TOOLS / 'test_run_acdd_mc2_dlr_paired120_from3927_v1.py'): args.tests_sha256})
    explicit.update(NEW_SOURCE_PINS)
    check_files(explicit, check)
    validate_supplemental_source_pins()
    parent = cross.read_json(PARENT_RUN / 'run_contract.json'); receipt = validate_parent_metadata(parent, cross.read_json(PARENT_RUN / 'completion.json'))
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if parent['configuration'] != config.as_dict(): raise RuntimeError('Keep mathematical configuration')
    development, dependencies = resume.load_development(DEV / 'development_contract.json')
    if parent['training_indices'] != development['retained_train_indices'] or parent['excluded_train_indices'] != development['excluded_train_indices']:
        raise RuntimeError('Development exclusions changed')
    files = dict(explicit); cross.merge_hashes(files, dependencies)
    cross.merge_hashes(files, imported_source_files(parent['file_sha256'], explicit))
    # 延迟import的生产网络源码也按原整体生产source指纹逐项核对。
    sources = training.collect_source_fingerprints(ROOT)
    for relative, digest in sources.items():
        path = str(ROOT / relative) if not Path(relative).is_absolute() else relative
        if parent['file_sha256'].get(path) != digest: raise RuntimeError('Production source changed: ' + path)
        files[path] = digest
    for asset in (base.TEACHER, base.VOCODER_DIR / 'generator_step_2500000.pt', ROOT / 'data/stats/mel_mean.npy', ROOT / 'data/stats/mel_std.npy'):
        digest = parent['file_sha256'].get(str(asset))
        if not digest: raise RuntimeError('Missing actual teacher/G/stats parent pin')
        files[str(asset)] = digest
    plan = parent['monitor64_plan']
    if plan != cross.read_json(previous.CROSS_IDENTITY)['plan']: raise RuntimeError('Original fixed64 plan changed')
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387: raise RuntimeError('Original data indexing changed')
    future = future_plan(development)
    future_files = previous.register_future_files(dataset, future, files, check)
    monitor_files = previous.register_future_files(dataset, [{'indices': plan['selected_indices']}], files, check)
    for name, digest in monitor_files.items():
        if parent['file_sha256'].get(name) != digest: raise RuntimeError('Fixed64 actual feature changed')
    if len(monitor_files) != 192: raise RuntimeError('Require all192 fixed64 feature files')
    check_files(files, check)
    return parent, receipt, config, development, dataset, future, future_files, monitor_files, files


def prepare_inputs(parent, dataset, diffusion, device):
    # 固定旧archive和已登记64句重建；不重新选句、不读取旧3227/3327模型合同。
    raw = resume.load_torch_checkpoint(cross.INPUT_FILE); inputs = {key: value.to(device) for key, value in raw.items()}; del raw
    plan = parent['monitor64_plan']; batches = []
    for b in range(2):
        batch = base.batch_for(dataset, plan['selected_indices'][b * 32:(b + 1) * 32], plan['crop_seeds'][b])
        entry = {'x0': batch['mel'].to(device), 'bnf': batch['bnf'].to(device), 'spk_emb': batch['spk_emb'].to(device)}
        training.seed_everything(plan['noise_seeds'][b])
        entry['source_noise'], entry['score_noise'] = torch.randn_like(entry['x0']), torch.randn_like(entry['x0'])
        entry['q950'] = diffusion.q_sample(entry['x0'], torch.full((32,), 950, device=device, dtype=torch.long), noise=entry['source_noise'])
        batches.append(entry)
    archive = resume.load_torch_checkpoint(cross.TARGET_FILE); target = archive['teacher_target'].to(device)
    if archive['global_source_step'] != 3107 or archive['sha256'] != parent['input_identities']['R3107']: raise RuntimeError('R3107 archive changed')
    def identities():
        return {'old': {key: probe.tensor_sha(value) for key, value in inputs.items()},
                'new64': [{key: cross.tensor_id(value) for key, value in batch.items()} for batch in batches], 'R3107': probe.tensor_sha(target)}
    if identities() != parent['input_identities']: raise RuntimeError('Original fixed input identities changed')
    return inputs, batches, target, identities


def joint_contract(args, parent, receipt, development, future, future_files, monitor_files, files):
    return {'kind': 'acdd_mc2_Dlr_paired120_development', 'protocol_sha256': PROTOCOL_SHA,
        'configuration': parent['configuration'], 'parent_checkpoint': str(PARENT), 'parent_checkpoint_sha256': PARENT_SHA,
        'parent_contract_sha256': PARENT_CANONICAL, 'parent_contract_file_sha256': PARENT_CONTRACT_SHA,
        'arms': list(ARMS), 'effective_optimizer_lrs_by_arm': ROLE_LRS, 'maximum_updates_per_arm': UPDATES,
        'start_after_global_step': START, 'stop_global_step': STOP, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'future_batches': future, 'future_training_feature_sha256': future_files, 'fixed64_feature_sha256': monitor_files,
        'training_indices': development['retained_train_indices'], 'excluded_train_indices': development['excluded_train_indices'],
        'runtime_files': files, 'input_identities': parent['input_identities'], 'monitor64_plan': parent['monitor64_plan'],
        'supplemental_runtime_source_pins': {'files_sha256': NEW_SOURCE_PINS, 'role': 'new_preexecution_binding',
            'parent_already_pinned_claimed': False, 'package_init_basis': 'exact_Git_HEAD_bytes', 'TOOLS_basis': 'frozen_importer_SHA_declaration'},
        'baseline3927_monitor': receipt['last_monitor'], 'baseline3327_development': cont.BASELINE3327,
        'original_step0_guard_anchor': cont.S0_ANCHOR, 'fixed_frozen_models_sha256': parent['fixed_frozen_models_sha256'],
        'monitor_updates': list(MONITOR_UPDATES), 'loss_windows': [[3928, 3987], [3988, 4047]],
        'per_arm_forward_limits': FORWARD_LIMITS, 'evaluation': EVALUATION, 'metric_identity_sha256': METRIC_IDENTITY_SHA,
        'evaluation_settings': {'pairs': 90, 'reference_characters': 4392, 'CER': 'corpus', 'SVA_threshold': .25,
            'all_metrics': ['cer', 'sva', 'dnsmos', 'utmos'], 'dnsmos_columns': ['P808', 'OVRL', 'SIG', 'BAK'],
            'batch_size': 4, 'DNS_workers': 1, 'ONNX_threads': 3, 'offline_assets': True},
        'archive_gate': {'path': str(OUTPUT / ARCHIVE_NAME), 'schema_version': 1,
                        'status': 'completed_private_archive_and_remote_removal', 'records': 91, 'file_sha_not_yet_known': True},
        'historical_ancestor': {'checkpoint_sha256': cont.PARENT_SHA, 'runtime_bytes_reverified': False,
                              'role': 'lineage_only_restore_private_archive_before_historical_reexecution'},
        'storage': {'joint_metadata_limit': META_LIMIT, 'system_reserve': previous.RESERVE, 'directory_reserve': previous.DIR_RESERVE,
                    'student_bytes': STUDENT_BYTES, 'full_bytes': FULL_BYTES, 'model_slack': previous.MODEL_SLACK, 'later_DEV90_bytes': EVAL_BYTES},
        'per_arm_compute_seconds': 900, 'cleanup_only_seconds': 90, 'candidate_design_ignores_control_metric_values': True,
        'no_automatic_training_evaluation_archive_or_retry': True, 'original_flags': cross.ORIGINAL_FLAGS}


def validate_control_archive(record, joint, contract_file_sha, control, inference, summary, exists=lambda p: Path(p).exists()):
    if record['schema_version'] != 1 or record['status'] != 'completed_private_archive_and_remote_removal': raise RuntimeError('No completed independent control archive')
    canonical = training.canonical_json_sha256(joint)
    if (record['joint_contract_file_sha256'], record['joint_contract_canonical_sha256']) != (contract_file_sha, canonical): raise RuntimeError('Archive bound to another joint contract')
    if control['cleanup_verified'] is not True or control['status'] not in ('completed120', 'protective_stop') or control['contract_sha256'] != canonical:
        raise RuntimeError('Control must have a clean nontechnical endpoint')
    state = control['state']; snapshot = state['student_snapshot']; step = state['completed_global_step']
    if state['partial_batch_may_have_updated_D'] or state.get('execution_error') or step not in (3987, 4047) or record['control_global_step'] != step:
        raise RuntimeError('Control is incomplete or technically failed')
    if (state['arm'], state['restore_completed'], state['last_student_saved_global_step'], snapshot['global_step'],
            snapshot['committed'], snapshot['resumable'], snapshot['frozen_readonly'], snapshot['path']) != (
            ARMS[0], True, step, step, True, False, True, str(OUTPUT / ARMS[0] / ('student_step_%06d.pt' % step))):
        raise RuntimeError('Control lacks its exact committed S-only endpoint')
    if inference['spec']['checkpoint'] != snapshot['path'] or inference['spec']['sha256'] != snapshot['sha256'] or inference['spec']['global_step'] != step:
        raise RuntimeError('Control evaluated another checkpoint')
    if inference['saved_training_contract_sha256'] != canonical or set(inference['wav_sha256']) != set(map(str, range(90))): raise RuntimeError('Control infer identity incomplete')
    if (summary['status'], summary['evaluated_pairs'], summary['metric_identity_sha256']) != ('development_only', 90, METRIC_IDENTITY_SHA):
        raise RuntimeError('Control objective evaluation is incomplete')
    folder = Path(EVALUATION[ARMS[0]]['output_root']) / EVALUATION[ARMS[0]]['label']
    expected = {snapshot['path']: snapshot['sha256']}
    expected.update({str(folder / 'wavs' / ('pair_%04d.wav' % i)): inference['wav_sha256'][str(i)] for i in range(90)})
    rows = record['records']
    if len(rows) != 91 or len({r['original_remote_path'] for r in rows}) != 91 or set(expected) != {r['original_remote_path'] for r in rows}:
        raise RuntimeError('Archive must contain exactly one controlS and its90 WAVs')
    for row in rows:
        path = row['original_remote_path']; private = PureWindowsPath(row['private_backup_absolute_path'])
        if (row['sha256'] != expected[path] or row['backup_sha256'] != row['sha256'] or row['backup_size_bytes'] != row['size_bytes']
                or not isinstance(row['size_bytes'], int) or row['size_bytes'] <= 0 or row['remote_removed'] is not True
                or not private.is_absolute() or '..' in private.parts or exists(path) or Path(path).is_symlink()):
            raise RuntimeError('Incomplete exact archive/removal proof: ' + path)
    # 分数仅验完整性，不读取其优劣决定候选参数或是否运行。
    return state


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf8') as handle: return list(csv.DictReader(handle))


def candidate_gate(args, joint, check=lambda: None):
    arm_path = OUTPUT / ARMS[0]; ev = Path(EVALUATION[ARMS[0]]['output_root']); system = ev / EVALUATION[ARMS[0]]['label']
    archive_path = OUTPUT / ARCHIVE_NAME
    check(); cross.regular(archive_path)
    if base.sha256_file(archive_path) != args.control_archive_sha256: raise RuntimeError('Archive receipt FILE SHA changed')
    record = cross.read_json(archive_path)
    paths = {'control_completion_file_sha256': arm_path / 'completion.json',
             'infer_completion_file_sha256': system / 'inference_complete.json',
             'evaluation_summary_file_sha256': system / 'metrics/development_summary_cer_dnsmos_sva_utmos.json'}
    for field, path in paths.items():
        check(); cross.regular(path)
        if base.sha256_file(path) != record[field]: raise RuntimeError('Retained archive proof changed: ' + field)
    control, inference, summary = (cross.read_json(paths[key]) for key in paths)
    state = validate_control_archive(record, joint, base.sha256_file(OUTPUT / 'run_contract.json'), control, inference, summary)
    if any((arm_path / name).exists() for name in ('failure.json', 'cleanup_failure.json')): raise RuntimeError('Control has technical failure evidence')
    if summary['inference_complete_sha256'] != record['infer_completion_file_sha256'] or summary['run_identity_sha256'] != inference['run_identity_sha256']:
        raise RuntimeError('Control evaluation chain differs')
    if base.sha256_file(ev / 'run_identity.json') != inference['run_identity_sha256']: raise RuntimeError('Retained run identity changed')
    identity = cross.read_json(ev / 'run_identity.json'); label = EVALUATION[ARMS[0]]['label']
    if (inference['status'], inference['system'], summary['system'], identity['purpose'], identity['pairs'],
            identity['candidate_training_contract'], identity['start_step'], identity['inference_batch_size']) != (
            'development_candidate_only', label, label, 'development_candidate_not_formal_8100', 90,
            str(OUTPUT / 'run_contract.json'), 950, 1): raise RuntimeError('Control evaluation uses another registered setting')
    if identity['systems'] != {label: inference['spec']}: raise RuntimeError('Evaluation system/checkpoint identity differs')
    expected_spec = dict(checkpoint=state['student_snapshot']['path'], sha256=state['student_snapshot']['sha256'],
                         state_key='student', epoch=0, global_step=state['completed_global_step'], steps=1, sampling='existing_student_t950_to_zero')
    if inference['spec'] != expected_spec: raise RuntimeError('Wrong control single-step student evaluator')
    if identity['inputs'].get(expected_spec['checkpoint']) != expected_spec['sha256'] or identity['inputs'].get(str(OUTPUT / 'run_contract.json')) != record['joint_contract_file_sha256']:
        raise RuntimeError('Actual evaluation source list lacks control checkpoint/contract identity')
    # 唯一允许不再读取的评估输入是已私有归档的控制S。其他原始源文件及小收据仍须现存验真。
    retained_inputs = {p: sha for p, sha in identity['inputs'].items() if p != expected_spec['checkpoint']}
    check_files(retained_inputs, check)
    if base.sha256_file(system / 'pairs.csv') != inference['pairs_csv_sha256']: raise RuntimeError('Control pairs CSV changed')
    original_rows = csv_rows(DEV / 'pairs.csv'); actual_rows = csv_rows(system / 'pairs.csv')
    if len(original_rows) != 90 or {r['pair_id'] for r in original_rows} != set(map(str, range(90))): raise RuntimeError('Require the original90 development pairs')
    expected_rows = [dict(r, output_wav=str(system / 'wavs' / ('pair_%04d.wav' % int(r['pair_id'])))) for r in original_rows]
    if actual_rows != expected_rows: raise RuntimeError('Control changed original pair inputs/order or output paths')
    if set(summary['metric_csv_sha256']) != {'cer', 'sva', 'dnsmos', 'utmos'}: raise RuntimeError('All control metric tables required')
    for name, digest in summary['metric_csv_sha256'].items():
        check(); path = system / 'metrics' / (name + '.csv')
        if base.sha256_file(path) != digest: raise RuntimeError('Retained control CSV changed')
        rows = csv_rows(path)
        if len(rows) != 90 or {r['pair_id'] for r in rows} != set(map(str, range(90))): raise RuntimeError('Control metric CSV misses development pairs')
        if name == 'cer' and sum(int(r['reference_chars']) for r in rows) != 4392: raise RuntimeError('Control CER denominator changed')
        if name == 'sva' and any(float(r['threshold']) != .25 for r in rows): raise RuntimeError('Control SVA threshold changed')
    rows = [json.loads(line) for line in (arm_path / 'step_losses.jsonl').read_text().splitlines()]
    if [r['global_step'] for r in rows] != list(range(START + 1, state['completed_global_step'] + 1)): raise RuntimeError('Control raw training records incomplete')
    return {r['global_step']: r for r in rows}, {'archive_receipt_file_sha256': args.control_archive_sha256, 'control_completion_file_sha256': record['control_completion_file_sha256']}


def launch_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true'); parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--driver-sha256'); parser.add_argument('--tests-sha256'); parser.add_argument('--control-archive-sha256')
    args = parser.parse_args(argv)
    if not args.run_approved or args.arm not in ARMS: raise RuntimeError('Require explicit stage approval')
    for key in ('driver_sha256', 'tests_sha256') + (('control_archive_sha256',) if args.arm == ARMS[1] else ()):
        v = getattr(args, key)
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v): raise RuntimeError('Require final reviewed FILE SHA: ' + key)
    if args.arm == ARMS[0] and args.control_archive_sha256 is not None: raise RuntimeError('No future receipt may be fabricated before control')
    return args


def main(argv=None):
    started = time.monotonic(); clock, budget, log = None, None, None; locks = []; owned = False
    state, completion, joint_sha, stage = {}, None, None, None; ledger = ForwardLedger()
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        args = launch_arguments(argv); stage = OUTPUT / args.arm
        clock = old.StopClock(900, started); clock.install(); previous.watchdog(900)
        training.assert_rented_server(ROOT); cross.regular(DEV, directory=True)
        locks = training.acquire_runtime_locks(ROOT)
        if len(locks) != 3: raise RuntimeError('Require three actual workflow locks')
        # 再持有inference锁，避免阶段间检查后与独立推理发生竞态。
        inference_lock = ROOT / 'exp/locks/text_unseen_fastvoicegrad_student_inference.lock'
        if inference_lock.is_symlink(): raise RuntimeError('Inference lock symlink')
        handle = inference_lock.open('a+')
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException: handle.close(); raise
        locks.append(handle)
        budget = Budget(OUTPUT, args.arm); disk = budget.require()
        if args.arm == ARMS[0]:
            if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('Control joint root must be new')
            OUTPUT.mkdir()
        else: cross.regular(OUTPUT, directory=True)
        log = StageLog(stage, budget); sys.stdout = sys.stderr = log; stage.mkdir(); owned = True
        def check(): clock.active(); budget.require()
        if training.git_head(ROOT) != cross.HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Require unchanged clean authoritative production code')
        if ((platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6')
                or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or cross.flags() != cross.ORIGINAL_FLAGS):
            raise RuntimeError('Keep actual training runtime and precision')
        check_files({str(TOOLS / HELPER_NAME): HELPER_SHA, str(TOOLS / HELPER_TEST_NAME): HELPER_TEST_SHA}, check)
        mc = importlib.import_module(HELPER_NAME[:-3])
        if (mc.START, mc.STOP, mc.MC2_SEED) != (START, STOP, 74000000): raise RuntimeError('Wrong frozenMC2 range')
        parent, parent_receipt, config, development, dataset, future, future_files, monitor_files, files = prepare_context(args, mc, check)
        joint = joint_contract(args, parent, parent_receipt, development, future, future_files, monitor_files, files)
        control_rows, archive_evidence = {}, None
        if args.arm == ARMS[0]: budget.write(OUTPUT / 'run_contract.json', joint)
        else:
            saved_joint = cross.read_json(OUTPUT / 'run_contract.json')
            if joint != saved_joint: raise RuntimeError('Candidate design differs from the pre-control joint contract')
            control_rows, archive_evidence = candidate_gate(args, joint, check)
        joint_sha = training.canonical_json_sha256(joint); state = new_state(args.arm, joint_sha)
        gpu = cross.gpu_gate(); device = torch.device('cuda'); check()
        diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
        inputs, batches, target, input_identity = prepare_inputs(parent, dataset, diffusion, device)
        check(); payload = resume.load_torch_checkpoint(PARENT)
        teacher, student, unused = base.load_teacher_and_student(base.TEACHER, config, device); del unused
        vocoder, vconfig, unused = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
            base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device); del unused
        discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed_d = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed = resume.load_torch_checkpoint(pair.PARENT / 'latest_full.pt')
        if fixed['global_step'] != 3107 or fixed['run_contract_sha256'] != pair.PARENT_CONTRACT: raise RuntimeError('Original fixedD3107 identity changed')
        fixed_d.load_state_dict(fixed['discriminator'], strict=True); del fixed
        fixed_d.eval(); training.set_requires_grad(fixed_d, False)
        mean, std = base.load_mel_statistics(ROOT / 'data', device)
        imported_source_files(files, files)  # 延迟网络import也必须早已纳入契约，不能训练后才追加pin。
        frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
        frozen_sha = {name: pair.model_sha(model) for name, model in frozen.items()}
        if frozen_sha != parent['fixed_frozen_models_sha256']: raise RuntimeError('Original fixed model identity changed')
        models = dict(frozen, student=student, active_D=discriminator, diffusion=diffusion)
        optimizer = torch.optim.Adam(student.parameters(), lr=PARENT_LRS['student'], betas=(.5, .9))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=PARENT_LRS['discriminator'], betas=(.5, .9))
        restored = restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, parent, parent_receipt, device, args.arm); del payload
        state.update(restore_completed=True, restore_identity=restored, optimizer_cursors=old.actual_cursors(optimizer, d_optimizer),
                     status='running', initial_disk=disk, initial_gpu=gpu, archive_evidence=archive_evidence)
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
            if ledger.delta(before) != {'teacher': 21, 'student': 4, 'vocoder': 2, 'active_D_branches': 0, 'fixed_D_branches': 16}: raise RuntimeError('Wrong monitoring forward count')
            entry['actual_forward_counts'] = ledger.delta(before); unchanged(); return entry
        initial = evaluate(START)
        for key in ('fixed_train', 'development', 'uniform64'): same_tree(initial[key], parent_receipt['last_monitor'][key])
        state['last_monitor'] = initial; budget.write(stage / 'monitor.jsonl', initial, append=True)
        def step_call(planned):
            step = planned['global_step']; batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
            training.seed_everything(planned['noise_seed']); identity = mc.original_input_identity(batch, device, config); receipt = {}
            check(); before = dict(ledger.counts)
            losses = complete_update(state, step, optimizer, d_optimizer, lambda: mc.train_batch_mc2(
                batch, teacher, student, vocoder, discriminator, diffusion, optimizer, d_optimizer, mean, std, device, config,
                global_step=step, mc_receipt=receipt))
            row = dict(losses, global_step=step, updates=step - START, indices=planned['indices'], crop_seed=planned['crop_seed'],
                noise_seed=planned['noise_seed'], input_identity=identity, mc=receipt,
                actual_optimizer_cursors=state['optimizer_cursors'], actual_forward_counts=ledger.delta(before))
            state['unpublished_step_record'] = row
            for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
                if receipt[key] != identity[key]: raise RuntimeError('Actual original RNG mismatch')
            if row['actual_forward_counts'] != {'teacher': 2, 'student': 2, 'vocoder': 3, 'active_D_branches': 32, 'fixed_D_branches': 0}: raise RuntimeError('Wrong train batch call count')
            previous.adam_counters(optimizer, d_optimizer, step)
            if [optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr']] != list(ROLE_LRS[args.arm].values()): raise RuntimeError('Role lr drift')
            if step in control_rows: paired_input_check(row, control_rows[step])
            return row
        result = run_updates(state, future, step_call, evaluate, lambda name, value, append=True: budget.write(stage / name, value, append), check, initial)
        state.update(reason=result['reason'], loss_windows=result['windows'], monitor_global_steps=[r['global_step'] for r in result['monitors']],
                     status='completed120' if result['reason'] == 'maximum120_updates_completed' else 'protective_stop', phase='saving')
        unchanged(); save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock)
        check_files(files, check); unchanged()
        if training.canonical_json_sha256(cross.read_json(OUTPUT / 'run_contract.json')) != joint_sha: raise RuntimeError('Joint contract mutated')
        if state['new_updates'] == UPDATES and ledger.counts != FORWARD_LIMITS: raise RuntimeError('Complete per-arm forward budget differs')
        completion = {'status': state['status'], 'cleanup_verified': True, 'state': dict(state), 'contract_sha256': joint_sha,
            'actual_forward_counts': dict(ledger.counts), 'elapsed_seconds': time.monotonic() - started,
            'actual_common_prefix_steps': min(state['new_updates'], len(control_rows)) if args.arm == ARMS[1] else None,
            'no_evaluation_or_archive_executed': True, 'automatic_next_stage_authorized': False}
        budget.write(stage / 'trial_completion.json', dict(completion, cleanup_verified=False, status='pending_cleanup')); check()
    except BaseException as error:
        if clock: clock.request('technical_failure')
        if budget: budget.cancel_future_on_fatal_abort(repr(error))
        state.update(status='technical_failure', execution_error=repr(error))
        if owned:
            try: budget.write(stage / 'failure.json', {'status': 'technical_failure', 'state': state, 'actual_forward_counts': ledger.counts, 'error': repr(error)})
            except BaseException as failed: saved_stderr.write('Failure record unavailable: %r; original=%r\n' % (failed, error))
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = old.cleanup_resources(ledger, clock, log, locks)
        if errors and budget: budget.cancel_future_on_fatal_abort(repr(errors))
        if errors and owned:
            try: budget.write(stage / 'cleanup_failure.json', {'status': 'technical_failure', 'errors': errors, 'state': state})
            except BaseException as error: saved_stderr.write('Cleanup failure record unavailable: %r\n' % error)
        if errors and original_error is None: raise RuntimeError('Stage cleanup failed: ' + repr(errors))
        if original_error is None and not errors and completion is not None:
            try:
                clock.active(); budget.publish_success(stage / 'completion.json', completion)
            except BaseException as error:
                budget.cancel_future_on_fatal_abort(repr(error))
                try: budget.write(stage / 'failure.json', {'status': 'technical_failure', 'state': state, 'error': repr(error)})
                except BaseException: pass
                raise


if __name__ == '__main__': main()
