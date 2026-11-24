"""MC1/MC2从真实3327各至多120步；默认拒绝运行，不做端点推理或评分。"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Run only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
MODULE_SHAS = {
    'run_acdd_timestep_paired60_v1.py': '39520259afddf3207355a0c901d029cec5c026f6ff611a558845b60fed8857f7',
    'run_acdd_sampler_replication3_v1.py': 'a2dcb3cd636560d8bdd351982726b186e3925e11cc876891628af1d906b3ca98',
    'acdd_mc2_train_batch_v1.py': 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16',
}
TEST_SHAS = {
    'test_acdd_mc2_train_batch_v1.py': '061e13bf26afa877a1cb7444525e651792175bd924fbd3d9f6cf8cce906cacce',
    'test_acdd_mc2_paired120_v1.py': '7b2747783c3243a0c3bfece250a67f037bad70efddcd8563dbfaa4e53318bbed',
}
for name, expected in MODULE_SHAS.items():
    if hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen helper changed: ' + name)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_timestep_paired60_v1 as previous
import run_acdd_sampler_replication3_v1 as replication
import acdd_mc2_train_batch_v1 as mc

cross, pair, base = previous.cross, previous.pair, previous.base
training, resume, probe = previous.training, previous.resume, previous.probe
DEV, PARENT = previous.DEV, previous.PARENT
START, STOP, UPDATES = 3327, 3447, 120
ARMS = ('control_mc1', 'candidate_mc2')
LRS = {'student': .00005, 'discriminator': .0002}
OUTPUT = DEV / 'acdd_mc2_paired120_from3327_v1'
PROTOCOL = DEV / 'ACDD3327_MC2_PAIRED120_PROTOCOL_CN.md'
PROTOCOL_SHA = '4cea00f3751782660f08a208fbba2d7d71965adb8a3b07ed10e13459e89e0394'
FIRST_ANCHOR = DEV / 'ACDD3327_MC2_FIRST_STEP_ANCHOR_V1.json'
FIRST_ANCHOR_SHA = '0a22b2f98196228e93a943d6ac9481d688ee9fec2e62161852f682b760dd1a7c'
PARENT_SHA, PARENT_CANONICAL = previous.PARENT_SHA, previous.PARENT_CANONICAL
FULL_BYTES, STUDENT_BYTES = 1424633856, 309268480
INITIAL_REQUIRED_BYTES = 5109268480
S0_ANCHOR = {'mel_l1': .19941353797912598, 'prediction_rms': .9994737505912781}
FORWARD_LIMITS = {'teacher': 490, 'student': 512, 'vocoder': 740,
                  'active_D_branches': 7680, 'fixed_D_branches': 160}
PAIR_UPDATES, UNIFORM64_UPDATES = (0, 30, 60, 90, 120), (0, 60, 120)
BoundaryStop, StopClock = previous.BoundaryStop, previous.StopClock


class Budget(previous.Budget):
    """沿用未来两份保存+系统保留预算，增加写入路径链和取消未来保存的明确记录。"""
    def __init__(self, output):
        super().__init__(output, FULL_BYTES, STUDENT_BYTES)
        self.cancelled = {}

    def write(self, path, value, append=False):
        path = Path(path)
        if self.output.resolve() not in path.resolve().parents or path.suffix not in ('.json', '.jsonl'):
            raise RuntimeError('Metadata must remain inside this new output')
        cross.regular(path.parent, directory=True)
        return super().write(path, value, append)

    def cancel_future_on_fatal_abort(self, reason):
        # 整段已终止且不再保存模型，才能取消尚未提交的预算；不删除任何.tmp或旧文件。
        self.cancelled.update({key: {'bytes': value, 'reason': reason} for key, value in self.pending.items()})
        self.pending.clear()


class ForwardLedger:
    """只观察实际模型调用；异常前向也记入预算，不产生额外S/T/V/D前向。"""
    def __init__(self):
        self.counts = {name: 0 for name in FORWARD_LIMITS}
        self.handles = []

    def before(self, name):
        def hook(module, arguments):
            if self.counts[name] >= FORWARD_LIMITS[name]:
                raise RuntimeError('Predeclared forward budget exhausted: ' + name)
            self.counts[name] += 1
        return hook

    def install(self, student, teacher, vocoder, active_d, fixed_d):
        try:
            for name, model in (('student', student), ('teacher', teacher), ('vocoder', vocoder)):
                self.handles.append(model.register_forward_pre_hook(self.before(name)))
            for name, model in (('active_D_branches', active_d), ('fixed_D_branches', fixed_d)):
                branches = tuple(model.loss_branches())
                if len(branches) != 8:
                    raise RuntimeError('The real MC2 trial requires exactly5MPD+3MRD')
                for branch in branches:
                    self.handles.append(branch.register_forward_pre_hook(self.before(name)))
        except BaseException:
            self.close(); raise

    def close(self):
        errors = []
        for handle in reversed(self.handles):
            try: handle.remove()
            except BaseException as error: errors.append(repr(error))
        self.handles.clear()
        if errors: raise RuntimeError('Forward hook cleanup failed: ' + repr(errors))

    def delta(self, before):
        return {key: self.counts[key] - before[key] for key in self.counts}


def new_arm_state(arm, contract_sha):
    if arm not in ARMS:
        raise RuntimeError('Undeclared MC arm')
    return {'arm': arm, 'status': 'restoring', 'phase': 'restore', 'restore_completed': False,
            'completed_global_step': START, 'new_updates': 0, 'attempted_global_step': None,
            'partial_batch_may_have_updated_D': False, 'optimizer_cursors': None,
            'last_full_saved_global_step': START, 'last_student_saved_global_step': None,
            'full_snapshot': None, 'student_snapshot': None, 'last_monitor': None,
            'contract_sha256': contract_sha, 'effective_optimizer_lrs': dict(LRS),
            'parent_checkpoint_sha256': PARENT_SHA, 'execution_stop_reasons': []}


def optimizer_cursor(optimizer):
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise RuntimeError('Empty or duplicated optimizer ownership')
    steps = []
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state or 'step' not in state:
            raise RuntimeError('Missing inherited Adam counter')
        value = float(state['step'])
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise RuntimeError('Invalid actual Adam counter')
        steps.append(int(value))
    unique = sorted(set(steps))
    return {'parameter_count': len(steps), 'distinct_parameter_steps': unique,
            'uniform_step': unique[0] if len(unique) == 1 else None}


def actual_cursors(optimizer, d_optimizer):
    return {'student': optimizer_cursor(optimizer), 'discriminator': optimizer_cursor(d_optimizer)}


@contextlib.contextmanager
def observe_adam_step_returns(state, optimizer, d_optimizer):
    # Adam可能先递增最后一个counter再在参数运算中失败；同时观察step正常返回才能证明完整。
    returned = {'student': 0, 'discriminator': 0}
    state['adam_step_returns'] = returned
    missing = object(); installed = []
    try:
        for name, opt in (('student', optimizer), ('discriminator', d_optimizer)):
            original, local = opt.step, opt.__dict__.get('step', missing)
            def observed(*args, _name=name, _original=original, **kwargs):
                if returned[_name] or (_name == 'student' and returned['discriminator'] != 1):
                    raise RuntimeError('Require one returned D.step followed by one S.step')
                result = _original(*args, **kwargs)
                returned[_name] += 1
                return result
            opt.step = observed; installed.append((opt, local))
        yield returned
    finally:
        for opt, local in reversed(installed):
            if local is missing: del opt.step
            else: opt.step = local


def complete_update(state, step, optimizer, d_optimizer, call):
    if step != state['completed_global_step'] + 1 or not START < step <= STOP:
        raise RuntimeError('Require exactly the next declared3328..3447 complete update')
    before = actual_cursors(optimizer, d_optimizer)
    if any(item['uniform_step'] != step - 1 for item in before.values()):
        raise RuntimeError('Both actual Adam cursors must match the previous complete step')
    state.update(attempted_global_step=step, partial_batch_may_have_updated_D=True, phase='D_S_update')
    returned = False
    try:
        with observe_adam_step_returns(state, optimizer, d_optimizer):
            losses = call()
        returned = True
    finally:
        # 即使helper在S.step之后日志/检查处抛错，也先报告两个Adam真正已经走到哪里。
        try:
            cursor = actual_cursors(optimizer, d_optimizer)
            state['optimizer_cursors'] = cursor
            if (all(item['uniform_step'] == step for item in cursor.values())
                    and state['adam_step_returns'] == {'student': 1, 'discriminator': 1}):
                state.update(completed_global_step=step, new_updates=step - START,
                             partial_batch_may_have_updated_D=False,
                             exception_after_both_adam_steps=not returned)
        except BaseException as cursor_error:
            state['cursor_inspection_error'] = repr(cursor_error)
    if state['partial_batch_may_have_updated_D']:
        raise RuntimeError('Helper returned without two complete uniform Adam updates')
    pair.finite_losses(losses)
    return losses


def window60(rows):
    if len(rows) != 60 or rows[0]['global_step'] not in (3328, 3388):
        raise RuntimeError('Only the two preregistered complete60 windows are allowed')
    if [row['global_step'] for row in rows] != list(range(rows[0]['global_step'], rows[0]['global_step'] + 60)):
        raise RuntimeError('Window must contain60 consecutive absolute steps')
    for row in rows:
        pair.finite_losses(row)
    return {'first_global_step': rows[0]['global_step'], 'global_step': rows[-1]['global_step'], 'count': 60,
            'means': {key: math.fsum(row[key] for row in rows) / 60 for key in pair.LOSS_KEYS},
            'first_last_difference': {key: rows[-1][key] - rows[0][key] for key in pair.LOSS_KEYS},
            'first': {key: rows[0][key] for key in pair.LOSS_KEYS},
            'last': {key: rows[-1][key] for key in pair.LOSS_KEYS}, 'used_as_stop_gate': False}


def guard_reasons(monitors, baseline3327):
    if not monitors or not math.isfinite(baseline3327['mel_l1']) or baseline3327['mel_l1'] <= 0:
        raise RuntimeError('Require a finite positive arm-specific3327 baseline')
    steps = [row['global_step'] for row in monitors]
    if steps != sorted(set(steps)):
        raise RuntimeError('Do not repeat a monitor to fabricate consecutive failures')
    for row in monitors:
        scores = row['development']
        if any(not math.isfinite(scores[name]) or scores[name] < 0 for name in ('mel_l1', 'prediction_rms')):
            raise RuntimeError('Nonfinite or negative development protection metric')
    scores = monitors[-1]['development']; reasons = []
    if scores['mel_l1'] > S0_ANCHOR['mel_l1'] * 1.10 or scores['prediction_rms'] < S0_ANCHOR['prediction_rms'] * .80:
        reasons.append('original_S0_extreme_guard')
    if len(monitors) >= 2 and all(row['development']['mel_l1'] > baseline3327['mel_l1'] * 1.05 for row in monitors[-2:]):
        reasons.append('two_consecutive_DEV32_above3327_by5percent')
    return reasons


def future_plan(development):
    plan = resume.GlobalBatchPlan(development['retained_train_indices'])
    if len(development['retained_train_indices']) != 39335 or plan.batches_per_pass != 1229:
        raise RuntimeError('The original global batch plan changed')
    rows = [{'global_step': step, 'indices': plan.for_step(step), 'crop_seed': 101234 + step,
             'noise_seed': 201234 + step} for step in range(START + 1, STOP + 1)]
    if any(set(row['indices']) & set(development['excluded_train_indices']) for row in rows):
        raise RuntimeError('Training intersects the52 development exclusions')
    return rows


def validate_first_anchor(anchor):
    # 此独立adapter留给主线程接入最终原生产3328锚点；不猜旧改t分支的loss。
    if (anchor.get('global_step') != START + 1 or anchor.get('parent_checkpoint_sha256') != PARENT_SHA
            or anchor.get('production_train_sha256') != mc.PRODUCTION_SHA
            or anchor.get('uses_original_uniform_t') is not True):
        raise RuntimeError('Require audited original-production3328 anchor, never a retimed replication')
    required = {'discriminator', 'student_adv', 'fm_raw', 'dist_raw', 'student_total'}
    if not required <= set(anchor.get('expected_losses', {})) or not anchor.get('source_sha256'):
        raise RuntimeError('Missing original3328 numeric or file-provenance anchors')
    if any(not math.isfinite(float(anchor['expected_losses'][key])) for key in required):
        raise RuntimeError('Nonfinite historical first-step anchor')
    if set(anchor.get('input_identity', {})) != {'batch_tensors', 'source_noise', 'score_noise1', 't1', 't1_values'}:
        raise RuntimeError('Incomplete original input identity')
    return anchor


def first_input_check(identity, anchor):
    if identity != anchor['input_identity']:
        raise RuntimeError('Original3328 input identity failed before any D/S update')


def first_plan_check(planned, anchor):
    if planned['global_step'] != START + 1 or planned['global_step'] != anchor['global_step']:
        raise RuntimeError('Original3328 registered plan mismatch: global_step')
    for key in ('indices', 'crop_seed', 'noise_seed'):
        if planned[key] != anchor['batch_plan'][key]:
            raise RuntimeError('Original3328 registered plan mismatch: ' + key)


def first_loss_check(arm, losses, receipt, anchor):
    expected = anchor['expected_losses']
    observed = {key: losses[key] for key in ('discriminator', 'student_adv', 'fm_raw')}
    if 'fm_weighted' in expected: observed['fm_weighted'] = losses['fm_weighted']
    observed['dist_raw'] = receipt['dist1']
    if arm == ARMS[0]:
        observed['student_total'] = losses['student_total']
        if 'dist_weighted' in expected: observed['dist_weighted'] = losses['dist_weighted']
    return previous.close_losses(observed, {key: expected[key] for key in observed})


def paired_input_check(row, control):
    for key in ('global_step', 'indices', 'input_identity'):
        if row[key] != control[key]:
            raise RuntimeError('Actual common-prefix paired input mismatch: ' + key)
    if row['global_step'] == START + 1:
        left = {key: row[key] for key in ('discriminator', 'student_adv', 'fm_raw', 'fm_weighted')}
        right = {key: control[key] for key in left}
        left['dist1'], right['dist1'] = row['mc']['dist1'], control['mc']['dist1']
        return previous.close_losses(left, right)
    return {}  # 从第二步开始模型轨迹可不同，不能要求loss继续相等。


def assert_saveable(state, student, discriminator, optimizer, d_optimizer):
    if not state['restore_completed'] or state['partial_batch_may_have_updated_D']:
        raise RuntimeError('Never save half or uncertain D/S state as a complete checkpoint')
    cursors = actual_cursors(optimizer, d_optimizer)
    state['optimizer_cursors'] = cursors
    if any(row['uniform_step'] != state['completed_global_step'] for row in cursors.values()):
        raise RuntimeError('Saved step must equal both actual uniform Adam cursors')
    for name, model, opt in (('student', student, optimizer), ('discriminator', discriminator, d_optimizer)):
        previous.prior.check_owner(opt, model)
        if len(opt.param_groups) != 1 or opt.param_groups[0]['lr'] != LRS[name] or opt.param_groups[0]['betas'] != (.5, .9):
            raise RuntimeError('Saved optimizer ownership or role hyperparameters changed')
    def finite_tree(value):
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError('Do not serialize nonfinite model/Adam as resumable state')
        elif isinstance(value, dict):
            for item in value.values(): finite_tree(item)
        elif isinstance(value, (tuple, list)):
            for item in value: finite_tree(item)
    for model in (student, discriminator):
        finite_tree(dict(model.named_parameters())); finite_tree(dict(model.named_buffers()))
    finite_tree(optimizer.state_dict()); finite_tree(d_optimizer.state_dict())


def atomic_checkpoint_commit(payload, path, committed):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    cross.regular(path.parent, directory=True)
    if any(item.exists() or item.is_symlink() for item in (path, temporary)):
        raise RuntimeError('Checkpoint and temporary targets must both be new')
    with temporary.open('xb') as handle:
        torch.save(dict(payload), handle)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    committed()  # 原子提交后第一件事：更新真实已保存游标；其后任何错误不得反悔。
    descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def save_endpoint(directory, state, common, student, discriminator, optimizer, d_optimizer, budget, clock):
    if state['completed_global_step'] == START:
        return
    assert_saveable(state, student, discriminator, optimizer, d_optimizer)
    directory = Path(directory)
    if directory != budget.output / state['arm']:
        raise RuntimeError('Endpoint must belong to this declared arm directory')
    step = state['completed_global_step']; candidate = state['arm'] == ARMS[1]
    if state['arm'] not in ARMS or common['global_step'] != step:
        raise RuntimeError('Unknown checkpoint arm or mismatched complete step')
    role, key = ('full', 'full_snapshot') if candidate else ('student', 'student_snapshot')
    path = directory / ('final_full.pt' if candidate else 'student_step_%06d.pt' % step)
    payload = dict(common, student=student.state_dict(), resumable=candidate)
    if candidate:
        payload.update(discriminator=discriminator.state_dict(), student_optimizer=optimizer.state_dict(),
                       discriminator_optimizer=d_optimizer.state_dict(), rng_state=training.capture_rng_state())
    clock.saving(); budget.require()
    def committed():
        state[key] = {'path': str(path), 'global_step': step, 'sha256': None, 'committed': True, 'resumable': candidate}
        state['last_full_saved_global_step' if candidate else 'last_student_saved_global_step'] = step
        budget.committed(role, path)
    atomic_checkpoint_commit(payload, path, committed)
    state[key]['sha256'] = base.sha256_file(path)
    path.chmod(0o444); state[key]['frozen_readonly'] = True
    clock.saving(); budget.write(directory / 'checkpoint_receipt.json', state)


def failure_record(output, state, summaries, error, budget, ledger):
    record = {'status': 'execution_aborted', 'error_type': type(error).__name__, 'error': str(error),
              'active_arm': state, 'completed_arms': summaries, 'actual_forward_counts': dict(ledger.counts),
              'old_artifacts_not_written_by_this_trial': True, 'new_training_retry_authorized': False,
              'cancelled_future_save_budgets': dict(budget.cancelled) if budget else {}}
    if output is not None:
        if (Path(output) / 'completion.json').exists():
            budget.write(Path(output) / 'completion.json', record)
        budget.write(Path(output) / 'failure.json', record)
    return record


def run_updates(state, future, step_call, evaluate, publish, check, baseline3327, initial):
    """可用CPU假回调测试完整调度；真正的D/S实现只由step_call注入。"""
    if [row['global_step'] for row in future] != list(range(START + 1, STOP + 1)):
        raise RuntimeError('The complete prospective120-step plan must be fixed before an arm starts')
    rows, pending, windows, monitors = [], [], [], [initial]
    reason = 'maximum120_updates_completed'
    guards = guard_reasons(monitors, baseline3327)
    if guards:
        return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': guards[0]}
    for planned in future:
        check()
        row = step_call(planned)
        state['unpublished_step_record'] = row
        publish('step_losses.jsonl', row)
        state.pop('unpublished_step_record', None)
        rows.append(row); pending.append(row)
        state.update(complete_loss_records=len(rows), partial_window_steps=len(pending))
        if len(pending) == 60:
            window = window60(pending); windows.append(window); pending = []
            state.update(partial_window_steps=0, loss_windows=windows)
            publish('loss_windows.jsonl', window)
        step = planned['global_step']
        if step - START in PAIR_UPDATES:
            check()
            entry = evaluate(step)
            monitors.append(entry); state['last_monitor'] = entry
            publish('monitor.jsonl', entry)
            guards = guard_reasons(monitors, baseline3327)
            state['guard_reasons'] = guards
            publish('guard_checks.jsonl', {'global_step': step, 'stop_reasons': guards})
            check()  # 同时发生的超时/空间停止不能被包装成普通保护停止。
            if guards:
                reason = guards[0]; break
        publish('status.json', state, append=False)
    return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': reason}


def launch_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    parser.add_argument('--parent-checkpoint-sha256')
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if not args.run_approved or args.parent_checkpoint_sha256 != PARENT_SHA or args.max_seconds != 1800:
        raise RuntimeError('Require explicit approval, exact3327 full parent and the fixed1800-second budget')
    for name, digest in [('final protocol', PROTOCOL_SHA), ('original3328 anchor', FIRST_ANCHOR_SHA)] + list(TEST_SHAS.items()):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise RuntimeError('Main-thread final review has not bound the ' + name)
    return args


def cleanup_resources(ledger, clock, log, locks):
    # 不因某个flush/close失败跳过其余资源；先关日志，再释放全部工作流锁。
    tasks = [('forward_hooks', ledger.close)]
    if clock is not None: tasks.append(('clock', clock.close))
    if log is not None: tasks.extend((('log_flush', log.flush), ('log_close', log.close)))
    tasks.extend(('workflow_lock_%d' % i, handle.close) for i, handle in enumerate(reversed(locks)))
    errors = []
    for name, close in tasks:
        try: close()
        except BaseException as error: errors.append({'resource': name, 'error': repr(error)})
    return errors


def publish_completed_after_cleanup(output, completion, budget):
    # 只有finally实际确认所有资源均清理成功，才把pending收据改为成功/普通保护停止终态。
    completion['status'] = completion.pop('intended_status')
    completion['cleanup_verified'] = True
    budget.write(Path(output) / 'completion.json', completion)


def checkpoint_common(state, config, anchor):
    last = state['last_monitor']; step = state['completed_global_step']
    return {'kind': 'acdd_mc2_paired120_checkpoint', 'global_step': step, 'epoch': 0,
            'configuration': config.as_dict(), 'effective_optimizer_lrs': dict(LRS),
            'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar',
            'mc_samples': 1 if state['arm'] == ARMS[0] else 2, 'mc_private_seed_base': mc.MC2_SEED,
            'development': last['development'] if last and last['global_step'] == step else None,
            'development_monitor_global_step': last['global_step'] if last else None,
            'run_contract_sha256': state['contract_sha256'],
            'lineage': {'parent_checkpoint_sha256': PARENT_SHA, 'parent_contract_sha256': PARENT_CANONICAL,
                        'start_after_global_step': START},
            'original_step0_guard_anchor': anchor, 'stop_reason': state['reason'],
            'execution_error': state.get('execution_error'), 'actual_optimizer_cursors': state['optimizer_cursors']}


def main(argv=None):
    started = time.monotonic()
    locks, clock, budget, log = [], None, None, None
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    state, summaries, owned, completion = {}, {}, False, None
    ledger = ForwardLedger()
    try:
        args = launch_arguments(argv)  # 审批或两项冻结SHA未齐全时，此处即拒绝，不创建输出或触碰GPU。
        clock = StopClock(1800, started); clock.install()
        external = previous.watchdog(1800)
        training.assert_rented_server(ROOT); cross.regular(DEV, directory=True)
        locks = training.acquire_runtime_locks(ROOT)
        if len(locks) != 3:
            raise RuntimeError('Require all three original workflow locks')
        budget = Budget(OUTPUT)
        disk = budget.require()
        if disk['required_bytes'] != INITIAL_REQUIRED_BYTES:
            raise RuntimeError('Initial model/metadata/reserve budget changed')
        log = replication.open_exclusive_log(OUTPUT, budget)
        sys.stdout = sys.stderr = log
        print('MC2_PREFLIGHT_STARTED', flush=True)
        if training.git_head(ROOT) != cross.HEAD or subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Require unchanged clean authoritative production source')
        if ((platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6')
                or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve()):
            raise RuntimeError('Use the original registered myconda runtime')
        if cross.flags() != cross.ORIGINAL_FLAGS:
            raise RuntimeError('Do not change the four original precision flags')
        allocated = lambda path: max(path.stat().st_size, path.stat().st_blocks * 512)
        if allocated(PARENT / 'final_full.pt') > FULL_BYTES or allocated(PARENT / 'student_step_003327.pt') > STUDENT_BYTES:
            raise RuntimeError('Declared model allocation is smaller than the real parent')
        bound = {str(PROTOCOL): PROTOCOL_SHA, str(FIRST_ANCHOR): FIRST_ANCHOR_SHA,
                 **{str(TOOLS / name): digest for name, digest in MODULE_SHAS.items()},
                 **{str(TOOLS / name): digest for name, digest in TEST_SHAS.items()},
                 str(previous.CROSS_RESULT): previous.PINNED[str(previous.CROSS_RESULT)],
                 str(previous.CROSS_IDENTITY): previous.PINNED[str(previous.CROSS_IDENTITY)]}
        cross.check_files(bound)
        first_anchor = validate_first_anchor(cross.read_json(FIRST_ANCHOR))
        cross.merge_hashes(bound, first_anchor['source_sha256']); cross.check_files(bound)
        files, contracts, receipts, development, unused_rows, config = cross.load_metadata()
        cross.merge_hashes(files, bound)
        cross.merge_hashes(files, {str(Path(__file__).resolve()): base.sha256_file(Path(__file__))})
        contract, parent = contracts[3327], receipts[3327]
        if parent['status'] != 'stopped' or parent['completed_global_step'] != START:
            raise RuntimeError('Old3327 remains stopped and is the only full parent')
        anchor = contract['original_step0_guard_anchor']
        if any(anchor[key] != value for key, value in S0_ANCHOR.items()):
            raise RuntimeError('Original S0 guard anchor changed')
        baseline3327 = parent['last_monitor']['development']
        future = future_plan(development)
        first_plan_check(future[0], first_anchor)
        clock.active(); gpu = cross.gpu_gate(); device = torch.device('cuda')
        old, batches, target, monitor_plan = cross.prepare_inputs(files, development, contracts, device)
        reference = cross.read_json(previous.CROSS_RESULT)
        if monitor_plan != cross.read_json(previous.CROSS_IDENTITY)['plan']:
            raise RuntimeError('Do not reselect fixed64 sentences, crops, t or noise')
        raw = resume.load_torch_checkpoint(cross.INPUT_FILE)
        inputs = {key: value.to(device) for key, value in raw.items()}; del raw, old
        diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
        for batch, expected in zip(batches, reference['readouts']['input_identities_after'][1:]):
            batch['q950'] = diffusion.q_sample(batch['x0'], torch.full((32,), 950, device=device, dtype=torch.long), noise=batch['source_noise'])
            if {key: cross.tensor_id(value) for key, value in batch.items()} != expected:
                raise RuntimeError('Fixed64 cached inputs changed')
        def input_identity():
            return {'old': {key: probe.tensor_sha(value) for key, value in inputs.items()},
                    'new64': [{key: cross.tensor_id(value) for key, value in batch.items()} for batch in batches],
                    'R3107': probe.tensor_sha(target)}
        input_hashes = input_identity()
        dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
        if len(dataset) != 39387:
            raise RuntimeError('Original training manifest changed')
        future_files = previous.register_future_files(dataset, future, files, clock.active)
        cross.check_files(files); clock.active(); budget.require()
        record = {'kind': 'acdd_mc2_paired120_development', 'protocol_sha256': PROTOCOL_SHA,
                  'first_original3328_anchor_sha256': FIRST_ANCHOR_SHA, 'parent_checkpoint': str(PARENT / 'final_full.pt'),
                  'parent_checkpoint_sha256': PARENT_SHA, 'parent_contract_sha256': PARENT_CANONICAL,
                  'seed': 1234, 'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS,
                  'arms': ARMS, 'maximum_updates_per_arm': UPDATES, 'start_after_global_step': START, 'stop_global_step': STOP,
                  'future_batches': future, 'future_training_feature_sha256': future_files,
                  'training_indices': development['retained_train_indices'], 'excluded_train_indices': development['excluded_train_indices'],
                  'development_contract_file_sha256': resume.DEVELOPMENT_SHA, 'monitor64_plan': monitor_plan,
                  'input_identities': input_hashes, 'file_sha256': dict(files), 'original_step0_guard_anchor': anchor,
                  'baseline3327_development': baseline3327, 'fixed_frozen_models_sha256': contract['fixed_frozen_models_sha256'],
                  'mc': {'control': 'actual production.train_batch', 'candidate': 'mean of two weighted L1 batch means',
                         'second_seed_base': mc.MC2_SEED, 'second_draw_order': ['score_noise2', 't2'], 'second_generator': 'private CUDA'},
                  'loss_windows': [[3328, 3387], [3388, 3447]], 'pair_monitor_updates': PAIR_UPDATES,
                  'uniform64_monitor_updates': UNIFORM64_UPDATES, 'forward_limits': FORWARD_LIMITS,
                  'stop_rules': ['original_S0_extreme_guard', 'two_consecutive_DEV32_above_own3327_by5percent', 'execution_safety'],
                  'no_old20_winner_gate_or_three_rising_gate': True,
                  'storage': {'control': 'S-only, not resumable', 'candidate': 'one full only, no duplicate S',
                              'system_reserve_bytes': previous.RESERVE, 'metadata_limit_bytes': previous.META_LIMIT,
                              'directory_reserve_bytes': previous.DIR_RESERVE, 'each_model_slack_bytes': previous.MODEL_SLACK,
                              'initial': disk, 'adjacent_log_exclusively_created_by_driver': True},
                  'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'flags': cross.flags(), 'gpu': gpu, 'external_timeout': external},
                  'maximum_compute_seconds': 1800, 'cleanup_only_seconds': 90,
                  'new_endpoint_inference_or_scoring_authorized': False, 'old3327_remains_stopped': True,
                  'interpretation': 'Two short60-step windows cannot establish long-term stability or CER/P808 improvement.'}
        OUTPUT.mkdir(); owned = True; budget.write(OUTPUT / 'run_contract.json', record)
        contract_sha = training.canonical_json_sha256(record)
        clock.active(); payload = resume.load_torch_checkpoint(PARENT / 'final_full.pt')
        teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
        vocoder, vconfig, _ = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
            base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
        discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed_d = base.build_paper_discriminator(ROOT, vconfig, config, device)
        old_d = resume.load_torch_checkpoint(pair.PARENT / 'latest_full.pt')
        if old_d['global_step'] != 3107 or old_d['run_contract_sha256'] != pair.PARENT_CONTRACT:
            raise RuntimeError('Original D3107 anchor changed')
        fixed_d.load_state_dict(old_d['discriminator'], strict=True); del old_d
        fixed_d.eval(); training.set_requires_grad(fixed_d, False)
        mean, std = base.load_mel_statistics(ROOT / 'data', device)
        frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
        frozen_sha = {name: pair.model_sha(model) for name, model in frozen.items()}
        if frozen_sha != contract['fixed_frozen_models_sha256']:
            raise RuntimeError('Frozen teacher/vocoder/D3107 state changed')
        models = dict(frozen, student=student, active_D=discriminator, diffusion=diffusion)
        ledger.install(student, teacher, vocoder, discriminator, fixed_d)
        def check():
            clock.active(); budget.require()
        def check_unchanged():
            pair.assert_frozen(frozen, frozen_sha)
            if input_identity() != input_hashes or cross.flags() != cross.ORIGINAL_FLAGS:
                raise RuntimeError('Fixed input or original precision changed')
        def evaluate(step):
            update = step - START
            if update not in PAIR_UPDATES:
                raise RuntimeError('Unregistered monitor call')
            check(); before = dict(ledger.counts)
            with previous.monitoring_state(models, device):
                entry = pair.monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, step)
                entry['legacy_monitor_updates_from3107'] = entry['updates']
                entry['updates'] = entry['trial_updates'] = update
                if update in UNIFORM64_UPDATES:
                    entry['uniform64'] = previous.uniform64_monitor(student, teacher, diffusion, batches, monitor_plan, config, device, clock.active)
            expected = {'teacher': 21 if update in UNIFORM64_UPDATES else 1,
                        'student': 4 if update in UNIFORM64_UPDATES else 2, 'vocoder': 2,
                        'active_D_branches': 0, 'fixed_D_branches': 16}
            if ledger.delta(before) != expected:
                raise RuntimeError('Actual monitoring forwards differ from the declared budget')
            entry['actual_forward_counts'] = ledger.delta(before)
            check_unchanged()
            return entry
        initial_restore, initial_monitor, control_rows = None, None, []
        for arm in ARMS:
            state = new_arm_state(arm, contract_sha)  # 先清新臂游标；异常不能借用上一臂完成状态。
            check(); directory = OUTPUT / arm; directory.mkdir()
            optimizer = torch.optim.Adam(student.parameters(), lr=LRS['student'], betas=(.5, .9))
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LRS['discriminator'], betas=(.5, .9))
            try:
                restored = previous.restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, contract, parent, device)
                if initial_restore is None: initial_restore = restored
                elif restored != initial_restore: raise RuntimeError('Arms did not independently restore identical3327 S/D/twoAdam/RNG')
                state.update(restore_completed=True, restore_identity=restored, status='running', phase='initial_monitor')
                state['optimizer_cursors'] = actual_cursors(optimizer, d_optimizer)
                initial = evaluate(START)
                previous.prior.validate_replay(initial, parent['last_monitor'])
                previous.validate_uniform_initial(initial['uniform64'], reference)
                if initial_monitor is None: initial_monitor = initial
                elif initial != initial_monitor: raise RuntimeError('Both independent initial monitors must exactly agree')
                state['last_monitor'] = initial
                own_baseline = dict(initial['development'])
                budget.write(directory / 'monitor.jsonl', initial, append=True)
                def step_call(planned):
                    step = planned['global_step']; state['phase'] = 'input_checks'
                    batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
                    training.seed_everything(planned['noise_seed'])
                    identity = mc.original_input_identity(batch, device, config)
                    if step == START + 1: first_input_check(identity, first_anchor)
                    if arm == ARMS[1] and step <= START + len(control_rows):
                        reference_row = control_rows[step - START - 1]
                        if identity != reference_row['input_identity'] or planned['indices'] != reference_row['indices']:
                            raise RuntimeError('Paired input differs before the next candidate D/S update')
                    receipt = {}; call = mc.train_batch_mc1 if arm == ARMS[0] else mc.train_batch_mc2
                    check()  # 预览/身份校验期间到期时，不再启动新的D/S更新。
                    before = dict(ledger.counts)
                    losses = complete_update(state, step, optimizer, d_optimizer, lambda: call(batch, teacher, student, vocoder,
                        discriminator, diffusion, optimizer, d_optimizer, mean, std, device, config, global_step=step, mc_receipt=receipt))
                    row = dict(losses, global_step=step, updates=step - START, arm=arm, indices=planned['indices'],
                               input_identity=identity, mc=receipt, actual_optimizer_cursors=state['optimizer_cursors'],
                               actual_forward_counts=ledger.delta(before),
                               comparison_scope='unpaired_extra' if arm == ARMS[1] and step > START + len(control_rows) else 'common_planned_prefix')
                    state['unpublished_step_record'] = row
                    for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
                        if receipt[key] != identity[key]: raise RuntimeError('Actual original RNG draw differs from the pre-update identity: ' + key)
                    expected = {'teacher': 1 if arm == ARMS[0] else 2, 'student': 2, 'vocoder': 3,
                                'active_D_branches': 32, 'fixed_D_branches': 0}
                    if row['actual_forward_counts'] != expected: raise RuntimeError('Actual complete-batch forward budget mismatch')
                    previous.adam_counters(optimizer, d_optimizer, step)
                    if [optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr']] != [LRS['student'], LRS['discriminator']]:
                        raise RuntimeError('Effective role learning rate changed')
                    if step == START + 1: row['original3328_loss_differences'] = first_loss_check(arm, losses, receipt, first_anchor)
                    if arm == ARMS[1] and step <= START + len(control_rows):
                        row['paired_first_loss_differences'] = paired_input_check(row, control_rows[step - START - 1])
                    return row
                def publish(name, value, append=True):
                    budget.write(directory / name, value, append=append)
                result = run_updates(state, future, step_call, evaluate, publish, check, own_baseline, initial)
                state.update(reason=result['reason'], phase='finalizing', loss_windows=result['windows'],
                             monitor_global_steps=[entry['global_step'] for entry in result['monitors']],
                             baseline3327_development=own_baseline,
                             development_first_last_difference=state['last_monitor']['development']['mel_l1'] - own_baseline['mel_l1'],
                             end_monitor_available=state['last_monitor']['global_step'] == state['completed_global_step'])
                check_unchanged()
                pair.validate_effective_adam(optimizer, d_optimizer, student, discriminator, LRS['student'], state['completed_global_step'])
                save_endpoint(directory, state, checkpoint_common(state, config, anchor), student, discriminator, optimizer, d_optimizer, budget, clock)
                check()  # 保存期间收到的停止信号仍终止后续臂，且保留已提交游标。
                state['status'] = 'completed120' if state['reason'] == 'maximum120_updates_completed' else 'protective_stop'
                state['phase'] = 'arm_complete'
                budget.write(directory / 'arm_completion.json', state); summaries[arm] = dict(state)
                if arm == ARMS[0]: control_rows = result['rows']
            except BaseException as error:
                clock.request('execution_abort')  # 非超时异常也明确进入最多90秒的保存清理阶段。
                state.update(status='execution_aborted', reason=str(error), execution_error=repr(error))
                state['execution_stop_reasons'].append(str(error))
                # 完整边界仍可保存；half/不确定游标绝不伪装成旧完整步。已提交文件绝不重试覆盖。
                if state['restore_completed'] and not state['partial_batch_may_have_updated_D'] and not (state['full_snapshot'] or state['student_snapshot']):
                    try:
                        check_unchanged()
                        save_endpoint(directory, state, checkpoint_common(state, config, anchor), student, discriminator, optimizer, d_optimizer, budget, clock)
                    except BaseException as save_error:
                        state['checkpoint_error'] = repr(save_error)
                raise
            finally:
                del optimizer, d_optimizer
        check_unchanged(); check(); cross.check_files(files)
        if training.canonical_json_sha256(cross.read_json(OUTPUT / 'run_contract.json')) != contract_sha:
            raise RuntimeError('Published contract changed')
        full120 = all(state['new_updates'] == UPDATES for state in summaries.values())
        if full120 and ledger.counts != FORWARD_LIMITS:
            raise RuntimeError('Complete paired120 run did not use exactly the registered forward budget')
        common_steps = sorted(set(summaries[ARMS[0]]['monitor_global_steps']) & set(summaries[ARMS[1]]['monitor_global_steps']))
        completion = {'status': 'finalization_pending', 'cleanup_verified': False,
                      'intended_status': 'both120_completed' if full120 else 'both_arms_reported_with_protective_stops',
                      'contract_sha256': contract_sha, 'arms': summaries, 'actual_forward_counts': dict(ledger.counts),
                      'same_length_ablation': summaries[ARMS[0]]['new_updates'] == summaries[ARMS[1]]['new_updates'],
                      'common_monitor_global_steps': common_steps, 'elapsed_seconds': time.monotonic() - started,
                      'no_new_endpoint_inference_or_scoring_performed': True, 'old3327_remains_stopped': True}
        budget.write(OUTPUT / 'completion.json', completion); check()
        print('MC2_COMPUTE_FINISHED_AWAITING_CLEANUP ' + json.dumps({'actual_forward_counts': ledger.counts}), flush=True)
    except BaseException as error:
        if budget is not None: budget.cancel_future_on_fatal_abort(repr(error))
        try:
            record = failure_record(OUTPUT if owned else None, state, summaries, error, budget, ledger)
            print('MC2_FAILED ' + json.dumps(record, ensure_ascii=False), flush=True)
        except BaseException as secondary:
            try: saved_stderr.write('MC2 failure receipt unavailable: %r; original=%r\n' % (secondary, error)); saved_stderr.flush()
            except BaseException: pass
        raise
    finally:
        original_error = sys.exc_info()[1]
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = cleanup_resources(ledger, clock, log, locks)
        if errors:
            state['cleanup_errors'] = errors
            cleanup_error = RuntimeError('MC2 resource cleanup failed: ' + repr(errors))
            try:
                if budget is not None: budget.cancel_future_on_fatal_abort(repr(cleanup_error))
                failure_record(OUTPUT if owned else None, state, summaries, original_error or cleanup_error, budget, ledger)
            except BaseException as receipt_error:
                try: saved_stderr.write('Cleanup failure receipt unavailable: %r\n' % receipt_error); saved_stderr.flush()
                except BaseException: pass
            try: saved_stderr.write(str(cleanup_error) + '\n'); saved_stderr.flush()
            except BaseException: pass
            if original_error is None: raise cleanup_error
        elif original_error is None and completion is not None:
            try:
                publish_completed_after_cleanup(OUTPUT, completion, budget)
            except BaseException as final_error:
                try: failure_record(OUTPUT if owned else None, state, summaries, final_error, budget, ledger)
                except BaseException:
                    try: saved_stderr.write('Final completion receipt failed: %r\n' % final_error); saved_stderr.flush()
                    except BaseException: pass
                raise


if __name__ == '__main__':
    main()  # 必须另行完成主线程审阅、独立复核、CPU测试与即时资源门禁；本文件不自动授权。
