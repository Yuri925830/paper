"""新MC2两个终点的只读FM四格；默认拒绝运行，绝不训练或评分。"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Only the registered Matpool rental may execute this diagnostic')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
OLD_FILE = TOOLS / 'inspect_acdd_fm_crossed_v1.py'
OLD_SHA = '1e4cca2b1d8f437458c158c6b838c685693fec2f8925602c5f2b880aeebab9a8'
if hashlib.sha256(OLD_FILE.read_bytes()).hexdigest() != OLD_SHA:
    raise RuntimeError('Frozen statistics helper changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import inspect_acdd_fm_crossed_v1 as old
from voicegrad.train import build_model
from fastvoicegrad.objectives import feature_matching_loss

training, baseline, resume = old.training, old.baseline, old.resume
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
OUTPUT = DEV / 'acdd_mc2_fm_cross_3447_3927_v1'
PROTOCOL = DEV / 'MC2_FM_CROSSED_3447_3927_DIAGNOSTIC_PLAN_CN.md'
PROTOCOL_SHA = '9a42c8139228a2b78a2903ec08a8183fa14ca3ce68547ac728c6258f8d2304ed'
NEW_G_ASSET_SHAS = {str(baseline.VOCODER_DIR / 'config.json'): '4b570fee140ffbda92bb217218792dca069b2d73627bc5e7bf7b1669869dd05a',
    str(baseline.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json'): '2112690afbe6fc2f7a5325d106c8dcfdd4503d7cfb09fee7675712d34b46205e'}
STEPS = (3447, 3927)
RUNS = {3447: DEV / 'acdd_mc2_paired120_from3327_v1', 3927: DEV / 'acdd_mc2_continuation480_from3447_v1'}
CHECKPOINTS = {3447: RUNS[3447] / 'candidate_mc2/final_full.pt', 3927: RUNS[3927] / 'final_full.pt'}
CHECKPOINT_SHAS = {3447: 'd108d1793a2473bb42f6b1e556c67b6339a04da8b9c8b39f0a7905bf5a26d4f6',
                  3927: '6a4f5beabcd3264e7aa016b3236f98da174af396b2ce7fb0c0b5b083f8aa2630'}
CONTRACT_SHAS = {3447: 'd939982c667e3b48228fe91b2216f93f572643f56837908a50382a0ecc069813',
                 3927: '4ba9228b4eb06f5315baa76778b63b20efb5bf309ac95ea0951d3694ad6c774e'}
COMPLETION_SHAS = {3447: '9ec17d2604adb63b692b9d03a7754af8c136ddbc1e6d3a08b008f09d334979ca',
                   3927: '22c29b860351604275fae3f9784ca158e535790443806f022e3b639bb2c800e0'}
INPUT = DEV / 'acdd_adam_descent_readonly_3107_v1/fixed_inputs.pt'
INPUT_SHA = '296a7b4a7c289f0d19409947cd2c4dcb54754ae5636a58af43e658bc12b17da8'
ANCHORS = {3447: {'mel_l1': .18723352253437042, 'rms': .9591109752655029},
           3927: {'mel_l1': .18618640303611755, 'rms': .9574350714683533}}
ORIGINAL_FLAGS = {'matmul_tf32': False, 'cudnn_tf32': True, 'benchmark': False, 'deterministic': False}
LIMITS = {'student': 2, 'vocoder': 3, 'D_branch': 48, 'D_top': 0}
RESERVE, META_LIMIT, DIR_RESERVE, FAILURE_RESERVE = 3355443200, 16 * 1024**2, 1024**2, 64 * 1024
EPSILON = 1e-8
INPUT_SHAPES = {'x0': (32, 80, 128), 'bnf': (32, 144, 128), 'spk_emb': (32, 256), 'source_noise': (32, 80, 128)}


def regular(path, directory=False):
    # 检查路径本身和所有父路径，拒绝符号链接把读写带到预定范围外。
    path = Path(path)
    if any(item.is_symlink() for item in (path,) + tuple(path.parents)):
        raise RuntimeError('Symlink path is forbidden: ' + str(path))
    if not (path.is_dir() if directory else path.is_file()): raise RuntimeError('Missing regular path: ' + str(path))


def flags():
    return {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'benchmark': torch.backends.cudnn.benchmark, 'deterministic': torch.backends.cudnn.deterministic}


def restore_flags(saved):
    torch.backends.cuda.matmul.allow_tf32 = saved['matmul_tf32']; torch.backends.cudnn.allow_tf32 = saved['cudnn_tf32']
    torch.backends.cudnn.benchmark = saved['benchmark']; torch.backends.cudnn.deterministic = saved['deterministic']


def rng_fingerprint(state):
    # NumPy状态数组必须完整取bytes；repr可能省略中间元素，不能据此认证恢复成功。
    if set(state) != {'python', 'numpy', 'torch', 'cuda_all'}: raise RuntimeError('Incomplete RNG identity')
    numpy_state = state['numpy']
    return training.canonical_json_sha256({'python': repr(state['python']),
        'numpy': {'algorithm': numpy_state[0], 'array_shape': list(numpy_state[1].shape),
                  'array_dtype': str(numpy_state[1].dtype), 'array_sha256': hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
                  'position': numpy_state[2], 'has_gauss': numpy_state[3], 'cached_gaussian': numpy_state[4]},
        'torch': old.tensor_identity(state['torch']), 'cuda_all': [old.tensor_identity(x) for x in state['cuda_all']]})


def read_json(path):
    regular(path); return json.loads(Path(path).read_text())


def verify_files(files, check=lambda: None):
    for path, expected in files.items():
        check(); regular(path)
        if baseline.sha256_file(Path(path)) != expected: raise RuntimeError('Pinned file changed: ' + path)


class Clock:
    def __init__(self):
        self.started = time.monotonic(); self.reason = None; self.stop_time = None; self.handlers = {}; self.timer = None

    def request(self, reason):
        if self.reason is None: self.reason, self.stop_time = reason, time.monotonic()

    def install(self):
        self.timer = signal.getitimer(signal.ITIMER_REAL)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            self.handlers[number] = signal.getsignal(number)
            signal.signal(number, lambda number, frame: self.request('time_or_stop_signal'))
        signal.setitimer(signal.ITIMER_REAL, max(.001, 300 - (time.monotonic() - self.started)))

    def active(self):
        if time.monotonic() - self.started >= 300: self.request('time_budget')
        if self.reason: raise TimeoutError(self.reason)

    def cleanup(self):
        if self.stop_time is not None and time.monotonic() - self.stop_time > 30: raise TimeoutError('cleanup_grace_expired')

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in self.handlers.items(): signal.signal(number, handler)
        if self.timer and self.timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(.001, self.timer[0] - (time.monotonic() - self.started)), self.timer[1])


class Budget:
    def __init__(self, output): self.output = Path(output); self.aborting = False

    def used(self):
        paths = list(self.output.rglob('*')) if self.output.exists() else []
        adjacent = Path(str(self.output) + '.log')
        if adjacent.exists(): paths.append(adjacent)
        total = 0
        for path in paths:
            regular(path, path.is_dir())
            if path.is_file(): total += path.stat().st_size
        return total

    def require(self, extra=0):
        used = self.used(); maximum = META_LIMIT if self.aborting else META_LIMIT - FAILURE_RESERVE
        if used + extra > maximum: raise RuntimeError('Complete output/log budget exhausted; never drop layers')
        stats = os.statvfs(self.output.parent); free = stats.f_bavail * stats.f_frsize
        required = RESERVE + DIR_RESERVE + (extra if self.aborting else META_LIMIT - used)
        if free < required: raise RuntimeError('Insufficient preserved disk: %d < %d' % (free, required))
        return {'available_bytes': free, 'required_bytes': required, 'used_bytes': used}

    def write(self, name, value, append=False, terminal=False):
        path = self.output / name
        if path.parent != self.output or path.suffix not in ('.json', '.jsonl'): raise RuntimeError('Only new direct JSON outputs')
        if terminal != (name == 'completion.json') or (terminal and append): raise RuntimeError('Success requires exclusive terminal publication')
        regular(self.output, True)
        data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf8')
        self.require(len(data) + 1)
        if append:
            if path.exists(): regular(path)
            with path.open('ab') as handle: handle.write(data + b'\n'); handle.flush(); os.fsync(handle.fileno())
        else:
            temporary = path.with_suffix(path.suffix + '.tmp')
            if any(p.exists() or p.is_symlink() for p in (path, temporary)): raise RuntimeError('Never overwrite existing output')
            with temporary.open('xb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.rename(temporary, path)
        # completion在rename前完成全部检查；发布成功后不再执行可能失败的后置检查。
        if not terminal: self.require()


class LimitedLog:
    # 相邻日志同样计入16MiB，不允许异常打印无限吃掉空间。
    def __init__(self, output, budget):
        path = Path(str(output) + '.log')
        if output.exists() or output.is_symlink() or path.exists() or path.is_symlink(): raise RuntimeError('Output and log must be new')
        self.budget = budget; self.handle = path.open('x', encoding='utf8')

    def write(self, text):
        self.budget.require(len(text.encode('utf8')))
        count = self.handle.write(text); self.handle.flush()
        return count
    def flush(self): self.handle.flush()
    def close(self): self.handle.close()


class Ledger:
    def __init__(self, check): self.check = check; self.counts = dict.fromkeys(LIMITS, 0); self.handles = []

    def before(self, name):
        def hook(module, arguments):
            self.check()  # 每次实际前向之前先核时间和空间，失败前向也计数。
            if self.counts[name] >= LIMITS[name]: raise RuntimeError('Forward budget exhausted: ' + name)
            self.counts[name] += 1
        return hook

    def add(self, model, name): self.handles.append(model.register_forward_pre_hook(self.before(name)))

    def close(self):
        errors = []
        for handle in reversed(self.handles):
            try: handle.remove()
            except BaseException as error: errors.append(repr(error))
        self.handles.clear()
        if errors: raise RuntimeError('Forward hook cleanup failed: ' + repr(errors))


def close_envelope(ledger, clock, log, locks, restore):
    # 一项清理报错也继续尝试其余全部清理，最后由调用方决定是否允许completion。
    tasks = [('readonly_models_rng_flags', restore), ('hooks', ledger.close)]
    if clock: tasks.append(('clock', clock.close))
    if log: tasks.extend((('log_flush', log.flush), ('log_close', log.close)))
    tasks.extend(('lock_%d' % i, lock.close) for i, lock in enumerate(reversed(locks)))
    errors = []
    for name, action in tasks:
        try: action()
        except BaseException as error: errors.append({'part': name, 'error': repr(error)})
    return errors


def restore_readonly(models, original_rng, original_flags, device, clock, receipt):
    # 即使模型核验失败，也继续恢复全局RNG与精度开关，并把每项失败一起报告。
    errors = []
    for name, (model, fingerprint) in models.items():
        try:
            if clock: clock.cleanup()
            old.assert_unchanged(model, fingerprint)
        except BaseException as error: errors.append(name + ': ' + repr(error))
    receipt['models_unchanged'] = not errors
    if original_rng is not None:
        try:
            training.restore_rng_state(original_rng, device)
            actual = training.capture_rng_state()
            receipt['rng_restored'] = rng_fingerprint(actual) == rng_fingerprint(original_rng)
            if not receipt['rng_restored']: raise RuntimeError('Original RNG failed to restore')
        except BaseException as error: errors.append('rng: ' + repr(error))
    if original_flags is not None:
        try:
            unchanged = flags() == original_flags; restore_flags(original_flags)
            receipt['flags_unchanged_and_restored'] = unchanged and flags() == original_flags
            if not receipt['flags_unchanged_and_restored']: raise RuntimeError('precision flags changed during diagnostic')
        except BaseException as error: errors.append('flags: ' + repr(error))
    if errors: raise RuntimeError('Read-only restoration failed: ' + repr(errors))


def publish_terminal(budget, clock, completion, cleanup_receipt, errors, original_error):
    # 清理阶段独立失败同样留下failure。没有全部成功，绝不创建completion。
    if errors or original_error is not None:
        budget.aborting = True
        failure = {'status': 'failed', 'cleanup_errors': errors,
                   'original_error': repr(original_error) if original_error is not None else None,
                   'readonly_cleanup': cleanup_receipt, 'automatic_retry_authorized': False}
        if not (budget.output / 'failure.json').exists(): budget.write('failure.json', failure)
        budget.write('cleanup_failure_or_abort.json', failure)
        return
    if completion is None: raise RuntimeError('Missing complete four-cell receipt')
    if not all(cleanup_receipt.values()): raise RuntimeError('Incomplete read-only cleanup cannot publish success')
    clock.active()
    completion = dict(completion, cleanup_verified=True, readonly_cleanup=cleanup_receipt)
    budget.write('completion.json', completion, terminal=True)


def close_numbers(actual, expected):
    for key, reference in expected.items():
        value = actual[key]
        if not math.isfinite(value) or abs(value - reference) > 1e-7 + 1e-6 * abs(reference):
            raise RuntimeError('Float32 replay or reduction mismatch: ' + key)


def effects(values):
    if set(values) != {'S3447_D3447', 'S3927_D3447', 'S3447_D3927', 'S3927_D3927'}:
        raise RuntimeError('All four declared combinations are required')
    if any(not math.isfinite(value) for value in values.values()): raise RuntimeError('Nonfinite crossed effect')
    a, b, c, e = (values[key] for key in ('S3447_D3447', 'S3927_D3447', 'S3447_D3927', 'S3927_D3927'))
    return {'student_change_at_D3447': b - a, 'student_change_at_D3927': e - c,
            'D_change_at_S3447': c - a, 'D_change_at_S3927': e - b, 'diagonal_change': e - a,
            'interaction': e - b - c + a, 'average_student_effect': ((b - a) + (e - c)) / 2,
            'average_D_effect': ((c - a) + (e - b)) / 2}


def branch_row(real_score, real_features, fake_score, fake_features, identity):
    if len(real_features) != 6 or len(fake_features) != 6: raise RuntimeError('Each branch must expose five hidden layers and one score')
    row = old.branch_measurement(real_score, real_features, fake_score, fake_features)
    row.update(identity)
    production = float(feature_matching_loss([real_features], [fake_features]))
    close_numbers({'fm': production}, {'fm': row['fm_raw']})
    row['production_float32_fm_raw'] = production
    row['export_sum_minus_production_float32'] = row['fm_raw'] - production
    for index, layer in enumerate(row['layers']):
        if layer['shape'][0] != 32 or len(layer['per_sample_fm_raw']) != 32: raise RuntimeError('Keep all original32 samples')
        layer.update(layer_index=index, numel_total=math.prod(layer['shape']), numel_per_sample=math.prod(layer['shape'][1:]),
                     rms_relative_l1_descriptive_only=layer['fm_raw'] / (layer['real']['rms'] + EPSILON),
                     real_mean_abs_near_zero=layer['real']['mean_abs'] <= EPSILON,
                     real_rms_near_zero=layer['real']['rms'] <= EPSILON)
    return row


def summarize_matrix(matrix):
    keys = ('fm_raw', 'fm_hidden_raw', 'fm_score_raw', 'student_adv', 'discriminator_loss_same_state')
    if set(matrix) != {'S3447_D3447', 'S3927_D3447', 'S3447_D3927', 'S3927_D3927'}: raise RuntimeError('Incomplete four-cell matrix')
    for cell in matrix.values():
        if [row['branch_index'] for row in cell['branches']] != list(range(8)): raise RuntimeError('Incomplete or reordered eight branches')
        for key in keys: cell[key] = math.fsum(row[key] for row in cell['branches'])
        cell['fm_weighted'] = 2 * cell['fm_raw']
        cell['families'] = {family: {key: math.fsum(row[key] for row in cell['branches'] if row['family'] == family) for key in keys} for family in ('MPD', 'MRD')}
        # 生产逐分支float32累加顺序与Python导出层和分别保留，不假称逐位相等。
        value = torch.zeros((), dtype=torch.float32)
        for row in cell['branches']: value.add_(row['production_float32_fm_raw'])
        cell['production_float32_fm_raw'] = float(value)
        close_numbers({'fm': cell['production_float32_fm_raw']}, {'fm': cell['fm_raw']})
    result = {'total': {key: effects({label: cell[key] for label, cell in matrix.items()}) for key in keys + ('fm_weighted',)},
              'families': {family: {key: effects({label: cell['families'][family][key] for label, cell in matrix.items()}) for key in keys} for family in ('MPD', 'MRD')},
              'branches': [], 'layers': [], 'real_activation_scale_ratios': []}
    for branch in range(8):
        result['branches'].append({'branch_index': branch, **{key: effects({label: cell['branches'][branch][key] for label, cell in matrix.items()}) for key in keys}})
        for layer in range(6):
            rows = {label: cell['branches'][branch]['layers'][layer] for label, cell in matrix.items()}
            result['layers'].append({'branch_index': branch, 'layer_index': layer,
                **{key: effects({label: row[key] for label, row in rows.items()}) for key in ('fm_raw', 'relative_l1_descriptive_only', 'rms_relative_l1_descriptive_only')}})
            for step in STEPS:
                if rows['S3447_D%d' % step]['real'] != rows['S3927_D%d' % step]['real']:
                    raise RuntimeError('One cached real activation must be independent of student choice')
            first, last = rows['S3447_D3447']['real'], rows['S3447_D3927']['real']
            result['real_activation_scale_ratios'].append({'branch_index': branch, 'layer_index': layer,
                'D3447_real': first, 'D3927_real': last,
                'mean_abs_ratio': last['mean_abs'] / (first['mean_abs'] + EPSILON),
                'rms_ratio': last['rms'] / (first['rms'] + EPSILON),
                'mean_abs_denominator_near_zero': first['mean_abs'] <= EPSILON,
                'rms_denominator_near_zero': first['rms'] <= EPSILON})
    return result


def branch_identity(branch, index):
    if index < 5:
        if branch.period != (2, 3, 5, 7, 11)[index]: raise RuntimeError('MPD period order changed')
        return {'branch_index': index, 'family': 'MPD', 'period': branch.period}
    r = branch.resolution
    return {'branch_index': index, 'family': 'MRD', 'fft_size': r.fft_size, 'hop_size': r.hop_size,
            'win_length': r.win_length, 'lrelu_slope': branch.lrelu_slope}


@torch.no_grad()
def crossed_forward(discriminator, states, real_wave, waves, check, remember, publish):
    matrix = {}
    for step in STEPS:
        check(); before = old.load_exact_weights(discriminator, states[step]['discriminator'])
        remember('discriminator', discriminator, before, 'D%d' % step)
        for student_step in STEPS: matrix['S%d_D%d' % (student_step, step)] = {'branches': []}
        branches = tuple(discriminator.loss_branches())
        if len(branches) != 8: raise RuntimeError('Exactly five MPD and three MRD branches required')
        for index, branch in enumerate(branches):
            check(); real_score, real_features = branch(real_wave)
            for student_step in STEPS:
                check(); fake_score, fake_features = branch(waves[student_step])
                row = branch_row(real_score, real_features, fake_score, fake_features, branch_identity(branch, index))
                matrix['S%d_D%d' % (student_step, step)]['branches'].append(row)
                publish({'S': student_step, 'D': step, 'branch': index, 'fm_raw': row['fm_raw']})
                del fake_score, fake_features
            del real_score, real_features
        old.assert_unchanged(discriminator, before)
    return matrix, summarize_matrix(matrix)


def select_payload(payload, step, config, contract, receipt):
    expected = (step, 0, 2, True, training.canonical_json_sha256(contract))
    actual = (payload['global_step'], payload['epoch'], payload['mc_samples'], payload['resumable'], payload['run_contract_sha256'])
    if actual != expected or payload['configuration'] != config.as_dict(): raise RuntimeError('Actual full checkpoint header/config mismatch')
    if payload['score_semantics'] != 'reconstruction_clean_x0_sqrt_alpha_bar' or payload['execution_error'] is not None:
        raise RuntimeError('Unexpected score semantics or failed checkpoint')
    if payload['development'] != receipt['last_monitor']['development'] or receipt['partial_batch_may_have_updated_D']:
        raise RuntimeError('Checkpoint disagrees with its complete endpoint receipt')
    if step == 3927 and payload['lineage']['parent_checkpoint_sha256'] != CHECKPOINT_SHAS[3447]:
        raise RuntimeError('3927 must descend from the exact3447 full state')
    return {'student': payload['student'], 'discriminator': payload['discriminator']}


def validated_context(args, clock):
    files = {str(PROTOCOL): PROTOCOL_SHA, str(OLD_FILE): OLD_SHA, str(INPUT): INPUT_SHA,
             str(Path(__file__).resolve()): args.driver_sha256, str(TOOLS / 'test_inspect_acdd_mc2_fm_cross_3447_3927_v1.py'): args.tests_sha256}
    files.update(NEW_G_ASSET_SHAS)  # 这两项为新方案追加绑定，绝不假称旧contract已包含。
    files.update({str(TOOLS / name): digest for name, digest in old.MODULE_SHAS.items()})
    for step in STEPS:
        files[str(CHECKPOINTS[step])] = CHECKPOINT_SHAS[step]
        files[str(RUNS[step] / 'run_contract.json')] = CONTRACT_SHAS[step]
        files[str(RUNS[step] / 'completion.json')] = COMPLETION_SHAS[step]
    verify_files(files, clock.active)
    contracts = {step: read_json(RUNS[step] / 'run_contract.json') for step in STEPS}
    done = {step: read_json(RUNS[step] / 'completion.json') for step in STEPS}
    receipts = {3447: done[3447]['arms']['candidate_mc2'], 3927: done[3927]['state']}
    arm_file = RUNS[3447] / 'candidate_mc2/arm_completion.json'; regular(arm_file)
    files[str(arm_file)] = baseline.sha256_file(arm_file)
    if read_json(arm_file) != receipts[3447]: raise RuntimeError('3447 arm receipt differs from root completion')
    config = baseline.replace(baseline.PAPER_STUDENT_CONFIG, num_workers=0)
    for step in STEPS:
        if (done[step]['status'] != ('both120_completed' if step == 3447 else 'completed480') or not done[step]['cleanup_verified']
                or done[step]['contract_sha256'] != training.canonical_json_sha256(contracts[step])):
            raise RuntimeError('Require original successful complete training receipts')
        receipt = receipts[step]
        if (receipt['completed_global_step'], receipt['last_full_saved_global_step'], receipt['partial_batch_may_have_updated_D']) != (step, step, False):
            raise RuntimeError('Only actual complete endpoint states may be diagnosed')
        if receipt['full_snapshot']['sha256'] != CHECKPOINT_SHAS[step] or contracts[step]['configuration'] != config.as_dict():
            raise RuntimeError('Endpoint configuration or snapshot identity changed')
        close_numbers({'mel_l1': receipt['last_monitor']['fixed_train']['train_gt_l1'], 'rms': receipt['last_monitor']['fixed_train']['train_prediction_rms']}, ANCHORS[step])
    if training.canonical_json_sha256(contracts[3447]) != '3be5667d5ab01c3def83536bc7e86e48433a08f9557a0f82d010955b4dbe51ad':
        raise RuntimeError('Canonical3447 contract mismatch')
    if contracts[3927]['parent_checkpoint_sha256'] != CHECKPOINT_SHAS[3447]: raise RuntimeError('Continuation parent mismatch')
    # 只挑本次导入的源码与G/统计资产，不继承五万份训练特征或旧实验权重。
    source_roots = ('fastvoicegrad/', 'voicegrad/', 'hifigan/custom/', 'hifigan/official/')
    selected = [Path(path) for path in contracts[3927]['file_sha256']
                if any(path.startswith(str(ROOT) + '/' + prefix) for prefix in source_roots) and path.endswith('.py')]
    selected += [baseline.VOCODER_DIR / 'generator_step_2500000.pt']
    selected += [ROOT / 'data/stats/mel_mean.npy', ROOT / 'data/stats/mel_std.npy']
    for path in selected:
        a, b = (contract['file_sha256'].get(str(path)) for contract in contracts.values())
        if not a or a != b: raise RuntimeError('Shared actual source/asset is not identically pinned: ' + str(path))
        files[str(path)] = a
    attestation = read_json(baseline.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json')
    if (attestation['status'], attestation['global_step'], attestation['generator_strict_load']) != ('verified', 2500000, True):
        raise RuntimeError('Require verified final G attestation')
    if (attestation['checkpoint_sha256'] != files[str(baseline.VOCODER_DIR / 'generator_step_2500000.pt')]
            or attestation['config_file_sha256'] != NEW_G_ASSET_SHAS[str(baseline.VOCODER_DIR / 'config.json')]):
        raise RuntimeError('New asset pins disagree with original G attestation')
    verify_files(files, clock.active)
    return files, contracts, receipts, config


def runtime_gate():
    if ((platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6')
            or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve()
            or flags() != ORIGINAL_FLAGS): raise RuntimeError('Keep original myconda and precision flags')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if processes: raise RuntimeError('GPU is in use; never overlap training/scoring')
    memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if memory < 8192 or available < 8 * 1024**3: raise RuntimeError('Need8GiB free GPU and available host memory')
    if not torch.cuda.is_available(): raise RuntimeError('Rental CUDA is unavailable')
    return {'initial_GPU_free_MiB': memory, 'initial_host_available_bytes': available}


def launch_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-approved', action='store_true'); p.add_argument('--driver-sha256'); p.add_argument('--tests-sha256')
    args = p.parse_args(argv)
    if not args.probe_approved: raise RuntimeError('Require explicit new read-only probe approval')
    for name in ('driver_sha256', 'tests_sha256'):
        digest = getattr(args, name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest): raise RuntimeError('Bind final reviewed code/test SHA')
    return args


def main(argv=None):
    locks, log, clock, budget, owned = [], None, None, None, False
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    original_rng, original_flags, device = None, None, None
    models, all_fingerprints, result, completion = {}, {}, None, None
    ledger = Ledger(lambda: (clock.active(), budget.require()))
    cleanup_receipt = {'models_unchanged': False, 'rng_restored': False, 'flags_unchanged_and_restored': False}
    try:
        args = launch_arguments(argv); clock = Clock(); clock.install()
        parent = Path('/proc') / str(os.getppid())
        command = [word.decode() for word in (parent / 'cmdline').read_bytes().split(b'\0') if word]
        if (parent / 'exe').resolve() != Path('/usr/bin/timeout').resolve() or not all(word in command for word in ('--signal=TERM', '--kill-after=30s', '300s')):
            raise RuntimeError('Require timeout --signal=TERM --kill-after=30s300s')
        training.assert_rented_server(ROOT); regular(DEV, True)
        locks = training.acquire_runtime_locks(ROOT)
        inference_lock = ROOT / 'exp/locks/text_unseen_fastvoicegrad_student_inference.lock'
        if inference_lock.is_symlink(): raise RuntimeError('Inference lock symlink forbidden')
        handle = inference_lock.open('a+')
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException: handle.close(); raise
        locks.append(handle)
        if len(locks) != 4: raise RuntimeError('All four workflow/inference locks required')
        budget = Budget(OUTPUT); disk = budget.require()
        if disk['required_bytes'] != 3373268992: raise RuntimeError('Initial preserved capacity changed')
        log = LimitedLog(OUTPUT, budget); sys.stdout = sys.stderr = log
        OUTPUT.mkdir(); owned = True
        if training.git_head(ROOT) != baseline.EXPECTED_HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Authoritative tracked code must remain unchanged')
        runtime = runtime_gate(); files, contracts, receipts, config = validated_context(args, clock)
        device = torch.device('cuda'); original_rng = training.capture_rng_state(); original_flags = flags()
        rng_identity = rng_fingerprint(original_rng)
        torch.cuda.reset_peak_memory_stats(device)
        def check(): clock.active(); budget.require()
        def remember(name, model, fingerprint, label):
            models[name] = (model, fingerprint); all_fingerprints[label] = fingerprint
        raw = resume.load_torch_checkpoint(INPUT)
        raw_ids = {key: old.tensor_sha(value) for key, value in raw.items()}
        if any(raw_ids != contract['input_identities']['old'] for contract in contracts.values()): raise RuntimeError('Fixed archive tensor identities changed')
        inputs = {key: raw[key].to(device) for key in INPUT_SHAPES}; del raw
        for key, shape in INPUT_SHAPES.items():
            if tuple(inputs[key].shape) != shape or inputs[key].dtype != torch.float32 or not torch.isfinite(inputs[key]).all(): raise RuntimeError('Original fixed input changed: ' + key)
        input_ids = {key: old.tensor_identity(value) for key, value in inputs.items()}
        states = {}
        for step in STEPS:
            check(); payload = resume.load_torch_checkpoint(CHECKPOINTS[step])
            states[step] = select_payload(payload, step, config, contracts[step], receipts[step]); del payload
        budget.write('run_contract.json', {'kind': 'readonly_mc2_FM_cross', 'protocol_sha256': PROTOCOL_SHA, 'files_sha256': files,
            'new_G_asset_pin_provenance': {'files_sha256': NEW_G_ASSET_SHAS,
                'source': 'new_preexecution_pins_approved_in_protocol_not_in_3447_or_3927_old_file_sha256'},
            'input_identities': input_ids, 'configuration': config.as_dict(), 'checkpoint_steps': STEPS,
            'input_provenance': {'train_step': 3108, 'crop_seed': 104342, 'noise_seed': 204342, 'archive_noise_reused_without_new_sampling': True},
            'forward_limits': LIMITS, 'epsilon': EPSILON, 'flags': original_flags, 'rng_identity_before': rng_identity,
            'initial_disk': disk, 'runtime': runtime, 'maximum_seconds': 300, 'cleanup_seconds': 30,
            'no_teacher_no_optimizer_no_backward_no_scoring': True})
        with torch.no_grad():
            check(); student = build_model('legacy_decimate').to(device).eval().requires_grad_(False)
            # 初始随机结构只作为半途加载失败的清理基线，不冒称它在随后有意load后仍不变。
            models['student'] = (student, old.fingerprint_model(student))
            check(); vocoder, vconfig, unused = baseline.load_frozen_vocoder(ROOT, baseline.VOCODER_DIR / 'generator_step_2500000.pt',
                baseline.VOCODER_DIR / 'config.json', baseline.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
            del unused
            remember('vocoder', vocoder, old.fingerprint_model(vocoder), 'vocoder'); ledger.add(student, 'student'); ledger.add(vocoder, 'vocoder')
            mean, std = baseline.load_mel_statistics(ROOT / 'data', device)
            diffusion = baseline.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
            diffusion.eval(); remember('diffusion', diffusion, old.fingerprint_model(diffusion), 'diffusion')
            real_wave = vocoder(training.denormalize_mel(inputs['x0'], mean, std))
            q950 = diffusion.q_sample(inputs['x0'], torch.full((32,), 950, device=device, dtype=torch.long), noise=inputs['source_noise'])
            waves, predictions = {}, {}
            for step in STEPS:
                check(); before = old.load_exact_weights(student, states[step]['student']); remember('student', student, before, 'S%d' % step)
                prediction, actual_q = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
                if not torch.equal(actual_q, q950): raise RuntimeError('Students must use identical archived q950 and conditions')
                measurements = {'mel_l1': float((prediction - inputs['x0']).abs().mean()), 'rms': float(prediction.square().mean().sqrt())}
                close_numbers(measurements, ANCHORS[step])
                predictions[step] = dict(measurements, identity=old.tensor_identity(prediction))
                waves[step] = vocoder(training.denormalize_mel(prediction, mean, std))
                old.assert_unchanged(student, before); del prediction, actual_q
            for wave in [real_wave] + list(waves.values()):
                if tuple(wave.shape) != (32, 1, 32768) or wave.dtype != torch.float32 or not torch.isfinite(wave).all(): raise RuntimeError('Waveform shape/dtype/finite contract changed')
            check(); discriminator = baseline.build_paper_discriminator(ROOT, vconfig, config, device).eval().requires_grad_(False)
            models['discriminator'] = (discriminator, old.fingerprint_model(discriminator))
            ledger.add(discriminator, 'D_top')
            for branch in discriminator.loss_branches(): ledger.add(branch, 'D_branch')
            matrix, decomposition = crossed_forward(discriminator, states, real_wave, waves, check, remember,
                lambda row: budget.write('progress.jsonl', dict(row, forward_counts=dict(ledger.counts)), append=True))
            identities = dict(input_ids, q950=old.tensor_identity(q950), real_wave=old.tensor_identity(real_wave),
                fake_S3447=old.tensor_identity(waves[3447]), fake_S3927=old.tensor_identity(waves[3927]))
        if ledger.counts != LIMITS: raise RuntimeError('Complete diagnostic requires exactly2S3V48D and zero topD')
        if {key: old.tensor_identity(value) for key, value in inputs.items()} != input_ids: raise RuntimeError('Fixed input mutation')
        for model, fingerprint in models.values(): old.assert_unchanged(model, fingerprint)
        verify_files(files, clock.active); check()
        result = {'status': 'results_ready_pending_readonly_cleanup', 'matrix': matrix, 'effects': decomposition,
            'input_and_output_identities': identities, 'train_reconstruction': predictions, 'actual_forward_counts': dict(ledger.counts),
            'model_fingerprints_before_equal_after': all_fingerprints, 'file_sha256': files,
            'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'process_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'elapsed_seconds': time.monotonic() - clock.started, 'optimizer_backward_teacher_calls': 0,
            'limits': ['One fixed TRAIN32 batch, not moving training windows or unseen speech-quality causality.',
                       'Raw and relative FM are descriptive; no new normalization or weight enters training.',
                       'Two-by-two marginals and interaction do not identify a unique causal percentage.']}
        budget.write('results.json', result); check()
        completion = {'status': 'completed_readonly_FM_cross', 'cleanup_verified': False, 'actual_forward_counts': dict(ledger.counts),
            'result_sha256': baseline.sha256_file(OUTPUT / 'results.json'), 'protocol_sha256': PROTOCOL_SHA}
    except BaseException as error:
        if clock: clock.request('execution_error')
        if budget: budget.aborting = True
        if owned:
            try: budget.write('failure.json', {'status': 'failed', 'error': repr(error), 'actual_forward_counts': dict(ledger.counts), 'automatic_retry_authorized': False})
            except BaseException as receipt_error: saved_stderr.write('Failure receipt unavailable: %r; original=%r\n' % (receipt_error, error))
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        restore = lambda: restore_readonly(models, original_rng, original_flags, device, clock, cleanup_receipt)
        errors = close_envelope(ledger, clock, log, locks, restore)
        if owned:
            try:
                publish_terminal(budget, clock, completion, cleanup_receipt, errors, original_error)
            except BaseException as error:
                budget.aborting = True
                try:
                    if not (OUTPUT / 'failure.json').exists():
                        budget.write('failure.json', {'status': 'failed', 'error': repr(error), 'cleanup_errors': errors,
                            'readonly_cleanup': cleanup_receipt, 'automatic_retry_authorized': False})
                except BaseException as receipt_error: saved_stderr.write('Terminal failure receipt unavailable: %r\n' % receipt_error)
                if original_error is None: raise
                saved_stderr.write('Terminal receipt unavailable: %r\n' % error)
        if errors and original_error is None: raise RuntimeError('Diagnostic cleanup failed: ' + repr(errors))


if __name__ == '__main__': main()
