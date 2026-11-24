"""按预先登记的时间步检查两个学生端点；只有只读前向，没有训练或评分。"""

import argparse  # 运行入口必须另行获得批准；导入模块只提供CPU测试函数。
import contextlib  # 正常结束和异常都恢复模型、随机流及精度设置。
import hashlib  # 先验辅助脚本字节，不能悄悄换成同名新实现。
import json  # 只保存小型身份和统计，不保存模型或噪声张量。
import math  # 所有报告数值均要求有限。
import os  # 排他创建输出、检查磁盘真实可用空间。
import platform
import random  # 新诊断的抽样使用独立Python随机流，不冒充历史CUDA随机流。
import signal  # 外部timeout的TERM及内部截止信号转为结构化中断，进入失败清理。
import stat
import subprocess  # 只查询GPU占用，不结束其他任务。
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
PAIR_FILE = TOOLS / 'run_acdd_paired_lr120_v1.py'
PAIR_SHA = '6bdf773152950226e8db6716ab64a977f30b3b72588322dd5aa54aee2b318bc0'
if hashlib.sha256(PAIR_FILE.read_bytes()).hexdigest() != PAIR_SHA:
    raise RuntimeError('Frozen helper identity changed')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_paired_lr120_v1 as pair

base, training, resume, probe = pair.base, pair.training, pair.resume, pair.probe
DEV = pair.DEV
PAIRED = DEV / 'acdd_lr_paired120_from3107_v1'
CURRENT = DEV / 'acdd_s5e5_staged120_from3227_v1'
PROTOCOL = DEV / 'ACDD3327_TIMESTEP_CROSS_PROTOCOL_CN.md'
OUTPUT = DEV / 'acdd_timestep_cross_readonly_3327_v1'
OLD_CROSS = DEV / 'acdd_teacher_target_cross_3227_3247_v1/results.json'
D_ANCHOR = DEV / 'acdd_d_joint_only_readonly_3327_v1/results.json'
INPUT_FILE = pair.INPUT_DIR / 'fixed_inputs.pt'
TARGET_FILE = PAIRED / 'fixed_teacher_target.pt'
HEAD = '0a2c646922704e746642f5f33c203869fe696b3f'
STEPS = (3227, 3327)  # 两个真实学生端点；不会把旧3247换标签充当3327。
STATE = {
    3227: {'path': PAIRED / 'primary_s1e4/student_step_003227.pt',
           'sha': '88da1f14234c9dcb164cdd249d47862b9ba105f1b7a1a858b4a18f9fa921b1dc',
           'contract': PAIRED / 'run_contract.json',
           'canonical': '25c6e14693a940d94e18a85f50e8e3c08bbfbf1e15f53580117662a0f86ad909',
           'receipt': PAIRED / 'primary_s1e4/arm_completion.json', 'lr': .0001},
    3327: {'path': CURRENT / 'student_step_003327.pt',
           'sha': '7bb68a0d053335a87a2c3c42684b864e27bf21a9f19d6f8d1b5bc8152955ca60',
           'contract': CURRENT / 'run_contract.json',
           'canonical': '32651a26cdf7074a9e8ca30c3cdd099697541fa4b261df757d2866f9e0622aba',
           'receipt': CURRENT / 'completion.json', 'lr': .00005},
}
PINNED = {
    str(PAIR_FILE): PAIR_SHA,
    str(PROTOCOL): 'aebb6be9c0ed9308238d73ad3031278f63a69723080835a589a814c1b6623839',
    str(PAIRED / 'run_contract.json'): 'f60a052e712080d48207be35a9595804a579ac152acdbb8c9b450108a8fde303',
    str(CURRENT / 'run_contract.json'): '787759339c886b2918bdda62abcc6818adf3cb410f923aca38c80bbd4b99b071',
    str(STATE[3227]['receipt']): '50eee0fadb5b25de06e06beec23fa2e4b4f6e9aa1ab38f6aee1128d17ce17dcb',
    str(STATE[3327]['receipt']): '68ee1cf0e680f3bf665be6dcf46e2aa75c3684262c2da3765ae2c7beebf931b2',
    str(PAIRED / 'primary_s1e4/final_full.pt'): 'f98b0763ec5a2edbd23d5d493e9dac549f85cf40cfac71e79f0b462f6fc6beb0',
    str(CURRENT / 'final_full.pt'): '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556',
    str(INPUT_FILE): '296a7b4a7c289f0d19409947cd2c4dcb54754ae5636a58af43e658bc12b17da8',
    str(TARGET_FILE): '2aedd82fb269ed07c98b206c41e071c73f64555c57ed4ce3faf1eb8db12144bb',
    str(OLD_CROSS): '4a95318d472eac6a5dad246744538f2a9c59fbb8eba81f59366a17be0cd08bca',
    str(D_ANCHOR): '17f56ae3424f5eb8ac8d0af2dbd1941d96365e0bd9a06f110e5f9e7652c89b09',
    str(CURRENT / 'step_losses.jsonl'): 'c5fd0974c6f2c3f943eb3af570014617bd19c6399c11d5bf5627287573b9ad72',
    str(CURRENT / 'monitor.jsonl'): '6c8e3e8c5369a82ac573d1816f3fb8a82c897c2ba0a34a9a26cabd22426de412',
}
PINNED.update({str(item['path']): item['sha'] for item in STATE.values()})
ORIGINAL_FLAGS = {'matmul_tf32': False, 'cudnn_tf32': True, 'benchmark': False, 'deterministic': False}
MAX_TOTAL = 5 * 1024**2  # 包括身份、结果、失败信息以及相邻启动日志。
MAX_RESULT = 2 * 1024**2
DIRECTORY_RESERVE = 1024**2  # 额外留出新目录与文件系统元数据空间。
SYSTEM_RESERVE = 3 * 1024**3 + 128 * 1024**2
INPUT_NAMES = ('x0', 'bnf', 'spk_emb', 'source_noise', 'score_noise', 'score_timesteps', 'q950')
LIMITS = [
    'Only retained TRAIN data. No formal-test forward, scoring or checkpoint selection.',
    '64 utterances are repeated over10 strata, not640 independent observations; sentinels excluded from uniform means.',
    'Endpoint3227 versus3327 does not reconstruct the missing3287 state or establish the cause of the final40 updates.',
    'A remains archivedR3107. New crossed columns areR3227 andR3327. No GAN total/adv/FM is replayed.',
    'Output derivatives hold the teacher target fixed; they are not parameter/Adam gradients or the full derivative of refreshedB.',
    'Float64 descriptive derivative reductions use actual float32 residuals/weights; large finite t1000 values alone are not a bug.',
    'Historical t values are regenerated after two exact CUDA anchors; they were not saved individually in the original step logs.',
    'Composition predictions apply historical bin fractions to fixed endpoint bin means; they are not reconstructed historical residuals.',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tensor_id(value):
    return {'sha256': probe.tensor_sha(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def merge_hashes(destination, extra):
    # 同一路径有两份不同证据时停止，不能让后读值覆盖预先冻结的SHA。
    for path, digest in extra.items():
        if path in destination and destination[path] != digest:
            raise RuntimeError('Conflicting frozen identity: ' + path)
        destination[path] = digest


def regular(path, directory=False):
    # 检查整个路径链，拒绝输出或依赖通过符号链接重定向。
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError('Absolute nonredirected path required: ' + str(path))
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise RuntimeError('Unexpected file type: ' + str(path))


def check_files(files):
    for path in files:
        regular(path)
    resume.verify_files(files)


def flags():
    return {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'benchmark': torch.backends.cudnn.benchmark, 'deterministic': torch.backends.cudnn.deterministic}


def restore_flags(values):
    torch.backends.cuda.matmul.allow_tf32 = values['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = values['cudnn_tf32']
    torch.backends.cudnn.benchmark = values['benchmark']
    torch.backends.cudnn.deterministic = values['deterministic']


class DiagnosticInterruption(RuntimeError):
    """中断只允许进入失败恢复，不能继续计算或发布成功。"""


class DeadlineSignals:
    def __init__(self, started, seconds):
        self.started, self.seconds = started, seconds
        self.reason, self.previous, self.previous_timer = None, {}, None

    def interrupt(self, number, frame):
        if self.reason is not None:
            return  # 已进入清理；最终330秒硬界限由独立GNUtimeout负责，不能取消。
        self.reason = {'signal': signal.Signals(number).name, 'elapsed_seconds': time.monotonic()-self.started}
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise DiagnosticInterruption('Diagnostic interrupted: ' + json.dumps(self.reason))

    def install(self):
        self.previous_timer = signal.getitimer(signal.ITIMER_REAL)
        for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            self.previous[number] = signal.getsignal(number)
            signal.signal(number, self.interrupt)
        signal.setitimer(signal.ITIMER_REAL, max(.001, self.started+self.seconds-time.monotonic()))

    def cleanup_only(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self.reason is None:
            self.reason = {'signal': None, 'failure_cleanup_only': True, 'elapsed_seconds': time.monotonic()-self.started}

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        if self.previous_timer is not None and self.previous_timer[0] > 0:
            remaining = max(.001, self.previous_timer[0]-(time.monotonic()-self.started))
            signal.setitimer(signal.ITIMER_REAL, remaining, self.previous_timer[1])


def timeout_parent(seconds):
    # 独立timeout进程不依赖Python能否响应信号；CUDA/I/O卡死时仍可在330秒硬退出。
    parent = Path('/proc') / str(os.getppid())
    command = [part.decode() for part in (parent/'cmdline').read_bytes().split(b'\0') if part]
    if (parent/'exe').resolve() != Path('/usr/bin/timeout').resolve() or not all(value in command for value in ('--signal=TERM', '--kill-after=30s', str(seconds)+'s')):
        raise RuntimeError('Launch through /usr/bin/timeout --signal=TERM --kill-after=30s %ss' % seconds)
    return {'pid': os.getppid(), 'command': command, 'hard_cleanup_limit_seconds': seconds+30,
            'hard_kill_cannot_guarantee_python_restoration': True}


def snapshot(model):
    # 全部named_buffers包含非持久窗口；仅state_dict不足以保证完整恢复。
    return {'parameters': {n: p.detach().cpu().clone() for n, p in model.named_parameters()},
            'buffers': {n: b.detach().cpu().clone() for n, b in model.named_buffers()},
            'modes': {n: m.training for n, m in model.named_modules()},
            'requires': {n: p.requires_grad for n, p in model.named_parameters()},
            'grads': {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in model.named_parameters()},
            'sha256': pair.model_sha(model)}


def restore_model(model, saved):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(saved['parameters'][name])
            parameter.requires_grad_(saved['requires'][name])
            old = saved['grads'][name]
            parameter.grad = None if old is None else old.to(parameter.device).clone()
        for name, buffer in model.named_buffers():
            buffer.copy_(saved['buffers'][name])
    for name, module in model.named_modules():
        module.training = saved['modes'][name]  # 逐模块恢复，避免父train()递归覆盖子模块原状态。
    actual = snapshot(model)
    if probe.tree_sha(actual) != probe.tree_sha(saved):
        raise RuntimeError('Model parameters/buffers/modes/requires_grad/grad restoration failed')


@contextlib.contextmanager
def preserved_state(models, receipt, on_failure=None):
    # models可在上下文内登记新创建对象；异常也逐一尝试恢复，报告真实成功/失败。
    saved_rng, saved_flags = training.capture_rng_state(), flags()
    originals = {}
    receipt.update({'rng_restored': False, 'flags_restored': False, 'models': {}, 'errors': []})
    def register(name, model):
        if name in models:
            raise RuntimeError('Model already registered')
        models[name] = model
        originals[name] = snapshot(model)
    try:
        yield register
    except BaseException:
        if on_failure is not None:
            on_failure()  # 错误或截止信号先切入清理状态，再尝试完整恢复，避免内部定时器二次打断。
        raise
    finally:
        for name, saved in originals.items():
            try:
                restore_model(models[name], saved)
                receipt['models'][name] = True
            except Exception as error:
                receipt['models'][name] = False
                receipt['errors'].append(name + ': ' + repr(error))
        try:
            training.restore_rng_state(saved_rng, torch.device('cuda' if saved_rng['cuda_all'] else 'cpu'))
            receipt['rng_restored'] = probe.tree_sha(training.capture_rng_state()) == probe.tree_sha(saved_rng)
        except Exception as error:
            receipt['errors'].append('rng: ' + repr(error))
        restore_flags(saved_flags)
        receipt['flags_restored'] = flags() == saved_flags
        receipt['registered_model_sets_match'] = set(models) == set(originals)  # 只证明已登记对象的集合一致，不推断失败构造器内部对象。
        print('TIMESTEP_RESTORED ' + json.dumps(receipt), flush=True)
        if receipt['errors'] or not receipt['rng_restored'] or not receipt['flags_restored'] or not all(receipt['models'].values()):
            raise RuntimeError('Read-only state restoration failed: ' + json.dumps(receipt))


class Budget:
    def __init__(self, output, started, seconds):
        self.output, self.started, self.seconds = Path(output), started, seconds

    def used(self):
        files = list(self.output.iterdir()) if self.output.exists() else []
        log = Path(str(self.output) + '.log')
        if log.exists():
            files.append(log)
        for path in files:
            regular(path)
        return sum(path.stat().st_size for path in files)

    def check(self, extra=0, enforce_time=True):
        used = self.used()
        if used + extra > MAX_TOTAL:
            raise RuntimeError('Total identity/results/failure/log budget exceeds5MiB')
        available = pair.available_bytes(self.output.parent)
        required = SYSTEM_RESERVE + DIRECTORY_RESERVE + MAX_TOTAL - used
        if available < required:
            raise RuntimeError('Insufficient reserve: available=%s required=%s' % (available, required))
        if enforce_time and time.monotonic() - self.started > self.seconds:
            raise RuntimeError('Entire diagnostic exceeded its predeclared time limit')
        return {'available_bytes': available, 'required_bytes': required, 'written_bytes': used}

    def write(self, name, value, enforce_time=True):
        # 每个文件仅写一次；失败临时文件保留，不覆盖旧实验或把半写入当成完成。
        data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if name == 'results.json' and len(data) > MAX_RESULT:
            raise RuntimeError('Complete result exceeds2MiB; do not discard conditions to make it fit')
        self.check(len(data), enforce_time)
        path, temporary = self.output / name, self.output / (name + '.tmp')
        if path.exists() or temporary.exists():
            raise RuntimeError('Refuse an existing result or partial output')
        with temporary.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.check(0, enforce_time)
        os.rename(temporary, path)
        return hashlib.sha256(data).hexdigest()


def choose_plan(retained, old_indices):
    if retained != sorted(set(retained)) or len(retained) != 39335 or len(set(old_indices)) != 32:
        raise RuntimeError('Original retained population or TRAIN3108 batch is invalid')
    pool = [index for index in retained if index not in set(old_indices)]
    selected = random.Random(73073328).sample(pool, 64)
    rng = random.Random(73073341)
    strata = [[rng.randint(100 * layer + 1, 100 * layer + 100) for _ in range(64)] for layer in range(10)]
    return {'selected_indices': selected, 'strata_timesteps': strata, 'sentinel_timesteps': [1, 950, 999, 1000],
            'selection_seed': 73073328, 'crop_seeds': [73073329, 73073330], 'noise_seeds': [73073331, 73073332],
            'timestep_seed': 73073341, 'student_start_step': 950}


def validate_step_rows(rows, retained):
    if [row['global_step'] for row in rows] != list(range(3228, 3328)):
        raise RuntimeError('Require all100 original completed steps in original order')
    planner = resume.GlobalBatchPlan(retained)
    for row in rows:
        if row['indices'] != planner.for_step(row['global_step']):
            raise RuntimeError('Historical batch indices differ from the absolute-step plan')
        pair.finite_losses(row)  # 已完成驱动把七项loss平铺在每行，未包在losses子字典中。


def draw_original(step, device):
    # 必须按生产CUDA随机调用顺序；CPU只用于小测试，main强制CUDA并核两个真实锚点。
    training.seed_everything(1234 + 200000 + step)
    template = torch.empty((32, 80, 128), dtype=torch.float32, device=device)
    source, score = torch.randn_like(template), torch.randn_like(template)
    t = torch.randint(1, 1001, (32,), device=device, dtype=torch.long)
    return {'source_noise': source, 'score_noise': score, 'score_timesteps': t}


def historical_timesteps(raw, d_anchor, rows, budget, device):
    anchors = {}
    for step, wanted in ((3108, {k: tensor_id(raw[k]) for k in ('source_noise', 'score_noise', 'score_timesteps')}),
                         (3328, d_anchor['input_identities'])):
        budget.check()
        drawn = draw_original(step, device)
        actual = {key: tensor_id(value) for key, value in drawn.items()}
        if any(actual[key] != wanted[key] for key in actual):
            raise RuntimeError('Historical CUDA RNG anchor did not replay at step' + str(step))
        anchors[str(step)] = actual
    records = []
    for row in rows:
        budget.check()
        t = draw_original(row['global_step'], device)['score_timesteps'].cpu().tolist()
        records.append({'global_step': row['global_step'], 'timesteps': t})
    windows = []
    for offset in range(0, 100, 20):
        counts = [0] * 10
        for row in records[offset:offset + 20]:
            for t in row['timesteps']:
                counts[(t - 1) // 100] += 1
        if sum(counts) != 640:
            raise RuntimeError('Each historical20-step window must contain640 timesteps')
        windows.append({'first_global_step': 3228 + offset, 'last_global_step': 3247 + offset,
                        'counts': counts, 'fractions': [count / 640 for count in counts]})
    return {'anchors': anchors, 'regenerated_not_original_logged': True, 'steps': records, 'windows': windows}


def derivative_statistics(residual, weights, batch_divisor=1, layer_divisor=1, coefficient=1):
    # 固定R的每句L1导数；零残差取0。这里只做代数统计，绝不backward。
    if residual.ndim != 3 or weights.shape != (residual.shape[0],) or min(batch_divisor, layer_divisor) <= 0:
        raise RuntimeError('Invalid derivative dimensions or averaging divisors')
    if not torch.isfinite(residual).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise RuntimeError('Derivative statistics require finite residuals and nonnegative weights')
    n = residual[0].numel()
    q = residual.ne(0).flatten(1).double().mean(1)
    scale = coefficient / (batch_divisor * layer_divisor)
    w = weights.double() * scale
    gradient = residual.sign().double() * w[:, None, None] / n
    l2, rms = gradient.flatten(1).norm(dim=1), gradient.square().flatten(1).mean(1).sqrt()
    return {'nonzero_fraction': q.cpu().tolist(), 'l2': l2.cpu().tolist(), 'rms': rms.cpu().tolist(),
            'expected_l2': (w * (q / n).sqrt()).cpu().tolist(), 'expected_rms': (w * q.sqrt() / n).cpu().tolist()}


def cell_readout(prediction, target, weights, include_derivative=True):
    if prediction.shape != target.shape or prediction.ndim != 3 or weights.shape != (prediction.shape[0],):
        raise RuntimeError('Crossed tensors have inconsistent dimensions')
    residual = prediction - target  # 保留真实float32减法；描述性导数范数稍后用float64汇总。
    if not all(bool(torch.isfinite(value).all()) for value in (prediction, target, residual, weights)):
        raise RuntimeError('Nonfinite crossed input or residual')
    per = residual.abs().flatten(1).mean(1)
    weighted = per * weights
    result = {'unweighted': float(per.mean()), 'weighted': float(weighted.mean()), 'weighted45': float(weighted.mean()) * 45,
              'per_unweighted': per.cpu().tolist(), 'per_weighted': weighted.cpu().tolist(),
              'per_weighted45': (weighted.double() * 45).cpu().tolist()}
    if include_derivative:
        result['fixed_target_output_derivative_per_sentence'] = derivative_statistics(residual, weights)
    return result  # R与R的漂移只报告误差，避免把对R的导数误读成学生输出导数。


def decomposition(c00, c10, c01, c11):
    # 两种先后顺序及其对称平均同时保留；只解释端点四格数值，不声称因果。
    return {'diagonal_change': c11 - c00, 'student_at_R3227': c10 - c00,
            'target_at_y3327': c11 - c10, 'target_at_y3227': c01 - c00,
            'student_at_R3327': c11 - c01, 'symmetric_student': .5 * (c10 - c00 + c11 - c01),
            'symmetric_target': .5 * (c01 - c00 + c11 - c10), 'interaction': c11 - c10 - c01 + c00}


def crossed(predictions, targets, weights):
    cells = {f'y{i}_R{j}': cell_readout(predictions[i], targets[j], weights) for i in STEPS for j in STEPS}
    order = ('y3227_R3227', 'y3327_R3227', 'y3227_R3327', 'y3327_R3327')
    result = {}
    for metric in ('weighted', 'unweighted'):
        result[metric] = decomposition(*(cells[key][metric] for key in order))
        per = [decomposition(*(cells[key]['per_' + metric][i] for key in order)) for i in range(len(weights))]
        result[metric]['per_utterance'] = {name: [row[name] for row in per] for name in per[0]}
    return {'cells': cells, 'decomposition': result,
            'target_drift': cell_readout(targets[3327], targets[3227], weights, include_derivative=False)}


def require_close(actual, expected, atol, rtol=0):
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol):
        raise RuntimeError('Frozen numerical replay tolerance failed: actual=%r expected=%r' % (actual, expected))
    return abs(actual - expected)


def validate_payload(payload, step, config):
    item = STATE[step]
    if payload['global_step'] != step or payload['run_contract_sha256'] != item['canonical']:
        raise RuntimeError('Student step/contract identity mismatch')
    if payload['configuration'] != config.as_dict() or payload['score_semantics'] != 'reconstruction_clean_x0_sqrt_alpha_bar':
        raise RuntimeError('Student objective/configuration mismatch')
    if payload['effective_optimizer_lrs'] != {'student': item['lr'], 'discriminator': .0002}:
        raise RuntimeError('Archived effective learning-rate identity mismatch')


def load_metadata():
    files = dict(PINNED)
    check_files(files)
    contracts, receipts = {}, {}
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    for step, item in STATE.items():
        contract, receipt = read_json(item['contract']), read_json(item['receipt'])
        if training.canonical_json_sha256(contract) != item['canonical'] or contract['configuration'] != config.as_dict():
            raise RuntimeError('Frozen contract or mathematical configuration changed')
        if receipt['completed_global_step'] != step or receipt['student_snapshot']['sha256'] != item['sha']:
            raise RuntimeError('Completion does not identify the selected actual student')
        if receipt['full_snapshot']['sha256'] != files[receipt['full_snapshot']['path']]:
            raise RuntimeError('Complete checkpoint lineage is not pinned')
        merge_hashes(files, contract['file_sha256'])
        contracts[step], receipts[step] = contract, receipt
    if contracts[3327]['lineage']['parent_contract_sha256'] != STATE[3227]['canonical']:
        raise RuntimeError('3327 is not the registered continuation of3227')
    development, dependencies = resume.load_development(DEV / 'development_contract.json')
    merge_hashes(files, dependencies)
    for contract in contracts.values():
        if contract['training_indices'] != development['retained_train_indices'] or contract['excluded_train_indices'] != development['excluded_train_indices']:
            raise RuntimeError('Retained training population differs from the frozen52 exclusions')
    if training.collect_source_fingerprints(ROOT) != read_json(resume.PARENT / 'run_contract.json')['source_sha256']:
        raise RuntimeError('Authoritative production source changed')
    rows = [json.loads(line) for line in (CURRENT / 'step_losses.jsonl').read_text().splitlines()]
    validate_step_rows(rows, development['retained_train_indices'])
    if (config.score_target, config.score_weight, config.lambda_fm, config.lambda_dist) != ('clean_x0', 'sqrt_alpha_bar', 2, 45):
        raise RuntimeError('Keep the unchanged objective and2/45 coefficients')
    return files, contracts, receipts, development, rows, config


def prepare_inputs(files, development, contracts, device):
    raw = resume.load_torch_checkpoint(INPUT_FILE)
    identities = {key: tensor_id(value)['sha256'] for key, value in raw.items()}
    if identities != contracts[3227]['input_sha256'] or identities != contracts[3327]['fixed_train_inputs_sha256']:
        raise RuntimeError('Fixed input archive differs between the two real runs')
    old = {key: raw[key].to(device) for key in INPUT_NAMES}
    old_indices = resume.GlobalBatchPlan(development['retained_train_indices']).for_step(3108)
    plan = choose_plan(development['retained_train_indices'], old_indices)
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment='legacy_length_linear')
    if len(dataset) != 39387:
        raise RuntimeError('Original dataset indexing changed')
    records, batches = [], []
    for b in range(2):
        indices = plan['selected_indices'][32 * b:32 * (b + 1)]
        for index in indices:
            item = dataset.items[index]
            paths = {key: str(item[key]) for key in ('mel_path', 'bnf_path', 'spk_emb_path')}
            for path in paths.values():
                regular(path)
                merge_hashes(files, {path: base.sha256_file(Path(path))})
            records.append({'original_index': index, 'speaker': item['speaker'], 'utt_id': item['utt_id'], 'feature_paths': paths})
        batch = base.batch_for(dataset, indices, plan['crop_seeds'][b])
        inputs = {'x0': batch['mel'].to(device), 'bnf': batch['bnf'].to(device), 'spk_emb': batch['spk_emb'].to(device)}
        training.seed_everything(plan['noise_seeds'][b])
        inputs['source_noise'], inputs['score_noise'] = torch.randn_like(inputs['x0']), torch.randn_like(inputs['x0'])
        batches.append(inputs)
    for inputs in [old] + batches:
        for name, shape in [('x0', (32, 80, 128)), ('bnf', (32, 144, 128)), ('spk_emb', (32, 256)), ('source_noise', (32, 80, 128)), ('score_noise', (32, 80, 128))]:
            if inputs[name].shape != shape or inputs[name].dtype != torch.float32 or not bool(torch.isfinite(inputs[name]).all()):
                raise RuntimeError('Invalid fixed normalized input: ' + name)
    target = resume.load_torch_checkpoint(TARGET_FILE)
    old_target = target['teacher_target'].to(device)
    if target['global_source_step'] != 3107 or target['sha256'] != contracts[3227]['fixed_teacher_target_sha256'] or probe.tensor_sha(old_target) != target['sha256']:
        raise RuntimeError('OldA must remain the exact archivedR3107 tensor')
    plan.update({'old_TRAIN3108_indices': old_indices, 'utterances': records, 'speaker_counts': dict(Counter(row['speaker'] for row in records)),
                 'input_identities': [{key: tensor_id(value) for key, value in inputs.items()} for inputs in [old] + batches]})
    return old, batches, old_target, plan


def gpu_gate():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if any(line.strip() != str(os.getpid()) for line in rows.splitlines() if line.strip()):
        raise RuntimeError('Another GPU process is active')
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    if len(free) != 1 or int(free[0]) < 24576:
        raise RuntimeError('Require the original single GPU with at least24GiB free')
    return {'compute_pids': rows, 'free_mib': int(free[0])}


@torch.no_grad()
def compute(models, register, old, batches, old_target, plan, config, receipts, budget, device):
    teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
    register('teacher', teacher)
    register('student', student)
    diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    register('diffusion', diffusion)
    teacher.eval()
    student.train()
    training.set_requires_grad(teacher, False)
    training.set_requires_grad(student, False)
    counts, predictions, targets, old_states = {'student': 0, 'teacher': 0}, {}, {}, {}
    old_receipt = read_json(OLD_CROSS)['states']['3227']
    conditions = [('stratum_' + str(i), row) for i, row in enumerate(plan['strata_timesteps'])]
    conditions += [('sentinel_' + str(t), [t] * 64) for t in plan['sentinel_timesteps']]
    for step in STEPS:
        budget.check()
        payload = resume.load_torch_checkpoint(STATE[step]['path'])
        validate_payload(payload, step, config)
        student.load_state_dict(payload['student'], strict=True)
        del payload
        state_before = {name: pair.model_sha(model) for name, model in models.items()}
        predictions[step], targets[step] = [], []
        for batch_index, inputs in enumerate([old] + batches):
            budget.check()
            prediction, q950 = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
            counts['student'] += 1
            if prediction.shape != inputs['x0'].shape or not bool(torch.isfinite(prediction).all()) or prediction.requires_grad:
                raise RuntimeError('Invalid no-gradient student prediction')
            if 'q950' in inputs and not torch.equal(q950, inputs['q950']):
                raise RuntimeError('Sourceq950 changed across students or old input')
            inputs['q950'] = q950  # 首个学生登记q950，第二个必须逐字节复现。
            predictions[step].append(prediction)
            targets[step].append({})
            local = [('old3108', old['score_timesteps'].cpu().tolist())] if batch_index == 0 else [(name, ts[32 * (batch_index - 1):32 * batch_index]) for name, ts in conditions]
            for name, ts in local:
                budget.check()
                t = torch.tensor(ts, dtype=torch.long, device=device)
                loss, target = training.score_distillation_loss(prediction, teacher, diffusion, t, inputs['score_noise'], inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
                counts['teacher'] += 1
                weights = diffusion.sqrt_alphas_cumprod[t]
                values = cell_readout(prediction, target, weights)
                require_close(values['weighted'], float(loss), 1e-7, 1e-6)
                targets[step][batch_index][name] = target
                if target.shape != prediction.shape or target.requires_grad:
                    raise RuntimeError('Teacher target shape or stop-gradient boundary changed')
                print('TIMESTEP_FORWARD S=%s batch=%s condition=%s calls=%s' % (step, batch_index, name, counts), flush=True)
            if batch_index == 0:
                target, w = targets[step][0]['old3108'], diffusion.sqrt_alphas_cumprod[old['score_timesteps']]
                a, b = cell_readout(prediction, old_target, w), cell_readout(prediction, target, w)
                drift = cell_readout(target, old_target, w, include_derivative=False)
                actual = {'diagnostic_frozen_target_score': a['weighted'], 'primary_moving_score': b['weighted'],
                          'teacher_target_weighted_l1_drift': drift['weighted'], 'train_gt_l1': float((prediction-old['x0']).abs().mean()),
                          'train_prediction_rms': float(prediction.square().mean().sqrt())}
                errors = {key: require_close(value, receipts[step]['last_monitor']['fixed_train'][key], 2e-5) for key, value in actual.items()}
                if step == 3227 and (tensor_id(prediction) != old_receipt['prediction'] or tensor_id(target) != old_receipt['teacher_target']):
                    raise RuntimeError('S3227 archived prediction/teacher target did not replay exactly')
                old_states[str(step)] = {'A_oldR3107': a, 'B_refreshed': b, 'target_drift': drift, 'actual': actual, 'replay_absolute_error': errors,
                                         'prediction': tensor_id(prediction), 'target': tensor_id(target)}
        if {name: pair.model_sha(model) for name, model in models.items()} != state_before:
            raise RuntimeError('Forward changed a parameter or persistent/nonpersistent buffer')
    if counts != {'student': 6, 'teacher': 58}:
        raise RuntimeError('Do not change the predeclared6S/58T forward budget')
    result_conditions, gt = {}, {}
    for step in STEPS:
        y, x = torch.cat(predictions[step][1:]), torch.cat([inputs['x0'] for inputs in batches])
        gt[str(step)] = {'per_utterance_l1': (y-x).abs().flatten(1).mean(1).cpu().tolist(),
                         'per_utterance_rms': y.square().flatten(1).mean(1).sqrt().cpu().tolist(),
                         'l1': float((y-x).abs().mean()), 'rms': float(y.square().mean().sqrt())}
    for name, ts in conditions:
        budget.check()
        t = torch.tensor(ts, dtype=torch.long, device=device)
        y = {step: torch.cat(predictions[step][1:]) for step in STEPS}
        r = {step: torch.cat([targets[step][b][name] for b in (1, 2)]) for step in STEPS}
        values = crossed(y, r, diffusion.sqrt_alphas_cumprod[t])
        values.update({'timesteps': ts, 't_identity': tensor_id(t), 'alpha_bar': diffusion.alphas_cumprod[t].cpu().tolist(),
                       'sqrt_alpha_bar': diffusion.sqrt_alphas_cumprod[t].cpu().tolist(),
                       'inverse_sqrt_alpha_bar': diffusion.sqrt_alphas_cumprod[t].reciprocal().cpu().tolist(),
                       'target_identities': {str(step): tensor_id(r[step]) for step in STEPS}})
        result_conditions[name] = values
    return {'actual_forward_counts': counts, 'old_TRAIN3108': old_states, 'new_TRAIN64_GT': gt,
            'conditions': result_conditions, 'input_identities_after': [{key: tensor_id(value) for key, value in inputs.items()} for inputs in [old] + batches],
            'student_prediction_identities': {str(step): [tensor_id(y) for y in predictions[step]] for step in STEPS},
            'fixed_target_output_derivative_scaling': {'per_sentence_N': 10240, 'actual_training_batch32_score': 1/32,
                'actual_training_batch32_45score': 45/32, 'diagnostic64_sentence_mean': 1/64,
                'diagnostic64_by10_uniform_mean': 1/640, 'diagnostic64_by10_uniform_45mean': 45/640}}


def aggregate(result, history):
    # 十层等权主均值，四个预设哨兵始终另列；历史频率只形成固定模型的组成预测。
    strata = [result['conditions']['stratum_' + str(i)] for i in range(10)]
    names = list(strata[0]['cells'])
    result['ten_stratum_uniform_means'] = {name: {metric: sum(row['cells'][name][metric] for row in strata)/10 for metric in ('weighted', 'unweighted', 'weighted45')} for name in names}
    for window in history['windows']:
        window['fixed_endpoint_composition_predictions_not_historical_losses'] = {
            name: sum(row['cells'][name]['weighted'] * fraction for row, fraction in zip(strata, window['fractions'])) for name in names}


def main():
    started = time.monotonic()  # 300秒包括门禁、文件哈希、模型、恢复和发布。
    restoration, models, owned, locks, deadline, budget = {}, {}, False, [], None, None
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--run-approved', action='store_true')
        parser.add_argument('--max-seconds', type=int, default=300)
        args = parser.parse_args()
        if not args.run_approved or not 1 <= args.max_seconds <= 300:
            raise RuntimeError('Require separate approval and at most300 seconds')
        deadline = DeadlineSignals(started, args.max_seconds)
        deadline.install()
        watchdog = timeout_parent(args.max_seconds)
        training.assert_rented_server(ROOT)
        regular(DEV, directory=True)
        if OUTPUT.exists() or OUTPUT.is_symlink():
            raise RuntimeError('Refuse any existing or redirected output')
        locks = training.acquire_runtime_locks(ROOT)  # finally显式关闭，不只依赖进程退出释放。
        if training.git_head(ROOT) != HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Frozen source must retain its clean recordedHEAD')
        if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6') or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve():
            raise RuntimeError('Use the original registered myconda runtime')
        if flags() != ORIGINAL_FLAGS:
            raise RuntimeError('Do not change the four original precision flags')
        budget = Budget(OUTPUT, started, args.max_seconds)
        disk, gpu = budget.check(), gpu_gate()
        files, contracts, receipts, development, rows, config = load_metadata()
        merge_hashes(files, {str(Path(__file__).resolve()): base.sha256_file(Path(__file__))})
        with preserved_state(models, restoration, deadline.cleanup_only) as register:
            device = torch.device('cuda')
            old, batches, old_target, plan = prepare_inputs(files, development, contracts, device)
            check_files(files)
            budget.check()
            identity = {'kind': 'readonly_timestep_cross_3227_3327', 'protocol_sha256': PINNED[str(PROTOCOL)],
                        'file_sha256': dict(files), 'plan': plan, 'flags': flags(), 'config': config.as_dict(),
                        'state_roles': {str(step): {'student_checkpoint': str(item['path']), 'student_sha256': item['sha'], 'contract_canonical': item['canonical']} for step, item in STATE.items()},
                        'old_target': tensor_id(old_target), 'max_seconds': args.max_seconds, 'forward_budget': {'student': 6, 'teacher': 58},
                        'initial_disk': disk, 'initial_gpu': gpu, 'external_watchdog': watchdog, 'limitations': LIMITS}
            OUTPUT.mkdir()  # 仅在预检通过后创建新目录；此后才能写属于本次诊断的失败记录。
            owned = True
            identity_sha = budget.write('run_identity.json', identity)
            # 身份已绑定64索引/t/所有文件；下面才开始历史CUDA锚点与模型读数。
            history = historical_timesteps(old, read_json(D_ANCHOR), rows, budget, device)
            computed = compute(models, register, old, batches, old_target, plan, config, receipts, budget, device)
            aggregate(computed, history)
            for before, after in zip(plan['input_identities'], computed['input_identities_after']):
                if any(after[key] != value for key, value in before.items()):
                    raise RuntimeError('A fixed input tensor changed during the diagnostic')
            if tensor_id(old_target) != identity['old_target'] or flags() != ORIGINAL_FLAGS:
                raise RuntimeError('Old target or precision flags changed')
        check_files(files)
        check_files(PINNED)  # 再钉住原静态证据，防止后续provenance覆盖先验值。
        if base.sha256_file(OUTPUT / 'run_identity.json') != identity_sha:
            raise RuntimeError('Published prospective identity changed')
        result = {'status': 'completed', 'run_identity_file_sha256': identity_sha, 'historical_t_replay': history,
                  'readouts': computed, 'restoration': restoration, 'flags_after': flags(),
                  'elapsed_seconds_before_publication': time.monotonic()-started, 'limits': LIMITS,
                  'optimizer_created': False, 'backward_called': False, 'model_files_written': False,
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
        digest = budget.write('results.json', result)
        print('TIMESTEP_COMPLETED ' + json.dumps({'sha256': digest, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    except BaseException as error:
        if deadline is not None:
            deadline.cleanup_only()  # 从此只写失败证据和关锁，不进行前向或成功发布。
        failure = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'restoration': restoration,
                   'interruption': None if deadline is None else deadline.reason,
                   'model_files_not_written_by_this_diagnostic': True, 'elapsed_seconds': time.monotonic()-started,
                   'output_directory_created_by_this_run': owned}
        print('TIMESTEP_FAILED ' + json.dumps(failure, ensure_ascii=False), flush=True)
        if owned:
            try:
                budget.write('failure.json', failure, enforce_time=False)  # 超时仍尝试留下失败证据，但不绕过容量保护。
            except Exception as second:
                print('FAILURE_RECORD_NOT_SAVED ' + repr(second), flush=True)
        raise
    finally:
        if deadline is not None:
            deadline.close()
        closed = []
        for handle in reversed(locks):
            try:
                handle.close()
                closed.append(True)
            except Exception as error:
                closed.append(False)
                print('LOCK_CLOSE_FAILED ' + repr(error), flush=True)
        print('TIMESTEP_LOCKS_RELEASED ' + json.dumps({'handles': len(locks), 'closed': closed}), flush=True)


if __name__ == '__main__':
    main()
