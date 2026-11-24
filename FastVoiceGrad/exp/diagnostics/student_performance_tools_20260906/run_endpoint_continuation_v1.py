# 独立的端点实验分段续训入口：保持v3及旧结果冻结，不创建新的S0或放宽保护。
"""Strict continuation of endpoint-v3 student/Adam state on Matpool only.

This continues the independent endpoint-L1 experiment, not original ACDD.
An explicitly approved immutable parent checkpoint and its original contract
are required. Global update numbers, the 100000-step LR schedule, shared
teacher inputs and the original S0 reconstruction guard never restart.
"""

import argparse  # 显式给出父权重、SHA、原契约及本段有限终点。
import hashlib  # 在导入冻结v3之前核对文件字节。
import importlib.util  # 在私有空间复用冻结v3的数学和保存辅助定义。
import json  # 保存只属于本段的来源、状态和每步记录。
import math  # 检查保护读数和梯度范数的有限性。
import platform  # 保持原训练Python和PyTorch环境。
import random  # 按全局步数重建原训练轮次的确定性shuffle。
import signal  # 只在完整更新边界响应停止。
import sys  # 只导入服务器的权威项目源码。
import time  # 所有分段都具有显式时间预算。
from pathlib import Path  # 使用严格归属实验区的新输出目录。

ROOT = Path('/root/FsatVoiceGrad')
V3_SHA = '44b89afa2a9b0550d205ecf5fc40d992a54ef1244910f9327902ae646259c97b'
CONTINUATION_KIND = 'endpoint_distillation_continuation_only'
BASE_LR = 0.0001  # 只能继续已由训练批次探针选定的1e-4策略，续段不再选lr。
DECAY_STEPS = 100000  # 全局日程不随本段长度变化。

# 本地只能编辑；导入这个训练工具也先检查运行地点。
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Endpoint continuation may only run on the registered Matpool rental')
sys.path.insert(0, str(ROOT))
v3_path = Path(__file__).resolve().with_name('run_endpoint_development_v3.py')
if hashlib.sha256(v3_path.read_bytes()).hexdigest() != V3_SHA:
    raise RuntimeError('Frozen endpoint-v3 source identity changed')
specification = importlib.util.spec_from_file_location('_frozen_endpoint_v3_continuation', v3_path)
core = importlib.util.module_from_spec(specification)
specification.loader.exec_module(core)
# 下面的模块已经由冻结v3使用；不另写一套扩散或数据处理实现。
torch, training = core.torch, core.training
from fastvoicegrad.components import load_torch_checkpoint


def require_new_output(path):
    # 解析符号链接之后检查范围，旧目录即使为空也不允许续段覆盖。
    resolved = Path(path).resolve()
    if ROOT / 'exp/diagnostics' not in resolved.parents or resolved.exists():
        raise ValueError('Choose a new, nonexistent output directory under exp/diagnostics')
    return resolved


def trusted_checkpoint(path, expected_sha):
    # 明确的完整SHA是读取pickle检查点的先决条件，不能接受仅有同名latest。
    if len(expected_sha) != 64 or any(letter not in '0123456789abcdef' for letter in expected_sha):
        raise ValueError('An explicit full lowercase parent checkpoint SHA-256 is required')
    if core.sha256_file(path) != expected_sha:
        raise ValueError('Immutable parent checkpoint SHA-256 does not match')
    return load_torch_checkpoint(path)  # 文件身份通过后才反序列化可信的项目归档。


def validate_parent_payload(payload, contract):
    # 父状态只能来自v3或使用相同来源链的续段，不能混进原ACDD学生。
    if contract.get('kind') not in ('endpoint_distillation_development_only', CONTINUATION_KIND):
        raise ValueError('Parent is not the declared endpoint-v3 experiment or its continuation')
    if contract.get('objective_id') != core.OBJECTIVE_ID or payload.get('objective_id') != core.OBJECTIVE_ID:
        raise ValueError('Parent endpoint objective identity differs')
    if payload.get('kind') != 'development_checkpoint' or payload.get('epoch') != 0:
        raise ValueError('Expected a complete endpoint development checkpoint with epoch=0')
    step = payload.get('global_step')
    if type(step) is not int or not 0 <= step <= DECAY_STEPS:
        raise ValueError('Parent global step is invalid')
    canonical = training.canonical_json_sha256(contract)
    if payload.get('run_contract_sha256') != canonical:
        raise ValueError('Parent checkpoint is not bound to the supplied original contract')
    required = {'student', 'student_optimizer', 'rng_state', 'development', 'read_asset_sha256', 'actual_optimizer_learning_rate'}
    if not required.issubset(payload):
        raise ValueError('Parent lacks the full student/Adam/RNG/development state needed for continuation')
    # 源配置中的0.0002仅为不生效模板；实际Adam约束独立核对。
    template = contract.get('source_configuration_template')
    if template != core.PAPER_STUDENT_CONFIG.as_dict() or payload.get('configuration') != template:
        raise ValueError('Parent teacher/student/data interface template differs from the audited source')
    if 'learning_rate' not in payload.get('inactive_configuration_fields', []):
        raise ValueError('Parent does not explicitly mark the old template learning rate inactive')
    optimizer_contract = contract.get('optimizer', {})
    expected = {'name': 'Adam', 'lr': BASE_LR, 'betas': [0.5, 0.9], 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False, 'gradient_clipping': None, 'ema_used': False}
    if any(optimizer_contract.get(key) != value for key, value in expected.items()):
        raise ValueError('Parent optimizer differs from fixed 1e-4 Adam without clipping or EMA')
    policy = optimizer_contract.get('learning_rate_policy', {})
    if (policy.get('policy_id'), policy.get('base_learning_rate'), policy.get('decay_steps'), policy.get('formula')) != (core.OPTIMIZER_POLICY_ID, BASE_LR, DECAY_STEPS, core.LR_FORMULA):
        raise ValueError('Parent linear learning-rate policy is not the fixed 100000-update schedule')
    groups = payload['student_optimizer'].get('param_groups', [])
    if len(groups) != 1:
        raise ValueError('Expected the original single Adam parameter group')
    actual_group = groups[0]
    actual_expected = {'lr': core.learning_rate_for_step(BASE_LR, max(1, step)), 'betas': (0.5, 0.9), 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False}
    if any(actual_group.get(key) != value for key, value in actual_expected.items()):
        raise ValueError('Saved Adam parameter group does not match the parent global step')
    # 不构造新优化器也能核对checkpoint记录和真正param_groups的lr。
    saved = payload['actual_optimizer_learning_rate']
    expected_record = {'policy_id': core.OPTIMIZER_POLICY_ID, 'completed_updates': step,
        'actual_param_group_learning_rates': [actual_expected['lr']], 'base_learning_rate': BASE_LR,
        'decay_steps': DECAY_STEPS, 'formula': core.LR_FORMULA, 'gradient_clipping': None, 'ema_used': False,
        'last_update_learning_rate': actual_expected['lr'] if step else None,
        'next_update_learning_rate': core.learning_rate_for_step(BASE_LR, step + 1)}
    if any(saved.get(key) != value for key, value in expected_record.items()):
        raise ValueError('Parent recorded learning rate disagrees with its actual Adam state')
    if set(payload['rng_state']) != {'python', 'numpy', 'torch', 'cuda_all'}:
        raise ValueError('Parent RNG state is incomplete')
    if not payload['student'] or (step > 0 and not payload['student_optimizer'].get('state')):
        raise ValueError('Parent student or trained Adam state is empty')
    # Adam内部更新计数也必须承接父步数，不能只修改检查点外面的global_step标签。
    for state in payload['student_optimizer']['state'].values():
        if not {'step', 'exp_avg', 'exp_avg_sq'}.issubset(state) or float(state['step']) != step:
            raise ValueError('Saved Adam moments or internal update count are inconsistent')
    # 固定原数据、裁剪、条件和采样约定，不允许续段同时换任务。
    constants = {'batch_size': 32, 'segment_frames': 128, 'num_workers': 0, 'seed': 1234,
        'per_step_crop_seed': '1234+100000+step', 'per_step_source_noise_seed': '1234+200000+step; CUDA randn_like first draw',
        'sampler_id': core.ddim_teacher.DDIM_SAMPLER_ID,
        'objective': {'formula': 'mean(abs(y_student_clean - stop_gradient(teacher_DDIM30(z950))))',
            'endpoint_weight': 1.0, 'gan_used': False, 'vocoder_used': False, 'random_score_t_used': False},
        'coupling': 'one q(source_mel,950,noise) state shared exactly by teacher and student; no reverse noise',
        'conditioning': 'reconstruction only: original speaker embedding and source BNF',
        'student_parameterization': 'epsilon; y=(z950-sqrt(1-alpha_bar950)*epsilon_student)/sqrt(alpha_bar950)',
        'development_regression_guard': {'GT_L1_ratio_above_initial': 1.10, 'prediction_RMS_ratio_below_initial': 0.80}}
    if any(contract.get(key) != value for key, value in constants.items()):
        raise ValueError('Parent sampling, data, seeds or original guard changed')
    return step, canonical


def guard_passes(scores, original_s0):
    # 门槛始终以首次v3未更新的S0为分母，父段自身的结果不能成为新基准。
    values = [float(scores[key]) for key in ('mel_l1', 'prediction_rms')]
    anchors = [float(original_s0[key]) for key in ('mel_l1', 'prediction_rms')]
    if not all(math.isfinite(value) for value in values + anchors) or any(value <= 0 for value in anchors):
        raise ValueError('Original S0 guard anchors or parent measurements are invalid')
    return values[0] <= anchors[0] * 1.10 and values[1] >= anchors[1] * 0.80


def step_batch_indices(retained, global_step, cached=None):
    # 与v3完全相同：每个pass只用完整batch32，最后不足32句不参加更新。
    if type(global_step) is not int or global_step < 1 or len(retained) < 32:
        raise ValueError('A positive global update and at least 32 retained utterances are required')
    per_pass = len(retained) // 32
    pass_index = (global_step - 1) // per_pass
    # 首次恢复即使位于pass中段，也立即重建该pass顺序，不能引用不存在的order。
    if cached is None or cached['pass_index'] != pass_index:
        order = list(retained)
        random.Random(1234 + pass_index).shuffle(order)
        cached = {'pass_index': pass_index, 'order': order}
    offset = ((global_step - 1) % per_pass) * 32
    return cached['order'][offset:offset + 32], cached


def runtime_identity():
    # 记录本次进程真实读取的开关，不能直接把父环境声明复制成当前实测环境。
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'cpu_threads': torch.get_num_threads(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}


def monitor_history(parent_contract_path, parent_step):
    # 只需最近四个完整50批窗口及未满窗口，续段不能把已有上涨趋势清空。
    if parent_step == 0:
        return [], {}  # S0尚未完成任何批次，自然没有历史loss可读取。
    first_step = max(1, (parent_step // 50 - 4) * 50 + 1)
    found, dependencies, visited = {}, {}, set()
    current_path = Path(parent_contract_path)
    while True:
        current_path = current_path.resolve()
        if current_path in visited:
            raise ValueError('Continuation monitoring lineage contains a cycle')
        visited.add(current_path)
        contract = json.loads(current_path.read_text(encoding='utf-8'))
        dependencies[str(current_path)] = core.sha256_file(current_path)
        log_path = current_path.with_name('step_losses.jsonl')
        dependencies[str(log_path)] = core.sha256_file(log_path)
        for line in log_path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            step = item['step']
            if first_step <= step <= parent_step:
                values = {name: float(item[name]) for name in ('student_total', 'dist_raw')}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError('Parent monitoring history contains nonfinite losses')
                if step in found and found[step] != values:
                    raise ValueError('Conflicting parent step-loss histories')
                found[step] = values
        if contract.get('kind') != CONTINUATION_KIND or contract['start_global_step'] < first_step:
            break
        ancestor = contract['parent']
        current_path = Path(ancestor['run_contract'])
        if core.sha256_file(current_path) != ancestor['run_contract_file_sha256']:
            raise ValueError('Ancestor contract changed while restoring loss windows')
    if set(found) != set(range(first_step, parent_step + 1)):
        raise ValueError('Parent step-loss history cannot reconstruct the original 50-batch windows')
    return [(step, found[step]) for step in sorted(found)], dependencies


def restore_monitor(output, history, parent_step):
    # 旧日志只读，新目录保存有来源标记的完整窗口；未满50批的累加值也带入。
    windows, totals, count = [], {}, 0
    for step, values in history:
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
        if count == 50:
            windows.append({'global_step': step, 'epoch': 0, 'window_steps': 50,
                'mean': {name: value / count for name, value in totals.items()},
                'provenance': 'reconstructed from immutable parent step-loss log; no training replay'})
            totals, count = {}, 0
    for item in windows:
        core.append_json(output / 'loss_windows.jsonl', item)
    monitor = training.LossMonitor(output, window_steps=50, resume_step=parent_step)
    monitor.totals, monitor.count = totals, count
    monitor.last_global_step = parent_step
    return monitor


def resolve_origin(args, parent_contract):
    # 首次续段锁定S0归档；之后各段继承已被父contract和checkpoint绑定的同一origin。
    if parent_contract['kind'] == CONTINUATION_KIND:
        origin = dict(parent_contract['origin'])
        if args.origin_initial_checkpoint_sha256 and args.origin_initial_checkpoint_sha256 != origin['initial_checkpoint_sha256']:
            raise ValueError('Requested S0 digest differs from the already-bound origin')
    else:
        if not args.origin_initial_checkpoint_sha256:
            raise ValueError('First continuation requires --origin-initial-checkpoint-sha256 for the original S0')
        directory = args.parent_contract.parent
        origin = {'run_contract': str(args.parent_contract), 'run_contract_file_sha256': core.sha256_file(args.parent_contract),
            'run_contract_sha256': training.canonical_json_sha256(parent_contract),
            'initial_checkpoint': str(directory / 'initial_full.pt'), 'initial_checkpoint_sha256': args.origin_initial_checkpoint_sha256,
            'fixed_targets': str(directory / 'fixed_development_targets.pt'), 'fixed_targets_sha256': core.sha256_file(directory / 'fixed_development_targets.pt'),
            'initial_development': str(directory / 'initial_development.json'), 'initial_development_sha256': core.sha256_file(directory / 'initial_development.json')}
    # 逐个原文件都保持存在及相同字节，不能在续段时重新生成教师目标。
    dependencies = {origin[key]: origin[key + '_sha256'] for key in ('initial_checkpoint', 'fixed_targets', 'initial_development')}
    dependencies[origin['run_contract']] = origin['run_contract_file_sha256']
    core.verify_hashes(dependencies)
    original = json.loads(Path(origin['run_contract']).read_text(encoding='utf-8'))
    if original.get('kind') != 'endpoint_distillation_development_only' or training.canonical_json_sha256(original) != origin['run_contract_sha256']:
        raise ValueError('Origin must remain the original v3 contract, not a later segment')
    initial = trusted_checkpoint(Path(origin['initial_checkpoint']), origin['initial_checkpoint_sha256'])
    initial_step, _ = validate_parent_payload(initial, original)
    if initial_step != 0 or initial['student_optimizer']['state']:
        raise ValueError('Original guard anchor must be the pre-update S0 checkpoint')
    s0 = json.loads(Path(origin['initial_development']).read_text(encoding='utf-8'))
    if s0 != initial['development']:
        raise ValueError('Original S0 JSON differs from its archived checkpoint measurements')
    del initial  # 初始化权重仅用于验真，不能拿来覆盖需要恢复的学生。
    fixed = load_torch_checkpoint(Path(origin['fixed_targets']))
    if set(fixed) != set(original['fixed_tensor_sha256']):
        raise ValueError('Original fixed development targets have a different tensor schema')
    for name, value in fixed.items():
        if not torch.is_tensor(value) or core.tensor_sha(value) != original['fixed_tensor_sha256'][name]:
            raise ValueError('Original fixed development tensor changed: ' + name)
    # 原S0教师文件、旧v3源码和全部开发数据也都必须仍是原字节。
    dependencies.update(original['file_sha256'])
    core.verify_hashes(dependencies)
    return origin, original, s0, fixed, dependencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--parent-checkpoint-sha256', required=True)
    parser.add_argument('--parent-contract', type=Path, required=True)
    parser.add_argument('--origin-initial-checkpoint-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-global-step', type=int, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--parent-evaluation-accepted', action='store_true')
    args = parser.parse_args()
    if not args.parent_evaluation_accepted:
        raise ValueError('Explicitly review the parent development objective evaluation before continuation')
    if not 1 <= args.max_global_step <= DECAY_STEPS or args.max_seconds < 1:
        raise ValueError('Require a finite positive time budget and max-global-step <=100000')
    output = require_new_output(args.output_dir)
    args.parent_checkpoint, args.parent_contract = args.parent_checkpoint.resolve(), args.parent_contract.resolve()
    if any(ROOT / 'exp/diagnostics' not in path.parents for path in (args.parent_checkpoint, args.parent_contract)):
        raise ValueError('Parent checkpoint and contract must belong to the independent experiment area')
    training.assert_rented_server(ROOT)
    if training.git_head(ROOT) != core.EXPECTED_HEAD:
        raise RuntimeError('Audited runtime source changed')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Use the original myconda training environment')
    sampler_path = Path(core.ddim_teacher.__file__).resolve()
    if sampler_path != Path(__file__).resolve().with_name('ddim_teacher.py') or core.sha256_file(sampler_path) != core.DDIM_SHA:
        raise RuntimeError('Actual DDIM sampler import differs from the frozen original module')
    # 所有父状态和保护检查先在CPU完成；已触发原保护的父状态没有跳过开关。
    parent = trusted_checkpoint(args.parent_checkpoint, args.parent_checkpoint_sha256)
    parent_contract = json.loads(args.parent_contract.read_text(encoding='utf-8'))
    parent_step, parent_canonical = validate_parent_payload(parent, parent_contract)
    if args.max_global_step <= parent_step:
        raise ValueError('The requested global endpoint must be after the parent checkpoint')
    origin, original, s0, fixed_cpu, dependencies = resolve_origin(args, parent_contract)
    if not guard_passes(parent['development'], s0):
        raise RuntimeError('Parent already violates the unchanged original-S0 guard; continuation is prohibited')
    # 固定开发划分应由原契约file_sha里的唯一锁定文件定位，不依赖目录深度猜测。
    candidates = [Path(path) for path, value in original['file_sha256'].items() if Path(path).name == 'development_contract.json' and value == core.CONTRACT_SHA]
    if len(candidates) != 1:
        raise ValueError('Cannot identify the one original development split contract')
    development_path = candidates[0]
    development, retained, _, current_dependencies = core.validate_development(development_path)
    if retained != original['training_indices'] or development['excluded_train_indices'] != original['excluded_train_indices']:
        raise ValueError('Continuation training/exclusion indices differ from the original split')
    # 子段的任务声明必须和起始v3完全相同；只有来源、起止步数及记录位置可以变化。
    immutable_fields = ('objective', 'sampler_id', 'teacher_timesteps_descending', 'coupling', 'conditioning',
        'student_parameterization', 'training_indices', 'excluded_train_indices', 'reconstruction_dev_indices',
        'fixed_tensor_sha256', 'fixed_source_noise_sha256', 'teacher_state_sha256', 'runtime')
    if any(parent_contract[key] != original[key] for key in immutable_fields):
        raise ValueError('Parent continuation changed an original endpoint input, target or runtime contract')
    history, history_dependencies = monitor_history(args.parent_contract, parent_step)
    dependencies.update(history_dependencies)
    core.verify_hashes(parent_contract['file_sha256'])
    dependencies.update(parent_contract['file_sha256'])
    dependencies.update(current_dependencies)
    dependencies.update({str(args.parent_checkpoint): args.parent_checkpoint_sha256, str(args.parent_contract): core.sha256_file(args.parent_contract), str(Path(__file__).resolve()): core.sha256_file(Path(__file__)), str(v3_path): V3_SHA})
    asset_hashes = dict(parent['read_asset_sha256'])
    core.verify_hashes(asset_hashes)
    # 共享锁到手后才允许加载GPU模型，绝不与正在运行的训练/推理争用。
    locks = training.acquire_runtime_locks(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError('Rental CUDA GPU unavailable')
    torch.set_num_threads(2)
    training.seed_everything(1234)
    actual_runtime = runtime_identity()
    if actual_runtime != original['runtime']:
        raise RuntimeError('Actual continuation runtime/backend flags differ from the original v3 process')
    device = torch.device('cuda')
    config = core.PAPER_STUDENT_CONFIG
    teacher, student, _ = core.load_teacher_and_student(core.TEACHER, config, device)
    student.load_state_dict(parent['student'], strict=True)  # 恢复父学生，不以教师重新起训。
    optimizer = torch.optim.Adam(student.parameters(), lr=BASE_LR, betas=(0.5, 0.9), eps=1e-8, weight_decay=0, amsgrad=False)
    optimizer.load_state_dict(parent['student_optimizer'])  # 恢复真实动量与内部step，不新建空Adam历史。
    core.optimizer_learning_rate_record(optimizer, BASE_LR, DECAY_STEPS, parent_step)
    diffusion = core.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    dataset = core.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387:
        raise RuntimeError('Original training manifest length changed')
    fixed = {name: value.to(device) for name, value in fixed_cpu.items()}
    training.restore_rng_state(parent['rng_state'], device)
    # 重新测量父状态只作一致性检查；保护分母始终是最初s0。
    last_scores, resume_prediction = core.evaluate_fixed(student, diffusion, fixed)
    for key in ('mel_l1', 'endpoint_l1', 'prediction_rms'):
        if not math.isclose(last_scores[key], parent['development'][key], rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError('Restored parent reconstruction differs from its saved measurement: ' + key)
    if not guard_passes(last_scores, s0):
        raise RuntimeError('Restored parent violates the unchanged S0 guard')
    teacher_initial = {name: core.tensor_sha(value) for name, value in teacher.state_dict().items()}
    if teacher_initial != original['teacher_state_sha256']:
        raise RuntimeError('Frozen teacher differs from the original v3 teacher state')
    # 新契约显式继承固定任务定义，并把本段实际起点、终点及来源链另外记录。
    inherited = ('objective_id', 'objective', 'sampler_id', 'teacher_timesteps_descending', 'coupling', 'conditioning', 'student_parameterization', 'source_configuration_template', 'inactive_template_fields', 'source_template_learning_rate_status', 'batch_size', 'segment_frames', 'num_workers', 'seed', 'per_step_crop_seed', 'per_step_source_noise_seed', 'development_contract_file_sha256', 'development_contract_sha256', 'training_indices', 'excluded_train_indices', 'reconstruction_dev_indices', 'teacher_exposure', 'fixed_tensor_sha256', 'fixed_source_noise_sha256', 'teacher_state_sha256', 'development_regression_guard', 'source_references', 'method_boundary')
    record = {key: original[key] for key in inherited}
    optimizer_contract = dict(original['optimizer'])
    optimizer_contract['learning_rate_policy'] = core.optimizer_learning_rate_record(optimizer, BASE_LR, DECAY_STEPS, parent_step)
    record.update({'kind': CONTINUATION_KIND, 'origin': origin,
        'parent': {'checkpoint': str(args.parent_checkpoint), 'checkpoint_sha256': args.parent_checkpoint_sha256, 'run_contract': str(args.parent_contract), 'run_contract_file_sha256': core.sha256_file(args.parent_contract), 'run_contract_sha256': parent_canonical, 'global_step': parent_step},
        'optimizer': optimizer_contract, 'start_global_step': parent_step, 'max_global_step': args.max_global_step, 'max_seconds': args.max_seconds, 'check_every': 100,
        'guard_anchor': 'original v3 S0 only; never reset to this segment start', 'formal_test_used_for_selection': False,
        'automatic_model_selection': False, 'file_sha256': dependencies, 'git_head': core.EXPECTED_HEAD,
        'runtime': actual_runtime, 'parent_prediction_sha256_match': last_scores['prediction_sha256'] == parent['development']['prediction_sha256'],
        'loss_monitor_continuity': 'reconstructed last four complete 50-step windows and pending partial window from immutable parent logs',
        'parent_replay_numeric_tolerance': {'rtol': 1e-5, 'atol': 1e-6},
        'research_deviations': original['research_deviations'] + ['explicit immutable-parent continuation; original S0 guard and global LR schedule retained'],
        'time_budget_scope': 'continued student updates only; CPU validation and restored-parent replay are separate preparation'})
    core.verify_hashes(dependencies)
    output.mkdir()
    training.atomic_write_json(output / 'run_contract.json', record)
    contract_sha = training.canonical_json_sha256(record)
    training.atomic_write_json(output / 'resume_development.json', {'global_step': parent_step, 'original_s0': s0, 'restored_parent': last_scores})
    training.atomic_torch_save({'prediction': resume_prediction, 'step': parent_step}, output / 'resume_development_prediction.pt')

    def checkpoint(step, scores, resume_initial=False):
        # 所有状态都以真实global_step保存；本段第一份只是父状态复存，没有重置Adam。
        training.atomic_write_json(output / 'read_asset_sha256.json', asset_hashes)
        payload = {'kind': 'development_checkpoint', 'objective_id': core.OBJECTIVE_ID, 'student': student.state_dict(), 'student_optimizer': optimizer.state_dict(), 'global_step': step, 'epoch': 0,
            'run_contract_sha256': contract_sha, 'configuration': config.as_dict(), 'inactive_configuration_fields': ['learning_rate', 'all GAN/discriminator/score/test-loss fields'],
            'actual_optimizer_learning_rate': core.optimizer_learning_rate_record(optimizer, BASE_LR, DECAY_STEPS, step), 'development': scores,
            'rng_state': training.capture_rng_state(), 'read_asset_sha256': dict(asset_hashes), 'origin': origin, 'parent': record['parent']}
        path = output / ('resume_initial_full.pt' if resume_initial else f'student_step_{step:06d}.pt')
        if path.exists():
            raise RuntimeError('Refusing to overwrite an already saved continuation checkpoint')
        training.atomic_torch_save(payload, path)
        training.atomic_write_json(output / 'latest_checkpoint.json', {'step': step, 'path': str(path), 'sha256': core.sha256_file(path), 'run_contract_sha256': contract_sha})

    checkpoint(parent_step, last_scores, resume_initial=True)
    del parent, fixed_cpu, resume_prediction  # 父Adam状态已复制进优化器，释放不再使用的CPU副本。
    stop = {'requested': False}
    def stop_at_boundary(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, stop_at_boundary)
    signal.signal(signal.SIGINT, stop_at_boundary)
    # 保留跨段50批窗口和已有报警状态，接续点即使不是50的倍数也不重计窗口。
    monitor = restore_monitor(output, history, parent_step)
    student.train()
    cached_order, final_step, reason = None, parent_step, 'global_step_budget_complete'
    started = time.monotonic()
    try:
        for step in range(parent_step + 1, args.max_global_step + 1):
            indices, cached_order = step_batch_indices(retained, step, cached_order)
            batch, current_assets = core.batch_for(dataset, indices, 1234 + 100000 + step, asset_hashes)
            core.append_json(output / 'batch_assets.jsonl', {'step': step, 'indices': indices, 'file_sha256': current_assets})
            training.seed_everything(1234 + 200000 + step)
            mel, speaker, bnf = (batch[name].to(device) for name in ('mel', 'spk_emb', 'bnf'))
            if tuple(mel.shape) != (32, 80, 128) or tuple(bnf.shape) != (32, 144, 128) or tuple(speaker.shape) != (32, 256):
                raise RuntimeError('Continued endpoint batch violates original 32/128 feature contract')
            noise = torch.randn_like(mel)
            times = torch.full((32,), 950, dtype=torch.long, device=device)
            x_start = diffusion.q_sample(mel, times, noise).detach()
            target_started = time.monotonic()
            target = core.ddim_teacher.sample_ddim_teacher(teacher=teacher, diffusion=diffusion, x_start=x_start, speaker_embedding=speaker, bnf=bnf)
            teacher_seconds = time.monotonic() - target_started
            optimizer.zero_grad(set_to_none=True)
            loss, prediction = core.student_endpoint_loss(student=student, diffusion=diffusion, x_start=x_start, speaker_embedding=speaker, bnf=bnf, teacher_target=target)
            loss.backward()
            gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
            if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError('Continuation student gradient is missing or nonfinite')
            grad_norm = float(torch.stack([gradient.detach().float().norm() for gradient in gradients]).norm())
            if not math.isfinite(grad_norm):
                raise FloatingPointError('Continuation gradient norm is nonfinite')
            actual_lr = core.set_optimizer_learning_rate(optimizer, BASE_LR, step, DECAY_STEPS)
            optimizer.step()
            final_step = step
            losses = {'student_total': float(loss.detach()), 'dist_raw': float(loss.detach())}
            monitor.observe(step, 0, losses)
            core.append_json(output / 'step_losses.jsonl', {'step': step, 'elapsed_seconds': time.monotonic() - started, 'teacher_target_seconds': teacher_seconds, 'student_gradient_norm': grad_norm, **losses,
                'learning_rate': actual_lr, 'base_learning_rate': BASE_LR, 'lr_decay_steps': DECAY_STEPS, 'lr_formula': core.LR_FORMULA, 'optimizer_policy_id': core.OPTIMIZER_POLICY_ID,
                'loss_phase': 'pre-update student prediction; fixed teacher target', 'objective_id': core.OBJECTIVE_ID})
            del gradients, loss, prediction, target, x_start, noise, times, mel, speaker, bnf, batch
            reached_time = time.monotonic() - started >= args.max_seconds
            if step % 100 == 0 or step == args.max_global_step or reached_time or stop['requested']:
                last_scores, prediction_cpu = core.evaluate_fixed(student, diffusion, fixed)
                item = {'step': step, 'elapsed_seconds': time.monotonic() - started, **last_scores}
                core.append_json(output / 'development.jsonl', item)
                training.atomic_torch_save({'prediction': prediction_cpu, 'step': step}, output / f'development_prediction_step_{step:06d}.pt')
                checkpoint(step, last_scores)
                training.atomic_write_json(output / 'status.json', {'status': 'running', **item})
                print(json.dumps(item, ensure_ascii=False), flush=True)
                if not guard_passes(last_scores, s0):
                    reason = 'original_s0_development_regression_guard'
                    break
                if reached_time or stop['requested']:
                    reason = 'requested_stop' if stop['requested'] else 'time_budget_complete'
                    break
        if any(parameter.requires_grad or parameter.grad is not None for parameter in teacher.parameters()) or teacher_initial != {name: core.tensor_sha(value) for name, value in teacher.state_dict().items()}:
            raise RuntimeError('Frozen teacher state or gradients changed')
        core.verify_hashes(dependencies)
        core.verify_hashes(asset_hashes)
    except Exception as error:
        monitor.finish('failed', final_step, 0)
        training.atomic_write_json(output / 'status.json', {'status': 'failed', 'step': final_step, 'error_type': type(error).__name__, 'error': str(error)})
        raise
    monitor.finish(reason, final_step, 0)
    summary = {'status': reason, 'start_global_step': parent_step, 'global_step': final_step, 'elapsed_seconds': time.monotonic() - started, 'original_s0': s0, 'final': last_scores,
        'run_contract_sha256': contract_sha, 'origin': origin, 'parent': record['parent'], 'objective_id': core.OBJECTIVE_ID,
        'actual_optimizer_learning_rate': core.optimizer_learning_rate_record(optimizer, BASE_LR, DECAY_STEPS, final_step)}
    training.atomic_write_json(output / 'completion.json', summary)
    training.atomic_write_json(output / 'status.json', summary)
    print('ENDPOINT_CONTINUATION_FINISHED', reason, final_step, flush=True)


if __name__ == '__main__':
    main()  # 只有主线程审核后显式运行，导入CPU测试不会启动任何训练。
