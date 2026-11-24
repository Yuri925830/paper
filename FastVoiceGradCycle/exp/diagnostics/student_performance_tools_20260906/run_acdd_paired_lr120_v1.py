# 两组均从同一个完整3107状态恢复；这次只比较学生Adam学习率，不改变生产损失或教师。
"""Paired, bounded ACDD continuation: original S LR versus half S LR."""

import argparse  # 只有主任务明确批准后才能执行GPU训练。
import contextlib  # 固定监测前后恢复随机状态，避免改变下一批训练。
import copy  # 恢复时复制CPU优化器状态，避免两组共享可修改的moment。
import hashlib  # 导入前锁定已审核辅助代码的字节内容。
import json  # 每步原始损失与各组结束原因都单独保存。
import math  # 立即拒绝NaN、Inf及无效均值。
import os  # 查询当前可用空间，绝不删除旧归档。
import platform  # 保持原myconda训练环境。
import signal  # 用户停止信号只在完整批次边界生效。
import sys  # Windows仅编辑，不能执行本项目计算。
import time  # 整个两组实验有统一时间上限。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
MODULE_SHAS = {
    'resume_acdd_development.py': '7b81570909cdf39cfa16c563c7c302e7c3886e5551060917ad61de7e8baf0014',
    'run_acdd_development.py': '5cd72f6199e542ecb64a49ed9c70883a4f9b6ea2d39d0faf4fb2a652ba0e2395',
    'inspect_acdd_adam_descent_v1.py': 'caef11ee20453e735b975d75de762fce4ae117b79d93f29f6176ef06f4bc17e7',
}  # 复用生产train_batch、固定数据计划与已审查的A/B监测公式。
for name, expected in MODULE_SHAS.items():
    if hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen helper changed: ' + name)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch  # 实际训练由main显式启动；CPU测试仅使用小网络。
import resume_acdd_development as resume
import inspect_acdd_adam_descent_v1 as probe

base, training = resume.baseline, resume.baseline.training
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
PARENT = DEV / 'acdd_s2000_to4000_v1'
PARENT_SHA = 'f007dd4375bbf23a78372549e6e79b87c06e94fd38c07b5e09eb6624ae50ce44'
PARENT_CONTRACT = '5fbd82bd161a7881f717b10b4d16e039f259e12ea7372b6cccb23e53b3326444'
PARENT_CONTRACT_FILE = '15287ed3c90e0d45dc7d39badf2b67d309a4e6be0ef268fe6ab4c1843a82633f'
INPUT_DIR = DEV / 'acdd_adam_descent_readonly_3107_v1'
INPUT_SHA = '296a7b4a7c289f0d19409947cd2c4dcb54754ae5636a58af43e658bc12b17da8'
RESULT_SHA = '61f5100aa2c8278bcfc023235f5d7962a8bc7704e2dbaadb71dea85f5c4d36f9'
PROTOCOL = DEV / 'ACDD3107_LR_PAIRED120_PROTOCOL_CN.md'
PROTOCOL_SHA = '048967f7012311b290dc1b690675081b74dbb6e734371be686263a9239293ef4'
START, UPDATES = 3107, 120  # 两组最多各120次新更新，不沿用旧4000步计划。
ARMS = (('control_s2e4', 0.0002), ('primary_s1e4', 0.0001))  # 顺序预先确定，不能依结果交换。
LOSS_KEYS = ('student_total', 'student_adv', 'fm_raw', 'fm_weighted', 'dist_raw', 'dist_weighted', 'discriminator')
SYSTEM_RESERVE = 3 * 1024**3
REPORT_RESERVE = 128 * 1024**2  # 包括小型目标张量、JSON、曲线等；不新增大输入副本。


class InsufficientDisk(RuntimeError):
    pass  # 空间不足时停止，不降低系统3GiB保护线。


def available_bytes(directory):
    info = os.statvfs(directory)
    return info.f_bavail * info.f_frsize


class StorageBudget:
    def __init__(self, directory, full_bytes, student_bytes):
        self.directory = directory
        self.pending = {'control_s2e4:S': student_bytes, 'primary_s1e4:S': student_bytes,
                        'primary_s1e4:full': full_bytes}  # 每份仅写一次，无中间或控制full。

    def require(self):
        required = SYSTEM_RESERVE + REPORT_RESERVE + sum(self.pending.values())
        free = available_bytes(self.directory)
        if free < required:
            raise InsufficientDisk(f'available={free}; required={required}; pending={self.pending}')
        return {'available_bytes': free, 'required_bytes': required, 'pending': dict(self.pending)}

    def committed(self, key):
        self.pending.pop(key)  # 文件已原子完成才扣除剩余预算，不能提前假定写入成功。


def append_json(path, record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()  # 逐步落盘，发生停止时仍能看到原始loss。


def finite_losses(losses):
    if any(key not in losses or not math.isfinite(float(losses[key])) for key in LOSS_KEYS):
        raise RuntimeError('Nonfinite or missing original training loss')


def compare_first_losses(control, candidate):
    finite_losses(control)
    finite_losses(candidate)
    differences = {key: abs(float(control[key]) - float(candidate[key])) for key in LOSS_KEYS}
    if any(differences[key] > 1e-7 + 1e-6 * abs(float(control[key])) for key in LOSS_KEYS):
        raise RuntimeError('First3108 pre-S-update losses differ across arms: ' + json.dumps(differences))
    return differences  # 首步D先更新过程相同，七项pre-S-update损失应在严格浮点容差内一致。


def window_record(rows):
    if len(rows) != 20 or [row['global_step'] for row in rows] != list(range(rows[0]['global_step'], rows[0]['global_step'] + 20)):
        raise RuntimeError('Only20 consecutive complete steps form a loss window')
    for row in rows:
        finite_losses(row)
    return {'updates': rows[-1]['updates'], 'global_step': rows[-1]['global_step'],
        'first_global_step': rows[0]['global_step'], 'count': 20,
        'means': {key: math.fsum(float(row[key]) for row in rows) / 20 for key in LOSS_KEYS}}


def moving_rise(windows):
    if len(windows) < 3:
        return False
    a, b, c = [row['means']['student_total'] for row in windows[-3:]]
    return 0 < a < b < c and c / a >= 1.02  # 最近3个完整窗口即2次连续上升，首尾至少涨2%。


def development_rise(records):
    if len(records) < 3:
        return False
    a, b, c = [row['development']['mel_l1'] for row in records[-3:]]
    initial = records[0]['development']['mel_l1']
    return 0 < initial and a < b < c and c / initial >= 1.01  # 最近3次含起点0；累计相对各组原起点。


def common_updates(arms):
    sets = [{row['updates'] for row in values} for values in arms]
    return sorted(set.intersection(*sets)) if len(sets) == 2 else []  # 不将不等终点冒充配对。


def model_sha(model):
    return probe.tree_sha({'parameters': dict(model.named_parameters()), 'buffers': dict(model.named_buffers())})
    # named_buffers包含MRD非持久Hann窗，不能仅检查state_dict。


def assert_frozen(models, fingerprints):
    for name, model in models.items():
        if model_sha(model) != fingerprints[name] or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen parameter/buffer changed: ' + name)


@contextlib.contextmanager
def preserve_rng(device):
    state = training.capture_rng_state()
    try:
        yield
    finally:
        training.restore_rng_state(state, device)
        if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(state):
            raise RuntimeError('Monitoring failed to restore the original RNG streams')


def restore_arm(payload, student, discriminator, optimizer, d_optimizer, lr, device):
    if lr not in (0.0002, 0.0001):
        raise RuntimeError('Undeclared student LR')
    if payload['configuration'] != base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0).as_dict():
        raise RuntimeError('Parent configuration changed before restore')
    # deepcopy避免CPU测试与两组优化器对同一个父moment就地修改；GPU也使用同一安全路径。
    restored = copy.deepcopy(payload)
    resume.restore_training_state(restored, student, discriminator, optimizer, d_optimizer,
                                  START, PARENT_CONTRACT, device)
    d_before = probe.tree_sha(d_optimizer.state_dict())
    s_moments = probe.tree_sha(optimizer.state_dict()['state'])
    for group in optimizer.param_groups:
        group['lr'] = lr  # 唯一试验改动：只改S步长，不改moment、step、beta或D学习率。
    if probe.tree_sha(d_optimizer.state_dict()) != d_before or probe.tree_sha(optimizer.state_dict()['state']) != s_moments:
        raise RuntimeError('LR override changed archived Adam moments or D')
    return {'student': lr, 'discriminator': d_optimizer.param_groups[0]['lr']}


def validate_effective_adam(optimizer, d_optimizer, student, discriminator, lr, step):
    # 老恢复校验器只接受2e-4；检查候选时仅在CPU副本还原这个字段，再检查真实shape/step/moments。
    original = optimizer.state_dict()
    saved = {'param_groups': copy.deepcopy(original['param_groups']), 'state': {
        identifier: {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                     for key, value in values.items()} for identifier, values in original['state'].items()}}
    if len(saved['param_groups']) != 1 or saved['param_groups'][0]['lr'] != lr:
        raise RuntimeError('Actual student LR differs from declared arm')
    saved['param_groups'][0]['lr'] = 0.0002
    resume.validate_adam_state(saved, student, step)
    resume.validate_adam_state(d_optimizer.state_dict(), discriminator, step)


def save_endpoint(directory, arm, step, common, student, discriminator, optimizer, d_optimizer,
                  budget, state, on_commit=None):
    # 所有大文件均为新路径，仅各写一次；即使后续SHA或JSON失败，已原子完成的恢复点仍立即记账。
    if step <= START:
        return
    if arm == 'primary_s1e4':
        path = directory / 'final_full.pt'
        if path.exists() or path.is_symlink():
            raise RuntimeError('Refuse to overwrite final full state')
        budget.require()
        payload = dict(common, student=student.state_dict(), discriminator=discriminator.state_dict(),
            student_optimizer=optimizer.state_dict(), discriminator_optimizer=d_optimizer.state_dict(),
            rng_state=training.capture_rng_state())
        training.atomic_torch_save(payload, path)
        state['last_full_saved_global_step'] = step
        state['full_snapshot'] = {'path': str(path), 'sha256': None, 'committed': True, 'global_step': step}
        budget.committed(arm + ':full')
        if on_commit:
            on_commit(state)
        state['full_snapshot']['sha256'] = base.sha256_file(path)
        path.chmod(0o444)
        state['full_snapshot']['frozen_readonly'] = True
    path = directory / f'student_step_{step:06d}.pt'
    if path.exists() or path.is_symlink():
        raise RuntimeError('Refuse to overwrite endpoint student')
    budget.require()
    training.atomic_torch_save(dict(common, student=student.state_dict()), path)
    state['student_snapshot'] = {'path': str(path), 'sha256': None, 'committed': True, 'global_step': step}
    budget.committed(arm + ':S')
    if on_commit:
        on_commit(state)
    state['student_snapshot']['sha256'] = base.sha256_file(path)
    path.chmod(0o444)
    state['student_snapshot']['frozen_readonly'] = True
    base.save_json(directory / 'checkpoint_receipt.json', state)  # 写失败仍由调用者报告真实保存游标。


def report_failure(directory, state, error):
    state.update(status='failed', error_type=type(error).__name__, error=str(error))
    print('PAIRED_ARM_FAILED ' + json.dumps(state, ensure_ascii=False), flush=True)
    try:
        base.save_json(directory / 'failure.json', state)
    except Exception as secondary:
        print('FAILURE_METADATA_WRITE_ERROR ' + repr(secondary), flush=True)
    # 不尝试把中途D已更新/S未更新的状态打包成完整新步，原父full仍可恢复。


@torch.no_grad()
def monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, step):
    with preserve_rng(device):
        prediction, _ = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'],
            inputs['bnf'], 950, inputs['source_noise'])
        fixed = probe.score_readout(prediction, inputs['x0'], target, teacher, diffusion,
            inputs['score_timesteps'], inputs['score_noise'], inputs['spk_emb'], inputs['bnf'],
            vocoder, fixed_d, mean, std, config)
        dev_batch = {key: inputs['dev_' + key] for key in ('mel', 'bnf', 'spk_emb')}
        development = base.evaluate_reconstruction(student, diffusion, dev_batch, inputs['dev_source_noise'], config, device)
        if any(not math.isfinite(value) for value in fixed.values()):
            raise RuntimeError('Nonfinite fixed TRAIN monitor')
    return {'global_step': step, 'updates': step - START, 'fixed_train': fixed, 'development': development}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--parent-checkpoint-sha256', required=True)
    parser.add_argument('--max-seconds', type=int, default=1200)
    parser.add_argument('--run-approved', action='store_true')
    args = parser.parse_args()
    if not args.run_approved or args.parent_checkpoint_sha256 != PARENT_SHA or not 1 <= args.max_seconds <= 1200:
        raise RuntimeError('Explicit approved parent and at most1200 seconds required')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)  # 全局GPU互斥，保留引用直到进程退出。
    if training.git_head(ROOT) != base.EXPECTED_HEAD:
        raise RuntimeError('Production source is not the frozen clean HEAD')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Original myconda runtime required')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Registered rental CUDA interpreter required')
    output = args.output_dir.resolve()
    if output.parent != DEV or output.exists():
        raise RuntimeError('Use a new direct child of DEV')
    dependencies = {str(PARENT / 'latest_full.pt'): PARENT_SHA, str(PARENT / 'run_contract.json'): PARENT_CONTRACT_FILE,
        str(PARENT / 'student_step_003107.pt'): 'a9128f1acc9adee16c18f27e5e0361fac95398b3281e5a71567eaaa2359980c0',
        str(INPUT_DIR / 'fixed_inputs.pt'): INPUT_SHA, str(INPUT_DIR / 'results.json'): RESULT_SHA,
        str(PROTOCOL): PROTOCOL_SHA, str(resume.PARENT / 'initial_full.pt'): resume.ANCHOR_SHA,
        str(resume.PARENT / 'run_contract.json'): resume.PARENT_CONTRACT_FILE_SHA,
        **{str(TOOLS / name): sha for name, sha in MODULE_SHAS.items()}}
    resume.verify_files(dependencies)
    parent = json.loads((PARENT / 'run_contract.json').read_text())
    grand = json.loads((resume.PARENT / 'run_contract.json').read_text())
    if training.canonical_json_sha256(parent) != PARENT_CONTRACT or training.canonical_json_sha256(grand) != resume.PARENT_CONTRACT_SHA:
        raise RuntimeError('Parent/grandparent canonical identity changed')
    dependencies.update(parent['file_sha256'])
    development, hashes = resume.load_development(DEV / 'development_contract.json')
    dependencies.update(hashes)
    if parent['training_indices'] != development['retained_train_indices'] or parent['reconstruction_dev_indices'] != development['reconstruction_dev_indices']:
        raise RuntimeError('Training/development split differs from the parent')
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if parent['configuration'] != config.as_dict() or grand['configuration'] != config.as_dict() or grand['seed'] != 1234:
        raise RuntimeError('Base ACDD configuration changed')
    if training.collect_source_fingerprints(ROOT) != grand['source_sha256']:
        raise RuntimeError('Production math, teacher, BNF or data source changed')
    dependencies[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
    resume.verify_files(dependencies)
    initial = resume.load_torch_checkpoint(resume.PARENT / 'initial_full.pt')
    anchor = copy.deepcopy(initial['development'])
    if initial['global_step'] != 0 or initial['run_contract_sha256'] != resume.PARENT_CONTRACT_SHA or anchor != parent['original_step0_guard_anchor']:
        raise RuntimeError('Original S0 protection anchor changed')
    del initial
    if anchor['mel_l1'] != 0.19941353797912598 or anchor['prediction_rms'] != 0.9994737505912781:
        raise RuntimeError('Original S0 thresholds cannot be rebased')
    payload = resume.load_torch_checkpoint(PARENT / 'latest_full.pt')
    resume.assert_guard(payload['development'], anchor)
    # 采用真实归档大小估算同结构输出；JSON和序列化表头差额包含在128MiB额度内。
    budget = StorageBudget(DEV, (PARENT / 'latest_full.pt').stat().st_size,
                           (PARENT / 'student_step_003107.pt').stat().st_size)
    capacity = budget.require()
    inputs_cpu = resume.load_torch_checkpoint(INPUT_DIR / 'fixed_inputs.pt')
    reference = json.loads((INPUT_DIR / 'results.json').read_text())
    input_hashes = {name: probe.tensor_sha(value) for name, value in inputs_cpu.items()}
    if input_hashes != reference['input_sha256']:
        raise RuntimeError('Frozen TRAIN/DEV tensors changed')
    torch.set_num_threads(2)  # 沿用已冻结开发驱动，其他TF32/cudnn开关不改。
    training.seed_everything(1234)
    device = torch.device('cuda')
    started = time.monotonic()
    stop = {'requested': False}
    def request_stop(signum, frame):
        stop['requested'] = True  # 不在D/S半步之间抛异常，完成当前批后安全结束。
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100,
                                   bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387:
        raise RuntimeError('Original training dataset length changed')
    plan = resume.GlobalBatchPlan(development['retained_train_indices'])
    if plan.batches_per_pass != 1229 or plan.for_step(3108) != reference['train_indices']:
        raise RuntimeError('Absolute3108 TRAIN plan changed')
    first = base.batch_for(dataset, plan.for_step(3108), 104342)
    for key, stored in (('mel', 'x0'), ('bnf', 'bnf'), ('spk_emb', 'spk_emb')):
        if probe.tensor_sha(first[key]) != input_hashes[stored]:
            raise RuntimeError('Current dataset does not reproduce fixed TRAIN3108')
    inputs = {name: value.to(device) for name, value in inputs_cpu.items()}
    del inputs_cpu, first
    teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
    vocoder, vconfig, _ = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
        base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
    fixed_d = base.build_paper_discriminator(ROOT, vconfig, config, device)
    fixed_d.load_state_dict(payload['discriminator'], strict=True)
    fixed_d.eval()
    training.set_requires_grad(fixed_d, False)  # 监测D与训练D是不同对象，永不向其传入优化器。
    diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = base.load_mel_statistics(ROOT / 'data', device)
    frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
    frozen_hashes = {name: model_sha(model) for name, model in frozen.items()}
    assert_frozen(frozen, frozen_hashes)
    # 先证明归档噪声就是生产3108次序；此检查不会消耗恢复后的训练随机流。
    with preserve_rng(device):
        training.seed_everything(204342)
        noise = torch.randn_like(inputs['x0'])
        score_noise = torch.randn_like(inputs['x0'])
        times = torch.randint(1, 1001, (32,), device=device, dtype=torch.long)
        q950 = diffusion.q_sample(inputs['x0'], torch.full((32,), 950, device=device, dtype=torch.long), noise=noise)
        if any(probe.tensor_sha(value) != input_hashes[key] for key, value in (
                ('source_noise', noise), ('score_noise', score_noise), ('score_timesteps', times), ('q950', q950))):
            raise RuntimeError('Fixed input noise/timestep order differs from production')
        del noise, score_noise, times, q950
    student.load_state_dict(payload['student'], strict=True)
    student.train()
    with torch.no_grad(), preserve_rng(device):
        prediction, _ = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
        _, target = training.score_distillation_loss(prediction, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
            inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
    target_sha = probe.tensor_sha(target)
    del prediction
    if target_sha != reference['models']['3107']['teacher_target_sha256']:
        raise RuntimeError('Initial fixed teacher target no longer matches readonly diagnostic')
    lineage = {'parent_checkpoint': str(PARENT / 'latest_full.pt'), 'parent_checkpoint_sha256': PARENT_SHA,
        'parent_contract_sha256': PARENT_CONTRACT, 'original_step0_checkpoint_sha256': resume.ANCHOR_SHA,
        'start_after_global_step': START}
    contract = {'kind': 'paired_acdd_student_lr120_development', 'lineage': lineage,
        'configuration': config.as_dict(), 'configuration_interpretation': 'Base restoration config; actual role-specific LR below overrides only S after full restore.',
        'effective_optimizer_lrs': {name: {'student': lr, 'discriminator': 0.0002} for name, lr in ARMS},
        'method_extension': 'Only primary S Adam constant LR1e-4; original2024 uses2e-4. Teacher/G/BNF/ACDD targets and2/45 unchanged.',
        'arms_order': [name for name, _ in ARMS], 'seed': 1234, 'maximum_updates_per_arm': UPDATES, 'max_seconds_pair': args.max_seconds,
        'development_contract_file_sha256': resume.DEVELOPMENT_SHA, 'training_indices': development['retained_train_indices'],
        'excluded_train_indices': development['excluded_train_indices'], 'reconstruction_dev_indices': development['reconstruction_dev_indices'],
        'formal_test_used_for_selection': False, 'cer_or_90pair_evaluation_planned': False,
        'input_sha256': input_hashes, 'fixed_inputs_file': str(INPUT_DIR / 'fixed_inputs.pt'),
        'fixed_development_inputs': {key: input_hashes['dev_' + key] for key in ('mel', 'bnf', 'spk_emb')},
        'fixed_development_source_noise_sha256': input_hashes['dev_source_noise'],
        'fixed_teacher_target_sha256': target_sha, 'fixed_frozen_models_sha256': frozen_hashes,
        'fixed_train_monitor_is_first_training_batch_not_independent_development': True,
        'batch_plan': {'first_global_step': 3108, 'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((step-1)/1229)',
            'crop_seed': '1234+100000+step', 'noise_seed': '1234+200000+step', 'draw_order': ['source_noise', 'score_noise', 'score_timesteps']},
        'original_step0_guard_anchor': anchor, 'guard_ratios': {'mel_l1_upper': 1.10, 'prediction_rms_lower': 0.80},
        'stops': {'moving_total': 'last3 nonoverlap complete20 means strictly rising and last/first>=1.02',
            'development': 'last3 checks including0 strictly rising and latest/arm0>=1.01',
            'other': ['nonfinite', 'OOM/exception', 'originalS0 guard', 'disk', 'time', 'requested_stop']},
        'monitor_semantics': 'Raw original train losses stay primary. FrozenD/frozen-target A and refreshed-target B on TRAIN3108 are separate diagnostics only.',
        'comparison': 'Only common update counts; all endpoints reported, no best checkpoint selection; unequal final weights are not paired performance.',
        'storage': {'endpoints_only': 'two S-only plus only primary full; no intermediate or control full',
            'atomic_full': 'new file only; no rolling overwrite or second full copy', 'initial_budget': capacity,
            'system_reserve_bytes': SYSTEM_RESERVE, 'report_reserve_bytes': REPORT_RESERVE},
        'file_sha256': dependencies, 'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda}}
    budget.require()
    output.mkdir()  # 所有只读身份检查通过后才建立唯一新目录。
    base.save_json(output / 'run_contract.json', contract)
    contract_sha = training.canonical_json_sha256(contract)
    training.atomic_torch_save({'teacher_target': target.cpu(), 'sha256': target_sha, 'global_source_step': START}, output / 'fixed_teacher_target.pt')
    parent_development = copy.deepcopy(payload['development'])
    del payload  # 每个arm重新从只读父文件载入，绝不承接控制组终点。
    results, histories, first_losses = {}, [], {}
    for arm, lr in ARMS:
        if stop['requested'] or time.monotonic() - started >= args.max_seconds:
            break  # 用户要求停止后不能再启动下一组。
        directory = output / arm
        directory.mkdir()
        state = {'status': 'initializing', 'arm': arm, 'completed_global_step': START, 'updates': 0,
            'last_full_saved_global_step': START, 'full_snapshot': None, 'student_snapshot': None,
            'contract_sha256': contract_sha, 'parent_checkpoint_sha256': PARENT_SHA, 'effective_optimizer_lrs': {'student': lr, 'discriminator': 0.0002},
            'partial_batch_may_have_updated_D': False}
        records, windows, pending = [], [], []
        try:
            budget.require()
            payload = resume.load_torch_checkpoint(PARENT / 'latest_full.pt')
            optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, betas=(0.5, 0.9))
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))
            state['effective_optimizer_lrs'] = restore_arm(payload, student, discriminator, optimizer, d_optimizer, lr, device)
            del payload
            with torch.no_grad(), preserve_rng(device):
                restored_prediction, _ = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
                _, restored_target = training.score_distillation_loss(restored_prediction, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
                    inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
                if probe.tensor_sha(restored_target) != target_sha:
                    raise RuntimeError('Arms do not start with identical teacher target')
                del restored_prediction, restored_target
            entry = monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, START)
            probe.validate_development_replay(entry['development'], parent_development)
            resume.assert_guard(entry['development'], anchor)
            for key, value in reference['models']['3107']['before_train'].items():
                if abs(entry['fixed_train'][key] - value) > 2e-5:
                    raise RuntimeError('FixedD3107 TRAIN initial monitor did not replay')
            records.append(entry)
            append_json(directory / 'monitor.jsonl', entry)
            state.update(status='running', last_monitor=entry)
            base.save_json(directory / 'status.json', state)
            reason = 'maximum120_updates_completed'
            for update in range(1, UPDATES + 1):
                if stop['requested'] or time.monotonic() - started >= args.max_seconds:
                    reason = 'requested_stop' if stop['requested'] else 'time_budget'
                    break
                try:
                    budget.require()  # 每批都为所有尚未提交的终点文件预留空间。
                except InsufficientDisk:
                    reason = 'disk_space_guard'
                    break
                step = START + update
                indices = plan.for_step(step)
                if set(indices) & set(development['excluded_train_indices']):
                    raise RuntimeError('Training batch overlaps52 held-out utterances')
                batch = base.batch_for(dataset, indices, 1234 + 100000 + step)
                training.seed_everything(1234 + 200000 + step)
                state['partial_batch_may_have_updated_D'] = True
                losses = training.train_batch(batch, teacher, student, vocoder, discriminator, diffusion,
                    optimizer, d_optimizer, mean, std, device, config)  # 唯一训练路径，固定target从未传入这里。
                state.update(completed_global_step=step, updates=update, partial_batch_may_have_updated_D=False)
                finite_losses(losses)
                if update == 1:
                    first_losses[arm] = {key: float(losses[key]) for key in LOSS_KEYS}
                    if arm == 'primary_s1e4':
                        state['first3108_loss_absolute_differences'] = compare_first_losses(first_losses['control_s2e4'], losses)
                row = {'global_step': step, 'updates': update, 'indices': indices,
                       'elapsed_seconds': time.monotonic() - started,
                       'actual_optimizer_lrs': {'student': optimizer.param_groups[0]['lr'], 'discriminator': d_optimizer.param_groups[0]['lr']}, **losses}
                append_json(directory / 'step_losses.jsonl', row)
                pending.append(row)
                if update % 20 == 0:
                    window = window_record(pending)
                    pending = []
                    windows.append(window)
                    append_json(directory / 'loss_windows.jsonl', window)
                    entry = monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, step)
                    records.append(entry)
                    state['last_monitor'] = entry
                    append_json(directory / 'monitor.jsonl', entry)
                    print(json.dumps({'arm': arm, 'updates': update, 'moving20': window['means'], 'monitor': entry}), flush=True)
                    try:
                        resume.assert_guard(entry['development'], anchor)
                    except RuntimeError:
                        reason = 'original_S0_guard'
                        break
                    if moving_rise(windows) or development_rise(records):
                        reason = 'moving_total_three_windows_rising' if moving_rise(windows) else 'development_three_checks_rising'
                        break  # 预声明趋势满足即停，不等待看到后续下降。
                base.save_json(directory / 'status.json', state)
            step = state['completed_global_step']
            if records[-1]['global_step'] != step:
                entry = monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, step)
                records.append(entry)
                state['last_monitor'] = entry
                append_json(directory / 'monitor.jsonl', entry)  # 不完整20步也保留终点，不能补成完整窗口。
                try:
                    resume.assert_guard(entry['development'], anchor)
                except RuntimeError:
                    reason = 'original_S0_guard'
            assert_frozen(frozen, frozen_hashes)
            if probe.tensor_sha(target) != target_sha or {name: probe.tensor_sha(value) for name, value in inputs.items()} != input_hashes:
                raise RuntimeError('Fixed target or input tensors changed during training')
            validate_effective_adam(optimizer, d_optimizer, student, discriminator, lr, step)
            state.update(reason=reason, last_monitor=records[-1], partial_window_steps=len(pending))
            common = {'kind': 'paired_acdd_lr_development_checkpoint', 'global_step': step, 'epoch': 0,
                'configuration': config.as_dict(), 'effective_optimizer_lrs': state['effective_optimizer_lrs'],
                'configuration_interpretation': contract['configuration_interpretation'], 'development': records[-1]['development'],
                'run_contract_sha256': contract_sha, 'lineage': lineage, 'arm': arm, 'original_step0_guard_anchor': anchor,
                'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'stop_reason': reason}
            save_endpoint(directory, arm, step, common, student, discriminator, optimizer, d_optimizer, budget, state)
            state['status'] = 'completed' if reason == 'maximum120_updates_completed' else 'stopped'
            base.save_json(directory / 'arm_completion.json', state)
            print('PAIRED_ARM_FINISHED ' + json.dumps(state), flush=True)
            results[arm] = state
            histories.append(records)
            del optimizer, d_optimizer
            if reason in ('disk_space_guard', 'requested_stop', 'time_budget'):
                break  # 资源或用户停止作用于全实验，不能悄悄继续下一组。
        except Exception as error:
            report_failure(directory, state, error)
            raise  # 错误不标completed；没有原子full时真实恢复点仍为父3107。
    resume.verify_files(dependencies)
    assert_frozen(frozen, frozen_hashes)
    completion = {'status': 'paired_short_experiment_finished' if len(results) == 2 else 'paired_short_experiment_incomplete',
        'arms': results, 'common_monitor_updates': common_updates(histories), 'contract_sha256': contract_sha,
        'elapsed_seconds': time.monotonic() - started,
        'no_formal_performance_claim': True, 'comparison_scope': 'Common scalar checkpoints only; unequal endpoint weights are unpaired.'}
    base.save_json(output / 'completion.json', completion)
    print('PAIRED_EXPERIMENT_FINISHED ' + json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()  # 只准备脚本和CPU测试不会触发训练。
