# 本研究只比较 ACDD600 后的601至1200步；源教师、BNF、声码器和旧实验文件全部保持不变。
"""Bounded fixed-BNF conversion-score continuation; not a 2024 reproduction."""

import argparse  # 父身份、终点和实际运行授权都必须显式给出。
import copy  # 保存原S0保护锚，不修改父检查点中的对象。
import hashlib  # 导入辅助模块之前先核验冻结文件的实际字节。
import json  # 新契约、检查点指针与日志只写入本次新目录。
import os  # statvfs读取当前可用空间，不删除任何归档。
import platform  # 锁定原myconda训练运行时。
import signal  # 用户停止在完整D/S更新之后处理。
import sys  # Windows只允许编辑，拒绝在本地导入执行此驱动。
import time  # 最多运行3600秒，不无限续训。
from pathlib import Path  # 路径解析后再限制父来源和新输出范围。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一获准的租用服务器工作区。
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'  # 冻结辅助模块的唯一来源。
MODULE_SHAS = {
    'resume_acdd_development.py': '7b81570909cdf39cfa16c563c7c302e7c3886e5551060917ad61de7e8baf0014',
    'conversion_score_step.py': '94d0a87feaa1ac00efd5e1a434d075afdb825732f1a4ecf3cdb8394ec9003b3e',
    'run_acdd_development.py': '5cd72f6199e542ecb64a49ed9c70883a4f9b6ea2d39d0faf4fb2a652ba0e2395',
}  # 原恢复逻辑、唯一新目标函数、原数据/测量函数都要保持已审查版本。
for module_name, digest in MODULE_SHAS.items():
    if hashlib.sha256((TOOLS / module_name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen helper changed before import: ' + module_name)
sys.path.insert(0, str(ROOT))  # 生产模型代码来自服务器工作区。
sys.path.insert(0, str(TOOLS))  # 辅助模块来自已验哈希的目录。

import torch  # import不执行GPU；只有main通过全部门禁后才构建真实模型。
import resume_acdd_development as resume  # 复用严格数据验真、全局批次计划和双Adam恢复。
import conversion_score_step as conversion  # 已通过6项CPU测试的冻结混合目标。

baseline = resume.baseline  # 完整复用同一份batch_for和固定32重建测量。
training = baseline.training  # 不修改生产train.py或它的全局函数。
DEV = ROOT / 'exp/diagnostics/student_development_20260906'  # 新run必须是此目录下的新直接子目录。
CONTROL = DEV / 'acdd_clean_resume_s2000_v1/full_step_001200.pt'  # 原目标600→1200的已保存对照。
CONTROL_SHA = 'b38fd8db9c137a7955dcbd1fb863a4a6658e00113a613cad39306a14e6be6123'
PERMUTATION_SEED = 991000  # 先验固定的独立重排种子，不根据转换难度挑配对。
GIB = 1024**3  # 所有磁盘门槛均用GiB而非含糊的十进制GB。
FULL_WRITE_BYTES = int(1.5 * GIB)  # 原完整状态约1.33GiB，为本次临时完整文件预留1.5GiB。
STUDENT_WRITE_BYTES = int(0.35 * GIB)  # 原学生快照约0.29GiB，单次另预留0.35GiB。
SYSTEM_RESERVE_BYTES = 3 * GIB  # 完整保存结束后仍给系统留下至少3GiB。


class InsufficientDisk(RuntimeError):
    pass  # 磁盘不足时停止；绝不删除旧run或假装当前步骤已经保存。


def available_bytes(directory):
    stats = os.statvfs(directory)  # 读取调用方新目录所在文件系统的真实可用空间。
    return stats.f_bavail * stats.f_frsize  # 使用普通进程可用块数，不计仅root保留块。


def require_checkpoint_space(directory, include_student):
    required = FULL_WRITE_BYTES + SYSTEM_RESERVE_BYTES + (STUDENT_WRITE_BYTES if include_student else 0)
    free = available_bytes(directory)  # 每次保存前重新查询，不只相信启动时的磁盘状态。
    if free < required:
        raise InsufficientDisk(f'Checkpoint requires {required} free bytes; available={free}')
    return free  # 记录保存前空间，便于区分训练异常和磁盘问题。


def validate_launch(parent_path, parent_sha, parent_contract, stop_step, max_seconds):
    if parent_path.resolve() != resume.PARENT / 'latest_full.pt' or parent_sha != resume.PARENT_FULL_SHA:
        raise RuntimeError('Only the frozen complete ACDD600 parent is accepted')
    if parent_contract != resume.PARENT_CONTRACT_SHA:
        raise RuntimeError('Declared parent contract is not the audited ACDD600 contract')
    if stop_step not in (601, 1200) or not 1 <= max_seconds <= 3600:
        raise RuntimeError('Choose explicit one-step601 smoke or matched1200 end, with at most3600 seconds')
    return range(601, stop_step + 1)  # 全局601是唯一允许的第一步，最多新增600次更新。


def start_boundary_monitor(output, parent_windows):
    archived = output / 'parent_reconstruction_loss_windows_through_600.jsonl'
    archived.write_bytes(parent_windows.read_bytes())  # 新目录保留完整父历史；源文件只读。
    baseline.save_json(output / 'objective_boundary.json', {
        'global_step': 600, 'parent_history': str(archived),
        'previous': 'reconstruction adv + 2 reconstruction FM + 45 reconstruction score',
        'current': 'reconstruction adv + 2 reconstruction FM + 45 conversion score',
        'cross_boundary_trend_comparison': False, 'new_window_steps': 50,
    })  # total/dist含义变化，旧窗口不得参加新目标的连续上涨判断。
    monitor = training.LossMonitor(output, window_steps=50, resume_step=600)
    if monitor.windows:
        raise RuntimeError('A new objective must start with no active pre-switch loss windows')
    monitor.last_global_step = 600  # 继承绝对训练游标，不把本轮伪装成从0步开始。
    return monitor


def save_checkpoint(output, step, scores, student, discriminator, optimizer, d_optimizer,
                    config, contract_sha, lineage, anchor, *, on_full_committed=None):
    # 此函数只覆盖本run自己的latest_full，学生快照一旦存在就拒绝覆盖。
    output = output.resolve()
    full_path = output / 'latest_full.pt'
    student_path = output / f'student_step_{step:06d}.pt'
    if full_path.is_symlink() or student_path.exists() or student_path.is_symlink():
        raise RuntimeError('Checkpoint targets must be local to this run; student snapshots are immutable')
    include_student = True  # 正常每100步及结束都保留不可覆盖的学生快照。
    try:
        before = require_checkpoint_space(output, include_student=True)
    except InsufficientDisk:
        before = require_checkpoint_space(output, include_student=False)  # 若只够完整恢复状态，优先保住它并停止。
        include_student = False
    common = {
        'kind': 'conversion_score_development_checkpoint', 'student': student.state_dict(),
        'epoch': 0, 'global_step': step, 'configuration': config.as_dict(), 'development': scores,
        'run_contract_sha256': contract_sha, 'lineage': lineage, 'original_step0_guard_anchor': anchor,
        'score_semantics': 'conversion_clean_x0_sqrt_alpha_bar',
    }  # 永远传mapping给真实atomic_torch_save，不保存裸Tensor。
    full = dict(common, discriminator=discriminator.state_dict(), student_optimizer=optimizer.state_dict(),
                discriminator_optimizer=d_optimizer.state_dict(), rng_state=training.capture_rng_state())
    receipt = {'global_step': step, 'path': str(full_path), 'checkpoint_sha256': None,
               'run_contract_sha256': contract_sha, 'free_bytes_before': before, 'student_snapshot': None,
               'stop_for_disk': not include_student, 'full_committed': False, 'write_stage': 'full_state_write'}
    training.atomic_torch_save(full, full_path)  # 临时完整文件写好后原子替换，仅覆盖此run上一次完整状态。
    receipt['full_committed'] = True  # 这一行仅在真实原子替换成功后执行；此前异常不能推进恢复步数。
    receipt['write_stage'] = 'full_state_committed'  # 后续SHA、快照或metadata失败，都不否认这个已落盘状态。
    if on_full_committed is not None:
        on_full_committed(receipt)  # 立即通知主循环；回调只赋内存游标，不进行任何磁盘写入。
    receipt['write_stage'] = 'full_sha256'
    receipt['checkpoint_sha256'] = baseline.sha256_file(full_path)  # 哈希失败时可恢复步仍已记账，但SHA保持未知。
    receipt['write_stage'] = 'initial_receipt_write'
    baseline.save_json(output / 'latest_full.json', receipt)  # 完整保存成功后立即更新可恢复指针。
    if include_student:
        if available_bytes(output) < STUDENT_WRITE_BYTES + SYSTEM_RESERVE_BYTES:
            receipt['stop_for_disk'] = True  # 外部空间突降时保留刚保存的full，停止而不继续写S快照。
        else:
            receipt['write_stage'] = 'student_snapshot_write'
            training.atomic_torch_save(common, student_path)  # OSError/RuntimeError均向上报告失败，不伪装成功。
            receipt['student_snapshot'] = {'path': str(student_path), 'sha256': None, 'committed': True}
            receipt['write_stage'] = 'student_snapshot_sha256'
            receipt['student_snapshot']['sha256'] = baseline.sha256_file(student_path)
    receipt['write_stage'] = 'final_receipt_write'
    baseline.save_json(output / 'latest_full.json', receipt)  # 完整归档与可选学生快照都明确记录。
    receipt['write_stage'] = 'checkpoint_history_write'
    with (output / 'checkpoint_history.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(receipt) + '\n')  # 记录历次full身份，虽其旧内容不再永久占用空间。
    receipt['write_stage'] = 'complete'  # 只有全部附属写入成功后，保存函数才正常返回。
    return receipt


def report_run_failure(output, monitor, final_step, last_saved_step, last_receipt, parent_path, error):
    # 磁盘错误可能使JSON和monitor也无法写盘；先打印完整事实，再尽力更新这些辅助文件。
    summary = {'status': 'failed', 'completed_global_step': final_step,
        'last_saved_global_step': last_saved_step, 'unsaved_completed_steps': final_step - last_saved_step,
        'error_type': type(error).__name__, 'error': str(error), 'parent_checkpoint': str(parent_path),
        'latest_full': last_receipt}  # last_receipt从full原子完成时更新，不能落后于真实可恢复权重。
    print('CONVERSION_SCORE_DEVELOPMENT_FAILED ' + json.dumps(summary), flush=True)
    try:
        monitor.finish('failed', final_step, 0)
    except Exception as metadata_error:
        print('FAILURE_MONITOR_WRITE_ERROR ' + repr(metadata_error), flush=True)  # 保留原始异常，避免被二次写盘异常盖掉。
    try:
        baseline.save_json(output / 'status.json', summary)
    except Exception as metadata_error:
        print('FAILURE_STATUS_WRITE_ERROR ' + repr(metadata_error), flush=True)  # stdout仍有可供主线程读取的真实恢复步。
    return summary  # 供CPU故障注入检查；main随后仍抛出原异常，不能标记候选完成。


def write_disk_stop_or_report_failure(output, monitor, final_step, last_saved_step, last_receipt, parent_path, disk_error):
    # 空间保护本身正常停止，但其停止记录若也无法写入，就必须走明确的失败报告路径。
    try:
        baseline.save_json(output / 'disk_stop.json', {'error': str(disk_error), 'completed_global_step': final_step,
            'last_saved_global_step': last_saved_step, 'unsaved_completed_steps': final_step - last_saved_step})
    except Exception as metadata_error:
        report_run_failure(output, monitor, final_step, last_saved_step, last_receipt, parent_path, metadata_error)
        raise  # 原样抛出停止metadata的异常；同级except不会自动捕获此处失败。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-checkpoint', type=Path, required=True)  # 明确完整600步来源。
    parser.add_argument('--parent-checkpoint-sha256', required=True)
    parser.add_argument('--parent-contract-sha256', required=True)
    parser.add_argument('--stop-global-step', type=int, choices=(601, 1200), required=True)  # 601只标作集成smoke。
    parser.add_argument('--max-seconds', type=int, required=True)  # 最长3600秒。
    parser.add_argument('--development-contract', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)  # 必须是从未存在过的新run。
    parser.add_argument('--run-approved', action='store_true')  # 写好脚本与CPU通过不构成GPU启动。
    args = parser.parse_args()
    steps = validate_launch(args.parent_checkpoint, args.parent_checkpoint_sha256,
                            args.parent_contract_sha256, args.stop_global_step, args.max_seconds)
    if not args.run_approved:
        raise RuntimeError('Explicit approval is required for this GPU experiment')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)  # 独占项目GPU工作流；其他任务占用时拒绝启动。
    if training.git_head(ROOT) != baseline.EXPECTED_HEAD:
        raise RuntimeError('Production source HEAD differs from the audited baseline')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Use the original Python3.8.15 torch1.13.1+cu116 training runtime')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve():
        raise RuntimeError('The actual interpreter must be myconda/bin/python')
    if not torch.cuda.is_available():
        raise RuntimeError('Rental GPU unavailable')
    output = args.output_dir.resolve()
    if output.parent != DEV or output.exists():
        raise RuntimeError('Create a new direct child of the development directory; old runs are protected')
    require_checkpoint_space(DEV, include_student=True)  # 大模型构建前先确认至少足够安全保存一次。
    parent_path = args.parent_checkpoint.resolve()
    dependencies = {str(parent_path): resume.PARENT_FULL_SHA,
        str(resume.PARENT / 'initial_full.pt'): resume.ANCHOR_SHA,
        str(resume.PARENT / 'run_contract.json'): resume.PARENT_CONTRACT_FILE_SHA,
        str(CONTROL): CONTROL_SHA,
        **{str(TOOLS / name): digest for name, digest in MODULE_SHAS.items()}}
    resume.verify_files(dependencies)  # 父状态、原S0和同段控制全部固定。
    if parent_path.stat().st_size > FULL_WRITE_BYTES:
        raise RuntimeError('Full checkpoint exceeds the declared1.5GiB temporary-write budget')
    parent = json.loads((resume.PARENT / 'run_contract.json').read_text(encoding='utf-8'))
    if training.canonical_json_sha256(parent) != resume.PARENT_CONTRACT_SHA:
        raise RuntimeError('Parent contract canonical identity changed')
    development, dev_hashes = resume.load_development(args.development_contract.resolve())
    dependencies.update(dev_hashes)  # 包括原mean/std、开发素材及52条排除收据。
    if parent['training_indices'] != development['retained_train_indices'] or parent['reconstruction_dev_indices'] != development['reconstruction_dev_indices']:
        raise RuntimeError('Train/development population differs from the parent')
    config = baseline.replace(baseline.PAPER_STUDENT_CONFIG, num_workers=0)
    if config.as_dict() != parent['configuration'] or parent['seed'] != 1234:
        raise RuntimeError('Parent hyperparameters, normalization or BNF settings changed')
    sources = training.collect_source_fingerprints(ROOT)
    if sources != parent['source_sha256']:
        raise RuntimeError('Production source fingerprints no longer match the parent')
    dependencies.update({str(ROOT / name): digest for name, digest in sources.items()})
    dependencies[str(Path(__file__).resolve())] = baseline.sha256_file(Path(__file__))
    payload = resume.load_torch_checkpoint(parent_path)  # 恢复必须包含S/D、双Adam、RNG。
    anchor_payload = resume.load_torch_checkpoint(resume.PARENT / 'initial_full.pt')
    if anchor_payload['global_step'] != 0 or anchor_payload['run_contract_sha256'] != resume.PARENT_CONTRACT_SHA:
        raise RuntimeError('Original step-zero anchor lineage changed')
    anchor = copy.deepcopy(anchor_payload['development'])
    del anchor_payload  # 保护线之外不使用S0模型，不浪费内存或混淆起点。
    if anchor['mel_l1'] != 0.19941353797912598 or anchor['prediction_rms'] != 0.9994737505912781:
        raise RuntimeError('OriginalS0 guard values changed')
    resume.assert_guard(payload['development'], anchor)
    if payload['configuration'] != config.as_dict():
        raise RuntimeError('Parent full checkpoint configuration differs from its contract')
    torch.set_num_threads(2)  # 与原开发基线相同，不改变TF32或cudnn设置。
    training.seed_everything(1234)
    device = torch.device('cuda')
    dataset = baseline.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100,
                                       bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387:
        raise RuntimeError('Original training manifest length changed')
    teacher, student, _ = baseline.load_teacher_and_student(baseline.TEACHER, config, device)
    vocoder, vconfig, _ = baseline.load_frozen_vocoder(ROOT, baseline.VOCODER_DIR / 'generator_step_2500000.pt',
        baseline.VOCODER_DIR / 'config.json', baseline.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = baseline.build_paper_discriminator(ROOT, vconfig, config, device)
    diffusion = baseline.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = baseline.load_mel_statistics(ROOT / 'data', device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2))
    resume.restore_training_state(payload, student, discriminator, optimizer, d_optimizer, 600, resume.PARENT_CONTRACT_SHA, device)
    parent_rng, parent_scores = payload['rng_state'], copy.deepcopy(payload['development'])
    del payload  # 当前学生和两组Adam已经恢复；不继续保留另一份大型CPU状态。
    protocol = development['reconstruction_protocol']
    if protocol['source_noise_device'] != 'cuda':
        raise RuntimeError('Fixed development source noise must use the original CUDA generator')
    dev_batch = baseline.batch_for(dataset, development['reconstruction_dev_indices'], protocol['crop_seed'])
    dev_noise = torch.randn(dev_batch['mel'].shape, device=device,
        generator=torch.Generator(device=device).manual_seed(protocol['source_noise_seed']))  # 一次生成完整32噪声。
    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if any(tensor_sha(dev_batch[key]) != parent['fixed_development_inputs'][key] for key in ('mel', 'bnf', 'spk_emb')):
        raise RuntimeError('Fixed32 feature identity changed')
    if tensor_sha(dev_noise) != parent['fixed_development_source_noise_sha256']:
        raise RuntimeError('Fixed32 CUDA noise identity changed')
    resumed_scores = baseline.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
    if any(abs(resumed_scores[key] - parent_scores[key]) > 1e-7 for key in ('mel_l1', 'prediction_rms', 'reference_rms', 'cosine_reference')):
        raise RuntimeError('RestoredS600 does not reproduce its saved development readout')
    resume.assert_guard(resumed_scores, anchor)
    plan = resume.GlobalBatchPlan(parent['training_indices'])  # 复用经过中轮恢复测试的原数据计划。
    if plan.batches_per_pass != 1229:
        raise RuntimeError('Expected39335 retained examples and1229 full batches per pass')
    parent_windows = resume.PARENT / 'loss_windows.jsonl'
    dependencies[str(parent_windows)] = baseline.sha256_file(parent_windows)
    dependencies[str(baseline.TEACHER)] = parent['teacher_sha256']
    dependencies[str(baseline.VOCODER_DIR / 'generator_step_2500000.pt')] = parent['vocoder_sha256']
    resume.verify_files(dependencies)  # 加载期间任何输入变更都会阻止开始更新。
    lineage = {'parent_directory': str(resume.PARENT), 'parent_full_checkpoint': str(parent_path),
        'parent_full_checkpoint_sha256': resume.PARENT_FULL_SHA, 'parent_contract_sha256': resume.PARENT_CONTRACT_SHA,
        'anchor_step0_checkpoint_sha256': resume.ANCHOR_SHA, 'start_after_global_step': 600,
        'matched_control_step1200': str(CONTROL), 'matched_control_step1200_sha256': CONTROL_SHA}
    record = {'kind': 'conversion_score_one_step_integration_smoke' if args.stop_global_step == 601 else 'conversion_score_matched_development',
        'lineage': lineage, 'configuration': config.as_dict(), 'seed': 1234, 'permutation_seed': PERMUTATION_SEED,
        'permutation_rule': 'CPU Generator seed991000+global_step; ordinary randperm permits fixed/same-speaker positions',
        'method': 'reconstruction adv +2 reconstruction FM +45 conversion clean_x0 sqrt_alpha_bar score; oneD then oneS update',
        'paper_basis': 'https://arxiv.org/html/2508.17868v1#S3.SS1 Eq9',
        'deviations_from_paper': ['fixed original BNF instead of trainable content encoder', 'GAN remains on reconstruction',
            'bounded600-update continuation of ACDD600, not formal100 epochs', 'training-source development exclusions and per-step replay seeds'],
        'training_indices': parent['training_indices'], 'excluded_train_indices': parent['excluded_train_indices'],
        'reconstruction_dev_indices': parent['reconstruction_dev_indices'], 'formal_test_used_for_selection': False,
        'development_contract_sha256': resume.DEVELOPMENT_SHA, 'original_step0_guard_anchor': anchor,
        'guard_ratios': {'mel_l1_upper': 1.10, 'prediction_rms_lower': 0.80}, 'parent_resumed_development': resumed_scores,
        'stop_global_step': args.stop_global_step, 'max_seconds': args.max_seconds, 'check_every': 100,
        'batch_plan': {'first_step': 601, 'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((step-1)/1229)',
                       'crop_seed': '1234+100000+step', 'noise_seed': '1234+200000+step'},
        'fixed_development_inputs': parent['fixed_development_inputs'], 'fixed_development_source_noise_sha256': parent['fixed_development_source_noise_sha256'],
        'monitoring': {'resume_step': 600, 'parent_history_preserved_separately': True,
                       'new_objective_only_trend_windows': 50, 'cross_objective_dist_comparison': False},
        'storage': {'student_snapshots': 'immutable every100 and end', 'full_state': 'atomic rolling latest_full.pt in this run only',
                    'full_temporary_write_bytes': FULL_WRITE_BYTES, 'student_write_bytes': STUDENT_WRITE_BYTES,
                    'system_reserve_bytes': SYSTEM_RESERVE_BYTES, 'low_disk': 'save full-only if safe then stop; otherwise preserve previous full and report unsaved steps'},
        'file_sha256': dependencies, 'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda}}
    require_checkpoint_space(DEV, include_student=True)  # 所有验真和模型加载后，再次检查可用空间。
    output.mkdir()  # 门禁全部通过才创建新的run，不会污染旧实验。
    baseline.save_json(output / 'run_contract.json', record)
    contract_sha = training.canonical_json_sha256(record)
    training.atomic_torch_save({'batch': dev_batch, 'source_noise': dev_noise.cpu()}, output / 'fixed_development_inputs.pt')
    monitor = start_boundary_monitor(output, parent_windows)
    training.restore_rng_state(parent_rng, device)  # 固定32诊断结束后恢复父完整RNG，然后按绝对步规则训练。
    del parent_rng
    baseline.save_json(output / 'resume_state.json', {'global_step': 600, 'next_global_step': 601, 'development': resumed_scores})
    final_step, last_saved_step, last_scores = 600, 600, resumed_scores  # 未产生新full时父600就是最后安全恢复点。
    reason, last_receipt = 'matched_global_step_budget_complete', None
    stop = {'requested': False}
    def stop_at_boundary(signum, frame):
        stop['requested'] = True  # 信号处理不保存半个D/S更新。
    signal.signal(signal.SIGTERM, stop_at_boundary)
    signal.signal(signal.SIGINT, stop_at_boundary)
    started = time.monotonic()

    def persist_current():
        nonlocal last_saved_step, last_receipt, last_scores
        if last_saved_step == final_step:
            return  # 最后100步边界已保存时，不重复覆盖不可变S快照。
        last_scores = baseline.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
        item = {'step': final_step, 'elapsed_seconds': time.monotonic() - started, **last_scores}
        with (output / 'development.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item) + '\n')
        def record_full_commit(receipt):
            nonlocal last_saved_step, last_receipt
            last_saved_step = receipt['global_step']  # full成功后立即推进，不能等S快照或日志也成功。
            last_receipt = receipt  # 保留同一字典引用，后续SHA/阶段变化可在异常报告中看到。
        save_checkpoint(output, final_step, last_scores, student, discriminator, optimizer,
                        d_optimizer, config, contract_sha, lineage, anchor, on_full_committed=record_full_commit)
        baseline.save_json(output / 'status.json', {'status': 'running', **item, 'latest_full': last_receipt})
        print(json.dumps(item), flush=True)

    try:
        for step in steps:
            if stop['requested'] or time.monotonic() - started >= args.max_seconds:
                reason = 'requested_stop' if stop['requested'] else 'time_budget_complete'
                break
            try:
                require_checkpoint_space(output, include_student=True)  # 每批前保留安全落盘能力，空间不足不再追加更新。
            except InsufficientDisk:
                reason = 'disk_space_guard'
                break
            indices = plan.for_step(step)
            batch = baseline.batch_for(dataset, indices, 1234 + 100000 + step)
            training.seed_everything(1234 + 200000 + step)  # 原ACDD控制组相同的源噪声/score噪声/t流。
            losses = conversion.train_batch_conversion_score(batch, teacher, student, vocoder, discriminator, diffusion,
                optimizer, d_optimizer, mean, std, device, config, permutation_seed=PERMUTATION_SEED, global_step=step)
            final_step = step  # 只有完整一次D更新和一次学生更新后才记完成。
            monitor.observe(step, 0, losses)  # 非有限即时报警，持续上涨只比较新CV窗口。
            with (output / 'step_losses.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'step': step, 'indices': indices, 'elapsed_seconds': time.monotonic() - started, **losses}) + '\n')
            if step % 100 == 0 or step == args.stop_global_step or stop['requested'] or time.monotonic() - started >= args.max_seconds:
                persist_current()
                try:
                    resume.assert_guard(last_scores, anchor)
                except RuntimeError:
                    reason = 'original_step0_development_regression_guard'
                    break
                if last_receipt['stop_for_disk']:
                    reason = 'disk_space_guard'
                    break
        persist_current()  # 超时/信号/低空间也尝试保存最后完整批次；已保存的边界不重复写。
        try:
            resume.assert_guard(last_scores, anchor)  # 非100步边界的结束检查也必须服从原保护线。
        except RuntimeError:
            reason = 'original_step0_development_regression_guard'
        if last_receipt and last_receipt['stop_for_disk']:
            reason = 'disk_space_guard'
        if reason == 'matched_global_step_budget_complete' and args.stop_global_step == 601:
            reason = 'one_step_integration_smoke_complete'
        for name, model in (('teacher', teacher), ('vocoder', vocoder)):
            if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Frozen model boundary changed: ' + name)
        resume.verify_files(dependencies)  # 结束时再次核验旧权重、数据、源代码和所有冻结模块。
    except InsufficientDisk as error:
        reason = 'disk_space_guard_last_full_preserved'  # 空间不够不得冒险挤占3GiB系统预留。
        write_disk_stop_or_report_failure(output, monitor, final_step, last_saved_step, last_receipt, parent_path, error)
    except Exception as error:
        report_run_failure(output, monitor, final_step, last_saved_step, last_receipt, parent_path, error)
        raise  # GPU异常/半步失败绝不保存成可续训完整状态或自动重启。
    try:
        monitor.finish(reason, final_step, 0)
        if last_receipt:
            full_path = Path(last_receipt['path'])
            frozen_sha = baseline.sha256_file(full_path)
            if frozen_sha != last_receipt['checkpoint_sha256']:
                raise RuntimeError('Final rolling checkpoint changed before freeze')
            full_path.chmod(0o444)  # 本run结束后不再滚动替换，保留最后完整状态的只读归档。
            baseline.save_json(output / 'final_full_frozen.json', dict(last_receipt, frozen=True, sha256=frozen_sha))
        summary = {'status': reason, 'global_step': final_step, 'start_after_global_step': 600,
            'completed_new_steps': final_step - 600, 'last_saved_global_step': last_saved_step,
            'unsaved_completed_steps': final_step - last_saved_step, 'stop_global_step': args.stop_global_step,
            'elapsed_seconds': time.monotonic() - started, 'initial_parent_development': resumed_scores,
            'final_development': last_scores, 'original_step0_guard_anchor': anchor, 'latest_full': last_receipt,
            'run_contract_sha256': contract_sha, 'lineage': lineage}
        baseline.save_json(output / 'status.json', summary)  # 完成标志最后写；前置metadata失败不留下成功completion。
        baseline.save_json(output / 'completion.json', summary)
        print('CONVERSION_SCORE_DEVELOPMENT_FINISHED', reason, final_step, flush=True)
    except Exception as error:
        report_run_failure(output, monitor, final_step, last_saved_step, last_receipt, parent_path, error)
        raise  # 收尾metadata失败同样是失败，不能只因模型落盘就声称候选全部完成。


if __name__ == '__main__':
    main()  # CPU导入测试不构建真实模型；只有主线程明确运行命令才启动GPU实验。
