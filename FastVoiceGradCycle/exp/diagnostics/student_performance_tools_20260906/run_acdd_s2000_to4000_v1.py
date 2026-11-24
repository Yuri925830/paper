# 从已完成ACDD2000恢复到4000；不改蒸馏目标，不接着训练转换分数候选。
"""Bounded unchanged-ACDD continuation from complete step2000 state."""

import argparse  # 父身份、预算和GPU授权均须显式给出。
import copy  # 原S0锚独立保存，不按2000步重基。
import hashlib  # 导入前验证冻结辅助模块。
import json  # 只在新run写合同、日志和检查点记录。
import os  # 查询实际可用磁盘，不删除任何旧数据。
import platform  # 锁定原训练运行时。
import signal  # 在完整批次边界响应停止。
import sys  # 导入项目之前拒绝Windows执行。
import time  # 有限3600秒预算。
from pathlib import Path  # 只接受已核实父目录和新的直接子目录。

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
MODULE_SHAS = {
    'resume_acdd_development.py': '7b81570909cdf39cfa16c563c7c302e7c3886e5551060917ad61de7e8baf0014',
    'run_acdd_development.py': '5cd72f6199e542ecb64a49ed9c70883a4f9b6ea2d39d0faf4fb2a652ba0e2395',
}  # 只复用已冻结的ACDD恢复、数据计划和生产训练路径。
for name, expected in MODULE_SHAS.items():
    if hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen ACDD helper changed: ' + name)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch  # 只有main获准后才构建GPU模型，CPU测试只用小网络。
import resume_acdd_development as resume

baseline, training = resume.baseline, resume.baseline.training
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
PARENT = DEV / 'acdd_clean_resume_s2000_v1'
PARENT_FULL_SHA = '77de35a153636d8156a45c6c23b21191b2c663d55963df874e17946e3a7ff72c'
PARENT_CONTRACT_SHA = 'f7d351f786c9cb7f4583327a8ff7a4b6d2c551c40bdbf36ce9e54dbe1fad933b'
PARENT_CONTRACT_FILE_SHA = '4ede290fc6c98e91fe91608410f74827d205d343b96137f9a57882b1ecd6a80e'
PARENT_WINDOWS_SHA = 'b00a6203ccc6a1f4a1156c3ef344905cb3bed210c2bc719435c55775492b7e5e'
GIB = 1024**3
FULL_WRITE_BYTES, STUDENT_WRITE_BYTES, SYSTEM_RESERVE_BYTES = int(1.5 * GIB), int(0.35 * GIB), 3 * GIB


class InsufficientDisk(RuntimeError):
    pass  # 不足空间时保留旧完整状态，不能挤占系统余量继续写。


def available_bytes(directory):
    value = os.statvfs(directory)
    return value.f_bavail * value.f_frsize  # 当前文件系统可用字节，而非总容量。


def require_space(directory, include_student):
    need = FULL_WRITE_BYTES + SYSTEM_RESERVE_BYTES + (STUDENT_WRITE_BYTES if include_student else 0)
    free = available_bytes(directory)
    if free < need:
        raise InsufficientDisk(f'Need {need} free bytes for safe checkpoint; available={free}')
    return free


def require_segment_capacity(directory, stop_step):
    # 正段保留一个full、3000/4000两个S，另留临时full与系统3GiB；smoke前还计入其永久full+S。
    need = 2 * FULL_WRITE_BYTES + 2 * STUDENT_WRITE_BYTES + SYSTEM_RESERVE_BYTES
    if stop_step == 2001:
        need += FULL_WRITE_BYTES + STUDENT_WRITE_BYTES
    free = available_bytes(directory)
    if free < need:
        raise InsufficientDisk(f'Smoke/segment storage plan requires {need} free bytes; available={free}')
    return {'available_bytes': free, 'required_bytes': need, 'includes_future_segment_after_smoke': stop_step == 2001}


def validate_launch(parent_path, parent_sha, contract_sha, stop_step, max_seconds):
    if parent_path.resolve() != PARENT / 'full_step_002000.pt' or parent_sha != PARENT_FULL_SHA:
        raise RuntimeError('Only the frozen complete ACDD2000 checkpoint is accepted')
    if contract_sha != PARENT_CONTRACT_SHA:
        raise RuntimeError('Wrong ACDD2000 parent contract')
    if stop_step not in (2001, 4000) or not 1 <= max_seconds <= 3600:
        raise RuntimeError('Only explicit2001 smoke or4000 endpoint is allowed, at most3600 seconds')
    return range(2001, stop_step + 1)  # 不重放2000，更不能重新从1步生成数据。


def should_snapshot(step, finishing):
    return step % 1000 == 0 or finishing  # 中间每100只滚动full；整1000及实际结束保留独立S。


def start_monitor(output, windows_path):
    (output / 'loss_windows.jsonl').write_bytes(windows_path.read_bytes())  # 同一目标，原2000之前窗口全部继承。
    monitor = training.LossMonitor(output, window_steps=50, resume_step=2000)
    if not monitor.windows or monitor.windows[-1]['global_step'] != 2000:
        raise RuntimeError('Expected complete unchanged-objective loss history through step2000')
    monitor.last_global_step = 2000
    return monitor  # 不清空旧dist或total趋势，也不改变报警规则。


def save_checkpoint(output, step, scores, student, discriminator, optimizer, d_optimizer,
                    config, contract_sha, lineage, anchor, *, include_student, on_full_committed=None):
    full_path, student_path = output / 'latest_full.pt', output / f'student_step_{step:06d}.pt'
    if full_path.is_symlink() or (include_student and (student_path.exists() or student_path.is_symlink())):
        raise RuntimeError('Unsafe full target or existing immutable student snapshot')
    requested_student = include_student
    try:
        before = require_space(output, include_student)
    except InsufficientDisk:
        if not include_student:
            raise
        before = require_space(output, False)  # 仅够完整恢复状态时先保住full，明确停止且报告缺S快照。
        include_student = False
    common = {'kind': 'acdd_development_checkpoint', 'student': student.state_dict(), 'epoch': 0,
        'global_step': step, 'configuration': config.as_dict(), 'development': scores,
        'run_contract_sha256': contract_sha, 'lineage': lineage, 'original_step0_guard_anchor': anchor,
        'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar'}
    full = dict(common, discriminator=discriminator.state_dict(), student_optimizer=optimizer.state_dict(),
                discriminator_optimizer=d_optimizer.state_dict(), rng_state=training.capture_rng_state())
    receipt = {'global_step': step, 'path': str(full_path), 'checkpoint_sha256': None,
        'run_contract_sha256': contract_sha, 'full_committed': False, 'write_stage': 'full_state_write',
        'student_snapshot': None, 'student_snapshot_requested': requested_student,
        'stop_for_disk': requested_student and not include_student, 'free_bytes_before': before}
    training.atomic_torch_save(full, full_path)  # 只原子覆盖本run的full，所有父归档保持只读。
    receipt['full_committed'], receipt['write_stage'] = True, 'full_state_committed'
    if on_full_committed is not None:
        on_full_committed(receipt)  # full完成立即推进内存恢复游标，不能等metadata成功才记账。
    receipt['write_stage'] = 'full_sha256'
    receipt['checkpoint_sha256'] = baseline.sha256_file(full_path)
    receipt['write_stage'] = 'initial_receipt_write'
    baseline.save_json(output / 'latest_full.json', receipt)
    if include_student:
        if available_bytes(output) < STUDENT_WRITE_BYTES + SYSTEM_RESERVE_BYTES:
            receipt['stop_for_disk'] = True
        else:
            receipt['write_stage'] = 'student_snapshot_write'
            training.atomic_torch_save(common, student_path)  # 保存mapping；没有D、优化器或RNG，避免空间膨胀。
            receipt['student_snapshot'] = {'path': str(student_path), 'sha256': None, 'committed': True}
            receipt['write_stage'] = 'student_snapshot_sha256'
            receipt['student_snapshot']['sha256'] = baseline.sha256_file(student_path)
    receipt['write_stage'] = 'final_receipt_write'
    baseline.save_json(output / 'latest_full.json', receipt)
    receipt['write_stage'] = 'checkpoint_history_write'
    with (output / 'checkpoint_history.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(receipt) + '\n')
    receipt['write_stage'] = 'complete'
    return receipt


def report_failure(output, monitor, final_step, saved_step, receipt, error):
    record = {'status': 'failed', 'completed_global_step': final_step, 'last_saved_global_step': saved_step,
        'unsaved_completed_steps': final_step - saved_step, 'latest_full': receipt,
        'parent_checkpoint': str(PARENT / 'full_step_002000.pt'), 'error_type': type(error).__name__, 'error': str(error)}
    print('ACDD_CONTINUATION_FAILED ' + json.dumps(record), flush=True)  # 即使磁盘不能写，终端仍有真实恢复点。
    try:
        monitor.finish('failed', final_step, 0)
    except Exception as secondary:
        print('FAILURE_MONITOR_WRITE_ERROR ' + repr(secondary), flush=True)
    try:
        baseline.save_json(output / 'status.json', record)
    except Exception as secondary:
        print('FAILURE_STATUS_WRITE_ERROR ' + repr(secondary), flush=True)
    return record  # main仍抛出原异常，不把附属写盘失败声称为候选完成。


def write_disk_stop(output, monitor, final_step, saved_step, receipt, error):
    try:
        baseline.save_json(output / 'disk_stop.json', {'error': str(error), 'completed_global_step': final_step,
            'last_saved_global_step': saved_step, 'unsaved_completed_steps': final_step - saved_step})
    except Exception as metadata_error:
        report_failure(output, monitor, final_step, saved_step, receipt, metadata_error)
        raise  # except内部的第二次写盘失败也必须诚实报告。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--parent-checkpoint-sha256', required=True)
    parser.add_argument('--parent-contract-sha256', required=True)
    parser.add_argument('--stop-global-step', type=int, choices=(2001, 4000), required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--development-contract', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-approved', action='store_true')  # CPU通过不等于GPU已获准自动启动。
    args = parser.parse_args()
    steps = validate_launch(args.parent_checkpoint, args.parent_checkpoint_sha256, args.parent_contract_sha256,
                            args.stop_global_step, args.max_seconds)
    if not args.run_approved:
        raise RuntimeError('Explicit GPU run approval is required')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)  # 与推理/评估互斥，锁持有到本进程退出。
    if training.git_head(ROOT) != baseline.EXPECTED_HEAD:
        raise RuntimeError('Production source HEAD changed')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Use the original myconda training runtime')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Original myconda interpreter and rental CUDA GPU required')
    output = args.output_dir.resolve()
    if output.parent != DEV or output.exists():
        raise RuntimeError('Output must be a new direct child of DEV; all old runs are protected')
    capacity = require_segment_capacity(DEV, args.stop_global_step)  # 先统筹完整smoke和正段，绝不降低安全阈值。
    dependencies = {str(PARENT / 'full_step_002000.pt'): PARENT_FULL_SHA,
        str(PARENT / 'run_contract.json'): PARENT_CONTRACT_FILE_SHA, str(PARENT / 'loss_windows.jsonl'): PARENT_WINDOWS_SHA,
        str(resume.PARENT / 'initial_full.pt'): resume.ANCHOR_SHA,
        str(resume.PARENT / 'run_contract.json'): resume.PARENT_CONTRACT_FILE_SHA,
        **{str(TOOLS / name): expected for name, expected in MODULE_SHAS.items()}}
    resume.verify_files(dependencies)
    parent = json.loads((PARENT / 'run_contract.json').read_text(encoding='utf-8'))
    grand = json.loads((resume.PARENT / 'run_contract.json').read_text(encoding='utf-8'))
    if training.canonical_json_sha256(parent) != PARENT_CONTRACT_SHA or training.canonical_json_sha256(grand) != resume.PARENT_CONTRACT_SHA:
        raise RuntimeError('Parent/grandparent contract canonical identity changed')
    resume.verify_files(parent['file_sha256'])  # 继承已核验教师、G、mean/std、特征和所有代码来源。
    dependencies.update(parent['file_sha256'])
    development, dev_hashes = resume.load_development(args.development_contract.resolve())
    dependencies.update(dev_hashes)
    if parent['training_indices'] != development['retained_train_indices'] or parent['reconstruction_dev_indices'] != development['reconstruction_dev_indices']:
        raise RuntimeError('The39335 training examples or held-out52 utterances changed')
    config = baseline.replace(baseline.PAPER_STUDENT_CONFIG, num_workers=0)
    if config.as_dict() != parent['configuration'] or config.as_dict() != grand['configuration'] or grand['seed'] != 1234:
        raise RuntimeError('ACDD objective, optimizer, BNF or normalization configuration changed')
    if training.collect_source_fingerprints(ROOT) != grand['source_sha256']:
        raise RuntimeError('Production source fingerprints no longer match original ACDD')
    dependencies[str(Path(__file__).resolve())] = baseline.sha256_file(Path(__file__))
    payload = resume.load_torch_checkpoint(PARENT / 'full_step_002000.pt')
    initial = resume.load_torch_checkpoint(resume.PARENT / 'initial_full.pt')
    if initial['global_step'] != 0 or initial['run_contract_sha256'] != resume.PARENT_CONTRACT_SHA:
        raise RuntimeError('OriginalS0 protection anchor has invalid lineage')
    anchor = copy.deepcopy(initial['development'])
    del initial
    if anchor != parent['original_step0_guard_anchor'] or anchor['mel_l1'] != 0.19941353797912598 or anchor['prediction_rms'] != 0.9994737505912781:
        raise RuntimeError('OriginalS0 guard changed')
    resume.assert_guard(payload['development'], anchor)
    if payload['configuration'] != config.as_dict():
        raise RuntimeError('Parent full state configuration differs from its contract')
    torch.set_num_threads(2)  # 与原开发驱动相同，不改TF32、cudnn或精度模式。
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
    resume.restore_training_state(payload, student, discriminator, optimizer, d_optimizer, 2000, PARENT_CONTRACT_SHA, device)
    parent_rng, parent_scores = payload['rng_state'], copy.deepcopy(payload['development'])
    del payload  # 已恢复S/D和双Adam，不继续保留大型CPU副本。
    protocol = development['reconstruction_protocol']
    if protocol['source_noise_device'] != 'cuda':
        raise RuntimeError('Fixed32 noise generator device changed')
    dev_batch = baseline.batch_for(dataset, development['reconstruction_dev_indices'], protocol['crop_seed'])
    dev_noise = torch.randn(dev_batch['mel'].shape, device=device,
        generator=torch.Generator(device=device).manual_seed(protocol['source_noise_seed']))
    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if any(tensor_sha(dev_batch[key]) != parent['fixed_development_inputs'][key] for key in ('mel', 'bnf', 'spk_emb')) or tensor_sha(dev_noise) != parent['fixed_development_source_noise_sha256']:
        raise RuntimeError('Fixed32 features or full-batch CUDA noise changed')
    resumed_scores = baseline.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
    if any(abs(resumed_scores[key] - parent_scores[key]) > 1e-7 for key in ('mel_l1', 'prediction_rms', 'reference_rms', 'cosine_reference')):
        raise RuntimeError('RestoredS2000 does not replay the saved fixed32 readout')
    resume.assert_guard(resumed_scores, anchor)
    plan = resume.GlobalBatchPlan(parent['training_indices'])  # 2001在第1轮，2459必须切到第2轮。
    if plan.batches_per_pass != 1229:
        raise RuntimeError('Expected1229 full batches per shuffle pass')
    resume.verify_files(dependencies)
    lineage = {'parent_directory': str(PARENT), 'parent_checkpoint': str(PARENT / 'full_step_002000.pt'),
        'parent_checkpoint_sha256': PARENT_FULL_SHA, 'parent_contract_sha256': PARENT_CONTRACT_SHA,
        'original_step0_checkpoint_sha256': resume.ANCHOR_SHA, 'start_after_global_step': 2000}
    record = {'kind': 'acdd2001_integration_smoke' if args.stop_global_step == 2001 else 'acdd2000_to4000_development',
        'lineage': lineage, 'configuration': config.as_dict(), 'method': 'unchanged production train_batch; reconstruction adv+2FM+45clean_x0 sqrt_alpha_bar score',
        'seed': 1234,  # 原ACDD种子；每轮shuffle和每步crop/noise在其基础上按下方绝对步规则派生。
        'development_contract_file_sha256': resume.DEVELOPMENT_SHA,  # 真实开发契约文件字节SHA；不使用去空白后的canonical SHA。
        'training_indices': parent['training_indices'], 'excluded_train_indices': parent['excluded_train_indices'],
        'reconstruction_dev_indices': parent['reconstruction_dev_indices'], 'formal_test_used_for_selection': False,
        'original_step0_guard_anchor': anchor, 'guard_ratios': {'mel_l1_upper': 1.10, 'prediction_rms_lower': 0.80},
        'parent_resumed_development': resumed_scores, 'stop_global_step': args.stop_global_step, 'max_seconds': args.max_seconds,
        'batch_plan': {'first_step': 2001, 'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((step-1)/1229)',
                       'crop_seed': '1234+100000+step', 'noise_seed': '1234+200000+step'},
        'fixed_development_inputs': parent['fixed_development_inputs'], 'fixed_development_source_noise_sha256': parent['fixed_development_source_noise_sha256'],
        'monitor': {'window_steps': 50, 'resume_step': 2000, 'same_objective_parent_windows_sha256': PARENT_WINDOWS_SHA},
        'storage': {'full': 'atomic rolling latest_full every100 and actual end', 'student': 'immutable every1000 and actual end',
            'full_write_bytes': FULL_WRITE_BYTES, 'student_write_bytes': STUDENT_WRITE_BYTES, 'system_reserve_bytes': SYSTEM_RESERVE_BYTES,
            'initial_capacity_plan': capacity, 'low_space': 'save full-only when safe and stop; otherwise retain previous saved state'},
        'scope': 'bounded unchanged-ACDD development continuation; no guarantee of improved CER or paper parity',
        'file_sha256': dependencies, 'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda}}
    require_segment_capacity(DEV, args.stop_global_step)
    output.mkdir()  # 所有门禁和父重放通过后才创建新run。
    baseline.save_json(output / 'run_contract.json', record)
    contract_sha = training.canonical_json_sha256(record)
    training.atomic_torch_save({'batch': dev_batch, 'source_noise': dev_noise.cpu()}, output / 'fixed_development_inputs.pt')
    monitor = start_monitor(output, PARENT / 'loss_windows.jsonl')
    training.restore_rng_state(parent_rng, device)
    del parent_rng
    final_step, saved_step, saved_student_step, last_scores, receipt = 2000, 2000, None, resumed_scores, None
    reason, stop = 'global_step_budget_complete', {'requested': False}
    baseline.save_json(output / 'resume_state.json', {'global_step': 2000, 'next_global_step': 2001, 'development': resumed_scores})
    def stop_at_boundary(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, stop_at_boundary)
    signal.signal(signal.SIGINT, stop_at_boundary)
    started = time.monotonic()

    def persist(finishing):
        nonlocal saved_step, saved_student_step, last_scores, receipt, reason
        if final_step == 2000 or (saved_step == final_step and (not finishing or saved_student_step == final_step
                or (receipt is not None and receipt['stop_for_disk']))):
            return  # 不复制父2000；已保存的结束S也不能再次覆盖。
        last_scores = baseline.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
        try:
            resume.assert_guard(last_scores, anchor)
        except RuntimeError:
            reason, finishing = 'original_step0_development_regression_guard', True  # 越线成为实际结束，仍保留本步完整证据。
        item = {'step': final_step, 'elapsed_seconds': time.monotonic() - started, **last_scores}
        with (output / 'development.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item) + '\n')
        def committed(value):
            nonlocal saved_step, receipt
            saved_step, receipt = value['global_step'], value  # full原子完成即记账，后续metadata异常也不回退。
        save_checkpoint(output, final_step, last_scores, student, discriminator, optimizer, d_optimizer,
            config, contract_sha, lineage, anchor, include_student=should_snapshot(final_step, finishing), on_full_committed=committed)
        if receipt['student_snapshot'] is not None:
            saved_student_step = final_step
        if receipt['stop_for_disk']:
            reason = 'disk_space_guard'
        baseline.save_json(output / 'status.json', {'status': 'running', **item, 'latest_full': receipt})
        print(json.dumps(item), flush=True)

    try:
        for step in steps:
            if stop['requested'] or time.monotonic() - started >= args.max_seconds:
                reason = 'requested_stop' if stop['requested'] else 'time_budget_complete'
                break
            try:
                require_space(output, True)  # 每批都预留实际结束时full+S的安全保存能力。
            except InsufficientDisk:
                reason = 'disk_space_guard'
                break
            indices = plan.for_step(step)
            batch = baseline.batch_for(dataset, indices, 1234 + 100000 + step)
            training.seed_everything(1234 + 200000 + step)
            losses = training.train_batch(batch, teacher, student, vocoder, discriminator, diffusion,
                optimizer, d_optimizer, mean, std, device, config)  # 原生产函数原样调用，不换任何梯度项。
            final_step = step
            monitor.observe(step, 0, losses)
            with (output / 'step_losses.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'step': step, 'indices': indices, 'elapsed_seconds': time.monotonic() - started, **losses}) + '\n')
            finishing = step == args.stop_global_step or stop['requested'] or time.monotonic() - started >= args.max_seconds
            if step % 100 == 0 or finishing:
                persist(finishing)
                if reason in ('original_step0_development_regression_guard', 'disk_space_guard'):
                    break
        persist(True)  # 实际结束也保留S；若刚保存了同一步full但没有S，可再原子滚动一次补齐。
        for name, model in (('teacher', teacher), ('vocoder', vocoder)):
            if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Frozen model boundary changed: ' + name)
        resume.verify_files(dependencies)
    except InsufficientDisk as error:
        reason = 'disk_space_guard_last_full_preserved'
        write_disk_stop(output, monitor, final_step, saved_step, receipt, error)
    except Exception as error:
        report_failure(output, monitor, final_step, saved_step, receipt, error)
        raise
    try:
        if args.stop_global_step == 2001 and reason == 'global_step_budget_complete':
            reason = 'one_step_integration_smoke_complete'
        monitor.finish(reason, final_step, 0)
        if receipt:
            path = Path(receipt['path'])
            if baseline.sha256_file(path) != receipt['checkpoint_sha256']:
                raise RuntimeError('Final full checkpoint changed before freeze')
            path.chmod(0o444)  # 结束后本run不再修改完整归档。
            baseline.save_json(output / 'final_full_frozen.json', dict(receipt, frozen=True))
        summary = {'status': reason, 'global_step': final_step, 'start_after_global_step': 2000,
            'completed_new_steps': final_step - 2000, 'last_saved_global_step': saved_step,
            'unsaved_completed_steps': final_step - saved_step, 'latest_full': receipt, 'stop_global_step': args.stop_global_step,
            'elapsed_seconds': time.monotonic() - started, 'initial_parent_development': resumed_scores,
            'final_development': last_scores, 'original_step0_guard_anchor': anchor, 'run_contract_sha256': contract_sha, 'lineage': lineage}
        baseline.save_json(output / 'status.json', summary)
        baseline.save_json(output / 'completion.json', summary)  # 所有收尾成功后才写完成记录。
        print('ACDD_CONTINUATION_FINISHED', reason, final_step, flush=True)
    except Exception as error:
        report_failure(output, monitor, final_step, saved_step, receipt, error)
        raise


if __name__ == '__main__':
    main()  # 不因文件准备好而自动启动任何GPU任务。
