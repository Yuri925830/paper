"""在租用服务器进行有步数上限的学生开发实验；绝不使用正式测试集挑模型。"""

import argparse  # 明确本次开发协议、输出目录和计算上限。
import hashlib  # 记录独立脚本与输入文件的内容身份。
import json  # 保存每个检查点的设置、损失和开发集结果。
import random  # 固定同一句语音的裁剪位置和训练顺序。
import signal  # 用户中止后在完整批次边界保存，不留下半个更新。
import sys  # 在导入计算库之前拒绝本地Windows执行。
import time  # 记录每一步用时，估算剩余运行时间。
from dataclasses import replace  # 派生开发配置，不改正式论文配置对象。
from pathlib import Path  # 所有真实路径都在租用服务器内。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一允许使用的项目运行目录。
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run this experiment only on the registered Matpool rental.')
sys.path.insert(0, str(ROOT))  # 只导入服务器现场的模型和数学实现。
import torch  # 全部模型计算在租用服务器执行。
from torch.utils.data import default_collate  # 保留正式batch字段和张量形状。
import fastvoicegrad.train as training  # 复用通过119项测试的真实ACDD更新。
from fastvoicegrad.components import load_teacher_and_student, load_frozen_vocoder, build_paper_discriminator, load_mel_statistics, sha256_file
from fastvoicegrad.config import PAPER_STUDENT_CONFIG  # 论文明确的2/45及Adam参数不变。
from fastvoicegrad.objectives import make_student_prediction  # 开发检查只看固定真实mel误差。
from voicegrad.dataset import VoiceGradDataset  # 保留原BNF、归一化及语音读取方法。
from voicegrad.diffusion import VoiceGradDiffusion  # 与教师共用原扩散日程。

EXPECTED_HEAD = '0a2c646922704e746642f5f33c203869fe696b3f'  # 本次已审计的代码版本。
TEACHER = ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt'
VOCODER_DIR = ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen'
TRAIN_SHA = '6b24abba2bbea1102ebd002980ec9e4ded9357a07e3871d2514c2474c6774e2b'
TEST_SHA = 'dec09943585580c782634900956ef80f2e9e8286ae8db539eebf0a2b2e7471a4'


def save_json(path, value):
    """以原子写入方式保存诊断状态，不改原始实验日志。"""
    training.atomic_write_json(path, value)  # 原函数会flush/fsync并原子替换。


def batch_for(dataset, indices, crop_seed):
    """同一检查步采用独立裁剪种子，重启后也能得到相同片段。"""
    saved = random.getstate()  # 不让开发集读数改变训练裁剪随机流。
    try:
        random.seed(crop_seed)  # 一批的裁剪由明确整数确定。
        return default_collate([dataset[index] for index in indices])
    finally:
        random.setstate(saved)  # 恢复调用者原本的随机状态。


@torch.no_grad()
def evaluate_reconstruction(student, diffusion, batch, noise, config, device):
    """固定真实mel作为参照，避免把判别器变化误当成学生退步。"""
    previous_mode = student.training  # 检查结束后恢复原训练模式。
    student.eval()
    try:
        mel = batch['mel'].to(device)  # 同一批固定开发片段。
        prediction, _ = make_student_prediction(
            student, diffusion, mel, batch['spk_emb'].to(device),
            batch['bnf'].to(device), config.student_start_step, noise,
        )
        errors = (prediction - mel).abs().flatten(1).mean(1)
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError('Non-finite development prediction')
        return {
            'mel_l1': float(errors.mean()),
            'per_utterance_mel_l1': errors.cpu().tolist(),
            'prediction_rms': float(prediction.square().mean().sqrt()),
            'reference_rms': float(mel.square().mean().sqrt()),
            'cosine_reference': float(torch.nn.functional.cosine_similarity(prediction.flatten(1), mel.flatten(1)).mean()),
        }
    finally:
        student.train(previous_mode)  # 只读检查不能改变后续更新模式。


def main():
    """执行固定开发协议的一条短训，不启动100轮正式训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-contract', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--check-every', type=int, default=100)
    parser.add_argument('--max-seconds', type=int, default=3600)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 5000 or args.check_every < 1 or args.max_seconds < 1:
        raise ValueError('Development run requires a bounded positive step/time budget')
    training.assert_rented_server(ROOT)  # 实验室101–104和本地均不能执行。
    locks = training.acquire_runtime_locks(ROOT)  # 独占GPU工作流直到进程退出。
    if training.git_head(ROOT) != EXPECTED_HEAD:
        raise RuntimeError('Source version changed; review the diagnostic before running')
    if not torch.cuda.is_available():
        raise RuntimeError('The rental GPU must be available')
    output = args.output_dir.resolve()  # 不允许写到正式模型或原数据目录。
    if ROOT / 'exp/diagnostics' not in output.parents or output.exists():
        raise RuntimeError('Use a new directory inside exp/diagnostics; existing runs are protected')
    contract_path = args.development_contract.resolve()
    # 这份开发划分已在服务器单独生成并核验；不能临时换句子让结果变好看。
    if sha256_file(contract_path) != '08737551d34e00ba0b80a5e34adbf73fcfc6c70ece6bc8dc4a6c466e9b87afb1':
        raise RuntimeError('Development contract differs from the reviewed fixed split')
    receipt = json.loads((contract_path.parent / 'development_manifest.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'complete':
        raise RuntimeError('Development split was not completely generated')
    for name, expected in receipt['artifacts'].items():
        if sha256_file(contract_path.parent / name) != expected:
            raise RuntimeError('Development artifact changed: ' + name)
    development = json.loads(contract_path.read_text(encoding='utf-8'))
    # 开发协议必须声明索引来自原训练manifest，且所有开发语句都从本次梯度更新中排除。
    excluded = set(development['excluded_train_indices'])
    dev_indices = list(development['reconstruction_dev_indices'])
    if len(dev_indices) != 32 or len(set(dev_indices)) != 32 or not set(dev_indices) <= excluded:
        raise RuntimeError('Expected 32 unique held-out reconstruction utterances')
    for name, expected in [('train_manifest.csv', TRAIN_SHA), ('test_manifest.csv', TEST_SHA)]:
        if sha256_file(ROOT / 'data/metadata' / name) != expected:
            raise RuntimeError('Authoritative train/test manifest changed')
    # 仅开发实验读单进程数据；模型、BNF和全部损失系数维持论文修复版本。
    config = replace(PAPER_STUDENT_CONFIG, num_workers=0)
    if (config.score_target, config.score_weight) != ('clean_x0', 'sqrt_alpha_bar'):
        raise RuntimeError('This development baseline requires the repaired distillation objective')
    torch.set_num_threads(2)  # 控制CPU读取和线性代数线程，GPU承担网络更新。
    training.seed_everything(args.seed)
    device = torch.device('cuda')
    dataset = VoiceGradDataset(ROOT / 'data', 'train', segment_length=128,
                               expected_speakers=100, bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387 or any(type(i) is not int or not 0 <= i < len(dataset) for i in excluded):
        raise RuntimeError('Development indices do not match the 39387-row training manifest')
    # 保存的训练语句名单只在新目录里使用，不更改data下的任何manifest或特征。
    train_indices = [i for i in range(len(dataset)) if i not in excluded]
    if len(excluded) != 52 or train_indices != development['retained_train_indices']:
        raise RuntimeError('Training indices must exactly exclude the fixed development split')
    if len(train_indices) < 32:
        raise RuntimeError('Not enough training utterances remain after development exclusion')
    teacher, student, _ = load_teacher_and_student(TEACHER, config, device)
    vocoder, vconfig, _ = load_frozen_vocoder(
        ROOT, VOCODER_DIR / 'generator_step_2500000.pt', VOCODER_DIR / 'config.json',
        VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device,
    )
    discriminator = build_paper_discriminator(ROOT, vconfig, config, device)
    diffusion = VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = load_mel_statistics(ROOT / 'data', device)
    student.train()  # 教师和G由严格加载器冻结；只有学生和D训练。
    discriminator.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate,
                                 betas=(config.adam_beta1, config.adam_beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate,
                                   betas=(config.adam_beta1, config.adam_beta2))
    # 延续原120步诊断的32句裁剪和CUDA噪声，使本轮起点可与已有证据核对。
    reconstruction_protocol = development['reconstruction_protocol']
    if reconstruction_protocol['source_noise_device'] != 'cuda':
        raise RuntimeError('The reconstruction noise bank must use the recorded CUDA generator')
    dev_batch = batch_for(dataset, dev_indices, reconstruction_protocol['crop_seed'])
    dev_noise = torch.randn(dev_batch['mel'].shape, device=device,
        generator=torch.Generator(device=device).manual_seed(reconstruction_protocol['source_noise_seed']))
    # 固定片段与固定噪声的实际字节也纳入记录，避免仅凭种子声称相同输入。
    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    archived = reconstruction_protocol['archived_input_identity']
    for actual, key in [(dev_batch['mel'], 'mel_sha256'), (dev_batch['bnf'], 'bnf_sha256'), (dev_noise, 'source_noise_sha256')]:
        if tensor_sha(actual) != archived[key]:
            raise RuntimeError('Fixed reconstruction inputs no longer replay the archived 32: ' + key)
    record = {
        'kind': 'student_training_development_only', 'formal_test_used_for_selection': False,
        'git_head': EXPECTED_HEAD, 'driver_sha256': sha256_file(Path(__file__)),
        'development_contract_sha256': sha256_file(contract_path),
        'configuration': config.as_dict(), 'training_indices': train_indices,
        'excluded_train_indices': sorted(excluded), 'reconstruction_dev_indices': dev_indices,
        'max_steps': args.max_steps, 'check_every': args.check_every, 'seed': args.seed,
        'max_seconds': args.max_seconds, 'teacher_sha256': sha256_file(TEACHER),
        'development_regression_guard': {'mel_l1_ratio_above_initial': 1.10,
            'prediction_rms_ratio_below_initial': 0.80,
            'basis': 'predeclared diagnostic safeguards, not paper hyperparameters'},
        'vocoder_sha256': sha256_file(VOCODER_DIR / 'generator_step_2500000.pt'),
        'fixed_development_inputs': {name: tensor_sha(dev_batch[name]) for name in ('mel', 'bnf', 'spk_emb')},
        'fixed_development_source_noise_sha256': tensor_sha(dev_noise),
        'source_sha256': training.collect_source_fingerprints(ROOT),
        'deviations_from_paper': ['student development utterances excluded from gradient updates',
            'bounded steps instead of formal 100 epochs', 'per-step fixed crop/noise seeds',
            'development-only checkpoint selection; no formal unseen selection'],
        'teacher_exposure': 'teacher trained on original train corpus, including these development utterances',
    }
    output.mkdir()  # 完成身份检查后才创建新的实验目录。
    save_json(output / 'run_contract.json', record)
    contract_sha = training.canonical_json_sha256(record)
    initial = evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
    save_json(output / 'initial_development.json', initial)
    # 记录所有可变初始状态，后续不同训练目标可以真正复用相同S/D/Adam起点。
    def checkpoint(path, step, scores, full):
        payload = {'kind': 'development_checkpoint', 'student': student.state_dict(),
                   'epoch': 0, 'global_step': step, 'run_contract_sha256': contract_sha,
                   'development': scores, 'configuration': config.as_dict()}
        if full:
            payload.update(discriminator=discriminator.state_dict(), student_optimizer=optimizer.state_dict(),
                           discriminator_optimizer=d_optimizer.state_dict(), rng_state=training.capture_rng_state())
        training.atomic_torch_save(payload, path)  # 原子保存，旧结果从不覆盖。
    checkpoint(output / 'initial_full.pt', 0, initial, True)
    stop = {'requested': False}
    def stop_at_boundary(signum, frame):
        stop['requested'] = True  # 信号处理器不执行模型或磁盘操作。
    signal.signal(signal.SIGTERM, stop_at_boundary)
    signal.signal(signal.SIGINT, stop_at_boundary)
    monitor = training.LossMonitor(output, window_steps=50, resume_step=0)
    started = time.monotonic()
    best = initial['mel_l1']  # 仅按留出的训练来源开发数据选候选。
    last_scores, final_step, reason = initial, 0, 'step_budget_complete'
    batches_per_pass = len(train_indices) // 32
    for step in range(1, args.max_steps + 1):
        if step == 1 or (step - 1) % batches_per_pass == 0:
            order = list(train_indices)
            random.Random(args.seed + (step - 1) // batches_per_pass).shuffle(order)
        offset = ((step - 1) % batches_per_pass) * 32
        batch = batch_for(dataset, order[offset:offset + 32], args.seed + 100000 + step)
        training.seed_everything(args.seed + 200000 + step)  # 各候选共享每步噪声。
        losses = training.train_batch(batch, teacher, student, vocoder, discriminator,
            diffusion, optimizer, d_optimizer, mean, std, device, config)
        final_step = step
        monitor.observe(step, 0, losses)  # 原报警逻辑保留，持续上涨会写loss_alerts。
        with (output / 'step_losses.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'step': step, 'elapsed_seconds': time.monotonic() - started, **losses}) + '\n')
        # 到完整批次边界才响应用户停止、预算到期或常规检查点。
        reached_time = time.monotonic() - started >= args.max_seconds
        if step % args.check_every == 0 or step == args.max_steps or stop['requested'] or reached_time:
            last_scores = evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
            item = {'step': step, 'elapsed_seconds': time.monotonic() - started, **last_scores}
            with (output / 'development.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(item) + '\n')
            checkpoint(output / f'student_step_{step:06d}.pt', step, last_scores, False)
            checkpoint(output / 'latest_full.pt', step, last_scores, True)
            if last_scores['mel_l1'] < best:
                best = last_scores['mel_l1']
                save_json(output / 'best_development.json', item)  # 只保存选择记录，保留所有候选权重。
            print(json.dumps(item), flush=True)
            save_json(output / 'status.json', {'status': 'running', **item})
            # 预先声明的止损线：明显重建恶化/频谱收缩时先暂停该候选，保存证据再换方案。
            if last_scores['mel_l1'] > initial['mel_l1'] * 1.10 or last_scores['prediction_rms'] < initial['prediction_rms'] * 0.80:
                reason = 'development_regression_guard'
                break
            if stop['requested'] or reached_time:
                reason = 'requested_stop' if stop['requested'] else 'time_budget_complete'
                break
    # 结束时再次检查冻结参数边界；本轮从来没有更新教师或声码器优化器。
    for name, model in [('teacher', teacher), ('vocoder', vocoder)]:
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError(f'{name} freezing boundary changed')
    save_json(output / 'completion.json', {'status': reason, 'global_step': final_step,
              'elapsed_seconds': time.monotonic() - started, 'initial': initial, 'final': last_scores,
              'best_development_mel_l1': best, 'run_contract_sha256': contract_sha})
    monitor.finish(reason, final_step, 0)  # 结束后监控状态不能遗留为running。
    save_json(output / 'status.json', {'status': reason, 'step': final_step, **last_scores})
    print('DEVELOPMENT_RUN_FINISHED', reason, final_step, flush=True)


if __name__ == '__main__':
    main()  # 只有明确运行脚本才会开始这一条有限预算实验。
