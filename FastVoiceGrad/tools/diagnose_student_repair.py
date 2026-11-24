# 仅在训练来源语音上进行新旧蒸馏设置的短程对照，不能替代正式 unseen 指标。
"""Paired short ACDD repair check on training-only utterances, on Matpool."""
# 解析诊断目录和步数，不使用正式测试集来选修复设置。
import argparse
# 两组实验之间主动释放已经不用的 Python 对象。
import gc
# 把样本、测量值和实验组设置保存成可复查的 JSON。
import json
# 保留标准库随机数接口；实际全局播种由 seed_everything 统一完成。
import random
# 只读调用 Git，记录当时所在的提交。
import subprocess
# 测量各组耗时并给诊断记录添加时间戳。
import time
# 从不可变配置复制一个新实例，只替换本组目标和权重模式。
from dataclasses import replace
# 用路径对象定位权重、数据和诊断输出。
from pathlib import Path

# NumPy 只负责可重复的样本索引打乱。
import numpy as np
# PyTorch 负责模型前向、梯度更新和张量测量。
import torch
# 将单句样本字典堆叠成每批 32 句的张量字典。
from torch.utils.data import default_collate

# 与正式训练共用数据读取、标准化、裁剪及 BNF 对齐实现。
from voicegrad.dataset import VoiceGradDataset
# 与正式训练共用扩散时间表和加噪/去噪系数。
from voicegrad.diffusion import VoiceGradDiffusion
# 所有诊断计算同样限制在当前登记的租用服务器。
from voicegrad.runtime_guard import assert_rented_server
# 读取学生训练固定参数，避免短程检查悄悄改变网络或优化器设置。
from fastvoicegrad.config import PAPER_STUDENT_CONFIG
# 复用正式模型加载和校验流程，保持教师、学生、声码器身份一致。
from fastvoicegrad.components import (
    # 分别构造判别器、加载冻结声码器，以及从教师权重初始化学生。
    build_paper_discriminator, load_frozen_vocoder, load_teacher_and_student,
    # 读取归一化统计与可信检查点，计算文件指纹以记录来源。
    load_mel_statistics, load_torch_checkpoint, sha256_file,
)
# 用学生完成一次去噪，诊断时固定输入和噪声。
from fastvoicegrad.objectives import make_student_prediction
# 更新步骤、随机种子及运行锁均使用正式训练的同一实现。
from fastvoicegrad.train import train_batch, seed_everything, acquire_runtime_locks
# 记录每组短程损失，并检查是否出现连续窗口上升。
from fastvoicegrad.loss_monitor import LossMonitor
# 对实际教师比较不同目标/权重组合的数值方向，不更新教师参数。
from tests.test_fastvoicegrad_score_repair import diagnose_teacher_targets


# 主流程顺序运行旧设置与修复设置，二者共享初始化、样本和随机数安排。
def main():
    # 构造命令行接口，允许明确选择诊断输出与短程长度。
    parser = argparse.ArgumentParser()
    # 权威运行代码只位于当前租用服务器的这个项目根目录。
    parser.add_argument("--project-root", type=Path, default=Path("/root/FsatVoiceGrad"))
    # 每次诊断必须指定独立目录；下文会拒绝覆盖已有诊断。
    parser.add_argument("--output-dir", type=Path, required=True)
    # 默认每组做 120 次更新，用于尽早发现幅度塌缩，不代表完整训练。
    parser.add_argument("--steps", type=int, default=120)
    # 将命令行文字解析成路径和整数。
    args = parser.parse_args()
    # 转成绝对路径，避免调用目录影响文件定位。
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    # 检查租用主机身份，失败立即终止，不能改在本地跑。
    assert_rented_server(root)
    # 持有与正式训练相同的互斥锁，防止同时开启冲突的训练任务。
    locks = acquire_runtime_locks(root)
    # 诊断目录存在就拒绝继续，保护之前的样本与实验记录。
    if out.exists():
        # 给出具体目录，要求改用新目录而非覆盖证据。
        raise RuntimeError(f"Refusing to overwrite existing diagnostic: {out}")
    # 仅为这一次对照创建输出目录。
    out.mkdir(parents=True)
    # 控制 CPU 线程数，避免小算子耗时受到线程争用影响。
    torch.set_num_threads(4)
    # 网络训练使用租用服务器 GPU，不提供本地 CPU 兜底执行。
    device = torch.device("cuda")
    # 固定模型加载与样本裁剪阶段的随机状态。
    seed_everything(1234)
    # 除目标和蒸馏权重外，两组沿用相同学生配置。
    config = PAPER_STUDENT_CONFIG
    # 固定第 500 epoch 教师，和既有教师客观评估采用相同起点。
    teacher_path = root / "exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt"
    # 严格加载教师，并将其参数复制给学生；不使用已退化的旧学生。
    teacher, student, teacher_metadata = load_teacher_and_student(teacher_path, config, device)
    # 在 CPU 上保存独立克隆，后续更新不会污染第二组的共同起点。
    initial_student_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    # 选择本实验自训的修改版 HiFi-GAN 目录。
    vocoder_dir = root / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
    # 加载并冻结声码器，其权重不随任一学生实验组更新。
    vocoder, vocoder_config, vocoder_metadata = load_frozen_vocoder(
        # 指定最终 250 万步生成器与其原始配置，不能串用公开版权重。
        root, vocoder_dir / "generator_step_2500000.pt", vocoder_dir / "config.json",
        # 验证最终证明、训练步数及现场实现后才允许生成波形。
        vocoder_dir / "final_checkpoint_attestation.json", config, device,
    )
    # 元数据已完成校验，释放无需长期保留的变量引用。
    del teacher_metadata, vocoder_metadata
    # 建立同一条余弦扩散时间表，所有组使用相同加噪系数。
    diffusion = VoiceGradDiffusion(config.diffusion_steps, offset=config.cosine_offset).to(device)
    # 读取训练集 mel 均值/标准差，声码器前需要逆标准化。
    mean, std = load_mel_statistics(root / "data", device)
    # 只打开 train 划分，每句取 128 帧，不从正式 unseen 集挑样本。
    dataset = VoiceGradDataset(root / "data", "train", segment_length=128,
                               # 检查训练说话人数，并沿用现有 BNF 时间对齐设定。
                               expected_speakers=100, bnf_alignment=config.bnf_alignment)
    # 独立固定种子打乱句子索引，让诊断样本选择可以重放。
    selection = np.random.default_rng(20260906).permutation(len(dataset))
    # 前 32 句只用于固定输入诊断；下方短程训练不会更新这些句子。
    dev_indices = selection[:32].tolist()
    # 后续每步分配 32 条不同训练语音，两组完全复用这一索引序列。
    train_indices = selection[32:32 + args.steps * 32].tolist()
    # 确认句子数量够用，避免无意复用样本或产生更小的尾批。
    if len(train_indices) != args.steps * 32:
        # 不满足成对诊断设计时直接停止。
        raise RuntimeError("Not enough distinct training utterances")
    # 提前读取并裁剪诊断语音，后续每次测量使用同一片段。
    dev = default_collate([dataset[i] for i in dev_indices])
    # 提前将每批训练样本也固定下来，避免两组随机裁剪不一致。
    batches = [default_collate([dataset[i] for i in train_indices[j:j + 32]])
               # 每隔 32 个索引建立一批，顺序与上面的打乱结果一致。
               for j in range(0, len(train_indices), 32)]
    # 将干净 mel、说话人嵌入和 BNF 放到 GPU；三种条件始终对应同一批语音。
    dev_x, dev_s, dev_b = (dev[k].to(device) for k in ("mel", "spk_emb", "bnf"))
    # 诊断时的高斯噪声只生成一次，防止噪声变化伪装成质量变化。
    dev_noise = torch.randn(dev_x.shape, device=device,
                            # 独立生成器不随训练随机流向前移动。
                            generator=torch.Generator(device=device).manual_seed(55001))
    # 保存本次设计及输入身份，使“相同样本条件”有可检查的记录。
    manifest = {
        # 标明只做训练来源的短程检查，不能当成论文测试指标。
        "purpose": "training-only paired diagnostic; not formal test metrics",
        # 记录每组更新次数及批大小，便于比较耗时与训练量。
        "steps_per_arm": args.steps, "batch_size": 32,
        # 保存原数据集中的索引，方便回查到底使用了哪些语音。
        "dev_indices": dev_indices, "train_indices": train_indices,
        # 显式检查诊断句与更新句没有交集。
        "dev_and_update_disjoint": not set(dev_indices).intersection(train_indices),
        # 教师哈希绑定实际权重字节，避免仅凭文件名判断实验起点。
        "teacher_sha256": sha256_file(teacher_path),
        # 两个修改同时切换，因而这项实验不单独估计各自的贡献。
        "arms": [["legacy_adjacent", "nominal_alpha"], ["clean_x0", "sqrt_alpha_bar"]],
        # 记录提交号；若工作区有未提交修改，还需结合事后源码快照查看。
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    # 先保存设计，再开始训练，保留实验实际使用的索引与设置。
    (out / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # 诊断教师目标时让三个网络都处于推理模式。
    teacher.eval(); vocoder.eval(); student.eval()
    # 这一步只查看目标数值，不构建梯度图，也不更新任何网络。
    with torch.no_grad():
        # 用固定噪声从第 950 步做一次学生预测，取得目标检查的共同输入。
        initial_prediction, _ = make_student_prediction(student, diffusion, dev_x, dev_s, dev_b, 950, dev_noise)
        # 取前 8 句比较多个教师目标模式，降低检查开销。
        probes = diagnose_teacher_targets(teacher, initial_prediction[:8], dev_s[:8], dev_b[:8])
    # 保存目标检查数值，便于把数学问题和训练结果联系起来。
    (out / "target_probe.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")

    # 每次固定样本测量都关闭梯度，避免诊断本身改变训练。
    @torch.no_grad()
    # label 区分实验组；step 表示已完成的优化器更新次数。
    def evaluate(label, step):
        # 记住调用前的训练/推理状态，以便测量后恢复。
        previous = student.training
        # 关闭训练模式专有行为，让各次固定输入可比。
        student.eval()
        # 每次都复用相同的干净输入、说话人、BNF 和噪声。
        prediction, _ = make_student_prediction(student, diffusion, dev_x, dev_s, dev_b, 950, dev_noise)
        # 这些数值均有固定参考，不依赖正在变化的判别器。
        record = {
            # 标记组别、更新进度和测量时间。
            "arm": label, "step": step, "time": time.time(),
            # 平均绝对误差衡量预测偏离真实 mel 的大小，越低越好。
            "fixed_dev_mel_l1": float((prediction - dev_x).abs().mean()),
            # 预测 mel 的均方根用于观察输出幅度是否塌缩。
            "prediction_rms": float(prediction.square().mean().sqrt()),
            # 同一批真实 mel 的均方根提供固定的幅度参照。
            "reference_rms": float(dev_x.square().mean().sqrt()),
            # 余弦相似度主要反映数值方向是否接近真实 mel，不能单独衡量音质。
            "cosine_reference": float(torch.nn.functional.cosine_similarity(
                # 每句展平后求相似度，再对 32 句平均，避免把不同句混成一个向量。
                prediction.flatten(1), dev_x.flatten(1), dim=1).mean()),
        }
        # 恢复调用前的状态，下一批训练继续按原模式执行。
        student.train(previous)
        # 追加记录而非覆盖，从而保留整条固定样本变化轨迹。
        with (out / "fixed_dev.jsonl").open("a", encoding="utf-8") as handle:
            # 一行一个完整 JSON 对象，读取时可按测量逐条处理。
            handle.write(json.dumps(record) + "\n")
        # 立即刷新日志，方便远程及时查看测量结果。
        print("FIXED_DEV " + json.dumps(record), flush=True)
        # 把记录交给组末汇总，避免重新计算得到不同状态。
        return record

    # 收集两组的起点、终点和耗时。
    results = []
    # 先旧写法再修复写法，二者不会继承彼此训练后的参数。
    for target, weight in manifest["arms"]:
        # 文件和日志使用明确模式名，降低混淆实验组的风险。
        label = target + "_" + weight
        # 每组重置随机流，使新判别器初始化与训练噪声可比。
        seed_everything(778899)
        # 严格恢复共同的学生起点，防止上一组结果泄漏到下一组。
        student.load_state_dict(initial_student_state, strict=True)
        # 只让学生训练；教师和声码器保持推理模式且参数已冻结。
        student.train(); teacher.eval(); vocoder.eval()
        # 复制配置并只改变目标与权重两项，不改网络、学习率或数据处理。
        current_config = replace(config, score_target=target, score_weight=weight)
        # 每组重新创建判别器，并使用相同种子得到相同初始化。
        discriminator = build_paper_discriminator(root, vocoder_config, current_config, device)
        # 判别器需要学习区分真实还原波形和学生生成波形。
        discriminator.train()
        # 新建学生优化器，不沿用上一组的动量或更新历史。
        optimizer_s = torch.optim.Adam(student.parameters(), lr=config.learning_rate,
                                      # Adam 的两个动量参数与正式训练一致。
                                      betas=(config.adam_beta1, config.adam_beta2))
        # 判别器持有独立的优化器，不混入学生或冻结模型的参数。
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate,
                                      # 保持两组完全相同的优化设置。
                                      betas=(config.adam_beta1, config.adam_beta2))
        # 短程只有 120 步，因此每 20 步统计；正式训练另外使用 50 步窗口。
        monitor = LossMonitor(out / label, window_steps=20)
        # 更新前测量起点，用于确认两组确实从相同预测出发。
        first = evaluate(label, 0)
        # 计时包含训练与各次固定样本测量。
        began = time.time()
        # 从 1 开始计数，与“已完成多少次更新”的日志含义一致。
        for step, batch in enumerate(batches, 1):
            # 完全复用正式的判别器/学生更新逻辑，只传入该组目标设置。
            losses = train_batch(batch, teacher, student, vocoder, discriminator, diffusion,
                                 # 传入两套优化器、mel 统计、GPU 与该组配置。
                                 optimizer_s, optimizer_d, mean, std, device, current_config)
            # 记录本步损失；这里 epoch=0 表示短程诊断，不伪装成正式训练轮数。
            monitor.observe(step, 0, losses)
            # 每隔 20 步观察一次固定输入变化，避免每步额外测量拖慢训练。
            if step % 20 == 0:
                # 误差和幅度同时检查，可区分 loss 变化与输出塌缩。
                evaluate(label, step)
                # 日志写出组别和已完成步数，以便外部轮询核实进度。
                print("PILOT_PROGRESS " + json.dumps({"arm": label, "step": step,
                      # 保留总步数、实测耗时及各项损失，不仅记录总损失。
                      "steps": args.steps, "seconds": time.time() - began, "losses": losses}), flush=True)
        # 组末再保存一次最终测量，确保步数不是 20 的倍数时也有终点。
        last = evaluate(label, args.steps)
        # 将这一组标记为诊断完成，不标记为正式训练完成。
        monitor.finish("diagnostic_complete", args.steps, 0)
        # 汇总共同起点和各自终点，后续制图直接读取这些测量。
        results.append({"arm": label, "initial": first, "final": last,
                        # 每组分别计时，可用于估计正式训练的大致时长。
                        "elapsed_seconds": time.time() - began})
        # 明确写入 diagnostic_only，避免把短程权重当成最终学生模型。
        torch.save({"purpose": "diagnostic_only", "step": args.steps,
                    # 除学生权重外记录使用的目标和权重模式。
                    "student": student.state_dict(), "target": target, "weight": weight},
                   # 每组各保存一份文件，不覆盖另一组。
                   out / f"{label}.pt")
        # 丢弃本组判别器和优化器，下一组必须重新初始化。
        del discriminator, optimizer_s, optimizer_d
        # 清理无用对象及空闲显存缓存，降低两组交替时的显存压力。
        gc.collect(); torch.cuda.empty_cache()
    # 两组都结束后才生成最终汇总，避免把半次实验误读为完整对照。
    (out / "paired_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # 通过明确完成标志和实测汇总告知远程调用者。
    print("PAIRED_DIAGNOSTIC_COMPLETE " + json.dumps(results), flush=True)


# 只有直接作为程序运行时才执行诊断；导入模块不会启动 GPU 任务。
if __name__ == "__main__":
    # 进入上述成对短程检查流程。
    main()
