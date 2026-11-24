# 固定 BNF 的研究对照：保留重建 GAN/FM，仅把分数监督改到跨说话人转换输出。
# 依据 FasterVoiceGrad (2025) §3.1 Eq9；本混合目标不等同于完整 ADCD 或 2024 原始复现。
# 本文件没有主入口、数据加载或自动训练；只能由获准的租用服务器驱动显式调用。
"""Independent conversion-score batch helper; no automatic experiment launch."""

from typing import Any, Dict, Mapping, Sequence, Tuple  # 限定批次、身份列表和返回损失的接口。

import torch  # 张量和自动求导只在明确获准的租用服务器执行。
from torch import nn  # 教师、学生、声码器和判别器沿用生产模块对象。

from fastvoicegrad import train as training  # 复用现有串行 D/VJP、反归一化和输入检查。
from fastvoicegrad.objectives import make_student_prediction, score_distillation_loss  # 不另写扩散数学。


def sample_target_permutation(
    batch_size: int,  # batch 中样本数；正式调用保持 32。
    *,
    permutation_seed: int,  # 本研究分支独立登记的重排种子，不使用 crop/noise 种子流。
    global_step: int,  # 绝对训练步，恢复后可以直接重建同一个重排。
) -> torch.Tensor:
    # CPU 专用 Generator 不触碰 Python、NumPy、默认 CPU 或任何 CUDA 随机状态。
    if batch_size <= 0 or global_step < 1 or permutation_seed < 0:
        raise ValueError("Positive batch/global step and nonnegative permutation seed required")
    seed = int(permutation_seed) + int(global_step)  # 同一个种子与绝对步始终给出相同配对。
    if seed >= 2**63:
        raise ValueError("Permutation seed plus global step exceeds signed int64 range")
    generator = torch.Generator(device="cpu")  # 创建独立生成器不会消耗默认生成器的随机数。
    generator.manual_seed(seed)  # 只设置这一条局部随机流。
    return torch.randperm(batch_size, generator=generator, device="cpu")  # 普通均匀重排，允许固定点。


def validate_permutation(
    permutation: torch.Tensor,  # 目标索引表，例如位置 i 使用 speaker[permutation[i]]。
    batch_size: int,  # 必须是一一重排同一个 batch，不能重复抽取有利目标。
) -> None:
    if permutation.device.type != "cpu" or permutation.dtype != torch.long:
        raise ValueError("Target permutation must be a CPU int64 tensor")
    if permutation.shape != (batch_size,):
        raise ValueError("Target permutation has the wrong batch length")
    if not torch.equal(torch.sort(permutation).values, torch.arange(batch_size)):
        raise ValueError("Target indices must be a true permutation of the batch")


def permutation_metrics(
    permutation: torch.Tensor,  # CPU 索引不需要同步整批特征到本地。
    speaker_names: Sequence[str],  # 使用真实说话人编号，而非“索引不同”冒充“身份不同”。
) -> Dict[str, float]:
    count = len(speaker_names)  # 一个说话人可能在同一批中出现多句。
    validate_permutation(permutation, count)  # 记录比例前也拒绝缺失或重复目标索引。
    if not all(isinstance(name, str) and name for name in speaker_names):
        raise ValueError("Each sample must carry its nonempty spk_name")
    indices = permutation.tolist()  # 仅 32 个整数，用于记录身份和固定点比例。
    different = sum(speaker_names[i] != speaker_names[j] for i, j in enumerate(indices))
    fixed = sum(i == j for i, j in enumerate(indices))  # 固定点不会被强行改成其他人。
    return {
        "conversion_different_speaker_fraction": different / float(count),  # 实际异说话人占比。
        "conversion_fixed_index_fraction": fixed / float(count),  # 索引原位不动的占比。
    }


def conversion_score_loss(
    student: nn.Module,  # 仍使用原始学生结构，不增加内容编码器。
    teacher: nn.Module,  # 参数冻结的同一个 epoch500 教师。
    diffusion: Any,  # 与重建分支共用的 1000 步日程。
    x_zero: torch.Tensor,  # 原 batch 的源 mel，不按目标说话人重排它。
    speaker: torch.Tensor,  # 原 batch 每条源语音的说话人嵌入。
    bnf: torch.Tensor,  # 原 batch 源内容 BNF；始终不重排内容。
    source_noise: torch.Tensor,  # 与重建分支逐元素完全相同的起始噪声。
    score_noise: torch.Tensor,  # 原训练顺序采得的独立分数噪声。
    score_timesteps: torch.Tensor,  # 原训练顺序采得的每例随机教师时刻。
    start_step: int,  # 正式调用固定 950。
    permutation: torch.Tensor,  # 唯一重排对象是 speaker 的 batch 维。
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_permutation(permutation, x_zero.shape[0])  # 在执行任何模型前向前检查配对。
    target_speaker = speaker.index_select(0, permutation.to(speaker.device))  # 转换的目标身份条件。
    converted, _ = make_student_prediction(
        student, diffusion, x_zero, target_speaker, bnf, start_step, source_noise
    )  # source mel、BNF、起始 x950 都与重建分支相同，只改身份条件。
    loss, target = score_distillation_loss(
        converted, teacher, diffusion, score_timesteps, score_noise, target_speaker, bnf,
        target_mode="clean_x0",  # 保持当前已核验的教师干净 mel 目标解释。
        weight_mode="sqrt_alpha_bar",  # 2025 Eq9 的累计信号幅度权重。
    )  # 生产函数在教师整条分支 no_grad；转换输出左侧仍保留学生梯度。
    return loss, target, converted  # 目标和预测返回供小测试检查边界，调用者不必存盘。


def _validate_update_ownership(
    teacher: nn.Module, student: nn.Module, vocoder: nn.Module, discriminator: nn.Module,
    student_optimizer: torch.optim.Optimizer, discriminator_optimizer: torch.optim.Optimizer,
) -> None:
    # 在第一次 D 更新前拒绝把冻结模型或错误参数登记到任何优化器。
    model_ids = [set(map(id, model.parameters())) for model in (teacher, student, vocoder, discriminator)]
    if any(model_ids[i] & model_ids[j] for i in range(4) for j in range(i)):
        raise ValueError("Teacher/student/vocoder/discriminator may not share parameter objects")
    for optimizer, expected in ((student_optimizer, model_ids[1]), (discriminator_optimizer, model_ids[3])):
        actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Optimizer ownership must exactly match its student or discriminator")
    for name, frozen in (("teacher", teacher), ("vocoder", vocoder)):
        if any(module.training for module in frozen.modules()):
            raise ValueError(f"Frozen {name} and all its submodules must be in eval mode")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in frozen.parameters()):
            raise ValueError(f"Frozen {name} must have requires_grad=False and no existing gradient")


def train_batch_conversion_score(
    batch: Mapping[str, Any],  # 与生产 train_batch 相同的完整 batch32，加原有 spk_name 元数据。
    teacher: nn.Module,  # 不更新的教师。
    student: nn.Module,  # 只有学生 Adam 更新这一组参数。
    vocoder: nn.Module,  # 冻结声码器，仍允许波形损失对其 mel 输入求导。
    discriminator: nn.Module,  # 仍使用原 MPD/MRD；先 D 更新，再学生更新。
    diffusion: Any,  # 复用原扩散日程。
    student_optimizer: torch.optim.Optimizer,  # 保留调用方恢复的学生 Adam 动量。
    discriminator_optimizer: torch.optim.Optimizer,  # 保留调用方恢复的 D Adam 动量。
    mel_mean: torch.Tensor,  # 沿用原反归一化均值。
    mel_std: torch.Tensor,  # 沿用原反归一化标准差。
    device: torch.device,  # 实验为租用 GPU；本辅助模块测试只在租用 CPU 执行。
    config: Any,  # 原配置必须显式是 clean_x0、sqrt_alpha_bar、2/45、950。
    *,
    permutation_seed: int,  # 调用驱动必须把这个独立随机种子写入新实验合同。
    global_step: int,  # 已恢复 checkpoint 后的绝对下一步，不能从 1 重置。
) -> Dict[str, float]:
    # 拒绝无意恢复历史目标或引入新的损失尺度；这不是通用调参入口。
    expected = ("clean_x0", "sqrt_alpha_bar", 2.0, 45.0, 950, 1000, 32)
    actual = (config.score_target, config.score_weight, config.lambda_fm, config.lambda_dist,
              config.student_start_step, config.diffusion_steps, config.batch_size)
    if actual != expected:
        raise ValueError(f"Conversion-score diagnostic requires fixed clean/sqrt/2/45/950 settings: {actual}")
    _validate_update_ownership(teacher, student, vocoder, discriminator, student_optimizer, discriminator_optimizer)
    tensors = training._slice_batch(batch, device)  # 原生产形状检查仍要求 [32,80,128]。
    x_zero, bnf, speaker = tensors["mel"], tensors["bnf"], tensors["spk_emb"]
    names = batch.get("spk_name")  # 原数据集已经提供真实身份编号，不猜测嵌入的相等含义。
    if not isinstance(names, (list, tuple)) or len(names) != x_zero.shape[0]:
        raise ValueError("A complete spk_name list is required to report actual conversion coverage")
    permutation = sample_target_permutation(
        x_zero.shape[0], permutation_seed=permutation_seed, global_step=global_step
    )  # 先在独立 CPU 流重排，不改变随后任何源噪声、分数噪声或时刻。
    coverage = permutation_metrics(permutation, names)  # 在 D step 之前验证身份记录完整。
    source_noise = torch.randn_like(x_zero)  # 与生产 train_batch 的第一次随机抽样完全一致。
    score_noise = torch.randn_like(x_zero)  # 第二次随机抽样仍为独立教师分数噪声。
    score_timesteps = torch.randint(1, config.diffusion_steps + 1, (config.batch_size,), device=device, dtype=torch.long)

    training.set_requires_grad(discriminator, True)  # 原 D 更新阶段，不给学生回传。
    discriminator_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        reconstruction_for_d, _ = make_student_prediction(
            student, diffusion, x_zero, speaker, bnf, config.student_start_step, source_noise
        )  # D 仍然评价源说话人重建，没有偷偷替换成转换假样本。
        real_waveform = vocoder(training.denormalize_mel(x_zero, mel_mean, mel_std))  # 源 GT mel 经同一个冻结 G。
        fake_proxy = vocoder(training.denormalize_mel(reconstruction_for_d, mel_mean, mel_std))  # 源身份重建经 G。
    training._assert_finite("D real waveform", real_waveform)
    training._assert_finite("D fake waveform", fake_proxy)
    discriminator_loss = training.serial_discriminator_backward(discriminator, real_waveform, fake_proxy)
    discriminator_optimizer.step()  # 全部分支梯度累积完成后，D 只更新一次。
    discriminator_optimizer.zero_grad(set_to_none=True)
    discriminator_loss_value = float(discriminator_loss.detach())  # 日志仍记录本次更新前的 D loss。
    del reconstruction_for_d, discriminator_loss  # 释放 D 阶段的临时预测。

    training.set_requires_grad(discriminator, False)  # 学生阶段 D 仅提供输入梯度。
    student_optimizer.zero_grad(set_to_none=True)
    adversarial_loss, fm_loss, waveform_gradient = training.serial_student_discriminator_gradient(
        discriminator, real_waveform, fake_proxy, config.lambda_fm
    )  # 真实和生成 FM 始终同一句话、同源身份；波形梯度已经包含 adv + 2FM。
    reconstruction, _ = make_student_prediction(
        student, diffusion, x_zero, speaker, bnf, config.student_start_step, source_noise
    )  # 和 D 阶段逐元素相同的重建，但这次保留学生计算图。
    fake_waveform = vocoder(training.denormalize_mel(reconstruction, mel_mean, mel_std))
    training._assert_finite("student reconstruction", reconstruction)
    training._assert_finite("student fake waveform", fake_waveform)
    if not torch.equal(fake_waveform.detach(), fake_proxy):
        difference = float(torch.max(torch.abs(fake_waveform.detach() - fake_proxy)))
        raise RuntimeError(f"Reconstruction waveform VJP proxy changed: max_abs_diff={difference:.9g}")
    del real_waveform, fake_proxy  # 波形 VJP 已准备好，释放两份无梯度代理。

    distillation_loss, teacher_target, converted = conversion_score_loss(
        student, teacher, diffusion, x_zero, speaker, bnf, source_noise,
        score_noise, score_timesteps, config.student_start_step, permutation,
    )  # 唯一变化：这一次教师调用监督转换输出，不再计算重建 score。
    training._assert_finite("student converted mel", converted)
    training._assert_finite("teacher conversion target", teacher_target)
    total_loss = training.acdd_student_loss(
        adversarial_loss, fm_loss, distillation_loss, lambda_fm=config.lambda_fm, lambda_dist=config.lambda_dist
    )  # 混合总目标为 reconstruction adv + 2 reconstruction FM + 45 conversion score。
    for name, value in (("student adversarial loss", adversarial_loss), ("raw feature matching loss", fm_loss),
                        ("raw conversion score loss", distillation_loss), ("student total loss", total_loss)):
        training._assert_finite(name, value)  # 不允许非有限分项进入学生优化器。
    torch.autograd.backward(
        (fake_waveform, distillation_loss),
        (waveform_gradient, distillation_loss.new_tensor(config.lambda_dist)),
    )  # 重建图只接收波形梯度；转换图只接收 45 倍 score 梯度；没有额外学生 step。
    gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
    if not gradients or not any(bool(torch.count_nonzero(gradient)) for gradient in gradients):
        raise RuntimeError("Student received no nonzero gradient")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError("Student gradient contains NaN or Inf")
    for name, frozen in (("teacher", teacher), ("vocoder", vocoder), ("discriminator", discriminator)):
        if any(parameter.grad is not None for parameter in frozen.parameters()):
            raise RuntimeError(f"Frozen {name} unexpectedly received gradients in student step")
    student_optimizer.step()  # 一次完整 batch 最后只执行一次学生 Adam。
    student_optimizer.zero_grad(set_to_none=True)
    training.set_requires_grad(discriminator, True)  # 为下一个 batch 恢复 D 可训练状态。
    return {
        "student_total": float(total_loss.detach()),  # 新混合目标；不能与旧总 loss 混成同一种实验曲线。
        "student_adv": float(adversarial_loss.detach()),  # 原重建 GAN 项。
        "fm_raw": float(fm_loss.detach()),  # 原重建 FM 项。
        "fm_weighted": float(config.lambda_fm * fm_loss.detach()),  # 权重保持 2。
        "dist_raw": float(distillation_loss.detach()),  # 此字段在新合同中明确表示 conversion score。
        "dist_weighted": float(config.lambda_dist * distillation_loss.detach()),  # 权重保持 45。
        "discriminator": discriminator_loss_value,  # 原 D 损失口径。
        **coverage,  # 实际异说话人比例和固定索引比例一起返回，防止无效转换被隐藏。
    }
