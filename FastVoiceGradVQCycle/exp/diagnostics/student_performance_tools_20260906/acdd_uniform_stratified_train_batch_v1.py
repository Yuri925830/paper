"""保留原ACDD训练步，只给时间步分层对照增加一个明确注入点。"""

import hashlib  # 导入前核对生产文件，禁止同名代码被暗中替换。
import random  # 每批独立局部随机流，不消耗原Python/NumPy/Torch状态。
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run project code only on the registered Matpool rental')
PRODUCTION = ROOT / 'fastvoicegrad/train.py'
PRODUCTION_SHA = 'd38131843b0b1ba51a10b65bc8bea226c2cecd1ec339d2bb04db5ede2febbaec'
if hashlib.sha256(PRODUCTION.read_bytes()).hexdigest() != PRODUCTION_SHA:
    raise RuntimeError('Frozen production train.py changed')
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from fastvoicegrad import train as production

# 所有原低层函数直接复用；这里只绑定本模块名称，不改生产模块或Torch全局属性。
PaperWaveformDiscriminator = production.PaperWaveformDiscriminator
VoiceGradDiffusion = production.VoiceGradDiffusion
FastVoiceGradStudentConfig = production.FastVoiceGradStudentConfig
_slice_batch = production._slice_batch
_assert_finite = production._assert_finite
set_requires_grad = production.set_requires_grad
make_student_prediction = production.make_student_prediction
denormalize_mel = production.denormalize_mel
serial_discriminator_backward = production.serial_discriminator_backward
serial_student_discriminator_gradient = production.serial_student_discriminator_gradient
score_distillation_loss = production.score_distillation_loss
acdd_student_loss = production.acdd_student_loss
SAMPLER_SEED = 73083328


def tensor_identity(value):
    # SHA取真实连续张量字节；转换到CPU仅为记录，不抽取任何随机数。
    array = value.detach().cpu().contiguous().numpy()
    return {'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def stratified_timesteps(global_step):
    if type(global_step) is not int or global_step < 1:
        raise ValueError('Require a positive absolute global step')
    rng = random.Random(SAMPLER_SEED + global_step)
    labels = [layer for layer in range(10) for _ in range(3)]  # 每层先放3句。
    bonus = rng.sample(range(10), 2)  # 两个不同层各加1句；没有跨批配额。
    labels.extend(bonus)
    rng.shuffle(labels)  # 32个标签随机分到32句，不能按说话人挑选。
    values = [rng.randint(100*layer+1, 100*layer+100) for layer in labels]
    return torch.tensor(values, dtype=torch.long), {'seed': SAMPLER_SEED+global_step,
        'labels': labels, 'bonus_bins': bonus, 'timesteps': values,
        'bin_counts': [labels.count(layer) for layer in range(10)]}


def inject_timesteps(original, override, source_noise, score_noise, receipt):
    # 即使候选要换t，调用此函数时原source、score和uniform t都已经真实消费。
    if original.shape != (32,) or original.dtype != torch.long:
        raise RuntimeError('Original time draw must be32 int64 values')
    if override is None:
        used = original
    else:
        if not isinstance(override, torch.Tensor) or override.shape != original.shape or override.dtype != torch.long or override.requires_grad:
            raise RuntimeError('Time override must be a detached32-element int64 tensor')
        if bool(((override<1)|(override>1000)).any()):
            raise RuntimeError('Time override must remain within1..1000')
        used = override.to(device=original.device)
    if receipt is not None:
        if not isinstance(receipt, dict) or receipt:
            raise RuntimeError('Supply a fresh empty sampler receipt')
        receipt.update({'source_noise': tensor_identity(source_noise), 'score_noise': tensor_identity(score_noise),
            'original_uniform_t': tensor_identity(original), 'used_t': tensor_identity(used),
            'original_uniform_values': original.detach().cpu().tolist(), 'used_values': used.detach().cpu().tolist(),
            'original_bin_counts': torch.bincount((original-1)//100, minlength=10).cpu().tolist(),
            'used_bin_counts': torch.bincount((used-1)//100, minlength=10).cpu().tolist(),
            'original_three_draws_preserved': True})
    return used


def train_batch_with_timesteps(
    batch: Mapping[str, Any],  # DataLoader 产生的一批训练样本及元数据。
    teacher: nn.Module,  # 冻结的 VoiceGrad epoch500 教师 phi。
    student: nn.Module,  # 从教师初始化、当前要训练的 FastVoiceGrad theta。
    vocoder: nn.Module,  # 冻结的自训练修改版 HiFi-GAN Generator V。
    discriminator: PaperWaveformDiscriminator,  # 可训练波形 D：MPD5+MRD3。
    diffusion: VoiceGradDiffusion,  # T=1000 的余弦扩散时间表。
    student_optimizer: torch.optim.Optimizer,  # 只拥有学生参数的 Adam。
    discriminator_optimizer: torch.optim.Optimizer,  # 只拥有 D 参数的 Adam。
    mel_mean: torch.Tensor,  # 训练集 80 维 mel 均值，广播形状与 mel 兼容。
    mel_std: torch.Tensor,  # 训练集 80 维 mel 标准差。
    device: torch.device,  # 正式运行是租用 A100 的 CUDA 设备。
    config: FastVoiceGradStudentConfig,  # 论文设置与已声明复现选择。
    *,  # 新接口仅增加关键字参数，原生产位置参数与计算顺序完全保留。
    score_timesteps_override=None,  # None保留原均匀抽样；候选传入独立分层时间步。
    sampler_receipt=None,  # 可选字典记录已真实消费的三次随机抽样。
# 结束 train_batch 的参数声明；返回类型为 Dict[str, float]，具体内容由下方实现给出。
) -> Dict[str, float]:
    # 搬运张量并强制检查 mel[32,80,128]、BNF[32,144,128]、spk[32,256]。
    tensors = _slice_batch(batch, device)
    # x_zero 表示干净、已标准化的目标 mel，即扩散记号中的 x_0。
    x_zero = tensors["mel"]
    # bnf 提供不随目标说话人变化的语音内容条件。
    bnf = tensors["bnf"]
    # speaker 提供 256 维目标说话人身份条件。
    speaker = tensors["spk_emb"]

    # 为一步学生从 S_K=950 构造输入噪声；形状与 x_zero 同为 [32,80,128]。
    source_noise = torch.randn_like(x_zero)
    # 为教师分数蒸馏另采独立噪声，避免和学生起始噪声错误共用。
    score_noise = torch.randn_like(x_zero)
    # 每条样本独立均匀采一个教师扩散时刻 t∈{1,…,1000}。
    score_timesteps = torch.randint(
        1,  # randint 左端包含 1；不抽 t=0。
        config.diffusion_steps + 1,  # 右端不包含，因此最大恰为 T=1000。
        (config.batch_size,),  # 每个 batch32 样本一个时间步，形状 [32]。
        # 本次推理的计算设备。
        device=device,  # 直接在 A100 生成，避免额外传输。
        # 数组数值存储类型。
        dtype=torch.long,  # 时间步作为网络索引必须为整数类型。
    )

    # 唯一新增计算入口：原三次抽样完成后，仅替换教师蒸馏的时间步。
    score_timesteps = inject_timesteps(score_timesteps, score_timesteps_override, source_noise, score_noise, sampler_receipt)

    # 论文式(15)：先用与学生图分离的完整 batch32 真假波形更新判别器一次。
    # 打开 D 参数梯度；上一批学生更新结束时虽已恢复，这里再次显式保证状态。
    set_requires_grad(discriminator, True)
    # 清掉旧 D 梯度，并用 None 节省显存/避免无意累加到下一批。
    discriminator_optimizer.zero_grad(set_to_none=True)
    # D 更新不应给学生/声码器建图，因此整个波形生成放在 no_grad 中。
    with torch.no_grad():
        # 用固定 source_noise 从 S_K=950 一步生成学生 mel，第二返回值此处不需要。
        x_theta_for_d, _ = make_student_prediction(
            student,  # 当前学生网络 theta。
            diffusion,  # 用于按 S_K 加噪和一步反推。
            x_zero,  # 干净训练目标 mel。
            speaker,  # 目标说话人条件。
            bnf,  # 语音内容条件。
            config.student_start_step,  # 论文 K=1 对应起始步 S_K=950。
            source_noise,  # 本批固定的学生输入噪声。
        )
        # 将真实标准化 mel 反归一化，再经冻结 V 合成 D 的真实波形。
        real_waveform_for_d = vocoder(denormalize_mel(x_zero, mel_mean, mel_std))
        # 将学生预测 mel 反归一化，再经同一冻结 V 合成 D 的生成波形。
        fake_waveform_for_d = vocoder(
            # 将学生标准化mel还原到HiFi-GAN训练使用的自然对数mel。
            denormalize_mel(x_theta_for_d, mel_mean, mel_std)
        )
    # 在进入 D 之前确认真实波形全部有限。
    _assert_finite("D real waveform", real_waveform_for_d)
    # 确认学生生成波形没有 NaN/Inf。
    _assert_finite("D fake waveform", fake_waveform_for_d)
    # 逐个 MPD/MRD 分支反传，累加的梯度仍对应完整论文 D 总损失。
    discriminator_loss = serial_discriminator_backward(
        # 判别器接收真实mel合成波形与学生mel合成波形，二者样本一一对应。
        discriminator, real_waveform_for_d, fake_waveform_for_d
    )
    # 所有 8 个分支梯度到齐后只执行一次 Adam 更新。
    discriminator_optimizer.step()
    # 更新后立即清 D 梯度，为学生阶段的“D 无梯度”断言创造明确状态。
    discriminator_optimizer.zero_grad(set_to_none=True)
    # 保存纯 Python D loss 数值供日志；detach 不保留图。
    discriminator_loss_value = float(discriminator_loss.detach())
    # D 用的学生 mel 和损失图不再需要，尽早释放显存。
    del x_theta_for_d, discriminator_loss

    # 论文式(14)：复用无梯度完整波形做代理，让更新后的冻结 D 逐分支贡献精确 VJP。
    # 冻结判别器参数；仍允许对 D 输入波形求导，从而把信号传给学生。
    set_requires_grad(discriminator, False)
    # 清除学生上一批梯度，保证本次更新只由当前 batch32 产生。
    student_optimizer.zero_grad(set_to_none=True)
    # 计算 8 分支 adv、FM 总值以及它们对生成波形的总梯度。
    adversarial_loss, fm_loss, waveform_gradient = (
        # 逐分支求对抗和FM对波形的梯度，控制峰值显存。
        serial_student_discriminator_gradient(
            discriminator,  # 已做完本批 D step 并冻结的判别器。
            real_waveform_for_d,  # 完整 batch32 真实波形代理。
            fake_waveform_for_d,  # 完整 batch32 生成波形代理。
            config.lambda_fm,  # 论文固定特征匹配权重 2。
        )
    )

    # 所有判别器特征图释放后，重新建立一次学生→声码器图，再注入累计波形梯度。
    # 使用完全相同的 x_zero/source_noise/条件重算，因此输出应逐位相同。
    x_theta, _ = make_student_prediction(
        student,  # 参数尚未 step，因此仍是与 D 采样时相同的学生。
        diffusion,  # 相同扩散时间表。
        x_zero,  # 相同干净 mel。
        speaker,  # 相同目标说话人条件。
        bnf,  # 相同内容条件。
        config.student_start_step,  # 相同 S_K=950。
        source_noise,  # 复用本批同一噪声，保证可比。
    )
    # 这次声码器前向保留“波形→mel→学生”的梯度图；V 参数仍冻结。
    fake_waveform = vocoder(denormalize_mel(x_theta, mel_mean, mel_std))
    # 确认重算学生 mel 数值正常。
    _assert_finite("student prediction", x_theta)
    # 确认保留梯度图的生成波形数值正常。
    _assert_finite("student fake waveform", fake_waveform)
    # 重算波形必须与之前代理逐位相等，否则注入 VJP 会对应错误输出点。
    if not torch.equal(fake_waveform.detach(), fake_waveform_for_d):
        # 计算最大绝对差，便于判断是数值微差还是模型含随机层。
        maximum_difference = float(
            # 若两次重算不一致，报告最大逐采样点偏差定位随机性或数值问题。
            torch.max(torch.abs(fake_waveform.detach() - fake_waveform_for_d))
        )
        # 任何非零差异都拒绝继续，严格保证显存优化不改变数学梯度。
        raise RuntimeError(
            # 波形代理与真实反向图输出必须逐位一致，否则预先求出的梯度作用点错误。
            "Recomputed student waveform is not bit-identical to the full-batch "
            # 把不一致的最大差值写入错误，避免继续用错误波形梯度训练。
            f"VJP proxy (max_abs_diff={maximum_difference:.9g})"
        )
    # 等价性确认后释放两份无梯度代理波形。
    del real_waveform_for_d, fake_waveform_for_d
    # 蒸馏梯度只从 L1 中的 x_theta 传回学生；教师重建目标整条分支已停止梯度。
    distillation_loss, _ = score_distillation_loss(
        x_theta,  # 学生一步预测的干净 mel。
        teacher,  # epoch500 冻结 VoiceGrad 教师。
        diffusion,  # T=1000 的同一扩散过程。
        score_timesteps,  # 本批 32 个随机教师时间步。
        score_noise,  # 独立蒸馏噪声 [32,80,128]。
        speaker,  # 目标说话人条件。
        bnf,  # 语音内容条件。
        # 训练和固定测试都显式传入本次选定的教师目标空间。
        target_mode=config.score_target,
        # 显式传入本次alpha权重解释，防止修复配置被默认值覆盖。
        weight_mode=config.score_weight,
    )
    # 仅用于记录/检查地组装论文总标量 L_adv+2L_FM+45L_dist。
    total_loss = acdd_student_loss(
        adversarial_loss,  # 8 个分支未加权学生对抗总和。
        fm_loss,  # 8 个分支未加权特征匹配总和。
        distillation_loss,  # 教师分数蒸馏原始损失。
        lambda_fm=config.lambda_fm,  # 论文权重 2。
        lambda_dist=config.lambda_dist,  # 论文权重 45。
    )
    # 在反传前检查学生目标所有组成，避免坏数进入 Adam 状态。
    for name, value in (
        # 检查学生波形对抗损失是否有限。
        ("student adversarial loss", adversarial_loss),  # L_adv。
        # 检查未乘 2 的特征匹配损失是否有限。
        ("raw feature matching loss", fm_loss),  # L_FM。
        # 检查未乘 45 的蒸馏损失是否有限。
        ("raw score distillation loss", distillation_loss),  # L_dist。
        # 检查三个已加权分项组成的学生总损失是否有限。
        ("student total loss", total_loss),  # L_adv+2L_FM+45L_dist。
    ):
        # 任何 NaN/Inf 都立即报出具体项。
        _assert_finite(name, value)
    # 一次联合反传：波形分支注入预先精确累加的 VJP，蒸馏分支乘 45。
    torch.autograd.backward(
        (fake_waveform, distillation_loss),  # 两个需要向学生回传的图输出。
        (
            waveform_gradient,  # 已包含 adv+2FM 的 dL/dwaveform。
            distillation_loss.new_tensor(config.lambda_dist),  # 蒸馏标量上游梯度固定 45。
        ),
    )

    # 收集所有实际收到梯度的学生参数，用于更新前完整性检查。
    student_gradients = [
        parameter.grad  # 取参数的梯度张量。
        for parameter in student.parameters()  # 遍历学生所有参数。
        if parameter.grad is not None  # 忽略结构上未参与本次计算的参数。
    ]
    # 一个梯度都没有说明计算图被错误切断。
    if not student_gradients:
        # 禁止空更新悄悄计为一个 global_step。
        raise RuntimeError("Student received no gradient")
    # 任何学生梯度出现 NaN/Inf 都不能交给 Adam。
    if any(not bool(torch.isfinite(gradient).all()) for gradient in student_gradients):
        # 立即停止并保留上一轮完整检查点。
        raise FloatingPointError("Student gradient contains NaN or Inf")
    # 如果所有梯度元素全为 0，学生不会学习，通常代表目标或图连接错误。
    if not any(bool(torch.count_nonzero(gradient)) for gradient in student_gradients):
        # 拒绝把无效 step 记录进训练进度。
        raise RuntimeError("All student gradients are zero")
    # 教师严格冻结，因此所有 teacher.parameter.grad 都应保持 None。
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        # 教师收到梯度表示冻结边界被破坏，会额外占显存并违反论文训练方式。
        raise RuntimeError("Frozen teacher unexpectedly received gradients")
    # 声码器虽然允许梯度穿过输入，但自身参数 grad 必须仍为 None。
    if any(parameter.grad is not None for parameter in vocoder.parameters()):
        # 若 V 参数收到梯度，说明误把已训练声码器一起训练了。
        raise RuntimeError("Frozen vocoder unexpectedly received parameter gradients")
    # 学生阶段 D 参数也应无 grad；输入梯度不等于更新 D 参数。
    if any(parameter.grad is not None for parameter in discriminator.parameters()):
        # 有 D 梯度说明冻结切换失败，会混淆两个优化目标。
        raise RuntimeError("Frozen discriminator unexpectedly received gradients in theta step")

    # 所有检查通过后，学生 Adam 对本完整 batch32 只更新一次。
    student_optimizer.step()
    # 清学生梯度，减少显存并防止下批意外累加。
    student_optimizer.zero_grad(set_to_none=True)
    # 恢复 D 可训练状态，为下一个 batch 的判别器更新做准备。
    set_requires_grad(discriminator, True)
    # 返回纯数值损失字典；这些值会先按 batch 再按 epoch 求平均并写 CSV。
    return {
        "student_total": float(total_loss.detach()),  # 完整 ACDD 学生总损失。
        "student_adv": float(adversarial_loss.detach()),  # 未加权对抗项。
        "fm_raw": float(fm_loss.detach()),  # 未加权 FM 项。
        # 记录两倍特征匹配项对总损失的实际贡献，先断开梯度再转标量。
        "fm_weighted": float(config.lambda_fm * fm_loss.detach()),  # 2*FM。
        "dist_raw": float(distillation_loss.detach()),  # 未加权蒸馏项。
        "dist_weighted": float(config.lambda_dist * distillation_loss.detach()),  # 45*蒸馏。
        "discriminator": discriminator_loss_value,  # 本批 D 更新前的总损失。
    }

