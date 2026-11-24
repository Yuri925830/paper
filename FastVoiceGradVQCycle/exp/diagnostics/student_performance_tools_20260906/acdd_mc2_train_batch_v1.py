"""只增加第二份教师监督；MC1实际调用原生产函数，MC2主干可做AST逐句对照。"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Run project code only on the registered Matpool rental')
PRODUCTION_SHA = 'd38131843b0b1ba51a10b65bc8bea226c2cecd1ec339d2bb04db5ede2febbaec'
OBJECTIVES_SHA = '5b1439550205ada6b4f6fca52010963df23f65f28bb185684a912f2c1e9bf58f'
for relative, expected in (('fastvoicegrad/train.py', PRODUCTION_SHA), ('fastvoicegrad/objectives.py', OBJECTIVES_SHA)):
    if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen production source changed: ' + relative)
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from fastvoicegrad import train as production
from fastvoicegrad.objectives import _extract

# 绑定原低层函数，不改生产模块、Torch全局API或模型forward方法。
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
MC2_SEED = 74000000
START, STOP = 3327, 3447


def tensor_identity(value):
    array = value.detach().cpu().contiguous().numpy()
    return {'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def validate_request(global_step, config, device, receipt):
    if type(global_step) is not int or not START < global_step <= STOP:
        raise RuntimeError('Require the next declared absolute step3328..3447')
    expected = {'batch_size': 32, 'diffusion_steps': 1000, 'student_start_step': 950,
                'score_target': 'clean_x0', 'score_weight': 'sqrt_alpha_bar',
                'lambda_fm': 2.0, 'lambda_dist': 45.0}
    if any(getattr(config, name) != value for name, value in expected.items()):
        raise RuntimeError('MC changes may not alter the original ACDD configuration')
    if device.type != 'cuda' and not (device.type == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''):
        raise RuntimeError('CPU is permitted only for explicitly GPU-hidden rental tests')
    if not isinstance(receipt, dict) or receipt:
        raise RuntimeError('Supply one fresh empty MC receipt for every complete batch')


def record_first_draws(source_noise, score_noise, score_timesteps, receipt):
    receipt.update(source_noise=tensor_identity(source_noise),
                   score_noise1=tensor_identity(score_noise),
                   t1=tensor_identity(score_timesteps),
                   t1_values=score_timesteps.detach().cpu().tolist())


def peek_production_draws(batch, device, config):
    """精确预览后恢复Python/NumPy/CPU/全部CUDA RNG；不前向或更新。"""
    x_zero = _slice_batch(batch, device)['mel']
    state = production.capture_rng_state()
    try:
        source_noise = torch.randn_like(x_zero)
        score_noise = torch.randn_like(x_zero)
        score_timesteps = torch.randint(1, config.diffusion_steps + 1, (config.batch_size,),
                                       device=device, dtype=torch.long)
    finally:
        production.restore_rng_state(state, device)
    return source_noise, score_noise, score_timesteps


def original_input_identity(batch, device, config):
    """供driver在首次D/S更新前检查真实3328数据、噪声与t身份，无模型调用。"""
    source_noise, score_noise, timesteps = peek_production_draws(batch, device, config)
    result = {'batch_tensors': {key: tensor_identity(batch[key]) for key in ('mel', 'bnf', 'spk_emb')}}
    record_first_draws(source_noise, score_noise, timesteps, result)
    return result


def second_score_draws(prediction, diffusion, global_step):
    if type(global_step) is not int or not START < global_step <= STOP:
        raise RuntimeError('Private MC2 seed requires the declared absolute global step')
    if diffusion.n_levels != 1000:
        raise RuntimeError('Private MC2 time distribution must remain1..1000')
    generator = torch.Generator(device=prediction.device).manual_seed(MC2_SEED + global_step)
    # 顺序必须是score_noise2然后t2；只使用该局部generator。
    noise = torch.randn(prediction.shape, device=prediction.device, dtype=prediction.dtype, generator=generator)
    timesteps = torch.randint(1, diffusion.n_levels + 1, (prediction.shape[0],),
                              device=prediction.device, dtype=torch.long, generator=generator)
    return noise, timesteps


@torch.no_grad()
def score_record(prediction, target, timesteps, diffusion, loss):
    if target.requires_grad or target.grad_fn is not None:
        raise RuntimeError('MC teacher targets must be completely detached')
    per_l1 = torch.abs(target - prediction.detach()).flatten(1).mean(1)
    weights = diffusion.sqrt_alphas_cumprod.gather(0, timesteps.long()).to(per_l1.dtype)
    per_weighted = weights * per_l1
    if not bool(torch.isfinite(per_weighted).all()) or not bool(torch.isfinite(target).all()):
        raise RuntimeError('Nonfinite MC teacher target or per-utterance loss')
    if not torch.equal(per_weighted.mean(), loss.detach()):
        raise RuntimeError('Recorded MC per-utterance mean differs from the actual loss')
    return {'per_utterance_l1': per_l1.cpu().tolist(), 'weights': weights.cpu().tolist(),
            'per_utterance_weighted_l1': per_weighted.cpu().tolist(),
            'batch_weighted_l1': float(loss.detach()), 'teacher_target': tensor_identity(target)}


def mc2_score_distillation_loss(student_prediction, teacher, diffusion, timesteps, score_noise,
                               speaker_embedding, bnf, *, target_mode, weight_mode, global_step, mc_receipt):
    if target_mode != 'clean_x0' or weight_mode != 'sqrt_alpha_bar':
        raise RuntimeError('MC2 retains clean_x0/sqrt_alpha_bar')
    first, first_target = score_distillation_loss(
        student_prediction, teacher, diffusion, timesteps, score_noise, speaker_embedding, bnf,
        target_mode=target_mode, weight_mode=weight_mode)
    noise2, t2 = second_score_draws(student_prediction, diffusion, global_step)
    second, second_target = score_distillation_loss(
        student_prediction, teacher, diffusion, t2, noise2, speaker_embedding, bnf,
        target_mode=target_mode, weight_mode=weight_mode)
    # 两份各自逐句L1→权重→batch均值后混合；绝不先平均teacher targets。
    mixed = (first + second) / 2.0
    mc_receipt.update(mc_samples=2, global_step=global_step, private_seed=MC2_SEED + global_step,
                      score_noise2=tensor_identity(noise2), t2=tensor_identity(t2),
                      t2_values=t2.detach().cpu().tolist(),
                      first=score_record(student_prediction, first_target, timesteps, diffusion, first),
                      second=score_record(student_prediction, second_target, t2, diffusion, second),
                      dist1=float(first.detach()), dist2=float(second.detach()), dist_mixed=float(mixed.detach()),
                      mixture='mean_of_two_weighted_L1_batch_means',
                      training_forward_budget={'student': 2, 'teacher': 2, 'vocoder': 3, 'D_branches': 32})
    return mixed, (first_target, second_target)


@torch.no_grad()
def _clean_from_observed_epsilon(diffusion, arguments, epsilon):
    # 仅从已经发生的模型forward结果还原记录值；运算次序与原reverse_mean_to_zero相同。
    alpha = _extract(diffusion.alphas_cumprod, arguments[1], arguments[0])
    return (arguments[0] - torch.sqrt((1.0 - alpha).clamp_min(0.0)) * epsilon) / torch.sqrt(alpha.clamp_min(1e-12))


def train_batch_mc1(batch, teacher, student, vocoder, discriminator, diffusion,
                    student_optimizer, discriminator_optimizer, mel_mean, mel_std, device, config,
                    *, global_step, mc_receipt):
    validate_request(global_step, config, device, mc_receipt)
    source_noise, score_noise, timesteps = peek_production_draws(batch, device, config)
    record_first_draws(source_noise, score_noise, timesteps, mc_receipt)
    tensors = _slice_batch(batch, device)
    expected_s_t = torch.full((32,), 950, device=device, dtype=torch.long)
    expected_q = diffusion.q_sample(tensors['mel'], expected_s_t, noise=source_noise)
    observed = {'student_count': 0, 'teacher_count': 0}
    handles = []

    @torch.no_grad()
    def student_hook(module, arguments, output):
        observed['student_count'] += 1
        if observed['student_count'] > 2 or not torch.equal(arguments[0], expected_q) or not torch.equal(arguments[1], expected_s_t):
            raise RuntimeError('Actual MC1 student call differs from the original noise preview')
        if not torch.equal(arguments[2], tensors['spk_emb']) or not torch.equal(arguments[3], tensors['bnf']):
            raise RuntimeError('MC1 student conditions changed')
        observed['prediction'] = _clean_from_observed_epsilon(diffusion, arguments, output.detach())

    @torch.no_grad()
    def teacher_hook(module, arguments, output):
        observed['teacher_count'] += 1
        if observed['student_count'] != 2 or observed['teacher_count'] != 1:
            raise RuntimeError('MC1 must perform exactly2S then1T')
        expected_t_q = diffusion.q_sample(observed['prediction'], timesteps, noise=score_noise)
        if not torch.equal(arguments[0], expected_t_q) or not torch.equal(arguments[1], timesteps):
            raise RuntimeError('Actual MC1 teacher input differs from the original noise/time preview')
        if not torch.equal(arguments[2], tensors['spk_emb']) or not torch.equal(arguments[3], tensors['bnf']):
            raise RuntimeError('MC1 teacher conditions changed')
        target = _clean_from_observed_epsilon(diffusion, arguments, output.detach())
        per_l1 = torch.abs(target - observed['prediction']).flatten(1).mean(1)
        weights = diffusion.sqrt_alphas_cumprod.gather(0, timesteps).to(per_l1.dtype)
        observed['first_loss'] = (weights * per_l1).mean()
        observed['first'] = score_record(observed['prediction'], target, timesteps, diffusion, observed['first_loss'])

    try:
        handles.append(student.register_forward_hook(student_hook))
        handles.append(teacher.register_forward_hook(teacher_hook))
        # 控制组确实执行原函数；没有复制/替换/猴子补丁生产train_batch。
        losses = production.train_batch(batch, teacher, student, vocoder, discriminator, diffusion,
                                        student_optimizer, discriminator_optimizer, mel_mean, mel_std, device, config)
        if observed['student_count'] != 2 or observed['teacher_count'] != 1 or losses['dist_raw'] != float(observed['first_loss']):
            raise RuntimeError('MC1 observation disagrees with actual production training')
        mc_receipt.update(mc_samples=1, global_step=global_step, private_seed=None,
                          score_noise2=None, t2=None, t2_values=None, first=observed['first'], second=None,
                          dist1=losses['dist_raw'], dist2=None, dist_mixed=losses['dist_raw'],
                          mixture='original_production_single_weighted_L1',
                          training_forward_budget={'student': 2, 'teacher': 1, 'vocoder': 3, 'D_branches': 32})
        return losses
    finally:
        for handle in reversed(handles):
            handle.remove()  # 异常同样清掉仅用于记录的hooks，不留在下一批或checkpoint中。


def train_batch_mc2(
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
    *, global_step, mc_receipt,
# 除两个明确hook及score调用适配外，函数体与冻结生产train_batch AST完全相同。
) -> Dict[str, float]:
    validate_request(global_step, config, device, mc_receipt)
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

    record_first_draws(source_noise, score_noise, score_timesteps, mc_receipt)

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
    distillation_loss, _ = mc2_score_distillation_loss(
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
        global_step=global_step, mc_receipt=mc_receipt,
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


# 遍历一整轮 1230 个完整 batch，累计训练 loss 平均值并更新全局步数。
