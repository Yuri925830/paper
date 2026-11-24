# 独立的教师目标生成器：不修改 VoiceGrad 正式采样器，也不训练或改写教师权重。
"""Deterministic DDIM targets from an explicitly shared, already noisy state.

This is an experimental target sampler, not the formal VoiceGrad-30 sampler.
Equation (12), DDIM (https://arxiv.org/html/2010.02502v3), with eta=0 gives
    clean = (x_t - sqrt(1-A_t) * epsilon_teacher) / sqrt(A_t)
    x_s = sqrt(A_s) * clean + sqrt(1-A_s) * epsilon_teacher.
A denotes cumulative alpha_bar, not the one-step alpha used by VoiceGrad.
At the final endpoint s=0, A_0=1 and the result is exactly clean.

The caller must form x_start = q(source_mel, start_step, shared_noise) once
and pass that same state to the student and this sampler. This module does
not add source noise, draw reverse noise, denormalize mel, or clip outputs.
Generating deterministic endpoints is motivated by Progressive Distillation
Section 3 (https://arxiv.org/html/2202.00512v2); direct 30-to-1 endpoint fitting
is a new experiment, not that paper's complete progressive-halving algorithm.
"""

# Integral 接受普通整数及 NumPy 整数，拒绝把小数时间步默默截断。
from numbers import Integral
# Optional 表示可以使用默认时间网格，Sequence 表示调用者可显式给出完整下降网格。
from typing import Optional, Sequence

# 只使用张量运算和 no_grad；本模块不创建优化器或随机生成器。
import torch
# 用类型提示说明教师必须是已有的神经网络对象。
from torch import nn

# 为日志提供与正式 ancestral VoiceGrad-30 不同的采样器身份。
DDIM_SAMPLER_ID = "ddim_eta0_shared_noisy_start_v1"


# 整条教师轨迹都停止求导；即使调用者在学生训练图内调用，也不反传进教师。
@torch.no_grad()
def sample_ddim_teacher(
    # 强制使用参数名，避免把已加噪状态与干净源 mel 混淆。
    *,
    # 接收已经加载并设为 eval 的教师；函数不会改变其训练模式或参数标志。
    teacher: nn.Module,
    # 使用既有 VoiceGradDiffusion 对象的累计日程和原始时间网格。
    diffusion,
    # 已处于 start_step 的标准化 mel [B,80,T]，必须与学生输入逐元素相同。
    x_start: torch.Tensor,
    # [B,C_speaker] 目标说话人条件；重建时用原说话人，转换时用目标说话人。
    speaker_embedding: torch.Tensor,
    # [B,C_bnf,T] 源内容条件，不能为了采样重新裁剪或变更对齐方式。
    bnf: torch.Tensor,
    # 显式网格必须严格下降、首项等于 start_step、不包含终点 0。
    timesteps: Optional[Sequence[int]] = None,
    # 默认走 30 次教师前向；若指定网格，其长度也必须等于这个值。
    num_steps: int = 30,
    # 既有学生从 950 开始；调用者换起点时必须同时改变其共享输入状态。
    start_step: int = 950,
) -> torch.Tensor:
    # 返回独立、无梯度的干净标准化 mel，不对教师质量作任何保证。
    """Return an eta=0 clean endpoint; x_start is already noisy, never clean."""
    # 教师的任何子模块处于训练模式都拒绝，防止 dropout 或 BN 改变目标和状态。
    if any(module.training for module in teacher.modules()):
        # 由调用者显式设置 eval，避免本函数暗中改动外部模型。
        raise ValueError("Teacher and all its submodules must be in eval mode")
    # 时间起点必须是整数，布尔值不能冒充 0 或 1。
    if isinstance(start_step, bool) or not isinstance(start_step, Integral):
        # 小数起点会造成系数表索引歧义，因此在教师前向之前停止。
        raise ValueError("start_step must be an integer")
    # 保持项目约定：起点大于 0 且低于训练总步数。
    if not 1 <= start_step < diffusion.n_levels:
        # 不接受干净输入的零步路径或超出训练日程的状态。
        raise ValueError("start_step must be in [1, diffusion.n_levels-1]")
    # 同样要求步数是有效整数，不允许四舍五入后改变教师计算次数。
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
        # num_steps 控制真实教师前向次数，不能以小数或布尔量指定。
        raise ValueError("num_steps must be an integer")
    # 正整数网格最多包含 start_step 个不重复的噪声时刻。
    if not 1 <= num_steps <= start_step:
        # 拒绝空轨迹以及不可能建立的时间网格。
        raise ValueError("num_steps must be in [1, start_step]")
    # 默认复用正式实现选出的相同 30 个时刻，只改变时刻之间的反向公式。
    if timesteps is None:
        # 原函数返回升序，这里反转为从 950 向 0 的去噪顺序。
        steps = list(reversed(diffusion.sampling_sequence(num_steps, start_step)))
    # 若实验显式提供网格，则完整保留它，不能偷偷重新采样或排序。
    else:
        # 转为独立列表，避免循环期间依赖调用者可变的序列对象。
        steps = list(timesteps)
    # 网格长度必须与声明步数相符，让“30 步”的标签可被严格核对。
    if len(steps) != num_steps:
        # 即使网格本身合法，也不接受标签与真实教师调用次数不一致。
        raise ValueError("timesteps length must equal num_steps")
    # 每个时刻都必须是整数，不自动将浮点时刻转为邻近整数。
    if any(isinstance(step, bool) or not isinstance(step, Integral) for step in steps):
        # 防止两个浮点值被截断到相同步而仍被当作不同去噪步。
        raise ValueError("Every timestep must be an integer")
    # 统一为 Python 整数，便于后面创建各样本的 long 类型时间条件。
    steps = [int(step) for step in steps]
    # 共享状态的噪声级必须与第一次教师前向的时间标签一致。
    if steps[0] != start_step:
        # 不能拿 x950 配上 t900，否则教师条件与输入分布不一致。
        raise ValueError("The first timestep must equal start_step")
    # 终点 0 由函数自动追加；显式网格只包含模型实际接受的正时间步。
    if any(step < 1 or step > start_step for step in steps):
        # 这也排除超过起点以及负数的时间标签。
        raise ValueError("Explicit timesteps must be positive and at most start_step")
    # 每次去噪都必须向更小时间步移动，不允许重复或反向跳回去。
    if any(left <= right for left, right in zip(steps, steps[1:])):
        # 不排序纠正调用错误，以便错误网格被及时发现。
        raise ValueError("timesteps must be strictly descending")
    # 本实验使用未补长的 [batch,mel频道,时间] 输入，不隐式广播批次。
    if x_start.ndim != 3 or any(size < 1 for size in x_start.shape):
        # 空批次或空帧无法构成合法语音采样任务。
        raise ValueError("x_start must have nonempty shape [B,C_mel,T]")
    # 先采用教师原始 float32 或诊断 float64，拒绝低精度下未经验证的除法放大。
    if x_start.dtype not in (torch.float32, torch.float64):
        # 不自动转精度，避免评估时悄悄偏离约定环境。
        raise ValueError("DDIM target generation requires float32 or float64")
    # 内容条件的批次与时间轴必须与共享输入完全相同。
    if bnf.ndim != 3 or bnf.shape[0] != x_start.shape[0] or bnf.shape[-1] != x_start.shape[-1]:
        # 这里不做插值、下采样或截断，BNF 处理保持原实验路径。
        raise ValueError("BNF must match the input batch and frame count")
    # 每条语音只接受一个说话人向量。
    if speaker_embedding.ndim != 2 or speaker_embedding.shape[0] != x_start.shape[0]:
        # 说话人条件的错配应在教师计算前被拒绝。
        raise ValueError("speaker_embedding must have shape [B,C_speaker]")
    # 条件与输入使用同一设备，避免循环中隐式搬运或落到另一张 GPU。
    if bnf.device != x_start.device or speaker_embedding.device != x_start.device:
        # 由调用者统一设备，不改变其原始条件张量。
        raise ValueError("Input and conditions must be on the same device")
    # 累计 alpha_bar 同时决定信号与噪声幅度，不能用单步 alpha 替代。
    cumulative = diffusion.alphas_cumprod.detach().to(device=x_start.device, dtype=x_start.dtype)
    # 保证索引 0 到 T 都存在；缺失的干净端点不能靠负索引补救。
    if cumulative.ndim != 1 or cumulative.numel() != diffusion.n_levels + 1:
        # 清楚报告日程格式错误，避免读取错误系数。
        raise ValueError("alphas_cumprod must contain the clean endpoint and all T steps")
    # 干净端点按定义严格为 1，否则最后的“x0”标注不成立。
    if float(cumulative[0]) != 1.0:
        # 不能静默覆盖系数表，因为这会掩盖另一套归一化或索引约定。
        raise ValueError("alpha_bar[0] must equal 1 exactly")
    # 只检查本次访问的日程，顺序为下降时间网格加干净端点。
    selected = cumulative[steps + [0]]
    # 每个信号功率必须有限、为正且不超过 1，开方与除法才有扩散意义。
    if not bool(torch.isfinite(selected).all()) or bool(((selected <= 0) | (selected > 1)).any()):
        # 不使用截断来掩盖错误日程，否则生成结果不再符合 Eq.12。
        raise ValueError("Selected cumulative alphas must be finite and in (0,1]")
    # 时间下降时累计信号应严格增大；退化的重复噪声级没有独立采样意义。
    if not bool((selected[1:] > selected[:-1]).all()):
        # 检查数值顺序，进一步防止传入单步 alpha 表冒充累计量。
        raise ValueError("Cumulative signal must increase along the reverse trajectory")
    # 已有 NaN 或无穷会污染全部目标，在任何教师前向前直接拒绝。
    if not bool(torch.isfinite(x_start).all()):
        # 输入错误不能被当作 DDIM 采样质量问题。
        raise ValueError("x_start contains nonfinite values")
    # 保留与学生输入完全相同的数值，同时切断图并避免写回调用者的张量存储。
    x = x_start.detach().clone()
    # 按选定网格依次调用同一个冻结教师，过程中不抽取任何新噪声。
    for index, current_step in enumerate(steps):
        # 每条样本本轮使用同一时间条件，张量形状为 [B]。
        step_tensor = torch.full((x.shape[0],), current_step, device=x.device, dtype=torch.long)
        # 教师输出当前状态中的噪声预测，仍使用原说话人与 BNF 条件。
        predicted_noise = teacher(x, step_tensor, speaker_embedding, bnf)
        # 噪声必须逐元素对应 mel；不接受能广播但语义不同的输出。
        if predicted_noise.shape != x.shape or predicted_noise.dtype != x.dtype or predicted_noise.device != x.device:
            # 也防止未声明的混合精度悄悄进入对照实验。
            raise ValueError("Teacher noise must match x_start shape, dtype, and device")
        # 当前噪声级的累计信号功率用来从 epsilon 参数化还原干净预测。
        alpha_t = cumulative[current_step]
        # 这是 DDIM Eq.12 第一项里的 x0 估计，不是相邻 t-1 的 DDPM 均值。
        clean = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        # 最后一次直接返回干净端点，避免在 t=0 再调用一次教师或加入噪声。
        if index + 1 == len(steps):
            # 终点必须有限；失败时不能把损坏目标保存为后续监督数据。
            if not bool(torch.isfinite(clean).all()):
                # 教师异常或数值溢出都应使目标生成明确失败。
                raise FloatingPointError("Teacher DDIM endpoint contains nonfinite values")
            # 返回由本次运算创建、已停止梯度的干净标准化 mel。
            return clean
        # 非末步仍保留下一噪声级所需的噪声方向，不能直接把 clean 当作 x_s。
        next_step = steps[index + 1]
        # s 是下一较小的选中时间步，而不是固定的 t-1。
        alpha_s = cumulative[next_step]
        # 完整 eta=0 DDIM 跳步公式；仅删 DDPM 的随机项得不到这个均值。
        x = torch.sqrt(alpha_s) * clean + torch.sqrt(1.0 - alpha_s) * predicted_noise
    # 合法非空网格必然在末轮 return；这里仅防未来维护意外打破控制流。
    raise RuntimeError("DDIM trajectory did not reach its clean endpoint")
