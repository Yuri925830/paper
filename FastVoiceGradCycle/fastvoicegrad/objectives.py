# 定义学生一步还原、显式教师目标空间与权重、波形对抗和特征匹配公式。
"""Differentiable FastVoiceGrad ACDD equations and explicit target semantics.

The student predicts clean x0 using the sparse transition 950 -> 0. A clean
teacher reconstruction uses the analogous t -> 0 transition. The adjacent
t -> t-1 target is retained only for reproducibility of the original run.

FastVoiceGrad Equation (13) references Equation (7) without resolving its
sparse-step interpretation. ADD Section 3.3 explicitly reconstructs clean x0.
The weighting notation differs between those papers. The same authors'
FasterVoiceGrad (Interspeech 2025), Section 2.2, Equation (6), explicitly uses
sqrt(alpha_bar_t) when reviewing FastVoiceGrad. Target and weight choices are
recorded so that the historical run and the corrected interpretation differ.
"""

# math.isfinite 用来拒绝 NaN 或无穷大的 Cycle 权重，避免无效总损失进入反向传播。
import math  # 这里只使用标准库的有限数检查，不改变任何张量计算。
# 导入 List、Sequence 和 Tuple，只用于描述函数输入、输出的数据类型，方便读代码和检查。
from typing import List, Sequence, Tuple

# 导入 PyTorch；Tensor、随机扩散计算和各种逐元素数学运算都由它完成。
import torch
# 导入神经网络模块命名空间；nn.Module 表示教师、学生或判别器这样的网络。
import torch.nn as nn

# 复用教师 VoiceGrad 已经审计过的扩散日程和加噪方法，避免另写一套造成公式偏差。
from voicegrad.diffusion import VoiceGradDiffusion


# 一个判别器由多个子判别器组成，所以“分数”是若干 Tensor 的有序序列。
ScoreList = Sequence[torch.Tensor]
# 每个子判别器又有多层中间特征，因此这里是“子判别器列表 × 层列表 × Tensor”。
FeatureList = Sequence[Sequence[torch.Tensor]]


# 从包含干净端点 0 的 T+1 项扩散系数表中，为每个样本取出其时间步对应的系数。
def _extract(values: torch.Tensor, timesteps: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # timesteps 必须是一维 [B]；B 是 batch 中样本数，必须与参照张量 like 的第 0 维一致。
    if timesteps.ndim != 1 or timesteps.shape[0] != like.shape[0]:
        # 形状不符时立即停止，防止广播机制悄悄把错误系数应用到样本上。
        raise ValueError("timesteps must be [B] and match the input batch")
    # gather 的索引必须是整数；调用者传来其他整数类型时统一转换为 long。
    if timesteps.dtype != torch.long:
        # long 即 64 位整数索引；这里只改变索引表示，不改变时间步数值。
        timesteps = timesteps.long()
    # values 通常形如 [T]，按每个样本的时间步取得 B 个标量，得到 [B]。
    selected = values.gather(0, timesteps)
    # 把 [B] 改成 [B, 1, 1, ...]，使系数能安全广播到 like 的通道和时间维。
    return selected.reshape(timesteps.shape[0], *((1,) * (like.ndim - 1)))


# 统一检查 mel、时间步、说话人向量和 BNF 条件能否按 batch 与时间正确对齐。
def _validate_condition_shapes(
    # x 是 mel 张量，约定形状为 [B, 80, T]。
    x: torch.Tensor,
    # timesteps 为每个样本各自选中的扩散时间步，形状为 [B]。
    timesteps: torch.Tensor,
    # speaker_embedding 是说话人条件向量，形状为 [B, C_speaker]。
    speaker_embedding: torch.Tensor,
    # bnf 是语音内容条件，形状为 [B, C_bnf, T]，最后一维必须与 mel 对齐。
    bnf: torch.Tensor,
# 结束 _validate_condition_shapes 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # mel 必须同时具有 batch、80 个频率通道和时间三个维度。
    if x.ndim != 3:
        # 少一维或多一维都会让后续卷积误解数据含义，所以直接报错。
        raise ValueError("mel input must have shape [B, 80, T]")
    # BNF 也必须为三维，并且样本数、时间帧数都要与 mel 完全一致。
    if bnf.ndim != 3 or bnf.shape[0] != x.shape[0] or bnf.shape[-1] != x.shape[-1]:
        # 这里不自动裁剪或补齐，确保训练不会在未知错位的数据上继续进行。
        raise ValueError("BNF must be [B, C, T] and time-aligned with mel")
    # 每个 mel 样本需要恰好一个一维说话人向量，所以总体应为二维且 batch 一致。
    if speaker_embedding.ndim != 2 or speaker_embedding.shape[0] != x.shape[0]:
        # 说话人条件错配会让模型学到错误身份对应关系，因此立即报错。
        raise ValueError("speaker embedding must have shape [B, C]")
    # 每个样本需要一个扩散步编号，所以 timesteps 必须是长度为 B 的一维向量。
    if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
        # 不允许依赖隐式广播，避免整个 batch 意外共用或错用时间步。
        raise ValueError("timesteps must have shape [B]")


# 实现论文公式（7）：让冻结教师执行普通的相邻反向过程 t -> t-1。
def reverse_mean_adjacent(
    # epsilon_model 预测 x_t 中所含噪声；此处正式训练时传入冻结教师。
    epsilon_model: nn.Module,
    # diffusion 保存 alpha、累计 alpha_bar 以及扩散步数等固定日程。
    diffusion: VoiceGradDiffusion,
    # x_t 是处于扩散时间步 t 的带噪 mel，形状 [B, 80, T_mel]。
    x_t: torch.Tensor,
    # timesteps 给出 batch 中每个样本各自的 t，形状 [B]。
    timesteps: torch.Tensor,
    # speaker_embedding 告诉模型目标说话人身份。
    speaker_embedding: torch.Tensor,
    # bnf 提供与 mel 帧对齐的语音内容信息。
    bnf: torch.Tensor,
# 结束 reverse_mean_adjacent 的参数声明；返回类型为 torch.Tensor，具体内容由下方实现给出。
) -> torch.Tensor:
    # 普通相邻反向公式只把状态从 t 移到 t-1；历史目标仍保留此路径。
    """Equation (7) for the ordinary adjacent transition t -> t-1."""

    # 先验证所有条件张量的 batch 和时间轴，避免公式在错误形状上照样广播运行。
    _validate_condition_shapes(x_t, timesteps, speaker_embedding, bnf)
    # 再由扩散对象检查 t 是否都位于合法范围 1 到 T 内；索引 0 只保留给干净端点。
    diffusion._validate_timesteps(timesteps)
    # 网络根据带噪 mel、时间步、说话人和内容条件预测噪声 epsilon_theta。
    predicted_noise = epsilon_model(x_t, timesteps, speaker_embedding, bnf)
    # 取单步 alpha_t；它描述从 t-1 到 t 这一个扩散步保留多少原信号。
    alpha_t = _extract(diffusion.alphas, timesteps, x_t)
    # 取累计 alpha_bar_t；它描述从第 0 步累计到 t 后还保留多少原信号。
    alpha_bar_t = _extract(diffusion.alphas_cumprod, timesteps, x_t)
    # 按公式（7）返回条件均值；这里没有随机 z 项，因为本函数只计算均值目标。
    return (
        # 从当前带噪状态 x_t 开始。
        x_t
        # 减去根据单步噪声增量和累计噪声规模校正后的预测噪声。
        - (1.0 - alpha_t)
        # 分母开方将预测噪声按当前累计噪声标准差进行归一化。
        / torch.sqrt((1.0 - alpha_bar_t).clamp_min(1e-12))
        # predicted_noise 是教师估计应从 x_t 中去掉的噪声方向。
        * predicted_noise
    # 最后除以 sqrt(alpha_t)，把尺度从时间步 t 还原到相邻的 t-1。
    ) / torch.sqrt(alpha_t.clamp_min(1e-12))


# 按累计系数直接估计干净梅尔；学生从 S_K 预测，clean_x0 教师目标从抽到的 t 预测。
def reverse_mean_to_zero(
    # epsilon_model 可以是待训练学生或冻结教师；两者都先输出噪声预测。
    epsilon_model: nn.Module,
    # diffusion 提供与教师完全相同的固定扩散系数表。
    diffusion: VoiceGradDiffusion,
    # 学生路径输入真实梅尔加噪后的状态；教师目标路径输入学生梅尔重新加噪后的状态。
    x_start_step: torch.Tensor,
    # 学生路径通常全为 950；教师目标路径使用批内各样本独立抽取的 t。
    start_timesteps: torch.Tensor,
    # speaker_embedding 是目标说话人条件。
    speaker_embedding: torch.Tensor,
    # bnf 是与 mel 时间帧对齐的内容条件。
    bnf: torch.Tensor,
# 结束 reverse_mean_to_zero 的参数声明；返回类型为 torch.Tensor，具体内容由下方实现给出。
) -> torch.Tensor:
    # 单步学生直接从起始噪声级预测干净状态，使用累计信号系数。
    """Sparse K=1 reverse mean from S_K directly to zero.

    VoiceGrad's accelerated subsequence redefines alpha for the sole selected
    step as alpha_bar[S_K].  The posterior variance of the final transition is
    zero, so no additional z term is added.
    """

    # 验证输入 mel、时间步和两个条件张量的形状关系。
    _validate_condition_shapes(
        # 第一个被检查的是本次反推时刻的带噪梅尔。
        x_start_step,
        # 第二个是每个样本的起始时间步。
        start_timesteps,
        # 第三个是说话人条件。
        speaker_embedding,
        # 第四个是 BNF 内容条件。
        bnf,
    )
    # 检查当前反推时刻在 1 到 T 内，适用于学生起点和教师目标两条路径。
    diffusion._validate_timesteps(start_timesteps)
    # 当前传入的学生或教师网络根据带噪梅尔和时间步预测其中的噪声。
    predicted_noise = epsilon_model(
        # 输入的带噪 mel。
        x_start_step,
        # 告诉模型当前位于哪个扩散步。
        start_timesteps,
        # 提供目标说话人身份。
        speaker_embedding,
        # 提供语音内容。
        bnf,
    )
    # 直接还原干净状态时，有效 alpha 是当前时刻的累计 alpha_bar，不能使用相邻步 alpha。
    effective_alpha = _extract(
        # alphas_cumprod 正是累计 alpha_bar 表。
        diffusion.alphas_cumprod,
        # 按每个样本本次反推的时间步索引取值。
        start_timesteps,
        # 以 x_start_step 的维数为参照整理广播形状。
        x_start_step,
    )
    # 用确定性公式估计干净梅尔；其角色由调用方决定，可以是学生输出或教师监督目标。
    return (
        # 从当前指定时刻的带噪梅尔开始。
        x_start_step
        # 减去“累计噪声标准差 × 学生预测噪声”。
        - torch.sqrt((1.0 - effective_alpha).clamp_min(0.0)) * predicted_noise
    # 再除以累计信号标准差，恢复到第 0 步的数值尺度。
    ) / torch.sqrt(effective_alpha.clamp_min(1e-12))


# 先按论文公式（9）制造学生输入，再让学生一步得到最终 mel 预测。
def make_student_prediction(
    # student 是由教师权重初始化、当前需要训练的 FastVoiceGrad 网络。
    student: nn.Module,
    # diffusion 用于把干净 mel 正向加噪到指定起始步。
    diffusion: VoiceGradDiffusion,
    # x_zero 是真实、标准化后的干净 mel，形状 [B, 80, T_mel]。
    x_zero: torch.Tensor,
    # speaker_embedding 指定目标说话人。
    speaker_embedding: torch.Tensor,
    # bnf 提供同一句话的内容条件。
    bnf: torch.Tensor,
    # start_step 是论文设置的 S_K，本实验锁定为 950。
    start_step: int,
    # source_noise 是预先抽样的标准高斯噪声，形状必须与 x_zero 一致。
    source_noise: torch.Tensor,
# 结束 make_student_prediction 的参数声明；返回类型为 Tuple[torch.Tensor, torch.Tensor]，具体内容由下方实现给出。
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 按正向公式给源梅尔加噪，再执行一次学生的干净梅尔预测。
    """Equation (9), followed by the paper's one-step student prediction."""

    # 加噪必须逐元素对应；形状不一致意味着样本、通道或时间帧发生了错位。
    if source_noise.shape != x_zero.shape:
        # 不允许自动广播噪声，以保证公式（9）严格逐元素执行。
        raise ValueError("source_noise and x_zero must have identical shapes")
    # S_K 不能为 0，也不能超出总扩散步数 T；本实验的 950 位于合法区间。
    if not 1 <= start_step < diffusion.n_levels:
        # 非法起始步无法表示从带噪状态到 0 的稀疏反向过程。
        raise ValueError("FastVoiceGrad requires 1 <= S_K < T")
    # 为 batch 中每个样本建立相同的起始时间步向量 [B]。
    timesteps = torch.full(
        # 向量长度等于 batch 大小 B。
        (x_zero.shape[0],),
        # 每个元素都写入论文指定的整数 S_K。
        int(start_step),
        # 直接在 x_zero 所在设备创建，避免 CPU/GPU 混用错误。
        device=x_zero.device,
        # 扩散系数查表要求 long 型整数索引。
        dtype=torch.long,
    )
    # 用固定公式 q(x_S_K | x_0) 把真实 mel 加噪到第 S_K 步。
    x_start_step = diffusion.q_sample(x_zero, timesteps, noise=source_noise)
    # 学生从 x_S_K 一步反推，得到对干净 mel 的预测 x_theta。
    x_theta = reverse_mean_to_zero(
        # 使用当前学生网络预测噪声。
        student,
        # 使用统一的扩散日程。
        diffusion,
        # 输入刚生成的 x_S_K。
        x_start_step,
        # 输入每个样本的 S_K。
        timesteps,
        # 输入目标说话人条件。
        speaker_embedding,
        # 输入语音内容条件。
        bnf,
    )
    # 同时返回最终预测和学生收到的带噪输入，后者可用于检查或诊断。
    return x_theta, x_start_step


# 用同一个单步学生完成 A→B→A，并把第一程的梯度边界集中锁在这一处。
def make_cycle_round_trip_prediction(  # 集中实现单步学生的 A→B→A 往返生成和梯度边界。
    student: nn.Module,  # student 是当前需要加入 Cycle Loss 的 FastVoiceGrad 单步网络。
    diffusion: VoiceGradDiffusion,  # diffusion 提供两程各自从干净端加噪到 S_K 的固定公式。
    source_mel: torch.Tensor,  # source_mel 是说话人 A 的标准化 mel，形状为 [B,80,T]。
    source_speaker: torch.Tensor,  # source_speaker 是说话人 A 的 256 维身份条件。
    target_speaker: torch.Tensor,  # target_speaker 是与 A 不同的说话人 B 的身份条件。
    source_bnf: torch.Tensor,  # source_bnf 是 A 的离线内容特征，本版在两程中如实复用。
    start_step: int,  # start_step 是 FastVoiceGrad 的单步起点，本实验固定为 950。
    forward_noise: torch.Tensor,  # forward_noise 只供 A→B 第一程加噪使用。
    backward_noise: torch.Tensor,  # backward_noise 只供 B→A 第二程重新加噪使用。
    *,  # 星号要求调用者显式写出梯度边界，防止位置参数误传。
    first_leg_stop_gradient: bool,  # True 对齐公开 CycleDiffusion 训练代码的第一程 no_grad。
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回生成的 mel_AB 与往返回来的 mel_ABA。
    # 该函数实现 mel 域条件式往返路径，但不会假装当前工程已经具备在线 BNF 重编码。
    """构造 FastVoiceGrad 的 A→B→A 路径，并明确限制第一程和第二程的梯度。"""  # 中文说明函数职责。

    # 两程噪声都必须与源 mel 逐元素同形，禁止依赖广播或复用形状不明的噪声。
    if forward_noise.shape != source_mel.shape or backward_noise.shape != source_mel.shape:  # 核对两个 [B,80,T] 噪声张量。
        raise ValueError("Both cycle noises must have the same shape as source_mel")  # 形状不符时立即停止。
    # 源与目标说话人向量必须完全同形，才能保证每个 A 都有一个对应的 B 条件。
    if source_speaker.shape != target_speaker.shape:  # 比较 batch 大小与说话人向量宽度。
        raise ValueError("Source and target speaker embeddings must have identical shapes")  # 拒绝错误广播。
    # 第一程默认不保留计算图，既符合公开实现，也避免同时保存两程图造成显存过高。
    if first_leg_stop_gradient:  # 只有明确配置为 True 时进入官方式 stop-gradient 分支。
        with torch.no_grad():  # no_grad 使 A→B 只制造伪平行 mel，不更新第一程图。
            converted_mel, _ = make_student_prediction(  # 从 A 的 mel 和内容生成 B 音色的 mel_AB。
                student,  # 第一程与第二程共享同一个 FastVoiceGrad 学生。
                diffusion,  # 两程使用同一套 T=1000 余弦日程。
                source_mel,  # 第一程从真实 A mel 加噪后开始。
                target_speaker,  # 第一程改用目标说话人 B 的身份向量。
                source_bnf,  # 内容仍使用 A 的 BNF，要求语言内容保持不变。
                start_step,  # FastVoiceGrad 每程只执行 S_K→0 的一个反向步。
                forward_noise,  # 第一程专属的独立标准高斯噪声。
            )  # 完成无梯度的 A→B 生成调用。
        converted_mel = converted_mel.detach()  # 再次显式切断，防止以后内部实现变化破坏边界。
    else:  # 该分支只供单独消融第一程是否反传，默认正式配置不会进入。
        converted_mel, _ = make_student_prediction(  # 建立可微的 A→B 第一程图。
            student,  # 使用当前学生参数。
            diffusion,  # 使用相同扩散定义。
            source_mel,  # 从源 A mel 开始。
            target_speaker,  # 条件切换到目标说话人 B。
            source_bnf,  # 两程仍复用 A 的内容条件。
            start_step,  # 使用固定单步起点。
            forward_noise,  # 使用第一程独立噪声。
        )  # 完成可微 A→B 消融路径。
    # 第二程必须把实际生成的 mel_AB 当作新的“干净端”重新加噪，不能再次使用原 mel_A。
    cycled_mel, _ = make_student_prediction(  # 从 mel_AB 生成返回源说话人 A 的 mel_ABA。
        student,  # 第二程梯度更新同一个学生网络。
        diffusion,  # 使用与第一程完全相同的扩散日程。
        converted_mel,  # 真正把第一程生成结果作为第二程输入。
        source_speaker,  # 第二程把说话人条件切回源说话人 A。
        source_bnf,  # 当前没有 Mel→BNF 编码器，因此明确复用 A 的 BNF。
        start_step,  # 第二程同样只走一个 FastVoiceGrad 反向步。
        backward_noise,  # 第二程使用与第一程独立抽取的噪声。
    )  # 完成可微的 B→A 往返重建。
    return converted_mel, cycled_mel  # 同时返回中间转换与最终往返结果供损失和诊断使用。


# 只比较每条语音真实存在的帧，避免短语音右侧补零影响 Cycle Loss。
def cycle_consistency_loss(  # 计算忽略补零帧的逐样本 Cycle 一致性损失。
    original_mel: torch.Tensor,  # original_mel 是原始 A mel，形状为 [B,80,T]。
    cycled_mel: torch.Tensor,  # cycled_mel 是 A→B→A 后的 mel，形状必须完全相同。
    valid_lengths: torch.Tensor,  # valid_lengths 保存每条语音补零前的真实帧数，形状为 [B]。
) -> torch.Tensor:  # 返回逐样本 masked L1 再按公开代码额外除 mel 维数的标量。
    # 该归约让长短语音各占相同样本权重，并保留 VQ-CycleDiffusion 的额外 1/n_mels 尺度。
    """先计算逐样本有效帧 L1，再按公开代码额外除以 mel 通道数。"""  # 中文说明归约与复现尺度。

    # 两个 mel 必须完全同形且为三维，禁止广播导致错误位置被比较。
    if original_mel.ndim != 3 or original_mel.shape != cycled_mel.shape:  # 核对 [B,C,T] 结构。
        raise ValueError("original_mel and cycled_mel must be identical [B,C,T] tensors")  # 形状不合即停止。
    # 本项目声学表示固定为 80 个 mel 通道，其他通道数说明拿错了特征。
    if original_mel.shape[1] != 80:  # 检查第二维是否为固定的 mel 频带数。
        raise ValueError("Cycle consistency expects exactly 80 mel channels")  # 拒绝 BNF 或波形误传。
    # 每条样本必须提供一个 long 型有效长度，浮点长度可能掩盖裁剪错误。
    if valid_lengths.ndim != 1 or valid_lengths.shape[0] != original_mel.shape[0]:  # 核对 [B]。
        raise ValueError("valid_lengths must have shape [B]")  # 数量错位时停止。
    if valid_lengths.dtype != torch.long:  # 扩散与掩码契约统一要求 64 位整数长度。
        raise TypeError("valid_lengths must use torch.long")  # 不做静默类型转换。
    # 长度至少为 1 且不能超过统一时间轴 T，否则掩码会表示不存在的帧。
    if bool(torch.any(valid_lengths < 1)) or bool(torch.any(valid_lengths > original_mel.shape[-1])):  # 检查合法范围。
        raise ValueError("valid_lengths must lie within 1..T")  # 拒绝空语音和越界长度。
    # 在 mel 所在设备建立时间索引 [1,1,T]，随后与每条长度广播比较。
    frame_index = torch.arange(original_mel.shape[-1], device=original_mel.device).view(1, 1, -1)  # 时间位置从 0 到 T-1。
    # 小于真实长度的位置为 1，其余补零位置为 0；再转换成 mel 的浮点类型。
    frame_mask = (frame_index < valid_lengths.to(original_mel.device).view(-1, 1, 1)).to(original_mel.dtype)  # 得到 [B,1,T] 掩码。
    # 对 80 个通道和有效时间帧累加绝对误差，补零区域乘 0 后既无数值贡献也无梯度。
    absolute_error_sum = (torch.abs(cycled_mel - original_mel) * frame_mask).sum(dim=(1, 2))  # 得到每条样本误差和 [B]。
    # 每条样本分母是 80×真实帧数，使不同长度语音先各自得到可比较的平均 L1。
    valid_element_count = valid_lengths.to(device=original_mel.device, dtype=original_mel.dtype) * original_mel.shape[1]  # 计算 [B] 分母。
    # 先对样本平均而不是把所有有效元素混在一起，避免长语音获得更大权重。
    masked_per_example_mae = torch.mean(absolute_error_sum / valid_element_count)  # 得到标准 masked MAE 标量。
    # 公开 VQ-CycleDiffusion 先用默认 mean 的 F.l1_loss，再额外除 n_mels；这里保留其 1/80 数值尺度。
    return masked_per_example_mae / original_mel.shape[1]  # 返回与公开解码器训练代码同尺度的 Cycle 标量。


# 计算分数蒸馏损失；教师对重新加噪的学生输出调用一次，目标空间由显式模式决定。
def score_distillation_loss(
    # student_prediction 是学生一步生成的干净 mel 估计 x_theta。
    student_prediction: torch.Tensor,
    # teacher 是冻结的 VoiceGrad-500 教师，只提供监督目标，不更新权重。
    teacher: nn.Module,
    # diffusion 提供再加噪与教师重建所需系数；clean_x0 直接还原干净状态，旧模式只退到 t-1。
    diffusion: VoiceGradDiffusion,
    # timesteps 是蒸馏时为每个样本抽到的随机扩散步 t。
    timesteps: torch.Tensor,
    # score_noise 是这次蒸馏加噪使用的高斯噪声，与 source_noise 相互独立。
    score_noise: torch.Tensor,
    # speaker_embedding 是目标说话人条件。
    speaker_embedding: torch.Tensor,
    # bnf 是语音内容条件。
    bnf: torch.Tensor,
    # 后面两种模式必须写关键字传参，避免相似字符串位置交换。
    *,
    # 调用方必须写明目标空间，漏填立即报错，避免旧脚本悄悄选择另一套公式。
    target_mode: str,
    # 权重同样必须显式写明；正式入口采用sqrt_alpha_bar，旧对照需要自行声明。
    weight_mode: str,
# 结束 score_distillation_loss 的参数声明；返回类型为 Tuple[torch.Tensor, torch.Tensor]，具体内容由下方实现给出。
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 计算蒸馏项，并显式区分干净重建或历史相邻目标及三种权重解释。
    """Equation (13) with an explicit reconstruction and weighting convention.

    The returned teacher target is detached and is provided for diagnostics.
    Gradients reach the student only through ``student_prediction`` in the L1
    difference, exactly matching sg(x_theta,t) in the paper.

    ``clean_x0`` reconstructs clean mel using cumulative alpha_bar_t and is the
    clean-target interpretation supported by ADD Section 3.3. Both mode keywords
    are required: ``legacy_adjacent`` is only an explicitly requested replay of
    the old run's t -> t-1 target. Omitting a mode never selects a historical
    objective or silently changes the meaning of an existing experiment.

    ``nominal_alpha`` uses 1-beta_t; ``alpha_bar`` uses cumulative signal power;
    ``sqrt_alpha_bar`` uses signal amplitude, as made explicit in the same
    authors' FasterVoiceGrad (2025) Section 2.2, Equation (6).
    """

    # 只接受已实现且有明确含义的两种教师目标空间。
    if target_mode not in {"legacy_adjacent", "clean_x0"}:
        # 目标名称拼错时立即报错，防止悄悄训练另一种目标。
        raise ValueError(f"Unknown score target mode: {target_mode!r}")
    # 检查权重名称是否属于单步保留量、累计功率或累计幅度。
    if weight_mode not in {"nominal_alpha", "alpha_bar", "sqrt_alpha_bar"}:
        # 未知权重不能自动猜测，直接提示配置错误。
        raise ValueError(f"Unknown score weight mode: {weight_mode!r}")
    # 蒸馏噪声必须与学生 mel 逐元素对应，不能依靠广播补齐。
    if score_noise.shape != student_prediction.shape:
        # 形状不同会改变 q(x_t | x_0) 的含义，因此立即报错。
        raise ValueError("score_noise and student_prediction must match")
    # 教师目标属于 stop-gradient 分支；no_grad 可防止保存其反向图并节省显存。
    with torch.no_grad():
        # 先把“已切断梯度”的学生结果加噪到随机时间步 t，得到 x_theta,t。
        x_theta_t = diffusion.q_sample(
            # detach 对应论文中的 sg：教师分支不能把梯度传回学生。
            student_prediction.detach(),
            # 每个样本可以使用不同的随机 t。
            timesteps,
            # 明确传入独立抽样的蒸馏噪声，保证过程可审计、可复现。
            noise=score_noise,
        )
        # 新实验显式指定 clean_x0；旧实验可重放原有相邻步目标。
        teacher_reconstruction = (
            # clean_x0使用t→0重建，其余已验证模式明确重放旧t→t−1均值。
            reverse_mean_to_zero if target_mode == "clean_x0" else reverse_mean_adjacent
        )
        # 按选定目标空间生成教师监督，处于no_grad中不会建立反向图。
        teacher_target = teacher_reconstruction(
            # 使用冻结教师预测噪声。
            teacher,
            # 使用相同扩散日程。
            diffusion,
            # 输入刚加噪得到的 x_theta,t。
            x_theta_t,
            # 输入对应的随机时间步 t。
            timesteps,
            # 输入目标说话人条件。
            speaker_embedding,
            # 输入内容条件。
            bnf,
        )
    # 对每个样本计算教师目标与学生预测间的逐元素 L1 距离，再平均通道和时间维。
    per_example_l1 = torch.abs(teacher_target - student_prediction).flatten(1).mean(1)
    # 论文的 alpha 记号存在解释差异；运行记录须保存所选权重的明确含义。
    weight_schedule = {
        # 旧模式取单个相邻扩散步的1−βt。
        "nominal_alpha": diffusion.alphas,
        # 对照模式取从0累计到t的信号功率。
        "alpha_bar": diffusion.alphas_cumprod,
        # 修复主模式取累计信号幅度，对应同作者2025年回顾FastVoiceGrad的Eq.6。
        "sqrt_alpha_bar": diffusion.sqrt_alphas_cumprod,
    # 用明确模式名选出本次实验的系数表。
    }[weight_mode]
    # 逐样本按t取权重，并与L1损失使用相同精度。
    weights = weight_schedule.gather(0, timesteps.long()).to(per_example_l1.dtype)
    # 逐样本加权后平均；同时返回已停止梯度的教师目标供诊断。
    return torch.mean(weights * per_example_l1), teacher_target


# 计算判别器的最小二乘 GAN 损失，对应论文公式（10）。
def least_squares_discriminator_loss(
    # real_scores 是各子判别器对真实波形输出的分数列表。
    real_scores: ScoreList,
    # fake_scores 是同一批子判别器对生成波形输出的分数列表。
    fake_scores: ScoreList,
# 结束 least_squares_discriminator_loss 的参数声明；返回类型为 torch.Tensor，具体内容由下方实现给出。
) -> torch.Tensor:
    # 按所有 MPD/MRD 子判别器相加得到真实与生成波形的判别器目标。
    """Equation (10), summed across the modified HiFi-GAN sub-discriminators."""

    # 两组结果必须非空且一一对应，否则无法按同一子判别器配对计算损失。
    if len(real_scores) != len(fake_scores) or not real_scores:
        # 拒绝静默截断；Python 的 zip 默认会忽略较长列表尾部，所以必须先检查。
        raise ValueError("real/fake discriminator score lists must be non-empty and aligned")
    # 将所有 MPD/MRD 子判别器的真实项和生成项直接相加，保持论文的“求和”定义。
    return sum(
        # 判别器对真实分数使用目标 1，对生成分数使用目标 0，分别计算均方误差。
        torch.mean((real - 1.0) ** 2) + torch.mean(fake**2)
        # 同时遍历一一对应的真实分数与生成分数。
        for real, fake in zip(real_scores, fake_scores)
    )


# 计算学生在波形判别器上的最小二乘对抗损失，对应论文公式（11）。
def least_squares_student_loss(fake_scores: ScoreList) -> torch.Tensor:
    # 对全部子判别器相加得到学生的对抗损失。
    """Equation (11), summed across the modified HiFi-GAN discriminators."""

    # 没有任何判别器输出时，返回 0 会掩盖结构错误，因此必须报错。
    if not fake_scores:
        # 明确告诉调用者生成分数列表为空。
        raise ValueError("fake discriminator score list must not be empty")
    # 学生希望每个判别器都把生成波形判为真实，所以所有 fake 分数的目标都是 1。
    return sum(torch.mean((fake - 1.0) ** 2) for fake in fake_scores)


# 比较真实/生成波形在判别器每一层的中间特征，对应论文公式（12）。
def feature_matching_loss(
    # real_features 按“子判别器 → 层”保存真实波形的特征图。
    real_features: FeatureList,
    # fake_features 以完全相同的结构保存生成波形的特征图。
    fake_features: FeatureList,
# 结束 feature_matching_loss 的参数声明；返回类型为 torch.Tensor，具体内容由下方实现给出。
) -> torch.Tensor:
    # 计算未加权的各层特征匹配 L1 之和，外部再乘固定系数 2。
    """Raw Equation (12), before applying lambda_FM=2.

    Each layer uses ``mean(abs(real-fake))``, which is exactly
    ``1/N_l * ||...||_1``.  Layers and sub-discriminators are summed, never
    averaged.  Unlike the upstream HiFi-GAN helper, this function does not
    contain an additional factor of two.
    """

    # 真实与生成两侧必须含相同数量且至少一个子判别器。
    if len(real_features) != len(fake_features) or not real_features:
        # 若数量不一致，zip 会漏算尾部；提前报错可以避免损失悄悄变小。
        raise ValueError("real/fake feature lists must be non-empty and aligned")
    # 建立列表，逐层保存尚未加权的平均绝对误差项。
    terms: List[torch.Tensor] = []
    # 成对遍历同一个子判别器在真实波形和生成波形上的各层特征。
    for real_layers, fake_layers in zip(real_features, fake_features):
        # 同一个子判别器的真实/生成前向必须返回相同层数。
        if len(real_layers) != len(fake_layers):
            # 层数不同无法逐层比较，说明网络输出契约被破坏。
            raise ValueError("real/fake discriminator layers are not aligned")
        # 将当前子判别器的所有逐层 L1 项追加到总列表。
        terms.extend(
            # detach 真实特征，使学生更新时只沿生成波形一侧传播梯度。
            torch.mean(torch.abs(real.detach() - fake))
            # 按层一一配对；mean 等价于论文的 1/N_l 乘 L1 范数。
            for real, fake in zip(real_layers, fake_layers)
        )
    # 理论上每个子判别器都应返回特征图；空列表意味着实现或调用发生异常。
    if not terms:
        # 不把空求和当成 0，以免学生在缺失特征匹配监督时继续训练。
        raise ValueError("discriminator returned no feature maps")
    # 对所有层、所有子判别器直接求和；这里不乘 2，权重由总损失函数统一应用。
    return sum(terms)


# 把三类学生损失按论文公式（14）组合成最终可反向传播的标量。
# 判别器也在更新，训练批次与噪声也会变化，因此该 GAN 总损失不保证逐步或逐轮下降。
def acdd_student_loss(
    # adversarial 是公式（11）的波形对抗损失。
    adversarial: torch.Tensor,
    # feature_matching 是公式（12）尚未乘权重的特征匹配损失。
    feature_matching: torch.Tensor,
    # score_distillation 是公式（13）的分数蒸馏损失。
    score_distillation: torch.Tensor,
    # 论文规定特征匹配权重 lambda_FM=2；增大时更强调波形中间特征相似。
    lambda_fm: float = 2.0,
    # 论文规定蒸馏权重 lambda_dist=45；增大时更强调接近教师目标。
    lambda_dist: float = 45.0,
# 结束 acdd_student_loss 的参数声明；返回类型为 torch.Tensor，具体内容由下方实现给出。
) -> torch.Tensor:
    # 学生总目标只含对抗项、2 倍特征匹配和 45 倍蒸馏项。
    """Equation (14), with no unreported auxiliary loss."""

    # 最终损失严格为 L_adv + 2*L_FM + 45*L_dist，没有额外辅助损失。
    return adversarial + lambda_fm * feature_matching + lambda_dist * score_distillation


# 在原 FastVoiceGrad 公式（14）外只增加一个显式加权的 Cycle Loss，保留基线函数原义。
def cycle_augmented_student_loss(  # 在原 ACDD 公式外只加入一次加权 Cycle 项。
    adversarial: torch.Tensor,  # adversarial 是原公式（11）的学生波形对抗损失。
    feature_matching: torch.Tensor,  # feature_matching 是原公式（12）的未加权特征匹配项。
    score_distillation: torch.Tensor,  # score_distillation 是原公式（13）的分数蒸馏项。
    cycle_consistency: torch.Tensor,  # cycle_consistency 是 masked mel L1 的 A→B→A 往返项。
    *,  # 所有权重必须写出名称，避免把相近的浮点数按错顺序。
    lambda_fm: float,  # lambda_fm 保持 FastVoiceGrad 论文规定的 2。
    lambda_dist: float,  # lambda_dist 保持 FastVoiceGrad 论文规定的 45。
    lambda_cycle: float,  # lambda_cycle 是新增实验权重，公开实现采用 1。
) -> torch.Tensor:  # 返回 L_ACDD + lambda_cycle×L_cycle 的可反向标量。
    # Cycle 权重必须是有限非负数；NaN、无穷大或负值都会让总损失失去数学意义。
    if not math.isfinite(lambda_cycle) or lambda_cycle < 0.0:  # 在建立总损失前完整验证新增超参数。
        raise ValueError("lambda_cycle must be finite and non-negative")  # 非法权重直接拒绝。
    # 先调用原函数得到未经改写的 FastVoiceGrad Eq.(14)，避免其语义被可选参数污染。
    baseline_loss = acdd_student_loss(  # 计算 L_adv + lambda_fm L_FM + lambda_dist L_dist。
        adversarial,  # 传入原始对抗项。
        feature_matching,  # 传入原始特征匹配项。
        score_distillation,  # 传入原始蒸馏项。
        lambda_fm=lambda_fm,  # 显式保留 FM 权重。
        lambda_dist=lambda_dist,  # 显式保留蒸馏权重。
    )  # 完成基线总目标计算。
    # 新实验只在基线目标后相加一次 Cycle 项，不把它混入判别器或教师分支。
    return baseline_loss + lambda_cycle * cycle_consistency  # 返回四项学生总目标。
