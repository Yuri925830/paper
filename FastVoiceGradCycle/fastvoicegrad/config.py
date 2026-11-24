# VCTK FastVoiceGrad 学生模型实验的只读配置。
"""VCTK FastVoiceGrad 学生模型实验的只读配置。

这个文件把两类设定严格分开：论文白纸黑字写出的设定由
``paper_explicit`` 返回；程序必须知道、但论文没有公布的细节由
``declared_reproduction_choices`` 返回。这样生成实验记录时，不会把我们的
复现选择误写成论文作者的原始设定。
"""

# ``asdict`` 把数据类转换成普通字典；``dataclass`` 用来声明结构化配置。
from dataclasses import asdict, dataclass
# 这些类型只服务于类型标注，帮助读代码的人看懂函数会接收或返回什么。
from typing import Dict, Mapping, Union


# 单个配置值只允许是字符串、整数、浮点数或布尔值，便于写入 JSON 运行契约。
Scalar = Union[str, int, float, bool]


# ``frozen=True`` 表示创建后不允许修改字段，防止训练中途悄悄改变实验设定。
@dataclass(frozen=True)
# 集中保存学生模型的公开参数和本项目补充选择，frozen保证运行中不被随意改写。
class FastVoiceGradStudentConfig:
    # 集中保存论文第 3 节、第 4.1 节设定以及明确登记的复现选择。
    """集中保存论文第 3 节、第 4.1 节设定以及明确登记的复现选择。"""

    # 以下是论文第 4.1 节明确给出的声学表示和 VCTK 划分协议。
    # 采样率 22050 Hz：每秒保留 22050 个波形点；更高能保留更多高频细节但更耗显存和算力。
    sample_rate: int = 22050
    # 梅尔频谱有 80 个频率通道；维数更大细节更多、输入更宽，维数更小则信息更粗略。
    n_mels: int = 80
    # FFT 长度 1024：每次用 1024 个采样点分析频率；更大频率分辨率高但时间定位更粗。
    n_fft: int = 1024
    # 相邻频谱帧移动 256 个采样点；更小会产生更密的帧并增加计算量，更大则时间细节减少。
    hop_size: int = 256
    # 分析窗长度 1024：每帧观察 1024 个采样点；增大更利于分辨频率，减小更利于定位瞬时变化。
    win_size: int = 1024
    # 训练集使用 100 位说话人；更多说话人通常提升泛化，但训练数据和耗时也会增加。
    train_speakers: int = 100
    # 严格 text-unseen 测试集使用 10 位说话人，用来覆盖论文规定的测试范围。
    test_speakers: int = 10
    # 每位测试说话人取 10 句，因此完整测试集一共是 10×10=100 句。
    test_sentences_per_speaker: int = 10

    # 学生模型沿用任意说话人到任意说话人的 VoiceGrad 噪声预测网络结构。
    # 教师扩散过程共有 1000 个噪声等级；更多等级过程更细但教师采样更慢。
    diffusion_steps: int = 1000
    # 余弦噪声日程的 0.008 偏移量论文未公开；它避免曲线端点过于极端，改动会改变各步噪声强度。
    cosine_offset: float = 0.008
    # 学生只走 1 个反向步；步数更少推理更快，但学习教师映射的难度更高。
    student_steps: int = 1
    # 单步学生从扩散步 950 开始；数值越大起点噪声越重，学生需要一次去除更多噪声。
    student_start_step: int = 950
    # 隐藏通道数为 512；更大通常提高容量，也会明显增加参数量、显存和计算时间。
    hidden_channels: int = 512
    # 主干包含 12 层一维卷积；层数增加可扩大建模能力，但训练和推理都会变慢。
    convolution_layers: int = 12
    # 使用 2 级下采样和上采样；级数影响感受野、时间分辨率及内存占用。
    down_up_stages: int = 2

    # 以下三个值对应论文公式（14），决定三个学生损失项如何组合。
    # 特征匹配损失乘 2；变大更强调波形判别器的中间特征，变小则对它约束更弱。
    lambda_fm: float = 2.0
    # 分数蒸馏损失乘 45；增大时更强调所选教师重建目标，减小时对抗与特征匹配的相对占比提高。
    lambda_dist: float = 45.0
    # 论文用 alpha_t 记号，但单步重定义的适用范围需要说明；具体实现单独登记。
    paper_score_weight_notation: str = "alpha_t; effective schedule interpretation disclosed separately"
    # 新训练默认让教师从随机噪声级t还原干净mel，与学生输出处在同一个空间。
    # 依据是原论文引用的ADD第3.3节干净重建公式；2024正文没有给出唯一可核验的实现。
    # 历史相邻步目标仍可显式回放，不能再由漏填参数触发。
    score_target: str = "clean_x0"
    # 累计信号幅度随噪声增大而减小，对应同作者2025论文第2.2节公式(6)。
    # 这里仅补充FastVoiceGrad蒸馏解释，不引入FasterVoiceGrad的新损失或改教师。
    score_weight: str = "sqrt_alpha_bar"

    # Cycle Loss 默认关闭，只有新实验显式传入 --enable-cycle-loss 才会改变目标函数。
    cycle_enabled: bool = False  # 默认关闭可保证原 FastVoiceGrad 的前向次数和随机数轨迹不变。
    # VQ-CycleDiffusion 从第 71 轮开始加入 Cycle Loss；第 1～70 轮保持原始重建训练。
    cycle_start_epoch: int = 71  # 这一轮次门控来自公开实现，便于先稳定基础生成能力。
    # Cycle Loss 的外层系数设为 1，与公开 CycleDiffusion 解码器训练设置一致。
    lambda_cycle: float = 1.0  # 该值属于新增实验选择，不是 FastVoiceGrad 原论文参数。
    # 每个 batch32 只取 3 条语音计算往返路径，以复现公开实现的显存控制办法。
    cycle_examples_per_batch: int = 3  # 数量太大会显著增加两次学生前向的显存占用。
    # 第一程 A→B 不保存反向图，第二程 B→A 才通过 L1 更新学生，符合公开训练代码。
    cycle_first_leg_stop_gradient: bool = True  # 该边界避免同时保留两程计算图导致显存过高。
    # 当前工程没有可微的 Mel→BNF 编码器，所以两程都如实复用源语音 BNF。
    cycle_content_mode: str = "reuse_source_bnf"  # 不能把此模式写成已经重新提取生成语音内容。
    # 训练时从同一 batch 中随机选择另一位说话人的 embedding，可重复使用同一个目标样本。
    cycle_target_sampling: str = "random_different_speaker_with_replacement"  # 每个 A 都强制满足 speaker_B != speaker_A。
    # Cycle 比较只统计真实帧，逐样本和 batch 平均后再按公开代码额外除以 80 个 mel 通道。
    cycle_loss_reduction: str = "masked_per_example_l1_then_batch_mean_divide_n_mels"  # 同时屏蔽补零并复现公开 1/80 尺度。
    # 两程分别抽取标准高斯噪声，禁止把 A→B 的噪声直接复用于 B→A。
    cycle_noise_policy: str = "independent_standard_normal_per_leg"  # 独立噪声避免人为耦合两个扩散输入。
    # 训练目标说话人配对使用独立随机流，避免改变原 ACDD 的噪声与时间步序列。
    cycle_train_pair_seed: int = 80_001_234  # 每个 global_step 在此基础上派生唯一配对种子。
    # A→B 训练噪声使用第二条独立随机流，不消费原训练全局 CUDA 随机状态。
    cycle_train_forward_noise_seed: int = 90_001_234  # 第一程每批按 global_step 派生可恢复种子。
    # B→A 训练噪声使用第三条独立随机流，并且不与第一程共享生成器状态。
    cycle_train_backward_noise_seed: int = 100_001_234  # 第二程每批按 global_step 派生可恢复种子。
    # FastVoiceGrad 学生每程仍只执行一个反向步，不照搬 CycleDiffusion 的六步采样。
    cycle_reverse_steps: int = 1  # 该值必须与本项目 K=1 的学生结构保持一致。
    # 固定测试从完整测试清单选择下一位不同说话人，避免末尾同说话人小批退化成 identity cycle。
    cycle_test_target_policy: str = "deterministic_next_different_speaker"  # 测试配对跨轮固定且不消耗训练随机数。

    # 以下是论文第 4.1 节公开的 VCTK 学生模型优化设置。
    # 每次更新同时处理 32 条语音；更大梯度通常更稳但占显存更多，更小则波动更大。
    batch_size: int = 32
    # 学习率为 0.0002，控制一次更新走多远；过大可能震荡，过小则收敛很慢。
    learning_rate: float = 2e-4
    # Adam 的一阶动量系数为 0.5；更大更依赖较久以前的梯度，更小更快响应当前梯度。
    adam_beta1: float = 0.5
    # Adam 的二阶动量系数为 0.9；更大时平方梯度估计更平滑，但适应变化会更慢。
    adam_beta2: float = 0.9
    # 完整遍历训练集 100 次；更多轮可能继续拟合，也会增加耗时并带来过拟合风险。
    epochs: int = 100

    # 下面是论文实验要求固定、不随学生训练更新的依赖模型。
    # 教师使用第 500 个 epoch 的权重；换成别的轮次会改变蒸馏目标，不能再直接对照本实验。
    teacher_epoch: int = 500
    # 声码器以 HiFi-GAN V1 为主体，保留 MPD，并用 UnivNet-MRD 替换原来的 MSD。
    vocoder_architecture: str = "HiFi-GAN-V1 + MPD + UnivNet-MRD (MSD replaced)"

    # 以下都是软件运行必需、但论文没有公开的复现选择；它们与上面的论文值故意分开登记。
    # 每个训练片段取 128 帧；更长能看到更多上下文但显存开销更高，更短则相反。
    train_frames: int = 128
    # 随机种子固定为 1234，使初始化、裁剪、噪声和批次顺序尽量可以复现。
    seed: int = 1234
    # BNF 与梅尔长度不同时按旧工程的长度线性规则对齐，保证继续使用既有数据流程。
    bnf_alignment: str = "legacy_length_linear"
    # BNF 时间轴按旧工程的抽点规则降采样；换规则会改变每个梅尔帧对应的语音内容。
    bnf_downsampling: str = "legacy_decimate"
    # 运行前必须确认协议是 text-unseen，避免误用 seen 文本的测试划分。
    expected_protocol: str = "text-unseen"
    # 500 epoch 教师检查点应对应 615000 次参数更新，用于发现拿错权重的情况。
    expected_teacher_global_step: int = 615_000
    # 教师文件的 SHA-256 指纹逐字节锁定正确权重，任何内容变化都会导致校验失败。
    expected_teacher_checkpoint_sha256: str = (
        # 这一整串就是服务器上已验证的教师权重指纹，不能凭文件名替代。
        "1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6"
    )
    # 固定使用训练到 250 万步的自训声码器；步数不一致时预检会直接拒绝开始训练。
    expected_vocoder_step: int = 2_500_000
    # 判别器从随机参数开始训练；若从预训练权重开始，早期损失和最终结果都会改变。
    discriminator_initialization: str = "scratch"
    # 每更新学生一次就更新判别器一次；比例更高会让判别器更强，也可能压制学生学习。
    discriminator_updates_per_student_update: int = 1
    # 全程保持学习率不变；若使用衰减日程，后期更新会更小，已属于另一套实验设定。
    learning_rate_schedule: str = "constant"
    # 权重衰减为 0，表示不额外惩罚大参数；增大可加强正则化但可能造成欠拟合。
    weight_decay: float = 0.0
    # Adam 分母稳定项为 1e-8；过大可能压小更新，过小在极小方差时数值稳定性较弱。
    adam_epsilon: float = 1e-8
    # 不启用 AMSGrad 变体，确保使用普通 Adam；开启后会改变二阶动量的更新规则。
    adam_amsgrad: bool = False
    # 不做梯度裁剪；若设置阈值会限制大梯度，也会改变论文目标下的真实更新量。
    gradient_clipping: str = "none"
    # 使用 float32 单精度；精度更低会省显存但带来舍入差异，float64 则更慢更占内存。
    numerical_precision: str = "float32"
    # 数据加载使用 4 个工作进程；更多可能加快供数但占用更多 CPU/内存，过多也可能拥堵。
    num_workers: int = 4
    # 每轮训练都打乱样本，避免模型长期看到固定顺序；关闭后更确定但可能产生顺序偏差。
    train_shuffle: bool = True
    # 丢弃不足 32 条的最后一批，确保论文规定的每次更新批量始终等于 32。
    train_drop_last: bool = True
    # 长语音每轮从所有合法整数起点中均匀随机裁 128 帧，增加训练片段多样性。
    train_crop_sampling: str = "uniform_integer_start_per_utterance_per_epoch"
    # 不足 128 帧的梅尔和对齐 BNF 在右侧补规范化空间中的 0，保证输入维度一致。
    short_segment_padding: str = "right_zero_in_normalized_mel_and_aligned_bnf"
    # 每 10 个 epoch 额外保存一个永久检查点；间隔更小更安全，但会占更多磁盘。
    save_every_epochs: int = 10
    # 每 10 个训练 step 输出一次进度；更频繁更实时，但终端和磁盘日志开销略高。
    log_every_steps: int = 10
    # 连续训练批次按窗口记录趋势；只报警，不用测试集损失改变训练。
    loss_monitor_window_steps: int = 50
    # 论文没有规定开发集损失协议；以下固定设置只为用户要求的观察曲线服务。
    # 每个 epoch 都算一次测试损失；间隔更大开销更小，但曲线时间分辨率会更低。
    test_loss_every_epochs: int = 1
    # 测试也统一取 128 帧，既与训练输入形状一致，也控制显存；更长会增加测试耗时。
    test_loss_frames: int = 128
    # 测试批量为 32，因此 100 条测试语音会组成 32、32、32、4 四批且不丢样本。
    test_loss_batch_size: int = 32
    # 固定测试裁剪种子，使同一语音每轮取到同一片段，曲线变化主要反映模型变化。
    test_loss_crop_seed: int = 10_001_234
    # 固定测试 DataLoader 种子，避免加载次序随机变化影响对比。
    test_loss_loader_seed: int = 20_001_234
    # 固定生成学生输入噪声的种子，使各 epoch 接受完全可比的起始噪声。
    test_loss_source_noise_seed: int = 30_001_234
    # 固定分数蒸馏目标噪声的另一种子，并与输入噪声保持独立。
    test_loss_score_noise_seed: int = 40_001_234
    # 固定随机扩散时刻的种子，使每条测试语音跨 epoch 使用相同时刻分布。
    test_loss_timestep_seed: int = 50_001_234
    # 固定测试 A→B 第一程噪声，确保各轮 Cycle 指标比较的是模型变化。
    test_loss_cycle_forward_noise_seed: int = 60_001_234  # 与已有三条测试随机流使用不同种子。
    # 固定测试 B→A 第二程噪声，并与第一程保持独立。
    test_loss_cycle_backward_noise_seed: int = 70_001_234  # 不复用 A→B 的噪声张量或随机序列。
    # 测试损失绝不参与反向传播或参数更新，避免测试集信息泄漏进模型训练。
    test_loss_use_for_training: bool = False
    # 禁止按测试损失挑检查点，只报告固定的第 100 epoch，维持事先确定的实验终点。
    test_loss_checkpoint_selection: str = "disabled_epoch_100_is_reported"
    # 先按每条样本加权，再覆盖全部 100 条 unseen 语音，避免末尾 4 条的小批权重被放大。
    test_loss_reduction: str = "sample_weighted_over_all_100_unseen_utterances"
    # 蒸馏时刻从 1 到 T 的离散整数中均匀采样；偏向某区间会改变模型重点学习的噪声级。
    score_t_sampling: str = "uniform_discrete_1_to_T"
    # 学生起始噪声与教师分数目标噪声各自独立采标准正态分布，避免错误耦合。
    source_and_score_noise: str = "independent_standard_normal"
    # 各层或各样本先求平均再相加，防止张量尺寸直接放大单个损失项。
    loss_reduction: str = "per_layer_or_example_mean_then_sum"
    # 判别器总损失是 MPD 与 MRD 所有分支损失之和；分支变多会相应增大未归一化总量。
    discriminator_loss_aggregation: str = "sum_over_MPD_and_MRD_branches"
    # 判别器使用独立 Adam，但学习率和 beta 与学生相同；共享优化器会混合两套状态。
    discriminator_optimizer_interpretation: str = "separate_Adam_with_student_hyperparameters"

    # 把冻结配置转换成普通字典，供检查点和运行契约记录。
    def as_dict(self) -> Dict[str, Scalar]:
        # 把当前全部只读字段转换成普通字典，方便序列化和分组筛选。
        """把当前全部只读字段转换成普通字典，方便序列化和分组筛选。"""

        # ``asdict`` 会复制数据类字段，不会把外部修改反写进这个冻结配置对象。
        return asdict(self)

    # 只挑出论文明确公开的字段，避免把本项目选择写成论文原设定。
    def paper_explicit(self) -> Mapping[str, Scalar]:
        # 只返回 FastVoiceGrad 论文明确写出的字段，供运行契约单独展示。
        """只返回 FastVoiceGrad 论文明确写出的字段，供运行契约单独展示。"""

        # 先取得完整配置字典，随后仅按下面白名单取值，避免误把复现选择混进来。
        values = self.as_dict()
        # 元组中的字段名就是“论文明确设定”的唯一白名单，增删都必须有论文依据。
        keys = (
            "sample_rate",  # 论文给出的音频采样率。
            "n_mels",  # 论文给出的梅尔通道数。
            "n_fft",  # 论文给出的 FFT 点数。
            "hop_size",  # 论文给出的相邻帧步长。
            "win_size",  # 论文给出的分析窗长度。
            "train_speakers",  # 论文给出的训练说话人数。
            "test_speakers",  # 论文给出的测试说话人数。
            "test_sentences_per_speaker",  # 论文给出的每位测试者句数。
            "diffusion_steps",  # 论文给出的教师扩散总步数。
            "student_steps",  # 论文给出的学生反向步数 K=1。
            "student_start_step",  # 论文给出的学生起始步 S_K=950。
            "hidden_channels",  # 论文给出的隐藏通道数。
            "convolution_layers",  # 论文给出的卷积层数。
            "down_up_stages",  # 论文给出的下采样/上采样级数。
            "lambda_fm",  # 论文公式（14）的特征匹配权重。
            "lambda_dist",  # 论文公式（14）的蒸馏权重。
            "paper_score_weight_notation",  # 保留原文记号，不把实现解释冒充明确设定。
            "batch_size",  # 论文给出的批量大小。
            "learning_rate",  # 论文给出的学习率。
            "adam_beta1",  # 论文给出的 Adam beta1。
            "adam_beta2",  # 论文给出的 Adam beta2。
            "epochs",  # 论文给出的学生训练轮数。
            "teacher_epoch",  # 论文实验采用的教师轮次。
            "vocoder_architecture",  # 论文描述的修改版声码器结构。
        )
        # 按白名单原顺序构造只读语义的映射，不会包含任何未公开字段。
        return {key: values[key] for key in keys}

    # 返回所有需要额外披露的实现选择，供报告解释其理由和影响。
    def declared_reproduction_choices(self) -> Mapping[str, Scalar]:
        # 返回论文没有写明、由本复现预先固定并公开登记的实现细节。
        """返回论文没有写明、由本复现预先固定并公开登记的实现细节。"""

        # 完整配置仍从同一个冻结对象取得，防止两套列表引用不同来源的数值。
        values = self.as_dict()
        # 下面白名单明确告诉读者：这些值是复现选择，不能声称由论文作者提供。
        keys = (
            "score_target",  # teacher 的相邻反向均值或一步干净 mel 预测。
            "score_weight",  # nominal_alpha、alpha_bar 或后续作者论文的 sqrt_alpha_bar。
            "cycle_enabled",  # 是否显式开启新增的 Cycle Loss 实验。
            "cycle_start_epoch",  # 从哪一轮开始真正建立往返路径。
            "lambda_cycle",  # Cycle Loss 在学生总目标中的外层权重。
            "cycle_examples_per_batch",  # 每个 batch 用于 Cycle 的样本数。
            "cycle_first_leg_stop_gradient",  # A→B 第一程是否停止梯度。
            "cycle_content_mode",  # 第二程采用何种内容条件，本版复用源 BNF。
            "cycle_target_sampling",  # 如何保证目标说话人与源说话人不同。
            "cycle_loss_reduction",  # 如何屏蔽补零帧并汇总 L1。
            "cycle_noise_policy",  # 两程扩散噪声之间的关系。
            "cycle_train_pair_seed",  # 训练目标说话人配对的独立随机流起始种子。
            "cycle_train_forward_noise_seed",  # 训练 A→B 噪声的独立随机流起始种子。
            "cycle_train_backward_noise_seed",  # 训练 B→A 噪声的独立随机流起始种子。
            "cycle_reverse_steps",  # 每一程实际执行的 FastVoiceGrad 反向步数。
            "cycle_test_target_policy",  # 固定测试跨 batch 选择目标说话人的规则。
            "train_frames",  # 训练裁剪长度。
            "cosine_offset",  # 余弦日程偏移。
            "seed",  # 总体随机种子。
            "bnf_alignment",  # BNF 长度对齐方法。
            "bnf_downsampling",  # BNF 降采样方法。
            "expected_protocol",  # 强制的 text-unseen 协议。
            "expected_teacher_global_step",  # 教师更新步数身份。
            "expected_teacher_checkpoint_sha256",  # 教师文件指纹。
            "expected_vocoder_step",  # 自训声码器更新步数。
            "discriminator_initialization",  # 判别器初始化来源。
            "discriminator_updates_per_student_update",  # 两种更新的比例。
            "learning_rate_schedule",  # 学习率是否变化。
            "weight_decay",  # Adam 权重衰减。
            "adam_epsilon",  # Adam 数值稳定项。
            "adam_amsgrad",  # 是否采用 AMSGrad。
            "gradient_clipping",  # 是否限制梯度大小。
            "numerical_precision",  # 训练浮点精度。
            "num_workers",  # 数据加载进程数。
            "train_shuffle",  # 每轮是否打乱训练样本。
            "train_drop_last",  # 是否丢弃不完整训练批。
            "train_crop_sampling",  # 随机裁剪规则。
            "short_segment_padding",  # 短语音补齐规则。
            "save_every_epochs",  # 永久检查点间隔。
            "log_every_steps",  # 终端日志间隔。
            "loss_monitor_window_steps",  # 训练趋势报警窗口。
            "test_loss_every_epochs",  # 测试损失间隔。
            "test_loss_frames",  # 测试固定片段长度。
            "test_loss_batch_size",  # 测试批量大小。
            "test_loss_crop_seed",  # 测试裁剪种子。
            "test_loss_loader_seed",  # 测试加载器种子。
            "test_loss_source_noise_seed",  # 学生输入噪声种子。
            "test_loss_score_noise_seed",  # 蒸馏目标噪声种子。
            "test_loss_timestep_seed",  # 扩散时刻种子。
            "test_loss_cycle_forward_noise_seed",  # 固定测试 Cycle 第一程噪声种子。
            "test_loss_cycle_backward_noise_seed",  # 固定测试 Cycle 第二程噪声种子。
            "test_loss_use_for_training",  # 是否让测试损失参与训练。
            "test_loss_checkpoint_selection",  # 是否按测试损失选模型。
            "test_loss_reduction",  # 100 条测试语音的汇总方法。
            "score_t_sampling",  # 蒸馏时刻抽样分布。
            "source_and_score_noise",  # 两类噪声之间的关系。
            "loss_reduction",  # 各损失张量的归约方式。
            "discriminator_loss_aggregation",  # 判别器分支汇总方式。
            "discriminator_optimizer_interpretation",  # 判别器优化器定义。
        )
        # 只按复现白名单返回字段，使报告可以逐项披露“论文未说明，我们如何固定”。
        return {key: values[key] for key in keys}


# 全工程统一导入这一个冻结实例，避免不同模块各自创建并修改实验设定。
PAPER_STUDENT_CONFIG = FastVoiceGradStudentConfig()
