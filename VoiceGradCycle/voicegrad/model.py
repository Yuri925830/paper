# VoiceGrad 网络根据目标身份和源内容估计 mel 中的噪声，支持未见说话人转换。
"""Any-to-any VoiceGrad noise estimator used by the FastVoiceGrad teacher.

The topology follows Figure 3 of the original VoiceGrad paper. In particular,
encoder/decoder skip connections are channel concatenations, not additions.
FastVoiceGrad replaces speaker-ID and one-hot timestep embeddings with an
external speaker embedding and a sinusoidal timestep embedding.
"""

# =============================================================================
# 先用最简单的话理解这个文件
# =============================================================================
# 这个文件定义“猜噪声的神经网络”，不是完整训练程序，也不负责把 mel 变成 wav。
# 网络一次接收四样东西：
#   x           ：某个扩散时刻的 noisy mel，形状 [B, 80, T]；
#   noise_level ：每条语音现在位于第几个扩散步，形状 [B]；
#   spk_emb     ：目标说话人的 256 维身份向量，形状 [B, 256]；
#   bnf         ：源语音的音素/内容特征，形状 [B, 144, T]。
# 最终输出 epsilon_theta，形状仍为 [B,80,T]，含义是“我认为 x 里有哪些噪声”。
#
# B=batch size，C=通道数，T=mel 时间帧数。PyTorch Conv1d 的固定轴顺序是
# [B,C,T]，因此不能把 80 个 mel 通道放到最后一维。
#
# 与论文的对应关系：
# 1. FastVoiceGrad 4.1 节规定 12 层一维卷积、512 隐藏通道、两次上下采样、
#    GLU 和 weight normalization。
# 2. 具体 kernel/stride 和 U-Net 连接来自原 VoiceGrad Figure 3。
# 3. FastVoiceGrad 把 speaker label 改成外部 speaker embedding，把 timestep
#    one-hot 改成 sinusoidal positional embedding。
# 4. 原图中的 ⊕ 是沿通道拼接，所以必须使用 torch.cat，绝对不是逐元素相加。

# --------------------------- 导入包逐项解释 ---------------------------
# math 是 Python 标准数学库，这里只用 log(10000) 计算正弦 timestep 频率。
import math
# Tuple 是类型提示，用来说明函数返回两个值组成的元组，不参与数值计算。
from typing import Tuple

# torch 是 PyTorch 主包，Tensor、随机数和基础张量操作都来自它。
import torch
# torch.nn 提供神经网络层和 Module 基类；把它简称 nn 是 PyTorch 社区惯例。
import torch.nn as nn
# functional 简称 F，提供无状态函数：pad、glu 等。它和 nn.Module 层的区别是
# 本身没有需要保存/学习的参数。
import torch.nn.functional as F


# 旧权重在低分辨率层直接每隔 2 或 4 帧抽取 BNF。
LEGACY_BNF_DOWNSAMPLING = "legacy_decimate"
# 独立对齐消融选项：先按隐藏层帧中心插值，再做通道投影。
CENTERED_BNF_DOWNSAMPLING = "centered_linear"
# 登记两种 BNF 下采样规则，避免拼写错误静默改变模型条件。
BNF_DOWNSAMPLING_MODES = (
    # 允许重放已有教师使用的直接抽点规则。
    LEGACY_BNF_DOWNSAMPLING,
    # 允许新实验显式选择按隐藏层中心插值的规则。
    CENTERED_BNF_DOWNSAMPLING,
)


# def 表示定义函数；前导下划线表示它只供本文件内部使用，不是给命令行直接调用的接口。
def _initialize(module: nn.Module) -> nn.Module:
    # 这个小函数统一初始化 Conv1d、ConvTranspose1d 和 Linear。
    # isinstance(a, (A,B,C)) 的意思是：检查 a 是否属于列出的任一种类型。
    # Xavier normal 让各层输入输出的数值尺度较稳定；gain=0.5 沿用 VoiceGrad 设置。
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        # 只重置传入层的 weight；函数名末尾的下划线表示“就地修改原张量”。
        nn.init.xavier_normal_(module.weight, gain=0.5)
        # 有些层可以关闭 bias，所以先判断偏置参数确实存在，不能对 None 做初始化。
        if module.bias is not None:
            # 把所有偏置清零，让网络初始输出不带人为的固定偏移。
            nn.init.zeros_(module.bias)
    # 把处理后的同一个层对象交还调用者，方便写成 layer = _initialize(layer)。
    return module


# 这个辅助函数把“初始化”和“添加 weight norm”组合为一步，避免每建一层都重复写两遍。
def _weight_norm(module: nn.Module) -> nn.Module:
    # 先初始化普通权重，再包装 weight normalization。weight norm 会把权重的
    # “大小”和“方向”分开参数化，通常能让深层卷积训练更稳定。
    _initialize(module)
    # 返回被 weight_norm 包装后的层；之后优化器会更新包装产生的可训练参数。
    return nn.utils.weight_norm(module)


# class 定义一个可重复使用的神经网络组件；继承 nn.Module 后才能被大模型登记为子模块。
class SinusoidalStepEmbedding(nn.Module):
    # timestep 原本只是一个整数，例如 500。神经网络很难直接从一个裸整数理解
    # “早期/中期/晚期噪声”，所以用很多不同频率的 sin/cos 把它展开成向量。
    # 这就是论文所说的 sinusoidal positional embedding，不需要学习查找表。
    def __init__(self, dim: int):
        # 调用 nn.Module 的构造函数，建立 PyTorch 管理子层、参数和设备所需的内部结构。
        super().__init__()
        # 至少需要一维 sin 和一维 cos；小于 2 无法形成成对的正余弦编码。
        if dim < 2:
            # raise 会立即停止并给出清楚原因，避免稍后才出现难懂的空张量错误。
            raise ValueError("The timestep embedding dimension must be >= 2.")
        # 保存维数，forward 中还要用它计算一半维数和判断奇偶。
        self.dim = dim

    # forward 定义该模块接到数据时怎样计算；调用 embedding(steps) 会自动执行它。
    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        # 输入通常是 [B]，输出是 [B,dim]。float() 是因为 sin/cos 需要浮点数；
        # reshape(-1) 把标量或多余维度统一整理成一维 batch。
        steps = steps.float().reshape(-1)
        # 整数除法 // 丢掉小数部分；一半位置放 sin，另一半位置放 cos。
        half = self.dim // 2
        # 频率在对数尺度上均匀铺开；max(...,1) 防止 half=1 时分母变成 0。
        scale = math.log(10000.0) / max(half - 1, 1)
        # 生成 half 个按指数变化的频率。device/dtype 跟 steps 保持一致，避免出现
        # “一个 Tensor 在 CPU、另一个在 GPU”或 float32/float64 不匹配错误。
        frequencies = torch.exp(
            # 生成按对数间隔递减的编码频率指数，低频编码慢变化、高频编码邻近步差异。
            -scale * torch.arange(half, device=steps.device, dtype=steps.dtype)
        )
        # None 是增加一个长度为 1 的维度：[B,1]×[1,half] 自动广播为 [B,half]。
        phases = steps[:, None] * frequencies[None, :]
        # 一半用 sin、一半用 cos，再沿特征维 dim=1 拼接。
        embedding = torch.cat((phases.sin(), phases.cos()), dim=1)
        # 对整数取模 `% 2`：结果为 1 就说明目标维数是奇数，需要补齐最后一维。
        if self.dim % 2:
            # 如果 dim 是奇数，sin/cos 两半只能提供 dim-1 维，最后补一个 0。
            embedding = F.pad(embedding, (0, 1))
        # 返回最终 [B, dim] 编码；return 之后本次函数调用结束。
        return embedding


# 这是主干反复使用的“带条件 GLU 卷积块”，每个对象对应论文网络图中的一层。
class ConditionalGLUBlock(nn.Module):
    # 一个卷积门控块同时接收扩散阶段、说话人身份与内容 BNF 三类条件。
    """A VoiceGrad GLU layer conditioned on timestep, speaker and BNF."""

    # 每个块的流程是：
    # ① 把 BNF 降维并调整到当前时间分辨率；
    # ② 把 timestep+speaker 条件复制到每个时间帧；
    # ③ x、全局条件、BNF 沿通道拼接；
    # ④ 做普通卷积或反卷积；
    # ⑤ 用 GLU 门控输出。
    # GLU 会把 2*out_channels 分成 A 和 B，计算 A*sigmoid(B)。B 像一个门，
    # 决定候选信息 A 中哪些应该通过。

    # 创建对象并固定后续读取或网络计算要遵循的参数；下面的参数分别限定输入数据和处理约束。
    def __init__(
        # 当前数据集或网络对象。
        self,
        # 卷积块输入的主干通道数。
        in_channels: int,
        # 经过 GLU 门控后输出的通道数。
        out_channels: int,
        # 卷积在时间轴上观察的相邻帧范围。
        kernel_size: int,
        # 卷积每次前移的时间间隔。
        stride: int,
        # 扩散步和说话人条件拼接后的通道数。
        cond_channels: int,
        # 进入投影层的原始 BNF 通道数。
        bnf_dim: int,
        # BNF 投影后送入当前块的通道数。
        bnf_channels: int,
        # 旧规则下 BNF 在时间轴的抽点间隔。
        bnf_stride: int,
        # 是否使用转置卷积恢复时间分辨率。
        transpose: bool = False,
        # 网络各时间分辨率上的 BNF 取样规则。
        bnf_downsampling: str = LEGACY_BNF_DOWNSAMPLING,
    # 本接口负责创建对象并固定后续读取或网络计算要遵循的参数，处理结果通过对象状态或文件保存，不额外返回数值。
    ) -> None:
        # 先初始化父类；`-> None` 表示构造函数只建立对象，不返回一个计算结果。
        super().__init__()
        # 记住本块给 BNF 使用的 stride，形状报错时可准确告诉我们是哪一级分辨率。
        self.bnf_stride = bnf_stride
        # 按网络各时间分辨率上的 BNF 取样规则决定是否进入此分支；判断条件为 bnf_downsampling not in BNF_DOWNSAMPLING_MODES。
        if bnf_downsampling not in BNF_DOWNSAMPLING_MODES:
            # 创建对象并固定后续读取或网络计算要遵循的参数未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(
                # 错误输出未被实现的内部 BNF 采样模式名。
                f"Unknown BNF downsampling {bnf_downsampling!r}; "
                # 列出当前允许选择的内部 BNF 对齐模式。
                f"choose one of {BNF_DOWNSAMPLING_MODES}"
            )
        # 网络各时间分辨率上的 BNF 取样规则。
        self.bnf_downsampling = bnf_downsampling
        # kernel_size=1 的卷积只混合特征通道；stride=1/2/4 负责把原始 BNF 的
        # 时间长度压到当前 U-Net 层的 T、T/2 或 T/4。
        self.bnf_projection = _weight_norm(
            # 创建只负责 BNF 通道投影的 1×1 卷积，其时间取样间隔由下面的规则决定。
            nn.Conv1d(
                # 进入投影层的原始 BNF 通道数。
                bnf_dim,
                # BNF 投影后送入当前块的通道数。
                bnf_channels,
                # 卷积在时间轴上观察的相邻帧范围。
                kernel_size=1,
                # 卷积每次前移的时间间隔。
                stride=(
                    # 旧规则由卷积 stride 直接每隔指定帧数取一个内容向量。
                    bnf_stride
                    # 按网络各时间分辨率上的 BNF 取样规则决定是否进入此分支；判断条件为 bnf_downsampling == LEGACY_BNF_DOWNSAMPLING。
                    if bnf_downsampling == LEGACY_BNF_DOWNSAMPLING
                    # 中心插值模式已在前向中完成时间缩放，投影层只逐帧变换通道。
                    else 1
                ),
            )
        )

        # 后面会 torch.cat 三部分，所以卷积输入通道必须是三者通道数之和。
        total_in = in_channels + cond_channels + bnf_channels
        # transpose=True 表示 decoder 上采样，使用 ConvTranspose1d；否则使用
        # 普通 Conv1d。stride=2 时普通卷积长度减半，反卷积长度加倍。
        if transpose:
            # decoder 的上采样块选择转置卷积；输出通道写 2 倍是为了后面的 GLU 对半切。
            convolution = nn.ConvTranspose1d(
                # 主干、说话人/步条件和 BNF 拼接后的总输入通道。
                total_in,
                # 先输出两倍通道，GLU 后一半作为门控，最终保留 out_channels。
                2 * out_channels,
                # 卷积在时间轴上观察的相邻帧范围。
                kernel_size=kernel_size,
                # 卷积每次前移的时间间隔。
                stride=stride,
                # 右侧补齐到固定长度所缺的帧数。
                padding=(kernel_size - stride) // 2,
            )
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # encoder 或不改变长度的块选择普通一维卷积，同样先产生 2*out_channels。
            convolution = nn.Conv1d(
                # 主干、说话人/步条件和 BNF 拼接后的总输入通道。
                total_in,
                # 先输出两倍通道，GLU 后一半作为门控，最终保留 out_channels。
                2 * out_channels,
                # 卷积在时间轴上观察的相邻帧范围。
                kernel_size=kernel_size,
                # 卷积每次前移的时间间隔。
                stride=stride,
                # 右侧补齐到固定长度所缺的帧数。
                padding=(kernel_size - 1) // 2,
            )
        # 对刚刚二选一建好的卷积统一初始化并加 weight norm，再登记为这个块的子层。
        self.convolution = _weight_norm(convolution)

    # 本块的前向计算同时接收当前特征、全局条件和与语音内容对齐的 BNF。
    def forward(
        # 输入当前声学隐藏特征、全局身份/步条件和整句内容特征。
        self, x: torch.Tensor, condition: torch.Tensor, bnf: torch.Tensor
    # 返回执行当前网络或条件模块的一次前向计算得到的张量；类型标注帮助调用方理解结果，不改变数值。
    ) -> torch.Tensor:
        # x 的最后一维永远是当前层的时间长度。
        length = x.shape[-1]
        # 旧 checkpoint 在 stride=2/4 时直接取第 0,2,... 或 0,4,... 帧。
        # 新实验可先按当前主干长度取样，再用完全相同形状的 1×1 权重做通道投影。
        if self.bnf_downsampling == CENTERED_BNF_DOWNSAMPLING:
            # 已经按当前层时间分辨率准备的 BNF。
            bnf_for_projection = (
                # 长度已经一致时直接使用原 BNF，不做额外插值。
                bnf
                # 先判断整句内容帧数是否已等于当前隐藏层长度。
                if bnf.shape[-1] == length
                # 低分辨率层将整句 BNF 线性取样到当前隐藏帧中心。
                else F.interpolate(
                    # 表示语音内容的 144 维 BNF 特征。
                    bnf,
                    # 目标时间帧数取当前主干隐藏层的长度。
                    size=length,
                    # 当前选择的时间对齐或插值模式。
                    mode="linear",
                    # 采用半像素中心插值，匹配两次偶数卷积下采样造成的中心位置。
                    align_corners=False,
                )
            )
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 已经按当前层时间分辨率准备的 BNF。
            bnf_for_projection = bnf
        # 经过 1×1 卷积投影的内容条件。
        bnf_condition = self.bnf_projection(bnf_for_projection)
        # 如果这里不相等，说明 mel/BNF 没有同样补到 4 的倍数，skip 后面必然出错。
        if bnf_condition.shape[-1] != length:
            # 立即抛错并同时报告实际长度、期望长度和 stride，方便定位预处理对齐问题。
            raise RuntimeError(
                # 错误说明某个下采样尺度的 BNF 时间长度与主干不一致。
                "BNF/time mismatch at stride "
                # 错误附带当前尺度、BNF 实际帧数与主干要求帧数。
                f"{self.bnf_stride}: got {bnf_condition.shape[-1]}, expected {length}. "
                # 提示 mel 与 BNF 需要同步补为四的倍数，以通过两次二倍下采样。
                "Pad the mel and BNF to a multiple of four before the network."
            )
        # condition 原形状 [B,2*cond_dim,1]。expand 的 -1 表示保持该维原大小，
        # 最后一维扩展到 length，让每个时间帧都看到同一个 timestep/speaker 条件。
        condition = condition.expand(-1, -1, length)
        # dim=1 就是沿“通道轴”拼接，而不是把不同语音或时间首尾接起来。
        # 卷积输出 2*out_channels，F.glu(dim=1) 沿通道对半分并做门控。
        return F.glu(
            # 先沿通道合并三种信息，再用主干卷积融合邻近时间上下文。
            self.convolution(torch.cat((x, condition, bnf_condition), dim=1)),
            # 特征或嵌入的通道宽度。
            dim=1,
        )


# 这个类把上面的 11 个条件 GLU 块和最后输出卷积组装成完整 VoiceGrad 教师网络。
class VoiceGrad(nn.Module):
    # 按当前教师重建设置组装 12 层、512 隐藏通道的 U-Net。
    """Twelve-layer, 512-channel any-to-any VoiceGrad network."""

    # U-Net 的直观理解：encoder 把时间分辨率压低以提取更抽象的信息，decoder
    # 再把长度恢复；skip connection 把 encoder 的细节直接送到对应 decoder，
    # 避免音素边界等局部信息在下采样中丢失。

    # 创建对象并固定后续读取或网络计算要遵循的参数；下面的参数分别限定输入数据和处理约束。
    def __init__(
        # 当前数据集或网络对象。
        self,
        # mel 频带数，本实验为 80。
        n_mels: int = 80,
        # BNF 内容维数，本实验为 144。
        n_bnf: int = 144,
        # VoiceGrad 主干隐藏通道数，论文设为 512。
        n_channels: int = 512,
        # 完整扩散链的步数。
        n_levels: int = 1000,
        # 步编号和说话人各自的投影宽度，本实验选 128。
        cond_dim: int = 128,
        # 各块 BNF 投影宽度，本实验选 32。
        bnf_out_dim: int = 32,
        # 说话人向量宽度，本实验为 256。
        spk_emb_dim: int = 256,
        # 网络各时间分辨率上的 BNF 取样规则。
        bnf_downsampling: str = LEGACY_BNF_DOWNSAMPLING,
    # 本接口负责创建对象并固定后续读取或网络计算要遵循的参数，处理结果通过对象状态或文件保存，不额外返回数值。
    ) -> None:
        # 初始化最外层 nn.Module；否则下面赋给 self 的子模块无法被 state_dict 正确收集。
        super().__init__()
        # 保存 mel 通道数，forward 会用它检查输入第二维是否为 80。
        self.n_mels = n_mels
        # 保存 BNF 通道数，forward 会检查内容特征是否为 144 维。
        self.n_bnf = n_bnf
        # 保存主干宽度 512，便于对象自描述，也与论文结构保持一致。
        self.n_channels = n_channels
        # 保存训练扩散总步数 1000，用来拒绝超出合法范围的 timestep。
        self.n_levels = n_levels
        # 保存 timestep/speaker 各自投影维数 128。
        self.cond_dim = cond_dim
        # 保存外部 speaker embedding 维数 256，用于严格形状检查。
        self.spk_emb_dim = spk_emb_dim
        # 按网络各时间分辨率上的 BNF 取样规则决定是否进入此分支；判断条件为 bnf_downsampling not in BNF_DOWNSAMPLING_MODES。
        if bnf_downsampling not in BNF_DOWNSAMPLING_MODES:
            # 创建对象并固定后续读取或网络计算要遵循的参数未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(
                # 错误输出未被实现的内部 BNF 采样模式名。
                f"Unknown BNF downsampling {bnf_downsampling!r}; "
                # 列出当前允许选择的内部 BNF 对齐模式。
                f"choose one of {BNF_DOWNSAMPLING_MODES}"
            )
        # 网络各时间分辨率上的 BNF 取样规则。
        self.bnf_downsampling = bnf_downsampling

        # timestep: [B] -> [B,128]；speaker: [B,256] -> [B,128]。
        # 建立扩散步的正弦编码，让相邻噪声阶段拥有有规律的条件表示。
        self.step_embedding = SinusoidalStepEmbedding(cond_dim)
        # Linear 做 256→128 的可学习投影，使说话人条件与时间步条件具有相同宽度。
        self.speaker_projection = _initialize(nn.Linear(spk_emb_dim, cond_dim))
        # 两部分随后拼接，所以每个卷积块收到的全局条件共 2*cond_dim=256 通道。
        condition_channels = 2 * cond_dim

        # 让全部 GLU 块共用同一 BNF 下采样规则；下面的参数分别限定输入数据和处理约束。
        def make_block(*block_args, **block_kwargs) -> ConditionalGLUBlock:
            # 把整网选择的 BNF 时间规则传给每一个条件卷积块。
            block_kwargs["bnf_downsampling"] = bnf_downsampling
            # 按统一规则构建一个带全局与内容条件的 GLU 层。
            return ConditionalGLUBlock(*block_args, **block_kwargs)

        # Encoder 的时间长度变化：
        # layer1 T；layer2 T/2；layer3 T/2；layer4 T/4；layer5 T/4；layer6 T/4。
        # k 是 kernel size（卷积核宽度），s 是 stride（每次移动多少帧）。
        # Encoder: k9/s1, k8/s2, k9/s1, k8/s2, k5/s1, k5/s1.
        self.encoder1 = make_block(
            # 80→512 通道，核宽 9、步长 1；保留完整时间轴和逐帧 BNF。
            n_mels, n_channels, 9, 1, condition_channels, n_bnf, bnf_out_dim, 1
        )
        # 第 2 层 stride=2，第一次把时间长度从 T 降到约 T/2。
        self.encoder2 = make_block(
            # 512→512 通道，核宽 8、步长 2；从全长输入进行第一次时间下采样。
            n_channels, n_channels, 8, 2, condition_channels, n_bnf, bnf_out_dim, 1
        )
        # 第 3 层保持 T/2，但 BNF stride=2，确保内容条件也处在 T/2。
        self.encoder3 = make_block(
            # 核宽 9、主干步长 1；在半长度上融合按 2 倍率取样的 BNF。
            n_channels, n_channels, 9, 1, condition_channels, n_bnf, bnf_out_dim, 2
        )
        # 第 4 层再次 stride=2，把主干时间长度从 T/2 降到 T/4。
        self.encoder4 = make_block(
            # 核宽 8、步长 2；输入为半长度，输出进入四分之一长度。
            n_channels, n_channels, 8, 2, condition_channels, n_bnf, bnf_out_dim, 2
        )
        # 第 5 层在最小分辨率 T/4 上继续提取上下文，卷积核宽度为 5。
        self.encoder5 = make_block(
            # 核宽 5、步长 1；在四分之一时间分辨率上融合 BNF 内容条件。
            n_channels, n_channels, 5, 1, condition_channels, n_bnf, bnf_out_dim, 4
        )
        # 第 6 层是 encoder 最深处，输出会直接送进 decoder1。
        self.encoder6 = make_block(
            # 核宽 5、步长 1；在四分之一时间分辨率上融合 BNF 内容条件。
            n_channels, n_channels, 5, 1, condition_channels, n_bnf, bnf_out_dim, 4
        )

        # Decoder 在原 VoiceGrad Figure 3 中从右向左读取。每次 skip 都是通道拼接，
        # 因此 C 通道与 C 通道拼完变 2C，后续层输入必须写 2*n_channels。
        # 时间变化：decoder1/2 为 T/4；decoder3 上采样至 T/2；decoder4 保持；
        # decoder5 上采样回 T；最后 output projection 输出 80 个 mel 通道。
        # Decoder (read right-to-left in Figure 3). Every incoming skip is a
        # channel concatenation, hence 2*n_channels after the first block.
        self.decoder1 = make_block(
            # 核宽 5、步长 1；在四分之一时间分辨率上融合 BNF 内容条件。
            n_channels, n_channels, 5, 1, condition_channels, n_bnf, bnf_out_dim, 4
        )
        # decoder2 的输入先拼接 encoder5，所以输入通道是 2C；时间仍保持 T/4。
        self.decoder2 = make_block(
            # 输入已拼接同尺度 skip 特征，主干通道数从 512 变为 1024。
            2 * n_channels,
            # VoiceGrad 主干隐藏通道数，论文设为 512。
            n_channels,
            # 此低分辨率层的卷积核观察相邻 5 个隐藏帧。
            5,
            # 主干步长为 1，当前层保持时间分辨率。
            1,
            # 扩散步条件与身份条件沿通道拼接后的总宽度，为 128+128=256。
            condition_channels,
            # BNF 内容维数，本实验为 144。
            n_bnf,
            # 各块 BNF 投影宽度，本实验选 32。
            bnf_out_dim,
            # 当前输入隐藏长度是原长度的四分之一，BNF 使用对应 4 倍取样比例。
            4,
        )
        # decoder3 使用转置卷积，接收与 encoder4 拼接的 2C，并从 T/4 上采样到 T/2。
        self.decoder3 = make_block(
            # 输入已拼接同尺度 skip 特征，主干通道数从 512 变为 1024。
            2 * n_channels,
            # VoiceGrad 主干隐藏通道数，论文设为 512。
            n_channels,
            # 此上采样层采用宽度 8 的转置卷积核。
            8,
            # 转置卷积步长设为 2，将当前隐藏时轴上采样为两倍。
            2,
            # 扩散步条件与身份条件沿通道拼接后的总宽度，为 128+128=256。
            condition_channels,
            # BNF 内容维数，本实验为 144。
            n_bnf,
            # 各块 BNF 投影宽度，本实验选 32。
            bnf_out_dim,
            # 当前输入隐藏长度是原长度的四分之一，BNF 使用对应 4 倍取样比例。
            4,
            # 是否使用转置卷积恢复时间分辨率。
            transpose=True,
        )
        # decoder4 与 encoder3 做 skip 拼接，在 T/2 分辨率上进一步融合信息。
        self.decoder4 = make_block(
            # 输入已拼接同尺度 skip 特征，主干通道数从 512 变为 1024。
            2 * n_channels,
            # VoiceGrad 主干隐藏通道数，论文设为 512。
            n_channels,
            # 此保持分辨率的层采用宽度 9 的普通卷积核。
            9,
            # 主干步长为 1，当前层保持时间分辨率。
            1,
            # 扩散步条件与身份条件沿通道拼接后的总宽度，为 128+128=256。
            condition_channels,
            # BNF 内容维数，本实验为 144。
            n_bnf,
            # 各块 BNF 投影宽度，本实验选 32。
            bnf_out_dim,
            # 此块输入隐藏长度为原时轴的一半，BNF 按 2 倍比例准备对应内容条件。
            2,
        )
        # decoder5 再次用转置卷积，从 T/2 恢复到原始时间长度 T。
        self.decoder5 = make_block(
            # 输入已拼接同尺度 skip 特征，主干通道数从 512 变为 1024。
            2 * n_channels,
            # VoiceGrad 主干隐藏通道数，论文设为 512。
            n_channels,
            # 此上采样层采用宽度 8 的转置卷积核。
            8,
            # 转置卷积步长设为 2，将当前隐藏时轴上采样为两倍。
            2,
            # 扩散步条件与身份条件沿通道拼接后的总宽度，为 128+128=256。
            condition_channels,
            # BNF 内容维数，本实验为 144。
            n_bnf,
            # 各块 BNF 投影宽度，本实验选 32。
            bnf_out_dim,
            # 此块输入隐藏长度为原时轴的一半，BNF 按 2 倍比例准备对应内容条件。
            2,
            # 是否使用转置卷积恢复时间分辨率。
            transpose=True,
        )
        # 最后一层接收 decoder5 与 encoder1 的 2C 通道，把它们变回 80 维噪声预测。
        self.output_projection = _weight_norm(
            # 最后一层没有 GLU，只把拼接后的 2C 通道映射回 80 个噪声 mel 通道。
            nn.Conv1d(2 * n_channels, n_mels, kernel_size=9, padding=4)
        )

    # 这个内部函数把原始 timestep 和说话人向量整理成所有卷积块共用的全局条件。
    def _prepare_condition(
        # 提供每条语音的扩散步、身份向量以及用于核对的批大小。
        self, steps: torch.Tensor, speaker_embedding: torch.Tensor, batch_size: int
    # 返回把时间步编码和目标说话人表示拼成全局条件得到的张量；类型标注帮助调用方理解结果，不改变数值。
    ) -> torch.Tensor:
        # 目标输出形状是 [B,2*cond_dim,1]，之后每层再沿时间轴 expand。
        if steps.ndim == 0:
            # 0 维 Tensor 是单个标量；repeat 给 batch 中每条语音复制同一个 t。
            steps = steps.repeat(batch_size)
        # 无论原来是 [B] 还是 [B,1]，都压成恰好一维，便于下面检查 batch 数量。
        steps = steps.reshape(-1)
        # 每条语音必须恰好有一个时间步；数量不一致通常是 batch 拼装错误。
        if steps.shape[0] != batch_size:
            # 报错中同时写期望值和实际值，让使用者不用猜是哪一维出了问题。
            raise ValueError(f"Expected {batch_size} timesteps, got {steps.shape[0]}.")
        # 本实现使用论文式 1-based timestep，所以合法范围是 1 到 n_levels，而不是从 0 开始。
        if torch.any(steps < 1) or torch.any(steps > self.n_levels):
            # 只要 batch 中任意一个 t 越界就停止，避免非法索引扩散系数表。
            raise ValueError(f"Timesteps must be in [1, {self.n_levels}].")

        # 如果只传来一条裸 [256] 向量，就给它补上 batch 轴，统一成二维格式。
        if speaker_embedding.ndim == 1:
            # 一条 [256] 向量没有 batch 轴，unsqueeze(0) 把它变成 [1,256]。
            speaker_embedding = speaker_embedding.unsqueeze(0)
        # 某些旧提取器保存成 [B,256,1]；squeeze(-1) 只删除末尾那个长度为 1 的轴。
        if speaker_embedding.ndim == 3 and speaker_embedding.shape[-1] == 1:
            # 删除的只是冗余维，数值和元素顺序都不改变。
            speaker_embedding = speaker_embedding.squeeze(-1)
        # 规范化后必须严格是 [batch_size, 256]，不能依赖 PyTorch 猜测或广播身份向量。
        if speaker_embedding.shape != (batch_size, self.spk_emb_dim):
            # 抛错时把真实 tuple(shape) 打印出来，便于回查 speaker embedding 预处理。
            raise ValueError(
                # 错误提示目标身份条件应是批次×身份维数。
                "Speaker embedding must have shape "
                # 错误列出应有批次×身份维数及实际张量形状。
                f"[{batch_size}, {self.spk_emb_dim}], got {tuple(speaker_embedding.shape)}."
            )

        # unsqueeze(-1) 在最后加一维：[B,128] -> [B,128,1]。
        step_condition = self.step_embedding(steps).unsqueeze(-1)
        # 先转 float32，再做可学习线性投影并补时间轴，得到 [B,128,1]。
        speaker_condition = self.speaker_projection(
            # 将参考身份向量转为 float32，再通过线性层映射到条件宽度。
            speaker_embedding.float()
        # 追加单元素时间轴，后续在各层广播同一身份条件。
        ).unsqueeze(-1)
        # 沿通道轴拼成 [B,256,1]。
        return torch.cat((step_condition, speaker_condition), dim=1)

    # 该辅助方法只使用传入的数据，不依赖网络实例内部状态。
    @staticmethod
    # 静态函数不读取 self；它只负责检查并拼接一对 decoder/encoder 特征。
    def _concatenate(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 这里就是论文图中的 ⊕。拼接前 batch 和时间长度必须一样，通道数可以不同。
        if x.shape[0] != skip.shape[0] or x.shape[-1] != skip.shape[-1]:
            # batch 或时间轴不一致时无法沿通道拼接，所以先给出双方完整形状并停止。
            raise RuntimeError(
                # 错误同时列出解码器与编码器跳接形状，帮助定位时间尺度错位。
                f"Invalid skip shapes: decoder={tuple(x.shape)}, encoder={tuple(skip.shape)}"
            )
        # dim=1 表示只增加通道数；batch 数和时间帧数保持不变。
        return torch.cat((x, skip), dim=1)

    # 该辅助方法只使用传入的数据，不依赖网络实例内部状态。
    @staticmethod
    # 该静态函数把时间长度补到 4 的倍数，并同时记住补零前的长度供最后裁切。
    def _pad_time(x: torch.Tensor, multiple: int = 4) -> Tuple[torch.Tensor, int]:
        # 网络连续两次 stride=2，总下采样倍数为 4。T 若不是 4 的倍数，decoder
        # 还原后可能和 skip 差一帧。(-T)%4 正好得到还需补的 0、1、2 或 3 帧。
        original_length = x.shape[-1]
        # Python 取模保证 amount 落在 0 到 multiple-1；已经整除时结果恰好为 0。
        amount = (-original_length) % multiple
        # 条件表达式“补帧数非零才 pad，否则直接用 x”；元组第二项始终返回原长度。
        return (F.pad(x, (0, amount)) if amount else x), original_length

    # 这是完整 VoiceGrad 的前向入口，输出和输入 mel 同形状的噪声估计。
    def forward(
        # 当前数据集或网络对象。
        self,
        # 当前扩散状态中的 mel 张量。
        x: torch.Tensor,
        # 当前扩散时刻的 1-based 编号。
        noise_level: torch.Tensor,
        # 256 维目标说话人条件。
        spk_emb: torch.Tensor,
        # 表示语音内容的 144 维 BNF 特征。
        bnf: torch.Tensor,
    # 返回执行当前网络或条件模块的一次前向计算得到的张量；类型标注帮助调用方理解结果，不改变数值。
    ) -> torch.Tensor:
        # forward 是 PyTorch 规定的前向函数；调用 model(...) 会自动进入这里。
        # 它只预测“一次当前 timestep 的噪声”，不是一次生成完整音频。
        # diffusion.sample 会在 30 个 timestep 中重复调用这个 forward。

        # 先做明确的形状检查，避免等到深层卷积才得到难以理解的 channel 报错。
        if x.ndim != 3 or x.shape[1] != self.n_mels:
            # mel 必须是 [B,80,T]；不满足就立即报告实际 shape。
            raise ValueError(f"Expected mel [B, {self.n_mels}, T], got {tuple(x.shape)}.")
        # BNF 也必须是三维，而且 batch 数和 144 内容通道必须与配置匹配。
        if bnf.ndim != 3 or bnf.shape[:2] != (x.shape[0], self.n_bnf):
            # 这类错误通常表示保存时把 [T,144] 忘记转置成 [144,T]。
            raise ValueError(f"Expected BNF [B, {self.n_bnf}, T], got {tuple(bnf.shape)}.")
        # 除通道格式外，还要保证每个 BNF 时间帧与一个 mel 帧一一对应。
        if bnf.shape[-1] != x.shape[-1]:
            # 网络不会在这里偷偷插值，因为静默对齐可能让音素和声学帧错位。
            raise ValueError("Mel and BNF must be aligned to the same frame count.")

        # mel 与 BNF 必须补同样数量的时间帧，才能一直保持内容条件对齐。
        x, original_length = self._pad_time(x)
        # 对 BNF 独立调用相同补齐逻辑，并保存它自己的原长度供交叉检查。
        bnf, bnf_original_length = self._pad_time(bnf)
        # 理论上前面已验证长度相等；这里是第二道防线，防止将来修改补齐函数后悄悄失配。
        if original_length != bnf_original_length:
            # 若真的不一致就停止，不能让错误内容条件进入 encoder。
            raise RuntimeError("Mel and BNF lengths changed inconsistently.")
        # 把本 batch 的扩散步和目标说话人身份合并成 [B,256,1] 条件。
        condition = self._prepare_condition(noise_level, spk_emb, x.shape[0])

        # ---------------- Encoder：T -> T/2 -> T/4 ----------------
        # 第一层在完整 mel 时轴融合全局条件与内容 BNF，并保存供同尺度跳接使用。
        enc1 = self.encoder1(x, condition, bnf)
        # enc2 接收上一层输出并第一次下采样；每行变量都会留给后面的 skip 使用。
        enc2 = self.encoder2(enc1, condition, bnf)
        # enc3 保持当前时间分辨率，扩大特征的上下文感受范围。
        enc3 = self.encoder3(enc2, condition, bnf)
        # enc4 第二次下采样到 T/4。
        enc4 = self.encoder4(enc3, condition, bnf)
        # enc5 在 T/4 上继续处理，并作为 decoder2 的 skip。
        enc5 = self.encoder5(enc4, condition, bnf)
        # enc6 得到 encoder 最深层表示，直接喂给 decoder1。
        enc6 = self.encoder6(enc5, condition, bnf)

        # ---------------- Decoder：每层先拼对应 encoder skip ----------------
        dec1 = self.decoder1(enc6, condition, bnf)
        # 把 dec1 与同分辨率 enc5 沿通道拼接，再经 decoder2 融合。
        dec2 = self.decoder2(self._concatenate(dec1, enc5), condition, bnf)
        # 拼 enc4 后由 decoder3 上采样到 T/2。
        dec3 = self.decoder3(self._concatenate(dec2, enc4), condition, bnf)
        # 拼 enc3 后在 T/2 上继续融合。
        dec4 = self.decoder4(self._concatenate(dec3, enc3), condition, bnf)
        # 拼 enc2 后由 decoder5 恢复到完整时间分辨率 T。
        dec5 = self.decoder5(self._concatenate(dec4, enc2), condition, bnf)
        # 最后拼最浅层 enc1，输出每个 mel 通道、每个时间帧的预测噪声。
        output = self.output_projection(self._concatenate(dec5, enc1))
        # ... 表示前面所有维度保持不变，只把最后的时间轴裁回补零前长度。
        return output[..., :original_length]


# 这个函数只做结构自检：确认主干卷积确实是论文要求的 12 层。
def count_convolution_layers(model: VoiceGrad) -> int:
    # 这是自检辅助函数。model.modules() 会遍历所有子层；isinstance 判断卷积类型。
    # 每个 GLU 块都有一个额外 BNF 1x1 投影，共 11 个，它们不算论文“12 层主干”，
    # 所以统计全部 Conv1d/ConvTranspose1d 后减去 11。
    return sum(
        # 只统计普通与转置卷积，不将激活或其他容器算作主干层。
        isinstance(module, (nn.Conv1d, nn.ConvTranspose1d))
        # 遍历完整网络的各个子层进行结构核对。
        for module in model.modules()
        # 排除最外层模型容器本身。
        if module is not model
    # 从卷积总数中扣去 11 个内容投影层，得到论文讨论的主干层数。
    ) - 11  # Exclude the eleven 1x1 BNF projections.


# 该判断是 Python 的脚本入口开关；文件被 import 时条件为假，不会运行下面的自测。
if __name__ == "__main__":
    # Python 标准入口保护：直接运行 python model.py 才执行，import 时不会执行。
    # 这里用随机假数据检查奇数时间长度、输入输出形状和主干层数，不做真实训练。
    network = VoiceGrad()
    # 随机创建 2 条、80 mel 通道、129 帧的 noisy mel；129 故意不是 4 的倍数。
    source = torch.randn(2, 80, 129)
    # 创建与 source 时间严格对齐的 144 维假 BNF。
    linguistic = torch.randn(2, 144, 129)
    # 为 batch 中两条语音分别创建一个 256 维假说话人向量。
    speakers = torch.randn(2, 256)
    # 两条语音分别测试扩散早期 t=1 和最晚 t=1000。
    steps = torch.tensor([1, 1000])
    # 真正调用一次完整网络，触发 padding、encoder、decoder 和最终裁切。
    result = network(source, steps, speakers, linguistic)
    # assert 表示“这个条件必须成立”；输出形状若不同，程序立刻报告自测失败。
    assert result.shape == source.shape
    # 第二个断言确认扣除 BNF 投影后，主干卷积层数严格等于论文的 12。
    assert count_convolution_layers(network) == 12
    # 两个断言都通过后打印成功信息，便于服务器上肉眼确认。
    print("VoiceGrad shape/topology check passed.")
