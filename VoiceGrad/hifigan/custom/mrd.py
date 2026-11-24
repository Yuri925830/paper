#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MRD 以三组频谱分辨率检查波形，线性幅度 STFT 保留回传生成一侧的梯度。
"""
FastVoiceGrad 论文声码器所需的“多分辨率频谱判别器”（MRD）。

先用一句最简单的话说明这个文件在干什么：

    一段真的声音和一段生成器造出来的声音，都会先被变成三张不同清晰度的频谱图；
    三个小判别器分别看一张频谱图，并判断它更像“真声音”还是“假声音”。

为什么单独写这个文件：

1. FastVoiceGrad 论文第 4.1 节明确写明，V（vocoder）和 D 使用修改版
   HiFi-GAN V1；修改内容是把 HiFi-GAN 原本的多尺度判别器 MSD 换成
   UnivNet 的多分辨率判别器 MRD。
2. 项目里的 ``hifi-gan/models.py`` 已经是官方 HiFi-GAN V1 代码，里面的
   Generator 和 MPD 可以直接复用，所以这里不抄一份重复代码，只补上论文要求的 MRD。
3. UnivNet 原文第 3.2、4.3 节和图 2 给出了三个 STFT 参数组和二维卷积层结构；
   本文件按这些公开信息实现。

必须诚实说明的复现边界：论文没有公开 STFT 库的每一个开关。Hann window 来自
UnivNet 论文；padding/center 采用广泛使用的 maum-ai UnivNet 参考实现约定：先左右
reflect-pad ``(n_fft-hop)/2``，再用 ``center=False`` 做 STFT。该仓库自称
unofficial implementation，所以这仍是写入配置的工程选择，不能冒充作者秘密参数。
"""

# ``dataclass`` 能把一组只读参数打包成一个小对象。
# 例如 STFTResolution(1024, 120, 600) 就表示一套频谱分辨率。
from dataclasses import dataclass

# ``List`` 和 ``Sequence`` 只是类型提示：帮助读代码的人和编辑器知道这里应放列表。
# 它们不会改变模型的计算结果。
from typing import List, Sequence, Tuple

# PyTorch 是本项目训练神经网络的核心库；Tensor 是多维数字数组。
import torch

# ``torch.nn`` 提供神经网络层的基类 Module、卷积层 Conv2d 和层列表 ModuleList。
import torch.nn as nn

# ``F`` 是 functional 的缩写；这里用它调用不需要单独保存参数的 LeakyReLU 激活函数。
import torch.nn.functional as F

# weight_norm 是论文和 HiFi-GAN/UnivNet 都使用的权重归一化技术。
# 它把卷积核的“方向”和“长度”分开参数化，通常能让训练更稳定。
from torch.nn.utils import weight_norm


# -----------------------------------------------------------------------------
# 一套 STFT 分辨率的只读说明书
# -----------------------------------------------------------------------------

# ``frozen=True`` 表示对象建立后不允许随手改数字，防止训练中参数被意外篡改。
@dataclass(frozen=True)
# 给每组频谱分辨率保存 FFT 点数、帧移和窗长。
class STFTResolution:
    # 一个只读对象对应一组 FFT 点数、帧移和 Hann 窗长度。
    """保存一次短时傅里叶变换需要的三个整数。"""

    # fft_size：每一帧做多少点 FFT；越大，频率刻度通常越细。
    fft_size: int
    # hop_size：相邻两帧向前走多少采样点；越小，时间观察越密。
    hop_size: int
    # win_length：每次真正乘 Hann 窗并观察多少个采样点。
    win_length: int


# 这三组数字逐字来自 UnivNet 论文第 4.3 节：
# (FFT 点数, 帧移, Hann 窗长度)。三种观察尺度同时使用，才叫“多分辨率”。
PAPER_MRD_RESOLUTIONS: Tuple[STFTResolution, ...] = (
    # 第一分支：1024 点 FFT、120 点帧移、600 点 Hann 窗。
    STFTResolution(fft_size=1024, hop_size=120, win_length=600),
    # 第二分支：更细的频率观察，FFT 2048、帧移 240、窗长 1200。
    STFTResolution(fft_size=2048, hop_size=240, win_length=1200),
    # 第三分支：更密的时间观察，FFT 512、帧移 50、窗长 240。
    STFTResolution(fft_size=512, hop_size=50, win_length=240),
)


# -----------------------------------------------------------------------------
# 单个分辨率的频谱判别器
# -----------------------------------------------------------------------------

# 一个 MRD 分支，只负责一种时间—频率分辨率。
class SpectrogramDiscriminator(nn.Module):
    # 单个频谱判别分支先做指定 STFT，再用二维卷积输出局部真假分数。
    """
    只负责一种 STFT 分辨率的小判别器。

    输入波形形状是 ``[B, 1, T]``：

    - B 是 batch 中有几条语音；
    - 1 表示单声道；
    - T 是每条语音的采样点数。

    它先把波形变成线性幅度频谱，再用 UnivNet 图 2 中的二维卷积逐层检查。
    """

    # 构造频谱判别器，固定分辨率与 LeakyReLU 负半轴斜率。
    def __init__(
        # 当前判别器对象，保存卷积权重和频谱参数。
        self,
        # 本分支使用的 FFT、帧移、窗长组合。
        resolution: STFTResolution,
        # 采用 UnivNet 披露的 LeakyReLU 负半轴斜率 0.2。
        lrelu_slope: float = 0.2,
    # 构造函数把参数和网络层保存在当前对象中，按 Python 约定不返回结果对象。
    ) -> None:
        # 所有 nn.Module 子类的构造函数开头都必须调用父类初始化。
        # 否则 PyTorch 无法正确登记后面创建的卷积参数。
        super().__init__()

        # 保存这一小判别器负责的 FFT、hop 和 window 参数，前向计算时会用到。
        self.resolution = resolution

        # UnivNet 第 4.3 节明确写出 LeakyReLU 的负半轴斜率 alpha=0.2。
        self.lrelu_slope = float(lrelu_slope)

        # Hann window 是做 STFT 前乘在每一帧上的平滑窗。
        # ``register_buffer`` 表示它属于模型、会跟随模型移动到 GPU，
        # 但它不是需要优化器学习的参数。
        # ``persistent=False`` 表示不用把这个可由公式重新生成的常量塞进检查点。
        self.register_buffer(
            # 把 Hann 窗登记为随模型移动的缓冲区。
            "window",
            # 按该分辨率窗长生成 Hann 窗。
            torch.hann_window(resolution.win_length),
            # 窗可以重新计算，无需写进权重检查点。
            persistent=False,
        )

        # 下面五层卷积对应 UnivNet 论文图 2 的 MRD 左半部分：
        # 1->32 的 (3,9) 卷积；三层 stride=(1,2) 的 (3,9) 卷积；
        # 最后一层 32->32 的 (3,3) 卷积。
        # padding 让频率轴高度基本不变，并让时间轴按 stride=2 逐步缩短。
        self.convs = nn.ModuleList(
            [
                # 把卷积权重拆成方向与幅度参数，沿用 HiFi-GAN/UnivNet 的训练约定。
                weight_norm(
                    # 将单通道频谱扩展为 32 个特征通道，观察 3 个频率格与 9 个时间帧。
                    nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))
                ),
                # 把卷积权重拆成方向与幅度参数，沿用 HiFi-GAN/UnivNet 的训练约定。
                weight_norm(
                    # 增加二维卷积，联合观察邻近频率和时间纹理。
                    nn.Conv2d(
                        # 卷积输入或输出均为 32 个隐藏特征通道。
                        32,
                        # 卷积输入或输出均为 32 个隐藏特征通道。
                        32,
                        # 卷积覆盖 3 个频率格、9 个时间帧。
                        kernel_size=(3, 9),
                        # 保留频率轴密度，时间轴每层缩短约一半。
                        stride=(1, 2),
                        # 在频率两侧补 1 格、时间两侧补 4 帧，配合该卷积核。
                        padding=(1, 4),
                    )
                ),
                # 把卷积权重拆成方向与幅度参数，沿用 HiFi-GAN/UnivNet 的训练约定。
                weight_norm(
                    # 增加二维卷积，联合观察邻近频率和时间纹理。
                    nn.Conv2d(
                        # 卷积输入或输出均为 32 个隐藏特征通道。
                        32,
                        # 卷积输入或输出均为 32 个隐藏特征通道。
                        32,
                        # 卷积覆盖 3 个频率格、9 个时间帧。
                        kernel_size=(3, 9),
                        # 保留频率轴密度，时间轴每层缩短约一半。
                        stride=(1, 2),
                        # 在频率两侧补 1 格、时间两侧补 4 帧，配合该卷积核。
                        padding=(1, 4),
                    )
                ),
                # 把卷积权重拆成方向与幅度参数，沿用 HiFi-GAN/UnivNet 的训练约定。
                weight_norm(
                    # 增加二维卷积，联合观察邻近频率和时间纹理。
                    nn.Conv2d(
                        # 卷积输入或输出均为 32 个隐藏特征通道。
                        32,
                        # 卷积输入或输出均为 32 个隐藏特征通道。
                        32,
                        # 卷积覆盖 3 个频率格、9 个时间帧。
                        kernel_size=(3, 9),
                        # 保留频率轴密度，时间轴每层缩短约一半。
                        stride=(1, 2),
                        # 在频率两侧补 1 格、时间两侧补 4 帧，配合该卷积核。
                        padding=(1, 4),
                    )
                ),
                # 把卷积权重拆成方向与幅度参数，沿用 HiFi-GAN/UnivNet 的训练约定。
                weight_norm(
                    # 最后隐藏层以 3×3 局部窗口整合 32 通道特征。
                    nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
                ),
            ]
        )

        # 最后一层把 32 个通道压成 1 个真假分数图。
        # 它后面不接激活函数，因为最小二乘 GAN 直接使用原始分数。
        self.conv_post = weight_norm(
            # 将隐藏特征变为单通道的局部真假分数。
            nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    # 把时域波形转为线性幅度频谱，保留回传生成器所需的梯度。
    def _linear_magnitude_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        # 波形先手动反射补边，再取复数 STFT 的线性幅度，不作 mel 或对数压缩。
        """把 ``[B,1,T]`` 波形变成 ``[B,1,F,frames]`` 线性幅度频谱。"""

        # 先检查维数，尽早把错误形状拦住；否则错误会在很深的卷积里才爆出来。
        if waveform.ndim != 3:
            # 报错原因：先检查维数，尽早把错误形状拦住；否则错误会在很深的卷积里才爆出来。
            raise ValueError(
                # 先检查维数，尽早把错误形状拦住；否则错误会在很深的卷积里才爆出来。
                f"MRD expects waveform [B,1,T], but got {tuple(waveform.shape)}"
            )

        # 第二维必须是 1，因为本项目的 VCTK 波形已经是单声道。
        if waveform.size(1) != 1:
            # 报错原因：第二维必须是 1，因为本项目的 VCTK 波形已经是单声道。
            raise ValueError(
                # 第二维必须是 1，因为本项目的 VCTK 波形已经是单声道。
                f"MRD expects one audio channel, but got {waveform.size(1)}"
            )

        # squeeze(1) 只删除长度为 1 的声道轴：[B,1,T] -> [B,T]。
        waveform_2d = waveform.squeeze(1)

        # maum-ai UnivNet 参考 MRD 先做 (n_fft-hop)/2 的 reflect padding，
        # 随后以 center=False 计算 STFT。这个细节不是论文正文公开参数，已在
        # config 的 undisclosed_choice_notes 中明确标记。
        pad_amount = int(
            # 根据 FFT 点数与帧移差计算左右反射补边长度。
            (self.resolution.fft_size - self.resolution.hop_size) / 2
        )
        # 按参考实现手动反射补边后再计算各分辨率 STFT。
        padded_waveform = F.pad(
            # 临时增加声道轴，以便对时间轴做反射补边。
            waveform_2d.unsqueeze(1),
            # 左右补相同点数，维持参考实现的帧位置。
            (pad_amount, pad_amount),
            # 边界使用波形反射，不额外插入突然的静音。
            mode="reflect",
        # 完成时间轴补边后去掉临时声道轴。
        ).squeeze(1)

        # torch.stft 把每条长波形切成很多重叠短帧，并计算每帧的复数频谱。
        # ``return_complex=True`` 是新版本 PyTorch 的正确写法，输出复数张量。
        # 这里绝对不能放在 ``torch.no_grad()`` 中：生成波形经过频谱判别器得到的
        # 梯度必须沿 STFT 回到生成器，生成器才能学会改善频谱细节。
        complex_spec = torch.stft(
            # 频谱输入已经手动补好边界。
            padded_waveform,
            # 本分支 FFT 点数决定频率格数量。
            n_fft=self.resolution.fft_size,
            # 本分支帧移决定时间采样密度。
            hop_length=self.resolution.hop_size,
            # 本分支 Hann 窗长度决定每帧实际观察范围。
            win_length=self.resolution.win_length,
            # 复用随模型移到同一设备的 Hann 窗。
            window=self.window,
            # 关闭自动居中补边，使用前面明确的反射补边。
            center=False,
            # 保留参考实现的 STFT 边界模式。
            pad_mode="reflect",
            # 不除以 FFT 长度，保持线性幅度标度。
            normalized=False,
            # 实数波形只需保存非负频率一半。
            onesided=True,
            # 输出复数频谱，下一步再取其幅度。
            return_complex=True,
        )

        # abs() 计算复数的模，也就是线性幅度频谱。
        # 注意：论文写的是 linear spectrogram magnitude，所以这里不取 log，也不做 mel 压缩。
        magnitude = complex_spec.abs()

        # 增加一个“图像通道”轴：[B,F,frames] -> [B,1,F,frames]，
        # 这样才能交给 Conv2d；Conv2d 把频率和时间当成图片的高和宽。
        return magnitude.unsqueeze(1)

    # 同时返回真假分数及各层特征，后者用于生成器的特征匹配损失。
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # 单分辨率分支返回局部真假分数和每层特征，生成器据此计算对抗及特征匹配损失。
        """返回最终真假分数和每层特征，后者用于 HiFi-GAN feature matching loss。"""

        # 先把原始波形转为该分辨率下的线性幅度频谱。
        x = self._linear_magnitude_spectrogram(waveform)

        # fmap 是 feature maps 的缩写；保存每层内部观察结果。
        # 真实语音和生成语音的这些结果会做 L1 距离，形成特征匹配损失。
        feature_maps: List[torch.Tensor] = []

        # 依次通过五个二维卷积层。
        for conv in self.convs:
            # 卷积学习局部的时间-频率纹理，例如谐波是否清晰、频谱是否过度平滑。
            x = conv(x)
            # LeakyReLU 在负数区域也保留小斜率，避免普通 ReLU 把负数梯度全部切断。
            x = F.leaky_relu(x, negative_slope=self.lrelu_slope)
            # 保存这一层特征，供生成器的 feature matching loss 使用。
            feature_maps.append(x)

        # 最后一层产生真假分数图，不接 sigmoid；LSGAN 直接把真分数推向 1、假分数推向 0。
        x = self.conv_post(x)

        # 最终分数也作为最后一层特征保存，行为与官方 HiFi-GAN 判别器一致。
        feature_maps.append(x)

        # flatten 从第 1 轴开始摊平，保留 batch 轴：每条语音得到一长串局部真假分数。
        scores = torch.flatten(x, start_dim=1)

        # 返回真假分数与中间特征列表。
        return scores, feature_maps


# -----------------------------------------------------------------------------
# 三个分辨率合在一起的 MRD
# -----------------------------------------------------------------------------

# 把三个独立频谱判别器并联，分别检查不同分辨率。
class MultiResolutionDiscriminator(nn.Module):
    # 三个独立分支比较同一对波形，并分别保留分数与每层匹配特征。
    """把三个 ``SpectrogramDiscriminator`` 组合成论文所说的 MRD。"""

    # 构造频谱判别器，固定分辨率与 LeakyReLU 负半轴斜率。
    def __init__(
        # 当前判别器对象，保存卷积权重和频谱参数。
        self,
        # 设置MRD 的多组时频分辨率。
        resolutions: Sequence[STFTResolution] = PAPER_MRD_RESOLUTIONS,
        # 采用 UnivNet 披露的 LeakyReLU 负半轴斜率 0.2。
        lrelu_slope: float = 0.2,
    # 构造函数把参数和网络层保存在当前对象中，按 Python 约定不返回结果对象。
    ) -> None:
        # 初始化 PyTorch Module 基类。
        super().__init__()

        # FastVoiceGrad 引用的 UnivNet MRD 使用恰好三个分辨率；数量不对就立刻报错。
        if len(resolutions) != 3:
            # 报错原因：FastVoiceGrad 引用的 UnivNet MRD 使用恰好三个分辨率；数量不对就立刻报错。
            raise ValueError(f"Paper-aligned MRD requires 3 resolutions, got {len(resolutions)}")

        # 为三套 STFT 参数各建一个独立的小判别器；它们不共享卷积权重。
        self.discriminators = nn.ModuleList(
            [
                # 为这一组分辨率创建独立卷积权重的判别器。
                SpectrogramDiscriminator(resolution, lrelu_slope=lrelu_slope)
                # 遍历三套规定的 STFT 参数。
                for resolution in resolutions
            ]
        )

    # 让同一组 MRD 分支分别观察参考和生成波形，并保留各层结果。
    def forward(
        # 当前判别器对象，保存卷积权重和频谱参数。
        self,
        # 参考波形；声码器训练时来自数据，学生 ACDD 中由冻结 V 还原真实 mel。
        real_waveform: torch.Tensor,
        # 生成波形；梯度可经 MRD、STFT 回传到负责生成的一侧。
        generated_waveform: torch.Tensor,
    # 按顺序返回参考分数、生成分数、参考各层特征和生成各层特征。
    ) -> Tuple[
        # 返回每个分辨率的一组真假分数张量。
        List[torch.Tensor],
        # 返回每个分辨率的一组真假分数张量。
        List[torch.Tensor],
        # 返回每个分辨率各层特征的列表，供特征匹配。
        List[List[torch.Tensor]],
        # 返回每个分辨率各层特征的列表，供特征匹配。
        List[List[torch.Tensor]],
    ]:
        # 三个分支分别处理同一对参考与生成波形，按官方接口汇总四组分数和特征列表。
        """
        同时判断真实波形和生成波形。

        返回顺序故意和官方 HiFi-GAN 的 MPD/MSD 完全相同，使训练损失函数可以原样复用。
        """

        # 下面四个列表分别保存：真实分数、生成分数、真实中间特征、生成中间特征。
        real_scores: List[torch.Tensor] = []
        # 收集三种分辨率对生成波形的真假分数。
        generated_scores: List[torch.Tensor] = []
        # 收集参考波形在三个分支各层的特征。
        real_feature_maps: List[List[torch.Tensor]] = []
        # 收集生成波形对应的特征，后续逐层比较。
        generated_feature_maps: List[List[torch.Tensor]] = []

        # 三个小判别器分别看同一对波形，但使用不同时间/频率分辨率。
        for discriminator in self.discriminators:
            # 真实声音经过当前小判别器。
            real_score, real_features = discriminator(real_waveform)
            # 生成声音经过同一个小判别器；共享权重才能进行公平真假比较。
            generated_score, generated_features = discriminator(generated_waveform)

            # 把当前分辨率的结果加入总列表。
            real_scores.append(real_score)
            # 加入当前分辨率对生成波形的打分。
            generated_scores.append(generated_score)
            # 保留本分支的参考特征。
            real_feature_maps.append(real_features)
            # 保留本分支的生成特征。
            generated_feature_maps.append(generated_features)

        # 返回格式与 HiFi-GAN MultiPeriodDiscriminator.forward 一致。
        return (
            # 先返回参考波形的各分支分数。
            real_scores,
            # 再返回生成波形的各分支分数。
            generated_scores,
            # 随后返回参考的多层特征。
            real_feature_maps,
            # 最后返回生成的多层特征，与官方判别器接口顺序一致。
            generated_feature_maps,
        )
