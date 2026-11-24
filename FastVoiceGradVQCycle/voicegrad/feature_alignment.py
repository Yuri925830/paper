#!/usr/bin/env python3
# 集中定义 BNF 与 mel 的时轴对齐规则，训练和推理必须遵循同一选择。
"""Shared BNF-to-mel time-axis alignment utilities.

The BNF extractor emits one frame every 10 ms with its first frame centred at
waveform time 0.  VoiceGrad mel extraction uses HiFi-GAN-style reflect padding;
with n_fft=1024 and hop=256 its first frame is centred at sample 128 and later
frames are 256 samples apart.  Equalising only the two array lengths ignores
that half-frame offset and lets rounding error accumulate over an utterance.

``legacy_length_linear`` preserves the alignment used by the already-trained
teacher checkpoints.  ``time_linear`` maps every mel-frame centre to the BNF
time grid explicitly and is intended for controlled A/B tests and new training
runs.  Keeping both modes prevents an old checkpoint from silently changing
behaviour.
"""

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Final

# 使用 PyTorch 张量表示声学特征、扩散状态及网络参数。
import torch
# 提供插值、补帧和损失等无需单独创建网络层的运算。
import torch.nn.functional as F


# 旧权重使用的规则：只按两个数组的总帧数做线性缩放。
LEGACY_ALIGNMENT: Final[str] = "legacy_length_linear"
# 物理时轴规则：以每个 mel 窗口在原始录音中的中心时刻取 BNF。
TIME_ALIGNMENT: Final[str] = "time_linear"
# 公开允许的两种规则；切换必须作为独立实验记录，不能偷偷改变旧权重输入。
ALIGNMENT_MODES: Final[tuple[str, ...]] = (LEGACY_ALIGNMENT, TIME_ALIGNMENT)


# 计算手动反射补齐后第一个 mel 窗口的物理中心；下面的参数分别限定输入数据和处理约束。
def mel_first_frame_center_samples(n_fft: int, hop_size: int) -> float:
    # 计算手动反射补齐后第一个 mel 窗口的物理中心。
    """Return the input-waveform centre of mel frame zero.

    HiFi-GAN manually reflect-pads by ``(n_fft-hop_size)/2`` and then calls an
    STFT with ``center=False``.  The first FFT window centre is therefore
    ``n_fft/2 - (n_fft-hop_size)/2 = hop_size/2`` in waveform coordinates.
    The unsimplified calculation is retained so a future configuration error
    is easier to diagnose.
    """

    # 按每个时窗的 FFT 点数决定是否进入此分支；判断条件为 n_fft <= 0 or hop_size <= 0 or n_fft < hop_size。
    if n_fft <= 0 or hop_size <= 0 or n_fft < hop_size:
        # 计算手动反射补齐后第一个 mel 窗口的物理中心未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Invalid STFT geometry: n_fft={n_fft}, hop_size={hop_size}")
    # STFT 前两侧各反射补齐的采样数。
    manual_padding = (float(n_fft) - float(hop_size)) / 2.0
    # 减去手动补齐量后，首窗中心回到原波形的 hop/2；当前为第 128 个采样点。
    return float(n_fft) / 2.0 - manual_padding


# 将 mel 窗口中心转换成 BNF 时间轴上的小数索引；下面的参数分别限定输入数据和处理约束。
def mel_centres_on_bnf_grid(
    # 需要对齐到的 mel 时间帧数。
    mel_length: int,
    # 从这里起的参数必须写明名称传入，减少路径或审核开关位置混淆。
    *,
    # 音频文件实际或预期的采样率。
    sample_rate: int = 22050,
    # 每个时窗的 FFT 点数。
    n_fft: int = 1024,
    # 相邻 mel 窗口的样本间距。
    hop_size: int = 256,
    # BNF 相邻帧的真实时间间距，当前提取器为 0.010 秒。
    bnf_frame_shift_seconds: float = 0.010,
    # 张量执行计算的设备。
    device: torch.device | None = None,
    # 张量的数值类型。
    dtype: torch.dtype = torch.float32,
# 返回将 mel 窗口中心转换成 BNF 时间轴上的小数索引得到的张量；类型标注帮助调用方理解结果，不改变数值。
) -> torch.Tensor:
    # 将 mel 窗口中心转换成 BNF 时间轴上的小数索引。
    """Return fractional BNF indices for all mel-frame centre times."""

    # 按需要对齐到的 mel 时间帧数决定是否进入此分支；判断条件为 mel_length <= 0。
    if mel_length <= 0:
        # 将 mel 窗口中心转换成 BNF 时间轴上的小数索引未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"mel_length must be positive, got {mel_length}")
    # 按音频文件实际或预期的采样率决定是否进入此分支；判断条件为 sample_rate <= 0 or bnf_frame_shift_seconds <= 0。
    if sample_rate <= 0 or bnf_frame_shift_seconds <= 0:
        # 将 mel 窗口中心转换成 BNF 时间轴上的小数索引未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 错误说明采样率和 BNF 帧移都必须是正数，才有合法物理时间轴。
            "sample_rate and bnf_frame_shift_seconds must be positive, got "
            # 错误附带实际采样率与 BNF 帧移，方便判断哪项非正。
            f"{sample_rate} and {bnf_frame_shift_seconds}"
        )

    # 第一个 mel 窗口在原波形上的中心采样位置。
    first_center = mel_first_frame_center_samples(n_fft, hop_size)
    # 从零起的 mel 帧编号。
    mel_indices = torch.arange(mel_length, device=device, dtype=dtype)
    # 各 mel 窗口中心在原始波形中的位置。
    centre_samples = first_center + mel_indices * float(hop_size)
    # 把波形采样位置除以每个 BNF 帧间隔对应的采样数，得到小数帧索引。
    return centre_samples / (float(sample_rate) * float(bnf_frame_shift_seconds))


# 在真实时间位置混合左右相邻的 BNF 帧；下面的参数分别限定输入数据和处理约束。
def _time_linear_interpolate(
    # 表示语音内容的 144 维 BNF 特征。
    bnf: torch.Tensor,
    # 需要对齐到的 mel 时间帧数。
    mel_length: int,
    # 从这里起的参数必须写明名称传入，减少路径或审核开关位置混淆。
    *,
    # 音频文件实际或预期的采样率。
    sample_rate: int,
    # 每个时窗的 FFT 点数。
    n_fft: int,
    # 相邻 mel 窗口的样本间距。
    hop_size: int,
    # BNF 相邻帧的真实时间间距，当前提取器为 0.010 秒。
    bnf_frame_shift_seconds: float,
# 返回在真实时间位置混合左右相邻的 BNF 帧得到的张量；类型标注帮助调用方理解结果，不改变数值。
) -> torch.Tensor:
    # 在真实时间位置混合左右相邻的 BNF 帧。
    """Linearly sample ``bnf`` at the physical centre time of each mel frame."""

    # 输入 BNF 的有效时间帧数。
    bnf_length = int(bnf.shape[-1])
    # 按输入 BNF 的有效时间帧数决定是否进入此分支；判断条件为 bnf_length <= 0。
    if bnf_length <= 0:
        # 在真实时间位置混合左右相邻的 BNF 帧未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError("BNF must contain at least one time frame")

    # 每个 mel 帧中心在 BNF 时轴上的小数索引。
    positions = mel_centres_on_bnf_grid(
        # 需要对齐到的 mel 时间帧数。
        mel_length,
        # 音频文件实际或预期的采样率。
        sample_rate=sample_rate,
        # 每个时窗的 FFT 点数。
        n_fft=n_fft,
        # 相邻 mel 窗口的样本间距。
        hop_size=hop_size,
        # BNF 相邻帧的真实时间间距，当前提取器为 0.010 秒。
        bnf_frame_shift_seconds=bnf_frame_shift_seconds,
        # 张量执行计算的设备。
        device=bnf.device,
        # 张量的数值类型。
        dtype=bnf.dtype,
    # 越过 BNF 首尾的取样位置限制在可用范围内，防止访问越界。
    ).clamp(0.0, float(bnf_length - 1))

    # 位于采样位置左侧的 BNF 帧索引。
    lower = positions.floor().to(torch.long)
    # 位于采样位置右侧的 BNF 帧索引。
    upper = (lower + 1).clamp_max(bnf_length - 1)
    # 两个相邻 BNF 帧线性混合的右侧权重。
    weight = positions - lower.to(positions.dtype)
    # 让插值权重只沿时间变化的广播形状。
    weight_shape = (1,) * (bnf.ndim - 1) + (mel_length,)
    # 两个相邻 BNF 帧线性混合的右侧权重。
    weight = weight.view(weight_shape)

    # 取出的左侧 BNF 帧特征。
    lower_values = bnf.index_select(-1, lower)
    # 取出的右侧 BNF 帧特征。
    upper_values = bnf.index_select(-1, upper)
    # 按距离混合左右两帧；位置越接近右帧，其权重越大。
    return lower_values * (1.0 - weight) + upper_values * weight


# 统一分派旧长度插值或物理时轴插值；下面的参数分别限定输入数据和处理约束。
def align_bnf_to_mel(
    # 表示语音内容的 144 维 BNF 特征。
    bnf: torch.Tensor,
    # 需要对齐到的 mel 时间帧数。
    mel_length: int,
    # 从这里起的参数必须写明名称传入，减少路径或审核开关位置混淆。
    *,
    # 当前选择的时间对齐或插值模式。
    mode: str = LEGACY_ALIGNMENT,
    # 音频文件实际或预期的采样率。
    sample_rate: int = 22050,
    # 每个时窗的 FFT 点数。
    n_fft: int = 1024,
    # 相邻 mel 窗口的样本间距。
    hop_size: int = 256,
    # BNF 相邻帧的真实时间间距，当前提取器为 0.010 秒。
    bnf_frame_shift_seconds: float = 0.010,
# 返回统一分派旧长度插值或物理时轴插值得到的张量；类型标注帮助调用方理解结果，不改变数值。
) -> torch.Tensor:
    # 统一分派旧长度插值或物理时轴插值。
    """Align ``[..., T_bnf]`` BNF features to ``mel_length`` frames.

    Args:
        bnf: A floating-point tensor whose final axis is time.  Both
            ``[channels,time]`` and ``[batch,channels,time]`` are supported.
        mel_length: Desired mel-frame count.
        mode: ``legacy_length_linear`` or ``time_linear``.

    Returns:
        A tensor with the same leading dimensions as ``bnf`` and a final axis
        of exactly ``mel_length``.
    """

    # 内容特征至少应包含通道与时间两个轴。
    if bnf.ndim < 2:
        # 统一分派旧长度插值或物理时轴插值未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"BNF must have at least two dimensions, got {tuple(bnf.shape)}")
    # 线性插值要求浮点特征，整数数组会丢失小数混合信息。
    if not torch.is_floating_point(bnf):
        # 统一分派旧长度插值或物理时轴插值未通过，抛出下方包含具体路径或实际值的错误。
        raise TypeError(f"BNF must be floating point, got {bnf.dtype}")
    # 按需要对齐到的 mel 时间帧数决定是否进入此分支；判断条件为 mel_length <= 0。
    if mel_length <= 0:
        # 统一分派旧长度插值或物理时轴插值未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"mel_length must be positive, got {mel_length}")
    # 按当前选择的时间对齐或插值模式决定是否进入此分支；判断条件为 mode not in ALIGNMENT_MODES。
    if mode not in ALIGNMENT_MODES:
        # 统一分派旧长度插值或物理时轴插值未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Unknown BNF alignment {mode!r}; choose one of {ALIGNMENT_MODES}")

    # 按当前选择的时间对齐或插值模式决定是否进入此分支；判断条件为 mode == LEGACY_ALIGNMENT。
    if mode == LEGACY_ALIGNMENT:
        # 旧规则只看帧数，相等时直接返回；这并不能证明物理帧中心完全一致。
        if bnf.shape[-1] == mel_length:
            # 返回表示语音内容的 144 维 BNF 特征，供调用方继续处理。
            return bnf
        # 原始二维 BNF 是否需要临时加 batch 轴。
        added_batch = bnf.ndim == 2
        # 当前正在校验的数值或记录。
        value = bnf.unsqueeze(0) if added_batch else bnf
        # 插值到 mel 帧数后的 BNF。
        aligned = F.interpolate(
            # 当前正在校验的数值或记录。
            value,
            # 将最后时间轴调整到 mel 的帧数。
            size=mel_length,
            # 当前选择的时间对齐或插值模式。
            mode="linear",
            # 按半像素中心规则插值，保持旧训练所使用的数值行为。
            align_corners=False,
        )
        # 仅移除本函数临时增加的 batch 轴，保留调用者原有维数。
        return aligned.squeeze(0) if added_batch else aligned

    # 物理时轴模式交给显式帧中心计算函数处理。
    return _time_linear_interpolate(
        # 表示语音内容的 144 维 BNF 特征。
        bnf,
        # 需要对齐到的 mel 时间帧数。
        mel_length,
        # 音频文件实际或预期的采样率。
        sample_rate=sample_rate,
        # 每个时窗的 FFT 点数。
        n_fft=n_fft,
        # 相邻 mel 窗口的样本间距。
        hop_size=hop_size,
        # BNF 相邻帧的真实时间间距，当前提取器为 0.010 秒。
        bnf_frame_shift_seconds=bnf_frame_shift_seconds,
    )
