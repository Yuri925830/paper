#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 数据层仅从训练清单选音频，整句归一化、随机裁切后现场计算条件 mel。
"""
论文架构对齐声码器的数据读取与 mel 计算工具。

整个文件只做三件事：

1. 只读取 ``train_manifest.csv``，确保 10 位未见测试说话人绝不进入声码器训练；
2. 按官方 HiFi-GAN 非 fine-tuning 路径逐 utterance 峰值归一化并乘 0.95，
   随机裁切 8192 个 waveform samples；
3. 从该归一化片段现场计算 conditioning mel，并提供相同函数给 mel loss。

这与 ``hifigan/official/meldataset.py`` 的 VCTK scratch-training 路径一致。
FastVoiceGrad 只公开冻结的 pretrained Generator，没有公开 standalone vocoder recipe；
因此这里是继承 HiFi-GAN V1 基座的、明确标注的复现选择。
"""

# csv 用来读取表头形式的 train_manifest.csv。
import csv

# math 提供 ceil；随机裁切时要把 8192 个采样点换算成 mel 帧数。
import math

# random 负责每次从一条长语音中随机选取一个训练片段。
import random

# dataclass 把一条语音的路径和身份信息打包为结构化对象。
from dataclasses import dataclass

# Path 用 ``/`` 安全拼接 Windows/Linux 路径；类型提示让函数输入输出更容易读懂。
from pathlib import Path
# 标明张量、清单和配置的输入输出类型，不改变运行数据。
from typing import Dict, List, Optional, Sequence, Tuple, Union

# librosa 只在这里生成 mel filter bank；真正的 STFT 由 PyTorch 完成。
import librosa

# NumPy 读取 .npy mel，并处理波形数组的 dtype 与有限值检查。
import numpy as np

# SoundFile 能把 WAV 可靠地读取为 float32 的 [-1,1] 波形。
import soundfile as sf

# PyTorch 提供张量、STFT 和 Dataset 基类。
import torch
# 调用反射补边函数，使 mel 帧位置沿用 HiFi-GAN 约定。
import torch.nn.functional as F
# 使用 PyTorch 的数据集、批次采样和加载接口。
from torch.utils.data import Dataset


# mel basis 和 Hann window 对同一套参数永远相同。
# 用字典缓存后，一个训练进程只创建一次，避免每个 batch 重算。
_MEL_BASIS_CACHE: Dict[str, torch.Tensor] = {}
# 按窗长、设备和精度缓存 Hann 窗，避免每批重建。
_HANN_WINDOW_CACHE: Dict[str, torch.Tensor] = {}


# 按 HiFi-GAN 的窗、补边和自然对数规则把波形转为 mel。
def mel_spectrogram_torch(
    # 输入参数：输入时域波形。
    waveform: torch.Tensor,
    # 输入参数：短时傅里叶变换点数。
    n_fft: int,
    # 输入参数：mel 频带数量。
    num_mels: int,
    # 输入参数：每秒波形采样点数。
    sampling_rate: int,
    # 输入参数：相邻 mel 帧之间的波形采样点间隔。
    hop_size: int,
    # 输入参数：分析窗的采样点长度。
    win_size: int,
    # 输入参数：mel 滤波器最低频率。
    fmin: int,
    # 输入参数：mel 滤波器最高频率，空值表示奈奎斯特频率。
    fmax: Optional[int],
# 返回 [批次, mel 频带, 时间帧] 的自然对数 mel 张量。
) -> torch.Tensor:
    # 按 HiFi-GAN 的窗、补边和自然对数规则把波形转为 mel。
    """把 ``[B,T]`` 波形变成 HiFi-GAN 风格的 ``[B,mel,Tm]`` 自然对数 mel。"""

    # 训练时一次传入一个 batch，因此必须是二维 [batch, samples]。
    if waveform.ndim != 2:
        # 报错原因：训练时一次传入一个 batch，因此必须是二维 [batch, samples]。
        raise ValueError(
            # 训练时一次传入一个 batch，因此必须是二维 [batch, samples]。
            f"mel_spectrogram_torch expects [B,T], got {tuple(waveform.shape)}"
        )

    # 记录张量所在设备；mel basis 和 Hann window 必须放在同一块 GPU/CPU 上。
    device = waveform.device

    # 缓存键包含所有会改变 mel 数值的参数，以及设备和 dtype。
    # fmax=None 表示一直使用到 Nyquist 频率，也必须成为键的一部分。
    basis_key = (
        # fmax=None 表示一直使用到 Nyquist 频率，也必须成为键的一部分。
        f"{sampling_rate}_{n_fft}_{num_mels}_{fmin}_{fmax}_"
        # fmax=None 表示一直使用到 Nyquist 频率，也必须成为键的一部分。
        f"{device}_{waveform.dtype}"
    )
    # 按窗长、设备与精度区分 Hann 窗缓存。
    window_key = f"{win_size}_{device}_{waveform.dtype}"

    # 第一次遇到这套配置时，用 librosa 建立 [num_mels, n_fft/2+1] 滤波矩阵。
    if basis_key not in _MEL_BASIS_CACHE:
        # 首次使用某组参数时创建线性频率到 mel 频带的投影矩阵。
        mel_basis_numpy = librosa.filters.mel(
            # mel 滤波器使用输入波形的真实采样率。
            sr=sampling_rate,
            # 设置短时傅里叶变换点数。
            n_fft=n_fft,
            # 把线性频率幅度投影到约定数量的 mel 频带。
            n_mels=num_mels,
            # 设置mel 滤波器最低频率。
            fmin=fmin,
            # 设置mel 滤波器最高频率，空值表示奈奎斯特频率。
            fmax=fmax,
        )
        # from_numpy 不复制数据；随后 to() 把它转换到训练设备和波形 dtype。
        _MEL_BASIS_CACHE[basis_key] = torch.from_numpy(mel_basis_numpy).to(
            # 缓存常量放到与波形相同的 CPU 或 GPU。
            device=device,
            # 设置张量的数值精度。
            dtype=waveform.dtype,
        )

    # 第一次使用该窗长度时建立 Hann window，并缓存在当前设备上。
    if window_key not in _HANN_WINDOW_CACHE:
        # 首次使用这一窗长、设备与精度时构造 Hann 窗。
        _HANN_WINDOW_CACHE[window_key] = torch.hann_window(
            # 使用与配置一致的 Hann 窗长度。
            win_size,
            # 缓存常量放到与波形相同的 CPU 或 GPU。
            device=device,
            # 设置张量的数值精度。
            dtype=waveform.dtype,
        )

    # HiFi-GAN 在 STFT 前手动做 reflect padding。
    # 1024 FFT、256 hop 时左右各补 (1024-256)/2=384 个采样点。
    pad_amount = int((n_fft - hop_size) / 2)
    # 先手动补齐波形两侧，再做不自动居中的 STFT。
    padded = F.pad(
        # 临时补出声道轴，满足一维反射补边接口。
        waveform.unsqueeze(1),
        # 波形左右两侧补相同采样点数，保持原有帧中心约定。
        (pad_amount, pad_amount),
        # 使用反射边界，复制邻近变化而非突然补零。
        mode="reflect",
    # 补边完成后移除临时声道轴，恢复 [批次, 采样点]。
    ).squeeze(1)

    # 做复数 STFT；center=False 是因为上一行已经手动补过边界。
    complex_spec = torch.stft(
        # 对已经手工补边的波形做短时傅里叶变换。
        padded,
        # 设置短时傅里叶变换点数。
        n_fft=n_fft,
        # 频谱帧移与 mel 时间轴保持一致。
        hop_length=hop_size,
        # 指定每帧实际参与分析的窗长。
        win_length=win_size,
        # 复用同设备、同精度的 Hann 窗。
        window=_HANN_WINDOW_CACHE[window_key],
        # 关闭 STFT 自动居中补边，防止与前面的手动补边重复。
        center=False,
        # 保留 HiFi-GAN 的边界模式约定。
        pad_mode="reflect",
        # 不施加 FFT 长度归一化，保留训练时的幅度标度。
        normalized=False,
        # 实数波形只保留非负频率一侧。
        onesided=True,
        # 返回复数频谱，后面显式计算幅度。
        return_complex=True,
    )

    # 复数模得到线性幅度；加 1e-9 避免零附近数值不稳定。
    magnitude = torch.sqrt(complex_spec.real.square() + complex_spec.imag.square() + 1e-9)

    # [mel,freq] @ [B,freq,time] -> [B,mel,time]，把线性频率压到 mel 频带。
    mel = torch.matmul(_MEL_BASIS_CACHE[basis_key], magnitude)

    # HiFi-GAN 的 spectral normalization 实际是自然对数压缩。
    # clamp 防止 log(0) 变成负无穷。
    return torch.log(torch.clamp(mel, min=1e-5))


# -----------------------------------------------------------------------------
# manifest 中一行数据的结构
# -----------------------------------------------------------------------------

# 使用只读记录保存语音身份，避免训练中意外改路径。
@dataclass(frozen=True)
# 定义训练语音条目的身份与文件位置。
class VocoderItem:
    # 单条训练记录绑定说话人、语音编号与对应波形和 mel 路径。
    """保存一条训练语音需要的最小信息。"""

    # speaker 例如 p233，用于统计训练说话人数和拼 mel 子目录。
    speaker: str
    # utt_id 例如 p233_001，是语音文件的唯一名字。
    utt_id: str
    # wav_path 是 22.05 kHz 单声道 wav 的完整路径。
    wav_path: Path
    # mel_path 是与 wav 对齐的 [80,T] float32 .npy 路径。
    mel_path: Path


# 优先采用清单中的绝对路径，必要时按项目数据布局重建路径。
def _resolve_wav_path(project_root: Path, row: Dict[str, str]) -> Path:
    # 优先采用清单中的绝对路径，必要时按项目数据布局重建路径。
    """优先使用服务器 manifest 路径，并提供本地下载副本的安全回退路径。"""

    # manifest 是按服务器根目录制作的，正式训练时这个绝对路径应该存在。
    manifest_path = Path(row["wav22k_path"])
    # 清单中的完整波形路径可用时优先采用它。
    if manifest_path.exists():
        # 返回清单记录的真实波形位置。
        return manifest_path

    # 本地电脑可能只下载了项目文件夹，绝对路径自然不存在。
    # 回退路径仍严格遵循 data/wav22k/<speaker>/<utt_id>.wav 的服务器布局。
    return (
        # 从明确的项目根目录定位训练数据。
        project_root
        # 进入项目数据目录。
        / "data"
        # 选择预处理为 22.05 kHz 的波形集合。
        / "wav22k"
        # 按清单中的说话人编号选择子目录。
        / row["speaker"]
        # 按清单语音编号定位同名 WAV。
        / f"{row['utt_id']}.wav"
    )


# 读取训练清单并拒绝测试行、未见说话人和重复语音。
def load_training_items(project_root: Path, manifest_path: Path) -> List[VocoderItem]:
    # 读取训练清单并拒绝测试行、未见说话人和重复语音。
    """读取并严格验证训练清单；任何 test/validation 行都会直接拒绝。"""

    # 清单不存在时立即停止，避免训练程序悄悄得到空数据集。
    if not manifest_path.is_file():
        # 报错原因：清单不存在时立即停止，避免训练程序悄悄得到空数据集。
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    # items 保存最终可交给 Dataset 的结构化条目；seen_keys 用于拒绝重复语音。
    items: List[VocoderItem] = []
    # 收集已读到的说话人—语音编号组合，以检查重复。
    seen_keys = set()

    # newline="" 是 csv 官方推荐写法；UTF-8 能正确处理文本字段中的字符。
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        # 按清单列名读取，避免列顺序变化引起身份错配。
        reader = csv.DictReader(handle)

        # 清单必须包含这些列；少一列说明文件格式不是本项目确认过的版本。
        required_columns = {
            # 字段含义：该语音所属的数据子集。
            "split",
            # 字段含义：说话人编号。
            "speaker",
            # 字段含义：语音编号。
            "utt_id",
            # 字段含义：语音编号。
            "wav22k_path",
            # 字段含义：是否属于禁止用于训练的未见说话人。
            "is_unseen_speaker",
        }
        # 取得 CSV 实际列名；没有表头时按空集合处理。
        actual_columns = set(reader.fieldnames or [])
        # 找出训练数据必需但清单中缺少的列。
        missing_columns = required_columns - actual_columns
        # 缺少必要身份字段时停止，不能猜测训练与测试归属。
        if missing_columns:
            # 报错原因：缺少必要身份字段时停止，不能猜测训练与测试归属。
            raise ValueError(
                # 错误信息列出训练清单缺少哪些身份字段。
                f"Training manifest is missing columns: {sorted(missing_columns)}"
            )

        # 一行一行读取训练清单。
        for line_number, row in enumerate(reader, start=2):
            # 本实验没有 validation；声码器训练更不能使用 test 行。
            if row["split"].strip().lower() != "train":
                # 报错原因：本实验没有 validation；声码器训练更不能使用 test 行。
                raise ValueError(
                    # 指出清单中混入非训练行的准确位置和实际 split。
                    f"Non-train row found at line {line_number}: split={row['split']!r}"
                )

            # 未见说话人标记必须是 0；这是第二道防止测试说话人泄漏的保险。
            if row["is_unseen_speaker"].strip() not in {"0", "False", "false"}:
                # 报错原因：未见说话人标记必须是 0；这是第二道防止测试说话人泄漏的保险。
                raise ValueError(
                    # 未见说话人标记必须是 0；这是第二道防止测试说话人泄漏的保险。
                    f"Unseen/test speaker leaked into training at line {line_number}"
                )

            # (speaker, utt_id) 组合应当唯一。
            key = (row["speaker"].strip(), row["utt_id"].strip())
            # 相同训练语音重复出现会改变采样权重，发现后直接报错。
            if key in seen_keys:
                # 报错原因：相同训练语音重复出现会改变采样权重，发现后直接报错。
                raise ValueError(f"Duplicate training utterance at line {line_number}: {key}")
            # 登记已通过身份检查的语音。
            seen_keys.add(key)

            # 拼出服务器 wav 路径和预提取 mel 路径。
            wav_path = _resolve_wav_path(project_root, row)
            # 设置预提取 mel 路径。
            mel_path = project_root / "data" / "mel" / key[0] / f"{key[1]}.npy"

            # 把这一行变为不可变的 VocoderItem。
            items.append(
                # 把清单字段整理为不可变语音条目。
                VocoderItem(
                    # 保存已去除空白的说话人编号。
                    speaker=key[0],
                    # 保存与波形文件对应的语音编号。
                    utt_id=key[1],
                    # 设置训练波形路径。
                    wav_path=wav_path,
                    # 设置预提取 mel 路径。
                    mel_path=mel_path,
                )
            )

    # 空清单肯定不能训练。
    if not items:
        # 报错原因：空清单肯定不能训练。
        raise ValueError(f"Training manifest contains zero utterances: {manifest_path}")

    # 返回全部训练条目。
    return items


# -----------------------------------------------------------------------------
# PyTorch Dataset：每次返回一个随机对齐片段
# -----------------------------------------------------------------------------

# 把训练清单包装成可按索引读取、可重复裁切的声码器数据集。
class ManifestMelWaveDataset(Dataset):
    # 训练样本从归一化波形片段现场生成条件 mel，测试身份行不会进入数据集。
    """从训练 manifest 读取波形并按官方 scratch recipe 生成 conditioning mel。"""

    # 保存波形裁切与 mel 计算参数；这些设置决定声码器看到的输入分布。
    def __init__(
        # 当前数据集实例，用于读取已保存的参数。
        self,
        # 输入参数：训练语音条目。
        items: Sequence[VocoderItem],
        # 输入参数：每个声码器训练片段的采样点数。
        segment_size: int,
        # 输入参数：相邻 mel 帧之间的波形采样点间隔。
        hop_size: int,
        # 输入参数：mel 频带数量。
        num_mels: int,
        # 输入参数：每秒波形采样点数。
        sampling_rate: int,
        # 输入参数：短时傅里叶变换点数。
        n_fft: int,
        # 输入参数：分析窗的采样点长度。
        win_size: int,
        # 输入参数：mel 滤波器最低频率。
        fmin: int,
        # 输入参数：mel 滤波器最高频率，空值表示奈奎斯特频率。
        fmax: Optional[int],
    # 构造函数把参数和网络层保存在当前对象中，按 Python 约定不返回结果对象。
    ) -> None:
        # 初始化 Dataset 父类。
        super().__init__()

        # list() 建立自己的列表，避免外部后来修改原 Sequence。
        self.items = list(items)
        # 每次训练使用多少个波形采样点；HiFi-GAN V1 官方配置是 8192。
        self.segment_size = int(segment_size)
        # 一帧 mel 对应多少波形采样点；论文明确是 256。
        self.hop_size = int(hop_size)
        # 预期 mel 频带数；论文明确是 80。
        self.num_mels = int(num_mels)
        # 预期波形采样率；论文明确是 22050 Hz。
        self.sampling_rate = int(sampling_rate)
        # 保存 STFT 点数，转换为整数供频谱计算。
        self.n_fft = int(n_fft)
        # 保存 Hann 分析窗长度。
        self.win_size = int(win_size)
        # 保存 mel 滤波器最低频率。
        self.fmin = int(fmin)
        # 最高频率为空时保留奈奎斯特上限约定。
        self.fmax = None if fmax is None else int(fmax)
        # 8192/256=32；ceil 保证非整数时也有足够帧覆盖整个波形片段。
        self.frames_per_segment = int(math.ceil(self.segment_size / self.hop_size))

        # 基本参数若不合理，尽早报错。
        if self.segment_size <= 0 or self.hop_size <= 0:
            # 报错原因：基本参数若不合理，尽早报错。
            raise ValueError("segment_size and hop_size must be positive")

    # 向 DataLoader 提供当前数据集或剩余批次的大小。
    def __len__(self) -> int:
        # len(dataset) 就是训练清单中的语音条数。
        return len(self.items)

    # 读取一个训练样本，并使用指定种子生成可恢复的对齐裁切。
    def __getitem__(
        # 接受普通语音索引，或可恢复采样器给出的索引与裁切种子。
        self, index: Union[int, Tuple[int, int]]
    # 依次返回 mel 张量、同位置波形张量和语音身份字符串。
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        # 读取一个训练样本，并使用指定种子生成可恢复的对齐裁切。
        """返回 ``(mel片段[80,32], 波形片段[8192], 唯一键)``。"""

        # 正式训练的确定性 batch sampler 会给出 ``(item_index, crop_seed)``。
        # crop_seed 把每个 epoch/位置的随机裁切固定下来，因此从 epoch 中间的
        # checkpoint 恢复时会得到与不中断运行完全相同的输入片段。
        crop_seed: Optional[int] = None
        # 保存本次实际选中的训练语音序号。
        item_index: int
        # 确定性采样器会同时给语音序号和独立裁切种子。
        if isinstance(index, tuple):
            # 确定性索引只能包含两个元素，避免误解恢复状态。
            if len(index) != 2:
                # 报错原因：确定性索引只能包含两个元素，避免误解恢复状态。
                raise ValueError(f"Invalid deterministic dataset index: {index!r}")
            # 拆出语音下标与该轮该位置的裁切种子。
            item_index, crop_seed = int(index[0]), int(index[1])
        # 此分支：普通索引只给语音下标，裁切采用常规随机状态。
        else:
            # 普通索引只给语音下标，裁切采用常规随机状态。
            item_index = int(index)

        # 根据 DataLoader 给出的下标选中一条语音。
        item = self.items[item_index]

        # 缺文件时把具体路径写进异常，方便直接定位预处理问题。
        if not item.wav_path.is_file():
            # 报错原因：缺文件时把具体路径写进异常，方便直接定位预处理问题。
            raise FileNotFoundError(f"WAV not found: {item.wav_path}")

        # dtype=float32 让 soundfile 直接返回神经网络需要的浮点波形。
        waveform, sample_rate = sf.read(
            # 读取当前条目所指向的真实训练 WAV。
            str(item.wav_path),
            # 直接读为网络所需的单精度浮点波形。
            dtype="float32",
            # 单声道文件保持一维，双声道随后显式合并。
            always_2d=False,
        )

        # 极少数文件若是双声道，沿声道轴平均成单声道。
        if waveform.ndim == 2:
            # 对左右声道求平均，得到一维单声道波形。
            waveform = waveform.mean(axis=1)

        # 正式数据必须是 22050 Hz；这里不偷偷重采样，以免掩盖预处理错误。
        if int(sample_rate) != self.sampling_rate:
            # 报错原因：正式数据必须是 22050 Hz；这里不偷偷重采样，以免掩盖预处理错误。
            raise ValueError(
                # 错误信息定位预处理采样率不符的波形文件。
                f"Sample-rate mismatch for {item.wav_path}: "
                # 正式数据必须是 22050 Hz；这里不偷偷重采样，以免掩盖预处理错误。
                f"got {sample_rate}, expected {self.sampling_rate}"
            )

        # 波形最终必须是一维、有限值数组。
        if waveform.ndim != 1:
            # 报错原因：波形最终必须是一维、有限值数组。
            raise ValueError(f"Waveform must be mono 1-D: {item.wav_path}")
        # 输入波形中不能包含 NaN 或无穷大。
        if not np.isfinite(waveform).all():
            # 报错原因：波形最终必须是一维、有限值数组。
            raise ValueError(f"Waveform contains NaN/Inf: {item.wav_path}")

        # 官方 HiFi-GAN 的非 fine-tuning 数据路径先做 librosa peak normalize，
        # 再乘 0.95。归一化发生在整条 utterance 上，而不是每个裁切片段上。
        waveform = librosa.util.normalize(
            # 确保整句波形连续存储，再做整句峰值归一化。
            np.ascontiguousarray(waveform, dtype=np.float32)
        # 整句峰值归一化后缩到 0.95，沿用官方 HiFi-GAN scratch 数据幅度规则。
        ) * 0.95
        # 归一化后再次固定为连续 float32 数组。
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)

        # 有固定裁切种子时使用独立随机器，续训不受 worker 调度顺序影响。
        crop_rng = random.Random(crop_seed) if crop_seed is not None else random
        # 计算还能容纳一个完整片段的最晚起点。
        max_audio_start = waveform.shape[0] - self.segment_size
        # 语音够长时从有效范围内随机选取完整片段。
        if max_audio_start >= 0:
            # 与官方实现一样，起点以 waveform sample 为单位，不强制 hop 对齐。
            audio_start = crop_rng.randint(0, int(max_audio_start))
            # 截出固定采样点数的连续训练波形。
            waveform_segment = waveform[audio_start : audio_start + self.segment_size]
        # 此分支：若极短语音不足一个片段，从开头取并在右侧补 0。
        else:
            # 若极短语音不足一个片段，从开头取并在右侧补 0。
            waveform_segment = waveform[: self.segment_size]
            # 短语音计算右侧缺少的采样点数。
            wav_pad = self.segment_size - waveform_segment.shape[0]
            # 只有片段不足目标长度时补零。
            if wav_pad > 0:
                # 在短片段右边补零，所有批次保持同一波形长度。
                waveform_segment = np.pad(waveform_segment, (0, wav_pad))

        # 把裁切后的波形转换为声码器训练张量。
        waveform_tensor = torch.from_numpy(
            # 保证切片在内存中连续，方便张量转换。
            np.ascontiguousarray(waveform_segment)
        # 把当前训练片段固定为 float32 波形张量。
        ).float()
        # 设置与该波形片段对应的对数 mel。
        mel_tensor = mel_spectrogram_torch(
            # 增加单样本批次轴，再计算对应条件 mel。
            waveform_tensor.unsqueeze(0),
            # 设置短时傅里叶变换点数。
            n_fft=self.n_fft,
            # 设置mel 频带数量。
            num_mels=self.num_mels,
            # 设置每秒波形采样点数。
            sampling_rate=self.sampling_rate,
            # 设置相邻 mel 帧之间的波形采样点间隔。
            hop_size=self.hop_size,
            # 设置分析窗的采样点长度。
            win_size=self.win_size,
            # 设置mel 滤波器最低频率。
            fmin=self.fmin,
            # 设置mel 滤波器最高频率，空值表示奈奎斯特频率。
            fmax=self.fmax,
        # 单条条件 mel 计算完成后去掉临时批次轴，交给 DataLoader 统一堆叠。
        ).squeeze(0)

        # 最终形状是训练网络的硬约束；不满足就停止而不是让错误悄悄传播。
        if tuple(mel_tensor.shape) != (self.num_mels, self.frames_per_segment):
            # 报错原因：最终形状是训练网络的硬约束；不满足就停止而不是让错误悄悄传播。
            raise RuntimeError(f"Unexpected mel segment shape: {tuple(mel_tensor.shape)}")
        # 训练波形片段必须恰好达到固定采样点数。
        if tuple(waveform_tensor.shape) != (self.segment_size,):
            # 报错原因：最终形状是训练网络的硬约束；不满足就停止而不是让错误悄悄传播。
            raise RuntimeError(
                # 错误信息给出真实片段形状，便于查找裁切或补零问题。
                f"Unexpected waveform segment shape: {tuple(waveform_tensor.shape)}"
            )

        # key 只用于日志定位，不参与模型计算。
        key = f"{item.speaker}/{item.utt_id}"

        # DataLoader 会把 batch 中的多个 tensor 自动堆叠。
        return mel_tensor, waveform_tensor, key


# 核对训练波形与预提取 mel 是否存在，可选全量检查。
def check_item_files(items: Sequence[VocoderItem], full: bool = False) -> Tuple[int, int]:
    # 核对训练波形与预提取 mel 是否存在，可选全量检查。
    """检查文件存在性；默认抽查首尾各 16 条，``full=True`` 检查全部。"""

    # 全量检查用于正式开跑；快速检查用于日常启动。
    selected = list(items) if full else list(items[:16]) + list(items[-16:])

    # missing_wav/missing_mel 是缺失文件计数。
    missing_wav = sum(1 for item in selected if not item.wav_path.is_file())
    # 同时检查预处理 mel 文件存在；当前 scratch 训练条件 mel 会现场重算。
    missing_mel = sum(1 for item in selected if not item.mel_path.is_file())

    # 返回两个计数，由训练入口决定是否终止。
    return missing_wav, missing_mel
