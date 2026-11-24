#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 从 22.05 kHz 音频提取 80 维自然对数 mel，完整性校验后原子保存。
"""
Extract 22.05k / 80-dim log-mel spectrograms for FastVoiceGrad teacher.

Design choices:
- Read utterances from data/metadata/all_manifest.csv
- Extract mel only for rows actually used by train/test
- Use HiFi-GAN-style mel_spectrogram implementation
- Fixed protocol from FastVoiceGrad paper:
    sr=22050, n_fft=1024, hop=256, win=1024, n_mels=80
- Practical choice for vocoder-compatible mel band:
    fmin=0, fmax=8000
- Save each mel to:
    data/mel/<speaker>/<utt_id>.npy
- Resume-friendly by default; use --force to replace old mel after a protocol change

Example:
    python3 preprocess/03_extract_mel_22050.py --limit 20 --num_workers 2
    python3 preprocess/03_extract_mel_22050.py --num_workers 4
"""

# =============================================================================
# 预处理第 03 步：把波形转换成论文使用的 80 维 log-mel
# =============================================================================
# wav 是一长串随时间变化的振幅，神经网络使用的是 [80,T] mel 频谱。转换路线：
# 波形 -> 切短窗 -> STFT 分析每帧频率 -> 取复数频谱幅度 -> mel filter bank
# 压成 80 个人耳尺度频带 -> 取自然对数压缩动态范围 -> 保存 float32 .npy。
#
# 论文明确给出 sr=22050、FFT=1024、hop=256、window=1024、mel=80，并说明
# 预处理 following DiffVC。DiffVC 官方实现还会先把波形截到 hop 的整数倍；本脚本
# 同样执行该细节，避免最后一帧与论文所引用实现出现细小差异。
# fmin=0/fmax=8000 是本工程为兼容 22.05 kHz HiFi-GAN 使用的实现细节。
# 训练、推理和 vocoder 必须使用相同 mel 定义，否则数值分布不能互相兼容。

# --------------------------- 导入包逐项解释 ---------------------------
# csv 读取 all_manifest；json 写处理报告；argparse 读取命令行参数。
import csv
# 保存或读取实验设置、数据指纹及处理进度的结构化记录。
import json
# 定义终端参数，使数据目录、检查点和运行选项明确记录。
import argparse
# os.replace 把完整写好的临时 .npy 原子替换到正式路径，防止中断留下半个文件。
import os
# Path 处理路径；Dict/List/Tuple 是类型提示，不改变运行。
from pathlib import Path
# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Dict, List, Tuple
# ProcessPoolExecutor 管理多进程；as_completed 按任务完成顺序收结果。
from concurrent.futures import ProcessPoolExecutor, as_completed

# NumPy 做波形范围裁剪、dtype 转换和 .npy 保存。
import numpy as np
# soundfile 读取 wav 及真实采样率；librosa 生成 mel filter bank。
import soundfile as sf
# 构造 mel 滤波器或重采样音频，具体参数由下面的声学设置限定。
import librosa
# tqdm 显示特征提取进度条。
from tqdm import tqdm

# torch 执行 STFT 和矩阵运算；F.pad 做反射边界填充。
import torch
# 提供插值、补帧和损失等无需单独创建网络层的运算。
import torch.nn.functional as F


# 优先使用环境变量；否则从 preprocess/ 返回上一层得到仓库根目录。
ROOT = Path(
    # 优先读取显式工程根目录；没有环境变量时回到本文件向上两级的仓库目录。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[1])
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()
# 默认读取 all_manifest 中真正参与 train/test 的语音，并按 speaker 分目录保存。
DEFAULT_MANIFEST_PATH = ROOT / "data" / "metadata" / "all_manifest.csv"
# 自然对数 mel 保存到 data/mel；标准化留到数据读取时执行。
DEFAULT_OUT_DIR = ROOT / "data" / "mel"
# mel 提取的参数、计数及失败示例保存到实验日志目录。
REPORT_DIR = ROOT / "exp" / "logs"

# 下面这些数字共同定义 mel 算法；任何一个数字改变都会产生另一种特征。
SR = 22050
# 每个时窗的 FFT 点数。
N_FFT = 1024
# 相邻 mel 窗口的样本间距。
HOP_SIZE = 256
# 每次频谱分析的波形窗口长度。
WIN_SIZE = 1024
# mel 频带数，本实验为 80。
N_MELS = 80
# mel 滤波器最低频率。
FMIN = 0
# mel 滤波器最高频率，本实验兼容选择为 8000 Hz。
FMAX = 8000

# log(0) 是负无穷，所以取 log 前先把最小幅度限制为 1e-5。
CLIP_VAL = 1e-5

# Per-process cache
# mel basis 和 Hann window 对同一配置永远一样。每个 worker 缓存后只创建一次，
# 不必为四万条语音重复计算。多进程内存独立，所以称 per-process cache。
mel_basis_cache = {}
# 缓存不同窗口长度与设备对应的 Hann 窗，避免每条语音重新生成。
hann_window_cache = {}


# 对幅度 mel 取自然对数并保护零值；下面的参数分别限定输入数据和处理约束。
def dynamic_range_compression_torch(x: torch.Tensor, clip_val: float = CLIP_VAL) -> torch.Tensor:
    # clamp 把小于 clip_val 的值抬高，再取自然对数。log 压缩极大/极小幅度差，
    # 更接近人耳对响度的感知，也让网络更容易训练。
    return torch.log(torch.clamp(x, min=clip_val))


# 执行 HiFi-GAN 式对数压缩，未做训练集 zscore；下面的参数分别限定输入数据和处理约束。
def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    # 这里沿用 HiFi-GAN 的函数名；实际只是 log compression，不是 mean/std
    # 标准化。训练集 mean/std 会由 04 脚本计算并在 dataset.py 中应用。
    return dynamic_range_compression_torch(magnitudes)


# 按 22.05 kHz 与既定 STFT 参数计算 80 维自然对数 mel；下面的参数分别限定输入数据和处理约束。
def mel_spectrogram_torch(
    # 待变换的单声道浮点波形。
    y: torch.Tensor,
    # 每个时窗的 FFT 点数。
    n_fft: int,
    # 输出 mel 频带数量。
    num_mels: int,
    # 波形每秒采样点数。
    sampling_rate: int,
    # 相邻 mel 窗口的样本间距。
    hop_size: int,
    # 每次频谱分析的波形窗口长度。
    win_size: int,
    # mel 滤波器最低频率。
    fmin: int,
    # mel 滤波器最高频率，本实验兼容选择为 8000 Hz。
    fmax: int,
# 返回按 22.05 kHz 与既定 STFT 参数计算 80 维自然对数 mel得到的张量；类型标注帮助调用方理解结果，不改变数值。
) -> torch.Tensor:
    # 按 22.05 kHz 与既定 STFT 参数计算 80 维自然对数 mel。
    """
    HiFi-GAN-style mel extraction.
    Input:
        y: shape [T] or [1, T], float32 waveform in [-1, 1]
    Output:
        mel: shape [1, n_mels, frames]
    """
    # 单条波形通常是 [samples]；unsqueeze(0) 增加 batch 轴成为 [1,samples]。
    if y.dim() == 1:
        # 将一维波形临时增加批次轴，形成 [1,N] 以便统一执行 STFT。
        y = y.unsqueeze(0)

    # 该提取函数一次只处理一条语音，拒绝把多个样本当作声道混合。
    if y.size(0) != 1:
        # 按 22.05 kHz 与既定 STFT 参数计算 80 维自然对数 mel未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Expected mono waveform with batch=1, but got shape {tuple(y.shape)}")

    # 张量执行计算的设备。
    device = y.device
    # 缓存键包含所有影响结果的配置和设备。CPU 上的 mel basis 不能直接与 GPU
    # spectrogram 相乘，所以 device 也必须写入 key。
    basis_key = f"{sampling_rate}_{n_fft}_{num_mels}_{fmin}_{fmax}_{device}"
    # 由窗口长度和设备组成的 Hann 窗缓存键。
    window_key = f"{win_size}_{device}"

    # 按由采样率、频率范围等组成的 mel 滤波器缓存键决定是否进入此分支；判断条件为 basis_key not in mel_basis_cache。
    if basis_key not in mel_basis_cache:
        # librosa 生成 [80,n_fft/2+1] mel 滤波器矩阵，每一行描述一个 mel 频带
        # 如何加权线性 FFT bins。
        mel = librosa.filters.mel(
            # 波形采样率。
            sr=sampling_rate,
            # 每个时窗的 FFT 点数。
            n_fft=n_fft,
            # mel 频带数，本实验为 80。
            n_mels=num_mels,
            # mel 滤波器最低频率。
            fmin=fmin,
            # mel 滤波器最高频率，本实验兼容选择为 8000 Hz。
            fmax=fmax,
        )
        # 把与当前采样率及频带范围匹配的 mel 滤波器矩阵转成单精度并缓存到计算设备。
        mel_basis_cache[basis_key] = torch.from_numpy(mel).float().to(device)

    # 按由窗口长度和设备组成的 Hann 窗缓存键决定是否进入此分支；判断条件为 window_key not in hann_window_cache。
    if window_key not in hann_window_cache:
        # Hann window 把短窗两端平滑压低，减少硬切片产生的频谱泄漏。
        hann_window_cache[window_key] = torch.hann_window(win_size).to(device)

    # HiFi-GAN 风格在 STFT 前手动 reflect padding。镜像边界比直接补 0 更不容易
    # 在语音开头/结尾制造突然跳变。
    pad_amt = int((n_fft - hop_size) / 2)
    # 在波形两端各反射补 (n_fft−hop_size)/2 个采样点，再去掉临时声道轴；这个规则决定 mel 首帧的物理中心。
    y = F.pad(y.unsqueeze(1), (pad_amt, pad_amt), mode="reflect").squeeze(1)

    # STFT（短时傅里叶变换）按 window/hop 把长波形切成重叠短帧并分析频率。
    # center=False 是因为上面已经按照 HiFi-GAN 规则手动 padding。
    spec = torch.stft(
        # 待变换的单声道浮点波形。
        y,
        # 每个时窗的 FFT 点数。
        n_fft=n_fft,
        # 相邻频谱窗口向前移动的样本数。
        hop_length=hop_size,
        # 频谱分析窗的采样点长度。
        win_length=win_size,
        # 使用已缓存的 Hann 窗，减轻相邻分析窗边界的不连续。
        window=hann_window_cache[window_key],
        # STFT 不再自动做居中补零；前面已按 HiFi-GAN 规则手动反射补边。
        center=False,
        # 声明反射补边模式；center=False 时实际边缘主要由前面的手动 padding 决定。
        pad_mode="reflect",
        # 不对 FFT 结果额外除以窗长，保持与声码器所用幅度定义一致。
        normalized=False,
        # 实值语音只保留非负频率一侧，避免共轭对称的重复频点。
        onesided=True,
        # 沿用当前实现的实部/虚部分离输出，随后显式计算幅度谱。
        return_complex=False,
    )

    # return_complex=False 时最后一轴的两个数是复数实部/虚部。
    # 幅度=sqrt(real^2+imag^2)；加 1e-9 防止 0 附近数值问题。
    # spec: [B, F, T, 2]
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)  # [B, F, T]

    # 矩阵乘法 [80,F] @ [B,F,T] -> [B,80,T]，把线性频率压为 mel 频带。
    mel = torch.matmul(mel_basis_cache[basis_key], spec)  # [B, n_mels, T]
    # 80 维自然对数 mel 声谱图。
    mel = spectral_normalize_torch(mel)
    # 返回80 维自然对数 mel 声谱图，供调用方继续处理。
    return mel


# 读取此次真正需要提取特征的清单并控制小样本检查范围；下面的参数分别限定输入数据和处理约束。
def load_manifest_rows(manifest_path: Path, limit: int = -1) -> List[Dict[str, str]]:
    # seen 集合用 (speaker,utt_id) 去重；同一语音只提取一次。
    if not manifest_path.exists():
        # 读取此次真正需要提取特征的清单并控制小样本检查范围未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # 本次选定的语音清单。
    rows = []
    # 已经读过的语音身份集合。
    seen = set()

    # 在受控文件句柄中完成读取此次真正需要提取特征的清单并控制小样本检查范围，退出此段后自动关闭文件。
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        # 按表头返回字典的 CSV 读取器。
        reader = csv.DictReader(f)
        # 逐项遍历按表头返回字典的 CSV 读取器，保持每条记录都经过相同检查。
        for row in reader:
            # 当前语音的唯一身份。
            key = (row["speaker"], row["utt_id"])
            # 按当前语音的唯一身份决定是否进入此分支；判断条件为 key in seen。
            if key in seen:
                # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
                continue
            # 登记当前语音身份，后续再次出现时可识别为重复。
            seen.add(key)
            # 将去重并核对必要字段后的语音加入本次 mel 提取清单。
            rows.append(row)

    # 按仅为诊断而限制处理的样本数决定是否进入此分支；判断条件为 limit is not None and limit > 0。
    if limit is not None and limit > 0:
        # --limit 20 可先验证少量语音；默认 -1 不截断，正式处理全量。
        rows = rows[:limit]

    # 返回本次选定的语音清单，供调用方继续处理。
    return rows


# 为一条音频计算并原子保存 mel 特征；下面的参数分别限定输入数据和处理约束。
def extract_one(
    # 清单中的当前一条语音记录。
    row: Dict[str, str], out_dir_str: str, force: bool = False
# 返回为一条音频计算并原子保存 mel 特征整理的 Tuple[str, str, str] 结果，内部各项保持前面约定的对应关系。
) -> Tuple[str, str, str]:
    # 为一条音频计算并原子保存 mel 特征。
    """
    Returns:
        (status, key, message)
    status in {"ok", "skip", "fail"}
    """
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 外层已经启动多个进程。若每个进程内部又开很多 Torch 线程会过度并行，
        # 因此每个 worker 限制为一个 Torch CPU 线程。
        torch.set_num_threads(1)

        # 特征输出目录。
        out_dir = Path(out_dir_str)

        # 当前语音所属说话人。
        speaker = row["speaker"]
        # 同时包含说话人和句号的样本编号。
        utt_id = row["utt_id"]
        # 当前语音的音频文件。
        wav_path = Path(row["wav22k_path"])
        # 当前样本的目标特征文件。
        out_path = out_dir / speaker / f"{utt_id}.npy"

        # 当前语音的唯一身份。
        key = f"{speaker}/{utt_id}"

        # 默认目标已存在就 skip，支持服务器任务中断后继续跑；--force=True 时会重算，
        # 用于把旧版“未截到 hop 整数倍”的 mel 全部替换成 DiffVC 对齐版本。
        if out_path.exists() and not force:
        # 此处只按文件存在跳过，不校验已有数组；正式来源审核另要求全量重新生成且 skip=0。
            return ("skip", key, str(out_path))

        # 检查当前语音的音频文件是否缺失；缺失时不能继续信任该输入。
        if not wav_path.exists():
            # 源 WAV 缺失时记录失败，不能用空数组补充数据。
            return ("fail", key, f"wav not found: {wav_path}")

        # soundfile 单声道通常读成 [samples]，多声道读成 [samples,channels]。
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

        # 读取到多声道波形时，进入合并为单声道的分支。
        if wav.ndim == 2:
            # 多声道沿 channel 轴求平均得到单声道。
            wav = wav.mean(axis=1)

        # 按波形采样率决定是否进入此分支；判断条件为 sr != SR。
        if sr != SR:
            # 采样率不符时停止当前语音提取，避免在错误时轴上计算 mel。
            return ("fail", key, f"sample rate mismatch: got {sr}, expected {SR}")

        # 统一声道后必须是单条一维波形。
        if wav.ndim != 1:
            # 波形维数异常时记录具体维度，留给数据排查。
            return ("fail", key, f"waveform ndim error: got {wav.ndim}")

        # 语音不足一个分析窗时拒绝提取，避免反射 padding 无法满足要求。
        if len(wav) < WIN_SIZE:
            # 报告过短语音的真实采样点数。
            return ("fail", key, f"waveform too short: {len(wav)} samples")

        # 浮点波形标准范围是 [-1,1]；clip 清除异常峰值，astype 确保 float32。
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        # 论文写“Following DiffVC”。DiffVC 官方 mel 实现先执行下面的整数除法与切片：
        # 例如长度 1003×256+17，会只保留前 1003×256 个采样点，丢掉不足一个 hop
        # 的最后 17 点。这样 STFT 的帧边界与它完全一致；最多只丢 255 个采样点。
        usable_samples = (len(wav) // HOP_SIZE) * HOP_SIZE
        # 读取后的单声道浮点音频。
        wav = wav[:usable_samples]
        # 送入 PyTorch 频谱计算的音频张量。
        wav_t = torch.from_numpy(wav)

        # 80 维自然对数 mel 声谱图。
        mel = mel_spectrogram_torch(
            # 送入 PyTorch 频谱计算的音频张量。
            wav_t,
            # 每个时窗的 FFT 点数。
            n_fft=N_FFT,
            # 输出 mel 频带数量。
            num_mels=N_MELS,
            # 波形每秒采样点数。
            sampling_rate=SR,
            # 相邻 mel 窗口的样本间距。
            hop_size=HOP_SIZE,
            # 每次频谱分析的波形窗口长度。
            win_size=WIN_SIZE,
            # mel 滤波器最低频率。
            fmin=FMIN,
            # mel 滤波器最高频率，本实验兼容选择为 8000 Hz。
            fmax=FMAX,
        )

        # 去掉 batch=1 轴并转回 NumPy，磁盘统一保存 [80,T] float32。
        mel = mel.squeeze(0).cpu().numpy().astype(np.float32)  # [80, T]

        # 最终 mel 必须为通道和时间构成的二维矩阵。
        if mel.ndim != 2:
            # 记录生成结果的维数异常，阻止保存不兼容特征。
            return ("fail", key, f"mel ndim error: got {mel.ndim}")
        # 要求通道优先排列且正好 80 个 mel 频带。
        if mel.shape[0] != N_MELS:
            # 将错误的通道数写入失败原因，避免后续网络收到错维输入。
            return ("fail", key, f"mel channel error: got {mel.shape[0]}, expected {N_MELS}")
        # 时间轴至少要有一帧，空 mel 不能参加训练。
        if mel.shape[1] <= 0:
            # 报告无效的 mel 时间长度。
            return ("fail", key, f"mel time length error: got {mel.shape[1]}")

        # 准备为一条音频计算并原子保存 mel 特征所需的目标目录，已有目录按给定选项复用。
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # .npy 会保存 shape/dtype，读取快；allow_pickle=False 只允许纯数值数组。
        # 先写同目录临时文件，确认 np.save 完成后再原子替换，强制重提中途断电也不会
        # 把原来完整的 mel 覆盖成半个损坏文件。
        temporary = out_path.with_suffix(out_path.suffix + ".tmp")
        # 在受控文件句柄中完成为一条音频计算并原子保存 mel 特征，退出此段后自动关闭文件。
        with temporary.open("wb") as handle:
            # 将检查后的 float32 自然对数 mel 写入临时 NPY，禁止任意对象序列化。
            np.save(handle, mel, allow_pickle=False)
        # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
        os.replace(temporary, out_path)

        # 原子发布完成后返回成功及实际通道×帧数。
        return ("ok", key, f"{mel.shape[0]}x{mel.shape[1]}")

    # 捕获为一条音频计算并原子保存 mel 特征中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as e:
        # 单条失败不让 worker 进程直接消失，而是把状态和错误文字返回主进程汇总。
        speaker = row.get("speaker", "UNK")
        # 同时包含说话人和句号的样本编号。
        utt_id = row.get("utt_id", "UNK")
        # 当前语音的唯一身份。
        key = f"{speaker}/{utt_id}"
        # 将当前语音提取或保存异常带回主流程，不把失败算作完成。
        return ("fail", key, repr(e))


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main():
    # --num_workers 控制并行进程；--manifest/--out_dir 可覆盖服务器默认路径。
    parser = argparse.ArgumentParser()
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--num_workers", type=int, default=4)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--limit", type=int, default=-1, help="For sanity check. Use -1 for full run.")
    # action="store_true"：命令行出现 --force 时为 True，不出现时为 False。
    parser.add_argument(
        # 显式要求重算已有特征；普通运行先尝试验证后复用。
        "--force",
        # 该文件本轮为新生成还是经过验证后复用。
        action="store_true",
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="Recompute and atomically replace existing mel files.",
    )
    # 终端传入并经解析的运行设置。
    args = parser.parse_args()

    # 本次实际使用的清单文件。
    manifest_path = Path(args.manifest)
    # 特征输出目录。
    out_dir = Path(args.out_dir)

    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    out_dir.mkdir(parents=True, exist_ok=True)
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 本次选定的语音清单。
    rows = load_manifest_rows(manifest_path, limit=args.limit)
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Manifest = {manifest_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Out dir  = {out_dir}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Total rows to process = {len(rows)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] num_workers = {args.num_workers}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] force overwrite = {args.force}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(
        # 日志记录实际采样率、FFT 点数和帧移。
        f"[INFO] Mel config: sr={SR}, n_fft={N_FFT}, hop={HOP_SIZE}, "
        # 继续记录窗长、mel 通道数和频率上下限，保留真实提取配置。
        f"win={WIN_SIZE}, n_mels={N_MELS}, fmin={FMIN}, fmax={FMAX}"
    )

    # 累计本轮新提取成功的 mel 数量。
    ok_count = 0
    # 累计已存在且检查通过的 mel 数量。
    skip_count = 0
    # 累计本轮无法生成有效 mel 的语音数量。
    fail_count = 0
    # 仅保存少量具体失败示例，避免报告被大量重复错误撑大。
    fail_examples = []

    # worker<=1 用普通循环，调试最直观；否则使用进程池并行提取。
    if args.num_workers <= 1:
        # 单进程模式按清单逐条提取，并显示真实完成进度。
        for row in tqdm(rows, total=len(rows), desc="Extract mel"):
            # 提取一条语音，得到状态、语音唯一键和形状或错误说明。
            status, key, msg = extract_one(row, str(out_dir), args.force)
            # 按本轮处理完成、部分完成或失败的状态决定是否进入此分支；判断条件为 status == "ok"。
            if status == "ok":
                # 一条新 mel 成功保存后增加完成计数。
                ok_count += 1
            # 按本轮处理完成、部分完成或失败的状态决定是否进入此分支；判断条件为 status == "skip"。
            elif status == "skip":
                # 一条已有 mel 被确认可用后增加复用计数。
                skip_count += 1
            # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
            else:
                # 一条语音处理失败后增加失败计数。
                fail_count += 1
                # 报告最多保存 20 个失败示例；总失败数仍完整累计。
                if len(fail_examples) < 20:
                    # 把语音身份与错误原因一起记下，方便准确重试。
                    fail_examples.append({"key": key, "error": msg})
    # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
    else:
        # submit 提交每条任务；as_completed 按完成先后返回，不等待原 CSV 顺序。
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            # 为并行提取创建任务列表，每条语音对应一个返回结果对象。
            futures = [
                # 逐条提交与单进程相同的 mel 提取函数，保持数值处理流程一致。
                ex.submit(extract_one, row, str(out_dir), args.force) for row in rows
            ]
            # 按子进程实际完成顺序回收结果并更新进度。
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extract mel"):
                # 取得当前已完成任务的状态、语音身份和详细说明。
                status, key, msg = fut.result()
                # 按本轮处理完成、部分完成或失败的状态决定是否进入此分支；判断条件为 status == "ok"。
                if status == "ok":
                    # 一条新 mel 成功保存后增加完成计数。
                    ok_count += 1
                # 按本轮处理完成、部分完成或失败的状态决定是否进入此分支；判断条件为 status == "skip"。
                elif status == "skip":
                    # 一条已有 mel 被确认可用后增加复用计数。
                    skip_count += 1
                # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
                else:
                    # 一条语音处理失败后增加失败计数。
                    fail_count += 1
                    # 报告最多保存 20 个失败示例；总失败数仍完整累计。
                    if len(fail_examples) < 20:
                        # 把语音身份与错误原因一起记下，方便准确重试。
                        fail_examples.append({"key": key, "error": msg})

    # JSON 报告永久记录特征参数及成功/跳过/失败数量，证明预处理是否完整。
    report = {
        # 记录待读取的样本清单。
        "manifest": str(manifest_path),
        # 记录特征输出目录。
        "out_dir": str(out_dir),
        # 记录本次计划处理的语音总数。
        "total_rows": len(rows),
        # 记录本轮新生成的有效 mel 数量。
        "ok": ok_count,
        # 记录从编码器直接传给解码器的同尺度细节。
        "skip": skip_count,
        # 记录失败数量，不能用成功加复用掩盖缺失数据。
        "fail": fail_count,
        # 记录波形采样率。
        "sr": SR,
        # 记录每个时窗的 FFT 点数。
        "n_fft": N_FFT,
        # 记录相邻 mel 窗口的样本间距。
        "hop_size": HOP_SIZE,
        # 记录每次频谱分析的波形窗口长度。
        "win_size": WIN_SIZE,
        # 记录mel 频带数，本实验为 80。
        "n_mels": N_MELS,
        # 记录mel 滤波器最低频率。
        "fmin": FMIN,
        # 记录mel 滤波器最高频率，本实验兼容选择为 8000 Hz。
        "fmax": FMAX,
        # 记录波形先截为 256 采样点整数倍的规则；这是与现有声码器兼容的实现选择。
        "truncate_waveform_to_hop_multiple": True,
        # 记录是否显式要求重算已有文件，以便区分复用和全新提取。
        "force_overwrite": args.force,
        # 附上有语音身份的失败示例供追查。
        "fail_examples": fail_examples,
    }

    # 数据处理审计报告的保存位置。
    report_path = REPORT_DIR / "mel_extraction_report_22050.json"
    # 在受控文件句柄中完成串联本文件的数据检查、模型处理和结果保存流程，退出此段后自动关闭文件。
    with report_path.open("w", encoding="utf-8") as f:
        # ensure_ascii=False 保留可读文字；indent=2 让 JSON 分行缩进。
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("\n[DONE] Mel extraction finished.")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  ok   = {ok_count}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  skip = {skip_count}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  fail = {fail_count}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  report = {report_path}")

    # 存在失败示例时在终端额外提示，便于启动后立即发现问题。
    if fail_examples:
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print("\n[FAIL EXAMPLES]")
        # 终端只展示前 10 个失败示例，完整计数保留在报告。
        for item in fail_examples[:10]:
            # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
            print(f"  {item['key']} -> {item['error']}")

    # 任意失败都返回非零退出码，阻止后续脚本从残缺 mel 计算统计量。
    if fail_count:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise SystemExit(
            # 报错说明 mel 提取未全量完成及详细报告位置。
            f"Mel extraction failed for {fail_count} utterances; see {report_path}."
        )


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 直接运行才开始全量提取；被其他 Python 文件 import 时不会自动执行。
    main()
