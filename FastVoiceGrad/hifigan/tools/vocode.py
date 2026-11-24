#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 此工具将既有 mel 转成标准 WAV，记录每条状态与波形统计，正式指标由评价脚本计算。
"""把 VoiceGrad-30 生成的 log-mel 通过 VCTK HiFi-GAN-V1 转换成 WAV。"""

# =============================================================================
# 这个文件解决什么问题？
# =============================================================================
# VoiceGrad-30 的输出不是人耳能直接播放的声音，而是一张形状通常为 [80, T] 的
# log-mel 频谱图。80 表示每一帧有 80 个 mel 频带，T 表示一共有多少个时间帧。
# HiFi-GAN 的 Generator（生成器）负责把这张频谱图还原成一长串波形采样点，最后
# 再把这些采样点保存成 22,050 Hz、单声道、16-bit PCM 的 WAV 文件。
#
# 项目自带的 hifigan/official/inference.py 不能直接完成我们的任务：它接收 WAV，先从
# WAV 重新提取 mel，再把这个 mel 送入声码器。我们需要的是“已经生成好的 .npy mel
# -> WAV”，所以必须提供这个独立入口。
#
# 这个脚本同时支持两种工作方式：
#   1. 单文件模式：用于现在这条 p225_001 -> p228_002 冒烟检查；
#   2. 目录模式：用于以后把 epoch 500 的 8,100 个 mel 一次性转换成 WAV。
#
# 默认采用公开 VCTK HiFi-GAN-V1；调用者也可明确指定自训 G。FastVoiceGrad 披露了
# 冻结的预训练 G 及 MPD+MRD 判别器，但未公布作者实际 G 检查点和独立声码器训练方案。
# 因此具体权重和训练方式需在实验记录中单独说明，不能仅凭结构相同声称完全同条件。

# ------------------------------- 导入标准库 --------------------------------
# argparse 用来读取终端里的 --input-mel、--output-wav 等命令行参数。
import argparse
# hashlib 用于计算配置和声码器检查点的 SHA-256；相同哈希代表文件内容完全相同。
import hashlib
# json 用来读取 HiFi-GAN 的 config.json，也用来写出机器可读的处理报告。
import json
# os.replace 能把完整写好的临时 WAV 原子替换成正式文件，避免中断留下半个 WAV。
import os
# sys.path 允许本工具从隔离的 hifigan/official 子目录导入官方模型代码。
import sys
# dataclasses.asdict 可以把下面的结果对象转换成普通字典，方便写入 JSON。
from dataclasses import asdict, dataclass
# Path 比手工拼字符串更安全地处理文件、目录、扩展名和绝对路径。
from pathlib import Path
# Dict、List、Optional、Tuple 只是类型提示，帮助读者和编辑器理解变量中装什么。
from typing import Dict, List, Optional, Tuple

# ------------------------------- 导入第三方库 -------------------------------
# NumPy 负责读取 .npy、检查 NaN/Inf、计算波形统计量和整理 float32 数据。
import numpy as np
# soundfile 负责把浮点波形写成标准 PCM_16 WAV，并重新读取文件头进行验证。
import soundfile as sf
# torch 负责建立 HiFi-GAN、加载神经网络权重，以及在 GPU 上执行声码器推理。
import torch
# tqdm 给批量处理显示进度条；单文件模式也复用同一套处理循环。
from tqdm import tqdm

# 复用 config.py 的可移植工程根目录。用户既可设置 VOICEGRAD_PROJECT_ROOT，
# 也可在命令行传 --project-root；代码不再依赖某台服务器的私人绝对路径。
from voicegrad.config import PROJECT_ROOT


# ------------------------------- 固定工程默认值 -----------------------------
# 保留旧变量名只是为了少改下方参数定义；它现在等于可移植的 PROJECT_ROOT。
SERVER_PROJECT_ROOT = PROJECT_ROOT
# 论文和本工程共同使用 80 维 log-mel；任何其他通道数都直接拒绝处理。
EXPECTED_MEL_CHANNELS = 80
# VCTK 版实验统一使用 22.05 kHz；写出的 WAV 也必须严格是这个采样率。
EXPECTED_SAMPLE_RATE = 22050
# 每个 mel 帧对应 256 个波形采样点，也是 HiFi-GAN 上采样率乘积 8*8*2*2。
EXPECTED_HOP_SIZE = 256


# ------------------------------- 单文件处理结果 -----------------------------
@dataclass
# 保存成功、跳过或失败的单条声码器处理结果。
class VocodeResult:
    # 每条结果同时保留文件身份、成功状态、长度和幅度统计，异常也有明确记录。
    """保存一条 mel 转 WAV 的结果；最后会原样写进 JSON 报告。"""

    # status 只有 ok、skip、fail 三种：成功、已有文件而跳过、发生错误。
    status: str
    # input_mel 是输入 .npy 的绝对路径，方便以后追踪这条 WAV 来自哪里。
    input_mel: str
    # output_wav 是输出 WAV 的绝对路径。
    output_wav: str
    # mel_shape 记录整理后的 [80, T]；失败或跳过时可以为空。
    mel_shape: Optional[List[int]] = None
    # samples 是最终波形采样点数量；正常情况下等于 T*256。
    samples: Optional[int] = None
    # duration_seconds 是 samples/22050，表示音频时长（秒）。
    duration_seconds: Optional[float] = None
    # peak 是最大绝对振幅；接近 1 表示靠近 PCM 可表示上限。
    peak: Optional[float] = None
    # rms 是均方根响度；非常接近 0 通常表示生成了静音。
    rms: Optional[float] = None
    # near_clipping_fraction 是 |波形|>=0.999 的采样点比例，用来检查严重削波。
    near_clipping_fraction: Optional[float] = None
    # message 在失败时保存错误原因，在跳过时说明文件已经存在。
    message: str = ""


# ------------------------------- 小型通用函数 -------------------------------
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # 按二进制文件内容计算 SHA-256，供后续检查是否混入其他版本。
    """分块计算文件 SHA-256；分块读取避免一次把大检查点全部塞进内存。"""

    # sha256() 创建一个空的哈希计算器，后面不断把文件字节喂给它。
    digest = hashlib.sha256()
    # rb 表示二进制只读；检查点不是文本，不能用普通字符串方式读取。
    with path.open("rb") as handle:
        # while True 表示持续读取，直到 read 返回空字节串才结束。
        while True:
            # 每次最多读 1 MiB，既节省内存，也适合几十 MiB 的声码器检查点。
            block = handle.read(chunk_size)
            # 空字节串说明已经读到文件末尾。
            if not block:
                # break 立即结束当前 while 循环。
                break
            # update 把这一块内容累加进 SHA-256 计算。
            digest.update(block)
    # hexdigest 把二进制哈希转成常见的 64 位十六进制字符串。
    return digest.hexdigest()


# 读取声码器配置并检查其根节点类型。
def load_json(path: Path) -> Dict[str, object]:
    # 声码器配置需要是 UTF-8 JSON 键值对象，缺文件或类型不符会直接报错。
    """读取 UTF-8 JSON，并保证最外层确实是键值字典。"""

    # 文件不存在时主动给出清楚错误，而不是等 open 抛出难懂的下游异常。
    if not path.is_file():
        # FileNotFoundError 明确告诉使用者缺少的是哪个配置文件。
        raise FileNotFoundError(f"JSON file not found: {path}")
    # utf-8 能正确读取普通英文配置，也不会依赖服务器系统默认编码。
    with path.open("r", encoding="utf-8") as handle:
        # json.load 把 JSON 文本变成 Python 对象。
        value = json.load(handle)
    # HiFi-GAN 配置最外层必须是 {键: 值}，也就是 Python 的 dict。
    if not isinstance(value, dict):
        # TypeError 表示文件存在，但内容类型不符合程序接口约定。
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    # 返回已验证的配置字典。
    return value


# 核对 mel 的物理尺度和波形输出步幅，避免特征接口错配。
def validate_hifigan_config(config: Dict[str, object], config_path: Path) -> None:
    # 核对 mel 的采样率、频带、窗长和上采样倍率，避免声码器接口错配。
    """阻止不匹配的 HiFi-GAN 配置悄悄生成不可比较的音频。"""

    # 这些键共同决定 mel 的物理含义，必须与 preprocess/03 完全相同。
    expected = {
        # 记录mel 频带数量。
        "num_mels": EXPECTED_MEL_CHANNELS,
        # 记录每秒波形采样点数。
        "sampling_rate": EXPECTED_SAMPLE_RATE,
        # 记录短时傅里叶变换点数。
        "n_fft": 1024,
        # 记录相邻 mel 帧之间的波形采样点间隔。
        "hop_size": EXPECTED_HOP_SIZE,
        # 记录分析窗的采样点长度。
        "win_size": 1024,
        # 记录mel 滤波器最低频率。
        "fmin": 0,
        # 记录mel 滤波器最高频率，空值表示奈奎斯特频率。
        "fmax": 8000,
    }
    # 逐项比较实际值与实验固定值；任何一项不一致都停止，而不是勉强运行。
    for key, expected_value in expected.items():
        # dict.get 在键不存在时返回 None，也会触发下面的不匹配错误。
        actual_value = config.get(key)
        # != 是“不等于”；这里只允许完全一致的数字。
        if actual_value != expected_value:
            # RuntimeError 同时写出配置路径、键、实际值和期望值，方便马上定位。
            raise RuntimeError(
                # 错误信息定位声码器配置文件，再逐项报告 mel 接口差异。
                f"HiFi-GAN config mismatch at {config_path}: "
                # RuntimeError 同时写出配置路径、键、实际值和期望值，方便马上定位。
                f"{key}={actual_value!r}, expected {expected_value!r}."
            )

    # 上采样率乘积决定一帧 mel 生成多少波形点，正常必须等于 hop_size=256。
    rates = config.get("upsample_rates")
    # 先保证 rates 是列表，而且每一项都是可以转成整数的上采样倍数。
    if not isinstance(rates, list) or not rates:
        # 配置缺少上采样表时不能可靠判断输出长度，因此直接拒绝。
        raise RuntimeError(f"Invalid upsample_rates in {config_path}: {rates!r}")
    # np.prod 计算 8*8*2*2；int 把 NumPy 整数转成普通 Python 整数。
    upsample_product = int(np.prod([int(rate) for rate in rates]))
    # 乘积不等于 hop 时，输出时长会和输入 mel 时长不对应。
    if upsample_product != EXPECTED_HOP_SIZE:
        # 报错中列出实际乘积，避免使用者只看到一个模糊的“形状错误”。
        raise RuntimeError(
            # 报错中列出实际乘积，避免使用者只看到一个模糊的“形状错误”。
            f"HiFi-GAN upsample product is {upsample_product}, "
            # 报错中列出实际乘积，避免使用者只看到一个模糊的“形状错误”。
            f"expected hop size {EXPECTED_HOP_SIZE}."
        )


# 从指定项目的官方源码目录取得声码器生成器定义。
def import_hifigan(project_root: Path):
    # 只从指定项目内的官方目录取得 G 定义，保持权重与实现对应。
    """从项目内 hifigan/official 目录导入官方 AttrDict 和 Generator。"""

    # 使用动态导入可以保持官方源码内部的 ``from utils import ...`` 不必大改。
    hifigan_dir = (project_root / "hifigan" / "official").resolve()
    # 如果目录不存在，说明服务器没有上传完整依赖，继续运行也只会更晚报错。
    if not hifigan_dir.is_dir():
        # FileNotFoundError 给出程序实际寻找的绝对目录。
        raise FileNotFoundError(f"HiFi-GAN source directory not found: {hifigan_dir}")
    # 把目录放到模块搜索路径最前面，models.py 才能继续导入同目录的 utils.py。
    if str(hifigan_dir) not in sys.path:
        # insert(0, ...) 比 append 优先级更高，避免意外导入系统里另一个同名 models。
        sys.path.insert(0, str(hifigan_dir))
    # AttrDict 让 config["x"] 同时可以写成 config.x，符合官方模型代码预期。
    from env import AttrDict  # type: ignore
    # Generator 就是把 [B,80,T] mel 变成 [B,1,T*256] 波形的神经网络。
    from models import Generator  # type: ignore

    # 返回类对象，而不是在这里直接建模型，使加载流程仍然清楚可检查。
    return AttrDict, Generator


# 建立 G、严格加载权重并冻结参数，准备纯推理。
def load_generator(
    # 输入参数：租用服务器上的项目根目录。
    project_root: Path,
    # 输入参数：模型权重路径。
    checkpoint_path: Path,
    # 输入参数：模型配置路径。
    config_path: Path,
    # 输入参数：张量和模型使用的计算设备。
    device: torch.device,
# 返回冻结的 G 网络对象和核对过的完整声码器配置。
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    # 检查配置后严格加载 G，并设为冻结推理状态；输入只接受可信来源权重。
    """建立 HiFi-GAN Generator、加载权重并切换到只推理状态。"""

    # 先检查配置，再建立网络；错误配置不应该占用 GPU 显存。
    config = load_json(config_path)
    # 严格核对 mel 和采样率相关参数。
    validate_hifigan_config(config, config_path)
    # 动态导入项目附带的官方 HiFi-GAN 网络类。
    AttrDict, Generator = import_hifigan(project_root)
    # AttrDict 只是给普通字典增加“点号取值”能力，不会修改配置数值。
    h = AttrDict(config)
    # 使用 config.json 建立与 generator_v1 权重完全同构的网络，然后搬到目标设备。
    generator = Generator(h).to(device)

    # 检查点必须是实际文件；目录或拼错路径都在这里立即停止。
    if not checkpoint_path.is_file():
        # FileNotFoundError 会把缺失路径直接显示在终端。
        raise FileNotFoundError(f"HiFi-GAN checkpoint not found: {checkpoint_path}")
    # PyTorch 2.6 默认 weights_only 行为发生过变化；这里显式写 False 兼容官方旧格式。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 官方检查点最外层是字典，真正的生成器参数放在 generator 键下面。
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        # KeyError 说明拿到的不是这个 HiFi-GAN Generator 所需的检查点结构。
        raise KeyError(
            # KeyError 说明拿到的不是这个 HiFi-GAN Generator 所需的检查点结构。
            f"Checkpoint {checkpoint_path} does not contain a 'generator' state dict."
        )
    # strict=True 要求每个权重键都对应，缺一个或多一个都报错，防止静默错配网络。
    generator.load_state_dict(checkpoint["generator"], strict=True)
    # eval 关闭训练行为；本网络虽没有 dropout，也应遵守标准推理流程。
    generator.eval()
    # 权重归一化是训练参数化方法；官方推理代码会移除它以减少运行开销。
    generator.remove_weight_norm()
    # 不再需要梯度，把所有参数设成 requires_grad=False 可进一步节省显存。
    for parameter in generator.parameters():
        # requires_grad_ 是原地修改，只影响是否记录反向传播，不改变权重数值。
        parameter.requires_grad_(False)
    # 返回已经可直接调用的生成器，以及经过检查的原始配置字典。
    return generator, config


# 读取数值 mel 并统一为连续的 [80, 时间帧] 浮点矩阵。
def load_mel(path: Path) -> np.ndarray:
    # 允许读取 [80,T] 或明确转置的 [T,80] 数组，输出统一 float32 且拒绝非有限数。
    """把 .npy 整理成连续的 [80,T] float32，并进行完整数值检查。"""

    # 输入路径必须是普通文件，而不是不存在的路径或目录。
    if not path.is_file():
        # FileNotFoundError 明确指向缺失的 mel。
        raise FileNotFoundError(f"Mel file not found: {path}")
    # allow_pickle=False 禁止在 .npy 中执行任意 Python 对象，只读取纯数值数组。
    mel = np.load(path, allow_pickle=False)
    # mel 必须是二维矩阵；一维或三维都说明上游保存格式不符合约定。
    if mel.ndim != 2:
        # ValueError 同时报告实际形状。
        raise ValueError(f"Expected a 2-D mel at {path}, got shape {mel.shape}.")
    # 正常保存格式是 [80,T]，这时无需转置。
    if mel.shape[0] == EXPECTED_MEL_CHANNELS:
        # astype(copy=False) 在已经是 float32 时避免无意义复制。
        mel = mel.astype(np.float32, copy=False)
    # 为兼容某些工具的 [T,80]，如果第二轴是80，则明确转成 [80,T]。
    elif mel.shape[1] == EXPECTED_MEL_CHANNELS:
        # .T 交换两个轴；astype 保证最终统一为 float32。
        mel = mel.T.astype(np.float32, copy=False)
    # 两个轴都不是80，说明它根本不是本实验定义的mel。
    else:
        # ValueError 阻止错误特征进入声码器并生成毫无意义的音频。
        raise ValueError(
            # ValueError 阻止错误特征进入声码器并生成毫无意义的音频。
            f"Expected one mel axis to equal {EXPECTED_MEL_CHANNELS} at {path}, "
            # ValueError 阻止错误特征进入声码器并生成毫无意义的音频。
            f"got {mel.shape}."
        )
    # 时间帧必须大于0，否则无法生成任何波形采样点。
    if mel.shape[1] <= 0:
        # 空 mel 通常意味着上游截断或保存失败。
        raise ValueError(f"Mel has no time frames: {path}")
    # isfinite 同时排除 NaN、正无穷和负无穷。
    if not np.isfinite(mel).all():
        # 非有限值进入卷积后会把整条波形污染成 NaN。
        raise ValueError(f"Mel contains NaN or Inf: {path}")
    # ascontiguousarray 确保转置后的内存连续，torch.from_numpy 才能高效读取。
    return np.ascontiguousarray(mel, dtype=np.float32)


# 完整写临时 WAV 后才替换正式输出，防止中断污染结果。
def atomic_write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    # 先写同目录临时 WAV，完成后原子发布，避免中断留下损坏的正式文件。
    """先写同目录临时 WAV，完整成功后再原子替换正式输出。"""

    # parents=True 会建立多层目录；exist_ok=True 表示目录已有时不报错。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件仍以 .wav 结尾，便于 soundfile 自动识别，也与正式文件处在同一磁盘。
    temporary = path.with_name(path.stem + ".tmp.wav")
    # 若上次异常留下同名临时文件，先只删除这个明确的临时目标。
    temporary.unlink(missing_ok=True)
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # PCM_16 是通用16-bit整数WAV；soundfile会把[-1,1]浮点数正确量化。
        sf.write(
            # 将波形先写入本次明确的临时文件。
            str(temporary),
            # 保存声码器产生的实际浮点波形，不额外做峰值归一化。
            audio,
            # 写出与项目一致的真实波形采样率。
            sample_rate,
            # 显式指定标准 WAV 容器格式。
            format="WAV",
            # 按 16 位整数 PCM 量化，和正式客观评价的输入格式保持一致。
            subtype="PCM_16",
        )
        # os.replace 在同一文件系统内原子完成改名；正式路径已有时会被完整替换。
        os.replace(temporary, path)
    # 写 WAV 失败时只清理本次临时文件，再保留原异常。
    except Exception:
        # 写入失败时清理明确的临时文件，正式输出不会成为半个损坏文件。
        temporary.unlink(missing_ok=True)
        # raise 保留原始错误和堆栈，让外层记录真正失败原因。
        raise


# 将一份 mel 还原成波形，逐项核对长度、有限值和文件头。
def vocode_one(
    # 输入参数：将 mel 转为波形的生成器 G。
    generator: torch.nn.Module,
    # 输入参数：张量和模型使用的计算设备。
    device: torch.device,
    # 输入参数：预提取 mel 路径。
    mel_path: Path,
    # 输入参数：训练波形路径。
    wav_path: Path,
    # 是否明确允许替换已经存在的生成 WAV。
    overwrite: bool,
# 返回 VocodeResult，保存当前一条语音的成功、跳过或失败及诊断信息。
) -> VocodeResult:
    # 每条 mel 单独生成和检查，失败原因写入结果记录，方便批量任务回查。
    """处理一条mel；错误会变成结果记录，而不会让8,100条任务整体丢失报告。"""

    # 所有路径先转为绝对路径，报告不受执行时工作目录变化影响。
    mel_path = mel_path.resolve()
    # resolve(strict=False) 即使输出尚不存在，也能得到清楚的绝对位置。
    wav_path = wav_path.resolve()
    # 默认不覆盖已有 WAV，方便长任务中断后继续；--overwrite 才强制重做。
    if wav_path.exists() and not overwrite:
        # skip 不是失败，只表示这一条此前已经完成。
        return VocodeResult(
            # 记录此前已有输出被跳过；此工具的跳过分支不等同于完整字节身份认证。
            status="skip",
            # 保存该生成结果所对应的输入 mel 绝对路径。
            input_mel=str(mel_path),
            # 保存实际输出波形的绝对路径。
            output_wav=str(wav_path),
            # 说明跳过原因；只有显式 overwrite 才重做该输出。
            message="output already exists; use --overwrite to replace it",
        )
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 读取、转置（如需要）并验证 [80,T] float32 mel。
        mel = load_mel(mel_path)
        # from_numpy 不复制连续数组；unsqueeze(0) 增加 batch 轴得到 [1,80,T]。
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(device)
        # inference_mode 比 no_grad 更彻底地关闭梯度记录，适合纯推理并节省显存。
        with torch.inference_mode():
            # Generator 输出 [1,1,samples]，squeeze 去掉两个长度为1的轴。
            audio_tensor = generator(mel_tensor).squeeze(0).squeeze(0)
        # 搬到CPU并统一float32，soundfile才能写WAV，NumPy也才能做统计。
        audio = audio_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        # 输出必须是一维单声道；维度异常说明模型或 squeeze 接口发生变化。
        if audio.ndim != 1:
            # ValueError 会被下面的 except 记录为 fail。
            raise ValueError(f"Generator returned waveform shape {audio.shape}, expected [samples].")
        # 一条正常语音必须至少包含一个采样点。
        if audio.size <= 0:
            # 空输出无法写成有效语音。
            raise ValueError("Generator returned an empty waveform.")
        # 非有限波形会破坏PCM量化，且说明声码器数值已经失败。
        if not np.isfinite(audio).all():
            # 直接停止这一条，绝不把NaN偷偷变成整数0。
            raise ValueError("Generator returned NaN or Inf waveform values.")
        # tanh理论范围是[-1,1]；clip只防浮点极小越界，不做响度归一化。
        audio = np.clip(audio, -1.0, 1.0)
        # peak 是整条波形的最大绝对幅度。
        peak = float(np.max(np.abs(audio)))
        # 先转float64再平方求均值，降低长音频累积时的数值误差。
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        # 统计接近满幅的采样点比例；比例很高时可能有可闻削波。
        near_clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))
        # 预期样本数由 mel 帧数乘 hop=256 得到。
        expected_samples = int(mel.shape[1] * EXPECTED_HOP_SIZE)
        # 官方VCTK_V1应严格输出T*256；不一致说明配置/模型接口错了。
        if audio.size != expected_samples:
            # 这里不允许悄悄裁剪，因为那会掩盖真正的协议不匹配。
            raise ValueError(
                # 这里不允许悄悄裁剪，因为那会掩盖真正的协议不匹配。
                f"Waveform length {audio.size} does not equal "
                # 这里不允许悄悄裁剪，因为那会掩盖真正的协议不匹配。
                f"mel_frames*hop={mel.shape[1]}*{EXPECTED_HOP_SIZE}={expected_samples}."
            )
        # RMS极低意味着几乎静音；阈值只排除明显失败，不用于评价语音质量。
        if rms <= 1e-7:
            # 把静音视为失败，避免它进入后续UTMOS/CER并污染平均值。
            raise ValueError(f"Generated waveform is effectively silent: rms={rms:.3e}.")
        # 通过全部内存检查后，才以原子方式写正式WAV。
        atomic_write_wav(wav_path, audio, EXPECTED_SAMPLE_RATE)
        # sf.info 重新读取真实文件头，验证磁盘上的格式，而不是只相信写入参数。
        info = sf.info(str(wav_path))
        # 必须是22.05kHz、单声道且帧数与内存波形相同。
        if (
            # 磁盘 WAV 的采样率必须等于 22.05 kHz。
            info.samplerate != EXPECTED_SAMPLE_RATE
            # 磁盘 WAV 必须为单声道。
            or info.channels != 1
            # 写入后样本数必须和模型实际生成的长度一致。
            or info.frames != audio.size
        ):
            # 文件头验证失败时报告所有实际值，便于找出是哪项不符。
            raise ValueError(
                # 写入后重新读取文件头，错误信息报告实际采样率。
                f"Written WAV validation failed: sr={info.samplerate}, "
                # 文件头验证失败时报告所有实际值，便于找出是哪项不符。
                f"channels={info.channels}, frames={info.frames}, expected_frames={audio.size}."
            )
        # 所有检查通过，返回可写入报告的成功记录。
        return VocodeResult(
            # 记录该文件生成和检查均已通过。
            status="ok",
            # 保存该生成结果所对应的输入 mel 绝对路径。
            input_mel=str(mel_path),
            # 保存实际输出波形的绝对路径。
            output_wav=str(wav_path),
            # 记录实际输入 80 频带及帧数。
            mel_shape=[int(mel.shape[0]), int(mel.shape[1])],
            # 记录最终波形采样点数。
            samples=int(audio.size),
            # 按采样点数与真实采样率计算秒数。
            duration_seconds=float(audio.size / EXPECTED_SAMPLE_RATE),
            # 保留最大绝对振幅以检查接近量化上限的情况。
            peak=peak,
            # 保留均方根幅度以排查异常静音或响度变化。
            rms=rms,
            # 记录接近 ±1 的采样点比例，供削波诊断。
            near_clipping_fraction=near_clipping_fraction,
            # 成功记录没有额外错误信息。
            message="",
        )
    # 把当前单条异常转为失败记录，批量任务仍能留下完整报告。
    except Exception as error:
        # 单条失败被转成结构化记录；批量模式仍能继续处理后面的mel。
        return VocodeResult(
            # 明确标记当前语音没有生成可接受结果。
            status="fail",
            # 保存该生成结果所对应的输入 mel 绝对路径。
            input_mel=str(mel_path),
            # 保存实际输出波形的绝对路径。
            output_wav=str(wav_path),
            # 保存异常类型和具体原因，便于回查哪条 mel 出问题。
            message=f"{type(error).__name__}: {error}",
        )


# 根据互斥的单文件或目录模式建立确定的输入输出配对。
def collect_jobs(args: argparse.Namespace) -> Tuple[List[Tuple[Path, Path]], Path]:
    # 单文件和目录两种模式互斥，输入 mel 与输出 WAV 路径保持明确对应。
    """根据单文件或目录参数，构造明确的 (输入mel, 输出wav) 任务列表。"""

    # 单文件模式由 --input-mel 触发。
    if args.input_mel is not None:
        # 单文件必须同时给 --output-wav，否则程序不知道结果写到哪里。
        if args.output_wav is None:
            # parser.error 会显示用法并以非零状态退出。
            raise ValueError("--output-wav is required when --input-mel is used.")
        # 单文件模式不允许再给目录输出，防止含义冲突。
        if args.output_dir is not None:
            # 明确告诉使用者删掉哪个多余参数。
            raise ValueError("--output-dir cannot be used with --input-mel.")
        # 默认报告与WAV放在同一目录，文件名带vocoder_report便于识别。
        default_report = args.output_wav.parent / f"{args.output_wav.stem}_vocoder_report.json"
        # 返回只有一项的任务列表。
        return [(args.input_mel, args.output_wav)], default_report

    # 走到这里就是目录模式；--input-dir 已由互斥参数组保证存在。
    if args.output_dir is None:
        # 目录输入必须给目录输出。
        raise ValueError("--output-dir is required when --input-dir is used.")
    # 目录模式不允许指定单个WAV路径。
    if args.output_wav is not None:
        # 避免用户误以为8,100条可以塞进同一个WAV。
        raise ValueError("--output-wav cannot be used with --input-dir.")
    # resolve 后再检查目录，错误消息会显示绝对路径。
    input_dir = args.input_dir.resolve()
    # 输入必须是目录。
    if not input_dir.is_dir():
        # FileNotFoundError 直接指出缺失目录。
        raise FileNotFoundError(f"Input mel directory not found: {input_dir}")
    # rglob 支持将来mel按子目录组织；sorted保证每次处理顺序一致。
    mel_paths = sorted(input_dir.rglob("*.npy"))
    # --limit 正数时只取前N条，用于先做小规模检查。
    if args.limit is not None and args.limit > 0:
        # Python切片不会超过实际长度，目录文件不足N条也安全。
        mel_paths = mel_paths[: args.limit]
    # 空目录不是成功任务，应当立即提醒使用者路径选错。
    if not mel_paths:
        # RuntimeError 表示目录存在但没有可处理内容。
        raise RuntimeError(f"No .npy mel files found under {input_dir}")
    # jobs 保存每一条输入和对应输出。
    jobs: List[Tuple[Path, Path]] = []
    # 逐个保持相对目录结构，例如a/b.npy会输出到output/a/b.wav。
    for mel_path in mel_paths:
        # relative_to 得到输入目录下的相对位置。
        relative = mel_path.relative_to(input_dir)
        # with_suffix 把.npy扩展名替换成.wav。
        wav_path = args.output_dir / relative.with_suffix(".wav")
        # append 把这一对路径加入任务列表。
        jobs.append((mel_path, wav_path))
    # 批量默认报告放在输出目录根部。
    default_report = args.output_dir / "vocoder_report.json"
    # 返回完整任务和默认报告位置。
    return jobs, default_report


# 读取运行路径、设备和诊断选项，默认值保持原实验约定。
def parse_args() -> argparse.Namespace:
    # 读取运行路径、设备和诊断选项，默认值保持原实验约定。
    """定义并读取命令行参数。"""

    # description 会出现在 python3 vocode_voicegrad_mels.py --help 顶部。
    parser = argparse.ArgumentParser(description=__doc__)
    # 工程根目录用于寻找 hifigan/official 源码和默认预训练权重。
    parser.add_argument("--project-root", type=Path, default=SERVER_PROJECT_ROOT)
    # 输入模式二选一，而且必须选择一个。
    input_group = parser.add_mutually_exclusive_group(required=True)
    # --input-mel 指定一张mel，适合冒烟测试。
    input_group.add_argument("--input-mel", type=Path)
    # --input-dir 递归处理目录内全部.npy，适合8,100条正式结果。
    input_group.add_argument("--input-dir", type=Path)
    # 单文件模式的输出WAV路径。
    parser.add_argument("--output-wav", type=Path)
    # 目录模式的输出根目录。
    parser.add_argument("--output-dir", type=Path)
    # 默认None表示稍后根据project-root寻找VCTK_V1/generator_v1。
    parser.add_argument("--checkpoint", type=Path, default=None)
    # 默认None表示使用checkpoint同目录的config.json。
    parser.add_argument("--config", type=Path, default=None)
    # 默认自动选CUDA；仍允许显式写cpu用于诊断。
    parser.add_argument(
        # 允许显式指定租用服务器上的 CPU 或 GPU 设备。
        "--device",
        # 存在 CUDA 时默认使用 GPU，否则由调用者核对诊断设备。
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # --overwrite 明确授权覆盖已有WAV；默认则安全跳过。
    parser.add_argument("--overwrite", action="store_true")
    # --limit只对目录模式生效，适合先处理10条检查。
    parser.add_argument("--limit", type=int, default=None)
    # --report允许显式指定JSON报告位置；不写时使用自动位置。
    parser.add_argument("--report", type=Path, default=None)
    # parse_args 真正读取 sys.argv 并返回带属性的 Namespace。
    return parser.parse_args()


# 执行本脚本的输入检查、主要处理和结果记录。
def main() -> None:
    # 执行本脚本的输入检查、主要处理和结果记录。
    """程序总入口：解析参数、加载一次生成器、处理全部mel、写报告并判定成功。"""

    # 读取终端参数。
    args = parse_args()
    # resolve 把相对工程路径变成绝对路径，后续寻找文件不受cd位置影响。
    project_root = args.project_root.resolve()
    # 默认声码器检查点固定为公开的VCTK V1 generator_v1。
    checkpoint_path = (
        # 优先采用调用者明确指定的声码器权重。
        args.checkpoint
        # 仅在用户提供权重路径时覆盖通用基线默认位置。
        if args.checkpoint is not None
        # 未提供权重时使用历史通用 VCTK V1 基线。
        else project_root / "pretrained_hifigan_22k" / "VCTK_V1" / "generator_v1"
    # 将最终选中的权重或配置路径规范为绝对路径。
    ).resolve()
    # 默认配置与检查点同目录，这是官方HiFi-GAN的目录结构。
    config_path = (
        # 配置默认从所选权重旁读取，避免错配另一版本结构。
        args.config if args.config is not None else checkpoint_path.parent / "config.json"
    # 将最终选中的权重或配置路径规范为绝对路径。
    ).resolve()
    # torch.device 把cuda/cpu字符串转换成PyTorch设备对象。
    device = torch.device(args.device)
    # 如果要求CUDA但当前不可用，必须停止，避免误用CPU跑8,100条耗费极长时间。
    if device.type == "cuda" and not torch.cuda.is_available():
        # RuntimeError 会告诉使用者检查节点、环境或CUDA_VISIBLE_DEVICES。
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    # 固定CPU随机数种子；HiFi-GAN推理理论上无随机层，但记录固定值更规范。
    torch.manual_seed(1234)
    # CUDA存在时也固定对应随机数种子。
    if torch.cuda.is_available():
        # manual_seed_all 覆盖当前程序可见的全部GPU；我们实际只暴露一张卡。
        torch.cuda.manual_seed_all(1234)

    # 先构造任务，参数组合错误会在占用GPU前立即停止。
    jobs, default_report = collect_jobs(args)
    # 显式--report优先，否则使用单文件/目录模式推导出的默认路径。
    report_path = (args.report if args.report is not None else default_report).resolve()
    # 打印关键协议，日志里可以直接确认没有误用LJ或Universal声码器。
    print(f"project_root={project_root}")
    # 打印真实节点可见设备。
    print(f"device={device}")
    # 打印声码器权重绝对路径。
    print(f"checkpoint={checkpoint_path}")
    # 打印配置绝对路径。
    print(f"config={config_path}")
    # 打印本次任务数；单文件应为1，正式批量应为8100。
    print(f"jobs={len(jobs)}")

    # 生成器只加载一次；这是批量模式比重复调用旧脚本高效的关键。
    generator, config = load_generator(
        # 设置租用服务器上的项目根目录。
        project_root=project_root,
        # 设置模型权重路径。
        checkpoint_path=checkpoint_path,
        # 设置模型配置路径。
        config_path=config_path,
        # 设置张量和模型使用的计算设备。
        device=device,
    )
    # results 收集每条ok/skip/fail，保证最终报告完整。
    results: List[VocodeResult] = []
    # tqdm 显示处理进度，总数来自明确任务列表。
    for mel_path, wav_path in tqdm(jobs, desc="HiFi-GAN vocoding"):
        # 每次处理一条，生成器和GPU权重不会重复加载。
        result = vocode_one(
            # 设置将 mel 转为波形的生成器 G。
            generator=generator,
            # 设置张量和模型使用的计算设备。
            device=device,
            # 设置预提取 mel 路径。
            mel_path=mel_path,
            # 设置训练波形路径。
            wav_path=wav_path,
            # 把是否覆盖已有文件的明确命令行选择传给单条处理。
            overwrite=args.overwrite,
        )
        # 把结果加入列表，稍后统一汇总。
        results.append(result)

    # 分别统计成功、跳过和失败数量。
    ok_count = sum(result.status == "ok" for result in results)
    # bool在Python里可当0/1求和，因此这个表达式直接得到数量。
    skip_count = sum(result.status == "skip" for result in results)
    # fail_count大于0时程序最后会用非零退出码提醒调用者。
    fail_count = sum(result.status == "fail" for result in results)
    # 报告包含声码器身份、配置、参数和逐条统计，方便审计本次 epoch-500 教师结果。
    report = {
        # 记录租用服务器上的项目根目录。
        "project_root": str(project_root),
        # 记录张量和模型使用的计算设备。
        "device": str(device),
        # 记录模型权重文件。
        "checkpoint": str(checkpoint_path),
        # 绑定本次真正加载的 G 权重字节身份。
        "checkpoint_sha256": sha256_file(checkpoint_path),
        # 记录模型配置。
        "config": str(config_path),
        # 记录配置内容指纹。
        "config_sha256": sha256_file(config_path),
        # 保存完整生效配置，便于回查频带与上采样参数。
        "hifigan_config": config,
        # 记录波形文件采样率。
        "sample_rate": EXPECTED_SAMPLE_RATE,
        # 记录输入 mel 固定为 80 个频带。
        "mel_channels": EXPECTED_MEL_CHANNELS,
        # 记录相邻 mel 帧之间的波形采样点间隔。
        "hop_size": EXPECTED_HOP_SIZE,
        # 记录本次计划处理并收集结果的条目总数。
        "total": len(results),
        # 记录新生成并核对成功的文件数。
        "ok": ok_count,
        # 记录因已有输出而未重做的文件数。
        "skip": skip_count,
        # 记录当前发生错误的文件数，非零时进程最终失败。
        "fail": fail_count,
        # 保留每条语音的状态、路径、波形统计和错误原因。
        "results": [asdict(result) for result in results],
    }
    # 建立报告目录。
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时JSON，确保中断不会留下语法不完整的正式报告。
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    # ensure_ascii=False让中文错误消息可直接阅读，indent=2便于人工检查。
    with temporary_report.open("w", encoding="utf-8") as handle:
        # sort_keys=True使字段顺序稳定，便于版本比较。
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        # 末尾换行符合普通文本文件习惯。
        handle.write("\n")
    # 完整写完后原子替换正式报告。
    os.replace(temporary_report, report_path)

    # 在终端打印简洁汇总。
    print("[DONE] HiFi-GAN vocoding finished.")
    # 分三行显示，用户无需打开JSON即可判断成功。
    print(f"  ok     = {ok_count}")
    # 跳过数量在断点续跑时会大于0。
    print(f"  skip   = {skip_count}")
    # 正式任务要求fail必须为0。
    print(f"  fail   = {fail_count}")
    # 输出报告路径供后续归档。
    print(f"  report = {report_path}")
    # 单文件成功时额外打印最重要的波形质量统计。
    if len(results) == 1 and results[0].status == "ok":
        # result变量在这里就是唯一一条成功记录。
        result = results[0]
        # 格式化到6位小数既易读，也保留足够诊断精度。
        print(
            # 格式化到6位小数既易读，也保留足够诊断精度。
            f"  wav    = {result.output_wav}\n"
            # 显示该条生成音频的实际秒数。
            f"  duration_seconds = {result.duration_seconds:.6f}\n"
            # 显示最大绝对幅度，检查接近满幅情况。
            f"  peak             = {result.peak:.6f}\n"
            # 显示均方根幅度，帮助识别异常静音或响度。
            f"  rms              = {result.rms:.6f}\n"
            # 显示接近满幅采样点占比，辅助检查削波。
            f"  near_clip        = {result.near_clipping_fraction:.8f}"
        )
    # 任意一条失败都用RuntimeError使Shell得到非零返回码，避免自动流程误判成功。
    if fail_count:
        # 报告已经写好，因此错误消息同时告诉使用者去哪里看逐条原因。
        raise RuntimeError(f"{fail_count} mel files failed; inspect {report_path}")


# Python入口保护：直接执行本文件时调用main，被其他脚本import时不自动开始处理。
if __name__ == "__main__":
    # 调用总入口。
    main()
