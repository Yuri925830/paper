#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 使用固定 GE2E 编码器产生 256 维参考身份向量，提取前沿用其独立的音频预处理。
"""
Extract utterance-level speaker embeddings for FastVoiceGrad teacher on VCTK.

Key idea:
- Reuse a separately supplied legacy speaker encoder implementation.
- But save results in the new project structure:
    data/spk_emb/<speaker>/<utt_id>.npy

Input:
  data/metadata/all_manifest.csv

Output:
  data/spk_emb/<speaker>/<utt_id>.npy

Behavior:
- Read wav path from manifest
- Use legacy preprocess_wav(...)
- Use legacy SpeakerEncoder(...).embed_utterance(...)
- Save one utterance embedding per utterance
- Resume-friendly: validate existing files and rebuild missing/invalid files
"""

# =============================================================================
# 预处理第 06 步：提取 256 维 speaker embedding（说话人身份条件）
# =============================================================================
# BNF 回答“说了什么”，speaker embedding 回答“谁在说”。FastVoiceGrad any-to-any
# 不使用只能认识训练人的 speaker label，而是从任意参考语音提取身份向量，因此
# 才能把 unseen target speaker 的 reference 作为目标音色。
# 本脚本复用旧工程 GE2E 风格 SpeakerEncoder，不训练它；每条保存一个 [256]
# float32 数组到 data/spk_emb/<speaker>/<utt_id>.npy。

# --------------------------- 导入包逐项解释 ---------------------------
# csv 读取 all_manifest；json 写报告；sys.path 临时加入旧工程；argparse 读参数；
# Path 拼路径；Dict/List 是类型提示。
import csv
# 保存或读取实验设置、数据指纹及处理进度的结构化记录。
import json
# hashlib 审核外部文件；tempfile 用于同目录原子写入。
import hashlib
# 为待发布的文件创建不重名临时路径，减少中断或并发写入的干扰。
import tempfile
# 处理解释器路径和旧特征提取器模块的导入位置。
import sys
# 定义终端参数，使数据目录、检查点和运行选项明确记录。
import argparse
# os.environ 读取可移植路径，不把实验室服务器目录公开在源码里。
import os
# 调用已有命令行工具，并检查外部程序是否正常完成。
import subprocess
# 用路径对象定位数据、权重和报告，减少手工拼接目录的错误。
from pathlib import Path
# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Dict, List

# NumPy 检查向量、统一 float32 并保存 .npy；tqdm 显示进度条。
import numpy as np


# 当前工程根目录：可由 VOICEGRAD_PROJECT_ROOT 覆盖。
ROOT = Path(
    # 优先读取显式工程根目录；没有环境变量时回到本文件向上两级的仓库目录。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[1])
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()
# 历史 speaker encoder 代码和权重不随公开仓库分发，由环境变量告诉脚本位置。
LEGACY_BNF_ROOT = Path(
    # 优先读取外部提取器目录环境变量，未设置时使用工程 external 子目录。
    os.environ.get("VOICEGRAD_LEGACY_BNF_ROOT", ROOT / "external" / "legacy_compute_bnf")
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()

# 全部实验语音清单。
DEFAULT_MANIFEST = ROOT / "data" / "metadata" / "all_manifest.csv"
# speaker embedding 输出根目录。
DEFAULT_OUT_DIR = ROOT / "data" / "spk_emb"
# JSON 处理报告。
DEFAULT_REPORT = ROOT / "exp" / "logs" / "spk_emb_extraction_report.json"

# 与旧 SpeakerEncoder 代码配套的预训练权重。
DEFAULT_SPK_ENCODER_CKPT = LEGACY_BNF_ROOT / "speaker_encoder" / "ckpt" / "pretrained_bak_5805000.pt"
# SpeakerEncoder 实际 import 的超参文件，其 SHA-256 写入审计报告。
DEFAULT_SPK_CONFIG = LEGACY_BNF_ROOT / "speaker_encoder" / "hparams.py"

# GE2E 每条参考语音应输出 256 维身份向量，这与 VoiceGrad 条件维度一致。
EXPECTED_EMBED_DIM = 256
# 固定本次复现审核过的外部提取器源码提交，防止升级依赖后混用特征。
EXPECTED_LEGACY_COMMIT = "b59cb9862cf4b82a3bdb589950e25cab85fc9b03"
# 允许使用的 GE2E 权重内容指纹。
EXPECTED_SPK_CHECKPOINT_SHA256 = (
    # 固定 GE2E 说话人编码器权重的完整 SHA-256 摘要。
    "bc7ff82ef75becd495aab2ede3a8220da393a717f178ae9534df355a6173bbca"
)
# 允许使用的 GE2E 配置内容指纹。
EXPECTED_SPK_CONFIG_SHA256 = (
    # 固定 GE2E 音频前端及模型配置的完整 SHA-256 摘要。
    "1ef3e1e56df97e078d02a8da5a5a47ef5e6a0950c5139d4440660fca54bb3946"
)
# 来源报告使用第二版结构，要求有逐文件内容与输入音频证据。
REPORT_SCHEMA_VERSION = 2
# BNF 或说话人嵌入的特征类别。
FEATURE_KIND = "speaker_embedding"


# 逐块读取文件并计算字节内容指纹；下面的参数分别限定输入数据和处理约束。
def sha256_file(path: Path) -> str:
    # 分块读取计算文件指纹，保持内存占用有上限。
    """Return a lowercase SHA-256 digest using bounded memory."""
    # 逐块累计文件内容的 SHA-256 对象。
    digest = hashlib.sha256()
    # 在受控文件句柄中完成逐块读取文件并计算字节内容指纹，退出此段后自动关闭文件。
    with path.open("rb") as handle:
        # 每次读取 1 MB 文件字节参与指纹计算，直到文件结束，避免把大权重一次载入内存。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
            digest.update(chunk)
    # 返回小写十六进制 SHA-256，用于比较本次与历史文件是否完全一致。
    return digest.hexdigest()


# 统一文本换行后核对提取器源码内容，避免跨系统换行干扰；下面的参数分别限定输入数据和处理约束。
def sha256_text_lf(path: Path) -> str:
    # 先把 CRLF 和 CR 统一为 Git 的 LF 形式，再对外部源码内容计算摘要。
    """Hash tracked text with Git's LF representation on every host OS."""

    # 当前读取并整理后的文件字节。
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    # 将规范化后的字节内容转成 SHA-256 指纹，避免仅凭文件名称认定版本一致。
    return hashlib.sha256(content).hexdigest()


# 核对旧特征提取器源码、配置和预训练权重身份；下面的参数分别限定输入数据和处理约束。
def verify_legacy_provenance(
    # 单独提供并固定版本的 ppg-vc 提取器源码目录。
    legacy_root: Path,
    # 特征提取器的固定预训练权重路径。
    checkpoint_path: Path,
    # 特征提取器结构与预处理参数文件。
    config_path: Path,
# 返回核对旧特征提取器源码、配置和预训练权重身份整理的 Dict[str, str] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, str]:
    # 核对旧特征提取器源码、配置和预训练权重身份。
    """Fail closed unless the audited ppg-vc checkout and speaker weight are used."""
    # 逐项核对提取器源码目录、权重和配置是否实际存在。
    for label, path in (
        # 需要检查外部 ppg-vc 的源码根目录。
        ("legacy ppg-vc root", legacy_root),
        # 需要检查固定 GE2E 说话人编码器权重。
        ("speaker checkpoint", checkpoint_path),
        # 需要检查固定 GE2E 预处理与模型参数文件。
        ("speaker config", config_path),
    ):
        # 检查当前待检查或读写的文件路径是否缺失；缺失时不能继续信任该输入。
        if not path.exists():
            # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
            raise FileNotFoundError(f"{label} not found: {path}")

    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 外部提取器当前 Git 提交编号。
        commit = subprocess.run(
            # 只读查询外部提取器的当前提交编号。
            ["git", "-C", str(legacy_root), "rev-parse", "HEAD"],
            # 外部命令失败立即抛异常，不能将空输出当成通过审核。
            check=True,
            # 捕获命令输出供程序核对，不只打印到终端。
            capture_output=True,
            # 将 Git 输出作为文本读取，便于比较提交编号及改动列表。
            text=True,
        # 取出提交编号并统一为无空白的小写形式。
        ).stdout.strip().lower()
        # 固定外部源码中相对基准提交的已跟踪文件改动。
        tracked_changes = subprocess.run(
            # 下面展开核对旧特征提取器源码、配置和预训练权重身份所需的这组条目；各字段随后逐一说明。
            [
                # 调用 Git 只读检查外部特征提取器来源。
                "git",
                # 指定后续 Git 命令运行的源码目录。
                "-C",
                # 让 Git 命令在已明确指定的外部提取器目录执行。
                str(legacy_root),
                # 比较已跟踪文件相对固定提交的内容变化。
                "diff",
                # 只输出发生变化的文件名，足够判断外部代码是否被改动。
                "--name-only",
                # 忽略文本行尾 CRLF 与 LF 差异，仍保留实际源码内容变化。
                "--ignore-cr-at-eol",
                # 以当前固定 Git 提交作为比较基准。
                "HEAD",
                # 结束 Git 选项解析，避免后续路径被解释成开关。
                "--",
            ],
            # 外部命令失败立即抛异常，不能将空输出当成通过审核。
            check=True,
            # 捕获命令输出供程序核对，不只打印到终端。
            capture_output=True,
            # 将 Git 输出作为文本读取，便于比较提交编号及改动列表。
            text=True,
        # 提取已跟踪文件改动列表，空串才表示源码未被修改。
        ).stdout.strip()
    # 捕获核对旧特征提取器源码、配置和预训练权重身份中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except (OSError, subprocess.CalledProcessError) as exc:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错记录外部提取器路径与 Git 来源查询失败原因。
            f"Cannot audit ppg-vc Git provenance at {legacy_root}: {exc}"
        # 保留 Git 查询的原始异常，便于追查身份编码器源码来源审核失败原因。
        ) from exc

    # 按外部提取器当前 Git 提交编号决定是否进入此分支；判断条件为 commit != EXPECTED_LEGACY_COMMIT。
    if commit != EXPECTED_LEGACY_COMMIT:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明外部提取器 Git 提交与固定审核版本不同。
            "Unexpected ppg-vc commit: "
            # 同时展示允许提交与实际提交，实际为空也明确显示。
            f"expected {EXPECTED_LEGACY_COMMIT}, got {commit or '<empty>'}"
        )
    # 按固定外部源码中相对基准提交的已跟踪文件改动决定是否进入此分支；判断条件为 tracked_changes。
    if tracked_changes:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明固定外部源码中的已跟踪文件被改动。
            "Tracked content in the pinned ppg-vc checkout is modified; refusing "
            # 列出未通过来源审核的源码修改，当前不继续提取身份特征。
            f"unaudited extraction:\n{tracked_changes}"
        )

    # 实际提取器权重文件的 SHA-256 指纹。
    checkpoint_sha256 = sha256_file(checkpoint_path)
    # 按实际提取器权重文件的 SHA-256 指纹决定是否进入此分支；判断条件为 checkpoint_sha256 != EXPECTED_SPK_CHECKPOINT_SHA256。
    if checkpoint_sha256 != EXPECTED_SPK_CHECKPOINT_SHA256:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明身份编码器预训练权重内容不匹配。
            "Unexpected speaker checkpoint SHA-256: "
            # 列出身份编码器权重的基准与实际摘要。
            f"expected {EXPECTED_SPK_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )
    # 按统一换行规范得到的提取器配置指纹。
    config_sha256 = sha256_text_lf(config_path)
    # 按按统一换行规范得到的提取器配置指纹决定是否进入此分支；判断条件为 config_sha256 != EXPECTED_SPK_CONFIG_SHA256。
    if config_sha256 != EXPECTED_SPK_CONFIG_SHA256:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明身份编码器配置内容不匹配。
            "Unexpected speaker config SHA-256: "
            # 列出身份编码器配置的基准与实际摘要。
            f"expected {EXPECTED_SPK_CONFIG_SHA256}, got {config_sha256}"
        )

    # 返回核对旧特征提取器源码、配置和预训练权重身份得到的完整字段集合，各字段用途在下面逐项说明。
    return {
        # 记录实际外部提取器源码提交编号。
        "legacy_git_commit": commit,
        # 记录预先固定的允许源码提交编号。
        "legacy_git_expected_commit": EXPECTED_LEGACY_COMMIT,
        # 记录实际提取器权重文件的 SHA-256 指纹。
        "checkpoint_sha256": checkpoint_sha256,
        # 记录允许使用的唯一提取器权重指纹。
        "checkpoint_expected_sha256": EXPECTED_SPK_CHECKPOINT_SHA256,
        # 记录按统一换行规范得到的提取器配置指纹。
        "config_sha256": config_sha256,
        # 记录允许使用的提取器参数文件指纹。
        "config_expected_sha256": EXPECTED_SPK_CONFIG_SHA256,
    }


# 核对参考语音身份嵌入的 256 维 float32 数组；下面的参数分别限定输入数据和处理约束。
def validate_speaker_file(
    # 当前待检查或读写的文件路径。
    path: Path,
    # 特征文件预期的维数。
    expected_dim: int = EXPECTED_EMBED_DIM,
# 返回核对参考语音身份嵌入的 256 维 float32 数组整理的 tuple 结果，内部各项保持前面约定的对应关系。
) -> tuple:
    # 核对参考语音身份嵌入的 256 维 float32 数组。
    """Validate an on-disk speaker embedding and return its shape."""
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 当前待校验或登记的数值特征数组。
        array = np.load(path, allow_pickle=False)
    # 捕获核对参考语音身份嵌入的 256 维 float32 数组中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as exc:
        # 核对参考语音身份嵌入的 256 维 float32 数组未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Unreadable speaker embedding {path}: {exc}") from exc
    # 要求读取对象符合核对参考语音身份嵌入的 256 维 float32 数组预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(array, np.ndarray):
        # 核对参考语音身份嵌入的 256 维 float32 数组未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Speaker embedding is not an ndarray: {path}")
    # 磁盘说话人特征必须已经是 [256] 单个向量；不接受混入时间轴的数组。
    if array.shape != (expected_dim,):
        # 核对参考语音身份嵌入的 256 维 float32 数组未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 报错要求身份特征为单个 256 维向量。
            f"Speaker embedding must have shape [{expected_dim}], got "
            # 补充实际数组形状和文件路径，便于修复错维文件。
            f"{array.shape}: {path}"
        )
    # 检查数组所有元素均为有限数；NaN 或无穷值会污染后续模型输入和统计量。
    if not np.isfinite(array).all():
        # 核对参考语音身份嵌入的 256 维 float32 数组未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Speaker embedding contains NaN or Inf: {path}")
    # 要求特征以单精度保存，避免来源混杂或推理时隐式改变数值精度。
    if array.dtype != np.float32:
        # 核对参考语音身份嵌入的 256 维 float32 数组未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 报错列出精度不合格的身份特征文件与实际类型。
            f"Speaker embedding must be float32, got {array.dtype}: {path}"
        )
    # 返回经过验证的数组形状，统一为普通整数供 JSON 来源台账保存。
    return tuple(int(value) for value in array.shape)


# 构造说话人和句子共同确定的特征身份；下面的参数分别限定输入数据和处理约束。
def feature_key(row: Dict[str, str]) -> str:
    # 构造说话人和句子共同确定的特征身份。
    """Return the manifest-stable relative name used in provenance ledgers."""

    # 把说话人目录和句子编号拼成稳定的相对特征名，作为来源台账的唯一键。
    return f"{row['speaker']}/{row['utt_id']}.npy"


# 按数组形状、类型和原始字节生成数值内容指纹；下面的参数分别限定输入数据和处理约束。
def sha256_array(array: np.ndarray) -> str:
    # 按数组形状、类型和原始字节生成数值内容指纹。
    """Hash numerical content canonically, independently of .npy container details."""

    # 当前正在校验的数值或记录。
    value = np.ascontiguousarray(array)
    # 逐块累计文件内容的 SHA-256 对象。
    digest = hashlib.sha256()
    # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
    digest.update(value.dtype.str.encode("ascii"))
    # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
    digest.update(b"\0")
    # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
    digest.update(",".join(str(int(item)) for item in value.shape).encode("ascii"))
    # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
    digest.update(b"\0")
    # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
    digest.update(memoryview(value).cast("B"))
    # 返回小写十六进制 SHA-256，用于比较本次与历史文件是否完全一致。
    return digest.hexdigest()


# 固定清单身份顺序并计算整组样本的摘要；下面的参数分别限定输入数据和处理约束。
def manifest_keys_sha256(rows: List[Dict[str, str]]) -> str:
    # 固定清单身份顺序并计算整组样本的摘要。
    """Bind a report to the exact ordered manifest key set."""

    # 逐块累计文件内容的 SHA-256 对象。
    digest = hashlib.sha256()
    # 逐项遍历本次选定的语音清单，保持每条记录都经过相同检查。
    for row in rows:
        # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
        digest.update(feature_key(row).encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
        digest.update(b"\n")
    # 返回小写十六进制 SHA-256，用于比较本次与历史文件是否完全一致。
    return digest.hexdigest()


# 记录单条音频与对应特征的内容指纹和提取设置；下面的参数分别限定输入数据和处理约束。
def make_feature_record(
    # 当前待检查或读写的文件路径。
    path: Path,
    # 当前待校验或登记的数值特征数组。
    array: np.ndarray,
    # 实际送入特征提取器的原始 WAV 路径。
    input_wav_path: Path = None,
    # 该文件本轮为新生成还是经过验证后复用。
    action: str = None,
# 返回记录单条音频与对应特征的内容指纹和提取设置整理的 Dict[str, object] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, object]:
    # 记录单条音频与对应特征的内容指纹和提取设置。
    """Create one compact, content-verifiable feature ledger entry."""

    # 特征文件的大小等磁盘属性。
    stat = path.stat()
    # 当前正在校验的数值或记录。
    value = np.asarray(array)
    # 当前特征的内容、形状、输入音频及处理动作记录。
    record = {
        # 记录包含数组类型、形状和数值字节的内容指纹。
        "array_sha256": sha256_array(value),
        # 记录当前文件的实际字节数。
        "bytes": int(stat.st_size),
        # 记录参与广播或校验的目标张量形状。
        "shape": [int(item) for item in value.shape],
        # 记录张量的数值类型。
        "dtype": value.dtype.str,
    }
    # 调用方提供本轮处理动作时，将它一并写入特征来源记录。
    if action is not None:
        # 只承认本轮新生成和来源验证后复用两种动作，拒绝无法追溯的完成标记。
        if action not in {"generated", "reused_verified"}:
            # 记录单条音频与对应特征的内容指纹和提取设置未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(f"Invalid feature-record action: {action!r}")
        # 在当前特征台账补充本轮新生成或经审核后复用的处理类别。
        record["run_action"] = action
    # 提供输入音频时，连同 WAV 内容指纹和文件大小一起绑定到当前特征。
    if input_wav_path is not None:
        # 输入 WAV 文件的大小等磁盘属性。
        wav_stat = input_wav_path.stat()
        # 在当前特征台账补充该特征实际使用的输入 WAV 内容指纹。
        record["input_wav_sha256"] = sha256_file(input_wav_path)
        # 在当前特征台账补充该输入 WAV 的字节数。
        record["input_wav_bytes"] = int(wav_stat.st_size)
    # 返回当前特征的内容、形状、输入音频及处理动作记录，供调用方继续处理。
    return record


# 汇总逐样本记录以验证整组特征的覆盖和来源；下面的参数分别限定输入数据和处理约束。
def summarize_feature_records(
    # 本次选定的语音清单。
    rows: List[Dict[str, str]],
    # 按语音唯一身份建立的逐文件来源台账。
    records: Dict[str, Dict[str, object]],
# 返回汇总逐样本记录以验证整组特征的覆盖和来源整理的 Dict[str, object] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, object]:
    # 汇总逐样本记录以验证整组特征的覆盖和来源。
    """Summarize a complete or partial ledger without re-reading feature contents."""

    # 按清单顺序累计特征数值内容的指纹。
    content_digest = hashlib.sha256()
    # 按清单顺序累计特征身份和文件大小的指纹。
    layout_digest = hashlib.sha256()
    # 按清单顺序累计输入 WAV 内容的指纹。
    input_content_digest = hashlib.sha256()
    # 按清单顺序累计输入 WAV 文件大小的指纹。
    input_layout_digest = hashlib.sha256()
    # 已经纳入当前统计或审核的文件数量。
    count = 0
    # 逐项遍历本次选定的语音清单，保持每条记录都经过相同检查。
    for row in rows:
        # 当前语音的唯一身份。
        key = feature_key(row)
        # 当前特征的内容、形状、输入音频及处理动作记录。
        record = records.get(key)
        # 当前语音尚无可信台账时跳过复用统计，留给后续生成或失败处理。
        if record is None:
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue
        # 累计已纳入本次统计的有效条目数量。
        count += 1
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征数值内容的指纹；字段顺序必须与读取审核时一致。
        content_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征数值内容的指纹；字段顺序必须与读取审核时一致。
        content_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征数值内容的指纹；字段顺序必须与读取审核时一致。
        content_digest.update(str(record["array_sha256"]).encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征数值内容的指纹；字段顺序必须与读取审核时一致。
        content_digest.update(b"\n")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征身份和文件大小的指纹；字段顺序必须与读取审核时一致。
        layout_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征身份和文件大小的指纹；字段顺序必须与读取审核时一致。
        layout_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征身份和文件大小的指纹；字段顺序必须与读取审核时一致。
        layout_digest.update(str(int(record["bytes"])).encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征身份和文件大小的指纹；字段顺序必须与读取审核时一致。
        layout_digest.update(b"\n")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 内容的指纹；字段顺序必须与读取审核时一致。
        input_content_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 内容的指纹；字段顺序必须与读取审核时一致。
        input_content_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 内容的指纹；字段顺序必须与读取审核时一致。
        input_content_digest.update(str(record["input_wav_sha256"]).encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 内容的指纹；字段顺序必须与读取审核时一致。
        input_content_digest.update(b"\n")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 文件大小的指纹；字段顺序必须与读取审核时一致。
        input_layout_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 文件大小的指纹；字段顺序必须与读取审核时一致。
        input_layout_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 文件大小的指纹；字段顺序必须与读取审核时一致。
        input_layout_digest.update(str(int(record["input_wav_bytes"])).encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 文件大小的指纹；字段顺序必须与读取审核时一致。
        input_layout_digest.update(b"\n")
    # 返回汇总逐样本记录以验证整组特征的覆盖和来源得到的完整字段集合，各字段用途在下面逐项说明。
    return {
        # 记录已经纳入当前统计或审核的文件数量。
        "count": count,
        # 记录按固定清单顺序汇总的特征数值指纹。
        "content_sha256": content_digest.hexdigest(),
        # 记录特征数值集合指纹的拼接与摘要规则。
        "content_algorithm": "sha256(manifest-order key + canonical-array-sha256)",
        # 记录语音身份和文件大小的集合指纹。
        "layout_sha256": layout_digest.hexdigest(),
        # 记录文件布局集合指纹的计算规则。
        "layout_algorithm": "sha256(manifest-order key + file-size)",
        # 记录输入 WAV 内容集合指纹。
        "input_content_sha256": input_content_digest.hexdigest(),
        # 记录输入 WAV 内容集合指纹的计算规则。
        "input_content_algorithm": "sha256(manifest-order key + wav-sha256)",
        # 记录输入 WAV 大小集合指纹。
        "input_layout_sha256": input_layout_digest.hexdigest(),
        # 记录输入 WAV 大小集合指纹的计算规则。
        "input_layout_algorithm": "sha256(manifest-order key + wav-file-size)",
    }


# 重用特征前逐条核对音频与已有发布证据；下面的参数分别限定输入数据和处理约束。
def verify_reused_feature_records(
    # 本次选定的语音清单。
    rows: List[Dict[str, str]],
    # 特征输出目录。
    out_dir: Path,
    # 数据处理审计报告的保存位置。
    report_path: Path,
    # 中断续作使用的逐条进度日志。
    journal_path: Path,
    # 本轮固定提取器源码、配置和权重的身份记录。
    output_provenance: Dict[str, str],
# 返回重用特征前逐条核对音频与已有发布证据整理的 Dict[str, Dict[str, object]] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, Dict[str, object]]:
    # 重用特征前逐条核对音频与已有发布证据。
    """Require per-file proof before assigning a producer to reused embeddings."""

    # 按本次选定的语音清单决定是否进入此分支；判断条件为 not rows。
    if not rows:
        # 没有可复用样本或既有日志时返回空台账，不捏造来源记录。
        return {}
    # 前次报告中的逐文件来源台账。
    previous_records = load_prior_feature_records(
        # 数据处理审计报告的保存位置。
        report_path,
        # 中断续作使用的逐条进度日志。
        journal_path,
        # 本轮固定提取器源码、配置和权重的身份记录。
        output_provenance,
    )

    # 逐条核对内容和输入音频后允许复用的特征台账。
    verified: Dict[str, Dict[str, object]] = {}
    # 逐项遍历本次选定的语音清单，保持每条记录都经过相同检查。
    for row in rows:
        # 当前语音的唯一身份。
        key = feature_key(row)
        # 该语音在历史来源报告中的记录。
        old_record = previous_records.get(key)
        # 要求读取对象符合重用特征前逐条核对音频与已有发布证据预期的数据结构，否则不能继续信任其中字段。
        if not isinstance(old_record, dict):
            # 重用特征前逐条核对音频与已有发布证据未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 指出已有身份向量缺少可验证来源记录，需要重新提取。
                f"Existing speaker embedding {key} has no trusted record; run with --force."
            )
        # 历史台账必须说明该文件来自新生成或已验证复用，不能只有形状信息。
        if old_record.get("run_action") not in {"generated", "reused_verified"}:
            # 重用特征前逐条核对音频与已有发布证据未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 指出已有身份台账没有记录该文件本轮的生成或复用方式。
                f"Existing speaker embedding {key} record does not identify how it "
                # 补充说明缺少生成或复用动作证据，需要重新提取。
                "was produced/reused; run with --force."
            )
        # 当前待检查或读写的文件路径。
        path = out_dir / key
        # 复用前检查说话人向量的维数、精度和有限数值。
        validate_speaker_file(path)
        # 当前待校验或登记的数值特征数组。
        array = np.load(path, allow_pickle=False)
        # 当前语音的音频文件。
        wav_path = Path(row["wav22k_path"])
        # 重新读取当前特征与输入 WAV 后得到的记录。
        current_record = make_feature_record(
            # 当前待检查或读写的文件路径。
            path,
            # 当前待校验或登记的数值特征数组。
            array,
            # 实际送入特征提取器的原始 WAV 路径。
            input_wav_path=wav_path,
            # 该文件本轮为新生成还是经过验证后复用。
            action="reused_verified",
        )
        # 逐项比较数组内容、形状、精度及其实际输入 WAV 的内容和大小。
        for field in (
            # 此字段保存包含数组类型、形状和数值字节的内容指纹。
            "array_sha256",
            # 此字段保存参与广播或校验的目标张量形状。
            "shape",
            # 此字段保存张量的数值类型。
            "dtype",
            # 此字段保存该特征实际使用的输入 WAV 内容指纹。
            "input_wav_sha256",
            # 此字段保存该输入 WAV 的字节数。
            "input_wav_bytes",
        ):
            # 任何关键来源字段改变都拒绝复用，要求明确重新提取。
            if old_record.get(field) != current_record[field]:
                # 重用特征前逐条核对音频与已有发布证据未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(
                    # 指出当前身份向量或输入 WAV 已偏离历史内容证据。
                    f"Existing speaker embedding {key} no longer matches its prior "
                    # 列出具体不匹配字段，并提示显式重新提取。
                    f"{field} attestation; run with --force."
                )
        # 只有所有内容和输入音频比较通过，才登记为可信复用。
        verified[key] = current_record
    # 返回全部逐项核验通过的特征台账。
    return verified


# 合并本次新提取及可信重用特征的完整来源报告；下面的参数分别限定输入数据和处理约束。
def build_extraction_report(
    # 从这里起的参数必须写明名称传入，减少路径或审核开关位置混淆。
    *,
    # 本次实际使用的清单文件。
    manifest_path: Path,
    # 本次选定的语音清单。
    rows: List[Dict[str, str]],
    # 特征输出目录。
    out_dir: Path,
    # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
    legacy_bnf_root: Path,
    # 预先固定的 GE2E 说话人编码器权重。
    spk_encoder_ckpt: Path,
    # 说话人编码器的预处理和结构参数。
    speaker_config: Path,
    # 本轮固定提取器源码、配置和权重的身份记录。
    output_provenance: Dict[str, str],
    # 是否明确要求重新计算已有特征。
    force: bool,
    # 本轮新提取成功的语音数量。
    generated_rows: int,
    # 本轮通过来源审核后复用的语音条目或数量。
    reused_rows: int,
    # 本轮提取失败的语音数量。
    failed_rows: int,
    # 按语音唯一身份建立的逐文件来源台账。
    records: Dict[str, Dict[str, object]],
    # 有限数量的失败语音及具体原因。
    fail_examples: List[Dict[str, str]],
    # 本轮实际观察到的说话人向量维数。
    embed_dim_seen: int = None,
# 返回合并本次新提取及可信重用特征的完整来源报告整理的 Dict[str, object] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, object]:
    # 合并本次新提取及可信重用特征的完整来源报告。
    """Build the single auditable schema consumed by validation and training."""

    # 是否完整覆盖当前清单且没有失败。
    complete = failed_rows == 0 and len(records) == len(rows)
    # 按本轮新提取成功的语音数量决定是否进入此分支；判断条件为 generated_rows and reused_rows。
    if generated_rows and reused_rows:
        # 全部新生成、全部可信复用或两者混合的特征来源类别。
        source_mode = "generated_and_verified_reused"
    # 按本轮新提取成功的语音数量决定是否进入此分支；判断条件为 generated_rows。
    elif generated_rows:
        # 全部新生成、全部可信复用或两者混合的特征来源类别。
        source_mode = "all_generated"
    # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
    else:
        # 全部新生成、全部可信复用或两者混合的特征来源类别。
        source_mode = "all_reused_verified"
    # 返回合并本次新提取及可信重用特征的完整来源报告得到的完整字段集合，各字段用途在下面逐项说明。
    return {
        # 记录特征来源报告的结构版本号。
        "report_schema_version": REPORT_SCHEMA_VERSION,
        # 记录BNF 或说话人嵌入的特征类别。
        "feature_kind": FEATURE_KIND,
        # 记录本轮处理完成、部分完成或失败的状态。
        "status": "complete" if complete else "incomplete",
        # 记录全部新生成、全部可信复用或两者混合的特征来源类别。
        "source_mode": source_mode,
        # 记录全量统一验证或仅部分验证的来源状态。
        "provenance_status": "uniform_verified" if complete else "partial_verified",
        # 记录待读取的样本清单。
        "manifest": {
            # 记录当前待检查或读写的文件路径。
            "path": str(manifest_path),
            # 记录当前文件的 SHA-256 内容指纹。
            "sha256": sha256_file(manifest_path),
            # 记录清单语音身份及顺序的指纹。
            "keys_sha256": manifest_keys_sha256(rows),
            # 记录本次选定的语音清单。
            "rows": len(rows),
        },
        # 记录特征输出目录。
        "out_dir": str(out_dir),
        # 记录固定 Git 提交的外部 BNF/说话人提取器代码目录。
        "legacy_bnf_root": str(legacy_bnf_root),
        # 记录预先固定的 GE2E 说话人编码器权重。
        "spk_encoder_ckpt": str(spk_encoder_ckpt),
        # 记录说话人编码器的预处理和结构参数。
        "speaker_config": str(speaker_config),
        # 记录本轮固定提取器源码、配置和权重的身份记录。
        "output_provenance": output_provenance,
        # 记录是否明确要求重新计算已有特征。
        "force": force,
        # 记录说话人向量预期的 256 维通道数。
        "expected_embed_dim": EXPECTED_EMBED_DIM,
        # 记录本轮清单中的语音总数。
        "total_rows": len(rows),
        # 记录本轮新提取成功的语音数量。
        "generated_rows": generated_rows,
        # 记录来源与内容均验证后复用的文件数量。
        "reused_verified_rows": reused_rows,
        # 记录未经来源验证便复用的文件数量，正式流程要求为零。
        "unverified_reused_rows": 0,
        # 记录本轮提取失败的语音数量。
        "failed_rows": failed_rows,
        # 记录运行开始时需要重新生成的语音数量，等于新成功加失败。
        "missing_rows": generated_rows + failed_rows,
        # 记录本轮新生成成功的数量。
        "ok": generated_rows,
        # 记录已有特征验证后跳过重算的数量。
        "skip": reused_rows,
        # 记录本轮处理失败的数量。
        "fail": failed_rows,
        # 记录本轮实际观察到的说话人向量维数。
        "embed_dim_seen": embed_dim_seen,
        # 记录整套特征及其输入 WAV 的汇总指纹。
        "feature_collection": summarize_feature_records(rows, records),
        # 记录以每条语音为单位保存的可核验特征台账。
        "file_records": records,
        # 记录有限数量的失败语音及具体原因。
        "fail_examples": fail_examples,
    }


# 检查说话人嵌入后原子保存数值文件；下面的参数分别限定输入数据和处理约束。
def atomic_save_speaker(path: Path, array: np.ndarray) -> None:
    # 检查说话人嵌入后原子保存数值文件。
    """Validate and atomically publish one speaker embedding."""
    # 准备检查说话人嵌入后原子保存数值文件所需的目标目录，已有目录按给定选项复用。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 当前操作专用的临时文件路径。
    temp_path = None
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在正式目录旁创建唯一临时文件，完整写入并校验后再原子发布。
        with tempfile.NamedTemporaryFile(
            # 临时文件名带原目标文件名，便于定位但不会覆盖已有正式文件。
            prefix=f".{path.name}.",
            # 临时文件使用明确后缀，不让下游误识别为正式特征。
            suffix=".tmp",
            # 临时文件留在目标同一文件系统，支持完整文件原子替换。
            dir=str(path.parent),
            # 先保留关闭后的临时文件，供验证和原子替换使用。
            delete=False,
        # 以下操作通过当前临时文件句柄写入；离开缩进块自动关闭句柄。
        ) as handle:
            # 当前操作专用的临时文件路径。
            temp_path = Path(handle.name)
            # 将已提取的数值数组写入 NPY，禁止保存任意 Python 对象。
            np.save(handle, array, allow_pickle=False)
            # 先将语言运行时缓冲写到文件，供后续同步落盘。
            handle.flush()
            # 要求操作系统同步已写内容，提高中断后的记录可恢复性。
            os.fsync(handle.fileno())
        # 发布前重新读取临时向量，确认文件完整且维数、精度、数值合格。
        validate_speaker_file(temp_path)
        # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
        os.replace(temp_path, path)
    # 无论处理成功或失败，都执行后面的临时资源清理。
    finally:
        # 按当前操作专用的临时文件路径决定是否进入此分支；判断条件为 temp_path is not None。
        if temp_path is not None:
            # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
            try:
                # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
                temp_path.unlink()
            # 捕获检查说话人嵌入后原子保存数值文件中的读取、格式或兼容错误，让调用者获得明确失败原因。
            except FileNotFoundError:
                # 此异常属于允许忽略的清理或兼容情形，后续流程继续执行。
                pass


# 原子保存本次特征提取报告；下面的参数分别限定输入数据和处理约束。
def write_json_report(path: Path, report: Dict) -> None:
    # 原子保存本次特征提取报告。
    """Atomically publish the extraction report."""
    # 准备原子保存本次特征提取报告所需的目标目录，已有目录按给定选项复用。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 当前操作专用的临时文件路径。
    temp_path = None
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在正式目录旁创建唯一临时文件，完整写入并校验后再原子发布。
        with tempfile.NamedTemporaryFile(
            # 当前选择的时间对齐或插值模式。
            mode="w",
            # 来源报告统一使用 UTF-8，保留中文路径与错误说明。
            encoding="utf-8",
            # 临时文件名带原目标文件名，便于定位但不会覆盖已有正式文件。
            prefix=f".{path.name}.",
            # 临时文件使用明确后缀，不让下游误识别为正式特征。
            suffix=".tmp",
            # 临时文件留在目标同一文件系统，支持完整文件原子替换。
            dir=str(path.parent),
            # 先保留关闭后的临时文件，供验证和原子替换使用。
            delete=False,
        # 以下操作通过当前临时文件句柄写入；离开缩进块自动关闭句柄。
        ) as handle:
            # 当前操作专用的临时文件路径。
            temp_path = Path(handle.name)
            # 把已经核对的结构化内容写成 UTF-8 JSON，保存时保留可读的中文及缩进。
            json.dump(report, handle, ensure_ascii=False, indent=2)
            # JSON 文件末尾补换行，便于文本检查和后续工具读取。
            handle.write("\n")
            # 先将语言运行时缓冲写到文件，供后续同步落盘。
            handle.flush()
            # 要求操作系统同步已写内容，提高中断后的记录可恢复性。
            os.fsync(handle.fileno())
        # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
        os.replace(temp_path, path)
    # 无论处理成功或失败，都执行后面的临时资源清理。
    finally:
        # 按当前操作专用的临时文件路径决定是否进入此分支；判断条件为 temp_path is not None。
        if temp_path is not None:
            # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
            try:
                # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
                temp_path.unlink()
            # 捕获原子保存本次特征提取报告中的读取、格式或兼容错误，让调用者获得明确失败原因。
            except FileNotFoundError:
                # 此异常属于允许忽略的清理或兼容情形，后续流程继续执行。
                pass


# 给小样本检查分配独立报告以保护全量来源证据；下面的参数分别限定输入数据和处理约束。
def choose_limited_report_output(
    # 数据处理审计报告的保存位置。
    report_path: Path,
    # 本轮根据限制选出的语音条目。
    selected_rows: int,
    # 仅为诊断而限制处理的样本数。
    limit: int,
# 返回给小样本检查分配独立报告以保护全量来源证据整理的 tuple 结果，内部各项保持前面约定的对应关系。
) -> tuple:
    # 给小样本检查分配独立报告以保护全量来源证据。
    """Keep a debug ``--limit`` run from replacing a larger formal report."""

    # 按仅为诊断而限制处理的样本数决定是否进入此分支；判断条件为 limit is None or limit <= 0 or not report_path.exists()。
    if limit is None or limit <= 0 or not report_path.exists():
        # 本次无需保护更大的旧报告，沿用指定路径并返回保护标志为假。
        return report_path, False
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在受控文件句柄中完成给小样本检查分配独立报告以保护全量来源证据，退出此段后自动关闭文件。
        with report_path.open("r", encoding="utf-8") as handle:
            # 旧正式报告覆盖的总样本数量。
            previous_rows = json.load(handle).get("total_rows")
    # 捕获给小样本检查分配独立报告以保护全量来源证据中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception:
        # 旧正式报告覆盖的总样本数量。
        previous_rows = None
    # 要求读取对象符合给小样本检查分配独立报告以保护全量来源证据预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(previous_rows, int) or previous_rows > selected_rows:
        # 带 limit 标记的独立诊断报告路径。
        debug_path = report_path.with_name(
            # 诊断报告名带实际样本限制数，避免覆盖全量正式报告。
            f"{report_path.stem}.limit-{limit}{report_path.suffix}"
        )
        # 少量诊断使用独立报告名，并明确禁止改写已有正式报告对应的数据。
        return debug_path, True
    # 本次无需保护更大的旧报告，沿用指定路径并返回保护标志为假。
    return report_path, False


# 按特征提取来源定位可追加的进度日志；下面的参数分别限定输入数据和处理约束。
def progress_journal_path(
    # 数据处理审计报告的保存位置。
    report_path: Path,
    # 本轮固定提取器源码、配置和权重的身份记录。
    output_provenance: Dict[str, str],
# 返回按特征提取来源定位可追加的进度日志整理的 Path 结果，内部各项保持前面约定的对应关系。
) -> Path:
    # 按特征提取来源定位可追加的进度日志。
    """Use a provenance-keyed append-only journal for crash-safe resume."""

    # 可信来源的模型参数与训练记录。
    checkpoint = output_provenance["checkpoint_sha256"][:12]
    # 网络、声学参数及实验约束的配置。
    config = output_provenance["config_sha256"][:12]
    # 生成带提取器权重及配置短指纹的日志文件名，不混用不同来源的续作记录。
    return report_path.with_name(
        # 进度日志名绑定权重与配置短指纹，防止不同提取器共用续作记录。
        f".{report_path.name}.progress-{checkpoint}-{config}.jsonl"
    )


# 读取已落盘的逐样本进度并拒绝身份不一致的记录；下面的参数分别限定输入数据和处理约束。
def load_progress_records(
    # 当前待检查或读写的文件路径。
    path: Path,
    # 本轮固定提取器源码、配置和权重的身份记录。
    output_provenance: Dict[str, str],
# 返回读取已落盘的逐样本进度并拒绝身份不一致的记录整理的 Dict[str, Dict[str, object]] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, Dict[str, object]]:
    # 读取已落盘的逐样本进度并拒绝身份不一致的记录。
    """Read complete journal lines; tolerate only a truncated final append."""

    # 检查当前待检查或读写的文件路径是否缺失；缺失时不能继续信任该输入。
    if not path.exists():
        # 没有可复用样本或既有日志时返回空台账，不捏造来源记录。
        return {}
    # 进度日志按行拆分后的文字。
    lines = path.read_text(encoding="utf-8").splitlines()
    # 按进度日志按行拆分后的文字决定是否进入此分支；判断条件为 not lines。
    if not lines:
        # 读取已落盘的逐样本进度并拒绝身份不一致的记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Empty speaker provenance journal: {path}")
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 进度日志首行的版本和模型身份信息。
        header = json.loads(lines[0])
    # 捕获读取已落盘的逐样本进度并拒绝身份不一致的记录中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except json.JSONDecodeError as exc:
        # 读取已落盘的逐样本进度并拒绝身份不一致的记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Corrupt speaker provenance journal header: {path}") from exc
    # 下面几个条件共同检查来源身份和格式；必须按该分支要求同时满足。
    if (
        # 进度第一行必须明确标为日志头。
        header.get("record_type") != "header"
        # 进度日志结构版本必须与当前来源审核规则相同。
        or header.get("report_schema_version") != REPORT_SCHEMA_VERSION
        # 日志必须属于本类特征，不能将 BNF 与身份向量混用。
        or header.get("feature_kind") != FEATURE_KIND
        # 日志绑定的源码、配置和权重必须与当前提取器完全一致。
        or header.get("output_provenance") != output_provenance
    ):
        # 读取已落盘的逐样本进度并拒绝身份不一致的记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Speaker provenance journal header mismatch: {path}")

    # 按语音唯一身份建立的逐文件来源台账。
    records: Dict[str, Dict[str, object]] = {}
    # 跳过首行来源头后，按写入顺序读取逐语音进度记录。
    for index, line in enumerate(lines[1:], start=1):
        # 允许日志中的空白行，不把空行当成一条已完成特征。
        if not line.strip():
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # 当前语音的特征路径与身份记录。
            item = json.loads(line)
        # 捕获读取已落盘的逐样本进度并拒绝身份不一致的记录中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except json.JSONDecodeError as exc:
            # 按当前条目在有序列表中的位置决定是否进入此分支；判断条件为 index == len(lines) - 1。
            if index == len(lines) - 1:
                # 结束当前扫描循环，保留此前已经收集的结果。
                break
            # 读取已落盘的逐样本进度并拒绝身份不一致的记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 报错定位身份特征进度日志中损坏的行，只有末尾未写完整一行可按恢复规则忽略。
                f"Corrupt speaker provenance journal line {index + 1}"
            # 保留 JSON 解析的原始异常和损坏行号，不能把日志中间损坏当作正常中断尾行。
            ) from exc
        # 当前语音的唯一身份。
        key = item.get("key")
        # 当前特征的内容、形状、输入音频及处理动作记录。
        record = item.get("record")
        # 每条进度必须有 feature 类别、字符串语音键和结构化来源记录。
        if item.get("record_type") != "feature" or not isinstance(key, str) or not isinstance(record, dict):
            # 读取已落盘的逐样本进度并拒绝身份不一致的记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 报错定位结构错误的身份进度行，不能把异常记录当作已完成。
                f"Malformed speaker provenance journal line {index + 1}"
            )
        # 按语音唯一身份保存本次核对后的特征来源记录。
        records[key] = record
    # 返回按语音唯一身份建立的逐文件来源台账，供调用方继续处理。
    return records


# 初始化或验证可恢复提取使用的日志头；下面的参数分别限定输入数据和处理约束。
def ensure_progress_journal(
    # 当前待检查或读写的文件路径。
    path: Path,
    # 本轮固定提取器源码、配置和权重的身份记录。
    output_provenance: Dict[str, str],
# 本接口负责初始化或验证可恢复提取使用的日志头，处理结果通过对象状态或文件保存，不额外返回数值。
) -> None:
    # 初始化或验证可恢复提取使用的日志头。
    """Atomically create a journal header, or validate the existing journal."""

    # 仅在当前待检查或读写的文件路径已存在时进入复用或校验分支。
    if path.exists():
        # 已有日志先完整核对身份与已完成行，再允许继续追加。
        load_progress_records(path, output_provenance)
        # 当前分支已经完成必要校验或复用，立即结束函数以免重复写入或提取。
        return
    # 准备初始化或验证可恢复提取使用的日志头所需的目标目录，已有目录按给定选项复用。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 进度日志首行的版本和模型身份信息。
    header = {
        # 记录当前 JSONL 行属于日志头还是单条特征记录。
        "record_type": "header",
        # 记录特征来源报告的结构版本号。
        "report_schema_version": REPORT_SCHEMA_VERSION,
        # 记录BNF 或说话人嵌入的特征类别。
        "feature_kind": FEATURE_KIND,
        # 记录本轮固定提取器源码、配置和权重的身份记录。
        "output_provenance": output_provenance,
    }
    # 当前操作专用的临时文件路径。
    temp_path = None
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在正式目录旁创建唯一临时文件，完整写入并校验后再原子发布。
        with tempfile.NamedTemporaryFile(
            # 当前选择的时间对齐或插值模式。
            mode="w",
            # 来源报告统一使用 UTF-8，保留中文路径与错误说明。
            encoding="utf-8",
            # 临时文件名带原目标文件名，便于定位但不会覆盖已有正式文件。
            prefix=f".{path.name}.",
            # 临时文件使用明确后缀，不让下游误识别为正式特征。
            suffix=".tmp",
            # 临时文件留在目标同一文件系统，支持完整文件原子替换。
            dir=str(path.parent),
            # 先保留关闭后的临时文件，供验证和原子替换使用。
            delete=False,
        # 以下操作通过当前临时文件句柄写入；离开缩进块自动关闭句柄。
        ) as handle:
            # 当前操作专用的临时文件路径。
            temp_path = Path(handle.name)
            # 把固定提取器身份写成日志首行，字段顺序稳定并保留中文。
            handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
            # 先将语言运行时缓冲写到文件，供后续同步落盘。
            handle.flush()
            # 要求操作系统同步已写内容，提高中断后的记录可恢复性。
            os.fsync(handle.fileno())
        # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
        os.replace(temp_path, path)
    # 无论处理成功或失败，都执行后面的临时资源清理。
    finally:
        # 按当前操作专用的临时文件路径决定是否进入此分支；判断条件为 temp_path is not None。
        if temp_path is not None:
            # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
            try:
                # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
                temp_path.unlink()
            # 捕获初始化或验证可恢复提取使用的日志头中的读取、格式或兼容错误，让调用者获得明确失败原因。
            except FileNotFoundError:
                # 此异常属于允许忽略的清理或兼容情形，后续流程继续执行。
                pass


# 在单条特征发布后同步写入可恢复的来源证明；下面的参数分别限定输入数据和处理约束。
def append_progress_record(
    # 当前待检查或读写的文件路径。
    path: Path,
    # 当前语音的唯一身份。
    key: str,
    # 当前特征的内容、形状、输入音频及处理动作记录。
    record: Dict[str, object],
# 本接口负责在单条特征发布后同步写入可恢复的来源证明，处理结果通过对象状态或文件保存，不额外返回数值。
) -> None:
    # 在单条特征发布后同步写入可恢复的来源证明。
    """Durably attest one published feature before moving to the next utterance."""

    # 当前语音的特征路径与身份记录。
    item = {"record_type": "feature", "key": key, "record": record}
    # 在受控文件句柄中完成在单条特征发布后同步写入可恢复的来源证明，退出此段后自动关闭文件。
    with path.open("a", encoding="utf-8") as handle:
        # 把一条已成功发布的特征来源追加为独立 JSON 行，随后同步落盘。
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        # 先将语言运行时缓冲写到文件，供后续同步落盘。
        handle.flush()
        # 要求操作系统同步已写内容，提高中断后的记录可恢复性。
        os.fsync(handle.fileno())


# 从上次全量报告中恢复可验证的特征记录；下面的参数分别限定输入数据和处理约束。
def load_prior_feature_records(
    # 数据处理审计报告的保存位置。
    report_path: Path,
    # 中断续作使用的逐条进度日志。
    journal_path: Path,
    # 本轮固定提取器源码、配置和权重的身份记录。
    output_provenance: Dict[str, str],
# 返回从上次全量报告中恢复可验证的特征记录整理的 Dict[str, Dict[str, object]] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, Dict[str, object]]:
    # 从上次全量报告中恢复可验证的特征记录。
    """Combine a prior formal report with later crash-safe progress records."""

    # 按语音唯一身份建立的逐文件来源台账。
    records: Dict[str, Dict[str, object]] = {}
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在受控文件句柄中完成从上次全量报告中恢复可验证的特征记录，退出此段后自动关闭文件。
        with report_path.open("r", encoding="utf-8") as handle:
            # 前一次完整来源报告。
            previous = json.load(handle)
    # 捕获从上次全量报告中恢复可验证的特征记录中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception:
        # 前一次完整来源报告。
        previous = None
    # 下面几个条件共同检查来源身份和格式；必须按该分支要求同时满足。
    if (
        # 旧报告必须能解析为字段字典。
        isinstance(previous, dict)
        # 旧报告结构版本必须匹配当前逐文件审计格式。
        and previous.get("report_schema_version") == REPORT_SCHEMA_VERSION
        # 旧报告必须属于当前特征类型。
        and previous.get("feature_kind") == FEATURE_KIND
        # 旧报告中的源码提交、配置及模型权重必须与当前一致。
        and previous.get("output_provenance") == output_provenance
        # 旧报告必须带有逐语音文件台账，单个总体计数不能证明来源。
        and isinstance(previous.get("file_records"), dict)
    ):
        # 将来源一致的旧正式报告台账先纳入恢复集合。
        records.update(previous["file_records"])
    # 再覆盖合并中断前新完成的持久化进度，保留较新的逐文件证据。
    records.update(load_progress_records(journal_path, output_provenance))
    # 返回按语音唯一身份建立的逐文件来源台账，供调用方继续处理。
    return records


# 读取此次真正需要提取特征的清单并控制小样本检查范围；下面的参数分别限定输入数据和处理约束。
def load_manifest_rows(manifest_path: Path, limit: int = -1) -> List[Dict[str, str]]:
    # 清单必须存在。
    if not manifest_path.exists():
        # 读取此次真正需要提取特征的清单并控制小样本检查范围未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # rows 保存去重后的行；seen 保存唯一键。
    rows = []
    # 已经读过的语音身份集合。
    seen = set()

    # 自动打开/关闭 UTF-8 CSV。
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        # 表头成为字典键。
        reader = csv.DictReader(f)
        # 逐行读取。
        for row in reader:
            # speaker+utt_id 唯一确定语音。
            key = (row["speaker"], row["utt_id"])
            # 跳过重复。
            if key in seen:
                # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
                continue
            # 记录唯一键。
            seen.add(key)
            # 保存行。
            rows.append(row)

    # 正数 limit 用于试跑前 N 条；-1 全量。
    if limit is not None and limit > 0:
        # 本次选定的语音清单。
        rows = rows[:limit]

    # 返回行列表。
    return rows


# 区分需提取的文件与可核验后重用的已有特征；下面的参数分别限定输入数据和处理约束。
def filter_missing_rows(
    # 本次选定的语音清单。
    rows: List[Dict[str, str]],
    # 特征输出目录。
    out_dir: Path,
    # 是否明确要求重新计算已有特征。
    force: bool = False,
# 返回区分需提取的文件与可核验后重用的已有特征整理的 List[Dict[str, str]] 结果，内部各项保持前面约定的对应关系。
) -> List[Dict[str, str]]:
    # 只跳过能重新加载且严格为 [256] 有限数的现有文件。
    missing = []
    # 遍历清单。
    for row in rows:
        # 取 speaker。
        speaker = row["speaker"]
        # 取 utterance ID。
        utt_id = row["utt_id"]
        # 拼标准输出路径。
        out_path = out_dir / speaker / f"{utt_id}.npy"
        # 按是否明确要求重新计算已有特征决定是否进入此分支；判断条件为 force or not out_path.exists()。
        if force or not out_path.exists():
            # 把缺失或校验失败的语音加入待重新提取清单。
            missing.append(row)
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # 已有输出先验证为合法 256 维 float32 身份向量。
            validate_speaker_file(out_path)
        # 捕获区分需提取的文件与可核验后重用的已有特征中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except ValueError:
            # 把缺失或校验失败的语音加入待重新提取清单。
            missing.append(row)
    # 返回待处理列表。
    return missing


# 去掉冗余单元素轴并核对说话人嵌入长度；下面的参数分别限定输入数据和处理约束。
def standardize_embed_shape(
    # 旧身份编码器返回的向量数组，允许有冗余长度为 1 的轴。
    x: np.ndarray,
    # 特征文件预期的维数。
    expected_dim: int = EXPECTED_EMBED_DIM,
# 返回去掉冗余单元素轴并核对说话人嵌入长度整理的 np.ndarray 结果，内部各项保持前面约定的对应关系。
) -> np.ndarray:
    # 去掉冗余单元素轴并核对说话人嵌入长度。
    """
    Speaker embedding should be a 1D vector.
    Accept shapes like:
      [D]
      [1, D]
      [D, 1]
    and convert to [D].
    """
    # 确保输入是 NumPy 数组。
    x = np.asarray(x)

    # 已是一维 [D] 就直接使用。
    if x.ndim == 1:
        # 身份输出已是一维向量，直接保留。
        y = x
    # [1,D] 取唯一一行变 [D]。
    elif x.ndim == 2 and x.shape[0] == 1:
        # 移除 [1,256] 输出中多余的首轴，得到 256 维向量。
        y = x[0]
    # [D,1] 取唯一一列变 [D]。
    elif x.ndim == 2 and x.shape[1] == 1:
        # 移除 [256,1] 输出中多余的末轴，得到 256 维向量。
        y = x[:, 0]
    # 其他形状无法明确解释，拒绝猜测。
    else:
        # 去掉冗余单元素轴并核对说话人嵌入长度未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Unexpected speaker embedding shape: {x.shape}")

    # 该已审计 encoder 的输出必须严格是 [256]。
    if y.ndim != 1 or y.shape != (expected_dim,):
        # 去掉冗余单元素轴并核对说话人嵌入长度未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 输出去掉冗余轴后身份向量的实际形状，与 256 维要求比较。
            f"Speaker embedding must have shape [{expected_dim}], got {y.shape}"
        )

    # 排除 NaN/Inf。
    if not np.isfinite(y).all():
        # 去掉冗余单元素轴并核对说话人嵌入长度未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError("Speaker embedding contains NaN or Inf")

    # 统一 float32 后返回。
    return y.astype(np.float32)


# 保留旧编码器音频处理参数并适配现有 librosa 与 NumPy 接口；下面的参数分别限定输入数据和处理约束。
def configure_legacy_audio_compat(speaker_audio):
    # 保留旧编码器音频处理参数并适配现有 librosa 与 NumPy 接口。
    """Adapt the frozen legacy speaker encoder to current librosa/NumPy APIs.

    The numerical recipe is unchanged: 16-kHz resampling, the legacy volume
    normalization and VAD parameters, and the original 40-channel mel setup.
    Only removed keyword/boolean APIs are replaced locally.
    """

    # 按旧编码器 VAD 规则筛除长静音区间；下面的参数分别限定输入数据和处理约束。
    def trim_long_silences(wav):
        # 每个 VAD 分析窗包含的采样点数。
        samples_per_window = (
            # 将旧编码器规定的 VAD 窗长毫秒数乘以采样率，准备换算为采样点数。
            speaker_audio.vad_window_length * speaker_audio.sampling_rate
        # 毫秒换成秒并向下取整，得到每个 VAD 窗的整数采样点数。
        ) // 1000
        # 读取后的单声道浮点音频。
        wav = wav[: len(wav) - (len(wav) % samples_per_window)]

        # 为 VAD 转换成 16 位整数格式的波形字节。
        pcm_wave = speaker_audio.struct.pack(
            # 给 struct 指定当前波形采样点数个 16 位整数，生成 VAD 接受的 PCM 字节。
            "%dh" % len(wav),
            # 把浮点波形放大、四舍五入并转为 16 位整数，符合 WebRTC VAD 输入要求。
            *(np.round(wav * speaker_audio.int16_max)).astype(np.int16),
        )
        # 按旧说话人提取器固定强度建立的语音活动检测器。
        vad = speaker_audio.webrtcvad.Vad(mode=3)
        # 各分析窗是否含语音的布尔标记。
        voice_flags = []
        # 按不重叠的 VAD 窗遍历整段参考语音。
        for window_start in range(0, len(wav), samples_per_window):
            # 当前 VAD 窗结束后的首个采样位置。
            window_end = window_start + samples_per_window
            # 为当前窗口保存一个是否含语音的判断结果。
            voice_flags.append(
                # 调用旧编码器使用的 WebRTC 语音活动检测规则。
                vad.is_speech(
                    # 16 位 PCM 每采样点占两个字节，按字节范围截取当前 VAD 窗。
                    pcm_wave[window_start * 2 : window_end * 2],
                    # 音频文件实际或预期的采样率。
                    sample_rate=speaker_audio.sampling_rate,
                )
            )

        # 各分析窗是否含语音的布尔标记。
        voice_flags = np.asarray(voice_flags)
        # 语音活动标记移动平均使用的窗口宽度。
        width = speaker_audio.vad_moving_average_width
        # 边缘补齐后的语音活动标记序列。
        padded = np.concatenate(
            # 在语音活动序列两端补零，使滑动平均后保持原窗口数量。
            (np.zeros((width - 1) // 2), voice_flags, np.zeros(width // 2))
        )
        # 用于快速计算滑窗平均的语音标记累积和。
        cumulative = np.cumsum(padded, dtype=float)
        # 用两个累积和之差求每个平滑窗口的语音标记总数。
        cumulative[width:] = cumulative[width:] - cumulative[:-width]
        # 将语音标记滑窗平均后四舍五入为布尔掩码，np.bool_ 仅替代新版 NumPy 删除的旧别名。
        audio_mask = np.round(cumulative[width - 1 :] / width).astype(np.bool_)
        # 按旧 VAD 的停顿容忍规则扩展语音标记，保留短静音连接。
        audio_mask = speaker_audio.binary_dilation(
            # 保留语音及允许短停顿的采样点掩码。
            audio_mask,
            # 按旧编码器允许的最大短静音长度扩展语音区域，保留语音之间的短停顿。
            np.ones(speaker_audio.vad_max_silence_length + 1),
        )
        # 将每个 VAD 窗的一位保留标记重复到对应采样点数，变成逐采样掩码。
        audio_mask = np.repeat(audio_mask, samples_per_window)
        # 仅返回保留区域的波形，删除过长静音；该处理只用于身份向量提取。
        return wav[audio_mask]

    # 重采样并按旧说话人编码器规则处理音量和静音；下面的参数分别限定输入数据和处理约束。
    def preprocess_wav(fpath_or_wav, source_sr=None):
        # 输入为路径时先读音频，已经是波形数组时直接使用。
        if isinstance(fpath_or_wav, (str, Path)):
            # 按文件真实采样率读取参考音频，此处先不自动重采样。
            wav, source_sr = speaker_audio.librosa.load(fpath_or_wav, sr=None)
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 读取后的单声道浮点音频。
            wav = fpath_or_wav

        # 按输入波形的原采样率决定是否进入此分支；判断条件为 source_sr is not None and source_sr != speaker_audio.sampling_rate。
        if source_sr is not None and source_sr != speaker_audio.sampling_rate:
            # 把参考音频重采样为固定身份编码器要求的 16 kHz，沿用原数值处理方案。
            wav = speaker_audio.librosa.resample(
                # 待变换的单声道浮点波形。
                y=wav,
                # 重采样前的输入音频采样率。
                orig_sr=source_sr,
                # 本次重采样要求的输出采样率。
                target_sr=speaker_audio.sampling_rate,
            )
        # 沿用 GE2E 预处理的音量标准，只提高过小音量，不等同于 VoiceGrad 的逐 mel 频带 zscore。
        wav = speaker_audio.normalize_volume(
            # 读取后的单声道浮点音频。
            wav,
            # 使用固定 GE2E 提取器的目标音量级，保持其预训练输入分布。
            speaker_audio.audio_norm_target_dBFS,
            # 沿用旧编码器规则，只提高过小音量，不压低已经更响的音频。
            increase_only=True,
        )
        # 音量处理后按固定 VAD 规则去除长静音，再交给身份编码器。
        return trim_long_silences(wav)

    # 计算旧说话人编码器使用的 40 维功率 mel；下面的参数分别限定输入数据和处理约束。
    def wav_to_mel_spectrogram(wav):
        # 计算固定身份编码器使用的 40 维功率 mel；这与 VoiceGrad 的 80 维自然对数 mel 分开处理。
        frames = speaker_audio.librosa.feature.melspectrogram(
            # 待变换的单声道浮点波形。
            y=wav,
            # 波形采样率。
            sr=speaker_audio.sampling_rate,
            # 每个时窗的 FFT 点数。
            n_fft=int(
                # 使用说话人编码器的 16 kHz 采样率，将毫秒窗长或步长转换为采样点。
                speaker_audio.sampling_rate
                # 乘以旧编码器规定的 mel 分析窗长，保持它自己的前端设置。
                * speaker_audio.mel_window_length
                # 将毫秒单位除以 1000 转成秒，再取整数采样点数。
                / 1000
            ),
            # 相邻频谱窗口向前移动的样本数。
            hop_length=int(
                # 使用说话人编码器的 16 kHz 采样率，将毫秒窗长或步长转换为采样点。
                speaker_audio.sampling_rate
                # 乘以旧编码器规定的 mel 帧移，保持它自己的时间采样间隔。
                * speaker_audio.mel_window_step
                # 将毫秒单位除以 1000 转成秒，再取整数采样点数。
                / 1000
            ),
            # mel 频带数，本实验为 80。
            n_mels=speaker_audio.mel_n_channels,
        )
        # 返回时间×40 频带的单精度身份编码器 mel；与 VoiceGrad 的 80 维自然对数 mel 分工不同。
        return frames.astype(np.float32).T

    # 仅替换旧模块已失效的布尔 API 兼容入口，保留原 VAD 参数和数值步骤。
    speaker_audio.trim_long_silences = trim_long_silences
    # 将新版库可接受的关键字调用接入原说话人预处理入口。
    speaker_audio.preprocess_wav = preprocess_wav
    # 给旧编码器接上兼容新版 librosa 的 40 维 mel 计算接口。
    speaker_audio.wav_to_mel_spectrogram = wav_to_mel_spectrogram
    # 返回兼容后的身份音频预处理函数，供主提取循环使用。
    return preprocess_wav


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main():
    # 创建命令行解析器。
    parser = argparse.ArgumentParser()
    # 下面参数依次覆盖清单、输出、报告、旧代码目录、权重和试跑数量。
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--legacy_bnf_root", type=str, default=str(LEGACY_BNF_ROOT))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--spk_encoder_ckpt", type=str, default=str(DEFAULT_SPK_ENCODER_CKPT))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--limit", type=int, default=-1, help="For sanity check. Use -1 for full run.")
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 显式要求重算已有特征；普通运行先尝试验证后复用。
        "--force",
        # 该文件本轮为新生成还是经过验证后复用。
        action="store_true",
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="Re-extract every selected embedding even when a valid output exists.",
    )
    # 从终端读取实际参数。
    args = parser.parse_args()

    # 把路径字符串转换为 Path。
    manifest_path = Path(args.manifest)
    # 特征输出目录。
    out_dir = Path(args.out_dir)
    # 数据处理审计报告的保存位置。
    report_path = Path(args.report)
    # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
    legacy_bnf_root = Path(args.legacy_bnf_root)
    # 预先固定的 GE2E 说话人编码器权重。
    spk_encoder_ckpt = Path(args.spk_encoder_ckpt)
    # 说话人编码器的预处理和结构参数。
    speaker_config = legacy_bnf_root / "speaker_encoder" / "hparams.py"

    # 创建输出和报告目录。
    out_dir.mkdir(parents=True, exist_ok=True)
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 在 import/建立 encoder 前强制核对代码 commit 和权重 SHA-256。
    provenance = verify_legacy_provenance(
        # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
        legacy_bnf_root,
        # 预先固定的 GE2E 说话人编码器权重。
        spk_encoder_ckpt,
        # 说话人编码器的预处理和结构参数。
        speaker_config,
    )

    # 读取清单。
    rows = load_manifest_rows(manifest_path, limit=args.limit)
    # 损坏、非 [256]、含 NaN/Inf 的旧输出会自动重做。
    rows_to_process = filter_missing_rows(rows, out_dir, force=args.force)
    # 本轮待重新计算的语音身份集合。
    process_keys = {feature_key(row) for row in rows_to_process}
    # 本轮通过来源审核后复用的语音条目或数量。
    reused_rows = [row for row in rows if feature_key(row) not in process_keys]
    # 检查少量诊断是否会覆盖更大的正式来源报告，并记录保护要求。
    _report_output_path, protects_larger_report = choose_limited_report_output(
        # 数据处理审计报告的保存位置。
        report_path,
        # 传入本轮实际选中的样本数，供全量覆盖或少量诊断比较。
        len(rows),
        # 传入用户的诊断条数限制；非正值表示全量处理。
        args.limit,
    )
    # 中断续作使用的逐条进度日志。
    journal_path = progress_journal_path(report_path, provenance)
    # Close the crash window between atomic feature publication and journal
    # fsync.  A valid array without a durable record is regenerated by itself;
    # it is never attributed to the current encoder merely from its shape.
    # 合并旧正式报告和中断进度，找到每个已有特征的可靠来源记录。
    prior_records = load_prior_feature_records(
        # 数据处理审计报告的保存位置。
        report_path,
        # 中断续作使用的逐条进度日志。
        journal_path,
        # 固定源码提交、配置和权重组成的提取器身份。
        provenance,
    )
    # 虽有特征文件但缺少可信来源记录、必须重算的语音。
    unattested_rows = [
        # 保留当前语音清单行，供下面的条件筛选。
        row
        # 逐条检查暂定可复用的语音是否确有来源证据。
        for row in reused_rows
        # 要求读取对象符合串联本文件的数据检查、模型处理和结果保存流程预期的数据结构，否则不能继续信任其中字段。
        if not isinstance(prior_records.get(feature_key(row)), dict)
    ]
    # 按虽有特征文件但缺少可信来源记录、必须重算的语音决定是否进入此分支；判断条件为 unattested_rows。
    if unattested_rows:
        # 无可靠来源的已有文件转入重算，防止把旧未知模型输出混入正式集合。
        rows_to_process.extend(unattested_rows)
        # 缺少来源证据而被转入重新提取的语音身份集合。
        unattested_keys = {feature_key(row) for row in unattested_rows}
        # 本轮通过来源审核后复用的语音条目或数量。
        reused_rows = [
            # 从复用列表中排除刚刚转入重算的语音，避免重复计数。
            row for row in reused_rows if feature_key(row) not in unattested_keys
        ]
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print(
            # 日志提示发现缺少持久来源证据的已有特征，正在重新生成。
            "[RECOVER] Regenerating "
            # 日志列出缺少持久来源证据而重算的身份文件数。
            f"{len(unattested_rows)} speaker file(s) published without a durable attestation."
        )
    # 只有上一份 schema-v2 报告逐文件证明内容和模型来源时才允许复用。
    reused_records = verify_reused_feature_records(
        # 本轮通过来源审核后复用的语音条目或数量。
        reused_rows,
        # 特征输出目录。
        out_dir,
        # 数据处理审计报告的保存位置。
        report_path,
        # 中断续作使用的逐条进度日志。
        journal_path,
        # 固定源码提交、配置和权重组成的提取器身份。
        provenance,
    )
    # 按当前少量诊断必须保护已有更大规模正式报告的标志决定是否进入此分支；判断条件为 protects_larger_report。
    if protects_larger_report:
        # 按本轮尚需通过提取模型计算的语音决定是否进入此分支；判断条件为 rows_to_process。
        if rows_to_process:
            # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 报错说明少量诊断会改动全量正式身份向量集合。
                "A limited run would modify canonical speaker embeddings covered by "
                # 指出受保护正式身份报告的位置，提示全量重算或独立诊断。
                f"a larger formal report ({report_path}). Run the full manifest, or "
                # 少量身份提取诊断必须同时隔离输出向量目录和报告。
                "use both a separate --out_dir and --report for the debug run."
            )
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print(f"[INFO] Preserving larger provenance report {report_path}.")

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Manifest           = {manifest_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Out dir            = {out_dir}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Legacy BNF root    = {legacy_bnf_root}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Speaker ckpt       = {spk_encoder_ckpt}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Speaker config     = {speaker_config}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Total rows         = {len(rows)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Rows to process    = {len(rows_to_process)}")

    # 所有输出存在时不加载 encoder，写报告后退出。
    if len(rows_to_process) == 0:
        # 按当前少量诊断必须保护已有更大规模正式报告的标志决定是否进入此分支；判断条件为 protects_larger_report。
        if protects_larger_report:
            # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
            print(
                # 少量核对通过时只报告结果，原全量身份来源报告保持不变。
                "\n[DONE] Selected speaker embeddings match the preserved full report."
            )
            # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
            print(f"  report = {report_path} (unchanged)")
            # 当前分支已经完成必要校验或复用，立即结束函数以免重复写入或提取。
            return
        # 本次处理的数据来源和完成情况报告。
        report = build_extraction_report(
            # 本次实际使用的清单文件。
            manifest_path=manifest_path,
            # 本次选定的语音清单。
            rows=rows,
            # 特征输出目录。
            out_dir=out_dir,
            # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
            legacy_bnf_root=legacy_bnf_root,
            # 预先固定的 GE2E 说话人编码器权重。
            spk_encoder_ckpt=spk_encoder_ckpt,
            # 说话人编码器的预处理和结构参数。
            speaker_config=speaker_config,
            # 本轮固定提取器源码、配置和权重的身份记录。
            output_provenance=provenance,
            # 是否明确要求重新计算已有特征。
            force=args.force,
            # 本轮新提取成功的语音数量。
            generated_rows=0,
            # 本轮通过来源审核后复用的语音条目或数量。
            reused_rows=len(reused_rows),
            # 本轮提取失败的语音数量。
            failed_rows=0,
            # 按语音唯一身份建立的逐文件来源台账。
            records=reused_records,
            # 有限数量的失败语音及具体原因。
            fail_examples=[],
        )
        # 报告补充说明本轮全部文件均由历史来源与内容核验后复用。
        report["note"] = (
            # 注明本轮所有身份向量都已逐文件对照旧报告完成内容核验。
            "All speaker embeddings were content-verified against the prior report."
        )
        # 将本轮完整状态和逐文件台账原子保存为正式报告。
        write_json_report(report_path, report)
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
            journal_path.unlink()
        # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except FileNotFoundError:
            # 此异常属于允许忽略的清理或兼容情形，后续流程继续执行。
            pass

        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print("\n[DONE] All target speaker embedding files already exist.")
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print(f"  report = {report_path}")
        # 提前结束 main。
        return

    # 提取前准备绑定本次固定模型身份的可恢复进度日志。
    ensure_progress_journal(journal_path, provenance)

    # Import legacy speaker encoder code
    # 真正需要提取时再载入进度条工具，显示逐语音完成情况。
    from tqdm import tqdm

    # 把旧工程放到模块搜索路径最前面。
    sys.path.insert(0, str(legacy_bnf_root))
    # 导入 encoder 类。
    from speaker_encoder.voice_encoder import SpeakerEncoder
    # 导入与该 encoder 训练设置完全配套的音频预处理函数。
    from speaker_encoder import audio as speaker_audio

    # The frozen encoder predates keyword-only librosa APIs and np.bool removal.
    # 建立兼容新库 API 的旧身份编码器预处理入口，保留其音量、VAD 和 40 维 mel 配方。
    preprocess_wav = configure_legacy_audio_compat(speaker_audio)

    # 建立模型并加载 checkpoint；此处不继续训练 encoder。
    encoder = SpeakerEncoder(spk_encoder_ckpt)

    # 初始化计数器、错误列表和首次观察到的 embedding 维数。
    ok = 0
    # 已有特征验证后跳过重算的数量。
    skip = len(rows) - len(rows_to_process)
    # 本轮处理失败的数量。
    fail = 0
    # 有限数量的失败语音及具体原因。
    fail_examples = []
    # 本轮实际观察到的说话人向量维数。
    embed_dim_seen = None
    # 新生成与可信复用记录最终合并成完整集合账本。
    records: Dict[str, Dict[str, object]] = dict(reused_records)

    # 逐条提取；tqdm 仅显示进度。
    for row in tqdm(rows_to_process, total=len(rows_to_process), desc="Extract spk emb"):
        # 当前说话人。
        speaker = row["speaker"]
        # 当前 utterance ID。
        utt_id = row["utt_id"]
        # 输入 wav 路径来自 manifest。
        wav_path = Path(row["wav22k_path"])
        # 输出路径按 speaker 分目录。
        out_path = out_dir / speaker / f"{utt_id}.npy"

        # 捕获单条失败，使前面成功结果仍被保存并能续跑。
        try:
            # 输入必须存在。
            if not wav_path.exists():
                # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
                raise FileNotFoundError(f"wav not found: {wav_path}")

            # 必须使用旧 encoder 自带的 preprocess_wav，保证采样率/静音处理与其训练一致。
            wav = preprocess_wav(wav_path)
            # 把整段参考语音压成固定长度身份向量。
            embed = encoder.embed_utterance(wav)
            # 统一向量形状和 dtype。
            embed = standardize_embed_shape(embed)

            # 只在第一条成功输出时记录实际维数，写进报告用于核对 256。
            if embed_dim_seen is None:
                # 本轮实际观察到的说话人向量维数。
                embed_dim_seen = int(embed.shape[0])

            # 同目录临时写入，重新加载验证 [256]/finite 后原子替换。
            atomic_save_speaker(out_path, embed)
            # 当前特征的内容、形状、输入音频及处理动作记录。
            record = make_feature_record(
                # 当前样本的目标特征文件。
                out_path,
                # 当前语句经过编码器得到的身份向量。
                embed,
                # 实际送入特征提取器的原始 WAV 路径。
                input_wav_path=wav_path,
                # 该文件本轮为新生成还是经过验证后复用。
                action="generated",
            )
            # 用语音唯一键登记刚刚成功保存的特征及其输入音频指纹。
            records[feature_key(row)] = record
            # 每成功一条就同步追加来源证据，突然中断后可准确恢复。
            append_progress_record(journal_path, feature_key(row), record)
            # 成功数加一。
            ok += 1

        # 捕获任意异常。
        except Exception as e:
            # 失败数加一。
            fail += 1
            # 只保留前 20 个详细例子，防止报告过大。
            if len(fail_examples) < 20:
                # 记录失败语音编号、输入路径和异常说明，方便后续定向补算。
                fail_examples.append({
                    # 记录当前语音的唯一身份。
                    "key": f"{speaker}/{utt_id}",
                    # 记录当前语音的音频文件。
                    "wav_path": str(wav_path),
                    # 记录本次操作捕获的具体错误。
                    "error": repr(e),
                })

    # 汇总本次输入输出、权重、计数、维数和逐文件内容账本。
    report = build_extraction_report(
        # 本次实际使用的清单文件。
        manifest_path=manifest_path,
        # 本次选定的语音清单。
        rows=rows,
        # 特征输出目录。
        out_dir=out_dir,
        # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
        legacy_bnf_root=legacy_bnf_root,
        # 预先固定的 GE2E 说话人编码器权重。
        spk_encoder_ckpt=spk_encoder_ckpt,
        # 说话人编码器的预处理和结构参数。
        speaker_config=speaker_config,
        # 本轮固定提取器源码、配置和权重的身份记录。
        output_provenance=provenance,
        # 是否明确要求重新计算已有特征。
        force=args.force,
        # 本轮新提取成功的语音数量。
        generated_rows=ok,
        # 本轮通过来源审核后复用的语音条目或数量。
        reused_rows=skip,
        # 本轮提取失败的语音数量。
        failed_rows=fail,
        # 按语音唯一身份建立的逐文件来源台账。
        records=records,
        # 有限数量的失败语音及具体原因。
        fail_examples=fail_examples,
        # 本轮实际观察到的说话人向量维数。
        embed_dim_seen=embed_dim_seen,
    )

    # 将本轮完整状态和逐文件台账原子保存为正式报告。
    write_json_report(report_path, report)
    # 按本次处理的数据来源和完成情况报告决定是否进入此分支；判断条件为 report["status"] == "complete"。
    if report["status"] == "complete":
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
            journal_path.unlink()
        # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except FileNotFoundError:
            # 此异常属于允许忽略的清理或兼容情形，后续流程继续执行。
            pass

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("\n[DONE] Speaker embedding extraction finished.")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  ok   = {ok}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  skip = {skip}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  fail = {fail}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  embed_dim_seen = {embed_dim_seen}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  report = {report_path}")

    # 终端只展示前 10 个失败例子。
    if fail_examples:
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print("\n[FAIL EXAMPLES]")
        # 终端展示最多 10 个失败例子，完整计数仍保存在报告。
        for item in fail_examples[:10]:
            # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
            print(item)

    # 任意失败返回非零退出码，防止残缺 embedding 集合进入训练。
    if fail:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise SystemExit(
            # 最终报错说明身份特征未全量提取成功及详细报告位置。
            f"Speaker embedding extraction failed for {fail} utterances; see {report_path}."
        )


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 直接执行才开始提取。
    main()
