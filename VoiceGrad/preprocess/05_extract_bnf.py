#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 使用固定外部提取器产生 144 维内容 BNF，并保存逐文件音频和模型来源证据。
"""
Extract 144-dim BNF for FastVoiceGrad teacher on VCTK.

Key idea:
- Reuse a separately supplied legacy conformer PPG/BNF model implementation.
- But DO NOT directly call the old 1_compute_ctc_att_bnf.py script,
  because that script:
    1) depends on glob2
    2) hardcodes CMU speaker names only
- Instead, this script imports load_ppg_model from the old project
  and performs VCTK-compatible extraction directly.

Input:
  data/metadata/all_manifest.csv

Output:
  data/bnf/<speaker>/<utt_id>.npy

Behavior:
- Read wav path from manifest
- If sr != 16000, resample to 16000
- Feed waveform into the same old PPG model
- Standardize output shape to [144, T]
- Resume-friendly: validate existing files and rebuild missing/invalid files

Example:
  python3 preprocess/05_extract_bnf.py --limit 20
  python3 preprocess/05_extract_bnf.py
"""

# =============================================================================
# 预处理第 05 步：提取 144 维 BNF（说话内容条件）
# =============================================================================
# BNF 回答“这句话说了什么”，speaker embedding 回答“是谁在说”。转换时用 source
# 的 BNF 和 target 的 speaker embedding，才能保持内容、替换音色。
# 论文说使用 BNE；实验室旧工程已有训练好的 Conformer PPG/BNF 模型，所以这里
# 只加载旧模型做前向，不重新训练。旧模型要求 16 kHz，故内存中临时重采样；
# data/wav22k 原文件不会被覆盖。最终每条保存 [144,T_bnf] float32。

# --------------------------- 导入包逐项解释 ---------------------------
# csv 读取 all_manifest；json 写报告；sys.path 用来临时找到旧工程 Python 模块；
# argparse 解析模型/路径参数；Path 处理文件；Dict/List 是类型提示。
import csv
# 保存或读取实验设置、数据指纹及处理进度的结构化记录。
import json
# hashlib 审核外部模型/配置；tempfile 用于同目录原子写入。
import hashlib
# 为待发布的文件创建不重名临时路径，减少中断或并发写入的干扰。
import tempfile
# 处理解释器路径和旧特征提取器模块的导入位置。
import sys
# 定义终端参数，使数据目录、检查点和运行选项明确记录。
import argparse
# os.environ 读取两个可移植路径变量，不在公开代码中写实验室绝对路径。
import os
# 调用已有命令行工具，并检查外部程序是否正常完成。
import subprocess
# 用路径对象定位数据、权重和报告，减少手工拼接目录的错误。
from pathlib import Path
# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Dict, List

# NumPy 整理并保存 BNF。其他推理依赖在来源审核通过后延迟导入。
import numpy as np


# 当前工程根路径：环境变量优先，否则使用仓库根目录。
ROOT = Path(
    # 优先读取显式工程根目录；没有环境变量时回到本文件向上两级的仓库目录。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[1])
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()
# 旧 BNF 工程含第三方/历史模型代码与权重，不应提交到公开 GitHub。请把它放在
# 任意目录并设置 VOICEGRAD_LEGACY_BNF_ROOT；未设置时给出一个清晰的占位默认值，
# 后面的文件存在检查会告诉读者缺少哪个文件。
LEGACY_BNF_ROOT = Path(
    # 优先读取外部提取器目录环境变量，未设置时使用工程 external 子目录。
    os.environ.get("VOICEGRAD_LEGACY_BNF_ROOT", ROOT / "external" / "legacy_compute_bnf")
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()

# 默认读取全部实验语音清单。
DEFAULT_MANIFEST = ROOT / "data" / "metadata" / "all_manifest.csv"
# 每条 BNF 输出目录。
DEFAULT_BNF_DIR = ROOT / "data" / "bnf"
# 提取结果 JSON 报告路径。
DEFAULT_REPORT = ROOT / "exp" / "logs" / "bnf_extraction_report.json"

# 旧 Conformer 的结构配置文件。
DEFAULT_TRAIN_CONFIG = LEGACY_BNF_ROOT / "conformer_ppg_model" / "en_conformer_ctc_att" / "config.yaml"
# 与上述结构配套的第 24 epoch 模型权重。
DEFAULT_MODEL_FILE = LEGACY_BNF_ROOT / "conformer_ppg_model" / "en_conformer_ctc_att" / "24epoch.pth"

# 旧 BNF 模型固定接收 16 kHz 波形。
SAMPLE_RATE = 16000
# 输出特征轴必须严格是 144 维。
EXPECTED_BNF_DIM = 144
# 严格锁定本复现已审计的 ppg-vc 版本和 BNF 权重。
EXPECTED_LEGACY_COMMIT = "b59cb9862cf4b82a3bdb589950e25cab85fc9b03"
# 允许使用的 BNF 权重内容指纹。
EXPECTED_BNF_CHECKPOINT_SHA256 = (
    # 固定 BNF 提取器权重的完整 SHA-256 内容摘要。
    "94362c99f408cbe891785537e222c2f1f9119edb11b95107b9faa4c0f0d679a5"
)
# 允许使用的 BNF 配置内容指纹。
EXPECTED_BNF_CONFIG_SHA256 = (
    # 固定 BNF 模型和前端配置的完整 SHA-256 内容摘要。
    "d8d3df407a9b88500cf62e32a8bdef78068b3a09e9f8bc47904843bef1fff6c4"
)
# 固定旧版位置编码源码指纹，兼容处理也必须与审核版本一致。
EXPECTED_BNF_COMPAT_EMBEDDING_SHA256 = (
    # 固定旧版 BNF 位置编码兼容源码的完整 SHA-256 摘要。
    "a9631e63a7d799c91b4d2a279f17520d98bc66e5be81cc2817957e75b1bf7a2e"
)
# 来源报告使用第二版结构，要求有逐文件内容与输入音频证据。
REPORT_SCHEMA_VERSION = 2
# BNF 或说话人嵌入的特征类别。
FEATURE_KIND = "bnf"


# 逐块读取文件并计算字节内容指纹；下面的参数分别限定输入数据和处理约束。
def sha256_file(path: Path) -> str:
    # 分块计算文件指纹，避免一次读入大型提取器权重。
    """Return a lowercase SHA-256 digest without loading a large checkpoint at once."""
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
    """Fail closed unless the audited ppg-vc checkout and BNF weight are in use."""
    # 定位外部提取器中已经审核的 ESPnet 兼容位置编码源码。
    compat_embedding_path = (
        # 使用固定 ppg-vc 仓库自带的 embedding.py，不从新装 ESPnet 随意替换。
        legacy_root / "conformer_ppg_model" / "encoder" / "embedding.py"
    )
    # 逐项核对提取器源码目录、权重和配置是否实际存在。
    for label, path in (
        # 需要检查外部 ppg-vc 的源码根目录。
        ("legacy ppg-vc root", legacy_root),
        # 核对 144 维 BNF 提取器预训练权重路径。
        ("BNF checkpoint", checkpoint_path),
        # 核对 BNF 网络和声学前端配置路径。
        ("BNF config", config_path),
        # 将位置编码兼容源码也纳入存在性及版本审核。
        ("pinned ESPnet-compatible embedding", compat_embedding_path),
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
        # 只读查询固定提取器源码相对 HEAD 的已跟踪改动。
        checkout_changes = subprocess.run(
            # 下面展开核对旧特征提取器源码、配置和预训练权重身份所需的这组条目；各字段随后逐一说明。
            [
                # 调用 Git 只读检查外部特征提取器来源。
                "git",
                # 指定后续 Git 命令运行的源码目录。
                "-C",
                # 让 Git 命令在已明确指定的外部提取器目录执行。
                str(legacy_root),
                # 检查外部源码当前工作区是否干净。
                "status",
                # 使用稳定、可供程序解析的状态格式。
                "--porcelain=v1",
                # 连未跟踪文件也列出，防止同名 Python 文件遮蔽固定实现。
                "--untracked-files=all",
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
        # 保留 Git 查询失败的原始异常链，便于追溯外部 BNF 源码审核为何无法完成。
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
    # 源码存在未审核改动时停止提取，防止新旧模型混产特征。
    if checkout_changes:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错指出固定源码目录存在修改或未跟踪内容，来源可能已被污染。
            "The pinned ppg-vc checkout has modified or untracked content; refusing "
            # 附上实际工作区变化列表，便于找出可能遮蔽 import 的文件。
            f"a shadowable/unaudited extraction:\n{checkout_changes}"
        )

    # 实际提取器权重文件的 SHA-256 指纹。
    checkpoint_sha256 = sha256_file(checkpoint_path)
    # 按实际提取器权重文件的 SHA-256 指纹决定是否进入此分支；判断条件为 checkpoint_sha256 != EXPECTED_BNF_CHECKPOINT_SHA256。
    if checkpoint_sha256 != EXPECTED_BNF_CHECKPOINT_SHA256:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明 BNF 预训练权重内容不匹配。
            "Unexpected BNF checkpoint SHA-256: "
            # 列出 BNF 权重的基准与实际摘要。
            f"expected {EXPECTED_BNF_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )
    # 按统一换行规范得到的提取器配置指纹。
    config_sha256 = sha256_text_lf(config_path)
    # 按按统一换行规范得到的提取器配置指纹决定是否进入此分支；判断条件为 config_sha256 != EXPECTED_BNF_CONFIG_SHA256。
    if config_sha256 != EXPECTED_BNF_CONFIG_SHA256:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明 BNF 模型或前端配置内容不匹配。
            "Unexpected BNF config SHA-256: "
            # 列出 BNF 配置的基准与实际摘要。
            f"expected {EXPECTED_BNF_CONFIG_SHA256}, got {config_sha256}"
        )
    # 统一换行后计算兼容位置编码源码指纹。
    compat_embedding_sha256 = sha256_text_lf(compat_embedding_path)
    # 兼容源码内容必须匹配固定版本，文件名相同不能代替内容校验。
    if compat_embedding_sha256 != EXPECTED_BNF_COMPAT_EMBEDDING_SHA256:
        # 核对旧特征提取器源码、配置和预训练权重身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明位置编码兼容源码内容偏离审核版本。
            "Unexpected pinned ESPnet-compatible embedding SHA-256: "
            # 输出兼容位置编码允许的源码摘要。
            f"expected {EXPECTED_BNF_COMPAT_EMBEDDING_SHA256}, "
            # 输出实际导入的兼容源码摘要，供逐字符核对。
            f"got {compat_embedding_sha256}"
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
        "checkpoint_expected_sha256": EXPECTED_BNF_CHECKPOINT_SHA256,
        # 记录按统一换行规范得到的提取器配置指纹。
        "config_sha256": config_sha256,
        # 记录允许使用的提取器参数文件指纹。
        "config_expected_sha256": EXPECTED_BNF_CONFIG_SHA256,
        # 记录实际使用的位置编码兼容源码指纹。
        "compat_embedding_sha256": compat_embedding_sha256,
        # 同时记录允许的基准指纹，便于来源报告独立核对。
        "compat_embedding_expected_sha256": EXPECTED_BNF_COMPAT_EMBEDDING_SHA256,
    }


# 导入已锁定的同源位置编码实现以兼容旧 BNF 提取器；下面的参数分别限定输入数据和处理约束。
def install_pinned_espnet_embedding_compat(legacy_root: Path) -> None:
    # 导入已锁定的同源位置编码实现以兼容旧 BNF 提取器。
    """Expose the pinned local positional encoder under the legacy ESPnet name.

    The audited ppg-vc checkout imports one class through ESPnet even though it
    carries the same implementation locally. Installing this narrow in-process
    alias avoids adding an unpinned full ESPnet distribution to the reproduction
    environment. The source file behind the alias is hash-checked above.
    """

    # 列出旧模型导入路径需要的 ESPnet 包层级，稍后仅建立兼容外壳。
    package_names = (
        # 旧位置编码导入路径最外层包名。
        "espnet",
        # 旧位置编码导入路径中的网络层级。
        "espnet.nets",
        # 旧 ESPnet PyTorch 后端包层级。
        "espnet.nets.pytorch_backend",
        # 旧 Transformer 包层级，最后将在其下挂接固定 embedding 实现。
        "espnet.nets.pytorch_backend.transformer",
    )
    # 兼容模块必须注册在旧模型实际 import 的完整路径。
    module_name = "espnet.nets.pytorch_backend.transformer.embedding"
    # 先检查当前进程是否已经导入其他 ESPnet 实现，避免污染固定来源。
    preloaded_espnet = [
        # 收集已经存在的 ESPnet 层级模块名。
        name for name in package_names + (module_name,) if name in sys.modules
    ]
    # 同时检查 Conformer 提取器是否在安装兼容入口前被提前导入。
    preloaded_conformer = sorted(
        # 保留发现的已加载模块名称，供冲突说明使用。
        name
        # 遍历当前 Python 进程已加载模块。
        for name in sys.modules
        # 只筛选本次固定 BNF 提取器及其子模块。
        if name == "conformer_ppg_model" or name.startswith("conformer_ppg_model.")
    )
    # 发现预加载冲突就停止，避免静默使用错误来源的位置编码。
    if preloaded_espnet or preloaded_conformer:
        # 导入已锁定的同源位置编码实现以兼容旧 BNF 提取器未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错要求用新进程建立兼容入口，不能混用已经加载的其他实现。
            "BNF compatibility setup requires a fresh Python process; preloaded "
            # 列出发现的预加载 ESPnet 和 Conformer 模块，定位导入来源冲突。
            f"modules: {preloaded_espnet + preloaded_conformer}"
        )

    # 使用标准库 ModuleType 创建最小兼容模块外壳。
    import types

    # 复用 embedding as local_embedding 的统一实现，具体输入和实验约束由调用处指定。
    from conformer_ppg_model.encoder import embedding as local_embedding

    # 建立唯一允许的位置编码源码绝对路径。
    expected_source = (
        # 使用固定 ppg-vc 仓库自带的 embedding.py，不从新装 ESPnet 随意替换。
        legacy_root / "conformer_ppg_model" / "encoder" / "embedding.py"
    # 消除相对路径和符号链接差异后再比较源码来源。
    ).resolve()
    # 读取实际导入的位置编码模块文件位置。
    actual_file = getattr(local_embedding, "__file__", None)
    # 无法追溯实际源码路径时拒绝继续安装兼容入口。
    if actual_file is None:
        # 导入已锁定的同源位置编码实现以兼容旧 BNF 提取器未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Pinned positional-encoding module has no source path")
    # 将实际导入来源规范为绝对路径。
    actual_source = Path(actual_file).resolve()
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 比较两路径是否对应同一实际文件，识别同名模块遮蔽。
        source_matches = actual_source.samefile(expected_source)
    # 捕获导入已锁定的同源位置编码实现以兼容旧 BNF 提取器中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except OSError:
        # 读取或比较源码位置失败时按来源不匹配处理。
        source_matches = False
    # 实际导入位置偏离固定源码时立即报错。
    if not source_matches:
        # 导入已锁定的同源位置编码实现以兼容旧 BNF 提取器未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错指出同名位置编码模块被其他文件遮蔽。
            "Positional-encoding import was shadowed: "
            # 同时列出应有源码路径和实际导入位置。
            f"expected {expected_source}, imported {actual_source}"
        )
    # 仅复用已校验本地源码中的位置编码类，不重新实现数学计算。
    PositionalEncoding = local_embedding.PositionalEncoding

    # 初始化逐层建立模块父子关系时的父模块指针。
    parent = None
    # 从顶层 ESPnet 到 transformer 按顺序建立兼容包层级。
    for package_name in package_names:
        # 读取当前层级是否已有兼容包对象。
        package = sys.modules.get(package_name)
        # 当前层级尚不存在时创建最小模块外壳。
        if package is None:
            # 创建具备正确导入名称的空模块对象。
            package = types.ModuleType(package_name)
            # 登记包名，使 Python 的相对导入机制识别当前层级。
            package.__package__ = package_name
            # 将该外壳标为包，但不加入额外搜索目录。
            package.__path__ = []
            # 把当前兼容包登记到进程模块表，供后续旧式 import 使用。
            sys.modules[package_name] = package
        # 从第二层开始，需要将当前包挂到上一级模块属性。
        if parent is not None:
            # 使用最后一级名称建立父包到子包的属性引用。
            setattr(parent, package_name.rsplit(".", 1)[1], package)
        # 当前层成为下一层兼容包的父对象。
        parent = package

    # 创建旧模型真正导入的 embedding 兼容模块。
    compat_module = types.ModuleType(module_name)
    # 把 embedding 归入 transformer 包层级。
    compat_module.__package__ = module_name.rsplit(".", 1)[0]
    # 保留真实固定源码路径，后续检查仍能追溯来源。
    compat_module.__file__ = str(expected_source)
    # 将已核对的原位置编码类暴露为旧导入路径的同名成员。
    compat_module.PositionalEncoding = PositionalEncoding
    # 注册最终兼容模块，旧 BNF 模型由此导入固定位置编码。
    sys.modules[module_name] = compat_module
    # 补齐 transformer.embedding 属性，与模块表中的路径保持一致。
    setattr(parent, "embedding", compat_module)


# 检查 BNF 是有限 float32 的 144 维内容特征；下面的参数分别限定输入数据和处理约束。
def validate_bnf_file(path: Path, expected_dim: int = EXPECTED_BNF_DIM) -> tuple:
    # 检查 BNF 是有限 float32 的 144 维内容特征。
    """Validate an on-disk canonical BNF array and return its shape."""
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 当前待校验或登记的数值特征数组。
        array = np.load(path, allow_pickle=False)
    # 捕获检查 BNF 是有限 float32 的 144 维内容特征中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as exc:
        # 检查 BNF 是有限 float32 的 144 维内容特征未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Unreadable BNF array {path}: {exc}") from exc
    # 要求读取对象符合检查 BNF 是有限 float32 的 144 维内容特征预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(array, np.ndarray):
        # 检查 BNF 是有限 float32 的 144 维内容特征未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"BNF output is not an ndarray: {path}")
    # 要求磁盘 BNF 为 [144,T] 且 T>0，防止通道方向或空时间轴错误。
    if array.ndim != 2 or array.shape[0] != expected_dim or array.shape[1] <= 0:
        # 检查 BNF 是有限 float32 的 144 维内容特征未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 报错说明 BNF 必须按内容通道×非空时间排列。
            f"BNF must have canonical shape [{expected_dim}, T>0], got "
            # 补充实际数组形状和文件路径，便于修复错维文件。
            f"{array.shape}: {path}"
        )
    # 检查数组所有元素均为有限数；NaN 或无穷值会污染后续模型输入和统计量。
    if not np.isfinite(array).all():
        # 检查 BNF 是有限 float32 的 144 维内容特征未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"BNF contains NaN or Inf: {path}")
    # 要求特征以单精度保存，避免来源混杂或推理时隐式改变数值精度。
    if array.dtype != np.float32:
        # 检查 BNF 是有限 float32 的 144 维内容特征未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"BNF must be float32, got {array.dtype}: {path}")
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
    # 144 维 BNF 的正式输出目录。
    bnf_dir: Path,
    # 数据处理审计报告的保存位置。
    report_path: Path,
    # 中断续作使用的逐条进度日志。
    journal_path: Path,
    # 本轮固定提取器源码、配置和权重的身份记录。
    output_provenance: Dict[str, str],
# 返回重用特征前逐条核对音频与已有发布证据整理的 Dict[str, Dict[str, object]] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, Dict[str, object]]:
    # 重用特征前逐条核对音频与已有发布证据。
    """Verify reused outputs against a prior per-file attestation.

    A valid shape alone says nothing about which model created a feature.  Therefore
    legacy reports without a content ledger are deliberately not accepted.
    """

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
                # 指出已有 BNF 缺少对应来源台账，需要明确重提取。
                f"Existing BNF {key} has no trusted record in {report_path}; run with --force."
            )
        # 历史台账必须说明该文件来自新生成或已验证复用，不能只有形状信息。
        if old_record.get("run_action") not in {"generated", "reused_verified"}:
            # 重用特征前逐条核对音频与已有发布证据未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 指出已有 BNF 虽有记录，却缺少生成或复用动作证据。
                f"Existing BNF {key} record does not identify how it was produced/reused; "
                # 错误提示要求显式全量重提取，不能无验证地继续跳过旧文件。
                "run with --force."
            )
        # 当前待检查或读写的文件路径。
        path = bnf_dir / key
        # 复用前检查当前 BNF 的通道数、帧数、单精度和有限数值。
        validate_bnf_file(path)
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
                    # 指出具体 BNF 及发生变化的来源字段。
                    f"Existing BNF {key} no longer matches its prior {field} attestation; "
                    # 错误提示要求显式全量重提取，不能无验证地继续跳过旧文件。
                    "run with --force."
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
    # 144 维 BNF 的正式输出目录。
    bnf_dir: Path,
    # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
    legacy_bnf_root: Path,
    # BNF 提取器使用的固定网络配置文件。
    train_config: Path,
    # BNF 提取器使用的固定模型权重文件。
    model_file: Path,
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
    # 张量执行计算的设备。
    device: str = "not-loaded",
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
        # 记录144 维 BNF 的正式输出目录。
        "bnf_dir": str(bnf_dir),
        # 记录固定 Git 提交的外部 BNF/说话人提取器代码目录。
        "legacy_bnf_root": str(legacy_bnf_root),
        # 记录BNF 提取器使用的固定网络配置文件。
        "train_config": str(train_config),
        # 记录BNF 提取器使用的固定模型权重文件。
        "model_file": str(model_file),
        # 记录本轮固定提取器源码、配置和权重的身份记录。
        "output_provenance": output_provenance,
        # 记录是否明确要求重新计算已有特征。
        "force": force,
        # 记录张量执行计算的设备。
        "device": device,
        # 记录内容 BNF 预期的 144 维通道数。
        "expected_bnf_dim": EXPECTED_BNF_DIM,
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
        # Backward-readable counters; their meaning is explicit above.
        # 兼容旧报告中的待生成数量，等于本轮新生成成功数加失败数，不含可信复用。
        "missing_rows": generated_rows + failed_rows,
        # 记录本轮新生成成功的数量。
        "ok": generated_rows,
        # 记录已有特征验证后跳过重算的数量。
        "skip": reused_rows,
        # 记录本轮处理失败的数量。
        "fail": failed_rows,
        # 记录整套特征及其输入 WAV 的汇总指纹。
        "feature_collection": summarize_feature_records(rows, records),
        # 记录以每条语音为单位保存的可核验特征台账。
        "file_records": records,
        # 记录有限数量的失败语音及具体原因。
        "fail_examples": fail_examples,
    }


# 验证 BNF 数组后通过临时文件发布；下面的参数分别限定输入数据和处理约束。
def atomic_save_bnf(path: Path, array: np.ndarray) -> None:
    # 验证 BNF 数组后通过临时文件发布。
    """Validate and atomically publish one BNF file beside its temporary file."""
    # 准备验证 BNF 数组后通过临时文件发布所需的目标目录，已有目录按给定选项复用。
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
        # 原子发布前重新打开临时 BNF，确认文件完整且数值格式正确。
        validate_bnf_file(temp_path)
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
            # 捕获验证 BNF 数组后通过临时文件发布中的读取、格式或兼容错误，让调用者获得明确失败原因。
            except FileNotFoundError:
                # 此异常属于允许忽略的清理或兼容情形，后续流程继续执行。
                pass


# 原子保存本次特征提取报告；下面的参数分别限定输入数据和处理约束。
def write_json_report(path: Path, report: Dict) -> None:
    # 原子保存本次特征提取报告。
    """Atomically write a JSON report so interruption cannot leave valid-looking JSON."""
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
        raise RuntimeError(f"Empty BNF provenance journal: {path}")
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 进度日志首行的版本和模型身份信息。
        header = json.loads(lines[0])
    # 捕获读取已落盘的逐样本进度并拒绝身份不一致的记录中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except json.JSONDecodeError as exc:
        # 读取已落盘的逐样本进度并拒绝身份不一致的记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Corrupt BNF provenance journal header: {path}") from exc
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
        raise RuntimeError(f"BNF provenance journal header mismatch: {path}")

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
            raise RuntimeError(f"Corrupt BNF provenance journal line {index + 1}") from exc
        # 当前语音的唯一身份。
        key = item.get("key")
        # 当前特征的内容、形状、输入音频及处理动作记录。
        record = item.get("record")
        # 每条进度必须有 feature 类别、字符串语音键和结构化来源记录。
        if item.get("record_type") != "feature" or not isinstance(key, str) or not isinstance(record, dict):
            # 读取已落盘的逐样本进度并拒绝身份不一致的记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Malformed BNF provenance journal line {index + 1}")
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
    # 清单不存在时立即报错，不允许返回空列表假装成功。
    if not manifest_path.exists():
        # 读取此次真正需要提取特征的清单并控制小样本检查范围未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # rows 保存最终不重复的数据行。
    rows = []
    # seen 保存已经见过的 (speaker,utt_id) 元组，用于去重。
    seen = set()

    # with 在代码块结束后自动关闭 CSV；newline="" 是 csv 模块推荐写法。
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        # DictReader 用表头作键，把每行变成字典。
        reader = csv.DictReader(f)
        # 逐行检查是否重复。
        for row in reader:
            # 同一说话人的同一 utt_id 唯一确定一条语音。
            key = (row["speaker"], row["utt_id"])
            # 重复项直接跳到下一轮循环。
            if key in seen:
                # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
                continue
            # 首次出现就加入集合。
            seen.add(key)
            # 并保存本行。
            rows.append(row)

    # limit>0 时只取前 N 条做小规模检查；-1 表示全量。
    if limit is not None and limit > 0:
        # 本次选定的语音清单。
        rows = rows[:limit]

    # 把处理后的行列表交给调用者。
    return rows


# 区分需提取的文件与可核验后重用的已有特征；下面的参数分别限定输入数据和处理约束。
def filter_missing_rows(
    # 本次选定的语音清单。
    rows: List[Dict[str, str]],
    # 144 维 BNF 的正式输出目录。
    bnf_dir: Path,
    # 是否明确要求重新计算已有特征。
    force: bool = False,
# 返回区分需提取的文件与可核验后重用的已有特征整理的 List[Dict[str, str]] 结果，内部各项保持前面约定的对应关系。
) -> List[Dict[str, str]]:
    # 缺失、损坏、形状错误或 --force 指定的行都需要重做。
    missing = []
    # 逐条构造预期输出路径。
    for row in rows:
        # 读取说话人字段。
        speaker = row["speaker"]
        # 读取 utterance ID。
        utt_id = row["utt_id"]
        # 目标格式 data/bnf/<speaker>/<utt_id>.npy。
        out_path = bnf_dir / speaker / f"{utt_id}.npy"
        # 按是否明确要求重新计算已有特征决定是否进入此分支；判断条件为 force or not out_path.exists()。
        if force or not out_path.exists():
            # 把缺失或校验失败的语音加入待重新提取清单。
            missing.append(row)
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # 已有特征先经过结构和数值检查，随后还要核对来源台账。
            validate_bnf_file(out_path)
        # 捕获区分需提取的文件与可核验后重用的已有特征中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except ValueError:
            # 把缺失或校验失败的语音加入待重新提取清单。
            missing.append(row)
    # 返回真正需要跑模型的子集。
    return missing


# 把旧提取器输出整理成 144 通道在前的 BNF；下面的参数分别限定输入数据和处理约束。
def standardize_bnf_shape(x: np.ndarray, expected_dim: int = EXPECTED_BNF_DIM) -> np.ndarray:
    # 把旧提取器输出整理成 144 通道在前的 BNF。
    """
    Convert model output to [144, T].
    Accept [144, T] or [T, 144].
    """
    # BNF 必须是“特征×时间”二维矩阵，其他维数直接拒绝。
    if x.ndim != 2:
        # 把旧提取器输出整理成 144 通道在前的 BNF未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"BNF ndim must be 2, got {x.ndim}")

    # 第一轴已经是 144 时不转置。
    if x.shape[0] == expected_dim:
        # BNF 已经是 [144,T] 通道优先排列，保持原数组方向。
        y = x
    # 第二轴是 144 时说明当前为 [T,144]，使用 .T 变为 [144,T]。
    elif x.shape[1] == expected_dim:
        # 将 [T,144] 转置为规范 [144,T]，只换轴，不改变内容特征数值。
        y = x.T
    # 两个轴都不是 144，不能猜测哪个是特征轴。
    else:
        # 把旧提取器输出整理成 144 通道在前的 BNF未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"BNF shape {x.shape} does not contain expected dim {expected_dim}")

    # 再检查标准化结果的 feature 轴和非空时间轴。
    if y.shape[0] != expected_dim or y.shape[1] <= 0:
        # 把旧提取器输出整理成 144 通道在前的 BNF未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Standardized BNF shape invalid: {y.shape}")

    # isfinite 排除 NaN 和正负无穷，这些数会污染训练 loss。
    if not np.isfinite(y).all():
        # 把旧提取器输出整理成 144 通道在前的 BNF未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError("BNF contains NaN or Inf")

    # 统一成训练使用的 float32 并返回。
    return y.astype(np.float32)


# 把输入音频变为旧 BNF 提取器需要的 16 kHz 单声道；下面的参数分别限定输入数据和处理约束。
def load_wav_mono_resample_16k(wav_path: Path) -> np.ndarray:
    # 延迟导入使来源审核/断点文件检查不需先加载整套模型依赖。
    import librosa
    # 把音频解码为浮点波形，保留文件真实采样率供核对。
    import soundfile as sf

    # soundfile 返回波形和文件中记录的真实采样率。
    wav, sr = sf.read(str(wav_path), always_2d=False)

    # 某些封装可能额外套 tuple，这里兼容取第一项波形。
    if isinstance(wav, tuple):
        # 读取后的单声道浮点音频。
        wav = wav[0]

    # 二维波形是 [samples,channels]，沿通道平均为 mono。
    if wav.ndim == 2:
        # 读取后的单声道浮点音频。
        wav = wav.mean(axis=1)

    # 先统一 dtype，后续 librosa 和 torch 都按 float32 工作。
    wav = wav.astype(np.float32)

    # 采样率不是 16k 时只在内存里重采样，不改磁盘 wav。
    if sr != SAMPLE_RATE:
        # 读取后的单声道浮点音频。
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

    # librosa 结果可能升级 dtype，再强制返回 float32。
    return wav.astype(np.float32)


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main():
    # 创建命令行解析器。
    parser = argparse.ArgumentParser()
    # 以下参数分别覆盖清单、输出、报告、旧工程、模型配置、模型权重和调试条数。
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--bnf_dir", type=str, default=str(DEFAULT_BNF_DIR))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--legacy_bnf_root", type=str, default=str(LEGACY_BNF_ROOT))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--train_config", type=str, default=str(DEFAULT_TRAIN_CONFIG))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--model_file", type=str, default=str(DEFAULT_MODEL_FILE))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--limit", type=int, default=-1, help="For sanity check. Use -1 for full run.")
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 显式要求重算已有特征；普通运行先尝试验证后复用。
        "--force",
        # 该文件本轮为新生成还是经过验证后复用。
        action="store_true",
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="Re-extract every selected BNF even when a valid output exists.",
    )
    # 真正读取终端参数。
    args = parser.parse_args()

    # 把 argparse 返回的字符串统一转为 Path 对象。
    manifest_path = Path(args.manifest)
    # 144 维 BNF 的正式输出目录。
    bnf_dir = Path(args.bnf_dir)
    # 数据处理审计报告的保存位置。
    report_path = Path(args.report)
    # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
    legacy_bnf_root = Path(args.legacy_bnf_root)
    # BNF 提取器使用的固定网络配置文件。
    train_config = Path(args.train_config)
    # BNF 提取器使用的固定模型权重文件。
    model_file = Path(args.model_file)

    # 预先创建输出和报告父目录。
    bnf_dir.mkdir(parents=True, exist_ok=True)
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 在加载任何推理代码前核对 Git commit、权重 SHA-256 和配置哈希。
    provenance = verify_legacy_provenance(
        # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
        legacy_bnf_root,
        # BNF 提取器使用的固定模型权重文件。
        model_file,
        # BNF 提取器使用的固定网络配置文件。
        train_config,
    )

    # 加载清单。
    rows = load_manifest_rows(manifest_path, limit=args.limit)
    # 只跳过经重新加载审核合格的现有 BNF。
    rows_to_process = filter_missing_rows(rows, bnf_dir, force=args.force)
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
    # 按当前少量诊断必须保护已有更大规模正式报告的标志决定是否进入此分支；判断条件为 protects_larger_report and rows_to_process。
    if protects_larger_report and rows_to_process:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明少量诊断会改动全量正式 BNF 集合。
            "A limited run would modify canonical BNF files covered by a larger "
            # 指出受保护的正式报告路径，提示改为全量处理或双路径隔离。
            f"formal report ({report_path}). Run the full manifest, or use both "
            # 少量 BNF 诊断必须同时使用独立特征目录和报告，保护全量来源账本。
            "a separate --bnf_dir and --report for the debug run."
        )
    # 中断续作使用的逐条进度日志。
    journal_path = progress_journal_path(report_path, provenance)
    # 按是否明确要求重新计算已有特征决定是否进入此分支；判断条件为 args.force。
    if args.force:
        # A forced run regenerates every selected feature, so an older journal
        # must not attribute those new files or block a provenance-schema update.
        # 先核对已有 BNF 的来源台账，出现无证据输出时按后面的恢复分支处理。
        try:
            # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
            journal_path.unlink()
        # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except FileNotFoundError:
            # 此异常属于允许忽略的清理或兼容情形，后续流程继续执行。
            pass
    # A crash can occur after the feature file is atomically published but
    # before its journal record is fsynced.  Such a shape-valid file is not
    # trusted reuse: schedule only that row for deterministic regeneration
    # instead of forcing an expensive full-manifest --force run.
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
            # 日志列出已有文件却没有持久来源证据、正在重算的 BNF 数量。
            f"{len(unattested_rows)} BNF file(s) published without a durable attestation."
        )
    # 形状正确只能证明“可读”，不能证明“由当前 BNE 生成”。逐文件核对上一份
    # schema-v2 内容摘要后才允许复用；来源不明的单文件会在上面自动重算。
    reused_records = verify_reused_feature_records(
        # 本轮通过来源审核后复用的语音条目或数量。
        reused_rows,
        # 144 维 BNF 的正式输出目录。
        bnf_dir,
        # 数据处理审计报告的保存位置。
        report_path,
        # 中断续作使用的逐条进度日志。
        journal_path,
        # 固定源码提交、配置和权重组成的提取器身份。
        provenance,
    )
    # 按当前少量诊断必须保护已有更大规模正式报告的标志决定是否进入此分支；判断条件为 protects_larger_report。
    if protects_larger_report:
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print(f"[INFO] Preserving larger provenance report {report_path}.")

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Manifest         = {manifest_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] BNF dir          = {bnf_dir}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Legacy BNF root  = {legacy_bnf_root}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Train config     = {train_config}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Model file       = {model_file}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Total rows       = {len(rows)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Rows to process  = {len(rows_to_process)}")

    # 全部输出已存在时，不浪费 GPU 加载模型，写报告后正常退出。
    if len(rows_to_process) == 0:
        # 按当前少量诊断必须保护已有更大规模正式报告的标志决定是否进入此分支；判断条件为 protects_larger_report。
        if protects_larger_report:
            # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
            print("\n[DONE] Selected BNF files match the preserved full report.")
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
            # 144 维 BNF 的正式输出目录。
            bnf_dir=bnf_dir,
            # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
            legacy_bnf_root=legacy_bnf_root,
            # BNF 提取器使用的固定网络配置文件。
            train_config=train_config,
            # BNF 提取器使用的固定模型权重文件。
            model_file=model_file,
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
        # 报告注明所有 BNF 都已和前次台账做内容核对，本轮无需重新计算。
        report["note"] = "All BNF files were content-verified against the prior report."
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
        print("\n[DONE] All target BNF files already exist.")
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print(f"  report = {report_path}")
        # 提前结束 main。
        return

    # 提取前准备绑定本次固定模型身份的可恢复进度日志。
    ensure_progress_journal(journal_path, provenance)

    # Import legacy model loader from old compute_bnf project
    # 需要执行特征模型时再载入 PyTorch，纯来源复用检查无需提前占用模型内存。
    import torch
    # 显示逐样本或逐批次进度，便于在服务器上观察是否仍在处理。
    from tqdm import tqdm

    # 把旧工程放到 Python 模块搜索路径第 0 位，优先使用指定旧实现。
    sys.path.insert(0, str(legacy_bnf_root))
    # 旧 subsampling.py 只从 ESPnet 导入 PositionalEncoding；将已经审核并锁定哈希的
    # 本地同源实现映射到该模块名，避免引入版本不明确的完整 ESPnet 包。
    install_pinned_espnet_embedding_compat(legacy_bnf_root)
    # 动态导入旧工程的模型加载函数。
    from conformer_ppg_model.build_ppg_model import load_ppg_model

    # 有 CUDA 使用 GPU，否则退回 CPU；这里只推理 BNF，不训练模型。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Device = {device}")

    # Build old PPG/BNF model exactly the old way
    # YAML 建结构、pth 装权重并放到 device。
    ppg_model_local = load_ppg_model(str(train_config), str(model_file), device)
    # eval 关闭 dropout 等训练行为。
    ppg_model_local.eval()

    # 初始化成功、已存在、失败计数及错误示例列表。
    ok = 0
    # 已有特征验证后跳过重算的数量。
    skip = len(rows) - len(rows_to_process)
    # 本轮处理失败的数量。
    fail = 0
    # 有限数量的失败语音及具体原因。
    fail_examples = []
    # 新生成的摘要与已验证复用摘要最后合并成完整集合账本。
    records: Dict[str, Dict[str, object]] = dict(reused_records)

    # 顺序跑 GPU 模型，避免多进程各加载一份权重爆显存。
    for row in tqdm(rows_to_process, total=len(rows_to_process), desc="Extract BNF"):
        # 当前语音所属说话人。
        speaker = row["speaker"]
        # 同时包含说话人和句号的样本编号。
        utt_id = row["utt_id"]
        # 当前语音的音频文件。
        wav_path = Path(row["wav22k_path"])
        # 当前样本的目标特征文件。
        out_path = bnf_dir / speaker / f"{utt_id}.npy"

        # 单条 try/except 使已成功结果能保存，同时汇总失败。
        try:
            # 检查当前语音的音频文件是否缺失；缺失时不能继续信任该输入。
            if not wav_path.exists():
                # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
                raise FileNotFoundError(f"wav not found: {wav_path}")

            # 读取、转 mono、临时重采样 16k。
            wav = load_wav_mono_resample_16k(wav_path)

            # BNF 提取器只接受单条一维波形，多声道或错维数据要先拒绝。
            if wav.ndim != 1:
                # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
                raise ValueError(f"waveform ndim must be 1, got {wav.ndim}")
            # 拒绝零长度音频，避免模型产生无意义的空 BNF。
            if len(wav) <= 0:
                # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
                raise ValueError("empty waveform")

            # [samples] NumPy -> float Tensor -> device -> [1,samples] batch。
            # 将已整理为 16 kHz 单声道的波形变成 [1,N] 单精度张量，供固定 BNF 提取器计算。
            wav_tensor = torch.from_numpy(wav).float().to(device).unsqueeze(0)
            # 序列模型需要真实样本长度，LongTensor 保存整数长度。
            wav_length = torch.LongTensor([wav.shape[0]]).to(device)

            # 不计算梯度，减少显存和运算。
            with torch.no_grad():
                # 表示语音内容的 144 维 BNF 特征。
                bnf = ppg_model_local(wav_tensor, wav_length)

            # 去 batch、脱离计算图、回 CPU、转 NumPy。
            bnf_npy = bnf.squeeze(0).detach().cpu().numpy()
            # 统一为 [144,T] float32。
            bnf_npy = standardize_bnf_shape(bnf_npy, expected_dim=EXPECTED_BNF_DIM)

            # 先写同目录临时文件，再重新加载验证并原子替换。
            atomic_save_bnf(out_path, bnf_npy)
            # 当前特征的内容、形状、输入音频及处理动作记录。
            record = make_feature_record(
                # 当前样本的目标特征文件。
                out_path,
                # 传入整理为 [144,T] 的单精度 BNF 数值，登记其内容摘要。
                bnf_npy,
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

        # 捕获单条语音的任何错误。
        except Exception as e:
            # 当前语音提取失败后增加失败计数，最终不能宣布全量成功。
            fail += 1
            # 来源报告最多附 20 个具体失败示例，失败总数仍完整记录。
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

    # 汇总本次所用路径、设备、数量和逐文件内容账本。
    report = build_extraction_report(
        # 本次实际使用的清单文件。
        manifest_path=manifest_path,
        # 本次选定的语音清单。
        rows=rows,
        # 144 维 BNF 的正式输出目录。
        bnf_dir=bnf_dir,
        # 固定 Git 提交的外部 BNF/说话人提取器代码目录。
        legacy_bnf_root=legacy_bnf_root,
        # BNF 提取器使用的固定网络配置文件。
        train_config=train_config,
        # BNF 提取器使用的固定模型权重文件。
        model_file=model_file,
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
        # 张量执行计算的设备。
        device=device,
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
    print("\n[DONE] BNF extraction finished.")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  ok   = {ok}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  skip = {skip}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  fail = {fail}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  report = {report_path}")

    # 按有限数量的失败语音及具体原因决定是否进入此分支；判断条件为 fail_examples。
    if fail_examples:
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print("\n[FAIL EXAMPLES]")
        # 终端展示最多 10 个失败例子，完整计数仍保存在报告。
        for item in fail_examples[:10]:
            # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
            print(item)

    # 任意失败都以非零状态结束，防止残缺 BNF 进入训练。
    if fail:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise SystemExit(
            # 最终报错列出 BNF 提取失败总数和来源报告位置。
            f"BNF extraction failed for {fail} utterances; see {report_path}."
        )


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 直接运行才加载模型并提取。
    main()
