#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 此入口训练独立声码器 G+MPD+MRD，完整断点包含优化器和随机状态。
"""
训练 FastVoiceGrad 论文架构对齐声码器的主程序。

真正执行的数据流可以压缩成下面六步：

    归一化训练波形现场计算 log-mel -> HiFi-GAN V1 Generator -> 生成波形
    真实波形 + 生成波形 -> HiFi-GAN MPD 与 UnivNet MRD -> 判别器损失
    生成波形 -> mel 重建损失 + 对抗损失 + feature matching -> 更新生成器

严格对齐与公开边界：

- FastVoiceGrad 第 4.1 节明确：modified HiFi-GAN V1，MSD 替换为 MRD。
- Generator、MPD 和三个 HiFi-GAN 损失函数直接复用项目内官方 HiFi-GAN 仓库。
- MRD 使用 UnivNet 论文公开的三套 STFT 参数和图 2 卷积结构。
- FastVoiceGrad 没有公开独立声码器训练时长和声码器检查点；正式终点 2,500,000
  steps 继承官方 HiFi-GAN 的 VCTK 实验，并明确标成复现选择而非作者隐藏值。
- 本程序只允许 train_manifest.csv，不创建 validation，也不使用 test 选择模型。
"""

# 延后求值类型提示，保持旧 Python 解释器兼容。
from __future__ import annotations

# 先忽略官方 weight_norm 在新 PyTorch 中的“未来弃用”提示；它不是训练错误。
import warnings
# 只隐藏旧 weight_norm 接口的弃用提醒，训练错误仍正常抛出。
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

# argparse 读取命令行参数，例如项目根目录和冒烟测试步数。
import argparse

# csv 把每一步 loss 写进简单、稳定、以后能画图的表格。
import csv

# hashlib 固定配置、清单和训练资产身份，防止 resume 时悄悄换数据。
import hashlib

# itertools.chain 把 MPD 和 MRD 参数串成一个判别器优化器的参数流。
import itertools

# json 读取固定配置，并保存本次实际运行配置快照。
import json

# os.environ 读取可移植路径，os.replace 用于原子替换检查点，避免断电留下半个文件。
import os

# random 和 NumPy/Torch 一起设随机种子。
import random

# signal lets nohup/background jobs convert SIGINT/SIGTERM into the existing
# resumable-checkpoint path instead of exiting without saving.
# 捕获普通停止信号，完成当前成对更新后先保存可续训状态。
import signal

# subprocess 只读取 Git commit/clean 状态并写进 checkpoint provenance。
import subprocess

# sys 用来把 hifigan/official 目录临时加入 Python 模块搜索路径。
import sys

# time 计算每个日志区间和每一步大约耗时。
import time

# Path 安全处理 Linux/Windows 路径。
from pathlib import Path

# 类型提示让配置字典和函数返回值更清楚。
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# NumPy 用于 DataLoader worker 的随机种子。
import numpy as np

# PyTorch 完成模型、反向传播、优化器、学习率调度和 GPU 操作。
import torch
# 计算 mel 的逐元素 L1 重建误差。
import torch.nn.functional as F
# 使用 PyTorch 的数据集、批次采样和加载接口。
from torch.utils.data import DataLoader, Sampler

# 自己的数据模块负责训练清单、官方式 waveform 裁切/归一化和 mel 计算。
from .dataset import (
    # 按训练清单读取波形，裁切后现场计算条件 mel。
    ManifestMelWaveDataset,
    # 预先检查对应波形与 mel 是否缺失。
    check_item_files,
    # 核对训练子集和未见说话人隔离，再创建训练条目。
    load_training_items,
    # 条件 mel 与重建损失共享明确的 HiFi-GAN 频谱函数。
    mel_spectrogram_torch,
)

# 自己的 MRD 模块是论文相对官方 HiFi-GAN V1 的唯一架构替换。
from .mrd import (
    # 联合三个频谱分辨率判断真假。
    MultiResolutionDiscriminator,
    # 为每个 MRD 分支保存 FFT 点数、帧移和窗长。
    STFTResolution,
)

# 限制实际计算只能在当前租用服务器执行。
from voicegrad.runtime_guard import assert_rented_server


# 完整训练档案和轻量生成器档案共同使用第二版结构。
CHECKPOINT_FORMAT_VERSION = 2
# 声明 G 与 MPD 复用 HiFi-GAN V1，频谱判别器采用 MRD。
ARCHITECTURE_NAME = "HiFi-GAN-V1 + MPD + UnivNet-MRD (MSD replaced)"


# 在完整批次更新后触发安全停机，避免把半次更新写成完整进度。
class SafeStopRequested(Exception):
    # 该异常只在完成一次 G/D 成对更新后触发，以进入保存断点的安全退出分支。
    """Internal control flow used after finishing the current optimizer step."""


# 把每轮乱序与裁切种子固定下来，能从轮中断点恢复相同输入序列。
class DeterministicEpochBatchSampler(Sampler):
    # 轮次、基础种子和批次游标共同决定语音顺序及裁切，worker 调度不改变样本。
    """Yield reproducible batches and crop seeds for an epoch, with a resume cursor."""

    # 保存训练条数、批大小、种子和恢复游标，并检查取值范围。
    def __init__(
        # 当前采样器实例，后续按这些固定参数生成批次。
        self,
        # 星号之后的参数必须写出参数名，防止位置传参把路径、字段或轮次放错。
        *,
        # 输入参数：训练数据集语音数量。
        dataset_size: int,
        # 输入参数：每批处理的语音数。
        batch_size: int,
        # 输入参数：随机种子。
        seed: int,
        # 输入参数：训练轮次。
        epoch: int,
        # 设置续训时本轮第一个尚未完成的批次。
        start_batch: int = 0,
    # 构造函数把参数和网络层保存在当前对象中，按 Python 约定不返回结果对象。
    ) -> None:
        # 固定训练语音总数，用于构造本轮随机排列。
        self.dataset_size = int(dataset_size)
        # 固定每个优化批次包含的语音数。
        self.batch_size = int(batch_size)
        # 保存整个声码器实验的基础随机种子。
        self.seed = int(seed)
        # 保存当前轮次，它参与乱序和裁切种子生成。
        self.epoch = int(epoch)
        # 从这一未完成批次开始，跳过断点前已训练数据。
        self.start_batch = int(start_batch)
        # 数据集和批大小必须为正，否则无法形成训练批次。
        if self.dataset_size <= 0 or self.batch_size <= 0:
            # 报错原因：数据集和批大小必须为正，否则无法形成训练批次。
            raise ValueError("dataset_size and batch_size must be positive")
        # 训练轮次和续训批次游标都不能为负。
        if self.epoch < 0 or self.start_batch < 0:
            # 报错原因：训练轮次和续训批次游标都不能为负。
            raise ValueError("epoch and start_batch must be non-negative")
        # 恢复位置不能超过本轮可形成的完整批次数。
        if self.start_batch > self.batches_per_epoch:
            # 报错原因：恢复位置不能超过本轮可形成的完整批次数。
            raise ValueError(
                # 恢复位置不能超过本轮可形成的完整批次数。
                f"start_batch={self.start_batch} exceeds {self.batches_per_epoch}"
            )

    # 把完整批次数暴露为只读计算属性。
    @property
    # 计算本轮完整批次数，尾部不足一批不参与更新。
    def batches_per_epoch(self) -> int:
        # 训练样本数除以批大小后向下取整，保持固定批大小。
        return self.dataset_size // self.batch_size

    # 向 DataLoader 提供当前数据集或剩余批次的大小。
    def __len__(self) -> int:
        # DataLoader 只需处理本轮断点后尚未完成的批次。
        return self.batches_per_epoch - self.start_batch

    # 按该轮固定乱序和批次游标依次产生训练样本索引。
    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        # A separate epoch seed makes the complete permutation a pure function of
        # (formal seed, epoch), independent of DataLoader worker scheduling.
        # 为采样建立独立 CPU 随机器，不消耗网络训练的全局随机序列。
        generator = torch.Generator()
        # 同一基础种子与轮次始终得到相同语音排列。
        generator.manual_seed(self.seed + self.epoch)
        # 对本轮全部训练语音生成随机排列。
        permutation = torch.randperm(
            # 明确排列长度和独立随机器，避免 worker 调度影响顺序。
            self.dataset_size, generator=generator
        # 将随机排列转为 Python 索引列表，交给批次采样器。
        ).tolist()
        # 从断点的下一批起，依次产出剩余完整批次。
        for batch_index in range(self.start_batch, self.batches_per_epoch):
            # 把批次编号换算为本轮排列中的起始位置。
            begin = batch_index * self.batch_size
            # 一批保存语音索引与对应裁切种子二元组。
            batch: List[Tuple[int, int]] = []
            # 遍历当前批次中每条语音及其批内位置。
            for offset, item_index in enumerate(
                # 截取本批固定数量的语音索引。
                permutation[begin : begin + self.batch_size]
            ):
                # 恢复该语音在本轮随机排列中的绝对位置。
                position = begin + offset
                # 设置该样本固定的随机裁切种子。
                crop_seed = (
                    # 用较大的固定整数混合基础种子，区分不同实验的裁切序列。
                    self.seed * 1_000_003
                    # 让不同训练轮次使用不同裁切种子区间。
                    + self.epoch * self.dataset_size
                    # 同一轮每个采样位置也获得独立裁切种子。
                    + position
                # 将混合裁切种子约束在可接受的 63 位非负整数范围。
                ) % (2**63 - 1)
                # 把语音编号与裁切种子共同交给 Dataset。
                batch.append((int(item_index), int(crop_seed)))
            # 产生一个完整且可重复的训练批次。
            yield batch


# 优先用 VOICEGRAD_PROJECT_ROOT；没有设置就用本文件所在的仓库目录。
# 公开代码因此不会泄露服务器目录，也不要求读者修改 Python 源码。
DEFAULT_PROJECT_ROOT = Path(
    # 优先使用显式配置的服务器项目路径，否则从当前源文件反推根目录。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[2])
# 展开用户目录并规范为绝对路径，避免工作目录影响定位。
).expanduser().resolve()


# 按二进制文件内容计算 SHA-256，供后续检查是否混入其他版本。
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # 按二进制文件内容计算 SHA-256，供后续检查是否混入其他版本。
    """Return a streaming SHA-256 digest without loading a large file at once."""

    # 初始化关键数据或源码文件的内容指纹累积器。
    digest = hashlib.sha256()
    # 按二进制读取，文件编码或换行不会被自动转换。
    with path.open("rb") as handle:
        # 每次读取一块文件内容，直到文件末尾，避免一次占满内存。
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            # 把当前二进制块加入同一个文件指纹。
            digest.update(chunk)
    # 返回便于写入 JSON 的十六进制文件指纹。
    return digest.hexdigest()


# 按统一 JSON 编码计算内容指纹，不受缩进和字段排列影响。
def canonical_json_sha256(value: Any) -> str:
    # 按统一 JSON 编码计算内容指纹，不受缩进和字段排列影响。
    """Hash JSON semantics with stable ordering and separators."""

    # 将配置按稳定排序编码，供内容指纹计算。
    payload = json.dumps(
        # 保留中文并固定字段顺序及分隔符，配置内容相同就得到同一编码。
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    # 将统一 JSON 表示转为 UTF-8 字节后计算指纹。
    ).encode("utf-8")
    # 返回规范化配置的 SHA-256。
    return hashlib.sha256(payload).hexdigest()


# 先写临时 JSON 再原子替换，减少中断时留下半份报告的风险。
def atomic_write_json(path: Path, value: Any) -> None:
    # 先写临时 JSON 再原子替换，减少中断时留下半份报告的风险。
    """Atomically publish a JSON file next to checkpoints."""

    # 创建结果文件的父目录；已有目录可继续使用。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 在正式 JSON 旁边先写临时文件，避免中断留下半份记录。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 先完整写入临时文件，避免正式路径出现半写入记录。
    temporary.write_text(
        # 把完整记录格式化成易读 JSON 文本。
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        # 保存为 UTF-8，保留中文配置说明。
        encoding="utf-8",
    )
    # 把写完的临时文件原子替换到正式结果路径。
    os.replace(temporary, path)


# 核对训练清单、划分与预处理记录，防止续训时换数据。
def collect_and_verify_data_fingerprints(
    # 输入当前服务器项目根目录与本次生效配置。
    project_root: Path, config: Dict[str, Any]
# 返回数据名称到已核对 SHA-256 的字典。
) -> Dict[str, str]:
    # 核对训练清单、划分与预处理记录，防止续训时换数据。
    """Bind the run to the exact active manifests and published mel collection."""

    # 从固定配置读取期望的训练数据指纹。
    expected = config.get("data_fingerprints")
    # 缺失数据身份表时禁止训练或续训。
    if not isinstance(expected, dict) or not expected:
        # 报错原因：缺失数据身份表时禁止训练或续训。
        raise ValueError("config.data_fingerprints must be a non-empty object")
    # 逐项检查下面列出的指纹格式条件是否有不满足的情况。
    if any(
        # 数据身份名称必须为文本。
        not isinstance(key, str)
        # 每个指纹必须保存为文本。
        or not isinstance(value, str)
        # SHA-256 指纹必须包含 64 个十六进制字符。
        or len(value) != 64
        # 检查指纹只含十六进制字符。
        or any(character not in "0123456789abcdef" for character in value.lower())
        # 覆盖配置中的全部数据指纹，不只抽查一项。
        for key, value in expected.items()
    ):
        # 报错原因：覆盖配置中的全部数据指纹，不只抽查一项。
        raise ValueError("config.data_fingerprints must contain only SHA-256 strings")

    # 定位本次训练使用的数据根目录。
    data = project_root / "data"
    # 定位训练/测试划分和波形清单目录。
    metadata = data / "metadata"
    # 定位训练集 mel 均值、标准差及其生成记录。
    stats = data / "stats"
    # 读取当前正式发布的 mel 统计批次指针。
    current_path = stats / "mel_stats_current.json"
    # 没有正式统计发布记录就不能确认当前归一化来源。
    if not current_path.is_file():
        # 报错原因：没有正式统计发布记录就不能确认当前归一化来源。
        raise FileNotFoundError(f"Mel statistics publication is missing: {current_path}")
    # 加载当前统计生成批次的身份记录。
    current = json.loads(current_path.read_text(encoding="utf-8"))
    # 取出不可混用的 mel 统计生成批次编号。
    generation_id = str(current.get("generation_id", "")).strip()
    # 统计批次编号缺失时中止，避免引用不明来源文件。
    if not generation_id:
        # 报错原因：统计批次编号缺失时中止，避免引用不明来源文件。
        raise ValueError("mel_stats_current.json lacks generation_id")
    # 定位该发布批次的完整统计来源报告。
    report_path = stats / ".mel_stats_generations" / generation_id / "report.json"
    # 发布指针必须能对应到真实完整报告。
    if not report_path.is_file():
        # 报错原因：发布指针必须能对应到真实完整报告。
        raise FileNotFoundError(f"Mel statistics report is missing: {report_path}")
    # 读取核对结果。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # BNF 提取记录中保存了原始输入波形集合指纹，供数据来源核对。
    bnf_report_path = project_root / "exp" / "logs" / "bnf_extraction_report.json"
    # 定位 22.05 kHz mel 提取来源记录。
    mel_extraction_report_path = (
        # 指定 22.05 kHz mel 提取来源报告。
        project_root / "exp" / "logs" / "mel_extraction_report_22050.json"
    )
    # 波形和 mel 提取来源记录必须同时存在。
    if not bnf_report_path.is_file() or not mel_extraction_report_path.is_file():
        # 报错原因：波形和 mel 提取来源记录必须同时存在。
        raise FileNotFoundError("Feature provenance reports required for WAV/mel identity are missing")
    # 读取 BNF 提取记录中的输入波形身份信息，不把 BNF 当声码器输入。
    bnf_report = json.loads(bnf_report_path.read_text(encoding="utf-8"))
    # 取得该报告中输入波形集合的内容与布局指纹。
    wav_collection = bnf_report.get("feature_collection", {})

    # 重新收集当前磁盘数据和来源记录的实际指纹。
    actual = {
        # 记录训练语音清单的文件指纹。
        "train_manifest_sha256": sha256_file(metadata / "train_manifest.csv"),
        # 记录测试语音清单的文件指纹。
        "test_manifest_sha256": sha256_file(metadata / "test_manifest.csv"),
        # 核对全量语音清单版本；它不作为声码器训练采样清单。
        "all_manifest_sha256": sha256_file(metadata / "all_manifest.csv"),
        # 记录数据划分约定指纹。
        "split_protocol_sha256": sha256_file(metadata / "split_protocol.json"),
        # 核对重采样后波形清单的身份。
        "vctk_wav22k_manifest_sha256": sha256_file(
            # 指定 22.05 kHz 波形文件清单。
            metadata / "vctk_wav22k_manifest.csv"
        ),
        # 记录用作波形来源证据的 BNF 提取报告指纹。
        "bnf_extraction_report_sha256": sha256_file(bnf_report_path),
        # 保留整批输入波形内容指纹。
        "wav_collection_content_sha256": str(
            # 从提取报告读取每条波形内容合成的集合指纹。
            wav_collection.get("input_content_sha256", "")
        ),
        # 保留输入波形相对路径布局的指纹。
        "wav_collection_layout_sha256": str(
            # 读取该波形集合的目录和文件名身份。
            wav_collection.get("input_layout_sha256", "")
        ),
        # 绑定 mel 提取参数和输入来源报告。
        "mel_extraction_report_sha256": sha256_file(mel_extraction_report_path),
        # 绑定正在生效的 mel 统计发布指针。
        "mel_stats_current_sha256": sha256_file(current_path),
        # 绑定该批 mel 统计完整报告。
        "mel_stats_report_sha256": sha256_file(report_path),
        # 记录mel 均值文件指纹。
        "mel_mean_sha256": sha256_file(stats / "mel_mean.npy"),
        # 记录mel 标准差文件指纹。
        "mel_std_sha256": sha256_file(stats / "mel_std.npy"),
        # 绑定预处理 mel 数值集合。
        "mel_collection_content_sha256": str(
            # 读取 mel 数值文件集合的整体内容指纹。
            report.get("mel_collection_content_sha256", "")
        ),
        # 绑定 mel 相对路径布局，防止语音身份与特征错配。
        "mel_collection_layout_sha256": str(
            # 读取 mel 文件目录及名称集合指纹。
            report.get("mel_collection_layout_sha256", "")
        ),
    }
    # 关键配置不符时中止，不能悄悄续训另一套实验。
    if actual != expected:
        # 把每项不一致的实际与预期指纹列出来，便于定位。
        differences = {
            # 同一数据项同时显示磁盘实际值和配置预期值。
            key: {"actual": actual.get(key), "expected": expected.get(key)}
            # 比较两侧所有键的并集，也能发现新增或缺失项。
            for key in sorted(set(actual) | set(expected))
            # 报告只保留不一致的数据项。
            if actual.get(key) != expected.get(key)
        }
        # 报错原因：报告只保留不一致的数据项。
        raise ValueError(f"Data fingerprint mismatch: {differences}")
    # 返回已通过精确比较的当前数据指纹表。
    return dict(actual)


# 记录当前 Git、关键源码指纹和依赖版本，约束续训来源。
def collect_runtime_provenance(project_root: Path) -> Dict[str, Any]:
    # 记录当前 Git、关键源码指纹和依赖版本，约束续训来源。
    """Record the clean source revision and core source hashes used by this run."""

    # 只读调用 Git，以核对本次训练源码提交和工作树状态。
    def git(*arguments: str) -> str:
        # 执行 Git 并捕获输出；命令失败会阻止来源记录生成。
        completed = subprocess.run(
            # 显式指定服务器项目目录，避免读到其他仓库。
            ["git", "-C", str(project_root), *arguments],
            # Git 返回非零状态时抛错，不使用不完整来源信息。
            check=True,
            # 捕获 Git 正常输出以写入来源记录。
            stdout=subprocess.PIPE,
            # 捕获 Git 错误信息以便失败定位。
            stderr=subprocess.PIPE,
            # 将 Git 输出作为文本处理。
            text=True,
        )
        # 去掉 Git 输出末尾换行后返回身份文本。
        return completed.stdout.strip()

    # 记录当前实际执行源码的提交编号。
    commit = git("rev-parse", "HEAD")
    # 检查提交之外是否还有未保存到 Git 的源码变更。
    dirty_status = git("status", "--porcelain")
    # 正式计算要求干净工作树，避免提交编号不能完整代表代码。
    if dirty_status:
        # 报错原因：正式计算要求干净工作树，避免提交编号不能完整代表代码。
        raise RuntimeError(
            # 正式计算要求干净工作树，避免提交编号不能完整代表代码。
            "Formal project compute requires a clean rented-server checkout; "
            # 显示服务器工作树具体未提交改动，说明为何正式训练被拒绝。
            f"git status returned:\n{dirty_status}"
        )
    # 额外逐文件绑定影响声码器训练的关键源码。
    source_paths = (
        # 训练循环、保存恢复与优化器设置来源。
        "hifigan/custom/train.py",
        # 波形归一化、随机裁切和条件 mel 的实现来源。
        "hifigan/custom/dataset.py",
        # 多分辨率判别器的实现来源。
        "hifigan/custom/mrd.py",
        # 官方 G、MPD 和损失函数的实现来源。
        "hifigan/official/models.py",
    )
    # 把已核对的字段组成返回记录，供后续身份检查或报告保存。
    return {
        # 记录生成当前代码的 Git 提交编号。
        "git_commit": commit,
        # 记录已实际通过工作树无未提交改动检查。
        "git_worktree_clean": True,
        # 先留配置指纹位置，主流程取得生效配置后补入。
        "config_semantic_sha256": None,
        # 记录关键训练源码的完整字节指纹。
        "source_sha256": {
            # 按相对路径记录全部关键源码的完整字节指纹，注释也计入。
            relative: sha256_file(project_root / relative) for relative in source_paths
        },
        # 记录实际 Python 解释器版本。
        "python": sys.version,
        # 记录实际 PyTorch 版本。
        "torch": torch.__version__,
        # 记录 PyTorch 构建所链接的 CUDA 版本。
        "cuda_runtime": torch.version.cuda,
        # 记录实际 cuDNN 版本。
        "cudnn": torch.backends.cudnn.version(),
    }


# 导入官方生成器、MPD 与三个损失函数，MRD 由本项目单独提供。
def import_official_hifigan(project_root: Path):
    # 取得当前项目官方 G 接口；检查点验证器还临时隔离并恢复同名模块缓存。
    """从项目内官方仓库导入 Generator、MPD 和三种损失函数。"""

    # 官方参考源码统一放在 hifigan/official，和自训练实现物理分开。
    hifigan_dir = project_root / "hifigan" / "official"

    # models.py 必须存在；否则无法保证用的是项目里审计过的官方代码。
    if not (hifigan_dir / "models.py").is_file():
        # 报错原因：models.py 必须存在；否则无法保证用的是项目里审计过的官方代码。
        raise FileNotFoundError(f"Official HiFi-GAN models.py not found: {hifigan_dir}")

    # 把字符串路径插到最前面，确保下面 import models 指向 official/models.py。
    sys.path.insert(0, str(hifigan_dir))

    # 导入发生在函数内部，是为了先确定正确的 project_root。
    # pylint/flake8 的位置警告在这里不适用。
    from env import AttrDict  # type: ignore
    # 从刚指定的官方目录加载 G、MPD 和损失定义。
    from models import (  # type: ignore
        # 官方 HiFi-GAN V1 生成器，将 mel 转成时域波形。
        Generator,
        # 官方多周期判别器，检查波形的不同周期结构。
        MultiPeriodDiscriminator,
        # 官方最小二乘判别器损失，真趋向 1、假趋向 0。
        discriminator_loss,
        # 官方逐层特征匹配损失，层内均值后求和并乘 2。
        feature_loss,
        # 官方生成器对抗损失，推动假分数接近 1。
        generator_loss,
    )

    # 用一个元组返回所有官方对象；调用方可按固定顺序解包。
    return (
        # 将字典配置包装成官方网络可用的属性访问形式。
        AttrDict,
        # 官方 HiFi-GAN V1 生成器，将 mel 转成时域波形。
        Generator,
        # 官方多周期判别器，检查波形的不同周期结构。
        MultiPeriodDiscriminator,
        # 官方最小二乘判别器损失，真趋向 1、假趋向 0。
        discriminator_loss,
        # 官方逐层特征匹配损失，层内均值后求和并乘 2。
        feature_loss,
        # 官方生成器对抗损失，推动假分数接近 1。
        generator_loss,
    )


# 读取运行路径、设备和诊断选项，默认值保持原实验约定。
def parse_args() -> argparse.Namespace:
    # 读取运行路径、设备和诊断选项，默认值保持原实验约定。
    """定义并读取命令行选项。"""

    # ArgumentParser 会自动生成 --help 页面。
    parser = argparse.ArgumentParser(
        # 命令行说明当前训练的是结构对齐的独立声码器衍生实验。
        description="Train paper-architecture-aligned HiFi-GAN V1 + MPD + MRD vocoder."
    )

    # 正式服务器默认路径；本地或别的副本可显式覆盖。
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)

    # 不传时使用项目根目录的固定 JSON 配置。
    parser.add_argument("--config", type=Path, default=None)

    # 正式输出目录与旧实验分开，绝不覆盖 pretrained_hifigan_22k/VCTK_V1。
    parser.add_argument("--checkpoint-dir", type=Path, default=None)

    # 覆盖项只允许与 --diagnostic-smoke 一起使用，且输出必须位于 exp/smoke。
    parser.add_argument("--max-steps", type=int, default=None)

    # --batch-size 可用来确认 2080 Ti 的显存上限；正式不传则保持官方 V1 的 16。
    parser.add_argument("--batch-size", type=int, default=None)

    # --num-workers 允许服务器文件系统太忙时暂时减小读取进程数。
    parser.add_argument("--num-workers", type=int, default=None)

    # --fresh 与 --resume 显式互斥，正式运行不再看到 latest.pt 就自动续训。
    parser.add_argument("--fresh", action="store_true")
    # 声明一个显式续训或诊断模式选项。
    parser.add_argument(
        # --fresh 与 --resume 显式互斥，正式运行不再看到 latest.pt 就自动续训。
        "--resume",
        # 将resume解析成可规范化的文件路径。
        type=Path,
        # 没有传入路径时保持为空，不自动猜测续训文件。
        default=None,
        # 帮助文本强调先核对精确断点身份，再进行任何写入。
        help="Exact path to this experiment's latest.pt; validated before any write.",
    )

    # 声明一个显式续训或诊断模式选项。
    parser.add_argument(
        # --fresh 与 --resume 显式互斥，正式运行不再看到 latest.pt 就自动续训。
        "--diagnostic-smoke",
        # 默认关闭；只有出现这一开关才启用诊断。
        action="store_true",
        # 诊断参数覆盖只能写到 exp/smoke，保护正式训练目录。
        help="Allow step/batch/worker overrides only in a separate exp/smoke directory.",
    )

    # --preflight-only 只核对数据、配置和 GPU，不建立模型、不训练。
    parser.add_argument("--preflight-only", action="store_true")

    # --full-file-check 在预检查时遍历 40392 条 wav/mel；正式第一次启动建议使用。
    parser.add_argument("--full-file-check", action="store_true")

    # 真正解析终端输入并返回 Namespace。
    return parser.parse_args()


# 读取并核对本项目固定配置；其中独立声码器训练时长等属于明确复现选择。
def load_config(config_path: Path) -> Dict[str, Any]:
    # 读取 JSON 配置并核对论文/官方架构不可随意改变的关键值。
    """读取 JSON 配置并核对论文/官方架构不可随意改变的关键值。"""

    # 配置不存在就停止。
    if not config_path.is_file():
        # 报错原因：配置不存在就停止。
        raise FileNotFoundError(f"Vocoder config not found: {config_path}")

    # UTF-8 读取配置；json.load 把 JSON 对象变为 Python 字典。
    with config_path.open("r", encoding="utf-8") as handle:
        # 载入完整 JSON 配置，随后逐项检查固定约定。
        config: Dict[str, Any] = json.load(handle)

    # 这些值来自 FastVoiceGrad mel 设置和官方 HiFi-GAN V1 架构，正式复现不能误改。
    required_exact = {
        # 记录生成器所用残差块类型。
        "resblock": "1",
        # 记录每批处理的语音数。
        "batch_size": 16,
        # 记录Adam 优化器的学习率。
        "learning_rate": 0.0002,
        # 记录Adam 一阶动量系数。
        "adam_b1": 0.8,
        # 记录Adam 二阶动量系数。
        "adam_b2": 0.99,
        # 记录AdamW 的权重衰减系数。
        "weight_decay": 0.01,
        # 记录每轮学习率衰减倍率。
        "lr_decay": 0.999,
        # 记录将 mel 时间轴逐层扩展为波形的倍率。
        "upsample_rates": [8, 8, 2, 2],
        # 记录转置卷积上采样的核长度。
        "upsample_kernel_sizes": [16, 16, 4, 4],
        # 记录声码器上采样前的隐藏通道数。
        "upsample_initial_channel": 512,
        # 记录残差块的多尺度卷积核长度。
        "resblock_kernel_sizes": [3, 7, 11],
        # 记录残差块的膨胀间隔。
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        # 记录每个声码器训练片段的采样点数。
        "segment_size": 8192,
        # 记录mel 频带数量。
        "num_mels": 80,
        # 记录短时傅里叶变换点数。
        "n_fft": 1024,
        # 记录相邻 mel 帧之间的波形采样点间隔。
        "hop_size": 256,
        # 记录分析窗的采样点长度。
        "win_size": 1024,
        # 记录每秒波形采样点数。
        "sampling_rate": 22050,
        # 记录mel 滤波器最低频率。
        "fmin": 0,
        # 记录mel 滤波器最高频率，空值表示奈奎斯特频率。
        "fmax": 8000,
        # 记录mel 重建损失的最高分析频率。
        "fmax_for_loss": None,
        # 记录生成波形 mel 重建损失系数。
        "mel_loss_weight": 45.0,
        # 记录整句波形峰值归一化后乘 0.95 的规则。
        "training_audio_normalization": "librosa.util.normalize_peak_then_scale_0.95",
        # 记录条件 mel 从归一化随机波形片段现场计算的规则。
        "input_mel_source": "on_the_fly_from_normalized_random_waveform_segment",
        # 记录随机裁切起点的单位，当前为波形采样点。
        "random_crop_unit": "waveform_sample",
        # 记录MRD 各分支的 FFT 点数。
        "mrd_fft_sizes": [1024, 2048, 512],
        # 记录MRD 各分支的频谱帧移。
        "mrd_hop_sizes": [120, 240, 50],
        # 记录MRD 各分支的窗长。
        "mrd_win_lengths": [600, 1200, 240],
        # 记录MRD LeakyReLU 负半轴斜率。
        "mrd_lrelu_slope": 0.2,
    }

    # 逐项比较；报错里同时显示实际值和要求值，便于发现上传错配置。
    for key, expected in required_exact.items():
        # 读取当前配置声明值，缺项会与固定值不符。
        actual = config.get(key)
        # 关键配置不符时中止，不能悄悄续训另一套实验。
        if actual != expected:
            # 报错原因：关键配置不符时中止，不能悄悄续训另一套实验。
            raise ValueError(
                # 指出不满足本项目固定配置约定的参数名；该约定也包括已声明的工程选择。
                f"Paper/official alignment check failed for {key}: "
                # 关键配置不符时中止，不能悄悄续训另一套实验。
                f"got {actual!r}, expected {expected!r}"
            )

    # 返回已经通过检查的配置。
    return config


# 给 Python、NumPy、CPU 与 CUDA 随机过程设置同一基础种子。
def seed_everything(seed: int) -> None:
    # 让 Python、NumPy、PyTorch CPU 和 CUDA 从固定基础随机种子开始。
    """设置主进程随机种子。"""

    # Python random 控制 Dataset 随机裁切。
    random.seed(seed)
    # NumPy 种子控制可能的 NumPy 随机操作。
    np.random.seed(seed)
    # CPU PyTorch 随机数。
    torch.manual_seed(seed)
    # 所有当前可见 CUDA 设备的随机数；本实验只暴露一块 GPU。
    torch.cuda.manual_seed_all(seed)


# 为 DataLoader 子进程设置可重复随机状态并限制 CPU 线程。
def seed_worker(worker_id: int) -> None:
    # 每个数据 worker 有独立可重复随机状态，并限制 CPU 线程避免 STFT 过量竞争。
    """给每个 DataLoader 子进程设置不同但可复现的随机种子。"""

    # PyTorch 会为 worker 安排基础种子；取 32 位后兼容 NumPy seed 范围。
    worker_seed = torch.initial_seed() % (2**32)
    # 每个 worker 的 NumPy 和 Python random 都用自己的种子。
    np.random.seed(worker_seed)
    # 让 worker 内 Python 随机裁切与其指定种子一致。
    random.seed(worker_seed)
    # Conditioning mel is calculated inside each worker.  One Torch CPU thread
    # per worker avoids 4 workers each spawning a full BLAS/STFT thread pool.
    # 每个数据读取 worker 只用一个 Torch CPU 线程，避免 STFT 线程过量竞争。
    torch.set_num_threads(1)


# 先写同目录临时检查点再替换正式文件，减少突然中断留下半文件。
def atomic_torch_save(payload: Dict[str, Any], destination: Path) -> None:
    # 完整状态先序列化到同目录临时文件，再替换 latest 或归档目标。
    """先写临时文件再原子替换，尽量避免 NFS 上留下损坏检查点。"""

    # 确保父目录存在。
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件放同一目录，os.replace 才能做同一文件系统内的原子替换。
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    # torch.save 序列化模型、优化器和训练状态。
    torch.save(payload, temporary)
    # 完整写完后再把临时文件改成正式名字。
    os.replace(temporary, destination)


# 把当前记录步的分项损失与耗时追加到 CSV，供真实曲线绘制。
def append_loss_row(csv_path: Path, row: Dict[str, Any]) -> None:
    # 追加当前记录步的分项损失、耗时和显存，续训保留已有曲线。
    """追加一行训练 loss；第一次写时自动创建表头。"""

    # 文件原来不存在或大小为 0 时需要写表头。
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    # newline="" 避免 Windows 产生空白行；服务器也兼容。
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        # 按本行固定字段顺序创建损失日志写入器。
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        # 新建或空文件才写列名，续训追加不重复表头。
        if write_header:
            # 写出固定列名，让后续续算和汇总能核对格式。
            writer.writeheader()
        # 保存当前记录步的全部分项损失和运行统计。
        writer.writerow(row)
        # flush 让远程查看日志时尽快看到最新点。
        handle.flush()


# 把三个 FFT、帧移和窗长列表逐位置配对，形成三个频谱判别分支。
def build_mrd_resolutions(config: Dict[str, Any]) -> Tuple[STFTResolution, ...]:
    # 把三份并行参数列表按位置配成三个 STFT 分辨率对象。
    """把 JSON 中三个并行列表组合成三个 STFTResolution 对象。"""

    # zip 按相同位置组合 FFT、hop、window。
    return tuple(
        # 将一组频谱分析参数构造成只读分辨率对象。
        STFTResolution(int(fft_size), int(hop_size), int(win_length))
        # 按同一分支下标匹配 FFT、帧移和窗长。
        for fft_size, hop_size, win_length in zip(
            # 各分支频率格精细程度所用 FFT 点数。
            config["mrd_fft_sizes"],
            # 各分支时间观察间隔。
            config["mrd_hop_sizes"],
            # 各分支实际乘 Hann 窗的长度。
            config["mrd_win_lengths"],
        )
    )


# 保存可续训完整状态，按较大间隔另外保存用于推理的轻量 G。
def save_training_checkpoint(
    # 输入参数：声码器检查点保存目录。
    checkpoint_dir: Path,
    # 输入参数：模型配置。
    config: Dict[str, Any],
    # 输入参数：将 mel 转为波形的生成器 G。
    generator: torch.nn.Module,
    # 输入参数：按多个周期检查波形的判别器 MPD。
    mpd: torch.nn.Module,
    # 输入参数：按多种频谱分辨率检查波形的判别器 MRD。
    mrd: torch.nn.Module,
    # 输入参数：生成器优化器状态。
    optim_g: torch.optim.Optimizer,
    # 输入参数：判别器优化器状态。
    optim_d: torch.optim.Optimizer,
    # 输入参数：生成器学习率调度器。
    scheduler_g: torch.optim.lr_scheduler.LRScheduler,
    # 输入参数：判别器学习率调度器。
    scheduler_d: torch.optim.lr_scheduler.LRScheduler,
    # 输入参数：已完成的优化器更新次数。
    global_step: int,
    # 输入参数：训练轮次。
    epoch: int,
    # 输入参数：本轮已经完成的批次数。
    batch_in_epoch: int,
    # 输入参数：训练数据及预处理记录的指纹。
    data_fingerprints: Dict[str, str],
    # 输入参数：训练软件和源码来源记录。
    runtime_provenance: Dict[str, Any],
    # 输入参数：是否额外保留当前生成器长期归档。
    archive_generator: bool,
# 把完整断点及可选 G 归档写到磁盘，函数不另返回对象。
) -> None:
    # latest 保存可继续优化的完整状态，轻量生成器归档仅供推理与阶段比较。
    """覆盖 latest.pt，并按较大间隔保存轻量生成器归档。"""

    # latest.pt 包含继续训练所需的全部状态；每次覆盖，避免磁盘无限增长。
    full_payload: Dict[str, Any] = {
        # 记录检查点结构版本。
        "format_version": CHECKPOINT_FORMAT_VERSION,
        # 记录完整训练状态或仅生成器权重的类型。
        "checkpoint_type": "full_training_state",
        # 记录声码器 G、MPD、MRD 的结构组合。
        "architecture": ARCHITECTURE_NAME,
        # 记录已完成的优化器更新次数。
        "global_step": int(global_step),
        # 记录训练轮次。
        "epoch": int(epoch),
        # 记录本轮已经完成的批次数。
        "batch_in_epoch": int(batch_in_epoch),
        # 记录模型配置。
        "config": config,
        # 把整份生效配置转成稳定内容指纹，后续加载可重算核对。
        "config_semantic_sha256": canonical_json_sha256(config),
        # 记录训练数据及预处理记录的指纹。
        "data_fingerprints": data_fingerprints,
        # 记录训练软件和源码来源记录。
        "runtime_provenance": runtime_provenance,
        # 记录将 mel 转为波形的生成器 G。
        "generator": generator.state_dict(),
        # 记录按多个周期检查波形的判别器 MPD。
        "mpd": mpd.state_dict(),
        # 记录按多种频谱分辨率检查波形的判别器 MRD。
        "mrd": mrd.state_dict(),
        # 记录生成器优化器状态。
        "optim_g": optim_g.state_dict(),
        # 记录判别器优化器状态。
        "optim_d": optim_d.state_dict(),
        # 记录生成器学习率调度器。
        "scheduler_g": scheduler_g.state_dict(),
        # 记录判别器学习率调度器。
        "scheduler_d": scheduler_d.state_dict(),
        # 记录Python 随机数发生器状态。
        "python_random_state": random.getstate(),
        # 记录NumPy 随机数发生器状态。
        "numpy_random_state": np.random.get_state(),
        # 记录PyTorch CPU 随机数发生器状态。
        "torch_random_state": torch.get_rng_state(),
        # 记录各可见 GPU 的随机数发生器状态。
        "cuda_random_state_all": torch.cuda.get_rng_state_all(),
    }
    # 更新最新完整断点；这里包含 G、D、优化器、调度器和随机状态。
    atomic_torch_save(full_payload, checkpoint_dir / "latest.pt")

    # 生成器归档间隔由配置控制；当前 text-unseen 实验固定每 500k steps 归档一次。
    if archive_generator:
        # 轻量归档只保留 G 和完整身份信息，适合推理但不能替代完整续训状态。
        generator_payload: Dict[str, Any] = {
            # 记录检查点结构版本。
            "format_version": CHECKPOINT_FORMAT_VERSION,
            # 记录完整训练状态或仅生成器权重的类型。
            "checkpoint_type": "generator_weights_only",
            # 记录声码器 G、MPD、MRD 的结构组合。
            "architecture": ARCHITECTURE_NAME,
            # 记录已完成的优化器更新次数。
            "global_step": int(global_step),
            # 记录训练轮次。
            "epoch": int(epoch),
            # 记录本轮已经完成的批次数。
            "batch_in_epoch": int(batch_in_epoch),
            # 记录模型配置。
            "config": config,
            # 把整份生效配置转成稳定内容指纹，后续加载可重算核对。
            "config_semantic_sha256": canonical_json_sha256(config),
            # 记录训练数据及预处理记录的指纹。
            "data_fingerprints": data_fingerprints,
            # 记录训练软件和源码来源记录。
            "runtime_provenance": runtime_provenance,
            # 记录将 mel 转为波形的生成器 G。
            "generator": generator.state_dict(),
        }
        # 把累计训练步数写进长期 G 归档文件名。
        archive_path = checkpoint_dir / f"generator_step_{global_step:07d}.pt"
        # 保存这一训练阶段的独立 G 权重，不覆盖较早归档。
        atomic_torch_save(generator_payload, archive_path)


# 在数据、配置和源码身份一致后恢复 G、两个 D 及全部训练状态。
def load_training_checkpoint(
    # 输入参数：模型权重路径。
    checkpoint_path: Path,
    # 输入参数：本次续训准备使用的配置。
    current_config: Dict[str, Any],
    # 输入参数：将 mel 转为波形的生成器 G。
    generator: torch.nn.Module,
    # 输入参数：按多个周期检查波形的判别器 MPD。
    mpd: torch.nn.Module,
    # 输入参数：按多种频谱分辨率检查波形的判别器 MRD。
    mrd: torch.nn.Module,
    # 输入参数：生成器优化器状态。
    optim_g: torch.optim.Optimizer,
    # 输入参数：判别器优化器状态。
    optim_d: torch.optim.Optimizer,
    # 输入参数：生成器学习率调度器。
    scheduler_g: torch.optim.lr_scheduler.LRScheduler,
    # 输入参数：判别器学习率调度器。
    scheduler_d: torch.optim.lr_scheduler.LRScheduler,
    # 输入参数：训练数据及预处理记录的指纹。
    data_fingerprints: Dict[str, str],
    # 输入参数：训练软件和源码来源记录。
    runtime_provenance: Dict[str, Any],
# 依次返回累计更新数、已完成轮数和当前轮已完成批次数。
) -> Tuple[int, int, int]:
    # 严格恢复 G、MPD、MRD 及优化器等状态，返回下一次训练对应的精确游标。
    """恢复 latest.pt，返回 ``(global_step, epoch, batch_in_epoch)``。"""

    # PyTorch 2.6 默认 weights_only=True；完整训练状态含 NumPy/Python RNG，
    # 所以对我们自己生成、路径受控的检查点显式使用 weights_only=False。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 架构标签不符就拒绝加载，防止把旧 MSD 声码器误接到 MRD 实验。
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        # 报错原因：架构标签不符就拒绝加载，防止把旧 MSD 声码器误接到 MRD 实验。
        raise ValueError(
            # 显示归档实际结构版本，防止以错格式恢复。
            f"Checkpoint format_version mismatch: {checkpoint.get('format_version')!r}"
        )
    # 只接受包含优化器等状态的完整断点，轻量 G 无法精确续训。
    if checkpoint.get("checkpoint_type") != "full_training_state":
        # 报错原因：只接受包含优化器等状态的完整断点，轻量 G 无法精确续训。
        raise ValueError(
            # 显示归档类型，轻量 G 不能替代完整续训状态。
            f"Checkpoint type mismatch: {checkpoint.get('checkpoint_type')!r}"
        )
    # 禁止把包含原始 MSD 的其他结构混入本 MRD 实验。
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        # 报错原因：禁止把包含原始 MSD 的其他结构混入本 MRD 实验。
        raise ValueError(
            # 显示实际结构标签，避免 MSD/MRD 不同实验混用。
            f"Checkpoint architecture mismatch: {checkpoint.get('architecture')!r}"
        )
    # 续训数据必须与断点保存时完全一致。
    if checkpoint.get("data_fingerprints") != data_fingerprints:
        # 报错原因：续训数据必须与断点保存时完全一致。
        raise ValueError("Resume checkpoint data_fingerprints do not match active data")
    # 读取断点保存的源码与软件来源。
    checkpoint_provenance = checkpoint.get("runtime_provenance", {})
    # 核对提交号及关键源文件字节指纹。
    for key in ("git_commit", "source_sha256"):
        # 当前代码与断点来源不一致时拒绝自动续训。
        if checkpoint_provenance.get(key) != runtime_provenance.get(key):
            # 报错原因：当前代码与断点来源不一致时拒绝自动续训。
            raise ValueError(f"Resume runtime provenance mismatch for {key}")

    # 继续训练时允许把 max_steps 从 1M 延长到更大，也允许改变日志/保存频率；
    # 但会改变网络、数据片段或优化轨迹的核心值必须与旧检查点完全相同。
    checkpoint_config = checkpoint.get("config", {})
    # 列出会改变网络、数据片段或优化轨迹的续训锁定字段。
    resume_locked_keys = (
        # 字段含义：生成器所用残差块类型。
        "resblock",
        # 字段含义：数据划分协议名称。
        "data_protocol",
        # 字段含义：每批处理的语音数。
        "batch_size",
        # 字段含义：Adam 优化器的学习率。
        "learning_rate",
        # 字段含义：Adam 一阶动量系数。
        "adam_b1",
        # 字段含义：Adam 二阶动量系数。
        "adam_b2",
        # 字段含义：AdamW 的权重衰减系数。
        "weight_decay",
        # 字段含义：每轮学习率衰减倍率。
        "lr_decay",
        # 字段含义：随机种子。
        "seed",
        # 字段含义：数据读取子进程数。
        "num_workers",
        # 字段含义：将 mel 时间轴逐层扩展为波形的倍率。
        "upsample_rates",
        # 字段含义：转置卷积上采样的核长度。
        "upsample_kernel_sizes",
        # 字段含义：声码器上采样前的隐藏通道数。
        "upsample_initial_channel",
        # 字段含义：残差块的多尺度卷积核长度。
        "resblock_kernel_sizes",
        # 字段含义：残差块的膨胀间隔。
        "resblock_dilation_sizes",
        # 字段含义：每个声码器训练片段的采样点数。
        "segment_size",
        # 字段含义：mel 频带数量。
        "num_mels",
        # 字段含义：短时傅里叶变换点数。
        "n_fft",
        # 字段含义：相邻 mel 帧之间的波形采样点间隔。
        "hop_size",
        # 字段含义：分析窗的采样点长度。
        "win_size",
        # 字段含义：每秒波形采样点数。
        "sampling_rate",
        # 字段含义：mel 滤波器最低频率。
        "fmin",
        # 字段含义：mel 滤波器最高频率，空值表示奈奎斯特频率。
        "fmax",
        # 字段含义：mel 重建损失的最高分析频率。
        "fmax_for_loss",
        # 字段含义：生成波形 mel 重建损失系数。
        "mel_loss_weight",
        # 字段含义：整句波形峰值归一化后乘 0.95 的规则。
        "training_audio_normalization",
        # 字段含义：条件 mel 从归一化随机波形片段现场计算的规则。
        "input_mel_source",
        # 字段含义：随机裁切起点的单位，当前为波形采样点。
        "random_crop_unit",
        # 字段含义：MRD 各分支的 FFT 点数。
        "mrd_fft_sizes",
        # 字段含义：MRD 各分支的频谱帧移。
        "mrd_hop_sizes",
        # 字段含义：MRD 各分支的窗长。
        "mrd_win_lengths",
        # 字段含义：MRD LeakyReLU 负半轴斜率。
        "mrd_lrelu_slope",
        # 字段含义：本划分预期训练语音总数。
        "expected_train_utterances",
        # 字段含义：本划分预期训练说话人数。
        "expected_train_speakers",
        # 字段含义：测试语音数，仅核对划分，不参与训练。
        "expected_test_utterances",
        # 字段含义：未见测试说话人数。
        "expected_test_speakers",
        # 字段含义：训练数据及预处理记录的指纹。
        "data_fingerprints",
    )
    # 逐个检查续训过程中不能变更的核心参数。
    for key in resume_locked_keys:
        # 任一关键值变化都会中止，避免混合两种训练约定。
        if checkpoint_config.get(key) != current_config.get(key):
            # 报错原因：任一关键值变化都会中止，避免混合两种训练约定。
            raise ValueError(
                # 定位续训时被改变的核心配置字段。
                f"Resume config mismatch for {key}: "
                # 任一关键值变化都会中止，避免混合两种训练约定。
                f"checkpoint={checkpoint_config.get(key)!r}, "
                # 任一关键值变化都会中止，避免混合两种训练约定。
                f"current={current_config.get(key)!r}"
            )

    # 恢复三个网络权重。
    generator.load_state_dict(checkpoint["generator"], strict=True)
    # 严格恢复 MPD 权重，不能把判别器重新随机初始化。
    mpd.load_state_dict(checkpoint["mpd"], strict=True)
    # 严格恢复 MRD 权重，保留之前学到的真假判断。
    mrd.load_state_dict(checkpoint["mrd"], strict=True)

    # 恢复优化器和学习率调度器，才能真正从断点继续而不是只加载权重重练。
    optim_g.load_state_dict(checkpoint["optim_g"])
    # 恢复两个判别器共有优化器的动量等状态。
    optim_d.load_state_dict(checkpoint["optim_d"])
    # 恢复 G 的学习率调度进度。
    scheduler_g.load_state_dict(checkpoint["scheduler_g"])
    # 恢复 D 的学习率调度进度。
    scheduler_d.load_state_dict(checkpoint["scheduler_d"])

    # 尽量恢复随机状态；这样续训的随机序列更接近不中断的运行。
    random.setstate(checkpoint["python_random_state"])
    # 恢复 NumPy 随机序列。
    np.random.set_state(checkpoint["numpy_random_state"])
    # 恢复 PyTorch CPU 随机序列。
    torch.set_rng_state(checkpoint["torch_random_state"])
    # 恢复每个可见 GPU 的随机序列。
    torch.cuda.set_rng_state_all(checkpoint["cuda_random_state_all"])

    # 返回下一步训练需要的计数。
    global_step = int(checkpoint["global_step"])
    # 设置训练轮次。
    epoch = int(checkpoint["epoch"])
    # 设置本轮已经完成的批次数。
    batch_in_epoch = int(checkpoint["batch_in_epoch"])
    # 累计步数、轮次与批次游标都必须非负。
    if global_step < 0 or epoch < 0 or batch_in_epoch < 0:
        # 报错原因：累计步数、轮次与批次游标都必须非负。
        raise ValueError("Checkpoint step/epoch/batch cursor must be non-negative")
    # 断点不能已经超过本次声明的训练终点。
    if global_step > int(current_config["max_steps"]):
        # 报错原因：断点不能已经超过本次声明的训练终点。
        raise ValueError(
            # 断点不能已经超过本次声明的训练终点。
            f"Checkpoint step {global_step} exceeds max_steps={current_config['max_steps']}"
        )
    # 设置每轮完整批次数。
    batches_per_epoch = (
        # 按断点锁定的训练条数核算每轮可用批次。
        int(current_config["expected_train_utterances"])
        # 尾部不足一批丢弃，批次数向下取整。
        // int(current_config["batch_size"])
    )
    # 轮内游标不能越过该轮末尾。
    if batch_in_epoch > batches_per_epoch:
        # 报错原因：轮内游标不能越过该轮末尾。
        raise ValueError(
            # 轮内游标不能越过该轮末尾。
            f"Checkpoint batch_in_epoch={batch_in_epoch} exceeds {batches_per_epoch}"
        )
    # 从轮次与轮内位置独立重算累计步数。
    expected_global_step = epoch * batches_per_epoch + batch_in_epoch
    # 进度字段互相矛盾时拒绝恢复。
    if global_step != expected_global_step:
        # 报错原因：进度字段互相矛盾时拒绝恢复。
        raise ValueError(
            # 错误信息说明累计步数与轮次、批次游标互相矛盾。
            "Checkpoint cursor is inconsistent: "
            # 进度字段互相矛盾时拒绝恢复。
            f"global_step={global_step}, epoch={epoch}, batch_in_epoch={batch_in_epoch}, "
            # 进度字段互相矛盾时拒绝恢复。
            f"expected_global_step={expected_global_step}"
        )
    # 返回准确恢复位置，主循环从下一未完成批次继续。
    return global_step, epoch, batch_in_epoch


# 执行本脚本的输入检查、主要处理和结果记录。
def main() -> None:
    # 执行本脚本的输入检查、主要处理和结果记录。
    """程序入口：预检查、建模、恢复检查点并训练到 max_steps。"""

    # 设置是否收到保存断点后安全停机的请求。
    safe_stop_requested = False

    # 收到普通中断或终止信号时先设置安全停机标志。
    def request_safe_stop(signum, _frame) -> None:
        # 修改外层训练入口持有的停机标志。
        nonlocal safe_stop_requested
        # 显示收到的停止信号，并说明将先保存可续训状态。
        print(f"Signal {signum} received; requesting resumable shutdown...", flush=True)
        # 收到停止请求后，在这次完整更新结束的位置退出。
        if safe_stop_requested:
            # 触发现有中断处理，尽力保存可续训状态。
            raise KeyboardInterrupt
        # 首次信号不打断半次更新，等当前 G/D 成对更新完成。
        safe_stop_requested = True

    # Background shells may inherit SIGINT=ignored.  Register both handlers
    # explicitly so future stop requests save latest.pt first.
    # 显式注册 Ctrl+C 信号，后台启动也能走保存路径。
    signal.signal(signal.SIGINT, request_safe_stop)
    # 普通 kill 请求同样转成安全保存停机。
    signal.signal(signal.SIGTERM, request_safe_stop)

    # 读取命令行。
    args = parse_args()

    # resolve() 得到规范绝对路径，后续日志更容易核对。
    project_root = args.project_root.resolve()
    # 真正建模和训练只能在当前租用服务器执行。
    assert_rented_server(project_root)
    # 设置模型配置路径。
    config_path = (
        # 用户显式指定配置时使用其准确绝对路径。
        args.config.resolve()
        # 仅在明确传入配置文件时覆盖默认 text-unseen 配置。
        if args.config is not None
        # 默认采用当前已固定的 text-unseen 声码器设置。
        else project_root / "hifigan" / "configs" / "text_unseen.json"
    )
    # 读取并严格验证架构配置。
    config = load_config(config_path)
    # 配置必须明确声明数据协议；这样不会拿 text-seen 清单误续训 text-unseen 权重。
    data_protocol = str(config.get("data_protocol", "")).strip()
    # 只接受已定义的数据协议，未知名称不能继续。
    if data_protocol not in {"text-unseen", "text-seen"}:
        # 报错原因：只接受已定义的数据协议，未知名称不能继续。
        raise ValueError(
            # 只接受已定义的数据协议，未知名称不能继续。
            "Vocoder config must set data_protocol to 'text-unseen' or 'text-seen'"
        )
    # 将协议名转换为检查点目录的固定标签。
    protocol_tag = data_protocol.replace("-", "_")
    # 设置声码器检查点保存目录。
    checkpoint_dir = (
        # 用户显式指定时使用独立检查点目录。
        args.checkpoint_dir.resolve()
        # 没有指定输出目录时才使用协议对应的默认位置。
        if args.checkpoint_dir is not None
        # 按 seen/unseen 协议分别保存声码器检查点。
        else project_root / "exp" / "checkpoints" / f"hifigan_v1_mrd_vctk_{protocol_tag}"
    )

    # 检查是否有人传入会覆盖正式训练规模的参数。
    override_requested = any(
        # 只要显式传过覆盖值，就算请求改变正式参数。
        value is not None
        # 检查总步数、批大小和数据 worker 三个可诊断覆盖项。
        for value in (args.max_steps, args.batch_size, args.num_workers)
    )
    # 正式训练拒绝参数覆盖，只有独立冒烟目录允许。
    if override_requested and not args.diagnostic_smoke:
        # 报错原因：正式训练拒绝参数覆盖，只有独立冒烟目录允许。
        raise ValueError(
            # 正式训练拒绝参数覆盖，只有独立冒烟目录允许。
            "Step/batch/worker overrides require --diagnostic-smoke and a separate exp/smoke path"
        )

    # 冒烟测试可以覆盖步数/batch/worker；正式运行绝不接受这些覆盖。
    if args.max_steps is not None:
        # 把诊断步数写进本次有效配置和来源记录。
        config["max_steps"] = int(args.max_steps)
    # 诊断可单独试验显存可承受的批大小。
    if args.batch_size is not None:
        # 保存本次实际使用的诊断批大小。
        config["batch_size"] = int(args.batch_size)
    # 诊断时可调整文件读取并行数。
    if args.num_workers is not None:
        # 保存实际数据读取进程数。
        config["num_workers"] = int(args.num_workers)

    # 覆盖值也必须是正数。
    if int(config["max_steps"]) <= 0:
        # 报错原因：覆盖值也必须是正数。
        raise ValueError("max_steps must be positive")
    # 批大小必须为正。
    if int(config["batch_size"]) <= 0:
        # 报错原因：批大小必须为正。
        raise ValueError("batch_size must be positive")
    # 读取 worker 可为零表示主进程加载，但不能为负。
    if int(config["num_workers"]) < 0:
        # 报错原因：读取 worker 可为零表示主进程加载，但不能为负。
        raise ValueError("num_workers cannot be negative")

    # 诊断产物只允许存放在这一独立子目录。
    smoke_root = (project_root / "exp" / "smoke").resolve()
    # 小规模诊断采用单独输出位置，不能覆盖正式断点。
    if args.diagnostic_smoke:
        # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
        try:
            # 确认诊断检查点路径确实位于 exp/smoke 内。
            checkpoint_dir.relative_to(smoke_root)
        # 诊断路径不在指定范围时中止，不写入正式实验目录。
        except ValueError as error:
            # 报错原因：诊断路径不在指定范围时中止，不写入正式实验目录。
            raise ValueError(
                # 同时给出允许的诊断根目录和当前请求目录。
                f"Diagnostic checkpoint directory must be below {smoke_root}: {checkpoint_dir}"
            # 保留原始底层异常作为原因，方便追查路径或参数检查为何失败。
            ) from error
    # 正式 text-unseen 声码器采用这次已确认的完整训练规模。
    elif data_protocol == "text-unseen":
        # 总步数 250 万、batch 16、worker 4 是当前实验固定约定。
        formal_requirements = {
            # 记录声码器计划训练总更新次数。
            "max_steps": 2_500_000,
            # 记录每批处理的语音数。
            "batch_size": 16,
            # 记录数据读取子进程数。
            "num_workers": 4,
        }
        # 逐项核对正式训练规模。
        for key, expected in formal_requirements.items():
            # 发现正式配置被改动时停止。
            if int(config[key]) != expected:
                # 报错原因：发现正式配置被改动时停止。
                raise ValueError(
                    # 发现正式配置被改动时停止。
                    f"Formal text-unseen run requires {key}={expected}, got {config[key]}"
                )

    # 仅检查模式不进入真正建模和优化流程。
    if args.preflight_only:
        # 只预检查不能同时声明真正新训或续训，避免模式含糊。
        if args.fresh or args.resume is not None:
            # 报错原因：只预检查不能同时声明真正新训或续训，避免模式含糊。
            raise ValueError("--preflight-only must not include --fresh or --resume")
    # 正式训练必须明确且仅选择新训或完整断点续训其中一种。
    elif (1 if args.fresh else 0) + (1 if args.resume is not None else 0) != 1:
        # 报错原因：正式训练必须明确且仅选择新训或完整断点续训其中一种。
        raise ValueError("Training requires exactly one explicit mode: --fresh or --resume latest.pt")

    # 正式只允许一块可见 GPU；CUDA_VISIBLE_DEVICES=4 后程序内部称它为 cuda:0。
    if not torch.cuda.is_available():
        # 报错原因：正式只允许一块可见 GPU；CUDA_VISIBLE_DEVICES=4 后程序内部称它为 cuda:0。
        raise RuntimeError("CUDA is not available; vocoder training requires a GPU")
    # 只允许一块显式可见 GPU，避免占用未授权卡。
    if torch.cuda.device_count() != 1:
        # 报错原因：只允许一块显式可见 GPU，避免占用未授权卡。
        raise RuntimeError(
            # 只允许一块显式可见 GPU，避免占用未授权卡。
            f"Expected exactly one visible GPU, got {torch.cuda.device_count()}. "
            # 只允许一块显式可见 GPU，避免占用未授权卡。
            "Set CUDA_VISIBLE_DEVICES to one free physical GPU."
        )

    # 内部设备编号永远是 0；它映射到外部 CUDA_VISIBLE_DEVICES 指定的物理 GPU 4。
    device = torch.device("cuda:0")

    # 设种子，然后允许 cuDNN 为固定卷积形状选择高性能算法；这与官方 HiFi-GAN 一致。
    seed_everything(int(config["seed"]))
    # 让 cuDNN 为固定卷积形状选择快算法；不保证跨环境逐位相同。
    torch.backends.cudnn.benchmark = True

    # 设置训练数据及预处理记录的指纹。
    data_fingerprints = collect_and_verify_data_fingerprints(project_root, config)
    # 设置训练软件和源码来源记录。
    runtime_provenance = collect_runtime_provenance(project_root)
    # 把已经生效的完整配置指纹补进运行来源。
    runtime_provenance["config_semantic_sha256"] = canonical_json_sha256(config)

    # 先读 02 脚本生成的审计报告，拒绝“配置说 text-seen、活动清单却是另一套”。
    metadata_dir = project_root / "data" / "metadata"
    # 定位实际发布的数据划分记录。
    protocol_report_path = metadata_dir / "split_protocol.json"
    # 没有划分报告不能确认训练清单的协议归属。
    if not protocol_report_path.is_file():
        # 报错原因：没有划分报告不能确认训练清单的协议归属。
        raise FileNotFoundError(f"Split protocol report not found: {protocol_report_path}")
    # 读取当前生效划分的审计记录。
    with protocol_report_path.open("r", encoding="utf-8") as handle:
        # 载入划分协议和样本数量事实。
        protocol_report = json.load(handle)
    # 从真实划分报告取得协议名称。
    active_protocol = str(protocol_report.get("protocol_cli_name", "")).strip()
    # 配置与实际清单协议不同就停止，避免 seen/unseen 混训。
    if active_protocol != data_protocol:
        # 报错原因：配置与实际清单协议不同就停止，避免 seen/unseen 混训。
        raise ValueError(
            # 配置与实际清单协议不同就停止，避免 seen/unseen 混训。
            f"Vocoder config protocol={data_protocol!r}, "
            # 配置与实际清单协议不同就停止，避免 seen/unseen 混训。
            f"but active manifests use {active_protocol!r}"
        )

    # 清单路径固定为 train_manifest.csv，不读取 all_manifest/test_manifest。
    manifest_path = metadata_dir / "train_manifest.csv"
    # 设置训练语音条目。
    items = load_training_items(project_root, manifest_path)

    # 统计并核对配置声明的训练语音数和 100 位训练说话人；语音数必须在官方
    # VCTK 0.92 mic1 split 生成后按 split_protocol.json 的审计值更新配置。
    speakers = sorted({item.speaker for item in items})
    # 取得固定划分下应有的训练语音数。
    expected_utterances = int(config["expected_train_utterances"])
    # 取得训练说话人数，用于防止清单错版。
    expected_speakers = int(config["expected_train_speakers"])
    # 训练清单实际条数必须与审计后的配置一致。
    if len(items) != expected_utterances:
        # 报错原因：训练清单实际条数必须与审计后的配置一致。
        raise ValueError(
            # 显示实际与预期训练条数，检查是否上传了另一份清单。
            f"Training utterance count mismatch: got {len(items)}, expected {expected_utterances}"
        )
    # 训练说话人总数不符时停止。
    if len(speakers) != expected_speakers:
        # 报错原因：训练说话人总数不符时停止。
        raise ValueError(
            # 显示训练说话人数差异，防止数据划分错版。
            f"Training speaker count mismatch: got {len(speakers)}, expected {expected_speakers}"
        )

    # 这里只读取 test_manifest 的身份字段来证明 train/test 说话人不重叠；
    # 不读取测试 wav/mel，更不会把测试数据交给 DataLoader 或反向传播。
    test_manifest_path = project_root / "data" / "metadata" / "test_manifest.csv"
    # 需要测试身份清单来证明没有训练/测试说话人重叠。
    if not test_manifest_path.is_file():
        # 报错原因：不读取测试 wav/mel，更不会把测试数据交给 DataLoader 或反向传播。
        raise FileNotFoundError(f"Test manifest not found: {test_manifest_path}")
    # 仅暂存测试身份元信息用于隔离审计，不加载测试波形训练。
    test_rows = []
    # 只读取测试清单字段以核对无训练泄漏。
    with test_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        # 逐行核对测试身份，并在出错时给出 CSV 行号。
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            # 测试清单中不能混入训练身份行。
            if row.get("split", "").strip().lower() != "test":
                # 报错原因：测试清单中不能混入训练身份行。
                raise ValueError(
                    # 测试清单中不能混入训练身份行。
                    f"Non-test row found in test manifest at line {line_number}"
                )
            # 测试身份必须明确标为未见说话人。
            if row.get("is_unseen_speaker", "").strip() not in {"1", "True", "true"}:
                # 报错原因：测试身份必须明确标为未见说话人。
                raise ValueError(
                    # 测试身份必须明确标为未见说话人。
                    f"Test row is not marked unseen at line {line_number}"
                )
            # 保存这条已通过身份检查的测试元信息。
            test_rows.append(row)
    # 提取去重后的测试说话人集合。
    test_speakers = sorted({row["speaker"].strip() for row in test_rows})
    # 核对固定测试清单条数，不读取其音频参与优化。
    if len(test_rows) != int(config["expected_test_utterances"]):
        # 报错原因：核对固定测试清单条数，不读取其音频参与优化。
        raise ValueError(
            # 报告身份审计读到的测试语音数。
            f"Test utterance count mismatch: got {len(test_rows)}, "
            # 核对固定测试清单条数，不读取其音频参与优化。
            f"expected {config['expected_test_utterances']}"
        )
    # 核对未见测试说话人数。
    if len(test_speakers) != int(config["expected_test_speakers"]):
        # 报错原因：核对未见测试说话人数。
        raise ValueError(
            # 报告测试说话人集合大小，用于核对固定划分。
            f"Test speaker count mismatch: got {len(test_speakers)}, "
            # 核对未见测试说话人数。
            f"expected {config['expected_test_speakers']}"
        )
    # 计算训练与测试说话人交集。
    overlap = sorted(set(speakers) & set(test_speakers))
    # 出现任何说话人重叠都会被视为泄漏并停止。
    if overlap:
        # 报错原因：出现任何说话人重叠都会被视为泄漏并停止。
        raise ValueError(f"Train/test speaker leakage detected: {overlap}")

    # 本实验协议没有 validation；旧 val_manifest 若重新出现就拒绝开跑。
    validation_manifest_path = project_root / "data" / "metadata" / "val_manifest.csv"
    # 此固定实验没有验证划分，拒绝意外残留的旧验证清单。
    if validation_manifest_path.exists():
        # 报错原因：此固定实验没有验证划分，拒绝意外残留的旧验证清单。
        raise ValueError(
            # 指出与本次无验证集约定冲突的残留清单路径。
            f"Validation manifest must not exist for this protocol: {validation_manifest_path}"
        )

    # 检查实际 wav/mel 文件；正式第一次启动使用 --full-file-check。
    missing_wav, missing_mel = check_item_files(items, full=args.full_file_check)
    # 必要数据文件缺失时中止，不让 DataLoader 运行后才失败。
    if missing_wav or missing_mel:
        # 报错原因：必要数据文件缺失时中止，不让 DataLoader 运行后才失败。
        raise FileNotFoundError(
            # 分别报告缺失波形与 mel 的数量。
            f"Feature file check failed: missing_wav={missing_wav}, missing_mel={missing_mel}"
        )

    # 把所有关键事实打印到训练日志。
    print("=" * 80, flush=True)
    # 日志标明开始自训声码器的输入和环境核对。
    print("PAPER-ARCHITECTURE-ALIGNED VOCODER PREFLIGHT", flush=True)
    # 向终端显示租用服务器上的项目根目录。
    print(f"project_root       = {project_root}", flush=True)
    # 向终端显示模型配置路径。
    print(f"config_path        = {config_path}", flush=True)
    # 向终端显示声码器检查点保存目录。
    print(f"checkpoint_dir     = {checkpoint_dir}", flush=True)
    # 显示本次真实训练清单路径。
    print(f"manifest           = {manifest_path}", flush=True)
    # 向终端显示数据划分协议名称。
    print(f"data_protocol      = {data_protocol}", flush=True)
    # 显示经过隔离核对的实际训练语音数。
    print(f"train_utterances   = {len(items)}", flush=True)
    # 向终端显示训练清单中的说话人集合。
    print(f"train_speakers     = {len(speakers)}", flush=True)
    # 显示仅用于身份审计的测试语音数，测试音频不进入训练。
    print(f"test_utterances    = {len(test_rows)} (metadata audit only)", flush=True)
    # 显示未见测试说话人数，供训练/测试隔离核对。
    print(f"test_speakers      = {len(test_speakers)} (metadata audit only)", flush=True)
    # 向终端显示当前独立声码器没有验证集，也不按测试指标挑选 checkpoint；未公开的作者选择规则不能猜作事实。
    print("validation         = not used", flush=True)
    # 日志再次声明禁止把测试数据用于训练。
    print("test_data_training = forbidden", flush=True)
    # 显示进程实际可见 GPU 数，核对单卡限制。
    print(f"visible_gpu_count  = {torch.cuda.device_count()}", flush=True)
    # 显示本次实际计算的 GPU 型号。
    print(f"visible_gpu_name   = {torch.cuda.get_device_name(0)}", flush=True)
    # 向终端显示每批处理的语音数。
    print(f"batch_size         = {config['batch_size']}", flush=True)
    # 向终端显示声码器计划训练总更新次数。
    print(f"max_steps          = {config['max_steps']}", flush=True)
    # 显示整份生效配置的稳定内容指纹。
    print(f"config_semantic_sha256 = {canonical_json_sha256(config)}", flush=True)
    # 向终端显示生成当前代码的 Git 提交编号。
    print(f"git_commit         = {runtime_provenance['git_commit']}", flush=True)
    # 显示当前训练清单内容指纹，便于核对是否换过数据。
    print(
        # 当前数据、路径或模型状态不满足明确的输入要求。
        f"train_manifest_sha256 = {data_fingerprints['train_manifest_sha256']}",
        # 立即刷新终端缓冲，远程查看日志时可及时看到进度或错误。
        flush=True,
    )
    # 向终端显示声码器 G、MPD、MRD 的结构组合。
    print("architecture       = HiFi-GAN V1 Generator + MPD + UnivNet MRD", flush=True)
    # 显示原 MSD 已被 MRD 替换的结构事实。
    print("replacement        = original MSD removed; MRD inserted", flush=True)
    # 画出日志分隔线，便于区分启动核对与训练进度。
    print("=" * 80, flush=True)

    # 只做预检查时到这里正常结束，不创建实验目录。
    if args.preflight_only:
        # 只在配置、数据隔离和设备检查通过后显示预检查完成。
        print("VOCODER PREFLIGHT PASSED", flush=True)
        # 只预检查到此结束，不创建模型或实验产物目录。
        return

    # Fresh 要求目录不存在或完全为空；resume 必须显式指向本目录的 latest.pt。
    latest_checkpoint = checkpoint_dir / "latest.pt"
    # 新训要求空目录，保护已有实验权重与日志。
    if args.fresh and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        # 报错原因：新训要求空目录，保护已有实验权重与日志。
        raise FileExistsError(f"--fresh requires an empty directory: {checkpoint_dir}")
    # 用户明确要求续训时核对完整断点和配置快照。
    if args.resume is not None:
        # 解析用户指定的续训文件为准确绝对路径。
        resume_path = args.resume.resolve()
        # 只允许该实验目录的 latest.pt，不猜测或自动选择其他文件。
        if resume_path != latest_checkpoint.resolve():
            # 报错原因：只允许该实验目录的 latest.pt，不猜测或自动选择其他文件。
            raise ValueError(
                # 只允许该实验目录的 latest.pt，不猜测或自动选择其他文件。
                f"--resume must point exactly to {latest_checkpoint.resolve()}, got {resume_path}"
            )
        # 所指完整断点必须存在。
        if not resume_path.is_file():
            # 报错原因：所指完整断点必须存在。
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        # 续训同时读取原实验旁边保存的配置快照。
        config_snapshot = checkpoint_dir / "config.json"
        # 没有原始生效配置就不能安全恢复。
        if not config_snapshot.is_file():
            # 报错原因：没有原始生效配置就不能安全恢复。
            raise FileNotFoundError(f"Resume config snapshot not found: {config_snapshot}")
        # 读取原实验完整配置。
        saved_config = json.loads(config_snapshot.read_text(encoding="utf-8"))
        # 外部本次配置必须与保存快照精确一致。
        if saved_config != config:
            # 报错原因：外部本次配置必须与保存快照精确一致。
            raise ValueError("Resume config.json does not exactly match the effective config")

    # 建立独立声码器实验目录。
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存本次“实际生效”的配置，包括任何冒烟测试覆盖项。
    # config.json 这个文件名与官方 HiFi-GAN 检查点目录一致，后续 vocode 脚本在
    # 指定 generator_step_xxx.pt 后就能自动找到它，不必再手写第二条配置路径。
    if args.fresh:
        # 新训先保存完整生效配置，后续权重加载与续训都要核对它。
        atomic_write_json(checkpoint_dir / "config.json", config)
        # 保存结构、数据和源码共同定义的运行约定。
        atomic_write_json(
            # 保存本实验结构、配置、数据和源码的联合约定。
            checkpoint_dir / "run_contract.json",
            {
                # 记录声码器 G、MPD、MRD 的结构组合。
                "architecture": ARCHITECTURE_NAME,
                # 记录归档结构的版本号。
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                # 记录模型配置。
                "config": config,
                # 把整份生效配置转成稳定内容指纹，后续加载可重算核对。
                "config_semantic_sha256": canonical_json_sha256(config),
                # 记录训练数据及预处理记录的指纹。
                "data_fingerprints": data_fingerprints,
                # 记录训练软件和源码来源记录。
                "runtime_provenance": runtime_provenance,
            },
        )

    # 从审计过的官方 hifi-gan 目录取出生成器、MPD 和损失函数。
    (
        # 将字典配置包装成官方网络可用的属性访问形式。
        AttrDict,
        # 官方 HiFi-GAN V1 生成器，将 mel 转成时域波形。
        Generator,
        # 官方多周期判别器，检查波形的不同周期结构。
        MultiPeriodDiscriminator,
        # 官方最小二乘判别器损失，真趋向 1、假趋向 0。
        discriminator_loss,
        # 官方逐层特征匹配损失，层内均值后求和并乘 2。
        feature_loss,
        # 官方生成器对抗损失，推动假分数接近 1。
        generator_loss,
    # 按前面顺序接收官方 G、MPD、配置包装器和三个损失。
    ) = import_official_hifigan(project_root)

    # AttrDict 让官方 Generator 可以用 h.upsample_rates 方式读取 JSON 字段。
    h = AttrDict(config)

    # 建立官方 HiFi-GAN V1 Generator 并移到当前可见 GPU。
    generator = Generator(h).to(device)
    # 建立官方 HiFi-GAN MPD；它从周期 2/3/5/7/11 的波形视角判断真假。
    mpd = MultiPeriodDiscriminator().to(device)
    # 建立论文要求替换进来的三分辨率 MRD。
    mrd = MultiResolutionDiscriminator(
        # 设置MRD 的多组时频分辨率。
        resolutions=build_mrd_resolutions(config),
        # MRD 使用配置中固定的负半轴斜率。
        lrelu_slope=float(config["mrd_lrelu_slope"]),
    # 将新建判别器及其 Hann 窗缓冲区送到指定 GPU。
    ).to(device)

    # 官方 HiFi-GAN V1 使用 AdamW，学习率 2e-4，betas=(0.8,0.99)。
    optim_g = torch.optim.AdamW(
        # G 优化器只持有生成器参数。
        generator.parameters(),
        # 按声码器固定配置设置 AdamW 学习率。
        lr=float(config["learning_rate"]),
        # 分别指定 Adam 一阶与二阶动量系数。
        betas=(float(config["adam_b1"]), float(config["adam_b2"])),
        # 设置AdamW 的权重衰减系数。
        weight_decay=float(config["weight_decay"]),
    )
    # MPD 与 MRD 都属于判别器，所以由同一个 AdamW 优化器更新。
    optim_d = torch.optim.AdamW(
        # D 优化器同时持有 MPD 和 MRD 参数，不包含 G。
        itertools.chain(mpd.parameters(), mrd.parameters()),
        # 按声码器固定配置设置 AdamW 学习率。
        lr=float(config["learning_rate"]),
        # 分别指定 Adam 一阶与二阶动量系数。
        betas=(float(config["adam_b1"]), float(config["adam_b2"])),
        # 设置AdamW 的权重衰减系数。
        weight_decay=float(config["weight_decay"]),
    )

    # 官方代码每完成一个 epoch，把学习率乘 0.999。
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        # 传入生成器优化器状态。
        optim_g,
        # 只有完成一轮时乘以该学习率衰减系数。
        gamma=float(config["lr_decay"]),
    )
    # 设置判别器学习率调度器。
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        # 传入判别器优化器状态。
        optim_d,
        # 只有完成一轮时乘以该学习率衰减系数。
        gamma=float(config["lr_decay"]),
    )

    # 默认从第 0 步、第 0 epoch 开始；只有显式 --resume 才恢复完整状态。
    global_step = 0
    # 新实验初始已完成轮次为零。
    start_epoch = 0
    # 新实验从第一批开始；续训会以检查点覆盖该游标。
    start_batch_in_epoch = 0
    # 用户明确要求续训时核对完整断点和配置快照。
    if args.resume is not None:
        # 恢复完整状态并覆盖零起点，训练从最后已保存进度继续。
        global_step, start_epoch, start_batch_in_epoch = load_training_checkpoint(
            # 传入这次已核对路径的完整 latest.pt。
            latest_checkpoint,
            # 传入模型配置。
            config,
            # 传入将 mel 转为波形的生成器 G。
            generator,
            # 传入按多个周期检查波形的判别器 MPD。
            mpd,
            # 传入按多种频谱分辨率检查波形的判别器 MRD。
            mrd,
            # 传入生成器优化器状态。
            optim_g,
            # 传入判别器优化器状态。
            optim_d,
            # 传入生成器学习率调度器。
            scheduler_g,
            # 传入判别器学习率调度器。
            scheduler_d,
            # 传入训练数据及预处理记录的指纹。
            data_fingerprints,
            # 传入训练软件和源码来源记录。
            runtime_provenance,
        )
        # 显示真正恢复的检查点、累计步数和轮内游标。
        print(
            # 日志显示真正恢复的完整断点路径与累计步数。
            f"RESUMED: checkpoint={latest_checkpoint}, step={global_step}, "
            # 传入训练软件和源码来源记录。
            f"epoch={start_epoch}, batch_in_epoch={start_batch_in_epoch}",
            # 立即刷新终端缓冲，远程查看日志时可及时看到进度或错误。
            flush=True,
        )
    # 此分支：Dataset 每次按官方 scratch 路径给出现场计算的 80x32 mel 和 8192 点波形。
    else:
        # 日志明确本次从随机初始化开始，不能把新训误认为断点续训。
        print("STARTING FROM SCRATCH", flush=True)

    # Dataset 每次按官方 scratch 路径给出现场计算的 80x32 mel 和 8192 点波形。
    dataset = ManifestMelWaveDataset(
        # 设置训练语音条目。
        items=items,
        # 设置每个声码器训练片段的采样点数。
        segment_size=int(config["segment_size"]),
        # 设置相邻 mel 帧之间的波形采样点间隔。
        hop_size=int(config["hop_size"]),
        # 设置mel 频带数量。
        num_mels=int(config["num_mels"]),
        # 设置每秒波形采样点数。
        sampling_rate=int(config["sampling_rate"]),
        # 设置短时傅里叶变换点数。
        n_fft=int(config["n_fft"]),
        # 设置分析窗的采样点长度。
        win_size=int(config["win_size"]),
        # 设置mel 滤波器最低频率。
        fmin=int(config["fmin"]),
        # 设置mel 滤波器最高频率，空值表示奈奎斯特频率。
        fmax=config["fmax"],
    )

    # 切换到训练模式，启用需要的训练行为。
    generator.train()
    # 将 MPD 设为训练模式。
    mpd.train()
    # 将 MRD 设为训练模式。
    mrd.train()

    # 先真正初始化 CUDA 上下文，再重置峰值显存计数；避免之前出现 Invalid device argument。
    _ = torch.empty(1, device=device)
    # 清零本轮监测的峰值显存基线。
    torch.cuda.reset_peak_memory_stats(device)

    # loss CSV 与检查点同目录，绘图不需要解析不断刷新的终端进度条。
    loss_csv_path = checkpoint_dir / "training_losses.csv"

    # 从配置读取循环控制参数。
    max_steps = int(config["max_steps"])
    # 按固定间隔输出训练损失记录。
    log_every = int(config["log_every_steps"])
    # 按固定间隔保存能恢复优化器的完整状态。
    save_every = int(config["save_every_steps"])
    # 按较长间隔保留独立轻量 G 归档。
    archive_every = int(config["archive_generator_every_steps"])

    # epoch 从恢复位置开始；while 让训练以“总步数”作为真正终止条件。
    epoch = start_epoch
    # 设置本轮已经完成的批次数。
    batch_in_epoch = start_batch_in_epoch
    # 设置每轮完整批次数。
    batches_per_epoch = len(dataset) // int(config["batch_size"])
    # 数据不足一个完整批次时禁止开始循环。
    if batches_per_epoch <= 0:
        # 报错原因：数据不足一个完整批次时禁止开始循环。
        raise ValueError("Training dataset is smaller than one batch")
    # 轮内游标不能越过该轮末尾。
    if batch_in_epoch > batches_per_epoch:
        # 报错原因：轮内游标不能越过该轮末尾。
        raise ValueError(
            # 轮内游标不能越过该轮末尾。
            f"Resume batch cursor {batch_in_epoch} exceeds {batches_per_epoch}"
        )
    # 记录上次日志时刻，计算接下来每步实际耗时。
    last_log_time = time.time()

    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # global_step 达到 max_steps 后循环结束。
        while global_step < max_steps:
            # 设置按轮次乱序并能从批次游标继续的采样器。
            batch_sampler = DeterministicEpochBatchSampler(
                # 设置训练数据集语音数量。
                dataset_size=len(dataset),
                # 设置每批处理的语音数。
                batch_size=int(config["batch_size"]),
                # 设置随机种子。
                seed=int(config["seed"]),
                # 设置训练轮次。
                epoch=epoch,
                # 设置续训时本轮第一个尚未完成的批次。
                start_batch=batch_in_epoch,
            )
            # 为 DataLoader worker 分配独立随机数发生器。
            worker_generator = torch.Generator()
            # 把 worker 随机序列与采样排列种子分离，并按轮次固定。
            worker_generator.manual_seed(int(config["seed"]) + epoch + 10_000_000)
            # 依据可恢复的批次计划读取数据。
            loader = DataLoader(
                # 传入现场计算条件 mel 的训练数据集。
                dataset,
                # 设置按轮次乱序并能从批次游标继续的采样器。
                batch_sampler=batch_sampler,
                # 设置数据读取子进程数。
                num_workers=int(config["num_workers"]),
                # 使用锁页 CPU 内存，加快复制到 GPU。
                pin_memory=True,
                # 每个数据读取子进程初始化固定随机状态和线程限制。
                worker_init_fn=seed_worker,
                # DataLoader 的 worker 种子使用独立发生器。
                generator=worker_generator,
                # 每轮重新创建 worker，保证轮次种子和恢复位置清楚。
                persistent_workers=False,
            )
            # 逐 batch 读取对齐的输入 mel 与真实波形。
            for input_mel, real_waveform, keys in loader:
                # 若上一个 batch 已经到达上限，立刻跳出当前 epoch。
                if global_step >= max_steps:
                    # 达到预定总步数后结束本轮读取，不再多做一次更新。
                    break

                # pin_memory 配合 non_blocking=True 加快 CPU 到 GPU 复制。
                input_mel = input_mel.to(device, non_blocking=True)
                # 把真实波形从锁页 CPU 内存异步复制到训练 GPU。
                real_waveform = real_waveform.to(device, non_blocking=True)

                # 判别器和卷积期望 [B,1,T]；Dataset 给的是 [B,T]，所以增加声道轴。
                real_waveform_3d = real_waveform.unsqueeze(1)

                # 生成器把 [B,80,32] mel 上采样 8*8*2*2=256 倍，得到 [B,1,8192] 波形。
                generated_waveform = generator(input_mel)

                # 真实波形的 loss mel 使用官方 fmax_for_loss=null，即直到 Nyquist。
                real_mel_for_loss = mel_spectrogram_torch(
                    # 传入与条件 mel 同一裁切位置的真实训练波形。
                    real_waveform,
                    # 设置短时傅里叶变换点数。
                    n_fft=int(config["n_fft"]),
                    # 设置mel 频带数量。
                    num_mels=int(config["num_mels"]),
                    # 设置每秒波形采样点数。
                    sampling_rate=int(config["sampling_rate"]),
                    # 设置相邻 mel 帧之间的波形采样点间隔。
                    hop_size=int(config["hop_size"]),
                    # 设置分析窗的采样点长度。
                    win_size=int(config["win_size"]),
                    # 设置mel 滤波器最低频率。
                    fmin=int(config["fmin"]),
                    # 设置mel 滤波器最高频率，空值表示奈奎斯特频率。
                    fmax=config["fmax_for_loss"],
                )

                # 生成波形也用完全相同定义转成 loss mel，梯度会沿此计算回到生成器。
                generated_mel_for_loss = mel_spectrogram_torch(
                    # 去掉单声道轴后计算生成波形 mel，梯度仍保留。
                    generated_waveform.squeeze(1),
                    # 设置短时傅里叶变换点数。
                    n_fft=int(config["n_fft"]),
                    # 设置mel 频带数量。
                    num_mels=int(config["num_mels"]),
                    # 设置每秒波形采样点数。
                    sampling_rate=int(config["sampling_rate"]),
                    # 设置相邻 mel 帧之间的波形采样点间隔。
                    hop_size=int(config["hop_size"]),
                    # 设置分析窗的采样点长度。
                    win_size=int(config["win_size"]),
                    # 设置mel 滤波器最低频率。
                    fmin=int(config["fmin"]),
                    # 设置mel 滤波器最高频率，空值表示奈奎斯特频率。
                    fmax=config["fmax_for_loss"],
                )

                # -----------------------------------------------------------------
                # 第一步：更新判别器 MPD + MRD
                # -----------------------------------------------------------------

                # 清空上一轮判别器梯度；set_to_none 节省一次显存写零操作。
                optim_d.zero_grad(set_to_none=True)

                # detach 切断生成器图：判别器更新时不应顺便改变生成器。
                mpd_real, mpd_fake, _, _ = mpd(
                    # 参考为当前训练数据裁切得到的真实波形 [B,1,T]。
                    real_waveform_3d,
                    # 判别器更新时切断 G 梯度，只学习判断真假。
                    generated_waveform.detach(),
                )
                # 官方 LSGAN 判别器损失：真分数靠近 1，假分数靠近 0。
                loss_disc_mpd, _, _ = discriminator_loss(mpd_real, mpd_fake)

                # MRD 从三张线性幅度频谱图判断真/假。
                mrd_real, mrd_fake, _, _ = mrd(
                    # 参考为当前训练数据裁切得到的真实波形 [B,1,T]。
                    real_waveform_3d,
                    # 判别器更新时切断 G 梯度，只学习判断真假。
                    generated_waveform.detach(),
                )
                # MRD 三个分支的真假损失按官方方式求和。
                loss_disc_mrd, _, _ = discriminator_loss(mrd_real, mrd_fake)

                # 总判别器损失是两个判别器族损失之和，和官方 MPD+MSD 训练结构一致。
                loss_disc_total = loss_disc_mpd + loss_disc_mrd

                # 反向计算判别器参数梯度并更新一次。
                loss_disc_total.backward()
                # 按当前梯度更新 MPD 与 MRD 一次。
                optim_d.step()

                # -----------------------------------------------------------------
                # 第二步：更新 HiFi-GAN V1 Generator
                # -----------------------------------------------------------------

                # 清空生成器上一轮梯度。
                optim_g.zero_grad(set_to_none=True)

                # mel L1 重建误差乘官方 HiFi-GAN 权重 45。
                loss_mel = F.l1_loss(
                    # 使用真实波形的完整损失频带 mel 作为重建参考。
                    real_mel_for_loss,
                    # 使用生成波形对应 mel，与参考逐点比较。
                    generated_mel_for_loss,
                # 将未加权 mel L1 乘以固定系数 45 后加入 G 目标。
                ) * float(config["mel_loss_weight"])

                # 再经过当前已更新的 MPD，保留真假中间特征用于 feature matching。
                mpd_real_g, mpd_fake_g, mpd_fmap_real, mpd_fmap_fake = mpd(
                    # 参考为当前训练数据裁切得到的真实波形 [B,1,T]。
                    real_waveform_3d,
                    # G 更新时保留生成波形计算图，让判别器信号传回 G。
                    generated_waveform,
                )
                # 再经过当前 MRD；这里绝不能 detach generated_waveform。
                mrd_real_g, mrd_fake_g, mrd_fmap_real, mrd_fmap_fake = mrd(
                    # 参考为当前训练数据裁切得到的真实波形 [B,1,T]。
                    real_waveform_3d,
                    # G 更新时保留生成波形计算图，让判别器信号传回 G。
                    generated_waveform,
                )

                # feature_loss 会比较每层真实/生成特征，并按官方实现整体乘 2。
                loss_fm_mpd = feature_loss(mpd_fmap_real, mpd_fmap_fake)
                # 求 MRD 各层参考/生成特征 L1 均值后相加，官方函数再乘 2。
                loss_fm_mrd = feature_loss(mrd_fmap_real, mrd_fmap_fake)

                # 生成器对抗目标把所有假分数推近 1，让判别器认为生成波形是真实的。
                loss_gen_mpd, _ = generator_loss(mpd_fake_g)
                # 推动三个频谱分支对生成波形的分数接近真标签 1。
                loss_gen_mrd, _ = generator_loss(mrd_fake_g)

                # 生成器总损失保留官方 HiFi-GAN 的五部分结构，只把 MSD 两项换为 MRD 两项。
                loss_gen_total = (
                    # 总目标包含乘 45 后的 mel 重建误差，量级不应直接理解为 0.x。
                    loss_mel
                    # 加入 MPD 各层特征匹配项。
                    + loss_fm_mpd
                    # 加入 MRD 各层特征匹配项。
                    + loss_fm_mrd
                    # 加入 MPD 对抗项。
                    + loss_gen_mpd
                    # 加入 MRD 对抗项；多项求和的总损失没有固定的 0.x 上限。
                    + loss_gen_mrd
                )

                # 反向传播到生成器并更新一次。
                loss_gen_total.backward()
                # 只更新 G 参数；D 的优化器本阶段不执行 step。
                optim_g.step()

                # 本 batch 完成后，总步数加一；这里的 step 表示一次 G/D 成对更新。
                global_step += 1
                # 这批 G/D 成对更新完整结束后，再推进轮内恢复游标。
                batch_in_epoch += 1

                # 按固定步数把 loss 写日志和 CSV；训练曲线以后就按这些真实记录画。
                if global_step % log_every == 0 or global_step == 1:
                    # 读取当前时刻用于估算训练吞吐。
                    now = time.time()
                    # 把上次记录后的总耗时除以实际步数。
                    seconds_per_step = (now - last_log_time) / (
                        # 首条记录仅完成一步，后续记录覆盖固定日志间隔。
                        1 if global_step == 1 else log_every
                    )
                    # 将当前时刻作为下个日志区间的起点。
                    last_log_time = now

                    # mel_error 不含乘 45，数值更容易与官方 HiFi-GAN 日志比较。
                    mel_error = F.l1_loss(
                        # 日志用的参考 mel 脱离计算图。
                        real_mel_for_loss.detach(),
                        # 日志重算未加权 mel L1，不额外建立训练梯度。
                        generated_mel_for_loss.detach(),
                    # 把仅用于日志的标量转为普通数字，不保留张量图。
                    ).item()

                    # 当前峰值显存以 GiB 输出。
                    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

                    # 把所有关键损失转为普通浮点数，避免 CSV 保存 tensor 字符串。
                    row = {
                        # 记录已完成的优化器更新次数。
                        "global_step": global_step,
                        # 记录训练轮次。
                        "epoch": epoch,
                        # 记录加权重建、特征匹配和对抗项之和。
                        "generator_total": float(loss_gen_total.detach().item()),
                        # 记录 MPD 与 MRD 的总真假判断误差。
                        "discriminator_total": float(loss_disc_total.detach().item()),
                        # 记录未乘 45 的 mel 平均绝对误差，便于单独观察重建。
                        "mel_error_unweighted": float(mel_error),
                        # 记录乘 45 后真正进入 G 总目标的 mel 项。
                        "mel_loss_weighted": float(loss_mel.detach().item()),
                        # 记录 MPD 特征匹配贡献。
                        "feature_matching_mpd": float(loss_fm_mpd.detach().item()),
                        # 记录 MRD 特征匹配贡献。
                        "feature_matching_mrd": float(loss_fm_mrd.detach().item()),
                        # 记录 G 试图通过 MPD 判断的对抗项。
                        "generator_adversarial_mpd": float(loss_gen_mpd.detach().item()),
                        # 记录 G 试图通过 MRD 判断的对抗项。
                        "generator_adversarial_mrd": float(loss_gen_mrd.detach().item()),
                        # 记录 MPD 自身的真假训练误差。
                        "discriminator_mpd": float(loss_disc_mpd.detach().item()),
                        # 记录 MRD 自身的真假训练误差。
                        "discriminator_mrd": float(loss_disc_mrd.detach().item()),
                        # 记录生成器当前学习率。
                        "learning_rate_g": float(optim_g.param_groups[0]["lr"]),
                        # 记录当前日志区间每步平均耗时。
                        "seconds_per_step": float(seconds_per_step),
                        # 记录训练开始后分配张量的峰值显存。
                        "peak_allocated_gib": float(peak_gib),
                        # 保留本批第一条语音身份，方便回查异常。
                        "example_key": str(keys[0]),
                    }
                    # 把当前分项损失写入持久 CSV。
                    append_loss_row(loss_csv_path, row)

                    # flush=True 让 nohup 日志立即可见。
                    print(
                        # 日志同时显示已经完成和计划总步数。
                        f"step={global_step:07d}/{max_steps:07d} "
                        # 日志显示当前轮次。
                        f"epoch={epoch:04d} "
                        # 日志显示加权求和后的生成器总损失。
                        f"g={row['generator_total']:.4f} "
                        # 日志显示 MPD+MRD 的总判别误差。
                        f"d={row['discriminator_total']:.4f} "
                        # 日志单列未乘 45 的 mel L1，便于观察重建变化。
                        f"mel={row['mel_error_unweighted']:.4f} "
                        # 日志显示近段每步平均秒数。
                        f"sec/step={seconds_per_step:.3f} "
                        # 日志显示峰值显存，便于判断批大小是否安全。
                        f"peak_GiB={peak_gib:.3f}",
                        # 立即刷新终端缓冲，远程查看日志时可及时看到进度或错误。
                        flush=True,
                    )

                # 定期覆盖完整 latest.pt；archive_generator 只在更大间隔发生。
                if global_step % save_every == 0:
                    # 保存 G、两个 D、优化器、调度器和游标，确保下次可完整恢复。
                    save_training_checkpoint(
                        # 传入声码器检查点保存目录。
                        checkpoint_dir,
                        # 传入模型配置。
                        config,
                        # 传入将 mel 转为波形的生成器 G。
                        generator,
                        # 传入按多个周期检查波形的判别器 MPD。
                        mpd,
                        # 传入按多种频谱分辨率检查波形的判别器 MRD。
                        mrd,
                        # 传入生成器优化器状态。
                        optim_g,
                        # 传入判别器优化器状态。
                        optim_d,
                        # 传入生成器学习率调度器。
                        scheduler_g,
                        # 传入判别器学习率调度器。
                        scheduler_d,
                        # 传入已完成的优化器更新次数。
                        global_step,
                        # 传入训练轮次。
                        epoch,
                        # 传入本轮已经完成的批次数。
                        batch_in_epoch,
                        # 传入训练数据及预处理记录的指纹。
                        data_fingerprints,
                        # 传入训练软件和源码来源记录。
                        runtime_provenance,
                        # 只有到较大归档间隔才另存轻量 G，平时仅更新 latest。
                        archive_generator=(global_step % archive_every == 0),
                    )
                    # 显示已完整保存的最新训练步数。
                    print(f"CHECKPOINT SAVED: step={global_step}", flush=True)

                # 信号处理器只设置标志；当前 G/D 成对更新完整结束后才走保存退出，
                # 避免异步异常留下半更新的 optimizer/checkpoint。
                if safe_stop_requested:
                    # 转入统一断点保存分支，不把停止误报为训练完成。
                    raise SafeStopRequested

            # 只有完整走完 epoch 才按官方规则衰减学习率。若 max_steps 落在
            # epoch 中间，cursor 保留原位置，避免最终 checkpoint 伪装成完整 epoch。
            if batch_in_epoch == batches_per_epoch:
                # 完整轮次结束后衰减 G 学习率。
                scheduler_g.step()
                # 同一轮结束后同步衰减 D 学习率。
                scheduler_d.step()
                # 只有全轮全部批次完成才增加已完成轮数。
                epoch += 1
                # 新轮从第一批开始，续训游标随之清零。
                batch_in_epoch = 0

        # 正常达到 max_steps 时一定保存最终完整状态和生成器归档。
        save_training_checkpoint(
            # 传入声码器检查点保存目录。
            checkpoint_dir,
            # 传入模型配置。
            config,
            # 传入将 mel 转为波形的生成器 G。
            generator,
            # 传入按多个周期检查波形的判别器 MPD。
            mpd,
            # 传入按多种频谱分辨率检查波形的判别器 MRD。
            mrd,
            # 传入生成器优化器状态。
            optim_g,
            # 传入判别器优化器状态。
            optim_d,
            # 传入生成器学习率调度器。
            scheduler_g,
            # 传入判别器学习率调度器。
            scheduler_d,
            # 传入已完成的优化器更新次数。
            global_step,
            # 传入训练轮次。
            epoch,
            # 传入本轮已经完成的批次数。
            batch_in_epoch,
            # 传入训练数据及预处理记录的指纹。
            data_fingerprints,
            # 传入训练软件和源码来源记录。
            runtime_provenance,
            # 达到正式终点时必定保存一份独立轻量生成器归档。
            archive_generator=True,
        )
        # 只有达到计划总步数后才显示训练完成。
        print(f"VOCODER TRAINING COMPLETED AT STEP {global_step}", flush=True)

    # 处理人工中断或安全停机请求，尽力保存完整最新断点。
    except (KeyboardInterrupt, SafeStopRequested) as stop_error:
        # 用户在前台按 Ctrl+C 时先保存再退出；kill -9/服务器断电无法执行任何 Python 清理。
        print("KeyboardInterrupt received; saving resumable latest.pt...", flush=True)
        # 保存 G、两个 D、优化器、调度器和游标，确保下次可完整恢复。
        save_training_checkpoint(
            # 传入声码器检查点保存目录。
            checkpoint_dir,
            # 传入模型配置。
            config,
            # 传入将 mel 转为波形的生成器 G。
            generator,
            # 传入按多个周期检查波形的判别器 MPD。
            mpd,
            # 传入按多种频谱分辨率检查波形的判别器 MRD。
            mrd,
            # 传入生成器优化器状态。
            optim_g,
            # 传入判别器优化器状态。
            optim_d,
            # 传入生成器学习率调度器。
            scheduler_g,
            # 传入判别器学习率调度器。
            scheduler_d,
            # 传入已完成的优化器更新次数。
            global_step,
            # 传入训练轮次。
            epoch,
            # 传入本轮已经完成的批次数。
            batch_in_epoch,
            # 传入训练数据及预处理记录的指纹。
            data_fingerprints,
            # 传入训练软件和源码来源记录。
            runtime_provenance,
            # 中断保存完整 latest，不把中途权重自动标为正式长期终点。
            archive_generator=False,
        )
        # 显示已保存断点的中途安全停机进度，不宣称训练完成。
        print(f"INTERRUPTED SAFELY AT STEP {global_step}", flush=True)
        # 普通终止信号用明确非零退出码告诉外部进程当前是安全停止。
        if isinstance(stop_error, SafeStopRequested):
            # 用终止状态 143 退出，防止自动化把中途停机认为完成。
            raise SystemExit(143) from None
        # 保存后继续传播人工中断，保留停止的真实原因。
        raise


# 用 ``python3 -m hifigan.custom.train`` 执行时进入 main；被别的文件 import 时不会开跑。
if __name__ == "__main__":
    # 启动本脚本的完整处理流程。
    main()
