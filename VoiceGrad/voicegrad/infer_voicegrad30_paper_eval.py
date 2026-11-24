#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 本文件说明正式教师评估流程；当前项目实际 speaker embedding 来自 256 维 GE2E，ECAPA 只用于评价音色相似度。
# 下方旧说明中保留的“ECAPA/GE2E”是历史泛称；本次送入教师的条件向量仅由 GE2E 提取。
"""
为 FastVoiceGrad 论文的 8,100 对客观评估批量生成指定 VoiceGrad-30 epoch 的波形。

这一个文件把“推理前必须固定的协议”全部集中起来，避免手工循环时漏掉某一步：

1. 读取所选协议中 10 位 unseen test speakers 的 100 条语音；
2. 每条 source 与另外 9 位说话人配对；
3. 每个目标说话人排除和 source 相同“测试句位”的那一句，再使用剩余 9 句；
4. 因此总数严格是 100 × 9 × 9 = 8,100；
5. 使用命令行指定并通过文件名/内部元数据一致性检查的 epoch、30 次反向扩散、S_K=950；
6. 按 FastVoiceGrad 第 3.1 节公式 (9)，从扩散后的 source x_src_SK 开始；
7. 用 source BNF 提供内容、每条 target reference 的 ECAPA/GE2E embedding 提供音色；
8. 最后用命令指定的“自训练 modified HiFi-GAN”或历史公开 VCTK_V1 声码器，
   把 mel 转成 22.05 kHz WAV。

重要边界：论文没有公开作者的 8,100-pair CSV、随机种子、完整 30 个整数 timestep，
也没有公开 modified HiFi-GAN 权重。本文件把所有这些未公开选择写进 CSV/JSON，
做到可复现，但不会把公开 VCTK_V1 声码器冒充论文作者的声码器。
"""

# 延迟解释类型提示，保持原有运行环境兼容。
from __future__ import annotations

# argparse 读取命令行；csv/json 写固定配对清单和运行报告。
import argparse
# 按表头读取测试清单和固定配对表。
import csv
# 计算模型、数据和源码的原始字节指纹，防止续算混用版本。
import hashlib
# 在内存中先生成确定的 CSV 字节，再决定是否发布到结果目录。
import io
# 读取、保存推理身份与完成报告。
import json
# 使用原子替换写报告，防止中断时留下半个正式文件。
import os
# 严格解析 epoch_NNNN.pt 文件名，并核对内部训练轮数。
import re
# 记录实际推理耗时。
import time

# defaultdict 按 source 组织 81 个目标配对，保证一个 batch 内 source 长度相同。
from collections import defaultdict

# dataclass 把一条配对记录变成字段明确的只读对象。
from dataclasses import asdict, dataclass

# Path 负责所有项目、检查点和输出路径。
from pathlib import Path

# 类型提示不改变运行，只帮助说明容器里放的对象。
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# NumPy 读取 mel/BNF/speaker embedding，并接收 GPU 推理结果。
import numpy as np

# SoundFile 用于快速检查已经存在的 WAV 是否完整。
import soundfile as sf

# PyTorch 加载 VoiceGrad 和 HiFi-GAN，并在选定 GPU 上批量推理。
import torch

# tqdm 显示 8,100 对任务进度。
from tqdm import tqdm

# 项目统一配置、扩散过程和教师模型入口。
from .config import PAPER_CONFIG, SERVER_PROJECT_ROOT
# 复用教师训练时的扩散定义，不能另造一套采样系数。
from .diffusion import VoiceGradDiffusion
# 外部 BNF 时间对齐的模式和统一实现。
from .feature_alignment import ALIGNMENT_MODES, LEGACY_ALIGNMENT, align_bnf_to_mel
# U-Net 内部 BNF 降采样的模式，与外部长度插值分开记录。
from .model import BNF_DOWNSAMPLING_MODES, LEGACY_BNF_DOWNSAMPLING
# 所有推理只允许在当前登记的租用服务器执行。
from .runtime_guard import assert_rented_server
# 按教师训练的同一结构构建网络，避免加载权重时接口漂移。
from .train import build_model

# 复用已经审计过的单条声码器工具：加载官方 Generator 和原子写 PCM16 WAV。
from hifigan.tools.vocode import atomic_write_wav, load_generator
# 自训声码器先严格验证内部步数和参数结构，再进行推理。
from hifigan.custom.validate_checkpoint import validate_checkpoint


# 两种协议的 test 都是同一组 10 人 × 001--010；区别只在 train 是否保留这些文本。
PROTOCOL_CHOICES = ("text-unseen", "text-seen")

# 声码器默认仍指向工程内公开 VCTK_V1 权重；命令行可改成我们自训练的权重。
DEFAULT_VOCODER_CHECKPOINT = (
    # 历史默认指向公开 VCTK V1 权重；正式 wrapper 会显式传入自训权重。
    SERVER_PROJECT_ROOT / "pretrained_hifigan_22k" / "VCTK_V1" / "generator_v1"
)
# 默认配置来自该公开权重同目录，命令行可明确指定其他配套配置。
DEFAULT_VOCODER_CONFIG = DEFAULT_VOCODER_CHECKPOINT.parent / "config.json"


# 将配对字段定义为不可修改的数据类，降低推理途中误改来源的风险。
@dataclass(frozen=True)
# 每个实例表示一条“源内容转成目标音色”的语音转换。
class EvaluationPair:
    # 固定配对编号供 WAV、指标 CSV 和最终汇总互相对应。
    """一行固定评估配对；pair_id 是 0 到 8099 的永久编号。"""

    # 该转换在完整清单中的唯一编号，采用 0 到 8099。
    pair_id: int
    # 源语音的说话人身份。
    source_speaker: str
    # 源句编号，用它查找需要保留的 mel/BNF 内容。
    source_utt: str
    # 源句在 001–010 公共测试网格中的位置。
    source_sentence_slot: int
    # 源语音真实文字，CER 要对照这一内容。
    source_text: str
    # 源语音的 22.05 kHz 波形路径。
    source_wav: str
    # 提供目标音色的参考说话人身份。
    reference_speaker: str
    # 目标参考语音编号，用于提取其音色嵌入。
    reference_utt: str
    # 参考句在相同测试句网格中的位置。
    reference_sentence_slot: int
    # 保留参考文本用于追踪，转换后应说的内容仍由源句决定。
    reference_text: str
    # 真实参考 WAV 路径，后续说话人验证以它为参照。
    reference_wav: str
    # 这条配对应写出的转换 WAV 绝对路径。
    output_wav: str


# 分块计算文件指纹，大模型无需整份放入内存。
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # 哈希只读文件，不解析或更改任何模型参数。
    """分块计算文件 SHA-256，不把 295 MB checkpoint 一次读进内存。"""

    # 初始化累计 SHA-256 状态。
    digest = hashlib.sha256()
    # 按二进制读取，避免编码和换行转换影响指纹。
    with path.open("rb") as handle:
        # 持续读取直到文件结束。
        while True:
            # 每次按 chunk_size 读取一个数据块，默认 1 MiB。
            block = handle.read(chunk_size)
            # 读到空字节串表示文件内容已经全部处理。
            if not block:
                # 结束读取循环，进入最终摘要返回。
                break
            # 把当前原始字节块加入累计摘要。
            digest.update(block)
    # 返回固定长度十六进制文件指纹。
    return digest.hexdigest()


# 完整报告先写到临时路径，再原子发布，读者不会看见半个 JSON。
def atomic_write_json(path: Path, value: Dict[str, object]) -> None:
    # 这项辅助函数只改变报告文件，不涉及模型训练。
    """Publish a run identity/report without leaving a half-written JSON file."""

    # 建立报告父目录，已存在目录保留。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 在同目录生成临时文件名，保证可以原子替换。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 先完整写临时 JSON，再更新正式文件。
    temporary.write_text(
        # 保留中文与缩进，并在文件尾写换行，方便核验准确字节。
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 一次性发布完整报告，避免外部读取部分写入的数据。
    os.replace(temporary, path)


# 读取来源 JSON 并要求其顶层为对象，格式不正确立即失败。
def read_json_object(path: Path) -> Dict[str, object]:
    # 来源记录缺失或结构异常时不能猜测继续。
    """Read a lineage JSON file and reject non-object payloads."""

    # 将 UTF-8 JSON 文件解析为 Python 对象。
    value = json.loads(path.read_text(encoding="utf-8"))
    # 来源契约必须为字段字典，不能接受列表或其他类型。
    if not isinstance(value, dict):
        # 报出异常类型及路径，便于找到损坏记录。
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    # 返回通过结构检查的完整记录。
    return value


# 从语音编号解析固定句位，不用排序后临时重新编号。
def utterance_number(utt_id: str) -> int:
    # 例如 p225_001 的最后部分 001 对应句位 1。
    """把 p225_001 的最后三位变成整数 1，只用于记录原始 VCTK 句号。"""

    # 对格式可能不符合约定的编号进行有提示的解析。
    try:
        # 从最后一个下划线右侧取得句号并转成整数。
        return int(utt_id.rsplit("_", 1)[1])
    # 编号缺少下划线或末尾不是整数都会进入错误处理。
    except (IndexError, ValueError) as exc:
        # 指出无法解析的实际语音编号，并保留原始异常原因。
        raise ValueError(f"Cannot parse sentence slot from utterance ID: {utt_id}") from exc


# 确认正式测试清单具有 10 位说话人、每位同样的 10 句。
def load_test_rows(manifest: Path) -> List[Dict[str, str]]:
    # 此函数核对测试网格；文本是否与训练隔离由独立协议审计负责。
    """读取测试清单并核对 10 人都具有完全相同的 001--010。

    注意：这里只检查 test 网格本身。测试文本是否在 train 出现由
    ``validate_setup.py`` 根据 ``split_protocol.json`` 判断。
    """

    # 清单必须真实存在，不能自动换用其他 split。
    if not manifest.is_file():
        # 缺失时报出完整测试清单路径。
        raise FileNotFoundError(f"Test manifest not found: {manifest}")

    # 按 CSV 规则和 UTF-8 编码读取测试记录。
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        # 把每行转换为具名字段字典。
        rows = list(csv.DictReader(handle))

    # 完整测试库存必须正好 100 句。
    if len(rows) != 100:
        # 数量不符立即失败，不把少量语音扩成貌似完整测试。
        raise ValueError(f"Paper evaluation requires 100 test utterances, got {len(rows)}")

    # 累计各测试说话人的句数，后续要求每位恰好 10 条。
    counts: Dict[str, int] = defaultdict(int)
    # 逐条核对划分标志与说话人是否属于 unseen 测试组。
    for row in rows:
        # 清理标签格式后要求 split 为 test。
        if row.get("split", "").strip().lower() != "test":
            # 遇到非测试记录立即指出，避免混入训练样本。
            raise ValueError(f"Non-test row found in {manifest}: {row}")
        # 要求说话人未在模型训练中出现的标志为真。
        if row.get("is_unseen_speaker", "").strip() not in {"1", "True", "true"}:
            # 缺少正确 unseen 标记就拒绝继续。
            raise ValueError(f"Test row is not marked unseen: {row}")
        # 将该句计入对应说话人的数量。
        counts[row["speaker"]] += 1

    # 不仅检查总数，还要求恰好 10 人且每人 10 句。
    if len(counts) != 10 or set(counts.values()) != {10}:
        # 报出实际每人句数，便于发现缺句或重复人。
        raise ValueError(f"Expected 10 speakers × 10 sentences, got {dict(counts)}")

    # 固定公共测试句编号集合为 001 至 010。
    expected_sentence_ids = {f"{number:03d}" for number in range(1, 11)}
    # 按说话人整理记录，以核对每人完整句网格。
    rows_by_speaker: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    # 按公共句号收集文本，核对不同人读的同号句确为同内容。
    texts_by_sentence_id: Dict[str, set] = defaultdict(set)
    # 遍历测试记录建立说话人和文本两种索引。
    for row in rows:
        # 将语音放入它所属说话人的列表。
        rows_by_speaker[row["speaker"]].append(row)
        # 从语音编号末尾读取三位句号。
        sentence_id = row["utt_id"].rsplit("_", 1)[-1]
        # 句号必须处于预登记的 001–010 范围。
        if sentence_id not in expected_sentence_ids:
            # 发现网格外语音立即拒绝。
            raise ValueError(
                # 在错误中列出不应进入测试的具体句子。
                f"Test row is outside the fixed 001--010 grid: {row['utt_id']}"
            )
        # 取得清单中的真实转写并去掉首尾空白。
        text = row.get("text", "").strip()
        # 真实转写为空则无法保证不同句规则与 CER 参考。
        if not text:
            # 报告缺少文本的语音编号。
            raise ValueError(f"Missing test transcript: {row['utt_id']}")
        # 记录这一句号对应的文本，后面核对跨说话人一致性。
        texts_by_sentence_id[sentence_id].add(text)

    # 逐人检查实际句号集合，避免总数正确却有缺句或重号。
    for speaker, speaker_rows in rows_by_speaker.items():
        # 建立该说话人实际拥有的句号集合。
        actual_sentence_ids = {
            # 从每条语音编号末尾取出句号；集合会合并重复编号。
            row["utt_id"].rsplit("_", 1)[-1] for row in speaker_rows
        }
        # 实际句号集合必须与 001–010 完全相同。
        if actual_sentence_ids != expected_sentence_ids:
            # 任何缺失或多余句位都停止正式配对。
            raise ValueError(
                # 错误先指出哪位说话人的测试网格不完整。
                f"{speaker} does not provide the exact 001--010 test grid: "
                # 再列出现场实际句号，方便定位缺项。
                f"{sorted(actual_sentence_ids)}"
            )

    # 找出同一句号在不同说话人中出现多种文本的情况。
    inconsistent_text = {
        # 保留异常句号以及对应的不同文本版本。
        sentence_id: sorted(texts)
        # 逐句号查看累计的文本集合。
        for sentence_id, texts in texts_by_sentence_id.items()
        # 正确情况下每个句号应只有一种真实转写。
        if len(texts) != 1
    }
    # 出现文本不一致就不能保证排除同句位等价于排除相同句子。
    if inconsistent_text:
        # 拒绝继续构建貌似符合规则的配对。
        raise ValueError(
            # 错误说明同号句在不同说话人间的文本发生冲突。
            "The same test sentence ID has different transcripts across speakers: "
            # 输出冲突句号与文本，便于修正清单来源。
            f"{inconsistent_text}"
        )

    # 测试网格和文本核对完成后返回原始顺序记录。
    return rows


# 按测试记录固定顺序构建源/目标组合，排除同说话人及同句位。
def build_pairs(rows: Sequence[Dict[str, str]], output_dir: Path) -> List[EvaluationPair]:
    # 配对数应为 100 个源句乘 9 位其他人乘 9 条不同句。
    """按 source顺序 × target顺序建立恰好 8,100 对。"""

    # 初始化完整配对列表，后续编号按追加顺序增长。
    pairs: List[EvaluationPair] = []
    # 每条转换波形统一放在该实验的 wav 子目录。
    wav_dir = output_dir / "wav"

    # 论文的 8,100 = 100×9×9：每位测试说话人都有完全相同的 001--010，
    # source 的句位 i 不能和目标说话人的同一句位 i 配对。
    # 把测试记录按说话人分组，供句位精确核对。
    rows_by_speaker: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    # 逐条访问测试记录。
    for row in rows:
        # 按真实说话人编号收集语音。
        rows_by_speaker[row["speaker"]].append(row)

    # 用“说话人+语音编号”作为键记录固定公共句位。
    test_slot: Dict[Tuple[str, str], int] = {}
    # 正确句位为整数 1 至 10，不使用任意排序后的代号。
    canonical_slots = set(range(1, 11))
    # 逐位说话人建立其句位映射。
    for speaker, speaker_rows in rows_by_speaker.items():
        # 先从每条真实语音编号解析句位。
        original_numbers = {
            # 把原 utt_id 与解析出的整数句号一一对应。
            row["utt_id"]: utterance_number(row["utt_id"]) for row in speaker_rows
        }
        # 取出该说话人的全部句位，检查重复和缺失。
        assigned_numbers = list(original_numbers.values())
        # 要求十个唯一句位且准确覆盖 1–10。
        if len(set(assigned_numbers)) != 10 or set(assigned_numbers) != canonical_slots:
            # 句位网格错误时停止，不自动补缺或重编号。
            raise ValueError(
                # 错误指出出现问题的说话人。
                f"Strict test grid for {speaker} must be exactly 001--010, got "
                # 同时列出实际句位序列，便于确认编号问题。
                f"{sorted(assigned_numbers)}"
            )

        # 把该说话人每条语音的句位写入全局查找表。
        for row in speaker_rows:
            # 使用前面已经核对的原始句号。
            number = original_numbers[row["utt_id"]]
            # 双字段键避免不同说话人的同号语音互相覆盖。
            test_slot[(speaker, row["utt_id"])] = number

        # 再次从已建立映射读取本人的句位集合。
        assigned = {
            # 检查映射过程确实保存了每条记录的正确句位。
            test_slot[(speaker, row["utt_id"])] for row in speaker_rows
        }
        # 要求最终映射仍准确覆盖 1–10。
        if assigned != canonical_slots:
            # 如映射与原始检查不一致，直接报告内部错误。
            raise RuntimeError(f"Invalid test sentence-slot assignment for {speaker}: {assigned}")

    # 每条测试语音依次作为需要转换的源句。
    for source in rows:
        # 取得源语音的固定公共句位。
        source_slot = test_slot[(source["speaker"], source["utt_id"])]
        # 遍历全部测试语音作为可能的目标音色参考。
        for reference in rows:
            # 取得该参考语音的公共句位。
            reference_slot = test_slot[(reference["speaker"], reference["utt_id"])]

            # 论文要求 different speaker：同一说话人的 10 句全部排除。
            if source["speaker"] == reference["speaker"]:
                # 跳过源说话人自己的全部语音，只保留跨说话人转换。
                continue

            # 论文要求 different sentence：排除相同测试句位，例如第 1 句对第 1 句。
            if source_slot == reference_slot:
                # 跳过与源语音读同一个公共句子的参考。
                continue

            # 剩余合法配对依次编号，编号与列表位置一致。
            pair_id = len(pairs)
            # 文件名同时保留源句和参考句身份。
            filename = (
                # 前半记录配对号与源语音编号。
                f"pair_{pair_id:04d}__src_{source['utt_id']}"
                # 后半记录目标参考编号，便于直接追溯 WAV。
                f"__ref_{reference['utt_id']}.wav"
            )
            # 将合法组合加入完整配对列表。
            pairs.append(
                # 构造具有固定字段的数据类记录。
                EvaluationPair(
                    # 记录唯一配对编号。
                    pair_id=pair_id,
                    # 记录源说话人。
                    source_speaker=source["speaker"],
                    # 记录源语音编号。
                    source_utt=source["utt_id"],
                    # 记录源语音固定句位。
                    source_sentence_slot=source_slot,
                    # 保存源文本作为转换内容的客观参照。
                    source_text=source["text"],
                    # 保存源语音路径供来源核对。
                    source_wav=source["wav22k_path"],
                    # 记录目标参考说话人。
                    reference_speaker=reference["speaker"],
                    # 记录提供音色的目标参考语音。
                    reference_utt=reference["utt_id"],
                    # 保存参考句位，便于复核 different sentence 规则。
                    reference_sentence_slot=reference_slot,
                    # 保存参考文本，避免仅靠文件名推断语义。
                    reference_text=reference["text"],
                    # 保存真实目标参考波形路径。
                    reference_wav=reference["wav22k_path"],
                    # 生成这条配对应写的绝对 WAV 路径。
                    output_wav=str(wav_dir / filename),
                )
            )

    # 所有排除规则执行后必须恰好得到 8100 对。
    if len(pairs) != 8100:
        # 数量不符立即失败，不以预期常数覆盖真实统计。
        raise RuntimeError(f"Pair construction must produce 8,100 rows, got {len(pairs)}")

    # 独立统计每条源语音对应多少合法目标参考。
    per_source: Dict[str, int] = defaultdict(int)
    # 遍历实际构建出的全部配对。
    for pair in pairs:
        # 将当前配对计入该源句数量。
        per_source[pair.source_utt] += 1
    # 每条源句必须恰好对应 9×9=81 个目标参考。
    if set(per_source.values()) != {81}:
        # 任一源句数量不符都报出实际分布。
        raise RuntimeError(f"Every source must have 81 pairs, got {dict(per_source)}")

    # 返回已经通过总数和每源句数量核对的完整清单。
    return pairs


# 先在内存渲染准确 CSV 字节，用于比较历史产物身份。
def render_pairs_csv(pairs: Sequence[EvaluationPair]) -> bytes:
    # 身份校验通过前不能覆盖旧配对表，否则会改变旧 WAV 的解释。
    """Render the canonical pair manifest before publishing any lineage files."""

    # 空配对清单无法生成有效表头与测试身份。
    if not pairs:
        # 空输入直接报错。
        raise ValueError("Cannot render an empty pair manifest")
    # 使用内存文本缓冲构建 CSV，不先改动磁盘文件。
    buffer = io.StringIO(newline="")
    # 按数据类字段顺序确定稳定列顺序。
    fieldnames = list(asdict(pairs[0]).keys())
    # 创建使用固定字段名的 CSV 写入器。
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    # 先写一行表头，后续 8100 行才是实际结果。
    writer.writeheader()
    # 按已经固定的配对顺序逐条写入。
    for pair in pairs:
        # 数据类转成字段字典后写一行。
        writer.writerow(asdict(pair))
    # 转成 UTF-8 字节，供准确哈希和原子文件发布。
    return buffer.getvalue().encode("utf-8")


# 把经过身份校验的完整配对清单原子写到磁盘。
def write_pairs_csv(pairs: Sequence[EvaluationPair], path: Path) -> None:
    # 每个 WAV 可用此表追溯源语音和目标参考。
    """原子写 pairs.csv；每个输出 WAV 都能追溯 source/reference。"""

    # 创建清单父目录，不删除已有实验。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 在目标旁使用临时文件，避免暴露半份清单。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 按上面同一渲染规则写准确 CSV 字节。
    temporary.write_bytes(render_pairs_csv(pairs))
    # 完整写出后才发布正式 pairs.csv。
    os.replace(temporary, path)


# 读取已有 mel 或 BNF，统一为特征维在前、时间在后的连续数组。
def feature_matrix(path: Path, dimension: int) -> np.ndarray:
    # 仅整理方向和 float32 类型，不在这里进行标准化。
    """读取二维特征并统一成 [feature_dim,time] float32。"""

    # 禁用 pickle，只读取普通数值特征数组。
    value = np.load(path, allow_pickle=False)
    # 只接受二维矩阵，额外批维或向量都不符合存储约定。
    if value.ndim != 2:
        # 形状异常立即指出文件，不能强行 reshape 掩盖。
        raise ValueError(f"Expected matrix at {path}, got {value.shape}")
    # 第一维已经等于所需特征维时保持方向。
    if value.shape[0] == dimension:
        # 转为连续 float32 数组，方便后续转 Tensor。
        return np.ascontiguousarray(value, dtype=np.float32)
    # 若第二维是特征维，说明磁盘使用时间在前的布局。
    if value.shape[1] == dimension:
        # 转置为 [特征,时间] 并整理连续内存。
        return np.ascontiguousarray(value.T, dtype=np.float32)
    # 两个轴都不符合要求就报错，不裁剪或补齐错误特征。
    raise ValueError(f"Expected dimension {dimension} at {path}, got {value.shape}")


# 统一按特征种类、说话人和语音编号定位 .npy。
def feature_path(data_dir: Path, feature: str, speaker: str, utt_id: str) -> Path:
    # mel、bnf、spk_emb 使用相同目录组织规则。
    """拼 data/mel|bnf|spk_emb/<speaker>/<utt>.npy。"""

    # 拼出既有特征路径，不在推理时重新提取。
    return data_dir / feature / speaker / f"{utt_id}.npy"


# 续算前检查已有 WAV 的采样率、长度和单通道格式。
def valid_existing_wav(path: Path, expected_sr: int = 22050) -> bool:
    # 此处主要检查文件头；不能据此证明主观音质正常。
    """续跑时只跳过采样率正确、至少含一个采样点的完整 WAV。"""

    # 路径必须为实际普通文件。
    if not path.is_file():
        # 缺失或不是文件时返回不可复用。
        return False
    # 尝试读取音频信息，损坏文件交给异常处理。
    try:
        # 读取 SoundFile 能解析的文件头元数据。
        info = sf.info(str(path))
        # 要求 22.05 kHz、至少一个采样点、且为单通道。
        return info.samplerate == expected_sr and info.frames > 0 and info.channels == 1
    # 文件无法解析时视为不完整，需要重新生成。
    except RuntimeError:
        # 返回不可复用，不悄悄算入已完成数。
        return False


# 按同一源句的配对切批，避免不同源句长度混合导致额外填充。
def fixed_chunks(pairs: Sequence[EvaluationPair], batch_size: int) -> Iterable[List[EvaluationPair]]:
    # 每个源句最多 81 条目标参考，按 batch_size 分成若干固定小批。
    """只在同一个 source 的 81 对内部切 batch，绝不混合不同长度 source。"""

    # 按源语音编号收集各目标参考配对。
    grouped: Dict[str, List[EvaluationPair]] = defaultdict(list)
    # 另外保存源句首次出现的顺序，保证续跑批次划分一致。
    source_order: List[str] = []
    # 遍历按固定规则生成的配对。
    for pair in pairs:
        # 源句第一次出现时登记其位置。
        if pair.source_utt not in grouped:
            # 保存源句顺序，不能改成每次随机排序。
            source_order.append(pair.source_utt)
        # 把当前配对归入对应源句。
        grouped[pair.source_utt].append(pair)

    # 严格按首次登记顺序逐源句处理。
    for source_utt in source_order:
        # 取得该源句的全部 81 个合法目标参考。
        source_pairs = grouped[source_utt]
        # 按固定 batch_size 建立不重叠起点。
        for start in range(0, len(source_pairs), batch_size):
            # 每次产生这一小批配对，尾批可少于 batch_size。
            yield source_pairs[start : start + batch_size]


# 加载可信教师并核对文件名、训练轮数、扩散配置与内部 BNF 接口。
def load_voicegrad(
    # checkpoint_path 指本次实际教师文件。
    checkpoint_path: Path,
    # device 指租用服务器上的计算设备。
    device: torch.device,
    # 默认保留教师原有内部 BNF 抽点方式。
    bnf_downsampling: str = LEGACY_BNF_DOWNSAMPLING,
    # 只有明确诊断授权才允许覆盖内部接口，正式推理应保持 False。
    allow_bnf_downsampling_override: bool = False,
# 返回冻结模型和原检查点元数据，后续报告据实记录。
) -> Tuple[torch.nn.Module, dict]:
    # 要求文件名与内部 epoch 一致，不能重命名旧权重伪装其他轮数。
    """严格加载 epoch_NNNN.pt，并核对文件名与检查点内部 epoch 一致。"""

    # 加载本项目可信检查点及元数据；weights_only=False 用于兼容原保存格式。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 严格要求文件名遵循 epoch_数字.pt。
    match = re.fullmatch(r"epoch_(\d+)\.pt", checkpoint_path.name)
    # 格式不符时无法明确追踪教师轮数。
    if match is None:
        # 拒绝不符合约定的教师文件名。
        raise ValueError(f"Teacher checkpoint must be named epoch_NNNN.pt: {checkpoint_path}")
    # 从实际文件名取得声明的 epoch。
    expected_epoch = int(match.group(1))
    # 从检查点内部取得实际训练 epoch，缺失用 -1 使校验失败。
    actual_epoch = int(checkpoint.get("epoch", -1))
    # 文件名轮数必须等于内部记录。
    if actual_epoch != expected_epoch:
        # 不一致时停止，避免混用不正确教师。
        raise ValueError(
            # 错误先说明文件名和内部元数据发生冲突。
            f"Teacher checkpoint filename/metadata mismatch: "
            # 同时列出两边的实际 epoch。
            f"filename epoch={expected_epoch}, metadata epoch={actual_epoch}"
        )
    # 读取训练时保存的论文配置。
    saved_config = checkpoint.get("paper_config", {})
    # 本实现要求教师使用正确的 1000 步扩散时间表。
    if saved_config.get("diffusion_steps") != 1000:
        # 扩散时间表不符就拒绝加载。
        raise ValueError("Checkpoint is not the 1000-step corrected teacher")

    # 读取教师训练时的 U-Net 内部 BNF 抽点设定。
    saved_downsampling = checkpoint.get("run_config", {}).get(
        # 旧检查点未登记时使用明确历史默认，不能默认为新物理中心模式。
        "bnf_downsampling", LEGACY_BNF_DOWNSAMPLING
    )
    # 如果本次接口不同且没有诊断覆盖授权，就停止。
    if saved_downsampling != bnf_downsampling and not allow_bnf_downsampling_override:
        # 内部接口不符时不能当作正式同条件推理。
        raise ValueError(
            # 错误先指出这是教师内部 BNF 降采样设置差异。
            "Checkpoint BNF U-Net downsampling mismatch: checkpoint uses "
            # 列出保存值与请求值，便于决定是否应另建诊断。
            f"{saved_downsampling!r}, requested {bnf_downsampling!r}."
        )

    # 用请求且已允许的内部接口创建同结构教师网络。
    model = build_model(bnf_downsampling).to(device)
    # strict=True 要求全部参数键和形状完全对应。
    model.load_state_dict(checkpoint["model"], strict=True)
    # 切换推理模式，不更新训练状态。
    model.eval()
    # 遍历模型参数，明确冻结教师。
    for parameter in model.parameters():
        # 关闭每个参数的梯度，推理不修改教师。
        parameter.requires_grad_(False)
    # 返回冻结教师与原元数据。
    return model, checkpoint


# 解析正式推理及明确标注的少量诊断选项。
def parse_args() -> argparse.Namespace:
    # 具体默认值都在下面逐项登记。
    """读取批量推理命令参数。"""

    # 构造命令行接口，帮助文字采用文件用途说明。
    parser = argparse.ArgumentParser(description=__doc__)
    # 默认从登记的租用项目目录读取数据与权重。
    parser.add_argument("--project-root", type=Path, default=SERVER_PROJECT_ROOT)
    # 选择活动数据协议，也可自动读取现场规则。
    parser.add_argument(
        # --protocol 控制 seen/unseen 标签与现场清单的一致性检查。
        "--protocol",
        # 允许 auto 或明确的两种实验协议，不接受任意字符串。
        choices=("auto",) + PROTOCOL_CHOICES,
        # 默认 auto 读取实际 split_protocol.json。
        default="auto",
        # 帮助说明显式协议也必须与现场规则一致。
        help="默认 auto：读取 split_protocol.json；也可显式指定并做一致性核对。",
    )
    # 选择要加载的教师 epoch，默认 500。
    parser.add_argument(
        # --epoch 同时用于默认权重文件名和报告标签。
        "--epoch",
        # 按整数解析训练轮数。
        type=int,
        # 默认与既有教师 500 epoch 对照一致。
        default=500,
        # 说明显式 checkpoint 可覆盖文件路径，但仍核对内部 epoch。
        help="未显式传 --checkpoint 时，用它拼出 epoch_NNNN.pt；默认 500。",
    )
    # 可显式指定教师文件，默认按协议与 epoch 定位。
    parser.add_argument("--checkpoint", type=Path, default=None)
    # 声码器权重参数默认保留历史公开 VCTK V1，正式 wrapper 显式传自训版本。
    parser.add_argument("--vocoder-checkpoint", type=Path, default=DEFAULT_VOCODER_CHECKPOINT)
    # 声码器配置必须与实际生成器配套。
    parser.add_argument("--vocoder-config", type=Path, default=DEFAULT_VOCODER_CONFIG)
    # 声码器来源标签必须与实际模型对应。
    parser.add_argument(
        # --vocoder-kind 指定自训、公开或根据路径推断。
        "--vocoder-kind",
        # 只接受 auto/custom/public 三个模式。
        choices=("auto", "custom", "public"),
        # 默认保留旧命令的路径推断兼容。
        default="auto",
        # 帮助说明推断标签会写入报告，不能只看 WAV 判断来源。
        help="写入审计报告的声码器身份；auto 根据检查点路径判断。",
    )
    # 可显式选择独立输出目录，默认按协议和教师 epoch 命名。
    parser.add_argument("--output-dir", type=Path, default=None)
    # 默认每批 4 个转换配对，影响显存和固定批次种子规则。
    parser.add_argument("--batch-size", type=int, default=4)
    # 记录基础随机种子，默认取统一教师配置。
    parser.add_argument("--seed", type=int, default=PAPER_CONFIG.seed)
    # 在租用服务器内选择 CUDA 或 CPU；项目规则仍禁止本地运行。
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # limit 只用于少量冒烟诊断，完整 pairs.csv 仍保留 8100 对；正式评估不得混用。
    parser.add_argument("--limit", type=int, default=None, help="Smoke test only; pairs.csv remains 8100 rows")
    # 只有明确请求 overwrite 才重生成有效已有 WAV。
    parser.add_argument("--overwrite", action="store_true")
    # 外部 BNF 时间对齐作为独立显式选项。
    parser.add_argument(
        # --bnf-alignment 选择长度插值或物理帧中心对齐。
        "--bnf-alignment",
        # 只接受登记的对齐模式。
        choices=ALIGNMENT_MODES,
        # 默认保留既有教师采用的历史长度插值。
        default=LEGACY_ALIGNMENT,
        # 帮助文字分别解释两个模式的时间轴含义。
        help=(
            # 历史长度线性插值保证与已有检查点接口一致。
            "legacy_length_linear exactly reproduces existing checkpoints; "
            # 物理时间模式按 10 毫秒 BNF 中心对齐到 mel 帧中心。
            "time_linear aligns 10 ms BNF centres to mel-frame centre times."
        ),
    )
    # 单独设置 U-Net 内部 BNF 抽点，不能把它与外部插值混为一项。
    parser.add_argument(
        # --bnf-downsampling 控制各网络尺度的 BNF 时间采样。
        "--bnf-downsampling",
        # 只允许源码已实现的内部模式。
        choices=BNF_DOWNSAMPLING_MODES,
        # 默认使用教师历史抽点规则。
        default=LEGACY_BNF_DOWNSAMPLING,
        # 帮助说明内部 T、T/2、T/4 三个时间尺度的接口。
        help="BNF alignment inside the T/T2/T4 U-Net levels.",
    )
    # 内部接口覆盖必须显式标明仅用于诊断。
    parser.add_argument(
        # 此开关允许在旧权重上试另一套内部抽点方法。
        "--diagnostic-allow-bnf-downsampling-override",
        # 该参数出现时才为真，默认保持关闭。
        action="store_true",
        # 帮助文字清楚说明覆盖旧检查点接口的限制。
        help=(
            # 允许受控诊断比较旧权重在另一采样规则下的行为。
            "Diagnostic only: apply a different U-Net BNF sampling rule to an old "
            # 正式推理必须保持这个覆盖选项关闭。
            "checkpoint. Formal inference must leave this disabled."
        ),
    )
    # 返回实际解析的路径、设置和诊断标记。
    return parser.parse_args()


# 从现场划分记录确定真实协议，阻止标签与数据不符。
def resolve_protocol(metadata: Path, requested: str) -> str:
    # 不能仅凭输出目录名判断 seen/unseen，必须核对元数据。
    """从审计 JSON 读取协议，并阻止命令行标签与真实清单不一致。"""

    # 定位当前活动划分的规则 JSON。
    protocol_path = metadata / "split_protocol.json"
    # 规则文件缺失就无法保证正确协议。
    if not protocol_path.is_file():
        # 拒绝在协议来源不明时继续推理。
        raise FileNotFoundError(f"Missing split protocol: {protocol_path}")
    # 打开现场 UTF-8 规则文件。
    with protocol_path.open("r", encoding="utf-8") as handle:
        # 解析真实划分报告。
        report = json.load(handle)
    # 读取登记的命令行协议名并清理外部空白。
    actual = str(report.get("protocol_cli_name", "")).strip()
    # 现场协议必须属于本项目支持的两种设置。
    if actual not in PROTOCOL_CHOICES:
        # 非法或缺失协议立即报错。
        raise ValueError(
            # 错误包含规则路径和读取到的实际标签。
            f"Invalid protocol_cli_name in {protocol_path}: {actual!r}"
        )
    # 显式请求协议时，必须与实际活动清单一致。
    if requested != "auto" and requested != actual:
        # 发现标签与数据冲突就停止，不能重命名目录掩盖。
        raise ValueError(
            # 列出请求值与现场值，便于切回正确划分。
            f"Requested protocol {requested!r}, but active manifests are {actual!r}"
        )
    # 返回已经核对的真实协议名。
    return actual


# 主流程先只读验证来源与已有产物，确认可安全启动后才生成音频并发布报告。
def main() -> None:
    # 先固定数据/模型身份，再按固定批次生成与保存 WAV。
    """建立配对、加载两个模型、批量生成并保存 WAV。"""

    # 读取本次实际命令行设置。
    args = parse_args()
    # 每批数量必须为正，不能构造空批或错误切片。
    if args.batch_size <= 0:
        # 非法 batch_size 在读取模型前就报错。
        raise ValueError("batch-size must be positive")
    # 教师 epoch 必须为正整数。
    if args.epoch <= 0:
        # 非法 epoch 不能用于拼检查点路径或报告标签。
        raise ValueError("epoch must be positive")

    # 规范化权威项目根目录。
    project_root = args.project_root.resolve()
    # 核对租用服务器身份和准确项目路径，不允许本地替代执行。
    assert_rented_server(project_root)
    # 预处理数据统一从 data 目录读取。
    data_dir = project_root / "data"
    # 清单和协议记录集中在 metadata。
    metadata = data_dir / "metadata"
    # 解析并核对实际活动 seen/unseen 协议。
    protocol_name = resolve_protocol(metadata, args.protocol)
    # 将标签连字符换成下划线，仅用于既有目录命名。
    protocol_tag = protocol_name.replace("-", "_")

    # ``or`` 表示：用户没传路径时才使用按协议自动拼出的默认路径。
    checkpoint_path = (
        # 用户显式给出的 checkpoint 优先使用。
        args.checkpoint
        # 没有显式权重时才按下面规则构造默认路径。
        or (
            # 默认路径从权威项目根目录展开。
            project_root
            # 进入实验产物目录。
            / "exp"
            # 进入模型检查点目录。
            / "checkpoints"
            # 根据真实活动协议选择相应教师训练目录。
            / f"voicegrad30_vctk_{protocol_tag}"
            # 按请求的 epoch 补齐四位权重文件名。
            / f"epoch_{args.epoch:04d}.pt"
        )
    # 消除相对部分和符号链接，得到实际教师路径。
    ).resolve()
    # 规范化本次实际声码器权重路径。
    vocoder_checkpoint = args.vocoder_checkpoint.resolve()
    # 规范化对应声码器配置路径。
    vocoder_config = args.vocoder_config.resolve()
    # auto 只用于方便旧命令；正式脚本会显式传 custom/public。路径判断结果也会
    # 写入 JSON，避免以后看到 WAV 却不知道它来自哪套声码器。
    # 仅 auto 模式需要依据路径推断来源，正式脚本应明确给出来源。
    if args.vocoder_kind == "auto":
        # 路径转换为统一大小写，避免大小写影响旧命令标签推断。
        checkpoint_text = str(vocoder_checkpoint).casefold()
        # 得到将写入报告的声码器来源标签。
        vocoder_kind = (
            # 路径符合项目自训命名规则时标为 custom，后面仍需实际验证权重。
            "custom"
            # 同时识别修改版目录名或 generator_step_ 检查点命名。
            if "hifigan_v1_mrd" in checkpoint_text or "generator_step_" in checkpoint_text
            # 不匹配自训命名时按历史公开版处理；这不等于作者未公开声码器。
            else "public"
        )
    # 显式指定来源时直接使用用户请求，并保留到运行身份。
    else:
        # 记录明确传入的 custom/public 标签。
        vocoder_kind = args.vocoder_kind
    # 为本次运行确定独立输出目录。
    output_dir = (
        # 优先使用命令行明确指定的输出位置。
        args.output_dir
        # 未指定时按协议和教师 epoch 组合默认目录。
        or (
            # 输出从权威项目根目录展开。
            project_root
            # 放在实验产物 exp 内。
            / "exp"
            # 归入评价产物目录。
            / "evaluation"
            # 目录名记录实际协议与请求教师轮数，避免和其他版本混淆。
            / f"voicegrad30_{protocol_tag}_epoch{args.epoch:04d}"
        )
    # 将输出位置规范化为绝对路径。
    ).resolve()
    # 本次配对只从活动测试清单读取。
    manifest = metadata / "test_manifest.csv"
    # 在租用服务器内选定实际推理设备。
    device = torch.device(args.device)

    # 自训声码器必须经过最终检查点验证。
    if vocoder_kind == "custom":
        # 严格检查声码器格式、配置、内部步数及模型参数。
        validate_checkpoint(
            # 从当前项目加载受控声码器实现。
            project_root=project_root,
            # 明确验证本次准备使用的生成器文件。
            checkpoint_path=vocoder_checkpoint,
            # 验证本次声码器对应的原始 JSON 配置。
            config_path=vocoder_config,
            # 要求内部更新步数精确等于最终 2500000。
            expected_step=2_500_000,
        )

    # 计算实际教师权重字节摘要，绑定推理来源。
    teacher_checkpoint_sha256 = sha256_file(checkpoint_path)
    # 计算实际声码器权重摘要，不能只凭文件名区分版本。
    vocoder_checkpoint_sha256 = sha256_file(vocoder_checkpoint)
    # 计算声码器配置原始文件摘要。
    vocoder_config_sha256 = sha256_file(vocoder_config)

    # Build the complete pair manifest before touching an existing output
    # directory.  Its canonical bytes are part of the run identity, so an old
    # WAV can never be silently reinterpreted under a newly generated pairs.csv.
    # 读取真实划分规则的位置，随后写入输入身份。
    split_protocol_path = metadata / "split_protocol.json"
    # 读取训练集 mel 均值文件位置。
    mel_mean_path = data_dir / "stats" / "mel_mean.npy"
    # 读取与均值配套的训练标准差位置。
    mel_std_path = data_dir / "stats" / "mel_std.npy"
    # 测试清单摘要固定本次语音库存版本。
    test_manifest_sha256 = sha256_file(manifest)
    # 协议文件摘要固定文本/说话人隔离规则版本。
    split_protocol_sha256 = sha256_file(split_protocol_path)
    # 均值摘要固定标准化统计量，防止续算时换表。
    mel_mean_sha256 = sha256_file(mel_mean_path)
    # 标准差摘要同样必须固定。
    mel_std_sha256 = sha256_file(mel_std_path)

    # 严格读取并核对 10 人×10 句测试网格。
    rows = load_test_rows(manifest)
    # 构造全部 8100 个合法跨人、跨句配对。
    all_pairs = build_pairs(rows, output_dir)
    # 先在内存得到规范配对 CSV 字节，不触碰旧输出。
    pairs_payload = render_pairs_csv(all_pairs)
    # 为准确配对字节计算摘要，旧 WAV 只能绑定相同配对表。
    expected_pairs_csv_sha256 = hashlib.sha256(pairs_payload).hexdigest()
    # 正式使用全部配对；limit 仅选前若干条作明确诊断。
    selected_pairs = all_pairs if args.limit is None else all_pairs[: args.limit]
    # 不允许空的实际选择。
    if not selected_pairs:
        # 选不到配对则立即报错，不能生成空完成报告。
        raise ValueError("No pairs selected")

    # 登记输入、模型、算法设置和源码字节身份，续跑时要求完全相同。
    run_identity: Dict[str, object] = {
        # 运行身份记录格式版本，区分不同结构的历史报告。
        "identity_format_version": 2,
        # 实际活动数据协议，由现场规则读取并核对。
        "data_protocol": protocol_name,
        # 本次正式测试清单路径。
        "test_manifest": str(manifest),
        # 本次正式测试清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "test_manifest_sha256": test_manifest_sha256,
        # 当前活动数据划分规则路径。
        "split_protocol": str(split_protocol_path),
        # 当前活动数据划分规则路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "split_protocol_sha256": split_protocol_sha256,
        # 训练集 mel 均值路径。
        "mel_mean": str(mel_mean_path),
        # 训练集 mel 均值路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "mel_mean_sha256": mel_mean_sha256,
        # 训练集 mel 标准差路径。
        "mel_std": str(mel_std_path),
        # 训练集 mel 标准差路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "mel_std_sha256": mel_std_sha256,
        # 完整规范配对清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "pairs_csv_sha256": expected_pairs_csv_sha256,
        # 按论文人数与跨人跨句规则构成的全部 8100 对。
        "paper_pair_count": len(all_pairs),
        # 本次实际选择数；少量 limit 诊断会小于全部配对。
        "selected_pair_count": len(selected_pairs),
        # 命令行请求的教师训练轮数。
        "teacher_epoch_requested": int(args.epoch),
        # 本次实际教师检查点路径。
        "teacher_checkpoint": str(checkpoint_path),
        # 本次实际教师检查点路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        # 本次实际声码器来源标签，区分公开与自训。
        "vocoder_kind": vocoder_kind,
        # 本次实际声码器生成器路径。
        "vocoder_checkpoint": str(vocoder_checkpoint),
        # 本次实际声码器生成器路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "vocoder_checkpoint_sha256": vocoder_checkpoint_sha256,
        # 与实际声码器配套的配置路径。
        "vocoder_config": str(vocoder_config),
        # 与实际声码器配套的配置路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "vocoder_config_sha256": vocoder_config_sha256,
        # 教师反向采样次数固定为 30。
        "voicegrad_steps": 30,
        # 源 mel 的起始加噪时间步固定为 950。
        "teacher_start_step": 950,
        # 本次基础随机种子；具体 batch 还会加其首个 pair_id。
        "seed": int(args.seed),
        # 本次固定批大小，属于随机噪声复现规则的一部分。
        "batch_size": int(args.batch_size),
        # 是否明确做少量诊断；正式运行应为 None。
        "selected_pair_limit": args.limit,
        # 实际采用的外部 BNF 时间对齐模式。
        "bnf_alignment": args.bnf_alignment,
        # 实际采用的 U-Net 内部 BNF 抽点规则。
        "bnf_downsampling": args.bnf_downsampling,
        # 是否显式允许诊断性覆盖教师原内部接口。
        "bnf_downsampling_override": bool(
            # 保存是否显式允许诊断性覆盖教师内部 BNF 接口。
            args.diagnostic_allow_bnf_downsampling_override
        ),
        # 关键实现的准确字节指纹，连注释变化也会记录。
        "implementation_sha256": {
            # 本配对与推理文件本身的字节摘要，也包含中文注释变化。
            "voicegrad/infer_voicegrad30_paper_eval.py": sha256_file(Path(__file__)),
            # 扩散数学和采样步骤实现的字节摘要。
            "voicegrad/diffusion.py": sha256_file(project_root / "voicegrad" / "diffusion.py"),
            # 教师网络结构及内部条件注入实现的摘要。
            "voicegrad/model.py": sha256_file(project_root / "voicegrad" / "model.py"),
            # 声码器加载、输入尺度及输出写入实现的摘要。
            "hifigan/tools/vocode.py": sha256_file(project_root / "hifigan" / "tools" / "vocode.py"),
            # 实际声码器网络结构实现的摘要。
            "hifigan/official/models.py": sha256_file(project_root / "hifigan" / "official" / "models.py"),
        },
    }
    # 按固定 JSON 排版建立运行身份的精确预期字节。
    expected_identity_payload = (
        # 保留中文和固定缩进，并与原子写函数保持相同末尾换行。
        json.dumps(run_identity, ensure_ascii=False, indent=2) + "\n"
    # 统一使用 UTF-8，避免同内容不同编码造成身份分歧。
    ).encode("utf-8")
    # 为整份运行身份计算预期字节摘要。
    expected_run_identity_sha256 = hashlib.sha256(
        # 摘要输入必须是将要实际写入磁盘的那些字节。
        expected_identity_payload
    # 得到后续可比较的十六进制身份摘要。
    ).hexdigest()
    # 该目录的运行身份必须在生成 WAV 前建立。
    identity_path = output_dir / "run_identity.json"
    # 完整配对表的正式保存位置。
    pairs_csv = output_dir / "pairs.csv"
    # 只有完整生成并通过末尾检查才写出完成报告。
    report_path = output_dir / "inference_report.json"
    # 旧完成报告进入修复状态时保存在此处，仍保留历史来源。
    recovery_report_path = output_dir / "inference_report.recovery.json"
    # 所有转换 WAV 存放在独立子目录。
    wav_dir = output_dir / "wav"
    # 只读枚举已有 WAV，判断是否已经存在不能重解释的产物。
    existing_wavs = list(wav_dir.glob("*.wav")) if wav_dir.exists() else []
    # 检查正式或恢复中的完成报告是否存在。
    has_completion_artifact = report_path.is_file() or recovery_report_path.is_file()
    # 已有 WAV 或完成记录都属于实质产物，不能因源码变化就随便替换身份。
    has_material_output = bool(existing_wavs) or has_completion_artifact

    # Everything through the completed-report checks below is read-only.  In
    # particular, pairs.csv is never overwritten before an old completion
    # report and its WAV lineage have been validated.
    # 先将已有身份设为未知，随后只读尝试加载。
    existing_identity: Optional[Dict[str, object]] = None
    # 旧身份文件存在时才读取。
    if identity_path.is_file():
        # 捕获 JSON 损坏与磁盘读取错误，区分有没有实质产物。
        try:
            # 读取旧身份并要求结构为 JSON 对象。
            existing_identity = read_json_object(identity_path)
        # 身份无法读取时进入安全处理分支。
        except (OSError, ValueError) as error:
            # 有 WAV 或完成记录就必须保留并停止，不能刷新为新身份。
            if has_material_output:
                # 拒绝复用来源无法读取的已有产物。
                raise RuntimeError(
                    # 错误指出已有音频/完成记录但身份损坏。
                    "Existing WAV/completion artifacts have an unreadable run identity: "
                    # 给出实际身份文件路径便于检查。
                    f"{identity_path}"
                # 把底层读取异常一并保留，方便定位损坏原因。
                ) from error
            # A lineage-only failed attempt has no reusable model output.  It is
            # safe to replace its malformed identity atomically below.
    # 旧身份除了字段相同，还要有准确字节摘要。
    existing_identity_sha256 = (
        # 身份文件不存在时记为 None，不伪造空文件摘要。
        sha256_file(identity_path) if identity_path.is_file() else None
    )
    # 同时检查结构内容和文件字节都匹配当前请求。
    identity_matches = (
        # 第一项要求全部身份字段精确一致。
        existing_identity == run_identity
        # 第二项要求磁盘 JSON 字节也与预期排版一致。
        and existing_identity_sha256 == expected_run_identity_sha256
    )
    # 身份不同且已有实质产物时必须拒绝续算。
    if not identity_matches and has_material_output:
        # 进一步区分是完全没有身份，还是存在另一身份。
        if existing_identity is None:
            # 缺少身份时明确拒绝复用旧结果。
            raise RuntimeError(
                # 指出已有 WAV/完成记录没有配套 run_identity.json。
                "Existing WAV/completion artifacts have no run_identity.json; "
                # 列出被保护的输出目录，避免意外覆盖。
                f"refusing to reuse {output_dir}"
            )
        # 有身份但属于其他实验同样拒绝复用。
        raise RuntimeError(
            # 说明已有结果属于另一套数据/模型/代码设置。
            "Existing WAV/completion artifacts belong to a different run identity; "
            # 保留原目录供用户审查，要求另选新实验目录。
            f"refusing to reuse {output_dir}"
        )

    # 只读计算已有配对表的准确字节摘要。
    existing_pairs_sha256 = (
        # 没有配对文件就记为 None，不能认为已有 WAV 自带配对身份。
        sha256_file(pairs_csv) if pairs_csv.is_file() else None
    )
    # 配对 CSV 必须与本次固定规则产生的字节一致。
    pairs_match = existing_pairs_sha256 == expected_pairs_csv_sha256
    # 配对不匹配且已有实质产物时，禁止改表重解释旧音频。
    if not pairs_match and has_material_output:
        # 发现配对来源冲突立即停止。
        raise RuntimeError(
            # 说明旧 WAV/完成记录没有绑定本次请求的 pairs.csv。
            "Existing WAV/completion artifacts are not bound to the requested pairs.csv; "
            # 输出实际与期望配对摘要，方便核对清单版本。
            f"actual={existing_pairs_sha256}, expected={expected_pairs_csv_sha256}"
        )

    # A completed report is immutable lineage evidence.  A report with valid
    # WAVs returns without rewriting anything.  If only WAV integrity has
    # failed, the verified report is moved to a recovery marker after all
    # read-only checks, and only missing/corrupt members are regenerated.
    # 初始不需要修复完成产物，只有后面发现坏 WAV 才设置为真。
    recovery_needed = False
    # 暂不假定已有完成报告可用，必须读取并逐项核验。
    completed_report: Optional[Dict[str, object]] = None
    # 用此次请求运行身份的确定摘要作为完成报告期望。
    expected_identity_sha256 = expected_run_identity_sha256

    # 独立核对旧完成/恢复报告是否属于本次完全相同的实验。
    def validate_report_lineage(
        # candidate 是已读取报告，candidate_path 用于准确指出出错文件。
        candidate: Dict[str, object], candidate_path: Path
    # 此函数只验证，不修改历史完成记录。
    ) -> None:
        # 列出完成报告必须共同匹配的核心模型、数据和配对字段。
        required_completed_fields = {
            # 按论文人数与跨人跨句规则构成的全部 8100 对。
            "paper_pair_count": len(all_pairs),
            # 本次实际选择数；少量 limit 诊断会小于全部配对。
            "selected_pair_count": len(selected_pairs),
            # 从实际教师检查点内部读取的训练轮数。
            "teacher_epoch": int(args.epoch),
            # 本次实际教师检查点路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            # 本次实际声码器来源标签，区分公开与自训。
            "vocoder_kind": vocoder_kind,
            # 本次实际声码器生成器路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "vocoder_checkpoint_sha256": vocoder_checkpoint_sha256,
            # 与实际声码器配套的配置路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "vocoder_config_sha256": vocoder_config_sha256,
            # 本次正式测试清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "test_manifest_sha256": test_manifest_sha256,
            # 当前活动数据划分规则路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "split_protocol_sha256": split_protocol_sha256,
            # 训练集 mel 均值路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "mel_mean_sha256": mel_mean_sha256,
            # 训练集 mel 标准差路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "mel_std_sha256": mel_std_sha256,
            # 完整规范配对清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "pairs_csv_sha256": expected_pairs_csv_sha256,
            # 本次完整输入与模型身份文件路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
            "run_identity_sha256": expected_identity_sha256,
        }
        # 收集全部字段差异，方便一次发现所有来源冲突。
        mismatches = {
            # 对每个不一致字段记录实际值和期望值。
            key: {"actual": candidate.get(key), "expected": expected}
            # 逐一检查规定的完成字段。
            for key, expected in required_completed_fields.items()
            # 只把实际不相同的字段加入差异表。
            if candidate.get(key) != expected
        }
        # 已完成量应为新生成数加经验证复用数。
        completed_count = int(candidate.get("generated", 0)) + int(
            # 把已复用的完整 WAV 数量加入完成量。
            candidate.get("skipped_existing", 0)
        )
        # 任何身份差异或数量不完整都不能接受旧完成报告。
        if mismatches or completed_count != len(selected_pairs):
            # 拒绝有冲突或不完整的完成记录。
            raise RuntimeError(
                # 错误指出具体是哪一份完成/恢复报告。
                f"Existing {candidate_path.name} is inconsistent with the requested run: "
                # 列出字段差异及报告声称的完成量。
                f"mismatches={mismatches}, completed_count={completed_count}, "
                # 再列本次请求应完成的数量，区分子集诊断和正式评估。
                f"expected_count={len(selected_pairs)}"
            )

    # 恢复标记存在时先验证它仍属于本次请求。
    if recovery_report_path.is_file():
        # 读取之前保留的完成报告副本。
        recovery_report = read_json_object(recovery_report_path)
        # 复核恢复标记的数据/模型/配对来源。
        validate_report_lineage(recovery_report, recovery_report_path)

    # 正式完成报告存在时也做同样严格核验。
    if report_path.is_file():
        # 读取原完成报告，不立即覆盖。
        completed_report = read_json_object(report_path)
        # 确认完成字段与本次请求精确一致。
        validate_report_lineage(completed_report, report_path)
        # 再逐条检查对应 WAV 是否仍可读取且格式正确。
        invalid_wavs = [
            # 仅记录损坏或缺失 WAV 的配对编号。
            pair.pair_id
            # 检查本次全部选择配对，不能只抽查几个文件。
            for pair in selected_pairs
            # 文件缺失、采样率不符或头损坏都需要恢复生成。
            if not valid_existing_wav(Path(pair.output_wav))
        ]
        # 若完成来源匹配且所有 WAV 完整，就保持原样返回。
        if not invalid_wavs:
            # 打印原来的完成报告，不伪造一次新的生成记录。
            print(json.dumps(completed_report, ensure_ascii=False, indent=2))
            # 明确告知该完整推理已经完成，无需重复计算。
            print("VOICEGRAD30_PAPER_PAIR_INFERENCE_ALREADY_COMPLETE")
            # 结束主流程，不重写模型、配对或完成报告。
            return
        # 只有确定旧来源正确但 WAV 不完整时才启动恢复。
        recovery_needed = True

    # A failed run that produced no WAV and no completion artifact may be
    # safely restarted after a code update: atomically refresh its identity and
    # canonical pairs.  Any material artifact above made drift fail closed.
    # 没有实质产物的失败尝试可以按当前请求刷新身份。
    if not identity_matches:
        # 原子写入完整运行身份，再允许后续生成音频。
        atomic_write_json(identity_path, run_identity)
    # 配对内容尚不匹配且没有受保护的旧产物时，发布规范配对表。
    if not pairs_match:
        # 写完整 8100 对清单，诊断 limit 也不改变表的总内容。
        write_pairs_csv(all_pairs, pairs_csv)
    # 重新读取磁盘身份摘要，确认发布字节与预期完全相同。
    if sha256_file(identity_path) != expected_run_identity_sha256:
        # 身份写入不一致时停止，不能开始产生不可追溯 WAV。
        raise RuntimeError("Published run_identity.json bytes do not match the requested identity")
    # 同样核对实际发布的 pairs.csv 字节摘要。
    if sha256_file(pairs_csv) != expected_pairs_csv_sha256:
        # 配对发布不一致时立即失败。
        raise RuntimeError("Published pairs.csv does not match the canonical pair manifest")

    # 旧报告已核验但有音频待修复时，先撤下当前完成标记。
    if recovery_needed:
        # Remove the stale active-completion signal before regeneration.  The
        # recovery marker preserves its lineage if GPU inference is interrupted.
        # 原完成报告移动为恢复记录，保留其来源证据；中断后仍可继续核验。
        os.replace(report_path, recovery_report_path)

    # 在来源核验全部通过后创建输出目录。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 创建波形子目录，已有有效文件保留。
    wav_dir.mkdir(parents=True, exist_ok=True)

    # 读训练 mel 均值并整理成 [80,1]，对时间轴广播。
    mean = np.load(mel_mean_path, allow_pickle=False).reshape(-1, 1)
    # 同样读取标准差，不用测试语音重拟合统计。
    std = np.load(mel_std_path, allow_pickle=False).reshape(-1, 1)
    # 要求两份统计量都有 80 个 mel 频带，防止混用其他前端。
    if mean.shape != (80, 1) or std.shape != (80, 1):
        # 统计形状错误时显示实际形状并停止。
        raise ValueError(f"Invalid mel stats shapes: mean={mean.shape}, std={std.shape}")

    # 来源检查完成后才加载实际教师网络。
    model, teacher_checkpoint = load_voicegrad(
        # 教师路径采用本次已经锁定摘要的文件。
        checkpoint_path,
        # 教师放到指定租用设备。
        device,
        # 内部 BNF 下采样使用明确登记的模式。
        bnf_downsampling=args.bnf_downsampling,
        # 是否覆盖旧教师内部接口由显式诊断开关决定。
        allow_bnf_downsampling_override=(
            # 将该诊断授权传给加载器，默认没有授权。
            args.diagnostic_allow_bnf_downsampling_override
        ),
    )
    # 用教师统一配置构建扩散时间表。
    diffusion = VoiceGradDiffusion(
        # 扩散总步数及余弦偏移取统一 PAPER_CONFIG。
        PAPER_CONFIG.diffusion_steps, PAPER_CONFIG.cosine_offset
    # 将系数张量放到和网络相同设备。
    ).to(device)
    # 加载本次声码器并取得实际运行配置用于报告。
    vocoder, vocoder_config_dict = load_generator(
        # 权威项目目录决定读取哪份受控实现。
        project_root=project_root,
        # 使用已经锁定摘要的声码器权重。
        checkpoint_path=vocoder_checkpoint,
        # 使用与该权重配套的配置。
        config_path=vocoder_config,
        # 教师和声码器在同一计算设备上工作。
        device=device,
    )

    # 固定 chunks 的起始 pair_id 决定随机 seed；中断续跑不会改变未完成 chunk 的噪声。
    # 按源句和固定批大小切批，续算时批次边界及噪声不改变。
    chunks = list(fixed_chunks(selected_pairs, args.batch_size))
    # 累计本次真正新写出的 WAV 数量。
    ok_count = 0
    # 累计已存在且通过格式验证的 WAV 数量。
    skip_count = 0
    # 记录实际生成阶段开始时间。
    start_time = time.time()

    # 按固定批次执行推理，进度条明确声码器版本。
    for chunk in tqdm(chunks, desc=f"VoiceGrad-30 + {vocoder_kind} HiFi-GAN"):
        # 先找出这一批尚需生成或被请求覆盖的配对。
        missing = [
            # 保留完整配对记录，后面仍使用原批次来固定随机噪声。
            pair
            # 遍历当前固定批次。
            for pair in chunk
            # 显式覆盖请求或音频不可复用时才需要重新写结果。
            if args.overwrite or not valid_existing_wav(Path(pair.output_wav))
        ]
        # 整批文件都完整且无需覆盖时，直接复用。
        if not missing:
            # 整批加入跳过数，仍计入最后完成量。
            skip_count += len(chunk)
            # 继续下一个固定批次，不浪费重复推理。
            continue

        # 批内所有配对共享源句，取第一条即可取得共同源内容。
        source = chunk[0]
        # 读取源语音的原始 log-mel。
        mel = feature_matrix(
            # 按源说话人及源句编号定位 mel，不能读目标参考 mel。
            feature_path(data_dir, "mel", source.source_speaker, source.source_utt),
            # 要求 mel 为 80 维，与教师输入及声码器一致。
            PAPER_CONFIG.n_mels,
        )
        # 读取同一源语音的 BNF 内容。
        bnf = feature_matrix(
            # BNF 路径必须与 mel 来自同一句，防止内容条件串位。
            feature_path(data_dir, "bnf", source.source_speaker, source.source_utt),
            # 要求内容特征维度为配置中的 144。
            PAPER_CONFIG.n_bnf,
        )

        # 逐频带减训练均值再除训练标准差，与教师训练输入一致。
        mel_norm = ((mel - mean) / std).astype(np.float32)
        # 增加批维得到 [1,80,T] 并转到设备。
        mel_tensor = torch.from_numpy(mel_norm).unsqueeze(0).to(device)
        # 将原始 BNF 时间轴插值到源 mel 的长度。
        bnf_tensor = align_bnf_to_mel(
            # 先得到 [1,144,T_bnf] 张量。
            torch.from_numpy(bnf).unsqueeze(0).to(device),
            # 目标帧数由当前源 mel 的实际时间长度决定。
            mel_tensor.shape[-1],
            # 外部对齐模式明确来自参数并写入身份。
            mode=args.bnf_alignment,
            # mel 物理时间轴采用 22050 Hz 采样率。
            sample_rate=PAPER_CONFIG.sample_rate,
            # FFT 窗长决定帧中心计算的一部分。
            n_fft=PAPER_CONFIG.n_fft,
            # hop 决定相邻 mel 帧的时间距离。
            hop_size=PAPER_CONFIG.hop_size,
            # 原始 BNF 帧步长为 10 毫秒，物理对齐模式据此计算。
            bnf_frame_shift_seconds=0.010,
        )

        # 本批配对数量可能是源句最后一个不足 4 条的尾批。
        batch_count = len(chunk)
        # 把共同源 mel 扩为 [批,80,T]，每条仅目标音色不同。
        mel_batch = mel_tensor.expand(batch_count, -1, -1).contiguous()
        # 把共同 BNF 扩为 [批,144,T]，与 mel 顺序一致。
        bnf_batch = bnf_tensor.expand(batch_count, -1, -1).contiguous()

        # 收集每个目标参考的 256 维 GE2E 音色向量。
        speaker_vectors = []
        # 严格按当前配对顺序逐条读取音色条件。
        for pair in chunk:
            # 为当前目标参考定位说话人嵌入。
            vector_path = feature_path(
                # 从同一套预处理数据目录读取。
                data_dir,
                # 明确特征种类是 spk_emb，不是用于内容的 BNF。
                "spk_emb",
                # 使用目标参考说话人的子目录。
                pair.reference_speaker,
                # 使用提供音色的实际参考语音编号。
                pair.reference_utt,
            )
            # 禁止 pickle，读取普通数组并去掉多余单例轴。
            vector = np.load(vector_path, allow_pickle=False).squeeze()
            # 要求音色向量维度与教师配置一致，不能随意补齐。
            if vector.shape != (PAPER_CONFIG.spk_emb_dim,):
                # 不匹配时报出具体形状和文件。
                raise ValueError(f"Invalid speaker embedding {vector.shape}: {vector_path}")
            # 统一 float32 后按配对順序收集。
            speaker_vectors.append(vector.astype(np.float32))
        # 堆叠为 [批,256] 并放到设备，逐条控制对应输出音色。
        speaker_batch = torch.from_numpy(np.stack(speaker_vectors)).to(device)

        # 每个固定批次用独立随机生成器，续跑不会受此前跳过批次影响。
        chunk_generator = torch.Generator(device=device).manual_seed(
            # 基础种子加该批第一条 pair_id，固定该批全部高斯噪声。
            int(args.seed) + int(chunk[0].pair_id)
        )

        # 只推理，关闭梯度与训练状态追踪，降低显存开销。
        with torch.inference_mode():
            # 用冻结教师进行 30 次反向扩散转换。
            converted_norm = diffusion.sample(
                # 使用已加载的教师网络。
                model,
                # 源 mel 提供初始加噪语音内容。
                mel_batch,
                # 参考 GE2E 向量提供目标说话人音色。
                speaker_batch,
                # 对齐后的源 BNF 提供语言内容条件。
                bnf_batch,
                # 固定执行 30 次反向采样，对应 VoiceGrad-30。
                num_steps=30,
                # 从第 950 扩散时间步开始，和已登记设置一致。
                start_step=950,
                # 使用当前固定批次的独立随机生成器。
                generator=chunk_generator,
                # 按论文公式（9）先给源 mel 加噪，而非直接从无条件纯噪声开始。
                diffuse_source=True,
            )
            # 训练均值转成同设备 float32 张量，准备逆标准化。
            mean_tensor = torch.from_numpy(mean.astype(np.float32)).to(device)
            # 标准差同样转成同设备张量。
            std_tensor = torch.from_numpy(std.astype(np.float32)).to(device)
            # 先乘回标准差再加均值，恢复声码器期望的原始 log-mel 尺度。
            converted_mel = converted_norm * std_tensor.unsqueeze(0) + mean_tensor.unsqueeze(0)
            # 声码器输出波形，去掉单声道维后转成 CPU NumPy 数组以保存。
            generated_audio = vocoder(converted_mel).squeeze(1).cpu().numpy()

        # 按相同配对顺序将生成音频和输出路径关联。
        for pair, audio in zip(chunk, generated_audio):
            # 取得该配对规定的目标 WAV 路径。
            destination = Path(pair.output_wav)
            # 未授权覆盖且当前文件完整时保留原音频。
            if not args.overwrite and valid_existing_wav(destination):
                # 这条保留的 WAV 计入复用数量。
                skip_count += 1
                # 跳过写入，继续下一条配对。
                continue
            # 发布之前检查所有音频采样值有限。
            if not np.isfinite(audio).all():
                # 发现 NaN/Inf 立即指出配对编号，不写坏音频。
                raise ValueError(f"Generated waveform contains NaN/Inf: pair {pair.pair_id}")
            # 使用原子写入，避免下游评价读到半个 WAV。
            atomic_write_wav(
                # 写入当前配对唯一指定的路径。
                destination,
                # 转为 float32，并将合法波形范围限制在 [-1,1]。
                np.clip(audio.astype(np.float32), -1.0, 1.0),
                # 保存为与前端和声码器一致的 22.05 kHz。
                PAPER_CONFIG.sample_rate,
            )
            # 确认一条写出后再增加本次生成数。
            ok_count += 1

    # 完成量由新生成数加有效复用数构成。
    completed_count = ok_count + skip_count
    # 必须完整覆盖本次实际选择的配对。
    if completed_count != len(selected_pairs):
        # 数量不完整时拒绝发布完成报告。
        raise RuntimeError(
            # 错误先给出实际完成量。
            f"Inference incomplete: generated+skipped={completed_count}, "
            # 再给出应完成的配对数量。
            f"expected={len(selected_pairs)}"
        )
    # 所有批次结束后再次逐文件核验 WAV，防止途中丢失或写坏。
    invalid_final_wavs = [
        # 记录最后发现异常的配对编号。
        pair.pair_id
        # 遍历本次所有选择的配对。
        for pair in selected_pairs
        # 不符合采样率、长度或单声道要求的文件视为异常。
        if not valid_existing_wav(Path(pair.output_wav))
    ]
    # 只要还有异常 WAV，就不能声明推理完成。
    if invalid_final_wavs:
        # 停止并保留现有产物，以便定位或后续恢复。
        raise RuntimeError(
            # 明确拒绝发布带有缺失/损坏音频的完成报告。
            "Refusing to publish inference_report.json with missing/corrupt WAVs: "
            # 给出前十个异常编号及总数，避免日志过长。
            f"examples={invalid_final_wavs[:10]}, total={len(invalid_final_wavs)}"
        )

    # Fail closed if an input asset changed during a long inference run.  No
    # completion report is published, so those WAVs cannot be consumed under a
    # false identity; a subsequent run will also reject identity drift while
    # material WAVs remain in this directory.
    # 长时间推理结束后再次核对关键数据与身份字节，防止途中被别的任务改动。
    final_input_hashes = {
        # 本次正式测试清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "test_manifest_sha256": sha256_file(manifest),
        # 当前活动数据划分规则路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "split_protocol_sha256": sha256_file(split_protocol_path),
        # 训练集 mel 均值路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "mel_mean_sha256": sha256_file(mel_mean_path),
        # 训练集 mel 标准差路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "mel_std_sha256": sha256_file(mel_std_path),
        # 完整规范配对清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "pairs_csv_sha256": sha256_file(pairs_csv),
        # 本次完整输入与模型身份文件路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "run_identity_sha256": sha256_file(identity_path),
    }
    # 期望值使用启动时保存的同一批摘要，不能临时接受变化后的文件。
    expected_final_input_hashes = {
        # 本次正式测试清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "test_manifest_sha256": test_manifest_sha256,
        # 当前活动数据划分规则路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "split_protocol_sha256": split_protocol_sha256,
        # 训练集 mel 均值路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "mel_mean_sha256": mel_mean_sha256,
        # 训练集 mel 标准差路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "mel_std_sha256": mel_std_sha256,
        # 完整规范配对清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "pairs_csv_sha256": expected_pairs_csv_sha256,
        # 本次完整输入与模型身份文件路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "run_identity_sha256": expected_run_identity_sha256,
    }
    # 前后摘要不同表示音频已无法完整对应最初登记的输入。
    if final_input_hashes != expected_final_input_hashes:
        # 拒绝在来源漂移时写完成报告。
        raise RuntimeError(
            # 错误说明输入或来源文件在生成期间发生变化。
            "Inference inputs/lineage changed while WAV generation was running: "
            # 列出前后摘要集合，便于定位哪份文件被修改。
            f"actual={final_input_hashes}, expected={expected_final_input_hashes}"
        )

    # 只有全部数量、WAV 完整性和来源检查通过后才建立最终报告。
    report = {
        # 标记采用 FastVoiceGrad VCTK 配对协议的客观评估。
        "protocol": "FastVoiceGrad VCTK 8,100-pair objective evaluation",
        # 按论文人数与跨人跨句规则构成的全部 8100 对。
        "paper_pair_count": 8100,
        # 本次实际选择数；少量 limit 诊断会小于全部配对。
        "selected_pair_count": len(selected_pairs),
        # 本次实际新生成并成功保存的 WAV 数量。
        "generated": ok_count,
        # 已有且通过格式检查而复用的 WAV 数量。
        "skipped_existing": skip_count,
        # 从实际教师检查点内部读取的训练轮数。
        "teacher_epoch": int(teacher_checkpoint["epoch"]),
        # 教师反向采样次数固定为 30。
        "voicegrad_steps": 30,
        # 源 mel 的起始加噪时间步固定为 950。
        "teacher_start_step": 950,
        # 完整保存程序生成的 30 个整数采样时间步升序。
        "sampling_sequence_ascending": diffusion.sampling_sequence(30, 950),
        # 保存实际反向执行的采样时间步顺序。
        "sampling_sequence_descending": list(
            # 将同一采样序列反转为实际由高噪声到低噪声的执行顺序。
            reversed(diffusion.sampling_sequence(30, 950))
        ),
        # 说明采用按公式（9）加噪后的源 mel 作为起点。
        "initial_state": "diffused source x_src_SK via paper Equation (9)",
        # 说明必须跨说话人并排除相同固定公共句号。
        "pair_rule": "different speaker and different fixed sentence ID within 001--010",
        # 实际活动数据协议，由现场规则读取并核对。
        "data_protocol": protocol_name,
        # 真实协议是否为 text-unseen，便于与 text-seen 对照区分。
        "strict_text_unseen_protocol": protocol_name == "text-unseen",
        # 本次正式测试清单路径。
        "test_manifest": str(manifest),
        # 本次正式测试清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "test_manifest_sha256": test_manifest_sha256,
        # 当前活动数据划分规则路径。
        "split_protocol": str(split_protocol_path),
        # 当前活动数据划分规则路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "split_protocol_sha256": split_protocol_sha256,
        # 训练集 mel 均值路径。
        "mel_mean": str(mel_mean_path),
        # 训练集 mel 均值路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "mel_mean_sha256": mel_mean_sha256,
        # 训练集 mel 标准差路径。
        "mel_std": str(mel_std_path),
        # 训练集 mel 标准差路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "mel_std_sha256": mel_std_sha256,
        # 本次真实使用的测试说话人列表。
        "test_speakers": sorted({row["speaker"] for row in rows}),
        # 固定公共测试句号 001–010。
        "test_sentence_ids": [f"{number:03d}" for number in range(1, 11)],
        # 本正式测试网格没有静默接受的缺句例外。
        "test_item_exceptions": None,
        # 固定批次首个 pair_id 加基础种子的明确噪声规则。
        "seed_rule": "base_seed + first_pair_id_of_fixed_source-local_chunk",
        # 本次基础随机种子。
        "base_seed": int(args.seed),
        # 本次固定批大小，属于随机噪声复现规则的一部分。
        "batch_size": int(args.batch_size),
        # 实际采用的外部 BNF 时间对齐模式。
        "bnf_alignment": args.bnf_alignment,
        # 实际采用的 U-Net 内部 BNF 抽点规则。
        "bnf_downsampling": args.bnf_downsampling,
        # 是否显式允许诊断性覆盖教师原内部接口。
        "bnf_downsampling_override": bool(
            # 记录诊断性内部接口覆盖是否开启，正式流程应保持关闭。
            args.diagnostic_allow_bnf_downsampling_override
        ),
        # 本次实际声码器来源标签，区分公开与自训。
        "vocoder_kind": vocoder_kind,
        # 用文字进一步区分实际自训架构和历史公开对照。
        "vocoder_status": (
            # 自训来源明确记录为 HiFi-GAN V1 生成器、MPD 和 UnivNet MRD 训练配置。
            "project-trained HiFi-GAN V1 + MPD + UnivNet MRD"
            # 仅在实际来源标签为 custom 时使用上述说明。
            if vocoder_kind == "custom"
            # 公开 VCTK V1 明确标为公开对照，不能冒充作者未公开声码器。
            else "temporary public HiFi-GAN VCTK_V1; not the unpublished paper vocoder"
        ),
        # 本次实际教师检查点路径。
        "teacher_checkpoint": str(checkpoint_path),
        # 本次实际教师检查点路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        # 本次实际声码器生成器路径。
        "vocoder_checkpoint": str(vocoder_checkpoint),
        # 本次实际声码器生成器路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "vocoder_checkpoint_sha256": vocoder_checkpoint_sha256,
        # 与实际声码器配套的配置路径。
        "vocoder_config": str(vocoder_config),
        # 与实际声码器配套的配置路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "vocoder_config_sha256": vocoder_config_sha256,
        # 声码器实际加载配置的完整内容。
        "vocoder_runtime_config": vocoder_config_dict,
        # 本次完整输入与模型身份文件路径。
        "run_identity": str(identity_path),
        # 本次完整输入与模型身份文件路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "run_identity_sha256": sha256_file(identity_path),
        # 完整规范配对清单路径。
        "pairs_csv": str(pairs_csv),
        # 完整规范配对清单路径的准确字节摘要，用于启动、续算与完成时核对同一版本。
        "pairs_csv_sha256": expected_pairs_csv_sha256,
        # 标明本次是否修复了先前完成后出现缺失/损坏 WAV 的结果。
        "recovered_from_prior_completion": recovery_report_path.is_file(),
        # 实际推理阶段耗时，包含本次生成与复用检查。
        "elapsed_seconds": time.time() - start_time,
    }
    # 原子发布完整推理报告，下游看到它后才可开始正式评价。
    atomic_write_json(report_path, report)
    # 如果这是从已核验的旧完成记录恢复，需要结束恢复状态。
    if recovery_report_path.is_file():
        # 新完成报告已发布后删除恢复标记；当前完成报告已保留完整来源，不删除任何权重或 WAV。
        recovery_report_path.unlink()

    # 日志中输出实际完成报告，便于远程核查。
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 最后打印推理通过标志，任何前述错误都会阻止执行到这里。
    print("VOICEGRAD30_PAPER_PAIR_INFERENCE_PASSED")


# 仅直接运行本文件时启动推理；导入配对和加载函数不会自动执行任务。
if __name__ == "__main__":
    # 进入已设置严格来源保护的主流程。
    main()
