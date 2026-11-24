#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 本模块对已生成语音计算四项客观指标，并绑定每条分数的模型、波形和清单来源。
"""
按照 FastVoiceGrad 论文的客观评估项目，评价正式生成的 8,100 条语音。

本文件只做“评价”，不再运行 VoiceGrad/FastVoiceGrad，也不修改任何 WAV。它读取批量推理脚本生成的
pairs.csv，然后依次计算论文表 1 使用的四项指标：

1. UTMOS：无需真人打分的自然度预测分数，越高越好；
2. DNSMOS：无需干净参考语音的总体质量分数，越高越好；
3. CER：wav2vec 2.0 识别结果相对 source 文本的字符错误率，越低越好；
4. SVA：ECAPA-TDNN 判断转换语音是否像 target reference，说话人验证准确率越高越好。

论文公开了指标名称，却没有公开作者当时的 8,100-pair CSV、各指标模型文件哈希、
SVA 阈值和 CER 文本清洗代码。为了不把“我们的合理补全”冒充作者原始实现，本文件会：

* 使用公开的 UTMOS strong learner、TorchAudio wav2vec2.0、SpeechBrain ECAPA-TDNN、
  Microsoft DNSMOS；
* 固定并记录 ECAPA 官方 verify_batch 默认阈值 0.25；
* 固定并记录 CER 文本标准化规则；
* 保存每一条语音的得分 CSV，而不是只留下一个无法追查的平均数；
* 把所有评价模型的 SHA-256、软件版本、样本数和计算方法写进 JSON。

默认目录保留给既有 VoiceGrad-30 教师结果。FastVoiceGrad 学生正式评价请使用固定包装：

bash text_unseen/08_evaluate_student.sh

冒烟测试示例（只算前 8 条，不会伪装成完整论文结果）：

python3 -m evaluation.evaluate_metrics --metric all --limit 8 \
        --output-dir /tmp/voicegrad30_metric_smoke
"""

# 延后求值类型提示，保持旧 Python 解释器兼容。
from __future__ import annotations

# argparse 读取命令行；csv/json 保存逐条结果和汇总；hashlib 记录模型指纹。
import argparse
# 使用 csv 读写带表头的逐条记录。
import csv
# 使用 hashlib 计算数据与权重文件指纹。
import hashlib
# 使用 importlib.metadata 查询实际安装的软件版本。
import importlib.metadata
# 使用 json 读写配置及来源记录。
import json
# 使用 math 执行有限数、取整等数值辅助检查。
import math
# 使用 multiprocessing 建立独立 CPU 工作进程。
import multiprocessing
# 使用 os 执行路径、原子替换和进程级辅助操作。
import os
# 使用 re 严格检查字符串格式。
import re
# 使用 sys 处理解释器版本和模块搜索路径。
import sys
# 使用 time 记录耗时与报告生成时刻。
import time
# 使用 unicodedata 统一文本字符编码形式。
import unicodedata

# 用进程池并行计算 CPU 指标，按提交顺序收回结果。
from concurrent.futures import ProcessPoolExecutor

# Path 比手工拼接斜杠更安全；类型提示让各函数的输入输出更容易看懂。
from pathlib import Path
# 标明张量、清单和配置的输入输出类型，不改变运行数据。
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# NumPy 做平均值和有限数检查；SoundFile 读取 22.05 kHz 单声道 WAV。
import numpy as np
# 使用 SoundFile 读取真实波形和 WAV 文件头。
import soundfile as sf

# PyTorch 承载波形和三个神经评价模型；torchaudio 负责重采样及 wav2vec2.0。
import torch
# 使用 TorchAudio 的重采样器与固定 ASR 模型。
import torchaudio
# 使用右侧补零等无独立参数的张量操作。
import torch.nn.functional as F

# tqdm 显示长达 8,100 条的评价进度。
from tqdm import tqdm

# 复用项目统一服务器根路径，避免在不同函数里写出互相矛盾的目录。
from voicegrad.config import SERVER_PROJECT_ROOT
# 限制实际计算只能在当前租用服务器执行。
from voicegrad.runtime_guard import assert_rented_server


# 正式推理目录必须和 voicegrad/infer_voicegrad30_paper_eval.py 的默认输出完全一致。
DEFAULT_INFERENCE_DIR = (
    # 从已经核对的服务器项目根目录开始拼接路径。
    SERVER_PROJECT_ROOT
    # 进入实验产物目录。
    / "exp"
    # 进入推理和指标结果目录。
    / "evaluation"
    # 默认读取教师 500 轮配自训声码器的 text-unseen 结果。
    / "voicegrad30_text_unseen_epoch0500_custom_hifigan"
)

# 逐条指标 CSV 和总报告默认放在推理目录下的 metrics 子文件夹。
DEFAULT_OUTPUT_DIR = DEFAULT_INFERENCE_DIR / "metrics"

# 官方 UTMOS Hugging Face Space 已经完整下载在项目内，不需要运行时联网。
DEFAULT_UTMOS_DIR = SERVER_PROJECT_ROOT / "UTMOS-demo"

# SpeechBrain 官方 ECAPA-TDNN 模型也已下载在本地；目录里的超参数会指定网络结构。
DEFAULT_ECAPA_DIR = SERVER_PROJECT_ROOT / "pretrained_spkrec_ecapa"

# SVA 的 0.25 是 SpeechBrain SpeakerRecognition.verify_batch 的公开默认阈值。
SVA_THRESHOLD = 0.25

# UTMOS 逐条结果只保存配对编号与预测分数。
UTMOS_FIELDS = ("pair_id", "utmos")
# DNSMOS 同时保留四个输出，避免总体分数与 P.808 分数混用。
DNSMOS_FIELDS = ("pair_id", "p808_mos", "mos_sig", "mos_bak", "mos_ovr")
# CER 留下两份规范化文本、错误数和分母，可逐句重算。
CER_FIELDS = (
    # 字段含义：转换配对编号。
    "pair_id",
    # 字段含义：统一书写格式后的正确文本。
    "reference_text_normalized",
    # 字段含义：统一书写格式后的 ASR 识别文本。
    "hypothesis_text_normalized",
    # 字段含义：字符插入、删除、替换总次数。
    "char_errors",
    # 字段含义：参考文本字符数。
    "reference_chars",
    # 字段含义：字符错误率百分数，越低越好。
    "cer_percent",
)
# SVA 同时记录连续相似度、阈值和二元判断。
SVA_FIELDS = ("pair_id", "cosine_score", "threshold", "verified")

# 论文表 1 的 VoiceGrad-30 数值只作为对照，不参与我们的计算。
PAPER_VOICEGRAD30_REFERENCE = {
    # 记录UTMOS 自然度预测分数，越高越好。
    "utmos": 3.88,
    # 记录DNSMOS 总体质量预测分数，越高越好。
    "dnsmos": 3.77,
    # 记录字符错误率百分数，越低越好。
    "cer_percent": 2.20,
    # 记录说话人验证通过比例，越高越好。
    "sva_percent": 82.9,
}

# 论文表 1 的完整 ACDD FastVoiceGrad 学生数值；同样只作对照。
PAPER_FASTVOICEGRAD_REFERENCE = {
    # 记录UTMOS 自然度预测分数，越高越好。
    "utmos": 3.96,
    # 记录DNSMOS 总体质量预测分数，越高越好。
    "dnsmos": 3.77,
    # 记录字符错误率百分数，越低越好。
    "cer_percent": 1.89,
    # 记录说话人验证通过比例，越高越好。
    "sva_percent": 83.0,
}
# 这是历史 100 轮学生权重的固定指纹；新重训结果需要单独审核身份。
EXPECTED_FASTVOICEGRAD_STUDENT_SHA256 = (
    # 固定历史学生权重的 64 位 SHA-256，不能代表新修复实验。
    "eb9ad626160372f1b45b29d6c6d3f058fa8e553cfe48a5ad6a8c007e00d7ca27"
)
# 这是历史学生训练约定的固定指纹，禁止无说明地混用新旧实验。
EXPECTED_FASTVOICEGRAD_RUN_CONTRACT_SHA256 = (
    # 固定历史训练约定的 SHA-256，防止新旧目标混在同份成绩里。
    "ea2eac524d970461e928cccf473c0ccf614d9873c6ca8b857f3c1d09ee18d3b1"
)


# 辨别教师与历史学生评估身份；固定 SHA 对应既有实验，不能直接冒充新重训模型。
def resolve_inference_system(inference_metadata: Mapping[str, object]) -> Dict[str, object]:
    # 辨别教师与历史学生评估身份；固定 SHA 对应既有实验，不能直接冒充新重训模型。
    """Read an explicit system identity while keeping old teacher reports compatible."""

    # 从已有记录取得教师或学生系统标识。
    system_id = str(inference_metadata.get("system_id", "voicegrad30")).strip().lower()
    # 设置教师模型训练轮数。
    teacher_epoch = int(inference_metadata["teacher_epoch"])
    # 教师结果沿用旧报告字段，并将学生字段明确置空。
    if system_id == "voicegrad30":
        # 教师轮数必须为正，缺失或无效记录不能进入正式评价。
        if teacher_epoch <= 0:
            # 报错原因：教师轮数必须为正，缺失或无效记录不能进入正式评价。
            raise ValueError("VoiceGrad inference report has an invalid teacher_epoch")
        # 把已核对的字段组成返回记录，供后续身份检查或报告保存。
        return {
            # 记录教师或学生系统标识。
            "system_id": system_id,
            # 记录报告中的模型名称。
            "system_label": str(
                # 读取推理阶段明确保存的模型标签，旧报告缺失时使用随后指定的兼容值。
                inference_metadata.get(
                    # 教师标签默认包含实际轮数。
                    "system_label", f"VoiceGrad-30 epoch {teacher_epoch}"
                )
            ),
            # 记录教师或学生角色。
            "model_role": "teacher",
            # 记录教师模型训练轮数。
            "teacher_epoch": teacher_epoch,
            # 记录学生模型训练轮数。
            "student_epoch": None,
            # 记录学生优化器累计更新次数。
            "student_global_step": None,
            # 记录学生推理去噪次数。
            "student_steps": None,
            # 记录学生推理起始噪声步。
            "student_start_step": None,
            # 记录学生权重路径。
            "student_checkpoint": None,
            # 记录学生权重的字节指纹。
            "student_checkpoint_sha256": None,
            # 记录学生训练约定的指纹。
            "student_run_contract_sha256": None,
            # 记录选用的论文模型行。
            "paper_reference_row": "VoiceGrad-30",
            # 记录论文公布的参考数值，不参与评分。
            "paper_reference": PAPER_VOICEGRAD30_REFERENCE,
        }
    # 学生结果执行更严格的完整性和固定权重身份检查。
    if system_id == "fastvoicegrad":
        # 从已有记录取得教师或学生角色。
        model_role = str(inference_metadata.get("model_role", "")).strip().lower()
        # 从已有记录取得流程完成状态。
        status = str(inference_metadata.get("status", "")).strip()
        # 从已有记录取得正式协议要求的配对总数。
        paper_pair_count = int(inference_metadata.get("paper_pair_count", -1))
        # 从已有记录取得本次选中的配对数。
        selected_pair_count = int(inference_metadata.get("selected_pair_count", -1))
        # 从已有记录取得学生模型训练轮数。
        student_epoch = int(inference_metadata.get("student_epoch", -1))
        # 从已有记录取得学生优化器累计更新次数。
        student_global_step = int(inference_metadata.get("student_global_step", -1))
        # 从已有记录取得学生推理去噪次数。
        student_steps = int(inference_metadata.get("student_steps", -1))
        # 从已有记录取得学生推理起始噪声步。
        student_start_step = int(inference_metadata.get("student_start_step", -1))
        # FastVoiceGrad 必须明确声明学生角色，避免教师结果误贴标签。
        if model_role != "student":
            # 报错原因：FastVoiceGrad 必须明确声明学生角色，避免教师结果误贴标签。
            raise ValueError("FastVoiceGrad inference report must declare model_role=student")
        # 组合检查下面各项：推理必须已经达到正式全量完成状态。
        if (
            # 推理必须已经达到正式全量完成状态。
            status != "formal_8100_complete"
            # 协议记录必须声明完整 8,100 对。
            or paper_pair_count != 8100
            # 实际推理选中数量也必须是 8,100。
            or selected_pair_count != 8100
        ):
            # 报错原因：实际推理选中数量也必须是 8,100。
            raise ValueError(
                # 实际推理选中数量也必须是 8,100。
                "FastVoiceGrad objective evaluation requires a completed formal 8,100-pair inference"
            )
        # 此历史正式配置只接受教师 500 轮、学生 100 轮及 123,000 次更新。
        if teacher_epoch != 500 or student_epoch != 100 or student_global_step != 123000:
            # 报错原因：此历史正式配置只接受教师 500 轮、学生 100 轮及 123,000 次更新。
            raise ValueError(
                # 此历史正式配置只接受教师 500 轮、学生 100 轮及 123,000 次更新。
                "Formal FastVoiceGrad report must use teacher epoch 500 and "
                # 此历史正式配置只接受教师 500 轮、学生 100 轮及 123,000 次更新。
                "student epoch/global_step 100/123000"
            )
        # 核对单步学生推理及起点 950，与既定正式协议一致。
        if student_steps != 1 or student_start_step != 950:
            # 报错原因：核对单步学生推理及起点 950，与既定正式协议一致。
            raise ValueError("Formal FastVoiceGrad report must declare K=1 and S_K=950")
        # 从已有记录取得学生权重路径。
        student_checkpoint = str(inference_metadata.get("student_checkpoint", "")).strip()
        # 设置学生权重的字节指纹。
        student_checkpoint_sha256 = str(
            # 读取学生权重字节指纹；缺失时保留为空并让身份检查失败。
            inference_metadata.get("student_checkpoint_sha256", "")
        # 去除身份字符串外围空白，避免路径或哈希意外带换行。
        ).strip()
        # 组合检查下面各项：必须明确保存学生权重路径。
        if (
            # 必须明确保存学生权重路径。
            not student_checkpoint
            # 权重指纹必须与登记的历史学生身份一致。
            or student_checkpoint_sha256 != EXPECTED_FASTVOICEGRAD_STUDENT_SHA256
        ):
            # 报错原因：权重指纹必须与登记的历史学生身份一致。
            raise ValueError("FastVoiceGrad report is missing the student checkpoint identity")
        # 从已有记录取得学生训练约定的指纹。
        student_run_contract_sha256 = inference_metadata.get(
            # 字段含义：学生训练约定的指纹。
            "student_run_contract_sha256"
        )
        # 拒绝使用其他训练目标或参数约定产生的学生权重。
        if student_run_contract_sha256 != EXPECTED_FASTVOICEGRAD_RUN_CONTRACT_SHA256:
            # 报错原因：拒绝使用其他训练目标或参数约定产生的学生权重。
            raise ValueError("FastVoiceGrad report has the wrong student run contract")
        # 组合检查下面各项：学生正式结果必须采用本次固定自训声码器。
        if (
            # 学生正式结果必须采用本次固定自训声码器。
            inference_metadata.get("vocoder_kind") != "custom"
            # 声码器必须是完整训练到 250 万步的归档。
            or int(inference_metadata.get("vocoder_global_step", -1)) != 2_500_000
        ):
            # 报错原因：声码器必须是完整训练到 250 万步的归档。
            raise ValueError("FastVoiceGrad report must use the 2,500,000-step custom vocoder")
        # 把已核对的字段组成返回记录，供后续身份检查或报告保存。
        return {
            # 记录教师或学生系统标识。
            "system_id": system_id,
            # 记录报告中的模型名称。
            "system_label": str(
                # 读取推理阶段明确保存的模型标签，旧报告缺失时使用随后指定的兼容值。
                inference_metadata.get(
                    # 兼容标签明确显示单步学生及 100 轮身份。
                    "system_label", "FastVoiceGrad (K=1) epoch 100"
                )
            ),
            # 记录教师或学生角色。
            "model_role": model_role,
            # 记录教师模型训练轮数。
            "teacher_epoch": teacher_epoch,
            # 记录学生模型训练轮数。
            "student_epoch": student_epoch,
            # 记录学生优化器累计更新次数。
            "student_global_step": student_global_step,
            # 记录学生推理去噪次数。
            "student_steps": student_steps,
            # 记录学生推理起始噪声步。
            "student_start_step": student_start_step,
            # 记录学生权重路径。
            "student_checkpoint": student_checkpoint,
            # 记录学生权重的字节指纹。
            "student_checkpoint_sha256": student_checkpoint_sha256,
            # 记录学生训练约定的指纹。
            "student_run_contract_sha256": student_run_contract_sha256,
            # 记录选用的论文模型行。
            "paper_reference_row": "FastVoiceGrad",
            # 记录论文公布的参考数值，不参与评分。
            "paper_reference": PAPER_FASTVOICEGRAD_REFERENCE,
        }
    # 未知系统名称会直接报错，不默认套用论文某一行。
    raise ValueError(f"Unsupported inference system_id: {system_id!r}")


# 读取运行路径、设备和诊断选项，默认值保持原实验约定。
def parse_args() -> argparse.Namespace:
    # 读取运行路径、设备和诊断选项，默认值保持原实验约定。
    """定义命令行参数；默认值就是本项目 epoch 500 的正式论文协议目录。"""

    # 设置命令行参数解析器。
    parser = argparse.ArgumentParser(description=__doc__)
    # 指定运行与数据所在的租用服务器项目根目录。
    parser.add_argument("--project-root", type=Path, default=SERVER_PROJECT_ROOT)
    # 指定已经生成波形、配对清单和推理记录的目录。
    parser.add_argument("--inference-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    # 指定四项逐条指标和总报告的保存目录。
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # 指定固定 UTMOS 模型与官方评分源码目录。
    parser.add_argument("--utmos-dir", type=Path, default=DEFAULT_UTMOS_DIR)
    # 指定 ECAPA 声纹评价模型目录。
    parser.add_argument("--ecapa-dir", type=Path, default=DEFAULT_ECAPA_DIR)
    # 定义一个显式评价选项，类型、默认值及限制由以下参数说明。
    parser.add_argument(
        # 选择全部四项客观指标或仅重算其中一项。
        "--metric",
        # 只允许这四项已实现指标及全量选项。
        choices=["all", "utmos", "dnsmos", "cer", "sva"],
        # 默认依次执行全部客观指标。
        default="all",
        # 命令行帮助说明客观指标名称及其使用限制。
        help="all 会依次计算四项；也能单独重跑某一项。",
    )
    # 默认每批四条同源生成语音，兼顾评价模型显存。
    parser.add_argument("--batch-size", type=int, default=4)
    # 定义一个显式评价选项，类型、默认值及限制由以下参数说明。
    parser.add_argument(
        # 单独控制 DNSMOS CPU 并行进程数。
        "--dnsmos-workers",
        # 要求数量、编号或进程预算以整数提供。
        type=int,
        # DNSMOS 默认三个 CPU 工作进程。
        default=3,
        # 命令行帮助说明DNSMOS 并行进程数及其使用限制。
        help="DNSMOS CPU 并行进程数；默认 3，机器较忙或内存不足时可调小。",
    )
    # 选择当前租用服务器上的 GPU 或 CPU 评价设备。
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 定义一个显式评价选项，类型、默认值及限制由以下参数说明。
    parser.add_argument(
        # 指定本次连续配对切片的起点。
        "--start",
        # 要求数量、编号或进程预算以整数提供。
        type=int,
        # 默认从配对清单第一条开始。
        default=0,
        # 这是保留的历史帮助文本；当前操作约束要求所有分片计算仍只在租用服务器进行。
        help="从 pair_id 的哪个位置开始；默认 0。仅用于把 DNSMOS 分给多台 CPU 服务器。",
    )
    # 定义一个显式评价选项，类型、默认值及限制由以下参数说明。
    parser.add_argument(
        # 限制小样本诊断条数；正式完整评价不传此项。
        "--limit",
        # 要求数量、编号或进程预算以整数提供。
        type=int,
        # 默认不截断配对数量。
        default=None,
        # 命令行帮助说明诊断时最多处理的配对数及其使用限制。
        help="只用于冒烟测试；正式报告必须省略该参数，才能覆盖 8,100 条。",
    )
    # 定义一个显式评价选项，类型、默认值及限制由以下参数说明。
    parser.add_argument(
        # 只有显式打开，才允许少于正式 8,100 条的受控诊断清单。
        "--diagnostic-allow-nonpaper-pairs",
        # 未传参数时关闭，出现该开关时才设为真。
        action="store_true",
        # 帮助文本明确划分诊断与正式结果身份。
        help=(
            # 帮助文本明确划分诊断与正式结果身份。
            "只用于受控诊断实验：允许 pairs.csv 少于 8,100 条。默认关闭，"
            # 帮助文本明确划分诊断与正式结果身份。
            "因此正式论文协议仍会严格要求 pair_id 0..8099。"
        ),
    )
    # 返回已经校验参数格式的运行选项。
    return parser.parse_args()


# 按二进制文件内容计算 SHA-256，供后续检查是否混入其他版本。
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # 按二进制文件内容计算 SHA-256，供后续检查是否混入其他版本。
    """分块计算 SHA-256；超大 UTMOS/ASR 权重也不会一次塞满内存。"""

    # 开始累计该文件的 SHA-256，后续分块加入所有字节。
    digest = hashlib.sha256()
    # 按二进制读取，文件编码或换行不会被自动转换。
    with path.open("rb") as handle:
        # 持续读入后续数据块，直到下面的空块条件终止循环。
        while True:
            # 按指定块大小读取权重或清单，避免大文件占用过多内存。
            block = handle.read(chunk_size)
            # 读到空块说明文件已经结束。
            if not block:
                # 结束已到末尾的文件分块读取。
                break
            # 把当前二进制块加入同一个文件指纹。
            digest.update(block)
    # 返回便于写入 JSON 的十六进制文件指纹。
    return digest.hexdigest()


# 先写临时 JSON 再原子替换，减少中断时留下半份报告的风险。
def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    # 先写临时 JSON 再原子替换，减少中断时留下半份报告的风险。
    """Atomically bind resumable metric CSVs to one exact inference run."""

    # 创建结果文件的父目录；已有目录可继续使用。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 结果先写同目录临时文件，完整后再发布正式 JSON。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 先完整写入临时文件，避免正式路径出现半写入记录。
    temporary.write_text(
        # 用 UTF-8 保留中文字段，并缩进输出便于人工核对。
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 把写完的临时文件原子替换到正式结果路径。
    os.replace(temporary, path)


# 记录实际安装的评价依赖版本，缺失元数据时明确报告不可得。
def package_version(distribution_name: str) -> str:
    # 记录当前实际安装版本；缺少发行版信息时明确返回不可得。
    """读取已安装包版本；找不到时返回清楚的字符串，而不是让报告生成失败。"""

    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 返回实际安装版本，供同环境复算时核对。
        return importlib.metadata.version(distribution_name)
    # 某个包没有发行版元信息时，报告显式说明版本不可得。
    except importlib.metadata.PackageNotFoundError:
        # 保留版本缺失状态，不能把未知版本编造为已知版本。
        return "not-installed-or-version-unavailable"


# 读取源语音—目标参考清单，正式模式检查 8,100 个连续配对编号。
def load_pairs(
    # 输入参数：固定源语音和目标参考的配对清单。
    pairs_csv: Path,
    # 输入参数：本次配对切片起点。
    start: int,
    # 输入参数：诊断时最多处理的配对数。
    limit: Optional[int],
    # 默认启用严格正式配对清单检查。
    require_formal_manifest: bool = True,
# 函数返回按清单顺序选中的配对字典列表，字符串保留 CSV 原始字段。
) -> List[Dict[str, str]]:
    # 读取源语音—目标参考清单，正式模式检查 8,100 个连续配对编号。
    """读取并检查推理清单；默认正式模式必须恰好有 pair_id 0~8099。"""

    # 配对清单必须存在，否则无法确认生成语音对应哪句话和说话人。
    if not pairs_csv.is_file():
        # 报错原因：配对清单必须存在，否则无法确认生成语音对应哪句话和说话人。
        raise FileNotFoundError(f"pairs.csv not found: {pairs_csv}")
    # 按 UTF-8 CSV 读取固定配对清单。
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        # 按表头名称读取每条记录，避免列顺序改变造成字段混淆。
        rows = list(csv.DictReader(handle))

    # 列出评价必需的身份、文本、参考波形和生成波形字段。
    required = {
        # 字段含义：转换配对编号。
        "pair_id",
        # 字段含义：源语音编号。
        "source_utt",
        # 字段含义：源语音对应文本，用于 CER 参考。
        "source_text",
        # 字段含义：目标说话人参考语音编号。
        "reference_utt",
        # 字段含义：目标说话人参考波形。
        "reference_wav",
        # 字段含义：待评估的生成波形。
        "output_wav",
    }
    # 空清单或缺少必要列会中止评价。
    if not rows or not required.issubset(rows[0]):
        # 报错原因：空清单或缺少必要列会中止评价。
        raise ValueError(f"pairs.csv is empty or missing columns {sorted(required)}")
    # 正式模式必须覆盖 8,100 条，小样本仅能走明确的诊断模式。
    if require_formal_manifest and len(rows) != 8100:
        # 报错原因：正式模式必须覆盖 8,100 条，小样本仅能走明确的诊断模式。
        raise ValueError(f"Formal pair manifest must contain 8,100 rows, got {len(rows)}")

    # 把每条配对编号转成整数以检查顺序和完整性。
    ids = [int(row["pair_id"]) for row in rows]
    # 编号必须从 0 连续递增，同时排除重复、遗漏和乱序。
    if ids != list(range(len(rows))):
        # 报错原因：编号必须从 0 连续递增，同时排除重复、遗漏和乱序。
        raise ValueError(
            # 编号必须从 0 连续递增，同时排除重复、遗漏和乱序。
            f"pair_id must be the continuous ordered range 0..{len(rows) - 1}"
        )

    # 诊断起点必须落在真实配对范围内。
    if not 0 <= start < len(rows):
        # 报错原因：诊断起点必须落在真实配对范围内。
        raise ValueError(f"start must be in [0, {len(rows) - 1}]")
    # 未指定诊断上限时，处理到清单末尾。
    if limit is None:
        # 把切片终点设为清单长度，从而包含所有剩余配对。
        end = len(rows)
    # 此分支：指定数量必须为正，且起点加数量不能越过清单末尾。
    else:
        # 指定数量必须为正，且起点加数量不能越过清单末尾。
        if limit < 1 or start + limit > len(rows):
            # 报错原因：指定数量必须为正，且起点加数量不能越过清单末尾。
            raise ValueError(
                # 指定数量必须为正，且起点加数量不能越过清单末尾。
                f"limit must be positive and start+limit must be <= {len(rows)}"
            )
        # 根据明确的诊断数量计算本次切片终点。
        end = start + limit
    # 只返回本次声明要处理的配对区间。
    return rows[start:end]


# 检查生成波形格式，并在提供指纹表时逐文件核对字节内容。
def validate_generated_wavs(
    # 输入参数：按 pair_id 排列的评估条目。
    rows: Sequence[Mapping[str, str]],
    # 可选的逐配对波形指纹；学生正式评价必须提供。
    expected_sha256: Optional[Mapping[str, str]] = None,
# 此函数只做文件检查，没有计算返回值；失败通过异常阻止评价。
) -> None:
    # 检查生成波形格式，并在提供指纹表时逐文件核对字节内容。
    """Confirm audio headers and, for the student, every waveform byte identity."""

    # 逐条核对生成文件，同时显示检查进度。
    for row in tqdm(rows, desc="Validate generated WAV files"):
        # 设置待处理文件路径。
        path = Path(row["output_wav"])
        # 缺失生成波形或指标文件时停止，不用不完整数据继续。
        if not path.is_file():
            # 报错原因：缺失生成波形或指标文件时停止，不用不完整数据继续。
            raise FileNotFoundError(f"Generated WAV missing for pair {row['pair_id']}: {path}")
        # 先读取 WAV 文件头，以检查采样率、声道和样本数。
        info = sf.info(str(path))
        # 要求非空的 22.05 kHz 单声道生成语音。
        if info.samplerate != 22050 or info.channels != 1 or info.frames <= 0:
            # 报错原因：要求非空的 22.05 kHz 单声道生成语音。
            raise ValueError(
                # 错误信息指出哪个转换配对的 WAV 文件头不符合要求。
                f"Invalid generated WAV for pair {row['pair_id']}: "
                # 要求非空的 22.05 kHz 单声道生成语音。
                f"sr={info.samplerate}, channels={info.channels}, frames={info.frames}"
            )
        # 学生正式结果还需要核对每份波形的完整字节指纹。
        if expected_sha256 is not None:
            # 设置转换配对编号。
            pair_id = row["pair_id"]
            # 重新计算当前波形文件的实际指纹。
            actual_sha256 = sha256_file(path)
            # 若波形被替换或损坏，就停止使用旧推理记录认证。
            if actual_sha256 != expected_sha256[pair_id]:
                # 报错原因：若波形被替换或损坏，就停止使用旧推理记录认证。
                raise RuntimeError(
                    # 错误信息同时给出配对编号和被更换或损坏的波形路径。
                    f"Generated WAV content hash mismatch for pair {pair_id}: {path}"
                )


# 核对每个生成波形的指纹及整张指纹表，绑定到确切推理结果。
def load_generated_wav_hash_manifest(
    # 推理阶段已经保存的模型、数据和波形身份信息。
    inference_metadata: Mapping[str, object],
    # 输入参数：按 pair_id 排列的评估条目。
    rows: Sequence[Mapping[str, str]],
# 返回配对编号到波形 SHA-256 的完整映射。
) -> Dict[str, str]:
    # 核对每个生成波形的指纹及整张指纹表，绑定到确切推理结果。
    """Validate the FastVoiceGrad report's complete pair-id to WAV-hash mapping."""

    # 读取推理阶段保存的逐配对波形指纹表。
    value = inference_metadata.get("generated_wav_sha256")
    # 必须有完整键值形式的波形指纹表，不能只依赖文件数量。
    if not isinstance(value, Mapping):
        # 报错原因：必须有完整键值形式的波形指纹表，不能只依赖文件数量。
        raise RuntimeError("FastVoiceGrad report has no generated_wav_sha256 mapping")
    # 从当前清单得到应当出现的全部波形编号。
    expected_ids = [row["pair_id"] for row in rows]
    # 指纹表必须恰好覆盖当前全部配对，不能缺失或多出其他实验语音。
    if set(value) != set(expected_ids):
        # 报错原因：指纹表必须恰好覆盖当前全部配对，不能缺失或多出其他实验语音。
        raise RuntimeError("FastVoiceGrad waveform hash manifest does not cover all pairs")
    # 限定 SHA-256 使用小写十六进制字符。
    hexadecimal = set("0123456789abcdef")
    # 收集经过格式核对的逐配对波形指纹。
    result: Dict[str, str] = {}
    # 按清单顺序检查每个配对的波形指纹。
    for pair_id in expected_ids:
        # 取出当前配对的预期波形指纹。
        digest = value.get(pair_id)
        # 组合检查下面各项：指纹必须是文本，不能接受其他 JSON 值。
        if (
            # 指纹必须是文本，不能接受其他 JSON 值。
            not isinstance(digest, str)
            # SHA-256 十六进制文本应恰好有 64 个字符。
            or len(digest) != 64
            # 排除非十六进制字符或大小写不规范的指纹。
            or not set(digest).issubset(hexadecimal)
        ):
            # 报错原因：排除非十六进制字符或大小写不规范的指纹。
            raise RuntimeError(f"Invalid waveform SHA-256 for pair {pair_id}")
        # 把通过格式检查的波形指纹写入结果映射。
        result[pair_id] = digest
    # 以固定字段排序和分隔符编码指纹表，保证整体哈希可重复。
    encoded = json.dumps(
        # 指纹表按固定键顺序与分隔符编码，避免格式差异影响整体身份。
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    # 将规范化 JSON 文本转成明确 UTF-8 字节后再哈希。
    ).encode("utf-8")
    # 对整张波形指纹表计算第二层指纹，检查记录内部一致性。
    collection_sha256 = hashlib.sha256(encoded).hexdigest()
    # 组合检查下面各项：读取推理记录中保存的波形集合整体指纹。
    if (
        # 读取推理记录中保存的波形集合整体指纹。
        inference_metadata.get("generated_wav_collection_sha256")
        # 与当前逐文件指纹表重新计算的整体指纹比较。
        != collection_sha256
    ):
        # 报错原因：与当前逐文件指纹表重新计算的整体指纹比较。
        raise RuntimeError("FastVoiceGrad waveform collection hash is inconsistent")
    # 返回本函数已经构造或校验的结果，供调用方继续处理。
    return result


# 把相邻且同源的语音分成小批，保留配对顺序并避免额外补零。
def grouped_chunks(
    # 输入有序配对列表与每个同源小批次上限。
    rows: Sequence[Mapping[str, str]], batch_size: int
# 返回按顺序逐批产生同源语音列表的迭代器。
) -> Iterable[List[Mapping[str, str]]]:
    # 把相邻且同源的语音分成小批，保留配对顺序并避免额外补零。
    """Chunk contiguous same-source runs without changing the selected-ID order."""

    # 暂存一段连续同源的转换配对。
    run: List[Mapping[str, str]] = []
    # 记录当前连续段的源语音编号。
    run_source: Optional[str] = None
    # 依照清单原有顺序处理配对，保证续算仍可按前缀验证。
    for row in rows:
        # 设置源语音编号。
        source_utt = row["source_utt"]
        # 源语音切换时先输出上一段，同一批不混入不同长度来源。
        if run and source_utt != run_source:
            # 按指定批大小切分当前同源语音段。
            for start in range(0, len(run), batch_size):
                # 产生一个保持原始顺序的同源小批次。
                yield run[start : start + batch_size]
            # 清空已经输出的同源段，开始收集下一个源语音。
            run = []
        # 把当前源编号设为这一连续段的标识。
        run_source = source_utt
        # 把当前配对加入同源段。
        run.append(row)

    # 按指定批大小切分当前同源语音段。
    for start in range(0, len(run), batch_size):
        # 产生一个保持原始顺序的同源小批次。
        yield run[start : start + batch_size]


# 读取单声道波形，拒绝采样率错误、空音频和非有限数。
def load_audio(path: Path, expected_sample_rate: int = 22050) -> torch.Tensor:
    # 读取单声道波形，拒绝采样率错误、空音频和非有限数。
    """读取一条单声道 float32 波形并检查 NaN/Inf，返回形状 [time]。"""

    # 按浮点波形读取文件，单声道保留为一维数组。
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    # 采样率不符直接报错，避免评价时暗中接受错误预处理。
    if sample_rate != expected_sample_rate:
        # 报错原因：采样率不符直接报错，避免评价时暗中接受错误预处理。
        raise ValueError(f"Expected {expected_sample_rate} Hz at {path}, got {sample_rate}")
    # 只接受有采样点的一维单声道波形。
    if audio.ndim != 1 or audio.size == 0:
        # 报错原因：只接受有采样点的一维单声道波形。
        raise ValueError(f"Expected non-empty mono WAV at {path}, got shape={audio.shape}")
    # 拒绝包含 NaN 或无穷大的音频。
    if not np.isfinite(audio).all():
        # 报错原因：拒绝包含 NaN 或无穷大的音频。
        raise ValueError(f"NaN/Inf waveform found: {path}")
    # 整理为连续内存数组，再转为无额外拷贝的 PyTorch 张量。
    return torch.from_numpy(np.ascontiguousarray(audio))


# 读取同源生成语音并要求长度一致，再堆成批次张量。
def load_equal_length_generated_batch(rows: Sequence[Mapping[str, str]]) -> torch.Tensor:
    # 读取同源生成语音并要求长度一致，再堆成批次张量。
    """读取一个同-source 小批次，严格要求生成长度完全相同后 stack 成 [B,time]。"""

    # 读取这一批每条生成波形。
    waves = [load_audio(Path(row["output_wav"])) for row in rows]
    # 收集不同波形长度，检查能否直接堆叠。
    lengths = {int(wave.numel()) for wave in waves}
    # 同源批次若有长度差异就报错，不用补零掩盖生成问题。
    if len(lengths) != 1:
        # 报错原因：同源批次若有长度差异就报错，不用补零掩盖生成问题。
        raise ValueError(f"Same-source generated WAV lengths differ: {sorted(lengths)}")
    # 沿批次轴堆成 [批大小, 采样点数]。
    return torch.stack(waves, dim=0)


# 检查 CSV 单元格是规范的非负整数，拒绝缺值和不合法字符。
def _strict_nonnegative_int(value: Optional[str], label: str) -> int:
    # 去除 CSV 单元格外围空白，同时保留缺值为空的事实。
    token = "" if value is None else value.strip()
    # 只接受 0 或不带前导零的非负整数文本。
    if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
        # 报错原因：只接受 0 或不带前导零的非负整数文本。
        raise ValueError(f"{label} must be a canonical non-negative integer, got {value!r}")
    # 把格式已核对的整数文本转成计数。
    return int(token)


# 进一步要求计数字段大于零，防止 CER 分母无效。
def _strict_positive_int(value: Optional[str], label: str) -> int:
    # 先做统一的整数格式检查，再增加正数约束。
    result = _strict_nonnegative_int(value, label)
    # 分母或必须为正的计数不能等于零。
    if result <= 0:
        # 报错原因：分母或必须为正的计数不能等于零。
        raise ValueError(f"{label} must be positive, got {value!r}")
    # 返回本函数已经构造或校验的结果，供调用方继续处理。
    return result


# 把数值字段转成浮点数，拒绝空字符串、NaN 和无穷大。
def _strict_finite_float(value: Optional[str], label: str) -> float:
    # 去除 CSV 单元格外围空白，同时保留缺值为空的事实。
    token = "" if value is None else value.strip()
    # 空数值单元格不能参与指标汇总。
    if not token:
        # 报错原因：空数值单元格不能参与指标汇总。
        raise ValueError(f"{label} must not be empty")
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 将经过空值检查的文本解析成浮点数。
        result = float(token)
    # 数值解析失败时补充字段位置，方便定位损坏的 CSV。
    except ValueError as error:
        # 报错原因：数值解析失败时补充字段位置，方便定位损坏的 CSV。
        raise ValueError(f"{label} must be numeric, got {value!r}") from error
    # 排除 NaN 和正负无穷大，避免它们污染平均分。
    if not math.isfinite(result):
        # 报错原因：排除 NaN 和正负无穷大，避免它们污染平均分。
        raise ValueError(f"{label} must be finite, got {value!r}")
    # 返回本函数已经构造或校验的结果，供调用方继续处理。
    return result


# 核对单行分数及其计算依据，防止损坏的续算 CSV 被跳过。
def validate_metric_row(
    # 本次待检查的指标名称，用来选择分数约束。
    metric_name: str,
    # 输入参数：当前语音条目。
    row: Mapping[str, str],
    # 星号之后的参数必须写出参数名，防止位置传参把路径、字段或轮次放错。
    *,
    # 输入参数：待处理文件路径。
    path: Path,
    # 当前 CSV 真实行号，错误信息可直接定位。
    line_number: int,
# 返回已经核对内容的整数配对编号，续算才可将其记为完成。
) -> int:
    # 核对单行分数及其计算依据，防止损坏的续算 CSV 被跳过。
    """Validate one resumable row completely before treating its pair as finished."""

    # 为错误信息附上文件和具体行号。
    prefix = f"{path}:{line_number}"
    # 从已有记录取得转换配对编号。
    pair_id = _strict_nonnegative_int(row.get("pair_id"), f"{prefix} pair_id")
    # UTMOS 分支检查预测分数是否为有限数。
    if metric_name == "utmos":
        # 核对 UTMOS 单元格存在且为有限数。
        _strict_finite_float(row.get("utmos"), f"{prefix} utmos")
    # DNSMOS 分支检查四列输出，不能只看主指标列。
    elif metric_name == "dnsmos":
        # 跳过编号列，依次验证 P.808、语音、背景与总体分数。
        for field in DNSMOS_FIELDS[1:]:
            # 核对当前 DNSMOS 输出列为有限数。
            _strict_finite_float(row.get(field), f"{prefix} {field}")
    # CER 续算记录要能从保存的两份文本完整重算。
    elif metric_name == "cer":
        # 读取已清洗的正确源文本。
        reference = row.get("reference_text_normalized", "")
        # 读取已清洗的 ASR 识别文本。
        hypothesis = row.get("hypothesis_text_normalized", "")
        # 参考文本不能为空，而且必须已满足同一套清洗规则。
        if not reference or normalize_cer_text(reference) != reference:
            # 报错原因：参考文本不能为空，而且必须已满足同一套清洗规则。
            raise ValueError(f"{prefix} has invalid normalized CER reference {reference!r}")
        # 识别文本也必须采用相同的大小写、标点和空格规则。
        if normalize_cer_text(hypothesis) != hypothesis:
            # 报错原因：识别文本也必须采用相同的大小写、标点和空格规则。
            raise ValueError(f"{prefix} has invalid normalized CER hypothesis {hypothesis!r}")
        # 字符错误次数必须为非负整数。
        errors = _strict_nonnegative_int(row.get("char_errors"), f"{prefix} char_errors")
        # 设置参考文本字符数。
        reference_chars = _strict_positive_int(
            # 取出并标识参考字符数，要求其为正分母。
            row.get("reference_chars"), f"{prefix} reference_chars"
        )
        # 从已有记录取得字符错误率百分数，越低越好。
        cer_percent = _strict_finite_float(row.get("cer_percent"), f"{prefix} cer_percent")
        # 检查 CSV 保存的分母是否确为参考文本长度。
        if reference_chars != len(reference):
            # 报错原因：检查 CSV 保存的分母是否确为参考文本长度。
            raise ValueError(
                # 检查 CSV 保存的分母是否确为参考文本长度。
                f"{prefix} reference_chars={reference_chars} but len(reference)={len(reference)}"
            )
        # 从保存文本重新计算字符编辑距离，验证旧结果没有损坏。
        recomputed_errors = levenshtein_distance(reference, hypothesis)
        # 若保存错误数与重新计算不一致，禁止跳过这一条。
        if errors != recomputed_errors:
            # 报错原因：若保存错误数与重新计算不一致，禁止跳过这一条。
            raise ValueError(
                # 若保存错误数与重新计算不一致，禁止跳过这一条。
                f"{prefix} char_errors={errors} but recomputed={recomputed_errors}"
            )
        # 按同一错误数和字符数重算该句 CER 百分数。
        recomputed_percent = 100.0 * errors / reference_chars
        # 只容许浮点书写级别的差异，不能接受计算口径变更。
        if not math.isclose(cer_percent, recomputed_percent, rel_tol=1e-9, abs_tol=1e-9):
            # 报错原因：只容许浮点书写级别的差异，不能接受计算口径变更。
            raise ValueError(
                # 只容许浮点书写级别的差异，不能接受计算口径变更。
                f"{prefix} cer_percent={cer_percent} but recomputed={recomputed_percent}"
            )
    # SVA 同时检查连续分数、阈值和最终通过标记。
    elif metric_name == "sva":
        # 从旧 SVA 行读取并核对有限的声纹余弦分数。
        score = _strict_finite_float(row.get("cosine_score"), f"{prefix} cosine_score")
        # 从已有记录取得说话人验证通过阈值。
        threshold = _strict_finite_float(row.get("threshold"), f"{prefix} threshold")
        # 从已有记录取得说话人验证是否通过，1 表示通过。
        verified = _strict_nonnegative_int(row.get("verified"), f"{prefix} verified")
        # 说话人验证的结果只能是未通过 0 或通过 1。
        if verified not in {0, 1}:
            # 报错原因：说话人验证的结果只能是未通过 0 或通过 1。
            raise ValueError(f"{prefix} verified must be 0 or 1, got {verified}")
        # 所有条目必须采用固定 0.25 阈值，不得混入其他阈值结果。
        if not math.isclose(threshold, SVA_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
            # 报错原因：所有条目必须采用固定 0.25 阈值，不得混入其他阈值结果。
            raise ValueError(
                # 所有条目必须采用固定 0.25 阈值，不得混入其他阈值结果。
                f"{prefix} threshold={threshold} does not equal {SVA_THRESHOLD}"
            )
        # 核对是否通过严格按照相似度大于阈值计算。
        if verified != int(score > threshold):
            # 报错原因：核对是否通过严格按照相似度大于阈值计算。
            raise ValueError(
                # 核对是否通过严格按照相似度大于阈值计算。
                f"{prefix} verified={verified} is inconsistent with score={score}"
            )
    # 此分支：返回已完整验证的配对编号，续算才可将其标记为完成。
    else:
        # 报错原因：核对是否通过严格按照相似度大于阈值计算。
        raise ValueError(f"Unknown metric CSV kind: {metric_name!r}")
    # 返回已完整验证的配对编号，续算才可将其标记为完成。
    return pair_id


# 续算只接受当前配对序列的有效前缀，不能拼接不同实验结果。
def validate_existing_metric_csv(
    # 输入参数：待处理文件路径。
    path: Path,
    # 输入参数：本次选中的评估条目。
    selected_rows: Sequence[Mapping[str, str]],
    # 本次待检查的指标名称，用来选择分数约束。
    metric_name: str,
    # 该指标 CSV 的固定字段名与顺序。
    fieldnames: Sequence[str],
# 返回通过格式、数值和前缀顺序核对的旧分数行。
) -> List[Dict[str, str]]:
    # 续算只接受当前配对序列的有效前缀，不能拼接不同实验结果。
    """Fail closed unless an existing CSV is exactly a valid selected-ID prefix."""

    # 尚无结果或只有空文件时，从第一条开始计算。
    if not path.is_file() or path.stat().st_size == 0:
        # 返回空的已完成列表，不跳过任何条目。
        return []
    # 以固定编码和 CSV 换行规则读取已有记录。
    with path.open("r", encoding="utf-8", newline="") as handle:
        # 按表头字段读取每个单元格。
        reader = csv.DictReader(handle)
        # 已有表头必须与当前指标精确匹配，包括列顺序。
        if reader.fieldnames != list(fieldnames):
            # 报错原因：已有表头必须与当前指标精确匹配，包括列顺序。
            raise ValueError(
                # 错误信息指出列名或顺序不符的指标文件。
                f"Metric CSV header mismatch at {path}: "
                # 已有表头必须与当前指标精确匹配，包括列顺序。
                f"got {reader.fieldnames!r}, expected {list(fieldnames)!r}"
            )
        # 收集通过基本列结构检查的旧分数记录。
        existing_rows = []
        # 建立该指标允许出现的列名集合。
        expected_columns = set(fieldnames)
        # 从 CSV 第二行开始逐条检查，并保留真实行号。
        for line_number, row in enumerate(reader, start=2):
            # 拒绝多列、少列和缺单元格的行。
            if set(row) != expected_columns or any(value is None for value in row.values()):
                # 报错原因：拒绝多列、少列和缺单元格的行。
                raise ValueError(
                    # 错误信息定位列数不完整的 CSV 具体行。
                    f"Malformed metric CSV row at {path}:{line_number}; "
                    # 拒绝多列、少列和缺单元格的行。
                    f"expected exactly {len(fieldnames)} columns"
                )
            # 保留结构完整的旧行，接着验证内容与编号。
            existing_rows.append(row)

    # 取得当前实际要计算的编号顺序。
    selected_ids = [int(row["pair_id"]) for row in selected_rows]
    # 选中清单不能有重复编号，否则续算可能误跳过语音。
    if len(selected_ids) != len(set(selected_ids)):
        # 报错原因：选中清单不能有重复编号，否则续算可能误跳过语音。
        raise ValueError("Selected metric rows contain duplicate pair_id values")
    # 旧结果数量不能超过本次选择范围。
    if len(existing_rows) > len(selected_ids):
        # 报错原因：旧结果数量不能超过本次选择范围。
        raise ValueError(
            # 同时报告旧结果条数和当前选中条数，说明两者不能续接。
            f"Metric CSV has {len(existing_rows)} rows but selection has only {len(selected_ids)}: {path}"
        )
    # 逐条校验旧行内容，并形成其实际编号序列。
    existing_ids = [
        # 逐行核对数值与可重算依据，然后取得有效编号。
        validate_metric_row(metric_name, row, path=path, line_number=line_number)
        # 保留 CSV 第二行起的真实数据行号。
        for line_number, row in enumerate(existing_rows, start=2)
    ]
    # 旧结果应当是当前清单前面已经连续完成的那一段。
    expected_prefix = selected_ids[: len(existing_ids)]
    # 不连续、乱序或来自另一切片的结果都不能继续追加。
    if existing_ids != expected_prefix:
        # 报错原因：不连续、乱序或来自另一切片的结果都不能继续追加。
        raise ValueError(
            # 错误信息说明续算必须满足同序连续前缀，不能任意拼分数。
            "Resumable metric CSV IDs must be the exact ordered prefix of the active selection: "
            # 列出出错文件和前 20 个实际/预期编号，方便定位乱序或错实验。
            f"path={path}, actual={existing_ids[:20]}, expected={expected_prefix[:20]}, "
            # 不连续、乱序或来自另一切片的结果都不能继续追加。
            f"actual_rows={len(existing_ids)}"
        )
    # 返回全部通过行内容和顺序核对的旧记录。
    return existing_rows


# 追加写逐条分数；新文件补表头，已有结果保留。
def open_result_writer(path: Path, fieldnames: Sequence[str]):
    # 追加写逐条分数；新文件补表头，已有结果保留。
    """以追加模式打开 CSV；新文件先写一次表头，旧文件继续追加而不覆盖结果。"""

    # 创建结果文件的父目录；已有目录可继续使用。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 判断是否需要为新结果文件写入表头。
    needs_header = not path.is_file() or path.stat().st_size == 0
    # 追加并采用行缓冲，尽早落盘每条已完成分数。
    handle = path.open("a", encoding="utf-8", newline="", buffering=1)
    # 固定写出列名和顺序，以便后续严格续算。
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
    # 仅在新建或空文件中写表头，避免中途重复表头。
    if needs_header:
        # 写出固定列名，让后续续算和汇总能核对格式。
        writer.writeheader()
    # 同时返回文件句柄和 CSV 写入器，调用方负责关闭。
    return handle, writer


# 过滤已验证完成的配对，续算只处理缺少的条目。
def pending_rows(
    # 输入有序配对清单及已经核对完成的编号集合。
    rows: Sequence[Mapping[str, str]], completed: Set[int]
# 返回仍缺少有效分数的配对列表，原有顺序不变。
) -> List[Mapping[str, str]]:
    # 过滤已验证完成的配对，续算只处理缺少的条目。
    """去掉已经存在于相应指标 CSV 的 pair，保留尚未完成的项目。"""

    # 只留下尚未通过有效旧结果验证的配对。
    return [row for row in rows if int(row["pair_id"]) not in completed]


# 用固定 UTMOS 官方模型预测自然度；它是客观模型评分，没有邀请听众打分。
def evaluate_utmos(
    # 输入参数：按 pair_id 排列的评估条目。
    rows: Sequence[Mapping[str, str]],
    # 输入参数：逐条指标 CSV。
    output_csv: Path,
    # UTMOS 官方评分源码与已核对权重所在目录。
    utmos_dir: Path,
    # 输入参数：张量和模型使用的计算设备。
    device: torch.device,
    # 输入参数：每批处理的语音数。
    batch_size: int,
# 分数逐条写到 UTMOS CSV；函数本身不返回汇总值。
) -> None:
    # 用固定 UTMOS 官方模型预测自然度；它是客观模型评分，没有邀请听众打分。
    """用 UTMOS 官方 strong learner 逐条预测自然度，原始输出按官方规则 ×2+3。"""

    # 设置模型权重文件。
    checkpoint = utmos_dir / "epoch=3-step=7459.ckpt"
    # UTMOS 还依赖自己的 wav2vec 特征模型，与 CER 模型分别固定。
    wav2vec_small = utmos_dir / "wav2vec_small.pt"
    # 两份 UTMOS 必要权重必须齐全，缺少任一项就停止。
    if not checkpoint.is_file() or not wav2vec_small.is_file():
        # 报错原因：两份 UTMOS 必要权重必须齐全，缺少任一项就停止。
        raise FileNotFoundError(f"Incomplete UTMOS assets in {utmos_dir}")

    # 先核对已有分数属于当前清单且内容有效，再决定从哪里继续。
    existing_rows = validate_existing_metric_csv(
        # 用 UTMOS 的两列格式核对当前输出与选中配对。
        output_csv, rows, "utmos", UTMOS_FIELDS
    )
    # 把已经验证过的旧结果编号组成集合，用于快速跳过。
    completed = {int(row["pair_id"]) for row in existing_rows}
    # 选择本项指标还没有算完的语音。
    todo = pending_rows(rows, completed)
    # 全部条目已有有效分数时，无需再次加载评价模型。
    if not todo:
        # 显示 UTMOS 所有选中语音已有有效分数，本次无需重算。
        print(f"[UTMOS] already complete: {len(completed)} rows")
        # 当前指标已经完成时提前返回，不再次加载模型或覆盖结果。
        return

    # UTMOS 的 lightning_module.py 用相对路径打开 wav2vec_small.pt，并写了
    # `from model import ...`。因此加载时要暂时把 UTMOS 目录放到 sys.path 首位，
    # 并把当前工作目录切过去；模型构造完成后再恢复项目工作目录。
    original_cwd = Path.cwd()
    # 优先解析 UTMOS 内部相对模块名。
    sys.path.insert(0, str(utmos_dir))
    # PyTorch 2.6 把 torch.load 的默认值改成 weights_only=True；而 2022 年发布的
    # UTMOS Lightning checkpoint 除权重外还含 OmegaConf 配置。旧版 Lightning 1.5
    # 没有传 weights_only 参数，于是会被 PyTorch 2.6 拒绝。这里仅在加载已经核对
    # SHA-256 的官方 UTMOS checkpoint 时，把“调用者没有指定”的情况补成 False；
    # scorer 构造完马上恢复原 torch.load，不让这个兼容处理影响其他任何模型。
    original_torch_load = torch.load

    # 仅在加载已核对的旧 UTMOS 权重时补旧版 Lightning 的兼容参数。
    def trusted_utmos_torch_load(*load_args, **load_kwargs):
        # 只补充旧调用者未明确指定的加载选项。
        if "weights_only" not in load_kwargs:
            # 允许可信旧 UTMOS 检查点中的 Lightning 配置对象。
            load_kwargs["weights_only"] = False
        # 把调用交回原始 PyTorch 加载器，其他参数原样保留。
        return original_torch_load(*load_args, **load_kwargs)

    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 临时切到 UTMOS 资源目录，使旧代码找到相对路径权重。
        os.chdir(utmos_dir)
        # 导入固定 UTMOS 官方评分包装器。
        from score import Score  # type: ignore

        # 在模型构造的短暂范围内启用兼容加载。
        torch.load = trusted_utmos_torch_load  # type: ignore[assignment]
        # 构造官方自然度评分器，输出已经按其接口变换后的分数。
        scorer = Score(
            # 指定固定 UTMOS checkpoint。
            ckpt_path=str(checkpoint),
            # 告诉 UTMOS 输入波形的真实采样率，内部按其要求处理。
            input_sample_rate=22050,
            # 设置张量和模型使用的计算设备。
            device=str(device),
        )
    # 无论前面成功或失败，都执行资源恢复和关闭操作。
    finally:
        # 恢复 PyTorch 原始加载行为，避免影响后续模型。
        torch.load = original_torch_load  # type: ignore[assignment]
        # 恢复进入 UTMOS 前的项目工作目录。
        os.chdir(original_cwd)

    # 打开 UTMOS 追加输出；通过核对的旧行保留。
    handle, writer = open_result_writer(output_csv, UTMOS_FIELDS)
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 将待算语音按同源连续段分批。
        chunks = list(grouped_chunks(todo, batch_size))
        # 逐批计算 UTMOS 并显示进度。
        for chunk in tqdm(chunks, desc="UTMOS"):
            # 整理为 UTMOS 所需的 [批次, 单声道, 时间] 并送到指定设备。
            waveforms = load_equal_length_generated_batch(chunk).unsqueeze(1).to(device)
            # 调用官方评分接口，转成每条语音一个双精度分数。
            values = np.asarray(scorer.score(waveforms), dtype=np.float64).reshape(-1)
            # 每条输入必须对应一个有限分数。
            if values.shape[0] != len(chunk) or not np.isfinite(values).all():
                # 报错原因：每条输入必须对应一个有限分数。
                raise ValueError(f"Invalid UTMOS output shape/values: {values}")
            # 按原批次顺序把模型分数配回语音编号。
            for row, value in zip(chunk, values):
                # 立即写出该配对的 UTMOS，不只保存批次均值。
                writer.writerow({"pair_id": int(row["pair_id"]), "utmos": float(value)})
    # 无论前面成功或失败，都执行资源恢复和关闭操作。
    finally:
        # 关闭结果文件，确保缓冲中的指标记录写入磁盘。
        handle.close()
        # 移除 UTMOS 模型引用，让后续模型释放所需内存。
        del scorer
        # 若刚才在 GPU 上计算，则执行 GPU 专用的资源清理。
        if device.type == "cuda":
            # 释放未被张量占用的 CUDA 缓存，给后续评价模型留出显存。
            torch.cuda.empty_cache()


# 用 CPU 运行非个性化 DNSMOS，总体质量 mos_ovr 用作论文主对照。
def evaluate_dnsmos(
    # 输入参数：按 pair_id 排列的评估条目。
    rows: Sequence[Mapping[str, str]],
    # 输入参数：逐条指标 CSV。
    output_csv: Path,
    # 输入参数：每批处理的语音数。
    batch_size: int,
    # 输入参数：DNSMOS 的 CPU 子进程数。
    worker_count: int,
# 四路质量分数写到 CSV；函数本身不返回整体平均分。
) -> None:
    # 用 CPU 运行非个性化 DNSMOS，总体质量 mos_ovr 用作论文主对照。
    """用非 personalized Microsoft DNSMOS 计算四个输出，论文主数取 mos_ovr。"""

    # 先核对已有分数属于当前清单且内容有效，再决定从哪里继续。
    existing_rows = validate_existing_metric_csv(
        # 用 DNSMOS 的编号加四列分数格式核对续算记录。
        output_csv, rows, "dnsmos", DNSMOS_FIELDS
    )
    # 把已经验证过的旧结果编号组成集合，用于快速跳过。
    completed = {int(row["pair_id"]) for row in existing_rows}
    # 选择本项指标还没有算完的语音。
    todo = pending_rows(rows, completed)
    # 全部条目已有有效分数时，无需再次加载评价模型。
    if not todo:
        # 显示 DNSMOS 所有选中语音已有有效四路分数。
        print(f"[DNSMOS] already complete: {len(completed)} rows")
        # 当前指标已经完成时提前返回，不再次加载模型或覆盖结果。
        return

    # 打开 DNSMOS 逐条追加输出。
    handle, writer = open_result_writer(output_csv, DNSMOS_FIELDS)
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 神经指标在 11 GB GPU 上用 batch=4 较稳；DNSMOS 是 CPU ONNX，batch 太小
        # 会重复付出较多调用开销。至少合并 16 条同长度语音不会改变任何逐条分数。
        dnsmos_batch_size = max(batch_size, 16)
        # 使用较大的同长度 CPU 批次减少 DNSMOS 调用开销。
        chunks = list(grouped_chunks(todo, dnsmos_batch_size))
        # DNSMOS 使用 CPU ONNX。默认让 3 个独立进程各使用 3 个线程，可并行处理
        # 三个 batch；资源较少的机器应通过 --dnsmos-workers 调小进程数。
        # spawn 不会继承已经建立的 CUDA 上下文，避免在先跑 UTMOS 后 fork 的隐患。
        payloads = [(list(chunk), 3) for chunk in chunks]
        # 一个工作进程时直接串行计算，避免不必要的进程池。
        if worker_count == 1:
            # 单 worker 必须真正在当前进程串行运行。共享服务器可能在两个独立
            # 评价任务同时建池时清理 multiprocessing semaphore。
            result_iterator = (dnsmos_chunk_worker(payload) for payload in payloads)
            # 按提交顺序收回已完成批次并显示进度。
            for result_rows in tqdm(result_iterator, total=len(chunks), desc="DNSMOS"):
                # 逐条写入该批次四列 DNSMOS 分数。
                for result in result_rows:
                    # 把配对编号与四个质量分数写入逐条 CSV。
                    writer.writerow(result)
        # 此分支：新建干净的子进程，避免继承之前评价模型的 CUDA 状态。
        else:
            # 新建干净的子进程，避免继承之前评价模型的 CUDA 状态。
            context = multiprocessing.get_context("spawn")
            # 创建受限规模的 CPU 进程池，退出时自动回收。
            with ProcessPoolExecutor(
                # 固定进程数量，并使用 spawn 启动方式。
                max_workers=worker_count, mp_context=context
            # 把此进程池交给上下文管理器，退出当前区域后自动关闭工作进程。
            ) as executor:
                # 提交批次并保持返回顺序，满足续算前缀约束。
                result_iterator = executor.map(
                    # 每个任务只分发一个同长度批次。
                    dnsmos_chunk_worker, payloads, chunksize=1
                )
                # 按进程池返回顺序接收批次结果并显示总进度。
                for result_rows in tqdm(
                    # 进度总数使用待算批次数，标签明确为 DNSMOS。
                    result_iterator, total=len(chunks), desc="DNSMOS"
                ):
                    # 逐条写入该批次四列 DNSMOS 分数。
                    for result in result_rows:
                        # 把配对编号与四个质量分数写入逐条 CSV。
                        writer.writerow(result)
    # 无论前面成功或失败，都执行资源恢复和关闭操作。
    finally:
        # 关闭结果文件，确保缓冲中的指标记录写入磁盘。
        handle.close()


# 在一个 CPU 工作进程里计算同长度批次的四列 DNSMOS 分数。
def dnsmos_chunk_worker(
    # 工作任务包含同长度语音列表及每个进程线程数。
    payload: Tuple[List[Mapping[str, str]], int]
# 返回本批每条语音的编号与四个 DNSMOS 分数组成的字典列表。
) -> List[Dict[str, object]]:
    # 在一个 CPU 工作进程里计算同长度批次的四列 DNSMOS 分数。
    """一个独立 CPU 进程处理一个同长度 batch，并把四个 DNSMOS 输出交回主进程。"""

    # import 放在 worker 内，使每个 spawn 进程各自建立一次 ONNX session；同一进程
    # 后续处理其他 batch 时，TorchMetrics 的 lru_cache 会复用 session，不重复加载。
    from torchmetrics.functional.audio.dnsmos import (
        # TorchMetrics 中封装的 Microsoft DNSMOS 评分函数。
        deep_noise_suppression_mean_opinion_score,
    )

    # 拆出本工作进程要算的语音批次及 CPU 线程预算。
    chunk, thread_count = payload
    # 读取该批生成波形，长度不一致时直接停止。
    waveforms = load_equal_length_generated_batch(chunk)
    # 收集模型四路质量输出，随后转为 CPU 数组。
    values = (
        # 调用 TorchMetrics 封装的 Microsoft DNSMOS。
        deep_noise_suppression_mean_opinion_score(
            # 传入没有额外响度归一化的实际生成波形。
            waveforms,
            # 输入波形的真实采样率为 22.05 kHz。
            fs=22050,
            # 采用公开非个性化 DNSMOS，不套用个性化模型校准。
            personalized=False,
            # DNSMOS 的 ONNX 推理固定在 CPU 执行。
            device="cpu",
            # 限制每个 ONNX 会话的 CPU 线程数。
            num_threads=thread_count,
        )
        # 将评分张量放到 CPU 以便整理和写 CSV。
        .cpu()
        # 把张量转为 NumPy 数组供有限值检查及逐条输出。
        .numpy()
    )
    # DNSMOS 每条必须返回四个有限分数。
    if values.shape != (len(chunk), 4) or not np.isfinite(values).all():
        # 报错原因：DNSMOS 每条必须返回四个有限分数。
        raise ValueError(f"Invalid DNSMOS output shape/values: {values}")

    # 收集本批逐配对评分，交回主进程统一写文件。
    results: List[Dict[str, object]] = []
    # 按原批次顺序把模型分数配回语音编号。
    for row, value in zip(chunk, values):
        # 将当前配对四个质量输出组成一条记录。
        results.append(
            {
                # 记录转换配对编号。
                "pair_id": int(row["pair_id"]),
                # 记录DNSMOS 的 P.808 质量预测分数。
                "p808_mos": float(value[0]),
                # 记录DNSMOS 语音部分质量分数。
                "mos_sig": float(value[1]),
                # 记录DNSMOS 背景部分质量分数。
                "mos_bak": float(value[2]),
                # 记录DNSMOS 总体质量分数，论文对照采用这一列。
                "mos_ovr": float(value[3]),
            }
        )
    # 返回当前 CPU 批次的全部逐条 DNSMOS 结果。
    return results


# 统一参考与识别文本的大小写、标点和空格，再比较字符。
def normalize_cer_text(text: str) -> str:
    # 统一参考与识别文本的大小写、标点和空格，再比较字符。
    """固定 CER 文本规则：转 ASCII 大写、去标点、保留字母/撇号/单个词间空格。"""

    # 先把弯撇号统一成 ASR 字典里存在的 ASCII 撇号，再做 Unicode 分解。
    text = text.replace("’", "'").replace("‘", "'")
    # 分解 Unicode 并保留 ASCII，保证参考和识别文本使用同一字符集合。
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # 统一为大写，与 ASR 字典对应。
    text = text.upper()
    # 不属于 A-Z、撇号、空白的字符全部换为空格；连续空白最后合并为一个空格。
    text = re.sub(r"[^A-Z'\s]+", " ", text)
    # 把连续空白合并为单个词间空格，并去掉首尾空格。
    return " ".join(text.split())


# 求把参考文本变成识别文本所需的最少插入、删除和替换次数。
def levenshtein_distance(reference: str, hypothesis: str) -> int:
    # 求把参考文本变成识别文本所需的最少插入、删除和替换次数。
    """动态规划计算最少字符插入、删除、替换次数；只保留一行以节省内存。"""

    # 把较短文本用作动态规划列，降低临时内存。
    if len(reference) < len(hypothesis):
        # 交换两者不改变编辑距离，却能保证工作数组使用较短字符串的长度。
        reference, hypothesis = hypothesis, reference
    # 初始化空参考变成每个识别前缀所需的插入次数。
    previous = list(range(len(hypothesis) + 1))
    # 逐个扩展参考文本前缀。
    for ref_index, ref_char in enumerate(reference, start=1):
        # 当前行首列是删除全部参考前缀的代价。
        current = [ref_index]
        # 逐个比较识别文本字符，计算当前前缀的编辑距离。
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            # 候选方案一：在已有结果后插入一个字符。
            insertion = current[hyp_index - 1] + 1
            # 候选方案二：删除一个参考字符。
            deletion = previous[hyp_index] + 1
            # 候选方案三：字符相同时不加代价，不同时替换一次。
            substitution = previous[hyp_index - 1] + (ref_char != hyp_char)
            # 选择三种操作中最少的编辑次数。
            current.append(min(insertion, deletion, substitution))
        # 本行成为下一参考字符的上一行，避免保存整张矩阵。
        previous = current
    # 最后一个单元格就是完整两句文本的字符错误总数。
    return previous[-1]


# 按 CTC 规则合并连续重复标记、删除空白标记并还原词间空格。
def decode_wav2vec_tokens(token_ids: torch.Tensor, labels: Sequence[str]) -> str:
    # 按 CTC 规则合并连续重复标记、删除空白标记并还原词间空格。
    """贪心 CTC 解码：合并连续重复 token、删 blank(-)、把 | 还原为空格。"""

    # CTC 先合并连续重复 token，非连续重复仍保留。
    collapsed = torch.unique_consecutive(token_ids).tolist()
    # 移除 CTC blank，并把其他 token 转回字符。
    characters = [labels[index] for index in collapsed if labels[index] != "-"]
    # 拼接字符，将词分隔符还原为空格。
    return "".join(characters).replace("|", " ").strip()


# 用固定 wav2vec 2.0 识别生成语音，与源语音文本计算字符错误率。
def evaluate_cer(
    # 输入参数：按 pair_id 排列的评估条目。
    rows: Sequence[Mapping[str, str]],
    # 输入参数：逐条指标 CSV。
    output_csv: Path,
    # 输入参数：张量和模型使用的计算设备。
    device: torch.device,
    # 输入参数：每批处理的语音数。
    batch_size: int,
# 参考文本、识别文本、错误数与 CER 逐条落盘；函数无独立返回值。
) -> None:
    # 用固定 wav2vec 2.0 识别生成语音，与源语音文本计算字符错误率。
    """用 TorchAudio 的 wav2vec2.0 LV-60k/LibriSpeech-960h ASR 计算 source 文本 CER。"""

    # 先核对已有分数属于当前清单且内容有效，再决定从哪里继续。
    existing_rows = validate_existing_metric_csv(
        # 用 CER 的文本、错误数和分母格式核对续算记录。
        output_csv, rows, "cer", CER_FIELDS
    )
    # 把已经验证过的旧结果编号组成集合，用于快速跳过。
    completed = {int(row["pair_id"]) for row in existing_rows}
    # 选择本项指标还没有算完的语音。
    todo = pending_rows(rows, completed)
    # 全部条目已有有效分数时，无需再次加载评价模型。
    if not todo:
        # 显示 CER 文本与错误计数已完整通过续算核对，无需再次 ASR。
        print(f"[CER] already complete: {len(completed)} rows")
        # 当前指标已经完成时提前返回，不再次加载模型或覆盖结果。
        return

    # 固定采用 LV-60k 预训练、LibriSpeech 960h 识别模型；论文未公布其权重哈希。
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    # 加载 ASR 到指定设备，并关闭训练时的随机层行为。
    model = bundle.get_model().to(device).eval()
    # 逐个冻结 ASR 参数，评价不训练识别器。
    for parameter in model.parameters():
        # 冻结评价网络权重，本次不会用评分数据更新模型。
        parameter.requires_grad_(False)
    # 读取与该 ASR 权重对应的字符字典。
    labels = bundle.get_labels()

    # 打开 CER 逐条文本与错误计数输出。
    handle, writer = open_result_writer(output_csv, CER_FIELDS)
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 将待算语音按同源连续段分批。
        chunks = list(grouped_chunks(todo, batch_size))
        # 按同源批次识别生成语音，显示 CER 计算进度。
        for chunk in tqdm(chunks, desc="wav2vec2.0 CER"):
            # 把当前生成波形批次送到 ASR 设备。
            waveforms = load_equal_length_generated_batch(chunk).to(device)
            # 从项目 22.05 kHz 重采样到 ASR 模型规定的采样率。
            waveforms = torchaudio.functional.resample(waveforms, 22050, bundle.sample_rate)
            # 关闭自动求导，仅执行评价或推理，节省显存。
            with torch.inference_mode():
                # 取得每个声学时间帧的字符输出。
                emissions, _ = model(waveforms)
                # 每帧选择分数最高的 token，采用贪心解码，不额外加入语言模型。
                predicted = emissions.argmax(dim=-1).cpu()

            # 把逐句识别结果与该句源文本对应。
            for row, token_ids in zip(chunk, predicted):
                # 以源语音文本为正确答案，按固定 CER 规则清洗。
                reference = normalize_cer_text(row["source_text"])
                # 先做 CTC 解码，再采用与参考相同的文本清洗。
                hypothesis = normalize_cer_text(decode_wav2vec_tokens(token_ids, labels))
                # 清洗后的正确文本不能为空，否则 CER 没有有效分母。
                if len(reference) == 0:
                    # 报错原因：清洗后的正确文本不能为空，否则 CER 没有有效分母。
                    raise ValueError(f"Empty CER reference after normalization: pair {row['pair_id']}")
                # 计算该句字符错误次数，保留整数以便语料级聚合。
                errors = levenshtein_distance(reference, hypothesis)
                # 写出当前配对的完整评分依据与结果。
                writer.writerow(
                    {
                        # 记录转换配对编号。
                        "pair_id": int(row["pair_id"]),
                        # 记录统一书写格式后的正确文本。
                        "reference_text_normalized": reference,
                        # 记录统一书写格式后的 ASR 识别文本。
                        "hypothesis_text_normalized": hypothesis,
                        # 记录字符插入、删除、替换总次数。
                        "char_errors": errors,
                        # 记录参考文本字符数。
                        "reference_chars": len(reference),
                        # 记录字符错误率百分数，越低越好。
                        "cer_percent": 100.0 * errors / len(reference),
                    }
                )
    # 无论前面成功或失败，都执行资源恢复和关闭操作。
    finally:
        # 关闭结果文件，确保缓冲中的指标记录写入磁盘。
        handle.close()
        # 移除 ASR 模型引用，释放后续评价需要的资源。
        del model
        # 若刚才在 GPU 上计算，则执行 GPU 专用的资源清理。
        if device.type == "cuda":
            # 释放未被张量占用的 CUDA 缓存，给后续评价模型留出显存。
            torch.cuda.empty_cache()


# 对较短波形右侧补零，同时返回有效长度比例，供声纹编码器忽略填充。
def pad_waveforms(waves: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # 对较短波形右侧补零，同时返回有效长度比例，供声纹编码器忽略填充。
    """把不同长度波形右侧补零，并返回 SpeechBrain 要求的相对有效长度 0~1。"""

    # 记录各波形真实长度，补零后仍可区分有效部分。
    lengths = torch.tensor([wave.numel() for wave in waves], dtype=torch.float32)
    # 找到当前批次最长语音作为统一长度。
    maximum = int(lengths.max().item())
    # 只在短句右侧补零，再堆成批次。
    padded = torch.stack([F.pad(wave, (0, maximum - wave.numel())) for wave in waves])
    # 返回补齐后的波形和 SpeechBrain 需要的有效长度比例。
    return padded, lengths / float(maximum)


# 比较生成语音与目标参考的 ECAPA 声纹，超过固定阈值就记为验证通过。
def evaluate_sva(
    # 输入参数：按 pair_id 排列的评估条目。
    rows: Sequence[Mapping[str, str]],
    # 输入参数：逐条指标 CSV。
    output_csv: Path,
    # 固定 ECAPA 配置和声纹模型的本地资源目录。
    ecapa_dir: Path,
    # 输入参数：张量和模型使用的计算设备。
    device: torch.device,
    # 输入参数：每批处理的语音数。
    batch_size: int,
# 相似度、阈值与验证标记逐条落盘；函数无独立返回值。
) -> None:
    # 比较生成语音与目标参考的 ECAPA 声纹，超过固定阈值就记为验证通过。
    """用 SpeechBrain ECAPA-TDNN 对生成语音和 target reference 做说话人验证。"""

    # 先核对已有分数属于当前清单且内容有效，再决定从哪里继续。
    existing_rows = validate_existing_metric_csv(
        # 用 SVA 的分数、阈值与二元判断格式核对续算记录。
        output_csv, rows, "sva", SVA_FIELDS
    )
    # 把已经验证过的旧结果编号组成集合，用于快速跳过。
    completed = {int(row["pair_id"]) for row in existing_rows}
    # 选择本项指标还没有算完的语音。
    todo = pending_rows(rows, completed)
    # 全部条目已有有效分数时，无需再次加载评价模型。
    if not todo:
        # 显示 SVA 相似度与阈值判断已有完整有效记录。
        print(f"[SVA] already complete: {len(completed)} rows")
        # 当前指标已经完成时提前返回，不再次加载模型或覆盖结果。
        return

    # ECAPA 结构配置必须存在，不能仅凭某个权重文件开始评分。
    if not (ecapa_dir / "hyperparams.yaml").is_file():
        # 报错原因：ECAPA 结构配置必须存在，不能仅凭某个权重文件开始评分。
        raise FileNotFoundError(f"SpeechBrain ECAPA assets missing: {ecapa_dir}")
    # 使用 SpeechBrain 的 ECAPA 说话人验证接口。
    from speechbrain.inference.speaker import SpeakerRecognition

    # 按固定本地配置构造声纹编码器和相似度函数。
    verifier = SpeakerRecognition.from_hparams(
        # 从已下载的 ECAPA 目录读取模型。
        source=str(ecapa_dir),
        # 复用同一资源目录，避免另建不明版本缓存。
        savedir=str(ecapa_dir),
        # 让声纹编码器使用指定 CPU 或 GPU。
        run_opts={"device": str(device)},
    )

    # 每个 target reference 会在 8,100 个 pair 中被重复使用很多次。先把全部
    # reference 的 ECAPA embedding 各算一次并缓存；这与 verify_batch 的计算
    # 完全等价，但避免重复读取、重采样和编码同一条 reference 约 80 次。
    reference_paths = list(dict.fromkeys(row["reference_wav"] for row in rows))
    # 按参考路径缓存声纹，每份参考只编码一次。
    reference_embeddings: Dict[str, torch.Tensor] = {}
    # 把去重后的参考语音按批大小分组。
    reference_batches = [
        # 选取当前参考语音小批次。
        reference_paths[start : start + batch_size]
        # 依次覆盖全部不同参考路径。
        for start in range(0, len(reference_paths), batch_size)
    ]
    # 预先提取目标参考声纹，并显示缓存进度。
    for path_batch in tqdm(reference_batches, desc="Cache target ECAPA embeddings"):
        # 读取这一批目标说话人参考语音。
        reference_22k = [load_audio(Path(path)) for path in path_batch]
        # ECAPA 要求 16 kHz，因此先逐句重采样参考。
        reference_16k = [
            # 把参考波形从 22.05 kHz 转为 16 kHz。
            torchaudio.functional.resample(waveform, 22050, 16000)
            # 逐条处理当前参考批次中的真实波形。
            for waveform in reference_22k
        ]
        # 补齐参考长度，同时传递真实有效比例。
        reference_batch, reference_lens = pad_waveforms(reference_16k)
        # 关闭自动求导，仅执行评价或推理，节省显存。
        with torch.inference_mode():
            # 把参考波形编码为 ECAPA 声纹向量。
            embeddings = verifier.encode_batch(
                # 把补齐后的参考波形送到编码器设备。
                reference_batch.to(device),
                # 同步传入每条参考的有效长度，避免把补零当成语音。
                reference_lens.to(device),
                # 沿用 verify_batch 的编码设置，后续由相似度函数比较。
                normalize=False,
            # 目标参考声纹无需梯度，缓存到 CPU 便于跨批次复用。
            ).detach().cpu()
        # 把每个参考声纹与其原始路径绑定。
        for path, embedding in zip(path_batch, embeddings):
            # 缓存该参考声纹，供多个转换配对复用。
            reference_embeddings[path] = embedding

    # 打开 SVA 逐条相似度、阈值及判断输出。
    handle, writer = open_result_writer(output_csv, SVA_FIELDS)
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 将待算语音按同源连续段分批。
        chunks = list(grouped_chunks(todo, batch_size))
        # 逐批计算生成语音与目标参考的说话人验证结果。
        for chunk in tqdm(chunks, desc="ECAPA-TDNN SVA"):
            # 读取当前配对的生成语音。
            converted_22k = [load_audio(Path(row["output_wav"])) for row in chunk]
            # grouped_chunks 保证同一 chunk 的转换语音来自同一 source、长度完全
            # 相同；一次批量重采样与逐条重采样等价，并避免重复构造 81 次滤波器。
            converted_batch = torchaudio.functional.resample(
                # 对同长度生成批次统一从 22.05 kHz 重采样到 16 kHz。
                torch.stack(converted_22k), 22050, 16000
            )
            # 同源生成批次没有补零，每条有效长度比例都是 1。
            converted_lens = torch.ones(len(chunk), dtype=torch.float32)
            # 按每条配对的目标参考顺序取得缓存声纹。
            target_embeddings = torch.stack(
                # 确保每条生成语音比较的是它指定的目标说话人参考。
                [reference_embeddings[row["reference_wav"]] for row in chunk]
            # 按本批目标参考顺序堆叠后的声纹张量送到评价设备。
            ).to(device)

            # 关闭自动求导，仅执行评价或推理，节省显存。
            with torch.inference_mode():
                # 提取生成语音的 ECAPA 声纹向量。
                converted_embeddings = verifier.encode_batch(
                    # 把 16 kHz 生成波形送到声纹编码器。
                    converted_batch.to(device),
                    # 指定全部波形采样点有效。
                    converted_lens.to(device),
                    # 沿用 verify_batch 的编码设置，后续由相似度函数比较。
                    normalize=False,
                )
                # 计算生成声纹与对应目标参考声纹的余弦相似度。
                scores = verifier.similarity(converted_embeddings, target_embeddings)
                # 严格大于 0.25 才判定目标说话人验证通过。
                predictions = scores > SVA_THRESHOLD
            # 脱离计算图并移回 CPU，整理逐句相似度。
            scores = scores.detach().cpu().reshape(-1).numpy()
            # 整理与每个相似度对应的布尔验证结果。
            predictions = predictions.detach().cpu().reshape(-1).numpy().astype(bool)
            # 检查声纹评分数量与输入一一对应，且没有异常数值。
            if scores.shape[0] != len(chunk) or not np.isfinite(scores).all():
                # 报错原因：检查声纹评分数量与输入一一对应，且没有异常数值。
                raise ValueError(f"Invalid SVA output: scores={scores}, predictions={predictions}")
            # 保持配对编号、连续分数和二元判断对应。
            for row, score, prediction in zip(chunk, scores, predictions):
                # 写出当前配对的完整评分依据与结果。
                writer.writerow(
                    {
                        # 记录转换配对编号。
                        "pair_id": int(row["pair_id"]),
                        # 记录转换语音与目标参考的声纹余弦相似度。
                        "cosine_score": float(score),
                        # 记录说话人验证通过阈值。
                        "threshold": SVA_THRESHOLD,
                        # 记录说话人验证是否通过，1 表示通过。
                        "verified": int(prediction),
                    }
                )
    # 无论前面成功或失败，都执行资源恢复和关闭操作。
    finally:
        # 关闭结果文件，确保缓冲中的指标记录写入磁盘。
        handle.close()
        # 移除声纹评价模型引用。
        del verifier
        # 若刚才在 GPU 上计算，则执行 GPU 专用的资源清理。
        if device.type == "cuda":
            # 释放未被张量占用的 CUDA 缓存，给后续评价模型留出显存。
            torch.cuda.empty_cache()


# 使用同一续算校验器读取完整指标表，缺条目时不生成正式汇总。
def read_metric_rows(
    # 输入参数：待处理文件路径。
    path: Path,
    # 输入参数：本次选中的评估条目。
    selected_rows: Sequence[Mapping[str, str]],
    # 本次待检查的指标名称，用来选择分数约束。
    metric_name: str,
    # 该指标 CSV 的固定字段名与顺序。
    fieldnames: Sequence[str],
# 返回完整且经过严格验证的该项指标记录列表。
) -> List[Dict[str, str]]:
    # 使用同一续算校验器读取完整指标表，缺条目时不生成正式汇总。
    """Read one completed metric CSV through the same strict resume validator."""

    # 缺失生成波形或指标文件时停止，不用不完整数据继续。
    if not path.is_file():
        # 报错原因：缺失生成波形或指标文件时停止，不用不完整数据继续。
        raise FileNotFoundError(f"Metric CSV not found: {path}")
    # 设置按 pair_id 排列的评估条目。
    rows = validate_existing_metric_csv(path, selected_rows, metric_name, fieldnames)
    # 完整汇总必须所有选定条目都有分数，只有有效前缀仍不算完成。
    if len(rows) != len(selected_rows):
        # 报错原因：完整汇总必须所有选定条目都有分数，只有有效前缀仍不算完成。
        raise ValueError(
            # 指出哪项指标只有部分有效结果，不能发布全量均值。
            f"Metric CSV is only a valid prefix, not complete: {path}, "
            # 完整汇总必须所有选定条目都有分数，只有有效前缀仍不算完成。
            f"got {len(rows)}, expected {len(selected_rows)}"
        )
    # 返回通过本函数检查的记录序列。
    return rows


# 在已知缓存位置查找 ASR 权重，供报告记录准确模型指纹。
def locate_torchaudio_asr_checkpoint() -> Optional[Path]:
    # 在已知缓存位置查找 ASR 权重，供报告记录准确模型指纹。
    """寻找 TorchAudio 已缓存的 wav2vec2.0 文件，以便报告精确权重哈希。"""

    # 在服务器用户与项目附近的已知缓存位置查找 ASR 权重。
    candidates = [
        # 使用当前租用服务器用户的缓存根目录。
        Path.home()
        # 进入公开模型的本地缓存目录。
        / ".cache"
        # 进入 PyTorch 下载资源缓存。
        / "torch"
        # 进入 Torch Hub 缓存。
        / "hub"
        # 进入模型权重文件缓存位置。
        / "checkpoints"
        # 固定本项目 CER 所用 wav2vec 2.0 权重文件名。
        / "wav2vec2_fairseq_large_lv60k_asr_ls960.pth",
        # 检查项目父目录下的同布局缓存作为兼容位置。
        SERVER_PROJECT_ROOT.parent
        # 进入公开模型的本地缓存目录。
        / ".cache"
        # 进入 PyTorch 下载资源缓存。
        / "torch"
        # 进入 Torch Hub 缓存。
        / "hub"
        # 进入模型权重文件缓存位置。
        / "checkpoints"
        # 固定本项目 CER 所用 wav2vec 2.0 权重文件名。
        / "wav2vec2_fairseq_large_lv60k_asr_ls960.pth",
    ]
    # 按固定顺序查找第一个真实存在的 ASR 权重。
    for path in candidates:
        # 只有普通文件才能用作模型字节身份来源。
        if path.is_file():
            # 返回实际找到的权重路径。
            return path
    # 没有找到缓存时明确返回不可得，不编造哈希。
    return None


# 记录所有评价模型的文件路径与哈希，缺少可选缓存会明确标注。
def metric_asset_manifest(utmos_dir: Path, ecapa_dir: Path) -> Dict[str, object]:
    # 记录所有评价模型的文件路径与哈希，缺少可选缓存会明确标注。
    """收集评价模型文件路径和哈希；不存在的可选缓存明确记为 unavailable。"""

    # 列出需要记入评价资产身份表的模型文件。
    assets: Dict[str, Path] = {
        # 记录固定 UTMOS 自然度模型权重。
        "utmos_checkpoint": utmos_dir / "epoch=3-step=7459.ckpt",
        # UTMOS 特征提取网络与 CER ASR 是两份不同权重。
        "utmos_wav2vec_small": utmos_dir / "wav2vec_small.pt",
        # 记录 SVA 使用的目标声纹编码器权重。
        "ecapa_embedding_model": ecapa_dir / "embedding_model.ckpt",
        # 记录DNSMOS 语音、背景和总体分支 ONNX 模型。
        "dnsmos_sig_bak_ovr": Path.home()
        # 进入服务器用户的 TorchMetrics 模型缓存。
        / ".torchmetrics"
        # 进入 Microsoft DNSMOS 资源缓存子目录。
        / "DNSMOS"
        # 进入 Microsoft DNSMOS 资源缓存子目录。
        / "DNSMOS"
        # 定位语音、背景及总体分数 ONNX 模型。
        / "sig_bak_ovr.onnx",
        # P.808 分支使用独立 ONNX 权重，单独记录身份。
        "dnsmos_p808_model_v8": Path.home()
        # 进入服务器用户的 TorchMetrics 模型缓存。
        / ".torchmetrics"
        # 进入 Microsoft DNSMOS 资源缓存子目录。
        / "DNSMOS"
        # 进入 Microsoft DNSMOS 资源缓存子目录。
        / "DNSMOS"
        # 定位 P.808 质量预测模型。
        / "model_v8.onnx",
    }
    # 查找 CER 模型的实际本地缓存。
    asr_checkpoint = locate_torchaudio_asr_checkpoint()
    # 找到 ASR 缓存后才能记录它的确切指纹。
    if asr_checkpoint is not None:
        # 把真实 ASR 权重加入资产记录。
        assets["wav2vec2_asr_checkpoint"] = asr_checkpoint

    # 收集逐模型文件的路径与内容指纹。
    result: Dict[str, object] = {}
    # 逐个评价模型计算文件身份。
    for name, path in assets.items():
        # 存在时规范化为绝对路径；缺失时保留原期望位置。
        resolved = path.resolve() if path.exists() else path
        # 为当前评价模型保存路径和是否可得的文件指纹。
        result[name] = {
            # 记录待处理文件路径。
            "path": str(resolved),
            # 真实文件才计算哈希，缺失时明确记录不可得。
            "sha256": sha256_file(resolved) if resolved.is_file() else "unavailable",
        }
    # 返回本函数已经构造或校验的结果，供调用方继续处理。
    return result


# 汇总客观指标，并记录样本、权重、软件和与论文比较的限制。
def write_summary(
    # 输入参数：命令行选项。
    args: argparse.Namespace,
    # 输入参数：本次选中的评估条目。
    selected_rows: Sequence[Mapping[str, str]],
    # 输入参数：固定源语音和目标参考的配对清单。
    pairs_csv: Path,
    # 输入参数：推理过程记录。
    inference_report: Path,
# 将汇总写成 JSON 和 CSV 并打印完成状态，无独立返回值。
) -> None:
    # 汇总客观指标，并记录样本、权重、软件和与论文比较的限制。
    """汇总四份逐条 CSV；正式模式产生可直接放入论文对照表的四个主数。"""

    # 确定本次结果目录。
    output_dir = args.output_dir.resolve()
    # 以实际选中清单长度判断完整评价或诊断状态。
    selected_count = len(selected_rows)
    # 必须有推理身份记录，才能确定这些分数属于哪个模型。
    if not inference_report.is_file():
        # 报错原因：必须有推理身份记录，才能确定这些分数属于哪个模型。
        raise FileNotFoundError(f"Inference report not found: {inference_report}")
    # 读取实际推理阶段保存的模型与数据来源。
    inference_metadata = json.loads(inference_report.read_text(encoding="utf-8"))
    # 从推理记录确定教师/学生身份及对应论文参考行。
    system = resolve_inference_system(inference_metadata)
    # 设置教师模型训练轮数。
    teacher_epoch = int(system["teacher_epoch"])
    # 从已有记录取得声码器权重路径。
    vocoder_checkpoint = str(inference_metadata.get("vocoder_checkpoint", ""))
    # 从已有记录取得通用或自训声码器类型。
    vocoder_kind = str(inference_metadata.get("vocoder_kind", "")).strip().lower()
    # 为缺少显式类型的旧教师报告保留路径识别兼容分支。
    if vocoder_kind not in {"public", "custom"}:
        # 不区分大小写检查旧权重路径标签。
        checkpoint_text = vocoder_checkpoint.casefold()
        # 设置通用或自训声码器类型。
        vocoder_kind = (
            # 路径符合自训归档命名时标记为自训声码器。
            "custom"
            # 依据旧实验的明确目录或归档文件名识别自训权重。
            if "hifigan_v1_mrd" in checkpoint_text or "generator_step_" in checkpoint_text
            # 其他历史默认路径归为通用声码器；正式新结果应显式记录类型。
            else "public"
        )
    # 通用基线报告保留公开 VCTK V1 与作者未公开权重的差异。
    if vocoder_kind == "public":
        # 使用能辨认历史通用基线的标签。
        vocoder_label = "temporary public pretrained_hifigan_22k/VCTK_V1"
        # 设置与作者未公开实现无法核实一致的条件。
        known_deviation = (
            # 设置与作者未公开实现无法核实一致的条件。
            "Waveforms were synthesized by public pretrained_hifigan_22k/VCTK_V1, not the "
            # 设置与作者未公开实现无法核实一致的条件。
            "paper authors' unpublished modified HiFi-GAN; do not present as an exact paper number."
        )
    # 此分支：自训标签同时写出网络结构与确切归档文件名。
    else:
        # 自训标签同时写出网络结构与确切归档文件名。
        vocoder_label = (
            # 自训标签同时写出网络结构与确切归档文件名。
            "self-trained HiFi-GAN V1 + MPD + UnivNet MRD "
            # 自训标签同时写出网络结构与确切归档文件名。
            f"({Path(vocoder_checkpoint).name})"
        )
        # 设置与作者未公开实现无法核实一致的条件。
        known_deviation = (
            # 设置与作者未公开实现无法核实一致的条件。
            "Waveforms were synthesized by the user-requested scratch-trained HiFi-GAN V1 + "
            # 设置与作者未公开实现无法核实一致的条件。
            "MPD + UnivNet MRD checkpoint. FastVoiceGrad instead discloses a frozen pretrained "
            # 设置与作者未公开实现无法核实一致的条件。
            "Generator and no standalone modified-vocoder training recipe/checkpoint. The scratch "
            # 设置与作者未公开实现无法核实一致的条件。
            "run inherits official HiFi-GAN VCTK normalization while the VoiceGrad mel interface's "
            # 设置与作者未公开实现无法核实一致的条件。
            "exact author-side amplitude treatment is unpublished; do not present this as the "
            # 设置与作者未公开实现无法核实一致的条件。
            "authors' exact internal number."
        )
    # 要求 UTMOS 每条分数完整通过校验。
    utmos_rows = read_metric_rows(
        # 读取与当前清单绑定的 UTMOS 两列结果。
        output_dir / "utmos.csv", selected_rows, "utmos", UTMOS_FIELDS
    )
    # 要求 DNSMOS 全部配对与四列输出完整。
    dnsmos_rows = read_metric_rows(
        # 读取当前完整 DNSMOS 逐条表。
        output_dir / "dnsmos.csv", selected_rows, "dnsmos", DNSMOS_FIELDS
    )
    # 要求所有 CER 文本与错误计数可重算。
    cer_rows = read_metric_rows(
        # 读取当前完整 CER 逐条依据和分数。
        output_dir / "cer.csv", selected_rows, "cer", CER_FIELDS
    )
    # 要求所有 SVA 分数、阈值与判断一致。
    sva_rows = read_metric_rows(
        # 读取当前完整 SVA 逐条表。
        output_dir / "sva.csv", selected_rows, "sva", SVA_FIELDS
    )

    # 按配对对 UTMOS 求算术平均。
    utmos_mean = float(np.mean([float(row["utmos"]) for row in utmos_rows]))
    # 只把 DNSMOS 的总体质量列作为论文主数。
    dnsmos_mean = float(np.mean([float(row["mos_ovr"]) for row in dnsmos_rows]))
    # 合并全语料字符编辑错误次数。
    total_errors = sum(int(row["char_errors"]) for row in cer_rows)
    # 合并全语料参考字符数，长句会提供更多字符分母。
    total_reference_chars = sum(int(row["reference_chars"]) for row in cer_rows)
    # 主 CER 使用总错误除以总字符数，乘 100 表示百分数。
    corpus_cer = 100.0 * total_errors / total_reference_chars
    # 另存每句等权的平均 CER，仅作诊断，不能替代语料级主 CER。
    macro_cer = float(np.mean([float(row["cer_percent"]) for row in cer_rows]))
    # 把二元验证结果的平均转换为通过百分比。
    sva_percent = 100.0 * float(np.mean([int(row["verified"]) for row in sva_rows]))
    # 保留声纹连续分数均值，帮助解释阈值附近的变化。
    sva_score_mean = float(np.mean([float(row["cosine_score"]) for row in sva_rows]))

    # 设置客观指标汇总。
    summary = {
        # 记录流程完成状态。
        "status": (
            # 仅当实际覆盖 8,100 对时标记正式全量完成。
            "formal_8100_complete"
            # 以实际评价条数决定是否满足正式完整规模。
            if selected_count == 8100
            # 数量不足 8,100 时，按显式诊断开关区分受控诊断与普通冒烟测试。
            else (
                # 非标准小清单明确标记为受控诊断。
                "controlled_diagnostic_nonpaper_manifest"
                # 只有用户显式诊断开关才能使用非标准清单身份。
                if args.diagnostic_allow_nonpaper_pairs
                # 普通截取小样本只标记冒烟测试，不能当论文正式结果。
                else "smoke_test_only"
            )
        ),
        # 记录已经完成指标计算的配对数。
        "evaluated_pairs": selected_count,
        # 记录教师或学生系统标识。
        "system_id": system["system_id"],
        # 记录报告中的模型名称。
        "system_label": system["system_label"],
        # 记录教师或学生角色。
        "model_role": system["model_role"],
        # 记录教师模型训练轮数。
        "teacher_epoch": teacher_epoch,
        # 记录学生模型训练轮数。
        "student_epoch": system["student_epoch"],
        # 记录学生优化器累计更新次数。
        "student_global_step": system["student_global_step"],
        # 记录学生推理去噪次数。
        "student_steps": system["student_steps"],
        # 记录学生推理起始噪声步。
        "student_start_step": system["student_start_step"],
        # 记录学生权重路径。
        "student_checkpoint": system["student_checkpoint"],
        # 记录学生权重的字节指纹。
        "student_checkpoint_sha256": system["student_checkpoint_sha256"],
        # 记录学生训练约定的指纹。
        "student_run_contract_sha256": system["student_run_contract_sha256"],
        # 记录通用或自训声码器类型。
        "vocoder_kind": vocoder_kind,
        # 记录声码器权重路径。
        "vocoder_checkpoint": vocoder_checkpoint,
        # 记录声码器权重指纹。
        "vocoder_checkpoint_sha256": inference_metadata.get(
            # 读取推理时的声码器指纹；旧记录没有则明确不可得。
            "vocoder_checkpoint_sha256", "unavailable"
        ),
        # 记录声码器配置指纹。
        "vocoder_config_sha256": inference_metadata.get(
            # 读取声码器配置指纹，缺失不作虚构。
            "vocoder_config_sha256", "unavailable"
        ),
        # 记录推理身份记录指纹。
        "run_identity_sha256": inference_metadata.get(
            # 读取本次推理约定指纹，旧记录缺失明确标注。
            "run_identity_sha256", "unavailable"
        ),
        # 记录用于论文对照的四项主指标。
        "headline_metrics": {
            # 记录逐句 UTMOS 的算术平均，越高越好。
            "UTMOS_mean_higher_is_better": utmos_mean,
            # 记录逐句 DNSMOS 总体分数平均，越高越好。
            "DNSMOS_mos_ovr_mean_higher_is_better": dnsmos_mean,
            # 记录全语料字符错误率，越低越好。
            "CER_corpus_percent_lower_is_better": corpus_cer,
            # 记录目标说话人验证通过率，越高越好。
            "SVA_percent_higher_is_better": sva_percent,
        },
        # 记录帮助定位差异的附加统计。
        "additional_diagnostics": {
            # 记录逐句 CER 百分数的平均，仅作辅助统计。
            "CER_macro_utterance_percent": macro_cer,
            # 记录全语料字符错误次数。
            "CER_total_character_errors": total_errors,
            # 记录全语料参考字符总数。
            "CER_total_reference_characters": total_reference_chars,
            # 记录目标声纹余弦分数均值。
            "SVA_mean_cosine_score": sva_score_mean,
            # 记录固定的声纹验证阈值。
            "SVA_threshold": SVA_THRESHOLD,
        },
        # 记录论文表 1 中所对应的模型行。
        "paper_table1_reference_row": system["paper_reference_row"],
        # 记录仅供比较的论文表 1 参考值。
        "paper_table1_reference_not_used_in_calculation": system["paper_reference"],
        # 仅为教师旧报告加入兼容参考字段，学生报告不附加。
        **(
            {
                # 保留旧教师报告读取器所需的论文参考字段。
                "paper_table1_voicegrad30_reference_not_used_in_calculation":
                    # 填入论文 VoiceGrad-30 参考数，完全不参与本次评价模型运算。
                    PAPER_VOICEGRAD30_REFERENCE
            }
            # 只有教师报告添加旧版教师兼容字段。
            if system["system_id"] == "voicegrad30"
            # 学生报告不加入教师专属兼容字段。
            else {}
        ),
        # 记录指标模型和计算规则。
        "method": {
            # 说明 UTMOS 沿用官方 strong learner 的输出标度。
            "UTMOS": "official UTMOS strong learner; official output transformation raw*2+3",
            # 说明 DNSMOS 使用非个性化模式，总体列作为主指标。
            "DNSMOS": "Microsoft DNSMOS via TorchMetrics; personalized=False; headline=mos_ovr",
            # 记录 ASR 权重类型、贪心 CTC 和文本清洗规则。
            "CER": (
                # 记录 ASR 权重类型、贪心 CTC 和文本清洗规则。
                "TorchAudio WAV2VEC2_ASR_LARGE_LV60K_960H; greedy CTC; source transcript; "
                # 记录 ASR 权重类型、贪心 CTC 和文本清洗规则。
                "ASCII uppercase; punctuation removed; apostrophe and single word spaces retained"
            ),
            # 记录主 CER 为语料级总错误除总字符数，非逐句等权平均。
            "CER_aggregation": "corpus sum(char_errors) / sum(reference_chars) * 100",
            # 记录目标参考声纹验证接口及固定 0.25 阈值。
            "SVA": "SpeechBrain spkrec-ecapa-voxceleb; converted vs target reference; threshold=0.25",
        },
        # 记录与作者未公开实现无法核实一致的条件。
        "known_deviation": known_deviation,
        # 记录固定源语音和目标参考的配对清单。
        "pairs_csv": str(pairs_csv),
        # 记录配对清单指纹。
        "pairs_csv_sha256": sha256_file(pairs_csv),
        # 记录推理过程记录。
        "inference_report": str(inference_report),
        # 记录推理记录的文件指纹。
        "inference_report_sha256": (
            # 对实际引用的推理报告重新计算文件指纹。
            sha256_file(inference_report) if inference_report.is_file() else "unavailable"
        ),
        # 记录评价模型的路径和指纹。
        "metric_assets": metric_asset_manifest(args.utmos_dir.resolve(), args.ecapa_dir.resolve()),
        # 记录评价环境中的软件版本。
        "software_versions": {
            # 保存解释器完整版本。
            "python": sys.version,
            # 记录当前 PyTorch 版本。
            "torch": torch.__version__,
            # 记录当前 TorchAudio 版本。
            "torchaudio": torchaudio.__version__,
            # 记录ECAPA 评价接口版本。
            "speechbrain": package_version("speechbrain"),
            # 记录DNSMOS 包装接口版本。
            "torchmetrics": package_version("torchmetrics"),
            # 记录兼容旧 UTMOS 权重的 Lightning 版本。
            "pytorch-lightning": package_version("pytorch-lightning"),
            # 记录DNSMOS 推理运行库版本。
            "onnxruntime": package_version("onnxruntime"),
            # 记录音频特征辅助库版本。
            "librosa": package_version("librosa"),
        },
        # 记录四份逐条指标表的指纹。
        "individual_metric_csv_sha256": {
            # 逐份指标 CSV 计算字节指纹，绑定汇总来源。
            name: sha256_file(output_dir / f"{name}.csv")
            # 覆盖四项正式客观结果文件。
            for name in ["utmos", "dnsmos", "cer", "sva"]
        },
        # 记录报告生成时刻。
        "created_unix_time": time.time(),
    }

    # 完整可追溯客观汇总的正式保存路径。
    summary_path = output_dir / "objective_summary.json"
    # 先写同目录临时汇总文件。
    temporary = summary_path.with_suffix(".json.tmp")
    # 把全部指标与来源完整写入临时 JSON。
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 完整写完后原子发布汇总。
    os.replace(temporary, summary_path)

    # 再写一份只有一行的 CSV，组会时用 Excel 打开或复制到表格更方便。
    table_path = output_dir / "objective_summary.csv"
    # 另外输出一行主指标 CSV，便于直接导入报告表格。
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        # 明确汇总表列顺序，和 JSON 中的主指标保持一致。
        writer = csv.DictWriter(
            # 向当前已打开的 CSV 文件写入记录。
            handle,
            # 列出一行简表中的模型身份、样本数量、四项指标和声码器。
            fieldnames=[
                # 表格中展示教师或学生模型名称。
                "system",
                # 字段含义：已经完成指标计算的配对数。
                "evaluated_pairs",
                # 字段含义：UTMOS 自然度预测分数，越高越好。
                "utmos",
                # 字段含义：DNSMOS 总体质量预测分数，越高越好。
                "dnsmos",
                # 字段含义：字符错误率百分数，越低越好。
                "cer_percent",
                # 字段含义：说话人验证通过比例，越高越好。
                "sva_percent",
                # 表格中展示通用或自训声码器标签。
                "vocoder",
            ],
        )
        # 写出固定列名，让后续续算和汇总能核对格式。
        writer.writeheader()
        # 写出当前配对的完整评分依据与结果。
        writer.writerow(
            {
                # 写出已经由推理记录核对的模型名称。
                "system": system["system_label"],
                # 记录已经完成指标计算的配对数。
                "evaluated_pairs": selected_count,
                # 记录UTMOS 自然度预测分数，越高越好。
                "utmos": utmos_mean,
                # 记录DNSMOS 总体质量预测分数，越高越好。
                "dnsmos": dnsmos_mean,
                # 记录字符错误率百分数，越低越好。
                "cer_percent": corpus_cer,
                # 记录说话人验证通过比例，越高越好。
                "sva_percent": sva_percent,
                # 写出对应权重的声码器说明。
                "vocoder": vocoder_label,
            }
        )

    # 在终端输出完整客观汇总与来源。
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # 输出指标校验与汇总流程通过标记，报告内仍明确正式或诊断状态。
    print("FORMAL_OBJECTIVE_METRICS_PASSED")
    # 教师结果保留旧自动化认识的完成标记。
    if system["system_id"] == "voicegrad30":
        # Keep the historical completion marker for existing teacher automation.
        # 告知教师评价包装器本次计算与汇总完成。
        print("VOICEGRAD30_PAPER_METRICS_PASSED")
    # 此分支：告知学生评价包装器本次计算与汇总完成。
    else:
        # 告知学生评价包装器本次计算与汇总完成。
        print("FASTVOICEGRAD_PAPER_METRICS_PASSED")


# 执行本脚本的输入检查、主要处理和结果记录。
def main() -> None:
    # 执行本脚本的输入检查、主要处理和结果记录。
    """总流程：检查输入、按选择顺序算指标、四项齐全时生成汇总报告。"""

    # 设置命令行选项。
    args = parse_args()
    # 把命令行项目根目录规范为绝对路径。
    args.project_root = args.project_root.resolve()
    # 评价属于项目计算，只允许当前租用服务器执行。
    assert_rented_server(args.project_root)
    # 每批条数必须为正。
    if args.batch_size <= 0:
        # 报错原因：每批条数必须为正。
        raise ValueError("batch-size must be positive")
    # DNSMOS 工作进程数必须为正。
    if args.dnsmos_workers <= 0:
        # 报错原因：DNSMOS 工作进程数必须为正。
        raise ValueError("dnsmos-workers must be positive")
    # 固定本次已完成推理结果所在目录。
    inference_dir = args.inference_dir.resolve()
    # 固定逐条指标和汇总的实际保存目录。
    args.output_dir = args.output_dir.resolve()
    # 固定本次 UTMOS 模型来源路径。
    args.utmos_dir = args.utmos_dir.resolve()
    # 固定本次 ECAPA 模型来源路径。
    args.ecapa_dir = args.ecapa_dir.resolve()
    # 设置固定源语音和目标参考的配对清单。
    pairs_csv = inference_dir / "pairs.csv"
    # 设置推理过程记录。
    inference_report = inference_dir / "inference_report.json"
    # 设置按 pair_id 排列的评估条目。
    rows = load_pairs(
        # 传入固定源语音和目标参考的配对清单。
        pairs_csv,
        # 传入明确的配对切片起点。
        args.start,
        # 传入可选诊断条数上限。
        args.limit,
        # 默认必须为正式 8,100 对，只有显式诊断才放宽清单规模。
        require_formal_manifest=not args.diagnostic_allow_nonpaper_pairs,
    )

    # 必须有推理身份记录，才能确定这些分数属于哪个模型。
    if not inference_report.is_file():
        # 报错原因：必须有推理身份记录，才能确定这些分数属于哪个模型。
        raise FileNotFoundError(f"Inference report not found: {inference_report}")
    # 读取实际推理阶段保存的模型与数据来源。
    inference_metadata = json.loads(inference_report.read_text(encoding="utf-8"))
    # 先确定模型身份，再执行相应的严格评价门禁。
    inference_system = resolve_inference_system(inference_metadata)
    # 默认旧教师记录没有逐波形指纹，学生分支会要求并载入。
    expected_generated_wav_sha256: Optional[Dict[str, str]] = None
    # 学生正式评价绑定固定目录、配对清单及所有波形内容。
    if inference_system["system_id"] == "fastvoicegrad":
        # 学生指标必须存放在本次推理目录的 metrics 子目录。
        expected_student_metrics_dir = (inference_dir / "metrics").resolve()
        # 禁止把学生分数写进其他模型的指标目录。
        if args.output_dir != expected_student_metrics_dir:
            # 报错原因：禁止把学生分数写进其他模型的指标目录。
            raise RuntimeError(
                # 错误提示要求学生指标目录与同一推理结果绑定，防止混入其他模型。
                "FastVoiceGrad metrics must stay inside their inference directory: "
                # 禁止把学生分数写进其他模型的指标目录。
                f"expected {expected_student_metrics_dir}, got {args.output_dir}"
            )
        # 重新计算当前配对清单内容指纹。
        actual_pairs_sha256 = sha256_file(pairs_csv)
        # 清单与生成波形时使用的不一致，就不能继续评价。
        if inference_metadata.get("pairs_csv_sha256") != actual_pairs_sha256:
            # 报错原因：清单与生成波形时使用的不一致，就不能继续评价。
            raise RuntimeError(
                # 清单与生成波形时使用的不一致，就不能继续评价。
                "FastVoiceGrad inference report is not bound to the current pairs.csv"
            )
        # 学生推理记录数量必须与本次清单完全一致。
        if int(inference_metadata.get("selected_pair_count", -1)) != len(rows):
            # 报错原因：学生推理记录数量必须与本次清单完全一致。
            raise RuntimeError(
                # 学生推理记录数量必须与本次清单完全一致。
                "FastVoiceGrad inference report pair count differs from the selected manifest"
            )
        # 读取并完整核对学生逐配对波形指纹。
        expected_generated_wav_sha256 = load_generated_wav_hash_manifest(
            # 以推理身份记录与实际配对清单双向检查波形表。
            inference_metadata, rows
        )
    # 把本次续算绑定到输入清单、推理记录、源码和波形集合。
    metrics_identity: Dict[str, object] = {
        # 记录身份记录的数据格式版本。
        "identity_format_version": 1,
        # 记录推理过程记录。
        "inference_report": str(inference_report),
        # 记录推理记录的文件指纹。
        "inference_report_sha256": sha256_file(inference_report),
        # 记录固定源语音和目标参考的配对清单。
        "pairs_csv": str(pairs_csv),
        # 记录配对清单指纹。
        "pairs_csv_sha256": sha256_file(pairs_csv),
        # 记录本次选中的配对数。
        "selected_pair_count": len(rows),
        # 记录本次配对切片起点。
        "start": int(args.start),
        # 记录诊断时最多处理的配对数。
        "limit": args.limit,
        # 记录是否明确允许非正式小样本诊断。
        "diagnostic_allow_nonpaper_pairs": bool(args.diagnostic_allow_nonpaper_pairs),
        # 记录执行源码的完整字节指纹。
        "implementation_sha256": sha256_file(Path(__file__)),
        # 记录全部生成波形指纹表的整体指纹。
        "generated_wav_collection_sha256": inference_metadata.get(
            # 保存波形集合整体指纹；旧教师没有记录时明确注明。
            "generated_wav_collection_sha256", "legacy_report_unavailable"
        ),
    }
    # 设置指标续算身份文件路径。
    identity_path = args.output_dir / "metrics_run_identity.json"
    # 检查输出目录是否已有可能属于旧实验的 CSV。
    existing_metric_csvs = list(args.output_dir.glob("*.csv")) if args.output_dir.exists() else []
    # 存在旧指标身份时先核对，不能直接追加。
    if identity_path.is_file():
        # 读取旧分数所绑定的输入与源码身份。
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        # 任一身份变化都会拒绝复用旧分数。
        if existing_identity != metrics_identity:
            # 报错原因：任一身份变化都会拒绝复用旧分数。
            raise RuntimeError(
                # 任一身份变化都会拒绝复用旧分数。
                "Existing metric CSVs belong to a different inference/code identity; "
                # 任一身份变化都会拒绝复用旧分数。
                f"refusing to reuse {args.output_dir}"
            )
    # 有 CSV 却没有身份记录时无法确认来源，必须拒绝续算。
    elif existing_metric_csvs:
        # 报错原因：有 CSV 却没有身份记录时无法确认来源，必须拒绝续算。
        raise RuntimeError(
            # 有 CSV 却没有身份记录时无法确认来源，必须拒绝续算。
            f"Found metric CSVs without metrics_run_identity.json in {args.output_dir}"
        )
    # 此分支：首次评分前保存输入、代码和波形的绑定关系，供中断后续算检查。
    else:
        # 首次评分前保存输入、代码和波形的绑定关系，供中断后续算检查。
        atomic_write_json(identity_path, metrics_identity)
    # 输入和旧状态核对通过后才建立本次输出目录。
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Refuse malformed resume state before scanning 8,100 WAV headers or loading any
    # evaluator. Validate every existing metric, including an unselected one that a
    # later summary would otherwise try to reuse.
    # 逐项列出所有指标 CSV 的固定结构，用于提前核对旧文件。
    metric_schemas = (
        # 核对 UTMOS 编号与分数列。
        ("utmos", UTMOS_FIELDS),
        # 核对 DNSMOS 四路分数列。
        ("dnsmos", DNSMOS_FIELDS),
        # 核对 CER 文本、错误数、字符数和百分数。
        ("cer", CER_FIELDS),
        # 核对 SVA 相似度、阈值和二元判断。
        ("sva", SVA_FIELDS),
    )
    # 连本次没有选择重算的旧指标也先验证，防止后面汇总误用。
    for metric_name, fieldnames in metric_schemas:
        # 定位该项指标的逐条文件。
        metric_csv = args.output_dir / f"{metric_name}.csv"
        # 已有文件必须通过严格前缀和内容检查。
        if metric_csv.is_file():
            # 使用统一的有效前缀校验器检查续算状态。
            validate_existing_metric_csv(
                # 同时传入当前清单、指标名和固定列结构。
                metric_csv, rows, metric_name, fieldnames
            )

    # 设置张量和模型使用的计算设备。
    device = torch.device(args.device)
    # 用户明确要求 GPU 时，CUDA 不可用就停止，避免静默换设备。
    if device.type == "cuda" and not torch.cuda.is_available():
        # 报错原因：用户明确要求 GPU 时，CUDA 不可用就停止，避免静默换设备。
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    # 评分前及汇总前都核对 WAV 格式；学生还逐文件检查字节身份。
    validate_generated_wavs(rows, expected_generated_wav_sha256)

    # 决定本次四项全算还是只运行指定指标。
    selected_metrics = (
        # 全算先完成 GPU 模型，再运行 CPU DNSMOS；分数计算口径不因顺序改变。
        ["utmos", "cer", "sva", "dnsmos"] if args.metric == "all" else [args.metric]
    )
    # 依次执行本次选择的客观评价项目。
    for metric in selected_metrics:
        # 日志标明开始哪一项，便于定位长任务进度。
        print(f"===== START {metric.upper()} =====")
        # 选择 UTMOS 自然度预测。
        if metric == "utmos":
            # 计算并逐条保存 UTMOS。
            evaluate_utmos(
                # 传入按 pair_id 排列的评估条目。
                rows,
                # 指定 UTMOS 的独立结果文件。
                args.output_dir / "utmos.csv",
                # 使用已经固定的 UTMOS 资源目录。
                args.utmos_dir,
                # 传入张量和模型使用的计算设备。
                device,
                # 使用本次明确的批大小，不改变逐条评分口径。
                args.batch_size,
            )
        # 选择 DNSMOS 质量预测。
        elif metric == "dnsmos":
            # 计算并逐条保存 DNSMOS 四路分数。
            evaluate_dnsmos(
                # 传入按 pair_id 排列的评估条目。
                rows,
                # 指定 DNSMOS 独立结果文件。
                args.output_dir / "dnsmos.csv",
                # 使用本次明确的批大小，不改变逐条评分口径。
                args.batch_size,
                # 限制 DNSMOS CPU 并行进程数。
                args.dnsmos_workers,
            )
        # 选择生成语音的字符错误率评价。
        elif metric == "cer":
            # 执行固定 ASR、文本清洗和字符编辑距离计算。
            evaluate_cer(
                # 传入按 pair_id 排列的评估条目。
                rows,
                # 保存逐条参考文本、识别文本与 CER 依据。
                args.output_dir / "cer.csv",
                # 传入张量和模型使用的计算设备。
                device,
                # 使用本次明确的批大小，不改变逐条评分口径。
                args.batch_size,
            )
        # 选择目标说话人验证准确率评价。
        elif metric == "sva":
            # 比较生成与目标参考 ECAPA 声纹，并逐条保存结果。
            evaluate_sva(
                # 传入按 pair_id 排列的评估条目。
                rows,
                # 保存逐条声纹相似度及验证结论。
                args.output_dir / "sva.csv",
                # 传入固定 ECAPA 评价模型目录。
                args.ecapa_dir,
                # 传入张量和模型使用的计算设备。
                device,
                # 使用本次明确的批大小，不改变逐条评分口径。
                args.batch_size,
            )

    # 只有四份 CSV 都存在时才汇总；单独跑第一项不会误报“完整评估”。
    required_csvs = [args.output_dir / f"{name}.csv" for name in ["utmos", "dnsmos", "cer", "sva"]]
    # 四项逐条文件均存在后，才尝试完整性检查和汇总。
    if all(path.is_file() for path in required_csvs):
        # The four metrics can take tens of minutes.  Recheck all bytes before
        # publishing a summary so a mid-run replacement cannot be certified.
        # 评分前及汇总前都核对 WAV 格式；学生还逐文件检查字节身份。
        validate_generated_wavs(rows, expected_generated_wav_sha256)
        # 组合检查下面各项：汇总发布前重新计算推理报告指纹。
        if (
            # 汇总发布前重新计算推理报告指纹。
            sha256_file(inference_report)
            # 报告必须仍与开始评分时完全一致。
            != metrics_identity["inference_report_sha256"]
            # 配对清单也不能在长时评分期间被替换。
            or sha256_file(pairs_csv) != metrics_identity["pairs_csv_sha256"]
        ):
            # 报错原因：配对清单也不能在长时评分期间被替换。
            raise RuntimeError(
                # 配对清单也不能在长时评分期间被替换。
                "Inference report or pairs.csv changed during objective evaluation"
            )
        # 在输入文件再次核对后发布客观指标汇总。
        write_summary(args, rows, pairs_csv, inference_report)
    # 此分支：当前单项计算结束；其余指标齐全前不发布完整成绩。
    else:
        # 当前单项计算结束；其余指标齐全前不发布完整成绩。
        print("Selected metric finished; full summary waits for all four metric CSV files.")


# 仅作为脚本直接运行时执行入口，导入模块时不启动任务。
if __name__ == "__main__":
    # 启动本脚本的完整处理流程。
    main()
