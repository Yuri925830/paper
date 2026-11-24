#!/usr/bin/env python3
# 生成固定第 100 轮学生的正式 8100 对音频，并严格核对模型、数据和输出身份。
"""Generate the formal 8,100-pair FastVoiceGrad epoch-100 student waveforms.

This entry point is intentionally separate from the VoiceGrad-30 teacher
inference module.  It loads the immutable epoch-100 student checkpoint, forms
``x_950`` from the source mel, performs exactly one learned transition to
``x_0``, and vocodes the result with the audited 2,500,000-step custom
HiFi-GAN.  Every material output is bound to a byte-level run identity so an
interrupted run can resume without silently mixing experiments.
"""

# 延迟解析类型注解，兼容项目使用的Python环境。
from __future__ import annotations

# 读取推理路径、设备、种子与诊断配对上限。
import argparse
# 在Linux上获取文件锁，防止同时训练或改动推理依赖。
import fcntl
# 核验模型、特征、配置和生成WAV的SHA-256指纹。
import hashlib
# 读写训练契约、运行身份和推理完成报告。
import json
# 执行原子文件替换和同步写盘，支持可靠中断恢复。
import os
# 读取Git状态和提交号，记录推理使用的代码版本。
import subprocess
# 记录生成阶段的实际耗时。
import time
# 让获取锁和释放锁成对执行，异常退出也能关闭句柄。
from contextlib import contextmanager
# 统一处理服务器模型、特征和输出的绝对路径。
from pathlib import Path
# 注明实验字典、配对列表、迭代器和函数返回值的结构。
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

# 读取预处理.npy特征，校验数值并转换音频数组。
import numpy as np
# 执行学生一步扩散推理和张量设备转换。
import torch
# 用神经网络模块类型描述冻结学生。
import torch.nn as nn
# 显示固定小批次推理进度。
from tqdm import tqdm

# 导入当前学生推理所需的fastvoicegrad.components接口。
from fastvoicegrad.components import (
    # 确认模型所有参数均关闭梯度。
    assert_frozen,
    # 加载并核验冻结的自训250万步HiFi-GAN。
    load_frozen_vocoder,
    # 兼容项目PyTorch版本读取可信检查点。
    load_torch_checkpoint,
    # 统一开关网络参数梯度，推理时全部关闭。
    set_requires_grad,
    # 以原始文件字节计算SHA-256。
    sha256_file,
)
# 读取学生的结构、BNF对齐和扩散设置，保持与训练一致。
from fastvoicegrad.config import PAPER_STUDENT_CONFIG, FastVoiceGradStudentConfig
# 导入复用训练中的源加噪和学生单步重建公式。
from fastvoicegrad.objectives import make_student_prediction
# 导入完整写好临时WAV后再发布目标文件。
from hifigan.tools.vocode import atomic_write_wav
# 复用教师音频规格及Matpool规范项目路径。
from voicegrad.config import PAPER_CONFIG, SERVER_PROJECT_ROOT
# 建立与教师和学生训练相同的余弦扩散日程。
from voicegrad.diffusion import VoiceGradDiffusion
# 导入把BNF时间轴对齐到源mel帧数。
from voicegrad.feature_alignment import align_bnf_to_mel
# 导入当前学生推理所需的voicegrad.infer_voicegrad30_paper_eval接口。
from voicegrad.infer_voicegrad30_paper_eval import (
    # 一组源语音、目标参考语音和输出WAV的结构化记录。
    EvaluationPair,
    # 按100×9×9的固定规则建立8100组客观转换配对。
    build_pairs,
    # 读取并检查特征矩阵的通道方向与形状。
    feature_matrix,
    # 按特征类型、说话人和句子定位特征文件。
    feature_path,
    # 保持固定源语音分组和4组批次边界。
    fixed_chunks,
    # 读取并验证测试清单中的100条语音。
    load_test_rows,
    # 读取JSON文件并检查顶层是字段字典。
    read_json_object,
    # 将配对记录按稳定格式序列化为CSV。
    render_pairs_csv,
    # 检查已存在音频是否可读且满足采样率、声道等要求。
    valid_existing_wav,
)
# 导入强制只在指定租用服务器运行项目推理。
from voicegrad.runtime_guard import assert_rented_server
# 导入按教师一致的U-Net和BNF下采样配置构建学生。
from voicegrad.train import build_model


# 固定正式评估学生为第100轮。
EXPECTED_STUDENT_EPOCH = 100
# 锁定第100轮对应123000次更新。
EXPECTED_STUDENT_GLOBAL_STEP = 123_000
# 锁定原epoch100学生权重的SHA-256。
EXPECTED_STUDENT_CHECKPOINT_SHA256 = (
    # 该固定 SHA-256 绑定历史正式实验身份，不能因换模型而静默替换。
    "eb9ad626160372f1b45b29d6c6d3f058fa8e553cfe48a5ad6a8c007e00d7ca27"
)
# 锁定原epoch100学生的训练契约指纹。
EXPECTED_STUDENT_RUN_CONTRACT_SHA256 = (
    # 该固定 SHA-256 绑定历史正式实验身份，不能因换模型而静默替换。
    "ea2eac524d970461e928cccf473c0ccf614d9873c6ca8b857f3c1d09ee18d3b1"
)

# 学生训练结果与契约的默认目录。
DEFAULT_STUDENT_DIR = (
    # 定位Matpool中原FastVoiceGrad学生实验目录。
    SERVER_PROJECT_ROOT / "exp/checkpoints/fastvoicegrad_vctk_text_unseen"
)
# 正式评估唯一允许的epoch_0100.pt路径。
DEFAULT_STUDENT_CHECKPOINT = DEFAULT_STUDENT_DIR / "epoch_0100.pt"
# 正式评估绑定的run_contract.json路径。
DEFAULT_STUDENT_RUN_CONTRACT = DEFAULT_STUDENT_DIR / "run_contract.json"
# 自训修改版HiFi-GAN的检查点目录。
DEFAULT_VOCODER_DIR = (
    # 定位学生训练依赖的自训修改版HiFi-GAN目录。
    SERVER_PROJECT_ROOT / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
)
# 冻结声码器的250万步归档文件。
DEFAULT_VOCODER_CHECKPOINT = DEFAULT_VOCODER_DIR / "generator_step_2500000.pt"
# 与最终声码器匹配的配置JSON。
DEFAULT_VOCODER_CONFIG = DEFAULT_VOCODER_DIR / "config.json"
# 声码器最终权重审核JSON。
DEFAULT_VOCODER_ATTESTATION = DEFAULT_VOCODER_DIR / "final_checkpoint_attestation.json"
# 将学生结果放入独立评估目录，避免覆盖教师。
DEFAULT_OUTPUT_DIR = (
    # 所有正式路径均以Matpool规范项目目录为根。
    SERVER_PROJECT_ROOT
    # 学生epoch100与自训声码器结果使用独立评估子目录。
    / "exp/evaluation/fastvoicegrad_text_unseen_epoch0100_custom_hifigan"
)

# 固定每个推理批次4组，保证随机分组可复现。
FORMAL_BATCH_SIZE = 4
# 固定基础种子1234。
FORMAL_SEED = 1234
# 运行身份格式使用版本1。
IDENTITY_FORMAT_VERSION = 1


# 将JSON规范化成稳定字节，键顺序变化不会改变内容指纹。
def canonical_json_bytes(value: Any) -> bytes:
    # 使用稳定 JSON 排序计算学生契约身份，不受字典插入顺序影响。
    """Return the stable JSON representation used by student checkpoints."""

    # 按固定JSON格式序列化，后续据此生成一致的内容指纹。
    return json.dumps(
        # 固定字典键顺序并去掉多余空格，中文保留UTF-8表示。
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    # 将稳定JSON文本编码为参与指纹计算的UTF-8字节。
    ).encode("utf-8")


# 计算规范化JSON的SHA-256，核对结构化实验身份。
def canonical_json_sha256(value: Any) -> str:
    # 对规范化JSON字节计算64字符SHA-256。
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# 得到发布到磁盘的缩进JSON精确字节，包含末尾换行。
def rendered_json_bytes(value: Mapping[str, object]) -> bytes:
    # 生成运行身份文件实际写入的格式化 JSON 字节，字节指纹可追溯。
    """Return the exact pretty JSON bytes used for run-identity publication."""

    # 发布JSON保留中文、两空格缩进和末尾换行，字节格式固定。
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


# 先写临时文件并同步磁盘，再原子替换正式文件。
def atomic_write_bytes(path: Path, payload: bytes) -> None:
    # 先保证目标文件的上级目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 正式写入前使用的同目录临时文件。
    temporary = path.with_name(path.name + ".tmp")
    # 二进制写临时文件，精确保留已计算指纹的字节。
    with temporary.open("wb") as handle:
        # 把本次完整文件内容写入临时目标。
        handle.write(payload)
        # 将Python写缓冲提交到操作系统。
        handle.flush()
        # 要求日志或身份文件落盘，降低中断时只写一半的风险。
        os.fsync(handle.fileno())
    # 临时文件写完后原子替换正式文件。
    os.replace(temporary, path)


# 将报告编码为固定JSON字节后原子写入。
def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    # 先写临时文件并同步磁盘，再原子替换正式文件。
    atomic_write_bytes(path, rendered_json_bytes(value))


# 要求正式推理使用已提交且干净的源码，并记录Git提交。
def git_head(project_root: Path) -> str:
    # 正式推理前要求源码已经提交且工作区干净，使运行身份能对应明确版本。
    """Require a clean committed worktree for a formal inference identity."""

    # 本次运行完成状态；仅全量8100组可标记正式完成。
    status = subprocess.run(
        # 读取简洁Git修改列表，正式推理要求该列表为空。
        ["git", "status", "--porcelain"],
        # Git命令明确在当前项目根目录执行。
        cwd=str(project_root),
        # 外部命令失败立即抛错，不能在版本不明时继续正式评估。
        check=True,
        # 把Git输出作为文本读取，方便检查空白和提交号。
        text=True,
        # 捕获命令的标准输出，用于验证源码状态。
        stdout=subprocess.PIPE,
        # 捕获错误输出，让Git查询失败可以定位原因。
        stderr=subprocess.PIPE,
    )
    # 存在未提交源码改动时，无法用单个Git提交准确表示本次实现。
    if status.stdout.strip():
        # 拒绝继续执行要求正式推理使用已提交且干净的源码，并记录Git提交：当前输入没有通过校验。
        raise RuntimeError(
            # 明确报告正式推理要求源码工作区干净且已提交。
            "Formal FastVoiceGrad inference requires a clean committed Git worktree"
        )
    # 当前检查或转换得到的返回记录。
    result = subprocess.run(
        # 取得当前提交完整哈希并写进运行身份。
        ["git", "rev-parse", "HEAD"],
        # Git命令明确在当前项目根目录执行。
        cwd=str(project_root),
        # 外部命令失败立即抛错，不能在版本不明时继续正式评估。
        check=True,
        # 把Git输出作为文本读取，方便检查空白和提交号。
        text=True,
        # 捕获命令的标准输出，用于验证源码状态。
        stdout=subprocess.PIPE,
        # 捕获错误输出，让Git查询失败可以定位原因。
        stderr=subprocess.PIPE,
    )
    # 返回去掉换行的Git提交编号。
    return result.stdout.strip()


# 检查元数据对象确实是字段映射，拒绝格式不符的输入。
def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    # 元数据必须按字段组织，其他格式无法逐项核验。
    if not isinstance(value, Mapping):
        # 拒绝继续执行检查元数据对象确实是字段映射，拒绝格式不符的输入：当前输入没有通过校验。
        raise RuntimeError(f"{label} must be a mapping")
    # 格式正确后保留原映射返回，不修改其内容。
    return value


# 逐字段核对实际值与该实验锁定的期望值。
def _require_exact(mapping: Mapping[str, Any], key: str, expected: object, label: str) -> None:
    # 当前文件或元数据实际值。
    actual = mapping.get(key)
    # 实际字段或指纹与锁定值不同，停止混用实验。
    if actual != expected:
        # 拒绝继续执行逐字段核对实际值与该实验锁定的期望值：当前输入没有通过校验。
        raise RuntimeError(
            # 打印字段所属分组、实际值与要求值，便于定位身份差异。
            f"{label}.{key} mismatch: actual={actual!r}, expected={expected!r}"
        )


# 核对原学生训练契约的指纹、协议、数据数量和模型设置。
def validate_student_run_contract(
    # 传入记录模型训练设置、数据和依赖的运行契约。
    run_contract: Mapping[str, Any],
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 已登记的正确SHA-256指纹。
    expected_sha256: str = EXPECTED_STUDENT_RUN_CONTRACT_SHA256,
# 结束 validate_student_run_contract 的参数声明；返回类型为 str，具体内容由下方实现给出。
) -> str:
    # 校验赋予学生权重具体实验含义的训练契约。
    """Validate the training contract that gives the weights their meaning."""

    # 从当前训练契约重新计算的指纹。
    actual_sha256 = canonical_json_sha256(run_contract)
    # 训练契约整体内容被改动或选错时立即拒绝。
    if actual_sha256 != expected_sha256:
        # 拒绝继续执行核对原学生训练契约的指纹、协议、数据数量和模型设置：当前输入没有通过校验。
        raise RuntimeError(
            # 报告学生训练契约文件身份不匹配。
            "Student run-contract identity mismatch: "
            # 同时报告契约实际和预期 SHA-256。
            f"actual={actual_sha256}, expected={expected_sha256}"
        )

    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 记录模型训练设置、数据和依赖的运行契约。
        run_contract,
        # 核对运行契约的实验名称字段。
        "experiment",
        # 只接受 VCTK text-unseen 的 ACDD 学生训练契约。
        "FastVoiceGrad VCTK text-unseen ACDD student",
        # 异常消息中标明正在检查运行契约。
        "run_contract",
    )
    # 训练契约保存的数据划分与样本数量。
    data = _require_mapping(run_contract.get("data"), "run_contract.data")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(data, "protocol", "text-unseen", "run_contract.data")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(data, "test_rows", 100, "run_contract.data")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(data, "test_speakers", 10, "run_contract.data")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 训练契约保存的数据划分与样本数量。
        data,
        # 本次学生训练预定总更新次数。
        "total_student_updates",
        # 锁定第100轮对应123000次更新。
        EXPECTED_STUDENT_GLOBAL_STEP,
        # 异常消息中标明数据划分与更新次数分组。
        "run_contract.data",
    )

    # 训练契约保存的论文设定部分。
    paper = _require_mapping(run_contract.get("paper"), "run_contract.paper")
    # 论文中明确公开的设定字段。
    explicit = _require_mapping(
        # 取出训练契约中论文明确设定的字段集合，并注明错误来源。
        paper.get("paper_explicit"), "run_contract.paper.paper_explicit"
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(explicit, "epochs", EXPECTED_STUDENT_EPOCH, "paper_explicit")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(explicit, "student_steps", 1, "paper_explicit")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(explicit, "student_start_step", 950, "paper_explicit")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(explicit, "diffusion_steps", 1000, "paper_explicit")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(explicit, "teacher_epoch", 500, "paper_explicit")

    # 本项目登记的未公开实现选择。
    choices = _require_mapping(
        # 读取训练时登记的未公开实现选择。
        run_contract.get("paper_undisclosed_declared_choices"),
        # 异常消息中标明论文未公开而显式登记的实现选择。
        "run_contract.paper_undisclosed_declared_choices",
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # BNF到mel的长度对齐必须与原学生使用的线性规则一致。
        choices, "bnf_alignment", "legacy_length_linear", "declared_choices"
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # BNF下采样必须保留原抽点规则。
        choices, "bnf_downsampling", "legacy_decimate", "declared_choices"
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(choices, "seed", FORMAL_SEED, "declared_choices")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(choices, "expected_vocoder_step", 2_500_000, "declared_choices")

    # 学生训练时冻结依赖的权重与配置记录。
    dependencies = _require_mapping(
        # 取得教师与声码器等冻结依赖的历史身份记录。
        run_contract.get("dependencies"), "run_contract.dependencies"
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 声码器训练进度必须是该学生绑定的250万步。
        dependencies, "vocoder_global_step", 2_500_000, "run_contract.dependencies"
    )
    # 返回从当前训练契约重新计算的指纹。
    return actual_sha256


# 逐个核对学生训练时登记的历史源码字节指纹。
def validate_historical_source_hashes(
    # 传入Matpool上的规范项目根目录。
    project_root: Path, run_contract: Mapping[str, Any]
# 结束 validate_historical_source_hashes 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # 核对契约登记的训练源码没有在权重保存后改变。
    """Ensure training-time source files listed by the contract are unchanged."""

    # 训练时的软件与源码运行记录。
    runtime = _require_mapping(run_contract.get("runtime"), "run_contract.runtime")
    # 检查点或训练契约保存的历史身份信息。
    saved = _require_mapping(
        # 从训练运行记录提取逐文件源码指纹。
        runtime.get("source_sha256"), "run_contract.runtime.source_sha256"
    )
    # 历史契约缺少源码指纹就不能证明实现一致性。
    if not saved:
        # 拒绝继续执行逐个核对学生训练时登记的历史源码字节指纹：当前输入没有通过校验。
        raise RuntimeError("Student run contract has no source fingerprints")
    # 逐个检查训练时登记的源码相对路径与指纹。
    for relative, expected in saved.items():
        # 当前需要核验或写入的具体文件路径。
        path = project_root / str(relative)
        # 当前文件或元数据实际值。
        actual = sha256_file(path)
        # 实际字段或指纹与锁定值不同，停止混用实验。
        if actual != expected:
            # 拒绝继续执行逐个核对学生训练时登记的历史源码字节指纹：当前输入没有通过校验。
            raise RuntimeError(
                # 指出保存学生后被修改的训练源码相对路径。
                f"Training source changed after checkpoint creation: {relative}: "
                # 报告当前值与训练登记值的差异。
                f"actual={actual}, expected={expected}"
            )


# 检查学生权重归档格式、轮次、步数、契约和参数数值。
def validate_student_checkpoint_payload(
    # 传入待核验的模型检查点。
    checkpoint: Mapping[str, Any],
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 传入检查点绑定的训练契约规范化指纹。
    run_contract_sha256: str,
    # 允许加载的学生训练轮次100。
    expected_epoch: int = EXPECTED_STUDENT_EPOCH,
    # 允许加载的学生优化器步数123000。
    expected_global_step: int = EXPECTED_STUDENT_GLOBAL_STEP,
# 结束 validate_student_checkpoint_payload 的参数声明；返回类型为 Mapping[str, torch.Tensor]，具体内容由下方实现给出。
) -> Mapping[str, torch.Tensor]:
    # 检查学生归档元数据不可变身份与张量有限性。
    """Validate immutable student-checkpoint metadata and tensor health."""

    # 学生归档检查点必须且只能具备的顶层字段。
    expected_top_level = {
        # 检查点序列化格式版本。
        "format_version",
        # 区分推理权重档案与训练恢复状态的类型标记。
        "checkpoint_type",
        # 当前完整训练轮次。
        "epoch",
        # 检查点记录的累计优化器更新步数。
        "global_step",
        # 检查点绑定的训练契约规范化指纹。
        "run_contract_sha256",
        # 该轮损失摘要，随权重归档保存。
        "epoch_losses",
        # 学生网络参数字典。
        "student",
    }
    # 顶层字段必须完全匹配学生精简归档格式。
    if set(checkpoint) != expected_top_level:
        # 拒绝继续执行检查学生权重归档格式、轮次、步数、契约和参数数值：当前输入没有通过校验。
        raise RuntimeError(
            # 报告学生归档顶层字段结构不一致。
            "Student checkpoint top-level schema mismatch: "
            # 列出当前与允许的归档字段名集合。
            f"actual={sorted(checkpoint)}, expected={sorted(expected_top_level)}"
        )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(checkpoint, "format_version", 3, "student_checkpoint")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 只接受完整epoch归档的学生权重文件，不把训练中间状态当正式结果。
        checkpoint, "checkpoint_type", "student_weights_only", "student_checkpoint"
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(checkpoint, "epoch", expected_epoch, "student_checkpoint")
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 检查内部更新步数等于原100轮对应的123000。
        checkpoint, "global_step", expected_global_step, "student_checkpoint"
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 待核验的模型检查点。
        checkpoint,
        # 检查点绑定的训练契约规范化指纹。
        "run_contract_sha256",
        # 检查点绑定的训练契约规范化指纹。
        run_contract_sha256,
        # 异常消息中标明学生归档身份分组。
        "student_checkpoint",
    )

    # 检查点最后一轮保存的分项损失。
    epoch_losses = _require_mapping(
        # 取出检查点最后完整轮的损失并检查字段映射类型。
        checkpoint.get("epoch_losses"), "student_checkpoint.epoch_losses"
    )
    # 归档必须含最后完整epoch的损失记录，不能接受空字段。
    if not epoch_losses:
        # 拒绝继续执行检查学生权重归档格式、轮次、步数、契约和参数数值：当前输入没有通过校验。
        raise RuntimeError("Student checkpoint epoch_losses is empty")
    # 逐项核验保存的损失值类型和数值有效性。
    for key, value in epoch_losses.items():
        # 排除布尔值或字符串假扮数值损失。
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # 拒绝继续执行检查学生权重归档格式、轮次、步数、契约和参数数值：当前输入没有通过校验。
            raise RuntimeError(f"Invalid epoch loss {key!r}: {value!r}")
        # 损失记录出现NaN/Inf说明归档不能作为正常完成结果。
        if not np.isfinite(float(value)):
            # 拒绝继续执行检查学生权重归档格式、轮次、步数、契约和参数数值：当前输入没有通过校验。
            raise RuntimeError(f"Non-finite epoch loss {key!r}: {value!r}")

    # 从检查点提取的学生参数名到张量映射。
    state = _require_mapping(checkpoint.get("student"), "student_checkpoint.student")
    # 空参数字典无法构建有效学生模型。
    if not state:
        # 拒绝继续执行检查学生权重归档格式、轮次、步数、契约和参数数值：当前输入没有通过校验。
        raise RuntimeError("Student checkpoint has an empty state dict")
    # 逐层检查学生参数是否为有效张量。
    for key, value in state.items():
        # 参数值必须是Tensor，不能混入其他对象。
        if not isinstance(value, torch.Tensor):
            # 拒绝继续执行检查学生权重归档格式、轮次、步数、契约和参数数值：当前输入没有通过校验。
            raise RuntimeError(f"Student state {key!r} is not a tensor")
        # 浮点和复数参数额外检查NaN/Inf，整数缓冲区不需此项。
        if (value.is_floating_point() or value.is_complex()) and not bool(
            # 要求当前参数所有元素均为有限数。
            torch.isfinite(value).all()
        ):
            # 拒绝继续执行检查学生权重归档格式、轮次、步数、契约和参数数值：当前输入没有通过校验。
            raise RuntimeError(f"Student state {key!r} contains NaN or Inf")
    # 返回已核验的学生参数映射供严格加载。
    return state


# 加载唯一登记的epoch100学生并冻结全部参数供正式推理。
def load_student_checkpoint(
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 传入Matpool上的规范项目根目录。
    project_root: Path,
    # 传入学生或声码器权重文件路径。
    checkpoint_path: Path,
    # 传入学生训练时保存的运行契约路径。
    run_contract_path: Path,
    # 传入本次推理的计算设备。
    device: torch.device,
    # 学生模型的固定结构与推理配置。
    config: FastVoiceGradStudentConfig = PAPER_STUDENT_CONFIG,
    # 预先审核的学生权重字节指纹。
    expected_checkpoint_sha256: str = EXPECTED_STUDENT_CHECKPOINT_SHA256,
    # 预先审核的训练契约指纹。
    expected_run_contract_sha256: str = EXPECTED_STUDENT_RUN_CONTRACT_SHA256,
    # 是否核对检查点训练时绑定的源代码。
    verify_historical_sources: bool = True,
# 结束 load_student_checkpoint 的参数声明；返回类型为 Tuple[nn.Module, Dict[str, Any], Dict[str, Any], str]，具体内容由下方实现给出。
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], str]:
    # 只加载允许的第 100 轮学生归档，并冻结参数用于推理。
    """Load the one allowed epoch-100 student and freeze it for inference."""

    # 原正式评估只允许第100轮归档，防止误用latest或较早权重。
    if checkpoint_path.name != "epoch_0100.pt":
        # 拒绝继续执行加载唯一登记的epoch100学生并冻结全部参数供正式推理：当前输入没有通过校验。
        raise RuntimeError(
            # 报告学生归档文件名没有指向固定第 100 轮。
            f"Formal student checkpoint must be named epoch_0100.pt: {checkpoint_path}"
        )
    # 从当前学生权重字节重新计算的SHA-256。
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    # 即便文件名相同，也必须核对原始权重字节完全一致。
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        # 拒绝继续执行加载唯一登记的epoch100学生并冻结全部参数供正式推理：当前输入没有通过校验。
        raise RuntimeError(
            # 报告学生权重归档实际字节不符。
            "Student checkpoint byte identity mismatch: "
            # 报告学生权重实际和允许的内容指纹。
            f"actual={actual_checkpoint_sha256}, expected={expected_checkpoint_sha256}"
        )

    # 记录模型训练设置、数据和依赖的运行契约。
    run_contract = read_json_object(run_contract_path)
    # 核验通过的训练契约内容指纹。
    contract_sha256 = validate_student_run_contract(
        # 按锁定训练契约指纹验证，而非只看文件名。
        run_contract, expected_sha256=expected_run_contract_sha256
    )
    # 启用严格核验时，继续核对训练当时登记的源码。
    if verify_historical_sources:
        # 逐个核对学生训练时登记的历史源码字节指纹。
        validate_historical_source_hashes(project_root, run_contract)

    # 待核验的模型检查点。
    checkpoint = load_torch_checkpoint(checkpoint_path)
    # 从检查点提取的学生参数名到张量映射。
    state = validate_student_checkpoint_payload(
        # 检查模型内部绑定的训练契约与刚核实的JSON一致。
        checkpoint, run_contract_sha256=contract_sha256
    )
    # 与教师相同结构的学生网络实例。
    model = build_model(config.bnf_downsampling)
    # 要求全部参数名称和形状一一对应，不允许漏层。
    model.load_state_dict(state, strict=True)
    # 将学生放到指定设备并切换推理模式。
    model.to(device).eval()
    # 关闭学生全部参数梯度，正式推理不更新模型。
    set_requires_grad(model, False)
    # 再次确认冻结完整，避免错误保留训练参数。
    assert_frozen(model, "FastVoiceGrad student")
    # 同时返回网络、权重元数据、训练契约和本次核实的字节指纹。
    return model, checkpoint, run_contract, actual_checkpoint_sha256


# 确认推理声码器与学生训练使用的声码器文件逐字节一致。
def validate_vocoder_dependency_identity(
    # 传入记录模型训练设置、数据和依赖的运行契约。
    run_contract: Mapping[str, Any],
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 传入学生或声码器权重文件路径。
    checkpoint_path: Path,
    # 传入声码器结构配置文件路径。
    config_path: Path,
    # 传入声码器最终检查点审核文件路径。
    attestation_path: Path,
# 结束 validate_vocoder_dependency_identity 的参数声明；返回类型为 Dict[str, str]，具体内容由下方实现给出。
) -> Dict[str, str]:
    # 推理声码器必须逐文件匹配学生训练时使用的版本。
    """Bind the inference vocoder to the exact one used for student training."""

    # 学生训练时冻结依赖的权重与配置记录。
    dependencies = _require_mapping(
        # 取得教师与声码器等冻结依赖的历史身份记录。
        run_contract.get("dependencies"), "run_contract.dependencies"
    )
    # 当前文件或元数据实际值。
    actual = {
        # 记录本次冻结声码器的最终权重文件的SHA-256字节指纹。
        "vocoder_checkpoint_sha256": sha256_file(checkpoint_path),
        # 记录声码器的结构与音频配置的SHA-256字节指纹。
        "vocoder_config_sha256": sha256_file(config_path),
        # 记录声码器250万步最终检查点的审核记录的SHA-256字节指纹。
        "vocoder_attestation_sha256": sha256_file(attestation_path),
    }
    # 该字段或指纹已登记的期望值。
    expected = {
        # 记录本次冻结声码器的最终权重文件的SHA-256字节指纹。
        "vocoder_checkpoint_sha256": dependencies.get("vocoder_checkpoint_sha256"),
        # 记录声码器的结构与音频配置的SHA-256字节指纹。
        "vocoder_config_sha256": dependencies.get("vocoder_config_sha256"),
        # 记录声码器250万步最终检查点的审核记录的SHA-256字节指纹。
        "vocoder_attestation_sha256": dependencies.get("vocoder_attestation_sha256"),
    }
    # 实际字段或指纹与锁定值不同，停止混用实验。
    if actual != expected:
        # 拒绝继续执行确认推理声码器与学生训练使用的声码器文件逐字节一致：当前输入没有通过校验。
        raise RuntimeError(
            # 指出推理声码器与学生训练时依赖不一致。
            f"Inference vocoder differs from the student training contract: "
            # 报告当前值与训练登记值的差异。
            f"actual={actual}, expected={expected}"
        )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 学生训练时冻结依赖的权重与配置记录。
        dependencies,
        # 本次冻结声码器的最终权重文件。
        "vocoder_checkpoint",
        # 将当前权重绝对路径与契约记录进行字符串精确比较。
        str(checkpoint_path),
        # 异常消息中标明冻结依赖的登记信息。
        "run_contract.dependencies",
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 学生训练时冻结依赖的权重与配置记录。
        dependencies,
        # 声码器的结构与音频配置。
        "vocoder_config",
        # 精确核对声码器配置绝对路径。
        str(config_path),
        # 异常消息中标明冻结依赖的登记信息。
        "run_contract.dependencies",
    )
    # 逐字段核对实际值与该实验锁定的期望值。
    _require_exact(
        # 学生训练时冻结依赖的权重与配置记录。
        dependencies,
        # 声码器250万步最终检查点的审核记录。
        "vocoder_attestation",
        # 精确核对声码器审核文件绝对路径。
        str(attestation_path),
        # 异常消息中标明冻结依赖的登记信息。
        "run_contract.dependencies",
    )
    # 把已经核对过的实际声码器指纹交给运行身份记录。
    return actual


# 核对当前数据拆分和预处理指纹与学生训练历史一致。
def validate_active_data_contract(
    # 传入Matpool上的规范项目根目录。
    project_root: Path, run_contract: Mapping[str, Any]
# 结束 validate_active_data_contract 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # 核对活动数据划分及特征统计量与训练历史完全一致。
    """Check active split and published data identities against training history."""

    # Imported lazily so simple checkpoint/unit tests do not import the training loop.
    # 导入当前学生推理所需的fastvoicegrad.train接口。
    from fastvoicegrad.train import validate_data_protocol

    # 当前数据目录实际发布的划分和指纹。
    active = validate_data_protocol(project_root, PAPER_STUDENT_CONFIG)
    # 检查点或训练契约保存的历史身份信息。
    saved = _require_mapping(run_contract.get("data"), "run_contract.data")
    # 按下面列出的字段逐一核对数据或解析命令行路径。
    for key in (
        # 当前数据清单声明的划分协议。
        "protocol",
        # 实际训练语音条数。
        "train_rows",
        # 训练使用100个说话人。
        "train_speakers",
        # 测试集语音条数。
        "test_rows",
        # 测试说话人数。
        "test_speakers",
        # 整个已发布数据清单的语音总数。
        "all_rows",
        # 划分协议文件字节指纹。
        "split_protocol_file_sha256",
        # 预处理产物及协议内容指纹。
        "fingerprints",
    ):
        # 当前数据字段与训练历史不同会破坏可比性，立即拒绝。
        if active.get(key) != saved.get(key):
            # 拒绝继续执行核对当前数据拆分和预处理指纹与学生训练历史一致：当前输入没有通过校验。
            raise RuntimeError(f"Active data contract differs for {key}")


# 为影响转换与客观指标的所有测试特征和原始波形建立指纹。
def collect_test_input_hashes(
    # 传入Matpool上的规范项目根目录。
    project_root: Path, rows: Sequence[Mapping[str, str]]
# 结束 collect_test_input_hashes 的参数声明；返回类型为 Dict[str, str]，具体内容由下方实现给出。
) -> Dict[str, str]:
    # 为可能影响学生推理和客观评估的所有源码文件登记内容指纹。
    """Hash every file that can affect student inference or objective evaluation."""

    # 源mel、BNF、说话人向量和统计量所在的数据目录。
    data_root = project_root / "data"
    # 去重后的所有测试输入文件路径。
    paths: Dict[str, Path] = {}
    # 遍历100条测试原语音，收集每条所需输入。
    for row in rows:
        # 当前测试原语音所属说话人。
        speaker = row["speaker"]
        # 当前测试原语音句子编号。
        utterance = row["utt_id"]
        # mel提供声学输入，BNF提供内容，说话人向量提供目标身份。
        for feature in ("mel", "bnf", "spk_emb"):
            # 当前需要核验或写入的具体文件路径。
            path = feature_path(data_root, feature, speaker, utterance)
            # 以项目相对路径去重，避免同一特征重复计入。
            paths[str(path.relative_to(project_root))] = path
        # 当前测试原始波形的绝对路径。
        wav = Path(row["wav22k_path"])
        # 清单中相对WAV路径需要按规范项目根目录解释。
        if not wav.is_absolute():
            # 当前测试原始波形的绝对路径。
            wav = project_root / wav
        # 当前测试原始波形的绝对路径。
        wav = wav.resolve()
        # 保存解析后的原始波形路径，用于客观指标输入指纹。
        paths[str(wav)] = wav
    # 按稳定键顺序逐文件哈希，生成可重复的输入集合指纹。
    return {key: sha256_file(path) for key, path in sorted(paths.items())}


# 先把源mel扩散到950，再让学生只执行一次950→0重建。
def student_one_step(
    # 传入冻结的FastVoiceGrad学生噪声预测器。
    student: nn.Module,
    # 传入与训练一致的余弦扩散系数表。
    diffusion: VoiceGradDiffusion,
    # 传入源语音标准化mel张量[B,80,T]。
    source_mel: torch.Tensor,
    # 传入目标说话人条件张量[B,256]。
    target_speaker: torch.Tensor,
    # 传入与源mel时间轴对齐的内容特征[B,144,T]。
    source_bnf: torch.Tensor,
    # 传入构造x950的独立标准高斯噪声。
    source_noise: torch.Tensor,
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 单步学生固定的起始时刻950。
    start_step: int = 950,
# 结束 student_one_step 的参数声明；返回类型为 torch.Tensor，具体内容由下方实现给出。
) -> torch.Tensor:
    # 先给源梅尔加噪到 950，再只调用一次学生直接预测干净梅尔。
    """Run q(x_950|x_source) followed by exactly one learned 950-to-0 step."""

    # 正式学生从950起步，拒绝其它扩散时刻改变推理任务。
    if start_step != PAPER_STUDENT_CONFIG.student_start_step:
        # 拒绝继续执行先把源mel扩散到950，再让学生只执行一次950→0重建：当前输入没有通过校验。
        raise RuntimeError(
            # 报告收到的错误一步起点，并给出固定要求 950。
            f"Formal student start step must be 950, received {start_step}"
        )
    # 复用训练公式执行源加噪和学生一步重建，额外带噪输入无需保留。
    converted, _ = make_student_prediction(
        # 冻结的FastVoiceGrad学生噪声预测器。
        student,
        # 与训练一致的余弦扩散系数表。
        diffusion,
        # 源语音标准化mel张量[B,80,T]。
        source_mel,
        # 目标说话人条件张量[B,256]。
        target_speaker,
        # 与源mel时间轴对齐的内容特征[B,144,T]。
        source_bnf,
        # 传入经过校验的SK=950。
        start_step=start_step,
        # 使用外部按固定种子生成的噪声，保证中断续跑结果一致。
        source_noise=source_noise,
    )
    # 返回一次学生前向得到的标准化目标mel。
    return converted


# 限定学生专属目录，防止覆盖教师结果或混入诊断输出。
def validate_output_directory(output_dir: Path, *, formal: bool) -> None:
    # 解析后的规范绝对输出目录。
    resolved = output_dir.resolve()
    # 输出路径任一层属于教师目录时拒绝写入。
    if any(part.startswith("voicegrad30_") for part in resolved.parts):
        # 拒绝继续执行限定学生专属目录，防止覆盖教师结果或混入诊断输出：当前输入没有通过校验。
        raise RuntimeError(
            # 打印被拒绝的教师目录，避免学生结果覆盖教师结果。
            f"Student inference refuses every VoiceGrad-30 teacher directory: {resolved}"
        )
    # 完整正式推理只能使用已锁定的学生输出目录。
    if formal and resolved != DEFAULT_OUTPUT_DIR.resolve():
        # 拒绝继续执行限定学生专属目录，防止覆盖教师结果或混入诊断输出：当前输入没有通过校验。
        raise RuntimeError(
            # 打印正式学生允许目录与实际目录。
            f"Formal student output must be {DEFAULT_OUTPUT_DIR.resolve()}, got {resolved}"
        )


# 把互斥锁的获取和释放包装成with上下文。
@contextmanager
# 按统一顺序取得流程互斥锁，避免训练和推理同时改动依赖。
def acquire_inference_locks(project_root: Path) -> Iterator[None]:
    # 按训练端相同的全局顺序取得依赖与输出锁，避免并发混写或死锁。
    """Hold dependency and output locks in the same global order as training."""

    # 保存流程互斥锁的目录。
    lock_dir = project_root / "exp/locks"
    # 创建互斥锁目录，允许首次启动。
    lock_dir.mkdir(parents=True, exist_ok=True)
    # 按统一顺序需要获取的声码器与学生运行锁。
    specifications = (
        (
            # 整个自训HiFi-GAN流水线锁，防止依赖正在被更新。
            lock_dir / "text_unseen_custom_hifigan_pipeline.lock",
            # 锁被占用时提示声码器流水线正在运行。
            "The custom HiFi-GAN pipeline is active",
        ),
        (
            # 声码器训练锁，确保合成时声码器权重稳定。
            lock_dir / "text_unseen_custom_hifigan_train.lock",
            # 锁被占用时提示声码器训练正在运行。
            "Custom HiFi-GAN training is active",
        ),
        (
            # 学生训练锁，避免同时训练和正式评估改动依赖。
            lock_dir / "text_unseen_fastvoicegrad_student.lock",
            # 锁被占用时提示学生训练正在运行。
            "FastVoiceGrad student training is active",
        ),
        (
            # 学生推理锁，防止两个任务同时写同一输出目录。
            lock_dir / "text_unseen_fastvoicegrad_student_inference.lock",
            # 锁被占用时提示已有学生推理进程。
            "FastVoiceGrad student inference is already active",
        ),
    )
    # 已经成功取得的锁文件句柄。
    handles = []
    # 保证检查或加锁中途失败时仍走清理流程。
    try:
        # 按全局固定顺序逐个获取锁，避免两个流程互相等待。
        for path, message in specifications:
            # 当前文件或互斥锁的打开句柄。
            handle = path.open("a+")
            # 保证检查或加锁中途失败时仍走清理流程。
            try:
                # 尝试非阻塞独占锁；其他任务占用时立即报错。
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # 锁已被其他训练或推理进程持有时进入冲突处理。
            except BlockingIOError as error:
                # 关闭未取得锁或已完成使用的文件句柄。
                handle.close()
                # 将底层锁冲突转为明确的实验流程占用消息。
                raise RuntimeError(message) from error
            # 保存成功取得的锁，退出时按反序释放。
            handles.append(handle)
        # 所有依赖都已锁定后，允许调用方执行推理。
        yield
    # 无论正常结束还是发生异常，都释放已获取的锁。
    finally:
        # 以获取时相反的顺序关闭锁文件。
        for handle in reversed(handles):
            # 关闭未取得锁或已完成使用的文件句柄。
            handle.close()


# 建立完成报告必须包含的系统、数据和权重身份字段。
def _report_required_fields(
    # 传入将本次模型、数据、代码和输出绑定在一起的身份记录。
    run_identity: Mapping[str, object],
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 传入已发布运行身份文件的字节指纹。
    run_identity_sha256: str,
    # 传入8100组转换配对清单的字节指纹。
    pairs_csv_sha256: str,
    # 传入本次实际选择的转换组数。
    selected_pair_count: int,
# 结束 _report_required_fields 的参数声明；返回类型为 Dict[str, object]，具体内容由下方实现给出。
) -> Dict[str, object]:
    # 汇集当前函数核验或生成的实验元数据字段。
    return {
        # 记录本次运行完成状态；仅全量8100组可标记正式完成。
        "status": (
            # 8100 对全量正式推理的完成状态。
            "formal_8100_complete"
            # 只有选择了全部8100组才能发布正式完成状态。
            if selected_pair_count == 8100
            # 少量配对只标记诊断完成，不能冒充论文8100组正式评估。
            else "diagnostic_partial_complete"
        ),
        # 记录系统机器标识，学生固定为fastvoicegrad。
        "system_id": "fastvoicegrad",
        # 记录报告中显示的学生名称与K=1、epoch100。
        "system_label": "FastVoiceGrad (K=1) epoch 100",
        # 记录明确标记student，防止与teacher评估混用。
        "model_role": "student",
        # 记录当前实验的text-unseen划分协议。
        "data_protocol": "text-unseen",
        # 记录声明采用严格的text-unseen测试隔离。
        "strict_text_unseen_protocol": True,
        # 记录论文客观评估要求的8100组转换。
        "paper_pair_count": 8100,
        # 记录本次实际选择的转换组数。
        "selected_pair_count": selected_pair_count,
        # 记录初始化学生的教师已训练500轮。
        "teacher_epoch": 500,
        # 记录学生已完成的训练轮次。
        "student_epoch": EXPECTED_STUDENT_EPOCH,
        # 记录学生已执行的优化器更新次数。
        "student_global_step": EXPECTED_STUDENT_GLOBAL_STEP,
        # 记录学生反向扩散次数K=1。
        "student_steps": 1,
        # 记录学生从SK=950开始单步重建。
        "student_start_step": 950,
        # 记录使用自训修改版HiFi-GAN。
        "vocoder_kind": "custom",
        # 记录声码器已完成2500000次训练更新。
        "vocoder_global_step": 2_500_000,
        # 记录加载的学生epoch100权重路径的SHA-256字节指纹。
        "student_checkpoint_sha256": run_identity["student_checkpoint_sha256"],
        # 记录赋予该学生检查点实验含义的训练契约的SHA-256字节指纹。
        "student_run_contract_sha256": run_identity[
            # 学生训练契约指纹。
            "student_run_contract_sha256"
        ],
        # 记录本次冻结声码器的最终权重文件的SHA-256字节指纹。
        "vocoder_checkpoint_sha256": run_identity["vocoder_checkpoint_sha256"],
        # 记录声码器的结构与音频配置的SHA-256字节指纹。
        "vocoder_config_sha256": run_identity["vocoder_config_sha256"],
        # 记录声码器250万步最终检查点的审核记录的SHA-256字节指纹。
        "vocoder_attestation_sha256": run_identity[
            # 声码器最终归档证明指纹。
            "vocoder_attestation_sha256"
        ],
        # 记录100条测试原语音的清单的SHA-256字节指纹。
        "test_manifest_sha256": run_identity["test_manifest_sha256"],
        # 记录训练/测试说话人与句子的拆分协议的SHA-256字节指纹。
        "split_protocol_sha256": run_identity["split_protocol_sha256"],
        # 记录训练集逐mel通道的均值的SHA-256字节指纹。
        "mel_mean_sha256": run_identity["mel_mean_sha256"],
        # 记录训练集逐mel通道的标准差的SHA-256字节指纹。
        "mel_std_sha256": run_identity["mel_std_sha256"],
        # 记录8100组转换配对清单的字节指纹。
        "pairs_csv_sha256": pairs_csv_sha256,
        # 记录已发布运行身份文件的字节指纹。
        "run_identity_sha256": run_identity_sha256,
    }


# 核验已有完成报告的身份及生成计数，防止错误跳过任务。
def validate_completed_report(
    # 传入已有或待发布的推理完成报告。
    report: Mapping[str, object],
    # 传入推理完成报告文件路径。
    report_path: Path,
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 传入将本次模型、数据、代码和输出绑定在一起的身份记录。
    run_identity: Mapping[str, object],
    # 传入已发布运行身份文件的字节指纹。
    run_identity_sha256: str,
    # 传入8100组转换配对清单的字节指纹。
    pairs_csv_sha256: str,
    # 传入本次实际选择的转换组数。
    selected_pair_count: int,
# 结束 validate_completed_report 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # 完成报告必须与当前请求一致的字段集合。
    required = _report_required_fields(
        # 将本次模型、数据、代码和输出绑定在一起的身份记录。
        run_identity,
        # 已发布运行身份文件的字节指纹。
        run_identity_sha256=run_identity_sha256,
        # 8100组转换配对清单的字节指纹。
        pairs_csv_sha256=pairs_csv_sha256,
        # 本次实际选择的转换组数。
        selected_pair_count=selected_pair_count,
    )
    # 报告实际字段与期望值不一致的逐项记录。
    mismatches = {
        # 逐项保存不匹配字段的实际值和本次期望值。
        key: {"actual": report.get(key), "expected": expected}
        # 检查每一个完成报告必需字段。
        for key, expected in required.items()
        # 只保留与本次请求不一致的字段。
        if report.get(key) != expected
    }
    # 本次实际新生成的WAV数量。
    generated = report.get("generated")
    # 报告声称已验证并跳过的WAV计数。
    skipped = report.get("skipped_existing")
    # 组合下面的有效性条件，任一异常都阻止发布完成结果。
    if (
        # 新生成计数必须是整数，不能用True或浮点值冒充。
        type(generated) is not int
        # 跳过计数同样必须是真正整数。
        or type(skipped) is not int
        # 已生成数量不可能为负。
        or generated < 0
        # 已跳过数量不可能为负。
        or skipped < 0
    ):
        # 拒绝继续执行核验已有完成报告的身份及生成计数，防止错误跳过任务：当前输入没有通过校验。
        raise RuntimeError(f"Invalid completion counters in {report_path}")
    # 新生成数加有效已存在数的总完成数量。
    completed = generated + skipped
    # 身份字段有差异或总计数不覆盖本次全部配对时拒绝复用完成报告。
    if mismatches or completed != selected_pair_count:
        # 拒绝继续执行核验已有完成报告的身份及生成计数，防止错误跳过任务：当前输入没有通过校验。
        raise RuntimeError(
            # 报告现有完成文件不属于当前请求的学生运行。
            f"Existing {report_path.name} does not match the requested student run: "
            # 打印不一致字段与已有完成数量。
            f"mismatches={mismatches}, completed={completed}, "
            # 给出本次应该完成的转换对总数。
            f"expected={selected_pair_count}"
        )


# 检查WAV音频头后，按pair_id记录每个生成文件的内容指纹。
def collect_generated_wav_hashes(
    # 传入转换配对及对应输出WAV路径。
    pairs: Sequence[EvaluationPair],
# 结束 collect_generated_wav_hashes 的参数声明；返回类型为 Dict[str, str]，具体内容由下方实现给出。
) -> Dict[str, str]:
    # 先检查每条输出 WAV 格式，再按配对顺序登记内容指纹。
    """Hash every selected WAV in pair order after validating its audio header."""

    # 当前检查或转换得到的返回记录。
    result: Dict[str, str] = {}
    # 按固定配对顺序检查每个输出文件。
    for pair in pairs:
        # 当前需要核验或写入的具体文件路径。
        path = Path(pair.output_wav)
        # 文件缺失、损坏或音频规格不符时不能记录成完成。
        if not valid_existing_wav(path):
            # 拒绝继续执行检查WAV音频头后，按pair_id记录每个生成文件的内容指纹：当前输入没有通过校验。
            raise RuntimeError(f"Cannot hash missing or invalid WAV for pair {pair.pair_id}")
        # 用pair_id绑定当前WAV的实际字节指纹。
        result[str(pair.pair_id)] = sha256_file(path)
    # 返回按字段或配对编号索引的核验结果。
    return result


# 确认WAV指纹表覆盖全部配对，且整体指纹与逐文件记录一致。
def validate_report_wav_hash_manifest(
    # 传入已有或待发布的推理完成报告。
    report: Mapping[str, object],
    # 传入转换配对及对应输出WAV路径。
    pairs: Sequence[EvaluationPair],
# 结束 validate_report_wav_hash_manifest 的参数声明；返回类型为 Mapping[str, str]，具体内容由下方实现给出。
) -> Mapping[str, str]:
    # 核对完成报告中每个转换对的 WAV 指纹及其集合签名。
    """Validate the completed report's per-pair waveform content manifest."""

    # 待序列化或检查类型的数据。
    value = report.get("generated_wav_sha256")
    # 元数据必须按字段组织，其他格式无法逐项核验。
    if not isinstance(value, Mapping):
        # 拒绝继续执行确认WAV指纹表覆盖全部配对，且整体指纹与逐文件记录一致：当前输入没有通过校验。
        raise RuntimeError("Completed report has no generated_wav_sha256 mapping")
    # 每个转换组应具备的字符串pair_id。
    expected_keys = [str(pair.pair_id) for pair in pairs]
    # 输出指纹表不能漏配对或多出其他实验配对。
    if set(value) != set(expected_keys):
        # 拒绝继续执行确认WAV指纹表覆盖全部配对，且整体指纹与逐文件记录一致：当前输入没有通过校验。
        raise RuntimeError("Completed report waveform-hash pair IDs are incomplete")
    # 已校验合法的pair_id到WAV指纹字典，供重算整体指纹。
    normalized: Dict[str, str] = {}
    # SHA-256允许的小写十六进制字符。
    hexadecimal = set("0123456789abcdef")
    # 按正式配对顺序逐一检查指纹。
    for key in expected_keys:
        # 当前转换WAV的64字符内容指纹。
        digest = value.get(key)
        # 组合下面的有效性条件，任一异常都阻止发布完成结果。
        if (
            # SHA-256应表示为字符串。
            not isinstance(digest, str)
            # SHA-256十六进制表示必须恰好64字符。
            or len(digest) != 64
            # 指纹字符必须全部属于小写十六进制集合。
            or not set(digest).issubset(hexadecimal)
        ):
            # 拒绝继续执行确认WAV指纹表覆盖全部配对，且整体指纹与逐文件记录一致：当前输入没有通过校验。
            raise RuntimeError(f"Invalid waveform SHA-256 for pair {key}: {digest!r}")
        # 将合法指纹按已知pair_id写入规范化字典。
        normalized[key] = digest
    # 规范化逐文件指纹字典之后的整体SHA-256。
    expected_collection_sha256 = canonical_json_sha256(normalized)
    # 整体WAV集合指纹必须能由逐文件指纹重新推导出来。
    if report.get("generated_wav_collection_sha256") != expected_collection_sha256:
        # 拒绝继续执行确认WAV指纹表覆盖全部配对，且整体指纹与逐文件记录一致：当前输入没有通过校验。
        raise RuntimeError("Completed report waveform collection hash is inconsistent")
    # 返回完成且合法的逐配对WAV指纹映射。
    return normalized


# 在写WAV前检查已有目录的模型身份和配对血缘，安全恢复中断。
def prepare_output_lineage(
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 传入本次学生推理的专属输出目录。
    output_dir: Path,
    # 传入将本次模型、数据、代码和输出绑定在一起的身份记录。
    run_identity: Mapping[str, object],
    # 传入按固定顺序编码的配对CSV字节。
    pairs_payload: bytes,
    # 传入本次需要生成的有序转换组列表。
    selected_pairs: Sequence[EvaluationPair],
# 结束 prepare_output_lineage 的参数声明；返回类型为 Tuple[bool, bool, str, str, set]，具体内容由下方实现给出。
) -> Tuple[bool, bool, str, str, set]:
    # 加载模型和写音频前，先验证输出目录身份并恢复中断状态。
    """Validate/recover an output directory before loading models or writing WAVs.

    Returns ``(already_complete, recovery_needed, identity_sha, pairs_sha,
    force_regenerate_pair_ids)``.
    """

    # 输出目录内发布的run_identity.json。
    identity_path = output_dir / "run_identity.json"
    # 输出目录内发布的pairs.csv。
    pairs_path = output_dir / "pairs.csv"
    # 推理完成报告文件路径。
    report_path = output_dir / "inference_report.json"
    # 恢复中暂存旧完成报告的路径。
    recovery_path = output_dir / "inference_report.recovery.json"
    # 本次生成语音的专属wav目录。
    wav_dir = output_dir / "wav"

    # 当前请求身份序列化后的精确JSON字节。
    expected_identity_payload = rendered_json_bytes(run_identity)
    # 当前身份文件应具有的精确字节指纹。
    expected_identity_sha256 = hashlib.sha256(expected_identity_payload).hexdigest()
    # 固定配对CSV应具有的字节指纹。
    expected_pairs_sha256 = hashlib.sha256(pairs_payload).hexdigest()

    # 输出目录已经留下的WAV文件。
    existing_wavs = list(wav_dir.glob("*.wav")) if wav_dir.exists() else []
    # 输出目录是否已经具有指标文件。
    metrics_material = (
        # 检查指标目录中是否已有任何文件，防止覆盖已评估结果。
        any(path.is_file() for path in (output_dir / "metrics").rglob("*"))
        # 指标目录存在时才递归检查。
        if (output_dir / "metrics").exists()
        # 没有指标目录时，视为尚无指标文件需要保护。
        else False
    )
    # 目录是否已存放WAV、完成报告或指标等实质结果。
    material = bool(existing_wavs) or report_path.is_file() or recovery_path.is_file() or metrics_material

    # 从已有运行身份文件读取的历史记录。
    existing_identity: Optional[Dict[str, object]] = None
    # 已有运行身份文件时先读取，决定是否可安全续跑。
    if identity_path.is_file():
        # 保证检查或加锁中途失败时仍走清理流程。
        try:
            # 从已有运行身份文件读取的历史记录。
            existing_identity = read_json_object(identity_path)
        # 身份文件不可读或JSON损坏时不能随意猜测原实验。
        except (OSError, ValueError) as error:
            # 已有实质结果时必须保留正确身份来源。
            if material:
                # 拒绝继续执行在写WAV前检查已有目录的模型身份和配对血缘，安全恢复中断：当前输入没有通过校验。
                raise RuntimeError(
                    # 目录已有音频但身份文件不可读时，明确指出问题文件。
                    f"Material student outputs have an unreadable identity: {identity_path}"
                # 保留底层读取异常作为原因，便于定位身份文件损坏。
                ) from error
    # 已有身份内容和字节指纹是否都与本次请求匹配。
    identity_matches = (
        # 身份内容本身必须完全相同。
        existing_identity == run_identity
        # 确认身份文件真实存在。
        and identity_path.is_file()
        # 进一步要求其发布字节与当前请求一致。
        and sha256_file(identity_path) == expected_identity_sha256
    )
    # 已有WAV或指标却属于另一运行时，拒绝复用目录。
    if material and not identity_matches:
        # 拒绝继续执行在写WAV前检查已有目录的模型身份和配对血缘，安全恢复中断：当前输入没有通过校验。
        raise RuntimeError(
            # 已有音频属于不同实验时，报告并拒绝复用该目录。
            f"Material outputs belong to another run; refusing to reuse {output_dir}"
        )

    # 已有配对清单是否与本次固定配对一致。
    pairs_match = pairs_path.is_file() and sha256_file(pairs_path) == expected_pairs_sha256
    # 已有结果必须绑定同一配对CSV，防止指标顺序与音频错配。
    if material and not pairs_match:
        # 拒绝继续执行在写WAV前检查已有目录的模型身份和配对血缘，安全恢复中断：当前输入没有通过校验。
        raise RuntimeError(
            # 已有音频没有绑定当前配对清单时，拒绝续推。
            f"Material outputs are not bound to the requested pairs.csv in {output_dir}"
        )

    # 按文件路径保存校验后的正式/恢复报告。
    reports: Dict[Path, Mapping[str, object]] = {}
    # 同时检查恢复暂存报告和正式完成报告。
    for candidate in (recovery_path, report_path):
        # 只有实际存在的报告才读取校验。
        if candidate.is_file():
            # 当前候选报告解析后的内容。
            candidate_report = read_json_object(candidate)
            # 核验已有完成报告的身份及生成计数，防止错误跳过任务。
            validate_completed_report(
                # 当前候选报告解析后的内容。
                candidate_report,
                # 错误信息保留当前正在检查的具体正式或恢复报告路径。
                candidate,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录。
                run_identity=run_identity,
                # 已发布运行身份文件的字节指纹。
                run_identity_sha256=expected_identity_sha256,
                # 8100组转换配对清单的字节指纹。
                pairs_csv_sha256=expected_pairs_sha256,
                # 本次实际选择的转换组数。
                selected_pair_count=len(selected_pairs),
            )
            # 确认WAV指纹表覆盖全部配对，且整体指纹与逐文件记录一致。
            validate_report_wav_hash_manifest(candidate_report, selected_pairs)
            # 保留校验通过的报告，随后读取旧WAV内容指纹。
            reports[candidate] = candidate_report

    # A recovery marker can survive another interruption.  Preserve that fact
    # in the final report even when the active completion report is absent.
    # 本次是否需要修复之前已声称完成的输出。
    recovery_needed = recovery_path.is_file()
    # 必须重新生成的缺失、损坏或内容改变的转换组编号。
    force_regenerate_pair_ids = set()
    # 用于核对旧WAV内容的正式或恢复报告。
    hash_report_path = report_path if report_path.is_file() else recovery_path
    # 存在旧完成报告时，根据其指纹核对每个WAV。
    if hash_report_path.is_file():
        # 旧报告记录的每一组正确WAV指纹。
        expected_wav_hashes = validate_report_wav_hash_manifest(
            # 拿旧报告和当前配对表对照，要求pair_id全部一致。
            reports[hash_report_path], selected_pairs
        )
        # 必须重新生成的缺失、损坏或内容改变的转换组编号。
        force_regenerate_pair_ids = {
            # 记录当前必须重生成或尚未完成的配对编号。
            pair.pair_id
            # 逐个检查本次选择的正式或诊断配对。
            for pair in selected_pairs
            # 组合下面的有效性条件，任一异常都阻止发布完成结果。
            if (
                # 缺失或音频规格不合法的WAV必须重新生成。
                not valid_existing_wav(Path(pair.output_wav))
                # 即使音频格式正常，也检查内容是否被替换。
                or sha256_file(Path(pair.output_wav))
                # 与旧完成报告的正确内容指纹不同就标记重生成。
                != expected_wav_hashes[str(pair.pair_id)]
            )
        }
        # 正式完成报告还存在时，判断是否真的所有文件都已完整。
        if report_path.is_file():
            # 没有缺失或变化的WAV，才允许直接确认全部完成。
            if not force_regenerate_pair_ids:
                # 按固定顺序返回完成标志、恢复标志、身份指纹和待重建配对。
                return (
                    # 第一项表示现有全部输出已验证完成。
                    True,
                    # 相应布尔位置表示无需恢复或本次仍需继续生成。
                    False,
                    # 当前身份文件应具有的精确字节指纹。
                    expected_identity_sha256,
                    # 固定配对CSV应具有的字节指纹。
                    expected_pairs_sha256,
                    # 现有WAV全部正确时，待重生成集合为空。
                    set(),
                )
            # 旧报告曾完成但WAV有问题，本次明确标记为恢复运行。
            recovery_needed = True

    # 没有实质冲突且身份尚未发布时，写入当前运行身份。
    if not identity_matches:
        # 先写临时文件并同步磁盘，再原子替换正式文件。
        atomic_write_bytes(identity_path, expected_identity_payload)
    # 没有实质冲突且配对尚未发布时，写入当前配对CSV。
    if not pairs_match:
        # 先写临时文件并同步磁盘，再原子替换正式文件。
        atomic_write_bytes(pairs_path, pairs_payload)
    # 发布后立即重新核对身份文件的精确字节。
    if sha256_file(identity_path) != expected_identity_sha256:
        # 拒绝继续执行在写WAV前检查已有目录的模型身份和配对血缘，安全恢复中断：当前输入没有通过校验。
        raise RuntimeError("Published student run identity bytes are inconsistent")
    # 发布后再次核对配对CSV，确保写入没有改变内容。
    if sha256_file(pairs_path) != expected_pairs_sha256:
        # 拒绝继续执行在写WAV前检查已有目录的模型身份和配对血缘，安全恢复中断：当前输入没有通过校验。
        raise RuntimeError("Published student pairs.csv bytes are inconsistent")

    # 开始修复时将旧完成报告移到恢复标记，避免外部误判当前完整。
    if recovery_needed and report_path.is_file():
        # 保留旧报告作为恢复证据，同时撤下暂时不成立的完成标记。
        os.replace(report_path, recovery_path)
    # 按固定顺序返回完成标志、恢复标志、身份指纹和待重建配对。
    return (
        # 相应布尔位置表示无需恢复或本次仍需继续生成。
        False,
        # 本次是否需要修复之前已声称完成的输出。
        recovery_needed,
        # 当前身份文件应具有的精确字节指纹。
        expected_identity_sha256,
        # 固定配对CSV应具有的字节指纹。
        expected_pairs_sha256,
        # 必须重新生成的缺失、损坏或内容改变的转换组编号。
        force_regenerate_pair_ids,
    )


# 解析模型、配置、输出、随机种子和诊断配对上限。
def parse_args() -> argparse.Namespace:
    # 把模块说明用作命令行帮助，明确本入口只负责学生推理。
    parser = argparse.ArgumentParser(description=__doc__)
    # 命令行指定Matpool上的规范项目根目录。
    parser.add_argument("--project-root", type=Path, default=SERVER_PROJECT_ROOT)
    # 命令行指定待核验的模型检查点。
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STUDENT_CHECKPOINT)
    # 登记下面的命令行选项及其严格默认值。
    parser.add_argument(
        # 训练契约路径默认指向锁定的历史学生实验。
        "--run-contract", type=Path, default=DEFAULT_STUDENT_RUN_CONTRACT
    )
    # 登记下面的命令行选项及其严格默认值。
    parser.add_argument(
        # 默认使用学生训练时绑定的最终自训声码器权重。
        "--vocoder-checkpoint", type=Path, default=DEFAULT_VOCODER_CHECKPOINT
    )
    # 命令行指定声码器的结构与音频配置。
    parser.add_argument("--vocoder-config", type=Path, default=DEFAULT_VOCODER_CONFIG)
    # 登记下面的命令行选项及其严格默认值。
    parser.add_argument(
        # 默认读取该声码器的最终归档证明。
        "--vocoder-attestation", type=Path, default=DEFAULT_VOCODER_ATTESTATION
    )
    # 命令行指定本次学生推理的专属输出目录。
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # 命令行指定每个推理小批次固定处理4个配对。
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    # 命令行指定固定整体随机种子1234。
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    # 命令行指定本次推理的计算设备。
    parser.add_argument("--device", default="cuda")
    # 登记下面的命令行选项及其严格默认值。
    parser.add_argument(
        # 显式限制转换对数量时属于诊断运行，不声明正式 8100 对完成。
        "--limit",
        # 将配对数或批量解析为整数，便于边界校验。
        type=int,
        # 未指定配对上限时执行全部8100组正式任务。
        default=None,
        # 帮助文字明确少量测试需要单独输出位置。
        help="Diagnostic only; requires a non-formal output directory.",
    )
    # 返回解析后的命令行参数，由下一步执行严格校验。
    return parser.parse_args()


# 将路径规范化并严格检查正式评估不能变动的参数。
def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    # 按下面列出的字段逐一核对数据或解析命令行路径。
    for key in (
        # 测试用项目根目录。
        "project_root",
        # 待核验的模型检查点。
        "checkpoint",
        # 异常消息中标明正在检查运行契约。
        "run_contract",
        # 本次冻结声码器的最终权重文件。
        "vocoder_checkpoint",
        # 声码器的结构与音频配置。
        "vocoder_config",
        # 声码器250万步最终检查点的审核记录。
        "vocoder_attestation",
        # 本次学生推理的专属输出目录。
        "output_dir",
    ):
        # 展开用户路径并转为规范绝对路径，避免相同文件有不同写法。
        setattr(args, key, getattr(args, key).expanduser().resolve())
    # 是否执行完整8100组正式推理。
    formal = args.limit is None
    # 限定学生专属目录，防止覆盖教师结果或混入诊断输出。
    validate_output_directory(args.output_dir, formal=formal)
    # 完整8100组正式评估要求模型和路径全部锁定。
    if formal:
        # 完整正式推理所锁定的模型与配置绝对路径。
        fixed_paths = {
            # 记录待核验的模型检查点。
            "checkpoint": DEFAULT_STUDENT_CHECKPOINT.resolve(),
            # 记录记录模型训练设置、数据和依赖的运行契约。
            "run_contract": DEFAULT_STUDENT_RUN_CONTRACT.resolve(),
            # 记录本次冻结声码器的最终权重文件。
            "vocoder_checkpoint": DEFAULT_VOCODER_CHECKPOINT.resolve(),
            # 记录声码器的结构与音频配置。
            "vocoder_config": DEFAULT_VOCODER_CONFIG.resolve(),
            # 记录声码器250万步最终检查点的审核记录。
            "vocoder_attestation": DEFAULT_VOCODER_ATTESTATION.resolve(),
        }
        # 逐条核对正式模式的学生、契约和声码器路径。
        for key, expected in fixed_paths.items():
            # 当前文件或元数据实际值。
            actual = getattr(args, key)
            # 实际字段或指纹与锁定值不同，停止混用实验。
            if actual != expected:
                # 拒绝继续执行将路径规范化并严格检查正式评估不能变动的参数：当前输入没有通过校验。
                raise RuntimeError(
                    # 报告正式学生参数与锁定配置之间的差异。
                    f"Formal student inference fixes {key}={expected}; got {actual}"
                )
    # 批量会改变固定随机分组，正式入口不允许任意更改。
    if args.batch_size != FORMAL_BATCH_SIZE:
        # 拒绝继续执行将路径规范化并严格检查正式评估不能变动的参数：当前输入没有通过校验。
        raise RuntimeError(f"Student inference batch size is fixed at {FORMAL_BATCH_SIZE}")
    # 保持原固定随机种子，保证重复运行可比较。
    if args.seed != FORMAL_SEED:
        # 拒绝继续执行将路径规范化并严格检查正式评估不能变动的参数：当前输入没有通过校验。
        raise RuntimeError(f"Student inference seed is fixed at {FORMAL_SEED}")
    # 指定少量配对时属于诊断任务，需要使用独立输出目录。
    if args.limit is not None:
        # 诊断配对上限只允许1到8099，完整数量由正式模式负责。
        if args.limit < 1 or args.limit >= 8100:
            # 拒绝继续执行将路径规范化并严格检查正式评估不能变动的参数：当前输入没有通过校验。
            raise ValueError("Diagnostic limit must be in [1, 8099]")
        # 少量诊断不能写入正式8100组输出位置。
        if args.output_dir == DEFAULT_OUTPUT_DIR.resolve():
            # 拒绝继续执行将路径规范化并严格检查正式评估不能变动的参数：当前输入没有通过校验。
            raise RuntimeError("A diagnostic limit cannot use the formal output directory")
    # 正式推理要求在租用服务器GPU执行。
    if formal and args.device != "cuda":
        # 拒绝继续执行将路径规范化并严格检查正式评估不能变动的参数：当前输入没有通过校验。
        raise RuntimeError("Formal student inference requires --device cuda")
    # 所有路径和参数校验通过后交给主流程。
    return args


# 列出所有能改变学生生成结果的源码文件。
def _implementation_paths(project_root: Path) -> Dict[str, Path]:
    # 影响本次推理的源码相对路径列表。
    relatives = (
        # 把 fastvoicegrad/infer_paper_eval.py 纳入当前推理源码身份检查。
        "fastvoicegrad/infer_paper_eval.py",
        # 把 fastvoicegrad/components.py 纳入当前推理源码身份检查。
        "fastvoicegrad/components.py",
        # 把 fastvoicegrad/config.py 纳入当前推理源码身份检查。
        "fastvoicegrad/config.py",
        # 把 fastvoicegrad/objectives.py 纳入当前推理源码身份检查。
        "fastvoicegrad/objectives.py",
        # 把 voicegrad/config.py 纳入当前推理源码身份检查。
        "voicegrad/config.py",
        # 把 voicegrad/diffusion.py 纳入当前推理源码身份检查。
        "voicegrad/diffusion.py",
        # 把 voicegrad/feature_alignment.py 纳入当前推理源码身份检查。
        "voicegrad/feature_alignment.py",
        # 把 voicegrad/infer_voicegrad30_paper_eval.py 纳入当前推理源码身份检查。
        "voicegrad/infer_voicegrad30_paper_eval.py",
        # 把 voicegrad/model.py 纳入当前推理源码身份检查。
        "voicegrad/model.py",
        # 把 hifigan/tools/vocode.py 纳入当前推理源码身份检查。
        "hifigan/tools/vocode.py",
        # 把 hifigan/official/models.py 纳入当前推理源码身份检查。
        "hifigan/official/models.py",
    )
    # 将列出的相对源码路径转换为项目中的具体文件。
    return {relative: project_root / relative for relative in relatives}


# 收集权重、统计量、协议和代码等关键输入，供前后两次核验。
def _critical_input_paths(
    # 以下参数要求按名称传入，避免多条相似路径或指纹传错位置。
    *,
    # 传入已解析并校验的推理命令行参数。
    args: argparse.Namespace,
    # 传入保存测试清单与拆分协议的元数据目录。
    metadata: Path,
    # 传入源mel、BNF、说话人向量和统计量所在的数据目录。
    data_root: Path,
    # 传入所有影响推理语义的源代码文件路径。
    implementation_paths: Mapping[str, Path],
# 结束 _critical_input_paths 的参数声明；返回类型为 Dict[str, Path]，具体内容由下方实现给出。
) -> Dict[str, Path]:
    # 当前检查或转换得到的返回记录。
    result = {
        # 记录加载的学生epoch100权重路径。
        "student_checkpoint": args.checkpoint,
        # 记录赋予该学生检查点实验含义的训练契约。
        "student_run_contract": args.run_contract,
        # 记录本次冻结声码器的最终权重文件。
        "vocoder_checkpoint": args.vocoder_checkpoint,
        # 记录声码器的结构与音频配置。
        "vocoder_config": args.vocoder_config,
        # 记录声码器250万步最终检查点的审核记录。
        "vocoder_attestation": args.vocoder_attestation,
        # 记录100条测试原语音的清单。
        "test_manifest": metadata / "test_manifest.csv",
        # 记录训练/测试说话人与句子的拆分协议。
        "split_protocol": metadata / "split_protocol.json",
        # 记录训练集逐mel通道的均值。
        "mel_mean": data_root / "stats/mel_mean.npy",
        # 记录训练集逐mel通道的标准差。
        "mel_std": data_root / "stats/mel_std.npy",
    }
    # 把源码也加入关键输入，前后核验时能发现实现被中途修改。
    result.update({f"implementation:{key}": path for key, path in implementation_paths.items()})
    # 返回按字段或配对编号索引的核验结果。
    return result


# 将文件路径映射转成按名称索引的SHA-256映射。
def _hash_paths(paths: Mapping[str, Path]) -> Dict[str, str]:
    # 对每个关键文件计算独立SHA-256指纹。
    return {key: sha256_file(path) for key, path in paths.items()}


# 依次完成正式预检、配对、单步转换、声码器合成和完成报告发布。
def main() -> None:
    # 已解析并校验的推理命令行参数。
    args = _resolve_args(parse_args())
    # 先确认当前是允许的租用服务器，再执行任何推理计算。
    assert_rented_server(args.project_root)
    # 请求GPU但CUDA不可用时，明确失败而非悄悄改用其他机器。
    if args.device == "cuda" and not torch.cuda.is_available():
        # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
        raise RuntimeError("CUDA was requested but is unavailable")
    # 本次推理的计算设备。
    device = torch.device(args.device)
    # 是否执行完整8100组正式推理。
    formal = args.limit is None

    # 依赖和输出互斥锁覆盖整个正式推理过程。
    with acquire_inference_locks(args.project_root):
        # 源mel、BNF、说话人向量和统计量所在的数据目录。
        data_root = args.project_root / "data"
        # 保存测试清单与拆分协议的元数据目录。
        metadata = data_root / "metadata"
        # 固定测试清单CSV路径。
        manifest = metadata / "test_manifest.csv"
        # 训练/测试说话人与句子的拆分协议。
        split_protocol = metadata / "split_protocol.json"
        # 训练集mel均值统计文件。
        mel_mean_path = data_root / "stats/mel_mean.npy"
        # 训练集mel标准差统计文件。
        mel_std_path = data_root / "stats/mel_std.npy"

        # 一次取得冻结学生、归档元数据、训练契约和核实后的权重指纹。
        student, student_checkpoint, run_contract, student_sha256 = (
            # 加载唯一登记的epoch100学生并冻结全部参数供正式推理。
            load_student_checkpoint(
                # Matpool上的规范项目根目录。
                project_root=args.project_root,
                # 学生或声码器权重文件路径。
                checkpoint_path=args.checkpoint,
                # 学生训练时保存的运行契约路径。
                run_contract_path=args.run_contract,
                # 在学生所在设备创建或读取张量，避免隐式设备迁移。
                device=device,
            )
        )
        # 核对当前数据拆分和预处理指纹与学生训练历史一致。
        validate_active_data_contract(args.project_root, run_contract)
        # 已验证与学生训练一致的声码器依赖指纹。
        vocoder_hashes = validate_vocoder_dependency_identity(
            # 记录模型训练设置、数据和依赖的运行契约。
            run_contract,
            # 学生或声码器权重文件路径。
            checkpoint_path=args.vocoder_checkpoint,
            # 声码器结构配置文件路径。
            config_path=args.vocoder_config,
            # 声码器最终检查点审核文件路径。
            attestation_path=args.vocoder_attestation,
        )
        # 同时加载冻结声码器、实际配置和已核验的权重元数据。
        vocoder, vocoder_config, vocoder_checkpoint = load_frozen_vocoder(
            # 在当前规范项目目录寻找声码器实现。
            args.project_root,
            # 加载启动参数指定且已通过身份核验的最终声码器权重。
            args.vocoder_checkpoint,
            # 使用该声码器绑定的结构与音频配置。
            args.vocoder_config,
            # 依据最终检查点审核记录验证声码器来源。
            args.vocoder_attestation,
            # 按原学生训练配置核对声码器结构和训练步数。
            PAPER_STUDENT_CONFIG,
            # 本次推理的计算设备。
            device,
        )

        # 测试清单中的100条原语音。
        rows = load_test_rows(manifest)
        # 按论文数量规则生成的全部8100组转换。
        all_pairs = build_pairs(rows, args.output_dir)
        # 按固定顺序编码的配对CSV字节。
        pairs_payload = render_pairs_csv(all_pairs)
        # 本次需要生成的有序转换组列表。
        selected_pairs = all_pairs if formal else all_pairs[: args.limit]
        # 开始推理前各测试特征和波形的指纹。
        test_input_hashes = collect_test_input_hashes(args.project_root, rows)
        # 所有影响推理语义的源代码文件路径。
        implementation_paths = _implementation_paths(args.project_root)
        # 逐个实现文件的源码字节指纹。
        implementation_sha256 = _hash_paths(implementation_paths)
        # 学生、声码器、数据统计和源码的关键文件列表。
        critical_paths = _critical_input_paths(
            # 已解析并校验的推理命令行参数。
            args=args,
            # 保存测试清单与拆分协议的元数据目录。
            metadata=metadata,
            # 源mel、BNF、说话人向量和统计量所在的数据目录。
            data_root=data_root,
            # 所有影响推理语义的源代码文件路径。
            implementation_paths=implementation_paths,
        )
        # 推理开始前关键文件的字节指纹。
        critical_hashes = _hash_paths(critical_paths)

        # 逐通道mel均值，形状[80,1]。
        mean = np.load(mel_mean_path, allow_pickle=False).reshape(-1, 1)
        # 逐通道mel标准差，形状[80,1]。
        std = np.load(mel_std_path, allow_pickle=False).reshape(-1, 1)
        # 归一化统计必须逐80个mel通道提供一个值。
        if mean.shape != (80, 1) or std.shape != (80, 1):
            # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
            raise RuntimeError(f"Invalid mel statistics: mean={mean.shape}, std={std.shape}")
        # 均值标准差必须有限且标准差为正，否则归一化无效。
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
            raise RuntimeError("Mel statistics contain invalid values")

        # 将本次模型、数据、代码和输出绑定在一起的身份记录。
        run_identity: Dict[str, object] = {
            # 记录运行身份记录的字段格式版本。
            "identity_format_version": IDENTITY_FORMAT_VERSION,
            # 记录系统机器标识，学生固定为fastvoicegrad。
            "system_id": "fastvoicegrad",
            # 记录报告中显示的学生名称与K=1、epoch100。
            "system_label": "FastVoiceGrad (K=1) epoch 100",
            # 记录明确标记student，防止与teacher评估混用。
            "model_role": "student",
            # 记录当前实验的text-unseen划分协议。
            "data_protocol": "text-unseen",
            # 记录论文客观评估要求的8100组转换。
            "paper_pair_count": 8100,
            # 记录本次实际选择的转换组数。
            "selected_pair_count": len(selected_pairs),
            # 记录学生已完成的训练轮次。
            "student_epoch": int(student_checkpoint["epoch"]),
            # 记录学生已执行的优化器更新次数。
            "student_global_step": int(student_checkpoint["global_step"]),
            # 记录初始化学生的教师已训练500轮。
            "teacher_epoch": 500,
            # 记录学生反向扩散次数K=1。
            "student_steps": 1,
            # 记录学生从SK=950开始单步重建。
            "student_start_step": PAPER_STUDENT_CONFIG.student_start_step,
            # 记录完整扩散日程的1000个噪声等级。
            "diffusion_steps": PAPER_STUDENT_CONFIG.diffusion_steps,
            # 记录余弦日程使用的0.008偏移。
            "cosine_offset": PAPER_STUDENT_CONFIG.cosine_offset,
            # 记录先按公式9将源mel加噪到x950。
            "initial_state": "q(x_950 | x_source_0), FastVoiceGrad Equation (9)",
            # 记录学生只执行一次确定性的950→0重建。
            "transition": "single deterministic learned reverse mean 950 -> 0",
            # 记录加载的学生epoch100权重路径。
            "student_checkpoint": str(args.checkpoint),
            # 记录加载的学生epoch100权重路径的SHA-256字节指纹。
            "student_checkpoint_sha256": student_sha256,
            # 记录赋予该学生检查点实验含义的训练契约。
            "student_run_contract": str(args.run_contract),
            # 记录赋予该学生检查点实验含义的训练契约的SHA-256字节指纹。
            "student_run_contract_sha256": canonical_json_sha256(run_contract),
            # 记录student_run_contract_file的SHA-256字节指纹。
            "student_run_contract_file_sha256": sha256_file(args.run_contract),
            # 记录使用自训修改版HiFi-GAN。
            "vocoder_kind": "custom",
            # 记录声码器已完成2500000次训练更新。
            "vocoder_global_step": int(vocoder_checkpoint["global_step"]),
            # 记录本次冻结声码器的最终权重文件。
            "vocoder_checkpoint": str(args.vocoder_checkpoint),
            # 记录声码器的结构与音频配置。
            "vocoder_config": str(args.vocoder_config),
            # 记录声码器250万步最终检查点的审核记录。
            "vocoder_attestation": str(args.vocoder_attestation),
            # 把核验过的声码器权重、配置和审核指纹合入运行身份。
            **vocoder_hashes,
            # 记录100条测试原语音的清单。
            "test_manifest": str(manifest),
            # 记录100条测试原语音的清单的SHA-256字节指纹。
            "test_manifest_sha256": sha256_file(manifest),
            # 记录训练/测试说话人与句子的拆分协议。
            "split_protocol": str(split_protocol),
            # 记录训练/测试说话人与句子的拆分协议的SHA-256字节指纹。
            "split_protocol_sha256": sha256_file(split_protocol),
            # 记录训练集逐mel通道的均值。
            "mel_mean": str(mel_mean_path),
            # 记录训练集逐mel通道的均值的SHA-256字节指纹。
            "mel_mean_sha256": sha256_file(mel_mean_path),
            # 记录训练集逐mel通道的标准差。
            "mel_std": str(mel_std_path),
            # 记录训练集逐mel通道的标准差的SHA-256字节指纹。
            "mel_std_sha256": sha256_file(mel_std_path),
            # 记录8100组转换配对清单的字节指纹。
            "pairs_csv_sha256": hashlib.sha256(pairs_payload).hexdigest(),
            # 记录100条源语音×9个其他说话人×9个不同句位。
            "pair_rule": "100 sources x 9 other speakers x 9 different sentence slots",
            # 记录保持与原教师相同的BNF到mel时间对齐方法。
            "bnf_alignment": PAPER_STUDENT_CONFIG.bnf_alignment,
            # 记录保持原教师内容条件的时间下采样方法。
            "bnf_downsampling": PAPER_STUDENT_CONFIG.bnf_downsampling,
            # 记录固定整体随机种子1234。
            "seed": FORMAL_SEED,
            # 记录小批次种子由基础种子加首个pair_id决定。
            "seed_rule": "base_seed + first_pair_id_of_fixed_source-local_chunk",
            # 记录每个推理小批次固定处理4个配对。
            "batch_size": FORMAL_BATCH_SIZE,
            # 记录仅诊断使用的配对上限，正式全量为None。
            "selected_pair_limit": args.limit,
            # 记录逐个测试特征和参考波形的字节指纹。
            "test_input_file_sha256": test_input_hashes,
            # 记录测试输入指纹集合的规范化整体指纹。
            "test_input_collection_sha256": canonical_json_sha256(test_input_hashes),
            # 记录逐个实现文件的源码字节指纹。
            "implementation_sha256": implementation_sha256,
            # 记录当前干净工作树对应的Git提交。
            "git_commit": git_head(args.project_root),
        }

        (
            # 现有8100组结果是否全部已核验完成。
            already_complete,
            # 本次是否需要修复之前已声称完成的输出。
            recovery_needed,
            # 本次已发布身份文件的字节指纹。
            identity_sha256,
            # 本次已发布配对清单的字节指纹。
            pairs_sha256,
            # 必须重新生成的缺失、损坏或内容改变的转换组编号。
            force_regenerate_pair_ids,
        # 按固定顺序接收目录恢复检查的五个结果，避免完成状态和指纹串位。
        ) = (
            # 在写WAV前检查已有目录的模型身份和配对血缘，安全恢复中断。
            prepare_output_lineage(
                # 本次学生推理的专属输出目录。
                output_dir=args.output_dir,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录。
                run_identity=run_identity,
                # 按固定顺序编码的配对CSV字节。
                pairs_payload=pairs_payload,
                # 本次需要生成的有序转换组列表。
                selected_pairs=selected_pairs,
            )
        )
        # 已有结果全部核验通过时，输出完成报告后结束，不重复生成。
        if already_complete:
            # 向运行日志输出当前身份、完成状态或可读报告。
            print(
                # 显示已验证的完成报告，便于用户检查实际数量和模型身份。
                (args.output_dir / "inference_report.json").read_text(encoding="utf-8")
            )
            # 向运行日志输出当前身份、完成状态或可读报告。
            print(
                # 打印可被编排脚本识别的全量结果已存在标记。
                "FASTVOICEGRAD_STUDENT_8100_INFERENCE_ALREADY_COMPLETE"
                # 根据是否全量8100组选择正式或诊断状态标记。
                if formal
                # 诊断结果已存在时打印专用完成标志，不宣称正式全量完成。
                else "FASTVOICEGRAD_STUDENT_DIAGNOSTIC_INFERENCE_ALREADY_COMPLETE"
            )
            # 当前请求已完成或全部输出已存在，结束推理入口。
            return

        # 本次生成语音的专属wav目录。
        wav_dir = args.output_dir / "wav"
        # 建立本次WAV目录，允许断点续跑保留已有文件。
        wav_dir.mkdir(parents=True, exist_ok=True)
        # 与训练一致的余弦扩散系数表。
        diffusion = VoiceGradDiffusion(
            # 完整日程使用1000个噪声等级。
            PAPER_STUDENT_CONFIG.diffusion_steps,
            # 保持训练余弦日程的0.008偏移。
            PAPER_STUDENT_CONFIG.cosine_offset,
        # 将所有扩散系数搬到学生模型所在设备。
        ).to(device)
        # 移到推理设备的[80,1]均值张量。
        mean_tensor = torch.from_numpy(mean.astype(np.float32)).to(device)
        # 移到推理设备的[80,1]标准差张量。
        std_tensor = torch.from_numpy(std.astype(np.float32)).to(device)

        # 本次实际新生成的WAV数量。
        generated = 0
        # 报告声称已验证并跳过的WAV计数。
        skipped = 0
        # 本次生成阶段开始的墙上时钟。
        start_time = time.time()
        # 保持源语音一致且顺序固定的4组推理小批次。
        chunks = list(fixed_chunks(selected_pairs, FORMAL_BATCH_SIZE))
        # 按固定4组小批次顺序生成，进度条显示学生与自训声码器流程。
        for chunk in tqdm(chunks, desc="FastVoiceGrad-1 + custom HiFi-GAN"):
            # 当前小批次缺失或必须重建的转换组。
            missing = [
                # 将需要生成的配对记录加入当前待处理列表。
                pair
                # 逐个检查当前4组小批次中的转换任务。
                for pair in chunk
                # 组合下面的有效性条件，任一异常都阻止发布完成结果。
                if (
                    # 旧报告指纹不匹配的文件即便格式可读也必须重生成。
                    pair.pair_id in force_regenerate_pair_ids
                    # 缺失、损坏或规格错误的WAV同样需要生成。
                    or not valid_existing_wav(Path(pair.output_wav))
                )
            ]
            # 这个小批次的WAV全部有效时，跳过模型计算。
            if not missing:
                # 把本批有效已存在的所有文件计入跳过数。
                skipped += len(chunk)
                # 当前配对或小批次已经完成，转到下一项。
                continue

            # 当前小批次共用的源语音记录。
            source = chunk[0]
            # 读取的源语音80通道自然对数mel。
            mel = feature_matrix(
                # 按特征类型、说话人和句子定位特征文件。
                feature_path(
                    # 按源说话人和句子编号读取对应mel，保持源内容一致。
                    data_root, "mel", source.source_speaker, source.source_utt
                ),
                # 要求源mel具有80个频率通道。
                PAPER_CONFIG.n_mels,
            )
            # 读取的源语音144通道内容特征。
            bnf = feature_matrix(
                # 按特征类型、说话人和句子定位特征文件。
                feature_path(
                    # 读取与同一源语音对应的BNF内容特征。
                    data_root, "bnf", source.source_speaker, source.source_utt
                ),
                # 要求BNF具有教师配置规定的144个通道。
                PAPER_CONFIG.n_bnf,
            )
            # 任何源特征包含NaN/Inf都必须在模型前向前阻止。
            if not np.isfinite(mel).all() or not np.isfinite(bnf).all():
                # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
                raise RuntimeError(
                    # 指出源梅尔或 BNF 含非有限值的具体语音。
                    f"Source mel/BNF contains NaN or Inf: {source.source_utt}"
                )
            # 逐通道减均值除标准差后的源mel。
            normalized = ((mel - mean) / std).astype(np.float32)
            # 源语音标准化mel张量[B,80,T]。
            source_mel = torch.from_numpy(normalized).unsqueeze(0).to(device)
            # 与源mel时间轴对齐的内容特征[B,144,T]。
            source_bnf = align_bnf_to_mel(
                # 将BNF转换为[B=1,C,T]张量并搬到推理设备。
                torch.from_numpy(bnf).unsqueeze(0).to(device),
                # 以源mel的真实帧数作为BNF对齐目标。
                source_mel.shape[-1],
                # 沿用学生训练登记的BNF长度对齐方式。
                mode=PAPER_STUDENT_CONFIG.bnf_alignment,
                # 使用22050Hz采样率解释mel时间位置。
                sample_rate=PAPER_CONFIG.sample_rate,
                # 使用1024点FFT规格对应时间窗中心。
                n_fft=PAPER_CONFIG.n_fft,
                # mel相邻帧间隔为256采样点。
                hop_size=PAPER_CONFIG.hop_size,
                # BNF原始帧移为10毫秒，用于解释内容特征时间轴。
                bnf_frame_shift_seconds=0.010,
            )

            # 当前固定小批次实际包含的转换组数。
            batch_count = len(chunk)
            # 源语音标准化mel张量[B,80,T]。
            source_mel = source_mel.expand(batch_count, -1, -1).contiguous()
            # 与源mel时间轴对齐的内容特征[B,144,T]。
            source_bnf = source_bnf.expand(batch_count, -1, -1).contiguous()
            # 当前小批次每个参考说话人的256维条件向量。
            target_vectors = []
            # 按当前批次的固定顺序读取每一组目标参考条件。
            for pair in chunk:
                # 当前需要核验或写入的具体文件路径。
                path = feature_path(
                    # 源mel、BNF、说话人向量和统计量所在的数据目录。
                    data_root,
                    # 从已审计的说话人向量目录加载目标音色条件。
                    "spk_emb",
                    # 目标条件来自该配对指定的参考说话人。
                    pair.reference_speaker,
                    # 使用指定参考句子的说话人向量，句子位置与源句不同。
                    pair.reference_utt,
                )
                # 当前目标参考语音提取出的说话人向量。
                vector = np.load(path, allow_pickle=False).squeeze()
                # 每条参考语音必须提供256维向量，不能误用其他编码器输出。
                if vector.shape != (PAPER_CONFIG.spk_emb_dim,):
                    # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
                    raise RuntimeError(f"Invalid speaker embedding shape {vector.shape}: {path}")
                # 说话人条件出现NaN/Inf时拒绝生成。
                if not np.isfinite(vector).all():
                    # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
                    raise RuntimeError(f"Non-finite speaker embedding: {path}")
                # 按配对顺序保存目标向量，确保输出语音与目标身份对应。
                target_vectors.append(vector.astype(np.float32))
            # 目标说话人条件张量[B,256]。
            target_speaker = torch.from_numpy(np.stack(target_vectors)).to(device)

            # 本小批次使用的独立随机数生成器。
            generator = torch.Generator(device=device).manual_seed(
                # 小批次独立种子由基础1234加首个pair_id组成，重跑边界固定。
                FORMAL_SEED + int(chunk[0].pair_id)
            )
            # 构造x950的独立标准高斯噪声。
            source_noise = torch.randn(
                # 噪声与源mel的批量、80通道和时间长度完全一致。
                source_mel.shape,
                # 在学生所在设备创建或读取张量，避免隐式设备迁移。
                device=device,
                # 噪声采用与mel相同的浮点精度。
                dtype=source_mel.dtype,
                # 从当前小批次独立生成器抽噪声，不受跳过其它批次影响。
                generator=generator,
            )
            # 整个生成前向关闭梯度记录，降低显存占用。
            with torch.inference_mode():
                # 学生一步重建的标准化目标mel。
                converted_norm = student_one_step(
                    # 冻结的FastVoiceGrad学生噪声预测器。
                    student,
                    # 与训练一致的余弦扩散系数表。
                    diffusion,
                    # 源语音标准化mel张量[B,80,T]。
                    source_mel,
                    # 目标说话人条件张量[B,256]。
                    target_speaker,
                    # 与源mel时间轴对齐的内容特征[B,144,T]。
                    source_bnf,
                    # 构造x950的独立标准高斯噪声。
                    source_noise,
                )
                # 学生输出出现NaN/Inf时停止，避免写出坏音频。
                if not bool(torch.isfinite(converted_norm).all()):
                    # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
                    raise RuntimeError(
                        # 指出学生输出非有限值的转换块起始编号。
                        f"Student mel contains NaN/Inf in chunk {chunk[0].pair_id}"
                    )
                # 反标准化后送入HiFi-GAN的自然对数mel。
                converted_mel = (
                    # 乘回逐通道标准差，将学生标准化输出还原幅度。
                    converted_norm * std_tensor.unsqueeze(0)
                    # 再加回逐通道均值，得到HiFi-GAN所需自然对数mel。
                    + mean_tensor.unsqueeze(0)
                )
                # HiFi-GAN合成的当前批次单声道波形。
                audio_batch = vocoder(converted_mel).squeeze(1).cpu().numpy()

            # 按固定配对顺序将模型批量输出逐条写成WAV。
            for pair, audio in zip(chunk, audio_batch):
                # 当前转换组要写入的WAV目标路径。
                destination = Path(pair.output_wav)
                # 组合下面的有效性条件，任一异常都阻止发布完成结果。
                if (
                    # 该文件不属于必须重建的内容变化集合。
                    pair.pair_id not in force_regenerate_pair_ids
                    # 且原WAV音频格式有效时，保留已验证输出。
                    and valid_existing_wav(destination)
                ):
                    # 记录一条已存在且可复用的转换语音。
                    skipped += 1
                    # 当前配对或小批次已经完成，转到下一项。
                    continue
                # 波形样本必须全部有限，不能把数值异常写进正式结果。
                if not np.isfinite(audio).all():
                    # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
                    raise RuntimeError(f"Generated audio contains NaN/Inf: pair {pair.pair_id}")
                # 完整写好临时WAV后再发布目标文件。
                atomic_write_wav(
                    # 当前转换组要写入的WAV目标路径。
                    destination,
                    # 音频转float32并限制在PCM允许幅度范围，避免越界写入。
                    np.clip(audio.astype(np.float32), -1.0, 1.0),
                    # 输出WAV采样率固定22050Hz。
                    PAPER_CONFIG.sample_rate,
                )
                # 一条WAV完成原子写入后，累计新生成数量。
                generated += 1

        # 所有选择配对都必须被新生成或验证跳过，计数不齐不能完成。
        if generated + skipped != len(selected_pairs):
            # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
            raise RuntimeError(
                # 同时打印新生成和已验证跳过数量，解释覆盖为何不足。
                f"Student inference incomplete: generated={generated}, skipped={skipped}, "
                # 打印本次选中应完成的配对数。
                f"expected={len(selected_pairs)}"
            )
        # 推理结束后仍缺失或格式无效的pair_id。
        invalid = [
            # 记录当前必须重生成或尚未完成的配对编号。
            pair.pair_id
            # 逐个检查本次选择的正式或诊断配对。
            for pair in selected_pairs
            # 结束前再次寻找缺失或格式无效的输出。
            if not valid_existing_wav(Path(pair.output_wav))
        ]
        # 仍有无效WAV时拒绝发布完成报告。
        if invalid:
            # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
            raise RuntimeError(
                # 列出前十个非法 WAV，禁止发布假完成报告。
                f"Refusing to publish completion with invalid WAVs: {invalid[:10]} "
                # 补充非法 WAV 的总数。
                f"(total={len(invalid)})"
            )

        # 结束时重新核对权重、配置、统计与源码，防止中途被替换。
        if _hash_paths(critical_paths) != critical_hashes:
            # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
            raise RuntimeError("A critical model/data/code input changed during inference")
        # 结束时重新计算的测试输入指纹。
        final_test_hashes = collect_test_input_hashes(args.project_root, rows)
        # 测试特征和参考波形在推理前后必须逐字节一致。
        if final_test_hashes != test_input_hashes:
            # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
            raise RuntimeError("A test input changed during inference")
        # 确认运行身份文件未被其他程序中途更改。
        if sha256_file(args.output_dir / "run_identity.json") != identity_sha256:
            # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
            raise RuntimeError("Student run identity changed during inference")
        # 确认配对顺序和输出路径清单在推理过程中保持一致。
        if sha256_file(args.output_dir / "pairs.csv") != pairs_sha256:
            # 拒绝继续执行依次完成正式预检、配对、单步转换、声码器合成和完成报告发布：当前输入没有通过校验。
            raise RuntimeError("Student pairs.csv changed during inference")

        # 每个pair_id对应WAV文件的SHA-256。
        generated_wav_sha256 = collect_generated_wav_hashes(selected_pairs)

        # 已有或待发布的推理完成报告。
        report: Dict[str, object] = {
            # 将已核实的模型、数据和运行身份字段汇入最终报告。
            **_report_required_fields(
                # 将本次模型、数据、代码和输出绑定在一起的身份记录。
                run_identity,
                # 已发布运行身份文件的字节指纹。
                run_identity_sha256=identity_sha256,
                # 8100组转换配对清单的字节指纹。
                pairs_csv_sha256=pairs_sha256,
                # 本次实际选择的转换组数。
                selected_pair_count=len(selected_pairs),
            ),
            # 记录本次运行完成状态；仅全量8100组可标记正式完成。
            "status": (
                # 按运行类型标记正式全量完成或局部诊断完成，防止混报。
                "formal_8100_complete" if formal else "diagnostic_partial_complete"
            ),
            # 记录当前数据清单声明的划分协议。
            "protocol": "FastVoiceGrad VCTK one-step objective evaluation",
            # 记录当前实验的text-unseen划分协议。
            "data_protocol": "text-unseen",
            # 记录声明采用严格的text-unseen测试隔离。
            "strict_text_unseen_protocol": True,
            # 记录本次实际新生成的WAV数量。
            "generated": generated,
            # 记录已经完整存在且身份一致、因而跳过的WAV数量。
            "skipped_existing": skipped,
            # 记录先按公式9将源mel加噪到x950。
            "initial_state": run_identity["initial_state"],
            # 记录学生只执行一次确定性的950→0重建。
            "transition": run_identity["transition"],
            # 记录100条源语音×9个其他说话人×9个不同句位。
            "pair_rule": run_identity["pair_rule"],
            # 记录保持与原教师相同的BNF到mel时间对齐方法。
            "bnf_alignment": PAPER_STUDENT_CONFIG.bnf_alignment,
            # 记录保持原教师内容条件的时间下采样方法。
            "bnf_downsampling": PAPER_STUDENT_CONFIG.bnf_downsampling,
            # 记录构造每个固定小批次随机种子的基础值1234。
            "base_seed": FORMAL_SEED,
            # 记录小批次种子由基础种子加首个pair_id决定。
            "seed_rule": run_identity["seed_rule"],
            # 记录每个推理小批次固定处理4个配对。
            "batch_size": FORMAL_BATCH_SIZE,
            # 记录使用自训修改版HiFi-GAN。
            "vocoder_kind": "custom",
            # 记录声码器已完成2500000次训练更新。
            "vocoder_global_step": int(vocoder_checkpoint["global_step"]),
            # 记录加载的学生epoch100权重路径。
            "student_checkpoint": str(args.checkpoint),
            # 记录赋予该学生检查点实验含义的训练契约。
            "student_run_contract": str(args.run_contract),
            # 记录本次冻结声码器的最终权重文件。
            "vocoder_checkpoint": str(args.vocoder_checkpoint),
            # 记录声码器的结构与音频配置。
            "vocoder_config": str(args.vocoder_config),
            # 记录声码器250万步最终检查点的审核记录。
            "vocoder_attestation": str(args.vocoder_attestation),
            # 记录声码器加载时实际采用的完整配置。
            "vocoder_runtime_config": vocoder_config,
            # 记录将本次模型、数据、代码和输出绑定在一起的身份记录。
            "run_identity": str(args.output_dir / "run_identity.json"),
            # 记录固定配对清单的发布路径。
            "pairs_csv": str(args.output_dir / "pairs.csv"),
            # 记录每个pair_id对应WAV文件的SHA-256。
            "generated_wav_sha256": generated_wav_sha256,
            # 记录按固定键顺序汇总全部输出WAV的整体指纹。
            "generated_wav_collection_sha256": canonical_json_sha256(
                # 每个pair_id对应WAV文件的SHA-256。
                generated_wav_sha256
            ),
            # 记录记录是否修复过原完成报告中的丢失或变化WAV。
            "recovered_from_prior_completion": recovery_needed,
            # 记录本次推理阶段实际耗时秒数。
            "elapsed_seconds": time.time() - start_time,
        }
        # 推理完成报告文件路径。
        report_path = args.output_dir / "inference_report.json"
        # 将报告编码为固定JSON字节后原子写入。
        atomic_write_json(report_path, report)
        # 恢复中暂存旧完成报告的路径。
        recovery_path = args.output_dir / "inference_report.recovery.json"
        # 恢复成功后若仍有旧报告标记，需要清理这个临时恢复状态。
        if recovery_path.is_file():
            # 删除已经由新完整报告替代的恢复标记，生成音频不受影响。
            recovery_path.unlink()
        # 在日志展示最终完整报告，便于直接查看模型、数量和耗时。
        print(json.dumps(report, ensure_ascii=False, indent=2))
        # 向运行日志输出当前身份、完成状态或可读报告。
        print(
            # 打印可被编排脚本识别的全量推理通过标记。
            "FASTVOICEGRAD_STUDENT_8100_INFERENCE_PASSED"
            # 根据是否全量8100组选择正式或诊断状态标记。
            if formal
            # 少量诊断结束时使用诊断通过标志，便于流水线区分。
            else "FASTVOICEGRAD_STUDENT_DIAGNOSTIC_INFERENCE_PASSED"
        )


# 仅直接运行本模块时开始正式推理，导入测试时不触发计算。
if __name__ == "__main__":
    # 启动完成参数校验、身份核验和生成的完整入口。
    main()
