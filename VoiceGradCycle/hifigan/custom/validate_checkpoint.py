#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 加载归档前联合核对权重、配置、数据与源码，任何一项不符都拒绝出具通过证明。
"""Fail-closed validation for an archived custom HiFi-GAN generator.

The custom vocoder checkpoint is a Python/PyTorch artifact, not merely a bag of
weights.  Before inference, this module binds together all of the facts that make
the artifact meaningful:

* the archive schema and the declared ``global_step``;
* the step encoded in ``generator_step_XXXXXXX.pt``;
* the complete canonical JSON training configuration;
* immutable data fingerprints embedded in both config and checkpoint; and
* a strict load into the audited HiFi-GAN V1 ``Generator`` implementation.

Only checkpoints produced by this repository and obtained through a trusted path
should be passed here.  PyTorch 1.13 cannot safely inspect every legacy pickle
without unpickling it, so the intentionally explicit ``weights_only=False`` is a
trust-boundary decision, not a claim that arbitrary checkpoint files are safe.
"""

# 延后求值类型提示，保持旧 Python 解释器兼容。
from __future__ import annotations

# 使用 argparse 读取命令行参数。
import argparse
# 使用 hashlib 计算数据与权重文件指纹。
import hashlib
# 使用 importlib 按独立模块名加载官方实现。
import importlib
# 使用 json 读写配置及来源记录。
import json
# 使用 os 执行路径、原子替换和进程级辅助操作。
import os
# 使用 re 严格检查字符串格式。
import re
# 使用 sys 处理解释器版本和模块搜索路径。
import sys
# 使用 tempfile 安全建立临时结果文件。
import tempfile
# 使用 warnings 控制已知依赖兼容警告。
import warnings
# 检查配置和状态确实是键值映射。
from collections.abc import Mapping
# 以带 UTC 时区的时间记录检查结果。
from datetime import datetime, timezone
# 用 Path 统一构造和检查文件路径。
from pathlib import Path
# 标明张量、清单和配置的输入输出类型，不改变运行数据。
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

# 使用 PyTorch 读取可信检查点并逐个验证生成器权重。
import torch

# 限制实际计算只能在当前租用服务器执行。
from voicegrad.runtime_guard import RuntimeIdentityError, assert_rented_server


# 只接受当前第二版归档结构。
EXPECTED_FORMAT_VERSION = 2
# 此入口核对用于推理的轻量生成器档案，不把完整训练状态误当同类文件。
EXPECTED_CHECKPOINT_TYPE = "generator_weights_only"
# 固定此次自训声码器的结构标签；结构声明与权重加载都要核对。
EXPECTED_ARCHITECTURE = "HiFi-GAN-V1 + MPD + UnivNet-MRD (MSD replaced)"

# Values disclosed by FastVoiceGrad/HiFi-GAN/UnivNet, plus the explicitly fixed
# VCTK protocol and reproducibility choices for this one formal run.  The complete
# config is compared separately; this table gives precise diagnostics for the
# settings whose accidental alteration would invalidate the experiment.
# 这些值混合论文公开项和本项目固定选择；完整来源见配置说明，不能统称作者原始设定。
REQUIRED_CONFIG_VALUES: Dict[str, Any] = {
    # 记录数据划分协议名称。
    "data_protocol": "text-unseen",
    # 记录生成器所用残差块类型。
    "resblock": "1",
    # 记录配置中使用的 GPU 数。
    "num_gpus": 1,
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
    # 记录随机种子。
    "seed": 1234,
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
    # 记录历史配置保留的频点字段；实际 STFT 频点由 n_fft 决定。
    "num_freq": 1025,
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
    # 记录数据读取子进程数。
    "num_workers": 4,
    # 记录声码器计划训练总更新次数。
    "max_steps": 2_500_000,
    # 记录终端和标量日志记录间隔。
    "log_every_steps": 10,
    # 记录可续训完整检查点保存间隔。
    "save_every_steps": 10_000,
    # 记录生成器长期归档间隔。
    "archive_generator_every_steps": 500_000,
    # 记录本划分预期训练语音总数。
    "expected_train_utterances": 39_387,
    # 记录本划分预期训练说话人数。
    "expected_train_speakers": 100,
    # 记录测试语音数，仅核对划分，不参与训练。
    "expected_test_utterances": 100,
    # 记录未见测试说话人数。
    "expected_test_speakers": 10,
    # 记录MRD 各分支的 FFT 点数。
    "mrd_fft_sizes": [1024, 2048, 512],
    # 记录MRD 各分支的频谱帧移。
    "mrd_hop_sizes": [120, 240, 50],
    # 记录MRD 各分支的窗长。
    "mrd_win_lengths": [600, 1200, 240],
    # 记录MRD LeakyReLU 负半轴斜率。
    "mrd_lrelu_slope": 0.2,
}

# 数据指纹只接受 64 位小写十六进制文本。
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# 优先读取明确的服务器根目录环境配置。
DEFAULT_PROJECT_ROOT = Path(
    # 未设环境路径时，以当前源码所在项目根目录为准。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[2])
# 展开用户目录写法，后续流程再严格解析绝对路径。
).expanduser()


# 统一表示任何检查不通过，调用方不得继续把文件认证为有效。
class CheckpointValidationError(RuntimeError):
    # 任何归档属性核对不通过都会抛出此错误，调用方不能继续认证。
    """Raised when any attested property fails closed."""


# 生成唯一的 JSON 表示，用于严格比较全部配置内容。
def canonical_json(value: Any) -> str:
    # 使用固定键顺序和分隔符编码配置，严格区分数字、布尔值和文本。
    """Return the sole JSON representation used for exact semantic comparison."""

    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 以固定排序和分隔符输出配置内容。
        return json.dumps(
            # 传入待比较或计算指纹的完整对象。
            value,
            # 中文说明按原字符保留，不转成 ASCII 转义。
            ensure_ascii=False,
            # 按字段名排序，让同内容对象获得相同编码。
            sort_keys=True,
            # 删除可有可无的空格，保证内容哈希使用唯一文本。
            separators=(",", ":"),
            # 拒绝 JSON 标准外的 NaN 和无穷大。
            allow_nan=False,
        )
    # 对象不能严格编码时中止，不能跳过不可保存字段。
    except (TypeError, ValueError) as error:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告配置中无法按严格 JSON 保存的对象或数值。
            f"Value is not canonical-JSON serializable: {error}"
        # 保留原始底层异常，核对记录能追溯真正失败原因。
        ) from error


# 把字典按固定 JSON 排列编码后计算指纹，避免键顺序影响身份比较。
def canonical_sha256(value: Any) -> str:
    # 把字典按固定 JSON 排列编码后计算指纹，避免键顺序影响身份比较。
    """Hash a value's canonical UTF-8 JSON representation."""

    # 对统一 UTF-8 JSON 内容计算语义指纹。
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# 按二进制文件内容计算 SHA-256，供后续检查是否混入其他版本。
def sha256_file(path: Path) -> str:
    # 按二进制文件内容计算 SHA-256，供后续检查是否混入其他版本。
    """Stream a file through SHA256 without loading another full copy in RAM."""

    # 初始化文件内容指纹累积器。
    digest = hashlib.sha256()
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 按二进制读取，文件编码或换行不会被自动转换。
        with path.open("rb") as handle:
            # 逐块读到文件末尾，避免复制整个大检查点到内存。
            while True:
                # 每次读取 1 MiB 文件内容。
                block = handle.read(1024 * 1024)
                # 空块说明读取完成。
                if not block:
                    # 结束已完成的文件分块读取。
                    break
                # 把当前二进制块加入同一个文件指纹。
                digest.update(block)
    # 文件读取或哈希失败时，保留操作系统的具体原因。
    except OSError as error:
        # 报错原因：文件读取或哈希失败时，保留操作系统的具体原因。
        raise CheckpointValidationError(f"Cannot hash file {path}: {error}") from error
    # 返回便于写入 JSON 的十六进制文件指纹。
    return digest.hexdigest()


# 解析 JSON 时显式拒绝 NaN、Infinity 等非标准常量。
def _reject_json_constant(token: str) -> None:
    # 报错原因：解析 JSON 时显式拒绝 NaN、Infinity 等非标准常量。
    raise CheckpointValidationError(f"Non-finite JSON constant is forbidden: {token}")


# 从原始键值对构造字典，先检查重复字段，避免后值悄悄覆盖前值。
def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    # 收集当前 JSON 对象的唯一字段。
    result: Dict[str, Any] = {}
    # 逐个检查解析器读到的原始键值对。
    for key, value in pairs:
        # 相同键第二次出现会中止，配置身份不允许含糊。
        if key in result:
            # 报错原因：相同键第二次出现会中止，配置身份不允许含糊。
            raise CheckpointValidationError(f"Duplicate JSON key is forbidden: {key!r}")
        # 只登记第一次出现且合法的字段。
        result[key] = value
    # 返回本函数已经构造或校验的结果，供调用方继续处理。
    return result


# 读取严格 JSON 对象，拒绝重复键、非有限数和非字典根节点。
def load_json_object(path: Path) -> Dict[str, Any]:
    # 保留原始键顺序进行重复键检查，并拒绝 NaN 与 Infinity 等非标准 JSON 值。
    """Read one strict JSON object, rejecting duplicates and NaN/Infinity."""

    # 输入路径必须指向真实文件，缺失时立即失败。
    if not path.is_file():
        # 报错原因：输入路径必须指向真实文件，缺失时立即失败。
        raise CheckpointValidationError(f"Config file does not exist: {path}")
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 按 UTF-8 读取配置文本。
        with path.open("r", encoding="utf-8") as handle:
            # 用加强的解析规则读取整份配置。
            value = json.load(
                # 把当前打开的配置文件交给 JSON 解析器。
                handle,
                # 保留原始键值序列用于重复键检查。
                object_pairs_hook=_unique_object,
                # 遇到非标准浮点常量立即报错。
                parse_constant=_reject_json_constant,
            )
    # 已有明确核对错误时保留原异常。
    except CheckpointValidationError:
        # 把已经带具体上下文的错误原样抛给调用方。
        raise
    # 区分文件、编码和 JSON 语法问题，均不能继续认证。
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        # 报错原因：区分文件、编码和 JSON 语法问题，均不能继续认证。
        raise CheckpointValidationError(f"Cannot read strict JSON config {path}: {error}") from error
    # 配置最外层必须是键值对象。
    if not isinstance(value, dict):
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            f"Config must contain one JSON object, got {type(value).__name__}"
        )
    # 返回经过本函数类型和格式核对的数值或对象。
    return value


# 步数等正整数明确排除布尔值，防止 True 被视为 1。
def _require_positive_integer(value: Any, label: str) -> int:
    # 值必须是真正整数且大于零。
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        # 报错原因：值必须是真正整数且大于零。
        raise CheckpointValidationError(f"{label} must be a positive integer, got {value!r}")
    # 返回经过本函数类型和格式核对的数值或对象。
    return value


# 轮次和批次游标允许零，但不能为负、浮点数或布尔值。
def _require_nonnegative_integer(value: Any, label: str) -> int:
    # 检查该恢复游标为真正的非负整数。
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            f"{label} must be a non-negative integer, got {value!r}"
        )
    # 返回经过本函数类型和格式核对的数值或对象。
    return value


# 检查配置满足本项目明确约定的声码器结构和训练设置。
def validate_config(config: Mapping[str, Any]) -> None:
    # 检查配置满足本项目明确约定的声码器结构和训练设置。
    """Check the experiment's fixed, scientifically material config values."""

    # 逐项核对会影响实验身份的固定设置。
    for key, expected in REQUIRED_CONFIG_VALUES.items():
        # 要求关键字段显式出现，不能用库默认值代替证据。
        if key not in config:
            # 报错原因：要求关键字段显式出现，不能用库默认值代替证据。
            raise CheckpointValidationError(f"Config is missing required key {key!r}")
        # 取出配置中实际声明的关键值。
        actual = config[key]
        # Python considers True == 1, including inside lists.  Canonical JSON does
        # not, and neither should a scientific config validator.
        # 用严格 JSON 比较类型和值，避免 True 与 1 被认为相同。
        if canonical_json(actual) != canonical_json(expected):
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            raise CheckpointValidationError(
                # 指出具体配置项及两侧数值，不能仅凭结构标签通过认证。
                f"Critical config mismatch for {key}: got {actual!r}, expected {expected!r}"
            )


# 核对非空的数据名称—SHA-256 表，绑定训练数据版本。
def validate_data_fingerprints(value: Any, label: str) -> Dict[str, str]:
    # 每个数据来源名称必须唯一，且对应完整小写 SHA-256。
    """Validate and copy a non-empty ``name -> lowercase SHA256`` object."""

    # 缺少数据指纹或空表都不能认证。
    if not isinstance(value, Mapping) or not value:
        # 报错原因：缺少数据指纹或空表都不能认证。
        raise CheckpointValidationError(f"{label} must be a non-empty mapping")
    # 收集已核对格式的数据指纹。
    result: Dict[str, str] = {}
    # 逐一检查每项数据名称和指纹。
    for key, digest in value.items():
        # 数据名称必须非空且没有外围空白，避免视觉相同但实际不同的键。
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            raise CheckpointValidationError(
                # 定位为空、带外围空白或类型不合法的数据指纹名称。
                f"{label} contains an invalid fingerprint name: {key!r}"
            )
        # 相同键第二次出现会中止，配置身份不允许含糊。
        if key in result:
            # 报错原因：相同键第二次出现会中止，配置身份不允许含糊。
            raise CheckpointValidationError(f"{label} contains duplicate name {key!r}")
        # 每个数据指纹都必须符合完整 SHA-256 格式。
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            raise CheckpointValidationError(
                # 报告当前核对失败的具体属性，阻止出具通过证明。
                f"{label}[{key!r}] must be a lowercase 64-hex SHA256 digest"
            )
        # 保存通过名称及指纹格式检查的数据项。
        result[key] = digest
    # 返回本函数已经构造或校验的结果，供调用方继续处理。
    return result


# 根据训练步数构造官方归档命名，核对文件名是否与内部进度一致。
def expected_checkpoint_filename(step: int) -> str:
    # 归档名中的七位步数必须与检查点内部累计更新数一致。
    """Return the exact filename emitted by ``custom.train`` for a step archive."""

    # 把累计步数补成至少七位，保持训练脚本的命名格式。
    return f"generator_step_{step:07d}.pt"


# 隔离同名模块后导入本项目审计过的官方 G，避免意外加载其他 models。
def import_official_hifigan(project_root: Path):
    # 取得当前项目官方 G 接口；检查点验证器还临时隔离并恢复同名模块缓存。
    """Import AttrDict/Generator from this checkout, never from site-packages.

    Upstream ``models.py`` imports ``utils`` by its bare module name.  We therefore
    isolate and restore those three temporary module names so an earlier unrelated
    import cannot silently select a different implementation.
    """

    # 指定当前项目内保留的官方 HiFi-GAN 源码目录。
    official_dir = project_root / "hifigan" / "official"
    # 严格加载依赖这三份源码，缺一份都不能确认网络实现。
    required_sources = tuple(official_dir / name for name in ("env.py", "models.py", "utils.py"))
    # 列出缺失的官方源文件，便于定位部署问题。
    missing = [str(path) for path in required_sources if not path.is_file()]
    # 缺少审计源码时停止导入。
    if missing:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 列出缺失的官方源码文件，阻止导入未知替代实现。
            f"Audited official HiFi-GAN sources are missing: {missing}"
        )

    # 官方代码用裸模块名导入，这三个名字需临时隔离。
    module_names = ("env", "models", "utils")
    # 保存进程中已有的同名模块，加载结束后恢复。
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    # 保存原模块搜索路径。
    old_path = list(sys.path)
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 逐个处理可能与其他项目冲突的裸模块名。
        for name in module_names:
            # 移除当前缓存中的同名模块，使接下来从指定官方目录重新解析。
            sys.modules.pop(name, None)
        # 把本项目官方源码放到搜索路径首位。
        sys.path.insert(0, str(official_dir))
        # 导入官方配置属性包装器。
        env_module = importlib.import_module("env")
        # 导入当前官方生成器实现。
        models_module = importlib.import_module("models")
        # 返回经过路径隔离的配置包装器与生成器类。
        return env_module.AttrDict, models_module.Generator
    # 已有明确核对错误时保留原异常。
    except CheckpointValidationError:
        # 把已经带具体上下文的错误原样抛给调用方。
        raise
    # 把底层导入、张量或严格加载异常附上当前检查上下文。
    except Exception as error:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 记录指定官方目录和具体导入异常。
            f"Cannot import audited HiFi-GAN Generator from {official_dir}: {error}"
        # 保留原始底层异常，核对记录能追溯真正失败原因。
        ) from error
    # 无论前面成功或失败，都执行资源恢复和关闭操作。
    finally:
        # 恢复原搜索路径，不把临时导入配置留给其他模块。
        sys.path[:] = old_path
        # 逐个处理可能与其他项目冲突的裸模块名。
        for name in module_names:
            # 移除当前缓存中的同名模块，使接下来从指定官方目录重新解析。
            sys.modules.pop(name, None)
            # 取得隔离前该名称的模块对象。
            previous = saved_modules[name]
            # 仅在之前确有模块时恢复缓存。
            if previous is not None:
                # 把原模块放回，避免检查器改变其他组件的导入身份。
                sys.modules[name] = previous


# 在 CPU 构建生成器并严格加载每个参数，拒绝缺项、意外项或非有限数。
def strict_load_generator(
    # 输入参数：租用服务器上的项目根目录。
    project_root: Path,
    # 输入参数：模型配置。
    config: Dict[str, Any],
    # 输入参数：待核对的生成器参数与缓冲区映射。
    state_dict: Any,
# 返回张量有限性及参数、缓冲区数量等严格加载证据。
) -> Dict[str, Any]:
    # 在 CPU 构建生成器并严格加载每个参数，拒绝缺项、意外项或非有限数。
    """Instantiate the audited Generator on CPU and strict-load the archive."""

    # 生成器状态必须是非空参数映射。
    if not isinstance(state_dict, Mapping) or not state_dict:
        # 报错原因：生成器状态必须是非空参数映射。
        raise CheckpointValidationError("Checkpoint 'generator' must be a non-empty state dict")
    # 逐个核对权重字段，而非仅确认检查点能打开。
    for name, tensor in state_dict.items():
        # 每个条目必须由文本参数名映射到真实张量。
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            raise CheckpointValidationError(
                # 报告当前核对失败的具体属性，阻止出具通过证明。
                "Checkpoint 'generator' must map string parameter names to tensors; "
                # 报告当前核对失败的具体属性，阻止出具通过证明。
                f"got {name!r} -> {type(tensor).__name__}"
            )
        # 浮点和复数权重需要逐元素有限值检查。
        if tensor.is_floating_point() or tensor.is_complex():
            # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
            try:
                # 确认当前张量所有元素都不是 NaN 或无穷大。
                finite = bool(torch.isfinite(tensor).all().item())
            # 把底层导入、张量或严格加载异常附上当前检查上下文。
            except Exception as error:
                # 报告当前核对失败的具体属性，阻止出具通过证明。
                raise CheckpointValidationError(
                    # 定位无法完成有限值检查的权重张量。
                    f"Cannot verify finiteness of Generator tensor {name!r}: {error}"
                # 保留原始底层异常，核对记录能追溯真正失败原因。
                ) from error
            # 任何非有限权重都会使模型认证失败。
            if not finite:
                # 报告当前核对失败的具体属性，阻止出具通过证明。
                raise CheckpointValidationError(
                    # 报告当前核对失败的具体属性，阻止出具通过证明。
                    f"Generator tensor {name!r} contains NaN or Inf"
                )

    # 取得路径已经核实的官方 G 定义。
    AttrDict, Generator = import_official_hifigan(project_root)
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 将已知归一化兼容警告限制在本次模型构造范围。
        with warnings.catch_warnings():
            # 仅隐藏旧 weight_norm API 的未来弃用提醒，不忽略参数错误。
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
            # 设置将 mel 转为波形的生成器 G。
            generator = Generator(AttrDict(config))
            # 严格加载全部 G 参数，缺项或多项直接抛错。
            incompatible = generator.load_state_dict(state_dict, strict=True)
    # 把底层导入、张量或严格加载异常附上当前检查上下文。
    except Exception as error:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 保留严格加载失败的缺键、形状或其他具体原因。
            f"Generator strict state-dict load failed: {error}"
        # 保留原始底层异常，核对记录能追溯真正失败原因。
        ) from error

    # PyTorch strict=True already raises, but checking the return object makes the
    # attestation fail closed even if a future/custom loader changes that behavior.
    # 再检查加载器返回的缺失键列表，作为额外保险。
    missing = list(getattr(incompatible, "missing_keys", ()))
    # 再检查权重多出的字段列表。
    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
    # 即使某版本加载器没有抛错，有任何字段不符仍拒绝认证。
    if missing or unexpected:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            f"Generator strict load returned missing={missing}, unexpected={unexpected}"
        )

    # 设置权重映射中的张量数量。
    tensor_count = len(state_dict)
    # 设置全部权重张量元素总数。
    tensor_numel = sum(int(tensor.numel()) for tensor in state_dict.values())
    # 设置模型可学习参数元素总数。
    parameter_numel = sum(int(parameter.numel()) for parameter in generator.parameters())
    # 设置模型非参数缓冲区元素总数。
    buffer_numel = sum(int(buffer.numel()) for buffer in generator.buffers())
    # 把已核对的字段组成返回记录，供后续身份检查或报告保存。
    return {
        # 声明已实际扫描全部浮点和复数权重的有限性。
        "all_floating_and_complex_tensors_finite": True,
        # 记录检查点中的张量个数。
        "state_tensor_count": tensor_count,
        # 记录状态字典中的张量元素总数。
        "state_tensor_numel": tensor_numel,
        # 记录实例化模型的参数元素总数。
        "model_parameter_numel": parameter_numel,
        # 记录模型缓冲区元素总数。
        "model_buffer_numel": buffer_numel,
    }


# 加载本项目可信训练检查点，兼容包含配置和状态的完整对象。
def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    # 输入路径必须指向真实文件，缺失时立即失败。
    if not path.is_file():
        # 报错原因：输入路径必须指向真实文件，缺失时立即失败。
        raise CheckpointValidationError(f"Checkpoint file does not exist: {path}")
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # Explicit for PyTorch 1.13 compatibility and stable behavior after the 2.6
        # default flip.  This is safe only for this trusted, locally trained file.
        # 仅加载可信的本项目归档；反序列化可执行对象，不能用于任意陌生权重。
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # 把底层导入、张量或严格加载异常附上当前检查上下文。
    except Exception as error:
        # 报错原因：把底层导入、张量或严格加载异常附上当前检查上下文。
        raise CheckpointValidationError(f"Cannot load checkpoint {path}: {error}") from error
    # 检查点根节点必须是带元数据的映射。
    if not isinstance(checkpoint, Mapping):
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            f"Checkpoint root must be a mapping, got {type(checkpoint).__name__}"
        )
    # 返回已读入并核对根类型的可信归档。
    return checkpoint


# 联合核对文件名、内部步数、配置、数据指纹及 G 参数，形成未自签名的证明。
def validate_checkpoint(
    # 输入参数：租用服务器上的项目根目录。
    project_root: Path,
    # 输入参数：模型权重路径。
    checkpoint_path: Path,
    # 输入参数：模型配置路径。
    config_path: Path,
    # 输入参数：此次核对要求的累计训练步数。
    expected_step: int,
# 返回尚未加入证明自身哈希的检查点核对记录。
) -> Dict[str, Any]:
    # 把文件名进度、内部游标、整份配置、数据指纹与实际 G 参数联合核对。
    """Validate a generator archive and return the unsigned attestation payload."""

    # 确定租用服务器上的项目根目录。
    project_root = project_root.expanduser().resolve()
    # 确定模型权重路径。
    checkpoint_path = checkpoint_path.expanduser().resolve()
    # 确定模型配置路径。
    config_path = config_path.expanduser().resolve()
    # 设置此次核对要求的累计训练步数。
    expected_step = _require_positive_integer(expected_step, "expected_step")

    # 由要求的步数得到唯一合法归档文件名。
    expected_name = expected_checkpoint_filename(expected_step)
    # 文件名与要认证的步数不一致时拒绝继续。
    if checkpoint_path.name != expected_name:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 显示与要求训练步数不对应的实际归档文件名。
            f"Checkpoint filename mismatch: got {checkpoint_path.name!r}, "
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            f"expected {expected_name!r}"
        )

    # 设置模型配置。
    config = load_json_object(config_path)
    # 先核对外部配置的关键实验设定。
    validate_config(config)
    # 核对外部配置中记录的数据版本。
    config_fingerprints = validate_data_fingerprints(
        # 明确指出正在核对配置侧的数据指纹表。
        config.get("data_fingerprints"), "config.data_fingerprints"
    )

    # 设置模型权重文件。
    checkpoint = _load_checkpoint(checkpoint_path)
    # 确定本检查器允许认证的版本、类型和网络结构。
    required_metadata = {
        # 记录检查点结构版本。
        "format_version": EXPECTED_FORMAT_VERSION,
        # 记录完整训练状态或仅生成器权重的类型。
        "checkpoint_type": EXPECTED_CHECKPOINT_TYPE,
        # 记录声码器 G、MPD、MRD 的结构组合。
        "architecture": EXPECTED_ARCHITECTURE,
    }
    # 逐项验证归档顶层身份。
    for key, expected in required_metadata.items():
        # 从归档中读取当前身份字段，缺失会在比较时失败。
        actual = checkpoint.get(key)
        # ``format_version=True`` must not pass as integer version 1/2.
        # 同时比较类型和数值，避免布尔值冒充版本号。
        if type(actual) is not type(expected) or actual != expected:  # noqa: E721
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            raise CheckpointValidationError(
                # 显示归档元数据哪个字段与固定版本、类型或结构不符。
                f"Checkpoint {key} mismatch: got {actual!r}, expected {expected!r}"
            )

    # 从权重内部读取严格正整数的累计更新次数。
    checkpoint_step = _require_positive_integer(
        # 从权重内部取得累计更新次数，并给错误信息标明字段位置。
        checkpoint.get("global_step"), "checkpoint.global_step"
    )
    # 权重内部累计步数必须与此次认证要求一致。
    if checkpoint_step != expected_step:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 比较内部真实保存进度与用户要求认证的训练步数。
            f"Checkpoint global_step mismatch: got {checkpoint_step}, expected {expected_step}"
        )
    # 读取已经完成的训练轮次。
    checkpoint_epoch = _require_nonnegative_integer(
        # 核对权重内部记录的已完成轮数。
        checkpoint.get("epoch"), "checkpoint.epoch"
    )
    # 读取本轮已完成的批次数，核对中途归档位置。
    checkpoint_batch_in_epoch = _require_nonnegative_integer(
        # 核对当前轮已完成批次游标。
        checkpoint.get("batch_in_epoch"), "checkpoint.batch_in_epoch"
    )
    # 设置每轮完整批次数。
    batches_per_epoch = (
        # 训练丢弃不足一批的尾部，完整批次数按向下取整计算。
        int(config["expected_train_utterances"]) // int(config["batch_size"])
    )
    # 本轮游标不能超过一轮可能拥有的批次数。
    if checkpoint_batch_in_epoch > batches_per_epoch:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 错误信息说明轮内游标超过实际可用批次数。
            f"checkpoint.batch_in_epoch exceeds {batches_per_epoch}: "
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            f"{checkpoint_batch_in_epoch}"
        )
    # 用完整轮数和本轮游标独立重建累计更新数。
    cursor_step = checkpoint_epoch * batches_per_epoch + checkpoint_batch_in_epoch
    # 进度字段彼此不一致说明归档身份有问题。
    if cursor_step != checkpoint_step:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 错误信息说明轮次与批次重建出的累计进度不符。
            "Checkpoint cursor/global_step mismatch: "
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            f"epoch={checkpoint_epoch}, batch_in_epoch={checkpoint_batch_in_epoch}, "
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            f"global_step={checkpoint_step}, cursor_step={cursor_step}"
        )

    # 读取直接随该权重归档的完整配置。
    embedded_config = checkpoint.get("config")
    # 权重必须带完整配置，单独外部 JSON 不足以证明来源。
    if not isinstance(embedded_config, Mapping):
        # 报错原因：权重必须带完整配置，单独外部 JSON 不足以证明来源。
        raise CheckpointValidationError("Checkpoint 'config' must be a mapping")
    # Material keys are checked independently for actionable diagnostics.  The
    # canonical comparison below then covers every note, frequency and future key.
    # 独立核对检查点内部关键设定。
    validate_config(embedded_config)
    # 外部与内部整份配置必须精确一致，连说明字段也不能静默不同。
    if canonical_json(embedded_config) != canonical_json(config):
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            "Checkpoint embedded config does not exactly match the supplied config "
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            "under canonical JSON"
        )
    # 对当前完整配置重新计算内容指纹。
    config_semantic_sha256 = canonical_sha256(config)
    # 读取归档时保存的配置内容指纹。
    embedded_config_digest = checkpoint.get("config_semantic_sha256")
    # 组合检查下面各项：保存的配置指纹必须是字符串。
    if (
        # 保存的配置指纹必须是字符串。
        not isinstance(embedded_config_digest, str)
        # 配置指纹必须满足完整 SHA-256 形式。
        or SHA256_PATTERN.fullmatch(embedded_config_digest) is None
        # 保存指纹必须与当前整份配置重新计算结果一致。
        or embedded_config_digest != config_semantic_sha256
    ):
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            "Checkpoint config_semantic_sha256 does not match the complete canonical config"
        )

    # 核对权重内部携带的训练数据指纹。
    checkpoint_fingerprints = validate_data_fingerprints(
        # 指定检查点侧的数据指纹表，方便错误定位。
        checkpoint.get("data_fingerprints"), "checkpoint.data_fingerprints"
    )
    # 权重和配置必须声明同一组数据内容。
    if checkpoint_fingerprints != config_fingerprints:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            "Checkpoint data_fingerprints do not exactly match config.data_fingerprints"
        )

    # 真正实例化 G 并严格加载，验证权重结构和有限性。
    generator_summary = strict_load_generator(
        # 设置租用服务器上的项目根目录。
        project_root=project_root,
        # 设置模型配置。
        config=dict(config),
        # 从已有记录取得待核对的生成器参数与缓冲区映射。
        state_dict=checkpoint.get("generator"),
    )

    # 指定当前项目内保留的官方 HiFi-GAN 源码目录。
    official_dir = project_root / "hifigan" / "official"
    # 在所有核对完成后形成完整检查点证明。
    report: Dict[str, Any] = {
        # 证明记录自身的结构版本，与模型检查点版本分开。
        "attestation_format_version": 1,
        # 记录流程完成状态。
        "status": "verified",
        # 记录此次核对发生的 UTC 时刻。
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        # 记录模型权重文件。
        "checkpoint": str(checkpoint_path),
        # 记录检查点文件大小。
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        # 重新计算被核对模型文件全部字节的指纹。
        "checkpoint_sha256": sha256_file(checkpoint_path),
        # 记录模型配置。
        "config": str(config_path),
        # 记录配置文件完整字节指纹。
        "config_file_sha256": sha256_file(config_path),
        # 记录按固定 JSON 顺序计算的配置指纹。
        "config_canonical_sha256": config_semantic_sha256,
        # 记录检查点结构版本。
        "format_version": EXPECTED_FORMAT_VERSION,
        # 记录完整训练状态或仅生成器权重的类型。
        "checkpoint_type": EXPECTED_CHECKPOINT_TYPE,
        # 记录声码器 G、MPD、MRD 的结构组合。
        "architecture": EXPECTED_ARCHITECTURE,
        # 记录已完成的优化器更新次数。
        "global_step": checkpoint_step,
        # 记录训练轮次。
        "epoch": checkpoint_epoch,
        # 记录本轮已经完成的批次数。
        "batch_in_epoch": checkpoint_batch_in_epoch,
        # 记录根据内部进度要求的归档文件名。
        "expected_filename": expected_name,
        # 记录训练数据及预处理记录的指纹。
        "data_fingerprints": config_fingerprints,
        # 给完整数据指纹表再计算整体内容指纹。
        "data_fingerprints_canonical_sha256": canonical_sha256(config_fingerprints),
        # 记录生成器参数完整性核对结果。
        "generator_strict_load": True,
        # 记录将 mel 转为波形的生成器 G。
        "generator": generator_summary,
        # 记录执行源码的完整字节指纹。
        "implementation_sha256": {
            # 绑定配置包装源码的所有字节；纯注释改动也会改变这个值。
            "hifigan/official/env.py": sha256_file(official_dir / "env.py"),
            # 绑定实际 G 结构源码的所有字节，新增注释版需新证明，不能修改旧证明。
            "hifigan/official/models.py": sha256_file(official_dir / "models.py"),
            # 绑定生成器初始化与辅助工具源码的所有字节。
            "hifigan/official/utils.py": sha256_file(official_dir / "utils.py"),
        },
    }
    # 返回已经通过全部核对的证明内容。
    return report


# 先写临时 JSON 再原子替换，减少中断时留下半份报告的风险。
def atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    # 先写临时 JSON 再原子替换，减少中断时留下半份报告的风险。
    """Durably publish JSON via a same-directory temporary file and replace."""

    # 先把证明输出目标解析成明确绝对路径。
    destination = destination.expanduser().resolve()
    # 创建证明文件所在目录。
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 记录本函数创建的临时文件，仅在失败时清理这一文件。
    temporary_name: Optional[str] = None
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 安全创建同目录唯一临时文件，避免并发重名。
        descriptor, temporary_name = tempfile.mkstemp(
            # 临时文件与目标处于同一文件系统，支持原子替换。
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        # 把新建文件描述符作为 UTF-8 文本写入器管理。
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            # 将完整证明写入临时文件。
            json.dump(
                # 传入待保存的完整记录。
                payload,
                # 把当前打开的配置文件交给 JSON 解析器。
                handle,
                # 中文说明按原字符保留，不转成 ASCII 转义。
                ensure_ascii=False,
                # 按字段名排序，让同内容对象获得相同编码。
                sort_keys=True,
                # 使用两空格缩进，便于人工审阅证明。
                indent=2,
                # 拒绝 JSON 标准外的 NaN 和无穷大。
                allow_nan=False,
            )
            # 文件尾加换行，保持文本报告格式。
            handle.write("\n")
            # 先刷新 Python 层写入缓冲。
            handle.flush()
            # 要求操作系统把临时文件内容同步到存储。
            os.fsync(handle.fileno())
        # 内容完整落盘后原子发布到正式证明路径。
        os.replace(temporary_name, destination)
        # 正式发布已成功，后续清理不能再删除它。
        temporary_name = None
        # Best effort directory fsync on Linux.  Windows may reject opening a
        # directory descriptor; the file-level flush + atomic replace still holds.
        # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
        try:
            # Linux 上打开父目录，尽力同步替换产生的目录项。
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        # 平台不支持目录同步时保留已经完成的文件同步结果。
        except OSError:
            # 明确记录没有可用目录描述符。
            directory_fd = None
        # 只有目录确实打开成功才执行同步和关闭。
        if directory_fd is not None:
            # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
            try:
                # 同步目录项，让原子替换信息更可靠地落盘。
                os.fsync(directory_fd)
            # 无论前面成功或失败，都执行资源恢复和关闭操作。
            finally:
                # 关闭目录描述符，不泄漏系统资源。
                os.close(directory_fd)
    # 文件发布或严格 JSON 序列化失败时，不出具成功标记。
    except (OSError, TypeError, ValueError) as error:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告独立证明文件未能完整发布，不能输出通过状态。
            f"Cannot atomically write attestation {destination}: {error}"
        # 保留原始底层异常，核对记录能追溯真正失败原因。
        ) from error
    # 无论前面成功或失败，都执行资源恢复和关闭操作。
    finally:
        # 只清理本次尚未正式发布的临时文件。
        if temporary_name is not None:
            # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
            try:
                # 删除本函数创建的未完成证明临时文件，模型和配置不受影响。
                os.unlink(temporary_name)
            # 临时文件已经不存在时无需再处理。
            except FileNotFoundError:
                # 允许清理对象已经不存在，保留原失败原因。
                pass


# 在核对通过后加入证明自身指纹，并原子发布到独立文件。
def attest_checkpoint(
    # 输入参数：租用服务器上的项目根目录。
    project_root: Path,
    # 输入参数：模型权重路径。
    checkpoint_path: Path,
    # 输入参数：模型配置路径。
    config_path: Path,
    # 输入参数：此次核对要求的累计训练步数。
    expected_step: int,
    # 独立证明的保存路径，不允许覆盖模型或配置。
    output_report: Path,
# 返回已经包含自身内容指纹、并写入磁盘的完整证明。
) -> Dict[str, Any]:
    # 验证通过后对证明本身计算内容指纹，再保存到独立报告路径。
    """Validate, self-hash the evidence payload, and atomically publish it."""

    # 确定租用服务器上的项目根目录。
    project_root = project_root.expanduser().resolve()
    # 确定模型权重路径。
    checkpoint_path = checkpoint_path.expanduser().resolve()
    # 确定模型配置路径。
    config_path = config_path.expanduser().resolve()
    # 将证明目标解析为准确绝对路径。
    output_report = output_report.expanduser().resolve()
    # 证明输出不能覆盖模型权重或配置本身。
    if output_report in {checkpoint_path, config_path}:
        # 报告当前核对失败的具体属性，阻止出具通过证明。
        raise CheckpointValidationError(
            # 报告当前核对失败的具体属性，阻止出具通过证明。
            "output_report must not overwrite the checkpoint or supplied config"
        )

    # 完成所有检查后才进入签注和发布步骤。
    report = validate_checkpoint(
        # 设置租用服务器上的项目根目录。
        project_root=project_root,
        # 设置模型权重路径。
        checkpoint_path=checkpoint_path,
        # 设置模型配置路径。
        config_path=config_path,
        # 设置此次核对要求的累计训练步数。
        expected_step=expected_step,
    )
    # 对尚未含自哈希字段的整份证明计算内容指纹。
    report["attestation_payload_sha256"] = canonical_sha256(report)
    # 可靠地发布新证明；旧实验的证明应保存在独立原路径。
    atomic_write_json(output_report, report)
    # 返回已经通过全部核对的证明内容。
    return report


# 把相对路径放到明确项目根目录下，绝对路径直接规范化。
def _resolve_from_project(project_root: Path, value: Path) -> Path:
    # 展开用户目录写法，随后再进行绝对路径解析。
    value = value.expanduser()
    # 统一得到准确绝对路径，避免当前工作目录改变目标。
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


# 读取运行路径、设备和诊断选项，默认值保持原实验约定。
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # 设置命令行参数解析器。
    parser = argparse.ArgumentParser(
        # 命令行帮助明确本工具只核对并出具自训 G 的归档证明。
        description="Strictly validate and attest a custom HiFi-GAN generator archive."
    )
    # 定义命令行选项：租用服务器上的项目根目录。
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    # 定义命令行选项：模型权重文件。
    parser.add_argument("--checkpoint", type=Path, required=True)
    # 定义命令行选项：模型配置。
    parser.add_argument("--config", type=Path, required=True)
    # 定义命令行选项：此次核对要求的累计训练步数。
    parser.add_argument("--expected-step", type=int, required=True)
    # 要求显式指定新证明位置，避免猜测覆盖旧记录。
    parser.add_argument("--output-report", type=Path, required=True)
    # 解析传入参数列表；空值时使用实际命令行。
    return parser.parse_args(argv)


# 执行命令行核对流程，并以退出码告知调用脚本是否通过。
def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    # 读取要认证的模型、配置、步数和报告位置。
    args = parse_args(argv)
    # 确定租用服务器上的项目根目录。
    project_root = args.project_root.expanduser().resolve()
    # 设置模型权重文件。
    checkpoint = _resolve_from_project(project_root, args.checkpoint)
    # 设置模型配置。
    config = _resolve_from_project(project_root, args.config)
    # 把相对证明路径基于本项目根目录解析。
    output_report = _resolve_from_project(project_root, args.output_report)
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # The validator instantiates the full Generator and scans every tensor, so
        # it is project compute and must obey the same positive rental identity
        # gate as training/inference/evaluation.
        # 严格加载和张量检查属于计算任务，只允许在当前租用服务器。
        assert_rented_server(project_root)
        # 设置核对结果。
        report = attest_checkpoint(
            # 设置租用服务器上的项目根目录。
            project_root=project_root,
            # 设置模型权重路径。
            checkpoint_path=checkpoint,
            # 设置模型配置路径。
            config_path=config,
            # 设置此次核对要求的累计训练步数。
            expected_step=args.expected_step,
            # 把明确的新证明输出路径交给核对发布函数。
            output_report=output_report,
        )
    # 模型核对或租用服务器身份检查失败都会返回失败状态。
    except (CheckpointValidationError, RuntimeIdentityError) as error:
        # 向标准错误输出核对失败的具体原因。
        print(f"CUSTOM HIFIGAN CHECKPOINT VALIDATION FAILED: {error}", file=sys.stderr)
        # 返回非零退出码，调用脚本必须停止后续训练或推理。
        return 2

    # 仅在模型、配置、游标和源码证明全部核对通过后显示成功。
    print("CUSTOM HIFIGAN CHECKPOINT VERIFIED")
    # 向终端显示模型权重文件。
    print(f"checkpoint={report['checkpoint']}")
    # 显示通过核对的检查点完整字节指纹。
    print(f"checkpoint_sha256={report['checkpoint_sha256']}")
    # 向终端显示已完成的优化器更新次数。
    print(f"global_step={report['global_step']}")
    # 向终端显示检查点核对证明。
    print(f"attestation={output_report}")
    # 全部检查完成，返回成功退出码。
    return 0


# 执行本脚本的输入检查、主要处理和结果记录。
def main() -> None:
    # 把检查器结果作为进程退出状态交给 shell。
    raise SystemExit(run_cli())


# 仅作为脚本直接运行时执行入口，导入模块时不启动任务。
if __name__ == "__main__":
    # 启动本脚本的完整处理流程。
    main()
