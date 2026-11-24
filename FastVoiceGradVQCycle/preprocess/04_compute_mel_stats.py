#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 仅使用训练集全部有效 mel 帧计算每个频带的均值与标准差，并将两者成套发布。
"""
Compute per-channel mel mean/std from TRAIN split only.

Input:
  data/metadata/train_manifest.csv
  data/mel/<speaker>/<utt_id>.npy

Output:
  data/stats/mel_mean.npy   # shape [80]
  data/stats/mel_std.npy    # shape [80]
  data/stats/mel_stats_current.json  # atomic generation commit + SHA-256
  data/stats/.mel_stats_generations/<id>/  # immutable source pair
  exp/logs/mel_stats_report.json

Notes:
- Statistics are computed over ALL frames of ALL train utterances.
- Mean/std are computed per mel channel.
- This matches the usual VoiceGrad normalization style:
    mel_norm = (mel - mean[:, None]) / std[:, None]
"""

# =============================================================================
# 预处理第 04 步：只用训练集计算每个 mel 通道的均值和标准差
# =============================================================================
# 为什么标准化：80 个 mel 通道的数值范围不同。变成大致均值 0、标准差 1 后，
# Adam 更容易同时学习所有频带，不会被某些绝对值特别大的通道支配。
#
# 为什么只能用 train_manifest：mean/std 也是从数据“学到”的信息。如果 test 数据
# 参与计算，就构成 data leakage，所谓 unseen 测试已经不严格。
#
# 这里统计“所有训练语音的所有帧”，不是先给每句话算均值再平均。输出 [80]
# mean/std，dataset.py 会 reshape 为 [80,1] 并广播到一条语音的全部时间帧。

# --------------------------- 导入包逐项解释 ---------------------------
# csv 读取 train_manifest；json 保存统计报告；argparse 解析可选路径参数。
import csv
# hashlib 为每一代 mean/std 和报告记录 SHA-256，训练前可据此识别混代文件。
import hashlib
# 保存或读取实验设置、数据指纹及处理进度的结构化记录。
import json
# 定义终端参数，使数据目录、检查点和运行选项明确记录。
import argparse
# os.environ 让同一份代码可在个人电脑、服务器和容器中使用。
import os
# shutil 用于把不可变 generation 中的文件发布到旧版兼容路径。
import shutil
# UTC 时间和随机 UUID 共同组成不会冲突的 generation id。
from datetime import datetime, timezone
# Path 管理 stats/log 路径；Dict/List 是类型提示。
from pathlib import Path
# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Dict, List, Tuple
# 复用 uuid4 的统一实现，具体输入和实验约束由调用处指定。
from uuid import uuid4

# NumPy 加载 mel，执行按通道求和/平方/开根并保存 mean/std。
import numpy as np
# tqdm 显示遍历约四万条训练语音的进度。
from tqdm import tqdm


# 本脚本在 preprocess/ 下，parents[1] 是仓库根目录；环境变量可覆盖。
ROOT = Path(
    # 优先读取显式工程根目录；没有环境变量时回到本文件向上两级的仓库目录。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[1])
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()
# 输入必须明确是 train_manifest，而不是 all_manifest 或 test_manifest。
DEFAULT_MANIFEST = ROOT / "data" / "metadata" / "train_manifest.csv"
# 自然对数 mel 数值文件的默认目录。
DEFAULT_MEL_DIR = ROOT / "data" / "mel"
# 训练集逐频带统计量的默认输出目录。
DEFAULT_STATS_DIR = ROOT / "data" / "stats"
# 统计量来源和数值范围的默认报告路径。
DEFAULT_REPORT_PATH = ROOT / "exp" / "logs" / "mel_stats_report.json"

# 论文规定的 mel 通道数。
N_MELS = 80
# 方差理论上非负，但大量浮点运算后可能得到极小负数；EPS 也避免 std=0。
EPS = 1e-8

# 一次成功计算产生一个不可变 generation。CURRENT_MANIFEST 是唯一原子提交点；
# TRANSACTION_MARKER 在兼容路径更新中断时保留，下一次运行会先完成同一代发布。
GENERATION_DIRNAME = ".mel_stats_generations"
# 当前完整 mean/std 版本的提交清单文件名。
CURRENT_MANIFEST_NAME = "mel_stats_current.json"
# 标记 mean/std 发布尚未完成的事务文件名。
TRANSACTION_MARKER_NAME = ".mel_stats_publish_transaction.json"
# 成套统计量发布协议的版本号。
PUBLICATION_SCHEMA_VERSION = 1


# 逐块读取文件并计算字节内容指纹；下面的参数分别限定输入数据和处理约束。
def sha256_file(path: Path) -> str:
    # 逐块读取文件并计算字节内容指纹。
    """Return a streaming SHA-256 digest without loading a whole file."""

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


# 按数组形状、类型和原始字节生成数值内容指纹；下面的参数分别限定输入数据和处理约束。
def sha256_array(array: np.ndarray) -> str:
    # 按数组形状、类型和原始字节生成数值内容指纹。
    """Canonical numerical digest shared with ``validate_setup.sha256_array``."""

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


# 刷新目录记录以保证文件替换在断机后可追溯；下面的参数分别限定输入数据和处理约束。
def _fsync_directory(path: Path) -> None:
    # 刷新目录记录以保证文件替换在断机后可追溯。
    """Best-effort directory fsync (supported on the Linux training host)."""

    # 以只读目录句柄准备同步元数据；兼容系统没有 O_DIRECTORY 常量的情况。
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 打开明确的目标目录句柄，供下方 fsync 保证重命名等目录变更落盘。
        descriptor = os.open(str(path), flags)
    # 捕获刷新目录记录以保证文件替换在断机后可追溯中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except OSError:
        # 系统不支持目录同步时结束这个辅助函数，不改变已完成的文件写入。
        return
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 要求操作系统同步已写内容，提高中断后的记录可恢复性。
        os.fsync(descriptor)
    # 捕获刷新目录记录以保证文件替换在断机后可追溯中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except OSError:
        # Windows does not fsync directory handles; file fsync + os.replace still
        # gives the atomicity needed by local tests.
        # 目录 fsync 在某些文件系统不支持；此处仅忽略辅助目录同步失败，文件写入本身仍按前面的规则检查。
        pass
    # 无论处理成功或失败，都执行后面的临时资源清理。
    finally:
        # 无论同步成功与否都关闭目录句柄，避免资源泄漏。
        os.close(descriptor)


# 写 JSON 并同步到磁盘；下面的参数分别限定输入数据和处理约束。
def _write_json_file(path: Path, payload: Dict[str, object], exclusive: bool = False) -> None:
    # 写 JSON 并同步到磁盘。
    """Write and fsync JSON at a path that is not yet visible to readers."""

    # 当前选择的时间对齐或插值模式。
    mode = "x" if exclusive else "w"
    # 在受控文件句柄中完成写 JSON 并同步到磁盘，退出此段后自动关闭文件。
    with path.open(mode, encoding="utf-8") as handle:
        # 把已经核对的结构化内容写成 UTF-8 JSON，保存时保留可读的中文及缩进。
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        # JSON 报告末尾加换行，便于命令行检查。
        handle.write("\n")
        # 先将语言运行时缓冲写到文件，供后续同步落盘。
        handle.flush()
        # 要求操作系统同步已写内容，提高中断后的记录可恢复性。
        os.fsync(handle.fileno())


# 拒绝覆盖已有记录地发布新的 JSON 事务文件；下面的参数分别限定输入数据和处理约束。
def _write_json_exclusive_atomic(path: Path, payload: Dict[str, object]) -> None:
    # 拒绝覆盖已有记录地发布新的 JSON 事务文件。
    """Atomically create a JSON marker without clobbering another publisher."""

    # 准备拒绝覆盖已有记录地发布新的 JSON 事务文件所需的目标目录，已有目录按给定选项复用。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 尚未正式发布的临时文件。
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 先独占创建完整临时事务 JSON，避免同名临时文件被覆盖。
        _write_json_file(temporary, payload, exclusive=True)
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # Linking a fully fsynced same-directory temp file is an atomic
            # create-if-absent operation on the Linux training filesystem.
            # 用硬链接独占创建事务标记；已有标记会报错，从而阻止两个发布者互相覆盖。
            os.link(temporary, path)
        # 捕获拒绝覆盖已有记录地发布新的 JSON 事务文件中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except FileExistsError as error:
            # 拒绝覆盖已有记录地发布新的 JSON 事务文件未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 报错指出另一个统计量发布事务已占用标记，当前进程不能覆盖。
                f"Another mel-stats publication is active: {path}"
            # 保留硬链接创建失败的原始异常链，明确标记冲突来自另一发布事务。
            ) from error
        # 同步事务标记所在目录，降低异常断电后标记丢失的风险。
        _fsync_directory(path.parent)
    # 无论处理成功或失败，都执行后面的临时资源清理。
    finally:
        # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
        temporary.unlink(missing_ok=True)


# 写入纯数值统计数组并同步磁盘；下面的参数分别限定输入数据和处理约束。
def _write_npy_file(path: Path, value: np.ndarray) -> None:
    # 写入纯数值统计数组并同步磁盘。
    """Write and fsync an NPY inside an unpublished staging directory."""

    # 在受控文件句柄中完成写入纯数值统计数组并同步磁盘，退出此段后自动关闭文件。
    with path.open("xb") as handle:
        # 向未发布的临时 NPY 写入纯数值统计量，禁止对象序列化。
        np.save(handle, value, allow_pickle=False)
        # 先将语言运行时缓冲写到文件，供后续同步落盘。
        handle.flush()
        # 要求操作系统同步已写内容，提高中断后的记录可恢复性。
        os.fsync(handle.fileno())


# 复制完整统计文件后再替换正式路径；下面的参数分别限定输入数据和处理约束。
def _copy_file_atomic(source: Path, destination: Path) -> None:
    # 复制完整统计文件后再替换正式路径。
    """Copy an immutable generation file and atomically replace its alias."""

    # 准备复制完整统计文件后再替换正式路径所需的目标目录，已有目录按给定选项复用。
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 尚未正式发布的临时文件。
    temporary = destination.with_name(
        # 临时文件名联合目标名、进程号和随机 ID，防止并行写入重名。
        f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在受控文件句柄中完成复制完整统计文件后再替换正式路径，退出此段后自动关闭文件。
        with source.open("rb") as reader, temporary.open("xb") as writer:
            # 分块复制不可变版本文件，每次最多 1 MB，避免额外占用大量内存。
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            # 先将语言运行时缓冲写到文件，供后续同步落盘。
            writer.flush()
            # 要求操作系统同步已写内容，提高中断后的记录可恢复性。
            os.fsync(writer.fileno())
        # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
        os.replace(temporary, destination)
        # 正式文件原子替换后同步父目录，保留已发布路径。
        _fsync_directory(destination.parent)
    # 无论处理成功或失败，都执行后面的临时资源清理。
    finally:
        # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
        temporary.unlink(missing_ok=True)


# 检查统计量是 80 维有限数组并按需要求标准差为正；下面的参数分别限定输入数据和处理约束。
def _validate_stat_array(path: Path, *, require_positive: bool = False) -> np.ndarray:
    # 当前正在校验的数值或记录。
    value = np.load(path, allow_pickle=False)
    # 发布的均值和标准差都必须为 80 维 float32，不能混入时间轴或其他精度。
    if value.shape != (N_MELS,) or value.dtype != np.float32:
        # 检查统计量是 80 维有限数组并按需要求标准差为正未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明统计文件必须是 80 维单精度向量。
            f"Invalid generated statistic {path}: expected float32 [{N_MELS}], "
            # 错误附带实际数组精度和形状，便于识别错维或混用。
            f"got dtype={value.dtype}, shape={value.shape}."
        )
    # 检查数组所有元素均为有限数；NaN 或无穷值会污染后续模型输入和统计量。
    if not np.isfinite(value).all():
        # 检查统计量是 80 维有限数组并按需要求标准差为正未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Invalid generated statistic {path}: contains NaN/Inf.")
    # 标准差必须每个频带严格大于零，防止标准化时除零。
    if require_positive and not np.all(value > 0):
        # 检查统计量是 80 维有限数组并按需要求标准差为正未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Invalid generated statistic {path}: std must be positive.")
    # 返回当前正在校验的数值或记录，供调用方继续处理。
    return value


# 将统计量发布批次映射到受控归档目录；下面的参数分别限定输入数据和处理约束。
def _generation_path(stats_dir: Path, generation_id: str) -> Path:
    # IDs are created internally, but validate a persisted transaction before
    # turning it into a path so a corrupted marker cannot escape stats_dir.
    # 版本名不能为空，也不能包含下面列出的目录分隔或上级跳转字符。
    if not generation_id or any(
        # 版本名禁止目录分隔符和父目录跳转，确保路径始终位于统计量版本目录内。
        token in generation_id for token in ("/", "\\", "..")
    ):
        # 将统计量发布批次映射到受控归档目录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Unsafe mel-stats generation id: {generation_id!r}")
    # 返回已验证版本名对应的不可变统计量目录。
    return stats_dir / GENERATION_DIRNAME / generation_id


# 核对一代均值与标准差归档的元数据及指纹；下面的参数分别限定输入数据和处理约束。
def _load_and_validate_generation(
    # 同时接收统计量根目录和待完成事务，两者共同定位唯一版本。
    stats_dir: Path, transaction: Dict[str, object]
# 返回核对一代均值与标准差归档的元数据及指纹整理的 Tuple[Path, Dict[str, object]] 结果，内部各项保持前面约定的对应关系。
) -> Tuple[Path, Dict[str, object]]:
    # 只恢复当前认识的事务格式，不猜测旧格式含义。
    if transaction.get("schema_version") != PUBLICATION_SCHEMA_VERSION:
        # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错给出不受支持的事务版本，不能猜测未知发布格式。
            f"Unsupported mel-stats transaction schema: {transaction.get('schema_version')!r}"
        )
    # 这一次成套发布的均值和标准差版本标识。
    generation_id = str(transaction.get("generation_id", ""))
    # 某一不可变统计量版本的存放目录。
    generation_dir = _generation_path(stats_dir, generation_id)
    # 本次实际使用的清单文件。
    manifest_path = generation_dir / "manifest.json"
    # 事务要求不可变版本清单匹配的 SHA-256。
    expected_manifest_hash = transaction.get("generation_manifest_sha256")
    # 检查本次实际使用的清单文件是否缺失；缺失时不能继续信任该输入。
    if not manifest_path.is_file():
        # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Incomplete mel-stats generation: {manifest_path} is missing.")
    # 不可变版本清单必须与准备事务时的内容完全一致。
    if sha256_file(manifest_path) != expected_manifest_hash:
        # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Mel-stats generation manifest hash mismatch: {manifest_path}")
    # 在受控文件句柄中完成核对一代均值与标准差归档的元数据及指纹，退出此段后自动关闭文件。
    with manifest_path.open("r", encoding="utf-8") as handle:
        # 待读取的样本清单。
        manifest = json.load(handle)
    # 同时核对下方版本、文件存在性或内容约束，任何不一致都停止发布。
    if (
        # 版本清单的数据结构必须与当前发布协议相同。
        manifest.get("schema_version") != PUBLICATION_SCHEMA_VERSION
        # 版本清单内的编号必须等于事务指向的目录编号。
        or manifest.get("generation_id") != generation_id
    ):
        # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Invalid mel-stats generation manifest: {manifest_path}")

    # 版本清单中 mean/std 的逐文件尺寸、类型、形状和摘要。
    files = manifest.get("files")
    # 要求读取对象符合核对一代均值与标准差归档的元数据及指纹预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(files, dict):
        # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Missing file table in {manifest_path}")
    # 分别检查均值和标准差；只有标准差额外要求逐元素为正。
    for filename, positive in (("mel_mean.npy", False), ("mel_std.npy", True)):
        # 训练与测试清单所在目录。
        metadata = files.get(filename)
        # 要求读取对象符合核对一代均值与标准差归档的元数据及指纹预期的数据结构，否则不能继续信任其中字段。
        if not isinstance(metadata, dict):
            # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Missing {filename} metadata in {manifest_path}")
        # 输入文件或源特征。
        source = generation_dir / filename
        # 文件必须存在且字节数与不可变清单相同，先排除缺失和截断。
        if not source.is_file() or source.stat().st_size != metadata.get("bytes"):
            # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Missing or truncated generated statistic: {source}")
        # 再次比较完整内容摘要，防止大小相同但数据已改变。
        if sha256_file(source) != metadata.get("sha256"):
            # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Generated statistic hash mismatch: {source}")
        # 加载数组检查 80 维、单精度、有限数值及标准差正值约束。
        _validate_stat_array(source, require_positive=positive)

    # 本次处理的数据来源和完成情况报告。
    report = generation_dir / "report.json"
    # 不可变版本清单保存的报告大小和摘要。
    report_metadata = manifest.get("report")
    # 同时核对下方版本、文件存在性或内容约束，任何不一致都停止发布。
    if (
        # 版本清单必须附带结构化报告身份。
        not isinstance(report_metadata, dict)
        # 与 mean/std 同一版本的来源报告必须存在。
        or not report.is_file()
        # 报告大小需要和版本清单记录一致。
        or report.stat().st_size != report_metadata.get("bytes")
        # 报告内容必须与同一版本清单绑定的摘要一致。
        or sha256_file(report) != report_metadata.get("sha256")
    ):
        # 核对一代均值与标准差归档的元数据及指纹未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Generated mel-stats report is missing or corrupted: {report}")
    # 只有整套均值、标准差和报告都通过检查才返回可发布版本。
    return generation_dir, manifest


# 按预先登记的事务同时发布均值、标准差和报告；下面的参数分别限定输入数据和处理约束。
def _apply_publication_transaction(
    # mel 标准化统计量保存目录。
    stats_dir: Path,
    # 数据处理审计报告的保存位置。
    report_path: Path,
    # 待继续完成的统计量发布事务。
    transaction: Dict[str, object],
# 返回按预先登记的事务同时发布均值、标准差和报告整理的 Dict[str, object] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, object]:
    # 按预先登记的事务同时发布均值、标准差和报告。
    """Idempotently finish a prepared generation, then clear its marker."""

    # 待恢复事务指定的原报告绝对路径。
    expected_report = Path(str(transaction.get("report_path", ""))).resolve()
    # 恢复时必须写回原事务指定的报告路径，不能把同一版本发布到另一个实验。
    if expected_report != report_path.resolve():
        # 按预先登记的事务同时发布均值、标准差和报告未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错说明恢复命令给出的报告路径与原事务不一致。
            "Pending mel-stats transaction targets a different report path: "
            # 提示原事务准确的报告路径，以及重新恢复时应指定的参数。
            f"{expected_report}. Re-run with --report_path {expected_report}."
        )
    # 先完整验证不可变来源版本，再开始更新对外可见的统计量文件。
    generation_dir, manifest = _load_and_validate_generation(stats_dir, transaction)

    # These aliases preserve compatibility with existing dataset readers. They
    # are not the commit point: CURRENT_MANIFEST below is. If either replace is
    # interrupted, TRANSACTION_MARKER remains and the next run repeats both.
    # 先从同一不可变版本更新均值别名，事务标记仍在，读取者应等待整套发布完成。
    _copy_file_atomic(generation_dir / "mel_mean.npy", stats_dir / "mel_mean.npy")
    # 从同一不可变版本原子更新标准差，事务标记仍保留以阻止读取半套版本。
    _copy_file_atomic(generation_dir / "mel_std.npy", stats_dir / "mel_std.npy")
    # 将同版本统计报告复制到正式报告位置。
    _copy_file_atomic(generation_dir / "report.json", report_path)

    # A single atomic replace makes the complete, hash-bound pair current.
    # 最后才提交完整版本清单，标记 mean/std 和报告已一起更新到同一版本。
    _copy_file_atomic(generation_dir / "manifest.json", stats_dir / CURRENT_MANIFEST_NAME)

    # Verify compatibility aliases before declaring the transaction complete.
    # 提交后逐一重验均值和标准差的真实内容摘要。
    for filename in ("mel_mean.npy", "mel_std.npy"):
        # 发布后再次验证正式 mean/std 内容，确认最终路径确实对应同一版本。
        if sha256_file(stats_dir / filename) != manifest["files"][filename]["sha256"]:
            # 按预先登记的事务同时发布均值、标准差和报告未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Published mel statistic failed verification: {filename}")

    # 当前统计量发布是否尚未完成的事务标记路径。
    marker = stats_dir / TRANSACTION_MARKER_NAME
    # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
    marker.unlink()
    # 完成提交并移除事务标记后同步统计量目录。
    _fsync_directory(stats_dir)
    # 返回待读取的样本清单，供调用方继续处理。
    return manifest


# 识别并完成上次中断的统计量发布事务；下面的参数分别限定输入数据和处理约束。
def recover_pending_publication(stats_dir: Path, report_path: Path) -> bool:
    # 识别并完成上次中断的统计量发布事务。
    """Complete an interrupted publication before computing another generation."""

    # 当前统计量发布是否尚未完成的事务标记路径。
    marker = stats_dir / TRANSACTION_MARKER_NAME
    # 检查当前统计量发布是否尚未完成的事务标记路径是否缺失；缺失时不能继续信任该输入。
    if not marker.exists():
        # 不存在待恢复事务，本轮可以正常计算新统计量。
        return False
    # 在受控文件句柄中完成识别并完成上次中断的统计量发布事务，退出此段后自动关闭文件。
    with marker.open("r", encoding="utf-8") as handle:
        # 待继续完成的统计量发布事务。
        transaction = json.load(handle)
    # 把识别并完成上次中断的统计量发布事务的当前进度或结果输出到服务器日志。
    print(
        # 日志提示正在先完成上次中断的统计量版本发布。
        "[RECOVER] Completing interrupted mel-stats generation "
        # 记录待恢复的版本编号，新统计计算要等旧事务处理完。
        f"{transaction.get('generation_id')} before starting new work."
    )
    # 按原事务完成同一版本的剩余发布步骤，避免先生成新版本造成混乱。
    _apply_publication_transaction(stats_dir, report_path, transaction)
    # 明确告知调用者此前中断的统计量发布已经恢复完成。
    return True


# 建立新统计量的可恢复发布批次；下面的参数分别限定输入数据和处理约束。
def publish_stats_transaction(
    # mel 标准化统计量保存目录。
    stats_dir: Path,
    # 数据处理审计报告的保存位置。
    report_path: Path,
    # 本次训练集逐 mel 频带均值，共 80 个数。
    mean: np.ndarray,
    # 训练集逐 mel 频带标准差，供标准化和反标准化共用。
    std: np.ndarray,
    # 本次处理的数据来源和完成情况报告。
    report: Dict[str, object],
# 返回建立新统计量的可恢复发布批次整理的 Dict[str, object] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, object]:
    # 建立新统计量的可恢复发布批次。
    """Stage an immutable mean/std pair and atomically commit its manifest."""

    # 发布前把 80 维训练集均值统一为单精度数组。
    mean = np.asarray(mean, dtype=np.float32)
    # 训练集逐 mel 频带标准差，供标准化和反标准化共用。
    std = np.asarray(std, dtype=np.float32)
    # 准备发布的两个数组都必须为 80 个 mel 频带各一个值。
    if mean.shape != (N_MELS,) or std.shape != (N_MELS,):
        # 建立新统计量的可恢复发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Expected mean/std shape [{N_MELS}], got {mean.shape}/{std.shape}.")
    # 检查数组所有元素均为有限数；NaN 或无穷值会污染后续模型输入和统计量。
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.all(std > 0):
        # 建立新统计量的可恢复发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError("Mean/std must be finite, and every std value must be positive.")

    # 准备建立新统计量的可恢复发布批次所需的目标目录，已有目录按给定选项复用。
    stats_dir.mkdir(parents=True, exist_ok=True)
    # 准备建立新统计量的可恢复发布批次所需的目标目录，已有目录按给定选项复用。
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # 当前统计量发布是否尚未完成的事务标记路径。
    marker = stats_dir / TRANSACTION_MARKER_NAME
    # 仅在当前统计量发布是否尚未完成的事务标记路径已存在时进入复用或校验分支。
    if marker.exists():
        # 建立新统计量的可恢复发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错指出待恢复标记的位置，要求先完成旧事务再发布新版本。
            f"Pending publication marker still exists: {marker}. Recover it before publishing."
        )

    # 这一次成套发布的均值和标准差版本标识。
    generation_id = (
        # 版本编号先包含 UTC 微秒时间，便于追溯统计量生成顺序。
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        # 在可读时间与随机标识之间加入分隔符。
        + "-"
        # 附加随机短标识，防止并行或极短间隔运行产生相同版本目录。
        + uuid4().hex[:12]
    )
    # 保存全部历史统计量版本的目录。
    generations_dir = stats_dir / GENERATION_DIRNAME
    # 准备建立新统计量的可恢复发布批次所需的目标目录，已有目录按给定选项复用。
    generations_dir.mkdir(parents=True, exist_ok=True)
    # 尚未提交完成的统计量版本暂存目录。
    staging_dir = generations_dir / f".{generation_id}.tmp"
    # 写入完成后不可变统计量版本的正式目录。
    final_dir = _generation_path(stats_dir, generation_id)
    # 准备建立新统计量的可恢复发布批次所需的目标目录，已有目录按给定选项复用。
    staging_dir.mkdir()
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 暂存版本中的 80 维均值文件。
        mean_stage = staging_dir / "mel_mean.npy"
        # 暂存版本中的 80 维标准差文件。
        std_stage = staging_dir / "mel_std.npy"
        # 在未发布目录独占写入本轮训练集均值。
        _write_npy_file(mean_stage, mean)
        # 在同一未发布目录独占写入本轮训练集标准差。
        _write_npy_file(std_stage, std)
        # 重新读取均值临时文件，确认保存格式和有限数值正确。
        _validate_stat_array(mean_stage)
        # 重新读取标准差临时文件，额外确认每个频带均大于零。
        _validate_stat_array(std_stage, require_positive=True)

        # 补齐版本编号、mean/std 摘要后的完整统计报告。
        complete_report = dict(report)
        # 在统计来源报告中加入本次版本编号和实际写出的文件指纹。
        complete_report.update(
            # 下面展开建立新统计量的可恢复发布批次所需的这组条目；各字段随后逐一说明。
            {
                # 记录成套统计量发布协议的版本号。
                "publication_schema_version": PUBLICATION_SCHEMA_VERSION,
                # 记录这一次成套发布的均值和标准差版本标识。
                "generation_id": generation_id,
                # 记录同一版本逐频带均值 NPY 的文件指纹。
                "mean_sha256": sha256_file(mean_stage),
                # 记录同一版本逐频带标准差 NPY 的文件指纹。
                "std_sha256": sha256_file(std_stage),
                # 记录当前完整统计量版本提交清单的路径。
                "current_manifest_path": str(stats_dir / CURRENT_MANIFEST_NAME),
                # 记录某一不可变统计量版本的存放目录。
                "generation_dir": str(final_dir),
            }
        )
        # 暂存版本中的完整来源报告。
        report_stage = staging_dir / "report.json"
        # 将完整来源报告写入与 mean/std 相同的未发布版本目录。
        _write_json_file(report_stage, complete_report, exclusive=True)

        # 待读取的样本清单。
        manifest = {
            # 记录来源报告的数据结构版本号。
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            # 记录这一次成套发布的均值和标准差版本标识。
            "generation_id": generation_id,
            # 记录该不可变统计量版本创建的 UTC 时间。
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            # 记录版本清单中 mean/std 的逐文件尺寸、类型、形状和摘要。
            "files": {
                # 以下字段描述本版本均值文件的大小、类型、形状和内容摘要。
                "mel_mean.npy": {
                    # 记录当前文件的实际字节数。
                    "bytes": mean_stage.stat().st_size,
                    # 记录张量的数值类型。
                    "dtype": "float32",
                    # 记录参与广播或校验的目标张量形状。
                    "shape": [N_MELS],
                    # 记录当前文件的 SHA-256 内容指纹。
                    "sha256": sha256_file(mean_stage),
                },
                # 以下字段描述本版本标准差文件，必须与均值属于同一 generation。
                "mel_std.npy": {
                    # 记录当前文件的实际字节数。
                    "bytes": std_stage.stat().st_size,
                    # 记录张量的数值类型。
                    "dtype": "float32",
                    # 记录参与广播或校验的目标张量形状。
                    "shape": [N_MELS],
                    # 记录当前文件的 SHA-256 内容指纹。
                    "sha256": sha256_file(std_stage),
                },
            },
            # 记录本次处理的数据来源和完成情况报告。
            "report": {
                # 记录当前文件的实际字节数。
                "bytes": report_stage.stat().st_size,
                # 记录当前文件的 SHA-256 内容指纹。
                "sha256": sha256_file(report_stage),
            },
        }
        # 在暂存目录写完整版本清单，将 mean、std 和报告的内容身份绑定。
        _write_json_file(staging_dir / "manifest.json", manifest, exclusive=True)
        # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
        os.replace(staging_dir, final_dir)
        # 版本目录从临时名称改为正式名称后同步父目录。
        _fsync_directory(generations_dir)
    # 无论处理成功或失败，都执行后面的临时资源清理。
    finally:
        # This is normally absent after rename. Leave any non-empty orphan for
        # audit instead of recursively deleting possibly valuable evidence.
        # 只允许清理确实存在且完全为空的暂存目录，保留有内容的故障现场。
        if staging_dir.exists() and not any(staging_dir.iterdir()):
            # 只清理已经确认为空的暂存目录，保留有内容的中断现场供调查。
            staging_dir.rmdir()

    # 一个不可变统计量版本的内容和指纹清单。
    generation_manifest = final_dir / "manifest.json"
    # 待继续完成的统计量发布事务。
    transaction = {
        # 记录来源报告的数据结构版本号。
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        # 记录这一次成套发布的均值和标准差版本标识。
        "generation_id": generation_id,
        # 记录事务绑定的不可变版本清单内容摘要。
        "generation_manifest_sha256": sha256_file(generation_manifest),
        # 记录数据处理审计报告的保存位置。
        "report_path": str(report_path.resolve()),
    }
    # 独占发布事务标记，若另一个发布者已占用则拒绝覆盖。
    _write_json_exclusive_atomic(marker, transaction)
    # 按准备好的事务更新正式别名，最后提交清单并清除标记。
    return _apply_publication_transaction(stats_dir, report_path, transaction)


# 读取参与均值和标准差统计的训练清单；下面的参数分别限定输入数据和处理约束。
def load_rows(manifest_path: Path, limit: int = -1) -> List[Dict[str, str]]:
    # DictReader 把 CSV 行变成字典；--limit 仅用于小规模调试，正式运行保持 -1。
    if not manifest_path.exists():
        # 读取参与均值和标准差统计的训练清单未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # 本次选定的语音清单。
    rows = []
    # 在受控文件句柄中完成读取参与均值和标准差统计的训练清单，退出此段后自动关闭文件。
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        # 按表头返回字典的 CSV 读取器。
        reader = csv.DictReader(f)
        # 逐项遍历按表头返回字典的 CSV 读取器，保持每条记录都经过相同检查。
        for row in reader:
            # 按训练清单原顺序保留每条语音，统计和来源摘要使用同一顺序。
            rows.append(row)

    # 按仅为诊断而限制处理的样本数决定是否进入此分支；判断条件为 limit is not None and limit > 0。
    if limit is not None and limit > 0:
        # 本次选定的语音清单。
        rows = rows[:limit]

    # 返回本次选定的语音清单，供调用方继续处理。
    return rows


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main():
    # argparse 允许必要时覆盖输入/输出目录，默认全部指向服务器正式路径。
    parser = argparse.ArgumentParser()
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--mel_dir", type=str, default=str(DEFAULT_MEL_DIR))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--stats_dir", type=str, default=str(DEFAULT_STATS_DIR))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--report_path", type=str, default=str(DEFAULT_REPORT_PATH))
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--limit", type=int, default=-1, help="For sanity check. Use -1 for full run.")
    # 终端传入并经解析的运行设置。
    args = parser.parse_args()

    # 本次实际使用的清单文件。
    manifest_path = Path(args.manifest)
    # 本轮统计读取的自然对数 mel 目录。
    mel_dir = Path(args.mel_dir)
    # mel 标准化统计量保存目录。
    stats_dir = Path(args.stats_dir)
    # 数据处理审计报告的保存位置。
    report_path = Path(args.report_path)

    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    stats_dir.mkdir(parents=True, exist_ok=True)
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 若上次在两个兼容文件之间被断电/kill，marker 会指向已经完整落盘的
    # generation；必须先幂等完成它，不能直接计算并覆盖成另一代。
    recover_pending_publication(stats_dir, report_path)

    # 本次选定的语音清单。
    rows = load_rows(manifest_path, limit=args.limit)

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Train manifest = {manifest_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Mel dir        = {mel_dir}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Stats dir      = {stats_dir}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"[INFO] Rows to use    = {len(rows)}")

    # 不把 39412 条 mel 一次堆进内存，只维护三个 running statistics：
    # sum_x[c] 是通道 c 所有帧之和；sum_x2[c] 是所有帧平方和；
    # count_frames 是所有 utterance 的总帧数。float64 累计可减少长序列求和误差。
    # Running sums over ALL train frames
    sum_x = np.zeros((N_MELS,), dtype=np.float64)
    # 逐 mel 频带累计的平方和。
    sum_x2 = np.zeros((N_MELS,), dtype=np.float64)
    # 所有有效训练语音累计的真实 mel 帧数。
    count_frames = 0

    # 检查中发现的缺失项目。
    missing = []
    # 无法读取、维度异常或有非有限数值的 mel 记录。
    bad = []

    # 续训不仅绑定最终 mean/std，也绑定生成它们的完整训练 mel 集合。
    # content 使用 canonical 数组摘要，不受 .npy header 细节影响；layout 只读文件大小，
    # 供训练启动时低成本复核。两者都严格按 manifest 行序累计。
    mel_content_digest = hashlib.sha256()
    # 按训练清单顺序累计的 mel 身份和文件大小指纹。
    mel_layout_digest = hashlib.sha256()
    # 纳入统计量来源集合的有效 mel 文件数。
    mel_collection_count = 0

    # tqdm 只显示进度条，不改变统计计算。
    for row in tqdm(rows, total=len(rows), desc="Compute mel stats"):
        # 当前语音所属说话人。
        speaker = row["speaker"]
        # 同时包含说话人和句号的样本编号。
        utt_id = row["utt_id"]

        # 当前训练语音的 mel NPY 路径。
        mel_path = mel_dir / speaker / f"{utt_id}.npy"
        # 检查当前训练语音的 mel NPY 路径是否缺失；缺失时不能继续信任该输入。
        if not mel_path.exists():
            # 记录缺失的训练 mel 路径；任何缺失最终都会阻止发布部分统计。
            missing.append(str(mel_path))
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue

        # 每条都检查能否读取、是否二维、是否 [80,T]、T 是否非空、有无 NaN/Inf。
        try:
            # 80 维自然对数 mel 声谱图。
            mel = np.load(mel_path, allow_pickle=False)
        # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except Exception as e:
            # 保留无法读取的 mel 路径和异常信息，避免静默跳过坏文件。
            bad.append((str(mel_path), repr(e)))
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue

        # 训练 mel 必须是频带×时间的二维矩阵。
        if mel.ndim != 2:
            # 将维数异常登记为坏特征。
            bad.append((str(mel_path), f"ndim={mel.ndim}"))
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue

        # 要求第一轴为 80 个频带，防止按错误轴累计统计。
        if mel.shape[0] != N_MELS:
            # 保存实际形状，方便定位通道和时间轴颠倒的问题。
            bad.append((str(mel_path), f"shape={mel.shape}"))
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue

        # 每条语音至少需要一个有效 mel 帧。
        if mel.shape[1] <= 0:
            # 空时间轴计作坏特征，不能贡献零帧假装已覆盖。
            bad.append((str(mel_path), f"empty time axis: shape={mel.shape}"))
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue

        # 检查数组所有元素均为有限数；NaN 或无穷值会污染后续模型输入和统计量。
        if not np.isfinite(mel).all():
            # 非有限数值会污染全局均值和方差，因此单独登记并阻止最终发布。
            bad.append((str(mel_path), "contains NaN or Inf"))
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue

        # 当前特征在数据目录中的稳定相对身份。
        collection_key = f"{speaker}/{utt_id}.npy"
        # 包含类型、形状和数值字节的规范数组摘要。
        canonical_array_hash = sha256_array(mel)
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 数值内容指纹；字段顺序必须与读取审核时一致。
        mel_content_digest.update(collection_key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 数值内容指纹；字段顺序必须与读取审核时一致。
        mel_content_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 数值内容指纹；字段顺序必须与读取审核时一致。
        mel_content_digest.update(canonical_array_hash.encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 数值内容指纹；字段顺序必须与读取审核时一致。
        mel_content_digest.update(b"\n")
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 身份和文件大小指纹；字段顺序必须与读取审核时一致。
        mel_layout_digest.update(collection_key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 身份和文件大小指纹；字段顺序必须与读取审核时一致。
        mel_layout_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 身份和文件大小指纹；字段顺序必须与读取审核时一致。
        mel_layout_digest.update(str(mel_path.stat().st_size).encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 身份和文件大小指纹；字段顺序必须与读取审核时一致。
        mel_layout_digest.update(b"\n")
        # 只有通过全部数值检查的 mel 才计入来源集合数量。
        mel_collection_count += 1

        # axis=1 沿时间轴求和，结果留下 [80]，实现每个 mel channel 独立统计。
        # mel: [80, T]
        sum_x += mel.sum(axis=1, dtype=np.float64)
        # 每个频带累计所有帧的平方和，用双精度减少长语料累加误差。
        sum_x2 += np.square(mel, dtype=np.float64).sum(axis=1, dtype=np.float64)
        # 按实际语音帧数累加总分母，所有训练帧等权，而非每条语音等权。
        count_frames += mel.shape[1]

    # 按检查中发现的缺失项目决定是否进入此分支；判断条件为 missing。
    if missing:
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print(f"[WARN] Missing mel files: {len(missing)}")
    # 按无法读取、维度异常或有非有限数值的 mel 记录决定是否进入此分支；判断条件为 bad。
    if bad:
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print(f"[WARN] Bad mel files: {len(bad)}")

    # 若跳过坏文件继续算，会产生“部分训练集统计量”，看起来能用但实验不可复现。
    # 因此任意缺失或损坏都直接停止，并且不会覆盖原有 mean/std。
    if missing or bad:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 明确拒绝用缺文件或坏文件后的剩余子集计算正式标准化统计。
            "Refusing to compute partial training statistics: "
            # 列出缺失 mel 与坏 mel 的真实数量。
            f"missing={len(missing)}, bad={len(bad)}. "
            # 列出前几个具体缺失和错误记录，便于定向补齐。
            f"Examples: missing={missing[:3]}, bad={bad[:3]}"
        )

    # 没有任何有效训练帧时无法定义均值和标准差，必须停止。
    if count_frames <= 0:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("No valid train mel frames found. Cannot compute statistics.")

    # 数学公式 E[x] = sum(x)/N。
    mean = sum_x / count_frames
    # 总体方差 Var[x] = E[x^2] - E[x]^2，可由运行和与平方和直接计算。
    var = sum_x2 / count_frames - np.square(mean)
    # maximum 逐元素取较大值，消除浮点误差产生的微小负方差并避免除零。
    var = np.maximum(var, EPS)
    # 标准差等于方差平方根。
    std = np.sqrt(var)

    # 双精度累加和除法完成后，将逐频带均值转为模型实际使用的 float32。
    mean = mean.astype(np.float32)
    # 训练集逐 mel 频带标准差，供标准化和反标准化共用。
    std = std.astype(np.float32)

    # 逐频带 mel 均值文件路径。
    mean_path = stats_dir / "mel_mean.npy"
    # 逐频带 mel 标准差文件路径。
    std_path = stats_dir / "mel_std.npy"

    # 报告保存使用行数、总帧数、缺失/坏文件数及统计范围，便于审计。
    report = {
        # 记录待读取的样本清单。
        "manifest": str(manifest_path),
        # 记录所用清单的文件指纹。
        "manifest_sha256": sha256_file(manifest_path),
        # 记录本轮统计读取的自然对数 mel 目录。
        "mel_dir": str(mel_dir),
        # 记录mel 标准化统计量保存目录。
        "stats_dir": str(stats_dir),
        # 记录计划纳入统计的训练语音数。
        "rows_used": len(rows),
        # 记录实际纳入均值与标准差计算的总训练帧数。
        "valid_total_frames": int(count_frames),
        # 记录清单中缺失的 mel 文件数量。
        "missing_count": len(missing),
        # 记录读取或数值校验失败的 mel 文件数量。
        "bad_count": len(bad),
        # 记录本次统计实际使用的 mel 数值集合指纹。
        "mel_collection_content_sha256": mel_content_digest.hexdigest(),
        # 记录本次统计实际使用的 mel 文件布局指纹。
        "mel_collection_layout_sha256": mel_layout_digest.hexdigest(),
        # 记录纳入统计量来源集合的有效 mel 文件数。
        "mel_collection_count": mel_collection_count,
        # 记录逐频带 mel 均值文件路径。
        "mean_path": str(mean_path),
        # 记录逐频带 mel 标准差文件路径。
        "std_path": str(std_path),
        # 记录最终逐频带均值的数组形状，应为 [80]。
        "mean_shape": list(mean.shape),
        # 记录最终逐频带标准差的数组形状，应为 [80]。
        "std_shape": list(std.shape),
        # 记录80 个频带均值中的最小值。
        "mean_min": float(mean.min()),
        # 记录80 个频带均值中的最大值。
        "mean_max": float(mean.max()),
        # 记录80 个频带均值的算术平均，仅供数值检查。
        "mean_avg": float(mean.mean()),
        # 记录80 个频带标准差中的最小值。
        "std_min": float(std.min()),
        # 记录80 个频带标准差中的最大值。
        "std_max": float(std.max()),
        # 记录80 个频带标准差的平均值，仅供数值检查。
        "std_avg": float(std.mean()),
        # 记录缺失 mel 的有限数量路径示例。
        "missing_examples": missing[:10],
        # 记录无效 mel 的路径与错误原因示例。
        "bad_examples": [{"path": p, "error": e} for p, e in bad[:10]],
    }

    # generation 目录先完整写入并验证；兼容路径更新时 marker 始终存在；
    # mel_stats_current.json 最后通过单次 os.replace 成为原子提交点。
    current_manifest = publish_stats_transaction(
        # mel 标准化统计量保存目录。
        stats_dir=stats_dir,
        # 数据处理审计报告的保存位置。
        report_path=report_path,
        # 传入与标准差同一次计算得到的训练集逐频带均值。
        mean=mean,
        # 训练集逐 mel 频带标准差，供标准化和反标准化共用。
        std=std,
        # 本次处理的数据来源和完成情况报告。
        report=report,
    )

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("\n[DONE] Mel stats computed.")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  total valid frames = {count_frames}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  missing = {len(missing)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  bad     = {len(bad)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  mean_path = {mean_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  std_path  = {std_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  mean shape = {mean.shape}, min={mean.min():.6f}, max={mean.max():.6f}, avg={mean.mean():.6f}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  std  shape = {std.shape}, min={std.min():.6f}, max={std.max():.6f}, avg={std.mean():.6f}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  report = {report_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  generation = {current_manifest['generation_id']}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"  current manifest = {stats_dir / CURRENT_MANIFEST_NAME}")


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 直接运行才重新计算并覆盖 stats；被 dataset.py import 时不会自动执行。
    main()
