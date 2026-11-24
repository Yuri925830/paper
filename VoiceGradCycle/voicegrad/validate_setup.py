# 核对 VCTK 身份与文本划分，并验证统计量和整套特征来源。
"""验证两种 100+10 VCTK 协议以及 VoiceGrad 训练所需的全部特征。

本文件同时认识 ``text-unseen`` 和 ``text-seen``。它不会凭目录名猜协议，而是读取
``data/metadata/split_protocol.json``；这样可以防止“文件夹叫 text_seen，里面却
仍然是 text_unseen 清单”这种肉眼很难发现的错误。
"""

# =============================================================================
# 为什么训练前一定要运行这个脚本？
# =============================================================================
# 一次 500-epoch 训练非常耗时。若数据少了一个目录、某个 .npy 维度写反、
# train/test 说话人重叠，模型可能跑几天后才暴露问题，甚至不报错但结果无效。
# 本脚本在占用 GPU 前做“体检”：
#   1. train/test/all 清单是否存在且非空，并确认没有残留 validation 清单；
#   2. 是否为论文所述的 100 train + 10 unseen test；
#   3. train 与 test 是否完全没有说话人交集；
#   4. 每个 test speaker 是否正好具有共同的 001--010 十句话；
#   5. text-unseen 是否彻底排除 001--010；text-seen 是否确实保留并覆盖测试文本；
#   6. all_manifest 是否恰好等于 train+test，没有重复、漏项或错误 split；
#   7. mel mean/std 是否由当前训练清单原子发布，且不存在未完成 transaction；
#   8. BNF/speaker 报告是否绑定固定权重、当前清单和完整逐文件内容账本；
#   9. mel/BNF/speaker embedding 文件是否存在、维度正确且无 NaN/Inf。
#
# 默认模式扫描完整特征布局并每人抽一条形状；--full 还重算每条内容摘要，正式训练必须用。
# 该脚本只读数据，不会修改、删除或重新生成任何特征。

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# --------------------------- 导入包逐项解释 ---------------------------
# argparse 解析 --project-root/--full；csv/json 读取清单与统计报告；sys 打印错误。
import argparse
# 按清单表头读取说话人、句子及文件路径，并写回实验索引。
import csv
# hashlib binds manifests, immutable statistics, and feature-ledger summaries.
# 用标准库 SHA-256 为数据、模型或来源记录计算可比较的内容指纹。
import hashlib
# json 读取 04_compute_mel_stats.py 写出的审计报告，确认统计量覆盖当前训练清单。
import json
# 处理解释器路径和旧特征提取器模块的导入位置。
import sys
# Counter 统计说话人和句子数；Path 管理路径；Dict/List 仅作类型提示。
from collections import Counter
# 用路径对象定位数据、权重和报告，减少手工拼接目录的错误。
from pathlib import Path
# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Dict, List

# NumPy 加载 .npy，并用 isfinite 检查 NaN/Inf、用 any 检查非法 std。
import numpy as np

# 统一配置告诉体检程序应期待 80/144/256 维以及 100/10 人。
from .config import PAPER_CONFIG, SERVER_PROJECT_ROOT


# 这两个名字必须与 preprocess/02_build_fastvoicegrad_split.py 写入的值一致。
# 这里不用导入划分器，是为了让验证器保持只读且可单独运行。
PROTOCOL_TEXT_UNSEEN = "text-unseen"
# 测试句子在其他训练说话人中出现过的协议名称。
PROTOCOL_TEXT_SEEN = "text-seen"
# 固定本次复现重建的十位测试人 ID；论文明确人数，具体 ID 依据官方演示证据补足。
EXPECTED_TEST_SPEAKERS = {
    # 固定测试说话人 p238；其所有语音均不能进入模型训练。
    "p238",
    # 固定测试说话人 p241；其所有语音均不能进入模型训练。
    "p241",
    # 固定测试说话人 p243；其所有语音均不能进入模型训练。
    "p243",
    # 固定测试说话人 p252；其所有语音均不能进入模型训练。
    "p252",
    # 固定测试说话人 p261；其所有语音均不能进入模型训练。
    "p261",
    # 固定测试说话人 p294；其所有语音均不能进入模型训练。
    "p294",
    # 固定测试说话人 p303；其所有语音均不能进入模型训练。
    "p303",
    # 固定测试说话人 p330；其所有语音均不能进入模型训练。
    "p330",
    # 固定测试说话人 p334；其所有语音均不能进入模型训练。
    "p334",
    # 固定测试说话人 p360；其所有语音均不能进入模型训练。
    "p360",
}
# 固定公共测试文本编号 001—010，确保 seen/unseen 比较使用同一批测试句。
EXPECTED_TEST_SENTENCE_IDS = {f"{index:03d}" for index in range(1, 11)}
# 要求包含逐文件内容证据的特征来源报告版本。
FEATURE_REPORT_SCHEMA_VERSION = 2
# 均值与标准差成套发布清单的格式版本。
MEL_STATS_PUBLICATION_SCHEMA_VERSION = 1
# 允许使用的外部提取器唯一 Git 提交。
EXPECTED_LEGACY_COMMIT = "b59cb9862cf4b82a3bdb589950e25cab85fc9b03"
# BNF 和身份提取器各自唯一允许的预训练权重摘要。
EXPECTED_FEATURE_CHECKPOINTS = {
    # 记录表示语音内容的 144 维 BNF 特征。
    "bnf": "94362c99f408cbe891785537e222c2f1f9119edb11b95107b9faa4c0f0d679a5",
    # 记录目标说话人的身份向量。
    "speaker_embedding": "bc7ff82ef75becd495aab2ede3a8220da393a717f178ae9534df355a6173bbca",
}
# 两类提取器各自唯一允许的配置文件摘要。
EXPECTED_FEATURE_CONFIGS = {
    # 记录表示语音内容的 144 维 BNF 特征。
    "bnf": "d8d3df407a9b88500cf62e32a8bdef78068b3a09e9f8bc47904843bef1fff6c4",
    # 记录目标说话人的身份向量。
    "speaker_embedding": "1ef3e1e56df97e078d02a8da5a5a47ef5e6a0950c5139d4440660fca54bb3946",
}
# 旧 BNF 位置编码兼容源码的固定内容摘要。
EXPECTED_BNF_COMPAT_EMBEDDING = (
    # 固定旧版 BNF 位置编码兼容源码的完整 SHA-256 摘要。
    "a9631e63a7d799c91b4d2a279f17520d98bc66e5be81cc2817957e75b1bf7a2e"
)


# 从样本编号提取句号并拒绝非法命名；下面的参数分别限定输入数据和处理约束。
def sentence_id(utt_id: str) -> str:
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 取完整语音编号最后的公共句号，供检查测试文本是否进入训练。
        return utt_id.rsplit("_", 1)[1]
    # 捕获从样本编号提取句号并拒绝非法命名中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except (AttributeError, IndexError) as error:
        # 从样本编号提取句号并拒绝非法命名未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Invalid utterance id: {utt_id!r}") from error


# 统一空格与大小写以核对文本重叠情况；下面的参数分别限定输入数据和处理约束。
def normalise_text(text: str) -> str:
    # 统一空格与大小写以核对文本重叠情况。
    """忽略大小写和重复空格，得到只用于“文本是否相同”检查的字符串。"""

    # 仅为文本一致性检查统一大小写与空白，不改变保存的官方原文。
    return " ".join(text.casefold().split())


# 逐块读取文件并计算字节内容指纹；下面的参数分别限定输入数据和处理约束。
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # 逐块读取文件并计算字节内容指纹。
    """Return a streaming SHA-256 without loading a whole artifact into RAM."""

    # 逐块累计文件内容的 SHA-256 对象。
    digest = hashlib.sha256()
    # 在受控文件句柄中完成逐块读取文件并计算字节内容指纹，退出此段后自动关闭文件。
    with path.open("rb") as handle:
        # 每次读取 1 MB 文件字节参与指纹计算，直到文件结束，避免把大权重一次载入内存。
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
            digest.update(chunk)
    # 返回小写十六进制 SHA-256，用于比较本次与历史文件是否完全一致。
    return digest.hexdigest()


# 按数组形状、类型和原始字节生成数值内容指纹；下面的参数分别限定输入数据和处理约束。
def sha256_array(array: np.ndarray) -> str:
    # 按数组形状、类型和原始字节生成数值内容指纹。
    """Match the canonical numerical hash written by feature extractors."""

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


# 构造说话人和句子共同确定的特征身份；下面的参数分别限定输入数据和处理约束。
def feature_key(row: Dict[str, str]) -> str:
    # 把说话人目录和句子编号拼成稳定的相对特征名，作为来源台账的唯一键。
    return f"{row['speaker']}/{row['utt_id']}.npy"


# 固定清单身份顺序并计算整组样本的摘要；下面的参数分别限定输入数据和处理约束。
def manifest_keys_sha256(rows: List[Dict[str, str]]) -> str:
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


# 检查记录中的指纹是否为规范的 64 位十六进制格式；下面的参数分别限定输入数据和处理约束。
def _is_sha256(value: object) -> bool:
    # 返回下方三个条件的共同结果，只有完整满足才算合法 SHA-256 文字。
    return (
        # 摘要必须是字符串，不能用缺失值或数字替代。
        isinstance(value, str)
        # SHA-256 小写十六进制表示必须恰好有 64 个字符。
        and len(value) == 64
        # 每个字符只能是小写十六进制数字，拒绝不规范或伪造的摘要字段。
        and all(character in "0123456789abcdef" for character in value)
    )


# 用固定字段排序计算审计对象的内容身份；下面的参数分别限定输入数据和处理约束。
def _canonical_json_sha256(value: object) -> str:
    # 准备保存的检查点或审计内容。
    payload = json.dumps(
        # 当前正在校验的数值或记录。
        value,
        # 规范 JSON 保留真实 Unicode 字符，再统一编码计算指纹。
        ensure_ascii=False,
        # 固定 JSON 字段顺序，避免字典排列改变摘要。
        sort_keys=True,
        # 去掉不必要空格，采用固定 JSON 字节形式。
        separators=(",", ":"),
    # 将规范化 JSON 转成 UTF-8 字节供 SHA-256 处理。
    ).encode("utf-8")
    # 将规范化后的字节内容转成 SHA-256 指纹，避免仅凭文件名称认定版本一致。
    return hashlib.sha256(payload).hexdigest()


# 读取特征并核对约定形状、float32 类型和有限值；下面的参数分别限定输入数据和处理约束。
def _load_strict_feature(path: Path, feature_kind: str) -> np.ndarray:
    # 当前正在校验的数值或记录。
    value = np.load(path, allow_pickle=False)
    # 按BNF 或说话人嵌入的特征类别决定是否进入此分支；判断条件为 feature_kind == "bnf"。
    if feature_kind == "bnf":
        # BNF 必须是 [144,T]，时间轴至少一帧，不能只凭任一轴恰好 144 判断。
        valid_shape = value.ndim == 2 and value.shape[0] == PAPER_CONFIG.n_bnf and value.shape[1] > 0
    # 按BNF 或说话人嵌入的特征类别决定是否进入此分支；判断条件为 feature_kind == "speaker_embedding"。
    elif feature_kind == "speaker_embedding":
        # 规范身份特征必须是单个 256 维向量。
        valid_shape = value.shape == (PAPER_CONFIG.spk_emb_dim,)
    # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
    else:
        # 读取特征并核对约定形状、float32 类型和有限值未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Unknown feature kind: {feature_kind}")
    # 形状、float32 精度和有限数值全部通过才接受该特征。
    if not valid_shape or value.dtype != np.float32 or not np.isfinite(value).all():
        # 读取特征并核对约定形状、float32 类型和有限值未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 错误给出具体特征种类、路径、形状和精度，定位不规范输入。
            f"Invalid canonical {feature_kind} {path}: shape={value.shape}, dtype={value.dtype}"
        )
    # 返回当前正在校验的数值或记录，供调用方继续处理。
    return value


# 把现有特征逐项对照其音频、提取器和发布记录；下面的参数分别限定输入数据和处理约束。
def audit_feature_provenance(
    # 本次实验的项目根目录。
    root: Path,
    # 本次选定的语音清单。
    rows: List[Dict[str, str]],
    # BNF 或说话人嵌入的特征类别。
    feature_kind: str,
    # 从这里起的参数必须写明名称传入，减少路径或审核开关位置混淆。
    *,
    # 默认扫描全部身份与文件尺寸；显式 full 时进一步重读每个数组和输入 WAV 计算内容指纹。
    check_content: bool = False,
    # 已核对输入 WAV 信息的缓存，避免跨特征重复读取。
    input_wav_cache: Dict[str, Dict[str, object]] = None,
# 返回把现有特征逐项对照其音频、提取器和发布记录整理的 Dict[str, str] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, str]:
    # 把现有特征逐项对照其音频、提取器和发布记录。
    """Validate one complete feature report and return checkpoint-stable fingerprints.

    Normal startup scans the full key/size layout but trusts the extractor's content
    ledger.  ``--full`` additionally loads every array and recomputes its numerical
    digest; this is expensive once, not on each resume inside ``train.py``.
    """

    # 按已核对输入 WAV 信息的缓存，避免跨特征重复读取决定是否进入此分支；判断条件为 input_wav_cache is None。
    if input_wav_cache is None:
        # 已核对输入 WAV 信息的缓存，避免跨特征重复读取。
        input_wav_cache = {}
    # 按BNF 或说话人嵌入的特征类别决定是否进入此分支；判断条件为 feature_kind == "bnf"。
    if feature_kind == "bnf":
        # 数据处理审计报告的保存位置。
        report_path = root / "exp" / "logs" / "bnf_extraction_report.json"
        # 当前特征类别在 data 下的正式目录。
        feature_dir = root / "data" / "bnf"
        # 报告中声明当前特征维数的字段名称。
        expected_shape_name = "expected_bnf_dim"
        # 当前特征应有的通道宽度，BNF 144 或身份向量 256。
        expected_shape_value = PAPER_CONFIG.n_bnf
    # 按BNF 或说话人嵌入的特征类别决定是否进入此分支；判断条件为 feature_kind == "speaker_embedding"。
    elif feature_kind == "speaker_embedding":
        # 数据处理审计报告的保存位置。
        report_path = root / "exp" / "logs" / "spk_emb_extraction_report.json"
        # 当前特征类别在 data 下的正式目录。
        feature_dir = root / "data" / "spk_emb"
        # 报告中声明当前特征维数的字段名称。
        expected_shape_name = "expected_embed_dim"
        # 当前特征应有的通道宽度，BNF 144 或身份向量 256。
        expected_shape_value = PAPER_CONFIG.spk_emb_dim
    # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
    else:
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Unknown feature kind: {feature_kind}")

    # 在受控文件句柄中完成把现有特征逐项对照其音频、提取器和发布记录，退出此段后自动关闭文件。
    with report_path.open("r", encoding="utf-8") as handle:
        # 本次处理的数据来源和完成情况报告。
        report = json.load(handle)
    # 旧版没有完整逐文件证据的来源报告不能用于当前正式训练。
    if report.get("report_schema_version") != FEATURE_REPORT_SCHEMA_VERSION:
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 指出来源报告没有当前逐文件证据结构，需要明确重提取。
            f"{report_path} is legacy/unversioned; regenerate features with --force."
        )
    # 报告所属的 BNF 或身份特征类别必须与当前检查对象一致。
    if report.get("feature_kind") != feature_kind:
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Feature kind mismatch in {report_path}")
    # 正式特征必须完整覆盖且全部来源统一验证，部分完成或来源混杂都会阻断训练。
    if report.get("status") != "complete" or report.get("provenance_status") != "uniform_verified":
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Incomplete or mixed provenance in {report_path}")
    # 只接受全部新生成、全部可信复用或二者混合三种可追溯来源。
    if report.get("source_mode") not in {
        # 允许全部特征由本次固定提取器新生成。
        "all_generated",
        # 允许全部特征在逐文件来源验证后复用。
        "all_reused_verified",
        # 允许新生成与来源验证后复用混合，但不能夹入无证据文件。
        "generated_and_verified_reused",
    }:
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Unknown source mode in {report_path}: {report.get('source_mode')!r}")
    # 报告声明的特征维数必须与当前模型要求一致。
    if report.get(expected_shape_name) != expected_shape_value:
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Feature dimension mismatch in {report_path}")

    # 本轮实际重新提取成功的文件数。
    generated = report.get("generated_rows")
    # 报告声称通过来源验证后复用的样本数量。
    reused = report.get("reused_verified_rows")
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # 新生成计数必须是整数。
        not isinstance(generated, int)
        # 可信复用计数必须是整数。
        or not isinstance(reused, int)
        # 新生成计数不能为负。
        or generated < 0
        # 可信复用计数不能为负。
        or reused < 0
        # 新生成加可信复用必须恰好覆盖当前清单所有语音。
        or generated + reused != len(rows)
        # 未经来源验证便复用的条目必须为零。
        or report.get("unverified_reused_rows") != 0
        # 任何提取失败都不能在正式集合中被忽略。
        or report.get("failed_rows") != 0
        # 报告计划总数必须与当前实际清单条数一致。
        or report.get("total_rows") != len(rows)
    ):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Invalid generated/reused counters in {report_path}")

    # 本次实际使用的清单文件。
    manifest_path = root / "data" / "metadata" / "all_manifest.csv"
    # 待读取的样本清单。
    manifest = report.get("manifest")
    # 要求读取对象符合把现有特征逐项对照其音频、提取器和发布记录预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(manifest, dict):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Missing manifest attestation in {report_path}")
    # 当前实际清单 CSV 的文件指纹。
    expected_manifest_sha = sha256_file(manifest_path)
    # 当前清单语音身份及顺序的指纹。
    expected_keys_sha = manifest_keys_sha256(rows)
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # 来源报告绑定的清单文件内容必须与现在相同。
        manifest.get("sha256") != expected_manifest_sha
        # 语音身份和清单顺序也必须一致，避免同数量换样本。
        or manifest.get("keys_sha256") != expected_keys_sha
        # 清单记录的条数必须与本次输入相同。
        or manifest.get("rows") != len(rows)
    ):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Feature report does not match the active all_manifest: {report_path}")

    # 固定源码提交、配置和权重组成的提取器身份。
    provenance = report.get("output_provenance")
    # 当前特征种类唯一允许的预训练权重内容指纹。
    expected_checkpoint = EXPECTED_FEATURE_CHECKPOINTS[feature_kind]
    # 当前特征种类唯一允许的模型配置指纹。
    expected_config = EXPECTED_FEATURE_CONFIGS[feature_kind]
    # 要求读取对象符合把现有特征逐项对照其音频、提取器和发布记录预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(provenance, dict):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Missing model provenance in {report_path}")
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # 报告登记的实际外部提取器提交必须等于已审核版本。
        provenance.get("legacy_git_commit") != EXPECTED_LEGACY_COMMIT
        # 报告的基准提交字段也必须一致，防止自行改写允许版本。
        or provenance.get("legacy_git_expected_commit") != EXPECTED_LEGACY_COMMIT
        # 实际提取权重摘要必须匹配当前特征种类的固定模型。
        or provenance.get("checkpoint_sha256") != expected_checkpoint
        # 报告内声明的允许权重也必须与当前固定基准相同。
        or provenance.get("checkpoint_expected_sha256") != expected_checkpoint
        # 实际模型及预处理配置摘要必须匹配固定基准。
        or provenance.get("config_sha256") != expected_config
        # 报告内配置基准字段不能偏离当前审核版本。
        or provenance.get("config_expected_sha256") != expected_config
    ):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Unaudited model provenance in {report_path}")
    # 按BNF 或说话人嵌入的特征类别决定是否进入此分支；判断条件为 feature_kind == "bnf" and (。
    if feature_kind == "bnf" and (
        # 额外读取 BNF 所用旧版位置编码兼容源码的实际摘要。
        provenance.get("compat_embedding_sha256")
        # 兼容源码必须匹配固定内容摘要，不能只保证模块名字相同。
        != EXPECTED_BNF_COMPAT_EMBEDDING
        # 同时核对报告所声明的位置编码基准摘要。
        or provenance.get("compat_embedding_expected_sha256")
        # 兼容源码必须匹配固定内容摘要，不能只保证模块名字相同。
        != EXPECTED_BNF_COMPAT_EMBEDDING
    ):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Unaudited ESPnet compatibility module in {report_path}")

    # 按语音唯一身份建立的逐文件来源台账。
    records = report.get("file_records")
    # 要求读取对象符合把现有特征逐项对照其音频、提取器和发布记录预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(records, dict):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Missing per-file content ledger in {report_path}")
    # 当前清单要求覆盖的全部语音身份。
    expected_keys = [feature_key(row) for row in rows]
    # 由语音唯一键快速找到输入 WAV 的清单索引。
    rows_by_key = {feature_key(row): row for row in rows}
    # 逐文件来源台账必须与当前清单一一对应，多条或少条都拒绝。
    if set(records) != set(expected_keys) or len(records) != len(expected_keys):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Feature ledger is not exactly the active manifest: {report_path}")

    # 按清单顺序累计特征数值内容的指纹。
    content_digest = hashlib.sha256()
    # 按清单顺序累计特征身份和文件大小的指纹。
    layout_digest = hashlib.sha256()
    # 按清单顺序累计输入 WAV 内容的指纹。
    input_content_digest = hashlib.sha256()
    # 按清单顺序累计输入 WAV 文件大小的指纹。
    input_layout_digest = hashlib.sha256()
    # 从每条台账动作重新统计的新生成数。
    actual_generated = 0
    # 从每条台账动作重新统计的可信复用数。
    actual_reused = 0
    # 逐项遍历当前清单要求覆盖的全部语音身份，保持每条记录都经过相同检查。
    for key in expected_keys:
        # 当前特征的内容、形状、输入音频及处理动作记录。
        record = records.get(key)
        # 要求读取对象符合把现有特征逐项对照其音频、提取器和发布记录预期的数据结构，否则不能继续信任其中字段。
        if not isinstance(record, dict):
            # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Malformed feature record {key} in {report_path}")
        # 历史台账绑定的当前特征数值摘要。
        expected_array_hash = record.get("array_sha256")
        # 历史台账或版本清单记录的文件字节数。
        expected_bytes = record.get("bytes")
        # 历史特征记录的数值精度标识。
        expected_dtype = record.get("dtype")
        # 历史特征记录的规范数组形状。
        expected_shape = record.get("shape")
        # 历史台账绑定的输入 WAV 内容摘要。
        expected_wav_hash = record.get("input_wav_sha256")
        # 历史台账绑定的输入 WAV 字节数。
        expected_wav_bytes = record.get("input_wav_bytes")
        # 本轮新生成或经审核后复用的处理类别。
        run_action = record.get("run_action")
        # 按本轮新生成或经审核后复用的处理类别决定是否进入此分支；判断条件为 run_action == "generated"。
        if run_action == "generated":
            # 从当前语音的 generated 动作重新累计新生成数。
            actual_generated += 1
        # 按本轮新生成或经审核后复用的处理类别决定是否进入此分支；判断条件为 run_action == "reused_verified"。
        elif run_action == "reused_verified":
            # 从当前语音的 reused_verified 动作重新累计可信复用数。
            actual_reused += 1
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Missing/invalid generated-vs-reused action for {key}")
        # 数组内容摘要必须格式正确，保存精度必须明确为 float32。
        if not _is_sha256(expected_array_hash) or expected_dtype != np.dtype(np.float32).str:
            # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Malformed content attestation for {key} in {report_path}")
        # 每条特征还必须有原始输入 WAV 的完整内容摘要和整数大小。
        if not _is_sha256(expected_wav_hash) or not isinstance(expected_wav_bytes, int):
            # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Missing input-WAV attestation for {key} in {report_path}")
        # 按BNF 或说话人嵌入的特征类别决定是否进入此分支；判断条件为 feature_kind == "bnf"。
        if feature_kind == "bnf":
            # 台账中记录的数组形状是否满足 BNF 或身份维数约束。
            valid_record_shape = (
                # 台账形状应为 JSON 列表。
                isinstance(expected_shape, list)
                # BNF 形状必须包含通道和时间两个轴。
                and len(expected_shape) == 2
                # BNF 第一个轴必须为 144 个内容通道。
                and expected_shape[0] == PAPER_CONFIG.n_bnf
                # BNF 时间帧数必须是整数。
                and isinstance(expected_shape[1], int)
                # BNF 时间帧数必须大于零。
                and expected_shape[1] > 0
            )
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 台账中记录的数组形状是否满足 BNF 或身份维数约束。
            valid_record_shape = expected_shape == [PAPER_CONFIG.spk_emb_dim]
        # 规范形状与正整数文件大小都有效，才能用台账核对真实文件。
        if not valid_record_shape or not isinstance(expected_bytes, int) or expected_bytes <= 0:
            # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Malformed shape/size attestation for {key} in {report_path}")

        # 当前待检查或读写的文件路径。
        path = feature_dir / key
        # 磁盘特征必须存在且字节数和提取时一致，先排除缺失或截断。
        if not path.is_file() or path.stat().st_size != expected_bytes:
            # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Missing or size-mismatched feature: {path}")
        # 全量内容审核时，进一步加载实际数组或 WAV，不能只信任台账中的摘要。
        if check_content:
            # 当前正在校验的数值或记录。
            value = _load_strict_feature(path, feature_kind)
            # 实际加载数组的形状必须与提取时登记的形状相同。
            if [int(item) for item in value.shape] != expected_shape:
                # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"Shape changed since extraction: {path}")
            # 对类型、形状和数值重新计算摘要，检出提取后被修改的特征。
            if sha256_array(value) != expected_array_hash:
                # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"Content changed since extraction: {path}")

        # 当前语音的音频文件。
        wav_path = Path(rows_by_key[key]["wav22k_path"])
        # 当前输入 WAV 的缓存索引路径。
        wav_cache_key = str(wav_path)
        # 当前输入 WAV 已核对的尺寸及可选内容摘要。
        cached_wav = input_wav_cache.get(wav_cache_key)
        # 按当前输入 WAV 已核对的尺寸及可选内容摘要决定是否进入此分支；判断条件为 cached_wav is None。
        if cached_wav is None:
            # 检查当前语音的音频文件是否缺失；缺失时不能继续信任该输入。
            if not wav_path.is_file():
                # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"Missing input WAV for attested feature: {wav_path}")
            # 当前输入 WAV 已核对的尺寸及可选内容摘要。
            cached_wav = {"bytes": int(wav_path.stat().st_size)}
            # 缓存这份 WAV 的已核对信息，BNF 与身份特征可共用而不重复读文件。
            input_wav_cache[wav_cache_key] = cached_wav
        # 特征所用输入 WAV 的文件大小必须仍与提取时一致。
        if cached_wav.get("bytes") != expected_wav_bytes:
            # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Input WAV size changed since extraction: {wav_path}")
        # 全量内容审核时，进一步加载实际数组或 WAV，不能只信任台账中的摘要。
        if check_content:
            # 本次重新读输入 WAV 算得的内容摘要。
            actual_wav_hash = cached_wav.get("sha256")
            # 按本次重新读输入 WAV 算得的内容摘要决定是否进入此分支；判断条件为 actual_wav_hash is None。
            if actual_wav_hash is None:
                # 本次重新读输入 WAV 算得的内容摘要。
                actual_wav_hash = sha256_file(wav_path)
                # 缓存真实 WAV 内容摘要，另一类特征审核时可直接复用。
                cached_wav["sha256"] = actual_wav_hash
            # 按本次重新读输入 WAV 算得的内容摘要决定是否进入此分支；判断条件为 actual_wav_hash != expected_wav_hash。
            if actual_wav_hash != expected_wav_hash:
                # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"Input WAV content changed since extraction: {wav_path}")

        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征数值内容的指纹；字段顺序必须与读取审核时一致。
        content_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征数值内容的指纹；字段顺序必须与读取审核时一致。
        content_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征数值内容的指纹；字段顺序必须与读取审核时一致。
        content_digest.update(expected_array_hash.encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征数值内容的指纹；字段顺序必须与读取审核时一致。
        content_digest.update(b"\n")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征身份和文件大小的指纹；字段顺序必须与读取审核时一致。
        layout_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征身份和文件大小的指纹；字段顺序必须与读取审核时一致。
        layout_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征身份和文件大小的指纹；字段顺序必须与读取审核时一致。
        layout_digest.update(str(expected_bytes).encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计特征身份和文件大小的指纹；字段顺序必须与读取审核时一致。
        layout_digest.update(b"\n")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 内容的指纹；字段顺序必须与读取审核时一致。
        input_content_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 内容的指纹；字段顺序必须与读取审核时一致。
        input_content_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 内容的指纹；字段顺序必须与读取审核时一致。
        input_content_digest.update(expected_wav_hash.encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 内容的指纹；字段顺序必须与读取审核时一致。
        input_content_digest.update(b"\n")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 文件大小的指纹；字段顺序必须与读取审核时一致。
        input_layout_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 文件大小的指纹；字段顺序必须与读取审核时一致。
        input_layout_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 文件大小的指纹；字段顺序必须与读取审核时一致。
        input_layout_digest.update(str(expected_wav_bytes).encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按清单顺序累计输入 WAV 文件大小的指纹；字段顺序必须与读取审核时一致。
        input_layout_digest.update(b"\n")

    # 逐文件动作累计值必须与报告总计完全一致。
    if actual_generated != generated or actual_reused != reused:
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 指出来源报告总体计数与逐文件真实处理动作矛盾。
            f"Per-file generated/reused actions disagree with counters in {report_path}"
        )
    # 由逐文件新生成/复用动作推回的合法来源模式。
    expected_source_mode = (
        # 有新生成也有可信复用时，将整套来源模式登记为混合可信来源。
        "generated_and_verified_reused"
        # 按本轮实际重新提取成功的文件数决定是否进入此分支；判断条件为 generated and reused。
        if generated and reused
        # 有新生成且没有复用时，合法来源模式为全部新生成。
        else "all_generated"
        # 按本轮实际重新提取成功的文件数决定是否进入此分支；判断条件为 generated。
        if generated
        # 没有新生成时，全部文件必须来自已验证复用。
        else "all_reused_verified"
    )
    # 报告来源模式必须与逐文件真实动作推导出的模式一致。
    if report.get("source_mode") != expected_source_mode:
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Source mode disagrees with per-file actions in {report_path}")
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # 兼容旧字段 ok 也必须等于已核对的新生成数。
        report.get("ok") != generated
        # 兼容旧字段 skip 必须等于已核对的可信复用数。
        or report.get("skip") != reused
        # 兼容旧字段 fail 也必须为零，不能与新计数互相矛盾。
        or report.get("fail") != 0
        # 在没有失败的完整运行中，最初缺失待生成数应等于本轮新生成数。
        or report.get("missing_rows") != generated
    ):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Legacy counters disagree with audited counters in {report_path}")

    # 报告中整套特征与输入 WAV 的汇总摘要。
    collection = report.get("feature_collection")
    # 从逐文件台账重算的特征数值集合摘要。
    calculated_content_sha = content_digest.hexdigest()
    # 从逐文件台账重算的特征布局集合摘要。
    calculated_layout_sha = layout_digest.hexdigest()
    # 从逐文件台账重算的输入 WAV 内容集合摘要。
    calculated_input_content_sha = input_content_digest.hexdigest()
    # 从逐文件台账重算的输入 WAV 布局集合摘要。
    calculated_input_layout_sha = input_layout_digest.hexdigest()
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # 报告必须有结构化的全量集合摘要。
        not isinstance(collection, dict)
        # 集合摘要覆盖条数必须与实际清单一致。
        or collection.get("count") != len(rows)
        # 整套特征数值摘要必须能由逐文件台账重新得到。
        or collection.get("content_sha256") != calculated_content_sha
        # 整套特征身份和大小摘要必须与逐文件台账一致。
        or collection.get("layout_sha256") != calculated_layout_sha
        # 输入 WAV 内容集合摘要必须与逐文件来源记录一致。
        or collection.get("input_content_sha256") != calculated_input_content_sha
        # 输入 WAV 布局集合摘要必须与逐文件来源记录一致。
        or collection.get("input_layout_sha256") != calculated_input_layout_sha
    ):
        # 把现有特征逐项对照其音频、提取器和发布记录未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Feature collection summary is inconsistent: {report_path}")

    # 过滤无关运行时间后用于检查点稳定比对的数据来源证据。
    attestation = {
        # 记录来源报告的数据结构版本号。
        "schema_version": FEATURE_REPORT_SCHEMA_VERSION,
        # 记录BNF 或说话人嵌入的特征类别。
        "feature_kind": feature_kind,
        # 记录所用清单的文件指纹。
        "manifest_sha256": expected_manifest_sha,
        # 记录清单中语音唯一身份及其顺序的指纹。
        "manifest_keys_sha256": expected_keys_sha,
        # 记录该份数据证据覆盖的清单条目数。
        "row_count": len(rows),
        # 记录按固定清单顺序汇总的特征数值指纹。
        "content_sha256": calculated_content_sha,
        # 记录语音身份和文件大小的集合指纹。
        "layout_sha256": calculated_layout_sha,
        # 记录输入 WAV 内容集合指纹。
        "input_content_sha256": calculated_input_content_sha,
        # 记录输入 WAV 大小集合指纹。
        "input_layout_sha256": calculated_input_layout_sha,
        # 记录本轮固定提取器源码、配置和权重的身份记录。
        "output_provenance": provenance,
    }
    # 返回把现有特征逐项对照其音频、提取器和发布记录得到的完整字段集合，各字段用途在下面逐项说明。
    return {
        # 以特征类别作为前缀，返回包含模型来源与数据集合的完整证据摘要。
        f"{feature_kind}.attestation_sha256": _canonical_json_sha256(attestation),
        # 返回当前特征实际绑定的全量语音清单指纹。
        f"{feature_kind}.manifest_sha256": expected_manifest_sha,
        # 返回当前特征数值集合摘要，供后续检查点锁定。
        f"{feature_kind}.content_sha256": calculated_content_sha,
        # 返回当前特征身份和文件大小集合摘要。
        f"{feature_kind}.layout_sha256": calculated_layout_sha,
        # 返回该特征所用输入 WAV 的内容集合摘要。
        f"{feature_kind}.input_content_sha256": calculated_input_content_sha,
        # 返回该特征所用输入 WAV 的布局集合摘要。
        f"{feature_kind}.input_layout_sha256": calculated_input_layout_sha,
        # 返回该特征固定预训练提取器的权重指纹。
        f"{feature_kind}.checkpoint_sha256": expected_checkpoint,
        # 返回该特征固定网络及声学配置的指纹。
        f"{feature_kind}.config_sha256": expected_config,
    }


# 核对 mel 均值与标准差来自同一完整训练集发布批次；下面的参数分别限定输入数据和处理约束。
def audit_mel_stats_publication(
    # 本次实验的项目根目录。
    root: Path,
    # 允许模型学习的训练语音记录。
    train_rows: List[Dict[str, str]],
    # 从这里起的参数必须写明名称传入，减少路径或审核开关位置混淆。
    *,
    # 默认扫描全部身份与文件尺寸；显式 full 时进一步重读每个数组和输入 WAV 计算内容指纹。
    check_content: bool = False,
# 返回核对 mel 均值与标准差来自同一完整训练集发布批次整理的 Dict[str, str] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, str]:
    # 核对 mel 均值与标准差来自同一完整训练集发布批次。
    """Validate the atomically published mean/std generation and compatibility aliases."""

    # mel 标准化统计量保存目录。
    stats_dir = root / "data" / "stats"
    # 当前统计量发布是否尚未完成的事务标记路径。
    marker = stats_dir / ".mel_stats_publish_transaction.json"
    # 仅在当前统计量发布是否尚未完成的事务标记路径已存在时进入复用或校验分支。
    if marker.exists():
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 指出统计量发布尚未完整提交，要求先用第 04 步恢复。
            f"Pending mel-statistics publication exists: {marker}; rerun step 04 to recover it."
        )
    # 指向当前完整统计量版本的提交清单路径。
    current_path = stats_dir / "mel_stats_current.json"
    # 在受控文件句柄中完成核对 mel 均值与标准差来自同一完整训练集发布批次，退出此段后自动关闭文件。
    with current_path.open("r", encoding="utf-8") as handle:
        # 当前正式发布的 mel 统计量版本清单。
        current = json.load(handle)
    # 这一次成套发布的均值和标准差版本标识。
    generation_id = current.get("generation_id")
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # 当前统计量提交清单必须采用认识的成套发布格式。
        current.get("schema_version") != MEL_STATS_PUBLICATION_SCHEMA_VERSION
        # 统计量版本编号必须是文字。
        or not isinstance(generation_id, str)
        # 版本编号不能留空，否则无法定位不可变来源。
        or not generation_id
        # 禁止版本编号包含路径跳转，确保读取仍局限于统计量历史目录。
        or any(token in generation_id for token in ("/", "\\", ".."))
    ):
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Invalid current mel-statistics manifest: {current_path}")

    # 某一不可变统计量版本的存放目录。
    generation_dir = stats_dir / ".mel_stats_generations" / generation_id
    # 一个不可变统计量版本的内容和指纹清单。
    generation_manifest = generation_dir / "manifest.json"
    # 当前提交清单必须是不可变版本清单的原样副本，不能只更换 mean 或 std。
    if sha256_file(generation_manifest) != sha256_file(current_path):
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Current mel-statistics manifest differs from its immutable generation.")

    # 版本清单中 mean/std 的逐文件尺寸、类型、形状和摘要。
    files = current.get("files")
    # 要求读取对象符合核对 mel 均值与标准差来自同一完整训练集发布批次预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(files, dict):
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Missing file table in {current_path}")
    # 已验证的 mean/std 文件摘要映射。
    statistic_hashes: Dict[str, str] = {}
    # 分别核对均值和标准差，其中标准差额外要求逐元素为正。
    for filename, require_positive in (("mel_mean.npy", False), ("mel_std.npy", True)):
        # 训练与测试清单所在目录。
        metadata = files.get(filename)
        # 要求读取对象符合核对 mel 均值与标准差来自同一完整训练集发布批次预期的数据结构，否则不能继续信任其中字段。
        if not isinstance(metadata, dict):
            # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Missing {filename} metadata in {current_path}")
        # 当前不可变版本清单要求的文件内容摘要。
        expected_hash = metadata.get("sha256")
        # 历史台账或版本清单记录的文件字节数。
        expected_bytes = metadata.get("bytes")
        # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
        if (
            # 版本清单必须声明单精度统计量。
            metadata.get("dtype") != "float32"
            # 版本清单必须声明每个 mel 频带一个数，共 80 维。
            or metadata.get("shape") != [PAPER_CONFIG.n_mels]
            # 文件内容摘要必须是合法 SHA-256。
            or not _is_sha256(expected_hash)
            # 记录的文件大小必须为整数。
            or not isinstance(expected_bytes, int)
        ):
            # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Invalid {filename} metadata in {current_path}")
        # 同时检查正式固定路径与不可变历史路径，两份都必须属于同一版本。
        for path in (stats_dir / filename, generation_dir / filename):
            # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
            if (
                # 统计量路径必须是实际文件，缺失或目录都不能接受。
                not path.is_file()
                # 文件字节数必须与版本清单相同。
                or path.stat().st_size != expected_bytes
                # 文件内容必须与版本清单绑定摘要相同。
                or sha256_file(path) != expected_hash
            ):
                # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"Published mel statistic is missing/corrupt: {path}")
            # 当前正在校验的数值或记录。
            value = np.load(path, allow_pickle=False)
            # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
            if (
                # 实际加载的均值或标准差必须为 80 维向量。
                value.shape != (PAPER_CONFIG.n_mels,)
                # 实际保存精度必须为 float32。
                or value.dtype != np.float32
                # 实际数组不得含 NaN 或无穷值。
                or not np.isfinite(value).all()
                # 若当前为标准差文件，每个频带的数值必须大于零。
                or (require_positive and not np.all(value > 0))
            ):
                # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"Invalid published mel statistic: {path}")
        # 登记验证通过的统计量摘要，供后续报告和检查点身份核对。
        statistic_hashes[filename] = str(expected_hash)

    # 不可变版本清单保存的报告大小和摘要。
    report_metadata = current.get("report")
    # 对外固定路径上的 mel 统计报告。
    report_alias = root / "exp" / "logs" / "mel_stats_report.json"
    # 不可变统计量版本目录内的原始报告。
    generation_report = generation_dir / "report.json"
    # 要求读取对象符合核对 mel 均值与标准差来自同一完整训练集发布批次预期的数据结构，否则不能继续信任其中字段。
    if not isinstance(report_metadata, dict):
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Missing report metadata in {current_path}")
    # 正式报告与不可变版本内报告也必须互相一致。
    for report_path in (report_alias, generation_report):
        # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
        if (
            # 两处报告路径都必须真实存在。
            not report_path.is_file()
            # 两处报告大小均需与版本清单相同。
            or report_path.stat().st_size != report_metadata.get("bytes")
            # 两处报告内容均需与版本清单记录的摘要相同。
            or sha256_file(report_path) != report_metadata.get("sha256")
        ):
            # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Published mel-statistics report is missing/corrupt: {report_path}")
    # 在受控文件句柄中完成核对 mel 均值与标准差来自同一完整训练集发布批次，退出此段后自动关闭文件。
    with report_alias.open("r", encoding="utf-8") as handle:
        # 本次处理的数据来源和完成情况报告。
        report = json.load(handle)
    # 决定标准化统计来源的训练清单文件。
    train_manifest = root / "data" / "metadata" / "train_manifest.csv"
    # 当前训练清单文件的内容指纹。
    train_manifest_sha = sha256_file(train_manifest)
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # 统计报告必须声明相同的成套发布格式版本。
        report.get("publication_schema_version") != MEL_STATS_PUBLICATION_SCHEMA_VERSION
        # 报告与当前 mean/std 必须登记同一个版本编号。
        or report.get("generation_id") != generation_id
        # 均值和标准差必须基于当前训练清单，不可复用另一次划分的统计。
        or report.get("manifest_sha256") != train_manifest_sha
        # 统计报告必须覆盖当前全部训练语音。
        or report.get("rows_used") != len(train_rows)
        # 训练统计不能遗漏任何清单中的 mel。
        or report.get("missing_count") != 0
        # 训练统计不能跳过无法读取或数值异常的 mel。
        or report.get("bad_count") != 0
        # 报告内均值摘要必须与本次实际核对的 mean 文件一致。
        or report.get("mean_sha256") != statistic_hashes["mel_mean.npy"]
        # 报告内标准差摘要必须与本次实际核对的 std 文件一致。
        or report.get("std_sha256") != statistic_hashes["mel_std.npy"]
    ):
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Mel-statistics report does not match the active training manifest/generation.")

    # 统计报告绑定的训练 mel 数值集合摘要。
    expected_mel_content_sha = report.get("mel_collection_content_sha256")
    # 统计报告绑定的训练 mel 文件布局摘要。
    expected_mel_layout_sha = report.get("mel_collection_layout_sha256")
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # 统计来源集合需覆盖全部训练语音数。
        report.get("mel_collection_count") != len(train_rows)
        # 训练 mel 数值集合摘要必须完整且格式有效。
        or not _is_sha256(expected_mel_content_sha)
        # 训练 mel 布局集合摘要必须完整且格式有效。
        or not _is_sha256(expected_mel_layout_sha)
    ):
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Mel-statistics report lacks a complete training-mel collection digest.")
    # 按训练清单顺序累计的 mel 数值内容指纹。
    mel_content_digest = hashlib.sha256()
    # 按训练清单顺序累计的 mel 身份和文件大小指纹。
    mel_layout_digest = hashlib.sha256()
    # 逐项遍历允许模型学习的训练语音记录，保持每条记录都经过相同检查。
    for row in train_rows:
        # 当前语音的唯一身份。
        key = feature_key(row)
        # 当前训练语音的 mel NPY 路径。
        mel_path = root / "data" / "mel" / key
        # 检查当前训练语音的 mel NPY 路径是否缺失；缺失时不能继续信任该输入。
        if not mel_path.is_file():
            # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Missing training mel target: {mel_path}")
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 身份和文件大小指纹；字段顺序必须与读取审核时一致。
        mel_layout_digest.update(key.encode("utf-8"))
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 身份和文件大小指纹；字段顺序必须与读取审核时一致。
        mel_layout_digest.update(b"\0")
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 身份和文件大小指纹；字段顺序必须与读取审核时一致。
        mel_layout_digest.update(str(int(mel_path.stat().st_size)).encode("ascii"))
        # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 身份和文件大小指纹；字段顺序必须与读取审核时一致。
        mel_layout_digest.update(b"\n")
        # 全量内容审核时，进一步加载实际数组或 WAV，不能只信任台账中的摘要。
        if check_content:
            # 80 维自然对数 mel 声谱图。
            mel = np.load(mel_path, allow_pickle=False)
            # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
            if (
                # 训练 mel 必须包含频带和时间两轴。
                mel.ndim != 2
                # 训练 mel 第一轴必须为 80 个频带。
                or mel.shape[0] != PAPER_CONFIG.n_mels
                # 训练 mel 时间轴必须非空。
                or mel.shape[1] <= 0
                # 训练 mel 必须按标准单精度保存。
                or mel.dtype != np.float32
                # 训练 mel 不得含有污染全局统计的非有限数。
                or not np.isfinite(mel).all()
            ):
                # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"Invalid canonical training mel: {mel_path}")
            # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 数值内容指纹；字段顺序必须与读取审核时一致。
            mel_content_digest.update(key.encode("utf-8"))
            # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 数值内容指纹；字段顺序必须与读取审核时一致。
            mel_content_digest.update(b"\0")
            # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 数值内容指纹；字段顺序必须与读取审核时一致。
            mel_content_digest.update(sha256_array(mel).encode("ascii"))
            # 将本行指定的身份、分隔符或数据内容追加到按训练清单顺序累计的 mel 数值内容指纹；字段顺序必须与读取审核时一致。
            mel_content_digest.update(b"\n")
    # 本次扫描训练 mel 文件大小重算的布局摘要。
    calculated_mel_layout_sha = mel_layout_digest.hexdigest()
    # 重新扫描出的训练 mel 文件布局必须与生成统计量时一致。
    if calculated_mel_layout_sha != expected_mel_layout_sha:
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Training mel layout differs from the step-04 attestation.")
    # full 模式还要核对所有训练 mel 数值内容，避免文件大小没变但数据已改。
    if check_content and mel_content_digest.hexdigest() != expected_mel_content_sha:
        # 核对 mel 均值与标准差来自同一完整训练集发布批次未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Training mel content differs from the step-04 attestation.")

    # 用于严格续训比较的统计量语义身份，不包含生成时间等无关字段。
    semantic_current = {
        # 记录来源报告的数据结构版本号。
        "schema_version": MEL_STATS_PUBLICATION_SCHEMA_VERSION,
        # 记录标准化统计实际对应的训练清单内容指纹。
        "train_manifest_sha256": train_manifest_sha,
        # 记录同一版本逐频带均值 NPY 的文件指纹。
        "mean_sha256": statistic_hashes["mel_mean.npy"],
        # 记录同一版本逐频带标准差 NPY 的文件指纹。
        "std_sha256": statistic_hashes["mel_std.npy"],
        # 记录本次统计实际使用的 mel 数值集合指纹。
        "mel_collection_content_sha256": expected_mel_content_sha,
        # 记录本次统计实际使用的 mel 文件布局指纹。
        "mel_collection_layout_sha256": expected_mel_layout_sha,
        # 记录参与广播或校验的目标张量形状。
        "shape": [PAPER_CONFIG.n_mels],
        # 记录张量的数值类型。
        "dtype": "float32",
    }
    # 返回核对 mel 均值与标准差来自同一完整训练集发布批次得到的完整字段集合，各字段用途在下面逐项说明。
    return {
        # The current manifest is fully validated above, but equality for resume is
        # semantic: generation UUID/timestamps/report paths do not change tensors.
        # 返回与数值有关的统计量身份摘要，不把创建时间等无关字段算作数据变化。
        "mel_stats_current.semantic_sha256": _canonical_json_sha256(semantic_current),
        # 返回 mean/std 实际对应的训练清单摘要。
        "mel_stats.train_manifest_sha256": train_manifest_sha,
        # 返回验证通过的均值文件摘要。
        "mel_stats.mean_sha256": statistic_hashes["mel_mean.npy"],
        # 返回验证通过的标准差文件摘要。
        "mel_stats.std_sha256": statistic_hashes["mel_std.npy"],
        # 返回标准化统计所用训练 mel 的数值集合摘要。
        "mel_collection.content_sha256": str(expected_mel_content_sha),
        # 返回训练 mel 的身份和文件大小集合摘要。
        "mel_collection.layout_sha256": str(expected_mel_layout_sha),
    }


# 核对采样率、STFT、频带和波形截尾约定；下面的参数分别限定输入数据和处理约束。
def audit_mel_extraction_recipe(
    # 本次实验的项目根目录。
    root: Path,
    # 训练及测试实际使用的全部记录。
    all_rows: List[Dict[str, str]],
    # 只用于评估的未见说话人语音记录。
    test_rows: List[Dict[str, str]],
# 返回核对采样率、STFT、频带和波形截尾约定整理的 Dict[str, str] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, str]:
    # 核对采样率、STFT、频带和波形截尾约定。
    """Bind the fixed mel recipe and the held-out mel collection used for endpoint loss."""

    # 数据处理审计报告的保存位置。
    report_path = root / "exp" / "logs" / "mel_extraction_report_22050.json"
    # 在受控文件句柄中完成核对采样率、STFT、频带和波形截尾约定，退出此段后自动关闭文件。
    with report_path.open("r", encoding="utf-8") as handle:
        # 本次处理的数据来源和完成情况报告。
        report = json.load(handle)
    # 本次复现固定的 mel 采样率、分析窗、频带及截尾规则。
    expected_recipe = {
        # 记录波形采样率。
        "sr": PAPER_CONFIG.sample_rate,
        # 记录每个时窗的 FFT 点数。
        "n_fft": PAPER_CONFIG.n_fft,
        # 记录相邻 mel 窗口的样本间距。
        "hop_size": PAPER_CONFIG.hop_size,
        # 记录每次频谱分析的波形窗口长度。
        "win_size": PAPER_CONFIG.win_size,
        # 记录mel 频带数，本实验为 80。
        "n_mels": PAPER_CONFIG.n_mels,
        # 记录mel 滤波器最低频率。
        "fmin": 0.0,
        # 记录mel 滤波器最高频率，本实验兼容选择为 8000 Hz。
        "fmax": 8000.0,
        # 锁定按 hop 整数倍截去波形尾部的现有规则；该细节属于实现补充设定。
        "truncate_waveform_to_hop_multiple": True,
    }
    # 逐个比较采样率、FFT、窗长、帧移、频带和截尾规则，任何不一致都拒绝。
    if any(report.get(key) != value for key, value in expected_recipe.items()):
        # 核对采样率、STFT、频带和波形截尾约定未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Mel extraction recipe mismatch in {report_path}")
    # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
    if (
        # mel 提取报告计划总数必须对应当前全量清单。
        report.get("total_rows") != len(all_rows)
        # 当前正式 mel 来源要求整套由一次明确新提取覆盖。
        or report.get("ok") != len(all_rows)
        # 此版本 mel 报告没有逐文件复用来源台账，因此正式审核不接受跳过已有文件。
        or report.get("skip") != 0
        # 兼容旧字段 fail 也必须为零，不能与新计数互相矛盾。
        or report.get("fail") != 0
    ):
        # 核对采样率、STFT、频带和波形截尾约定未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 提示当前 mel 来源报告只接受完整新生成证据，不能用无逐文件来源的旧复用报告。
            "Mel report must attest a fresh all-generated collection; rerun step 03 with --force."
        )
    # 本次实际使用的清单文件。
    manifest_path = root / "data" / "metadata" / "all_manifest.csv"
    # 绑定声学处理参数和数据清单的提取规则证据。
    recipe_attestation = {
        # 记录固定 mel 提取参数表，包含论文设定及已披露的实现选择。
        "recipe": expected_recipe,
        # 记录所用清单的文件指纹。
        "manifest_sha256": sha256_file(manifest_path),
        # 记录清单中语音唯一身份及其顺序的指纹。
        "manifest_keys_sha256": manifest_keys_sha256(all_rows),
        # 记录该份数据证据覆盖的清单条目数。
        "row_count": len(all_rows),
    }
    # 逐条累计未见测试 mel 数值的摘要对象。
    test_content = hashlib.sha256()
    # 逐条累计未见测试 mel 文件尺寸的摘要对象。
    test_layout = hashlib.sha256()
    # 逐项遍历只用于评估的未见说话人语音记录，保持每条记录都经过相同检查。
    for row in test_rows:
        # 当前语音的唯一身份。
        key = feature_key(row)
        # 当前待检查或读写的文件路径。
        path = root / "data" / "mel" / key
        # 当前正在校验的数值或记录。
        value = np.load(path, allow_pickle=False)
        # 以下条件逐项核对来源和完整性；任一失败都进入报错分支。
        if (
            # 测试 mel 必须是频带×时间的二维数组。
            value.ndim != 2
            # 测试 mel 第一轴必须是 80 个频带。
            or value.shape[0] != PAPER_CONFIG.n_mels
            # 测试 mel 时间轴至少要有一帧。
            or value.shape[1] <= 0
            # 实际保存精度必须为 float32。
            or value.dtype != np.float32
            # 实际数组不得含 NaN 或无穷值。
            or not np.isfinite(value).all()
        ):
            # 核对采样率、STFT、频带和波形截尾约定未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Invalid held-out mel: {path}")
        # 把测试语音唯一身份和分隔符加入数值集合摘要。
        test_content.update(key.encode("utf-8") + b"\0")
        # 再加入该测试 mel 的规范数值摘要，逐语音换行分隔。
        test_content.update(sha256_array(value).encode("ascii") + b"\n")
        # 把同一测试语音身份加入文件布局摘要。
        test_layout.update(key.encode("utf-8") + b"\0")
        # 加入该测试 mel 的真实文件大小，锁定布局证据。
        test_layout.update(str(path.stat().st_size).encode("ascii") + b"\n")
    # 返回核对采样率、STFT、频带和波形截尾约定得到的完整字段集合，各字段用途在下面逐项说明。
    return {
        # 返回 mel 参数与所用清单共同构成的稳定身份摘要。
        "mel_recipe.semantic_sha256": _canonical_json_sha256(recipe_attestation),
        # 返回固定测试 mel 集合的真实数值摘要。
        "test_mel.content_sha256": test_content.hexdigest(),
        # 返回固定测试 mel 集合的真实布局摘要。
        "test_mel.layout_sha256": test_layout.hexdigest(),
    }


# 读取样本 CSV 并校验必需字段及重复身份；下面的参数分别限定输入数据和处理约束。
def read_manifest(path: Path) -> List[Dict[str, str]]:
    # DictReader 把 CSV 每一行变成字符串字典；list(...) 一次读入便于反复统计。
    if not path.is_file():
        # 读取样本 CSV 并校验必需字段及重复身份未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(path)
    # 在受控文件句柄中完成读取样本 CSV 并校验必需字段及重复身份，退出此段后自动关闭文件。
    with path.open("r", encoding="utf-8", newline="") as handle:
        # 本次选定的语音清单。
        rows = list(csv.DictReader(handle))
    # 按本次选定的语音清单决定是否进入此分支；判断条件为 not rows。
    if not rows:
        # 读取样本 CSV 并校验必需字段及重复身份未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Empty manifest: {path}")
    # 返回本次选定的语音清单，供调用方继续处理。
    return rows


# 核对特征数组维数、长度与有效数值；下面的参数分别限定输入数据和处理约束。
def check_array(path: Path, feature_dim: int, vector: bool = False) -> None:
    # vector=False 用于 mel/BNF 二维矩阵，只要求其中一轴是 80/144；
    # vector=True 用于 speaker embedding，squeeze 后必须严格是一维 [256]。
    value = np.load(path, allow_pickle=False)
    # 按当前特征是否应为单个说话人向量决定是否进入此分支；判断条件为 vector。
    if vector:
        # 当前正在校验的数值或记录。
        value = np.asarray(value).squeeze()
        # 当前数组形状是否符合该特征类别的标志。
        valid_shape = value.shape == (feature_dim,)
    # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
    else:
        # 当前数组形状是否符合该特征类别的标志。
        valid_shape = value.ndim == 2 and feature_dim in value.shape
    # np.isfinite 同时排除 NaN、+Inf、-Inf；这些数进入网络会让 loss 变 NaN。
    if not valid_shape or not np.isfinite(value).all():
        # 核对特征数组维数、长度与有效数值未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Invalid feature {path}: shape={value.shape}")


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main() -> None:
    # action="store_true" 表示：命令出现 --full 时 args.full=True，不出现则 False。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--project-root", type=Path, default=SERVER_PROJECT_ROOT)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 要求重新加载全部特征与输入 WAV，逐个重算内容摘要。
        "--full",
        # 该文件本轮为新生成还是经过验证后复用。
        action="store_true",
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="Load and validate every feature array; default checks one utterance per speaker.",
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 要求磁盘划分报告与给定 seen/unseen 协议严格一致。
        "--expected-protocol",
        # 启动检查可以显式锁定 seen/unseen 协议，防止只凭目录名误判。
        choices=(PROTOCOL_TEXT_UNSEEN, PROTOCOL_TEXT_SEEN),
        # 未显式指定时保留为空，由本函数随后确定默认路径或处理范围。
        default=None,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help=(
            # 帮助文字要求磁盘协议记录与指定名称严格一致。
            "Require split_protocol.json to describe this exact active protocol. "
            # 帮助文字说明训练包装脚本会显式设置协议，防止 seen/unseen 混用。
            "Training wrappers set this to prevent a text-seen/unseen mix-up."
        ),
    )
    # 终端传入并经解析的运行设置。
    args = parser.parse_args()
    # 本次实验的项目根目录。
    root = args.project_root.resolve()
    # 项目中保存音频、特征和清单的数据目录。
    data = root / "data"

    # 论文实验协议只有 train/test；all 只是后续特征提取使用的二者合集。
    metadata = data / "metadata"
    # 只允许参与参数学习的训练语音记录。
    train = read_manifest(metadata / "train_manifest.csv")
    # 预先固定、与训练身份分离的测试语音记录。
    test = read_manifest(metadata / "test_manifest.csv")
    # all_manifest 本身不是第三种数据划分；它必须只是 train 与 test 的无重复并集。
    all_rows = read_manifest(metadata / "all_manifest.csv")
    # Counter 的键是 speaker，值是该人在对应 split 中有多少条 utterance。
    counts = {
        # 记录只允许参与参数学习的训练语音记录。
        "train": Counter(row["speaker"] for row in train),
        # 记录预先固定、与训练身份分离的测试语音记录。
        "test": Counter(row["speaker"] for row in test),
    }
    # 不在第一个问题处立刻停止，而是把多个错误集中收集后一次打印，减少反复运行。
    errors: List[str] = []

    # split_protocol.json 是划分器写出的“实验身份证”。必须先知道它是哪一种协议，
    # 后面才能判断 001--010 出现在 train 到底是错误还是有意设计。
    protocol_path = metadata / "split_protocol.json"
    # 当前 seen/unseen 文本划分规则。
    protocol: Dict[str, object] = {}
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在受控文件句柄中完成串联本文件的数据检查、模型处理和结果保存流程，退出此段后自动关闭文件。
        with protocol_path.open("r", encoding="utf-8") as handle:
            # 当前 seen/unseen 文本划分规则。
            protocol = json.load(handle)
    # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as error:
        # 记录协议文件缺失或无法解析的具体原因。
        errors.append(f"cannot read split protocol {protocol_path}: {error}")

    # 同时描述说话人和文本是否交叠的协议全称。
    protocol_name = str(protocol.get("protocol_cli_name", "")).strip()
    # 按同时描述说话人和文本是否交叠的协议全称决定是否进入此分支；判断条件为 protocol_name not in {PROTOCOL_TEXT_UNSEEN, PROTOCOL_TEXT_SEEN}。
    if protocol_name not in {PROTOCOL_TEXT_UNSEEN, PROTOCOL_TEXT_SEEN}:
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误说明协议短名称缺失或不属于已实现的两种协议。
            "split_protocol.json has invalid protocol_cli_name; "
            # 错误列出实际协议名，并说明只支持两种已实现协议。
            f"expected text-unseen or text-seen, got {protocol_name!r}"
        )
    # 按本次允许使用的 seen/unseen 数据划分协议决定是否进入此分支；判断条件为 args.expected_protocol is not None and protocol_name != args.expected_protocol。
    if args.expected_protocol is not None and protocol_name != args.expected_protocol:
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误说明本次请求与磁盘真实数据协议不同。
            "active split protocol mismatch: "
            # 错误比较请求协议与活动清单实际协议。
            f"requested {args.expected_protocol!r}, got {protocol_name!r}"
        )
    # 训练集是否完全排除公共测试文本的标志。
    strict_text_unseen = protocol_name == PROTOCOL_TEXT_UNSEEN

    # 划分器把最终数量写进审计报告；验证器再反向核对，防止报告与 CSV 不一致。
    try:
        # 划分协议声明的训练语音总数。
        expected_train_utterances = int(protocol["train_utterances"])
        # 划分协议声明的测试语音总数。
        expected_test_utterances = int(protocol["test_utterances"])
    # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except (KeyError, TypeError, ValueError) as error:
        # 记录协议缺少合法训练/测试总数，后续不猜测正确条数。
        errors.append(f"split protocol lacks valid utterance counts: {error}")
        # 划分协议声明的训练语音总数。
        expected_train_utterances = -1
        # 划分协议声明的测试语音总数。
        expected_test_utterances = -1
    # 最终训练清单应覆盖论文要求的 100 位说话人。
    if len(counts["train"]) != PAPER_CONFIG.expected_train_speakers:
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误记录训练清单实际说话人数。
            f"train has {len(counts['train'])} speakers, "
            # 错误补充固定要求为 100 位训练说话人。
            f"expected {PAPER_CONFIG.expected_train_speakers}"
        )
    # 旧版脚本留下的 val_manifest 会造成“是不是还有验证集”的歧义；要求先重跑 02 清掉它。
    if (metadata / "val_manifest.csv").exists():
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 提示仍有旧验证集清单，需要重新发布仅 train/test 的划分。
            "stale val_manifest.csv exists; rerun preprocess/02_build_fastvoicegrad_split.py"
        )
    # 最终测试清单应覆盖十位未见说话人。
    if len(counts["test"]) != PAPER_CONFIG.expected_test_speakers:
        # 把实际测试人数与要求十人的差异写入失败清单。
        errors.append(f"test has {len(counts['test'])} speakers, expected 10")
    # set 的 & 是集合交集。unseen-to-unseen 测试要求这个交集必须为空。
    overlap = set(counts["train"]) & set(counts["test"])
    # 按训练和测试说话人身份的交集，应为空决定是否进入此分支；判断条件为 overlap。
    if overlap:
        # 列出同时进入训练和测试的说话人，便于修复身份泄漏。
        errors.append(f"seen/unseen speaker overlap: {sorted(overlap)}")
    # 这是 dictionary comprehension（字典推导式）：只收集数量不等于 10 的人。
    bad_test = {
        # 记录语句数量异常的测试说话人与实际条数。
        speaker: count
        # 逐位检查测试说话人的语音数量。
        for speaker, count in counts["test"].items()
        # 按已经纳入当前统计或审核的文件数量决定是否进入此分支；判断条件为 count != PAPER_CONFIG.test_utterances_per_speaker。
        if count != PAPER_CONFIG.test_utterances_per_speaker
    }
    # 按测试语句条数不等于十的说话人及实际计数决定是否进入此分支；判断条件为 bad_test。
    if bad_test:
        # 列出测试句数异常的说话人与实际数量。
        errors.append(f"test utterance count is not 10 per speaker: {bad_test}")

    # 本次可审计重建固定两条轴；只隔离说话人仍不足以形成 text-unseen，因为
    # VCTK 的公共提示句 001--010 会跨说话人重复。
    actual_test_speakers = set(counts["test"])
    # 按测试清单中实际出现的说话人集合决定是否进入此分支；判断条件为 actual_test_speakers != EXPECTED_TEST_SPEAKERS。
    if actual_test_speakers != EXPECTED_TEST_SPEAKERS:
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误说明实际测试人集合偏离本次固定 ID。
            "unexpected test speaker set: "
            # 同时列出实际和固定测试 ID，便于逐人找出差异。
            f"got={sorted(actual_test_speakers)}, expected={sorted(EXPECTED_TEST_SPEAKERS)}"
        )
    # 固定十位说话人乘十个公共句号的完整测试网格。
    expected_test_grid = {
        # 建立固定测试人和公共文本句号对应的唯一语音身份。
        (speaker, f"{speaker}_{slot}")
        # 覆盖本次固定的十位测试说话人。
        for speaker in EXPECTED_TEST_SPEAKERS
        # 覆盖固定的 001—010 十个公共文本编号。
        for slot in EXPECTED_TEST_SENTENCE_IDS
    }
    # 测试清单实际包含的说话人和句号组合。
    actual_test_grid = {(row["speaker"], row["utt_id"]) for row in test}
    # 按测试清单实际包含的说话人和句号组合决定是否进入此分支；判断条件为 actual_test_grid != expected_test_grid。
    if actual_test_grid != expected_test_grid:
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误说明测试记录没有完整覆盖固定十人×001—010 网格。
            "test manifest is not the exact 10 speakers x fixed 001-010 sentence grid"
        )
    # 变量名沿用旧版：这里收集 train 中所有 001--010。
    # 在 text-unseen 里它们是“泄漏”，在 text-seen 里它们是“必须保留的重叠样本”。
    leaked_sentence_ids = sorted(
        # 下面展开串联本文件的数据检查、模型处理和结果保存流程所需的这组条目；各字段随后逐一说明。
        {
            # 保留在训练中发现的公共测试语句完整编号，供泄漏报错定位。
            row["utt_id"]
            # 遍历最终训练清单，独立检查文本去留和重复情况。
            for row in train
            # 选出训练中仍含 001—010 公共测试文本的条目。
            if sentence_id(row["utt_id"]) in EXPECTED_TEST_SENTENCE_IDS
        }
    )
    # 按训练集是否完全排除公共测试文本的标志决定是否进入此分支；判断条件为 strict_text_unseen and leaked_sentence_ids。
    if strict_text_unseen and leaked_sentence_ids:
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误说明公共测试文本句号在 text-unseen 的训练集合中出现。
            "held-out sentence IDs 001-010 leaked into train; examples="
            # 展示前十个泄漏进训练的公共测试语音编号。
            f"{leaked_sentence_ids[:10]}"
        )
    # 按同时描述说话人和文本是否交叠的协议全称决定是否进入此分支；判断条件为 protocol_name == PROTOCOL_TEXT_SEEN and not leaked_sentence_ids。
    if protocol_name == PROTOCOL_TEXT_SEEN and not leaked_sentence_ids:
        # text-seen 协议若没有保留公共测试句，就登记协议实施失败。
        errors.append("text-seen protocol did not retain any 001-010 rows in train")

    # 实际训练总数必须与划分报告声明一致。
    if len(train) != expected_train_utterances:
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误列出训练清单实际语音数。
            f"train has {len(train)} utterances, split report expects "
            # 补充划分报告登记的预期训练语音数。
            f"{expected_train_utterances}"
        )
    # 实际测试总数必须与划分报告声明一致。
    if len(test) != expected_test_utterances:
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误列出测试清单实际语音数。
            f"test has {len(test)} utterances, split report expects "
            # 补充划分报告登记的预期测试语音数。
            f"{expected_test_utterances}"
        )

    # 按公共句号分组的跨说话人测试转写集合。
    test_texts_by_slot = {}
    # 逐项遍历预先固定、与训练身份分离的测试语音记录，保持每条记录都经过相同检查。
    for row in test:
        # 语音编号中的三位公共文本句号。
        slot = sentence_id(row["utt_id"])
        # 官方转写文字，教师训练不以此作为模型输入。
        text = normalise_text(row.get("text", ""))
        # 按官方转写文字，教师训练不以此作为模型输入决定是否进入此分支；判断条件为 not text。
        if not text:
            # 指出哪条测试语音缺少 CER 所需转写。
            errors.append(f"empty test transcript: {row['speaker']}/{row['utt_id']}")
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue
        # 按句号收集规范化转写，检查跨十位说话人是否真正同句。
        test_texts_by_slot.setdefault(slot, set()).add(text)
    # 必须覆盖十个固定句号，并逐句检查其跨说话人文本是否唯一。
    if set(test_texts_by_slot) != EXPECTED_TEST_SENTENCE_IDS or any(
        # 同一句号若对应多个不同文本，测试网格的配对前提就不成立。
        len(texts) != 1 for texts in test_texts_by_slot.values()
    ):
        # 记录公共句号的跨说话人转写不一致或编号不完整。
        errors.append("each test sentence slot must have one identical text across all 10 speakers")
    # 去重后的测试原文集合，实际应为十句。
    unique_test_texts = set().union(*test_texts_by_slot.values()) if test_texts_by_slot else set()
    # 十个句号还必须对应十句不同原文，避免重复文本冒充十句。
    if len(unique_test_texts) != 10:
        # 报告实际不同测试文本数，不能只用编号数量代替文本覆盖。
        errors.append(f"test must contain 10 distinct sentence texts, got {len(unique_test_texts)}")
    # 训练集中有转写的语音规范化文字集合。
    known_train_texts = {
        # 统一训练转写大小写和空白，仅用于内容交集检查。
        normalise_text(row.get("text", ""))
        # 遍历最终训练清单，独立检查文本去留和重复情况。
        for row in train
        # 只将确实有转写的训练语音加入文字集合，p315 缺文本另由句号规则控制。
        if row.get("text", "").strip()
    }
    # 训练和测试原文的交集，用于独立验证 seen/unseen。
    text_overlap = known_train_texts & unique_test_texts
    # 按训练集是否完全排除公共测试文本的标志决定是否进入此分支；判断条件为 strict_text_unseen and text_overlap。
    if strict_text_unseen and text_overlap:
        # 将 text-unseen 中泄漏到训练的完整公共文本列出。
        errors.append(f"held-out test text appears in train: {sorted(text_overlap)}")
    # 按同时描述说话人和文本是否交叠的协议全称决定是否进入此分支；判断条件为 protocol_name == PROTOCOL_TEXT_SEEN and text_overlap != unique_test_texts。
    if protocol_name == PROTOCOL_TEXT_SEEN and text_overlap != unique_test_texts:
        # text-seen 中测试句没有在训练文字出现的缺失项。
        missing_overlap = sorted(unique_test_texts - text_overlap)
        # 将此处发现的不一致及实际值加入错误清单，最终一起报告而不静默通过。
        errors.append(
            # 错误说明 text-seen 要求每句测试原文都能在其他训练人语音中找到。
            "text-seen protocol requires every test text to occur in train; "
            # 列出 text-seen 中仍未在训练出现的具体测试原文。
            f"missing={missing_overlap}"
        )

    # 固定测试人编号的独立文本清单。
    speakers_file = metadata / "test_speakers.txt"
    # 固定测试文本编号的独立清单。
    sentence_ids_file = metadata / "test_sentence_ids.txt"
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 独立文本文件声明的测试说话人集合。
        recorded_speakers = {
            # 逐行读取独立测试人清单，忽略空行与两端空白。
            line.strip() for line in speakers_file.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        # 按独立文本文件声明的测试说话人集合决定是否进入此分支；判断条件为 recorded_speakers != EXPECTED_TEST_SPEAKERS。
        if recorded_speakers != EXPECTED_TEST_SPEAKERS:
            # 独立测试人清单和固定 ID 不一致时登记错误。
            errors.append(f"{speakers_file} does not match the fixed test speakers")
    # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as error:
        # 记录独立测试人清单无法读取或核对的原因。
        errors.append(f"cannot validate {speakers_file}: {error}")
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 独立文本文件声明的测试句号集合。
        recorded_sentence_ids = {
            # 去掉测试句号两端空白。
            line.strip()
            # 逐行读取独立保存的十个测试文本编号。
            for line in sentence_ids_file.read_text(encoding="utf-8").splitlines()
            # 忽略空白行，不将其当成额外文本编号。
            if line.strip()
        }
        # 按独立文本文件声明的测试句号集合决定是否进入此分支；判断条件为 recorded_sentence_ids != EXPECTED_TEST_SENTENCE_IDS。
        if recorded_sentence_ids != EXPECTED_TEST_SENTENCE_IDS:
            # 独立句号清单偏离 001—010 时登记错误。
            errors.append(f"{sentence_ids_file} does not match fixed sentence IDs 001-010")
    # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as error:
        # 记录测试句号清单无法读取或核对的原因。
        errors.append(f"cannot validate {sentence_ids_file}: {error}")

    # 每条语音由 (speaker, utt_id) 唯一标识。集合适合检查并集/交集，列表长度适合查重复。
    train_ids = [(row["speaker"], row["utt_id"]) for row in train]
    # 测试清单中每条语音的唯一身份列表。
    test_ids = [(row["speaker"], row["utt_id"]) for row in test]
    # 总清单中每条语音的唯一身份列表。
    all_ids = [(row["speaker"], row["utt_id"]) for row in all_rows]
    # 训练语音身份不能重复，避免同一个样本被重复加权。
    if len(set(train_ids)) != len(train_ids):
        # 指出训练清单存在重复语音身份。
        errors.append("train_manifest.csv contains duplicate speaker/utt_id pairs")
    # 测试语音身份不能重复，避免扩大测试数量表象。
    if len(set(test_ids)) != len(test_ids):
        # 指出测试清单存在重复语音身份。
        errors.append("test_manifest.csv contains duplicate speaker/utt_id pairs")
    # 全量清单也必须每条语音只有一行。
    if len(set(all_ids)) != len(all_ids):
        # 指出全量清单存在重复语音身份。
        errors.append("all_manifest.csv contains duplicate speaker/utt_id pairs")
    # 同一条录音不能同时存在于训练和测试。
    if set(train_ids) & set(test_ids):
        # 同一录音跨训练测试出现时记录直接数据泄漏。
        errors.append("the same utterance appears in both train and test manifests")
    # 训练与测试语音身份集合的准确并集。
    expected_all_ids = set(train_ids) | set(test_ids)
    # 全量清单必须准确等于训练与测试并集，不能多混入其他语音。
    if set(all_ids) != expected_all_ids:
        # 指出全量特征清单与训练、测试实际并集不同。
        errors.append("all_manifest.csv is not exactly the union of train and test")

    # split 列也要与文件职责一致，防止文件名正确但某些行内部仍标成旧的 val。
    if any(row.get("split") != "train" for row in train):
        # 记录训练清单中错误的 split 标签。
        errors.append("train_manifest.csv contains a row whose split is not 'train'")
    # 测试清单每行 split 标签都必须写 test。
    if any(row.get("split") != "test" for row in test):
        # 记录测试清单中错误的 split 标签。
        errors.append("test_manifest.csv contains a row whose split is not 'test'")
    # 总清单只能包含 train/test 两类，不接受额外验证集或未知集合。
    if any(row.get("split") not in {"train", "test"} for row in all_rows):
        # 记录总清单混入未知集合或额外验证集的问题。
        errors.append("all_manifest.csv contains a split other than train/test")

    # mean/std 必须来自一次完整原子发布；pending marker、旧兼容文件和 current
    # manifest 任何一处不一致都 fail closed。
    audit_fingerprints: Dict[str, str] = {}
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 汇集本次已通过审核的数据指纹，便于检查点或预检报告保存。
        audit_fingerprints.update(
            # 审核统计量成套发布与训练 mel 来源；full 时重读全部训练 mel 内容。
            audit_mel_stats_publication(root, train, check_content=args.full)
        )
        # 再核对 mel 声学参数及固定测试 mel 内容身份。
        audit_fingerprints.update(audit_mel_extraction_recipe(root, all_rows, test))
    # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as error:
        # 保留成套统计量或 mel 参数来源审核失败的具体原因。
        errors.append(f"cannot verify atomically published mel statistics: {error}")

    # BNF / speaker 报告既绑定模型权重，也绑定当前 all_manifest 和每条特征内容。
    # 快速模式只扫描完整路径/字节布局；正式 --full 会重算全部 numerical digest。
    feature_provenance_ok: Dict[str, bool] = {}
    # 已核对输入 WAV 信息的缓存，避免跨特征重复读取。
    input_wav_cache: Dict[str, Dict[str, object]] = {}
    # 分别审核内容 BNF 和说话人身份向量的固定提取器来源。
    for feature_kind in ("bnf", "speaker_embedding"):
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # 汇集本次已通过审核的数据指纹，便于检查点或预检报告保存。
            audit_fingerprints.update(
                # 审核每种特征的来源台账、文件布局以及所用模型版本。
                audit_feature_provenance(
                    # 本次实验的项目根目录。
                    root,
                    # 训练及测试实际使用的全部记录。
                    all_rows,
                    # BNF 或说话人嵌入的特征类别。
                    feature_kind,
                    # 由 --full 控制是否重读当前特征及其输入 WAV 的全部内容。
                    check_content=args.full,
                    # 已核对输入 WAV 信息的缓存，避免跨特征重复读取。
                    input_wav_cache=input_wav_cache,
                )
            )
            # 记录该特征已经通过完整来源审核，后续无需重复加载其数组。
            feature_provenance_ok[feature_kind] = True
        # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except Exception as error:
            # 来源审核失败时保留失败标志，后续仍尽量检查数组并汇总具体问题。
            feature_provenance_ok[feature_kind] = False
            # 按特征类别记录来源审核失败，不将后续形状合格误当来源已通过。
            errors.append(f"cannot verify {feature_kind} provenance: {error}")

    # 特征检查直接按已经验证过的 all_manifest 进行，避免同一语音重复读取。
    rows = all_rows
    # 未请求 full 时，额外数组形状检查只抽每位说话人一条；来源布局检查仍覆盖全量。
    if not args.full:
        # 快速模式用 setdefault 为每个 speaker 只保留第一次遇到的一条。
        # key 已存在时 setdefault 不覆盖，所以最终正好“一人一条”。
        selected = {}
        # 逐项遍历本次选定的语音清单，保持每条记录都经过相同检查。
        for row in rows:
            # 每位说话人固定取清单中首条语音，避免同一人重复抽样。
            selected.setdefault(row["speaker"], row)
        # 本次选定的语音清单。
        rows = list(selected.values())
    # 正式 --full 时这里会遍历约 40492 条，每条检查三个 .npy。
    for row in rows:
        # 从当前清单行拆出身份，构造三个特征路径。
        speaker, utterance = row["speaker"], row["utt_id"]
        # 每个 tuple 依次保存：(文件路径，期望特征维数，是否是一维向量)。
        paths = [
            # 检查当前语音的 80 维二维 mel 矩阵。
            (data / "mel" / speaker / f"{utterance}.npy", PAPER_CONFIG.n_mels, False),
        ]
        # 成功的 full provenance audit 已经逐条加载并验证 BNF / speaker，避免
        # 再读一遍数 GB 特征；快速模式仍抽样检查形状。若 audit 失败则保留逐条诊断。
        if not args.full or not feature_provenance_ok.get("bnf", False):
            # 将尚未在完整来源审核中加载过的特征加入形状检查列表。
            paths.append(
                # 检查当前语音的 144 维二维内容 BNF。
                (data / "bnf" / speaker / f"{utterance}.npy", PAPER_CONFIG.n_bnf, False)
            )
        # 抽查模式或身份来源审核失败时，仍单独加载身份向量检查数值结构。
        if not args.full or not feature_provenance_ok.get("speaker_embedding", False):
            # 将尚未在完整来源审核中加载过的特征加入形状检查列表。
            paths.append(
                # 下面展开串联本文件的数据检查、模型处理和结果保存流程所需的这组条目；各字段随后逐一说明。
                (
                    # 定位当前语音的说话人嵌入 NPY。
                    data / "spk_emb" / speaker / f"{utterance}.npy",
                    # 身份向量要求 256 维。
                    PAPER_CONFIG.spk_emb_dim,
                    # 将身份特征按单个向量检查，不能误当通道×时间矩阵。
                    True,
                )
            )
        # 逐项取出特征路径、应有维数及是否为单向量的标志。
        for path, dimension, vector in paths:
            # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
            try:
                # 实际读取当前特征，确认维数和有限数值。
                check_array(path, dimension, vector)
            # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
            except Exception as error:
                # 最多累计 30 个错误，足够定位问题，又不会打印几万行。
                errors.append(str(error))
                # 达到 30 个错误时停止继续刷同类信息，保留足够定位线索并最终非零退出。
                if len(errors) >= 30:
                    # 结束当前扫描循环，保留此前已经收集的结果。
                    break
        # 达到 30 个错误时停止继续刷同类信息，保留足够定位线索并最终非零退出。
        if len(errors) >= 30:
            # 结束当前扫描循环，保留此前已经收集的结果。
            break

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"project_root={root}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"train: {len(train)} utterances, {len(counts['train'])} speakers")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"test:  {len(test)} utterances, {len(counts['test'])} speakers")
    # all 的数量应等于 train+test；它只是特征提取总清单，不是 validation。
    print(f"all:   {len(all_rows)} utterances (train + test; no validation split)")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"feature_check={'all utterances' if args.full else 'one utterance per speaker'}")
    # 按汇总的审核失败说明决定是否进入此分支；判断条件为 errors。
    if errors:
        # stderr 是错误输出流；SystemExit(1) 返回非零退出码，让 shell 的
        # set -e 能阻止 run_train_voicegrad30.sh 继续启动错误训练。
        print("\nSETUP CHECK FAILED:", file=sys.stderr)
        # 逐项遍历汇总的审核失败说明，保持每条记录都经过相同检查。
        for error in errors:
            # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
            print(f"  - {error}", file=sys.stderr)
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise SystemExit(1)
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("SETUP CHECK PASSED: the train/test data are ready for VoiceGrad-30 training.")


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # Python 入口保护：直接运行才体检，import read_manifest 时不会自动执行。
    main()
