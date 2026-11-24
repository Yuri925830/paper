# 读取完整教师客观汇总并与论文比较；此历史模块保留论文主观值占位，不组织听评也不计算本项目主观分数。
"""Create an auditable VoiceGrad-30 epoch-500 comparison with the paper.

This module deliberately uses only the Python standard library.  It consumes the
formal report produced by :mod:`evaluation.evaluate_metrics`; diagnostic or
partial metric runs are rejected before any comparison files are written.
"""

# 延后求值类型提示，保持旧 Python 解释器兼容。
from __future__ import annotations

# 使用 argparse 读取命令行参数。
import argparse
# 使用 csv 读写带表头的逐条记录。
import csv
# 使用 hashlib 计算数据与权重文件指纹。
import hashlib
# 使用 json 读写配置及来源记录。
import json
# 使用 math 执行有限数、取整等数值辅助检查。
import math
# 使用 os 执行路径、原子替换和进程级辅助操作。
import os
# 用 Path 统一构造和检查文件路径。
from pathlib import Path
# 标明张量、清单和配置的输入输出类型，不改变运行数据。
from typing import Any, Dict, List, Mapping, Optional, Sequence


# 记录对照来源；论文参考值只用于描述性比较，不参与评价计算。
PAPER_SOURCE = {
    # 记录论文参考分数。
    "paper": (
        # 保留论文标题前半，用于报告来源链接。
        "FastVoiceGrad: One-step Diffusion-Based Voice Conversion with "
        # 标题后半说明对抗条件扩散蒸馏方法，与前一字符串合成完整标题。
        "Adversarial Conditional Diffusion Distillation"
    ),
    # 记录论文发表会议与年份。
    "venue": "Interspeech 2024",
    # 记录所引用的论文表格和模型行。
    "table": "Table 1, VoiceGrad-30, VCTK",
    # 记录论文官方原文页面。
    "url": "https://www.isca-archive.org/interspeech_2024/kaneko24_interspeech.html",
}

# 列出四项客观指标及对应字段、单位和改善方向。
OBJECTIVE_METRICS = (
    {
        # 记录客观指标名称。
        "metric": "UTMOS",
        # 记录客观汇总中对应的指标字段名。
        "headline_key": "UTMOS_mean_higher_is_better",
        # 记录数值使用的单位。
        "unit": "score",
        # 记录数值较好的一侧方向。
        "direction": "higher_is_better",
        # 记录论文表格中已舍入的参考数值，不参与模型评价。
        "paper_value": 3.88,
    },
    {
        # 记录客观指标名称。
        "metric": "DNSMOS",
        # 记录客观汇总中对应的指标字段名。
        "headline_key": "DNSMOS_mos_ovr_mean_higher_is_better",
        # 记录数值使用的单位。
        "unit": "score",
        # 记录数值较好的一侧方向。
        "direction": "higher_is_better",
        # 记录论文表格中已舍入的参考数值，不参与模型评价。
        "paper_value": 3.77,
    },
    {
        # 记录客观指标名称。
        "metric": "CER",
        # 记录客观汇总中对应的指标字段名。
        "headline_key": "CER_corpus_percent_lower_is_better",
        # 记录数值使用的单位。
        "unit": "percent",
        # 记录数值较好的一侧方向。
        "direction": "lower_is_better",
        # 记录论文表格中已舍入的参考数值，不参与模型评价。
        "paper_value": 2.20,
    },
    {
        # 记录客观指标名称。
        "metric": "SVA",
        # 记录客观汇总中对应的指标字段名。
        "headline_key": "SVA_percent_higher_is_better",
        # 记录数值使用的单位。
        "unit": "percent",
        # 记录数值较好的一侧方向。
        "direction": "higher_is_better",
        # 记录论文表格中已舍入的参考数值，不参与模型评价。
        "paper_value": 82.9,
    },
)

# 历史报告只抄录论文 qMOS/sMOS 原值，本项目未开展主观听测，不产生本地分数。
SUBJECTIVE_METRICS = (
    {
        # 记录客观指标名称。
        "metric": "qMOS",
        # 记录数值使用的单位。
        "unit": "score",
        # 记录数值较好的一侧方向。
        "direction": "higher_is_better",
        # 记录论文表格中已舍入的参考数值，不参与模型评价。
        "paper_value": 3.95,
        # 记录论文主观表格公布的 ± 范围，本项目没有重新测量。
        "paper_uncertainty_plus_minus": 0.11,
    },
    {
        # 记录客观指标名称。
        "metric": "sMOS",
        # 记录数值使用的单位。
        "unit": "score",
        # 记录数值较好的一侧方向。
        "direction": "higher_is_better",
        # 记录论文表格中已舍入的参考数值，不参与模型评价。
        "paper_value": 2.42,
        # 记录论文主观表格公布的 ± 范围，本项目没有重新测量。
        "paper_uncertainty_plus_minus": 0.16,
    },
)

# 只接受正式全量完成状态，拒绝小样本诊断结果。
EXPECTED_STATUS = "formal_8100_complete"
# 正式 text-unseen 对照固定覆盖 8,100 个转换配对。
EXPECTED_PAIRS = 8100
# 教师对照使用第 500 轮权重。
EXPECTED_EPOCH = 500


# 表示输入未通过正式结果检查，禁止发布比较表。
class ComparisonInputError(ValueError):
    # 用专用异常区分正式输入不合格，失败后不会写比较结果。
    """Raised when the objective summary is not the required formal result."""


# 读取完整客观汇总路径和比较报告输出目录，不启动推理或修改模型。
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # 设置命令行参数解析器。
    parser = argparse.ArgumentParser(description=__doc__)
    # 定义当前命令行选项及其路径类型、必填约束或默认值。
    parser.add_argument(
        # 用户必须指定已经生成的客观汇总 JSON。
        "--summary",
        # 把命令行路径转换为 Path，统一处理绝对路径。
        type=Path,
        # 此项必须显式提供，避免猜测输入实验。
        required=True,
        # 帮助说明输入必须来自完整客观评价的 objective_summary.json。
        help="Path to the formal objective_summary.json produced by evaluate_metrics.",
    )
    # 定义当前命令行选项及其路径类型、必填约束或默认值。
    parser.add_argument(
        # 可选择独立报告目录；未指定时与输入汇总放在一起。
        "--output-dir",
        # 把命令行路径转换为 Path，统一处理绝对路径。
        type=Path,
        # 没有覆盖值时按下游明确规定的默认目录处理。
        default=None,
        # 帮助说明未指定输出目录时，报告保存到输入汇总旁。
        help="Output directory; defaults to the objective summary's directory.",
    )
    # 返回解析后的命令行参数。
    return parser.parse_args(argv)


# 为引用的结果文件计算字节指纹，保证比较表能追溯来源。
def _sha256(path: Path) -> str:
    # 初始化累计文件摘要，用于锁定比较表实际读取的汇总字节。
    digest = hashlib.sha256()
    # 按二进制读取，文件编码或换行不会被自动转换。
    with path.open("rb") as handle:
        # 每次读取一块文件内容，直到文件末尾，避免一次占满内存。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            # 把当前二进制块加入同一个文件指纹。
            digest.update(chunk)
    # 返回便于写入 JSON 的十六进制文件指纹。
    return digest.hexdigest()


# 严格验证整数身份字段，避免布尔值或字符串冒充轮次。
def _require_exact_int(summary: Mapping[str, Any], key: str, expected: int) -> None:
    # 读取待核对的身份字段，缺项会导致下面检查失败。
    value = summary.get(key)
    # 禁止布尔值冒充整数轮次，同时检查准确数值。
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        # 带上实际值与要求值后报错，中止正式比较。
        raise ComparisonInputError(
            # 错误列出字段名、预期整数和实际值；字符串或布尔值不能冒充轮数/配对数。
            f"{key} must be the integer {expected}; received {value!r}"
        )


# 要求比较用的分数是有限数，不把缺失值当成零。
def _finite_number(value: Any, label: str) -> float:
    # 指标必须是 JSON 数值，布尔值或字符串都不能当分数。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # 拒绝非数值指标，报告具体字段和收到的内容。
        raise ComparisonInputError(f"{label} must be a finite JSON number; received {value!r}")
    # 统一为浮点数，便于差值计算。
    result = float(value)
    # 拒绝 NaN 和无穷大，不能让异常数值进入比较表。
    if not math.isfinite(result):
        # 拒绝 NaN/Inf 分数，不将异常值当成零参与差值计算。
        raise ComparisonInputError(f"{label} must be finite; received {value!r}")
    # 返回本函数已经构造或校验的结果，供调用方继续处理。
    return result


# 只接受满足固定实验身份和完整条目数的客观指标汇总。
def load_formal_summary(path: Path) -> Dict[str, Any]:
    # 加载前先要求完整 8100 对与第 500 epoch 教师身份。
    """Load and strictly validate an epoch-500, 8,100-pair objective report."""

    # 确定待处理文件路径。
    path = path.resolve()
    # 必须能找到用户指定的完整客观汇总。
    if not path.is_file():
        # 缺少客观汇总时直接报出准确路径。
        raise FileNotFoundError(f"Objective summary not found: {path}")
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 读取客观指标汇总。
        summary = json.loads(path.read_text(encoding="utf-8"))
    # 汇总 JSON 语法损坏时中止并指出路径。
    except json.JSONDecodeError as error:
        # JSON 解析失败时保留文件路径和底层原因，禁止继续生成比较表。
        raise ComparisonInputError(f"Invalid objective summary JSON: {path}: {error}") from error
    # 汇总根节点必须为键值对象。
    if not isinstance(summary, dict):
        # 汇总必须为具名字段对象，不能接受列表或裸数值。
        raise ComparisonInputError("Objective summary must be a JSON object")
    # 仅有部分指标或诊断结果不能获得正式完成身份。
    if summary.get("status") != EXPECTED_STATUS:
        # 带上实际值与要求值后报错，中止正式比较。
        raise ComparisonInputError(
            # 完成状态必须准确匹配正式 8100 对完成标记。
            f"status must be {EXPECTED_STATUS!r}; received {summary.get('status')!r}"
        )
    # 核对实际评价了 8,100 个配对。
    _require_exact_int(summary, "evaluated_pairs", EXPECTED_PAIRS)
    # 核对教师为正式第 500 轮。
    _require_exact_int(summary, "teacher_epoch", EXPECTED_EPOCH)
    # 从已有记录取得通用或自训声码器类型。
    vocoder_kind = summary.get("vocoder_kind")
    # 声码器来源必须是 public/custom 字符串，避免空标签或其他模型混入。
    if not isinstance(vocoder_kind, str) or vocoder_kind not in {"public", "custom"}:
        # 带上实际值与要求值后报错，中止正式比较。
        raise ComparisonInputError(
            # 错误先说明可接受的两个真实来源标签。
            "vocoder_kind must be the authoritative string 'public' or 'custom'; "
            # 再输出现场收到的错误值，便于修正调用。
            f"received {vocoder_kind!r}"
        )

    # 取得用于论文比较的四项主指标。
    headline = summary.get("headline_metrics")
    # 四项主指标必须有独立键值结构，不能缺失。
    if not isinstance(headline, dict):
        # 四项主要指标必须保存在字段对象中，缺失不能默认为零。
        raise ComparisonInputError("headline_metrics must be a JSON object")
    # 按固定顺序逐项处理 UTMOS、DNSMOS、CER、SVA。
    for definition in OBJECTIVE_METRICS:
        # 取得该指标在 objective_summary 中的准确字段名。
        key = str(definition["headline_key"])
        # 在计算差值前逐项确认分数存在且有限。
        _finite_number(headline.get(key), f"headline_metrics.{key}")
    # 返回通过模型身份、样本数量及指标有效性检查的汇总。
    return summary


# 对差值做固定精度舍入，减少浮点尾数影响报告显示。
def _stable_delta(value: float) -> float:
    # Avoid exposing binary floating-point tails such as 0.020000000000000018.
    # 舍入到 12 位小数，消除差值中的浮点尾数噪声。
    return round(value, 12)


# 将方向统一后的差值解释成易读文字，不据此宣称统计显著。
def _delta_interpretation(metric: str, favorable_delta: float) -> str:
    # 统一改善方向差值为正，表示本次数字沿该指标较好方向变化。
    if favorable_delta > 0:
        # 正的有利差值表示指标数值向期望方向移动。
        relation = "数值上朝有利方向偏移"
    # 统一改善方向差值为负，表示本次数字较论文参考差。
    elif favorable_delta < 0:
        # 负的有利差值表示指标数值向不利方向移动。
        relation = "数值上朝不利方向偏移"
    # 零差值单独处理，不硬说成改善或退化。
    else:
        # 按当前数字精度相同，只描述这一事实。
        relation = "按报告精度数值相同"
    # 组合指标名、数值方向及解释限制。
    return (
        # 说明与论文的差别只是描述性数值比较。
        f"{metric} 相对论文报告值{relation}；这是描述性差值，"
        # 保留评价资产和实验条件不能证明完全一致的边界。
        "不表示统计显著，也不表示两套实验严格同条件。"
    )


# 把实测与论文客观分数排在同一行，并按各指标方向计算差值。
def _objective_rows(headline: Mapping[str, Any]) -> List[Dict[str, Any]]:
    # 初始化四项客观指标的比较行，后续按固定顺序追加。
    rows: List[Dict[str, Any]] = []
    # 按固定顺序逐项处理 UTMOS、DNSMOS、CER、SVA。
    for definition in OBJECTIVE_METRICS:
        # 设置客观指标名称。
        metric = str(definition["metric"])
        # 取得该指标在 objective_summary 中的准确字段名。
        key = str(definition["headline_key"])
        # 设置数值较好的一侧方向。
        direction = str(definition["direction"])
        # 从实际客观汇总读取这一项实测分数。
        reproduction = _finite_number(headline[key], f"headline_metrics.{key}")
        # 设置论文参考分数。
        paper = float(definition["paper_value"])
        # 原始差值固定采用实测减论文。
        raw_delta = _stable_delta(reproduction - paper)
        # CER 越低越好，因此翻转符号；其他三项保留原始差值。
        favorable_delta = raw_delta if direction == "higher_is_better" else -raw_delta
        # 统一改善差值也做相同舍入，避免负零或尾数干扰。
        favorable_delta = _stable_delta(favorable_delta)
        # 把当前指标的实测、论文、差值和限制说明保存为一行。
        rows.append(
            {
                # 记录客观指标名称。
                "metric": metric,
                # 记录本项目是否实际测量该指标。
                "measurement_status": "automatically_measured",
                # 记录数值使用的单位。
                "unit": definition["unit"],
                # 记录数值较好的一侧方向。
                "direction": direction,
                # 记录本次客观实测值；未测指标明确为空。
                "reproduction_value": reproduction,
                # 记录论文表格中已舍入的参考数值，不参与模型评价。
                "paper_value": paper,
                # 记录论文主观表格公布的 ± 范围，本项目没有重新测量。
                "paper_uncertainty_plus_minus": None,
                # 记录实测减论文的原始差值。
                "raw_delta": raw_delta,
                # 记录统一为正值代表改善的差值。
                "favorable_delta": favorable_delta,
                # 记录该差值方向的文字解释。
                "explanation": _delta_interpretation(metric, favorable_delta),
            }
        )
    # 返回通过本函数检查的记录序列。
    return rows


# 仅保留论文原有主观分数作历史参考；本项目未实施主观评估。
def _subjective_rows() -> List[Dict[str, Any]]:
    # 历史模块为论文主观指标保留空值行，复现值不会被填入。
    rows: List[Dict[str, Any]] = []
    # 仅遍历预登记的论文主观参考数值，不运行任何主观评价。
    for definition in SUBJECTIVE_METRICS:
        # 设置客观指标名称。
        metric = str(definition["metric"])
        # 把当前指标的实测、论文、差值和限制说明保存为一行。
        rows.append(
            {
                # 记录客观指标名称。
                "metric": metric,
                # 记录本项目是否实际测量该指标。
                "measurement_status": "not_automatically_measured",
                # 记录数值使用的单位。
                "unit": definition["unit"],
                # 记录数值较好的一侧方向。
                "direction": definition["direction"],
                # 记录本次客观实测值；未测指标明确为空。
                "reproduction_value": None,
                # 记录论文表格中已舍入的参考数值，不参与模型评价。
                "paper_value": definition["paper_value"],
                # 记录论文主观表格公布的 ± 范围，本项目没有重新测量。
                "paper_uncertainty_plus_minus": definition[
                    # 字段含义：论文主观表格公布的 ± 范围，本项目没有重新测量。
                    "paper_uncertainty_plus_minus"
                ],
                # 记录实测减论文的原始差值。
                "raw_delta": None,
                # 记录统一为正值代表改善的差值。
                "favorable_delta": None,
                # 记录该差值方向的文字解释。
                "explanation": (
                    # 明确本项目没有测 qMOS/sMOS，论文值仅用于来源说明。
                    f"{metric} 是主观听测指标，本自动流程未测量；论文值仅作参考，"
                    # 禁止从客观指标估计或伪造本项目主观分数。
                    "不能由客观指标替代或推断。"
                ),
            }
        )
    # 返回通过本函数检查的记录序列。
    return rows


# 组装论文对照记录，同时保留无法核实同条件的限制。
def build_report(summary: Mapping[str, Any], summary_path: Path) -> Dict[str, Any]:
    # 输入通过完整性和教师身份检查之后，才能建立比较报告。
    """Build the comparison payload after ``load_formal_summary`` validation."""

    # 读取已核验的四项客观指标，后续只计算与论文的差值。
    headline = summary["headline_metrics"]
    # 从已有记录取得与作者未公开实现无法核实一致的条件。
    known_deviation = str(summary.get("known_deviation", "")).strip()
    # 设置通用或自训声码器类型。
    vocoder_kind = str(summary["vocoder_kind"])
    # 按真实来源判断是否为公开 HiFi-GAN，以选择对应的可比性说明。
    uses_public_hifigan = vocoder_kind == "public"

    # 保留作者未公开权重、评价资产和逐条结果所造成的可比性边界。
    limitations = [
        (
            # 公开版分支明确指出使用的是公开 VCTK V1 权重。
            "本次波形使用 public HiFi-GAN（pretrained_hifigan_22k/VCTK_V1），"
            # 说明公开版与论文描述的修改架构存在差异。
            "并非论文描述的 modified HiFi-GAN 架构。"
            # 仅在实际来源为公开版时选用这段说明。
            if uses_public_hifigan
            # 自训版另用下方说明，不沿用公开版权重的身份描述。
            else (
                # 自训版说明本项目采用 V1+MPD+MRD，但没有作者原始声码器权重可核对。
                "本次使用自训练 HiFi-GAN V1+MPD+MRD；论文没有发布可核验的声码器"
                # 保留训练配置和权重身份无法与作者逐项验证的限制。
                "训练 recipe、训练时长或 checkpoint identity，因此不能证明与作者内部权重相同。"
            )
        ),
        (
            # 列明本项目使用的公开客观评价工具。
            "自动评估依赖公开替代/代理的 UTMOS、DNSMOS、ASR/CER 和 ECAPA/SVA 评价模型；"
            # 说明缺少作者全部资产版本与逐样本输出，不能声称完全同条件。
            "论文未公开足够的资产版本、全部配置和逐样本输出，无法证明完全同条件。"
        ),
        (
            # 论文表格已舍入，且没有可配对的逐样本数值。
            "论文表格值已经舍入，且没有可配对的逐样本统计量或置信区间；"
            # 因此这里只描述均值差，不进行显著性推断。
            "本报告只能给出描述性差值，不能进行或声称统计显著性判断。"
        ),
        # 此历史文字仅说明主观指标未测；当前项目继续只做客观评价。
        "qMOS 和 sMOS 需要按论文协议组织真人主观听测，本自动流程未测量。",
    ]
    # 输入汇总已有偏差说明时，比较报告继续保留。
    if known_deviation:
        # 把原始实验的偏差声明继续附在比较报告中。
        limitations.append(f"objective_summary.json 自带偏差说明：{known_deviation}")

    # 返回包含实际来源、四项客观差值和比较限制的结构化报告。
    return {
        # 记录流程完成状态。
        "status": "paper_comparison_complete",
        # 记录正式样本数量和模型身份核对结果。
        "formal_input_validation": {
            # 记录输入客观汇总的完成状态。
            "objective_summary_status": summary["status"],
            # 记录已经完成指标计算的配对数。
            "evaluated_pairs": summary["evaluated_pairs"],
            # 记录教师模型训练轮数。
            "teacher_epoch": summary["teacher_epoch"],
        },
        # 记录比较所引用的客观汇总文件。
        "source_objective_summary": str(summary_path.resolve()),
        # 记录被引用汇总的内容指纹。
        "source_objective_summary_sha256": _sha256(summary_path.resolve()),
        # 记录通用或自训声码器类型。
        "vocoder_kind": vocoder_kind,
        # 记录声码器权重指纹。
        "vocoder_checkpoint_sha256": summary.get(
            # 保留原声码器权重摘要；上游未提供时明确标为 unavailable，不伪造哈希。
            "vocoder_checkpoint_sha256", "unavailable"
        ),
        # 记录声码器配置指纹。
        "vocoder_config_sha256": summary.get(
            # 同样保留配置摘要或明确不可用标记。
            "vocoder_config_sha256", "unavailable"
        ),
        # 记录论文标题、会议、表格和原文链接。
        "paper_source": PAPER_SOURCE,
        # 记录原始差值与统一改善方向差值的定义。
        "delta_semantics": {
            # 记录实测减论文的原始差值。
            "raw_delta": "reproduction_value - paper_value",
            # 记录统一为正值代表改善的差值。
            "favorable_delta": (
                # 有利差值统一规定为正数代表方向较好。
                "positive means a numerically favorable direction; for lower-is-better metrics "
                # 对 CER 这类越低越好的指标，将原始差值符号反转。
                "the raw delta is sign-inverted"
            ),
        },
        # 记录是否能确认与论文作者所有条件完全相同。
        "strictly_same_conditions": False,
        # 记录是否开展统计显著性检验，当前为否。
        "statistical_significance_assessed": False,
        # 记录实测和论文的逐指标对照行。
        "comparisons": _objective_rows(headline) + _subjective_rows(),
        # 记录无法核实与作者完全一致的设置及比较边界。
        "limitations": limitations,
    }


# 先写临时文本，再原子替换正式报告。
def _atomic_write_text(path: Path, text: str) -> None:
    # 临时文件与正式报告在同一目录，保证替换不跨文件系统。
    temporary = path.with_name(path.name + ".tmp")
    # 把完整报告内容按 UTF-8 写入临时文件。
    temporary.write_text(text, encoding="utf-8")
    # 把写完的临时文件原子替换到正式结果路径。
    os.replace(temporary, path)


# 把同一组对照行保存为 CSV，便于教授或组会表格核对。
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    # 固定各列含义和顺序，保证三种格式输出同一组数据。
    fieldnames = [
        # 字段含义：客观指标名称。
        "metric",
        # 字段含义：本项目是否实际测量该指标。
        "measurement_status",
        # 字段含义：数值使用的单位。
        "unit",
        # 字段含义：数值较好的一侧方向。
        "direction",
        # 字段含义：本次客观实测值；未测指标明确为空。
        "reproduction_value",
        # 字段含义：论文表格中已舍入的参考数值，不参与模型评价。
        "paper_value",
        # 字段含义：论文主观表格公布的 ± 范围，本项目没有重新测量。
        "paper_uncertainty_plus_minus",
        # 字段含义：实测减论文的原始差值。
        "raw_delta",
        # 字段含义：统一为正值代表改善的差值。
        "favorable_delta",
        # 字段含义：该差值方向的文字解释。
        "explanation",
    ]
    # 临时文件与正式报告在同一目录，保证替换不跨文件系统。
    temporary = path.with_name(path.name + ".tmp")
    # 创建临时 CSV，换行交由 csv 模块处理。
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        # 按明确列名输出，避免字典顺序改变对照表。
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # 写出固定列名，让后续续算和汇总能核对格式。
        writer.writeheader()
        # 逐行写出已核对的客观对照数据。
        writer.writerows(rows)
    # 把写完的临时文件原子替换到正式结果路径。
    os.replace(temporary, path)


# 统一表格数值显示精度，未测值明确显示为空占位。
def _format_value(value: Any) -> str:
    # 没有实际测量的数据不能显示为零。
    if value is None:
        # 用破折号表示未测，不产生虚构分数。
        return "—"
    # 浮点分数使用固定有效数字显示。
    if isinstance(value, float):
        # 最多保留 12 位有效数字，避免过长的二进制浮点尾数。
        return f"{value:.12g}"
    # 非浮点字段按原含义转成文本。
    return str(value)


# 处理 Markdown 表格中的转义，保持每个值留在自己的单元格。
def _markdown_cell(value: Any) -> str:
    # 转义竖线并把换行改为空格，避免拆裂表格列。
    return _format_value(value).replace("|", "\\|").replace("\n", " ")


# 把已核对报告排成便于人工阅读的中文表格。
def render_markdown(report: Mapping[str, Any]) -> str:
    # 取得已经通过的实验身份与样本数，用于报告说明。
    validation = report["formal_input_validation"]
    # 依次组装报告标题、输入身份、差值定义和表头。
    lines = [
        # 写教师与论文比较的 Markdown 标题。
        "# VoiceGrad-30 教师模型与论文表 1 的比较",
        # 空字符串生成空白行，分隔标题与正文，不是遗漏的测量值。
        "",
        (
            # 正文披露实际教师 epoch，避免比较表隐去训练版本。
            f"输入已通过正式门禁：epoch {validation['teacher_epoch']}，"
            # 同时显示实际配对数量与完成状态。
            f"{validation['evaluated_pairs']:,} 个转换对，状态 "
            # 打印通过校验的原完成状态文字。
            f"`{validation['objective_summary_status']}`。"
        ),
        # 用空行隔开身份说明和差值说明。
        "",
        # 显式解释原始差值与有利差值，避免把 CER 上升误认为改善。
        "原始差值 = 复现值 − 论文值；有利差值为正表示仅在指标方向上数值有利。",
        # 空行使后面的 Markdown 表格正确独立显示。
        "",
        # 定义比较表的列名，包括来源状态与解释。
        "| 指标 | 状态 | 方向 | 复现值 | 论文值 | 原始差值 | 有利差值 | 解释 |",
        # 定义 Markdown 列分隔及数值列右对齐。
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    # 按报告中的指标顺序输出表格行。
    for row in report["comparisons"]:
        # 设置论文表格中已舍入的参考数值，不参与模型评价。
        paper_value = _format_value(row["paper_value"])
        # 读取论文是否给出不确定度；None 表示没有该数值。
        uncertainty = row["paper_uncertainty_plus_minus"]
        # 只有论文确实提供不确定度才显示 ± 项。
        if uncertainty is not None:
            # 把论文原有不确定度附到论文值，不套用到本项目结果。
            paper_value += f" ± {_format_value(uncertainty)}"
        # 追加当前指标对应的 Markdown 表格行。
        lines.append(
            # 开始这一行 Markdown 表格的左边界。
            "| "
            # 用 Markdown 列分隔符连接已转义单元格。
            + " | ".join(
                [
                    # 输出指标名，特殊竖线会转义以免破坏表格。
                    _markdown_cell(row["metric"]),
                    # 输出是否实际测量的状态，主观占位仍显示未测。
                    _markdown_cell(row["measurement_status"]),
                    # 输出指标方向，让读者知道应看高还是低。
                    _markdown_cell(row["direction"]),
                    # 输出本项目实际客观值或明确空值。
                    _markdown_cell(row["reproduction_value"]),
                    # 输出论文值，若提供则同时展示原不确定度。
                    _markdown_cell(paper_value),
                    # 输出复现值减论文值的原始差值。
                    _markdown_cell(row["raw_delta"]),
                    # 输出按指标好坏方向统一后的差值。
                    _markdown_cell(row["favorable_delta"]),
                    # 输出描述性解释及不能推断显著性的提醒。
                    _markdown_cell(row["explanation"]),
                ]
            )
            # 补齐当前表格行右边界。
            + " |"
        )

    # 取得论文标题、会议和原文链接。
    source = report["paper_source"]
    # 在指标表后追加可比性限制和来源说明。
    lines.extend(
        [
            # 空行将指标表与可比性说明分开。
            "",
            # 加入可比性限制的小节标题。
            "## 可比性限制",
            # 标题之后保留空行，保证 Markdown 列表排版正常。
            "",
            # 将每条限制单独列出，不把数值差异解释成已证实同条件优势。
            *[f"- {item}" for item in report["limitations"]],
            # 用空行分开限制列表和论文出处。
            "",
            (
                # 用真实论文标题与官方地址构造来源链接。
                f"论文来源：[{source['paper']}]({source['url']})，"
                # 同时写明发表会议和被比较的表格行。
                f"{source['venue']}，{source['table']}。"
            ),
            # 结尾空行保证后续追加文字不会挤到来源段落。
            "",
        ]
    )
    # 按行拼出完整可读报告。
    return "\n".join(lines)


# 核对正式汇总后输出三种格式的同一份对照报告。
def write_comparison(summary_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    # 验证通过后分别原子写出 JSON、CSV 与 Markdown。
    """Validate a formal summary and atomically write JSON, CSV, and Markdown."""

    # 固定输入汇总绝对路径，便于记录其来源。
    summary_path = summary_path.resolve()
    # 设置客观指标汇总。
    summary = load_formal_summary(summary_path)
    # 输出目录采用显式值，否则与输入汇总放在一起。
    destination = (output_dir or summary_path.parent).resolve()
    # 创建报告输出目录。
    destination.mkdir(parents=True, exist_ok=True)
    # 设置核对结果。
    report = build_report(summary, summary_path)

    # 机器可复查的完整比较记录。
    json_path = destination / "paper_comparison.json"
    # 可在表格软件中打开的逐指标对照表。
    csv_path = destination / "paper_comparison.csv"
    # 便于人工阅读的比较报告。
    markdown_path = destination / "paper_comparison.md"
    # 原子写出完整 JSON 或 Markdown 文本。
    _atomic_write_text(
        # 将完整来源与限制保存到 JSON 文件。
        json_path,
        # 保留中文并缩进输出完整比较记录。
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    # 把相同对照行写为 CSV，不重新计算另一套分数。
    _write_csv(csv_path, report["comparisons"])
    # 发布从同一报告对象渲染的可读版本。
    _atomic_write_text(markdown_path, render_markdown(report))
    # 返回实际生成的三个文件路径，方便调用方定位。
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


# 执行本脚本的输入检查、主要处理和结果记录。
def main(argv: Optional[Sequence[str]] = None) -> None:
    # 读取汇总路径和可选输出目录。
    args = parse_args(argv)
    # 执行正式输入核对并写出比较文件。
    outputs = write_comparison(args.summary, args.output_dir)
    # 打印刚生成的报告文件位置，便于远程找到输出。
    print(
        # 用 JSON 格式组织文件类型与路径映射。
        json.dumps(
            # 把各格式文件的实际路径作为 JSON 输出给调用者。
            {name: str(path) for name, path in outputs.items()},
            # 保留路径和报告中的中文字符。
            ensure_ascii=False,
            # 对输出路径记录做易读缩进。
            indent=2,
        )
    )
    # 三种比较报告都写出后才打印完成标志。
    print("VOICEGRAD30_PAPER_COMPARISON_COMPLETE")


# 仅作为脚本直接运行时执行入口，导入模块时不启动任务。
if __name__ == "__main__":
    # 启动本脚本的完整处理流程。
    main()
