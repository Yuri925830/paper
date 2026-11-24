# 只比较完整 FastVoiceGrad 学生四项客观指标；本项目不组织主观听评。
"""Compare a formal FastVoiceGrad student evaluation with the paper.

This module accepts only the completed 8,100-pair VCTK text-unseen student
summary.  It deliberately reports the four objective metrics only; subjective
MOS values are outside this project's evaluation protocol.
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
        # 论文标题前半用于构造可追溯来源。
        "FastVoiceGrad: One-step Diffusion-Based Voice Conversion with "
        # 论文标题后半与前一字符串连接，保持完整原题。
        "Adversarial Conditional Diffusion Distillation"
    ),
    # 记录论文发表会议与年份。
    "venue": "Interspeech 2024",
    # 记录所引用的论文表格和模型行。
    "table": "Table 1, FastVoiceGrad, VCTK",
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
        "paper_value": 3.96,
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
        "paper_value": 1.89,
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
        "paper_value": 83.0,
    },
)

# 只接受正式全量完成状态，拒绝小样本诊断结果。
EXPECTED_STATUS = "formal_8100_complete"
# 正式 text-unseen 对照固定覆盖 8,100 个转换配对。
EXPECTED_PAIRS = 8100
# 学生比较必须明确声明 FastVoiceGrad 身份。
EXPECTED_SYSTEM_ID = "fastvoicegrad"
# 此历史比较配置要求学生已经训练 100 轮。
EXPECTED_STUDENT_EPOCH = 100
# 历史训练 100 轮对应 123,000 次更新。
EXPECTED_STUDENT_GLOBAL_STEP = 123000
# 核对学生初始化及蒸馏所用教师为第 500 轮。
EXPECTED_TEACHER_EPOCH = 500
# 此学生对照使用固定自训声码器，不混入通用基线。
EXPECTED_VOCODER_KIND = "custom"
# 锁定历史学生权重；新修复训练不能直接沿用这一身份常量。
EXPECTED_STUDENT_CHECKPOINT_SHA256 = (
    # 旧学生 epoch100 的准确权重 SHA-256；新修复学生必须另行登记，不能偷换此值。
    "eb9ad626160372f1b45b29d6c6d3f058fa8e553cfe48a5ad6a8c007e00d7ca27"
)
# 锁定历史训练目标与参数约定，新实验需独立审核。
EXPECTED_STUDENT_RUN_CONTRACT_SHA256 = (
    # 旧学生训练契约摘要，绑定它当时的目标、权重与数据设置。
    "ea2eac524d970461e928cccf473c0ccf614d9873c6ca8b857f3c1d09ee18d3b1"
)


# 表示输入未通过正式结果检查，禁止发布比较表。
class ComparisonInputError(ValueError):
    # 正式学生输入不合格时抛出此异常，禁止写出误标的比较报告。
    """Raised when an input is not the required formal student result."""


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
        # 输入必须是正式学生评价生成的 objective_summary.json。
        help="Formal FastVoiceGrad objective_summary.json.",
    )
    # 定义当前命令行选项及其路径类型、必填约束或默认值。
    parser.add_argument(
        # 可选择独立报告目录；未指定时与输入汇总放在一起。
        "--output-dir",
        # 把命令行路径转换为 Path，统一处理绝对路径。
        type=Path,
        # 没有覆盖值时按下游明确规定的默认目录处理。
        default=None,
        # 未显式指定输出目录时，比较报告与原汇总存放在一起。
        help="Output directory; defaults to the summary directory.",
    )
    # 返回解析后的命令行参数。
    return parser.parse_args(argv)


# 为引用的结果文件计算字节指纹，保证比较表能追溯来源。
def _sha256(path: Path) -> str:
    # 初始化 SHA-256 累计器，用于锁定实际输入汇总。
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
    # 要求类型为真正整数且与固定身份值一致。
    if type(value) is not int or value != expected:
        # 带上实际值与要求值后报错，中止正式比较。
        raise ComparisonInputError(
            # 报出整数身份字段的预期与实际值，拒绝把字符串/布尔值当作训练轮数。
            f"{key} must be the integer {expected}; received {value!r}"
        )


# 严格验证文本身份字段，避免引用其他模型或不完整结果。
def _require_exact_string(
    # summary 是待核验报告，key 指身份字段，expected 是此历史实验允许的准确字符串。
    summary: Mapping[str, Any], key: str, expected: str
# 校验通过不返回数据；任何不一致都直接抛错。
) -> None:
    # 读取待核对的身份字段，缺项会导致下面检查失败。
    value = summary.get(key)
    # 要求类型为文本且内容与预期模型身份完全一致。
    if type(value) is not str or value != expected:
        # 带上实际值与要求值后报错，中止正式比较。
        raise ComparisonInputError(
            # 说明哪个字符串身份字段不符，并同时显示要求和实际值。
            f"{key} must be {expected!r}; received {value!r}"
        )


# 要求比较用的分数是有限数，不把缺失值当成零。
def _finite_number(value: Any, label: str) -> float:
    # 指标必须是 JSON 数值，布尔值或字符串都不能当分数。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # 带上实际值与要求值后报错，中止正式比较。
        raise ComparisonInputError(
            # 拒绝非数值指标，不能把缺失或文本分数当成零。
            f"{label} must be a finite JSON number; received {value!r}"
        )
    # 统一为浮点数，便于差值计算。
    result = float(value)
    # 拒绝 NaN 和无穷大，不能让异常数值进入比较表。
    if not math.isfinite(result):
        # 分数为 NaN 或无穷大时终止比较。
        raise ComparisonInputError(f"{label} must be finite; received {value!r}")
    # 返回本函数已经构造或校验的结果，供调用方继续处理。
    return result


# 只接受满足固定实验身份和完整条目数的客观指标汇总。
def load_formal_summary(path: Path) -> Dict[str, Any]:
    # 要求输入属于已登记的旧 epoch100、123000 步学生，修复学生需独立身份。
    """Load and validate the formal epoch-100 FastVoiceGrad result."""

    # 确定待处理文件路径。
    path = path.resolve()
    # 必须能找到用户指定的完整客观汇总。
    if not path.is_file():
        # 客观汇总不存在时指出准确文件位置。
        raise FileNotFoundError(f"Objective summary not found: {path}")
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 读取客观指标汇总。
        summary = json.loads(path.read_text(encoding="utf-8"))
    # 汇总 JSON 语法损坏时中止并指出路径。
    except json.JSONDecodeError as error:
        # JSON 损坏时保留解析异常原因，不继续输出半可信比较表。
        raise ComparisonInputError(f"Invalid objective summary JSON: {path}: {error}") from error
    # 汇总根节点必须为键值对象。
    if not isinstance(summary, dict):
        # 汇总顶层必须为字段对象。
        raise ComparisonInputError("Objective summary must be a JSON object")

    # 核对全量完成标记。
    _require_exact_string(summary, "status", EXPECTED_STATUS)
    # 核对实际评价了 8,100 个配对。
    _require_exact_int(summary, "evaluated_pairs", EXPECTED_PAIRS)
    # 禁止把其他系统的结果标为 FastVoiceGrad。
    _require_exact_string(summary, "system_id", EXPECTED_SYSTEM_ID)
    # 明确该表比较学生模型。
    _require_exact_string(summary, "model_role", "student")
    # 核对学生训练轮数。
    _require_exact_int(summary, "student_epoch", EXPECTED_STUDENT_EPOCH)
    # 核对累计优化器更新次数。
    _require_exact_int(summary, "student_global_step", EXPECTED_STUDENT_GLOBAL_STEP)
    # 核对单步学生推理。
    _require_exact_int(summary, "student_steps", 1)
    # 核对单步推理的起始噪声步为 950。
    _require_exact_int(summary, "student_start_step", 950)
    # 核对配套教师训练轮数。
    _require_exact_int(summary, "teacher_epoch", EXPECTED_TEACHER_EPOCH)
    # 核对声码器类型与此固定学生实验一致。
    _require_exact_string(summary, "vocoder_kind", EXPECTED_VOCODER_KIND)
    # 严格核对下面指定的权重或训练约定指纹。
    _require_exact_string(
        # 传入客观指标汇总。
        summary,
        # 字段含义：学生权重的字节指纹。
        "student_checkpoint_sha256",
        # 仅接受已登记的历史学生权重字节身份。
        EXPECTED_STUDENT_CHECKPOINT_SHA256,
    )
    # 严格核对下面指定的权重或训练约定指纹。
    _require_exact_string(
        # 传入客观指标汇总。
        summary,
        # 字段含义：学生训练约定的指纹。
        "student_run_contract_sha256",
        # 仅接受已登记的历史训练约定。
        EXPECTED_STUDENT_RUN_CONTRACT_SHA256,
    )

    # 取得用于论文比较的四项主指标。
    headline = summary.get("headline_metrics")
    # 四项主指标必须有独立键值结构，不能缺失。
    if not isinstance(headline, dict):
        # 四项主要指标必须是完整字典，不能缺失。
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
    # 舍入到 12 位小数，消除差值中的浮点尾数噪声。
    return round(value, 12)


# 根据指标改善方向描述差值，不据此宣称统计显著。
def _delta_explanation(metric: str, favorable_delta: float) -> str:
    # 统一改善方向差值为正，表示本次数字沿该指标较好方向变化。
    if favorable_delta > 0:
        # 方向统一后为正，表示指标数值朝较好方向变化。
        relation = "数值方向优于论文报告值"
    # 统一改善方向差值为负，表示本次数字较论文参考差。
    elif favorable_delta < 0:
        # 方向统一后为负，表示指标数值朝较差方向变化。
        relation = "数值方向低于论文报告值"
    # 差值为零单独说明，避免硬判改善。
    else:
        # 仅说明按当前报告精度与论文相同。
        relation = "按当前报告精度与论文值相同"
    # 返回数值方向解释，同时保留评价可比性边界。
    return (
        # 描述当前指标与论文均值的差异，不扩展为未测的机制结论。
        f"{metric} {relation}。该差值用于描述结果；两边的评价资产和未公开实验细节"
        # 资产版本与未公开设置无法全部核对，因此不用于显著性判断。
        "无法逐项确认一致，因此不用于统计显著性判断。"
    )


# 把实测与论文客观分数排在同一行，并按各指标方向计算差值。
def _objective_rows(headline: Mapping[str, Any]) -> List[Dict[str, Any]]:
    # 收集四项客观比较结果，不创建主观指标行。
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
                # 记录数值使用的单位。
                "unit": definition["unit"],
                # 记录数值较好的一侧方向。
                "direction": direction,
                # 记录本次客观实测值；未测指标明确为空。
                "reproduction_value": reproduction,
                # 记录论文表格中已舍入的参考数值，不参与模型评价。
                "paper_value": paper,
                # 记录实测减论文的原始差值。
                "raw_delta": raw_delta,
                # 记录统一为正值代表改善的差值。
                "favorable_delta": favorable_delta,
                # 记录该差值方向的文字解释。
                "explanation": _delta_explanation(metric, favorable_delta),
            }
        )
    # 返回通过本函数检查的记录序列。
    return rows


# 组装论文对照记录，同时保留无法核实同条件的限制。
def build_report(summary: Mapping[str, Any], summary_path: Path) -> Dict[str, Any]:
    # 从已有记录取得与作者未公开实现无法核实一致的条件。
    known_deviation = str(summary.get("known_deviation", "")).strip()
    # 保留作者未公开权重、评价资产和逐条结果所造成的可比性边界。
    limitations = [
        (
            # 记录本项目自训声码器采用的修改架构。
            "本次使用项目自训练的 HiFi-GAN V1+MPD+MRD；论文没有公布作者实际使用的"
            # 说明作者未公开原权重、清单与完整训练配置。
            "声码器 checkpoint、训练清单和完整训练 recipe。"
        ),
        (
            # 列明当前实际固定的四类公开评价模型。
            "本次评价固定了公开 UTMOS、DNSMOS、wav2vec 2.0 ASR 和 ECAPA 模型；"
            # 提醒论文没有提供全部资产哈希、参数和逐条输出。
            "论文没有提供这些评价资产的文件哈希、全部参数和逐条输出。"
        ),
        (
            # 论文只有舍入均值，没有可与本实验配对的 8100 条逐样本分数。
            "论文只给出表格中的舍入结果，没有可配对的 8,100 条逐样本分数；"
            # 比较限定为描述性差值，不据此宣称统计显著。
            "这里报告描述性差值，不进行统计显著性判断。"
        ),
    ]
    # 输入汇总已有偏差说明时，比较报告继续保留。
    if known_deviation:
        # 原样继承实验汇总中的已知偏差，避免比较环节丢失限制。
        limitations.append(f"objective_summary.json 记录的偏差说明：{known_deviation}")

    # 返回包含身份校验、四指标差值和限制说明的结构化结果。
    return {
        # 记录流程完成状态。
        "status": "fastvoicegrad_paper_comparison_complete",
        # 记录正式样本数量和模型身份核对结果。
        "formal_input_validation": {
            # 记录输入客观汇总的完成状态。
            "objective_summary_status": summary["status"],
            # 记录已经完成指标计算的配对数。
            "evaluated_pairs": summary["evaluated_pairs"],
            # 记录教师或学生系统标识。
            "system_id": summary["system_id"],
            # 记录教师或学生角色。
            "model_role": summary["model_role"],
            # 记录学生模型训练轮数。
            "student_epoch": summary["student_epoch"],
            # 记录学生优化器累计更新次数。
            "student_global_step": summary["student_global_step"],
            # 记录学生推理去噪次数。
            "student_steps": summary["student_steps"],
            # 记录学生推理起始噪声步。
            "student_start_step": summary["student_start_step"],
            # 记录学生权重的字节指纹。
            "student_checkpoint_sha256": summary["student_checkpoint_sha256"],
            # 记录学生训练约定的指纹。
            "student_run_contract_sha256": summary[
                # 字段含义：学生训练约定的指纹。
                "student_run_contract_sha256"
            ],
            # 记录教师模型训练轮数。
            "teacher_epoch": summary["teacher_epoch"],
            # 记录通用或自训声码器类型。
            "vocoder_kind": summary["vocoder_kind"],
        },
        # 记录比较所引用的客观汇总文件。
        "source_objective_summary": str(summary_path.resolve()),
        # 记录被引用汇总的内容指纹。
        "source_objective_summary_sha256": _sha256(summary_path.resolve()),
        # 记录论文标题、会议、表格和原文链接。
        "paper_source": PAPER_SOURCE,
        # 记录原始差值与统一改善方向差值的定义。
        "delta_semantics": {
            # 记录实测减论文的原始差值。
            "raw_delta": "reproduction_value - paper_value",
            # 记录统一为正值代表改善的差值。
            "favorable_delta": (
                # 定义有利差值的正负含义，与原始相减值分开记录。
                "positive means a numerically favorable direction; lower-is-better "
                # CER 等越低越好的指标需要反转差值符号。
                "metrics use the sign-inverted raw delta"
            ),
        },
        # 记录声明本报告仅比较客观指标。
        "objective_metrics_only": True,
        # 记录是否能确认与论文作者所有条件完全相同。
        "strictly_same_conditions": False,
        # 记录是否开展统计显著性检验，当前为否。
        "statistical_significance_assessed": False,
        # 记录实测和论文的逐指标对照行。
        "comparisons": _objective_rows(summary["headline_metrics"]),
        # 记录无法核实与作者完全一致的设置及比较边界。
        "limitations": limitations,
    }


# 先写临时文本，再原子发布完整报告。
def _atomic_write_text(path: Path, value: str) -> None:
    # 临时文件与正式报告在同一目录，保证替换不跨文件系统。
    temporary = path.with_name(path.name + ".tmp")
    # 把完整报告内容按 UTF-8 写入临时文件。
    temporary.write_text(value, encoding="utf-8")
    # 把写完的临时文件原子替换到正式结果路径。
    os.replace(temporary, path)


# 把同一组对照行保存为 CSV，便于教授或组会表格核对。
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    # 固定各列含义和顺序，保证三种格式输出同一组数据。
    fieldnames = [
        # 字段含义：客观指标名称。
        "metric",
        # 字段含义：数值使用的单位。
        "unit",
        # 字段含义：数值较好的一侧方向。
        "direction",
        # 字段含义：本次客观实测值；未测指标明确为空。
        "reproduction_value",
        # 字段含义：论文表格中已舍入的参考数值，不参与模型评价。
        "paper_value",
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
        # 写学生四项客观指标比较的 Markdown 标题。
        "# FastVoiceGrad 学生模型与论文表 1 的客观指标比较",
        # 空行分隔标题和实验身份正文。
        "",
        (
            # 正文列出实际学生 epoch，不仅写模型名称。
            f"输入检查通过：FastVoiceGrad epoch {validation['student_epoch']}，"
            # 同时披露初始化教师 epoch 与完整配对数量。
            f"教师 epoch {validation['teacher_epoch']}，{validation['evaluated_pairs']:,} 个转换对，"
            # 注明自训声码器来源，方便与公开声码器对照区分。
            f"声码器为 {validation['vocoder_kind']}。"
        ),
        # 用空行分隔身份信息和差值说明。
        "",
        # 解释原始差值与方向统一后的有利差值。
        "原始差值 = 复现值 - 论文值；有利差值为正表示指标方向较好。",
        # 为 Markdown 表格开始留一个空行。
        "",
        # 定义四指标比较表的列，不包含主观听评项目。
        "| 指标 | 方向 | 复现值 | 论文值 | 原始差值 | 有利差值 | 说明 |",
        # 使用 Markdown 分隔行，并将数值列右对齐。
        "|---|---|---:|---:|---:|---:|---|",
    ]
    # 按报告中的指标顺序输出表格行。
    for row in report["comparisons"]:
        # 追加当前指标对应的 Markdown 表格行。
        lines.append(
            # 写当前表格行的左边界。
            "| "
            # 用 Markdown 列分隔符连接已转义单元格。
            + " | ".join(
                [
                    # 输出指标名称并处理可能破坏表格的特殊字符。
                    _markdown_cell(row["metric"]),
                    # 输出越高越好或越低越好的方向。
                    _markdown_cell(row["direction"]),
                    # 输出本项目真实客观评估值。
                    _markdown_cell(row["reproduction_value"]),
                    # 输出论文相应 FastVoiceGrad 行的数值。
                    _markdown_cell(row["paper_value"]),
                    # 输出复现值减论文值。
                    _markdown_cell(row["raw_delta"]),
                    # 输出统一指标方向后的有利差值。
                    _markdown_cell(row["favorable_delta"]),
                    # 输出差值的描述性解释。
                    _markdown_cell(row["explanation"]),
                ]
            )
            # 写当前表格行的右边界。
            + " |"
        )

    # 取得论文标题、会议和原文链接。
    source = report["paper_source"]
    # 在指标表后追加可比性限制和来源说明。
    lines.extend(
        [
            # 空行把指标表与后续说明分开。
            "",
            # 添加可比性说明小节标题。
            "## 可比性说明",
            # 标题后留空行，以正确显示后面的限制列表。
            "",
            # 将每条限制单独列出，不把数值差异解释成已证实同条件优势。
            *[f"- {item}" for item in report["limitations"]],
            # 空行分隔限制列表与论文来源。
            "",
            (
                # 用论文原题和官方地址生成来源链接。
                f"论文来源：[{source['paper']}]({source['url']})，"
                # 写出会议和表格位置，避免与 VoiceGrad-30 教师行混用。
                f"{source['venue']}，{source['table']}。"
            ),
            # 在文档末尾保留空行。
            "",
        ]
    )
    # 按行拼出完整可读报告。
    return "\n".join(lines)


# 核对正式汇总后一次输出 JSON、CSV 与 Markdown。
def write_comparison(
    # summary_path 指已完成学生汇总，output_dir 可选择独立报告目录。
    summary_path: Path, output_dir: Optional[Path] = None
# 返回三种生成文件的路径映射，方便调用者显示或下载。
) -> Dict[str, Path]:
    # 先验证完整学生身份，再原子写出 JSON、CSV 和 Markdown。
    """Validate the summary and atomically write JSON, CSV, and Markdown."""

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
    # 将生成文件的位置输出到终端。
    print(
        # 路径按 JSON 对象打印，便于远程脚本读取。
        json.dumps(
            # 把各格式文件的实际路径作为 JSON 输出给调用者。
            {name: str(path) for name, path in outputs.items()},
            # 保留路径和报告中的中文字符。
            ensure_ascii=False,
            # 对输出路径记录做易读缩进。
            indent=2,
        )
    )
    # 所有比较文件成功写出后才输出客观比较完成标志。
    print("FASTVOICEGRAD_PAPER_OBJECTIVE_COMPARISON_COMPLETE")


# 仅作为脚本直接运行时执行入口，导入模块时不启动任务。
if __name__ == "__main__":
    # 启动本脚本的完整处理流程。
    main()
