# 本文件围绕 compare_fastvoicegrad_with_paper 的生产接口进行小规模回归测试。
"""Tests for the FastVoiceGrad student objective-only paper comparison."""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# 读写评估表格和语音清单，核对字段和行顺序。
import csv
# 读写实验身份、指纹和报告的 JSON。
import json
# 提供 NaN 和无穷数反例，检查指标有限性。
import math
# 将故意损坏的测试文件隔离在自动清理目录中。
import tempfile
# 使用标准库测试框架组织断言和失败报告。
import unittest
# 组合临时测试文件和仓库路径。
from pathlib import Path

# 导入本组测试实际调用的 evaluation.compare_fastvoicegrad_with_paper 接口；测试直接约束生产实现。
from evaluation.compare_fastvoicegrad_with_paper import (
    # 论文比较输入不合法时使用的异常类型。
    ComparisonInputError,
    # 从客观摘要生成论文对照表的生产入口。
    write_comparison,
)


# 构造身份完整且四项客观分数可手算的学生评估摘要，数值仅供测试。
def formal_student_summary() -> dict:
    # 按本辅助函数的接口构造最小有效测试字典。
    return {
        # 本次运行完成状态；仅全量8100组可标记正式完成；此处为测试使用的明确值。
        "status": "formal_8100_complete",
        # 实际完成客观评估的转换对数量。
        "evaluated_pairs": 8100,
        # 系统机器标识，学生固定为fastvoicegrad；此处为测试使用的明确值。
        "system_id": "fastvoicegrad",
        # 明确标记student，防止与teacher评估混用；此处为测试使用的明确值。
        "model_role": "student",
        # 学生已完成的训练轮次；此处为测试使用的明确值。
        "student_epoch": 100,
        # 学生已执行的优化器更新次数；此处为测试使用的明确值。
        "student_global_step": 123000,
        # 学生反向采样次数 K；此处为测试使用的明确值。
        "student_steps": 1,
        # 一步学生的加噪起始索引；此处为测试使用的明确值。
        "student_start_step": 950,
        # 学生权重文件内容指纹。
        "student_checkpoint_sha256": (
            # 该固定 SHA-256 绑定历史正式实验身份，不能因换模型而静默替换。
            "eb9ad626160372f1b45b29d6c6d3f058fa8e553cfe48a5ad6a8c007e00d7ca27"
        ),
        # 学生训练契约指纹。
        "student_run_contract_sha256": (
            # 该固定 SHA-256 绑定历史正式实验身份，不能因换模型而静默替换。
            "ea2eac524d970461e928cccf473c0ccf614d9873c6ca8b857f3c1d09ee18d3b1"
        ),
        # 作为来源的教师训练轮次；此处为测试使用的明确值。
        "teacher_epoch": 500,
        # 使用自训修改版HiFi-GAN；此处为测试使用的明确值。
        "vocoder_kind": "custom",
        # 报告中四项客观分数；此处为测试使用的明确值。
        "headline_metrics": {
            # UTMOS 平均预测质量，越高越好；此处为测试使用的明确值。
            "UTMOS_mean_higher_is_better": 4.01,
            # DNSMOS 总体质量均值，越高越好；此处为测试使用的明确值。
            "DNSMOS_mos_ovr_mean_higher_is_better": 3.70,
            # 全语料字符错误率百分比，越低越好；此处为测试使用的明确值。
            "CER_corpus_percent_lower_is_better": 1.70,
            # 目标说话人验证通过率百分比，越高越好；此处为测试使用的明确值。
            "SVA_percent_higher_is_better": 84.0,
        },
        # 与论文条件不同的已知实现说明；此处为测试使用的明确值。
        "known_deviation": "Audited public metric implementations were used.",
    }


# 把同一类故障保护与期望行为组织成一组测试：FastVoiceGradPaperComparisonTests。
class FastVoiceGradPaperComparisonTests(unittest.TestCase):
    # 把测试摘要写进临时目录，供生产比较脚本读取。
    def write_summary(self, directory: Path, payload: dict) -> Path:
        # 测试摘要固定写到 objective_summary.json，供生产论文比较器读取。
        path = directory / "objective_summary.json"
        # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
        path.write_text(json.dumps(payload), encoding="utf-8")
        # 返回已写入文件路径，供生产校验或比较函数读取。
        return path

    # 核对学生比较报告只含四项客观指标，并使用论文 FastVoiceGrad 行。
    def test_writes_only_four_objective_rows_with_fastvoicegrad_values(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as temporary:
            # 本测试独享且结束后清理的目录。
            directory = Path(temporary)
            # 供论文比较脚本读取的客观摘要路径。
            summary_path = self.write_summary(directory, formal_student_summary())
            # 比较脚本生成的 JSON、CSV 和 Markdown 路径。
            outputs = write_comparison(summary_path)

            # 核对 set(outputs), {"json", "csv", "markdown"}：应完全相等。
            self.assertEqual(set(outputs), {"json", "csv", "markdown"})
            # 核对 all(path.is_file() for path in outputs.values())：必须成立。
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            # 读回生产比较器刚生成的 JSON，对照表格字段和差值。
            report = json.loads(outputs["json"].read_text(encoding="utf-8"))
            # 核对 report[“本次运行完成状态；仅全量8100组可标记正式完成”], "fastvoicegrad_paper_comparison_complete"：应完全相等。
            self.assertEqual(report["status"], "fastvoicegrad_paper_comparison_complete")
            # 核对 report[“是否只报告客观评估指标”]：必须成立。
            self.assertTrue(report["objective_metrics_only"])
            # 核对 report[“是否与论文所有评估条件严格一致”]：必须不成立。
            self.assertFalse(report["strictly_same_conditions"])
            # 核对 report[“是否实际进行了统计显著性检验”]：必须不成立。
            self.assertFalse(report["statistical_significance_assessed"])
            # 核对 len(report[“按指标列出的论文与复现比较结果”]), 4：应完全相等。
            self.assertEqual(len(report["comparisons"]), 4)

            # 用指标名索引比较结果，便于逐项核对 UTMOS、DNSMOS、CER 和 SVA。
            rows = {row["metric"]: row for row in report["comparisons"]}
            # 核对 set(rows), {"UTMOS", "DNSMOS", "CER", "SVA"}：应完全相等。
            self.assertEqual(set(rows), {"UTMOS", "DNSMOS", "CER", "SVA"})
            # 核对 rows["UTMOS"][“论文表格中的参考数值”], 3.96：应在浮点容差内相等。
            self.assertAlmostEqual(rows["UTMOS"]["paper_value"], 3.96)
            # 核对 rows["DNSMOS"][“论文表格中的参考数值”], 3.77：应在浮点容差内相等。
            self.assertAlmostEqual(rows["DNSMOS"]["paper_value"], 3.77)
            # 核对 rows["CER"][“论文表格中的参考数值”], 1.89：应在浮点容差内相等。
            self.assertAlmostEqual(rows["CER"]["paper_value"], 1.89)
            # 核对 rows["SVA"][“论文表格中的参考数值”], 83.0：应在浮点容差内相等。
            self.assertAlmostEqual(rows["SVA"]["paper_value"], 83.0)
            # 核对 rows["UTMOS"][“统一为正值表示改善的差值”], 0.05：应在浮点容差内相等。
            self.assertAlmostEqual(rows["UTMOS"]["favorable_delta"], 0.05)
            # 核对 rows["DNSMOS"][“统一为正值表示改善的差值”], -0.07：应在浮点容差内相等。
            self.assertAlmostEqual(rows["DNSMOS"]["favorable_delta"], -0.07)
            # 核对 rows["CER"][“复现值减去论文值的原始差”], -0.19：应在浮点容差内相等。
            self.assertAlmostEqual(rows["CER"]["raw_delta"], -0.19)
            # 核对 rows["CER"][“统一为正值表示改善的差值”], 0.19：应在浮点容差内相等。
            self.assertAlmostEqual(rows["CER"]["favorable_delta"], 0.19)
            # 核对 rows["SVA"][“统一为正值表示改善的差值”], 1.0：应在浮点容差内相等。
            self.assertAlmostEqual(rows["SVA"]["favorable_delta"], 1.0)

            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with outputs["csv"].open("r", encoding="utf-8", newline="") as handle:
                # 从生成 CSV 读回的行，用于核对列顺序。
                csv_rows = list(csv.DictReader(handle))
            # 核对下面完整给出的断言条件。
            self.assertEqual(
                # 提取生成 CSV 中指标名称的实际排列顺序。
                [row["metric"] for row in csv_rows],
                # 学生正式报告仅允许这四项客观指标，且顺序固定。
                ["UTMOS", "DNSMOS", "CER", "SVA"],
            )

            # 生成的可读报告文本，用于检查标签和说明。
            markdown = outputs["markdown"].read_text(encoding="utf-8")
            # 核对 "FastVoiceGrad epoch 100", markdown：应包含目标内容。
            self.assertIn("FastVoiceGrad epoch 100", markdown)
            # 核对 "qMOS", markdown：不应包含目标内容。
            self.assertNotIn("qMOS", markdown)
            # 核对 "sMOS", markdown：不应包含目标内容。
            self.assertNotIn("sMOS", markdown)

    # 逐项污染模型、数据或检查点身份，确认错误结果不能进入论文比较。
    def test_rejects_wrong_formal_identity_fields(self) -> None:
        # 逐项注入的非法身份字段和值。
        invalid_fields = (
            # 把本次运行完成状态；仅全量8100组可标记正式完成改为该行给出的非法值，独立测试拒绝路径。
            ("status", "smoke_test_only"),
            # 实际完成客观评估的转换对数量。
            ("evaluated_pairs", 8099),
            # 把系统机器标识，学生固定为fastvoicegrad改为该行给出的非法值，独立测试拒绝路径。
            ("system_id", "voicegrad30"),
            # 把明确标记student，防止与teacher评估混用改为该行给出的非法值，独立测试拒绝路径。
            ("model_role", "teacher"),
            # 把学生已完成的训练轮次改为该行给出的非法值，独立测试拒绝路径。
            ("student_epoch", 99),
            # 把学生已执行的优化器更新次数改为该行给出的非法值，独立测试拒绝路径。
            ("student_global_step", 122999),
            # 把学生反向采样次数 K改为该行给出的非法值，独立测试拒绝路径。
            ("student_steps", 30),
            # 把一步学生的加噪起始索引改为该行给出的非法值，独立测试拒绝路径。
            ("student_start_step", 949),
            # 学生权重文件内容指纹。
            ("student_checkpoint_sha256", "a" * 64),
            # 学生训练契约指纹。
            ("student_run_contract_sha256", "b" * 64),
            # 把作为来源的教师训练轮次改为该行给出的非法值，独立测试拒绝路径。
            ("teacher_epoch", 499),
            # 把使用自训修改版HiFi-GAN改为该行给出的非法值，独立测试拒绝路径。
            ("vocoder_kind", "public"),
        )
        # 逐个注入非法字段，检查每项保护能独立触发。
        for key, value in invalid_fields:
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temporary:
                # 本测试独享且结束后清理的目录。
                directory = Path(temporary)
                # 每个反例都从完整有效的学生客观摘要重新开始，避免污染累积。
                payload = formal_student_summary()
                # 每次只替换一个身份字段，使错误可归因到该项。
                payload[key] = value
                # 供论文比较脚本读取的客观摘要路径。
                summary_path = self.write_summary(directory, payload)
                # 非法输入必须明确报错，不能生成正式论文比较。
                with self.assertRaises(ComparisonInputError):
                    # 调用生产比较入口，检查错误输入在写结果前被拒绝。
                    write_comparison(summary_path)
                # 核对 (directory / "paper_comparison.json").exists()：必须不成立。
                self.assertFalse((directory / "paper_comparison.json").exists())
                # 核对 (directory / "paper_comparison.csv").exists()：必须不成立。
                self.assertFalse((directory / "paper_comparison.csv").exists())
                # 核对 (directory / "paper_comparison.md").exists()：必须不成立。
                self.assertFalse((directory / "paper_comparison.md").exists())

    # 确认 Python 布尔值不能冒充训练轮次、步数或评估数量。
    def test_rejects_bool_identity_values(self) -> None:
        # 依次检查下面列出的身份字段。
        for key in (
            # 实际完成客观评估的转换对数量。
            "evaluated_pairs",
            # 学生已完成的训练轮次。
            "student_epoch",
            # 学生已执行的优化器更新次数。
            "student_global_step",
            # 学生反向采样次数 K。
            "student_steps",
            # 一步学生的加噪起始索引。
            "student_start_step",
            # 作为来源的教师训练轮次。
            "teacher_epoch",
        ):
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                # 本测试独享且结束后清理的目录。
                directory = Path(temporary)
                # 每个反例都从完整有效的学生客观摘要重新开始，避免污染累积。
                payload = formal_student_summary()
                # 用布尔 True 冒充整数，检验严格类型约束。
                payload[key] = True
                # 供论文比较脚本读取的客观摘要路径。
                summary_path = self.write_summary(directory, payload)
                # 非法输入必须明确报错，不能生成正式论文比较。
                with self.assertRaises(ComparisonInputError):
                    # 调用生产比较入口，检查错误输入在写结果前被拒绝。
                    write_comparison(summary_path)

    # 确认空值、字符串和 NaN/Inf 客观分数均被拒绝。
    def test_rejects_missing_nonnumeric_or_nonfinite_metric(self) -> None:
        # 选 UTMOS 字段作为非法数值反例，其他客观分数保持有效。
        key = "UTMOS_mean_higher_is_better"
        # 依次测试缺失、字符串、NaN 和正负无穷分数。
        for value in (None, "4.01", math.nan, math.inf, -math.inf):
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                # 本测试独享且结束后清理的目录。
                directory = Path(temporary)
                # 每个反例都从完整有效的学生客观摘要重新开始，避免污染累积。
                payload = formal_student_summary()
                # 把当前客观分数替换成缺失或非有限反例。
                payload["headline_metrics"][key] = value
                # 供论文比较脚本读取的客观摘要路径。
                summary_path = self.write_summary(directory, payload)
                # 非法输入必须明确报错，不能生成正式论文比较。
                with self.assertRaises(ComparisonInputError):
                    # 调用生产比较入口，检查错误输入在写结果前被拒绝。
                    write_comparison(summary_path)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
