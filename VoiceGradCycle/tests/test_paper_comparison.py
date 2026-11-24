# 本文件围绕 paper_comparison 的生产接口进行小规模回归测试。
"""Tests for the standard-library paper-comparison report."""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# 读写评估表格和语音清单，核对字段和行顺序。
import csv
# 读写实验身份、指纹和报告的 JSON。
import json
# 将故意损坏的测试文件隔离在自动清理目录中。
import tempfile
# 使用标准库测试框架组织断言和失败报告。
import unittest
# 组合临时测试文件和仓库路径。
from pathlib import Path

# 导入本组测试实际调用的 evaluation.compare_with_paper 接口；测试直接约束生产实现。
from evaluation.compare_with_paper import ComparisonInputError, write_comparison


# 构造通用声码器教师评估摘要，检验论文表格的差值和未测项目。
def formal_summary() -> dict:
    # 按本辅助函数的接口构造最小有效测试字典。
    return {
        # 本次运行完成状态；仅全量8100组可标记正式完成；此处为测试使用的明确值。
        "status": "formal_8100_complete",
        # 实际完成客观评估的转换对数量。
        "evaluated_pairs": 8100,
        # 作为来源的教师训练轮次；此处为测试使用的明确值。
        "teacher_epoch": 500,
        # 使用自训修改版HiFi-GAN；此处为测试使用的明确值。
        "vocoder_kind": "public",
        # 报告中四项客观分数；此处为测试使用的明确值。
        "headline_metrics": {
            # UTMOS 平均预测质量，越高越好；此处为测试使用的明确值。
            "UTMOS_mean_higher_is_better": 3.90,
            # DNSMOS 总体质量均值，越高越好；此处为测试使用的明确值。
            "DNSMOS_mos_ovr_mean_higher_is_better": 3.50,
            # 全语料字符错误率百分比，越低越好；此处为测试使用的明确值。
            "CER_corpus_percent_lower_is_better": 1.90,
            # 目标说话人验证通过率百分比，越高越好；此处为测试使用的明确值。
            "SVA_percent_higher_is_better": 80.0,
        },
        # 与论文条件不同的已知实现说明；此处为测试使用的明确值。
        "known_deviation": (
            # 旧教师对照报告明确使用公开 VCTK V1 声码器。
            "Waveforms were synthesized by public pretrained_hifigan_22k/VCTK_V1, "
            # 报告继续说明它与作者未公开修改版声码器存在差异。
            "not the paper authors' unpublished modified HiFi-GAN."
        ),
    }


# 把同一类故障保护与期望行为组织成一组测试：PaperComparisonTests。
class PaperComparisonTests(unittest.TestCase):
    # 把测试摘要写进临时目录，供生产比较脚本读取。
    def write_summary(self, directory: Path, payload: dict) -> Path:
        # 测试摘要固定写到 objective_summary.json，供生产论文比较器读取。
        path = directory / "objective_summary.json"
        # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
        path.write_text(json.dumps(payload), encoding="utf-8")
        # 返回已写入文件路径，供生产校验或比较函数读取。
        return path

    # 核对旧教师表格的四项客观结果及论文主观指标占位；本复现主观分数保持空值。
    def test_writes_objective_and_unmeasured_subjective_rows(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as temporary:
            # 本测试独享且结束后清理的目录。
            directory = Path(temporary)
            # 供论文比较脚本读取的客观摘要路径。
            summary_path = self.write_summary(directory, formal_summary())
            # 比较脚本生成的 JSON、CSV 和 Markdown 路径。
            outputs = write_comparison(summary_path)

            # 核对 set(outputs), {"json", "csv", "markdown"}：应完全相等。
            self.assertEqual(set(outputs), {"json", "csv", "markdown"})
            # 核对 all(path.is_file() for path in outputs.values())：必须成立。
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            # 读回生产比较器刚生成的 JSON，对照表格字段和差值。
            report = json.loads(outputs["json"].read_text(encoding="utf-8"))
            # 核对 report[“本次运行完成状态；仅全量8100组可标记正式完成”], "paper_comparison_complete"：应完全相等。
            self.assertEqual(report["status"], "paper_comparison_complete")
            # 核对 report[“是否与论文所有评估条件严格一致”]：必须不成立。
            self.assertFalse(report["strictly_same_conditions"])
            # 核对 report[“是否实际进行了统计显著性检验”]：必须不成立。
            self.assertFalse(report["statistical_significance_assessed"])
            # 核对 len(report[“按指标列出的论文与复现比较结果”]), 6：应完全相等。
            self.assertEqual(len(report["comparisons"]), 6)

            # 用指标名索引比较结果，便于逐项核对 UTMOS、DNSMOS、CER 和 SVA。
            rows = {row["metric"]: row for row in report["comparisons"]}
            # 核对 rows["UTMOS"][“复现值减去论文值的原始差”], 0.02：应在浮点容差内相等。
            self.assertAlmostEqual(rows["UTMOS"]["raw_delta"], 0.02)
            # 核对 rows["UTMOS"][“统一为正值表示改善的差值”], 0.02：应在浮点容差内相等。
            self.assertAlmostEqual(rows["UTMOS"]["favorable_delta"], 0.02)
            # 核对 rows["DNSMOS"][“统一为正值表示改善的差值”], -0.27：应在浮点容差内相等。
            self.assertAlmostEqual(rows["DNSMOS"]["favorable_delta"], -0.27)
            # 核对 rows["CER"][“复现值减去论文值的原始差”], -0.30：应在浮点容差内相等。
            self.assertAlmostEqual(rows["CER"]["raw_delta"], -0.30)
            # 核对 rows["CER"][“统一为正值表示改善的差值”], 0.30：应在浮点容差内相等。
            self.assertAlmostEqual(rows["CER"]["favorable_delta"], 0.30)
            # 核对 rows["SVA"][“统一为正值表示改善的差值”], -2.9：应在浮点容差内相等。
            self.assertAlmostEqual(rows["SVA"]["favorable_delta"], -2.9)
            # 核对 rows["qMOS"][“论文表格中的参考数值”], 3.95：应完全相等。
            self.assertEqual(rows["qMOS"]["paper_value"], 3.95)
            # 核对 rows["qMOS"][“论文主观均值报告的加减区间”], 0.11：应完全相等。
            self.assertEqual(rows["qMOS"]["paper_uncertainty_plus_minus"], 0.11)
            # 核对 rows["qMOS"][“本次复现测得的指标；未测项目为空”]：应为空。
            self.assertIsNone(rows["qMOS"]["reproduction_value"])
            # 核对 rows["qMOS"][“统一为正值表示改善的差值”]：应为空。
            self.assertIsNone(rows["qMOS"]["favorable_delta"])
            # 核对 rows["sMOS"][“论文表格中的参考数值”], 2.42：应完全相等。
            self.assertEqual(rows["sMOS"]["paper_value"], 2.42)
            # 核对 rows["sMOS"][“论文主观均值报告的加减区间”], 0.16：应完全相等。
            self.assertEqual(rows["sMOS"]["paper_uncertainty_plus_minus"], 0.16)

            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with outputs["csv"].open("r", encoding="utf-8", newline="") as handle:
                # 从生成 CSV 读回的行，用于核对列顺序。
                csv_rows = list(csv.DictReader(handle))
            # 核对 [row["metric"] for row in csv_rows], [：应完全相等。
            self.assertEqual([row["metric"] for row in csv_rows], [
                # 旧教师表格保留四项客观结果及论文两项主观占位；复现主观值保持空白。
                "UTMOS", "DNSMOS", "CER", "SVA", "qMOS", "sMOS"
            ])
            # 核对 csv_rows[4][“本次复现测得的指标；未测项目为空”], ""：应完全相等。
            self.assertEqual(csv_rows[4]["reproduction_value"], "")

            # 生成的可读报告文本，用于检查标签和说明。
            markdown = outputs["markdown"].read_text(encoding="utf-8")
            # 核对 "public HiFi-GAN", markdown：应包含目标内容。
            self.assertIn("public HiFi-GAN", markdown)
            # 核对 "不能进行或声称统计显著性判断", markdown：应包含目标内容。
            self.assertIn("不能进行或声称统计显著性判断", markdown)
            # 核对 "3.95 ± 0.11", markdown：应包含目标内容。
            self.assertIn("3.95 ± 0.11", markdown)

    # 确认试跑、少于 8100 对和错误教师轮次不能生成正式比较报告。
    def test_rejects_nonformal_partial_or_wrong_epoch_input(self) -> None:
        # 逐项注入的非法身份字段和值。
        invalid_fields = (
            # 把本次运行完成状态；仅全量8100组可标记正式完成改为该行给出的非法值，独立测试拒绝路径。
            ("status", "smoke_test_only"),
            # 实际完成客观评估的转换对数量。
            ("evaluated_pairs", 8099),
            # 把作为来源的教师训练轮次改为该行给出的非法值，独立测试拒绝路径。
            ("teacher_epoch", 499),
        )
        # 逐个注入非法字段，检查每项保护能独立触发。
        for key, value in invalid_fields:
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temporary:
                # 本测试独享且结束后清理的目录。
                directory = Path(temporary)
                # 每个反例都重新构造有效教师摘要，再单独改变待检查项。
                payload = formal_summary()
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

    # 确认缺失或非法客观分数会阻止教师论文比较。
    def test_rejects_missing_or_nonfinite_headline_metric(self) -> None:
        # 用缺失、字符串与 NaN 检验教师客观分数保护。
        for value in (None, "3.9", float("nan")):
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                # 本测试独享且结束后清理的目录。
                directory = Path(temporary)
                # 每个反例都重新构造有效教师摘要，再单独改变待检查项。
                payload = formal_summary()
                # 只污染 UTMOS 分数，检查非法数值会阻止整份报告发布。
                payload["headline_metrics"]["UTMOS_mean_higher_is_better"] = value
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
