# 验证学生训练/测试损失 CSV 能稳定生成完整且原子写入的曲线图。
"""验证学生训练/测试损失 CSV 能稳定生成完整且原子写入的曲线图。"""

# unittest 负责发现两个绘图契约测试并报告失败原因。
import unittest
# Path 用于在临时目录中组合 loss.csv 与 PNG 路径。
from pathlib import Path
# TemporaryDirectory 隔离测试文件并在测试结束后自动清除。
from tempfile import TemporaryDirectory

# 从生产绘图模块导入固定表头、CSV 校验器和两张图的生成函数。
from fastvoicegrad.plot_loss_curves import (
    LOSS_FIELDS as PLOT_LOSS_FIELDS,  # 绘图端认为合法的 CSV 完整列顺序。
    load_loss_rows,  # 严格读取并验证每轮损失记录。
    plot_loss_curves,  # 生成总损失图和四面板分项损失图。
)
# 从训练模块导入损失字段、写 CSV 和合并每轮记录的生产实现。
from fastvoicegrad.train import (
    LOSS_COMPONENTS,  # 九个需要同时记录的损失分量，其中两项属于 Cycle。
    LOSS_FIELDS as TRAIN_LOSS_FIELDS,  # 训练端实际写出的 CSV 列顺序。
    append_loss_csv,  # 按正式格式追加一轮损失。
    make_epoch_loss_record,  # 把训练/测试字典和覆盖数量合成一条记录。
)


# 锁定训练记录与绘图脚本之间的表头、覆盖数量和原子文件契约。
class FastVoiceGradStudentPlotTests(unittest.TestCase):
    # 锁定训练记录与绘图脚本之间的表头、覆盖数量和原子文件契约。
    """锁定训练记录与绘图脚本之间的表头、覆盖数量和原子文件契约。"""

    # 记录生成不依赖测试实例状态，因此声明为静态辅助方法。
    @staticmethod
    # 构造数值各异的一轮完整训练/测试损失，便于发现列映射错误。
    def _record(scale):
        # 构造数值各异的一轮完整训练/测试损失，便于发现列映射错误。
        """构造数值各异的一轮完整训练/测试损失，便于发现列映射错误。"""

        # 第 i 个训练分量设为 (i+1)×scale，使七列不会因数值相同而混淆。
        train = {key: float(index + 1) * scale for index, key in enumerate(LOSS_COMPONENTS)}
        # 测试值在对应训练值上加 0.25，确保图中两条曲线不会完全重合。
        test = {
            # 保持与 train 相同的分量顺序，同时用小偏移区分数据来源。
            key: float(index + 1) * scale + 0.25
            # enumerate 提供稳定序号，LOSS_COMPONENTS 提供生产字段名。
            for index, key in enumerate(LOSS_COMPONENTS)
        }
        # 登记完整测试覆盖 100 条、4 批，与正式 text-unseen 评估契约一致。
        return make_epoch_loss_record(train, test, 100, 4)

    # 合法两轮 CSV 必须生成两张非空 PNG，而且不能残留任何临时文件。
    def test_complete_schema_generates_two_nonempty_atomic_pngs(self):
        # 合法两轮 CSV 必须生成两张非空 PNG，而且不能残留任何临时文件。
        """合法两轮 CSV 必须生成两张非空 PNG，而且不能残留任何临时文件。"""

        # 创建测试完成后自动删除的隔离目录，不接触真实训练输出。
        with TemporaryDirectory() as directory:
            # 绘图端和训练端表头必须逐列相等，否则训练后自动绘图会失败或错列。
            self.assertEqual(PLOT_LOSS_FIELDS, TRAIN_LOSS_FIELDS)
            # 把临时目录字符串转换成 Path，供后续路径拼接。
            output = Path(directory)
            # 正式风格的损失表路径位于实验输出目录根部。
            csv_path = output / "loss.csv"
            # 写第 1 轮，1230 是 batch32、39387 条训练数据 drop_last 后的每轮步数。
            append_loss_csv(csv_path, 1, 1230, self._record(1.0))
            # 写第 2 轮，global_step 累计到 2460，并让损失整体缩小到一半。
            append_loss_csv(csv_path, 2, 2460, self._record(0.5))
            # 先独立调用读取校验器，确认 CSV 能被绘图端按正确类型解析。
            rows = load_loss_rows(csv_path)
            # 解析结果必须按 [1,2] 连续排列，没有漏轮、重复或排序改变。
            self.assertEqual([row["epoch"] for row in rows], [1, 2])
            # 生成总损失 PNG 和分项 PNG，并取得两个正式输出路径。
            paths = plot_loss_curves(csv_path, output)
            # 绘图函数的公开契约恰好返回两张图。
            self.assertEqual(len(paths), 2)
            # 对两张图逐一验证文件存在且确实包含足够的 PNG 内容。
            for path in paths:
                # 正式输出路径必须是已落盘的普通文件。
                self.assertTrue(path.is_file())
                # 文件应大于 1000 字节，排除只创建空壳或严重截断的情况。
                self.assertGreater(path.stat().st_size, 1000)
            # 原子保存结束后所有形如 .名字.tmp 的临时文件都必须清理干净。
            self.assertFalse(list(output.glob(".*.tmp")))

    # 测试样本少于固定 100 条时必须拒绝作图，不能展示成完整 text-unseen 曲线。
    def test_plot_rejects_incomplete_test_coverage(self):
        # 测试样本少于固定 100 条时必须拒绝作图，不能展示成完整 text-unseen 曲线。
        """测试样本少于固定 100 条时必须拒绝作图，不能展示成完整 text-unseen 曲线。"""

        # 使用自动清理目录隔离故意损坏的 CSV。
        with TemporaryDirectory() as directory:
            # 转换成 Path 以组合文件名。
            output = Path(directory)
            # 在临时目录内定义损失表路径。
            csv_path = output / "loss.csv"
            # 先生成字段和值都完整的一轮基准记录。
            record = self._record(1.0)
            # 故意把覆盖条数从 100 改成 96，模拟最后 4 条测试语音被遗漏。
            record["test_examples"] = 96
            # 仍按正式写入函数保存，以保证唯一错误就是覆盖数量。
            append_loss_csv(csv_path, 1, 1230, record)
            # 读取器必须抛 RuntimeError，且消息明确包含“不完整留出测试”提示。
            with self.assertRaisesRegex(RuntimeError, "Incomplete held-out"):
                # 如果这句没有抛错，测试就失败，阻止错误曲线进入报告。
                load_loss_rows(csv_path)


# 只有直接执行这个文件时才启动测试；被统一测试发现器导入时不会重复运行。
if __name__ == "__main__":
    # 自动执行本类两个 test_ 方法并返回标准成功/失败状态码。
    unittest.main()
