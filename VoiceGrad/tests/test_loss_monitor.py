# 解析报警文件，核对报警类别、步数和证据窗口。
import json
# 为监控文件建立临时目录，测试结束后自动清理测试数据。
import tempfile
# 组织损失趋势和断点恢复的单元测试。
import unittest
# 定位临时目录中的窗口和报警JSON文件。
from pathlib import Path

# 直接测试正式训练所用的损失监控器。
from fastvoicegrad.loss_monitor import LossMonitor


# 覆盖涨幅阈值、首次报警、非有限损失及续训窗口回滚。
class LossMonitorTest(unittest.TestCase):
    # 区分持续上涨与单批噪声波动。
    def test_sustained_not_single_noisy_batch(self):
        # 交替升降的数列不能触发持续上涨报警。
        self.assertFalse(LossMonitor.sustained_rise([1, 2, 1, 2]))
        # 虽然连续变大，累计仅上涨0.3%，低于2%阈值，不应报警。
        self.assertFalse(LossMonitor.sustained_rise([1, 1.001, 1.002, 1.003]))
        # 连续三个窗口上涨且累计3%，应触发报警。
        self.assertTrue(LossMonitor.sustained_rise([1, 1.01, 1.02, 1.03]))

    # 验证报警必须等完整窗口收满，并且同一次上涨只通知一次。
    def test_complete_window_alert_and_no_duplicate(self):
        # 全部监控文件放在独立临时目录，避免污染真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试窗口缩短到2步，用10个小数值就能覆盖完整趋势。
            monitor = LossMonitor(directory, window_steps=2)
            # 每两个相同损失形成一个窗口，窗口均值依次1、1.01、1.02、1.03、1.04。
            for step, total in enumerate([1, 1, 1.01, 1.01, 1.02, 1.02, 1.03, 1.03, 1.04, 1.04], 1):
                # 仅总损失上升，蒸馏损失固定0.1，用于检查报警归因。
                monitor.observe(step, 1, {"student_total": total, "dist_raw": 0.1})
            # 读回持久化报警行，逐条解析JSON。
            records = [json.loads(s) for s in (Path(directory) / "loss_alerts.jsonl").read_text().splitlines()]
            # 连续上涨期间应只有一条通知记录。
            self.assertEqual(len(records), 1)
            # 第四个窗口结束于step8，报警应在这个时刻出现。
            self.assertEqual(records[0]["global_step"], 8)
            # 报警只应列出正在上涨的student_total。
            self.assertEqual(records[0]["metrics"], ["student_total"])
            # 记录必须正确声明每个证据窗口含2个批次。
            self.assertEqual(records[0]["windows"][0]["window_steps"], 2)

    # 验证NaN损失会立即报警，不必等待窗口统计。
    def test_nonfinite_alert(self):
        # 使用独立临时目录保存本项测试的报警。
        with tempfile.TemporaryDirectory() as directory:
            # 按默认50步窗口建立监控器，说明异常检查不依赖窗口结束。
            monitor = LossMonitor(directory)
            # 收到NaN时应抛出浮点异常。
            with self.assertRaises(FloatingPointError):
                # 在step2注入NaN总损失，模拟训练数值异常。
                monitor.observe(2, 1, {"student_total": float("nan")})
            # 报警文件必须明确标记为nonfinite_loss。
            self.assertEqual(json.loads((Path(directory) / "loss_alerts.jsonl").read_text())["kind"], "nonfinite_loss")

    # 续训从较早检查点恢复时，不得混入后来已经回滚的窗口。
    def test_resume_excludes_abandoned_later_windows(self):
        # 用临时目录模拟一次中断前后的共同实验目录。
        with tempfile.TemporaryDirectory() as directory:
            # 每2步记录窗口，便于检查恢复边界。
            monitor = LossMonitor(directory, window_steps=2)
            # 先生成step1到10的旧训练历史。
            for step in range(1, 11):
                # 输入逐步上升的总损失，留下step2/4/6/8/10窗口。
                monitor.observe(step, 1, {"student_total": float(step), "dist_raw": 0.1})
            # 模拟模型实际只恢复到step4。
            resumed = LossMonitor(directory, window_steps=2, resume_step=4)
            # 只允许保留step2和step4的历史窗口。
            self.assertEqual([r["global_step"] for r in resumed.windows], [2, 4])
            # 恢复后的step5进入一个新的损失窗口。
            resumed.observe(5, 1, {"student_total": 1., "dist_raw": 0.1})
            # step6补满新窗口，当前新训练的总损失已降为1。
            resumed.observe(6, 1, {"student_total": 1., "dist_raw": 0.1})
            # 有效历史应为旧step2、旧step4和新step6，不能混入旧step6/8/10。
            self.assertEqual([r["global_step"] for r in resumed.windows], [2, 4, 6])
            # 有效序列不再持续上涨，旧报警状态必须解除。
            self.assertFalse(resumed.active_alert)


# 直接运行文件时启动测试，导入时只定义测试类。
if __name__ == "__main__":
    # 汇总单元测试结果，并通过退出码告知调用方。
    unittest.main()
