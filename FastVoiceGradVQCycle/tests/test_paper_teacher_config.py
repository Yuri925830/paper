# 本文件围绕 paper_teacher_config 的生产接口进行小规模回归测试。
"""Regression checks for the settings explicitly fixed by the papers."""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# 使用标准库测试框架组织断言和失败报告。
import unittest

# 导入本组测试实际调用的 voicegrad.config 接口；测试直接约束生产实现。
from voicegrad.config import PAPER_CONFIG


# 把同一类故障保护与期望行为组织成一组测试：PaperTeacherConfigTests。
class PaperTeacherConfigTests(unittest.TestCase):
    # 锁定教师实验的说话人数及梅尔提取设置。
    def test_vctk_and_mel_settings(self) -> None:
        # 核对 PAPER_CONFIG.expected_total_speakers, 110：应完全相等。
        self.assertEqual(PAPER_CONFIG.expected_total_speakers, 110)
        # 核对 PAPER_CONFIG.expected_train_speakers, 100：应完全相等。
        self.assertEqual(PAPER_CONFIG.expected_train_speakers, 100)
        # 核对 PAPER_CONFIG.expected_test_speakers, 10：应完全相等。
        self.assertEqual(PAPER_CONFIG.expected_test_speakers, 10)
        # 核对 PAPER_CONFIG.test_utterances_per_speaker, 10：应完全相等。
        self.assertEqual(PAPER_CONFIG.test_utterances_per_speaker, 10)
        # 核对 PAPER_CONFIG.sample_rate, 22050：应完全相等。
        self.assertEqual(PAPER_CONFIG.sample_rate, 22050)
        # 核对 PAPER_CONFIG.n_mels, 80：应完全相等。
        self.assertEqual(PAPER_CONFIG.n_mels, 80)
        # 核对 PAPER_CONFIG.n_fft, 1024：应完全相等。
        self.assertEqual(PAPER_CONFIG.n_fft, 1024)
        # 核对 PAPER_CONFIG.hop_size, 256：应完全相等。
        self.assertEqual(PAPER_CONFIG.hop_size, 256)
        # 核对 PAPER_CONFIG.win_size, 1024：应完全相等。
        self.assertEqual(PAPER_CONFIG.win_size, 1024)

    # 锁定教师扩散日程、网络宽度、优化器和 500 轮终点。
    def test_teacher_training_settings(self) -> None:
        # 核对 PAPER_CONFIG.diffusion_steps, 1000：应完全相等。
        self.assertEqual(PAPER_CONFIG.diffusion_steps, 1000)
        # 核对 PAPER_CONFIG.hidden_channels, 512：应完全相等。
        self.assertEqual(PAPER_CONFIG.hidden_channels, 512)
        # 核对 PAPER_CONFIG.batch_size, 32：应完全相等。
        self.assertEqual(PAPER_CONFIG.batch_size, 32)
        # 核对 PAPER_CONFIG.learning_rate, 2e-4：应完全相等。
        self.assertEqual(PAPER_CONFIG.learning_rate, 2e-4)
        # 核对 PAPER_CONFIG.adam_beta1, 0.9：应完全相等。
        self.assertEqual(PAPER_CONFIG.adam_beta1, 0.9)
        # 核对 PAPER_CONFIG.adam_beta2, 0.999：应完全相等。
        self.assertEqual(PAPER_CONFIG.adam_beta2, 0.999)
        # 核对 PAPER_CONFIG.paper_epochs, 500：应完全相等。
        self.assertEqual(PAPER_CONFIG.paper_epochs, 500)
        # 核对 PAPER_CONFIG.epochs, 500：应完全相等。
        self.assertEqual(PAPER_CONFIG.epochs, 500)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
