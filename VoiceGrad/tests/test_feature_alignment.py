# 本文件围绕 feature_alignment 的生产接口进行小规模回归测试。
"""Small regression tests for the optional BNF-to-mel alignment modes."""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# 使用标准库测试框架组织断言和失败报告。
import unittest

# 构造小张量和参考公式，不加载正式训练数据。
import torch
# 使用独立的插值和补边参考运算进行回归比较。
import torch.nn.functional as F

# 导入本组测试实际调用的 voicegrad.feature_alignment 接口；测试直接约束生产实现。
from voicegrad.feature_alignment import (
    # 历史按长度线性插值的对齐模式名称。
    LEGACY_ALIGNMENT,
    # 按实际帧中心时间插值的对齐模式名称。
    TIME_ALIGNMENT,
    # 把BNF时间轴对齐到源mel帧数。
    align_bnf_to_mel,
    # 将梅尔帧中心换算到 BNF 采样网格的位置。
    mel_centres_on_bnf_grid,
    # 按 HiFi-GAN 补边规则计算首帧中心采样点。
    mel_first_frame_center_samples,
)


# 把同一类故障保护与期望行为组织成一组测试：FeatureAlignmentTests。
class FeatureAlignmentTests(unittest.TestCase):
    # 核对 HiFi-GAN 梅尔第一帧中心位于半个 hop，即 128 个采样点。
    def test_hifigan_first_mel_frame_center_is_half_a_hop(self) -> None:
        # 核对 mel_first_frame_center_samples(1024, 256), 128.0：应完全相等。
        self.assertEqual(mel_first_frame_center_samples(1024, 256), 128.0)

    # 确认旧长度插值模式逐元素复现历史 BNF 对齐。
    def test_legacy_mode_matches_previous_interpolate_call(self) -> None:
        # 形状固定且数值可辨认的 BNF 测试输入。
        bnf = torch.arange(2 * 5, dtype=torch.float32).reshape(1, 2, 5)
        # 直接调用历史长度线性插值，作为旧 BNF 对齐模式的独立基准。
        expected = F.interpolate(
            # 传入形状固定且数值可辨认的 BNF 测试输入。
            bnf,
            # 对齐后的目标帧数，按下列值传给生产函数。
            size=7,
            # 插值或补边方式，按下列值传给生产函数。
            mode="linear",
            # 是否强制首尾像素中心对齐，按下列值传给生产函数。
            align_corners=False,
        )
        # 生产函数实际输出，供与独立参考比较。
        actual = align_bnf_to_mel(bnf, 7, mode=LEGACY_ALIGNMENT)
        # 核对 torch.equal(actual, expected)：必须成立。
        self.assertTrue(torch.equal(actual, expected))

    # 确认按真实时间对齐时，梅尔帧中心正确映射到 10 ms BNF 网格。
    def test_time_grid_positions_use_physical_frame_centres(self) -> None:
        # 梅尔帧中心映射到 BNF 网格的位置。
        positions = mel_centres_on_bnf_grid(
            # 计算前三个梅尔帧中心，以便独立手算。
            3,
            # 音频每秒采样点数，按下列值传给生产函数。
            sample_rate=22050,
            # 傅里叶变换窗口点数，按下列值传给生产函数。
            n_fft=1024,
            # 相邻梅尔帧的采样点间隔，按下列值传给生产函数。
            hop_size=256,
            # BNF 两帧之间的真实时间间隔，按下列值传给生产函数。
            bnf_frame_shift_seconds=0.010,
        )
        # 三个梅尔帧中心分别在 128、384、640 采样点；除以每个 BNF 帧 220.5 点换算网格位置。
        expected = torch.tensor([128.0, 384.0, 640.0]) / 220.5
        # 核对 torch.allclose(positions, expected)：必须成立。
        self.assertTrue(torch.allclose(positions, expected))

    # 确认时间对齐后帧数正确，边界值不越过原 BNF 范围。
    def test_time_mode_preserves_shape_and_endpoint_bounds(self) -> None:
        # 形状固定且数值可辨认的 BNF 测试输入。
        bnf = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
        # 按真实帧中心插值得到的 BNF。
        aligned = align_bnf_to_mel(bnf, 11, mode=TIME_ALIGNMENT)
        # 核对 tuple(aligned.shape), (1, 1, 11)：应完全相等。
        self.assertEqual(tuple(aligned.shape), (1, 1, 11))
        # 核对 float(aligned.min()), 0.0：应不小于给定下限。
        self.assertGreaterEqual(float(aligned.min()), 0.0)
        # 核对 float(aligned.max()), 5.0：应不超过给定上限。
        self.assertLessEqual(float(aligned.max()), 5.0)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
