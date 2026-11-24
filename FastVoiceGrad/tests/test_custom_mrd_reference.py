# 本文件围绕 custom_mrd_reference 的生产接口进行小规模回归测试。
"""MRD STFT regression against the declared UnivNet reference convention."""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# 使用标准库测试框架组织断言和失败报告。
import unittest

# 构造小张量和参考公式，不加载正式训练数据。
import torch
# 使用独立的插值和补边参考运算进行回归比较。
import torch.nn.functional as F

# 导入本组测试实际调用的 hifigan.custom.mrd 接口；测试直接约束生产实现。
from hifigan.custom.mrd import SpectrogramDiscriminator, STFTResolution


# 把同一类故障保护与期望行为组织成一组测试：MRDReferenceSTFTTest。
class MRDReferenceSTFTTest(unittest.TestCase):
    # 把 MRD 频谱与手动反射补边、center=False 的参考 STFT 逐元素比较。
    def test_manual_reflect_padding_and_center_false(self) -> None:
        # 固定本例张量随机种子，保证参考公式比较可重复。
        torch.manual_seed(7)
        # MRD 测试的 FFT、帧移和窗长组合。
        resolution = STFTResolution(fft_size=512, hop_size=50, win_length=240)
        # 使用该分辨率的频谱判别器。
        discriminator = SpectrogramDiscriminator(resolution)
        # 人工构造的测试波形，不来自真实实验音频。
        waveform = torch.randn(2, 1, 8192)

        # 生产函数实际输出，供与独立参考比较。
        actual = discriminator._linear_magnitude_spectrogram(waveform)
        # 按 FFT 点数和帧移计算的单侧反射补边长度。
        pad = (resolution.fft_size - resolution.hop_size) // 2
        # 手工补边并去掉单通道维后的波形。
        padded = F.pad(waveform, (pad, pad), mode="reflect").squeeze(1)
        # 独立调用 PyTorch STFT 参考运算，再与 MRD 内部频谱逐元素核对。
        expected = torch.stft(
            # 传入手工补边并去掉单通道维后的波形。
            padded,
            # 傅里叶变换窗口点数，按下列值传给生产函数。
            n_fft=resolution.fft_size,
            # 当前 STFT 分支的帧移点数，按下列值传给生产函数。
            hop_length=resolution.hop_size,
            # 当前 STFT 分支的窗长，按下列值传给生产函数。
            win_length=resolution.win_length,
            # 与生产 MRD 一致的分析窗，按下列值传给生产函数。
            window=discriminator.window,
            # 是否由 STFT 自动在两侧补边，按下列值传给生产函数。
            center=False,
            # 边界延拓方式，按下列值传给生产函数。
            pad_mode="reflect",
            # 是否对 STFT 额外做幅度归一化，按下列值传给生产函数。
            normalized=False,
            # 实数信号是否仅保留非负频率，按下列值传给生产函数。
            onesided=True,
            # 是否先保留复数谱再取幅度，按下列值传给生产函数。
            return_complex=True,
        # 先取复数 STFT 幅度，再补回判别器需要的单通道维。
        ).abs().unsqueeze(1)

        # 核对 actual.shape, expected.shape：应完全相等。
        self.assertEqual(actual.shape, expected.shape)
        # 核对 torch.equal(actual, expected)：必须成立。
        self.assertTrue(torch.equal(actual, expected))


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
