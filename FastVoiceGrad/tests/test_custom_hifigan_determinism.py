# 本文件围绕 custom_hifigan_determinism 的生产接口进行小规模回归测试。
"""Deterministic data/crop and rented-host guard tests for custom HiFi-GAN."""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# 将故意损坏的测试文件隔离在自动清理目录中。
import tempfile
# 使用标准库测试框架组织断言和失败报告。
import unittest
# 组合临时测试文件和仓库路径。
from pathlib import Path

# 创建和读取极小的人工音频、梅尔或特征数组。
import numpy as np
# 写出合法 WAV，测试文件头正确但内容变化的情况。
import soundfile as sf
# 构造小张量和参考公式，不加载正式训练数据。
import torch

# 导入本组测试实际调用的 hifigan.custom.dataset 接口；测试直接约束生产实现。
from hifigan.custom.dataset import ManifestMelWaveDataset, VocoderItem
# 导入本组测试实际调用的 hifigan.custom.train 接口；测试直接约束生产实现。
from hifigan.custom.train import DeterministicEpochBatchSampler
# 导入本组测试实际调用的 voicegrad.runtime_guard 接口；测试直接约束生产实现。
from voicegrad.runtime_guard import RuntimeIdentityError, validate_runtime_identity


# 把同一类故障保护与期望行为组织成一组测试：DeterministicSamplerTest。
class DeterministicSamplerTest(unittest.TestCase):
    # 确认断点恢复后的批顺序精确等于原轮次未完成的尾段。
    def test_resume_tail_matches_uninterrupted_epoch(self) -> None:
        # 固定数据量、批大小、种子和轮次的采样参数。
        arguments = dict(dataset_size=23, batch_size=4, seed=1234, epoch=7)
        # 不中断时一整轮的确定性采样批列表。
        full = list(DeterministicEpochBatchSampler(**arguments))
        # 跳过已完成批后恢复的采样列表。
        resumed = list(
            # 按轮次和种子构造确定性批列表，使断点前后采样可逐项比较。
            DeterministicEpochBatchSampler(**arguments, start_batch=3)
        )
        # 核对 resumed, full[3:]：应完全相等。
        self.assertEqual(resumed, full[3:])
        # 核对 len(full), 23 // 4：应完全相等。
        self.assertEqual(len(full), 23 // 4)

    # 确认同轮同种子裁剪一致，不同轮次有新的采样顺序。
    def test_crop_seed_is_stable_and_epoch_specific(self) -> None:
        # 第 0 轮确定性采样结果。
        epoch_zero = list(
            # 按轮次和种子构造确定性批列表，使断点前后采样可逐项比较。
            DeterministicEpochBatchSampler(
                # 12 条样本每批 3 条，固定种子 1234，构造第 0 轮采样。
                dataset_size=12, batch_size=3, seed=1234, epoch=0
            )
        )
        # 同一轮同一种子重复采样的结果。
        repeated = list(
            # 按轮次和种子构造确定性批列表，使断点前后采样可逐项比较。
            DeterministicEpochBatchSampler(
                # 12 条样本每批 3 条，固定种子 1234，构造第 0 轮采样。
                dataset_size=12, batch_size=3, seed=1234, epoch=0
            )
        )
        # 第 1 轮重新生成的采样结果。
        epoch_one = list(
            # 按轮次和种子构造确定性批列表，使断点前后采样可逐项比较。
            DeterministicEpochBatchSampler(
                # 保持数据与种子不变，只改为第 1 轮，采样顺序应相应更新。
                dataset_size=12, batch_size=3, seed=1234, epoch=1
            )
        )
        # 核对 epoch_zero, repeated：应完全相等。
        self.assertEqual(epoch_zero, repeated)
        # 核对 epoch_zero, epoch_one：应不同。
        self.assertNotEqual(epoch_zero, epoch_one)

    # 确认样本索引加裁剪种子可复现完全相同的梅尔与波形片段。
    def test_dataset_composite_index_repeats_exact_crop(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 测试音频文件路径。
            wav_path = root / "sample.wav"
            # 对应测试音频的梅尔数组路径。
            mel_path = root / "sample.npy"
            # 人工构造的测试波形，不来自真实实验音频。
            waveform = np.linspace(-0.5, 0.5, 6_000, dtype=np.float32)
            # 有可辨认数值的 80 通道测试梅尔。
            mel = np.arange(80 * 100, dtype=np.float32).reshape(80, 100)
            # 写入测试 WAV；即使格式合法，后续也必须通过内容指纹检查。
            sf.write(wav_path, waveform, 22_050, subtype="FLOAT")
            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(mel_path, mel, allow_pickle=False)
            # 按正式接口构造的极小声码器数据集。
            dataset = ManifestMelWaveDataset(
                # 梅尔与波形一一配对的样本列表，按下列值传给生产函数。
                items=[VocoderItem("p001", "p001_001", wav_path, mel_path)],
                # 单次声码器训练的波形点数，按下列值传给生产函数。
                segment_size=8192,
                # 相邻梅尔帧的采样点间隔，按下列值传给生产函数。
                hop_size=256,
                # 声码器梅尔通道数，按下列值传给生产函数。
                num_mels=80,
                # 声码器音频采样率，按下列值传给生产函数。
                sampling_rate=22_050,
                # 傅里叶变换窗口点数，按下列值传给生产函数。
                n_fft=1024,
                # 频谱分析窗长度，按下列值传给生产函数。
                win_size=1024,
                # 梅尔滤波最低频率，按下列值传给生产函数。
                fmin=0,
                # 梅尔滤波最高频率，按下列值传给生产函数。
                fmax=8000,
            )
            # 相同样本和裁剪种子的第一次读取结果。
            first = dataset[(0, 987654321)]
            # 相同样本和裁剪种子的第二次读取结果。
            second = dataset[(0, 987654321)]
            # 核对 torch.equal(first[0], second[0])：必须成立。
            self.assertTrue(torch.equal(first[0], second[0]))
            # 核对 torch.equal(first[1], second[1])：必须成立。
            self.assertTrue(torch.equal(first[1], second[1]))
            # 核对 float(first[1].abs().max()), 0.95, places=5：应在浮点容差内相等。
            self.assertAlmostEqual(float(first[1].abs().max()), 0.95, places=5)


# 把同一类故障保护与期望行为组织成一组测试：RuntimeGuardTest。
class RuntimeGuardTest(unittest.TestCase):
    # 确认已登记的租用主机身份可以通过运行保护。
    def test_registered_identity_passes(self) -> None:
        # 测试目录的 Path 对象。
        root = Path("/synthetic/rental")
        # 调用租用主机身份保护，只有登记的主机与路径组合才允许通过。
        validate_runtime_identity(
            # 当前主机名，按下列值传给生产函数。
            hostname="rental",
            # 运行主机 machine-id 的不可逆指纹，按下列值传给生产函数。
            machine_id_sha256="a" * 64,
            # 测试用项目根目录，按下列值传给生产函数。
            project_root=root,
            # 事先登记的租用主机名，按下列值传给生产函数。
            expected_hostname="rental",
            # 事先登记的租用主机指纹，按下列值传给生产函数。
            expected_machine_id_sha256="a" * 64,
            # 事先登记的租用项目绝对路径，按下列值传给生产函数。
            expected_project_root=root,
        )

    # 确认本机与禁止服务器都无法绕过运行保护。
    def test_unknown_and_forbidden_hosts_fail(self) -> None:
        # 允许主机和项目目录的公共校验参数。
        common = dict(
            # 运行主机 machine-id 的不可逆指纹，按下列值传给生产函数。
            machine_id_sha256="a" * 64,
            # 测试用项目根目录，按下列值传给生产函数。
            project_root=Path("/synthetic/rental"),
            # 事先登记的租用主机名，按下列值传给生产函数。
            expected_hostname="rental",
            # 事先登记的租用主机指纹，按下列值传给生产函数。
            expected_machine_id_sha256="a" * 64,
            # 事先登记的租用项目绝对路径，按下列值传给生产函数。
            expected_project_root=Path("/synthetic/rental"),
        )
        # 该反例应被明确拒绝；同时核对异常类型。
        with self.assertRaises(RuntimeIdentityError):
            # 调用租用主机身份保护，只有登记的主机与路径组合才允许通过。
            validate_runtime_identity(hostname="local", **common)
        # 该反例应被明确拒绝；同时核对异常类型。
        with self.assertRaises(RuntimeIdentityError):
            # 调用租用主机身份保护，只有登记的主机与路径组合才允许通过。
            validate_runtime_identity(hostname="speech101", **common)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
