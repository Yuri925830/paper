# 按路径导入名称以数字开头的预处理脚本。
import importlib.util
# 将故意损坏的测试文件隔离在自动清理目录中。
import tempfile
# 使用标准库测试框架组织断言和失败报告。
import unittest
# 用标准 WAV 接口生成短 PCM 文件，控制声道和采样率。
import wave
# 组合临时测试文件和仓库路径。
from pathlib import Path

# 创建和读取极小的人工音频、梅尔或特征数组。
import numpy as np


# 由测试文件位置定位的仓库根目录。
ROOT = Path(__file__).resolve().parents[1]


# 按文件位置导入数字开头的预处理脚本，不启动其命令行主流程。
def load_script(module_name, relative_path):
    # 按文件路径建立的 Python 模块加载描述。
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    # 尚待执行脚本定义的模块对象。
    module = importlib.util.module_from_spec(spec)
    # 执行模块定义；__main__ 入口不会因此启动。
    spec.loader.exec_module(module)
    # 返回已加载定义的预处理模块，供测试直接调用。
    return module


# 用于 WAV 重采样续作校验的生产脚本模块。
resample = load_script("voicegrad_resample", "preprocess/01_resample_to_22050.py")
# 导入 BNF 提取脚本模块，测试数组校验和原子保存入口。
bnf = load_script("voicegrad_extract_bnf", "preprocess/05_extract_bnf.py")
# 导入说话人向量脚本模块，测试 256 维和有限性保护。
speaker = load_script("voicegrad_extract_speaker", "preprocess/06_extract_spk_emb.py")


# 把同一类故障保护与期望行为组织成一组测试：ResampleResumeSafetyTest。
class ResampleResumeSafetyTest(unittest.TestCase):
    # 辅助函数不访问实例状态，因此声明为静态方法。
    @staticmethod
    # 写极短的测试 PCM WAV，独立控制声道、采样率和长度。
    def write_wav(path, channels=1, sample_rate=22050, frames=8):
        # 按指定编码打开测试文件，作用域结束后自动关闭。
        with wave.open(str(path), "wb") as handle:
            # 设置 WAV 声道数，区分单声道与双声道反例。
            handle.setnchannels(channels)
            # 每个采样点使用 2 字节 PCM16 编码。
            handle.setsampwidth(2)
            # 写指定采样率，以测试 22050 Hz 限制。
            handle.setframerate(sample_rate)
            # 按声道数和帧数写零 PCM，frames=0 模拟空音频。
            handle.writeframes(b"\x00\x00" * channels * frames)

    # 确认只有完整、非空、单声道 22050 Hz WAV 才允许续作跳过。
    def test_validation_requires_mono_22050_and_nonempty(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 满足单声道和采样率要求的测试 WAV。
            valid = root / "valid.wav"
            # 生成当前指定的 WAV 反例，仅改变这一例需要检验的格式条件。
            self.write_wav(valid)
            # 从实际 WAV 文件核对声道、采样率、非空和长度完整性。
            resample.validate_resampled_wav(valid)

            # 故意构造的双声道反例 WAV。
            stereo = root / "stereo.wav"
            # 生成当前指定的 WAV 反例，仅改变这一例需要检验的格式条件。
            self.write_wav(stereo, channels=2)
            # 该反例应被明确拒绝；同时核对异常类型。
            with self.assertRaises(ValueError):
                # 从实际 WAV 文件核对声道、采样率、非空和长度完整性。
                resample.validate_resampled_wav(stereo)

            # 故意构造的错误采样率反例 WAV。
            wrong_rate = root / "wrong-rate.wav"
            # 生成当前指定的 WAV 反例，仅改变这一例需要检验的格式条件。
            self.write_wav(wrong_rate, sample_rate=16000)
            # 该反例应被明确拒绝；同时核对异常类型。
            with self.assertRaises(ValueError):
                # 从实际 WAV 文件核对声道、采样率、非空和长度完整性。
                resample.validate_resampled_wav(wrong_rate)

            # 仅有 WAV 头而没有采样点的反例。
            empty = root / "empty.wav"
            # 生成当前指定的 WAV 反例，仅改变这一例需要检验的格式条件。
            self.write_wav(empty, frames=0)
            # 该反例应被明确拒绝；同时核对异常类型。
            with self.assertRaises(ValueError):
                # 从实际 WAV 文件核对声道、采样率、非空和长度完整性。
                resample.validate_resampled_wav(empty)

            # 将被截去末尾字节的反例 WAV。
            truncated = root / "truncated.wav"
            # 生成当前指定的 WAV 反例，仅改变这一例需要检验的格式条件。
            self.write_wav(truncated)
            # 需要完整写入文件的字节内容，按下列值传给生产函数。
            payload = truncated.read_bytes()
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            truncated.write_bytes(payload[:-1])
            # 该反例应被明确拒绝；同时核对异常类型。
            with self.assertRaises(ValueError):
                # 从实际 WAV 文件核对声道、采样率、非空和长度完整性。
                resample.validate_resampled_wav(truncated)


# 把同一类故障保护与期望行为组织成一组测试：FeatureResumeSafetyTest。
class FeatureResumeSafetyTest(unittest.TestCase):
    # 确认 BNF 续作仅跳过合法的 144 通道数组，转置错误必须重做。
    def test_bnf_skip_requires_valid_canonical_array(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 预处理测试的特征输出目录。
            out_dir = Path(directory)
            # 按说话人/语音.npy 规范组织人工特征文件。
            path = out_dir / "p001" / "p001_001.npy"
            # 人工清单只含 p001_001 一条，测试逐文件续作或统计发布保护。
            rows = [{"speaker": "p001", "utt_id": "p001_001"}]

            # 通过生产 BNF 原子写入接口保存；新内容非法时应保留旧文件。
            bnf.atomic_save_bnf(path, np.ones((144, 3), dtype=np.float32))
            # 核对 bnf.validate_bnf_file(path), (144, 3)：应完全相等。
            self.assertEqual(bnf.validate_bnf_file(path), (144, 3))
            # 核对 bnf.filter_missing_rows(rows, out_dir), []：应完全相等。
            self.assertEqual(bnf.filter_missing_rows(rows, out_dir), [])
            # 核对 bnf.filter_missing_rows(rows, out_dir, force=True), rows：应完全相等。
            self.assertEqual(bnf.filter_missing_rows(rows, out_dir, force=True), rows)

            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(path, np.ones((3, 144), dtype=np.float32), allow_pickle=False)
            # 核对 bnf.filter_missing_rows(rows, out_dir), rows：应完全相等。
            self.assertEqual(bnf.filter_missing_rows(rows, out_dir), rows)

    # 确认含 NaN 的新 BNF 不会覆盖先前有效文件。
    def test_failed_bnf_validation_does_not_replace_valid_destination(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 当前需要核验或写入的具体文件路径，按下列值传给生产函数。
            path = Path(directory) / "feature.npy"
            # 形状合法且全部有限的 BNF 数组。
            good = np.ones((144, 2), dtype=np.float32)
            # 通过生产 BNF 原子写入接口保存；新内容非法时应保留旧文件。
            bnf.atomic_save_bnf(path, good)
            # 复制有效数组后注入非法值的反例。
            bad = good.copy()
            # 向合法 BNF 注入一个 NaN，检查发布前保护。
            bad[0, 0] = np.nan
            # 该反例应被明确拒绝；同时核对异常类型。
            with self.assertRaises(ValueError):
                # 通过生产 BNF 原子写入接口保存；新内容非法时应保留旧文件。
                bnf.atomic_save_bnf(path, bad)
            # 核对 np.load(path, allow_pickle=False), good：数组应逐元素相等。
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), good)

    # 确认说话人特征必须恰好 256 维且全有限，才能在续作时跳过。
    def test_speaker_skip_requires_exact_256_finite_values(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 预处理测试的特征输出目录。
            out_dir = Path(directory)
            # 按说话人/语音.npy 规范组织人工特征文件。
            path = out_dir / "p001" / "p001_001.npy"
            # 人工清单只含 p001_001 一条，测试逐文件续作或统计发布保护。
            rows = [{"speaker": "p001", "utt_id": "p001_001"}]

            # 通过生产原子写入接口保存 256 维说话人特征。
            speaker.atomic_save_speaker(path, np.ones(256, dtype=np.float32))
            # 核对 speaker.validate_speaker_file(path), (256,)：应完全相等。
            self.assertEqual(speaker.validate_speaker_file(path), (256,))
            # 核对 speaker.filter_missing_rows(rows, out_dir), []：应完全相等。
            self.assertEqual(speaker.filter_missing_rows(rows, out_dir), [])

            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(path, np.ones(255, dtype=np.float32), allow_pickle=False)
            # 核对 speaker.filter_missing_rows(rows, out_dir), rows：应完全相等。
            self.assertEqual(speaker.filter_missing_rows(rows, out_dir), rows)
            # 该反例应被明确拒绝；同时核对异常类型。
            with self.assertRaises(ValueError):
                # 检查说话人向量维度，不能把 255 维错误输入自动当作 256 维。
                speaker.standardize_embed_shape(np.ones(255, dtype=np.float32))

            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(path, np.full(256, np.inf, dtype=np.float32), allow_pickle=False)
            # 核对 speaker.filter_missing_rows(rows, out_dir), rows：应完全相等。
            self.assertEqual(speaker.filter_missing_rows(rows, out_dir), rows)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
