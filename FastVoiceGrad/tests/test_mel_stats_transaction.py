# 本文件围绕 mel_stats_transaction 的生产接口进行小规模回归测试。
"""Regression tests for interruption-safe mel-statistics publication."""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# 按路径导入名称以数字开头的预处理脚本。
import importlib.util
# 计算文件或 JSON 的 SHA-256，检测内容被替换。
import hashlib
# 读写实验身份、指纹和报告的 JSON。
import json
# 沿用当前服务器 Python 解释器启动测试子进程。
import sys
# 将故意损坏的测试文件隔离在自动清理目录中。
import tempfile
# 创建可控的模块替身，隔离可选依赖或导入污染反例。
import types
# 使用标准库测试框架组织断言和失败报告。
import unittest
# 组合临时测试文件和仓库路径。
from pathlib import Path
# 临时替换依赖或注入故障，测试后自动恢复。
from unittest import mock

# 创建和读取极小的人工音频、梅尔或特征数组。
import numpy as np


# 由测试文件位置定位的仓库根目录。
ROOT = Path(__file__).resolve().parents[1]
# 发布事务本身不依赖进度条；租用服务器的精简 CPU 测试环境缺少 tqdm 时可使用替身。
# 本项目测试仍只允许在租用服务器运行，本地电脑仅负责读写与上传源码。
# 执行可能失败的测试操作，并为后面的恢复或依赖回退保留入口。
try:
    # 优先使用已安装进度条库；这里只验证能否导入。
    import tqdm  # noqa: F401
# 租用服务器的精简测试环境若没有进度条库，则使用无计算副作用的替身。
except ModuleNotFoundError:
    # 未安装进度条库时使用的无效果模块替身。
    tqdm_stub = types.ModuleType("tqdm")
    # 让进度条替身原样返回迭代对象，不改统计数据顺序。
    tqdm_stub.tqdm = lambda iterable, **_kwargs: iterable
    # 登记替身以满足统计脚本导入；不影响统计公式。
    sys.modules["tqdm"] = tqdm_stub
# 按路径导入统计量脚本的模块描述。
SPEC = importlib.util.spec_from_file_location(
    # 指定统计量脚本的导入名和仓库内真实路径。
    "voicegrad_compute_mel_stats", ROOT / "preprocess" / "04_compute_mel_stats.py"
)
# 只加载定义的统计量计算与原子发布模块。
mel_stats = importlib.util.module_from_spec(SPEC)
# 加载统计计算与事务函数定义，不自动执行命令行入口。
SPEC.loader.exec_module(mel_stats)


# 把同一类故障保护与期望行为组织成一组测试：MelStatsTransactionTests。
class MelStatsTransactionTests(unittest.TestCase):
    # 辅助函数不访问实例状态，因此声明为静态方法。
    @staticmethod
    # 生成彼此可区分的 80 维均值与正标准差用于发布事务测试。
    def arrays(offset: float) -> tuple[np.ndarray, np.ndarray]:
        # 逐梅尔通道的人工均值数组。
        mean = np.arange(mel_stats.N_MELS, dtype=np.float32) + offset
        # 逐梅尔通道的正标准差数组。
        std = np.arange(1, mel_stats.N_MELS + 1, dtype=np.float32) + offset
        # 成对返回均值和标准差，供同一代次发布测试。
        return mean, std

    # 确认一份代次清单同时绑定均值、标准差和报告的内容指纹。
    def test_generation_manifest_binds_complete_pair_with_hashes(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 同一测试代次的梅尔均值和标准差发布目录。
            stats_dir = root / "stats"
            # 本次统计事务的计算证明报告位置。
            report_path = root / "logs" / "mel_stats_report.json"
            # 生成偏移为零的基础 80 维均值和正标准差。
            mean, std = self.arrays(0.0)

            # 同时发布均值和标准差，取得绑定这一完整代次的清单。
            manifest = mel_stats.publish_stats_transaction(
                # 一次发布两个统计数组，并登记参与计算的 7 条样本。
                stats_dir, report_path, mean, std, {"rows_used": 7}
            )

            # 核对下面完整给出的断言条件。
            self.assertFalse(
                # 检查事务标记是否残留；正常完整提交后应不存在。
                (stats_dir / mel_stats.TRANSACTION_MARKER_NAME).exists()
            )
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with (stats_dir / mel_stats.CURRENT_MANIFEST_NAME).open(
                # 以 UTF-8 只读方式打开报告，避免任何写入。
                "r", encoding="utf-8"
            # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
            ) as handle:
                # 当前已提交生效的统计代次清单。
                current = json.load(handle)
            # 核对 current, manifest：应完全相等。
            self.assertEqual(current, manifest)
            # 不可变代次目录，存放成对统计量和证明。
            generation = (
                # 传入同一测试代次的梅尔均值和标准差发布目录。
                stats_dir
                # 进入保存不可变统计历史的代次目录。
                / mel_stats.GENERATION_DIRNAME
                # 再进入刚提交代次对应的唯一子目录。
                / manifest["generation_id"]
            )
            # 逐项处理 filename, expected in ，覆盖这组反例的全部输入。
            for filename, expected in (
                # 把均值文件与原始均值数组配对，核对发布结果。
                ("mel_mean.npy", mean),
                # 把标准差文件与原始标准差数组配对，核对发布结果。
                ("mel_std.npy", std),
            ):
                # 核对下面完整给出的断言条件。
                np.testing.assert_array_equal(
                    # 读取兼容路径的实际数组，与该代次原始数组逐元素比较。
                    np.load(stats_dir / filename, allow_pickle=False), expected
                )
                # 核对下面完整给出的断言条件。
                self.assertEqual(
                    # 独立计算不可变代次目录中文件的真实指纹。
                    mel_stats.sha256_file(generation / filename),
                    # 取代次清单登记的同文件指纹，要求与真实字节一致。
                    manifest["files"][filename]["sha256"],
                )
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with report_path.open("r", encoding="utf-8") as handle:
                # 读回刚发布的统计计算报告，核对代次、样本数和梅尔集合指纹。
                report = json.load(handle)
            # 核对 report[“把均值与标准差绑定到同一次计算的代次标识”], manifest[“把均值与标准差绑定到同一次计算的代次标识”]：应完全相等。
            self.assertEqual(report["generation_id"], manifest["generation_id"])
            # 核对 report[“实际参与统计的训练语音条数”], 7：应完全相等。
            self.assertEqual(report["rows_used"], 7)

    # 模拟均值已更新而标准差尚未更新时中断，验证先恢复同一代统计量。
    def test_interrupted_alias_update_is_recovered_before_new_generation(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 同一测试代次的梅尔均值和标准差发布目录。
            stats_dir = root / "stats"
            # 本次统计事务的计算证明报告位置。
            report_path = root / "logs" / "mel_stats_report.json"
            # 生成上一已提交代次的均值和标准差。
            old_mean, old_std = self.arrays(0.0)
            # 故障注入前已完整提交的旧统计代次。
            old_manifest = mel_stats.publish_stats_transaction(
                # 先完整发布旧代次，作为中断恢复的已提交基准。
                stats_dir, report_path, old_mean, old_std, {"label": "old"}
            )
            # 两种统计量都加 100，使半新半旧状态容易被识别。
            new_mean, new_std = self.arrays(100.0)

            # 生产原子复制函数的原始引用，用于非故障调用。
            original_copy = mel_stats._copy_file_atomic
            # 故障是否已注入一次的标记。
            failed = False

            # 只在第一次更新标准差别名时注入中断，准确模拟半发布故障。
            def interrupt_once(source: Path, destination: Path) -> None:
                # 让内层故障注入函数共享外层的一次性标记。
                nonlocal failed
                # 仅第一次写标准差别名时中断，模拟均值已新、标准差仍旧。
                if destination.name == "mel_std.npy" and not failed:
                    # 故障是否已注入一次的标记。
                    failed = True
                    # 抛出人工磁盘中断异常，检验恢复逻辑。
                    raise OSError("simulated interruption between mean and std")
                # 其他复制按真实原子实现继续，不额外改变文件行为。
                original_copy(source, destination)

            # 在限定作用域临时替换一个依赖，退出后恢复。
            with mock.patch.object(
                # 只对原子复制入口注入一次均值与标准差之间的中断。
                mel_stats, "_copy_file_atomic", side_effect=interrupt_once
            ):
                # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    # 执行真实的成对统计发布事务。
                    mel_stats.publish_stats_transaction(
                        # 传入同一测试代次的梅尔均值和标准差发布目录。
                        stats_dir,
                        # 传入测试证明或提取报告的输出路径。
                        report_path,
                        # 传入新代次的均值数组。
                        new_mean,
                        # 传入新代次的标准差数组。
                        new_std,
                        # 把报告标记为新代次，以辨认恢复最终指向。
                        {"label": "new"},
                    )

            # 尚未完成统计量发布的事务标记路径。
            marker = stats_dir / mel_stats.TRANSACTION_MARKER_NAME
            # 核对 marker.is_file()：必须成立。
            self.assertTrue(marker.is_file())
            # The compatibility pair may be mixed during an interrupted update,
            # but the one-file commit point must still identify the old pair.
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with (stats_dir / mel_stats.CURRENT_MANIFEST_NAME).open(
                # 以 UTF-8 只读方式打开报告，避免任何写入。
                "r", encoding="utf-8"
            # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
            ) as handle:
                # 核对 json.load(handle)[“把均值与标准差绑定到同一次计算的代次标识”], old_manifest[“把均值与标准差绑定到同一次计算的代次标识”]：应完全相等。
                self.assertEqual(json.load(handle)["generation_id"], old_manifest["generation_id"])
            # 核对下面完整给出的断言条件。
            np.testing.assert_array_equal(
                # 核对均值兼容文件已是新代次内容。
                np.load(stats_dir / "mel_mean.npy", allow_pickle=False), new_mean
            )
            # 核对下面完整给出的断言条件。
            np.testing.assert_array_equal(
                # 核对中断瞬间标准差兼容文件仍是旧代次。
                np.load(stats_dir / "mel_std.npy", allow_pickle=False), old_std
            )

            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with marker.open("r", encoding="utf-8") as handle:
                # 中断后待恢复的新统计代次 ID。
                pending_generation = json.load(handle)["generation_id"]
            # 核对下面完整给出的断言条件。
            self.assertTrue(
                # 读取事务标记并完成中断代次，禁止带着混合统计量继续训练。
                mel_stats.recover_pending_publication(stats_dir, report_path)
            )
            # 核对 marker.exists()：必须不成立。
            self.assertFalse(marker.exists())
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with (stats_dir / mel_stats.CURRENT_MANIFEST_NAME).open(
                # 以 UTF-8 只读方式打开报告，避免任何写入。
                "r", encoding="utf-8"
            # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
            ) as handle:
                # 核对 json.load(handle)[“把均值与标准差绑定到同一次计算的代次标识”], pending_generation：应完全相等。
                self.assertEqual(json.load(handle)["generation_id"], pending_generation)
            # 核对下面完整给出的断言条件。
            np.testing.assert_array_equal(
                # 核对均值兼容文件已是新代次内容。
                np.load(stats_dir / "mel_mean.npy", allow_pickle=False), new_mean
            )
            # 核对下面完整给出的断言条件。
            np.testing.assert_array_equal(
                # 恢复完成后标准差也必须变成新代次，与新均值成对。
                np.load(stats_dir / "mel_std.npy", allow_pickle=False), new_std
            )

    # 确认标准差含零时在创建事务前拒绝发布。
    def test_invalid_std_is_rejected_before_any_transaction(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 生成偏移为零的基础 80 维均值和正标准差。
            mean, std = self.arrays(0.0)
            # 把第一通道标准差改为零，测试除零风险会被预先阻止。
            std[0] = 0.0
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(ValueError, "positive"):
                # 执行真实的成对统计发布事务。
                mel_stats.publish_stats_transaction(
                    # 传入非法标准差但其他有效的发布参数，验证事务尚未开启就失败。
                    root / "stats", root / "report.json", mean, std, {}
                )
            # 核对下面完整给出的断言条件。
            self.assertFalse(
                # 确认无效输入没有留下半完成事务标记。
                (root / "stats" / mel_stats.TRANSACTION_MARKER_NAME).exists()
            )

    # 用两条人工梅尔验证真实命令行统计出可手算均值和标准差并一致发布。
    def test_cli_computes_and_publishes_one_consistent_generation(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 人工训练或全量样本清单路径。
            manifest_path = root / "train_manifest.csv"
            # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
            manifest_path.write_text(
                # 人工训练清单恰好含同一说话人的两条语音，用于独立手算统计值。
                "speaker,utt_id\np001,p001_001\np001,p001_002\n",
                # 文件文本编码，保证中文和摘要可稳定读写，按下列值传给生产函数。
                encoding="utf-8",
            )
            # 人工梅尔文件所在目录。
            mel_dir = root / "mel" / "p001"
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            mel_dir.mkdir(parents=True)
            # 80 个通道各自的固定数值基线。
            channels = np.arange(mel_stats.N_MELS, dtype=np.float32)[:, None]
            # 第一条人工梅尔，时间偏移为 0 和 1。
            first_mel = channels + np.array([[0.0, 1.0]], dtype=np.float32)
            # 第二条人工梅尔，时间偏移为 2 和 3。
            second_mel = channels + np.array([[2.0, 3.0]], dtype=np.float32)
            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(
                # 第一条人工训练梅尔的文件路径。
                mel_dir / "p001_001.npy",
                # 传入第一条人工梅尔，时间偏移为 0 和 1。
                first_mel,
                # 是否允许读取任意 Python 对象；特征数组禁止，按下列值传给生产函数。
                allow_pickle=False,
            )
            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(
                # 第二条人工训练梅尔的文件路径。
                mel_dir / "p001_002.npy",
                # 传入第二条人工梅尔，时间偏移为 2 和 3。
                second_mel,
                # 是否允许读取任意 Python 对象；特征数组禁止，按下列值传给生产函数。
                allow_pickle=False,
            )
            # 同一测试代次的梅尔均值和标准差发布目录。
            stats_dir = root / "stats"
            # 本例完整统计 CLI 的计算证明输出位置。
            report_path = root / "logs" / "report.json"

            # 为统计量 CLI 指定人工清单、梅尔目录、统计目录和报告位置。
            arguments = [
                # 模拟统计量 CLI 的程序名。
                "04_compute_mel_stats.py",
                # 指定本次统计或提取使用的人工清单。
                "--manifest",
                # 将人工训练或全量样本清单路径转换为命令行字符串。
                str(manifest_path),
                # 指定人工梅尔数组所在目录。
                "--mel_dir",
                # 将root / "mel"转换为命令行字符串。
                str(root / "mel"),
                # 指定均值与标准差的事务发布目录。
                "--stats_dir",
                # 将同一测试代次的梅尔均值和标准差发布目录转换为命令行字符串。
                str(stats_dir),
                # 指定统计量计算证明报告路径。
                "--report_path",
                # 将测试证明或提取报告的输出路径转换为命令行字符串。
                str(report_path),
            ]
            # 在租用服务器当前测试进程内运行 CLI，进度条被替换但真实统计与发布逻辑保持不变。
            # 这样可以直接检查生成的均值、标准差和内容证明，无需另起环境。
            # 临时设置真实 CLI 参数并移除进度条显示，不改变统计计算。
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                # 将进度条显示替换为原样迭代，使测试输出保持安静。
                mel_stats, "tqdm", side_effect=lambda iterable, **_kwargs: iterable
            ):
                # 在租用服务器进程内调用生产统计脚本的完整入口。
                mel_stats.main()
            # 核对下面完整给出的断言条件。
            np.testing.assert_allclose(
                # 读取最终发布的 80 维均值。
                np.load(stats_dir / "mel_mean.npy", allow_pickle=False),
                # 四帧偏移 0、1、2、3 的平均为 1.5，加各通道基线。
                np.arange(mel_stats.N_MELS, dtype=np.float32) + 1.5,
            )
            # 核对下面完整给出的断言条件。
            np.testing.assert_allclose(
                # 读取最终发布的 80 维标准差。
                np.load(stats_dir / "mel_std.npy", allow_pickle=False),
                # 四帧的总体方差为 1.25，因此每通道标准差为其平方根。
                np.full(mel_stats.N_MELS, np.sqrt(1.25), dtype=np.float32),
                # 采用百万分之一相对容差，允许正常浮点舍入。
                rtol=1e-6,
            )
            # 核对 (stats_dir / mel_stats.CURRENT_MANIFEST_NAME).is_file()：必须成立。
            self.assertTrue((stats_dir / mel_stats.CURRENT_MANIFEST_NAME).is_file())
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with report_path.open("r", encoding="utf-8") as handle:
                # 读回刚发布的统计计算报告，核对代次、样本数和梅尔集合指纹。
                report = json.load(handle)
            # 手工累计的梅尔内容集合 SHA-256。
            expected_content = hashlib.sha256()
            # 手工累计的梅尔路径与文件大小集合 SHA-256。
            expected_layout = hashlib.sha256()
            # 逐项处理 key, path, value in ，覆盖这组反例的全部输入。
            for key, path, value in (
                # 把语音键、实际梅尔文件和原始数组三者配对，独立计算集合指纹。
                ("p001/p001_001.npy", mel_dir / "p001_001.npy", first_mel),
                # 把语音键、实际梅尔文件和原始数组三者配对，独立计算集合指纹。
                ("p001/p001_002.npy", mel_dir / "p001_002.npy", second_mel),
            ):
                # 向手工累计的梅尔内容集合 SHA-256追加当前语音身份键。
                expected_content.update(key.encode("utf-8"))
                # 向手工累计的梅尔内容集合 SHA-256追加键值分隔符，防止拼接歧义。
                expected_content.update(b"\0")
                # 向手工累计的梅尔内容集合 SHA-256追加当前特征数组内容指纹。
                expected_content.update(mel_stats.sha256_array(value).encode("ascii"))
                # 向手工累计的梅尔内容集合 SHA-256追加记录结束换行，使逐文件边界明确。
                expected_content.update(b"\n")
                # 向手工累计的梅尔路径与文件大小集合 SHA-256追加当前语音身份键。
                expected_layout.update(key.encode("utf-8"))
                # 向手工累计的梅尔路径与文件大小集合 SHA-256追加键值分隔符，防止拼接歧义。
                expected_layout.update(b"\0")
                # 向手工累计的梅尔路径与文件大小集合 SHA-256追加当前文件字节长度。
                expected_layout.update(str(path.stat().st_size).encode("ascii"))
                # 向手工累计的梅尔路径与文件大小集合 SHA-256追加记录结束换行，使逐文件边界明确。
                expected_layout.update(b"\n")
            # 核对 report[“参与统计的梅尔文件数量”], 2：应完全相等。
            self.assertEqual(report["mel_collection_count"], 2)
            # 核对下面完整给出的断言条件。
            self.assertEqual(
                # 取报告登记的梅尔集合指纹，与下面手工累计结果比较。
                report["mel_collection_content_sha256"], expected_content.hexdigest()
            )
            # 核对下面完整给出的断言条件。
            self.assertEqual(
                # 取报告登记的梅尔集合指纹，与下面手工累计结果比较。
                report["mel_collection_layout_sha256"], expected_layout.hexdigest()
            )


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
