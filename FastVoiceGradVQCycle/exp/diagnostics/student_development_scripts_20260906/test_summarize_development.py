"""在Matpool用纯合成数字核验开发汇总；不读取语音或模型。"""

import importlib.util  # 从独立脚本位置导入统计函数。
import sys  # 通过命令行指定已上传的脚本，避免本地运行。
import unittest  # 使用标准库测试，不依赖训练测试框架。
from pathlib import Path  # 限定出租服务器路径。

ROOT = Path('/root/FsatVoiceGrad')  # 所有测试也只能在Matpool执行。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地编辑不运行测试。
    raise RuntimeError('Tests run only on the Matpool rental.')  # 提前阻断错误环境。
SCRIPT = Path(sys.argv.pop(1)).resolve()  # 第一个参数是待审核的独立汇总脚本。
spec = importlib.util.spec_from_file_location('development_summary', SCRIPT)  # 不导入项目训练代码。
summary = importlib.util.module_from_spec(spec)  # 准备独立命名空间。
spec.loader.exec_module(summary)  # 脚本自己的机器指纹检查仍会执行。
np = summary.np  # 复用相同NumPy版本的纯CPU小数组。


class SummaryTests(unittest.TestCase):  # 每个测试都只用人工可核算的数字。
    def test_cer_is_ratio_of_sums(self):  # 防止平均句CER产生错误主数。
        sums = np.array([[1, 2, 1, 3, 4, 3, 3, 3, 0.4, 1], [1, 100, 0, 2, 3, 2, 2, 2, 0.2, 1]], dtype=float)  # 两句分母相差很大。
        point = summary.values_from_sums(sums.sum(axis=0))  # 正确总CER为2/102。
        self.assertAlmostEqual(point[0], 100 * 2 / 102)  # 不等于(50+1)/2。
        self.assertNotAlmostEqual(point[0], 25.5)  # 显式拒绝宏平均混用。
        self.assertAlmostEqual(point[1], 50)  # SVA仍按两条验证标记平均。

    def test_sva_boundary_is_strict(self):  # 恰好0.25必须未通过。
        self.assertEqual(summary.validate_sva({'cosine_score': '0.25', 'threshold': '0.25', 'verified': '0'}), (0.25, 0))  # 与原评分器严格大于相同。
        with self.assertRaises(ValueError):  # 错把边界判通过会被拒绝。
            summary.validate_sva({'cosine_score': '0.25', 'threshold': '0.25', 'verified': '1'})  # 不允许改成大于等于。
        with self.assertRaises(ValueError):  # 其他阈值也必须失败。
            summary.validate_sva({'cosine_score': '0.3', 'threshold': '0.2', 'verified': '1'})  # 禁止开发时挑阈值。

    def test_cluster_draws_and_reproducibility(self):  # 每次抽10个整簇，四系统复用同一权重。
        weights = summary.bootstrap_weights(10, 1000, 1234)  # 生成一小批CPU随机整数。
        np.testing.assert_array_equal(weights.sum(axis=1), np.full(1000, 10))  # 每次抽样数量恰好10。
        np.testing.assert_array_equal(weights, summary.bootstrap_weights(10, 1000, 1234))  # 同种子完全复现。
        self.assertTrue(np.any(weights > 1))  # 有放回抽样应能重复某簇。

    def test_pairing_cancels_identical_systems(self):  # 同一系统相减的区间必须严格为0。
        sums = np.tile(np.array([1, 100, 8, 27, 36, 27, 27, 27, 4.5, 9], dtype=float), (10, 1))  # 每簇9个目标。
        sums[:, 3] += np.arange(10)  # 各源语音的DNSMOS可不同。
        values = summary.values_from_sums(summary.bootstrap_weights(10, 1000, 3) @ sums)  # 重采样带来绝对分数变化。
        np.testing.assert_array_equal(summary.confidence_interval(values - values), np.zeros((8, 2)))  # 配对不能分别抽样。

    def test_constant_gain_has_constant_paired_interval(self):  # 每条语音同幅改善时差值区间应保留该幅度。
        baseline = np.tile(np.array([1, 100, 8, 27, 36, 27, 27, 27, 4.5, 9], dtype=float), (10, 1))  # DNSMOS原均值3。
        changed = baseline.copy()  # 两系统唯一区别是DNSMOS每对加0.4。
        changed[:, 3] += 9 * 0.4  # 每簇9目标的总分增加3.6。
        weights = summary.bootstrap_weights(10, 1000, 19)  # 两系统共享整簇抽样。
        delta = summary.values_from_sums(weights @ changed) - summary.values_from_sums(weights @ baseline)  # 同一抽样先相减。
        np.testing.assert_allclose(summary.confidence_interval(delta)[2], [0.4, 0.4], atol=1e-12)  # 不产生虚假的独立抽样噪声。

    def test_cer_bootstrap_reweights_denominators(self):  # 重复长句簇时CER分母也应重复。
        sums = np.array([[1, 2, 1, 3, 4, 3, 3, 3, 0.4, 1], [1, 100, 0, 2, 3, 2, 2, 2, 0.2, 1]], dtype=float)  # 两个人工簇。
        values = summary.values_from_sums(np.array([[2, 0], [1, 1], [0, 2]]) @ sums)  # 三种已知抽样结果。
        np.testing.assert_allclose(values[:, 0], [50, 100 * 2 / 102, 1])  # 逐次采用总错误除以总字符。

    def test_text_distance_and_empty_hypothesis(self):  # 无需加载ASR即可核验保存文本的计数。
        self.assertEqual(summary.normalize_cer("Don't stop, 12!"), "DON'T STOP")  # 撇号保留，数字与标点移除。
        self.assertEqual(summary.edit_distance('ABC', 'ADC'), 1)  # 一次替换。
        self.assertEqual(summary.edit_distance('ABC', ''), 3)  # 空识别是三次删除，不可忽略。

    def test_nonfinite_and_empty_denominator_rejected(self):  # 有坏分数就停止，不从均值中剔除。
        with self.assertRaises(ValueError):  # NaN不可参与区间。
            summary.finite('nan')  # 不把NaN写为JSON数值。
        with self.assertRaises(ValueError):  # CER不能用零字符分母。
            summary.values_from_sums(np.zeros(10))  # 不自动改用宏平均兜底。

    def test_average_ranks_handle_duplicate_durations(self):  # 两条源句帧数相等时Spearman使用平均秩。
        np.testing.assert_array_equal(summary.average_ranks([3, 1, 1, 5]), [2, 0.5, 0.5, 3])  # 并列最短句各取0.5。
        self.assertIsNone(summary.descriptive_correlation([1, 1, 1], [2, 3, 4]))  # 时长完全一样则不报相关系数。


if __name__ == '__main__':  # 命令行直接启动纯CPU单元测试。
    unittest.main()  # 不生成音频、不重新评价、不触碰训练锁。
