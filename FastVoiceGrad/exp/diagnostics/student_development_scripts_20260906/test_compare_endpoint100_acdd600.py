"""只在出租机用合成90对记录验证两个候选的配对差值方向。"""

import importlib.util  # 从显式路径加载独立比较脚本。
import sys  # 第一个参数指定服务器脚本位置。
import unittest  # 不调用训练测试或模型。
from pathlib import Path  # 本地禁止执行项目测试。

if not sys.platform.startswith('linux') or not Path('/root/FsatVoiceGrad').is_dir():  # 所有CPU测试仍限定Matpool。
    raise RuntimeError('Run only on Matpool.')  # 错误运行环境直接停止。
spec = importlib.util.spec_from_file_location('_endpoint_pairwise_test', Path(sys.argv.pop(1)).resolve())  # 私有加载辅助函数。
module = importlib.util.module_from_spec(spec)  # 准备隔离命名空间。
spec.loader.exec_module(module)  # 真实主机指纹与冻结辅助代码哈希仍会核验。


class PairwiseTests(unittest.TestCase):  # 数据全部人工可核算，实际评估文件不参与测试。
    def test_identical_rows_have_zero_interval(self):  # 完全相同的两候选不应出现虚假差值。
        clusters = [(f'p{i}', f'utt{i}') for i in range(10)]  # 合成10条源录音。
        rows = []  # 每个源固定9个目标。
        for speaker, utterance in clusters:  # 构造相同的计数和评分。
            for target in range(9):  # 目标无需加载真实说话人。
                rows.append({'source_speaker': speaker, 'source_utt': utterance, 'char_errors': 1, 'reference_chars': 50, 'verified': 1, 'mos_ovr': 3, 'utmos': 4, 'mos_sig': 3, 'mos_bak': 4, 'p808_mos': 3, 'cosine_score': 0.8})  # 完整的统计输入字段。
        values, _ = module.paired_statistics(rows, rows, clusters)  # 两系统复用同一行记录。
        for item in values.values():  # 所有8项指标差值都应严格为0。
            self.assertEqual(item['delta_endpoint100_minus_acdd600'], 0)  # 原始点差为0。
            self.assertEqual(item['cluster_bootstrap_95_percentile_ci'], [0, 0])  # 配对区间同样为0。
        changed = [{**row, 'utmos': row['utmos'] + 0.5, 'char_errors': 0} for row in rows]  # endpoint每对UTMOS提升0.5且字符错误消失。
        values, _ = module.paired_statistics(rows, changed, clusters)  # 原规则的差值方向为endpoint减ACDD。
        self.assertAlmostEqual(values['UTMOS']['delta_endpoint100_minus_acdd600'], 0.5)  # 高者为好，正差正确。
        self.assertAlmostEqual(values['CER_corpus_percent']['delta_endpoint100_minus_acdd600'], -2)  # CER改善2个百分点应为负差。


if __name__ == '__main__':  # 显式命令行运行合成数字测试。
    unittest.main()  # 不用GPU，不修改现有结果。
