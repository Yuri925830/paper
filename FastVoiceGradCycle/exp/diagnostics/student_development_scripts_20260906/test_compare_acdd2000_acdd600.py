"""在Matpool用合成契约和分数测试2000/600比较，不加载模型。"""

import copy  # 为每个不合法契约案例创建独立副本。
import importlib.util  # 导入独立待测统计脚本。
import sys  # 第一个参数指定已上传的真实脚本位置。
import unittest  # 使用标准库测试框架。
from pathlib import Path  # 限定运行环境。

if not sys.platform.startswith('linux') or not Path('/root/FsatVoiceGrad').is_dir():  # 本地只编辑不执行测试。
    raise RuntimeError('Run these CPU tests only on Matpool.')  # 不允许错误服务器运行。
spec = importlib.util.spec_from_file_location('_test_acdd_checkpoint_comparison', Path(sys.argv.pop(1)).resolve())  # 私有模块名。
module = importlib.util.module_from_spec(spec)  # 不影响训练或现有评价模块。
spec.loader.exec_module(module)  # 真实机器身份及冻结helper哈希照常核验。


class ComparisonTests(unittest.TestCase):  # 全部输入均为合成数字和小契约。
    def contracts(self):  # 构造可人工核算的父子关系。
        parent = {'configuration': {'target': 'fixed'}, 'training_indices': [0, 2], 'excluded_train_indices': [1], 'reconstruction_dev_indices': [1]}  # 字段缩小但含义相同。
        child = {**copy.deepcopy(parent), 'kind': 'acdd_development_continuation', 'lineage': {'parent_contract_sha256': module.IDENTITIES[module.REFERENCE]['training_contract_sha256'], 'start_after_global_step': 600}, 'stop_global_step': 2000}  # 正确续训起点和终点。
        return parent, child  # 不读真实训练状态。

    def test_valid_lineage(self):  # 正确父子关系应通过。
        module.validate_lineage(*self.contracts())  # 不因契约schema更新误拒绝。

    def test_changed_configuration_is_rejected(self):  # 不把改目标后的实验冒充单纯续训。
        parent, child = self.contracts()  # 每次测试使用新数据。
        child['configuration']['target'] = 'changed'  # 仅修改训练目标字段。
        with self.assertRaises(ValueError):  # 应明确拒绝。
            module.validate_lineage(parent, child)  # 不放宽为只看步数。

    def test_wrong_parent_or_step_is_rejected(self):  # 父契约和600步起点必须同时满足。
        for key, wrong in [('parent_contract_sha256', 'wrong'), ('start_after_global_step', 0)]:  # 分别模拟错恢复和从头训练。
            parent, child = self.contracts()  # 两种错误互相独立。
            child['lineage'][key] = wrong  # 当前只改一个来源条件。
            with self.subTest(key=key), self.assertRaises(ValueError):  # 每个错误都应阻断。
                module.validate_lineage(parent, child)  # 不能仍标lineage_verified。

    def test_pairing_and_metric_direction(self):  # 确认2000减600的正负号没有写反。
        clusters = [(f'p{i}', f'u{i}') for i in range(10)]  # 合成10条独立源。
        rows = [{'source_speaker': speaker, 'source_utt': utterance, 'char_errors': 1, 'reference_chars': 50, 'verified': 1, 'mos_ovr': 3, 'utmos': 4, 'mos_sig': 3, 'mos_bak': 4, 'p808_mos': 3, 'cosine_score': 0.8} for speaker, utterance in clusters for _ in range(9)]  # 每源9目标。
        values, _ = module.paired_statistics(rows, rows, clusters)  # 完全相同模型的差应为0。
        for item in values.values():  # 所有指标的配对区间也应精确为0。
            self.assertEqual(item['cluster_bootstrap_95_percentile_ci'], [0, 0])  # 不采用独立两次抽样。
        changed = [{**row, 'char_errors': 0, 'utmos': 4.5} for row in rows]  # 2000步合成CER降低、UTMOS升高。
        values, _ = module.paired_statistics(rows, changed, clusters)  # 检查方向定义。
        self.assertAlmostEqual(values['CER_corpus_percent']['delta_acdd2000_minus_acdd600'], -2)  # CER改善为负百分点。
        self.assertAlmostEqual(values['UTMOS']['delta_acdd2000_minus_acdd600'], 0.5)  # 质量预测改善为正。


if __name__ == '__main__':  # 显式命令行运行纯CPU测试。
    unittest.main()  # 不读取真实metrics，不执行GPU模型。
