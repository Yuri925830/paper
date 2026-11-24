"""出租机CPU测试候选汇总的契约兼容性与同配对统计，不加载模型。"""

import copy  # 合成契约逐项修改，检查保护能否拒绝错误输入。
import importlib.util  # 私有导入独立候选汇总脚本。
import sys  # 第一个参数为远端待测脚本位置。
import unittest  # 只使用标准库单元测试。
from pathlib import Path  # 主机范围先于任何测试计算检查。

if not sys.platform.startswith('linux') or not Path('/root/FsatVoiceGrad').is_dir():  # 不允许在本地执行。
    raise RuntimeError('Run these CPU tests only on Matpool.')  # 测试也遵守服务器约束。
path = Path(sys.argv.pop(1)).resolve()  # 明确待测脚本，不从同名模块猜测。
spec = importlib.util.spec_from_file_location('_candidate_summary_test', path)  # 不覆盖公共模块。
candidate = importlib.util.module_from_spec(spec)  # 准备隔离测试命名空间。
spec.loader.exec_module(candidate)  # 同样执行真实机器指纹与辅助代码哈希保护。


class CandidateSummaryTests(unittest.TestCase):  # 全部使用人工构造的小契约，无语音读取。
    def setUp(self):  # 每个测试得到全新测试数据。
        self.dev = {'excluded_train_indices': [1, 3], 'retained_train_indices': [0, 2, 4], 'reconstruction_dev_indices': [3]}  # 缩小的合成划分。
        self.contract = {'formal_test_used_for_selection': False, 'excluded_train_indices': [1, 3], 'training_indices': [0, 2, 4], 'reconstruction_dev_indices': [3], 'development_contract_sha256': 'file-sha'}  # ACDD字段约定。

    def test_accepts_acdd_file_sha_field(self):  # ACDD旧字段保存文件SHA，不能误按canonical解释。
        candidate.validate_training_exclusion(self.contract, self.dev, 'file-sha')  # 正确来源应直接通过。

    def test_accepts_endpoint_explicit_file_sha(self):  # endpoint分开保存文件和canonical身份。
        self.contract.update(development_contract_file_sha256='file-sha', development_contract_sha256='canonical-sha')  # 新明确字段优先。
        candidate.validate_training_exclusion(self.contract, self.dev, 'file-sha')  # 两种训练runner使用同一开发文件。

    def test_rejects_missing_exclusion_or_changed_pool(self):  # 不能只靠声明声称没见开发语音。
        for key, wrong in [('excluded_train_indices', [1]), ('training_indices', [0, 1, 2, 4]), ('reconstruction_dev_indices', [1]), ('formal_test_used_for_selection', True)]:  # 分别模拟污染。
            changed = copy.deepcopy(self.contract)  # 原合成契约不受修改。
            changed[key] = wrong  # 当前测试只改一个条件。
            with self.subTest(key=key), self.assertRaises(ValueError):  # 每一种错误都应明确拒绝。
                candidate.validate_training_exclusion(changed, self.dev, 'file-sha')  # 不能回退到宽松判断。

    def test_rejects_wrong_development_file(self):  # 内容相似但文件身份不同也不能混用。
        with self.assertRaises(ValueError):  # 所有比较必须绑定冻结开发协议。
            candidate.validate_training_exclusion(self.contract, self.dev, 'another-sha')  # 错误来源必须停止。

    def test_canonical_sha_is_order_independent(self):  # 训练绑定内容级哈希，不依赖JSON字段书写次序。
        self.assertEqual(candidate.canonical_sha({'a': 1, 'b': 2}), candidate.canonical_sha({'b': 2, 'a': 1}))  # 相同内容得到相同指纹。
        self.assertNotEqual(candidate.canonical_sha({'a': 1}), candidate.canonical_sha({'a': 2}))  # 数值变化必然改变身份。


if __name__ == '__main__':  # 显式命令行运行纯CPU测试。
    unittest.main()  # 不访问GPU、不重评估、不改任何训练记录。
