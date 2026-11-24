"""Matpool CPU核验字符操作路径，不调用ASR或其他模型。"""

import importlib.util  # 从显式服务器路径导入待测函数。
import sys  # 测试脚本第一个参数为已上传的分析脚本。
import unittest  # 标准库合成文字测试。
from pathlib import Path  # 限定项目运行环境。

if not sys.platform.startswith('linux') or not Path('/root/FsatVoiceGrad').is_dir():  # 禁止本地项目测试。
    raise RuntimeError('Run only on Matpool.')  # 此限制同样适用于轻量诊断。
spec = importlib.util.spec_from_file_location('_transcript_alignment_test', Path(sys.argv.pop(1)).resolve())  # 私有模块名。
module = importlib.util.module_from_spec(spec)  # 不影响现有评价模块。
spec.loader.exec_module(module)  # 真实机器和辅助代码哈希保护仍会执行。


class AlignmentTests(unittest.TestCase):  # 使用人工可核算的短字符序列。
    def test_operations_sum_to_original_distance(self):  # 每个操作都应贡献一个字符错误。
        for reference, hypothesis, expected in [('ABC', 'ADC', 1), ('ABC', 'AB', 1), ('AB', 'ABC', 1), ('ABC', '', 3), ('', 'ABC', 3), ('A B', 'AB', 1)]:  # 包括空识别与空格删除。
            with self.subTest(reference=reference, hypothesis=hypothesis):  # 显示具体失败样例。
                result = module.align_edits(reference, hypothesis)  # 不使用库内置结果替代待测路径。
                self.assertEqual(result['distance'], expected)  # 总CER定义正确。
                self.assertEqual(len(result['edits']), expected)  # 保存路径长度必须一致。

    def test_tie_is_marked_and_reproducible(self):  # 重复字符可导致不唯一删除位置。
        result = module.align_edits('AA', 'A')  # 删除第一或第二个A同样最优。
        self.assertTrue(result['ambiguous_optimal_alignment'])  # 明确标记歧义。
        self.assertEqual(result, module.align_edits('AA', 'A'))  # 固定规则下重复运行同路径。

    def test_word_and_character_paths_are_separate(self):  # 词级只能辅助描述，不能替代CER分母。
        words = module.align_edits(['REAL', 'REWARD'], ['REAL', 'WERE', 'BROAD'])  # 一个词变成两个词的现象。
        self.assertEqual(words['distance'], 2)  # 词级两次编辑。
        chars = module.align_edits('REAL REWARD', 'REAL WERE BROAD')  # 字符级另按完整字符串计算。
        self.assertEqual(chars['distance'], module.base.edit_distance('REAL REWARD', 'REAL WERE BROAD'))  # 必须与原CER算法相同。

    def test_position_is_word_based_not_audio_time(self):  # 句首末只按文字首词/末词定义。
        self.assertEqual(module.locate_character_edit('ONE TWO THREE', {'reference_index': 0})['zone'], 'first_word')  # 首字符。
        self.assertEqual(module.locate_character_edit('ONE TWO THREE', {'reference_index': 5})['zone'], 'interior')  # 中间词。
        self.assertEqual(module.locate_character_edit('ONE TWO THREE', {'reference_index': 13})['zone'], 'last_word')  # 句尾插入间隙。


if __name__ == '__main__':  # 显式命令行启动合成字符串测试。
    unittest.main()  # 纯CPU，不改任何指标或模型。
