"""仅在租用服务器用CPU小张量检查交叉诊断；不执行GPU主入口。"""

import ast  # 检查新脚本没有隐藏的训练、优化器更新或模型保存调用。
import importlib.util  # 只导入同目录冻结诊断，不调用main。
import json  # 读取固定开发清单，核对实际训练批次的排除规则。
import random  # 独立重建原始shuffle次序。
import sys  # 查找上传到租用服务器的诊断脚本。
import unittest  # 小型回归测试无需额外测试框架。
from pathlib import Path  # 所有输入都是现有文件，测试不写项目模型。
from types import SimpleNamespace  # 只伪造statvfs返回值，验证实际磁盘门槛。
from unittest.mock import patch  # 磁盘测试不创建大文件、不更改真实空间。

import torch  # 环境变量CUDA_VISIBLE_DEVICES为空，所有测试张量在CPU。

TOOLS = Path('/root/FsatVoiceGrad/exp/diagnostics/student_performance_tools_20260906')
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('fm_crossed_probe', TOOLS / 'inspect_acdd_fm_crossed_v1.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
from hifigan.custom.mrd import SpectrogramDiscriminator, STFTResolution  # 用实际MRD分支，不用零输出替身。
from fastvoicegrad.objectives import feature_matching_loss, least_squares_discriminator_loss, least_squares_student_loss


class CrossedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)  # 小型CPU回归不干扰其他任务，也不会触碰CUDA。

    def frozen_linear(self):
        model = torch.nn.Linear(2, 2)  # 非零小权重检验加载与只读指纹，不构造大模型。
        for value in model.parameters():
            value.requires_grad_(False)
        model.eval()
        return model

    def test_known_matrix_effects(self):
        measured = probe.matrix_effects({'S2000_D2000': 1., 'S3107_D2000': 3.,
                                        'S2000_D3107': 6., 'S3107_D3107': 10.})
        self.assertEqual(measured, {'student_change_at_D2000': 2., 'student_change_at_D3107': 4.,
            'D_change_at_S2000': 5., 'D_change_at_S3107': 7., 'interaction': 2.,
            'moving_diagonal_change': 9., 'average_student_marginal': 3., 'average_D_marginal': 6.})

    def test_layer_reduction_weight_and_relative_label(self):
        real = torch.tensor([[1., 3.], [2., 6.]])
        fake = torch.tensor([[0., 1.], [4., 2.]])
        row = probe.layer_measurement(real, fake, False)
        self.assertEqual(row['fm_raw'], 2.25)  # 四个绝对差1、2、2、4取平均。
        self.assertEqual(row['fm_weighted'], 4.5)  # FM的2只乘一次。
        self.assertEqual(row['per_sample_fm_raw'], [1.5, 3.])
        self.assertAlmostEqual(row['relative_l1_descriptive_only'], 2.25 / (3 + 1e-8))
        self.assertEqual(row['kind'], 'hidden_feature')

    def test_actual_mrd_agrees_with_production_terms(self):
        torch.manual_seed(301)  # 固定非零真实MRD卷积权重，避免不同次测试换一把尺。
        branch = SpectrogramDiscriminator(STFTResolution(512, 50, 240)).eval()
        for parameter in branch.parameters():
            parameter.requires_grad_(False)
        before = probe.fingerprint_model(branch)
        with torch.no_grad():
            real_wave = torch.randn(2, 1, 2048) * .1
            fake_wave = real_wave * .83 + torch.randn_like(real_wave) * .03
            real_score, real_features = branch(real_wave)
            fake_score, fake_features = branch(fake_wave)
            row = probe.branch_measurement(real_score, real_features, fake_score, fake_features)
            self.assertEqual(len(row['layers']), 6)  # 五层中间特征和一层最终分数全部保留。
            self.assertEqual([item['kind'] for item in row['layers']], ['hidden_feature'] * 5 + ['final_score'])
            self.assertAlmostEqual(row['fm_raw'], float(feature_matching_loss([real_features], [fake_features])), places=6)
            self.assertAlmostEqual(row['student_adv'], float(least_squares_student_loss([fake_score])), places=7)
            self.assertAlmostEqual(row['discriminator_loss_same_state'], float(least_squares_discriminator_loss([real_score], [fake_score])), places=7)
            self.assertAlmostEqual(row['fm_raw'], row['fm_hidden_raw'] + row['fm_score_raw'])
        probe.assert_unchanged(branch, before)  # 包括非持久Hann窗口，不能只查state_dict。

    def test_reject_zeroed_weight(self):
        model = self.frozen_linear()
        model.weight.zero_()  # 只清空CPU测试模型，证明诊断会拒绝这种伪低损失。
        with self.assertRaisesRegex(RuntimeError, 'zeroed'):
            probe.assert_readonly_model(model)

    def test_changed_parameter_or_buffer_is_detected(self):
        for change in ('parameter', 'buffer'):
            model = self.frozen_linear()
            model.register_buffer('fixed_window', torch.ones(2), persistent=False)
            before = probe.fingerprint_model(model)
            (model.weight if change == 'parameter' else model.fixed_window).add_(.1)
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                probe.assert_unchanged(model, before)

    def test_exact_loading_and_malformed_state(self):
        model = self.frozen_linear()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        before = probe.load_exact_weights(model, state)
        probe.assert_unchanged(model, before)
        with self.assertRaises(RuntimeError):
            probe.load_exact_weights(model, {'weight': state['weight']})  # 缺失bias不能宽松加载。

    def test_reject_gradients_and_training_submodule(self):
        model = self.frozen_linear()
        model.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'eval, frozen'):
            probe.assert_readonly_model(model)
        model.weight.requires_grad_(False)
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(RuntimeError, 'gradients'):
            probe.assert_readonly_model(model)
        model.weight.grad = None
        model.add_module('unexpected_active_child', torch.nn.Dropout())
        with self.assertRaisesRegex(RuntimeError, 'eval, frozen'):
            probe.assert_readonly_model(model)  # 父对象eval但新子层仍train也拒绝。

    def test_bad_shapes_scores_and_nonfinite_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'broadcasting'):
            probe.layer_measurement(torch.zeros(2, 2), torch.zeros(1, 2), False)
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            probe.statistics(torch.tensor([float('nan')]))
        with self.assertRaisesRegex(RuntimeError, 'exact output'):
            probe.branch_measurement(torch.zeros(2, 2), [torch.ones(2, 2)], torch.zeros(2, 2), [torch.zeros(2, 2)])

    def test_production_noise_draw_order_on_cpu(self):
        mel = torch.empty(32, 3, 7)  # CPU验证抽样先后；GPU随机数本身由主线程诊断核对SHA。
        source, score, times = probe.noise_inputs(mel, torch.device('cpu'))
        torch.manual_seed(204342)
        self.assertTrue(torch.equal(source, torch.randn_like(mel)))
        self.assertTrue(torch.equal(score, torch.randn_like(mel)))
        self.assertTrue(torch.equal(times, torch.randint(1, 1001, (32,), dtype=torch.long)))
        self.assertEqual(probe.CROP_SEED, 1234 + 100000 + 3108)
        self.assertEqual(probe.NOISE_SEED, 1234 + 200000 + 3108)

    def test_actual_retained_batch_plan_3108(self):
        development = json.loads((probe.DEV / 'development_contract.json').read_text())
        retained = development['retained_train_indices']
        self.assertEqual(len(retained), 39335)
        self.assertEqual(len(development['excluded_train_indices']), 52)
        expected = list(retained)
        random.Random(1234 + 2).shuffle(expected)  # 3108在第2个从零编号的shuffle轮次。
        offset = ((3108 - 1) % 1229) * 32
        actual = probe.resume.GlobalBatchPlan(retained).for_step(3108)
        self.assertEqual(actual, expected[offset:offset + 32])
        self.assertEqual(len(actual), 32)
        self.assertFalse(set(actual) & set(development['excluded_train_indices']))

    def test_source_has_no_update_or_model_save_calls(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        prohibited = {'step', 'backward', 'train_batch', 'restore_training_state', 'save_checkpoint', 'Adam', 'AdamW'}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
            self.assertNotIn(name, prohibited)
            self.assertFalse(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                             and node.func.value.id == 'torch' and node.func.attr == 'save')

    def test_disk_capacity_threshold_preserved(self):
        threshold = 3 * 1024**3 + 64 * 1024**2
        with patch.object(probe.os, 'statvfs', return_value=SimpleNamespace(f_bavail=threshold - 1, f_frsize=1)):
            with self.assertRaisesRegex(RuntimeError, 'Preserve3GiB'):
                probe.require_output_capacity()
        with patch.object(probe.os, 'statvfs', return_value=SimpleNamespace(f_bavail=threshold, f_frsize=1)):
            probe.require_output_capacity()  # 恰达到门槛可通过，绝不在测试中调低生产门槛。

    def test_gpu_capacity_precheck_and_hard_300_second_limit(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        checks = [node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == 'require_output_capacity']
        device_line = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == 'device')
        self.assertEqual(len(checks), 2)
        self.assertLess(min(node.lineno for node in checks), device_line)
        comparisons = [node for node in ast.walk(main) if isinstance(node, ast.Compare)
                       and any(isinstance(value, ast.Attribute) and value.attr == 'max_seconds' for value in node.comparators)]
        self.assertTrue(any(isinstance(node.comparators[-1], ast.Constant) and node.comparators[-1].value == 300
                            for node in comparisons))


if __name__ == '__main__':
    unittest.main()  # 只运行上面的小型CPU测试，绝不调用probe.main。
