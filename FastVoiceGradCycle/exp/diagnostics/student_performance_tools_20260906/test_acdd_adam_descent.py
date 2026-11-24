# 只在租用CPU验证新诊断代码；其中torchAdam更新仅作用于随机生成的几个参数，不读真实权重。
"""CPU tests for Adam arithmetic, fixed-target scoring and reversible probes."""

import ast  # 检查实际GPU入口没有optimizer或权重文件写入。
import copy  # 为参考torchAdam保留独立原始状态。
import os  # 明确遮蔽GPU。
import unittest
from pathlib import Path

if os.name != 'posix' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("Execute only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")

import numpy as np
import torch
from torch import nn
import inspect_acdd_adam_descent_v1 as probe


class TinyState(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.randn(5))
        self.second = nn.Parameter(torch.randn(3))
        self.skipped = nn.Parameter(torch.randn(2))
        self.register_buffer('fixed_buffer', torch.tensor([2.0, 3.0]))


class TinyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 3, 1)
    def forward(self, wave):
        hidden = torch.tanh(self.conv(wave))
        score = hidden.mean(1, keepdim=True)
        return score, [hidden, score]


class TinyD(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([TinyBranch(), TinyBranch()])
    def loss_branches(self):
        return tuple(self.branches)


class TinyTeacher(nn.Module):
    def forward(self, x, t, speaker, bnf):
        return -x  # 仅用于反例：固定网络的输出仍会随学生输入变化。


class DescentProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_float32_adam_matches_actual_optimizer2000_and3107_including_none(self):
        torch.manual_seed(202)
        model = TinyState()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.9))
        snapshots = {}
        for step in range(1, 3108):
            for parameter in model.parameters():
                parameter.grad = torch.randn_like(parameter)
            optimizer.step()  # CPU小模型真实累计，不手填2000/3107计数。
            optimizer.zero_grad(set_to_none=True)
            if step in (2000, 3107):
                snapshots[step] = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
        for step, (weights, saved) in snapshots.items():
            with self.subTest(step=step):
                model.load_state_dict(weights)
                gradient = {'first': torch.randn(5), 'second': torch.zeros(3), 'skipped': None}
                before = probe.tree_sha(saved)
                candidate, delta = probe.adam_candidate(list(model.named_parameters()), gradient, saved, step)
                self.assertEqual(probe.tree_sha(saved), before)
                reference = TinyState()
                reference.load_state_dict(weights)
                real_adam = torch.optim.Adam(reference.parameters(), lr=0.0002, betas=(0.5, 0.9))
                real_adam.load_state_dict(copy.deepcopy(saved))
                for name, parameter in reference.named_parameters():
                    parameter.grad = None if gradient[name] is None else gradient[name].clone()
                real_adam.step()
                for name, parameter in reference.named_parameters():
                    torch.testing.assert_close(candidate[name], parameter, atol=0, rtol=0)  # float32逐bit一致。
                    if gradient[name] is None:
                        self.assertIsNone(delta[name])
                        torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)
                    else:
                        torch.testing.assert_close(delta[name], parameter.detach() - weights[name], atol=0, rtol=0)
                self.assertFalse(torch.equal(candidate['second'], weights['second']))  # 零梯度仍有历史动量；不能当None跳过。

    def test_adam_rejects_wrong_counter_or_configuration(self):
        model = TinyState()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.9))
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()
        saved = optimizer.state_dict()
        banks = {name: torch.ones_like(parameter) for name, parameter in model.named_parameters()}
        with self.assertRaises(RuntimeError):
            probe.adam_candidate(list(model.named_parameters()), banks, saved, 2000)
        wrong = copy.deepcopy(saved)
        wrong['param_groups'][0]['lr'] = 0.0001
        with self.assertRaises(RuntimeError):
            probe.adam_candidate(list(model.named_parameters()), banks, wrong, 1)

    def test_temporary_trial_restores_parameters_and_buffers_on_all_scales_and_error(self):
        model = TinyState()
        original = copy.deepcopy(model.state_dict())
        candidate = {name: p.detach().clone() + 0.25 for name, p in model.named_parameters()}
        delta = {name: candidate[name] - original[name] for name in candidate}
        expected = probe.tree_sha(original)
        for scale in probe.SCALES:
            with probe.trial_state(model, original, candidate, delta, scale):
                self.assertNotEqual(probe.tree_sha(model.state_dict()), expected)
                torch.testing.assert_close(model.fixed_buffer, original['fixed_buffer'], atol=0, rtol=0)
            self.assertEqual(probe.tree_sha(model.state_dict()), expected)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with probe.trial_state(model, original, candidate, delta, 0.5):
                model.fixed_buffer.add_(5)  # 模拟诊断中buffer意外变化，finally也必须恢复。
                raise RuntimeError('injected failure during trial')
        self.assertEqual(probe.tree_sha(model.state_dict()), expected)
        self.assertEqual(probe.tree_sha(original), expected)

    def test_serial_wave_and_parameter_components_match_full_graph(self):
        torch.manual_seed(35)
        student = nn.Conv1d(80, 80, 1)
        vocoder, discriminator = nn.Conv1d(80, 1, 1), TinyD()
        for model in (vocoder, discriminator):
            probe.training.set_requires_grad(model, False)
        x0, source = torch.randn(2, 80, 8), torch.randn(2, 80, 8)
        mean, std = torch.zeros(1, 80, 1), torch.full((1, 80, 1), 1.7)
        prediction = student(source)
        leaf = prediction.detach().requires_grad_(True)
        values, adv_y, fm_y = probe.wave_output_gradients(leaf, x0, vocoder, discriminator, mean, std, lambda: None)
        fixed_target = torch.randn_like(leaf)
        score_y = (leaf.detach() - fixed_target).sign() / leaf.numel()
        gt_y = (leaf.detach() - x0).sign() / leaf.numel()
        banks = probe.parameter_banks(prediction, list(student.named_parameters()),
            {'adv': adv_y, '2fm': 2 * fm_y, '45score': 45 * score_y, 'gt_l1': gt_y}, lambda: None)
        direct_prediction = student(source)
        fake = vocoder(probe.training.denormalize_mel(direct_prediction, mean, std))
        with torch.no_grad():
            real = vocoder(probe.training.denormalize_mel(x0, mean, std))
        adversarial, matching = 0, 0
        for branch in discriminator.loss_branches():
            with torch.no_grad():
                _, real_features = branch(real)
            score, features = branch(fake)
            adversarial = adversarial + probe.training.least_squares_student_loss([score])
            matching = matching + probe.training.feature_matching_loss([real_features], [features])
        self.assertAlmostEqual(values['adv'], float(adversarial), places=6)
        self.assertAlmostEqual(values['fm_raw'], float(matching), places=6)
        objectives = {'adv': adversarial, '2fm': 2 * matching,
                      '45score': 45 * (direct_prediction - fixed_target).abs().mean(),
                      'gt_l1': (direct_prediction - x0).abs().mean()}
        objectives['total'] = sum(objectives[key] for key in ('adv', '2fm', '45score'))
        max_difference = 0.0
        for label, value in objectives.items():
            grads = torch.autograd.grad(value, tuple(student.parameters()), retain_graph=True)
            for (name, _), direct in zip(student.named_parameters(), grads):
                max_difference = max(max_difference, float((banks[label][name] - direct).abs().max()))
                torch.testing.assert_close(banks[label][name], direct, rtol=2e-5, atol=2e-6)
        print('CPU_SERIAL_FULL_GRADIENT_MAX_ABSDIFF', max_difference)
        self.assertTrue(all(p.grad is None for model in (student, vocoder, discriminator) for p in model.parameters()))

    def test_frozen_teacher_target_can_fall_while_refreshed_value_rises(self):
        torch.manual_seed(314)
        teacher, diffusion = TinyTeacher(), probe.base.VoiceGradDiffusion(1000, 0.008)
        vocoder, discriminator = nn.Conv1d(80, 1, 1), TinyD()
        for model in (vocoder, discriminator):
            probe.training.set_requires_grad(model, False)
        y = torch.ones(2, 80, 8)
        reference, t, noise = torch.zeros_like(y), torch.tensor([600, 600]), torch.full_like(y, 0.5)
        speaker, bnf = torch.zeros(2, 256), torch.zeros(2, 144, 8)
        config = probe.base.PAPER_STUDENT_CONFIG
        leaf = y.clone().requires_grad_(True)
        score, target = probe.training.score_distillation_loss(leaf, teacher, diffusion, t, noise, speaker, bnf,
            target_mode='clean_x0', weight_mode='sqrt_alpha_bar')
        derivative = probe.checked_grad(score, leaf)
        self.assertTrue(bool((derivative < 0).all()))
        self.assertFalse(target.requires_grad)
        mean, std = torch.zeros(1, 80, 1), torch.ones(1, 80, 1)
        before = probe.score_readout(y, reference, target, teacher, diffusion, t, noise, speaker, bnf,
            vocoder, discriminator, mean, std, config)
        after = probe.score_readout(y + 0.01, reference, target, teacher, diffusion, t, noise, speaker, bnf,
            vocoder, discriminator, mean, std, config)
        self.assertAlmostEqual(before['primary_moving_score'], before['diagnostic_frozen_target_score'])
        self.assertLess(after['diagnostic_frozen_target_score'], before['diagnostic_frozen_target_score'])
        self.assertGreater(after['primary_moving_score'], before['primary_moving_score'])
        self.assertGreater(after['teacher_target_weighted_l1_drift'], 0)
        self.assertEqual(before['teacher_target_weighted_l1_drift'], 0)

    def test_statistics_preserve_none_and_delta_sign(self):
        values = {'a': {'p': torch.tensor([1., 2.]), 'q': None}, 'b': {'p': torch.tensor([-1., -2.]), 'q': None}}
        result = probe.gradient_statistics(values, {'p': torch.tensor([-0.1, -0.2]), 'q': None})
        self.assertAlmostEqual(result['cosines']['a']['b'], -1.)
        self.assertLess(result['dot_products']['a']['adam_delta'], 0)
        self.assertEqual(result['none_parameter_names']['a'], ['q'])

    def test_real_batch32_float32_score_gradient_reports_strict_rounding_tolerance(self):
        torch.manual_seed(414)
        y = torch.randn(32, 80, 128, dtype=torch.float32, requires_grad=True)
        noise = torch.randn_like(y)
        diffusion = probe.base.VoiceGradDiffusion(1000, 0.008)
        t = torch.linspace(1, 1000, 32).long()
        score, target = probe.training.score_distillation_loss(y, TinyTeacher(), diffusion, t, noise,
            torch.zeros(32, 256), torch.zeros(32, 144, 128), target_mode='clean_x0', weight_mode='sqrt_alpha_bar')
        actual = probe.checked_grad(score, y)
        analytic = diffusion.sqrt_alphas_cumprod[t].reshape(-1, 1, 1) * (y.detach() - target).sign() / y.numel()
        agreement = probe.validate_score_gradient(actual, analytic)
        self.assertLessEqual(agreement['max_absolute_difference'], 1e-12)
        self.assertLessEqual(agreement['max_relative_difference_on_nonzero_reference'], 1e-6)
        with self.assertRaises(RuntimeError):
            probe.validate_score_gradient(-actual, analytic)  # 错误符号远超舍入容差，仍应明确拒绝。

    def test_fixed32_replay_requires_archived_values_not_only_input_identity(self):
        value = {'mel_l1': .20179936289787292, 'prediction_rms': 1.003182053565979,
                 'reference_rms': 1.0232422351837158, 'cosine_reference': .9609888792037964}
        self.assertEqual(max(probe.validate_development_replay(value, value).values()), 0.0)
        for wrong in (float('nan'), value['mel_l1'] + 1e-5):
            changed = dict(value, mel_l1=wrong)
            with self.assertRaises(RuntimeError):
                probe.validate_development_replay(changed, value)

    def test_numpy_rng_digest_covers_middle_array_elements(self):
        value = np.arange(10000)
        changed = value.copy()
        changed[5000] += 1
        self.assertNotEqual(probe.tree_sha(value), probe.tree_sha(changed))  # repr通常会省略中间项，字节SHA必须检测到。

    def test_main_never_constructs_optimizer_and_only_saves_fixed_inputs(self):
        tree = ast.parse(Path(probe.__file__).read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr in ('Adam', 'step', 'train_batch') for node in calls))
        saves = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'atomic_torch_save']
        self.assertEqual(len(saves), 1)
        self.assertIsInstance(saves[0].args[0], ast.DictComp)
        self.assertEqual(saves[0].args[1].right.value, 'fixed_inputs.pt')
        self.assertEqual(probe.SCALES, (0.25, 0.5, 1.0))
        self.assertLessEqual(probe.MAX_OUTPUT_BYTES, 100 * 1024**2)


if __name__ == '__main__':
    unittest.main()  # 该入口只运行租用CPU小模型测试，不执行任何GPU诊断。
