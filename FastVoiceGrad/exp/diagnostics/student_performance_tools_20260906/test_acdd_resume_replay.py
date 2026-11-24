# 在租用服务器CPU上验证恢复两套Adam后的逐步等价，不加载真实S/D或启动GPU训练。
"""Uninterrupted versus step-600 resume checks, including D, Adam and global seeds."""

import ast  # 从实际续训驱动提取保存函数，检验真正将被调用的payload构造。
import copy  # 保留不中断分支的最终状态，避免后续测试修改参照对象。
import json  # 核对真实写入的最新完整检查点指针。
import random  # 生成原全局步数对应的打乱和裁剪序列。
import tempfile  # 测试产物仅位于租用服务器/tmp的独立目录。
import unittest  # 以独立测试报告等价性、拒绝错误父状态和保护阈值。
from pathlib import Path  # 定位测试自己的临时归档。

import numpy as np  # 核对完整NumPy随机状态的恢复。
import torch  # 全部小网络与Adam均在CPU运行。
from torch import nn  # 两个很小的网络模拟学生和判别器。
import resume_acdd_development as resume  # 导入不会调用GPU主入口。


def make_models():  # 小模型也使用weight_norm，检查真实参数登记顺序及双Adam恢复。
    student = nn.Sequential(nn.utils.weight_norm(nn.Linear(3, 4)), nn.Tanh(), nn.utils.weight_norm(nn.Linear(4, 2)))
    discriminator = nn.Sequential(nn.utils.weight_norm(nn.Linear(2, 4)), nn.Tanh(), nn.utils.weight_norm(nn.Linear(4, 1)))
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, betas=(0.5, 0.9))  # 学生独立Adam。
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))  # D的动量绝不与学生共享。
    return student, discriminator, optimizer, d_optimizer  # 没有真实教师、声码器或数据。


def toy_step(models, plan, step):  # 按真实顺序执行一次D更新、再一次学生更新。
    student, discriminator, optimizer, d_optimizer = models
    indices = plan.for_step(step)  # 从绝对步数取得当前批次，不依赖新进程是否从1开始。
    crop = random.Random(1234 + 100000 + step).randint(0, 50)  # 模拟原全局裁剪seed。
    base = torch.tensor([[index / 71.0, crop / 50.0, 0.2] for index in indices])  # 确定性的假输入。
    resume.baseline.training.seed_everything(1234 + 200000 + step)  # 原全局噪声seed。
    source_noise = torch.randn_like(base)  # 第一次随机抽样对应学生起点噪声。
    score_noise = torch.randn_like(base)  # 第二次对应独立教师分数噪声。
    timesteps = torch.randint(1, 1001, (len(indices),))  # 第三次对应每例随机扩散时刻。
    inputs = base + 0.05 * source_noise  # 小网络的有噪输入。
    target = 0.5 * base[:, :2] + 0.001 * score_noise[:, :2]  # 固定批次目标，只有微小独立噪声。
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)  # D阶段允许D求导。
    d_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake = student(inputs)  # D的生成样本不向学生反传。
    d_loss = (discriminator(target) - 1).square().mean() + discriminator(fake).square().mean()
    d_loss.backward()
    d_optimizer.step()  # 第一次真实CPU优化器更新只属于假判别器。
    d_optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)  # 学生阶段冻结D参数，但保留输入梯度。
    optimizer.zero_grad(set_to_none=True)
    prediction = student(inputs)
    adversarial = (discriminator(prediction) - 1).square().mean()
    real_feature = discriminator[0](target).detach()  # 固定目标特征不能反传。
    fm = (discriminator[0](prediction) - real_feature).abs().mean()
    dist = (prediction - target).abs().mean()
    loss = adversarial + 2 * fm + 45 * dist  # toy保留D/S顺序和原2/45尺度，而非验证真实声学目标。
    loss.backward()
    optimizer.step()  # 第二次真实CPU更新只属于假学生。
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)  # 与原train_batch批次末端一样恢复D状态。
    return {'step': step, 'indices': indices, 'crop': crop, 'noise': source_noise.clone(),
            'timesteps': timesteps.clone(), 'student_loss': float(loss), 'd_loss': float(d_loss)}  # 逐步核对数据及更新。


def state_payload(models, step):  # 构造与生产恢复函数相同的完整字段，不省略D或优化器。
    student, discriminator, optimizer, d_optimizer = models
    return {'student': student.state_dict(), 'discriminator': discriminator.state_dict(),
            'student_optimizer': optimizer.state_dict(), 'discriminator_optimizer': d_optimizer.state_dict(),
            'global_step': step, 'rng_state': resume.baseline.training.capture_rng_state(),
            'run_contract_sha256': 'cpu_toy_parent', 'configuration': {},
            'development': {'mel_l1': 0.19, 'prediction_rms': 1.0}}  # 这些读数明确仅用于CPU假实验。


class ACDDResumeTests(unittest.TestCase):
    def equal_nested(self, actual, expected):  # 递归比较模型、Adam及RNG中的全部张量和值。
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)  # 数值、dtype、形状必须精确相同。
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)  # NumPy随机状态数组精确恢复。
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.equal_nested(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.equal_nested(value, reference)
        else:
            self.assertEqual(actual, expected)  # 学习率、beta、步数等标量也不能被重置。

    def test_true_population_midpass_and_nextpass_batch_identity(self):
        indices = list(range(39335))  # 使用真实保留样本数，因而每轮正好1229个batch32。
        uninterrupted = resume.GlobalBatchPlan(indices)
        recorded = {}  # 模拟不中断计划直到2000步，跨过真实1229/1230边界。
        for step in range(1, 2001):
            batch = uninterrupted.for_step(step)
            if step in (601, 1229, 1230, 2000):
                recorded[step] = batch
        resumed = resume.GlobalBatchPlan(indices)  # 新进程缓存为空，从601中途进入。
        self.assertEqual(resumed.batches_per_pass, 1229)
        for step in (601, 1229, 1230, 2000):
            self.assertEqual(resumed.for_step(step), recorded[step])  # 不得重播第1批或误用下一轮shuffle。

    def test_two_adam_models_and_rng_equal_uninterrupted_after_step600(self):
        torch.manual_seed(811)  # 初始假模型固定，不加载真实项目权重。
        models = make_models()
        plan = resume.GlobalBatchPlan(list(range(71)), batch_size=4)  # 小批量加快CPU测试，同时存在轮内恢复。
        for step in range(1, 601):
            toy_step(models, plan, step)  # 建立真实600次CPU Adam动量，而非伪造step字段。
        with tempfile.TemporaryDirectory(prefix='acdd_resume_cpu_', dir='/tmp') as directory:
            path = Path(directory) / 'full600.pt'
            resume.baseline.training.atomic_torch_save(state_payload(models, 600), path)  # 经过真实原子保存与磁盘往返。
            checkpoint = torch.load(path, map_location='cpu')
            expected_trace = [toy_step(models, plan, step) for step in range(601, 609)]  # 原对象不中断继续。
            expected_final = copy.deepcopy(state_payload(models, 608))  # 保留两个模型、两个Adam与所有RNG最终值。
            torch.manual_seed(999)  # 新进程先用不同随机值建模，防止碰巧与旧状态相同而漏检恢复。
            restored = make_models()
            resume.restore_training_state(checkpoint, *restored, 600, 'cpu_toy_parent', torch.device('cpu'))
            self.equal_nested(resume.baseline.training.capture_rng_state(), checkpoint['rng_state'])  # 重设每步seed前先单独验证RNG确实恢复。
            continued = resume.GlobalBatchPlan(list(range(71)), batch_size=4)  # 缓存为空，从601重建当前轮。
            actual_trace = [toy_step(restored, continued, step) for step in range(601, 609)]
            actual_final = state_payload(restored, 608)
            self.equal_nested(actual_trace, expected_trace)  # 数据索引、裁剪、噪声、t及两项loss逐步完全一致。
            self.equal_nested(actual_final, expected_final)  # S/D参数、双方moment/step/LR及RNG全部精确相同。

    def test_original_guard_is_not_rebased_to_parent(self):
        anchor = {'mel_l1': 0.19941353797912598, 'prediction_rms': 0.9994737505912781}  # 原S0的实际保护锚。
        resume.assert_guard({'mel_l1': 0.1909332424402237, 'prediction_rms': 1.0017690658569336}, anchor)  # 已保存S600合法。
        with self.assertRaises(RuntimeError):
            resume.assert_guard({'mel_l1': anchor['mel_l1'] * 1.10001, 'prediction_rms': 1.0}, anchor)
        with self.assertRaises(RuntimeError):
            resume.assert_guard({'mel_l1': 0.19, 'prediction_rms': anchor['prediction_rms'] * 0.79999}, anchor)

    def test_incomplete_parent_is_rejected_before_reinitializing_optimizers(self):
        models = make_models()  # 全新假模型，不允许以此冒充已经训练600步的状态。
        incomplete = {'student': models[0].state_dict(), 'global_step': 600}  # 对应错误地传入精简S600。
        with self.assertRaises(RuntimeError):
            resume.restore_training_state(incomplete, *models, 600, 'cpu_toy_parent', torch.device('cpu'))

    def test_actual_child_checkpoint_saves_both_models_and_adams(self):
        source = Path(resume.__file__).resolve()
        tree = ast.parse(source.read_text(encoding='utf-8'))  # 提取实际嵌套checkpoint函数，不执行main。
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint']
        self.assertEqual(len(functions), 1)
        student, discriminator, optimizer, d_optimizer = make_models()
        with tempfile.TemporaryDirectory(prefix='acdd_child_save_cpu_', dir='/tmp') as directory:
            output = Path(directory)
            namespace = {'baseline': resume.baseline, 'student': student, 'discriminator': discriminator,
                         'optimizer': optimizer, 'd_optimizer': d_optimizer, 'config': resume.baseline.PAPER_STUDENT_CONFIG,
                         'contract_sha': 'cpu-child', 'record': {'lineage': {'parent': 'cpu-only'}},
                         'anchor': {'mel_l1': 0.2, 'prediction_rms': 1.0}, 'output': output}
            module = ast.Module(body=functions, type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(source), 'exec'), namespace)  # 定义真实保存函数，不改写其payload代码。
            namespace['checkpoint'](0, {'mel_l1': 0.19, 'prediction_rms': 1.0})  # 只保存未更新的CPU假模型。
            path = output / 'full_step_000000.pt'
            saved = torch.load(path, map_location='cpu')
            for name in ('student', 'discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state', 'lineage'):
                self.assertIn(name, saved)  # 任何一个缺失都会破坏后续精确恢复。
            self.assertEqual(saved['global_step'], 0)
            pointer = json.loads((output / 'latest_full.json').read_text(encoding='utf-8'))
            self.assertEqual(pointer['checkpoint_sha256'], resume.baseline.sha256_file(path))  # 指针必须绑定实际完整文件。


if __name__ == '__main__':  # 所有CPU测试都由明确的远端命令调用。
    unittest.main()  # 本地Windows不能执行本文件。
