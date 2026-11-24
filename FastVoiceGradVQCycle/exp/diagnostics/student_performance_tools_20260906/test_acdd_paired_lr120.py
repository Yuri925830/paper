# 只在租用机CPU运行；用小模型验证真实双Adam恢复、仅S学习率变化和异常保存边界。
"""CPU acceptance tests for the paired120 development experiment."""

import ast  # 检查训练唯一入口没有被固定监测目标替代。
import contextlib  # 捕获故障报告，检查恢复游标是否诚实。
import copy  # 保存父状态独立副本，避免测试自己污染基准。
import io  # 读取stdout中的实际失败步。
import json  # 检查真实保存收据。
import os  # CPU测试必须显式隐藏CUDA。
import random  # 验证Python随机流和独立批次计划。
import tempfile  # 所有测试文件只落在本测试创建的临时目录中。
import unittest
from pathlib import Path
from unittest import mock  # 假容量仅用于KB级测试文件，不改变生产门禁。

if os.name != 'posix' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("Execute only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
import numpy as np
import torch
from torch import nn
import run_acdd_paired_lr120_v1 as driver


def models():
    s, d = nn.Linear(3, 2), nn.Linear(2, 1)
    return s, d, torch.optim.Adam(s.parameters(), lr=0.0002, betas=(0.5, 0.9)), torch.optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.9))


def update(items):
    s, d, so, do = items
    random.random(), np.random.random()
    source, target = torch.randn(4, 3), torch.randn(4, 2)
    driver.training.set_requires_grad(d, True)
    do.zero_grad(set_to_none=True)
    dloss = (d(target) - 1).square().mean() + d(s(source).detach()).square().mean()
    dloss.backward()
    do.step()
    do.zero_grad(set_to_none=True)
    driver.training.set_requires_grad(d, False)
    so.zero_grad(set_to_none=True)
    prediction = s(source)
    loss = (prediction - target).abs().mean() + (d(prediction) - 1).square().mean()
    before = float(loss.detach())
    loss.backward()
    so.step()
    so.zero_grad(set_to_none=True)
    driver.training.set_requires_grad(d, True)
    return before, float(dloss.detach())  # 两组首步返回pre-S-update值，应完全相同。


def snapshot(items):
    s, d, so, do = items
    return copy.deepcopy({'student': s.state_dict(), 'discriminator': d.state_dict(),
        'student_optimizer': so.state_dict(), 'discriminator_optimizer': do.state_dict(),
        'rng_state': driver.training.capture_rng_state(), 'global_step': driver.START,
        'run_contract_sha256': driver.PARENT_CONTRACT,
        'configuration': driver.base.replace(driver.base.PAPER_STUDENT_CONFIG, num_workers=0).as_dict(),
        'development': {'mel_l1': 0.19941353797912598, 'prediction_rms': 0.9994737505912781}})


class Paired120Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        driver.training.seed_everything(177)
        cls.initial = models()
        for _ in range(driver.START):
            update(cls.initial)  # 真实执行3107次小双Adam，所有moments和计数不是手工伪造。
        cls.parent = snapshot(cls.initial)

    def setUp(self):
        space = mock.patch.object(driver, 'available_bytes', return_value=20 * 1024**3)
        space.start()
        self.addCleanup(space.stop)

    def restored(self, lr):
        items = models()
        driver.restore_arm(self.parent, *items, lr, torch.device('cpu'))
        return items

    def test_real_dual_adam_restore_only_student_lr_and_identical_first_d(self):
        parent_hash = driver.probe.tree_sha(self.parent)
        control = self.restored(0.0002)
        c_loss = update(control)
        c_rng = driver.probe.tree_sha(driver.training.capture_rng_state())
        half = self.restored(0.0001)
        h_loss = update(half)
        self.assertEqual(c_loss, h_loss)
        self.assertEqual(c_rng, driver.probe.tree_sha(driver.training.capture_rng_state()))
        self.assertEqual(driver.model_sha(control[1]), driver.model_sha(half[1]))
        self.assertEqual(driver.probe.tree_sha(control[3].state_dict()), driver.probe.tree_sha(half[3].state_dict()))
        self.assertEqual(driver.probe.tree_sha(control[2].state_dict()['state']), driver.probe.tree_sha(half[2].state_dict()['state']))
        self.assertNotEqual(driver.model_sha(control[0]), driver.model_sha(half[0]))
        for (name, original), (_, c), (_, h) in zip(self.parent['student'].items(), control[0].state_dict().items(), half[0].state_dict().items()):
            torch.testing.assert_close(h - original, 0.5 * (c - original), rtol=0, atol=2e-7)
        driver.validate_effective_adam(control[2], control[3], control[0], control[1], 0.0002, 3108)
        driver.validate_effective_adam(half[2], half[3], half[0], half[1], 0.0001, 3108)
        self.assertEqual(parent_hash, driver.probe.tree_sha(self.parent))

    def test_uninterrupted_control_matches_cold_restore(self):
        driver.training.restore_rng_state(self.parent['rng_state'], torch.device('cpu'))
        update(self.initial)
        expected = snapshot(self.initial)
        restored = self.restored(0.0002)
        update(restored)
        actual = snapshot(restored)
        self.assertEqual(driver.probe.tree_sha(expected), driver.probe.tree_sha(actual))
        # 恢复类共享模型，避免其他测试因执行顺序不同而误用3108状态。
        driver.restore_arm(self.parent, *self.initial, 0.0002, torch.device('cpu'))

    def test_wrong_parent_and_adam_override_rejected(self):
        for change in ('step', 'contract', 'configuration', 'd_lr'):
            wrong = copy.deepcopy(self.parent)
            if change == 'step': wrong['global_step'] = 3106
            if change == 'contract': wrong['run_contract_sha256'] = 'wrong'
            if change == 'configuration': wrong['configuration']['lambda_dist'] = 1
            if change == 'd_lr': wrong['discriminator_optimizer']['param_groups'][0]['lr'] = 0.0001
            with self.assertRaises(RuntimeError):
                driver.restore_arm(wrong, *models(), 0.0001, torch.device('cpu'))
        with self.assertRaises(RuntimeError):
            self.restored(0.00005)

    def test_three_complete_window_stop_boundary(self):
        rows = []
        windows = []
        for n in range(1, 61):
            value = (1.0, 1.01, 1.02)[(n - 1) // 20]
            rows.append({'global_step': 3107 + n, 'updates': n, **{key: value for key in driver.LOSS_KEYS}})
            if n % 20 == 0:
                windows.append(driver.window_record(rows[-20:]))
            if n < 60:
                self.assertFalse(driver.moving_rise(windows))
        self.assertTrue(driver.moving_rise(windows))
        self.assertFalse(driver.moving_rise([{'means': {'student_total': v}} for v in (1, 1, 1.03)]))
        self.assertFalse(driver.moving_rise([{'means': {'student_total': v}} for v in (1, 1.005, 1.019)]))
        with self.assertRaises(RuntimeError):
            driver.window_record(rows[-19:])

    def test_development_includes_zero_and_threshold_does_not_rebase(self):
        records = [{'updates': n, 'development': {'mel_l1': v}} for n, v in zip((0, 20, 40), (1.0, 1.004, 1.01))]
        self.assertFalse(driver.development_rise(records[:2]))
        self.assertTrue(driver.development_rise(records))
        self.assertFalse(driver.development_rise([{'development': {'mel_l1': v}} for v in (1, .98, .99)]))
        anchor = self.parent['development']
        driver.resume.assert_guard({'mel_l1': anchor['mel_l1'] * 1.10, 'prediction_rms': anchor['prediction_rms'] * .80}, anchor)
        with self.assertRaises(RuntimeError):
            driver.resume.assert_guard({'mel_l1': anchor['mel_l1'] * 1.1001, 'prediction_rms': 1}, anchor)

    def test_plan3108_rng_and_common_prefix(self):
        indices = list(range(39335))
        plan = driver.resume.GlobalBatchPlan(indices)
        rng = driver.probe.tree_sha(driver.training.capture_rng_state())
        for step in (3108, 3127, 3227):
            wanted = indices.copy()
            random.Random(1234 + (step - 1) // 1229).shuffle(wanted)
            offset = ((step - 1) % 1229) * 32
            self.assertEqual(plan.for_step(step), wanted[offset:offset + 32])
        self.assertEqual(rng, driver.probe.tree_sha(driver.training.capture_rng_state()))
        with driver.preserve_rng(torch.device('cpu')):
            random.random(), np.random.random(), torch.randn(5)
        self.assertEqual(rng, driver.probe.tree_sha(driver.training.capture_rng_state()))
        first = [{'updates': n} for n in (0, 20, 40, 45)]
        second = [{'updates': n} for n in (0, 20, 40, 60, 80)]
        self.assertEqual(driver.common_updates([first, second]), [0, 20, 40])

    def test_first_losses_and_nonfinite_checks(self):
        losses = {key: 1.0 for key in driver.LOSS_KEYS}
        self.assertEqual(driver.compare_first_losses(losses, losses), {key: 0 for key in driver.LOSS_KEYS})
        with self.assertRaises(RuntimeError):
            driver.compare_first_losses(losses, dict(losses, dist_raw=1.01))
        with self.assertRaises(RuntimeError):
            driver.finite_losses(dict(losses, student_total=float('nan')))

    def test_nonpersistent_buffers_included(self):
        model = nn.Linear(2, 2)
        model.register_buffer('window', torch.ones(2), persistent=False)
        original = driver.model_sha(model)
        state = driver.probe.tree_sha(model.state_dict())
        model.window.add_(1)
        self.assertNotEqual(original, driver.model_sha(model))
        self.assertEqual(state, driver.probe.tree_sha(model.state_dict()))

    def test_exact_storage_plan_decrements_only_on_commit(self):
        budget = driver.StorageBudget(Path('/tmp'), 1424630885, 309265751)
        required = 1424630885 + 2 * 309265751 + driver.SYSTEM_RESERVE + driver.REPORT_RESERVE
        with mock.patch.object(driver, 'available_bytes', return_value=required):
            self.assertEqual(budget.require()['required_bytes'], required)
        with mock.patch.object(driver, 'available_bytes', return_value=required - 1):
            with self.assertRaises(driver.InsufficientDisk): budget.require()
        budget.committed('control_s2e4:S')
        self.assertEqual(budget.require()['required_bytes'], required - 309265751)

    def save_case(self, directory, arm='primary_s1e4', fault=None):
        items = self.restored(0.0001 if arm == 'primary_s1e4' else 0.0002)
        update(items)
        state = {'completed_global_step': 3108, 'last_full_saved_global_step': 3107, 'student_snapshot': None, 'full_snapshot': None}
        common = {'global_step': 3108, 'configuration': self.parent['configuration'], 'effective_optimizer_lrs': {'student': items[2].param_groups[0]['lr'], 'discriminator': .0002}}
        budget = driver.StorageBudget(directory, 100000, 10000)
        saver = driver.training.atomic_torch_save
        def inject(payload, path):
            if fault == 'full': raise RuntimeError('full failed')
            if path.name.startswith('student') and fault in ('s_os', 's_runtime'):
                raise OSError('S failed') if fault == 's_os' else RuntimeError('S failed')
            saver(payload, path)
        try:
            with mock.patch.object(driver.training, 'atomic_torch_save', side_effect=inject):
                driver.save_endpoint(directory, arm, 3108, common, *items, budget, state)
        except Exception as error:
            return state, error, items
        return state, None, items

    def test_true_mapping_endpoint_saver_and_only_primary_full(self):
        for arm in ('control_s2e4', 'primary_s1e4'):
            with tempfile.TemporaryDirectory(prefix='paired120_cpu_') as temp:
                path = Path(temp)
                state, error, items = self.save_case(path, arm)
                self.assertIsNone(error)
                saved = torch.load(path / 'student_step_003108.pt', map_location='cpu')
                self.assertEqual(saved['global_step'], 3108)
                self.assertEqual(driver.probe.tree_sha(saved['student']), driver.probe.tree_sha(items[0].state_dict()))
                self.assertNotIn('student_optimizer', saved)
                self.assertEqual((path / 'final_full.pt').exists(), arm == 'primary_s1e4')
                if arm == 'primary_s1e4':
                    full = torch.load(path / 'final_full.pt', map_location='cpu')
                    self.assertEqual(full['student_optimizer']['param_groups'][0]['lr'], .0001)
                    self.assertEqual(full['discriminator_optimizer']['param_groups'][0]['lr'], .0002)
                    self.assertEqual(driver.probe.tree_sha(full['rng_state']), driver.probe.tree_sha(driver.training.capture_rng_state()))
                    self.assertEqual(driver.probe.tree_sha(full['student']), driver.probe.tree_sha(saved['student']))
                    self.assertEqual(state['last_full_saved_global_step'], 3108)
                    self.assertEqual((path / 'final_full.pt').stat().st_mode & 0o222, 0)
                else:
                    self.assertEqual(state['last_full_saved_global_step'], 3107)

    def test_full_success_then_s_fail_keeps_real_cursor(self):
        for fault in ('s_os', 's_runtime'):
            with tempfile.TemporaryDirectory(prefix='paired120_failure_') as temp:
                state, error, _ = self.save_case(Path(temp), fault=fault)
                self.assertIsNotNone(error)
                self.assertEqual(state['last_full_saved_global_step'], 3108)
                self.assertTrue(state['full_snapshot']['committed'])
                self.assertIsNone(state['student_snapshot'])
                self.assertIsNotNone(state['full_snapshot']['sha256'])

    def test_full_failure_never_advances_cursor(self):
        with tempfile.TemporaryDirectory(prefix='paired120_failure_') as temp:
            state, error, _ = self.save_case(Path(temp), fault='full')
            self.assertIsInstance(error, RuntimeError)
            self.assertEqual(state['last_full_saved_global_step'], 3107)
            self.assertIsNone(state['full_snapshot'])

    def test_metadata_failure_truthful_state_even_if_failure_json_fails(self):
        with tempfile.TemporaryDirectory(prefix='paired120_metadata_') as temp:
            with mock.patch.object(driver.base, 'save_json', side_effect=OSError('metadata full')):
                state, error, _ = self.save_case(Path(temp))
                self.assertIsInstance(error, OSError)
                self.assertEqual(state['last_full_saved_global_step'], 3108)
                self.assertTrue(state['student_snapshot']['committed'])
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    driver.report_failure(Path(temp), state, error)
                self.assertIn('"last_full_saved_global_step": 3108', stream.getvalue())
                self.assertEqual(state['status'], 'failed')

    def test_partial_train_failure_source_does_not_checkpoint(self):
        source = Path(driver.__file__).read_text()
        tree = ast.parse(source)
        production_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'train_batch']
        self.assertEqual(len(production_calls), 1)
        self.assertEqual(ast.unparse(production_calls[0].func) if hasattr(ast, 'unparse') else production_calls[0].func.value.id, 'training.train_batch' if hasattr(ast, 'unparse') else 'training')
        self.assertEqual(source.count("state['partial_batch_may_have_updated_D'] = True"), 1)
        self.assertIn("partial_batch_may_have_updated_D=False", source)
        self.assertNotIn('target', [node.id for node in ast.walk(production_calls[0]) if isinstance(node, ast.Name)])
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        for handler in [node for node in ast.walk(main) if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) and node.type.id == 'Exception']:
            self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'save_endpoint' for node in ast.walk(handler)))


if __name__ == '__main__':
    unittest.main()
