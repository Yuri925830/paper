# 本文件只在租用CPU运行，用很小的真实双Adam模型验证恢复和保存；不读取或更新真实模型权重。
"""CPU acceptance tests for the bounded unchanged-ACDD2000 continuation."""

import ast  # 静态核对主循环只调用原train_batch，不把其他候选目标带进来。
import contextlib  # 故障注入结束后自动还原每个被模拟的函数。
import copy  # 复制真实训练状态，避免后续更新改变预期值。
import io  # 验证磁盘写入失败时stdout仍报告真实恢复位置。
import importlib.util  # 私有导入冻结的候选汇总校验器，不修改评估模块。
import json  # 读取实际合同、窗口和保存指针。
import os  # 禁止此测试接触GPU。
import random  # 独立重建原绝对步对应的乱序批次。
import tempfile  # 所有测试写入都限定在专门创建的临时目录。
import unittest  # 组织恢复、批次、保护和保存故障测试。
from pathlib import Path
from unittest import mock  # 只对KB级临时文件模拟容量，不改变生产磁盘门禁。

if os.name != 'posix' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("Run only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")

import numpy as np  # 验证NumPy随机状态逐元素一致。
import torch  # 仅在CPU上构建几个参数的小网络。
from torch import nn
import run_acdd_s2000_to4000_v1 as driver  # 只导入辅助函数，main不会运行。


ANCHOR = {'mel_l1': 0.19941353797912598, 'prediction_rms': 0.9994737505912781}


def models():
    student, discriminator = nn.Linear(3, 2), nn.Linear(2, 1)  # 模拟互不共享参数的S与D。
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    return student, discriminator, optimizer, d_optimizer


def update(items):
    student, discriminator, optimizer, d_optimizer = items
    random.random(), np.random.random()  # 每次更新也消耗两种非torch随机流，验证它们确实能恢复。
    source, target = torch.randn(3, 3), torch.randn(3, 2)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    d_optimizer.zero_grad(set_to_none=True)
    d_loss = (discriminator(target) - 1).square().mean() + discriminator(student(source).detach()).square().mean()
    d_loss.backward()
    d_optimizer.step()  # D先做一次真实Adam更新。
    d_optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)  # 学生目标允许梯度穿过D输入，但不更新D参数。
    optimizer.zero_grad(set_to_none=True)
    prediction = student(source)
    ((prediction - target).abs().mean() + (discriminator(prediction) - 1).square().mean()).backward()
    optimizer.step()  # S再做一次真实Adam更新；不是手工伪造moment或step。
    optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)


def snapshot(items):
    student, discriminator, optimizer, d_optimizer = items
    return copy.deepcopy({'student': student.state_dict(), 'discriminator': discriminator.state_dict(),
        'student_optimizer': optimizer.state_dict(), 'discriminator_optimizer': d_optimizer.state_dict(),
        'rng_state': driver.training.capture_rng_state()})  # 含S、D、双Adam与Python/NumPy/torchCPU随机流。


class ACDD2000ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)  # 只限制当前小CPU测试进程。

    def setUp(self):
        # CPU小模型检查点只有KB级；假容量仅隔离测试，不填满实际磁盘，也不绕过生产main门禁。
        space = mock.patch.object(driver, 'available_bytes', return_value=12 * driver.GIB)
        space.start()
        self.addCleanup(space.stop)

    def equal_nested(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.equal_nested(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.equal_nested(a, b)
        else:
            self.assertEqual(actual, expected)  # Adam的学习率、beta、步数必须完全一致。

    def save(self, output, step, items, *, include_student=False, on_full_committed=None):
        return driver.save_checkpoint(output, step, {'mel_l1': 0.1958128809928894, 'prediction_rms': 1.0058488845825195},
            *items, driver.baseline.PAPER_STUDENT_CONFIG, 'cpu-child', {'parent': 'cpu-toy'}, ANCHOR,
            include_student=include_student, on_full_committed=on_full_committed)  # 使用真实mapping与生产原子保存函数。

    def test_true_dual_adam_rng2000_resume_matches_uninterrupted2001(self):
        driver.training.seed_everything(20001)
        continuous = models()
        for _ in range(2000):
            update(continuous)  # 两套优化器都真实累计2000步，绝不只修改计数器冒充恢复。
        state2000 = snapshot(continuous)
        with tempfile.TemporaryDirectory(prefix='acdd2000_resume_cpu_', dir='/tmp') as directory:
            output = Path(directory)
            first = self.save(output, 2000, continuous)
            full = torch.load(output / 'latest_full.pt', map_location='cpu')
            self.assertEqual(full['global_step'], 2000)
            self.assertEqual(full['kind'], 'acdd_development_checkpoint')
            for key in state2000:
                self.equal_nested(full[key], state2000[key])
            update(continuous)
            uninterrupted2001 = snapshot(continuous)
            restored = models()  # 新建对象会消耗随机数，随后真实恢复必须把随机流也还原。
            driver.resume.restore_training_state(full, *restored, 2000, 'cpu-child', torch.device('cpu'))
            self.equal_nested(snapshot(restored), state2000)
            update(restored)
            self.equal_nested(snapshot(restored), uninterrupted2001)  # 下一步S/D、两组moment、全部CPU随机流逐bit相等。
            second = self.save(output, 2001, restored, include_student=True)
            self.assertNotEqual(first['checkpoint_sha256'], second['checkpoint_sha256'])
            current = torch.load(output / 'latest_full.pt', map_location='cpu')
            again = models()
            driver.resume.restore_training_state(current, *again, 2001, 'cpu-child', torch.device('cpu'))
            self.equal_nested(snapshot(again), uninterrupted2001)
            saved_student = torch.load(output / 'student_step_002001.pt', map_location='cpu')
            self.equal_nested(saved_student['student'], uninterrupted2001['student'])
            for field in ('discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state'):
                self.assertNotIn(field, saved_student)  # 独立S快照不偷偷包含体积较大的完整状态。
            for mutation in ('wrong_step', 'missing_d_adam', 'changed_lr'):
                invalid = copy.deepcopy(full)
                if mutation == 'wrong_step':
                    invalid['global_step'] = 1999
                elif mutation == 'missing_d_adam':
                    del invalid['discriminator_optimizer']
                else:
                    invalid['student_optimizer']['param_groups'][0]['lr'] = 0.0001
                with self.assertRaises(RuntimeError):
                    driver.resume.restore_training_state(invalid, *models(), 2000, 'cpu-child', torch.device('cpu'))

    def test_parent_identity_and_bounded_endpoints(self):
        parent = driver.PARENT / 'full_step_002000.pt'
        args = (parent, driver.PARENT_FULL_SHA, driver.PARENT_CONTRACT_SHA)
        self.assertEqual(list(driver.validate_launch(*args, 2001, 60)), [2001])
        steps = list(driver.validate_launch(*args, 4000, 3600))
        self.assertEqual((steps[0], steps[-1], len(steps)), (2001, 4000, 2000))
        for end in (2000, 2002, 3999, 4001, 6000):
            with self.assertRaises(RuntimeError):
                driver.validate_launch(*args, end, 60)
        for seconds in (0, 3601):
            with self.assertRaises(RuntimeError):
                driver.validate_launch(*args, 4000, seconds)
        for wrong in ((parent, 'wrong', driver.PARENT_CONTRACT_SHA),
                      (parent, driver.PARENT_FULL_SHA, 'wrong'),
                      (parent.with_name('student_step_002000.pt'), driver.PARENT_FULL_SHA, driver.PARENT_CONTRACT_SHA)):
            with self.assertRaises(RuntimeError):
                driver.validate_launch(*wrong, 4000, 60)

    def test_real39335_plan2001_and2458_to2459_preserves_rng_and_exclusions(self):
        path = driver.DEV / 'development_contract.json'
        self.assertEqual(driver.baseline.sha256_file(path), driver.resume.DEVELOPMENT_SHA)
        value = json.loads(path.read_text(encoding='utf-8'))
        indices, excluded = value['retained_train_indices'], set(value['excluded_train_indices'])
        self.assertEqual((len(indices), len(excluded)), (39335, 52))
        plan = driver.resume.GlobalBatchPlan(indices)
        self.assertEqual(plan.batches_per_pass, 1229)
        driver.training.seed_everything(239)
        rng = driver.training.capture_rng_state()
        for step in (2001, 2458, 2459, 3000, 3687, 3688, 4000):
            pass_index = (step - 1) // 1229
            order = list(indices)
            random.Random(1234 + pass_index).shuffle(order)  # 从固定原顺序独立构造该轮，不能接着乱序上一轮。
            offset = ((step - 1) % 1229) * 32
            self.assertEqual(plan.for_step(step), order[offset:offset + 32])
            self.assertFalse(set(plan.for_step(step)) & excluded)
            self.assertEqual(plan.pass_index, pass_index)
        self.equal_nested(driver.training.capture_rng_state(), rng)  # 批次规划不能消耗训练噪声或裁剪随机流。
        self.assertEqual(((2458 - 1) // 1229, (2459 - 1) // 1229), (1, 2))

    def test_parent2000_loss_windows_are_preserved_and_continue_at2050(self):
        parent_path = driver.PARENT / 'loss_windows.jsonl'
        self.assertEqual(driver.baseline.sha256_file(parent_path), driver.PARENT_WINDOWS_SHA)
        original = parent_path.read_bytes()
        with tempfile.TemporaryDirectory(prefix='acdd_windows_cpu_', dir='/tmp') as directory:
            output = Path(directory)
            monitor = driver.start_monitor(output, parent_path)
            self.assertEqual((output / 'loss_windows.jsonl').read_bytes(), original)
            expected = [json.loads(line) for line in original.decode('utf-8').splitlines() if line][-4:]
            self.assertEqual(monitor.windows, expected)  # 不清空旧目标的最近四个窗口，也不替换成转换目标。
            self.assertEqual(monitor.last_global_step, 2000)
            restored_alert = any(driver.training.LossMonitor.sustained_rise([entry['mean'][name] for entry in expected])
                                 for name in ('student_total', 'dist_raw'))
            self.assertEqual(monitor.active_alert, restored_alert)
            for step in range(2001, 2051):
                monitor.observe(step, 0, {'student_total': 1.0, 'dist_raw': 0.01})
            self.assertEqual(monitor.windows[-1]['global_step'], 2050)
            self.assertEqual([entry['global_step'] for entry in monitor.windows[:-1]],
                             [entry['global_step'] for entry in expected[-3:]])
        self.assertEqual(parent_path.read_bytes(), original)  # 权威父窗口保持只读。

    def test_original_s0_guard_is_not_rebased_to_parent2000(self):
        driver.resume.assert_guard({'mel_l1': 0.1958128809928894, 'prediction_rms': 1.0058488845825195}, ANCHOR)
        driver.resume.assert_guard({'mel_l1': ANCHOR['mel_l1'] * 1.10, 'prediction_rms': ANCHOR['prediction_rms'] * 0.80}, ANCHOR)
        for scores in ({'mel_l1': ANCHOR['mel_l1'] * 1.1001, 'prediction_rms': 1.0},
                       {'mel_l1': 0.19, 'prediction_rms': ANCHOR['prediction_rms'] * 0.7999},
                       {'mel_l1': float('nan'), 'prediction_rms': 1.0}):
            with self.assertRaises(RuntimeError):
                driver.resume.assert_guard(scores, ANCHOR)

    def test_rolling_full_every100_and_only1000_or_actual_end_student_snapshots(self):
        regular = [step for step in range(2100, 4001, 100) if driver.should_snapshot(step, False)]
        self.assertEqual(regular, [3000, 4000])
        self.assertTrue(driver.should_snapshot(2017, True))
        driver.training.seed_everything(43)
        items = models()
        update(items)
        with tempfile.TemporaryDirectory(prefix='acdd_storage_schedule_cpu_', dir='/tmp') as directory:
            output = Path(directory)
            self.save(output, 2100, items, include_student=driver.should_snapshot(2100, False))
            self.assertFalse(list(output.glob('student_step_*.pt')))
            self.save(output, 3000, items, include_student=driver.should_snapshot(3000, False))
            preserved = driver.baseline.sha256_file(output / 'student_step_003000.pt')
            self.save(output, 3100, items, include_student=driver.should_snapshot(3100, False))
            last = self.save(output, 3117, items, include_student=driver.should_snapshot(3117, True))
            self.assertEqual(driver.baseline.sha256_file(output / 'student_step_003000.pt'), preserved)
            self.assertEqual(sorted(path.name for path in output.glob('student_step_*.pt')),
                             ['student_step_003000.pt', 'student_step_003117.pt'])
            self.assertEqual(len(list(output.glob('latest_full.pt'))), 1)
            full = torch.load(output / 'latest_full.pt', map_location='cpu')
            self.assertEqual(full['global_step'], 3117)
            self.assertEqual(full['score_semantics'], 'reconstruction_clean_x0_sqrt_alpha_bar')
            self.assertEqual(last['global_step'], 3117)
            with self.assertRaises(RuntimeError):
                self.save(output, 3000, items, include_student=True)  # 已归档S永远不能覆盖。

    def test_capacity_keeps_system_reserve_and_budgets_smoke_plus_segment(self):
        normal = 2 * driver.FULL_WRITE_BYTES + 2 * driver.STUDENT_WRITE_BYTES + driver.SYSTEM_RESERVE_BYTES
        smoke = normal + driver.FULL_WRITE_BYTES + driver.STUDENT_WRITE_BYTES
        self.assertEqual(driver.SYSTEM_RESERVE_BYTES, 3 * driver.GIB)
        for endpoint, need in ((4000, normal), (2001, smoke)):
            with mock.patch.object(driver, 'available_bytes', return_value=need - 1):
                with self.assertRaises(driver.InsufficientDisk):
                    driver.require_segment_capacity(Path('/tmp'), endpoint)
            with mock.patch.object(driver, 'available_bytes', return_value=need):
                self.assertEqual(driver.require_segment_capacity(Path('/tmp'), endpoint)['required_bytes'], need)
        minimum = driver.FULL_WRITE_BYTES + driver.STUDENT_WRITE_BYTES + driver.SYSTEM_RESERVE_BYTES
        with mock.patch.object(driver, 'available_bytes', return_value=minimum - 1):
            with self.assertRaises(driver.InsufficientDisk):
                driver.require_space(Path('/tmp'), True)
        with tempfile.TemporaryDirectory(prefix='acdd_low_disk_cpu_', dir='/tmp') as directory:
            output, items = Path(directory), models()
            update(items)
            with mock.patch.object(driver, 'available_bytes', return_value=driver.FULL_WRITE_BYTES + driver.SYSTEM_RESERVE_BYTES):
                receipt = self.save(output, 2001, items, include_student=True)
            self.assertTrue(receipt['full_committed'])
            self.assertTrue(receipt['stop_for_disk'])
            self.assertIsNone(receipt['student_snapshot'])
            self.assertTrue((output / 'latest_full.pt').is_file())  # 能保住full时记录真实状态，但明确停止且不假称S存在。
            before = driver.baseline.sha256_file(output / 'latest_full.pt')
            with mock.patch.object(driver, 'available_bytes', return_value=driver.FULL_WRITE_BYTES + driver.SYSTEM_RESERVE_BYTES - 1):
                with self.assertRaises(driver.InsufficientDisk):
                    self.save(output, 2002, items, include_student=True)
            self.assertEqual(driver.baseline.sha256_file(output / 'latest_full.pt'), before)

    def test_commit_cursor_is_truthful_for_snapshot_metadata_hash_and_full_failures(self):
        for fault in ('student_oserror', 'student_runtimeerror', 'initial_receipt', 'final_receipt', 'history', 'full_hash', 'full_write'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix='acdd_failure_cpu_', dir='/tmp') as directory:
                output, items = Path(directory), models()
                update(items)
                first = self.save(output, 1, items, include_student=True)
                original_student = driver.baseline.sha256_file(output / 'student_step_000001.pt')
                update(items)
                cursor = {'step': 1, 'receipt': first}
                def committed(receipt):
                    cursor.update(step=receipt['global_step'], receipt=receipt)  # 与main相同：full一成功就记录同一个可变收据。
                original_save, original_json = driver.training.atomic_torch_save, driver.baseline.save_json
                original_open, original_hash = Path.open, driver.baseline.sha256_file
                counts = {'metadata': 0}
                def failing_save(payload, path):
                    if path.name == 'student_step_000002.pt' and fault in ('student_oserror', 'student_runtimeerror'):
                        raise (OSError if fault == 'student_oserror' else RuntimeError)('injected snapshot failure')
                    if path.name == 'latest_full.pt' and fault == 'full_write':
                        raise RuntimeError('injected full failure before replace')
                    return original_save(payload, path)
                def failing_json(path, value):
                    if path.name == 'latest_full.json':
                        counts['metadata'] += 1
                        if (fault, counts['metadata']) in (('initial_receipt', 1), ('final_receipt', 2)):
                            raise OSError('injected receipt failure')
                    return original_json(path, value)
                def failing_open(path, *args, **kwargs):
                    if path.name == 'checkpoint_history.jsonl' and fault == 'history' and args and args[0] == 'a':
                        raise OSError('injected history failure')
                    return original_open(path, *args, **kwargs)
                def failing_hash(path):
                    if Path(path).name == 'latest_full.pt' and fault == 'full_hash':
                        raise OSError('injected hash read failure')
                    return original_hash(path)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(driver.training, 'atomic_torch_save', side_effect=failing_save))
                    stack.enter_context(mock.patch.object(driver.baseline, 'save_json', side_effect=failing_json))
                    stack.enter_context(mock.patch.object(Path, 'open', new=failing_open))
                    stack.enter_context(mock.patch.object(driver.baseline, 'sha256_file', side_effect=failing_hash))
                    with self.assertRaises((OSError, RuntimeError)) as error:
                        self.save(output, 2, items, include_student=True, on_full_committed=committed)
                expected_step = 1 if fault == 'full_write' else 2
                self.assertEqual(cursor['step'], expected_step)
                full = torch.load(output / 'latest_full.pt', map_location='cpu')
                self.assertEqual(full['global_step'], expected_step)
                if fault == 'full_hash':
                    self.assertIsNone(cursor['receipt']['checkpoint_sha256'])  # 新full哈希未知时不能借用旧full哈希。
                if fault == 'full_write':
                    self.assertEqual(driver.baseline.sha256_file(output / 'latest_full.pt'), first['checkpoint_sha256'])
                else:
                    restored = models()
                    driver.resume.restore_training_state(full, *restored, 2, 'cpu-child', torch.device('cpu'))
                    self.equal_nested(restored[0].state_dict(), items[0].state_dict())
                    self.equal_nested(restored[2].state_dict(), items[2].state_dict())
                    self.equal_nested(restored[3].state_dict(), items[3].state_dict())
                self.assertEqual(driver.baseline.sha256_file(output / 'student_step_000001.pt'), original_student)
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    report = driver.report_failure(output, mock.Mock(), 2, cursor['step'], cursor['receipt'], error.exception)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['last_saved_global_step'], expected_step)
                self.assertEqual(report['unsaved_completed_steps'], 2 - expected_step)
                self.assertFalse((output / 'completion.json').exists())
                self.assertIn('ACDD_CONTINUATION_FAILED', stream.getvalue())

    def test_failure_reporting_survives_second_metadata_error(self):
        with tempfile.TemporaryDirectory(prefix='acdd_failed_metadata_cpu_', dir='/tmp') as directory:
            monitor = mock.Mock()
            monitor.finish.side_effect = OSError('secondary monitor failure')
            stream = io.StringIO()
            with mock.patch.object(driver.baseline, 'save_json', side_effect=OSError('secondary status failure')), contextlib.redirect_stdout(stream):
                result = driver.report_failure(Path(directory), monitor, 2105, 2100, {'global_step': 2100}, RuntimeError('original snapshot failure'))
            printed = json.loads(stream.getvalue().splitlines()[0].split(' ', 1)[1])
            self.assertEqual(printed['last_saved_global_step'], 2100)
            self.assertEqual(printed['unsaved_completed_steps'], 5)
            self.assertEqual(result['error'], 'original snapshot failure')
            self.assertIn('FAILURE_MONITOR_WRITE_ERROR', stream.getvalue())
            self.assertIn('FAILURE_STATUS_WRITE_ERROR', stream.getvalue())

    def test_disk_stop_metadata_error_is_failed_and_reraises_original_exception(self):
        for error_type in (OSError, RuntimeError):
            with tempfile.TemporaryDirectory(prefix='acdd_disk_stop_cpu_', dir='/tmp') as directory:
                error = error_type('disk stop metadata failure')
                stream = io.StringIO()
                with mock.patch.object(driver.baseline, 'save_json', side_effect=error), contextlib.redirect_stdout(stream):
                    with self.assertRaises(error_type) as captured:
                        driver.write_disk_stop(Path(directory), mock.Mock(), 2110, 2100, {'global_step': 2100}, driver.InsufficientDisk('space low'))
                self.assertIs(captured.exception, error)
                value = json.loads(stream.getvalue().splitlines()[0].split(' ', 1)[1])
                self.assertEqual((value['status'], value['last_saved_global_step'], value['unsaved_completed_steps']), ('failed', 2100, 10))
                self.assertFalse((Path(directory) / 'completion.json').exists())

    def test_source_calls_only_original_training_batch_and_absolute_seed(self):
        source = Path(driver.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        batch_calls = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'train_batch']
        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(ast.dump(batch_calls[0].func), ast.dump(ast.parse('training.train_batch', mode='eval').body))
        self.assertNotIn('conversion_score_step', source)
        self.assertNotIn('permutation_seed', source)
        self.assertIn('1234 + 100000 + step', source)
        self.assertIn('1234 + 200000 + step', source)
        self.assertIn('resume.restore_training_state(payload, student, discriminator, optimizer, d_optimizer, 2000', source)

    def test_actual_contract_fields_pass_frozen_candidate_exclusion_validator(self):
        path = driver.ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development_candidate.py'
        self.assertEqual(driver.baseline.sha256_file(path), '2c08b3700f405dc9deb85e6eecc5ff63309642e15113012242f81580704078e6')
        spec = importlib.util.spec_from_file_location('_frozen_candidate_exclusion_test', path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)  # 只导入；冻结评估器内部仍会检查运行主机和其统计辅助文件SHA。
        development_path = driver.DEV / 'development_contract.json'
        file_sha = driver.baseline.sha256_file(development_path)
        self.assertEqual(file_sha, driver.resume.DEVELOPMENT_SHA)
        development = json.loads(development_path.read_text(encoding='utf-8'))
        parent = json.loads((driver.PARENT / 'run_contract.json').read_text(encoding='utf-8'))
        tree = ast.parse(Path(driver.__file__).read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        record = next(node.value for node in main.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'record' for target in node.targets))
        needed = {'formal_test_used_for_selection', 'excluded_train_indices', 'training_indices',
                  'reconstruction_dev_indices', 'development_contract_file_sha256', 'seed'}
        actual = {}
        for key, value in zip(record.keys, record.values):
            if isinstance(key, ast.Constant) and key.value in needed:
                expression = ast.fix_missing_locations(ast.Expression(body=value))
                actual[key.value] = eval(compile(expression, str(driver.__file__), 'eval'),
                                        {'parent': parent, 'resume': driver.resume})
        self.assertEqual(set(actual), needed)  # 从main真实合同表达式取值；不能只在测试里补字段掩盖驱动缺项。
        self.assertEqual(actual['seed'], 1234)
        self.assertNotEqual(file_sha, candidate.canonical_sha(development))  # 明确文件SHA与canonical属于两种身份。
        candidate.validate_training_exclusion(actual, development, file_sha)  # 实际调用冻结评估校验器，不复制其实现。
        for wrong in (None, candidate.canonical_sha(development)):
            invalid = copy.deepcopy(actual)
            if wrong is None:
                del invalid['development_contract_file_sha256']
            else:
                invalid['development_contract_file_sha256'] = wrong
            with self.assertRaises(ValueError):
                candidate.validate_training_exclusion(invalid, development, file_sha)  # 缺文件身份或误填canonical都拒绝。


if __name__ == '__main__':
    unittest.main()  # 只有显式CPU测试命令执行测试；绝不自动启动GPU主入口。
