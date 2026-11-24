# 租用CPU上的真实保存/恢复测试；不读真实大权重，也不运行GPU主入口。
"""CPU acceptance tests for the independent rolling-checkpoint CV driver."""

import copy  # 保存CPU小模型各时刻的真实参数和Adam状态。
import contextlib  # 同时注入不同写盘阶段的故障，测试结束自动恢复真实函数。
import io  # 捕获失败日志，核验磁盘metadata不可写时stdout仍报告真实恢复步。
import json  # 读取实际写入的检查点指针与目标切换说明。
import os  # 强制CPU运行。
import tempfile  # 所有生成检查点只在租用机的临时目录。
import unittest  # 自动检查边界、磁盘、保存、恢复和监控。
from pathlib import Path  # 不使用任何旧实验目录作为写入目标。
from unittest import mock  # 模拟磁盘不足，不实际填满磁盘。

if os.name != 'posix' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("These project tests must run on Matpool with CUDA_VISIBLE_DEVICES=''")

import numpy as np  # 比较NumPy RNG数组，不重新实现恢复方法。
import torch  # 仅CPU上的小线性网络和Adam。
from torch import nn
import run_conversion_score_development_v1 as driver  # 导入会验冻结模块哈希，但不会调用main。


def models():
    student, discriminator = nn.Linear(3, 2), nn.Linear(2, 1)  # 两组不共享参数的小网络。
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    return student, discriminator, optimizer, d_optimizer  # 每一组Adam只管理自己的模型。


def update(items):
    student, discriminator, optimizer, d_optimizer = items
    source, target = torch.randn(3, 3), torch.randn(3, 2)  # CPU假输入；不使用项目音频。
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    d_optimizer.zero_grad(set_to_none=True)
    d_loss = (discriminator(target) - 1).square().mean() + discriminator(student(source).detach()).square().mean()
    d_loss.backward()
    d_optimizer.step()  # 第一次真实Adam更新属于D。
    d_optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    prediction = student(source)
    loss = (prediction - target).abs().mean() + (discriminator(prediction) - 1).square().mean()
    loss.backward()
    optimizer.step()  # 第二次真实Adam更新属于学生。
    optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)


def snapshot(items):
    student, discriminator, optimizer, d_optimizer = items
    return copy.deepcopy({'student': student.state_dict(), 'discriminator': discriminator.state_dict(),
        'student_optimizer': optimizer.state_dict(), 'discriminator_optimizer': d_optimizer.state_dict(),
        'rng_state': driver.training.capture_rng_state()})  # 保留完整实际状态，而非伪造step。


class ConversionDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)  # 仅当前CPU测试进程限线程。

    def setUp(self):
        # 当前租用盘约3GiB：仅这些KB级CPU临时文件模拟8GiB可用；生产主入口门槛完全未改。
        space = mock.patch.object(driver, 'available_bytes', return_value=8 * driver.GIB)
        space.start()
        self.addCleanup(space.stop)  # 每个测试结束恢复函数；磁盘故障测试可在内层覆盖为更低值。

    def equal_nested(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.equal_nested(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.equal_nested(a, b)
        else:
            self.assertEqual(actual, expected)  # 学习率、beta和step必须精确一致。

    def save(self, output, step, items, *, on_full_committed=None):
        return driver.save_checkpoint(output, step, {'mel_l1': 0.19, 'prediction_rms': 1.0}, *items,
            driver.baseline.PAPER_STUDENT_CONFIG, 'cpu-child', {'parent': 'cpu-toy'},
            {'mel_l1': 0.19941353797912598, 'prediction_rms': 0.9994737505912781},
            on_full_committed=on_full_committed)  # 调用真实mapping构造与真实原子保存。

    def test_fault_injection_records_full_commit_before_snapshot_or_metadata_failure(self):
        faults = ('student_oserror', 'student_runtimeerror', 'initial_receipt', 'final_receipt',
                  'history', 'full_hash', 'full_write')  # 覆盖成功full之后的每个危险阶段及真正full失败。
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix='conversion_fault_cpu_', dir='/tmp') as directory:
                output = Path(directory)
                driver.training.seed_everything(662)
                items = models()
                update(items)
                first = self.save(output, 1, items)  # 先有真实可恢复旧状态，便于判断是否错误推进。
                old_student_sha = driver.baseline.sha256_file(output / 'student_step_000001.pt')
                update(items)  # 内存里现在是第二个完整小模型更新。
                committed = {'step': 1, 'receipt': first}

                def record_commit(receipt):
                    committed['step'] = receipt['global_step']  # 与main中的即时内存赋值回调保持相同语义。
                    committed['receipt'] = receipt  # 不复制，让后续hash或失败阶段立即可见。

                original_save, original_json = driver.training.atomic_torch_save, driver.baseline.save_json
                original_open, original_hash = Path.open, driver.baseline.sha256_file
                metadata_calls = {'count': 0}

                def failing_save(payload, path):
                    if path.name == 'student_step_000002.pt' and fault in ('student_oserror', 'student_runtimeerror'):
                        error_type = OSError if fault == 'student_oserror' else RuntimeError
                        raise error_type('injected student snapshot write failure')
                    if path.name == 'latest_full.pt' and fault == 'full_write':
                        raise RuntimeError('injected full write failure before atomic replacement')
                    return original_save(payload, path)

                def failing_json(path, value):
                    if path.name == 'latest_full.json':
                        metadata_calls['count'] += 1
                        if ((fault == 'initial_receipt' and metadata_calls['count'] == 1)
                                or (fault == 'final_receipt' and metadata_calls['count'] == 2)):
                            raise OSError('injected receipt write failure')
                    return original_json(path, value)

                def failing_open(path, *args, **kwargs):
                    if path.name == 'checkpoint_history.jsonl' and fault == 'history' and args and args[0] == 'a':
                        raise OSError('injected history append failure')
                    return original_open(path, *args, **kwargs)

                def failing_hash(path):
                    if Path(path).name == 'latest_full.pt' and fault == 'full_hash':
                        raise OSError('injected hash read failure after complete replacement')
                    return original_hash(path)

                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(driver.training, 'atomic_torch_save', side_effect=failing_save))
                    stack.enter_context(mock.patch.object(driver.baseline, 'save_json', side_effect=failing_json))
                    stack.enter_context(mock.patch.object(Path, 'open', new=failing_open))
                    stack.enter_context(mock.patch.object(driver.baseline, 'sha256_file', side_effect=failing_hash))
                    with self.assertRaises((OSError, RuntimeError)) as captured:
                        self.save(output, 2, items, on_full_committed=record_commit)
                expected_step = 1 if fault == 'full_write' else 2
                self.assertEqual(committed['step'], expected_step)  # full成功即推进；S或metadata失败不能回退。
                actual = torch.load(output / 'latest_full.pt', map_location='cpu')
                self.assertEqual(actual['global_step'], expected_step)  # 报告步数必须与真实磁盘恢复状态一致。
                self.assertEqual(driver.baseline.sha256_file(output / 'student_step_000001.pt'), old_student_sha)
                if fault == 'full_write':
                    self.assertEqual(driver.baseline.sha256_file(output / 'latest_full.pt'), first['checkpoint_sha256'])
                else:
                    self.assertTrue(committed['receipt']['full_committed'])
                    if fault == 'full_hash':
                        self.assertIsNone(committed['receipt']['checkpoint_sha256'])  # 未算出SHA不伪造值，但已保存步明确。
                    else:
                        self.assertEqual(committed['receipt']['checkpoint_sha256'], driver.baseline.sha256_file(output / 'latest_full.pt'))
                    restored = models()
                    driver.resume.restore_training_state(actual, *restored, 2, 'cpu-child', torch.device('cpu'))
                    self.equal_nested(restored[0].state_dict(), items[0].state_dict())  # 失败后full确实能恢复学生和双Adam。
                    self.equal_nested(restored[2].state_dict(), items[2].state_dict())
                    self.equal_nested(restored[3].state_dict(), items[3].state_dict())
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    summary = driver.report_run_failure(output, mock.Mock(), 2, committed['step'], committed['receipt'],
                                                        output / 'parent.pt', captured.exception)
                self.assertEqual(summary['status'], 'failed')  # 模型落盘不能把附属写入失败改称候选完成。
                self.assertEqual(summary['last_saved_global_step'], expected_step)
                self.assertEqual(summary['unsaved_completed_steps'], 2 - expected_step)
                self.assertEqual(json.loads((output / 'status.json').read_text())['last_saved_global_step'], expected_step)
                self.assertIn('CONVERSION_SCORE_DEVELOPMENT_FAILED', stream.getvalue())
                self.assertFalse((output / 'completion.json').exists())

    def test_failure_report_retains_truth_when_monitor_and_status_cannot_be_written(self):
        with tempfile.TemporaryDirectory(prefix='conversion_failure_stdout_cpu_', dir='/tmp') as directory:
            receipt = {'global_step': 702, 'full_committed': True, 'write_stage': 'initial_receipt_write',
                       'path': str(Path(directory) / 'latest_full.pt'), 'checkpoint_sha256': 'cpu-test-placeholder'}
            monitor = mock.Mock()
            monitor.finish.side_effect = OSError('injected monitor disk failure')
            stream = io.StringIO()
            with mock.patch.object(driver.baseline, 'save_json', side_effect=OSError('injected status disk failure')), contextlib.redirect_stdout(stream):
                summary = driver.report_run_failure(Path(directory), monitor, 702, 702, receipt,
                    Path(directory) / 'parent.pt', RuntimeError('original snapshot error'))
            self.assertEqual(summary['last_saved_global_step'], 702)
            self.assertEqual(summary['error'], 'original snapshot error')  # 二次metadata错误不能盖掉原失败原因。
            lines = stream.getvalue().splitlines()
            printed = json.loads(lines[0].split(' ', 1)[1])
            self.assertEqual(printed['status'], 'failed')
            self.assertEqual(printed['last_saved_global_step'], 702)
            self.assertIn('FAILURE_MONITOR_WRITE_ERROR', stream.getvalue())
            self.assertIn('FAILURE_STATUS_WRITE_ERROR', stream.getvalue())

    def test_disk_stop_metadata_failure_reports_real_recovery_step_and_reraises_same_error(self):
        for error_type in (OSError, RuntimeError):
            with self.subTest(error_type=error_type), tempfile.TemporaryDirectory(prefix='conversion_disk_stop_cpu_', dir='/tmp') as directory:
                output = Path(directory)
                metadata_error = error_type('injected disk_stop metadata write failure')
                disk_error = driver.InsufficientDisk('injected low-space stop')
                receipt = {'global_step': 700, 'full_committed': True, 'write_stage': 'complete',
                           'path': str(output / 'latest_full.pt'), 'checkpoint_sha256': 'cpu-test-placeholder'}
                stream = io.StringIO()
                with mock.patch.object(driver.baseline, 'save_json', side_effect=metadata_error), contextlib.redirect_stdout(stream):
                    with self.assertRaises(error_type) as captured:
                        driver.write_disk_stop_or_report_failure(output, mock.Mock(), 704, 700, receipt,
                                                                 output / 'parent.pt', disk_error)
                self.assertIs(captured.exception, metadata_error)  # 必须抛出原metadata异常，不替换成成功或普通低盘退出。
                printed = json.loads(stream.getvalue().splitlines()[0].split(' ', 1)[1])
                self.assertEqual(printed['status'], 'failed')
                self.assertEqual(printed['completed_global_step'], 704)
                self.assertEqual(printed['last_saved_global_step'], 700)
                self.assertEqual(printed['unsaved_completed_steps'], 4)
                self.assertEqual(printed['error'], str(metadata_error))
                self.assertFalse((output / 'completion.json').exists())  # 二次写盘失败不能留下“候选完成”的文件。

    def test_only_parent600_and_explicit601_or1200_bounds(self):
        path = driver.resume.PARENT / 'latest_full.pt'
        arguments = (path, driver.resume.PARENT_FULL_SHA, driver.resume.PARENT_CONTRACT_SHA)
        self.assertEqual(list(driver.validate_launch(*arguments, 601, 60)), [601])  # smoke只有一次更新。
        steps = list(driver.validate_launch(*arguments, 1200, 3600))
        self.assertEqual((steps[0], steps[-1], len(steps)), (601, 1200, 600))  # 正常边界不多跑1201，也不重播600。
        for end in (600, 602, 1199, 1201, 2000):
            with self.assertRaises(RuntimeError):
                driver.validate_launch(*arguments, end, 60)
        for seconds in (0, 3601):
            with self.assertRaises(RuntimeError):
                driver.validate_launch(*arguments, 1200, seconds)
        with self.assertRaises(RuntimeError):
            driver.validate_launch(path, 'wrong', driver.resume.PARENT_CONTRACT_SHA, 1200, 60)
        with self.assertRaises(RuntimeError):
            driver.validate_launch(path.with_name('student_step_000600.pt'), driver.resume.PARENT_FULL_SHA,
                                   driver.resume.PARENT_CONTRACT_SHA, 1200, 60)

    def test_true_global601_plan_keeps_original_train_population(self):
        contract_path = driver.DEV / 'development_contract.json'
        self.assertEqual(driver.baseline.sha256_file(contract_path), driver.resume.DEVELOPMENT_SHA)
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        plan = driver.resume.GlobalBatchPlan(contract['retained_train_indices'])
        self.assertEqual(plan.batches_per_pass, 1229)
        self.assertEqual(plan.for_step(601), [5870, 6470, 13794, 10241, 31793, 17531, 3684, 30419,
            32108, 20396, 5673, 6424, 16926, 37134, 33859, 19164, 14254, 5178, 2024, 13401,
            20386, 35375, 15160, 25202, 5482, 14484, 36684, 35452, 16953, 12028, 6156, 21134])
        self.assertFalse(set(plan.for_step(1200)) & set(contract['excluded_train_indices']))

    def test_real_mapping_s_snapshot_and_two_rolling_full_adam_rng_restores(self):
        driver.training.seed_everything(6018)
        items = models()
        for _ in range(601):
            update(items)  # 小CPU网络真实更新601次，双方step和moment不是手填的。
        state601 = snapshot(items)
        with tempfile.TemporaryDirectory(prefix='conversion_rolling_cpu_', dir='/tmp') as directory:
            output = Path(directory)
            first = self.save(output, 601, items)
            first_full = torch.load(output / 'latest_full.pt', map_location='cpu')
            first_student = torch.load(output / 'student_step_000601.pt', map_location='cpu')
            self.assertIsInstance(first_student, dict)
            self.assertEqual(first_student['global_step'], 601)
            self.equal_nested(first_student['student'], state601['student'])
            for field in ('discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state'):
                self.assertNotIn(field, first_student)  # 永久快照只保留学生，完整状态在rolling文件。
                self.equal_nested(first_full[field], state601[field])
            original_student_sha = driver.baseline.sha256_file(output / 'student_step_000601.pt')
            restored = models()  # 先故意消耗RNG建新模型，再用真实恢复函数验证恢复。
            driver.resume.restore_training_state(first_full, *restored, 601, 'cpu-child', torch.device('cpu'))
            self.equal_nested(snapshot(restored), state601)
            update(restored)  # 从完整601状态继续真实602；不能重置两套Adam。
            state602 = snapshot(restored)
            second = self.save(output, 602, restored)
            self.assertNotEqual(first['checkpoint_sha256'], second['checkpoint_sha256'])
            self.assertEqual(driver.baseline.sha256_file(output / 'student_step_000601.pt'), original_student_sha)
            self.assertEqual(len(list(output.glob('latest_full.pt'))), 1)
            self.assertEqual(len(list(output.glob('student_step_*.pt'))), 2)
            current = torch.load(output / 'latest_full.pt', map_location='cpu')
            second_restored = models()
            driver.resume.restore_training_state(current, *second_restored, 602, 'cpu-child', torch.device('cpu'))
            self.equal_nested(snapshot(second_restored), state602)  # 第二次rolling也精确恢复S/D、双Adam和RNG。
            self.assertEqual(json.loads((output / 'latest_full.json').read_text())['checkpoint_sha256'], driver.baseline.sha256_file(output / 'latest_full.pt'))
            with self.assertRaises(RuntimeError):
                self.save(output, 602, second_restored)  # 已存在的学生快照绝不覆盖。
            bad = copy.deepcopy(current)
            del bad['discriminator_optimizer']
            with self.assertRaises(RuntimeError):
                driver.resume.restore_training_state(bad, *models(), 602, 'cpu-child', torch.device('cpu'))

    def test_disk_rejection_preserves_previous_full_and_safe_full_only_stops(self):
        driver.training.seed_everything(991)
        items = models()
        update(items)
        with tempfile.TemporaryDirectory(prefix='conversion_disk_cpu_', dir='/tmp') as directory:
            output = Path(directory)
            first = self.save(output, 1, items)  # CPU磁盘测试使用小step，不执行主入口的600父限制。
            with mock.patch.object(driver, 'available_bytes', return_value=4 * driver.GIB):
                with self.assertRaises(driver.InsufficientDisk):
                    self.save(output, 2, items)  # 不足full1.5+系统3GiB时任何旧完整文件都不能被覆盖。
            self.assertEqual(driver.baseline.sha256_file(output / 'latest_full.pt'), first['checkpoint_sha256'])
            self.assertFalse((output / 'student_step_000002.pt').exists())
            update(items)
            with mock.patch.object(driver, 'available_bytes', return_value=int(4.6 * driver.GIB)):
                result = self.save(output, 2, items)  # 足够full但不足额外S预算：先保存完整状态，再要求停止。
            self.assertTrue(result['stop_for_disk'])
            self.assertIsNone(result['student_snapshot'])
            self.assertEqual(torch.load(output / 'latest_full.pt', map_location='cpu')['global_step'], 2)
            self.assertFalse((output / 'student_step_000002.pt').exists())

    def test_original_guard_cannot_be_rebased_to_new_parent_or_target(self):
        anchor = {'mel_l1': 0.19941353797912598, 'prediction_rms': 0.9994737505912781}
        driver.resume.assert_guard({'mel_l1': 0.20, 'prediction_rms': 0.99}, anchor)
        with self.assertRaises(RuntimeError):
            driver.resume.assert_guard({'mel_l1': anchor['mel_l1'] * 1.10001, 'prediction_rms': 1.0}, anchor)
        with self.assertRaises(RuntimeError):
            driver.resume.assert_guard({'mel_l1': 0.19, 'prediction_rms': anchor['prediction_rms'] * 0.79999}, anchor)

    def test_parent_windows_preserved_but_new_cv_trend_has_no_old_rec_reference(self):
        with tempfile.TemporaryDirectory(prefix='conversion_boundary_cpu_', dir='/tmp') as directory:
            root = Path(directory)
            parent = root / 'parent_windows.jsonl'
            old = [dict(global_step=450 + i * 50, mean={'student_total': 1.0 + i * 0.1, 'dist_raw': 0.01 + i * 0.001}) for i in range(4)]
            parent.write_text(''.join(json.dumps(item) + '\n' for item in old))  # 只创建测试临时历史，不触碰真实父日志。
            original_bytes = parent.read_bytes()
            output = root / 'child'
            output.mkdir()
            monitor = driver.start_boundary_monitor(output, parent)
            self.assertEqual(monitor.last_global_step, 600)
            self.assertEqual(monitor.windows, [])
            self.assertFalse(monitor.active_alert)
            self.assertEqual((output / 'parent_reconstruction_loss_windows_through_600.jsonl').read_bytes(), original_bytes)
            for step in range(601, 651):
                monitor.observe(step, 0, {'student_total': 100.0, 'dist_raw': 3.0})  # 新目标尺度即使更大，也不能跨旧历史报上涨。
            self.assertEqual(len(monitor.windows), 1)
            self.assertEqual(monitor.windows[0]['global_step'], 650)
            self.assertFalse(monitor.active_alert)
            self.assertFalse((output / 'loss_alerts.jsonl').exists())
            self.assertEqual(parent.read_bytes(), original_bytes)
            self.assertFalse(json.loads((output / 'objective_boundary.json').read_text())['cross_boundary_trend_comparison'])
            with self.assertRaises(FloatingPointError):
                monitor.observe(651, 0, {'student_total': float('nan'), 'dist_raw': 3.0})  # 非有限异常仍需立即报警。
            self.assertTrue((output / 'loss_alerts.jsonl').exists())


if __name__ == '__main__':
    unittest.main()  # 只通过明确的远端CPU命令执行。
