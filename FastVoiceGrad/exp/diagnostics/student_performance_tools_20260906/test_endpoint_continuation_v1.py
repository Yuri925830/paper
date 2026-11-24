# 仅在Matpool的CPU核对续训状态，不加载教师、不使用GPU、不启动项目训练。
"""CPU tests for immutable endpoint continuation and exact toy Adam replay."""

import ast  # 执行真实保存闭包，避免测试自行写另一套保存逻辑。
import copy  # 为负例复制契约，保证正确基线不被修改。
import json  # 检查新输出的来源和日志文件。
import random  # 独立手算v3的每pass打乱顺序。
import tempfile  # 微型CPU状态只写入服务器/tmp。
import unittest  # 运行独立的小型回归测试。
from pathlib import Path  # 读取实际驱动脚本及临时保存结果。
from unittest.mock import patch  # 证明不合格父状态会在GPU/加锁之前被拒绝。

import torch  # 只处理微型CPU线性层和Adam状态。
import run_endpoint_continuation_v1 as driver  # 导入不会进入训练main。

core, training = driver.core, driver.training


def toy_setup():
    # 保持每条测试轨迹的初始参数完全一致。
    training.seed_everything(901)
    student = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(student.parameters(), lr=driver.BASE_LR,
        betas=(0.5, 0.9), eps=1e-8, weight_decay=0, amsgrad=False)
    return student, optimizer


def toy_updates(student, optimizer, first, last, cached=None):
    # 用39335个索引保留真实1229批/pass边界，张量仅32×2，完全不运行真实网络。
    trace = []
    for step in range(first, last + 1):
        indices, cached = driver.step_batch_indices(range(39335), step, cached)
        crop_probe = random.Random(1234 + 100000 + step).randint(0, 1000)
        training.seed_everything(1234 + 200000 + step)
        source_noise = torch.randn(32, 2)  # 每步第一份CPU噪声对应真实训练的首份CUDA噪声规则。
        inputs = source_noise + torch.tensor(indices, dtype=torch.float32).unsqueeze(1) / 40000
        optimizer.zero_grad(set_to_none=True)
        (student(inputs) - inputs.detach() * 0.25).abs().mean().backward()
        lr = core.set_optimizer_learning_rate(optimizer, driver.BASE_LR, step)
        optimizer.step()
        trace.append((step, tuple(indices), crop_probe, core.tensor_sha(source_noise), lr))
    return trace


def parent_fixture(student, optimizer, step):
    # 构造结构完整的小型checkpoint；字段含义与真实v3契约保持相同。
    contract = {'kind': 'endpoint_distillation_development_only', 'objective_id': core.OBJECTIVE_ID,
        'source_configuration_template': core.PAPER_STUDENT_CONFIG.as_dict(),
        'optimizer': {'name': 'Adam', 'lr': driver.BASE_LR, 'betas': [0.5, 0.9], 'eps': 1e-8,
            'weight_decay': 0, 'amsgrad': False, 'gradient_clipping': None, 'ema_used': False,
            'learning_rate_policy': core.optimizer_learning_rate_record(optimizer, driver.BASE_LR, 100000, step)},
        'batch_size': 32, 'segment_frames': 128, 'num_workers': 0, 'seed': 1234,
        'per_step_crop_seed': '1234+100000+step',
        'per_step_source_noise_seed': '1234+200000+step; CUDA randn_like first draw',
        'sampler_id': core.ddim_teacher.DDIM_SAMPLER_ID,
        'objective': {'formula': 'mean(abs(y_student_clean - stop_gradient(teacher_DDIM30(z950))))',
            'endpoint_weight': 1.0, 'gan_used': False, 'vocoder_used': False, 'random_score_t_used': False},
        'coupling': 'one q(source_mel,950,noise) state shared exactly by teacher and student; no reverse noise',
        'conditioning': 'reconstruction only: original speaker embedding and source BNF',
        'student_parameterization': 'epsilon; y=(z950-sqrt(1-alpha_bar950)*epsilon_student)/sqrt(alpha_bar950)',
        'development_regression_guard': {'GT_L1_ratio_above_initial': 1.10, 'prediction_RMS_ratio_below_initial': 0.80}}
    payload = {'kind': 'development_checkpoint', 'objective_id': core.OBJECTIVE_ID, 'global_step': step, 'epoch': 0,
        'student': copy.deepcopy(student.state_dict()), 'student_optimizer': copy.deepcopy(optimizer.state_dict()),
        'configuration': core.PAPER_STUDENT_CONFIG.as_dict(), 'inactive_configuration_fields': ['learning_rate'],
        'development': {'mel_l1': 0.2, 'endpoint_l1': 0.1, 'prediction_rms': 1.0},
        'rng_state': training.capture_rng_state(), 'read_asset_sha256': {},
        'actual_optimizer_learning_rate': core.optimizer_learning_rate_record(optimizer, driver.BASE_LR, 100000, step),
        'run_contract_sha256': training.canonical_json_sha256(contract)}
    return payload, contract


class EndpointContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)  # 不抢占同一服务器上的其他CPU诊断资源。
        self.source = Path(driver.__file__)
        self.tree = ast.parse(self.source.read_text(encoding='utf-8'))

    def assert_state_equal(self, left, right):
        # 包括参数、Adam一阶/二阶动量及内部更新计数的逐元素严格相等。
        self.assertEqual(set(left), set(right))
        for key in left:
            if torch.is_tensor(left[key]):
                torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
            elif isinstance(left[key], dict):
                self.assert_state_equal(left[key], right[key])
            else:
                self.assertEqual(left[key], right[key])

    # 覆盖100、600及1229的续点；第一次恢复必须立即构造正确pass顺序。
    def test_global_order_lr_and_first_resumed_batch(self):
        retained = list(range(39335))
        for step in (1, 100, 101, 600, 601, 1229, 1230, 1231, 100000):
            indices, cache = driver.step_batch_indices(retained, step, None)
            pass_index = (step - 1) // 1229
            expected = retained.copy()
            random.Random(1234 + pass_index).shuffle(expected)
            start = ((step - 1) % 1229) * 32
            self.assertEqual(indices, expected[start:start + 32])
            self.assertEqual(cache['pass_index'], pass_index)
            self.assertAlmostEqual(core.learning_rate_for_step(0.0001, step),
                0.0001 * (100000 - step) / 99999, delta=1e-18)
        with self.assertRaises(ValueError):
            driver.step_batch_indices(retained, 0)

    # 先连续运行，再分别在600与1229保存恢复；参数、Adam、采样顺序都应完全一样。
    def test_toy_continuous_equals_save_and_resume(self):
        continuous, continuous_opt = toy_setup()
        expected_trace = toy_updates(continuous, continuous_opt, 1, 1231)
        with tempfile.TemporaryDirectory(prefix='endpoint_resume_cpu_', dir='/tmp') as directory:
            for split in (600, 1229):
                with self.subTest(split=split):
                    first, first_opt = toy_setup()
                    before_trace = toy_updates(first, first_opt, 1, split)
                    payload, contract = parent_fixture(first, first_opt, split)
                    path = Path(directory) / ('parent_%d.pt' % split)
                    training.atomic_torch_save(payload, path)
                    restored = driver.trusted_checkpoint(path, core.sha256_file(path))
                    self.assertEqual(driver.validate_parent_payload(restored, contract)[0], split)
                    second, second_opt = toy_setup()  # 故意重建模型后再恢复，不能靠同一个实例通过测试。
                    second.load_state_dict(restored['student'], strict=True)
                    second_opt.load_state_dict(restored['student_optimizer'])
                    training.restore_rng_state(restored['rng_state'], torch.device('cpu'))
                    after_trace = toy_updates(second, second_opt, split + 1, 1231, cached=None)
                    self.assertEqual(before_trace + after_trace, expected_trace)
                    self.assert_state_equal(second.state_dict(), continuous.state_dict())
                    self.assert_state_equal(second_opt.state_dict(), continuous_opt.state_dict())

    # SHA先检查后反序列化；契约不匹配、旧lr、重置动量或改目标都必须拒绝。
    def test_hash_and_parent_contract_rejections(self):
        student, optimizer = toy_setup()
        toy_updates(student, optimizer, 1, 2)
        payload, contract = parent_fixture(student, optimizer, 2)
        self.assertEqual(driver.validate_parent_payload(payload, contract)[0], 2)
        with tempfile.TemporaryDirectory(prefix='endpoint_resume_sha_', dir='/tmp') as directory:
            path = Path(directory) / 'parent.pt'
            training.atomic_torch_save(payload, path)
            with patch.object(driver, 'load_torch_checkpoint') as loader, self.assertRaises(ValueError):
                driver.trusted_checkpoint(path, '0' * 64)
            loader.assert_not_called()
            for bad in ('abc', 'A' * 64):
                with self.assertRaises(ValueError):
                    driver.trusted_checkpoint(path, bad)
        modified = copy.deepcopy(contract)
        modified['conditioning'] = 'another task'
        with self.assertRaisesRegex(ValueError, 'bound'):
            driver.validate_parent_payload(payload, modified)
        for mutation in ('old_lr', 'empty_adam', 'wrong_adam_step', 'wrong_record', 'changed_objective'):
            changed, changed_contract = copy.deepcopy(payload), copy.deepcopy(contract)
            if mutation == 'old_lr':
                changed['student_optimizer']['param_groups'][0]['lr'] = 0.0002
            elif mutation == 'empty_adam':
                changed['student_optimizer']['state'] = {}
            elif mutation == 'wrong_adam_step':
                next(iter(changed['student_optimizer']['state'].values()))['step'] = torch.tensor(0.0)
            elif mutation == 'wrong_record':
                changed['actual_optimizer_learning_rate']['next_update_learning_rate'] = 0.0002
            else:
                changed_contract['objective']['endpoint_weight'] = 2.0
                changed['run_contract_sha256'] = training.canonical_json_sha256(changed_contract)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                driver.validate_parent_payload(changed, changed_contract)

    # 当前100步的GT结果会触发原保护，绝不能以父段结果为新分母来放行。
    def test_original_s0_guard_is_not_reanchored(self):
        s0 = {'mel_l1': 0.1994135, 'prediction_rms': 0.988}
        parent = {'mel_l1': 0.2257485, 'prediction_rms': 1.0162}
        self.assertFalse(driver.guard_passes(parent, s0))
        self.assertTrue(driver.guard_passes(parent, parent))  # 展示错误重设基准会产生相反判定。
        self.assertFalse(driver.guard_passes({'mel_l1': 0.1, 'prediction_rms': 0.7}, s0))
        with self.assertRaises(ValueError):
            driver.guard_passes({'mel_l1': float('nan'), 'prediction_rms': 1}, s0)

    # 输出目录必须新建且位于实验区；已存在空目录也不能复用。
    def test_existing_or_outside_output_is_rejected(self):
        for path in (driver.ROOT, driver.ROOT / 'exp/diagnostics', Path('/tmp/new-run')):
            with self.assertRaises(ValueError):
                driver.require_new_output(path)
        with tempfile.TemporaryDirectory(prefix='endpoint_output_cpu_', dir=str(driver.ROOT / 'exp/diagnostics')) as directory:
            with self.assertRaises(ValueError):
                driver.require_new_output(Path(directory))
            self.assertEqual(driver.require_new_output(Path(directory) / 'new'), Path(directory) / 'new')

    # 运行真实checkpoint闭包，确认恢复起点和下一步均记录真实LR且禁止覆盖。
    def test_real_checkpoint_saver_preserves_global_step_and_adam(self):
        student, optimizer = toy_setup()
        toy_updates(student, optimizer, 1, 100)
        node = next(node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint')
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        with tempfile.TemporaryDirectory(prefix='endpoint_resume_save_', dir='/tmp') as directory:
            output = Path(directory)
            namespace = {'training': training, 'core': core, 'output': output, 'asset_hashes': {}, 'student': student,
                'optimizer': optimizer, 'contract_sha': 'cpu-contract', 'config': core.PAPER_STUDENT_CONFIG,
                'BASE_LR': driver.BASE_LR, 'DECAY_STEPS': 100000, 'origin': {'initial_checkpoint_sha256': 'cpu-s0'},
                'record': {'parent': {'global_step': 100, 'checkpoint_sha256': 'cpu-parent'}}}
            exec(compile(module, str(self.source), 'exec'), namespace)
            namespace['checkpoint'](100, {'mel_l1': 0.2}, resume_initial=True)
            initial = torch.load(output / 'resume_initial_full.pt', map_location='cpu')
            self.assertEqual(initial['global_step'], 100)
            self.assert_state_equal(initial['student_optimizer'], optimizer.state_dict())
            self.assertEqual(initial['origin']['initial_checkpoint_sha256'], 'cpu-s0')
            toy_updates(student, optimizer, 101, 101)
            namespace['checkpoint'](101, {'mel_l1': 0.19})
            saved = torch.load(output / 'student_step_000101.pt', map_location='cpu')
            self.assertEqual(saved['global_step'], 101)
            self.assertEqual(saved['actual_optimizer_learning_rate']['last_update_learning_rate'], core.learning_rate_for_step(0.0001, 101))
            self.assertEqual(saved['actual_optimizer_learning_rate']['next_update_learning_rate'], core.learning_rate_for_step(0.0001, 102))
            self.assertEqual(saved['configuration']['learning_rate'], 0.0002)  # 旧模板字段继续明确为inactive。
            self.assertIn('learning_rate', saved['inactive_configuration_fields'])
            pointer = json.loads((output / 'latest_checkpoint.json').read_text())
            self.assertEqual(pointer['sha256'], core.sha256_file(output / 'student_step_000101.pt'))
            with self.assertRaises(RuntimeError):
                namespace['checkpoint'](101, {'mel_l1': 0.19})

    # 分段停在123步时带回100步完整窗口及23步累加值；下个窗口仍结束于150。
    def test_monitor_windows_continue_across_partial_boundary(self):
        with tempfile.TemporaryDirectory(prefix='endpoint_resume_monitor_', dir='/tmp') as directory:
            root = Path(directory)
            parent = root / 'parent'
            parent.mkdir()
            contract = parent / 'run_contract.json'
            training.atomic_write_json(contract, {'kind': 'endpoint_distillation_development_only'})
            for step in range(1, 124):
                core.append_json(parent / 'step_losses.jsonl', {'step': step, 'student_total': float(step), 'dist_raw': float(step)})
            history, dependencies = driver.monitor_history(contract, 123)
            core.verify_hashes(dependencies)
            output = root / 'continued'
            output.mkdir()
            monitor = driver.restore_monitor(output, history, 123)
            self.assertEqual(monitor.count, 23)
            self.assertEqual([item['global_step'] for item in monitor.windows], [50, 100])
            for step in range(124, 151):
                monitor.observe(step, 0, {'student_total': step, 'dist_raw': step})
            self.assertEqual(monitor.count, 0)
            self.assertEqual(monitor.windows[-1]['global_step'], 150)
            self.assertEqual(monitor.windows[-1]['mean']['student_total'], sum(range(101, 151)) / 50)
            # 完整性检查能识别缺失历史，不把缺一批的窗口当作同一量尺。
            (parent / 'step_losses.jsonl').write_text('', encoding='utf-8')
            with self.assertRaises(ValueError):
                driver.monitor_history(contract, 123)

    # 保存固定开发预测仍为v2兼容字典；没有新增GAN、裁剪、EMA或自动best选模。
    def test_frozen_math_and_no_extra_update_path(self):
        self.assertEqual(core.sha256_file(driver.v3_path), driver.V3_SHA)
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
            for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        self.assertEqual(calls.count('step'), 1)
        self.assertEqual(calls.count('student_endpoint_loss'), 1)
        self.assertEqual(calls.count('sample_ddim_teacher'), 1)
        for prohibited in ('clip_grad_norm_', 'clip_grad_value_', 'MultiPeriodDiscriminator', 'Generator'):
            self.assertNotIn(prohibited, calls)
        text = self.source.read_text(encoding='utf-8')
        self.assertNotIn('best_development', text)
        self.assertIn("'automatic_model_selection': False", text)
        self.assertIn('range(parent_step + 1, args.max_global_step + 1)', text)
        for call in (node for node in ast.walk(self.tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'atomic_torch_save' and node.args and isinstance(node.args[0], ast.Dict)):
            names = [key.value for key in call.args[0].keys if isinstance(key, ast.Constant)]
            if 'prediction' in names:
                self.assertIn('step', names)


if __name__ == '__main__':
    unittest.main()  # 远程CPU执行；主入口和GPU模型不会被本文件调用。
