# 租用服务器CPU回归测试：仅验证v3学习率策略、记录与保存，绝不启动真实教师或训练任务。
"""Check endpoint-v3 LR arithmetic, unchanged objective and actual save/log paths."""

import argparse  # 模拟已解析的显式学习率参数。
import ast  # 从真实脚本提取调用，避免测试另写正确的保存函数却漏掉生产调用错误。
import copy  # 保留Adam真实状态，核对调学习率没有更改动量。
import io  # 捕获预期的参数错误文本。
import json  # 读回实际逐步记录和检查点指针。
import tempfile  # 仅在服务器/tmp创建可回收的CPU测试输出。
import unittest  # 提供断言与测试发现。
from contextlib import redirect_stderr  # 抑制预期的argparse帮助输出。
from pathlib import Path  # 使用实际导入脚本和明确临时目录。
from types import SimpleNamespace  # 为提取的日志表达式提供固定时钟等少量上下文。
from unittest.mock import patch  # Python3.8需显式导入mock子模块，不能依赖先前测试间接导入。

import torch  # 只执行CPU小张量与微型Adam验证。
import fastvoicegrad.train as training  # 复用真实原子保存接口。
import run_endpoint_development_v2 as old  # 旧v2只能读取，不能修改其脚本或结果。
import run_endpoint_development_v3 as driver  # 导入不会进入main训练入口。


class EndpointV3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)  # CPU测试不占用其他诊断的并行资源。
        self.source = Path(driver.__file__).resolve()
        self.tree = ast.parse(self.source.read_text(encoding='utf-8'))
        self.old_tree = ast.parse(Path(old.__file__).read_text(encoding='utf-8'))

    # 精确核对第1、600、N和N+1步，短期试验不能提前退火到0。
    def test_schedule_endpoints_and_short_run_horizon(self):
        for base in (0.00005, 0.0001):
            with self.subTest(base=base):
                self.assertEqual(driver.learning_rate_for_step(base, 1), base)
                # 手算和实现的乘除顺序不同，可相差一个float64末位，仅容许1e-18舍入。
                self.assertAlmostEqual(driver.learning_rate_for_step(base, 600), base * (100000 - 600) / (100000 - 1), delta=1e-18)
                self.assertGreater(driver.learning_rate_for_step(base, 600), base * 0.99)
                self.assertEqual(driver.learning_rate_for_step(base, 100000), 0.0)
                self.assertEqual(driver.learning_rate_for_step(base, 100001), 0.0)
                self.assertGreater(driver.learning_rate_for_step(base, 599), driver.learning_rate_for_step(base, 600))
        # 错误步数、旧lr和把日程压成600步都应拒绝。
        for base, step, horizon in ((0.0002, 1, 100000), (0.0001, 0, 100000), (0.0001, True, 100000), (0.0001, 1, 600)):
            with self.assertRaises(ValueError):
                driver.learning_rate_for_step(base, step, horizon)

    # 直接编译main开头的真实参数解析段，不执行环境检查、模型加载或加锁。
    def parse_actual_cli(self, arguments):
        main = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        setup = []
        for statement in main.body:
            setup.append(statement)
            if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'args' for target in statement.targets):
                break
        tree = ast.Module(body=setup, type_ignores=[])
        ast.fix_missing_locations(tree)
        namespace = {'argparse': argparse, 'Path': Path, '__doc__': driver.__doc__, 'LR_DECAY_STEPS': driver.LR_DECAY_STEPS}
        with patch('sys.argv', ['run_endpoint_development_v3.py', *arguments]):
            exec(compile(tree, str(self.source), 'exec'), namespace)
        return namespace['args']

    # 学习率必须显式声明，退火终点固定100000，与600/3600的运行预算分开。
    def test_cli_requires_lr_and_rejects_shortened_annealing(self):
        common = ['--development-contract', '/tmp/contract.json', '--output-dir', '/tmp/new-output']
        for lr in ('0.00005', '0.0001'):
            args = self.parse_actual_cli(common + ['--learning-rate', lr])
            self.assertEqual(args.learning_rate, float(lr))
            self.assertEqual((args.lr_decay_steps, args.max_steps, args.max_seconds), (100000, 600, 3600))
        for extra in ([], ['--learning-rate', '0.0002'], ['--learning-rate', '0.0001', '--lr-decay-steps', '600']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.parse_actual_cli(common + extra)
            self.assertEqual(error.exception.code, 2)

    # 仅调整lr时，已有Adam动量、参数、梯度及其他配置均须原样保留。
    def test_setting_lr_changes_no_other_adam_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.0001, betas=(0.5, 0.9), eps=1e-8, weight_decay=0, amsgrad=False)
        parameter.square().sum().backward()
        optimizer.step()  # 只更新两个CPU标量，构造真实非空动量以检验调度器隔离。
        before, value, gradient = copy.deepcopy(optimizer.state_dict()), parameter.detach().clone(), parameter.grad.clone()
        actual = driver.set_optimizer_learning_rate(optimizer, 0.0001, 600)
        after = optimizer.state_dict()
        self.assertEqual(actual, driver.learning_rate_for_step(0.0001, 600))
        for prior, current in zip(before['param_groups'], after['param_groups']):
            self.assertEqual({key: item for key, item in prior.items() if key != 'lr'}, {key: item for key, item in current.items() if key != 'lr'})
        for parameter_id, state in before['state'].items():
            for key, item in state.items():
                if torch.is_tensor(item):
                    torch.testing.assert_close(after['state'][parameter_id][key], item, atol=0, rtol=0)
                else:
                    self.assertEqual(after['state'][parameter_id][key], item)
        torch.testing.assert_close(parameter, value, atol=0, rtol=0)
        torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)

    # 用源码AST证明数学目标、采样调用、固定数据与保护条件没有在v3一起更改。
    def test_objective_sampler_data_and_guards_are_unchanged(self):
        names = ('tensor_sha', 'append_json', 'verify_hashes', 'validate_development', 'batch_for', 'student_endpoint_loss', 'evaluate_fixed')
        for name in names:
            prior = next(node for node in self.old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            current = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(prior), ast.dump(current), name)
        # 关键函数的所有调用也保持完全相同，包括源加噪和两处教师DDIM目标构造。
        for name in ('sample_ddim_teacher', 'q_sample', 'seed_everything', 'batch_for', 'student_endpoint_loss', 'evaluate_fixed'):
            collect = lambda tree: sorted(ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == name) or (isinstance(node.func, ast.Attribute) and node.func.attr == name)))
            self.assertEqual(collect(self.old_tree), collect(self.tree), name)
        # Adam的参数、动量、eps等关键字不能改变，唯一差异应是lr来自显式参数。
        find_adam = lambda tree: next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'Adam')
        prior, current = copy.deepcopy(find_adam(self.old_tree)), copy.deepcopy(find_adam(self.tree))
        prior.keywords = [item for item in prior.keywords if item.arg != 'lr']
        current.keywords = [item for item in current.keywords if item.arg != 'lr']
        self.assertEqual(ast.dump(prior), ast.dump(current))
        # 禁止新增裁剪、EMA、优化器更新次数或其他学生训练路径。
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else '' for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        self.assertNotIn('clip_grad_norm_', calls)
        self.assertNotIn('clip_grad_value_', calls)
        self.assertEqual(calls.count('step'), 1)
        # 开发保护判定的条件表达式不变，避免学习率修改同时放宽停止门槛。
        guards = lambda tree: [ast.dump(node.test) for node in ast.walk(tree) if isinstance(node, ast.If) and any(isinstance(item, ast.Constant) and item.value == 'development_regression_guard' for item in ast.walk(node))]
        self.assertEqual(guards(self.old_tree), guards(self.tree))

    # 执行v3真实checkpoint闭包，确认初始化和第600次CPU更新后记录实际lr。
    def test_actual_checkpoint_and_v2_prediction_saver_roundtrip(self):
        checkpoint = next(node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint')
        with tempfile.TemporaryDirectory(prefix='endpoint_v3_cpu_', dir='/tmp') as directory:
            output = Path(directory)
            student = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(student.parameters(), lr=0.00005, betas=(0.5, 0.9), eps=1e-8, weight_decay=0, amsgrad=False)
            namespace = {'training': training, 'output': output, 'asset_hashes': {}, 'student': student, 'optimizer': optimizer,
                'OBJECTIVE_ID': driver.OBJECTIVE_ID, 'config': driver.PAPER_STUDENT_CONFIG, 'contract_sha': 'cpu-only-v3',
                'sha256_file': driver.sha256_file, 'args': argparse.Namespace(learning_rate=0.00005, lr_decay_steps=100000),
                'optimizer_learning_rate_record': driver.optimizer_learning_rate_record}
            module = ast.Module(body=[checkpoint], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(self.source), 'exec'), namespace)
            namespace['checkpoint'](0, {'mel_l1': 0.2})
            initial = torch.load(output / 'initial_full.pt', map_location='cpu')
            self.assertIsNone(initial['actual_optimizer_learning_rate']['last_update_learning_rate'])
            self.assertEqual(initial['actual_optimizer_learning_rate']['next_update_learning_rate'], 0.00005)
            self.assertEqual(initial['student_optimizer']['state'], {})
            # 微型CPU模型真实完成600次更新，保存游标与Adam内部计数一致。
            for step in range(1, 601):
                optimizer.zero_grad(set_to_none=True)
                student(torch.ones(1, 2)).square().mean().backward()
                driver.set_optimizer_learning_rate(optimizer, 0.00005, step)
                optimizer.step()
            namespace['checkpoint'](600, {'mel_l1': 0.1})
            restored = torch.load(output / 'student_step_000600.pt', map_location='cpu')
            actual = restored['actual_optimizer_learning_rate']
            self.assertEqual(actual['last_update_learning_rate'], driver.learning_rate_for_step(0.00005, 600))
            self.assertEqual(actual['next_update_learning_rate'], driver.learning_rate_for_step(0.00005, 601))
            self.assertEqual(restored['student_optimizer']['param_groups'][0]['lr'], actual['last_update_learning_rate'])
            self.assertEqual(restored['configuration']['learning_rate'], 0.0002)
            self.assertIn('learning_rate', restored['inactive_configuration_fields'])
            self.assertEqual((actual['gradient_clipping'], actual['ema_used']), (None, False))
            self.assertNotIn('discriminator', restored)
            pointer = json.loads((output / 'latest_checkpoint.json').read_text(encoding='utf-8'))
            self.assertEqual(pointer['sha256'], driver.sha256_file(output / 'student_step_000600.pt'))
            # 两处prediction保存沿用v2的字典包装，执行真实表达式并读回。
            for variable, step in (('initial_prediction', 0), ('prediction_cpu', 600)):
                calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'atomic_torch_save' and node.args and any(isinstance(item, ast.Name) and item.id == variable for item in ast.walk(node.args[0]))]
                self.assertEqual(len(calls), 1)
                self.assertIsInstance(calls[0].args[0], ast.Dict)
                value = torch.ones(2, 3, 4)
                expression = ast.Expression(body=calls[0])
                ast.fix_missing_locations(expression)
                eval(compile(expression, str(self.source), 'eval'), {'training': training, 'output': output, 'step': step, variable: value})
                saved = torch.load(output / f'development_prediction_step_{step:06d}.pt', map_location='cpu')
                self.assertEqual(saved['step'], step)
                torch.testing.assert_close(saved['prediction'], value, atol=0, rtol=0)

    # 实际每步日志表达式必须保存刚写入Adam的lr，不能继续沿用模板0.0002。
    def test_actual_step_log_records_applied_learning_rate(self):
        call = next(node for node in ast.walk(self.tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'append_json' and any(isinstance(item, ast.Constant) and item.value == 'step_losses.jsonl' for item in ast.walk(node.args[0])))
        with tempfile.TemporaryDirectory(prefix='endpoint_v3_log_', dir='/tmp') as directory:
            output = Path(directory)
            expected = driver.learning_rate_for_step(0.0001, 600)
            namespace = {'append_json': driver.append_json, 'output': output, 'step': 600, 'time': SimpleNamespace(monotonic=lambda: 10.0), 'started': 1.0,
                'teacher_seconds': 0.25, 'grad_norm': 0.5, 'losses': {'student_total': 0.1, 'dist_raw': 0.1}, 'actual_lr': expected,
                'args': argparse.Namespace(learning_rate=0.0001, lr_decay_steps=100000), 'LR_FORMULA': driver.LR_FORMULA,
                'OPTIMIZER_POLICY_ID': driver.OPTIMIZER_POLICY_ID, 'OBJECTIVE_ID': driver.OBJECTIVE_ID}
            expression = ast.Expression(body=call)
            ast.fix_missing_locations(expression)
            eval(compile(expression, str(self.source), 'eval'), namespace)
            saved = json.loads((output / 'step_losses.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(saved['learning_rate'], expected)
            self.assertEqual(saved['lr_decay_steps'], 100000)
            self.assertEqual(saved['lr_formula'], driver.LR_FORMULA)
            self.assertEqual(saved['objective_id'], old.OBJECTIVE_ID)

    # 当前优化器值与宣称日程不一致时，checkpoint元数据不能伪造已正确执行。
    def test_optimizer_record_rejects_wrong_actual_lr(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=0.0002)
        with self.assertRaises(ValueError):
            driver.optimizer_learning_rate_record(optimizer, 0.0001, 100000, 0)

    # 旧v1/v2的程序证据必须完整保留；这些测试也不会修改任何旧实验输出。
    def test_frozen_v1_and_v2_are_unchanged(self):
        directory = self.source.parent
        self.assertEqual(driver.sha256_file(directory / 'run_endpoint_development.py'), 'fa27cafeb0ba6ec3a8e4d328136714b502f76559d4b208273c8f61ca6c15011b')
        self.assertEqual(driver.sha256_file(directory / 'run_endpoint_development_v2.py'), 'e198a87b6515bde8b6cdbf9ec5e41dd6081e801a4c4170e92c5c4d1e6fc4d933')


if __name__ == '__main__':
    unittest.main()
