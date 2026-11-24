# 在租用服务器 CPU 上真实写入并读回预测文件，覆盖此前未测到的保存接口。
"""Exercise the actual v2 prediction-save calls with the production writer."""

import ast  # 从真实 v2 主函数提取两处保存表达式，避免测试另写一份正确却未被使用的代码。
import tempfile  # 只在租用服务器 /tmp 创建独立测试目录，退出后回收测试产物。
import unittest  # 将真实磁盘往返测试与已有数学测试一起运行。
from pathlib import Path  # 安全定位脚本及当前测试临时文件。

import torch  # 测试张量固定放在 CPU，不占用正在工作的 GPU。
import fastvoicegrad.train as training  # 调用生产 atomic_torch_save，而非简化替身。
import run_endpoint_development_v2 as driver  # 导入不会执行 main 或创建模型。
from fastvoicegrad.components import sha256_file  # 保证 v1 归档没有被这次修复覆盖。


class EndpointPredictionSaveV2Tests(unittest.TestCase):
    # 提取真实代码中的初始保存和后续保存，并核对它们确实都接收字典。
    def save_calls(self):
        path = Path(driver.__file__).resolve()  # 只检查本次实际导入的 v2 文件。
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))  # 解析源码，不执行主训练逻辑。
        found = {}  # 分别保存引用 initial_prediction 和 prediction_cpu 的调用节点。
        for node in ast.walk(tree):  # 检查两个保存位置的真实函数参数。
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):  # 只看属性形式的函数调用。
                continue  # 普通计算表达式与本回归测试无关。
            if node.func.attr != 'atomic_torch_save' or not node.args:  # 跳过其他辅助保存或没有参数的调用。
                continue
            identifiers = {item.id for item in ast.walk(node.args[0]) if isinstance(item, ast.Name)}  # 找到保存的数据来源。
            for name in ('initial_prediction', 'prediction_cpu'):  # 覆盖失败的第 0 步和原定第 100 步。
                if name in identifiers:  # 匹配真实保存表达式中的预测变量。
                    self.assertNotIn(name, found)  # 不接受同名变量出现多处而只测其中一处。
                    self.assertIsInstance(node.args[0], ast.Dict)  # 生产函数要求 mapping，不能再传 bare Tensor。
                    found[name] = node  # 后面直接执行这个实际调用，而非测试重新拼接的保存逻辑。
        self.assertEqual(set(found), {'initial_prediction', 'prediction_cpu'})  # 两处都必须被识别和覆盖。
        return path, found  # 返回源码位置，供编译表达式时保留真实错误定位。

    # 每种精度分别真实写入初始与后续预测，核对文件字段、数值和 dtype。
    def test_actual_initial_and_later_saves_roundtrip(self):
        source, calls = self.save_calls()  # 从当前 v2 获取真正会在训练中执行的两处保存语句。
        with tempfile.TemporaryDirectory(prefix='endpoint_save_cpu_', dir='/tmp') as directory:  # 不写任何旧实验目录。
            output = Path(directory)  # 两份实际检查点只属于这一轮 CPU 回归测试。
            for dtype in (torch.float32, torch.float64):  # 正式 float32 和更高精度都不能被保存函数改写。
                value = torch.linspace(-2, 2, 32 * 80 * 128, dtype=dtype).reshape(32, 80, 128)  # 使用真实预测形状。
                for variable, step in (('initial_prediction', 0), ('prediction_cpu', 100)):  # 第 0 步与第100步都走实际保存表达式。
                    with self.subTest(dtype=str(dtype), step=step):  # 分别报告具体失败的保存阶段及精度。
                        namespace = {'training': training, 'output': output, 'step': step, variable: value}  # 只提供这一行需要的局部变量。
                        expression = ast.Expression(body=calls[variable])  # 不编译 main，避免意外进入训练或加载教师。
                        ast.fix_missing_locations(expression)  # 保留合法 AST 位置供 Python 编译调用。
                        eval(compile(expression, str(source), 'eval'), namespace)  # 真正执行 v2 中的生产原子保存调用。
                        path = output / f'development_prediction_step_{step:06d}.pt'  # 与主训练脚本完全相同的文件命名。
                        self.assertTrue(path.is_file())  # 只有已发布完整文件才算保存成功。
                        restored = torch.load(path, map_location='cpu')  # 文件由本测试刚写入，CPU只读回自己的可信内容。
                        self.assertEqual(set(restored), {'prediction', 'step'})  # 字典包装不能遗漏真实预测或步数。
                        self.assertEqual(restored['step'], step)  # 初始不能误记成100步，后续也不能一直记0。
                        self.assertEqual(restored['prediction'].dtype, dtype)  # 不允许序列化时隐式转精度。
                        self.assertEqual(restored['prediction'].device.type, 'cpu')  # 本测试禁止出现GPU张量。
                        torch.testing.assert_close(restored['prediction'], value, atol=0, rtol=0)  # 数值必须逐元素精确保留。
                self.assertEqual(len(list(output.glob('*.pt'))), 2)  # 没有留下额外的半份发布文件。

    # 单独核对旧版本字节身份，失败实验必须能按原脚本复查。
    def test_v1_archive_is_unchanged(self):
        original = Path(driver.__file__).resolve().with_name('run_endpoint_development.py')  # 指向独立保留的 v1。
        self.assertEqual(sha256_file(original), 'fa27cafeb0ba6ec3a8e4d328136714b502f76559d4b208273c8f61ca6c15011b')

    # 真实执行 v2 中的 checkpoint(0, ...) 函数，覆盖初始模型/Adam/RNG 的保存流程。
    def test_actual_checkpoint_zero_payload_roundtrip(self):
        source = Path(driver.__file__).resolve()  # 仅从实际 v2 脚本提取被测试的保存函数。
        tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))  # 不运行 main 的模型加载逻辑。
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint']
        self.assertEqual(len(functions), 1)  # 必须覆盖真正的唯一嵌套保存函数，不能误选另一实现。
        with tempfile.TemporaryDirectory(prefix='endpoint_checkpoint0_cpu_', dir='/tmp') as directory:
            output = Path(directory)  # 只创建本次测试自己的输出目录。
            student = torch.nn.Linear(2, 2)  # 用四个权重的CPU假学生，不加载或修改真实模型。
            optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, betas=(0.5, 0.9))  # 不调用step，仅核对初始状态保存。
            scores = {'mel_l1': 0.123, 'endpoint_l1': 0.234}  # 明确的测试数据，不写入真实实验日志。
            namespace = {'training': training, 'output': output, 'asset_hashes': {}, 'student': student,
                         'optimizer': optimizer, 'OBJECTIVE_ID': driver.OBJECTIVE_ID, 'config': driver.PAPER_STUDENT_CONFIG,
                         'contract_sha': 'cpu-only-roundtrip-test', 'sha256_file': sha256_file}  # 提供原函数实际捕获的变量。
            module = ast.Module(body=functions, type_ignores=[])  # 只编译保存函数本身，没有GPU主训练入口。
            ast.fix_missing_locations(module)  # 让提取后的AST具有合法位置。
            exec(compile(module, str(source), 'exec'), namespace)  # 定义真实checkpoint函数，不重写其payload。
            namespace['checkpoint'](0, scores)  # 执行第0步真实原子保存与latest指针发布。
            path = output / 'initial_full.pt'  # 失败的v1从未到达这个位置，本测试专门覆盖它。
            restored = torch.load(path, map_location='cpu')  # 读回当前测试自己的可信文件。
            self.assertEqual(restored['global_step'], 0)  # 初始模型不能错误标注为已训练状态。
            self.assertEqual(restored['epoch'], 0)  # 与有限开发实验的初始轮次约定一致。
            self.assertEqual(restored['objective_id'], driver.OBJECTIVE_ID)  # 保留独立端点目标身份。
            self.assertEqual(restored['development'], scores)  # 实际初始读数完整保留。
            self.assertEqual(restored['student_optimizer']['state'], {})  # 未更新的Adam不得虚构动量或步数。
            self.assertIn('rng_state', restored)  # 原随机状态必须随初始化模型一起保存。
            self.assertNotIn('discriminator', restored)  # 纯端点实验不能夹带D状态。
            self.assertNotIn('teacher', restored)  # 教师保持外部只读依赖，不作为被训练模型保存。
            for name, value in student.state_dict().items():  # 检查真实保存函数没有改变学生参数。
                torch.testing.assert_close(restored['student'][name], value, atol=0, rtol=0)  # CPU值和dtype保持一致。
            self.assertTrue((output / 'read_asset_sha256.json').is_file())  # 已读资产账本与模型同步落盘。
            pointer = __import__('json').loads((output / 'latest_checkpoint.json').read_text(encoding='utf-8'))
            self.assertEqual(pointer['sha256'], sha256_file(path))  # latest指针指向完整文件的实际字节摘要。
            self.assertEqual(pointer['step'], 0)  # 指针同样不能标注成已经完成更新。


if __name__ == '__main__':  # 导入测试文件不会自动执行任何项目计算。
    unittest.main()  # 仅通过租用服务器 CPU 命令运行。
