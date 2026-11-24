# 只在租用服务器 CPU 使用小模型验证转换监督，不加载真实 checkpoint、不执行 GPU 训练。
"""CPU checks of the real conversion helper and production batch mathematics."""

import copy  # 复制小模型和 Adam 状态，得到公平的原函数/新函数起点。
import hashlib  # 真实批次 smoke 绑定开发合同、特征张量和随机状态的字节。
import json  # 只读开发名单并输出身份核对结果，不创建实验或修改文件。
import os  # 在导入模型代码之前拒绝本地 Windows 或可见 GPU 的测试环境。
import random  # 检查 Python 随机流也未被重排扰动。
import unittest  # 自动核对更新、身份条件、梯度边界和随机流。
from pathlib import Path  # 只允许真实租用工作区运行测试。
from unittest import mock  # 仅在测试进程记录生产函数输入，或指定恒等测试配对。

if os.name != "posix" or not Path("/root/FsatVoiceGrad/AGENTS.md").is_file():
    raise RuntimeError("Run these project tests only on the rented Matpool server")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CPU tests require CUDA_VISIBLE_DEVICES='' explicitly")

import numpy as np  # 仅核对当前进程 NumPy 随机状态。
import torch  # 所有张量都留在 CPU，使用完整 batch32 形状的小计算图。
from torch import nn  # 几个可手算的参数替代真实大网络。

import conversion_score_step as candidate  # 新辅助函数；此模块没有自动启动入口。
from fastvoicegrad import train as production  # 真正的原 train_batch 用作更新对照。
from fastvoicegrad.config import PAPER_STUDENT_CONFIG  # 使用真实 clean/sqrt/2/45/950 配置。
from fastvoicegrad.objectives import make_student_prediction  # 真实一步反推，不由测试重写公式。
from voicegrad.diffusion import VoiceGradDiffusion  # 使用真实 1000 步扩散表。


class ConditionalEpsilon(nn.Module):
    def __init__(self, coefficient=0.94):
        super().__init__()  # 登记四个小参数，让输入、说话人、内容分别影响输出。
        self.x_coefficient = nn.Parameter(torch.tensor(coefficient))
        self.speaker_coefficient = nn.Parameter(torch.tensor(0.012))
        self.bnf_coefficient = nn.Parameter(torch.tensor(0.025))
        self.bias = nn.Parameter(torch.tensor(0.001))
        self.trace = []  # 仅记录测试输入，绝不属于真实模型或 checkpoint。

    def forward(self, x, timesteps, speaker, bnf):
        self.trace.append({"x": x.detach().clone(), "t": timesteps.detach().clone(),
                           "speaker": speaker.detach().clone(), "bnf": bnf.detach().clone(),
                           "grad_enabled": torch.is_grad_enabled(), "x_requires_grad": x.requires_grad})
        return (self.x_coefficient * x + self.speaker_coefficient * speaker[:, :1, None]
                + self.bnf_coefficient * bnf[:, :1, :] + self.bias)  # 条件取错会真实改变损失和梯度。


class TinyVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.1))  # 测试中冻结，仍允许 mel 输入得到梯度。
        self.trace = []

    def forward(self, mel):
        self.trace.append(mel.detach().clone())  # 查真实/生成波形究竟用了重建还是转换 mel。
        return self.gain * mel.mean(dim=1, keepdim=True)  # 小波形只保留 128 帧，节省 CPU 计算。


class TinyBranch(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain))
        self.bias = nn.Parameter(torch.tensor(0.02))

    def forward(self, waveform):
        feature = waveform * self.gain + self.bias  # 真正可微的判别器特征。
        second = torch.tanh(feature)  # 第二层特征用于验证多层 FM。
        return second.mean(dim=-1), [feature, second]  # 接口符合原串行 D/VJP 函数。


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([TinyBranch(0.7), TinyBranch(0.8)])  # 两个分支验证逐分支累积。

    def loss_branches(self):
        return tuple(self.branches)  # 原生产辅助函数直接调用这两个小分支。


def reset_rng(seed):
    random.seed(seed)  # 默认 Python 流只用于被动核对。
    np.random.seed(seed)  # 默认 NumPy 流也不应被重排消耗。
    torch.manual_seed(seed)  # 模拟原驱动逐绝对步设置训练噪声种子。


def make_fixture(dtype=torch.float64):
    reset_rng(8821)  # 两份测试 fixture 拥有完全相同的 CPU 输入。
    batch = {"mel": torch.randn(32, 80, 128, dtype=dtype),
             "bnf": torch.randn(32, 144, 128, dtype=dtype),
             "spk_emb": torch.randn(32, 256, dtype=dtype),
             "spk_name": [f"p{i // 2:03d}" for i in range(32)]}  # 同身份两条，避免把索引差异当身份差异。
    teacher = ConditionalEpsilon(0.97).to(dtype=dtype).eval()
    student = ConditionalEpsilon(0.94).to(dtype=dtype).train()
    vocoder = TinyVocoder().to(dtype=dtype).eval()
    discriminator = TinyDiscriminator().to(dtype=dtype).train()
    for module in (teacher, vocoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)  # 与真实任务相同的冻结边界。
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    diffusion = VoiceGradDiffusion(n_levels=1000, offset=0.008)  # 实际起点 950；不缩短时间表。
    mean = torch.linspace(-7.5, -4.0, 80, dtype=dtype).view(1, 80, 1)  # 非平凡反归一化也被测试。
    std = torch.linspace(0.7, 2.5, 80, dtype=dtype).view(1, 80, 1)
    args = (batch, teacher, student, vocoder, discriminator, diffusion, optimizer, d_optimizer,
            mean, std, torch.device("cpu"), PAPER_STUDENT_CONFIG)  # 和生产函数完全相同的参数顺序。
    return args


def model_optimizer_state(args):
    return {"student": copy.deepcopy(args[2].state_dict()), "discriminator": copy.deepcopy(args[4].state_dict()),
            "student_optimizer": copy.deepcopy(args[6].state_dict()),
            "discriminator_optimizer": copy.deepcopy(args[7].state_dict())}  # 双 Adam 的 step/moment 都比较。


class ConversionScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)  # 仅本 CPU 测试进程限线程，不改正在运行的训练进程设置。

    def assert_nested_close(self, actual, expected, *, atol, rtol):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            if expected.is_floating_point() and expected.numel():
                return float(torch.max(torch.abs(actual - expected)))  # 汇总最大实测舍入误差。
            return 0.0
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            return max([self.assert_nested_close(actual[key], expected[key], atol=atol, rtol=rtol)
                        for key in expected] + [0.0])
        if isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            return max([self.assert_nested_close(a, b, atol=atol, rtol=rtol)
                        for a, b in zip(actual, expected)] + [0.0])
        self.assertEqual(actual, expected)  # Adam 的学习率、beta、步数和布尔项不允许近似相等。
        return 0.0

    def test_independent_shuffle_preserves_all_default_rng_and_allows_fixed_points(self):
        reset_rng(713)
        py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        permutation = candidate.sample_target_permutation(32, permutation_seed=991000, global_step=601)
        expected = torch.randperm(32, generator=torch.Generator().manual_seed(991601))  # 普通 randperm，没有拒绝采样。
        torch.testing.assert_close(permutation, expected, atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), cpu_state, atol=0, rtol=0)
        self.assertEqual(random.getstate(), py_state)
        after_numpy = np.random.get_state()
        np.testing.assert_array_equal(after_numpy[1], np_state[1])
        self.assertEqual(after_numpy[0], np_state[0])
        self.assertEqual(after_numpy[2:], np_state[2:])
        self.assertEqual(candidate.sample_target_permutation(1, permutation_seed=1, global_step=1).tolist(), [0])
        stats = candidate.permutation_metrics(torch.tensor([1, 0, 2, 3]), ["p1", "p1", "p2", "p3"])
        self.assertEqual(stats["conversion_different_speaker_fraction"], 0.0)  # 两个索引互换仍可能是同一人。
        self.assertEqual(stats["conversion_fixed_index_fraction"], 0.5)

    def test_conversion_score_has_only_conversion_student_gradient_and_detached_teacher(self):
        args = make_fixture()
        batch, teacher, student, _, _, diffusion = args[:6]
        reset_rng(770)
        source_noise, score_noise = torch.randn_like(batch["mel"]), torch.randn_like(batch["mel"])
        timesteps = torch.randint(1, 1001, (32,))
        reconstruction, _ = make_student_prediction(student, diffusion, batch["mel"], batch["spk_emb"], batch["bnf"], 950, source_noise)
        reconstruction.retain_grad()  # 单独存在的重建图不能意外收到转换 score 梯度。
        permutation = torch.roll(torch.arange(32), shifts=1)  # 测试强制非恒等配对，实际调用仍用普通 randperm。
        loss, target, converted = candidate.conversion_score_loss(
            student, teacher, diffusion, batch["mel"], batch["spk_emb"], batch["bnf"],
            source_noise, score_noise, timesteps, 950, permutation,
        )
        converted.retain_grad()
        self.assertFalse(target.requires_grad)
        self.assertIsNone(target.grad_fn)
        self.assertEqual(len(teacher.trace), 1)  # 没有偷偷再计算一次重建教师目标。
        self.assertFalse(teacher.trace[0]["grad_enabled"])
        self.assertFalse(teacher.trace[0]["x_requires_grad"])
        torch.testing.assert_close(teacher.trace[0]["speaker"], batch["spk_emb"][permutation], atol=0, rtol=0)
        torch.testing.assert_close(teacher.trace[0]["bnf"], batch["bnf"], atol=0, rtol=0)
        teacher_input = diffusion.q_sample(converted.detach(), timesteps, noise=score_noise)
        torch.testing.assert_close(teacher.trace[0]["x"], teacher_input, atol=0, rtol=0)
        loss.backward()  # 真实生产 score 函数的导数与解析 L1 stop-gradient 导数比对。
        weights = diffusion.sqrt_alphas_cumprod[timesteps].to(converted.dtype).view(32, 1, 1)
        expected_gradient = weights * torch.sign(converted.detach() - target) / converted.numel()
        torch.testing.assert_close(converted.grad, expected_gradient, atol=1e-20, rtol=1e-12)
        self.assertIsNone(reconstruction.grad)  # 重建 score 没有参与梯度。
        self.assertTrue(any(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)) for parameter in student.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_nonidentity_batch_keeps_fm_on_reconstruction_and_updates_only_s_d(self):
        args = make_fixture()
        batch, teacher, student, vocoder, discriminator, diffusion = args[:6]
        teacher_before, vocoder_before = copy.deepcopy(teacher.state_dict()), copy.deepcopy(vocoder.state_dict())
        update_before = model_optimizer_state(args)
        reset_rng(351)
        expected_source = torch.randn_like(batch["mel"])
        expected_score = torch.randn_like(batch["mel"])
        expected_t = torch.randint(1, 1001, (32,))
        expected_rng = torch.get_rng_state().clone()  # 后续模型均无随机层，所以实际结束状态必须精确等于这里。
        reset_rng(351)
        py_before, np_before = random.getstate(), np.random.get_state()
        permutation = torch.roll(torch.arange(32), 1)
        predictions, noises, fm_inputs = [], [], []  # 捕获 helper 实際调用的生产函数，不替换其计算结果。
        original_make = candidate.make_student_prediction
        original_fm = production.serial_student_discriminator_gradient

        def record_prediction(*positional, **keywords):
            result = original_make(*positional, **keywords)
            predictions.append(result[0].detach().clone())
            noises.append(positional[6].detach().clone())  # 第七个参数是同源起始噪声。
            return result

        def record_fm(d, real, fake, weight):
            fm_inputs.append((real.detach().clone(), fake.detach().clone(), weight))
            return original_fm(d, real, fake, weight)  # 真正运行原多分支波形 VJP。

        with mock.patch.object(candidate, "sample_target_permutation", return_value=permutation), \
             mock.patch.object(candidate, "make_student_prediction", side_effect=record_prediction), \
             mock.patch.object(production, "serial_student_discriminator_gradient", side_effect=record_fm):
            losses = candidate.train_batch_conversion_score(*args, permutation_seed=991000, global_step=601)
        self.assertEqual(len(student.trace), 3)  # D 无梯度重建、学生重建、学生转换各一次。
        self.assertEqual([entry["grad_enabled"] for entry in student.trace], [False, True, True])
        self.assertEqual(len(teacher.trace), 1)
        self.assertEqual(len(vocoder.trace), 3)  # 没有为转换新增波形 GAN 或 FM。
        for noise in noises:
            torch.testing.assert_close(noise, expected_source, atol=0, rtol=0)
        for index in (0, 1):
            torch.testing.assert_close(student.trace[index]["speaker"], batch["spk_emb"], atol=0, rtol=0)
        torch.testing.assert_close(student.trace[2]["speaker"], batch["spk_emb"][permutation], atol=0, rtol=0)
        for entry in student.trace:
            torch.testing.assert_close(entry["bnf"], batch["bnf"], atol=0, rtol=0)
            torch.testing.assert_close(entry["x"], student.trace[0]["x"], atol=0, rtol=0)  # 三次前向的 x950 完全共用。
        torch.testing.assert_close(teacher.trace[0]["t"], expected_t, atol=0, rtol=0)
        torch.testing.assert_close(teacher.trace[0]["x"], diffusion.q_sample(predictions[2], expected_t, noise=expected_score), atol=0, rtol=0)
        torch.testing.assert_close(teacher.trace[0]["speaker"], student.trace[2]["speaker"], atol=0, rtol=0)
        real_mel = production.denormalize_mel(batch["mel"], args[8], args[9])
        rec_mel = production.denormalize_mel(predictions[1], args[8], args[9])
        torch.testing.assert_close(vocoder.trace[0], real_mel, atol=0, rtol=0)
        torch.testing.assert_close(vocoder.trace[1], rec_mel, atol=0, rtol=0)
        torch.testing.assert_close(vocoder.trace[2], rec_mel, atol=0, rtol=0)
        torch.testing.assert_close(fm_inputs[0][0], vocoder.gain.detach() * real_mel.mean(1, keepdim=True), atol=0, rtol=0)
        torch.testing.assert_close(fm_inputs[0][1], vocoder.gain.detach() * rec_mel.mean(1, keepdim=True), atol=0, rtol=0)
        self.assertEqual(fm_inputs[0][2], 2.0)
        self.assertFalse(torch.equal(predictions[1], predictions[2]))  # 测试条件确实有影响，不是无条件假网络。
        self.assert_nested_close(teacher.state_dict(), teacher_before, atol=0, rtol=0)
        self.assert_nested_close(vocoder.state_dict(), vocoder_before, atol=0, rtol=0)
        after = model_optimizer_state(args)
        self.assertGreater(max(float(torch.max(torch.abs(after["student"][key] - update_before["student"][key]))) for key in after["student"]), 0)
        self.assertGreater(max(float(torch.max(torch.abs(after["discriminator"][key] - update_before["discriminator"][key]))) for key in after["discriminator"]), 0)
        for optimizer in args[6:8]:
            self.assertTrue(all(int(state["step"]) == 1 for state in optimizer.state.values()))
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.0002)
            self.assertEqual(optimizer.param_groups[0]["betas"], (0.5, 0.9))
        for model in (teacher, student, vocoder, discriminator):
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in discriminator.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
        self.assertEqual(random.getstate(), py_before)
        np.testing.assert_array_equal(np.random.get_state()[1], np_before[1])
        self.assertEqual(np.random.get_state()[2:], np_before[2:])
        self.assertEqual(losses["conversion_different_speaker_fraction"], 0.5)
        self.assertEqual(losses["conversion_fixed_index_fraction"], 0.0)

    def test_identity_permutation_matches_production_model_and_two_adam_updates(self):
        for dtype, atol, rtol in ((torch.float64, 1e-12, 1e-11), (torch.float32, 1e-7, 2e-6)):
            original, changed = make_fixture(dtype), make_fixture(dtype)
            maximum = 0.0  # 分别报告两种 dtype 下三步后模型与所有 Adam 状态的最大绝对差。
            for step in range(1, 4):
                reset_rng(900 + step)
                expected = production.train_batch(*original)  # 真正原生产 train_batch，而非另写简化基准。
                expected_rng = torch.get_rng_state().clone()
                reset_rng(900 + step)
                with mock.patch.object(candidate, "sample_target_permutation", return_value=torch.arange(32)):
                    actual = candidate.train_batch_conversion_score(*changed, permutation_seed=991000, global_step=step)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)  # 随机流要求逐位相等。
                for key in expected:
                    torch.testing.assert_close(torch.tensor(actual[key], dtype=dtype), torch.tensor(expected[key], dtype=dtype), atol=atol, rtol=rtol)
                difference = self.assert_nested_close(model_optimizer_state(changed), model_optimizer_state(original), atol=atol, rtol=rtol)
                maximum = max(maximum, difference)
                self.assertEqual(actual["conversion_different_speaker_fraction"], 0.0)
                self.assertEqual(actual["conversion_fixed_index_fraction"], 1.0)
            actual_state, expected_state = model_optimizer_state(changed), model_optimizer_state(original)
            details = {}  # Adam 二阶动量可能量纲较大，不能把它的绝对差误报成权重差。
            for category in ("student", "discriminator"):
                details[category + "_parameter_max_abs"] = max(float((actual_state[category][key] - expected_state[category][key]).abs().max()) for key in expected_state[category])
            for category in ("student_optimizer", "discriminator_optimizer"):
                for moment in ("exp_avg", "exp_avg_sq"):
                    pairs = [(actual_state[category]["state"][key][moment], expected_state[category]["state"][key][moment]) for key in expected_state[category]["state"]]
                    details[category + "_" + moment + "_max_abs"] = max(float((a - b).abs().max()) for a, b in pairs)
                    details[category + "_" + moment + "_max_relative"] = max(float(((a - b).abs() / b.abs().clamp_min(1e-30)).max()) for a, b in pairs)
            print("IDENTITY_EQUIVALENCE " + json.dumps({"dtype": str(dtype), "steps": 3, "atol": atol, "rtol": rtol,
                  "max_model_or_Adam_abs_difference": maximum, "details": details}, sort_keys=True))

    def test_real_global601_batch_metadata_and_read_shuffle_rng_smoke(self):
        import run_acdd_development as baseline  # 只导入既有批次读取函数；不调用 main 或构造真实模型。
        root = Path("/root/FsatVoiceGrad")
        contract_path = root / "exp/diagnostics/student_development_20260906/development_contract.json"
        self.assertEqual(hashlib.sha256(contract_path.read_bytes()).hexdigest(), "08737551d34e00ba0b80a5e34adbf73fcfc6c70ece6bc8dc4a6c466e9b87afb1")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        retained = list(contract["retained_train_indices"])
        self.assertEqual(len(retained), 39335)
        order = list(retained)
        random.Random(1234).shuffle(order)  # global601 位于第一遍1229个批次中，重建原基线相同打乱。
        indices = order[600 * 32:601 * 32]
        self.assertFalse(set(indices) & set(contract["excluded_train_indices"]))  # 52条开发句不能进入候选训练。
        reset_rng(5577)
        py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        dataset = baseline.VoiceGradDataset(root / "data", "train", segment_length=128,
                                           expected_speakers=100, bnf_alignment=PAPER_STUDENT_CONFIG.bnf_alignment)
        batch = baseline.batch_for(dataset, indices, 1234 + 100000 + 601)  # 真正读取相同32条特征、相同裁剪。
        self.assertEqual(len(dataset), 39387)
        self.assertEqual(tuple(batch["mel"].shape), (32, 80, 128))
        self.assertEqual(tuple(batch["bnf"].shape), (32, 144, 128))
        self.assertEqual(tuple(batch["spk_emb"].shape), (32, 256))
        self.assertEqual(len(batch["spk_name"]), 32)
        permutation = candidate.sample_target_permutation(32, permutation_seed=991000, global_step=601)
        metrics = candidate.permutation_metrics(permutation, batch["spk_name"])
        self.assertEqual(random.getstate(), py_state)  # 真实 batch_for 恢复 Python 裁剪流。
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), cpu_state, atol=0, rtol=0)  # 读特征及重排都不改变默认训练噪声流。
        tensor_hashes = {key: hashlib.sha256(batch[key].contiguous().numpy().tobytes()).hexdigest() for key in ("mel", "bnf", "spk_emb")}
        print("REAL_BATCH_METADATA_SMOKE " + json.dumps({"global_step": 601, "permutation_seed": 991000,
              "indices": indices, "spk_name": batch["spk_name"], "permutation": permutation.tolist(),
              "default_python_numpy_torch_rng_unchanged": True, "feature_sha256": tensor_hashes, **metrics}, sort_keys=True))

    def test_bad_contract_metadata_and_optimizer_are_rejected_before_updates(self):
        args = make_fixture()
        original = model_optimizer_state(args)
        invalid_batch = dict(args[0])
        del invalid_batch["spk_name"]  # 没有身份元数据不能用索引差异猜测转换比例。
        with self.assertRaises(ValueError):
            candidate.train_batch_conversion_score(invalid_batch, *args[1:], permutation_seed=1, global_step=1)
        with self.assertRaises(ValueError):
            candidate.train_batch_conversion_score(*args[:6], args[7], args[6], *args[8:], permutation_seed=1, global_step=1)
        bad_config = copy.copy(PAPER_STUDENT_CONFIG)
        object.__setattr__(bad_config, "score_target", "legacy_adjacent")  # 只污染测试副本，生产 frozen dataclass 未改。
        with self.assertRaises(ValueError):
            candidate.train_batch_conversion_score(*args[:-1], bad_config, permutation_seed=1, global_step=1)
        with mock.patch.object(candidate, "sample_target_permutation", return_value=torch.zeros(32, dtype=torch.long)):
            with self.assertRaises(ValueError):
                candidate.train_batch_conversion_score(*args, permutation_seed=1, global_step=1)
        self.assert_nested_close(model_optimizer_state(args), original, atol=0, rtol=0)
        self.assertEqual(len(args[1].trace), 0)
        self.assertEqual(len(args[2].trace), 0)  # 所有错误在模型前向和两次 optimizer.step 之前发现。


if __name__ == "__main__":
    unittest.main()  # 显式远端 CPU 命令才运行，导入不会触发训练。
