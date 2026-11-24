# 只在Matpool运行的小型CPU测试；不读取真实模型权重、不调用任何GPU。
"""Check isolated candidate identities and their shared noisy input."""

# 导入路径仅指向独立工具目录，不替换正式推理模块。
import argparse
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# 测试时由远程CPU执行这些小张量运算。
import torch
from torch import nn

# 固定项目根和工具目录，避免意外导入Windows上的开发副本。
ROOT = Path("/root/FsatVoiceGrad")
TOOLS = ROOT / "exp/diagnostics/student_performance_tools_20260906"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import evaluate_development_candidates as candidates
from voicegrad.diffusion import VoiceGradDiffusion


# 可计数教师把测试聚焦在采样器接口，不需要加载真实VoiceGrad权重。
class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        # 参数仅用于检查推理没有创建梯度或更新数值。
        self.value = nn.Parameter(torch.tensor(0.2), requires_grad=False)
        self.calls = 0
        self.first_input = None

    def forward(self, x, timesteps, speaker_embedding, bnf):
        self.calls += 1
        if self.first_input is None:
            self.first_input = x.detach().clone()
        return torch.zeros_like(x) + self.value


# 所有单元测试均使用CPU小张量，并固定自己的随机生成器。
class DevelopmentCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.diffusion = VoiceGradDiffusion()
        self.source = torch.randn(2, 2, 8, generator=torch.Generator().manual_seed(21))
        self.speaker = torch.zeros(2, 3)
        self.bnf = torch.ones(2, 4, 8)
        self.seed = 2026090600

    # 建立完整CLI字段，测试只改与各失败情形直接有关的部分。
    def arguments(self, **changes):
        values = {"system": "teacher500_ddim30", "checkpoint": None, "checkpoint_sha256": None, "run_contract": None, "label": None, "epoch": None, "global_step": None}
        values.update(changes)
        return argparse.Namespace(**values)

    # DDIM教师名字、模型SHA和采样步数必须明确区别于正式教师30步。
    def test_ddim_teacher_identity_is_fixed_and_distinct(self):
        label, spec = candidates.candidate_spec(self.arguments())
        self.assertEqual(label, "teacher500_ddim30")
        self.assertEqual((spec["steps"], spec["epoch"], spec["global_step"]), (30, 500, 615000))
        self.assertEqual(spec["sha256"], candidates.TEACHER_SHA)
        self.assertEqual(spec["sampling"], "ddim_eta0_shared_noisy_start_v1")
        # 提供学生参数会制造身份歧义，不能被悄悄忽略。
        with self.assertRaises(ValueError):
            candidates.candidate_spec(self.arguments(label="student_typo"))

    # 开发600步权重必须用epoch0/step600登记，不被解释为旧完整训练归档。
    def test_student_identity_requires_explicit_checkpoint_cursor_and_sha(self):
        arguments = self.arguments(system="student_k1", checkpoint=ROOT / "exp/diagnostics/example/student_step_000600.pt", checkpoint_sha256="a" * 64, label="student_acdd600", epoch=0, global_step=600)
        label, spec = candidates.candidate_spec(arguments)
        self.assertEqual(label, "student_acdd600")
        self.assertEqual((spec["epoch"], spec["global_step"], spec["steps"]), (0, 600, 1))
        # 缺失摘要、缩短摘要、目录穿越和漏写游标均应在加载前被拒绝。
        for field, invalid in (("checkpoint_sha256", None), ("checkpoint_sha256", "a" * 12), ("label", "../teacher500_k30"), ("epoch", None), ("global_step", -1)):
            with self.subTest(field=field, invalid=invalid):
                changed = argparse.Namespace(**vars(arguments))
                setattr(changed, field, invalid)
                with self.assertRaises(ValueError):
                    candidates.candidate_spec(changed)

    # 辅助函数在私有模块空间复用，不能改变另一份基线模块的四系统常量。
    def test_private_helper_namespace_does_not_mutate_baseline_systems(self):
        path = TOOLS / "evaluate_development_baselines.py"
        first = candidates.load_verified_module(path, candidates.BASELINE_SHA, "_candidate_test_first")
        second = candidates.load_verified_module(path, candidates.BASELINE_SHA, "_candidate_test_second")
        first.SYSTEMS = {"student_example": {"steps": 1}}
        self.assertEqual(len(second.SYSTEMS), 4)
        self.assertIn("teacher500_k30", second.SYSTEMS)
        # 假摘要必须阻止模块执行，不能仅记录警告后继续。
        with self.assertRaises(ValueError):
            candidates.load_verified_module(path, "0" * 64, "_candidate_test_bad")

    # 开发权重必须与原训练契约、实际游标和固定BNF接口同时对应。
    def test_payload_contract_cursor_and_interface_binding(self):
        configuration = {"bnf_alignment": "legacy_length_linear", "bnf_downsampling": "legacy_decimate", "diffusion_steps": 1000, "cosine_offset": 0.008}
        contract = {"kind": "student_training_development_only", "configuration": configuration, "说明": "固定中文字段检查规范化摘要"}
        expected_sha = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        spec = {"state_key": "student", "epoch": 0, "global_step": 600}
        payload = {"epoch": 0, "global_step": 600, "student": {"value": torch.tensor(1.0)}, "configuration": configuration, "run_contract_sha256": expected_sha}
        self.assertEqual(candidates.validate_payload(payload, spec, contract), expected_sha)
        # 换步数、换契约摘要以及换接口都必须单独报错。
        for changed in ({**payload, "global_step": 601}, {**payload, "run_contract_sha256": "0" * 64}, {**payload, "configuration": {**configuration, "bnf_alignment": "physical_centers"}}):
            with self.assertRaises(ValueError):
                candidates.validate_payload(changed, spec, contract)

    # 学生候选与四基线同seed下现有K1推理应逐元素相同。
    def test_student_candidate_matches_existing_one_step_sample(self):
        model = TinyTeacher().eval()
        with torch.inference_mode():
            expected = self.diffusion.sample(model, self.source, self.speaker, self.bnf, num_steps=1, start_step=950, generator=torch.Generator().manual_seed(self.seed), diffuse_source=True)
        model.calls = 0
        model.first_input = None
        actual, x_start = candidates.predict_candidate(model, self.diffusion, self.source, self.speaker, self.bnf, self.seed, "student_k1")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(model.first_input, x_start, atol=0, rtol=0)
        self.assertEqual(model.calls, 1)
        self.assertFalse(actual.requires_grad)
        self.assertIsNone(model.value.grad)

    # DDIM分支必须直接收到已有q950，不能把干净source传给不再加噪的采样器。
    def test_ddim_receives_same_q950_exactly_once(self):
        model = TinyTeacher().eval()
        fake_sampler = Mock(side_effect=lambda **arguments: arguments["x_start"].detach().clone())
        with patch.object(self.diffusion, "q_sample", wraps=self.diffusion.q_sample) as q_sample:
            converted, x_start = candidates.predict_candidate(model, self.diffusion, self.source, self.speaker, self.bnf, self.seed, "teacher500_ddim30", fake_sampler)
        self.assertEqual(q_sample.call_count, 1)
        self.assertEqual(fake_sampler.call_count, 1)
        called = fake_sampler.call_args.kwargs
        torch.testing.assert_close(called["x_start"], x_start, atol=0, rtol=0)
        torch.testing.assert_close(converted, x_start, atol=0, rtol=0)
        self.assertIs(called["speaker_embedding"], self.speaker)
        self.assertIs(called["bnf"], self.bnf)
        self.assertEqual((called["num_steps"], called["start_step"]), (30, 950))

    # 与真实已审核DDIM模块连通时仍应正好30次教师前向，只进行一次源加噪。
    def test_real_ddim_calls_teacher_thirty_times_without_input_mutation(self):
        ddim = candidates.load_verified_module(TOOLS / "ddim_teacher.py", candidates.DDIM_SHA, "_candidate_test_real_ddim")
        model = TinyTeacher().eval()
        before = [value.clone() for value in (self.source, self.speaker, self.bnf)]
        random_state = torch.random.get_rng_state().clone()
        with patch.object(self.diffusion, "q_sample", wraps=self.diffusion.q_sample) as q_sample:
            converted, x_start = candidates.predict_candidate(model, self.diffusion, self.source, self.speaker, self.bnf, self.seed, "teacher500_ddim30", ddim.sample_ddim_teacher)
        self.assertEqual(q_sample.call_count, 1)
        self.assertEqual(model.calls, 30)
        torch.testing.assert_close(model.first_input, x_start, atol=0, rtol=0)
        self.assertTrue(bool(torch.isfinite(converted).all()))
        self.assertFalse(converted.requires_grad)
        self.assertIsNone(model.value.grad)
        # 独立Generator不消耗全局随机序列，也不改源mel或两类教师条件。
        self.assertTrue(torch.equal(torch.random.get_rng_state(), random_state))
        for actual, expected in zip((self.source, self.speaker, self.bnf), before):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


# 只有租用机显式运行本测试文件时才启动CPU小测试。
if __name__ == "__main__":
    unittest.main()
