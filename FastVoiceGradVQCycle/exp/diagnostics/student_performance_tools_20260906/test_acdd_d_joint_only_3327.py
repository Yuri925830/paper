"""仅在租用CPU检查联合D诊断，不启动GPU、不生成真实实验结果。"""

import ast  # 静态确认没有分项梯度、模型保存或旧helper全局修改。
import copy
import hashlib
import json
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
import numpy as np
import torch
from torch import nn
import inspect_acdd_d_joint_only_3327_v1 as j

# 复用已冻结的三个独立小图/真实Adam用例，不复制或修改旧测试。
OLD_TEST = j.TOOLS/'test_acdd_d_update_3327.py'
if hashlib.sha256(OLD_TEST.read_bytes()).hexdigest() != 'ab5c7ba6dc7b0364365fb7408fe8c548edee366781297836e3d1b684c2c283ea':
    raise RuntimeError('Frozen independent Adam/math tests changed')
import test_acdd_d_update_3327 as previous


class JointOnlyTests(unittest.TestCase):
    setUpClass = classmethod(previous.DUpdateTests.setUpClass.__func__)
    fixture = previous.DUpdateTests.fixture
    test_true_Adam_None_zero_next_counter = previous.DUpdateTests.test_true3327_adam_next_step_none_zero_and_moment_identity
    test_D_update_independent_sum = previous.DUpdateTests.test_D_update_matches_independent_summed_scalar_and_changes_no_student
    test_S_actual_joint_independent_scalar = previous.DUpdateTests.test_production_joint_student_step_equals_direct_tiny_joint_objective

    def tiny_shared(self, student=None):
        # 目标依赖当前学生输出，以便确实检查候选更新后R也可以变化。
        student = nn.Conv1d(1, 1, 1) if student is None else student
        teacher, vocoder = nn.Identity(), nn.Conv1d(1, 1, 1)
        x = torch.randn(2, 1, 7)
        inputs = {'x0':x, 'bnf':x, 'spk_emb':torch.ones(2,1), 'source_noise':torch.zeros_like(x),
                  'score_noise':torch.zeros_like(x), 'score_timesteps':torch.tensor([1,2])}
        diffusion = SimpleNamespace(sqrt_alphas_cumprod=torch.tensor([1.,.4,.8]))
        config = SimpleNamespace(score_target='clean_x0',score_weight='sqrt_alpha_bar')
        def predict(model, diffusion, mel, *args):
            return model(mel), mel*.7
        def score(y, teacher, diffusion, t, noise, speaker, bnf, **kwargs):
            target = y.detach()*.6+.17
            return (diffusion.sqrt_alphas_cumprod[t]*(y-target).abs().flatten(1).mean(1)).mean(), target
        return student, teacher, vocoder, diffusion, inputs, config, predict, score

    def test_each_student_row_reuses_same_wave_target_for_both_D(self):
        student, teacher, vocoder, diffusion, inputs, config, predict, score = self.tiny_shared()
        discriminator = previous.TinyD()
        with mock.patch.object(j.training,'make_student_prediction',side_effect=predict), \
             mock.patch.object(j.training,'score_distillation_loss',side_effect=score) as teacher_call, \
             mock.patch.object(j.training,'denormalize_mel',side_effect=lambda x,*args:x):
            shared = j.shared_readout(student,teacher,vocoder,diffusion,inputs,None,None,config)
            before = copy.deepcopy(shared['identities'])
            a = j.discriminator_readout(discriminator,shared)
            with torch.no_grad():
                discriminator.branches[0].conv.weight.add_(.12)
            b = j.discriminator_readout(discriminator,shared)
            self.assertEqual(teacher_call.call_count,1)
            self.assertEqual(shared['identities'],before)
            for field in shared['scalars']: self.assertEqual(a[field],b[field])
            self.assertNotEqual(a['adv'],b['adv'])
            self.assertEqual(a['diagnostic_frozen_target_total'],a['adv']+2*a['fm_raw']+45*a['diagnostic_frozen_target_score'])
            self.assertEqual(b['primary_moving_total'],b['adv']+2*b['fm_raw']+45*b['primary_moving_score'])
            with torch.no_grad(): student.weight.add_(.2)
            candidate = j.shared_readout(student,teacher,vocoder,diffusion,inputs,None,None,config,shared['tensors']['teacher_target'])
            self.assertEqual(teacher_call.call_count,2)
            self.assertNotEqual(candidate['identities']['teacher_target'],shared['identities']['teacher_target'])
            self.assertEqual(candidate['identities']['q950'],shared['identities']['q950'])
            self.assertGreater(candidate['scalars']['teacher_target_weighted_l1_drift'],0)
            self.assertTrue(all(p.grad is None for m in (student,teacher,vocoder,discriminator) for p in m.parameters()))

    def test_shared_preupdate_point_requires_actual_bitwise_identity(self):
        student, teacher, vocoder, diffusion, inputs, config, predict, score = self.tiny_shared()
        with mock.patch.object(j.training,'make_student_prediction',side_effect=predict), \
             mock.patch.object(j.training,'score_distillation_loss',side_effect=score), \
             mock.patch.object(j.training,'denormalize_mel',side_effect=lambda x,*args:x):
            a = j.shared_readout(student,teacher,vocoder,diffusion,inputs,None,None,config)
            b = j.shared_readout(student,teacher,vocoder,diffusion,inputs,None,None,config)
            j.require_same_shared(a,b)
            b['tensors']['fake_waveform'].add_(.001)  # 即使伪造同一identity字典，也会被真实张量比较发现。
            with self.assertRaisesRegex(RuntimeError,'bitwise'): j.require_same_shared(a,b)
            b = copy.deepcopy(a); b['identities']['teacher_target']['sha256']='bad'
            with self.assertRaises(RuntimeError): j.require_same_shared(a,b)

    def test_actual_direction_norms_None_and_cross_support_are_explicit(self):
        student, _, _, _ = self.fixture()
        reference = j.d.snapshot(student)
        gradients = {n:None if n=='unused' else torch.ones_like(p) for n,p in student.named_parameters()}
        delta = {n:None if v is None else -.3*v for n,v in gradients.items()}
        result = j.actual_direction_summary(gradients,delta,reference)
        self.assertEqual(result['actual_joint_and_delta']['cosines']['actual_joint']['adam_delta'],-1.)
        self.assertTrue(result['per_parameter']['unused']['is_none'])
        j.validate_cross_support(gradients,copy.deepcopy(gradients))
        wrong = copy.deepcopy(gradients); wrong['unused']=torch.zeros(2)
        with self.assertRaises(RuntimeError):j.validate_cross_support(gradients,wrong)
        wrong = copy.deepcopy(delta); wrong['conv.weight'].fill_(float('nan'))
        with self.assertRaises(RuntimeError):j.actual_direction_summary(gradients,wrong,reference)
        # 梯度内积不作为人为通过阈值；它可以是正值，仍须如实报告。
        positive = {n:None if v is None else .1*v for n,v in gradients.items()}
        result = j.actual_direction_summary(gradients,positive,reference)
        self.assertGreater(result['actual_joint_and_delta']['dot_products']['actual_joint']['adam_delta'],0)
        json.dumps(result,allow_nan=False)

    def test_four_models_two_Adam_rng_modes_grad_flags_restore_on_success_and_failure(self):
        entry_flags = j.read_flags()
        try:
            j.restore_flags(j.ORIGINAL_FLAGS)
            for fail in (False,True):
                student, discriminator, so, do = self.fixture()
                teacher, vocoder = previous.TinyStudent(), previous.TinyStudent()
                models = {'student':student,'discriminator':discriminator,'teacher':teacher,'vocoder':vocoder}
                for model in models.values():
                    model.eval()
                    for parameter in model.parameters():
                        parameter.requires_grad_(False); parameter.grad=torch.full_like(parameter,.7)
                student.conv.train()  # 子层模式不要求与顶层相同。
                originals = {n:j.d.snapshot(m) for n,m in models.items()}
                opts = {'student':so,'discriminator':do}
                adam = {n:j.probe.tree_sha(o.state_dict()) for n,o in opts.items()}
                rng = j.training.capture_rng_state()
                def operation():
                    with j.restored_state(models,opts,rng,torch.device('cpu')):
                        for model in models.values():
                            model.train()
                            with torch.no_grad():
                                for buffer in model.buffers():buffer.add_(.15)
                                for parameter in model.parameters():parameter.add_(.03)
                            for parameter in model.parameters():parameter.requires_grad_(True);parameter.grad=torch.ones_like(parameter)
                        so.step(); do.step()
                        random.random(); np.random.rand(); torch.rand(4)
                        j.restore_flags({**j.ORIGINAL_FLAGS,'cudnn_allow_tf32':False})
                        if fail:raise ValueError('injected after temporary update')
                if fail:
                    with self.assertRaisesRegex(ValueError,'injected'):operation()
                else:operation()
                for name,model in models.items():j.verify_snapshot(model,originals[name])
                self.assertEqual({n:j.probe.tree_sha(o.state_dict()) for n,o in opts.items()},adam)
                self.assertEqual(j.probe.tree_sha(j.training.capture_rng_state()),j.probe.tree_sha(rng))
                self.assertEqual(j.read_flags(),j.ORIGINAL_FLAGS)
        finally:j.restore_flags(entry_flags)

    def test_final_snapshot_check_includes_nonpersistent_modes_requires_and_grad(self):
        student, _, _, _ = self.fixture()
        original = j.d.snapshot(student)
        for change in ('mode','requires','grad','nonpersistent'):
            j.d.restore(student,original)
            if change=='mode':student.conv.eval()
            elif change=='requires':student.unused.requires_grad_(False)
            elif change=='grad':student.unused.grad=torch.ones_like(student.unused)
            else:student.nonpersistent.add_(1)
            with self.assertRaises(RuntimeError):j.verify_snapshot(student,original)
        j.d.restore(student,original);j.verify_snapshot(student,original)

    def test_all_3x2_differences_compare_same_D_and_report_D_effect(self):
        matrix={s:{r:{'adv':10*i+j,'diagnostic_frozen_target_total':20*i+2*j,'primary_moving_total':30*i+3*j}
                   for j,r in enumerate(j.d.DISCRIMINATORS)} for i,s in enumerate(j.d.STUDENTS)}
        development={s:{'mel_l1':.2+i*.01} for i,s in enumerate(j.d.STUDENTS)}
        result=j.d.matrix_differences(matrix,development)
        self.assertEqual(result['candidate_D1_minus_candidate_D0_same_D']['D0']['adv'],10)
        self.assertEqual(result['candidate_D1_minus_candidate_D0_same_D']['D1']['adv'],10)
        with self.assertRaises(RuntimeError):j.d.matrix_differences({'original':matrix['original']},development)

    def test_original_flags_and_time_disk_are_hard_requirements(self):
        with mock.patch.object(j,'read_flags',return_value={**j.ORIGINAL_FLAGS,'cudnn_allow_tf32':False}):
            with self.assertRaises(RuntimeError):j.require_original_flags()
        with mock.patch.object(j.cross,'capacity',return_value={}),mock.patch.object(j.d.time,'monotonic',return_value=301):
            with self.assertRaises(TimeoutError):j.d.check_limits(0,300)
        with mock.patch.object(j.cross,'capacity',side_effect=RuntimeError('low disk')):
            with self.assertRaisesRegex(RuntimeError,'low disk'):j.d.check_limits(0,300)

    def test_real_parent_protocol_flags_retained_next_batch_and_original_DEV(self):
        # 原full的完整双Adam字段另由复用用例逐项验证；这里核验本次新增协议与现场精度契约。
        j.resume.verify_files({**j.d.PINNED,str(j.HELPER):j.HELPER_SHA,str(j.PROTOCOL):j.PROTOCOL_SHA})
        contract=j.cross.read_json(j.d.RUN/'run_contract.json')
        self.assertEqual({k:contract['runtime'][k] for k in j.ORIGINAL_FLAGS},j.ORIGINAL_FLAGS)
        self.assertEqual(j.training.canonical_json_sha256(contract),j.d.CONTRACT_SHA)
        indices=j.resume.GlobalBatchPlan(contract['training_indices']).for_step(3328)
        self.assertEqual(len(indices),32)
        self.assertFalse(set(indices)&set(contract['excluded_train_indices']))
        archived=j.resume.load_torch_checkpoint(j.pair.INPUT_DIR/'fixed_inputs.pt')
        for name in ('dev_mel','dev_bnf','dev_spk_emb','dev_source_noise'):
            self.assertEqual(j.cross.identity(archived[name])['sha256'],contract['fixed_train_inputs_sha256'][name])
        self.assertEqual(archived['dev_mel'].shape,(32,80,128))

    test_real_full3327_Adam_contract_rejections = previous.DUpdateTests.test_real3327_contract_checkpoint_moments_and_dev_input_schema

    def test_source_joint_only_no_model_writes_old_globals_or_new_training(self):
        source=Path(j.__file__).read_text();tree=ast.parse(source)
        forbidden={'component_gradients','parameter_banks','wave_output_gradients','bank_difference','train_batch','train_epoch',
                   'save','save_checkpoint','unlink','rmdir','backward','grad'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotIn(node.func.attr,forbidden)
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('d','old','pair','base','training','probe','resume','cross') for t in targets))
        self.assertIn('actual, adam_record = d.s_update(',source)
        self.assertIn('d_record = d.d_update(',source)
        self.assertIn("for role in d.DISCRIMINATORS:",source)
        self.assertIn("for label in d.STUDENTS:",source)
        self.assertIn('1 <= args.max_seconds <= 300',source)
        self.assertEqual(j.OUTPUT.name,'acdd_d_joint_only_readonly_3327_v1')
        # 必要的严格fake/target门禁仍来自旧冻结helper，不因去掉额外估算器而放宽。
        helper=Path(j.d.__file__).read_text()
        self.assertIn('torch.equal(waveform.detach(), fake)',helper)
        self.assertIn('torch.equal(actual_target, target)',helper)


if __name__=='__main__':
    unittest.main()
