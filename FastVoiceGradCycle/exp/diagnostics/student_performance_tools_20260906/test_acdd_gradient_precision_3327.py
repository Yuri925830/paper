"""Matpool CPU验收精度诊断；CPU小图检查公式与恢复，不预测GPU实测误差。"""

import ast
import copy
import hashlib
import json
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
import numpy as np
import torch
from torch import nn
import inspect_acdd_gradient_precision_3327_v1 as p


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv1d(1,1,1)
        self.unused=nn.Parameter(torch.ones(2))  # 用来验证None支持一致，不会删除这个参数。
        self.register_buffer('persistent',torch.tensor([1.]))
        self.register_buffer('hann_like',torch.tensor([.2,.8]),persistent=False)

    def forward(self,x):
        return torch.tanh(self.conv(x))


class TinyBranch(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv1d(1,2,1)

    def forward(self,wave):
        feature=torch.tanh(self.conv(wave));score=feature.mean(1,keepdim=True)
        return score,[feature,score]


class TinyD(nn.Module):
    def __init__(self):
        super().__init__();self.branches=nn.ModuleList([TinyBranch(),TinyBranch()])

    def loss_branches(self):
        return tuple(self.branches)


class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(812)
        student,teacher,vocoder,discriminator=TinyStudent(),nn.Identity(),nn.Conv1d(1,1,1),TinyD()
        for model in (teacher,vocoder,discriminator):p.training.set_requires_grad(model,False)
        x=torch.randn(2,1,9);target=torch.randn_like(x);weights=torch.tensor([.3,.9])
        inputs={'x0':x,'spk_emb':torch.ones(2,1),'bnf':x,'source_noise':torch.zeros_like(x),
                'score_noise':torch.zeros_like(x),'score_timesteps':torch.tensor([1,2])}
        config=SimpleNamespace(lambda_fm=2,lambda_dist=45,score_target='clean_x0',score_weight='sqrt_alpha_bar')
        predict=lambda model,diffusion,mel,*args:(model(mel),mel)
        score=lambda y,*args,**kwargs:((weights*(y-target).abs().flatten(1).mean(1)).mean(),target)
        args=(student,teacher,vocoder,discriminator,None,inputs,None,None,config,lambda:None)
        return args,predict,score

    def test_split_and_two_production_joint_gradients_on_independent_tiny_graph(self):
        args,predict,score=self.fixture();student,teacher,vocoder,discriminator=args[:4]
        before={n:p.d.snapshot(m) for n,m in [('s',student),('t',teacher),('v',vocoder),('d',discriminator)]}
        with mock.patch.object(p.training,'make_student_prediction',side_effect=predict), \
             mock.patch.object(p.training,'score_distillation_loss',side_effect=score), \
             mock.patch.object(p.training,'denormalize_mel',side_effect=lambda x,*args:x):
            split,trace_s,record_s=p.split_gradient(*args)
            joint1,trace_j1,record_j1=p.joint_gradient(*args)
            # 人工放入旧grad，验证joint2入口确实清空，不能与joint1累加。
            for parameter in student.parameters():parameter.grad=torch.full_like(parameter,123.)
            joint2,trace_j2,record_j2=p.joint_gradient(*args)
            p.require_same_trace(trace_s,trace_j1);p.require_same_trace(trace_s,trace_j2)
            for name in split:
                if split[name] is None:
                    self.assertIsNone(joint1[name]);self.assertIsNone(joint2[name])
                else:
                    torch.testing.assert_close(split[name],joint1[name],rtol=1e-6,atol=1e-6)
                    self.assertTrue(torch.equal(joint1[name],joint2[name]))
            # 独立完整标量目标验证2/45与所有D分支之和，不复用split或joint求导组合。
            x=args[5]['x0'];y=student(x);fake=vocoder(y)
            with torch.no_grad():real=vocoder(x)
            scores,real_maps,fake_maps=[],[],[]
            for branch in discriminator.loss_branches():
                with torch.no_grad():real_maps.append(branch(real)[1])
                item,maps=branch(fake);scores.append(item);fake_maps.append(maps)
            direct=p.training.least_squares_student_loss(scores)+2*p.training.feature_matching_loss(real_maps,fake_maps)+45*score(y)[0]
            names,parameters=zip(*student.named_parameters())
            direct_grads=torch.autograd.grad(direct,parameters,allow_unused=True)
            for name,gradient in zip(names,direct_grads):
                if gradient is None:self.assertIsNone(joint1[name])
                else:torch.testing.assert_close(joint1[name],gradient,rtol=1e-6,atol=1e-6)
            self.assertAlmostEqual(record_j1['losses']['total'],float(direct),places=5)
        self.assertEqual(record_j1['losses'],record_j2['losses'])
        self.assertEqual(set(record_s['component_norms']),{'adv','2fm','45score','total'})
        self.assertTrue(all(v.grad is None for model in (student,teacher,vocoder,discriminator) for v in model.parameters()))
        for name,model in [('s',student),('t',teacher),('v',vocoder),('d',discriminator)]:
            self.assertEqual(p.probe.tree_sha(p.d.snapshot(model)),p.probe.tree_sha(before[name]))

    def test_raw_differences_keep_large_errors_and_all_parameters_without_a_gate(self):
        left={'a':torch.tensor([4.,6.]),'none':None,'zero':torch.zeros(1)}
        right={'a':torch.tensor([1.,2.]),'none':None,'zero':torch.zeros(1)}
        result=p.gradient_difference(left,right)
        self.assertEqual(result['absolute_l2_difference'],5.)
        self.assertEqual(result['max_absolute_difference'],4.)
        self.assertAlmostEqual(result['relative_l2_difference'],5/(5**.5))
        self.assertEqual(result['parameter_count'],3)
        self.assertEqual(set(result['per_parameter']),set(left))
        self.assertEqual(result['none_parameters'],['none'])
        # 远大于旧1e-4的差异依然作为原始结果返回，不据大小选择通过。
        self.assertGreater(result['relative_l2_difference'],1.)

    def test_zero_reference_is_explicitly_undefined_not_inf_or_epsilon(self):
        result=p.gradient_difference({'x':torch.tensor([3.,4.])},{'x':torch.zeros(2)})
        self.assertIsNone(result['relative_l2_difference'])
        self.assertTrue(result['relative_l2_undefined_zero_reference'])
        self.assertEqual(result['absolute_l2_difference'],5.)
        self.assertEqual(result['per_parameter']['x']['max_absolute_difference'],4.)
        json.dumps(result,allow_nan=False)
        equal=p.gradient_difference({'x':None},{'x':None})
        self.assertTrue(equal['actual_exact']);self.assertIsNone(equal['relative_l2_difference'])

    def test_none_support_dtype_nonfinite_and_trace_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):p.gradient_difference({'x':None},{'x':torch.zeros(1)})
        with self.assertRaises(RuntimeError):p.difference(torch.zeros(1),torch.zeros(1,dtype=torch.float64))
        with self.assertRaises(RuntimeError):p.difference(torch.tensor([float('nan')]),torch.ones(1))
        trace={name:torch.ones(2) for name in ('prediction','q950','real_waveform','fake_waveform','teacher_target')}
        p.require_same_trace(trace,copy.deepcopy(trace))
        for name in trace:
            other=copy.deepcopy(trace);other[name][0]+=1e-6
            with self.subTest(name=name),self.assertRaises(RuntimeError):p.require_same_trace(other,trace)

    def test_cross_precision_trace_can_differ_and_reports_original_values(self):
        left,reference=torch.tensor([1.001,2.]),torch.tensor([1.,2.])
        result=p.difference(left,reference)
        self.assertFalse(result['actual_exact'])
        self.assertGreater(result['absolute_l2_difference'],0)
        self.assertEqual(result['numel'],2)

    def test_student_requires_gradient_is_not_mistaken_for_weight_mutation(self):
        args,_,_=self.fixture();models=dict(zip(('student','teacher','vocoder','discriminator'),args[:4]))
        hashes={name:p.pair.model_sha(model) for name,model in models.items()}
        # 直接复现旧helper的语义，避免只测试新断言而漏掉真实启动阻断。
        with self.assertRaises(RuntimeError):p.pair.assert_frozen(models,hashes)
        p.verify_models(models,hashes,discriminator_frozen=True)
        p.training.set_requires_grad(models['discriminator'],True)
        with self.assertRaises(RuntimeError):p.verify_models(models,hashes,discriminator_frozen=True)
        p.verify_models(models,hashes,discriminator_frozen=False)
        models['student'].conv.weight.grad=torch.ones_like(models['student'].conv.weight)
        with self.assertRaises(RuntimeError):p.verify_models(models,hashes,discriminator_frozen=False)
        models['student'].conv.weight.grad=None
        with torch.no_grad():models['student'].conv.weight.add_(.01)
        with self.assertRaises(RuntimeError):p.verify_models(models,hashes,discriminator_frozen=False)

    def test_flags_restored_after_success_and_injected_exception(self):
        original=p.read_flags()
        desired={name:not value for name,value in original.items()}
        with p.precision_scope(desired):self.assertEqual(p.read_flags(),desired)
        self.assertEqual(p.read_flags(),original)
        with self.assertRaisesRegex(ValueError,'injected'):
            with p.precision_scope(desired):raise ValueError('injected')
        self.assertEqual(p.read_flags(),original)
        with self.assertRaises(RuntimeError):p.write_flags({'cudnn_allow_tf32':False})
        self.assertEqual(p.read_flags(),original)

    def test_full_model_flags_buffers_grad_modes_rng_restored_on_exception_without_optimizer(self):
        args,_,_=self.fixture();models=dict(zip(('student','teacher','vocoder','discriminator'),args[:4]))
        models['student'].conv.eval()
        originals={n:p.d.snapshot(m) for n,m in models.items()}
        rng=p.training.capture_rng_state();flags=p.read_flags()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with p.d.restore_everything(models,{},rng,torch.device('cpu')):
                with p.precision_scope(dict(p.ORIGINAL_FLAGS,cudnn_allow_tf32=False)):
                    for model in models.values():
                        model.train()
                        for parameter in model.parameters():
                            parameter.requires_grad_(True);parameter.grad=torch.ones_like(parameter)
                            with torch.no_grad():parameter.add_(.2)
                    models['student'].hann_like.add_(3)
                    random.random();np.random.rand();torch.rand(4)
                    raise RuntimeError('injected')
        for name,model in models.items():self.assertEqual(p.probe.tree_sha(p.d.snapshot(model)),p.probe.tree_sha(originals[name]))
        self.assertEqual(p.probe.tree_sha(p.training.capture_rng_state()),p.probe.tree_sha(rng))
        self.assertEqual(p.read_flags(),flags)

    def test_recorded_protocol_parent_runtime_and_next3328_sampling_identity(self):
        self.assertEqual(hashlib.sha256(p.PROTOCOL.read_bytes()).hexdigest(),p.PROTOCOL_SHA)
        self.assertEqual(hashlib.sha256(p.HELPER.read_bytes()).hexdigest(),p.HELPER_SHA)
        contract=p.cross.read_json(p.d.RUN/'run_contract.json')
        self.assertEqual(p.training.canonical_json_sha256(contract),p.d.CONTRACT_SHA)
        self.assertEqual({k:contract['runtime'][k] for k in p.ORIGINAL_FLAGS},p.ORIGINAL_FLAGS)
        self.assertEqual(p.d.PARENT.stat().st_size,1424630725)
        indices=p.resume.GlobalBatchPlan(contract['training_indices']).for_step(3328)
        self.assertEqual(len(indices),32);self.assertFalse(set(indices)&set(contract['excluded_train_indices']))
        self.assertEqual(p.d.NEXT,3328)
        self.assertEqual(p.ROUTES,('split','joint1','joint2'))
        a=dict(p.ORIGINAL_FLAGS);b=dict(p.ORIGINAL_FLAGS,cudnn_allow_tf32=False)
        self.assertEqual([k for k in a if a[k]!=b[k]],['cudnn_allow_tf32'])

    def test_600second_and_disk_guards_without_GPU(self):
        with mock.patch.object(p.cross,'capacity',return_value={}),mock.patch.object(p.time,'monotonic',return_value=601):
            with self.assertRaises(TimeoutError):p.check_limits(0,600)
        with mock.patch.object(p.cross,'capacity',side_effect=RuntimeError('low disk')):
            with self.assertRaisesRegex(RuntimeError,'low disk'):p.check_limits(0,600)

    def test_source_never_constructs_optimizer_updates_models_or_applies_old_error_gate(self):
        source=Path(p.__file__).read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('step','Adam','actual_adam_step','s_update','d_update','train_batch','train_epoch','save','save_checkpoint','unlink','bank_difference'))
            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name):
                self.assertFalse(node.value.id=='torch' and node.attr=='optim')
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('d','pair','base','training','resume','probe','cross') for t in targets))
        self.assertIn('training.serial_student_discriminator_gradient(',source)
        self.assertIn('torch.autograd.backward((fake,score)',source)
        self.assertIn('d.restore_everything(models,{},payload',source)
        self.assertEqual(p.MAX_JSON_BYTES,2*1024**2)


if __name__=='__main__':
    unittest.main()
