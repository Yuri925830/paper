# 测试只在Matpool CPU运行，小模型真实Adam用于核对算术；不计算实际学生的GPU预测。
"""CPU acceptance for the matched3227 half-LR read-only direction probe."""

import ast
import copy  # 参考更新和拒绝案例均使用独立状态，不能污染夹具。
import os  # 明确隐藏GPU，禁止从Windows执行项目测试。
import random
import unittest
from pathlib import Path
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
import numpy as np
import torch
from torch import nn
import inspect_acdd_adam_3227_v1 as d


class TinyState(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.randn(5))
        self.zero_gradient = nn.Parameter(torch.randn(3))
        self.no_gradient = nn.Parameter(torch.randn(2))
        self.register_buffer('persistent', torch.tensor([1., 2.]))
        self.register_buffer('nonpersistent', torch.tensor([3., 4.]), persistent=False)


class TinyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 3, 1)

    def forward(self, wave):
        feature = torch.tanh(self.conv(wave))
        score = feature.mean(1, keepdim=True)
        return score, [feature, score]


class TinyD(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([TinyBranch(), TinyBranch()])

    def loss_branches(self):
        return tuple(self.branches)


class Adam3227Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(997)
        student, discriminator = TinyState(), TinyState()
        optimizer = torch.optim.Adam(student.parameters(), lr=.0001, betas=(.5,.9))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(.5,.9))
        for _ in range(3227):  # 小CPU状态真实累计3227次，不人工伪造moment计数。
            for model, opt in ((student,optimizer),(discriminator,d_optimizer)):
                for parameter in model.parameters():
                    parameter.grad = torch.randn_like(parameter)
                opt.step()
                opt.zero_grad(set_to_none=True)
        cls.config = d.base.replace(d.base.PAPER_STUDENT_CONFIG,num_workers=0)
        cls.parent = copy.deepcopy({'student':student.state_dict(),'discriminator':discriminator.state_dict(),
            'student_optimizer':optimizer.state_dict(),'discriminator_optimizer':d_optimizer.state_dict(),
            'rng_state':d.training.capture_rng_state(),'global_step':3227,'run_contract_sha256':d.PARENT_CONTRACT,
            'configuration':cls.config.as_dict(),'effective_optimizer_lrs':{'student':.0001,'discriminator':.0002}})

    def original_models(self):
        student,discriminator = TinyState(),TinyState()
        student.load_state_dict(self.parent['student'])
        discriminator.load_state_dict(self.parent['discriminator'])
        return student,discriminator

    def test_real_half_lr_adam3227_to3228_matches_torch_exactly_none_vs_zero(self):
        student,discriminator = self.original_models()
        before = d.probe.tree_sha(self.parent)
        d.validate_parent(self.parent,student,discriminator,self.config)
        gradients = {'first':torch.tensor([1.,-.5,.3,0.,-.2]),'zero_gradient':torch.zeros(3),'no_gradient':None}
        candidate,delta = d.adam_candidate(list(student.named_parameters()),gradients,self.parent['student_optimizer'],3227)
        reference,_ = self.original_models()
        actual = torch.optim.Adam(reference.parameters(),lr=.0001,betas=(.5,.9))
        actual.load_state_dict(copy.deepcopy(self.parent['student_optimizer']))
        for name,parameter in reference.named_parameters():
            parameter.grad = gradients[name]
        actual.step()  # 仅CPU测试中的小模型参考更新；主诊断无优化器。
        for name,parameter in reference.named_parameters():
            self.assertTrue(torch.equal(candidate[name],parameter))
            if gradients[name] is None:
                self.assertIsNone(delta[name])
                self.assertEqual(float(actual.state[parameter]['step']),3227)
                self.assertTrue(torch.equal(parameter,self.parent['student'][name]))
            else:
                self.assertEqual(float(actual.state[parameter]['step']),3228)
                self.assertTrue(torch.equal(delta[name],parameter.detach()-self.parent['student'][name]))
        self.assertFalse(torch.equal(candidate['zero_gradient'],self.parent['student']['zero_gradient']))
        self.assertEqual(d.probe.tree_sha(self.parent),before)

    def test_saved_lr_validator_does_not_reset_true_student_lr(self):
        student,discriminator=self.original_models()
        state=copy.deepcopy(self.parent)
        before=d.probe.tree_sha(state)
        d.validate_parent(state,student,discriminator,self.config)
        self.assertEqual(state['student_optimizer']['param_groups'][0]['lr'],.0001)
        self.assertEqual(state['discriminator_optimizer']['param_groups'][0]['lr'],.0002)
        self.assertEqual(before,d.probe.tree_sha(state))

    def test_reject_wrong_lr_counter_moment_shape_and_shared_models(self):
        student,discriminator=self.original_models()
        gradients={name:torch.ones_like(parameter) for name,parameter in student.named_parameters()}
        for mutation in ('lr','counter','shape','negative','duplicate','beta'):
            saved=copy.deepcopy(self.parent['student_optimizer'])
            identifier=saved['param_groups'][0]['params'][0]
            if mutation=='lr':saved['param_groups'][0]['lr']=.0002
            if mutation=='counter':saved['state'][identifier]['step']=torch.tensor(3226.)
            if mutation=='shape':saved['state'][identifier]['exp_avg']=torch.ones(1)
            if mutation=='negative':saved['state'][identifier]['exp_avg_sq'].fill_(-1)
            if mutation=='duplicate':saved['param_groups'][0]['params'][1]=identifier
            if mutation=='beta':saved['param_groups'][0]['betas']=(.9,.999)
            with self.assertRaises(RuntimeError):
                d.adam_candidate(list(student.named_parameters()),gradients,saved,3227)
        with self.assertRaises(RuntimeError):
            d.validate_parent(self.parent,student,student,self.config)
        wrong=copy.deepcopy(self.parent);wrong['global_step']=3247
        with self.assertRaises(RuntimeError):d.validate_parent(wrong,student,discriminator,self.config)

    def test_four_scales_are_independent_and_restore_all_buffers(self):
        model,_=self.original_models();original=d.snapshot_model(model)
        candidate={name:value.detach().clone()+.25 for name,value in model.named_parameters()}
        delta={name:candidate[name]-original['state'][name] for name in candidate}
        for scale in d.SCALES:
            with d.trial_state(model,original,candidate,delta,scale):
                for name,parameter in model.named_parameters():
                    expected=candidate[name] if scale==1 else original['state'][name]+scale*delta[name]
                    self.assertTrue(torch.equal(parameter,expected))
            self.assertEqual(d.pair.model_sha(model),original['sha256'])
        with self.assertRaises(ValueError):
            with d.trial_state(model,original,candidate,delta,.1):pass
        self.assertEqual(d.SCALES,(0.,.25,.5,1.))

    def test_trial_exception_restores_parameters_nonpersistent_buffer_and_all_rng(self):
        model,_=self.original_models();original=d.snapshot_model(model)
        candidate={name:p.detach().clone()+.2 for name,p in model.named_parameters()}
        delta={name:candidate[name]-original['state'][name] for name in candidate}
        rng=d.training.capture_rng_state();rng_sha=d.probe.tree_sha(rng)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with d.pair.preserve_rng(torch.device('cpu')),d.trial_state(model,original,candidate,delta,.5):
                model.persistent.add_(4);model.nonpersistent.add_(7)
                random.random();np.random.random();torch.rand(9)
                raise RuntimeError('injected trial failure')
        self.assertEqual(d.pair.model_sha(model),original['sha256'])
        self.assertEqual(d.probe.tree_sha(d.training.capture_rng_state()),rng_sha)

    def test_component_parameter_vjp_equals_independent_joint_loss_without_GT(self):
        torch.manual_seed(39)
        student=nn.Conv1d(80,80,1)
        vocoder,disc=nn.Conv1d(80,1,1),TinyD()
        d.training.set_requires_grad(vocoder,False);d.training.set_requires_grad(disc,False)
        source,reference=torch.randn(2,80,8),torch.randn(2,80,8)
        target=torch.randn_like(source);weights=torch.tensor([.2,.8])
        mean,std=torch.zeros(1,80,1),torch.full((1,80,1),1.7)
        prediction=student(source);leaf=prediction.detach().requires_grad_(True)
        score=(weights*(leaf-target).abs().flatten(1).mean(1)).mean()
        score_y,=torch.autograd.grad(score,leaf)
        wave,adv_y,fm_y=d.probe.wave_output_gradients(leaf,reference,vocoder,disc,mean,std,lambda:None)
        vectors={'adv':adv_y,'2fm':2*fm_y,'45score':45*score_y,'gt_l1':(leaf.detach()-reference).sign()/leaf.numel()}
        banks=d.probe.parameter_banks(prediction,list(student.named_parameters()),vectors,lambda:None)
        # 另建完整小图，直接用联合标量求梯度；不是把待测banks再相加验证自身。
        y=student(source);real=vocoder(d.training.denormalize_mel(reference,mean,std));fake=vocoder(d.training.denormalize_mel(y,mean,std))
        adversarial,fm=0.,0.
        for branch in disc.loss_branches():
            _,rfeatures=branch(real);fscores,ffeatures=branch(fake)
            adversarial=adversarial+d.training.least_squares_student_loss([fscores])
            fm=fm+d.training.feature_matching_loss([rfeatures],[ffeatures])
        dist=(weights*(y-target).abs().flatten(1).mean(1)).mean()
        total=adversarial+2*fm+45*dist
        expected=torch.autograd.grad(total,tuple(student.parameters()))
        for (name,_),gradient in zip(student.named_parameters(),expected):
            self.assertTrue(torch.allclose(banks['total'][name],gradient,rtol=2e-5,atol=2e-6))
        self.assertTrue(all(p.grad is None for model in (student,vocoder,disc) for p in model.parameters()))
        stats=d.probe.gradient_statistics(banks)
        self.assertIn('gt_l1',stats['norms'])  # GT有单独记录，但上述联合标量没有GT项。

    def test_finite_step_residual_uses_fixed_A_not_refreshed_B(self):
        before={'adv':1.,'fm_raw':2.,'diagnostic_frozen_target_score':.2,'diagnostic_frozen_target_total':14.,'train_gt_l1':.3,'primary_moving_score':.2}
        after=dict(before,adv=1.1,fm_raw=1.9,diagnostic_frozen_target_score=.21,diagnostic_frozen_target_total=14.35,train_gt_l1=.29,primary_moving_score=.1)
        stats={'dot_products':{name:{'adam_delta':-.4} for name in ('adv','2fm','45score','total','gt_l1')}}
        result=d.finite_step_error(.5,after,before,stats)
        self.assertAlmostEqual(result['45score']['actual_fixed_objective_change'],.45)
        self.assertAlmostEqual(result['total']['first_order_scale_g_dot_delta'],-.2)
        self.assertAlmostEqual(result['total']['finite_step_residual'],.55)
        changed=dict(after,primary_moving_score=100.)
        self.assertEqual(result,d.finite_step_error(.5,changed,before,stats))

    def test_dot_products_and_None_convention(self):
        banks={'adv':{'x':torch.tensor([1.,2.]),'unused':None},'total':{'x':torch.tensor([1.,2.]),'unused':None}}
        delta={'x':torch.tensor([-.1,-.2]),'unused':None}
        result=d.probe.gradient_statistics(banks,delta)
        self.assertLess(result['dot_products']['total']['adam_delta'],0)
        self.assertAlmostEqual(result['cosines']['total']['adam_delta'],-1.,places=7)
        self.assertEqual(result['none_parameter_names']['total'],['unused'])

    def test_frozen_real_material_identity_and_schemas_CPU_read_only(self):
        self.assertEqual(d.base.sha256_file(d.PROTOCOL),d.PROTOCOL_SHA)
        self.assertEqual(d.base.sha256_file(d.CROSS_RESULTS),d.CROSS_RESULTS_SHA)
        self.assertEqual(d.base.sha256_file(d.PARENT),d.PARENT_SHA)
        payload=d.resume.load_torch_checkpoint(d.PARENT)
        self.assertEqual(payload['global_step'],3227)
        self.assertEqual(payload['run_contract_sha256'],d.PARENT_CONTRACT)
        self.assertEqual(payload['effective_optimizer_lrs'],{'student':.0001,'discriminator':.0002})
        self.assertEqual(payload['configuration'],self.config.as_dict())
        for name,lr in (('student_optimizer',.0001),('discriminator_optimizer',.0002)):
            self.assertEqual(payload[name]['param_groups'][0]['lr'],lr)
            self.assertTrue(all(float(state['step'])==3227 for state in payload[name]['state'].values()))
        cross=d.cross.read_json(d.CROSS_RESULTS)
        self.assertEqual(cross['states']['3227']['teacher_target']['sha256'],'01e35e4b98270e681530ad271778846ecf929aaca6b77909cb17d2820578299e')
        self.assertEqual(len(payload['development']['per_utterance_mel_l1']),32)

    def test_main_no_optimizer_no_weight_write_time_disk_and_VRAM_gates(self):
        tree=ast.parse(Path(d.__file__).read_text())
        calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
        for name in ('Adam','step','atomic_torch_save','save','train_batch'):
            self.assertNotIn(name,calls)
        for name in ('acquire_runtime_locks','assert_rented_server','gpu_idle','capacity','max_memory_allocated','max_memory_reserved','preserve_rng'):
            self.assertIn(name,calls)
        text=Path(d.__file__).read_text()
        self.assertIn("args.max_seconds <= 300",text)
        self.assertIn("gpu_before['free_mib'] < 24576",text)
        with mock.patch.object(d.cross.os,'statvfs',return_value=mock.Mock(f_bavail=1,f_frsize=1024)):
            with self.assertRaises(RuntimeError):d.cross.capacity(d.DEV)

    def test_deadline_boundary_and_disk_recheck_are_executed(self):
        with mock.patch.object(d.cross,'capacity',return_value={}) as capacity:
            with mock.patch.object(d.time,'monotonic',return_value=301.):
                with self.assertRaises(TimeoutError):d.check_limits(0.,300)
            with mock.patch.object(d.time,'monotonic',return_value=300.):
                d.check_limits(0.,300)  # 定义是超过上限才停止；恰好边界仍与主条件一致。
            self.assertEqual(capacity.call_count,2)
        with mock.patch.object(d.cross,'capacity',side_effect=RuntimeError('disk changed')):
            with self.assertRaisesRegex(RuntimeError,'disk changed'):d.check_limits(0.,300)


if __name__ == '__main__':
    unittest.main(verbosity=2)
