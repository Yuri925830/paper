"""租用CPU验收D单次更新诊断；不执行实际GPU前向，也不发布诊断结果。"""

import ast  # 检查入口没有train_batch、模型保存或旧全局常量修改。
import copy
import json
import os  # CUDA必须显式隐藏，Windows禁止运行。
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
import inspect_acdd_d_update_3327_v1 as d


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)
        self.unused = nn.Parameter(torch.randn(2))  # 已有Adam状态，但本次图可以没有梯度。
        self.register_buffer('persistent', torch.tensor([1.]))
        self.register_buffer('nonpersistent', torch.tensor([2.]), persistent=False)

    def forward(self, x):
        return torch.tanh(self.conv(x))


class TinyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 2, 1)

    def forward(self, wave):
        feature = torch.tanh(self.conv(wave))
        score = feature.mean(1, keepdim=True)
        return score, [feature, score]  # 末端score也保留，与实际FM接口一致。


class TinyD(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([TinyBranch(), TinyBranch()])
        self.register_buffer('hann', torch.tensor([.2, .8]), persistent=False)

    def loss_branches(self):
        return tuple(self.branches)


class DUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(991)
        student, discriminator = TinyStudent(), TinyD()
        opts = [torch.optim.Adam(student.parameters(), lr=.00005, betas=(.5, .9)),
                torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(.5, .9))]
        # 真正累计3327次微型Adam，不用把新优化器计数强填成3327冒充恢复。
        for _ in range(d.STEP):
            for model, opt in zip((student, discriminator), opts):
                for parameter in model.parameters():
                    parameter.grad = torch.randn_like(parameter)
                opt.step(); opt.zero_grad(set_to_none=True)
        cls.states = [copy.deepcopy(m.state_dict()) for m in (student, discriminator)]
        cls.adams = [copy.deepcopy(o.state_dict()) for o in opts]

    def fixture(self):
        student, discriminator = TinyStudent(), TinyD()
        student.load_state_dict(self.states[0]); discriminator.load_state_dict(self.states[1])
        so = torch.optim.Adam(student.parameters(), lr=.00005, betas=(.5, .9))
        do = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(.5, .9))
        d.restore_optimizer(so, self.adams[0]); d.restore_optimizer(do, self.adams[1])
        return student, discriminator, so, do

    def test_true3327_adam_next_step_none_zero_and_moment_identity(self):
        student, _, opt, _ = self.fixture()
        reference, _, reference_opt, _ = self.fixture()
        gradients = {'unused': None, 'conv.weight': torch.full_like(student.conv.weight, .7),
                     'conv.bias': torch.zeros_like(student.conv.bias)}
        for model in (student, reference):
            for name, parameter in model.named_parameters():
                parameter.grad = None if gradients[name] is None else gradients[name].clone()
        actual, record = d.actual_adam_step(student, opt, d.STEP)
        reference_opt.step(); reference_opt.zero_grad(set_to_none=True)
        self.assertEqual(d.probe.tree_sha(student.state_dict()), d.probe.tree_sha(reference.state_dict()))
        self.assertEqual(d.probe.tree_sha(opt.state_dict()), d.probe.tree_sha(reference_opt.state_dict()))
        self.assertEqual(record['counters']['unused'], 3327)
        self.assertEqual(record['counters']['conv.bias'], 3328)
        self.assertIsNone(actual['unused'])
        self.assertFalse(torch.equal(student.conv.bias, self.states[0]['conv.bias']))
        self.assertTrue(all(p.grad is None for p in student.parameters()))

    def test_D_update_matches_independent_summed_scalar_and_changes_no_student(self):
        student, discriminator, _, opt = self.fixture()
        _, reference, _, reference_opt = self.fixture()
        real, fake = torch.randn(2, 1, 9), torch.randn(2, 1, 9)
        s_before = d.pair.model_sha(student)
        record = d.d_update(discriminator, opt, real, fake)
        real_scores, fake_scores = [], []
        for branch in reference.loss_branches():
            real_scores.append(branch(real)[0]); fake_scores.append(branch(fake)[0])
        loss = d.training.least_squares_discriminator_loss(real_scores, fake_scores)
        loss.backward(); reference_opt.step(); reference_opt.zero_grad(set_to_none=True)
        self.assertEqual(record['loss_before_update'], float(loss))
        self.assertEqual(d.probe.tree_sha(discriminator.state_dict()), d.probe.tree_sha(reference.state_dict()))
        self.assertEqual(d.probe.tree_sha(opt.state_dict()), d.probe.tree_sha(reference_opt.state_dict()))
        self.assertEqual(d.pair.model_sha(student), s_before)
        self.assertTrue(all(x == 3328 for x in record['counters'].values()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in discriminator.parameters()))
        with self.assertRaises(RuntimeError): d.d_update(discriminator, opt, real.requires_grad_(), fake)

    def test_production_joint_student_step_equals_direct_tiny_joint_objective(self):
        student, discriminator, opt, _ = self.fixture()
        reference, reference_d, reference_opt, _ = self.fixture()
        d.training.set_requires_grad(discriminator, False); d.training.set_requires_grad(reference_d, False)
        vocoder = nn.Conv1d(1, 1, 1); d.training.set_requires_grad(vocoder, False)
        teacher = nn.Identity()
        x0, target = torch.randn(2, 1, 9), torch.randn(2, 1, 9)
        weights = torch.tensor([.3, .9])
        inputs = {'x0':x0, 'spk_emb':torch.ones(2,1), 'bnf':x0, 'source_noise':torch.zeros_like(x0),
                  'score_noise':torch.zeros_like(x0), 'score_timesteps':torch.tensor([1,2])}
        config = SimpleNamespace(lambda_fm=2, lambda_dist=45, score_target='clean_x0', score_weight='sqrt_alpha_bar')
        predict = lambda model, diffusion, mel, *args: (model(mel), mel)
        score = lambda y, *args, **kwargs: ((weights * (y-target).abs().flatten(1).mean(1)).mean(), target)
        with torch.no_grad():
            real, fake = vocoder(x0), vocoder(student(x0))
        with mock.patch.object(d.training, 'make_student_prediction', side_effect=predict), \
             mock.patch.object(d.training, 'score_distillation_loss', side_effect=score), \
             mock.patch.object(d.training, 'denormalize_mel', side_effect=lambda x,*args:x):
            actual, record = d.s_update(student, teacher, vocoder, discriminator, None, inputs, real, fake, target, None, None, config, opt)
            y = reference(x0); wave = vocoder(y)
            real_maps, fake_maps, fake_scores = [], [], []
            for branch in reference_d.loss_branches():
                with torch.no_grad(): real_maps.append(branch(real)[1])
                scores, maps = branch(wave); fake_scores.append(scores); fake_maps.append(maps)
            direct = d.training.least_squares_student_loss(fake_scores) + 2*d.training.feature_matching_loss(real_maps,fake_maps) + 45*score(y)[0]
            direct.backward()
            for name,p in reference.named_parameters():
                if p.grad is None: self.assertIsNone(actual[name])
                else: torch.testing.assert_close(actual[name], p.grad, rtol=1e-6, atol=1e-6)
            reference_opt.step(); reference_opt.zero_grad(set_to_none=True)
            for p,q in zip(student.parameters(), reference.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=1e-7)
        self.assertEqual(record['lr'], .00005)
        self.assertEqual(record['counters']['unused'], 3327)

    def test_success_and_exception_restore_nonpersistent_modes_grad_flags_adams_rng(self):
        for failure in (False, True):
            student, discriminator, so, do = self.fixture()
            student.conv.eval(); student.unused.requires_grad_(False)
            original = {'s':d.snapshot(student), 'd':d.snapshot(discriminator)}
            adams = [d.probe.tree_sha(x.state_dict()) for x in (so,do)]
            rng = d.training.capture_rng_state()
            def operation():
                with d.restore_everything({'s':student,'d':discriminator},{'s':so,'d':do},rng,torch.device('cpu')):
                    with torch.no_grad():
                        student.nonpersistent.add_(9); discriminator.hann.add_(3); student.conv.weight.add_(.1)
                    student.train(); student.unused.requires_grad_(True)
                    for p in student.parameters(): p.grad=torch.ones_like(p)
                    so.step(); random.random(); np.random.rand(); torch.rand(3)
                    if failure: raise ValueError('injected')
            if failure:
                with self.assertRaisesRegex(ValueError,'injected'):operation()
            else: operation()
            self.assertEqual(d.probe.tree_sha(d.snapshot(student)),d.probe.tree_sha(original['s']))
            self.assertEqual(d.probe.tree_sha(d.snapshot(discriminator)),d.probe.tree_sha(original['d']))
            self.assertEqual([d.probe.tree_sha(x.state_dict()) for x in (so,do)],adams)
            self.assertEqual(d.probe.tree_sha(d.training.capture_rng_state()),d.probe.tree_sha(rng))

    def test_optimizer_load_and_CPU_snapshot_do_not_alias_parent(self):
        student, _, opt, _ = self.fixture()
        saved=copy.deepcopy(self.adams[0]); before=d.probe.tree_sha(saved)
        d.restore_optimizer(opt,saved)
        for p in student.parameters():p.grad=torch.ones_like(p)
        opt.step()
        self.assertEqual(d.probe.tree_sha(saved),before)
        cpu=d.cpu_copy(opt.state_dict()); original=d.probe.tree_sha(opt.state_dict())
        first=next(iter(cpu['state'].values()));first['exp_avg'].add_(1)
        self.assertEqual(d.probe.tree_sha(opt.state_dict()),original)

    def test_gradient_invariants_exact_support_and_joint_rounding_limit(self):
        a={'used':torch.tensor([1.,2.]),'unused':None}
        self.assertEqual(d.bank_difference(a,copy.deepcopy(a),tight=True)['max_absolute_difference'],0)
        b={'used':a['used']+1e-6,'unused':None}
        self.assertLess(d.bank_difference(a,b)['relative_l2_difference'],1e-4)
        self.assertFalse(d.bank_difference(a,b,tight=True)['actual_exact'])
        with self.assertRaises(RuntimeError):d.bank_difference(a,{'used':a['used']+1e-4,'unused':None},tight=True)
        with self.assertRaises(RuntimeError):d.bank_difference(a,{'used':a['used']+1.,'unused':None})
        with self.assertRaises(RuntimeError):d.bank_difference(a,{'used':a['used'],'unused':torch.zeros(1)})

    def test_complete_3S_by_2D_changes_keep_the_same_D_column(self):
        matrix={s:{r:{'adv':10*i+j,'diagnostic_frozen_target_total':20*i+2*j,'primary_moving_total':30*i+3*j}
                   for j,r in enumerate(d.DISCRIMINATORS)} for i,s in enumerate(d.STUDENTS)}
        development={s:{'mel_l1':.2+i*.01} for i,s in enumerate(d.STUDENTS)}
        differences=d.matrix_differences(matrix,development)
        self.assertEqual(differences['candidate_D1_minus_candidate_D0_same_D']['D0']['adv'],10)
        self.assertEqual(differences['candidate_D1_minus_candidate_D0_same_D']['D1']['adv'],10)
        self.assertAlmostEqual(differences['development_D1_minus_D0'],.01)
        broken=copy.deepcopy(matrix);del broken['candidate_D1']['D0']
        with self.assertRaises(RuntimeError):d.matrix_differences(broken,development)

    def test_time_and_disk_guards_fail_without_GPU(self):
        with mock.patch.object(d.cross,'capacity',return_value={}),mock.patch.object(d.time,'monotonic',return_value=301):
            with self.assertRaises(TimeoutError):d.check_limits(0,300)
        with mock.patch.object(d.cross,'capacity',side_effect=RuntimeError('low disk')):
            with self.assertRaisesRegex(RuntimeError,'low disk'):d.check_limits(0,300)

    def test_real3327_contract_checkpoint_moments_and_dev_input_schema(self):
        # CPU只读实际full，创建与moment形状相同的占位参数用于完整Adam字段检查，不运行大网络。
        d.resume.verify_files(d.PINNED)
        payload=d.resume.load_torch_checkpoint(d.PARENT)
        contract=d.cross.read_json(d.RUN/'run_contract.json');receipt=d.cross.read_json(d.RUN/'completion.json')
        models=[]
        for role in ('student','discriminator'):
            saved=payload[role+'_optimizer'];ids=saved['param_groups'][0]['params']
            models.append(nn.ParameterList([nn.Parameter(torch.empty_like(saved['state'][k]['exp_avg'])) for k in ids]))
        config=d.base.replace(d.base.PAPER_STUDENT_CONFIG,num_workers=0)
        d.validate_parent(payload,*models,config,contract,receipt)
        self.assertEqual([len(m) for m in models],[71,144])
        archived=d.resume.load_torch_checkpoint(d.pair.INPUT_DIR/'fixed_inputs.pt')
        for key in ('dev_mel','dev_bnf','dev_spk_emb','dev_source_noise'):
            self.assertEqual(d.cross.identity(archived[key])['sha256'],contract['fixed_train_inputs_sha256'][key])
        self.assertEqual(archived['dev_mel'].shape,(32,80,128))
        indices=d.resume.GlobalBatchPlan(contract['training_indices']).for_step(3328)
        self.assertEqual(len(indices),32)
        self.assertFalse(set(indices)&set(contract['excluded_train_indices']))
        self.assertEqual(d.LRS,{'student':.00005,'discriminator':.0002})
        for bad in ('lr','step','score','partial'):
            changed=dict(payload);rec=copy.deepcopy(receipt)
            if bad=='lr':changed['effective_optimizer_lrs']={'student':.0001,'discriminator':.0002}
            if bad=='step':changed['global_step']=3247
            if bad=='score':changed['score_semantics']='legacy'
            if bad=='partial':rec['partial_batch_may_have_updated_D']=True
            with self.assertRaises(RuntimeError):d.validate_parent(changed,*models,config,contract,rec)

    def test_source_has_no_model_write_training_loop_or_old_global_override(self):
        source=Path(d.__file__).read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('save','save_checkpoint','train_batch','train_epoch','unlink','rmdir'))
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('old','pair','base','training','probe','resume','cross') for t in targets))
        self.assertIn('training.serial_discriminator_backward(discriminator, real, fake)',source)
        self.assertIn('training.serial_student_discriminator_gradient(discriminator, real, fake, config.lambda_fm)',source)
        self.assertIn('torch.autograd.backward((waveform, score)',source)
        self.assertEqual(d.STUDENTS,('original','candidate_D0','candidate_D1'))
        self.assertEqual(d.DISCRIMINATORS,('D0','D1'))


if __name__=='__main__':
    unittest.main()
