# CPU只用小网络真实运行Adam，验证完整3227状态与严格停止规则；不接触真实GPU训练。
"""CPU acceptance for inherited half-LR3227 bounded continuation."""

import ast  # 确认仅有原train_batch且不存在旧模块全局常量修改。
import copy  # 各拒绝案例不污染父模型状态。
import contextlib  # 捕获故障报告，核对真实恢复游标。
import io
import json  # 读取真实冻结历史和终点收据作独立身份fixture。
import os  # 明确隐藏CUDA才能执行本测试。
import random
import tempfile  # 测试文件仅限本次创建的临时目录。
import unittest
from pathlib import Path
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
import numpy as np
import torch
from torch import nn
import run_acdd_half_lr_continue480_v1 as d


def models():
    student, discriminator = nn.Linear(3,2), nn.Linear(2,1)
    so = torch.optim.Adam(student.parameters(), lr=.0001, betas=(.5,.9))
    do = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(.5,.9))
    return student,discriminator,so,do


def update(items):
    s,disc,so,do=items
    random.random(),np.random.random()
    source,target=torch.randn(4,3),torch.randn(4,2)
    d.training.set_requires_grad(disc,True)
    do.zero_grad(set_to_none=True)
    dloss=(disc(target)-1).square().mean()+disc(s(source).detach()).square().mean()
    dloss.backward();do.step();do.zero_grad(set_to_none=True)
    d.training.set_requires_grad(disc,False)
    so.zero_grad(set_to_none=True)
    prediction=s(source)
    loss=(prediction-target).abs().mean()+(disc(prediction)-1).square().mean()
    loss.backward();so.step();so.zero_grad(set_to_none=True)
    d.training.set_requires_grad(disc,True)
    return {key:float(loss.detach()) for key in d.pair.LOSS_KEYS}


def snapshot(items,step=3227):
    s,disc,so,do=items
    return copy.deepcopy({'student':s.state_dict(),'discriminator':disc.state_dict(),'student_optimizer':so.state_dict(),
        'discriminator_optimizer':do.state_dict(),'rng_state':d.training.capture_rng_state(),'global_step':step,
        'run_contract_sha256':d.PARENT_CONTRACT,'configuration':d.base.replace(d.base.PAPER_STUDENT_CONFIG,num_workers=0).as_dict(),
        'effective_optimizer_lrs':dict(d.ACTUAL_LRS),'development':{'mel_l1':.1948028951883316,'prediction_rms':1.0114428997039795}})


class Continue480Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        d.training.seed_everything(221)
        cls.items=models()
        for _ in range(3227):update(cls.items)  # 小双Adam真实计数3227，不手造moment或计数。
        cls.parent=snapshot(cls.items)
        cls.config=d.base.replace(d.base.PAPER_STUDENT_CONFIG,num_workers=0)

    def setUp(self):
        patch=mock.patch.object(d.pair,'available_bytes',return_value=20*1024**3)
        patch.start();self.addCleanup(patch.stop)

    def restored(self):
        items=models()
        d.restore_actual_parent(self.parent,*items,self.config,torch.device('cpu'))
        return items

    def histories(self):
        parent=json.loads((d.PARENT/'arm_completion.json').read_text())
        return d.load_history(parent)

    def test_true3227_dual_adam_lr_counter_and_rng_restoration(self):
        original=d.probe.tree_sha(self.parent)
        restored=self.restored()
        self.assertEqual(d.probe.tree_sha(snapshot(restored)),original)
        self.assertEqual(restored[2].param_groups[0]['lr'],.0001)
        self.assertEqual(restored[3].param_groups[0]['lr'],.0002)
        update(restored)
        expected=snapshot(restored,3228)
        again=self.restored();update(again)
        self.assertEqual(d.probe.tree_sha(snapshot(again,3228)),d.probe.tree_sha(expected))
        for optimizer in again[2:]:
            self.assertTrue(all(float(v['step'])==3228 for v in optimizer.state.values()))
        self.assertEqual(d.probe.tree_sha(self.parent),original)

    def test_wrong_identity_configuration_saved_lr_counter_and_ownership_rejected(self):
        for mutation in ('step','contract','baseconfig','effective_lr','s_lr','d_lr','counter','nonfinite','rng'):
            p=copy.deepcopy(self.parent)
            if mutation=='step':p['global_step']=3226
            if mutation=='contract':p['run_contract_sha256']='wrong'
            if mutation=='baseconfig':p['configuration']['lambda_dist']=1
            if mutation=='effective_lr':p['effective_optimizer_lrs']['student']=.0002
            if mutation=='s_lr':p['student_optimizer']['param_groups'][0]['lr']=.0002
            if mutation=='d_lr':p['discriminator_optimizer']['param_groups'][0]['lr']=.0001
            if mutation=='counter':next(iter(p['student_optimizer']['state'].values()))['step']=torch.tensor(3226.)
            if mutation=='nonfinite':next(iter(p['student_optimizer']['state'].values()))['exp_avg'].flatten()[0]=float('nan')
            if mutation=='rng':p['rng_state'].pop('numpy')
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):d.restore_actual_parent(p,*models(),self.config,torch.device('cpu'))
        items=models()
        wrong=torch.optim.Adam(items[1].parameters(),lr=.0001,betas=(.5,.9))
        with self.assertRaises(RuntimeError):d.restore_actual_parent(self.parent,items[0],items[1],wrong,items[3],self.config,torch.device('cpu'))
        items=models();items[2].param_groups[0]['lr']=.0002
        with self.assertRaises(RuntimeError):d.restore_actual_parent(self.parent,*items,self.config,torch.device('cpu'))

    def test_inherited_six_windows_seven_monitors_and_first_new_check(self):
        windows,monitors=self.histories()
        self.assertEqual([r['updates'] for r in windows],[20,40,60,80,100,120])
        self.assertEqual([r['updates'] for r in monitors],[0,20,40,60,80,100,120])
        self.assertEqual(monitors[-1]['global_step'],3227)
        new=copy.deepcopy(windows[-1]);new.update(global_step=3247,updates=140,first_global_step=3228)
        # 旧100窗口6.0058、120窗口6.00999，追加仅微弱上涨也须立即停止，不能再要求2%。
        new['means']['student_total']=windows[-1]['means']['student_total']+1e-5
        anchor={'mel_l1':.19941353797912598,'prediction_rms':.9994737505912781}
        result=d.trend_state(windows+[new],monitors,anchor,monitors[-1]['development'])
        self.assertIn('moving_total_three_windows_rising',result['stop_reasons'])
        self.assertEqual(len(monitors),7)  # 开始重放不会添加第二份3227点。

    def test_strict_rules_no_old_percentage_gates_and_discriminator_alone(self):
        self.assertTrue(d.strict_three([1,1.000001,1.000002]))
        self.assertFalse(d.strict_three([1,1,1.01]))
        self.assertFalse(d.strict_three([1,1.01,1]))
        with self.assertRaises(RuntimeError):d.strict_three([1,float('nan'),2])
        windows,monitors=self.histories();base=copy.deepcopy(monitors[-1]['development'])
        for i,v in enumerate((.19480,.1948001,.1948002)):
            monitors[-3+i]['development']['mel_l1']=v
        anchor={'mel_l1':.19941353797912598,'prediction_rms':.9994737505912781}
        result=d.trend_state(windows,monitors,anchor,base)
        self.assertIn('development_three_checks_rising',result['stop_reasons'])
        windows,monitors=self.histories()
        for i in range(3):windows[-3+i]['means']['discriminator']=1+i/1000000
        result=d.trend_state(windows,monitors,anchor,monitors[-1]['development'])
        self.assertIn('discriminator',result['strictly_rising_loss_fields'])
        self.assertEqual(result['stop_reasons'],[])

    def test_single_gt_onepercent_and_original_s0_rms_guards(self):
        windows,monitors=self.histories();base=copy.deepcopy(monitors[-1]['development'])
        anchor={'mel_l1':.19941353797912598,'prediction_rms':.9994737505912781}
        monitors[-1]['development']['mel_l1']=base['mel_l1']*1.01
        self.assertNotIn('development_above3227_by1percent',d.trend_state(windows,monitors,anchor,base)['stop_reasons'])
        monitors[-1]['development']['mel_l1']+=1e-9
        self.assertIn('development_above3227_by1percent',d.trend_state(windows,monitors,anchor,base)['stop_reasons'])
        monitors[-1]['development']['prediction_rms']=anchor['prediction_rms']*.799
        self.assertIn('original_S0_guard',d.trend_state(windows,monitors,anchor,base)['stop_reasons'])

    def test_first3228_and3688_cross_shuffle_plan_excludes_all52(self):
        development,_=d.resume.load_development(d.DEV/'development_contract.json')
        plan=d.resume.GlobalBatchPlan(development['retained_train_indices'])
        rng=d.probe.tree_sha(d.training.capture_rng_state())
        for step in (3228,3247,3687,3688,3707):
            expected=development['retained_train_indices'].copy()
            random.Random(1234+(step-1)//1229).shuffle(expected)
            offset=((step-1)%1229)*32
            self.assertEqual(plan.for_step(step),expected[offset:offset+32])
            self.assertFalse(set(plan.for_step(step))&set(development['excluded_train_indices']))
        self.assertEqual(rng,d.probe.tree_sha(d.training.capture_rng_state()))
        self.assertEqual(list(range(d.START+1,d.STOP+1))[0],3228)
        self.assertEqual(len(range(d.START+1,d.STOP+1)),480)

    def test_monitor_replay_exact32_rng_and_nonpersistent_frozen_buffers(self):
        _,monitors=self.histories();expected=monitors[-1]
        d.validate_replay(copy.deepcopy(expected),expected)
        wrong=copy.deepcopy(expected);wrong['fixed_train']['diagnostic_frozen_target_score']+=.001
        with self.assertRaises(RuntimeError):d.validate_replay(wrong,expected)
        wrong=copy.deepcopy(expected);wrong['development']['per_utterance_mel_l1'].pop()
        with self.assertRaises(RuntimeError):d.validate_replay(wrong,expected)
        model=nn.Linear(2,1);model.register_buffer('window',torch.ones(3),persistent=False)
        d.training.set_requires_grad(model,False)
        frozen={'probe':model};fingerprints={'probe':d.pair.model_sha(model)}
        rng=d.probe.tree_sha(d.training.capture_rng_state())
        def call():
            random.random(),np.random.random(),torch.randn(5)
            return 5
        self.assertEqual(d.monitored(call,frozen,fingerprints,torch.device('cpu')),5)
        self.assertEqual(rng,d.probe.tree_sha(d.training.capture_rng_state()))
        def corrupt():model.window.add_(1)
        with self.assertRaises(RuntimeError):d.monitored(corrupt,frozen,fingerprints,torch.device('cpu'))

    def test_half_step_failure_never_advances_or_saves(self):
        state={'completed_global_step':3227,'new_updates':0,'updates':120,'last_full_saved_global_step':3227,
            'partial_batch_may_have_updated_D':False}
        def failed():raise RuntimeError('D updated then S failed')
        with self.assertRaises(RuntimeError):d.complete_batch(state,3228,failed)
        self.assertEqual(state['completed_global_step'],3227)
        self.assertTrue(state['partial_batch_may_have_updated_D'])
        # 即使不完整状态的游标被外部写成新步，保存函数仍拒绝半步标记。
        state['completed_global_step']=3228
        with tempfile.TemporaryDirectory(prefix='half480_cpu_') as temp:
            with self.assertRaises(RuntimeError):d.save_final(Path(temp),state,{},*models(),d.FinalBudget(Path(temp),10000,1000))
        safe={'completed_global_step':3227,'partial_batch_may_have_updated_D':False}
        d.complete_batch(safe,3228,lambda:{key:1. for key in d.pair.LOSS_KEYS})
        self.assertEqual(safe['completed_global_step'],3228)
        self.assertEqual(safe['new_updates'],1)

    def test_budget_is_only_one_full_one_s_plus_unchanged_reserves(self):
        budget=d.FinalBudget(Path('/tmp'),1424630213,309265943)
        required=1424630213+309265943+3*1024**3+128*1024**2
        with mock.patch.object(d.pair,'available_bytes',return_value=required):self.assertEqual(budget.require()['required_bytes'],required)
        with mock.patch.object(d.pair,'available_bytes',return_value=required-1):
            with self.assertRaises(d.pair.InsufficientDisk):budget.require()
        budget.committed('full');self.assertEqual(budget.require()['required_bytes'],required-1424630213)

    def test_disk_stop_cannot_claim_unsaved_completed_step(self):
        # 模拟上一完整新批已结束、下一批门禁发现空间下降，终点写盘也无法维持系统余量。
        items=self.restored();update(items)
        state={'status':'running','completed_global_step':3228,'new_updates':1,'updates':121,
            'last_full_saved_global_step':3227,'full_snapshot':None,'student_snapshot':None,
            'partial_batch_may_have_updated_D':False,'reason':'disk_space_guard'}
        with tempfile.TemporaryDirectory(prefix='half480_lowdisk_') as temp:
            path=Path(temp);budget=d.FinalBudget(path,100000,10000)
            with mock.patch.object(d.pair,'available_bytes',return_value=d.pair.SYSTEM_RESERVE):
                with self.assertRaises(d.pair.InsufficientDisk):budget.require()
                try:d.save_final(path,state,{'global_step':3228},*items,budget)
                except d.pair.InsufficientDisk as error:
                    stream=io.StringIO()
                    with contextlib.redirect_stdout(stream):d.pair.report_failure(path,state,error)
                else:self.fail('Unsafe final save was unexpectedly accepted')
            self.assertEqual(state['status'],'failed')
            self.assertEqual(state['completed_global_step'],3228)
            self.assertEqual(state['last_full_saved_global_step'],3227)
            self.assertFalse(state['partial_batch_may_have_updated_D'])
            self.assertEqual(list(path.glob('*.pt')),[])
            self.assertIn('"last_full_saved_global_step": 3227',stream.getvalue())

    def save_case(self,path,fault=None):
        items=self.restored();update(items)
        state={'completed_global_step':3228,'new_updates':1,'updates':121,'last_full_saved_global_step':3227,
            'full_snapshot':None,'student_snapshot':None,'partial_batch_may_have_updated_D':False}
        common={'global_step':3228,'effective_optimizer_lrs':dict(d.ACTUAL_LRS),'run_contract_sha256':'test-contract'}
        budget=d.FinalBudget(path,100000,10000);save=d.training.atomic_torch_save
        def injected(value,filename):
            if fault=='full':raise RuntimeError('full failure')
            if filename.name.startswith('student') and fault in ('s_runtime','s_os'):
                raise RuntimeError('S failure') if fault=='s_runtime' else OSError('S failure')
            save(value,filename)
        try:
            with mock.patch.object(d.training,'atomic_torch_save',side_effect=injected):d.save_final(path,state,common,*items,budget)
        except Exception as error:return state,error,items
        return state,None,items

    def test_true_final_full_s_mapping_dualadam_rng_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='half480_save_') as temp:
            path=Path(temp);state,error,items=self.save_case(path)
            self.assertIsNone(error)
            self.assertEqual(sorted(p.name for p in path.glob('*.pt')),['final_full.pt','student_step_003228.pt'])
            full=torch.load(path/'final_full.pt',map_location='cpu');student=torch.load(path/'student_step_003228.pt',map_location='cpu')
            self.assertEqual(d.probe.tree_sha(full['student']),d.probe.tree_sha(student['student']))
            self.assertEqual(d.probe.tree_sha(full['student_optimizer']),d.probe.tree_sha(items[2].state_dict()))
            self.assertEqual(d.probe.tree_sha(full['discriminator_optimizer']),d.probe.tree_sha(items[3].state_dict()))
            self.assertEqual(d.probe.tree_sha(full['rng_state']),d.probe.tree_sha(d.training.capture_rng_state()))
            self.assertEqual((path/'final_full.pt').stat().st_mode&0o222,0)
            with self.assertRaises(RuntimeError):d.save_final(path,state,{},*items,d.FinalBudget(path,100000,10000))

    def test_save_failures_preserve_actual_full_cursor_and_truthful_metadata(self):
        for fault in ('full','s_runtime','s_os','metadata'):
            with tempfile.TemporaryDirectory(prefix='half480_fault_') as temp:
                with contextlib.ExitStack() as stack:
                    if fault=='metadata':stack.enter_context(mock.patch.object(d.base,'save_json',side_effect=OSError('metadata failure')))
                    state,error,_=self.save_case(Path(temp),fault)
                    self.assertIsNotNone(error)
                    self.assertEqual(state['last_full_saved_global_step'],3227 if fault=='full' else 3228)
                    stream=io.StringIO()
                    with contextlib.redirect_stdout(stream):d.pair.report_failure(Path(temp),state,error)
                    self.assertEqual(state['status'],'failed')
                    self.assertIn('last_full_saved_global_step',stream.getvalue())

    def test_source_locks_runtime_and_does_not_patch_old_modules_or_score(self):
        source=Path(d.__file__).read_text();tree=ast.parse(source)
        changed=[node for node in ast.walk(tree) if isinstance(node,(ast.Assign,ast.AugAssign))]
        for node in changed:
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('pair','resume','base','training') for t in targets))
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
        self.assertEqual(sum(node.func.attr=='train_batch' for node in calls),1)
        self.assertTrue(any(node.func.attr=='acquire_runtime_locks' for node in calls))
        self.assertTrue(any(node.func.attr=='assert_rented_server' for node in calls))
        self.assertIn("('3.8.15', '1.13.1+cu116', '11.6')",source)
        self.assertIn("/root/miniconda3/envs/myconda/bin/python",source)
        self.assertIn("automatic_metric_evaluation': False",source)


if __name__=='__main__':
    unittest.main()
