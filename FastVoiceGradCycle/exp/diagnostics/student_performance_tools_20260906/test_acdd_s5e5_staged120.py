# 只在租用CPU验证两阶段控制、真实小Adam和保存；不启动实际学生训练。
"""CPU acceptance for the5e-5 staged20-plus100 development branch."""

import ast
import contextlib
import copy
import io
import json
import os
import tempfile  # 只使用本测试创建的KB级临时目录，测试结束自动清理。
import unittest
from pathlib import Path
from unittest import mock

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
import torch
import run_acdd_s5e5_staged120_v1 as d
import test_acdd_half_lr_continue480 as fixture  # 复用已审小CPU模型/双Adam更新夹具，不运行其main。


class StagedS5e5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        d.training.seed_everything(551)
        cls.items=fixture.models()
        for _ in range(3227):fixture.update(cls.items)
        cls.parent=fixture.snapshot(cls.items)
        cls.config=d.base.replace(d.base.PAPER_STUDENT_CONFIG,num_workers=0)

    def setUp(self):
        patch=mock.patch.object(d.pair,'available_bytes',return_value=20*1024**3)
        patch.start();self.addCleanup(patch.stop)

    def restored(self):
        items=fixture.models()
        record=d.restore_then_change_student_lr(self.parent,*items,self.config,torch.device('cpu'))
        return items,record

    def test_restore_only_student_lr_changes_all_moments_D_and_rng_unchanged(self):
        before=d.probe.tree_sha(self.parent)
        items,record=self.restored()
        expected=copy.deepcopy(self.parent['student_optimizer'])
        expected['param_groups'][0]['lr']=.00005
        self.assertEqual(d.probe.tree_sha(items[2].state_dict()),d.probe.tree_sha(expected))
        self.assertEqual(d.probe.tree_sha(items[3].state_dict()),d.probe.tree_sha(self.parent['discriminator_optimizer']))
        self.assertEqual(d.probe.tree_sha(items[0].state_dict()),d.probe.tree_sha(self.parent['student']))
        self.assertEqual(d.probe.tree_sha(items[1].state_dict()),d.probe.tree_sha(self.parent['discriminator']))
        self.assertEqual(d.probe.tree_sha(d.training.capture_rng_state()),d.probe.tree_sha(self.parent['rng_state']))
        self.assertEqual(d.probe.tree_sha(self.parent),before)
        self.assertEqual(record['effective_optimizer_lrs'],{'student':.00005,'discriminator':.0002})

    def test_true_next_update3228_replays_and_keeps_effective_lr_counters(self):
        items,_=self.restored();fixture.update(items)
        expected=d.probe.tree_sha([model.state_dict() for model in items])
        again,_=self.restored();fixture.update(again)
        self.assertEqual(d.probe.tree_sha([model.state_dict() for model in again]),expected)
        d.pair.validate_effective_adam(*again[2:],*again[:2],.00005,3228)
        for optimizer in again[2:]:self.assertTrue(all(float(v['step'])==3228 for v in optimizer.state.values()))
        self.assertEqual(again[2].param_groups[0]['lr'],.00005)
        self.assertEqual(again[3].param_groups[0]['lr'],.0002)

    def gate_args(self,gt=d.CONTROL_GT,total=6.):
        return ({'global_step':3247,'first_global_step':3228,'count':20,'means':{'student_total':total}},
                {'global_step':3247,'development':{'mel_l1':gt}}, {'stop_reasons':[]}, .1948028951883316)

    def test_stage1_all_conditions_are_intersection_including_execution_stops(self):
        args=self.gate_args(total=d.CONTROL_TOTAL)
        self.assertTrue(d.stage1_gate(*args)['passed'])  # 数值等于上界允许，但仍需原strict保护另行通过。
        for gt,total,parent in ((d.CONTROL_GT+1e-8,6.,args[-1]),(.194,6.,.193),(.194,d.CONTROL_TOTAL+1e-8,args[-1])):
            changed=list(self.gate_args(gt,total));changed[-1]=parent
            self.assertFalse(d.stage1_gate(*changed)['passed'])
        changed=list(args);changed[2]={'stop_reasons':['moving_total_three_windows_rising']}
        self.assertFalse(d.stage1_gate(*changed)['passed'])
        for reason in ('requested_stop','time_budget','disk_space_guard'):
            result=d.stage1_gate(*args,external_stop_reasons=[reason])
            self.assertFalse(result['passed'])
            self.assertEqual(result['external_stop_reasons'],[reason])
            self.assertIn('execution_guards_clear',result['failed_conditions'])
            with self.assertRaises(RuntimeError):d.require_stage_permission(3248,result)

    def test_stage1_nonfinite_or_wrong_stage_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(RuntimeError):d.stage1_gate(*self.gate_args(gt=value))
        for key,value in (('global_step',3246),('first_global_step',3229),('count',19)):
            args=list(self.gate_args());args[0][key]=value
            with self.assertRaises(RuntimeError):d.stage1_gate(*args)

    def test_stage2_permission_never_enters_without_pass_or_beyond120(self):
        for step in range(3228,3248):d.require_stage_permission(step,None)
        for gate in (None,{'passed':False}):
            with self.assertRaises(RuntimeError):d.require_stage_permission(3248,gate)
        for step in range(3248,3348):d.require_stage_permission(step,{'passed':True})
        for step in (3227,3348):
            with self.assertRaises(RuntimeError):d.require_stage_permission(step,{'passed':True})
        self.assertEqual((d.prior.START,d.prior.STOP,d.prior.ACTUAL_LRS),(3227,3707,{'student':.0001,'discriminator':.0002}))

    def test_real_control20_history_six_windows_seven_monitors_and_batch_plan(self):
        d.resume.verify_files(d.CONTROL_FILES)
        rows,receipt=d.load_control()
        parent=json.loads((d.PARENT/'arm_completion.json').read_text())
        windows,monitors=d.prior.load_history(parent)
        self.assertEqual(len(rows),20);self.assertEqual(len(windows),6);self.assertEqual(len(monitors),7)
        self.assertEqual(windows[-1]['global_step'],3227)
        self.assertEqual(monitors[-1]['global_step'],3227)
        development,_=d.resume.load_development(d.DEV/'development_contract.json')
        plan=d.resume.GlobalBatchPlan(development['retained_train_indices'])
        for row in rows:
            self.assertEqual(plan.for_step(row['global_step']),row['indices'])
            self.assertFalse(set(row['indices'])&set(development['excluded_train_indices']))
        self.assertEqual(plan.batches_per_pass,1229)
        self.assertEqual(receipt['status'],'stopped')
        self.assertEqual(d.base.sha256_file(d.PROTOCOL),d.PROTOCOL_SHA)

    def test_original_strict_trend_can_reject_even_when_control_threshold_passes(self):
        parent=json.loads((d.PARENT/'arm_completion.json').read_text())
        contract=json.loads((d.PAIRED/'run_contract.json').read_text())
        windows,monitors=d.prior.load_history(parent)
        window=copy.deepcopy(windows[-1]);window.update(global_step=3247,first_global_step=3228,count=20)
        window['means']['student_total']=d.CONTROL_TOTAL
        entry=copy.deepcopy(monitors[-1]);entry['global_step']=3247;entry['development']['mel_l1']=d.CONTROL_GT
        trend=d.prior.trend_state(windows+[window],monitors+[entry],contract['original_step0_guard_anchor'],parent['last_monitor']['development'])
        gate=d.stage1_gate(window,entry,trend,parent['last_monitor']['development']['mel_l1'])
        self.assertTrue(gate['checks']['total_not_worse_than_existing_control20'])
        self.assertFalse(gate['passed'])
        self.assertIn('moving_total_three_windows_rising',trend['stop_reasons'])

    def test_first_preupdate_all_seven_losses_match_and_mismatch_rejected(self):
        rows,_=d.load_control();first=rows[0]
        self.assertEqual(d.first_step_check(first,dict(first)),{key:0. for key in d.pair.LOSS_KEYS})
        for key in d.pair.LOSS_KEYS:
            wrong=dict(first);wrong[key]+=1e-3
            with self.assertRaises(RuntimeError):d.first_step_check(first,wrong)
        wrong=dict(first);wrong['dist_raw']=float('nan')
        with self.assertRaises(RuntimeError):d.first_step_check(first,wrong)

    def save_case(self,path,fault=None):
        items,_=self.restored();fixture.update(items)
        state={'status':'running','completed_global_step':3228,'new_updates':1,'updates':121,'last_full_saved_global_step':3227,
               'partial_batch_may_have_updated_D':False,'full_snapshot':None,'student_snapshot':None}
        common={'global_step':3228,'effective_optimizer_lrs':dict(d.ACTUAL_LRS),'run_contract_sha256':'test-contract','stage1_gate':None}
        budget=d.prior.FinalBudget(path,100000,10000);real=d.training.atomic_torch_save
        def injected(value,filename):
            if fault=='full':raise RuntimeError('full failure')
            if filename.name.startswith('student') and fault in ('S_os','S_runtime'):
                raise OSError('S failure') if fault=='S_os' else RuntimeError('S failure')
            real(value,filename)
        try:
            with mock.patch.object(d.training,'atomic_torch_save',side_effect=injected):
                d.prior.save_final(path,state,common,*items,budget)
        except Exception as error:return state,error,items
        return state,None,items

    def test_actual_endpoint_full_S_two_adam_lr_counters_RNG_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='s5e5_endpoint_') as temp:
            path=Path(temp);state,error,items=self.save_case(path)
            self.assertIsNone(error)
            self.assertEqual(sorted(p.name for p in path.glob('*.pt')),['final_full.pt','student_step_003228.pt'])
            full=torch.load(path/'final_full.pt',map_location='cpu');student=torch.load(path/'student_step_003228.pt',map_location='cpu')
            self.assertEqual(d.probe.tree_sha(full['student']),d.probe.tree_sha(student['student']))
            self.assertEqual(full['effective_optimizer_lrs'],d.ACTUAL_LRS)
            self.assertEqual(full['student_optimizer']['param_groups'][0]['lr'],.00005)
            self.assertEqual(full['discriminator_optimizer']['param_groups'][0]['lr'],.0002)
            for key in ('student_optimizer','discriminator_optimizer'):
                self.assertTrue(all(float(x['step'])==3228 for x in full[key]['state'].values()))
            self.assertEqual(d.probe.tree_sha(full['rng_state']),d.probe.tree_sha(d.training.capture_rng_state()))
            with self.assertRaises(RuntimeError):d.prior.save_final(path,state,{},*items,d.prior.FinalBudget(path,100000,10000))

    def test_full_commit_cursor_survives_S_and_metadata_failure(self):
        for fault in ('full','S_os','S_runtime','metadata'):
            with tempfile.TemporaryDirectory(prefix='s5e5_fault_') as temp,contextlib.ExitStack() as stack:
                path=Path(temp)
                if fault=='metadata':stack.enter_context(mock.patch.object(d.base,'save_json',side_effect=OSError('metadata failure')))
                state,error,_=self.save_case(path,fault)
                self.assertIsNotNone(error)
                self.assertEqual(state['last_full_saved_global_step'],3227 if fault=='full' else 3228)
                stream=io.StringIO()
                with contextlib.redirect_stdout(stream):d.pair.report_failure(path,state,error)
                self.assertEqual(state['status'],'failed')
                self.assertIn('last_full_saved_global_step',stream.getvalue())

    def test_half_batch_failure_cannot_advance_or_be_saved(self):
        state={'completed_global_step':3227,'last_full_saved_global_step':3227,'partial_batch_may_have_updated_D':False}
        def fail():raise RuntimeError('D changed and S failed')
        with self.assertRaises(RuntimeError):d.prior.complete_batch(state,3228,fail)
        self.assertEqual(state['completed_global_step'],3227)
        self.assertTrue(state['partial_batch_may_have_updated_D'])
        state['completed_global_step']=3228
        with tempfile.TemporaryDirectory(prefix='s5e5_partial_') as temp:
            with self.assertRaises(RuntimeError):
                d.prior.save_final(Path(temp),state,{},*fixture.models(),d.prior.FinalBudget(Path(temp),100000,10000))

    def test_dynamic_disk_stop_reports_completed_but_unsaved_state(self):
        items,_=self.restored();fixture.update(items)
        state={'completed_global_step':3228,'new_updates':1,'last_full_saved_global_step':3227,
               'partial_batch_may_have_updated_D':False,'full_snapshot':None,'student_snapshot':None}
        with tempfile.TemporaryDirectory(prefix='s5e5_lowdisk_') as temp:
            path=Path(temp);budget=d.prior.FinalBudget(path,100000,10000)
            with mock.patch.object(d.pair,'available_bytes',return_value=d.pair.SYSTEM_RESERVE):
                with self.assertRaises(d.pair.InsufficientDisk):budget.require()
                with self.assertRaises(d.pair.InsufficientDisk):d.prior.save_final(path,state,{'global_step':3228},*items,budget)
            self.assertEqual(state['completed_global_step'],3228)
            self.assertEqual(state['last_full_saved_global_step'],3227)
            self.assertEqual(list(path.glob('*.pt')),[])
        required=1424630213+309265943+3*1024**3+128*1024**2
        self.assertEqual(required,5089339356)
        budget=d.prior.FinalBudget(d.DEV,1424630213,309265943)
        with mock.patch.object(d.pair,'available_bytes',return_value=required-1):
            with self.assertRaises(d.pair.InsufficientDisk):budget.require()

    def test_gpu_busy_check_and_source_single_train_call_no_monkeypatch(self):
        with mock.patch.object(d.subprocess,'run',return_value=mock.Mock(stdout='999999, other-job')):
            with self.assertRaises(RuntimeError):d.assert_gpu_idle()
        source=Path(d.__file__).read_text();tree=ast.parse(source)
        calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
        self.assertEqual(calls.count('train_batch'),1)
        self.assertEqual(calls.count('save_final'),1)
        self.assertIn('acquire_runtime_locks',calls);self.assertIn('assert_rented_server',calls)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Assign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id in ('prior','pair','training','base','resume') for x in targets))
        self.assertIn('args.max_seconds<=600',source)
        self.assertEqual((d.START,d.STAGE1_END,d.STOP),(3227,3247,3347))
        self.assertIn("'automatic_metric_evaluation':False",source)


if __name__=='__main__':
    unittest.main(verbosity=2)
