"""只在租用服务器CPU验证新诊断；不调用真实学生/教师或任何GPU。"""

import copy  # 故障注入只改独立小字典，不改真实收据。
import contextlib
import io
import json
import math
import random
import subprocess
import sys
import tempfile  # 所有临时写入严格限于测试自己创建的目录。
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import inspect_acdd_timestep_cross_3327_v1 as m


class Tiny(torch.nn.Module):
    # 一个参数、一个非持久buffer及混合子模块模式，覆盖state_dict看不到的状态。
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2., 3.]))
        self.child = torch.nn.Identity()
        self.child.eval()
        self.register_buffer('temporary_window', torch.tensor([4.]), persistent=False)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_available():
            raise RuntimeError('Run these tests with CUDA_VISIBLE_DEVICES empty')
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)  # 仅控制CPU测试耗时；主诊断不更改线程或精度设置。

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_sampling_is_fixed_excluded_and_independent(self):
        retained = list(range(39335))
        old = retained[100:132]
        state = random.getstate()
        plan = m.choose_plan(retained, old)
        self.assertEqual(state, random.getstate())
        self.assertEqual(plan, m.choose_plan(retained, old))
        self.assertEqual(len(set(plan['selected_indices'])), 64)
        self.assertFalse(set(plan['selected_indices']) & set(old))
        self.assertEqual(plan['sentinel_timesteps'], [1, 950, 999, 1000])
        for layer, row in enumerate(plan['strata_timesteps']):
            self.assertEqual(len(row), 64)
            self.assertTrue(all(100 * layer + 1 <= t <= 100 * layer + 100 for t in row))
        with self.assertRaises(RuntimeError):
            m.choose_plan(retained[:-1], old)

    def test_derivative_finite_difference_and_zero_convention(self):
        y = torch.tensor([[[.3, -.7], [1.3, -.4]], [[.8, .2], [-.9, .6]]], dtype=torch.float64)
        target = torch.zeros_like(y)
        weights = torch.tensor([.31, .77], dtype=torch.float64)
        direction = torch.tensor([[[.2, .1], [.3, -.1]], [[-.2, .4], [.3, -.1]]], dtype=torch.float64)
        scalar = lambda value: ((value-target).abs().flatten(1).mean(1) * weights).mean()
        h = 1e-6
        numerical = float((scalar(y+h*direction)-scalar(y-h*direction))/(2*h))
        analytic = float((weights[:,None,None] * (y-target).sign() / y[0].numel() / 2 * direction).sum())
        self.assertAlmostEqual(numerical, analytic, places=10)
        residual = y-target
        residual[0,0,0] = 0  # L1恰为零的点采用生产sign(0)=0，不称其唯一可微导数。
        stats = m.derivative_statistics(residual, weights)
        self.assertEqual(stats['nonzero_fraction'], [.75, 1.])
        for actual, expected in zip(stats['l2'], stats['expected_l2']):
            self.assertAlmostEqual(actual, expected, places=15)
        zero = m.derivative_statistics(torch.zeros_like(y), weights)
        self.assertEqual(zero['l2'], [0.,0.])

    def test_actual_N_and_batch_layer_coefficient_scales(self):
        residual = torch.ones((32,80,128), dtype=torch.float32)
        residual[0,:,:64] = 0
        weights = torch.linspace(.00004928252, .99, 32)
        per = m.derivative_statistics(residual, weights)
        self.assertEqual(per['nonzero_fraction'][0], .5)
        for divisor, layers, coefficient in [(32,1,1),(32,1,45),(64,1,1),(64,10,1),(64,10,45)]:
            value = m.derivative_statistics(residual, weights, divisor, layers, coefficient)
            for index in range(32):
                self.assertAlmostEqual(value['l2'][index], per['l2'][index]*coefficient/divisor/layers, places=14)
                self.assertAlmostEqual(value['rms'][index], value['expected_rms'][index], places=14)
        with self.assertRaises(RuntimeError):
            m.derivative_statistics(residual, weights, 0)

    def test_cross_cells_use_target_columns_and_decompose(self):
        y = {3227: torch.tensor([[[1.,2.]]]), 3327: torch.tensor([[[3.,-1.]]])}
        r = {3227: torch.tensor([[[0.,0.]]]), 3327: torch.tensor([[[2.,1.]]])}
        before = {k:m.tensor_id(v) for k,v in r.items()}
        output = m.crossed(y,r,torch.tensor([.5]))
        self.assertEqual(output['cells']['y3327_R3227']['weighted'], 1.)
        self.assertEqual(output['cells']['y3227_R3327']['weighted'], .5)
        d = output['decomposition']['weighted']
        self.assertAlmostEqual(d['diagonal_change'], d['symmetric_student']+d['symmetric_target'])
        self.assertEqual(before, {k:m.tensor_id(v) for k,v in r.items()})
        self.assertFalse(any('3247' in key for key in output['cells']))
        field = 'fixed_target_output_derivative_per_sentence'
        self.assertNotIn(field, output['target_drift'])
        self.assertTrue(all(field in cell for cell in output['cells'].values()))

    def test_nonfinite_and_frozen_tolerances_reject(self):
        with self.assertRaises(RuntimeError):
            m.cell_readout(torch.tensor([[[float('inf')]]]), torch.zeros(1,1,1), torch.ones(1))
        with self.assertRaises(RuntimeError):
            m.require_close(1.00003, 1., 2e-5)
        self.assertLess(m.require_close(1.0000001,1.,1e-7,1e-6),2e-7)

    def test_full_restore_on_exception_includes_nonpersistent_and_rng(self):
        model = Tiny()
        model.weight.grad = torch.tensor([7.,8.])
        original = m.snapshot(model)
        rng, flags = m.probe.tree_sha(m.training.capture_rng_state()), m.flags()
        restored, models = {}, {}
        with self.assertRaisesRegex(ValueError,'intentional'):
            with m.preserved_state(models,restored) as register:
                register('student',model)
                with torch.no_grad():
                    model.weight.add_(10)
                    model.temporary_window.zero_()
                model.train()
                model.weight.requires_grad_(False)
                model.weight.grad = None
                random.random(); np.random.rand(); torch.rand(3)
                torch.backends.cudnn.benchmark = not flags['benchmark']
                raise ValueError('intentional')
        self.assertEqual(m.probe.tree_sha(original),m.probe.tree_sha(m.snapshot(model)))
        self.assertEqual(rng,m.probe.tree_sha(m.training.capture_rng_state()))
        self.assertEqual(flags,m.flags())
        self.assertTrue(restored['models']['student'])
        self.assertTrue(restored['rng_restored'])

    def test_restore_failure_is_not_claimed_success(self):
        receipt = {}
        with mock.patch.object(m,'restore_model',side_effect=OSError('injected restoration failure')):
            with self.assertRaisesRegex(RuntimeError,'restoration failed'):
                with m.preserved_state({},receipt) as register:
                    register('student',Tiny())
        self.assertFalse(receipt['models']['student'])
        self.assertTrue(receipt['errors'])
        self.assertTrue(receipt['rng_restored'])

    def test_hash_conflict_and_redirect_rejected(self):
        hashes = {'/a':'123'}
        with self.assertRaises(RuntimeError):
            m.merge_hashes(hashes,{'/a':'456'})
        self.assertEqual(hashes,{'/a':'123'})
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            target = root/'real'; target.write_text('small test')
            link = root/'link'; link.symlink_to(target)
            with self.assertRaises(RuntimeError):
                m.regular(link)

    def test_budget_atomic_mapping_no_overwrite_disk_and_time(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(m.pair,'available_bytes',return_value=10*1024**3):
            output = Path(root)/'new'; output.mkdir()
            budget = m.Budget(output,time.monotonic(),300)
            digest = budget.write('run_identity.json',{'plan':[1,2]})
            self.assertEqual(digest,m.base.sha256_file(output/'run_identity.json'))
            with self.assertRaises(RuntimeError):
                budget.write('run_identity.json',{'plan':[3]})
            with mock.patch.object(m.pair,'available_bytes',return_value=m.SYSTEM_RESERVE):
                with self.assertRaises(RuntimeError):
                    budget.check()
            budget.started = time.monotonic()-301
            with self.assertRaises(RuntimeError):
                budget.check()
            budget.write('failure.json',{'status':'failed'},enforce_time=False)
            with self.assertRaises(RuntimeError):
                budget.write('results.json',{'payload':'x'*m.MAX_RESULT},enforce_time=False)

    def test_two_anchors_required_before100_replayed_steps(self):
        t = torch.arange(1,33,dtype=torch.long)
        drawn = {'source_noise':torch.zeros(32,1,1),'score_noise':torch.ones(32,1,1),'score_timesteps':t}
        raw = dict(drawn)
        anchor = {'input_identities':{key:m.tensor_id(value) for key,value in drawn.items()}}
        rows = [{'global_step':step} for step in range(3228,3328)]
        budget = mock.Mock()
        with mock.patch.object(m,'draw_original',return_value=drawn) as draw:
            result = m.historical_timesteps(raw,anchor,rows,budget,torch.device('cpu'))
            self.assertEqual([c.args[0] for c in draw.call_args_list], [3108,3328]+list(range(3228,3328)))
        self.assertEqual(sum(len(row['timesteps']) for row in result['steps']),3200)
        self.assertEqual(result['windows'][0]['counts'],[640]+[0]*9)
        bad = copy.deepcopy(anchor); bad['input_identities']['score_timesteps']['sha256']='bad'
        with mock.patch.object(m,'draw_original',return_value=drawn) as draw:
            with self.assertRaises(RuntimeError):
                m.historical_timesteps(raw,bad,rows,budget,torch.device('cpu'))
            self.assertEqual(draw.call_count,2)

    def test_history_flat_rows_and_absolute_order(self):
        retained = list(range(39335)); planner = m.resume.GlobalBatchPlan(retained)
        rows = [{'global_step':step,'indices':planner.for_step(step),**{key:1. for key in m.pair.LOSS_KEYS}} for step in range(3228,3328)]
        m.validate_step_rows(rows,retained)
        rows[0]['indices'] = rows[0]['indices'][::-1]
        with self.assertRaises(RuntimeError):
            m.validate_step_rows(rows,retained)

    def test_stratum_aggregate_excludes_sentinels(self):
        names = ['y3227_R3227','y3327_R3227','y3227_R3327','y3327_R3327']
        conditions = {'stratum_'+str(i):{'cells':{name:{'weighted':float(i),'unweighted':float(i),'weighted45':45.*i} for name in names}} for i in range(10)}
        conditions['sentinel_1000'] = {'cells':{name:{'weighted':1e9} for name in names}}
        result = {'conditions':conditions}
        history = {'windows':[{'fractions':[.1]*10}]}
        m.aggregate(result,history)
        self.assertEqual(result['ten_stratum_uniform_means'][names[0]]['weighted'],4.5)
        self.assertAlmostEqual(history['windows'][0]['fixed_endpoint_composition_predictions_not_historical_losses'][names[0]],4.5)

    def test_real_metadata_and_payloads_only_CPU_no_forward(self):
        # 真收据/权重只读验schema，防止toy测试掩盖字段名字不一致；不创建网络。
        files,contracts,receipts,development,rows,config = m.load_metadata()
        self.assertEqual(len(rows),100)
        self.assertEqual(len(development['retained_train_indices']),39335)
        self.assertEqual(len(development['excluded_train_indices']),52)
        for step in m.STEPS:
            payload = m.resume.load_torch_checkpoint(m.STATE[step]['path'])
            m.validate_payload(payload,step,config)
            self.assertEqual(receipts[step]['completed_global_step'],step)
            self.assertIn('fixed_train',receipts[step]['last_monitor'])
            del payload
        raw = m.resume.load_torch_checkpoint(m.INPUT_FILE)
        self.assertEqual({k:m.tensor_id(v)['sha256'] for k,v in raw.items()},contracts[3227]['input_sha256'])
        self.assertEqual(tuple(raw['bnf'].shape),(32,144,128))
        self.assertEqual(tuple(raw['spk_emb'].shape),(32,256))

    def test_real_prepare_inputs_CPU_schema_and192_feature_files(self):
        # 仅在CPU读取原64句特征、BNF对齐/裁剪；新CPU噪声只验shape，不冒充历史CUDA字节。
        contracts={step:m.read_json(item['contract']) for step,item in m.STATE.items()}
        development=m.read_json(m.DEV/'development_contract.json')
        files={}
        receipt={}
        with m.preserved_state({},receipt):
            old,batches,target,plan=m.prepare_inputs(files,development,contracts,torch.device('cpu'))
        self.assertEqual(len(plan['utterances']),64)
        self.assertEqual(len(files),192)
        self.assertEqual(sum(plan['speaker_counts'].values()),64)
        self.assertFalse(set(plan['selected_indices'])&set(development['excluded_train_indices']))
        self.assertFalse(set(plan['selected_indices'])&set(plan['old_TRAIN3108_indices']))
        for batch in batches:
            self.assertEqual(tuple(batch['bnf'].shape),(32,144,128))
            self.assertEqual(tuple(batch['spk_emb'].shape),(32,256))
            self.assertEqual(tuple(batch['source_noise'].shape),(32,80,128))
        self.assertEqual(m.tensor_id(target)['sha256'],contracts[3227]['fixed_teacher_target_sha256'])
        self.assertTrue(receipt['rng_restored'])

    def test_preflight_failure_reports_without_output_and_closes_three_locks(self):
        # 源码身份不符时在GPU门禁前失败；不生成假恢复收据或新输出目录。
        handles=[mock.Mock() for _ in range(3)]
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'must_remain_absent'
            console=io.StringIO()
            with mock.patch.object(m,'OUTPUT',output),mock.patch.object(m,'timeout_parent',return_value={'CPU_test':True}), \
                 mock.patch.object(m.training,'acquire_runtime_locks',return_value=handles), \
                 mock.patch.object(m.training,'git_head',return_value='invalid'), \
                 mock.patch.object(sys,'argv',['diagnostic','--run-approved']),contextlib.redirect_stdout(console):
                with self.assertRaisesRegex(RuntimeError,'Frozen source'):
                    m.main()
            self.assertFalse(output.exists())
            for handle in handles:
                handle.close.assert_called_once()
            self.assertIn('TIMESTEP_FAILED',console.getvalue())
            self.assertIn('"restoration": {}',console.getvalue())
            self.assertIn('TIMESTEP_LOCKS_RELEASED',console.getvalue())

    def test_internal_deadline_and_SIGTERM_restore_in_CPU_subprocess(self):
        # 真子进程信号测试：没有CUDA或项目网络，截止后仅执行恢复，返回明确失败状态。
        program="""
import os,signal,sys,time,torch
import inspect_acdd_timestep_cross_3327_v1 as m
g=m.DeadlineSignals(time.monotonic(),.1 if sys.argv[1]=='deadline' else 10)
model=torch.nn.Linear(2,2)
saved=m.snapshot(model)
models,receipt={},{}
g.install()
try:
    with m.preserved_state(models,receipt,g.cleanup_only) as register:
        register('toy',model)
        with torch.no_grad(): model.weight.add_(1)
        if sys.argv[1]=='term': os.kill(os.getpid(),signal.SIGTERM)
        else: time.sleep(.3)
    raise AssertionError('interruption did not occur')
except m.DiagnosticInterruption:
    assert m.probe.tree_sha(saved)==m.probe.tree_sha(m.snapshot(model))
    assert receipt['models']['toy'] and receipt['rng_restored']
    print('EXPECTED_INTERRUPTION_RESTORED')
finally: g.close()
"""
        for mode in ('deadline','term'):
            result=subprocess.run([sys.executable,'-c',program,mode],cwd=str(Path(__file__).resolve().parent),
                                  text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('EXPECTED_INTERRUPTION_RESTORED',result.stdout)

    def test_complete_schema_size_under_frozen_cap(self):
        # 保留14条件、64句、全部四格和导数字段估计实际JSON上界，不删难例或端点。
        generator = torch.Generator().manual_seed(17)
        y = {step:torch.randn(64,2,8,generator=generator) for step in m.STEPS}
        r = {step:torch.randn(64,2,8,generator=generator) for step in m.STEPS}
        weights = torch.linspace(.00004928252,.99,64)
        value = m.crossed(y,r,weights)
        conditions = {name:copy.deepcopy(value) for name in [*('stratum_'+str(i) for i in range(10)),*('sentinel_'+str(t) for t in [1,950,999,1000])]}
        result = {'readouts':{'conditions':conditions},'reserved_other_fields':'x'*200000}
        encoded = json.dumps(result,separators=(',',':'),allow_nan=False).encode()
        self.assertLess(len(encoded),m.MAX_RESULT)
        self.assertEqual(len(conditions),14)

    def test_small_CPU_workflow_exact6S58T_and_shared_target_columns(self):
        # 独立toy仅替代网络，保留compute完整循环/四格/重放，真实项目模型没有前向。
        teacher, student = Tiny(), Tiny()
        config = m.base.replace(m.base.PAPER_STUDENT_CONFIG,num_workers=0)
        diffusion = m.base.VoiceGradDiffusion(config.diffusion_steps,config.cosine_offset)
        x = torch.linspace(-.5,.5,64).reshape(32,1,2)
        old = {'x0':x,'bnf':torch.zeros(32,1,2),'spk_emb':torch.zeros(32,2),
               'source_noise':torch.ones_like(x)*.1,'score_noise':torch.ones_like(x)*.2,
               'score_timesteps':torch.arange(1,33,dtype=torch.long)}
        old['q950'] = x+old['source_noise']*.1
        batches = [{k:v.clone() for k,v in old.items() if k not in ('q950','score_timesteps')} for _ in range(2)]
        old_target = x*.8
        plan = {'strata_timesteps':[[100*i+1]*64 for i in range(10)],'sentinel_timesteps':[1,950,999,1000]}
        payloads,receipts = {},{}
        for step,value in [(3227,2.),(3327,3.)]:
            state = copy.deepcopy(student.state_dict()); state['weight'][0]=value
            payloads[str(m.STATE[step]['path'])]={'student':state}
            y=x+value*.01; target=.9*y+.02
            w=diffusion.sqrt_alphas_cumprod[old['score_timesteps']]
            receipts[step]={'last_monitor':{'fixed_train':{
                'diagnostic_frozen_target_score':m.cell_readout(y,old_target,w)['weighted'],
                'primary_moving_score':m.cell_readout(y,target,w)['weighted'],
                'teacher_target_weighted_l1_drift':m.cell_readout(target,old_target,w)['weighted'],
                'train_gt_l1':float((y-x).abs().mean()),'train_prediction_rms':float(y.square().mean().sqrt())}}}
        y0=x+.02; target0=.9*y0+.02
        archived={'states':{'3227':{'prediction':m.tensor_id(y0),'teacher_target':m.tensor_id(target0)}}}
        teacher_calls=[]
        def predict(s,diff,mel,spk,bnf,start,noise):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(start,950)
            return mel+s.weight[0]*.01,mel+noise*.1
        def score(y,tmodel,diff,t,noise,spk,bnf,**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(kwargs,{'target_mode':'clean_x0','weight_mode':'sqrt_alpha_bar'})
            target=.9*y+.02
            teacher_calls.append((m.tensor_id(y),m.tensor_id(target),t.tolist()))
            return ((y-target).abs().flatten(1).mean(1)*diff.sqrt_alphas_cumprod[t]).mean(),target
        with mock.patch.object(m.base,'load_teacher_and_student',return_value=(teacher,student,{})), \
             mock.patch.object(m.resume,'load_torch_checkpoint',side_effect=lambda path:copy.deepcopy(payloads[str(path)])), \
             mock.patch.object(m,'validate_payload'),mock.patch.object(m,'read_json',return_value=archived), \
             mock.patch.object(m.training,'make_student_prediction',side_effect=predict) as sf, \
             mock.patch.object(m.training,'score_distillation_loss',side_effect=score):
            models,receipt={},{}
            with m.preserved_state(models,receipt) as register:
                result=m.compute(models,register,old,batches,old_target,plan,config,receipts,mock.Mock(),torch.device('cpu'))
        self.assertEqual(sf.call_count,6)
        self.assertEqual(len(teacher_calls),58)
        self.assertEqual(result['actual_forward_counts'],{'student':6,'teacher':58})
        self.assertEqual(len(result['conditions']),14)
        self.assertEqual(len(result['conditions']['sentinel_1000']['cells']['y3327_R3227']['per_weighted']),64)
        field='fixed_target_output_derivative_per_sentence'
        for entry in result['old_TRAIN3108'].values():
            self.assertNotIn(field,entry['target_drift'])
            self.assertIn(field,entry['A_oldR3107'])
            self.assertIn(field,entry['B_refreshed'])
        self.assertTrue(all(receipt['models'].values()))


if __name__ == '__main__':
    unittest.main()
