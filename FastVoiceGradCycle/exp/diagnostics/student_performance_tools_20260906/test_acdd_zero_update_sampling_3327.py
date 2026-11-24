"""租用CPU小图检查：全部120抽样、122T重放、真实无更新和异常恢复。"""

import ast
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile  # 仅在本测试独占临时目录写小JSON，绝不写模型。
import unittest
from pathlib import Path
from unittest import mock

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only rented Matpool CPU tests are allowed')
import torch
from torch import nn
import inspect_acdd_zero_update_sampling_3327_v1 as driver
import test_conversion_score_step as fixture


def tiny_case():
    # 冻结fixture中的两optimizer不参与此检查；测试随后单独证明诊断入口没有构造它们。
    args=fixture.make_fixture(torch.float32)
    teacher,student,vocoder,disc,diffusion=args[1:6]
    inputs={'x0':args[0]['mel'],'bnf':args[0]['bnf'],'spk_emb':args[0]['spk_emb']}
    driver.training.seed_everything(204562)
    inputs.update(source_noise=torch.randn_like(inputs['x0']),score_noise=torch.randn_like(inputs['x0']),
        score_timesteps=torch.randint(1,1001,(32,),dtype=torch.long))
    # 用一次预备读数为小fixture建立锚点；正式被测段从清空计数后开始。
    with torch.no_grad():
        y,q=driver.training.make_student_prediction(student,diffusion,inputs['x0'],inputs['spk_emb'],inputs['bnf'],950,inputs['source_noise'])
        real=vocoder(driver.training.denormalize_mel(inputs['x0'],args[8],args[9]))
        fake=vocoder(driver.training.denormalize_mel(y,args[8],args[9]))
        loss_d,adv,fm=driver.training.serial_discriminator_metrics(disc,real,fake)
        loss,target=driver.training.score_distillation_loss(y,teacher,diffusion,inputs['score_timesteps'],inputs['score_noise'],inputs['spk_emb'],inputs['bnf'],
            target_mode=args[-1].score_target,weight_mode=args[-1].score_weight)
    anchor={'input_identities':{name:driver.cross.tensor_id(value) for name,value in inputs.items()},
        'shared_row_tensor_identities':{'original':{name:driver.cross.tensor_id(value) for name,value in
            {'prediction':y,'q950':q,'real_waveform':real,'fake_waveform':fake,'teacher_target':target}.items()}},
        'matrix_3S_by_2D':{'original':{'D0':{'adv':float(adv),'fm_raw':float(fm),'discriminator_loss':float(loss_d),'primary_moving_score':float(loss)}}}}
    for model in (teacher,student,vocoder):model.trace.clear()
    return args,inputs,anchor


def actual_plan():
    return driver.planned_draws(driver.cross.read_json(driver.REPLICATION_CONTRACT))


class ZeroUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)  # 仅当前CPU测试进程，不改变原GPU执行设置。

    def test_actual_contract_contains_all120_vectors_in_predeclared_order(self):
        driver.cross.check_files(driver.PINNED)
        contract=driver.cross.read_json(driver.REPLICATION_CONTRACT);draws=driver.planned_draws(contract)
        self.assertEqual(len(draws),120)
        self.assertEqual([(r['seed_base'],r['arm'],r['absolute_seed_label']) for r in draws],
            [(seed,arm,step) for seed in (73100000,73200000,73300000) for arm in ('iid_control','stratified_candidate') for step in range(3328,3348)])
        self.assertEqual([r['readout_index'] for r in draws],list(range(120)))
        changed=copy.deepcopy(contract);changed['timestep_draws']['73200000']['iid_control'][4]['timesteps'][0]+=1
        with self.assertRaises(RuntimeError):driver.planned_draws(changed)

    def test_real_train3328_schema_and_only_original_input_assets(self):
        contract=driver.cross.read_json(driver.PARENT/'run_contract.json')
        dev,_=driver.resume.load_development(driver.DEV/'development_contract.json')
        config=driver.base.replace(driver.base.PAPER_STUDENT_CONFIG,num_workers=0)
        dataset=driver.base.VoiceGradDataset(driver.ROOT/'data','train',segment_length=128,expected_speakers=100,bnf_alignment=config.bnf_alignment)
        indices=driver.resume.GlobalBatchPlan(dev['retained_train_indices']).for_step(3328)
        anchor=driver.cross.read_json(driver.ANCHOR)
        self.assertEqual(indices,anchor['training_indices']);self.assertFalse(set(indices)&set(dev['excluded_train_indices']))
        before=driver.probe.tree_sha(driver.training.capture_rng_state())
        with driver.pair.preserve_rng(torch.device('cpu')):
            inputs=driver.prepare_inputs(dataset,indices,torch.device('cpu'))
        self.assertEqual(before,driver.probe.tree_sha(driver.training.capture_rng_state()))
        self.assertEqual(set(inputs),set(driver.INPUT_KEYS))
        self.assertEqual(inputs['x0'].shape,(32,80,128));self.assertEqual(inputs['bnf'].shape,(32,144,128));self.assertEqual(inputs['spk_emb'].shape,(32,256))
        for name in ('x0','bnf','spk_emb'):self.assertEqual(driver.cross.tensor_id(inputs[name]),anchor['input_identities'][name])
        # CPU噪声绝不冒称历史CUDA噪声；实际入口仍必须过全部六项GPU SHA门。
        with self.assertRaises(RuntimeError):driver.validate_inputs(inputs,anchor)

    def test_exact_full_1S2V1D122T_and_independent_weighted_reductions_no_updates(self):
        args,inputs,anchor=tiny_case();teacher,student,vocoder,disc,diffusion=args[1:6]
        models={'student':student,'teacher':teacher,'vocoder':vocoder,'D':disc,'diffusion':diffusion}
        before={name:driver.model_description(model) for name,model in models.items()}
        input_before=driver.validate_inputs(inputs,anchor)
        counts={key:0 for key in driver.EXPECTED_COUNTS};observed={'student':0,'teacher':0,'vocoder':0,'D':0};captured=[]
        handles=[]
        def hook(name):
            def run(module,unused,output):
                observed[name]+=1
                if name=='teacher':module.trace.clear()  # 仅清fixture测试追踪列表，避免120份小图输入积存。
            return run
        for name,model in (('student',student),('teacher',teacher),('vocoder',vocoder)):handles.append(model.register_forward_hook(hook(name)))
        original_d=driver.training.serial_discriminator_metrics;original_score=driver.training.score_distillation_loss
        def d_read(*a,**kw):observed['D']+=1;return original_d(*a,**kw)
        def score(*a,**kw):
            loss,target=original_score(*a,**kw)
            raw=(a[0]-target).abs().flatten(1).mean(1)
            weighted=raw*diffusion.sqrt_alphas_cumprod[a[3]]
            captured.append((raw.tolist(),weighted.tolist(),float(loss)))
            return loss,target
        emitted=[]
        # 实际运行段禁止构造Adam或保存权重，不只依赖AST检查。
        with mock.patch.object(torch.optim,'Adam',side_effect=AssertionError('no optimizer')), mock.patch.object(torch,'save',side_effect=AssertionError('no save')), mock.patch.object(driver.training,'serial_discriminator_metrics',side_effect=d_read), mock.patch.object(driver.training,'score_distillation_loss',side_effect=score):
            y,ids,gan=driver.fixed_gan(student,vocoder,disc,diffusion,inputs,args[8],args[9],anchor,counts,lambda:None)
            rows,old,repeat=driver.run_scores(y,teacher,diffusion,inputs,args[-1],gan,anchor,actual_plan(),counts,lambda:None,lambda kind,row:emitted.append((kind,row)))
        for handle in handles:handle.remove()
        self.assertEqual(counts,driver.EXPECTED_COUNTS);self.assertEqual(observed,{'student':1,'teacher':122,'vocoder':2,'D':1})
        self.assertEqual(len(rows),120);self.assertEqual([kind for kind,row in emitted],['anchor']+['draw']*120+['repeat'])
        for row,(raw,weighted,loss) in zip([old]+rows+[repeat],captured):
            self.assertEqual(row['per_utterance_unweighted_l1'],raw);self.assertEqual(row['per_utterance_weighted_l1'],weighted)
            self.assertEqual(row['dist_raw'],loss);self.assertEqual(row['student_total'],gan['adv']+2*gan['fm_raw']+45*loss)
        self.assertEqual(repeat,{key:value for key,value in rows[0].items() if key not in ('seed_base','arm','absolute_seed_label','timestep_seed','readout_index','model_global_step','optimizer_updates')})
        driver.assert_unchanged(models,before,inputs,input_before)
        self.assertEqual(gan['per_utterance_GT_l1'],(y-inputs['x0']).abs().flatten(1).mean(1).tolist())
        summary=driver.summarize(rows);self.assertEqual(len(summary['groups']),6)
        for group,offset in zip(summary['groups'],range(0,120,20)):
            values=[row['dist_raw'] for row in rows[offset:offset+20]]
            self.assertEqual(group['statistics']['dist_raw']['mean'],math.fsum(values)/20)
            self.assertEqual(group['statistics']['dist_raw']['population_std'],statistics.pstdev(values))

    def test_fixed_gan_rejects_changed_waveform_identity_before_D_readout(self):
        args,inputs,anchor=tiny_case();anchor['shared_row_tensor_identities']['original']['fake_waveform']['sha256']='bad'
        counts={key:0 for key in driver.EXPECTED_COUNTS}
        with self.assertRaisesRegex(RuntimeError,'waveform SHA'):
            driver.fixed_gan(args[2],args[3],args[4],args[5],inputs,args[8],args[9],anchor,counts,lambda:None)
        self.assertEqual(counts,{'student':1,'teacher':0,'vocoder':2,'serial_D_readouts':0})

    def test_score_rejects_nonfinite_and_bad_time_without_extra_calls(self):
        args,inputs,anchor=tiny_case();counts={key:0 for key in driver.EXPECTED_COUNTS}
        gan={'adv':2.,'fm_raw':.8};y=inputs['x0'];t=inputs['score_timesteps']
        for bad in (torch.zeros(32,dtype=torch.long),torch.ones(31,dtype=torch.long),torch.ones(32)):
            with self.assertRaises(RuntimeError):driver.score_record(y,args[1],args[5],inputs,bad,args[-1],gan,counts,lambda:None)
        self.assertEqual(counts['teacher'],0)
        with mock.patch.object(driver.training,'score_distillation_loss',return_value=(torch.tensor(float('nan')),y)):
            with self.assertRaisesRegex(RuntimeError,'Nonfinite'):driver.score_record(y,args[1],args[5],inputs,t,args[-1],gan,counts,lambda:None)
        self.assertEqual(counts['teacher'],1)

    def test_repeat_mismatch_and_execution_error_do_not_continue(self):
        with self.assertRaises(RuntimeError):driver.require_exact_repeat({'target':'a'},{'target':'b'})
        args,inputs,anchor=tiny_case();counts={key:0 for key in driver.EXPECTED_COUNTS}
        values=[]
        def fake(*a,**kw):
            if len(values)==5:raise RuntimeError('teacher failure')
            draw=actual_plan()[0];t=a[4].tolist()
            row={'teacher_target_identity':anchor['shared_row_tensor_identities']['original']['teacher_target'],
                'dist_raw':anchor['matrix_3S_by_2D']['original']['D0']['primary_moving_score'],
                'timesteps':t,'bin_counts':[((i-1)//100) for i in t]}
            row['bin_counts']=[row['bin_counts'].count(i) for i in range(10)];values.append(row);return row
        with mock.patch.object(driver,'score_record',side_effect=fake):
            with self.assertRaisesRegex(RuntimeError,'teacher failure'):
                driver.run_scores(inputs['x0'],args[1],args[5],inputs,args[-1],{},anchor,actual_plan(),counts,lambda:None,lambda *a:None)
        self.assertEqual(len(values),5)

    def test_model_change_detected_before_restore_and_real_exception_restores_all_state(self):
        model=nn.Sequential(nn.Linear(2,2),nn.ReLU());model[1].eval()
        model.register_buffer('window',torch.tensor([2.]),persistent=False)
        model[0].bias.grad=torch.ones_like(model[0].bias)
        inputs={'x0':torch.ones(2)};input_before={k:driver.cross.tensor_id(v) for k,v in inputs.items()}
        before={'model':driver.model_description(model)};models={};receipt={}
        rng=driver.probe.tree_sha(driver.training.capture_rng_state());flags=driver.cross.flags()
        with self.assertRaisesRegex(RuntimeError,'Read-only forward changed'):
            with driver.cross.preserved_state(models,receipt) as register:
                register('model',model)
                model.window.add_(1);model.eval();model[0].weight.requires_grad_(False);torch.randn(2)
                driver.assert_unchanged(models,before,inputs,input_before)
        self.assertEqual(driver.model_description(model),before['model'])
        self.assertEqual(driver.probe.tree_sha(driver.training.capture_rng_state()),rng);self.assertEqual(driver.cross.flags(),flags)
        self.assertTrue(receipt['rng_restored']);self.assertTrue(receipt['models']['model'])
        changed={k:v.clone() for k,v in inputs.items()};changed['x0'].add_(1)
        with self.assertRaisesRegex(RuntimeError,'input tensor'):driver.assert_unchanged(models,before,changed,input_before)

    def test_prediction_stats_and_precision_changes_are_rejected_before_restoration(self):
        prediction=torch.ones(2);identity=driver.cross.tensor_id(prediction)
        inputs={'mel_mean':torch.ones(2),'mel_std':torch.ones(2)}
        before={name:driver.cross.tensor_id(value) for name,value in inputs.items()}
        prediction.add_(1)
        with self.assertRaisesRegex(RuntimeError,'cached student'):driver.assert_unchanged({}, {},inputs,before,prediction,identity)
        inputs['mel_std'].add_(1)
        with self.assertRaisesRegex(RuntimeError,'input tensor'):driver.assert_unchanged({}, {},inputs,before)
        flags=driver.cross.flags()
        try:
            torch.backends.cudnn.allow_tf32=not flags['cudnn_tf32']
            with self.assertRaisesRegex(RuntimeError,'precision'):driver.require_original_precision()
            with self.assertRaisesRegex(RuntimeError,'precision'):driver.assert_unchanged({}, {},{}, {})
        finally:driver.cross.restore_flags(flags)

    def test_teacher_readout_cannot_mutate_cached_prediction_and_continue(self):
        args,inputs,anchor=tiny_case();y=inputs['x0'].clone();count={key:0 for key in driver.EXPECTED_COUNTS}
        def mutate(*a,**kw):a[0].add_(1);return {}
        with mock.patch.object(driver,'score_record',side_effect=mutate) as changed:
            with self.assertRaisesRegex(RuntimeError,'cached student'):
                driver.run_scores(y,args[1],args[5],inputs,args[-1],{},anchor,actual_plan(),count,lambda:None,lambda *a:None)
        self.assertEqual(changed.call_count,1)

    def test_summary_refuses_missing_or_reordered_measurements(self):
        rows=[dict(seed_base=x['seed_base'],arm=x['arm'],absolute_seed_label=x['absolute_seed_label'],dist_raw=.05,dist_weighted=2.25,student_total=6.) for x in actual_plan()]
        result=driver.summarize(rows)
        self.assertFalse(result['three_windows_in_predeclared_seed_order']['iid_control']['dist_raw']['strictly_increasing'])
        with self.assertRaises(RuntimeError):driver.summarize(rows[:-1])
        rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(RuntimeError):driver.summarize(rows)

    def test_import_and_ast_have_no_optimizer_backward_model_save_or_global_patch(self):
        tree=ast.parse(Path(driver.__file__).read_text());bad=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                if name in ('Adam','AdamW','SGD','backward','grad','save','atomic_torch_save','train_batch','train_batch_with_timesteps'):bad.append(name)
                if isinstance(node.func,ast.Name) and name in ('eval','exec'):bad.append(name)
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id in ('training','torch','replication','cross'):bad.append(ast.dump(target))
        self.assertEqual(bad,[])

    def test_original_600_30_watchdog_and_budget_output_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/'trial';budget=driver.replication.MetadataBudget(output)
            with mock.patch.object(driver.pair,'available_bytes',return_value=100*1024**3):
                log=driver.replication.open_exclusive_log(output,budget);log.write('test\n');log.close()
                with self.assertRaises(RuntimeError):driver.replication.open_exclusive_log(output,budget)
                output.mkdir()
                with self.assertRaises(RuntimeError):budget.write(Path(name)/'escape.json',{})
        code="""import os,signal,time
import inspect_acdd_zero_update_sampling_3327_v1 as d
s=d.cross.DeadlineSignals(time.monotonic(),600);s.install()
try:
    try:signal.raise_signal(signal.SIGTERM)
    except d.cross.DiagnosticInterruption:pass
    else:raise AssertionError('signal ignored')
    s.cleanup_only();assert s.reason is not None
    print('READONLY_DEADLINE_OK')
finally:s.close()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parent,env=dict(os.environ),
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('READONLY_DEADLINE_OK',result.stdout)

    def test_actual_gnu_timeout_parent_passes_and_direct_python_parent_is_rejected(self):
        # 真实CPU子进程读取/proc父进程，不能用mock伪造外层600+30秒保护。
        code="import inspect_acdd_zero_update_sampling_3327_v1 as d; r=d.watchdog(); assert r['maximum_compute_seconds']==600 and r['cleanup_only_seconds']==30; print('WATCHDOG_REAL_OK')"
        success=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=30s','600s',sys.executable,'-c',code],
            cwd=Path(__file__).resolve().parent,env=dict(os.environ),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=25)
        self.assertEqual(success.returncode,0,success.stderr);self.assertIn('WATCHDOG_REAL_OK',success.stdout)
        reject="""import inspect_acdd_zero_update_sampling_3327_v1 as d
try:d.watchdog()
except RuntimeError as error:
    assert 'Require external timeout' in str(error)
    print('DIRECT_PARENT_REJECTED')
else:raise AssertionError('direct Python parent must be refused')
"""
        failure=subprocess.run([sys.executable,'-c',reject],cwd=Path(__file__).resolve().parent,env=dict(os.environ),
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=25)
        self.assertEqual(failure.returncode,0,failure.stderr);self.assertIn('DIRECT_PARENT_REJECTED',failure.stdout)


if __name__=='__main__':
    unittest.main()
