"""租用机CPU测试：只用小计算图和只读真实收据，不启动实际GPU训练。"""

import ast  # 静态证明新训练函数除唯一hook外与原函数完全一致。
import copy
import inspect
import json
import os
import random
import signal
import subprocess
import sys
import tempfile  # 所有可写测试只在自己创建的临时目录内进行。
import time
import unittest
from pathlib import Path
from unittest import mock

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Run CPU tests only on the rented Matpool with CUDA_VISIBLE_DEVICES empty')

import torch
from torch import nn
import acdd_uniform_stratified_train_batch_v1 as helper
import run_acdd_timestep_paired60_v1 as driver
import test_conversion_score_step as fixture  # 冻结小条件网络，仅用于独立CPU计算对照。


def fixture_state(args):
    return {'models':fixture.model_optimizer_state(args),'rng':driver.training.capture_rng_state()}


def loss_row(step,value):
    return {'global_step':step,'updates':step-driver.START,**{key:float(value) for key in driver.pair.LOSS_KEYS}}


def entry(step,gt=.19,score=.045,train_gt=.19):
    return {'global_step':step,'development':{'mel_l1':gt,'prediction_rms':1.},
            'uniform64':{'uniform_weighted_score':score,'GT_l1':train_gt}}


class Paired60Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_static_clone_has_exactly_one_hook_and_only_two_added_keyword_arguments(self):
        original=ast.parse(inspect.getsource(helper.production.train_batch)).body[0]
        clone=ast.parse(inspect.getsource(helper.train_batch_with_timesteps)).body[0]
        self.assertEqual([node.arg for node in clone.args.kwonlyargs],['score_timesteps_override','sampler_receipt'])
        self.assertTrue(all(isinstance(node,ast.Constant) and node.value is None for node in clone.args.kw_defaults))
        hook=[i for i,node in enumerate(clone.body) if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
              and isinstance(node.value.func,ast.Name) and node.value.func.id=='inject_timesteps']
        self.assertEqual(len(hook),1)
        previous=clone.body[hook[0]-1]
        self.assertIsInstance(previous,ast.Assign)
        self.assertEqual(previous.value.func.attr,'randint')  # 必须在真实第三个原draw之后。
        clone.body.pop(hook[0]);clone.name=original.name;clone.args.kwonlyargs=[];clone.args.kw_defaults=[]
        self.assertEqual(ast.dump(clone,include_attributes=False),ast.dump(original,include_attributes=False))
        source=ast.parse(Path(helper.__file__).read_text())
        forbidden=[n for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('eval','exec')]
        self.assertEqual(forbidden,[])  # 静态源码，运行时不生成/替换生产函数。

    def test_production_none_and_explicit_original_t_match_models_two_adam_rng(self):
        for override_mode in ('none','same_t'):
            original=fixture.make_fixture(torch.float32);changed=fixture.make_fixture(torch.float32)
            for args in (original,changed):args[6].param_groups[0]['lr']=5e-5
            for step in (3328,3329):
                seed=201234+step
                driver.training.seed_everything(seed)
                torch.randn_like(changed[0]['mel']);torch.randn_like(changed[0]['mel'])
                t=torch.randint(1,1001,(32,),dtype=torch.long)
                driver.training.seed_everything(seed)
                expected=helper.production.train_batch(*original)
                expected_state=fixture_state(original)
                driver.training.seed_everything(seed);receipt={}
                actual=helper.train_batch_with_timesteps(*changed,score_timesteps_override=t if override_mode=='same_t' else None,sampler_receipt=receipt)
                self.assertEqual(actual,expected)
                self.assertEqual(driver.probe.tree_sha(fixture_state(changed)),driver.probe.tree_sha(expected_state))
                self.assertEqual(receipt['original_uniform_values'],t.tolist())
                self.assertEqual(receipt['used_values'],t.tolist())

    def test_stratification_counts_ranges_seed_and_no_global_rng_consumption(self):
        driver.training.seed_everything(9123);before=driver.probe.tree_sha(driver.training.capture_rng_state())
        for step in range(3328,3388):
            values,meta=helper.stratified_timesteps(step)
            self.assertEqual(values.dtype,torch.long);self.assertEqual(values.shape,(32,))
            self.assertEqual(sorted(meta['bin_counts']),[3]*8+[4]*2)
            self.assertEqual(len(set(meta['bonus_bins'])),2)
            self.assertEqual(meta['seed'],73083328+step)
            self.assertEqual(((values-1)//100).tolist(),meta['labels'])
            local=random.Random(73083328+step)
            labels=[i for i in range(10) for _ in range(3)]+local.sample(range(10),2)
            local.shuffle(labels)
            self.assertEqual(values.tolist(),[local.randint(100*i+1,100*i+100) for i in labels])
        self.assertEqual(driver.probe.tree_sha(driver.training.capture_rng_state()),before)
        self.assertEqual((3+2/10)/32/100,.001)  # 每句的条件边际仍是1/1000；不推断Adam期望。

    def test_candidate_only_replaces_t_keeps_original_three_draws_and_gan_inputs(self):
        a,b=fixture.make_fixture(torch.float32),fixture.make_fixture(torch.float32)
        receipts=[];losses=[]
        for args,override in ((a,None),(b,helper.stratified_timesteps(3328)[0])):
            args[6].param_groups[0]['lr']=5e-5
            driver.training.seed_everything(204562);receipt={}
            losses.append(helper.train_batch_with_timesteps(*args,score_timesteps_override=override,sampler_receipt=receipt));receipts.append(receipt)
        for key in ('source_noise','score_noise','original_uniform_t','original_uniform_values'):
            self.assertEqual(receipts[0][key],receipts[1][key])
        self.assertNotEqual(receipts[0]['used_values'],receipts[1]['used_values'])
        for key in ('student_adv','fm_raw','fm_weighted','discriminator'):self.assertEqual(losses[0][key],losses[1][key])
        self.assertFalse(torch.equal(a[1].trace[-1]['t'],b[1].trace[-1]['t']))
        self.assertEqual(driver.probe.tree_sha(a[4].state_dict()),driver.probe.tree_sha(b[4].state_dict()))
        for args in (a,b):
            self.assertTrue(all(p.grad is None for model in args[1:5] for p in model.parameters()))

    def test_bad_time_override_rejected_before_any_D_or_S_update(self):
        for bad in (torch.ones(31,dtype=torch.long),torch.ones(32),torch.zeros(32,dtype=torch.long),torch.full((32,),1001,dtype=torch.long)):
            args=fixture.make_fixture(torch.float32)
            before=driver.probe.tree_sha(fixture.model_optimizer_state(args))
            with self.assertRaises(RuntimeError):helper.train_batch_with_timesteps(*args,score_timesteps_override=bad)
            self.assertEqual(driver.probe.tree_sha(fixture.model_optimizer_state(args)),before)

    def test_monitor_restores_modes_requires_all_named_buffers_and_rng_on_error(self):
        model=nn.Sequential(nn.Linear(2,2),nn.ReLU());model[1].eval()
        model.register_buffer('nonpersistent',torch.tensor(3.),persistent=False)
        before_modes=[m.training for m in model.modules()];before_rng=driver.probe.tree_sha(driver.training.capture_rng_state())
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with driver.monitoring_state({'tiny':model},torch.device('cpu')):
                model.eval();model[0].weight.requires_grad_(False);torch.randn(2)
                raise RuntimeError('injected')
        self.assertEqual(before_modes,[m.training for m in model.modules()]);self.assertTrue(model[0].weight.requires_grad)
        self.assertEqual(before_rng,driver.probe.tree_sha(driver.training.capture_rng_state()))
        with self.assertRaisesRegex(RuntimeError,'changed'):
            with driver.monitoring_state({'tiny':model},torch.device('cpu')):model.nonpersistent.add_(1)

    def test_fresh_windows_and_original_plus_both_parent_hard_guards(self):
        anchor={'mel_l1':.19941353797912598,'prediction_rms':.9994737505912781}
        old={'mel_l1':.1948028951883316};parent={'mel_l1':.19254307448863983}
        windows=[driver.pair.window_record([loss_row(3328+20*i+j,x) for j in range(20)]) for i,x in enumerate((5.9,6.,6.1))]
        monitors=[entry(3327),entry(3347,.189),entry(3367,.188),entry(3387,.187)]
        self.assertNotIn('moving_total_three_windows_rising',driver.stop_rules(windows[:2],monitors[:3],anchor,old,parent)['stop_reasons'])
        self.assertIn('moving_total_three_windows_rising',driver.stop_rules(windows,monitors,anchor,old,parent)['stop_reasons'])
        self.assertIn('development_three_checks_rising',driver.stop_rules([], [entry(3327,.19),entry(3347,.1901),entry(3367,.1902)],anchor,old,parent)['stop_reasons'])
        self.assertIn('development_above3327_by1percent',driver.stop_rules([], [entry(3347,parent['mel_l1']*1.010001)],anchor,old,parent)['stop_reasons'])
        self.assertIn('development_above3227_by1percent',driver.stop_rules([], [entry(3347,.20)],anchor,old,parent)['stop_reasons'])
        self.assertIn('original_S0_guard',driver.stop_rules([], [entry(3347,.23)],anchor,old,parent)['stop_reasons'])

    def test_candidate20_gate_all_four_metrics_and_every_execution_guard(self):
        parent={'mel_l1':.1925};control=entry(3347,.192,.047,.192)
        passing=entry(3347,.191,.046,.191)
        self.assertTrue(driver.candidate_gate(passing,control,parent,[])['passed'])
        for path in (('development','mel_l1'),('uniform64','uniform_weighted_score'),('uniform64','GT_l1')):
            bad=copy.deepcopy(passing);bad[path[0]][path[1]]=1.
            self.assertFalse(driver.candidate_gate(bad,control,parent,[])['passed'])
        for reason in ('requested_stop','time_budget','disk','moving_total_three_windows_rising'):
            self.assertFalse(driver.candidate_gate(passing,control,parent,[reason])['passed'])
        with self.assertRaises(RuntimeError):driver.candidate_gate(passing,None,parent,[])

    def test_complete_batch_half_update_error_never_advances_or_saves(self):
        state={'arm':'primary_stratified','completed_global_step':3327,'partial_batch_may_have_updated_D':False}
        with self.assertRaisesRegex(RuntimeError,'half'):
            driver.complete_batch(state,3328,lambda:(_ for _ in ()).throw(RuntimeError('half')))
        self.assertEqual(state['completed_global_step'],3327);self.assertTrue(state['partial_batch_may_have_updated_D'])
        with self.assertRaisesRegex(RuntimeError,'half'):
            driver.save_primary(Path('/irrelevant'),state,{},None,None,None,None,None,None)

    def test_every_actual_adam_counter_is_checked_without_resetting_moments(self):
        args=fixture.make_fixture(torch.float32);args[6].param_groups[0]['lr']=5e-5
        helper.production.train_batch(*args)
        before=driver.probe.tree_sha([args[6].state_dict(),args[7].state_dict()])
        result=driver.adam_counters(args[6],args[7],1)
        self.assertEqual(result['student']['parameter_count'],4)
        self.assertEqual(result['discriminator']['parameter_count'],4)
        self.assertEqual(driver.probe.tree_sha([args[6].state_dict(),args[7].state_dict()]),before)
        with self.assertRaises(RuntimeError):driver.adam_counters(args[6],args[7],2)
        next(iter(args[7].state.values()))['step'].zero_()
        with self.assertRaises(RuntimeError):driver.adam_counters(args[6],args[7],1)

    def test_future_feature_binding_reads_actual_files_and_rejects_conflicting_sha(self):
        with tempfile.TemporaryDirectory() as name:
            paths=[]
            for i in range(3):
                path=Path(name)/str(i);path.write_bytes(('test feature%d'%i).encode());paths.append(path)
            dataset=mock.Mock(items=[dict(zip(('mel_path','bnf_path','spk_emb_path'),paths))])
            files={};registered=driver.register_future_files(dataset,[{'indices':[0,0]}],files,lambda:None)
            self.assertEqual(len(registered),3);self.assertEqual(files,registered)
            self.assertEqual(files[str(paths[0])],driver.base.sha256_file(paths[0]))
            with self.assertRaisesRegex(RuntimeError,'Conflicting'):
                driver.register_future_files(dataset,[{'indices':[0]}],{str(paths[0]):'wrong'},lambda:None)

    def test_real_metadata_and64_input_schema_no_model_forward(self):
        contracts={step:driver.cross.read_json(item['contract']) for step,item in driver.cross.STATE.items()}
        self.assertEqual(driver.training.canonical_json_sha256(contracts[3327]),driver.PARENT_CANONICAL)
        receipt=driver.cross.read_json(driver.PARENT/'completion.json')
        self.assertEqual(receipt['completed_global_step'],3327);self.assertEqual(receipt['status'],'stopped')
        self.assertEqual(receipt['full_snapshot']['sha256'],driver.PARENT_SHA)
        development,files=driver.resume.load_development(driver.DEV/'development_contract.json')
        old,batches,target,plan=driver.cross.prepare_inputs(files,development,contracts,torch.device('cpu'))
        reference=driver.cross.read_json(driver.CROSS_IDENTITY)['plan']
        for key in ('selected_indices','strata_timesteps','crop_seeds','noise_seeds','utterances'):
            self.assertEqual(plan[key],reference[key])
        self.assertEqual(len(plan['selected_indices']),64)
        self.assertFalse(set(plan['selected_indices'])&set(development['excluded_train_indices']))
        for batch in batches:
            self.assertEqual(batch['x0'].shape,(32,80,128));self.assertEqual(batch['bnf'].shape,(32,144,128));self.assertEqual(batch['spk_emb'].shape,(32,256))
        self.assertEqual(target.shape,(32,80,128))
        # CPU生成的新noise不声称等于历史CUDA；真实入口还会比较已发布的GPU tensorSHA。
        self.assertEqual(driver.probe.tensor_sha(old['score_timesteps']),reference['input_identities'][0]['score_timesteps']['sha256'])

    def test_real_fixed64_replay_schema_keeps_all_per_sentence_fields(self):
        r=driver.cross.read_json(driver.CROSS_RESULT);read=r['readouts'];gt=read['new_TRAIN64_GT']['3327']
        actual={'GT_l1':gt['l1'],'uniform_weighted_score':read['ten_stratum_uniform_means']['y3327_R3327']['weighted'],
            'prediction_identities':read['student_prediction_identities']['3327'][1:],
            'teacher_target_identities':[read['conditions']['stratum_'+str(i)]['target_identities']['3327'] for i in range(10)],
            'per_stratum_per_utterance_weighted_score':[read['conditions']['stratum_'+str(i)]['cells']['y3327_R3327']['per_weighted'] for i in range(10)]}
        driver.validate_uniform_initial(actual,r)
        broken=copy.deepcopy(actual);broken['per_stratum_per_utterance_weighted_score'][0].pop()
        with self.assertRaises(RuntimeError):driver.validate_uniform_initial(broken,r)

    def test_actual_uniform64_monitor_two_students_twenty_teachers_and_independent_reductions(self):
        args=fixture.make_fixture(torch.float32)
        teacher,student,diffusion,config=args[1],args[2],args[5],args[-1]
        batches=[]
        for index in range(2):
            batch=args[0]
            x=batch['mel'].clone()+.05*index
            source,score=torch.randn_like(x),torch.randn_like(x)
            t=torch.full((32,),950,dtype=torch.long)
            batches.append({'x0':x,'bnf':batch['bnf'].clone(),'spk_emb':batch['spk_emb'].clone(),
                'source_noise':source,'score_noise':score,'q950':diffusion.q_sample(x,t,noise=source)})
        local=random.Random(73073341)
        plan={'strata_timesteps':[[local.randint(100*i+1,100*i+100) for _ in range(64)] for i in range(10)]}
        inputs_before=driver.probe.tree_sha(batches)
        models={'student':student,'teacher':teacher,'diffusion':diffusion}
        model_before={key:driver.pair.model_sha(value) for key,value in models.items()}
        rng_before=driver.probe.tree_sha(driver.training.capture_rng_state())
        captured=[];original=driver.training.score_distillation_loss
        def record(*positional,**keywords):
            loss,target=original(*positional,**keywords)
            captured.append((positional[0].detach().clone(),target.detach().clone(),positional[3].clone()))
            return loss,target  # 记录同一次真实teacher输出，不再额外前向制造参考。
        with mock.patch.object(driver.training,'score_distillation_loss',side_effect=record):
            with driver.monitoring_state(models,torch.device('cpu')):
                actual=driver.uniform64_monitor(student,teacher,diffusion,batches,plan,config,torch.device('cpu'),lambda:None)
        self.assertEqual(len(student.trace),2);self.assertEqual(len(teacher.trace),20)
        self.assertEqual(actual['actual_forward_counts'],{'student':2,'teacher':20})
        values=[]
        for layer in range(10):
            pieces=[]
            for block in range(2):
                y,target,t=captured[10*block+layer]
                independent=(y-target).abs().reshape(32,-1).mean(1)*diffusion.sqrt_alphas_cumprod[t]
                pieces.extend(independent.tolist())
            self.assertEqual(len(pieces),64)
            self.assertEqual(actual['per_stratum_per_utterance_weighted_score'][layer],pieces)
            self.assertAlmostEqual(actual['per_stratum_weighted_score'][layer],sum(pieces)/64,places=14)
            values.extend(pieces)
        self.assertAlmostEqual(actual['uniform_weighted_score'],sum(values)/640,places=13)
        y=torch.cat([captured[0][0],captured[10][0]]);x=torch.cat([row['x0'] for row in batches])
        self.assertEqual(actual['GT_l1'],float((y-x).abs().mean()))
        self.assertEqual(actual['prediction_rms'],float(y.square().mean().sqrt()))
        self.assertEqual(actual['per_utterance_GT_l1'],(y-x).abs().reshape(64,-1).mean(1).tolist())
        self.assertEqual(driver.probe.tree_sha(batches),inputs_before)
        self.assertEqual({key:driver.pair.model_sha(value) for key,value in models.items()},model_before)
        self.assertEqual(driver.probe.tree_sha(driver.training.capture_rng_state()),rng_before)

    def test_real_parent_two_adam_roles_counters_shapes_and_rng_without_update(self):
        payload=driver.resume.load_torch_checkpoint(driver.PARENT/'final_full.pt')
        self.assertEqual(payload['global_step'],3327);self.assertEqual(payload['effective_optimizer_lrs'],driver.LRS)
        self.assertEqual(payload['run_contract_sha256'],driver.PARENT_CANONICAL)
        for role,lr in (('student',5e-5),('discriminator',2e-4)):
            state=payload[role+'_optimizer'];self.assertEqual(state['param_groups'][0]['lr'],lr)
            self.assertEqual(state['param_groups'][0]['betas'],(.5,.9))
            self.assertEqual(len(state['param_groups']),1)
            self.assertEqual(set(state['state']),set(state['param_groups'][0]['params']))
            for item in state['state'].values():
                self.assertEqual(float(item['step']),3327.)
                self.assertEqual(item['exp_avg'].shape,item['exp_avg_sq'].shape)
                self.assertTrue(bool(torch.isfinite(item['exp_avg']).all() and torch.isfinite(item['exp_avg_sq']).all()))
        self.assertEqual(set(payload['rng_state']),{'python','numpy','torch','cuda_all'})

    def test_restore_real_logic_with_two_toy_adams_and_original_rng_bytes(self):
        args=fixture.make_fixture(torch.float32);student,disc=args[2],args[4];opt,dopt=args[6:8]
        opt.param_groups[0]['lr']=5e-5
        driver.training.seed_everything(204562);helper.production.train_batch(*args)
        for optimizer in (opt,dopt):
            for value in optimizer.state.values():value['step'].fill_(3327)
        payload={'student':copy.deepcopy(student.state_dict()),'discriminator':copy.deepcopy(disc.state_dict()),
            'student_optimizer':copy.deepcopy(opt.state_dict()),'discriminator_optimizer':copy.deepcopy(dopt.state_dict()),
            'rng_state':driver.training.capture_rng_state()}
        before=driver.probe.tree_sha(payload)
        # only父身份文案在小fixture中替代；真正模型/双Adam/RNG恢复函数均执行。
        with mock.patch.object(driver.dstate,'validate_parent'):
            restored=driver.restore_parent(payload,student,disc,opt,dopt,args[-1],{}, {},torch.device('cpu'))
            driver.training.seed_everything(9);helper.production.train_batch(*args)
            again=driver.restore_parent(payload,student,disc,opt,dopt,args[-1],{}, {},torch.device('cpu'))
        self.assertEqual(restored,again);self.assertEqual(driver.probe.tree_sha(payload),before)
        self.assertEqual(opt.param_groups[0]['lr'],5e-5);self.assertEqual(dopt.param_groups[0]['lr'],2e-4)

    def test_budget_keeps_system_slack_and_all_metadata_and_refuses_low_disk(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/'trial';output.mkdir()
            budget=driver.Budget(output,100,200)
            need=driver.RESERVE+driver.DIR_RESERVE+driver.META_LIMIT+2*driver.MODEL_SLACK+300
            with mock.patch.object(driver.pair,'available_bytes',return_value=need):self.assertEqual(budget.require()['required_bytes'],need)
            with mock.patch.object(driver.pair,'available_bytes',return_value=need-1):
                with self.assertRaises(driver.pair.InsufficientDisk):budget.require()
            with mock.patch.object(budget,'metadata_used',return_value=driver.META_LIMIT+1):
                with self.assertRaises(driver.pair.InsufficientDisk):budget.require()

    def test_real_full_S_save_and_faults_preserve_truthful_committed_cursor(self):
        args=fixture.make_fixture(torch.float32);student,disc,opt,dopt=args[2],args[4],args[6],args[7]
        for fault in ('none','full','S_OSError','S_RuntimeError','metadata'):
            with tempfile.TemporaryDirectory() as name:
                output=Path(name);budget=driver.Budget(output,1024**2,1024**2)
                state={'arm':'primary_stratified','completed_global_step':3347,'last_full_saved_global_step':3327,
                    'partial_batch_may_have_updated_D':False,'full_snapshot':None,'student_snapshot':None}
                common={'global_step':3347,'effective_optimizer_lrs':driver.LRS}
                real=driver.training.atomic_torch_save
                def save(payload,path):
                    if fault=='full' and path.name=='final_full.pt':raise OSError('full fault')
                    if fault.startswith('S_') and path.name.startswith('student_'):
                        raise (OSError('S fault') if fault=='S_OSError' else RuntimeError('S fault'))
                    real(payload,path)
                with mock.patch.object(driver.pair,'available_bytes',return_value=100*1024**3),mock.patch.object(driver.training,'atomic_torch_save',side_effect=save):
                    if fault=='metadata':budget.write=mock.Mock(side_effect=OSError('metadata fault'))
                    if fault=='none':driver.save_primary(output,state,common,student,disc,opt,dopt,budget,driver.StopClock(1200))
                    else:
                        with self.assertRaises((RuntimeError,OSError)):driver.save_primary(output,state,common,student,disc,opt,dopt,budget,driver.StopClock(1200))
                self.assertEqual(state['last_full_saved_global_step'],3327 if fault=='full' else 3347)
                if fault!='full':
                    saved=torch.load(output/'final_full.pt',map_location='cpu');self.assertEqual(saved['global_step'],3347)
                    self.assertIn('student_optimizer',saved);self.assertIn('discriminator_optimizer',saved);self.assertIn('rng_state',saved)
                if fault=='none':
                    s=torch.load(output/'student_step_003347.pt',map_location='cpu')
                    self.assertEqual(driver.probe.tree_sha(saved['student']),driver.probe.tree_sha(s['student']))

    def test_control_never_saves_and_failure_receipt_does_not_claim_full(self):
        with self.assertRaisesRegex(RuntimeError,'Control'):
            driver.save_primary(Path('/irrelevant'),{'arm':'control_iid'},None,None,None,None,None,None,None)
        with tempfile.TemporaryDirectory() as name:
            budget=driver.Budget(Path(name),100,100)
            state={'completed_global_step':3347,'last_full_saved_global_step':3327,'partial_batch_may_have_updated_D':False}
            with mock.patch.object(budget,'write',side_effect=OSError('disk full')),mock.patch('builtins.print') as report:
                driver.failure_record(Path(name),state,RuntimeError('metadata failed'),budget)
            self.assertIn('3327',str(report.call_args_list));self.assertNotIn('original_failure_preserved',str(report.call_args_list))

    def test_control_execution_stop_cannot_launch_candidate_even_if_space_recovers(self):
        state={'status':'stopped','reason':'moving_total_three_windows_rising','new_updates':60,'partial_batch_may_have_updated_D':False}
        self.assertTrue(driver.permits_candidate(state))
        for reason in ('available=1 required=2 pending={}','disk_space_guard','time_budget','requested_stop','OOM','half_update'):
            self.assertFalse(driver.permits_candidate(dict(state,reason=reason)))
        self.assertFalse(driver.permits_candidate(dict(state,new_updates=19)))
        self.assertFalse(driver.permits_candidate(dict(state,partial_batch_may_have_updated_D=True)))
        self.assertFalse(driver.permits_candidate(dict(state,execution_stop_reasons=['disk_space_guard'])))  # 趋势和空间同时停止也不能漏掉后者。

    def test_stop_during_final_save_cannot_claim_normal_completion(self):
        clock=driver.StopClock(1200);state={'reason':'maximum60_updates_completed'}
        self.assertEqual(driver.terminal_status(state,clock),'completed')
        clock.request('time_budget')
        self.assertEqual(driver.terminal_status(state,clock),'stopped')
        self.assertEqual(state['reason'],'time_budget')
        self.assertEqual(state['execution_stop_during_finalization'],'time_budget')

    def test_historical_preupdate_and_common_input_checks_fail_closed(self):
        anchor=driver.cross.read_json(driver.cross.D_ANCHOR)
        inputs=anchor['input_identities']
        row={'global_step':3328,'indices':anchor['training_indices'],'batch_tensors':{a:inputs[b] for a,b in (('mel','x0'),('bnf','bnf'),('spk_emb','spk_emb'))},
            'sampler':{'source_noise':inputs['source_noise'],'score_noise':inputs['score_noise'],'original_uniform_t':inputs['score_timesteps'],
                       'original_uniform_values':anchor['score_timesteps']},'postupdate_state_sha256':{},
            'discriminator':3.91922926902771,'student_adv':2.151906728744507,'fm_raw':.8101478815078735,
            'fm_weighted':2*.8101478815078735,'dist_raw':.054745227098464966,'student_total':6.2357378005981445}
        driver.historical_first_check(row,anchor);driver.common_input_check(row,copy.deepcopy(row))
        bad=copy.deepcopy(row);bad['student_adv']+=.01
        with self.assertRaises(RuntimeError):driver.historical_first_check(bad,anchor)
        with self.assertRaises(RuntimeError):driver.common_input_check(bad,row)
        bad=copy.deepcopy(row);bad['dist_raw']+=.01;bad['student_total']+=.45
        driver.common_input_check(bad,row)  # 候选t变了，不能错误要求score/total仍相同。

    def test_signal_boundary_stop_and_no_monitor_after_deadline(self):
        code="""import signal,time
import run_acdd_timestep_paired60_v1 as d
c=d.StopClock(1200);c.install()
try:
    signal.raise_signal(signal.SIGTERM)
    assert c.reason=='requested_stop'
    c.saving()
    try:c.active()
    except d.BoundaryStop:print('BOUNDARY_STOP_OK')
    else:raise AssertionError('missed stop')
finally:c.close()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parent,env=dict(os.environ),
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('BOUNDARY_STOP_OK',result.stdout)
        clock=driver.StopClock(.001,started=time.monotonic()-1)
        with self.assertRaises(driver.BoundaryStop):clock.active()
        self.assertEqual(clock.reason,'time_budget')
        clock.stop_time=time.monotonic()-91
        with self.assertRaisesRegex(driver.BoundaryStop,'grace'):clock.saving()


if __name__=='__main__':
    unittest.main()
