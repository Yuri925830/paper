"""只在Matpool租用机CPU验证六臂编排、真实小图更新、身份与仅日志保存。"""

import ast  # 静态检查新入口没有保存模型或修改旧模块全局的语句。
import copy
import inspect
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile  # 测试写入只位于自己创建的临时目录。
import unittest
from pathlib import Path
from unittest import mock

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only rented Matpool CPU tests with CUDA_VISIBLE_DEVICES empty are allowed')

import torch
import run_acdd_sampler_replication3_v1 as driver
import test_conversion_score_step as fixture
import test_acdd_timestep_paired60 as frozen_tests  # 重跑原恢复/监测断言，不修改冻结测试。


def loss_row(step,value):
    return {'global_step':step,'updates':step-driver.START,**{key:float(value) for key in driver.pair.LOSS_KEYS}}


def monitored(step,gt=.19,score=.045,train_gt=.19):
    return {'global_step':step,'development':{'mel_l1':gt,'per_utterance_mel_l1':[gt]*32,'prediction_rms':1.},
        'uniform64':{'uniform_weighted_score':score,'GT_l1':train_gt,'per_utterance_GT_l1':[train_gt]*64,
            'per_stratum_per_utterance_weighted_score':[[score]*64 for _ in range(10)]}}


def arm_result(guard=None,gt=.19,score=.045,train_gt=.19):
    return {'last_monitor':monitored(3347,gt,score,train_gt),'guard_reasons':guard or [],
            'window':driver.describe_window([loss_row(step,5.9) for step in range(3328,3348)])}


class ReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)  # 只限制这个CPU测试进程，不改变真实训练环境。

    def test_registered_sampling_exact_recipe_and_no_global_rng_consumption(self):
        driver.training.seed_everything(61)
        before=driver.probe.tree_sha(driver.training.capture_rng_state())
        for seed in driver.SEED_BASES:
            for step in range(3328,3348):
                for arm in driver.ARMS:
                    actual=driver.sample_timesteps(seed,step,arm)
                    local=random.Random(seed+step)
                    if arm=='iid_control':
                        expected=[local.randint(1,1000) for _ in range(32)]
                        self.assertEqual(actual['bonus_bins'],[])
                    else:
                        bonus=local.sample(range(10),2)
                        labels=[i for i in range(10) for _ in range(3)]+bonus;local.shuffle(labels)
                        expected=[local.randint(100*i+1,100*i+100) for i in labels]
                        self.assertEqual(actual['bonus_bins'],bonus)
                        self.assertEqual(sorted(actual['bin_counts']),[3]*8+[4]*2)
                    self.assertEqual(actual['timesteps'],expected)
                    self.assertEqual(actual['labels'],[(t-1)//100 for t in expected])
                    self.assertEqual(sum(actual['bin_counts']),32)
        self.assertEqual(driver.probe.tree_sha(driver.training.capture_rng_state()),before)

    def test_bounds_and_fixed_six_arm_order(self):
        for args in ((73083328,3328,driver.ARMS[0]),(73100000,3327,driver.ARMS[0]),(73100000,3348,driver.ARMS[0]),(73100000,3328,'best_seed')):
            with self.assertRaises(RuntimeError):driver.sample_timesteps(*args)
        calls=[]
        def run(seed,arm,control):
            calls.append((seed,arm));return arm_result(['ordinary_guard_failure'])
        results=driver.execute_registered(run,lambda s,c,e:driver.pair_result(s,c,e,{'mel_l1':.1925}))
        self.assertEqual(calls,[(seed,arm) for seed in driver.SEED_BASES for arm in driver.ARMS])
        self.assertEqual(len(results),3)
        self.assertTrue(all(not row['candidate_gate']['passed'] for row in results))

    def test_early_success_never_skips_predeclared_seed_pairs(self):
        calls=[]
        def run(seed,arm,control):
            calls.append((seed,arm));return arm_result()
        results=driver.execute_registered(run,lambda s,c,e:driver.pair_result(s,c,e,{'mel_l1':.1925}))
        self.assertEqual(len(calls),6);self.assertEqual(len(results),3)
        self.assertTrue(all(row['candidate_gate']['passed'] for row in results))

    def test_every_execution_error_aborts_all_later_arms(self):
        for error in (driver.previous.BoundaryStop('time_budget'),driver.pair.InsufficientDisk('disk'),
                      RuntimeError('nonfinite'),RuntimeError('identity mismatch'),RuntimeError('half D step')):
            calls=[];published=[]
            def run(seed,arm,control):
                calls.append((seed,arm))
                if len(calls)==3:raise error
                return arm_result()
            with self.assertRaises(type(error)):
                driver.execute_registered(run,lambda s,c,e:published.append(s))
            self.assertEqual(len(calls),3);self.assertEqual(published,[73100000])

    def test_control_guard_failure_is_not_hidden_by_candidate_success(self):
        control=arm_result(['original_S0_guard'],gt=.191)
        candidate=arm_result(gt=.19)
        result=driver.pair_result(73100000,control,candidate,{'mel_l1':.1925})
        self.assertFalse(result['candidate_gate']['passed'])
        self.assertEqual(result['candidate_gate']['stop_reasons'],['control:original_S0_guard'])
        self.assertEqual(result['DEV32']['improved'],32)
        self.assertEqual(len(result['uniform64_score']['candidate_minus_control']),64)

    def test_raw20_means_population_std_and_all_utterances(self):
        rows=[loss_row(step,float(step-3327)) for step in range(3328,3348)]
        actual=driver.describe_window(rows)
        self.assertEqual(actual['means']['student_total'],10.5)
        self.assertEqual(actual['population_standard_deviations']['dist_raw'],statistics.pstdev(range(1,21)))
        with self.assertRaises(RuntimeError):driver.describe_window(rows[:-1])
        with self.assertRaises(RuntimeError):driver.count_changes([0.]*63,[1.]*64,64)
        with self.assertRaises(RuntimeError):driver.count_changes([float('nan')]*32,[1.]*32,32)
        changes=driver.count_changes([0.,1.,2.],[1.,1.,1.],3)
        self.assertEqual((changes['improved'],changes['equal'],changes['worsened']),(1,1,1))

    def test_real_prior_contract_first20_plan_identity_and_no_dev_leak(self):
        old,completion,rows,initial=driver.load_old_trial()
        development,files=driver.resume.load_development(driver.DEV/'development_contract.json')
        future=driver.future_plan(development,old,rows)
        self.assertEqual(len(future),20);self.assertEqual(future[0]['global_step'],3328)
        self.assertEqual(future[-1]['global_step'],3347)
        self.assertEqual(initial['global_step'],3327)
        self.assertEqual(completion['arms']['control_iid']['restore_identity']['student'],
            'd878d91bbe16d6b950d40558d46ee8e89eacbf2670ed7d8bfe8fd1b5e8269131')
        self.assertEqual(len(development['excluded_train_indices']),52)
        for planned,row in zip(future,rows):
            self.assertEqual(planned['indices'],row['indices'])
            self.assertFalse(set(planned['indices'])&set(development['excluded_train_indices']))
            changed=copy.deepcopy(row);changed['dist_raw']+=.1;changed['student_total']+=4.5
            driver.previous.common_input_check(changed,row)  # 新t允许score/total改变。
        changed=copy.deepcopy(rows[0]);changed['fm_raw']+=.1
        with self.assertRaises(RuntimeError):driver.previous.common_input_check(changed,rows[0])
        changed=copy.deepcopy(rows[-1]);changed['sampler']['source_noise']['sha256']='bad'
        with self.assertRaises(RuntimeError):driver.previous.common_input_check(changed,rows[-1])

    def test_actual_hook_both_new_streams_keep_three_draws_and_first_gan(self):
        for seed in driver.SEED_BASES:
            values=[];states=[];receipts=[]
            for arm in driver.ARMS:
                args=fixture.make_fixture(torch.float32);args[6].param_groups[0]['lr']=5e-5
                draw=driver.sample_timesteps(seed,3328,arm);receipt={}
                driver.training.seed_everything(204562)
                values.append(driver.previous.sampler.train_batch_with_timesteps(*args,
                    score_timesteps_override=torch.tensor(draw['timesteps']),sampler_receipt=receipt))
                receipts.append(receipt);states.append(driver.probe.tree_sha(driver.training.capture_rng_state()))
                self.assertEqual(receipt['used_values'],draw['timesteps'])
                self.assertEqual(receipt['used_bin_counts'],draw['bin_counts'])
                self.assertEqual(args[1].trace[-1]['t'].tolist(),draw['timesteps'])
                self.assertTrue(all(p.grad is None for model in args[1:5] for p in model.parameters()))
                driver.previous.adam_counters(args[6],args[7],1)
            for key in ('source_noise','score_noise','original_uniform_t','original_uniform_values'):
                self.assertEqual(receipts[0][key],receipts[1][key])
            self.assertEqual(states[0],states[1])
            self.assertNotEqual(receipts[0]['used_values'],receipts[1]['used_values'])
            for key in ('discriminator','student_adv','fm_raw','fm_weighted'):self.assertEqual(values[0][key],values[1][key])

    def test_actual_cold_restore_two_adams_rng_and_final_restore(self):
        args=fixture.make_fixture(torch.float32);s,d=args[2],args[4];opt,dopt=args[6:8]
        opt.param_groups[0]['lr']=5e-5
        driver.previous.sampler.production.train_batch(*args)
        for optimizer in (opt,dopt):
            for item in optimizer.state.values():item['step'].fill_(3327)
        payload={'student':copy.deepcopy(s.state_dict()),'discriminator':copy.deepcopy(d.state_dict()),
            'student_optimizer':copy.deepcopy(opt.state_dict()),'discriminator_optimizer':copy.deepcopy(dopt.state_dict()),
            'rng_state':driver.training.capture_rng_state()}
        payload_sha=driver.probe.tree_sha(payload);restorations=[]
        with mock.patch.object(driver.previous.dstate,'validate_parent'):
            for seed in driver.SEED_BASES:
                for arm in driver.ARMS:
                    restorations.append(driver.previous.restore_parent(payload,s,d,opt,dopt,args[-1],{}, {},torch.device('cpu')))
                    draw=driver.sample_timesteps(seed,3328,arm)
                    driver.training.seed_everything(204562)
                    driver.previous.sampler.train_batch_with_timesteps(*args,score_timesteps_override=torch.tensor(draw['timesteps']))
                    driver.previous.adam_counters(opt,dopt,3328)
            final=driver.previous.restore_parent(payload,s,d,opt,dopt,args[-1],{}, {},torch.device('cpu'))
        self.assertTrue(all(item==final for item in restorations))
        self.assertEqual(driver.probe.tree_sha(payload),payload_sha)
        self.assertEqual((opt.param_groups[0]['lr'],dopt.param_groups[0]['lr']),(5e-5,2e-4))

    def test_real_parent_adam_and_original_fixed64_schema(self):
        # 重用冻结真实CPU校验；其内部只读真实父权重，检查双Adam/RNG并不前向大模型。
        frozen_tests.Paired60Tests().test_real_parent_two_adam_roles_counters_shapes_and_rng_without_update()
        frozen_tests.Paired60Tests().test_real_fixed64_replay_schema_keeps_all_per_sentence_fields()

    def test_actual_full_monitor_all_twelve_calls_all640_values_and_no_mutation(self):
        args=fixture.make_fixture(torch.float32)
        teacher,student,diffusion,config=args[1],args[2],args[5],args[-1]
        batches=[]
        for index in range(2):
            x=args[0]['mel'].clone()+index*.05;source,score=torch.randn_like(x),torch.randn_like(x)
            batches.append({'x0':x,'bnf':args[0]['bnf'].clone(),'spk_emb':args[0]['spk_emb'].clone(),
                'source_noise':source,'score_noise':score,'q950':diffusion.q_sample(x,torch.full((32,),950,dtype=torch.long),noise=source)})
        local=random.Random(73073341)
        plan={'strata_timesteps':[[local.randint(100*i+1,100*i+100) for _ in range(64)] for i in range(10)]}
        models={'student':student,'teacher':teacher,'diffusion':diffusion}
        before=(driver.probe.tree_sha(batches),{key:driver.pair.model_sha(value) for key,value in models.items()},
                driver.probe.tree_sha(driver.training.capture_rng_state()))
        original=driver.training.score_distillation_loss;count={'student':0,'teacher':0}
        for _ in range(12):
            teacher.trace.clear();student.trace.clear();captured=[]
            def record(*a,**kw):
                loss,target=original(*a,**kw);captured.append((a[0],target,a[3]))
                return loss,target  # 同一次输出直接留作独立按句归约，不增加前向。
            with mock.patch.object(driver.training,'score_distillation_loss',side_effect=record):
                with driver.previous.monitoring_state(models,torch.device('cpu')):
                    result=driver.previous.uniform64_monitor(student,teacher,diffusion,batches,plan,config,torch.device('cpu'),lambda:None)
            self.assertEqual((len(student.trace),len(teacher.trace)),(2,20))
            count['student']+=len(student.trace);count['teacher']+=len(teacher.trace)
            all_values=[]
            for layer in range(10):
                per=[]
                for block in range(2):
                    y,target,t=captured[10*block+layer]
                    per.extend(((y-target).abs().flatten(1).mean(1)*diffusion.sqrt_alphas_cumprod[t]).tolist())
                self.assertEqual(result['per_stratum_per_utterance_weighted_score'][layer],per);all_values.extend(per)
            self.assertAlmostEqual(result['uniform_weighted_score'],sum(all_values)/640,places=13)
            y=torch.cat([captured[0][0],captured[10][0]]);x=torch.cat([item['x0'] for item in batches])
            self.assertEqual(result['GT_l1'],float((y-x).abs().mean()))
            self.assertEqual(result['prediction_rms'],float(y.square().mean().sqrt()))
        self.assertEqual(count,{'student':24,'teacher':240})
        after=(driver.probe.tree_sha(batches),{key:driver.pair.model_sha(value) for key,value in models.items()},
               driver.probe.tree_sha(driver.training.capture_rng_state()))
        self.assertEqual(before,after)

    def test_half_update_and_metadata_failure_never_claim_new_checkpoint(self):
        state={'completed_global_step':3327,'new_updates':0,'last_full_saved_global_step':3327,'partial_batch_may_have_updated_D':False}
        with self.assertRaisesRegex(RuntimeError,'half'):
            driver.complete_update(state,3328,lambda:(_ for _ in ()).throw(RuntimeError('half D update')))
        self.assertEqual(state['completed_global_step'],3327);self.assertTrue(state['partial_batch_may_have_updated_D'])
        record=driver.failure_payload(state,{},OSError('disk failure'))
        self.assertEqual(record['last_full_saved_global_step'],3327)
        self.assertFalse(record['new_checkpoint_created']);self.assertFalse(record['successful_completion_valid'])
        self.assertFalse(record['parent_restoration_verified'])

    def test_new_twenty_update_bound_refuses_old_helper_sixty_limit(self):
        state={'completed_global_step':3347,'new_updates':20,'partial_batch_may_have_updated_D':False}
        call=mock.Mock()
        for step in (3327,3348,3387):
            with self.assertRaises(RuntimeError):driver.complete_update(state,step,call)
        call.assert_not_called();self.assertEqual(state['completed_global_step'],3347)
        self.assertEqual(driver.previous.STOP,3387)  # 旧模块常量没有为本次20步被修改。

    def test_new_arm_failure_cannot_inherit_previous_completed_state(self):
        old={'completed_global_step':3347,'seed_base':73100000,'arm':'stratified_candidate'}
        new=driver.new_arm_state(73200000,'iid_control','contract')
        record=driver.failure_payload(new,{'old':old},RuntimeError('optimizer construction failed'))
        self.assertEqual(record['active_arm']['completed_global_step'],3327)
        self.assertEqual(record['active_arm']['phase'],'restoring')
        self.assertFalse(record['active_arm']['restore_completed'])
        self.assertEqual(record['completed_arms']['old']['completed_global_step'],3347)

    def test_postpublication_deadline_invalidates_completion_and_tracks_actual_restore(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);budget=driver.MetadataBudget(output)
            with mock.patch.object(driver.pair,'available_bytes',return_value=100*1024**3):
                budget.write(output/'completion.json',{'status':'all_three_registered_pairs_completed'})
                failure=driver.failure_payload({}, {},driver.previous.BoundaryStop('time_budget'),True)
                self.assertEqual(driver.write_failure(output,failure,budget),[])
                actual=json.loads((output/'completion.json').read_text())
                self.assertEqual(actual['status'],'execution_aborted')
                self.assertTrue(actual['parent_restoration_verified'])
                self.assertFalse(actual['successful_completion_valid'])
                self.assertEqual(actual,json.loads((output/'failure.json').read_text()))

    def test_budget_exact_threshold_log_and_no_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/'trial';budget=driver.MetadataBudget(output)
            required=driver.RESERVE+driver.DIR_RESERVE+driver.META_LIMIT
            with mock.patch.object(driver.pair,'available_bytes',return_value=required):
                self.assertEqual(budget.require()['required_bytes'],required)
                log=driver.open_exclusive_log(output,budget);log.write('small log\n');log.close()
                output.mkdir();budget.write(output/'data.json',{'keep':[1,2,3]})
                self.assertGreater(budget.used(),10)
                with self.assertRaises(RuntimeError):budget.write(output/'bad.pt',{})
            with mock.patch.object(driver.pair,'available_bytes',return_value=1):
                with self.assertRaises(driver.pair.InsufficientDisk):budget.require()
            with mock.patch.object(budget,'used',return_value=driver.META_LIMIT):
                with self.assertRaises(driver.pair.InsufficientDisk):budget.require(1)

    def test_exclusive_log_refuses_existing_files_and_does_not_use_stdout(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/'trial';budget=driver.MetadataBudget(output)
            with mock.patch.object(driver.pair,'available_bytes',return_value=100*1024**3):
                log=driver.open_exclusive_log(output,budget)
                with mock.patch.object(sys,'stdout',mock.Mock(write=mock.Mock(side_effect=BrokenPipeError))):log.write('kept despite unrelated closed pipe\n')
                log.close()
                original=Path(str(output)+'.log').read_bytes()
                with self.assertRaises(RuntimeError):driver.open_exclusive_log(output,budget)
                self.assertEqual(Path(str(output)+'.log').read_bytes(),original)
                other=Path(name)/'already';other.mkdir()
                with self.assertRaises(RuntimeError):driver.open_exclusive_log(other,driver.MetadataBudget(other))

    def test_metadata_atomic_failure_preserves_original_and_truthful_failure(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);budget=driver.MetadataBudget(output)
            with mock.patch.object(driver.pair,'available_bytes',return_value=100*1024**3):
                budget.write(output/'status.json',{'completed':3327})
                with mock.patch.object(driver.os,'replace',side_effect=OSError('atomic commit fault')):
                    with self.assertRaises(OSError):budget.write(output/'status.json',{'completed':3347})
                self.assertEqual(json.loads((output/'status.json').read_text()),{'completed':3327})
                with self.assertRaises(FileExistsError):budget.write(output/'status.json',{'retry':True})

    def test_metadata_refuses_parent_symlink_and_dotdot_escape(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);output=root/'trial';output.mkdir();outside=root/'other';outside.mkdir()
            (output/'link').symlink_to(outside,target_is_directory=True)
            budget=driver.MetadataBudget(output)
            with mock.patch.object(driver.pair,'available_bytes',return_value=100*1024**3):
                for path in (output/'link'/'bad.json',output/'..'/'other'/'bad.json'):
                    with self.assertRaises(RuntimeError):budget.write(path,{})
            self.assertFalse((outside/'bad.json').exists())

    def test_no_model_save_no_global_mutation_and_original_clone_still_matches(self):
        tree=ast.parse(Path(driver.__file__).read_text())
        forbidden=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                if name in ('save','atomic_torch_save','save_primary') or (isinstance(node.func,ast.Name) and name in ('exec','eval')):
                    forbidden.append(name)
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id in ('previous','torch','training'):
                        forbidden.append(ast.dump(target))
        self.assertEqual(forbidden,[])
        frozen_tests.Paired60Tests().test_static_clone_has_exactly_one_hook_and_only_two_added_keyword_arguments()

    def test_internal_signal_aborts_schedule_and_does_not_start_next_arm(self):
        code="""import signal
import run_acdd_sampler_replication3_v1 as d
clock=d.previous.StopClock(1200);clock.install();calls=[]
try:
    def run(seed,arm,control):
        calls.append((seed,arm));signal.raise_signal(signal.SIGTERM);clock.active()
    try:d.execute_registered(run,lambda *a:None)
    except d.previous.BoundaryStop:pass
    else:raise AssertionError('signal was ignored')
    assert len(calls)==1 and clock.reason=='requested_stop'
    clock.saving();print('SIGNAL_ABORT_OK')
finally:clock.close()
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parent,env=dict(os.environ),
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('SIGNAL_ABORT_OK',result.stdout)


if __name__=='__main__':
    unittest.main()
