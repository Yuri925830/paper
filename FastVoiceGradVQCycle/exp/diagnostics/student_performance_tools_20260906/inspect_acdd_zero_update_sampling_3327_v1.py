"""固定S3327/D0与TRAIN3328，只比较120组时间步；不创建优化器或更新权重。"""

import argparse  # 导入文件不运行GPU，正式检查仍须另行批准。
import hashlib
import json
import math
import os
import platform
import statistics  # 每20组为固定模型重复测量，不称训练曲线。
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run only on the registered Matpool rental')
TOOLS=ROOT/'exp/diagnostics/student_performance_tools_20260906'
REPLICATION_SHA='a2dcb3cd636560d8bdd351982726b186e3925e11cc876891628af1d906b3ca98'
if hashlib.sha256((TOOLS/'run_acdd_sampler_replication3_v1.py').read_bytes()).hexdigest()!=REPLICATION_SHA:
    raise RuntimeError('Frozen replication helper changed')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TOOLS))
import torch
import run_acdd_sampler_replication3_v1 as replication  # 只复用采样、预算和日志，无调用其训练main。

cross,pair,base=replication.cross,replication.pair,replication.base
training,resume,probe=replication.training,replication.resume,replication.probe
DEV,PARENT=replication.DEV,replication.PARENT
OUTPUT=DEV/'acdd_zero_update_sampling_3327_v1'
PROTOCOL=DEV/'ACDD3327_ZERO_UPDATE_SAMPLING_PROTOCOL_CN.md'
PROTOCOL_SHA='84eb96c14c4807338b887b221754cd0f4559f23449a53e9e296b9a7bda1609f8'
ANCHOR=DEV/'acdd_d_joint_only_readonly_3327_v1/results.json'
REPLICATION_CONTRACT=replication.OUTPUT/'run_contract.json'
TEST_FILE=TOOLS/'test_acdd_zero_update_sampling_3327.py'
PINNED={
    str(PROTOCOL):PROTOCOL_SHA,
    str(TOOLS/'run_acdd_sampler_replication3_v1.py'):REPLICATION_SHA,
    str(ANCHOR):'17f56ae3424f5eb8ac8d0af2dbd1941d96365e0bd9a06f110e5f9e7652c89b09',
    str(REPLICATION_CONTRACT):'315be69dbe5f57848506a35b987dab2b92c55740379f1f3207ca3b6e95f76b84',
    str(PARENT/'final_full.pt'):replication.previous.PARENT_SHA,
    str(PARENT/'run_contract.json'):'787759339c886b2918bdda62abcc6818adf3cb410f923aca38c80bbd4b99b071',
    str(PARENT/'completion.json'):'68ee1cf0e680f3bf665be6dcf46e2aa75c3684262c2da3765ae2c7beebf931b2',
}
EXPECTED_COUNTS={'student':1,'vocoder':2,'serial_D_readouts':1,'teacher':122}
INPUT_KEYS=('x0','bnf','spk_emb','source_noise','score_noise','score_timesteps')


def watchdog():
    # 超过600秒只准进入恢复，外部timeout再留30秒后强制退出。
    parent=Path('/proc')/str(os.getppid())
    args=[part.decode() for part in (parent/'cmdline').read_bytes().split(b'\0') if part]
    if (parent/'exe').resolve()!=Path('/usr/bin/timeout').resolve() or not all(word in args for word in ('--signal=TERM','--kill-after=30s','600s')):
        raise RuntimeError('Require external timeout --signal=TERM --kill-after=30s 600s')
    return {'pid':os.getppid(),'command':args,'maximum_compute_seconds':600,'cleanup_only_seconds':30}


def planned_draws(contract):
    # 取原登记120个向量并独立调用冻结采样函数核对，顺序不能依据loss重新排列。
    if contract['seed_bases']!=list(replication.SEED_BASES) or contract['arm_order']!=list(replication.ARMS):
        raise RuntimeError('Replication seed/arm ordering changed')
    result=[]
    for seed in replication.SEED_BASES:
        for arm in replication.ARMS:
            saved=contract['timestep_draws'][str(seed)][arm]
            if len(saved)!=20:raise RuntimeError('Each registered arm must keep all20 t vectors')
            for offset,step in enumerate(range(3328,3348)):
                draw=replication.sample_timesteps(seed,step,arm)
                if draw!=saved[offset]:raise RuntimeError('Actual replication120-vector recipe changed')
                result.append(dict(draw,absolute_seed_label=step,readout_index=len(result)))
    return result


def model_description(model):
    # 参数和全部buffer的SHA之外，还保留每层模式、requires_grad及None梯度身份。
    return {'sha256':pair.model_sha(model),'modes':{name:item.training for name,item in model.named_modules()},
        'requires_grad':{name:item.requires_grad for name,item in model.named_parameters()},
        'gradients':{name:None if item.grad is None else cross.tensor_id(item.grad) for name,item in model.named_parameters()}}


def require_original_precision():
    if cross.flags()!=cross.ORIGINAL_FLAGS:
        raise RuntimeError('A forward changed the original precision flags')


def assert_unchanged(models,before,inputs,input_before,prediction=None,prediction_identity=None):
    # 在恢复之前执行；若前向偷偷改变buffer，不能靠finally恢复把变化掩盖。
    if {name:model_description(model) for name,model in models.items()}!=before:
        raise RuntimeError('Read-only forward changed parameters/buffers/modes/requires_grad/grad')
    if {name:cross.tensor_id(value) for name,value in inputs.items()}!=input_before:
        raise RuntimeError('Read-only forward changed an input tensor')
    require_original_precision()
    if prediction is not None and cross.tensor_id(prediction)!=prediction_identity:
        raise RuntimeError('Read-only forward changed the cached student prediction')


def prepare_inputs(dataset,indices,device):
    # 只取TRAIN3328这32句；不读取任何旧DEV张量或64句监测缓存。
    batch=training._slice_batch(base.batch_for(dataset,indices,104562),device)
    training.seed_everything(204562)
    inputs={'x0':batch['mel'],'bnf':batch['bnf'],'spk_emb':batch['spk_emb'],
        'source_noise':torch.randn_like(batch['mel']),'score_noise':torch.randn_like(batch['mel']),
        'score_timesteps':torch.randint(1,1001,(32,),device=device,dtype=torch.long)}
    return inputs


def validate_inputs(inputs,anchor):
    actual={name:cross.tensor_id(value) for name,value in inputs.items()}
    if set(actual)!=set(INPUT_KEYS) or any(actual[name]!=anchor['input_identities'][name] for name in INPUT_KEYS):
        raise RuntimeError('HistoricalTRAIN3328 data/source/score/t identity did not replay')
    return actual


@torch.no_grad()
def fixed_gan(student,vocoder,discriminator,diffusion,inputs,mean,std,anchor,counts,check):
    check();counts['student']+=1
    prediction,q950=training.make_student_prediction(student,diffusion,inputs['x0'],inputs['spk_emb'],inputs['bnf'],950,inputs['source_noise'])
    check();counts['vocoder']+=1
    real=vocoder(training.denormalize_mel(inputs['x0'],mean,std))
    check();counts['vocoder']+=1
    fake=vocoder(training.denormalize_mel(prediction,mean,std))
    tensors={'prediction':prediction,'q950':q950,'real_waveform':real,'fake_waveform':fake}
    if any(not bool(torch.isfinite(value).all()) or value.requires_grad for value in tensors.values()):
        raise RuntimeError('Invalid fixed no-gradient prediction or waveform')
    identities={name:cross.tensor_id(value) for name,value in tensors.items()}
    expected=anchor['shared_row_tensor_identities']['original']
    if any(value!=expected[name] for name,value in identities.items()):
        raise RuntimeError('OriginalS3327 prediction/q950/waveform SHA did not replay')
    check();counts['serial_D_readouts']+=1
    loss_d,adv,fm=training.serial_discriminator_metrics(discriminator,real,fake)
    scalars={'discriminator_loss':float(loss_d),'adv':float(adv),'fm_raw':float(fm)}
    for name,value in scalars.items():cross.require_close(value,anchor['matrix_3S_by_2D']['original']['D0'][name],1e-6,0)
    gt=(prediction-inputs['x0']).abs().flatten(1).mean(1)
    scalars.update(train_GT_l1=float((prediction-inputs['x0']).abs().mean()),per_utterance_GT_l1=gt.cpu().tolist(),
        prediction_rms=float(prediction.square().mean().sqrt()))
    return prediction,identities,scalars


@torch.no_grad()
def score_record(prediction,teacher,diffusion,inputs,t,config,gan,counts,check):
    if t.shape!=(32,) or t.dtype!=torch.long or bool(((t<1)|(t>1000)).any()):
        raise RuntimeError('Require32 actual integer timesteps within1..1000')
    check();counts['teacher']+=1  # 尝试前向即记数，异常不能假称所有调用都已完成。
    loss,target=training.score_distillation_loss(prediction,teacher,diffusion,t,inputs['score_noise'],inputs['spk_emb'],inputs['bnf'],
        target_mode=config.score_target,weight_mode=config.score_weight)
    if target.shape!=prediction.shape or target.requires_grad or not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite or incompatible frozen clean teacher target')
    unweighted=(prediction-target).abs().flatten(1).mean(1)
    weighted=unweighted*diffusion.sqrt_alphas_cumprod[t]
    cross.require_close(float(weighted.mean()),float(loss),1e-7,1e-6)
    total=gan['adv']+2*gan['fm_raw']+45*float(loss)
    if not math.isfinite(total) or not bool(torch.isfinite(unweighted).all() and torch.isfinite(weighted).all()):
        raise RuntimeError('Nonfinite score or original-coefficient total')
    values=t.cpu().tolist();labels=[(value-1)//100 for value in values]
    return {'timesteps':values,'t_identity':cross.tensor_id(t),'bin_counts':[labels.count(i) for i in range(10)],
        'per_utterance_unweighted_l1':unweighted.cpu().tolist(),'per_utterance_weighted_l1':weighted.cpu().tolist(),
        'teacher_target_identity':cross.tensor_id(target),'dist_raw':float(loss),'dist_weighted':45*float(loss),
        'student_adv':gan['adv'],'fm_raw':gan['fm_raw'],'fm_weighted':2*gan['fm_raw'],'student_total':total}


def require_exact_repeat(actual,first):
    # 相同模型/噪声/t的实际目标和每句数值都相同，才称精确重放通过。
    if actual!=first:raise RuntimeError('The final repeat of the firstIID sample was not exact')


def run_scores(prediction,teacher,diffusion,inputs,config,gan,anchor,draws,counts,check,emit):
    prediction_identity=cross.tensor_id(prediction)
    def measure(t):
        value=score_record(prediction,teacher,diffusion,inputs,t,config,gan,counts,check)
        if cross.tensor_id(prediction)!=prediction_identity:raise RuntimeError('A score readout changed the cached student prediction')
        return value
    old=measure(inputs['score_timesteps'])
    if old['teacher_target_identity']!=anchor['shared_row_tensor_identities']['original']['teacher_target']:
        raise RuntimeError('Historical original-t teacher target SHA did not replay')
    cross.require_close(old['dist_raw'],anchor['matrix_3S_by_2D']['original']['D0']['primary_moving_score'],1e-7,0)
    emit('anchor',old)
    rows=[];first=None
    for draw in draws:
        t=torch.tensor(draw['timesteps'],device=prediction.device,dtype=torch.long)
        value=measure(t)
        if value['timesteps']!=draw['timesteps'] or value['bin_counts']!=draw['bin_counts']:
            raise RuntimeError('Actual score timestep differs from predeclared vector')
        if first is None:first=value
        record=dict(value,seed_base=draw['seed_base'],arm=draw['arm'],absolute_seed_label=draw['absolute_seed_label'],
            timestep_seed=draw['seed'],readout_index=draw['readout_index'],model_global_step=3327,optimizer_updates=0)
        rows.append(record);emit('draw',record)
    if len(rows)!=120 or first is None:raise RuntimeError('Keep all120 predeclared measurements')
    repeat=measure(torch.tensor(draws[0]['timesteps'],device=prediction.device,dtype=torch.long))
    require_exact_repeat(repeat,first);emit('repeat',repeat)
    return rows,old,repeat


def summarize(rows):
    expected=[(seed,arm,step) for seed in replication.SEED_BASES for arm in replication.ARMS for step in range(3328,3348)]
    if [(x['seed_base'],x['arm'],x['absolute_seed_label']) for x in rows]!=expected:
        raise RuntimeError('Summary must retain original120 order, not a selected prefix')
    groups=[]
    for i in range(0,120,20):
        block=rows[i:i+20]
        values={name:[item[name] for item in block] for name in ('dist_raw','dist_weighted','student_total')}
        groups.append({'seed_base':block[0]['seed_base'],'arm':block[0]['arm'],'count':20,
            'statistics':{name:{'mean':math.fsum(value)/20,'population_std':statistics.pstdev(value),'minimum':min(value),'maximum':max(value)} for name,value in values.items()}})
    order={}
    for arm in replication.ARMS:
        by_seed=[group for group in groups if group['arm']==arm]
        order[arm]={}
        for name in ('dist_raw','student_total'):
            means=[group['statistics'][name]['mean'] for group in by_seed]
            order[arm][name]={'means_in_seed_order':means,'adjacent_differences':[means[1]-means[0],means[2]-means[1]],
                'strictly_increasing':means[0]<means[1]<means[2],'last_minus_first':means[2]-means[0]}
    return {'groups':groups,'three_windows_in_predeclared_seed_order':order,
        'interpretation':'Fixed-model repeated sampling windows, not optimizer steps or independent utterances; no real-training false-alarm estimate.'}


def main():
    started=time.monotonic();locks=[];signals=None;budget=None;log=None;owned=False
    saved_stdout,saved_stderr=sys.stdout,sys.stderr
    models={};restoration={};counts={key:0 for key in EXPECTED_COUNTS};stage='preflight'
    try:
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-approved',action='store_true')
        if not parser.parse_args().run_approved:raise RuntimeError('Explicit approval is required')
        signals=cross.DeadlineSignals(started,600);signals.install()
        external=watchdog();training.assert_rented_server(ROOT);cross.regular(DEV,directory=True)
        locks=training.acquire_runtime_locks(ROOT);budget=replication.MetadataBudget(OUTPUT)
        log=replication.open_exclusive_log(OUTPUT,budget);sys.stdout=sys.stderr=log
        def check():
            if signals.reason is not None or time.monotonic()-started>=600:raise cross.DiagnosticInterruption('Zero-update600-second deadline')
            require_original_precision()  # 不能等finally恢复后才检查是否曾改过数值精度。
            budget.require()
        def progress(name):
            nonlocal stage
            stage=name;check();print('ZERO_UPDATE_STAGE '+name+' '+json.dumps(counts),flush=True)
        progress('runtime_and_source')
        if training.git_head(ROOT)!=cross.HEAD or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Require unchanged clean production source')
        if (platform.python_version(),torch.__version__,torch.version.cuda)!=('3.8.15','1.13.1+cu116','11.6') or Path(sys.executable).resolve()!=Path('/root/miniconda3/envs/myconda/bin/python').resolve():
            raise RuntimeError('Require the original myconda runtime')
        if cross.flags()!=cross.ORIGINAL_FLAGS:raise RuntimeError('Original precision flags changed')
        torch.set_num_threads(2)  # 与原D-joint只读入口保持相同的CPU线程设定。
        gpu=cross.gpu_gate();disk=budget.require();cross.check_files(PINNED)
        anchor=cross.read_json(ANCHOR);contract=cross.read_json(PARENT/'run_contract.json');receipt=cross.read_json(PARENT/'completion.json')
        if (anchor['status'],anchor['parent_step'],anchor['next_batch_step'],anchor['parent_checkpoint_sha256'])!=('completed_readonly_D_joint_only_probe',3327,3328,replication.previous.PARENT_SHA):
            raise RuntimeError('Require the actual completed original3327/D0 anchor')
        original=anchor['matrix_3S_by_2D']['original']['D0']
        if {name:original[name] for name in ('adv','fm_raw','discriminator_loss','primary_moving_score')}!={'adv':2.0234532356262207,'fm_raw':.8152695894241333,'discriminator_loss':3.91922926902771,'primary_moving_score':.054745227098464966}:
            raise RuntimeError('Do not substitute the updatedD1 readout for originalD0')
        replication_contract=cross.read_json(REPLICATION_CONTRACT)
        if training.canonical_json_sha256(contract)!=replication.previous.PARENT_CANONICAL or training.canonical_json_sha256(replication_contract)!='f6cc75f05f0c2b9c0e349279b642a796d8a8ee6b84fa294059d0e730ae67e781':
            raise RuntimeError('Parent or actual replication contract changed')
        draws=planned_draws(replication_contract)
        development,development_files=resume.load_development(DEV/'development_contract.json')
        if contract['training_indices']!=development['retained_train_indices'] or contract['excluded_train_indices']!=development['excluded_train_indices']:
            raise RuntimeError('Original retained training pool or52 exclusions changed')
        indices=resume.GlobalBatchPlan(development['retained_train_indices']).for_step(3328)
        if len(indices)!=32 or indices!=anchor['training_indices'] or set(indices)&set(development['excluded_train_indices']):
            raise RuntimeError('Require original TRAIN3328 indices, outside all52 exclusions')
        config=base.replace(base.PAPER_STUDENT_CONFIG,num_workers=0)
        if config.as_dict()!=contract['configuration'] or (config.score_target,config.score_weight,config.lambda_fm,config.lambda_dist)!=('clean_x0','sqrt_alpha_bar',2,45):
            raise RuntimeError('Original objective/configuration changed')
        # 不读取旧DEV缓存或其他学生断点；只继承非pt依赖及本次真正使用的T/G权重。
        allowed_weights={str(base.TEACHER),str(base.VOCODER_DIR/'generator_step_2500000.pt')}
        files={path:digest for path,digest in contract['file_sha256'].items() if not path.endswith('.pt') or path in allowed_weights}
        cross.merge_hashes(files,PINNED);cross.merge_hashes(files,development_files)
        # 当前真正导入的本项目Python源也全部登记；与既有固定锚点冲突时拒绝。
        for module in list(sys.modules.values()):
            value=getattr(module,'__file__',None)
            if value:
                path=Path(value).resolve()
                if ROOT in path.parents and path.suffix=='.py':cross.merge_hashes(files,{str(path):base.sha256_file(path)})
        for path in (Path(__file__).resolve(),TEST_FILE):cross.merge_hashes(files,{str(path):base.sha256_file(path)})
        dataset=base.VoiceGradDataset(ROOT/'data','train',segment_length=128,expected_speakers=100,bnf_alignment=config.bnf_alignment)
        features=replication.previous.register_future_files(dataset,[{'indices':indices}],files,check)
        cross.check_files(files);check()
        identity={'kind':'zero_update_fixed3327_timestep_sampling','protocol_sha256':PROTOCOL_SHA,
            'parent_checkpoint_sha256':replication.previous.PARENT_SHA,'parent_contract_sha256':replication.previous.PARENT_CANONICAL,
            'model_global_step':3327,'discriminator_state':'original_D0_before_any_update','configuration':config.as_dict(),
            'training_indices':indices,'crop_seed':104562,'noise_seed':204562,'noise_draw_order':['source_noise','score_noise','score_timesteps'],
            'expected_input_identities':{key:anchor['input_identities'][key] for key in INPUT_KEYS},
            'expected_model_identities':anchor['model_state_sha256'],'draws':draws,'file_sha256':dict(files),'batch_feature_sha256':features,
            'expected_forward_counts':EXPECTED_COUNTS,'optimizer_instances':0,'backward_calls':0,'model_writes':0,
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'flags':cross.flags(),'cpu_threads':torch.get_num_threads(),'gpu':gpu,'external_timeout':external},
            'initial_disk':disk,'limits':['TRAIN3328 fixed32 reused120 times, not3840 independent utterances.',
                'No actual training steps, no proposed change to loss/stopping rule, no DEV/test score or checkpoint selection.']}
        OUTPUT.mkdir();owned=True;budget.write(OUTPUT/'run_identity.json',identity)
        identity_sha=training.canonical_json_sha256(identity)
        device=torch.device('cuda');torch.cuda.reset_peak_memory_stats(device)
        with cross.preserved_state(models,restoration,signals.cleanup_only) as register:
            progress('original_inputs_and_models')
            inputs=prepare_inputs(dataset,indices,device);input_before=validate_inputs(inputs,anchor)
            payload=resume.load_torch_checkpoint(PARENT/'final_full.pt')
            saved_adam_rng=probe.tree_sha({'s':payload['student_optimizer'],'d':payload['discriminator_optimizer'],'rng':payload['rng_state']})
            if saved_adam_rng!=anchor['source_twoAdam_rng_sha256']:raise RuntimeError('Original saved Adam/RNG metadata changed')
            teacher,student,_=base.load_teacher_and_student(base.TEACHER,config,device)
            vocoder,vconfig,_=base.load_frozen_vocoder(ROOT,base.VOCODER_DIR/'generator_step_2500000.pt',base.VOCODER_DIR/'config.json',
                base.VOCODER_DIR/'final_checkpoint_attestation_comments_20260906.json',config,device)
            discriminator=base.build_paper_discriminator(ROOT,vconfig,config,device)
            replication.previous.dstate.validate_parent(payload,student,discriminator,config,contract,receipt)
            student.load_state_dict(payload['student'],strict=True);discriminator.load_state_dict(payload['discriminator'],strict=True)
            student.train();discriminator.train();teacher.eval();vocoder.eval()
            training.set_requires_grad(student,True);training.set_requires_grad(discriminator,True)
            diffusion=base.VoiceGradDiffusion(config.diffusion_steps,config.cosine_offset).to(device)
            for name,model in (('student',student),('discriminator',discriminator),('teacher',teacher),('vocoder',vocoder),('diffusion',diffusion)):
                register(name,model)
            before={name:model_description(model) for name,model in models.items()}
            expected={'student':anchor['model_state_sha256']['students']['original'],'discriminator':anchor['model_state_sha256']['discriminators']['D0'],
                'teacher':anchor['model_state_sha256']['teacher'],'vocoder':anchor['model_state_sha256']['vocoder']}
            if any(before[name]['sha256']!=value for name,value in expected.items()):raise RuntimeError('OriginalS/D0/T/G parameters or buffers changed')
            pair.assert_frozen({'teacher':teacher,'vocoder':vocoder},{name:expected[name] for name in ('teacher','vocoder')})
            mean,std=base.load_mel_statistics(ROOT/'data',device)
            protected_inputs=dict(inputs,mel_mean=mean,mel_std=std)
            protected_before={name:cross.tensor_id(value) for name,value in protected_inputs.items()}
            progress('fixed_oneS_twoV_oneD')
            prediction,shared_ids,gan=fixed_gan(student,vocoder,discriminator,diffusion,inputs,mean,std,anchor,counts,check)
            assert_unchanged(models,before,protected_inputs,protected_before,prediction,shared_ids['prediction'])
            budget.write(OUTPUT/'fixed_readout.json',{'input_identities':input_before,'protected_input_and_stats_identities':protected_before,
                'shared_tensor_identities':shared_ids,'gan_and_GT':gan,'model_state_before':before})
            def emit(kind,value):
                check();budget.write(OUTPUT/(('samples.jsonl' if kind=='draw' else kind+'.json')),value,append=kind=='draw')
                if kind!='draw' or (value['readout_index']+1)%20==0:progress(kind+'_'+str(counts['teacher']))
            progress('anchor_then120_then_repeat')
            rows,old,repeat=run_scores(prediction,teacher,diffusion,inputs,config,gan,anchor,draws,counts,check,emit)
            if counts!=EXPECTED_COUNTS:raise RuntimeError('Registered1S2V1D122T count mismatch')
            progress('verify_before_restoration')
            assert_unchanged(models,before,protected_inputs,protected_before,prediction,shared_ids['prediction'])
            if probe.tree_sha({'s':payload['student_optimizer'],'d':payload['discriminator_optimizer'],'rng':payload['rng_state']})!=saved_adam_rng:
                raise RuntimeError('Read-only loading changed archived Adam/RNG payload')
            summary=summarize(rows);check()
        # finally已恢复RNG/标志及所有登记对象；前面的未变化断言仍是独立必要条件。
        progress('final_identity_validation')
        cross.check_files(files);cross.check_files(PINNED)
        if training.canonical_json_sha256(cross.read_json(OUTPUT/'run_identity.json'))!=identity_sha:raise RuntimeError('Published identity changed')
        result={'status':'zero_update_sampling_complete','run_identity_sha256':identity_sha,'forward_counts':counts,
            'fixed_model_global_step':3327,'original_t_anchor':old,'first_sample_repeat':repeat,'repeat_exact':True,
            'fixed_gan_and_train_GT':gan,'statistics':summary,'all_models_inputs_unchanged_before_restore':True,
            'restoration':restoration,'source_twoAdam_rng_unchanged':True,'optimizer_instances':0,'backward_calls':0,'model_checkpoint_writes':0,
            'runtime_peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'runtime_peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'elapsed_seconds':time.monotonic()-started,'limitations':identity['limits']}
        check();budget.write(OUTPUT/'results.json',result)  # 最后一个正常发布操作，之后不再计算。
    except BaseException as error:
        if signals is not None:signals.cleanup_only()
        failure={'status':'failed','stage':stage,'error_type':type(error).__name__,'error':str(error),
            'attempted_forward_counts':counts,'restoration':restoration,'model_checkpoint_writes':0,'completion_valid':False,
            'old_artifacts_not_written_by_this_diagnostic':True}
        try:print('ZERO_UPDATE_FAILED '+json.dumps(failure,ensure_ascii=False),flush=True)
        except BaseException:pass
        if owned:
            # 元数据commit后若预算校验失败，本次results也必须标成失败，不能保留假成功。
            for path in ([OUTPUT/'results.json'] if (OUTPUT/'results.json').exists() else [])+[OUTPUT/'failure.json']:
                try:budget.write(path,failure)
                except BaseException:pass
        raise
    finally:
        if signals is not None:
            try:signals.close()
            except BaseException:pass
        sys.stdout,sys.stderr=saved_stdout,saved_stderr
        if log is not None:
            try:log.flush();log.close()
            except BaseException:pass
        for handle in reversed(locks):
            try:handle.close()
            except BaseException:pass


if __name__=='__main__':
    main()  # 不自动训练、不评分、不提升模型，须根线程另行批准一次GPU诊断。
