"""122900步完整学生的唯一正式8100对推理；不训练、不评分、不改旧入口。"""
import argparse
import gc
import hashlib
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT=Path('/root/FsatVoiceGrad')
TOOLS=ROOT/'exp/diagnostics/student_performance_tools_20260906'
DEV=ROOT/'exp/diagnostics/student_development_20260906'
OLD=TOOLS/'infer_formal_student3327_v1.py'
OLD_SHA='1699a7961306244a3df73f02ce277276ed9d6f217c8f7fe5c6985e84a6482f32'
if os.name!='posix' or not (ROOT/'AGENTS.md').is_file():raise RuntimeError('Execution is restricted to the Matpool rental')
if hashlib.sha256(OLD.read_bytes()).hexdigest()!=OLD_SHA:raise RuntimeError('Frozen original formal inference source changed')
sys.path.insert(0,str(TOOLS));sys.path.insert(0,str(ROOT))
import infer_formal_student3327_v1 as old

# 只复用不锁旧模型身份的函数；旧模块和旧常量均不修改。
np,torch,formal,training,teacher_infer=old.np,old.torch,old.formal,old.training,old.teacher_infer
digest,read_json,json_bytes=old.digest,old.read_json,old.json_bytes
verify_files,merge_hashes,ensure_regular=old.verify_files,old.merge_hashes,old.ensure_regular
publish_exact,prepare_output,append_record=old.publish_exact,old.prepare_output,old.append_record
scan_journal,finish_prepared,validate_wave=old.scan_journal,old.finish_prepared,old.validate_wave
tensor_sha,parameter_fingerprint=old.tensor_sha,old.parameter_fingerprint
current_runtime,gpu_idle=old.current_runtime,old.gpu_idle
RUN=DEV/'acdd_mc1_longtrain_from3327_v1'
OUTPUT=ROOT/'exp/evaluation/fastvoicegrad_text_unseen_s122900_custom_hifigan_20260909'
STUDENT=RUN/'latest_full.pt'
STUDENT_SHA='2ca44269eefac2dd3f8ffa7afe0337f1f3684469aec9fadeff623237cf37b830'
CONTRACT=RUN/'run_contract.json'
CONTRACT_SHA='79d007d0a78796abfe3eeb8c80ff8482cf19dff55bfa33306a31628c2043d1ba'
CONTRACT_FILE_SHA='8774d819b07a6bce0fca166309cbd4493fae4349b0fa2f913700a43f94a4d99c'
COMPLETION=RUN/'attempt_0001/completion.json'
COMPLETION_SHA='d09147b8561f15953a4ade37d38f2ee94beb220e8b2ca6188a9a1a7558720546'
POINTER=RUN/'latest_full.json'
POINTER_SHA='5310f6e2284d5addf09b7b80596a3eff7642d3390b96dd57dbf731b9397334a6'
PARENT_SHA='11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556'
SYSTEM='fastvoicegrad_student122900'
SYSTEM_LABEL='FastVoiceGrad学生122900步（100个完整训练pass）'
PAIR_COUNT,BATCH_SIZE,SEED=8100,4,1234
STEP,PASSES=122900,100
LRS={'student':.00005,'discriminator':.0002}
RESERVE=1024**3+128*1024**2
METADATA_BUDGET,DIRECTORY_BUDGET=32*1024**2,1024**2
EXPECTED_START_FREE=3234639872
EXPECTED_RUNTIME,EXPECTED_HEAD=old.EXPECTED_RUNTIME,old.EXPECTED_HEAD

# 仅当前full/完成记录、实际推理源码及固定正式输入。旧3327权重和训练历史资产不是依赖。
PINNED={str(ROOT/key):sha for key,sha in old.SOURCE_SHA.items()}
PINNED.update({str(OLD):OLD_SHA,str(STUDENT):STUDENT_SHA,str(CONTRACT):CONTRACT_FILE_SHA,
    str(COMPLETION):COMPLETION_SHA,str(POINTER):POINTER_SHA})
for path in (old.TEACHER_PAIRS,old.TEACHER_DIR/'inference_report.json',old.TEACHER_DIR/'run_identity.json',
             old.OLD_IDENTITY,old.VOCODER,old.VOCODER_DIR/'config.json',old.ATTESTATION):
    PINNED[str(path)]=old.PINNED[str(path)]


def validate_end(contract,completion,pointer):
    if training.canonical_json_sha256(contract)!=CONTRACT_SHA:raise RuntimeError('Actual long-training contract changed')
    if contract.get('kind')!='user_selected3327_MC1_longtrain_v1' or contract.get('stop_global_step')!=STEP or contract.get('total_passes')!=PASSES:
        raise RuntimeError('Wrong long-training endpoint contract')
    if contract.get('mc_samples')!=1 or contract.get('effective_optimizer_lrs')!=LRS or contract.get('origin_parent_sha256')!=PARENT_SHA:
        raise RuntimeError('Wrong actual MC1 training lineage/settings')
    if completion.get('status')!='completed100passes' or completion.get('completed_global_step')!=STEP or completion.get('last_saved_global_step')!=STEP or completion.get('partial_batch') is not False:
        raise RuntimeError('Training has not completed and saved the full endpoint')
    expected={'student':{'parameter_count':71,'uniform_step':STEP},'discriminator':{'parameter_count':144,'uniform_step':STEP}}
    if completion.get('optimizer_cursors')!=expected or completion.get('adam_returns')!={'student':1,'discriminator':1}:
        raise RuntimeError('Terminal S/D Adam updates are not complete')
    if pointer!={'path':str(STUDENT),'global_step':STEP,'checkpoint_sha256':STUDENT_SHA,'run_contract_sha256':CONTRACT_SHA}:
        raise RuntimeError('Actual rolling full pointer differs')
    return replace(old.PAPER_STUDENT_CONFIG,num_workers=0)


def validate_adam(saved,state,count,lr):
    # 此网络的持久state仅包含参数；严格核71/144张量及每个相应moment，不构建D、不作更新。
    if not isinstance(state,dict) or len(state)!=count:raise RuntimeError('Unexpected full model parameter count')
    groups=saved['param_groups']
    if len(groups)!=1:raise RuntimeError('Require one original Adam parameter group')
    group=groups[0];identifiers=group['params']
    expected={'lr':lr,'betas':(.5,.9),'eps':1e-8,'weight_decay':0,'amsgrad':False}
    if any(group.get(k)!=v for k,v in expected.items()) or any(group.get(k,False) for k in ('maximize','capturable','differentiable','fused')):
        raise RuntimeError('Actual Adam hyperparameters differ')
    if len(identifiers)!=count or len(set(identifiers))!=count or set(saved['state'])!=set(identifiers):raise RuntimeError('Incomplete Adam state')
    for identifier,(name,parameter) in zip(identifiers,state.items()):
        if not isinstance(parameter,torch.Tensor) or not bool(torch.isfinite(parameter).all()):raise RuntimeError('Nonfinite full model: '+name)
        moment=saved['state'][identifier]
        if set(moment)!={'step','exp_avg','exp_avg_sq'} or float(moment['step'])!=STEP:raise RuntimeError('Adam cursor differs from122900')
        for key in ('exp_avg','exp_avg_sq'):
            value=moment[key]
            if not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=parameter.dtype or not bool(torch.isfinite(value).all()):
                raise RuntimeError('Adam tensor differs: '+name+'/'+key)
        if bool((moment['exp_avg_sq']<0).any()):raise RuntimeError('Negative Adam second moment')


def validate_payload(payload,contract,completion):
    config=validate_end(contract,completion,read_json(POINTER))
    fields={'kind':'acdd_mc1_longtrain_from3327_full_v1','epoch':0,'global_step':STEP,'completed_passes':PASSES,
        'run_contract_sha256':CONTRACT_SHA,'mc_samples':1,'effective_optimizer_lrs':LRS,'origin_parent_sha256':PARENT_SHA,
        'partial_batch':False,'resumable':True,'score_semantics':'reconstruction_clean_x0_sqrt_alpha_bar'}
    if any(payload.get(k)!=v for k,v in fields.items()):raise RuntimeError('Wrong actual complete122900 payload')
    if payload.get('configuration')!=contract['configuration'] or config.as_dict()!=payload['configuration']:
        raise RuntimeError('Original source normalization/BNF/student inference configuration changed')
    for role,count in (('student',71),('discriminator',144)):
        validate_adam(payload[role+'_optimizer'],payload[role],count,LRS[role])
    if payload.get('actual_optimizer_cursors')!=completion['optimizer_cursors']:raise RuntimeError('Saved optimizer cursor receipt differs')
    if set(payload.get('rng_state',{}))!={'python','numpy','torch','cuda_all'}:raise RuntimeError('Full RNG state missing')
    return config


def load_plan(verify_dependencies=True):
    if verify_dependencies:verify_files(PINNED)
    contract,completion,pointer=read_json(CONTRACT),read_json(COMPLETION),read_json(POINTER)
    config=validate_end(contract,completion,pointer)
    if config.as_dict()!=contract['configuration']:raise RuntimeError('Training/inference configuration differs')
    historical=read_json(old.OLD_IDENTITY);teacher_report=read_json(old.TEACHER_DIR/'inference_report.json')
    sources=teacher_infer.load_test_rows(ROOT/'data/metadata/test_manifest.csv')
    pairs=teacher_infer.build_pairs(sources,OUTPUT);pair_bytes=teacher_infer.render_pairs_csv(pairs);rows=old.csv_rows(pair_bytes)
    old.match_teacher_pairs(rows,old.csv_rows(old.TEACHER_PAIRS.read_bytes()))
    files=dict(PINNED);inputs={}
    for key,sha in historical['test_input_file_sha256'].items():
        path=Path(key);path=(path if path.is_absolute() else ROOT/path).resolve()
        if ROOT/'data' not in path.parents or str(path) in inputs:raise RuntimeError('Duplicated or escaped formal input')
        inputs[str(path)]=sha
    if len(inputs)!=400:raise RuntimeError('Require the original400 formal inputs')
    merge_hashes(files,inputs)
    for key in ('test_manifest','split_protocol','mel_mean','mel_std'):
        merge_hashes(files,{teacher_report[key]:teacher_report[key+'_sha256']})
    files[str(Path(__file__).resolve())]=digest(Path(__file__))
    # 绑定实际已导入的项目源码；从新训练契约只选这些.py，不继承训练权重/数据历史表。
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        if path.suffix!='.py' or ROOT not in path.parents:continue
        # 评分器稍后也会调用此只读接口；调用方TOOLS/evaluation模块不属于推理源码身份。
        if path.relative_to(ROOT).parts[0] not in ('voicegrad','fastvoicegrad','hifigan'):continue
        expected=files.get(str(path),contract['runtime_files'].get(str(path)))
        if expected is None:
            try:expected=hashlib.sha256(subprocess.check_output(['git','show',EXPECTED_HEAD+':'+str(path.relative_to(ROOT))],cwd=ROOT,stderr=subprocess.DEVNULL)).hexdigest()
            except subprocess.CalledProcessError:raise RuntimeError('Unpinned imported inference source: '+str(path))
        merge_hashes(files,{str(path):expected})
    if verify_dependencies:
        verify_files(files)
        if formal.collect_test_input_hashes(ROOT,sources)!=historical['test_input_file_sha256']:raise RuntimeError('Original formal input collection differs')
    lengths={}
    for row in sources:
        shape=np.load(teacher_infer.feature_path(ROOT/'data','mel',row['speaker'],row['utt_id']),mmap_mode='r',allow_pickle=False).shape
        if len(shape)!=2 or 80 not in shape:raise RuntimeError('Invalid source mel shape')
        lengths[row['utt_id']]=(shape[1] if shape[0]==80 else shape[0])*256
    samples={r['pair_id']:lengths[r['source_utt']] for r in rows};unit=os.statvfs(OUTPUT.parent).f_frsize
    sizes={k:44+2*v for k,v in samples.items()};allocations={k:old.allocated_size(v,unit) for k,v in sizes.items()}
    if sum(sizes.values())!=old.EXPECTED_WAV_BYTES or (unit==4096 and sum(allocations.values())!=old.EXPECTED_WAV_ALLOCATED):
        raise RuntimeError('Original complete PCM16 WAV size plan changed')
    chunks=list(teacher_infer.fixed_chunks(pairs,BATCH_SIZE))
    if len(chunks)!=2100:raise RuntimeError('Original source-local grouping changed')
    return {'contract':contract,'completion':completion,'pairs':pairs,'rows':rows,'pair_bytes':pair_bytes,'chunks':chunks,
        'file_sha256':files,'test_input_file_sha256':inputs,'expected_samples':samples,'expected_wav_bytes':sizes,
        'expected_allocations':allocations,'allocation_unit':unit,'actual_max_wav_allocation':max(allocations.values()),
        'max_temporary_wav_allocation':max(old.TEMPORARY_BUDGET,max(allocations.values()))}


def make_identity(plan):
    # 显示真实epoch0存储字段，并另记已经完成100个训练pass，不假冒旧epoch100格式。
    return {'identity_format_version':1,'system_id':SYSTEM,'system_label':SYSTEM_LABEL,'model_role':'student','output_dir':str(OUTPUT),
        'student_epoch':0,'student_epoch_semantics':'Stored epoch0; actual complete global cursor122900 represents100 passes of1229 batches.',
        'student_global_step':STEP,'completed_passes':PASSES,'teacher_epoch':500,'student_steps':1,'student_start_step':950,
        'completed_paper_training':False,'student_checkpoint':str(STUDENT),'student_checkpoint_sha256':STUDENT_SHA,
        'parent_full_checkpoint':str(STUDENT),'parent_full_checkpoint_sha256':STUDENT_SHA,
        'origin_parent_sha256':PARENT_SHA,'student_run_contract':str(CONTRACT),'student_run_contract_sha256':CONTRACT_SHA,
        'student_run_contract_file_sha256':CONTRACT_FILE_SHA,'effective_optimizer_lrs':LRS,'configuration':plan['contract']['configuration'],
        'vocoder_checkpoint':str(old.VOCODER),'vocoder_checkpoint_sha256':old.VOCODER_SHA,'vocoder_kind':'custom','vocoder_global_step':2500000,
        'data_protocol':'text-unseen','paper_pair_count':PAIR_COUNT,'selected_pair_count':PAIR_COUNT,'teacher_pairs_sha256':old.TEACHER_PAIRS_SHA,
        'pairs_csv_sha256':hashlib.sha256(plan['pair_bytes']).hexdigest(),'file_sha256':plan['file_sha256'],'test_input_file_sha256':plan['test_input_file_sha256'],
        'batch_size':BATCH_SIZE,'base_seed':SEED,'seed_rule':'base_seed + first_pair_id_of_fixed_source-local_chunk',
        'noise_recipe':'unchanged original full source-local CUDA Generator shape/order; no DEV audio',
        'bnf_alignment':'legacy_length_linear','bnf_downsampling':'legacy_decimate',
        'waveform_format':{'samplerate':22050,'channels':1,'subtype':'PCM_16','sample_count':'source_mel_T*256'},
        'expected_samples_per_pair':plan['expected_samples'],'expected_logical_wav_bytes':sum(plan['expected_wav_bytes'].values()),
        'expected_allocated_wav_bytes':sum(plan['expected_allocations'].values()),'allocation_unit':plan['allocation_unit'],
        'temporary_wav_budget':plan['max_temporary_wav_allocation'],'runtime':EXPECTED_RUNTIME,'git_commit':EXPECTED_HEAD,
        'inference_driver_sha256':digest(Path(__file__)),'original_inference_source_sha256':OLD_SHA,
        'storage':{'reserve_bytes':RESERVE,'metadata_budget_bytes':METADATA_BUDGET,'directory_budget_bytes':DIRECTORY_BUDGET,
            'reason':'Training is complete; no concurrent rolling-full save. New explicit evaluation-only reserve1GiB+128MiB, unchanged whole8100 WAV quality.'},
        'limitations':['The user selected3327 for direct training; this endpoint is evaluated once without DEV checkpoint screening.',
            'Actual student LR5e-5 and52 excluded TRAIN utterances differ from exact paper training;100 passes are complete.',
            'DNSMOS version correspondence to the paper remains unverified; keep P808/SIG/BAK/OVRL separately.',
            'Historical teacher does not have a complete per-pair noise SHA receipt; its frozen pairs and scoring records are retained.']}


def required_free(remaining,used=0,directories=0,temporary=425984):
    if min(remaining,used,directories,temporary)<0:raise ValueError('Negative disk accounting')
    if used>METADATA_BUDGET or directories>DIRECTORY_BUDGET:raise RuntimeError('Metadata allocation exhausted')
    return RESERVE+remaining+METADATA_BUDGET-used+DIRECTORY_BUDGET-directories+temporary


def capacity(output=OUTPUT,remaining=0,temporary=425984):
    used,directories=old.metadata_usage(Path(output));fs=os.statvfs(Path(output).parent);free=fs.f_bavail*fs.f_frsize
    # 原函数计相邻推理.log；本次另把正式评分.metrics.log纳入同一32MiB额度。
    metric_log=Path(output).parent/(Path(output).name+'.metrics.log')
    if metric_log.exists():ensure_regular(metric_log);used+=metric_log.stat().st_blocks*512
    needed=required_free(remaining,used,directories,temporary)
    if free<needed:raise RuntimeError('Insufficient evaluation disk: free=%d required=%d'%(free,needed))
    return {'free_bytes':free,'required_bytes':needed,'remaining_WAV_allocation':remaining,'metadata_allocated':used,'directory_allocated':directories}


def readout_report(plan,identity,records,elapsed,generated,skipped):
    hashes={r['pair_id']:r['wav_sha256'] for r in records}
    return {'status':'formal_8100_complete','system_id':SYSTEM,'system_label':SYSTEM_LABEL,'model_role':'student',
        'student_epoch':0,'student_global_step':STEP,'completed_passes':PASSES,'teacher_epoch':500,'student_steps':1,'student_start_step':950,
        'paper_pair_count':PAIR_COUNT,'selected_pair_count':len(records),'completed_paper_training':False,'data_protocol':'text-unseen',
        'student_checkpoint':str(STUDENT),'student_checkpoint_sha256':STUDENT_SHA,'parent_full_checkpoint_sha256':STUDENT_SHA,
        'student_run_contract_sha256':CONTRACT_SHA,'student_run_contract_file_sha256':CONTRACT_FILE_SHA,
        'vocoder_checkpoint_sha256':old.VOCODER_SHA,'vocoder_kind':'custom','vocoder_global_step':2500000,
        'run_identity_sha256':digest(OUTPUT/'run_identity.json'),'pairs_csv_sha256':identity['pairs_csv_sha256'],
        'inference_driver_sha256':identity['inference_driver_sha256'],'file_sha256':identity['file_sha256'],
        'generated_wav_sha256':hashes,'generated_wav_collection_sha256':training.canonical_json_sha256(hashes),
        'noise_sha256_per_pair':{r['pair_id']:r['source_noise_sha256'] for r in records},'x950_sha256_per_pair':{r['pair_id']:r['x950_sha256'] for r in records},
        'wav_progress_sha256':digest(OUTPUT/'wav_progress.jsonl'),'expected_samples_per_pair':plan['expected_samples'],
        'generated_this_invocation':generated,'skipped_verified_from_previous_invocations':skipped,'elapsed_seconds':elapsed,
        'runtime':identity['runtime'],'batch_size':BATCH_SIZE,'base_seed':SEED,'seed_rule':identity['seed_rule'],
        'model_checkpoint_writes':0,'original_model_parameters_and_buffers_unchanged':True,'limitations':identity['limitations']}


def validate_complete_output(output_dir=OUTPUT,verify_dependencies=True):
    output=Path(output_dir)
    if output.is_symlink() or output.resolve()!=OUTPUT:raise RuntimeError('Require the declared122900 evaluation directory')
    plan=load_plan(verify_dependencies);identity=make_identity(plan)
    for name in ('run_identity.json','pairs.csv','inference_report.json'):ensure_regular(output/name)
    if (output/'run_identity.json').read_bytes()!=json_bytes(identity) or (output/'pairs.csv').read_bytes()!=plan['pair_bytes']:raise RuntimeError('Inference identity changed')
    identity_sha=digest(output/'run_identity.json')
    records,pending,count=scan_journal(output,plan['chunks'],plan['rows'],plan['expected_samples'],identity_sha)
    if pending is not None or count!=2100 or len(records)!=PAIR_COUNT:raise RuntimeError('Incomplete8100 inference')
    report=read_json(output/'inference_report.json')
    expected=readout_report(plan,identity,records,report.get('elapsed_seconds'),report.get('generated_this_invocation'),report.get('skipped_verified_from_previous_invocations'))
    if report!=expected:raise RuntimeError('Complete report differs from actual journal')
    if not isinstance(report['elapsed_seconds'],(int,float)) or not math.isfinite(report['elapsed_seconds']) or not 0<=report['elapsed_seconds']<=10800:raise RuntimeError('Invalid elapsed time')
    if any(type(report[k]) is not int or report[k]<0 for k in ('generated_this_invocation','skipped_verified_from_previous_invocations')) or report['generated_this_invocation']+report['skipped_verified_from_previous_invocations']!=PAIR_COUNT:
        raise RuntimeError('Invalid invocation pair counts')
    return {'rows':plan['rows'],'identity':identity,'report':report,'report_sha256':digest(output/'inference_report.json'),'identity_sha256':identity_sha}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=10800);args=parser.parse_args()
    if not args.preflight and not args.run_approved:parser.error('Require explicit formal inference authorization')
    if not 1<=args.max_seconds<=10800:parser.error('At most10800 seconds')
    if OUTPUT.is_symlink():raise RuntimeError('Output cannot be redirected')
    if args.preflight:
        plan=load_plan();print(json_bytes({'status':'CPU_preflight_no_model','pairs':len(plan['pairs']),'chunks':len(plan['chunks']),
            'input_files':len(plan['test_input_file_sha256']),'dependencies':len(plan['file_sha256']),
            'disk':capacity(OUTPUT,sum(plan['expected_allocations'].values()),plan['max_temporary_wav_allocation']),
            'cuda_initialized':torch.cuda.is_initialized()}).decode(),flush=True);return
    started=time.monotonic();stop={'requested':False};locks=[];handlers={};failure=None;result=None
    try:
        training.assert_rented_server(ROOT);locks=training.acquire_runtime_locks(ROOT)
        import fcntl
        lock_path=ROOT/'exp/locks/text_unseen_fastvoicegrad_student_inference.lock'
        if lock_path.is_symlink():raise RuntimeError('Inference lock cannot be redirected')
        extra=lock_path.open('a+')
        try:fcntl.flock(extra.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:extra.close();raise
        locks.append(extra)
        for signum in (signal.SIGTERM,signal.SIGINT):
            handlers[signum]=signal.getsignal(signum);signal.signal(signum,lambda *_:stop.update(requested=True))
        if formal.git_head(ROOT)!=EXPECTED_HEAD or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise RuntimeError('Frozen production HEAD is not clean')
        if current_runtime()!=EXPECTED_RUNTIME:raise RuntimeError('Original inference runtime changed')
        gpu_idle()
        def check(remaining=0,temporary=425984):
            if stop['requested']:raise RuntimeError('External stop requested; keep verified/partial artifacts')
            if time.monotonic()-started>args.max_seconds:raise TimeoutError('Formal inference exceeded10800s budget')
            return capacity(OUTPUT,remaining,temporary)
        plan=load_plan();identity=make_identity(plan);identity_sha=hashlib.sha256(json_bytes(identity)).hexdigest()
        if (OUTPUT/'inference_report.json').exists():
            validate_complete_output();print('FORMAL122900_ALREADY_COMPLETE_VERIFIED',flush=True);return
        if OUTPUT.exists():
            for name in ('run_identity.json','pairs.csv'):ensure_regular(OUTPUT/name)
            if (OUTPUT/'run_identity.json').read_bytes()!=json_bytes(identity) or (OUTPUT/'pairs.csv').read_bytes()!=plan['pair_bytes']:raise RuntimeError('Existing output identity differs')
            complete,pending,position=scan_journal(OUTPUT,plan['chunks'],plan['rows'],plan['expected_samples'],identity_sha)
        else:complete,pending,position=[],None,0
        present={r['pair_id'] for r in complete}
        if pending:present.update(r['pair_id'] for r in pending['pairs'])
        remaining=sum(v for k,v in plan['expected_allocations'].items() if k not in present)
        check(remaining,plan['max_temporary_wav_allocation'])
        if not OUTPUT.exists():prepare_output(OUTPUT,identity,plan['pair_bytes'])
        if pending:finish_prepared(OUTPUT,pending);complete.extend(pending['pairs']);position+=1
        skipped=len(complete);generated=0
        if position<len(plan['chunks']):
            # 完整payload只反序列化一次；S/D双Adam均核验，只有S进入推理，D和动量随后释放。
            payload=old.load_torch_checkpoint(STUDENT);config=validate_payload(payload,plan['contract'],plan['completion'])
            torch.set_num_threads(4);device=torch.device('cuda')
            model=old.build_model('legacy_decimate');model.load_state_dict(payload['student'],strict=True);del payload;gc.collect()
            model.to(device).eval();old.set_requires_grad(model,False);old.assert_frozen(model,'student122900')
            vocoder,_,_=old.load_frozen_vocoder(ROOT,old.VOCODER,old.VOCODER_DIR/'config.json',old.ATTESTATION,config,device)
            mean,std=old.load_mel_statistics(ROOT/'data',device)
            mean_np=np.load(ROOT/'data/stats/mel_mean.npy',allow_pickle=False).astype(np.float32).reshape(80,1)
            std_np=np.load(ROOT/'data/stats/mel_std.npy',allow_pickle=False).astype(np.float32).reshape(80,1)
            diffusion=old.VoiceGradDiffusion(1000,offset=.008).to(device)
            fingerprints={'student':parameter_fingerprint(model),'vocoder':parameter_fingerprint(vocoder)}
            for chunk in plan['chunks'][position:]:
                check(remaining,plan['max_temporary_wav_allocation'])
                if current_runtime()!=EXPECTED_RUNTIME:raise RuntimeError('Inference precision flags changed')
                source=chunk[0]
                mel=teacher_infer.feature_matrix(teacher_infer.feature_path(ROOT/'data','mel',source.source_speaker,source.source_utt),80)
                bnf=teacher_infer.feature_matrix(teacher_infer.feature_path(ROOT/'data','bnf',source.source_speaker,source.source_utt),144)
                if not np.isfinite(mel).all() or not np.isfinite(bnf).all():raise RuntimeError('Nonfinite source features')
                x=torch.from_numpy(((mel-mean_np)/std_np).astype(np.float32)).unsqueeze(0).to(device)
                condition=old.align_bnf_to_mel(torch.from_numpy(bnf).unsqueeze(0).to(device),x.shape[-1],mode='legacy_length_linear',sample_rate=22050,n_fft=1024,hop_size=256,bnf_frame_shift_seconds=.010)
                x=x.expand(len(chunk),-1,-1).contiguous();condition=condition.expand(len(chunk),-1,-1).contiguous();vectors=[]
                for pair in chunk:
                    vector=np.load(teacher_infer.feature_path(ROOT/'data','spk_emb',pair.reference_speaker,pair.reference_utt),allow_pickle=False).squeeze()
                    if vector.shape!=(256,) or not np.isfinite(vector).all():raise RuntimeError('Invalid target speaker vector')
                    vectors.append(vector.astype(np.float32))
                speaker=torch.from_numpy(np.stack(vectors)).to(device);seed=SEED+int(chunk[0].pair_id)
                generator=torch.Generator(device=device).manual_seed(seed);noise=torch.randn(x.shape,device=device,dtype=x.dtype,generator=generator)
                with torch.inference_mode():
                    converted=formal.student_one_step(model,diffusion,x,speaker,condition,noise)
                    q950=diffusion.q_sample(x,torch.full((len(chunk),),950,device=device,dtype=torch.long),noise=noise)
                    audio=vocoder(converted*std+mean).squeeze(1).cpu().numpy()
                if not bool(torch.isfinite(converted).all()) or not np.isfinite(audio).all():raise RuntimeError('Nonfinite inference output')
                prepared={'type':'prepared','chunk_head':int(chunk[0].pair_id),'run_identity_sha256':identity_sha,'pairs':[]}
                for index,pair in enumerate(chunk):
                    key=str(pair.pair_id);path=OUTPUT/'staging'/Path(pair.output_wav).name
                    if path.exists() or path.with_name(path.stem+'.tmp.wav').exists() or Path(pair.output_wav).exists():raise RuntimeError('Never overwrite/recompute an existing WAV')
                    if audio[index].ndim!=1 or len(audio[index])!=plan['expected_samples'][key]:raise RuntimeError('Unexpected source duration')
                    old.atomic_write_wav(path,np.clip(audio[index].astype(np.float32),-1.,1.),22050)
                    record={'pair_id':key,'output_wav':pair.output_wav,'num_samples':len(audio[index]),'wav_sha256':digest(path),
                        'seed':seed,'chunk_index':index,'chunk_size':len(chunk),'source_noise_sha256':tensor_sha(noise[index]),'x950_sha256':tensor_sha(q950[index])}
                    validate_wave(path,record);prepared['pairs'].append(record);remaining-=plan['expected_allocations'][key]
                    check(remaining,plan['max_temporary_wav_allocation'])
                append_record(OUTPUT/'wav_progress.jsonl',prepared);finish_prepared(OUTPUT,prepared)
                complete.extend(prepared['pairs']);generated+=len(chunk)
                print(old.json.dumps({'stage':'chunk_committed','chunk_head':prepared['chunk_head'],'verified_pairs':len(complete),'remaining_wave_allocation':remaining}),flush=True)
            if fingerprints!={'student':parameter_fingerprint(model),'vocoder':parameter_fingerprint(vocoder)}:raise RuntimeError('Inference changed model parameters/buffers')
            old.assert_frozen(model,'student122900');old.assert_frozen(vocoder,'fixed customG')
        check(0,plan['max_temporary_wav_allocation']);verify_files(plan['file_sha256'])
        records,pending,count=scan_journal(OUTPUT,plan['chunks'],plan['rows'],plan['expected_samples'],identity_sha)
        if pending or count!=2100 or len(records)!=PAIR_COUNT:raise RuntimeError('Incomplete8100 inference')
        if (OUTPUT/'run_identity.json').read_bytes()!=json_bytes(identity) or (OUTPUT/'pairs.csv').read_bytes()!=plan['pair_bytes']:raise RuntimeError('Identity changed during inference')
        result=readout_report(plan,identity,records,time.monotonic()-started,generated,skipped)
        check(0,plan['max_temporary_wav_allocation'])
    except BaseException as error:
        failure=error
        if OUTPUT.exists():
            try:publish_exact(OUTPUT/('failure_%d.json'%os.getpid()),json_bytes({'status':'failed_no_complete_claim','error':repr(error),'elapsed_seconds':time.monotonic()-started}))
            except BaseException:pass
        raise
    finally:
        cleanup=[]
        for signum,handler in handlers.items():
            try:signal.signal(signum,handler)
            except BaseException as error:cleanup.append(repr(error))
        for handle in reversed(locks):
            try:handle.close()
            except BaseException as error:cleanup.append(repr(error))
        if cleanup:
            if OUTPUT.exists():publish_exact(OUTPUT/('cleanup_failure_%d.json'%os.getpid()),json_bytes({'errors':cleanup}))
            if failure is None:raise RuntimeError('Inference cleanup failed: '+repr(cleanup))
    if result is not None:
        publish_exact(OUTPUT/'inference_report.json',json_bytes(result))
        print('FORMAL122900_8100_COMPLETE '+str(OUTPUT/'inference_report.json'),flush=True)


if __name__=='__main__':main()
