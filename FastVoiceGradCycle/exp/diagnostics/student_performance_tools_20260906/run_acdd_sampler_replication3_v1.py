"""固定3327父状态，三组时间步种子各作IID/分层20步；仅记录，不保存新模型。"""

import argparse  # 导入模块不启动实验，真实入口还须显式批准。
import hashlib  # 先核冻结辅助模块，避免悄悄换成别的训练逻辑。
import json
import math
import os
import platform
import random  # 每批使用局部Random，不改变原Python/CUDA随机流。
import statistics  # 20步标准差只作描述，不把相邻更新当独立试验。
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run only on the registered Matpool rental')
TOOLS = ROOT/'exp/diagnostics/student_performance_tools_20260906'
PAIRED_MODULE_SHA = '39520259afddf3207355a0c901d029cec5c026f6ff611a558845b60fed8857f7'
if hashlib.sha256((TOOLS/'run_acdd_timestep_paired60_v1.py').read_bytes()).hexdigest()!=PAIRED_MODULE_SHA:
    raise RuntimeError('Frozen paired60 helper changed')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TOOLS))
import torch
import run_acdd_timestep_paired60_v1 as previous  # 只调用函数，不改其常量或全局函数。

cross, pair, base = previous.cross, previous.pair, previous.base
training, resume, probe = previous.training, previous.resume, previous.probe
DEV, PARENT = previous.DEV, previous.PARENT
START, STOP, UPDATES = 3327, 3347, 20
SEED_BASES = (73100000,73200000,73300000)
ARMS = ('iid_control','stratified_candidate')
LRS = {'student':.00005,'discriminator':.0002}
OUTPUT = DEV/'acdd_sampler_replication3_from3327_v1'
PROTOCOL = DEV/'ACDD3327_SAMPLER_REPLICATION3_PROTOCOL_CN.md'
PROTOCOL_SHA = '433bfcb8008b98ae7377e8acce5f8a8cfcf52f49e8242f9d6f9586e34dab0571'
OLD = previous.OUTPUT
OLD_CANONICAL = '2025948dd40b3339a0c82c7f23637b4e3fbb8179936165e9cbf570b6a4c882b9'
PINNED = {
    str(PROTOCOL):PROTOCOL_SHA,
    str(TOOLS/'run_acdd_timestep_paired60_v1.py'):PAIRED_MODULE_SHA,
    str(OLD/'run_contract.json'):'07bbce5c7f13b8effa9ab605d8c64ee096584e0699decd115466193c2fea2b93',
    str(OLD/'control_iid/step_losses.jsonl'):'cf334964262c6ced8e7a16f235e1f5d83b1feb0c55ff2b502242c107bd129e56',
    str(OLD/'control_iid/monitor.jsonl'):'8d4bf3af234f7719c4d8e76a29dc0bdc4007f96bfc3eaede5c4bf126f8595c7f',
    str(OLD/'control_iid/arm_completion.json'):'f7793e9cef3aa5120708203de733e9f87142548b188acb0735a007ca8170f3b0',
    str(OLD/'completion.json'):'6e30218d743a961a05b0616d811e99812565c9cb33ec1d4776cb5b1378c280f8',
}
META_LIMIT, DIR_RESERVE = 32*1024**2, 1024**2
RESERVE = 3*1024**3+128*1024**2  # 系统保留空间不能用来装实验日志。


def sample_timesteps(seed_base, step, arm):
    # 两臂均替换t：IID也不用旧torch的第三次抽样值，确保比较的是三组新独立流。
    if seed_base not in SEED_BASES or not START<step<=STOP or arm not in ARMS:
        raise RuntimeError('Unregistered seed, arm or step')
    rng=random.Random(seed_base+step)
    bonus=[]
    if arm==ARMS[0]:
        values=[rng.randint(1,1000) for _ in range(32)]
        labels=[(value-1)//100 for value in values]
    else:
        bonus=rng.sample(range(10),2)  # 两个不同区间各多一句，其余每区间三句。
        labels=[layer for layer in range(10) for _ in range(3)]+bonus
        rng.shuffle(labels)
        values=[rng.randint(100*layer+1,100*layer+100) for layer in labels]
    return {'seed_base':seed_base,'seed':seed_base+step,'arm':arm,'timesteps':values,
            'labels':labels,'bonus_bins':bonus,'bin_counts':[labels.count(i) for i in range(10)]}


def read_lines(path):
    # 畸形或空行不能通过悄悄删行来恢复；完整旧日志本来就应能逐行解析。
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def load_old_trial():
    cross.check_files(PINNED)
    contract=cross.read_json(OLD/'run_contract.json')
    completion=cross.read_json(OLD/'completion.json')
    if training.canonical_json_sha256(contract)!=OLD_CANONICAL or completion['contract_sha256']!=OLD_CANONICAL:
        raise RuntimeError('Old paired60 contract or receipt changed')
    rows=read_lines(OLD/'control_iid/step_losses.jsonl')
    monitors=read_lines(OLD/'control_iid/monitor.jsonl')
    if [row['global_step'] for row in rows]!=list(range(3328,3388)) or [row['global_step'] for row in monitors]!=[3327,3347,3367,3387]:
        raise RuntimeError('The original complete control60 identity is required')
    return contract,completion,rows[:UPDATES],monitors[0]


def future_plan(development, old_contract, old_rows):
    # 绝对步数决定原shuffle/crop/noise；不用新的时间步种子改动这些数据安排。
    plan=resume.GlobalBatchPlan(development['retained_train_indices'])
    if plan.batches_per_pass!=1229 or len(development['retained_train_indices'])!=39335:
        raise RuntimeError('Original retained training plan changed')
    rows=[]
    for offset,step in enumerate(range(START+1,STOP+1)):
        item={'global_step':step,'indices':plan.for_step(step),'crop_seed':1234+100000+step,'noise_seed':1234+200000+step}
        if set(item['indices'])&set(development['excluded_train_indices']):
            raise RuntimeError('A training batch intersects the52 development exclusions')
        if any(item[key]!=old_contract['future_batches'][offset][key] for key in item) or item['indices']!=old_rows[offset]['indices']:
            raise RuntimeError('Original first20 data/crop/noise plan did not replay')
        rows.append(item)
    return rows


class MetadataBudget:
    # 只许可本次JSON/JSONL和相邻日志；没有权重预算，也没有权重保存入口。
    def __init__(self,output):
        self.output=Path(output)
        self.log=Path(str(self.output)+'.log')

    def used(self):
        paths=list(self.output.rglob('*')) if self.output.exists() else []
        if self.log.exists() or self.log.is_symlink():paths.append(self.log)
        total=0
        for path in paths:
            cross.regular(path,directory=path.is_dir())
            if path.is_file():
                if '.pt' in path.suffixes:raise RuntimeError('This experiment must never create model checkpoints')
                total+=path.stat().st_size
        return total

    def require(self,extra=0):
        used=self.used()
        if extra<0 or used+extra>META_LIMIT:
            raise pair.InsufficientDisk('Complete metadata exceeds32MiB; retain all observations and abort')
        free=pair.available_bytes(self.output.parent)
        required=RESERVE+DIR_RESERVE+META_LIMIT-used
        if free<required:raise pair.InsufficientDisk('available=%s required=%s'%(free,required))
        return {'available_bytes':free,'required_bytes':required,'metadata_used_bytes':used}

    def write(self,path,value,append=False):
        path=Path(path)
        if (self.output not in path.parents or self.output.resolve() not in path.resolve().parents
                or path.is_symlink() or path.suffix not in ('.json','.jsonl')):
            raise RuntimeError('Only this new trial JSON/JSONL may be written')
        cross.regular(path.parent,directory=True)  # 拒绝父目录中的符号链接，不能借它写到别处。
        data=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
        self.require(len(data)+1)
        if append:
            with path.open('ab') as handle:
                handle.write(data+b'\n');handle.flush();os.fsync(handle.fileno())
        else:
            temporary=path.with_suffix(path.suffix+'.tmp')
            with temporary.open('xb') as handle:
                handle.write(data);handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,path)  # 只更新本次状态，不覆盖旧实验或残留临时文件。
        self.require()


class BoundedLog:
    # 连生产辅助函数的print也计入同一32MiB预算，文件写失败不能伪称结果已完整。
    def __init__(self,handle,budget):
        self.handle,self.budget=handle,budget

    def write(self,text):
        self.budget.require(len(text.encode('utf8')))
        result=self.handle.write(text);self.handle.flush()
        return result

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()


def open_exclusive_log(output,budget):
    # 调用者先持有三把锁；root不得预先把shell输出重定向到这个同名日志。
    output=Path(output);log=Path(str(output)+'.log')
    if any(path.exists() or path.is_symlink() for path in (output,log)):
        raise RuntimeError('Both the output directory and adjacent log must be new')
    budget.require()
    return BoundedLog(log.open('x',encoding='utf8',buffering=1),budget)


def failure_payload(state,summaries,error,parent_restoration_verified=False):
    # 临时参数可能已经变过，但磁盘上最后完整状态始终是原3327。
    return {'status':'execution_aborted','error_type':type(error).__name__,'error':str(error),
            'active_arm':state,'completed_arms':summaries,'last_full_saved_global_step':START,
            'new_checkpoint_created':False,'old_artifacts_not_written_by_this_trial':True,
            'parent_restoration_verified':parent_restoration_verified,'missing_arms_are_not_retried':True,'successful_completion_valid':False}


def new_arm_state(seed_base,arm,contract_sha):
    # 在建目录/优化器和恢复前登记新臂，失败时不借用上一臂的已完成步数。
    return {'seed_base':seed_base,'arm':arm,'status':'restoring','phase':'restoring','completed_global_step':START,'new_updates':0,
        'last_full_saved_global_step':START,'new_checkpoint_created':False,'partial_batch_may_have_updated_D':False,
        'contract_sha256':contract_sha,'restore_completed':False,'effective_optimizer_lrs':dict(LRS),'last_monitor':None}


def write_failure(output,failure,budget):
    # 若成功发布后刚好收到截止信号，先作废本次completion；旧实验完全不写。
    errors=[]
    paths=([output/'completion.json'] if (output/'completion.json').exists() else [])+[output/'failure.json']
    for path in paths:
        try:budget.write(path,failure)
        except BaseException as error:errors.append({'path':str(path),'error':repr(error)})
    return errors  # 极端磁盘故障不能承诺落盘；调用方还保留日志和非零退出。


def describe_window(rows):
    if len(rows)!=UPDATES or [row['global_step'] for row in rows]!=list(range(START+1,STOP+1)):
        raise RuntimeError('A complete20-update window is required')
    record=pair.window_record(rows)
    record['population_standard_deviations']={key:statistics.pstdev([row[key] for row in rows]) for key in pair.LOSS_KEYS}
    record['interpretation']='Descriptive20 successive updates, not20 independent training runs'
    return record


def complete_update(state,step,call):
    # 旧辅助函数最多允许60步；本次在调用前另行锁死20步边界，不修改旧STOP。
    if not START<step<=STOP:
        raise RuntimeError('Replication arms permit only3328 through3347')
    return previous.complete_batch(state,step,call)


def count_changes(candidate,control,expected_count):
    if len(candidate)!=expected_count or len(control)!=expected_count:
        raise RuntimeError('Every fixed utterance must be retained')
    delta=[float(a)-float(b) for a,b in zip(candidate,control)]
    if not all(math.isfinite(value) for value in delta):raise RuntimeError('Nonfinite paired monitor')
    return {'candidate_minus_control':delta,'improved':sum(value<0 for value in delta),
            'worsened':sum(value>0 for value in delta),'equal':sum(value==0 for value in delta)}


def pair_result(seed_base,control,candidate,baseline3327):
    # 控制自己的保护失败也使整对未通过；不能只展示候选的有利结果。
    c,e=control['last_monitor'],candidate['last_monitor']
    reasons=['control:'+reason for reason in control['guard_reasons']]+['candidate:'+reason for reason in candidate['guard_reasons']]
    gate=previous.candidate_gate(e,c,baseline3327,reasons)
    differences={key:float(candidate['window']['means'][key])-float(control['window']['means'][key]) for key in pair.LOSS_KEYS}
    differences.update(DEV32=e['development']['mel_l1']-c['development']['mel_l1'],
        uniform64_score=e['uniform64']['uniform_weighted_score']-c['uniform64']['uniform_weighted_score'],
        TRAIN64_GT=e['uniform64']['GT_l1']-c['uniform64']['GT_l1'])
    return {'seed_base':seed_base,'candidate_gate':gate,'control_guard_reasons':control['guard_reasons'],
        'candidate_guard_reasons':candidate['guard_reasons'],'candidate_minus_control':differences,
        'DEV32':count_changes(e['development']['per_utterance_mel_l1'],c['development']['per_utterance_mel_l1'],32),
        'TRAIN64_GT':count_changes(e['uniform64']['per_utterance_GT_l1'],c['uniform64']['per_utterance_GT_l1'],64),
        'uniform64_score':count_changes([sum(layer[i] for layer in e['uniform64']['per_stratum_per_utterance_weighted_score'])/10 for i in range(64)],
            [sum(layer[i] for layer in c['uniform64']['per_stratum_per_utterance_weighted_score'])/10 for i in range(64)],64),
        'no_checkpoint_promotion':True}


def execute_registered(run_arm,publish_pair):
    # 普通保护失败只是观察值；只有异常会抛出并终止全部后续预登记分支。
    comparisons=[]
    for seed_base in SEED_BASES:
        control=run_arm(seed_base,ARMS[0],None)
        candidate=run_arm(seed_base,ARMS[1],control)
        comparisons.append(publish_pair(seed_base,control,candidate))
    if len(comparisons)!=3:raise RuntimeError('Never select a successful seed and omit the others')
    return comparisons


def main():
    started=time.monotonic()
    locks,clock,budget,log=[],None,None,None
    saved_stdout,saved_stderr=sys.stdout,sys.stderr
    owned=False;state={};summaries={};parent_restoration_verified=False
    try:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--run-approved',action='store_true')
        parser.add_argument('--parent-checkpoint-sha256',required=True)
        args=parser.parse_args()
        if not args.run_approved or args.parent_checkpoint_sha256!=previous.PARENT_SHA:
            raise RuntimeError('Require explicit approval and the exact full3327 parent')
        clock=previous.StopClock(1200,started);clock.install()
        external=previous.watchdog(1200)
        training.assert_rented_server(ROOT);cross.regular(DEV,directory=True)
        locks=training.acquire_runtime_locks(ROOT)
        budget=MetadataBudget(OUTPUT)
        log=open_exclusive_log(OUTPUT,budget)
        sys.stdout=sys.stderr=log  # 文件独占记录；不复制到可能已经断开的stdout管道。
        print('REPLICATION3_PREFLIGHT_STARTED',flush=True)
        if training.git_head(ROOT)!=cross.HEAD or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Require unchanged clean authoritative source')
        if (platform.python_version(),torch.__version__,torch.version.cuda)!=('3.8.15','1.13.1+cu116','11.6') or Path(sys.executable).resolve()!=Path('/root/miniconda3/envs/myconda/bin/python').resolve():
            raise RuntimeError('Use the registered original myconda runtime')
        if cross.flags()!=cross.ORIGINAL_FLAGS:raise RuntimeError('Do not change any original precision flag')
        disk=budget.require();gpu=cross.gpu_gate();clock.active()
        old_contract,old_completion,old_rows,old_initial=load_old_trial()
        files,contracts,receipts,development,unused_rows,config=cross.load_metadata()
        cross.merge_hashes(files,old_contract['file_sha256'])
        cross.merge_hashes(files,PINNED)
        cross.merge_hashes(files,{str(Path(__file__).resolve()):base.sha256_file(Path(__file__))})
        cross.check_files(files)
        contract,parent=contracts[3327],receipts[3327]
        if parent['status']!='stopped':raise RuntimeError('Original3327 must remain stopped')
        anchor=contract['original_step0_guard_anchor'];baseline3227=contract['baseline3227_development']
        baseline3327=parent['last_monitor']['development']
        future=future_plan(development,old_contract,old_rows)
        # 各臂全部20个t在读模型作数值计算前写定，不依据前一组结果选择。
        draws={str(seed):{arm:[sample_timesteps(seed,row['global_step'],arm) for row in future] for arm in ARMS} for seed in SEED_BASES}
        device=torch.device('cuda')
        old,batches,target,monitor_plan=cross.prepare_inputs(files,development,contracts,device)
        reference=cross.read_json(previous.CROSS_RESULT)
        if monitor_plan!=old_contract['monitor64_plan']:raise RuntimeError('Original64 samples/t/noises changed')
        raw=resume.load_torch_checkpoint(cross.INPUT_FILE)
        inputs={key:value.to(device) for key,value in raw.items()};del raw,old
        diffusion=base.VoiceGradDiffusion(config.diffusion_steps,config.cosine_offset).to(device)
        for batch,expected in zip(batches,reference['readouts']['input_identities_after'][1:]):
            t=torch.full((32,),950,device=device,dtype=torch.long)
            batch['q950']=diffusion.q_sample(batch['x0'],t,noise=batch['source_noise'])
            if {key:cross.tensor_id(value) for key,value in batch.items()}!=expected:raise RuntimeError('Fixed64 cached tensor identity changed')
        def input_identity():
            return {'old':{key:probe.tensor_sha(value) for key,value in inputs.items()},
                'new64':[{key:cross.tensor_id(value) for key,value in batch.items()} for batch in batches],'R3107':probe.tensor_sha(target)}
        input_hashes=input_identity()
        if input_hashes!=old_contract['input_identities']:raise RuntimeError('Original fixed32/64/R3107 inputs changed')
        dataset=base.VoiceGradDataset(ROOT/'data','train',segment_length=128,expected_speakers=100,bnf_alignment=config.bnf_alignment)
        if len(dataset)!=39387:raise RuntimeError('Original full training manifest changed')
        future_files=previous.register_future_files(dataset,future,files,clock.active)
        cross.check_files(files);clock.active();budget.require()
        record={'kind':'acdd_sampler_replication3_log_only','protocol_sha256':PROTOCOL_SHA,
            'parent_checkpoint':str(PARENT/'final_full.pt'),'parent_checkpoint_sha256':previous.PARENT_SHA,
            'parent_contract_sha256':previous.PARENT_CANONICAL,'seed':1234,'seed_bases':SEED_BASES,'arm_order':ARMS,
            'configuration':config.as_dict(),'effective_optimizer_lrs':LRS,'maximum_updates_per_arm':UPDATES,'start_after_global_step':START,'stop_global_step':STOP,
            'future_batches':future,'timestep_draws':draws,'future_training_feature_sha256':future_files,
            'training_indices':development['retained_train_indices'],'excluded_train_indices':development['excluded_train_indices'],
            'development_contract_file_sha256':resume.DEVELOPMENT_SHA,'monitor64_plan':monitor_plan,'input_identities':input_hashes,
            'file_sha256':dict(files),'old_paired60_contract_sha256':OLD_CANONICAL,'old_paired60_is_not_one_of_the_three_replications':True,
            'original_step0_guard_anchor':anchor,'baseline3227_development':baseline3227,'baseline3327_development':baseline3327,
            'original_frozen_model_sha256':contract['fixed_frozen_models_sha256'],
            'sampling':'Both arms use local PythonRandom(seed_base+absolute_step) after genuinely consuming all3 original CUDA draws; no cross-batch quota.',
            'unchanged_objective':'clean_x0/sqrt_alpha_bar, adv+2FM+45dist, original D/S order, source950, same teacher/G/BNF/stats',
            'monitor_anchors':{'old_TRAIN':3108,'old_D':3107,'old_fixed_target_R':3107,'uniform64':'same10 strata and64 source utterances'},
            'uniform64_forward_budget':{'student':24,'teacher':240,'monitor_calls':12,'old_monitor_forwards_separate':True},
            'storage':{'model_saves':0,'last_full_saved_global_step':START,'metadata_limit_bytes':META_LIMIT,
                'directory_reserve_bytes':DIR_RESERVE,'system_and_report_reserve_bytes':RESERVE,'initial':disk,'adjacent_log_exclusively_created_by_driver':True},
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'flags':cross.flags(),'gpu':gpu,'external_timeout':external},
            'maximum_compute_seconds':1200,'cleanup_only_seconds':90,'old3327_run_remains_stopped':True,
            'formal_test_used_for_this_selection':False,'new_scoring_or_promotion_authorized':False,
            'limits':['Three t-seed repetitions with fixed data/noise/start, not general significance or CER evidence.',
                'One20-step window per arm does not demonstrate sustained loss decrease.',
                '64 TRAIN sentences repeated at10 timesteps are not640 independent sentences.',
                'Ordinary protective failures remain reported; execution errors abort all remaining arms without retries.']}
        OUTPUT.mkdir();owned=True;budget.write(OUTPUT/'run_contract.json',record)
        contract_sha=training.canonical_json_sha256(record)
        # 先发布不可混淆的契约，再加载真正模型；所有旧文件只读。
        clock.active();payload=resume.load_torch_checkpoint(PARENT/'final_full.pt')
        teacher,student,_=base.load_teacher_and_student(base.TEACHER,config,device)
        vocoder,vconfig,_=base.load_frozen_vocoder(ROOT,base.VOCODER_DIR/'generator_step_2500000.pt',base.VOCODER_DIR/'config.json',
            base.VOCODER_DIR/'final_checkpoint_attestation_comments_20260906.json',config,device)
        discriminator=base.build_paper_discriminator(ROOT,vconfig,config,device)
        fixed_d=base.build_paper_discriminator(ROOT,vconfig,config,device)
        old_d=resume.load_torch_checkpoint(pair.PARENT/'latest_full.pt')
        if old_d['global_step']!=3107 or old_d['run_contract_sha256']!=pair.PARENT_CONTRACT:raise RuntimeError('Old D3107 anchor changed')
        fixed_d.load_state_dict(old_d['discriminator'],strict=True);del old_d
        fixed_d.eval();training.set_requires_grad(fixed_d,False)
        mean,std=base.load_mel_statistics(ROOT/'data',device)
        frozen={'teacher':teacher,'vocoder':vocoder,'fixed_D3107':fixed_d}
        frozen_sha={name:pair.model_sha(model) for name,model in frozen.items()}
        if frozen_sha!=contract['fixed_frozen_models_sha256']:raise RuntimeError('Frozen teacher/G/D3107 identity changed')
        all_models=dict(frozen,student=student,active_D=discriminator,diffusion=diffusion)
        monitor_count={'calls':0,'student':0,'teacher':0}
        restored_reference=old_completion['arms']['control_iid']['restore_identity']
        optimizers=[None,None]  # 只保留当前臂的优化器；结束时再冷恢复父状态作核验。

        def check_unchanged():
            pair.assert_frozen(frozen,frozen_sha)
            if input_identity()!=input_hashes or cross.flags()!=cross.ORIGINAL_FLAGS:
                raise RuntimeError('Fixed inputs or precision changed')

        def evaluate(step):
            clock.active();budget.require()
            with previous.monitoring_state(all_models,device):
                entry=pair.monitor(student,teacher,vocoder,fixed_d,diffusion,inputs,target,mean,std,config,device,step)
                entry['legacy_monitor_updates_from3107']=entry['updates'];entry['updates']=step-START;entry['trial_updates']=step-START
                entry['uniform64']=previous.uniform64_monitor(student,teacher,diffusion,batches,monitor_plan,config,device,clock.active)
            monitor_count['calls']+=1
            for key in ('student','teacher'):monitor_count[key]+=entry['uniform64']['actual_forward_counts'][key]
            check_unchanged();clock.active();budget.require()
            return entry

        def run_arm(seed_base,arm,control):
            nonlocal state
            clock.active();budget.require()
            state=new_arm_state(seed_base,arm,contract_sha)
            directory=OUTPUT/(str(seed_base)+'_'+arm);directory.mkdir()
            optimizer=torch.optim.Adam(student.parameters(),lr=LRS['student'],betas=(.5,.9))
            d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=LRS['discriminator'],betas=(.5,.9))
            optimizers[:]=[optimizer,d_optimizer]
            restored=previous.restore_parent(payload,student,discriminator,optimizer,d_optimizer,config,contract,parent,device)
            if restored!=restored_reference:raise RuntimeError('Arm did not cold-restore the true3327 S/D/twoAdam/RNG/flags')
            state.update(status='running',phase='initial_monitor',restore_identity=restored,restore_completed=True)
            initial=evaluate(START)
            previous.prior.validate_replay(initial,parent['last_monitor']);previous.validate_uniform_initial(initial['uniform64'],reference)
            if initial!=old_initial:raise RuntimeError('All original3327 monitoring values must replay identically')
            if previous.stop_rules([], [initial],anchor,baseline3227,baseline3327)['stop_reasons']:
                raise RuntimeError('Parent itself violates the unchanged hard guards; do not start an arm')
            budget.write(directory/'monitor.jsonl',initial,append=True)
            rows=[];state['last_monitor']=initial
            state['phase']='twenty_updates'
            for offset,planned in enumerate(future):
                step=planned['global_step'];clock.active();budget.require()
                batch=base.batch_for(dataset,planned['indices'],planned['crop_seed'])
                draw=draws[str(seed_base)][arm][offset]
                receipt={};override=torch.tensor(draw['timesteps'],dtype=torch.long)
                training.seed_everything(planned['noise_seed'])
                losses=complete_update(state,step,lambda:previous.sampler.train_batch_with_timesteps(
                    batch,teacher,student,vocoder,discriminator,diffusion,optimizer,d_optimizer,mean,std,device,config,
                    score_timesteps_override=override,sampler_receipt=receipt))
                row={'global_step':step,'updates':step-START,'trial_updates':step-START,'indices':planned['indices'],
                    'seed_base':seed_base,'timestep_seed':draw['seed'],'arm':arm,'sampler':receipt,
                    'batch_tensors':{key:cross.tensor_id(batch[key]) for key in ('mel','bnf','spk_emb')},
                    'actual_optimizer_lrs':{'student':optimizer.param_groups[0]['lr'],'discriminator':d_optimizer.param_groups[0]['lr']},
                    'actual_adam_counters':previous.adam_counters(optimizer,d_optimizer,step),**losses}
                if receipt['used_values']!=draw['timesteps'] or receipt['used_bin_counts']!=draw['bin_counts'] or row['actual_optimizer_lrs']!=LRS:
                    raise RuntimeError('Actual timestep plan or role learning rate differs from the contract')
                row['old_control_input_and_first_GAN_differences']=previous.common_input_check(row,old_rows[offset])
                if control is not None:row['same_pair_input_and_first_GAN_differences']=previous.common_input_check(row,control['rows'][offset])
                if step==START+1:row['postupdate_state_sha256']={'student':pair.model_sha(student),'discriminator':pair.model_sha(discriminator)}
                budget.write(directory/'step_losses.jsonl',row,append=True);rows.append(row)
                budget.write(directory/'status.json',state)
                print('REPLICATION3_STEP '+json.dumps({'seed_base':seed_base,'arm':arm,'step':step,'total':row['student_total']}),flush=True)
            window=describe_window(rows);budget.write(directory/'loss_windows.jsonl',window,append=True)
            state['phase']='endpoint_monitor'
            endpoint=evaluate(STOP);state['last_monitor']=endpoint
            budget.write(directory/'monitor.jsonl',endpoint,append=True)
            guards=previous.stop_rules([window],[initial,endpoint],anchor,baseline3227,baseline3327)
            clock.active();budget.require()  # 执行停止优先；不能把磁盘/截止失败仅写成普通保护失败。
            previous.pair.validate_effective_adam(optimizer,d_optimizer,student,discriminator,LRS['student'],STOP)
            check_unchanged()
            state.update(status='guard_failed' if guards['stop_reasons'] else 'completed20',guard_reasons=guards['stop_reasons'],
                window=window,endpoint_state_sha256=previous.content_state(student,discriminator,optimizer,d_optimizer),
                phase='arm_complete',no_promotion=True,state_hashes_are_not_recoverable_checkpoints=True)
            budget.write(directory/'arm_completion.json',state)
            summaries[str(seed_base)+'_'+arm]=dict(state)
            return dict(state,rows=rows)  # 仅内存行用于配对，不在大收据重复整个逐步日志。

        def publish_pair(seed_base,control,candidate):
            clock.active();budget.require()
            result=pair_result(seed_base,control,candidate,baseline3327)
            budget.write(OUTPUT/(str(seed_base)+'_comparison.json'),result)
            print('REPLICATION3_PAIR '+json.dumps({'seed_base':seed_base,'gate':result['candidate_gate'],'delta':result['candidate_minus_control']}),flush=True)
            return result

        comparisons=execute_registered(run_arm,publish_pair)
        if monitor_count!={'calls':12,'student':24,'teacher':240}:raise RuntimeError('Incomplete registered12 fixed64 monitors')
        clock.active();budget.require();check_unchanged()
        cross.check_files(files);cross.check_files(PINNED)
        if training.canonical_json_sha256(cross.read_json(OUTPUT/'run_contract.json'))!=contract_sha:
            raise RuntimeError('Published contract changed')
        # 六个终点全部只是临时内存状态；最后真正恢复父状态，不调用任何权重保存函数。
        clock.active()
        restored_final=previous.restore_parent(payload,student,discriminator,*optimizers,config,contract,parent,device)
        if restored_final!=restored_reference:raise RuntimeError('Final true parent restoration failed')
        parent_restoration_verified=True  # 最后恢复已实核，即使随后JSON/截止失败也不应否认这个事实。
        check_unchanged();clock.active();budget.require()
        directions={key:{'negative':sum(item['candidate_minus_control'][key]<0 for item in comparisons),
                         'positive':sum(item['candidate_minus_control'][key]>0 for item in comparisons),
                         'zero':sum(item['candidate_minus_control'][key]==0 for item in comparisons)} for key in comparisons[0]['candidate_minus_control']}
        completion={'status':'all_three_registered_pairs_completed','contract_sha256':contract_sha,'completed_arms':summaries,
            'comparisons':comparisons,'three_seed_direction_counts':directions,'uniform64_forward_counts':monitor_count,
            'final_parent_restore_identity':restored_final,'parent_restoration_verified':True,'last_full_saved_global_step':START,
            'new_checkpoint_created':False,'promoted_model':None,'old3327_run_remains_stopped':True,
            'elapsed_seconds':time.monotonic()-started,'no_metric_evaluation_performed':True,
            'interpretation':'All predeclared pairs reported, including protective failures; no independent performance or sustained-loss claim.'}
        budget.write(OUTPUT/'completion.json',completion)
        clock.active()  # 截止在发布边界到达时仍产生失败记录，不能隐去超时事实。
        print('REPLICATION3_FINISHED '+json.dumps({'pairs':3,'arms':6,'temporary_updates':120,'new_checkpoints':0}),flush=True)
    except BaseException as error:
        failure=failure_payload(state,summaries,error,parent_restoration_verified)
        try:print('REPLICATION3_FAILED '+json.dumps(failure,ensure_ascii=False),flush=True)
        except BaseException:pass  # 日志盘满不能替换最初的异常或伪称收据成功。
        if owned:
            try:
                if clock is not None:clock.saving()
                errors=write_failure(OUTPUT,failure,budget)
                if errors:print('FAILURE_RECEIPT_NOT_SAVED '+json.dumps(errors),flush=True)
            except BaseException as secondary:
                try:print('FAILURE_RECEIPT_NOT_SAVED '+repr(secondary),flush=True)
                except BaseException:pass
        raise
    finally:
        if clock is not None:
            try:clock.close()
            except BaseException:pass  # 信号清理异常也不能跳过日志和其余锁的关闭。
        sys.stdout,sys.stderr=saved_stdout,saved_stderr
        # 日志先flush/关闭，之后才释放三把工作流锁；失败仍逐一尝试关闭。
        if log is not None:
            try:log.flush();log.close()
            except BaseException:pass
        for handle in reversed(locks):
            try:handle.close()
            except BaseException:pass


if __name__=='__main__':
    main()  # 仅主线程全文审查、CPU测试和实际资源门禁通过后另行启动。
