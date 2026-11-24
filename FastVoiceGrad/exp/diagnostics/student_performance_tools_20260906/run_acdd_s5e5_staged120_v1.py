# 从原3227完整状态只调整学生学习率；前20步验收通过后，最多再继续100步。
"""Staged20-plus100 development training with student5e-5 and unchangedD2e-4."""

import argparse  # 必须显式确认父断点和运行授权。
import copy  # 只在独立CPU副本校验学习率变更，原moment保持不动。
import hashlib  # 冻结辅助代码通过SHA校验后再导入。
import json
import math  # 非有限值不能参与验收或继续训练。
import os  # GPU查询时识别本进程，不结束其他任务。
import platform
import signal  # 用户停止在完整D/S批次边界处理。
import subprocess  # 仅查询GPU占用，避免与其他训练或评估并行。
import sys
import time  # 总120步只是上限，600秒或任一保护触发就停止。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
PRIOR_FILE = TOOLS / 'run_acdd_half_lr_continue480_v1.py'
PRIOR_SHA = 'f7678ad423e352a3979af5cb7f43057544e237d0c089e22b754757d569ca18ba'
if hashlib.sha256(PRIOR_FILE.read_bytes()).hexdigest() != PRIOR_SHA:
    raise RuntimeError('Frozen continuation helper changed')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(TOOLS))
import torch
import run_acdd_half_lr_continue480_v1 as prior

pair,base,training,resume,probe = prior.pair,prior.base,prior.training,prior.resume,prior.probe
DEV,PAIRED,PARENT = prior.DEV,prior.PAIRED,prior.PARENT
PARENT_SHA,PARENT_CONTRACT = prior.PARENT_SHA,prior.PARENT_CONTRACT
START,STAGE1_END,STOP,ORIGIN = 3227,3247,3347,3107  # 不修改任何旧模块全局常量。
ACTUAL_LRS = {'student':.00005,'discriminator':.0002}
CONTROL = DEV / 'acdd_half_lr_continuation480_v1'
CONTROL_GT = .19456790387630463
CONTROL_TOTAL = 6.013833093643188
PROTOCOL = DEV / 'ACDD_S5E5_STAGED120_PROTOCOL_CN.md'
PROTOCOL_SHA = 'b936ccb27b64dc7434afdefe9b45e3aae169b5d28bb3b98682106551c3550628'
CONTROL_FILES = {
    str(CONTROL/'run_contract.json'):'9396b0609d5a77a6203d928339d6acd07ac02e51899e500c126da736e51517e8',
    str(CONTROL/'completion.json'):'0781f8b8928b850215c0b30892e8cedc106db9cc0c331637d487a586460e857b',
    str(CONTROL/'step_losses.jsonl'):'9d08568d18832749b0f38a0d9a0aac92ec07b1c3fddc75cdff8565f748f9c48c',
    str(CONTROL/'loss_windows.jsonl'):'12d26cf5764e25054bdd8a219f0691fa6f84a6c271d2f8c69c777b13ce606e0d',
    str(CONTROL/'monitor.jsonl'):'1747807971077569565e5a820ad72a2276eee5beee80dad6ee9c22d3d0c7d6ad',
}


def restore_then_change_student_lr(payload,student,discriminator,optimizer,d_optimizer,config,device):
    restored = prior.restore_actual_parent(payload,student,discriminator,optimizer,d_optimizer,config,device)
    # 上一步严格恢复原S1e-4/D2e-4；现在唯一允许变动的是S参数组中的lr字段。
    expected = prior.cpu_adam_copy(optimizer.state_dict())
    expected['param_groups'][0]['lr'] = ACTUAL_LRS['student']
    d_sha = probe.tree_sha(d_optimizer.state_dict())
    rng_sha = probe.tree_sha(training.capture_rng_state())
    model_sha = {'student':pair.model_sha(student),'discriminator':pair.model_sha(discriminator)}
    optimizer.param_groups[0]['lr'] = ACTUAL_LRS['student']
    if probe.tree_sha(optimizer.state_dict()) != probe.tree_sha(expected) or probe.tree_sha(d_optimizer.state_dict()) != d_sha:
        raise RuntimeError('Only the studentLR field may differ from the restored Adam state')
    if probe.tree_sha(training.capture_rng_state()) != rng_sha or model_sha != {'student':pair.model_sha(student),'discriminator':pair.model_sha(discriminator)}:
        raise RuntimeError('LR change altered models or RNG')
    pair.validate_effective_adam(optimizer,d_optimizer,student,discriminator,ACTUAL_LRS['student'],START)
    return {'restored_parent':restored,'effective_optimizer_lrs':dict(ACTUAL_LRS),
            'student_adam_after_lr_override_sha256':probe.tree_sha(expected),'discriminator_unchanged_sha256':d_sha,
            'runtime_rng_unchanged_sha256':rng_sha,'models_unchanged_sha256':model_sha}


def first_step_check(control,candidate):
    pair.finite_losses(control)
    pair.finite_losses(candidate)
    differences = {key:abs(float(control[key])-float(candidate[key])) for key in pair.LOSS_KEYS}
    if any(differences[key]>1e-7+1e-6*abs(float(control[key])) for key in pair.LOSS_KEYS):
        raise RuntimeError('First3228 pre-S-update losses differ from the existing1e-4 control: '+json.dumps(differences))
    return differences  # 两次D先行更新相同，改变S学习率不应改变本步更新前的七项读数。


def stage1_gate(window,entry,trend,parent_gt,external_stop_reasons=()):
    if window['global_step'] != STAGE1_END or window['first_global_step'] != START+1 or window['count'] != 20 or entry['global_step'] != STAGE1_END:
        raise RuntimeError('Stage1 acceptance requires the exact first20 completed updates')
    gt,total = entry['development']['mel_l1'],window['means']['student_total']
    if not all(math.isfinite(value) for value in (gt,total,parent_gt)):
        raise RuntimeError('Stage1 cannot accept nonfinite losses')
    checks = {'GT_not_worse_than_parent3227':gt<=parent_gt,'GT_not_worse_than_existing_control3247':gt<=CONTROL_GT,
              'total_not_worse_than_existing_control20':total<=CONTROL_TOTAL,'original_strict_guards_clear':not trend['stop_reasons'],
              'execution_guards_clear':not external_stop_reasons}
    return {'global_step':STAGE1_END,'passed':all(checks.values()),'checks':checks,'candidate_GT':gt,'candidate20_total':total,
            'parent3227_GT':parent_gt,'control3247_GT':CONTROL_GT,'control20_total':CONTROL_TOTAL,
            'failed_conditions':[name for name,passed in checks.items() if not passed],'external_stop_reasons':list(external_stop_reasons)}


def require_stage_permission(step,gate):
    if not START < step <= STOP:
        raise RuntimeError('Only the predeclared3228-through3347 range is allowed')
    if step>STAGE1_END and (gate is None or gate.get('passed') is not True):
        raise RuntimeError('Do not enter the unpaired extension before passing all20-step conditions')


def load_control():
    read = lambda name:[json.loads(line) for line in (CONTROL/name).read_text().splitlines()]
    rows,windows,monitors = read('step_losses.jsonl'),read('loss_windows.jsonl'),read('monitor.jsonl')
    receipt = json.loads((CONTROL/'completion.json').read_text())
    if [row['global_step'] for row in rows] != list(range(START+1,STAGE1_END+1)) or receipt['completed_global_step'] != STAGE1_END:
        raise RuntimeError('Existingcontrol must contain exactly20 new updates and remain stopped')
    if receipt['status'] != 'stopped' or receipt['new_updates'] != 20 or receipt['parent_checkpoint_sha256'] != PARENT_SHA:
        raise RuntimeError('Existingcontrol identity or stopped status changed')
    if pair.window_record(rows) != windows[-1] or windows[-1]['means']['student_total'] != CONTROL_TOTAL or monitors[-1] != receipt['last_monitor'] or monitors[-1]['development']['mel_l1'] != CONTROL_GT:
        raise RuntimeError('Control thresholds do not match the actual frozen20-step experiment')
    return rows,receipt


def assert_gpu_idle():
    command=['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits']
    rows=subprocess.run(command,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    if any(line.split(',')[0].strip()!=str(os.getpid()) for line in rows.splitlines() if line.strip()):
        raise RuntimeError('AnotherGPU task is active: '+rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--parent-checkpoint-sha256',required=True)
    parser.add_argument('--max-seconds',type=int,default=600)
    parser.add_argument('--run-approved',action='store_true')
    args=parser.parse_args()
    if not args.run_approved or args.parent_checkpoint_sha256!=PARENT_SHA or not 1<=args.max_seconds<=600:
        raise RuntimeError('Explicit reviewed3227 parent and at most600 seconds required')
    training.assert_rented_server(ROOT)
    locks=training.acquire_runtime_locks(ROOT)
    if training.git_head(ROOT)!=base.EXPECTED_HEAD:
        raise RuntimeError('Frozen clean production source changed')
    if (platform.python_version(),torch.__version__,torch.version.cuda)!=('3.8.15','1.13.1+cu116','11.6'):
        raise RuntimeError('Original training runtime required')
    if Path(sys.executable).resolve()!=Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Registered myconda CUDA interpreter required')
    output=args.output_dir.resolve()
    if output.exists() or output!=DEV/'acdd_s5e5_staged120_from3227_v1':
        raise RuntimeError('Use the exact new output directory declared in the protocol')
    gpu_before=assert_gpu_idle()
    hashes={str(PRIOR_FILE):PRIOR_SHA,str(PROTOCOL):PROTOCOL_SHA,str(PARENT/'final_full.pt'):PARENT_SHA,
        str(PAIRED/'run_contract.json'):'f60a052e712080d48207be35a9595804a579ac152acdbb8c9b450108a8fde303',
        str(PARENT/'arm_completion.json'):'50eee0fadb5b25de06e06beec23fa2e4b4f6e9aa1ab38f6aee1128d17ce17dcb',
        str(PARENT/'step_losses.jsonl'):'01cbc2832a7efa32acf9d153a2bb7372054e6c67dd64e5010a7f3684f42c524a',
        str(PARENT/'loss_windows.jsonl'):prior.WINDOW_SHA,str(PARENT/'monitor.jsonl'):prior.MONITOR_SHA,
        str(PAIRED/'fixed_teacher_target.pt'):prior.TARGET_SHA,
        str(PARENT/'student_step_003227.pt'):'88da1f14234c9dcb164cdd249d47862b9ba105f1b7a1a858b4a18f9fa921b1dc',**CONTROL_FILES}
    resume.verify_files(hashes)
    contract=json.loads((PAIRED/'run_contract.json').read_text())
    parent=json.loads((PARENT/'arm_completion.json').read_text())
    if training.canonical_json_sha256(contract)!=PARENT_CONTRACT or parent['completed_global_step']!=START or parent['full_snapshot']['sha256']!=PARENT_SHA:
        raise RuntimeError('Parent contract or completed endpoint changed')
    hashes.update(contract['file_sha256'])
    development,dev_hashes=resume.load_development(DEV/'development_contract.json')
    hashes.update(dev_hashes)
    if development['retained_train_indices']!=contract['training_indices'] or development['excluded_train_indices']!=contract['excluded_train_indices']:
        raise RuntimeError('The52 excluded utterances or retained training indices changed')
    config=base.replace(base.PAPER_STUDENT_CONFIG,num_workers=0)
    if config.as_dict()!=contract['configuration'] or contract['seed']!=1234:
        raise RuntimeError('Base math/BNF/data configuration changed')
    grand=json.loads((resume.PARENT/'run_contract.json').read_text())
    if training.collect_source_fingerprints(ROOT)!=grand['source_sha256']:
        raise RuntimeError('Production source fingerprints changed')
    hashes[str(Path(__file__).resolve())]=base.sha256_file(Path(__file__))
    resume.verify_files(hashes)
    windows,monitors=prior.load_history(parent)  # 仅继承到3227，不把旧control3247的失败窗口加入候选。
    controls,control_receipt=load_control()
    anchor,baseline3227=contract['original_step0_guard_anchor'],parent['last_monitor']['development']
    if anchor['mel_l1']!=.19941353797912598 or anchor['prediction_rms']!=.9994737505912781:
        raise RuntimeError('OriginalS0 protection cannot be rebased')
    if prior.trend_state(windows,monitors,anchor,baseline3227)['stop_reasons']:
        raise RuntimeError('Inherited3227 endpoint already violates the strict stop rules')
    budget=prior.FinalBudget(DEV,(PARENT/'final_full.pt').stat().st_size,(PARENT/'student_step_003227.pt').stat().st_size)
    capacity=budget.require()
    torch.set_num_threads(2)
    training.seed_everything(1234)
    device=torch.device('cuda')
    started=time.monotonic()
    stop={'requested':False}
    def stop_at_boundary(signum,frame):stop['requested']=True
    signal.signal(signal.SIGTERM,stop_at_boundary)
    signal.signal(signal.SIGINT,stop_at_boundary)
    payload=resume.load_torch_checkpoint(PARENT/'final_full.pt')
    if payload['development']!=baseline3227:
        raise RuntimeError('Complete parent does not contain the recorded3227 development endpoint')
    dataset=base.VoiceGradDataset(ROOT/'data','train',segment_length=128,expected_speakers=100,bnf_alignment=config.bnf_alignment)
    if len(dataset)!=39387:
        raise RuntimeError('Original training manifest length changed')
    plan=resume.GlobalBatchPlan(contract['training_indices'])
    if plan.batches_per_pass!=1229 or any(plan.for_step(row['global_step'])!=row['indices'] for row in controls):
        raise RuntimeError('First20 candidate batches do not match the existingcontrol plan')
    raw=resume.load_torch_checkpoint(pair.INPUT_DIR/'fixed_inputs.pt')
    if {key:probe.tensor_sha(value) for key,value in raw.items()}!=contract['input_sha256']:
        raise RuntimeError('OriginalTRAIN3108/fixed32 input tensors changed')
    inputs={key:value.to(device) for key,value in raw.items()}
    del raw
    old_target=resume.load_torch_checkpoint(PAIRED/'fixed_teacher_target.pt')
    target=old_target['teacher_target'].to(device)
    if old_target['global_source_step']!=3107 or probe.tensor_sha(target)!=contract['fixed_teacher_target_sha256'] or old_target['sha256']!=contract['fixed_teacher_target_sha256']:
        raise RuntimeError('ComparableA must remain the oldR3107 target')
    del old_target
    teacher,student,_=base.load_teacher_and_student(base.TEACHER,config,device)
    vocoder,vconfig,_=base.load_frozen_vocoder(ROOT,base.VOCODER_DIR/'generator_step_2500000.pt',base.VOCODER_DIR/'config.json',
        base.VOCODER_DIR/'final_checkpoint_attestation_comments_20260906.json',config,device)
    discriminator=base.build_paper_discriminator(ROOT,vconfig,config,device)
    fixed_d=base.build_paper_discriminator(ROOT,vconfig,config,device)
    original_d=resume.load_torch_checkpoint(pair.PARENT/'latest_full.pt')
    if original_d['global_step']!=3107 or original_d['run_contract_sha256']!=pair.PARENT_CONTRACT:
        raise RuntimeError('Fixed monitorD must be original3107')
    fixed_d.load_state_dict(original_d['discriminator'],strict=True)
    del original_d
    fixed_d.eval();training.set_requires_grad(fixed_d,False)
    diffusion=base.VoiceGradDiffusion(config.diffusion_steps,config.cosine_offset).to(device)
    mean,std=base.load_mel_statistics(ROOT/'data',device)
    frozen={'teacher':teacher,'vocoder':vocoder,'fixed_D3107':fixed_d}
    fingerprints={name:pair.model_sha(model) for name,model in frozen.items()}
    if fingerprints!=contract['fixed_frozen_models_sha256']:
        raise RuntimeError('Frozen teacher/G/D3107 parameters or buffers changed')
    pair.assert_frozen(frozen,fingerprints)
    optimizer=torch.optim.Adam(student.parameters(),lr=.0001,betas=(.5,.9))
    d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=.0002,betas=(.5,.9))
    restored=restore_then_change_student_lr(payload,student,discriminator,optimizer,d_optimizer,config,device)
    del payload
    def evaluate(step):
        return prior.monitored(lambda:pair.monitor(student,teacher,vocoder,fixed_d,diffusion,inputs,target,mean,std,config,device,step),frozen,fingerprints,device)
    replay=evaluate(START)
    prior.validate_replay(replay,monitors[-1])  # 不复制一次3227点进入历史，也不把R3227新诊断混进A。
    budget.require()
    lineage={'parent_checkpoint':str(PARENT/'final_full.pt'),'parent_checkpoint_sha256':PARENT_SHA,'parent_contract_sha256':PARENT_CONTRACT,
             'start_after_global_step':START,'original_step0_checkpoint_sha256':resume.ANCHOR_SHA}
    record={'kind':'acdd_s5e5_staged120_development','lineage':lineage,'configuration':config.as_dict(),'effective_optimizer_lrs':ACTUAL_LRS,
        'configuration_interpretation':'Base template2e-4 remains unchanged; actual student5e-5 and D2e-4 optimizer LRs are authoritative.',
        'method_extension':'Only studentLR changes from actual parent1e-4 to5e-5. This is a development extension, not strict2024LR reproduction.',
        'seed':1234,'stop_global_step':STOP,'stage1_end_global_step':STAGE1_END,'maximum_new_updates':120,'max_seconds':args.max_seconds,
        'training_indices':contract['training_indices'],'excluded_train_indices':contract['excluded_train_indices'],
        'reconstruction_dev_indices':contract['reconstruction_dev_indices'],'development_contract_file_sha256':resume.DEVELOPMENT_SHA,
        'formal_test_used_for_selection':False,'automatic_metric_evaluation':False,
        'batch_plan':{'first_step':3228,'batches_per_pass':1229,'shuffle_seed':'1234+floor((step-1)/1229)',
                      'crop_seed':'1234+100000+step','noise_seed':'1234+200000+step'},
        'inherited_history':{'windows_count':6,'windows_sha256':prior.WINDOW_SHA,'monitor_count':7,'monitor_sha256':prior.MONITOR_SHA,'updates_origin_global_step':ORIGIN},
        'baseline3227_development':baseline3227,'baseline3227_replay':replay,'original_step0_guard_anchor':anchor,
        'guard_ratios':{'mel_l1_upper':1.10,'prediction_rms_lower':.80},
        'stop_rules':{'total':'last3 complete20 means strictly increasing, no2percent gate','development':'last3 fixedGT strictly increasing',
                      'single_development':'GT > parent3227*1.01','originalS0':'unchanged','other':['nonfinite','OOM/exception','disk','time','user_stop']},
        'stage1_acceptance':{'parent_GT_upper':baseline3227['mel_l1'],'existing_control_GT_upper':CONTROL_GT,'existing_control20_total_upper':CONTROL_TOTAL,
                             'all_original_strict_guards_must_also_pass':True,'on_failure':'save actual endpoint and stop','on_pass':'at most100 further unpaired updates'},
        'existing_control':{'directory':str(CONTROL),'file_sha256':CONTROL_FILES,'completed_updates':20,'never_resume_control':True,
                            'paired_comparison_only_global_steps':[3228,3247],'after3247_is_unpaired':True,'first_loss_rtol':1e-6,'first_loss_atol':1e-7},
        'fixed_train_inputs_sha256':contract['input_sha256'],'fixed_teacher_target_sha256':contract['fixed_teacher_target_sha256'],
        'fixed_teacher_target_file':str(PAIRED/'fixed_teacher_target.pt'),'fixed_target_anchor_step':3107,'fixed_discriminator_step':3107,
        'fixed_frozen_models_sha256':fingerprints,'fixed_development_inputs':contract['fixed_development_inputs'],
        'fixed_development_source_noise_sha256':contract['fixed_development_source_noise_sha256'],
        'storage':{'policy':'one actual-end full and oneS only; no intermediate checkpoints or new controlweights','initial_budget':capacity,
                   'system_reserve_bytes':pair.SYSTEM_RESERVE,'report_reserve_bytes':pair.REPORT_RESERVE},
        'restore_and_only_student_lr_change':restored,'file_sha256':hashes,
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'executable':str(Path(sys.executable).resolve()),
                   'gpu_name':torch.cuda.get_device_name(0),'initial_GPU_process_query':gpu_before,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
                   'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}}
    output.mkdir()
    base.save_json(output/'run_contract.json',record)
    contract_sha=training.canonical_json_sha256(record)
    for name in ('loss_windows.jsonl','monitor.jsonl'):
        with (output/name).open('xb') as handle:handle.write((PARENT/name).read_bytes())
    state={'status':'running','completed_global_step':START,'new_updates':0,'updates':120,'last_full_saved_global_step':START,
        'full_snapshot':None,'student_snapshot':None,'partial_batch_may_have_updated_D':False,'contract_sha256':contract_sha,
        'parent_checkpoint_sha256':PARENT_SHA,'effective_optimizer_lrs':ACTUAL_LRS,'last_monitor':monitors[-1],
        'stage1_gate':None,'phase':'matched_first20','existing_control_updates':20}
    base.save_json(output/'resume_state.json',dict(state,replay_verified=True,next_global_step=3228))
    pending,reason=[],'maximum120_updates_completed'
    try:
        for step in range(START+1,STOP+1):
            if stop['requested'] or time.monotonic()-started>=args.max_seconds:
                reason='requested_stop' if stop['requested'] else 'time_budget';break
            require_stage_permission(step,state['stage1_gate'])
            try:budget.require()
            except pair.InsufficientDisk:reason='disk_space_guard';break
            indices=plan.for_step(step)
            if set(indices)&set(contract['excluded_train_indices']):
                raise RuntimeError('Training batch overlaps52 excluded utterances')
            if step<=STAGE1_END and indices!=controls[step-START-1]['indices']:
                raise RuntimeError('Candidate first20 data order differs from existingcontrol')
            batch=base.batch_for(dataset,indices,1234+100000+step)
            training.seed_everything(1234+200000+step)
            losses=prior.complete_batch(state,step,lambda:training.train_batch(batch,teacher,student,vocoder,discriminator,
                diffusion,optimizer,d_optimizer,mean,std,device,config))  # 原训练函数，不向梯度加入监测目标或GT。
            state['phase']='matched_first20' if step<=STAGE1_END else 'unpaired_extension'
            row={'global_step':step,'updates':step-ORIGIN,'new_updates':step-START,'indices':indices,'phase':state['phase'],
                 'actual_optimizer_lrs':{'student':optimizer.param_groups[0]['lr'],'discriminator':d_optimizer.param_groups[0]['lr']},
                 'elapsed_seconds':time.monotonic()-started,**losses}
            pair.append_json(output/'step_losses.jsonl',row)
            if step==START+1:
                state['first_step_loss_absolute_differences']=first_step_check(controls[0],row)
                base.save_json(output/'first_step_comparison.json',{'global_step':step,'differences':state['first_step_loss_absolute_differences']})
            pending.append(row)
            if (step-START)%20==0:
                window=pair.window_record(pending);pending=[];windows.append(window)
                pair.append_json(output/'loss_windows.jsonl',window)
                entry=evaluate(step);monitors.append(entry);state['last_monitor']=entry
                pair.append_json(output/'monitor.jsonl',entry)
                trend=prior.trend_state(windows,monitors,anchor,baseline3227);state['trend']=trend
                if step==STAGE1_END:
                    external=[]  # 验收时也检查时间、信号、空间；不能把已触发保护的阶段标为通过。
                    if stop['requested']:external.append('requested_stop')
                    if time.monotonic()-started>=args.max_seconds:external.append('time_budget')
                    try:budget.require()
                    except pair.InsufficientDisk:external.append('disk_space_guard')
                    state['stage1_gate']=stage1_gate(window,entry,trend,baseline3227['mel_l1'],external)
                    base.save_json(output/'stage1_gate.json',state['stage1_gate'])
                pair.append_json(output/'trend.jsonl',{'global_step':step,'new_updates':step-START,'stage1_gate':state['stage1_gate'],**trend})
                print(json.dumps({'global_step':step,'means':window['means'],'monitor':entry,'trend':trend,'stage1_gate':state['stage1_gate']}),flush=True)
                if trend['stop_reasons']:reason=trend['stop_reasons'][0];break
                if step==STAGE1_END and state['stage1_gate']['external_stop_reasons']:reason=state['stage1_gate']['external_stop_reasons'][0];break
                if step==STAGE1_END and not state['stage1_gate']['passed']:reason='stage1_acceptance_failed';break
            base.save_json(output/'status.json',state)
        step=state['completed_global_step']
        if monitors[-1]['global_step']!=step:
            entry=evaluate(step);monitors.append(entry);state['last_monitor']=entry
            pair.append_json(output/'monitor.jsonl',entry)
        trend=prior.trend_state(windows,monitors,anchor,baseline3227);state['trend']=trend
        if trend['stop_reasons']:reason=trend['stop_reasons'][0]
        pair.assert_frozen(frozen,fingerprints)
        if {key:probe.tensor_sha(value) for key,value in inputs.items()}!=contract['input_sha256'] or probe.tensor_sha(target)!=contract['fixed_teacher_target_sha256']:
            raise RuntimeError('Fixed inputs or oldR3107 target changed')
        pair.validate_effective_adam(optimizer,d_optimizer,student,discriminator,ACTUAL_LRS['student'],step)
        state.update(reason=reason,partial_window_steps=len(pending),paired_updates=min(step-START,20),unpaired_updates=max(0,step-STAGE1_END))
        common={'kind':'acdd_s5e5_staged120_checkpoint','global_step':step,'epoch':0,'configuration':config.as_dict(),
            'effective_optimizer_lrs':ACTUAL_LRS,'configuration_interpretation':record['configuration_interpretation'],
            'development':monitors[-1]['development'],'run_contract_sha256':contract_sha,'lineage':lineage,'original_step0_guard_anchor':anchor,
            'score_semantics':'reconstruction_clean_x0_sqrt_alpha_bar','stop_reason':reason,'stage1_gate':state['stage1_gate'],
            'paired_updates':state['paired_updates'],'unpaired_updates':state['unpaired_updates']}
        prior.save_final(output,state,common,student,discriminator,optimizer,d_optimizer,budget)
        resume.verify_files(hashes)
        state['status']='completed' if reason=='maximum120_updates_completed' else 'stopped'
        state['elapsed_seconds']=time.monotonic()-started
        base.save_json(output/'completion.json',state)
        print('S5E5_STAGED_FINISHED '+json.dumps(state),flush=True)
    except Exception as error:
        pair.report_failure(output,state,error)  # full成功后即记账；半D/S步或保存失败不能伪装完整成功。
        raise


if __name__=='__main__':
    main()  # 只有root完整审查、CPU验收并显式启动之后才执行真实训练。
