"""从真实3327比较原均匀抽样与批内十层抽样；每组至多60步。"""

import argparse  # 必须另行批准真实运行，导入此文件不会训练。
import contextlib  # 监测结束后恢复模式、随机流和梯度开关。
import hashlib  # 所有复用模块先核字节，再导入。
import json
import math  # 拒绝NaN/Inf，禁止异常值通过保护条件。
import os  # 原子元数据、真实磁盘余量和timeout父进程。
import platform
import signal  # 信号只请求停机，当前完整D/S批次结束后保存。
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
MODULES = {
    'inspect_acdd_timestep_cross_3327_v1.py': 'fb2b94ba1aedf1beb29a333ac629c4095286e86adc70e89fdd551a1efa015db9',
    'inspect_acdd_d_update_3327_v1.py': 'c14d0997173bb734ae9d461d95d1637d2b7503725bbcc474febdb04e12c2cd49',
    'run_acdd_half_lr_continue480_v1.py': 'f7678ad423e352a3979af5cb7f43057544e237d0c089e22b754757d569ca18ba',
    'acdd_uniform_stratified_train_batch_v1.py': 'd29b4ec674521a23936a9cc5c3dfa62245a82be7cd51fe6b79ab97816918abbb',
}
for name, digest in MODULES.items():
    if hashlib.sha256((TOOLS/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen helper changed: '+name)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch
import inspect_acdd_timestep_cross_3327_v1 as cross
import inspect_acdd_d_update_3327_v1 as dstate
import run_acdd_half_lr_continue480_v1 as prior
import acdd_uniform_stratified_train_batch_v1 as sampler

pair, base, training, resume, probe = cross.pair, cross.base, cross.training, cross.resume, cross.probe
DEV, PARENT = cross.DEV, cross.CURRENT
START, STOP = 3327, 3387
ARMS = ('control_iid', 'primary_stratified')
LRS = {'student': .00005, 'discriminator': .0002}
OUTPUT = DEV / 'acdd_timestep_paired60_from3327_v1'
PROTOCOL = DEV / 'ACDD3327_TIMESTEP_PAIRED60_PROTOCOL_CN.md'
PROTOCOL_SHA = 'd4db9929f401d60b466cb936f7cb85df2e6655c62e8194ecf0a7ca95b28669aa'
PARENT_SHA = '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556'
PARENT_CANONICAL = '32651a26cdf7074a9e8ca30c3cdd099697541fa4b261df757d2866f9e0622aba'
CROSS_RESULT = cross.OUTPUT / 'results.json'
CROSS_IDENTITY = cross.OUTPUT / 'run_identity.json'
PINNED = {
    str(PROTOCOL): PROTOCOL_SHA,
    str(CROSS_RESULT): '2b40c4b6641e7bf49143c7c2a5ff62622f67a1e77b3d4f7c61227fd7389b23a5',
    str(CROSS_IDENTITY): 'b9183198b8d671b24c3c6a0af0384353b11a51874a1f6357f8e77acb77031491',
    str(PARENT/'loss_windows.jsonl'): '7a589e277c408f6f868bfd48eb14f3dfb5f5dd22df44fda28b8fd884aabdf773',
    **{str(TOOLS/name): digest for name, digest in MODULES.items()},
}
META_LIMIT = 16*1024**2  # 两组完整逐步日志、64句十层监测和外部日志一并计算。
DIR_RESERVE, MODEL_SLACK = 1024**2, 1024**2
RESERVE = 3*1024**3 + 128*1024**2  # 不挪用系统3GiB或128MiB保留空间。


class BoundaryStop(RuntimeError):
    """停止新计算；已完成的批次仍可在清理时间内保存。"""


class StopClock:
    def __init__(self, seconds, started=None):
        self.started = time.monotonic() if started is None else started
        self.seconds, self.reason, self.stop_time = seconds, None, None
        self.handlers, self.old_timer = {}, None

    def request(self, reason):
        if self.reason is None:
            self.reason, self.stop_time = reason, time.monotonic()

    def signal_handler(self, number, frame):
        self.request('time_budget' if number==signal.SIGALRM else 'requested_stop')

    def install(self):
        self.old_timer = signal.getitimer(signal.ITIMER_REAL)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            self.handlers[number] = signal.getsignal(number)
            signal.signal(number, self.signal_handler)
        signal.setitimer(signal.ITIMER_REAL, max(.001, self.started+self.seconds-time.monotonic()))

    def active(self):
        if time.monotonic()-self.started >= self.seconds:
            self.request('time_budget')
        if self.reason is not None:
            raise BoundaryStop(self.reason)  # 批次外检查；不在D/S半步中抛停止异常。

    def saving(self):
        # 外部timeout最终强制退出；Python能响应时还限制清理阶段不得超过90秒。
        if self.stop_time is not None and time.monotonic()-self.stop_time > 90:
            raise BoundaryStop('save_cleanup_grace_expired')

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in self.handlers.items():
            signal.signal(number, handler)
        if self.old_timer and self.old_timer[0]>0:
            signal.setitimer(signal.ITIMER_REAL, max(.001,self.old_timer[0]-(time.monotonic()-self.started)), self.old_timer[1])


def watchdog(seconds):
    parent = Path('/proc')/str(os.getppid())
    args = [part.decode() for part in (parent/'cmdline').read_bytes().split(b'\0') if part]
    if (parent/'exe').resolve()!=Path('/usr/bin/timeout').resolve() or not all(word in args for word in ('--signal=TERM','--kill-after=90s',str(seconds)+'s')):
        raise RuntimeError('Require external timeout --signal=TERM --kill-after=90s %ss' % seconds)
    return {'pid':os.getppid(),'command':args,'cleanup_only_seconds':90,'hard_kill_cannot_guarantee_checkpoint':True}


class Budget:
    def __init__(self, output, full_bytes, student_bytes):
        self.output = Path(output)
        self.pending = {'full':full_bytes+MODEL_SLACK,'student':student_bytes+MODEL_SLACK}

    def metadata_used(self):
        paths = list(self.output.rglob('*')) if self.output.exists() else []
        adjacent = Path(str(self.output)+'.log')
        if adjacent.exists():
            paths.append(adjacent)
        total = 0
        for path in paths:
            cross.regular(path,directory=path.is_dir())
            if path.is_file() and not (path.name.endswith('.pt') or path.name.endswith('.pt.tmp')):
                total += path.stat().st_size
        return total

    def require(self, extra_metadata=0):
        used = self.metadata_used()
        if used+extra_metadata>META_LIMIT:
            raise pair.InsufficientDisk('Complete metadata/log exceeds16MiB; do not omit observations')
        need = RESERVE+DIR_RESERVE+sum(self.pending.values())+META_LIMIT-used
        free = pair.available_bytes(self.output.parent)
        if free<need:
            raise pair.InsufficientDisk('available=%s required=%s pending=%s' % (free,need,self.pending))
        return {'available_bytes':free,'required_bytes':need,'pending_bytes':dict(self.pending),'metadata_bytes':used}

    def committed(self, name, path):
        allowance = self.pending.pop(name)  # 文件已原子成功即扣预算，不能伪称仍未保存。
        if max(path.stat().st_size,path.stat().st_blocks*512)>allowance:
            raise RuntimeError('Actual checkpoint allocation exceeded predeclared model slack')

    def write(self, path, value, append=False):
        path = Path(path)
        if self.output not in path.parents or path.is_symlink():
            raise RuntimeError('Metadata writes must remain inside this new trial')
        data = json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
        self.require(len(data)+1)
        if append:
            with path.open('ab') as handle:
                handle.write(data+b'\n');handle.flush();os.fsync(handle.fileno())
        else:
            temporary = path.with_suffix(path.suffix+'.tmp')
            if temporary.exists() or temporary.is_symlink():
                raise RuntimeError('Do not overwrite an interrupted metadata write')
            with temporary.open('xb') as handle:
                handle.write(data);handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,path)  # 只覆盖本次自己创建的状态文件，不触碰任何旧run。
        self.require()


def content_state(student, discriminator, optimizer, d_optimizer):
    return {'student':pair.model_sha(student),'discriminator':pair.model_sha(discriminator),
            'student_optimizer':probe.tree_sha(optimizer.state_dict()),'discriminator_optimizer':probe.tree_sha(d_optimizer.state_dict()),
            'rng':probe.tree_sha(training.capture_rng_state()),'flags':cross.flags()}


def restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, contract, receipt, device):
    dstate.validate_parent(payload,student,discriminator,config,contract,receipt)
    prior.check_owner(optimizer,student);prior.check_owner(d_optimizer,discriminator)
    student.load_state_dict(payload['student'],strict=True)
    discriminator.load_state_dict(payload['discriminator'],strict=True)
    dstate.restore_optimizer(optimizer,payload['student_optimizer'])
    dstate.restore_optimizer(d_optimizer,payload['discriminator_optimizer'])
    for model in (student,discriminator):
        model.train();training.set_requires_grad(model,True)
        for parameter in model.parameters():parameter.grad=None
    training.restore_rng_state(payload['rng_state'],device)
    expected_rng = payload['rng_state'] if device.type=='cuda' else dict(payload['rng_state'],cuda_all=[])
    if probe.tree_sha(training.capture_rng_state())!=probe.tree_sha(expected_rng):
        raise RuntimeError('Parent RNG failed to restore')
    pair.validate_effective_adam(optimizer,d_optimizer,student,discriminator,LRS['student'],START)
    prior.check_owner(optimizer,student);prior.check_owner(d_optimizer,discriminator)
    return content_state(student,discriminator,optimizer,d_optimizer)


@contextlib.contextmanager
def monitoring_state(models, device):
    # 不复制大参数；完整参数/buffer SHA前后相等，同时恢复每个子模块的模式和梯度开关。
    fingerprints = {name:pair.model_sha(model) for name,model in models.items()}
    modes = {name:{key:m.training for key,m in model.named_modules()} for name,model in models.items()}
    requires = {name:{key:p.requires_grad for key,p in model.named_parameters()} for name,model in models.items()}
    with pair.preserve_rng(device):
        try:yield
        finally:
            for name,model in models.items():
                for key,module in model.named_modules():module.training=modes[name][key]
                for key,parameter in model.named_parameters():parameter.requires_grad_(requires[name][key])
                if pair.model_sha(model)!=fingerprints[name] or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Monitor changed model/buffer or left gradients: '+name)


@torch.no_grad()
def uniform64_monitor(student, teacher, diffusion, batches, plan, config, device, check):
    # 两个固定32句块各一次学生和十次教师；不构造D/G，不加入训练梯度。
    predictions, by_layer, targets = [], [[] for _ in range(10)], [[] for _ in range(10)]
    counts = {'student':0,'teacher':0}
    for batch_index, inputs in enumerate(batches):
        check()
        y,q = training.make_student_prediction(student,diffusion,inputs['x0'],inputs['spk_emb'],inputs['bnf'],950,inputs['source_noise'])
        counts['student']+=1
        if y.shape!=inputs['x0'].shape or not bool(torch.isfinite(y).all()) or not torch.equal(q,inputs['q950']):
            raise RuntimeError('Fixed64 student output/source input is invalid')
        predictions.append(y)
        for layer in range(10):
            check()
            t=torch.tensor(plan['strata_timesteps'][layer][32*batch_index:32*(batch_index+1)],device=device,dtype=torch.long)
            loss,target=training.score_distillation_loss(y,teacher,diffusion,t,inputs['score_noise'],inputs['spk_emb'],inputs['bnf'],
                target_mode=config.score_target,weight_mode=config.score_weight)
            counts['teacher']+=1
            per=(y-target).abs().flatten(1).mean(1)*diffusion.sqrt_alphas_cumprod[t]
            if target.requires_grad or not bool(torch.isfinite(per).all()):
                raise RuntimeError('Fixed64 clean teacher target or weighted score is invalid')
            cross.require_close(float(per.mean()),float(loss),1e-7,1e-6)
            by_layer[layer].append(per.cpu());targets[layer].append(target)
    y,x=torch.cat(predictions),torch.cat([item['x0'] for item in batches])
    values=[torch.cat(items).tolist() for items in by_layer]
    if counts!={'student':2,'teacher':20} or any(len(row)!=64 for row in values):
        raise RuntimeError('Keep the complete64-by10 monitor and2S20T budget')
    return {'uniform_weighted_score':sum(sum(row)/64 for row in values)/10,
        'per_stratum_weighted_score':[sum(row)/64 for row in values],'per_stratum_per_utterance_weighted_score':values,
        'GT_l1':float((y-x).abs().mean()),'prediction_rms':float(y.square().mean().sqrt()),
        'per_utterance_GT_l1':(y-x).abs().flatten(1).mean(1).cpu().tolist(),
        'per_utterance_prediction_rms':y.square().flatten(1).mean(1).sqrt().cpu().tolist(),
        'prediction_identities':[cross.tensor_id(value) for value in predictions],
        'teacher_target_identities':[cross.tensor_id(torch.cat(items)) for items in targets],
        'actual_forward_counts':counts,'role':'fixed retainedTRAIN64 repeated over10 strata, not640 independent sentences'}


def validate_uniform_initial(actual, reference):
    saved=reference['readouts']
    gt=saved['new_TRAIN64_GT']['3327']
    cross.require_close(actual['GT_l1'],gt['l1'],1e-7,1e-6)
    cross.require_close(actual['uniform_weighted_score'],saved['ten_stratum_uniform_means']['y3327_R3327']['weighted'],1e-7,1e-6)
    if actual['prediction_identities']!=saved['student_prediction_identities']['3327'][1:]:
        raise RuntimeError('Initial fixed64 student tensors did not exactly replay3327')
    for layer in range(10):
        condition=saved['conditions']['stratum_'+str(layer)]
        if actual['teacher_target_identities'][layer]!=condition['target_identities']['3327']:
            raise RuntimeError('Initial fixed64 teacher target did not exactly replay')
        expected=condition['cells']['y3327_R3327']['per_weighted']
        if len(expected)!=64 or len(actual['per_stratum_per_utterance_weighted_score'][layer])!=64:
            raise RuntimeError('Every monitor must retain all64 utterances')
        for a,b in zip(actual['per_stratum_per_utterance_weighted_score'][layer],expected):cross.require_close(a,b,1e-7,1e-6)


def complete_batch(state, step, call):
    if step!=state['completed_global_step']+1 or not START<step<=STOP:
        raise RuntimeError('Only the next absolute step within the declared60 is allowed')
    state['partial_batch_may_have_updated_D']=True
    losses=call()  # 异常时不伪造S已完成；禁止随后保存半个交替更新。
    state.update(completed_global_step=step,new_updates=step-START,partial_batch_may_have_updated_D=False)
    pair.finite_losses(losses)
    return losses


def adam_counters(optimizer, d_optimizer, expected_step):
    # 每个真实更新后只检查所有参数的step，不搬运大moment；完整moment校验仍在恢复和终点执行。
    result={}
    for role,opt in (('student',optimizer),('discriminator',d_optimizer)):
        parameters=[parameter for group in opt.param_groups for parameter in group['params']]
        steps=[]
        for parameter in parameters:
            if parameter not in opt.state or 'step' not in opt.state[parameter]:
                raise RuntimeError('Adam parameter has no update counter: '+role)
            value=float(opt.state[parameter]['step'])
            if not math.isfinite(value) or value!=expected_step:
                raise RuntimeError('Actual Adam counter differs from complete step: '+role)
            steps.append(value)
        if not steps or len(set(map(id,parameters)))!=len(parameters):
            raise RuntimeError('Empty or duplicated optimizer parameter ownership')
        result[role]={'parameter_count':len(parameters),'all_parameter_steps':expected_step,
                      'lr':opt.param_groups[0]['lr']}
    return result


def register_future_files(dataset, future, files, check):
    # 未来60批的源文件全部在契约发布前绑定；重复使用同一资产只哈希一次。
    registered={}
    for row in future:
        for index in row['indices']:
            item=dataset.items[index]
            for key in ('mel_path','bnf_path','spk_emb_path'):
                path=Path(item[key]);name=str(path)
                if name not in registered:
                    check();cross.regular(path)
                    digest=base.sha256_file(path)
                    cross.merge_hashes(files,{name:digest});registered[name]=digest
    return registered


def stop_rules(windows, monitors, anchor, baseline3227, baseline3327):
    result=prior.trend_state(windows,monitors,anchor,baseline3227)  # windows只含本轮新完整20步窗口。
    if monitors[-1]['development']['mel_l1']>baseline3327['mel_l1']*1.01:
        result['stop_reasons'].append('development_above3327_by1percent')
    return result


def candidate_gate(entry, control20, baseline3327, reasons):
    if entry['global_step']!=START+20 or control20 is None or control20['global_step']!=START+20:
        raise RuntimeError('Candidate gate requires the common first20 endpoint')
    g=entry['development']['mel_l1'];u=entry['uniform64'];c=control20['uniform64']
    if not all(math.isfinite(v) for v in (g,u['GT_l1'],u['uniform_weighted_score'],c['GT_l1'],c['uniform_weighted_score'])):
        raise RuntimeError('Nonfinite protective candidate gate')
    checks={'DEV32_not_worse_than_parent':g<=baseline3327['mel_l1'],
            'DEV32_not_worse_than_control20':g<=control20['development']['mel_l1'],
            'uniform64_score_not_worse_than_control20':u['uniform_weighted_score']<=c['uniform_weighted_score'],
            'TRAIN64_GT_not_worse_than_control20':u['GT_l1']<=c['GT_l1'],'all_stop_guards_clear':not reasons}
    return {'global_step':START+20,'passed':all(checks.values()),'checks':checks,'stop_reasons':list(reasons),
            'interpretation':'Protective development gate only; not evidence of objective performance improvement'}


def close_losses(actual, expected):
    differences={key:abs(float(actual[key])-float(value)) for key,value in expected.items()}
    if any(value>1e-7+1e-6*abs(float(expected[key])) for key,value in differences.items()):
        raise RuntimeError('Pre-update loss anchor mismatch: '+json.dumps(differences))
    return differences


def historical_first_check(row, anchor):
    for key in ('source_noise','score_noise'):
        if row['sampler'][key]!=anchor['input_identities'][key]:raise RuntimeError('Historical3328 noise mismatch: '+key)
    if row['sampler']['original_uniform_t']!=anchor['input_identities']['score_timesteps']:
        raise RuntimeError('Historical3328 time draw mismatch')
    for new,old in (('mel','x0'),('bnf','bnf'),('spk_emb','spk_emb')):
        if row['batch_tensors'][new]!=anchor['input_identities'][old]:raise RuntimeError('Historical3328 data mismatch')
    # 只比较历史确实保存的原始五项；不声称旧记录存在独立2FM/45score字段。
    expected={'discriminator':3.91922926902771,'student_adv':2.151906728744507,'fm_raw':.8101478815078735,
              'dist_raw':.054745227098464966,'student_total':6.2357378005981445}
    return {'loss_absolute_differences':close_losses(row,expected),'historical_postupdate_comparison_is_descriptive':True,
        'actual_postupdate':row['postupdate_state_sha256'],
        'historical_postupdate':{'student':'aaadcf83287f3e2c06f95a278a3e79baaff2a083788b0f214bc964f7ab0e0d92',
            'discriminator':'a90b0e1036e3a5c025621487640ff26e179287b9a213b44e2d98aa925f2a1f8f'}}


def common_input_check(row, control):
    for key in ('global_step','indices','batch_tensors'):
        if row[key]!=control[key]:raise RuntimeError('Paired training input differs: '+key)
    for key in ('source_noise','score_noise','original_uniform_t','original_uniform_values'):
        if row['sampler'][key]!=control['sampler'][key]:raise RuntimeError('Paired original RNG draw differs: '+key)
    if row['global_step']==START+1:
        return close_losses(row,{key:control[key] for key in ('discriminator','student_adv','fm_raw','fm_weighted')})
    return {}  # 更改t后学生轨迹会分开，不能要求后续loss相同。


def permits_candidate(control):
    # 仅完整、可解释的保护性停止可进入下一组；空间/时间/信号/执行错误不能自动恢复。
    allowed={'maximum60_updates_completed','moving_total_three_windows_rising','development_three_checks_rising',
             'original_S0_guard','development_above3227_by1percent','development_above3327_by1percent'}
    return (control.get('status') in ('completed','stopped') and control.get('reason') in allowed
            and not control.get('execution_stop_reasons')
            and not control.get('partial_batch_may_have_updated_D',True) and control.get('new_updates',0)>=20)


def terminal_status(state, clock):
    # 保存过程中到达时间/收到停止信号也要反映在收据，不能继续声称正常完成。
    try:clock.active()
    except BoundaryStop:
        state['execution_stop_during_finalization']=clock.reason
        state['reason']=clock.reason
    state['status']='completed' if state['reason']=='maximum60_updates_completed' else 'stopped'
    return state['status']


def save_primary(directory,state,common,student,discriminator,optimizer,d_optimizer,budget,clock):
    if state['arm']!='primary_stratified':
        raise RuntimeError('Control is hash/log only; no control checkpoint is authorized')
    if state['partial_batch_may_have_updated_D']:raise RuntimeError('Never serialize a half D/S update as complete')
    step=state['completed_global_step']
    if step==START:return
    full_path,s_path=directory/'final_full.pt',directory/('student_step_%06d.pt'%step)
    if any(path.exists() or path.is_symlink() or path.with_suffix(path.suffix+'.tmp').exists() for path in (full_path,s_path)):
        raise RuntimeError('Final checkpoint targets must be new; no retry or overwrite')
    clock.saving();budget.require()
    full=dict(common,student=student.state_dict(),discriminator=discriminator.state_dict(),student_optimizer=optimizer.state_dict(),
              discriminator_optimizer=d_optimizer.state_dict(),rng_state=training.capture_rng_state())
    training.atomic_torch_save(full,full_path)
    state['last_full_saved_global_step']=step  # 必须早于hash、S保存和任何JSON写入。
    state['full_snapshot']={'path':str(full_path),'sha256':None,'global_step':step,'committed':True}
    budget.committed('full',full_path)
    state['full_snapshot']['sha256']=base.sha256_file(full_path);full_path.chmod(0o444)
    clock.saving();budget.require()
    training.atomic_torch_save(dict(common,student=student.state_dict()),s_path)
    state['student_snapshot']={'path':str(s_path),'sha256':None,'global_step':step,'committed':True}
    budget.committed('student',s_path)
    state['student_snapshot']['sha256']=base.sha256_file(s_path);s_path.chmod(0o444)
    budget.write(directory/'checkpoint_receipt.json',state)


def failure_record(output,state,error,budget):
    record={'status':'failed','error_type':type(error).__name__,'error':str(error),'actual_arm_state':state,
            'old_artifacts_not_written_by_this_trial':True}
    print('PAIRED60_FAILED '+json.dumps(record,ensure_ascii=False),flush=True)
    if output is not None:
        try:budget.write(output/'failure.json',record)
        except BaseException as secondary:print('FAILURE_RECEIPT_NOT_SAVED '+repr(secondary),flush=True)


def main():
    started=time.monotonic()
    locks,clock,budget,owned,state=[],None,None,False,{}
    try:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--run-approved',action='store_true')
        parser.add_argument('--parent-checkpoint-sha256',required=True)
        parser.add_argument('--max-seconds',type=int,default=1200)
        args=parser.parse_args()
        if not args.run_approved or args.parent_checkpoint_sha256!=PARENT_SHA or not 1<=args.max_seconds<=1200:
            raise RuntimeError('Require reviewed3327 parent, explicit approval and at most1200 seconds')
        clock=StopClock(args.max_seconds,started);clock.install()
        external=watchdog(args.max_seconds)
        training.assert_rented_server(ROOT);cross.regular(DEV,directory=True)
        if OUTPUT.exists() or OUTPUT.is_symlink():raise RuntimeError('Refuse an existing or redirected trial directory')
        locks=training.acquire_runtime_locks(ROOT)
        if training.git_head(ROOT)!=cross.HEAD or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Require the unchanged clean authoritative source')
        if (platform.python_version(),torch.__version__,torch.version.cuda)!=('3.8.15','1.13.1+cu116','11.6') or Path(sys.executable).resolve()!=Path('/root/miniconda3/envs/myconda/bin/python').resolve():
            raise RuntimeError('Use the original registered myconda runtime')
        if cross.flags()!=cross.ORIGINAL_FLAGS:raise RuntimeError('The four original precision flags must remain unchanged')
        allocated=lambda path:max(path.stat().st_size,path.stat().st_blocks*512)
        budget=Budget(OUTPUT,allocated(PARENT/'final_full.pt'),allocated(PARENT/'student_step_003327.pt'))
        disk=budget.require();gpu=cross.gpu_gate();clock.active()
        files,contracts,receipts,development,old_rows,config=cross.load_metadata()
        cross.merge_hashes(files,PINNED)
        cross.merge_hashes(files,{str(Path(__file__).resolve()):base.sha256_file(Path(__file__))})
        cross.check_files(files)
        contract,parent=contracts[3327],receipts[3327]
        if parent['status']!='stopped':raise RuntimeError('The old3327 experiment must remain stopped')
        anchor=contract['original_step0_guard_anchor'];baseline3227=contract['baseline3227_development']
        baseline3327=parent['last_monitor']['development']
        if anchor!={'mel_l1':.19941353797912598,'prediction_rms':.9994737505912781}:
            # 原记录可能还有逐句信息；核心数值必须一致，保留完整原anchor到契约。
            if anchor['mel_l1']!=.19941353797912598 or anchor['prediction_rms']!=.9994737505912781:raise RuntimeError('OriginalS0 guard changed')
        device=torch.device('cuda')
        old,batches,target,monitor_plan=cross.prepare_inputs(files,development,contracts,device)
        reference=cross.read_json(CROSS_RESULT);monitor_identity=cross.read_json(CROSS_IDENTITY)
        if monitor_plan!=monitor_identity['plan']:raise RuntimeError('Do not reselect the fixed64 sentences, crops, t or noises')
        raw=resume.load_torch_checkpoint(cross.INPUT_FILE)
        inputs={key:value.to(device) for key,value in raw.items()};del raw,old
        diffusion=base.VoiceGradDiffusion(config.diffusion_steps,config.cosine_offset).to(device)
        for b,expected in zip(batches,reference['readouts']['input_identities_after'][1:]):
            # q950只由同一个冻结扩散表算出，必须等于已经发表的只读输入SHA。
            t=torch.full((32,),950,device=device,dtype=torch.long)
            b['q950']=diffusion.q_sample(b['x0'],t,noise=b['source_noise'])
            if {k:cross.tensor_id(v) for k,v in b.items()}!=expected:raise RuntimeError('Fixed64 cached inputs changed')
        input_hashes={'old':{k:probe.tensor_sha(v) for k,v in inputs.items()},
                      'new64':[{k:cross.tensor_id(v) for k,v in b.items()} for b in batches],'R3107':probe.tensor_sha(target)}
        dataset=base.VoiceGradDataset(ROOT/'data','train',segment_length=128,expected_speakers=100,bnf_alignment=config.bnf_alignment)
        plan=resume.GlobalBatchPlan(development['retained_train_indices'])
        if len(dataset)!=39387 or plan.batches_per_pass!=1229:raise RuntimeError('Original batch plan changed')
        future=[{'global_step':step,'indices':plan.for_step(step),'crop_seed':1234+100000+step,'noise_seed':1234+200000+step,
                 'stratified':sampler.stratified_timesteps(step)[1]} for step in range(START+1,STOP+1)]
        if any(set(row['indices'])&set(development['excluded_train_indices']) for row in future):raise RuntimeError('Training overlaps52 exclusions')
        future_files=register_future_files(dataset,future,files,clock.active)
        cross.check_files(files);clock.active();budget.require()
        record={'kind':'acdd_sampler_only_paired60_development','protocol_sha256':PROTOCOL_SHA,'parent_checkpoint':str(PARENT/'final_full.pt'),
            'parent_checkpoint_sha256':PARENT_SHA,'parent_contract_sha256':PARENT_CANONICAL,'seed':1234,
            'configuration':config.as_dict(),'effective_optimizer_lrs':LRS,'training_indices':development['retained_train_indices'],
            'excluded_train_indices':development['excluded_train_indices'],'development_contract_file_sha256':resume.DEVELOPMENT_SHA,
            'future_batches':future,'future_training_feature_sha256':future_files,
            'monitor64_plan':monitor_plan,'input_identities':input_hashes,'file_sha256':dict(files),
            'arms':ARMS,'maximum_updates_per_arm':60,'start_after_global_step':START,'stop_global_step':STOP,
            'sampling_extension':'Only primary replaces already-consumed uniform t by independent per-batch10strata3plus2; each sentence marginal uniform1..1000.',
            'objective_unchanged':'reconstruction clean_x0/sqrt_alpha_bar; adv+2FM+45dist; original GAN/D-S order and source950',
            'window_semantics':'Fresh prospective20-step windows from3328 in each arm. Archived3327 remains stopped; no inherited failures relabelled as passed.',
            'original_step0_guard_anchor':anchor,'baseline3227_development':baseline3227,'baseline3327_development':baseline3327,
            'old_monitor_anchor':{'D':3107,'teacher_target_A':3107,'TRAIN':3108},
            'fixed_frozen_models_sha256':contract['fixed_frozen_models_sha256'],
            'stop_rules':['three new complete20 total means strictly rise','three DEV32 readings strictly rise','originalS0','DEV32>3227*1.01','DEV32>3327*1.01','nonfinite/zero_grad','disk/time/signal/half_update_error'],
            'candidate20_gate':['DEV32<=parent3327','DEV32<=control20','uniform64score<=control20','TRAIN64GT<=control20','all execution and strict guards clear'],
            'comparison':'Only common completed20-step endpoints; no best-checkpoint selection or unequal-length performance claim.',
            'storage':{'only_primary_final_full_and_S':True,'control_hashes_are_not_recoverable_checkpoints':True,'metadata_limit_bytes':META_LIMIT,
                'model_slack_bytes_each':MODEL_SLACK,'directory_reserve_bytes':DIR_RESERVE,'system_and_report_reserve_bytes':RESERVE,'initial':disk},
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'flags':cross.flags(),'gpu':gpu,'external_timeout':external},
            'max_seconds':args.max_seconds,'formal_test_used_for_this_selection':False,'metric_evaluation_authorized':False,
            'historical_formal3327_already_observed':True,'limits':['Adaptive development experiment; no independent generalization or CER claim.','Stratification preserves marginal score sampling, not necessarily Adam trajectory expectation.']}
        OUTPUT.mkdir();owned=True;budget.write(OUTPUT/'run_contract.json',record)
        contract_sha=training.canonical_json_sha256(record)
        # 身份与将用的全部采样方案先冻结，然后才读取学生/教师模型作本次读数。
        clock.active();payload=resume.load_torch_checkpoint(PARENT/'final_full.pt')
        teacher,student,_=base.load_teacher_and_student(base.TEACHER,config,device)
        vocoder,vconfig,_=base.load_frozen_vocoder(ROOT,base.VOCODER_DIR/'generator_step_2500000.pt',base.VOCODER_DIR/'config.json',
            base.VOCODER_DIR/'final_checkpoint_attestation_comments_20260906.json',config,device)
        discriminator=base.build_paper_discriminator(ROOT,vconfig,config,device)
        fixed_d=base.build_paper_discriminator(ROOT,vconfig,config,device)
        old_d=resume.load_torch_checkpoint(pair.PARENT/'latest_full.pt')
        if old_d['global_step']!=3107 or old_d['run_contract_sha256']!=pair.PARENT_CONTRACT:raise RuntimeError('Old fixedD anchor changed')
        fixed_d.load_state_dict(old_d['discriminator'],strict=True);del old_d
        fixed_d.eval();training.set_requires_grad(fixed_d,False)
        mean,std=base.load_mel_statistics(ROOT/'data',device)
        frozen={'teacher':teacher,'vocoder':vocoder,'fixed_D3107':fixed_d}
        frozen_sha={name:pair.model_sha(model) for name,model in frozen.items()}
        if frozen_sha!=contract['fixed_frozen_models_sha256']:raise RuntimeError('Teacher/G/fixedD parameters or buffers changed')
        all_models=dict(frozen,student=student,active_D=discriminator,diffusion=diffusion)
        fixed_inputs=lambda:{'old':{k:probe.tensor_sha(v) for k,v in inputs.items()},'new64':[{k:cross.tensor_id(v) for k,v in b.items()} for b in batches],'R3107':probe.tensor_sha(target)}
        def evaluate(step):
            clock.active();budget.require()
            with monitoring_state(all_models,device):
                entry=pair.monitor(student,teacher,vocoder,fixed_d,diffusion,inputs,target,mean,std,config,device,step)
                entry['legacy_monitor_updates_from3107']=entry['updates']  # 留下旧监测函数原横轴，不混成新窗口计数。
                entry['updates']=step-START
                entry['trial_updates']=step-START
                entry['uniform64']=uniform64_monitor(student,teacher,diffusion,batches,monitor_plan,config,device,clock.active)
            pair.assert_frozen(frozen,frozen_sha)
            if fixed_inputs()!=input_hashes:raise RuntimeError('Monitoring changed a fixed input orR3107')
            return entry
        anchor3328=cross.read_json(cross.D_ANCHOR)
        summaries,control_rows,control_monitors,initial_restore,initial_monitor={},[],[],None,None
        primary_not_started_reason=None
        for arm in ARMS:
            if arm==ARMS[1] and not permits_candidate(summaries[ARMS[0]]):
                primary_not_started_reason='control_execution_stop_or_no_complete_control20';break
            try:clock.active();budget.require()
            except (BoundaryStop,pair.InsufficientDisk) as error:
                if arm==ARMS[1]:primary_not_started_reason=str(error);break
                raise
            if arm==ARMS[1] and not any(row['global_step']==START+20 for row in control_monitors):
                raise RuntimeError('No complete valid control20; candidate is not permitted')
            directory=OUTPUT/arm;directory.mkdir()
            optimizer=torch.optim.Adam(student.parameters(),lr=LRS['student'],betas=(.5,.9))
            d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=LRS['discriminator'],betas=(.5,.9))
            restored=restore_parent(payload,student,discriminator,optimizer,d_optimizer,config,contract,parent,device)
            if initial_restore is None:initial_restore=restored
            elif restored!=initial_restore:raise RuntimeError('Arms did not cold-restore the same S/D/twoAdam/RNG/flags')
            state={'arm':arm,'status':'running','completed_global_step':START,'new_updates':0,'last_full_saved_global_step':START,
                'full_snapshot':None,'student_snapshot':None,'partial_batch_may_have_updated_D':False,'contract_sha256':contract_sha,
                'parent_checkpoint_sha256':PARENT_SHA,'effective_optimizer_lrs':LRS,'last_monitor':None,'candidate20_gate':None,
                'execution_stop_reasons':[],
                'restore_identity':restored,'control_model_will_not_be_saved':arm==ARMS[0]}
            entry=evaluate(START);prior.validate_replay(entry,parent['last_monitor']);validate_uniform_initial(entry['uniform64'],reference)
            if initial_monitor is None:initial_monitor=entry
            elif entry!=initial_monitor:raise RuntimeError('Both arms must have exactly the same initial monitoring values')
            monitors,windows,rows,pending=[entry],[],[],[];state['last_monitor']=entry
            budget.write(directory/'monitor.jsonl',entry,append=True)
            reason='maximum60_updates_completed'
            try:
                for planned in future:
                    step=planned['global_step'];clock.active();budget.require()
                    if arm==ARMS[1] and step>START+20 and not (state['candidate20_gate'] and state['candidate20_gate']['passed']):
                        raise RuntimeError('No candidate continuation beyond20 before all prospective gates pass')
                    batch=base.batch_for(dataset,planned['indices'],planned['crop_seed'])
                    receipt={};override=None
                    if arm==ARMS[1]:override=torch.tensor(planned['stratified']['timesteps'],dtype=torch.long)
                    training.seed_everything(planned['noise_seed'])
                    losses=complete_batch(state,step,lambda:sampler.train_batch_with_timesteps(batch,teacher,student,vocoder,discriminator,diffusion,
                        optimizer,d_optimizer,mean,std,device,config,score_timesteps_override=override,sampler_receipt=receipt))
                    counters=adam_counters(optimizer,d_optimizer,step)
                    row={'global_step':step,'updates':step-START,'trial_updates':step-START,'indices':planned['indices'],
                        'batch_tensors':{key:cross.tensor_id(batch[key]) for key in ('mel','bnf','spk_emb')},'sampler':receipt,
                        'actual_optimizer_lrs':{'student':optimizer.param_groups[0]['lr'],'discriminator':d_optimizer.param_groups[0]['lr']},
                        'actual_adam_counters':counters,
                        'comparison_scope':'unpaired_extra' if arm==ARMS[1] and step>START+len(control_rows) else 'common_planned_prefix',**losses}
                    if row['actual_optimizer_lrs']!=LRS:raise RuntimeError('An optimizer learning rate changed')
                    if step==START+1:
                        row['postupdate_state_sha256']={'student':pair.model_sha(student),'discriminator':pair.model_sha(discriminator)}
                        if arm==ARMS[0]:state['historical_first_step']=historical_first_check(row,anchor3328)
                    if arm==ARMS[1] and step<=START+len(control_rows):
                        row['paired_preupdate_absolute_differences']=common_input_check(row,control_rows[step-START-1])
                    budget.write(directory/'step_losses.jsonl',row,append=True)
                    rows.append(row);pending.append(row)
                    if len(pending)==20:
                        window=pair.window_record(pending);pending=[];windows.append(window)
                        budget.write(directory/'loss_windows.jsonl',window,append=True)
                        entry=evaluate(step);monitors.append(entry);state['last_monitor']=entry
                        budget.write(directory/'monitor.jsonl',entry,append=True)
                        trend=stop_rules(windows,monitors,anchor,baseline3227,baseline3327);state['trend']=trend
                        execution=[]
                        try:clock.active();budget.require()
                        except (BoundaryStop,pair.InsufficientDisk) as error:execution.append(str(error))
                        state['execution_stop_reasons'].extend(execution)
                        if arm==ARMS[1] and step==START+20:
                            control20=next(item for item in control_monitors if item['global_step']==START+20)
                            state['candidate20_gate']=candidate_gate(entry,control20,baseline3327,trend['stop_reasons']+execution)
                            budget.write(directory/'candidate20_gate.json',state['candidate20_gate'])
                        budget.write(directory/'trend.jsonl',{'global_step':step,**trend,'execution_stops':execution},append=True)
                        print('PAIRED60_WINDOW '+json.dumps({'arm':arm,'step':step,'means':window['means'],'DEV32':entry['development']['mel_l1'],
                            'uniform64':entry['uniform64']['uniform_weighted_score'],'TRAIN64_GT':entry['uniform64']['GT_l1'],'stop':trend['stop_reasons'],'gate':state['candidate20_gate']}),flush=True)
                        if execution:reason=execution[0];break
                        if trend['stop_reasons']:reason=trend['stop_reasons'][0];break
                        if arm==ARMS[1] and step==START+20 and not state['candidate20_gate']['passed']:reason='candidate20_protective_gate_failed';break
                    budget.write(directory/'status.json',state)
            except (BoundaryStop,pair.InsufficientDisk) as error:
                reason=str(error)  # 仅批次边界/监测中止允许保存；生产半步异常仍走最外层failure。
                state['execution_stop_reasons'].append(reason)
                if state['partial_batch_may_have_updated_D']:raise
            step=state['completed_global_step']
            # 不为截止时间后的终点补一次新监测；标明最后一次监测的真实步数。
            state.update(reason=reason,partial_window_steps=len(pending),end_monitor_available=state['last_monitor']['global_step']==step,
                endpoint_state_sha256=content_state(student,discriminator,optimizer,d_optimizer))
            state['unpaired_extra_updates']=max(0,step-START-len(control_rows)) if arm==ARMS[1] else None
            state['common_paired20_global_steps']=sorted(set(item['global_step'] for item in control_monitors)
                &set(item['global_step'] for item in monitors)-{START}) if arm==ARMS[1] else []
            pair.validate_effective_adam(optimizer,d_optimizer,student,discriminator,LRS['student'],step)
            pair.assert_frozen(frozen,frozen_sha)
            if fixed_inputs()!=input_hashes or cross.flags()!=cross.ORIGINAL_FLAGS:raise RuntimeError('Frozen input/model precision changed')
            if arm==ARMS[1]:
                common={'kind':'acdd_stratified_paired60_checkpoint','global_step':step,'epoch':0,'configuration':config.as_dict(),
                    'effective_optimizer_lrs':LRS,'score_semantics':'reconstruction_clean_x0_sqrt_alpha_bar',
                    'sampler_semantics':'independent_per_batch_10strata3plus2_uniform_marginal','sampler_seed':sampler.SAMPLER_SEED,
                    'development':state['last_monitor']['development'] if state['end_monitor_available'] else None,
                    'development_monitor_global_step':state['last_monitor']['global_step'],'run_contract_sha256':contract_sha,
                    'lineage':{'parent_checkpoint_sha256':PARENT_SHA,'parent_contract_sha256':PARENT_CANONICAL,'start_after_global_step':START},
                    'original_step0_guard_anchor':anchor,'stop_reason':reason,'candidate20_gate':state['candidate20_gate']}
                save_primary(directory,state,common,student,discriminator,optimizer,d_optimizer,budget,clock)
            terminal_status(state,clock)
            budget.write(directory/'arm_completion.json',state)
            summaries[arm]=dict(state)
            if arm==ARMS[0]:control_rows,control_monitors=rows,monitors
            del optimizer,d_optimizer  # 后一组另建优化器，再从相同父moment冷恢复。
        cross.check_files(files);cross.check_files(PINNED)
        if training.canonical_json_sha256(cross.read_json(OUTPUT/'run_contract.json'))!=contract_sha:raise RuntimeError('Published contract changed')
        common_steps=sorted(set(item['global_step'] for item in control_monitors)&set(item['global_step'] for item in monitors)) if ARMS[1] in summaries else []
        try:clock.active()
        except BoundaryStop:pass
        completion={'status':'bounded_trial_stopped' if clock.reason or primary_not_started_reason else 'bounded_trial_completed',
                    'contract_sha256':contract_sha,'arms':summaries,'common_monitor_global_steps':common_steps,
                    'common_paired20_global_steps':[step for step in common_steps if step>START],
                    'primary_unpaired_extra_updates':summaries.get(ARMS[1],{}).get('unpaired_extra_updates'),
                    'primary_not_started_reason':primary_not_started_reason,'execution_stop_during_finalization':clock.reason,
                    'elapsed_seconds':time.monotonic()-started,'no_metric_evaluation_performed':True,'old3327_run_remains_stopped':True}
        budget.write(OUTPUT/'completion.json',completion)
        print('PAIRED60_FINISHED '+json.dumps({'common_steps':common_steps,'endpoints':{key:value['completed_global_step'] for key,value in summaries.items()}}),flush=True)
    except BaseException as error:
        failure_record(OUTPUT if owned else None,state,error,budget)
        raise
    finally:
        if clock is not None:clock.close()
        closed=[]
        for handle in reversed(locks):
            try:handle.close();closed.append(True)
            except Exception as error:closed.append(False);print('LOCK_CLOSE_ERROR '+repr(error),flush=True)
        print('PAIRED60_LOCKS_RELEASED '+json.dumps({'handles':len(locks),'closed':closed}),flush=True)


if __name__=='__main__':
    main()  # 仅主线程审核代码/CPU测试、容量和身份后，才可另行启动GPU。
