"""只检查分项与联合梯度的数值差异；没有优化器、D更新或模型保存。"""

import argparse  # 真GPU诊断由主线程显式批准，导入不运行。
import contextlib  # 精度开关、全部模型状态和RNG在异常时也恢复。
import hashlib  # 先核验冻结helper，不能导入同名未审核代码。
import json  # 只允许新增不超过2MiB的JSON。
import math  # 全部差异原样报告，不设通过阈值。
import os  # 查询余量和原子发布小报告，不删除文件。
import platform
import sys
import time  # 整个实验最多600秒。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
HELPER = TOOLS / 'inspect_acdd_d_update_3327_v1.py'
HELPER_SHA = 'c14d0997173bb734ae9d461d95d1637d2b7503725bbcc474febdb04e12c2cd49'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Frozen state/source helper changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import inspect_acdd_d_update_3327_v1 as d

pair, base, training, resume, probe, cross = d.pair, d.base, d.training, d.resume, d.probe, d.cross
DEV = d.DEV
PROTOCOL = DEV / 'ACDD3327_GRADIENT_PRECISION_PROTOCOL_CN.md'
PROTOCOL_SHA = 'c29a4b67eadf87751c70772ee0b96836ad38fbad7e2561618d18cb58e9313ab5'
OUTPUT = DEV / 'acdd_gradient_precision_readonly_3327_v1'
CONDITIONS = ('original_cudnn_tf32_on', 'diagnostic_cudnn_tf32_off')
ROUTES = ('split', 'joint1', 'joint2')
ORIGINAL_FLAGS = {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True,
                  'cudnn_benchmark': False, 'cudnn_deterministic': False}
MAX_JSON_BYTES = 2 * 1024**2


def read_flags():
    return {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic}


def write_flags(values):
    if set(values) != set(ORIGINAL_FLAGS) or any(type(v) is not bool for v in values.values()):
        raise RuntimeError('All four precision flags must be explicit booleans')
    torch.backends.cuda.matmul.allow_tf32 = values['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = values['cudnn_allow_tf32']
    torch.backends.cudnn.benchmark = values['cudnn_benchmark']
    torch.backends.cudnn.deterministic = values['cudnn_deterministic']


@contextlib.contextmanager
def precision_scope(values):
    original = read_flags()
    try:
        write_flags(values)
        if read_flags() != values:
            raise RuntimeError('Requested precision flags were not applied')
        yield
    finally:
        write_flags(original)
        if read_flags() != original:
            raise RuntimeError('Original precision flags failed to restore')


def difference(left, reference):
    # 差值为left-reference；分母为reference的L2。零分母保持undefined，不夹成任意小数。
    if left.shape != reference.shape or left.dtype != reference.dtype:
        raise RuntimeError('Tensor shape or dtype differs')
    if not bool(torch.isfinite(left).all() and torch.isfinite(reference).all()):
        raise RuntimeError('Nonfinite tensor cannot enter precision comparison')
    a, b = left.detach().cpu().double(), reference.detach().cpu().double()
    delta = a - b
    error2, reference2, left2 = float(delta.square().sum()), float(b.square().sum()), float(a.square().sum())
    return {'max_absolute_difference': float(delta.abs().max()) if delta.numel() else 0.,
            'absolute_l2_difference': math.sqrt(error2), 'reference_l2_norm': math.sqrt(reference2),
            'left_l2_norm': math.sqrt(left2), 'relative_l2_difference': math.sqrt(error2/reference2) if reference2 else None,
            'relative_l2_undefined_zero_reference': reference2 == 0., 'actual_exact': torch.equal(left, reference),
            'numel': left.numel()}


def gradient_difference(left, reference):
    # 全71个参数逐项报告；None支持差异仍是计算图错误，不能藏在总范数里。
    if set(left) != set(reference):
        raise RuntimeError('Gradient parameter names differ')
    per, absent = {}, []
    error2 = reference2 = left2 = maximum = 0.
    for name in left:
        a, b = left[name], reference[name]
        if a is None or b is None:
            if a is not None or b is not None:
                raise RuntimeError('None gradient support differs: ' + name)
            absent.append(name); per[name] = {'left_is_none': True, 'reference_is_none': True}
            continue
        item = difference(a, b)
        per[name] = dict(item, left_is_none=False, reference_is_none=False)
        error2 += item['absolute_l2_difference']**2
        reference2 += item['reference_l2_norm']**2
        left2 += item['left_l2_norm']**2
        maximum = max(maximum, item['max_absolute_difference'])
    return {'max_absolute_difference': maximum, 'absolute_l2_difference': math.sqrt(error2),
            'reference_l2_norm': math.sqrt(reference2), 'left_l2_norm': math.sqrt(left2),
            'relative_l2_difference': math.sqrt(error2/reference2) if reference2 else None,
            'relative_l2_undefined_zero_reference': reference2 == 0., 'actual_exact': error2 == 0.,
            'none_parameters': absent, 'per_parameter': per, 'parameter_count': len(left)}


def gradient_norms(bank):
    # 只输出范数、最大绝对值和None名单，完整梯度永远不落盘。
    per, absent, total2 = {}, [], 0.
    for name, value in bank.items():
        if value is None:
            absent.append(name); per[name] = {'is_none': True}
            continue
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError('Nonfinite component gradient: ' + name)
        squared = float(value.double().square().sum()); total2 += squared
        per[name] = {'is_none': False, 'l2_norm': math.sqrt(squared), 'max_abs': float(value.abs().max()), 'numel': value.numel()}
    return {'l2_norm': math.sqrt(total2), 'none_parameters': absent, 'per_parameter': per}


def fixed_trace(prediction, q950, real, fake, target):
    # 保存这次真实forward的CPU小张量用于内存比较；不是事后另跑一次不同前向。
    return {name: value.detach().cpu().clone() for name, value in
            [('prediction', prediction), ('q950', q950), ('real_waveform', real), ('fake_waveform', fake), ('teacher_target', target)]}


def trace_identities(trace):
    return {name: cross.identity(value) for name, value in trace.items()}


def require_same_trace(actual, reference):
    if set(actual) != set(reference) or any(not torch.equal(actual[name], reference[name]) for name in reference):
        raise RuntimeError('Within-condition student/real/fake/target/noisy-input identity differs')


def verify_models(models, expected_hashes, *, discriminator_frozen):
    # S需要requires_grad=True才能求导；这里核权重未变，不能误用“所有参数冻结”的旧断言。
    for name, model in models.items():
        if pair.model_sha(model) != expected_hashes[name]:
            raise RuntimeError('Model parameters or buffers changed: ' + name)
    pair.assert_frozen({name:models[name] for name in ('teacher','vocoder')},
                       {name:expected_hashes[name] for name in ('teacher','vocoder')})
    if any(p.grad is not None for model in models.values() for p in model.parameters()):
        raise RuntimeError('A completed route left parameter.grad populated')
    if discriminator_frozen and any(p.requires_grad for p in models['discriminator'].parameters()):
        raise RuntimeError('D parameters must be frozen during student gradient comparison')


def losses(adv, fm, score):
    result = {'adv': float(adv), 'fm_raw': float(fm), 'fm_weighted': float(2*fm),
              'dist_raw': float(score), 'dist_weighted': float(45*score), 'total': float(adv+2*fm+45*score)}
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError('Nonfinite objective in numerical comparison')
    return result


def split_gradient(student, teacher, vocoder, discriminator, diffusion, inputs, mean, std, config, check):
    # 与原wave_output_gradients的分开求导顺序相同；直接保留原forward以核验共同作用点。
    prediction, q950 = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
    leaf = prediction.detach().requires_grad_(True)
    score, target = training.score_distillation_loss(leaf, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
        inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
    score_y = probe.checked_grad(score, leaf)
    fake = vocoder(training.denormalize_mel(leaf, mean, std))
    with torch.no_grad():
        real = vocoder(training.denormalize_mel(inputs['x0'], mean, std))
    proxy = fake.detach().requires_grad_(True)
    adv_gradient, fm_gradient = torch.zeros_like(proxy), torch.zeros_like(proxy)
    adv_value, fm_value = proxy.new_zeros(()), proxy.new_zeros(())
    for branch in discriminator.loss_branches():
        check()
        with torch.no_grad():
            _, real_maps = branch(real)
        fake_score, fake_maps = branch(proxy)
        adv = training.least_squares_student_loss([fake_score])
        fm = training.feature_matching_loss([real_maps], [fake_maps])
        adv_gradient.add_(probe.checked_grad(adv, proxy, retain_graph=True))
        fm_gradient.add_(probe.checked_grad(fm, proxy))
        adv_value.add_(adv.detach()); fm_value.add_(fm.detach())
        del real_maps, fake_maps, fake_score, adv, fm
    adv_y, = torch.autograd.grad(fake, leaf, grad_outputs=adv_gradient, retain_graph=True)
    fm_y, = torch.autograd.grad(fake, leaf, grad_outputs=fm_gradient)
    vectors = {'adv': adv_y.detach(), '2fm': 2*fm_y.detach(), '45score': 45*score_y}
    banks = probe.parameter_banks(prediction, list(student.named_parameters()), vectors, check)
    component_norms = {name: gradient_norms(bank) for name, bank in banks.items()}
    denominator = component_norms['total']['l2_norm']
    cancellation = sum(component_norms[k]['l2_norm'] for k in ('adv','2fm','45score')) / denominator if denominator else None
    record = {'losses': losses(adv_value,fm_value,score), 'component_norms':component_norms,
              'sum_component_norms_over_total_norm':cancellation, 'total_norm_zero':denominator == 0.}
    return banks['total'], fixed_trace(leaf,q950,real,fake,target), record


def joint_gradient(student, teacher, vocoder, discriminator, diffusion, inputs, mean, std, config, check):
    # 原生产学生阶段直到取得梯度为止：全batch代理、合并wave VJP、联合backward；没有step。
    for parameter in student.parameters():
        parameter.grad = None
    with torch.no_grad():
        proxy_prediction, _ = training.make_student_prediction(student,diffusion,inputs['x0'],inputs['spk_emb'],inputs['bnf'],950,inputs['source_noise'])
        real = vocoder(training.denormalize_mel(inputs['x0'],mean,std))
        fake_proxy = vocoder(training.denormalize_mel(proxy_prediction,mean,std))
    check()
    adv,fm,wave_gradient = training.serial_student_discriminator_gradient(discriminator,real,fake_proxy,config.lambda_fm)
    prediction,q950 = training.make_student_prediction(student,diffusion,inputs['x0'],inputs['spk_emb'],inputs['bnf'],950,inputs['source_noise'])
    fake = vocoder(training.denormalize_mel(prediction,mean,std))
    if not torch.equal(fake.detach(),fake_proxy):
        raise RuntimeError('Production joint waveform proxy did not replay exactly')
    score,target = training.score_distillation_loss(prediction,teacher,diffusion,inputs['score_timesteps'],inputs['score_noise'],
        inputs['spk_emb'],inputs['bnf'],target_mode=config.score_target,weight_mode=config.score_weight)
    check()
    torch.autograd.backward((fake,score),(wave_gradient,score.new_tensor(config.lambda_dist)))
    gradients = {name:None if p.grad is None else p.grad.detach().cpu().clone() for name,p in student.named_parameters()}
    if not any(v is not None for v in gradients.values()):
        raise RuntimeError('The joint graph produced no student gradient')
    norms = gradient_norms(gradients)  # 非有限数立即拒绝，但不按差异大小判断通过。
    if any(p.grad is not None for model in (teacher,vocoder,discriminator) for p in model.parameters()):
        raise RuntimeError('Frozen model unexpectedly received gradients')
    for parameter in student.parameters():
        parameter.grad = None  # joint2不能累积joint1梯度；此处没有优化器。
    return gradients,fixed_trace(prediction,q950,real,fake,target),{'losses':losses(adv,fm,score),'gradient_norms':norms}


def check_limits(started,maximum):
    cross.capacity(DEV)
    if time.monotonic()-started > maximum:
        raise TimeoutError('The predeclared600-second numerical diagnostic budget is exhausted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-seconds',type=int,default=600)
    parser.add_argument('--run-approved',action='store_true')
    args=parser.parse_args()
    if not args.run_approved or not 1<=args.max_seconds<=600:
        raise RuntimeError('Explicit approval and at most600 seconds required')
    training.assert_rented_server(ROOT)
    locks=training.acquire_runtime_locks(ROOT)  # 三把原锁持续覆盖全部六次反传及结果写入。
    if args.output_dir.is_symlink() or args.output_dir.resolve()!=OUTPUT or OUTPUT.exists():
        raise RuntimeError('Use the exact predeclared new directory')
    if (platform.python_version(),torch.__version__,torch.version.cuda)!=('3.8.15','1.13.1+cu116','11.6'):
        raise RuntimeError('Original training runtime required')
    if Path(sys.executable).resolve()!=Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Registered myconda CUDA interpreter required')
    if training.git_head(ROOT)!=base.EXPECTED_HEAD:
        raise RuntimeError('Frozen production source changed')
    initial_flags=read_flags()
    disk,gpu=cross.capacity(DEV),cross.gpu_idle()
    if gpu['free_mib']<24576:
        raise RuntimeError('Require at least24GiB freeVRAM for complete batch32')
    started=time.monotonic()
    def check():
        check_limits(started,args.max_seconds)
    def progress(stage):
        check();print('PRECISION3327 '+stage,flush=True)
    hashes=dict(d.PINNED)
    hashes.update({str(HELPER):HELPER_SHA,str(PROTOCOL):PROTOCOL_SHA,str(Path(__file__).resolve()):base.sha256_file(Path(__file__))})
    resume.verify_files(hashes)
    contract,receipt=cross.read_json(d.RUN/'run_contract.json'),cross.read_json(d.RUN/'completion.json')
    if training.canonical_json_sha256(contract)!=d.CONTRACT_SHA or {k:contract['runtime'][k] for k in ORIGINAL_FLAGS}!=ORIGINAL_FLAGS:
        raise RuntimeError('Parent contract or its archived precision flags changed')
    hashes.update(contract['file_sha256'])
    development,dev_hashes=resume.load_development(DEV/'development_contract.json');hashes.update(dev_hashes)
    resume.verify_files(hashes)
    grand=cross.read_json(resume.PARENT/'run_contract.json')
    if training.collect_source_fingerprints(ROOT)!=grand['source_sha256']:
        raise RuntimeError('Production source fingerprints changed')
    config=base.replace(base.PAPER_STUDENT_CONFIG,num_workers=0)
    if config.as_dict()!=contract['configuration'] or (config.score_target,config.score_weight,config.lambda_fm,config.lambda_dist)!=('clean_x0','sqrt_alpha_bar',2,45):
        raise RuntimeError('Original mathematical objective changed')
    if contract['training_indices']!=development['retained_train_indices'] or contract['excluded_train_indices']!=development['excluded_train_indices']:
        raise RuntimeError('Original retained training pool or52 exclusions changed')
    torch.set_num_threads(2)
    device=torch.device('cuda');torch.cuda.reset_peak_memory_stats(device)
    teacher,student,_=base.load_teacher_and_student(base.TEACHER,config,device)
    vocoder,vconfig,_=base.load_frozen_vocoder(ROOT,base.VOCODER_DIR/'generator_step_2500000.pt',base.VOCODER_DIR/'config.json',
        base.VOCODER_DIR/'final_checkpoint_attestation_comments_20260906.json',config,device)
    discriminator=base.build_paper_discriminator(ROOT,vconfig,config,device)
    diffusion=base.VoiceGradDiffusion(config.diffusion_steps,config.cosine_offset).to(device)
    mean,std=base.load_mel_statistics(ROOT/'data',device)
    payload=resume.load_torch_checkpoint(d.PARENT)
    d.validate_parent(payload,student,discriminator,config,contract,receipt)  # 只核Adam归档，没有优化器对象。
    student.load_state_dict(payload['student'],strict=True);discriminator.load_state_dict(payload['discriminator'],strict=True)
    student.train();discriminator.train();teacher.eval();vocoder.eval()
    training.set_requires_grad(student,True);training.set_requires_grad(discriminator,True)
    models={'student':student,'discriminator':discriminator,'teacher':teacher,'vocoder':vocoder}
    original_model_hashes={name:pair.model_sha(model) for name,model in models.items()}
    if any(original_model_hashes[n]!=contract['fixed_frozen_models_sha256'][n] for n in ('teacher','vocoder')):
        raise RuntimeError('The original frozen teacher/G changed')
    parent_optimizer_rng_sha=probe.tree_sha({'s':payload['student_optimizer'],'d':payload['discriminator_optimizer'],'rng':payload['rng_state']})
    dataset=base.VoiceGradDataset(ROOT/'data','train',segment_length=128,expected_speakers=100,bnf_alignment=config.bnf_alignment)
    indices=resume.GlobalBatchPlan(development['retained_train_indices']).for_step(d.NEXT)
    if len(indices)!=32 or set(indices)&set(development['excluded_train_indices']):
        raise RuntimeError('Require the same next full TRAIN3328 batch, excluding52 development utterances')
    conditions,precision_traces={},{}
    with d.restore_everything(models,{},payload['rng_state'],device) as originals:
        with pair.preserve_rng(device):
            tensors=training._slice_batch(base.batch_for(dataset,indices,1234+100000+d.NEXT),device)
            x0=tensors['mel'];training.seed_everything(1234+200000+d.NEXT)
            inputs={'x0':x0,'bnf':tensors['bnf'],'spk_emb':tensors['spk_emb'],
                'source_noise':torch.randn_like(x0),'score_noise':torch.randn_like(x0),
                'score_timesteps':torch.randint(1,1001,(32,),device=device,dtype=torch.long)}
            input_identities={n:cross.identity(v) for n,v in inputs.items()}
            route_rng=training.capture_rng_state();route_rng_sha=probe.tree_sha(route_rng)
            for condition in CONDITIONS:
                flags=dict(ORIGINAL_FLAGS,cudnn_allow_tf32=condition==CONDITIONS[0])
                with precision_scope(flags):
                    record={'flags_before':read_flags(),'routes':{}}
                    print('PRECISION3327_FLAGS_BEFORE '+json.dumps({'condition':condition,'flags':record['flags_before']}),flush=True)
                    gradients={};condition_trace=None
                    for route in ROUTES:
                        progress(condition+'/'+route)
                        for name,model in models.items():
                            d.restore(model,originals[name])
                        training.set_requires_grad(discriminator,False)
                        training.restore_rng_state(route_rng,device)
                        with pair.preserve_rng(device):
                            function=split_gradient if route=='split' else joint_gradient
                            gradient,trace,values=function(student,teacher,vocoder,discriminator,diffusion,inputs,mean,std,config,check)
                        if probe.tree_sha(training.capture_rng_state())!=route_rng_sha:
                            raise RuntimeError('A gradient route changed the shared random state')
                        if condition_trace is None:
                            condition_trace=trace
                        else:
                            require_same_trace(trace,condition_trace)
                        verify_models(models,original_model_hashes,discriminator_frozen=True)
                        if read_flags()!=flags:
                            raise RuntimeError('Precision flags changed inside a route')
                        record['routes'][route]=dict(values,trace_identities=trace_identities(trace),flags=read_flags())
                        gradients[route]=gradient
                    record['split_minus_joint1']=gradient_difference(gradients['split'],gradients['joint1'])
                    record['joint2_minus_joint1']=gradient_difference(gradients['joint2'],gradients['joint1'])
                    record['flags_after']=read_flags();conditions[condition]=record
                    print('PRECISION3327_FLAGS_AFTER '+json.dumps({'condition':condition,'flags':record['flags_after']}),flush=True)
                    precision_traces[condition]=condition_trace
                    del gradients,gradient
            if read_flags()!=initial_flags:
                raise RuntimeError('Initial precision flags were not restored')
            if {n:cross.identity(v) for n,v in inputs.items()}!=input_identities:
                raise RuntimeError('An original CUDA random/input tensor changed')
            cross_precision={name:difference(precision_traces[CONDITIONS[1]][name],precision_traces[CONDITIONS[0]][name])
                for name in precision_traces[CONDITIONS[0]]}
            if not cross_precision['q950']['actual_exact']:
                raise RuntimeError('The precision toggle changed the supposedly fixed initial q950 input')
    # 恢复边界退出后核全模型、原双Adam/RNG与磁盘文件，再发布成功报告。
    verify_models(models,original_model_hashes,discriminator_frozen=False)
    if probe.tree_sha({'s':payload['student_optimizer'],'d':payload['discriminator_optimizer'],'rng':payload['rng_state']})!=parent_optimizer_rng_sha:
        raise RuntimeError('Original twoAdam/RNG payload changed')
    if read_flags()!=initial_flags or probe.tree_sha(training.capture_rng_state())!=probe.tree_sha(payload['rng_state']):
        raise RuntimeError('Original process precision or parent RNG did not restore')
    resume.verify_files(hashes);progress('all_states_restored_JSON_only')
    result={'status':'completed_readonly_gradient_precision_diagnostic','parent_checkpoint_sha256':d.PARENT_SHA,
        'parent_contract_sha256':d.CONTRACT_SHA,'parent_step':d.STEP,'training_step_used_only_for_inputs':d.NEXT,
        'file_sha256':hashes,'training_indices':indices,'crop_seed':1234+100000+d.NEXT,'noise_seed':1234+200000+d.NEXT,
        'input_identities':input_identities,'score_timesteps':inputs['score_timesteps'].cpu().tolist(),
        'noise_draw_order':['source_noise_CUDA','score_noise_CUDA','score_timesteps_CUDA'],'route_start_rng_sha256':route_rng_sha,
        'initial_process_flags':initial_flags,'final_process_flags':read_flags(),'archived_parent_flags':ORIGINAL_FLAGS,
        'conditions':conditions,'cross_precision_tf32off_minus_on_trace_differences':cross_precision,
        'all_model_states_restored':True,'models_before_after_sha256':original_model_hashes,'source_twoAdam_rng_sha256':parent_optimizer_rng_sha,
        'original_twoAdam_rng_unchanged':True,'optimizer_objects_created':0,'optimizer_updates':0,'model_checkpoint_writes':0,
        'initial_gpu':gpu,'initial_disk':disk,'elapsed_seconds':time.monotonic()-started,
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)},
        'limitations':['No error-size acceptance threshold was introduced; all raw differences and both conditions are retained.',
            'Within each precision condition, all routes use identical prediction/real/fake/target/q950 tensors.',
            'Across precision settings, predictions/waveforms/teacher targets may differ; changes are recorded, not treated as identical objectives.',
            'One TRAIN3328 batch/noise/t draw; this is not evidence that TF32 caused CER errors, D problems or the training trend.',
            'cuDNN deterministic remains false in both conditions; joint repeat quantifies observed repeat variation but does not prove universal determinism.',
            'No optimizer was constructed, no D or S step executed, no candidate or model checkpoint saved, and no metrics were rerun.']}
    encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    if len(encoded)>MAX_JSON_BYTES:
        raise RuntimeError('The result exceeds the2MiB JSON-only budget')
    OUTPUT.mkdir()
    with (OUTPUT/'results.json.tmp').open('xb') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())
    os.replace(OUTPUT/'results.json.tmp',OUTPUT/'results.json')
    print('PRECISION3327_COMPLETED '+str(OUTPUT/'results.json'),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print('PRECISION3327_FAILED '+json.dumps({'status':'failed','error_type':type(error).__name__,'error':str(error),
            'current_precision_flags':read_flags()}),flush=True)
        raise
