# 只在第3227步保存状态上检查当前梯度和有限步幅；不训练、不写候选权重。
"""Read-only matched-D3227 Adam direction with the actual half student LR."""

import argparse  # 主入口需要显式批准；导入时不会访问GPU。
import contextlib  # 无论前向成功还是异常，都恢复临时改变的参数和全部buffer。
import hashlib  # 冻结依赖的实际文件字节必须与登记值相同。
import json  # 只输出轻量数值报告，不写梯度或模型大文件。
import math  # Adam偏差校正和有限值检查。
import os  # 磁盘余量查询与小报告的原子发布。
import platform  # 保留原Python/PyTorch/CUDA环境。
import sys
import time  # 300秒到期后不再尝试其他尺度。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
CROSS_FILE = TOOLS / 'inspect_acdd_teacher_target_cross_v1.py'
CROSS_SHA = 'e2373af110a3e2a828a128bb76621036883b19963a7aa6b27722ef7439bce668'
if hashlib.sha256(CROSS_FILE.read_bytes()).hexdigest() != CROSS_SHA:
    raise RuntimeError('Frozen target-cross helper changed')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch
import inspect_acdd_teacher_target_cross_v1 as cross

pair, base, training, resume, probe = cross.pair, cross.base, cross.training, cross.resume, cross.probe
DEV = cross.DEV
PARENT = cross.PAIRED / 'primary_s1e4/final_full.pt'
PARENT_SHA = 'f98b0763ec5a2edbd23d5d493e9dac549f85cf40cfac71e79f0b462f6fc6beb0'
PARENT_CONTRACT = '25c6e14693a940d94e18a85f50e8e3c08bbfbf1e15f53580117662a0f86ad909'
CROSS_RESULTS = DEV / 'acdd_teacher_target_cross_3227_3247_v1/results.json'
CROSS_RESULTS_SHA = '4a95318d472eac6a5dad246744538f2a9c59fbb8eba81f59366a17be0cd08bca'
PROTOCOL = DEV / 'ACDD3227_ADAM_DIRECTION_PROTOCOL_CN.md'
PROTOCOL_SHA = 'eb14ef5779be13388349b0544677f01366a9acde71821f0ffb75105a4b0189df'
STEP, STUDENT_LR, D_LR = 3227, .0001, .0002
SCALES = (0., .25, .5, 1.)  # 预先确定全部尺度；每次都从同一个3227状态出发。


def validate_saved_adam(saved, model, expected_step, actual_lr):
    # 旧检查器只允许2e-4；仅替换独立参数组副本中的校验值，不碰归档真实LR或moment。
    before = probe.tree_sha(saved)
    if len(saved['param_groups']) != 1 or saved['param_groups'][0]['lr'] != actual_lr:
        raise RuntimeError('Actual role-specific Adam learning rate changed')
    inspection = {'param_groups': [dict(saved['param_groups'][0], lr=.0002)], 'state': saved['state']}
    resume.validate_adam_state(inspection, model, expected_step)
    if probe.tree_sha(saved) != before:
        raise RuntimeError('Validation altered original Adam state')


def adam_candidate(named_parameters, gradients, saved, expected_step):
    # 计算真实1e-4 Adam的float32候选参数；不能先求2e-4已舍入位移再乘0.5。
    groups = saved['param_groups']
    if len(groups) != 1:
        raise RuntimeError('Exactly one inherited student Adam group required')
    group = groups[0]
    required = {'lr': STUDENT_LR, 'betas': (.5, .9), 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False}
    if any(group.get(key) != value for key, value in required.items()) or any(group.get(key, False) for key in ('maximize', 'capturable', 'differentiable', 'fused')):
        raise RuntimeError('Use the actual unchanged half-LR student Adam')
    identifiers = group['params']
    names = [name for name, parameter in named_parameters]
    if len(identifiers) != len(names) or len(set(identifiers)) != len(identifiers) or set(saved['state']) != set(identifiers) or set(gradients) != set(names):
        raise RuntimeError('Adam parameter identities or gradient layout mismatch')
    beta1, beta2 = group['betas']
    candidate, delta = {}, {}
    for (name, parameter), identifier in zip(named_parameters, identifiers):
        original = parameter.detach().cpu()
        state, gradient = saved['state'][identifier], gradients[name]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != expected_step:
            raise RuntimeError('Saved Adam counter or moment fields changed')
        if any(state[key].shape != original.shape or state[key].dtype != original.dtype or not bool(torch.isfinite(state[key]).all()) for key in ('exp_avg', 'exp_avg_sq')) or bool((state['exp_avg_sq'] < 0).any()):
            raise RuntimeError('Saved Adam moment shape/dtype/value changed')
        after = original.clone()
        if gradient is None:
            candidate[name], delta[name] = after, None  # 无梯度时Adam不更新该参数；零梯度仍会受历史动量影响。
            continue
        if gradient.shape != original.shape or gradient.dtype != original.dtype or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('Invalid current parameter gradient')
        first, second = state['exp_avg'].cpu().clone(), state['exp_avg_sq'].cpu().clone()
        first.mul_(beta1).add_(gradient, alpha=1 - beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        step = expected_step + 1
        denominator = (second.sqrt() / math.sqrt(1 - beta2 ** step)).add_(group['eps'])
        after.addcdiv_(first, denominator, value=-group['lr'] / (1 - beta1 ** step))
        if not bool(torch.isfinite(after).all()):
            raise RuntimeError('Nonfinite algebraic Adam candidate')
        candidate[name], delta[name] = after, after - original  # 保留实际float32加法舍入。
    return candidate, delta


def snapshot_model(model):
    # state_dict不含非持久buffer，所以另外逐项保存所有named_buffers。
    return {'state': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            'buffers': {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
            'sha256': pair.model_sha(model)}


def restore_model(model, original):
    model.load_state_dict(original['state'], strict=True)
    buffers = dict(model.named_buffers())
    if set(buffers) != set(original['buffers']):
        raise RuntimeError('Model buffer registration changed')
    with torch.no_grad():
        for name, value in original['buffers'].items():
            buffers[name].copy_(value)
    if pair.model_sha(model) != original['sha256']:
        raise RuntimeError('Temporary student parameters/buffers were not restored exactly')


@contextlib.contextmanager
def trial_state(model, original, candidate, delta, scale):
    if scale not in SCALES:
        raise ValueError('Unregistered scale')
    if pair.model_sha(model) != original['sha256']:
        raise RuntimeError('Each trial must start at the exact3227 state, not the previous candidate')
    try:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if scale == 0. or delta[name] is None:
                    value = original['state'][name]
                elif scale == 1.:
                    value = candidate[name]
                else:
                    value = original['state'][name] + scale * delta[name]
                parameter.copy_(value)  # 只临时改变隔离模型内存，不写回任何归档。
        yield
    finally:
        restore_model(model, original)


def validate_parent(payload, student, discriminator, config):
    if payload['global_step'] != STEP or payload['run_contract_sha256'] != PARENT_CONTRACT or payload['configuration'] != config.as_dict():
        raise RuntimeError('Require the original complete3227 primary state')
    if payload['effective_optimizer_lrs'] != {'student': STUDENT_LR, 'discriminator': D_LR}:
        raise RuntimeError('Actual inherited role-specific learning rates changed')
    validate_saved_adam(payload['student_optimizer'], student, STEP, STUDENT_LR)
    validate_saved_adam(payload['discriminator_optimizer'], discriminator, STEP, D_LR)
    if {id(p) for p in student.parameters()} & {id(p) for p in discriminator.parameters()}:
        raise RuntimeError('Student and discriminator parameters cannot share ownership')


def finite_step_error(scale, values, before, statistics):
    # 比较有限步实际变化与g·delta；A才对应已计算的SG梯度，刷新目标B不能代入这里。
    measured = {'adv':values['adv']-before['adv'], '2fm':2*(values['fm_raw']-before['fm_raw']),
                '45score':45*(values['diagnostic_frozen_target_score']-before['diagnostic_frozen_target_score']),
                'total':values['diagnostic_frozen_target_total']-before['diagnostic_frozen_target_total'],
                'gt_l1':values['train_gt_l1']-before['train_gt_l1']}
    return {name:{'actual_fixed_objective_change':value,
                  'first_order_scale_g_dot_delta':scale*statistics['dot_products'][name]['adam_delta'],
                  'finite_step_residual':value-scale*statistics['dot_products'][name]['adam_delta']}
            for name,value in measured.items()}


def check_limits(started, max_seconds):
    cross.capacity(DEV)  # 每个阶段都重新看真实余量，不只相信启动时的空闲空间。
    if time.monotonic() - started > max_seconds:
        raise TimeoutError('No further trials after the300-second limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--run-approved', action='store_true')
    args = parser.parse_args()
    if not args.run_approved or not 1 <= args.max_seconds <= 300:
        raise RuntimeError('Explicit approval and at most300 seconds required')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)
    if training.git_head(ROOT) != base.EXPECTED_HEAD:
        raise RuntimeError('Frozen clean production source changed')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Original myconda training runtime required')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Registered myconda CUDA interpreter required')
    output = args.output_dir.resolve()
    if output.exists() or output != DEV / 'acdd_adam_descent_readonly_3227_v1':
        raise RuntimeError('Use the exact new directory declared in the protocol')
    disk_before, gpu_before = cross.capacity(DEV), cross.gpu_idle()
    if gpu_before['free_mib'] < 24576:
        raise RuntimeError('Retain at least24GiB freeVRAM for the intact32-example gradient graph')
    started = time.monotonic()
    def check_budget():
        check_limits(started, args.max_seconds)
    def progress(stage):
        check_budget()
        print('ADAM3227 ' + stage, flush=True)
    hashes = {str(CROSS_FILE): CROSS_SHA, str(CROSS_RESULTS): CROSS_RESULTS_SHA, str(PROTOCOL): PROTOCOL_SHA,
              str(PARENT): PARENT_SHA, str(Path(__file__).resolve()): base.sha256_file(Path(__file__))}
    resume.verify_files(hashes)
    cross_record = cross.read_json(CROSS_RESULTS)
    hashes.update(cross_record['file_sha256'])
    resume.verify_files(hashes)
    parent = cross.read_json(cross.PAIRED / 'run_contract.json')
    if training.canonical_json_sha256(parent) != PARENT_CONTRACT:
        raise RuntimeError('Parent contract identity changed')
    grand = cross.read_json(resume.PARENT / 'run_contract.json')
    if training.collect_source_fingerprints(ROOT) != grand['source_sha256']:
        raise RuntimeError('Production source fingerprints changed')
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if config.as_dict() != parent['configuration'] or (config.score_target, config.score_weight, config.lambda_fm, config.lambda_dist) != ('clean_x0', 'sqrt_alpha_bar', 2, 45):
        raise RuntimeError('Score convention, normalization or original coefficients changed')
    raw = resume.load_torch_checkpoint(pair.INPUT_DIR / 'fixed_inputs.pt')
    identities = {name: cross.identity(value) for name, value in raw.items()}
    if identities != cross_record['input_identities'] or {k:v['sha256'] for k,v in identities.items()} != parent['input_sha256']:
        raise RuntimeError('Fixed TRAIN3108 or32DEV input identity changed')
    if raw['x0'].shape != (32,80,128) or raw['bnf'].shape != (32,144,128) or raw['spk_emb'].shape != (32,256):
        raise RuntimeError('Original complete32-example shape required')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    inputs = {name: value.to(device) for name,value in raw.items()}
    del raw
    dev_batch = {key: inputs['dev_'+key] for key in ('mel','bnf','spk_emb')}
    teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
    vocoder, vconfig, _ = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
        base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
    diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = base.load_mel_statistics(ROOT / 'data', device)
    payload = resume.load_torch_checkpoint(PARENT)
    validate_parent(payload, student, discriminator, config)
    student.load_state_dict(payload['student'], strict=True)
    discriminator.load_state_dict(payload['discriminator'], strict=True)  # 匹配D3227，无旧D3107分支。
    student.train(), discriminator.train()
    teacher.eval(), vocoder.eval()
    training.set_requires_grad(student, True)
    training.set_requires_grad(discriminator, False)
    original = snapshot_model(student)
    fixed = {'teacher':teacher, 'vocoder':vocoder, 'D3227':discriminator}
    frozen_hashes = {name:pair.model_sha(model) for name,model in fixed.items()}
    if any(frozen_hashes[name] != parent['fixed_frozen_models_sha256'][name] for name in ('teacher','vocoder')):
        raise RuntimeError('Original teacher/G model or buffers changed')
    pair.assert_frozen(fixed, frozen_hashes)
    state_before = probe.tree_sha({'s':payload['student_optimizer'], 'd':payload['discriminator_optimizer'], 'rng':payload['rng_state']})
    training.restore_rng_state(payload['rng_state'], device)  # 构建网络后恢复原四种随机状态；不替换已存噪声。
    if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(payload['rng_state']):
        raise RuntimeError('Saved3227 RNG did not restore')
    result = None
    try:
        with pair.preserve_rng(device):
            progress('fixed_inputs_and_matched3227_state_ready')
            prediction, q950 = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
            if not torch.equal(q950, inputs['q950']):
                raise RuntimeError('Original exactq950 did not replay')
            leaf = prediction.detach().requires_grad_(True)
            score, target = training.score_distillation_loss(leaf, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
                inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
            if cross.identity(target) != cross_record['states']['3227']['teacher_target'] or cross.identity(leaf) != cross_record['states']['3227']['prediction']:
                raise RuntimeError('Currenty3227/R3227 differ from the verified crossed-target experiment')
            score_y = probe.checked_grad(score, leaf)
            analytic = diffusion.sqrt_alphas_cumprod[inputs['score_timesteps']][:,None,None] * (leaf.detach()-target).sign() / leaf.numel()
            agreement = probe.validate_score_gradient(score_y, analytic)
            progress('score_SG_verified_waveform_VJP')
            wave, adv_y, fm_y = probe.wave_output_gradients(leaf, inputs['x0'], vocoder, discriminator, mean, std, check_budget)
            progress('parameter_components_VJP_to_CPU')
            vectors = {'adv':adv_y, '2fm':2*fm_y, '45score':45*score_y, 'gt_l1':(leaf.detach()-inputs['x0']).sign()/leaf.numel()}
            named = list(student.named_parameters())
            banks = probe.parameter_banks(prediction, named, vectors, check_budget)
            candidate, delta = adam_candidate(named, banks['total'], payload['student_optimizer'], STEP)
            statistics = probe.gradient_statistics(banks, delta)
            before = probe.score_readout(leaf.detach(), inputs['x0'], target, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
                inputs['spk_emb'], inputs['bnf'], vocoder, discriminator, mean, std, config)
            if abs(before['adv']-wave['adv']) > 2e-5 or abs(before['fm_raw']-wave['fm_raw']) > 2e-5 or abs(before['primary_moving_score']-float(score)) > 2e-5 or abs(before['diagnostic_frozen_target_score']-before['primary_moving_score']) > 2e-5:
                raise RuntimeError('Gradient branches or newR3227 baseline disagree with their numeric readout')
            before_dev = base.evaluate_reconstruction(student, diffusion, dev_batch, inputs['dev_source_noise'], config, device)
            dev_replay = probe.validate_development_replay(before_dev, payload['development'])
            if len(before_dev['per_utterance_mel_l1']) != 32 or len(payload['development']['per_utterance_mel_l1']) != 32 or any(abs(a-b)>1e-7 for a,b in zip(before_dev['per_utterance_mel_l1'],payload['development']['per_utterance_mel_l1'])):
                raise RuntimeError('The original32 individualDEV errors did not replay')
            trials = [{'scale':0., 'train':before, 'development':before_dev,
                       'delta_vs_before':{key:0. for key in before}, 'dev_gt_l1_delta_vs_before':0.,
                       'finite_step_error':finite_step_error(0.,before,before,statistics)}]
            del prediction, leaf, q950, score_y, analytic, vectors, adv_y, fm_y, banks, score
            for scale in SCALES[1:]:
                progress('independent_scale_'+str(scale))
                with pair.preserve_rng(device), trial_state(student, original, candidate, delta, scale):
                    with torch.no_grad():
                        changed, _ = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
                    values = probe.score_readout(changed, inputs['x0'], target, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
                        inputs['spk_emb'], inputs['bnf'], vocoder, discriminator, mean, std, config)
                    dev_values = base.evaluate_reconstruction(student, diffusion, dev_batch, inputs['dev_source_noise'], config, device)
                    trials.append({'scale':scale, 'train':values, 'development':dev_values,
                        'delta_vs_before':{key:values[key]-before[key] for key in values},
                        'dev_gt_l1_delta_vs_before':dev_values['mel_l1']-before_dev['mel_l1'],
                        'finite_step_error':finite_step_error(scale,values,before,statistics)})
                    del changed
                pair.assert_frozen(fixed, frozen_hashes)
            if [row['scale'] for row in trials] != list(SCALES):
                raise RuntimeError('All predeclared scales must be reported')
            result = {'status':'completed', 'kind':'readonly_3227_adam_direction', 'scope':'theta-only at matched fixedD3227; not a production alternating step',
                'parent_checkpoint_sha256':PARENT_SHA, 'parent_contract_sha256':PARENT_CONTRACT, 'file_sha256':hashes,
                'cross_results_sha256':CROSS_RESULTS_SHA, 'input_archive':str(pair.INPUT_DIR/'fixed_inputs.pt'), 'input_identities':identities,
                'fixed_training_batch':cross_record['fixed_training_batch'], 'score_timesteps':inputs['score_timesteps'].cpu().tolist(),
                'fixed_target_anchor_step':STEP, 'teacher_target':cross.identity(target), 'matched_discriminator_step':STEP,
                'saved_adam_step':STEP, 'candidate_adam_step':STEP+1, 'actual_optimizer_lrs':{'student':STUDENT_LR,'discriminator':D_LR},
                'parameter_gradients':statistics, 'score_analytic_gradient_agreement':agreement, 'before_train':before, 'before_dev':before_dev,
                'development_replay_absolute_differences':dev_replay, 'trials':trials, 'scales':list(SCALES),
                'models_before_after_sha256':dict(frozen_hashes, student=original['sha256']), 'source_optimizer_rng_sha256':state_before,
                'runtime':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'executable':str(Path(sys.executable).resolve()),
                    'student_training_mode':student.training,'discriminator_training_mode':discriminator.training,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)},
                'initial_gpu':gpu_before,'initial_disk':disk_before,
                'limitations':['A fixes the freshly verifiedR3227 target; it is not the olderR3107 diagnostic.',
                    'D3227 never updates. SavedAdam moments are used algebraically, but no optimizer exists and no model checkpoint is written.',
                    'The current SG gradient need not equal the derivative of refreshed-target B; no teacher Jacobian is used.',
                    'A first-order dot product is local; finiteL1 kinks, curvature and momentum can affect finite trial steps.',
                    'Separate component parameterVJPs and CPUfloat32 Adam have possible small rounding differences from an actualCUDA joint backward/update.',
                    'Only oneTRAIN32/noise/t draw and fixed32DEV; no CER/UTMOS/DNSMOS inference or generalization guarantee.']}
    finally:
        restore_model(student, original)
        pair.assert_frozen(fixed, frozen_hashes)
        if any(p.grad is not None for p in student.parameters()):
            raise RuntimeError('The diagnostic unexpectedly populated student parameter.grad')
        if probe.tree_sha({'s':payload['student_optimizer'],'d':payload['discriminator_optimizer'],'rng':payload['rng_state']}) != state_before:
            raise RuntimeError('Saved twoAdam/RNG payload changed')
        if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(payload['rng_state']):
            raise RuntimeError('Saved runtimeRNG was not preserved')
    if {name:cross.identity(value) for name,value in inputs.items()} != identities:
        raise RuntimeError('An original input tensor changed')
    resume.verify_files(hashes)
    progress('all_states_restored_writing_small_report')
    result.update(elapsed_seconds=time.monotonic()-started, temporary_models_restored=True, original_optimizer_rng_unchanged=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    if len(encoded)>cross.MAX_JSON_BYTES:
        raise RuntimeError('Report exceeds the1MiB small-output budget')
    output.mkdir()
    with (output/'results.json.tmp').open('xb') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())
    os.replace(output/'results.json.tmp',output/'results.json')
    print('ADAM3227_COMPLETED '+str(output/'results.json'),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('ADAM3227_FAILED '+json.dumps({'status':'failed','error_type':type(error).__name__,'error':str(error)},ensure_ascii=False),flush=True)
        raise
