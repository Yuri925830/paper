"""在隔离内存检查一次D更新怎样改变学生下一步；不续训、不保存模型。"""

import argparse  # GPU诊断必须由主线程明确批准。
import contextlib  # 包括异常路径在内，还原模型、优化器和随机状态。
import copy  # 优化器加载独立副本，不能让CPU张量与原归档共享写入。
import hashlib  # 读取真实文件与张量身份，不凭文件名确认模型。
import json  # 唯一新增产物是小型JSON报告。
import math  # 比较梯度数值以及有限值。
import os  # 原子发布JSON；不提供删除或模型保存入口。
import platform  # 保留原训练Python、PyTorch和CUDA版本。
import sys
import time  # 整项检查最多300秒，不超时继续追加对照。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute this diagnostic only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
HELPER = TOOLS / 'inspect_acdd_adam_3227_v1.py'
HELPER_SHA = 'c193ad8b8a96eda8058856c4946bf207631bb8d6b6ee5940770740a1c7a8212e'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Frozen state/gradient helper changed')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch  # 只有main会分配CUDA；CPU测试用很小的模型。
import inspect_acdd_adam_3227_v1 as old

pair, base, training, resume, probe, cross = old.pair, old.base, old.training, old.resume, old.probe, old.cross
DEV = cross.DEV
RUN = DEV / 'acdd_s5e5_staged120_from3227_v1'
PARENT = RUN / 'final_full.pt'
PARENT_SHA = '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556'
CONTRACT_SHA = '32651a26cdf7074a9e8ca30c3cdd099697541fa4b261df757d2866f9e0622aba'
PROTOCOL = DEV / 'ACDD3327_D_UPDATE_PROBE_PROTOCOL_CN.md'
PROTOCOL_SHA = 'c2b38d48bf6b7b1b15e68902fc0fe652b0934d6cb1f765c8d583c2ad42c23d04'
OUTPUT = DEV / 'acdd_d_update_readonly_3327_v1'
STEP, NEXT = 3327, 3328
LRS = {'student': .00005, 'discriminator': .0002}
STUDENTS, DISCRIMINATORS = ('original', 'candidate_D0', 'candidate_D1'), ('D0', 'D1')
MAX_JSON_BYTES = 2 * 1024**2  # 包含全部3×2条件和3组32句结果，不写权重或梯度文件。
PINNED = {
    str(HELPER): HELPER_SHA, str(PROTOCOL): PROTOCOL_SHA, str(PARENT): PARENT_SHA,
    str(RUN / 'run_contract.json'): '787759339c886b2918bdda62abcc6818adf3cb410f923aca38c80bbd4b99b071',
    str(RUN / 'completion.json'): '68ee1cf0e680f3bf665be6dcf46e2aa75c3684262c2da3765ae2c7beebf931b2',
    str(RUN / 'student_step_003327.pt'): '7bb68a0d053335a87a2c3c42684b864e27bf21a9f19d6f8d1b5bc8152955ca60',
    str(pair.INPUT_DIR / 'fixed_inputs.pt'): '296a7b4a7c289f0d19409947cd2c4dcb54754ae5636a58af43e658bc12b17da8',
}


def snapshot(model):
    # 非持久Hann窗口、子层模式、requires_grad和已有grad都不能被state_dict漏掉。
    value = old.snapshot_model(model)
    value['modes'] = {n: m.training for n, m in model.named_modules()}
    value['requires_grad'] = {n: p.requires_grad for n, p in model.named_parameters()}
    value['grads'] = {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
    return value


def restore(model, value):
    old.restore_model(model, value)  # 参数和全部named_buffers先恢复并核SHA。
    for name, module in model.named_modules():
        module.training = value['modes'][name]  # 逐子层恢复，不用顶层train覆盖不同子层模式。
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(value['requires_grad'][name])
        gradient = value['grads'][name]
        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()


def restore_optimizer(optimizer, saved):
    optimizer.load_state_dict(copy.deepcopy(saved))  # 防止load_state_dict在CPU时保留可变moment别名。
    if probe.tree_sha(optimizer.state_dict()) != probe.tree_sha(saved):
        raise RuntimeError('Optimizer restore did not preserve every moment, counter and option')


def cpu_copy(value):
    # Adam快照只放CPU，避免为了异常恢复在GPU额外复制两套大moment。
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


@contextlib.contextmanager
def restore_everything(models, optimizers, saved_rng, device):
    # 此边界内部只有隔离内存被暂改；失败时也必须回到进入前的完整状态。
    originals = {name: snapshot(model) for name, model in models.items()}
    saved_optimizers = {name: cpu_copy(opt.state_dict()) for name, opt in optimizers.items()}
    training.restore_rng_state(saved_rng, device)
    try:
        yield originals
    finally:
        for name, model in models.items():
            restore(model, originals[name])
        for name, optimizer in optimizers.items():
            restore_optimizer(optimizer, saved_optimizers[name])
        training.restore_rng_state(saved_rng, device)
        if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(saved_rng):
            raise RuntimeError('The original Python/NumPy/CPU/CUDA RNG did not restore')


def actual_adam_step(model, optimizer, expected_step):
    # 直接调用原torch.Adam；None不推进该参数，零梯度仍保留历史moment的作用。
    before = {id(p): float(optimizer.state[p]['step']) for p in model.parameters()}
    gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
    if any(v != expected_step for v in before.values()):
        raise RuntimeError('Candidate must begin at the original per-parameter Adam counters')
    active = [v for v in gradients.values() if v is not None]
    if not active or any(not bool(torch.isfinite(v).all()) for v in active):
        raise RuntimeError('Missing or nonfinite current Adam gradients')
    optimizer.step()  # 仅本诊断的临时模型；不调用train_batch，不写任何断点。
    counters = {}
    for name, parameter in model.named_parameters():
        expected = expected_step + int(gradients[name] is not None)
        counters[name] = float(optimizer.state[parameter]['step'])
        if counters[name] != expected or not bool(torch.isfinite(parameter).all()):
            raise RuntimeError('Actual Adam counter or candidate parameter is invalid')
        if any(not bool(torch.isfinite(optimizer.state[parameter][key]).all()) for key in ('exp_avg', 'exp_avg_sq')):
            raise RuntimeError('Actual Adam produced nonfinite moments')
    optimizer.zero_grad(set_to_none=True)
    return gradients, {'counters': counters, 'lr': optimizer.param_groups[0]['lr'],
        'state_sha256': probe.tree_sha(optimizer.state_dict()), 'none_parameters': [n for n, v in gradients.items() if v is None]}


def d_update(discriminator, optimizer, real, fake):
    # D公式、分支顺序与一次step直接来自生产函数，绝不同时更新学生。
    if real.requires_grad or fake.requires_grad or real.shape != fake.shape:
        raise RuntimeError('D must receive matching detached full-batch waveforms')
    training.set_requires_grad(discriminator, True)
    optimizer.zero_grad(set_to_none=True)
    loss = training.serial_discriminator_backward(discriminator, real, fake)
    gradients, record = actual_adam_step(discriminator, optimizer, STEP)
    del gradients  # D梯度不落盘，也不留到学生反向阶段。
    training.set_requires_grad(discriminator, False)
    return dict(record, loss_before_update=float(loss))


def bank_difference(left, right, *, tight=False):
    # 分项VJP与联合backward可能有浮点累加差；真正候选始终使用联合backward。
    if set(left) != set(right):
        raise RuntimeError('Parameter gradient names differ')
    maximum, error2, reference2 = 0., 0., 0.
    for name in left:
        a, b = left[name], right[name]
        if a is None or b is None:
            if a is not None or b is not None:
                raise RuntimeError('None-gradient support changed: ' + name)
            continue
        if a.shape != b.shape or a.dtype != b.dtype or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise RuntimeError('Invalid gradient comparison: ' + name)
        error = a.double() - b.double()
        maximum = max(maximum, float(error.abs().max()))
        error2 += float(error.square().sum())
        reference2 += float(a.double().square().sum())
    relative = math.sqrt(error2 / reference2) if reference2 else (0. if not error2 else float('inf'))
    # 同S不同D的score/GT参数梯度只容许很紧的反传舍入差；不是把较大的差异解释成D影响。
    limit = 1e-6 if tight else 1e-4
    if relative > limit:
        raise RuntimeError('Gradient invariant failed: max=%g relativeL2=%g' % (maximum, relative))
    return {'max_absolute_difference': maximum, 'relative_l2_difference': relative,
            'actual_exact': maximum == 0., 'relative_l2_limit': limit, 'cross_D_score_GT_invariant': tight}


def validate_parent(payload, student, discriminator, config, contract, receipt):
    if training.canonical_json_sha256(contract) != CONTRACT_SHA or contract['configuration'] != config.as_dict():
        raise RuntimeError('Original3327 training contract or configuration changed')
    if (payload['global_step'], payload['run_contract_sha256'], payload['effective_optimizer_lrs']) != (STEP, CONTRACT_SHA, LRS):
        raise RuntimeError('Require the actual complete3327 state with S5e-5/D2e-4')
    if payload['configuration'] != config.as_dict() or payload['score_semantics'] != 'reconstruction_clean_x0_sqrt_alpha_bar':
        raise RuntimeError('The original score configuration changed')
    if receipt['completed_global_step'] != STEP or receipt['last_full_saved_global_step'] != STEP or receipt['partial_batch_may_have_updated_D']:
        raise RuntimeError('The parent is not a complete stopped batch boundary')
    if receipt['full_snapshot']['sha256'] != PARENT_SHA or payload['development'] != receipt['last_monitor']['development']:
        raise RuntimeError('Full state and its original fixedDEV receipt disagree')
    old.validate_saved_adam(payload['student_optimizer'], student, STEP, LRS['student'])
    old.validate_saved_adam(payload['discriminator_optimizer'], discriminator, STEP, LRS['discriminator'])
    if {id(p) for p in student.parameters()} & {id(p) for p in discriminator.parameters()}:
        raise RuntimeError('Student and D cannot share parameter ownership')


def component_gradients(student, teacher, vocoder, discriminator, diffusion, inputs, mean, std, config, check):
    prediction, _ = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
    leaf = prediction.detach().requires_grad_(True)
    score, target = training.score_distillation_loss(leaf, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
        inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
    score_y = probe.checked_grad(score, leaf)
    analytic = diffusion.sqrt_alphas_cumprod[inputs['score_timesteps']][:, None, None] * (leaf.detach() - target).sign() / leaf.numel()
    agreement = probe.validate_score_gradient(score_y, analytic)
    wave, adv_y, fm_y = probe.wave_output_gradients(leaf, inputs['x0'], vocoder, discriminator, mean, std, check)
    vectors = {'adv': adv_y, '2fm': 2 * fm_y, '45score': 45 * score_y, 'gt_l1': (leaf.detach() - inputs['x0']).sign() / leaf.numel()}
    banks = probe.parameter_banks(prediction, list(student.named_parameters()), vectors, check)
    return banks, target.detach(), {'prediction': cross.identity(leaf), 'target': cross.identity(target),
        'score_y': cross.identity(score_y), 'gt_y': cross.identity(vectors['gt_l1']), 'analytic_agreement': agreement,
        'score': float(score), 'adv': wave['adv'], 'fm_raw': wave['fm_raw']}


def s_update(student, teacher, vocoder, discriminator, diffusion, inputs, real, fake, target, mean, std, config, optimizer):
    # 以下顺序与生产学生阶段相同：合并wave VJP，再通过G/S与45score一次联合反传。
    training.set_requires_grad(discriminator, False)
    optimizer.zero_grad(set_to_none=True)
    adv, fm, waveform_gradient = training.serial_student_discriminator_gradient(discriminator, real, fake, config.lambda_fm)
    prediction, _ = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
    waveform = vocoder(training.denormalize_mel(prediction, mean, std))
    if not torch.equal(waveform.detach(), fake):
        raise RuntimeError('Production student waveform did not replay the original full-batch proxy')
    score, actual_target = training.score_distillation_loss(prediction, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
        inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
    if not torch.equal(actual_target, target):
        raise RuntimeError('A changed D must not change the current S3327 teacher target')
    torch.autograd.backward((waveform, score), (waveform_gradient, score.new_tensor(config.lambda_dist)))
    if any(p.grad is not None for model in (teacher, vocoder, discriminator) for p in model.parameters()):
        raise RuntimeError('Student backward populated frozen model gradients')
    if not any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in student.parameters()):
        raise RuntimeError('The production student step must not accept all-zero gradients')
    gradients, adam = actual_adam_step(student, optimizer, STEP)
    return gradients, dict(adam, losses={'adv': float(adv), 'fm_raw': float(fm), 'dist_raw': float(score),
        'total': float(adv + 2 * fm + 45 * score)})


def matrix_differences(matrix, development):
    # 每列固定同一D，比候选和原学生；另外完整保留D1候选减D0候选。
    if set(matrix) != set(STUDENTS) or any(set(v) != set(DISCRIMINATORS) for v in matrix.values()) or set(development) != set(STUDENTS):
        raise RuntimeError('All three students and both discriminator columns are required')
    changes = {s: {d: {k: value - matrix['original'][d][k] for k, value in matrix[s][d].items()}
                   for d in DISCRIMINATORS} for s in STUDENTS[1:]}
    between = {d: {k: matrix['candidate_D1'][d][k] - matrix['candidate_D0'][d][k]
                  for k in matrix['original'][d]} for d in DISCRIMINATORS}
    return {'candidate_minus_original_same_D': changes, 'candidate_D1_minus_candidate_D0_same_D': between,
        'development_D1_minus_D0': development['candidate_D1']['mel_l1'] - development['candidate_D0']['mel_l1'],
        'development_candidate_minus_original': {s: development[s]['mel_l1'] - development['original']['mel_l1'] for s in STUDENTS[1:]}}


def check_limits(started, maximum):
    cross.capacity(DEV)  # 3GiB系统保留量与128MiB缓冲始终不挪用。
    if time.monotonic() - started > maximum:
        raise TimeoutError('The predeclared300-second diagnostic budget is exhausted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--run-approved', action='store_true')
    args = parser.parse_args()
    if not args.run_approved or not 1 <= args.max_seconds <= 300:
        raise RuntimeError('Explicit approval and at most300 seconds are required')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)  # 三把原工作流锁保持到main结束，不与评分或训练并行。
    if args.output_dir.is_symlink() or args.output_dir.resolve() != OUTPUT or OUTPUT.exists():
        raise RuntimeError('Use only the predeclared new output directory')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('The original training runtime is required')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Use the registered myconda CUDA interpreter')
    if training.git_head(ROOT) != base.EXPECTED_HEAD:
        raise RuntimeError('Frozen production source changed')
    disk, gpu = cross.capacity(DEV), cross.gpu_idle()
    if gpu['free_mib'] < 24576:
        raise RuntimeError('Require at least24GiB free VRAM for unchanged batch32')
    started = time.monotonic()
    def check():
        check_limits(started, args.max_seconds)
    def progress(stage):
        check()
        print('D3327 ' + stage, flush=True)
    hashes = dict(PINNED)
    hashes[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
    resume.verify_files(hashes)
    contract, receipt = cross.read_json(RUN / 'run_contract.json'), cross.read_json(RUN / 'completion.json')
    if training.canonical_json_sha256(contract) != CONTRACT_SHA:
        raise RuntimeError('The actual3327 contract changed')
    hashes.update(contract['file_sha256'])
    development, development_hashes = resume.load_development(DEV / 'development_contract.json')
    hashes.update(development_hashes)
    resume.verify_files(hashes)
    grand = cross.read_json(resume.PARENT / 'run_contract.json')
    if training.collect_source_fingerprints(ROOT) != grand['source_sha256']:
        raise RuntimeError('Production source fingerprints changed')
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if config.as_dict() != contract['configuration'] or (config.score_target, config.score_weight, config.lambda_fm, config.lambda_dist) != ('clean_x0', 'sqrt_alpha_bar', 2, 45):
        raise RuntimeError('Original objective or coefficients changed')
    if contract['training_indices'] != development['retained_train_indices'] or contract['excluded_train_indices'] != development['excluded_train_indices']:
        raise RuntimeError('The retained training pool or52 exclusions changed')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
    vocoder, vconfig, _ = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt', base.VOCODER_DIR / 'config.json',
        base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
    diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = base.load_mel_statistics(ROOT / 'data', device)
    payload = resume.load_torch_checkpoint(PARENT)
    validate_parent(payload, student, discriminator, config, contract, receipt)
    student.load_state_dict(payload['student'], strict=True)
    discriminator.load_state_dict(payload['discriminator'], strict=True)
    student.train(); discriminator.train(); teacher.eval(); vocoder.eval()
    training.set_requires_grad(student, True); training.set_requires_grad(discriminator, True)
    opts = {'student': torch.optim.Adam(student.parameters(), lr=LRS['student'], betas=(.5, .9)),
            'discriminator': torch.optim.Adam(discriminator.parameters(), lr=LRS['discriminator'], betas=(.5, .9))}
    for name, opt in opts.items():
        restore_optimizer(opt, payload[name + '_optimizer'])
    saved_state_sha = probe.tree_sha({'s': payload['student_optimizer'], 'd': payload['discriminator_optimizer'], 'rng': payload['rng_state']})
    frozen = {'teacher': teacher, 'vocoder': vocoder}
    frozen_sha = {n: pair.model_sha(m) for n, m in frozen.items()}
    if any(frozen_sha[n] != contract['fixed_frozen_models_sha256'][n] for n in frozen):
        raise RuntimeError('The original frozen teacher or vocoder changed')
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    indices = resume.GlobalBatchPlan(development['retained_train_indices']).for_step(NEXT)
    if len(indices) != 32 or set(indices) & set(development['excluded_train_indices']):
        raise RuntimeError('Next absolute batch must contain32 retained training utterances')
    result = None
    with restore_everything({'student': student, 'discriminator': discriminator, **frozen}, opts, payload['rng_state'], device) as original:
        with pair.preserve_rng(device):
            batch = base.batch_for(dataset, indices, 1234 + 100000 + NEXT)
            batch = training._slice_batch(batch, device)
            x0 = batch['mel']
            training.seed_everything(1234 + 200000 + NEXT)
            inputs = {'x0': x0, 'bnf': batch['bnf'], 'spk_emb': batch['spk_emb'],
                'source_noise': torch.randn_like(x0), 'score_noise': torch.randn_like(x0),
                'score_timesteps': torch.randint(1, 1001, (32,), device=device, dtype=torch.long)}
            # 使用原归档DEV张量与55001噪声，并逐项与原固定输入身份核对。
            archived = resume.load_torch_checkpoint(pair.INPUT_DIR / 'fixed_inputs.pt')
            for name in ('dev_mel', 'dev_bnf', 'dev_spk_emb', 'dev_source_noise'):
                if cross.identity(archived[name])['sha256'] != contract['fixed_train_inputs_sha256'][name]:
                    raise RuntimeError('Original32DEV input identity changed: ' + name)
                inputs[name] = archived[name].to(device)
            del archived
            dev_batch = {k: inputs['dev_' + k] for k in ('mel', 'bnf', 'spk_emb')}
            identities = {n: cross.identity(v) for n, v in inputs.items()}
            progress('parent_and_next3328_inputs_ready')
            with torch.no_grad():
                y0, q950 = training.make_student_prediction(student, diffusion, x0, inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
                real, fake = vocoder(training.denormalize_mel(x0, mean, std)), vocoder(training.denormalize_mel(y0, mean, std))
            identities['q950'] = cross.identity(q950)
            del q950
            students, discriminators = {'original': original['student']}, {'D0': original['discriminator']}
            arm_records, banks_for_cross, deltas = {}, {}, {}
            target = None
            for role in DISCRIMINATORS:
                progress(role + '_original_state_and_gradients')
                restore(student, original['student']); restore(discriminator, original['discriminator'])
                restore_optimizer(opts['student'], payload['student_optimizer'])
                restore_optimizer(opts['discriminator'], payload['discriminator_optimizer'])
                if role == 'D1':
                    with pair.preserve_rng(device):  # D临时试算不推进比较所用的原随机流。
                        d_record = d_update(discriminator, opts['discriminator'], real, fake)
                    discriminators['D1'] = snapshot(discriminator)
                else:
                    d_record = {'optimizer_updates': 0, 'state_sha256': probe.tree_sha(opts['discriminator'].state_dict()), 'lr': LRS['discriminator']}
                training.set_requires_grad(discriminator, False)
                with pair.preserve_rng(device):
                    banks, current_target, components = component_gradients(student, teacher, vocoder, discriminator, diffusion, inputs, mean, std, config, check)
                    if target is None:
                        target = current_target
                    elif not torch.equal(target, current_target):
                        raise RuntimeError('D1 changed the target although S and all inputs are fixed')
                    if role == 'D1':
                        for key in ('prediction', 'target', 'score_y', 'gt_y'):
                            if components[key] != arm_records['D0']['components'][key]:
                                raise RuntimeError('D condition changed a shared input/score/GT gradient: ' + key)
                        invariant = {key: bank_difference(banks_for_cross[key], banks[key], tight=True) for key in ('45score', 'gt_l1')}
                    else:
                        invariant = {}
                    progress(role + '_production_joint_backward_and_actual_S_Adam')
                    actual, adam_record = s_update(student, teacher, vocoder, discriminator, diffusion, inputs, real, fake, target, mean, std, config, opts['student'])
                    agreement = bank_difference(actual, banks['total'])
                    banks['total'] = actual  # 后续内积使用实际联合backward，分项仅用于解释。
                    candidate = snapshot(student)
                    delta = {n: None if actual[n] is None else candidate['state'][n] - original['student']['state'][n] for n in actual}
                    students['candidate_' + role], deltas[role] = candidate, delta
                    statistics = probe.gradient_statistics(banks, delta)
                    arm_records[role] = {'D_update': d_record, 'S_update': adam_record, 'components': components,
                        'parameter_gradients': statistics, 'component_sum_vs_actual_joint': agreement, 'D_invariant_score_GT': invariant}
                    if role == 'D0':
                        banks_for_cross = {key: banks[key] for key in ('45score', 'gt_l1')}
                    banks_for_cross[role + '_total'] = actual
                    del banks
                restore(student, original['student'])
                pair.assert_frozen(frozen, frozen_sha)
            progress('all_candidates_created_common_D0_D1_readouts')
            matrix, dev_results, predictions = {}, {}, {}
            for label in STUDENTS:
                restore(student, students[label])
                with pair.preserve_rng(device), torch.no_grad():
                    prediction, _ = training.make_student_prediction(student, diffusion, x0, inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
                    predictions[label] = cross.identity(prediction)
                    dev_results[label] = base.evaluate_reconstruction(student, diffusion, dev_batch, inputs['dev_source_noise'], config, device)
                    matrix[label] = {}
                    for role in DISCRIMINATORS:
                        progress(label + '_readout_' + role)
                        restore(discriminator, discriminators[role])
                        matrix[label][role] = probe.score_readout(prediction, x0, target, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
                            inputs['spk_emb'], inputs['bnf'], vocoder, discriminator, mean, std, config)
                if len(dev_results[label]['per_utterance_mel_l1']) != 32:
                    raise RuntimeError('All32 individualDEV errors must be retained')
            replay = probe.validate_development_replay(dev_results['original'], payload['development'])
            if max(abs(a-b) for a,b in zip(dev_results['original']['per_utterance_mel_l1'], payload['development']['per_utterance_mel_l1'])) > 1e-7:
                raise RuntimeError('Original32 individualDEV errors did not replay')
            direction = probe.gradient_statistics({'D0_actual_total': banks_for_cross['D0_total'], 'D1_actual_total': banks_for_cross['D1_total'],
                                                   'D0_actual_delta': deltas['D0'], 'D1_actual_delta': deltas['D1']})
            differences = matrix_differences(matrix, dev_results)
            if {n: cross.identity(v) for n,v in inputs.items()} != {n:v for n,v in identities.items() if n != 'q950'}:
                raise RuntimeError('A fixed input changed during the probe')
            result = {'status': 'completed_readonly_D_update_probe', 'parent_step': STEP, 'next_batch_step': NEXT,
                'parent_checkpoint_sha256': PARENT_SHA, 'parent_contract_sha256': CONTRACT_SHA, 'file_sha256': hashes,
                'input_identities': identities, 'training_indices': indices, 'crop_seed': 1234+100000+NEXT, 'noise_seed': 1234+200000+NEXT,
                'noise_draw_order': ['source_noise_CUDA', 'score_noise_CUDA', 'score_timesteps_CUDA'], 'score_timesteps': inputs['score_timesteps'].cpu().tolist(),
                'original_fixed32DEV_source_noise_seed': 55001, 'target_anchor': 'fresh fixed R3327 at TRAIN3328, independent of D',
                'fixed_target_identity': cross.identity(target), 'actual_optimizer_lrs': LRS,
                'temporary_optimizer_updates': {'D': 1, 'S': 2}, 'arms': arm_records, 'matrix_3S_by_2D': matrix,
                'fixed32_development': dev_results, 'development_replay_absolute_differences': replay, 'differences': differences,
                'directions': direction, 'student_prediction_identities': predictions,
                'model_state_sha256': {'students': {n:s['sha256'] for n,s in students.items()}, 'discriminators': {n:s['sha256'] for n,s in discriminators.items()}, **frozen_sha},
                'source_twoAdam_rng_sha256': saved_state_sha, 'initial_gpu': gpu, 'initial_disk': disk,
                'limitations': ['One retainedTRAIN32 batch and one noise/t draw; not the entire training curve or a CER/quality cause.',
                    'S candidates use actual production joint backward and torch.Adam; split parameter VJPs only describe components.',
                    'A fixes freshR3327; B recomputes teacher targets and has no teacher Jacobian in the update.',
                    'All threeS/bothD columns and all32DEV errors are reported; no scale or condition was selected after seeing values.',
                    'No original model, optimizer, source code, metric definition, training directory or checkpoint was written.']}
    # 退出恢复边界后再次核验原状态、teacher/G、输入来源和归档文件。
    pair.assert_frozen(frozen, frozen_sha)
    if probe.tree_sha({'s': payload['student_optimizer'], 'd': payload['discriminator_optimizer'], 'rng': payload['rng_state']}) != saved_state_sha:
        raise RuntimeError('The immutable in-memory parent payload changed')
    resume.verify_files(hashes)
    progress('restored_all_states_writing_small_JSON_only')
    result.update(elapsed_seconds=time.monotonic()-started, all_temporary_states_restored=True,
        original_twoAdam_rng_unchanged=True, original_model_files_unchanged=True, model_checkpoint_writes=0,
        runtime={'python':platform.python_version(), 'torch':torch.__version__, 'cuda':torch.version.cuda,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(device), 'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_JSON_BYTES:
        raise RuntimeError('The result exceeds the2MiB JSON-only budget')
    OUTPUT.mkdir()
    with (OUTPUT/'results.json.tmp').open('xb') as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    os.replace(OUTPUT/'results.json.tmp', OUTPUT/'results.json')
    print('D3327_COMPLETED ' + str(OUTPUT/'results.json'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('D3327_FAILED ' + json.dumps({'status':'failed', 'error_type':type(error).__name__, 'error':str(error)}), flush=True)
        raise
