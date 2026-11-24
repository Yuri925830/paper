"""只在隔离内存比较一次D更新；学生使用真实联合反传，不估算分项梯度。"""

import argparse  # CPU导入不会运行GPU；正式诊断仍须显式批准。
import contextlib  # 异常时也恢复四个模型、双Adam、精度标志和RNG。
import hashlib  # 同名helper必须与已经审核的文件逐字节相同。
import json  # 只发布小报告，不发布模型或完整梯度。
import math  # 范数和余弦只描述真实梯度，没有事后调整的过关阈值。
import os  # 小JSON使用排他创建和原子重命名，不删除旧产物。
import platform
import sys
import time  # 300秒包含输入、全部对照、恢复和验真。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
HELPER = TOOLS / 'inspect_acdd_d_update_3327_v1.py'
HELPER_SHA = 'c14d0997173bb734ae9d461d95d1637d2b7503725bbcc474febdb04e12c2cd49'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Frozen production-step helper changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import inspect_acdd_d_update_3327_v1 as d

pair, base, training, resume, probe, cross = d.pair, d.base, d.training, d.resume, d.probe, d.cross
DEV = d.DEV
PROTOCOL = DEV / 'ACDD3327_D_JOINT_ONLY_PROTOCOL_CN.md'
PROTOCOL_SHA = '2e63908395d9223eeaa353b904e90ca1af64d3831ad7b1a7dfbc40603a88904a'
OUTPUT = DEV / 'acdd_d_joint_only_readonly_3327_v1'
ORIGINAL_FLAGS = {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True,
                  'cudnn_benchmark': False, 'cudnn_deterministic': False}
MAX_JSON_BYTES = 2 * 1024**2


def read_flags():
    return {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic}


def restore_flags(values):
    # 只在退出边界恢复进入前值；正常试验不切换任何精度设置。
    torch.backends.cuda.matmul.allow_tf32 = values['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = values['cudnn_allow_tf32']
    torch.backends.cudnn.benchmark = values['cudnn_benchmark']
    torch.backends.cudnn.deterministic = values['cudnn_deterministic']


def require_original_flags():
    values = read_flags()
    if values != ORIGINAL_FLAGS:
        raise RuntimeError('The four original precision flags must remain unchanged: ' + repr(values))
    return values


def verify_snapshot(model, expected):
    # model_sha涵盖全部named_buffers，包括state_dict不保存的窗口。
    if pair.model_sha(model) != expected['sha256']:
        raise RuntimeError('Model parameters or a persistent/nonpersistent buffer changed')
    if {n: m.training for n, m in model.named_modules()} != expected['modes']:
        raise RuntimeError('A submodule training/evaluation mode failed to restore')
    if {n: p.requires_grad for n, p in model.named_parameters()} != expected['requires_grad']:
        raise RuntimeError('Parameter requires_grad flags failed to restore')
    for name, parameter in model.named_parameters():
        wanted, actual = expected['grads'][name], parameter.grad
        if wanted is None or actual is None:
            if wanted is not None or actual is not None:
                raise RuntimeError('Parameter grad/None state failed to restore: ' + name)
        elif not torch.equal(actual.detach().cpu(), wanted):
            raise RuntimeError('An original parameter gradient failed to restore: ' + name)


@contextlib.contextmanager
def restored_state(models, optimizers, saved_rng, device):
    flags = require_original_flags()
    adam_hashes = {n: probe.tree_sha(opt.state_dict()) for n, opt in optimizers.items()}
    originals = None
    try:
        with d.restore_everything(models, optimizers, saved_rng, device) as originals:
            yield originals
    finally:
        # 即使内层前向、Adam或报告构造失败，也不遗留模式、梯度或精度切换。
        restore_flags(flags)
        if read_flags() != flags:
            raise RuntimeError('Precision flags failed to restore')
        if originals is not None:
            for name, model in models.items():
                verify_snapshot(model, originals[name])
            if {n: probe.tree_sha(opt.state_dict()) for n, opt in optimizers.items()} != adam_hashes:
                raise RuntimeError('An original Adam moment/counter/option failed to restore')
            if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(saved_rng):
                raise RuntimeError('Original Python/NumPy/CPU/CUDA RNG failed to restore')
        print('DJOINT3327_RESTORED ' + json.dumps({'flags': read_flags(), 'states_verified': originals is not None}), flush=True)


@torch.no_grad()
def shared_readout(student, teacher, vocoder, diffusion, inputs, mean, std, config, fixed_target=None):
    # 同一学生只生成一份y、波形和R，再交给两个D；更新后的学生允许有不同的R。
    prediction, q950 = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
    moving, target = training.score_distillation_loss(prediction, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'],
        inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight)
    anchor = target if fixed_target is None else fixed_target
    weights = diffusion.sqrt_alphas_cumprod[inputs['score_timesteps']]
    fixed = (weights * (prediction-anchor).abs().flatten(1).mean(1)).mean()
    drift = (weights * (target-anchor).abs().flatten(1).mean(1)).mean()
    real = vocoder(training.denormalize_mel(inputs['x0'], mean, std))
    fake = vocoder(training.denormalize_mel(prediction, mean, std))
    # 这里只核对mel空间的解析导数身份，不进行分项反传，也不把GT加入训练。
    score_y = weights[:, None, None] * (prediction-target).sign() / prediction.numel()
    gt_y = (prediction-inputs['x0']).sign() / prediction.numel()
    tensors = {'prediction': prediction, 'q950': q950, 'teacher_target': target, 'real_waveform': real, 'fake_waveform': fake,
               'analytic_score_mel_gradient': score_y, 'analytic_GT_mel_gradient': gt_y}
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise RuntimeError('A shared prediction, target, waveform or analytic readout is nonfinite')
    scalars = {'primary_moving_score': float(moving), 'diagnostic_frozen_target_score': float(fixed),
        'teacher_target_weighted_l1_drift': float(drift), 'train_gt_l1': float((prediction-inputs['x0']).abs().mean()),
        'train_prediction_rms': float(prediction.square().mean().sqrt())}
    if any(not math.isfinite(value) for value in scalars.values()):
        raise RuntimeError('A shared scalar is nonfinite')
    return {'tensors': tensors, 'identities': {n: cross.identity(t) for n, t in tensors.items()}, 'scalars': scalars}


def require_same_shared(current, reference):
    # 这条逐位检查只用于更新前同一个S，绝不要求不同候选具有同一教师目标。
    if current['identities'] != reference['identities'] or current['scalars'] != reference['scalars']:
        raise RuntimeError('D condition changed the common pre-update S/input/waveform/teacher-target point')
    for name in current['tensors']:
        if not torch.equal(current['tensors'][name], reference['tensors'][name]):
            raise RuntimeError('Shared tensor did not replay bitwise: ' + name)


@torch.no_grad()
def discriminator_readout(discriminator, shared):
    # A与B复用同一adv/FM，仅教师目标定义不同；没有第二次教师或声码器前向。
    loss_d, adv, fm = training.serial_discriminator_metrics(discriminator, shared['tensors']['real_waveform'], shared['tensors']['fake_waveform'])
    values = dict(shared['scalars'], discriminator_loss=float(loss_d), adv=float(adv), fm_raw=float(fm))
    values['primary_moving_total'] = values['adv'] + 2*values['fm_raw'] + 45*values['primary_moving_score']
    values['diagnostic_frozen_target_total'] = values['adv'] + 2*values['fm_raw'] + 45*values['diagnostic_frozen_target_score']
    if any(not math.isfinite(v) for v in values.values()):
        raise RuntimeError('A discriminator readout is nonfinite')
    return values


def actual_direction_summary(gradients, delta, reference):
    # 全部71个实际参数均有记录；None参数没有Adam更新，必须逐位不动。
    if set(gradients) != set(delta) or set(gradients) != set(reference['requires_grad']):
        raise RuntimeError('Actual gradient/delta parameter ownership differs')
    per = {}
    for name, gradient in gradients.items():
        update = delta[name]
        if gradient is None or update is None:
            if gradient is not None or update is not None:
                raise RuntimeError('Actual gradient and Adam delta have different None support')
            per[name] = {'is_none': True}
            continue
        if gradient.shape != reference['state'][name].shape or gradient.dtype != reference['state'][name].dtype or update.shape != gradient.shape or update.dtype != gradient.dtype:
            raise RuntimeError('Actual gradient or delta shape/dtype differs: ' + name)
        if not bool(torch.isfinite(gradient).all() and torch.isfinite(update).all()):
            raise RuntimeError('Nonfinite actual gradient or delta: ' + name)
        g, change = gradient.double().flatten(), update.double().flatten()
        per[name] = {'is_none': False, 'gradient_l2': float(g.norm()), 'delta_l2': float(change.norm()),
                     'gradient_dot_delta': float(torch.dot(g, change)), 'gradient_max_abs': float(gradient.abs().max()),
                     'delta_max_abs': float(update.abs().max()), 'numel': gradient.numel()}
    return {'actual_joint_and_delta': probe.gradient_statistics({'actual_joint': gradients}, delta), 'per_parameter': per}


def validate_cross_support(left, right):
    if set(left) != set(right):
        raise RuntimeError('D0/D1 actual gradient parameter names differ')
    for name in left:
        a, b = left[name], right[name]
        if a is None or b is None:
            if a is not None or b is not None:
                raise RuntimeError('Changing D changed None gradient support: ' + name)
        elif a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError('Changing D changed gradient shape/dtype: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--run-approved', action='store_true')
    args = parser.parse_args()
    if not args.run_approved or not 1 <= args.max_seconds <= 300:
        raise RuntimeError('Explicit approval and at most300 seconds are required')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)  # 持有原三把锁直至进程结束，不启动其他训练或评分。
    if args.output_dir.is_symlink() or args.output_dir.resolve() != OUTPUT or OUTPUT.exists():
        raise RuntimeError('Only the predeclared new output directory is allowed')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Original myconda runtime is required')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Use the original rented CUDA interpreter')
    if training.git_head(ROOT) != base.EXPECTED_HEAD:
        raise RuntimeError('Frozen production HEAD changed')
    disk, gpu = cross.capacity(DEV), cross.gpu_idle()
    if gpu['free_mib'] < 24576:
        raise RuntimeError('At least24GiB VRAM must be free before model allocation')
    started = time.monotonic()
    initial_flags = require_original_flags()
    print('DJOINT3327_FLAGS_START ' + json.dumps(initial_flags), flush=True)
    def check():
        d.check_limits(started, args.max_seconds)
    def progress(stage):
        check()
        print('DJOINT3327 ' + stage + ' ' + json.dumps(require_original_flags()), flush=True)
    hashes = dict(d.PINNED)
    hashes.update({str(HELPER): HELPER_SHA, str(PROTOCOL): PROTOCOL_SHA, str(Path(__file__).resolve()): base.sha256_file(Path(__file__))})
    resume.verify_files(hashes)
    contract, receipt = cross.read_json(d.RUN/'run_contract.json'), cross.read_json(d.RUN/'completion.json')
    if training.canonical_json_sha256(contract) != d.CONTRACT_SHA or {k: contract['runtime'][k] for k in ORIGINAL_FLAGS} != initial_flags:
        raise RuntimeError('Original3327 contract or archived precision flags differ')
    hashes.update(contract['file_sha256'])
    development, development_hashes = resume.load_development(DEV/'development_contract.json')
    hashes.update(development_hashes)
    resume.verify_files(hashes)
    grand = cross.read_json(resume.PARENT/'run_contract.json')
    if training.collect_source_fingerprints(ROOT) != grand['source_sha256']:
        raise RuntimeError('Production source fingerprints changed')
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if config.as_dict() != contract['configuration'] or (config.score_target, config.score_weight, config.lambda_fm, config.lambda_dist) != ('clean_x0', 'sqrt_alpha_bar', 2, 45):
        raise RuntimeError('Original objective or2/45 coefficients changed')
    if contract['training_indices'] != development['retained_train_indices'] or contract['excluded_train_indices'] != development['excluded_train_indices']:
        raise RuntimeError('Retained training pool or52 exclusions changed')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
    vocoder, vconfig, _ = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR/'generator_step_2500000.pt', base.VOCODER_DIR/'config.json',
        base.VOCODER_DIR/'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
    diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = base.load_mel_statistics(ROOT/'data', device)
    payload = resume.load_torch_checkpoint(d.PARENT)
    d.validate_parent(payload, student, discriminator, config, contract, receipt)
    student.load_state_dict(payload['student'], strict=True); discriminator.load_state_dict(payload['discriminator'], strict=True)
    student.train(); discriminator.train(); teacher.eval(); vocoder.eval()
    training.set_requires_grad(student, True); training.set_requires_grad(discriminator, True)
    opts = {'student': torch.optim.Adam(student.parameters(), lr=d.LRS['student'], betas=(.5, .9)),
            'discriminator': torch.optim.Adam(discriminator.parameters(), lr=d.LRS['discriminator'], betas=(.5, .9))}
    for name, opt in opts.items():
        d.restore_optimizer(opt, payload[name+'_optimizer'])
    saved_state_sha = probe.tree_sha({'s': payload['student_optimizer'], 'd': payload['discriminator_optimizer'], 'rng': payload['rng_state']})
    frozen = {'teacher': teacher, 'vocoder': vocoder}
    frozen_sha = {n: pair.model_sha(m) for n, m in frozen.items()}
    if any(frozen_sha[n] != contract['fixed_frozen_models_sha256'][n] for n in frozen):
        raise RuntimeError('Original frozen teacher or vocoder changed')
    pair.assert_frozen(frozen, frozen_sha)
    models = {'student': student, 'discriminator': discriminator, **frozen}
    dataset = base.VoiceGradDataset(ROOT/'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    indices = resume.GlobalBatchPlan(development['retained_train_indices']).for_step(d.NEXT)
    if len(indices) != 32 or set(indices) & set(development['excluded_train_indices']):
        raise RuntimeError('Next absolute batch must contain32 retained utterances')
    with restored_state(models, opts, payload['rng_state'], device) as original:
        with pair.preserve_rng(device):
            batch = training._slice_batch(base.batch_for(dataset, indices, 1234+100000+d.NEXT), device)
            training.seed_everything(1234+200000+d.NEXT)
            inputs = {'x0': batch['mel'], 'bnf': batch['bnf'], 'spk_emb': batch['spk_emb'],
                'source_noise': torch.randn_like(batch['mel']), 'score_noise': torch.randn_like(batch['mel']),
                'score_timesteps': torch.randint(1, 1001, (32,), device=device, dtype=torch.long)}
            route_rng = training.capture_rng_state()  # 两臂都从同一实际抽样后的随机位置开始。
            archived = resume.load_torch_checkpoint(pair.INPUT_DIR/'fixed_inputs.pt')
            for name in ('dev_mel', 'dev_bnf', 'dev_spk_emb', 'dev_source_noise'):
                if cross.identity(archived[name])['sha256'] != contract['fixed_train_inputs_sha256'][name]:
                    raise RuntimeError('Original32DEV input changed: ' + name)
                inputs[name] = archived[name].to(device)
            del archived
            identities = {n: cross.identity(t) for n, t in inputs.items()}
            dev_batch = {k: inputs['dev_'+k] for k in ('mel', 'bnf', 'spk_emb')}
            with pair.preserve_rng(device):
                common = shared_readout(student, teacher, vocoder, diffusion, inputs, mean, std, config)
            target = common['tensors']['teacher_target']
            students, discriminators = {'original': original['student']}, {'D0': original['discriminator']}
            arms, actuals, deltas = {}, {}, {}
            for role in d.DISCRIMINATORS:
                progress(role+'_restore_original3327')
                for name, model in models.items(): d.restore(model, original[name])
                for name, opt in opts.items(): d.restore_optimizer(opt, payload[name+'_optimizer'])
                training.restore_rng_state(route_rng, device)
                if role == 'D1':
                    with pair.preserve_rng(device):
                        d_record = d.d_update(discriminator, opts['discriminator'], common['tensors']['real_waveform'], common['tensors']['fake_waveform'])
                    discriminators['D1'] = d.snapshot(discriminator)
                    verify_snapshot(student, original['student'])
                    if probe.tree_sha(opts['student'].state_dict()) != probe.tree_sha(payload['student_optimizer']):
                        raise RuntimeError('D-only update changed student Adam')
                    pair.assert_frozen(frozen, frozen_sha)
                else:
                    d_record = {'optimizer_updates': 0, 'lr': d.LRS['discriminator'], 'state_sha256': probe.tree_sha(opts['discriminator'].state_dict())}
                training.set_requires_grad(discriminator, False)
                with pair.preserve_rng(device):
                    before = shared_readout(student, teacher, vocoder, diffusion, inputs, mean, std, config)
                    require_same_shared(before, common)
                    del before
                    progress(role+'_production_joint_and_actual_S_Adam')
                    actual, adam_record = d.s_update(student, teacher, vocoder, discriminator, diffusion, inputs,
                        common['tensors']['real_waveform'], common['tensors']['fake_waveform'], target, mean, std, config, opts['student'])
                    candidate = d.snapshot(student)
                    delta = {n: None if value is None else candidate['state'][n]-original['student']['state'][n] for n, value in actual.items()}
                    for name, value in actual.items():
                        if value is None and not torch.equal(candidate['state'][name], original['student']['state'][name]):
                            raise RuntimeError('None-gradient parameter moved: '+name)
                    arms[role] = {'D_update': d_record, 'S_update': adam_record, 'precision_flags': require_original_flags(),
                        'common_preupdate_identities': common['identities'], 'actual_directions': actual_direction_summary(actual, delta, original['student'])}
                    actuals[role], deltas[role], students['candidate_'+role] = actual, delta, candidate
                pair.assert_frozen(frozen, frozen_sha)
            validate_cross_support(actuals['D0'], actuals['D1'])
            matrix, dev_results, row_identities = {}, {}, {}
            for label in d.STUDENTS:
                progress(label+'_shared_readout_and_fixed32DEV')
                d.restore(student, students[label])
                training.restore_rng_state(route_rng, device)
                with pair.preserve_rng(device), torch.no_grad():
                    shared = shared_readout(student, teacher, vocoder, diffusion, inputs, mean, std, config, target)
                    if shared['identities']['q950'] != common['identities']['q950']:
                        raise RuntimeError('Candidate evaluation changed source q950')
                    if label == 'original': require_same_shared(shared, common)
                    row_identities[label] = shared['identities']
                    dev_results[label] = base.evaluate_reconstruction(student, diffusion, dev_batch, inputs['dev_source_noise'], config, device)
                    matrix[label] = {}
                    for role in d.DISCRIMINATORS:
                        progress(label+'_common_'+role)
                        d.restore(discriminator, discriminators[role])
                        matrix[label][role] = discriminator_readout(discriminator, shared)
                    if {n: cross.identity(t) for n, t in shared['tensors'].items()} != shared['identities']:
                        raise RuntimeError('D readout mutated reused y/waveform/teacher target')
                    del shared
                if len(dev_results[label]['per_utterance_mel_l1']) != 32:
                    raise RuntimeError('All32DEV errors must be retained')
            replay = probe.validate_development_replay(dev_results['original'], payload['development'])
            if max(abs(a-b) for a,b in zip(dev_results['original']['per_utterance_mel_l1'], payload['development']['per_utterance_mel_l1'])) > 1e-7:
                raise RuntimeError('Original32DEV individual errors did not replay')
            directions = probe.gradient_statistics({'D0_actual_joint': actuals['D0'], 'D1_actual_joint': actuals['D1'],
                'D0_actual_delta': deltas['D0'], 'D1_actual_delta': deltas['D1']})
            if {n: cross.identity(t) for n, t in inputs.items()} != identities:
                raise RuntimeError('A shared fixed input changed')
            differences = d.matrix_differences(matrix, dev_results)
            differences['same_student_D1_minus_D0'] = {s: {k: matrix[s]['D1'][k]-matrix[s]['D0'][k] for k in matrix[s]['D0']} for s in d.STUDENTS}
            result = {'status': 'completed_readonly_D_joint_only_probe', 'parent_step': d.STEP, 'next_batch_step': d.NEXT,
                'parent_checkpoint_sha256': d.PARENT_SHA, 'parent_contract_sha256': d.CONTRACT_SHA, 'file_sha256': hashes,
                'input_identities': identities, 'training_indices': indices, 'crop_seed':1234+100000+d.NEXT, 'noise_seed':1234+200000+d.NEXT,
                'noise_draw_order':['source_noise_CUDA','score_noise_CUDA','score_timesteps_CUDA'], 'score_timesteps':inputs['score_timesteps'].cpu().tolist(),
                'original_fixed32DEV_source_noise_seed':55001, 'target_anchor':'fresh R(S3327) at fixed TRAIN3328; A shared, B refreshed separately per student',
                'fixed_target_identity':cross.identity(target), 'actual_optimizer_lrs':d.LRS, 'temporary_optimizer_updates':{'D':1,'S':2},
                'arms':arms, 'matrix_3S_by_2D':matrix, 'fixed32_development':dev_results, 'development_replay_absolute_differences':replay,
                'differences':differences, 'directions':directions, 'shared_row_tensor_identities':row_identities,
                'model_state_sha256':{'students':{n:s['sha256'] for n,s in students.items()},'discriminators':{n:s['sha256'] for n,s in discriminators.items()},**frozen_sha},
                'source_twoAdam_rng_sha256':saved_state_sha, 'initial_gpu':gpu, 'initial_disk':disk, 'initial_precision_flags':initial_flags,
                'limitations':['One retainedTRAIN32 batch/noise/t draw cannot establish a CER or whole-curve cause.',
                    'S candidates use production joint backward and actual Adam; no component-gradient estimator or equivalence claim.',
                    'Different candidate students can produce different frozen-teacher targets; both D columns reuse each row target and waveform.',
                    'GT is read-only and never contributes a gradient; all3S/2D and32DEV utterances are reported.',
                    'Earlier D and off-precision probes failed; their failures do not establish a TF32 cause.']}
    # 所有临时状态已由双层finally恢复，这里再次钉住归档文件与冻结模型。
    pair.assert_frozen(frozen, frozen_sha)
    if probe.tree_sha({'s':payload['student_optimizer'],'d':payload['discriminator_optimizer'],'rng':payload['rng_state']}) != saved_state_sha:
        raise RuntimeError('Original in-memory parent Adam/RNG payload changed')
    resume.verify_files(hashes)
    progress('all_models_twoAdam_modes_grad_buffers_rng_restored')
    result.update(elapsed_seconds=time.monotonic()-started, all_temporary_states_restored=True,
        original_twoAdam_rng_unchanged=True, original_model_files_unchanged=True, model_checkpoint_writes=0,
        final_precision_flags=require_original_flags(), runtime={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_JSON_BYTES:
        raise RuntimeError('JSON-only result exceeds2MiB budget')
    check()
    OUTPUT.mkdir()
    with (OUTPUT/'results.json.tmp').open('xb') as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    (OUTPUT/'results.json.tmp').rename(OUTPUT/'results.json')
    print('DJOINT3327_COMPLETE ' + str(OUTPUT/'results.json'), flush=True)


if __name__ == '__main__':
    main()
