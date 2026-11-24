# 对停止后的两个完整状态做有限步长诊断；只改隔离模型的内存并立即还原，绝不写新模型权重。
"""Fixed-D Adam direction probe with frozen-target and refreshed-target readouts."""

import argparse  # 必须显式批准诊断；导入模块不会访问GPU。
import contextlib  # 临时试算无论成功还是报错都恢复原参数和buffer。
import hashlib  # 将源文件、数据、随机张量和模型状态绑定到报告。
import json  # 新目录只保存轻量诊断记录。
import math  # Adam偏差校正及向量范数。
import os  # 查询磁盘余量，不删除任何已有材料。
import platform  # 只用原myconda运行环境。
import sys  # 本地Windows只允许编辑，不允许执行项目检查。
import time  # 整个GPU诊断最多300秒，超时不继续尝试其他尺度。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
MODULE_SHAS = {'resume_acdd_development.py': '7b81570909cdf39cfa16c563c7c302e7c3886e5551060917ad61de7e8baf0014',
    'run_acdd_development.py': '5cd72f6199e542ecb64a49ed9c70883a4f9b6ea2d39d0faf4fb2a652ba0e2395'}
for name, sha in MODULE_SHAS.items():
    if hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() != sha:
        raise RuntimeError('Frozen helper changed: ' + name)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch  # 真诊断只求梯度，没有torch.optim实例或optimizer.step。
import numpy as np  # RNG中的NumPy状态数组也按全部字节哈希，不能只用省略后的repr。
import resume_acdd_development as resume

base, training = resume.baseline, resume.baseline.training
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
PARENT = DEV / 'acdd_clean_resume_s2000_v1'
STOPPED = DEV / 'acdd_s2000_to4000_v1'
STATES = {
    2000: (PARENT / 'full_step_002000.pt', '77de35a153636d8156a45c6c23b21191b2c663d55963df874e17946e3a7ff72c',
           'f7d351f786c9cb7f4583327a8ff7a4b6d2c551c40bdbf36ce9e54dbe1fad933b'),
    3107: (STOPPED / 'latest_full.pt', 'f007dd4375bbf23a78372549e6e79b87c06e94fd38c07b5e09eb6624ae50ce44',
           '5fbd82bd161a7881f717b10b4d16e039f259e12ea7372b6cccb23e53b3326444'),
}  # 使用每个学生自己对应的D和Adam；跨D比较由另一个只读诊断负责。
SCALES = (0.25, 0.5, 1.0)  # 预先确定全部尺度，不根据结果隐藏不利尺度。
MAX_OUTPUT_BYTES = 100 * 1024**2


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tree_sha(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()  # 同时覆盖参数、buffer、Adam计数和超参数，不只比较文件名。


def checked_grad(loss, leaf, retain_graph=False):
    value, = torch.autograd.grad(loss, leaf, retain_graph=retain_graph)
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError('Nonfinite diagnostic gradient')
    return value.detach()


def validate_score_gradient(actual, analytic):
    difference = (actual - analytic).abs()
    nonzero = analytic != 0
    relative = difference[nonzero] / analytic[nonzero].abs() if bool(nonzero.any()) else difference.new_zeros(1)
    record = {'max_absolute_difference': float(difference.max()), 'max_relative_difference_on_nonzero_reference': float(relative.max()),
              'rtol': 1e-6, 'atol': 1e-12}
    if not torch.allclose(actual, analytic, rtol=record['rtol'], atol=record['atol']):
        raise RuntimeError('Score stop-gradient disagrees with L1 derivative: ' + json.dumps(record))
    return record  # 两次mean与一次总元素除法可能差几个ULP，记录实差而不将舍入差判成公式错误。


def validate_development_replay(actual, archived):
    keys = ('mel_l1', 'prediction_rms', 'reference_rms', 'cosine_reference')
    differences = {key: abs(actual[key] - archived[key]) for key in keys}
    if any(not math.isfinite(actual[key]) or differences[key] > 1e-7 for key in keys):
        raise RuntimeError('Original fixed32 development readout did not replay: ' + json.dumps(differences))
    return differences  # 固定开发输入的SHA正确还不够，更新前读数也须与真实归档对应。


def wave_output_gradients(y, reference, vocoder, discriminator, mean, std, check_budget):
    # y是与学生图分开的输出叶子；先求波形梯度，再经同一个冻结G拉回y。
    fake = vocoder(training.denormalize_mel(y, mean, std))
    with torch.no_grad():
        real = vocoder(training.denormalize_mel(reference, mean, std))
    proxy = fake.detach().requires_grad_(True)
    grad_adv, grad_fm = torch.zeros_like(proxy), torch.zeros_like(proxy)
    values = {'adv': 0.0, 'fm_raw': 0.0}
    for branch in discriminator.loss_branches():
        check_budget()
        with torch.no_grad():
            _, real_features = branch(real)
        fake_score, fake_features = branch(proxy)
        adv = training.least_squares_student_loss([fake_score])
        fm = training.feature_matching_loss([real_features], [fake_features])
        grad_adv.add_(checked_grad(adv, proxy, retain_graph=True))
        grad_fm.add_(checked_grad(fm, proxy))
        values['adv'] += float(adv.detach())
        values['fm_raw'] += float(fm.detach())
        del real_features, fake_features, fake_score, adv, fm
    adv_y, = torch.autograd.grad(fake, y, grad_outputs=grad_adv, retain_graph=True)
    fm_y, = torch.autograd.grad(fake, y, grad_outputs=grad_fm)
    if not bool(torch.isfinite(adv_y).all() and torch.isfinite(fm_y).all()):
        raise RuntimeError('Invalid waveform-to-mel VJP')
    return values, adv_y.detach(), fm_y.detach()  # 不拆batch，不改变每层均值和分支相加的定义。


def parameter_banks(prediction, named_parameters, output_gradients, check_budget):
    names, parameters = zip(*named_parameters)
    banks = {}
    for number, (label, vector) in enumerate(output_gradients.items()):
        check_budget()
        values = torch.autograd.grad(prediction, parameters, grad_outputs=vector,
            retain_graph=number + 1 < len(output_gradients), allow_unused=True)
        banks[label] = {name: None if value is None else value.detach().cpu() for name, value in zip(names, values)}
        if any(value is not None and not bool(torch.isfinite(value).all()) for value in banks[label].values()):
            raise RuntimeError('Nonfinite parameter gradient: ' + label)
        del values
    # 各项拉回参数后再相加；保留浮点累加次序与生产联合backward可能有极小差别的说明。
    banks['total'] = {}
    for name in names:
        present = [banks[label][name] for label in ('adv', '2fm', '45score') if banks[label][name] is not None]
        banks['total'][name] = sum(present[1:], present[0].clone()) if present else None
    return banks


def gradient_statistics(banks, delta=None):
    values = dict(banks)
    if delta is not None:
        values['adam_delta'] = delta
    labels, names = list(values), list(next(iter(values.values())))
    gram = {a: {b: 0.0 for b in labels} for a in labels}
    none = {label: [] for label in labels}
    for name in names:
        vectors = {}
        for label in labels:
            value = values[label][name]
            if value is None:
                none[label].append(name)
            else:
                vectors[label] = value.flatten().double()
        for i, a in enumerate(labels):
            for b in labels[i:]:
                dot = float(torch.dot(vectors[a], vectors[b])) if a in vectors and b in vectors else 0.0
                gram[a][b] += dot
                if a != b:
                    gram[b][a] += dot
    norms = {name: math.sqrt(max(gram[name][name], 0.0)) for name in labels}
    cosine = {a: {b: gram[a][b] / (norms[a] * norms[b]) if norms[a] and norms[b] else None for b in labels} for a in labels}
    return {'norms': norms, 'cosines': cosine, 'dot_products': gram, 'none_parameter_names': none,
            'delta_sign': 'dot(gradient,adam_delta)<0 predicts a first-order reduction of that fixed objective'}


def adam_candidate(named_parameters, banks, saved, expected_step):
    # 只对原状态的CPU副本做标准Adam算术，不创建优化器、不改变任何归档moment。
    group = saved['param_groups'][0]
    if len(saved['param_groups']) != 1 or (group['lr'], group['betas'], group['eps'], group['weight_decay'], group['amsgrad']) != (0.0002, (0.5, 0.9), 1e-8, 0, False):
        raise RuntimeError('Only original unmodified Adam is supported')
    if any(group.get(name, False) for name in ('maximize', 'capturable', 'differentiable', 'fused')):
        raise RuntimeError('Unsupported Adam execution variant')
    identifiers = group['params']
    if len(identifiers) != len(named_parameters) or set(saved['state']) != set(identifiers):
        raise RuntimeError('Adam parameter ownership/layout mismatch')
    beta1, beta2 = group['betas']
    candidate, delta = {}, {}
    for (name, parameter), identifier in zip(named_parameters, identifiers):
        original = parameter.detach().cpu()
        state, gradient = saved['state'][identifier], banks[name]
        if float(state['step']) != expected_step or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise RuntimeError('Adam moment step or fields changed')
        if any(state[key].shape != original.shape or state[key].dtype != original.dtype for key in ('exp_avg', 'exp_avg_sq')):
            raise RuntimeError('Adam moment shape/dtype mismatch')
        after = original.clone()
        if gradient is None:
            candidate[name], delta[name] = after, None  # None与零梯度不同：没有当前梯度时Adam不会推进该参数。
            continue
        if gradient.shape != original.shape or gradient.dtype != original.dtype or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('Invalid parameter gradient for Adam')
        first, second = state['exp_avg'].cpu().clone(), state['exp_avg_sq'].cpu().clone()
        if not bool(torch.isfinite(first).all() and torch.isfinite(second).all() and (second >= 0).all()):
            raise RuntimeError('Invalid saved Adam moments')
        first.mul_(beta1).add_(gradient, alpha=1 - beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        step = expected_step + 1
        denominator = (second.sqrt() / math.sqrt(1 - beta2 ** step)).add_(group['eps'])
        after.addcdiv_(first, denominator, value=-group['lr'] / (1 - beta1 ** step))
        if not bool(torch.isfinite(after).all()):
            raise RuntimeError('Invalid algebraic Adam candidate')
        candidate[name], delta[name] = after, after - original  # 差值包含float32参数加法舍入，不以float64公式冒称实际精度。
    return candidate, delta


@contextlib.contextmanager
def trial_state(model, original, candidate, delta, scale):
    if scale not in SCALES:
        raise ValueError('Unregistered trial scale')
    before = tree_sha(original)
    try:
        state = dict(original)  # buffer全部沿用原值，仅参数拥有候选位移。
        for name in candidate:
            state[name] = candidate[name] if scale == 1.0 else (
                original[name] if delta[name] is None else original[name] + scale * delta[name])
        model.load_state_dict(state, strict=True)
        yield
    finally:
        model.load_state_dict(original, strict=True)
        if tree_sha(model.state_dict()) != before:
            raise RuntimeError('Temporary model was not restored exactly')


@torch.no_grad()
def score_readout(prediction, reference, target, teacher, diffusion, t, noise, speaker, bnf,
                  vocoder, discriminator, mean, std, config):
    # B与原训练日志相同：教师根据当前学生结果重新生成目标；A只用于判断本次固定目标是否下降。
    moving, current_target = training.score_distillation_loss(prediction, teacher, diffusion, t, noise, speaker, bnf,
        target_mode=config.score_target, weight_mode=config.score_weight)
    weights = diffusion.sqrt_alphas_cumprod[t]
    fixed = (weights * (prediction - target).abs().flatten(1).mean(1)).mean()
    drift = (weights * (current_target - target).abs().flatten(1).mean(1)).mean()
    real = vocoder(training.denormalize_mel(reference, mean, std))
    fake = vocoder(training.denormalize_mel(prediction, mean, std))
    _, adv, fm = training.serial_discriminator_metrics(discriminator, real, fake)
    return {'primary_moving_score': float(moving), 'primary_moving_total': float(adv + 2 * fm + 45 * moving),
        'diagnostic_frozen_target_score': float(fixed), 'diagnostic_frozen_target_total': float(adv + 2 * fm + 45 * fixed),
        'teacher_target_weighted_l1_drift': float(drift), 'adv': float(adv), 'fm_raw': float(fm),
        'train_gt_l1': float((prediction - reference).abs().mean()),
        'train_prediction_rms': float(prediction.square().mean().sqrt())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--run-approved', action='store_true')
    args = parser.parse_args()
    if not args.run_approved or not 1 <= args.max_seconds <= 300:
        raise RuntimeError('Explicit approval and at most300 seconds are required')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Use original myconda runtime')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Registered myconda CUDA runtime required')
    if training.git_head(ROOT) != base.EXPECTED_HEAD:
        raise RuntimeError('Production source changed')
    output = args.output_dir.resolve()
    if output.parent != DEV or output.exists():
        raise RuntimeError('Use a new direct child of DEV')
    fs = os.statvfs(DEV)
    if fs.f_bavail * fs.f_frsize < 3 * 1024**3 + MAX_OUTPUT_BYTES:
        raise RuntimeError('Retain3GiB system reserve plus100MiB diagnostic budget')
    contracts = {step: json.loads(path.parent.joinpath('run_contract.json').read_text()) for step, (path, _, _) in STATES.items()}
    for step, contract in contracts.items():
        if training.canonical_json_sha256(contract) != STATES[step][2]:
            raise RuntimeError('Archived contract identity mismatch')
    dependencies = dict(contracts[3107]['file_sha256'])
    dependencies.update({str(path): sha for path, sha, _ in STATES.values()})
    dependencies.update({str(PARENT / 'run_contract.json'): '4ede290fc6c98e91fe91608410f74827d205d343b96137f9a57882b1ecd6a80e',
        str(STOPPED / 'run_contract.json'): '15287ed3c90e0d45dc7d39badf2b67d309a4e6be0ef268fe6ab4c1843a82633f',
        str(STOPPED / 'completion.json'): '9bb61d34a87ff9bfaf4a9a4cf10172dc231f012b493ecc86593762f41a67d688',
        str(Path(__file__).resolve()): base.sha256_file(Path(__file__))})
    development, dev_hashes = resume.load_development(DEV / 'development_contract.json')
    dependencies.update(dev_hashes)
    resume.verify_files(dependencies)
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if any(contract['configuration'] != config.as_dict() for contract in contracts.values()):
        raise RuntimeError('Archived training configuration changed')
    if (config.score_target, config.score_weight, config.lambda_fm, config.lambda_dist) != ('clean_x0', 'sqrt_alpha_bar', 2, 45):
        raise RuntimeError('Diagnostic must preserve current objective and coefficients')
    torch.set_num_threads(2)
    training.seed_everything(1234)
    device = torch.device('cuda')
    started = time.monotonic()
    def progress(stage, **details):
        print(json.dumps({'stage': stage, 'elapsed_seconds': time.monotonic() - started, **details}), flush=True)
    def check_budget():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError('Bounded diagnostic time budget exhausted; no further trials')
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    indices = resume.GlobalBatchPlan(development['retained_train_indices']).for_step(3108)
    if set(indices) & set(development['excluded_train_indices']):
        raise RuntimeError('TRAIN probe overlaps52 excluded utterances')
    batch = base.batch_for(dataset, indices, 104342)
    tensors = training._slice_batch(batch, device)
    x0, bnf, speaker = tensors['mel'], tensors['bnf'], tensors['spk_emb']
    training.seed_everything(204342)
    source_noise = torch.randn_like(x0)
    score_noise = torch.randn_like(x0)
    timesteps = torch.randint(1, 1001, (32,), device=device, dtype=torch.long)  # 与生产同次序，不能用CPU噪声代替。
    protocol = development['reconstruction_protocol']
    dev_batch = base.batch_for(dataset, development['reconstruction_dev_indices'], protocol['crop_seed'])
    dev_noise = torch.randn(dev_batch['mel'].shape, device=device,
        generator=torch.Generator(device=device).manual_seed(protocol['source_noise_seed']))
    teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
    progress('teacher_and_student_loaded')
    vocoder, vconfig, _ = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
        base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
    diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = base.load_mel_statistics(ROOT / 'data', device)
    progress('frozen_vocoder_discriminator_and_inputs_ready')
    input_tensors = {'x0': x0, 'bnf': bnf, 'spk_emb': speaker, 'source_noise': source_noise,
        'score_noise': score_noise, 'score_timesteps': timesteps,
        'q950': diffusion.q_sample(x0, torch.full((32,), 950, device=device, dtype=torch.long), noise=source_noise),
        'dev_mel': dev_batch['mel'], 'dev_bnf': dev_batch['bnf'], 'dev_spk_emb': dev_batch['spk_emb'], 'dev_source_noise': dev_noise}
    input_sha = {name: tensor_sha(value) for name, value in input_tensors.items()}
    if any(input_sha['dev_' + key] != contracts[2000]['fixed_development_inputs'][key] for key in ('mel', 'bnf', 'spk_emb')) or input_sha['dev_source_noise'] != contracts[2000]['fixed_development_source_noise_sha256']:
        raise RuntimeError('Original fixed32 development tensors changed')
    frozen_sha = {name: tree_sha(model.state_dict()) for name, model in (('teacher', teacher), ('vocoder', vocoder))}
    output.mkdir()
    # 实际随机张量不到100MiB；这里只写输入，绝不保存任何候选模型或优化器状态。
    training.atomic_torch_save({name: value.detach().cpu() for name, value in input_tensors.items()}, output / 'fixed_inputs.pt')
    result = {'status': 'running', 'scope': 'diagnostic theta update at matched fixed D; not full production3108 replay',
        'source_sha256': dependencies, 'input_sha256': input_sha, 'train_indices': indices, 'global_input_step': 3108,
        'crop_seed': 104342, 'noise_seed': 204342, 'score_timesteps': timesteps.cpu().tolist(), 'scales': list(SCALES),
        'formal_test_used': False, 'coefficients': {'fm': 2, 'dist': 45}, 'models': {},
        'limitations': ['D is not updated; this is not a full alternating GAN training step.',
            'Frozen-target A is diagnostic and is never relabeled as original moving-target training loss B.',
            'CPU float32 Adam algebra matches CPU torchAdam; separate parameter VJPs and CUDA kernel rounding can differ slightly.',
            'All declared scales are reported; no candidate checkpoint is written and no training is resumed.']}
    try:
        for step, (path, _, contract_sha) in STATES.items():
            check_budget()
            progress('loading_archived_full_state', archived_step=step)
            payload = resume.load_torch_checkpoint(path)
            if payload['global_step'] != step or payload['run_contract_sha256'] != contract_sha or payload['configuration'] != config.as_dict():
                raise RuntimeError('Full checkpoint metadata does not match its archive')
            student.load_state_dict(payload['student'], strict=True)
            discriminator.load_state_dict(payload['discriminator'], strict=True)
            student.train()
            training.set_requires_grad(discriminator, False)
            resume.validate_adam_state(payload['student_optimizer'], student, step)
            resume.validate_adam_state(payload['discriminator_optimizer'], discriminator, step)
            original = {name: value.detach().cpu().clone() for name, value in student.state_dict().items()}
            original_sha, d_sha = tree_sha(original), tree_sha(discriminator.state_dict())
            adam_sha = tree_sha({'s': payload['student_optimizer'], 'd': payload['discriminator_optimizer'], 'rng': payload['rng_state']})
            progress('matched_state_validated', archived_step=step)
            try:
                prediction, _ = training.make_student_prediction(student, diffusion, x0, speaker, bnf, 950, source_noise)
                leaf = prediction.detach().requires_grad_(True)
                score, target = training.score_distillation_loss(leaf, teacher, diffusion, timesteps, score_noise, speaker, bnf,
                    target_mode=config.score_target, weight_mode=config.score_weight)
                score_y = checked_grad(score, leaf)
                analytic = diffusion.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1) * (leaf.detach() - target).sign() / leaf.numel()
                score_agreement = validate_score_gradient(score_y, analytic)
                progress('score_gradient_validated_wave_vjp_starting', archived_step=step)
                wave_values, adv_y, fm_y = wave_output_gradients(leaf, x0, vocoder, discriminator, mean, std, check_budget)
                progress('wave_vjp_done_parameter_vjp_starting', archived_step=step)
                gtys = {'adv': adv_y, '2fm': 2 * fm_y, '45score': 45 * score_y, 'gt_l1': (leaf.detach() - x0).sign() / leaf.numel()}
                banks = parameter_banks(prediction, list(student.named_parameters()), gtys, check_budget)
                progress('parameter_gradients_done', archived_step=step)
                before = score_readout(leaf.detach(), x0, target, teacher, diffusion, timesteps, score_noise, speaker, bnf,
                    vocoder, discriminator, mean, std, config)
                if abs(before['adv'] - wave_values['adv']) > 2e-5 or abs(before['fm_raw'] - wave_values['fm_raw']) > 2e-5:
                    raise RuntimeError('Gradient and no-grad waveform objectives disagree')
                candidate, delta = adam_candidate(list(student.named_parameters()), banks['total'], payload['student_optimizer'], step)
                before_dev = base.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
                dev_replay = validate_development_replay(before_dev, payload['development'])
                progress('adam_candidate_and_archived_development_replay_done', archived_step=step)
                entry = {'matched_discriminator_step': step, 'saved_adam_step': step, 'candidate_adam_step': step + 1,
                    'before_train': before, 'before_dev': before_dev, 'archived_development_replay_absolute_differences': dev_replay,
                    'parameter_gradients': gradient_statistics(banks, delta), 'score_analytic_gradient_agreement': score_agreement,
                    'teacher_target_sha256': tensor_sha(target), 'trials': []}
                del prediction, leaf, gtys, adv_y, fm_y, score_y, analytic, banks
                for scale in SCALES:
                    check_budget()
                    progress('trial_starting', archived_step=step, scale=scale)
                    with trial_state(student, original, candidate, delta, scale):
                        with torch.no_grad():
                            changed, _ = training.make_student_prediction(student, diffusion, x0, speaker, bnf, 950, source_noise)
                        values = score_readout(changed, x0, target, teacher, diffusion, timesteps, score_noise, speaker, bnf,
                            vocoder, discriminator, mean, std, config)
                        dev_values = base.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
                        entry['trials'].append({'scale': scale, 'train': values, 'development': dev_values,
                            'delta_vs_before': {key: values[key] - before[key] for key in values},
                            'dev_gt_l1_delta_vs_before': dev_values['mel_l1'] - entry['before_dev']['mel_l1']})
                        del changed
                    print(json.dumps({'archived_step': step, 'scale': scale, 'train': values, 'dev_gt_l1': dev_values['mel_l1']}), flush=True)
                result['models'][str(step)] = entry
            finally:
                student.load_state_dict(original, strict=True)
                if tree_sha(student.state_dict()) != original_sha or tree_sha(discriminator.state_dict()) != d_sha:
                    raise RuntimeError('Student or matched discriminator was not preserved')
                if tree_sha({'s': payload['student_optimizer'], 'd': payload['discriminator_optimizer'], 'rng': payload['rng_state']}) != adam_sha:
                    raise RuntimeError('Archived Adam/RNG in-memory snapshot was altered')
            del payload, original, candidate, delta, target
        progress('all_trials_done_final_frozen_identity_checks')
        check_budget()
        if {name: tensor_sha(value) for name, value in input_tensors.items()} != input_sha:
            raise RuntimeError('Fixed inputs mutated during the diagnostic')
        for name, model in (('teacher', teacher), ('vocoder', vocoder)):
            if tree_sha(model.state_dict()) != frozen_sha[name] or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise RuntimeError('Frozen model changed: ' + name)
        resume.verify_files(dependencies)
        check_budget()
        result.update(status='complete_readonly_descent_diagnostic', elapsed_seconds=time.monotonic() - started,
                      temporary_models_restored=True, archived_optimizer_rng_unchanged=True, frozen_models_unchanged=True)
        if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > MAX_OUTPUT_BYTES - 1024**2:
            raise RuntimeError('Diagnostic exceeds100MiB output budget')
        base.save_json(output / 'results.json', result)
    except Exception as error:
        result.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - started)
        base.save_json(output / 'failure.json', result)
        raise


if __name__ == '__main__':
    main()  # 只有root审核并显式启动时才执行GPU诊断。
