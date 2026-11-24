"""固定真实输入，交叉读取S2000/S3107与D2000/D3107；不更新任何模型。"""

import argparse  # GPU只读诊断仍须主线程明确批准。
import hashlib  # 绑定原始权重、源码和实际输入张量。
import json  # 仅保存小型数值报告，不保存模型或音频。
import os  # 为轻量结果保留系统磁盘余量。
import platform  # 与实际训练保持同一Python/PyTorch运行时。
import sys  # 禁止在本地Windows执行项目计算。
import time  # 限制本次检查的运行时间。
from pathlib import Path  # 只读父归档，输出限定全新开发子目录。

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run this diagnostic only on the Matpool rental.')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
MODULE_SHAS = {
    'run_acdd_development.py': '5cd72f6199e542ecb64a49ed9c70883a4f9b6ea2d39d0faf4fb2a652ba0e2395',
    'resume_acdd_development.py': '7b81570909cdf39cfa16c563c7c302e7c3886e5551060917ad61de7e8baf0014',
    'run_acdd_s2000_to4000_v1.py': '2894c58180d8b2934de788800930267fac412d51aa2cda5cf257d6d92452e373',
}
for filename, expected in MODULE_SHAS.items():
    if hashlib.sha256((TOOLS / filename).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Reviewed dependency changed: ' + filename)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch  # CPU测试用小张量；main批准后才触及CUDA。
import resume_acdd_development as resume  # 只复用数据计划和文件校验，不调用恢复优化器或训练入口。

baseline, training = resume.baseline, resume.baseline.training
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
TRAIN_STEP, SEED, CROP_SEED, NOISE_SEED = 3108, 1234, 104342, 204342
CHECKPOINTS = {
    2000: {'directory': DEV / 'acdd_clean_resume_s2000_v1', 'file': 'full_step_002000.pt',
           'sha256': '77de35a153636d8156a45c6c23b21191b2c663d55963df874e17946e3a7ff72c',
           'contract_file_sha256': '4ede290fc6c98e91fe91608410f74827d205d343b96137f9a57882b1ecd6a80e',
           'contract_sha256': 'f7d351f786c9cb7f4583327a8ff7a4b6d2c551c40bdbf36ce9e54dbe1fad933b'},
    3107: {'directory': DEV / 'acdd_s2000_to4000_v1', 'file': 'latest_full.pt',
           'sha256': 'f007dd4375bbf23a78372549e6e79b87c06e94fd38c07b5e09eb6624ae50ce44',
           'contract_file_sha256': '15287ed3c90e0d45dc7d39badf2b67d309a4e6be0ef268fe6ab4c1843a82633f',
           'contract_sha256': '5fbd82bd161a7881f717b10b4d16e039f259e12ea7372b6cccb23e53b3326444'},
}


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()  # 与现有诊断一致，形状及dtype另行记录。


def tensor_identity(value):
    return {'sha256': tensor_sha(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def fingerprint_model(model):
    result = {}
    for category, values in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, value in values:
            result[category + ':' + name] = tensor_identity(value)  # 包括非持久Hann窗口缓冲区。
    return result


def assert_readonly_model(model):
    if any(layer.training for layer in model.modules()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Diagnostic model must be eval, frozen, and free of gradients.')
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError('Non-finite model parameter: ' + name)
        if name.endswith(('weight', 'weight_g', 'weight_v')) and not bool(torch.count_nonzero(parameter)):
            raise RuntimeError('A model weight tensor was zeroed: ' + name)  # 不容许清空D伪造低损失。


def load_exact_weights(model, state):
    model.load_state_dict(state, strict=True)  # 只加载指定S或D参数，从不创建或恢复优化器。
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert_readonly_model(model)
    current = model.state_dict()
    if set(current) != set(state) or any(not torch.equal(current[name].detach().cpu(), state[name]) for name in current):
        raise RuntimeError('Loaded model does not exactly match the archived state.')
    return fingerprint_model(model)


def assert_unchanged(model, before):
    assert_readonly_model(model)
    if fingerprint_model(model) != before:
        raise RuntimeError('Model parameters or buffers changed during the read-only diagnostic.')


def statistics(value):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError('Non-finite discriminator feature or score.')
    flat = value.float()
    return {'mean': float(flat.mean()), 'mean_abs': float(flat.abs().mean()),
            'rms': float(flat.square().mean().sqrt()), 'centered_std': float(flat.std(unbiased=False))}


def layer_measurement(real, fake, is_score):
    if real.shape != fake.shape:
        raise RuntimeError('Feature shapes differ; broadcasting is forbidden.')
    real_stats, fake_stats = statistics(real), statistics(fake)
    difference = (real - fake).abs()
    per_sample = difference.flatten(1).mean(1)
    raw = float(difference.mean())  # 与生产公式同样一次对整层取mean；逐句值另外报告。
    return {'shape': list(real.shape), 'kind': 'final_score' if is_score else 'hidden_feature',
            'fm_raw': raw, 'fm_weighted': 2.0 * raw, 'per_sample_fm_raw': per_sample.cpu().tolist(),
            'real': real_stats, 'fake': fake_stats,
            'relative_l1_descriptive_only': raw / (real_stats['mean_abs'] + 1e-8)}


def branch_measurement(real_score, real_features, fake_score, fake_features):
    if not real_features or len(real_features) != len(fake_features) or real_score.shape != fake_score.shape:
        raise RuntimeError('Discriminator score/features contract changed.')
    for score, features in ((real_score, real_features), (fake_score, fake_features)):
        if not torch.equal(features[-1].flatten(1), score):
            raise RuntimeError('Expected the last FM layer to be the exact output score map.')
    layers = [layer_measurement(a, b, i == len(real_features) - 1)
              for i, (a, b) in enumerate(zip(real_features, fake_features))]
    return {'layers': layers, 'fm_raw': sum(item['fm_raw'] for item in layers),
            'fm_hidden_raw': sum(item['fm_raw'] for item in layers[:-1]), 'fm_score_raw': layers[-1]['fm_raw'],
            'student_adv': float(((fake_score - 1) ** 2).mean()),
            'discriminator_loss_same_state': float(((real_score - 1) ** 2).mean() + fake_score.square().mean()),
            'real_score': statistics(real_score), 'fake_score': statistics(fake_score),
            'mean_real_minus_fake_score': float((real_score - fake_score).mean())}


def matrix_effects(values):
    m00, m10 = values['S2000_D2000'], values['S3107_D2000']
    m01, m11 = values['S2000_D3107'], values['S3107_D3107']
    return {'student_change_at_D2000': m10 - m00, 'student_change_at_D3107': m11 - m01,
            'D_change_at_S2000': m01 - m00, 'D_change_at_S3107': m11 - m10,
            'interaction': m11 - m10 - m01 + m00, 'moving_diagonal_change': m11 - m00,
            'average_student_marginal': ((m10 - m00) + (m11 - m01)) / 2,
            'average_D_marginal': ((m01 - m00) + (m11 - m10)) / 2}  # 数值分解不解释成唯一因果百分比。


def noise_inputs(mel, device):
    training.seed_everything(NOISE_SEED)
    source_noise = torch.randn_like(mel)  # 生产顺序第一项，供q950构造。
    score_noise = torch.randn_like(mel)  # 第二项也生成并存SHA，供math独立probe核对。
    timesteps = torch.randint(1, 1001, (32,), device=device, dtype=torch.long)  # 第三项，范围1..1000。
    return source_noise, score_noise, timesteps


def require_output_capacity():
    free = os.statvfs(DEV)
    if free.f_bavail * free.f_frsize < 3 * 1024**3 + 64 * 1024**2:
        raise RuntimeError('Preserve3GiB system space plus64MiB diagnostic allowance.')


def verify_context():
    dependencies = {str(TOOLS / name): value for name, value in MODULE_SHAS.items()}
    contracts = {}
    for step, item in CHECKPOINTS.items():
        dependencies[str(item['directory'] / item['file'])] = item['sha256']
        path = item['directory'] / 'run_contract.json'
        dependencies[str(path)] = item['contract_file_sha256']
        contract = json.loads(path.read_text(encoding='utf-8'))
        if training.canonical_json_sha256(contract) != item['contract_sha256']:
            raise RuntimeError('Training contract identity changed.')
        contracts[step] = contract
        dependencies.update(contract['file_sha256'])
    resume.verify_files(dependencies)
    grand = json.loads((resume.PARENT / 'run_contract.json').read_text(encoding='utf-8'))
    if training.canonical_json_sha256(grand) != resume.PARENT_CONTRACT_SHA or training.collect_source_fingerprints(ROOT) != grand['source_sha256']:
        raise RuntimeError('Original production source or initial development contract changed.')
    development, dev_hashes = resume.load_development(DEV / 'development_contract.json')
    dependencies.update(dev_hashes)
    config = baseline.replace(baseline.PAPER_STUDENT_CONFIG, num_workers=0)
    for contract in contracts.values():
        if contract['configuration'] != config.as_dict() or contract['training_indices'] != development['retained_train_indices'] or contract['excluded_train_indices'] != development['excluded_train_indices']:
            raise RuntimeError('Original ACDD settings or52-exclusion training pool changed.')
    child = contracts[3107]
    if child['kind'] != 'acdd2000_to4000_development' or child['lineage']['parent_checkpoint_sha256'] != CHECKPOINTS[2000]['sha256']:
        raise RuntimeError('Stopped checkpoint is not the unchanged ACDD2000 continuation.')
    if child['original_step0_guard_anchor'] != contracts[2000]['original_step0_guard_anchor']:
        raise RuntimeError('S0 anchor changed.')
    completion_path = CHECKPOINTS[3107]['directory'] / 'completion.json'
    completion = json.loads(completion_path.read_text(encoding='utf-8'))
    dependencies[str(completion_path)] = baseline.sha256_file(completion_path)
    if completion.get('status') != 'requested_stop' or completion.get('global_step') != 3107 or completion.get('last_saved_global_step') != 3107 or completion.get('run_contract_sha256') != CHECKPOINTS[3107]['contract_sha256']:
        raise RuntimeError('Expected the confirmed user-requested stop at3107.')
    return config, contracts, development, dependencies


def load_network_states(step, config):
    item = CHECKPOINTS[step]
    payload = resume.load_torch_checkpoint(item['directory'] / item['file'])  # CPU读取可信原归档，不调用模型或优化器。
    if payload['global_step'] != step or payload['run_contract_sha256'] != item['contract_sha256'] or payload['configuration'] != config.as_dict():
        raise RuntimeError('Full checkpoint metadata differs from its frozen contract.')
    selected = {'student': payload['student'], 'discriminator': payload['discriminator'], 'development': payload['development']}
    del payload  # 丢弃未使用的Adam/RNG归档，不创建优化器，不执行任何更新。
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--probe-approved', action='store_true')
    args = parser.parse_args()
    if not args.probe_approved or not 1 <= args.max_seconds <= 300:
        raise RuntimeError('Explicit read-only GPU probe approval and a bounded duration are required.')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)  # 与训练和其他GPU诊断互斥，main之外不获取锁。
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6') or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve():
        raise RuntimeError('Use the original myconda training runtime.')
    if not torch.cuda.is_available() or training.git_head(ROOT) != baseline.EXPECTED_HEAD:
        raise RuntimeError('Rental GPU or reviewed production HEAD unavailable.')
    output = args.output_dir.resolve()
    if output.parent != DEV or output.exists():
        raise RuntimeError('Use a new direct development child directory; all archives stay unchanged.')
    require_output_capacity()  # GPU前先查一次，不能做完模型计算才发现无法保留报告。
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError('Read-only diagnostic time budget exceeded.')
    config, contracts, development, dependencies = verify_context()
    print('Frozen inputs and checkpoint identities verified; preparing shared TRAIN3108 and fixed32.', flush=True)
    deadline()
    torch.set_num_threads(2)
    device = torch.device('cuda')
    training.seed_everything(SEED)
    dataset = baseline.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100,
                                       bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387:
        raise RuntimeError('Training manifest changed.')
    plan = resume.GlobalBatchPlan(development['retained_train_indices'])
    indices = plan.for_step(TRAIN_STEP)
    if len(indices) != 32 or set(indices) & set(development['excluded_train_indices']):
        raise RuntimeError('Probe must use a full retained TRAIN batch and exclude all52 development utterances.')
    batch = baseline.batch_for(dataset, indices, CROP_SEED)
    tensors = {name: batch[name].to(device) for name in ('mel', 'bnf', 'spk_emb')}
    source_noise, score_noise, score_timesteps = noise_inputs(tensors['mel'], device)
    protocol = development['reconstruction_protocol']
    if protocol['source_noise_device'] != 'cuda':
        raise RuntimeError('Original development noise device changed.')
    dev_batch = baseline.batch_for(dataset, development['reconstruction_dev_indices'], protocol['crop_seed'])
    dev_noise = torch.randn(dev_batch['mel'].shape, device=device,
        generator=torch.Generator(device=device).manual_seed(protocol['source_noise_seed']))
    parent = contracts[2000]
    if any(tensor_sha(dev_batch[key]) != parent['fixed_development_inputs'][key] for key in ('mel', 'bnf', 'spk_emb')) or tensor_sha(dev_noise) != parent['fixed_development_source_noise_sha256']:
        raise RuntimeError('Original fixed32 development inputs/noise changed.')
    teacher, student, _ = baseline.load_teacher_and_student(baseline.TEACHER, config, device)
    for model in (teacher, student):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    vocoder, vconfig, _ = baseline.load_frozen_vocoder(ROOT, baseline.VOCODER_DIR / 'generator_step_2500000.pt',
        baseline.VOCODER_DIR / 'config.json', baseline.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = baseline.build_paper_discriminator(ROOT, vconfig, config, device)
    diffusion = baseline.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = baseline.load_mel_statistics(ROOT / 'data', device)
    fixed_fingerprints = {name: fingerprint_model(model) for name, model in (('teacher', teacher), ('vocoder', vocoder))}
    states = {step: load_network_states(step, config) for step in CHECKPOINTS}
    print('S/D checkpoint parameters loaded; beginning read-only student predictions.', flush=True)
    dev_scores, mel_scores, waves, model_fingerprints = {}, {}, {}, {}
    with torch.no_grad():  # 全段禁止构建参数梯度图，也没有优化器入口。
        real_wave = vocoder(training.denormalize_mel(tensors['mel'], mean, std))
        noisy_start = diffusion.q_sample(tensors['mel'], torch.full((32,), 950, device=device, dtype=torch.long), source_noise)
        for step in CHECKPOINTS:
            deadline()
            print('Reading S{} on the common TRAIN batch and original fixed32.'.format(step), flush=True)
            before = load_exact_weights(student, states[step]['student'])
            prediction, actual_start = baseline.make_student_prediction(student, diffusion, tensors['mel'], tensors['spk_emb'],
                tensors['bnf'], config.student_start_step, source_noise)
            if not torch.equal(actual_start, noisy_start):
                raise RuntimeError('Student did not receive the recorded common q950 input.')
            waves[step] = vocoder(training.denormalize_mel(prediction, mean, std))
            mel_scores[step] = {'mel_l1': float((prediction - tensors['mel']).abs().mean()),
                               'prediction': statistics(prediction), 'prediction_identity': tensor_identity(prediction)}
            dev_scores[step] = baseline.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
            saved = states[step]['development']
            if any(abs(dev_scores[step][k] - saved[k]) > 1e-7 for k in ('mel_l1', 'prediction_rms', 'reference_rms', 'cosine_reference')):
                raise RuntimeError('Original fixed32 checkpoint readouts failed to replay.')
            assert_unchanged(student, before)
            model_fingerprints['S' + str(step)] = before
        matrix = {}
        for d_step in CHECKPOINTS:
            before = load_exact_weights(discriminator, states[d_step]['discriminator'])
            for s_step in CHECKPOINTS:
                matrix[f'S{s_step}_D{d_step}'] = {'branches': []}
            for number, branch in enumerate(discriminator.loss_branches()):
                deadline()
                print('Reading D{} branch {}/8 against both cached student waveforms.'.format(d_step, number + 1), flush=True)
                real_score, real_features = branch(real_wave)
                for s_step in CHECKPOINTS:
                    fake_score, fake_features = branch(waves[s_step])
                    row = branch_measurement(real_score, real_features, fake_score, fake_features)
                    row.update(branch_index=number, family='MPD' if number < 5 else 'MRD')
                    matrix[f'S{s_step}_D{d_step}']['branches'].append(row)
                    del fake_score, fake_features
                del real_score, real_features  # 一次只保留一个D分支，避免八支特征同时驻留。
            assert_unchanged(discriminator, before)
            model_fingerprints['D' + str(d_step)] = before
    for name, model in (('teacher', teacher), ('vocoder', vocoder)):
        assert_unchanged(model, fixed_fingerprints[name])
    model_fingerprints.update(fixed_fingerprints)
    for cell in matrix.values():
        for key in ('fm_raw', 'fm_hidden_raw', 'fm_score_raw', 'student_adv', 'discriminator_loss_same_state'):
            cell[key] = sum(row[key] for row in cell['branches'])
        cell['fm_weighted'] = 2 * cell['fm_raw']
    effects = {key: matrix_effects({name: cell[key] for name, cell in matrix.items()})
               for key in ('fm_raw', 'fm_weighted', 'fm_hidden_raw', 'fm_score_raw', 'student_adv', 'discriminator_loss_same_state')}
    identities = {name: tensor_identity(value) for name, value in tensors.items()}
    identities.update(source_noise=tensor_identity(source_noise), score_noise=tensor_identity(score_noise),
                      score_timesteps=tensor_identity(score_timesteps), q950=tensor_identity(noisy_start),
                      real_wave=tensor_identity(real_wave),
                      fake_S2000=tensor_identity(waves[2000]), fake_S3107=tensor_identity(waves[3107]))
    resume.verify_files(dependencies)
    deadline()
    require_output_capacity()  # 检查期间其他工作也可能占盘，写报告前再次确认。
    result = {'status': 'complete_readonly_crossed_GAN_diagnostic', 'training_updates': 0,
        'train_global_step_for_inputs_only': TRAIN_STEP, 'indices': indices, 'crop_seed': CROP_SEED, 'noise_seed': NOISE_SEED,
        'score_timesteps': score_timesteps.cpu().tolist(),  # 列出32个真实时刻，SHA同时保留在input_identities中。
        'input_identities': identities, 'matrix': matrix, 'effects': effects,
        'train_reconstruction': mel_scores, 'fixed32_development': dev_scores,
        'model_parameters_buffers_unchanged': True, 'all_parameter_gradients_absent': True,
        'model_fingerprints_before_equal_after': model_fingerprints,
        'source_sha256': dependencies, 'script_sha256': baseline.sha256_file(Path(__file__)),
        'elapsed_seconds': time.monotonic() - started,
        'limitations': ['One retained TRAIN minibatch, shared crop and noise; not the moving50-step training window.',
                       'D checkpoints are evaluated in the same fixed state; logged training D loss precedes each D update whereas student FM follows it.',
                       'Hidden features and final score maps are separated. Weight/feature changes need not be pure scalar rescaling.',
                       'Relative feature normalization is descriptive only; the unchanged training objective still uses raw sum of layer means and weight2.',
                       'The2x2 interaction and marginals describe fixed-input checkpoint differences, not a unique causal allocation or speech-quality metric.',
                       'No optimizer was constructed/restored/stepped; no model/audio is saved. Fixed32 reconstruction is separate from text-unseen conversion performance.']}
    output.mkdir(exist_ok=False)
    baseline.save_json(output / 'results.json', result)
    print(json.dumps({'output': str(output), 'status': result['status'], 'effects': effects,
                      'fixed32_mel_l1': {step: scores['mel_l1'] for step, scores in dev_scores.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()  # 仅主线程批准的只读GPU调用执行这里；导入测试不触发main。
