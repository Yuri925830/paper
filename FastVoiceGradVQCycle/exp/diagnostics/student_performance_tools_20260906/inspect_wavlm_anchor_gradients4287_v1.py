"""固定 TRAIN8 的只读 WavLM 锚点梯度检查；不更新、评分、保存波形或模型。"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
OUTPUT = DEV / 'wavlm_anchor_gradients4287_v1'
PROTOCOL = DEV / 'WAVLM_ANCHOR_GRADIENTS4287_PROTOCOL_CN.md'
INPUT = DEV / 'acdd_adam_descent_readonly_3107_v1/fixed_inputs.pt'
INPUT_SHA = '296a7b4a7c289f0d19409947cd2c4dcb54754ae5636a58af43e658bc12b17da8'
BASE = 'inspect_teacher30_anchor_train16_v1.py'
BASE_SHA = '5bb79c3c4174800ed8296bccd82691cd4bd7856d8aab138466336410dc4e86db'
MC = 'acdd_mc2_train_batch_v1.py'
MC_SHA = 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16'
STATS = 'mc2_component_gradients_v1.py'
STATS_SHA = 'fefc4195a45107dae5d0b8c66dbe600a17abfcff183059893bd0c1c60bb09a50'
SHUFFLE = 'acdd_mc2_conversion_anchor4287_v1.py'
SHUFFLE_SHA = '56d366da65492210b2b3754afbd54dcf3d2fef416bae74c2e3f0489e8473c640'
LOSS = 'wavlm_conv_perceptual_loss_v1.py'
ASSET = TOOLS / 'assets/wavlm_large_conv_c1423ed_v1.pt'
ASSET_SHA = '82976c7b48f79da0d80a5277a1faedbae66bfae86942379f041aca62301be49b'
ASSET_RECEIPT = ASSET.with_suffix('.receipt.json')
ASSET_RECEIPT_SHA = 'acbf3cfdd59fa1b4e0ca15c8b31f01d114e554fe480200a260bab05c0d32ac6b'
COUNTS = {'teacher': 62, 'student': 1, 'vocoder': 3}
CAP, RESERVE = 8 * 1024**2, 1024**3
SHAPES = {'x0': (32, 80, 128), 'bnf': (32, 144, 128), 'spk_emb': (32, 256),
          'source_noise': (32, 80, 128), 'score_noise': (32, 80, 128),
          'score_timesteps': (32,), 'q950': (32, 80, 128)}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''): value.update(block)
    return value.hexdigest()


def load(name, sha):
    path = TOOLS / name
    if path.is_symlink() or digest(path) != sha: raise RuntimeError('Source SHA mismatch: ' + name)
    spec = importlib.util.spec_from_file_location('_wavlm_probe_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def shuffle_function():
    # 只提取已经逐行核过的排列函数；不导入尚未运行的训练附件或修改其全局变量。
    path = TOOLS / SHUFFLE
    if digest(path) != SHUFFLE_SHA: raise RuntimeError('Shuffle source changed')
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'speaker_shuffle']
    if len(nodes) != 1 or nodes[0].decorator_list: raise RuntimeError('Unexpected shuffle definition')
    namespace = {'random': random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['speaker_shuffle']


def choose(manifest, indices, development, test_rows, shuffle):
    retained, excluded = set(development['retained_train_indices']), set(development['excluded_train_indices'])
    if len(excluded) != 52 or retained & excluded or retained | excluded != set(range(len(manifest))):
        raise RuntimeError('Original training complement changed')
    if len(indices) != 32 or len(set(indices)) != 32 or any(type(i) is not int or i not in retained for i in indices):
        raise RuntimeError('Require the fixed32 distinct training indices')
    test_speakers = {row['speaker'] for row in test_rows}
    speakers = [manifest[i]['speaker'] for i in indices]
    if any(manifest[i]['split'] != 'train' or manifest[i]['speaker'] in test_speakers for i in indices):
        raise RuntimeError('Training/reference identity overlaps formal test')
    permutation = shuffle(speakers, 88007000)
    if sorted(permutation) != list(range(32)) or any(speakers[i] == speakers[j] for i, j in enumerate(permutation)):
        raise RuntimeError('Cross-speaker mapping is not a full bijection')
    rows = [{'pair_id': i, 'source_index': indices[i], 'reference_index': indices[permutation[i]],
             'source_speaker': speakers[i], 'source_utt': manifest[indices[i]]['utt_id'],
             'reference_speaker': speakers[permutation[i]], 'reference_utt': manifest[indices[permutation[i]]]['utt_id']}
            for i in range(8)]
    return rows, permutation


def validate_asset_receipt(receipt, sha):
    if (sha, receipt.get('status'), receipt.get('source_revision'), receipt.get('source_sha256'),
            receipt.get('source_size_bytes'), receipt.get('tensor_count'), receipt.get('parameter_count'),
            receipt.get('cuda_initialized'), receipt.get('forward_calls'), receipt.get('model_initializations')) != (
            ASSET_SHA, 'completed_verified_official_conv_extraction', 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c',
            'fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d', 1261990257, 21, 4206592, False, 0, 0):
        raise RuntimeError('Official convolution asset acquisition identity differs')
    if receipt.get('output') != {'path': str(ASSET), 'size_bytes': 16832166, 'sha256': ASSET_SHA, 'state_format': 'plain_conv_only_state_dict'}:
        raise RuntimeError('Wrong extracted convolution asset')
    if receipt.get('source_config_sha256') != 'a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d': raise RuntimeError('Official WavLM config differs')


def prepare(args, base, check=lambda: None):
    # 只取本次真实依赖；不会验证旧已归档WAV、3447或其他已退休学生文件。
    pins = {path: sha for path, sha in base.FIXED.items()
            if path not in {str(TOOLS / 'evaluate_development_baselines.py'), str(TOOLS / 'evaluate_development_candidates.py'), str(base.METRIC_ID_SOURCE)}}
    pins.update({str(TOOLS / n): h for n, h in ((BASE, BASE_SHA), (MC, MC_SHA), (STATS, STATS_SHA), (SHUFFLE, SHUFFLE_SHA),
        (LOSS, args.helper_sha256), ('test_' + LOSS, args.helper_tests_sha256),
        (Path(__file__).name, args.driver_sha256), ('test_' + Path(__file__).name, args.tests_sha256))})
    pins.update({str(INPUT): INPUT_SHA, str(ASSET): args.asset_sha256, str(PROTOCOL): args.protocol_sha256})
    pins[str(ASSET_RECEIPT)] = ASSET_RECEIPT_SHA
    if digest(base.regular(ASSET_RECEIPT)) != ASSET_RECEIPT_SHA: raise RuntimeError('Official extraction receipt changed')
    asset_receipt = base.read_json(ASSET_RECEIPT); validate_asset_receipt(asset_receipt, args.asset_sha256)
    pins.update(asset_receipt['source_pins'])
    pins.update({str(ROOT / n): h for n, h in base.CORE.items() if not n.startswith('evaluation/') and 'infer_voicegrad30' not in n})
    # Git干净HEAD加新预执行SHA绑定实际依赖包；并不冒称这些全是父契约里的pin。
    for package in ('voicegrad', 'fastvoicegrad', 'hifigan'):
        for path in sorted((ROOT / package).rglob('*.py')): pins.setdefault(str(base.regular(path)), digest(path))
    for path, sha in pins.items():
        check()
        if digest(base.regular(path)) != sha: raise RuntimeError('Bound input differs: ' + path)
    contract = base.read_json(base.PARENT_RUN / 'run_contract.json')
    base.validate_parent(contract, base.read_json(base.PARENT_RUN / 'candidate_mc2/completion.json'))
    proof = base.read_json(base.TRAIN_PROOF)
    if (proof.get('global_input_step'), proof.get('crop_seed'), proof.get('noise_seed')) != (3108, 104342, 204342):
        raise RuntimeError('Wrong original TRAIN32 input provenance')
    rows, permutation = choose(base.read_csv(ROOT / 'data/metadata/train_manifest.csv'), proof['train_indices'],
        base.read_json(DEV / 'development_contract.json'), base.read_csv(ROOT / 'data/metadata/test_manifest.csv'), shuffle_function())
    for row in rows:
        path = ROOT / 'data/mel' / row['source_speaker'] / (row['source_utt'] + '.npy')
        pins[str(base.regular(path))] = digest(path)
        frames = base.frame_count(base.npy_shape(path), 80)
        row.update(original_frames=frames, valid_frames=min(frames, 128), valid_samples=min(frames, 128) * 256)
    lengths = [row['valid_samples'] for row in rows]
    wavlm_calls = (2 if len(set(lengths)) == 1 else 16) + (2 if len(set(lengths[:4])) == 1 else 8)
    return {'kind': 'readonly_conversion_TRAIN8_LMOS_gradient4287', 'files': pins, 'pairs': rows,
        'full32_train_indices': proof['train_indices'], 'speaker_permutation32': permutation,
        'raw_input_sha256': {k: proof['input_sha256'][k] for k in SHAPES}, 'configuration': contract['configuration'],
        'forward_limits': dict(COUNTS, wavlm=wavlm_calls), 'source_step': 3108, 'private_second_draw_step_label': 4288,
        'not_actual_next_training_batch': True, 'reverse_seed': 88008000, 'repeat_reverse_seed': 89008000,
        'calibration': 'norm(45*localMC2_conversion)/norm(LMOS), one aggregateTRAIN8, static engineering weight',
        'no_metrics_no_optimizer_no_checkpoint_no_wav': True, 'maximum_seconds': 600,
        'output_bytes_limit': CAP, 'reserve_bytes': RESERVE}


class Budget:
    def __init__(self, output=OUTPUT): self.output = Path(output)
    def used(self):
        total = 0
        if self.output.is_symlink(): raise RuntimeError('Unsafe output')
        if self.output.exists():
            for path in self.output.rglob('*'):
                if path.is_symlink() or not path.is_file(): raise RuntimeError('Only flat JSON evidence is permitted')
                if '.json' not in path.name: raise RuntimeError('No model/waveform output')
                total += path.stat().st_size
        return total
    def check(self):
        used = self.used(); status = os.statvfs(self.output.parent)
        free = status.f_bavail * status.f_frsize
        if used > CAP or free < RESERVE + CAP - used + 1024**2: raise RuntimeError('Output/reserve budget exhausted')
        return {'used_bytes': used, 'free_bytes': free}
    def write(self, name, value, terminal=False):
        if Path(name).name != name or not name.endswith('.json'): raise RuntimeError('Only exact child JSON paths')
        data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        if self.used() + len(data) > CAP - (0 if terminal else 262144): raise RuntimeError('Reserve terminal evidence space')
        self.check(); path = self.output / name; temporary = self.output / (name + '.tmp')
        if path.exists() or temporary.exists(): raise RuntimeError('No evidence overwrite')
        with temporary.open('xb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.rename(temporary, path)


def second32(core, full32, diffusion):
    # 只改变新函数私有globals的准入范围，函数code、采样顺序、seed公式原样复用。
    if tuple(full32.shape) != (32, 80, 128): raise RuntimeError('Generate MC2 draw at original32 shape first')
    context = dict(core.second_score_draws.__globals__, START=4287, STOP=4288)
    clone = types.FunctionType(core.second_score_draws.__code__, context, core.second_score_draws.__name__)
    return clone(full32, diffusion, 4288)


def teacher_endpoint(torch, teacher, diffusion, q, speaker, bnf, seed, tensor_id):
    observed = []
    def first(module, values):
        if not observed: observed.append((tensor_id(values[0]), values[1].detach().cpu().tolist(), tensor_id(values[2]), tensor_id(values[3])))
    handle = teacher.register_forward_pre_hook(first)
    try:
        with torch.no_grad():
            target = diffusion.sample(teacher, q, speaker, bnf, num_steps=30, start_step=950,
                generator=torch.Generator(device=q.device).manual_seed(seed), diffuse_source=False)
        expected = (tensor_id(q), [950] * q.shape[0], tensor_id(speaker), tensor_id(bnf))
        if observed != [expected]: raise RuntimeError('Teacher actual first input/condition differs')
        if target.shape != q.shape or not bool(torch.isfinite(target).all()) or target.requires_grad: raise RuntimeError('Invalid teacher endpoint')
        return target, {'first_input': observed[0], 'endpoint': tensor_id(target), 'reverse_seed': seed}
    finally: handle.remove()


def readout(value):
    # 只把现有标量/逐句结果转JSON，绝不保存梯度向量或重算评价指标。
    result = {}
    for key in ('loss', 'feature_mse', 'feature_weighted', 'stft_magnitude_l1', 'per_example_loss',
                'per_example_feature_mse', 'per_example_stft_magnitude_l1'):
        result[key] = value[key].detach().cpu().tolist()
    for key in ('resampled_lengths', 'feature_frame_lengths', 'execution_path', 'method'): result[key] = value[key]
    return result


def calibration(statistics):
    a, b = statistics['norms']['45MC2_conversion'], statistics['norms']['LMOS']
    if not all(math.isfinite(x) and x > 0 for x in (a, b)): raise RuntimeError('Cannot calibrate zero/nonfinite gradient norm')
    ratio = a / b
    if not math.isfinite(ratio) or ratio <= 0: raise RuntimeError('Nonfinite calibration ratio')
    return {'anchor_weight': ratio, 'numerator_norm': a, 'denominator_norm': b,
            'cosine': statistics['cosines']['LMOS']['45MC2_conversion'],
            'automatic_training_approval': False, 'not_GradNorm': True, 'not_performance_evidence': True}


def compute(args, plan, base, budget, check, ledger):
    import dataclasses
    import numpy as np
    import torch
    from fastvoicegrad import train as training
    from fastvoicegrad.components import load_teacher_and_student, load_torch_checkpoint, load_frozen_vocoder, load_mel_statistics
    from fastvoicegrad.config import PAPER_STUDENT_CONFIG
    from voicegrad.diffusion import VoiceGradDiffusion
    core, helper, stats = load(MC, MC_SHA), load(LOSS, args.helper_sha256), load(STATS, STATS_SHA)
    config = dataclasses.replace(PAPER_STUDENT_CONFIG, num_workers=0)
    if config.as_dict() != plan['configuration']: raise RuntimeError('Parent/actual configuration differs')
    if (config.score_target, config.score_weight, config.lambda_dist, config.lambda_fm) != ('clean_x0', 'sqrt_alpha_bar', 45., 2.): raise RuntimeError('Current MC2 math changed')
    flags = lambda: {'matmul': torch.backends.cuda.matmul.allow_tf32, 'cudnn': torch.backends.cudnn.allow_tf32,
        'benchmark': torch.backends.cudnn.benchmark, 'deterministic': torch.backends.cudnn.deterministic,
        'threads': torch.get_num_threads()}
    original_flags = flags()
    if (torch.__version__, torch.version.cuda) != ('1.13.1+cu116', '11.6') or any(original_flags[k] for k in ('matmul', 'benchmark', 'deterministic')) or not original_flags['cudnn']:
        raise RuntimeError('Frozen inference/training numerical runtime changed')
    device = torch.device('cuda'); rng = training.capture_rng_state(); models, snapshots, cleanup = {}, {}, {}
    try:
        check(); raw = load_torch_checkpoint(INPUT)
        for key, shape in SHAPES.items():
            value = raw[key]
            if tuple(value.shape) != shape or value.dtype != (torch.long if key == 'score_timesteps' else torch.float32) or not bool(torch.isfinite(value).all()): raise RuntimeError('Invalid archived tensor: ' + key)
            if base.tensor_id(value)['sha256'] != plan['raw_input_sha256'][key]: raise RuntimeError('Actual raw tensor identity differs: ' + key)
        if not bool(((raw['score_timesteps'] >= 1) & (raw['score_timesteps'] <= 1000)).all()): raise RuntimeError('Invalid first teacher times')
        inputs = {key: raw[key][:8].clone().to(device) for key in SHAPES}
        full32 = raw['x0'].to(device)
        speaker = raw['spk_emb'][plan['speaker_permutation32'][:8]].clone().to(device)
        del raw
        input_ids = {key: base.tensor_id(value) for key, value in inputs.items()}; input_ids['target_speaker'] = base.tensor_id(speaker)
        teacher, student, ignored = load_teacher_and_student(base.TEACHER, config, device); del ignored
        models.update(teacher=teacher, student=student)
        payload = load_torch_checkpoint(base.STUDENT)
        base.validate_parent(base.read_json(base.PARENT_RUN / 'run_contract.json'), base.read_json(base.PARENT_RUN / 'candidate_mc2/completion.json'), payload)
        student.load_state_dict(payload['student'], strict=True); del payload
        vocoder, ignored, ignored2 = load_frozen_vocoder(ROOT, base.VOCODER / 'generator_step_2500000.pt',
            base.VOCODER / 'config.json', base.VOCODER / 'final_checkpoint_attestation_comments_20260906.json', config, device)
        models['vocoder'] = vocoder; del ignored, ignored2
        diffusion = VoiceGradDiffusion(1000, offset=.008).to(device); models['diffusion'] = diffusion
        asset = load_torch_checkpoint(ASSET)
        asset_receipt = base.read_json(ASSET_RECEIPT)
        if set(asset) != {row['saved_key'] for row in asset_receipt['tensors']}: raise RuntimeError('Actual convolution tensor key set differs')
        for row in asset_receipt['tensors']:
            value = asset[row['saved_key']]
            if base.tensor_id(value) != {'sha256': row['sha256'], 'shape': row['shape'], 'dtype': 'torch.float32'} or value.numel() != row['numel']: raise RuntimeError('Actual official convolution tensor differs')
        encoder = helper.FrozenWavLMConvEncoder(asset).to(device); models['wavlm'] = encoder; del asset
        for name, model in models.items():
            model.train(name == 'student').requires_grad_(name == 'student')
            if any(p.grad is not None for p in model.parameters()): raise RuntimeError('Unexpected existing gradient')
            snapshots[name] = base.snapshot(model)
            if name in ledger.limits: ledger.add(model, name)
        params = dict(student.named_parameters())
        if len(params) != 71 or not all(p.requires_grad for p in params.values()): raise RuntimeError('Require all71 student parameters')
        mean, std = load_mel_statistics(ROOT / 'data', device)
        training.restore_rng_state(rng, device); check()
        prediction, q = training.make_student_prediction(student, diffusion, inputs['x0'], speaker, inputs['bnf'], 950, inputs['source_noise'])
        if not torch.equal(q, inputs['q950']) or prediction.shape != q.shape or not bool(torch.isfinite(prediction).all()): raise RuntimeError('Source q950 replay or student prediction failed')
        target, primary = teacher_endpoint(torch, teacher, diffusion, q, speaker, inputs['bnf'], 88008000, base.tensor_id)
        repeat, repeated = teacher_endpoint(torch, teacher, diffusion, q[:4], speaker[:4], inputs['bnf'][:4], 89008000, base.tensor_id)
        # 学生波形保留完整G梯度图；冻结G参数，不截幅、不转PCM、不做评价归一化。
        generated = vocoder(training.denormalize_mel(prediction, mean, std))
        with torch.no_grad():
            target_wave = vocoder(training.denormalize_mel(target, mean, std))
            repeat_wave = vocoder(training.denormalize_mel(repeat, mean, std))
        if generated.shape != (8, 1, 32768) or target_wave.shape != generated.shape or repeat_wave.shape != (4, 1, 32768): raise RuntimeError('Unexpected128-frame waveform shape')
        lengths = [r['valid_samples'] for r in plan['pairs']]
        lmos = helper.wavlm_conv_perceptual_loss(encoder, generated, target_wave, lengths)
        with torch.no_grad(): repeated_loss = helper.wavlm_conv_perceptual_loss(encoder, target_wave[:4], repeat_wave, lengths[:4])
        noise2, t2 = second32(core, full32, diffusion)
        mc_rows, losses = [], []
        for noise, times in ((inputs['score_noise'], inputs['score_timesteps']), (noise2[:8], t2[:8])):
            check()
            loss, local_target = core.score_distillation_loss(prediction, teacher, diffusion, times, noise, speaker, inputs['bnf'], target_mode='clean_x0', weight_mode='sqrt_alpha_bar')
            losses.append(loss); mc_rows.append(core.score_record(prediction, local_target, times, diffusion, loss))
        weighted_mc2 = 45. * ((losses[0] + losses[1]) / 2.)
        banks, mel_gradients = {}, {}
        for label, loss in (('LMOS', lmos['loss']), ('45MC2_conversion', weighted_mc2)):
            check()
            if loss.ndim or not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite/nonscalar differentiable loss')
            gradients = torch.autograd.grad(loss, tuple(params.values()) + (prediction,), retain_graph=label == 'LMOS', allow_unused=True)
            if gradients[-1] is None or any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients): raise RuntimeError('Disconnected/nonfinite diagnostic gradient')
            banks[label] = {name: None if value is None else value.detach().cpu() for name, value in zip(params, gradients[:-1])}
            mel_gradients[label] = gradients[-1].detach().cpu().double().flatten(1).norm(dim=1).tolist()
            del gradients
        statistics = stats.blockwise_statistics(torch=torch, banks=banks, check=check)
        if statistics['parameter_count'] != 71: raise RuntimeError('Gradient parameter identity count differs')
        if ledger.counts != plan['forward_limits']: raise RuntimeError('Actual forward budget mismatch')
        result = {'counts': dict(ledger.counts), 'autograd_grad_calls': 2, 'optimizer_updates': 0, 'checkpoint_saves': 0,
            'input_identities': input_ids, 'student_prediction': base.tensor_id(prediction), 'primary_teacher': primary, 'repeat_teacher': repeated,
            'student_teacher': readout(lmos), 'teacher_teacher_first4': readout(repeated_loss),
            'mel_L1_student_teacher': (prediction.detach() - target).abs().flatten(1).mean(1).cpu().tolist(),
            'mel_L1_teacher_repeat_first4': (target[:4] - repeat).abs().flatten(1).mean(1).cpu().tolist(),
            'MC2': {'first': mc_rows[0], 'second': mc_rows[1], 'weighted45_mean': float(weighted_mc2.detach()),
                'full32_noise2': base.tensor_id(noise2), 'full32_t2': base.tensor_id(t2), 't2_first8': t2[:8].cpu().tolist(),
                'second_seed': 74004288, 'mixture': 'mean_of_two_weighted_L1_batch_means'},
            'gradient_statistics': statistics, 'per_utterance_mel_output_gradient_norm': mel_gradients,
            'calibration': calibration(statistics), 'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'flags': original_flags}}
        for key, identity in input_ids.items():
            if base.tensor_id(speaker if key == 'target_speaker' else inputs[key]) != identity: raise RuntimeError('Diagnostic input mutated')
        budget.write('results.json', result)
        return result, cleanup
    finally:
        errors = []
        for name, model in models.items():
            try:
                if name in snapshots and base.snapshot(model) != snapshots[name]: raise RuntimeError('Model/state/buffer/gradient changed: ' + name)
            except BaseException as caught: errors.append(repr(caught))
        try:
            cleanup['rng_unchanged'] = base.same_rng(torch, np, training.capture_rng_state(), rng)
            if not cleanup['rng_unchanged']: errors.append('default RNG changed')
        except BaseException as caught: errors.append(repr(caught))
        for action in (lambda: training.restore_rng_state(rng, device), ledger.close):
            try: action()
            except BaseException as caught: errors.append(repr(caught))
        cleanup.update(model_snapshots_unchanged=not errors, checked_models=list(snapshots), flags_unchanged=flags() == original_flags)
        if not cleanup['flags_unchanged']: errors.append('runtime flags/threads changed')
        try:
            torch.backends.cuda.matmul.allow_tf32 = original_flags['matmul']; torch.backends.cudnn.allow_tf32 = original_flags['cudnn']
            torch.backends.cudnn.benchmark = original_flags['benchmark']; torch.backends.cudnn.deterministic = original_flags['deterministic']
            if torch.get_num_threads() != original_flags['threads']: torch.set_num_threads(original_flags['threads'])
            cleanup['restored_rng_and_flags'] = base.same_rng(torch, np, training.capture_rng_state(), rng) and flags() == original_flags
            if not cleanup['restored_rng_and_flags']: errors.append('RNG/flags restoration verification failed')
        except BaseException as caught: errors.append(repr(caught))
        # 调用栈中的大张量离开compute后，由main在仍持锁时做最后显存回收。
        models.clear()
        if errors: raise RuntimeError('Read-only cleanup failed: ' + repr(errors))


def finish(base, budget, clock, ledger, locks, owned, result, cleanup, error):
    failures = []
    for action in (ledger.close, clock.close):
        try: action()
        except BaseException as caught: failures.append(repr(caught))
    published = False
    def failure():
        if owned and not (OUTPUT / 'failure.json').exists():
            budget.write('failure.json', {'error': repr(error), 'cleanup_errors': failures, 'counts': dict(ledger.counts), 'no_retry': True}, terminal=True)
    try:
        if error is not None or failures:
            if error is None: error = RuntimeError(repr(failures))
            failure()
        elif owned:
            clock.check()
            budget.write('completion.json', {'status': 'completed_readonly_TRAIN8_gradient_diagnostic', 'cleanup': cleanup,
                'cleanup_verified': True, 'counts': result['counts'], 'results_sha256': digest(OUTPUT / 'results.json'),
                'contract_sha256': digest(OUTPUT / 'run_contract.json'), 'elapsed_seconds': time.monotonic() - clock.started,
                'automatic_training_approval': False}, terminal=True)
            published = True
    except BaseException as caught:
        if error is None: error = caught
        try: failure()
        except BaseException as last: print('Failure receipt publication failed: ' + repr(last), file=sys.stderr)
    for handle in reversed(locks):
        try: handle.close()
        except BaseException as caught: failures.append(repr(caught))
    if failures:
        if error is None: error = RuntimeError('Lock/cleanup failed: ' + repr(failures))
        if published: os.rename(OUTPUT / 'completion.json', OUTPUT / 'completion_invalidated.json')
        failure()
    return error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('preflight', 'run'))
    parser.add_argument('--run-approved', action='store_true'); parser.add_argument('--plan-sha256')
    for name in ('driver', 'tests', 'protocol', 'helper', 'helper-tests', 'asset'): parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if name.endswith('sha256') and value is not None and (len(value) != 64 or any(c not in '0123456789abcdef' for c in value)): raise RuntimeError('Full lowercaseSHA required')
    if not sys.platform.startswith('linux') or Path.cwd().resolve() != ROOT or not (ROOT / 'AGENTS.md').is_file(): raise RuntimeError('Only registered Matpool rental')
    base = load(BASE, BASE_SHA)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != base.HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip(): raise RuntimeError('Clean frozen GitHEAD required')
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('No repeat/overwrite')
    sys.path.insert(0, str(ROOT)); budget = Budget()
    if args.phase == 'preflight':
        plan = prepare(args, base); disk = budget.check()
        if 'torch' in sys.modules: raise RuntimeError('Preflight must not import Torch')
        print(json.dumps({'plan': plan, 'plan_sha256': base.canonical(plan), 'disk': disk}, ensure_ascii=False)); return
    if not args.run_approved or not args.plan_sha256: raise RuntimeError('Reviewed plan approval required')
    parent = Path('/proc') / str(os.getppid()); command = (parent / 'cmdline').read_bytes().split(b'\0')
    if (parent / 'exe').resolve() != Path('/usr/bin/timeout').resolve() or any(x not in command for x in (b'--signal=TERM', b'--kill-after=30s', b'600s')): raise RuntimeError('Require outer timeout TERM600/KILL30')
    clock, locks, owned = base.Clock(), [], False
    def check(): clock.check(); budget.check()
    ledger = base.Ledger(check, COUNTS); result = error = None; cleanup = {}
    try:
        import fcntl
        clock.install()
        for name in base.LOCK_NAMES:
            path = ROOT / 'exp/locks' / name
            if path.is_symlink(): raise RuntimeError('Unsafe lock')
            handle = path.open('a+')
            try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException: handle.close(); raise
            locks.append(handle)
        plan = prepare(args, base, check)
        ledger.limits = plan['forward_limits']
        if base.canonical(plan) != args.plan_sha256: raise RuntimeError('Preflight plan differs')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(): raise RuntimeError('GPU occupied')
        free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        if len(free) != 1 or int(free[0]) < 32768: raise RuntimeError('Require one GPU with32GiB free')
        check(); OUTPUT.mkdir(); owned = True; budget.write('run_contract.json', plan)
        result, cleanup = compute(args, plan, base, budget, check, ledger)
        for path, sha in plan['files'].items():
            check()
            if digest(base.regular(path)) != sha: raise RuntimeError('Runtime asset/source changed: ' + path)
        check()
    except BaseException as caught:
        # 保留错误类型/内容，但释放失败前向的traceback帧，避免它继续引用整张GPU图。
        error = RuntimeError(type(caught).__name__ + ': ' + str(caught)); caught.__traceback__ = None
    finally:
        try:
            gc.collect()
            if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized(): sys.modules['torch'].cuda.empty_cache()
        except BaseException as caught:
            if error is None: error = RuntimeError('Final memory cleanup failed: ' + repr(caught))
        error = finish(base, budget, clock, ledger, locks, owned, result, cleanup, error)
    if error is not None: raise error


if __name__ == '__main__': main()
