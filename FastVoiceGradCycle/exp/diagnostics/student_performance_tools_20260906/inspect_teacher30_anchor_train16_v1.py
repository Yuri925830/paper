"""16对训练语音的只读教师端点检查；不训练、不保存模型、不自动进入下一阶段。"""

import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
OUTPUT = DEV / 'teacher30_anchor_train16_v1'
PARENT_RUN = DEV / 'acdd_mc2_independent360_from3927_v1'
STUDENT = PARENT_RUN / 'candidate_mc2/final_full.pt'
TEACHER = ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt'
VOCODER = ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen'
TRAIN_PROOF = DEV / 'acdd_adam_descent_readonly_3107_v1/results.json'
PROTOCOL = DEV / 'TEACHER30_ANCHOR_TRAIN16_DIAGNOSTIC_CN.md'
TEST_NAME = 'test_inspect_teacher30_anchor_train16_v1.py'
HEAD = '0a2c646922704e746642f5f33c203869fe696b3f'
PARENT_CANONICAL = '45213c2fc1b760d08780e1d209c8c9d589eb232d0bc62fdf588c7c1bccb3ad6d'
STUDENT_SHA = '5f680fccd37855c0ed58d6655df9dfd0a3ea11410fd629a32020e11eb8bddfdb'
TEACHER_SHA = '1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6'
BASELINE_SHA = '14142b4909583f9a79424e9a23c188a00605c8cbfd9797daa7eff04848849d21'
EVALUATOR_SHA = '7e3b5676fd57e4ffa83bd9d705edcc690979af1ce55688b59bd69a39538797d8'
METRIC_ID_FILE_SHA = 'c0ff0e76a25b5656f1b2c110a2a6664342b03d8cf244971a8590074c75a37d86'
METRIC_ID_CANONICAL = '9490fd42770b151776d2d0c60a799135dd559bb133a67c25b726749b8e4f8427'
METRIC_ID_SOURCE = DEV / 'evaluation_mc2_independent360_endpoint_v1/metric_identity.json'
LABELS = ('teacher500_formal_k30_train16', 'student4287_k1_train16')
COUNTS = {'teacher': 600, 'student': 16, 'vocoder': 32}
WAV_LIMIT, META_LIMIT = 32 * 1024**2, 8 * 1024**2
# 本轮只读，输出硬上限40MiB且不保存模型；另保留2GiB应对同盘空间波动。
RESERVE, DIR_RESERVE = 2 * 1024**3, 1024**2
SECONDS, CLEANUP_SECONDS = 600, 30
SEED_BASE = 87007000
LOCK_NAMES = ('text_unseen_custom_hifigan_pipeline.lock', 'text_unseen_custom_hifigan_train.lock',
              'text_unseen_fastvoicegrad_student.lock', 'text_unseen_fastvoicegrad_student_inference.lock')
CORE = {
    'fastvoicegrad/train.py': 'd38131843b0b1ba51a10b65bc8bea226c2cecd1ec339d2bb04db5ede2febbaec',
    'fastvoicegrad/objectives.py': '5b1439550205ada6b4f6fca52010963df23f65f28bb185684a912f2c1e9bf58f',
    'fastvoicegrad/components.py': 'aa15c81b2c0b1e614db997df52ca97c2fa494434a0f10b84d317fe2669583b5f',
    'fastvoicegrad/config.py': 'c4b161f89839b22dcd6711f39b29231a4dbd194b6021990e2dff23967eebc81a',
    'voicegrad/model.py': '5a533381bdfebf1b72f3cc284de24c8ffbd97d7f10d74bbae4b89e7013a6d771',
    'voicegrad/dataset.py': '4575107114cbd0fdb03faa019440983bf55e31b12bc0626b0293db5b71c0d68c',
    'voicegrad/diffusion.py': 'c89d7bcee03c5dfbdf967eadd436941da7c28b2c2f9c3db5c482ea9dc3c8bf57',
    'voicegrad/feature_alignment.py': '9bedea5455a118281337ebf72fe50024098b9b3161b32d0cc9efc6ced916109f',
    'voicegrad/infer_voicegrad30_paper_eval.py': 'd885f9d4dec7fa35d4c3ae7fb1e102fea0373a4dd57d734ebb099c005518a9af',
    'evaluation/evaluate_metrics.py': '9797b724c7296517a1e0b31a74dbe05e6f06454669902b351e42caea2cba42ab',
}
FIXED = {
    str(STUDENT): STUDENT_SHA, str(TEACHER): TEACHER_SHA,
    str(PARENT_RUN / 'run_contract.json'): 'acd379068d66c51a01dcd62b1e9d91ac1e0434f3b28a54e2e169ef90cadc0263',
    str(PARENT_RUN / 'candidate_mc2/completion.json'): '331d6911e6763b4d16b40e16f7009103ea544fb6e7e2c725f5dedd2b6bb35506',
    str(TRAIN_PROOF): '61f5100aa2c8278bcfc023235f5d7962a8bc7704e2dbaadb71dea85f5c4d36f9',
    str(DEV / 'development_contract.json'): '08737551d34e00ba0b80a5e34adbf73fcfc6c70ece6bc8dc4a6c466e9b87afb1',
    str(ROOT / 'data/metadata/train_manifest.csv'): '6b24abba2bbea1102ebd002980ec9e4ded9357a07e3871d2514c2474c6774e2b',
    str(ROOT / 'data/metadata/test_manifest.csv'): 'dec09943585580c782634900956ef80f2e9e8286ae8db539eebf0a2b2e7471a4',
    str(ROOT / 'data/stats/mel_mean.npy'): '4a858ffc0ea5fdba37751a5925dd6213a8cd87ece14978cda39606ca49f2cc31',
    str(ROOT / 'data/stats/mel_std.npy'): '9e5d30a98a2fa2d65f63ae0d853bd404b716af5d90f519eb0ad06209c7a0773b',
    str(VOCODER / 'generator_step_2500000.pt'): 'dacd362c4db46cdf7bd8428ace6f38d7d03976c60a3cd4c55022d38383094fb9',
    str(VOCODER / 'config.json'): '4b570fee140ffbda92bb217218792dca069b2d73627bc5e7bf7b1669869dd05a',
    str(VOCODER / 'final_checkpoint_attestation_comments_20260906.json'): '2112690afbe6fc2f7a5325d106c8dcfdd4503d7cfb09fee7675712d34b46205e',
    str(TOOLS / 'evaluate_development_baselines.py'): BASELINE_SHA,
    str(TOOLS / 'evaluate_development_candidates.py'): EVALUATOR_SHA,
    str(METRIC_ID_SOURCE): METRIC_ID_FILE_SHA,
}


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or ROOT not in path.resolve().parents:
        raise RuntimeError('Require a regular authoritative project file: ' + str(path))
    return path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''): result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle: return list(csv.DictReader(handle))


def npy_shape(path):
    # 预检只读取NPY头；不导入NumPy/Torch、不读取数组用于任何特征计算。
    with Path(path).open('rb') as handle:
        if handle.read(6) != b'\x93NUMPY': raise RuntimeError('Invalid NPY signature')
        version = tuple(handle.read(2))
        if version not in ((1, 0), (2, 0), (3, 0)): raise RuntimeError('Unsupported NPY version')
        width = 2 if version == (1, 0) else 4
        length = struct.unpack('<H' if width == 2 else '<I', handle.read(width))[0]
        if not 0 < length <= 65536: raise RuntimeError('Unreasonable NPY header')
        header = ast.literal_eval(handle.read(length).decode('utf-8' if version == (3, 0) else 'latin1'))
    shape = header.get('shape')
    if (not isinstance(shape, tuple) or not shape or any(type(n) is not int or n <= 0 for n in shape)
            or header.get('descr') not in ('<f4', '<f8', '=f4', '=f8')):
        raise RuntimeError('Require a nonempty floating-point feature array')
    return shape


def frame_count(shape, channels):
    if len(shape) != 2: raise RuntimeError('Feature must have two axes')
    if shape[0] == channels: return shape[1]
    if shape[1] == channels: return shape[0]
    raise RuntimeError('Wrong feature channel count')


def select_pairs(manifest, proof_indices, development, test_rows):
    # 固定旧TRAIN32的顺序；只按文本是否已提供筛选，不读取任何历史分数。
    retained, excluded = set(development['retained_train_indices']), set(development['excluded_train_indices'])
    if len(excluded) != 52 or retained & excluded or retained | excluded != set(range(len(manifest))):
        raise RuntimeError('The original52 exclusions must remain complete')
    if len(proof_indices) != 32 or len(set(proof_indices)) != 32 or not set(proof_indices) <= retained:
        raise RuntimeError('Fixed TRAIN32 must be distinct and exclude all52')
    test_ids = {(r['speaker'], r['utt_id']) for r in test_rows}
    test_speakers = {r['speaker'] for r in test_rows}
    for index in proof_indices:
        row = manifest[index]
        if (row['split'] != 'train' or row['speaker'] in test_speakers
                or (row['speaker'], row['utt_id']) in test_ids): raise RuntimeError('Formal-test overlap')
    sources = [i for i in proof_indices if manifest[i]['text'].strip()][:16]
    if len(sources) != 16 or len({manifest[i]['utt_id'] for i in sources}) != 16:
        raise RuntimeError('Need16 distinct complete training utterances with supplied text')
    result = []
    for pair_id, source_index in enumerate(sources):
        position = proof_indices.index(source_index)
        choices = proof_indices[position + 1:] + proof_indices[:position]
        reference_index = next((i for i in choices if manifest[i]['speaker'] != manifest[source_index]['speaker']), None)
        if reference_index is None: raise RuntimeError('No other-speaker training reference')
        result.append({'pair_id': pair_id, 'source_index': source_index, 'reference_index': reference_index,
            'seed': SEED_BASE + pair_id, 'source': manifest[source_index], 'reference': manifest[reference_index]})
    return result


def model_specs():
    return {
        LABELS[0]: {'checkpoint': str(TEACHER), 'sha256': TEACHER_SHA, 'state_key': 'model',
            'epoch': 500, 'global_step': 615000, 'steps': 30, 'sampling': 'formal_sparse_DDPM_diffuse_source_true'},
        LABELS[1]: {'checkpoint': str(STUDENT), 'sha256': STUDENT_SHA, 'state_key': 'student',
            'epoch': 0, 'global_step': 4287, 'steps': 1, 'sampling': 'existing_student_t950_to_zero'},
    }


def validate_parent(contract, done, payload=None):
    if canonical(contract) != PARENT_CANONICAL: raise RuntimeError('Wrong4287 parent contract')
    state = done.get('state', {})
    if (done.get('status'), done.get('cleanup_verified'), state.get('completed_global_step'),
            state.get('partial_batch_may_have_updated_D'), state.get('execution_error'),
            done.get('contract_sha256')) != ('completed360', True, 4287, False, None, PARENT_CANONICAL):
        raise RuntimeError('Require the audited complete4287 endpoint')
    if payload is not None:
        if (payload.get('kind'), payload.get('global_step'), payload.get('epoch'), payload.get('resumable'),
                payload.get('run_contract_sha256'), payload.get('execution_error')) != (
                'MC2_FM2_independent360_full', 4287, 0, True, PARENT_CANONICAL, None):
            raise RuntimeError('Wrong actual4287 full payload')
        if payload.get('configuration') != contract['configuration'] or not payload.get('student'):
            raise RuntimeError('Actual student configuration or weights differ')


def prepare_plan(args, check=lambda: None):
    # 这里只核必要当前文件，不要求已私有归档的3927/3447或旧WAV仍存在。
    pins = dict(FIXED)
    pins.update({str(ROOT / p): sha for p, sha in CORE.items()})
    pins.update({str(Path(__file__).resolve()): args.driver_sha256,
        str(TOOLS / TEST_NAME): args.tests_sha256, str(PROTOCOL): args.protocol_sha256})
    for path, expected in pins.items():
        check()
        if digest(regular(path)) != expected: raise RuntimeError('Input/source SHA mismatch: ' + path)
    contract = read_json(PARENT_RUN / 'run_contract.json')
    if canonical(read_json(METRIC_ID_SOURCE)) != METRIC_ID_CANONICAL: raise RuntimeError('Original metric identity canonical differs')
    validate_parent(contract, read_json(PARENT_RUN / 'candidate_mc2/completion.json'))
    development = read_json(DEV / 'development_contract.json')
    manifest = read_csv(ROOT / 'data/metadata/train_manifest.csv')
    selected = select_pairs(manifest, read_json(TRAIN_PROOF)['train_indices'], development,
        read_csv(ROOT / 'data/metadata/test_manifest.csv'))
    rows = []
    for entry in selected:
        src, ref = entry['source'], entry['reference']
        bnf = ROOT / 'data/bnf' / src['speaker'] / (src['utt_id'] + '.npy')
        if not bnf.exists(): bnf = bnf.with_name(src['utt_id'] + '.ling_feat.npy')
        row = {k: str(entry[k]) for k in ('pair_id', 'source_index', 'reference_index', 'seed')}
        row.update(source_speaker=src['speaker'], source_utt=src['utt_id'], source_text=src['text'],
            reference_speaker=ref['speaker'], reference_utt=ref['utt_id'], source_wav=src['wav22k_path'],
            reference_wav=ref['wav22k_path'], source_mel=str(ROOT / 'data/mel' / src['speaker'] / (src['utt_id'] + '.npy')),
            source_bnf=str(bnf), reference_spk_emb=str(ROOT / 'data/spk_emb' / ref['speaker'] / (ref['utt_id'] + '.npy')),
            output_wav='')
        for key in ('source_wav', 'reference_wav', 'source_mel', 'source_bnf', 'reference_spk_emb'):
            path = regular(row[key])
            if ROOT / 'data' not in path.resolve().parents: raise RuntimeError('Feature outside data/')
            pins.setdefault(str(path), digest(path)); row[key + '_sha256'] = pins[str(path)]
        frames = frame_count(npy_shape(row['source_mel']), 80)
        frame_count(npy_shape(row['source_bnf']), 144)
        if math.prod(npy_shape(row['reference_spk_emb'])) != 256: raise RuntimeError('Wrong speaker vector')
        row['expected_num_samples'] = str(frames * 256); rows.append(row)
    # 只登记推理/评分实际相关源码包；它们作为新预执行绑定，不伪称父契约都已登记。
    for package in ('voicegrad', 'fastvoicegrad', 'hifigan', 'evaluation'):
        for path in sorted((ROOT / package).rglob('*.py')): pins.setdefault(str(regular(path)), digest(path))
    wav_upper = 2 * sum(2 * int(r['expected_num_samples']) + 4096 for r in rows)
    if wav_upper > WAV_LIMIT: raise RuntimeError('All32 whole-utterance PCM16 WAVs exceed32MiB; do not crop or reselect')
    return {'kind': 'readonly_TRAIN16_formal_teacher30_anchor', 'pairs': rows, 'systems': model_specs(),
        'inputs': pins, 'parent_contract_canonical_sha256': PARENT_CANONICAL, 'excluded_indices': development['excluded_train_indices'],
        'selection': 'first16 supplied-text sources from frozenTRAIN32; next cyclic other-speaker reference in sameTRAIN32',
        'source_binding': 'new_preexecution_binding_with_explicit_core_pins', 'sample_role': 'training_only_not_unseen_validation',
        'sampler': 'unchanged VoiceGradDiffusion.sample K30 S_K950 diffuse_source=True; never DDIM',
        'full_utterances': True, 'seed_base': SEED_BASE, 'forward_limits': COUNTS, 'wav_upper_bytes': wav_upper,
        'teacher_repeat': {'pair_ids': [0, 1, 2, 3], 'reverse_seed_offset': 1000000,
            'initial_state': 'same actual q950; sample diffuse_source=False', 'saved_WAVs': 0, 'metrics': 0},
        'wav_limit_bytes': WAV_LIMIT, 'metadata_limit_bytes': META_LIMIT, 'compute_seconds_per_phase': SECONDS,
        'cleanup_seconds': CLEANUP_SECONDS, 'optimizer_updates': 0, 'backward_calls': 0, 'checkpoint_saves': 0,
        'metrics': ['cer', 'dnsmos', 'utmos'], 'SVA_or_OVRL_gate': False,
        'limitations': ['Teacher and student previously saw training data; this only examines target usefulness.',
            'A higher-quality teacher target does not prove a student can learn it or reach formal thresholds.',
            'Earlier pure DDIM30 reconstruction L1 trial is distinct and remains a recorded quality regression.']}


class Budget:
    def __init__(self, output=OUTPUT): self.output = Path(output)
    def usage(self):
        wav = metadata = 0
        for path in self.output.rglob('*') if self.output.exists() else ():
            if path.is_symlink(): raise RuntimeError('Symlink in diagnostic output')
            if not path.is_file(): continue
            if '.pt' in path.suffixes: raise RuntimeError('No checkpoint artifacts allowed')
            size = path.stat().st_size
            if '.wav' in path.suffixes: wav += size
            else: metadata += size
        return wav, metadata
    def check(self):
        wav, metadata = self.usage()
        if wav > WAV_LIMIT or metadata > META_LIMIT: raise RuntimeError('Diagnostic output quota exceeded')
        status = os.statvfs(self.output.parent)
        free = status.f_bavail * status.f_frsize
        required = RESERVE + DIR_RESERVE + (WAV_LIMIT - wav) + (META_LIMIT - metadata)
        if free < required: raise RuntimeError('Insufficient reserved space: %d < %d' % (free, required))
        return {'free_bytes': free, 'required_bytes': required, 'wav_bytes': wav, 'metadata_bytes': metadata}
    def write(self, path, value, terminal=False):
        path = Path(path)
        if self.output not in path.parents: raise RuntimeError('Output path escapes new diagnostic')
        encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        if self.usage()[1] + len(encoded) > META_LIMIT - (0 if terminal else 262144):
            raise RuntimeError('Reserve metadata room for terminal failure receipt')
        self.check(); temporary = path.with_name(path.name + '.tmp')
        if path.exists() or temporary.exists() or path.is_symlink() or temporary.is_symlink(): raise RuntimeError('Never overwrite evidence')
        with temporary.open('xb') as handle: handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.rename(temporary, path)
        # 完成发布后不再执行可能失败的后置检查，以免留下假成功收据。
        if not terminal: self.check()


class Clock:
    def __init__(self): self.started, self.reason, self.handlers = time.monotonic(), None, {}
    def install(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            self.handlers[sig] = signal.getsignal(sig); signal.signal(sig, self.stop)
        signal.alarm(SECONDS)
    def stop(self, signum, frame): self.reason = 'signal_%s' % signum
    def check(self):
        if self.reason or time.monotonic() - self.started >= SECONDS: raise TimeoutError(self.reason or 'phase deadline')
    def close(self):
        signal.alarm(0)
        for sig, handler in self.handlers.items(): signal.signal(sig, handler)


def load_private(name, expected):
    path = TOOLS / name
    if digest(regular(path)) != expected: raise RuntimeError('Frozen helper source changed')
    spec = importlib.util.spec_from_file_location('_train16_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def tensor_id(tensor):
    value = tensor.detach().cpu().contiguous()
    return {'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest(), 'shape': list(value.shape), 'dtype': str(value.dtype)}


def snapshot(model):
    # 逐参数及buffer核对真实数值；模式和requires_grad也必须保持，不能只看文件SHA。
    if any(v.is_floating_point() and not bool(v.isfinite().all()) for v in model.state_dict().values()):
        raise RuntimeError('Model state contains nonfinite values')
    return {'state': {k: tensor_id(v) for k, v in model.state_dict().items()},
        'modes': {k: v.training for k, v in model.named_modules()},
        'requires_grad': {k: v.requires_grad for k, v in model.named_parameters()},
        'grads_none': all(v.grad is None for v in model.parameters())}


def same_rng(torch, np, actual, expected):
    if set(actual) != {'python', 'numpy', 'torch', 'cuda_all'} or set(actual) != set(expected): return False
    a, b = actual['numpy'], expected['numpy']
    return (actual['python'] == expected['python'] and a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
        and torch.equal(actual['torch'], expected['torch']) and len(actual['cuda_all']) == len(expected['cuda_all'])
        and all(torch.equal(x, y) for x, y in zip(actual['cuda_all'], expected['cuda_all'])))


class Ledger:
    def __init__(self, check, limits=None): self.check, self.limits, self.counts, self.handles = check, limits or COUNTS, {}, []
    def add(self, model, name):
        self.counts[name] = 0
        def before(module, inputs):
            self.check()
            if self.counts[name] >= self.limits[name]: raise RuntimeError('Forward budget exceeded: ' + name)
            self.counts[name] += 1
        self.handles.append(model.register_forward_pre_hook(before))
    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()


def one_pair(torch, teacher, student, vocoder, diffusion, source, speaker, bnf, seed, mean, std, predict, repeat_seed=None):
    # 两个新Generator使用同一seed；教师sample自己正常消耗第一份源噪声及后续29份反向噪声。
    # hook只读取教师第一次实际输入，不重写sample、不额外做教师前向。
    first = []
    def observe(module, values):
        if not first: first.append((tensor_id(values[0]), values[1].detach().cpu().tolist(), tensor_id(values[2]), tensor_id(values[3])))
    handle = teacher.register_forward_pre_hook(observe)
    try:
        with torch.inference_mode():
            generator = torch.Generator(device=source.device).manual_seed(int(seed))
            target = diffusion.sample(teacher, source, speaker, bnf, num_steps=30, start_step=950,
                generator=generator, diffuse_source=True)
            prediction, q = predict(student, diffusion, source, speaker, bnf, seed, 'student_k1')
            expected = (tensor_id(q), [950] * source.shape[0], tensor_id(speaker), tensor_id(bnf))
            if first != [expected]: raise RuntimeError('Teacher/student actual initial state or conditioning mismatch')
            if target.shape != prediction.shape or not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(prediction).all()):
                raise RuntimeError('Nonfinite or wrong-shaped independent endpoint')
            record = {'initial_q950': first[0][0], 'source': tensor_id(source), 'bnf': tensor_id(bnf),
                'target_speaker': tensor_id(speaker), 'teacher_endpoint': tensor_id(target), 'student_endpoint': tensor_id(prediction),
                'student_teacher_mel_l1': float((prediction - target).abs().mean()),
                'teacher_rms': float(target.square().mean().sqrt()), 'student_rms': float(prediction.square().mean().sqrt())}
            if repeat_seed is not None:
                # 输入已是实际q950，关闭再次源加噪；后续30步仍完整使用原sample数学。
                repeat = diffusion.sample(teacher, q, speaker, bnf, num_steps=30, start_step=950,
                    generator=torch.Generator(device=source.device).manual_seed(int(repeat_seed)), diffuse_source=False)
                if repeat.shape != target.shape or not bool(torch.isfinite(repeat).all()): raise RuntimeError('Invalid repeated teacher endpoint')
                record['teacher_repeat'] = {'reverse_seed': int(repeat_seed), 'initial_q950': tensor_id(q),
                    'endpoint': tensor_id(repeat), 'mel_l1_to_primary_teacher': float((repeat - target).abs().mean()),
                    'rms': float(repeat.square().mean().sqrt()), 'primary_teacher_replaced': False,
                    'waveform_generated': False, 'metric_computed': False}
            waves = [vocoder(mel * std + mean).squeeze().detach().cpu().numpy() for mel in (target, prediction)]
            return waves, record
    finally: handle.remove()


def write_rows(path, rows):
    # 两系统的CSV除output_wav外完全相同；后续旧评分器用同一source_text计算CER。
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def infer(plan, budget, check, ledger):
    import numpy as np
    import soundfile as sf
    import torch
    from fastvoicegrad import train as training
    from fastvoicegrad.components import load_teacher_and_student, load_torch_checkpoint, load_frozen_vocoder, load_mel_statistics
    from fastvoicegrad.config import PAPER_STUDENT_CONFIG
    from hifigan.tools.vocode import atomic_write_wav
    from voicegrad.diffusion import VoiceGradDiffusion
    from voicegrad.feature_alignment import align_bnf_to_mel
    from voicegrad.infer_voicegrad30_paper_eval import feature_matrix
    evaluator = load_private('evaluate_development_candidates.py', EVALUATOR_SHA)
    if (torch.__version__, torch.version.cuda) != ('1.13.1+cu116', '11.6'): raise RuntimeError('Use frozen myconda inference environment')
    flags = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'benchmark': torch.backends.cudnn.benchmark, 'deterministic': torch.backends.cudnn.deterministic}
    if flags != {'matmul_tf32': False, 'cudnn_tf32': True, 'benchmark': False, 'deterministic': False}:
        raise RuntimeError('Original inference precision changed')
    old_threads = torch.get_num_threads(); torch.set_num_threads(4)
    models, before, rng, records = {}, {}, None, []
    try:
        device = torch.device('cuda'); check()
        teacher, student, ignored = load_teacher_and_student(TEACHER, PAPER_STUDENT_CONFIG, device); del ignored
        payload = load_torch_checkpoint(STUDENT); contract = read_json(PARENT_RUN / 'run_contract.json')
        validate_parent(contract, read_json(PARENT_RUN / 'candidate_mc2/completion.json'), payload)
        evaluator.validate_payload(payload, model_specs()[LABELS[1]], contract)
        student.load_state_dict(payload['student'], strict=True); del payload; gc.collect()
        vocoder, ignored, ignored2 = load_frozen_vocoder(ROOT, VOCODER / 'generator_step_2500000.pt',
            VOCODER / 'config.json', VOCODER / 'final_checkpoint_attestation_comments_20260906.json', PAPER_STUDENT_CONFIG, device)
        del ignored, ignored2
        diffusion = VoiceGradDiffusion(1000, offset=0.008).to(device)
        models = {'teacher': teacher, 'student': student, 'vocoder': vocoder, 'diffusion': diffusion}
        for name, model in models.items():
            model.eval().requires_grad_(False)
            if any(p.grad is not None for p in model.parameters()): raise RuntimeError('Unexpected existing gradient')
            before[name] = snapshot(model)
            if name in COUNTS: ledger.add(model, name)
        rng = training.capture_rng_state()
        mean, std = load_mel_statistics(ROOT / 'data', device)
        means = np.load(ROOT / 'data/stats/mel_mean.npy', allow_pickle=False).astype(np.float32).reshape(80, 1)
        stds = np.load(ROOT / 'data/stats/mel_std.npy', allow_pickle=False).astype(np.float32).reshape(80, 1)
        system_rows, hashes = {n: [] for n in LABELS}, {n: {} for n in LABELS}
        for label in LABELS: (OUTPUT / label / 'wavs').mkdir(parents=True)
        for row in plan['pairs']:
            check()
            mel, bnf = feature_matrix(Path(row['source_mel']), 80), feature_matrix(Path(row['source_bnf']), 144)
            source = torch.from_numpy(((mel - means) / stds).astype(np.float32)).unsqueeze(0).to(device)
            condition = align_bnf_to_mel(torch.from_numpy(bnf).unsqueeze(0).to(device), source.shape[-1],
                mode='legacy_length_linear', sample_rate=22050, n_fft=1024, hop_size=256, bnf_frame_shift_seconds=.010)
            vector = np.load(row['reference_spk_emb'], allow_pickle=False).squeeze()
            if vector.shape != (256,) or not np.isfinite(vector).all(): raise RuntimeError('Invalid reference speaker vector')
            speaker = torch.from_numpy(vector.astype(np.float32)).unsqueeze(0).to(device)
            waves, record = one_pair(torch, teacher, student, vocoder, diffusion, source, speaker, condition,
                int(row['seed']), mean, std, evaluator.predict_candidate,
                repeat_seed=int(row['seed']) + 1000000 if int(row['pair_id']) < 4 else None)
            for label, wave in zip(LABELS, waves):
                check()
                if wave.ndim != 1 or len(wave) != int(row['expected_num_samples']) or not np.isfinite(wave).all():
                    raise RuntimeError('Invalid whole-utterance waveform')
                path = OUTPUT / label / 'wavs' / ('pair_%04d.wav' % int(row['pair_id']))
                atomic_write_wav(path, np.clip(wave.astype(np.float32), -1, 1), 22050)
                info = sf.info(path)
                if (info.samplerate, info.channels, info.frames, info.subtype) != (22050, 1, len(wave), 'PCM_16'):
                    raise RuntimeError('WAV format differs from frozen evaluation')
                hashes[label][row['pair_id']] = digest(path); system_rows[label].append(dict(row, output_wav=str(path)))
            records.append(dict(record, pair_id=row['pair_id'])); budget.check()
            print(json.dumps({'pair_completed': row['pair_id'], 'forward_counts': ledger.counts}), flush=True)
        if ledger.counts != COUNTS: raise RuntimeError('Wrong final inference call counts')
        for label in LABELS:
            path = OUTPUT / label / 'pairs.csv'; write_rows(path, system_rows[label])
            budget.write(OUTPUT / label / 'inference_complete.json', {'system': label, 'spec': model_specs()[label],
                'run_identity_sha256': digest(OUTPUT / 'run_identity.json'), 'pairs_csv_sha256': digest(path),
                'wav_sha256': hashes[label], 'n_pairs': 16, 'diagnostic_only': True})
        budget.write(OUTPUT / 'endpoint_readouts.json', records)
        return {'counts': dict(ledger.counts), 'runtime': {'flags': flags, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'threads': torch.get_num_threads(), 'previous_threads': old_threads},
            'waves': 32, 'optimizer_updates': 0, 'backward_calls': 0, 'checkpoint_saves': 0}
    finally:
        errors = []
        try:
            if rng is not None:
                # 全部噪声使用私有Generator；严格检查默认RNG没有被采样改变。
                current = training.capture_rng_state()
                if not same_rng(torch, np, current, rng): errors.append('global_rng_changed')
                training.restore_rng_state(rng, torch.device('cuda'))
            for name, model in models.items():
                if name in before and snapshot(model) != before[name]: errors.append('model_changed:' + name)
        finally:
            ledger.close(); models.clear(); gc.collect(); torch.cuda.empty_cache(); torch.set_num_threads(old_threads)
        if errors: raise RuntimeError('Read-only preservation failure: ' + repr(errors))


def score_phase(args, plan, budget, check):
    # 保持既有评分模型/算法；新16对由本文件验证，不能伪造旧90对的选择收据。
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    shared = load_private('evaluate_development_baselines.py', BASELINE_SHA)
    shared.SYSTEMS = model_specs()  # 只改变私有实例的本次标签，不改冻结源码或公共模块。
    identity = shared.metric_identity(); identity.update(device='cuda', batch_size=4, dnsmos_workers=1, dnsmos_onnx_threads=3)
    if canonical(identity) != METRIC_ID_CANONICAL or identity != read_json(METRIC_ID_SOURCE):
        raise RuntimeError('Frozen objective evaluator identity differs')
    identity_path = OUTPUT / 'metric_identity.json'
    if identity_path.exists():
        if read_json(identity_path) != identity: raise RuntimeError('Metric identity changed between phases')
    else: budget.write(identity_path, identity)
    import torch
    def flags():
        return {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'benchmark': torch.backends.cudnn.benchmark, 'deterministic': torch.backends.cudnn.deterministic}
    before_flags = flags()
    original = {'matmul_tf32': False, 'cudnn_tf32': True, 'benchmark': False, 'deterministic': False}
    if before_flags != original: raise RuntimeError('Require a fresh metric process with original flags')
    if args.phase == 'quality':
        # 原全指标流程先导入SpeechBrain；保留其精度副作用，但不构建SVA模型、不算SVA。
        from speechbrain.inference.speaker import SpeakerRecognition
    after_flags = flags()
    if after_flags != dict(original, matmul_tf32=args.phase == 'quality'):
        raise RuntimeError('Original SpeechBrain import precision side effects differ')
    from evaluation import evaluate_metrics as metrics
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    names = ['cer'] if args.phase == 'cer' else ['dnsmos', 'utmos']
    fields = {'cer': metrics.CER_FIELDS, 'dnsmos': metrics.DNSMOS_FIELDS, 'utmos': metrics.UTMOS_FIELDS}
    result = {}
    try:
        for label in LABELS:
            check(); rows = shared.verify_wavs(OUTPUT / label, plan['pairs']); folder = OUTPUT / label / 'metrics'; folder.mkdir(exist_ok=True)
            for name in names:
                destination = folder / (name + '.csv')
                if destination.exists(): raise RuntimeError('Existing metric CSV forbids retry')
                check()
                if name == 'cer': metrics.evaluate_cer(rows, destination, torch.device('cuda'), 4)
                elif name == 'dnsmos': metrics.evaluate_dnsmos(rows, destination, 4, 1)
                else: metrics.evaluate_utmos(rows, destination, ROOT / 'UTMOS-demo', torch.device('cuda'), 4)
                check(); metrics.read_metric_rows(destination, rows, name, fields[name])
            result[label] = {name: digest(folder / (name + '.csv')) for name in names}
        after = shared.metric_identity(); after.update(device='cuda', batch_size=4, dnsmos_workers=1, dnsmos_onnx_threads=3)
        if identity != after: raise RuntimeError('Metric assets changed during evaluation')
        if flags() != after_flags: raise RuntimeError('Metric precision changed during scoring')
        return {'metric_identity_canonical_sha256': canonical(identity), 'original_metric_identity_file_sha256': METRIC_ID_FILE_SHA,
            'actual_metric_csv_sha256': result, 'SVA_executed': False,
            'precision_before_SpeechBrain_import': before_flags, 'precision_after_SpeechBrain_import': after_flags}
    finally:
        gc.collect(); torch.cuda.empty_cache(); torch.set_num_threads(old_threads)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'infer', 'cer', 'quality'), required=True)
    parser.add_argument('--run-approved', action='store_true')
    for name in ('driver', 'tests', 'protocol'): parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--plan-sha256')
    args = parser.parse_args(argv)
    if args.phase != 'preflight' and (not args.run_approved or not args.plan_sha256): raise RuntimeError('Reviewed plan and bounded run approval required')
    for value in (args.driver_sha256, args.tests_sha256, args.protocol_sha256, args.plan_sha256):
        if value is not None and (len(value) != 64 or any(c not in '0123456789abcdef' for c in value)): raise RuntimeError('Require full lowercase SHA256')
    return args


def finish_phase(args, budget, clock, ledger, locks, owned, result, error):
    # 模型/钩子清理后仍持有四锁，发布本阶段收据，最后才释放锁。
    # 若释放失败，撤下的只是本阶段新完成标记；原记录改名保留，绝不覆盖旧证据。
    cleanup = []
    for action in (ledger.close, clock.close):
        try: action()
        except BaseException as caught: cleanup.append(repr(caught))
    completion = OUTPUT / (args.phase + '_completion.json')
    failure = OUTPUT / (args.phase + '_failure.json')
    published = False
    def fail_record(cause):
        if owned and not failure.exists():
            budget.write(failure, {'phase': args.phase, 'error': repr(cause), 'cleanup_errors': list(cleanup),
                'actual_forward_counts': dict(ledger.counts), 'no_retry': True}, terminal=True)
    if owned:
        try:
            if error is not None or cleanup:
                if error is None: error = RuntimeError('Cleanup failed: ' + repr(cleanup))
                fail_record(error)
            else:
                clock.check()
                budget.write(completion, {'phase': args.phase, 'cleanup_verified': True,
                    'plan_sha256': args.plan_sha256, 'result': result, 'elapsed_seconds': time.monotonic() - clock.started,
                    'automatic_next_stage': False}, terminal=True)
                published = True
        except BaseException as caught:
            if error is None: error = caught
            try: fail_record(caught)
            except BaseException as final_error: print('Could not publish failure receipt: ' + repr(final_error), file=sys.stderr)
    lock_errors = []
    for handle in reversed(locks):
        try: handle.close()
        except BaseException as caught: lock_errors.append(repr(caught))
    if lock_errors:
        cleanup.extend(lock_errors)
        if error is None: error = RuntimeError('Lock release failed: ' + repr(lock_errors))
        try:
            if published:
                invalidated = OUTPUT / (args.phase + '_completion_invalidated.json')
                if invalidated.exists() or invalidated.is_symlink(): raise RuntimeError('Do not overwrite invalidation evidence')
                os.rename(completion, invalidated)
            fail_record(error)
        except BaseException as caught: print('Could not preserve terminal invalidation: ' + repr(caught), file=sys.stderr)
    if cleanup and error is None: error = RuntimeError('Cleanup failed: ' + repr(cleanup))
    return error


def main(argv=None):
    args = arguments(argv)
    if not sys.platform.startswith('linux') or Path.cwd().resolve() != ROOT or not (ROOT / 'AGENTS.md').is_file():
        raise RuntimeError('Run only inside the registered Matpool rental project')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != HEAD:
        raise RuntimeError('Frozen project HEAD changed')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip(): raise RuntimeError('Tracked code is dirty')
    sys.path.insert(0, str(ROOT))
    if args.phase == 'preflight':
        if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('Diagnostic output must still be absent')
        plan = prepare_plan(args); disk = Budget().check()
        if 'torch' in sys.modules: raise RuntimeError('CPU preflight may not import Torch or load a model')
        print(json.dumps({'status': 'preflight_only', 'plan_sha256': canonical(plan), 'plan': plan, 'disk': disk, 'torch_imported': False}, ensure_ascii=False))
        return
    clock, budget, locks, owned = Clock(), Budget(), [], False
    ledger = Ledger(lambda: (clock.check(), budget.check())); result = error = None
    try:
        import fcntl
        clock.install()
        for name in LOCK_NAMES:
            path = ROOT / 'exp/locks' / name
            if path.is_symlink(): raise RuntimeError('Unsafe lock path')
            handle = path.open('a+')
            try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException: handle.close(); raise
            locks.append(handle)
        def check(): clock.check(); budget.check()
        plan = prepare_plan(args, check)
        if canonical(plan) != args.plan_sha256: raise RuntimeError('Preflight plan changed')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU is occupied; never join another workload')
        if args.phase == 'infer':
            if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('No inference retry or overwrite')
            OUTPUT.mkdir(); owned = True
            budget.write(OUTPUT / 'run_identity.json', plan)
        else:
            if OUTPUT.is_symlink() or read_json(OUTPUT / 'run_identity.json') != plan: raise RuntimeError('Wrong diagnostic output identity')
            if read_json(OUTPUT / 'infer_completion.json').get('cleanup_verified') is not True: raise RuntimeError('Inference is incomplete')
            for suffix in ('started', 'failure', 'completion'):
                if (OUTPUT / (args.phase + '_' + suffix + '.json')).exists(): raise RuntimeError('No metric phase retry')
            owned = True
        budget.write(OUTPUT / (args.phase + '_started.json'), {'phase': args.phase, 'plan_sha256': args.plan_sha256})
        result = infer(plan, budget, check, ledger) if args.phase == 'infer' else score_phase(args, plan, budget, check)
        for path, sha in plan['inputs'].items():
            check()
            if digest(regular(path)) != sha: raise RuntimeError('Input changed during phase: ' + path)
        check()
    except BaseException as caught: error = caught
    finally:
        error = finish_phase(args, budget, clock, ledger, locks, owned, result, error)
    if error is not None: raise error


if __name__ == '__main__': main()
