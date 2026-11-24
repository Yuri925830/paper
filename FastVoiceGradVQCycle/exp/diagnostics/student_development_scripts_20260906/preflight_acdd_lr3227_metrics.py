"""仅在租用CPU核对三份真实权重、训练来源和90对评估预算；不启动任何GPU工作。"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
EXPECTED = {'parent': ('a9128f1acc9adee16c18f27e5e0361fac95398b3281e5a71567eaaa2359980c0', 3107),
            'matched_control': ('4457563a142e39e296994bb8c169314c5618a95d1fc24d28c1c02ce8f55449af', 3227),
            'primary': ('88da1f14234c9dcb164cdd249d47862b9ba105f1b7a1a858b4a18f9fa921b1dc', 3227)}
HELPERS = {'evaluate_development_candidates.py': '7e3b5676fd57e4ffa83bd9d705edcc690979af1ce55688b59bd69a39538797d8',
           'evaluate_development_baselines.py': '14142b4909583f9a79424e9a23c188a00605c8cbfd9797daa7eff04848849d21',
           'ddim_teacher.py': '0e3100d14e421a6f7eb1a06b1dc79d195091c2f20a35f4c3e4266e77dbd3c6cc'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux') or Path.cwd() != ROOT or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Rental CPU only; explicitly hide CUDA')
    if args.selection.resolve().parent != DEV or args.receipt.resolve().parent != DEV or args.receipt.exists():
        raise ValueError('Read registered selection and write a new direct DEV receipt')
    for name, expected in HELPERS.items():
        if sha(TOOLS / name) != expected:
            raise ValueError('Frozen helper changed: ' + name)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TOOLS))
    import torch
    import evaluate_development_candidates as candidate
    from fastvoicegrad.train import canonical_json_sha256
    torch.set_num_threads(1)
    selected = json.loads(args.selection.read_text())
    if selected['schema_version'] != 1:
        raise ValueError('Unexpected selection schema')
    shared = candidate.load_verified_module(TOOLS / 'evaluate_development_baselines.py', candidate.BASELINE_SHA, '_lr3227_preflight_baseline')
    paired_root = DEV / 'acdd_lr_paired120_from3107_v1'
    paired_end = json.loads((paired_root / 'completion.json').read_text())
    if paired_end['status'] != 'paired_short_experiment_finished':
        raise ValueError('Both short trials must have actually ended')
    development = json.loads((DEV / 'development_contract.json').read_text())
    verified_files, models = {}, {}
    for role in ('parent', 'matched_control', 'primary'):
        item = selected[role]
        checkpoint, contract_path, destination = Path(item['checkpoint']), Path(item['run_contract']), Path(item['candidate_root'])
        if destination.parent != DEV or destination.exists() or (item['checkpoint_sha256'], item['global_step']) != EXPECTED[role] or item['epoch'] != 0:
            raise ValueError('Wrong candidate, endpoint or occupied output namespace: ' + role)
        if sha(checkpoint) != item['checkpoint_sha256']:
            raise ValueError('Actual checkpoint SHA mismatch')
        contract = json.loads(contract_path.read_text())
        if canonical_json_sha256(contract) != item['training_contract_sha256']:
            raise ValueError('Actual training contract mismatch')
        if contract['training_indices'] != development['retained_train_indices'] or contract['excluded_train_indices'] != development['excluded_train_indices']:
            raise ValueError('Original52-exclusion training pool changed')
        for path, expected in contract['file_sha256'].items():
            if path in verified_files:
                if verified_files[path] != expected:
                    raise ValueError('Training contracts disagree on a shared dependency')
            else:
                if sha(path) != expected:
                    raise ValueError('Training source/dependency changed: ' + path)
                verified_files[path] = expected
        namespace = argparse.Namespace(system='student_k1', checkpoint=checkpoint, checkpoint_sha256=item['checkpoint_sha256'],
            run_contract=contract_path, label=item['label'], epoch=0, global_step=item['global_step'])
        label, spec = candidate.candidate_spec(namespace)
        payload = torch.load(checkpoint, map_location='cpu')
        if candidate.validate_payload(payload, spec, contract) != item['training_contract_sha256'] or payload['configuration'] != contract['configuration']:
            raise ValueError('Frozen evaluator rejected actual checkpoint metadata')
        if role != 'parent':
            arm = 'control_s2e4' if role == 'matched_control' else 'primary_s1e4'
            actual_end = paired_end['arms'][arm]
            arm_end = json.loads((paired_root / arm / 'arm_completion.json').read_text())
            expected_lrs = {'student': .0002 if role == 'matched_control' else .0001, 'discriminator': .0002}
            if actual_end != arm_end or actual_end['completed_global_step'] != 3227 or actual_end['updates'] != 120:
                raise ValueError('Paired endpoint receipt mismatch')
            receipt = actual_end['student_snapshot']
            if receipt['sha256'] != item['checkpoint_sha256'] or Path(receipt['path']) != checkpoint or not receipt['committed']:
                raise ValueError('Selected snapshot differs from the actual saved endpoint')
            if payload['arm'] != arm or payload['effective_optimizer_lrs'] != expected_lrs or contract['effective_optimizer_lrs'][arm] != expected_lrs or actual_end['effective_optimizer_lrs'] != expected_lrs:
                raise ValueError('Actual role-specific learning rates changed')
        else:
            parent_end = json.loads((contract_path.parent / 'completion.json').read_text())
            if parent_end['status'] != 'requested_stop' or parent_end['global_step'] != 3107:
                raise ValueError('Expected the original user-requested stop at3107')
            expected_lrs = {'student': .0002, 'discriminator': .0002}
        del payload
        shared.SYSTEMS = {label: spec}
        rows, inputs = shared.validate_inputs(argparse.Namespace(pairs=DEV / 'pairs.csv', development_contract=DEV / 'development_contract.json'))
        if len(rows) != 90 or len({row['source_speaker'] for row in rows}) != 10:
            raise ValueError('Use all original90 development pairs')
        wav_bytes = sum(int(row['expected_num_samples']) * 2 + 44 for row in rows)
        if wav_bytes != 14749560:
            raise ValueError('Expected PCM16 waveform storage changed')
        models[role] = {'spec': spec, 'run_contract': str(contract_path), 'run_contract_file_sha256': sha(contract_path),
            'training_contract_sha256': item['training_contract_sha256'], 'effective_optimizer_lrs': expected_lrs,
            'pairs': len(rows), 'expected_wav_bytes': wav_bytes, 'validated_input_file_count': len(inputs)}
    # 三份输出按已有实测15.3MB向上预留32MiB/份，并额外保留128MiB阶段临时空间。
    required = 3 * 1024**3 + 128 * 1024**2 + 3 * 32 * 1024**2
    fs = os.statvfs(DEV)
    free = fs.f_bavail * fs.f_frsize
    if free < required:
        raise RuntimeError('Insufficient space for allthree outputs plus3GiB system reserve')
    record = {'status': 'cpu_preflight_passed_no_gpu_started', 'selection_file_sha256': sha(args.selection),
        'source_file_sha256': sha(Path(__file__)), 'frozen_helpers': HELPERS, 'models': models,
        'verified_training_dependency_sha256': verified_files, 'space': {'free_bytes': free, 'required_bytes': required,
            'system_reserve_bytes': 3 * 1024**3, 'per_candidate_output_budget_bytes': 32 * 1024**2, 'transient_budget_bytes': 128 * 1024**2},
        'phase_order': ['parent_infer', 'parent_metrics', 'matched_control_infer', 'matched_control_metrics', 'primary_infer', 'primary_metrics'],
        'limitations': ['No speech has been synthesized or scored by this check.', 'Recheck remaining disk budget before every subsequent phase.']}
    with args.receipt.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    print(json.dumps({'status': record['status'], 'models': models, 'space': record['space']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
