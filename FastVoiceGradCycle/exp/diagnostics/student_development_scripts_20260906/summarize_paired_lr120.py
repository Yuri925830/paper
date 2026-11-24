"""读取完整短对照，核对共同步骤并画出全部曲线；不训练、不挑最好的一步。"""

import argparse  # 必须给出真实实验目录和全新结果目录。
import hashlib  # 记录读过的每一个结果文件，便于追溯。
import json  # 原始每步日志与监测日志保持原样。
import math  # 拒绝NaN与无穷大，避免坏读数进入均值或图。
import os  # 本脚本只在租用服务器CPU运行。
import sys
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
ARMS = ('control_s2e4', 'primary_s1e4')
LOSS_KEYS = ('student_total', 'student_adv', 'fm_raw', 'fm_weighted', 'dist_raw', 'dist_weighted', 'discriminator')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run project analysis only on the registered rental server')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def file_sha(path):
    # 大权重只流式读取，不把整个文件搬进内存，也不载入模型执行推理。
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-6)


def indexed(rows, name):
    # 一个实际更新数只能有一个读数；重复行不能偷偷用后写的覆盖前写的。
    result = {}
    for row in rows:
        key = row['updates']
        if not isinstance(key, int) or key in result or row['global_step'] != 3107 + key:
            raise ValueError('Duplicate/invalid update identity: ' + name)
        result[key] = row
    return result


def validate_arm(steps, windows, monitors, completion):
    count = completion['updates']
    if not isinstance(count, int) or not 0 <= count <= 120 or completion['completed_global_step'] != 3107 + count:
        raise ValueError('Completion does not match the bounded3107 continuation')
    by_step, by_window, by_monitor = indexed(steps, 'steps'), indexed(windows, 'windows'), indexed(monitors, 'monitors')
    if list(by_step) != list(range(1, count + 1)):
        raise ValueError('Missing/reordered training steps')
    if list(by_window) != list(range(20, count + 1, 20)):
        raise ValueError('Missing/noncontiguous full20-step windows')
    expected_monitor = sorted({0, count, *range(20, count + 1, 20)})
    if list(by_monitor) != expected_monitor:
        raise ValueError('Missing/reordered fixed monitor checkpoints')
    for row in steps:
        if row['actual_optimizer_lrs'] != completion['effective_optimizer_lrs']:
            raise ValueError('Runtime optimizer learning rates differ from the declared arm')
        if len(row['indices']) != 32 or len(set(row['indices'])) != 32:
            raise ValueError('Expected a full32-example training batch')
        if any(not math.isfinite(row[key]) for key in LOSS_KEYS):
            raise ValueError('Nonfinite training readout')
        if not close(row['fm_weighted'], 2 * row['fm_raw']) or not close(row['dist_weighted'], 45 * row['dist_raw']):
            raise ValueError('Original2/45 weighting changed')
        if not close(row['student_total'], row['student_adv'] + row['fm_weighted'] + row['dist_weighted']):
            raise ValueError('Total-loss identity changed')
    for update, row in by_window.items():
        if row['count'] != 20 or row['first_global_step'] != 3107 + update - 19:
            raise ValueError('Window is not the declared nonoverlapping20-step mean')
        for key in LOSS_KEYS:
            expected = math.fsum(by_step[i][key] for i in range(update - 19, update + 1)) / 20
            if not close(row['means'][key], expected):
                raise ValueError('Window differs from original steps: ' + key)
    for row in monitors:
        fixed, development = row['fixed_train'], row['development']
        if len(development['per_utterance_mel_l1']) != 32 or not close(development['mel_l1'], sum(development['per_utterance_mel_l1']) / 32):
            raise ValueError('Fixed32 development mean/sample count changed')
        for key in ('primary_moving_score', 'primary_moving_total', 'diagnostic_frozen_target_score', 'diagnostic_frozen_target_total', 'adv', 'fm_raw', 'train_gt_l1'):
            if not math.isfinite(fixed[key]):
                raise ValueError('Nonfinite fixed-reference readout')
        for score, total in (('primary_moving_score', 'primary_moving_total'), ('diagnostic_frozen_target_score', 'diagnostic_frozen_target_total')):
            if not close(fixed[total], fixed['adv'] + 2 * fixed['fm_raw'] + 45 * fixed[score]):
                raise ValueError('Fixed-reference loss decomposition changed')
    return {'steps': steps, 'windows': windows, 'monitors': monitors, 'completion': completion,
            'by_step': by_step, 'by_window': by_window, 'by_monitor': by_monitor}


def rising_triplets(rows, value, relative_start=None):
    # 原停止定义：三个读数，两次严格增加；阈值和窗口位置不能根据结果更改。
    found = []
    for i in range(2, len(rows)):
        triple = rows[i-2:i+1]
        numbers = [value(row) for row in triple]
        denominator = numbers[0] if relative_start is None else relative_start
        threshold_ratio = 1.02 if relative_start is None else 1.01
        change = (numbers[-1] - denominator) / denominator
        if denominator > 0 and numbers[0] < numbers[1] < numbers[2] and numbers[-1] / denominator >= threshold_ratio:
            found.append({'updates': [row['updates'] for row in triple], 'values': numbers, 'relative_change': change})
    return found


def compare(arms):
    control, candidate = [arms[key] for key in ARMS]
    common_steps = sorted(set(control['by_step']) & set(candidate['by_step']))
    for step in common_steps:
        if control['by_step'][step]['indices'] != candidate['by_step'][step]['indices']:
            raise ValueError('Paired arms used different training examples')
    if common_steps and any(not close(control['by_step'][1][key], candidate['by_step'][1][key]) for key in LOSS_KEYS):
        raise ValueError('First pre-student-update losses differ despite identical archived initialization')
    common_monitors = sorted(set(control['by_monitor']) & set(candidate['by_monitor']))
    rows = []
    for update in common_monitors:
        left, right = control['by_monitor'][update], candidate['by_monitor'][update]
        rows.append({'updates': update, 'global_step': update + 3107,
            'gt_l1_control': left['development']['mel_l1'], 'gt_l1_candidate': right['development']['mel_l1'],
            'gt_l1_candidate_minus_control': right['development']['mel_l1'] - left['development']['mel_l1'],
            'fixed_target_total_control': left['fixed_train']['diagnostic_frozen_target_total'],
            'fixed_target_total_candidate': right['fixed_train']['diagnostic_frozen_target_total'],
            'refreshed_target_total_control': left['fixed_train']['primary_moving_total'],
            'refreshed_target_total_candidate': right['fixed_train']['primary_moving_total']})
    common_windows = sorted(set(control['by_window']) & set(candidate['by_window']))
    window_rows = [{'updates': step, 'control': control['by_window'][step]['means'],
                    'candidate': candidate['by_window'][step]['means']} for step in common_windows]
    details = {}
    for label, arm in arms.items():
        initial, last = arm['monitors'][0], arm['monitors'][-1]
        details[label] = {'completion': arm['completion'],
            'original_moving_loss_stop_matches': rising_triplets(arm['windows'], lambda r: r['means']['student_total']),
            'development_error_stop_matches': rising_triplets(arm['monitors'], lambda r: r['development']['mel_l1'], initial['development']['mel_l1']),
            'initial_gt_l1': initial['development']['mel_l1'], 'actual_endpoint_gt_l1': last['development']['mel_l1'],
            'endpoint_gt_relative_change': last['development']['mel_l1'] / initial['development']['mel_l1'] - 1,
            'complete_window_total_values': [row['means']['student_total'] for row in arm['windows']]}
        moving_stops = details[label]['original_moving_loss_stop_matches']
        dev_stops = details[label]['development_error_stop_matches']
        triggers = moving_stops + dev_stops
        if triggers and arm['completion']['updates'] > min(item['updates'][-1] for item in triggers):
            raise ValueError('Arm continued after a predeclared trend stop should have fired')
        reason = arm['completion']['reason']
        for expected_reason, matches in (('moving_total_three_windows_rising', moving_stops), ('development_three_checks_rising', dev_stops)):
            if reason == expected_reason and not any(item['updates'][-1] == arm['completion']['updates'] for item in matches):
                raise ValueError('Reported trend-stop reason does not match the actual recorded endpoint')
    return {'status': 'complete_all_logged_steps_compared', 'arms': details, 'common_monitor_rows': rows,
        'common_window_rows': window_rows, 'latest_common_monitor_updates': common_monitors[-1],
        'limitations': ['One short deterministic paired optimization diagnostic; no claim of long-term convergence.',
            'Fixed TRAIN3108 is a training batch, not an independent validation set.',
            'Fixed32 utterances are excluded from these new updates but have already been inspected during development; they are not an untouched final test.',
            'FixedD/old-teacher-target loss and original moving training loss are different quantities and remain separate.',
            'Unequal arm endpoints are reported separately; only shared checkpoints are paired.',
            'Only endpoint weights are saved, so a shared earlier scalar checkpoint need not have a saved model for CER.',
            'No objective conversion scoring, formal8100 evaluation, best-checkpoint selection or new training occurs here.']}


def plot_all(arms, path):
    import matplotlib
    matplotlib.use('Agg')  # 服务器只生成PNG，不打开桌面窗口。
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    colors = {'control_s2e4': '#3366aa', 'primary_s1e4': '#cc6633'}
    labels = {'control_s2e4': 'Original student LR 0.0002', 'primary_s1e4': 'Student LR 0.0001'}
    for name in ARMS:
        arm, color, label = arms[name], colors[name], labels[name]
        axes[0].plot([r['updates'] for r in arm['windows']], [r['means']['student_total'] for r in arm['windows']], 'o-', color=color, label=label)
        for field, style, suffix in (('diagnostic_frozen_target_total', '-', 'fixed teacher target'), ('primary_moving_total', '--', 'refreshed teacher target')):
            axes[1].plot([r['updates'] for r in arm['monitors']], [r['fixed_train'][field] for r in arm['monitors']], style, marker='o', color=color, label=label + ', ' + suffix)
        axes[2].plot([r['updates'] for r in arm['monitors']], [r['development']['mel_l1'] for r in arm['monitors']], 'o-', color=color, label=label)
    axes[0].set_title('Original training total: all non-overlapping 20-update means')
    axes[1].set_title('Fixed TRAIN3108 and D3107: two distinct teacher-target readouts')
    axes[2].set_title('Same 32 development utterances: reconstruction error')
    for axis, label in zip(axes, ('Training loss', 'Diagnostic loss', 'Normalized mel L1')):
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
        axis.ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[2].set_xlabel('Actual updates after the same full step3107 checkpoint')
    figure.text(.5, .01, 'All checkpoints shown; curves stop at actual endpoints. Panels use separate numeric y-scales.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .025, 1, 1))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Explicitly hide GPU for this read-only CPU report')
    source, output = args.run_dir.resolve(), args.output_dir.resolve()
    if source.parent != DEV or output.parent != DEV or output.exists():
        raise ValueError('Read one direct DEV run and write one new direct DEV report directory')
    contract_path, completion_path = source / 'run_contract.json', source / 'completion.json'
    contract, completed = json.loads(contract_path.read_text()), json.loads(completion_path.read_text())
    if canonical_sha(contract) != args.contract_sha256 or completed['contract_sha256'] != args.contract_sha256:
        raise ValueError('Run contract does not match the explicitly selected finished experiment')
    if contract['kind'] != 'paired_acdd_student_lr120_development' or completed['status'] != 'paired_short_experiment_finished':
        raise ValueError('Both bounded arms must have completed or stopped with recorded endpoints')
    if contract['effective_optimizer_lrs'] != {'control_s2e4': {'student': .0002, 'discriminator': .0002}, 'primary_s1e4': {'student': .0001, 'discriminator': .0002}}:
        raise ValueError('Declared paired learning rates differ')
    if contract['lineage']['parent_checkpoint_sha256'] != 'f007dd4375bbf23a78372549e6e79b87c06e94fd38c07b5e09eb6624ae50ce44':
        raise ValueError('Comparison must start from the preserved full3107 state')
    arms, hashes = {}, {str(path): file_sha(path) for path in (contract_path, completion_path)}
    for label in ARMS:
        directory = source / label
        paths = {key: directory / value for key, value in {'steps': 'step_losses.jsonl', 'windows': 'loss_windows.jsonl', 'monitors': 'monitor.jsonl', 'completion': 'arm_completion.json'}.items()}
        for path in paths.values():
            hashes[str(path)] = file_sha(path)
        arm_completion = json.loads(paths['completion'].read_text())
        if arm_completion['contract_sha256'] != args.contract_sha256 or arm_completion != completed['arms'][label]:
            raise ValueError('Root and arm completion identities differ')
        if arm_completion['effective_optimizer_lrs'] != contract['effective_optimizer_lrs'][label]:
            raise ValueError('Actual arm/completion learning rates differ from the frozen contract')
        for field in ('student_snapshot', 'full_snapshot'):
            receipt = arm_completion[field]
            if receipt is None:
                if field == 'student_snapshot' or label == 'primary_s1e4':
                    raise ValueError('Expected endpoint checkpoint was not saved')
                continue
            path = Path(receipt['path']).resolve()
            if path.parent != directory or not receipt['committed'] or receipt['global_step'] != arm_completion['completed_global_step']:
                raise ValueError('Endpoint checkpoint receipt is inconsistent')
            actual = file_sha(path)
            if actual != receipt['sha256']:
                raise ValueError('Saved endpoint checkpoint no longer matches receipt')
            hashes[str(path)] = actual
        arms[label] = validate_arm(read_jsonl(paths['steps']), read_jsonl(paths['windows']), read_jsonl(paths['monitors']), arm_completion)
        retained, excluded = set(contract['training_indices']), set(contract['excluded_train_indices'])
        if len(retained) != 39335 or len(excluded) != 52 or retained & excluded:
            raise ValueError('Declared52-exclusion development pool changed')
        if any(not set(row['indices']).issubset(retained) or set(row['indices']) & excluded for row in arms[label]['steps']):
            raise ValueError('Training log contains examples outside the frozen retained pool')
    result = compare(arms)
    result.update(source_run=str(source), run_contract_sha256=args.contract_sha256, source_sha256=hashes, script_sha256=file_sha(Path(__file__)))
    output.mkdir()
    plot_all(arms, output / 'paired_loss_curves.png')
    (output / 'paired_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'status': result['status'], 'arms': result['arms'], 'latest_common': result['common_monitor_rows'][-1]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
