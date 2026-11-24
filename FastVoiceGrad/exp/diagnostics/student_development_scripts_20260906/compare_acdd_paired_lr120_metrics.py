"""只读比较学习率短对照的三组固定90对客观结果；不启动模型或评分器。"""

import argparse  # 调用者须指定预先写好的三组身份清单。
import hashlib  # 校验冻结统计代码、权重及所有已完成结果的字节内容。
import importlib.util  # 私有导入已审查的候选读取器，不修改历史统计实现。
import json  # 仅在全新目录保存比较结果。
import re  # 检查逐对q950哈希，拒绝缺项及占位符。
import sys  # Windows只能编辑，全部汇总计算限定租用服务器。
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run this CPU comparison only on the Matpool rental')
HELPER = ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development_candidate.py'
HELPER_SHA = '2c08b3700f405dc9deb85e6eecc5ff63309642e15113012242f81580704078e6'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Frozen candidate validator changed')
spec = importlib.util.spec_from_file_location('_paired_lr90_frozen_helper', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)  # 此helper同时真实核对租用机身份，且只导入CPU统计代码。
base, np, DEV = helper.base, helper.np, helper.base.DEV
PAIRED = DEV / 'acdd_lr_paired120_from3107_v1'
PARENT = DEV / 'acdd_s2000_to4000_v1'
PAIRED_SHA = '25c6e14693a940d94e18a85f50e8e3c08bbfbf1e15f53580117662a0f86ad909'
PARENT_SHA = '5fbd82bd161a7881f717b10b4d16e039f259e12ea7372b6cccb23e53b3326444'
PARENT_FULL_SHA = 'f007dd4375bbf23a78372549e6e79b87c06e94fd38c07b5e09eb6624ae50ce44'
FULL_SHA = 'f98b0763ec5a2edbd23d5d493e9dac549f85cf40cfac71e79f0b462f6fc6beb0'
HISTORY = DEV / 'comparison_acdd2000_vs_acdd600_v1/comparison.json'
HISTORY_SHA = 'e85521587ee37c11237fb0bb7d2847f2f5f4f43939c952287bf6c310285e0a39'
ROLES = ('primary', 'matched_control', 'parent')  # 两项差值的方向固定，不能事后改主要参照。
EXPECTED = {
    'primary': ('student_acdd_lr_half_s3227', PAIRED / 'primary_s1e4/student_step_003227.pt',
        '88da1f14234c9dcb164cdd249d47862b9ba105f1b7a1a858b4a18f9fa921b1dc', PAIRED / 'run_contract.json', PAIRED_SHA, 3227),
    'matched_control': ('student_acdd_lr_control_s3227', PAIRED / 'control_s2e4/student_step_003227.pt',
        '4457563a142e39e296994bb8c169314c5618a95d1fc24d28c1c02ce8f55449af', PAIRED / 'run_contract.json', PAIRED_SHA, 3227),
    'parent': ('student_acdd_parent_s3107', PARENT / 'student_step_003107.pt',
        'a9128f1acc9adee16c18f27e5e0361fac95398b3281e5a71567eaaa2359980c0', PARENT / 'run_contract.json', PARENT_SHA, 3107),
}
MAIN_METRICS = ('CER_corpus_percent', 'UTMOS', 'DNSMOS_OVRL', 'SVA_percent')


def validate_selection(value):
    if not isinstance(value, dict) or set(value) != {'schema_version', *ROLES, 'output_dir'} or type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Use the exact three-way version-one selection schema')
    fields = {'label', 'candidate_root', 'checkpoint', 'checkpoint_sha256', 'run_contract', 'training_contract_sha256', 'global_step', 'epoch'}
    for role in ROLES:
        entry = value[role]
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError('Incomplete or extra identity fields: ' + role)
        label, checkpoint, sha, contract, contract_sha, step = EXPECTED[role]
        expected = {'label': label, 'candidate_root': str(DEV / ('evaluation_candidate_' + label)),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha, 'run_contract': str(contract),
            'training_contract_sha256': contract_sha, 'global_step': step, 'epoch': 0}
        if entry != expected or type(entry['global_step']) is not int or type(entry['epoch']) is not int:
            raise ValueError('Selection differs from the predeclared exact role identity: ' + role)
    output = Path(value['output_dir'])
    if not output.is_absolute() or output.resolve().parent != DEV or output.exists() or output.is_symlink():
        raise ValueError('Comparison output must be a new direct child of DEV')
    return {role: value[role] for role in ROLES}, output.resolve()


def validate_lineage(parent, paired):
    if helper.canonical_sha(parent) != PARENT_SHA or helper.canonical_sha(paired) != PAIRED_SHA:
        raise ValueError('Frozen training contract canonical identity changed')
    if paired.get('kind') != 'paired_acdd_student_lr120_development' or paired.get('maximum_updates_per_arm') != 120:
        raise ValueError('Only the completed matched120 learning-rate experiment is accepted')
    expected = {'parent_checkpoint': str(PARENT / 'latest_full.pt'), 'parent_checkpoint_sha256': PARENT_FULL_SHA,
        'parent_contract_sha256': PARENT_SHA, 'original_step0_checkpoint_sha256': parent['lineage']['original_step0_checkpoint_sha256'],
        'start_after_global_step': 3107}
    if paired.get('lineage') != expected:
        raise ValueError('Both arms must start from the same complete3107 parent')
    if paired.get('effective_optimizer_lrs') != {'control_s2e4': {'student': .0002, 'discriminator': .0002},
                                               'primary_s1e4': {'student': .0001, 'discriminator': .0002}}:
        raise ValueError('Only the declared student LR may differ')
    for key in ('configuration', 'training_indices', 'excluded_train_indices', 'reconstruction_dev_indices',
                'fixed_development_inputs', 'fixed_development_source_noise_sha256', 'original_step0_guard_anchor'):
        if key not in parent or paired.get(key) != parent[key]:
            raise ValueError('A held-fixed source/data/configuration field changed: ' + key)


def validate_completion(completion, entries):
    if completion.get('status') != 'paired_short_experiment_finished' or completion.get('contract_sha256') != PAIRED_SHA:
        raise ValueError('The complete two-arm experiment receipt is required')
    if completion.get('common_monitor_updates') != [0, 20, 40, 60, 80, 100, 120]:
        raise ValueError('Both selected endpoints must have all common120-step checks')
    if not isinstance(completion.get('arms'), dict) or set(completion['arms']) != {'primary_s1e4', 'control_s2e4'}:
        raise ValueError('Both and only the declared arm receipts are required')
    for role, arm in (('primary', 'primary_s1e4'), ('matched_control', 'control_s2e4')):
        item = completion['arms'][arm]
        if not isinstance(item, dict):
            raise ValueError('Arm receipt must be a mapping')
        expected_reason = 'maximum120_updates_completed' if role == 'primary' else 'development_three_checks_rising'
        if (item.get('status'), item.get('reason'), item.get('completed_global_step'), item.get('updates'),
                item.get('partial_batch_may_have_updated_D')) != ('completed' if role == 'primary' else 'stopped', expected_reason, 3227, 120, False):
            raise ValueError('Selection is not the actual predeclared full120 endpoint: ' + role)
        snapshot = item.get('student_snapshot', {})
        if not isinstance(snapshot, dict) or not snapshot.get('committed') or not snapshot.get('frozen_readonly') or snapshot.get('global_step') != 3227:
            raise ValueError('Selected student snapshot lacks a final frozen receipt')
        if snapshot.get('path') != entries[role]['checkpoint'] or snapshot.get('sha256') != entries[role]['checkpoint_sha256']:
            raise ValueError('Receipt and selected checkpoint differ')
    primary = completion['arms']['primary_s1e4']
    full = primary.get('full_snapshot', {})
    if not isinstance(full, dict) or primary.get('last_full_saved_global_step') != 3227 or full.get('sha256') != FULL_SHA or full.get('path') != str(PAIRED / 'primary_s1e4/final_full.pt') or not full.get('committed') or not full.get('frozen_readonly'):
        raise ValueError('The primary endpoint must retain its declared complete resume state')
    first = primary.get('first3108_loss_absolute_differences', {})
    if not isinstance(first, dict) or set(first) != {'student_total', 'student_adv', 'fm_raw', 'fm_weighted', 'dist_raw', 'dist_weighted', 'discriminator'} or any(value != 0 for value in first.values()):
        raise ValueError('The paired first-step loss identity receipt changed')


def validate_same_noise(reports):
    expected_keys = {str(i) for i in range(90)}
    first = reports[ROLES[0]].get('x950_sha256_per_pair', {})
    if set(first) != expected_keys or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in first.values()):
        raise ValueError('All90 actualq950 tensor hashes are required')
    if any(reports[role].get('x950_sha256_per_pair') != first for role in ROLES[1:]):
        raise ValueError('The three systems did not use identical actualq950 inputs')
    return first  # 把全部哈希原样写入结果，不凭相同种子宣称张量相同。


def source_clusters(template):
    if len(template) != 90 or [int(row['pair_id']) for row in template] != list(range(90)):
        raise ValueError('Keep all90 original ordered pairs')
    clusters = sorted({(row['source_speaker'], row['source_utt']) for row in template})
    if len(clusters) != 10 or any(sum((r['source_speaker'], r['source_utt']) == cluster for r in template) != 9 for cluster in clusters):
        raise ValueError('Bootstrap units must be the original10 source utterances with9 targets each')
    return clusters


def compare_rows(reference, candidate, clusters, weights):
    # 同一次源簇抽样同时用于两模型；每次CER均先合计错误与字符数，再相除。
    left, right = base.cluster_sums(reference, clusters), base.cluster_sums(candidate, clusters)
    before, after = base.values_from_sums(left.sum(axis=0)), base.values_from_sums(right.sum(axis=0))
    interval = base.confidence_interval(base.values_from_sums(weights @ right) - base.values_from_sums(weights @ left))
    return {name: {'reference': float(before[i]), 'candidate': float(after[i]), 'delta': float(after[i] - before[i]),
        'cluster_bootstrap_95_percentile_ci': interval[i].tolist(),
        'unit': 'percentage_points' if name in ('CER_corpus_percent', 'SVA_percent') else 'score',
        'favorable_delta_sign': 'negative' if name == 'CER_corpus_percent' else 'positive'} for i, name in enumerate(base.METRICS)}


def absolute_values(rows, clusters):
    values = base.values_from_sums(base.cluster_sums(rows, clusters).sum(axis=0))
    return {name: float(values[i]) for i, name in enumerate(base.METRICS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    selection_path = args.selection.resolve()
    if selection_path.parent != DEV:
        raise ValueError('Keep the predeclared selection in the fixed development directory')
    hashes = {}
    selection = base.read_json(selection_path, hashes)
    entries, output = validate_selection(selection)
    history = base.read_json(HISTORY, hashes)
    if hashes[str(HISTORY)] != HISTORY_SHA:
        raise ValueError('The historical600/2000 summary changed')
    historical, template, development, metric_identity, originals = base.load_results(DEV / 'evaluation_baselines_v1')
    hashes.update(originals)  # 原四基线与评分器全部用冻结校验器检查，绝不重新评分。
    systems, contracts, reports, identities = {}, {}, {}, {}
    for role in ROLES:
        entry = entries[role]
        namespace = SimpleNamespace(**{k: v for k, v in entry.items() if k not in ('checkpoint', 'run_contract')})
        namespace.candidate_root = Path(namespace.candidate_root)
        rows, contract, identity, report = helper.load_candidate(namespace, template, development, metric_identity, hashes)
        if Path(identity['systems'][entry['label']]['checkpoint']).resolve() != Path(entry['checkpoint']) or Path(identity['candidate_training_contract']).resolve() != Path(entry['run_contract']):
            raise ValueError('Inference used a different checkpoint or shared training contract: ' + role)
        systems[role], contracts[role], identities[role], reports[role] = rows, contract, identity, report
    if contracts['primary'] != contracts['matched_control']:
        raise ValueError('The two arms must point to the identical shared root contract')
    validate_lineage(contracts['parent'], contracts['primary'])
    completion = base.read_json(PAIRED / 'completion.json', hashes)
    validate_completion(completion, entries)
    for role, arm in (('primary', 'primary_s1e4'), ('matched_control', 'control_s2e4')):
        if base.read_json(PAIRED / arm / 'arm_completion.json', hashes) != completion['arms'][arm]:
            raise ValueError('Arm and root completion receipts disagree')
    full_path = PAIRED / 'primary_s1e4/final_full.pt'
    if base.digest(full_path) != FULL_SHA:
        raise ValueError('The actual primary full checkpoint changed')
    hashes[str(full_path)] = FULL_SHA
    for path, expected in contracts['primary']['file_sha256'].items():
        if ROOT not in Path(path).resolve().parents or base.digest(path) != expected:
            raise ValueError('Frozen source or training dependency changed: ' + path)
        hashes[path] = expected
    q950 = validate_same_noise(reports)
    clusters = source_clusters(template)
    weights = base.bootstrap_weights(10, 10000, 2026090617)  # 同一矩阵用于全部八列和两项预声明比较。
    comparisons = {role: compare_rows(systems[role], systems['primary'], clusters, weights) for role in ('matched_control', 'parent')}
    absolute = {entries[role]['label']: absolute_values(systems[role], clusters) for role in ROLES}
    history_absolute = {label: absolute_values(rows, clusters) for label, rows in historical.items()}
    for label, field in (('student_acdd_clean_s600', 'acdd600'), ('student_acdd_clean_s2000', 'acdd2000')):
        history_absolute[label] = {metric: history['metrics'][metric][field] for metric in base.METRICS}
    result = {'status': 'complete_paired_lr90_comparison_only', 'selection': selection,
        'primary_comparison': 'primary minus matched_control', 'secondary_comparison': 'primary minus parent',
        'systems': absolute, 'primary_minus_references': comparisons, 'main_metric_order': list(MAIN_METRICS),
        'all_metric_order': list(base.METRICS), 'n_pairs_per_system': 90, 'n_source_clusters': 10,
        'actual_q950_sha256_per_pair': q950, 'training_completion': completion, 'metric_identity': metric_identity,
        'historical_absolute_references_only': history_absolute, 'historical600_2000_summary': {'path': str(HISTORY), 'sha256': HISTORY_SHA},
        'bootstrap': {'method': 'paired source-utterance cluster percentile bootstrap', 'repetitions': 10000,
            'seed': 2026090617, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__, 'clusters': [list(c) for c in clusters],
            'targets_per_source': 9, 'confidence_level': .95, 'multiple_comparison_adjustment': None,
            'weight_matrix_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()},
        'definitions': {'CER': '100*sum(char_errors)/sum(reference_chars)', 'SVA': '100*mean(cosine>0.25)', 'DNSMOS_headline': 'mos_ovr'},
        'source_sha256': hashes, 'frozen_helper_sha256': HELPER_SHA, 'script_sha256': base.digest(Path(__file__)),
        'limitations': ['All90 fixed development pairs are retained. These are not the formal8100 unseen pairs or paper scores.',
            'Only10 source clusters and fixed target references; intervals are exploratory and not multiplicity adjusted.',
            'The primary matched120 contrast changes only student LR; the parent contrast additionally includes120 optimization steps.',
            'Model selection already uses this development set; this comparison cannot establish independent final generalization.',
            'Teacher, historical models and original normalization statistics were exposed to development utterances; new updates excluded52 utterances.',
            'Historical600/2000 and four baseline scores are context only; no new paired significance contrasts against them are asserted.',
            'No training, inference, ASR, audio scoring, or checkpoint modification is performed by this script.']}
    for path, expected in hashes.items():
        if base.digest(path) != expected:
            raise RuntimeError('An input changed while comparing completed results: ' + path)
    output.mkdir(exist_ok=False)  # 任一结果缺失或身份不符时不会创建成功报告。
    with (output / 'comparison.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    lines = ['# 学生学习率短对照的客观指标', '',
        '我保留同一组90个转换配对，先比较训练步数相同的两组，再比较降低学习率后的学生与3107步父模型。下面的区间按10条源录音整组重抽样计算，未作多重比较校正。', '']
    for role in ('matched_control', 'parent'):
        lines += [f'## 降低学习率组减{entries[role]["label"]}', '', '| 指标 | 参照 | 降低学习率组 | 差值 | 95%配对区间 |', '|---|---:|---:|---:|---|']
        for name in (*MAIN_METRICS, *(key for key in base.METRICS if key not in MAIN_METRICS)):
            item = comparisons[role][name]
            low, high = item['cluster_bootstrap_95_percentile_ci']
            lines.append(f"| {name} | {item['reference']:.4f} | {item['candidate']:.4f} | {item['delta']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
        lines.append('')
    lines += ['CER负差表示错误减少，其他主指标正差表示分数提高。全部DNSMOS输出列、历史参照和90对输入哈希保留在JSON中。这里只检验这次短对照，不能据此保证长期训练效果或正式unseen性能。', '']
    with (output / 'comparison_CN.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps({'output': str(output), 'status': result['status'], 'primary_minus_references': comparisons}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()  # CPU测试导入不会读取未来评分结果，更不会自动启动GPU。
