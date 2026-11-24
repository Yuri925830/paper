# 只汇总已完成的90对结果，不运行ASR、声码器、推理或训练，不修改任何原始转写。
"""Read-only, SHA-bound pairwise CV1200 error-count and UTMOS audit."""

import csv  # 原样读取已有CER和UTMOS逐对结果。
import hashlib  # 绑定已由主线程验真的comparison及本次读取的每个输入。
import io  # 直接解析已验哈希的字节，避免验完文件又打开一份不同内容。
import json  # 仅在新的独立目录保存审计报告。
import sys  # 拒绝本地Windows执行项目汇总。
from collections import Counter  # 检查10条源语音各有9个目标，不能悄悄少样本。
from pathlib import Path  # 真实数据只读，新输出不覆盖已有publication。

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run project analysis only on the rented Matpool server')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
COMPARISON = DEV / 'comparison_cv1200_vs_controls_v1/comparison.json'
COMPARISON_SHA = '4864f5013c11f3bbaba74a671d542ae2c16633d9ecbec6f12ea372542c1528a4'
OUTPUT = DEV / 'audit_cv1200_pair_changes_v1'  # 仅创建这份新的小审计，不删除或覆盖任何旧目录。
LABELS = ('student_conversion_score_s1200', 'student_acdd_clean_s1200', 'student_acdd_clean_s600')


def main():
    raw = COMPARISON.read_bytes()
    if hashlib.sha256(raw).hexdigest() != COMPARISON_SHA:
        raise RuntimeError('The reviewed CV comparison has changed')
    comparison = json.loads(raw)
    if comparison['matched_control_status'] != 'verified_complete_same_step_control':
        raise RuntimeError('Same-step control has not been verified')
    if (comparison['n_pairs_per_system'], comparison['n_source_clusters']) != (90, 10):
        raise RuntimeError('Expected the complete90-pair,10-source development protocol')
    if comparison['selection']['candidate']['checkpoint_sha256'] != '583999eea4777c3f3ad85ec7850f5d79c2d023b460d32fabc1ffa8d12499756f':
        raise RuntimeError('Unexpected CV1200 checkpoint identity')
    read_hashes = {str(COMPARISON): COMPARISON_SHA}

    def load_rows(path):
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != comparison['source_sha256'][str(path)]:
            raise RuntimeError('Metric or pair manifest differs from its verified receipt: ' + str(path))
        read_hashes[str(path)] = actual
        rows = list(csv.DictReader(io.StringIO(content.decode('utf-8'))))
        ids = [int(row['pair_id']) for row in rows]
        if len(ids) != 90 or set(ids) != set(range(90)):
            raise RuntimeError('Pair identity is incomplete or duplicated')
        return {int(row['pair_id']): row for row in rows}  # 对齐基于pair_id，不依赖CSV排序。

    def metric_path(label, name):
        matches = [Path(path) for path in comparison['source_sha256'] if path.endswith('/' + label + '/metrics/' + name)]
        if len(matches) != 1:
            raise RuntimeError('Metric identity is missing or ambiguous')
        return matches[0]

    pairs = load_rows(DEV / 'pairs.csv')
    counts = Counter(row['source_utt'] for row in pairs.values())
    if len(counts) != 10 or set(counts.values()) != {9}:
        raise RuntimeError('Every one of the10 source utterances must retain9 targets')
    cer = {label: load_rows(metric_path(label, 'cer.csv')) for label in LABELS}
    utmos = {label: load_rows(metric_path(label, 'utmos.csv')) for label in LABELS}
    rows = []
    for pair_id in range(90):
        entries = [cer[label][pair_id] for label in LABELS]
        if len({row['reference_text_normalized'] for row in entries}) != 1 or len({row['reference_chars'] for row in entries}) != 1:
            raise RuntimeError('Reference text or official CER denominator changed across systems')
        errors = {label: int(cer[label][pair_id]['char_errors']) for label in LABELS}
        scores = {label: float(utmos[label][pair_id]['utmos']) for label in LABELS}
        rows.append({'pair_id': pair_id, 'source_speaker': pairs[pair_id]['source_speaker'],
            'source_utt': pairs[pair_id]['source_utt'], 'target_speaker': pairs[pair_id]['reference_speaker'],
            'reference_text_normalized': entries[0]['reference_text_normalized'],
            'reference_chars': int(entries[0]['reference_chars']), 'char_errors': errors,
            'hypothesis_text_normalized': {label: cer[label][pair_id]['hypothesis_text_normalized'] for label in LABELS},
            'utmos': scores, 'error_delta_cv_minus_matched1200': errors[LABELS[0]] - errors[LABELS[1]],
            'error_delta_cv_minus_parent600': errors[LABELS[0]] - errors[LABELS[2]],
            'utmos_delta_cv_minus_parent600': scores[LABELS[0]] - scores[LABELS[2]]})
    denominator = sum(row['reference_chars'] for row in rows)  # 沿用评价器已保存的分母，不重新规范化文本。
    totals = {label: sum(row['char_errors'][label] for row in rows) for label in LABELS}
    for label in LABELS[1:]:
        recorded = comparison['candidate_minus_references'][label]
        if abs(100 * totals[label] / denominator - recorded['CER_corpus_percent']['reference']) > 1e-12:
            raise RuntimeError('Pairwise CER does not reproduce the verified aggregate')
        if abs(100 * totals[LABELS[0]] / denominator - recorded['CER_corpus_percent']['candidate']) > 1e-12:
            raise RuntimeError('Candidate pairwise CER differs from its aggregate')
    source_groups = []
    for source in sorted(counts):
        group = [row for row in rows if row['source_utt'] == source]
        delta = [row['error_delta_cv_minus_matched1200'] for row in group]
        ud = [row['utmos_delta_cv_minus_parent600'] for row in group]
        source_groups.append({'source_utt': source, 'source_speaker': group[0]['source_speaker'], 'pairs': len(group),
            'char_errors': {label: sum(row['char_errors'][label] for row in group) for label in LABELS},
            'gross_positive_pair_error_count_delta': sum(max(value, 0) for value in delta),
            'gross_negative_pair_error_count_delta_magnitude': sum(max(-value, 0) for value in delta),
            'net_error_delta_vs_matched1200': sum(delta), 'error_worse_pairs': sum(value > 0 for value in delta),
            'error_better_pairs': sum(value < 0 for value in delta), 'error_equal_pairs': sum(value == 0 for value in delta),
            'utmos_mean_delta_vs_parent600': sum(ud) / len(ud),
            'utmos_down_pairs': sum(value < 0 for value in ud), 'utmos_up_pairs': sum(value > 0 for value in ud)})
    changed = [row for row in rows if row['hypothesis_text_normalized'][LABELS[0]] != row['hypothesis_text_normalized'][LABELS[1]]]
    headline = {
        'n_pairs': 90, 'n_source_groups': 10, 'reference_chars': denominator, 'char_errors': totals,
        'net_error_delta_vs_matched1200': sum(row['error_delta_cv_minus_matched1200'] for row in rows),
        'gross_positive_pair_error_count_delta': sum(max(row['error_delta_cv_minus_matched1200'], 0) for row in rows),
        'gross_negative_pair_error_count_delta_magnitude': sum(max(-row['error_delta_cv_minus_matched1200'], 0) for row in rows),
        'error_worse_pairs': sum(row['error_delta_cv_minus_matched1200'] > 0 for row in rows),
        'error_better_pairs': sum(row['error_delta_cv_minus_matched1200'] < 0 for row in rows),
        'error_equal_pairs': sum(row['error_delta_cv_minus_matched1200'] == 0 for row in rows),
        'source_groups_with_any_error_worse_pair': sum(group['error_worse_pairs'] > 0 for group in source_groups),
        'source_groups_net_worse': sum(group['net_error_delta_vs_matched1200'] > 0 for group in source_groups),
        'source_groups_net_better': sum(group['net_error_delta_vs_matched1200'] < 0 for group in source_groups),
        'source_groups_net_equal': sum(group['net_error_delta_vs_matched1200'] == 0 for group in source_groups),
        'hypothesis_changed_pairs': len(changed),
        'hypothesis_changed_but_error_count_equal_pairs': [row['pair_id'] for row in changed if row['error_delta_cv_minus_matched1200'] == 0],
        'utmos_down_pairs_vs_parent600': sum(row['utmos_delta_cv_minus_parent600'] < 0 for row in rows),
        'utmos_up_pairs_vs_parent600': sum(row['utmos_delta_cv_minus_parent600'] > 0 for row in rows),
        'utmos_source_groups_mean_down_vs_parent600': sum(group['utmos_mean_delta_vs_parent600'] < 0 for group in source_groups),
        'utmos_mean_delta_vs_parent600': sum(row['utmos_delta_cv_minus_parent600'] for row in rows) / 90,
    }
    if headline['net_error_delta_vs_matched1200'] != 13:
        raise RuntimeError('The reviewed13-error difference was not reproduced')
    if abs(headline['utmos_mean_delta_vs_parent600'] - comparison['candidate_minus_references'][LABELS[2]]['UTMOS']['delta']) > 1e-12:
        raise RuntimeError('Pairwise UTMOS does not reproduce the reviewed aggregate')
    for path, expected in read_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('An input changed during read-only analysis')
    result = {'status': 'complete_existing_metrics_readonly_audit', 'headline': headline, 'by_source': source_groups,
        'changed_transcripts_vs_matched1200': changed, 'all90_pairs': rows, 'source_sha256': read_hashes,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['All90 pairs retained; no normalization, spelling correction or metric recomputation.',
            'Gross additions/removals are positive/negative per-pair error-count differences, not tracked identities of individual edit operations.',
            'Equal CER counts can accompany changed hypotheses; all such pairs are retained.',
            'Text comparison does not localize acoustic events or establish a causal failure mechanism.',
            'Only10 source utterances with9 target speakers each; this is development analysis, not independent formal-test evidence.']}
    OUTPUT.mkdir()  # 已有同名审计时直接失败，绝不覆盖旧结果。
    (OUTPUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT / 'summary.json'), 'headline': headline, 'by_source': source_groups}, ensure_ascii=False))


if __name__ == '__main__':
    main()  # 明确的租用CPU命令才执行这个小汇总。
