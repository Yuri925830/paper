"""只读比较原ACDD的实际续训终点与2000步、600步及四个历史基线。"""

import argparse  # 必须显式提供候选选择清单，不能自动挑高分检查点。
import hashlib  # 核对冻结统计代码和输入文件的真实字节。
import importlib.util  # 在独立命名空间复用旧校验器，不修改它。
import json  # 保存完整来源和所有客观指标。
import re  # 拒绝不完整SHA及带路径符号的系统标签。
import sys  # 所有统计计算限定租用服务器。
from pathlib import Path  # 新报告必须写入开发区内的新目录。
from types import SimpleNamespace  # 向冻结候选校验器传递显式身份。

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run this CPU comparison only on the Matpool rental.')
HELPER = ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development_candidate.py'
HELPER_SHA = '2c08b3700f405dc9deb85e6eecc5ff63309642e15113012242f81580704078e6'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Frozen candidate validation/bootstrap helper changed.')
spec = importlib.util.spec_from_file_location('_acdd_continuation_summary_helper', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)  # 原helper同时核验租用机身份，只导入CPU统计函数。
base, np, DEV = helper.base, helper.np, helper.base.DEV
PARENT_LABEL, EARLY_LABEL = 'student_acdd_clean_s2000', 'student_acdd_clean_s600'
PARENT_DIR = DEV / 'acdd_clean_resume_s2000_v1'
PARENT_FULL_SHA = '77de35a153636d8156a45c6c23b21191b2c663d55963df874e17946e3a7ff72c'
PARENT_CONTRACT_SHA = 'f7d351f786c9cb7f4583327a8ff7a4b6d2c551c40bdbf36ce9e54dbe1fad933b'
EARLY_CONTRACT_SHA = 'f9472aea9332e8c261e4045ceac6630efe8f9ba012f135d5b4aba45216b32c13'
DEVELOPMENT_FILE_SHA = '08737551d34e00ba0b80a5e34adbf73fcfc6c70ece6bc8dc4a6c466e9b87afb1'
ANCHOR_SHA = 'ec504a627350c102cab1647befdc74234969b616238d2cd07d5350ef0c322162'
DRIVER = ROOT / 'exp/diagnostics/student_performance_tools_20260906/run_acdd_s2000_to4000_v1.py'
DRIVER_SHA = '2894c58180d8b2934de788800930267fac412d51aa2cda5cf257d6d92452e373'
FIXED = [  # 两个已完成的同轨迹参照固定身份，不允许从选择清单换掉。
    {'label': PARENT_LABEL, 'candidate_root': str(DEV / ('evaluation_candidate_' + PARENT_LABEL)),
     'checkpoint': str(PARENT_DIR / 'full_step_002000.pt'), 'checkpoint_sha256': PARENT_FULL_SHA,
     'training_contract_sha256': PARENT_CONTRACT_SHA, 'global_step': 2000, 'epoch': 0},
    {'label': EARLY_LABEL, 'candidate_root': str(DEV / ('evaluation_candidate_' + EARLY_LABEL)),
     'checkpoint': str(DEV / 'acdd_clean_600_v1/student_step_000600.pt'),
     'checkpoint_sha256': 'b3eedcddc465b8cc79756f8aedcfc126eef5a807c35334a712b74c6691562882',
     'training_contract_sha256': EARLY_CONTRACT_SHA, 'global_step': 600, 'epoch': 0},
]
MAIN_METRICS = ('CER_corpus_percent', 'UTMOS', 'DNSMOS_OVRL', 'SVA_percent')


def validate_selection(value):
    required = {'schema_version', 'candidate', 'output_dir'}  # 无更换参照、改评分器或删除配对的开关。
    if not isinstance(value, dict) or set(value) != required or type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Use the exact version-one selection schema.')
    entry = value['candidate']
    fields = {'label', 'candidate_root', 'checkpoint', 'checkpoint_sha256', 'training_contract_sha256', 'global_step', 'epoch'}
    if not isinstance(entry, dict) or set(entry) != fields:
        raise ValueError('Candidate identity fields must be complete and exact.')
    if not isinstance(entry['label'], str) or not re.fullmatch(r'student_[a-z0-9_]{1,80}', entry['label']) or entry['label'] in (PARENT_LABEL, EARLY_LABEL, *base.SYSTEMS):
        raise ValueError('Use a distinct explicit student label.')
    if type(entry['global_step']) is not int or not 2000 < entry['global_step'] <= 4000 or type(entry['epoch']) is not int or entry['epoch'] != 0:
        raise ValueError('Use the actual saved step after2000 and at most4000, with epoch zero.')
    for field in ('checkpoint_sha256', 'training_contract_sha256'):
        if not isinstance(entry[field], str) or not re.fullmatch(r'[0-9a-f]{64}', entry[field]):
            raise ValueError('Pass full lowercase SHA-256 values; placeholders are forbidden.')
    for field in ('checkpoint', 'candidate_root'):
        path = Path(entry[field])
        if not path.is_absolute() or DEV not in path.resolve().parents:
            raise ValueError('Candidate files must resolve inside the fixed development directory.')
    output = Path(value['output_dir'])
    if not output.is_absolute() or output.resolve().parent != DEV or output.exists() or output.is_symlink():
        raise ValueError('Output must be a new direct child of the development directory.')
    return entry, output.resolve()  # 此处只校验，不创建目录或改写选择清单。


def validate_lineage(early, parent, child):
    if parent.get('kind') != 'acdd_development_continuation' or parent.get('stop_global_step') != 2000:
        raise ValueError('The primary reference must be the original ACDD2000 continuation.')
    parent_lineage = parent.get('lineage', {})
    if parent_lineage.get('parent_contract_sha256') != EARLY_CONTRACT_SHA or parent_lineage.get('start_after_global_step') != 600:
        raise ValueError('ACDD2000 does not descend from the frozen600 contract.')
    if child.get('kind') != 'acdd2000_to4000_development' or child.get('stop_global_step') != 4000:
        raise ValueError('Only the declared continuation is accepted; smoke and other objectives are excluded.')
    expected = {'parent_directory': str(PARENT_DIR), 'parent_checkpoint': str(PARENT_DIR / 'full_step_002000.pt'),
                'parent_checkpoint_sha256': PARENT_FULL_SHA, 'parent_contract_sha256': PARENT_CONTRACT_SHA,
                'original_step0_checkpoint_sha256': ANCHOR_SHA, 'start_after_global_step': 2000}
    if child.get('lineage') != expected:
        raise ValueError('Candidate did not restore the declared complete2000 parent and originalS0 anchor.')
    for field in ('configuration', 'training_indices', 'excluded_train_indices', 'reconstruction_dev_indices',
                  'fixed_development_inputs', 'fixed_development_source_noise_sha256'):
        if field not in early or parent.get(field) != early[field] or child.get(field) != parent[field]:
            raise ValueError('Held-fixed ACDD configuration or data changed: ' + field)
    if child.get('seed') != 1234 or early.get('seed') != 1234:
        raise ValueError('The original seed changed.')
    if child.get('development_contract_file_sha256') != DEVELOPMENT_FILE_SHA:
        raise ValueError('Wrong development file SHA; canonical SHA is not interchangeable.')
    if child.get('method') != 'unchanged production train_batch; reconstruction adv+2FM+45clean_x0 sqrt_alpha_bar score':
        raise ValueError('Candidate is not the reviewed unchanged reconstruction ACDD objective.')
    plan = {'first_step': 2001, 'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((step-1)/1229)',
            'crop_seed': '1234+100000+step', 'noise_seed': '1234+200000+step'}
    if child.get('batch_plan') != plan or not 1 <= child.get('max_seconds', 0) <= 3600:
        raise ValueError('Absolute batch/crop/noise replay or bounded duration changed.')
    if child.get('original_step0_guard_anchor') != parent.get('original_step0_guard_anchor') or child.get('guard_ratios') != {'mel_l1_upper': 1.10, 'prediction_rms_lower': 0.80}:
        raise ValueError('OriginalS0 protection was rebased or changed.')
    if child.get('file_sha256', {}).get(str(DRIVER)) != DRIVER_SHA:
        raise ValueError('The continuation source is not the reviewed frozen driver.')


def validate_completion(completion, frozen, entry, contract_path):
    accepted = {'global_step_budget_complete', 'requested_stop', 'time_budget_complete',
                'original_step0_development_regression_guard', 'disk_space_guard', 'disk_space_guard_last_full_preserved'}
    step, contract_sha = entry['global_step'], entry['training_contract_sha256']
    completed = completion.get('global_step')  # 实际更新游标可能高于低盘时保住的最后full。
    if completion.get('status') not in accepted or type(completed) is not int or not step <= completed <= 4000:
        raise ValueError('A completed bounded segment is required, not running/failed/smoke status.')
    if completion.get('run_contract_sha256') != contract_sha or completion.get('last_saved_global_step') != step or completion.get('start_after_global_step') != 2000:
        raise ValueError('Selection must identify the actual final saved checkpoint of this segment.')
    if completion.get('unsaved_completed_steps') != completed - step or completion.get('completed_new_steps') != completed - 2000:
        raise ValueError('Completion update/save counters are inconsistent.')
    if completion['status'] == 'global_step_budget_complete' and (step != 4000 or completed != 4000):
        raise ValueError('The4000 budget cannot be marked complete at an earlier checkpoint.')
    receipt = completion.get('latest_full')
    if not isinstance(receipt, dict) or not receipt.get('full_committed') or receipt.get('global_step') != step or receipt.get('run_contract_sha256') != contract_sha:
        raise ValueError('Final full-state receipt is missing or refers to another checkpoint.')
    if frozen != dict(receipt, frozen=True):
        raise ValueError('Final frozen-full attestation differs from completion.')
    directory = contract_path.parent.resolve()
    if Path(receipt['path']).resolve() != directory / 'latest_full.pt':
        raise ValueError('Final full checkpoint escapes its new continuation directory.')
    selected = Path(entry['checkpoint']).resolve()
    if selected == Path(receipt['path']).resolve():
        expected_sha = receipt.get('checkpoint_sha256')
    else:
        snapshot = receipt.get('student_snapshot') or {}
        if selected != directory / f'student_step_{step:06d}.pt' or snapshot.get('path') != str(selected) or snapshot.get('committed') is not True:
            raise ValueError('Selected student snapshot is not the actual final saved step.')
        expected_sha = snapshot.get('sha256')
    if expected_sha != entry['checkpoint_sha256']:
        raise ValueError('Selected checkpoint SHA does not match the final save receipt.')
    return Path(receipt['path']).resolve(), receipt['checkpoint_sha256']


def same_noise(left, right):
    a, b = left.get('x950_sha256_per_pair', {}), right.get('x950_sha256_per_pair', {})
    if set(a) != {str(i) for i in range(90)} or a != b or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in a.values()):
        raise ValueError('All90 actual q950 tensor hashes must agree.')


def compare_rows(reference, candidate, clusters, weights):
    before, after = base.cluster_sums(reference, clusters), base.cluster_sums(candidate, clusters)
    p0, p1 = base.values_from_sums(before.sum(axis=0)), base.values_from_sums(after.sum(axis=0))
    interval = base.confidence_interval(base.values_from_sums(weights @ after) - base.values_from_sums(weights @ before))
    return {name: {'reference': float(p0[i]), 'candidate': float(p1[i]), 'delta': float(p1[i] - p0[i]),
                   'cluster_bootstrap_95_percentile_ci': interval[i].tolist(),
                   'unit': 'percentage_points' if name in ('CER_corpus_percent', 'SVA_percent') else 'score',
                   'favorable_delta_sign': 'negative' if name == 'CER_corpus_percent' else 'positive'}
            for i, name in enumerate(base.METRICS)}  # 保留四主指标、DNSMOS其他头及连续声纹余弦。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    selection_path = args.selection.resolve()
    if ROOT / 'exp/diagnostics' not in selection_path.parents:
        raise ValueError('Keep the explicit selection inside rental diagnostics.')
    hashes = {}
    selection = base.read_json(selection_path, hashes)
    candidate, output = validate_selection(selection)
    systems, template, development, metric_identity, originals = base.load_results(DEV / 'evaluation_baselines_v1')
    hashes.update(originals)  # 原四基线、固定评分器与源文件按旧逻辑完整验证。
    contracts, identities, reports = {}, {}, {}
    for entry in [*FIXED, candidate]:
        namespace = SimpleNamespace(**{k: v for k, v in entry.items() if k != 'checkpoint'})
        namespace.candidate_root = Path(namespace.candidate_root)
        rows, contract, identity, report = helper.load_candidate(namespace, template, development, metric_identity, hashes)
        label = entry['label']
        if Path(identity['systems'][label]['checkpoint']).resolve() != Path(entry['checkpoint']).resolve():
            raise ValueError('Inference used a checkpoint other than the explicit selection.')
        systems[label], contracts[label], identities[label], reports[label] = rows, contract, identity, report
    label, step = candidate['label'], candidate['global_step']
    validate_lineage(contracts[EARLY_LABEL], contracts[PARENT_LABEL], contracts[label])
    for reference in (PARENT_LABEL, EARLY_LABEL):
        same_noise(reports[reference], reports[label])  # 对新轨迹参照核对每对实际张量，不只看seed文字。
    contract_path = Path(identities[label]['candidate_training_contract']).resolve()
    completion = base.read_json(contract_path.parent / 'completion.json', hashes)
    frozen = base.read_json(contract_path.parent / 'final_full_frozen.json', hashes)
    full_path, full_sha = validate_completion(completion, frozen, candidate, contract_path)
    if base.digest(full_path) != full_sha:
        raise ValueError('Frozen final full checkpoint bytes changed.')
    hashes[str(full_path)] = full_sha  # 即使评分只读S，也核对完整恢复状态的真实身份。
    for path, expected in contracts[label]['file_sha256'].items():
        if ROOT not in Path(path).resolve().parents or base.digest(path) != expected:
            raise ValueError('Continuation source/dependency changed: ' + path)
        hashes[path] = expected  # 教师、G、统计量及生产源码也进入最终复核。
    clusters = sorted({(r['source_speaker'], r['source_utt']) for r in template})
    if len(clusters) != 10 or any(sum((r['source_speaker'], r['source_utt']) == c for r in template) != 9 for c in clusters):
        raise ValueError('Expected all90 pairs grouped as10 source utterances with9 targets each.')
    weights = base.bootstrap_weights(10, 10000, 2026090617)
    references = [PARENT_LABEL, EARLY_LABEL, *base.SYSTEMS]  # 主要对照2000固定排首位，四历史基线全部保留。
    comparisons = {r: compare_rows(systems[r], systems[label], clusters, weights) for r in references}
    audit = base.cer_concentration_audit(systems, clusters)  # 逐源/目标分解不改变主数分母。
    utmos_high = comparisons[PARENT_LABEL]['UTMOS']['cluster_bootstrap_95_percentile_ci'][1]
    result = {
        'status': 'complete_acdd2000_continuation_comparison_only', 'selection': selection,
        'primary_reference': PARENT_LABEL, 'main_metric_order': list(MAIN_METRICS),
        'candidate_minus_references': comparisons, 'actual_saved_step': step, 'additional_saved_updates': step - 2000,
        'training_completion': completion, 'lineage_verified': True, 'n_pairs_per_system': 90, 'n_source_clusters': 10,
        'protocol_review': {'pause_next_segment_due_to_UTMOS': bool(utmos_high < 0),
                            'rule': 'Pause next segment if candidate-minus2000 UTMOS paired95% upper bound is below zero; no automatic continuation otherwise.'},
        'bootstrap': {'method': 'paired source-utterance cluster percentile bootstrap', 'repetitions': 10000,
                      'seed': 2026090617, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__,
                      'weight_matrix_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()},
        'verified_q950_hash_comparisons': [PARENT_LABEL, EARLY_LABEL],
        'historical_baselines_without_archived_q950_hashes': list(base.SYSTEMS),
        'cer_concentration_audit': audit,
        'target_p308_p317_breakdown': {name: [r for r in item['by_target'] if r['reference_speaker'] in ('p308', 'p317')] for name, item in audit.items()},
        'metric_identity': metric_identity, 'source_sha256': hashes, 'frozen_helper_sha256': HELPER_SHA,
        'script_sha256': base.digest(Path(__file__)),
        'limitations': ['All90 pairs retained; CER is corpus sum(errors)/sum(chars), SVA cosine>0.25, DNSMOS headline mos_ovr.',
                       'Only10 source clusters and fixed targets; repeated developmental comparisons and multiple metrics are exploratory, without multiplicity adjustment.',
                       'The historical four baselines lack per-pair q950 tensor receipts; only their frozen data/seed/batch1/code protocol is verifiable.',
                       'The2000,600 and new endpoint share one training trajectory; this does not estimate variation across training seeds.',
                       'The teacher/historical models and original normalization statistics saw these development utterances; new development updates exclude all52.',
                       'Stopping and selecting on this development set does not replace an independent final unseen assessment or permit direct paper-score equivalence.'],
    }
    for path, expected in hashes.items():
        if base.digest(path) != expected:
            raise RuntimeError('Input changed during continuation comparison: ' + path)
    output.mkdir(exist_ok=False)  # 验真全部完成后才创建新报告，不覆盖旧目录。
    lines = ['# ACDD续训终点的开发集比较', '', f'我比较实际保存的{step}步与2000步，主要观察继续训练后的变化。600步、初始化学生、教师和另外两个历史学生仍完整保留。全部90对均计入，区间按10条源录音整组重抽样。', '']
    for reference, metrics in comparisons.items():
        prefix = '主要对照：' if reference == PARENT_LABEL else '补充对照：'
        lines += [f'## {prefix}候选减{reference}', '', '| 指标 | 参照 | 候选 | 差值 | 95%配对区间 |', '|---|---:|---:|---:|---|']
        for name in (*MAIN_METRICS, *(m for m in base.METRICS if m not in MAIN_METRICS)):
            item = metrics[name]
            low, high = item['cluster_bootstrap_95_percentile_ci']
            lines.append(f"| {name} | {item['reference']:.4f} | {item['candidate']:.4f} | {item['delta']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
        lines.append('')
    decision = 'UTMOS相对2000步的区间全低于零，按预先安排暂停下一段，先做针对性诊断。' if utmos_high < 0 else 'UTMOS的预定暂停条件未触发；这不构成自动续训许可，还需一起查看其他指标。'
    lines += [decision, '', 'CER负差表示改善，其余主指标正差表示改善。停止原因与评分分别记录，区间跨零不能说明性能相同。2000、600和本次候选核对了全部实际q950张量哈希；原四基线没有保存这项证据，因此我只核验其输入、种子和固定代码协议。', '', '这里有10条源语音和固定目标参考；重复开发比较、单条训练轨迹与多项指标都限制了结论。完整转写及逐源/目标分解保留在JSON中，没有删除p308或p317。这些结果不能替代正式unseen评估。', '']
    with (output / 'comparison_CN.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    with (output / 'comparison.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(output), 'status': result['status'], 'primary_reference': PARENT_LABEL,
                      'primary_comparison': comparisons[PARENT_LABEL], 'protocol_review': result['protocol_review']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':  # 导入CPU测试不会自动读取结果或启动训练、推理、评分。
    main()
