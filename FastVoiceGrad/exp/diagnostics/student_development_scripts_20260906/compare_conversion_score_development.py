"""只读比较已完成的 CV 开发候选、同一步 ACDD 控制、父600和原四基线。"""

import argparse  # 调用者必须提供真实候选身份清单，不从目录名猜模型。
import hashlib  # 输入、冻结代码和新报告均记录实际字节身份。
import importlib.util  # 复用已审核的读取和配对统计函数，不导入训练模型。
import json  # 小型选择清单和新结果都使用可核验的JSON。
import re  # 拒绝含路径字符的标签和不完整哈希。
import sys  # 在导入项目代码前拒绝本地运行。
from pathlib import Path  # 所有计算限定Matpool开发区。
from types import SimpleNamespace  # 向冻结读取器传递完整显式身份。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一允许执行项目计算的租用机根目录。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地Windows只编辑文件。
    raise RuntimeError('Run this CPU comparison only on the Matpool rental.')
sys.path.insert(0, str(ROOT))  # 只加载主机身份保护。
from voicegrad.runtime_guard import assert_rented_server  # 不建立GPU或运行时训练锁。
assert_rented_server(ROOT)  # 数值汇总也必须在授权机器执行。
HELPER = ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development_candidate.py'  # 已冻结的候选校验器。
HELPER_SHA = '2c08b3700f405dc9deb85e6eecc5ff63309642e15113012242f81580704078e6'  # 保持历史评价口径。
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:  # 导入之前检查代码字节。
    raise RuntimeError('Frozen candidate helper changed.')
module_spec = importlib.util.spec_from_file_location('_cv_comparison_helper', HELPER)  # 私有模块不会覆盖旧工具。
helper = importlib.util.module_from_spec(module_spec)  # 原main保护使导入不读取实际结果。
module_spec.loader.exec_module(helper)  # 只复用只读验证和CPU统计。
base, np = helper.base, helper.np  # CER分母、SVA阈值和bootstrap与旧报告相同。
DEV = base.DEV  # 固定开发目录，不接触正式8100对或其他归档。
PARENT_LABEL = 'student_acdd_clean_s600'  # 两个续训分支共同的600步父模型。
PARENT_CONTRACT_SHA = 'f9472aea9332e8c261e4045ceac6630efe8f9ba012f135d5b4aba45216b32c13'  # 父训练契约canonical。
PARENT_FULL_SHA = '70dc5ed188e0eaeccc4c6b8fd80ed79595605b3af5569bafe87264645c54cf43'  # 父完整S/D/Adam/RNG状态。
CONTROL_CONTRACT_SHA = 'f7d351f786c9cb7f4583327a8ff7a4b6d2c551c40bdbf36ce9e54dbe1fad933b'  # 原ACDD600→2000轨迹。
PARENT = {  # 父600的推理只读学生快照，训练恢复来源仍由lineage指向full。
    'label': PARENT_LABEL, 'candidate_root': str(DEV / ('evaluation_candidate_' + PARENT_LABEL)),
    'checkpoint': str(DEV / 'acdd_clean_600_v1/student_step_000600.pt'),
    'checkpoint_sha256': 'b3eedcddc465b8cc79756f8aedcfc126eef5a807c35334a712b74c6691562882',
    'training_contract_sha256': PARENT_CONTRACT_SHA, 'global_step': 600, 'epoch': 0,
}


def validate_entry(entry):  # 一个候选必须写出全部身份，不自动填未知SHA。
    required = {'label', 'candidate_root', 'checkpoint', 'checkpoint_sha256', 'training_contract_sha256', 'global_step', 'epoch'}
    if not isinstance(entry, dict) or set(entry) != required:  # 拒绝拼错字段被静默忽略。
        raise ValueError('Each model entry must contain exactly the documented identity fields.')
    if not re.fullmatch(r'student_[a-z0-9_]{1,80}', entry['label']) or entry['label'] in base.SYSTEMS:
        raise ValueError('Expected a separate student label.')  # 不能覆盖教师/S0身份。
    if type(entry['global_step']) is not int or type(entry['epoch']) is not int or entry['epoch'] != 0:
        raise ValueError('Use actual integer global_step and epoch zero.')  # 不拿计划终点代替实际保存步数。
    if any(not isinstance(entry[k], str) or not re.fullmatch(r'[0-9a-f]{64}', entry[k]) for k in ('checkpoint_sha256', 'training_contract_sha256')):
        raise ValueError('Full lowercase SHA-256 values are required.')  # 不接受前缀或占位符。
    for key in ('candidate_root', 'checkpoint'):  # 只读当前开发区内的明确路径。
        path = Path(entry[key])
        if not path.is_absolute() or DEV not in path.resolve().parents:
            raise ValueError('Candidate paths must resolve inside the fixed development directory.')
    return entry  # 返回原记录，既不改路径也不重命名模型。


def validate_selection(selection):  # 真实结果运行时的轻量身份清单。
    if not isinstance(selection, dict) or set(selection) != {'schema_version', 'candidate', 'matched_control', 'output_dir'} or selection['schema_version'] != 1:
        raise ValueError('Use the exact version-one selection schema.')
    candidate = validate_entry(selection['candidate'])  # 所有候选字段必须齐全。
    step = candidate['global_step']  # 后续只按这个已保存的实际游标找控制。
    if not 600 < step <= 1200 or candidate['label'] == PARENT_LABEL:
        raise ValueError('CV comparison accepts actual endpoints after600 and at most1200.')
    control = selection['matched_control']  # null明确表示未提供可比较的同一步完成评价。
    if control is not None:
        validate_entry(control)  # 同一步控制也必须由主线程提供全部已知身份。
        if control['global_step'] != step or control['label'] in (candidate['label'], PARENT_LABEL):
            raise ValueError('The ACDD control must have the identical actual update count and a distinct label.')
        expected = DEV / f'acdd_clean_resume_s2000_v1/full_step_{step:06d}.pt'  # 唯一合法控制轨迹。
        if Path(control['checkpoint']).resolve() != expected.resolve() or control['training_contract_sha256'] != CONTROL_CONTRACT_SHA:
            raise ValueError('Use the original ACDD archive at precisely the CV saved step.')
    output = Path(selection['output_dir']).resolve()  # 仅创建开发区下全新的直接子目录。
    if output.parent != DEV or output.exists():
        raise ValueError('The comparison output must be a new direct child of the development directory.')
    return candidate, control, output  # 未知控制绝不自动训练、生成或评分。


def validate_lineage(parent, candidate, step, control=None):  # 同步数之外，还要检查确实恢复同一个父状态。
    if candidate.get('kind') != 'conversion_score_matched_development' or candidate.get('stop_global_step') != 1200:
        raise ValueError('A one-step smoke is not a scored CV performance candidate.')
    if candidate.get('method') != 'reconstruction adv +2 reconstruction FM +45 conversion clean_x0 sqrt_alpha_bar score; oneD then oneS update' or candidate.get('permutation_seed') != 991000:
        raise ValueError('Candidate is not the declared fixed-BNF conversion-score experiment.')
    if candidate.get('seed') != parent.get('seed') or candidate.get('seed') != 1234:
        raise ValueError('The common parent seed changed.')
    branches = [candidate] + ([control] if control is not None else [])  # 两分支分别核查来源。
    for branch in branches:
        lineage = branch.get('lineage', {})
        expected = {'parent_contract_sha256': PARENT_CONTRACT_SHA, 'parent_full_checkpoint_sha256': PARENT_FULL_SHA, 'start_after_global_step': 600}
        if any(lineage.get(key) != value for key, value in expected.items()):
            raise ValueError('The branch does not restore the declared common full600 state.')
        for key in ('configuration', 'training_indices', 'excluded_train_indices', 'reconstruction_dev_indices', 'fixed_development_inputs', 'fixed_development_source_noise_sha256'):
            if key not in parent or branch.get(key) != parent[key]:  # 不允许在新目标之外偷偷换配置或数据。
                raise ValueError('A held-fixed parent field changed: ' + key)
    cv_plan = candidate.get('batch_plan', {})  # 按绝对步数重放的原训练顺序。
    if cv_plan != {'first_step': 601, 'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((step-1)/1229)', 'crop_seed': '1234+100000+step', 'noise_seed': '1234+200000+step'}:
        raise ValueError('CV no longer declares the reviewed original batch/crop/noise replay.')
    if control is not None:  # 缺控制时不能写“等步数对照已通过”。
        if control.get('kind') != 'acdd_development_continuation' or not step <= control.get('stop_global_step', -1):
            raise ValueError('The control contract cannot supply this actual saved step.')
        if control.get('batch_plan') != {'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((global_step-1)/1229)', 'crop_seed': '1234+100000+global_step', 'noise_seed': '1234+200000+global_step', 'first_resumed_global_step': 601}:
            raise ValueError('The control batch/crop/noise replay changed.')


def compare_rows(reference, candidate, clusters, weights):  # 每次整组抽样同时用于两模型。
    before, after = base.cluster_sums(reference, clusters), base.cluster_sums(candidate, clusters)
    p0, p1 = base.values_from_sums(before.sum(axis=0)), base.values_from_sums(after.sum(axis=0))  # CER由总错误除以总字符。
    interval = base.confidence_interval(base.values_from_sums(weights @ after) - base.values_from_sums(weights @ before))
    return {key: {'reference': float(p0[i]), 'candidate': float(p1[i]), 'delta': float(p1[i] - p0[i]), 'cluster_bootstrap_95_percentile_ci': interval[i].tolist(), 'unit': 'percentage_points' if key in ('CER_corpus_percent', 'SVA_percent') else 'score', 'favorable_delta_sign': 'negative' if key == 'CER_corpus_percent' else 'positive'} for i, key in enumerate(base.METRICS)}


def same_noise(left, right):  # 实际q950张量的逐对哈希必须全90对一致。
    a, b = left.get('x950_sha256_per_pair', {}), right.get('x950_sha256_per_pair', {})
    if set(a) != {str(i) for i in range(90)} or a != b or any(not re.fullmatch(r'[0-9a-f]{64}', value) for value in a.values()):
        raise ValueError('The candidates do not share all90 actual source-q950 hashes.')


def main():  # 只在四项真实评分完成后运行，不等待GPU、不启动评分器。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)  # 独立JSON文件固定本次候选与控制。
    args = parser.parse_args()
    selection_path = args.selection.resolve()  # 选择清单本身也进入来源哈希。
    if ROOT / 'exp/diagnostics' not in selection_path.parents:
        raise ValueError('Keep the selection manifest inside the rental diagnostics directory.')
    hashes = {}
    selection = base.read_json(selection_path, hashes)  # 不接受内嵌未知模型身份。
    candidate_spec, control_spec, output = validate_selection(selection)
    systems, template, development, metric_identity, base_hashes = base.load_results(DEV / 'evaluation_baselines_v1')
    hashes.update(base_hashes)  # 原四基线、全部源文件及评分器身份仍使用完整验证。
    contracts, identities, reports = {}, {}, {}
    for entry in [PARENT, candidate_spec] + ([control_spec] if control_spec else []):
        namespace = SimpleNamespace(**{k: v for k, v in entry.items() if k != 'checkpoint'})
        namespace.candidate_root = Path(namespace.candidate_root)
        rows, contract, identity, report = helper.load_candidate(namespace, template, development, metric_identity, hashes)
        label = entry['label']
        if Path(identity['systems'][label]['checkpoint']).resolve() != Path(entry['checkpoint']).resolve():
            raise ValueError('The inference checkpoint path differs from the explicit selection.')
        systems[label], contracts[label], identities[label], reports[label] = rows, contract, identity, report
    label, step = candidate_spec['label'], candidate_spec['global_step']  # 实际保存步数仍由冻结读取器核权重及推理收据。
    control_label = control_spec['label'] if control_spec else None
    validate_lineage(contracts[PARENT_LABEL], contracts[label], step, contracts.get(control_label))
    same_noise(reports[PARENT_LABEL], reports[label])  # 父600和CV输入实际哈希完全一致。
    if control_label:
        same_noise(reports[label], reports[control_label])  # 控制对照不能仅凭相同seed文字认定配对。
    completion_path = Path(identities[label]['candidate_training_contract']).parent / 'completion.json'
    completion = base.read_json(completion_path, hashes)  # 防止运行中或只完成601smoke的模型进入性能报告。
    accepted = {'matched_global_step_budget_complete', 'requested_stop', 'time_budget_complete', 'original_step0_development_regression_guard', 'disk_space_guard', 'disk_space_guard_last_full_preserved'}
    if completion.get('status') not in accepted or completion.get('run_contract_sha256') != candidate_spec['training_contract_sha256'] or completion.get('last_saved_global_step') != step or completion.get('global_step', -1) < step:
        raise ValueError('CV completion does not identify this actual last saved checkpoint.')
    clusters = sorted({(r['source_speaker'], r['source_utt']) for r in template})
    if len(clusters) != 10 or any(sum((r['source_speaker'], r['source_utt']) == c for r in template) != 9 for c in clusters):
        raise ValueError('Expected exactly10 source utterances with all9 targets each.')
    weights = base.bootstrap_weights(10, 10000, 2026090617)  # 与所有旧汇总完全相同的配对抽样。
    references = [PARENT_LABEL, *base.SYSTEMS] + ([control_label] if control_label else [])
    comparisons = {reference: compare_rows(systems[reference], systems[label], clusters, weights) for reference in references}
    archive = DEV / f'acdd_clean_resume_s2000_v1/full_step_{step:06d}.pt'
    control_status = 'verified_complete_same_step_control' if control_label else ('same_step_archive_missing' if not archive.is_file() else 'same_step_complete_evaluation_not_supplied')
    audit = base.cer_concentration_audit(systems, clusters)  # 继续保留每源/每目标，包括p308、p317。
    result = {
        'status': 'complete_cv_development_comparison_only', 'selection': selection, 'candidate_minus_references': comparisons,
        'matched_control_status': control_status, 'expected_control_archive': str(archive), 'same_step_conclusion_available': bool(control_label),
        'actual_saved_step': step, 'training_completion': completion, 'n_pairs_per_system': 90, 'n_source_clusters': 10,
        'bootstrap': {'method': 'paired source-utterance cluster percentile bootstrap', 'repetitions': 10000, 'seed': 2026090617, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__, 'weight_matrix_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()},
        'verified_q950_hash_comparisons': [PARENT_LABEL] + ([control_label] if control_label else []),
        'cer_concentration_audit': audit, 'target_p308_p317_breakdown': {name: [r for r in item['by_target'] if r['reference_speaker'] in ('p308', 'p317')] for name, item in audit.items()},
        'source_sha256': hashes, 'frozen_helper_sha256': HELPER_SHA, 'script_sha256': base.digest(Path(__file__)),
        'limitations': ['All90 pairs retained; CER is corpus sum(errors)/sum(chars), SVA cosine>0.25, DNSMOS headline mos_ovr.', 'Only10 source clusters and fixed targets; paired intervals are exploratory without multiple-comparison adjustment or multiple training seeds.', 'The original four baseline receipts lack q950 tensor hashes; their common input/seed/batch1/frozen code protocol is verified, without claiming an archived tensor-hash equality.', 'CV changes the teacher-score speaker condition while retaining fixed original BNF and reconstruction GAN; this is a separate related-paper diagnostic, not a strict2024 reproduction.', 'A missing same-step control forbids a matched-update objective comparison; no training or evaluation is automatically launched.', 'Development selection cannot replace independent final unseen evaluation or direct paper-score comparison.'],
    }
    for path, expected in hashes.items():  # 所有已读文件在发布之前必须仍保持相同字节。
        if base.digest(path) != expected:
            raise RuntimeError('Input changed during CV comparison: ' + path)
    output.mkdir(exist_ok=False)  # 所有真实收据齐全且核验通过后才创建轻量结果。
    with (output / 'comparison.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    lines = ['# CV学生的开发集配对比较', '', f'我评价的是实际完整保存的第{step}步模型。全部90对语音保持不变，区间按10条源录音整组重抽样。', '']
    if not control_label:
        lines += [f'同一步ACDD控制目前不能比较，记录状态为{control_status}。以下父600和S0、教师对照不能替代等步数的训练目标比较。', '']
    for reference, metrics in comparisons.items():
        lines += [f'## CV减{reference}', '', '| 指标 | 参照 | CV | 差值 | 95%配对区间 |', '|---|---:|---:|---:|---|']
        for key, item in metrics.items():
            low, high = item['cluster_bootstrap_95_percentile_ci']
            lines.append(f"| {key} | {item['reference']:.4f} | {item['candidate']:.4f} | {item['delta']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
        lines.append('')
    lines += ['CER负差表示识别错误减少，质量分数正差表示预测分数上升。CER按全部字符计数计算，SVA保持余弦严格大于0.25。训练停止原因与客观质量结果分别记录，不根据loss或GT mel保护线预设本次评分结论。', '', '父600、CV与已提供的同一步控制核验了逐对实际q950哈希。原四基线只归档了相同输入、seed和固定代码协议，没有补造缺失的q950哈希证明。开发语音来自原训练语句，教师曾见过；10源簇和单次训练不能代表全部泛化或训练随机性。此处不作正式unseen或论文分数结论。', '']
    with (output / 'comparison_CN.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))  # 新文档独立保存，不改任何冻结报告。
    print(json.dumps({'output': str(output), 'status': result['status'], 'matched_control_status': control_status, 'candidate_minus_references': comparisons}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':  # 测试导入不会读取候选或写真实结果。
    main()  # 不存在默认候选、自动训练或自动评分入口。
