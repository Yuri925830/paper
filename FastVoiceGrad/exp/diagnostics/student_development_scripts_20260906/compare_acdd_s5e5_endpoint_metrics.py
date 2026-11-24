"""只读比较实际3327步学生、父3227步和冻结教师的90对客观指标。"""

import hashlib  # 核对原始文件字节，不能用同名模型替换已声明候选。
import importlib.util  # 复用已冻结的读取器，不改旧模块的系统列表或其他全局变量。
import json  # 仅在全新目录新增小型比较结果。
import re  # 要求学生逐对q950身份为完整SHA，不能填unknown。
import sys  # 本地Windows只能阅读和编辑，统计限定Matpool CPU。
from pathlib import Path  # 全部候选和输出路径在此脚本中预先固定。
from types import SimpleNamespace  # 适配冻结loader的显式候选参数。

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Run this CPU-only comparison on the Matpool rental')
HELPER = ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development_candidate.py'
HELPER_SHA = '2c08b3700f405dc9deb85e6eecc5ff63309642e15113012242f81580704078e6'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Frozen generic candidate validator changed')
spec = importlib.util.spec_from_file_location('_s5e5_endpoint_generic', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)  # 导入同时执行原租用机身份保护，不启动任何模型。
base, np, DEV = helper.base, helper.np, helper.base.DEV
RUN = DEV / 'acdd_s5e5_staged120_from3227_v1'
PAIRED = DEV / 'acdd_lr_paired120_from3107_v1'
BASELINES = DEV / 'evaluation_baselines_v1'
OUTPUT = DEV / 'comparison_acdd_s5e5_3327_v1'
PROTOCOL = DEV / 'ACDD_S5E5_ENDPOINT_METRICS_PROTOCOL_CN.md'
PROTOCOL_SHA = '2876f2fe1893c05d6ba868f46869c6e88ff5da207ff396b1e3fc90747542dda3'
RUN_SHA = '32651a26cdf7074a9e8ca30c3cdd099697541fa4b261df757d2866f9e0622aba'
PARENT_SHA = '25c6e14693a940d94e18a85f50e8e3c08bbfbf1e15f53580117662a0f86ad909'
FULL_SHA = '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556'
PARENT_FULL_SHA = 'f98b0763ec5a2edbd23d5d493e9dac549f85cf40cfac71e79f0b462f6fc6beb0'
TEACHER = 'teacher500_k30'  # 原教师30步，不能替换成DDIM30或单步S0。
STUDENTS = {  # 当前比较仅接受这两个实际独立学生快照。
    'candidate': ('student_acdd_s5e5_s3327', RUN / 'student_step_003327.pt',
        '7bb68a0d053335a87a2c3c42684b864e27bf21a9f19d6f8d1b5bc8152955ca60', RUN / 'run_contract.json', RUN_SHA, 3327),
    'parent': ('student_acdd_lr_half_s3227', PAIRED / 'primary_s1e4/student_step_003227.pt',
        '88da1f14234c9dcb164cdd249d47862b9ba105f1b7a1a858b4a18f9fa921b1dc', PAIRED / 'run_contract.json', PARENT_SHA, 3227),
}
FROZEN = {  # 推理之前已确定的协议、实际停止收据及旧教师身份。
    PROTOCOL: PROTOCOL_SHA,
    RUN / 'run_contract.json': '787759339c886b2918bdda62abcc6818adf3cb410f923aca38c80bbd4b99b071',
    RUN / 'completion.json': '68ee1cf0e680f3bf665be6dcf46e2aa75c3684262c2da3765ae2c7beebf931b2',
    BASELINES / 'run_identity.json': '3e318582d9fc30b570d6a9bb7a93bf8715b68198acd643e7fa45a5c9d521bf96',
    BASELINES / TEACHER / 'inference_complete.json': '11dc3e16534d122bf83364be345cf3e2cc21f0742ebef51d94a2c673ba9b673b',
}


def verify_files(files, hashes):
    # 所有权重仅分块读取计算SHA；不反序列化、不调用训练或评分模型。
    for raw, expected in files.items():
        path = Path(raw)
        if not path.is_absolute() or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('Expected an absolute path and full SHA identity')
        if str(path) in hashes and hashes[str(path)] != expected:
            raise ValueError('Conflicting identities for the same dependency: ' + str(path))
        if base.digest(path) != expected:
            raise ValueError('Frozen file changed: ' + str(path))
        hashes[str(path)] = expected  # 最后再读一次确认统计期间没有文件被覆盖。


def validate_training(candidate, parent, receipt, parent_receipt):
    # 内容SHA与文件SHA分别检查；真实训练设置不能由报告口头推断。
    if helper.canonical_sha(candidate) != RUN_SHA or helper.canonical_sha(parent) != PARENT_SHA:
        raise ValueError('Unexpected training contract identity')
    lineage = {'original_step0_checkpoint_sha256': 'ec504a627350c102cab1647befdc74234969b616238d2cd07d5350ef0c322162',
        'parent_checkpoint': str(PAIRED / 'primary_s1e4/final_full.pt'), 'parent_checkpoint_sha256': PARENT_FULL_SHA,
        'parent_contract_sha256': PARENT_SHA, 'start_after_global_step': 3227}
    if candidate.get('kind') != 'acdd_s5e5_staged120_development' or candidate.get('lineage') != lineage:
        raise ValueError('Candidate must descend from the actual complete3227 parent')
    if candidate.get('effective_optimizer_lrs') != {'student': .00005, 'discriminator': .0002}:
        raise ValueError('Candidate effective optimizer LR identity changed')
    for key in ('configuration', 'training_indices', 'excluded_train_indices', 'reconstruction_dev_indices',
                'fixed_development_inputs', 'fixed_development_source_noise_sha256', 'original_step0_guard_anchor'):
        if candidate.get(key) != parent.get(key) or key not in candidate:
            raise ValueError('Held-fixed training/data field changed: ' + key)
    if (receipt.get('status'), receipt.get('completed_global_step'), receipt.get('new_updates'),
            receipt.get('contract_sha256'), receipt.get('partial_batch_may_have_updated_D')) != ('stopped', 3327, 100, RUN_SHA, False):
        raise ValueError('Only the actual complete100-update stopped endpoint is accepted')
    if set(receipt.get('trend', {}).get('stop_reasons', [])) != {'development_three_checks_rising', 'moving_total_three_windows_rising'}:
        raise ValueError('The selected endpoint must match its recorded strict stops')
    if receipt.get('paired_updates') != 20 or receipt.get('unpaired_updates') != 80 or receipt.get('stage1_gate', {}).get('passed') is not True:
        raise ValueError('Do not call the additional80 updates a paired training control')
    # full与S必须均已原子保存、冻结；半批或仅口头宣布成功的终点不能进入比较。
    for role, item, full_sha, full_path in (
            ('candidate', receipt, FULL_SHA, RUN / 'final_full.pt'),
            ('parent', parent_receipt, PARENT_FULL_SHA, PAIRED / 'primary_s1e4/final_full.pt')):
        _, student_path, sha, _, contract_sha, step = STUDENTS[role]
        if item.get('contract_sha256') != contract_sha or item.get('last_full_saved_global_step') != step or item.get('partial_batch_may_have_updated_D') is not False:
            raise ValueError('Incomplete resume-state receipt: ' + role)
        for key, path, digest in (('student_snapshot', student_path, sha), ('full_snapshot', full_path, full_sha)):
            expected = {'path': str(path), 'sha256': digest, 'global_step': step, 'committed': True, 'frozen_readonly': True}
            if item.get(key) != expected:
                raise ValueError('Frozen snapshot identity changed: ' + role + '/' + key)


def validate_noise_and_recipe(candidate, parent, teacher, identities):
    # 新学生与父学生有实际张量哈希；教师只保留配方证据，不能补造旧哈希。
    values = candidate.get('x950_sha256_per_pair', {})
    if set(values) != {str(i) for i in range(90)} or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in values.values()):
        raise ValueError('All90 actual candidate q950 hashes are required')
    if parent.get('x950_sha256_per_pair') != values:
        raise ValueError('Candidate and parent actual q950 inputs differ')
    if 'x950_sha256_per_pair' in teacher:
        raise ValueError('The frozen legacy teacher receipt must not be retroactively enriched')
    expected_seed = 'manifest per-pair seed, same initial q950 noise as frozen four-system baselines'
    for role in ('candidate', 'parent'):
        if identities[role].get('seed_rule') != expected_seed:
            raise ValueError('Student per-pair seed recipe changed')
    teacher_identity = identities['teacher']
    if teacher_identity.get('seed_rule') != 'per-pair manifest seed; inference batch1; same initial source noise for all systems' or teacher.get('inference_batch_size') != 1:
        raise ValueError('Teacher per-pair batch1 seed recipe changed')
    for identity in identities.values():
        if (identity.get('start_step'), identity.get('bnf_alignment'), identity.get('bnf_downsampling')) != (950, 'legacy_length_linear', 'legacy_decimate'):
            raise ValueError('The frozen source interface changed')
    # 候选loader已核对整张pairs表中的源、参考、文本、种子；此处再核对公共代码及特征文件身份。
    common = set.intersection(*(set(value['inputs']) for value in identities.values()))
    required = {str(ROOT / 'voicegrad/diffusion.py'), str(ROOT / 'data/stats/mel_mean.npy'), str(ROOT / 'data/stats/mel_std.npy')}
    if not required <= common:
        raise ValueError('Missing shared noising/statistics source evidence')
    if any(len({identity['inputs'][path] for identity in identities.values()}) != 1 for path in common):
        raise ValueError('Shared source/data/recipe file identities differ')
    return {'candidate_parent_actual_q950_equal': True, 'actual_q950_sha256_per_pair': values,
        'teacher_actual_q950_hashes_available': False, 'teacher_same_pairs_seed_and_frozen_recipe_verified': True,
        'teacher_tensor_equality_proven': False, 'shared_recipe_input_count': len(common)}


def compare(systems, template):
    # 输入必须保留原90行及10个源簇，每簇9个转换目标；不按指标删除语音。
    if [int(row['pair_id']) for row in template] != list(range(90)):
        raise ValueError('Keep the original ordered90 pairs')
    clusters = sorted({(r['source_speaker'], r['source_utt']) for r in template})
    if len(clusters) != 10 or len({c[0] for c in clusters}) != 10 or any(sum((r['source_speaker'], r['source_utt']) == c for r in template) != 9 for c in clusters):
        raise ValueError('Expected10 distinct source clusters with9 targets each')
    expected_labels = {STUDENTS['candidate'][0], STUDENTS['parent'][0], TEACHER}
    if set(systems) != expected_labels:
        raise ValueError('Exactly candidate, parent and teacher are required')
    weights = base.bootstrap_weights(10, 10000, 2026090617)  # 两条比较及八指标完全共享这一次抽样。
    points, draws, absolute, per_source = {}, {}, {}, {}
    for label, rows in systems.items():
        if [(int(r['pair_id']), r['source_speaker'], r['source_utt'], r['reference_speaker']) for r in rows] != [(int(r['pair_id']), r['source_speaker'], r['source_utt'], r['reference_speaker']) for r in template]:
            raise ValueError('System rows do not preserve the complete source/target pairing')
        sums = base.cluster_sums(rows, clusters)  # CER先累计整数分子分母；其余分数按相同配对平均。
        points[label], draws[label] = base.values_from_sums(sums.sum(axis=0)), base.values_from_sums(weights @ sums)
        intervals = base.confidence_interval(draws[label])
        absolute[label] = {name: {'estimate': float(points[label][i]), 'cluster_bootstrap_95_percentile_ci': intervals[i].tolist()} for i, name in enumerate(base.METRICS)}
        absolute[label].update(CER_total_errors=sum(r['char_errors'] for r in rows), CER_total_reference_chars=sum(r['reference_chars'] for r in rows))
        per_source[label] = [{'source_speaker': c[0], 'source_utt': c[1], 'pairs': 9,
            'CER_total_errors': int(sums[i, 0]), 'CER_total_reference_chars': int(sums[i, 1]),
            **dict(zip(base.METRICS, base.values_from_sums(sums[i]).tolist()))} for i, c in enumerate(clusters)]
    candidate = STUDENTS['candidate'][0]
    comparisons, source_deltas = {}, {}
    for reference in (STUDENTS['parent'][0], TEACHER):
        delta = points[candidate] - points[reference]
        ci = base.confidence_interval(draws[candidate] - draws[reference])  # 同一次源簇抽样后相减，不能独立抽两模型。
        comparisons[reference] = {name: {'reference': float(points[reference][i]), 'candidate': float(points[candidate][i]),
            'delta': float(delta[i]), 'cluster_bootstrap_95_percentile_ci': ci[i].tolist(),
            'favorable_delta_sign': 'negative' if name == 'CER_corpus_percent' else 'positive',
            'unit': 'percentage_points' if name in ('CER_corpus_percent', 'SVA_percent') else 'score'} for i, name in enumerate(base.METRICS)}
        source_deltas[reference] = [{'source_speaker': a['source_speaker'], 'source_utt': a['source_utt'],
            'CER_error_count_delta': a['CER_total_errors'] - b['CER_total_errors'],
            **{name: a[name] - b[name] for name in base.METRICS}} for a, b in zip(per_source[candidate], per_source[reference])]
    return absolute, comparisons, per_source, source_deltas, clusters, weights


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink() or OUTPUT.resolve().parent != DEV:
        raise ValueError('Comparison output must be the declared new direct child of DEV')
    hashes = {}
    verify_files(FROZEN, hashes)  # 先确认协议在推理前冻结，且终点身份没有变化。
    historical, template, development, metric_identity, originals = base.load_results(BASELINES)
    hashes.update(originals)  # 原冻结loader核验四基线，但本入口仅比较预声明教师这一列。
    systems = {TEACHER: historical[TEACHER]}
    contracts, identities, reports = {}, {}, {}
    for role, (label, checkpoint, sha, contract_path, contract_sha, step) in STUDENTS.items():
        args = SimpleNamespace(candidate_root=DEV / ('evaluation_candidate_' + label), label=label,
            checkpoint_sha256=sha, training_contract_sha256=contract_sha, global_step=step, epoch=0)
        rows, contract, identity, report = helper.load_candidate(args, template, development, metric_identity, hashes)
        if identity['systems'][label]['checkpoint'] != str(checkpoint) or identity['candidate_training_contract'] != str(contract_path):
            raise ValueError('Checkpoint path or shared-root training contract differs: ' + role)
        systems[label], contracts[role], identities[role], reports[role] = rows, contract, identity, report
    receipt = base.read_json(RUN / 'completion.json', hashes)
    parent_receipt = base.read_json(PAIRED / 'primary_s1e4/arm_completion.json', hashes)
    validate_training(contracts['candidate'], contracts['parent'], receipt, parent_receipt)
    identities['teacher'] = base.read_json(BASELINES / 'run_identity.json', hashes)
    reports['teacher'] = base.read_json(BASELINES / TEACHER / 'inference_complete.json', hashes)
    noise = validate_noise_and_recipe(reports['candidate'], reports['parent'], reports['teacher'], identities)
    for identity in identities.values():
        verify_files(identity['inputs'], hashes)  # 对实际模型、BNF/mel、扩散代码和原推理源码逐文件验真。
    for contract in contracts.values():
        verify_files(contract['file_sha256'], hashes)  # 训练源码及父谱系保留原SHA，不重签任何旧契约。
    verify_files({RUN / 'final_full.pt': FULL_SHA, PAIRED / 'primary_s1e4/final_full.pt': PARENT_FULL_SHA}, hashes)
    verify_files({asset['path']: asset['sha256'] for asset in metric_identity['assets'].values()}, hashes)
    verify_files(metric_identity['auxiliary_source_sha256'], hashes)  # 评分器权重及其源码也只读核对，不加载模型。
    absolute, differences, per_source, source_deltas, clusters, weights = compare(systems, template)
    result = {'status': 'complete_s5e5_endpoint_development_comparison_only', 'candidate_label': STUDENTS['candidate'][0],
        'candidate_global_step': 3327, 'candidate_training_contract_sha256': RUN_SHA, 'training_completion': receipt,
        'systems': absolute, 'candidate_minus_references': differences, 'per_source': per_source,
        'per_source_candidate_minus_references': source_deltas, 'all90_rows': systems,
        'metric_order': list(base.METRICS), 'n_pairs_per_system': 90, 'n_source_clusters': 10,
        'input_noise_evidence': noise, 'inference_identities': identities, 'metric_identity': metric_identity,
        'bootstrap': {'method': 'paired source-utterance cluster percentile bootstrap', 'repetitions': 10000,
            'seed': 2026090617, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__, 'clusters': [list(c) for c in clusters],
            'targets_per_source': 9, 'confidence_level': .95, 'multiple_comparison_adjustment': None,
            'weight_matrix_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()},
        'definitions': {'CER': '100*sum(char_errors)/sum(reference_chars), including each bootstrap draw',
            'SVA': '100*mean(cosine>0.25)', 'DNSMOS_headline': 'mos_ovr'},
        'source_sha256': hashes, 'frozen_generic_helper_sha256': HELPER_SHA, 'script_sha256': base.digest(Path(__file__)),
        'limitations': ['All90 development pairs retained; this is not formal8100 or a comparison with paper scores.',
            'Candidate3327 has100 additional updates after parent3227, with only20 updates paired to the old3247 control.',
            'The final80 updates have no contemporaneous matched-length control; this is not a100-step paired LR trial.',
            'Development data influenced LR and stopping selection; these are not independent final generalization estimates.',
            'Only10 source clusters, fixed target references; both contrasts and all metric intervals are exploratory and unadjusted.',
            'Historical teacher and normalization statistics saw development utterances; new student updates excluded52 utterances.',
            'Teacher legacy receipt lacks actual q950 hashes; same pair/seed/frozen recipe does not prove identical initial tensors.',
            'No inference, scoring, optimizer update or checkpoint write is performed.']}
    verify_files(FROZEN, hashes)  # 原读取器会登记新读到的SHA；发布前重钉协议和终点固定锚点。
    verify_files(hashes.copy(), {})  # 所有来源再次核验后才发布；中途变化就拒绝生成成功报告。
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT / 'comparison.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')  # 新文件保留全部270条记录，旧评分和模型文件不变。
    print(json.dumps({'output': str(OUTPUT), 'status': result['status'], 'candidate_minus_references': differences}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()  # CPU测试导入只访问冻结helper，不运行比较、更不触发GPU。
