"""把显式学生候选与冻结四基线逐对比较；只读取已完成的客观指标。"""

import argparse  # 候选目录、权重哈希、训练契约和步数均由调用者明确给出。
import csv  # 另存候选逐条结果，不改原指标CSV。
import hashlib  # 先核验冻结辅助代码，再导入纯统计函数。
import importlib.util  # 在私有模块中复用已审核的源簇bootstrap函数。
import json  # 保存候选比较及其完整来源身份。
import re  # 拒绝含路径字符或基线名称的候选标签。
import sys  # 在任何计算前限制运行主机。
from pathlib import Path  # 所有操作限定出租服务器开发区。

ROOT = Path('/root/FsatVoiceGrad')  # 当前唯一获准的权威项目目录。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地Windows只能阅读与编辑。
    raise RuntimeError('Run candidate summaries only on the Matpool rental.')  # 不在其他服务器执行。
sys.path.insert(0, str(ROOT))  # 导入主机指纹保护，不导入训练模型。
from voicegrad.runtime_guard import assert_rented_server  # 再核对机器身份。
assert_rented_server(ROOT)  # 此脚本没有GPU或评分器入口。
HELPER = ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development.py'  # 固定四基线汇总的已审核纯统计实现。
HELPER_SHA = '806475c23f75310df469bc102f1dbefa473a993f4f9df5e02c07c87fd071aca3'  # 原汇总已经生成，辅助代码不再修改。
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:  # 必须先核验再执行导入。
    raise RuntimeError('Frozen baseline summary helper changed; review before reuse.')  # 不靠文件名推断实现相同。
specification = importlib.util.spec_from_file_location('_frozen_development_summary', HELPER)  # 不覆盖其他模块命名空间。
base = importlib.util.module_from_spec(specification)  # 私有模块仅用于读取与统计函数。
specification.loader.exec_module(base)  # main保护使导入不运行原四基线汇总。
np = base.np  # 复用同一NumPy版本，所有数组计算均在CPU。


def canonical_sha(value):  # 原训练检查点绑定的是内容级JSON哈希。
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')  # 与原训练签名规则相同。
    return hashlib.sha256(payload).hexdigest()  # 不混用带缩进的文件字节哈希。


def validate_training_exclusion(contract, development, development_file_sha):  # 新候选必须真正排除开发语音的整句更新。
    if contract.get('formal_test_used_for_selection') is not False:  # 正式unseen不可用于挑选本次候选。
        raise ValueError('Candidate must explicitly declare no formal-test selection.')  # 缺字段也不推断为安全。
    if contract.get('excluded_train_indices') != development['excluded_train_indices']:  # 源、目标参考和自重建52句必须全部排除。
        raise ValueError('Candidate development exclusion differs from the frozen 52 utterances.')  # 不允许只排除其中的裁剪片段。
    if contract.get('training_indices') != development['retained_train_indices']:  # 保留39335句的原索引顺序也须一致。
        raise ValueError('Candidate training pool differs from the declared retained indices.')  # 不把其他划分的结果混入同条件比较。
    if contract.get('reconstruction_dev_indices') != development['reconstruction_dev_indices']:  # 候选选择时的固定32句也需要同身份。
        raise ValueError('Candidate reconstruction development identities changed.')  # 禁止候选之间重挑开发片段来源。
    # ACDD旧字段记录文件哈希，endpoint单独区分内容和文件哈希，两种均明确兼容。
    stored_file_sha = contract.get('development_contract_file_sha256', contract.get('development_contract_sha256'))  # 有新明确字段时优先使用。
    if stored_file_sha != development_file_sha:  # 两个协议都必须指向同一份真实开发契约文件。
        raise ValueError('Candidate did not use the frozen development contract file.')  # 不改变旧账本来绕过校验。


def load_candidate(args, template, development, baseline_metric_identity, hashes):  # 只有全90对四指标完成才接受候选。
    evaluation = args.candidate_root.resolve()  # 一个显式候选对应一个独立评价目录。
    directory = evaluation / args.label  # 其子目录与实际系统标签完全一致。
    identity_path, metric_identity_path = evaluation / 'run_identity.json', evaluation / 'metric_identity.json'  # 记录推理与评分身份。
    identity = base.read_json(identity_path, hashes)  # 不需要加载候选模型即可核对其原始凭据。
    metric_identity = base.read_json(metric_identity_path, hashes)  # 指标模型、源码、版本、批量等与基线一致。
    if identity.get('purpose') != 'development_candidate_not_formal_8100' or identity.get('pairs') != 90 or set(identity.get('systems', {})) != {args.label}:  # 拒绝混入正式评估或多个候选目录。
        raise ValueError('Expected one explicitly identified 90-pair development candidate.')  # 不由目录名猜测模型身份。
    if metric_identity != baseline_metric_identity:  # 全部评分口径和权重身份必须相同。
        raise ValueError('Candidate metric identity differs from the frozen baselines.')  # 不将新模型评分器结果直接相减。
    model = identity['systems'][args.label]  # 读取实际推理时保存的权重身份和游标。
    if (model.get('sha256'), model.get('global_step'), model.get('epoch'), model.get('steps'), model.get('state_key'), model.get('sampling')) != (args.checkpoint_sha256, args.global_step, args.epoch, 1, 'student', 'existing_student_t950_to_zero'):  # ACDD和endpoint均使用同单步接口。
        raise ValueError('Candidate checkpoint/step/sampling identity does not match explicit arguments.')  # 不能冒用同标签的其他权重。
    if (identity.get('bnf_alignment'), identity.get('bnf_downsampling'), identity.get('start_step'), identity.get('vocoder')) != ('legacy_length_linear', 'legacy_decimate', 950, 'same frozen custom step2500000'):  # 本候选对照仍保持已声明公共接口。
        raise ValueError('Candidate changed a held-fixed input/vocoder interface.')  # 新时轴实验需要另一份明确协议。
    for path in (base.DEV / 'pairs.csv', base.DEV / 'development_contract.json'):  # 候选和四基线使用同一个源、参考、种子模板。
        if identity['inputs'].get(str(path)) != hashes[str(path)]:  # 与刚核验的开发文件字节相连。
            raise ValueError(f'Candidate development input mismatch: {path}')  # 不允许新增候选偷偷换样本。
    checkpoint_path = Path(model['checkpoint']).resolve()  # 仅做文件哈希核验，不反序列化模型。
    if ROOT not in checkpoint_path.parents or base.digest(checkpoint_path) != args.checkpoint_sha256 or identity['inputs'].get(str(checkpoint_path)) != args.checkpoint_sha256:  # 权重真实字节和原推理登记须同时相符。
        raise ValueError('Candidate checkpoint bytes do not match the declared identity.')  # 文件名或步数不足以证明相同模型。
    hashes[str(checkpoint_path)] = args.checkpoint_sha256  # 发布汇总前复查权重未被覆盖。
    contract_path = Path(identity['candidate_training_contract']).resolve()  # 由实际推理身份提供原训练契约位置。
    if ROOT not in contract_path.parents:  # 不读取其他机器或工作区来源。
        raise ValueError('Candidate training contract is outside the authoritative project.')  # 限制来源范围。
    training_contract = base.read_json(contract_path, hashes)  # 只读训练设置与划分证据。
    if canonical_sha(training_contract) != args.training_contract_sha256 or identity['inputs'].get(str(contract_path)) != hashes[str(contract_path)]:  # 内容级SHA与实际文件SHA分别核对。
        raise ValueError('Candidate training contract does not match the declared identity.')  # 不后补新的训练来源。
    validate_training_exclusion(training_contract, development, hashes[str(base.DEV / 'development_contract.json')])  # 明确教师暴露与新学生排除的区别。
    report_path = directory / 'inference_complete.json'  # 候选90条波形必须已经完整生成。
    report = base.read_json(report_path, hashes)  # 记录实际权重、输出波形、噪声状态身份。
    actual = base.read_csv(directory / 'pairs.csv', hashes)  # 相同pair_id0..89，输出路径独立。
    if report.get('status') != 'development_candidate_only' or report.get('system') != args.label or report.get('spec') != model or report.get('saved_training_contract_sha256') != args.training_contract_sha256 or report.get('student_updates') != args.global_step:  # 检查点与训练契约链不能缺一环。
        raise ValueError('Candidate inference completion identity mismatch.')  # 半成品不能进入汇总。
    if report.get('run_identity_sha256') != hashes[str(identity_path)] or report.get('pairs_csv_sha256') != hashes[str(directory / 'pairs.csv')]:  # 完成凭据绑定当前run与实际配对表。
        raise ValueError('Candidate inference receipt hashes do not match.')  # 不使用改过的输出表。
    if [{k: v for k, v in row.items() if k != 'output_wav'} for row in actual] != [{k: v for k, v in row.items() if k != 'output_wav'} for row in template]:  # 唯一允许差异是各自生成wav路径。
        raise ValueError('Candidate changed source/reference/text/seed pairing.')  # 保持完全配对比较。
    if set(report.get('x950_sha256_per_pair', {})) != {str(i) for i in range(90)}:  # 候选driver须记录每对实际初始状态。
        raise ValueError('Candidate lacks complete per-pair noisy-start identities.')  # 不在此汇总中重算GPU噪声。
    for row in actual:  # 读取已评分波形的字节，不解码、不重新合成。
        path = Path(row['output_wav']).resolve()  # 波形必须属于这个候选。
        if directory not in path.parents or not path.is_file() or base.digest(path) != report['wav_sha256'].get(row['pair_id']):  # 查验生成和评分使用同一文件。
            raise ValueError(f'Candidate waveform identity mismatch: pair {row["pair_id"]}')  # 不能借用其他模型的输出。
        hashes[str(path)] = report['wav_sha256'][row['pair_id']]  # 汇总发布前再核对。
    summary_path = directory / 'metrics/development_summary_cer_dnsmos_sva_utmos.json'  # 四项完成才进行比较。
    complete = base.read_json(summary_path, hashes)  # 文件不存在时直接失败，不启动评分器补算。
    if complete.get('status') != 'development_only' or complete.get('system') != args.label or complete.get('evaluated_pairs') != 90 or complete.get('inference_complete_sha256') != hashes[str(report_path)] or complete.get('run_identity_sha256') != hashes[str(identity_path)] or complete.get('metric_identity_sha256') != hashes[str(metric_identity_path)]:  # 四阶段身份须一致。
        raise ValueError('Candidate complete-metrics receipt mismatch.')  # 不把部分结果当最终结果。
    tables = {}  # 每项指标保存全部90个固定pair。
    for metric, fields in base.FIELDS.items():  # 精确沿用四基线CSV字段定义。
        path = directory / f'metrics/{metric}.csv'  # 仅读取已生成指标文件。
        tables[metric] = base.read_csv(path, hashes, fields)  # 检查表头、顺序、行数。
        if complete['metric_csv_sha256'].get(metric) != hashes[str(path)]:  # 完成后改动的指标拒绝使用。
            raise ValueError(f'Candidate metric CSV hash mismatch: {metric}')  # 不修改旧完成收据。
    rows = base.merged_rows(args.label, actual, tables)  # 独立重算CER编辑距离/分母，检查SVA>0.25和各有限数。
    base.validate_driver_summary(complete, rows)  # 原主数必须能从逐条记录完整复算。
    return rows, training_contract, identity, report  # 后续仅对这些固定数值做CPU统计。


def make_comparisons(systems, candidate_label, clusters, repetitions, seed):  # 所有系统共享同一次10源簇抽样。
    weights = base.bootstrap_weights(len(clusters), repetitions, seed)  # 每次整组抽10条源，其9个目标一起带入。
    point, draws, absolute = {}, {}, {}  # 原主数、重采样分数、绝对区间分别保留。
    for label, rows in systems.items():  # CER分母始终从抽中语句的原字符数累加。
        sums = base.cluster_sums(rows, clusters)  # 将每源9目标的计数及各分数求和。
        point[label] = base.values_from_sums(sums.sum(axis=0))  # 完整90对主数，不删失败配对。
        draws[label] = base.values_from_sums(weights @ sums)  # 每次bootstrap仍先加分子分母再相除。
        interval = base.confidence_interval(draws[label])  # 绝对区间只作开发描述。
        absolute[label] = {metric: {'estimate': float(point[label][i]), 'cluster_bootstrap_95_percentile_ci': interval[i].tolist()} for i, metric in enumerate(base.METRICS)}  # 保留DNSMOS四列和连续余弦。
        absolute[label].update(CER_total_errors=sum(row['char_errors'] for row in rows), CER_total_reference_chars=sum(row['reference_chars'] for row in rows))  # CER整数计数便于核对。
    comparisons = {}  # 每条比较都明确为“候选减参照”。
    for reference in base.SYSTEMS:  # S0、教师、旧学生、repair2460四个冻结参照全部保留。
        delta = point[candidate_label] - point[reference]  # 同一配对的原始总分差。
        interval = base.confidence_interval(draws[candidate_label] - draws[reference])  # 先按共同重采样作差，再取分位数。
        comparisons[reference] = {'candidate_system': candidate_label, 'reference_system': reference, 'delta_definition': 'candidate minus reference', 'metrics': {metric: {'delta': float(delta[i]), 'cluster_bootstrap_95_percentile_ci': interval[i].tolist(), 'favorable_delta_sign': 'negative' if metric == 'CER_corpus_percent' else 'positive', 'unit': 'percentage_points' if metric in ('CER_corpus_percent', 'SVA_percent') else 'score'} for i, metric in enumerate(base.METRICS)}}  # 不以不同抽样制造虚假的独立比较。
    return absolute, comparisons, weights  # 返回数值供统一存档和简短中文表格。


def main():  # 完成结果统计，不触碰训练/推理锁或GPU。
    parser = argparse.ArgumentParser(description=__doc__)  # 缺少显式身份就拒绝运行。
    parser.add_argument('--candidate-root', type=Path, required=True)  # 例如evaluation_candidate_student_acdd_clean_s600。
    parser.add_argument('--label', required=True)  # 实际候选子目录的系统名。
    parser.add_argument('--checkpoint-sha256', required=True)  # 完整权重字节哈希，不接受前缀。
    parser.add_argument('--training-contract-sha256', required=True)  # 原训练契约canonical SHA。
    parser.add_argument('--global-step', type=int, required=True)  # 以实际更新数标识候选，不冒称整轮训练。
    parser.add_argument('--epoch', type=int, default=0)  # 短开发训练的epoch字段固定为0。
    parser.add_argument('--baseline-root', type=Path, default=base.DEV / 'evaluation_baselines_v1')  # 冻结四基线结果位置。
    parser.add_argument('--output-dir', type=Path)  # 默认候选评价目录内独立paired_summary_v1。
    args = parser.parse_args()  # 不提供改指标、删pair或重评分开关。
    args.candidate_root, args.baseline_root = args.candidate_root.resolve(), args.baseline_root.resolve()  # 使用真实绝对路径。
    output = args.output_dir.resolve() if args.output_dir else args.candidate_root / 'paired_summary_v1'  # 与四基线汇总分开。
    if base.DEV not in args.candidate_root.parents or base.DEV not in args.baseline_root.parents or args.candidate_root not in output.parents or output.exists():  # 所有目录限定开发区且输出必须全新。
        raise ValueError('Use independent development directories and a new output folder.')  # 拒绝覆盖冻结汇总。
    if not re.fullmatch(r'student_[a-z0-9_]{1,80}', args.label) or args.label in base.SYSTEMS or args.global_step < 0 or args.epoch < 0:  # 标签、步数和epoch须合理。
        raise ValueError('Expected a new explicit student candidate identity.')  # 教师DDIM对照另有专门标签，不能冒充学生。
    if any(not re.fullmatch(r'[0-9a-f]{64}', value) for value in (args.checkpoint_sha256, args.training_contract_sha256)):  # 两个哈希都要求完整小写十六进制。
        raise ValueError('Pass full lowercase SHA-256 identities, not prefixes.')  # 避免误认另一个断点。
    systems, template, development, metric_identity, hashes = base.load_results(args.baseline_root)  # 再核验全部冻结基线，不改它们。
    rows, training_contract, identity, report = load_candidate(args, template, development, metric_identity, hashes)  # 新候选四项指标都完整才返回。
    systems[args.label] = rows  # 只修改本次内存字典，不改基线模块常量或任何已有文件。
    clusters = sorted({(row['source_speaker'], row['source_utt']) for row in template})  # 与原汇总相同的源簇次序。
    if len(clusters) != 10 or any(sum((r['source_speaker'], r['source_utt']) == cluster for r in template) != 9 for cluster in clusters):  # 固定10源、每源9目标。
        raise ValueError('Expected exactly 10 source clusters with 9 targets each.')  # 保持bootstrap单位真实。
    absolute, comparisons, weights = make_comparisons(systems, args.label, clusters, 10000, 2026090617)  # 与冻结基线汇总同种子和次数。
    concentration = base.cer_concentration_audit(systems, clusters)  # 保留全部源时长、各target错误与原ASR文本。
    target_breakdown = {label: [row for row in audit['by_target'] if row['reference_speaker'] in ('p308', 'p317')] for label, audit in concentration.items()}  # 预先已定位的两个目标固定报告，不重挑更有利目标。
    result = {  # 单独声明本候选用途和训练暴露范围。
        'status': 'complete_development_candidate_summary_only', 'candidate_label': args.label, 'candidate_checkpoint_sha256': args.checkpoint_sha256, 'candidate_training_contract_sha256': args.training_contract_sha256, 'candidate_global_step': args.global_step,  # 明确实际候选身份。
        'systems': absolute, 'candidate_minus_baselines': comparisons, 'target_p308_p317_breakdown': target_breakdown, 'cer_concentration_audit': concentration,  # 完整分数、配对区间与目标失效排查。
        'metric_order': list(base.METRICS), 'n_pairs_per_system': 90, 'n_unique_source_utterances': 10, 'SVA_threshold': 0.25, 'CER_definition': '100 * sum(char_errors) / sum(reference_chars), including every bootstrap draw',  # 原口径不改。
        'bootstrap': {'method': 'paired source-utterance cluster percentile bootstrap', 'repetitions': 10000, 'seed': 2026090617, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__, 'clusters': [list(c) for c in clusters], 'targets_kept_together_per_source': 9, 'confidence_level': 0.95, 'weight_matrix_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()},  # 可与基线原权重矩阵比较。
        'exposure': {'baseline_models_saw_development_utterances_during_original_training': True, 'candidate_excluded_all_52_utterances_from_updates': True, 'frozen_normalization_statistics_include_these_utterances': True},  # 候选的独立性仅指其参数更新，不隐瞒教师与统计量暴露。
        'limitations': ['This is the unchanged 90-pair development set, not formal text-unseen 8100 or paper scores.', 'Only 10 source clusters; percentile intervals are exploratory and condition on the fixed reference-target set.', 'Four baseline contrasts and multiple metrics are descriptive, without multiplicity-adjusted significance claims.', 'Candidate selection uses this development set; final unseen generalization must be evaluated separately after selection.', 'All 90 pairs remain in the main scores; p308/p317 and posthoc target subsets are diagnostic breakdowns only.', 'Endpoint and ACDD candidates may have different declared training objectives; the paired evaluation holds data, inference interface, vocoder and scoring fixed, not the training objective.'],  # 不将开发改善冒称论文级泛化提升。
        'candidate_training_objective': training_contract.get('objective_id', training_contract.get('kind')), 'candidate_inference_identity': identity, 'candidate_x950_sha256_per_pair': report['x950_sha256_per_pair'],  # 保留新候选的实际训练类别与初始状态身份。
        'metric_identity': metric_identity, 'source_sha256': hashes, 'frozen_summary_helper_sha256': HELPER_SHA, 'script_sha256': base.digest(Path(__file__)),  # 汇总本身也留字节指纹。
    }  # 无重新识别、生成音频或模型更新。
    for path, expected in hashes.items():  # 结果、来源与checkpoint在汇总期间不应被任何任务覆盖。
        if base.digest(path) != expected:  # 输入变化即停止发布。
            raise RuntimeError(f'Input changed while summarizing candidate: {path}')  # 不用新文件解释旧指标。
    output.mkdir(parents=True, exist_ok=False)  # 所有检查完成后才写独立结果。
    with (output / 'summary.json').open('x', encoding='utf-8') as handle:  # 已有结果不会覆盖。
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 拒绝NaN写入。
        handle.write('\n')  # 保存完整文本末尾换行。
    with (output / 'candidate_paired_metrics.csv').open('x', encoding='utf-8', newline='') as handle:  # 新候选90行结果及原ASR转写。
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))  # 沿用冻结汇总的逐条字段。
        writer.writeheader()  # 方便研究人员按列复核。
        writer.writerows(rows)  # 顺序仍为pair_id0..89。
    lines = [f'# {args.label} 的固定开发集对照', '', f'我使用同一组90个转换配对，比较实际训练{args.global_step}步的候选和四个固定参照。新候选训练时排除了52条开发语音，教师及历史学生训练时见过这些语音。', '', '| 系统 | UTMOS | DNSMOS OVRL | CER（%） | SVA（%） |', '|---|---:|---:|---:|---:|']  # 报告明确候选步数和暴露区别。
    for label in (*base.SYSTEMS, args.label):  # 全部四参照与新候选都展示。
        value = absolute[label]  # 这些均为完整90对主数。
        lines.append(f"| {label} | {value['UTMOS']['estimate']:.4f} | {value['DNSMOS_OVRL']['estimate']:.4f} | {value['CER_corpus_percent']['estimate']:.4f} | {value['SVA_percent']['estimate']:.4f} |")  # 不隐藏退步指标。
    lines += ['', '## 候选减参照的配对差值', '', '括号内是按10条源语音整组重抽样得到的95%区间。CER差值单位为百分点，负值表示改善；UTMOS、DNSMOS正值表示改善。', '', '| 参照 | Δ UTMOS | Δ DNSMOS OVRL | Δ CER |', '|---|---:|---:|---:|']  # 解释差值方向与统计单位。
    for reference in ('teacher_initialized_s0_k1', 'teacher500_k30', 'repair_custom_step2460_k1', 'old_student_epoch100_k1'):  # 优先回答与S0、教师、既有修复的差异。
        cells = []  # 每格保留点差及上下界。
        for metric in ('UTMOS', 'DNSMOS_OVRL', 'CER_corpus_percent'):  # 同时展示三个主要质量与内容指标。
            item = comparisons[reference]['metrics'][metric]  # 同一簇抽样下候选减参照。
            low, high = item['cluster_bootstrap_95_percentile_ci']  # 未对多重比较作校正。
            cells.append(f"{item['delta']:+.4f} [{low:+.4f}, {high:+.4f}]")  # 保留正负号。
        lines.append(f"| {reference} | {' | '.join(cells)} |")  # 全部参照均列出，不挑有利比较。
    lines += ['', '## 两个固定目标的补充检查', '', '| 系统 | 目标 | CER（%） | SVA通过数 | 平均声纹余弦 |', '|---|---|---:|---:|---:|']  # 延续对p308/p317的失效排查。
    for label, group in target_breakdown.items():  # 所有系统相同两目标。
        for row in group:  # 每目标固定9条不同源语音。
            lines.append(f"| {label} | {row['reference_speaker']} | {row['CER_corpus_percent']:.4f} | {row['SVA_passes']}/9 | {row['SVA_mean_cosine']:.4f} |")  # 不以删掉困难目标后的指标替代主数。
    lines += ['', '我保留了完整90对的CER主数，计算方法是总字符错误数除以总参考字符数。p308、p317只用于检查之前发现的条件失效；没有因为它们较难就换目标或删样本。DNSMOS其余三列、全部ASR转写和逐源长度分解保存在summary.json。', '', '这里只涉及10条源语音，目标参考固定，区间是开发阶段的探索性统计。它不能覆盖重新抽取目标说话人的不确定性，也不能代替正式unseen评估。开发指标参与候选选择，后续需要在不用于选模型的正式数据上确认效果。', '']  # 不强行与论文分数直接对齐。
    with (output / 'summary_CN.md').open('x', encoding='utf-8') as handle:  # 中文说明独立新增。
        handle.write('\n'.join(lines))  # 不改已有基线报告。
    print(json.dumps({'output': str(output), 'status': result['status'], 'candidate': absolute[args.label], 'candidate_minus_baselines': comparisons}, ensure_ascii=False, indent=2), flush=True)  # 让主线程直接读取核心比较。


if __name__ == '__main__':  # 导入函数供CPU测试时不自动汇总真实候选。
    main()  # 只有显式调用才读取已完成结果。
