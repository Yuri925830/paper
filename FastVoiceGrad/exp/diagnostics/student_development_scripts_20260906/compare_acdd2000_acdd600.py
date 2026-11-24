"""比较同一ACDD训练轨迹的600步与2000步客观指标；只读完整开发评价。"""

import hashlib  # 先核对已冻结的候选读取和统计辅助代码。
import importlib.util  # 在私有模块中复用现有只读校验函数。
import json  # 保存独立的前后对照结果与来源。
import sys  # 计算开始前限制运行环境。
from pathlib import Path  # 全部输入和输出限定当前Matpool开发区。
from types import SimpleNamespace  # 明确传递两个候选的权重、契约和训练步数。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一获准执行项目计算的路径。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地只编辑，不运行此脚本。
    raise RuntimeError('Run only on the registered Matpool rental.')  # 不回落到实验室服务器。
sys.path.insert(0, str(ROOT))  # 导入真实机器身份保护。
from voicegrad.runtime_guard import assert_rented_server  # 无需导入训练、ASR或声码器。
assert_rented_server(ROOT)  # CPU统计同样检查机器指纹。
HELPER = ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development_candidate.py'  # 已冻结的候选验真实现。
HELPER_SHA = '2c08b3700f405dc9deb85e6eecc5ff63309642e15113012242f81580704078e6'  # 与已有候选汇总完全相同。
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:  # 先验字节，再执行导入。
    raise RuntimeError('Frozen candidate-summary helper changed.')  # 不修改辅助代码来迁就新结果。
spec = importlib.util.spec_from_file_location('_acdd_2000_600_summary_helper', HELPER)  # 私有模块名不影响其他进程。
helper = importlib.util.module_from_spec(spec)  # 只复用读取和纯CPU统计函数。
spec.loader.exec_module(helper)  # 原main保护确保不自动汇总任何真实文件。
base, np = helper.base, helper.np  # 共用同样的CER、SVA和10源簇bootstrap实现。
REFERENCE, CANDIDATE = 'student_acdd_clean_s600', 'student_acdd_clean_s2000'  # 差值固定为2000减600。
IDENTITIES = {  # 两个检查点身份由主线程事先明确指定。
    REFERENCE: {  # 原600步独立学生权重。
        'checkpoint_sha256': 'b3eedcddc465b8cc79756f8aedcfc126eef5a807c35334a712b74c6691562882',  # 已完整评价的S600。
        'training_contract_sha256': 'f9472aea9332e8c261e4045ceac6630efe8f9ba012f135d5b4aba45216b32c13',  # 原开发训练契约canonical身份。
        'global_step': 600, 'epoch': 0,  # 实际更新步数，不用epoch标签冒充整轮。
    },
    CANDIDATE: {  # 从600续训完成2000后保留的完整断点。
        'checkpoint_sha256': '77de35a153636d8156a45c6c23b21191b2c663d55963df874e17946e3a7ff72c',  # 主线程已核实的完整S/D/Adam文件。
        'training_contract_sha256': 'f7d351f786c9cb7f4583327a8ff7a4b6d2c551c40bdbf36ce9e54dbe1fad933b',  # 子续训契约的canonical身份。
        'global_step': 2000, 'epoch': 0,  # 已完成的实际全局步数。
    },
}  # 不使用可被后续训练覆盖的latest名称。


def validate_lineage(parent, child):  # 确认两个模型属于本次已经登记的连续训练轨迹。
    lineage = child.get('lineage', {})  # 续训必须明确保存父关系。
    if child.get('kind') != 'acdd_development_continuation' or lineage.get('parent_contract_sha256') != IDENTITIES[REFERENCE]['training_contract_sha256'] or lineage.get('start_after_global_step') != 600:  # 父契约和续训起点都不能缺失。
        raise ValueError('The step-2000 model is not the declared continuation of step 600.')  # 不把另一实验当作多训练1400步。
    for field in ('configuration', 'training_indices', 'excluded_train_indices', 'reconstruction_dev_indices'):  # 训练目标配置与数据池须保持相同。
        if field not in parent or child.get(field) != parent[field]:  # 缺字段不做推断。
            raise ValueError(f'Continuation changes a held-fixed field: {field}')  # 改目标或划分应另作比较。
    if child.get('stop_global_step') != 2000:  # 当前契约就是有界的2000步阶段。
        raise ValueError('Expected the audited continuation ending at global step 2000.')  # 不接受另一个终点阶段。


def paired_statistics(reference_rows, candidate_rows, clusters):  # 同一次源簇抽样同时用于600和2000。
    weights = base.bootstrap_weights(10, 10000, 2026090617)  # 与旧汇总同PCG64种子、次数。
    before = base.cluster_sums(reference_rows, clusters)  # 每条源语音连同9个目标的分子/分母与分数和。
    after = base.cluster_sums(candidate_rows, clusters)  # 相同源和目标，唯一变化是模型状态。
    point_before, point_after = base.values_from_sums(before.sum(axis=0)), base.values_from_sums(after.sum(axis=0))  # 原完整90对主数。
    intervals = base.confidence_interval(base.values_from_sums(weights @ after) - base.values_from_sums(weights @ before))  # 每次CER重新累计字符计数后作差。
    metrics = {metric: {'acdd600': float(point_before[i]), 'acdd2000': float(point_after[i]), 'delta_acdd2000_minus_acdd600': float(point_after[i] - point_before[i]), 'cluster_bootstrap_95_percentile_ci': intervals[i].tolist(), 'favorable_delta_sign': 'negative' if metric == 'CER_corpus_percent' else 'positive', 'unit': 'percentage_points' if metric in ('CER_corpus_percent', 'SVA_percent') else 'score'} for i, metric in enumerate(base.METRICS)}  # DNSMOS各头和连续余弦均保留。
    return metrics, weights  # 返回配对差值及可复查抽样身份。


def main():  # 主线程确认scores-ready后才显式启动，不主动轮询。
    output = base.DEV / 'comparison_acdd2000_vs_acdd600_v1'  # 新的独立比较目录。
    if output.exists():  # 不覆盖任何历史汇总。
        raise RuntimeError('The independent step-2000 comparison already exists.')  # 重算须另行明确版本。
    _, template, development, metric_identity, hashes = base.load_results(base.DEV / 'evaluation_baselines_v1')  # 复用已冻结90对和评价器身份。
    systems, contracts, inference_reports = {}, {}, {}  # 两模型的数值与原证据分开保留。
    for label, identity in IDENTITIES.items():  # 任一候选缺少完整四项完成收据都会拒绝。
        args = SimpleNamespace(label=label, candidate_root=base.DEV / ('evaluation_candidate_' + label), **identity)  # 显式固定checkpoint/contract/global_step。
        rows, contract, _, report = helper.load_candidate(args, template, development, metric_identity, hashes)  # 检查52句排除、相同CSV/波形/评分资产与真实文件哈希。
        systems[label], contracts[label], inference_reports[label] = rows, contract, report  # 不加载模型或优化器。
    validate_lineage(contracts[REFERENCE], contracts[CANDIDATE])  # 两个已验真的契约再交叉验证父子关系。
    if inference_reports[REFERENCE]['x950_sha256_per_pair'] != inference_reports[CANDIDATE]['x950_sha256_per_pair']:  # 600和2000都登记了每对实际初始状态。
        raise ValueError('The two candidates did not use identical per-pair noisy starts.')  # 不只凭随机种子文字认定输入相同。
    clusters = sorted({(row['source_speaker'], row['source_utt']) for row in template})  # 相同10条源语音的固定顺序。
    if len(clusters) != 10 or any(sum((row['source_speaker'], row['source_utt']) == cluster for row in template) != 9 for cluster in clusters):  # 一簇包含该源的全部9目标。
        raise ValueError('Expected exactly 10 source clusters with nine target pairs each.')  # 不把90当90条独立源语音。
    metrics, weights = paired_statistics(systems[REFERENCE], systems[CANDIDATE], clusters)  # 只处理CPU小数组。
    breakdown = base.cer_concentration_audit(systems, clusters)  # 保留逐源时长、原ASR文本和全部目标分组。
    result = {  # 不根据损失告警预设2000步会改善或退步。
        'status': 'complete_candidate_pairwise_development_only', 'reference': REFERENCE, 'candidate': CANDIDATE, 'identities': IDENTITIES, 'metrics': metrics,  # 完整客观主数及差值。
        'lineage_verified': True, 'identical_per_pair_x950_verified': True, 'additional_updates': 1400,  # 只有真实契约和输入哈希通过才写这些标记。
        'n_pairs': 90, 'n_source_clusters': 10, 'bootstrap': {'method': 'paired source-utterance cluster percentile bootstrap', 'repetitions': 10000, 'seed': 2026090617, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__, 'weight_matrix_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()},  # 统计口径与已有比较一致。
        'cer_concentration_audit': breakdown, 'target_p308_p317_breakdown': {label: [row for row in item['by_target'] if row['reference_speaker'] in ('p308', 'p317')] for label, item in breakdown.items()},  # 预先固定的两个目标继续检查，不重挑困难目标。
        'source_sha256': hashes, 'frozen_helper_sha256': HELPER_SHA, 'script_sha256': base.digest(Path(__file__)),  # 完整来源记录。
        'limitations': ['Same training trajectory and seed, not independent training replicates; this contrast measures checkpoint differences on a fixed development set.', 'Only ten source clusters, fixed target references and unadjusted descriptive intervals; no population or multiple-testing significance claim.', 'CER is corpus sum(errors)/sum(chars), SVA remains cosine >0.25 and DNSMOS headline remains mos_ovr.', 'All 90 development pairs remain in the main metrics, including p308 and p317; breakdowns do not redefine the main score.', 'Development results select candidates and cannot replace a separately held final unseen evaluation or direct paper-test comparison.'],  # 如实交代开发用途与统计限制。
    }  # 没有重评估、重新识别或主观听评。
    for path, expected in hashes.items():  # 防止模型或CSV在读取期间发生并发覆盖。
        if base.digest(path) != expected:  # 任一输入变动就停止发布。
            raise RuntimeError(f'Input changed during paired checkpoint comparison: {path}')  # 不更改旧账本。
    output.mkdir(exist_ok=False)  # 全部检查通过后创建新结果目录。
    with (output / 'comparison.json').open('x', encoding='utf-8') as handle:  # 完整JSON结果独立保存。
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 非有限值不能伪装为正常分数。
        handle.write('\n')  # 为文本工具保留末尾换行。
    lines = ['# ACDD2000与ACDD600的固定开发集比较', '', '我比较了同一条训练轨迹在600步和2000步时的实际客观指标。两份训练契约的配置、数据划分与父子关系一致，90对推理的实际x950哈希也完全一致。2000步比600步增加1400次更新，没有在本次比较中重新抽取语音。', '', '| 指标 | ACDD600 | ACDD2000 | 2000减600 | 95%配对区间 |', '|---|---:|---:|---:|---|']  # 用实际步数标识模型。
    for metric, item in metrics.items():  # 主要及辅助客观指标全部保留。
        low, high = item['cluster_bootstrap_95_percentile_ci']  # 同一10源簇的配对区间。
        lines.append(f"| {metric} | {item['acdd600']:.4f} | {item['acdd2000']:.4f} | {item['delta_acdd2000_minus_acdd600']:+.4f} | [{low:+.4f}, {high:+.4f}] |")  # 不挑选对2000步有利的列。
    lines += ['', 'CER由全部错误字符数除以全部参考字符数得到；每次bootstrap也重新累加分子和分母。CER负差为改善，UTMOS和DNSMOS正差为改善。SVA沿用余弦严格大于0.25，CER/SVA差值单位是百分点。', '', '90个配对来自10条源录音，每个源转换到9个固定目标。我按源语音整组重抽样，保留目标间的组内关系；目标参考本身没有重新抽样。单条训练轨迹和10个源簇不足以代表多次训练与其他说话人的不确定性。', '', '全部90对仍计入主数。p308、p317及各源句的原ASR转写分解保存在comparison.json，仅用于检查错误分布。这份开发结果不替代正式unseen评估，也没有直接套用论文分数。', '']  # 不把开发改善冒称最终复现成功。
    with (output / 'comparison_CN.md').open('x', encoding='utf-8') as handle:  # 中文说明新建，不碰旧比较。
        handle.write('\n'.join(lines))  # 保留冻结历史结果。
    print(json.dumps({'output': str(output), 'status': result['status'], 'metrics': metrics}, ensure_ascii=False, indent=2), flush=True)  # 向主线程报告真实完成结果。


if __name__ == '__main__':  # 测试导入不会读取模型评价文件。
    main()  # 只在收到scores-ready后显式执行。
