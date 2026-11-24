"""对同90个配对比较endpoint100与ACDD600的既有客观指标，不再运行模型。"""

import hashlib  # 核验已冻结的候选汇总辅助代码。
import importlib.util  # 在私有命名空间复用只读校验函数。
import json  # 独立保存两个候选的配对差值与区间。
import sys  # 本地Windows只能编辑本文件。
from pathlib import Path  # 输入和新输出均限定出租服务器开发区。
from types import SimpleNamespace  # 向冻结校验函数传递明确候选身份。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一获准进行项目计算的目录。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 不在本地或实验室服务器执行。
    raise RuntimeError('Run only on the registered Matpool rental.')  # 提前拒绝错误环境。
sys.path.insert(0, str(ROOT))  # 只导入机器指纹保护。
from voicegrad.runtime_guard import assert_rented_server  # 不导入训练或评分模型。
assert_rented_server(ROOT)  # 在全部数组计算前验证主机。
HELPER = ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development_candidate.py'  # 已通过5项CPU测试的候选汇总实现。
HELPER_SHA = '2c08b3700f405dc9deb85e6eecc5ff63309642e15113012242f81580704078e6'  # 保持辅助代码原字节，不改其适用接口。
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:  # 先验证再执行导入。
    raise RuntimeError('Frozen candidate-summary helper changed.')  # 不能用另一版本重新解释旧指标。
spec = importlib.util.spec_from_file_location('_endpoint_acdd_pairwise_helper', HELPER)  # 独立模块名不覆盖其他流程。
helper = importlib.util.module_from_spec(spec)  # 准备纯读取与统计函数。
spec.loader.exec_module(helper)  # 原main保护确保不会自动运行候选汇总。
base, np = helper.base, helper.np  # 共用完全相同的CER、SVA和源簇bootstrap实现。

IDENTITIES = {  # 本次两个候选都使用用户已明确的完整身份。
    'student_acdd_clean_s600': {  # 已完整评价的600步ACDD参照。
        'checkpoint_sha256': 'b3eedcddc465b8cc79756f8aedcfc126eef5a807c35334a712b74c6691562882',  # 600步独立权重文件字节。
        'training_contract_sha256': 'f9472aea9332e8c261e4045ceac6630efe8f9ba012f135d5b4aba45216b32c13',  # ACDD原训练契约内容SHA。
        'global_step': 600, 'epoch': 0,  # 开发训练实际更新步数，不冒称完成整轮。
    },
    'student_endpoint_lr1e4_s100': {  # 因GTmel守护停止后接受客观评价的100步候选。
        'checkpoint_sha256': '447c9de695f893c336991bd21feb4274c70833ef68c1bde7fb20590eb9944920',  # 已完整保存的100步权重。
        'training_contract_sha256': '73f7923f78cdec7d95d6e2acb7276ed25e4493eaf09ae12502bf80dd57375aba',  # endpoint目标及数据排除的原契约。
        'global_step': 100, 'epoch': 0,  # 不读取尚未完成或不存在的后续断点。
    },
}  # 此比较不重训，不更换开发语音或评分器。


def paired_statistics(reference_rows, candidate_rows, clusters):  # 整组配对抽样保证两候选使用相同抽中源语音。
    weights = base.bootstrap_weights(10, 10000, 2026090617)  # 与先前全部汇总使用同一PCG64重采样。
    reference_sums = base.cluster_sums(reference_rows, clusters)  # ACDD每源9目标计数与评分总和。
    candidate_sums = base.cluster_sums(candidate_rows, clusters)  # endpoint同源同目标的计数与总和。
    reference_point = base.values_from_sums(reference_sums.sum(axis=0))  # ACDD完整90对主数。
    candidate_point = base.values_from_sums(candidate_sums.sum(axis=0))  # endpoint完整90对主数。
    difference = candidate_point - reference_point  # 固定差值方向endpoint减ACDD。
    draws = base.values_from_sums(weights @ candidate_sums) - base.values_from_sums(weights @ reference_sums)  # 每次CER先从总错误/总字符计算，再做配对差值。
    intervals = base.confidence_interval(draws)  # 按10000次差值的2.5/97.5分位数给区间。
    metrics = {metric: {'endpoint100': float(candidate_point[i]), 'acdd600': float(reference_point[i]), 'delta_endpoint100_minus_acdd600': float(difference[i]), 'cluster_bootstrap_95_percentile_ci': intervals[i].tolist(), 'favorable_delta_sign': 'negative' if metric == 'CER_corpus_percent' else 'positive', 'unit': 'percentage_points' if metric in ('CER_corpus_percent', 'SVA_percent') else 'score'} for i, metric in enumerate(base.METRICS)}  # DNSMOS四个输出完整保留。
    return metrics, weights  # 返回主数、配对区间及可核对抽样矩阵。


def main():  # 仅读取四项已完成指标，不创建ASR、声纹或自然度模型。
    output = base.DEV / 'comparison_endpoint100_vs_acdd600_v1'  # 两候选之间单独的新比较目录。
    if output.exists():  # 避免覆盖已完成证据。
        raise RuntimeError('The independent pairwise comparison already exists.')  # 如需另一次统计应人工另起协议。
    _, template, development, metric_identity, hashes = base.load_results(base.DEV / 'evaluation_baselines_v1')  # 用同一冻结模板和评分身份作为共同参照。
    systems, contracts = {}, {}  # 两个候选分别保存完整逐条记录与训练目标信息。
    for label, identity in IDENTITIES.items():  # 任何候选缺少完整四项CSV都会停止。
        args = SimpleNamespace(label=label, candidate_root=base.DEV / ('evaluation_candidate_' + label), **identity)  # 显式传入权重、契约、步数身份。
        rows, contract, _, _ = helper.load_candidate(args, template, development, metric_identity, hashes)  # 校验52句排除、同配对、同指标版本及全部CSV/wav/检查点哈希。
        systems[label], contracts[label] = rows, contract  # 不从GTmel告警推断实际音质。
    clusters = sorted({(row['source_speaker'], row['source_utt']) for row in template})  # 相同10条独立源语音。
    if len(clusters) != 10 or any(sum((r['source_speaker'], r['source_utt']) == cluster for r in template) != 9 for cluster in clusters):  # 每个源必须带着全部9目标。
        raise ValueError('Expected the frozen ten-by-nine development pairing.')  # 不接受其他样本规模。
    metrics, weights = paired_statistics(systems['student_acdd_clean_s600'], systems['student_endpoint_lr1e4_s100'], clusters)  # 纯CPU的小数组统计。
    audit = base.cer_concentration_audit(systems, clusters)  # 两个系统的逐源长度、目标条件、原转写一并保留。
    result = {  # 与已冻结的各自候选汇总分开存档。
        'status': 'complete_candidate_pairwise_development_only', 'reference': 'student_acdd_clean_s600', 'candidate': 'student_endpoint_lr1e4_s100', 'identities': IDENTITIES, 'metrics': metrics,  # 明确两个候选、差值和身份。
        'n_pairs': 90, 'n_source_clusters': 10, 'bootstrap': {'method': 'paired source-utterance cluster percentile bootstrap', 'repetitions': 10000, 'seed': 2026090617, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__, 'weight_matrix_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()},  # 相同抽样规则可完整复核。
        'target_p308_p317_breakdown': {label: [row for row in item['by_target'] if row['reference_speaker'] in ('p308', 'p317')] for label, item in audit.items()},  # 固定检查曾发生旧学生条件失效的目标。
        'cer_concentration_audit': audit, 'source_sha256': hashes, 'frozen_helper_sha256': HELPER_SHA, 'script_sha256': base.digest(Path(__file__)),  # 原转写及结果来源完整保存。
        'training_objectives': {label: contract.get('objective_id', contract.get('kind')) for label, contract in contracts.items()},  # 训练目标不同必须明确，不宣称单因素消融。
        'limitations': ['Different objectives and update counts: endpoint100 versus ACDD600, not a matched-step controlled ablation.', 'Both candidates use the same 90 development pairs and fixed evaluation assets; all pairs remain in the main scores.', 'Only ten source clusters and fixed target references; intervals are exploratory and not adjusted for multiple metrics.', 'A GT-mel guard event is a training diagnostic, not an objective waveform-quality conclusion; this report uses actual completed CER/SVA/DNSMOS/UTMOS only.', 'Development data are used for selection, so these scores do not replace final unseen evaluation or paper-test comparisons.'],  # 不混淆守护告警、客观质量与正式泛化。
    }  # 完整90对总CER的分子和分母已经逐条校验。
    for path, expected in hashes.items():  # 所有来源在读取期间须保持原样。
        if base.digest(path) != expected:  # 输入变化就不发布结果。
            raise RuntimeError(f'Input changed while comparing candidates: {path}')  # 不接受跨版本指标混合。
    output.mkdir(exist_ok=False)  # 所有校验通过后才创建新目录。
    with (output / 'comparison.json').open('x', encoding='utf-8') as handle:  # 原候选汇总始终不改。
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 非有限值不能被写作有效指标。
        handle.write('\n')  # 文本末尾保留换行。
    lines = ['# endpoint100 与 ACDD600 的开发集配对比较', '', '我只用已完成的客观评估结果判断两者差异。endpoint训练期间的GT mel告警单独记录，不预先据此认定语音质量变差。两次训练目标和更新步数不同，这份结果用于比较现有候选，不能看成等步数的单因素消融。', '', '| 指标 | ACDD600 | endpoint100 | endpoint减ACDD | 95%配对区间 |', '|---|---:|---:|---:|---|']  # 中文报告先交代比较范围。
    for metric, item in metrics.items():  # 主指标及其他DNSMOS输出全部显示。
        low, high = item['cluster_bootstrap_95_percentile_ci']  # 10源簇的探索性区间。
        lines.append(f"| {metric} | {item['acdd600']:.4f} | {item['endpoint100']:.4f} | {item['delta_endpoint100_minus_acdd600']:+.4f} | [{low:+.4f}, {high:+.4f}] |")  # 保留改善与退步的正负方向。
    lines += ['', 'CER按全部字符错误数除以全部参考字符数计算，每次bootstrap也重新累计分子和分母。SVA保持余弦严格大于0.25的规则。CER和SVA差值单位为百分点，其他指标差值为分数。', '', '90个转换配对只对应10条源录音。我按源语音整组重抽样，每组9个目标一起带入；目标参考本身固定。全部90对仍计入主数，p308、p317和逐源转写分解保存在comparison.json，仅用于定位问题。这个开发结果不能代替正式unseen评估。', '']  # 明确统计限制且不删除困难目标。
    with (output / 'comparison_CN.md').open('x', encoding='utf-8') as handle:  # 新比较文档独立保存。
        handle.write('\n'.join(lines))  # 不覆盖冻结基线报告。
    print(json.dumps({'output': str(output), 'status': result['status'], 'metrics': metrics}, ensure_ascii=False, indent=2), flush=True)  # 返回实际指标，无任何模型运算。


if __name__ == '__main__':  # 导入纯配对函数用于测试时不会读取真实结果。
    main()  # 显式调用后才检查已完成指标。
