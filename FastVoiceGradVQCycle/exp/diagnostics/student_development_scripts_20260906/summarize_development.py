"""汇总固定开发集的既有四项指标，以源语音为单位计算配对 bootstrap 区间。"""

import argparse  # 只指定既有评价目录与新汇总目录，不提供重评估开关。
import csv  # 逐条读取已完成的90对结果，不能只照抄平均值。
import hashlib  # 核对指标、配对表和生成记录的来源。
import itertools  # 四个系统共有六组两两比较。
import json  # 读取已有完成凭据，写入独立汇总证据。
import math  # 拒绝NaN、Inf以及超出误差范围的汇总值。
import re  # 按原CER约定核验文本，不重新识别语音。
import sys  # 在任何项目计算前限定Matpool运行位置。
import unicodedata  # 与既有CER评分器采用相同ASCII文本规范化。
from pathlib import Path  # 所有输入均为权威服务器中的独立开发结果。

ROOT = Path('/root/FsatVoiceGrad')  # 禁止本地Windows或实验室服务器执行。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地只允许编辑与审阅。
    raise RuntimeError('Run only on the registered Matpool rental.')  # 未连接出租机就停止。
sys.path.insert(0, str(ROOT))  # 只导入现有的主机身份保护。
from voicegrad.runtime_guard import assert_rented_server  # 不导入训练、推理或评分模型。
assert_rented_server(ROOT)  # 在NumPy计算之前核对机器身份。
import numpy as np  # bootstrap只处理几十个数的CPU数组，不使用GPU。

DEV = ROOT / 'exp/diagnostics/student_development_20260906'  # 已冻结的52句开发语音来源。
SYSTEMS = ('teacher500_k30', 'old_student_epoch100_k1', 'teacher_initialized_s0_k1', 'repair_custom_step2460_k1')  # 报表按教师、旧学生、S0、修复学生排列。
CHECKPOINTS = {  # 绑定本轮四个固定系统，不能混入后续同名latest。
    'teacher500_k30': ('1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6', 30),  # 教师500轮、30次前向。
    'old_student_epoch100_k1': ('eb9ad626160372f1b45b29d6c6d3f058fa8e553cfe48a5ad6a8c007e00d7ca27', 1),  # 旧100轮单步学生。
    'teacher_initialized_s0_k1': ('1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6', 1),  # 同教师权重的单步起点。
    'repair_custom_step2460_k1': ('2f5c56f819b98994a25c4b891412cadac1af5bfa0753019a06e1f8a392974da2', 1),  # 已归档的2460步修复学生。
}  # 这里只读取身份，不加载这些大权重。
FIELDS = {  # 字段与既有evaluate_metrics.py的CSV完全相同。
    'cer': ('pair_id', 'reference_text_normalized', 'hypothesis_text_normalized', 'char_errors', 'reference_chars', 'cer_percent'),  # 保存文本便于核对编辑距离。
    'sva': ('pair_id', 'cosine_score', 'threshold', 'verified'),  # 阈值固定0.25，且使用严格大于。
    'dnsmos': ('pair_id', 'p808_mos', 'mos_sig', 'mos_bak', 'mos_ovr'),  # OVRL主值与其他三列分开保留。
    'utmos': ('pair_id', 'utmos'),  # 自然度预测器原分数。
}  # 不接受缺列、重复行或部分指标结果。
METRICS = ('CER_corpus_percent', 'SVA_percent', 'DNSMOS_OVRL', 'UTMOS', 'DNSMOS_SIG', 'DNSMOS_BAK', 'DNSMOS_P808', 'SVA_cosine')  # 所有bootstrap列保持同一次抽样。


def digest(path):  # 按原文件字节记录身份，所有输入只读。
    result = hashlib.sha256()  # 每份文件各自生成一个摘要。
    with Path(path).open('rb') as handle:  # 不修改已有CSV或波形。
        for block in iter(lambda: handle.read(1048576), b''):  # 分块读取避免不必要内存占用。
            result.update(block)  # 原字节顺序参与校验。
    return result.hexdigest()  # 结果可直接与完成凭据比较。


def read_json(path, hashes):  # 读结果同时把实际文件哈希放进本次证据表。
    hashes[str(path)] = digest(path)  # 保存读取时的原字节身份。
    return json.loads(path.read_text(encoding='utf-8'))  # 仅读取结构化记录，不加载模型。


def read_csv(path, hashes, required=None):  # 输入顺序应恰好是pair_id的0到89。
    hashes[str(path)] = digest(path)  # 读取前登记文件哈希。
    with path.open(encoding='utf-8-sig', newline='') as handle:  # 正确处理转写内的逗号和引号。
        reader = csv.DictReader(handle)  # 首行定义字段。
        if required is not None and tuple(reader.fieldnames or ()) != tuple(required):  # 不以猜测填补缺失字段。
            raise ValueError(f'Unexpected CSV schema: {path}')  # 评分器版本或结果损坏需要先处理。
        rows = list(reader)  # 只保存90条小记录。
    if [row.get('pair_id') for row in rows] != [str(i) for i in range(90)]:  # 行缺失、重复、乱序都拒绝。
        raise ValueError(f'Expected complete ordered pair_id 0..89: {path}')  # 不用部分结果冒充完成。
    return rows  # 四系统随后按同一pair_id逐行比较。


def normalize_cer(text):  # 逐字沿用原评分器的文本约定，包含词间空格。
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').upper()  # 统一ASCII与大写。
    return ' '.join(re.sub(r"[^A-Z'\s]+", ' ', text).split())  # 撇号保留，其他非字母符号转为空格。


def edit_distance(reference, hypothesis):  # 仅重算保存文本的字符编辑距离，不调用ASR。
    previous = list(range(len(hypothesis) + 1))  # 空参考到每个识别前缀的插入次数。
    for i, ref in enumerate(reference, 1):  # 逐字符扩展参考文本。
        current = [i]  # 当前参考前缀变为空串需要i次删除。
        for j, hyp in enumerate(hypothesis, 1):  # 每格取插入、删除、替换的最小值。
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))  # 与原Levenshtein定义一致。
        previous = current  # 只保留上一行，节约内存。
    return previous[-1]  # 完整参考与识别文本的错误总次数。


def finite(value):  # 空值或非有限数不能进入bootstrap。
    result = float(value)  # 解析原CSV数值字符串。
    if not math.isfinite(result):  # 包括NaN及正负Inf。
        raise ValueError(f'Non-finite metric value: {value}')  # 拒绝污染整组结果。
    return result  # 保留原评分精度。


def nonnegative_int(value):  # 错误数、字符数及验证标记必须是严格非负整数。
    if not str(value).isdigit():  # 1.0或负数不能冒充计数。
        raise ValueError(f'Expected a nonnegative integer: {value}')  # 不默默取整。
    return int(value)  # 后续CER聚合始终用原整数计数。


def validate_sva(row):  # SVA阈值及通过定义都固定，不重新挑阈值。
    cosine, threshold, verified = finite(row['cosine_score']), finite(row['threshold']), nonnegative_int(row['verified'])  # 三列交叉核对。
    if not math.isclose(threshold, 0.25, abs_tol=1e-12, rel_tol=0) or verified not in (0, 1) or verified != int(cosine > 0.25):  # 恰好0.25判未通过。
        raise ValueError('SVA must use cosine_score > 0.25 without changing the threshold.')  # 防止边界规则改变。
    return cosine, verified  # 返回连续分数和二元结果供两种汇总。


def merged_rows(label, pairs, tables):  # 把同一pair的四份CSV组合为一份可追溯记录。
    merged = []  # 每系统应返回固定90条。
    for i, pair in enumerate(pairs):  # 所有表已验证按同一编号顺序排列。
        cer, sva, dns, ut = (tables[key][i] for key in ('cer', 'sva', 'dnsmos', 'utmos'))  # 取同一生成语音的分数。
        reference, hypothesis = cer['reference_text_normalized'], cer['hypothesis_text_normalized']  # 原ASR输出不再变化。
        errors, chars = nonnegative_int(cer['char_errors']), nonnegative_int(cer['reference_chars'])  # 原始CER分子与分母。
        if not reference or reference != normalize_cer(pair['source_text']) or normalize_cer(hypothesis) != hypothesis or chars != len(reference) or errors != edit_distance(reference, hypothesis):  # 文本身份、长度和错误数全部核对。
            raise ValueError(f'CER text/count mismatch: {label} pair {i}')  # 禁止混入另一配对的识别结果。
        if not math.isclose(finite(cer['cer_percent']), 100 * errors / chars, rel_tol=1e-9, abs_tol=1e-9):  # 单句百分数只作为辅助核验。
            raise ValueError(f'CER percentage mismatch: {label} pair {i}')  # 正式汇总随后从整数重新相除。
        cosine, verified = validate_sva(sva)  # 同时验证0.25边界和二元标记。
        merged.append({  # 保留身份及所有数值，不只保留主分数。
            'system': label, 'pair_id': i, 'source_speaker': pair['source_speaker'], 'source_utt': pair['source_utt'], 'reference_speaker': pair['reference_speaker'],  # 聚类由源语音身份决定。
            'source_mel_frames': int(pair['source_mel_frames']), 'expected_num_samples': int(pair['expected_num_samples']), 'generated_duration_seconds': int(pair['expected_num_samples']) / 22050,  # 统一记录完整mel对应的生成时长，不把它当作原录音精确时长。
            'reference_text_normalized': reference, 'hypothesis_text_normalized': hypothesis,  # 只读原ASR转写，用于追查错误集中位置，不试听。
            'char_errors': errors, 'reference_chars': chars, 'cer_percent': finite(cer['cer_percent']), 'verified': verified, 'cosine_score': cosine, 'threshold': 0.25,  # 字符计数与SVA口径。
            'mos_ovr': finite(dns['mos_ovr']), 'mos_sig': finite(dns['mos_sig']), 'mos_bak': finite(dns['mos_bak']), 'p808_mos': finite(dns['p808_mos']), 'utmos': finite(ut['utmos']),  # DNSMOS四个头与UTMOS分开。
        })  # 每条配对只出现一次。
    return merged  # 后续bootstrap直接复用这些冻结计数和分数。


def cluster_sums(rows, clusters):  # 每条源语音带着其全部9个目标，形成一个抽样单位。
    result = []  # 每簇10个充分统计量，避免每次bootstrap重复遍历波形或文本。
    for speaker, utterance in clusters:  # 源身份按固定次序排列，四系统一致。
        group = [row for row in rows if (row['source_speaker'], row['source_utt']) == (speaker, utterance)]  # 保留这句的所有目标。
        result.append([sum(row[key] for row in group) for key in ('char_errors', 'reference_chars', 'verified', 'mos_ovr', 'utmos', 'mos_sig', 'mos_bak', 'p808_mos', 'cosine_score')] + [len(group)])  # 最后一列为配对数。
    return np.asarray(result, dtype=np.float64)  # 计数范围很小，float64可精确表示所有整数。


def values_from_sums(sums):  # 既用于原始总分，也用于每一次成簇抽样后的总分。
    sums = np.asarray(sums, dtype=np.float64)  # 支持一行原总量或多行bootstrap总量。
    if np.any(sums[..., 1] <= 0) or np.any(sums[..., 9] <= 0):  # CER字符数和配对数都是正分母。
        raise ValueError('Positive character and pair denominators are required.')  # 不丢弃空分母样本来美化结果。
    return np.stack([100 * sums[..., 0] / sums[..., 1], 100 * sums[..., 2] / sums[..., 9], sums[..., 3] / sums[..., 9], sums[..., 4] / sums[..., 9], sums[..., 5] / sums[..., 9], sums[..., 6] / sums[..., 9], sums[..., 7] / sums[..., 9], sums[..., 8] / sums[..., 9]], axis=-1)  # CER始终采用总错误/总字符，不平均句CER。


def bootstrap_weights(cluster_count, repetitions, seed):  # 每次整簇抽10次，允许同一源语音重复出现。
    rng = np.random.Generator(np.random.PCG64(seed))  # 显式登记随机算法，不依赖隐含全局状态。
    draws = rng.integers(0, cluster_count, size=(repetitions, cluster_count))  # 一行对应一次完整的簇抽样。
    weights = np.zeros((repetitions, cluster_count), dtype=np.int64)  # 记录每个簇这次被抽中几次。
    for cluster in range(cluster_count):  # 同一权重矩阵用于所有系统和指标。
        weights[:, cluster] = np.sum(draws == cluster, axis=1)  # 9个目标一起被复制，不能独立抽目标。
    return weights  # 两系统相减时使用同一批抽样，形成配对区间。


def confidence_interval(values):  # 双侧95% percentile区间；10簇样本量很小，结果只供开发探索。
    low, high = np.percentile(values, [2.5, 97.5], axis=0)  # 使用默认线性插值，NumPy版本会一并登记。
    return np.stack([low, high], axis=-1)  # 每个指标对应[下界,上界]。


def validate_driver_summary(summary, rows):  # 从逐条计数重算主数，再与driver完成记录交叉验证。
    total = cluster_sums(rows, sorted({(r['source_speaker'], r['source_utt']) for r in rows})).sum(axis=0)  # 不依赖driver自己的聚合实现。
    values = values_from_sums(total)  # 得到相同口径的主数及辅助数。
    expected = {'CER_corpus_percent_lower_is_better': values[0], 'SVA_percent_higher_is_better': values[1], 'DNSMOS_mos_ovr_mean_higher_is_better': values[2], 'UTMOS_mean_higher_is_better': values[3]}  # 明确四个原字段名。
    for key, value in expected.items():  # 每项主数都必须相符。
        if not math.isclose(finite(summary['metrics'][key]), float(value), abs_tol=1e-10, rel_tol=1e-10):  # 只接受浮点累加级别的差别。
            raise ValueError(f'Driver summary does not match per-pair counts: {key}')  # 不输出相互矛盾的结果。
    extra = summary['additional']  # 原完成记录中的分母与阈值同样需要核验。
    if extra['CER_total_errors'] != int(total[0]) or extra['CER_total_reference_chars'] != int(total[1]) or extra['SVA_threshold'] != 0.25:  # CER原始计数不做近似。
        raise ValueError('Driver summary count/threshold mismatch.')  # 守住统一评价口径。


def descriptive_correlation(left, right):  # 仅描述10个源句的长度关系，不作因果或显著性检验。
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)  # 将同顺序源句的两个量放在一起。
    if np.std(left) == 0 or np.std(right) == 0:  # 常数没有可定义的相关系数。
        return None  # JSON用null明确表示未定义，不伪造0。
    return float(np.corrcoef(left, right)[0, 1])  # 结果只能作为小样本描述。


def average_ranks(values):  # 计算有并列值时的平均秩，不额外依赖SciPy。
    values = np.asarray(values, dtype=float)  # 两条源语音可能有相同mel长度。
    order = np.argsort(values, kind='stable')  # 相等值保持稳定顺序。
    ranks = np.empty(len(values), dtype=float)  # 每个原位置对应一个秩。
    first = 0  # 从最小值开始逐组处理。
    while first < len(order):  # 每次把所有相同值作为一组。
        last = first + 1  # 至少包含一个样本。
        while last < len(order) and values[order[last]] == values[order[first]]:  # 向后合并并列值。
            last += 1  # 找到本组末尾的下一个位置。
        ranks[order[first:last]] = (first + last - 1) / 2  # 零起始平均秩不影响相关系数。
        first = last  # 继续处理下一组不同取值。
    return ranks  # 用于Spearman描述性相关。


def cer_concentration_audit(systems, clusters):  # 从已验证的逐条转写定位源句与目标条件上的错误分布。
    audit = {}  # 所有系统都报告同样的字段，避免只观察旧学生。
    targets = sorted({row['reference_speaker'] for row in next(iter(systems.values()))})  # 同一组10个固定目标。
    for label, rows in systems.items():  # 主数依然来自完整90对，下面的分组仅供定位。
        total_errors = sum(row['char_errors'] for row in rows)  # 原完整错误数作为贡献率分母。
        source_rows, target_rows = [], []  # 源视角与目标视角分开，不混淆其含义。
        for speaker, utterance in clusters:  # 每源对应原来固定的9个转换目标。
            group = [row for row in rows if (row['source_speaker'], row['source_utt']) == (speaker, utterance)]  # 取出相同源音频。
            errors, chars = sum(row['char_errors'] for row in group), sum(row['reference_chars'] for row in group)  # 仍按总错误/总字符计算。
            source_rows.append({  # 源语音长度与各目标错误一并记录。
                'source_speaker': speaker, 'source_utt': utterance, 'mel_frames': group[0]['source_mel_frames'], 'generated_duration_seconds': group[0]['generated_duration_seconds'],  # 长度是mel帧对应的输出长度。
                'char_errors': errors, 'reference_chars': chars, 'CER_corpus_percent': 100 * errors / chars, 'share_of_system_errors_percent': 100 * errors / total_errors if total_errors else 0,  # 错误分布只作诊断。
                'reference_text_normalized': group[0]['reference_text_normalized'],  # 9个目标共同的正确文本。
                'targets': [{key: row[key] for key in ('pair_id', 'reference_speaker', 'char_errors', 'reference_chars', 'cer_percent', 'hypothesis_text_normalized', 'cosine_score', 'verified')} for row in group],  # 每个目标保留原ASR文本和声纹判断。
            })  # 不对输出文本进行人工修正。
        for target in targets:  # 每个目标接收其他9个源语音。
            group = [row for row in rows if row['reference_speaker'] == target]  # 固定同一参考语音条件。
            errors, chars = sum(row['char_errors'] for row in group), sum(row['reference_chars'] for row in group)  # 目标间文本长度不同，不能只比平均句CER。
            target_rows.append({'reference_speaker': target, 'pairs': len(group), 'char_errors': errors, 'reference_chars': chars, 'CER_corpus_percent': 100 * errors / chars, 'share_of_system_errors_percent': 100 * errors / total_errors if total_errors else 0, 'SVA_passes': sum(row['verified'] for row in group), 'SVA_mean_cosine': float(np.mean([row['cosine_score'] for row in group])), 'hypothesis_chars_total': sum(len(row['hypothesis_text_normalized']) for row in group)})  # 识别文本缩短是客观转写现象，不直接认定音频被截断。
        worst_targets = sorted(target_rows, key=lambda row: (-row['char_errors'], row['reference_speaker']))[:2]  # 事后定位错误贡献最多的两个目标，不影响主评价集。
        worst_names = [row['reference_speaker'] for row in worst_targets]  # 明确这是描述性排序。
        other = [row for row in rows if row['reference_speaker'] not in worst_names]  # 其余配对仅作诊断，不成为新主数。
        durations = [row['generated_duration_seconds'] for row in source_rows]  # 10个独立源句的生成长度。
        source_cer = [row['CER_corpus_percent'] for row in source_rows]  # 每源合并其全部9目标。
        audit[label] = {  # 单独保存错误集中和时长关系的证据。
            'by_source': source_rows, 'by_target': target_rows,  # 每句、每目标的错误分解。
            'length_association': {'n_source_utterances': 10, 'duration_min_seconds': min(durations), 'duration_max_seconds': max(durations), 'pearson_duration_vs_source_corpus_CER': descriptive_correlation(durations, source_cer), 'spearman_duration_vs_source_corpus_CER': descriptive_correlation(average_ranks(durations), average_ranks(source_cer)), 'interpretation': 'Descriptive correlation over only 10 source utterances, without adjustment for speaker/text/target effects; not evidence of a length mechanism.'},  # 不用小样本相关证明因果。
            'posthoc_top2_target_error_concentration': {'targets': worst_names, 'error_count': sum(row['char_errors'] for row in worst_targets), 'total_system_error_count': total_errors, 'share_of_system_errors_percent': 100 * sum(row['char_errors'] for row in worst_targets) / total_errors if total_errors else 0, 'other_pairs_count': len(other), 'other_pairs_CER_for_diagnosis_only_percent': 100 * sum(row['char_errors'] for row in other) / sum(row['reference_chars'] for row in other), 'not_used_to_remove_pairs_or_redefine_the_main_score': True},  # 完整90对主数永远保留。
        }  # 不筛样本、不改阈值、不调用任何评分模型。
    return audit  # 后续中文报告引用其具体数字。


def load_results(evaluation):  # 全部结果完整且来源一致之后才开始统计。
    hashes = {}  # 最后复查，避免一边写metrics一边汇总半成品。
    receipt = read_json(DEV / 'development_manifest.json', hashes)  # 固定开发集生成凭据。
    contract = read_json(DEV / 'development_contract.json', hashes)  # 暴露范围和选择规则也保存在汇总中。
    pairs = read_csv(DEV / 'pairs.csv', hashes)  # 唯一配对模板。
    if receipt.get('status') != 'complete' or receipt['pairs_csv_sha256'] != hashes[str(DEV / 'pairs.csv')] or receipt['development_contract_file_sha256'] != hashes[str(DEV / 'development_contract.json')]:  # 不接受未冻结开发集。
        raise ValueError('Development manifest receipt mismatch.')  # 不重新选择语音。
    identity_path, metric_identity_path = evaluation / 'run_identity.json', evaluation / 'metric_identity.json'  # 全系统共同身份。
    identity = read_json(identity_path, hashes)  # 四个系统的来源与推理规则。
    metric_identity = read_json(metric_identity_path, hashes)  # 指标模型和软件版本已由实际评分driver核验。
    if identity.get('purpose') != 'development_diagnostic_not_formal_text_unseen_8100' or identity.get('pairs') != 90:  # 正式8100结果不在本工具范围内。
        raise ValueError('Expected the fixed 90-pair development evaluation.')  # 不把不同实验混合。
    for path in (DEV / 'pairs.csv', DEV / 'development_contract.json'):  # 推理实际使用的模板必须就是现冻结文件。
        if identity['inputs'].get(str(path)) != hashes[str(path)]:  # 将选样与推理身份连接起来。
            raise ValueError(f'Inference used a different development input: {path}')  # 不接受后补的不同配对。
    systems = {}  # 保存四系统完整逐条结果。
    for label in SYSTEMS:  # 此版汇总不静默跳过任一系统。
        directory = evaluation / label  # 每个系统使用独立子目录。
        report_path = directory / 'inference_complete.json'  # 必须已完整生成90条音频。
        report = read_json(report_path, hashes)  # 不因wav存在就冒认完成。
        actual = read_csv(directory / 'pairs.csv', hashes)  # 波形路径由实际推理输出记录。
        expected_sha, expected_steps = CHECKPOINTS[label]  # 核对冻结模型身份与推理步数。
        if report.get('status') != 'development_diagnostic_only' or report['system'] != label or report['spec'] != identity['systems'][label] or (report['spec']['sha256'], report['spec']['steps']) != (expected_sha, expected_steps):  # 避免同名系统换权重。
            raise ValueError(f'Baseline model identity mismatch: {label}')  # 不加载检查点也能核对原凭据。
        if report['run_identity_sha256'] != hashes[str(identity_path)] or report['pairs_csv_sha256'] != hashes[str(directory / 'pairs.csv')]:  # 推理凭据绑定本系统CSV和全局身份。
            raise ValueError(f'Inference receipt mismatch: {label}')  # 任何改动都会阻断汇总。
        if [{k: v for k, v in row.items() if k != 'output_wav'} for row in actual] != [{k: v for k, v in row.items() if k != 'output_wav'} for row in pairs]:  # 只允许模型输出路径变化。
            raise ValueError(f'Pairing changed across systems: {label}')  # 同pair的源、目标、文本、种子都须相同。
        for row in actual:  # 只读核对已评分波形的字节，不解码、不重新评分。
            path = Path(row['output_wav']).resolve()  # 确认生成语音属于对应系统输出区。
            if directory not in path.parents or not path.is_file() or digest(path) != report['wav_sha256'].get(row['pair_id']):  # 不允许跨目录借用另一系统输出。
                raise ValueError(f'Generated waveform identity mismatch: {label} pair {row["pair_id"]}')  # 读已有分数也保留波形来源链。
            hashes[str(path)] = report['wav_sha256'][row['pair_id']]  # 汇总结束前再次复查。
        summary_path = directory / 'metrics/development_summary_cer_dnsmos_sva_utmos.json'  # 必须已有四项完整评分收据。
        summary = read_json(summary_path, hashes)  # 缺少文件就停止，不触发ASR或评分模型。
        if summary.get('status') != 'development_only' or summary['system'] != label or summary['evaluated_pairs'] != 90 or summary['inference_complete_sha256'] != hashes[str(report_path)] or summary['run_identity_sha256'] != hashes[str(identity_path)] or summary['metric_identity_sha256'] != hashes[str(metric_identity_path)]:  # 同时核对阶段和指标身份。
            raise ValueError(f'Complete metric receipt mismatch: {label}')  # 分批或半途评分必须先由原driver完成。
        tables = {}  # 四项CSV每项应有90行。
        for metric, fields in FIELDS.items():  # 逐列检查原评分器schema。
            path = directory / f'metrics/{metric}.csv'  # 所有数据已存在，本工具不写此目录。
            tables[metric] = read_csv(path, hashes, fields)  # 行顺序、数量、表头一起核对。
            if summary['metric_csv_sha256'].get(metric) != hashes[str(path)]:  # 指标CSV须与完成凭据完全一致。
                raise ValueError(f'Metric CSV changed after completion: {path}')  # 不重新签发旧评估身份。
        systems[label] = merged_rows(label, actual, tables)  # 严格核验CER/SVA与数值后组合。
        validate_driver_summary(summary, systems[label])  # 独立从90条数据重算主数。
    return systems, pairs, contract, metric_identity, hashes  # 到这里才允许进行bootstrap。


def main():  # 仅消费完成结果并生成新汇总，不占用任何模型运行锁。
    parser = argparse.ArgumentParser(description=__doc__)  # 固定四系统，禁止拿formal输入套本开发协议。
    parser.add_argument('--evaluation-root', type=Path, default=DEV / 'evaluation_baselines_v1')  # 当前四系统结果位置。
    parser.add_argument('--output-dir', type=Path)  # 默认建立结果目录下的新子目录，不覆盖已有汇总。
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)  # 10簇重复10000次，只有CPU小数组运算。
    parser.add_argument('--bootstrap-seed', type=int, default=2026090617)  # 固定重采样次序，便于复核。
    args = parser.parse_args()  # 不提供GPU、下载或重评分参数。
    evaluation = args.evaluation_root.resolve()  # 消除相对路径歧义。
    output = args.output_dir.resolve() if args.output_dir else evaluation / 'paired_summary_v1'  # 独立的新结果目录。
    if DEV not in evaluation.parents or evaluation not in output.parents or output.exists():  # 输入和输出均限定本开发区，拒绝覆盖。
        raise ValueError('Use the fixed development area and a new child output directory.')  # 旧结果始终保留。
    if not 1000 <= args.bootstrap_repetitions <= 100000 or args.bootstrap_seed < 0:  # 防止意外开启无上限计算。
        raise ValueError('Bootstrap requires 1000..100000 repetitions and a nonnegative seed.')  # 次数与种子必须明确可追溯。
    systems, pairs, contract, metric_identity, hashes = load_results(evaluation)  # 全部CSV完成后才进入统计。
    clusters = sorted({(row['source_speaker'], row['source_utt']) for row in pairs})  # 一条源语音就是一个簇。
    counts = [sum((row['source_speaker'], row['source_utt']) == cluster for row in pairs) for cluster in clusters]  # 每簇应固定9个目标。
    if len(clusters) != 10 or counts != [9] * 10 or len({speaker for speaker, _ in clusters}) != 10:  # 本协议每位源说话人只有一条源句。
        raise ValueError('Expected 10 distinct source-utterance clusters, each with 9 target conversions.')  # 不把其他配对拓扑套入本区间。
    weights = bootstrap_weights(10, args.bootstrap_repetitions, args.bootstrap_seed)  # 所有系统共享这同一矩阵。
    points, bootstraps, per_source, estimates = {}, {}, {}, {}  # 原分数、重采样分数和簇级证据分别保留。
    for label, rows in systems.items():  # 不通过平均单句CER改变论文原统计口径。
        sums = cluster_sums(rows, clusters)  # 10簇的分子、分母和评分总和。
        points[label] = values_from_sums(sums.sum(axis=0))  # 原90对全部字符的正确主数。
        bootstraps[label] = values_from_sums(weights @ sums)  # 每次抽样之后重新计算CER比率。
        intervals = confidence_interval(bootstraps[label])  # 绝对分数的探索性区间。
        estimates[label] = {metric: {'estimate': float(points[label][i]), 'cluster_bootstrap_95_percentile_ci': intervals[i].tolist()} for i, metric in enumerate(METRICS)}  # DNSMOS四列完整保留。
        estimates[label]['CER_total_errors'] = sum(row['char_errors'] for row in rows)  # 保留原整数分子。
        estimates[label]['CER_total_reference_chars'] = sum(row['reference_chars'] for row in rows)  # 保留原整数分母。
        per_source[label] = [{'source_speaker': cluster[0], 'source_utt': cluster[1], 'pairs': 9, 'CER_total_errors': int(sums[i, 0]), 'CER_total_reference_chars': int(sums[i, 1]), **dict(zip(METRICS, values_from_sums(sums[i]).tolist()))} for i, cluster in enumerate(clusters)]  # 教授可查看改善是否只来自个别源句。
    differences = []  # 每组固定定义为后列系统减前列系统，正负含义不混淆。
    for reference, candidate in itertools.combinations(SYSTEMS, 2):  # 四系统六组比较全部报告，避免挑有利组合。
        difference = points[candidate] - points[reference]  # 同一批配对的原始分数差。
        interval = confidence_interval(bootstraps[candidate] - bootstraps[reference])  # 同次簇抽样先作差，再取分位数。
        differences.append({'reference_system': reference, 'candidate_system': candidate, 'delta_definition': 'candidate minus reference', 'metrics': {metric: {'delta': float(difference[i]), 'cluster_bootstrap_95_percentile_ci': interval[i].tolist(), 'favorable_delta_sign': 'negative' if metric == 'CER_corpus_percent' else 'positive', 'unit': 'percentage_points' if metric in ('CER_corpus_percent', 'SVA_percent') else 'score'} for i, metric in enumerate(METRICS)}})  # CER和SVA差值是百分点。
    result = {  # 开发比较与正式论文结果严格分开。
        'status': 'complete_development_summary_only', 'systems': estimates, 'paired_comparisons': differences, 'per_source': per_source,  # 主数、配对区间与逐源结果。
        'metric_order': list(METRICS), 'n_pairs_per_system': 90, 'n_unique_source_utterances': 10, 'n_unique_reference_utterances': 10,  # 重复来源规模如实记录。
        'bootstrap': {'method': 'paired source-utterance cluster nonparametric percentile bootstrap', 'repetitions': args.bootstrap_repetitions, 'seed': args.bootstrap_seed, 'rng': 'numpy.PCG64', 'numpy_version': np.__version__, 'clusters': [list(c) for c in clusters], 'drawn_clusters_per_repetition': 10, 'targets_kept_together_per_cluster': 9, 'confidence_level': 0.95, 'weight_matrix_sha256': hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest()},  # 可复核同一权重矩阵。
        'aggregation': {'CER': '100 * sum(char_errors) / sum(reference_chars), also inside every bootstrap replicate', 'SVA': '100 * mean(cosine_score > 0.25), strict greater-than', 'DNSMOS_primary': 'mos_ovr / OVRL', 'DNSMOS_other_columns': ['mos_sig', 'mos_bak', 'p808_mos'], 'UTMOS': 'arithmetic mean over the same 90 pairs'},  # 不混用DNSMOS不同输出头。
        'limitations': ['Only 10 source clusters; percentile intervals are exploratory and may be unstable.', 'Intervals condition on the fixed target-reference set; repeated target references are not treated as independently resampled units.', 'The 90 pairs are dependent repetitions of 10 source recordings, not 90 independent recordings.', 'All four historical baselines saw these training-source utterances; this is not an independent final unseen evaluation.', 'Multiple metrics and six contrasts are descriptive; no multiplicity-adjusted significance or p-values are claimed.', 'No comparison with paper scores is made because this development protocol differs from the paper test protocol.'],  # 不夸大统计证据或泛化范围。
        'development_exposure': contract['exposure'], 'metric_identity': metric_identity, 'source_sha256': hashes, 'script_sha256': digest(Path(__file__)),  # 保留评价器与原始结果的完整身份。
        'cer_concentration_audit': cer_concentration_audit(systems, clusters),  # 追加源句时长、目标效应和原始客观转写的只读分解。
    }  # 本次只计算已有分数的统计量。
    for path, expected in hashes.items():  # 所有读入的完成记录、CSV和生成波形在汇总期间须不变。
        if digest(path) != expected:  # 仍在修改结果时拒绝发布。
            raise RuntimeError(f'Input changed while summarizing: {path}')  # 不接受跨阶段拼接数据。
    output.mkdir(parents=True, exist_ok=False)  # 通过所有检查后才创建新汇总目录。
    with (output / 'paired_metrics.csv').open('x', encoding='utf-8', newline='') as handle:  # 每系统90行，合计360行。
        writer = csv.DictWriter(handle, fieldnames=list(systems[SYSTEMS[0]][0]))  # 原计数与所有指标列均保留。
        writer.writeheader()  # 固定字段便于后续研究核验。
        for rows in systems.values():  # 报表次序按固定四系统。
            writer.writerows(rows)  # 不改变源配对顺序。
    with (output / 'summary.json').open('x', encoding='utf-8') as handle:  # 不覆盖任何原完成汇总。
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 非有限值不能写成有效结果。
        handle.write('\n')  # 文件保留末尾换行。
    lines = ['# 固定开发集的四系统客观对照', '', '我使用相同的90个转换配对比较四个系统。它们来自10条源语音，每条转换到9个固定目标；所有系统都使用同一自训HiFiGAN。', '', '| 系统 | UTMOS | DNSMOS OVRL | CER（%） | SVA（%） |', '|---|---:|---:|---:|---:|']  # 中文报告先显示四项主数。
    for label in SYSTEMS:  # 教师、旧学生、初始化、修复学生依次排列。
        point = points[label]  # 未重采样的原始点估计。
        lines.append(f'| {label} | {point[3]:.4f} | {point[2]:.4f} | {point[0]:.4f} | {point[1]:.4f} |')  # CER来自全部字符数的比率。
    lines += ['', 'CER由总字符错误数除以总参考字符数得到。SVA保持余弦分数严格大于0.25的规则。DNSMOS以OVRL为主，SIG、BAK、P.808和声纹余弦值均完整保存在summary.json。', '', '我按源语音整组重抽样，每次把它对应的9个目标一同带入。各系统使用同一套抽样，因此区间衡量的是同配对的系统差值。完整的95%区间和每条源语音结果保存在summary.json，没有把90个配对当成90个独立样本。', '', '这里仅有10组源语音，区间存在较大不确定性；目标参考固定，也没有覆盖重新抽取目标说话人的不确定性。四个历史模型训练时见过这批语音，我把本结果用于开发排查，不将它与论文正式测试分数直接比较。', '']  # 如实交代用途与统计限制。
    old_audit = result['cer_concentration_audit']['old_student_epoch100_k1']  # 用户关心旧学生CER为何明显增大。
    concentration = old_audit['posthoc_top2_target_error_concentration']  # 事后错误贡献排序只用于说明现象。
    lines += ['## 旧学生的字符错误分布', '', '| 源语音 | mel帧数 | 生成时长（秒） | 9目标总错误/总字符 | CER（%） |', '|---|---:|---:|---:|---:|']  # 先逐源显示时长和错误数。
    for row in old_audit['by_source']:  # 完整列出10源，不只挑最差语音。
        lines.append(f"| {row['source_utt']} | {row['mel_frames']} | {row['generated_duration_seconds']:.3f} | {row['char_errors']}/{row['reference_chars']} | {row['CER_corpus_percent']:.3f} |")  # 时长来自melT×256。
    lines += ['', f"我进一步按目标说话人检查了原ASR转写。旧学生错误贡献最多的两个目标为 {'、'.join(concentration['targets'])}，共贡献 {concentration['error_count']}/{concentration['total_system_error_count']} 个字符错误，占 {concentration['share_of_system_errors_percent']:.2f}%。这些都是事后定位结果，90个配对全部保留。", '', '| 固定目标 | 教师 CER（%） | 旧学生 CER（%） | S0 CER（%） | 修复学生 CER（%） | 旧学生 SVA通过数 |', '|---|---:|---:|---:|---:|---:|']  # 比较相同目标在四系统的表现。
    target_index = {label: {row['reference_speaker']: row for row in result['cer_concentration_audit'][label]['by_target']} for label in SYSTEMS}  # 目标顺序统一。
    for target in sorted(target_index[SYSTEMS[0]]):  # 同样列出全部10目标。
        values = [target_index[label][target]['CER_corpus_percent'] for label in SYSTEMS]  # 全部从原整数计数重算。
        lines.append(f"| {target} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {values[3]:.3f} | {target_index['old_student_epoch100_k1'][target]['SVA_passes']}/9 |")  # 声纹结果用于独立交叉观察。
    lines += ['', '完整参考文本和每个目标的ASR识别文本均保留在summary.json及paired_metrics.csv。我没有试听，也没有人工修订转写。部分识别文本明显变短，只能据此说ASR未识别出完整内容；还不能直接判定生成音频在文件层面被截断。', '', '我还记录了10源句的时长与分组CER的描述性相关。样本少，且说话人、文字与目标条件同时变化，这个相关不能证明长度造成退步。若相同源、参考、BNF和声码器在其他模型上正常，而旧学生在固定目标上反复失效，可以进一步检查旧学生的条件响应；当前结果尚不能单独确定失效发生在哪一层。', '']  # 明确已证实现象与待查机制的边界。
    with (output / 'summary_CN.md').open('x', encoding='utf-8') as handle:  # 可给主线程直接引用的简短说明。
        handle.write('\n'.join(lines))  # 只新增本次汇总文档。
    print(json.dumps({'output': str(output), 'status': result['status'], 'systems': {label: dict(zip(METRICS, points[label].tolist())) for label in SYSTEMS}}, ensure_ascii=False, indent=2), flush=True)  # 展示结果，绝不触发再次评估。


if __name__ == '__main__':  # 导入纯统计函数不会自动读结果或写汇总。
    main()  # 仅在Matpool进行只读结果核验和CPU bootstrap。
