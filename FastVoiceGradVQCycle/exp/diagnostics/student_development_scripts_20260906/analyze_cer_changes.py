"""逐条核对四系统ASR转写的字符变化及同配对质量分数；不重识别、不试听。"""

import csv  # 新增逐对错误变化表，不修改原评分CSV。
import hashlib  # 先核验已冻结的只读校验辅助代码。
import importlib.util  # 复用已审核的身份校验和原指标读取函数。
import json  # 保存字符对齐、目标分布及全部原转写。
import re  # 在参考文本中定位词边界，仅定位文字位置。
import sys  # 运行前限制机器环境。
from collections import Counter  # 汇总插入、删除、替换和位置变化。
from pathlib import Path  # 真实输入和新诊断目录均位于Matpool。
from types import SimpleNamespace  # 显式传入两个候选的模型与契约身份。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一获准的运行目录。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地只允许编辑代码。
    raise RuntimeError('Run transcript audits only on the Matpool rental.')  # 不在实验室服务器执行。
sys.path.insert(0, str(ROOT))  # 导入主机身份保护。
from voicegrad.runtime_guard import assert_rented_server  # 不加载任何模型。
assert_rented_server(ROOT)  # 在CPU统计前核对机器指纹。
HELPER = ROOT / 'exp/diagnostics/student_development_scripts_20260906/summarize_development_candidate.py'  # 冻结只读候选校验实现。
HELPER_SHA = '2c08b3700f405dc9deb85e6eecc5ff63309642e15113012242f81580704078e6'  # 已完成候选汇总的原字节。
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:  # 必须先核验再导入。
    raise RuntimeError('Frozen candidate-summary helper changed.')  # 不执行身份不符的辅助代码。
module_spec = importlib.util.spec_from_file_location('_cer_change_frozen_helper', HELPER)  # 使用独立模块名。
helper = importlib.util.module_from_spec(module_spec)  # 只复用读取、哈希及数值校验函数。
module_spec.loader.exec_module(helper)  # main保护保证导入不会重新评分。
base, np = helper.base, helper.np  # 全部数组统计仅在CPU执行。
T, S, A, E = 'teacher500_k30', 'teacher_initialized_s0_k1', 'student_acdd_clean_s600', 'student_endpoint_lr1e4_s100'  # 四个系统完整标签。
EXPECTED_CANDIDATES = {  # 两个新候选的权重及训练契约均为已明确的冻结身份。
    A: {'checkpoint_sha256': 'b3eedcddc465b8cc79756f8aedcfc126eef5a807c35334a712b74c6691562882', 'training_contract_sha256': 'f9472aea9332e8c261e4045ceac6630efe8f9ba012f135d5b4aba45216b32c13', 'global_step': 600, 'epoch': 0},  # ACDD600。
    E: {'checkpoint_sha256': '447c9de695f893c336991bd21feb4274c70833ef68c1bde7fb20590eb9944920', 'training_contract_sha256': '73f7923f78cdec7d95d6e2acb7276ed25e4493eaf09ae12502bf80dd57375aba', 'global_step': 100, 'epoch': 0},  # endpoint100。
}  # 不使用latest或后续其他候选。


def align_edits(reference, hypothesis):  # 返回一条固定规则的最小编辑路径，可同时用于字符和词。
    nr, nh = len(reference), len(hypothesis)  # 两边序列长度，字符CER包含空格。
    distance = [[0] * (nh + 1) for _ in range(nr + 1)]  # 每格为两个前缀的最少编辑次数。
    ways = [[0] * (nh + 1) for _ in range(nr + 1)]  # 只区分一个或至少两个最优路径，避免大整数。
    for i in range(nr + 1):  # 空识别文本需要删除全部参考字符。
        distance[i][0], ways[i][0] = i, 1  # 边界路径唯一。
    for j in range(nh + 1):  # 空参考文本需要插入全部识别字符。
        distance[0][j], ways[0][j] = j, 1  # 另一边界路径唯一。
    for i in range(1, nr + 1):  # 按参考前缀长度填表。
        for j in range(1, nh + 1):  # 按识别前缀长度填表。
            candidates = [(distance[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]), ways[i - 1][j - 1]), (distance[i - 1][j] + 1, ways[i - 1][j]), (distance[i][j - 1] + 1, ways[i][j - 1])]  # 匹配/替换、删除、插入三种路径。
            best = min(value for value, _ in candidates)  # 原CER的Levenshtein最小值。
            distance[i][j] = best  # 保存当前最少错误数。
            ways[i][j] = min(2, sum(count for value, count in candidates if value == best))  # 标记对齐是否存在歧义。
    i, j, edits = nr, nh, []  # 从完整文本向前回溯一条确定的路径。
    while i or j:  # 回溯到两个序列都为空。
        if i and j and distance[i][j] == distance[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):  # 并列最优时优先对角线：匹配或替换。
            if reference[i - 1] != hypothesis[j - 1]:  # 相等字符不计错误。
                edits.append({'op': 'S', 'reference_index': i - 1, 'hypothesis_index': j - 1, 'reference': reference[i - 1], 'hypothesis': hypothesis[j - 1]})  # 一个字符或词被替换。
            i, j = i - 1, j - 1  # 两侧各退一位。
        elif i and distance[i][j] == distance[i - 1][j] + 1:  # 次优先删除参考字符。
            edits.append({'op': 'D', 'reference_index': i - 1, 'hypothesis_index': j, 'reference': reference[i - 1], 'hypothesis': ''})  # 识别文本缺少此字符。
            i -= 1  # 仅参考侧退一位。
        else:  # 剩余最优路径为插入识别字符。
            edits.append({'op': 'I', 'reference_index': i, 'hypothesis_index': j - 1, 'reference': '', 'hypothesis': hypothesis[j - 1]})  # reference_index表示插入前的参考间隙。
            j -= 1  # 仅识别侧退一位。
    edits.reverse()  # 以原句从左到右的顺序保存。
    return {'distance': distance[nr][nh], 'ambiguous_optimal_alignment': ways[nr][nh] > 1, 'edits': edits}  # 总数唯一，但S/D/I拆分可能依赖并列路径规则。


def locate_character_edit(reference, edit):  # 只定位标准化文字，不将字符位置当作音频时间。
    words = list(re.finditer(r'\S+', reference))  # 原参考文本的逐词起止字符位置。
    position = edit['reference_index']  # 删除/替换取参考字符，插入取参考间隙。
    anchor = min(position, len(reference) - 1)  # 句末插入归到最后一个词边缘。
    zone = 'first_word' if anchor < words[0].end() else ('last_word' if anchor >= words[-1].start() else 'interior')  # 句首末明确限定首词/末词。
    nearby = [word.group() for word in words if word.start() <= anchor < word.end()]  # 对应的参考词；空格单独标记。
    return {'zone': zone, 'reference_word_or_gap': nearby[0] if nearby else '<SPACE_OR_WORD_GAP>', 'reference_context': reference[max(0, position - 12):min(len(reference), position + 13)]}  # 保留短上下文方便人工核对。


def analyze_transcript(row):  # 每条已验证ASR结果分别做字符与词级对齐。
    reference, hypothesis = row['reference_text_normalized'], row['hypothesis_text_normalized']  # 原转写完全不改。
    characters = align_edits(reference, hypothesis)  # 字符级路径总数须与原CER精确一致。
    if characters['distance'] != row['char_errors']:  # 防止诊断用了另一套字符距离。
        raise ValueError('Edit path does not reproduce stored CER error count.')  # 不重写原CER来强行相符。
    for edit in characters['edits']:  # 为每个字符错误附上明确文字位置。
        edit.update(locate_character_edit(reference, edit))  # 无声学时间信息，不声称停顿位置。
    ref_words, hyp_words = reference.split(), hypothesis.split()  # 词级仅帮助阅读，不替代字符CER。
    words = align_edits(ref_words, hyp_words)  # 记录REWARD变为WERE BROAD等词边界变化。
    adjacent = [hyp_words[i] for i in range(1, len(hyp_words)) if hyp_words[i] == hyp_words[i - 1]]  # 直接记录相邻重复词。
    extra_repeat = {word: count - Counter(ref_words)[word] for word, count in Counter(hyp_words).items() if count > 1 and count > Counter(ref_words)[word]}  # 非相邻重复只是描述，可能含常见功能词。
    return {'character_alignment': characters, 'word_alignment_for_description_only': words, 'S_D_I': {op: sum(edit['op'] == op for edit in characters['edits']) for op in ('S', 'D', 'I')}, 'zones': dict(Counter(edit['zone'] for edit in characters['edits'])), 'adjacent_repeated_hypothesis_words': adjacent, 'extra_repeated_hypothesis_word_counts': extra_repeat}  # 不用重复词标记自动判定真实口吃。


def operation_multiset(analysis):  # 按固定对齐路径比较新增/消失的文字编辑事件。
    return Counter((edit['op'], edit['reference_index'], edit['reference'], edit['hypothesis'], edit['zone'], edit['reference_word_or_gap']) for edit in analysis['character_alignment']['edits'])  # 包含参考位置，避免同字不同处被合并。


def compare_systems(reference_label, candidate_label, systems, analyses):  # 比较相同pair_id的ASR与质量分数，不筛样。
    rows = []  # 完整保留90对的变化，包括错误数不变但识别文字改变的情况。
    for reference, candidate in zip(systems[reference_label], systems[candidate_label]):  # 两系统已核验同一模板与顺序。
        i = reference['pair_id']  # 全流程固定的配对编号。
        before, after = analyses[reference_label][i], analyses[candidate_label][i]  # 各自的固定字符对齐。
        added = operation_multiset(after) - operation_multiset(before)  # 新出现的编辑事件，不等同于真实声学错误。
        removed = operation_multiset(before) - operation_multiset(after)  # 原编辑事件的消失。
        rows.append({  # 逐对保留净变化与可检查文字证据。
            'pair_id': i, 'source_utt': reference['source_utt'], 'reference_speaker': reference['reference_speaker'], 'duration_seconds': reference['generated_duration_seconds'],  # 相同源句与目标条件。
            'reference_text': reference['reference_text_normalized'], 'before_hypothesis': reference['hypothesis_text_normalized'], 'after_hypothesis': candidate['hypothesis_text_normalized'],  # 真实ASR文本，不作人工纠错。
            'before_errors': reference['char_errors'], 'after_errors': candidate['char_errors'], 'error_delta': candidate['char_errors'] - reference['char_errors'], 'reference_chars': reference['reference_chars'],  # 总CER仍按全部字符相除。
            'operation_count_delta': {op: after['S_D_I'][op] - before['S_D_I'][op] for op in ('S', 'D', 'I')},  # 固定回溯规则下S/D/I数的变化。
            'added_alignment_events': [{'op': event[0], 'reference_index': event[1], 'reference': event[2], 'hypothesis': event[3], 'zone': event[4], 'reference_word_or_gap': event[5], 'count': count} for event, count in added.items()],  # 增加事件可能伴随同位置另一个事件消失。
            'removed_alignment_events': [{'op': event[0], 'reference_index': event[1], 'reference': event[2], 'hypothesis': event[3], 'zone': event[4], 'reference_word_or_gap': event[5], 'count': count} for event, count in removed.items()],  # 不能把事件增加总数冒充净错误增量。
            'UTMOS_delta': candidate['utmos'] - reference['utmos'], 'DNSMOS_OVRL_delta': candidate['mos_ovr'] - reference['mos_ovr'], 'SVA_cosine_delta': candidate['cosine_score'] - reference['cosine_score'],  # 定位减少CER时质量代价出现在哪些样本。
            'before_UTMOS': reference['utmos'], 'after_UTMOS': candidate['utmos'], 'before_DNSMOS_OVRL': reference['mos_ovr'], 'after_DNSMOS_OVRL': candidate['mos_ovr'],  # 保留绝对分数以免只看差值。
        })  # 所有90对都进入统计。
    groups = {}  # 从源句和目标两种视角定位变化。
    for field in ('source_utt', 'reference_speaker'):  # 不能预先认定变化只属于目标条件。
        groups[field] = []  # 固定10组，全部报告。
        for value in sorted({row[field] for row in rows}):  # 名字排序便于横向核对。
            subset = [row for row in rows if row[field] == value]  # 每组恰好9对。
            groups[field].append({field: value, 'pairs': len(subset), 'before_errors': sum(row['before_errors'] for row in subset), 'after_errors': sum(row['after_errors'] for row in subset), 'error_delta': sum(row['error_delta'] for row in subset), 'worse_pairs': sum(row['error_delta'] > 0 for row in subset), 'better_pairs': sum(row['error_delta'] < 0 for row in subset), 'UTMOS_delta_mean': float(np.mean([row['UTMOS_delta'] for row in subset])), 'DNSMOS_OVRL_delta_mean': float(np.mean([row['DNSMOS_OVRL_delta'] for row in subset]))})  # 只作描述性分解，不重定义主数。
    quality_by_cer = {}  # 看CER改善的语音是否同时得到质量预测器改善。
    for label, predicate in [('CER_decreased', lambda value: value < 0), ('CER_unchanged', lambda value: value == 0), ('CER_increased', lambda value: value > 0)]:  # 三组完整覆盖90对。
        subset = [row for row in rows if predicate(row['error_delta'])]  # 分组依据是已有结果，仅作事后描述。
        quality_by_cer[label] = {'pairs': len(subset), 'error_delta_sum': sum(row['error_delta'] for row in subset), 'UTMOS_delta_mean': float(np.mean([row['UTMOS_delta'] for row in subset])) if subset else None, 'DNSMOS_OVRL_delta_mean': float(np.mean([row['DNSMOS_OVRL_delta'] for row in subset])) if subset else None, 'UTMOS_decreased_pairs': sum(row['UTMOS_delta'] < 0 for row in subset), 'DNSMOS_OVRL_decreased_pairs': sum(row['DNSMOS_OVRL_delta'] < 0 for row in subset)}  # 无数据时明确null。
    added_zones, removed_zones, added_words, removed_words = Counter(), Counter(), Counter(), Counter()  # 编辑事件的位置与词语聚合。
    for row in rows:  # 对净变化为零的改写也保留两侧事件。
        for key, zones, words in [('added_alignment_events', added_zones, added_words), ('removed_alignment_events', removed_zones, removed_words)]:  # 增加与减少不混为一个数。
            for event in row[key]:  # 位置依赖固定的最优路径规则。
                zones[event['zone']] += event['count']  # 首词、末词与句内分开统计。
                words[event['reference_word_or_gap']] += event['count']  # 同源多目标共有错误可据此看出。
    return {  # 主数及所有分解均可追溯到原90对。
        'reference_system': reference_label, 'candidate_system': candidate_label, 'net_error_delta': sum(row['error_delta'] for row in rows), 'gross_increase_on_worsened_pairs': sum(max(row['error_delta'], 0) for row in rows), 'gross_reduction_on_improved_pairs': sum(max(-row['error_delta'], 0) for row in rows),  # 例如净+13可能由大幅新增与减少抵消。
        'pairs': rows, 'groups': groups, 'quality_by_CER_change': quality_by_cer, 'operation_count_delta': {op: sum(row['operation_count_delta'][op] for row in rows) for op in ('S', 'D', 'I')},  # 完整S/D/I变化。
        'added_alignment_event_zones': dict(added_zones), 'removed_alignment_event_zones': dict(removed_zones), 'added_alignment_event_reference_words': dict(added_words.most_common()), 'removed_alignment_event_reference_words': dict(removed_words.most_common()),  # 不能把事件数当作独立语音错误数。
        'largest_UTMOS_decreases': sorted(rows, key=lambda row: (row['UTMOS_delta'], row['pair_id']))[:10], 'largest_DNSMOS_OVRL_decreases': sorted(rows, key=lambda row: (row['DNSMOS_OVRL_delta'], row['pair_id']))[:10],  # 保留具体质量代价最大的样本，主数仍全90对。
    }  # 无任何重新评分、文本替换或样本排除。


def main():  # 只消费冻结的客观评估结果。
    output = base.DEV / 'cer_transcript_change_audit_v1'  # 独立追加诊断目录。
    if output.exists():  # 既有证据不覆盖。
        raise RuntimeError('Transcript audit already exists.')  # 重新分析需另行明确版本。
    old, template, development, metric_identity, hashes = base.load_results(base.DEV / 'evaluation_baselines_v1')  # 完整核验原指标CSV、波形和凭据。
    systems = {T: old[T], S: old[S]}  # 当前问题只比较教师、S0及两个新候选。
    for label, identity in EXPECTED_CANDIDATES.items():  # 两个候选均绑定真实权重与训练契约。
        args = SimpleNamespace(label=label, candidate_root=base.DEV / ('evaluation_candidate_' + label), **identity)  # 显式身份无需加载模型。
        systems[label], _, _, _ = helper.load_candidate(args, template, development, metric_identity, hashes)  # 验证同90对、指标哈希、字符计数和52句排除。
    if {label: sum(row['char_errors'] for row in rows) for label, rows in systems.items()} != {T: 124, S: 110, A: 123, E: 101}:  # 与此前已核验的总计相符。
        raise ValueError('Unexpected frozen CER totals.')  # 防止误读其他批次。
    analyses = {label: [analyze_transcript(row) for row in rows] for label, rows in systems.items()}  # 原360条转写全部分析。
    summaries = {}  # 每系统的操作分类及拼写/重复文字现象。
    for label, rows in systems.items():  # 各模型用相同规则，避免某系统单独清洗。
        summaries[label] = {'errors': sum(row['char_errors'] for row in rows), 'reference_chars': sum(row['reference_chars'] for row in rows), 'S_D_I': {op: sum(item['S_D_I'][op] for item in analyses[label]) for op in ('S', 'D', 'I')}, 'ambiguous_character_alignment_pairs': sum(item['character_alignment']['ambiguous_optimal_alignment'] for item in analyses[label]), 'adjacent_repetition_pairs': [i for i, item in enumerate(analyses[label]) if item['adjacent_repeated_hypothesis_words']], 'favourites_spelled_favorites_pairs': [row['pair_id'] for row in rows if 'FAVOURITES' in row['reference_text_normalized'].split() and 'FAVORITES' in row['hypothesis_text_normalized'].split()], 'last_word_FORE_as_FOR_or_FOUR_pairs': [row['pair_id'] for row in rows if row['reference_text_normalized'].endswith(' FORE') and row['hypothesis_text_normalized'].split()[-1:] in (['FOR'], ['FOUR'])]}  # 拼写现象保留在原CER中，不自动纠正。
    comparisons = {'ACDD600_minus_S0': compare_systems(S, A, systems, analyses), 'endpoint100_minus_ACDD600': compare_systems(A, E, systems, analyses), 'endpoint100_minus_S0': compare_systems(S, E, systems, analyses), 'teacher30_minus_S0': compare_systems(S, T, systems, analyses)}  # 用户关心的四种同配对变化。
    result = {  # 字符计数确定，音频机制仍需另外证明。
        'status': 'complete_readonly_transcript_change_audit', 'systems': summaries, 'comparisons': comparisons, 'per_pair_original_metrics': systems, 'per_pair_alignment': analyses,  # 全360条原文本、对齐和质量分数保留。
        'alignment_convention': 'Levenshtein unit cost including spaces; backtrace prioritizes diagonal match/substitution, then deletion, then insertion. S/D/I decomposition and exact event positions can be nonunique when optimal paths tie; total CER remains unique.',  # 不把一个算法路径当作唯一真实错音。
        'word_position_convention': 'first_word and last_word mean the first/last whitespace-separated reference word; other positions are interior. Character indices are not acoustic timestamps.',  # 不从文字位置推断停顿时间。
        'limitations': ['No audio was listened to or re-recognized; analyses use original cached ASR hypotheses and reference texts.', 'ASR spelling, lexical segmentation and sensitivity to timbre may contribute; these transcripts alone do not prove that the generated audio contains the same lexical or phonetic errors.', 'Pauses and phoneme timing cannot be located from these transcripts. Silence, repeated audio or truncation need independent waveform/forced-alignment evidence.', 'All 90 development pairs remain in CER and quality totals; no subset is selected for training, no scoring threshold/text normalization is changed, and no formal 8100 output is used for selection.', 'Quality differences within CER-improved/worsened groups are posthoc descriptive summaries, not independent statistical experiments.'],  # 约束可操作判断的证据边界。
        'source_sha256': hashes, 'frozen_helper_sha256': HELPER_SHA, 'script_sha256': base.digest(Path(__file__)),  # 所有读入结果和代码有身份记录。
    }  # 不将训练GTmel告警写作音质结论。
    for path, expected in hashes.items():  # 原CSV、wav、权重及完成凭据读取前后必须相同。
        if base.digest(path) != expected:  # 读到并发变化就停止。
            raise RuntimeError(f'Input changed during transcript audit: {path}')  # 不混合不同运行数据。
    output.mkdir(exist_ok=False)  # 通过所有检查才创建新诊断目录。
    with (output / 'transcript_audit.json').open('x', encoding='utf-8') as handle:  # 完整证据单独保存。
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 非有限数不进入报告。
        handle.write('\n')  # 便于文本工具读取。
    fields = ('comparison', 'pair_id', 'source_utt', 'reference_speaker', 'before_errors', 'after_errors', 'error_delta', 'UTMOS_delta', 'DNSMOS_OVRL_delta', 'reference_text', 'before_hypothesis', 'after_hypothesis')  # 简短CSV用于定位具体配对。
    with (output / 'pair_changes.csv').open('x', encoding='utf-8', newline='') as handle:  # 所有四组×90对都保留。
        writer = csv.DictWriter(handle, fieldnames=fields)  # 不将嵌套对齐信息塞进CSV。
        writer.writeheader()  # JSON保存完整操作路径。
        for name, comparison in comparisons.items():  # 每组同一pair的文字变化可直接查阅。
            for row in comparison['pairs']:  # 不仅导出错误变化较大的样本。
                writer.writerow({'comparison': name, **{key: row[key] for key in fields if key != 'comparison'}})  # 保留原文本与各指标差值。
    lines = ['# 四系统ASR转写变化检查', '', '我逐条核对了同一组90个配对的原始ASR转写，并从保存文本重新计算字符编辑距离。全部360条转写与原CER计数一致，没有修改文本、评分规则或配对。', '', '| 系统 | 字符错误 | 替换S | 删除D | 插入I |', '|---|---:|---:|---:|---:|']  # 先明确总数及拆分。
    for label, item in summaries.items():  # 四系统完整展示。
        lines.append(f"| {label} | {item['errors']}/4392 | {item['S_D_I']['S']} | {item['S_D_I']['D']} | {item['S_D_I']['I']} |")  # S/D/I总和必须等于原错误数。
    lines += ['', 'S/D/I采用固定最小编辑路径，若有多条同样短的路径，拆分和定位可能不同；字符错误总数不受这个选择影响。下方句首末仅表示参考文本首词和末词，不能当作音频时间。', '', '## ACDD600相对S0', '']  # 明确路径歧义和定位范围。
    comp = comparisons['ACDD600_minus_S0']  # 净+13的真实来源。
    lines += [f"ACDD共增加{comp['gross_increase_on_worsened_pairs']}个错误，同时减少{comp['gross_reduction_on_improved_pairs']}个错误，抵消后净增加{comp['net_error_delta']}个。", '', '| 源语音 | S0错误 | ACDD错误 | 净变化 | 变差配对 | 变好配对 |', '|---|---:|---:|---:|---:|---:|']  # 不把净+13误写成仅13个事件。
    for row in comp['groups']['source_utt']:  # 全部10源，不挑样本。
        lines.append(f"| {row['source_utt']} | {row['before_errors']} | {row['after_errors']} | {row['error_delta']:+d} | {row['worse_pairs']} | {row['better_pairs']} |")  # 每源9个目标合计。
    lines += ['', '| 目标说话人 | S0错误 | ACDD错误 | 净变化 |', '|---|---:|---:|---:|']  # 目标效应与源句效应分开展示。
    for row in comp['groups']['reference_speaker']:  # 全部10目标。
        lines.append(f"| {row['reference_speaker']} | {row['before_errors']} | {row['after_errors']} | {row['error_delta']:+d} |")  # 原错误计数不改变。
    lines += ['', '## endpoint100相对ACDD600', '']  # CER降低时同步检查质量分数。
    comp = comparisons['endpoint100_minus_ACDD600']  # 两候选真实逐对变化。
    lines += [f"endpoint共增加{comp['gross_increase_on_worsened_pairs']}个错误，同时减少{comp['gross_reduction_on_improved_pairs']}个错误，净变化{comp['net_error_delta']}个。", '', '| 事后分组 | 配对数 | 净错误变化 | 平均ΔUTMOS | 平均ΔDNSMOS OVRL |', '|---|---:|---:|---:|---:|']  # 分组只作定位，不作为新的主指标。
    for name, row in comp['quality_by_CER_change'].items():  # 改善、不变、变差三组全部列出。
        lines.append(f"| {name} | {row['pairs']} | {row['error_delta_sum']:+d} | {row['UTMOS_delta_mean']:.4f} | {row['DNSMOS_OVRL_delta_mean']:.4f} |")  # 指标来自已完成评分器，不使用听感。
    lines += ['', '| 源语音 | Δ字符错误 | 平均ΔUTMOS | 平均ΔDNSMOS OVRL |', '|---|---:|---:|---:|']  # 查看质量代价是否只集中少数源句。
    for row in comp['groups']['source_utt']:  # 每源9目标等权质量均值。
        lines.append(f"| {row['source_utt']} | {row['error_delta']:+d} | {row['UTMOS_delta_mean']:.4f} | {row['DNSMOS_OVRL_delta_mean']:.4f} |")  # 字符计数仍合并求和。
    lines += ['', '## 文字证据与后续判断的范围', '', '完整参考、各系统识别文本、每个字符操作和质量变化最大的配对都保存在transcript_audit.json，pair_changes.csv保留全部配对。文本中的拼写、分词或识别词变化可能来自ASR对音色和声学细节的敏感性；目前不能仅凭它们确定真实音频说错了同样的词。', '', '我没有用这份结果更换开发语音或改CER参考文本。停顿、真实重复发音和音频截断尚无时间对齐证据，本次不据转写作出肯定判断。', '']  # 不将客观识别结果扩大为已证实的声学根因。
    with (output / 'transcript_audit_CN.md').open('x', encoding='utf-8') as handle:  # 先保存自动可核算的证据表。
        handle.write('\n'.join(lines))  # 简短人工解释随后可另写独立说明。
    print(json.dumps({'output': str(output), 'systems': summaries, 'comparisons': {name: {key: comp[key] for key in ('net_error_delta', 'gross_increase_on_worsened_pairs', 'gross_reduction_on_improved_pairs', 'operation_count_delta', 'added_alignment_event_zones', 'removed_alignment_event_zones', 'added_alignment_event_reference_words', 'removed_alignment_event_reference_words', 'quality_by_CER_change', 'groups')} for name, comp in comparisons.items()}}, ensure_ascii=False, indent=2), flush=True)  # 向主线程报告可验证计数。


if __name__ == '__main__':  # 导入对齐函数测试时不读取真实指标。
    main()  # 显式调用才进行租用CPU上的只读诊断。
