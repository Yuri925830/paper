"""只在 Matpool 建立独立的学生开发清单；不改原数据，不加载模型。"""

import argparse  # 输出位置必须明确，已有开发记录不能被覆盖。
import csv  # 读取权威训练名单，写出沿用原字段的子集名单。
import hashlib  # 固定选择顺序，并登记输入文件的字节身份。
import json  # 保存可供训练与评估共同核验的开发协议。
import random  # 重放旧 32 句诊断的裁剪位置。
import re  # 只用于排除已知重复文本，不替代 CER 的文本处理。
import subprocess  # 只读记录生成清单时服务器的代码版本。
import sys  # 先拒绝在本地 Windows 导入计算库。
import time  # 记录清单生成时间，不影响样本选择。
import wave  # 只读检查原音频头部，不解码或合成语音。
from collections import Counter, defaultdict  # 统计文本重复次数和各说话人的候选句。
from pathlib import Path  # 限定权威输入与独立输出目录。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一获准进行项目计算的出租服务器目录。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地只能编辑此脚本。
    raise RuntimeError('Run only on the registered Matpool rental.')  # 路径或系统不符即停止。
sys.path.insert(0, str(ROOT))  # 导入服务器现场的主机身份保护。
from voicegrad.runtime_guard import assert_rented_server  # 进一步核对主机名及机器指纹。
assert_rented_server(ROOT)  # 在 NumPy 导入和特征读取之前完成主机身份核验。
import numpy as np  # 仅在出租机用 CPU 检查少量已选特征的形状和有限性。

SELECTION_SEED = 20260906  # 一次确定的开发集选择编号，绝不按指标重新抽取。
PAIR_NOISE_BASE = 2026090600  # 同一 pair 的各模型使用同一输入噪声种子。
TRAIN = ROOT / 'data/metadata/train_manifest.csv'  # 原始 39,387 句训练清单。
TEST = ROOT / 'data/metadata/test_manifest.csv'  # 只读身份信息，以核对语句和说话人不重叠。
PILOT = ROOT / 'exp/diagnostics/acdd_repair_20260906/diagnostic_manifest.json'  # 旧 32 句的索引来源。
OLD_RESULTS = ROOT / 'exp/diagnostics/student_stop_audit_20260906/results.json'  # 旧固定输入的张量指纹。
MEAN = ROOT / 'data/stats/mel_mean.npy'  # 保留教师原有的逐频带统计量。
STD = ROOT / 'data/stats/mel_std.npy'  # 此开发划分不重算统计量，也不改变教师输入。
EXPECTED = {  # 固定已核实的来源；任何变化都需要人工重新审核。
    TRAIN: '6b24abba2bbea1102ebd002980ec9e4ded9357a07e3871d2514c2474c6774e2b',  # 权威训练名单。
    TEST: 'dec09943585580c782634900956ef80f2e9e8286ae8db539eebf0a2b2e7471a4',  # 正式测试名单。
    PILOT: 'd4b83766559d8535f03db6de1bd0e5f69063215968eb2dd8d787605d7f2adc98',  # 原诊断样本选择。
    OLD_RESULTS: 'ee34a05bca5350a45a08fcaa73802f89adc68a683efcc06128d4da2dd1697912',  # 原输入哈希与结果。
    MEAN: '4a858ffc0ea5fdba37751a5925dd6213a8cd87ece14978cda39606ca49f2cc31',  # 原均值字节。
    STD: '9e5d30a98a2fa2d65f63ae0d853bd404b716af5d90f519eb0ad06209c7a0773b',  # 原标准差字节。
}  # 哈希针对原文件，不因新增开发清单而更新。


def file_sha(path):  # 分块读取文件，避免一次把音频全放入内存。
    digest = hashlib.sha256()  # 每个文件独立计算 SHA-256。
    with Path(path).open('rb') as handle:  # 所有来源均以只读方式打开。
        for block in iter(lambda: handle.read(1024 * 1024), b''):  # 最多每次读取 1 MiB。
            digest.update(block)  # 保留文件原始字节的顺序。
    return digest.hexdigest()  # 返回适合写入 JSON/CSV 的十六进制值。


def canonical_sha(value):  # 此指纹表示 JSON 内容身份，与缩进后的文件字节哈希分开。
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))  # 固定字段顺序与分隔符。
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()  # 不把文件末尾换行算入内容指纹。


def text_identity(text):  # 选择阶段只判断已知文本是否重复，不据此计算 CER。
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split())  # 去除标点差异并合并空格。


def selection_rank(label):  # 用 SHA 排序可避免不同随机数库版本造成抽样差异。
    return hashlib.sha256(f'{SELECTION_SEED}:{label}'.encode('utf-8')).hexdigest()  # 相同标签始终得到相同次序。


def read_csv(path):  # 不改变原清单的行顺序，行索引由此定义。
    with path.open('r', encoding='utf-8', newline='') as handle:  # 正确读取包含逗号的原转写。
        reader = csv.DictReader(handle)  # 首行字段名作为每条记录的键。
        return list(reader.fieldnames or []), list(reader)  # 同时保留原字段次序与所有行。


def save_json_new(path, value):  # 排除意外覆盖旧实验的可能。
    with path.open('x', encoding='utf-8') as handle:  # x 模式要求文件尚不存在。
        json.dump(value, handle, ensure_ascii=False, indent=2)  # 中文解释原样保存。
        handle.write('\n')  # 文件结尾保留换行，便于人工阅读。


def save_csv_new(path, fields, rows):  # 子集仍用原 CSV 字段，不能伪造新的官方 split。
    with path.open('x', encoding='utf-8', newline='') as handle:  # 新文件独立保存。
        writer = csv.DictWriter(handle, fieldnames=fields)  # 明确输出列顺序。
        writer.writeheader()  # 让后续工具按字段名读取。
        writer.writerows(rows)  # 输入顺序决定输出顺序，不另外打乱。


def data_path(path):  # 所有特征必须来自本项目 data 目录中的现存文件。
    resolved = Path(path).resolve()  # 符号链接也检查真实落点。
    if ROOT / 'data' not in resolved.parents or not resolved.is_file():  # 防止误读其他实验或失效路径。
        raise RuntimeError(f'Unexpected or missing data asset: {path}')  # 不以其他句子替代损坏样本。
    return resolved  # 配对清单统一写真实绝对路径。


def inspect_utterance(index, row, source_hashes):  # 仅检查已选的 52 句，不扫描全库特征。
    speaker, utterance = row['speaker'], row['utt_id']  # 从原名单获取唯一语音身份。
    paths = {  # 四种资产须属于同一句语音。
        'wav': data_path(row['wav22k_path']),  # 原 22.05 kHz 录音。
        'mel': data_path(ROOT / f'data/mel/{speaker}/{utterance}.npy'),  # 80 维原始 log-mel。
        'bnf': data_path(ROOT / f'data/bnf/{speaker}/{utterance}.npy'),  # 已缓存的 144 维内容特征。
        'spk_emb': data_path(ROOT / f'data/spk_emb/{speaker}/{utterance}.npy'),  # 256 维 GE2E 条件。
    }  # 不重新提特征，不更换 BNF 或说话人模型。
    hashes = {key: file_sha(path) for key, path in paths.items()}  # 在检查内容前先登记字节。
    arrays = {key: np.load(paths[key], allow_pickle=False, mmap_mode='r') for key in ('mel', 'bnf', 'spk_emb')}  # 仅读取数值数组。
    for key, dimensions in [('mel', 80), ('bnf', 144)]:  # 缓存必须是维度在前、时间在后的已知格式。
        value = arrays[key]  # 取当前一种特征的只读视图。
        if value.ndim != 2 or value.shape[0] != dimensions or value.shape[1] < 1 or not np.isfinite(value).all():  # 拒绝异常形状或非有限值。
            raise RuntimeError(f'Invalid selected {key}: {paths[key]} {value.shape}')  # 留下具体坏资产路径。
    embedding = arrays['spk_emb']  # 只检查条件向量，不对其重新归一化。
    if embedding.size != 256 or not np.isfinite(embedding).all() or float(np.linalg.norm(embedding)) == 0:  # 零向量不能作为有效说话人条件。
        raise RuntimeError(f'Invalid selected GE2E embedding: {paths["spk_emb"]}')  # 维数或内容不符则停止。
    with wave.open(str(paths['wav']), 'rb') as handle:  # 音频头部给出无需解码的长度与格式。
        audio = {'sample_rate': handle.getframerate(), 'channels': handle.getnchannels(), 'sample_width': handle.getsampwidth(), 'samples': handle.getnframes()}  # 记录原格式。
    mel_frames = int(arrays['mel'].shape[1])  # 推理的目标长度直接由缓存 mel 决定。
    if (audio['sample_rate'], audio['channels'], audio['sample_width']) != (22050, 1, 2):  # 与原 PCM16 单声道预处理协议核对。
        raise RuntimeError(f'Unexpected waveform header: {paths["wav"]} {audio}')  # 禁止临时重采样掩盖差异。
    if mel_frames != audio['samples'] // 256:  # 原提取流程按 hop=256 截取整数帧。
        raise RuntimeError(f'Wave/mel frame rule changed: {paths["wav"]}')  # 边界规则变化需要单独排查。
    for key, path in paths.items():  # 检查读取过程中没有其他任务改写输入。
        if file_sha(path) != hashes[key]:  # 前后哈希须完全一致。
            raise RuntimeError(f'Input changed while reading: {path}')  # 不保存混合版本的开发清单。
        source_hashes[str(path)] = hashes[key]  # 供整个生成过程结束前再次核对。
    return {  # 返回可复用的语句描述，与任何待比较模型无关。
        'train_index': index, 'speaker': speaker, 'utt_id': utterance, 'gender': row['gender'], 'text': row['text'],  # 原名单身份。
        **{key: str(path) for key, path in paths.items()},  # 真实数据路径。
        **{f'{key}_sha256': value for key, value in hashes.items()},  # 每个文件独立字节身份。
        'mel_frames': mel_frames, 'bnf_frames': int(arrays['bnf'].shape[1]), 'wav_header': audio,  # 时间长度及录音格式。
    }  # 此处没有模型预测、损失或语音质量筛选。


def main():  # 生成一套只追加、不替换的开发证据。
    parser = argparse.ArgumentParser(description=__doc__)  # 命令行仅允许选择新诊断目录。
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'exp/diagnostics/student_development_20260906')  # 默认固定本次开发协议目录。
    args = parser.parse_args()  # 不提供改变种子或样本数的临时开关。
    output = args.output_dir.resolve()  # 写入范围按真实路径核对。
    if ROOT / 'exp/diagnostics' not in output.parents or output.exists():  # 必须位于独立诊断区且目录尚不存在。
        raise RuntimeError('Use a new directory inside exp/diagnostics; never overwrite a development set.')  # 旧清单留作可追溯证据。
    for path, expected in EXPECTED.items():  # 先检查全部固定来源，再选择语音。
        if file_sha(path) != expected:  # 任一名单、旧诊断或统计量变化都会停止。
            raise RuntimeError(f'Authoritative input identity changed: {path}')  # 不自动接受新文件。
    fields, train = read_csv(TRAIN)  # 保存原索引顺序，禁止排序后重新编号。
    _, test = read_csv(TEST)  # 仅核对正式测试名单身份，不读取测试音频、指标或特征。
    if len(train) != 39387 or len({row['speaker'] for row in train}) != 100 or len(test) != 100:  # 与已登记规模核对。
        raise RuntimeError('Unexpected authoritative train/test split size.')  # 划分变化需要重新审核。
    if any(row['split'] != 'train' or row['is_unseen_speaker'] != '0' for row in train):  # 新开发语音全部来自原 seen 说话人训练集。
        raise RuntimeError('The development pool must be the original training split.')  # 不混入正式 unseen。
    train_ids = [(row['speaker'], row['utt_id']) for row in train]  # 语句身份由说话人与句号共同确定。
    test_ids = {(row['speaker'], row['utt_id']) for row in test}  # 测试信息只用来验证不重叠。
    if len(set(train_ids)) != len(train_ids) or set(train_ids) & test_ids or {r['speaker'] for r in train} & {r['speaker'] for r in test}:  # 同时检查语句与说话人划分。
        raise RuntimeError('Duplicate training identity or overlap with the formal test split.')  # 不在污染划分上继续。
    pilot = json.loads(PILOT.read_text(encoding='utf-8'))  # 只取旧 pilot 的开发索引，不启动旧训练程序。
    reconstruction = pilot['dev_indices']  # 保留原 32 句的批内顺序。
    if len(reconstruction) != 32 or len(set(reconstruction)) != 32 or any(type(i) is not int or not 0 <= i < len(train) for i in reconstruction):  # 索引必须有效且唯一。
        raise RuntimeError('Expected the archived 32 unique reconstruction indices.')  # 不重新抽取旧固定开发批次。
    old = json.loads(OLD_RESULTS.read_text(encoding='utf-8'))['input_identities']['training_source_32']  # 只保留旧训练来源32句的身份，不使用正式测试指标。
    if len(old) != 1 or old[0]['utterances'] != [train[i]['utt_id'] for i in reconstruction] or old[0]['speakers'] != [train[i]['speaker'] for i in reconstruction]:  # 与旧记录逐句核对。
        raise RuntimeError('Archived input order does not match pilot indices.')  # 相同集合但不同顺序也不接受。
    normalized = [text_identity(row['text']) for row in train]  # 仅根据原训练文本选择跨说话人配对。
    counts = Counter(normalized)  # 已知全文重复的句子不作新的跨说话人源/参考。
    candidates = defaultdict(list)  # 将符合预先规则的语句按说话人归组。
    for index, row in enumerate(train):  # 遍历 CSV 元数据，不扫描全部音频。
        text = normalized[index]  # 当前原转写的去标点身份串。
        suffix = row['utt_id'].rsplit('_', 1)[-1]  # 使用 VCTK 句号避开前面的通用朗读段落。
        if index not in reconstruction and counts[text] == 1 and 8 <= len(text.split()) <= 35 and suffix.isdigit() and int(suffix) >= 25:  # 长度适中、文本完整且与旧32分开。
            candidates[row['speaker']].append(index)  # 入选条件完全不涉及模型输出和指标。
    selected_speakers = []  # 固定五位女性和五位男性，减少小集合的性别偏斜。
    for gender in ('F', 'M'):  # 两组均按相同的 SHA 排序规则选择。
        eligible = [speaker for speaker, indices in candidates.items() if len(indices) >= 2 and {train[i]['gender'] for i in indices} == {gender}]  # 每人需两句不同语音。
        if len(eligible) < 5:  # 候选不足时不偷偷放宽规则。
            raise RuntimeError(f'Not enough eligible speakers for gender {gender}')  # 要求重新审核开发定义。
        selected_speakers.extend(sorted(eligible, key=lambda speaker: (selection_rank('speaker:' + speaker), speaker))[:5])  # 一次固定选择。
    selected_speakers.sort()  # pair_id 按说话人名称排列，便于人工核对。
    roles = {}  # 每人固定一条源句与一条参考句，所有模型共用。
    for speaker in selected_speakers:  # 不随转换目标更换源句。
        chosen = sorted(candidates[speaker], key=lambda i: (selection_rank('utterance:' + train[i]['utt_id']), i))[:2]  # SHA 排第一作源，第二作参考。
        roles[speaker] = {'source_train_index': chosen[0], 'reference_train_index': chosen[1]}  # 记录原清单索引。
    cross_indices = sorted({index for role in roles.values() for index in role.values()})  # 90 对实际只涉及 20 条原录音。
    excluded = sorted(set(reconstruction) | set(cross_indices))  # 新学生训练必须排除源、参考和自重建的全部语句。
    retained = [i for i in range(len(train)) if i not in set(excluded)]  # 剩余索引保持原 CSV 次序。
    if len(cross_indices) != 20 or len(excluded) != 52 or len(retained) != 39335 or len({train[i]['speaker'] for i in retained}) != 100:  # 保持原100说话人仍参与学生训练。
        raise RuntimeError('Unexpected development exclusion size or speaker coverage.')  # 排除规模不符即停止。
    source_hashes = {str(path): expected for path, expected in EXPECTED.items()}  # 开始登记全套只读来源。
    assets = {index: inspect_utterance(index, train[index], source_hashes) for index in excluded}  # 仅检查52句缓存与wav头部。
    crop_random = random.Random(1234)  # 原 pilot 在读取32句前设置的同一个裁剪随机流。
    self_rows = []  # 以原批内次序写出32条自重建记录。
    for index in reconstruction:  # 长句才消费一次 randint，与原 Dataset 一致。
        asset = dict(assets[index])  # 同时附带各特征哈希与语句身份。
        frames = asset['mel_frames']  # 128 帧片段的裁剪范围来自原完整 mel。
        asset.update(crop_start=crop_random.randint(0, frames - 128) if frames > 128 else 0, segment_frames=128, valid_frames=min(frames, 128))  # 短句在标准化后补零。
        self_rows.append(asset)  # 顺序决定同一 CUDA 噪声张量的批内位置。
    pairs = []  # 10 个源说话人各转换到另外9人，固定90个配对。
    for source_speaker in selected_speakers:  # 每个 source 录音在9个目标条件下复用。
        source = assets[roles[source_speaker]['source_train_index']]  # 完整源mel与BNF只来自这一句。
        for reference_speaker in selected_speakers:  # 参考条件按目标说话人固定。
            if source_speaker == reference_speaker:  # 自重建单独计量，不混入跨说话人90对。
                continue  # 保留全部不同说话人的组合。
            reference = assets[roles[reference_speaker]['reference_train_index']]  # 参考GE2E从其固定参考语音读取。
            pair_id = len(pairs)  # 从0到89，跨模型身份保持一致。
            pairs.append({  # 前12项沿用正式客观评分工具的字段名。
                'pair_id': pair_id, 'source_speaker': source_speaker, 'source_utt': source['utt_id'],  # 固定源身份。
                'source_sentence_slot': source['utt_id'].rsplit('_', 1)[-1], 'source_text': source['text'], 'source_wav': source['wav'],  # 此slot是VCTK句号，不是正式测试1..10。
                'reference_speaker': reference_speaker, 'reference_utt': reference['utt_id'],  # 目标说话人及其参考句。
                'reference_sentence_slot': reference['utt_id'].rsplit('_', 1)[-1], 'reference_text': reference['text'], 'reference_wav': reference['wav'],  # SVA使用同一参考wav。
                'output_wav': '', 'target_speaker': reference_speaker,  # 空输出由每个系统各自填入，原模板不改写。
                'source_train_index': source['train_index'], 'reference_train_index': reference['train_index'],  # 直接对应排除集合。
                'source_mel': source['mel'], 'source_bnf': source['bnf'], 'reference_spk_emb': reference['spk_emb'],  # 完整推理的三项条件。
                'source_wav_sha256': source['wav_sha256'], 'reference_wav_sha256': reference['wav_sha256'],  # 原音频身份。
                'source_mel_sha256': source['mel_sha256'], 'source_bnf_sha256': source['bnf_sha256'], 'reference_spk_emb_sha256': reference['spk_emb_sha256'],  # 特征身份。
                'source_mel_frames': source['mel_frames'], 'expected_num_samples': source['mel_frames'] * 256, 'seed': PAIR_NOISE_BASE + pair_id,  # 输出保持完整mel长度，不裁到原wav长度。
            })  # 未生成波形，也未读取任何系统的指标。
    contract = {  # 下游训练和评估必须共同读取这一份开发定义。
        'schema_version': 1, 'kind': 'student_training_source_development', 'created_unix': time.time(),  # 开发集身份与创建时间。
        'output_directory': str(output), 'selection_seed': SELECTION_SEED, 'train_manifest': str(TRAIN), 'train_manifest_sha256': EXPECTED[TRAIN],  # 原索引的唯一来源。
        'excluded_train_indices': excluded, 'retained_train_indices': retained, 'reconstruction_dev_indices': reconstruction,  # 训练runner直接使用的顶层索引。
        'cross_speaker_dev_indices': cross_indices, 'cross_speaker_roles': roles, 'cross_speaker_pair_count': len(pairs),  # 配对语音与角色。
        'reconstruction_protocol': {  # 完全延续旧32句的输入身份；不为方便而更换CPU噪声。
            'crop_seed': 1234, 'source_noise_seed': 55001, 'source_noise_device': 'cuda', 'source_noise_dtype': 'float32',  # 原裁剪和源噪声。
            'score_noise_seed': 66002, 'timestep_seed': 77003, 'batch_size': 32, 'segment_frames': 128,  # 原移动dist诊断的独立噪声及t种子。
            'source_noise_generation': 'one torch.randn([32,80,128], device=cuda, generator=torch.Generator(device=cuda).manual_seed(55001))',  # 逐句重新设种子不能复现旧批量噪声。
            'archived_input_identity': old[0], 'crop_starts': [row['crop_start'] for row in self_rows],  # mel/BNF/source_noise哈希须由模型诊断再次核对。
            'external_bnf_alignment': 'legacy_length_linear', 'internal_bnf_alignment': 'legacy_decimate',  # 现有基线输入规则，不伪称改时轴后身份不变。
            'normalization': '(raw_log_mel - frozen_train_mean) / frozen_train_std; pad zero after normalization',  # 与固定教师的数值约定相同。
        },  # 自重建仅用于固定真实mel尺度，不能替代跨说话人性能。
        'selection_rules': {  # 写清楚自行选择的规则及计数边界。
            'gender_counts': {'F': 5, 'M': 5}, 'known_word_count_inclusive': [8, 35], 'minimum_utterance_number': 25,  # 小规模开发协议，并非论文规定。
            'known_full_text_occurrences_in_original_train': 1, 'unknown_training_transcript_rows': sum(not text for text in normalized),  # 166条缺文本阻止全库严格文本不重叠证明。
            'ranking': 'sha256(20260906:speaker:SPEAKER); sha256(20260906:utterance:UTT_ID)',  # 性别分组后各取前5人，每人前2句。
            'metric_based_selection': False, 'formal_test_content_used_for_selection': False,  # 不看旧模型好坏挑样本。
        },  # 词数只是减少极短句造成的CER波动，不按听感筛选。
        'exposure': {  # 明确各模型已经见过什么，避免把开发结果冒充独立测试。
            'teacher_trained_on_all_development_utterances': True, 'historical_full_students_trained_on_all_development_utterances': True,  # 原教师与旧完整学生都见过原train。
            'new_candidate_training_must_exclude_all_52_utterances': True, 'frozen_normalization_statistics_include_development_utterances': True,  # 新学生梯度排除，教师原统计保持不动。
            'old_32_were_previously_used_for_diagnostics': True, 'formal_unseen_evaluation_used_for_tuning': False,  # 已用过的旧32仅作可复现固定量尺。
        },  # 这是seen-speaker训练来源开发，不是正式unseen100/8100评估。
        'limitations': [  # 后续报告须保留这些范围说明。
            'The 90 pairs contain 10 unique source utterances and 10 fixed references; repeated sources are statistically dependent.',  # 90个组合不等于90条独立源语音。
            'The 32 reconstruction utterances span 27 original training speakers, separately from the 10 cross-speaker development speakers.',  # 两个开发子集的说话人范围不同。
            'One archived reconstruction utterance has no transcript; reconstruction does not compute CER.',  # 无文本句只作mel固定误差。
            'Known selected cross-speaker texts are unique in the original training metadata, but 166 unknown transcripts prevent a full text-disjointness claim.',  # 不夸大文本隔离证明。
            'Teacher and historical students are exposed baselines; this development comparison is not an independent final generalization estimate.',  # 教师与新学生训练暴露不同。
            'Any new BNF alignment arm must use a distinct run/input contract and may not claim equality to archived legacy BNF tensor hashes.',  # 时轴实验另立身份。
        ],  # 正式测试只在候选确定后另行进行。
        'artifacts': {name: str(output / name) for name in ('pairs.csv', 'self_reconstruction.jsonl', 'student_train_manifest.csv', 'dev_excluded_manifest.csv')},  # 各文件绝对路径。
        'source_sha256': source_hashes, 'script_sha256': file_sha(Path(__file__)),  # 完整记录使用过的文件身份。
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),  # 只读记录服务器当前代码版本。
    }  # 不包含测试指标、模型选择结果或伪造的训练状态。
    for path, expected in source_hashes.items():  # 全部选择和元数据检查完成后复查源文件。
        if file_sha(path) != expected:  # 原文件发生变化则不能签发完成收据。
            raise RuntimeError(f'Source changed during manifest construction: {path}')  # 不更改旧数据来强行通过。
    output.mkdir(parents=True, exist_ok=False)  # 所有检查通过才创建新目录，失败时绝不复用旧目录。
    save_csv_new(output / 'pairs.csv', list(pairs[0]), pairs)  # 供每个系统复用的固定配对模板。
    save_csv_new(output / 'student_train_manifest.csv', fields, [train[i] for i in retained])  # 单独保存排除52句后的训练名单。
    save_csv_new(output / 'dev_excluded_manifest.csv', fields, [train[i] for i in excluded])  # 保存被排除的原始记录，方便教授或研究者核对。
    with (output / 'self_reconstruction.jsonl').open('x', encoding='utf-8') as handle:  # 一行一条原自重建输入。
        for row in self_rows:  # 批内顺序与旧诊断相同。
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')  # 保留裁剪位置及来源资产哈希。
    save_json_new(output / 'development_contract.json', contract)  # 最后写出供训练runner读取的开发协议。
    artifact_hashes = {name: file_sha(output / name) for name in ('development_contract.json', 'pairs.csv', 'self_reconstruction.jsonl', 'student_train_manifest.csv', 'dev_excluded_manifest.csv')}  # 文件字节哈希单独记账。
    receipt = {  # 完成标记最后写入；任何半成品目录都不能被下游接受。
        'status': 'complete', 'schema_version': 1, 'development_contract_sha256': canonical_sha(contract),  # 内容级指纹。
        'development_contract_file_sha256': artifact_hashes['development_contract.json'], 'pairs_csv_sha256': artifact_hashes['pairs.csv'],  # 实际文件字节指纹。
        'artifacts': artifact_hashes, 'excluded_utterance_count': 52, 'retained_utterance_count': 39335, 'cross_speaker_pair_count': 90, 'reconstruction_count': 32,  # 各角色数量。
        'project_compute': 'CPU metadata and selected cached-array checks only; no model, GPU, optimizer or feature extraction',  # 明确生成工具未做训练或评估。
    }  # 收据不包含自身哈希，避免自引用。
    save_json_new(output / 'development_manifest.json', receipt)  # 下游先确认complete，再逐一核验artifact hashes。
    print(json.dumps({'output': str(output), **receipt}, ensure_ascii=False, indent=2), flush=True)  # 返回新产物位置与简要身份。


if __name__ == '__main__':  # 被其他脚本读取或审阅时不自动生成清单。
    main()  # 无训练、无推理、无正式测试指标计算。
