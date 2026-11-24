"""只读检查固定开发集的270条生成波形；不加载模型、不评分、不改音频。"""

import argparse
import collections  # 按源句和目标说话人分别保留分组，不能把90对当90个独立源句。
import contextlib
import csv
import fcntl  # 只持有原三把工作流锁，不接触publication任务的锁。
import hashlib
import json
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np  # float64时间能量与rFFT，完全不导入torch或评分器。
import soundfile as sf  # 仅解码已有PCM16文件，代码中没有音频写入调用。

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT/'exp/diagnostics/student_development_20260906'
TOOLS = ROOT/'exp/diagnostics/student_performance_tools_20260906'
OUTPUT = DEV/'acdd_dev90_waveform_readonly_v1'
PROTOCOL = DEV/'DEV90_WAVEFORM_AUDIT_PROTOCOL_CN.md'
PROTOCOL_SHA = '416f345668608fefb7089d75f7b7dc51a10c5bb787e00dfcda8e5392464ff266'
COMPARISON = DEV/'comparison_acdd_s5e5_3327_v1/comparison.json'
COMPARISON_SHA = '7e63195d6b5e696394c2e0312d67b287e06a541d34776e2bc47f636f7f844534'
METRIC_SHA = 'c0ff0e76a25b5656f1b2c110a2a6664342b03d8cf244971a8590074c75a37d86'
HEAD = '0a2c646922704e746642f5f33c203869fe696b3f'
SYSTEMS = {
    'teacher500_k30': DEV/'evaluation_baselines_v1/teacher500_k30',
    'student_acdd_s5e5_s3327': DEV/'evaluation_candidate_student_acdd_s5e5_s3327/student_acdd_s5e5_s3327',
    'student_acdd_lr_half_s3227': DEV/'evaluation_candidate_student_acdd_lr_half_s3227/student_acdd_lr_half_s3227',
}
ROLES = dict(zip(SYSTEMS, ('teacher', 'candidate', 'parent')))
METRICS = ('cer', 'sva', 'dnsmos', 'utmos')
METRIC_FIELDS = ('p808_mos', 'mos_ovr', 'mos_sig', 'mos_bak', 'utmos', 'cosine_score', 'verified')
LOCK_NAMES = ('text_unseen_custom_hifigan_pipeline.lock', 'text_unseen_custom_hifigan_train.lock', 'text_unseen_fastvoicegrad_student.lock')
RATE, FRAME, HOP = 22050, 1024, 256  # 帧起点为j*256；只取完整1024点，不补边。
RESERVE = 3*1024**3 + 128*1024**2
CAP, DIRECTORY = 16*1024**2, 1024**2  # 日志也计算在16MiB内，另外预留目录空间。
BANDS = ('0_1000', '1000_4000', '4000_8000', '8000_11025')


def canonical(value):
    return hashlib.sha256(encode(value)).hexdigest()


def encode(value):
    # 禁止NaN/Infinity进入JSON；真正无定义的量必须显式写None/null。
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def regular(path):
    # 文件和所有父目录都不允许符号链接，避免登记路径被重定向。
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise RuntimeError('Expected absolute regular input: '+str(path))
    if not stat.S_ISREG(path.stat().st_mode):
        raise RuntimeError('Nonregular input: '+str(path))
    return path


def sha(path, check=lambda: None):
    result = hashlib.sha256()
    with regular(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''):
            check(); result.update(block)  # 大文件校验也遵守同一个300秒时限。
    return result.hexdigest()


def bind(files, path, expected):
    # 同路径的两个证据必须一致，不能用后读到的SHA覆盖已固定锚点。
    key = str(path)
    if key in files and files[key] != expected:
        raise RuntimeError('Conflicting input identity: '+key)
    files[key] = expected


def verify(files, check=lambda: None):
    for path, expected in files.items():
        check()
        if sha(path, check) != expected:
            raise RuntimeError('Input/source SHA changed: '+path)


def read_json(path):
    return json.loads(regular(path).read_text(encoding='utf-8'))


def rows90(path):
    with regular(path).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    # 不自动排序错位的CSV，也不去重或丢弃失败行。
    if len(rows) != 90 or [row['pair_id'] for row in rows] != [str(i) for i in range(90)]:
        raise RuntimeError('CSV must contain exactly ordered pair_id 0..89: '+str(path))
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise RuntimeError('Malformed CSV row: '+str(path))
    return rows


def pair_equal(rows, original):
    # 只有输出路径可因系统改变，文本、源/参考身份、种子和长度都必须相同。
    if len(rows) != len(original):
        raise RuntimeError('Pair count mismatch')
    for row, reference in zip(rows, original):
        if {k:v for k,v in row.items() if k != 'output_wav'} != {k:v for k,v in reference.items() if k != 'output_wav'}:
            raise RuntimeError('Pair recipe changed: '+row['pair_id'])


def metric_records(tables):
    # 这里只解析已生成的原评分CSV，绝不执行评分模型。
    result = []
    for index in range(90):
        row = {'pair_id': index}
        for metric in METRICS:
            for key, value in tables[metric][index].items():
                if key == 'pair_id':
                    continue
                if key in ('reference_text_normalized', 'hypothesis_text_normalized'):
                    row[key] = value
                elif key in ('char_errors', 'reference_chars', 'verified'):
                    row[key] = int(value)
                else:
                    row[key] = float(value)
                    if not math.isfinite(row[key]):
                        raise RuntimeError('Nonfinite original metric')
        if row['reference_chars'] <= 0 or row['char_errors'] < 0 or row['verified'] not in (0, 1):
            raise RuntimeError('Invalid original metric counts')
        if abs(row['cer_percent']-100*row['char_errors']/row['reference_chars']) > 1e-12:
            raise RuntimeError('CER row does not match original integer counts')
        result.append(row)
    return result


def load_plan(check=lambda: None):
    # 只读身份、CSV和SHA；CPU测试可调用此函数，但不会读波形样本或计算特征。
    files = {str(PROTOCOL): PROTOCOL_SHA, str(COMPARISON): COMPARISON_SHA}
    verify(files, check)
    comparison = read_json(COMPARISON)
    inherited = comparison['source_sha256']

    def inherited_file(path):
        if str(path) not in inherited:
            raise RuntimeError('File is not pinned by original comparison: '+str(path))
        bind(files, path, inherited[str(path)])
        verify({str(path): files[str(path)]}, check)

    inherited_file(DEV/'pairs.csv')
    original = rows90(DEV/'pairs.csv')
    sources = {}
    for row in original:
        # 实际数据字段沿用原CSV；源音频、参考音频、mel/BNF/说话人向量都只验SHA。
        for field in ('source_wav', 'reference_wav', 'source_mel', 'source_bnf', 'reference_spk_emb'):
            bind(files, row[field], row[field+'_sha256'])
            if inherited.get(row[field]) != row[field+'_sha256']:
                raise RuntimeError('Pair data differs from frozen comparison')
        description = {k: row[k] for k in ('source_utt', 'source_speaker', 'source_wav', 'source_wav_sha256', 'expected_num_samples')}
        if row['source_utt'] in sources and sources[row['source_utt']] != description:
            raise RuntimeError('Source identity/length differs between targets')
        sources[row['source_utt']] = description
    if len(sources) != 10 or set(collections.Counter(row['source_utt'] for row in original).values()) != {9}:
        raise RuntimeError('Expected ten source groups of nine pairs')
    if len(set(row['reference_speaker'] for row in original)) != 10 or set(collections.Counter(row['reference_speaker'] for row in original).values()) != {9}:
        raise RuntimeError('Expected ten target groups of nine pairs')
    systems = {}
    for label, folder in SYSTEMS.items():
        check()
        paths = [folder/'pairs.csv', folder/'inference_complete.json', folder/'metrics/development_summary_cer_dnsmos_sva_utmos.json', folder.parent/'run_identity.json', folder.parent/'metric_identity.json']
        paths += [folder/'metrics'/(name+'.csv') for name in METRICS]
        for path in paths:
            inherited_file(path)
        rows = rows90(folder/'pairs.csv'); pair_equal(rows, original)
        receipt = read_json(folder/'inference_complete.json')
        identity = read_json(folder.parent/'run_identity.json')
        metric_identity = read_json(folder.parent/'metric_identity.json')
        summary = read_json(folder/'metrics/development_summary_cer_dnsmos_sva_utmos.json')
        if identity != comparison['inference_identities'][ROLES[label]] or metric_identity != comparison['metric_identity']:
            raise RuntimeError('Actual inference/metric identity differs from comparison')
        if summary['system'] != label or receipt['system'] != label or summary['evaluated_pairs'] != 90:
            raise RuntimeError('System role/count changed')
        if summary['run_identity_sha256'] != files[str(folder.parent/'run_identity.json')] or receipt['run_identity_sha256'] != summary['run_identity_sha256']:
            raise RuntimeError('Run identity FILE SHA mismatch')
        if summary['metric_identity_sha256'] != METRIC_SHA or files[str(folder.parent/'metric_identity.json')] != METRIC_SHA:
            raise RuntimeError('Scoring identity changed')
        if summary['inference_complete_sha256'] != files[str(folder/'inference_complete.json')] or receipt['pairs_csv_sha256'] != files[str(folder/'pairs.csv')]:
            raise RuntimeError('Inference receipt/CSV SHA mismatch')
        for name in METRICS:
            if summary['metric_csv_sha256'][name] != files[str(folder/'metrics'/(name+'.csv'))]:
                raise RuntimeError('Original metric CSV identity changed')
        if set(receipt['wav_sha256']) != {str(i) for i in range(90)}:
            raise RuntimeError('WAV receipt must retain all90 pairs')
        for row in rows:
            path = Path(row['output_wav'])
            if path.parent != folder/'wavs' or path.resolve() != path:
                raise RuntimeError('Unexpected generated WAV path')
            bind(files, path, receipt['wav_sha256'][row['pair_id']])
            if inherited.get(str(path)) != files[str(path)]:
                raise RuntimeError('WAV differs between inference and comparison receipts')
        metrics = metric_records({name: rows90(folder/'metrics'/(name+'.csv')) for name in METRICS})
        for actual, saved in zip(metrics, comparison['all90_rows'][label]):
            if any(saved[key] != value for key, value in actual.items()):
                raise RuntimeError('Original metric values differ from frozen comparison')
        if sum(row['reference_chars'] for row in metrics) != 4392 or sum(row['char_errors'] for row in metrics) != summary['additional']['CER_total_errors']:
            raise RuntimeError('Corpus CER counts changed')
        systems[label] = {'rows': rows, 'metrics': metrics, 'inference': receipt}
    verify(files, check)  # 所有280音频和实际使用的元数据在特征读取前固定身份。
    return {'files': files, 'pairs': original, 'sources': sources, 'systems': systems}


def check_header(info, expected, source=False):
    if info.samplerate != RATE or info.channels != 1 or info.subtype != 'PCM_16' or info.format != 'WAV':
        raise RuntimeError('Require 22050Hz mono WAV PCM16')
    extra = info.frames-int(expected)
    if (not source and extra != 0) or (source and not 0 <= extra <= 255):
        raise RuntimeError('Unexpected generated/source waveform length')
    if int(expected) < FRAME:
        raise RuntimeError('No complete1024-sample frame')
    return int(extra)


def read_wave(path, expected, source=False):
    info = sf.info(str(regular(path)))
    extra = check_header(info, expected, source)
    wave, rate = sf.read(str(path), dtype='float64', always_2d=False)
    if rate != RATE or wave.ndim != 1 or len(wave) != info.frames or not np.isfinite(wave).all():
        raise RuntimeError('Decoded audio/header mismatch or nonfinite samples')
    # 只有source允许在分析内去掉已登记的0..255点尾巴，原文件完全不写。
    return wave[:int(expected)] if source else wave, extra


def frames_of(wave):
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 1 or len(wave) < FRAME or not np.isfinite(wave).all():
        raise RuntimeError('Invalid waveform for fixed frame grid')
    count = 1+(len(wave)-FRAME)//HOP
    offsets = np.arange(count)[:, None]*HOP + np.arange(FRAME)[None, :]
    return wave[offsets]  # 显式索引不会移动输入波形，也不会作补零。


def source_mask(wave):
    energy = np.mean(frames_of(wave)**2, axis=1, dtype=np.float64)
    maximum = float(energy.max())
    mask = None if maximum == 0 else energy < maximum*1e-4
    return {'frame_mean_square': energy.tolist(), 'low_energy_mask': None if mask is None else mask.tolist(),
            'maximum_frame_mean_square': maximum, 'threshold': maximum*1e-4,
            'low_count': None if mask is None else int(mask.sum()),
            'other_count': None if mask is None else int((~mask).sum()),
            'mask_undefined_reason': 'source_all_zero' if mask is None else None}


def band_masks(frequencies):
    # 上三个频带为左闭右开；最后一个显式包含11025Hz的Nyquist点。
    edges = ((0, 1000), (1000, 4000), (4000, 8000), (8000, 11025))
    masks = [(frequencies >= low) & ((frequencies <= high) if index == 3 else (frequencies < high)) for index, (low, high) in enumerate(edges)]
    if not np.all(np.stack(masks).sum(axis=0) == 1):
        raise RuntimeError('Frequency bands do not cover every bin exactly once')
    return masks


def spectrum(frame_values):
    # 周期Hann分母为1024，不用np.hanning的对称窗；不减均值、不去DC。
    window = 0.5-0.5*np.cos(2*np.pi*np.arange(FRAME, dtype=np.float64)/FRAME)
    power = np.abs(np.fft.rfft(frame_values*window, axis=1))**2
    power[:, 1:-1] *= 2  # 单侧谱的DC和Nyquist不加倍，其余频点补上负频率能量。
    masks = band_masks(np.fft.rfftfreq(FRAME, d=1/RATE))
    energy = np.asarray([power[:, mask].sum(dtype=np.float64) for mask in masks])
    total = float(energy.sum(dtype=np.float64))
    fractions = None if total == 0 else (energy/total).tolist()
    if fractions is not None and abs(sum(fractions)-1) > 1e-12:
        raise RuntimeError('Band fractions must sum to one')
    return energy.tolist(), fractions, total


def waveform_features(wave, shared_mask):
    frames = frames_of(wave)
    energy = np.mean(frames**2, axis=1, dtype=np.float64)
    if len(energy) != len(shared_mask['frame_mean_square']):
        raise RuntimeError('Source/generated frame grids differ')
    low_mask = shared_mask['low_energy_mask']
    reasons = {}
    if low_mask is None:
        low = other = ratio = None
        reasons = dict.fromkeys(('low_frame_rms', 'other_frame_rms', 'low_to_other_db'), 'source_all_zero')
    else:
        mask = np.asarray(low_mask, dtype=bool)
        low = float(np.sqrt(energy[mask].mean())) if mask.any() else None
        other = float(np.sqrt(energy[~mask].mean())) if (~mask).any() else None
        if low is None: reasons['low_frame_rms'] = 'empty_source_low_mask'
        if other is None: reasons['other_frame_rms'] = 'empty_source_other_mask'
        if low is None or other is None:
            ratio = None; reasons['low_to_other_db'] = 'empty_or_undefined_frame_set'
        elif low == 0 or other == 0:
            ratio = None; reasons['low_to_other_db'] = 'zero_low_or_other_rms'
        else:
            ratio = float(20*np.log10(low/other))
    bands, fractions, total = spectrum(frames)
    near_count = int(np.count_nonzero(np.abs(wave) >= 0.999))
    values = {'num_samples': len(wave), 'peak': float(np.max(np.abs(wave))),
              'rms': float(np.sqrt(np.mean(wave**2, dtype=np.float64))), 'dc_mean': float(np.mean(wave, dtype=np.float64)),
              'near_boundary_count': near_count, 'near_boundary_fraction': near_count/len(wave),
              'frame_count': len(frames), 'uncovered_tail_samples': len(wave)-((len(frames)-1)*HOP+FRAME),
              'frame_mean_square_mean': float(energy.mean()), 'low_frame_rms': low, 'other_frame_rms': other,
              'low_to_other_db': ratio, 'spectral_energy_total': total}
    for index, band in enumerate(BANDS):
        values['spectral_energy_'+band] = bands[index]
        values['spectral_fraction_'+band] = None if fractions is None else fractions[index]
    return {'features': values, 'frame_mean_square': energy.tolist(), 'undefined_reasons': reasons,
            'spectral_fraction_undefined_reason': 'all_spectral_energy_zero' if total == 0 else None}


def describe(values):
    # 每个描述量保留有效数和缺失数，不用零替代null再求平均。
    valid = [float(value) for value in values if value is not None]
    if not all(math.isfinite(value) for value in valid):
        raise RuntimeError('Nonfinite descriptor')
    return {'valid': len(valid), 'missing': len(values)-len(valid),
            'mean': float(np.mean(valid)) if valid else None, 'median': float(np.median(valid)) if valid else None,
            'min': min(valid) if valid else None, 'max': max(valid) if valid else None,
            'positive': sum(value > 0 for value in valid), 'negative': sum(value < 0 for value in valid), 'zero': sum(value == 0 for value in valid)}


def differences(candidate, reference):
    if candidate['pair_id'] != reference['pair_id']:
        raise RuntimeError('Cannot subtract different pairs')
    left, right = candidate['features'], reference['features']
    if set(left) != set(right) or len(candidate['frame_mean_square']) != len(reference['frame_mean_square']):
        raise RuntimeError('Feature schema/grid mismatch')
    values = {key: None if left[key] is None or right[key] is None else left[key]-right[key] for key in left}
    a, b = candidate['metrics'], reference['metrics']
    metric_delta = {key: a[key]-b[key] for key in METRIC_FIELDS+('char_errors', 'reference_chars', 'cer_percent')}
    return {'pair_id': candidate['pair_id'], 'source_utt': candidate['source_utt'], 'reference_speaker': candidate['reference_speaker'],
            'features': values, 'frame_mean_square_difference': (np.asarray(candidate['frame_mean_square'])-np.asarray(reference['frame_mean_square'])).tolist(),
            'metric_differences': metric_delta}


def summarize(rows, contrast=False):
    metrics_key = 'metric_differences' if contrast else 'metrics'
    summary = {'n_pairs': len(rows), 'features': {key: describe([row['features'][key] for row in rows]) for key in rows[0]['features']},
               'metrics': {key: describe([row[metrics_key][key] for row in rows]) for key in METRIC_FIELDS}}
    # 总体CER严格使用错误数之和/字符数之和；差值用两系统共用的原分母。
    errors = sum(row[metrics_key]['char_errors'] for row in rows)
    chars = sum(row['reference_chars'] if contrast else row['metrics']['reference_chars'] for row in rows)
    summary['CER'] = {'errors_difference' if contrast else 'errors': errors, 'reference_chars': chars,
                      'corpus_percent_difference' if contrast else 'corpus_percent': 100*errors/chars}
    return summary


def grouped(rows, contrast=False):
    return {'all90': summarize(rows, contrast), **{
        field: {value: summarize([row for row in rows if row[field] == value], contrast) for value in sorted({row[field] for row in rows})}
        for field in ('source_utt', 'reference_speaker')}}


class Interrupted(RuntimeError):
    pass


class Clock:
    def __init__(self, seconds=300):
        self.start = time.monotonic(); self.seconds = seconds; self.reason = None

    def handle(self, number, frame):
        self.reason = signal.Signals(number).name  # 信号只设停止标记，下一检查点转入失败清理。

    def check(self):
        if self.reason or time.monotonic()-self.start >= self.seconds:
            raise Interrupted(self.reason or '300-second deadline')


def watchdog():
    parent = Path('/proc')/str(os.getppid())
    args = [word.decode() for word in (parent/'cmdline').read_bytes().split(b'\0') if word]
    if (parent/'exe').resolve() != Path('/usr/bin/timeout').resolve() or not all(word in args for word in ('--signal=TERM', '--kill-after=30s', '300s')):
        raise RuntimeError('Require timeout --signal=TERM --kill-after=30s 300s')
    return {'pid': os.getppid(), 'command': args, 'seconds': 300, 'cleanup_only_seconds': 30}


def output_path(path, parent=DEV):
    # 输出只能是已登记父目录的直接新子目录，既有目录、日志或符号链接全部拒绝。
    path, parent = Path(path), Path(parent)
    if path.parent != parent or path.resolve() != path or parent.resolve() != parent:
        raise RuntimeError('Output must be an unredirected direct child')
    if os.path.lexists(str(path)) or os.path.lexists(str(path)+'.log'):
        raise RuntimeError('Output/log already exists; no automatic retry')
    return path


class Budget:
    def __init__(self, output):
        self.output = Path(output); self.log_path = Path(str(output)+'.log'); self.used = 0; self.log_handle = None

    def check(self, amount=0):
        disk = os.statvfs(self.output.parent)
        remaining = CAP-self.used
        if amount > remaining or disk.f_bavail*disk.f_frsize < RESERVE+DIRECTORY+remaining:
            raise RuntimeError('Metadata capacity/system reserve would be exceeded')

    def open(self):
        self.check()
        self.log_handle = self.log_path.open('x', encoding='utf-8')
        self.output.mkdir()  # 只创建本次独占的新目录；失败也不触碰旧结果。

    def log(self, message):
        data = message+'\n'; self.check(len(data.encode('utf-8')))
        self.log_handle.write(data); self.log_handle.flush(); self.used += len(data.encode('utf-8'))
        try: print(message, flush=True)
        except (OSError, BrokenPipeError): pass  # stdout关闭不应抹去独占日志。

    def write(self, name, value, check=lambda: None):
        if Path(name).name != name or not name.endswith(('.json', '.jsonl')):
            raise RuntimeError('Illegal output filename')
        data = encode(value)+b'\n'; self.check(len(data)); check()
        destination = self.output/name
        if destination.exists() or destination.is_symlink():
            raise RuntimeError('Refuse existing output file')
        temporary = self.output/(name+'.tmp')
        with temporary.open('xb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        self.used += len(data); check()
        os.link(str(temporary), str(destination))  # 原子且不覆盖已有文件，再去掉本次临时名字。
        temporary.unlink()

    def append(self, name, value):
        if name != 'waveforms.jsonl': raise RuntimeError('Only the registered waveform journal may append')
        data = encode(value)+b'\n'; self.check(len(data))
        path = self.output/name
        if path.is_symlink(): raise RuntimeError('Redirected waveform journal')
        with path.open('ab') as handle: handle.write(data); handle.flush()
        self.used += len(data)

    def close(self):
        if self.log_handle is not None: self.log_handle.close()


def environment():
    if not sys.platform.startswith('linux') or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve():
        raise RuntimeError('Use the original rented Matpool myconda interpreter')
    if platform.python_version() != '3.8.15' or np.__version__ != '1.23.5' or sf.__version__ != '0.12.1' or sf.__libsndfile_version__ != '1.2.0':
        raise RuntimeError('CPU waveform runtime changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise RuntimeError('This audit must be CPU-only without torch')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=str(ROOT), text=True)
    if head != HEAD or dirty.strip(): raise RuntimeError('Production HEAD/worktree changed')
    return {'python': platform.python_version(), 'numpy': np.__version__, 'soundfile': sf.__version__,
            'libsndfile': sf.__libsndfile_version__, 'git_head': head, 'cuda_visible_devices': ''}


def fail_output(budget, state, error):
    record = dict(state, status='failed', error=repr(error), original_artifacts_not_written=True)
    # 若发布后的日志/截止检查失败，保留failure并明确撤销本次成功结果，不碰旧实验。
    if budget is not None and budget.output.is_dir():
        try:
            result = budget.output/'results.json'
            if result.exists():
                os.rename(str(result), str(budget.output/'results.invalidated.json'))
            budget.write('failure.json', record)
        except BaseException as secondary:
            record['failure_record_write_error'] = repr(secondary)
    try: print('WAVEFORM_AUDIT_FAILED '+json.dumps(record, ensure_ascii=False), flush=True)
    except OSError: pass


def cleanup(budget, locks, old_handlers):
    # 即使日志关闭失败，也逐一尝试释放所有锁、恢复全部信号处理器。
    errors = []
    actions = ([('log_close', budget.close)] if budget is not None else [])
    actions += [('lock_close_'+str(index), handle.close) for index, handle in enumerate(reversed(locks))]
    actions += [('restore_signal_'+str(number), lambda n=number,h=handler: signal.signal(n,h))
                for number, handler in old_handlers.items()]
    for label, action in actions:
        try: action()
        except BaseException as error: errors.append({'action': label, 'error': repr(error)})
    return errors


def main():
    clock = Clock(); locks = []; budget = None; primary_error = None
    state = {'phase': 'preflight', 'generated_rows_completed': 0, 'source_rows_completed': 0, 'final_input_verification_passed': False}
    old_handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in old_handlers: signal.signal(number, clock.handle)
    try:
        runtime = environment(); parent = watchdog(); clock.check()
        for name in LOCK_NAMES:
            path = ROOT/'exp/locks'/name; regular(path)
            handle = path.open('r+'); locks.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        output_path(OUTPUT); budget = Budget(OUTPUT); budget.open()
        def check(): clock.check(); budget.check()
        budget.log('START CPU-only270-WAV audit; no models/no scoring/no audio writes')
        plan = load_plan(check)
        for path in (Path(__file__).resolve(), TOOLS/'test_dev90_waveform_v1.py'):
            bind(plan['files'], path, sha(path, check))
        identity = {'kind': 'preregistered_dev90_waveform_readonly', 'runtime': runtime, 'watchdog': parent,
                    'file_sha256': dict(plan['files']), 'systems': list(SYSTEMS), 'source_count': 10, 'generated_count': 270,
                    'protocol_sha256': PROTOCOL_SHA, 'comparison_sha256': COMPARISON_SHA,
                    'mask': 'source_frame_MSE < max_source_frame_MSE * 1e-4', 'frame': FRAME, 'hop': HOP,
                    'center': False, 'window': 'periodic_Hann', 'audio_dtype': 'float64', 'metric_recomputation': False}
        budget.write('run_identity.json', identity, check)
        identity_file_sha = sha(OUTPUT/'run_identity.json', check)
        state['identity_sha256'] = canonical(identity); state['phase'] = 'source_masks'
        masks = {}; source_records = []
        for source, spec in plan['sources'].items():
            check(); wave, extra = read_wave(spec['source_wav'], int(spec['expected_num_samples']), True)
            mask = source_mask(wave); masks[source] = mask
            source_records.append(dict(spec, original_samples=len(wave)+extra, analysis_samples=len(wave), extra_tail_samples=extra,
                                       frame_count=len(mask['frame_mean_square']), uncovered_tail_samples=(len(wave)-FRAME)%HOP, **mask))
            state['source_rows_completed'] += 1
        budget.write('source_frames.json', source_records, check)
        records = {}; state['phase'] = 'generated_waveforms'
        for label, system in plan['systems'].items():
            current = []
            for row, metrics in zip(system['rows'], system['metrics']):
                check(); wave, extra = read_wave(row['output_wav'], int(row['expected_num_samples']))
                features = waveform_features(wave, masks[row['source_utt']])
                item = dict(features, system=label, pair_id=int(row['pair_id']), source_utt=row['source_utt'],
                            reference_speaker=row['reference_speaker'], output_wav=row['output_wav'],
                            wav_sha256=plan['files'][row['output_wav']], metrics=metrics)
                check(); budget.append('waveforms.jsonl', item); current.append(item)
                state['generated_rows_completed'] += 1
                if len(current)%30 == 0: budget.log('PROGRESS '+label+' '+str(len(current))+'/90')
            records[label] = current
        check(); state['phase'] = 'all_pair_differences'
        primary = records['student_acdd_s5e5_s3327']; contrasts = {}
        for reference in ('teacher500_k30', 'student_acdd_lr_half_s3227'):
            pairs = []
            for candidate, old in zip(primary, records[reference]):
                item = differences(candidate, old); item['reference_chars'] = candidate['metrics']['reference_chars']; pairs.append(item)
            contrasts[reference] = {'all90': pairs, 'summaries': grouped(pairs, True)}
        state['phase'] = 'final_identity_check'; verify(plan['files'], check)
        verify({str(OUTPUT/'run_identity.json'): identity_file_sha}, check); environment(); check()
        state['final_input_verification_passed'] = True
        result = {'status': 'dev90_waveform_readonly_complete', 'identity_sha256': canonical(identity),
                  'run_identity_file_sha256': identity_file_sha,
                  'counts': {'source': 10, 'generated': 270, 'contrasts': 2, 'pairs_per_contrast': 90},
                  'system_summaries': {label: grouped(values) for label, values in records.items()},
                  'candidate_minus_references': contrasts, 'all_inputs_unchanged': True,
                  'source_frames_sha256': sha(OUTPUT/'source_frames.json', check), 'waveforms_sha256': sha(OUTPUT/'waveforms.jsonl', check),
                  'elapsed_seconds': time.monotonic()-clock.start,
                  'limitations': ['Only10source utterances; target groups share those sources; development exposure remains.',
                                  'Shared source low-energy mask is not noise/SNR and can include consonants/breath/timing differences.',
                                  'Near-boundary count cannot recover pre-save clipping; high-frequency energy alone is not distortion.',
                                  'DNSMOS resamples to16kHz, so energy above8kHz cannot alone explain P808.',
                                  'Descriptive preregistered features only; no feature selection, causal claim or new metric scoring.']}
        budget.write('results.json', result, check); check()
        budget.log('FINISHED270 generated/10source/2all90 contrasts; all input SHA unchanged')
        check(); state['phase'] = 'completed'
    except BaseException as error:
        primary_error = error
    finally:
        cleanup_errors = cleanup(budget, locks, old_handlers)
    if primary_error is not None or cleanup_errors:
        state['cleanup_errors'] = cleanup_errors
        error = primary_error if primary_error is not None else RuntimeError('Cleanup failed: '+repr(cleanup_errors))
        fail_output(budget, state, error)
        raise error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true', required=True,
                        help='Only after full review, rented CPU tests and fresh resource checks.')
    parser.parse_args()
    main()
