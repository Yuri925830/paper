# 只读检查声码器对音频幅度的适应情况；不创建优化器、不反传、不改模型或原数据。
import csv  # 按原训练清单顺序找到 pilot 中登记的语音。
import hashlib  # 给文件和实际计算数组保存身份摘要。
import importlib.util  # 直接载入原预处理脚本的 mel 函数，不另写一套 mel 算法。
import json  # 将逐句测量和汇总保存为新的诊断文件。
import sys  # 先检查操作系统，再允许导入远端项目。
import time  # 记录本次检查的起止时间。
from pathlib import Path  # 所有读取和结果位置均使用明确路径。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一允许运行项目计算的出租服务器目录。


def main():  # 导入本文件不会开始计算，只有直接执行才进入诊断。
    if not sys.platform.startswith('linux') or not ROOT.is_dir():  # Windows 或缺少远端工程立即拒绝。
        raise RuntimeError('This diagnostic can run only on the registered Matpool checkout.')  # 不提供本地兜底。
    sys.dont_write_bytecode = True  # 导入只读代码时也不额外写入 __pycache__。
    sys.path.insert(0, str(ROOT))  # 后续只导入租用服务器现场项目代码。
    from voicegrad.runtime_guard import assert_rented_server  # 用登记的主机名和机器指纹二次确认。
    assert_rented_server(ROOT)  # 先通过出租主机身份检查，再导入计算依赖。
    import librosa  # 精确沿用自训声码器的整句峰值归一化函数。
    import numpy as np  # 读取数组、整理连续内存和计算八句等权汇总。
    import soundfile as sf  # 从原 WAV 读取单精度波形及采样率。
    import torch  # 只做不保留梯度的 mel 计算和冻结声码器前向。
    import torch.nn.functional as F  # 频谱测量沿用原 mel 的反射补边规则。
    from fastvoicegrad.config import PAPER_STUDENT_CONFIG  # 复用声码器步数和结构的严格检查设置。
    from fastvoicegrad.components import load_frozen_vocoder, sha256_file  # 不绕开权重、配置和证明校验。
    from fastvoicegrad.train import acquire_runtime_locks  # 防止与训练或其他独占诊断冲突。
    locks = acquire_runtime_locks(ROOT)  # 将锁对象保留到函数结束；没有启动任何训练。
    out = ROOT / 'exp/diagnostics/student_stop_audit_20260906/amplitude_results.json'  # 与已有 results.json 分开。
    if out.exists():  # 绝不覆盖此前完成的这项诊断。
        raise RuntimeError(f'Refusing to overwrite {out}')  # 重跑需由主代理另行安排新结果位置。
    torch.set_num_threads(4)  # 限制 CPU 特征计算线程，避免不必要的资源争用。
    if not torch.cuda.is_available():  # 没有出租 GPU 就停止，不改为其他服务器运行。
        raise RuntimeError('The registered rental GPU is unavailable.')  # 保留明确失败原因。
    device = torch.device('cuda')  # 仅声码器前向放到出租 GPU，每次处理一句。
    began = time.time()  # 总耗时包含读取、检查和两路前向。
    pilot_path = ROOT / 'exp/diagnostics/acdd_repair_20260906/diagnostic_manifest.json'  # 读取原 pilot 样本身份。
    train_path = ROOT / 'data/metadata/train_manifest.csv'  # 索引只能在原训练清单解释。
    contract_path = ROOT / 'exp/checkpoints/fastvoicegrad_vctk_text_unseen_repair_x0_sqrt_20260906/run_contract.json'  # 锁定正式训练的数据版本。
    pilot = json.loads(pilot_path.read_text(encoding='utf-8'))  # 只读已有诊断设计。
    contract = json.loads(contract_path.read_text(encoding='utf-8'))  # 只读已停止训练的契约。
    if sha256_file(train_path) != contract['data']['fingerprints']['train_manifest.csv']:  # 避免清单排序改变后索引指向别句。
        raise RuntimeError('The train manifest differs from the stopped run.')  # 数据版本不符时不做混合比较。
    with train_path.open(encoding='utf-8', newline='') as handle:  # 不重排清单，保持原 dataset 索引含义。
        rows = list(csv.DictReader(handle))  # 训练清单中的每行对应一个语音条目。
    indices = pilot['dev_indices'][:8]  # 按要求固定取原 pilot 的前八句完整录音。
    if len(indices) != 8 or len(set(indices)) != 8:  # 样本数量与唯一性必须符合本次设计。
        raise RuntimeError('Exactly eight distinct original pilot indices are required.')  # 不悄悄补选其他样本。
    if any(type(index) is not int or not 0 <= index < len(rows) for index in indices):  # 所有索引须能在当前清单解释。
        raise RuntimeError('Invalid pilot index in the pinned train manifest.')  # 拒绝越界或非整数索引。
    selected = [rows[index] for index in indices]  # 不采用随机裁剪，八句都使用完整录音。
    if any(row['split'] != 'train' or row['is_unseen_speaker'] not in {'0', 'False', 'false'} for row in selected):  # 检查没有混入测试数据。
        raise RuntimeError('This amplitude check accepts only the original training-source samples.')  # 不以正式测试集选择方案。
    feature_path = ROOT / 'preprocess/03_extract_mel_22050.py'  # A/B 及输出评价共用这份现场原函数。
    spec = importlib.util.spec_from_file_location('amplitude_original_mel', feature_path)  # 为数字开头的脚本建立可导入名称。
    if spec is None or spec.loader is None:  # 无法载入原实现时不使用替代算法。
        raise RuntimeError('Cannot import the original preprocessing mel function.')  # 保持特征定义明确。
    feature = importlib.util.module_from_spec(spec)  # 建立模块对象，此时尚未执行预处理任务。
    spec.loader.exec_module(feature)  # 原文件的 main 保护确保只载入函数，不批量提取特征。
    acoustic = {  # 明确记录两路完全相同的声学参数，名称直接对应原函数参数。
        'n_fft': feature.N_FFT,  # 每个频谱窗使用 1024 点 FFT。
        'num_mels': feature.N_MELS,  # 使用原来的 80 个 mel 频带。
        'sampling_rate': feature.SR,  # 音频固定为 22050 Hz。
        'hop_size': feature.HOP_SIZE,  # 相邻帧间隔 256 个采样点。
        'win_size': feature.WIN_SIZE,  # 分析窗长度仍为 1024。
        'fmin': feature.FMIN,  # 最低频率沿用现场设置。
        'fmax': feature.FMAX,  # 最高频率沿用现场设置，不为 B 路另改频带。
    }
    if acoustic != dict(n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000):  # 阻止无意换了特征定义。
        raise RuntimeError(f'Unexpected current mel recipe: {acoustic}')  # 显示实际参数以便审查。

    def array_sha(value):  # 记录实际参与运算的单精度数组，方便之后复放。
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()  # 形状另在每句记录中保存。

    def log_mel(wave):  # 输入和生成输出都在 CPU 使用原预处理函数，避免设备差异污染缓存核对。
        return feature.mel_spectrogram_torch(wave, **acoustic)  # 得到未做 z-score 的自然对数 mel。

    def magnitude(wave):  # 额外记录不含相位的线性频率幅度谱误差。
        padding = (acoustic['n_fft'] - acoustic['hop_size']) // 2  # 与原 mel 一样，两侧各反射补 384 点。
        padded = F.pad(wave[None, None], (padding, padding), mode='reflect').squeeze(1)  # 两路和参考使用同样边界。
        spectrum = torch.stft(  # 仅查看频谱幅度，不逐点比较生成波形的相位。
            padded, n_fft=acoustic['n_fft'], hop_length=acoustic['hop_size'],  # FFT 和帧移与原 mel 相同。
            win_length=acoustic['win_size'], window=torch.hann_window(acoustic['win_size']),  # 采用相同周期 Hann 窗。
            center=False, normalized=False, onesided=True, return_complex=True,  # 手动补边后不再居中或额外归一化。
        )
        return torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-9)  # 沿用原频谱稳定项。

    def wave_stats(wave):  # 幅度统计不修改波形，也不做额外响度对齐。
        return {  # 同时保留峰值、总体能量和接近削波的比例。
            'rms': float(wave.square().mean().sqrt()),  # 整句均方根，反映音量。
            'peak': float(wave.abs().max()),  # 整句最大绝对幅度。
            'dc_mean': float(wave.mean()),  # 观察是否出现直流偏置。
            'abs_ge_0_999_fraction': float((wave.abs() >= 0.999).float().mean()),  # 只是接近幅度上限的比例，不直接断言已削波。
            'abs_gt_1_fraction': float((wave.abs() > 1.0).float().mean()),  # 还原 B 路增益后是否超出标准范围。
        }

    def measure(output, reference_mel, reference_mag, reference_rms):  # 所有质量误差都与同一原幅度参考比较。
        reconstructed_mel = log_mel(output)  # 不缩放或 clip 输出后再测量，以免隐藏幅度误差。
        reconstructed_mag = magnitude(output)  # 与参考使用同长度、同帧中心的幅度谱。
        delta_mel = (reconstructed_mel - reference_mel).abs()  # 固定的原始 log-mel 误差，不经过训练判别器。
        delta_mag = reconstructed_mag - reference_mag  # 只比较模长，不比较复数相位。
        values = wave_stats(output)  # 保存输出本身的幅度信息。
        values.update({  # 所有误差均越小越接近固定原录音。
            'raw_log_mel_l1': float(delta_mel.mean()),  # 原始自然对数 mel 的整句平均绝对误差。
            'raw_log_mel_l1_inner_4frames': float(delta_mel[..., 4:-4].mean()),  # 同时看去掉两端四帧的内部区域，暴露边界影响。
            'spectral_magnitude_mae': float(delta_mag.abs().mean()),  # 线性频率幅度谱的平均绝对误差。
            'spectral_magnitude_relative_l1': float(delta_mag.abs().sum() / reference_mag.abs().sum()),  # 用参考总幅度消除各句能量差异。
            'spectral_convergence': float(torch.linalg.vector_norm(delta_mag) / torch.linalg.vector_norm(reference_mag)),  # 幅度谱相对二范数误差。
            'rms_ratio_to_reference': values['rms'] / reference_rms,  # 1 表示总体幅度与原录音接近。
        })
        return values  # 不计算相位敏感的 waveform L1，也不生成主观评价。

    source_hashes = {str(path): sha256_file(path) for path in (pilot_path, train_path, contract_path, feature_path)}  # 固定输入设计和源码身份。
    prepared = []  # 先完成八句缓存核对，再允许 GPU 合成。
    with torch.inference_mode():  # 特征读取和计算均不创建梯度图。
        for index, row in zip(indices, selected):  # 保持原 pilot 前八句顺序。
            wav_path = Path(row['wav22k_path']).resolve()  # 只读取训练清单实际指向的音频。
            cache_path = ROOT / 'data/mel' / row['speaker'] / (row['utt_id'] + '.npy')  # 找到同句已登记的原始 mel。
            if ROOT / 'data/wav22k' not in wav_path.parents:  # 不接受指向工程以外音频的临时清单。
                raise RuntimeError(f'Unexpected source WAV path: {wav_path}')  # 防止无意扩大读取范围。
            source_hashes[str(wav_path)] = sha256_file(wav_path)  # 记录原 WAV 的文件摘要。
            source_hashes[str(cache_path)] = sha256_file(cache_path)  # 记录被核对的缓存数组文件摘要。
            full, sr = sf.read(str(wav_path), dtype='float32')  # 与预处理和自训声码器相同的浮点读取方式。
            if full.ndim == 2:  # 多声道时与原预处理一样取通道均值。
                full = full.mean(axis=1)  # 之后 A/B 和参考都使用同一单声道结果。
            if sr != feature.SR or full.ndim != 1 or not np.isfinite(full).all():  # 采样率、形状或数值异常即停止。
                raise RuntimeError(f'Invalid source WAV: {wav_path}')  # 不临时重采样或修复数据。
            if float(np.max(np.abs(full))) > 1.0:  # 超范围输入会使原预处理 clip 与声码器训练规则出现额外差别。
                raise RuntimeError(f'Out-of-range source needs a separate audit: {wav_path}')  # 本次只检验幅度缩放，避免引入第二个变量。
            full = np.ascontiguousarray(np.clip(full, -1.0, 1.0), dtype=np.float32)  # 与第 03 步保持一致；上述门禁保证 clip 不改变有效值。
            peak = float(np.max(np.abs(full)))  # 在截掉尾部不足一帧前计算整句峰值，符合自训规则。
            if peak <= float(librosa.util.tiny(full)):  # 全静音或极小输入不能定义可靠的峰值还原增益。
                raise RuntimeError(f'Near-silent source cannot support this gain comparison: {wav_path}')  # 不人为规定静音增益。
            gain = 0.95 / peak  # B 路提取 mel 前的整句放大倍数，合成后须除回同一个数。
            normalized_full = np.ascontiguousarray(librosa.util.normalize(full) * 0.95, dtype=np.float32)  # 精确复用自训声码器的归一化次序。
            usable = len(full) // feature.HOP_SIZE * feature.HOP_SIZE  # 两路一起截到相同 hop 整数倍，沿用原缓存边界。
            original = torch.from_numpy(np.ascontiguousarray(full[:usable]))  # 原幅度参考与 A 路输入使用相同音频。
            normalized = torch.from_numpy(np.ascontiguousarray(normalized_full[:usable]))  # B 路只改变幅度，不改变句长或裁剪起点。
            if usable // feature.HOP_SIZE <= 8:  # 确保内部去边界指标仍有真实帧可测。
                raise RuntimeError(f'Source is too short for the predeclared comparison: {wav_path}')  # 不让空张量均值变成 NaN。
            mel_a, mel_b = log_mel(original), log_mel(normalized)  # 两路使用完全同一函数及参数。
            cached = torch.from_numpy(np.load(cache_path, allow_pickle=False)).unsqueeze(0)  # 缓存约定 [80,T]，补一个批次轴。
            if cached.shape != mel_a.shape or not torch.isfinite(cached).all():  # 缓存形状错误或坏值不能直接比较。
                raise RuntimeError(f'Cached mel is malformed: {cache_path}')  # 禁止自动转置或截短掩盖差异。
            cache_max = float((cached - mel_a).abs().max())  # 记录原幅度重算与缓存的最大差。
            if not torch.allclose(cached, mel_a, rtol=0.0, atol=2e-5):  # 明确绝对容差，不能靠较大负 log 数值放宽检查。
                raise RuntimeError(f'Recomputed mel differs from cache: {cache_path}; max={cache_max}')  # 特征不一致时不继续声码器比较。
            metadata = {  # 保存样本、增益和缓存核对结果，方便结果可追溯。
                'train_index': index, 'key': row['speaker'] + '/' + row['utt_id'],  # 原清单位置和语音身份。
                'wav_path': str(wav_path), 'cached_mel_path': str(cache_path),  # 原输入文件位置。
                'original_samples': len(full), 'compared_samples': usable, 'dropped_tail_samples': len(full) - usable,  # 只去掉不足一帧的共同尾部。
                'mel_frames': int(mel_a.shape[-1]), 'full_utterance_peak': peak, 'peak095_gain': gain,  # 明确音量缩放是在整句计算。
                'cache_max_abs_difference': cache_max, 'cache_mean_abs_difference': float((cached - mel_a).abs().mean()),  # 证明测量尺度没有换算法。
                'mel_a_sha256': array_sha(mel_a.numpy()), 'mel_b_sha256': array_sha(mel_b.numpy()),  # 锁定实际输入 G 的两份 mel。
                'reference': wave_stats(original),  # 两路比较对象始终是这份原幅度录音。
            }
            prepared.append((metadata, original, mel_a, mel_b))  # CPU 缓存八句输入，不预占 GPU 特征空间。

    vd = ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen'  # 唯一自训声码器来源。
    weight_path = vd / 'generator_step_2500000.pt'  # 固定 250 万步生成器，不切换其他权重。
    config_path = vd / 'config.json'  # 使用权重配套配置。
    proof_path = vd / 'final_checkpoint_attestation_comments_20260906.json'  # 使用注释等价性核验后的独立证明。
    for path in (weight_path, config_path, proof_path):  # 记录校验所依赖的模型文件身份。
        source_hashes[str(path)] = sha256_file(path)  # 结束前再次核对这些只读文件没有变化。
    vocoder, vconfig, checkpoint = load_frozen_vocoder(ROOT, weight_path, config_path, proof_path, PAPER_STUDENT_CONFIG, device)  # 严格验证后只前向使用。
    del checkpoint  # 不保留多余的 CPU 权重副本。
    if any(parameter.requires_grad for parameter in vocoder.parameters()):  # 附加检查全部参数已冻结。
        raise RuntimeError('Vocoder parameters must remain frozen.')  # 不允许有可训练参数进入本诊断。
    if any(vconfig[key] != acoustic[alias] for key, alias in [('sampling_rate', 'sampling_rate'), ('num_mels', 'num_mels'), ('n_fft', 'n_fft'), ('hop_size', 'hop_size'), ('win_size', 'win_size'), ('fmin', 'fmin'), ('fmax', 'fmax')]):  # 对照声码器配置与原 mel 参数。
        raise RuntimeError('Vocoder and preprocessing acoustic settings differ.')  # 不通过临时改参数补救。
    results = []  # 每句单独保留 A/B，避免平均数隐藏例外。
    with torch.inference_mode():  # 生成和评价均关闭梯度；没有 optimizer 或 backward。
        for metadata, original, mel_a, mel_b in prepared:  # 每次只合成一句两路，控制显存占用。
            output_a = vocoder(mel_a.to(device)).detach().cpu().reshape(-1)  # A：原幅度 mel 直接合成。
            output_b_native = vocoder(mel_b.to(device)).detach().cpu().reshape(-1)  # B：符合自训峰值范围的 mel 合成。
            output_b = output_b_native / metadata['peak095_gain']  # 恢复到原音频幅度后再公平比较。
            if output_a.shape != original.shape or output_b.shape != original.shape:  # 绝不额外截断、移位或重采样输出。
                raise RuntimeError(f'Vocoder length mismatch: {metadata["key"]}')  # 长度不符会改变频谱边界，必须另查。
            if not torch.isfinite(output_a).all() or not torch.isfinite(output_b).all():  # 排除非有限输出。
                raise RuntimeError(f'Non-finite vocoder output: {metadata["key"]}')  # 不把 NaN 汇总成貌似正常的分数。
            reference_mag = magnitude(original)  # 两路共用同一个原幅度参考谱。
            metadata['A_raw_mel'] = measure(output_a, mel_a, reference_mag, metadata['reference']['rms'])  # A 路固定参考误差。
            metadata['B_peak095_mel_then_undo_gain'] = measure(output_b, mel_a, reference_mag, metadata['reference']['rms'])  # B 路与同一 mel_a 比较。
            metadata['B_before_undo_gain'] = wave_stats(output_b_native)  # 保留还原音量前的波形幅度，检查生成器是否已接近饱和。
            results.append(metadata)  # 收集完整逐句结果。
            print('AMPLITUDE_RESULT ' + json.dumps(metadata, ensure_ascii=False), flush=True)  # 只打印本句结果，不改旧日志。
    metric_names = list(results[0]['A_raw_mel'])  # A/B 两路具有相同指标字段。
    summary = {label: {key: float(np.mean([row[label][key] for row in results])) for key in metric_names} for label in ('A_raw_mel', 'B_peak095_mel_then_undo_gain')}  # 八句等权平均，不用长句压过短句。
    for path_text, expected_hash in source_hashes.items():  # 结束前再次确认原数据、模型及源码未被并发改变。
        if sha256_file(Path(path_text)) != expected_hash:  # 内容有变化时拒绝发布完整报告。
            raise RuntimeError(f'Input changed during the diagnostic: {path_text}')  # 不修改账本来接受漂移。
    report = {  # 结果仅用于幅度兼容性诊断，不能写成正式 text-unseen 指标。
        'purpose': 'Read-only full-utterance vocoder amplitude compatibility check; eight training-source pilot sentences.',  # 明确样本与用途。
        'started_unix': began, 'elapsed_seconds': time.time() - began, 'sample_count': len(results),  # 保存运行时间和实际覆盖。
        'script_sha256': sha256_file(Path(__file__).resolve()), 'source_sha256': source_hashes,  # 绑定脚本与全部被读取的关键资源。
        'acoustic_parameters': acoustic, 'cache_absolute_tolerance': 2e-5,  # 记录固定特征定义及缓存核对门槛。
        'normalization_order': 'Full-utterance peak normalization to 0.95 before identical hop-multiple truncation; divide generated B waveform by the same gain.',  # 不能按片段重新估计增益。
        'summary_reduction': 'Equal mean over eight utterances; per-utterance frame/sample means inside each metric.',  # 明确汇总分母。
        'summary': summary, 'per_utterance': results,  # 同时保留均值与全部明细。
        'limitations': ['Only eight training-source utterances; no formal cross-speaker VC or subjective evaluation.', 'No student weights are used; this does not establish the cause of rising student loss.', 'STFT errors use magnitudes only; sample-wise waveform L1 is intentionally absent.', 'Near-ceiling sample fraction is an amplitude diagnostic, not proof of audible clipping.'],  # 限制结论外推范围。
    }
    out.parent.mkdir(parents=True, exist_ok=True)  # 只创建本次诊断结果目录，不触碰数据或检查点目录。
    with out.open('x', encoding='utf-8') as handle:  # 排他新建；即使中途出现同名文件也不覆盖。
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 非有限汇总不允许写成完整 JSON。
    print('AMPLITUDE_DIAGNOSTIC_COMPLETE ' + str(out), flush=True)  # 完成后通知主代理读取新结果。


if __name__ == '__main__':  # 导入本脚本不会运行诊断或加载任何模型。
    main()  # 唯一执行路径含租用机门禁，完全没有训练更新。
