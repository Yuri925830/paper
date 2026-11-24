"""按固定开发配对和正式测试配对评估新 FastVoiceGrad 实验。"""  # 本模块只在 Colab 的训练入口调用后执行模型计算。
import argparse  # 读取项目、实验和断点的位置。
import csv  # 保存每一条转换的真实评分和来源。
import hashlib  # 给实验断点和评估输入登记内容身份。
import json  # 原子保存评估状态与汇总结果。
import math  # 拒绝非有限数分数，避免错误成绩进入报告。
import os  # 完整写入后一次替换正式文件。
import sys  # 导入同一发布目录中的训练模块。
import time  # 测量真实推理速度。
from dataclasses import asdict  # 将旧评估配对对象转换为普通字段。
from pathlib import Path  # 安全处理中文云盘路径。

def atomic_json(path, value):  # 先写完整临时文件，再提交正式 JSON。
    path = Path(path)  # 接受字符串和路径对象。
    path.parent.mkdir(parents=True, exist_ok=True)  # 首次评估时创建输出目录。
    temporary = path.with_name(path.name + ".writing")  # 未完成写入不使用正式文件名。
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # 拒绝 NaN 并保存可读 UTF8。
    os.replace(temporary, path)  # 完整写入后原子替换结果。

def file_sha256(path):  # 分块读取文件，避免大断点占用双份内存。
    digest = hashlib.sha256()  # 建立当前文件的摘要对象。
    with Path(path).open("rb") as handle:  # 只读打开真实文件。
        for block in iter(lambda: handle.read(8 * 1024**2), b""):  # 每次最多读取八 MiB。
            digest.update(block)  # 将当前数据块加入摘要。
    return digest.hexdigest()  # 返回文件内容指纹。

def read_csv(path):  # 读取实际保存的 CSV 行。
    if not Path(path).is_file():  # 首次运行没有旧评分文件。
        return []  # 明确表示没有已完成条目。
    with Path(path).open("r", encoding="utf-8", newline="") as handle:  # 使用 CSV 标准换行规则。
        return list(csv.DictReader(handle))  # 返回具名字段记录。

def relocate_wave(root, value):  # 把历史服务器波形路径映射到当前项目。
    value = str(value)  # 将清单字段统一为字符串。
    marker = "/data/"  # 历史数据路径包含稳定的项目相对部分。
    if marker in value:  # 只迁移确实属于数据树的路径。
        return str(Path(root) / "data" / value.split(marker, 1)[1])  # 保留 wav22k 下原说话人与句号。
    return str(Path(value))  # 对其他明确路径保持原样。

def load_pairs(root, output, split):  # 固定开发九十对或测试八千一百对。
    root, output = Path(root), Path(output)  # 统一路径类型。
    if split == "development":  # 开发转换只使用预先固定的开发配对。
        rows = read_csv(root / "exp/diagnostics/student_development_20260906/pairs.csv")  # 不在评估时重新挑选语音。
        if len(rows) != 90:  # 开发配对数量必须等于既定协议。
            raise RuntimeError(f"开发转换必须有90对，实际为{len(rows)}。")  # 不把少量诊断伪装成完整开发评估。
    else:  # 测试采用原项目严格不同说话人且不同句子的配对规则。
        from voicegrad.infer_voicegrad30_paper_eval import load_test_rows, build_pairs  # 只复用已审计的配对函数。
        test_rows = load_test_rows(root / "data/metadata/test_manifest.csv")  # 严格检查十人十句测试网格。
        rows = [asdict(pair) for pair in build_pairs(test_rows, output)]  # 按原排序生成八千一百个确定配对。
    for index, row in enumerate(rows):  # 为当前评估目录填入稳定输出路径。
        row["pair_id"] = int(row["pair_id"])  # 保留原配对编号。
        if row["pair_id"] != index:  # 编号必须连续且顺序不可漂移。
            raise RuntimeError("评估配对编号不连续。")  # 避免续算结果套到错误语音上。
        row["source_wav"] = relocate_wave(root, row["source_wav"])  # 映射原始源语音路径。
        row["reference_wav"] = relocate_wave(root, row["reference_wav"])  # 映射真实目标参考语音路径。
        row["output_wav"] = str(output / "wav" / f"pair_{index:04d}.wav")  # 将新生成声音保存到独立实验目录。
    return rows  # 返回固定且来源明确的配对表。

def write_pairs(path, rows):  # 保存后续指标直接读取的最小标准配对表。
    fields = ("pair_id", "source_speaker", "source_utt", "source_text", "source_wav", "reference_speaker", "reference_utt", "reference_wav", "output_wav")  # 保留内容、身份和输出来源。
    temporary = Path(path).with_name(Path(path).name + ".writing")  # 避免中断后留下半份正式清单。
    with temporary.open("w", encoding="utf-8", newline="") as handle:  # 顺序生成完整 CSV。
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")  # 忽略开发清单额外的审计字段。
        writer.writeheader()  # 写入固定表头。
        writer.writerows(rows)  # 保留确定的配对顺序。
    os.replace(temporary, path)  # 完整写好后提交。

def recover_records(path):  # 读取已提交记录，允许最后一次断线留下未完成尾行。
    records = {}  # 每个 pair_id 只保留唯一的完整记录。
    if not Path(path).is_file():  # 初次评估没有可恢复日志。
        return records  # 返回空进度。
    lines = Path(path).read_text(encoding="utf-8").splitlines()  # 读取小型逐配对日志以定位尾部中断。
    accepted = []  # 保留可以确认的完整前缀。
    for index, line in enumerate(lines):  # 按原记录顺序解析。
        try:  # 最后一行可能因虚拟机中断没有写完。
            record = json.loads(line)  # 解码真实完成记录。
        except json.JSONDecodeError:  # 只有最后一行不完整才属于正常断线恢复。
            if index != len(lines) - 1:  # 中间损坏不能静默隐藏。
                raise RuntimeError(f"评估日志中间损坏：{path}第{index + 1}行。")  # 显示需要排查的真实损坏位置。
            backup = Path(path).with_name(Path(path).name + f".interrupted_tail_{time.time_ns()}")  # 保留未完成尾行供追溯。
            backup.write_text(line, encoding="utf-8")  # 不永久删除中断证据。
            temporary = Path(path).with_name(Path(path).name + ".writing")  # 在另一文件中整理完整前缀。
            temporary.write_text("\n".join(accepted) + ("\n" if accepted else ""), encoding="utf-8")  # 防止后续追加与坏尾行粘连。
            os.replace(temporary, path)  # 提交可安全追加的完整日志。
            break  # 下一次评分会重新计算未提交配对。
        records[int(record["pair_id"])] = record  # 同一配对重算时采用最近完整记录。
        accepted.append(line)  # 保存已确认的前缀原文。
    if records and Path(path).read_bytes()[-1:] != b"\n":  # 合法JSON也可能在最后换行写入前断线。
        with Path(path).open("a", encoding="utf-8") as handle:  # 保留完整最后一行，只补上安全分隔符。
            handle.write("\n")  # 防止下一条完整记录与上一行连接成坏JSON。
    return records  # 返回真实已完成进度。

def validate_metric_counts(rows, result_rows, label):  # 汇总前确认每一条配对都已有真实评分。
    expected = {int(row["pair_id"]) for row in rows}  # 使用实际配对集合。
    actual = {int(row["pair_id"]) for row in result_rows}  # 读取已评分集合。
    if len(result_rows) != len(expected) or actual != expected:  # 重复、漏项和额外条目都不允许。
        raise RuntimeError(f"{label}评估尚未完整：{len(actual)}/{len(expected)}。")  # 保留已完成结果供下一次续算。

def score_p808(runtime, rows, output):  # DNSMOS只保存用户选定的P.808输出。
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score  # 使用固定微软DNSMOS模型封装。
    import soundfile as sf  # 读取实际生成的音频，不使用mel代理音质。
    from tqdm.auto import tqdm  # 显示逐语音评分进度。
    path = Path(output) / "dnsmos_p808.jsonl"  # 输出只含P.808分数。
    records = recover_records(path)  # 恢复已完成语音编号。
    todo = [row for row in rows if int(row["pair_id"]) not in records]  # 已完成条目不再计算。
    with path.open("a", encoding="utf-8") as handle:  # 每个评分完成后立即持久保存。
        for row in tqdm(todo, total=len(todo), desc="DNSMOS P.808", unit="语音"):  # 显示真实剩余量。
            waveform, sr = sf.read(row["output_wav"], dtype="float32")  # 保持生成波形的真实幅度。
            if sr != 22050 or waveform.ndim != 1:  # 评分输入必须为单声道22.05kHz。
                raise RuntimeError("DNSMOS收到错误的波形采样率或声道数。")  # 拒绝偷偷归一化或改变音频。
            tensor = runtime.torch.from_numpy(waveform).unsqueeze(0)  # 增加单条batch维度。
            values = deep_noise_suppression_mean_opinion_score(tensor, fs=sr, personalized=False, device="cpu", num_threads=2)  # 使用非个性化官方DNSMOS。
            p808 = float(values.reshape(-1, 4)[0, 0])  # 按官方接口第一列提取P.808。
            if not math.isfinite(p808):  # 异常模型输出不能被当作评分。
                raise RuntimeError("DNSMOS P.808产生非有限分数。")  # 保留此前成功进度并停止。
            record = {"pair_id": int(row["pair_id"]), "p808_mos": p808}  # 不保存其他DNSMOS输出。
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")  # 持久追加真实分数。
            handle.flush()  # 每条分数完成后立即交给Drive文件系统。
            records[record["pair_id"]] = record  # 更新本次进程的完成集合。
    validate_metric_counts(rows, list(records.values()), "DNSMOS P.808")  # 全部完成后才返回。
    return list(records.values())  # 返回实际评分，交给统一汇总。

def score_all(runtime, rows, output, root):  # 使用原项目固定评估器计算所有报告指标。
    import evaluation.evaluate_metrics as metrics  # 复用已审计UTMOS、CER、ECAPA逐条评分函数。
    output, root = Path(output), Path(root)  # 统一目录类型。
    device = runtime.torch.device("cuda")  # 与正式转换使用同一GPU环境。
    metrics.evaluate_utmos(rows, output / "utmos.csv", root / "UTMOS-demo", device, 4)  # 保持官方strong learner及原评分变换。
    p808 = score_p808(runtime, rows, output)  # 仅计算并报告用户选定的P.808字段。
    metrics.evaluate_cer(rows, output / "cer.csv", device, 4)  # 使用原LV60K-960h ASR和文本标准化。
    metrics.evaluate_sva(rows, output / "sva.csv", root / "pretrained_spkrec_ecapa", device, 4)  # 保持ECAPA余弦阈值严格大于0.25。
    utmos, cer, sva = [read_csv(output / name) for name in ("utmos.csv", "cer.csv", "sva.csv")]  # 读取所有真实逐条评分。
    for label, values in (("UTMOS", utmos), ("CER", cer), ("SVA", sva)):  # 同时检查所有指标覆盖。
        validate_metric_counts(rows, values, label)  # 缺项不得写入完成汇总。
    n = len(rows)  # 分母采用真实配对数量。
    characters = sum(int(row["reference_chars"]) for row in cer)  # 语料CER按总正确字符数加权。
    if characters <= 0:  # 无参考文本时无法定义CER。
        raise RuntimeError("CER参考文本总字符数为零。")  # 不构造虚假的零错误率。
    return {"count": n, "utmos": sum(float(row["utmos"]) for row in utmos) / n, "dnsmos_p808": sum(float(row["p808_mos"]) for row in p808) / n, "corpus_cer_percent": 100 * sum(int(row["char_errors"]) for row in cer) / characters, "utterance_cer_percent": sum(float(row["cer_percent"]) for row in cer) / n, "sva_percent": 100 * sum(float(row["cosine_score"]) > 0.25 for row in sva) / n, "ecapa_cosine": sum(float(row["cosine_score"]) for row in sva) / n, "sva_threshold": 0.25}  # 汇总所有报告规定客观指标。

def time_end_to_end(runtime, components, rows, output, root, seed):  # 在固定32对测试语音上真实计时完整原始波形到转换波形的流程。
    import soundfile as sf  # 在计时范围内读取真实源波形。
    from tqdm.auto import tqdm  # 显示端到端计时样本的进度。
    from fastvoicegrad_formal import load_python  # 复用明确路径的原模块加载器。
    torch, device = runtime.torch, components["device"]  # 使用与正式转换相同的GPU及精度。
    mel_helper = load_python(Path(root) / "preprocess/03_extract_mel_22050.py", "formal_timing_mel")  # 使用原80通道mel定义。
    spk_helper = load_python(Path(root) / "preprocess/06_extract_spk_emb.py", "formal_timing_speaker")  # 使用原说话人前端兼容代码。
    from speaker_encoder import audio as speaker_audio  # 导入与现有256维embedding相同的GE2E音频处理包。
    from speaker_encoder.voice_encoder import SpeakerEncoder  # 使用原固定身份编码器。
    checkpoint = Path(root) / "external/legacy_compute_bnf/speaker_encoder/ckpt/pretrained_bak_5805000.pt"  # 端到端计时也要求原始权重存在。
    if file_sha256(checkpoint) != spk_helper.EXPECTED_SPK_CHECKPOINT_SHA256:  # 防止使用其他提取器冒充原完整流程。
        raise RuntimeError("端到端RTF的GE2E权重身份不匹配。")  # 缺失或替换权重不能产生正式计时结果。
    preprocess_speaker = spk_helper.configure_legacy_audio_compat(speaker_audio)  # 保留原16k重采样、音量处理和静音筛除规则。
    encoder_module = sys.modules[SpeakerEncoder.__module__]  # 只兼容原GE2E模块的加载调用，不修改全局PyTorch。
    original_module_torch = encoder_module.torch  # 保存该作者模块原来的torch引用。
    class TrustedSpeakerLoad:  # 已核对固定SHA的自有权重需要读取完整作者字典。
        def __getattr__(self, name):  # 所有计算接口仍委托给真实PyTorch。
            return getattr(torch, name)  # 不替换网络运算或数值。
        def load(self, *values, **options):  # 唯一兼容新版默认weights_only的文件加载行为。
            options.setdefault("weights_only", False)  # 固定可信检查点可能含作者元数据。
            return torch.load(*values, **options)  # 真正加载仍由原PyTorch执行。
    encoder_module.torch = TrustedSpeakerLoad()  # 临时局部代理只影响这个作者模块的初始化。
    try:  # 构造失败也必须恢复作者模块原引用。
        encoder = SpeakerEncoder(checkpoint, device=device, verbose=False).eval()  # 模型加载不计入稳态单句推理时间。
    finally:  # 后续embed_utterance使用真正原torch接口。
        encoder_module.torch = original_module_torch  # 不留下长期或全局加载补丁。
    selected = rows[:min(32, len(rows))]  # 固定取清单前32对，不挑选较快或较短的语音。
    path = Path(output) / "end_to_end_timing.jsonl"  # 每个实际完成的计时条目可在断线后复用。
    records = recover_records(path)  # 恢复已提交的固定计时编号。
    def measure(row):  # 一次完整计时包括两份波形读取、全部内容身份前端和声学合成。
        torch.cuda.synchronize(device)  # 计时前等待之前的GPU任务。
        started = time.monotonic()  # 从实际源波形读取前开始计时。
        with torch.inference_mode():  # 稳态推理完全不更新任何参数。
            waveform, sr = sf.read(row["source_wav"], dtype="float32")  # 源wav读取也计入端到端成本。
            if sr != 22050 or waveform.ndim != 1:  # 沿用原数据22.05k单声道协议。
                raise RuntimeError("RTF源波形格式与原实验不一致。")  # 不偷偷更改测试语音。
            source = torch.from_numpy(waveform[:len(waveform) // 256 * 256]).clamp(-1, 1).to(device)  # 按原前端裁幅并截到hop256整数倍。
            natural_mel = mel_helper.mel_spectrogram_torch(source, 1024, 80, 22050, 256, 1024, 0, 8000)  # 真实计算原log-mel而非读取缓存。
            mel = (natural_mel - components["mean"]) / components["std"]  # 采用原模型训练统计进行标准化。
            speech = runtime.torchaudio.functional.resample(torch.from_numpy(waveform).to(device), 22050, 16000).unsqueeze(0)  # 源完整波形重采样到内容提取器的16k输入。
            lengths = torch.tensor([speech.shape[-1]], device=device, dtype=torch.long)  # 指定原始波形有效长度。
            raw = components["bridge"].model(speech, lengths)[0].transpose(0, 1)  # 真实提取源语音144维Conformer内容。
            bnf = runtime.align_bnf_to_mel(raw, mel.shape[-1], mode="legacy_length_linear").unsqueeze(0)  # 保留原内容与mel对齐方式。
            target = torch.from_numpy(encoder.embed_utterance(preprocess_speaker(row["reference_wav"]))).to(device).unsqueeze(0)  # 读取指定目标参考wav并真实提取256维身份。
            generator = torch.Generator(device=device).manual_seed(90000000 + int(seed) + int(row["pair_id"]))  # 使用与正常转换相同的确定噪声。
            noise = torch.randn(mel.shape, device=device, generator=generator)  # 端到端成本包含噪声生成。
            predicted, _ = runtime.training.make_student_prediction(components["student"], components["diffusion"], mel, target, bnf, 950, noise)  # 单次FastVoiceGrad声学转换。
            converted = components["vocoder"](runtime.training.denormalize_mel(predicted, components["mean"], components["std"])).reshape(-1)  # 真实合成并等待可用的输出波形。
            converted.cpu()  # 把完整波形返回主机也计入端到端成本，文件保存不计入。
        torch.cuda.synchronize(device)  # 完整声学流程完成后才停止计时。
        return {"pair_id": int(row["pair_id"]), "total_vc_seconds": time.monotonic() - started, "audio_seconds": converted.numel() / 22050}  # 仅保存真实测得的运行时间与输出时长。
    for row in selected[:min(3, len(selected))]:  # 用固定前三对预热GPU及前端，不将预热结果计入正式计时。
        measure(row)  # 每次恢复都执行同样预热，避免首条冷启动污染比较。
    with path.open("a", encoding="utf-8") as handle:  # 每个计时样本成功后立即持久化。
        for row in tqdm(selected, desc="端到端RTF", unit="配对"):  # 进度条显示固定32对完整推理的完成比例。
            if int(row["pair_id"]) in records:  # 已完成计时不重复占用算力。
                continue  # 直接复用同模型同配对的真实记录。
            record = measure(row)  # 对当前固定配对进行完整端到端测量。
            if any(not math.isfinite(float(record[key])) or float(record[key]) <= 0 for key in ("total_vc_seconds", "audio_seconds")):  # 时间和时长必须有效且为正。
                raise RuntimeError("端到端RTF出现无效计时。")  # 不发布无法计算的速度指标。
            handle.write(json.dumps(record, allow_nan=False) + "\n")  # 保存当前真实计时。
            handle.flush()  # 每个配对完成后立即提交。
            records[record["pair_id"]] = record  # 更新本次完成集合。
    validate_metric_counts(selected, list(records.values()), "端到端RTF")  # 所有固定计时样本必须完成。
    return {"total_vc_rtf": sum(record["total_vc_seconds"] for record in records.values()) / sum(record["audio_seconds"] for record in records.values()), "total_vc_timing_count": len(selected), "total_vc_rtf_scope": "source_wav_io_mel_bnf_reference_wav_io_ge2e_student_vocoder_output_to_cpu", "total_vc_rtf_status": "complete_fixed_first_32_pairs_3_warmups"}  # 明确计时范围、样本数和预热协议。

def evaluate_conversion_in_memory(runtime, components, root, run_dir, split, seed, global_step):  # 训练程序在固定时点调用，不重复建立教师与判别器。
    import soundfile as sf  # 保存和检查真实生成音频。
    from tqdm.auto import tqdm  # 显示每一对转换进度。
    torch = runtime.torch  # 使用同一训练进程的PyTorch。
    root = Path(root)  # 模型和固定数据根目录。
    output = Path(run_dir) / "evaluation" / split / f"step_{int(global_step):09d}"  # 各checkpoint与集合分别存储结果。
    output.mkdir(parents=True, exist_ok=True)  # 不覆盖其他评估时点。
    (output / "wav").mkdir(exist_ok=True)  # 声音子目录用于转换续算。
    rows = load_pairs(root, output, split)  # 选择固定90或8100对。
    weight_digest = hashlib.sha256()  # 精确绑定当前学生参数，避免同一步其他模型误复用旧声音。
    for name, tensor in components["student"].state_dict().items():  # 顺序读取真实模型参数与缓冲区，不运行网络。
        weight_digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode("utf-8"))  # 参数名称、类型及形状也构成模型身份。
        weight_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())  # 对真实参数内容计算摘要。
    source_identity = {"step": int(global_step), "seed": int(seed), "split": split, "student_weights_sha256": weight_digest.hexdigest(), "pairs": [{key: row[key] for key in ("pair_id", "source_utt", "reference_utt")} for row in rows]}  # 将结果绑定当前模型内容、时点和配对。
    checkpoint_identity = Path(run_dir) / "run_contract.json"  # 加入训练合同以区分重建与微调实验。
    if checkpoint_identity.is_file():  # 新训练入口会先保存正式合同。
        source_identity["run_contract_sha256"] = file_sha256(checkpoint_identity)  # 防止同一目录误接不同实验。
    identity_path = output / "identity.json"  # 保存所有评分共享的模型与配对身份。
    if identity_path.is_file() and json.loads(identity_path.read_text(encoding="utf-8")) != source_identity:  # 历史结果必须严格属于当前评估请求。
        raise RuntimeError("评估目录已有其他实验结果，不能混用。")  # 保留旧实验而不是删除。
    atomic_json(identity_path, source_identity)  # 固定评估身份后再写声音。
    write_pairs(output / "pairs.csv", rows)  # 让后续指标读取同一份配对。
    dataset_split = "train" if split == "development" else "test"  # 开发语音源于排除的train索引。
    dataset = runtime.baseline.VoiceGradDataset(root / "data", split=dataset_split, segment_length=None, bnf_alignment=components["config"].bnf_alignment)  # 转换使用完整句子，不切训练片段。
    index_by_utt = {str(item["utt_id"]): index for index, item in enumerate(dataset.items)}  # 索引实际数据，不重新挑样本。
    student, vocoder = components["student"], components["vocoder"]  # 使用当前FastVoiceGrad与冻结声码器。
    device = next(student.parameters()).device  # 与当前训练GPU保持一致。
    modules = [student, vocoder]  # 记录评估结束后要恢复的模型模式。
    previous_modes = [module.training for module in modules]  # 不让评估模式改变下一批训练。
    rng = runtime.training.capture_rng_state()  # 开发计算不应消耗训练随机序列。
    records_path = output / "conversion.jsonl"  # 每对转换完成后立即保存。
    records = recover_records(records_path)  # 网络重连后从真实已完成编号继续。
    for module in modules:  # 关闭评估期间的随机训练层。
        module.eval()  # 只读评估不更新模型。
    try:  # 所有异常也必须恢复训练随机状态和模式。
        with records_path.open("a", encoding="utf-8") as handle:  # 保留此前转换记录。
            for row in tqdm(rows, total=len(rows), desc=f"{split}转换", unit="配对"):  # 每条语音显示百分比。
                pair_id = int(row["pair_id"])  # 当前固定配对编号。
                wav_path = Path(row["output_wav"])  # 当前输出波形的永久位置。
                old = records.get(pair_id)  # 查找真实完成记录。
                if old and wav_path.is_file() and file_sha256(wav_path) == old.get("wav_sha256"):  # 完成记录与声音内容同时一致才跳过。
                    continue  # 已提交配对不重复运行模型。
                source = dataset[index_by_utt[row["source_utt"]]]  # 读取完整源句的标准化mel和BNF。
                reference = dataset[index_by_utt[row["reference_utt"]]]  # 目标speaker embedding来自指定参考句。
                mel, bnf = [source[key].unsqueeze(0).to(device) for key in ("mel", "bnf")]  # 模型输入增加batch维度。
                if components.get("vq_quantizer") is not None:  # VQ小实验的转换必须使用与训练相同的冻结码本。
                    bnf = components["vq_quantizer"](bnf)  # 只量化内容条件，不改变mel或身份条件。
                source_spk = source["spk_emb"].unsqueeze(0).to(device)  # 返回转换使用原说话人身份。
                target_spk = reference["spk_emb"].unsqueeze(0).to(device)  # 第一次转换使用目标说话人身份。
                lengths = source["mel_len"].reshape(1).to(device)  # 全句有效帧，没有训练补零干扰。
                generator1 = torch.Generator(device=device).manual_seed(90000000 + int(seed) + pair_id)  # 第一次噪声与其他配对独立。
                generator2 = torch.Generator(device=device).manual_seed(100000000 + int(seed) + pair_id)  # 第二次噪声使用不同随机流。
                noise1 = torch.randn(mel.shape, device=device, generator=generator1)  # 固定每对首次转换噪声。
                noise2 = torch.randn(mel.shape, device=device, generator=generator2)  # 固定每对返回转换噪声。
                torch.cuda.synchronize(device)  # 测量前排空GPU异步任务。
                started = time.monotonic()  # 记录mel转换真实开始时间。
                with torch.inference_mode():  # 全部转换评分都不建立参数梯度。
                    generated, _ = runtime.training.make_student_prediction(student, components["diffusion"], mel, target_spk, bnf, start_step=950, source_noise=noise1)  # 严格K1从950到0。
                torch.cuda.synchronize(device)  # 等待该次转换真正完成。
                mel_seconds = time.monotonic() - started  # 记录纯mel网络推理时间。
                with torch.inference_mode():  # 声码器也保持冻结只读。
                    waveform = vocoder(runtime.training.denormalize_mel(generated, components["mean"], components["std"])).reshape(-1)  # 按训练统计恢复mel后还原波形。
                torch.cuda.synchronize(device)  # 等待波形合成结束。
                synthesis_seconds = time.monotonic() - started  # mel转换加vocoder的总推理时间。
                values = waveform[:int(lengths[0]) * 256].float().cpu().numpy()  # 只保存真实句长对应的波形。
                temporary = wav_path.with_name(wav_path.name + ".writing")  # 半份WAV不进入正式评分。
                sf.write(temporary, values, 22050, format="WAV", subtype="FLOAT")  # 保存真实浮点波形，不添加响度归一化。
                os.replace(temporary, wav_path)  # 完成后提交正式声音。
                with torch.inference_mode():  # CycleMAE在Base和Cycle上使用相同路径。
                    errors = __import__("fastvoicegrad_formal").cycle_per_sample(runtime, components, mel, bnf, source_spk, target_spk, lengths, noise1, noise2, gradient=False)  # 使用生成语音重新提BNF的正式往返路径。
                record = {"pair_id": pair_id, "cycle_mae": float(errors[0]), "mel_seconds": mel_seconds, "mel_vocoder_seconds": synthesis_seconds, "audio_seconds": len(values) / 22050, "rtf_scope": "cached_source_features_to_waveform", "wav_sha256": file_sha256(wav_path)}  # 缓存源特征的RTF范围明确保存，不冒称含特征提取。
                if not all(math.isfinite(float(record[key])) for key in ("cycle_mae", "mel_seconds", "mel_vocoder_seconds", "audio_seconds")):  # 无效模型输出立即停下。
                    raise RuntimeError("转换结果包含非有限数。")  # 不把失败声学结果计入成功。
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")  # 每条完成记录独立可恢复。
                handle.flush()  # 立即持久提交，不等待全部8100对结束。
                records[pair_id] = record  # 更新本轮进度。
        validate_metric_counts(rows, list(records.values()), "转换")  # 只有完整转换才开始正式指标汇总。
        summary = score_all(runtime, rows, output, root)  # 输出全部报告中的客观评分。
        summary["cycle_mae"] = sum(float(record["cycle_mae"]) for record in records.values()) / len(rows)  # 每个配对等权重平均。
        total_audio = sum(float(record["audio_seconds"]) for record in records.values())  # 用真实累计音频时长作RTF分母。
        summary["mel_rtf"] = sum(float(record["mel_seconds"]) for record in records.values()) / total_audio  # 纯mel转换实时因子。
        summary["mel_vocoder_rtf"] = sum(float(record["mel_vocoder_seconds"]) for record in records.values()) / total_audio  # 明确不包含预缓存特征提取。
        if split == "test":  # 正式端点测试才增加32对完整前端计时，避免每次开发重复浪费算力。
            summary.update(time_end_to_end(runtime, components, rows, output, root, seed))  # 与缓存特征RTF分别报告真实端到端RTF。
        else:  # 开发评估以模型选择与客观音频质量为主。
            summary.update(total_vc_rtf=None, total_vc_rtf_status="endpoint_test_only")  # 不把未计时的开发流程伪装成完整端到端速度。
        summary.update(global_step=int(global_step), split=split, seed=int(seed), status="complete")  # 记录实际模型时点与完整覆盖。
        atomic_json(output / "summary.json", summary)  # 所有指标成功后提交汇总。
        return summary  # 训练入口可用于development选参。
    finally:  # 评估失败也不改变后续训练状态。
        for module, mode in zip(modules, previous_modes):  # 逐个恢复进入评估前的模式。
            module.train(mode)  # 冻结模块不会被意外改为训练模式。
        runtime.training.restore_rng_state(rng, device)  # 恢复全部训练随机数位置。

def main():  # 独立Notebook可在训练完成后续算未完成的评估。
    parser = argparse.ArgumentParser(description=__doc__)  # 给出明确评估入口。
    parser.add_argument("--project-root", type=Path, required=True)  # 不猜测永久项目或缓存位置。
    parser.add_argument("--run-dir", type=Path, required=True)  # 指定这次实验独立输出目录。
    parser.add_argument("--checkpoint", type=Path, required=True)  # 只评估已完整提交的模型断点。
    parser.add_argument("--split", choices=("development", "test"), default="test")  # 集合用途固定并显式区分。
    parser.add_argument("--seed", type=int, default=1234)  # 固定每对噪声可重复生成。
    args = parser.parse_args()  # 解析实际启动参数。
    from fastvoicegrad_formal import load_runtime, build_components  # 延迟导入计算库，Mac静态检查不会启动模型。
    runtime = load_runtime(args.project_root)  # 由训练入口执行相同环境与GPU约束。
    device = runtime.torch.device("cuda")  # 评估在已授权ColabGPU执行。
    args.lambda_cycle, args.arm, args.bnf_checkpoint = 0.0125, "base_post", None  # 构造共用模型，不改变已保存权重。
    components = build_components(runtime, args, device)  # 使用同一个教师架构和自训声码器。
    try:  # 兼容旧PyTorch没有weights_only参数。
        payload = runtime.torch.load(args.checkpoint, map_location="cpu", weights_only=False)  # 读取用户自己的完整训练断点。
    except TypeError:  # 旧版本PyTorch使用同样的完整加载语义。
        payload = runtime.torch.load(args.checkpoint, map_location="cpu")  # 不改变checkpoint字段。
    components["student"].load_state_dict(payload["student"], strict=True)  # 严格恢复FastVoiceGrad参数。
    components.pop("teacher", None)  # 转换评估无需占用教师显存。
    components.pop("discriminator", None)  # 转换评估无需占用判别器显存。
    components.pop("optimizer", None)  # 独立评分不需要学生训练优化器状态。
    components.pop("d_optimizer", None)  # 优化器也持有D参数引用，必须一同释放才真正回收显存。
    runtime.torch.cuda.empty_cache()  # 释放已无引用的GPU缓存。
    result = evaluate_conversion_in_memory(runtime, components, args.project_root, args.run_dir, args.split, args.seed, payload["global_step"])  # 从每对转换与评分记录继续。
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 显示实际汇总，不打印虚构参考成绩。

if __name__ == "__main__":  # 导入本模块只加载函数，不自动执行实验。
    main()  # 用户运行独立入口时才执行评估。
