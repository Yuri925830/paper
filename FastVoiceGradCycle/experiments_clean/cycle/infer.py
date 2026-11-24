# 本文件负责固定配对、真实语音转换和推理计时，不计算客观音质分数。
import argparse  # 读取用户明确提供的模型断点及输出位置。
import csv  # 读写每条转换的来源清单。
import hashlib  # 核验模型参数及生成音频的内容身份。
import json  # 保存可恢复的转换进度及来源记录。
import math  # 拒绝无穷大和非数值的转换结果。
import os  # 将完整临时文件原子替换为正式文件。
import sys  # 定位原说话人编码器模块的兼容接口。
import time  # 测量真实推理耗时。
from dataclasses import asdict  # 将原配对对象转成可保存的字典。
from pathlib import Path  # 安全处理中文及绝对路径。
import config as settings  # 从同一文件夹读取本组唯一的实验设置。

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
        if len(rows) != settings.DEVELOPMENT_PAIRS:  # 开发配对数量必须等于既定协议。
            raise RuntimeError(f"开发转换必须有{settings.DEVELOPMENT_PAIRS}对，实际为{len(rows)}。")  # 不把少量诊断伪装成完整开发评估。
    else:  # 测试采用原项目严格不同说话人且不同句子的配对规则。
        from voicegrad.infer_voicegrad30_paper_eval import load_test_rows, build_pairs  # 只复用已审计的配对函数。
        test_rows = load_test_rows(root / "data/metadata/test_manifest.csv")  # 严格检查十人十句测试网格。
        rows = [asdict(pair) for pair in build_pairs(test_rows, output)]  # 按原排序生成八千一百个确定配对。
    expected_count = settings.DEVELOPMENT_PAIRS if split == "development" else settings.TEST_PAIRS  # 两个集合各有固定的配对总数。
    if split not in ("development", "test") or len(rows) != expected_count:  # 集合名称和配对数量都不能被临时缩减。
        raise RuntimeError(f"配对协议不符：{split}，实际{len(rows)}对。")  # 不把少量调试结果作为完整评估。
    for index, row in enumerate(rows):  # 为当前评估目录填入稳定输出路径。
        row["pair_id"] = int(row["pair_id"])  # 保留原配对编号。
        if row["pair_id"] != index:  # 编号必须连续且顺序不可漂移。
            raise RuntimeError("评估配对编号不连续。")  # 避免续算结果套到错误语音上。
        row["source_wav"] = relocate_wave(root, row["source_wav"])  # 映射原始源语音路径。
        row["reference_wav"] = relocate_wave(root, row["reference_wav"])  # 映射真实目标参考语音路径。
        row["output_wav"] = str(output / "wav" / f"pair_{index:04d}.wav")  # 将新生成声音保存到独立实验目录。
    return rows  # 返回固定且来源明确的配对表。

def write_pairs(path, rows):  # 保存后续指标直接读取的最小标准配对表。
    fields = (  # 固定清单列名，供推理恢复和后续评分共同读取。
        "pair_id", "source_speaker", "source_utt", "source_text", "source_wav",  # 保存源语音的编号、说话人、文字及文件路径。
        "reference_speaker", "reference_utt", "reference_wav", "output_wav",  # 保存目标参考语音与生成语音的位置。
    )  # 列顺序与原正式推理清单保持相同。
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

def time_end_to_end(runtime, components, rows, output, root, seed):  # 在固定32对测试语音上真实计时完整原始波形到转换波形的流程。
    import soundfile as sf  # 在计时范围内读取真实源波形。
    from tqdm.auto import tqdm  # 显示端到端计时样本的进度。
    from train import load_python  # 复用明确路径的原模块加载器。
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
    selected = rows[:min(settings.TIMING_PAIRS, len(rows))]  # 固定取清单前32对，不挑选较快或较短的语音。
    path = Path(output) / "end_to_end_timing.jsonl"  # 每个实际完成的计时条目可在断线后复用。
    records = recover_records(path)  # 恢复已提交的固定计时编号。
    def measure(row):  # 一次完整计时包括两份波形读取、全部内容身份前端和声学合成。
        torch.cuda.synchronize(device)  # 计时前等待之前的GPU任务。
        started = time.monotonic()  # 从实际源波形读取前开始计时。
        with torch.inference_mode():  # 稳态推理完全不更新任何参数。
            waveform, sr = sf.read(row["source_wav"], dtype="float32")  # 源wav读取也计入端到端成本。
            if sr != settings.SAMPLE_RATE or waveform.ndim != 1:  # 沿用原数据22.05k单声道协议。
                raise RuntimeError("RTF源波形格式与原实验不一致。")  # 不偷偷更改测试语音。
            source = torch.from_numpy(  # 将源音频从数组转成PyTorch张量。
                waveform[:len(waveform) // settings.HOP_SIZE * settings.HOP_SIZE]  # 只保留hop长度整数倍的波形，沿用原前端规则。
            ).clamp(-1, 1).to(device)  # 沿用原幅度裁切，再送入当前GPU。
            natural_mel = mel_helper.mel_spectrogram_torch(  # 用原前端实时计算源语音的log-mel，计时不读取缓存特征。
                source, settings.N_FFT, settings.N_MELS, settings.SAMPLE_RATE,  # 传入波形、FFT点数、mel通道数和采样率。
                settings.HOP_SIZE, settings.WIN_SIZE, settings.F_MIN, settings.F_MAX,  # 传入帧移、窗长及频率上下界。
            )  # 保持原前端的参数顺序和实际计算。
            mel = (natural_mel - components["mean"]) / components["std"]  # 采用原模型训练统计进行标准化。
            speech = runtime.torchaudio.functional.resample(  # 将完整源语音变为内容编码器要求的采样率。
                torch.from_numpy(waveform).to(device),  # 将完整波形转成张量并放入当前GPU。
                settings.SAMPLE_RATE, settings.CONTENT_SAMPLE_RATE,  # 明确原语音和内容提取器使用的两种采样率。
            ).unsqueeze(0)  # 增加单条语音的batch维度。
            lengths = torch.tensor([speech.shape[-1]], device=device, dtype=torch.long)  # 指定原始波形有效长度。
            raw = components["bridge"].model(speech, lengths)[0].transpose(0, 1)  # 真实提取源语音144维Conformer内容。
            bnf = runtime.align_bnf_to_mel(raw, mel.shape[-1], mode=settings.BNF_ALIGNMENT).unsqueeze(0)  # 保留原内容与mel对齐方式。
            target = torch.from_numpy(  # 将目标参考语音的真实身份向量转为PyTorch张量。
                encoder.embed_utterance(preprocess_speaker(row["reference_wav"]))  # 读取指定参考音频并用原GE2E模型提取256维身份。
            ).to(device).unsqueeze(0)  # 把身份向量放到GPU并增加batch维度。
            generator = torch.Generator(device=device).manual_seed(  # 使用独立生成器，不消耗训练的随机序列。
                settings.INFERENCE_FORWARD_SEED + int(seed) + int(row["pair_id"])  # 按实验种子和配对编号生成与普通推理相同的噪声种子。
            )  # 固定噪声使不同模型使用同一配对条件。
            noise = torch.randn(mel.shape, device=device, generator=generator)  # 端到端成本包含噪声生成。
            predicted, _ = runtime.training.make_student_prediction(  # 调用原FastVoiceGrad的单步转换公式。
                components["student"], components["diffusion"], mel, target, bnf,  # 传入学生、扩散日程、源mel、目标身份及内容条件。
                settings.STUDENT_START_STEP, noise,  # 保留原950起始时刻和本配对固定噪声。
            )  # 返回转换后的mel及原接口的辅助输出。
            converted = components["vocoder"](  # 用冻结的原HiFi-GAN合成转换后的真实波形。
                runtime.training.denormalize_mel(predicted, components["mean"], components["std"])  # 用训练集统计恢复mel的原始尺度。
            ).reshape(-1)  # 将单条音频整理为一维采样点序列。
            converted.cpu()  # 把完整波形返回主机也计入端到端成本，文件保存不计入。
        torch.cuda.synchronize(device)  # 完整声学流程完成后才停止计时。
        return {  # 保存这对语音实际完成的端到端测量。
            "pair_id": int(row["pair_id"]),  # 保留固定清单中的配对编号。
            "total_vc_seconds": time.monotonic() - started,  # 记录从读取源音频到GPU任务完成的真实耗时。
            "audio_seconds": converted.numel() / settings.SAMPLE_RATE,  # 用输出采样点数换算音频秒数。
        }  # 不在推理计时中构造音质预测分数。
    for row in selected[:min(settings.TIMING_WARMUPS, len(selected))]:  # 用固定前三对预热GPU及前端，不将预热结果计入正式计时。
        measure(row)  # 每次恢复都执行同样预热，避免首条冷启动污染比较。
    with path.open("a", encoding="utf-8") as handle:  # 每个计时样本成功后立即持久化。
        for row in tqdm(selected, desc="端到端RTF", unit="配对"):  # 进度条显示固定32对完整推理的完成比例。
            if int(row["pair_id"]) in records:  # 已完成计时不重复占用算力。
                continue  # 直接复用同模型同配对的真实记录。
            record = measure(row)  # 对当前固定配对进行完整端到端测量。
            if any(  # 真实耗时和音频时长都必须是有限正数。
                not math.isfinite(float(record[key])) or float(record[key]) <= 0  # 检查当前字段是否无效或不大于零。
                for key in ("total_vc_seconds", "audio_seconds")  # 分别检查测量耗时与输出音频时长。
            ):  # 任一字段无效就停止发布该条计时。
                raise RuntimeError("端到端RTF出现无效计时。")  # 不发布无法计算的速度指标。
            handle.write(json.dumps(record, allow_nan=False) + "\n")  # 保存当前真实计时。
            handle.flush()  # 每个配对完成后立即提交。
            records[record["pair_id"]] = record  # 更新本次完成集合。
    validate_metric_counts(selected, list(records.values()), "端到端RTF")  # 所有固定计时样本必须完成。
    return {  # 保存与旧正式结果完全相同的端到端计时字段。
        "total_vc_rtf": (  # 端到端实时因子使用累计耗时与累计音频时长的比值。
            sum(record["total_vc_seconds"] for record in records.values())  # 累计所有固定计时配对的真实运行秒数。
            / sum(record["audio_seconds"] for record in records.values())  # 除以这些配对累计生成的音频秒数。
        ),  # 保持原正式评估的计算顺序。
        "total_vc_timing_count": len(selected),  # 记录固定计时清单实际包含的配对数。
        "total_vc_rtf_scope": "source_wav_io_mel_bnf_reference_wav_io_ge2e_student_vocoder_output_to_cpu",  # 说明波形读取、全部前端和输出回传都在范围内。
        "total_vc_rtf_status": "complete_fixed_first_32_pairs_3_warmups",  # 保留原32对和3次预热的计时协议名称。
    }  # 返回真实测量结果，不运行音质评分器。

def generate_conversion(runtime, components, root, run_dir, split, seed, global_step):  # 只生成完整配对的语音及转换诊断，返回清单和输出目录。
    import soundfile as sf  # 保存和检查真实生成音频。
    from train import cycle_per_sample  # 复用本文件夹训练文件中唯一的循环重建定义。
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
    source_identity = {  # 绑定模型参数、训练步数和固定配对，防止误用其他音频。
        "step": int(global_step),  # 记录学生已经完成的真实优化步数。
        "seed": int(seed),  # 记录本次配对噪声使用的固定种子。
        "split": split,  # 记录开发集或正式测试集的用途。
        "student_weights_sha256": weight_digest.hexdigest(),  # 绑定学生权重的真实内容哈希。
        "pairs": [{key: row[key] for key in ("pair_id", "source_utt", "reference_utt")} for row in rows],  # 保留每对语音编号及源句、参考句身份。
    }  # 身份字段与旧正式推理结果保持兼容。
    checkpoint_identity = Path(run_dir) / "run_contract.json"  # 加入训练合同以区分重建与微调实验。
    if checkpoint_identity.is_file():  # 新训练入口会先保存正式合同。
        source_identity["run_contract_sha256"] = file_sha256(checkpoint_identity)  # 防止同一目录误接不同实验。
    identity_path = output / "identity.json"  # 保存所有评分共享的模型与配对身份。
    if identity_path.is_file() and json.loads(identity_path.read_text(encoding="utf-8")) != source_identity:  # 历史结果必须严格属于当前评估请求。
        raise RuntimeError("评估目录已有其他实验结果，不能混用。")  # 保留旧实验而不是删除。
    atomic_json(identity_path, source_identity)  # 固定评估身份后再写声音。
    write_pairs(output / "pairs.csv", rows)  # 让后续指标读取同一份配对。
    dataset_split = "train" if split == "development" else "test"  # 开发语音源于排除的train索引。
    dataset = runtime.baseline.VoiceGradDataset(  # 沿用baseline的数据集实现，转换时读取完整句子。
        root / "data", split=dataset_split, segment_length=None,  # 指定数据根、集合以及不裁剪的完整句长。
        bnf_alignment=components["config"].bnf_alignment,  # 保留训练时相同的内容特征对齐方法。
    )  # 推理数据不使用训练的随机128帧片段。
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
                source_spk = source["spk_emb"].unsqueeze(0).to(device)  # 返回转换使用原说话人身份。
                target_spk = reference["spk_emb"].unsqueeze(0).to(device)  # 第一次转换使用目标说话人身份。
                lengths = source["mel_len"].reshape(1).to(device)  # 全句有效帧，没有训练补零干扰。
                generator1 = torch.Generator(device=device).manual_seed(settings.INFERENCE_FORWARD_SEED + int(seed) + pair_id)  # 第一次噪声与其他配对独立。
                generator2 = torch.Generator(device=device).manual_seed(settings.INFERENCE_BACKWARD_SEED + int(seed) + pair_id)  # 第二次噪声使用不同随机流。
                noise1 = torch.randn(mel.shape, device=device, generator=generator1)  # 固定每对首次转换噪声。
                noise2 = torch.randn(mel.shape, device=device, generator=generator2)  # 固定每对返回转换噪声。
                torch.cuda.synchronize(device)  # 测量前排空GPU异步任务。
                started = time.monotonic()  # 记录mel转换真实开始时间。
                with torch.inference_mode():  # 全部转换评分都不建立参数梯度。
                    generated, _ = runtime.training.make_student_prediction(  # 每个配对严格执行一次学生声学转换。
                        student, components["diffusion"], mel, target_spk, bnf,  # 传入当前学生、扩散日程、源mel及目标条件。
                        start_step=settings.STUDENT_START_STEP, source_noise=noise1,  # 保留950起始时刻和固定第一程噪声。
                    )  # 单步采样规则与原正式推理保持一致。
                torch.cuda.synchronize(device)  # 等待该次转换真正完成。
                mel_seconds = time.monotonic() - started  # 记录纯mel网络推理时间。
                with torch.inference_mode():  # 声码器也保持冻结只读。
                    waveform = vocoder(  # 用冻结声码器将转换mel合成为真实音频。
                        runtime.training.denormalize_mel(generated, components["mean"], components["std"])  # 先按原训练统计恢复mel尺度。
                    ).reshape(-1)  # 将单条输出整理为一维波形。
                torch.cuda.synchronize(device)  # 等待波形合成结束。
                synthesis_seconds = time.monotonic() - started  # mel转换加vocoder的总推理时间。
                values = waveform[:int(lengths[0]) * settings.HOP_SIZE].float().cpu().numpy()  # 只保存真实句长对应的波形。
                temporary = wav_path.with_name(wav_path.name + ".writing")  # 半份WAV不进入正式评分。
                sf.write(temporary, values, settings.SAMPLE_RATE, format="WAV", subtype="FLOAT")  # 保存真实浮点波形，不添加响度归一化。
                os.replace(temporary, wav_path)  # 完成后提交正式声音。
                with torch.inference_mode():  # CycleMAE在Base和Cycle上使用相同路径。
                    errors = cycle_per_sample(  # Base和Cycle都用同一个生成语音重提BNF的往返诊断。
                        runtime, components, mel, bnf, source_spk, target_spk,  # 传入模型组件、源内容及往返说话人身份。
                        lengths, noise1, noise2, gradient=False,  # 保留真实有效帧和独立两程噪声，诊断不计算梯度。
                    )  # 每个样本返回一个有效帧平均绝对误差。
                record = {  # 保存这对语音实际完成的转换诊断及音频哈希。
                    "pair_id": pair_id,  # 将记录绑定清单中的唯一配对编号。
                    "cycle_mae": float(errors[0]),  # 保存往返转换的有效帧平均绝对误差。
                    "mel_seconds": mel_seconds,  # 保存学生mel转换的真实耗时。
                    "mel_vocoder_seconds": synthesis_seconds,  # 保存学生和声码器合计的真实耗时。
                    "audio_seconds": len(values) / settings.SAMPLE_RATE,  # 由输出样本数计算真实音频时长。
                    "rtf_scope": "cached_source_features_to_waveform",  # 明确这些耗时不包含预先缓存的特征提取。
                    "wav_sha256": file_sha256(wav_path),  # 记录已经写入磁盘的音频内容身份。
                }  # 记录字段与旧正式推理结果保持一致。
                if not all(  # 保存转换记录前必须检查全部数值是否有限。
                    math.isfinite(float(record[key]))  # 检查当前误差、耗时或时长字段没有无穷大和非数值。
                    for key in ("cycle_mae", "mel_seconds", "mel_vocoder_seconds", "audio_seconds")  # 覆盖本条转换的四个数值字段。
                ):  # 任一无效输出都不能被计入成功转换。
                    raise RuntimeError("转换结果包含非有限数。")  # 不把失败声学结果计入成功。
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")  # 每条完成记录独立可恢复。
                handle.flush()  # 立即持久提交，不等待全部8100对结束。
                records[pair_id] = record  # 更新本轮进度。
        validate_metric_counts(rows, list(records.values()), "转换")  # 只有完整转换才开始正式指标汇总。
        return rows, output  # 转换完整后交给独立评估文件计算客观分数。
    finally:  # 评估失败也不改变后续训练状态。
        for module, mode in zip(modules, previous_modes):  # 逐个恢复进入评估前的模式。
            module.train(mode)  # 冻结模块不会被意外改为训练模式。
        runtime.training.restore_rng_state(rng, device)  # 恢复全部训练随机数位置。

def main(argv=None):  # 用户显式运行本文件时，才加载模型并生成真实转换音频。
    parser = argparse.ArgumentParser(description="按固定配对运行本组推理；不计算UTMOS等分数。")  # 说明推理与评估的职责边界。
    parser.add_argument("--project-root", type=Path, default=settings.PROJECT_ROOT)  # 默认读取设置文件中已有的baseline项目。
    parser.add_argument("--run-dir", type=Path, default=settings.RUN_DIR)  # 默认写入本次重构专用的新实验目录。
    parser.add_argument("--checkpoint", type=Path, required=True)  # 必须明确选择已有的完整模型断点。
    parser.add_argument("--split", choices=("development", "test"), default="test")  # 只允许固定开发集或正式测试集。
    parser.add_argument("--seed", type=int, default=settings.SEED)  # 保留与训练记录相同的配对噪声种子。
    parser.add_argument("--run-approved", action="store_true")  # 防止查看代码或准备文件时误启动GPU推理。
    args = parser.parse_args(argv)  # 读取本次明确的运行参数。
    if not args.run_approved:  # 仅导入文件或查看帮助都不表示授权运行模型。
        parser.error("实际推理必须显式添加 --run-approved。")  # 在加载机器学习库前停止误启动。
    from train import load_runtime, build_components, load_checkpoint  # 复用本组训练文件的模型构建及运行环境。
    runtime = load_runtime(args.project_root)  # 只在登记的Linux CUDA主机加载baseline依赖。
    args.arm, args.lambda_cycle, args.bnf_checkpoint = settings.ARM, settings.LAMBDA_CYCLE, None  # 结构与本组设置一致，推理不执行优化。
    device = runtime.torch.device("cuda")  # 使用当前进程明确可见的GPU。
    components = build_components(runtime, args, device)  # 按原顺序构建同一学生、声码器及内容提取器。
    payload = load_checkpoint(args.checkpoint)  # 完整读取用户指定的自有断点。
    is_baseline = (  # 只有原始内容哈希和完整状态身份都匹配，才接受未标记本组的baseline。
        file_sha256(args.checkpoint) == settings.LEGACY_PARENT_SHA256  # 验证文件正是已经归档的学生12号父断点。
        and payload.get("kind") == settings.PARENT_KIND  # baseline必须采用原完整训练状态格式。
        and payload.get("global_step") == settings.START_STEP  # baseline必须停在122900步，而不是任意旧模型。
        and not payload.get("cycle_enabled", False)  # 原baseline不能已经包含Cycle更新。
    )  # 普通无标签模型不会被误认为已验证baseline。
    if (payload.get("arm") != settings.ARM and not is_baseline) or "student" not in payload:  # 本组子断点或固定baseline才允许进入本组推理。
        raise RuntimeError("断点不是本组实验或baseline的学生状态。")  # 防止Base与Cycle权重被错误标记。
    components["student"].load_state_dict(payload["student"], strict=True)  # 严格恢复所有学生参数及缓冲区。
    for key in ("teacher", "discriminator", "optimizer", "d_optimizer"):  # 推理不需要教师、判别器或训练优化器。
        components.pop(key, None)  # 释放多余引用以降低显存占用。
    runtime.torch.cuda.empty_cache()  # 回收已没有参数引用的GPU缓存。
    components["student"].eval()  # 独立推理和计时都必须关闭学生的随机训练层。
    components["vocoder"].eval()  # 冻结声码器始终使用评估模式合成音频。
    rows, output = generate_conversion(  # 按明确选择的完整断点继续生成固定配对音频。
        runtime, components, args.project_root, args.run_dir,  # 传入实际运行时、模型、baseline项目和本次独立输出。
        args.split, args.seed, payload["global_step"],  # 保留集合用途、固定噪声种子和真实模型步数。
    )  # 返回实际完成的清单及转换目录。
    if args.split == "test":  # 正式端点测试保留原先固定32对、3次预热的完整流程计时。
        time_end_to_end(runtime, components, rows, output, args.project_root, args.seed)  # 计时结果留给evaluate.py汇总，不运行客观评分器。
    print(f"已完成{len(rows)}对转换，输出目录：{output}")  # 只报告真实生成数量及保存位置。

if __name__ == "__main__":  # 导入本文件只定义函数，不自动启动推理。
    main()  # 用户明确执行本文件时读取命令行并运行。
