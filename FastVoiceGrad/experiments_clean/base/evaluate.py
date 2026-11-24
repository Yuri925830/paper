# 本文件只评分已有转换音频，并汇总完整配对的客观指标。
import argparse  # 读取用户明确选择的已有转换目录。
import json  # 读取来源身份并显示真实评分汇总。
import math  # 检查评分及计时数值是否有效。
from pathlib import Path  # 安全处理已有项目和评估路径。
import config as settings  # 从同组设置文件读取评分协议。
from infer import atomic_json, file_sha256, read_csv, recover_records, validate_metric_counts  # 复用同目录的文件读写及完整性核验。

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
            if sr != settings.SAMPLE_RATE or waveform.ndim != 1:  # 评分输入必须为单声道22.05kHz。
                raise RuntimeError("DNSMOS收到错误的波形采样率或声道数。")  # 拒绝偷偷归一化或改变音频。
            tensor = runtime.torch.from_numpy(waveform).unsqueeze(0)  # 增加单条batch维度。
            values = deep_noise_suppression_mean_opinion_score(  # 使用原非个性化DNSMOS官方封装。
                tensor, fs=sr, personalized=False, device="cpu",  # 保留真实音频采样率与原CPU评分方式。
                num_threads=settings.DNSMOS_THREADS,  # 线程数统一读取本组实验设置。
            )  # 接口返回P.808及其他输出，下一行只取既定主报告列。
            p808 = float(values.reshape(-1, 4)[0, 0])  # 按官方接口第一列提取P.808。
            if not math.isfinite(p808):  # 异常模型输出不能被当作评分。
                raise RuntimeError("DNSMOS P.808产生非有限分数。")  # 保留此前成功进度并停止。
            record = {"pair_id": int(row["pair_id"]), "p808_mos": p808}  # 不保存其他DNSMOS输出。
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")  # 持久追加真实分数。
            handle.flush()  # 每条分数完成后立即交给永久结果目录。
            records[record["pair_id"]] = record  # 更新本次进程的完成集合。
    validate_metric_counts(rows, list(records.values()), "DNSMOS P.808")  # 全部完成后才返回。
    return list(records.values())  # 返回实际评分，交给统一汇总。

def score_all(runtime, rows, output, root):  # 使用原项目固定评估器计算所有报告指标。
    import evaluation.evaluate_metrics as metrics  # 复用已审计UTMOS、CER、ECAPA逐条评分函数。
    output, root = Path(output), Path(root)  # 统一目录类型。
    device = runtime.torch.device("cuda")  # 与正式转换使用同一GPU环境。
    metrics.evaluate_utmos(rows, output / "utmos.csv", root / "UTMOS-demo", device, settings.METRIC_BATCH_SIZE)  # 保持官方strong learner及原评分变换。
    p808 = score_p808(runtime, rows, output)  # 仅计算并报告用户选定的P.808字段。
    metrics.evaluate_cer(rows, output / "cer.csv", device, settings.METRIC_BATCH_SIZE)  # 使用原LV60K-960h ASR和文本标准化。
    metrics.evaluate_sva(rows, output / "sva.csv", root / "pretrained_spkrec_ecapa", device, settings.METRIC_BATCH_SIZE)  # 保持ECAPA余弦阈值严格大于0.25。
    utmos, cer, sva = [read_csv(output / name) for name in ("utmos.csv", "cer.csv", "sva.csv")]  # 读取所有真实逐条评分。
    for label, values in (("UTMOS", utmos), ("CER", cer), ("SVA", sva)):  # 同时检查所有指标覆盖。
        validate_metric_counts(rows, values, label)  # 缺项不得写入完成汇总。
    return summarize_scores(rows, utmos, p808, cer, sva)  # 逐条评分与纯数值汇总分开，方便核对已有结果。

def summarize_scores(rows, utmos, p808, cer, sva):  # 仅用已完成的逐条分数计算汇总，不加载或执行模型。
    for label, values in (("UTMOS", utmos), ("DNSMOS P.808", p808), ("CER", cer), ("SVA", sva)):  # 四种分数必须覆盖同一配对集合。
        validate_metric_counts(rows, values, label)  # 拒绝漏项、重复或额外评分。
    n = len(rows)  # 分母采用真实配对数量。
    if n == 0:  # 空清单没有可报告的完整评估结果。
        raise RuntimeError("不能汇总空配对清单。")  # 防止除零或误报零分。
    characters = sum(int(row["reference_chars"]) for row in cer)  # 语料CER按总参考字符数加权。
    if characters <= 0:  # 无参考文本时无法定义CER。
        raise RuntimeError("CER参考文本总字符数为零。")  # 不构造虚假的零错误率。
    return {  # 保持原正式指标的字段、分母和计算顺序。
        "count": n,  # 记录真正完成四种评分的配对总数。
        "utmos": sum(float(row["utmos"]) for row in utmos) / n,  # 对每对语音的UTMOS预测分数等权平均。
        "dnsmos_p808": sum(float(row["p808_mos"]) for row in p808) / n,  # 只汇总DNSMOS的P.808列。
        "corpus_cer_percent": 100 * sum(int(row["char_errors"]) for row in cer) / characters,  # 总字符编辑错误数除以总参考字符数。
        "utterance_cer_percent": sum(float(row["cer_percent"]) for row in cer) / n,  # 保留逐句CER均值作辅助读数，不替代语料级CER。
        "sva_percent": 100 * sum(float(row["cosine_score"]) > settings.SVA_THRESHOLD for row in sva) / n,  # 严格大于0.25的配对才通过验证。
        "ecapa_cosine": sum(float(row["cosine_score"]) for row in sva) / n,  # 汇总与同一参考语音比较的平均余弦分数。
        "sva_threshold": settings.SVA_THRESHOLD,  # 明确保存固定的验证阈值。
    }  # 返回可直接写入正式汇总的真实数值。

def summarize_conversion(rows, records):  # 从真实转换记录汇总循环误差与缓存特征推理速度。
    validate_metric_counts(rows, list(records.values()), "转换")  # 先确认所有配对都确实完成转换。
    total_audio = sum(float(record["audio_seconds"]) for record in records.values())  # 累计真实转换音频时长作为速度分母。
    if total_audio <= 0 or not math.isfinite(total_audio):  # 时长必须为有限正数。
        raise RuntimeError("转换记录的累计音频时长无效。")  # 不发布无法定义的实时因子。
    return {  # 保持原正式汇总使用的字段名称和计算顺序。
        "cycle_mae": sum(float(record["cycle_mae"]) for record in records.values()) / len(rows),  # 每个配对的有效帧MAE等权平均。
        "mel_rtf": sum(float(record["mel_seconds"]) for record in records.values()) / total_audio,  # 只统计学生mel网络的真实耗时。
        "mel_vocoder_rtf": sum(float(record["mel_vocoder_seconds"]) for record in records.values()) / total_audio,  # 统计学生加声码器的耗时，仍不含缓存特征提取。
    }  # 返回实际计算的三个转换诊断值。

def saved_timing(rows, output, split):  # 只汇总已存在的端到端计时，不重新运行推理。
    if split != "test":  # 开发集按原协议不计算完整端到端速度。
        return {"total_vc_rtf": None, "total_vc_rtf_status": "endpoint_test_only"}  # 明确标注原协议未计时。
    records = recover_records(Path(output) / "end_to_end_timing.jsonl")  # 读取推理文件已提交的固定32对计时。
    selected = rows[:min(settings.TIMING_PAIRS, len(rows))]  # 保留原清单的固定前32对，不按速度挑选。
    if not records:  # 用户只生成音频而没有做完整前端计时的情况必须显式保留。
        return {"total_vc_rtf": None, "total_vc_rtf_status": "not_measured"}  # 不用缓存特征RTF冒充端到端RTF。
    validate_metric_counts(selected, list(records.values()), "端到端RTF")  # 有部分记录时要求补齐后再发布完整汇总。
    for record in records.values():  # 每条计时都必须满足原有限正数要求。
        if any(not math.isfinite(float(record[key])) or float(record[key]) <= 0 for key in ("total_vc_seconds", "audio_seconds")):  # 检查耗时和音频时长。
            raise RuntimeError("端到端RTF含无效计时。")  # 不接受零、负数或非数值时间。
    return {  # 与原正式端到端汇总保持同样的字段和分母。
        "total_vc_rtf": (  # 完整端到端实时因子采用累计耗时与音频时长的比值。
            sum(record["total_vc_seconds"] for record in records.values())  # 累计所有固定配对实际测得的运行秒数。
            / sum(record["audio_seconds"] for record in records.values())  # 除以这些配对生成的音频总秒数。
        ),  # 保持与原正式汇总相同的计算顺序。
        "total_vc_timing_count": len(selected),  # 记录实际使用的固定计时数量。
        "total_vc_rtf_scope": "source_wav_io_mel_bnf_reference_wav_io_ge2e_student_vocoder_output_to_cpu",  # 明确包含两份波形读取、特征前端和输出回传。
        "total_vc_rtf_status": "complete_fixed_first_32_pairs_3_warmups",  # 保留既定32对及3次预热协议标签。
    }  # 返回真实存在的端到端计时汇总。

def evaluate_conversion_in_memory(runtime, components, root, run_dir, split, seed, global_step):  # 训练到预定时点时，先调用推理文件再调用评分逻辑。
    from infer import generate_conversion, time_end_to_end  # 推理过程只有infer.py中的一份定义。
    modules = [components["student"], components["vocoder"]]  # 评分前保存学生和声码器的训练状态。
    modes = [module.training for module in modules]  # 记录原训练或评估模式。
    device = next(components["student"].parameters()).device  # 使用学生所在设备恢复随机状态。
    rng = runtime.training.capture_rng_state()  # 评分和计时不能消耗后续训练的随机序列。
    for module in modules:  # 禁用评估期间可能存在的随机训练层。
        module.eval()  # 本次转换、评分和计时不更新模型。
    try:  # 任一步失败也必须恢复原训练状态。
        rows, output = generate_conversion(runtime, components, root, run_dir, split, seed, global_step)  # 完成同模型同配对的真实转换。
        summary = score_all(runtime, rows, output, root)  # 计算四种客观评分及ECAPA平均余弦。
        summary.update(summarize_conversion(rows, recover_records(output / "conversion.jsonl")))  # 汇总真实循环误差和缓存特征速度。
        if split == "test":  # 只有正式端点测试执行完整前端计时。
            summary.update(time_end_to_end(runtime, components, rows, output, root, seed))  # 与原流程相同，在评分之后执行端到端测量。
        else:  # 开发评估不额外执行32对完整计时。
            summary.update(saved_timing(rows, output, split))  # 明确保留未计时状态。
        summary.update(global_step=int(global_step), split=split, seed=int(seed), status="complete")  # 发布真实模型步数及集合用途。
        atomic_json(output / "summary.json", summary)  # 所有规定结果都完成后原子写入正式汇总。
        return summary  # 将客观结果交回训练器记录开发和端点评估。
    finally:  # 包括转换、评分或计时异常的路径都恢复原状态。
        for module, mode in zip(modules, modes):  # 逐个恢复进入评估前的训练开关。
            module.train(mode)  # 声码器等冻结模块不会被意外切换训练模式。
        runtime.training.restore_rng_state(rng, device)  # 恢复Python、NumPy、CPU和CUDA随机位置。

def validate_existing_conversion(output):  # 在加载评分器之前，核对用户明确指定的已有转换目录。
    output = Path(output)  # 统一为路径对象。
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))  # 模型步数、权重身份和配对来源必须已经存在。
    rows = read_csv(output / "pairs.csv")  # 只读取原推理保存的配对，评分时不重新选样本。
    split = identity.get("split")  # 集合用途由推理来源记录确定。
    count = settings.DEVELOPMENT_PAIRS if split == "development" else settings.TEST_PAIRS  # 保留固定90或8100对协议。
    if split not in ("development", "test") or len(rows) != count:  # 集合名称与完整配对数量必须同时有效。
        raise RuntimeError("已有转换目录的集合或配对数量不符合协议。")  # 不能把部分转换伪称完整测试。
    expected_pairs = [  # 从原清单逐条恢复推理身份中保存的配对字段。
        {  # 每个配对只保留编号、源句和参考句身份。
            key: int(row[key]) if key == "pair_id" else row[key]  # CSV中的配对编号转回整数，句号标识保持文本。
            for key in ("pair_id", "source_utt", "reference_utt")  # 字段集合与推理保存的身份记录一致。
        }  # 当前配对的身份字典已经完整。
        for row in rows  # 按原清单顺序遍历全部配对。
    ]  # 顺序和字段类型都参与下一步身份核验。
    if identity.get("pairs") != expected_pairs or [int(row["pair_id"]) for row in rows] != list(range(count)):  # 身份、顺序及连续编号都必须匹配。
        raise RuntimeError("已有配对清单与推理身份记录不一致。")  # 防止把分数附到其他配对。
    contract = output.parents[2] / "run_contract.json"  # 原目录结构为实验目录下evaluation、集合和步数。
    if "run_contract_sha256" in identity and (  # 推理曾记录训练合同时，评分必须继续核对同一份合同。
        not contract.is_file() or file_sha256(contract) != identity["run_contract_sha256"]  # 合同缺失或内容变化都属于来源不匹配。
    ):  # 来源不符时不能复用此前的生成音频和分数。
        raise RuntimeError("已有转换的训练合同身份不一致。")  # 保存历史结果而不是覆盖其身份。
    summary_path = output / "summary.json"  # 已经完成的汇总可以被验证，但不能混入另一个模型时点。
    if summary_path.is_file():  # 评分续算需要先检查此前已发布的步数和集合。
        old = json.loads(summary_path.read_text(encoding="utf-8"))  # 读取已有汇总身份。
        if any(  # 已有汇总必须属于当前推理的同一步数、集合和种子。
            old.get(key) != identity.get(identity_key)  # 对比汇总字段与推理身份中的对应字段。
            for key, identity_key in (("global_step", "step"), ("split", "split"), ("seed", "seed"))  # 明确三对字段名映射。
        ):  # 任一字段不符都拒绝覆盖已有成绩。
            raise RuntimeError("已有汇总属于其他模型步数、集合或种子。")  # 不覆盖不匹配的旧成绩。
    records = recover_records(output / "conversion.jsonl")  # 恢复每条已完成的转换及音频内容哈希。
    validate_metric_counts(rows, list(records.values()), "转换")  # 缺少转换记录时不运行评分器。
    for row in rows:  # 逐条检查真实生成音频，避免原音频被改写后复用旧分数。
        path = Path(row["output_wav"])  # 保留推理保存的真实输出位置。
        if not path.is_file() or file_sha256(path) != records[int(row["pair_id"])].get("wav_sha256"):  # 文件必须存在且与已提交内容相同。
            raise RuntimeError(f"生成音频缺失或哈希不符：{path}")  # 不偷偷重生成音频或改变评分对象。
    return rows, identity, records  # 交给独立评分入口，不返回或加载学生模型。

def main(argv=None):  # 用户明确运行本文件时，只评分已有转换音频。
    parser = argparse.ArgumentParser(description="评分已有转换目录；不会重新训练或生成音频。")  # 说明独立评估入口的范围。
    parser.add_argument("--project-root", type=Path, default=settings.PROJECT_ROOT)  # 读取设置文件中固定的评分器和模型资源目录。
    parser.add_argument("--conversion-dir", type=Path, required=True)  # 必须明确指定已有转换目录，默认不接触历史结果。
    parser.add_argument("--run-approved", action="store_true")  # 实际评分会占用GPU，必须明确请求运行。
    args = parser.parse_args(argv)  # 读取明确的用户运行参数。
    if not args.run_approved:  # 查看代码、导入函数和查看帮助都不触发评分。
        parser.error("实际评估必须显式添加 --run-approved。")  # 在导入评分模型前阻止误启动。
    rows, identity, records = validate_existing_conversion(args.conversion_dir)  # 先只读核对模型、清单和音频身份。
    from train import load_runtime  # 使用本组训练入口相同的Linux CUDA环境要求。
    runtime = load_runtime(args.project_root)  # 只加载原项目依赖，不构建学生、教师或判别器。
    summary = score_all(runtime, rows, args.conversion_dir, args.project_root)  # 仅对已存在且内容核验通过的音频计算分数。
    summary.update(summarize_conversion(rows, records))  # 转换诊断来自原推理记录，不重新执行往返网络。
    summary.update(saved_timing(rows, args.conversion_dir, identity["split"]))  # 已有计时则汇总，没有计时则明确标为未测。
    summary.update(global_step=int(identity["step"]), split=identity["split"], seed=int(identity["seed"]), status="complete")  # 保持最终汇总身份与原转换一致。
    atomic_json(args.conversion_dir / "summary.json", summary)  # 评分覆盖完整后才发布正式汇总。
    print(json.dumps(summary, ensure_ascii=False, indent=2))  # 显示真实结果及缺少的计时状态。

if __name__ == "__main__":  # 导入文件只提供函数，避免整理代码时运行评分。
    main()  # 用户明确执行本文件时才读取参数并评分。
