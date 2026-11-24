# 四系统开发集客观对照；只读既有模型，输出独立于正式8100对评估。
"""Run fixed development conversions and reuse the existing objective scorers.

Run only on Matpool. This is a development diagnostic, never a formal paper
evaluation. No optimizer, backward pass, feature extraction or training occurs.
"""

# 所有标准库导入只用于参数、身份核对和独立结果文件管理。
import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean

# 权威模型和特征始终来自这台租用服务器，不允许回落到本地副本。
ROOT = Path("/root/FsatVoiceGrad")
# 诊断只使用训练说话人中预留的开发样本，不碰正式测试输出。
DEV = ROOT / "exp/diagnostics/student_development_20260906"
# 四系统共同使用的冻结自训声码器，避免在此对照里同时改声码器。
VOCODER = ROOT / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
# 已经审计过的教师500轮文件是教师30步和学生初始化单步的共同起点。
TEACHER = ROOT / "exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt"
# 系统身份写死，禁止误把后续latest替换成这里的历史基线。
SYSTEMS = {
    # 教师30步是本开发集内的性能参照，权重本身保持不动。
    "teacher500_k30": {
        "checkpoint": str(TEACHER),
        "sha256": "1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6",
        "state_key": "model", "epoch": 500, "global_step": 615000, "steps": 30,
    },
    # S0只把同一教师权重改为单步推理，学生尚未接受任何蒸馏更新。
    "teacher_initialized_s0_k1": {
        "checkpoint": str(TEACHER),
        "sha256": "1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6",
        "state_key": "model", "epoch": 500, "global_step": 615000, "steps": 1,
    },
    # 旧100轮学生保留原字节，仅用于解释已发生的性能差距。
    "old_student_epoch100_k1": {
        "checkpoint": str(ROOT / "exp/checkpoints/fastvoicegrad_vctk_text_unseen/epoch_0100.pt"),
        "sha256": "eb9ad626160372f1b45b29d6c6d3f058fa8e553cfe48a5ad6a8c007e00d7ca27",
        "state_key": "student", "epoch": 100, "global_step": 123000, "steps": 1,
    },
    # 两轮修复学生必须读取已经冻结的停训归档，不能使用会被续训覆盖的latest。
    "repair_custom_step2460_k1": {
        "checkpoint": str(ROOT / "exp/checkpoints/fastvoicegrad_vctk_text_unseen_repair_x0_sqrt_20260906/stopped_epoch_0002_step_00002460_full.pt"),
        "sha256": "2f5c56f819b98994a25c4b891412cadac1af5bfa0753019a06e1f8a392974da2",
        "state_key": "student", "epoch": 2, "global_step": 2460, "steps": 1,
    },
}


# 分块读取大检查点，避免为了记录指纹再把整个文件读入内存。
def digest(path):
    # SHA-256绑定实际字节，文件名相同也不能替代身份验证。
    result = hashlib.sha256()
    # 二进制读取不改动文件或其模型内容。
    with Path(path).open("rb") as handle:
        # 每次最多读取1MB，适用于包含优化器状态的完整停训归档。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            # 依照原字节顺序累计摘要。
            result.update(chunk)
    # 返回便于写入JSON和逐项比对的十六进制字符串。
    return result.hexdigest()


# 所有JSON只写入独立诊断目录；临时文件发布后不会留下半份完成报告。
def write_json(path, value):
    # 临时文件与最终文件位于同一文件系统，以便原子替换。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 保留中文说明，禁止把NaN当作有效数值写进JSON。
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # 只有完整写完才对外发布；调用方事先负责检查是否允许创建。
    temporary.replace(path)


# 读取开发集CSV，不在这里重新抽样或改变配对顺序。
def read_csv(path):
    # 兼容UTF-8 BOM，保持引号中的完整源文本。
    with path.open(encoding="utf-8-sig", newline="") as handle:
        # 返回保留字段名的逐行字典。
        return list(csv.DictReader(handle))


# 核对所有输入存在、身份一致，并把本次实际读取文件摘要交给运行身份记录。
def validate_inputs(args):
    # 明确要求单独的开发契约文件，不能拿正式test_manifest冒充开发清单。
    contract = json.loads(args.development_contract.read_text(encoding="utf-8"))
    # 空契约无法说明这批样本的用途，拒绝继续。
    if not isinstance(contract, dict) or not contract:
        raise ValueError("Development contract must be a non-empty JSON object")
    # 选样程序完成后才会发布收据；不接受尚在生成中的半份开发集。
    receipt_path = args.pairs.parent / "development_manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # 文件字节摘要和规范化JSON摘要分别使用，避免把两种哈希混为同一概念。
    if receipt.get("status") != "complete" or receipt.get("pairs_csv_sha256") != digest(args.pairs) or receipt.get("development_contract_file_sha256") != digest(args.development_contract):
        raise ValueError("Development selection receipt does not match pairs/contract files")
    # 四个系统共用一次读取的配对模板。
    rows = read_csv(args.pairs)
    # 本诊断只接受已经约定的小开发集，不接受8100正式测试清单。
    if not 90 <= len(rows) <= 120:
        raise ValueError("Expected 90..120 development pairs, never the formal 8100 pairs")
    # 连续唯一编号保证四系统和评分表可以严格逐对对应。
    if [int(row["pair_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Development pair_id must be exactly 0..N-1 in order")
    # 把模板、划分契约与训练统计一起绑定，后面禁止输入静默变化。
    inputs = {str(path.resolve()): digest(path) for path in (
        args.pairs, args.development_contract, receipt_path,
        ROOT / "data/stats/mel_mean.npy", ROOT / "data/stats/mel_std.npy",
        VOCODER / "generator_step_2500000.pt", VOCODER / "config.json",
        VOCODER / "final_checkpoint_attestation_comments_20260906.json",
    )}
    # 四系统必须使用此前正式结果的同一自训G，不能仅靠“250万步”文件名认定相同。
    if inputs[str(VOCODER / "generator_step_2500000.pt")] != "dacd362c4db46cdf7bd8428ace6f38d7d03976c60a3cd4c55022d38383094fb9" or inputs[str(VOCODER / "config.json")] != "4b570fee140ffbda92bb217218792dca069b2d73627bc5e7bf7b1669869dd05a":
        raise ValueError("The common frozen vocoder differs from the audited custom baseline")
    # 排除清单及剩余训练清单也绑定本轮身份，证明开发集来自预先固定的划分。
    for name, expected in receipt["artifacts"].items():
        # 收据只能引用同目录内的单个文件，禁止路径穿越到其他实验。
        if Path(name).name != name:
            raise ValueError(f"Unsafe development receipt artifact name: {name}")
        path = args.pairs.parent / name
        if digest(path) != expected:
            raise ValueError(f"Development receipt artifact changed: {path}")
        inputs[str(path.resolve())] = expected
    # 需要从模板逐条核对的真实语音与预处理资产。
    asset_fields = ("source_wav", "reference_wav", "source_mel", "source_bnf", "reference_spk_emb")
    # 每条配对必须具有明确源内容、目标音色和可复现随机种子。
    for row in rows:
        # 同说话人转换不能充当跨人语音转换开发指标。
        if row["source_speaker"] == row["reference_speaker"]:
            raise ValueError(f"Same-speaker pair: {row['pair_id']}")
        # 空文本或空时长无法计算有效CER，也不能进入波形推理。
        if not row["source_text"].strip() or int(row["expected_num_samples"]) <= 0 or int(row["seed"]) < 0:
            raise ValueError(f"Invalid text, output length or seed: {row['pair_id']}")
        # 核对每份实际输入，重复来源只计算一次哈希。
        for key in asset_fields:
            # 路径来自已固定模板，统一解析以消除相对目录歧义。
            path = Path(row[key]).resolve(strict=True)
            # 所有资产必须属于当前项目的数据目录，禁止意外读其他实验缓存。
            if ROOT / "data" not in path.parents:
                raise ValueError(f"Input is outside authoritative data/: {path}")
            # 首次遇到文件才读取其字节，后续重复配对复用相同摘要。
            if str(path) not in inputs:
                inputs[str(path)] = digest(path)
            # 清单必须提供预先选样时的摘要，不能只在运行时接受任何同名文件。
            if row.get(key + "_sha256") != inputs[str(path)]:
                raise ValueError(f"Development asset hash mismatch: {key}, pair={row['pair_id']}")
    # 加入当前实现和驱动本身，防止用修改后的代码解释之前的输出。
    sources = [Path(__file__).resolve()]
    # 覆盖实际推理和评分相关的项目源码，保留完整实现身份。
    for package in ("voicegrad", "fastvoicegrad", "hifigan", "evaluation"):
        # 路径排序保证身份JSON在不同启动中顺序稳定。
        sources.extend(sorted((ROOT / package).rglob("*.py")))
    # 代码文件较小，逐个绑定完整字节即可。
    inputs.update({str(path): digest(path) for path in sources})
    # 启动时核验固定四系统，不使用任何未声明的latest权重。
    for spec in SYSTEMS.values():
        # 同一个教师文件供两系统共用，仅计算一次大文件哈希。
        if spec["checkpoint"] not in inputs:
            inputs[spec["checkpoint"]] = digest(spec["checkpoint"])
        # 先验证字节，再允许后面反序列化可信模型。
        if inputs[spec["checkpoint"]] != spec["sha256"]:
            raise ValueError(f"Baseline checkpoint identity mismatch: {spec['checkpoint']}")
    # 返回固定模板与所有可追溯输入，供推理和评分共享。
    return rows, inputs


# 推理只使用现有低层公式，正式8100入口及其旧权重哈希保护完全保持原样。
def infer(args, rows, inputs):
    # 计算库仅在已通过Matpool运行检查之后导入。
    import numpy as np
    import soundfile as sf
    import torch
    # 声码器仍走已有完整验真链；本诊断没有绕过它。
    from fastvoicegrad.components import load_frozen_vocoder, load_torch_checkpoint, set_requires_grad, load_mel_statistics
    # 使用固定训练配置中的教师及声码器接口，归一化和BNF均不改变。
    from fastvoicegrad.config import PAPER_STUDENT_CONFIG
    # 复用已有全局互斥锁，训练未暂停时本诊断会安全拒绝。
    from fastvoicegrad.infer_paper_eval import acquire_inference_locks
    # 使用与正式输出相同的PCM保存函数。
    from hifigan.tools.vocode import atomic_write_wav
    # 扩散推理与正式教师/学生共用一套实现。
    from voicegrad.diffusion import VoiceGradDiffusion
    # 对齐模式保留legacy_length_linear，不改教师条件。
    from voicegrad.feature_alignment import align_bnf_to_mel
    # 使用已有矩阵维度验证，避免mel/BNF轴翻转。
    from voicegrad.infer_voicegrad30_paper_eval import feature_matrix
    # 网络结构仍是既有legacy_decimate U-Net。
    from voicegrad.train import build_model
    # 基线必须保留已核验的旧接口，不能让后续配置默认值悄悄改变教师输入。
    if (PAPER_STUDENT_CONFIG.bnf_alignment, PAPER_STUDENT_CONFIG.bnf_downsampling, PAPER_STUDENT_CONFIG.diffusion_steps, PAPER_STUDENT_CONFIG.cosine_offset) != ("legacy_length_linear", "legacy_decimate", 1000, 0.008):
        raise ValueError("Fixed baseline inference interface differs from audited legacy settings")
    # 主线程显式选择设备；默认CPU不意味着可以在本地执行。
    device = torch.device(args.device)
    # 多线程上限控制在4以内，不同时启动数据加载worker。
    torch.set_num_threads(4)
    # 只有所有共享锁都取得后才开始加载任何GPU模型。
    with acquire_inference_locks(ROOT), torch.inference_mode():
        # 自训声码器只加载一次，四系统输出均通过完全同一个冻结V。
        vocoder, _, _ = load_frozen_vocoder(
            ROOT, VOCODER / "generator_step_2500000.pt", VOCODER / "config.json",
            VOCODER / "final_checkpoint_attestation_comments_20260906.json", PAPER_STUDENT_CONFIG, device,
        )
        # 读取原训练统计，不对这90对重新估计均值或标准差。
        mean_tensor, std_tensor = load_mel_statistics(ROOT / "data", device)
        # 预处理保留正式推理的NumPy float32计算顺序。
        mean_array = np.load(ROOT / "data/stats/mel_mean.npy", allow_pickle=False).astype(np.float32).reshape(80, 1)
        std_array = np.load(ROOT / "data/stats/mel_std.npy", allow_pickle=False).astype(np.float32).reshape(80, 1)
        # 各系统共享同一1000步余弦日程和950起点。
        diffusion = VoiceGradDiffusion(PAPER_STUDENT_CONFIG.diffusion_steps, offset=PAPER_STUDENT_CONFIG.cosine_offset).to(device)
        # 顺序处理系统，每次只保留一份大模型，防止四模型同时占用显存。
        for label in args.systems:
            # 系统文件名直接来自受限choices，没有用户提供的任意路径片段。
            spec, out = SYSTEMS[label], args.output_root / label
            # 已完成结果只能校验后复用；部分目录不做猜测式续跑。
            if (out / "inference_complete.json").is_file():
                # 确认旧波形仍对应相同运行身份，再跳过这个完整系统。
                verify_wavs(out, rows)
                continue
            # 每个未完成系统使用新目录，禁止覆盖半成品或已有证据。
            out.mkdir(parents=True, exist_ok=False)
            # 只加载事先核验SHA的可信归档，不加载优化器、不恢复训练状态。
            payload = load_torch_checkpoint(Path(spec["checkpoint"]))
            # 文件名与内部epoch/step共同核验，避免归档误标。
            if (payload.get("epoch"), payload.get("global_step")) != (spec["epoch"], spec["global_step"]):
                raise ValueError(f"Checkpoint cursor mismatch: {label}")
            # 不改教师和旧学生的内部BNF下采样规则。
            model = build_model(PAPER_STUDENT_CONFIG.bnf_downsampling)
            # strict=True要求权重字段名称和形状全部吻合。
            model.load_state_dict(payload[spec["state_key"]], strict=True)
            # 仅记录原训练契约摘要，不声称通过另一时期正式训练契约。
            saved_contract_sha = payload.get("run_contract_sha256")
            # 完整停训文件还含判别器和优化器，尽早释放这些不用的数据。
            del payload
            gc.collect()
            # 四系统都冻结并使用eval，整个诊断没有反向传播。
            model.to(device).eval()
            set_requires_grad(model, False)
            # 逐系统只改输出位置，其余配对字段和顺序保持模板原样。
            system_rows, wav_hashes = [], {}
            # batch固定为1，每对独立seed，四系统共享同一份源加噪。
            for index, row in enumerate(rows):
                # 只读取源句mel和BNF，目标参考仅提供音色向量。
                mel = feature_matrix(Path(row["source_mel"]), 80)
                bnf = feature_matrix(Path(row["source_bnf"]), 144)
                # NumPy归一化顺序与正式推理相同，随后添加批维。
                source = torch.from_numpy(((mel - mean_array) / std_array).astype(np.float32)).unsqueeze(0).to(device)
                # 沿用教师现有外部BNF接口，物理参数也不改变。
                condition = align_bnf_to_mel(
                    torch.from_numpy(bnf).unsqueeze(0).to(device), source.shape[-1],
                    mode=PAPER_STUDENT_CONFIG.bnf_alignment, sample_rate=22050,
                    n_fft=1024, hop_size=256, bnf_frame_shift_seconds=0.010,
                )
                # 加载已核验的目标参考GE2E嵌入，绝不替换为ECAPA评分特征。
                vector = np.load(row["reference_spk_emb"], allow_pickle=False).squeeze()
                # 缺维度或非有限输入必须停止，不能广播掩盖特征错误。
                if vector.shape != (256,) or not all(np.isfinite(value).all() for value in (mel, bnf, vector)):
                    raise ValueError(f"Invalid feature values: pair={row['pair_id']}")
                # 目标音色条件与当前pair唯一对应。
                speaker = torch.from_numpy(vector.astype(np.float32)).unsqueeze(0).to(device)
                # 同一pair在四系统重置同种子，教师30步首次源噪声与单步完全相同。
                generator = torch.Generator(device=device).manual_seed(int(row["seed"]))
                # 使用正式sample路径，K1已由数学测试证明等价于学生训练单步公式。
                converted = diffusion.sample(model, source, speaker, condition, num_steps=spec["steps"], start_step=950, generator=generator, diffuse_source=True)
                # 逆标准化后进入同一V，不做额外响度归一化或波形裁剪对齐。
                audio = vocoder(converted * std_tensor + mean_tensor).squeeze().cpu().numpy()
                # 输出必须是完整mel长度×256采样点，不能静默取原wav较短长度。
                if audio.ndim != 1 or len(audio) != int(row["expected_num_samples"]) or not np.isfinite(audio).all():
                    raise ValueError(f"Invalid generated waveform: {label}, pair={row['pair_id']}")
                # 每个系统保存独立文件，永不覆盖旧正式推理目录。
                destination = out / "wavs" / f"pair_{int(row['pair_id']):04d}.wav"
                destination.parent.mkdir(parents=True, exist_ok=True)
                # 与正式推理一致，仅在PCM保存前限制波形到[-1,1]。
                atomic_write_wav(destination, np.clip(audio.astype(np.float32), -1.0, 1.0), 22050)
                # 核对实际WAV格式与完整长度，并绑定写盘后的真实字节。
                info = sf.info(destination)
                if (info.samplerate, info.channels, info.frames) != (22050, 1, int(row["expected_num_samples"])):
                    raise ValueError(f"Written WAV format mismatch: {destination}")
                wav_hashes[row["pair_id"]] = digest(destination)
                # 评分表保留原模板全部身份字段，只替换本系统输出路径。
                system_rows.append({**row, "output_wav": str(destination)})
                # 明确显示系统和进度，长教师推理阶段不让主线程误以为卡死。
                print(f"[{label}] {index + 1}/{len(rows)}", flush=True)
            # 写出当前系统完整配对表，客观评价直接读取这些已生成波形。
            with (out / "pairs.csv").open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(system_rows[0]))
                writer.writeheader()
                writer.writerows(system_rows)
            # 完成之前重核所有输入，任何变化都不允许发布成功记录。
            assert_inputs_unchanged(inputs)
            # 独立报告明确区分教师训练游标和S0学生尚无更新的事实。
            write_json(out / "inference_complete.json", {
                "status": "development_diagnostic_only", "system": label, "spec": spec,
                "student_updates": 0 if label == "teacher_initialized_s0_k1" else (None if spec["steps"] == 30 else spec["global_step"]),
                "saved_training_contract_sha256": saved_contract_sha,
                "device": str(device), "inference_batch_size": 1,
                "pairs_csv_sha256": digest(out / "pairs.csv"), "wav_sha256": wav_hashes,
                "run_identity_sha256": digest(args.output_root / "run_identity.json"),
            })
            # 当前系统释放后才加载下一系统，避免不必要的GPU内存叠加。
            del model, converted, audio, source, condition, speaker
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


# 长任务结束前再次核验关键输入，防止运行中发生来源漂移。
def assert_inputs_unchanged(inputs):
    # 记录表包含数据、模型、统计量和代码。
    for path, expected in inputs.items():
        # 任一字节变化就停止，旧输出仍保留供排查。
        if digest(path) != expected:
            raise RuntimeError(f"Input changed during development evaluation: {path}")


# 评分和复用必须以完整推理记录为依据，不凭WAV文件存在就跳过。
def verify_wavs(out, rows):
    # SoundFile仅在服务器调用本函数后导入。
    import soundfile as sf
    # 读取已经完整发布的系统身份和波形指纹。
    report = json.loads((out / "inference_complete.json").read_text(encoding="utf-8"))
    # 完成记录必须绑定当前四系统共同的运行身份。
    if report["run_identity_sha256"] != digest(out.parent / "run_identity.json"):
        raise ValueError(f"Inference run identity mismatch: {out}")
    # 配对表也必须是推理时保存的原字节，不能调换波形后再评分。
    if report["pairs_csv_sha256"] != digest(out / "pairs.csv"):
        raise ValueError(f"Inference pairs changed: {out}")
    # 读取当前系统输出路径。
    actual_rows = read_csv(out / "pairs.csv")
    # 模板各字段除输出路径外必须逐行完全相同。
    if [{k: v for k, v in row.items() if k != "output_wav"} for row in actual_rows] != [{k: v for k, v in row.items() if k != "output_wav"} for row in rows]:
        raise ValueError(f"Cross-system development pair mismatch: {out}")
    # 完整记录的系统spec必须等于本脚本固定的历史身份。
    if report["spec"] != SYSTEMS[out.name] or report["system"] != out.name:
        raise ValueError(f"System identity mismatch: {out}")
    # 每个波形都必须满足原采样率、长度和字节哈希。
    for row in actual_rows:
        path = Path(row["output_wav"])
        info = sf.info(path)
        if (info.samplerate, info.channels, info.frames) != (22050, 1, int(row["expected_num_samples"])) or digest(path) != report["wav_sha256"][row["pair_id"]]:
            raise ValueError(f"Generated waveform changed: {path}")
    # 返回已核验的配对，后面直接送入原评分函数。
    return actual_rows


# 评分权重和依赖版本沿用旧正式结果，缺少资产直接失败，不临时下载另一份模型。
def metric_identity():
    # 读取软件版本只依赖包元数据，不构造任何评分网络。
    from importlib import metadata
    # 这份已核验的旧学生正式汇总只提供评估器身份，不提供本开发集的分数。
    baseline = ROOT / "exp/evaluation/fastvoicegrad_text_unseen_epoch0100_custom_hifigan/metrics/objective_summary.json"
    # 汇总本身固定哈希，防止评分器身份定义被悄悄换掉。
    if digest(baseline) != "315be70e4b54c1ffebb04047870f5a55710b58ef063675f4bfb8f49f4a5c8a69":
        raise ValueError("Audited historical evaluator summary changed")
    # 取出原资产路径、SHA和版本，复用其全部公开记录。
    old = json.loads(baseline.read_text(encoding="utf-8"))
    assets, versions = old["metric_assets"], {}
    # 所有四项指标的必要权重必须已经存在并与此前评价完全相同。
    for name, record in assets.items():
        # 缺文件会报错，不允许torch或评分器自动回落到在线下载。
        if digest(record["path"]) != record["sha256"]:
            raise ValueError(f"Objective evaluator asset changed: {name}")
    # 这些包直接影响评分数值，必须与原正式环境版本一致。
    for name in ("torch", "torchaudio", "speechbrain", "torchmetrics", "pytorch-lightning", "onnxruntime", "librosa"):
        # 使用分发包版本，不加载整套模型。
        versions[name] = metadata.version(name)
        # 出现版本变化时停止，不把新评分值直接混入旧口径对照。
        if versions[name] != old["software_versions"][name]:
            raise ValueError(f"Use the existing voicegrad-eval environment; version mismatch: {name}")
    # DNSMOS实现也与此前已单独审计的源码字节一致。
    dnsmos_source = Path(metadata.distribution("torchmetrics").locate_file("torchmetrics/functional/audio/dnsmos.py"))
    if digest(dnsmos_source) != "921a393586987e4344da0ccc264d325f63fc2a211a5c099d922b5b1a5acfc8ae":
        raise ValueError("Audited TorchMetrics DNSMOS implementation changed")
    # 保存评估器附属配置和源码身份，后续评分阶段不能换配置重解释旧CSV。
    auxiliary = {str(dnsmos_source): digest(dnsmos_source)}
    # ECAPA加载器还使用这三份附属状态，先确认齐全，避免运行时补下载。
    for name in ("classifier.ckpt", "label_encoder.ckpt", "mean_var_norm_emb.ckpt"):
        path = ROOT / "pretrained_spkrec_ecapa" / name
        auxiliary[str(path)] = digest(path)
    # UTMOS源码及ECAPA配置也是计算流程的一部分，不能只记录主要权重。
    for folder in (ROOT / "UTMOS-demo", ROOT / "pretrained_spkrec_ecapa"):
        for path in sorted(folder.rglob("*")):
            # 只读记录代码和配置，不把运行过程中生成的pycache混进身份。
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".json", ".txt") and ".git" not in path.parts:
                auxiliary[str(path)] = digest(path)
    # 记录评分口径，不复制历史8100主数作为本开发集结果。
    return {"baseline_summary_sha256": digest(baseline), "assets": assets, "software_versions": versions, "auxiliary_source_sha256": auxiliary, "method": old["method"]}


# 四项指标直接调用旧正式评价函数；只有汇总标签和开发集身份单独编写。
def evaluate_or_summarize(args, rows, inputs):
    # 所有模型依赖只允许从服务器现有缓存读取，禁止诊断时改变模型来源。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # 先核验评分资源和版本，再导入真实评价实现。
    evaluation_identity = metric_identity()
    import torch
    from evaluation import evaluate_metrics as metrics
    # 与推理同样使用既有非阻塞锁，根任务未暂停训练时安全拒绝争用。
    from fastvoicegrad.infer_paper_eval import acquire_inference_locks
    # 小开发集逐句/小批评分即可，不启动多个CPU评分进程。
    torch.set_num_threads(1)
    device = torch.device(args.device)
    # 评分设备会影响数值舍入，因此在本轮评分身份中锁定，不跨设备续算。
    evaluation_identity.update({"device": str(device), "batch_size": 4, "dnsmos_workers": 1, "dnsmos_onnx_threads": 3})
    # 评分身份文件与推理身份分离，两个阶段可以使用对应的已固定软件环境。
    identity_path = args.output_root / "metric_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != evaluation_identity:
            raise ValueError("Evaluator identity differs from existing development scores")
    else:
        # 若已有指标文件却没有评分身份，不能事后补造来源。
        if any(args.output_root.glob("*/metrics/*.csv")):
            raise ValueError("Existing score CSVs have no evaluator identity")
        write_json(identity_path, evaluation_identity)
    # 每项CSV沿用正式字段定义与完整性校验，包括CER计数和固定SVA阈值。
    fields = {"cer": metrics.CER_FIELDS, "sva": metrics.SVA_FIELDS, "dnsmos": metrics.DNSMOS_FIELDS, "utmos": metrics.UTMOS_FIELDS}
    # 仅汇总时也验证输入和WAV来源；模型运算只在metrics阶段执行。
    with acquire_inference_locks(ROOT):
        # 将本轮选中的所有系统结果放在同一张开发集对照表。
        comparison = {}
        for label in args.systems:
            # 校验本系统波形完整且逐对对应共同开发模板。
            out = args.output_root / label
            system_rows = verify_wavs(out, rows)
            score_dir = out / "metrics"
            score_dir.mkdir(exist_ok=True)
            # summary阶段只读取评分表，不加载ASR、ECAPA或自然度模型。
            if args.phase == "metrics":
                for name in args.metrics:
                    # 四个分支直接调用原函数，不重写预处理或更换评分定义。
                    if name == "cer":
                        metrics.evaluate_cer(system_rows, score_dir / "cer.csv", device, 4)
                    elif name == "sva":
                        metrics.evaluate_sva(system_rows, score_dir / "sva.csv", ROOT / "pretrained_spkrec_ecapa", device, 4)
                    elif name == "utmos":
                        metrics.evaluate_utmos(system_rows, score_dir / "utmos.csv", ROOT / "UTMOS-demo", device, 4)
                    else:
                        # DNSMOS内部固定CPU、personalized=False，保留全部四列和原OVR主列。
                        metrics.evaluate_dnsmos(system_rows, score_dir / "dnsmos.csv", 4, 1)
            # 完整读取全部请求指标；缺一行或重复编号时旧校验器直接报错。
            tables = {name: metrics.read_metric_rows(score_dir / f"{name}.csv", system_rows, name, fields[name]) for name in args.metrics}
            # 结果标签明确只是开发集，不放入正式paper comparison入口。
            summary = {"status": "development_only", "system": label, "evaluated_pairs": len(rows), "metrics": {}, "additional": {}}
            if "cer" in tables:
                # 主CER继续用全语料总错误/总字符，不换成逐句平均。
                errors = sum(int(row["char_errors"]) for row in tables["cer"])
                chars = sum(int(row["reference_chars"]) for row in tables["cer"])
                summary["metrics"]["CER_corpus_percent_lower_is_better"] = 100.0 * errors / chars
                # 同时保存原始整数分母，90条之间文本长度不一样也可复查。
                summary["additional"].update({"CER_total_errors": errors, "CER_total_reference_chars": chars, "CER_macro_percent": mean(float(row["cer_percent"]) for row in tables["cer"])})
            if "sva" in tables:
                # 原0.25阈值通过率和连续余弦分数同时呈现。
                summary["metrics"]["SVA_percent_higher_is_better"] = 100.0 * mean(int(row["verified"]) for row in tables["sva"])
                summary["additional"].update({"SVA_threshold": metrics.SVA_THRESHOLD, "SVA_mean_cosine": mean(float(row["cosine_score"]) for row in tables["sva"])})
            if "dnsmos" in tables:
                # 四列都保留；主结果仍是原先使用的mos_ovr。
                summary["metrics"]["DNSMOS_mos_ovr_mean_higher_is_better"] = mean(float(row["mos_ovr"]) for row in tables["dnsmos"])
                summary["additional"]["DNSMOS_all_four_means"] = {field: mean(float(row[field]) for row in tables["dnsmos"]) for field in metrics.DNSMOS_FIELDS[1:]}
            if "utmos" in tables:
                # 正式函数已经使用官方输出变换，这里只对逐对分数取平均。
                summary["metrics"]["UTMOS_mean_higher_is_better"] = mean(float(row["utmos"]) for row in tables["utmos"])
            # 把每份分数表的实际字节与两个阶段的身份记录绑定。
            summary.update({"metric_csv_sha256": {name: digest(score_dir / f"{name}.csv") for name in args.metrics}, "metric_identity_sha256": digest(identity_path), "inference_complete_sha256": digest(out / "inference_complete.json"), "run_identity_sha256": digest(args.output_root / "run_identity.json")})
            # 不同指标子集保存到不同文件，避免后算两项覆盖先前四项汇总。
            summary_name = "development_summary_" + "_".join(sorted(args.metrics)) + ".json"
            destination = score_dir / summary_name
            # 已完成汇总只能逐字核验数值相同，不能覆盖成另一套结果。
            if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != summary:
                raise ValueError(f"Existing development summary changed: {destination}")
            if not destination.exists():
                write_json(destination, summary)
            comparison[label] = summary
            print(json.dumps({"system": label, **summary["metrics"]}, ensure_ascii=False), flush=True)
        # 完成后再次确认输入和评分资源均未漂移。
        assert_inputs_unchanged(inputs)
        final_identity = metric_identity()
        # 执行参数不属于资产信息，比较前补回相同的明示字段。
        final_identity.update({"device": str(device), "batch_size": 4, "dnsmos_workers": 1, "dnsmos_onnx_threads": 3})
        if final_identity != evaluation_identity:
            raise ValueError("Objective evaluator assets changed during scoring")
        # 最终对照表只包含本次实际完整计算的系统与指标，没有预填论文数。
        table_path = args.output_root / ("development_comparison_" + "_".join(args.systems) + "_" + "_".join(sorted(args.metrics)) + ".json")
        result = {"status": "development_diagnostic_only", "n_pairs": len(rows), "systems": comparison, "limitations": "Training-derived development data; historical models saw these sentences. Not the official 8100-pair unseen result, and 90 pairs are not 90 independent source utterances."}
        if table_path.exists() and json.loads(table_path.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing development comparison has different results")
        if not table_path.exists():
            write_json(table_path, result)


# 命令行把推理和评分分开，主线程可分别选择原训练环境与固定evalenv。
def main():
    # 此处尚未导入PyTorch，Windows上误运行会在任何项目计算前被拒绝。
    if os.name != "posix" or Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run only from /root/FsatVoiceGrad on the rented Matpool server")
    # 把权威服务器代码放到导入路径，不导入下载到本地的参考副本。
    sys.path.insert(0, str(ROOT))
    from voicegrad.runtime_guard import assert_rented_server
    assert_rented_server(ROOT)
    # 参数仅控制独立诊断的输入和设备，不允许改论文模型设定或评分阈值。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("infer", "metrics", "summary"), required=True)
    parser.add_argument("--pairs", type=Path, default=DEV / "pairs.csv")
    parser.add_argument("--development-contract", type=Path, default=DEV / "development_contract.json")
    parser.add_argument("--output-root", type=Path, default=DEV / "evaluation_baselines_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--systems", nargs="+", choices=tuple(SYSTEMS), default=list(SYSTEMS))
    parser.add_argument("--metrics", nargs="+", choices=("cer", "sva", "dnsmos", "utmos"), default=["cer", "sva", "dnsmos", "utmos"])
    args = parser.parse_args()
    # 只允许新开发区输出，不可能误指向正式模型或评估目录。
    args.output_root = args.output_root.resolve()
    if DEV not in args.output_root.parents:
        raise ValueError("Output must be a child directory of the independent development area")
    # 固定输入模板，未核验前不加载任何教师、学生或指标模型。
    rows, inputs = validate_inputs(args)
    # 运行身份不随执行阶段、设备和系统子集改变，以支持分阶段完成同一对照。
    identity = {
        "purpose": "development_diagnostic_not_formal_text_unseen_8100",
        "limitations": ["Teacher and historical students have already seen original training data.", "Only new diagnostic training excludes the declared development sentences.", "90-pair development metrics cannot establish paper-level unseen performance."],
        "systems": SYSTEMS, "inputs": inputs, "pairs": len(rows),
        "seed_rule": "per-pair manifest seed; inference batch1; same initial source noise for all systems",
        "bnf_alignment": "legacy_length_linear", "bnf_downsampling": "legacy_decimate",
        "vocoder": "same frozen custom step2500000 for all systems", "start_step": 950,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    # 新目录只创建一次；后续阶段必须严格匹配原运行身份。
    args.output_root.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_root / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing development output belongs to different inputs or code")
    else:
        # 已有其他产物却缺身份时拒绝补造身份记录。
        if any(args.output_root.iterdir()):
            raise ValueError("Non-empty development output without run identity")
        write_json(identity_path, identity)
    # 推理先行；评分分支在同一脚本后续定义，使用完全相同的身份核验。
    if args.phase == "infer":
        infer(args, rows, inputs)
    else:
        # 评分和汇总仅消费完整推理输出，不允许临时混入其他系统波形。
        evaluate_or_summarize(args, rows, inputs)


# 被导入时只提供定义，绝不自动执行推理或加载模型。
if __name__ == "__main__":
    main()
