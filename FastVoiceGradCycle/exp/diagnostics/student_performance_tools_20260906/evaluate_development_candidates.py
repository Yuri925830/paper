# 候选模型开发评估入口：独立保存DDIM教师对照和显式学生权重，不修改四系统基线文件。
"""Evaluate explicitly identified candidates on the unchanged 90 development pairs.

This is not the formal 8100-pair evaluation. The DDIM teacher is a separately
labelled diagnostic sampler; no formal teacher code, weights or inputs change.
"""

# 标准库负责身份、路径和命令行；导入本脚本本身不会加载任何模型。
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# 所有真实数据和模型运算只允许在当前租用服务器发生。
ROOT = Path("/root/FsatVoiceGrad")
DEV = ROOT / "exp/diagnostics/student_development_20260906"
# 两个复用模块必须与当前已经审核的冻结文件逐字节一致。
BASELINE_SHA = "14142b4909583f9a79424e9a23c188a00605c8cbfd9797daa7eff04848849d21"
DDIM_SHA = "0e3100d14e421a6f7eb1a06b1dc79d195091c2f20a35f4c3e4266e77dbd3c6cc"
# 教师身份与四系统基线完全相同，只有新DDIM对照的反向采样公式不同。
TEACHER_SHA = "1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6"
TEACHER_PATH = ROOT / "exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt"


# 加载辅助模块之前先独立核对字节，不能先执行未核验代码再检查其哈希。
def file_sha(path):
    result = hashlib.sha256()
    # 分块读取大检查点，避免额外占满CPU内存。
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


# 每次加载新建私有模块对象，修改候选系统字典不会影响正式入口或正在运行的基线进程。
def load_verified_module(path, expected_sha, namespace):
    # SHA不符立即停止，绝不把改过的辅助文件冒充原冻结版本。
    if file_sha(path) != expected_sha:
        raise ValueError(f"Frozen helper identity mismatch: {path}")
    # 使用独立模块名，不覆盖voicegrad、fastvoicegrad或基线公开模块名。
    specification = importlib.util.spec_from_file_location(namespace, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# 只接受独立候选名称及显式权重身份，拒绝把新候选写成四系统旧基线。
def candidate_spec(args):
    # DDIM教师使用固定权重、固定30次前向和独立标签。
    if args.system == "teacher500_ddim30":
        if any(value is not None for value in (args.checkpoint, args.checkpoint_sha256, args.run_contract, args.label, args.epoch, args.global_step)):
            raise ValueError("DDIM teacher identity is fixed; do not supply student options")
        return "teacher500_ddim30", {
            "checkpoint": str(TEACHER_PATH), "sha256": TEACHER_SHA,
            "state_key": "model", "epoch": 500, "global_step": 615000,
            "steps": 30, "sampling": "ddim_eta0_shared_noisy_start_v1",
        }
    # 学生必须明确给出路径、字节摘要、游标和可识别标签，避免默默使用latest。
    required = (args.checkpoint, args.checkpoint_sha256, args.label, args.epoch, args.global_step)
    if any(value is None for value in required):
        raise ValueError("student_k1 requires --checkpoint, --checkpoint-sha256, --label, --epoch and --global-step")
    # 标签限定普通目录字符，并强制与教师和旧基线名称区分。
    if not re.fullmatch(r"student_[a-z0-9_]{1,80}", args.label):
        raise ValueError("Student label must match student_[a-z0-9_]{1,80}")
    # 大小写摘要统一后只接受完整64位SHA，不接受前缀匹配。
    expected_sha = args.checkpoint_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("Expected a full 64-character checkpoint SHA-256")
    if args.epoch < 0 or args.global_step < 0:
        raise ValueError("Epoch and global step must be nonnegative integers")
    # 路径以调用时的明确位置为准，禁止从名称推断另一个检查点。
    checkpoint = args.checkpoint.resolve()
    if ROOT not in checkpoint.parents:
        raise ValueError("Candidate checkpoint must be inside the authoritative project")
    # 开发权重使用epoch0及实际更新步数，不把600步写成已完成一轮。
    return args.label, {
        "checkpoint": str(checkpoint), "sha256": expected_sha, "state_key": "student",
        "epoch": args.epoch, "global_step": args.global_step, "steps": 1,
        "sampling": "existing_student_t950_to_zero",
    }


# 检查权重自身的游标、网络字段和原训练契约；只读验证，不恢复训练状态。
def validate_payload(payload, spec, run_contract):
    # 文件名不是身份依据，必须同时核对真实内部epoch和global_step。
    if (payload.get("epoch"), payload.get("global_step")) != (spec["epoch"], spec["global_step"]):
        raise ValueError("Candidate checkpoint cursor does not match explicit epoch/step")
    if spec["state_key"] not in payload or not payload[spec["state_key"]]:
        raise ValueError("Candidate checkpoint has no nonempty requested state dictionary")
    # 教师的完整字节已锁定，不需要人为补造学生训练契约。
    if spec["state_key"] == "model":
        return None
    # 学生必须提供原训练契约，不能仅凭一个state_dict声称完成开发训练。
    if run_contract is None:
        raise ValueError("Student candidate must have its original run_contract.json")
    # checkpoint里的run_contract_sha256是规范化JSON摘要，与文件字节摘要分开核对。
    canonical = json.dumps(run_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract_sha = hashlib.sha256(canonical).hexdigest()
    if payload.get("run_contract_sha256") != contract_sha:
        raise ValueError("Student checkpoint is not bound to the supplied training contract")
    # 新开发权重配置与旧学生契约字段不同，分别读取，不静默补值。
    configuration = payload.get("configuration", run_contract.get("configuration", run_contract.get("config", {})))
    # 两个BNF接口以及扩散日程在此实验保持不变；字段缺失则由已绑定契约另行明确给出。
    if not isinstance(configuration, dict):
        raise ValueError("Candidate configuration must be a mapping")
    for key, expected in (("bnf_alignment", "legacy_length_linear"), ("bnf_downsampling", "legacy_decimate"), ("diffusion_steps", 1000), ("cosine_offset", 0.008)):
        if configuration.get(key) != expected:
            raise ValueError(f"Candidate changes an interface held fixed in this comparison: {key}")
    return contract_sha


# 四基线和候选共用每对seed产生的第一份标准高斯源噪声，任何分支都只加噪一次。
def predict_candidate(model, diffusion, source, speaker, bnf, seed, mode, ddim_sampler=None):
    import torch
    from fastvoicegrad.objectives import reverse_mean_to_zero
    # 生成器只在当前pair内部使用，不消耗前一对或其他系统的随机序列。
    generator = torch.Generator(device=source.device).manual_seed(int(seed))
    noise = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
    # 源加噪固定到950，与baseline的diffuse_source=True对应同一初始状态。
    timesteps = torch.full((source.shape[0],), 950, device=source.device, dtype=torch.long)
    x_start = diffusion.q_sample(source, timesteps, noise=noise)
    # 整条候选推理关闭梯度，不修改教师、学生或任何调用者提供的输入。
    with torch.inference_mode():
        if mode == "teacher500_ddim30":
            if ddim_sampler is None:
                raise ValueError("DDIM teacher mode requires the verified sampler")
            # 传入已经加噪的同一x950，DDIM模块不会再次加噪或生成随机反向噪声。
            converted = ddim_sampler(teacher=model, diffusion=diffusion, x_start=x_start, speaker_embedding=speaker, bnf=bnf, num_steps=30, start_step=950)
        elif mode == "student_k1":
            # 使用生产学生公式从同一x950一步还原，不改变正式教师sample实现。
            converted = reverse_mean_to_zero(model, diffusion, x_start, timesteps, speaker, bnf)
        else:
            raise ValueError(f"Unknown candidate mode: {mode}")
    # 返回实际初始状态供身份诊断，便于确认候选没有重复加噪。
    return converted, x_start


# 候选推理只新增两种已声明模型路径，其他预处理、声码器和WAV保存完全沿用基线。
def infer_candidate(args, shared, rows, inputs, spec, contract):
    import numpy as np
    import soundfile as sf
    import torch
    from fastvoicegrad.components import load_frozen_vocoder, load_torch_checkpoint, load_mel_statistics, set_requires_grad
    from fastvoicegrad.config import PAPER_STUDENT_CONFIG
    from fastvoicegrad.infer_paper_eval import acquire_inference_locks
    from hifigan.tools.vocode import atomic_write_wav
    from voicegrad.diffusion import VoiceGradDiffusion
    from voicegrad.feature_alignment import align_bnf_to_mel
    from voicegrad.infer_voicegrad30_paper_eval import feature_matrix
    from voicegrad.train import build_model
    # 每次只加载一个候选和共同G；共享锁会在训练仍活跃时拒绝争用。
    device = torch.device(args.device)
    torch.set_num_threads(4)
    # 完成文件存在时只能验真复用，部分目录不进行猜测式覆盖或续算。
    out = args.output_root / args.systems[0]
    if (out / "inference_complete.json").exists():
        shared.verify_wavs(out, rows)
        return
    if out.exists():
        raise ValueError("Partial candidate output exists; use a new independent output-root")
    # 不允许运行配置在后续维护中偷偷改变本对照的固定接口。
    if (PAPER_STUDENT_CONFIG.bnf_alignment, PAPER_STUDENT_CONFIG.bnf_downsampling, PAPER_STUDENT_CONFIG.diffusion_steps, PAPER_STUDENT_CONFIG.cosine_offset) != ("legacy_length_linear", "legacy_decimate", 1000, 0.008):
        raise ValueError("Candidate inference requires the unchanged audited legacy interface")
    with acquire_inference_locks(ROOT), torch.inference_mode():
        # 哈希已经核验；反序列化后仍须验证实际游标和训练契约。
        payload = load_torch_checkpoint(Path(spec["checkpoint"]))
        saved_contract_sha = validate_payload(payload, spec, contract)
        model = build_model("legacy_decimate")
        model.load_state_dict(payload[spec["state_key"]], strict=True)
        del payload
        gc.collect()
        # 全部参数冻结、全部子模块eval，避免DDIM目标含训练随机层。
        model.to(device).eval()
        set_requires_grad(model, False)
        # G仍通过已有完整证明验证，不能用公开G冒充自训G。
        vocoder, _, _ = load_frozen_vocoder(ROOT, shared.VOCODER / "generator_step_2500000.pt", shared.VOCODER / "config.json", shared.VOCODER / "final_checkpoint_attestation_comments_20260906.json", PAPER_STUDENT_CONFIG, device)
        mean_tensor, std_tensor = load_mel_statistics(ROOT / "data", device)
        mean_array = np.load(ROOT / "data/stats/mel_mean.npy", allow_pickle=False).astype(np.float32).reshape(80, 1)
        std_array = np.load(ROOT / "data/stats/mel_std.npy", allow_pickle=False).astype(np.float32).reshape(80, 1)
        diffusion = VoiceGradDiffusion(1000, offset=0.008).to(device)
        # DDIM辅助模块与baseline辅助模块都固定字节，不修改它们的文件。
        ddim = load_verified_module(args.ddim_module, DDIM_SHA, "_development_candidate_ddim")
        if ddim.DDIM_SAMPLER_ID != "ddim_eta0_shared_noisy_start_v1":
            raise ValueError("Unexpected DDIM sampler identifier")
        # 输入与模型全部通过核验之后才建立本候选输出目录。
        out.mkdir(parents=True, exist_ok=False)
        system_rows, wav_hashes, initial_hashes = [], {}, {}
        for index, row in enumerate(rows):
            # 预处理与四基线逐行相同，不对候选单独调标准化或BNF对齐。
            mel = feature_matrix(Path(row["source_mel"]), 80)
            bnf = feature_matrix(Path(row["source_bnf"]), 144)
            source = torch.from_numpy(((mel - mean_array) / std_array).astype(np.float32)).unsqueeze(0).to(device)
            condition = align_bnf_to_mel(torch.from_numpy(bnf).unsqueeze(0).to(device), source.shape[-1], mode="legacy_length_linear", sample_rate=22050, n_fft=1024, hop_size=256, bnf_frame_shift_seconds=0.010)
            vector = np.load(row["reference_spk_emb"], allow_pickle=False).squeeze()
            if vector.shape != (256,) or not all(np.isfinite(value).all() for value in (mel, bnf, vector)):
                raise ValueError(f"Invalid candidate inputs: pair={row['pair_id']}")
            speaker = torch.from_numpy(vector.astype(np.float32)).unsqueeze(0).to(device)
            # 同一个seed产生与baseline相同的源噪声，DDIM与单步都从同一x950开始。
            converted, x_start = predict_candidate(model, diffusion, source, speaker, condition, int(row["seed"]), args.system, ddim.sample_ddim_teacher)
            initial_hashes[row["pair_id"]] = hashlib.sha256(x_start.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            # 波形前不增加响度归一化，输出完整mel帧数对应的采样点。
            audio = vocoder(converted * std_tensor + mean_tensor).squeeze().cpu().numpy()
            if audio.ndim != 1 or len(audio) != int(row["expected_num_samples"]) or not np.isfinite(audio).all():
                raise ValueError(f"Invalid candidate waveform: pair={row['pair_id']}")
            destination = out / "wavs" / f"pair_{int(row['pair_id']):04d}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            # 与baseline一致，只在保存PCM前裁到合法振幅范围。
            atomic_write_wav(destination, np.clip(audio.astype(np.float32), -1.0, 1.0), 22050)
            info = sf.info(destination)
            if (info.samplerate, info.channels, info.frames) != (22050, 1, int(row["expected_num_samples"])):
                raise ValueError(f"Candidate WAV format mismatch: {destination}")
            wav_hashes[row["pair_id"]] = shared.digest(destination)
            system_rows.append({**row, "output_wav": str(destination)})
            print(f"[{args.systems[0]}] {index + 1}/{len(rows)}", flush=True)
        # 配对模板除输出位置外原样保留，原指标函数直接消费同一格式。
        with (out / "pairs.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(system_rows[0]))
            writer.writeheader()
            writer.writerows(system_rows)
        shared.assert_inputs_unchanged(inputs)
        # 完成记录包含实际checkpoint、DDIM及baseline源码身份，绝不伪装为正式推理。
        shared.write_json(out / "inference_complete.json", {
            "status": "development_candidate_only", "system": args.systems[0], "spec": spec,
            "student_updates": spec["global_step"] if args.system == "student_k1" else None,
            "saved_training_contract_sha256": saved_contract_sha,
            "pairs_csv_sha256": shared.digest(out / "pairs.csv"), "wav_sha256": wav_hashes,
            "x950_sha256_per_pair": initial_hashes, "device": str(device), "inference_batch_size": 1,
            "baseline_driver_sha256": BASELINE_SHA, "ddim_module_sha256": DDIM_SHA,
            "run_identity_sha256": shared.digest(args.output_root / "run_identity.json"),
        })


# 只有主线程显式启动此入口才会做推理或评分，导入测试时没有项目工作负载。
def main():
    if os.name != "posix" or Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run candidate diagnostics only from the rented Matpool project root")
    sys.path.insert(0, str(ROOT))
    from voicegrad.runtime_guard import assert_rented_server
    assert_rented_server(ROOT)
    # 模式、checkpoint游标和SHA都要写清楚，省略不会默认加载另一个学生。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("infer", "metrics", "summary"), required=True)
    parser.add_argument("--system", choices=("teacher500_ddim30", "student_k1"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--run-contract", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--global-step", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pairs", type=Path, default=DEV / "pairs.csv")
    parser.add_argument("--development-contract", type=Path, default=DEV / "development_contract.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--metrics", nargs="+", choices=("cer", "sva", "dnsmos", "utmos"), default=["cer", "sva", "dnsmos", "utmos"])
    args = parser.parse_args()
    label, spec = candidate_spec(args)
    # 固定辅助文件与本驱动同目录，避免从sys.path导入另一份同名采样器。
    helper_path = Path(__file__).resolve().with_name("evaluate_development_baselines.py")
    args.ddim_module = Path(__file__).resolve().with_name("ddim_teacher.py")
    if file_sha(args.ddim_module) != DDIM_SHA:
        raise ValueError("Candidate DDIM helper does not match its frozen SHA")
    shared = load_verified_module(helper_path, BASELINE_SHA, "_development_candidate_shared")
    # 此私有空间只用于通用身份/CSV/评分函数；不调用原固定四系统infer或main。
    shared.SYSTEMS = {label: spec}
    args.systems = [label]
    args.output_root = (args.output_root or DEV / ("evaluation_candidate_" + label)).resolve()
    if DEV not in args.output_root.parents:
        raise ValueError("Candidate output must stay inside the independent development area")
    # 复用完整90对和receipt校验，所选候选权重另以显式SHA核对。
    rows, inputs = shared.validate_inputs(args)
    # 只读原学生训练契约；默认使用权重所在目录的run_contract.json。
    contract = None
    if args.system == "student_k1":
        args.run_contract = (args.run_contract or Path(spec["checkpoint"]).parent / "run_contract.json").resolve()
        if ROOT not in args.run_contract.parents:
            raise ValueError("Candidate training contract must be inside the project")
        contract = json.loads(args.run_contract.read_text(encoding="utf-8"))
        inputs[str(args.run_contract)] = shared.digest(args.run_contract)
    # 双重记录辅助模块固定摘要和本驱动摘要，推理后变更实现会拒绝评分。
    for path in (Path(__file__).resolve(), helper_path, args.ddim_module):
        inputs[str(path)] = shared.digest(path)
    identity = {
        "purpose": "development_candidate_not_formal_8100", "systems": shared.SYSTEMS,
        "pairs": len(rows), "inputs": inputs, "baseline_driver_sha256": BASELINE_SHA,
        "ddim_module_sha256": DDIM_SHA, "start_step": 950, "inference_batch_size": 1,
        "seed_rule": "manifest per-pair seed, same initial q950 noise as frozen four-system baselines",
        "bnf_alignment": "legacy_length_linear", "bnf_downsampling": "legacy_decimate",
        "vocoder": "same frozen custom step2500000", "candidate_training_contract": str(args.run_contract) if args.run_contract else None,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "limitations": ["DDIM teacher is a separate diagnostic sampler, not formal VoiceGrad-30.", "Only 10 independent source sentences underlie the 90 conversion pairs.", "The teacher saw these original training utterances; development scores cannot replace unseen evaluation."],
    }
    # 新命名空间仍采用严格目录身份，不复用基线或另一候选的波形。
    args.output_root.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_root / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing candidate directory has another run identity")
    else:
        if any(args.output_root.iterdir()):
            raise ValueError("Nonempty candidate directory lacks its original run identity")
        shared.write_json(identity_path, identity)
    if args.phase == "infer":
        infer_candidate(args, shared, rows, inputs, spec, contract)
    else:
        # 原CER/SVA/DNSMOS/UTMOS实现和模型/版本检查直接复用，不增设替代指标口径。
        shared.evaluate_or_summarize(args, rows, inputs)


# 启动保护避免测试导入此文件时运行推理。
if __name__ == "__main__":
    main()
