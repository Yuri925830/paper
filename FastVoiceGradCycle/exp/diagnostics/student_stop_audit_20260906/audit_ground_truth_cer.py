"""只在Matpool用单线程CPU，补齐原始100句的CER基准；不修改训练。"""

# 先限制计算资源，再导入Torch，确保这个评估进程看不到GPU。
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 标准库负责只读清单、核对哈希、记录时间和独立诊断结果。
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import mean

# 只允许在已经指定的租用服务器项目目录运行。
ROOT = Path("/root/FsatVoiceGrad")
if os.name != "posix" or Path.cwd().resolve() != ROOT:
    raise RuntimeError("请只在Matpool的 /root/FsatVoiceGrad 运行。")
sys.path.insert(0, str(ROOT))

# 流式哈希避免为了核对大型ASR权重额外占用一整份文件大小的内存。
def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

# 核对已有识别模型，不允许因选错缓存而改用其他权重。
checkpoint = Path("/root/.cache/torch/hub/checkpoints/wav2vec2_fairseq_large_lv60k_asr_ls960.pth")
expected = "7a88965716fbd598a595209bf45c1210a18a6935cfb0cf53527fc986c5543ac7"
if digest(checkpoint) != expected:
    raise RuntimeError("ASR权重与三组正式评估不一致。")

# 复用正式评估的文本处理和CTC解码，避免手写第二套规则。
import numpy as np
import soundfile as sf
import torch
import torchaudio
from evaluation.evaluate_metrics import normalize_cer_text, decode_wav2vec_tokens, levenshtein_distance
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if Path(torch.hub.get_dir()).resolve() != checkpoint.parent.parent:
    raise RuntimeError("Torch缓存路径与已核验权重不一致。")

# 清单必须与DNSMOS和正式评估已核验的100句清单一致。
manifest = ROOT / "data/metadata/test_manifest.csv"
if digest(manifest) != "dec09943585580c782634900956ef80f2e9e8286ae8db539eebf0a2b2e7471a4":
    raise RuntimeError("测试清单发生变化。")
with manifest.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 100 or len({row["utt_id"] for row in rows}) != 100:
    raise RuntimeError("需要100条不重复的原始测试语音。")
# 加载大模型之前先核对文字分母，若文本口径变了就不浪费计算。
references = [normalize_cer_text(row["text"]) for row in rows]
if not all(references) or sum(map(len, references)) * 81 != 566190:
    raise RuntimeError("参考文字或分母与正式结果不一致。")

# 结果进入独立诊断目录；只创建新文件，拒绝覆盖。
out = ROOT / "exp/diagnostics/evaluation_baseline_deep_audit_20260906"
csv_path = out / "ground_truth_cer.csv"
json_path = out / "ground_truth_cer.json"
if not out.is_dir() or csv_path.exists() or json_path.exists():
    raise RuntimeError("诊断目录尚未准备好，或已有CER结果。")

# 与正式评估相同的模型，单线程CPU、eval模式和无梯度前向。
bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
model = bundle.get_model().cpu().eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
labels = bundle.get_labels()
results = []
started = time.monotonic()

# 每句立即保存识别文本，若运行失败，已完成部分仍可复查。
with csv_path.open("x", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["utt_id", "speaker", "wav_sha256", "reference_text", "hypothesis_text", "char_errors", "reference_chars", "cer_percent"])
    writer.writeheader()
    for index, row in enumerate(rows, 1):
        # 句间软检查配合外层timeout的600秒硬上限，限制CPU诊断时长。
        if time.monotonic() - started > 600:
            raise RuntimeError("达到CPU诊断十分钟上限；已完成记录保留，尚未生成完整汇总。")
        wav_path = Path(row["wav22k_path"])
        waveform, sample_rate = sf.read(wav_path, dtype="float32")
        if sample_rate != 22050 or waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise RuntimeError("原始音频格式异常：" + str(wav_path))
        # 使用正式评估相同的TorchAudio重采样，无额外响度处理。
        wave = torchaudio.functional.resample(torch.from_numpy(waveform).unsqueeze(0), 22050, bundle.sample_rate)
        with torch.inference_mode():
            emissions, _ = model(wave)
            token_ids = emissions.argmax(dim=-1)[0]
        reference = normalize_cer_text(row["text"])
        hypothesis = normalize_cer_text(decode_wav2vec_tokens(token_ids, labels))
        if not reference:
            raise RuntimeError("正确文本不能为空。")
        errors = levenshtein_distance(reference, hypothesis)
        result = {
            "utt_id": row["utt_id"], "speaker": row["speaker"], "wav_sha256": digest(wav_path),
            "reference_text": reference, "hypothesis_text": hypothesis,
            "char_errors": errors, "reference_chars": len(reference), "cer_percent": 100 * errors / len(reference),
        }
        results.append(result)
        writer.writerow(result)
        handle.flush()
        if index % 10 == 0:
            print("GT_CER", index, "/100", "seconds", round(time.monotonic() - started, 1), flush=True)

# 全语料分母与正式8100配对的分母相差固定81倍，才能直接比较内部CER。
total_errors = sum(row["char_errors"] for row in results)
total_chars = sum(row["reference_chars"] for row in results)
if total_chars * 81 != 566190:
    raise RuntimeError("归一化文本分母与正式结果不一致。")
report = {
    "status": "complete_100_ground_truth", "n": len(results),
    "char_errors": total_errors, "reference_chars": total_chars,
    "CER_corpus_percent": 100 * total_errors / total_chars,
    "CER_utterance_mean_percent_diagnostic_only": mean(row["cer_percent"] for row in results),
    "elapsed_seconds": time.monotonic() - started,
    "asr_checkpoint_sha256": expected, "test_manifest_sha256": digest(manifest),
    "evaluation_source_sha256": digest(ROOT / "evaluation/evaluate_metrics.py"),
    "torch": torch.__version__, "torchaudio": torchaudio.__version__, "device": "cpu", "threads": 1,
    "limitations": ["原评估使用GPU批处理，本次CPU单条处理可能有微小浮点差异。", "原始录音完整长度与合成语音的帧移对齐长度可能略有不同。", "本结果不能确认作者的ASR权重、解码及文本处理规则。"],
}
with json_path.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False), flush=True)
