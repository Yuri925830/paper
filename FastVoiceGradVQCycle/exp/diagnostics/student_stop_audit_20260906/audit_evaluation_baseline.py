"""仅在租用服务器上检查评估记录，并给100条原始测试语音计算DNSMOS基准。

本脚本不加载学生、教师或声码器，不训练，不使用GPU，也不改写已有结果。
四列分数全部保存；不能挑选更接近论文的一列来替换原来的主指标。
"""

# 在导入任何计算库前，屏蔽GPU并把CPU并行量限制为一个线程。
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 这些标准库只负责读取表格、核对身份和保存独立的排查记录。
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# 只允许从当前租用服务器的项目根目录运行，防止误在Windows执行。
ROOT = Path("/root/FsatVoiceGrad")
if os.name != "posix" or Path.cwd().resolve() != ROOT:
    raise RuntimeError("请只在Matpool服务器的 /root/FsatVoiceGrad 中运行。")

# 新目录只用于本次诊断；已有目录时停止，避免覆盖任何既有记录。
OUT = ROOT / "exp/diagnostics/evaluation_baseline_deep_audit_20260906"
OUT.mkdir(exist_ok=False)

# 哈希用于记录文件身份，不读取或展示任何连接凭据。
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# 统一读取UTF-8表格，保留原字段内容以便逐列比较实验配对。
def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

# 预先核对两个已有评分模型，文件不存在或身份不符时不允许自动下载。
assets = {
    "/root/.torchmetrics/DNSMOS/DNSMOS/sig_bak_ovr.onnx": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    "/root/.torchmetrics/DNSMOS/DNSMOS/model_v8.onnx": "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
}
for path, expected in assets.items():
    if sha256(path) != expected:
        raise RuntimeError("DNSMOS评分模型身份不符：" + path)

# 三组正式结果都只读；当前两轮学生声码器对照尚未做正式指标，不混入此表。
tags = [
    "voicegrad30_text_unseen_epoch0500_public_hifigan",
    "voicegrad30_text_unseen_epoch0500_custom_hifigan",
    "fastvoicegrad_text_unseen_epoch0100_custom_hifigan",
]
fields = ["p808_mos", "mos_sig", "mos_bak", "mos_ovr"]
report = {"created_utc": datetime.now(timezone.utc).isoformat(), "assets": assets, "runs": {}}
first_pairs = None
for tag in tags:
    # 依赖正式汇总文件指出配对表，避免凭文件名猜测使用了哪些样本。
    metrics = ROOT / "exp/evaluation" / tag / "metrics"
    summary = json.loads((metrics / "objective_summary.json").read_text())
    pairs = read_csv(summary["pairs_csv"])
    scores = read_csv(metrics / "dnsmos.csv")
    if len(pairs) != 8100 or len(scores) != 8100:
        raise RuntimeError("正式结果必须完整包含8100条：" + tag)
    # 去掉必然不同的输出文件路径，其余11列要求与首组逐条一致。
    identities = [{k: v for k, v in row.items() if k != "output_wav"} for row in pairs]
    if first_pairs is None:
        first_pairs = identities
    counts = Counter(row["source_utt"] for row in pairs)
    report["runs"][tag] = {
        "n": len(scores),
        "pairs_nonoutput_identical_to_public_teacher": identities == first_pairs,
        "unique_source_count": len(counts),
        "source_repetition_counts": sorted(set(counts.values())),
        "dnsmos_means": {key: mean(float(row[key]) for row in scores) for key in fields},
        "headline_metrics": summary["headline_metrics"],
        "method": summary["method"],
        "dnsmos_csv_sha256": sha256(metrics / "dnsmos.csv"),
    }

# 所有项目计算到这里才开始，且已屏蔽GPU，两个计算库也限制为单线程。
import numpy as np
import soundfile as sf
import torch
import torchmetrics
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 原始100条测试语音每条在正式配对中重复81次，等权均值因此可作内部基准。
manifest = ROOT / "data/metadata/test_manifest.csv"
rows = read_csv(manifest)
if len(rows) != 100 or {row["utt_id"] for row in rows} != set(counts):
    raise RuntimeError("原始测试语音身份与正式评估来源不一致。")
results = []
for index, row in enumerate(rows, 1):
    # 与原评估一致：读取PCM浮点波形，不额外做峰值或响度归一化。
    wav_path = Path(row["wav22k_path"])
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if sample_rate != 22050 or waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise RuntimeError("原始测试语音格式异常：" + str(wav_path))
    # 每次评分一条完整语音；短于模型窗口时由原评分函数按官方方式重复语音。
    with torch.inference_mode():
        values = deep_noise_suppression_mean_opinion_score(
            torch.from_numpy(waveform).unsqueeze(0), fs=22050,
            personalized=False, device="cpu", num_threads=1,
        ).cpu().numpy()[0]
    if values.shape != (4,) or not np.isfinite(values).all():
        raise RuntimeError("DNSMOS返回了异常分数。")
    results.append({
        "utt_id": row["utt_id"], "speaker": row["speaker"],
        "wav_sha256": sha256(wav_path), "num_samples": len(waveform),
        **{key: float(value) for key, value in zip(fields, values)},
    })
    if index % 10 == 0:
        print("GT_DNSMOS", index, "/", len(rows), flush=True)

# 原始语音基准只是本数据和本评分器的对照，不能据此断言作者用了哪列分数。
report["ground_truth"] = {
    "n": len(results), "manifest_sha256": sha256(manifest),
    "dnsmos_means": {key: mean(row[key] for row in results) for key in fields},
    "rows": results,
}
report["runtime"] = {"torch": torch.__version__, "torchmetrics": torchmetrics.__version__, "device": "cpu", "num_threads": 1}
report["limitations"] = [
    "尚无作者评分模型及输出列确认，不能把P.808改称论文DNSMOS主结果。",
    "这里只新增原始语音DNSMOS基准，没有新增主观评估或正式学生推理。",
    "旧100轮学生与当前clean_x0两轮对照的训练目标不同，不合并解释。",
]
# 只写新建诊断目录；既有模型、源码、配对和指标文件保持不变。
(OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"ground_truth_means": report["ground_truth"]["dnsmos_means"], "output": str(OUT / "results.json")}, ensure_ascii=False), flush=True)
