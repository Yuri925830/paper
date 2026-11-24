#!/usr/bin/env python3
# 预检只验证固定评价环境与资源，并用一条真 WAV 检查 UTMOS 可运行，不代表正式指标完成。
"""在 8,100-pair 推理前验证固定的论文评价环境和模型资产。"""

# 延后求值类型提示，保持旧 Python 解释器兼容。
from __future__ import annotations

# 使用 argparse 读取命令行参数。
import argparse
# 使用 csv 读写带表头的逐条记录。
import csv
# 使用 hashlib 计算数据与权重文件指纹。
import hashlib
# 使用 importlib 按独立模块名加载官方实现。
import importlib
# 使用 importlib.metadata 查询实际安装的软件版本。
import importlib.metadata
# 使用 json 读写配置及来源记录。
import json
# 使用 math 执行有限数、取整等数值辅助检查。
import math
# 使用 sys 处理解释器版本和模块搜索路径。
import sys
# 使用 tempfile 安全建立临时结果文件。
import tempfile
# 用 Path 统一构造和检查文件路径。
from pathlib import Path
# 标明张量、清单和配置的输入输出类型，不改变运行数据。
from typing import Dict, Tuple

# 复用严格自训声码器验证器，核对步数、配置、数据和权重完整性。
from hifigan.custom.validate_checkpoint import validate_checkpoint
# 限制实际计算只能在当前租用服务器执行。
from voicegrad.runtime_guard import assert_rented_server


# 固定本项目使用的公开 UTMOS 版本；论文未给出这一具体提交身份。
UTMOS_REVISION = "47212055c2ecfb02d40cec2395233b83295d3d30"
# 固定本项目使用的公开 ECAPA 版本，作为可复查的评价资产选择。
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"

# 每项同时固定字节数和 SHA-256，避免把 Git-LFS 指针或中断下载误当成模型。
PROJECT_ASSETS: Dict[str, Tuple[str, int, str]] = {
    # 记录固定 UTMOS 自然度模型权重。
    "utmos_checkpoint": (
        # UTMOS 最终评分网络的固定检查点相对路径。
        "UTMOS-demo/epoch=3-step=7459.ckpt",
        # 该检查点必须具有这一精确字节数，先快速排除截断文件。
        1238128841,
        # 进一步用完整 SHA-256 核对 UTMOS 权重身份。
        "44c57e3e4135a243b43d2c82b6a693fcd56f15f9ad0e1eb2a8b31fdecd3a49b8",
    ),
    # 记录UTMOS 所用 wav2vec 特征模型。
    "utmos_wav2vec": (
        # UTMOS 依赖的 wav2vec 表征模型位置。
        "UTMOS-demo/wav2vec_small.pt",
        # 该上游表征模型的固定文件字节数。
        950500491,
        # 该表征权重必须与登记摘要一致，不能静默下载其他版本。
        "c66c39eaed1b79a61ea8573f71e08f6641ff156b6a8f458cfaab53877dfa4a26",
    ),
    # 记录UTMOS 官方评分包装源码。
    "utmos_score_source": (
        # UTMOS 评分入口源码也纳入资源锁定。
        "UTMOS-demo/score.py",
        # 评分源码的预登记字节长度。
        3982,
        # 评分源码准确摘要，防止同权重配上修改过的评分逻辑。
        "95b0ef25c6fc7e719986584bb076195cf08a3522846678d48ca1fcc8848375d0",
    ),
    # 记录UTMOS 官方网络定义源码。
    "utmos_model_source": (
        # UTMOS 网络结构定义源码位置。
        "UTMOS-demo/model.py",
        # 网络定义文件准确字节数。
        6929,
        # UTMOS 网络定义的原始字节指纹。
        "79ca330b7d7fd83304b7254e10c5c3a0e0c30c2b8ae3af90847353ae8f6336d4",
    ),
    # 记录UTMOS 官方旧版 Lightning 权重加载源码。
    "utmos_lightning_source": (
        # UTMOS 使用的 Lightning 包装源码位置。
        "UTMOS-demo/lightning_module.py",
        # 包装源码的准确字节长度。
        1370,
        # 锁定其模型加载与调用包装实现。
        "def23a75821bb662ba95220bbb241f6762d59689f763bc007d9b8c3b254c313f",
    ),
    # 记录ECAPA 声纹嵌入模型权重。
    "ecapa_embedding": (
        # SVA 使用的 ECAPA 音色嵌入网络位置，与教师输入 GE2E 有不同用途。
        "pretrained_spkrec_ecapa/embedding_model.ckpt",
        # ECAPA 嵌入模型预期字节数。
        83316686,
        # 核对 SVA 所用 ECAPA 权重的准确身份。
        "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    ),
    # 记录ECAPA 发布模型所带分类层权重。
    "ecapa_classifier": (
        # ECAPA 公共检查点包中的分类器资源。
        "pretrained_spkrec_ecapa/classifier.ckpt",
        # 分类器文件的准确字节数。
        5534328,
        # 锁定该配套资源的 SHA-256。
        "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
    ),
    # 记录通用 VCTK V1 声码器权重。
    "public_hifigan_generator": (
        # 历史公开 VCTK V1 声码器生成器位置，自训模式不要求读取它。
        "pretrained_hifigan_22k/VCTK_V1/generator_v1",
        # 该公开生成器的准确字节长度。
        55788858,
        # 公开生成器固定权重摘要，用于历史对照身份。
        "27cdeb835874516f9404d6c1a9ea229b092fd98c329b8444f5955c24cf7b29a1",
    ),
    # 记录通用 VCTK V1 声码器配置。
    "public_hifigan_config": (
        # 公开声码器对应的 JSON 配置位置。
        "pretrained_hifigan_22k/VCTK_V1/config.json",
        # 公开配置的准确字节数，配置格式也保持来源一致。
        795,
        # 公开配置原始字节摘要。
        "12107bd79ec9e854cecb2043aa58345e3079ffeb159c9ab9a68ab25fa63deb9b",
    ),
}

# 检查服务器用户缓存目录里的 ASR 与 DNSMOS 模型。
HOME_ASSETS: Dict[str, Tuple[str, int, str]] = {
    # 记录CER 使用的 wav2vec 2.0 ASR 权重。
    "wav2vec2_asr": (
        # CER 所用固定 wav2vec2 ASR 权重在用户缓存中的相对位置。
        ".cache/torch/hub/checkpoints/wav2vec2_fairseq_large_lv60k_asr_ls960.pth",
        # ASR 检查点准确字节长度，用于排除未完整下载。
        1262046195,
        # CER 识别器固定文件摘要，不能用另一套 ASR 混算。
        "7a88965716fbd598a595209bf45c1210a18a6935cfb0cf53527fc986c5543ac7",
    ),
    # 记录DNSMOS P.808 质量分支 ONNX 模型。
    "dnsmos_p808": (
        # DNSMOS P.808 分支 ONNX 模型的缓存位置。
        ".torchmetrics/DNSMOS/DNSMOS/model_v8.onnx",
        # 该 ONNX 文件预期字节数。
        224860,
        # 锁定该 DNSMOS 分支版本的 SHA-256。
        "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
    ),
    # 记录DNSMOS 语音、背景和总体分支 ONNX 模型。
    "dnsmos_sig_bak_ovr": (
        # DNSMOS 信号/背景/总体质量网络的缓存位置。
        ".torchmetrics/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        # 该 DNSMOS ONNX 网络预期字节数。
        1157965,
        # 准确锁定总体质量评分网络版本。
        "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    ),
}

# 以下版本属于本项目已验证环境，不能把每个版本号都称为论文明确设定。
EXPECTED_DISTRIBUTIONS = {
    # 记录当前 PyTorch 版本。
    "torch": "2.6.0+cu124",
    # 记录当前 TorchAudio 版本。
    "torchaudio": "2.6.0+cu124",
    # 记录ECAPA 评价接口版本。
    "speechbrain": "1.0.3",
    # 记录DNSMOS 包装接口版本。
    "torchmetrics": "1.5.2",
    # 记录兼容旧 UTMOS 权重的 Lightning 版本。
    "pytorch-lightning": "1.5.10",
    # 记录DNSMOS 推理运行库版本。
    "onnxruntime": "1.19.2",
    # 记录下载与加载公开模型资产的接口版本。
    "huggingface-hub": "0.24.7",
    # 记录UTMOS/fairseq 旧配置解析版本。
    "hydra-core": "1.0.7",
    # 记录旧 checkpoint 配置对象的解析版本。
    "omegaconf": "2.0.6",
    # 记录音频特征辅助库版本。
    "librosa": "0.11.0",
    # This is the distribution version emitted by the pinned d03f4e7 commit.
    # 记录UTMOS wav2vec 特征依赖的固定源码发行版本。
    "fairseq": "1.0.0a0+d03f4e7",
}


# 按二进制文件内容计算 SHA-256，供后续检查是否混入其他版本。
def sha256_file(path: Path) -> str:
    # 初始化累计 SHA-256 计算器，资源按块读取。
    digest = hashlib.sha256()
    # 按二进制读取，文件编码或换行不会被自动转换。
    with path.open("rb") as handle:
        # 每次读取一块文件内容，直到文件末尾，避免一次占满内存。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            # 把当前二进制块加入同一个文件指纹。
            digest.update(chunk)
    # 返回便于写入 JSON 的十六进制文件指纹。
    return digest.hexdigest()


# 核对评价模型文件的大小与 SHA-256，确保使用已固定版本。
def check_asset(name: str, path: Path, size: int, expected_sha256: str) -> Dict[str, object]:
    # 每个固定评价资产都必须是真实文件。
    if not path.is_file():
        # 资源缺失时给出名字和路径，不把模型未加载误判为低分。
        raise FileNotFoundError(f"Missing evaluation asset {name}: {path}")
    # 先读取文件实际字节数，快速排查下载未完成或 Git LFS 指针。
    actual_size = path.stat().st_size
    # 文件大小与固定资产不一致时先停止，不继续加载。
    if actual_size != size:
        # 资源字节数不符立即报错，避免加载半个模型。
        raise RuntimeError(f"Size mismatch for {name}: got {actual_size}, expected {size}")
    # 对完整文件计算内容指纹。
    actual_sha256 = sha256_file(path)
    # 文件大小相同仍需核对字节身份，防止混用其他版本。
    if actual_sha256 != expected_sha256:
        # 摘要不符时拒绝当前资源。
        raise RuntimeError(
            # 错误同时记录实际与预期摘要，便于定位模型版本差异。
            f"SHA-256 mismatch for {name}: got {actual_sha256}, expected {expected_sha256}"
        )
    # 返回已通过核对的路径、大小和指纹。
    return {"path": str(path.resolve()), "bytes": actual_size, "sha256": actual_sha256}


# 在租用服务器上用一条真实波形检查 UTMOS 能否给出有限分数。
def run_utmos_cpu_smoke(root: Path) -> Dict[str, object]:
    # 用固定真语音在 CPU 上跑一条 UTMOS 冒烟测试，仅检查模型加载和接口是否正常。
    """Load the audited legacy checkpoint and score one real 22.05-kHz WAV on CPU."""

    # 使用项目内一条真实波形做接口诊断；这一分数不算正式模型性能。
    smoke_wav = root / "data" / "wav22k" / "p238" / "p238_001.wav"
    # 诊断波形缺失时停止，避免改用不明替代音频。
    if not smoke_wav.is_file():
        # 冒烟所需真实 WAV 缺失时停止预检。
        raise FileNotFoundError(f"UTMOS smoke-test WAV is missing: {smoke_wav}")

    # 加载 PyTorch 以检查设备和执行服务器上的评分诊断。
    import torch

    # Call the production evaluator rather than duplicating its Torch 2.6 compatibility
    # shim.  This exercises Lightning checkpoint loading, fairseq wav2vec loading,
    # resampling, and one actual forward pass without occupying the training GPU.
    # 直接测试正式 UTMOS 评分函数，使兼容层与正式流程保持一致。
    from evaluation.evaluate_metrics import evaluate_utmos

    # 保存当前 CPU 线程数，诊断结束后恢复。
    previous_threads = torch.get_num_threads()
    # 开始可能失败的文件或模型操作，后面的异常/清理分支负责收尾。
    try:
        # 诊断最多使用两个 CPU 线程，避免过度占用资源。
        torch.set_num_threads(min(previous_threads, 2))
        # 使用独立临时结果目录，退出时只清理这次诊断产物。
        with tempfile.TemporaryDirectory(prefix="voicegrad-utmos-preflight-") as temp_dir:
            # 设置逐条指标 CSV。
            output_csv = Path(temp_dir) / "utmos.csv"
            # 真实加载 UTMOS 及 wav2vec 权重，并计算一条波形的自然度分数。
            evaluate_utmos(
                [
                    {
                        # 记录转换配对编号。
                        "pair_id": "1",
                        # 记录源语音编号。
                        "source_utt": "p238_001",
                        # 记录待评估的生成波形。
                        "output_wav": str(smoke_wav),
                    }
                ],
                # 传入逐条指标 CSV。
                output_csv,
                # 指定项目内已经核对的 UTMOS 资源目录。
                root / "UTMOS-demo",
                # 仅在租用服务器 CPU 上做预检查，不占用训练 GPU。
                torch.device("cpu"),
                # 诊断只处理一条语音，不涉及正式 8,100 对汇总。
                batch_size=1,
            )
            # 重新读取正式评分器写出的 CSV，连写入接口一起核对。
            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                # 读取诊断实际产生的逐条分数。
                rows = list(csv.DictReader(handle))
    # 无论前面成功或失败，都执行资源恢复和关闭操作。
    finally:
        # 恢复运行预检查前的线程预算。
        torch.set_num_threads(previous_threads)

    # 诊断必须恰好生成指定编号的一条记录。
    if len(rows) != 1 or int(rows[0]["pair_id"]) != 1:
        # 冒烟必须返回恰好一条且 pair_id 正确，不能接受空输出或重复行。
        raise RuntimeError(f"UTMOS CPU smoke test produced invalid rows: {rows}")
    # 从实际 CSV 读取 UTMOS 分数。
    score = float(rows[0]["utmos"])
    # 真实前向评分必须给出有限值。
    if not math.isfinite(score):
        # 冒烟分数必须有限，NaN/Inf 意味着环境或模型执行异常。
        raise RuntimeError(f"UTMOS CPU smoke test produced a non-finite score: {score}")
    # 记录接口诊断通过及所用波形，不把该值当正式 unseen 平均成绩。
    return {"status": "passed", "wav": str(smoke_wav.resolve()), "utmos": score}


# 执行本脚本的输入检查、主要处理和结果记录。
def main() -> None:
    # 设置命令行参数解析器。
    parser = argparse.ArgumentParser(description=__doc__)
    # 要求明确项目根目录，不默认使用可能过期的路径。
    parser.add_argument("--project-root", type=Path, required=True)
    # 选择自训或公开声码器来源，默认自训。
    parser.add_argument("--vocoder-kind", choices=("custom", "public"), default="custom")
    # 可以明确指定自训生成器检查点，否则使用项目固定最终版本。
    parser.add_argument("--vocoder-checkpoint", type=Path, default=None)
    # 可以明确指定配套配置，否则使用同一固定声码器目录。
    parser.add_argument("--vocoder-config", type=Path, default=None)
    # 设置命令行选项。
    args = parser.parse_args()
    # 确定项目根目录。
    root = args.project_root.expanduser().resolve()
    # 核对允许的租用主机与项目路径，预检也不能在本地计算。
    assert_rented_server(root)

    # 锁定本项目已验证的 Python 3.9 评价环境。
    if sys.version_info[:2] != (3, 9):
        # 评价环境要求 Python 3.9，和训练环境分开维护。
        raise RuntimeError(f"Evaluation environment requires Python 3.9, got {sys.version}")

    # 读取实际安装的全部固定依赖版本。
    versions = {
        # 逐个查询发行版元信息，缺包会直接失败。
        name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTIONS
    }
    # 收集实际版本与本项目固定版本不一致的依赖。
    mismatches = {
        # 同时记录实际版本和预期版本，便于修复环境。
        name: {"actual": versions[name], "expected": expected}
        # 覆盖所有固定评价依赖。
        for name, expected in EXPECTED_DISTRIBUTIONS.items()
        # 只保留版本不同的项目。
        if versions[name] != expected
    }
    # 有任一环境差异时不开始正式长时评价。
    if mismatches:
        # 任一固定依赖版本不符都立即报告差异，避免同名指标不同实现。
        raise RuntimeError(f"Evaluation package version mismatch: {mismatches}")

    # 不只看 pip 元数据；这里会实际走到四项指标依赖的导入路径。
    for module_name in ("torch", "torchaudio", "fairseq", "pytorch_lightning", "onnxruntime"):
        # 实际导入依赖，发现仅 pip 版本正确但二进制不兼容的问题。
        importlib.import_module(module_name)
    # 检查 ECAPA 正式接口可以导入，不只检查包名存在。
    from speechbrain.inference.speaker import SpeakerRecognition  # noqa: F401
    # 导入实际 DNSMOS 函数，验证依赖真正能加载，而不仅查安装列表。
    from torchmetrics.functional.audio.dnsmos import (  # noqa: F401
        # 检查 TorchMetrics 的 DNSMOS 函数接口可用。
        deep_noise_suppression_mean_opinion_score,
    )

    # 加载 PyTorch 以检查设备和执行服务器上的评分诊断。
    import torch

    # 正式评价环境必须能识别 CUDA，避免意外使用错误环境。
    if not torch.cuda.is_available():
        # 预检发现当前评价环境看不到 CUDA 就报错，不静默改成另一执行路径。
        raise RuntimeError("Evaluation environment cannot see CUDA")

    # 收集所有已经核对的评价资产身份。
    assets: Dict[str, Dict[str, object]] = {}
    # 逐项核对项目目录内的固定资源。
    for name, (relative, size, digest) in PROJECT_ASSETS.items():
        # 自训声码器实验不强制要求通用基线声码器文件。
        if args.vocoder_kind == "custom" and name.startswith("public_hifigan_"):
            # 跳过本次不使用的通用声码器资源，其余评价资产仍检查。
            continue
        # 同时核对项目资产的路径、大小及哈希。
        assets[name] = check_asset(name, root / relative, size, digest)
    # 逐项核对服务器用户缓存中的固定评价权重。
    for name, (relative, size, digest) in HOME_ASSETS.items():
        # 在当前服务器用户缓存路径核对资源，未用其他服务器计算。
        assets[name] = check_asset(name, Path.home() / relative, size, digest)

    # 资产与环境通过后，再实际跑一条 UTMOS 前向诊断。
    utmos_cpu_smoke = run_utmos_cpu_smoke(root)

    # 保存本次所选声码器的独立核对结果。
    vocoder_report: Dict[str, object]
    # 自训声码器还要通过完整 250 万步检查点验证。
    if args.vocoder_kind == "custom":
        # 设置模型权重文件。
        checkpoint = (
            # 优先使用用户明确指定的自训生成器文件。
            args.vocoder_checkpoint
            # 未指定时从项目根目录构造固定默认路径。
            or root
            # 进入实验产物目录。
            / "exp"
            # 进入模型检查点目录。
            / "checkpoints"
            # 选择自训 HiFi-GAN V1+MRD 的 text-unseen 实验。
            / "hifigan_v1_mrd_vctk_text_unseen"
            # 明确要求最终 250 万步生成器。
            / "generator_step_2500000.pt"
        # 将权重路径规范化为真实绝对位置。
        ).resolve()
        # 设置模型配置。
        config = (
            # 优先采用用户明确指定的声码器配置。
            args.vocoder_config
            # 未指定配置时从项目根目录展开默认位置。
            or root
            # 进入实验产物目录。
            / "exp"
            # 进入检查点目录。
            / "checkpoints"
            # 配置与最终自训生成器来自同一实验文件夹。
            / "hifigan_v1_mrd_vctk_text_unseen"
            # 使用其原始 config.json，不重新猜测结构参数。
            / "config.json"
        # 规范化配套配置绝对路径。
        ).resolve()
        # 核对自训声码器累计步数、配置、数据及严格参数加载。
        vocoder_report = validate_checkpoint(root, checkpoint, config, 2_500_000)
    # 公开声码器分支只记录已核验的历史公开资产身份。
    else:
        # 通用基线使用前面已核对的公开权重与配置身份。
        vocoder_report = {
            # 记录流程完成状态。
            "status": "verified_historical_public_baseline",
            # 记录模型权重文件。
            "checkpoint": assets["public_hifigan_generator"],
            # 记录模型配置。
            "config": assets["public_hifigan_config"],
        }

    # 设置核对结果。
    report = {
        # 记录流程完成状态。
        "status": "paper_metric_preflight_passed",
        # 保存实际 Python 完整版本字符串，便于以后复查环境。
        "python": sys.version,
        # 记录实际可见 GPU 型号。
        "cuda_device": torch.cuda.get_device_name(0),
        # 保存本次实际核对过的软件版本。
        "package_versions": versions,
        # 记录本项目固定的 UTMOS 公开版本。
        "utmos_revision": UTMOS_REVISION,
        # 记录本项目固定的 ECAPA 公开版本。
        "ecapa_revision": ECAPA_REVISION,
        # 保留真实加载与单句前向诊断证据。
        "utmos_cpu_smoke": utmos_cpu_smoke,
        # 记录通用或自训声码器类型。
        "vocoder_kind": args.vocoder_kind,
        # 加入本次所选声码器的核对证明。
        "vocoder": vocoder_report,
        # 保存全部评价资源的字节身份，方便未来同条件复算。
        "assets": assets,
    }
    # 输出完整预检查证据，保留中文且便于人工查看。
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 全部版本、文件及真实接口诊断通过后，才输出成功标记。
    print("VOICEGRAD30_PAPER_METRIC_PREFLIGHT_PASSED")


# 仅作为脚本直接运行时执行入口，导入模块时不启动任务。
if __name__ == "__main__":
    # 启动本脚本的完整处理流程。
    main()
