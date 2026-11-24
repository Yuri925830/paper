# 说明本函数或文件的核验用途；所有不一致都应中止，不能静默放过。
"""Final fail-closed audit for the custom-vocoder 8,100-pair pipeline."""

# 延迟处理类型提示，保持与既有 Python 版本兼容。
from __future__ import annotations

# 解析项目根目录和最终报告路径，不从其他任务推断输入。
import argparse
# 读取配对清单和每项指标 CSV 的列名与数据行。
import csv
# 计算 SHA-256，记录模型、配置、代码和结果的准确文件身份。
import hashlib
# 读取各阶段结构化报告并编码最终证据记录。
import json
# 用原子替换发布完整验证报告。
import os
# 安全组合绝对路径并检查文件存在。
from pathlib import Path
# 类型提示描述报告字段与映射，不改变模型或指标数值。
from typing import Any, Dict, Mapping

# 复用正向租用服务器身份检查，避免在未授权主机执行项目审计。
from voicegrad.runtime_guard import assert_rented_server


# 固定这份教师对照实验已经登记的模型字节身份；新学生实验不能自动替换它。
EXPECTED_TEACHER_SHA256 = (
    # 固定教师权重的完整 SHA-256；必须逐字符匹配，不能仅凭同名文件认定身份。
    "1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6"
)
# 自训声码器必须完成的总更新次数。
EXPECTED_STEP = 2_500_000
# 完整评估的固定配对数。
EXPECTED_PAIRS = 8_100
# 四项客观指标名称，UTMOS/DNSMOS/SVA 越高越好，CER 越低越好。
METRIC_NAMES = ("utmos", "dnsmos", "cer", "sva")
# 所有会影响教师推理的核心源码，逐份记录字节摘要。
INFERENCE_IMPLEMENTATION_PATHS = (
    # 正式教师配对、条件准备和推理流程纳入实现审计。
    "voicegrad/infer_voicegrad30_paper_eval.py",
    # 扩散系数、加噪及反向采样数学实现纳入审计。
    "voicegrad/diffusion.py",
    # 教师网络结构与内部 BNF 接口纳入审计。
    "voicegrad/model.py",
    # mel 到波形的加载、尺度和声码器调用接口纳入审计。
    "hifigan/tools/vocode.py",
    # 生成器/判别器结构的实现指纹，哪怕注释变化也会被检测。
    "hifigan/official/models.py",
)


# 分块计算文件字节摘要，适合较大的模型检查点。
def sha256_file(path: Path) -> str:
    # 初始化文件 SHA-256 累计器，分块读取时逐步更新。
    digest = hashlib.sha256()
    # 二进制读取，文件指纹不受文字解码与换行转换影响。
    with path.open("rb") as handle:
        # 每次最多读取 1 MiB，读到空字节串就结束，避免大权重占满内存。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            # 将这一块原始字节加入累计文件摘要。
            digest.update(chunk)
    # 返回可写入 JSON 并逐字符比较的十六进制文件指纹。
    return digest.hexdigest()


# 按与声码器证明相同的稳定 JSON 规则计算内容摘要。
def canonical_sha256(value: Any) -> str:
    # 说明本函数或文件的核验用途；所有不一致都应中止，不能静默放过。
    """Hash the same canonical JSON representation used by checkpoint attestation."""

    # 把 JSON 转为键序和空白固定的 UTF-8 字节，保证内容摘要稳定。
    encoded = json.dumps(
        # 把待验证的完整对象交给稳定 JSON 编码，包含所有字段。
        value,
        # 保留中文文字，编码规则与生成证明时保持一致。
        ensure_ascii=False,
        # 固定字典键的顺序，避免同内容不同排列产生不同规范摘要。
        sort_keys=True,
        # 去掉 JSON 中无意义的空白，生成稳定内容表示。
        separators=(",", ":"),
        # 拒绝 NaN 和无穷大，防止出现不规范且难以稳定比较的 JSON。
        allow_nan=False,
    # 统一转为 UTF-8 字节后再计算规范摘要。
    ).encode("utf-8")
    # 对稳定 JSON 字节计算 SHA-256，与证明使用完全相同规则。
    return hashlib.sha256(encoded).hexdigest()


# 读取必需的 JSON 报告并核对其顶层结构。
def load_object(path: Path) -> Dict[str, Any]:
    # 要求目标是实际普通文件，不把目录或缺失路径当成有效报告。
    if not path.is_file():
        # 缺文件立即停止，并给出准确路径。
        raise FileNotFoundError(path)
    # 读取或接收待验证 JSON 内容，后续要求它为对象。
    value = json.loads(path.read_text(encoding="utf-8"))
    # 要求报告顶层为 JSON 对象，数组或标量均不符合验证格式。
    if not isinstance(value, dict):
        # 指出结构不符合约定的报告文件，不猜测如何兼容。
        raise ValueError(f"Expected JSON object: {path}")
    # 返回已经通过类型检查的对象，供后续字段核对。
    return value


# 集中执行精确一致性断言，给出清晰的差异标签。
def require_equal(actual: Any, expected: Any, label: str) -> None:
    # 实际值与期望值必须精确相同，任何偏差都不能标为完成。
    if actual != expected:
        # 错误消息同时给出项目名、实际值和预期值，便于定位是哪条证据不一致。
        raise ValueError(f"{label}: got {actual!r}, expected {expected!r}")


# 确保摘要集合使用有效键值映射。
def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    # 要求字段具有键值映射结构，以便逐项比较指纹。
    if not isinstance(value, Mapping):
        # 类型错误立即指出实际类型，防止缺字段被空值蒙混通过。
        raise ValueError(f"{label}: expected a JSON object, got {type(value).__name__}")
    # 返回已经通过类型检查的对象，供后续字段核对。
    return value


# 要求一项客观指标恰好覆盖完整且唯一的 8100 个配对。
def validate_metric_csv(path: Path) -> None:
    # 按 CSV 的换行规则读取完整指标文件。
    with path.open("r", encoding="utf-8", newline="") as handle:
        # 每一行按列名读为字典，保留配对编号。
        rows = list(csv.DictReader(handle))
    # 从指标 CSV 中读取全部配对编号，用于发现缺失或重复。
    ids = [int(row["pair_id"]) for row in rows]
    # 核对 f{path.name} row count；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(len(rows), EXPECTED_PAIRS, f"{path.name} row count")
    # 核对 f{path.name} pair IDs；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(sorted(ids), list(range(EXPECTED_PAIRS)), f"{path.name} pair IDs")


# 只在报告完整写好后原子发布，避免外部读取半份结果。
def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    # 只创建最终审计报告父目录，已有目录保持不变。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 在正式报告旁建立临时文件名，完整写好前不发布新报告。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 先写完整临时报告，避免调用者读取半份最终证据。
    temporary.write_text(
        # 用可读 UTF-8 JSON 保存全部字段，并保留文件末尾换行。
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 临时报告完整写出后一次性发布为正式文件。
    os.replace(temporary, path)


# 按模型、数据、推理、逐句指标、汇总和论文比较顺序核对完整证据链。
def main() -> None:
    # 创建本次最终审计的命令行参数读取器。
    parser = argparse.ArgumentParser(description=__doc__)
    # 必须明确权威项目根目录，不能默认误查本地副本。
    parser.add_argument("--project-root", type=Path, required=True)
    # 必须指定最终审计报告位置，避免覆盖不相关结果。
    parser.add_argument("--output-report", type=Path, required=True)
    # 解析实际项目根目录和输出报告位置。
    args = parser.parse_args()

    # 规范化项目绝对路径，所有数据与模型身份从此目录核对。
    root = args.project_root.resolve()
    # 核对租用主机、机器身份和项目路径，任何不符都停止。
    assert_rented_server(root)
    # 固定本项目 text-unseen 教师第 500 epoch 检查点。
    teacher = (
        # 从已核验的项目根目录展开下方路径。
        root
        # 明确指向第 500 epoch 的教师对照权重。
        / "exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt"
    )
    # 自训 HiFi-GAN V1+MRD 的检查点目录。
    vocoder_dir = root / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
    # 选择最终 generator_step_2500000.pt，不按最近修改时间猜模型。
    checkpoint = vocoder_dir / "generator_step_2500000.pt"
    # 与最终声码器配套的原始 JSON 配置。
    config_path = vocoder_dir / "config.json"
    # 原最终验证证明路径；注释后字节变化需要独立的新证明，不能改写历史记录。
    attestation_path = vocoder_dir / "final_checkpoint_attestation.json"
    # 这份教师加自训声码器的正式推理目录，和学生评估分开。
    inference_dir = (
        # 从已核验的项目根目录展开下方路径。
        root
        # 指向教师加自训声码器的完整 8100 对结果目录。
        / "exp/evaluation/voicegrad30_text_unseen_epoch0500_custom_hifigan"
    )
    # 本次教师结果的四指标目录。
    metrics_dir = inference_dir / "metrics"
    # 正式测试语音清单的现场文件。
    test_manifest_path = root / "data/metadata/test_manifest.csv"
    # 记录 text-unseen 隔离规则的现场 JSON。
    split_protocol_path = root / "data/metadata/split_protocol.json"
    # 训练集均值文件，验证推理没有换用另一份统计量。
    mel_mean_path = root / "data/stats/mel_mean.npy"
    # 训练集标准差文件，必须与生成时完全一致。
    mel_std_path = root / "data/stats/mel_std.npy"
    # 实际 8100 对转换的配对 CSV。
    pairs_path = inference_dir / "pairs.csv"
    # 推理前登记的完整系统和输入身份 JSON。
    run_identity_path = inference_dir / "run_identity.json"
    # 只有完整推理完成才写出的结果报告。
    inference_report_path = inference_dir / "inference_report.json"
    # 四指标计算采用的输入与实现记录。
    metrics_identity_path = metrics_dir / "metrics_run_identity.json"
    # 完整四指标数值汇总位置。
    summary_path = metrics_dir / "objective_summary.json"
    # 与论文比较的结构化结果位置。
    comparison_path = metrics_dir / "paper_comparison.json"

    # 现场计算教师权重的字节摘要，不仅相信文件名。
    teacher_sha = sha256_file(teacher)
    # 核对 teacher SHA-256；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(teacher_sha, EXPECTED_TEACHER_SHA256, "teacher SHA-256")
    # 现场重新计算最终声码器权重摘要。
    checkpoint_sha = sha256_file(checkpoint)
    # 现场重新计算声码器配置原始字节摘要。
    config_sha = sha256_file(config_path)
    # 读取现场完整声码器配置，以核对实际参数。
    config_object = load_object(config_path)
    # 现场测试清单的字节摘要。
    test_manifest_sha = sha256_file(test_manifest_path)
    # 现场划分规则文件摘要。
    split_protocol_sha = sha256_file(split_protocol_path)
    # 现场 mel 均值文件摘要。
    mel_mean_sha = sha256_file(mel_mean_path)
    # 现场 mel 标准差文件摘要。
    mel_std_sha = sha256_file(mel_std_path)

    # 读取原来的声码器最终证明，各关键字段都将与现场重新比较。
    attestation = load_object(attestation_path)
    # 核对 attestation status；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(attestation.get("status"), "verified", "attestation status")
    # 核对 attested step；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(attestation.get("global_step"), EXPECTED_STEP, "attested step")
    # 核对 attested checkpoint path；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(attestation.get("checkpoint"), str(checkpoint), "attested checkpoint path")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的证明登记的生成器检查点路径的原始字节 SHA-256，必须与现场文件一致。
        attestation.get("checkpoint_sha256"), checkpoint_sha, "attested checkpoint SHA-256"
    )
    # 核对 attested config path；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(attestation.get("config"), str(config_path), "attested config path")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的声码器 JSON 配置原始字节摘要。
        attestation.get("config_file_sha256"), config_sha, "attested config SHA-256"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的配置按稳定 JSON 形式编码后的内容摘要。
        attestation.get("config_canonical_sha256"),
        # 用现场完整配置重新计算规范内容摘要。
        canonical_sha256(config_object),
        # 为这次精确比较提供具体错误标签，失败时可定位到 attested canonical config SHA-256。
        "attested canonical config SHA-256",
    )
    # 计算当前声码器实现摘要，与证明逐项比较；不忽略注释导致的变化。
    expected_attestation_implementation = {
        # 逐份源码计算现场字节摘要，以相对路径作为对应键。
        relative_path: sha256_file(root / relative_path)
        # 遍历声码器证明必须锁定的三个官方实现文件。
        for relative_path in (
            # 配置包装实现也纳入指纹，防止构造模型时换用不同配置解释。
            "hifigan/official/env.py",
            # 生成器/判别器结构的实现指纹，哪怕注释变化也会被检测。
            "hifigan/official/models.py",
            # 声码器辅助加载和工具实现同样登记。
            "hifigan/official/utils.py",
        )
    }
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的当前现场实现文件的原始字节指纹；新增注释也会改变它。
        attestation.get("implementation_sha256"),
        # 期望值来自现场三份声码器实现的准确字节摘要。
        expected_attestation_implementation,
        # 为这次精确比较提供具体错误标签，失败时可定位到 attestation implementation SHA-256。
        "attestation implementation SHA-256",
    )
    # 复制证明内容，以便移除自哈希字段后重新计算，不修改磁盘证明。
    attestation_payload = dict(attestation)
    # 取出原记录的自哈希，后续比较其余内容重新算出的摘要。
    attestation_payload_sha = attestation_payload.pop(
        # 取出并移除证明的自哈希字段；缺失时得到 None，随后校验会失败。
        "attestation_payload_sha256", None
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 比较左侧为证明文件原来保存的自哈希。
        attestation_payload_sha,
        # 比较右侧为剔除自哈希字段后重新计算的内容摘要。
        canonical_sha256(attestation_payload),
        # 为这次精确比较提供具体错误标签，失败时可定位到 attestation self-hash。
        "attestation self-hash",
    )

    # 读取正式教师推理完成报告，核对数据、步数、随机种子与权重来源。
    inference = load_object(inference_report_path)
    # 核对 inference pairs；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("selected_pair_count"), EXPECTED_PAIRS, "inference pairs")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 新生成数加经验证复用数必须等于全部配对，支持严格续算。
        int(inference.get("generated", 0)) + int(inference.get("skipped_existing", 0)),
        # 正式完成要求完整的 8100 对，不能把少量试跑标为完成。
        EXPECTED_PAIRS,
        # 为这次精确比较提供具体错误标签，失败时可定位到 generated + resumed inference pairs。
        "generated + resumed inference pairs",
    )
    # 核对 teacher epoch；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("teacher_epoch"), 500, "teacher epoch")
    # 核对 paper pair count；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("paper_pair_count"), EXPECTED_PAIRS, "paper pair count")
    # 核对 data protocol；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("data_protocol"), "text-unseen", "data protocol")
    # 核对 strict protocol；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("strict_text_unseen_protocol"), True, "strict protocol")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的正式测试清单的路径。
        inference.get("test_manifest"),
        # 要求登记路径指向这份实际测试清单。
        str(test_manifest_path),
        # 为这次精确比较提供具体错误标签，失败时可定位到 inference test manifest path。
        "inference test manifest path",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的正式测试清单的路径的原始字节 SHA-256，必须与现场文件一致。
        inference.get("test_manifest_sha256"),
        # 要求摘要与现场测试清单字节完全相同。
        test_manifest_sha,
        # 为这次精确比较提供具体错误标签，失败时可定位到 inference test manifest SHA-256。
        "inference test manifest SHA-256",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的划分规则文件，记录文本和说话人的隔离。
        inference.get("split_protocol"),
        # 要求登记路径指向这份实际划分规则。
        str(split_protocol_path),
        # 为这次精确比较提供具体错误标签，失败时可定位到 inference split protocol path。
        "inference split protocol path",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的划分规则文件，记录文本和说话人的隔离的原始字节 SHA-256，必须与现场文件一致。
        inference.get("split_protocol_sha256"),
        # 要求规则文件摘要与现场版本一致。
        split_protocol_sha,
        # 为这次精确比较提供具体错误标签，失败时可定位到 inference split protocol SHA-256。
        "inference split protocol SHA-256",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的训练集逐 mel 频带均值文件。
        inference.get("mel_mean"), str(mel_mean_path), "inference mel mean path"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的训练集逐 mel 频带均值文件的原始字节 SHA-256，必须与现场文件一致。
        inference.get("mel_mean_sha256"), mel_mean_sha, "inference mel mean SHA-256"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的训练集逐 mel 频带标准差文件。
        inference.get("mel_std"), str(mel_std_path), "inference mel std path"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的训练集逐 mel 频带标准差文件的原始字节 SHA-256，必须与现场文件一致。
        inference.get("mel_std_sha256"), mel_std_sha, "inference mel std SHA-256"
    )
    # 核对 VoiceGrad inference steps；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("voicegrad_steps"), 30, "VoiceGrad inference steps")
    # 核对 teacher start step；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("teacher_start_step"), 950, "teacher start step")
    # 核对 inference base seed；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("base_seed"), 1234, "inference base seed")
    # 核对 inference batch size；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("batch_size"), 4, "inference batch size")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的外部 BNF 到 mel 时间轴采用已登记的 legacy_length_linear。
        inference.get("bnf_alignment"),
        # 这里固定外部长度线性插值的历史接口，不能在正式对照中悄悄切换。
        "legacy_length_linear",
        # 为这次精确比较提供具体错误标签，失败时可定位到 BNF temporal alignment。
        "BNF temporal alignment",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的U-Net 内部 BNF 抽点采用教师原有 legacy_decimate。
        inference.get("bnf_downsampling"),
        # 这里固定教师原有的内部 BNF 抽点方式，未进行接口覆盖。
        "legacy_decimate",
        # 为这次精确比较提供具体错误标签，失败时可定位到 BNF U-Net downsampling。
        "BNF U-Net downsampling",
    )
    # 核对 vocoder kind；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("vocoder_kind"), "custom", "vocoder kind")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的固定第 500 epoch 教师的路径。
        inference.get("teacher_checkpoint"), str(teacher), "inference teacher path"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的固定第 500 epoch 教师的路径的原始字节 SHA-256，必须与现场文件一致。
        inference.get("teacher_checkpoint_sha256"), teacher_sha, "inference teacher SHA-256"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的最终 250 万步自训生成器文件。
        inference.get("vocoder_checkpoint"),
        # 要求登记的声码器路径就是本次最终 250 万步权重。
        str(checkpoint),
        # 为这次精确比较提供具体错误标签，失败时可定位到 inference vocoder path。
        "inference vocoder path",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的最终 250 万步自训生成器文件的原始字节 SHA-256，必须与现场文件一致。
        inference.get("vocoder_checkpoint_sha256"),
        # 要求声码器文件摘要与现场权重一致。
        checkpoint_sha,
        # 为这次精确比较提供具体错误标签，失败时可定位到 inference vocoder SHA-256。
        "inference vocoder SHA-256",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的该生成器对应的完整配置。
        inference.get("vocoder_config"), str(config_path), "inference config path"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的该生成器对应的完整配置的原始字节 SHA-256，必须与现场文件一致。
        inference.get("vocoder_config_sha256"), config_sha, "inference config SHA-256"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的推理时实际采用的完整声码器配置。
        inference.get("vocoder_runtime_config"),
        # 完整比较实际声码器参数，不能只比较几个代表性字段。
        config_object,
        # 为这次精确比较提供具体错误标签，失败时可定位到 inference runtime config。
        "inference runtime config",
    )
    # 核对 BNF override；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("bnf_downsampling_override"), False, "BNF override")

    # 读取推理实际使用的 pairs.csv，不临时重新选择一套配对。
    with pairs_path.open("r", encoding="utf-8", newline="") as handle:
        # 读取实际配对清单的全部行，随后精确检查数量和编号顺序。
        pair_rows = list(csv.DictReader(handle))
    # 核对 pairs.csv row count；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(len(pair_rows), EXPECTED_PAIRS, "pairs.csv row count")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 取得当前清单记录的真实配对顺序。
        [int(row["pair_id"]) for row in pair_rows],
        # 期望配对编号按顺序为 0 到 8099，不能缺失、重复或乱序。
        list(range(EXPECTED_PAIRS)),
        # 为这次精确比较提供具体错误标签，失败时可定位到 pairs.csv IDs/order。
        "pairs.csv IDs/order",
    )
    # 保存实际配对清单的原始字节摘要。
    pairs_sha = sha256_file(pairs_path)
    # 核对 inference pairs path；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(inference.get("pairs_csv"), str(pairs_path), "inference pairs path")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的完整配对清单的准确路径的原始字节 SHA-256，必须与现场文件一致。
        inference.get("pairs_csv_sha256"), pairs_sha, "inference pairs SHA-256"
    )

    # 读取推理开始时固定的系统与数据身份。
    run_identity = load_object(run_identity_path)
    # 计算推理身份 JSON 文件摘要，后续要求推理报告与指标汇总共同指向它。
    run_identity_sha = sha256_file(run_identity_path)
    # 现场计算全部核心推理源码的摘要，不只核对模型权重。
    expected_inference_implementation = {
        # 逐份源码计算现场字节摘要，以相对路径作为对应键。
        relative_path: sha256_file(root / relative_path)
        # 完整遍历文件顶部登记的核心推理源码列表。
        for relative_path in INFERENCE_IMPLEMENTATION_PATHS
    }
    # 根据现场文件和这份正式教师协议重建期望身份，要求与保存记录完全一致。
    expected_run_identity = {
        # 来源记录的格式版本，避免将不同结构的旧记录直接比较。
        "identity_format_version": 2,
        # 活动数据划分，必须为 text-unseen。
        "data_protocol": "text-unseen",
        # 正式测试清单的路径。
        "test_manifest": str(test_manifest_path),
        # 正式测试清单的路径的原始字节 SHA-256，必须与现场文件一致。
        "test_manifest_sha256": test_manifest_sha,
        # 划分规则文件，记录文本和说话人的隔离。
        "split_protocol": str(split_protocol_path),
        # 划分规则文件，记录文本和说话人的隔离的原始字节 SHA-256，必须与现场文件一致。
        "split_protocol_sha256": split_protocol_sha,
        # 训练集逐 mel 频带均值文件。
        "mel_mean": str(mel_mean_path),
        # 训练集逐 mel 频带均值文件的原始字节 SHA-256，必须与现场文件一致。
        "mel_mean_sha256": mel_mean_sha,
        # 训练集逐 mel 频带标准差文件。
        "mel_std": str(mel_std_path),
        # 训练集逐 mel 频带标准差文件的原始字节 SHA-256，必须与现场文件一致。
        "mel_std_sha256": mel_std_sha,
        # 完整配对清单的准确路径的原始字节 SHA-256，必须与现场文件一致。
        "pairs_csv_sha256": pairs_sha,
        # 论文协议重建所需的 8100 对。
        "paper_pair_count": EXPECTED_PAIRS,
        # 本次实际选择的配对数，正式流程不能只跑子集。
        "selected_pair_count": EXPECTED_PAIRS,
        # 用户明确请求的教师轮数 500。
        "teacher_epoch_requested": 500,
        # 固定第 500 epoch 教师的路径。
        "teacher_checkpoint": str(teacher),
        # 固定第 500 epoch 教师的路径的原始字节 SHA-256，必须与现场文件一致。
        "teacher_checkpoint_sha256": teacher_sha,
        # 声码器来源为自训 custom，不能混成公开权重。
        "vocoder_kind": "custom",
        # 最终 250 万步自训生成器文件。
        "vocoder_checkpoint": str(checkpoint),
        # 最终 250 万步自训生成器文件的原始字节 SHA-256，必须与现场文件一致。
        "vocoder_checkpoint_sha256": checkpoint_sha,
        # 该生成器对应的完整配置。
        "vocoder_config": str(config_path),
        # 该生成器对应的完整配置的原始字节 SHA-256，必须与现场文件一致。
        "vocoder_config_sha256": config_sha,
        # 教师反向采样次数固定为 30。
        "voicegrad_steps": 30,
        # 源 mel 的起始加噪时间步固定为 950。
        "teacher_start_step": 950,
        # 基础随机种子固定为 1234。
        "seed": 1234,
        # 推理每批固定 4 个配对。
        "batch_size": 4,
        # 正式流程不允许设置只取部分配对的上限。
        "selected_pair_limit": None,
        # 外部 BNF 到 mel 时间轴采用已登记的 legacy_length_linear。
        "bnf_alignment": "legacy_length_linear",
        # U-Net 内部 BNF 抽点采用教师原有 legacy_decimate。
        "bnf_downsampling": "legacy_decimate",
        # 正式教师评估没有覆盖教师训练时的内部接口。
        "bnf_downsampling_override": False,
        # 当前现场实现文件的原始字节指纹；新增注释也会改变它。
        "implementation_sha256": expected_inference_implementation,
    }
    # 核对 inference run identity；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(run_identity, expected_run_identity, "inference run identity")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的推理模型、数据与设置的整体身份记录。
        inference.get("run_identity"), str(run_identity_path), "run identity path"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的推理模型、数据与设置的整体身份记录的原始字节 SHA-256，必须与现场文件一致。
        inference.get("run_identity_sha256"), run_identity_sha, "run identity SHA-256"
    )

    # 只统计实际普通 WAV 文件数量，要求 8100 条全部存在。
    wav_count = sum(1 for path in (inference_dir / "wav").glob("*.wav") if path.is_file())
    # 核对 WAV file count；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(wav_count, EXPECTED_PAIRS, "WAV file count")

    # 集中记录每份逐句指标 CSV 的准确字节摘要。
    metric_hashes: Dict[str, str] = {}
    # 四项指标分别检查，不能只算一部分就宣布完整评估。
    for metric in METRIC_NAMES:
        # 定位当前一项客观指标的逐句 CSV 文件。
        metric_path = metrics_dir / f"{metric}.csv"
        # 核对当前指标恰好覆盖 8100 个唯一配对编号。
        validate_metric_csv(metric_path)
        # 锁定这一份逐句结果文件，汇总必须指向同一字节版本。
        metric_hashes[metric] = sha256_file(metric_path)

    # 读取指标计算的运行身份，确认没有串用别次推理。
    metrics_identity = load_object(metrics_identity_path)
    # 用现场推理报告、配对表和评价源码重建期望指标身份。
    expected_metrics_identity = {
        # 来源记录的格式版本，避免将不同结构的旧记录直接比较。
        "identity_format_version": 1,
        # 已经完成推理的报告文件。
        "inference_report": str(inference_report_path),
        # 已经完成推理的报告文件的原始字节 SHA-256，必须与现场文件一致。
        "inference_report_sha256": sha256_file(inference_report_path),
        # 完整配对清单的准确路径。
        "pairs_csv": str(pairs_path),
        # 完整配对清单的准确路径的原始字节 SHA-256，必须与现场文件一致。
        "pairs_csv_sha256": pairs_sha,
        # 本次实际选择的配对数，正式流程不能只跑子集。
        "selected_pair_count": EXPECTED_PAIRS,
        # 正式评估从第 0 个配对开始。
        "start": 0,
        # 正式评估没有只算部分配对的数量限制。
        "limit": None,
        # 不允许将非正式少量配对诊断当作完整评估。
        "diagnostic_allow_nonpaper_pairs": False,
        # 当前现场实现文件的原始字节指纹；新增注释也会改变它。
        "implementation_sha256": sha256_file(root / "evaluation/evaluate_metrics.py"),
    }
    # 核对 metrics run identity；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(metrics_identity, expected_metrics_identity, "metrics run identity")
    # 保留已核验指标身份文件摘要，写进最终完成记录。
    metrics_identity_sha = sha256_file(metrics_identity_path)

    # 读取四指标汇总，再核对它引用的模型、配对和逐句文件。
    summary = load_object(summary_path)
    # 核对 summary status；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(summary.get("status"), "formal_8100_complete", "summary status")
    # 核对 summary pairs；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(summary.get("evaluated_pairs"), EXPECTED_PAIRS, "summary pairs")
    # 核对 summary teacher epoch；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(summary.get("teacher_epoch"), 500, "summary teacher epoch")
    # 核对 summary vocoder kind；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(summary.get("vocoder_kind"), "custom", "summary vocoder kind")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的最终 250 万步自训生成器文件。
        summary.get("vocoder_checkpoint"), str(checkpoint), "summary vocoder path"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的最终 250 万步自训生成器文件的原始字节 SHA-256，必须与现场文件一致。
        summary.get("vocoder_checkpoint_sha256"), checkpoint_sha, "summary vocoder SHA-256"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的该生成器对应的完整配置的原始字节 SHA-256，必须与现场文件一致。
        summary.get("vocoder_config_sha256"), config_sha, "summary config SHA-256"
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的推理模型、数据与设置的整体身份记录的原始字节 SHA-256，必须与现场文件一致。
        summary.get("run_identity_sha256"), run_identity_sha, "summary run identity"
    )
    # 核对 summary pairs path；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(summary.get("pairs_csv"), str(pairs_path), "summary pairs path")
    # 核对 summary pairs SHA-256；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(summary.get("pairs_csv_sha256"), pairs_sha, "summary pairs SHA-256")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的已经完成推理的报告文件。
        summary.get("inference_report"),
        # 汇总必须指向当前已完成的推理报告。
        str(inference_report_path),
        # 为这次精确比较提供具体错误标签，失败时可定位到 summary inference path。
        "summary inference path",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的已经完成推理的报告文件的原始字节 SHA-256，必须与现场文件一致。
        summary.get("inference_report_sha256"),
        # 重新核对推理报告字节版本，维持来源对应关系。
        sha256_file(inference_report_path),
        # 为这次精确比较提供具体错误标签，失败时可定位到 summary inference lineage。
        "summary inference lineage",
    )
    # 要求汇总中的逐指标摘要是完整映射，不能缺省后跳过核对。
    summary_metric_hashes = require_mapping(
        # 读取记录中的四份逐句指标 CSV 的独立文件摘要。
        summary.get("individual_metric_csv_sha256"),
        # 为这次精确比较提供具体错误标签，失败时可定位到 summary individual metric CSV hashes。
        "summary individual metric CSV hashes",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 四项逐句 CSV 的摘要必须与汇总保存的逐项一致，不能替换文件后沿用旧平均值。
        dict(summary_metric_hashes), metric_hashes, "summary metric CSV SHA-256 lineage"
    )

    # 读取论文比较记录，核实它确实来自本次完整客观汇总。
    comparison = load_object(comparison_path)
    # 核对 comparison status；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(comparison.get("status"), "paper_comparison_complete", "comparison status")
    # 核对 comparison vocoder kind；必须与下方明确给出的正式设置或现场指纹一致。
    require_equal(comparison.get("vocoder_kind"), "custom", "comparison vocoder kind")
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的最终 250 万步自训生成器文件的原始字节 SHA-256，必须与现场文件一致。
        comparison.get("vocoder_checkpoint_sha256"),
        # 要求声码器文件摘要与现场权重一致。
        checkpoint_sha,
        # 为这次精确比较提供具体错误标签，失败时可定位到 comparison vocoder SHA-256。
        "comparison vocoder SHA-256",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的该生成器对应的完整配置的原始字节 SHA-256，必须与现场文件一致。
        comparison.get("vocoder_config_sha256"),
        # 要求声码器配置文件字节摘要一致。
        config_sha,
        # 为这次精确比较提供具体错误标签，失败时可定位到 comparison config SHA-256。
        "comparison config SHA-256",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的论文比较实际读取的那份四指标汇总。
        comparison.get("source_objective_summary"),
        # 论文比较必须来自这一份已经核验的指标汇总。
        str(summary_path),
        # 为这次精确比较提供具体错误标签，失败时可定位到 comparison summary path。
        "comparison summary path",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的论文比较实际读取的那份四指标汇总的原始字节 SHA-256，必须与现场文件一致。
        comparison.get("source_objective_summary_sha256"),
        # 重新计算汇总文件摘要，防止比较表引用旧数值。
        sha256_file(summary_path),
        # 为这次精确比较提供具体错误标签，失败时可定位到 comparison summary lineage。
        "comparison summary lineage",
    )
    # 逐项比较下面给出的实际记录与现场期望；第三个参数用于指出具体不一致位置。
    require_equal(
        # 读取记录中的保留与论文评价环境不完全相同的事实，不得标成已经证明条件完全一致。
        comparison.get("strictly_same_conditions"), False, "comparison honesty flag"
    )

    # 全部检查通过后才组织最终完成记录，绑定模型、数据、源码和结果。
    completion = {
        # 阶段完成状态，必须与本函数要求的正式完成标记一致。
        "status": "custom_hifigan_teacher_paper_comparison_complete",
        # 固定第 500 epoch 教师的路径。
        "teacher_checkpoint": str(teacher),
        # 固定第 500 epoch 教师的路径的原始字节 SHA-256，必须与现场文件一致。
        "teacher_checkpoint_sha256": teacher_sha,
        # 这次自训声码器权重路径。
        "custom_vocoder_checkpoint": str(checkpoint),
        # 这次自训声码器权重路径的原始字节 SHA-256，必须与现场文件一致。
        "custom_vocoder_checkpoint_sha256": checkpoint_sha,
        # 自训声码器内部记录的最终更新步数 2500000。
        "custom_vocoder_global_step": EXPECTED_STEP,
        # 自训声码器配套配置文件。
        "custom_vocoder_config": str(config_path),
        # 自训声码器配套配置文件的原始字节 SHA-256，必须与现场文件一致。
        "custom_vocoder_config_sha256": config_sha,
        # 该声码器的最终验证证明文件。
        "checkpoint_attestation": str(attestation_path),
        # 该声码器的最终验证证明文件的原始字节 SHA-256，必须与现场文件一致。
        "checkpoint_attestation_sha256": sha256_file(attestation_path),
        # 推理模型、数据与设置的整体身份记录。
        "run_identity": str(run_identity_path),
        # 推理模型、数据与设置的整体身份记录的原始字节 SHA-256，必须与现场文件一致。
        "run_identity_sha256": run_identity_sha,
        # 已经完成推理的报告文件。
        "inference_report": str(inference_report_path),
        # 已经完成推理的报告文件的原始字节 SHA-256，必须与现场文件一致。
        "inference_report_sha256": sha256_file(inference_report_path),
        # 完整配对清单的准确路径。
        "pairs_csv": str(pairs_path),
        # 完整配对清单的准确路径的原始字节 SHA-256，必须与现场文件一致。
        "pairs_csv_sha256": pairs_sha,
        # 正式测试清单的路径。
        "test_manifest": str(test_manifest_path),
        # 正式测试清单的路径的原始字节 SHA-256，必须与现场文件一致。
        "test_manifest_sha256": test_manifest_sha,
        # 划分规则文件，记录文本和说话人的隔离。
        "split_protocol": str(split_protocol_path),
        # 划分规则文件，记录文本和说话人的隔离的原始字节 SHA-256，必须与现场文件一致。
        "split_protocol_sha256": split_protocol_sha,
        # 训练集逐 mel 频带均值文件。
        "mel_mean": str(mel_mean_path),
        # 训练集逐 mel 频带均值文件的原始字节 SHA-256，必须与现场文件一致。
        "mel_mean_sha256": mel_mean_sha,
        # 训练集逐 mel 频带标准差文件。
        "mel_std": str(mel_std_path),
        # 训练集逐 mel 频带标准差文件的原始字节 SHA-256，必须与现场文件一致。
        "mel_std_sha256": mel_std_sha,
        # 本次客观指标输入与实现的身份记录。
        "metrics_run_identity": str(metrics_identity_path),
        # 本次客观指标输入与实现的身份记录的原始字节 SHA-256，必须与现场文件一致。
        "metrics_run_identity_sha256": metrics_identity_sha,
        # 四份逐句指标 CSV 的独立文件摘要。
        "individual_metric_csv_sha256": metric_hashes,
        # 四项客观指标的完整汇总文件。
        "objective_summary": str(summary_path),
        # 四项客观指标的完整汇总文件的原始字节 SHA-256，必须与现场文件一致。
        "objective_summary_sha256": sha256_file(summary_path),
        # 汇总结果与论文表格的比较记录。
        "paper_comparison": str(comparison_path),
        # 汇总结果与论文表格的比较记录的原始字节 SHA-256，必须与现场文件一致。
        "paper_comparison_sha256": sha256_file(comparison_path),
        # 直接从已核验汇总读取的主要指标，不在此重新挑选结果。
        "headline_metrics": summary.get("headline_metrics"),
    }
    # 全部证据通过后才发布完整通过报告，不更新任何模型、配对或原始证明。
    atomic_write_json(args.output_report.resolve(), completion)
    # 日志输出同一份已核验的完成记录，便于远程查看。
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    # 最后打印明确通过标记，前面任一异常都会阻止执行到这里。
    print("CUSTOM_HIFIGAN_TEACHER_PIPELINE_VERIFIED_COMPLETE")


# 只有直接运行该审计程序才开始核验；导入模块不自动评估。
if __name__ == "__main__":
    # 进入上面定义的全链路只读核验与报告流程。
    main()
