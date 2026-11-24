# 不需要真实音频的双协议单元测试。
"""不需要真实音频的双协议单元测试。

本测试故意只建立“假的路径清单”，因为 02 脚本负责的是数据身份划分，不会打开 WAV。
这样普通电脑几秒内就能验证最危险的逻辑：测试人是否泄漏、001--010 是否从
text-unseen 训练集删除、又是否在 text-seen 训练集保留。

运行方法：

    python -m unittest tests.test_split_protocols

``unittest`` 与 ``tempfile`` 都来自 Python 标准库，不需要额外安装 pytest。
"""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# csv/json 写和读取测试清单；subprocess 用“真实命令行”执行待测脚本。
import csv
# 读写实验身份、指纹和报告的 JSON。
import json
# 计算文件或 JSON 的 SHA-256，检测内容被替换。
import hashlib
# 在临时目录复制或恢复文件，模拟统计量发布事务。
import shutil
# 在租用服务器中调用生产脚本的真实命令行入口。
import subprocess
# 沿用当前服务器 Python 解释器启动测试子进程。
import sys
# 将故意损坏的测试文件隔离在自动清理目录中。
import tempfile
# 使用标准库测试框架组织断言和失败报告。
import unittest
# Path 管理临时目录和仓库路径。
from pathlib import Path
# 导入本组测试实际调用的 typing 接口；测试直接约束生产实现。
from typing import Dict, Iterable, List

# NumPy 只用来生成极小的假特征，验证形状检查；不进行任何模型计算。
import numpy as np

# 导入本组测试实际调用的 voicegrad 接口；测试直接约束生产实现。
from voicegrad import validate_setup as validator


# 通过测试文件位置确定的仓库根目录。
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# 生产文本 seen/unseen 划分脚本位置。
SPLIT_SCRIPT = REPOSITORY_ROOT / "preprocess" / "02_build_fastvoicegrad_split.py"
# 生产 VCTK 库存构建脚本位置。
INVENTORY_SCRIPT = REPOSITORY_ROOT / "preprocess" / "00_build_vctk_manifest.py"
# 生产训练前完整验证模块名。
VALIDATE_MODULE = "voicegrad.validate_setup"
# 人工测试库存采用的十个测试说话人 ID。
TEST_SPEAKERS = (
    # 将 p238 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p238",
    # 将 p241 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p241",
    # 将 p243 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p243",
    # 将 p252 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p252",
    # 将 p261 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p261",
    # 将 p294 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p294",
    # 将 p303 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p303",
    # 将 p330 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p330",
    # 将 p334 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p334",
    # 将 p360 纳入固定十人测试集合，训练清单必须排除这个身份。
    "p360",
)


# 写入人工 CSV 清单，严格保留指定字段和行顺序。
def write_csv(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    # 写测试 CSV；with 结束时文件会自动关闭。
    """写测试 CSV；with 结束时文件会自动关闭。"""

    # 按指定编码打开测试文件，作用域结束后自动关闭。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 按给定字段顺序写入 CSV 的记录器。
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        # 先写与正式清单相同的列名。
        writer.writeheader()
        # 按固定字段顺序写人工样本清单。
        writer.writerows(rows)


# 建立 110 人各 12 句的人工库存，包含无转写的 p315 和特殊 ID s5。
def build_synthetic_inventory(project_root: Path) -> None:
    # 建立 110 人 × 12 句的最小完整库存，路径存在性不属于本单元测试范围。
    """建立 110 人 × 12 句的最小完整库存，路径存在性不属于本单元测试范围。"""

    # 人工项目的训练、测试、总库存与协议清单都位于此元数据目录。
    metadata = project_root / "data" / "metadata"
    # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
    metadata.mkdir(parents=True)

    # 官方 0.92 的 110 人包含特殊 ID s5；p315 有音频但无转写。
    # 其余 98 个训练人用 p400--p497，使总数仍为 100 train + 10 test。
    train_speakers = ("p315", "s5") + tuple(
        # 用 p400 到 p497 构造 98 个训练人，连同 p315 和 s5 共 100 人。
        f"p{number}" for number in range(400, 498)
    )
    # 将一百训练人和十测试人拼成的完整库存。
    speakers = train_speakers + TEST_SPEAKERS
    # 重采样 WAV 清单记录。
    wav_rows: List[Dict[str, object]] = []
    # 原始音频、文本、麦克风和说话人配对记录。
    pair_rows: List[Dict[str, object]] = []

    # 逐项处理 speaker_index, speaker in enumerate(speakers)，覆盖这组反例的全部输入。
    for speaker_index, speaker in enumerate(speakers):
        # 按人工说话人序号交替设置女、男标签，覆盖两种库存解析分支。
        gender = "F" if speaker_index % 2 == 0 else "M"
        # 逐项处理 number in range(1, 13)，覆盖这组反例的全部输入。
        for number in range(1, 13):
            # 三位数格式的句子槽位。
            slot = f"{number:03d}"
            # 用说话人 ID 加三位数句子槽位组成语音唯一 ID。
            utterance_id = f"{speaker}_{slot}"
            # 001--010 在所有说话人之间文本完全相同；011--012 加入说话人 ID。
            text = (
                # 各人前十句使用相同文本，便于验证文本是否泄漏。
                f"公共测试句 {slot}"
                # 前十句设置为所有说话人共享的测试文本。
                if number <= 10
                # 第十一、十二句带说话人 ID，作为互不重复的训练文本。
                else f"{speaker} 的私有训练句 {slot}"
            )
            # 测试音频文件路径。
            wav_path = project_root / "data" / "wav22k" / speaker / f"{utterance_id}.wav"
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            wav_path.write_bytes(f"synthetic-{utterance_id}".encode("utf-8"))
            # 登记当前人工语音的重采样 WAV 路径。
            wav_rows.append(
                {
                    # 语音所属说话人的 ID。
                    "speaker": speaker,
                    # 语音的唯一身份 ID。
                    "utt_id": utterance_id,
                    # 22050 Hz 源音频路径。
                    "wav22k_path": str(wav_path),
                }
            )
            # 登记原始音频、文本和说话人信息。
            pair_rows.append(
                {
                    # 语音所属说话人的 ID。
                    "speaker": speaker,
                    # 语音的唯一身份 ID。
                    "utt_id": utterance_id,
                    # 说话人性别标签，用于库存完整性检查；此处为测试使用的明确值。
                    "gender": gender,
                    # 原始 VCTK 音频路径；此处为测试使用的明确值。
                    "wav48_path": f"/synthetic/flac/{speaker}/{utterance_id}_mic1.flac",
                    # 原始文字转写路径；缺文本者留空；此处为测试使用的明确值。
                    "txt_path": (
                        # p315 缺少转写路径，明确保留空值而不编造文本。
                        ""
                        # p315 人工模拟官方语料有音频但缺文本的情况。
                        if speaker == "p315"
                        # 其他说话人登记正常转写文件路径。
                        else f"/synthetic/txt/{speaker}/{utterance_id}.txt"
                    ),
                    # 语音转写文本或文本模式标记，具体取决于当前调用；此处为测试使用的明确值。
                    "text": "" if speaker == "p315" else text,
                    # 原始音频采用的 VCTK 目录布局；此处为测试使用的明确值。
                    "audio_layout": "wav48_silence_trimmed",
                    # 选择的录音麦克风通道；此处为测试使用的明确值。
                    "microphone": "mic1",
                }
            )

    # 把人工库存写成生产划分器实际读取的 CSV。
    write_csv(
        # 重采样语音清单的生产约定路径。
        metadata / "vctk_wav22k_manifest.csv",
        # 重采样清单依次给出说话人、语音 ID 和 WAV 路径。
        ["speaker", "utt_id", "wav22k_path"],
        # 传入重采样 WAV 清单记录。
        wav_rows,
    )
    # 把人工库存写成生产划分器实际读取的 CSV。
    write_csv(
        # 原始 VCTK 音频与文本配对清单位置。
        metadata / "vctk_all_pairs.csv",
        [
            # 语音所属说话人的 ID。
            "speaker",
            # 语音的唯一身份 ID。
            "utt_id",
            # 说话人性别标签，用于库存完整性检查。
            "gender",
            # 原始 VCTK 音频路径。
            "wav48_path",
            # 原始文字转写路径；缺文本者留空。
            "txt_path",
            # 语音转写文本或文本模式标记，具体取决于当前调用。
            "text",
            # 原始音频采用的 VCTK 目录布局。
            "audio_layout",
            # 选择的录音麦克风通道。
            "microphone",
        ],
        # 传入原始音频、文本、麦克风和说话人配对记录。
        pair_rows,
    )


# 按活动清单构建最小合法梅尔、BNF、说话人向量及其来源证明。
def build_synthetic_features(project_root: Path) -> None:
    # 按活动 all_manifest 为每条语音建立合法但极小的 mel/BNF/说话人特征。
    """按活动 all_manifest 为每条语音建立合法但极小的 mel/BNF/说话人特征。"""

    # 人工项目的训练、测试、总库存与协议清单都位于此元数据目录。
    metadata = project_root / "data" / "metadata"
    # 按指定编码打开测试文件，作用域结束后自动关闭。
    with (metadata / "all_manifest.csv").open(
        # 只读 UTF-8 CSV，并由 csv 模块统一处理换行。
        "r", encoding="utf-8", newline=""
    # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
    ) as handle:
        # 从活动全量清单读出人工训练与测试语音，为每条记录建立特征。
        rows = list(csv.DictReader(handle))
    # 按 BNF 与说话人向量分类保存的逐文件证明。
    feature_records = {"bnf": {}, "speaker_embedding": {}}
    # 逐项处理 row in rows，覆盖这组反例的全部输入。
    for row in rows:
        # 从当前人工库存行读取说话人 ID，用于组织对应特征路径。
        speaker = row["speaker"]
        # 从当前库存行读取语音 ID，保证特征路径与清单身份一致。
        utterance_id = row["utt_id"]
        # 梅尔、BNF 和说话人向量三种小数组。
        arrays = {
            # 80 通道、短帧长的人工梅尔数组；此处为测试使用的明确值。
            "mel": np.zeros((80, 2), dtype=np.float32),
            # 144 通道的人工内容特征数组；此处为测试使用的明确值。
            "bnf": np.zeros((144, 2), dtype=np.float32),
            # 256 维说话人特征数组；此处为测试使用的明确值。
            "spk_emb": np.ones((256,), dtype=np.float32),
        }
        # 逐项处理 feature_name, array in arrays.items()，覆盖这组反例的全部输入。
        for feature_name, array in arrays.items():
            # 按当前特征类型、说话人和语音 ID 拼接 .npy 输出路径。
            destination = (
                # 按特征类型/说话人/语音 ID 组织人工特征路径。
                project_root / "data" / feature_name / speaker / f"{utterance_id}.npy"
            )
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            destination.parent.mkdir(parents=True, exist_ok=True)
            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(destination, array, allow_pickle=False)
            # BNF 和说话人向量额外需要模型与源音频来源证明。
            if feature_name in {"bnf", "spk_emb"}:
                # 当前处理的 BNF 或说话人特征类型。
                kind = "bnf" if feature_name == "bnf" else "speaker_embedding"
                # 用说话人/语音.npy 作为特征集合中的稳定身份键。
                key = f"{speaker}/{utterance_id}.npy"
                # 以特征类型和语音键登记当前数组的完整证明。
                feature_records[kind][key] = {
                    # 不受 NumPy 文件头影响的数组内容指纹；此处为测试使用的明确值。
                    "array_sha256": validator.sha256_array(array),
                    # 文件实际字节数；此处为测试使用的明确值。
                    "bytes": destination.stat().st_size,
                    # 数组维度，区分通道与时间；此处为测试使用的明确值。
                    "shape": list(array.shape),
                    # 数组数值存储类型；此处为测试使用的明确值。
                    "dtype": array.dtype.str,
                    # 当前提取运行对该文件的处理动作；此处为测试使用的明确值。
                    "run_action": "generated",
                    # 生成特征所用源音频的内容指纹；此处为测试使用的明确值。
                    "input_wav_sha256": validator.sha256_file(Path(row["wav22k_path"])),
                    # 源音频的字节长度；此处为测试使用的明确值。
                    "input_wav_bytes": Path(row["wav22k_path"]).stat().st_size,
                }
    # 人工项目的统计量目录。
    stats = project_root / "data" / "stats"
    # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
    stats.mkdir(parents=True, exist_ok=True)
    # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
    np.save(stats / "mel_mean.npy", np.zeros(80, dtype=np.float32), allow_pickle=False)
    # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
    np.save(stats / "mel_std.npy", np.ones(80, dtype=np.float32), allow_pickle=False)
    # validator 不只看数组，还要求报告证明统计量来自“当前训练清单”的全部行。
    with (metadata / "train_manifest.csv").open(
        # 只读 UTF-8 CSV，并由 csv 模块统一处理换行。
        "r", encoding="utf-8", newline=""
    # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
    ) as handle:
        # 读回当前协议实际生成的训练记录列表，随后检查人数和句子范围。
        train_rows = list(csv.DictReader(handle))
    # 人工项目中的提取与统计证明日志目录。
    logs = project_root / "exp" / "logs"
    # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
    logs.mkdir(parents=True, exist_ok=True)
    # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
    (logs / "mel_extraction_report_22050.json").write_text(
        # 按所列 JSON 编码参数写测试结构，保证后续指纹计算一致。
        json.dumps(
            {
                # 此次提取所覆盖的清单身份；此处为测试使用的明确值。
                "manifest": str(metadata / "all_manifest.csv"),
                # 清单总条数；此处为测试使用的明确值。
                "total_rows": len(rows),
                # 成功条数；此处为测试使用的明确值。
                "ok": len(rows),
                # 跳过条数；此处为测试使用的明确值。
                "skip": 0,
                # 失败条数；此处为测试使用的明确值。
                "fail": 0,
                # 梅尔提取的采样率；此处为测试使用的明确值。
                "sr": 22050,
                # 傅里叶变换窗口点数；此处为测试使用的明确值。
                "n_fft": 1024,
                # 相邻梅尔帧的采样点间隔；此处为测试使用的明确值。
                "hop_size": 256,
                # 频谱分析窗长度；此处为测试使用的明确值。
                "win_size": 1024,
                # 梅尔频率通道数；此处为测试使用的明确值。
                "n_mels": 80,
                # 梅尔滤波最低频率；此处为测试使用的明确值。
                "fmin": 0.0,
                # 梅尔滤波最高频率；此处为测试使用的明确值。
                "fmax": 8000.0,
                # 是否先裁齐到整数个梅尔帧移；此处为测试使用的明确值。
                "truncate_waveform_to_hop_multiple": True,
            }
        ),
        # 文件文本编码，保证中文和摘要可稳定读写，按下列值传给生产函数。
        encoding="utf-8",
    )

    # 按稳定语音键计算特征及输入 WAV 的内容和布局四类集合指纹。
    def summarize(records):
        # 逐特征内容集合哈希。
        content = hashlib.sha256()
        # 逐特征路径和大小集合哈希。
        layout = hashlib.sha256()
        # 逐源 WAV 内容集合哈希。
        input_content = hashlib.sha256()
        # 逐源 WAV 路径和大小集合哈希。
        input_layout = hashlib.sha256()
        # 逐项处理 row in rows，覆盖这组反例的全部输入。
        for row in rows:
            # 按清单当前语音生成同一套特征键，保证逐文件哈希顺序一致。
            key = f"{row['speaker']}/{row['utt_id']}.npy"
            # 当前语音的特征来源记录。
            record = records[key]
            # 向逐特征内容集合哈希追加当前语音身份键。
            content.update(key.encode("utf-8"))
            # 向逐特征内容集合哈希追加键值分隔符，防止拼接歧义。
            content.update(b"\0")
            # 向逐特征内容集合哈希追加当前特征数组内容指纹。
            content.update(record["array_sha256"].encode("ascii"))
            # 向逐特征内容集合哈希追加记录结束换行，使逐文件边界明确。
            content.update(b"\n")
            # 向逐特征路径和大小集合哈希追加当前语音身份键。
            layout.update(key.encode("utf-8"))
            # 向逐特征路径和大小集合哈希追加键值分隔符，防止拼接歧义。
            layout.update(b"\0")
            # 向逐特征路径和大小集合哈希追加当前文件字节长度。
            layout.update(str(record["bytes"]).encode("ascii"))
            # 向逐特征路径和大小集合哈希追加记录结束换行，使逐文件边界明确。
            layout.update(b"\n")
            # 向逐源 WAV 内容集合哈希追加当前语音身份键。
            input_content.update(key.encode("utf-8"))
            # 向逐源 WAV 内容集合哈希追加键值分隔符，防止拼接歧义。
            input_content.update(b"\0")
            # 向逐源 WAV 内容集合哈希追加当前源 WAV 内容指纹。
            input_content.update(record["input_wav_sha256"].encode("ascii"))
            # 向逐源 WAV 内容集合哈希追加记录结束换行，使逐文件边界明确。
            input_content.update(b"\n")
            # 向逐源 WAV 路径和大小集合哈希追加当前语音身份键。
            input_layout.update(key.encode("utf-8"))
            # 向逐源 WAV 路径和大小集合哈希追加键值分隔符，防止拼接歧义。
            input_layout.update(b"\0")
            # 向逐源 WAV 路径和大小集合哈希追加当前文件字节长度。
            input_layout.update(str(record["input_wav_bytes"]).encode("ascii"))
            # 向逐源 WAV 路径和大小集合哈希追加记录结束换行，使逐文件边界明确。
            input_layout.update(b"\n")
        # 按本辅助函数的接口构造最小有效测试字典。
        return {
            # 记录覆盖数量；此处为测试使用的明确值。
            "count": len(rows),
            # 按样本键累计的内容集合指纹；此处为测试使用的明确值。
            "content_sha256": content.hexdigest(),
            # 按样本键和大小累计的布局指纹；此处为测试使用的明确值。
            "layout_sha256": layout.hexdigest(),
            # 对应源 WAV 内容集合指纹；此处为测试使用的明确值。
            "input_content_sha256": input_content.hexdigest(),
            # 对应源 WAV 路径和大小集合指纹；此处为测试使用的明确值。
            "input_layout_sha256": input_layout.hexdigest(),
        }

    # 合并训练和测试的活动清单。
    all_manifest = metadata / "all_manifest.csv"
    # 供两类特征共享的清单身份摘要。
    common_manifest = {
        # 当前需要核验或写入的具体文件路径；此处为测试使用的明确值。
        "path": str(all_manifest),
        # 当前内容的 SHA-256 指纹；此处为测试使用的明确值。
        "sha256": validator.sha256_file(all_manifest),
        # 样本身份键集合指纹；此处为测试使用的明确值。
        "keys_sha256": validator.manifest_keys_sha256(rows),
        # 登记当前活动清单的实际记录数量，随 seen/unseen 协议而变化。
        "rows": len(rows),
    }
    # 逐项处理 kind, report_name, expected_field, expected_dimension in ，覆盖这组反例的全部输入。
    for kind, report_name, expected_field, expected_dimension in (
        # BNF 报告必须声明 144 个内容通道。
        ("bnf", "bnf_extraction_report.json", "expected_bnf_dim", 144),
        (
            # 说话人嵌入特征在来源报告中的类型名。
            "speaker_embedding",
            # 说话人向量提取证明的固定文件名。
            "spk_emb_extraction_report.json",
            # 报告记录预期说话人向量维度的字段名。
            "expected_embed_dim",
            # 说话人向量固定为 256 维。
            256,
        ),
    ):
        # 读取验证器锁定的当前特征模型权重 SHA-256；这里是指纹字符串。
        checkpoint = validator.EXPECTED_FEATURE_CHECKPOINTS[kind]
        # 锁定的当前特征模型配置指纹。
        config_hash = validator.EXPECTED_FEATURE_CONFIGS[kind]
        # 从人工特征证明中取出当前 BNF 或说话人向量的逐文件记录。
        records = feature_records[kind]
        # 组合当前特征模型的代码、权重和配置指纹，声明统一来源。
        output_provenance = {
            # 特征提取旧代码的实际 Git 提交；此处为测试使用的明确值。
            "legacy_git_commit": validator.EXPECTED_LEGACY_COMMIT,
            # 审计允许的旧代码 Git 提交；此处为测试使用的明确值。
            "legacy_git_expected_commit": validator.EXPECTED_LEGACY_COMMIT,
            # 特征模型权重或检查点文件的实际指纹；此处为测试使用的明确值。
            "checkpoint_sha256": checkpoint,
            # 事先锁定的特征模型权重指纹；此处为测试使用的明确值。
            "checkpoint_expected_sha256": checkpoint,
            # 特征提取配置的实际指纹；此处为测试使用的明确值。
            "config_sha256": config_hash,
            # 事先锁定的特征提取配置指纹；此处为测试使用的明确值。
            "config_expected_sha256": config_hash,
        }
        # BNF 额外绑定位置编码兼容实现的来源。
        if kind == "bnf":
            # 向已有模型来源补充 BNF 特有的兼容源码指纹。
            output_provenance.update(
                {
                    # 本地位置编码兼容源码的实际指纹；此处为测试使用的明确值。
                    "compat_embedding_sha256": validator.EXPECTED_BNF_COMPAT_EMBEDDING,
                    # 允许使用的位置编码兼容源码指纹；此处为测试使用的明确值。
                    "compat_embedding_expected_sha256": validator.EXPECTED_BNF_COMPAT_EMBEDDING,
                }
            )
        # 为当前 BNF 或说话人特征构造完整的提取来源报告。
        report = {
            # 特征提取报告结构版本；此处为测试使用的明确值。
            "report_schema_version": validator.FEATURE_REPORT_SCHEMA_VERSION,
            # 特征类型，如 BNF 或说话人向量；此处为测试使用的明确值。
            "feature_kind": kind,
            # 本次运行完成状态；仅全量8100组可标记正式完成；此处为测试使用的明确值。
            "status": "complete",
            # 整批特征的生成或复用方式；此处为测试使用的明确值。
            "source_mode": "all_generated",
            # 所有特征来源是否已验证且一致；此处为测试使用的明确值。
            "provenance_status": "uniform_verified",
            # 此次提取所覆盖的清单身份；此处为测试使用的明确值。
            "manifest": common_manifest,
            # 产生这些特征的模型、配置和代码来源；此处为测试使用的明确值。
            "output_provenance": output_provenance,
            # 按特征种类登记对应维度字段，BNF 为 144、说话人向量为 256。
            expected_field: expected_dimension,
            # 清单总条数；此处为测试使用的明确值。
            "total_rows": len(rows),
            # 本次新生成特征条数；此处为测试使用的明确值。
            "generated_rows": len(rows),
            # 经过来源检查后复用的条数；此处为测试使用的明确值。
            "reused_verified_rows": 0,
            # 未验证来源却复用的条数，正式流程应为零；此处为测试使用的明确值。
            "unverified_reused_rows": 0,
            # 提取失败条数；此处为测试使用的明确值。
            "failed_rows": 0,
            # 成功条数；此处为测试使用的明确值。
            "ok": len(rows),
            # 跳过条数；此处为测试使用的明确值。
            "skip": 0,
            # 失败条数；此处为测试使用的明确值。
            "fail": 0,
            # 运行开始时需生成的特征数；此处为测试使用的明确值。
            "missing_rows": len(rows),
            # 整批特征与源 WAV 的集合指纹；此处为测试使用的明确值。
            "feature_collection": summarize(records),
            # 逐文件记录的内容与输入指纹；此处为测试使用的明确值。
            "file_records": records,
        }
        # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
        (logs / report_name).write_text(
            # 按所列 JSON 编码参数写测试结构，保证后续指纹计算一致。
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )

    # Construct the same immutable/current publication schema as preprocessing step 04.
    # 用固定人工代次 ID 把均值、标准差及清单指针绑定在一起。
    generation_id = "synthetic-generation"
    # 保存同一代均值、标准差和证明的不可变目录。
    generation_dir = stats / ".mel_stats_generations" / generation_id
    # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
    generation_dir.mkdir(parents=True)
    # 逐项处理 filename in ("mel_mean.npy", "mel_std.npy")，覆盖这组反例的全部输入。
    for filename in ("mel_mean.npy", "mel_std.npy"):
        # 在本例临时目录复制统计量或证明，为不可变代次和兼容别名建立一致基准。
        shutil.copyfile(stats / filename, generation_dir / filename)
    # 当前均值文件的 SHA-256。
    mean_hash = validator.sha256_file(stats / "mel_mean.npy")
    # 当前标准差文件的 SHA-256。
    std_hash = validator.sha256_file(stats / "mel_std.npy")
    # 按语音键累计梅尔数值指纹的哈希对象。
    mel_content = hashlib.sha256()
    # 按语音键累计文件大小的布局哈希对象。
    mel_layout = hashlib.sha256()
    # 逐项处理 row in train_rows，覆盖这组反例的全部输入。
    for row in train_rows:
        # 按清单当前语音生成同一套特征键，保证逐文件哈希顺序一致。
        key = f"{row['speaker']}/{row['utt_id']}.npy"
        # 对应测试音频的梅尔数组路径。
        mel_path = project_root / "data" / "mel" / key
        # 安全读回本条人工梅尔，以其真实数值计算训练集内容指纹。
        mel = np.load(mel_path, allow_pickle=False)
        # 向按语音键累计梅尔数值指纹的哈希对象追加当前语音身份键。
        mel_content.update(key.encode("utf-8") + b"\0")
        # 向按语音键累计梅尔数值指纹的哈希对象追加当前特征数组内容指纹。
        mel_content.update(validator.sha256_array(mel).encode("ascii") + b"\n")
        # 向按语音键累计文件大小的布局哈希对象追加当前语音身份键。
        mel_layout.update(key.encode("utf-8") + b"\0")
        # 向按语音键累计文件大小的布局哈希对象追加当前文件字节长度。
        mel_layout.update(str(mel_path.stat().st_size).encode("ascii") + b"\n")
    # 训练集梅尔统计计算的完整证明。
    mel_report = {
        # 归一化统计量发布结构版本；此处为测试使用的明确值。
        "publication_schema_version": 1,
        # 把均值与标准差绑定到同一次计算的代次标识；此处为测试使用的明确值。
        "generation_id": generation_id,
        # 计算统计量所依据训练清单的指纹；此处为测试使用的明确值。
        "manifest_sha256": validator.sha256_file(metadata / "train_manifest.csv"),
        # 实际参与统计的训练语音条数；此处为测试使用的明确值。
        "rows_used": len(train_rows),
        # 缺失梅尔文件数；此处为测试使用的明确值。
        "missing_count": 0,
        # 格式错误梅尔文件数；此处为测试使用的明确值。
        "bad_count": 0,
        # 80 维梅尔均值文件指纹；此处为测试使用的明确值。
        "mean_sha256": mean_hash,
        # 80 维梅尔标准差文件指纹；此处为测试使用的明确值。
        "std_sha256": std_hash,
        # 参与统计的全部梅尔内容集合指纹；此处为测试使用的明确值。
        "mel_collection_content_sha256": mel_content.hexdigest(),
        # 参与统计的梅尔路径和大小集合指纹；此处为测试使用的明确值。
        "mel_collection_layout_sha256": mel_layout.hexdigest(),
        # 参与统计的梅尔文件数量；此处为测试使用的明确值。
        "mel_collection_count": len(train_rows),
    }
    # 统计证明的 UTF-8 JSON 文本。
    report_payload = json.dumps(mel_report, ensure_ascii=False) + "\n"
    # 不可变代次内的统计计算报告路径。
    generation_report = generation_dir / "report.json"
    # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
    generation_report.write_text(report_payload, encoding="utf-8")
    # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
    (logs / "mel_stats_report.json").write_text(report_payload, encoding="utf-8")
    # 当前已提交生效的统计代次清单。
    current = {
        # 当前 JSON 身份或发布清单结构版本。
        "schema_version": 1,
        # 把均值与标准差绑定到同一次计算的代次标识；此处为测试使用的明确值。
        "generation_id": generation_id,
        # 发布记录的 UTC 时间；此处为测试使用的明确值。
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        # 同一发布代次包含的统计文件；此处为测试使用的明确值。
        "files": {
            # 当前代次均值文件及其字节身份。
            "mel_mean.npy": {
                # 文件实际字节数；此处为测试使用的明确值。
                "bytes": (stats / "mel_mean.npy").stat().st_size,
                # 数组数值存储类型；此处为测试使用的明确值。
                "dtype": "float32",
                # 数组维度，区分通道与时间；此处为测试使用的明确值。
                "shape": [80],
                # 当前内容的 SHA-256 指纹；此处为测试使用的明确值。
                "sha256": mean_hash,
            },
            # 当前代次标准差文件及其字节身份。
            "mel_std.npy": {
                # 文件实际字节数；此处为测试使用的明确值。
                "bytes": (stats / "mel_std.npy").stat().st_size,
                # 数组数值存储类型；此处为测试使用的明确值。
                "dtype": "float32",
                # 数组维度，区分通道与时间；此处为测试使用的明确值。
                "shape": [80],
                # 当前内容的 SHA-256 指纹；此处为测试使用的明确值。
                "sha256": std_hash,
            },
        },
        # 当前代次统计报告的文件身份；此处为测试使用的明确值。
        "report": {
            # 文件实际字节数；此处为测试使用的明确值。
            "bytes": generation_report.stat().st_size,
            # 当前内容的 SHA-256 指纹；此处为测试使用的明确值。
            "sha256": validator.sha256_file(generation_report),
        },
    }
    # 当前统计代次指针的 JSON 文本。
    current_payload = json.dumps(current, ensure_ascii=False) + "\n"
    # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
    (generation_dir / "manifest.json").write_text(current_payload, encoding="utf-8")
    # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
    (stats / "mel_stats_current.json").write_text(current_payload, encoding="utf-8")


# 把同一类故障保护与期望行为组织成一组测试：InventoryBuilderTest。
class InventoryBuilderTest(unittest.TestCase):
    # 验证官方 0.92 双麦 FLAC、p315 无转写和旧 wav48 兼容。
    """验证官方 0.92 双麦 FLAC、p315 无转写和旧 wav48 兼容。"""

    # 辅助函数不访问实例状态，因此声明为静态方法。
    @staticmethod
    # 在租用服务器用真实库存脚本处理临时 VCTK 目录。
    def run_inventory(project_root: Path, vctk_root: Path, *extra: str) -> Path:
        # 用真实命令行运行 stage 00，返回此次指定的输出目录。
        """用真实命令行运行 stage 00，返回此次指定的输出目录。"""

        # 按当前麦克风或布局参数区分库存输出目录，避免各次对照互相覆盖。
        output_dir = project_root / ("metadata-" + (extra[-1] if extra else "default"))
        # 只在租用服务器启动生产脚本子进程，验证真实 CLI 行为。
        subprocess.run(
            [
                # 使用当前服务器测试环境的同一 Python 解释器。
                sys.executable,
                # 将生产 VCTK 库存构建脚本位置转换为命令行字符串。
                str(INVENTORY_SCRIPT),
                # 指定临时人工项目根，避免脚本接触真实实验文件。
                "--project-root",
                # 将本例完全隔离的人工项目根目录转换为命令行字符串。
                str(project_root),
                # 把库存扫描限定在当前临时 VCTK 根目录。
                "--vctk-root",
                # 将临时构造的 VCTK 语料根目录转换为命令行字符串。
                str(vctk_root),
                # 指定库存清单的临时输出目录。
                "--output-dir",
                # 将生产库存脚本输出的人工清单目录转换为命令行字符串。
                str(output_dir),
                # 追加本例指定的麦克风或旧目录布局参数。
                *extra,
            ],
            # 子进程失败时是否立即抛异常，按下列值传给生产函数。
            check=True,
            # 是否收集脚本标准输出与报错供断言检查，按下列值传给生产函数。
            capture_output=True,
            # 把子进程标准输出和错误解码成文本，供成功标记和报错断言检查。
            text=True,
            # 文件文本编码，保证中文和摘要可稳定读写，按下列值传给生产函数。
            encoding="utf-8",
            # 非法字节的解码处理策略，按下列值传给生产函数。
            errors="replace",
        )
        # 返回生产库存脚本生成的清单目录。
        return output_dir

    # 核对默认选官方去静音 mic1，并保留有音频无文本的说话人。
    def test_official_trimmed_default_mic1_includes_speaker_without_text(self) -> None:
        # auto 优先 trimmed/mic1，p315 仍入库且文本字段明确为空。
        """auto 优先 trimmed/mic1，p315 仍入库且文本字段明确为空。"""

        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory(prefix="voicegrad-vctk-inventory-") as temp:
            # 将当前自动清理临时目录作为人工项目根，不接触真实实验。
            project_root = Path(temp)
            # 临时构造的 VCTK 语料根目录。
            vctk_root = project_root / "VCTK-Corpus-0.92"
            # 逐项处理 speaker in ("p225", "p315", "s5")，覆盖这组反例的全部输入。
            for speaker in ("p225", "p315", "s5"):
                # 去静音录音布局下的说话人目录。
                trimmed = vctk_root / "wav48_silence_trimmed" / speaker
                # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
                trimmed.mkdir(parents=True)
                # 逐项处理 microphone in ("mic1", "mic2")，覆盖这组反例的全部输入。
                for microphone in ("mic1", "mic2"):
                    # 写入本例规定的文件字节，用于构造或篡改内容指纹。
                    (trimmed / f"{speaker}_001_{microphone}.flac").write_bytes(b"not-audio")
            # 旧 wav48 布局目录。
            legacy = vctk_root / "wav48" / "p225"
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            legacy.mkdir(parents=True)
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            (legacy / "p225_001.wav").write_bytes(b"not-audio")
            # 人工 VCTK 文本转写目录。
            transcript_dir = vctk_root / "txt" / "p225"
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            transcript_dir.mkdir(parents=True)
            # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
            (transcript_dir / "p225_001.txt").write_text(
                # 写带首尾空白的转写，验证库存脚本会正确清理。
                "  Please call Stella.  \n", encoding="utf-8"
            )
            # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
            (vctk_root / "speaker-info.txt").write_text(
                # 人工说话人信息表的字段表头。
                "ID AGE GENDER ACCENTS REGION\n"
                # 登记有转写说话人的人工年龄、性别和口音信息。
                "p225 23 F English Southern\n"
                # 登记无转写说话人，验证缺文本不会丢掉音频身份。
                "p315 24 M English Northern\n"
                # 登记特殊 ID s5，验证库存解析不只接受 p 加数字。
                "s5 22 F British auxiliary speaker-dependent corpus\n",
                # 文件文本编码，保证中文和摘要可稳定读写，按下列值传给生产函数。
                encoding="utf-8",
            )

            # 使用默认 mic1 规则生成的库存清单目录。
            default_output = self.run_inventory(project_root, vctk_root)
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with (default_output / "vctk_all_pairs.csv").open(
                # 只读 UTF-8 CSV，并由 csv 模块统一处理换行。
                "r", encoding="utf-8", newline=""
            # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
            ) as handle:
                # 读回生产库存脚本生成的全部语音记录，核对麦克风和缺文本说话人。
                rows = list(csv.DictReader(handle))
            # 核对 {row["speaker"] for row in rows}, {"p225", "p315", "s5"}：应完全相等。
            self.assertEqual({row["speaker"] for row in rows}, {"p225", "p315", "s5"})
            # 核对 all(row[“选择的录音麦克风通道”] == "mic1" for row in rows)：必须成立。
            self.assertTrue(all(row["microphone"] == "mic1" for row in rows))
            # 核对 all(row[“原始 VCTK 音频路径”].endswith("_mic1.flac") for row in rows)：必须成立。
            self.assertTrue(all(row["wav48_path"].endswith("_mic1.flac") for row in rows))
            # 有转写的人工说话人记录。
            p225 = next(row for row in rows if row["speaker"] == "p225")
            # 有音频无转写的人工说话人记录。
            p315 = next(row for row in rows if row["speaker"] == "p315")
            # 特殊非 pXXX 编号说话人的库存记录。
            s5 = next(row for row in rows if row["speaker"] == "s5")
            # 核对 p225[“说话人性别标签，用于库存完整性检查”], "F"：应完全相等。
            self.assertEqual(p225["gender"], "F")
            # 核对 p225[“语音转写文本或文本模式标记，具体取决于当前调用”], "Please call Stella."：应完全相等。
            self.assertEqual(p225["text"], "Please call Stella.")
            # 核对 p315[“说话人性别标签，用于库存完整性检查”], "M"：应完全相等。
            self.assertEqual(p315["gender"], "M")
            # 核对 p315[“原始文字转写路径；缺文本者留空”], ""：应完全相等。
            self.assertEqual(p315["txt_path"], "")
            # 核对 p315[“语音转写文本或文本模式标记，具体取决于当前调用”], ""：应完全相等。
            self.assertEqual(p315["text"], "")
            # 核对 s5[“说话人性别标签，用于库存完整性检查”], "F"：应完全相等。
            self.assertEqual(s5["gender"], "F")
            # 核对 s5[“原始文字转写路径；缺文本者留空”], ""：应完全相等。
            self.assertEqual(s5["txt_path"], "")

            # 显式选择 mic2 后的库存目录。
            mic2_output = self.run_inventory(
                # 传入本例完全隔离的人工项目根目录。
                project_root,
                # 传入临时构造的 VCTK 语料根目录。
                vctk_root,
                # 显式指定 VCTK 麦克风通道，检查默认值可被正确覆盖。
                "--microphone",
                # 本反例选择第二麦克风，产物路径应全部带 mic2。
                "mic2",
            )
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with (mic2_output / "vctk_all_pairs.csv").open(
                # 只读 UTF-8 CSV，并由 csv 模块统一处理换行。
                "r", encoding="utf-8", newline=""
            # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
            ) as handle:
                # mic2 库存实际读回的语音行。
                mic2_rows = list(csv.DictReader(handle))
            # 核对 all(row[“选择的录音麦克风通道”] == "mic2" for row in mic2_rows)：必须成立。
            self.assertTrue(all(row["microphone"] == "mic2" for row in mic2_rows))
            # 核对下面完整给出的断言条件。
            self.assertTrue(
                # 要求清单中所有项目均满足当前身份约束。
                all(row["wav48_path"].endswith("_mic2.flac") for row in mic2_rows)
            )

            # 显式选择旧 wav48 布局后的库存目录。
            legacy_output = self.run_inventory(
                # 传入本例完全隔离的人工项目根目录。
                project_root,
                # 传入临时构造的 VCTK 语料根目录。
                vctk_root,
                # 显式指定原始音频布局，检查兼容旧版目录。
                "--audio-layout",
                # 选旧 wav48 布局而非默认去静音目录。
                "wav48",
            )
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with (legacy_output / "vctk_all_pairs.csv").open(
                # 只读 UTF-8 CSV，并由 csv 模块统一处理换行。
                "r", encoding="utf-8", newline=""
            # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
            ) as handle:
                # 旧布局实际读回的语音行。
                legacy_rows = list(csv.DictReader(handle))
            # 核对 len(legacy_rows), 1：应完全相等。
            self.assertEqual(len(legacy_rows), 1)
            # 核对 legacy_rows[0][“原始 VCTK 音频路径”].endswith("p225_001.wav")：必须成立。
            self.assertTrue(legacy_rows[0]["wav48_path"].endswith("p225_001.wav"))


# 把同一类故障保护与期望行为组织成一组测试：SplitProtocolTest。
class SplitProtocolTest(unittest.TestCase):
    # 分别运行两个协议，并核对唯一应当变化的训练提示句集合。
    """分别运行两个协议，并核对唯一应当变化的训练提示句集合。"""

    # 在独立临时目录运行指定 seen 或 unseen 划分并读回清单。
    def run_protocol(self, protocol: str) -> tuple[List[Dict[str, str]], List[Dict[str, str]], dict]:
        # 在自动清理的临时目录运行一次 02 脚本并返回三类结果。
        """在自动清理的临时目录运行一次 02 脚本并返回三类结果。"""

        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory(prefix=f"voicegrad-{protocol}-") as temp:
            # 将当前自动清理临时目录作为人工项目根，不接触真实实验。
            project_root = Path(temp)
            # 创建独立的 110 人各 12 句人工库存。
            build_synthetic_inventory(project_root)
            # 只在租用服务器启动生产脚本子进程，验证真实 CLI 行为。
            subprocess.run(
                [
                    # 使用当前服务器测试环境的同一 Python 解释器。
                    sys.executable,
                    # 将生产文本 seen/unseen 划分脚本位置转换为命令行字符串。
                    str(SPLIT_SCRIPT),
                    # 指定临时人工项目根，避免脚本接触真实实验文件。
                    "--project-root",
                    # 将本例完全隔离的人工项目根目录转换为命令行字符串。
                    str(project_root),
                    # 指定本例要运行 seen 还是 unseen 文本划分。
                    "--protocol",
                    # 传入当前数据清单声明的划分协议。
                    protocol,
                    # 人工临时库存无需保留历史备份。
                    "--no-backup",
                ],
                # 子进程失败时是否立即抛异常，按下列值传给生产函数。
                check=True,
                # 是否收集脚本标准输出与报错供断言检查，按下列值传给生产函数。
                capture_output=True,
                # 把子进程标准输出和错误解码成文本，供成功标记和报错断言检查。
                text=True,
                # 文件文本编码，保证中文和摘要可稳定读写，按下列值传给生产函数。
                encoding="utf-8",
            )
            # 人工项目的训练、测试、总库存与协议清单都位于此元数据目录。
            metadata = project_root / "data" / "metadata"
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with (metadata / "train_manifest.csv").open(
                # 只读 UTF-8 CSV，并由 csv 模块统一处理换行。
                "r", encoding="utf-8", newline=""
            # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
            ) as handle:
                # 读回当前协议实际生成的训练记录列表，随后检查人数和句子范围。
                train_rows = list(csv.DictReader(handle))
            # 按指定编码打开测试文件，作用域结束后自动关闭。
            with (metadata / "test_manifest.csv").open(
                # 只读 UTF-8 CSV，并由 csv 模块统一处理换行。
                "r", encoding="utf-8", newline=""
            # 将已打开的测试清单或报告绑定为 handle，退出 with 后自动关闭。
            ) as handle:
                # 读回当前协议实际生成的测试记录列表，随后检查未知说话人边界。
                test_rows = list(csv.DictReader(handle))
            # 读回生产划分报告，检查 seen/unseen 规则与说话人数。
            report = json.loads(
                # 读回划分报告，核对协议名、人数和公共句处理方式。
                (metadata / "split_protocol.json").read_text(encoding="utf-8")
            )
        # 把训练清单、测试清单和协议报告一起交给断言。
        return train_rows, test_rows, report

    # 确认 unseen 训练集删除公共测试句，且说话人与测试集不相交。
    def test_text_unseen_removes_shared_prompts(self) -> None:
        # 严格协议训练人只剩 011--012，因此测试文本从未进入训练清单。
        """严格协议训练人只剩 011--012，因此测试文本从未进入训练清单。"""

        # 实际运行 unseen 划分，检查公共句是否全部从训练人中剔除。
        train_rows, test_rows, report = self.run_protocol("text-unseen")
        # 核对 len(train_rows), 100 * 2：应完全相等。
        self.assertEqual(len(train_rows), 100 * 2)
        # 核对 len(test_rows), 10 * 10：应完全相等。
        self.assertEqual(len(test_rows), 10 * 10)
        # 核对 report["strict_text_unseen"]：必须成立。
        self.assertTrue(report["strict_text_unseen"])
        # 核对 report[“训练使用100个说话人”], 100：应完全相等。
        self.assertEqual(report["train_speakers"], 100)
        # 核对 report["legacy_all_manifest_used"]：必须不成立。
        self.assertFalse(report["legacy_all_manifest_used"])
        # 当前特征提取模型、配置和代码指纹。
        provenance = report["exact_split_id_provenance"]
        # 核对 provenance[“已在论文中明确给出的设定”]：必须不成立。
        self.assertFalse(provenance["paper_explicit"])
        # 核对 provenance["reconstruction_confidence"], "high"：应完全相等。
        self.assertEqual(provenance["reconstruction_confidence"], "high")
        # 核对 any(row["speaker"] == "p315" for row in train_rows)：必须成立。
        self.assertTrue(any(row["speaker"] == "p315" for row in train_rows))
        # 核对下面完整给出的断言条件。
        self.assertTrue(
            # 要求清单中所有项目均满足当前身份约束。
            all(not row["text"] for row in train_rows if row["speaker"] == "p315")
        )
        # 核对下面完整给出的断言条件。
        self.assertEqual(
            # 提取每条训练语音最后的句子槽位，检查剩余文本范围。
            {row["utt_id"].rsplit("_", 1)[-1] for row in train_rows},
            # 严格 unseen 的人工训练集应只保留第十一和十二句。
            {"011", "012"},
        )
        # 核对下面完整给出的断言条件。
        self.assertTrue(
            # 把训练说话人转成集合，要求与测试人没有交集。
            {row["speaker"] for row in train_rows}.isdisjoint(
                # 构造测试说话人集合，检查未知说话人边界。
                {row["speaker"] for row in test_rows}
            )
        )

    # 确认 seen 对照保留公共句，但仍隔离训练和测试说话人。
    def test_text_seen_keeps_shared_prompts(self) -> None:
        # 对照协议保留训练人的 001--010，但测试说话人仍与训练完全分离。
        """对照协议保留训练人的 001--010，但测试说话人仍与训练完全分离。"""

        # 实际运行 seen 对照划分，检查公共句保留而说话人仍隔离。
        train_rows, test_rows, report = self.run_protocol("text-seen")
        # 核对 len(train_rows), 100 * 12：应完全相等。
        self.assertEqual(len(train_rows), 100 * 12)
        # 核对 len(test_rows), 10 * 10：应完全相等。
        self.assertEqual(len(test_rows), 10 * 10)
        # 核对 report["strict_text_unseen"]：必须不成立。
        self.assertFalse(report["strict_text_unseen"])
        # 核对 report["retained_train_prompt_rows"], 100 * 10：应完全相等。
        self.assertEqual(report["retained_train_prompt_rows"], 100 * 10)
        # 核对下面完整给出的断言条件。
        self.assertTrue(
            # 把训练说话人转成集合，要求与测试人没有交集。
            {row["speaker"] for row in train_rows}.isdisjoint(
                # 构造测试说话人集合，检查未知说话人边界。
                {row["speaker"] for row in test_rows}
            )
        )

    # 确认两类合法划分连同一致特征和统计证明都能通过完整启动验证。
    def test_full_validator_accepts_each_protocol(self) -> None:
        # 活动报告、文本边界和三种特征全部一致时，两版都应通过完整验证。
        """活动报告、文本边界和三种特征全部一致时，两版都应通过完整验证。"""

        # 逐项处理 protocol in ("text-unseen", "text-seen")，覆盖这组反例的全部输入。
        for protocol in ("text-unseen", "text-seen"):
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(protocol=protocol):
                # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
                with tempfile.TemporaryDirectory(prefix=f"voicegrad-valid-{protocol}-") as temp:
                    # 将当前自动清理临时目录作为人工项目根，不接触真实实验。
                    project_root = Path(temp)
                    # 创建独立的 110 人各 12 句人工库存。
                    build_synthetic_inventory(project_root)
                    # 只在租用服务器启动生产脚本子进程，验证真实 CLI 行为。
                    subprocess.run(
                        [
                            # 使用当前服务器测试环境的同一 Python 解释器。
                            sys.executable,
                            # 将生产文本 seen/unseen 划分脚本位置转换为命令行字符串。
                            str(SPLIT_SCRIPT),
                            # 指定临时人工项目根，避免脚本接触真实实验文件。
                            "--project-root",
                            # 将本例完全隔离的人工项目根目录转换为命令行字符串。
                            str(project_root),
                            # 指定本例要运行 seen 还是 unseen 文本划分。
                            "--protocol",
                            # 传入当前数据清单声明的划分协议。
                            protocol,
                            # 人工临时库存无需保留历史备份。
                            "--no-backup",
                        ],
                        # 子进程失败时是否立即抛异常，按下列值传给生产函数。
                        check=True,
                        # 是否收集脚本标准输出与报错供断言检查，按下列值传给生产函数。
                        capture_output=True,
                        # 把子进程标准输出和错误解码成文本，供成功标记和报错断言检查。
                        text=True,
                        # 文件文本编码，保证中文和摘要可稳定读写，按下列值传给生产函数。
                        encoding="utf-8",
                    )
                    # 按刚生成的活动清单建立合法特征及完整来源证明。
                    build_synthetic_features(project_root)
                    # 真实命令行执行结果，包含退出码与标准输出。
                    completed = subprocess.run(
                        [
                            # 使用当前服务器测试环境的同一 Python 解释器。
                            sys.executable,
                            # 通过 Python 模块入口执行生产验证器。
                            "-m",
                            # 传入生产训练前完整验证模块名。
                            VALIDATE_MODULE,
                            # 指定临时人工项目根，避免脚本接触真实实验文件。
                            "--project-root",
                            # 将本例完全隔离的人工项目根目录转换为命令行字符串。
                            str(project_root),
                            # 要求完整核对特征内容与来源，不只做轻量布局检查。
                            "--full",
                        ],
                        # 子进程失败时是否立即抛异常，按下列值传给生产函数。
                        check=True,
                        # 是否收集脚本标准输出与报错供断言检查，按下列值传给生产函数。
                        capture_output=True,
                        # 把子进程标准输出和错误解码成文本，供成功标记和报错断言检查。
                        text=True,
                        # 文件文本编码，保证中文和摘要可稳定读写，按下列值传给生产函数。
                        encoding="utf-8",
                        # 子进程的工作目录，按下列值传给生产函数。
                        cwd=REPOSITORY_ROOT,
                    )
                    # 核对 "SETUP CHECK PASSED", completed.stdout：应包含目标内容。
                    self.assertIn("SETUP CHECK PASSED", completed.stdout)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # verbosity=2 会打印每个测试方法名，初学者能清楚看到哪一步通过或失败。
    unittest.main(verbosity=2)
