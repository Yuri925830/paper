#!/usr/bin/env python3
# 这份历史工具在训练说话人上比较两种 BNF 对齐方式，输出不能冒充正式 8100 对 unseen 指标。
"""Run a small, controlled BNF-alignment A/B on training speakers.

This is deliberately not the 8,100-pair text-unseen paper evaluation.  It uses
the same trained teacher, self-trained HiFi-GAN, source utterances, target
references, diffusion noise and metric implementations for both branches.  The
only changed variable is BNF-to-mel alignment, so the result can diagnose
whether the time-grid correction is promising before any new teacher training.
"""

# 延迟解析类型提示；只影响类型标注处理，不改变推理数值。
from __future__ import annotations

# 解析命令行指定的教师、声码器、样本数和对齐模式。
import argparse
# 读取有表头的训练清单及其文本来源。
import csv
# 保存诊断设置、来源指纹与运行汇总。
import json
# 用原子替换发布完整 JSON，避免中断留下半份正式报告。
import os
# 直接运行 tools 下文件时，帮助登记正确的项目导入路径。
import sys
# 记录每种对齐模式的实际耗时。
import time
# 按说话人或源句将记录分组，缺少分组时自动建立空列表。
from collections import defaultdict
# 导入数据类字典转换工具；当前这份脚本未实际调用它。
from dataclasses import asdict
# 使用路径对象组合特征、权重和输出文件位置。
from pathlib import Path
# 类型提示描述清单字典与样本序列，便于阅读函数输入。
from typing import Dict, List, Sequence

# 读取数值特征、堆叠说话人向量，并检查音频有限值。
import numpy as np
# 执行教师扩散采样和声码器推理，不更新模型参数。
import torch
# 显示各源句的诊断推理进度。
from tqdm import tqdm

# Running ``python tools/this_file.py`` puts tools/ rather than the repository
# root on sys.path.  Add the parent explicitly before importing project modules.
# 从脚本路径定位仓库根目录，避免终端当前目录影响导入。
PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[1]
# 只有路径尚未登记时才添加，避免重复路径。
if str(PROJECT_SOURCE_ROOT) not in sys.path:
    # 把当前仓库放到导入搜索首位，读取本项目的实现。
    sys.path.insert(0, str(PROJECT_SOURCE_ROOT))

# 复用教师论文配置及登记的租用服务器目录。
from voicegrad.config import PAPER_CONFIG, SERVER_PROJECT_ROOT
# 使用与正式教师相同的余弦扩散与采样实现。
from voicegrad.diffusion import VoiceGradDiffusion
# 列出支持的外部 BNF 对齐模式，并导入统一对齐函数。
from voicegrad.feature_alignment import ALIGNMENT_MODES, align_bnf_to_mel
# 复用正式教师推理中的配对、文件验证和模型加载工具。
from voicegrad.infer_voicegrad30_paper_eval import (
    # EvaluationPair 保存一对源语音与目标参考的完整身份。
    EvaluationPair,
    # feature_matrix 核对 mel/BNF 维度并统一特征方向。
    feature_matrix,
    # feature_path 按说话人和句号定位已有特征文件。
    feature_path,
    # load_voicegrad 严格建立教师网络并加载检查点。
    load_voicegrad,
    # sha256_file 记录权重、清单和配对文件的字节指纹。
    sha256_file,
    # utterance_number 从句子编号取出语句槽位，方便追查文本。
    utterance_number,
    # valid_existing_wav 核对已经写出的波形是否可用，支持中断后跳过完整文件。
    valid_existing_wav,
    # write_pairs_csv 保存实际生成的配对清单，后续指标必须与之对应。
    write_pairs_csv,
)
# 内部 BNF 下采样选项单独登记，避免与外部时间轴插值混为一项设置。
from voicegrad.model import BNF_DOWNSAMPLING_MODES, LEGACY_BNF_DOWNSAMPLING
# 复用声码器加载和原子写 WAV，防止保留半成品音频。
from hifigan.tools.vocode import atomic_write_wav, load_generator


# 这份历史诊断默认使用第 500 epoch 教师。
DEFAULT_TEACHER = (
    # 从统一的租用服务器根目录开始定位权重。
    SERVER_PROJECT_ROOT
    # 进入实验产物目录 exp。
    / "exp"
    # 进入模型检查点目录。
    / "checkpoints"
    # 选择 text-unseen 协议训练出的 VoiceGrad 教师。
    / "voicegrad30_vctk_text_unseen"
    # 固定默认 epoch_0500.pt；实际可由命令行显式覆盖。
    / "epoch_0500.pt"
)
# 历史诊断默认声码器为自训 100 万步版本，不等于当前修复训练使用的 250 万步。
DEFAULT_VOCODER = (
    # 声码器同样从租用项目目录定位。
    SERVER_PROJECT_ROOT
    # 使用实验产物目录。
    / "exp"
    # 进入声码器检查点存放位置。
    / "checkpoints"
    # 明确选择修改版 HiFi-GAN V1+MRD 的 text-unseen 训练目录。
    / "hifigan_v1_mrd_vctk_text_unseen"
    # 保留历史默认的 generator_step_1000000.pt，注释整理不暗改实验版本。
    / "generator_step_1000000.pt"
)
# 默认配置与该声码器权重处于同一目录，避免结构和权重串用。
DEFAULT_VOCODER_CONFIG = DEFAULT_VOCODER.parent / "config.json"
# 历史诊断有独立输出目录，不覆盖正式测试结果。
DEFAULT_OUTPUT_ROOT = (
    # 输出从统一项目根目录展开。
    SERVER_PROJECT_ROOT
    # 放入实验产物目录。
    / "exp"
    # 归入 evaluation，但下文会明确标注仅为诊断。
    / "evaluation"
    # 目录名保留训练集来源、教师 epoch 与声码器步数，便于区分版本。
    / "bnf_alignment_diagnostic_seen_train_epoch0500_hifigan_step1000000"
)


# 解析此次对齐诊断的路径和样本控制参数。
def parse_args() -> argparse.Namespace:
    # 使用文件用途说明生成命令行帮助。
    parser = argparse.ArgumentParser(description=__doc__)
    # 明确项目根目录，默认使用登记的租用服务器位置。
    parser.add_argument("--project-root", type=Path, default=SERVER_PROJECT_ROOT)
    # 可选补充清单只填补文本，不改变实际样本选择。
    parser.add_argument(
        # --text-manifest 指向文本完整的补充清单。
        "--text-manifest",
        # 将清单参数解析为 Path 对象。
        type=Path,
        # 默认没有补充清单，完全使用服务器现有清单。
        default=None,
        # 帮助文字说明允许补充的范围。
        help=(
            # 第一段说明只填原清单中的空文本字段。
            "Optional text-complete manifest used only to fill blank transcript fields; "
            # 第二段强调说话人和特征位置始终由服务器训练清单确定。
            "speaker selection and feature paths always come from the server train manifest."
        ),
    )
    # 允许明确指定教师检查点，默认第 500 epoch。
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    # 允许明确指定声码器权重，默认保留历史 100 万步版本。
    parser.add_argument("--vocoder", type=Path, default=DEFAULT_VOCODER)
    # 指定和声码器配套的结构配置。
    parser.add_argument("--vocoder-config", type=Path, default=DEFAULT_VOCODER_CONFIG)
    # 指定诊断输出根目录，每种对齐模式会有子目录。
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    # 默认选取 10 位训练说话人提供源语音内容。
    parser.add_argument("--source-speakers", type=int, default=10)
    # 另外选取 10 位不重叠的训练说话人提供目标音色参考。
    parser.add_argument("--target-speakers", type=int, default=10)
    # 固定扩散噪声种子，使两种对齐模式共用相同随机输入。
    parser.add_argument("--seed", type=int, default=PAPER_CONFIG.seed)
    # 内部下采样模式独立设定，作为本次诊断明确记录的条件。
    parser.add_argument(
        # --bnf-downsampling 选择 U-Net 内部的 BNF 降采样实现。
        "--bnf-downsampling",
        # 只接受源码登记的模式，错误拼写不能静默生效。
        choices=BNF_DOWNSAMPLING_MODES,
        # 默认保留教师历史内部抽点方式，外部对齐比较不自动改它。
        default=LEGACY_BNF_DOWNSAMPLING,
        # 帮助说明这是受控诊断选项，不能宣称原教师在该接口上重训过。
        help="Controlled U-Net BNF downsampling branch to diagnose.",
    )
    # 在租用服务器内选择可用 CUDA 或 CPU；本地 Windows 仍不允许执行本工具。
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 仅显式 --overwrite 才重新生成已有有效音频。
    parser.add_argument("--overwrite", action="store_true")
    # 返回已经解析的本次实际参数。
    return parser.parse_args()


# 读取并严格确认所用清单只包含训练来源语音。
def load_train_rows(manifest: Path) -> List[Dict[str, str]]:
    # 先检查清单确实存在，避免后续空输入。
    if not manifest.is_file():
        # 清单缺失就报出完整路径，不替换为另一份划分。
        raise FileNotFoundError(f"Training manifest not found: {manifest}")
    # 以 UTF-8 和 CSV 约定读取清单，保持字段内容。
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        # 按表头把每句变成字典，不依赖固定列号。
        rows = list(csv.DictReader(handle))
    # 训练清单为空则无法选择诊断样本。
    if not rows:
        # 空清单立即报错，避免生成空结果却显示成功。
        raise ValueError(f"Training manifest is empty: {manifest}")
    # 列出配对及文本追踪必需的字段。
    required = {
        # split 说明这条语音属于哪一个数据划分。
        "split",
        # speaker 标记说话人身份，用来建立不重叠分组。
        "speaker",
        # utt_id 标记具体语音，关联各类特征。
        "utt_id",
        # text 提供真实转写，后续 CER 可据此比较。
        "text",
        # txt_path 保留原始文本文件的追踪位置。
        "txt_path",
        # wav22k_path 指向 22.05 kHz 波形。
        "wav22k_path",
        # is_unseen_speaker 防止混入正式测试说话人。
        "is_unseen_speaker",
    }
    # 核对清单表头包含全部必需字段。
    if not required.issubset(rows[0]):
        # 报出缺失列名，要求补齐数据而非猜测字段。
        raise ValueError(f"Training manifest is missing {sorted(required - set(rows[0]))}")
    # 逐条检查划分和说话人标记，不能只检查第一句。
    for row in rows:
        # 清理大小写和外部空白后要求 split 为 train。
        if row["split"].strip().lower() != "train":
            # 遇到非训练样本立即停止，保护正式测试集的独立性。
            raise ValueError(f"Non-training row found: {row['utt_id']}")
        # 只接受明确的否定标记，表示这位说话人未被标成 unseen。
        if row["is_unseen_speaker"].strip() not in {"0", "False", "false"}:
            # 混入 unseen 说话人时指出具体语音编号。
            raise ValueError(f"Training row is marked unseen: {row['utt_id']}")
    # 确认全部行符合训练来源要求后返回。
    return rows


# 从训练说话人中选择源句与目标参考句，二者说话人组不重叠。
def choose_items(
    # rows 是候选语音，两个 count 分别控制源/目标说话人数。
    rows: Sequence[Dict[str, str]], source_count: int, target_count: int
# 返回两份语音列表，后续构成全部交叉配对。
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    # 两侧说话人数都必须为正，空诊断无法比较。
    if source_count <= 0 or target_count <= 0:
        # 样本数量参数无效时立即报错。
        raise ValueError("source-speakers and target-speakers must be positive")
    # 按说话人聚合有可追踪文本的候选语音。
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    # 检查每条语音能否取得真实文本。
    for row in rows:
        # 清单里有非空文本，或存在原始文本文件，才可用于后续客观检查。
        has_traceable_text = bool(row.get("text", "").strip()) or Path(
            # 从当前语音记录提取文本文件路径，缺失则用空字符串表示。
            row.get("txt_path", "")
        # 判断该路径是否为实际文件，不把目录或不存在路径算作有效来源。
        ).is_file()
        # 只将可追踪文本的语音加入候选集合。
        if has_traceable_text:
            # 把这条语音追加到其说话人名下。
            grouped[row["speaker"]].append(row)
    # 排序说话人编号，保证同一清单每次选择一致。
    speakers = sorted(grouped)
    # 必须有足够多说话人，才能保持源组与参考组互不重叠。
    if len(speakers) < source_count + target_count:
        # 人数不足时停止，不能悄悄让源和目标复用同一说话人。
        raise ValueError(
            # 错误信息列出要求人数和实际人数，便于调整诊断样本规模。
            f"Need {source_count + target_count} disjoint speakers, found {len(speakers)}"
        )

    # 保存源语音列表，源句决定要保留的语言内容。
    sources: List[Dict[str, str]] = []
    # 保存目标参考列表，参考句决定转换音色。
    references: List[Dict[str, str]] = []
    # 先选排序后前 source_count 位作为源说话人。
    for speaker in speakers[:source_count]:
        # 每位说话人的语音按句号排序，固定选择顺序。
        speaker_rows = sorted(grouped[speaker], key=lambda row: row["utt_id"])
        # 每位源说话人取第一句，避免随机选择影响两组比较。
        sources.append(speaker_rows[0])
    # 紧接着选另外 target_count 位作为目标参考说话人。
    for speaker in speakers[source_count : source_count + target_count]:
        # 同样固定目标语音的排序。
        speaker_rows = sorted(grouped[speaker], key=lambda row: row["utt_id"])
        # 目标有两句时取第二句，否则取唯一一条；这属于诊断选句约定。
        references.append(speaker_rows[1] if len(speaker_rows) > 1 else speaker_rows[0])
    # 返回源句与参考句，不修改原训练清单。
    return sources, references


# 优先读取清单文本，缺失时回到清单登记的原始文本文件。
def transcript(row: Dict[str, str]) -> str:
    # 函数说明保留原文；下方中文分别解释两条文本读取路径。
    """Return manifest text, falling back to the recorded VCTK txt file."""

    # 先清理清单文本两端空白。
    value = row.get("text", "").strip()
    # 清单文本非空即可直接使用。
    if value:
        # 返回已登记文本，不调用 ASR 来猜参考答案。
        return value
    # 清单没有文本时取得原始转写文件路径。
    txt_path = Path(row.get("txt_path", ""))
    # 不存在真实文本文件就无法追踪这条参考文本。
    if not txt_path.is_file():
        # 拒绝用空字符串凑齐样本。
        raise ValueError(
            # 错误中保留语音编号与路径，便于补齐正确文本。
            f"Missing transcript and txt_path for {row.get('utt_id')}: {txt_path}"
        )
    # 读取原始 UTF-8 文本并去掉首尾空白。
    value = txt_path.read_text(encoding="utf-8").strip()
    # 即使文件存在，内容为空也不能作为转写。
    if not value:
        # 报出空文本文件及语音编号。
        raise ValueError(f"Empty transcript file for {row.get('utt_id')}: {txt_path}")
    # 返回从原始转写读取的非空内容。
    return value


# 将每个源句与每个目标参考组合，建立诊断配对清单。
def build_pairs(
    # sources 提供被转换的语音及其文字。
    sources: Sequence[Dict[str, str]],
    # references 提供目标说话人的音色样本。
    references: Sequence[Dict[str, str]],
    # output_dir 指当前对齐模式的独立结果目录。
    output_dir: Path,
# 函数返回带完整来源信息的 EvaluationPair 列表。
) -> List[EvaluationPair]:
    # 初始化配对列表，依次写入所有交叉组合。
    pairs: List[EvaluationPair] = []
    # 枚举源句，索引用于生成稳定的配对编号。
    for source_index, source in enumerate(sources):
        # 取得源语音真实文字，客观识别错误率应对照它。
        source_text = transcript(source)
        # 对当前源句遍历每一条目标参考。
        for reference_index, reference in enumerate(references):
            # 记录参考文字以保留完整来源，但转换内容仍由源句决定。
            reference_text = transcript(reference)
            # 二维源/参考索引转换为唯一整数编号。
            pair_id = source_index * len(references) + reference_index
            # 构造可读且不会在不同配对间冲突的 WAV 文件名。
            output_wav = (
                # 文件放到当前模式的结果目录。
                output_dir
                # 具体波形集中保存在 wav 子目录。
                / "wav"
                # 下面两个格式化片段连成同一个文件名。
                / (
                    # 文件名前半保留配对号和源句编号。
                    f"pair_{pair_id:04d}__src_{source['utt_id']}"
                    # 后半保留参考句编号及 .wav 后缀。
                    f"__ref_{reference['utt_id']}.wav"
                )
            )
            # 把当前源/参考组合追加到结果列表。
            pairs.append(
                # 使用与正式推理相同的数据结构，方便复用指标读取器。
                EvaluationPair(
                    # 记录当前唯一配对号。
                    pair_id=pair_id,
                    # 记录源说话人身份。
                    source_speaker=source["speaker"],
                    # 记录源句编号，用于关联 mel 和 BNF。
                    source_utt=source["utt_id"],
                    # 从源句编号解析语句槽位，保留选句位置。
                    source_sentence_slot=utterance_number(source["utt_id"]),
                    # 源文本是转换后应保留的语义内容。
                    source_text=source_text,
                    # 源 WAV 路径可用于追溯原音频。
                    source_wav=source["wav22k_path"],
                    # 记录目标参考说话人身份。
                    reference_speaker=reference["speaker"],
                    # 记录用于提供目标音色的具体参考语音。
                    reference_utt=reference["utt_id"],
                    # 同时记录参考句的语句槽位。
                    reference_sentence_slot=utterance_number(reference["utt_id"]),
                    # 保存参考文字用于审计，不替代源文本。
                    reference_text=reference_text,
                    # 参考 WAV 是说话人验证等诊断的真实参照。
                    reference_wav=reference["wav22k_path"],
                    # 记录将要生成的 WAV 路径，供后续客观工具读取。
                    output_wav=str(output_wav),
                )
            )
    # 返回全部交叉配对，不在这里执行推理。
    return pairs


# 将报告完整写到临时文件，再一次性发布，避免读取半份结果。
def atomic_write_json(path: Path, payload: dict) -> None:
    # 创建报告父目录，已经存在则保留。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件与目标同目录，支持同一文件系统上的原子替换。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 先写出完整 UTF-8 JSON，中文可直接阅读。
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 最后替换正式路径，此时读者才会看到新报告。
    os.replace(temporary, path)


# 运行受控对齐诊断；这是历史工具，正式修复训练不自动调用它。
def main() -> None:
    # 读取命令行中的模型、样本与模式选择。
    args = parse_args()
    # 规范化权威项目根目录。
    project_root = args.project_root.resolve()
    # 特征和元数据都从该项目的 data 目录读取。
    data_dir = project_root / "data"
    # 明确本次实际教师检查点路径。
    teacher_path = args.teacher.resolve()
    # 明确本次实际声码器权重路径。
    vocoder_path = args.vocoder.resolve()
    # 明确本次实际声码器配置路径。
    vocoder_config_path = args.vocoder_config.resolve()
    # 规范化诊断输出目录，两个模式会分别保存在其下。
    output_root = args.output_root.resolve()
    # 在当前租用服务器创建所选计算设备。
    device = torch.device(args.device)

    # 这里只打开训练清单，避免用正式测试结果选择对齐方案。
    manifest = data_dir / "metadata" / "train_manifest.csv"
    # 读取并确认所有样本都属于 train 且未标为 unseen。
    rows = load_train_rows(manifest)
    # 可选补充文本清单存在时转成绝对路径。
    text_manifest = args.text_manifest.resolve() if args.text_manifest is not None else None
    # 只在用户提供文本补充清单时执行补全。
    if text_manifest is not None:
        # 补充清单也必须通过训练来源检查。
        supplemental_rows = load_train_rows(text_manifest)
        # 建立以说话人和句子编号为键的文本索引。
        supplemental_text = {
            # 双字段共同匹配，避免不同说话人的同号句互相串用。
            (row["speaker"], row["utt_id"]): row.get("text", "").strip()
            # 遍历补充训练清单的全部记录。
            for row in supplemental_rows
            # 只收录确实有非空文本的行。
            if row.get("text", "").strip()
        }
        # 依次查看服务器权威训练清单的记录。
        for row in rows:
            # 只补原来为空的文本，已有文本不覆盖。
            if not row.get("text", "").strip():
                # 按同一说话人和同一句号查找补充文字。
                value = supplemental_text.get((row["speaker"], row["utt_id"]))
                # 匹配到非空文字才填入。
                if value:
                    # 仅修改内存中这一行的 text；不改清单文件和特征路径。
                    row["text"] = value
    # 按固定排序选出源和目标参考，不让补充清单重新决定说话人。
    sources, references = choose_items(rows, args.source_speakers, args.target_speakers)

    # 读取训练 mel 均值并整理为 [80,1]，可沿时间帧广播。
    mean = np.load(data_dir / "stats" / "mel_mean.npy", allow_pickle=False).reshape(80, 1)
    # 读取同一套训练标准差，不能用诊断或测试数据重新拟合。
    std = np.load(data_dir / "stats" / "mel_std.npy", allow_pickle=False).reshape(80, 1)
    # 均值转成 float32 张量并放到同一设备，供逆标准化。
    mean_tensor = torch.from_numpy(mean.astype(np.float32)).to(device)
    # 标准差同样转换，避免与网络数据类型不匹配。
    std_tensor = torch.from_numpy(std.astype(np.float32)).to(device)

    # 加载教师及其原始检查点信息。
    model, teacher_checkpoint = load_voicegrad(
        # 使用命令行明确选择的教师权重。
        teacher_path,
        # 将教师放到当前选定的租用服务器设备。
        device,
        # 内部 BNF 下采样采用指定诊断模式。
        bnf_downsampling=args.bnf_downsampling,
        # 这里允许诊断性覆盖内部模式；报告会披露是否与原教师训练接口不同。
        allow_bnf_downsampling_override=True,
    )
    # 创建与教师相同的扩散时间表。
    diffusion = VoiceGradDiffusion(
        # 总时间步数和余弦偏移都取统一论文配置。
        PAPER_CONFIG.diffusion_steps, PAPER_CONFIG.cosine_offset
    # 将扩散系数缓存放到网络同一设备。
    ).to(device)
    # 加载声码器并记录实际运行配置。
    vocoder, vocoder_runtime_config = load_generator(
        # 指定权威项目根目录来寻找受控实现。
        project_root=project_root,
        # 传入本次声码器权重，不根据目录名猜最新版本。
        checkpoint_path=vocoder_path,
        # 传入与权重对应的结构配置。
        config_path=vocoder_config_path,
        # 声码器与教师放在同一计算设备。
        device=device,
    )

    # 计算教师字节指纹，保证两种对齐模式实际用同一文件。
    teacher_hash = sha256_file(teacher_path)
    # 同样锁定声码器权重身份。
    vocoder_hash = sha256_file(vocoder_path)
    # 声码器配置单独记录哈希，避免只锁权重漏掉接口变化。
    vocoder_config_hash = sha256_file(vocoder_config_path)

    # 依次运行登记的外部 BNF 对齐方式。
    for mode in ALIGNMENT_MODES:
        # 为当前模式单独计时。
        start_time = time.time()
        # 每种模式独立子目录，防止 WAV 互相覆盖。
        output_dir = output_root / mode
        # 创建当前模式的波形目录。
        (output_dir / "wav").mkdir(parents=True, exist_ok=True)
        # 两种模式使用相同源句、参考句和配对顺序。
        pairs = build_pairs(sources, references, output_dir)
        # 先写配对清单，后续指标可准确关联生成波形。
        write_pairs_csv(pairs, output_dir / "pairs.csv")
        # 统计当前实际新生成的 WAV 数量。
        generated = 0
        # 另记已存在且通过检查而跳过的数量。
        skipped = 0

        # 按源句聚合配对，复用同一句的 mel 和 BNF。
        pairs_by_source: Dict[str, List[EvaluationPair]] = defaultdict(list)
        # 遍历所有已建立配对。
        for pair in pairs:
            # 将同一源句的不同目标参考放在一组。
            pairs_by_source[pair.source_utt].append(pair)

        # 逐源句推理，并显示当前外部对齐模式。
        for source_index, source in enumerate(tqdm(sources, desc=f"Diagnostic {mode}")):
            # 取得当前源句对应的所有目标参考配对。
            source_pairs = pairs_by_source[source["utt_id"]]
            # 未要求覆盖且这一组音频全都有效时可以跳过。
            if not args.overwrite and all(
                # 逐一核对目标 WAV，而不是仅凭文件名存在就认定成功。
                valid_existing_wav(Path(pair.output_wav)) for pair in source_pairs
            ):
                # 将整组配对计入跳过数量。
                skipped += len(source_pairs)
                # 转到下一条源语音，避免重复推理浪费租用时长。
                continue

            # 读取源语音的未标准化 mel 特征。
            mel = feature_matrix(
                # 路径由同一源说话人和语音编号共同确定。
                feature_path(data_dir, "mel", source["speaker"], source["utt_id"]),
                # 要求 mel 特征维度为论文配置的 80。
                PAPER_CONFIG.n_mels,
            )
            # 读取同一源语音的 BNF 内容特征。
            bnf = feature_matrix(
                # BNF 必须和上面的 mel 来自同一条语音。
                feature_path(data_dir, "bnf", source["speaker"], source["utt_id"]),
                # 要求 BNF 特征维度为登记的 144。
                PAPER_CONFIG.n_bnf,
            )
            # 按训练均值和标准差逐频带标准化，保持教师输入分布一致。
            mel_norm = ((mel - mean) / std).astype(np.float32)
            # 增加批维，形成 [1,80,T] 并放到计算设备。
            mel_tensor = torch.from_numpy(mel_norm).unsqueeze(0).to(device)
            # 将 BNF 时间轴对齐到当前 mel 的帧数。
            bnf_tensor = align_bnf_to_mel(
                # 先把 BNF 整理为 [1,144,T_bnf] 张量。
                torch.from_numpy(bnf).unsqueeze(0).to(device),
                # 目标时间长度由真实 mel 决定，不硬设与 BNF 原始长度相同。
                mel_tensor.shape[-1],
                # 这里是两组外部时间对齐的变化点。
                mode=mode,
                # mel 物理时间轴使用真实采样率。
                sample_rate=PAPER_CONFIG.sample_rate,
                # FFT 窗长影响 mel 帧中心位置。
                n_fft=PAPER_CONFIG.n_fft,
                # hop 决定相邻 mel 帧的时间间隔。
                hop_size=PAPER_CONFIG.hop_size,
                # BNF 原始帧步长为 10 毫秒，用于物理中心对齐。
                bnf_frame_shift_seconds=0.010,
            )

            # 当前源句需要生成的数量等于目标参考数。
            count = len(source_pairs)
            # 同一源 mel 扩成一批，内容固定，仅目标音色不同。
            mel_batch = mel_tensor.expand(count, -1, -1).contiguous()
            # BNF 同样扩批，与 mel 的配对顺序保持一致。
            bnf_batch = bnf_tensor.expand(count, -1, -1).contiguous()
            # 收集当前所有参考语音的说话人嵌入。
            speaker_vectors = []
            # 按照 source_pairs 顺序读取，防止目标音色与文件名串位。
            for pair in source_pairs:
                # 从已提取的 .npy 加载参考说话人向量。
                vector = np.load(
                    # 用统一特征路径函数定位目标音色向量。
                    feature_path(
                        # 从当前项目数据目录读取。
                        data_dir,
                        # 明确要读 spk_emb，不混用 mel 或 BNF。
                        "spk_emb",
                        # 使用目标参考说话人的目录。
                        pair.reference_speaker,
                        # 使用当前目标参考语音的编号。
                        pair.reference_utt,
                    ),
                    # 禁用 pickle，只读取普通数值数组。
                    allow_pickle=False,
                # 去掉多余的单例轴，将向量整理为一维。
                ).squeeze()
                # 要求嵌入维度与教师配置一致，不能擅自裁剪或补齐。
                if vector.shape != (PAPER_CONFIG.spk_emb_dim,):
                    # 形状不匹配就停止，避免给教师错误条件。
                    raise ValueError(
                        # 报出实际维度及对应参考语音。
                        f"Invalid speaker embedding {vector.shape}: {pair.reference_utt}"
                    )
                # 统一为 float32 并按当前配对顺序收集。
                speaker_vectors.append(vector.astype(np.float32))
            # 堆叠成 [目标数,嵌入维] 后放到设备。
            speaker_batch = torch.from_numpy(np.stack(speaker_vectors)).to(device)
            # 每个源句使用独立且可重复的随机生成器。
            generator = torch.Generator(device=device).manual_seed(
                # 种子由基础种子加源句索引确定，两种对齐模式重复同一规则。
                int(args.seed) + source_index
            )

            # 仅推理，关闭梯度及训练状态追踪以降低开销。
            with torch.inference_mode():
                # 用教师执行固定 30 步扩散转换。
                converted_norm = diffusion.sample(
                    # 传入同一个已加载教师网络。
                    model,
                    # 源 mel 决定加噪后的起始语音内容。
                    mel_batch,
                    # 每条参考嵌入决定该输出的目标音色。
                    speaker_batch,
                    # 对齐后的 BNF 提供语言内容条件。
                    bnf_batch,
                    # 固定 30 次反向步骤，与 VoiceGrad-30 名称对应。
                    num_steps=30,
                    # 从扩散时间步 950 开始，而非最高噪声的 1000。
                    start_step=950,
                    # 显式传入固定生成器，保证两种对齐方式共用噪声。
                    generator=generator,
                    # 先对源 mel 加噪作为起点，保留源语音结构。
                    diffuse_source=True,
                )
                # 把标准化预测恢复到声码器期望的原始 log-mel 尺度。
                converted_mel = (
                    # 先逐频带乘回训练标准差。
                    converted_norm * std_tensor.unsqueeze(0)
                    # 再加回训练均值；次序与输入标准化互为逆运算。
                    + mean_tensor.unsqueeze(0)
                )
                # 声码器输出 [批,1,采样点]，去掉单通道轴后转到 CPU 数组以保存。
                generated_audio = vocoder(converted_mel).squeeze(1).cpu().numpy()

            # 按相同配对顺序逐条保存生成音频。
            for pair, audio in zip(source_pairs, generated_audio):
                # 从配对记录取得应写入的目标文件。
                destination = Path(pair.output_wav)
                # 若这条音频已有效且没有覆盖授权，就保留原文件。
                if not args.overwrite and valid_existing_wav(destination):
                    # 更新跳过计数，避免与新生成量混在一起。
                    skipped += 1
                    # 继续下一配对，不覆盖已确认的波形。
                    continue
                # 保存前确认每个音频采样点都是有限数。
                if not np.isfinite(audio).all():
                    # 出现 NaN/Inf 立即报出配对号，拒绝发布坏 WAV。
                    raise ValueError(f"Generated waveform has NaN/Inf: pair {pair.pair_id}")
                # 采用原子写入，避免外部评价读取未完成的音频。
                atomic_write_wav(
                    # 写入当前配对指定的路径。
                    destination,
                    # 转换 float32 并裁剪到 WAV 合法幅度范围 [-1,1]。
                    np.clip(audio.astype(np.float32), -1.0, 1.0),
                    # 输出采样率与 mel 和声码器约定保持一致。
                    PAPER_CONFIG.sample_rate,
                )
                # 一条完整写出后才增加生成计数。
                generated += 1

        # 记录本模式的实际来源、参数和完成量，为后续比较保留边界。
        report = {
            # 明确是受控诊断，不能标成正式论文评估。
            "status": "controlled_diagnostic_not_formal_paper_evaluation",
            # 记录用途为重训前检查 BNF 对齐。
            "purpose": "BNF alignment A/B before retraining",
            # 实际样本选择来自这份服务器训练清单。
            "training_manifest": str(manifest),
            # 训练清单指纹绑定准确版本。
            "training_manifest_sha256": sha256_file(manifest),
            # 披露是否使用额外文本清单及其路径。
            "supplemental_text_manifest": str(text_manifest) if text_manifest else None,
            # 额外文本来源也要有独立指纹。
            "supplemental_text_manifest_sha256": (
                # 未提供补充清单时使用 null，避免虚构来源。
                sha256_file(text_manifest) if text_manifest else None
            ),
            # 保存当前外部时间对齐模式。
            "bnf_alignment": mode,
            # 保存独立的内部 BNF 下采样模式。
            "bnf_downsampling": args.bnf_downsampling,
            # 单独披露内部模式是否覆盖了原教师检查点配置。
            "bnf_downsampling_is_checkpoint_override": (
                # 本次实际下采样选项作为比较左侧。
                args.bnf_downsampling
                # 从教师保存的 run_config 读取原始训练选项。
                != teacher_checkpoint.get("run_config", {}).get(
                    # 旧检查点未登记时按明确的历史默认解释。
                    "bnf_downsampling", LEGACY_BNF_DOWNSAMPLING
                )
            ),
            # 本模式的总配对数，不等同于新生成数量。
            "pair_count": len(pairs),
            # 本次实际写出的 WAV 数量。
            "generated": generated,
            # 保留而未重新生成的有效 WAV 数量。
            "skipped_existing": skipped,
            # 完整列出源说话人，便于检查两组是否一致。
            "source_speakers": [row["speaker"] for row in sources],
            # 列出源语音编号，防止只同说话人却换了句子。
            "source_utterances": [row["utt_id"] for row in sources],
            # 完整列出目标参考说话人。
            "target_speakers": [row["speaker"] for row in references],
            # 列出实际提供音色的参考句编号。
            "target_utterances": [row["utt_id"] for row in references],
            # 声明按设计两模式复用输入与种子；应结合清单哈希核实。
            "same_inputs_and_seed_across_alignment_modes": True,
            # 明确每个源句的种子推导规则。
            "seed_rule": "base_seed + source_index",
            # 保存本次基础随机种子。
            "base_seed": int(args.seed),
            # 从实际教师检查点读取 epoch，不根据文件名猜。
            "teacher_epoch": int(teacher_checkpoint["epoch"]),
            # 记录真实设置的教师反向采样次数。
            "voicegrad_steps": 30,
            # 记录加噪起始时间步。
            "teacher_start_step": 950,
            # 保存教师权重的绝对路径。
            "teacher_checkpoint": str(teacher_path),
            # 保存该权重准确字节哈希。
            "teacher_checkpoint_sha256": teacher_hash,
            # 保存声码器权重路径。
            "vocoder_checkpoint": str(vocoder_path),
            # 保存声码器字节哈希。
            "vocoder_checkpoint_sha256": vocoder_hash,
            # 保存声码器配置路径。
            "vocoder_config": str(vocoder_config_path),
            # 保存声码器配置字节哈希。
            "vocoder_config_sha256": vocoder_config_hash,
            # 展开实际声码器运行配置，便于后续检查接口。
            "vocoder_runtime_config": vocoder_runtime_config,
            # 记录本模式配对 CSV 的位置。
            "pairs_csv": str(output_dir / "pairs.csv"),
            # 锁定配对 CSV 内容，评价时可以防止混用样本。
            "pairs_csv_sha256": sha256_file(output_dir / "pairs.csv"),
            # 保存当前模式实测耗时。
            "elapsed_seconds": time.time() - start_time,
        }
        # 完整推理结束后原子写出该模式报告。
        atomic_write_json(output_dir / "inference_report.json", report)
        # 在日志中打印同一份报告，便于远程核实。
        print(json.dumps(report, ensure_ascii=False, indent=2))


# 只有命令行直接执行时才开始此诊断，导入不触发推理。
if __name__ == "__main__":
    # 进入受控诊断主流程。
    main()
