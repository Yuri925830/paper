#!/usr/bin/env python3
# 从官方 VCTK 音频库存建立完整 110 人数据清单。
"""预处理第 00 步：扫描官方 VCTK 0.92，建立完整音频库存。

官方 0.92 发行包的推荐输入是
``wav48_silence_trimmed/<speaker>/<utt_id>_mic1.flac``。脚本默认只选择 mic1，
也可用 ``--microphone mic2`` 显式切换；为兼容旧版 VCTK，它仍接受
``wav48/<speaker>/<utt_id>.wav``。

库存以“存在音频”为准，而不是以“音频和文本都存在”为准。这个区别很重要：官方
0.92 的 p315 有音频、在 ``speaker-info.txt`` 中也有性别信息，但没有转写目录。
p315 因此必须进入 110 人库存，其 ``txt_path`` 和 ``text`` 留空。输出的
``vctk_all_pairs.csv`` 保留历史文件名和 ``wav48_path`` 列以兼容 stage 01，同时新增
gender/text 等自包含字段，后续划分不再需要从旧 ``all_manifest.csv`` 补信息。
"""

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# 定义终端参数，使数据目录、检查点和运行选项明确记录。
import argparse
# 按清单表头读取说话人、句子及文件路径，并写回实验索引。
import csv
# 读取项目目录设置，使用操作系统的原子替换和落盘接口。
import os
# 用路径对象定位数据、权重和报告，减少手工拼接目录的错误。
from pathlib import Path
# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Dict, List, Mapping, Tuple


# 环境变量或脚本位置确定的默认工程根目录。
DEFAULT_PROJECT_ROOT = Path(
    # 优先读取显式工程根目录；没有环境变量时回到本文件向上两级的仓库目录。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[1])
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()

# 自动选择官方 trimmed 音频目录的模式名。
AUDIO_LAYOUT_AUTO = "auto"
# VCTK 0.92 官方去长静音音频目录名称。
AUDIO_LAYOUT_TRIMMED = "wav48_silence_trimmed"
# 兼容旧版 VCTK 的 wav48 音频目录名称。
AUDIO_LAYOUT_LEGACY = "wav48"
# 脚本允许的自动、官方 trimmed 和旧版音频布局。
AUDIO_LAYOUTS: Tuple[str, ...] = (
    # 自动选择官方 trimmed 音频目录的模式名。
    AUDIO_LAYOUT_AUTO,
    # VCTK 0.92 官方去长静音音频目录名称。
    AUDIO_LAYOUT_TRIMMED,
    # 兼容旧版 VCTK 的 wav48 音频目录名称。
    AUDIO_LAYOUT_LEGACY,
)
# 官方双麦录音允许选择的 mic1/mic2 名称。
MICROPHONES: Tuple[str, ...] = ("mic1", "mic2")
# 原始库存 CSV 的固定列顺序。
INVENTORY_FIELDS = (
    # 此字段保存当前语音所属说话人。
    "speaker",
    # 此字段保存同时包含说话人和句号的样本编号。
    "utt_id",
    # 此字段保存官方元数据中的性别标记，仅用于数据核对。
    "gender",
    # 历史 stage 01 读取这个列名；值可以是官方 0.92 的 FLAC 路径。
    "wav48_path",
    # 此字段保存当前句子可用的转写文件路径，缺失时留空。
    "txt_path",
    # 此字段保存官方转写文字，教师训练不以此作为模型输入。
    "text",
    # 此字段保存本次库存实际采用的 VCTK 音频布局。
    "audio_layout",
    # 此字段保存本次唯一保留的麦克风编号，默认 mic1 是实现选择。
    "microphone",
)


# 说明并解析本脚本的输入目录、实验协议和可选运行参数；下面的参数分别限定输入数据和处理约束。
def parse_args() -> argparse.Namespace:
    # 说明并解析本脚本的输入目录、实验协议和可选运行参数。
    """定义命令行参数并返回解析结果。"""

    # 定义本脚本参数与帮助文字的解析器。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 指定本次工程根目录，数据与实验路径都据此解析。
        "--project-root",
        # 把命令行路径直接解析为 Path 对象，后续统一处理相对和绝对目录。
        type=Path,
        # 默认从工程根目录查找 VCTK 和元数据，允许命令显式改路径。
        default=DEFAULT_PROJECT_ROOT,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="仓库/实验目录；默认读取 VOICEGRAD_PROJECT_ROOT。",
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 显式指定官方 VCTK 数据包根目录。
        "--vctk-root",
        # 把命令行路径直接解析为 Path 对象，后续统一处理相对和绝对目录。
        type=Path,
        # 未显式指定时保留为空，由本函数随后确定默认路径或处理范围。
        default=None,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help=(
            # 帮助文字先说明此参数应指向官方数据包根目录。
            "VCTK-Corpus-0.92 根目录；默认 "
            # 帮助文字给出未指定参数时实际使用的默认子目录。
            "<project-root>/data/raw/VCTK-Corpus-0.92。"
        ),
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 显式指定本次结果目录，避免和其他实验共用输出。
        "--output-dir",
        # 把命令行路径直接解析为 Path 对象，后续统一处理相对和绝对目录。
        type=Path,
        # 未显式指定时保留为空，由本函数随后确定默认路径或处理范围。
        default=None,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="清单输出目录；默认 <project-root>/data/metadata。",
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 选择官方 trimmed 或历史 wav48 音频目录布局。
        "--audio-layout",
        # 音频布局只能选择 auto、官方 trimmed 或旧 wav48。
        choices=AUDIO_LAYOUTS,
        # 自动模式先尝试官方 VCTK 0.92 trimmed 目录，缺失时再尝试旧布局。
        default=AUDIO_LAYOUT_AUTO,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help=(
            # 帮助文字说明自动寻找音频布局的优先顺序。
            "auto 优先使用官方 wav48_silence_trimmed，找不到时退回 wav48；"
            # 帮助文字提示可以用命令行锁定布局，避免自动选择变化。
            "也可显式固定一种布局。"
        ),
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 选择 mic1 或 mic2，一次库存只保留同一路麦克风。
        "--microphone",
        # 每次库存只接受指定的一路麦克风，避免同一句双麦录音重复。
        choices=MICROPHONES,
        # 本次实现默认选择 mic1；论文未逐项交代麦克风时需在报告披露此选择。
        default="mic1",
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="官方双麦录音选择；论文复现默认 mic1。",
    )
    # 完成终端参数解析，返回经过类型和选项检查的运行设置。
    return parser.parse_args()


# 把完整清单写到临时文件后原子替换正式 CSV；下面的参数分别限定输入数据和处理约束。
def atomic_write_csv(
    # 当前待检查或读写的文件路径。
    path: Path,
    # 本次 CSV 输出的列名及固定顺序。
    fieldnames: List[str] | Tuple[str, ...],
    # 本次选定的语音清单。
    rows: List[Dict[str, object]],
# 本接口负责把完整清单写到临时文件后原子替换正式 CSV，处理结果通过对象状态或文件保存，不额外返回数值。
) -> None:
    # 把完整清单写到临时文件后原子替换正式 CSV。
    """先写临时文件再替换正式文件，避免中断后留下半张 CSV。"""

    # 尚未正式发布的临时文件。
    temporary = path.with_name(path.name + ".tmp")
    # 在受控文件句柄中完成把完整清单写到临时文件后原子替换正式 CSV，退出此段后自动关闭文件。
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        # 固定表头顺序的 CSV 写入器。
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # 先写列名，使后续行能按相同字段重新读取。
        writer.writeheader()
        # 把已整理的清单记录按固定字段顺序写入 CSV。
        writer.writerows(rows)
    # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
    temporary.replace(path)


# 优先识别官方裁静音 VCTK 布局并兼容旧版目录；下面的参数分别限定输入数据和处理约束。
def choose_audio_root(vctk_root: Path, requested_layout: str) -> Tuple[Path, str]:
    # 优先识别官方裁静音 VCTK 布局并兼容旧版目录。
    """选择官方 trimmed 布局或旧 wav48 布局，并返回实际选择。"""

    # 按命令行要求采用的音频目录布局决定是否进入此分支；判断条件为 requested_layout == AUDIO_LAYOUT_AUTO。
    if requested_layout == AUDIO_LAYOUT_AUTO:
        # 当前待按优先级检查的目录或音频候选项。
        candidates = (AUDIO_LAYOUT_TRIMMED, AUDIO_LAYOUT_LEGACY)
    # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
    else:
        # 当前待按优先级检查的目录或音频候选项。
        candidates = (requested_layout,)
    # 逐项遍历当前待按优先级检查的目录或音频候选项，保持每条记录都经过相同检查。
    for layout in candidates:
        # 当前检查的音频根目录候选路径。
        candidate = vctk_root / layout
        # 仅在当前检查的音频根目录候选路径已存在时进入复用或校验分支。
        if candidate.is_dir():
            # 返回实际存在的音频目录及布局名，报告据此记录真实来源。
            return candidate, layout
    # 已尝试寻找的所有音频目录，供缺失报错使用。
    searched = ", ".join(str(vctk_root / name) for name in candidates)
    # 优先识别官方裁静音 VCTK 布局并兼容旧版目录未通过，抛出下方包含具体路径或实际值的错误。
    raise FileNotFoundError(f"找不到 VCTK 音频目录；已检查：{searched}")


# 从官方说话人信息解析编号及性别；下面的参数分别限定输入数据和处理约束。
def read_gender_map(path: Path) -> Dict[str, str]:
    # 从官方说话人信息解析编号及性别。
    """从官方 speaker-info.txt 的 ID/AGE/GENDER 前三列读取性别。"""

    # 检查当前待检查或读写的文件路径是否缺失；缺失时不能继续信任该输入。
    if not path.is_file():
        # 从官方说话人信息解析编号及性别未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(f"找不到官方说话人信息：{path}")
    # 从官方说话人信息表读取的性别映射。
    genders: Dict[str, str] = {}
    # 逐行读取官方说话人元数据，同时保留行号供格式报错定位。
    for line_number, raw_line in enumerate(
        # 兼容 UTF-8 文件头标记，从第一行开始编号。
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        # 当前读取的一行文字。
        line = raw_line.strip()
        # 按当前读取的一行文字决定是否进入此分支；判断条件为 not line or line.startswith("#")。
        if not line or line.startswith("#"):
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue
        # 当前 CSV 表头或文本行拆出的字段。
        fields = line.split()
        # 按当前 CSV 表头或文本行拆出的字段决定是否进入此分支；判断条件为 fields[0].casefold() == "id"。
        if fields[0].casefold() == "id":
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue
        # The official VCTK 0.92 archive writes IDs as ``p225`` while some
        # community copies use ``225``.  Accept both without weakening the
        # numeric speaker-ID check.
        # 将官方说话人 ID 统一为小写，兼容 P225 与 p225 等文字形式。
        speaker_token = fields[0].casefold()
        # Official 0.92 includes ``s5`` as the 110th released speaker (109
        # ``pNNN`` directories plus ``s5``).  Preserve that identifier exactly;
        # excluding it would silently train on only 109 speakers / 44,057 files.
        # 保留旧版 VCTK 特殊说话人 s5，不强行把它解析成 p 加数字。
        if speaker_token == "s5":
            # 当前语音所属说话人。
            speaker = "s5"
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 去掉 p 前缀后的说话人数值编号。
            numeric_id = speaker_token[1:] if speaker_token.startswith("p") else speaker_token
            # 除特殊 s5 外，要求说话人编号为数字且至少有 ID、年龄、性别三列。
            if len(fields) < 3 or not numeric_id.isdigit():
                # 从官方说话人信息解析编号及性别未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"speaker-info.txt 第 {line_number} 行格式异常：{raw_line!r}")
            # 当前语音所属说话人。
            speaker = f"p{int(numeric_id):03d}"
        # 官方元数据中的性别标记，仅用于数据核对。
        gender = fields[2].upper()
        # 按官方元数据中的性别标记，仅用于数据核对决定是否进入此分支；判断条件为 gender not in {"F", "M", "U"}。
        if gender not in {"F", "M", "U"}:
            # 从官方说话人信息解析编号及性别未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"speaker-info.txt 中 {speaker} 性别异常：{gender!r}")
        # 按当前语音所属说话人决定是否进入此分支；判断条件为 speaker in genders and genders[speaker] != gender。
        if speaker in genders and genders[speaker] != gender:
            # 从官方说话人信息解析编号及性别未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"speaker-info.txt 中 {speaker} 性别冲突")
        # 登记官方性别信息，后续只用于数据一致性检查。
        genders[speaker] = gender
    # 按从官方说话人信息表读取的性别映射决定是否进入此分支；判断条件为 not genders。
    if not genders:
        # 从官方说话人信息解析编号及性别未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"没有从 {path} 读取到任何说话人")
    # 返回从官方说话人信息表读取的性别映射，供调用方继续处理。
    return genders


# 选择指定麦克风音频并核对说话人与文件编号；下面的参数分别限定输入数据和处理约束。
def selected_audio_files(
    # 当前说话人的原始音频目录。
    speaker_dir: Path,
    # 实际找到并使用的原始音频布局。
    layout: str,
    # 本次唯一保留的麦克风编号，默认 mic1 是实现选择。
    microphone: str,
# 返回选择指定麦克风音频并核对说话人与文件编号整理的 Dict[str, Path] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, Path]:
    # 选择指定麦克风音频并核对说话人与文件编号。
    """返回 ``utt_id -> 音频``；官方双麦文件只保留指定 microphone。"""

    # 当前 VCTK 布局允许扫描的音频后缀。
    extensions = ("*.flac", "*.wav") if layout == AUDIO_LAYOUT_TRIMMED else ("*.wav",)
    # 当前待按优先级检查的目录或音频候选项。
    candidates: List[Path] = []
    # 逐项遍历当前 VCTK 布局允许扫描的音频后缀，保持每条记录都经过相同检查。
    for pattern in extensions:
        # 收集当前说话人目录中符合当前后缀的录音文件。
        candidates.extend(speaker_dir.glob(pattern))

    # 按句子编号索引的单麦有效音频。
    selected: Dict[str, Path] = {}
    # 是否发现旧版无麦克风后缀的录音。
    saw_unsuffixed_legacy = False
    # 逐项遍历当前待按优先级检查的目录或音频候选项，保持每条记录都经过相同检查。
    for path in sorted(candidates):
        # 当前音频去掉扩展名后的文件名。
        stem = path.stem
        # 麦克风编号到文件名后缀的映射。
        mic_suffixes = {mic: f"_{mic}" for mic in MICROPHONES}
        # 从文件名后缀识别出的实际麦克风。
        matched_mic = next((mic for mic, suffix in mic_suffixes.items() if stem.endswith(suffix)), None)
        # 按从文件名后缀识别出的实际麦克风决定是否进入此分支；判断条件为 matched_mic is not None。
        if matched_mic is not None:
            # 按从文件名后缀识别出的实际麦克风决定是否进入此分支；判断条件为 matched_mic != microphone。
            if matched_mic != microphone:
                # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
                continue
            # 同时包含说话人和句号的样本编号。
            utt_id = stem[: -len(mic_suffixes[matched_mic])]
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 旧 wav48 是单麦且文件名没有 _mic1；把它视为 mic1 兼容输入。
            if layout != AUDIO_LAYOUT_LEGACY:
                # 选择指定麦克风音频并核对说话人与文件编号未通过，抛出下方包含具体路径或实际值的错误。
                raise RuntimeError(f"trimmed 音频缺少 _mic1/_mic2 后缀：{path}")
            # 是否发现旧版无麦克风后缀的录音。
            saw_unsuffixed_legacy = True
            # 按本次唯一保留的麦克风编号，默认 mic1 是实现选择决定是否进入此分支；判断条件为 microphone != "mic1"。
            if microphone != "mic1":
                # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
                continue
            # 同时包含说话人和句号的样本编号。
            utt_id = stem

        # 由说话人目录名构成的合法语音编号前缀。
        expected_prefix = f"{speaker_dir.name}_"
        # 文件名必须属于当前说话人目录，拒绝错放录音。
        if not utt_id.startswith(expected_prefix):
            # 选择指定麦克风音频并核对说话人与文件编号未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"说话人与文件名不一致：{path}")
        # 按同时包含说话人和句号的样本编号决定是否进入此分支；判断条件为 utt_id in selected。
        if utt_id in selected:
            # 选择指定麦克风音频并核对说话人与文件编号未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"选择 {microphone} 后仍有重复 utterance：{utt_id}")
        # 将唯一句子编号绑定到选定麦克风的音频路径。
        selected[utt_id] = path

    # 按实际找到并使用的原始音频布局决定是否进入此分支；判断条件为 layout == AUDIO_LAYOUT_LEGACY and microphone == "mic2" and saw_unsuffixed_legacy。
    if layout == AUDIO_LAYOUT_LEGACY and microphone == "mic2" and saw_unsuffixed_legacy:
        # 选择指定麦克风音频并核对说话人与文件编号未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("旧 wav48 单麦文件没有 mic2；请使用 --microphone mic1。")
    # 返回按句子编号索引的单麦有效音频，供调用方继续处理。
    return selected


# 读取官方转写文字并压平多余空白；下面的参数分别限定输入数据和处理约束。
def read_transcript(path: Path) -> str:
    # 读取官方转写文字并压平多余空白。
    """读取一条官方转写并压平多余空白。"""

    # 读取官方转写并把换行或连续空白合并为单个空格，不改写句子内容。
    return " ".join(path.read_text(encoding="utf-8-sig").split())


# 建立包含无转写音频在内的完整 VCTK 库存；下面的参数分别限定输入数据和处理约束。
def build_inventory(
    # 官方 VCTK 数据包根目录。
    vctk_root: Path,
    # 命令行要求采用的音频目录布局。
    requested_layout: str,
    # 本次唯一保留的麦克风编号，默认 mic1 是实现选择。
    microphone: str,
# 返回建立包含无转写音频在内的完整 VCTK 库存整理的 Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str], str] 结果，内部各项保持前面约定的对应关系。
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str], str]:
    # 建立包含无转写音频在内的完整 VCTK 库存。
    """扫描音频说话人，连同 gender/text 生成自包含库存。"""

    # 按请求选择实际音频布局，同时记录最后采用的目录名称。
    audio_root, layout = choose_audio_root(vctk_root, requested_layout)
    # 官方转写文本总目录。
    txt_root = vctk_root / "txt"
    # 检查官方转写文本总目录是否缺失；缺失时不能继续信任该输入。
    if not txt_root.is_dir():
        # 建立包含无转写音频在内的完整 VCTK 库存未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(f"找不到 VCTK 文本目录：{txt_root}")
    # 从官方说话人信息表读取的性别映射。
    genders = read_gender_map(vctk_root / "speaker-info.txt")

    # 按音频目录而非转写目录发现的完整说话人集合。
    audio_speakers = sorted(path.name for path in audio_root.iterdir() if path.is_dir())
    # 按按音频目录而非转写目录发现的完整说话人集合决定是否进入此分支；判断条件为 not audio_speakers。
    if not audio_speakers:
        # 建立包含无转写音频在内的完整 VCTK 库存未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"音频目录没有任何说话人：{audio_root}")
    # 存在音频但官方信息表中缺少性别记录的说话人。
    missing_gender = sorted(set(audio_speakers) - set(genders))
    # 按存在音频但官方信息表中缺少性别记录的说话人决定是否进入此分支；判断条件为 missing_gender。
    if missing_gender:
        # 建立包含无转写音频在内的完整 VCTK 库存未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"speaker-info.txt 缺少音频说话人：{missing_gender}")

    # 训练及测试实际使用的全部记录。
    all_rows: List[Dict[str, object]] = []
    # 按说话人分组的语音清单。
    speaker_rows: List[Dict[str, object]] = []
    # 逐项遍历按音频目录而非转写目录发现的完整说话人集合，保持每条记录都经过相同检查。
    for speaker in audio_speakers:
        # 当前说话人按句号索引的选定麦克风音频。
        audio_by_utt = selected_audio_files(audio_root / speaker, layout, microphone)
        # 按当前说话人按句号索引的选定麦克风音频决定是否进入此分支；判断条件为 not audio_by_utt。
        if not audio_by_utt:
            # 建立包含无转写音频在内的完整 VCTK 库存未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"{speaker} 没有选中的 {microphone} 音频")
        # 当前说话人的转写目录，p315 可能没有此目录。
        txt_dir = txt_root / speaker
        # 当前说话人实际存在的转写文件。
        txt_files = sorted(txt_dir.glob("*.txt")) if txt_dir.is_dir() else []
        # 按句号索引的官方转写文件。
        txt_by_utt: Mapping[str, Path] = {path.stem: path for path in txt_files}
        # 同时有音频和转写的句子编号集合。
        paired = set(audio_by_utt) & set(txt_by_utt)

        # 保存当前说话人的音频、转写和缺转写数量。
        speaker_rows.append(
            # 下面展开建立包含无转写音频在内的完整 VCTK 库存所需的这组条目；各字段随后逐一说明。
            {
                # 记录当前语音所属说话人。
                "speaker": speaker,
                # 记录官方元数据中的性别标记，仅用于数据核对。
                "gender": genders[speaker],
                # 记录当前说话人选定麦克风的音频总数。
                "n_audio": len(audio_by_utt),
                # 兼容旧审计消费者；这里表示选中音频数，即使源文件是 FLAC。
                "n_wav": len(audio_by_utt),
                # 记录当前说话人实际存在的转写文件数。
                "n_txt": len(txt_files),
                # 记录同时有音频与转写的条目数。
                "n_pairs": len(paired),
                # 记录有音频但缺少转写的条目数，仍保留在库存中。
                "n_missing_txt": len(audio_by_utt) - len(paired),
            }
        )
        # 按句号稳定排序所有选中音频，库存由音频存在性决定。
        for utterance_id, audio_path in sorted(audio_by_utt.items()):
            # 当前句子可用的转写文件路径，缺失时留空。
            txt_path = txt_by_utt.get(utterance_id)
            # 即使转写缺失也保留该音频记录，确保 p315 等说话人不被误删。
            all_rows.append(
                # 下面展开建立包含无转写音频在内的完整 VCTK 库存所需的这组条目；各字段随后逐一说明。
                {
                    # 记录当前语音所属说话人。
                    "speaker": speaker,
                    # 记录同时包含说话人和句号的样本编号。
                    "utt_id": utterance_id,
                    # 记录官方元数据中的性别标记，仅用于数据核对。
                    "gender": genders[speaker],
                    # 记录库存中记录的原始高采样率音频路径。
                    "wav48_path": str(audio_path.resolve()),
                    # 记录当前句子可用的转写文件路径，缺失时留空。
                    "txt_path": str(txt_path.resolve()) if txt_path is not None else "",
                    # 记录官方转写文字，教师训练不以此作为模型输入。
                    "text": read_transcript(txt_path) if txt_path is not None else "",
                    # 记录本次库存实际采用的 VCTK 音频布局。
                    "audio_layout": layout,
                    # 记录本次唯一保留的麦克风编号，默认 mic1 是实现选择。
                    "microphone": microphone,
                }
            )
    # 共同返回全量库存、逐说话人计数、说话人列表和实际音频布局。
    return all_rows, speaker_rows, audio_speakers, layout


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main() -> None:
    # 串联本文件的数据检查、模型处理和结果保存流程。
    """扫描目录、写清单并打印可审计摘要。"""

    # 终端传入并经解析的运行设置。
    args = parse_args()
    # 保存代码、数据与检查点的项目目录。
    project_root = args.project_root.expanduser().resolve()
    # 官方 VCTK 数据包根目录。
    vctk_root = (
        # 显式指定数据包路径时，展开用户目录并转为绝对位置。
        args.vctk_root.expanduser().resolve()
        # 按官方 VCTK 数据包根目录决定是否进入此分支；判断条件为 args.vctk_root is not None。
        if args.vctk_root is not None
        # 未指定数据包路径时，使用工程下的官方 VCTK 0.92 默认目录。
        else project_root / "data" / "raw" / "VCTK-Corpus-0.92"
    )
    # 本次运行的输出目录。
    output_dir = (
        # 显式输出目录展开后转换为绝对路径。
        args.output_dir.expanduser().resolve()
        # 按本次运行的输出目录决定是否进入此分支；判断条件为 args.output_dir is not None。
        if args.output_dir is not None
        # 未指定输出目录时，把库存存到统一元数据目录。
        else project_root / "data" / "metadata"
    )
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描并核对整份数据包，取得四类后续保存的信息。
    all_rows, speaker_rows, speakers, layout = build_inventory(
        # 官方 VCTK 数据包根目录。
        vctk_root,
        # 传入本次用户选择的音频目录布局。
        args.audio_layout,
        # 传入本次唯一使用的麦克风编号。
        args.microphone,
    )
    # 保存全部音频库存的 CSV 路径。
    all_csv = output_dir / "vctk_all_pairs.csv"
    # 逐说话人音频与转写计数表路径。
    speaker_csv = output_dir / "vctk_speaker_counts.csv"
    # 按固定列顺序原子发布完整库存，后续重采样以此为依据。
    atomic_write_csv(all_csv, INVENTORY_FIELDS, all_rows)
    # 另存逐说话人计数 CSV，便于检查 110 人是否完整。
    atomic_write_csv(
        # 逐说话人音频与转写计数表路径。
        speaker_csv,
        # 计数表按说话人、官方性别、音频数、转写数及缺失量固定排序。
        ("speaker", "gender", "n_audio", "n_wav", "n_txt", "n_pairs", "n_missing_txt"),
        # 按说话人分组的语音清单。
        speaker_rows,
    )

    # 完整说话人编号文本清单路径。
    speakers_txt = output_dir / "vctk_all_speakers.txt"
    # 尚未原子发布的说话人清单临时文件。
    temporary_txt = speakers_txt.with_name(speakers_txt.name + ".tmp")
    # 每行写一个说话人编号到临时文本，完整写完再替换正式清单。
    temporary_txt.write_text("\n".join(speakers) + "\n", encoding="utf-8")
    # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
    temporary_txt.replace(speakers_txt)

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print("===== VCTK 清单摘要 =====")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"音频布局：{layout}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"microphone：{args.microphone}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"音频说话人：{len(speakers)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"选中音频总数：{len(all_rows)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"缺少转写的音频：{sum(not str(row['txt_path']) for row in all_rows)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"已保存：{all_csv}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"已保存：{speaker_csv}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"已保存：{speakers_txt}")


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 执行本文件的参数检查及处理入口，错误会返回到终端。
    main()
