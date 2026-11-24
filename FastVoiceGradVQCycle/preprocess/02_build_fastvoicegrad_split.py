#!/usr/bin/env python3
# 构建测试网格相同、公共训练文本去留不同的 seen/unseen 数据划分。
"""为 FastVoiceGrad 构造两种可直接比较的 VCTK 数据划分。

先用一句大白话概括本文件：

* ``text-unseen``：测试说话人没见过，测试句子也没见过；
* ``text-seen``：测试说话人没见过，但测试句子在其他训练说话人那里见过。

两种协议都固定使用同一批 10 位测试说话人、同一组 001--010 句子，并且都没有
validation 集。这样比较两组结果时，唯一主动改变的因素就是“测试文本是否曾在
训练集中出现”，不会把说话人、测试句子数或模型结构一起换掉。

共同规则：

* test = 10 位固定未见说话人 × VCTK 公共句子 001--010，共 100 条；
* 这 10 位说话人的 011 及以后语音不进入任何集合；
* 另外 100 位说话人属于 train；
* 不创建 validation，因为论文教师模型协议只说明了 train/test。

差异规则：

* ``text-unseen`` 会从 100 位训练说话人中删除 001--010，因此测试文本未见；
* ``text-seen`` 会保留训练说话人的 001--010，因此测试文本已见。

stage 00 的 ``vctk_all_pairs.csv`` 已自包含 gender/text/txt_path，并覆盖官方 0.92
全部 110 位音频说话人（包括没有转写的 p315）；stage 01 的
``vctk_wav22k_manifest.csv`` 只负责证明对应的 22.05 kHz 音频已成功生成。因此本脚本
不读取也不依赖任何旧 ``all_manifest.csv``。模型训练本身不读取文本，真正的数据
泄漏边界由 VCTK 句号 001--010 决定。
"""

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# 定义终端参数，使数据目录、检查点和运行选项明确记录。
import argparse
# 按清单表头读取说话人、句子及文件路径，并写回实验索引。
import csv
# 保存或读取实验设置、数据指纹及处理进度的结构化记录。
import json
# 读取项目目录设置，使用操作系统的原子替换和落盘接口。
import os
# 复制已经保存的清单或统计文件，用于备份及可恢复发布。
import shutil
# 统计每位说话人的样本数量，核对训练与测试覆盖范围。
from collections import Counter
# 生成 UTC 时间标记，区分不同次数据划分和发布记录。
from datetime import datetime, timezone
# 用路径对象定位数据、权重和报告，减少手工拼接目录的错误。
from pathlib import Path
# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


# 优先读取环境变量；没有设置时，用本脚本父目录的父目录（也就是仓库根目录）。
# ``expanduser`` 会展开用户输入的 ``~``，``resolve`` 会整理成绝对路径。
DEFAULT_PROJECT_ROOT = Path(
    # 优先读取 VOICEGRAD_PROJECT_ROOT 指定的工程位置。
    os.environ.get(
        # 用于显式指定工程根目录的环境变量名称。
        "VOICEGRAD_PROJECT_ROOT",
        # 没有环境变量时，默认使用本脚本上两级的仓库目录。
        Path(__file__).resolve().parents[1],
    )
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()

# argparse 的 choices 会拒绝拼错的协议名，防止本想跑 text-unseen 却静默跑成另一套。
PROTOCOL_TEXT_UNSEEN = "text-unseen"
# 测试句子在其他训练说话人中出现过的协议名称。
PROTOCOL_TEXT_SEEN = "text-seen"
# 允许的文本未见和文本已见两种数据划分。
PROTOCOL_CHOICES: Tuple[str, ...] = (PROTOCOL_TEXT_UNSEEN, PROTOCOL_TEXT_SEEN)

# 官方 VCTK 0.92 完整音频库存有 110 位说话人：100 位训练，10 位测试。
EXPECTED_TOTAL_SPEAKERS = 110
# 按论文人数设定要求训练集覆盖 100 位说话人。
EXPECTED_TRAIN_SPEAKERS = 100
# 按论文人数设定要求留出 10 位完全未见的测试说话人。
EXPECTED_TEST_SPEAKERS = 10

# 这 10 位说话人都完整拥有 001--010。FastVoiceGrad 官方演示直接出现其中 8 位；
# 后续 FasterVoiceGrad 使用相同的 10 人×10 句 VCTK 协议、实现沿用 FastVoiceGrad，
# 并在与 FastVoiceGrad 的配对比较演示中补出 p243 与 p303。两页文件名的并集恰为
# 这 10 位。论文仍未公开私有 split 文件，所以这里只能标成高置信官方证据重建，
# 不能声称作者在正文逐项打印过这份名单。
TEST_SPEAKERS: Tuple[str, ...] = (
    # 固定测试说话人 p238；其所有语音均不能进入模型训练。
    "p238",
    # 固定测试说话人 p241；其所有语音均不能进入模型训练。
    "p241",
    # 固定测试说话人 p243；其所有语音均不能进入模型训练。
    "p243",
    # 固定测试说话人 p252；其所有语音均不能进入模型训练。
    "p252",
    # 固定测试说话人 p261；其所有语音均不能进入模型训练。
    "p261",
    # 固定测试说话人 p294；其所有语音均不能进入模型训练。
    "p294",
    # 固定测试说话人 p303；其所有语音均不能进入模型训练。
    "p303",
    # 固定测试说话人 p330；其所有语音均不能进入模型训练。
    "p330",
    # 固定测试说话人 p334；其所有语音均不能进入模型训练。
    "p334",
    # 固定测试说话人 p360；其所有语音均不能进入模型训练。
    "p360",
)
# 本次固定的十个公共测试文本编号。
TEST_SENTENCE_IDS: Tuple[str, ...] = tuple(f"{index:03d}" for index in range(1, 11))

# 每一行 CSV 都按下面顺序写列。元组用圆括号表示不可随意增删的有序序列。
MANIFEST_FIELDS = (
    # 此字段保存训练或测试集合的名称。
    "split",
    # 此字段保存当前语音所属说话人。
    "speaker",
    # 此字段保存同时包含说话人和句号的样本编号。
    "utt_id",
    # 此字段保存官方元数据中的性别标记，仅用于数据核对。
    "gender",
    # 此字段保存已经重采样到 22.05 kHz 的音频路径。
    "wav22k_path",
    # 此字段保存当前句子可用的转写文件路径，缺失时留空。
    "txt_path",
    # 此字段保存官方转写文字，教师训练不以此作为模型输入。
    "text",
    # 此字段保存测试说话人是否从训练中完全排除的标记。
    "is_unseen_speaker",
)


# 说明并解析本脚本的输入目录、实验协议和可选运行参数；下面的参数分别限定输入数据和处理约束。
def parse_args() -> argparse.Namespace:
    # 说明并解析本脚本的输入目录、实验协议和可选运行参数。
    """把终端命令转换为 Python 对象，返回值例如 ``args.protocol``。"""

    # 定义本脚本参数与帮助文字的解析器。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 选择公共测试文本是否允许进入其他人的训练语音。
        "--protocol",
        # 命令只能选择明确实现的 text-unseen 或 text-seen 协议。
        choices=PROTOCOL_CHOICES,
        # 默认从训练说话人中剔除公共测试文本，采用文本未见协议。
        default=PROTOCOL_TEXT_UNSEEN,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help=(
            # 帮助文字说明文本未见模式会剔除训练人的公共测试句。
            "text-unseen 删除训练人的 001--010；"
            # 帮助文字说明文本已见模式保留公共句，默认选择文本未见。
            "text-seen 保留这些公共句子。默认使用更严格的 text-unseen。"
        ),
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 只有显式使用此开关才跳过旧清单备份，默认会保留历史划分。
        "--no-backup",
        # 该文件本轮为新生成还是经过验证后复用。
        action="store_true",
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="Do not archive the currently active split before replacing it.",
    )
    # 完成终端参数解析，返回经过类型和选项检查的运行设置。
    return parser.parse_args()


# 读取清单并检查缺少的必需列及空文件；下面的参数分别限定输入数据和处理约束。
def read_csv(path: Path, required: Iterable[str]) -> List[Dict[str, str]]:
    # 读取清单并检查缺少的必需列及空文件。
    """读取一个 CSV，并在真正处理前检查必需列和空文件。

    ``required`` 是必须存在的列名集合；返回值是“字典组成的列表”，其中每个字典
    就是一行，例如 ``row["speaker"] == "p238"``。
    """

    # 检查当前待检查或读写的文件路径是否缺失；缺失时不能继续信任该输入。
    if not path.is_file():
        # 读取清单并检查缺少的必需列及空文件未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(path)
    # 在受控文件句柄中完成读取清单并检查缺少的必需列及空文件，退出此段后自动关闭文件。
    with path.open("r", encoding="utf-8", newline="") as handle:
        # 按表头返回字典的 CSV 读取器。
        reader = csv.DictReader(handle)
        # 当前 CSV 表头或文本行拆出的字段。
        fields = set(reader.fieldnames or [])
        # 检查中发现的缺失项目。
        missing = set(required) - fields
        # 按检查中发现的缺失项目决定是否进入此分支；判断条件为 missing。
        if missing:
            # 读取清单并检查缺少的必需列及空文件未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        # 本次选定的语音清单。
        rows = [dict(row) for row in reader]
    # 按本次选定的语音清单决定是否进入此分支；判断条件为 not rows。
    if not rows:
        # 读取清单并检查缺少的必需列及空文件未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Empty CSV: {path}")
    # 返回本次选定的语音清单，供调用方继续处理。
    return rows


# 用说话人和完整句号定位清单中的唯一录音；下面的参数分别限定输入数据和处理约束。
def key(row: Mapping[str, str]) -> Tuple[str, str]:
    # 用说话人和完整句号定位清单中的唯一录音。
    """用 ``(speaker, utt_id)`` 组成一条语音的全局唯一身份。"""

    # 返回说话人和语句编号组成的唯一键，用于跨清单匹配。
    return str(row["speaker"]), str(row["utt_id"])


# 从样本编号提取句号并拒绝非法命名；下面的参数分别限定输入数据和处理约束。
def sentence_id(utt_id: str) -> str:
    # 从样本编号提取句号并拒绝非法命名。
    """从 ``p238_001`` 中取出 ``001``，同时拒绝格式异常的文件名。"""

    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 从最后一个下划线拆出说话人前缀和三位句号。
        speaker, slot = utt_id.rsplit("_", 1)
    # 捕获从样本编号提取句号并拒绝非法命名中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except ValueError as error:
        # 从样本编号提取句号并拒绝非法命名未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Invalid VCTK utterance id: {utt_id!r}") from error
    # 语音编号前缀符合 p 加数字或历史 s5 的格式标志。
    valid_speaker = speaker == "s5" or (
        # 普通 VCTK 说话人编号需由 p 加数字构成。
        speaker.startswith("p") and speaker[1:].isdigit()
    )
    # 按语音编号前缀符合 p 加数字或历史 s5 的格式标志决定是否进入此分支；判断条件为 not valid_speaker or len(slot) != 3 or not slot.isdigit()。
    if not valid_speaker or len(slot) != 3 or not slot.isdigit():
        # 从样本编号提取句号并拒绝非法命名未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Invalid VCTK utterance id: {utt_id!r}")
    # 返回语音编号中的三位公共文本句号，供调用方继续处理。
    return slot


# 统一空格与大小写以核对文本重叠情况；下面的参数分别限定输入数据和处理约束。
def normalise_text(text: str) -> str:
    # 统一空格与大小写以核对文本重叠情况。
    """统一大小写和多余空格，仅用于检查同一句是否被写成了不同文本。"""

    # 仅为文本一致性检查统一大小写与空白，不改变保存的官方原文。
    return " ".join(text.casefold().split())


# 构造无重复身份的清单索引；下面的参数分别限定输入数据和处理约束。
def unique_rows(rows: Sequence[Dict[str, str]], source: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    # 构造无重复身份的清单索引。
    """把行列表变成按唯一身份索引的字典，并提前阻止重复行。"""

    # 当前函数生成的结果。
    result: Dict[Tuple[str, str], Dict[str, str]] = {}
    # 逐项遍历本次选定的语音清单，保持每条记录都经过相同检查。
    for row in rows:
        # 由说话人和句号构成的索引键。
        row_key = key(row)
        # 按由说话人和句号构成的索引键决定是否进入此分支；判断条件为 row_key in result。
        if row_key in result:
            # 构造无重复身份的清单索引未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Duplicate {row_key} in {source}")
        # 按由说话人和句号构成的索引键决定是否进入此分支；判断条件为 row_key[0] not in row_key[1]。
        if row_key[0] not in row_key[1]:
            # 构造无重复身份的清单索引未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Speaker/utterance mismatch in {source}: {row_key}")
        # 在建立索引前验证每条语音的句号格式是否合法。
        sentence_id(row_key[1])
        # 将这条唯一记录放入索引，后续按同一身份合并音频和文本。
        result[row_key] = row
    # 返回当前函数生成的结果，供调用方继续处理。
    return result


# 把完整清单写到临时文件后原子替换正式 CSV；下面的参数分别限定输入数据和处理约束。
def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    # 把完整清单写到临时文件后原子替换正式 CSV。
    """先写临时文件再替换正式文件，避免中断后留下“半张 CSV”。"""

    # 尚未正式发布的临时文件。
    temporary = path.with_name(path.name + ".tmp")
    # 在受控文件句柄中完成把完整清单写到临时文件后原子替换正式 CSV，退出此段后自动关闭文件。
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        # 固定表头顺序的 CSV 写入器。
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        # 先写列名，使后续行能按相同字段重新读取。
        writer.writeheader()
        # 把已整理的清单记录按固定字段顺序写入 CSV。
        writer.writerows(rows)
    # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
    temporary.replace(path)


# 通过临时文件发布完整文本记录；下面的参数分别限定输入数据和处理约束。
def atomic_write_text(path: Path, text: str) -> None:
    # 通过临时文件发布完整文本记录。
    """原子写普通文本；原理与上面的 CSV 写法相同。"""

    # 尚未正式发布的临时文件。
    temporary = path.with_name(path.name + ".tmp")
    # 先把完整文本写到临时文件，再由下一行原子替换。
    temporary.write_text(text, encoding="utf-8")
    # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
    temporary.replace(path)


# 在改划分前保存当前清单快照；下面的参数分别限定输入数据和处理约束。
def archive_active_split(metadata: Path) -> Path | None:
    # 在改划分前保存当前清单快照。
    """把当前活动清单备份到带 UTC 时间戳的目录，便于回退和审计。"""

    # 更换数据划分前需要归档的清单和协议文件名。
    filenames = (
        # 备份决定参数学习样本的训练清单。
        "train_manifest.csv",
        # 备份固定未见说话人测试清单。
        "test_manifest.csv",
        # 备份训练和测试合并的特征提取清单。
        "all_manifest.csv",
        # 备份本次固定测试说话人的编号列表。
        "test_speakers.txt",
        # 备份本次固定测试公共句号列表。
        "test_sentence_ids.txt",
        # 备份公共测试句号与官方原文对照。
        "test_sentences.csv",
        # 备份 seen/unseen 划分规则及具体 ID 的来源披露。
        "split_protocol.json",
        # 若存在历史验证集清单也先归档，当前协议不再沿用它。
        "val_manifest.csv",
    )
    # 当前磁盘上实际存在、需要先备份的旧清单。
    existing = [metadata / filename for filename in filenames if (metadata / filename).is_file()]
    # 按当前磁盘上实际存在、需要先备份的旧清单决定是否进入此分支；判断条件为 not existing。
    if not existing:
        # 返回空结果，表示当前没有可用的归档或条目。
        return None
    # UTC 格式的备份目录时间戳。
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # 原子发布后的正式文件路径。
    destination = metadata / "split_history" / timestamp
    # 准备在改划分前保存当前清单快照所需的目标目录，已有目录按给定选项复用。
    destination.mkdir(parents=True, exist_ok=False)
    # 逐项遍历当前磁盘上实际存在、需要先备份的旧清单，保持每条记录都经过相同检查。
    for source in existing:
        # 复制原清单及其文件元数据到时间戳备份目录，保留回退依据。
        shutil.copy2(source, destination / source.name)
    # 返回原子发布后的正式文件路径，供调用方继续处理。
    return destination


# 合并原始库存信息与重采样后的音频路径；下面的参数分别限定输入数据和处理约束。
def build_row(
    # 训练或测试集合的名称。
    split: str,
    # 重采样清单中证明音频已成功生成的记录。
    wav_row: Mapping[str, str],
    # 原始库存中提供性别、文本和源路径的记录。
    inventory_row: Mapping[str, str],
# 返回合并原始库存信息与重采样后的音频路径整理的 Dict[str, object] 结果，内部各项保持前面约定的对应关系。
) -> Dict[str, object]:
    # 合并原始库存信息与重采样后的音频路径。
    """把重采样路径和 stage-00 自包含库存合并为最终清单的一行。"""

    # 返回合并原始库存信息与重采样后的音频路径得到的完整字段集合，各字段用途在下面逐项说明。
    return {
        # 记录训练或测试集合的名称。
        "split": split,
        # 记录当前语音所属说话人。
        "speaker": wav_row["speaker"],
        # 记录同时包含说话人和句号的样本编号。
        "utt_id": wav_row["utt_id"],
        # 记录官方元数据中的性别标记，仅用于数据核对。
        "gender": inventory_row["gender"].upper(),
        # 记录已经重采样到 22.05 kHz 的音频路径。
        "wav22k_path": wav_row["wav22k_path"],
        # 记录当前句子可用的转写文件路径，缺失时留空。
        "txt_path": inventory_row.get("txt_path", ""),
        # 记录官方转写文字，教师训练不以此作为模型输入。
        "text": inventory_row.get("text", "").strip(),
        # 记录测试说话人是否从训练中完全排除的标记。
        "is_unseen_speaker": int(split == "test"),
    }


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main() -> None:
    # 串联本文件的数据检查、模型处理和结果保存流程。
    """执行完整的读取、核对、划分、备份、写入和审计报告流程。"""

    # 终端传入并经解析的运行设置。
    args = parse_args()
    # 本次实验的项目根目录。
    root = args.project_root.resolve()
    # 训练与测试清单所在目录。
    metadata = root / "data" / "metadata"
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    metadata.mkdir(parents=True, exist_ok=True)

    # stage 01 生成的全量有效 22.05 kHz 音频清单。
    full_wav_path = metadata / "vctk_wav22k_manifest.csv"
    # stage 00 生成的自包含原始音频和转写库存。
    full_inventory_path = metadata / "vctk_all_pairs.csv"

    # 重采样完成的全部音频记录。
    wav_rows = read_csv(full_wav_path, {"speaker", "utt_id", "wav22k_path"})
    # 包含性别和转写信息的全部原始库存记录。
    inventory_rows = read_csv(
        # stage 00 生成的自包含原始音频和转写库存。
        full_inventory_path,
        # 原始库存必须自带身份、性别、文本路径和文本列，不能从旧实验偷偷补字段。
        {"speaker", "utt_id", "gender", "txt_path", "text"},
    )

    # 按语音唯一键索引的重采样音频清单。
    wav_by_key = unique_rows(wav_rows, full_wav_path)
    # 按同一语音唯一键索引的原始库存。
    inventory_by_key = unique_rows(inventory_rows, full_inventory_path)
    # 原始库存与重采样成功清单必须逐语音一致，少转换一条也先报错。
    if set(wav_by_key) != set(inventory_by_key):
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Stage-00 and stage-01 full utterance inventories differ.")

    # 重采样库存覆盖的全部说话人。
    all_speakers = sorted({speaker for speaker, _ in wav_by_key})
    # 要求官方库存覆盖 110 位音频说话人，避免因缺转写误删 p315。
    if len(all_speakers) != EXPECTED_TOTAL_SPEAKERS:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 报错同时列出库存真实人数和预期 110 人，便于定位缺失。
            f"Full inventory has {len(all_speakers)} speakers; expected {EXPECTED_TOTAL_SPEAKERS}."
        )
    # 固定测试说话人中库存尚未覆盖的编号。
    missing_test_speakers = sorted(set(TEST_SPEAKERS) - set(all_speakers))
    # 按固定测试说话人中库存尚未覆盖的编号决定是否进入此分支；判断条件为 missing_test_speakers。
    if missing_test_speakers:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Test speakers are absent: {missing_test_speakers}")

    # 从官方说话人信息表读取的性别映射。
    genders: Dict[str, str] = {}
    # 逐项遍历包含性别和转写信息的全部原始库存记录，保持每条记录都经过相同检查。
    for row in inventory_rows:
        # 当前语音所属说话人。
        speaker = row["speaker"]
        # 官方元数据中的性别标记，仅用于数据核对。
        gender = row["gender"].upper()
        # 按官方元数据中的性别标记，仅用于数据核对决定是否进入此分支；判断条件为 gender not in {"F", "M", "U"}。
        if gender not in {"F", "M", "U"}:
            # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Invalid gender for {speaker}: {gender!r}")
        # 按当前语音所属说话人决定是否进入此分支；判断条件为 speaker in genders and genders[speaker] != gender。
        if speaker in genders and genders[speaker] != gender:
            # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Conflicting genders for {speaker}")
        # 把核对无冲突的官方性别绑定到说话人编号。
        genders[speaker] = gender
    # 每位有音频的说话人都应有官方性别元数据。
    if set(genders) != set(all_speakers):
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Existing manifest does not cover gender metadata for all speakers.")

    # 允许出现在参数训练中的说话人集合。
    train_speakers = sorted(set(all_speakers) - set(TEST_SPEAKERS))
    # 排除固定十位测试说话人后必须恰好剩 100 位训练人。
    if len(train_speakers) != EXPECTED_TRAIN_SPEAKERS:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Expected 100 training speakers, got {len(train_speakers)}")

    # 在碰任何活动清单前，先确认 10×10 测试网格真实存在。集合推导式会生成
    # (p238, p238_001) 这类应有键；缺一个都立即停止。
    expected_test_keys = {
        # 将测试说话人与公共句号组合成一条必须存在的测试语音身份。
        (speaker, f"{speaker}_{slot}")
        # 覆盖本次固定的十位测试说话人；具体 ID 来自官方演示重建，论文未逐项列出。
        for speaker in TEST_SPEAKERS
        # 覆盖 001—010 十个公共文本编号，形成完整十人×十句网格。
        for slot in TEST_SENTENCE_IDS
    }
    # 固定测试网格中缺失的语音身份。
    missing_test_keys = sorted(expected_test_keys - set(wav_by_key))
    # 按固定测试网格中缺失的语音身份决定是否进入此分支；判断条件为 missing_test_keys。
    if missing_test_keys:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Incomplete fixed 10x10 test grid: {missing_test_keys}")
    # 固定测试语音中缺少官方转写的条目。
    missing_test_text = sorted(
        # 保留缺失转写的具体语音键，报错时可直接定位。
        row_key
        # 逐条检查固定 100 条测试语音。
        for row_key in expected_test_keys
        # 测试转写为空时列为缺失，因为后续 CER 需要可靠原文。
        if not inventory_by_key.get(row_key, {}).get("text", "").strip()
    )
    # 按固定测试语音中缺少官方转写的条目决定是否进入此分支；判断条件为 missing_test_text。
    if missing_test_text:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Test transcripts are unavailable: {missing_test_text}")

    # 十个公共测试句号对应的转写文字。
    prompt_texts: Dict[str, str] = {}
    # 逐项遍历本次固定的十个公共测试文本编号，保持每条记录都经过相同检查。
    for slot in TEST_SENTENCE_IDS:
        # 同一公共句号在各测试说话人中出现的规范化转写集合。
        texts = {
            # 取同一公共句号在每位测试人中的转写，并统一大小写及空白后比较。
            normalise_text(inventory_by_key[(speaker, f"{speaker}_{slot}")]["text"])
            # 覆盖本次固定的十位测试说话人；具体 ID 来自官方演示重建，论文未逐项列出。
            for speaker in TEST_SPEAKERS
        }
        # 同一个公共句号必须对应同一句文字，跨说话人有差异时停止划分。
        if len(texts) != 1:
            # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Test slot {slot} is not the same sentence for all speakers: {texts}")
        # 跨说话人转写一致后选作展示的一份原始文字。
        representative = inventory_by_key[
            # 文本一致性通过后，用固定第一位测试人的原文作为该句展示文本。
            (TEST_SPEAKERS[0], f"{TEST_SPEAKERS[0]}_{slot}")
        # 取官方原文并去掉首尾空白，保留原有大小写和标点。
        ]["text"].strip()
        # 保存公共句号到原文的映射，后续评估按这份文本核对内容。
        prompt_texts[slot] = representative
    # 十个测试句号还必须对应十句不同文字，防止重复文本降低实际覆盖。
    if len({normalise_text(text) for text in prompt_texts.values()}) != len(TEST_SENTENCE_IDS):
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("The ten test slots do not contain ten distinct prompts.")

    # 下面两个列表先只存在内存里；所有检查通过以后才一次性写到磁盘。
    train_rows: List[Dict[str, object]] = []
    # 只用于评估的未见说话人语音记录。
    test_rows: List[Dict[str, object]] = []
    # text-unseen 中从训练说话人排除的公共测试句数。
    excluded_train_prompts = 0
    # 测试说话人 011 以后未进入训练或测试的语音数。
    unused_test_speaker_rows = 0

    # 逐项遍历按语音唯一键索引的重采样音频清单，保持每条记录都经过相同检查。
    for row_key in sorted(wav_by_key):
        # 拆出当前语音所属说话人与完整句号，分别决定身份和文本划分。
        speaker, utt_id = row_key
        # 语音编号中的三位公共文本句号。
        slot = sentence_id(utt_id)
        # 重采样清单中证明音频已成功生成的记录。
        wav_row = wav_by_key[row_key]
        # 原始库存中提供性别、文本和源路径的记录。
        inventory_row = inventory_by_key[row_key]

        # 测试说话人的 001--010 进入 test，其余语音明确不用。
        if speaker in TEST_SPEAKERS:
            # 按语音编号中的三位公共文本句号决定是否进入此分支；判断条件为 slot in TEST_SENTENCE_IDS。
            if slot in TEST_SENTENCE_IDS:
                # 固定测试人的 001—010 进入测试集，并合并音频路径与官方文字。
                test_rows.append(build_row("test", wav_row, inventory_row))
            # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
            else:
                # 测试人的其他句子不进入任何集合，保持这些身份完全未参与训练。
                unused_test_speaker_rows += 1
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue

        # 严格 text-unseen 才删除训练说话人的公共测试句。
        # text-seen 则故意保留，从而只隔离“人”而不隔离“文本”。
        if args.protocol == PROTOCOL_TEXT_UNSEEN and slot in TEST_SENTENCE_IDS:
            # text-unseen 模式下，训练人的公共测试句被排除并计数。
            excluded_train_prompts += 1
            # 跳过已排除或已处理的当前条目，继续检查清单中的下一项。
            continue
        # 将通过协议筛选的其余训练人语音写入训练集合。
        train_rows.append(build_row("train", wav_row, inventory_row))

    # 每位训练说话人在最终清单中的语音数量。
    train_counts = Counter(str(row["speaker"]) for row in train_rows)
    # 每位测试说话人在最终清单中的语音数量。
    test_counts = Counter(str(row["speaker"]) for row in test_rows)
    # 最终训练记录仍需覆盖预定 100 位说话人。
    if set(train_counts) != set(train_speakers):
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Training speaker set changed while constructing rows.")
    # 测试集必须是固定十人且每人正好十句，不能少人或少句。
    if set(test_counts) != set(TEST_SPEAKERS) or set(test_counts.values()) != {10}:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Invalid test grid counts: {dict(test_counts)}")
    # 两种协议的文本边界正好相反，所以必须分别断言，不能只打印数量。
    train_prompt_rows = [
        # 保留满足公共句号条件的训练记录以便协议复核。
        row
        # 重新遍历最终训练清单，独立检查公共测试文本的去留。
        for row in train_rows
        # 筛出训练集中句号为 001—010 的记录。
        if sentence_id(str(row["utt_id"])) in TEST_SENTENCE_IDS
    ]
    # 按当前 seen/unseen 文本划分规则决定是否进入此分支；判断条件为 args.protocol == PROTOCOL_TEXT_UNSEEN and train_prompt_rows。
    if args.protocol == PROTOCOL_TEXT_UNSEEN and train_prompt_rows:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("text-unseen 协议中，001--010 错误地进入了 train_manifest.csv")
    # 按当前 seen/unseen 文本划分规则决定是否进入此分支；判断条件为 args.protocol == PROTOCOL_TEXT_SEEN and not train_prompt_rows。
    if args.protocol == PROTOCOL_TEXT_SEEN and not train_prompt_rows:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("text-seen 协议中，训练集没有保留任何 001--010，协议选择可能失效")
    # 训练与测试身份有任何交集都拒绝发布，防止未见说话人泄漏。
    if {str(row["speaker"]) for row in train_rows} & {str(row["speaker"]) for row in test_rows}:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("A held-out speaker leaked into train_manifest.csv")

    # 切换协议前旧数据划分的可恢复备份目录。
    backup = None if args.no_backup else archive_active_split(metadata)

    # 原子发布只用于参数学习的训练清单。
    atomic_write_csv(metadata / "train_manifest.csv", train_rows)
    # 原子发布固定 100 条测试清单。
    atomic_write_csv(metadata / "test_manifest.csv", test_rows)
    # 汇总本次实际使用的训练和测试语音，供特征提取统一覆盖。
    atomic_write_csv(metadata / "all_manifest.csv", train_rows + test_rows)
    # 清理当前函数明确定位的临时或已过期记录，避免下次误当有效结果。
    (metadata / "val_manifest.csv").unlink(missing_ok=True)
    # 单独保存本次固定测试人编号，便于报告与脚本共同核对。
    atomic_write_text(metadata / "test_speakers.txt", "\n".join(TEST_SPEAKERS) + "\n")
    # 单独保存十个公共文本编号，明确文本泄漏边界。
    atomic_write_text(metadata / "test_sentence_ids.txt", "\n".join(TEST_SENTENCE_IDS) + "\n")

    # 保存十个公共测试句原文的 CSV 路径。
    test_sentences_path = metadata / "test_sentences.csv"
    # 尚未原子发布的测试文本临时 CSV。
    temporary_sentences = test_sentences_path.with_name(test_sentences_path.name + ".tmp")
    # 在受控文件句柄中完成串联本文件的数据检查、模型处理和结果保存流程，退出此段后自动关闭文件。
    with temporary_sentences.open("w", encoding="utf-8", newline="") as handle:
        # 固定表头顺序的 CSV 写入器。
        writer = csv.DictWriter(handle, fieldnames=("sentence_id", "text"))
        # 先写列名，使后续行能按相同字段重新读取。
        writer.writeheader()
        # 把已整理的清单记录按固定字段顺序写入 CSV。
        writer.writerows(
            # 逐句写出固定编号和已核对跨说话人一致的官方原文。
            {"sentence_id": slot, "text": prompt_texts[slot]} for slot in TEST_SENTENCE_IDS
        )
    # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
    temporary_sentences.replace(test_sentences_path)

    # 训练语音中缺少转写的条数，模型本身不使用转写输入。
    missing_train_transcripts = sum(not str(row["text"]).strip() for row in train_rows)
    # 同时描述说话人和文本是否交叠的协议全称。
    protocol_name = (
        # 协议全名注明说话人与公共文本均与训练分离，测试规模为十人×十句。
        "speaker-disjoint_and_text-disjoint_10x10"
        # 按当前 seen/unseen 文本划分规则决定是否进入此分支；判断条件为 args.protocol == PROTOCOL_TEXT_UNSEEN。
        if args.protocol == PROTOCOL_TEXT_UNSEEN
        # text-seen 模式仍保持说话人完全分离，但允许训练文本与测试文本重叠。
        else "speaker-disjoint_text-overlap_10x10"
    )
    # 当前 seen/unseen 文本划分规则。
    protocol = {
        # 记录当前 seen/unseen 文本划分规则。
        "protocol": protocol_name,
        # 记录与训练命令严格核对的协议短名称。
        "protocol_cli_name": args.protocol,
        # 记录训练集是否完全排除公共测试文本的标志。
        "strict_text_unseen": args.protocol == PROTOCOL_TEXT_UNSEEN,
        # 记录训练完全不可见、只供测试的说话人集合。
        "test_speakers": list(TEST_SPEAKERS),
        # 记录本次固定的十个公共测试文本编号。
        "test_sentence_ids": list(TEST_SENTENCE_IDS),
        # 记录固定测试句号到原始转写的映射。
        "test_sentences": prompt_texts,
        # 记录具体说话人和句号如何重建的来源说明。
        "exact_split_id_provenance": {
            # 记录论文是否逐一公布这些具体 ID，此处明确为否。
            "paper_explicit": False,
            # 记录本次依据官方演示文件名重建 ID 的证据置信说明。
            "reconstruction_confidence": "high",
            # 记录重建具体测试 ID 所参考的官方证据列表。
            "basis": [
                # 八位测试人 ID 来自官方 FastVoiceGrad 演示文件名。
                "official FastVoiceGrad demo filenames (8 of 10 speakers)",
                # 其余 p243、p303 的证据来自后续官方成对比较演示。
                "official FasterVoiceGrad paired-comparison demo filenames (p243 and p303)",
                # 后续论文声明沿用同类 VCTK 十人十句设计，为跨演示重建提供辅助依据。
                "FasterVoiceGrad paper: same VCTK 10-speaker/10-sentence design and implementation following FastVoiceGrad",
                # 使用 VCTK 所有测试人共有的 001—010 公共文本编号。
                "VCTK common-prompt IDs 001-010",
            ],
            # 记录对论文明确设定与本次补充假设的完整披露。
            "disclosure": (
                # 披露论文明确给出的训练 100 人、未见测试 10 人设定。
                "The paper specifies 100 training speakers, 10 unseen test speakers, "
                # 披露每位测试人十句是明确设定，下一行说明具体 ID 的缺失。
                "and 10 test sentences per speaker, but does not explicitly publish the "
                # 论文没有逐个公布说话人与句号，本工程 ID 属于证据重建结果。
                "exact speaker IDs or sentence IDs. The recorded IDs are a high-confidence "
                # 重建时联合比较官方 FastVoiceGrad 和协议对应的后续演示。
                "reconstruction from the union of official FastVoiceGrad and protocol-matched "
                # 继续说明后续演示文件名的作用及公共文本编号来源。
                "FasterVoiceGrad demo filenames; 001-010 are the VCTK common prompts. This is "
                # 明确限制证据表述，不能把重建 ID 写成原论文逐项明示。
                "not a claim that the 2024 paper printed these exact IDs."
            ),
        },
        # 记录允许出现在参数训练中的说话人集合。
        "train_speakers": len(train_counts),
        # 记录最终测试说话人数量，应为十人。
        "test_speakers_count": len(test_counts),
        # 记录最终进入参数训练的语音条数。
        "train_utterances": len(train_rows),
        # 记录最终进入测试的语音条数，应为 100 条。
        "test_utterances": len(test_rows),
        # 记录从训练集排除的公共测试句条数。
        "excluded_train_prompt_rows": excluded_train_prompts,
        # 记录text-seen 中保留的公共测试句条数。
        "retained_train_prompt_rows": len(train_prompt_rows),
        # 记录测试说话人 011 以后未进入训练或测试的语音数。
        "unused_test_speaker_rows": unused_test_speaker_rows,
        # 记录有音频但没有转写的训练样本数。
        "training_rows_without_recovered_transcript": missing_train_transcripts,
        # 记录说明文本只用于划分核对、不参与教师网络输入。
        "transcript_note": (
            # 报告说明教师网络不读文字；text-unseen 通过删除公共句号控制文本泄漏。
            "Training does not consume text. Prompt IDs 001-010 are removed from train."
            # 按当前 seen/unseen 文本划分规则决定是否进入此分支；判断条件为 args.protocol == PROTOCOL_TEXT_UNSEEN。
            if args.protocol == PROTOCOL_TEXT_UNSEEN
            # 报告明确 text-seen 的训练清单有意保留公共句子；文本本身仍不作为教师模型输入。
            else "Training does not consume text. Prompt IDs 001-010 are intentionally retained in train."
        ),
        # 记录当前协议记录创建的 UTC 时间。
        "created_utc": datetime.now(timezone.utc).isoformat(),
        # 记录构建划分时读取的完整重采样库存来源。
        "source_resampled_inventory": str(full_wav_path),
        # 记录构建划分时读取的原始音频与转写库存来源。
        "source_inventory": str(full_inventory_path),
        # 记录是否使用旧 all_manifest 补数据，此处固定为否。
        "legacy_all_manifest_used": False,
        # 记录上一版划分的归档路径，便于回退和审计。
        "backup_of_previous_split": str(backup) if backup else None,
    }
    # 最后原子保存整个数据划分协议及其来源披露。
    atomic_write_text(
        # 后续训练启动时从此文件严格核对 seen/unseen 协议。
        metadata / "split_protocol.json",
        # 协议保存成便于人工阅读的 UTF-8 JSON，保留中文和完整字段。
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
    )

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"project_root={root}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"protocol={args.protocol}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"backup={backup}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"train: utterances={len(train_rows)}, speakers={len(train_counts)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"test:  utterances={len(test_rows)}, speakers={len(test_counts)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"excluded train prompt rows={excluded_train_prompts}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"retained train prompt rows={len(train_prompt_rows)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"unused test-speaker rows={unused_test_speaker_rows}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"training rows without recovered transcript={missing_train_transcripts}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"test speakers={list(TEST_SPEAKERS)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"test sentence IDs={list(TEST_SENTENCE_IDS)}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(
        # 全部核对通过后输出说话人分离的成功标志。
        "SPLIT BUILD PASSED: speakers are disjoint; "
        # 成功信息同时打印实际使用的 seen/unseen 文本协议。
        f"text protocol is {args.protocol}."
    )


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 执行本文件的参数检查及处理入口，错误会返回到终端。
    main()
