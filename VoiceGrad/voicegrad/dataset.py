# 教师数据接口：统一读取 VCTK mel、BNF 与身份特征，并核对同句对应关系。
"""Strict VCTK feature dataset for the VoiceGrad-30 teacher."""

# =============================================================================
# 这个文件负责什么？
# =============================================================================
# 神经网络不能直接吃 wav 文件。本文件负责根据 CSV 清单找到预处理后的三个文件：
#   data/mel/<speaker>/<utt>.npy      ：模型要还原的 80 维声学特征；
#   data/bnf/<speaker>/<utt>.npy      ：源语音 144 维音素/内容条件；
#   data/spk_emb/<speaker>/<utt>.npy  ：目标说话人 256 维身份条件。
# 然后完成：检查 -> 读取 -> 调整形状 -> BNF 对齐 -> mel 归一化 -> 裁剪/补零 ->
# 转成 PyTorch Tensor -> 组成 batch。
#
# 为什么要写得“严格”？旧代码遇到坏文件会随机换另一条语音，这会让你以为训练
# 正常，实际数据已经错了。这里宁可在训练前直接报错，也不静默污染实验。
#
# Dataset 是“如何按索引取一条样本”；DataLoader 是“如何多进程取样并组成 batch”。

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# --------------------------- 导入包逐项解释 ---------------------------
# csv 是 Python 标准库，用来把 manifest.csv 每行读成字典。
import csv
# random 是标准伪随机库，这里只用于选择 128 帧随机裁剪起点。
import random
# Counter 是“计数型字典”，用于统计每个 speaker 有多少条和总说话人数。
from collections import Counter
# Path 负责跨平台地拼接/检查文件路径。
from pathlib import Path
# Dict/List/Optional 是类型提示，帮助读者和编辑器理解数据结构。
from typing import Dict, List, Optional

# NumPy 负责读取 .npy、检查 NaN/Inf 和整理数组 dtype/shape。
import numpy as np
# torch 把 NumPy 数据转成能进入神经网络的 Tensor。
import torch
# F.interpolate 对齐 BNF 时间轴，F.pad 为短语音补零。
import torch.nn.functional as F
# Dataset 规定“怎样取一条”；DataLoader 负责打乱、多进程读取和组成 batch。
from torch.utils.data import DataLoader, Dataset

# 共享同一 BNF 时间对齐入口，使数据读取与推理遵循权重训练时的帧对应关系。
from .feature_alignment import LEGACY_ALIGNMENT, align_bnf_to_mel


# 集中实现训练及测试清单上的特征读取、对齐和裁剪。
class VoiceGradDataset(Dataset):
    # 数据集同时提供对齐内容帧、标准化后的 mel 和外部说话人向量。
    """Load aligned log-mel, BNF and external speaker embeddings.

    Missing or malformed data raises an error instead of being silently skipped
    or replaced with a random utterance. This makes a long server training run
    fail early when preprocessing is incomplete.
    """

    # 创建对象并固定后续读取或网络计算要遵循的参数；下面的参数分别限定输入数据和处理约束。
    def __init__(
        # 当前数据集或网络对象。
        self,
        # 预处理数据根目录。
        data_root,
        # 训练或测试集合的名称。
        split: str = "train",
        # 每个训练片段的 mel 帧数。
        segment_length: Optional[int] = 128,
        # 当前集合应包含的说话人数。
        expected_speakers: Optional[int] = None,
        # mel 频带数，本实验为 80。
        n_mels: int = 80,
        # BNF 内容维数，本实验为 144。
        n_bnf: int = 144,
        # 说话人向量宽度，本实验为 256。
        spk_emb_dim: int = 256,
        # 外部 BNF 到 mel 的时间对齐规则。
        bnf_alignment: str = LEGACY_ALIGNMENT,
    # 本接口负责创建对象并固定后续读取或网络计算要遵循的参数，处理结果通过对象状态或文件保存，不额外返回数值。
    ) -> None:
        # __init__ 在创建数据集对象时只执行一次，保存设置并提前检查数据；
        # 真正每次读取 .npy 的操作在后面的 __getitem__ 中。
        self.data_root = Path(data_root)
        # 训练或测试集合的名称。
        self.split = self._normalise_split(split)
        # 每个训练片段的 mel 帧数。
        self.segment_length = segment_length
        # mel 频带数，本实验为 80。
        self.n_mels = n_mels
        # BNF 内容维数，本实验为 144。
        self.n_bnf = n_bnf
        # 说话人向量宽度，本实验为 256。
        self.spk_emb_dim = spk_emb_dim
        # Existing teacher checkpoints were trained with length-only linear
        # interpolation.  New controlled runs can select physical time-grid
        # alignment without silently changing old checkpoint semantics.
        # 保存当前权重所需的外部 BNF 对齐方式，后面每条语音都按同一规则处理。
        self.bnf_alignment = bnf_alignment

        # f"{self.split}_manifest.csv" 是 f-string：把 split 的实际值插入字符串。
        # 例如 split="train" 时得到 train_manifest.csv。
        self.manifest_path = (
            # 从 metadata 目录选择 train_manifest.csv 或 test_manifest.csv，防止两个集合混用。
            self.data_root / "metadata" / f"{self.split}_manifest.csv"
        )
        # mel 目录保存按说话人分组的 80 维原始对数频谱。
        self.mel_dir = self.data_root / "mel"
        # BNF 目录保存同句的语音内容特征。
        self.bnf_dir = self.data_root / "bnf"
        # spk_emb 目录保存各参考语音的说话人向量。
        self.spk_emb_dir = self.data_root / "spk_emb"

        # mean/std 必须只由训练集计算。形状 [80,1] 可自动广播到 [80,T] 每一帧。
        self.mel_mean = self._load_stat("mel_mean.npy", n_mels)
        # 只用训练集统计的逐频带标准差。
        self.mel_std = self._load_stat("mel_std.npy", n_mels).clamp_min(1e-8)
        # 按唯一说话人和句子组织的特征条目。
        self.items = self._load_manifest()

        # Counter 会统计每个 speaker 出现多少次；len(Counter) 就是不同说话人数。
        # 括号中的 item[...] for item in ... 是 generator expression（生成器表达式）。
        speakers = Counter(item["speaker"] for item in self.items)
        # 按当前集合应包含的说话人数决定是否进入此分支；判断条件为 expected_speakers is not None and len(speakers) != expected_speakers。
        if expected_speakers is not None and len(speakers) != expected_speakers:
            # 创建对象并固定后续读取或网络计算要遵循的参数未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 列出当前数据清单路径和实际说话人数。
                f"{self.manifest_path} contains {len(speakers)} speakers, but "
                # 列出此运行要求人数，提示先修复数据划分。
                f"this run expects {expected_speakers}. Rebuild the split manifest first."
            )
        # 立即检查清单中的三个特征路径，缺文件时在启动训练前报错。
        self._validate_feature_paths()
        # 把创建对象并固定后续读取或网络计算要遵循的参数的当前进度或结果输出到服务器日志。
        print(
            # 数据加载日志显示实际集合名称和语音条数。
            f"[Dataset] split={self.split}, utterances={len(self.items)}, "
            # 同时记录实际人数与清单来源，便于和协议核对。
            f"speakers={len(speakers)}, manifest={self.manifest_path}"
        )

    # 该辅助方法只使用传入的数据，不依赖网络实例内部状态。
    @staticmethod
    # 规范集合名称并限制为已登记的训练或测试协议；下面的参数分别限定输入数据和处理约束。
    def _normalise_split(split: str) -> str:
        # @staticmethod 表示这个工具函数不需要读取 self。
        # strip 去掉首尾空格，lower 统一小写。本次论文复现明确只有 train/test，
        # 因此连底层 Dataset 也拒绝 val/valid/dev，防止其他脚本以后误画 validation。
        split = str(split).strip().lower()
        # 按训练或测试集合的名称决定是否进入此分支；判断条件为 split not in {"train", "test"}。
        if split not in {"train", "test"}:
            # 规范集合名称并限制为已登记的训练或测试协议未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(
                # 错误说明此协议只接受 train/test，显示收到的非法集合名。
                f"split must be 'train' or 'test' for this protocol, got {split!r}."
            )
        # 检查通过后原样返回规范的小写字符串。
        return split

    # 读取并核对训练集每个 mel 频带的归一化统计量；下面的参数分别限定输入数据和处理约束。
    def _load_stat(self, filename: str, expected_dim: int) -> torch.Tensor:
        # 统计量不存在时绝不能偷偷用 0/1 代替，否则训练分布和论文不一致。
        path = self.data_root / "stats" / filename
        # 检查当前待检查或读写的文件路径是否缺失；缺失时不能继续信任该输入。
        if not path.is_file():
            # 读取并核对训练集每个 mel 频带的归一化统计量未通过，抛出下方包含具体路径或实际值的错误。
            raise FileNotFoundError(
                # 指出缺失的训练集 mean/std 文件，不能临时用每句统计替代。
                f"Required training-set statistic is missing: {path}. "
                # 提示按训练清单完整生成并成套发布均值与标准差。
                "Run preprocess/04_compute_mel_stats.py."
            )
        # allow_pickle=False 是安全做法：这里只允许纯数值数组，不执行 pickle 对象。
        value = np.load(path, allow_pickle=False)
        # 要求每个 mel 频带恰好一个有限统计值，拒绝维数不符或 NaN/Inf。
        if value.shape != (expected_dim,) or not np.isfinite(value).all():
            # 读取并核对训练集每个 mel 频带的归一化统计量未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(f"Invalid {path}: expected [{expected_dim}], got {value.shape}.")
        # NumPy 数组转 PyTorch Tensor，再从 [80] reshape 为 [80,1] 以便广播。
        return torch.from_numpy(value.astype(np.float32)).view(expected_dim, 1)

    # 把清单中的每条语音解析为三个对应特征路径；下面的参数分别限定输入数据和处理约束。
    def _load_manifest(self) -> List[Dict[str, object]]:
        # 返回类型 List[Dict[str,object]] 的意思是：返回“由很多字典组成的列表”。
        # 每个字典代表一条 utterance，保存 speaker、utt_id 和三个特征路径。
        if not self.manifest_path.is_file():
            # 把清单中的每条语音解析为三个对应特征路径未通过，抛出下方包含具体路径或实际值的错误。
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        # 按唯一说话人和句子组织的特征条目。
        items: List[Dict[str, object]] = []
        # set（集合）用于快速查重；同一 (speaker,utt_id) 不能出现两次。
        seen = set()
        # 在受控文件句柄中完成把清单中的每条语音解析为三个对应特征路径，退出此段后自动关闭文件。
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            # DictReader 用 CSV 第一行做键，每一行变成例如 row["speaker"] 的字典。
            reader = csv.DictReader(handle)
            # 当前文件或记录必须提供的字段。
            required = {"speaker", "utt_id"}
            # 表头必须提供 speaker 和 utt_id，才能把同句三个特征正确配对。
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                # 把清单中的每条语音解析为三个对应特征路径未通过，抛出下方包含具体路径或实际值的错误。
                raise ValueError(
                    # 列出当前语音清单必须提供的列名。
                    f"{self.manifest_path} must contain columns {sorted(required)}."
                )
            # enumerate 同时给出行号和内容。start=2 是因为 CSV 第 1 行是表头。
            for row_number, row in enumerate(reader, start=2):
                # 当前语音所属说话人。
                speaker = (row.get("speaker") or "").strip()
                # 当前句子的完整编号。
                utterance = (row.get("utt_id") or "").strip()
                # 按当前语音所属说话人决定是否进入此分支；判断条件为 not speaker or not utterance。
                if not speaker or not utterance:
                    # 把清单中的每条语音解析为三个对应特征路径未通过，抛出下方包含具体路径或实际值的错误。
                    raise ValueError(f"Empty speaker/utt_id at manifest line {row_number}.")
                # 圆括号创建 tuple（不可变元组），很适合作为 set 中的唯一键。
                key = (speaker, utterance)
                # 按当前语音的唯一身份决定是否进入此分支；判断条件为 key in seen。
                if key in seen:
                    # 把清单中的每条语音解析为三个对应特征路径未通过，抛出下方包含具体路径或实际值的错误。
                    raise ValueError(f"Duplicate manifest item at line {row_number}: {key}")
                # 登记当前语音身份，后续再次出现时可识别为重复。
                seen.add(key)
                # 默认使用新格式 utt.npy；若不存在，兼容旧格式 utt.ling_feat.npy。
                bnf_path = self.bnf_dir / speaker / f"{utterance}.npy"
                # 检查当前句子的 BNF 特征文件是否缺失；缺失时不能继续信任该输入。
                if not bnf_path.is_file():
                    # 兼容旧提取器的 .ling_feat.npy 命名；只更换候选文件名，不改变内容。
                    legacy_name = self.bnf_dir / speaker / f"{utterance}.ling_feat.npy"
                    # 当前句子的 BNF 特征文件。
                    bnf_path = legacy_name if legacy_name.is_file() else bnf_path
                # 将当前语音的身份及三个对应特征路径作为一个不可拆散的样本条目加入列表。
                items.append(
                    # 下面展开把清单中的每条语音解析为三个对应特征路径所需的这组条目；各字段随后逐一说明。
                    {
                        # 记录当前语音所属说话人。
                        "speaker": speaker,
                        # 记录同时包含说话人和句号的样本编号。
                        "utt_id": utterance,
                        # 目标 mel 的路径，由当前说话人与句子身份共同确定。
                        "mel_path": self.mel_dir / speaker / f"{utterance}.npy",
                        # 记录当前句子的 BNF 特征文件。
                        "bnf_path": bnf_path,
                        # 这条语音对应的说话人嵌入路径，训练时与 mel 来自同一录音。
                        "spk_emb_path": self.spk_emb_dir / speaker / f"{utterance}.npy",
                    }
                )
        # 按按唯一说话人和句子组织的特征条目决定是否进入此分支；判断条件为 not items。
        if not items:
            # 把清单中的每条语音解析为三个对应特征路径未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Manifest contains no utterances: {self.manifest_path}")
        # 返回按唯一说话人和句子组织的特征条目，供调用方继续处理。
        return items

    # 在进入训练前发现缺失的 mel、BNF 或说话人文件；下面的参数分别限定输入数据和处理约束。
    def _validate_feature_paths(self) -> None:
        # 在真正占 GPU 前先检查所有清单项是否有三个特征文件。
        # 最多记录 20 个例子，避免 4 万个缺失路径把终端刷满。
        missing = []
        # 逐项遍历按唯一说话人和句子组织的特征条目，保持每条记录都经过相同检查。
        for item in self.items:
            # 对每条样本分别检查声学、内容和身份三种必需文件。
            for feature in ("mel_path", "bnf_path", "spk_emb_path"):
                # 当前待检查或读写的文件路径。
                path = item[feature]
                # 检查当前待检查或读写的文件路径是否缺失；缺失时不能继续信任该输入。
                if not path.is_file():
                    # 记录缺失文件的具体路径，以便终端报告能直接定位补提取目标。
                    missing.append(str(path))
                    # 错误示例最多保留 20 个，避免大量缺文件刷满日志。
                    if len(missing) >= 20:
                        # 结束当前扫描循环，保留此前已经收集的结果。
                        break
            # 错误示例最多保留 20 个，避免大量缺文件刷满日志。
            if len(missing) >= 20:
                # 结束当前扫描循环，保留此前已经收集的结果。
                break
        # 按检查中发现的缺失项目决定是否进入此分支；判断条件为 missing。
        if missing:
            # 把缺失路径逐行排列，便于阅读并复制到服务器检查。
            examples = "\n  ".join(missing)
            # 在进入训练前发现缺失的 mel、BNF 或说话人文件未通过，抛出下方包含具体路径或实际值的错误。
            raise FileNotFoundError(
                # 展示缺失的 mel、BNF 或身份文件示例，训练开始前必须补齐。
                "Preprocessing is incomplete. Missing feature examples:\n  " + examples
            )

    # 该辅助方法只使用传入的数据，不依赖网络实例内部状态。
    @staticmethod
    # 把时间与特征轴整理为约定通道在前的二维数组；下面的参数分别限定输入数据和处理约束。
    def _feature_matrix(value: np.ndarray, dim: int, name: str) -> np.ndarray:
        # mel 期望 [80,T]，BNF 期望 [144,T]。某些旧工具可能保存成 [T,80]，
        # 因此先判断哪一轴是 feature dim，必要时用 .T 转置。
        value = np.asarray(value)
        # mel 和 BNF 都应具有特征轴、时间轴，其他维数无法可靠解释。
        if value.ndim != 2:
            # 把时间与特征轴整理为约定通道在前的二维数组未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(f"{name} must be 2-D, got {value.shape}.")
        # 第一轴已是所需特征维数，保留通道在前的排布。
        if value.shape[0] == dim:
            # 当前函数生成的结果。
            result = value
        # 第二轴才是特征维数时转置，恢复为 [通道,时间]。
        elif value.shape[1] == dim:
            # 当前函数生成的结果。
            result = value.T
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 把时间与特征轴整理为约定通道在前的二维数组未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(f"{name} does not contain feature dimension {dim}: {value.shape}.")
        # 拒绝空时间轴以及含 NaN/Inf 的特征，避免污染训练。
        if result.shape[1] < 1 or not np.isfinite(result).all():
            # 把时间与特征轴整理为约定通道在前的二维数组未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(f"{name} is empty or contains NaN/Inf.")
        # 模型统一用 float32。copy=False 表示类型本来正确时尽量不复制内存。
        return result.astype(np.float32, copy=False)

    # 核对说话人条件是有限的 256 维向量；下面的参数分别限定输入数据和处理约束。
    def _speaker_vector(self, value: np.ndarray) -> np.ndarray:
        # squeeze 删除长度为 1 的轴，使 [1,256]、[256,1] 都能变为 [256]。
        value = np.asarray(value).squeeze()
        # 说话人向量必须恰为 256 维有限值，不能按时间序列误读。
        if value.shape != (self.spk_emb_dim,) or not np.isfinite(value).all():
            # 核对说话人条件是有限的 256 维向量未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(
                # 列出身份向量要求维数与磁盘实际形状。
                f"Speaker embedding must be [{self.spk_emb_dim}], got {value.shape}."
            )
        # 返回单精度身份向量；已有 float32 时避免额外复制。
        return value.astype(np.float32, copy=False)

    # 按权重训练时选定的规则将内容帧对齐到 mel 时轴；下面的参数分别限定输入数据和处理约束。
    def _align_bnf(self, bnf: torch.Tensor, mel_length: int) -> torch.Tensor:
        # BNF 是 10 ms 一帧，VoiceGrad mel 是 256/22050 秒一帧。统一入口既能
        # 原样复现旧检查点使用的长度插值，也能让新实验按真实帧中心时间对齐。
        return align_bnf_to_mel(
            # 表示语音内容的 144 维 BNF 特征。
            bnf,
            # 需要对齐到的 mel 时间帧数。
            mel_length,
            # 当前选择的时间对齐或插值模式。
            mode=self.bnf_alignment,
            # 音频文件实际或预期的采样率。
            sample_rate=22050,
            # 每个时窗的 FFT 点数。
            n_fft=1024,
            # 相邻 mel 窗口的样本间距。
            hop_size=256,
            # BNF 相邻帧的真实时间间距，当前提取器为 0.010 秒。
            bnf_frame_shift_seconds=0.010,
        )

    # 用同一时间窗口裁剪 mel 和 BNF 并保留真实帧数；下面的参数分别限定输入数据和处理约束。
    def _crop_or_pad(
        # 输入同一句且已在同一时间轴上的 mel 与 BNF。
        self, mel: torch.Tensor, bnf: torch.Tensor
    # 返回用同一时间窗口裁剪 mel 和 BNF 并保留真实帧数得到的张量；类型标注帮助调用方理解结果，不改变数值。
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # DataLoader 要把 32 条语音堆成 [32,C,T]，所以训练时 T 必须相同。
        # 长于 128：mel 与 BNF 使用完全相同的随机起点裁剪；短于 128：右侧补 0。
        # 验证/测试 segment_length=None，不裁整句，batch size 会被强制设为 1。
        valid_length = mel.shape[-1]
        # 按每个训练片段的 mel 帧数决定是否进入此分支；判断条件为 self.segment_length is None。
        if self.segment_length is None:
            # 同时返回两个对齐特征和真实有效长度，补帧不能冒充真实录音。
            return mel, bnf, valid_length
        # 按裁剪或补零前的真实有效帧数决定是否进入此分支；判断条件为 valid_length > self.segment_length。
        if valid_length > self.segment_length:
            # randint 两端都可能取到；最大起点保证 stop 不越过语音末尾。
            start = random.randint(0, valid_length - self.segment_length)
            # 当前片段末尾之后的首个索引。
            stop = start + self.segment_length
            # 两个特征使用完全相同的起止帧，裁出目标长度后有效帧数即片段长度。
            return mel[..., start:stop], bnf[..., start:stop], self.segment_length
        # 右侧补齐到固定长度所缺的帧数。
        padding = self.segment_length - valid_length
        # 按右侧补齐到固定长度所缺的帧数决定是否进入此分支；判断条件为 padding。
        if padding:
            # 对已标准化的 80 维 mel 在时间末尾补零，补值对应各通道训练均值。
            mel = F.pad(mel, (0, padding))
            # 表示语音内容的 144 维 BNF 特征。
            bnf = F.pad(bnf, (0, padding))
        # 同时返回两个对齐特征和真实有效长度，补帧不能冒充真实录音。
        return mel, bnf, valid_length

    # 向加载器提供可用语音条目的总数；下面的参数分别限定输入数据和处理约束。
    def __len__(self) -> int:
        # Python 的 len(dataset) 会自动调用这里，返回清单中有多少条语音。
        return len(self.items)

    # 按清单索引读取同一句的 mel、BNF 和说话人条件并同步裁剪；下面的参数分别限定输入数据和处理约束。
    def __getitem__(self, index: int) -> Dict[str, object]:
        # DataLoader 请求 dataset[index] 时进入这里。index 是 0 到 len-1 的整数。
        item = self.items[index]
        # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
        try:
            # 一条 utterance 必须同时成功读出 mel、BNF 和 speaker embedding。
            mel_array = np.load(item["mel_path"], allow_pickle=False)
            # 磁盘读出的 BNF 数组。
            bnf_array = np.load(item["bnf_path"], allow_pickle=False)
            # 参考语音的说话人嵌入数组。
            speaker_array = np.load(item["spk_emb_path"], allow_pickle=False)
            # 80 维自然对数 mel 声谱图。
            mel = torch.from_numpy(self._feature_matrix(mel_array, self.n_mels, "mel"))
            # 表示语音内容的 144 维 BNF 特征。
            bnf = torch.from_numpy(self._feature_matrix(bnf_array, self.n_bnf, "BNF"))
            # 当前语音所属说话人。
            speaker = torch.from_numpy(self._speaker_vector(speaker_array))
        # 捕获按清单索引读取同一句的 mel、BNF 和说话人条件并同步裁剪中的读取、格式或兼容错误，让调用者获得明确失败原因。
        except Exception as error:
            # ``raise ... from error`` 会保留原始异常原因，方便定位具体坏文件。
            raise RuntimeError(
                # 将加载异常绑定到具体说话人和句子，便于定位坏特征。
                f"Failed to load {item['speaker']}/{item['utt_id']}: {error}"
            # 保留底层文件读取或形状检查异常，并与当前语音身份一起报告。
            ) from error

        # 先把 BNF 对齐到原始 mel 全长，之后二者才能用同一位置裁剪。
        bnf = self._align_bnf(bnf, mel.shape[-1])
        # 归一化公式 (mel-训练集均值)/训练集标准差，逐 mel 通道执行。
        # 必须先归一化真实帧、再补 0；补的 0 才代表“归一化空间中的均值”。
        # Normalize real frames first, then zero-pad in normalized space.
        mel = (mel - self.mel_mean) / self.mel_std
        # 先完成标准化，再同步随机裁剪或在标准化空间补零。
        mel, bnf, valid_length = self._crop_or_pad(mel, bnf)
        # contiguous 把 Tensor 内存整理为连续布局，送入卷积时更稳妥高效。
        # 返回字典后，DataLoader 会自动把 32 条相同键的数据堆成一个 batch。
        return {
            # 返回标准化后的 80 维 mel，时间长度已与 BNF 对齐。
            "mel": mel.contiguous(),
            # 记录表示语音内容的 144 维 BNF 特征。
            "bnf": bnf.contiguous(),
            # 记录256 维目标说话人条件。
            "spk_emb": speaker.contiguous(),
            # 记录补零之前的有效 mel 帧数。
            "mel_len": torch.tensor(valid_length, dtype=torch.long),
            # 保留说话人文本编号供日志及覆盖检查，不把它当模型的身份标签。
            "spk_name": item["speaker"],
            # 记录同时包含说话人和句号的样本编号。
            "utt_id": item["utt_id"],
        }


# 构造与训练或测试用途匹配的特征加载器；下面的参数分别限定输入数据和处理约束。
def get_dataloader(
    # 预处理数据根目录。
    data_root,
    # 训练或测试集合的名称。
    split: str,
    # 每批共同送入模型的语音条数。
    batch_size: int,
    # 并行读取特征的进程数。
    num_workers: int = 4,
    # 每个训练片段的 mel 帧数。
    segment_length: int = 128,
    # 当前集合应包含的说话人数。
    expected_speakers: Optional[int] = None,
    # 外部 BNF 到 mel 的时间对齐规则。
    bnf_alignment: str = LEGACY_ALIGNMENT,
# 返回构造与训练或测试用途匹配的特征加载器整理的 DataLoader 结果，内部各项保持前面约定的对应关系。
) -> DataLoader:
    # 这是给 train.py 使用的便捷工厂函数：一次创建 Dataset 和 DataLoader。
    split = VoiceGradDataset._normalise_split(split)
    # 将已核对的 split、路径和对齐参数传给共享数据集。
    dataset = VoiceGradDataset(
        # 预处理数据根目录。
        data_root=data_root,
        # 训练或测试集合的名称。
        split=split,
        # 每个训练片段的 mel 帧数。
        segment_length=segment_length if split == "train" else None,
        # 当前集合应包含的说话人数。
        expected_speakers=expected_speakers,
        # 外部 BNF 到 mel 的时间对齐规则。
        bnf_alignment=bnf_alignment,
    )
    # 当前加载器是否用于参数训练。
    is_train = split == "train"
    # shuffle：每个 epoch 打乱训练顺序；test 不打乱，保证每次监控顺序一致。
    # drop_last：训练最后不足 32 条的残 batch 丢弃，使每步 batch size 稳定。
    # num_workers：并行读取数据的子进程数。
    # pin_memory：有 CUDA 时使用页锁定内存，加快 CPU->GPU 拷贝。
    # persistent_workers=False：每个 epoch 重新按 DataLoader base seed 启动 worker。
    # 这样 latest.pt 恢复父进程 RNG 后，下一轮的 shuffle 与 Python-random 裁剪也能
    # 沿着同一条随机轨迹继续；常驻 worker 的内部 Python RNG 无法写进 checkpoint。
    return DataLoader(
        # 把逐样本特征读取规则交给批次加载器。
        dataset,
        # 每批共同送入模型的语音条数。
        batch_size=batch_size if is_train else 1,
        # 训练每轮打乱次序，测试保持清单顺序。
        shuffle=is_train,
        # 训练丢弃不足设定 batch 的尾批，测试保留所有样本。
        drop_last=is_train,
        # 并行读取特征的进程数。
        num_workers=num_workers,
        # 使用 GPU 时启用锁页内存，加快特征搬运。
        pin_memory=torch.cuda.is_available(),
        # 每轮重建工作进程，避免其裁剪随机状态脱离检查点恢复轨迹。
        persistent_workers=False,
    )
