# 在 VCTK text-unseen 数据上训练 ACDD 单步学生，并登记完整复现契约。
"""Train the paper-aligned VCTK FastVoiceGrad student with ACDD.

This entry point is intentionally narrow: epoch-500 VoiceGrad initialization,
K=1/S_K=950, the final step-2,500,000 custom modified HiFi-GAN Generator,
global batch 32, Adam 2e-4/(0.5,0.9), and 100 epochs are hard requirements.
"""

# argparse 用来读取命令行参数，例如项目目录、检查点路径和是否只做预检查。
import argparse
# csv 用来逐轮保存训练/测试损失；CSV 可直接用 Excel、Python 或 VS Code 查看。
import csv
# 复制冻结训练配置并填入命令行指定的修复模式，保留全局默认。
from dataclasses import replace
# fcntl 提供 Linux 文件锁，防止同一块数据和 GPU 被两个训练进程同时占用。
import fcntl
# hashlib 提供 SHA-256：固定测试裁剪位置，并核对文件是否被改动。
import hashlib
# json 用来读写可审计的实验契约 run_contract.json。
import json
# math 提供有限数检查、向上取整和浮点近似比较。
import math
# os 用来读取环境变量、强制落盘并原子替换文件。
import os
# random 是 Python 自带随机数生成器；它的状态也会写入断点，保证续训可复现。
import random
# signal 捕获 Ctrl+C/SIGTERM，让训练在完整 epoch 边界安全保存后再停。
import signal
# subprocess 调用仓库已有的数据审计和 Git 命令，并要求子命令成功才继续。
import subprocess
# sys 提供当前 Python 解释器路径，保证审计脚本使用同一环境。
import sys
# Counter 统计每位说话人的句数；defaultdict 用零值累加每种 loss。
from collections import Counter, defaultdict
# Path 以跨平台、可读的方式拼接并规范化文件路径。
from pathlib import Path
# 这些类型标注说明函数输入输出结构，方便读代码和静态检查，不改变训练数值。
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

# NumPy 负责 NumPy 随机种子和随机状态，确保整个数据流程可复现。
import numpy as np
# PyTorch 是模型、张量、自动微分、优化器及 CUDA 训练的核心库。
import torch
# nn 是神经网络模块命名空间，这里用于通用模型类型标注。
import torch.nn as nn
# F 提供张量函数；这里主要用零填充把短测试语音补到固定 128 帧。
import torch.nn.functional as F
# Dataset 定义测试样本读取规则，DataLoader 负责按 batch 送入模型。
from torch.utils.data import DataLoader, Dataset
# tqdm 在终端显示每轮训练/测试的实时进度条。
from tqdm import tqdm

# VoiceGradDataset 读取 mel、BNF、说话人向量；get_dataloader 构造严格的训练批次。
from voicegrad.dataset import VoiceGradDataset, get_dataloader
# VoiceGradDiffusion 实现论文使用的扩散时间表和相关计算。
from voicegrad.diffusion import VoiceGradDiffusion
# 运行位置保护：正式计算只能在用户租用的 Matpool 服务器上进行。
from voicegrad.runtime_guard import assert_rented_server
# 以下审计函数逐项检查数据来源、mel 提取设置、统计量和 manifest。
from voicegrad.validate_setup import (
    audit_feature_provenance,  # 核对 BNF/说话人嵌入由哪份音频、哪套工具产生。
    audit_mel_extraction_recipe,  # 核对 22.05 kHz、FFT/hop/window、80 维 mel 等设置。
    audit_mel_stats_publication,  # 核对训练集 mel 均值和标准差的发布记录。
    read_manifest,  # 读取 train/test/all 三份 CSV 样本清单。
)

# 以下组件负责组装学生、冻结教师/声码器、判别器及各种安全断言。
from .components import (
    PaperWaveformDiscriminator,  # 论文波形判别器的类型：5 个 MPD + 3 个 MRD 分支。
    assert_disjoint_parameter_sets,  # 防止教师、学生、声码器、判别器意外共享参数。
    assert_frozen,  # 确认指定模型不会在训练中更新。
    assert_optimizer_ownership,  # 确认每个优化器只持有它应该更新的参数。
    assert_trainable,  # 确认学生和判别器确实打开了梯度。
    build_paper_discriminator,  # 根据论文/参考实现创建 MPD+MRD，明确移除 MSD。
    denormalize_mel,  # 将标准化 mel 恢复到声码器训练时使用的数值尺度。
    load_frozen_vocoder,  # 加载本项目训练到 250 万步的修改版 HiFi-GAN 并冻结。
    load_mel_statistics,  # 加载训练集统计出的 80 维 mel 均值和标准差。
    load_teacher_and_student,  # 加载 epoch500 教师，并用其参数初始化学生。
    load_torch_checkpoint,  # 以统一、安全方式读取 PyTorch 检查点。
    set_requires_grad,  # 在判别器更新和学生更新之间切换判别器梯度开关。
    sha256_file,  # 计算文件指纹，锁定代码、数据和权重的具体版本。
)
# 读取唯一的论文严格配置对象及其类型定义。
from .config import PAPER_STUDENT_CONFIG, FastVoiceGradStudentConfig
# 引入训练损失窗口与报警记录，及时捕捉持续上涨。
from .loss_monitor import LossMonitor
# 以下函数分别实现 ACDD 总损失、特征匹配、对抗损失、一步预测与蒸馏损失。
from .objectives import (
    acdd_student_loss,  # 合成 L_adv + 2*L_FM + 45*L_dist。
    feature_matching_loss,  # 比较真假波形在判别器各层中的中间特征。
    least_squares_discriminator_loss,  # 判别器的最小二乘 GAN 损失。
    least_squares_student_loss,  # 学生生成波形“骗过”判别器的最小二乘损失。
    make_student_prediction,  # 从 S_K=950 的噪声状态一步预测干净 mel。
    score_distillation_loss,  # 用冻结 VoiceGrad 教师监督学生的一步预测。
)


# 检查点格式版本为 3；续训时版本不一致会拒绝加载，避免字段错读。
CHECKPOINT_FORMAT_VERSION = 3
# 这 7 个名字是每轮都要记录的完整损失组成，顺序也是 CSV/绘图固定顺序。
LOSS_COMPONENTS = (
    "student_total",  # 学生总损失：对抗项 + 2 倍特征匹配项 + 45 倍蒸馏项。
    "student_adv",  # 未额外加权的学生对抗损失。
    "fm_raw",  # 未乘权重的判别器特征匹配损失。
    "fm_weighted",  # 乘论文权重 lambda_fm=2 后的特征匹配贡献。
    "dist_raw",  # 未乘权重的教师分数蒸馏损失。
    "dist_weighted",  # 乘论文权重 lambda_dist=45 后的蒸馏贡献。
    "discriminator",  # 区分真实/生成波形的判别器训练损失。
)
# 给每个损失名前加 train_，形成一轮训练集平均值的 CSV 列名。
TRAIN_LOSS_FIELDS = tuple(f"train_{key}" for key in LOSS_COMPONENTS)
# 给同样的损失名前加 test_，形成固定 text-unseen 测试集诊断列名。
TEST_LOSS_FIELDS = tuple(f"test_{key}" for key in LOSS_COMPONENTS)
# 附加记录测试覆盖的样本数和 batch 数，防止漏测或重复测。
LOSS_COUNT_FIELDS = ("test_examples", "test_batches")
# loss.csv 的最终、不可随意改变的表头：轮次、步数、训练损失、测试损失、覆盖计数。
LOSS_FIELDS = (
    "epoch",  # 当前训练轮次，范围 1～100。
    "global_step",  # 已完成的学生/判别器更新次数；每个完整 batch 加 1。
    *TRAIN_LOSS_FIELDS,  # 展开全部 train_* 列。
    *TEST_LOSS_FIELDS,  # 展开全部 test_* 列。
    *LOSS_COUNT_FIELDS,  # 展开 test_examples 与 test_batches 两列。
)


# 固定测试裁剪的数据集只用于每轮 loss 诊断，绝不会被优化器用于训练。
class FixedSegmentTestDataset(Dataset):
    # 为每条测试语音固定选取 128 帧片段，确保各轮诊断用同一内容。
    """Return one fixed 128-frame diagnostic segment for every test utterance.

    The official text-unseen test manifest contains variable-length utterances,
    while the training objective sees 128-frame segments.  A SHA-256-derived
    crop makes the comparison shape-matched, independent of worker scheduling,
    and exactly repeatable across epochs and resume.  This dataset is diagnostic
    only; none of its samples can reach an optimizer step.
    """

    # 初始化时锁定数据目录、固定帧数、裁剪种子、说话人数和 BNF 对齐方式。
    def __init__(
        self,  # 当前 FixedSegmentTestDataset 对象自身。
        data_root: Path,  # 已预处理数据根目录，内部包含 test manifest 和特征文件。
        segment_length: int,  # 每条测试样本统一取 128 帧，与训练输入长度完全一致。
        crop_seed: int,  # 只决定每条语音的固定裁剪位置，所有 epoch 保持不变。
        expected_speakers: int,  # text-unseen 测试集应恰好包含 10 位未见说话人。
        bnf_alignment: str,  # BNF 与 mel 的时间轴对齐策略，必须与训练配置一致。
    # 结束 __init__ 的参数声明；返回类型为 None，具体内容由下方实现给出。
    ) -> None:
        # 长度必须至少为 1，否则没有有效帧可交给模型。
        if segment_length < 1:
            # 立即报错比产生空张量后在网络深处失败更容易定位问题。
            raise ValueError("Fixed test segment length must be positive")
        # 底层数据集读取完整测试语音；固定裁剪在本包装类中统一完成。
        self.base = VoiceGradDataset(
            # 源mel、BNF、说话人向量和统计量所在的数据目录。
            data_root=data_root,  # 指向当前实验唯一的数据根目录。
            # 当前数据属于训练还是测试。
            split="test",  # 明确只读 text-unseen 测试 manifest，避免训练数据泄漏。
            segment_length=None,  # 先读取完整长度，不使用训练集的随机裁剪。
            expected_speakers=expected_speakers,  # 读取时检查 10 位测试说话人。
            # BNF 到梅尔的帧对齐模式。
            bnf_alignment=bnf_alignment,  # 使用已登记的时间对齐规则。
        )
        # 转为普通整数，保证后续长度运算和序列化结果稳定。
        self.segment_length = int(segment_length)
        # 保存固定种子；相同语音身份始终算出相同起点。
        self.crop_seed = int(crop_seed)

    # DataLoader 会调用此方法确定测试样本总数。
    def __len__(self) -> int:
        # 直接返回底层测试集长度，不能增加、删除或重复样本。
        return len(self.base)

    # 根据种子、说话人和句子 ID 计算确定性裁剪起点。
    def _start(self, speaker: str, utterance: str, frames: int) -> int:
        # 可选起点数=总帧数-片段帧数+1；这样最后一个合法窗口也能被选中。
        choices = frames - self.segment_length + 1
        # 语音不长于目标片段时只能从第 0 帧开始，短缺部分稍后补零。
        if choices <= 1:
            # 返回 0 表示从语音最开头取数据。
            return 0
        # 将种子和样本唯一身份组合成字节串；\0 分隔可避免字符串拼接歧义。
        identity = (
            # UTF-8 编码让同一文本在不同机器上得到相同 SHA-256。
            f"{self.crop_seed}\0{speaker}\0{utterance}".encode("utf-8")
        )
        # 取 SHA-256 前 8 字节作为稳定的 64 位大端整数，不依赖 Python 随机哈希。
        number = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        # 对合法起点数取余，结果一定落在 0～choices-1。
        return number % choices

    # DataLoader 用索引读取一个已对齐、固定为 128 帧的测试样本。
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 复制底层字典，避免在这里更新字段时污染缓存或原始数据。
        sample = dict(self.base[index])
        # mel 形状应为 [80, T]：80 个声学频带、T 个时间帧。
        mel = sample["mel"]
        # BNF 形状应为 [144, T]：144 维语音内容特征、同一时间长度 T。
        bnf = sample["bnf"]
        # 两者都必须是二维且时间帧数相同，否则无法逐帧条件生成。
        if mel.ndim != 2 or bnf.ndim != 2 or mel.shape[-1] != bnf.shape[-1]:
            # 在进入 GPU 前明确报告损坏或错位的测试特征。
            raise RuntimeError("Fixed test crop received malformed mel/BNF tensors")
        # 读取当前语音实际帧数 T，转成 Python int 便于分支判断。
        frames = int(mel.shape[-1])
        # 长语音需要选一个固定窗口，保证输入正好为 128 帧。
        if frames > self.segment_length:
            # 利用样本身份计算可复现起点，而不是每轮重新随机抽取。
            start = self._start(
                # 说话人名、句子 ID、总帧数共同决定合法起点。
                str(sample["spk_name"]), str(sample["utt_id"]), frames
            )
            # Python 切片右端不包含 stop，因此窗口长度恰为 segment_length。
            stop = start + self.segment_length
            # `...` 保留 mel 的 80 个通道，只裁时间维 [start:stop]。
            mel = mel[..., start:stop]
            # BNF 必须用完全相同的起止帧，维持声学与内容逐帧对齐。
            bnf = bnf[..., start:stop]
            # 长语音裁出的窗口每一帧都是真实有效帧。
            valid_length = self.segment_length
        # 短语音或刚好 128 帧时，从第 0 帧开始并按需在右侧补零。
        else:
            # 固定起点为 0，便于记录和复查。
            start = 0
            # 有效长度保留补零前的真实帧数。
            valid_length = frames
            # padding 表示还缺多少帧才能达到固定的 128 帧。
            padding = self.segment_length - frames
            # 只有确实短于 128 帧时才调用填充函数。
            if padding:
                # mel 时间维右侧补 0；标准化空间中的 0 对应训练集均值。
                mel = F.pad(mel, (0, padding))
                # BNF 同样右补 0，保证两种特征长度始终一致。
                bnf = F.pad(bnf, (0, padding))
        # 用裁剪/补齐后的张量及诊断元数据覆盖样本字典中的旧值。
        sample.update(
            {  # 以下四个字段一起替换原样本中的未裁剪特征与长度信息。
                # 80 通道、短帧长的人工梅尔数组。
                "mel": mel.contiguous(),  # 连续内存的 [80,128] mel，GPU 复制更稳定。
                # 144 通道的人工内容特征数组。
                "bnf": bnf.contiguous(),  # 连续内存的 [144,128] BNF。
                "mel_len": torch.tensor(valid_length, dtype=torch.long),  # 真实有效帧数。
                "fixed_crop_start": torch.tensor(start, dtype=torch.long),  # 固定裁剪起点。
            }
        )
        # 返回一个样本；DataLoader 会把多个样本堆叠成 [B,C,128]。
        return sample


# 构造每个 epoch 都完整遍历 100 条 text-unseen 样本的确定性测试加载器。
def build_test_loss_loader(
    data_root: Path, config: FastVoiceGradStudentConfig  # 数据位置与全部实验固定设置。
# 结束 build_test_loss_loader 的参数声明；返回类型为 DataLoader，具体内容由下方实现给出。
) -> DataLoader:
    # 用上面的包装数据集保证每条测试语音都得到同一个 128 帧窗口。
    dataset = FixedSegmentTestDataset(
        # 源mel、BNF、说话人向量和统计量所在的数据目录。
        data_root=data_root,  # 读取 /data 下已审计的特征。
        segment_length=config.test_loss_frames,  # 固定为 128 帧，与训练片段相同。
        crop_seed=config.test_loss_crop_seed,  # 固定裁剪种子，不随 epoch 变化。
        expected_speakers=config.test_speakers,  # 应为 10 位未见说话人。
        # BNF 到梅尔的帧对齐模式。
        bnf_alignment=config.bnf_alignment,  # 沿用项目登记的 BNF 对齐方式。
    )
    # 论文 text-unseen 网格为 10 位说话人×每人 10 句=100 条。
    expected_examples = config.test_speakers * config.test_sentences_per_speaker
    # 若实际样本数不是 100，停止而不是画一条覆盖不完整、不可比较的曲线。
    if len(dataset) != expected_examples:
        # 错误中同时显示期望值和实际值，便于定位 manifest 问题。
        raise RuntimeError(
            f"Held-out test loss requires {expected_examples} examples, got {len(dataset)}"  # 期望/实际数量。
        )
    # 单独建立 CPU 随机生成器，隔离 DataLoader 与正式训练的随机数流。
    loader_generator = torch.Generator(device="cpu")
    # 使用配置中登记的固定种子；即便以后加载器内部用到随机数也可复现。
    loader_generator.manual_seed(config.test_loss_loader_seed)
    # 返回测试 DataLoader；它不打乱、不丢最后 4 条样本，也不常驻 worker。
    return DataLoader(
        dataset,  # 上面创建的 100 条固定窗口测试集。
        # 每个固定测试批的最大样本数，末尾不足时仍完整保留。
        batch_size=config.test_loss_batch_size,  # 通常为 32，得到 32/32/32/4 四批。
        shuffle=False,  # 保持 manifest 顺序，使每轮测试顺序完全相同。
        drop_last=False,  # 保留最后不足 32 的 4 条，确保测试覆盖 100%。
        num_workers=config.num_workers,  # 并行读取进程数，只影响速度、不改目标函数。
        pin_memory=torch.cuda.is_available(),  # 有 CUDA 时锁页内存可加快 CPU→GPU 传输。
        persistent_workers=False,  # 每次迭代结束释放 worker，避免跨轮隐藏状态。
        # DataLoader 使用独立的随机数生成器，避免固定测试加载消耗训练的全局随机序列。
        generator=loader_generator,  # 使用隔离的固定随机数生成器。
    )


# 定义并解析启动脚本接受的全部命令行参数。
def parse_args() -> argparse.Namespace:
    # 优先读取 VOICEGRAD_PROJECT_ROOT；未设置时用当前目录，并转为绝对路径。
    root = Path(os.environ.get("VOICEGRAD_PROJECT_ROOT", Path.cwd())).resolve()
    # 创建参数解析器；帮助文字直接使用文件顶部说明。
    parser = argparse.ArgumentParser(description=__doc__)
    # 项目根目录参数控制所有相对数据、代码和输出位置。
    parser.add_argument("--project-root", type=Path, default=root)
    # 教师权重固定指向 text-unseen VoiceGrad 的 epoch500 检查点。
    parser.add_argument(
        "--teacher-checkpoint",  # 命令行参数名。
        type=Path,  # 将输入文本转换成 Path 对象。
        # 默认从当前项目的正式教师输出目录读取。
        default=root  # 从项目根目录开始拼接。
        / "exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt",  # 教师 epoch500 文件。
    )
    # 声码器权重固定使用本实验训练完成的修改版 HiFi-GAN 250 万步模型。
    parser.add_argument(
        "--vocoder-checkpoint",  # 声码器检查点参数名。
        type=Path,  # 路径类型便于后面做存在性和绝对路径检查。
        # 默认值明确锁定 2,500,000 步，而不是公开预训练声码器。
        default=root  # 从项目根目录开始拼接。
        / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/generator_step_2500000.pt",  # 250 万步权重。
    )
    # 同一声码器的网络/声学配置文件必须与权重配套。
    parser.add_argument(
        "--vocoder-config",  # HiFi-GAN 配置参数名。
        type=Path,  # 将命令行值解析为文件路径。
        default=root / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/config.json",  # 默认配置。
    )
    # attestation 记录最终声码器步数、哈希和依赖，用来防止拿错权重。
    parser.add_argument(
        "--vocoder-attestation",  # 声码器验真记录参数名。
        type=Path,  # 解析为 Path。
        # 默认读取完成 250 万步后生成的最终验真 JSON。
        default=root  # 从项目根目录开始拼接。
        # 这份证明核验的是同一个250万步声码器，源码指纹对应逐行注释后的当前版本。
        / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/final_checkpoint_attestation_comments_20260906.json",
    )
    # 学生权重、loss.csv、运行契约和 loss 图都写入这一目录。
    parser.add_argument(
        "--output-dir",  # 输出目录参数名。
        type=Path,  # 转为路径对象。
        # 修复采用独立的新目录，保留旧100轮结果及其契约，绝不覆盖旧实验。
        default=root / "exp/checkpoints/fastvoicegrad_vctk_text_unseen_distillation_clean_v1",
    )
    # --resume 可指定 latest.pt 续训；不提供时必须从空目录开始新训练。
    parser.add_argument("--resume", type=Path, default=None)
    # --preflight-only 只做完整审计和模型加载，不进入任何训练 step。
    parser.add_argument("--preflight-only", action="store_true")
    # 默认使用 cuda；正式训练还会再次强制检查，禁止误用 CPU 或本机。
    parser.add_argument("--device", default="cuda")
    # 公开教师目标空间选项，区分历史相邻状态与干净梅尔重建。
    parser.add_argument(
        # 目标空间只能选历史相邻步或修复后的干净x0重建。
        "--score-target", choices=("legacy_adjacent", "clean_x0"),
        # 命令行和配置类共用一个默认值，避免两处设定不一致时误回到旧目标。
        default=PAPER_STUDENT_CONFIG.score_target,
        # 帮助文字说明clean_x0会直接从抽到的t还原到干净状态。
        help="Teacher distillation target; clean_x0 performs the selected t -> 0 transition.",
    )
    # 公开三种噪声权重解释，训练记录中必须明确选择。
    parser.add_argument(
        # 公开三种明确的alpha解释，避免同一名称对应不同数学系数。
        "--score-weight", choices=("nominal_alpha", "alpha_bar", "sqrt_alpha_bar"),
        # 使用已登记的累计信号幅度权重，遗漏命令行参数也能进入修复路径。
        default=PAPER_STUDENT_CONFIG.score_weight,
        # 说明第三种权重依据同作者2025年对FastVoiceGrad的明确公式。
        help="Nominal alpha_t, effective alpha_bar_t, or sqrt(alpha_bar_t) clarified in the authors' 2025 paper.",
    )
    # 非默认目标/权重只供有意安排的消融或历史回放，需增加第二道显式声明。
    # 此开关只允许选择目标，不会绕过旧检查点的代码、数据和运行契约校验。
    parser.add_argument(
        "--allow-score-ablation", action="store_true",
        help="Explicitly allow a non-default score objective for a separately recorded diagnostic run.",
    )
    # 每50个训练批次形成一个不重叠窗口，减少单批噪声对报警的影响。
    parser.add_argument("--loss-monitor-window", type=int, default=50)
    # 公开整轮暂停点，便于先检查修复训练走势后继续。
    parser.add_argument(
        # 可指定完成某个整轮后暂停，便于先检查修复训练的趋势。
        "--stop-after-epoch", type=int, default=None,
        # 暂停只影响本次运行时长，实验总目标仍保存为100轮。
        help="Pause after this complete epoch; the experiment still targets 100 epochs.",
    )
    # 先解析参数，再检查是否误用了旧目标；此时尚未加载模型或占用GPU。
    args = parser.parse_args()
    # 只把clean_x0与sqrt_alpha_bar这一组合当作本轮修复的默认训练方案。
    if (args.score_target, args.score_weight) != ("clean_x0", "sqrt_alpha_bar"):
        # 旧目标、旧权重或混合组合都需要明确声明为对照，避免复制旧命令误开训。
        if not args.allow_score_ablation:
            # argparse会解释原因并退出，不会创建新训练状态。
            parser.error(
                "Non-default score objective requires --allow-score-ablation; "
                "the repaired default is clean_x0 + sqrt_alpha_bar."
            )
    # 返回已验证的配置；正式续训仍须通过后面的完整契约匹配。
    return args


# 同时固定三套随机数系统，使数据裁剪、噪声和模型训练可以精确续跑。
def seed_everything(seed: int) -> None:
    # 固定 Python 标准库随机数。
    random.seed(seed)
    # 固定 NumPy 随机数。
    np.random.seed(seed)
    # 固定 PyTorch CPU 随机数，也影响后续生成器初始化。
    torch.manual_seed(seed)
    # 只有 CUDA 存在时才调用 CUDA 种子接口，预检查 CPU 模式也能正常运行。
    if torch.cuda.is_available():
        # 固定所有可见 GPU 的随机数状态；正式运行这里只有租用 A100。
        torch.cuda.manual_seed_all(seed)


# 获取三个非阻塞文件锁，确保声码器流程、声码器训练和学生训练不会互相冲突。
def acquire_runtime_locks(project_root: Path):
    # 训练全程持有声码器依赖锁与学生输出锁，退出才释放。
    """Hold the full custom-vocoder pipeline and student locks until exit."""

    # 所有锁文件集中放在项目 exp/locks 下，便于人工查看当前流程占用情况。
    lock_dir = project_root / "exp" / "locks"
    # 若目录不存在就递归创建；已存在时不报错。
    lock_dir.mkdir(parents=True, exist_ok=True)
    # 每项包含锁文件路径和抢锁失败时给用户看的明确原因。
    lock_specs = (
        (  # 第一把锁覆盖完整声码器流水线。
            lock_dir / "text_unseen_custom_hifigan_pipeline.lock",  # 覆盖声码器训练/评估全流程。
            "custom HiFi-GAN pipeline/evaluation is still running",  # 表示声码器流水线仍占用资源。
        ),
        (  # 第二把锁覆盖声码器训练进程。
            lock_dir / "text_unseen_custom_hifigan_train.lock",  # 单独保护 HiFi-GAN 训练进程。
            "custom HiFi-GAN training is still running",  # 提醒先等待声码器训练退出。
        ),
        (  # 第三把锁覆盖本学生训练进程。
            lock_dir / "text_unseen_fastvoicegrad_student.lock",  # 防止两个学生训练同时启动。
            "another FastVoiceGrad student process is already running",  # 重复运行时的错误说明。
        ),
    )
    # 保存打开的文件句柄；句柄活着，Linux flock 才会一直有效。
    handles = []
    # 如果中途任一锁失败，外层 try 会负责释放之前已成功获取的锁。
    try:
        # 依次尝试三个锁，顺序固定可减少并发死锁风险。
        for path, message in lock_specs:
            # 以追加/读取模式打开锁文件，不会清空其中已有内容。
            handle = path.open("a+", encoding="utf-8")
            # 内层 try 专门处理当前这一个锁的占用错误。
            try:
                # LOCK_EX=独占锁；LOCK_NB=立即返回，已有进程占用时不无限等待。
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # BlockingIOError 表示另一个活进程已经持有这把锁。
            except BlockingIOError as error:
                # 当前锁没拿到，先关闭刚打开的句柄，避免文件描述符泄漏。
                handle.close()
                # 用人能看懂的流程冲突说明包装底层错误并终止启动。
                raise RuntimeError(message) from error
            # 只有成功抢到锁才把句柄加入列表并保留到进程结束。
            handles.append(handle)
    # 捕获任何启动异常，保证此前取得的锁全部释放。
    except Exception:
        # 按获取的相反顺序关闭，逻辑与资源栈一致。
        for handle in reversed(handles):
            # 关闭句柄会由操作系统自动释放对应 flock。
            handle.close()
        # 保留原始异常和回溯继续向上传递。
        # 保留原异常继续抛出，让调用方和日志看到真正失败原因。
        raise
    # 返回句柄列表；main 中一直持有它们，直到训练进程退出。
    return handles


# 调用仓库的完整只读审计，核对所有 manifest、特征和 text-unseen 协议。
def run_full_data_audit(project_root: Path, expected_protocol: str) -> None:
    # 正式训练前执行完整只读数据与特征审计，不修改已有实验产物。
    """Run the repository's exhaustive, read-only feature/protocol validator."""

    # 使用独立子进程运行审计模块；check=True 使任何失败都阻止训练。
    subprocess.run(
        [  # 子进程命令按列表传递，不经过 shell 字符串解释。
            sys.executable,  # 使用当前环境的 Python，避免包版本不一致。
            "-u",  # 关闭标准输出缓冲，让 VS Code 立即看到审计进度。
            "-m",  # 以模块方式运行，正确使用仓库的 Python 包路径。
            "voicegrad.validate_setup",  # 现有的权威审计入口。
            "--project-root",  # 下一项是项目根目录参数。
            str(project_root),  # Path 转字符串交给子进程。
            "--expected-protocol",  # 下一项声明必须采用的拆分协议。
            expected_protocol,  # 固定要求 vctk-text-unseen。
            "--full",  # 全量检查内容指纹，而不是只抽查文件是否存在。
        ],
        # 子进程的工作目录。
        cwd=str(project_root),  # 从项目根目录运行，保证相对路径一致。
        # 子进程失败时是否立即抛异常。
        check=True,  # 审计退出码非 0 时抛异常，绝不带病开始训练。
    )


# 对任意 JSON 可表示对象计算与键顺序无关的稳定 SHA-256。
def canonical_json_sha256(value: Any) -> str:
    # 键排序、紧凑分隔且保留 Unicode，确保等价对象得到相同字节串。
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False  # 稳定 JSON 文本。
    ).encode("utf-8")  # 统一编码为字节后才能计算 SHA-256。
    # 返回 64 位十六进制摘要，供运行契约和协议语义比对。
    return hashlib.sha256(encoded).hexdigest()


# 原子写 JSON：先完整写临时文件，再一次替换正式文件，防止断电留下半个 JSON。
def atomic_write_json(path: Path, value: Any) -> None:
    # 递归创建父目录；已存在时保持不变。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件与目标在同一目录，从而保证 os.replace 的原子性。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 以 UTF-8 文本方式打开临时文件。
    with temporary.open("w", encoding="utf-8") as handle:
        # 缩进 2 空格、键排序，方便人阅读和复核；中文不转义。
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        # 补标准结尾换行，便于终端和 Git 工具显示。
        handle.write("\n")
        # 将 Python 缓冲区内容推给操作系统。
        handle.flush()
        # 要求操作系统把内容真正同步到磁盘，降低突然关机时丢失风险。
        os.fsync(handle.fileno())
    # 原子替换目标：外界只会看到旧完整文件或新完整文件。
    os.replace(temporary, path)


# 原子保存 PyTorch 检查点，避免训练中断产生无法续训的半写权重。
def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    # 确保检查点父目录存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用同目录 .tmp 文件，完成后再原子改名。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 复制为普通 dict 后交给 torch.save 序列化张量和优化器状态。
    torch.save(dict(payload), temporary)
    # 一次性替换正式文件，读者不会读到尚未写完的检查点。
    os.replace(temporary, path)


# 捕获所有随机数状态；保存/绘图/测试后可恢复，保证训练轨迹不受诊断操作影响。
def capture_rng_state() -> Dict[str, Any]:
    # 返回包含 Python、NumPy、PyTorch CPU 和全部 CUDA 状态的完整字典。
    return {
        "python": random.getstate(),  # Python random 当前内部状态。
        "numpy": np.random.get_state(),  # NumPy 全局生成器状态。
        "torch": torch.get_rng_state(),  # PyTorch CPU 生成器状态张量。
        "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],  # 每张 GPU 状态。
    }


# 将断点或诊断前捕获的随机状态完整恢复。
def restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    # 明确要求四个键一个不少，也不接受未知键，防止“不完全可复现”的续训。
    required = {"python", "numpy", "torch", "cuda_all"}
    # 集合不完全一致说明检查点格式损坏或来自旧实现。
    if set(state) != required:
        # 报错中列出正确键名，便于修复数据而不是静默继续。
        raise RuntimeError(f"Incomplete RNG state; expected keys {sorted(required)}")
    # 恢复 Python 随机数流到捕获时的精确位置。
    random.setstate(state["python"])
    # 恢复 NumPy 随机数流。
    np.random.set_state(state["numpy"])
    # 恢复 PyTorch CPU 随机数流。
    torch.set_rng_state(state["torch"])
    # CUDA 运行还必须恢复 GPU 随机数，CPU 预检查则无需处理。
    if device.type == "cuda":
        # CUDA 续训检查点若没有 GPU 状态就无法保证下一批噪声相同。
        if not state["cuda_all"]:
            # 明确拒绝不精确续训。
            raise RuntimeError("CUDA resume checkpoint has no CUDA RNG state")
        # 一次恢复所有可见 GPU 状态；本实验正式计算在租用 A100 上。
        torch.cuda.set_rng_state_all(state["cuda_all"])


# 在不扰动训练随机数的前提下，从 loss.csv 更新训练/测试 loss 图。
def render_loss_plots_without_rng_side_effects(
    loss_path: Path, output_dir: Path, device: torch.device  # CSV、图片目录和当前设备。
):
    # 隔离画图和导入产生的随机状态变化，保证训练随机序列可恢复。
    """Keep plotting/import side effects outside the formal training RNG stream."""

    # 绘图前保存完整随机状态，因为首次导入库也可能间接消耗随机数。
    state = capture_rng_state()
    # 无论绘图成功还是报错，finally 都会恢复训练状态。
    try:
        # 延迟导入绘图函数，避免仅做预检查时加载 Matplotlib。
        from .plot_loss_curves import plot_loss_curves

        # 读取 CSV 并返回生成的两张 PNG 路径。
        return plot_loss_curves(loss_path, output_dir)
    # finally 即使 return 或异常也必定执行。
    finally:
        # 恢复绘图前的所有随机数状态，使下一训练 batch 的噪声完全不变。
        restore_rng_state(state, device)


# 对当前数据集做第二层结构检查，并收集会写进运行契约的完整指纹。
def validate_data_protocol(
    project_root: Path, config: FastVoiceGradStudentConfig  # 项目根目录与论文严格配置。
# 结束 validate_data_protocol 的参数声明；返回类型为 Dict[str, Any]，具体内容由下方实现给出。
) -> Dict[str, Any]:
    # 所有训练/测试特征都应位于项目 data 目录。
    data_root = project_root / "data"
    # manifest、拆分协议和数据来源说明集中在 data/metadata。
    metadata = data_root / "metadata"
    # split_protocol.json 明确记录当前启用的训练/测试拆分名称。
    protocol_path = metadata / "split_protocol.json"
    # 以 UTF-8 打开协议 JSON，with 结束时自动关闭文件。
    with protocol_path.open("r", encoding="utf-8") as handle:
        # 将 JSON 解析成 Python 字典供下面逐项核对。
        protocol = json.load(handle)
    # 当前协议名必须与配置要求的 vctk-text-unseen 完全相同。
    if protocol.get("protocol_cli_name") != config.expected_protocol:
        # 不允许误用 seen、speech-unseen 或其他拆分继续训练。
        raise RuntimeError(
            f"Active protocol must be {config.expected_protocol!r}; "  # 显示要求协议。
            f"got {protocol.get('protocol_cli_name')!r}"  # 显示实际协议。
        )

    # 训练 manifest 列出所有用于优化器更新的语音。
    train_path = metadata / "train_manifest.csv"
    # 测试 manifest 列出 10 位未见说话人各 10 句，共 100 条。
    test_path = metadata / "test_manifest.csv"
    # all manifest 应恰好是训练与测试的无重叠并集。
    all_path = metadata / "all_manifest.csv"
    # 读取训练样本行，每行至少包含 speaker 和 utt_id。
    train = read_manifest(train_path)
    # 读取测试样本行。
    test = read_manifest(test_path)
    # 读取全量样本行。
    all_rows = read_manifest(all_path)
    # 统计训练集中每位说话人的句数；Counter 的键数就是说话人数。
    train_speakers = Counter(row["speaker"] for row in train)
    # 同样统计测试说话人及每人的测试句数。
    test_speakers = Counter(row["speaker"] for row in test)
    # 用 (说话人, 句子ID) 二元组作为训练样本的唯一身份。
    train_ids = {(row["speaker"], row["utt_id"]) for row in train}
    # 构造测试样本唯一身份集合，用于查重和泄漏检查。
    test_ids = {(row["speaker"], row["utt_id"]) for row in test}
    # 构造全量 manifest 的唯一身份集合。
    all_ids = {(row["speaker"], row["utt_id"]) for row in all_rows}
    # 集合长度小于原始行数说明同一身份被重复列出。
    if len(train_ids) != len(train) or len(test_ids) != len(test) or len(all_ids) != len(all_rows):
        # 重复样本会改变样本权重，因此直接终止严格复现。
        raise RuntimeError("Duplicate speaker/utterance key in an active manifest")
    # 训练/测试不能共享语句，all 也不能多一条或少一条。
    if train_ids.intersection(test_ids) or all_ids != train_ids.union(test_ids):
        # 任一条件失败都说明拆分存在泄漏或清单不完整。
        raise RuntimeError("all_manifest must be the exact disjoint union of train and test")
    # 论文设置要求训练集恰好包含 100 位说话人。
    if len(train_speakers) != config.train_speakers:
        # 显示要求人数和实际人数，方便检查拆分生成步骤。
        raise RuntimeError(
            f"Strict paper mode requires {config.train_speakers} train speakers; "  # 期望训练人数。
            f"found {len(train_speakers)}"  # 实际训练人数。
        )
    # text-unseen 测试集必须恰好包含 10 位训练中从未出现的说话人。
    if len(test_speakers) != config.test_speakers:
        # 人数不符时拒绝生成不可与论文比较的结果。
        raise RuntimeError(
            f"Strict paper mode requires {config.test_speakers} test speakers; "  # 期望测试人数。
            f"found {len(test_speakers)}"  # 实际测试人数。
        )
    # 收集句数不是 10 的测试说话人；正常情况下这个字典为空。
    wrong_test_counts = {
        speaker: count  # 保留错误说话人名及其实际句数，便于诊断。
        for speaker, count in test_speakers.items()  # 逐位遍历测试说话人计数。
        if count != config.test_sentences_per_speaker  # 只选出不等于每人 10 句者。
    }
    # 同时检查逐说话人句数和总数 10×10=100，二者缺一不可。
    if wrong_test_counts or len(test) != (
        # 固定测试覆盖是 10 位说话人各 10 句，共 100 条。
        config.test_speakers * config.test_sentences_per_speaker  # 10×10=100。
    ):
        # 报告异常计数；若只有总数异常，则打印完整 Counter。
        raise RuntimeError(
            "Strict paper test grid requires exactly 10 utterances for each of 10 "  # 正确网格定义。
            f"speakers; got {wrong_test_counts or dict(test_speakers)}"  # 实际异常计数。
        )
    # 取训练/测试说话人集合交集；text-unseen 的交集必须为空。
    overlap = set(train_speakers).intersection(test_speakers)
    # 若交集非空，说明测试说话人在训练中出现过，违反核心实验协议。
    if overlap:
        # 列出泄漏说话人，立即中止。
        raise RuntimeError(f"Train/test speaker leakage detected: {sorted(overlap)}")

    # 定义最关键的三份清单和两份 mel 统计量，逐文件计算 SHA-256。
    fingerprint_paths = {
        # 训练语音清单指纹。
        "train_manifest.csv": train_path,  # 锁定训练样本和顺序。
        # 测试语音清单指纹。
        "test_manifest.csv": test_path,  # 锁定测试样本和顺序。
        "all_manifest.csv": all_path,  # 锁定全数据全集。
        # 当前代次均值文件及其字节身份。
        "mel_mean.npy": data_root / "stats/mel_mean.npy",  # 80 维训练 mel 均值。
        # 当前代次标准差文件及其字节身份。
        "mel_std.npy": data_root / "stats/mel_std.npy",  # 80 维训练 mel 标准差。
    }
    # 文件内容只要改变一个字节，对应指纹就会不同。
    fingerprints = {
        key: sha256_file(path) for key, path in fingerprint_paths.items()  # 逐文件哈希。
    }
    # 另存协议“语义”哈希，只包含必须一致的协议名而不受 JSON 排版影响。
    fingerprints["split_protocol.semantic_sha256"] = canonical_json_sha256(
        {"protocol_cli_name": config.expected_protocol}  # 固定 text-unseen 语义。
    )
    # 加入 mel 均值/标准差发布清单指纹；这里核元数据，完整内容已由外层审计检查。
    fingerprints.update(
        audit_mel_stats_publication(project_root, train, check_content=False)  # 返回统计量发布指纹。
    )
    # 加入 mel 提取配方和全量/测试集合对应关系的审计指纹。
    fingerprints.update(audit_mel_extraction_recipe(project_root, all_rows, test))
    # 缓存输入 wav 的审计结果，避免 BNF 和说话人嵌入核对时重复读同一音频。
    input_wav_cache: Dict[str, Dict[str, object]] = {}
    # 两类条件特征都必须能追溯到同一批已登记原始音频。
    for feature_kind in ("bnf", "speaker_embedding"):
        # 将这一类特征的来源/布局指纹合并进总指纹字典。
        fingerprints.update(
            audit_feature_provenance(  # 执行当前特征种类的来源审计。
                project_root,  # 从当前项目根目录解析审计文件。
                all_rows,  # 覆盖 train+test 的全部语音。
                feature_kind,  # 本轮是 bnf 或 speaker_embedding。
                check_content=False,  # 内容全检已在 run_full_data_audit 完成，避免重复耗时。
                input_wav_cache=input_wav_cache,  # 复用已读取的 wav 元数据。
            )
        )
    # 返回可 JSON 序列化的数据契约，后面会写进 run_contract.json。
    return {
        # 当前数据清单声明的划分协议。
        "protocol": config.expected_protocol,  # 实验协议名 vctk-text-unseen。
        # 实际训练语音条数。
        "train_rows": len(train),  # 训练语音总条数，本次应为 39,387。
        # 训练使用100个说话人。
        "train_speakers": len(train_speakers),  # 训练说话人数，应为 100。
        # 测试集语音条数。
        "test_rows": len(test),  # 测试语音总条数，应为 100。
        # 测试说话人数。
        "test_speakers": len(test_speakers),  # 测试说话人数，应为 10。
        # 整个已发布数据清单的语音总数。
        "all_rows": len(all_rows),  # 全部语音条数，应等于 train_rows+test_rows。
        # 划分协议文件字节指纹。
        "split_protocol_file_sha256": sha256_file(protocol_path),  # 协议文件原始字节哈希。
        # 预处理产物及协议内容指纹。
        "fingerprints": fingerprints,  # 完整的多项数据来源指纹表。
    }


# 确认教师和声码器确实由当前这套 text-unseen 数据及 mel 统计量训练得到。
def validate_dependency_data_fingerprints(
    data_contract: Mapping[str, Any],  # 刚从当前磁盘数据重新计算的指纹契约。
    teacher_checkpoint: Mapping[str, Any],  # epoch500 教师检查点中的历史元数据。
    vocoder_checkpoint: Mapping[str, Any],  # 250 万步声码器检查点中的历史元数据。
# 结束 validate_dependency_data_fingerprints 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # active 是当前准备送入学生训练的数据指纹真值。
    active = data_contract["fingerprints"]
    # 教师检查点保存了训练当时的完整 31 键数据指纹。
    teacher_saved = teacher_checkpoint.get("run_config", {}).get("data_fingerprints", {})
    # 必须逐键完全相同，才可证明教师与当前学生数据来自同一版本。
    if teacher_saved != active:
        # 任一数据文件变化都拒绝混用教师，避免结果来源不清。
        raise RuntimeError(
            "Teacher checkpoint does not match the complete 31-key active data provenance"  # 31 项需全等。
        )

    # 声码器检查点采用另一套字段命名，下面建立显式映射逐项核对。
    vocoder_saved = vocoder_checkpoint.get("data_fingerprints", {})
    # 左边是声码器字段，右边是当前数据契约中的对应字段。
    vocoder_to_active = {
        "train_manifest_sha256": "train_manifest.csv",  # 训练清单。
        # 测试语音清单指纹。
        "test_manifest_sha256": "test_manifest.csv",  # 测试清单。
        "all_manifest_sha256": "all_manifest.csv",  # 全量清单。
        # 梅尔均值文件指纹。
        "mel_mean_sha256": "mel_mean.npy",  # mel 均值文件。
        # 梅尔标准差文件指纹。
        "mel_std_sha256": "mel_std.npy",  # mel 标准差文件。
        # 参与统计的全部梅尔内容集合指纹。
        "mel_collection_content_sha256": "mel_collection.content_sha256",  # mel 内容集合。
        # 参与统计的梅尔路径和大小集合指纹。
        "mel_collection_layout_sha256": "mel_collection.layout_sha256",  # mel 路径布局。
    }
    # 遍历每个对应关系，不能只检查权重步数而忽略训练数据身份。
    for vocoder_key, active_key in vocoder_to_active.items():
        # 当前值和声码器保存值不同就说明声码器不是为这套 mel 训练的。
        if vocoder_saved.get(vocoder_key) != active.get(active_key):
            # 同时打印两端值，便于找出具体不一致项目。
            raise RuntimeError(
                f"Vocoder data fingerprint mismatch for {vocoder_key}: "  # 冲突字段名。
                f"vocoder={vocoder_saved.get(vocoder_key)!r}, "  # 权重历史值。
                f"active={active.get(active_key)!r}"  # 当前数据值。
            )
    # 声码器保存的拆分协议文件原始哈希也必须一致。
    if vocoder_saved.get("split_protocol_sha256") != data_contract.get(
        "split_protocol_file_sha256"  # 当前协议文件的原始 SHA-256。
    ):
        # 协议文件不同可能代表训练/测试定义不同，因此立即停止。
        raise RuntimeError("Vocoder split-protocol file fingerprint mismatch")


# 对所有会影响学生结果的源文件计算 SHA-256，形成可追溯代码快照。
def collect_source_fingerprints(project_root: Path) -> Dict[str, str]:
    # 列表同时覆盖学生、教师、数据、声码器和启动脚本，不能只哈希当前文件。
    relative_paths = [
        "fastvoicegrad/__init__.py",  # 学生包入口。
        "fastvoicegrad/config.py",  # 学生论文参数与未公开项声明。
        "fastvoicegrad/components.py",  # 模型加载、判别器和安全检查。
        "fastvoicegrad/objectives.py",  # ACDD 全部数学目标。
        "fastvoicegrad/loss_monitor.py",  # 每批训练趋势记录及报警。
        "fastvoicegrad/plot_loss_curves.py",  # train/test loss 作图逻辑。
        "fastvoicegrad/train.py",  # 本学生训练主程序。
        "voicegrad/config.py",  # 教师网络/声学配置。
        "voicegrad/dataset.py",  # mel/BNF/说话人向量读取与训练裁剪。
        "voicegrad/diffusion.py",  # 扩散过程公式。
        "voicegrad/feature_alignment.py",  # 特征时间轴对齐规则。
        "voicegrad/model.py",  # VoiceGrad 教师/学生共享 U-Net 结构。
        "voicegrad/runtime_guard.py",  # 租用服务器运行保护。
        "voicegrad/train.py",  # 教师训练实现，影响教师权重解释。
        "voicegrad/validate_setup.py",  # 数据完整性审计。
        "hifigan/custom/dataset.py",  # 修改版声码器数据读取。
        "hifigan/custom/mrd.py",  # UnivNet 多分辨率判别器实现。
        "hifigan/custom/train.py",  # 修改版 HiFi-GAN 训练实现。
        "hifigan/custom/validate_checkpoint.py",  # 声码器权重验真。
        "hifigan/official/env.py",  # HiFi-GAN 配置环境工具。
        "hifigan/official/models.py",  # Generator 和 MPD 基础结构。
        "hifigan/official/utils.py",  # HiFi-GAN 权重等工具函数。
        "scripts/common/06_train_fastvoicegrad_student.sh",  # 通用学生启动脚本。
        "scripts/common/require_rented_server.sh",  # shell 层服务器保护。
        "text_unseen/06_train_student.sh",  # text-unseen 一键入口。
    ]
    # 返回“相对路径→文件哈希”字典，写入 run_contract 后可逐文件复核。
    return {
        relative: sha256_file(project_root / relative) for relative in relative_paths  # 逐文件计算。
    }


# 要求正式运行来自干净 Git 提交，并返回该提交 ID。
def git_head(project_root: Path) -> str:
    # `git status --porcelain` 用稳定、机器可读格式列出未提交改动。
    status = subprocess.run(
        ["git", "status", "--porcelain"],  # 只查询状态，不修改仓库。
        # 子进程的工作目录。
        cwd=str(project_root),  # 在权威项目根目录查询。
        # 子进程失败时是否立即抛异常。
        check=True,  # Git 命令失败就终止。
        # 语音转写文本或文本模式标记，具体取决于当前调用。
        text=True,  # 将输出解码为 Python 字符串。
        stdout=subprocess.PIPE,  # 捕获状态结果供程序判断。
        stderr=subprocess.PIPE,  # 捕获错误信息并保留在异常中。
    )
    # 任意非空输出都代表有未提交、未跟踪或已暂存变化。
    if status.stdout.strip():
        # 拒绝从无法唯一追溯的“脏工作区”启动正式实验。
        raise RuntimeError(
            "Formal FastVoiceGrad training requires a clean committed Git worktree"  # 可追溯性要求。
        )
    # 工作区干净后查询当前 HEAD 的完整提交哈希。
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # 只读获取当前提交 ID。
        # 子进程的工作目录。
        cwd=str(project_root),  # 使用同一项目目录。
        # 子进程失败时是否立即抛异常。
        check=True,  # 无有效 Git 提交时停止。
        # 语音转写文本或文本模式标记，具体取决于当前调用。
        text=True,  # 输出为文本。
        stdout=subprocess.PIPE,  # 捕获提交哈希。
        stderr=subprocess.PIPE,  # 捕获 Git 错误。
    )
    # 去掉末尾换行，返回可直接写入 JSON 的 40 位提交哈希。
    return result.stdout.strip()


# 汇总论文明确项、复现自定项、依赖哈希、数据身份和运行环境，形成实验“身份证”。
def build_run_contract(
    project_root: Path,  # 权威服务器项目根目录。
    args: argparse.Namespace,  # 已解析并规范化的命令行路径/设备设置。
    config: FastVoiceGradStudentConfig,  # 不可随意变化的论文严格配置。
    data_contract: Mapping[str, Any],  # 上一步验证过的数据数量和指纹。
    teacher_checkpoint: Mapping[str, Any],  # 教师权重元数据，用于记录真实训练步数。
    vocoder_checkpoint: Mapping[str, Any],  # 自训练声码器元数据。
# 结束 build_run_contract 的参数声明；返回类型为 Dict[str, Any]，具体内容由下方实现给出。
) -> Dict[str, Any]:
    # 返回纯 Python 字典，随后会规范哈希并原子保存为 run_contract.json。
    return {
        # 明确区分FastVoiceGrad text-unseen学生实验。
        "experiment": "FastVoiceGrad VCTK text-unseen ACDD student",  # 实验唯一名称。
        # paper 区记录论文明确写出的核心方法，便于和自定项分开说明。
        "paper": {
            "title": "FastVoiceGrad: One-step Diffusion-Based Voice Conversion with Adversarial Conditional Diffusion Distillation",  # 论文全名。
            "interspeech_year": 2024,  # 论文发表年份/会议届次。
            # 已在论文中明确给出的设定。
            "paper_explicit": dict(config.paper_explicit()),  # config 中逐项登记的论文原文设置。
            "objective": "L_adv(theta) + 2*L_FM(theta) + 45*L_dist(theta)",  # 学生总目标及权重。
            "discriminator_objective": "L_adv(D)",  # 判别器独立最小二乘对抗目标。
            "frozen": ["VoiceGrad teacher phi", "neural vocoder Generator V"],  # 只提供监督、不更新。
            "trainable": ["FastVoiceGrad student theta", "waveform discriminator D"],  # 两组会更新参数。
        },
        # reference_derived 记录由论文描述和官方/参考代码共同确定的实现细节。
        "reference_derived": {
            "student_network": "existing audited any-to-any VoiceGrad U-Net",  # 学生沿用已审计 U-Net。
            "discriminator": "HiFi-GAN V1 MPD + UnivNet MRD; MSD removed",  # 5 个 MPD+3 个 MRD。
            "feature_matching": "per-layer mean absolute error summed over MPD+MRD",  # 各层 L1 均值求和。
            "mel_to_vocoder": "denormalize with training mel mean/std before frozen V",  # 声码器输入还原尺度。
        },
        # 论文未说明的细节不伪装成论文设置，而是完整登记我们的可复现选择。
        "paper_undisclosed_declared_choices": dict(
            # 把论文未明确给出的实现选择完整写进运行记录。
            config.declared_reproduction_choices()
        ),
        # 单独记录教师目标和alpha解释，让修复结果可与旧实验区分。
        "score_distillation_interpretation": {
            # 保存实际传入损失函数的教师目标模式。
            "target_mode": config.score_target,
            # 保存实际损失权重名称。
            "weight_mode": config.score_weight,
            # 旧对照仍可保存，但会明确标成消融，不能混入修复主实验的结果。
            "profile": (
                "clean_x0_signal_amplitude_v1"
                if (config.score_target, config.score_weight) == ("clean_x0", "sqrt_alpha_bar")
                else "explicit_score_ablation"
            ),
            # 记录实际使用的公式；旧对照必须如实登记相邻步，不能写成干净目标。
            "target_formula": (
                "(x_t - sqrt(1-alpha_bar_t)*epsilon_teacher(x_t,t,s,p)) / sqrt(alpha_bar_t)"
                if config.score_target == "clean_x0"
                else "(x_t - (1-alpha_t)/sqrt(1-alpha_bar_t)*epsilon_teacher(x_t,t,s,p)) / sqrt(alpha_t)"
            ),
            # 教师目标整条分支不反传；学生仍通过L1差值直接获得梯度。
            "gradient_boundary": "detach student before re-noising; no_grad teacher target; direct student L1 gradient",
            # 保留已有归约和外层权重，不额外叠加ADD的L2或其他经验系数。
            "reduction": "mean over mel channels/frames per example, multiply c(t), then mean over examples; outer lambda_dist=45",
            # 用文字说明目标所在的扩散状态，便于不读代码也能核对。
            "target_space": (
                # 修复目标直接还原干净x0，其跳步系数是累计alpha_bar。
                "clean x_0 predicted from x_t with effective alpha=alpha_bar_t"
                # 当前选择干净目标时记录这条解释。
                if config.score_target == "clean_x0"
                # 旧模式如实记录比较了相邻含噪目标和干净学生输出的空间差异。
                else "legacy adjacent x_(t-1) reverse mean compared against clean student x_0"
            ),
            # 展开权重的数学含义，避免报告只出现难以区分的alpha符号。
            "weight": (
                # 平方根累计权重采用同作者后续回顾论文的解释。
                "sqrt(alpha_bar_t), as clarified for FastVoiceGrad in the authors' 2025 paper"
                # 先判断是否为修复主设定的信号幅度权重。
                if config.score_weight == "sqrt_alpha_bar" else
                # 累计功率解释仍作为可复现的对照选择。
                "alpha_bar_t (effective coefficient of the selected t -> 0 transition)"
                # 当前选择累计功率时使用这条文字说明。
                if config.score_weight == "alpha_bar"
                # 剩余合法模式为旧相邻日程的单步保留量。
                else "nominal alpha_t=1-beta_t of the adjacent schedule"
            ),
            # 明确指出哪些解释尚未得到作者原始训练代码的独立验证。
            "paper_status": (
                # 记录2024论文公式13引用公式7，第二节又对跳步alpha进行重定义的背景。
                "Equation 13 references Equation 7; Section 2 redefines alpha for sparse "
                # 说明教师目标的反向步跨度和alpha含义需要在复现时单独交代。
                "transitions. The teacher-target transition and alpha interpretation "
                # 声明记录的是公开披露的复现解释，避免夸大为已核实作者实现。
                "are disclosed reproduction choices, not independently verified author code."
            ),
            # 保留2025年同作者明确写出sqrt(alpha_bar)的官方出处。
            "later_author_clarification": "https://www.isca-archive.org/interspeech_2025/kaneko25_interspeech.pdf Section 2.2, Equation 6",
            # 原论文引用的ADD明确写出干净重建目标，提供目标空间的补充依据。
            "clean_target_reference": "https://stability.ai/s/adversarial_diffusion_distillation.pdf page 5, Section 3.3, Equation 4 and following reconstruction formula",
        },
        # 记录报警观察的训练数据和规则，便于解释通知依据。
        "training_loss_monitor": {
            # 窗口长度来自当前配置，默认50个批次。
            "window_steps": config.loss_monitor_window_steps,
            # 监控输入是每次更新对应的训练批次损失。
            "input": "per-training-batch losses after each optimizer update",
            # 监控只保存记录与报警，不会用测试集结果决定梯度或检查点。
            "action": "record and alert; does not alter gradients or select a checkpoint",
        },
        # 这一段专门说明用户要求的每轮测试 loss 如何计算，以及它不会参与选模。
        "held_out_test_loss_instrumentation": {
            "paper_status": "not specified by the paper; user-requested diagnostic",  # 属于额外记录工具。
            # 当前数据清单声明的划分协议。
            "protocol": config.expected_protocol,  # 仍严格使用 vctk-text-unseen。
            # 此次提取所覆盖的清单身份。
            "manifest": "data/metadata/test_manifest.csv",  # 唯一测试样本来源。
            # 测试样本数=10 位说话人×每人 10 句=100。
            "examples": config.test_speakers
            # 每位测试说话人数乘测试句数，得到100条固定诊断语音。
            * config.test_sentences_per_speaker,
            "speakers": config.test_speakers,  # 10 位训练未见说话人。
            "sentences_per_speaker": config.test_sentences_per_speaker,  # 每人 10 句。
            "segment_frames": config.test_loss_frames,  # 每句固定 128 帧以匹配训练形状。
            "crop": "sha256(seed, speaker, utterance) fixed across epochs",  # 确定性裁剪规则。
            # 每个训练批的样本数。
            "batch_size": config.test_loss_batch_size,  # 测试 batch 最大 32。
            "drop_last": False,  # 最后一批 4 条也必须统计。
            "expected_batch_sizes": [32, 32, 32, 4],  # 100 条的实际四批布局。
            "reduction": config.test_loss_reduction,  # 按样本加权平均，短尾批不会权重过大。
            "randomness": "fixed independent source-noise, score-noise, and timestep generators",  # 独立固定噪声。
            "student_metrics": list(LOSS_COMPONENTS[:-1]),  # 总损失及三个学生目标的原始/加权值。
            # 判别器测试 loss 只看泛化趋势，不加入学生总损失。
            "discriminator_metric": (
                # 固定测试上的辅助诊断值不纳入学生训练总损失。
                "held-out diagnostic only; excluded from student_total"
            ),
            "frequency_epochs": config.test_loss_every_epochs,  # 每 1 个 epoch 测一次并画点。
            "affects_optimizer_or_rng": False,  # 不反传、不 step，并恢复训练随机状态。
            "early_stopping": False,  # 不根据测试结果提前停止。
            "checkpoint_selection": config.test_loss_checkpoint_selection,  # 明确禁止按测试 loss 选模型。
            "reported_checkpoint": "epoch_0100 regardless of test loss",  # 固定报告第 100 轮。
        },
        # strict_guards 记录程序主动强制的核心不可变条件。
        "strict_guards": {
            "teacher_epoch_must_equal": config.teacher_epoch,  # 教师必须为 epoch500。
            "vocoder_step_must_equal": config.expected_vocoder_step,  # 声码器必须 250 万步。
            "global_batch_must_equal": config.batch_size,  # 全局 batch 必须为 32。
            "gradient_accumulation": False,  # 没有用小 batch 累积冒充 batch32。
            # D 在完整 batch32 上依次处理 5 个 MPD 与 3 个 MRD，再只 step 一次。
            "discriminator_gradient_schedule": (
                # 记录判别器逐分支累加完整batch32梯度，全部结束后仅更新一次。
                "full_batch32_serial_MPD5_then_MRD3_sum_then_one_Adam_step"
            ),
            # 学生同样保持完整 batch32，只串行释放判别器特征图以节省 A100 显存。
            "student_gradient_schedule": (
                # 记录学生波形梯度先逐分支求和，再一次回传并更新学生。
                "full_batch32_exact_waveform_VJP_serial_MPD5_then_MRD3_then_one_Adam_step"
            ),
            # 串行仅改变自动微分执行顺序，数学目标、精度和更新次数都不变。
            "gradient_schedule_effect": (
                # 声明顺序反传仅降低显存需求，目标、批量、精度和更新次数均相同。
                "memory-only autograd scheduling; objective, batch, precision, and update counts unchanged"
            ),
            "automatic_mixed_precision": False,  # 使用 FP32，不启用混合精度近似。
            "scheduler": False,  # 学习率恒定为 2e-4，不额外衰减。
            "early_stopping": False,  # 固定训练 100 轮。
            "test_loss_controls_training": False,  # 测试曲线不会改变梯度或保存选择。
            "extra_student_losses": [],  # 不加入论文目标之外的学生损失。
        },
        # dependencies 精确记录三套前置产物的绝对路径、哈希和实际步数。
        "dependencies": {
            "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),  # epoch500 教师绝对路径。
            "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint.resolve()),  # 教师文件指纹。
            "teacher_global_step": int(teacher_checkpoint.get("global_step", -1)),  # 教师累计更新数。
            # 本次冻结声码器的最终权重文件。
            "vocoder_checkpoint": str(args.vocoder_checkpoint.resolve()),  # 自训练声码器绝对路径。
            # 冻结声码器权重文件指纹。
            "vocoder_checkpoint_sha256": sha256_file(args.vocoder_checkpoint.resolve()),  # 声码器指纹。
            # 声码器已完成2500000次训练更新。
            "vocoder_global_step": int(vocoder_checkpoint.get("global_step", -1)),  # 必须为 2,500,000。
            # 声码器的结构与音频配置。
            "vocoder_config": str(args.vocoder_config.resolve()),  # 声码器结构/声学配置路径。
            # 声码器配置文件指纹。
            "vocoder_config_sha256": sha256_file(args.vocoder_config.resolve()),  # 配置内容指纹。
            # 声码器250万步最终检查点的审核记录。
            "vocoder_attestation": str(args.vocoder_attestation.resolve()),  # 最终验真 JSON 路径。
            # attestation 自身也哈希，避免验真记录被替换。
            "vocoder_attestation_sha256": sha256_file(
                # 将声码器审核记录转成绝对路径，便于之后核对同一文件。
                args.vocoder_attestation.resolve()
            ),
        },
        # data 既保留完整指纹，也明确记录 DataLoader 会怎样使用 39,387 条训练数据。
        "data": {
            **dict(data_contract),  # 展开协议、数量和所有数据指纹。
            "train_shuffle": config.train_shuffle,  # 每轮训练会随机打乱样本。
            "train_drop_last": config.train_drop_last,  # 丢弃不足 32 的尾批，保持全局 batch32。
            "train_batch_size": config.batch_size,  # 论文 batch size=32。
            # 整除得到每轮完整 batch 数：39387//32=1230。
            "train_batches_per_epoch": int(data_contract["train_rows"])
            # 只计算完整训练批次，与drop_last=True保持一致。
            // config.batch_size,
            # 余数得到每轮尾部丢弃数：39387%32=27，下一轮打乱后不是固定样本。
            "train_examples_dropped_per_epoch": int(data_contract["train_rows"])
            # 记录每轮丢弃的不完整尾批样本数。
            % config.batch_size,
            # 总学生更新数=每轮 1230×100 轮=123,000。
            "total_student_updates": (
                # 由已核实的训练语音数推算每个epoch实际更新次数。
                int(data_contract["train_rows"]) // config.batch_size
            )
            # 乘计划训练轮数，得到完整100轮对应的优化器总步数。
            * config.epochs,
        },
        # runtime 锁定代码版本和实际软件/CUDA 环境，便于以后完整复查。
        "runtime": {
            # 测试用项目根目录。
            "project_root": str(project_root),  # 租用服务器上的权威代码根目录。
            # 当前干净工作树对应的Git提交。
            "git_commit": git_head(project_root),  # 当前干净提交的唯一哈希。
            "git_worktree_clean": True,  # git_head 已实际检查，无未提交代码。
            "source_sha256": collect_source_fingerprints(project_root),  # 每个关键源码文件指纹。
            "torch": torch.__version__,  # 实际 PyTorch 版本。
            "cuda_runtime": torch.version.cuda,  # PyTorch 所使用的 CUDA 运行时版本。
            "device_request": args.device,  # 正式训练应为 cuda。
        },
    }


# 通用数值安全门：任何张量出现 NaN/Inf 都立刻停止，避免污染后续权重和曲线。
def _assert_finite(name: str, value: torch.Tensor) -> None:
    # torch.isfinite 逐元素检查，all 要求每个元素都是正常有限数。
    if not bool(torch.isfinite(value).all()):
        # name 标明是哪一步出错，便于从训练日志定位。
        raise FloatingPointError(f"{name} contains NaN or Inf")


# 将一个正式训练 batch 搬到 A100，并严格验证论文要求的 B=32 及特征维度。
def _slice_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    # 学生预测最少需要目标 mel、内容 BNF 和目标说话人嵌入三项。
    required = {"mel", "bnf", "spk_emb"}
    # issubset 检查输入字典是否一个必需键都不少。
    if not required.issubset(batch):
        # 差集会列出具体缺失字段，避免含糊的 KeyError。
        raise KeyError(f"Batch lacks required keys: {sorted(required - set(batch))}")
    # 仅提取必需张量并异步复制到当前设备；形状在下方逐项检查。
    result = {
        # 只搬运必须的梅尔、BNF 和说话人张量到租用 GPU，保持同一批样本配对。
        key: batch[key].to(device, non_blocking=True) for key in required  # CPU→A100。
    }
    # mel 是归一化目标声谱图，期望 [32,80,128]。
    mel = result["mel"]
    # BNF 是语音内容特征，期望 [32,144,128]。
    bnf = result["bnf"]
    # speaker 是目标说话人表示，期望 [32,256]。
    speaker = result["spk_emb"]
    # 正式训练不允许更小 batch 或梯度累积；80 是 mel 维，128 是时间帧数。
    if mel.shape != (PAPER_STUDENT_CONFIG.batch_size, 80, PAPER_STUDENT_CONFIG.train_frames):
        # 形状不符会改变目标函数统计或网络输入，因此立即终止。
        raise RuntimeError(
            # 形状不符合batch32、80通道、128帧时，明确说明训练接口要求。
            "Strict student batch must be [32,80,128] with no gradient accumulation; "
            # 同时打印收到的实际形状，方便定位数据裁剪或批量错误。
            f"got {tuple(mel.shape)}"
        )
    # BNF batch/时间必须跟 mel 一致，中间固定为 144 维。
    if bnf.shape != (mel.shape[0], 144, mel.shape[-1]):
        # 报告实际 BNF 形状，常见原因是特征对齐或 collate 出错。
        raise RuntimeError(f"BNF shape mismatch: {tuple(bnf.shape)}")
    # 每条样本必须有一个 256 维目标说话人嵌入。
    if speaker.shape != (mel.shape[0], 256):
        # 错误维度会导致条件网络语义不正确。
        raise RuntimeError(f"Speaker embedding shape mismatch: {tuple(speaker.shape)}")
    # 返回已在 GPU 上且通过严格形状检查的三个张量。
    return result


# 测试最后一批允许小于 32，但其余通道和时间维必须与正式训练一致。
def _slice_test_batch(
    batch: Mapping[str, Any],  # DataLoader 给出的测试字典。
    device: torch.device,  # 通常是租用 A100 的 cuda 设备。
    config: FastVoiceGradStudentConfig,  # 测试 batch 上限与固定帧数来源。
# 结束 _slice_test_batch 的参数声明；返回类型为 Dict[str, torch.Tensor]，具体内容由下方实现给出。
) -> Dict[str, torch.Tensor]:
    # 测试允许末尾小批，但训练仍严格要求完整的 32 条样本。
    """Validate a held-out batch without weakening the strict B=32 train gate."""

    # 测试 ACDD 损失需要与训练完全相同的三种输入。
    required = {"mel", "bnf", "spk_emb"}
    # 先确认字段完整，再访问字典。
    if not required.issubset(batch):
        # 列出缺失测试字段。
        raise KeyError(f"Test batch lacks required keys: {sorted(required - set(batch))}")
    # 将必需张量搬到评估设备，不处理说话人/句子 ID 等 CPU 元数据。
    result = {
        key: batch[key].to(device, non_blocking=True) for key in required  # 锁页内存时可异步复制。
    }
    # 测试 mel 应为 [B,80,128]。
    mel = result["mel"]
    # 测试 BNF 应为 [B,144,128]。
    bnf = result["bnf"]
    # 说话人嵌入应为 [B,256]。
    speaker = result["spk_emb"]
    # 三维分别代表 batch、mel 通道和时间；少/多一维都说明数据错误。
    if mel.ndim != 3:
        # 明确显示实际形状。
        raise RuntimeError(f"Test mel must be [B,80,128], got {tuple(mel.shape)}")
    # 从实际张量读取 B；正常四批依次为 32、32、32、4。
    batch_size = int(mel.shape[0])
    # 测试 B 至少 1 且不超过配置上限 32。
    if not 1 <= batch_size <= config.test_loss_batch_size:
        # 空 batch 或超大 batch 都拒绝。
        raise RuntimeError(f"Invalid held-out test batch size {batch_size}")
    # 除 B 外，mel 必须严格为 80 通道×128 帧。
    if tuple(mel.shape[1:]) != (80, config.test_loss_frames):
        # 维度不符时报告全部形状。
        raise RuntimeError(f"Test mel shape mismatch: {tuple(mel.shape)}")
    # BNF 的 B/T 与 mel 对齐，特征通道固定 144。
    if bnf.shape != (batch_size, 144, config.test_loss_frames):
        # 防止悄悄广播或错误时间对齐。
        raise RuntimeError(f"Test BNF shape mismatch: {tuple(bnf.shape)}")
    # 每个测试样本必须配一个 256 维目标说话人向量。
    if speaker.shape != (batch_size, 256):
        # 目标说话人条件错误会直接影响转换声音身份。
        raise RuntimeError(f"Test speaker embedding shape mismatch: {tuple(speaker.shape)}")
    # 返回通过检查的 GPU 张量。
    return result


# 在完整 batch32 上逐分支反传判别器，降低峰值显存且保持一次 Adam 更新。
def serial_discriminator_backward(
    discriminator: PaperWaveformDiscriminator,  # 含 5 个 MPD 和 3 个 MRD 的判别器集合。
    real_waveform: torch.Tensor,  # 声码器从真实 mel 合成的参考波形 [B,1,Twav]。
    fake_waveform: torch.Tensor,  # 声码器从学生 mel 合成的生成波形，形状相同。
# 结束 serial_discriminator_backward 的参数声明；返回类型为 torch.Tensor，具体内容由下方实现给出。
) -> torch.Tensor:
    # 判别器逐分支累积同一总损失的梯度，全部完成后只更新一次。
    """Backpropagate the unchanged D-loss one branch at a time, then step once.

    The five MPD and three MRD parameter sets are disjoint.  Summing their
    gradients before the single optimizer step is mathematically identical to
    backpropagating their summed scalar loss, while releasing every branch's
    feature maps before constructing the next branch.
    """

    # 真假波形必须逐样本、逐采样点对应，才能公平累加各分支损失。
    if real_waveform.shape != fake_waveform.shape:
        # 不允许依赖广播掩盖数据/声码器问题。
        raise ValueError("Real and fake waveforms must have identical shapes")
    # D 更新阶段至少一个判别器参数必须开启梯度。
    if not any(parameter.requires_grad for parameter in discriminator.parameters()):
        # 若全冻结，backward 无法给 D 产生可更新梯度。
        raise RuntimeError("Discriminator parameters must be trainable for its update")
    # 固定取出 8 个互不共享参数的分支：MPD5+MRD3。
    branches = tuple(discriminator.loss_branches())
    # 空分支说明判别器构建错误，不允许把零损失当正常训练。
    if not branches:
        # 立即停止。
        raise RuntimeError("Paper discriminator returned no MPD/MRD branches")
    # 创建与波形同设备/精度的标量 0，用来累计日志损失，不保留计算图。
    total = real_waveform.new_zeros(())
    # 逐分支构图、反传并释放特征图，避免 8 个大图同时占满 40GB 显存。
    for branch in branches:
        # 当前分支处理真实波形，得到判别分数和中间特征。
        real_score, real_features = branch(real_waveform)
        # 同一分支处理生成波形。
        fake_score, fake_features = branch(fake_waveform)
        # 计算该分支的 LSGAN 判别器损失；列表接口与标准多分支函数一致。
        branch_loss = least_squares_discriminator_loss([real_score], [fake_score])
        # 反传前检查数值，NaN/Inf 不能进入参数梯度。
        _assert_finite("discriminator branch loss", branch_loss)
        # 将当前分支梯度累加到判别器参数；优化器会在所有分支结束后只 step 一次。
        branch_loss.backward()
        # detach 后只累加数值用于日志，不把已反传计算图挂在 total 上。
        total.add_(branch_loss.detach())
        # 显式删除当前分支大张量引用，让下一分支前尽快回收显存。
        del real_score, real_features, fake_score, fake_features, branch_loss
    # 再检查 8 个分支总和是否为有限数。
    _assert_finite("summed discriminator loss", total)
    # 返回总 D 损失标量；真正 optimizer.step 由 train_batch 统一调用一次。
    return total


# 逐判别器分支求学生波形梯度 VJP，保持精确目标同时避免保存全部特征图。
def serial_student_discriminator_gradient(
    discriminator: PaperWaveformDiscriminator,  # 此阶段参数冻结，但对输入波形仍可求导。
    real_waveform: torch.Tensor,  # 真实 mel 经冻结声码器生成的参考波形。
    fake_waveform: torch.Tensor,  # 学生 mel 经冻结声码器生成的完整 batch32 波形代理。
    lambda_fm: float,  # 论文特征匹配权重，固定为 2。
# 结束 serial_student_discriminator_gradient 的参数声明；返回类型为 Tuple[torch.Tensor, torch.Tensor, torch.Tensor]，具体内容由下方实现给出。
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 逐分支求完整批量的对抗、特征匹配损失及对生成波形的梯度。
    """Return exact full-batch adversarial/FM losses and their waveform VJP.

    A detached full-batch waveform proxy lets each frozen discriminator branch
    release its graph immediately.  The summed waveform derivative is injected
    through the real vocoder/student graph once by ``train_batch``.  This does
    not split the batch, approximate a loss, or add an optimizer update.
    """

    # 真假波形必须完全同形，保证特征层逐元素比较合法。
    if real_waveform.shape != fake_waveform.shape:
        # 拒绝广播导致的错误损失。
        raise ValueError("Real and fake waveforms must have identical shapes")
    # 学生更新时判别器只能提供梯度方向，自己的参数不得积累梯度。
    if any(parameter.requires_grad for parameter in discriminator.parameters()):
        # 若仍可训练，会浪费显存且破坏一次 D/一次学生的更新边界。
        raise RuntimeError("Discriminator parameters must be frozen for the theta step")
    # 得到 5 个 MPD+3 个 MRD 的固定分支序列。
    branches = tuple(discriminator.loss_branches())
    # 没有分支就无法计算论文要求的波形对抗和特征匹配项。
    if not branches:
        # 明确终止。
        raise RuntimeError("Paper discriminator returned no MPD/MRD branches")

    # 切断旧学生/声码器图，只让这个全 batch 波形代理本身接收分支梯度。
    fake_proxy = fake_waveform.detach().requires_grad_(True)
    # 创建学生对抗损失累计标量，不挂接分支计算图。
    adversarial_total = fake_proxy.new_zeros(())
    # 创建原始特征匹配损失累计标量。
    feature_matching_total = fake_proxy.new_zeros(())
    # 创建与 [B,1,Twav] 同形的总波形梯度，稍后一次注入真实学生图。
    waveform_gradient = torch.zeros_like(fake_proxy)
    # 每次只保留一个判别器分支的中间特征，显著降低 A100 峰值显存。
    for branch in branches:
        # 真实波形只充当比较目标，不需要任何输入梯度。
        with torch.no_grad():
            # `_real_score` 不参与学生损失，但统一接收返回值以便显式释放。
            _real_score, real_features = branch(real_waveform)
        # 生成波形代理要保留输入梯度，判别器参数本身已冻结。
        fake_score, fake_features = branch(fake_proxy)
        # 当前分支学生对抗项希望 fake_score 接近“真实”标签。
        branch_adversarial = least_squares_student_loss([fake_score])
        # 当前分支逐层比较真实/生成特征的平均绝对误差。
        branch_feature_matching = feature_matching_loss(
            # 按同一个判别器分支配对真实和生成特征，保证逐层FM可比较。
            [real_features], [fake_features]
        )
        # 用论文权重 2 合成当前分支的波形相关学生损失。
        branch_loss = branch_adversarial + lambda_fm * branch_feature_matching
        # 在求导前逐项检查三个标量，便于精确定位异常来源。
        for name, value in (
            ("student branch adversarial loss", branch_adversarial),  # 未加权对抗项。
            ("student branch feature matching loss", branch_feature_matching),  # 未加权 FM。
            # 检查当前分支对抗项加两倍特征匹配后是否仍为有限数。
            ("student weighted branch loss", branch_loss),  # adv+2*FM。
        ):
            # 任一分支非有限就停止，不能把坏梯度加进总 VJP。
            _assert_finite(name, value)
        # 只求 branch_loss 对完整波形代理的梯度，不给模型参数反传。
        (branch_gradient,) = torch.autograd.grad(branch_loss, fake_proxy)
        # 确认本分支 [B,1,Twav] 梯度没有 NaN/Inf。
        _assert_finite("student branch waveform gradient", branch_gradient)
        # 保存未加权对抗损失数值，用于精确日志和总损失显示。
        adversarial_total.add_(branch_adversarial.detach())
        # 保存未加权特征匹配数值。
        feature_matching_total.add_(branch_feature_matching.detach())
        # 线性累加 8 个分支的波形梯度，等价于先求和标量再一次反传。
        waveform_gradient.add_(branch_gradient)
        # 立即释放当前分支所有图和大特征张量，控制峰值显存。
        del (
            _real_score,  # 当前真实分数。
            real_features,  # 当前真实中间特征。
            fake_score,  # 当前生成分数。
            fake_features,  # 当前生成中间特征。
            branch_adversarial,  # 当前对抗标量及其图。
            branch_feature_matching,  # 当前 FM 标量及其图。
            branch_loss,  # 当前加权分支标量。
            branch_gradient,  # 已加到总梯度的临时张量。
        )
    # 所有分支完成后，再检查累计波形梯度数值。
    _assert_finite("student summed waveform gradient", waveform_gradient)
    # 返回对抗总值、FM 总值和可一次注入声码器/学生图的精确 VJP。
    return adversarial_total, feature_matching_total, waveform_gradient


# 测试阶段逐分支计算 D/学生对抗/FM 三项数值，全函数受 inference_mode 保护。
def serial_discriminator_metrics(
    discriminator: PaperWaveformDiscriminator,  # 评估模式的 MPD+MRD。
    real_waveform: torch.Tensor,  # 测试真实 mel 对应波形。
    fake_waveform: torch.Tensor,  # 测试学生预测 mel 对应波形。
# 结束 serial_discriminator_metrics 的参数声明；返回类型为 Tuple[torch.Tensor, torch.Tensor, torch.Tensor]，具体内容由下方实现给出。
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 不建梯度图地逐分支测量判别器、学生对抗和特征匹配总量。
    """Compute no-gradient D, student-adversarial, and FM sums branch-wise."""

    # 初始化判别器测试损失总和。
    discriminator_total = real_waveform.new_zeros(())
    # 初始化学生对抗测试损失总和。
    adversarial_total = real_waveform.new_zeros(())
    # 初始化特征匹配测试损失总和。
    feature_matching_total = real_waveform.new_zeros(())
    # 顺序遍历 8 个分支，避免测试时也同时保留所有中间特征。
    for branch in tuple(discriminator.loss_branches()):
        # 当前分支提取真实分数与各层特征。
        real_score, real_features = branch(real_waveform)
        # 当前分支提取生成分数与各层特征。
        fake_score, fake_features = branch(fake_waveform)
        # 累加该分支的 D 最小二乘损失。
        discriminator_total.add_(
            # 计算本分支真实标签1、生成标签0的最小二乘损失。
            least_squares_discriminator_loss([real_score], [fake_score])
        )
        # 累加该分支的学生对抗诊断损失。
        adversarial_total.add_(least_squares_student_loss([fake_score]))
        # 累加该分支真假中间特征的 L1 匹配损失。
        feature_matching_total.add_(
            # 累计本分支真实/生成中间特征的逐层平均绝对误差。
            feature_matching_loss([real_features], [fake_features])
        )
        # 当前分支算完立即释放引用；这里不需要反传。
        del real_score, real_features, fake_score, fake_features
    # 三项总和都必须是有限数，任何异常都会阻止错误曲线写盘。
    for name, value in (
        ("test discriminator loss", discriminator_total),  # 测试 D loss。
        ("test student adversarial loss", adversarial_total),  # 测试学生 adv。
        ("test feature matching loss", feature_matching_total),  # 测试 FM。
    ):
        # 逐项执行 NaN/Inf 检查。
        _assert_finite(name, value)
    # 返回三个未转 Python float 的设备标量，供上层组装完整 ACDD 记录。
    return discriminator_total, adversarial_total, feature_matching_total


# 完成一个严格的全局 batch32：先更新 D 一次，再更新学生一次，并返回全部 loss。
def train_batch(
    batch: Mapping[str, Any],  # DataLoader 产生的一批训练样本及元数据。
    teacher: nn.Module,  # 冻结的 VoiceGrad epoch500 教师 phi。
    student: nn.Module,  # 从教师初始化、当前要训练的 FastVoiceGrad theta。
    vocoder: nn.Module,  # 冻结的自训练修改版 HiFi-GAN Generator V。
    discriminator: PaperWaveformDiscriminator,  # 可训练波形 D：MPD5+MRD3。
    diffusion: VoiceGradDiffusion,  # T=1000 的余弦扩散时间表。
    student_optimizer: torch.optim.Optimizer,  # 只拥有学生参数的 Adam。
    discriminator_optimizer: torch.optim.Optimizer,  # 只拥有 D 参数的 Adam。
    mel_mean: torch.Tensor,  # 训练集 80 维 mel 均值，广播形状与 mel 兼容。
    mel_std: torch.Tensor,  # 训练集 80 维 mel 标准差。
    device: torch.device,  # 正式运行是租用 A100 的 CUDA 设备。
    config: FastVoiceGradStudentConfig,  # 论文设置与已声明复现选择。
# 结束 train_batch 的参数声明；返回类型为 Dict[str, float]，具体内容由下方实现给出。
) -> Dict[str, float]:
    # 搬运张量并强制检查 mel[32,80,128]、BNF[32,144,128]、spk[32,256]。
    tensors = _slice_batch(batch, device)
    # x_zero 表示干净、已标准化的目标 mel，即扩散记号中的 x_0。
    x_zero = tensors["mel"]
    # bnf 提供不随目标说话人变化的语音内容条件。
    bnf = tensors["bnf"]
    # speaker 提供 256 维目标说话人身份条件。
    speaker = tensors["spk_emb"]

    # 为一步学生从 S_K=950 构造输入噪声；形状与 x_zero 同为 [32,80,128]。
    source_noise = torch.randn_like(x_zero)
    # 为教师分数蒸馏另采独立噪声，避免和学生起始噪声错误共用。
    score_noise = torch.randn_like(x_zero)
    # 每条样本独立均匀采一个教师扩散时刻 t∈{1,…,1000}。
    score_timesteps = torch.randint(
        1,  # randint 左端包含 1；不抽 t=0。
        config.diffusion_steps + 1,  # 右端不包含，因此最大恰为 T=1000。
        (config.batch_size,),  # 每个 batch32 样本一个时间步，形状 [32]。
        # 本次推理的计算设备。
        device=device,  # 直接在 A100 生成，避免额外传输。
        # 数组数值存储类型。
        dtype=torch.long,  # 时间步作为网络索引必须为整数类型。
    )

    # 论文式(15)：先用与学生图分离的完整 batch32 真假波形更新判别器一次。
    # 打开 D 参数梯度；上一批学生更新结束时虽已恢复，这里再次显式保证状态。
    set_requires_grad(discriminator, True)
    # 清掉旧 D 梯度，并用 None 节省显存/避免无意累加到下一批。
    discriminator_optimizer.zero_grad(set_to_none=True)
    # D 更新不应给学生/声码器建图，因此整个波形生成放在 no_grad 中。
    with torch.no_grad():
        # 用固定 source_noise 从 S_K=950 一步生成学生 mel，第二返回值此处不需要。
        x_theta_for_d, _ = make_student_prediction(
            student,  # 当前学生网络 theta。
            diffusion,  # 用于按 S_K 加噪和一步反推。
            x_zero,  # 干净训练目标 mel。
            speaker,  # 目标说话人条件。
            bnf,  # 语音内容条件。
            config.student_start_step,  # 论文 K=1 对应起始步 S_K=950。
            source_noise,  # 本批固定的学生输入噪声。
        )
        # 将真实标准化 mel 反归一化，再经冻结 V 合成 D 的真实波形。
        real_waveform_for_d = vocoder(denormalize_mel(x_zero, mel_mean, mel_std))
        # 将学生预测 mel 反归一化，再经同一冻结 V 合成 D 的生成波形。
        fake_waveform_for_d = vocoder(
            # 将学生标准化mel还原到HiFi-GAN训练使用的自然对数mel。
            denormalize_mel(x_theta_for_d, mel_mean, mel_std)
        )
    # 在进入 D 之前确认真实波形全部有限。
    _assert_finite("D real waveform", real_waveform_for_d)
    # 确认学生生成波形没有 NaN/Inf。
    _assert_finite("D fake waveform", fake_waveform_for_d)
    # 逐个 MPD/MRD 分支反传，累加的梯度仍对应完整论文 D 总损失。
    discriminator_loss = serial_discriminator_backward(
        # 判别器接收真实mel合成波形与学生mel合成波形，二者样本一一对应。
        discriminator, real_waveform_for_d, fake_waveform_for_d
    )
    # 所有 8 个分支梯度到齐后只执行一次 Adam 更新。
    discriminator_optimizer.step()
    # 更新后立即清 D 梯度，为学生阶段的“D 无梯度”断言创造明确状态。
    discriminator_optimizer.zero_grad(set_to_none=True)
    # 保存纯 Python D loss 数值供日志；detach 不保留图。
    discriminator_loss_value = float(discriminator_loss.detach())
    # D 用的学生 mel 和损失图不再需要，尽早释放显存。
    del x_theta_for_d, discriminator_loss

    # 论文式(14)：复用无梯度完整波形做代理，让更新后的冻结 D 逐分支贡献精确 VJP。
    # 冻结判别器参数；仍允许对 D 输入波形求导，从而把信号传给学生。
    set_requires_grad(discriminator, False)
    # 清除学生上一批梯度，保证本次更新只由当前 batch32 产生。
    student_optimizer.zero_grad(set_to_none=True)
    # 计算 8 分支 adv、FM 总值以及它们对生成波形的总梯度。
    adversarial_loss, fm_loss, waveform_gradient = (
        # 逐分支求对抗和FM对波形的梯度，控制峰值显存。
        serial_student_discriminator_gradient(
            discriminator,  # 已做完本批 D step 并冻结的判别器。
            real_waveform_for_d,  # 完整 batch32 真实波形代理。
            fake_waveform_for_d,  # 完整 batch32 生成波形代理。
            config.lambda_fm,  # 论文固定特征匹配权重 2。
        )
    )

    # 所有判别器特征图释放后，重新建立一次学生→声码器图，再注入累计波形梯度。
    # 使用完全相同的 x_zero/source_noise/条件重算，因此输出应逐位相同。
    x_theta, _ = make_student_prediction(
        student,  # 参数尚未 step，因此仍是与 D 采样时相同的学生。
        diffusion,  # 相同扩散时间表。
        x_zero,  # 相同干净 mel。
        speaker,  # 相同目标说话人条件。
        bnf,  # 相同内容条件。
        config.student_start_step,  # 相同 S_K=950。
        source_noise,  # 复用本批同一噪声，保证可比。
    )
    # 这次声码器前向保留“波形→mel→学生”的梯度图；V 参数仍冻结。
    fake_waveform = vocoder(denormalize_mel(x_theta, mel_mean, mel_std))
    # 确认重算学生 mel 数值正常。
    _assert_finite("student prediction", x_theta)
    # 确认保留梯度图的生成波形数值正常。
    _assert_finite("student fake waveform", fake_waveform)
    # 重算波形必须与之前代理逐位相等，否则注入 VJP 会对应错误输出点。
    if not torch.equal(fake_waveform.detach(), fake_waveform_for_d):
        # 计算最大绝对差，便于判断是数值微差还是模型含随机层。
        maximum_difference = float(
            # 若两次重算不一致，报告最大逐采样点偏差定位随机性或数值问题。
            torch.max(torch.abs(fake_waveform.detach() - fake_waveform_for_d))
        )
        # 任何非零差异都拒绝继续，严格保证显存优化不改变数学梯度。
        raise RuntimeError(
            # 波形代理与真实反向图输出必须逐位一致，否则预先求出的梯度作用点错误。
            "Recomputed student waveform is not bit-identical to the full-batch "
            # 把不一致的最大差值写入错误，避免继续用错误波形梯度训练。
            f"VJP proxy (max_abs_diff={maximum_difference:.9g})"
        )
    # 等价性确认后释放两份无梯度代理波形。
    del real_waveform_for_d, fake_waveform_for_d
    # 蒸馏梯度只从 L1 中的 x_theta 传回学生；教师重建目标整条分支已停止梯度。
    distillation_loss, _ = score_distillation_loss(
        x_theta,  # 学生一步预测的干净 mel。
        teacher,  # epoch500 冻结 VoiceGrad 教师。
        diffusion,  # T=1000 的同一扩散过程。
        score_timesteps,  # 本批 32 个随机教师时间步。
        score_noise,  # 独立蒸馏噪声 [32,80,128]。
        speaker,  # 目标说话人条件。
        bnf,  # 语音内容条件。
        # 训练和固定测试都显式传入本次选定的教师目标空间。
        target_mode=config.score_target,
        # 显式传入本次alpha权重解释，防止修复配置被默认值覆盖。
        weight_mode=config.score_weight,
    )
    # 仅用于记录/检查地组装论文总标量 L_adv+2L_FM+45L_dist。
    total_loss = acdd_student_loss(
        adversarial_loss,  # 8 个分支未加权学生对抗总和。
        fm_loss,  # 8 个分支未加权特征匹配总和。
        distillation_loss,  # 教师分数蒸馏原始损失。
        lambda_fm=config.lambda_fm,  # 论文权重 2。
        lambda_dist=config.lambda_dist,  # 论文权重 45。
    )
    # 在反传前检查学生目标所有组成，避免坏数进入 Adam 状态。
    for name, value in (
        # 检查学生波形对抗损失是否有限。
        ("student adversarial loss", adversarial_loss),  # L_adv。
        # 检查未乘 2 的特征匹配损失是否有限。
        ("raw feature matching loss", fm_loss),  # L_FM。
        # 检查未乘 45 的蒸馏损失是否有限。
        ("raw score distillation loss", distillation_loss),  # L_dist。
        # 检查三个已加权分项组成的学生总损失是否有限。
        ("student total loss", total_loss),  # L_adv+2L_FM+45L_dist。
    ):
        # 任何 NaN/Inf 都立即报出具体项。
        _assert_finite(name, value)
    # 一次联合反传：波形分支注入预先精确累加的 VJP，蒸馏分支乘 45。
    torch.autograd.backward(
        (fake_waveform, distillation_loss),  # 两个需要向学生回传的图输出。
        (
            waveform_gradient,  # 已包含 adv+2FM 的 dL/dwaveform。
            distillation_loss.new_tensor(config.lambda_dist),  # 蒸馏标量上游梯度固定 45。
        ),
    )

    # 收集所有实际收到梯度的学生参数，用于更新前完整性检查。
    student_gradients = [
        parameter.grad  # 取参数的梯度张量。
        for parameter in student.parameters()  # 遍历学生所有参数。
        if parameter.grad is not None  # 忽略结构上未参与本次计算的参数。
    ]
    # 一个梯度都没有说明计算图被错误切断。
    if not student_gradients:
        # 禁止空更新悄悄计为一个 global_step。
        raise RuntimeError("Student received no gradient")
    # 任何学生梯度出现 NaN/Inf 都不能交给 Adam。
    if any(not bool(torch.isfinite(gradient).all()) for gradient in student_gradients):
        # 立即停止并保留上一轮完整检查点。
        raise FloatingPointError("Student gradient contains NaN or Inf")
    # 如果所有梯度元素全为 0，学生不会学习，通常代表目标或图连接错误。
    if not any(bool(torch.count_nonzero(gradient)) for gradient in student_gradients):
        # 拒绝把无效 step 记录进训练进度。
        raise RuntimeError("All student gradients are zero")
    # 教师严格冻结，因此所有 teacher.parameter.grad 都应保持 None。
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        # 教师收到梯度表示冻结边界被破坏，会额外占显存并违反论文训练方式。
        raise RuntimeError("Frozen teacher unexpectedly received gradients")
    # 声码器虽然允许梯度穿过输入，但自身参数 grad 必须仍为 None。
    if any(parameter.grad is not None for parameter in vocoder.parameters()):
        # 若 V 参数收到梯度，说明误把已训练声码器一起训练了。
        raise RuntimeError("Frozen vocoder unexpectedly received parameter gradients")
    # 学生阶段 D 参数也应无 grad；输入梯度不等于更新 D 参数。
    if any(parameter.grad is not None for parameter in discriminator.parameters()):
        # 有 D 梯度说明冻结切换失败，会混淆两个优化目标。
        raise RuntimeError("Frozen discriminator unexpectedly received gradients in theta step")

    # 所有检查通过后，学生 Adam 对本完整 batch32 只更新一次。
    student_optimizer.step()
    # 清学生梯度，减少显存并防止下批意外累加。
    student_optimizer.zero_grad(set_to_none=True)
    # 恢复 D 可训练状态，为下一个 batch 的判别器更新做准备。
    set_requires_grad(discriminator, True)
    # 返回纯数值损失字典；这些值会先按 batch 再按 epoch 求平均并写 CSV。
    return {
        "student_total": float(total_loss.detach()),  # 完整 ACDD 学生总损失。
        "student_adv": float(adversarial_loss.detach()),  # 未加权对抗项。
        "fm_raw": float(fm_loss.detach()),  # 未加权 FM 项。
        # 记录两倍特征匹配项对总损失的实际贡献，先断开梯度再转标量。
        "fm_weighted": float(config.lambda_fm * fm_loss.detach()),  # 2*FM。
        "dist_raw": float(distillation_loss.detach()),  # 未加权蒸馏项。
        "dist_weighted": float(config.lambda_dist * distillation_loss.detach()),  # 45*蒸馏。
        "discriminator": discriminator_loss_value,  # 本批 D 更新前的总损失。
    }


# 遍历一整轮 1230 个完整 batch，累计训练 loss 平均值并更新全局步数。
def train_epoch(
    loader,  # 训练 DataLoader：shuffle=True、drop_last=True、batch=32。
    epoch: int,  # 当前轮次 1～100，仅用于进度显示。
    global_step: int,  # 进入本轮前已完成的更新次数。
    teacher: nn.Module,  # 冻结教师。
    student: nn.Module,  # 可训练学生。
    vocoder: nn.Module,  # 冻结声码器。
    discriminator: PaperWaveformDiscriminator,  # 可训练 MPD+MRD。
    diffusion: VoiceGradDiffusion,  # 扩散时间表。
    student_optimizer: torch.optim.Optimizer,  # 学生 Adam。
    discriminator_optimizer: torch.optim.Optimizer,  # 判别器 Adam。
    mel_mean: torch.Tensor,  # mel 反归一化均值。
    mel_std: torch.Tensor,  # mel 反归一化标准差。
    device: torch.device,  # 租用 A100 CUDA 设备。
    config: FastVoiceGradStudentConfig,  # 固定论文配置。
    # 可选监控器在每次完整训练更新后接收损失；无监控时数学更新相同。
    monitor: Optional[LossMonitor] = None,
# 结束 train_epoch 的参数声明；返回类型为 Tuple[Dict[str, float], int]，具体内容由下方实现给出。
) -> Tuple[Dict[str, float], int]:
    # 教师保持 eval，禁用可能存在的训练态行为；参数本身也已冻结。
    teacher.eval()
    # 声码器同样固定在 eval，保证重算波形逐位一致。
    vocoder.eval()
    # 学生进入 train 模式，明确这是会更新参数的阶段。
    student.train()
    # 判别器进入 train 模式，逐 batch 接受一次更新。
    discriminator.train()
    # 以 0.0 为默认值累计 7 个训练损失的 batch 均值。
    totals: MutableMapping[str, float] = defaultdict(float)
    # 记录本轮实际完成 batch 数，用于平均和空加载器保护。
    batches = 0
    # 创建实时进度条，显示当前 epoch 编号。
    progress = tqdm(loader, desc=f"FastVoiceGrad epoch {epoch:03d}")
    # 逐批读取训练数据；严格配置下每轮应进入 1230 次。
    for batch in progress:
        # 完成一次 D step 和一次学生 step，并取得该批全部损失。
        losses = train_batch(
            batch,  # 当前完整 batch32。
            teacher,  # 冻结教师。
            student,  # 待更新学生。
            vocoder,  # 冻结声码器。
            discriminator,  # 先更新后冻结提供学生梯度。
            diffusion,  # T=1000 扩散对象。
            student_optimizer,  # 本批执行一次 step。
            discriminator_optimizer,  # 本批执行一次 step。
            mel_mean,  # 80 维均值。
            mel_std,  # 80 维标准差。
            # 在当前租用服务器选定的 CUDA 设备执行。
            device,  # CUDA。
            config,  # 论文/复现参数。
        )
        # 每个完整 batch 完成后全局更新计数加 1；100 轮最终为 123000。
        global_step += 1
        # 训练启动时配置了监控器才写窗口和报警。
        if monitor is not None:
            # 立即提交当前step、epoch和分项损失，让持续上涨能尽早被发现。
            monitor.observe(global_step, epoch, losses)
        # 本轮已完成批次数加 1。
        batches += 1
        # 将当前批的 7 个 loss 数值分别累加。
        for key, value in losses.items():
            # key 保持固定 schema，value 已在 train_batch 转为 float。
            totals[key] += value
        # 每隔配置的步数（默认 10）刷新一次进度条右侧数值。
        if global_step % config.log_every_steps == 0:
            # 显示本批而非累计平均，便于实时发现爆炸或异常波动。
            progress.set_postfix(
                total=f"{losses['student_total']:.4f}",  # ACDD 总损失。
                adv=f"{losses['student_adv']:.4f}",  # 学生对抗项。
                fm=f"{losses['fm_raw']:.4f}",  # 原始 FM 项。
                dist=f"{losses['dist_raw']:.4f}",  # 原始蒸馏项。
                d=f"{losses['discriminator']:.4f}",  # D 损失。
            )
    # 理论上不应为 0；若数据过少/drop_last 设置错误则明确失败。
    if batches == 0:
        # 防止除以零或写出虚假的 0 loss。
        raise RuntimeError("Training DataLoader produced no full batch")
    # 每项除以实际批数得到 epoch 训练平均，同时返回更新后的 global_step。
    return {key: value / batches for key, value in totals.items()}, global_step


# 把测试 batch 均值按真实样本数加权，避免最后 4 条与完整 32 条权重相同。
def add_sample_weighted_losses(
    totals: MutableMapping[str, float],  # 当前累计的“损失×样本数”总和。
    losses: Mapping[str, float],  # 当前测试 batch 的 7 项均值。
    batch_size: int,  # 当前批真实样本数，可能是 32 或最后的 4。
# 结束 add_sample_weighted_losses 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # 按实际样本数累积各批均值，避免末尾 4 条测试样本被赋予过大权重。
    """Accumulate batch means without overweighting a short final batch."""

    # batch 不得为空，且字段必须与固定 7 项 schema 完全一致。
    if batch_size < 1 or set(losses) != set(LOSS_COMPONENTS):
        # 拒绝缺项、多项或无样本的贡献。
        raise RuntimeError("Invalid sample-weighted loss contribution")
    # 按固定顺序逐项检查并累计。
    for key in LOSS_COMPONENTS:
        # 统一转换成 Python float，避免将计算图或设备张量带入长期累计。
        value = float(losses[key])
        # 非有限测试值不能写入 CSV/图表。
        if not math.isfinite(value):
            # 报告具体损失名。
            raise RuntimeError(f"Non-finite {key} contribution")
        # batch 均值×B 还原该批样本总贡献，最后再除总样本数 100。
        totals[key] += value * batch_size


# inference_mode 彻底关闭自动微分，测试 loss 只观察、不更新模型并节省显存。
@torch.inference_mode()
# 在固定 100 条 text-unseen 样本上计算与训练同定义的完整 loss 诊断。
def evaluate_test_epoch(
    loader,  # 不打乱、不丢尾批的固定测试 DataLoader。
    teacher: nn.Module,  # 冻结 epoch500 教师。
    student: nn.Module,  # 当前 epoch 训练后的学生。
    vocoder: nn.Module,  # 冻结修改版 HiFi-GAN。
    discriminator: PaperWaveformDiscriminator,  # 当前 epoch 训练后的 D。
    diffusion: VoiceGradDiffusion,  # 与训练相同的扩散时间表。
    mel_mean: torch.Tensor,  # mel 反归一化均值。
    mel_std: torch.Tensor,  # mel 反归一化标准差。
    device: torch.device,  # CUDA 评估设备。
    config: FastVoiceGradStudentConfig,  # 测试种子、帧数、batch 等固定设置。
# 结束 evaluate_test_epoch 的参数声明；返回类型为 Tuple[Dict[str, float], int, int]，具体内容由下方实现给出。
) -> Tuple[Dict[str, float], int, int]:
    # 在固定 100 条未知文本测试语音上测量全部 ACDD 分项，只作诊断。
    """Measure all ACDD terms on the fixed 100-item unseen diagnostic set.

    This function deliberately owns independent generators and performs no
    optimizer operation.  Consequently evaluating every epoch neither consumes
    the training RNG stream nor changes the fixed 100-epoch trajectory.
    """

    # 先记住四个模型进入评估前各自的 train/eval 状态，结束后原样恢复。
    modes = {
        "teacher": teacher.training,  # 教师原模式，通常 False。
        # 学生网络参数字典。
        "student": student.training,  # 学生原模式，训练后通常 True。
        "vocoder": vocoder.training,  # 声码器原模式，通常 False。
        "discriminator": discriminator.training,  # D 原模式，训练后通常 True。
    }
    # 捕获正式训练的全部随机状态，保证测试不改变下一轮噪声与样本顺序。
    rng_state = capture_rng_state()
    # 所有模型进入确定性评估模式。
    teacher.eval()
    # 学生切到 eval，但不改变参数。
    student.eval()
    # 声码器 eval 保证合成行为固定。
    vocoder.eval()
    # 判别器 eval 保证诊断定义每轮一致。
    discriminator.eval()

    # 测试学生起始噪声使用独立 Generator，不消费训练 RNG。
    source_generator = torch.Generator(device=device)
    # 每个 epoch 都从相同 source seed 开始，比较的是模型变化而非噪声变化。
    source_generator.manual_seed(config.test_loss_source_noise_seed)
    # 蒸馏噪声使用第二个独立 Generator。
    score_generator = torch.Generator(device=device)
    # 固定教师目标使用的再加噪随机序列；学生输出随 epoch 改变，实际教师监督目标仍会改变。
    score_generator.manual_seed(config.test_loss_score_noise_seed)
    # 蒸馏时间步使用第三个独立 Generator。
    timestep_generator = torch.Generator(device=device)
    # 固定时间步 seed，使每条测试样本的 t 跨 epoch 一致。
    timestep_generator.manual_seed(config.test_loss_timestep_seed)

    # 累计 7 项“均值×样本数”。
    totals: MutableMapping[str, float] = defaultdict(float)
    # 累计实际覆盖样本数，结束必须恰为 100。
    examples = 0
    # 累计 batch 数，结束必须恰为 4。
    batches = 0
    # 保存已见 (speaker,utt_id)，同时检查重复和完整唯一覆盖。
    seen_ids = set()
    # try/finally 确保测试中即使报错也恢复模型模式和训练 RNG。
    try:
        # 创建不残留额外进度行的测试进度条。
        progress = tqdm(loader, desc="FastVoiceGrad unseen-test loss", leave=False)
        # 按固定 manifest 顺序遍历 32/32/32/4 四批。
        for batch in progress:
            # 将特征搬到 GPU 并验证 [B,80/144,128]、[B,256]。
            tensors = _slice_test_batch(batch, device, config)
            # 测试干净 mel [B,80,128]。
            x_zero = tensors["mel"]
            # 测试 BNF [B,144,128]。
            bnf = tensors["bnf"]
            # 测试说话人嵌入 [B,256]。
            speaker = tensors["spk_emb"]
            # 读取当前 B；前三批 32，最后一批 4。
            batch_size = int(x_zero.shape[0])

            # 说话人名仍保留在 CPU 字典中，用于检查测试身份覆盖。
            batch_speakers = batch.get("spk_name")
            # 句子 ID 同样用于构成唯一身份。
            batch_utterances = batch.get("utt_id")
            # 两项都必须存在，且列表长度与张量 B 一致。
            if (
                batch_speakers is None  # 缺说话人名。
                or batch_utterances is None  # 缺句子 ID。
                or len(batch_speakers) != batch_size  # 说话人名数量与 B 不符。
                or len(batch_utterances) != batch_size  # 句子 ID 数量与 B 不符。
            ):
                # 元数据错位会让覆盖验证失真，因此停止。
                raise RuntimeError("Test batch lacks aligned speaker/utterance identities")
            # 将当前批说话人与句子 ID 一一配对。
            for identity in zip(batch_speakers, batch_utterances):
                # 统一转字符串，避免 NumPy/Python 字符串类型造成集合比较差异。
                normalized = (str(identity[0]), str(identity[1]))
                # 同一身份再次出现说明 DataLoader/manifest 有重复。
                if normalized in seen_ids:
                    # 明确列出重复项。
                    raise RuntimeError(f"Duplicate held-out test identity: {normalized}")
                # 记录本批这个唯一测试身份。
                seen_ids.add(normalized)

            # 用独立固定生成器产生学生起始噪声，形状与 mel 完全相同。
            source_noise = torch.randn(
                # 按测试梅尔的批数、80 通道和固定 128 帧生成同形状噪声。
                x_zero.shape,  # [B,80,128]。
                # 数组数值存储类型。
                dtype=x_zero.dtype,  # 与 mel 同为训练使用的浮点精度。
                # 本次推理的计算设备。
                device=device,  # 直接在 A100 上生成。
                # 这里是 torch.Generator，单独控制学生起始噪声的随机序列。
                generator=source_generator,  # 不使用全局训练 RNG。
            )
            # 独立生成教师分数蒸馏噪声。
            score_noise = torch.randn(
                # 按测试梅尔的批数、80 通道和固定 128 帧生成同形状噪声。
                x_zero.shape,  # [B,80,128]。
                # 数组数值存储类型。
                dtype=x_zero.dtype,  # 与输入同精度。
                # 噪声直接创建在当前训练 GPU，避免设备不匹配。
                device=device,  # CUDA。
                # 这里是 torch.Generator，单独控制教师目标再加噪的随机序列。
                generator=score_generator,  # 固定独立序列。
            )
            # 为当前测试批固定抽取每样本一个 t∈[1,1000]。
            score_timesteps = torch.randint(
                1,  # 包含最小扩散步 1。
                config.diffusion_steps + 1,  # 排除 1001，因此最大为 1000。
                (batch_size,),  # 当前每个测试样本一个 t。
                # 本次推理的计算设备。
                device=device,  # 在 A100 上生成。
                # 数组数值存储类型。
                dtype=torch.long,  # 整数索引。
                # 这里是 torch.Generator，单独控制蒸馏时间步抽样。
                generator=timestep_generator,  # 固定独立时间步序列。
            )
            # 从 S_K=950 一步预测测试 mel；全函数 inference_mode，不建梯度图。
            x_theta, _ = make_student_prediction(
                student,  # 当前 epoch 学生。
                diffusion,  # 相同 T=1000 时间表。
                x_zero,  # 固定测试目标。
                speaker,  # 目标说话人。
                bnf,  # 内容条件。
                # 把学生噪声起点固定在配置指定的第 950 级。
                config.student_start_step,  # S_K=950。
                source_noise,  # 固定测试起始噪声。
            )
            # 真实测试 mel 反归一化后经冻结声码器得到参考波形。
            real_waveform = vocoder(denormalize_mel(x_zero, mel_mean, mel_std))
            # 学生测试 mel 经同一声码器得到生成波形。
            fake_waveform = vocoder(denormalize_mel(x_theta, mel_mean, mel_std))
            # 检查学生输出数值。
            _assert_finite("test student prediction", x_theta)
            # 检查参考波形数值。
            _assert_finite("test real waveform", real_waveform)
            # 检查生成波形数值。
            _assert_finite("test fake waveform", fake_waveform)

            # 一次取得测试 D loss、学生对抗 loss 和 FM loss。
            (
                discriminator_loss,  # D 区分真假波形的诊断值。
                adversarial_loss,  # 学生骗过 D 的诊断值。
                fm_loss,  # 真假中间特征差异。
            # 固定测试通过逐分支前向得到D、对抗和FM三个诊断值。
            ) = serial_discriminator_metrics(
                # 同一测试批次的参考波形与生成波形送入冻结评估判别器。
                discriminator, real_waveform, fake_waveform
            )
            # 在固定噪声/时间步上计算教师分数蒸馏测试 loss。
            distillation_loss, _ = score_distillation_loss(
                x_theta,  # 当前学生预测。
                teacher,  # 冻结教师。
                diffusion,  # 相同扩散定义。
                score_timesteps,  # 固定测试 t。
                score_noise,  # 固定测试噪声。
                speaker,  # 目标说话人。
                bnf,  # 内容特征。
                # 训练和固定测试都显式传入本次选定的教师目标空间。
                target_mode=config.score_target,
                # 显式传入本次alpha权重解释，防止修复配置被默认值覆盖。
                weight_mode=config.score_weight,
            )
            # 按论文同一公式合成学生测试总损失。
            total_loss = acdd_student_loss(
                # 传入未额外加权的学生对抗损失。
                adversarial_loss,  # L_adv。
                # 传入原始特征匹配损失，由总目标函数统一乘 2。
                fm_loss,  # L_FM。
                # 传入原始蒸馏损失，由总目标函数统一乘 45。
                distillation_loss,  # L_dist。
                lambda_fm=config.lambda_fm,  # 权重 2。
                lambda_dist=config.lambda_dist,  # 权重 45。
            )
            # 保留 7 个张量形式的测试指标，名称与训练完全对应。
            tensor_losses = {
                # 记录学生三项加权总目标，不含判别器自身训练损失。
                "student_total": total_loss,  # L_adv+2L_FM+45L_dist。
                "student_adv": adversarial_loss,  # 原始对抗项。
                "fm_raw": fm_loss,  # 原始 FM。
                # 记录已乘两倍系数的特征匹配贡献。
                "fm_weighted": config.lambda_fm * fm_loss,  # 2*FM。
                "dist_raw": distillation_loss,  # 原始蒸馏项。
                "dist_weighted": config.lambda_dist * distillation_loss,  # 45*蒸馏。
                "discriminator": discriminator_loss,  # D 诊断值。
            }
            # 新建纯 Python 数值字典，避免长期保存 GPU 张量。
            scalar_losses = {}
            # 逐项验证并转换。
            for key, value in tensor_losses.items():
                # 写 CSV 前禁止 NaN/Inf。
                _assert_finite(f"test {key}", value)
                # inference 标量转 float，释放后不依赖 GPU。
                scalar_losses[key] = float(value)
            # 每段形状相同；batch 均值乘 B 后累计，确保最后 4 条不会被当成 32 条权重。
            # 把当前批 7 项按真实 B 加入总体。
            add_sample_weighted_losses(totals, scalar_losses, batch_size)
            # 实际覆盖样本数增加当前 B。
            examples += batch_size
            # 测试批计数加 1。
            batches += 1
    # 无论正常结束或异常，都恢复进入测试前的状态。
    finally:
        # 按保存的布尔值恢复教师 train/eval 模式。
        teacher.train(modes["teacher"])
        # 恢复学生模式，通常回到 train。
        student.train(modes["student"])
        # 恢复声码器模式。
        vocoder.train(modes["vocoder"])
        # 恢复判别器模式，通常回到 train。
        discriminator.train(modes["discriminator"])
        # 精确恢复训练 RNG，测试过程对下一轮训练轨迹零影响。
        restore_rng_state(rng_state, device)

    # 严格测试期望样本数=10×10=100。
    expected_examples = config.test_speakers * config.test_sentences_per_speaker
    # batch32 且不丢尾批时，应有 ceil(100/32)=4 批。
    expected_batches = math.ceil(expected_examples / config.test_loss_batch_size)
    # 同时核对累计样本数和唯一身份数，防止重复一条又漏掉另一条。
    if examples != expected_examples or len(seen_ids) != expected_examples:
        # 报告总数、唯一数与期望数。
        raise RuntimeError(
            # 测试覆盖数量或唯一语音数不符时停止写曲线。
            "Held-out test coverage mismatch: "
            # 打印实际样本数、去重数与预期100条，定位重复或遗漏。
            f"examples={examples}, unique={len(seen_ids)}, expected={expected_examples}"
        )
    # 批次数也必须精确为 4，保证 DataLoader 配置未被改变。
    if batches != expected_batches:
        # 显示实际和期望批数。
        raise RuntimeError(
            # 固定100条测试应组成32、32、32、4四批，数量不符则报告。
            f"Held-out test batch count is {batches}, expected {expected_batches}"
        )
    # 返回按 100 个样本加权的 7 项平均、样本数和批数，供 CSV 覆盖证明。
    return (
        {key: totals[key] / examples for key in LOSS_COMPONENTS},  # 每项测试均值。
        examples,  # 固定 100。
        batches,  # 固定 4。
    )


# 将一轮训练均值和对应测试均值合并成 loss.csv 的一行数据。
def make_epoch_loss_record(
    train_losses: Mapping[str, float],  # train_epoch 返回的 7 项训练均值。
    test_losses: Mapping[str, float],  # evaluate_test_epoch 返回的 7 项测试均值。
    test_examples: int,  # 本轮实际测试样本数，应为 100。
    test_batches: int,  # 本轮实际测试批数，应为 4。
# 结束 make_epoch_loss_record 的参数声明；返回类型为 Dict[str, float]，具体内容由下方实现给出。
) -> Dict[str, float]:
    # 训练损失键必须与固定 schema 完全一致，防止漏记某项。
    if set(train_losses) != set(LOSS_COMPONENTS):
        # 字段不符立即终止，不写不完整 CSV。
        raise RuntimeError("Training loss components do not match the fixed schema")
    # 测试损失也必须包含完全相同的 7 项。
    if set(test_losses) != set(LOSS_COMPONENTS):
        # 防止训练/测试曲线错列或无法对齐。
        raise RuntimeError("Test loss components do not match the fixed schema")
    # 新建本轮记录，稍后填入 train_*、test_* 和覆盖计数。
    record: Dict[str, float] = {}
    # 训练键统一加 train_ 前缀并转 float。
    record.update({f"train_{key}": float(train_losses[key]) for key in LOSS_COMPONENTS})
    # 测试键统一加 test_ 前缀并转 float。
    record.update({f"test_{key}": float(test_losses[key]) for key in LOSS_COMPONENTS})
    # 记录测试样本覆盖数，转换为普通整数便于 JSON/CSV 保存。
    record["test_examples"] = int(test_examples)
    # 记录测试批覆盖数。
    record["test_batches"] = int(test_batches)
    # 对本行全部数值做最后一次有限数检查；整数计数也可安全转 float。
    if not all(math.isfinite(float(value)) for value in record.values()):
        # 禁止把 NaN/Inf 写进历史日志和检查点。
        raise RuntimeError("Epoch loss record contains NaN or Inf")
    # 返回正好对应 LOSS_FIELDS 去掉 epoch/step 后的字段集合。
    return record


# 将一轮 loss 原子追加到 CSV；实际实现重写最多 100 行以避免半行损坏。
def append_loss_csv(path: Path, epoch: int, global_step: int, losses: Mapping[str, float]) -> None:
    # 输入字段必须等于固定表头从第 3 列开始的全部列。
    if set(losses) != set(LOSS_FIELDS[2:]):
        # 字段缺失/额外都拒绝。
        raise RuntimeError("Epoch loss record does not match the fixed CSV schema")
    # 默认当前 CSV 尚不存在，没有历史行。
    rows = []
    # 从第 2 轮起读取已有完整历史，以便验证表头和轮次连续性。
    if path.is_file():
        # newline="" 遵循 csv 模块要求，避免 Windows/Unix 空行差异。
        with path.open("r", encoding="utf-8", newline="") as handle:
            # DictReader 按列名读取每一轮记录。
            reader = csv.DictReader(handle)
            # 表头顺序与内容必须和当前代码固定 schema 完全一致。
            if tuple(reader.fieldnames or ()) != LOSS_FIELDS:
                # 防止拿旧版本 CSV 继续写造成列错位。
                raise RuntimeError("Student loss.csv header differs from the fixed schema")
            # 将最多 100 行读入内存，便于随后原子重写。
            rows = list(reader)
    # 新增 epoch 必须恰好等于“已有行数+1”，不允许跳轮或重复写轮次。
    if int(epoch) != len(rows) + 1:
        # 报告本轮和历史行数，便于恢复逻辑诊断。
        raise RuntimeError(
            # CSV轮次游标不一致时拒绝追加，防止曲线重复或跳轮。
            f"Student loss.csv append cursor mismatch: epoch={epoch}, rows={len(rows)}"
        )
    # 将 epoch、步数和全部损失格式化为字符串后追加为新行。
    rows.append(
        {
            # 当前完整训练轮次。
            "epoch": str(int(epoch)),  # 轮次保存为无小数整数文本。
            # 检查点记录的累计优化器更新步数。
            "global_step": str(int(global_step)),  # 累计更新次数保存为整数文本。
            # 展开剩余所有固定列；计数用整数，loss 固定保留 10 位小数。
            **{
                # key 依次遍历 train_*、test_*、test_examples、test_batches。
                key: (
                    str(int(losses[key]))  # 覆盖计数字段不能写成浮点。
                    if key in LOSS_COUNT_FIELDS  # 判断是否为 examples/batches。
                    else f"{float(losses[key]):.10f}"  # 损失保留足够精度供续训核对。
                )
                for key in LOSS_FIELDS[2:]  # 跳过已单独填写的 epoch/global_step。
            },
        }
    )
    # 通过同目录临时文件重写最多 100 个短行，强制停止也不会留下半条 CSV。
    # 调用原子写帮助函数完成 flush、fsync 和 replace。
    _atomic_write_loss_rows(path, rows)


# 将完整 loss 行列表写入同目录临时文件，再原子替换正式 CSV。
def _atomic_write_loss_rows(path: Path, rows) -> None:
    # 临时路径例如 loss.csv.tmp，与目标在同一文件系统。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # UTF-8 文本写入，newline="" 交由 csv 模块正确控制换行。
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        # 固定 LOSS_FIELDS 作为严格表头及每行列顺序。
        writer = csv.DictWriter(handle, fieldnames=LOSS_FIELDS)
        # 先写一次表头。
        writer.writeheader()
        # 再写全部历史行和本轮新行。
        writer.writerows(rows)
        # 将 Python 用户态缓冲推给操作系统。
        handle.flush()
        # 强制同步到磁盘，减少实例异常关机导致的 loss 丢失。
        os.fsync(handle.fileno())
    # 原子替换正式 loss.csv，读者只会看到完整旧版或完整新版。
    os.replace(temporary, path)


# 续训前核对 CSV 与 latest.pt，并只修复“CSV 已写但 checkpoint 未写完”的单行尾巴。
def reconcile_loss_csv(
    path: Path,  # 当前输出目录的 loss.csv。
    completed_epoch: int,  # latest.pt 声明已完成的 epoch。
    batches_per_epoch: int,  # 严格训练每轮完整 batch 数，应为 1230。
    checkpoint_losses: Mapping[str, float],  # latest.pt 保存的最后一轮完整 loss。
# 结束 reconcile_loss_csv 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # 核对逐轮日志，仅允许修复先写日志但尚未保存检查点的一条合法尾记录。
    """Validate the epoch log and repair only one known append-before-save tail."""

    # 正式续训必须同时有 loss.csv；缺失时无法证明历史曲线完整。
    if not path.is_file():
        # 拒绝凭空续写。
        raise FileNotFoundError(f"Resume loss log is missing: {path}")
    # 以 CSV 标准换行方式读取。
    with path.open("r", encoding="utf-8", newline="") as handle:
        # 将列名映射到每行字典。
        reader = csv.DictReader(handle)
        # 旧/损坏表头不能和当前格式混用。
        if tuple(reader.fieldnames or ()) != LOSS_FIELDS:
            # 立即停止。
            raise RuntimeError("Student loss.csv header differs from the fixed schema")
        # 历史最多 100 行，一次读入便于完整验证。
        rows = list(reader)
    # 正常应等于 checkpoint epoch；崩溃可能留下唯一一个先写 CSV 的尾行。
    if len(rows) not in {completed_epoch, completed_epoch + 1}:
        # 少行、多于一行尾巴都不是已知原子保存顺序能解释的状态。
        raise RuntimeError(
            # 日志记录数必须与最新完整检查点对应。
            "loss.csv row count is inconsistent with latest.pt: "
            # 列出日志行数和模型已完成轮数，定位断点与日志不一致。
            f"rows={len(rows)}, checkpoint_epoch={completed_epoch}"
        )
    # 从 1 开始逐行验证轮次、步数、有限数和测试覆盖。
    for expected_epoch, row in enumerate(rows, start=1):
        # 格式转换可能失败，用 try 捕获并给出具体行号。
        try:
            # epoch 必须是整数文本。
            row_epoch = int(row["epoch"])
            # global_step 必须是整数文本。
            row_step = int(row["global_step"])
            # 读取 14 个 train/test loss 为浮点数。
            loss_values = {
                key: float(row[key])  # 将十进制文本恢复为数值。
                for key in (*TRAIN_LOSS_FIELDS, *TEST_LOSS_FIELDS)  # 全部训练/测试列。
            }
            # 测试样本数和批数必须是整数。
            count_values = {key: int(row[key]) for key in LOSS_COUNT_FIELDS}
        # 任一字段缺失、为空或格式错误都会进入这里。
        except (KeyError, TypeError, ValueError) as error:
            # 保留底层异常原因并指出 CSV 行号。
            raise RuntimeError(f"Invalid loss.csv row {expected_epoch}: {error}") from error
        # 每行轮次必须连续，步数必须严格等于 epoch×1230。
        if row_epoch != expected_epoch or row_step != expected_epoch * batches_per_epoch:
            # 跳轮、重复轮或中途 batch 检查点都不作为精确续训入口。
            raise RuntimeError(
                # 逐行校验时指出第一条轮次或步数错误的位置。
                f"loss.csv cursor mismatch at row {expected_epoch}: "
                # 显示该行实际epoch和step以便人工核对。
                f"epoch={row_epoch}, global_step={row_step}"
            )
        # 14 项 loss 都必须是有限数。
        if not all(math.isfinite(value) for value in loss_values.values()):
            # 指出问题所在 epoch。
            raise RuntimeError(f"Non-finite loss.csv value at epoch {expected_epoch}")
        # 每轮测试期望覆盖固定的 10×10=100 条。
        expected_examples = (
            # 固定测试人数来自论文配置。
            PAPER_STUDENT_CONFIG.test_speakers
            # 乘每人10句得到全部100条测试语音。
            * PAPER_STUDENT_CONFIG.test_sentences_per_speaker
        )
        # 100 条按 batch32 且保留尾批，应为 4 批。
        expected_batches = math.ceil(
            # 用100条除以测试batch32，向上取整后应为4批。
            expected_examples / PAPER_STUDENT_CONFIG.test_loss_batch_size
        )
        # 对比 CSV 实记覆盖与期望覆盖。
        if count_values != {
            # 恢复日志应精确覆盖全部 100 条测试语音。
            "test_examples": expected_examples,  # 100。
            # 100 条按 32 条分批后应完成四批，末批为四条。
            "test_batches": expected_batches,  # 4。
        }:
            # 覆盖不完整的测试点不能留在正式曲线里。
            raise RuntimeError(
                # 逐轮日志必须保留正确测试覆盖数，异常时报告该轮原值。
                f"Invalid held-out coverage at epoch {expected_epoch}: {count_values}"
            )

    # 若 CSV 比 latest.pt 多恰好一行，说明上一进程在两次原子写之间中断。
    if len(rows) == completed_epoch + 1:
        # loss.csv 先 fsync、latest.pt 后保存；尾行对应的模型状态未必存在，必须删掉。
        # 原子重写，只保留与 latest.pt 已完成轮数一致的前缀。
        _atomic_write_loss_rows(path, rows[:completed_epoch])
        # 内存中的 rows 同步截断，后续末行核对使用修复后历史。
        rows = rows[:completed_epoch]
    # 可续训 latest.pt 至少来自完成第 1 轮，因此 CSV 不应为空。
    if not rows:
        # 空历史无法与 checkpoint loss 互证。
        raise RuntimeError("A resumable checkpoint must have at least one loss row")
    # 取 CSV 最后一条已完成轮记录。
    final = rows[-1]
    # 对照 latest.pt 中保存的同轮全部 loss/覆盖字段。
    for key in LOSS_FIELDS[2:]:
        # 覆盖计数按整数精确比较。
        if key in LOSS_COUNT_FIELDS:
            # 计数不同表示 CSV 与 checkpoint 不属于同一状态。
            if int(final[key]) != int(checkpoint_losses[key]):
                # 指出冲突字段。
                raise RuntimeError(f"loss.csv and latest.pt disagree for {key}")
            # 当前计数已通过，进入下一字段。
            continue
        # loss.csv 只保存 10 位小数，因此用 1e-9 绝对/相对容差比较 checkpoint 浮点。
        if not math.isclose(
            float(final[key]),  # CSV 末行值。
            float(checkpoint_losses[key]),  # latest.pt 原始浮点值。
            rel_tol=1e-9,  # 随数值大小缩放的相对容差。
            abs_tol=1e-9,  # 接近 0 时仍允许十位小数格式误差。
        ):
            # 超出格式化可解释误差说明两个文件不匹配。
            raise RuntimeError(f"loss.csv and latest.pt disagree for {key}")


# 保存学生检查点；latest.pt 含完整续训状态，epoch_XXXX.pt 只含报告/推理所需学生权重。
def save_checkpoint(
    path: Path,  # 目标 .pt 文件路径。
    student: nn.Module,  # 当前学生模型。
    discriminator: nn.Module,  # 当前波形判别器。
    student_optimizer: torch.optim.Optimizer,  # 学生 Adam 及其动量状态。
    discriminator_optimizer: torch.optim.Optimizer,  # D Adam 及其动量状态。
    epoch: int,  # 已完整完成并测试的轮次。
    global_step: int,  # 已完成的完整 batch 更新数。
    run_contract_sha256: str,  # 当前代码/数据/依赖契约的稳定哈希。
    epoch_losses: Mapping[str, float],  # 与本轮 CSV 行对应的全部 train/test loss。
    include_training_state: bool,  # True=可续训 latest；False=精简学生权重归档。
# 结束 save_checkpoint 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # 所有检查点共同保存格式、位置、契约、loss 和学生权重。
    payload: Dict[str, Any] = {
        # 检查点序列化格式版本。
        "format_version": CHECKPOINT_FORMAT_VERSION,  # 当前固定版本 3。
        # 根据是否包含优化器/D/RNG 明确标记文件用途。
        "checkpoint_type": (
            # 完整断点含优化器和随机状态，精简归档只保留学生权重与身份。
            "full_training_state" if include_training_state else "student_weights_only"
        ),
        # 当前完整训练轮次。
        "epoch": int(epoch),  # 转普通整数，避免 NumPy/PyTorch 标量兼容问题。
        # 检查点记录的累计优化器更新步数。
        "global_step": int(global_step),  # 精确续训游标。
        # 检查点绑定的训练契约规范化指纹。
        "run_contract_sha256": run_contract_sha256,  # 防止不同实验配置之间误续训。
        # 把最后一轮所有 loss/覆盖计数复制进 checkpoint，与 CSV 双向核对。
        "epoch_losses": {
            # 覆盖字段保留整数，损失字段保存完整 Python float 精度。
            key: (
                # 测试样本数和批数属于计数，保存时转成整数。
                int(epoch_losses[key])  # test_examples/test_batches。
                if key in LOSS_COUNT_FIELDS  # 判断是否为计数列。
                else float(epoch_losses[key])  # train/test 损失列。
            )
            for key in LOSS_FIELDS[2:]  # 按固定 schema 保存所有非游标字段。
        },
        # 学生网络参数字典。
        "student": student.state_dict(),  # 学生所有参数和缓冲区。
    }
    # latest.pt 必须附带下一轮精确继续所需的全部可变状态。
    if include_training_state:
        # 把续训专用字段合并到 payload。
        payload.update(
            {
                "discriminator": discriminator.state_dict(),  # D 参数和缓冲区。
                "student_optimizer": student_optimizer.state_dict(),  # 学生 Adam 动量/步数。
                "discriminator_optimizer": discriminator_optimizer.state_dict(),  # D Adam 状态。
                "rng_state": capture_rng_state(),  # Python/NumPy/CPU/CUDA 随机游标。
            }
        )
    # 无论大检查点还是精简权重都采用临时文件+原子替换。
    atomic_torch_save(payload, path)


# 严格加载 latest.pt，核对格式/契约/游标/loss 后恢复模型、优化器和 RNG。
def load_resume_checkpoint(
    path: Path,  # 必须是当前输出目录的 latest.pt。
    student: nn.Module,  # 已按当前配置新建的学生，接收权重。
    discriminator: nn.Module,  # 已按当前配置新建的 D。
    student_optimizer: torch.optim.Optimizer,  # 新建学生 Adam，接收历史状态。
    discriminator_optimizer: torch.optim.Optimizer,  # 新建 D Adam。
    device: torch.device,  # 用于判断是否必须恢复 CUDA RNG。
    expected_run_contract_sha256: str,  # 当前运行重新计算的契约哈希。
    batches_per_epoch: int,  # 当前数据应有每轮 1230 批。
    max_epochs: int,  # 最大 100 轮，检查游标范围。
# 结束 load_resume_checkpoint 的参数声明；返回类型为 Tuple[int, int, Dict[str, float]]，具体内容由下方实现给出。
) -> Tuple[int, int, Dict[str, float]]:
    # 读取检查点到 CPU 映射结构，再按 state_dict 加入当前设备模型。
    checkpoint = load_torch_checkpoint(path)
    # 只接受当前格式版本 3，避免旧字段被误解释。
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        # 版本不符拒绝续训。
        raise RuntimeError("Student resume checkpoint format mismatch")
    # 精简 epoch_XXXX.pt 没有 D/优化器/RNG，不能作为精确续训来源。
    if checkpoint.get("checkpoint_type") != "full_training_state":
        # 指明只能用 latest.pt 类型。
        raise RuntimeError("Only latest.pt full_training_state can resume training")
    # 检查点必须属于当前完全相同的代码、数据、教师、声码器与配置契约。
    if checkpoint.get("run_contract_sha256") != expected_run_contract_sha256:
        # 契约变化时从旧状态继续会混合实验，因此拒绝。
        raise RuntimeError("Run contract changed; refusing a non-identical resume")
    # 读取已完成 epoch 游标。
    epoch = checkpoint.get("epoch")
    # 读取累计更新步数。
    global_step = checkpoint.get("global_step")
    # epoch 必须是真正 Python int，且在 1～100 范围内。
    if type(epoch) is not int or not 1 <= epoch <= max_epochs:
        # bool 虽是 int 子类但 type(...) is not int 也会被排除。
        raise RuntimeError(f"Invalid resume epoch cursor: {epoch!r}")
    # 只在完整 epoch 边界保存，因此步数必须严格等于 epoch×1230。
    if type(global_step) is not int or global_step != epoch * batches_per_epoch:
        # 任一不符都无法证明模型和数据迭代器处于同一位置。
        raise RuntimeError(
            # 恢复训练前核对epoch与优化器step是否一致。
            "Invalid resume optimizer-step cursor: "
            # 错误中列出断点声称的轮次和步数。
            f"epoch={epoch!r}, global_step={global_step!r}, "
            # 同时显示每轮实际批次数，便于验证epoch×步数关系。
            f"batches_per_epoch={batches_per_epoch}"
        )
    # 读取检查点附带的最后一轮 loss 记录。
    saved_losses = checkpoint.get("epoch_losses")
    # 必须是映射且字段集合与当前固定 CSV schema 完全一致。
    if not isinstance(saved_losses, Mapping) or set(saved_losses) != set(LOSS_FIELDS[2:]):
        # 字段不全/旧格式都拒绝。
        raise RuntimeError("Resume checkpoint has an invalid epoch_losses record")
    # 将覆盖计数恢复为整数、loss 恢复为浮点。
    epoch_losses = {
        # 每个 key 按字段类型转换。
        key: (
            int(saved_losses[key])  # 覆盖计数。
            if key in LOSS_COUNT_FIELDS  # examples/batches 两项。
            else float(saved_losses[key])  # 其余 14 项 loss。
        )
        for key in LOSS_FIELDS[2:]  # 固定顺序遍历。
    }
    # 所有 loss 和计数必须能表示为有限数。
    if not all(math.isfinite(value) for value in epoch_losses.values()):
        # 检查点中已有 NaN/Inf 时不能继续污染训练。
        raise RuntimeError("Resume checkpoint contains a non-finite epoch loss")
    # strict=True 要求学生每个参数名/形状完全匹配当前代码。
    student.load_state_dict(checkpoint["student"], strict=True)
    # 同样严格恢复判别器 8 分支权重。
    discriminator.load_state_dict(checkpoint["discriminator"], strict=True)
    # 恢复学生 Adam 的一阶/二阶矩和内部步数。
    student_optimizer.load_state_dict(checkpoint["student_optimizer"])
    # 恢复判别器 Adam 状态。
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    # 最后恢复全部随机游标，使下一 epoch 的打乱和噪声与不中断时一致。
    restore_rng_state(checkpoint["rng_state"], device)
    # 下一轮从 epoch+1 开始；global_step 不增加；同时返回末轮 loss 供 CSV 互证。
    return epoch + 1, global_step, epoch_losses


# 新训练启动前保护现有正式产物；只有显式 --resume 才能使用已有目录。
def ensure_output_safety(output_dir: Path, resume: Optional[Path]) -> None:
    # 收集任何代表一次训练已开始/完成的核心文件。
    existing = [
        path  # 将实际存在的路径加入列表。
        # 检查本次目录中的已有实验文件，避免无意覆盖历史训练。
        for path in (
            output_dir / "latest.pt",  # 可续训完整状态。
            output_dir / "loss.csv",  # 每轮 train/test loss。
            output_dir / "run_contract.json",  # 实验身份契约。
            output_dir / "student_total_loss_curves.png",  # 总损失曲线。
            output_dir / "student_loss_component_curves.png",  # 分项曲线。
        )
        if path.exists()  # 不存在的候选不加入 existing。
    ]
    # 同时把每 10 轮保存的 epoch_XXXX.pt 归档权重加入冲突检查。
    existing.extend(output_dir.glob("epoch_*.pt"))
    # 没有 --resume 却发现任何旧产物时，禁止覆盖。
    if resume is None and existing:
        # 提醒用户必须明确从 latest.pt 续训，而不是静默重开覆盖历史。
        raise RuntimeError(
            # 已存在训练结果时明确拒绝直接覆盖。
            f"Refusing to overwrite an existing student run in {output_dir}; "
            # 提示使用完整最新断点续训，以恢复正确优化器与随机状态。
            "use --resume with latest.pt"
        )


# 主入口按“解析→审计→加载→建优化器→可选续训→100 轮训练/测试/保存/画图”执行。
def main() -> None:
    # 读取项目/权重/输出/续训/设备等命令行参数。
    args = parse_args()
    # 创建当前运行的配置副本，把命令行修复模式写入正式契约。
    config = replace(
        # 以锁定的论文及复现默认配置为基础。
        PAPER_STUDENT_CONFIG,
        # 采用启动命令指定的教师目标空间。
        score_target=args.score_target,
        # 采用启动命令指定的噪声权重解释。
        score_weight=args.score_weight,
        # 采用本次要求的监控窗口长度。
        loss_monitor_window_steps=args.loss_monitor_window,
    )
    # 窗口至少需要一个训练批次。
    if config.loss_monitor_window_steps < 1:
        # 在启动昂贵训练前拒绝无效报警配置。
        raise ValueError("--loss-monitor-window must be positive")
    # 计划暂停轮次必须落在完整100轮范围内。
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= config.epochs:
        # 拒绝0轮、负轮或超出实验终点的暂停位置。
        raise ValueError("--stop-after-epoch must be within the planned 1..100 epochs")
    # 用户要求每轮都有 train/test loss 曲线，因此频率必须固定为每 1 轮。
    if config.test_loss_every_epochs != 1:
        # 配置若被改成稀疏测试就停止，避免产生不符合要求的报告数据。
        raise RuntimeError(
            # 要求每轮都有固定测试点，确保训练和测试曲线可逐轮比较。
            "The requested student train/test curves require one held-out point per epoch"
        )
    # 展开 ~ 并转绝对路径，避免运行中工作目录变化影响定位。
    project_root = args.project_root.expanduser().resolve()
    # 教师检查点路径同样规范化。
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    # 声码器权重路径规范化。
    args.vocoder_checkpoint = args.vocoder_checkpoint.expanduser().resolve()
    # 声码器配置路径规范化。
    args.vocoder_config = args.vocoder_config.expanduser().resolve()
    # 声码器验真记录路径规范化。
    args.vocoder_attestation = args.vocoder_attestation.expanduser().resolve()
    # 学生输出目录规范化。
    args.output_dir = args.output_dir.expanduser().resolve()
    # 只有用户提供 --resume 时才进入严格续训路径检查。
    if args.resume is not None:
        # 将提供的续训文件路径规范化。
        args.resume = args.resume.expanduser().resolve()
        # 唯一允许的续训文件是当前输出目录内的 latest.pt。
        expected_resume = (args.output_dir / "latest.pt").resolve()
        # 防止误拿 epoch 权重或另一次实验的 latest。
        if args.resume != expected_resume:
            # 同时显示要求路径和实际路径。
            raise RuntimeError(
                # 恢复路径必须指向当前实验的正确latest.pt，不能混入其他运行。
                f"Strict resume accepts only {expected_resume}, got {args.resume}"
            )

    # 读取服务器身份/路径标记，强制所有计算只在租用 Matpool 实例进行。
    assert_rented_server(project_root)
    # 返回的打开句柄故意在整个进程存活，使三把独占锁始终有效。
    # 下划线变量仍被 main 持有，不会提前垃圾回收释放锁。
    _runtime_lock_handles = acquire_runtime_locks(project_root)
    # 要求 Git 工作区干净且有有效 HEAD；返回值稍后也会写进运行契约。
    git_head(project_root)
    # 启动前逐个确认四个不可缺少的教师/声码器依赖文件真实存在。
    for required in (
        args.teacher_checkpoint,  # VoiceGrad epoch500 教师。
        args.vocoder_checkpoint,  # 修改版 HiFi-GAN 250 万步 Generator。
        args.vocoder_config,  # 与该 Generator 配套的配置。
        args.vocoder_attestation,  # 最终检查点验真记录。
    ):
        # is_file 同时排除不存在路径和同名目录。
        if not required.is_file():
            # 缺任何一个都不允许回退到公开模型或随机初始化。
            raise FileNotFoundError(f"Strict student dependency is missing: {required}")

    # 把 "cuda" 等文本转换为 PyTorch 设备对象。
    device = torch.device(args.device)
    # 正式训练只允许 CUDA；CPU 只可用于 --preflight-only 的只读结构检查。
    if not args.preflight_only and device.type != "cuda":
        # 防止误在本机/CPU 开始极慢且不符合 A100 设定的训练。
        raise RuntimeError("Formal FastVoiceGrad training requires CUDA on the rented server")
    # 即便请求 cuda，也必须确认当前 PyTorch 实际能看到 CUDA。
    if device.type == "cuda" and not torch.cuda.is_available():
        # 驱动、容器或 GPU 分配异常时立即停止。
        raise RuntimeError("CUDA was requested but is unavailable")
    # 在模型初始化、DataLoader 和噪声生成之前固定全局种子。
    seed_everything(config.seed)

    # 首先运行仓库现有的全量、只读数据审计；任何失败都不会创建训练输出。
    run_full_data_audit(project_root, config.expected_protocol)
    # 再做本文件的 text-unseen 结构验证并收集数据契约。
    data_contract = validate_data_protocol(project_root, config)
    # 加载冻结 epoch500 教师、由其初始化的学生及教师检查点元数据。
    teacher, student, teacher_checkpoint = load_teacher_and_student(
        args.teacher_checkpoint, config, device  # 固定权重、配置和 CUDA 设备。
    )
    # 加载冻结的自训练修改版 HiFi-GAN，并取得配置与检查点元数据。
    vocoder, hifigan_config, vocoder_checkpoint = load_frozen_vocoder(
        project_root,  # 用于解析源码/相对依赖。
        args.vocoder_checkpoint,  # 2,500,000 步 Generator。
        args.vocoder_config,  # 配套网络配置。
        args.vocoder_attestation,  # 哈希/步数验真。
        config,  # 核对声学和学生设置。
        device,  # 加载到 A100。
    )
    # 根据论文/参考结构创建新的可训练波形判别器 MPD5+MRD3。
    discriminator = build_paper_discriminator(
        project_root, hifigan_config, config, device  # 所需代码、声码器配置、严格设置和设备。
    )
    # 创建 T=1000 的 VoiceGrad 扩散对象并搬到 GPU。
    diffusion = VoiceGradDiffusion(
        config.diffusion_steps, offset=config.cosine_offset  # 1000 步与余弦 offset。
    # 构建完成的模型或扩散系数统一搬到本次计算设备。
    ).to(device)
    # 加载训练集 80 维 mel 均值/标准差到 GPU，供每批声码器输入反归一化。
    mel_mean, mel_std = load_mel_statistics(project_root / "data", device)

    # 对比当前数据与教师/声码器内保存指纹，保证三阶段使用完全同一数据版本。
    validate_dependency_data_fingerprints(
        data_contract, teacher_checkpoint, vocoder_checkpoint  # 当前、教师历史、声码器历史。
    )
    # 确保四个模型没有意外引用同一 Parameter 对象。
    assert_disjoint_parameter_sets(
        (
            ("teacher", teacher),  # 冻结教师参数集合。
            # 学生网络参数字典。
            ("student", student),  # 可训练学生参数集合。
            ("vocoder", vocoder),  # 冻结声码器参数集合。
            ("discriminator", discriminator),  # 可训练 D 参数集合。
        )
    )
    # 确认教师每个参数 requires_grad=False。
    assert_frozen(teacher, "teacher")
    # 确认声码器每个参数 requires_grad=False。
    assert_frozen(vocoder, "vocoder")
    # 确认学生至少有并且所有应训练参数已开启梯度。
    assert_trainable(student, "student")
    # 确认判别器可训练。
    assert_trainable(discriminator, "discriminator")

    # 组装包含论文项、自定项、数据/代码/权重指纹和运行环境的完整契约。
    run_contract = build_run_contract(
        project_root,  # 权威服务器路径。
        args,  # 本次命令行设置。
        config,  # 论文严格配置。
        data_contract,  # 当前数据身份。
        teacher_checkpoint,  # 教师元数据。
        vocoder_checkpoint,  # 声码器元数据。
    )
    # 对规范化契约计算 SHA-256，检查点以后只接受这一完全相同契约。
    run_contract_sha = canonical_json_sha256(run_contract)
    # 在日志完整打印契约，VS Code 中可实时查看所有设定和指纹。
    print(json.dumps(run_contract, ensure_ascii=False, indent=2, sort_keys=True))
    # 单独打印契约哈希，便于检索和报告引用。
    print(f"run_contract_sha256={run_contract_sha}")
    # 预检查模式在所有数据/模型/依赖审计成功后到此退出，不创建训练产物。
    if args.preflight_only:
        # 明确告诉日志“检查通过且没有开始训练”。
        print("[OK] Strict FastVoiceGrad dependency preflight passed; no training was started.")
        # 退出 main；进程结束时自动释放运行锁。
        return

    # 正式训练前防止无 --resume 覆盖旧 checkpoint/CSV/图片。
    ensure_output_safety(args.output_dir, args.resume)
    # 递归创建学生输出目录；已存在的合法续训目录保持不变。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 运行契约固定保存在输出目录根部。
    contract_path = args.output_dir / "run_contract.json"
    # 新训练需要首次原子写入本次契约。
    if args.resume is None:
        # 写盘并 fsync，保证权重产生前已有完整实验身份记录。
        atomic_write_json(contract_path, run_contract)
    # 续训不能覆盖契约，而要读取旧契约并重新验证。
    else:
        # 旧契约必须存在。
        if not contract_path.is_file():
            # 缺契约的权重无法证明来源，因此拒绝续训。
            raise FileNotFoundError(f"Run contract missing for resume: {contract_path}")
        # 以 UTF-8 读取保存契约。
        with contract_path.open("r", encoding="utf-8") as handle:
            # 解析旧 JSON。
            saved_contract = json.load(handle)
        # 重新规范哈希并与当前契约比较，排除代码/数据/参数变化。
        if canonical_json_sha256(saved_contract) != run_contract_sha:
            # 不允许跨契约续训。
            raise RuntimeError("Saved run_contract.json differs from active dependencies/code")

    # 构造正式训练加载器：39,387 条、shuffle、drop_last、每批 32、每段 128 帧。
    train_loader = get_dataloader(
        project_root / "data",  # 已审计数据根目录。
        "train",  # 只读取训练 manifest。
        # 每个训练批的样本数。
        batch_size=config.batch_size,  # 论文全局 batch=32。
        num_workers=config.num_workers,  # 数据读取 worker 数，只影响吞吐。
        segment_length=config.train_frames,  # 每条随机裁 128 帧。
        expected_speakers=config.train_speakers,  # 必须为 100 位训练说话人。
        # BNF 到梅尔的帧对齐模式。
        bnf_alignment=config.bnf_alignment,  # mel/BNF 对齐方式。
    )
    # 构造每轮完整测试 100 条的固定、不打乱加载器。
    test_loader = build_test_loss_loader(project_root / "data", config)
    # 创建只更新学生参数的 Adam；学习率/beta/eps 等全部显式登记。
    student_optimizer = torch.optim.Adam(
        student.parameters(),  # 优化器唯一拥有学生 theta 参数。
        lr=config.learning_rate,  # 论文学习率 2e-4；越大更新更快但更易不稳定。
        betas=(config.adam_beta1, config.adam_beta2),  # 论文 Adam (0.5,0.9) 动量衰减。
        eps=config.adam_epsilon,  # 数值稳定项 1e-8。
        weight_decay=config.weight_decay,  # 设为 0，不加额外 L2 正则。
        amsgrad=config.adam_amsgrad,  # False，使用标准 Adam 而非 AMSGrad 变体。
    )
    # 判别器使用相同 Adam 超参数，但参数集合只包含 D。
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),  # 8 个判别器分支参数。
        lr=config.learning_rate,  # 同为 2e-4。
        betas=(config.adam_beta1, config.adam_beta2),  # 同为 (0.5,0.9)。
        eps=config.adam_epsilon,  # 同为 1e-8。
        # 按已登记选择关闭权重衰减，值为零。
        weight_decay=config.weight_decay,  # 0。
        # 按已登记选择关闭 Adam 的 AMSGrad 变体。
        amsgrad=config.adam_amsgrad,  # False。
    )
    # 证明学生 Adam 只含学生参数，且不含教师/声码器/D。
    assert_optimizer_ownership(
        student_optimizer,  # 被检查的优化器。
        student,  # 它应完整拥有的模型。
        (teacher, vocoder, discriminator),  # 它绝不能拥有的模型。
        "student Adam",  # 错误信息中的易读名称。
    )
    # 证明判别器 Adam 只含 D 参数，且不含教师/声码器/学生。
    assert_optimizer_ownership(
        discriminator_optimizer,  # 被检查的 D 优化器。
        discriminator,  # 应完整拥有的 D。
        (teacher, vocoder, student),  # 不允许混入的模型。
        "discriminator Adam",  # 日志名称。
    )

    # 每轮训练/测试损失统一写在输出目录 loss.csv。
    loss_path = args.output_dir / "loss.csv"
    # 新训练默认从第 1 轮、全局第 0 步开始。
    start_epoch, global_step = 1, 0
    # 提供 --resume 时恢复上一个完整 epoch 边界。
    if args.resume is not None:
        # 严格加载学生/D/两个 Adam/RNG，并取得下一轮游标和末轮 loss。
        start_epoch, global_step, checkpoint_losses = load_resume_checkpoint(
            args.resume,  # 当前输出目录 latest.pt。
            student,  # 接收学生状态。
            discriminator,  # 接收 D 状态。
            student_optimizer,  # 接收学生 Adam 状态。
            discriminator_optimizer,  # 接收 D Adam 状态。
            device,  # 恢复 CUDA RNG。
            run_contract_sha,  # 必须匹配的当前契约。
            batches_per_epoch=len(train_loader),  # 应为 1230。
            max_epochs=config.epochs,  # 最多 100。
        )
        # 核对/必要时修复 CSV 唯一合法尾行，再与 checkpoint loss 逐项比较。
        reconcile_loss_csv(
            loss_path,  # 已有历史 CSV。
            completed_epoch=start_epoch - 1,  # checkpoint 已完成轮次。
            batches_per_epoch=len(train_loader),  # 每轮 1230 步。
            checkpoint_losses=checkpoint_losses,  # checkpoint 末轮记录。
        )
        # 根据修复/核对后的 CSV 重新生成最新两张曲线，不改变已恢复 RNG。
        render_loss_plots_without_rng_side_effects(
            loss_path, args.output_dir, device  # 输入 CSV、输出目录、当前 CUDA 设备。
        )
        # 在实时日志显示实际恢复位置。
        print(f"Resumed epoch={start_epoch}, global_step={global_step}")

    # 训练开始前创建持久化损失窗口和报警记录。
    monitor = LossMonitor(
        # 监控文件与本次检查点放在同一实验目录，使用已登记窗口长度。
        args.output_dir, window_steps=config.loss_monitor_window_steps,
        # 只恢复不晚于当前模型断点的历史窗口。
        resume_step=global_step,
    )
    # 根据下一轮起点推回已经完整训练并保存的轮次。
    completed_epoch = start_epoch - 1
    # 本次运行要求中途检查时，提前公布整轮暂停位置。
    if args.stop_after_epoch is not None:
        # 在日志说明本次预设暂停位置与完整实验终点。
        print(
            # 日志明确显示本次将在指定轮完成后暂停。
            f"Scheduled pause after epoch {args.stop_after_epoch}; planned training "
            # 再次声明实验完整终点仍为100轮，避免把暂停误判为训练完毕。
            f"endpoint remains epoch {config.epochs}.", flush=True,
        )
        # 恢复断点已达到暂停点时，无须再执行额外训练。
        if completed_epoch >= args.stop_after_epoch:
            # 将完成、暂停或失败状态写到监控快照，供外部确认运行结果。
            monitor.finish(
                # 只有完成计划全部轮次才标记completed，中途暂停标记stopped。
                "completed" if completed_epoch == config.epochs else "stopped",
                # 结束状态携带实际更新步数和最后完整轮次。
                global_step, completed_epoch,
            )
            # 断点已达到本次暂停位置，结束当前运行而不再更新模型。
            return

    # 可变字典让嵌套信号处理函数无需 nonlocal 就能标记停止请求。
    stop_requested = {"value": False}

    # SIGINT/SIGTERM 到来时只做轻量标记，让当前 epoch 完整结束并保存。
    def request_stop(signum, _frame) -> None:
        # 设置标志；训练循环会在本轮 checkpoint/图表写完后检查。
        stop_requested["value"] = True
        # 立即把安全停止策略打印到日志。
        print(
            # 收到停止信号后完成当前整轮再保存，保留可精确恢复的断点。
            f"Signal {signum} received; finishing the current epoch before an exact checkpoint.",
            flush=True,  # 立刻刷新，VS Code tail 能马上看到。
        )

    # Ctrl+C 对应 SIGINT，注册同一个安全停止处理器。
    signal.signal(signal.SIGINT, request_stop)
    # 系统 kill -TERM/实例停止通常发 SIGTERM，也注册安全处理器。
    signal.signal(signal.SIGTERM, request_stop)
    # 从新训练第 1 轮或续训下一轮开始，一直到固定第 100 轮（右端+1）。
    try:
        # 从恢复起点逐轮训练直到100轮，终点在范围中包含。
        for epoch in range(start_epoch, config.epochs + 1):
            # 先完成本轮所有 1230 个训练 batch，得到 7 项 train loss 均值。
            train_losses, global_step = train_epoch(
                train_loader,  # 正式 batch32 训练数据。
                epoch,  # 当前轮次。
                global_step,  # 进入轮次时的累计步数。
                teacher,  # 冻结教师。
                student,  # 可训练学生。
                vocoder,  # 冻结自训练声码器。
                discriminator,  # 可训练 D。
                diffusion,  # T=1000 扩散对象。
                student_optimizer,  # 学生 Adam。
                # 传入只拥有判别器参数的独立 Adam 优化器。
                discriminator_optimizer,  # D Adam。
                mel_mean,  # 反归一化均值。
                mel_std,  # 反归一化标准差。
                # 使用租用服务器当前训练 GPU 设备。
                device,  # A100 CUDA。
                config,  # 固定设置。
                # 把同一个监控器传给每轮训练，窗口可以跨轮连续观察。
                monitor=monitor,
            )
            # 训练结束立即在固定 100 条未见说话人数据上计算 7 项 test loss。
            test_losses, test_examples, test_batches = evaluate_test_epoch(
                test_loader,  # 100 条固定测试窗口。
                teacher,  # 冻结教师。
                student,  # 本轮更新后的学生。
                vocoder,  # 冻结声码器。
                discriminator,  # 本轮更新后的 D。
                diffusion,  # 相同扩散定义。
                mel_mean,  # 相同 mel 均值。
                mel_std,  # 相同 mel 标准差。
                # 在当前租用服务器选定的 CUDA 设备执行。
                device,  # CUDA。
                config,  # 固定测试种子/设置。
            )
            # 合并 train/test 前缀字段与 100 样本/4 批覆盖计数。
            losses = make_epoch_loss_record(
                train_losses,  # 7 项训练均值。
                test_losses,  # 7 项测试均值。
                # 将实际覆盖的 100 条测试语音数写入逐轮记录。
                test_examples,  # 100。
                # 将含末尾小批在内的四批测试数量写入逐轮记录。
                test_batches,  # 4。
            )
            # 先原子写 CSV；若后续 checkpoint 中断，续训会识别并裁掉唯一尾行。
            append_loss_csv(loss_path, epoch, global_step, losses)
            # 将本轮主要 train/test 数值写入实时日志，便于无需打开 CSV 快速查看。
            print(
                f"epoch={epoch:03d} step={global_step} "  # 轮次与累计步数。
                f"train_total={losses['train_student_total']:.6f} "  # 训练总 ACDD。
                f"test_total={losses['test_student_total']:.6f} "  # 测试总 ACDD。
                f"train_adv={losses['train_student_adv']:.6f} "  # 训练对抗项。
                f"test_adv={losses['test_student_adv']:.6f} "  # 测试对抗项。
                f"train_fm={losses['train_fm_raw']:.6f} "  # 训练原始 FM。
                f"test_fm={losses['test_fm_raw']:.6f} "  # 测试原始 FM。
                f"train_dist={losses['train_dist_raw']:.6f} "  # 训练原始蒸馏项。
                f"test_dist={losses['test_dist_raw']:.6f} "  # 测试原始蒸馏项。
                f"train_D={losses['train_discriminator']:.6f} "  # 训练 D loss。
                f"test_D={losses['test_discriminator']:.6f} "  # 测试 D 诊断。
                f"test_examples={test_examples}",  # 证明覆盖 100 条。
                flush=True,  # 立即显示到 VS Code 日志。
            )
            # 每轮都原子覆盖 latest.pt，保存可精确续训的完整状态。
            save_checkpoint(
                args.output_dir / "latest.pt",  # 固定完整状态路径。
                student,  # 当前学生参数。
                discriminator,  # 当前 D 参数。
                student_optimizer,  # 学生 Adam 状态。
                discriminator_optimizer,  # D Adam 状态。
                epoch,  # 当前已完成轮。
                global_step,  # 当前步数。
                run_contract_sha,  # 实验契约哈希。
                losses,  # 本轮 train/test 记录。
                include_training_state=True,  # 包含 D/优化器/RNG，可续训。
            )
            # 每 10 轮以及最终第 100 轮额外保存不会被覆盖的学生权重归档。
            if epoch % config.save_every_epochs == 0 or epoch == config.epochs:
                # 精简归档用于推理/评估和里程碑保留，不作为续训入口。
                save_checkpoint(
                    args.output_dir / f"epoch_{epoch:04d}.pt",  # 如 epoch_0010.pt。
                    student,  # 学生权重会写入。
                    discriminator,  # 参数传入但精简模式不写 D。
                    student_optimizer,  # 精简模式不写优化器。
                    discriminator_optimizer,  # 精简模式不写优化器。
                    epoch,  # 里程碑轮次。
                    global_step,  # 对应累计步数。
                    run_contract_sha,  # 仍绑定实验契约。
                    losses,  # 保存同轮损失供追溯。
                    include_training_state=False,  # 只保留学生与公共元数据。
                )
            # 每轮 CSV/checkpoint 保存后更新总损失图和分项损失图。
            plot_paths = render_loss_plots_without_rng_side_effects(
                loss_path, args.output_dir, device  # 从完整 CSV 画到正式输出目录。
            )
            # 将两张图片绝对路径打印到日志，方便在 VS Code 中直接打开。
            print(
                "loss_plots=" + ",".join(str(path) for path in plot_paths),  # 逗号分隔路径。
                flush=True,  # 立即刷新日志。
            )
            # 若本轮期间收到停止信号，现在已有完整 CSV、latest、归档和图，可以安全退出。
            completed_epoch = epoch
            # 收到用户停止信号或达到预定检查轮次时，退出后续轮次循环。
            if stop_requested["value"] or (
                # 只在完整轮次保存后检查是否到达计划暂停位置。
                args.stop_after_epoch is not None and epoch >= args.stop_after_epoch
            ):
                # 明确记录安全停止发生在 epoch 边界。
                print("Safe stop completed at an epoch boundary.", flush=True)
                # 跳出 100 轮循环，随后 main 正常返回并释放文件锁。
                break
    # 包括训练错误和中断在内的异常都会更新失败状态。
    except BaseException:
        # 将完成、暂停或失败状态写到监控快照，供外部确认运行结果。
        monitor.finish(
            # 失败记录尽量使用最近实际观察到的step，避免只显示上一轮进度。
            "failed", getattr(monitor, "last_global_step", global_step),
            # 同时记录异常发生的最近epoch。
            getattr(monitor, "last_epoch", completed_epoch),
        )
        # 保留原异常继续抛出，让调用方和日志看到真正失败原因。
        raise
    # 将完成、暂停或失败状态写到监控快照，供外部确认运行结果。
    monitor.finish(
        # 只有完成计划全部轮次才标记completed，中途暂停标记stopped。
        "completed" if completed_epoch == config.epochs else "stopped",
        # 结束状态携带实际更新步数和最后完整轮次。
        global_step, completed_epoch,
    )


# 只有 `python -m fastvoicegrad.train` 或直接运行本文件时才启动训练；被测试导入时不启动。
if __name__ == "__main__":
    # 调用上面的完整主流程。
    main()
