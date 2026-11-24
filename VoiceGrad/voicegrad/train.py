# 执行固定教师训练及严格续训，启动前核对当前数据划分与特征来源。
"""Train the paper-aligned FastVoiceGrad VoiceGrad teacher on VCTK."""

# =============================================================================
# 这是整个教师训练的“总指挥”文件
# =============================================================================
# 一次训练迭代的完整路线：
#   DataLoader 取出干净 mel/BNF/speaker -> 随机选 t -> 随机生成真实噪声 epsilon ->
#   diffusion.q_sample 得到 noisy mel x_t -> model 预测 epsilon_theta ->
#   L1(epsilon_theta, epsilon) -> backward 算梯度 -> Adam 更新权重。
#
# 论文明确的教师训练参数：batch=32、lr=0.0002、Adam betas=(0.9,0.999)、
# 论文训练 500 epoch；本次主复现实验也固定在 500 epoch 结束。
#
# Python/PyTorch 最重要的区别：epoch 是完整遍历一遍训练集；batch 是一次送给
# GPU 的 32 条片段；global_step 是实际执行了多少次 optimizer.step()。

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# --------------------------- 导入包逐项解释 ---------------------------
# argparse 解析命令行，例如 --epochs 500；csv 写 loss.csv；json 漂亮打印配置。
import argparse
# 按清单表头读取说话人、句子及文件路径，并写回实验索引。
import csv
# hashlib records the exact active manifests/statistics in every checkpoint.
# 用标准库 SHA-256 为数据、模型或来源记录计算可比较的内容指纹。
import hashlib
# 保存或读取实验设置、数据指纹及处理进度的结构化记录。
import json
# os.replace 用于原子替换 checkpoint；random 用于固定 Python 随机种子。
import os
# 管理训练顺序与片段裁剪使用的 Python 随机数流。
import random
# Path 用路径对象管理 data/exp/checkpoint 目录。
from pathlib import Path
# Dict 是类型提示，表示“键和值组成的字典”，不会改变程序运行结果。
from typing import Dict

# NumPy 这里只用于设置随机种子；torch 负责模型、Tensor、GPU、优化器和 checkpoint。
import numpy as np
# 使用 PyTorch 张量表示声学特征、扩散状态及网络参数。
import torch
# F.l1_loss 实现论文的 L1 噪声预测损失。
import torch.nn.functional as F
# tqdm 给训练/测试循环加终端进度条，不影响数学结果。
from tqdm import tqdm

# 下面是本工程自己的模块，而不是 pip 包：
# config 给统一参数；dataset 给 batch；diffusion 负责加噪；model 是噪声预测网络。
from .config import PAPER_CONFIG, SERVER_PROJECT_ROOT
# 复用 get_dataloader 的统一实现，具体输入和实验约束由调用处指定。
from .dataset import get_dataloader
# 复用 VoiceGradDiffusion 的统一实现，具体输入和实验约束由调用处指定。
from .diffusion import VoiceGradDiffusion
# 复用 ALIGNMENT_MODES, LEGACY_ALIGNMENT 的统一实现，具体输入和实验约束由调用处指定。
from .feature_alignment import ALIGNMENT_MODES, LEGACY_ALIGNMENT
# 复用 ( 的统一实现，具体输入和实验约束由调用处指定。
from .model import (
    # 导入网络内部 BNF 下采样的合法模式名称。
    BNF_DOWNSAMPLING_MODES,
    # 导入历史模型所用的间隔抽帧规则，保护既有检查点输入契约。
    LEGACY_BNF_DOWNSAMPLING,
    # 导入教师与学生共用的噪声预测网络结构。
    VoiceGrad,
)
# 复用 ( 的统一实现，具体输入和实验约束由调用处指定。
from .validate_setup import (
    # 导入 mel 采样率、窗口和频带设置的来源审核函数。
    audit_mel_extraction_recipe,
    # 导入逐特征提取器身份及来源台账审核函数。
    audit_feature_provenance,
    # 导入均值与标准差成套发布一致性的检查函数。
    audit_mel_stats_publication,
    # 导入统一的语音清单读取入口。
    read_manifest,
)


# 说明并解析本脚本的输入目录、实验协议和可选运行参数；下面的参数分别限定输入数据和处理约束。
def parse_args() -> argparse.Namespace:
    # argparse 把终端命令解析成 args 对象。例如 --epochs 500 最后是 args.epochs。
    # default 表示用户不写该参数时用什么；required=True 则表示必须提供。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--project-root", type=Path, default=SERVER_PROJECT_ROOT)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 显式指定本次结果目录，避免和其他实验共用输出。
        "--output-dir",
        # 把命令行路径直接解析为 Path 对象，后续统一处理相对和绝对目录。
        type=Path,
        # 未显式指定时保留为空，由本函数随后确定默认路径或处理范围。
        default=None,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help=(
            # 帮助文字说明相对检查点目录如何基于工程根目录解析。
            "Checkpoint/loss directory. Relative paths are resolved under project-root; "
            # 记录历史默认教师检查点路径，便于兼容旧运行。
            "the legacy default is exp/checkpoints/voicegrad30_vctk."
        ),
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--epochs", type=int, default=PAPER_CONFIG.epochs)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--batch-size", type=int, default=PAPER_CONFIG.batch_size)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--num-workers", type=int, default=4)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--save-every", type=int, default=100)
    # 工程只有 train/test，没有 validation。为了不在训练期间反复查看 held-out test，
    # 本次固定只在第 500 epoch 终点记录一次 train/test 去噪 L1。
    parser.add_argument(
        # 设置训练/测试 L1 的记录间隔，本教师入口要求固定 500 轮终点。
        "--loss-every",
        # 此选项必须为整数，防止轮数、批大小或条数出现小数。
        type=int,
        # 默认只在预先固定的第 500 轮记录未见测试集诊断，避免反复查看测试集来选模型。
        default=PAPER_CONFIG.paper_epochs,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="Log train/test L1 only at the fixed 500-epoch endpoint.",
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--resume", type=Path, default=None)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--seed", type=int, default=PAPER_CONFIG.seed)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 要求磁盘划分报告与给定 seen/unseen 协议严格一致。
        "--expected-protocol",
        # 允许显式指定文本未见或文本已见协议，随后必须与磁盘划分报告一致。
        choices=("text-unseen", "text-seen"),
        # 默认使用本次正式教师实验的文本未见协议。
        default="text-unseen",
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help="Refuse to train unless the active split report has this exact protocol.",
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 选择 BNF 与 mel 之间的外部时间轴对应方式。
        "--bnf-alignment",
        # 参数解析器只接受已经实现的外部 BNF 时间对齐方式。
        choices=ALIGNMENT_MODES,
        # 默认沿用历史长度比例插值，避免续训时突然改变输入。
        default=LEGACY_ALIGNMENT,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help=(
            # 帮助文字强调现有权重已依赖固定 BNF/mel 外部对应方式。
            "BNF/mel time-axis rule. Existing checkpoints require "
            # 历史权重保持长度比例插值；物理时轴线性对齐需要另立新实验。
            "legacy_length_linear; use time_linear only for a new run."
        ),
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument(
        # 选择 U-Net 内部不同时间分辨率上如何采样 BNF。
        "--bnf-downsampling",
        # 参数解析器只接受已经实现的网络内部 BNF 采样方式。
        choices=BNF_DOWNSAMPLING_MODES,
        # 默认保留历史间隔抽点规则；中心对齐只能另立实验。
        default=LEGACY_BNF_DOWNSAMPLING,
        # 下面的命令行帮助说明此选项的用途和实验限制。
        help=(
            # 帮助文字说明内部 BNF 对齐分别发生在完整、二分之一和四分之一时间尺度。
            "BNF alignment inside T/T2/T4 U-Net levels. Existing checkpoints use "
            # 历史权重沿用间隔抽帧，中心对齐规则只能作为新的消融实验。
            "legacy_decimate; centered_linear is a separate new-run ablation."
        ),
    )
    # 完成终端参数解析，返回经过类型和选项检查的运行设置。
    return parser.parse_args()


# 同步固定 Python、NumPy 和 PyTorch 的随机数流；下面的参数分别限定输入数据和处理约束。
def seed_everything(seed: int) -> None:
    # 工程同时用到 Python random、NumPy random、PyTorch CPU/GPU random。
    # 只设置其中一个不够，所以全部设为同一个 seed。
    random.seed(seed)
    # 固定 NumPy 的随机流，使依赖 NumPy 的抽样可重复。
    np.random.seed(seed)
    # 固定 PyTorch 随机流，让加噪或测试诊断可复现。
    torch.manual_seed(seed)
    # 只有 CUDA 实际可用时才处理 GPU 专属随机数状态。
    if torch.cuda.is_available():
        # 同时固定可见 GPU 的随机流，避免模型采样随设备改变。
        torch.cuda.manual_seed_all(seed)


# 依据教师参数表创建同一套 VoiceGrad 网络；下面的参数分别限定输入数据和处理约束。
def build_model(
    # 网络各时间分辨率上的 BNF 取样规则。
    bnf_downsampling: str = LEGACY_BNF_DOWNSAMPLING,
# 返回依据教师参数表创建同一套 VoiceGrad 网络整理的 VoiceGrad 结果，内部各项保持前面约定的对应关系。
) -> VoiceGrad:
    # 所有参数只从 config.py 读取，训练和推理都调用本函数，保证结构完全相同。
    return VoiceGrad(
        # mel 频带数，本实验为 80。
        n_mels=PAPER_CONFIG.n_mels,
        # BNF 内容维数，本实验为 144。
        n_bnf=PAPER_CONFIG.n_bnf,
        # VoiceGrad 主干隐藏通道数，论文设为 512。
        n_channels=PAPER_CONFIG.hidden_channels,
        # 完整扩散链的步数。
        n_levels=PAPER_CONFIG.diffusion_steps,
        # 步编号和说话人各自的投影宽度，本实验选 128。
        cond_dim=PAPER_CONFIG.cond_dim,
        # 各块 BNF 投影宽度，本实验选 32。
        bnf_out_dim=PAPER_CONFIG.bnf_channels,
        # 说话人向量宽度，本实验为 256。
        spk_emb_dim=PAPER_CONFIG.spk_emb_dim,
        # 网络各时间分辨率上的 BNF 取样规则。
        bnf_downsampling=bnf_downsampling,
    )


# 通过临时文件原子发布检查点；下面的参数分别限定输入数据和处理约束。
def atomic_torch_save(payload: Dict[str, object], destination: Path) -> None:
    # “原子保存”的原因：若直接写 destination 时服务器断电，旧文件已被截断，
    # 新文件又没写完。这里先完整写 .tmp，再用 os.replace 一瞬间替换正式文件。
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 尚未正式发布的临时文件。
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    # 先将整份检查点写入临时文件，完整写完再替换正式文件。
    torch.save(payload, temporary)
    # 用完整的临时文件替换正式路径，避免读到只写了一部分的结果。
    os.replace(temporary, destination)


# 逐块读取文件并计算字节内容指纹；下面的参数分别限定输入数据和处理约束。
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # 逐块读取文件并计算字节内容指纹。
    """Return a streaming SHA-256 without loading a whole artifact into RAM."""

    # 逐块累计文件内容的 SHA-256 对象。
    digest = hashlib.sha256()
    # 在受控文件句柄中完成逐块读取文件并计算字节内容指纹，退出此段后自动关闭文件。
    with path.open("rb") as handle:
        # 持续读取文件块，直到遇到文件末尾再结束摘要计算。
        while True:
            # 本次读取的一块文件字节。
            chunk = handle.read(chunk_size)
            # 按本次读取的一块文件字节决定是否进入此分支；判断条件为 not chunk。
            if not chunk:
                # 结束当前扫描循环，保留此前已经收集的结果。
                break
            # 将本行指定的身份、分隔符或数据内容追加到逐块累计文件内容的 SHA-256 对象；字段顺序必须与读取审核时一致。
            digest.update(chunk)
    # 返回小写十六进制 SHA-256，用于比较本次与历史文件是否完全一致。
    return digest.hexdigest()


# 保存影响训练顺序、裁剪和噪声的全部随机数状态；下面的参数分别限定输入数据和处理约束。
def capture_rng_state() -> Dict[str, object]:
    # 保存影响训练顺序、裁剪和噪声的全部随机数状态。
    """Capture every RNG stream that can affect shuffle, crops, t, and noise."""

    # 检查点保存的网络或随机数状态。
    state: Dict[str, object] = {
        # 保存 Python 随机流，恢复后继续相同的数据裁剪过程。
        "python": random.getstate(),
        # 保存 NumPy 随机流，防止续训重启时重新抽样。
        "numpy": np.random.get_state(),
        # 保存 CPU 上的 PyTorch 随机流，覆盖数据打乱等操作。
        "torch_cpu": torch.get_rng_state(),
    }
    # 只有 CUDA 实际可用时才处理 GPU 专属随机数状态。
    if torch.cuda.is_available():
        # 保存全部可见 GPU 的随机流，包括扩散时间步和噪声采样状态。
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    # 返回检查点保存的网络或随机数状态，供调用方继续处理。
    return state


# 恢复随机数状态使下一轮延续原训练抽样轨迹；下面的参数分别限定输入数据和处理约束。
def restore_rng_state(state: Dict[str, object], device: torch.device) -> None:
    # 恢复随机数状态使下一轮延续原训练抽样轨迹。
    """Restore a full-resume checkpoint's RNG streams before the next epoch."""

    # 当前文件或记录必须提供的字段。
    required = {"python", "numpy", "torch_cpu"}
    # 检查中发现的缺失项目。
    missing = required - set(state)
    # 按检查中发现的缺失项目决定是否进入此分支；判断条件为 missing。
    if missing:
        # 恢复随机数状态使下一轮延续原训练抽样轨迹未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Checkpoint RNG state is incomplete: {sorted(missing)}")
    # 恢复 Python 随机流，不在续训时从初始种子重来。
    random.setstate(state["python"])
    # 恢复 NumPy 随机流到检查点保存时的位置。
    np.random.set_state(state["numpy"])
    # 把 CPU 随机状态恢复到保存位置，状态字节本身需位于 CPU。
    torch.set_rng_state(state["torch_cpu"].cpu())
    # 当前运行使用 GPU 时，还要处理该设备的随机数状态。
    if device.type == "cuda":
        # 检查点记录的每张 GPU 随机数流状态。
        cuda_state = state.get("torch_cuda")
        # 按检查点记录的每张 GPU 随机数流状态决定是否进入此分支；判断条件为 cuda_state is None。
        if cuda_state is None:
            # 恢复随机数状态使下一轮延续原训练抽样轨迹未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError("CUDA resume requested but checkpoint has no CUDA RNG state.")
        # 将保存的 GPU 随机状态字节逐一恢复，延续原训练采样序列。
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])


# 保存权重、轮次、配置和可选的完整续训状态；下面的参数分别限定输入数据和处理约束。
def save_checkpoint(
    # 原子发布后的正式文件路径。
    destination: Path,
    # 预测 mel 噪声的 VoiceGrad 网络。
    model: VoiceGrad,
    # 更新教师参数并保存动量的 Adam 优化器。
    optimizer: torch.optim.Optimizer,
    # 当前完整遍历训练集的轮次。
    epoch: int,
    # 从训练开始累计完成的参数更新次数。
    global_step: int,
    # 终端传入并经解析的运行设置。
    args: argparse.Namespace,
    # 是否同时保存 Adam 和随机数状态供严格续训。
    include_optimizer: bool = True,
# 本接口负责保存权重、轮次、配置和可选的完整续训状态，处理结果通过对象状态或文件保存，不额外返回数值。
) -> None:
    # include_optimizer=True 生成“完整恢复检查点”：模型权重之外还保存 Adam 的两个
    # 移动平均状态。只有 latest.pt 使用这种大文件，训练中断时用它才能原样继续。
    # include_optimizer=False 生成“轻量评估检查点”：只保存模型和实验配置，适合推理、
    # 画图和比较 epoch，不能恢复 Adam。每 100 epoch 共保留 5 个阶段权重；它们不会
    # 各自重复保存体积很大的 optimizer state，显著减少服务器磁盘占用。
    payload = {
        # 记录持久化记录的格式版本号。
        "format_version": 5,
        # 记录检查点是完整续训档案还是仅用于推理的权重。
        "checkpoint_type": "full_resume" if include_optimizer else "weights_only",
        # state_dict 是“参数名字 -> Tensor”的字典，不是把整个 Python 对象 pickle。
        "model": model.state_dict(),
        # 记录当前完整遍历训练集的轮次。
        "epoch": epoch,
        # 记录从训练开始累计完成的参数更新次数。
        "global_step": global_step,
        # 记录随权重保存的统一教师实验配置。
        "paper_config": PAPER_CONFIG.as_dict(),
        # 记录检查点登记的实际训练设置。
        "run_config": {
            # 记录保存代码、数据与检查点的项目目录。
            "project_root": str(args.project_root),
            # 记录预先确定的训练轮数。
            "epochs": args.epochs,
            # 记录每批共同送入模型的语音条数。
            "batch_size": args.batch_size,
            # 记录并行读取特征的进程数。
            "num_workers": args.num_workers,
            # 记录复现随机抽样过程使用的固定种子。
            "seed": args.seed,
            # 同一个值决定 train/test 两列在哪些 epoch 同时记录。
            "loss_every": args.loss_every,
            # 记录另存权重快照的轮次间隔。
            "save_every": args.save_every,
            # 记录本次允许使用的 seen/unseen 数据划分协议。
            "expected_protocol": args.expected_protocol,
            # 记录外部 BNF 到 mel 的时间对齐规则。
            "bnf_alignment": args.bnf_alignment,
            # 记录网络各时间分辨率上的 BNF 取样规则。
            "bnf_downsampling": args.bnf_downsampling,
            # 记录绑定数据清单、标准化统计量和特征来源的摘要。
            "data_fingerprints": args.data_fingerprints,
        },
    }
    # 只有完整恢复检查点才加入 optimizer 键；权重文件没有这个键，体积因此显著减小。
    if include_optimizer:
        # 完整续训档额外保存 Adam 动量及步数，不能只加载网络权重。
        payload["optimizer"] = optimizer.state_dict()
        # 把各类随机流状态与同一轮网络、优化器一起保存。
        payload["rng_state"] = capture_rng_state()
    # 将组装好的同一时点状态原子发布为目标检查点。
    atomic_torch_save(payload, destination)


# 核对训练条件后恢复模型、Adam 和随机状态；下面的参数分别限定输入数据和处理约束。
def load_checkpoint(
    # 当前待检查或读写的文件路径。
    path: Path,
    # 预测 mel 噪声的 VoiceGrad 网络。
    model: VoiceGrad,
    # 更新教师参数并保存动量的 Adam 优化器。
    optimizer: torch.optim.Optimizer,
    # 张量执行计算的设备。
    device: torch.device,
    # 本次允许使用的 seen/unseen 数据划分协议。
    expected_protocol: str,
    # 严格续训要求的总训练轮数。
    expected_epochs: int,
    # 严格续训要求的批大小。
    expected_batch_size: int,
    # 严格续训要求的数据加载进程数。
    expected_num_workers: int,
    # 严格续训要求的随机种子。
    expected_seed: int,
    # 当前数据清单、统计量及特征来源的审核指纹。
    expected_data_fingerprints: Dict[str, str],
    # 本次运行要求的外部 BNF 对齐方式。
    expected_bnf_alignment: str = LEGACY_ALIGNMENT,
    # 本次运行要求的网络内部 BNF 对齐方式。
    expected_bnf_downsampling: str = LEGACY_BNF_DOWNSAMPLING,
# 返回核对训练条件后恢复模型、Adam 和随机状态整理的 tuple[int, int] 结果，内部各项保持前面约定的对应关系。
) -> tuple[int, int]:
    # map_location 保证 checkpoint Tensor 被放到当前指定 device，而不是保存时设备。
    checkpoint = torch.load(path, map_location=device)
    # 网络、声学参数及实验约束的配置。
    config = checkpoint.get("paper_config", {})
    # 主实验恢复要求完整配置逐项相同；旧 20-step 或缺字段权重都会被明确拒绝。
    if config != PAPER_CONFIG.as_dict():
        # 核对训练条件后恢复模型、Adam 和随机状态未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 错误说明检查点参数表偏离当前固定 500 轮教师配置。
            "Checkpoint paper_config differs from the fixed 500-epoch teacher "
            # 继续说明该检查点不能接入当前严格续训。
            "configuration; it must not be resumed into this run."
        )
    # 检查点登记的实际训练设置。
    run_config = checkpoint.get("run_config", {})
    # 从当前命令汇集的续训一致性要求。
    expected_run_values = {
        # 记录预先确定的训练轮数。
        "epochs": expected_epochs,
        # 记录每批共同送入模型的语音条数。
        "batch_size": expected_batch_size,
        # 记录并行读取特征的进程数。
        "num_workers": expected_num_workers,
        # 记录复现随机抽样过程使用的固定种子。
        "seed": expected_seed,
        # 记录本次允许使用的 seen/unseen 数据划分协议。
        "expected_protocol": expected_protocol,
    }
    # 检查点值与当前请求不一致的字段。
    mismatches = {
        # 为不一致字段同时保留旧值和本次请求值，报错时能解释具体差异。
        key: {"checkpoint": run_config.get(key), "requested": value}
        # 逐个比较轮数、批大小、工作进程数、种子和数据协议。
        for key, value in expected_run_values.items()
        # 只收集实际不一致的字段，匹配项无需报错。
        if run_config.get(key) != value
    }
    # 按检查点值与当前请求不一致的字段决定是否进入此分支；判断条件为 mismatches。
    if mismatches:
        # 核对训练条件后恢复模型、Adam 和随机状态未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Checkpoint run configuration mismatch: {mismatches}")
    # 检查点绑定的数据指纹。
    saved_fingerprints = run_config.get("data_fingerprints")
    # 按检查点绑定的数据指纹决定是否进入此分支；判断条件为 saved_fingerprints != expected_data_fingerprints。
    if saved_fingerprints != expected_data_fingerprints:
        # 核对训练条件后恢复模型、Adam 和随机状态未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 错误说明当前清单与检查点绑定的数据身份不符。
            "Checkpoint data fingerprints do not match the active manifests, "
            # 进一步列明不一致也可能来自成套 mean/std 或内容、身份特征。
            "atomically published mel statistics, or audited BNF/speaker feature "
            # 提示恢复原始数据状态，或把数据变化作为独立新实验。
            "collections. Restore the original data state or start a new experiment "
            # 续接提示，新实验必须另用目录，保留原结果的可追溯性。
            "directory."
        )
    # 检查点保存的外部 BNF 时间对齐规则。
    saved_alignment = run_config.get(
        # 读取旧检查点外部对齐字段；旧档缺字段时明确按历史长度插值解释。
        "bnf_alignment", LEGACY_ALIGNMENT
    )
    # 按检查点保存的外部 BNF 时间对齐规则决定是否进入此分支；判断条件为 saved_alignment != expected_bnf_alignment。
    if saved_alignment != expected_bnf_alignment:
        # 核对训练条件后恢复模型、Adam 和随机状态未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 续训错误提示明确禁止更换外部 BNF/mel 对齐规则。
            "Cannot resume with a different BNF alignment: checkpoint uses "
            # 续训报错同时列出检查点外部对齐规则与当前请求。
            f"{saved_alignment!r}, requested {expected_bnf_alignment!r}. "
            # 修改外部时轴对齐需要新实验目录，不能覆盖历史运行。
            "Start a new experiment directory for an alignment change."
        )
    # 检查点保存的内部 BNF 下采样规则。
    saved_downsampling = run_config.get(
        # 读取旧检查点内部 BNF 采样字段；缺失字段按历史间隔抽点解释。
        "bnf_downsampling", LEGACY_BNF_DOWNSAMPLING
    )
    # 按检查点保存的内部 BNF 下采样规则决定是否进入此分支；判断条件为 saved_downsampling != expected_bnf_downsampling。
    if saved_downsampling != expected_bnf_downsampling:
        # 核对训练条件后恢复模型、Adam 和随机状态未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 续训错误提示明确禁止更换 U-Net 内部 BNF 采样规则。
            "Cannot resume with a different BNF U-Net downsampling rule: checkpoint "
            # 续训报错同时列出检查点内部 BNF 采样规则与当前请求。
            f"uses {saved_downsampling!r}, requested {expected_bnf_downsampling!r}. "
            # 内部采样对齐改动需另开消融实验目录。
            "Start a new experiment directory for this ablation."
        )
    # epoch_XXXX.pt 是故意不保存 Adam 状态的轻量权重，只能推理/测评。
    # 恢复训练必须用 latest.pt；这里提前给出人能看懂的错误，避免后面出现 KeyError。
    if "optimizer" not in checkpoint:
        # 核对训练条件后恢复模型、Adam 和随机状态未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 指出该文件只有网络权重，缺少 Adam 状态，无法完整续训。
            f"Checkpoint {path} is weights-only and cannot resume Adam. "
            # 提示使用包含 Adam 与随机状态的完整 latest.pt 续训档案。
            "Use exp/checkpoints/voicegrad30_vctk/latest.pt instead."
        )
    # 缺少随机数状态时不能保证严格续训，必须停止而非自动重置。
    if "rng_state" not in checkpoint:
        # 核对训练条件后恢复模型、Adam 和随机状态未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 指出该文件缺少随机流状态，无法保证延续原训练轨迹。
            f"Checkpoint {path} has no RNG state and cannot provide a strict resume."
        )
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # strict=True 要求每个参数名和形状都完全一致，防止只加载一半权重还继续跑。
        model.load_state_dict(checkpoint["model"], strict=True)
        # 把检查点保存的状态完整恢复到对应模型或优化器，确保续作沿用旧状态。
        optimizer.load_state_dict(checkpoint["optimizer"])
    # 捕获核对训练条件后恢复模型、Adam 和随机状态中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except (KeyError, RuntimeError) as error:
        # 核对训练条件后恢复模型、Adam 和随机状态未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Incompatible checkpoint {path}: {error}") from error
    # 网络和优化器加载成功后再恢复采样随机流，准备从下一轮继续。
    restore_rng_state(checkpoint["rng_state"], device)
    # checkpoint 已完成 epoch N，恢复后必须从 N+1 开始，否则会重复训练一轮。
    return int(checkpoint["epoch"]) + 1, int(checkpoint.get("global_step", 0))


# 比较预测噪声与前向加噪时的已知噪声；下面的参数分别限定输入数据和处理约束。
def diffusion_loss(
    # 预测 mel 噪声的 VoiceGrad 网络。
    model: VoiceGrad,
    # 保存扩散日程并负责加噪、去噪的对象。
    diffusion: VoiceGradDiffusion,
    # 同一批 mel、BNF 与身份条件。
    batch: Dict[str, object],
    # 张量执行计算的设备。
    device: torch.device,
# 返回比较预测噪声与前向加噪时的已知噪声得到的张量；类型标注帮助调用方理解结果，不改变数值。
) -> torch.Tensor:
    # non_blocking=True 与 DataLoader pin_memory 配合，可让 CPU->GPU 拷贝更高效。
    mel = batch["mel"].to(device, non_blocking=True)
    # 表示语音内容的 144 维 BNF 特征。
    bnf = batch["bnf"].to(device, non_blocking=True)
    # 当前语音所属说话人。
    speaker = batch["spk_emb"].to(device, non_blocking=True)
    # 论文要求 t 在 {1,...,1000} 均匀抽样。torch.randint 的 high 不包含自身，
    # 所以 high 必须写 n_levels+1 才有机会抽到 1000。
    timesteps = torch.randint(
        # 随机扩散步下界为 1；第 0 步是干净数据，不参加加噪训练采样。
        1,
        # torch.randint 上界不包含端点，因此加 1 才能抽到第 1000 步。
        diffusion.n_levels + 1,
        # 每条批内语音独立抽取一个扩散时间步。
        (mel.shape[0],),
        # 张量执行计算的设备。
        device=device,
        # 张量的数值类型。
        dtype=torch.long,
    )
    # 真实监督答案 epsilon~N(0,I)，形状与干净 mel 一样。
    noise = torch.randn_like(mel)
    # 使用论文公式 (3) 把 x_0 和 epsilon 混合成随机时刻 x_t。
    noisy_mel = diffusion.q_sample(mel, timesteps, noise)
    # 网络根据 x_t、t、目标说话人和 BNF 预测 epsilon_theta。
    predicted_noise = model(noisy_mel, timesteps, speaker, bnf)
    # 论文明确使用 L1 而不是原 DDPM 常见的 L2。L1 是逐元素绝对差再取平均。
    return F.l1_loss(predicted_noise, noise)


# 遍历一轮训练批次并执行反向传播和 Adam 更新；下面的参数分别限定输入数据和处理约束。
def train_epoch(
    # 预测 mel 噪声的 VoiceGrad 网络。
    model: VoiceGrad,
    # 保存扩散日程并负责加噪、去噪的对象。
    diffusion: VoiceGradDiffusion,
    # 逐批返回已对齐特征的数据加载器。
    loader,
    # 更新教师参数并保存动量的 Adam 优化器。
    optimizer: torch.optim.Optimizer,
    # 张量执行计算的设备。
    device: torch.device,
    # 当前完整遍历训练集的轮次。
    epoch: int,
    # 从训练开始累计完成的参数更新次数。
    global_step: int,
# 返回遍历一轮训练批次并执行反向传播和 Adam 更新整理的 tuple[float, int] 结果，内部各项保持前面约定的对应关系。
) -> tuple[float, int]:
    # train() 打开训练模式。当前网络没有 dropout/batchnorm，但仍是规范写法。
    model.train()
    # 当前轮已处理批次的累计损失。
    total_loss = 0.0
    # 当前轮已经处理的批次数。
    batches = 0
    # tqdm 只负责显示进度条；:03d 表示整数至少显示 3 位，如 5 -> 005。
    progress = tqdm(loader, desc=f"train {epoch:03d}", dynamic_ncols=True)
    # 逐批读取训练片段，每批对应一次参数更新。
    for batch in progress:
        # 清掉上一个 batch 的梯度。set_to_none=True 比全部写 0 更省内存。
        optimizer.zero_grad(set_to_none=True)
        # 预测噪声与已知噪声之间的 L1 误差。
        loss = diffusion_loss(model, diffusion, batch, device)
        # 损失出现 NaN 或无穷时立即中止，防止无效梯度破坏后续检查点。
        if not torch.isfinite(loss):
            # 遍历一轮训练批次并执行反向传播和 Adam 更新未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(f"Non-finite training loss at step {global_step}.")
        # backward 根据计算图用链式法则求每个可学习参数的梯度。
        loss.backward()
        # optimizer.step 根据 Adam 规则和固定 lr=2e-4 真正修改模型参数。
        optimizer.step()
        # detach 从计算图分离；float 转成普通 Python 数字，便于累计和打印。
        value = float(loss.detach())
        # 累计每批的标量 L1，供轮末计算平均误差。
        total_loss += value
        # 累计本轮已经处理的批次数，作为平均损失分母。
        batches += 1
        # 一次优化器更新完成后，累计训练步数增加 1。
        global_step += 1
        # 在服务器进度条显示当前批的 L1，保留五位小数。
        progress.set_postfix(loss=f"{value:.5f}")
    # 返回本轮平均训练误差和更新后的累计步数。
    return total_loss / batches, global_step


# 关闭这段评估或采样的梯度记录，避免把诊断过程当成训练更新。
@torch.no_grad()
# 使用固定噪声评估教师误差而不更新参数；下面的参数分别限定输入数据和处理约束。
def evaluate_test_loss(
    # 预测 mel 噪声的 VoiceGrad 网络。
    model: VoiceGrad,
    # 保存扩散日程并负责加噪、去噪的对象。
    diffusion: VoiceGradDiffusion,
    # 逐批返回已对齐特征的数据加载器。
    loader,
    # 张量执行计算的设备。
    device: torch.device,
    # 测试噪声的固定种子，便于不同轮比较。
    evaluation_seed: int,
# 返回使用固定噪声评估教师误差而不更新参数整理的 float 结果，内部各项保持前面约定的对应关系。
) -> float:
    # test 只观察 loss，不做 backward/optimizer.step；@no_grad 可节省大量显存。
    model.eval()
    # 当前轮已处理批次的累计损失。
    total_loss = 0.0
    # 当前轮已经处理的批次数。
    batches = 0
    # diffusion_loss 会随机抽 timestep 和高斯噪声。若每个 epoch 换一套随机“考题”，
    # test 曲线波动会同时包含模型变化和考题变化，epoch 间就不够公平。
    # fork_rng 先保存训练 RNG 状态，测试结束再原样恢复：固定测试题的同时，测试也
    # 不会消耗下一轮训练原本要使用的随机数，因此不会暗中改变训练轨迹。
    cuda_devices = []
    # 当前运行使用 GPU 时，还要处理该设备的随机数状态。
    if device.type == "cuda":
        # 随机数隔离时需要保存和恢复的 GPU 编号。
        cuda_devices = [
            # 优先使用显式 GPU 编号；只写 cuda 时使用当前设备编号。
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    # 隔离测试随机流，测试结束后恢复训练随机状态，避免测试改变后续训练噪声。
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        # 每次评估重设同一个 seed；test 清单也不 shuffle，所以 t/noise/语音顺序都固定。
        torch.manual_seed(evaluation_seed)
        # 逐条或逐批计算未见说话人的固定噪声预测诊断，不执行参数更新。
        for batch in tqdm(loader, desc="unseen test", leave=False, dynamic_ncols=True):
            # 每条完整语音算一次 L1，再对 100 条语音等权平均。
            total_loss += float(diffusion_loss(model, diffusion, batch, device))
            # 累计本轮已经处理的批次数，作为平均损失分母。
            batches += 1
    # 按当前轮已经处理的批次数决定是否进入此分支；判断条件为 batches == 0。
    if batches == 0:
        # 空测试集没有可定义的平均 loss；主动报错比稍后出现“除以 0”更容易理解。
        raise RuntimeError("The unseen-speaker test loader is empty.")
    # 返回所有测试批次的平均噪声预测 L1。
    return total_loss / batches


# 向损失 CSV 追加当前轮训练与测试记录；下面的参数分别限定输入数据和处理约束。
def append_log(path: Path, epoch: int, train_loss: float, test_loss: float) -> None:
    # 以追加模式 "a" 写 loss.csv。文件首次创建时先写表头，以后不重复表头。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 输出 CSV 尚不存在、需要先写列名的标志。
    write_header = not path.exists()
    # 在受控文件句柄中完成向损失 CSV 追加当前轮训练与测试记录，退出此段后自动关闭文件。
    with path.open("a", encoding="utf-8", newline="") as handle:
        # 固定表头顺序的 CSV 写入器。
        writer = csv.writer(handle)
        # 按输出 CSV 尚不存在、需要先写列名的标志决定是否进入此分支；判断条件为 write_header。
        if write_header:
            # 把已整理的清单记录按固定字段顺序写入 CSV。
            writer.writerow(["epoch", "train_l1", "test_l1"])
        # 两列使用同一 epoch 写在同一行，不存在一列有点、另一列没有点的情况。
        writer.writerow([epoch, f"{train_loss:.8f}", f"{test_loss:.8f}"])


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main() -> None:
    # main 是程序入口，负责按正确顺序组装前面所有独立函数。
    args = parse_args()
    # 这些参数会出现在取模、range 或 DataLoader 中，必须是正整数；提前给出清楚错误。
    if args.epochs < 1:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError("--epochs must be at least 1.")
    # 按记录训练与测试损失的轮次间隔决定是否进入此分支；判断条件为 args.loss_every < 1。
    if args.loss_every < 1:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError("--loss-every must be at least 1.")
    # 按另存权重快照的轮次间隔决定是否进入此分支；判断条件为 args.save_every < 1。
    if args.save_every < 1:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError("--save-every must be at least 1.")
    # 本模块是固定的论文教师入口，而不是任意超参数搜索器。即使绕过 shell
    # wrapper 直接调用 Python，也不能把主实验静默改成另一套训练。
    if args.epochs != PAPER_CONFIG.paper_epochs:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 错误说明教师正式运行的终点固定为 500 个 epoch。
            f"paper teacher requires exactly {PAPER_CONFIG.paper_epochs} epochs; "
            # 输出命令实际请求的总轮数，便于与固定终点比较。
            f"got {args.epochs}"
        )
    # 按每批共同送入模型的语音条数决定是否进入此分支；判断条件为 args.batch_size != PAPER_CONFIG.batch_size。
    if args.batch_size != PAPER_CONFIG.batch_size:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 错误说明教师正式训练采用固定批大小 32。
            f"paper teacher requires batch size {PAPER_CONFIG.batch_size}; "
            # 输出命令实际请求的批大小。
            f"got {args.batch_size}"
        )
    # 按复现随机抽样过程使用的固定种子决定是否进入此分支；判断条件为 args.seed != PAPER_CONFIG.seed。
    if args.seed != PAPER_CONFIG.seed:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 错误比较本次重建固定种子 1234 与实际请求；种子属于实现可复现选择。
            f"audited reconstruction fixes seed={PAPER_CONFIG.seed}; got {args.seed}"
        )
    # 按外部 BNF 到 mel 的时间对齐规则决定是否进入此分支；判断条件为 args.bnf_alignment != LEGACY_ALIGNMENT。
    if args.bnf_alignment != LEGACY_ALIGNMENT:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 错误提醒该教师基线固定沿用历史外部长度插值。
            f"audited teacher fixes BNF alignment {LEGACY_ALIGNMENT!r}; "
            # 输出当前命令请求的外部对齐方式。
            f"got {args.bnf_alignment!r}"
        )
    # 按网络各时间分辨率上的 BNF 取样规则决定是否进入此分支；判断条件为 args.bnf_downsampling != LEGACY_BNF_DOWNSAMPLING。
    if args.bnf_downsampling != LEGACY_BNF_DOWNSAMPLING:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 提示本次已审核教师保持原内部 BNF 下采样规则。
            "audited teacher fixes BNF U-Net downsampling "
            # 比较历史内部抽帧规则与当前请求。
            f"{LEGACY_BNF_DOWNSAMPLING!r}; got {args.bnf_downsampling!r}"
        )
    # 按记录训练与测试损失的轮次间隔决定是否进入此分支；判断条件为 args.loss_every != PAPER_CONFIG.paper_epochs。
    if args.loss_every != PAPER_CONFIG.paper_epochs:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 提示测试集只在预先确定的 500 轮终点诊断，避免依测试表现选停止点。
            "the held-out test set is evaluated only at the fixed 500-epoch endpoint; "
            # 提示当前入口要求仅在第 500 轮记录测试诊断，并给出实际错误间隔。
            f"set --loss-every {PAPER_CONFIG.paper_epochs}, got {args.loss_every}"
        )
    # 训练开始前统一设置 Python、NumPy 和 PyTorch 随机种子。
    seed_everything(args.seed)
    # 张量执行计算的设备。
    device = torch.device(args.device)
    # 显式请求 GPU 却没有可用 CUDA 时直接报错，避免不知情地转为慢速 CPU 训练。
    if device.type == "cuda" and not torch.cuda.is_available():
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("CUDA was requested but is not available.")

    # resolve 转成规范绝对路径。data 和 exp 都相对于服务器工程根目录构造。
    project_root = args.project_root.resolve()
    # 正式数据划分协议记录路径。
    protocol_path = project_root / "data" / "metadata" / "split_protocol.json"
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 在受控文件句柄中完成串联本文件的数据检查、模型处理和结果保存流程，退出此段后自动关闭文件。
        with protocol_path.open("r", encoding="utf-8") as handle:
            # 当前磁盘数据清单声明的协议名称。
            active_protocol = str(
                # 从划分报告读取机器可核对的协议名，缺字段时返回空串并触发后续失败。
                json.load(handle).get("protocol_cli_name", "")
            # 去掉协议名两侧空白，随后进行严格名称比较。
            ).strip()
    # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as error:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Cannot read active split protocol {protocol_path}: {error}") from error
    # 按当前磁盘数据清单声明的协议名称决定是否进入此分支；判断条件为 active_protocol != args.expected_protocol。
    if active_protocol != args.expected_protocol:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 指出当前数据划分与训练请求的 seen/unseen 协议不一致。
            "Active split protocol does not match this training run: "
            # 同时输出期望协议和磁盘真实协议，便于修正路径或划分。
            f"expected {args.expected_protocol!r}, got {active_protocol!r}."
        )
    # 需要登记内容身份的依赖文件。
    fingerprint_paths = {
        # 登记决定参数学习样本的训练清单。
        "train_manifest.csv": project_root / "data" / "metadata" / "train_manifest.csv",
        # 登记只供固定终点评估的测试清单。
        "test_manifest.csv": project_root / "data" / "metadata" / "test_manifest.csv",
        # 登记特征提取应完整覆盖的总清单。
        "all_manifest.csv": project_root / "data" / "metadata" / "all_manifest.csv",
        # 登记实际标准化使用的逐频带均值文件。
        "mel_mean.npy": project_root / "data" / "stats" / "mel_mean.npy",
        # 登记与均值同版本的逐频带标准差文件。
        "mel_std.npy": project_root / "data" / "stats" / "mel_std.npy",
    }
    # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
    try:
        # 绑定数据清单、标准化统计量和特征来源的摘要。
        args.data_fingerprints = {
            # 为每份数据依赖计算文件内容摘要，随检查点锁定输入版本。
            name: sha256_file(path) for name, path in fingerprint_paths.items()
        }
        # 单独对协议含义计算指纹，不让无关报告时间字段改变数据身份。
        args.data_fingerprints["split_protocol.semantic_sha256"] = hashlib.sha256(
            # 把协议字段序列化为稳定文字，以便不同运行生成相同摘要。
            json.dumps(
                # 指纹只绑定实际 seen/unseen 协议名称。
                {"protocol_cli_name": active_protocol},
                # 固定 JSON 字段顺序，避免字典排列改变摘要。
                sort_keys=True,
                # 去掉不必要空格，采用固定 JSON 字节形式。
                separators=(",", ":"),
            # 将规范化 JSON 转成 UTF-8 字节供 SHA-256 处理。
            ).encode("utf-8")
        # 将协议内容摘要转换为可保存、可比较的十六进制文字。
        ).hexdigest()
        # These audits fail closed on legacy/mixed/missing feature provenance and on
        # an interrupted mel-statistics publication.  They scan every expected path
        # and byte size but defer full feature-content hashing to validate_setup
        # --full, which the official wrapper runs immediately before this process.
        # 读取当前训练清单，用于核对 mean/std 确实只基于这一训练集合。
        train_rows = read_manifest(fingerprint_paths["train_manifest.csv"])
        # 训练及测试实际使用的全部记录。
        all_rows = read_manifest(fingerprint_paths["all_manifest.csv"])
        # 把当前审计返回的数据身份摘要并入检查点将保存的依赖指纹。
        args.data_fingerprints.update(
            # 核对 mean/std、不可变版本清单和统计报告是否属于同一次完整发布。
            audit_mel_stats_publication(
                # 保存代码、数据与检查点的项目目录。
                project_root,
                # 允许模型学习的训练语音记录。
                train_rows,
                # 启动时核对来源报告、文件尺寸和已登记指纹，不在此处重新遍历读取所有数值内容。
                check_content=False,
            )
        )
        # 只用于评估的未见说话人语音记录。
        test_rows = read_manifest(fingerprint_paths["test_manifest.csv"])
        # 把当前审计返回的数据身份摘要并入检查点将保存的依赖指纹。
        args.data_fingerprints.update(
            # 核对 mel 提取参数、全量覆盖范围及测试输入的身份。
            audit_mel_extraction_recipe(project_root, all_rows, test_rows)
        )
        # 已核对输入 WAV 信息的缓存，避免跨特征重复读取。
        input_wav_cache: Dict[str, Dict[str, object]] = {}
        # 分别审核内容 BNF 和说话人身份向量的固定提取器来源。
        for feature_kind in ("bnf", "speaker_embedding"):
            # 把当前审计返回的数据身份摘要并入检查点将保存的依赖指纹。
            args.data_fingerprints.update(
                # 审核每种特征的来源台账、文件布局以及所用模型版本。
                audit_feature_provenance(
                    # 保存代码、数据与检查点的项目目录。
                    project_root,
                    # 训练及测试实际使用的全部记录。
                    all_rows,
                    # BNF 或说话人嵌入的特征类别。
                    feature_kind,
                    # 启动时核对来源报告、文件尺寸和已登记指纹，不在此处重新遍历读取所有数值内容。
                    check_content=False,
                    # 已核对输入 WAV 信息的缓存，避免跨特征重复读取。
                    input_wav_cache=input_wav_cache,
                )
            )
    # 捕获串联本文件的数据检查、模型处理和结果保存流程中的读取、格式或兼容错误，让调用者获得明确失败原因。
    except Exception as error:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(
            # 保留数据来源预检失败的具体原因，训练不能绕过此检查。
            f"Cannot verify/fingerprint active training data provenance: {error}"
        # 保留预检失败的底层异常链，帮助区分清单、统计量或特征来源问题。
        ) from error
    # 预处理数据根目录。
    data_root = project_root / "data"
    # 按本次运行的输出目录决定是否进入此分支；判断条件为 args.output_dir is None。
    if args.output_dir is None:
        # 本次运行的输出目录。
        output_dir = project_root / "exp" / "checkpoints" / "voicegrad30_vctk"
    # 按本次运行的输出目录决定是否进入此分支；判断条件为 args.output_dir.is_absolute()。
    elif args.output_dir.is_absolute():
        # 本次运行的输出目录。
        output_dir = args.output_dir.resolve()
    # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
    else:
        # 本次运行的输出目录。
        output_dir = (project_root / args.output_dir).resolve()
    # 不带 --resume 却指向已有实验时拒绝继续，防止把新 loss 追加到旧曲线、把旧 latest
    # 覆盖掉。真正断点续训必须显式写 --resume .../latest.pt，意图才足够明确。
    if args.resume is None and output_dir.exists():
        # 旧实验已生成的检查点和损失记录。
        existing_outputs = list(output_dir.glob("epoch_*.pt"))
        # 除轮次快照外，还检查可续训档案和旧损失记录。
        existing_outputs += [
            # 只收集实际存在的旧文件，用于阻止新实验覆盖已有运行。
            path for path in (output_dir / "latest.pt", output_dir / "loss.csv") if path.exists()
        ]
        # 按旧实验已生成的检查点和损失记录决定是否进入此分支；判断条件为 existing_outputs。
        if existing_outputs:
            # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
            raise RuntimeError(
                # 指出目标目录已有实验产物，禁止将新训练混入旧记录。
                f"Refusing to mix a new run with existing outputs in {output_dir}. "
                # 提示已有实验只能明确续训，或使用空目录建立新实验。
                "Use --resume with latest.pt or choose an empty experiment directory."
            )
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(json.dumps(PAPER_CONFIG.as_dict(), indent=2))
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"project_root={project_root}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"device={device}")

    # 创建训练集：严格要求 100 个 seen speakers，每条随机裁 128 帧，batch=32。
    train_loader = get_dataloader(
        # 预处理数据根目录。
        data_root,
        # 加载只用于参数学习的训练集合，启用训练裁剪和打乱规则。
        "train",
        # 每批共同送入模型的语音条数。
        batch_size=args.batch_size,
        # 并行读取特征的进程数。
        num_workers=args.num_workers,
        # 每个训练片段的 mel 帧数。
        segment_length=PAPER_CONFIG.train_frames,
        # 当前集合应包含的说话人数。
        expected_speakers=PAPER_CONFIG.expected_train_speakers,
        # 外部 BNF 到 mel 的时间对齐规则。
        bnf_alignment=args.bnf_alignment,
    )
    # 工程没有 validation。test loader 读取 10 位训练中从未出现的说话人，每人 10 句。
    # 非训练 split 会自动使用完整语音、batch=1、shuffle=False，保证每次顺序一致。
    test_loader = get_dataloader(
        # 预处理数据根目录。
        data_root,
        # 加载固定未见测试集合，不作为参数学习数据。
        "test",
        # 每批共同送入模型的语音条数。
        batch_size=1,
        # 并行读取特征的进程数。
        num_workers=args.num_workers,
        # 当前集合应包含的说话人数。
        expected_speakers=PAPER_CONFIG.expected_test_speakers,
        # 外部 BNF 到 mel 的时间对齐规则。
        bnf_alignment=args.bnf_alignment,
    )

    # .to(device) 把网络参数和扩散 buffer 搬到 CUDA GPU 或 CPU。
    model = build_model(args.bnf_downsampling).to(device)
    # 保存扩散日程并负责加噪、去噪的对象。
    diffusion = VoiceGradDiffusion(
        # 按训练固定的 1000 步余弦日程和偏移重建扩散系数。
        PAPER_CONFIG.diffusion_steps, PAPER_CONFIG.cosine_offset
    # 将扩散系数缓冲量一起放到训练设备。
    ).to(device)
    # 论文教师优化器：固定 lr=0.0002，betas=(0.9,0.999)，没有 scheduler。
    optimizer = torch.optim.Adam(
        # Adam 只更新 VoiceGrad 网络的可训练参数。
        model.parameters(),
        # 使用固定学习率 0.0002，不在本入口自动调度。
        lr=PAPER_CONFIG.learning_rate,
        # 每一步新增噪声比例表。
        betas=(PAPER_CONFIG.adam_beta1, PAPER_CONFIG.adam_beta2),
    )

    # 普通新训练从 epoch 1/step 0 开始；--resume 才会被 checkpoint 覆盖。
    start_epoch, global_step = 1, 0
    # 只有明确提供检查点时才进入严格续训，不自动猜测旧实验。
    if args.resume is not None:
        # 恢复完整状态并取得下一轮编号与累计训练步数。
        start_epoch, global_step = load_checkpoint(
            # 使用用户显式指定的完整续训检查点。
            args.resume,
            # 预测 mel 噪声的 VoiceGrad 网络。
            model,
            # 更新教师参数并保存动量的 Adam 优化器。
            optimizer,
            # 张量执行计算的设备。
            device,
            # 本次允许使用的 seen/unseen 数据划分协议。
            expected_protocol=args.expected_protocol,
            # 严格续训要求的总训练轮数。
            expected_epochs=args.epochs,
            # 严格续训要求的批大小。
            expected_batch_size=args.batch_size,
            # 严格续训要求的数据加载进程数。
            expected_num_workers=args.num_workers,
            # 严格续训要求的随机种子。
            expected_seed=args.seed,
            # 当前数据清单、统计量及特征来源的审核指纹。
            expected_data_fingerprints=args.data_fingerprints,
            # 本次运行要求的外部 BNF 对齐方式。
            expected_bnf_alignment=args.bnf_alignment,
            # 本次运行要求的网络内部 BNF 对齐方式。
            expected_bnf_downsampling=args.bnf_downsampling,
        )
        # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
        print(f"Resumed {args.resume} at epoch {start_epoch}, step {global_step}.")

    # 逐轮训练结果的记录文件。
    log_path = output_dir / "loss.csv"
    # range 的右端不包含，所以写 args.epochs+1 才会真正训练到第 500 epoch。
    for epoch in range(start_epoch, args.epochs + 1):
        # 执行一轮训练，接收本轮平均误差及新的累计步数。
        train_loss, global_step = train_epoch(
            # 将当前网络、固定扩散日程、训练数据和优化器状态共同交给训练循环。
            model, diffusion, train_loader, optimizer, device, epoch, global_step
        )
        # 每一轮训练 loss 都会在终端打印；只有到统一采样 epoch 才同时计算并记录 test。
        if epoch % args.loss_every == 0:
            # 加上一个固定大偏移只是把测试随机序列与训练 seed 区分开；每个 epoch 仍相同。
            test_loss = evaluate_test_loss(
                # 预测 mel 噪声的 VoiceGrad 网络。
                model,
                # 保存扩散日程并负责加噪、去噪的对象。
                diffusion,
                # 按完整语句读取未见说话人数据的测试加载器。
                test_loader,
                # 张量执行计算的设备。
                device,
                # 测试噪声的固定种子，便于不同轮比较。
                evaluation_seed=args.seed + 10_000_000,
            )
            # 同一行保存同一轮训练和测试误差，防止曲线取点错配。
            append_log(log_path, epoch, train_loss, test_loss)
            # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
            print(
                # 日志打印本轮编号和平均训练 L1，保留六位小数。
                f"epoch={epoch:04d} train_l1={train_loss:.6f} "
                # 同一条日志追加相同轮次的固定测试 L1。
                f"test_l1={test_loss:.6f}"
            )
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 例如 --loss-every 10 时，第 1~9 轮只打印训练结果，不往曲线 CSV 写点。
            print(f"epoch={epoch:04d} train_l1={train_loss:.6f} test_l1=not-scheduled")

        # latest.pt 每轮覆盖，方便断点续训；阶段 checkpoint 每 100 轮永久保留，
        # 每 100 epoch 保存一次，并且一定包含论文终点 epoch_0500。
        save_checkpoint(
            # 每轮更新 latest.pt，保存网络、Adam 和随机流的同一时点状态。
            output_dir / "latest.pt", model, optimizer, epoch, global_step, args
        )
        # 按当前完整遍历训练集的轮次决定是否进入此分支；判断条件为 epoch % args.save_every == 0 or epoch == args.epochs。
        if epoch % args.save_every == 0 or epoch == args.epochs:
            # 保存当前轮检查点；是否带优化器由下方 include_optimizer 参数区分。
            save_checkpoint(
                # 用四位轮次编号另存可用于推理的权重快照，便于历史结果追溯。
                output_dir / f"epoch_{epoch:04d}.pt",
                # 预测 mel 噪声的 VoiceGrad 网络。
                model,
                # 更新教师参数并保存动量的 Adam 优化器。
                optimizer,
                # 当前完整遍历训练集的轮次。
                epoch,
                # 从训练开始累计完成的参数更新次数。
                global_step,
                # 终端传入并经解析的运行设置。
                args,
                # 阶段权重只用于推理/测评，不重复保存体积很大的 Adam 状态。
                include_optimizer=False,
            )


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 直接执行 python train.py 时调用 main；其他文件 import build_model 时不启动训练。
    main()
