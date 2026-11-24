# 用训练好的教师权重转换一条预处理语音，输出未标准化的自然对数 mel。
"""Convert one preprocessed utterance with a trained VoiceGrad-30 checkpoint."""

# =============================================================================
# 推理（voice conversion）到底在做什么？
# =============================================================================
# 训练结束后，这个脚本用一条 source 语音提供“说什么”，用另一条 reference 语音
# 提供“像谁说”，生成转换后的 log-mel：
#   source mel  -> 提供源语音的整体初始状态；
#   source BNF  -> 提供源语音的音素内容；
#   reference speaker embedding -> 提供目标音色；
#   checkpoint -> 提供训练好的噪声预测网络；
#   diffusion.sample -> 进行 VoiceGrad-30 的 30 次反向更新；
#   输出 .npy -> 未归一化的 80 维 log-mel，之后才交给 HiFi-GAN 变成 wav。
#
# 这个脚本不会训练模型，不会调用 optimizer，也不会改 checkpoint。

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# --------------------------- 导入包逐项解释 ---------------------------
# argparse 读取推理命令参数；Path 负责拼 source/reference/output 路径。
import argparse
# 用路径对象定位数据、权重和报告，减少手工拼接目录的错误。
from pathlib import Path

# NumPy 读取/保存 .npy 并做反归一化；torch 建模型、加载权重并运行 GPU 推理。
import numpy as np
# 使用 PyTorch 张量表示声学特征、扩散状态及网络参数。
import torch
# F.interpolate 只用于把 BNF 时间长度对齐到 source mel。
import torch.nn.functional as F

# PAPER_CONFIG 保证推理参数与训练一致；VoiceGradDiffusion 执行 30 步更新；
# build_model 使用 train.py 中同一入口建立完全相同的网络结构。
from .config import PAPER_CONFIG, SERVER_PROJECT_ROOT
# 复用 VoiceGradDiffusion 的统一实现，具体输入和实验约束由调用处指定。
from .diffusion import VoiceGradDiffusion
# 复用 build_model 的统一实现，具体输入和实验约束由调用处指定。
from .train import build_model


# 根据说话人和句子编号定位对应的预提取特征；下面的参数分别限定输入数据和处理约束。
def feature_path(data: Path, feature: str, item_id: str) -> Path:
    # 命令行用 p225/p225_001 这种紧凑格式指定语音。
    # replace 把 Windows 反斜杠统一成 /，split 再拆成 [speaker, utterance]。
    parts = item_id.replace("\\", "/").split("/")
    # 要求语音编号恰好由非空说话人目录和句子编号两段构成。
    if len(parts) != 2 or not all(parts):
        # 根据说话人和句子编号定位对应的预提取特征未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError("Item IDs must have form speaker/utterance, e.g. p225/p225_001.")
    # 根据 feature="mel"/"bnf"/"spk_emb" 拼出标准工程路径。
    return data / feature / parts[0] / f"{parts[1]}.npy"


# 读取并整理输入特征矩阵的通道与时间轴；下面的参数分别限定输入数据和处理约束。
def matrix(path: Path, dimension: int) -> np.ndarray:
    # 读取二维特征，并统一为 [feature_dim, time]。
    # 有些工具保存 [T,80]，有些保存 [80,T]；这里判断哪一轴等于 dimension。
    value = np.load(path, allow_pickle=False)
    # mel 和 BNF 都应为通道×时间矩阵；拒绝标量、向量或更高维输入。
    if value.ndim != 2:
        # 读取并整理输入特征矩阵的通道与时间轴未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Expected a matrix at {path}, got {value.shape}.")
    # 若第一轴已经是 80 或 144 个特征通道，就保留原方向。
    if value.shape[0] == dimension:
        # 保留通道优先排列，并转换成模型使用的单精度浮点数。
        return value.astype(np.float32)
    # 若通道在第二轴，下面先转置成通道优先排列。
    if value.shape[1] == dimension:
        # .T 是二维转置：行列互换，[T,80] -> [80,T]。
        return value.T.astype(np.float32)
    # 读取并整理输入特征矩阵的通道与时间轴未通过，抛出下方包含具体路径或实际值的错误。
    raise ValueError(f"Expected feature dimension {dimension} at {path}, got {value.shape}.")


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main() -> None:
    # argparse 把终端参数转换成 args.xxx。required=True 的参数漏写会直接给出帮助。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--project-root", type=Path, default=SERVER_PROJECT_ROOT)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--checkpoint", type=Path, required=True)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--source", required=True, help="speaker/utterance used for mel and BNF")
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--reference", required=True, help="speaker/utterance used for target speaker embedding")
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--output", type=Path, required=True, help="output unnormalised log-mel .npy")
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--steps", type=int, default=PAPER_CONFIG.teacher_sampling_steps)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--start-step", type=int, default=PAPER_CONFIG.teacher_start_step)
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--seed", type=int, default=PAPER_CONFIG.seed)
    # FastVoiceGrad 后续实验采用公式 (9) 的 x_src_SK；clean 只保留给原 VoiceGrad 对照。
    parser.add_argument(
        # 选择教师从加噪源 mel 还是干净源 mel 开始反向转换。
        "--initial-state",
        # 只允许源 mel 先加噪或直接使用干净源 mel 两种起点。
        choices=("diffused", "clean"),
        # 默认按论文后续实验使用加噪源起点；干净源保留作对照。
        default="diffused",
        # 帮助文字说明默认起点依据，并提醒 clean 属于额外对照实验。
        help="Use paper subsequent-experiment x_src_SK by default; clean is ablation only.",
    )
    # 登记下列终端选项；默认值及允许范围随后明确列出。
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 终端传入并经解析的运行设置。
    args = parser.parse_args()

    # torch.device 把 "cuda"/"cpu" 字符串变成 PyTorch 设备对象。
    device = torch.device(args.device)
    # resolve 得到绝对工程路径，特征固定从其 data 子目录读取。
    data = args.project_root.resolve() / "data"
    # source 同时提供 mel 和 BNF；reference 只提供目标 speaker embedding。
    mel = matrix(feature_path(data, "mel", args.source), PAPER_CONFIG.n_mels)
    # 表示语音内容的 144 维 BNF 特征。
    bnf = matrix(feature_path(data, "bnf", args.source), PAPER_CONFIG.n_bnf)
    # squeeze 把 [1,256] 或 [256,1] 整理为 [256]；形状错误会立即停止。
    speaker = np.load(feature_path(data, "spk_emb", args.reference), allow_pickle=False).squeeze()
    # 目标身份必须是 256 维单个向量，防止把时间序列或错误模型输出当身份。
    if speaker.shape != (PAPER_CONFIG.spk_emb_dim,):
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(f"Invalid target speaker embedding shape: {speaker.shape}")

    # 模型在标准化空间训练，所以 source mel 推理前必须使用同一训练集 mean/std。
    # reshape(-1,1) 把 [80] 变成 [80,1]，可广播到所有时间帧。
    mean = np.load(data / "stats" / "mel_mean.npy", allow_pickle=False).reshape(-1, 1)
    # 加载与训练相同的 80 维标准差，增加时间轴用于逐频带标准化。
    std = np.load(data / "stats" / "mel_std.npy", allow_pickle=False).reshape(-1, 1)
    # NumPy [80,T] -> float32 -> PyTorch -> 增加 batch 轴 [1,80,T] -> 搬到 GPU。
    mel_tensor = torch.from_numpy(((mel - mean) / std).astype(np.float32)).unsqueeze(0).to(device)
    # 给源 BNF 加上大小为 1 的批次轴，形成 [1,144,T_bnf] 后放到推理设备。
    bnf_tensor = torch.from_numpy(bnf).unsqueeze(0).to(device)
    # 当 BNF 与 mel 总帧数不同时才进行旧版长度比例插值。
    if bnf_tensor.shape[-1] != mel_tensor.shape[-1]:
        # BNF 与 mel 帧率不同，用线性插值把 BNF 时间轴对齐到 mel。
        bnf_tensor = F.interpolate(
            # 仅沿时间轴把 BNF 调整为源 mel 帧数；本入口沿用历史长度对齐，不显式校正物理窗口中心。
            bnf_tensor, size=mel_tensor.shape[-1], mode="linear", align_corners=False
        )
    # 把目标参考身份变为 [1,256] 单精度张量，与源内容分开送入网络。
    speaker_tensor = torch.from_numpy(speaker.astype(np.float32)).unsqueeze(0).to(device)

    # map_location 确保 checkpoint 被加载到当前设备，而不是保存它时的 GPU 编号。
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # 读取检查点保存的教师配置，先检查是否属于当前 1000 步扩散版本。
    saved_config = checkpoint.get("paper_config", {})
    # 拒绝扩散总步数与当前教师不一致的检查点。
    if saved_config.get("diffusion_steps") != PAPER_CONFIG.diffusion_steps:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError("Checkpoint is not the corrected 1000-step VoiceGrad teacher.")
    # 先用和训练完全相同的 config 建空模型，再严格灌入 checkpoint 权重。
    model = build_model().to(device)
    # 严格加载全部网络参数，缺键、多键或形状不符都会停止推理。
    model.load_state_dict(checkpoint["model"], strict=True)
    # eval 切换推理模式；本模型虽无 dropout/batchnorm，仍应遵循规范。
    model.eval()
    # 保存扩散日程并负责加噪、去噪的对象。
    diffusion = VoiceGradDiffusion(
        # 用教师训练的 1000 步余弦日程和 0.008 偏移重建扩散系数。
        PAPER_CONFIG.diffusion_steps, PAPER_CONFIG.cosine_offset
    # 把扩散系数一起移到模型所用设备，避免 CPU/GPU 混用。
    ).to(device)
    # 独立随机数生成器固定中间步骤的随机 z，同样输入和 seed 可得到同样结果。
    generator = torch.Generator(device=device).manual_seed(args.seed)
    # 这里真正执行 VoiceGrad-30：从 source mel 开始，使用 source BNF+target speaker。
    converted = diffusion.sample(
        # 预测 mel 噪声的 VoiceGrad 网络。
        model,
        # 源语音的标准化 mel，提供转换起始状态。
        mel_tensor,
        # 目标参考语音的身份条件，与源说话人身份分开。
        speaker_tensor,
        # 源语音的内容条件，负责保持原句表达内容。
        bnf_tensor,
        # 实际执行的反向采样步数。
        num_steps=args.steps,
        # 开始反向转换的扩散时间步。
        start_step=args.start_step,
        # 单独控制采样噪声的随机数生成器。
        generator=generator,
        # 是否按论文公式先将源 mel 加噪。
        diffuse_source=args.initial_state == "diffused",
    )
    # 去 batch 轴、搬回 CPU、转 NumPy，然后执行反归一化 mel=norm*std+mean。
    converted = converted.squeeze(0).cpu().numpy() * std + mean
    # parents=True 自动创建多层输出目录；exist_ok=True 表示目录已有也不报错。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 保存反标准化后的自然对数 mel；这里只生成特征，尚未合成 WAV。
    np.save(args.output, converted.astype(np.float32), allow_pickle=False)
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"saved {args.output} shape={converted.shape}")


# 仅在直接运行本脚本时进入工作流程；被其他模块导入时不自动启动。
if __name__ == "__main__":
    # 直接执行脚本时进入 main；被别的脚本 import 时不会自动推理。
    main()
