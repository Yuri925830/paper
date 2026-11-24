#!/usr/bin/env python3
# 本脚本只统计已保存的 BNF 分布，不重新提取、不额外归一化，也不训练网络。
"""Audit the saved 144-dimensional BNF scale without changing any features."""

# 延迟解析类型提示，方便在既有 Python 环境中导入这些函数。
from __future__ import annotations

# 命令行指定抽样数量、种子与报告路径。
import argparse
# 按表头读取训练和测试语音清单。
import csv
# 给清单计算指纹，记录统计对应的准确样本来源。
import hashlib
# 用 JSON 保存每个通道和总体统计量。
import json
# os.replace 用于将完整临时报告原子发布为正式报告。
import os
# 调整导入路径，使直接执行 tools 下脚本也能找到项目包。
import sys
# Path 安全拼接清单、特征和输出路径。
from pathlib import Path
# 类型提示说明样本行是字符串字典，多个样本组成序列。
from typing import Dict, List, Sequence

# NumPy 读取 .npy 特征并计算均值、方差、分位数。
import numpy as np


# 从脚本位置向上两级得到项目根目录，不依赖终端当前目录。
PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[1]
# 仅在尚未登记项目路径时添加，避免重复导入路径。
if str(PROJECT_SOURCE_ROOT) not in sys.path:
    # 将当前项目放到最前，优先读取这份仓库内的实现。
    sys.path.insert(0, str(PROJECT_SOURCE_ROOT))

# 复用项目登记的租用服务器根目录。
from voicegrad.config import SERVER_PROJECT_ROOT


# 默认路径跟随统一配置，不另外猜测实验室旧路径。
PROJECT_ROOT = SERVER_PROJECT_ROOT


# 解析只读统计所需的参数，返回具名参数对象。
def parse_args() -> argparse.Namespace:
    # 帮助文字采用文件开头的用途说明。
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据从这个项目目录下的 data 读取。
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    # 默认随机抽取 1000 条训练语音；测试清单则全部统计。
    parser.add_argument("--train-samples", type=int, default=1000)
    # 固定种子使抽样结果可重复，便于比较两次审计。
    parser.add_argument("--seed", type=int, default=1234)
    # 允许指定报告文件，但不会因此改写任何 BNF 特征。
    parser.add_argument(
        # --output 是命令行中选择报告目的文件的选项。
        "--output",
        # 将路径文字直接转成 Path，后面无需再手工拼接字符串。
        type=Path,
        # 默认与其他检查日志保存在同一目录。
        default=PROJECT_ROOT / "exp" / "logs" / "bnf_normalization_audit.json",
    )
    # 根据实际命令行得到本次抽样参数。
    return parser.parse_args()


# 按原始字节计算清单指纹；相同指纹说明清单没有被替换。
def sha256_file(path: Path) -> str:
    # 初始化累计摘要，不把大文件一次性装入内存。
    digest = hashlib.sha256()
    # 二进制读取避免换行和文字编码转换影响指纹。
    with path.open("rb") as handle:
        # 每次最多读取 1 MiB，读到空字节串表示文件结束。
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            # 将当前块加入累计摘要。
            digest.update(block)
    # 返回便于写入 JSON 和核对的十六进制字符串。
    return digest.hexdigest()


# 读取清单，保证至少有一条语音可供统计。
def read_rows(path: Path) -> List[Dict[str, str]]:
    # newline="" 交由 csv 处理行结束符，保留字段内容。
    with path.open("r", encoding="utf-8", newline="") as handle:
        # 按列名创建每条语音的字典，避免依赖列的排列顺序。
        rows = list(csv.DictReader(handle))
    # 空清单没有可用统计含义，不继续计算除以零的平均值。
    if not rows:
        # 报错包含具体清单路径，便于检查数据准备阶段。
        raise ValueError(f"Empty manifest: {path}")
    # 返回全部清单行，训练抽样由主函数统一完成。
    return rows


# 根据说话人和句子编号定位已经保存的 BNF 数组。
def bnf_path(data_dir: Path, row: Dict[str, str]) -> Path:
    # 路径结构与预处理保存规则一致：data/bnf/说话人/句子.npy。
    return data_dir / "bnf" / row["speaker"] / f"{row['utt_id']}.npy"


# 将两种常见存储方向统一为 [144,时间帧]，仅改变内存视图和统计精度。
def load_bnf(path: Path) -> np.ndarray:
    # 禁用 pickle，只接受普通数值数组，避免执行对象反序列化。
    value = np.load(path, allow_pickle=False)
    # BNF 应只有特征维和时间维，额外维度说明文件格式不符合约定。
    if value.ndim != 2:
        # 报出原始形状，避免把不正确的维度强行 reshape 掩盖问题。
        raise ValueError(f"Expected 2-D BNF at {path}, got {value.shape}")
    # 第一个维度已经是 144 时，保持原方向。
    if value.shape[0] == 144:
        # value 的每一行对应一个内容特征通道。
        result = value
    # 若第二维为 144，原文件使用 [时间帧,特征] 排列。
    elif value.shape[1] == 144:
        # 转置后所有统计代码都按通道在前处理。
        result = value.T
    # 两个方向都找不到 144 维，则不属于本实验约定的 BNF。
    else:
        # 拒绝猜测特征含义，提示检查提取器或文件路径。
        raise ValueError(f"No 144-dimensional axis at {path}: {value.shape}")
    # 统计采用 float64 降低大量累加的舍入误差；磁盘特征仍保持原样。
    result = np.asarray(result, dtype=np.float64)
    # 没有时间帧或包含 NaN/Inf 都会让统计失去意义。
    if result.shape[1] == 0 or not np.isfinite(result).all():
        # 发现异常直接报具体文件，不忽略坏样本后继续给出貌似正常的结果。
        raise ValueError(f"Invalid BNF values at {path}")
    # 返回已核对方向和有限性的数组。
    return result


# 对选定语音计算逐通道统计；长句按帧数贡献更多观测值。
def summarize(data_dir: Path, rows: Sequence[Dict[str, str]]) -> dict:
    # 144 个通道分别累计所有时间帧的数值总和。
    channel_sum = np.zeros(144, dtype=np.float64)
    # 同时累计平方和，最后计算总体方差。
    channel_square_sum = np.zeros(144, dtype=np.float64)
    # 从正无穷开始取最小值，第一条语音即可给出有效边界。
    channel_min = np.full(144, np.inf, dtype=np.float64)
    # 从负无穷开始取最大值，避免错误地把零当作初始上界。
    channel_max = np.full(144, -np.inf, dtype=np.float64)
    # 用实际帧数作均值分母，不把每句先平均后等权混合。
    total_frames = 0
    # 分位数只保留每句少量均匀抽取的帧，控制内存占用。
    sampled_values = []
    # 逐帧通道均值可帮助检查提取器末端 LayerNorm 的结果。
    sampled_frame_means = []
    # 逐帧通道标准差与跨语音逐通道标准差含义不同，分别保存。
    sampled_frame_stds = []

    # 逐句统计，不需要一次加载整套 BNF。
    for row in rows:
        # 从清单定位文件，并核对 144 维和有限值。
        value = load_bnf(bnf_path(data_dir, row))
        # 第二维是本句时间帧数。
        frames = value.shape[1]
        # 更新跨语音总帧数，用于后续按帧加权平均。
        total_frames += frames
        # 沿时间轴求和，保留每个通道的独立累计量。
        channel_sum += value.sum(axis=1)
        # 平方和用于 E[x²]−E[x]² 形式的总体方差。
        channel_square_sum += np.square(value).sum(axis=1)
        # 逐通道比较当前最小值与本句最小值。
        channel_min = np.minimum(channel_min, value.min(axis=1))
        # 同样更新逐通道最大值，便于发现极端异常幅度。
        channel_max = np.maximum(channel_max, value.max(axis=1))

        # At most 32 equally spaced frames per utterance keep percentile memory bounded.
        # 每句最多均匀取 32 帧；这些抽样只用于分位数和逐帧统计。
        indices = np.linspace(0, frames - 1, min(frames, 32), dtype=np.int64)
        # 同时取出这些帧的全部 144 个通道。
        sample = value[:, indices]
        # 展平后集中计算取值分位数，避免保存所有原始时间帧。
        sampled_values.append(sample.reshape(-1))
        # 每一帧在 144 个通道上求均值，观察是否接近零。
        sampled_frame_means.append(sample.mean(axis=0))
        # 每一帧在 144 个通道上求标准差，观察是否接近一致尺度。
        sampled_frame_stds.append(sample.std(axis=0))

    # 用全部观测帧计算每个通道的总体均值。
    mean = channel_sum / total_frames
    # 浮点舍入可能产生极小负方差；截到零后再开平方。
    variance = np.maximum(channel_square_sum / total_frames - np.square(mean), 0.0)
    # 标准差与原 BNF 数值具有相同尺度。
    std = np.sqrt(variance)
    # 合并各句抽样取值，供分位数计算。
    values = np.concatenate(sampled_values)
    # 合并抽样帧的跨通道均值。
    frame_means = np.concatenate(sampled_frame_means)
    # 合并抽样帧的跨通道标准差。
    frame_stds = np.concatenate(sampled_frame_stds)
    # 返回完整 144 维统计与便于阅读的总体摘要。
    return {
        # 实际参与这部分统计的句子数。
        "utterances": len(rows),
        # 均值和方差的实际帧数分母。
        "total_frames": int(total_frames),
        # 每个特征通道跨全部时间帧的平均值。
        "channel_mean": mean.tolist(),
        # 每个通道跨全部时间帧的标准差。
        "channel_std": std.tolist(),
        # 每个通道出现过的最小数值。
        "channel_min": channel_min.tolist(),
        # 每个通道出现过的最大数值。
        "channel_max": channel_max.tolist(),
        # 汇总字段方便快速判断是否有整体偏移或幅度异常。
        "summary": {
            # 144 个通道均值中最小的一个。
            "channel_mean_min": float(mean.min()),
            # 144 个通道均值中最大的一个。
            "channel_mean_max": float(mean.max()),
            # 对 144 个通道均值再平均，查看整体中心位置。
            "channel_mean_average": float(mean.mean()),
            # 最不变化的通道，其标准差可能很小。
            "channel_std_min": float(std.min()),
            # 波动最大的通道，便于发现尺度不均衡。
            "channel_std_max": float(std.max()),
            # 所有通道标准差的平均值。
            "channel_std_average": float(std.mean()),
            # 下列分位数来自均匀抽帧值，不能称为所有时间帧的精确分位数。
            "sampled_value_percentiles": {
                # 将分位位置作为字符串键，JSON 中保留相应测量值。
                str(q): float(np.percentile(values, q))
                # 同时查看左右尾部和中位数，避免只看极值被单点影响。
                for q in (0.1, 1.0, 50.0, 99.0, 99.9)
            },
            # 每帧跨通道均值离零的平均距离，反映帧级中心化程度。
            "sampled_per_frame_channel_mean_abs_average": float(
                # 先取绝对值再平均，正负偏移不会相互抵消。
                np.mean(np.abs(frame_means))
            ),
            # 帧级中心偏移最明显的一帧。
            "sampled_per_frame_channel_mean_abs_max": float(
                # 在所有抽样帧中寻找最大绝对均值。
                np.max(np.abs(frame_means))
            ),
            # 平均帧级跨通道标准差；此量不同于上面的逐通道跨时间标准差。
            "sampled_per_frame_channel_std_average": float(frame_stds.mean()),
            # 抽样帧中最小的跨通道标准差。
            "sampled_per_frame_channel_std_min": float(frame_stds.min()),
            # 抽样帧中最大的跨通道标准差。
            "sampled_per_frame_channel_std_max": float(frame_stds.max()),
        },
    }


# 读取清单、抽样统计、比较训练/测试分布，最后单独保存审计报告。
def main() -> None:
    # 获取本次指定的目录、抽样数量及随机种子。
    args = parse_args()
    # 零条或负条训练样本不能作为分布参照。
    if args.train_samples <= 0:
        # 在读数据前给出明确的参数错误。
        raise ValueError("--train-samples must be positive")
    # 规范化根目录，后续所有文件都从这里派生。
    project_root = args.project_root.resolve()
    # 预处理特征与元数据统一位于 data 下。
    data_dir = project_root / "data"
    # 训练清单决定随机抽样的候选语音。
    train_manifest = data_dir / "metadata" / "train_manifest.csv"
    # 测试清单用于完整描述测试 BNF 分布，不用于拟合变换。
    test_manifest = data_dir / "metadata" / "test_manifest.csv"
    # 读取训练样本并确认清单非空。
    train_rows = read_rows(train_manifest)
    # 读取全部测试样本，保持既有划分不变。
    test_rows = read_rows(test_manifest)

    # 使用独立随机数生成器，固定种子确保抽样可重放。
    generator = np.random.default_rng(args.seed)
    # 当请求多于清单总数时，统计全部训练语音。
    count = min(args.train_samples, len(train_rows))
    # 不放回抽样，排序仅为了读取顺序稳定，不改变选中的语音。
    selected_indices = np.sort(generator.choice(len(train_rows), count, replace=False))
    # 根据抽中的索引取得实际清单行。
    selected_train = [train_rows[int(index)] for index in selected_indices]
    # 对训练抽样统计跨帧、跨通道数值。
    train_report = summarize(data_dir, selected_train)
    # 对完整测试清单进行相同统计，保证量的定义一致。
    test_report = summarize(data_dir, test_rows)

    # 把训练均值列表恢复成数组，用于逐通道比较。
    train_mean = np.asarray(train_report["channel_mean"], dtype=np.float64)
    # 训练标准差作为比较分布偏移的尺度基准。
    train_std = np.asarray(train_report["channel_std"], dtype=np.float64)
    # 测试均值只参与描述性比较，不写回训练统计文件。
    test_mean = np.asarray(test_report["channel_mean"], dtype=np.float64)
    # 测试标准差用于检查特征波动是否明显放大或缩小。
    test_std = np.asarray(test_report["channel_std"], dtype=np.float64)
    # 给几乎不变的通道设分母下限，防止除以零。
    safe_train_std = np.maximum(train_std, 1e-12)
    # 均值差除以训练标准差，表示偏移相当于几个训练标准差。
    standardized_mean_shift = (test_mean - train_mean) / safe_train_std
    # 大于 1 表示测试波动较大，小于 1 表示较小。
    std_ratio = test_std / safe_train_std

    # 报告同时保存数据身份、完整测量及解释边界。
    payload = {
        # 标记只读审计，避免误认为已实施了 BNF 归一化修复。
        "status": "read_only_bnf_normalization_audit",
        # 保存抽样种子，方便重新选中同一批训练语音。
        "seed": int(args.seed),
        # 说明训练语音按句均匀无放回抽样，不宣称按说话人均衡抽样。
        "train_sampling": "uniform random utterance sample without replacement",
        # 保存训练清单位置供人工追查。
        "train_manifest": str(train_manifest),
        # 清单字节指纹用于确认后续查看的还是同一份划分。
        "train_manifest_sha256": sha256_file(train_manifest),
        # 保存测试清单路径，明确测试样本来源。
        "test_manifest": str(test_manifest),
        # 测试清单指纹与训练清单分别记录。
        "test_manifest_sha256": sha256_file(test_manifest),
        # 训练抽样的逐通道统计和摘要。
        "train": train_report,
        # 全部测试语音的同定义统计。
        "test": test_report,
        # 仅比较分布，不能用这些结果证明某个额外归一化一定改善性能。
        "test_vs_sampled_train": {
            # 144 维均值偏移的均方根，概括整体偏移幅度。
            "channel_mean_shift_z_rms": float(
                # 使用标准化偏移先平方、平均再开方。
                np.sqrt(np.mean(np.square(standardized_mean_shift)))
            ),
            # 找出均值偏移最严重的通道。
            "channel_mean_shift_z_max_abs": float(
                # 取绝对值避免正负方向影响偏移大小判断。
                np.max(np.abs(standardized_mean_shift))
            ),
            # 测试/训练波动比最小的通道。
            "channel_std_ratio_min": float(std_ratio.min()),
            # 测试/训练波动比最大的通道。
            "channel_std_ratio_max": float(std_ratio.max()),
            # 所有通道波动比的平均值。
            "channel_std_ratio_average": float(std_ratio.mean()),
        },
        # 解释上游已有逐句 CMVN 和帧级 LayerNorm；统计异常不能自动推出再标准化方案。
        "interpretation_guardrail": (
            # 第一段说明 80 维输入前端已有逐句均值方差归一化。
            "The extractor applies utterance-level mean/variance normalization to its "
            # 第二段说明 144 维编码器每帧还经过 LayerNorm。
            "80-bin frontend and final LayerNorm to every 144-D encoder frame. This "
            # 最后强调本报告仅核对现有结果，不构成额外变换的依据。
            "report checks the saved result; it does not justify an extra transform."
        ),
    }
    # 确定本次只读审计的输出报告路径。
    output = args.output.resolve()
    # 报告目录不存在时创建；不动原始特征目录。
    output.parent.mkdir(parents=True, exist_ok=True)
    # 先写相邻临时文件，避免中断留下半份正式 JSON。
    temporary = output.with_suffix(output.suffix + ".tmp")
    # 保留中文并缩进，方便人工查阅所有逐通道统计。
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 临时报告完整写出后才替换正式报告，原 BNF 数组始终只读。
    os.replace(temporary, output)
    # 同时输出到终端，远程日志可直接看到统计结果。
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# 被导入时只提供工具函数，直接执行时才开始统计。
if __name__ == "__main__":
    # 运行上面的读取、抽样、统计和报告流程。
    main()
