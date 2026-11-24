#!/usr/bin/env python3
# 读取同轮配对的 train/test L1 记录并保存损失曲线。
"""把 train.py 生成的 train/test L1 记录画成严格同一 epoch 取点的曲线。"""

# =============================================================================
# 这个脚本解决什么问题？
# =============================================================================
# train.py 会把 loss 写进一个普通 CSV 表格：
#
#     epoch,train_l1,test_l1
#     1,0.81234567,0.82345678
#     2,0.74567890,0.75678901
#
# 每一行同时保存“同一个 epoch”的 train 与 test，绝不会出现 train 在第 10 轮有点、
# test 却在第 7 轮有点。真正决定取点间隔的是 train.py 的唯一参数 --loss-every：
#   --loss-every 1  -> 两条曲线每 1 epoch 一个点；
#   --loss-every 10 -> 两条曲线每 10 epoch 一个点。
# 本脚本不会再次抽样、平均或补点，只把 CSV 里真实存在的点原样画出来。
#
# 请特别注意：test_l1 是“固定随机 timestep/噪声下的未见说话人去噪误差”，它不是
# 论文最终报告的 qMOS/sMOS，也不能代替转换语音的 UTMOS、DNSMOS、CER、SVA。

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# argparse 负责读取命令行里的 --csv 和 --output；csv 负责读取逗号分隔表格。
import argparse
# 按清单表头读取说话人、句子及文件路径，并写回实验索引。
import csv
# math.isfinite 用来拒绝 NaN、正无穷和负无穷，避免画出没有意义的曲线。
import math
# Path 把路径当作对象处理，可检查文件、创建父目录和拼接文件名。
from pathlib import Path
# List/Tuple 是类型提示，只帮助人和编辑器理解数据结构，不会改变运行结果。
from typing import List, Tuple

# matplotlib 默认可能尝试打开服务器图形窗口。服务器通常没有桌面，所以必须先
# 选择 Agg 后端：它只在后台生成 PNG，不需要显示器，也不会卡在 plt.show()。
import matplotlib

# 使用无窗口绘图后端，出租服务器只保存图片文件。
matplotlib.use("Agg")
# pyplot 提供 figure、plot、savefig 等真正的画图函数。
import matplotlib.pyplot as plt

# 从全工程统一配置读取服务器根目录，避免在多个脚本里写不同路径。
from .config import SERVER_PROJECT_ROOT


# 默认输入就是 train.py 的输出位置；用户仍可用 --csv 指向别的实验。
DEFAULT_CSV = (
    # 从当前工程根目录开始定位教师损失表。
    SERVER_PROJECT_ROOT
    # 进入实验产物目录。
    / "exp"
    # 进入模型检查点及其配套日志目录。
    / "checkpoints"
    # 选择教师 text-unseen 实验，不混入学生实验曲线。
    / "voicegrad30_vctk_text_unseen"
    # 最后定位记录 epoch、train_l1、test_l1 的 CSV 文件。
    / "loss.csv"
)


# 说明并解析本脚本的输入目录、实验协议和可选运行参数；下面的参数分别限定输入数据和处理约束。
def parse_args() -> argparse.Namespace:
    # 说明并解析本脚本的输入目录、实验协议和可选运行参数。
    """把命令行文字转换成带属性的 args 对象。"""

    # description 会出现在 python3 plot_loss_curves.py --help 的开头。
    parser = argparse.ArgumentParser(description=__doc__)
    # type=Path 表示 argparse 自动把字符串路径变成 Path；default 是不写参数时的值。
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    # output 默认 None；main 会把它解释成“与 loss.csv 同目录的 loss_curves.png”。
    parser.add_argument("--output", type=Path, default=None)
    # parse_args 真正读取当前终端命令并返回结果，函数到这里结束。
    return parser.parse_args()


# 读取可绘制的损失记录并检查轮次及数值；下面的参数分别限定输入数据和处理约束。
def load_losses(path: Path) -> Tuple[List[int], List[float], List[float]]:
    # 读取可绘制的损失记录并检查轮次及数值。
    """读取并严格检查 CSV，返回 epoch、train loss、test loss 三个等长列表。"""

    # is_file 同时要求路径存在而且确实是文件；目录不能当 CSV 读取。
    if not path.is_file():
        # 读取可绘制的损失记录并检查轮次及数值未通过，抛出下方包含具体路径或实际值的错误。
        raise FileNotFoundError(f"Loss CSV not found: {path}")

    # 三个空列表稍后按行追加。相同下标永远代表 CSV 的同一行、同一个 epoch。
    epochs: List[int] = []
    # 按时间顺序收集每轮训练噪声预测 L1。
    train_losses: List[float] = []
    # 按相同时间顺序收集固定测试噪声下的 L1。
    test_losses: List[float] = []

    # with 会在缩进块结束后自动关闭文件，即使中途报错也不会一直占用文件句柄。
    with path.open("r", encoding="utf-8", newline="") as handle:
        # DictReader 用第一行表头作为键，之后可写 row["train_l1"]，比记列号更安全。
        reader = csv.DictReader(handle)
        # 当前文件或记录必须提供的字段。
        required = {"epoch", "train_l1", "test_l1"}
        # fieldnames 可能为 None（空文件），所以先用 set(reader.fieldnames or []) 兜底。
        if not required.issubset(set(reader.fieldnames or [])):
            # 读取可绘制的损失记录并检查轮次及数值未通过，抛出下方包含具体路径或实际值的错误。
            raise ValueError(
                # 报错说明损失 CSV 所需列名和实际文件位置。
                f"{path} must contain columns {sorted(required)}; "
                # 列出实际表头，并明确旧 val_l1 日志不适用于当前测试曲线。
                f"found {reader.fieldnames}. Old val_l1 logs are incompatible."
            )

        # enumerate(..., start=2) 从 2 开始，因为 CSV 第 1 行是表头，报错行号更直观。
        for row_number, row in enumerate(reader, start=2):
            # 将接下来的文件读取或转换放入异常检查范围，失败时保留可定位的原因。
            try:
                # int 把例如 "100" 转成整数 100；float 把小数字符串转成浮点数。
                epoch = int(row["epoch"])
                # 该轮训练批次平均的噪声预测误差。
                train_loss = float(row["train_l1"])
                # 固定测试条件下的噪声预测误差。
                test_loss = float(row["test_l1"])
            # 捕获读取可绘制的损失记录并检查轮次及数值中的读取、格式或兼容错误，让调用者获得明确失败原因。
            except (TypeError, ValueError) as error:
                # 读取可绘制的损失记录并检查轮次及数值未通过，抛出下方包含具体路径或实际值的错误。
                raise ValueError(f"Invalid numeric value at CSV line {row_number}: {row}") from error

            # epoch 必须是正整数；0/负数不属于本训练循环。
            if epoch < 1:
                # 读取可绘制的损失记录并检查轮次及数值未通过，抛出下方包含具体路径或实际值的错误。
                raise ValueError(f"Epoch must be positive at CSV line {row_number}: {epoch}")
            # 两个 loss 都必须是有限数；NaN/Inf 往往说明训练已经数值崩溃。
            if not math.isfinite(train_loss) or not math.isfinite(test_loss):
                # 读取可绘制的损失记录并检查轮次及数值未通过，抛出下方包含具体路径或实际值的错误。
                raise ValueError(f"Non-finite loss at CSV line {row_number}: {row}")
            # CSV 必须按训练时间严格递增；重复 epoch 会让续训曲线含两份互相矛盾的点。
            if epochs and epoch <= epochs[-1]:
                # 读取可绘制的损失记录并检查轮次及数值未通过，抛出下方包含具体路径或实际值的错误。
                raise ValueError(
                    # 错误列出相邻重复或倒序轮次，避免续训日志拼接后产生假曲线。
                    f"Epochs must be strictly increasing: {epochs[-1]} then {epoch}."
                )

            # 保存本行的训练轮次，作为两条曲线共用的横坐标。
            epochs.append(epoch)
            # 将该轮训练误差加入训练曲线。
            train_losses.append(train_loss)
            # 将同一轮测试误差加入测试曲线，保持下标与训练轮次对应。
            test_losses.append(test_loss)

    # 只有表头、没有任何数据点时也不能画图，给出清楚提示。
    if not epochs:
        # 读取可绘制的损失记录并检查轮次及数值未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeError(f"Loss CSV contains no data rows: {path}")
    # 共同返回横坐标和两组同长度损失，绘图时不再重新配对。
    return epochs, train_losses, test_losses


# 串联本文件的数据检查、模型处理和结果保存流程；下面的参数分别限定输入数据和处理约束。
def main() -> None:
    # 串联本文件的数据检查、模型处理和结果保存流程。
    """读取 loss.csv、检查共同间隔、画两条曲线并保存 PNG。"""

    # 终端传入并经解析的运行设置。
    args = parse_args()
    # resolve 把相对路径整理成绝对路径，终端输出时能明确知道读写的是哪个文件。
    csv_path = args.csv.resolve()
    # 若没写 --output，就把结果放在 CSV 旁边；否则尊重用户给定路径。
    output_path = (
        # 未指定图片路径时，把图保存到原始损失表旁边。
        csv_path.with_name("loss_curves.png")
        # 按恢复原时间长度前的噪声预测输出决定是否进入此分支；判断条件为 args.output is None。
        if args.output is None
        # 指定输出位置时，将其转换为明确的绝对路径。
        else args.output.resolve()
    )
    # 读入并验证损失表，得到已对齐的轮次和两组误差。
    epochs, train_losses, test_losses = load_losses(csv_path)

    # 相邻 epoch 相减可还原 CSV 的取点间隔。正常情况下全部等于 --loss-every。
    intervals = [right - left for left, right in zip(epochs, epochs[1:])]
    # 若相邻记录的轮次间隔不统一，就拒绝画成固定间隔曲线。
    if intervals and len(set(intervals)) != 1:
        # 串联本文件的数据检查、模型处理和结果保存流程未通过，抛出下方包含具体路径或实际值的错误。
        raise ValueError(
            # 报错列出实际出现的多个取点间隔，当前绘图要求固定间隔。
            f"Loss points do not use one constant epoch interval: {sorted(set(intervals))}"
        )
    # 将实际记录间隔写进标题；只有一个点时说明还不能判断间隔。
    interval_text = str(intervals[0]) if intervals else "only one point so far"

    # figsize=(10,6) 的单位是英寸；dpi=160 决定输出图片清晰度。
    figure, axis = plt.subplots(figsize=(10, 6))
    # 两次 plot 使用完全相同的 epochs；marker='.' 让每个真实 CSV 点都肉眼可见。
    axis.plot(epochs, train_losses, label="Train denoising L1", marker=".", linewidth=1.4)
    # 用与训练相同的横坐标画未见说话人测试 L1，点标记对应原始记录。
    axis.plot(epochs, test_losses, label="Unseen-test denoising L1", marker=".", linewidth=1.4)
    # 横轴标为完整训练轮次。
    axis.set_xlabel("Epoch")
    # 纵轴明确标为噪声预测 L1，避免误读为生成音频的客观性能。
    axis.set_ylabel("L1 noise-prediction loss")
    # 标题注明教师模型和两条曲线共用的真实取点间隔。
    axis.set_title(f"VoiceGrad-30 loss curves (shared interval: {interval_text} epoch)")
    # alpha<1 的网格线比较淡，不会盖住曲线；legend 显示两种颜色分别代表什么。
    axis.grid(True, alpha=0.25)
    # 显示图例，区分训练损失和未见说话人测试损失。
    axis.legend()
    # tight_layout 自动调整边距，防止坐标轴标题在 PNG 边缘被裁掉。
    figure.tight_layout()
    # 准备串联本文件的数据检查、模型处理和结果保存流程所需的目标目录，已有目录按给定选项复用。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 以每英寸 160 像素保存 PNG，保留足够清晰的坐标和图例。
    figure.savefig(output_path, dpi=160)
    # 主动关闭 figure 释放内存；脚本被批量调用时不会越积越多。
    plt.close(figure)

    # min 找最小 test loss 的数值，index 找它在列表中的位置，再映射回对应 epoch。
    minimum_test = min(test_losses)
    # 从最小测试误差的下标找到其实际训练轮次；不据此自动改用最佳模型。
    minimum_epoch = epochs[test_losses.index(minimum_test)]
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"loss_csv={csv_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"loss_curve={output_path}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(f"shared_epoch_interval={interval_text}")
    # 把串联本文件的数据检查、模型处理和结果保存流程的当前进度或结果输出到服务器日志。
    print(
        # 日志给出记录范围内的最小测试 L1 及对应轮次，不据此自动选模型。
        f"lowest_recorded_test_l1={minimum_test:.8f} at epoch={minimum_epoch} "
        # 图表日志明确这个最小测试 L1 只是诊断，不能代替转换语音客观指标。
        "(diagnostic only; not the paper's conversion-performance metric)"
    )


# 只有直接运行本文件才调用 main；被测试代码 import load_losses 时不会自动画图。
if __name__ == "__main__":
    # 执行本文件的参数检查及处理入口，错误会返回到终端。
    main()
