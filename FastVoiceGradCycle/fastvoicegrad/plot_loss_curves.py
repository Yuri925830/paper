#!/usr/bin/env python3
# 校验 FastVoiceGrad 训练/测试损失 CSV，并以原子方式生成两张曲线图。
"""校验 FastVoiceGrad 训练/测试损失 CSV，并以原子方式生成两张曲线图。"""

# 启用较新的类型标注行为，让这里的标注不会在导入时造成额外运行依赖。
from __future__ import annotations

# ``argparse`` 负责读取命令行的 CSV 路径和输出目录。
import argparse
# ``csv`` 负责按表头解析每个 epoch 的训练和测试损失。
import csv
# ``math.isfinite`` 用来拒绝 NaN 和正负无穷，防止画出误导性曲线。
import math
# ``os`` 提供进程号、强制落盘和原子替换文件等操作。
import os
# ``Path`` 让 Windows/Linux 路径处理保持统一且易读。
from pathlib import Path
# 这些容器类型说明每个函数接收和返回的数据结构，不改变实际计算。
from typing import Dict, List, Mapping, Sequence, Tuple

# 先导入 matplotlib 主包，以便在导入 pyplot 前指定无窗口后端。
import matplotlib

# ``Agg`` 不需要显示器，适合租用服务器后台训练时自动画 PNG。
matplotlib.use("Agg")
# pyplot 提供折线图、坐标轴、图例和关闭画布等绘图操作。
import matplotlib.pyplot as plt


# 这些名称必须与 Cycle v4 训练代码输出的九个损失分量一一对应，顺序也写入 CSV 契约。
COMPONENTS = (
    "student_total",  # 学生总损失：原有 ACDD 三项加上按 lambda_cycle 计权的 Cycle 项。
    "student_adv",  # 学生对抗损失，衡量生成波形骗过判别器的程度。
    "fm_raw",  # 尚未乘权重 2 的原始特征匹配损失。
    "fm_weighted",  # 已乘权重 2、真正计入学生总损失的特征匹配项。
    "dist_raw",  # 尚未乘权重 45 的原始分数蒸馏损失。
    "dist_weighted",  # 已乘权重 45、真正计入学生总损失的蒸馏项。
    "cycle_raw",  # 尚未乘 lambda_cycle、但已按公开代码额外除 80 的有效帧 A→B→A 损失。
    "cycle_weighted",  # 已乘 lambda_cycle、真正计入学生总损失的 Cycle 项。
    "discriminator",  # 判别器自身诊断损失，它不加进学生总损失。
)
# CSV 固定先写训练游标，再写九个训练值、九个测试值以及测试覆盖数量。
LOSS_FIELDS = (
    "epoch",  # 当前完成的是第几个完整训练轮次，从 1 开始。
    "global_step",  # 到该轮结束为止完成了多少次学生参数更新。
    *(f"train_{key}" for key in COMPONENTS),  # 为九个分量依次生成 train_ 前缀列。
    *(f"test_{key}" for key in COMPONENTS),  # 为同一批九个分量依次生成 test_ 前缀列。
    "test_examples",  # 本轮实际覆盖的测试语音条数，严格应为 100。
    "test_batches",  # 批量 32 时 100 条测试语音应分成 4 批。
)


# 读取并严格校验损失表，只有完整、有限且连续的数据才交给绘图函数。
def load_loss_rows(path: Path) -> List[Dict[str, float]]:
    # 读取并严格校验损失表，只有完整、有限且连续的数据才交给绘图函数。
    """读取并严格校验损失表，只有完整、有限且连续的数据才交给绘图函数。"""

    # 路径不是一个现有文件时立即报错，避免后面得到含糊的空图。
    if not path.is_file():
        # 错误消息带上完整调用路径，方便在服务器目录中定位问题。
        raise FileNotFoundError(f"Student loss CSV not found: {path}")
    # 新建结果列表；每个元素最终都是一轮训练的数值字典。
    rows: List[Dict[str, float]] = []
    # 以 UTF-8 和 CSV 推荐的 newline 模式只读打开，退出代码块后文件自动关闭。
    with path.open("r", encoding="utf-8", newline="") as handle:
        # DictReader 用第一行表头把后续每个单元格映射到列名。
        reader = csv.DictReader(handle)
        # 表头必须与训练模块的固定契约逐列完全一致，不能少列、多列或换顺序。
        if tuple(reader.fieldnames or ()) != LOSS_FIELDS:
            # 把实际表头写入异常，方便判断是否误用了旧版本 CSV。
            raise RuntimeError(
                # 报告 CSV 实际表头，帮助定位遗漏或误名的损失列。
                f"Unexpected student loss schema: {reader.fieldnames!r}"
            )
        # 从物理第 2 行开始读取数据，因为第 1 行已经是 CSV 表头。
        for line_number, raw in enumerate(reader, start=2):
            # 数值转换单独放进 try，任何缺列或非法字符串都会变成带行号的统一错误。
            try:
                # epoch、step 和覆盖数量必须是整数；真正的损失允许是浮点数。
                parsed = {
                    # 同名列保留为字典键；轮次和数量解析为整数，其余损失解析为浮点数。
                    key: (
                        # 游标和计数列用 int，避免把 100.5 之类不合法计数悄悄接受。
                        int(raw[key])
                        # 这四列记录进度或样本数量，必须使用整数。
                        if key in {"epoch", "global_step", "test_examples", "test_batches"}
                        # 其余十八个训练/测试损失统一转成 float 供检查和绘图。
                        else float(raw[key])
                    )
                    # 遍历固定表头而不是原始行，确保不会漏掉契约要求的字段。
                    for key in LOSS_FIELDS
                }
            # KeyError 表示缺列，TypeError/ValueError 表示内容无法按规定转换。
            except (KeyError, TypeError, ValueError) as error:
                # 保留原始异常作为原因，同时给用户显示准确的 CSV 行号。
                raise RuntimeError(
                    # 报告损失表中非法行的位置与原始转换错误。
                    f"Invalid student loss row at line {line_number}: {error}"
                # 保留底层异常链，便于追查 CSV 解析失败的具体原因。
                ) from error
            # 除第一条外，每一轮都必须恰好接在上一轮之后，防止断点或重复轮次混入曲线。
            if rows and parsed["epoch"] != rows[-1]["epoch"] + 1:
                # 不连续时拒绝作图，因为连接断裂数据会产生看似平滑的错误趋势。
                raise RuntimeError("Student loss epochs must be contiguous and increasing")
            # epoch 与 global_step 都从正数开始，0 或负数说明记录游标损坏。
            if parsed["epoch"] < 1 or parsed["global_step"] < 1:
                # 异常中带行号，便于直接在 CSV 中找到损坏记录。
                raise RuntimeError(f"Invalid student loss cursor at line {line_number}")
            # text-unseen 测试必须覆盖 100 条，批量 32 时必须正好处理 4 批（32+32+32+4）。
            if parsed["test_examples"] != 100 or parsed["test_batches"] != 4:
                # 覆盖不完整就拒绝画“测试损失”，避免把部分样本结果冒充完整测试结果。
                raise RuntimeError(f"Incomplete held-out test coverage at line {line_number}")
            # 对所有真正的损失列检查有限性；计数列不需要浮点有限性检查。
            if not all(
                # NaN 或正负无穷都会让 isfinite 返回 False。
                math.isfinite(float(parsed[key]))
                # 逐列遍历固定契约，排除四个只表示位置/数量的整数列。
                for key in LOSS_FIELDS
                # 排除进度和数量列，只对实际损失值实施非负检查。
                if key not in {"epoch", "global_step", "test_examples", "test_batches"}
            ):
                # 一旦任意损失非有限就立即报告对应 CSV 行号。
                raise RuntimeError(f"Non-finite student loss at line {line_number}")
            # 通过全部检查后才加入结果，保证返回列表中没有半合法记录。
            rows.append(parsed)
    # 只有表头、没有任何 epoch 数据时也不能画图，否则空图容易被误认为真实结果。
    if not rows:
        # 报错中保留输入路径，帮助确认训练是否尚未完成第一轮。
        raise RuntimeError(f"Student loss CSV has no rows: {path}")
    # 返回已经转换和校验好的按 epoch 排列的数值记录。
    return rows


# 先完整写临时 PNG，再一次性替换正式文件，避免 VS Code 读到半张图。
def _atomic_save(figure, path: Path) -> None:
    # 先完整写临时 PNG，再一次性替换正式文件，避免 VS Code 读到半张图。
    """先完整写临时 PNG，再一次性替换正式文件，避免 VS Code 读到半张图。"""

    # 自动创建目标目录及其缺失的父目录；目录已存在时不会报错。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件名包含当前进程号，降低并发绘图时互相覆盖的可能性。
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # try/finally 确保即使保存失败，也会清理残留临时文件。
    try:
        # 以 180 dpi 保存 PNG，并紧凑裁掉图像周围多余空白；正式路径此时仍未变化。
        figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        # 二进制只读重新打开刚写完的临时文件，准备强制同步到磁盘。
        with temporary.open("rb") as handle:
            # fsync 等待操作系统把文件内容真正刷盘，降低断电时留下损坏 PNG 的风险。
            os.fsync(handle.fileno())
        # 原子替换正式 PNG：读者看到的要么是旧完整图，要么是新完整图。
        os.replace(temporary, path)
    # 无论保存、同步或替换在哪一步异常，都执行下面的残留清理。
    finally:
        # 正常替换后临时路径已不存在；异常时若仍存在才需要删除。
        if temporary.exists():
            # 删除不完整临时图，避免它被误当成训练产物。
            temporary.unlink()


# 按 epoch 顺序抽出指定损失列，并统一转成 matplotlib 可画的浮点列表。
def _series(rows: Sequence[Mapping[str, float]], key: str) -> List[float]:
    # 按 epoch 顺序抽出指定损失列，并统一转成 matplotlib 可画的浮点列表。
    """按 epoch 顺序抽出指定损失列，并统一转成 matplotlib 可画的浮点列表。"""

    # 保留 rows 原有顺序，从每轮记录中取同一列，因此横纵坐标能够正确一一对应。
    return [float(row[key]) for row in rows]


# 从完整 CSV 重画总损失图和分项图，两张 PNG 都分别原子替换。
def plot_loss_curves(csv_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    # 从完整 CSV 重画总损失图和分项图，两张 PNG 都分别原子替换。
    """从完整 CSV 重画总损失图和分项图，两张 PNG 都分别原子替换。"""

    # 再包装为 Path 后读取，确保字符串路径调用者也能正常使用，并先执行严格数据校验。
    rows = load_loss_rows(Path(csv_path))
    # 横轴使用整数 epoch，而不是行号或 global_step，方便直接对应检查点轮次。
    epochs = [int(row["epoch"]) for row in rows]
    # 输出目录也统一转换成 Path，后续拼接文件名时兼容不同操作系统。
    output_dir = Path(output_dir)

    # 第一张图只比较最重要的训练总损失和固定测试总损失。
    total_path = output_dir / "student_total_loss_curves.png"
    # 创建 9.4×5.8 英寸画布和一个坐标轴，尺寸兼顾报告可读性与文件体积。
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    # 绘制每个 epoch 的在线训练总损失均值。
    axis.plot(
        epochs,  # 横坐标：完成的训练轮次。
        _series(rows, "train_student_total"),  # 纵坐标：该轮所有训练批的平均总损失。
        # 错误消息中使用的字段来源名称。
        label="Train total (online epoch mean)",  # 图例说明这是训练期间在线统计值。
        linewidth=1.6,  # 线宽稍粗，投影到 PPT 后仍清楚。
        marker=".",  # 每个 epoch 标一个点，避免把插值线误当成更密集观测。
    )
    # 在同一坐标轴绘制固定 100 条 text-unseen 测试语音的总损失。
    axis.plot(
        epochs,  # 使用相同 epoch 横轴，便于逐轮比较训练与测试差距。
        _series(rows, "test_student_total"),  # 每轮在固定裁剪/噪声下得到的测试总损失。
        # 错误消息中使用的字段来源名称。
        label="Held-out unseen test total (fixed 100)",  # 明确是 100 条留出测试样本。
        linewidth=1.6,  # 与训练曲线同线宽，不人为强调任何一边。
        marker=".",  # 同样显示每一轮的真实观测点。
    )
    # 标题说明这是 FastVoiceGrad 学生原有 ACDD 与可选 Cycle 联合目标的总损失。
    axis.set_title("FastVoiceGrad student total loss (ACDD + optional cycle)")  # 用英文标题兼容论文图表排版。
    # 横轴标签写 Epoch，即完整遍历训练集的次数。
    axis.set_xlabel("Epoch")
    # 纵轴直接写联合公式，提醒读者 Cycle 项还要乘实验配置中的 lambda_cycle。
    axis.set_ylabel("L_adv + 2 L_FM + 45 L_dist + lambda_cycle L_cycle")  # 显示训练器实际优化的联合目标。
    # 加浅色网格辅助读取趋势，alpha 越小网格越淡，不会盖住曲线。
    axis.grid(alpha=0.25)
    # 显示训练与测试两条曲线的图例。
    axis.legend()
    # 自动整理标题、标签和图例间距，避免保存时文字互相覆盖。
    figure.tight_layout()
    # 通过临时文件原子写入总损失图，训练过程中打开它也不会读到半成品。
    _atomic_save(figure, total_path)
    # 立即释放第一张画布占用的内存，长期训练每轮绘图时不会不断累积。
    plt.close(figure)

    # 第二张图把四个已计权学生分项和判别器诊断项拆成五个面板。
    component_path = output_dir / "student_loss_component_curves.png"
    # 创建 3×2 画布并共享 epoch 横轴，五个有效面板留出足够空间显示 Cycle 曲线。
    figure, axes = plt.subplots(3, 2, figsize=(12.4, 11.2), sharex=True)  # 两列三行中的最后一格稍后隐藏。
    # 每个三元组依次给出 CSV 分量名、面板标题和是否使用对数纵轴。
    panels = (
        # 对抗损失保持线性纵轴；越低通常表示学生更能骗过当前判别器。
        ("student_adv", "Student adversarial loss", False),
        # 画乘 2 后的特征匹配项，数值就是它对总损失的真实贡献。
        ("fm_weighted", "Weighted feature matching (2 x raw)", False),
        # 画乘 45 后的蒸馏项，方便与其他三个实际计入学生总目标的分量比较量级。
        ("dist_weighted", "Weighted score distillation (45 x raw)", False),
        # 判别器损失只用于诊断对抗平衡，不属于学生总损失；当前同样使用线性轴。
        (
            # 读取判别器损失列，单独作为对抗训练诊断。
            "discriminator",
            # 面板标题明确判别器损失不计入学生总损失。
            "Discriminator diagnostic (excluded from student total)",
            # 当前面板使用普通线性纵轴。
            False,
        ),
        # 把 Cycle 放在左下角，使隐藏右下角后它仍显示共享横轴的刻度和标签。
        ("cycle_weighted", "Weighted cycle consistency (lambda_cycle x raw)", False),  # 展示 Cycle 对联合目标的真实贡献。
    )
    # axes.flat 按行展开前五个坐标轴，与 panels 中五项一一配对。
    for axis, (component, title, use_log_scale) in zip(axes.flat, panels):
        # 在当前面板画这个分量的训练曲线。
        axis.plot(
            epochs,  # 横轴仍是 epoch。
            _series(rows, f"train_{component}"),  # 根据分量名动态读取对应 train_ 列。
            # 错误消息中使用的字段来源名称。
            label="Train",  # 图例标明训练曲线。
            linewidth=1.4,  # 小面板使用略细线条，减少视觉拥挤。
            marker=".",  # 显示每轮实际记录点。
        )
        # 紧接着在同一面板画完全相同分量的留出测试曲线。
        axis.plot(
            epochs,  # 与训练分量共享横坐标。
            _series(rows, f"test_{component}"),  # 动态读取对应 test_ 列。
            # 错误消息中使用的字段来源名称。
            label="Held-out unseen test",  # 明确它来自 text-unseen 测试集。
            linewidth=1.4,  # 与训练线相同粗细，避免产生偏重。
            marker=".",  # 标出每轮测试点。
        )
        # 当前五项都传入 False；保留此分支便于将来量级跨度大时显式启用对数轴。
        if use_log_scale:
            # 对数轴会拉开小数值差异，但包含 0/负数时不适用，所以必须由面板配置决定。
            axis.set_yscale("log")
        # 为当前小图设置直白标题，说明正在查看哪个损失项。
        axis.set_title(title)
        # 使用浅色网格帮助比较相邻 epoch，透明度 0.25 不遮挡折线。
        axis.grid(alpha=0.25)
        # 小面板图例字号设为 8，保证说明完整又不压住过多曲线区域。
        axis.legend(fontsize=8)
    # 第六个坐标轴没有对应损失项，明确隐藏它，避免留下容易被误解的空白图框。
    axes.flat[-1].set_visible(False)  # 关闭多出来的右下角坐标轴及其边框、刻度和背景。
    # 只给仍可见的最下面一排面板加横轴标签，减少重复文字造成的拥挤。
    for axis in axes[-1]:  # 遍历左下有效面板和右下隐藏面板，保持统一布局代码。
        # 可见的底部面板标明横坐标是 Epoch，隐藏面板设置标签也不会显示。
        axis.set_xlabel("Epoch")
    # 整张五面板图的总标题说明测试输入固定，且测试曲线不参与挑选检查点。
    figure.suptitle(
        # 图标题明确固定测试噪声和裁剪；曲线只供诊断，不用于挑选检查点。
        "Fixed test noise/crops; curves are diagnostic and never select a checkpoint",
        # 总标题字号为 11，避免挤占五张分项图。
        fontsize=11,
    )
    # 自动调整五个面板与总标题的间距，避免保存后发生裁切或重叠。
    figure.tight_layout()
    # 原子保存分项损失图，防止服务器后台改图时 VS Code 打开到不完整文件。
    _atomic_save(figure, component_path)
    # 关闭第二张画布，释放 matplotlib 对象及其缓存。
    plt.close(figure)
    # 返回两张正式 PNG 的路径，让训练器可以打印或测试文件是否生成成功。
    return total_path, component_path


# 定义并解析命令行参数，不在这里读取或修改任何实验文件。
def parse_args() -> argparse.Namespace:
    # 定义并解析命令行参数，不在这里读取或修改任何实验文件。
    """定义并解析命令行参数，不在这里读取或修改任何实验文件。"""

    # description 直接使用模块说明，使 ``--help`` 自动告诉用户脚本用途。
    parser = argparse.ArgumentParser(description=__doc__)
    # ``--csv`` 必填并转成 Path，调用时必须明确指出要画哪个 loss.csv。
    parser.add_argument("--csv", type=Path, required=True)
    # ``--output-dir`` 可选；不写时主函数会把图片放在 CSV 所在目录。
    parser.add_argument("--output-dir", type=Path, default=None)
    # 让 argparse 校验实际命令行并返回带 ``csv``、``output_dir`` 属性的对象。
    return parser.parse_args()


# 命令行入口：规范化路径、生成两张图并逐行打印最终位置。
def main() -> None:
    # 命令行入口：规范化路径、生成两张图并逐行打印最终位置。
    """命令行入口：规范化路径、生成两张图并逐行打印最终位置。"""

    # 解析用户传入的两个参数。
    args = parse_args()
    # 展开 ``~`` 并转成绝对路径，日志里打印的位置不会受当前工作目录影响。
    csv_path = args.csv.expanduser().resolve()
    # 若未指定输出目录就使用 CSV 父目录；指定时同样展开并转绝对路径。
    output_dir = (
        # 默认路径保证 loss.csv 与两张曲线图放在同一个实验目录中。
        csv_path.parent
        # 未指定图片目录时，选择损失 CSV 所在文件夹。
        if args.output_dir is None
        # 用户自定义目录也规范化，避免相对路径在后台任务中指向意外位置。
        else args.output_dir.expanduser().resolve()
    )
    # 生成两张图，并让终端分别显示它们的绝对路径，方便在 VS Code 中打开。
    for path in plot_loss_curves(csv_path, output_dir):
        # 每次 print 一条路径，因此日志中可以直接搜索两个固定 PNG 文件名。
        print(path)


# 只有直接执行本脚本时才运行 main；被训练模块或单元测试导入时不会自动画图。
if __name__ == "__main__":
    # 进入上面的命令行主流程。
    main()
