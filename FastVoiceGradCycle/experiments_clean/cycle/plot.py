"""读取已提交的真实损失，导出表格并绘制单组或两组对照图。"""  # 这个文件只处理日志和图表，不加载模型。
from __future__ import annotations  # 延后解释类型名称，导入时不需要训练依赖。
import argparse  # 读取实验目录、对照目录和图片输出位置。
import csv  # 将真实损失导出为可以复查的表格。
import hashlib  # 校验日志内容和图片文件的身份。
import json  # 读取断点指针、损失清单和逐行记录。
import math  # 拒绝无法绘制的无穷大或非数值损失。
import os  # 在文件完整写入后原子替换正式文件。
import time  # 记录本次生成图表的真实时间。
import uuid  # 为临时文件生成不会冲突的名称。
from pathlib import Path  # 安全处理实验和图片的文件路径。

SPLITS = ("train", "development", "validation", "test")  # 固定图例顺序，不把不同用途的数据混在一起。


def atomic_json(path, value):  # 完整写好收据后再发布，避免中断留下半份正式文件。
    target = Path(path)  # 将字符串统一转换成路径对象。
    target.parent.mkdir(parents=True, exist_ok=True)  # 创建收据所需的目录。
    temporary = target.with_name(target.name + ".writing." + uuid.uuid4().hex)  # 为本次写入建立独立临时文件。
    with temporary.open("w", encoding="utf-8") as handle:  # 先将内容写入临时文件。
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 保存中文并拒绝非法数值。
        handle.write("\n")  # 给文本末尾添加一个换行。
        handle.flush()  # 将文本缓冲交给文件系统。
        os.fsync(handle.fileno())  # 要求文件系统接收完整收据内容。
    os.replace(temporary, target)  # 完成写入后一次替换正式收据。


def sha256(path):  # 分块计算文件摘要，用内容校验代替只看文件名。
    digest = hashlib.sha256()  # 创建当前文件的摘要计算器。
    with Path(path).open("rb") as handle:  # 以只读二进制方式打开文件。
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):  # 每次最多读取八兆字节。
            digest.update(block)  # 将当前分块加入文件摘要。
    return digest.hexdigest()  # 返回可以记录在收据里的摘要文字。


def _pointer_sequence(run_dir):  # 读取最近完整断点共同提交的损失记录边界。
    pointer = Path(run_dir) / "checkpoint_pointer.json"  # 定位原实验使用的断点指针。
    if not pointer.is_file():  # 没有完整指针时不能自动认可尚未提交的日志。
        return 0  # 使用零边界，避免把中断前的尾部日志算作正式结果。
    value = json.loads(pointer.read_text(encoding="utf-8"))  # 读取真实断点收据。
    return int(value["current"].get("loss_sequence", 0))  # 返回当前完整训练状态对应的最后记录编号。


def read_committed_records(run_dir, committed_sequence=None):  # 只读取与完整断点一致的连续日志。
    root = Path(run_dir)  # 固定需要读取的实验目录。
    limit = _pointer_sequence(root) if committed_sequence is None else int(committed_sequence)  # 优先采用调用方明确提供的恢复边界。
    if limit < 0:  # 损失记录编号不能小于零。
        raise ValueError("损失提交边界不能为负数。")  # 拒绝不合法的日志范围。
    manifest = root / "loss_manifest.json"  # 定位正式发布的日志段清单。
    if not manifest.is_file():  # 实验尚未产生日志时允许返回空列表。
        if limit > 0:  # 已有正数边界却没有清单说明资料不完整。
            raise RuntimeError("完整断点对应的损失清单缺失，不能绘制正式曲线。")  # 明确报告缺项，不补造数据。
        return []  # 零边界且无日志时不生成虚假的曲线。
    value = json.loads(manifest.read_text(encoding="utf-8"))  # 读取清单中的已发布日志段。
    records = {}  # 用唯一记录编号防止重复计入同一次更新。
    for segment in value.get("segments", []):  # 逐段读取当前实验实际登记的日志。
        path = root / segment["path"]  # 按清单定位当前日志段。
        valid_file = path.is_file() and path.stat().st_size == int(segment["bytes"])  # 先确认文件存在且字节数完整。
        if not valid_file or sha256(path) != segment["sha256"]:  # 再检查真实内容，发现等长损坏也会停止。
            raise RuntimeError(f"损失日志段缺失或不完整：{path}")  # 显示需要核查的具体文件。
        boundary = min(limit, int(segment["last_sequence"]))  # 同时遵守断点边界和恢复后被截短的日志段边界。
        with path.open("r", encoding="utf-8") as handle:  # 逐行读取当前日志段。
            for line in handle:  # 每一行对应真实发生的一次损失记录。
                record = json.loads(line)  # 解析当前记录，而不插值或补齐缺失记录。
                sequence = int(record["sequence"])  # 读取当前记录的唯一编号。
                if sequence <= boundary:  # 忽略完整断点之后必须重跑的尾部记录。
                    if sequence in records and records[sequence] != record:  # 相同编号只能保留内容相同的记录。
                        raise RuntimeError(f"损失序号重复且内容不一致：{sequence}")  # 防止回滚前后两条轨迹被错误拼接。
                    records[sequence] = record  # 保存提交边界内的实际记录。
    ordered = [records[key] for key in sorted(records)]  # 按真实发生顺序整理记录。
    if [record["sequence"] for record in ordered] != list(range(1, limit + 1)):  # 检查从第一条到提交边界没有缺行。
        raise RuntimeError("已提交损失日志存在缺行，不能绘制正式曲线。")  # 拒绝用连线掩盖日志缺口。
    for record in ordered:  # 检查每条已提交记录可以安全用于图表。
        if record["split"] not in SPLITS:  # 数据用途必须能明确区分。
            raise ValueError(f"未知损失集合：{record['split']}")  # 不把未知集合误标为训练或验证。
        if not all(math.isfinite(float(number)) for number in record["losses"].values()):  # 所有实际损失必须是有限数。
            raise FloatingPointError(f"损失记录含非法数值：{record['sequence']}")  # 不绘制隐藏数值异常的曲线。
    return ordered  # 返回已经核对内容和完整性的正式记录。


def _plotting():  # 只有真正画图时才导入绘图库。
    import matplotlib  # 使用标准图形库，不依赖训练框架。
    matplotlib.use("Agg")  # 在没有显示器的服务器上输出图片。
    import matplotlib.pyplot as pyplot  # 提供创建坐标轴和保存图片的接口。
    return pyplot  # 让纯日志检查与命令行帮助不需要安装绘图库。


def _save_figure(figure, path):  # 将完整渲染的图片一次发布。
    target = Path(path)  # 统一图片的目标路径类型。
    target.parent.mkdir(parents=True, exist_ok=True)  # 创建图片输出目录。
    temporary = target.with_name("." + target.name + ".writing." + uuid.uuid4().hex)  # 避免读取到尚未渲染完成的图片。
    figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")  # 先完整生成清晰的图片。
    os.replace(temporary, target)  # 图片完整后再发布正式文件名。


def _safe_name(value):  # 将损失名称变成可用的文件名。
    return "".join(character if character.isalnum() or character in "_-" else "_" for character in value)  # 只替换文件名字符，不改变损失数值。


def _export_csv(records, names, target):  # 保存所有逐条数值供其他软件重新检查。
    temporary = target.with_name(target.name + ".writing." + uuid.uuid4().hex)  # 避免中断产生半份正式表格。
    with temporary.open("w", encoding="utf-8", newline="") as handle:  # 使用标准表格换行规则写入临时文件。
        writer = csv.DictWriter(handle, fieldnames=["sequence", "split", "global_step"] + names)  # 固定记录身份与实际损失字段。
        writer.writeheader()  # 先写列名，方便直接打开检查。
        for record in records:  # 原样导出每条已提交记录。
            row = {key: record[key] for key in ("sequence", "split", "global_step")}  # 保存编号、集合与更新步数。
            writer.writerow({**row, **record["losses"]})  # 缺少的损失留空，不用零代替未计算项目。
    os.replace(temporary, target)  # 表格写完后发布正式文件。


def plot_losses(run_dir, committed_sequence=None, output_dir=None):  # 为单组实验分别绘制每种真实损失。
    records = read_committed_records(run_dir, committed_sequence)  # 只使用完整断点认可的记录。
    if not records:  # 尚无正式记录时不生成误导性的空图。
        print(f"当前尚无已提交损失：{run_dir}")  # 说明为什么没有图片产物。
        return []  # 返回空产物列表供训练器继续处理。
    root = Path(output_dir) if output_dir is not None else Path(run_dir) / "loss_plots"  # 兼容训练自动绘图，也允许另存图表。
    root.mkdir(parents=True, exist_ok=True)  # 创建用户指定的图片输出目录。
    names = sorted({name for record in records for name in record["losses"]})  # 从真实记录发现损失字段，不假设未计算的项目存在。
    csv_path = root / "all_losses.csv"  # 沿用既有完整损失表的文件名。
    _export_csv(records, names, csv_path)  # 导出未经平滑或改写的真实损失。
    pyplot = _plotting()  # 完成日志检查后才加载绘图库。
    outputs = [str(csv_path)]  # 记录本次实际写出的表格路径。
    for name in names:  # 每种损失独立成图，方便核查不同尺度。
        figure, axis = pyplot.subplots(figsize=(9, 4.8))  # 创建清晰的白底坐标轴。
        for split in SPLITS:  # 明确区分训练、开发、验证和测试。
            items = [record for record in records if record["split"] == split and name in record["losses"]]  # 只取当前集合实际计算过的损失。
            if items:  # 当前集合有数据才画线，不生成虚构图例。
                steps = [record["global_step"] for record in items]  # 用累计优化更新次数作为横轴。
                values = [record["losses"][name] for record in items]  # 直接读取未经平滑的损失数值。
                marker = None if split == "train" else "o"  # 用圆点显示数量较少的真实评估时点。
                axis.plot(steps, values, label=split, linewidth=0.9, alpha=0.8, marker=marker, markersize=3)  # 保留每条真实记录，不插入缺失点。
        axis.set(xlabel="Global optimizer step", ylabel=name, title=name)  # 让标题和纵轴准确指向当前损失。
        axis.grid(alpha=0.18)  # 添加浅网格帮助读数。
        axis.legend()  # 标出各条曲线的真实数据用途。
        figure.tight_layout()  # 调整边距，避免文字被裁切。
        path = root / (_safe_name(name) + ".png")  # 沿用每个损失各一张图片的稳定文件名。
        _save_figure(figure, path)  # 发布完整图片。
        pyplot.close(figure)  # 释放当前画布占用的内存。
        outputs.append(str(path))  # 登记已经成功生成的图片。
    receipt = {  # 汇总图表对应的真实输入范围。
        "committed_sequence": records[-1]["sequence"],  # 保存这次绘图的完整断点边界。
        "records": len(records),  # 保存实际使用的记录数量。
        "files": outputs,  # 保存成功生成的文件清单。
        "created_at_unix": time.time(),  # 保存这次绘图的真实时间。
        "smoothing": "none",  # 明确图片展示原始数值，不用平滑曲线代替原值。
    }  # 完成绘图收据内容。
    atomic_json(root / "plot_receipt.json", receipt)  # 原子保存可追溯的绘图收据。
    print(f"损失表格和 {len(names)} 张曲线已保存：{root}")  # 告知实际产物数量和位置。
    return outputs  # 返回真实产物供训练日志记录。


def plot_comparison(run_dir, compare_dir, output_dir):  # 用相同集合和损失字段比较两组实验。
    first = read_committed_records(run_dir)  # 核对当前组的已提交记录。
    second = read_committed_records(compare_dir)  # 核对对照组的已提交记录。
    if not first or not second:  # 两组都必须有实际记录才能进行比较。
        raise RuntimeError("两组都需要已提交的真实损失记录，才能绘制对照图。")  # 不用空线或假数据代表缺失实验。
    pyplot = _plotting()  # 所有日志核对通过后才载入绘图库。
    root = Path(output_dir)  # 使用独立的对照图输出目录。
    root.mkdir(parents=True, exist_ok=True)  # 创建新的图表位置。
    outputs = []  # 收集本次实际生成的对照图。
    for split in SPLITS:  # 每个数据集合单独画图，防止训练和测试被混在同一比较中。
        left = [record for record in first if record["split"] == split]  # 选取第一组当前集合的真实记录。
        right = [record for record in second if record["split"] == split]  # 选取第二组相同用途的记录。
        left_names = {name for record in left for name in record["losses"]}  # 查找第一组实际计算过的损失。
        right_names = {name for record in right for name in record["losses"]}  # 查找第二组实际计算过的损失。
        names = sorted(left_names & right_names)  # 仅比较两组共同存在的损失字段。
        if not names:  # 没有共同字段时明确跳过当前集合。
            continue  # 不为不存在的共同指标生成图片。
        columns = min(3, len(names))  # 每行最多三个面板，便于阅读。
        rows = (len(names) + columns - 1) // columns  # 根据共同字段数量安排足够的面板。
        figure, axes = pyplot.subplots(rows, columns, figsize=(5 * columns, 3.6 * rows), squeeze=False)  # 创建规则的多面板布局。
        for axis, name in zip(axes.flat, names):  # 每个面板对应一种共同损失。
            for records, directory in ((left, run_dir), (right, compare_dir)):  # 对两组使用相同的绘制规则。
                items = [record for record in records if name in record["losses"]]  # 不把未计算的损失补成零。
                steps = [record["global_step"] for record in items]  # 使用真实绝对训练步数。
                values = [record["losses"][name] for record in items]  # 使用原始损失值进行比较。
                axis.plot(steps, values, label=Path(directory).name, linewidth=0.9, alpha=0.8)  # 用目录名标明各组来源。
            axis.set(title=name, xlabel="Global optimizer step", ylabel=name)  # 清楚标注当前字段和坐标轴。
            axis.grid(alpha=0.18)  # 添加不遮挡曲线的浅网格。
            axis.legend(fontsize="small")  # 标明两组实际实验目录。
        for axis in list(axes.flat)[len(names):]:  # 处理最后一行没有使用的面板。
            axis.set_visible(False)  # 隐藏没有数据的空白坐标轴。
        figure.suptitle(f"FastVoiceGrad comparison: {split}")  # 标明这张图对应的数据用途。
        figure.tight_layout()  # 整理面板间距和标题空间。
        path = root / f"comparison_{split}.png"  # 不同集合使用不同的稳定文件名。
        _save_figure(figure, path)  # 原子发布当前集合的对照图。
        pyplot.close(figure)  # 释放已经完成的画布。
        outputs.append(str(path))  # 登记真实写出的对照图。
    atomic_json(root / "comparison_receipt.json", {"runs": [str(run_dir), str(compare_dir)], "files": outputs})  # 保存两组来源和实际产物。
    return outputs  # 返回当前生成的全部对照图。


def main(argv=None):  # 提供只读日志、另存图表的命令行入口。
    import config as settings  # 只从本组唯一设置文件读取默认实验路径。
    parser = argparse.ArgumentParser(description=__doc__)  # 显示这个文件的绘图职责。
    parser.add_argument("--run-dir", type=Path, default=settings.RUN_DIR)  # 指定需要绘图的真实实验目录。
    parser.add_argument("--compare-dir", type=Path)  # 可选另一组目录，生成相同集合的对照图。
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "analysis")  # 命令行默认将图另存到新整理目录。
    args = parser.parse_args(argv)  # 读取用户实际提供的路径参数。
    plot_losses(args.run_dir, output_dir=args.output_dir)  # 输出单组原始曲线和完整损失表。
    if args.compare_dir is not None:  # 用户明确给出另一组才生成对照图。
        plot_comparison(args.run_dir, args.compare_dir, args.output_dir)  # 将对照图保存到相同的新输出目录。


if __name__ == "__main__":  # 只有直接运行文件才会写出图片。
    main()  # 执行日志核对和图表生成。
