from __future__ import annotations  # 延后解释类型提示，避免导入时加载训练依赖。
import csv  # 将所有真实损失记录导出为便于画图的表格。
import hashlib  # 为每段已保存的损失日志记录文件身份。
import json  # 读取和保存逐行损失及提交清单。
import math  # 拒绝无穷大及无法表示的损失数值。
import os  # 用同目录替换保证日志只出现完整文件。
import time  # 为每条损失记录保存实际发生时间。
import uuid  # 为原子写入和新日志段分配互不冲突的名称。
from pathlib import Path  # 安全处理 Drive 中文路径和本地路径。

def atomic_json(path, value):  # 将完整 JSON 写入临时文件后再发布到正式位置。
    target = Path(path)  # 将字符串路径统一转换为路径对象。
    target.parent.mkdir(parents=True, exist_ok=True)  # 创建目标文件所需的父目录。
    temporary = target.with_name(target.name + ".writing." + uuid.uuid4().hex)  # 避免不同写入者共用同一临时文件名。
    with temporary.open("w", encoding="utf-8") as handle:  # 先写入尚未被读取者使用的临时文件。
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)  # 保留中文并禁止非法 JSON 数值。
        handle.write("\n")  # 为可读文本增加结束换行。
        handle.flush()  # 将 Python 缓冲区交给文件系统。
        os.fsync(handle.fileno())  # 要求当前文件系统接收所有已写字节。
    os.replace(temporary, target)  # 完整写入后替换正式文件。

def _sha256(path):  # 只对小型日志段计算内容摘要。
    digest = hashlib.sha256()  # 为当前文件创建摘要计算器。
    with Path(path).open("rb") as handle:  # 只读打开当前文件。
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):  # 分块读取以避免额外的大内存占用。
            digest.update(block)  # 将当前块加入文件身份。
    return digest.hexdigest()  # 返回可以保存进 JSON 的摘要字符串。

def _pointer_sequence(run_dir):  # 从最近完整断点指针读取损失日志的安全边界。
    pointer = Path(run_dir) / "checkpoint_pointer.json"  # 定位共享断点工具发布的指针。
    if not pointer.is_file():  # 尚未发布完整断点时没有可自动确定的安全边界。
        return 0  # 返回零以避免把未提交损失误当作正式训练结果。
    value = json.loads(pointer.read_text(encoding="utf-8"))  # 读取轻量断点收据。
    return int(value["current"].get("loss_sequence", 0))  # 读取与当前训练态共同提交的最后一行编号。

def read_committed_records(run_dir, committed_sequence=None):  # 只读取属于最近完整训练态的损失记录。
    root = Path(run_dir)  # 固定当前实验的永久目录。
    limit = _pointer_sequence(root) if committed_sequence is None else int(committed_sequence)  # 使用调用方恢复断点的游标或正式指针。
    manifest = root / "loss_manifest.json"  # 定位已发布的损失段目录。
    if not manifest.is_file():  # 没有任何已提交损失时返回空列表。
        if limit > 0:  # 完整断点声明已有损失时缺清单属于损坏。
            raise RuntimeError("完整断点对应的损失清单缺失，禁止画不完整曲线。")  # 保留真实训练态并明确报告日志缺项。
        return []  # 不伪造训练、验证或测试曲线。
    value = json.loads(manifest.read_text(encoding="utf-8"))  # 读取正式日志段清单。
    records = {}  # 按唯一顺序号收集真实记录。
    for segment in value.get("segments", []):  # 按提交清单读取仍属于当前实验轨迹的日志段。
        path = root / segment["path"]  # 定位当前段的永久文件。
        if not path.is_file() or path.stat().st_size != int(segment["bytes"]) or _sha256(path) != segment["sha256"]:  # 对小日志核对完整内容，防止等长损坏混入曲线。
            raise RuntimeError(f"损失日志段缺失或不完整：{path}")  # 明确列出损坏的日志段。
        with path.open("r", encoding="utf-8") as handle:  # 流式读取当前日志段。
            for line in handle:  # 每一行保存一个实际 batch 或实际验证批次的记录。
                record = json.loads(line)  # 将当前 JSON 行解析为字典。
                sequence = int(record["sequence"])  # 读取当前记录的唯一游标。
                if sequence <= min(limit, int(segment["last_sequence"])):  # 同时排除恢复时截断的旧段尾部和断点之后的尾部。
                    if sequence in records and records[sequence] != record:  # 同一编号不能对应两次不同训练结果。
                        raise RuntimeError(f"损失序号重复且内容不一致：{sequence}")  # 防止把回滚前后的轨迹错误合并。
                    records[sequence] = record  # 保存安全边界内的记录。
    ordered = [records[key] for key in sorted(records)]  # 恢复实际发生顺序。
    if [record["sequence"] for record in ordered] != list(range(1, limit + 1)):  # 已提交轨迹必须连续，正游标对应空日志也视为缺失。
        raise RuntimeError("已提交损失日志存在缺行，暂时不能画正式曲线。")  # 显式报告而不插值伪造缺失损失。
    return ordered  # 返回可安全用于报告和绘图的真实记录。

class LossJournal:  # 为训练器提供逐批记录及与断点共同提交的损失日志。
    def __init__(self, run_dir, committed_sequence=None):  # 恢复后从最近完整训练态的损失游标继续记录。
        self.root = Path(run_dir)  # 保存当前实验唯一的永久输出目录。
        self.root.mkdir(parents=True, exist_ok=True)  # 创建实验目录但不覆盖任何既有结果。
        self.limit = _pointer_sequence(self.root) if committed_sequence is None else int(committed_sequence)  # 固定恢复时的日志安全边界。
        self.records = read_committed_records(self.root, self.limit)  # 加载已提交损失并丢弃已回滚尾部。
        self.sequence = self.limit  # 下一行编号从最近正式编号之后开始。
        self.pending = []  # 在本地进程内积累每个 batch 的真实损失。
        self.manifest_path = self.root / "loss_manifest.json"  # 保存当前轨迹使用的段清单位置。
        old = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.is_file() else {}  # 读取旧清单用于恢复。
        self.segments = [{**item, "last_sequence": min(int(item["last_sequence"]), self.limit)} for item in old.get("segments", []) if int(item["first_sequence"]) <= self.limit]  # 保留安全段并明确截断旧轨迹尾部，避免重训后编号冲突。

    def append(self, split, step, losses, **metadata):  # 逐个实际 batch 记录对应集合及所有计算出的损失项。
        if split not in {"train", "development", "validation", "test"}:  # 集合名称必须明确，避免把测试曲线画成验证曲线。
            raise ValueError(f"不支持的损失集合：{split}")  # 提示维护者使用真实集合名称。
        values = {}  # 为当前批次建立纯数值损失字典。
        for name, value in losses.items():  # 保留训练器本批次实际提供的每一种损失。
            number = float(value)  # 将已脱离计算图的标量统一转换为普通浮点数。
            if not math.isfinite(number):  # 非有限值不能继续发布为有效训练结果。
                raise FloatingPointError(f"{split} step {step} 的 {name} 不是有限数：{number}")  # 立即报告异常损失项。
            values[str(name)] = number  # 保存当前损失项的真实值。
        self.sequence += 1  # 为当前记录分配新的连续编号。
        record = {"sequence": self.sequence, "split": split, "global_step": int(step), "time_unix": time.time(), "losses": values, "metadata": metadata}  # 保存步数、集合、损失及附加统计。
        self.pending.append(record)  # 把当前真实记录加入等待断点提交的日志尾部。
        return self.sequence  # 返回编号供训练器确认日志游标。

    def commit(self, committed_step):  # 在完整 batch 训练态保存前发布该游标内的真实日志段。
        accepted = [record for record in self.pending if int(record["global_step"]) <= int(committed_step)]  # 只提交不晚于完整训练态的记录。
        if not accepted:  # 本轮没有新日志时直接返回上一正式边界。
            return self.limit  # 避免写入空日志段。
        first, last = accepted[0]["sequence"], accepted[-1]["sequence"]  # 记录当前段的连续编号范围。
        relative = Path("loss_segments") / f"segment_{first:09d}_{last:09d}_{uuid.uuid4().hex}.jsonl"  # 新段使用唯一名称，保留回滚前段作为可审计旧结果。
        target = self.root / relative  # 定位日志段的永久路径。
        target.parent.mkdir(parents=True, exist_ok=True)  # 创建日志段专用目录。
        temporary = target.with_name(target.name + ".writing")  # 当前段先写入明确的临时路径。
        with temporary.open("w", encoding="utf-8") as handle:  # 顺序写入已完成批次的日志。
            for record in accepted:  # 每个 batch 或验证批次独占一行。
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")  # 写入真实损失并禁止非法数值。
            handle.flush()  # 刷新 Python 文本缓冲。
            os.fsync(handle.fileno())  # 要求挂载文件系统接收当前段字节。
        os.replace(temporary, target)  # 发布完整日志段。
        self.segments.append({"path": relative.as_posix(), "first_sequence": first, "last_sequence": last, "bytes": target.stat().st_size, "sha256": _sha256(target)})  # 将当前段纳入轨迹清单。
        atomic_json(self.manifest_path, {"format_version": 1, "last_sequence": last, "segments": self.segments})  # 原子发布新日志清单。
        self.records.extend(accepted)  # 更新当前进程的已提交记录。
        self.pending = [record for record in self.pending if record["sequence"] > last]  # 仅保留尚未纳入完整训练态的尾部。
        self.limit = last  # 保存新的正式日志边界。
        return last  # 训练器必须把该值共同保存进完整断点。

def plot_losses(run_dir, committed_sequence=None):  # 按真实集合和真实损失项分别绘制已提交曲线。
    records = read_committed_records(run_dir, committed_sequence)  # 自动排除已回滚尾部记录。
    if not records:  # 尚未完成任何完整训练态时不生成误导性空曲线。
        print(f"当前尚无已提交损失：{run_dir}")  # 向用户说明训练状态。
        return []  # 返回空文件清单。
    import matplotlib  # 只在需要绘图时导入图形库。
    matplotlib.use("Agg")  # 在 Colab 后台使用无需图形窗口的输出模式。
    import matplotlib.pyplot as plt  # 用标准绘图工具保存学术报告可用曲线。
    root = Path(run_dir) / "loss_plots"  # 将所有损失图放入当前实验永久目录。
    root.mkdir(parents=True, exist_ok=True)  # 创建曲线输出目录。
    names = sorted({name for record in records for name in record["losses"]})  # 找出本训练器实际记录的所有损失项。
    rows = [{"sequence": item["sequence"], "split": item["split"], "global_step": item["global_step"], **item["losses"]} for item in records]  # 为全部真实逐批记录建立表格行。
    csv_path = root / "all_losses.csv"  # 保存可用其他软件重新作图的完整数值表。
    with csv_path.open("w", encoding="utf-8", newline="") as handle:  # 按标准 CSV 格式写入损失数值。
        writer = csv.DictWriter(handle, fieldnames=["sequence", "split", "global_step"] + names)  # 固定表头并允许不存在的损失留空。
        writer.writeheader()  # 保存字段名称供用户查看。
        writer.writerows(rows)  # 保存每一条真实批次损失。
    outputs = [str(csv_path)]  # 首先登记完整损失表。
    for name in names:  # 为每个真实损失项独立绘制所有实际存在的集合曲线。
        figure, axes = plt.subplots(figsize=(9, 4.8))  # 使用清晰的白底标准坐标轴。
        for split in ("train", "development", "validation", "test"):  # 固定图例顺序以保持各实验可比较。
            items = [item for item in records if item["split"] == split and name in item["losses"]]  # 只绘制确实计算过该损失的记录。
            if items:  # 当前集合没有该损失时不伪造对应曲线。
                axes.plot([item["global_step"] for item in items], [item["losses"][name] for item in items], label=split, linewidth=0.9, alpha=0.8, marker="o" if split != "train" else None, markersize=3)  # 保留原始数值，并让仅评估一次的测试点也清晰可见。
        axes.set_xlabel("Global optimizer step")  # 横轴显示累计优化更新次数。
        axes.set_ylabel(name)  # 纵轴使用训练器真实损失名称。
        axes.set_title(name)  # 标题直接注明对应损失。
        axes.grid(alpha=0.18)  # 使用淡网格帮助读取数值。
        axes.legend()  # 明确标出训练、开发、验证和测试集合。
        figure.tight_layout()  # 避免标签和坐标轴被裁切。
        safe_name = "".join(character if character.isalnum() or character in "_-" else "_" for character in name)  # 防止损失名产生非法文件路径。
        path = root / (safe_name + ".png")  # 为当前损失设置稳定输出文件名。
        figure.savefig(path, dpi=180)  # 保存清晰且可直接放入报告的图片。
        plt.close(figure)  # 主动释放图形内存。
        outputs.append(str(path))  # 登记当前已经实际输出的图。
    atomic_json(root / "plot_receipt.json", {"committed_sequence": records[-1]["sequence"], "records": len(records), "files": outputs})  # 保存曲线对应的正式日志边界。
    print(f"✓ 损失 CSV 和 {len(names)} 张曲线已保存：{root}")  # 告知用户实际产物数量和永久位置。
    return outputs  # 返回所有真实生成的文件位置。
