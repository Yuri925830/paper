#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 此入口只合并已经计算好的 DNSMOS 分片；当前所有分片计算仍限定在租用服务器。
"""合并三台 CPU 服务器各自计算的 DNSMOS CSV，并严格检查 8,100 对没有漏项或重复。"""

# argparse 读取三个分片路径；csv 读写表格；os.replace 完成原子替换。
import argparse
# 使用 csv 读写带表头的逐条记录。
import csv
# 使用 os 执行路径、原子替换和进程级辅助操作。
import os

# 统计配对编号重复次数，查出分片重叠。
from collections import Counter

# Path 管理文件路径；List/Dict 只是帮助读代码时看懂容器内容。
from pathlib import Path
# 标明张量、清单和配置的输入输出类型，不改变运行数据。
from typing import Dict, List


# 这五列必须和 evaluation/evaluate_metrics.py 完全相同。
FIELDNAMES = ["pair_id", "p808_mos", "mos_sig", "mos_bak", "mos_ovr"]


# 读取运行路径、设备和诊断选项，默认值保持原实验约定。
def parse_args() -> argparse.Namespace:
    # 读取运行路径、设备和诊断选项，默认值保持原实验约定。
    """要求显式给出分片和正式目标，防止把结果误写到别的实验目录。"""

    # 设置命令行参数解析器。
    parser = argparse.ArgumentParser(description=__doc__)
    # 显式列出至少一份待合并 DNSMOS 分片路径。
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    # 必须明确指定合并后的目标 CSV，避免猜测正式实验目录。
    parser.add_argument("--output", type=Path, required=True)
    # 返回已经校验参数格式的运行选项。
    return parser.parse_args()


# 执行本脚本的输入检查、主要处理和结果记录。
def main() -> None:
    # 执行本脚本的输入检查、主要处理和结果记录。
    """读取、验证、按 pair_id 排序、原子写正式 dnsmos.csv。"""

    # 设置命令行选项。
    args = parse_args()
    # 收集所有分片的原始逐配对分数，不改写分数值。
    rows: List[Dict[str, str]] = []

    # 逐个读取指定分片；当前项目所有计算仍只允许在租用服务器执行。
    for shard in args.shards:
        # 确定当前分片的绝对路径。
        shard = shard.resolve()
        # 任何分片缺失都会中止合并。
        if not shard.is_file():
            # 报错原因：任何分片缺失都会中止合并。
            raise FileNotFoundError(f"DNSMOS shard not found: {shard}")
        # 按 UTF-8 CSV 读取一个分片。
        with shard.open("r", encoding="utf-8", newline="") as handle:
            # 用列名对应 DNSMOS 的四种输出。
            reader = csv.DictReader(handle)
            # 列名及顺序必须与正式评价脚本一致。
            if reader.fieldnames != FIELDNAMES:
                # 报错原因：列名及顺序必须与正式评价脚本一致。
                raise ValueError(
                    # 同时显示当前表头与规定列名，帮助发现选错分片。
                    f"Unexpected columns in {shard}: {reader.fieldnames}; expected {FIELDNAMES}"
                )
            # 合并分片中的全部原始分数，不重新计算或改变数值。
            rows.extend(reader)

    # pair_id 转整数后排序，服务器完成顺序不影响最终正式 CSV 的固定顺序。
    rows.sort(key=lambda row: int(row["pair_id"]))
    # 提取排序后的配对编号。
    ids = [int(row["pair_id"]) for row in rows]
    # 完整正式结果应当逐一覆盖编号 0 到 8099。
    expected = list(range(8100))
    # 存在缺号、重复或越界时拒绝发布。
    if ids != expected:
        # 列出预期存在但任何分片都没有的编号。
        missing = sorted(set(expected) - set(ids))
        # 列出由重叠分片造成的重复编号。
        duplicates = sorted(pair_id for pair_id, count in Counter(ids).items() if count > 1)
        # 报错原因：列出由重叠分片造成的重复编号。
        raise ValueError(
            # 列出由重叠分片造成的重复编号。
            f"DNSMOS shards are not exactly pair_id 0..8099; "
            # 报告总条数和前 20 个缺失、重复编号，便于补算。
            f"rows={len(rows)}, missing_first20={missing[:20]}, duplicates_first20={duplicates[:20]}"
        )

    # 将合并目标解析为明确绝对路径。
    output = args.output.resolve()
    # 建立目标 CSV 所在目录。
    output.parent.mkdir(parents=True, exist_ok=True)
    # 先写同目录临时文件，防止半份合并结果被误读。
    temporary = output.with_suffix(output.suffix + ".tmp")
    # 创建完整合并 CSV 的临时写入器。
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        # 固定输出的编号与四个 DNSMOS 列。
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        # 写出固定列名，让后续续算和汇总能核对格式。
        writer.writeheader()
        # 按配对编号顺序写出所有原始分数。
        writer.writerows(rows)
    # 全部条目完成后原子发布到正式路径。
    os.replace(temporary, output)

    # 报告实际合并了多少份分片。
    print(f"merged_shards={len(args.shards)}")
    # 报告最终条目数，核对为 8,100。
    print(f"rows={len(rows)}")
    # 显示最终 CSV 的实际保存位置。
    print(f"output={output}")
    # 只在编号完整性检查和写入成功后输出通过标记。
    print("DNSMOS_8100_MERGE_PASSED")


# 仅作为脚本直接运行时执行入口，导入模块时不启动任务。
if __name__ == "__main__":
    # 启动本脚本的完整处理流程。
    main()
