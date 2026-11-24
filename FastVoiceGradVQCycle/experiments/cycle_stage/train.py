"""从完整学生12号断点公平分叉 Base、Mid-Cycle 与 Post-Cycle 实验。"""  # 本入口只在显式授权参数存在时训练。
import argparse  # 读取计划路径和人工训练授权开关。
import fcntl  # 给同一实验加非阻塞文件锁，防止两个 Colab 单元重复启动。
import hashlib  # 核验父断点、源码和运行合同的 SHA256 身份。
import json  # 读取实验计划并保存合同、状态和损失窗口。
import math  # 检查损失和优化器游标是否为有限数。
import os  # 原子替换状态文件并记录当前进程信息。
import random  # 按绝对训练步数重建学生12号的批次顺序。
import shutil  # 检查 Colab 本地盘和 Google Drive 的可用空间。
import signal  # 在用户停止时等当前完整批次结束后保存断点。
import sys  # 设置历史工具目录的导入路径并读取解释器信息。
import threading  # 在长达数分钟的完整断点上传期间用后台线程持续刷新Drive租约心跳。
import time  # 记录训练耗时和估算剩余时间。
from dataclasses import replace  # 从冻结的学生配置派生阶段实验配置。
from pathlib import Path  # 安全处理 Colab、Drive 和项目绝对路径。

import numpy as np  # 精确比较恢复前后的 NumPy 随机状态数组。
from tqdm.auto import tqdm  # 为每个训练batch显示可重连的百分比、速度和预计剩余时间。

ROOT = Path(__file__).resolve().parents[2]  # 从当前阶段模块位置解析恢复项目根，允许部署到分配给임서的数据盘目录。
TOOLS = ROOT / "exp/diagnostics/student_performance_tools_20260906"  # 指向学生12号原训练辅助代码目录。
DEVELOPMENT = ROOT / "exp/diagnostics/student_development_20260906/development_contract.json"  # 指向固定52句开发排除协议。
DEVELOPMENT_SHA256 = "08737551d34e00ba0b80a5e34adbf73fcfc6c70ece6bc8dc4a6c466e9b87afb1"  # 锁定学生12号固定52句排除协议的真实文件身份。
DEVELOPMENT_MANIFEST = DEVELOPMENT.parent / "development_manifest.json"  # 指向固定开发划分的完整发布收据。
DEVELOPMENT_MANIFEST_SHA256 = "4b530a15caa2804f817437870e91a0da1fed843f0f9ec0666f8f0bc1e8064cb3"  # 锁定开发划分收据，防止同步替换划分和哈希。
DRIVE_MOUNT = Path("/content/drive")  # 指向 Colab 官方 Google Drive 挂载点。
DRIVE_RUN_ROOT = DRIVE_MOUNT / "MyDrive/FsatVoiceGrad_服务器原文件_20260906/FastVoiceGrad_Cycle阶段实验运行记录_20261005"  # 限定所有可恢复阶段断点只能写入用户指定 Drive 目录。
DATA_RESTORE_RECEIPT = ROOT / ".essential_data_restore_identity.json"  # 指向准备 Notebook 生成的必要数据快速恢复身份收据。
POST_PARENT_PATH = ROOT / "exp/diagnostics/student_development_20260906/acdd_mc1_longtrain_from3327_v1/latest_full.pt"  # 硬锁已经确认的学生12号100-pass完整父断点路径。
POST_PARENT_SHA256 = "2ca44269eefac2dd3f8ffa7afe0337f1f3684469aec9fadeff623237cf37b830"  # 硬锁已经确认的122900步完整父断点字节身份。
MINIMUM_LOCAL_FREE_BYTES = 4 * 1024**3  # 本地至少保留四 GiB，确保清理旧暂存后仍可安全写入约1.4GB完整断点。
MINIMUM_DRIVE_FREE_BYTES = 4 * 1024**3  # Drive 至少保留四 GiB，容纳旧完整断点、原子临时文件和日志。
DRIVE_LEASE_STALE_SECONDS = 30 * 60  # Drive 租约连续三十分钟没有心跳后才允许新运行时接管，兼顾断线恢复和并发安全。
DRIVE_LEASE_HEARTBEAT_SECONDS = 60  # 后台线程每六十秒刷新一次租约，远短于三十分钟接管阈值。
LOSS_CHUNK_UPDATES = 200  # 每两百个batch原子提交一块逐batch损失，并在任何完整断点前额外强制提交以减少Drive小文件开销。
BATCH_SIZE = 32  # 学生12号每次更新固定使用32条语音。
UPDATES_PER_PASS = 1229  # 排除52句开发数据后每个完整 pass 恰有1229个完整批次。
TORCH_NUM_THREADS = 2  # 完全沿用学生12号长期训练入口显式设置的CPU线程数。
STUDENT_LR = 5e-5  # 学生12号真实有效学习率来自断点记录，不采用配置模板中的2e-4。
DISCRIMINATOR_LR = 2e-4  # 判别器继续使用学生12号断点中的真实有效学习率。
PER_UPDATE_SEED_OFFSET = 201234  # 每个正式更新都按这个固定偏移加绝对步数重建Python、NumPy、CPU和CUDA随机流。
LOSS_KEYS = ("student_total", "student_adv", "fm_raw", "fm_weighted", "dist_raw", "dist_weighted", "cycle_raw", "cycle_weighted", "discriminator")  # 固定九项阶段训练损失字段。
PARENT_KIND = "acdd_mc1_longtrain_from3327_full_v1"  # 接受已经恢复的122900步学生12号完整断点类型。
CHILD_KIND = "fastvoicegrad_cycle_stage_v2_full"  # 为本入口保存的可恢复完整断点使用独立类型。
FORMAT_VERSION = 2  # 本入口的计划、合同和断点共同使用第二版格式。

if str(ROOT) not in sys.path:  # 避免重复插入项目根目录。
    sys.path.insert(0, str(ROOT))  # 让 Python 从恢复后的项目导入模型和训练公式。
if str(TOOLS) not in sys.path:  # 避免重复插入学生12号工具目录。
    sys.path.insert(0, str(TOOLS))  # 让 Python 导入原来的数据批处理辅助函数。

import torch  # PyTorch 只会在用户显式运行本训练入口时加载。
import fastvoicegrad.train as training  # 复用已经实现 Cycle Loss 的真实 ACDD 单批更新。
import run_acdd_development as baseline  # 复用学生12号的模型构建、数据读取和固定裁剪方法。


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:  # 以固定小内存分块计算任意大文件的 SHA256。
    digest = hashlib.sha256()  # 为当前文件创建新的哈希计算器。
    with Path(path).open("rb") as handle:  # 以二进制只读方式打开文件。
        while True:  # 持续读取直到文件末尾。
            block = handle.read(chunk_size)  # 每次最多读取八 MiB，避免一次载入1.4GB断点。
            if not block:  # 空数据表示已经读到文件末尾。
                break  # 退出分块读取循环。
            digest.update(block)  # 把当前数据块加入哈希计算。
    return digest.hexdigest()  # 返回小写十六进制 SHA256。


def canonical_sha256(value: object) -> str:  # 对结构化合同生成与键顺序无关的稳定哈希。
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # 生成唯一规范 JSON 字节。
    return hashlib.sha256(encoded).hexdigest()  # 返回规范合同的内容身份。


def read_json(path: Path) -> dict:  # 读取必须为对象的 UTF-8 JSON 文件。
    value = json.loads(Path(path).read_text(encoding="utf-8"))  # 解析计划或合同文本。
    if not isinstance(value, dict):  # 阶段计划和合同都必须是键值对象。
        raise RuntimeError(f"JSON 顶层必须是对象：{path}")  # 拒绝数组或标量造成的含糊配置。
    return value  # 返回已经确认的字典。


def write_json_atomic(path: Path, value: object) -> None:  # 用同目录临时文件原子提交小型状态 JSON。
    path = Path(path)  # 统一将调用方输入转换为 Path。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建目标文件所需的父目录。
    temporary = path.with_name(path.name + ".partial")  # 使用不会被误认成正式结果的临时文件名。
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # 一次写完可读 JSON 文本。
    os.replace(temporary, path)  # 写完后原子替换正式文件。


def append_json_line(path: Path, value: object) -> None:  # 追加一条紧凑 JSON 记录并立即刷新。
    path = Path(path)  # 统一路径类型以便创建父目录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 首次写日志时创建实验目录。
    with path.open("a", encoding="utf-8") as handle:  # 以追加模式保留此前所有窗口记录。
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")  # 每条记录独占一行便于断点读取。
        handle.flush()  # 立即把 Python 缓冲区交给文件系统。


def append_json_lines(path: Path, values: list) -> None:  # 把一批逐batch损失作为独立原子日志块提交到Drive，兼顾可恢复性和训练速度。
    if not values:  # 没有尚未持久化的batch记录时不执行任何Drive写入。
        return  # 直接返回并保留调用方正常训练流程。
    path = Path(path)  # 统一路径类型以便创建本次尝试目录。
    chunk_root = path.parent / "batch_loss_chunks"  # 使用独立小文件避免虚拟机中断在单个大JSONL末尾留下残行。
    chunk_root.mkdir(parents=True, exist_ok=True)  # 首次刷新逐batch日志时创建原子日志块目录。
    first_step = int(values[0]["global_step"])  # 读取当前日志块第一个已经完整返回的绝对训练步。
    last_step = int(values[-1]["global_step"])  # 读取当前日志块最后一个已经完整返回的绝对训练步。
    expected_steps = list(range(first_step, last_step + 1))  # 构造当前日志块必须覆盖的连续绝对步序列。
    actual_steps = [int(value["global_step"]) for value in values]  # 提取当前缓冲区真实保存的全部绝对步。
    if actual_steps != expected_steps:  # 同一日志块不允许缺步、重复或乱序。
        raise RuntimeError("逐batch损失缓冲区的global_step不连续。")  # 拒绝发布无法可靠合并的日志块。
    final_path = chunk_root / f"steps_{first_step:09d}_{last_step:09d}.jsonl"  # 用固定宽度起止步生成可排序且唯一的正式文件名。
    partial_path = final_path.with_name(final_path.name + ".partial")  # 在同一Drive目录先写不会被绘图读取的临时文件。
    if final_path.exists():  # 当前尝试内相同步数块理论上不应被重复提交。
        raise RuntimeError(f"逐batch损失日志块已经存在：{final_path}")  # 不覆盖既有记录以免掩盖控制流错误。
    if partial_path.exists():  # 同一进程上一次未完成的小块写入可能留下精确临时文件。
        partial_path.unlink()  # 只清理当前确定文件名的未提交临时块。
    with partial_path.open("x", encoding="utf-8") as handle:  # 排他创建临时文件并一次写入当前完整日志块。
        for value in values:  # 按真实完成顺序逐条写入当前缓冲区记录。
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")  # 每个batch独占一行并保存九项完整损失。
        handle.flush()  # 一次刷新整批小记录，避免每个batch单独访问Drive拖慢GPU。
    os.replace(partial_path, final_path)  # 全部行写完后原子发布正式日志块，中断只会留下可忽略的partial文件。


def semantic_plan(plan: dict) -> dict:  # 去掉每次重跑都会变化但不影响实验数学的创建时间。
    return {key: value for key, value in plan.items() if key != "created_at_utc"}  # 保留其他全部实验字段供合同锁定。


def validate_plan(plan: dict) -> dict:  # 对 Notebook 生成的实验计划做失败关闭式检查。
    required = {"format_version", "plan_name", "status", "parent_checkpoint", "parent_checkpoint_sha256", "parent_pass", "end_pass", "updates_per_pass", "cycle_enabled", "cycle_step_start", "cycle_start_relative_pass", "paired_group", "shared_parent_required", "data_protocol", "save_every_updates", "save_every_seconds", "status_every_updates", "local_run_dir", "persistent_run_dir"}  # 列出训练前不可缺少的预注册字段。
    missing = sorted(required - set(plan))  # 找出计划中缺失的必需字段。
    if missing:  # 任一字段缺失都不能靠默认值猜测。
        raise RuntimeError(f"实验计划缺少字段：{missing}")  # 报告全部缺失项后停止。
    if plan["format_version"] != FORMAT_VERSION:  # 只接受本入口明确支持的计划版本。
        raise RuntimeError(f"不支持的计划版本：{plan['format_version']}")  # 防止旧计划被错误解释。
    if plan["status"] != "READY":  # Notebook 会把缺少完整父断点的 Mid 计划标为阻塞。
        raise RuntimeError(f"计划当前不可运行：{plan.get('blocked_reason')}")  # 保留真实阻塞原因，不允许绕过。
    if plan["updates_per_pass"] != UPDATES_PER_PASS:  # 学生12号排除协议决定每轮必须是1229批。
        raise RuntimeError("计划的每 pass 更新数不是学生12号的1229。")  # 拒绝把1230批普通训练混入比较。
    if plan["data_protocol"] != "student12_longtrain_exact":  # 只允许学生12号固定排除和批次协议。
        raise RuntimeError("计划没有声明学生12号精确训练协议。")  # 防止切换成普通 DataLoader。
    if type(plan["parent_pass"]) is not int or type(plan["end_pass"]) is not int or not 0 <= plan["parent_pass"] < plan["end_pass"]:  # 核对阶段起止 pass 为合法递增整数。
        raise RuntimeError("计划的 parent_pass/end_pass 非法。")  # 非法范围无法定义公平分叉终点。
    allowed = {"base_from_50_to_100": (50, 100, False, "mid_50_to_100"), "mid_cycle_from_50_to_100": (50, 100, True, "mid_50_to_100"), "base_from_100_to_150": (100, 150, False, "post_100_to_150"), "post_cycle_from_100_to_150": (100, 150, True, "post_100_to_150")}  # 只开放论文设计中预注册的两组严格配对实验。
    actual_identity = (plan["parent_pass"], plan["end_pass"], plan["cycle_enabled"], plan["paired_group"])  # 汇总当前计划不可自由改写的实验身份。
    if plan["plan_name"] not in allowed or actual_identity != allowed[plan["plan_name"]]:  # 名称、区间、Cycle开关和配对组必须相互一致。
        raise RuntimeError("计划不属于预注册的四个阶段实验之一。")  # 防止临时改轮数后仍沿用正式实验名称。
    expected_start = plan["parent_pass"] * UPDATES_PER_PASS  # 根据学生12号协议计算父断点绝对步数。
    if plan.get("expected_start_global_step") != expected_start:  # Notebook 写入的步数必须与 pass 计算一致。
        raise RuntimeError("计划的父 pass 与起始 global_step 不一致。")  # 拒绝标签和真实训练量不一致。
    if type(plan["cycle_enabled"]) is not bool or plan["cycle_step_start"] != 0:  # Cycle 必须显式布尔并把已完成更新游标从0开始。
        raise RuntimeError("计划必须明确 cycle_enabled，且 cycle_step_start 必须为0。")  # 防止用绝对步数改变 Cycle 随机序列。
    expected_cycle_start = 1 if plan["cycle_enabled"] else None  # Cycle分支从阶段第一轮启用，Base分支没有Cycle门控轮次。
    if plan["cycle_start_relative_pass"] != expected_cycle_start or plan["shared_parent_required"] is not True:  # 核对Cycle门控和共享父断点声明。
        raise RuntimeError("计划的 Cycle 起始轮次或共享父断点声明不符。")  # 不允许把公平配对设计改成其他实验。
    if plan["save_every_updates"] != 2000 or plan["save_every_seconds"] != 1800 or plan["status_every_updates"] != 20:  # 锁定按步数、按时间和轻量状态频率，保证四分支一致。
        raise RuntimeError("计划的断点或状态保存频率不符合预注册值。")  # 防止只对某一分支改变I/O节奏。
    parent = Path(str(plan["parent_checkpoint"]))  # 将完整父断点路径转换为 Path。
    if not parent.is_absolute() or not parent.is_file() or parent.is_symlink():  # 父断点必须是存在的绝对普通文件。
        raise FileNotFoundError(f"完整父断点不存在或不是普通文件：{parent}")  # 不接受权重-only别名或软链接替代。
    expected_sha = str(plan["parent_checkpoint_sha256"]).lower()  # 规范化计划中登记的父断点哈希。
    if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):  # SHA256 必须是64位小写十六进制。
        raise RuntimeError("父断点 SHA256 格式非法。")  # 没有完整身份就不能公平分叉。
    if plan["parent_pass"] == 100 and (parent != POST_PARENT_PATH or expected_sha != POST_PARENT_SHA256):  # Post两臂只能从已经确认的同一个学生12号父状态分叉。
        raise RuntimeError("Post 计划的父断点路径或 SHA256 不是已确认的学生12号100-pass完整状态。")  # 防止误改计划后混入另一条同样122900步的运行。
    persistent = Path(str(plan["persistent_run_dir"]))  # 将 Drive 持久输出目录转换为 Path。
    if not persistent.is_absolute() or str(persistent) in {"/", "/content", "/content/drive", "/root", str(ROOT)}:  # 拒绝过宽或危险的输出根目录。
        raise RuntimeError(f"持久输出目录不安全：{persistent}")  # 防止状态文件写到项目或磁盘根。
    if DRIVE_RUN_ROOT not in persistent.parents or persistent.parent != DRIVE_RUN_ROOT:  # 每个计划必须直接位于用户指定的阶段运行记录目录下。
        raise RuntimeError(f"持久输出目录不在指定 Google Drive 根目录：{persistent}")  # 防止断挂时误写本地同名目录或写进其他项目。
    local = Path(str(plan["local_run_dir"]))  # 读取 Notebook 预留的当前计划本地高速目录。
    if not local.is_absolute() or local.parent != ROOT / "exp/cycle_stage_runs":  # 本地目录只能是专用总目录的直接子目录。
        raise RuntimeError(f"本地运行目录不安全：{local}")  # 防止未来暂存文件覆盖模型或项目根。
    if local.name != plan["plan_name"] or persistent.name != plan["plan_name"]:  # 两个输出目录必须与实验计划名完全相同。
        raise RuntimeError("本地或持久目录名称与计划名称不一致。")  # 防止不同计划共享输出目录。
    plan["_parent_path"] = parent  # 保存经过验证的父断点 Path 供主流程使用。
    plan["_local_path"] = local  # 保存经过验证的本地高速目录供运行时空间检查。
    plan["_persistent_path"] = persistent  # 保存经过验证的持久输出 Path 供主流程使用。
    plan["_target_global_step"] = plan["end_pass"] * UPDATES_PER_PASS  # 计算并缓存阶段终点绝对步数。
    return plan  # 返回补充了安全派生字段的计划。


class GlobalBatchPlan:  # 按绝对步数重建学生12号每一轮的固定随机打乱顺序。
    def __init__(self, indices: list, seed: int = 1234) -> None:  # 保存39335条保留索引和原始打乱种子。
        self.indices = list(indices)  # 复制索引，避免调用方后来修改原列表。
        self.seed = int(seed)  # 规范化原学生12号批次种子。
        self.batches_per_pass = len(self.indices) // BATCH_SIZE  # 用整除丢弃每轮最后7条不足批样本。
        self.pass_index = None  # 新进程尚未重建任何轮次顺序。
        self.order = None  # 当前轮次的打乱索引将在第一次请求时生成。
        if self.batches_per_pass != UPDATES_PER_PASS:  # 39335条保留数据必须恰好形成1229个完整批。
            raise RuntimeError(f"学生12号批次数变化：{self.batches_per_pass}")  # 数据人口不一致时停止。

    def for_step(self, global_step: int) -> list:  # 返回指定一基绝对更新步对应的32条数据索引。
        if type(global_step) is not int or global_step < 1:  # 全局更新步从1开始且必须是普通整数。
            raise ValueError("global_step 必须是一基正整数。")  # 拒绝0、负数或浮点游标。
        current_pass = (global_step - 1) // self.batches_per_pass  # 计算当前步位于第几个零基 pass。
        if self.pass_index != current_pass:  # 跨 pass 或首次调用时需要重建顺序。
            self.order = list(self.indices)  # 每轮都从同一原始索引顺序开始。
            random.Random(self.seed + current_pass).shuffle(self.order)  # 用原规则独立打乱当前 pass。
            self.pass_index = current_pass  # 缓存当前 pass，避免每个 batch 重复打乱。
        offset = ((global_step - 1) % self.batches_per_pass) * BATCH_SIZE  # 计算当前 batch 在打乱列表中的起点。
        return self.order[offset : offset + BATCH_SIZE]  # 返回严格32条且不含尾部余数的索引。


def load_training_indices() -> list:  # 读取学生12号固定52句排除协议中的训练索引。
    if not DEVELOPMENT.is_file():  # 恢复目录必须包含原开发协议。
        raise FileNotFoundError(f"缺少学生12号开发排除协议：{DEVELOPMENT}")  # 缺少协议时不能退回普通全数据训练。
    if sha256_file(DEVELOPMENT) != DEVELOPMENT_SHA256:  # 固定协议必须逐字节等于学生12号训练时归档版本。
        raise RuntimeError("学生12号固定52句排除协议的 SHA256 不一致。")  # 不允许换另一组数量相同的52句。
    if not DEVELOPMENT_MANIFEST.is_file() or sha256_file(DEVELOPMENT_MANIFEST) != DEVELOPMENT_MANIFEST_SHA256:  # 同时核对完整发布收据的固定身份。
        raise RuntimeError("学生12号开发划分发布收据缺失或 SHA256 不一致。")  # 防止同步替换协议和自报哈希。
    receipt = read_json(DEVELOPMENT_MANIFEST)  # 读取已经固定身份的开发划分发布收据。
    expected_artifacts = {"development_contract.json", "pairs.csv", "self_reconstruction.jsonl", "student_train_manifest.csv", "dev_excluded_manifest.csv"}  # 固定构成该划分的五份发布文件。
    if receipt.get("status") != "complete" or set(receipt.get("artifacts", {})) != expected_artifacts:  # 收据必须声明完成且文件集合不多不少。
        raise RuntimeError("学生12号开发划分发布收据不完整。")  # 不从缺失配套证据的协议继续训练。
    for name, expected_sha in receipt["artifacts"].items():  # 逐份核对固定收据登记的配套文件。
        artifact = DEVELOPMENT.parent / name  # 将收据内安全的固定文件名定位到同一发布目录。
        if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != expected_sha:  # 文件必须是普通文件且内容身份一致。
            raise RuntimeError(f"学生12号开发划分配套文件损坏：{name}")  # 报告具体损坏文件并停止。
    contract = read_json(DEVELOPMENT)  # 读取已经归档的固定开发划分。
    frozen_sources = contract.get("source_sha256")  # 读取固定协议登记的原训练清单和归一化统计身份。
    if not isinstance(frozen_sources, dict):  # 缺少来源哈希时无法证明数据协议仍与学生12号相同。
        raise RuntimeError("学生12号固定协议缺少 source_sha256。")  # 不根据当前文件自行建立新的有利身份。
    for source in (ROOT / "data/metadata/train_manifest.csv", ROOT / "data/metadata/test_manifest.csv", ROOT / "data/stats/mel_mean.npy", ROOT / "data/stats/mel_std.npy"):  # 核对训练索引语义与mel归一化直接依赖的四份资产。
        archived_source = Path("/root/FsatVoiceGrad") / source.relative_to(ROOT)  # 用旧发布根拼出合同中保留的历史键，不改变文件内容身份。
        expected_source_sha = frozen_sources.get(str(archived_source))  # 从固定合同的历史绝对路径键读取该资产发布时哈希。
        if not source.is_file() or source.is_symlink() or expected_source_sha is None or sha256_file(source) != expected_source_sha:  # 路径、普通文件属性和内容都必须匹配。
            raise RuntimeError(f"学生12号固定数据资产发生变化：{source}")  # 拒绝在漂移的数据或统计量上继续对照实验。
    retained = contract.get("retained_train_indices")  # 取得原长期训练实际使用的索引顺序。
    excluded = contract.get("excluded_train_indices")  # 取得从梯度更新中排除的52句索引。
    if not isinstance(retained, list) or not isinstance(excluded, list):  # 两组索引都必须是显式列表。
        raise RuntimeError("开发协议缺少保留或排除索引列表。")  # 不根据其他文件重新推断划分。
    if len(retained) != 39335 or len(excluded) != 52:  # 学生12号固定人口分别是39335和52。
        raise RuntimeError(f"学生12号数据人口变化：retained={len(retained)}, excluded={len(excluded)}")  # 报告实际数量后停止。
    if any(type(index) is not int or not 0 <= index < 39387 for index in retained + excluded):  # 每个索引必须落在原39387行训练清单内。
        raise RuntimeError("开发协议包含非法训练索引。")  # 拒绝负数、浮点或越界索引。
    excluded_set = set(excluded)  # 只构建一次排除集合，后续顺序核验无需重复转换。
    if len(set(retained)) != len(retained) or len(excluded_set) != len(excluded) or set(retained) & excluded_set:  # 保留和排除集合必须各自唯一且互斥。
        raise RuntimeError("学生12号保留与排除索引发生重复或交叉。")  # 防止开发样本重新进入训练。
    expected_retained = [index for index in range(39387) if index not in excluded_set]  # 按固定训练清单顺序重建唯一合法的保留索引列表。
    if retained != expected_retained:  # 不只核对集合，还要锁定进入每轮打乱前的原始顺序。
        raise RuntimeError("学生12号保留索引的集合或原始顺序发生变化。")  # 顺序变化会改变相同随机种子的批次内容。
    return retained  # 保留原列表顺序并交给固定批次规划器。


def require_runtime() -> torch.device:  # 核对当前确实是用户授权的 Colab CUDA 运行环境。
    if not sys.platform.startswith("linux") or not ROOT.is_dir():  # 本地 Mac 和未恢复项目的运行时都不允许训练。
        raise RuntimeError("阶段训练只允许在已恢复 /root/FsatVoiceGrad 的 Linux Colab 中运行。")  # 明确拒绝本地误启动。
    if not DRIVE_MOUNT.is_dir() or not os.path.ismount(DRIVE_MOUNT) or not (DRIVE_MOUNT / "MyDrive").is_dir():  # 正式训练前确认官方Drive挂载点仍然真实存在。
        raise RuntimeError("Google Drive 没有真实挂载；拒绝把持久断点误写到临时本地目录。")  # Drive断挂时立即停止而不是制造会随会话消失的文件。
    if not DRIVE_RUN_ROOT.is_dir():  # 准备 Notebook 应当已经创建阶段实验运行记录目录。
        raise FileNotFoundError(f"缺少 Drive 阶段运行目录：{DRIVE_RUN_ROOT}")  # 目录缺失时要求重新运行准备 Notebook。
    local_free = shutil.disk_usage(ROOT).free  # 读取 Colab 本地项目所在磁盘的实际剩余空间。
    drive_free = shutil.disk_usage(DRIVE_RUN_ROOT).free  # 读取 Google Drive 挂载目录报告的实际剩余空间。
    if local_free < MINIMUM_LOCAL_FREE_BYTES or drive_free < MINIMUM_DRIVE_FREE_BYTES:  # 两边空间都必须满足完整断点的安全余量。
        raise RuntimeError(f"存储空间不足：本地剩余{local_free}字节，Drive剩余{drive_free}字节。")  # 防止原子保存临时文件写到一半失败。
    if not torch.cuda.is_available():  # FastVoiceGrad 阶段训练必须有可用 CUDA GPU。
        raise RuntimeError("当前 Colab 没有可用 CUDA GPU；不要退回 CPU 训练。")  # CPU 会极慢且不符合计划。
    device = torch.device("cuda")  # 使用当前 Colab 分配的唯一 CUDA 设备。
    properties = torch.cuda.get_device_properties(device)  # 读取真实 GPU 名称和总显存供合同记录。
    if properties.total_memory < 15 * 1024**3:  # 低于约15GiB时两程 Cycle 很可能无法容纳原批量。
        raise RuntimeError(f"GPU 显存不足15GiB：{properties.name}，{properties.total_memory}字节。")  # 条件不足时只准备不训练。
    return device  # 返回已经确认的 CUDA 设备。


def runtime_identity(device: torch.device) -> dict:  # 记录并锁定配对实验真正使用的软件与GPU数学环境。
    properties = torch.cuda.get_device_properties(device)  # 读取当前可见GPU的型号、显存和计算能力。
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu_name": properties.name, "gpu_total_memory": properties.total_memory, "compute_capability": list(torch.cuda.get_device_capability(device)), "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(), "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())}  # 用纯JSON字段描述会影响数值轨迹的运行条件。


def optimizer_uniform_step(optimizer: torch.optim.Optimizer) -> int:  # 读取一个 Adam 对全部参数一致的真实更新游标。
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]  # 保留优化器登记顺序展开所有参数。
    if not parameters or len({id(parameter) for parameter in parameters}) != len(parameters):  # 参数列表必须非空且不能重复拥有同一对象。
        raise RuntimeError("优化器参数所有权为空或重复。")  # 错误所有权会让游标检查失去意义。
    steps = []  # 收集每个参数对应的 Adam step。
    for parameter in parameters:  # 逐个检查所有受该优化器控制的参数。
        state = optimizer.state.get(parameter)  # 读取 PyTorch 已映射到真实参数对象的状态。
        if not isinstance(state, dict) or "step" not in state:  # 已训练完整断点的每个参数都必须有 Adam 状态。
            raise RuntimeError("完整断点缺少某个参数的 Adam step。")  # 不允许把无动量参数当作完整恢复。
        value = float(state["step"])  # 将标量张量或数字统一转成 Python 浮点数。
        if not math.isfinite(value) or value != int(value):  # Adam 游标必须是有限整数。
            raise RuntimeError("Adam step 不是有限整数。")  # 拒绝 NaN、无穷或分数游标。
        steps.append(int(value))  # 保存当前参数的整数更新次数。
    if len(set(steps)) != 1:  # 同一模型全部参数必须完成相同次数的更新。
        raise RuntimeError("同一优化器内部的 Adam step 不一致。")  # 混合游标无法定义严格续训起点。
    return steps[0]  # 返回唯一的共同更新步数。


def validate_optimizer_hyperparameters(optimizer: torch.optim.Optimizer, expected_lr: float, role: str) -> None:  # 核对加载后真实 Adam 超参数。
    if len(optimizer.param_groups) != 1:  # 学生12号的学生和判别器各只有一个参数组。
        raise RuntimeError(f"{role} Adam 参数组数量变化。")  # 多组学习率会改变原训练解释。
    group = optimizer.param_groups[0]  # 读取唯一参数组的实际运行设置。
    expected = {"lr": expected_lr, "betas": (0.5, 0.9), "eps": 1e-8, "weight_decay": 0, "amsgrad": False}  # 锁定学生12号真实优化器设置。
    for key, value in expected.items():  # 逐项核对学习率、动量和数值稳定设置。
        if group.get(key) != value:  # 任何一项不同都表示分叉后优化条件改变。
            raise RuntimeError(f"{role} Adam 的 {key}={group.get(key)!r}，期望 {value!r}。")  # 报告具体差异后停止。


def validate_optimizer_state(optimizer: torch.optim.Optimizer, model: torch.nn.Module, expected_lr: float, expected_step: int, role: str) -> None:  # 深入核对恢复后的 Adam 所有权、游标和动量张量。
    validate_optimizer_hyperparameters(optimizer, expected_lr, role)  # 先核对学习率、betas和其他超参数。
    model_parameters = list(model.parameters())  # 按模型登记顺序收集全部可训练参数。
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]  # 展开优化器实际拥有的参数对象。
    if {id(parameter) for parameter in optimizer_parameters} != {id(parameter) for parameter in model_parameters}:  # Adam必须且只能拥有当前模型的全部参数。
        raise RuntimeError(f"{role} Adam 的参数所有权与模型不一致。")  # 防止遗漏、额外或跨模型参数混入。
    if len(optimizer.state) != len(model_parameters):  # 完整训练态应当为每个模型参数保存一份Adam状态。
        raise RuntimeError(f"{role} Adam 状态数量与模型参数数量不一致。")  # 不用缺失动量的近似恢复冒充完整恢复。
    for parameter in model_parameters:  # 逐参数检查一阶、二阶动量和真实更新步数。
        state = optimizer.state.get(parameter)  # 读取映射到当前参数对象的恢复状态。
        if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= set(state):  # 标准Adam三项状态必须齐全。
            raise RuntimeError(f"{role} Adam 缺少 step、exp_avg 或 exp_avg_sq。")  # 缺项时停止而不是让PyTorch重新初始化。
        step_value = float(state["step"])  # 将标量张量或数字转换成可比较游标。
        if not math.isfinite(step_value) or step_value != expected_step:  # 每个参数都必须正好位于断点声明步数。
            raise RuntimeError(f"{role} Adam 参数游标与 global_step 不一致。")  # 防止部分参数少更新或多更新。
        for key in ("exp_avg", "exp_avg_sq"):  # 逐项检查一阶与二阶矩估计。
            moment = state[key]  # 读取当前动量张量。
            if not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape or moment.dtype != parameter.dtype or moment.device != parameter.device:  # 动量的形状、精度和设备必须与参数完全对应。
                raise RuntimeError(f"{role} Adam 的 {key} 形状、精度或设备不符。")  # 拒绝PyTorch可能广播或隐式转换的异常状态。
            if not bool(torch.isfinite(moment).all()):  # 任何NaN或无穷都会污染后续全部更新。
                raise RuntimeError(f"{role} Adam 的 {key} 含非有限数。")  # 在第一批训练前失败关闭。
        if not bool((state["exp_avg_sq"] >= 0).all()):  # Adam二阶矩按定义不能出现负值。
            raise RuntimeError(f"{role} Adam 的 exp_avg_sq 含负值。")  # 拒绝损坏或格式误读的状态。


def validate_model_state(model: torch.nn.Module, role: str) -> None:  # 核对模型全部参数和缓冲区仍为有限数。
    for name, value in model.state_dict().items():  # 逐项检查权重、偏置和归一化缓冲区。
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):  # 正式模型状态必须是有限张量。
            raise RuntimeError(f"{role} 模型状态含非有限数或非法对象：{name}")  # 在恢复或保存边界阻止坏状态继续传播。


def rng_state_equal(expected: object, actual: object) -> bool:  # 精确比较Python、NumPy、CPU与CUDA四类随机状态。
    if not isinstance(expected, dict) or not isinstance(actual, dict) or set(expected) != {"python", "numpy", "torch", "cuda_all"} or set(actual) != set(expected):  # 两边必须具有完全相同的固定键集合。
        return False  # 格式缺失或多出未知字段时直接判为不一致。
    if expected["python"] != actual["python"]:  # Python random 状态由纯整数元组构成，可直接精确比较。
        return False  # Python随机流位置不同即恢复失败。
    expected_numpy, actual_numpy = expected["numpy"], actual["numpy"]  # 读取NumPy的算法名、状态数组和游标元组。
    if len(expected_numpy) != len(actual_numpy) or expected_numpy[0] != actual_numpy[0] or not np.array_equal(expected_numpy[1], actual_numpy[1]) or expected_numpy[2:] != actual_numpy[2:]:  # NumPy状态数组和其余标量都必须逐项一致。
        return False  # NumPy随机流位置不同即恢复失败。
    if not torch.equal(expected["torch"].cpu(), actual["torch"].cpu()):  # CPU生成器状态按字节张量精确比较。
        return False  # PyTorch CPU随机流不同即恢复失败。
    if len(expected["cuda_all"]) != len(actual["cuda_all"]):  # 可见CUDA设备数量必须与保存状态列表一致。
        return False  # 设备状态数量不同无法声称完整恢复。
    return all(torch.equal(left.cpu(), right.cpu()) for left, right in zip(expected["cuda_all"], actual["cuda_all"]))  # 逐张GPU精确比较CUDA生成器状态。


def restore_stage_rng_state(payload: dict, device: torch.device, completed_step: int) -> dict:  # 对阶段断点严格恢复全部随机状态，并只为旧学生父断点兼容CUDA状态长度变化。
    saved_state = payload["rng_state"]  # 读取父断点或本阶段断点保存的四类随机数状态。
    checkpoint_kind = payload.get("kind")  # 依据断点类型区分必须精确恢复的本阶段状态与旧版学生父状态。
    required_keys = {"python", "numpy", "torch", "cuda_all"}  # 固定任何可接受随机状态必须具有的完整键集合。
    if not isinstance(saved_state, dict) or set(saved_state) != required_keys:  # 旧父断点也不能缺少任一随机状态类别。
        raise RuntimeError(f"完整随机状态字段不符：期望{sorted(required_keys)}。")  # 不允许用兼容逻辑掩盖残缺断点。
    if checkpoint_kind != PARENT_KIND:  # 本阶段自己保存的断点必须在合同锁定的同一后端逐字节精确恢复。
        training.restore_rng_state(saved_state, device)  # 恢复Python、NumPy、PyTorch CPU和当前CUDA生成器状态。
        if not rng_state_equal(saved_state, training.capture_rng_state()):  # 恢复后立即重新捕获并比较四类状态。
            raise RuntimeError("完整随机状态恢复后回读不一致。")  # 本阶段续训不开放任何近似或重建路径。
        exact_sizes = [int(state.numel()) for state in saved_state["cuda_all"]]  # 记录本阶段断点逐字节恢复的CUDA状态长度供续训审计。
        return {"mode": "exact_checkpoint_rng_state", "checkpoint_kind": checkpoint_kind, "saved_cuda_device_count": len(exact_sizes), "runtime_cuda_device_count": len(exact_sizes), "saved_cuda_state_sizes": exact_sizes, "runtime_cuda_state_sizes": exact_sizes}  # 记录本阶段断点已经完成严格全量恢复。
    random.setstate(saved_state["python"])  # 对旧学生父断点仍逐项精确恢复Python随机状态。
    np.random.set_state(saved_state["numpy"])  # 对旧学生父断点仍逐项精确恢复NumPy随机状态。
    torch.set_rng_state(saved_state["torch"])  # 对旧学生父断点仍逐项精确恢复PyTorch CPU随机状态。
    restored_non_cuda = training.capture_rng_state()  # 回读当前状态以核对前三类旧状态没有因兼容路径被重置。
    non_cuda_equal = saved_state["python"] == restored_non_cuda["python"] and len(saved_state["numpy"]) == len(restored_non_cuda["numpy"]) and saved_state["numpy"][0] == restored_non_cuda["numpy"][0] and np.array_equal(saved_state["numpy"][1], restored_non_cuda["numpy"][1]) and saved_state["numpy"][2:] == restored_non_cuda["numpy"][2:] and torch.equal(saved_state["torch"].cpu(), restored_non_cuda["torch"].cpu())  # 精确比较Python、NumPy与CPU生成器状态。
    if not non_cuda_equal:  # 兼容仅允许改变无法载入的旧CUDA字节格式。
        raise RuntimeError("旧父断点的非CUDA随机状态恢复后回读不一致。")  # 非CUDA状态有任何差异都停止。
    saved_cuda_states = saved_state["cuda_all"]  # 读取旧父断点保存的全部CUDA生成器字节张量。
    runtime_cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else []  # 读取当前后端期望的CUDA状态数量和字节长度。
    if not saved_cuda_states or not runtime_cuda_states:  # 正式CUDA训练的旧父状态和当前后端都必须至少包含一个生成器状态。
        raise RuntimeError("旧父断点或当前运行时缺少CUDA随机状态。")  # 缺失状态不是版本长度差异，不能兼容。
    if any(not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1 for state in saved_cuda_states):  # 旧CUDA状态必须仍是标准一维字节张量。
        raise RuntimeError("旧父断点CUDA随机状态类型或维度损坏。")  # 对象、精度或维度异常不是允许兼容的后端长度变化。
    if any(not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1 for state in runtime_cuda_states):  # 当前后端也必须提供标准一维字节状态。
        raise RuntimeError("当前运行时CUDA随机状态类型或维度异常。")  # 当前后端格式异常时不能建立可审计的替代状态。
    saved_sizes = [int(state.numel()) for state in saved_cuda_states]  # 记录旧后端每张GPU随机状态的实际字节数。
    runtime_sizes = [int(state.numel()) for state in runtime_cuda_states]  # 记录当前后端每张GPU随机状态要求的实际字节数。
    exact_cuda_compatible = len(saved_cuda_states) == len(runtime_cuda_states) and all(saved.numel() == current.numel() for saved, current in zip(saved_cuda_states, runtime_cuda_states))  # 已确认类型与维度合法后，只有设备数和长度完全相同时才逐字节恢复。
    if exact_cuda_compatible:  # 旧父断点碰巧来自兼容CUDA生成器格式时保留原始状态。
        torch.cuda.set_rng_state_all(saved_cuda_states)  # 在已经确认尺寸兼容后恢复旧CUDA生成器字节。
        if not rng_state_equal(saved_state, training.capture_rng_state()):  # 四类状态都必须与断点逐项完全一致。
            raise RuntimeError("旧父断点的兼容CUDA随机状态恢复后回读不一致。")  # 接口无报错但内容不同仍然停止。
        return {"mode": "legacy_parent_exact_all_rng_state", "checkpoint_kind": checkpoint_kind, "saved_cuda_device_count": len(saved_sizes), "runtime_cuda_device_count": len(runtime_sizes), "saved_cuda_state_sizes": saved_sizes, "runtime_cuda_state_sizes": runtime_sizes}  # 永久记录本次无需重建旧CUDA状态。
    next_step = int(completed_step) + 1  # 计算恢复后即将执行的第一个绝对更新步。
    replacement_seed = PER_UPDATE_SEED_OFFSET + next_step  # 使用与正式循环第一批完全相同的绝对步种子公式。
    torch.cuda.manual_seed_all(replacement_seed)  # 只重建无法载入的CUDA生成器，不改变已经精确恢复的Python、NumPy和CPU状态。
    rebuilt_once = [state.clone() for state in torch.cuda.get_rng_state_all()]  # 保存第一次确定性重建后的当前格式字节。
    torch.cuda.manual_seed_all(replacement_seed)  # 用同一整数再重建一次以验证当前后端结果可重复。
    rebuilt_twice = torch.cuda.get_rng_state_all()  # 回读第二次重建结果用于确定性核验。
    if len(rebuilt_once) != len(rebuilt_twice) or not all(torch.equal(left, right) for left, right in zip(rebuilt_once, rebuilt_twice)):  # 同种子两次结果必须逐字节相同。
        raise RuntimeError("旧父断点CUDA随机状态的确定性重建失败。")  # 当前后端不能稳定重建时拒绝进入训练。
    return {"mode": "legacy_parent_non_cuda_exact_cuda_reseeded", "checkpoint_kind": checkpoint_kind, "fallback_reason": "saved_and_runtime_cuda_state_layout_mismatch", "saved_cuda_device_count": len(saved_sizes), "runtime_cuda_device_count": len(runtime_sizes), "saved_cuda_state_sizes": saved_sizes, "runtime_cuda_state_sizes": runtime_sizes, "cuda_replacement_seed": replacement_seed}  # 记录旧格式与当前格式长度、兼容原因以及唯一替代种子。


def validate_core_configuration(configuration: object) -> None:  # 核对旧断点中与学生12号数学直接相关的配置子集。
    if not isinstance(configuration, dict):  # 完整断点必须带有可读配置字典。
        raise RuntimeError("完整断点缺少配置字典。")  # 没有配置无法证明来源一致。
    expected = {"score_target": "clean_x0", "score_weight": "sqrt_alpha_bar", "batch_size": 32, "student_start_step": 950, "lambda_fm": 2.0, "lambda_dist": 45.0, "train_frames": 128, "bnf_alignment": "legacy_length_linear"}  # 固定学生12号核心目标和数据设置。
    for key, value in expected.items():  # 逐项核对旧配置中必须存在的核心字段。
        if configuration.get(key) != value:  # 缺失或取值变化都不允许静默继续。
            raise RuntimeError(f"父断点核心配置不符：{key}={configuration.get(key)!r}，期望 {value!r}。")  # 报告确切字段。


def validate_parent_checkpoint(payload: object, plan: dict) -> dict:  # 确认外部父断点是真正的学生12号完整训练状态。
    if not isinstance(payload, dict):  # torch.load 结果必须是映射而非裸权重张量。
        raise RuntimeError("父断点不是完整训练状态字典。")  # 明确拒绝 weight-only 文件。
    required = {"kind", "global_step", "completed_passes", "partial_batch", "resumable", "student", "discriminator", "student_optimizer", "discriminator_optimizer", "rng_state", "configuration", "effective_optimizer_lrs", "mc_samples", "score_semantics"}  # 列出公平分叉所需字段，并要求明确完整批次状态。
    missing = sorted(required - set(payload))  # 计算实际缺失的完整状态部分。
    if missing:  # 任一项缺失都不能通过重新初始化补齐。
        raise RuntimeError(f"父断点不是完整状态，缺少：{missing}")  # 报告缺项并停止。
    if payload.get("kind") not in {PARENT_KIND, CHILD_KIND}:  # 父状态只能来自原学生12号或本阶段入口生成的完整基线。
        raise RuntimeError(f"不接受的父断点类型：{payload.get('kind')!r}")  # 拒绝其他实验的同名文件。
    if payload.get("kind") == CHILD_KIND and payload.get("cycle_enabled") is not False:  # 本阶段生成的父断点只能来自无Cycle基线分支。
        raise RuntimeError("不能把 Cycle 子断点当作新的共享基线父断点。")  # 防止后续比较继承已经施加过的Cycle目标。
    if payload.get("global_step") != plan["expected_start_global_step"]:  # 真实绝对步数必须与计划父 pass 一致。
        raise RuntimeError(f"父断点 global_step={payload.get('global_step')!r}，计划要求 {plan['expected_start_global_step']}。")  # 防止错误分叉点。
    if payload.get("completed_passes") != plan["parent_pass"]:  # 完整pass标签必须与绝对更新步数和计划共同一致。
        raise RuntimeError("父断点 completed_passes 与计划分叉轮次不一致。")  # 防止仅修改global_step标签的错误文件通过。
    if payload.get("effective_optimizer_lrs") != {"student": STUDENT_LR, "discriminator": DISCRIMINATOR_LR}:  # 核对断点声明的真实两组学习率。
        raise RuntimeError("父断点真实学习率不是学生5e-5、判别器2e-4。")  # 不采用模板里的学生2e-4覆盖实际状态。
    if payload.get("mc_samples") != 1 or payload.get("score_semantics") != "reconstruction_clean_x0_sqrt_alpha_bar":  # 锁定学生12号 MC1 和蒸馏语义。
        raise RuntimeError("父断点的 MC 样本数或分数蒸馏语义变化。")  # 不把其他候选当作学生12号。
    if payload.get("partial_batch") is not False or payload.get("resumable") is not True:  # 父状态必须明确位于完整批次边界且声明可恢复。
        raise RuntimeError("父断点位于不完整批次或被标为不可恢复。")  # 禁止从半次 D/S 更新继续。
    validate_core_configuration(payload.get("configuration"))  # 核对与当前实验相关的核心配置子集。
    return payload  # 返回已经确认的完整父状态。


def validate_child_checkpoint(payload: object, contract_sha: str, plan: dict) -> dict:  # 核对本阶段入口保存的滚动续训状态。
    if not isinstance(payload, dict) or payload.get("kind") != CHILD_KIND or payload.get("format_version") != FORMAT_VERSION:  # 子断点必须使用本入口的明确类型和版本。
        raise RuntimeError("滚动断点类型或格式版本不符。")  # 不尝试猜测其他格式。
    required = {"global_step", "completed_passes", "cycle_step", "student", "discriminator", "student_optimizer", "discriminator_optimizer", "rng_state", "configuration", "mc_samples", "score_semantics", "run_contract_sha256", "plan_name", "cycle_enabled", "effective_optimizer_lrs", "partial_batch", "resumable", "origin_parent_sha256"}  # 列出恢复阶段训练所需字段，并要求完整数学语义与父断点身份。
    missing = sorted(required - set(payload))  # 找出滚动断点缺失项。
    if missing:  # 缺少任一完整状态都禁止恢复。
        raise RuntimeError(f"滚动断点不完整，缺少：{missing}")  # 明确报告损坏部分。
    if payload.get("run_contract_sha256") != contract_sha or payload.get("plan_name") != plan["plan_name"] or payload.get("cycle_enabled") is not plan["cycle_enabled"]:  # 合同、计划名和目标开关必须完全一致。
        raise RuntimeError("滚动断点属于不同实验合同。")  # 防止 Base 与 Cycle 互相续训。
    if payload.get("partial_batch") is not False or payload.get("resumable") is not True:  # 子断点必须明确位于一次完整 D/S 更新之后并允许恢复。
        raise RuntimeError("滚动断点不是明确可恢复的完整批次状态。")  # 不接受缺省、半批或被撤销恢复资格的文件。
    if payload.get("origin_parent_sha256") != plan["parent_checkpoint_sha256"]:  # 滚动断点必须继续来自计划登记的同一个共享父断点。
        raise RuntimeError("滚动断点的原始父断点身份与计划不同。")  # 防止同名分支在更换父模型后混续。
    step = payload.get("global_step")  # 读取已经完成的绝对训练步。
    if type(step) is not int or not plan["expected_start_global_step"] < step <= plan["_target_global_step"]:  # 子游标必须位于本阶段有效范围。
        raise RuntimeError(f"滚动断点 global_step 非法：{step!r}")  # 拒绝倒退或越过终点。
    expected_cycle_step = step - plan["expected_start_global_step"] if plan["cycle_enabled"] else 0  # 计算本阶段应完成的 Cycle 更新数。
    if payload.get("cycle_step") != expected_cycle_step:  # 独立 Cycle 游标必须与已完成阶段步严格一致。
        raise RuntimeError("滚动断点的 cycle_step 与阶段进度不一致。")  # 防止配对和噪声序列错位。
    if payload.get("completed_passes") != step // UPDATES_PER_PASS:  # 子断点的完整pass标签必须由绝对步数唯一决定。
        raise RuntimeError("滚动断点的 completed_passes 与 global_step 不一致。")  # 拒绝标签漂移的滚动文件。
    if payload.get("effective_optimizer_lrs") != {"student": STUDENT_LR, "discriminator": DISCRIMINATOR_LR}:  # 子断点也必须保留真实学习率。
        raise RuntimeError("滚动断点的真实学习率发生变化。")  # 不允许恢复时悄悄改学习率。
    if payload.get("mc_samples") != 1 or payload.get("score_semantics") != "reconstruction_clean_x0_sqrt_alpha_bar":  # 子断点也必须保留学生12号蒸馏语义。
        raise RuntimeError("滚动断点的 MC 样本数或分数蒸馏语义变化。")  # 不允许续训途中更换基础目标。
    validate_core_configuration(payload.get("configuration"))  # 核对滚动断点保存的基础数学配置。
    return payload  # 返回已经确认可恢复的子状态。


def load_checkpoint(path: Path) -> dict:  # 用兼容新旧 PyTorch 的方式把完整训练状态加载到 CPU。
    try:  # 新版 PyTorch 支持显式关闭 weights_only 限制。
        value = torch.load(path, map_location="cpu", weights_only=False)  # 旧断点包含优化器和 RNG 字典，必须完整读取。
    except TypeError:  # PyTorch 1.13 尚没有 weights_only 参数。
        value = torch.load(path, map_location="cpu")  # 在旧环境中使用等价的完整加载路径。
    if not isinstance(value, dict):  # 上层验证依赖明确字段映射。
        raise RuntimeError(f"断点顶层不是字典：{path}")  # 裸模型权重不能用于分叉。
    return value  # 返回仍在 CPU 的完整状态，稍后严格加载到模型。


def build_components(device: torch.device, cycle_enabled: bool):  # 按学生12号原顺序构建教师、学生、声码器和判别器。
    config = replace(baseline.PAPER_STUDENT_CONFIG, num_workers=0, cycle_enabled=cycle_enabled, cycle_start_epoch=1, epochs=50)  # 阶段内第一步即按计划开关 Cycle，其他核心参数不变。
    if config.cycle_enabled is not cycle_enabled or config.cycle_start_epoch != 1 or config.lambda_cycle != 1.0:  # 构建后立即核对Cycle开关、门控轮次和预注册权重。
        raise RuntimeError("阶段配置没有正确锁定 Cycle 开关、起始轮次或权重。")  # 防止计划标签和真实训练目标不一致。
    teacher, student, _ = baseline.load_teacher_and_student(baseline.TEACHER, config, device)  # 用500轮教师结构建立并初始化学生。
    vocoder, vocoder_config, _ = baseline.load_frozen_vocoder(ROOT, baseline.VOCODER_DIR / "generator_step_2500000.pt", baseline.VOCODER_DIR / "config.json", baseline.VOCODER_DIR / "final_checkpoint_attestation_comments_20260906.json", config, device)  # 严格加载自训250万步 HiFi-GAN。
    discriminator = baseline.build_paper_discriminator(ROOT, vocoder_config, config, device)  # 建立与父断点相同的 MPD+MRD 判别器。
    diffusion = baseline.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)  # 建立相同1000步余弦扩散日程。
    mel_mean, mel_std = baseline.load_mel_statistics(ROOT / "data", device)  # 读取原训练集固定 mel 均值和标准差。
    student_optimizer = torch.optim.Adam(student.parameters(), lr=STUDENT_LR, betas=(0.5, 0.9))  # 用真实5e-5学生学习率建立待恢复 Adam 容器。
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(0.5, 0.9))  # 用真实2e-4判别器学习率建立待恢复 Adam 容器。
    return config, teacher, student, vocoder, discriminator, diffusion, mel_mean, mel_std, student_optimizer, discriminator_optimizer  # 返回训练一批所需全部对象。


def restore_state(payload: dict, student: torch.nn.Module, discriminator: torch.nn.Module, student_optimizer: torch.optim.Optimizer, discriminator_optimizer: torch.optim.Optimizer, device: torch.device) -> tuple:  # 恢复两模型、两 Adam 和可审计的随机状态。
    student.load_state_dict(payload["student"], strict=True)  # 严格恢复学生全部参数和缓冲区。
    discriminator.load_state_dict(payload["discriminator"], strict=True)  # 严格恢复共同训练的判别器状态。
    validate_model_state(student, "学生")  # 恢复后立即确认学生权重和缓冲区全部有限。
    validate_model_state(discriminator, "判别器")  # 恢复后立即确认判别器权重和缓冲区全部有限。
    student_optimizer.load_state_dict(payload["student_optimizer"])  # 恢复学生 Adam 一阶、二阶动量和步数。
    discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])  # 恢复判别器 Adam 一阶、二阶动量和步数。
    step = int(payload["global_step"])  # 读取完整断点已经完成的绝对更新数。
    validate_optimizer_state(student_optimizer, student, STUDENT_LR, step, "学生")  # 深入确认学生Adam完整恢复且每项状态合法。
    validate_optimizer_state(discriminator_optimizer, discriminator, DISCRIMINATOR_LR, step, "判别器")  # 深入确认判别器Adam完整恢复且每项状态合法。
    student.train()  # 模型训练模式不在 state_dict 中，因此显式恢复。
    discriminator.train()  # 判别器同样回到训练模式。
    rng_restore_audit = restore_stage_rng_state(payload, device, step)  # 精确恢复本阶段断点；旧父断点仅在CUDA长度不兼容时按下一绝对步确定性重建。
    cycle_step = int(payload.get("cycle_step", 0))  # 原学生12号父断点没有该字段时从0开始独立 Cycle 编号。
    return step, cycle_step, rng_restore_audit  # 返回两个进度游标以及本次随机状态恢复的完整审计信息。


def validate_data_restore_receipt() -> dict:  # 快速确认准备 Notebook 生成的数据恢复收据结构完整且没有被手工简化。
    if not DATA_RESTORE_RECEIPT.is_file() or DATA_RESTORE_RECEIPT.is_symlink():  # 收据必须是项目根中的真实普通文件。
        raise FileNotFoundError(f"缺少必要数据快速恢复身份收据：{DATA_RESTORE_RECEIPT}")  # 要求先完整运行新的准备 Notebook。
    receipt = read_json(DATA_RESTORE_RECEIPT)  # 读取由持久分卷清单和本地文件元数据生成的确定性收据。
    expected_policy = "manifests_and_markers_plus_exact_feature_layouts_plus_128_retained_train_bnf_and_spk_samples_each_plus_456_published_sentinels_without_full_48gb_rehash"  # 锁定本版本的全路径尺寸与固定内容哨兵策略。
    expected_level = "all_paths_and_sizes_plus_fixed_content_sentinels_not_full_content_verification"  # 明确禁止把快速收据误写成48GB全内容验证。
    if receipt.get("format_version") != 1 or receipt.get("kind") != "fastvoicegrad_essential_data_restore_identity" or receipt.get("verification_policy") != expected_policy or receipt.get("verification_level") != expected_level or receipt.get("files_per_shard") != 2000:  # 核对收据版本、类型、核验范围、策略和原分卷规则。
        raise RuntimeError("必要数据快速恢复身份收据的版本或策略不符合本实验要求。")  # 拒绝把其他摘要文件冒充正式收据。
    expected_report_hashes = {"bnf_extraction_report.json": "127bdad5d9a50377c6cdeb5d59addab0f0df1877a9c16105a1087991c0e24ebd", "spk_emb_extraction_report.json": "8ef508eab880fbadff0d919b8e5f7fec8984b852b33dd11bd12b40ad1c82bf3f", "mel_extraction_report_22050.json": "e96b84398a8ec036ac6b37ed433f20f4c0e173c6601a35f5b4d4ef7571debe29", "mel_stats_report.json": "25823115dc00431775938a1491fe8462f36a41e2c15b02a04f7e83fc41053810"}  # 固定四份服务器正式数据来源报告身份。
    if receipt.get("provenance_report_sha256") != expected_report_hashes:  # 收据必须明确绑定四份权威来源报告。
        raise RuntimeError("必要数据快速恢复身份收据的正式来源报告身份不一致。")  # 不接受当前数据自行生成的替代报告。
    layouts = receipt.get("feature_layouts")  # 读取全部BNF、说话人向量和训练mel路径尺寸集合摘要。
    if not isinstance(layouts, dict) or set(layouts) != {"bnf", "spk_emb", "mel_train"}:  # 三类布局摘要必须不多不少。
        raise RuntimeError("必要数据快速恢复身份收据缺少三类全量布局摘要。")  # 防止只抽样而漏掉其余文件结构。
    for layout_name, layout in layouts.items():  # 逐类检查全量布局摘要的数量和SHA256字段。
        expected_count = 39387 if layout_name == "mel_train" else 39487  # mel统计只使用训练集，BNF和说话人向量覆盖全量清单。
        layout_sha = layout.get("layout_sha256") if isinstance(layout, dict) else None  # 读取当前布局的顺序敏感SHA256。
        layout_count = layout.get("count") if isinstance(layout, dict) else None  # 读取当前布局正式纳入的文件数量。
        if layout_count != expected_count or not isinstance(layout_sha, str) or len(layout_sha) != 64:  # 数量和摘要格式都必须符合正式协议。
            raise RuntimeError(f"必要数据全量布局摘要非法：{layout_name}")  # 报告具体数据组并停止。
    if receipt.get("retained_train_content_sample_count_per_group") != 128 or receipt.get("published_sentinel_counts") != {"mel": 152, "bnf": 152, "spk_emb": 152}:  # 固定训练内容抽样和原发布哨兵数量不能减少。
        raise RuntimeError("必要数据固定内容抽样或发布哨兵数量不一致。")  # 防止把较弱收据冒充当前版本。
    for digest_name in ("retained_train_content_sample_set_sha256", "selected12_manifest_sha256", "published_sentinel_set_sha256", "overall_marker_sha256"):  # 检查四项固定来源或抽样集合摘要。
        digest_value = receipt.get(digest_name)  # 读取当前摘要字段。
        if not isinstance(digest_value, str) or len(digest_value) != 64 or any(character not in "0123456789abcdef" for character in digest_value.lower()):  # 每项摘要都必须为完整SHA256。
            raise RuntimeError(f"必要数据快速恢复身份收据摘要非法：{digest_name}")  # 报告具体坏字段并停止。
    if receipt.get("selected12_manifest_sha256") != "4c4378317957d566159c608cf48173ded084e8a00ad55935a80b69d0bea83454":  # 原发布哨兵必须来自固定学生12号外层清单。
        raise RuntimeError("必要数据收据没有绑定权威学生12号SHA256清单。")  # 不接受替换后的哨兵来源。
    groups = receipt.get("groups")  # 取得五类必要数据各自的恢复摘要。
    expected_groups = {"mel", "bnf", "spk_emb", "metadata", "stats"}  # 固定学生12号训练读取的全部数据组。
    if not isinstance(groups, dict) or set(groups) != expected_groups:  # 数据组必须不多不少且名称完全匹配。
        raise RuntimeError("必要数据快速恢复身份收据没有完整覆盖五类数据。")  # 防止缺少mel、BNF或说话人向量仍进入训练。
    for group_name, group in groups.items():  # 逐组检查清单哈希、文件数量、总大小和分卷表。
        manifest_sha = group.get("manifest_sha256") if isinstance(group, dict) else None  # 读取当前组固定文件清单的SHA256。
        shards = group.get("shards") if isinstance(group, dict) else None  # 读取当前组完整持久分卷摘要。
        marker_sha = group.get("group_marker_sha256") if isinstance(group, dict) else None  # 读取当前数据组持久完成标记的SHA256。
        if not isinstance(manifest_sha, str) or len(manifest_sha) != 64 or any(character not in "0123456789abcdef" for character in manifest_sha.lower()) or not isinstance(marker_sha, str) or len(marker_sha) != 64:  # 清单与完成标记都必须有完整SHA256。
            raise RuntimeError(f"必要数据组的清单哈希非法：{group_name}")  # 报告具体组并停止。
        if type(group.get("file_count")) is not int or group["file_count"] < 1 or type(group.get("local_total_bytes")) is not int or group["local_total_bytes"] < 1:  # 文件数和本地总字节数都必须为正整数。
            raise RuntimeError(f"必要数据组的文件数量或总字节数非法：{group_name}")  # 防止空目录生成可接受收据。
        if not isinstance(shards, list) or not shards:  # 每类必要数据必须来自至少一个已完成正式分卷。
            raise RuntimeError(f"必要数据组没有登记持久分卷：{group_name}")  # 缺少恢复来源时停止。
        if any(not isinstance(shard, dict) or not isinstance(shard.get("name"), str) or type(shard.get("bytes")) is not int or shard["bytes"] < 1 for shard in shards):  # 每个分卷必须登记名称和正字节数。
            raise RuntimeError(f"必要数据组包含非法分卷记录：{group_name}")  # 拒绝不完整或手工缩减的分卷摘要。
    return receipt  # 返回已经通过结构检查的收据供后续合同记录。


def source_hashes() -> dict:  # 记录会直接决定本阶段更新数学的源码身份。
    validate_data_restore_receipt()  # 在哈希收据前先确认其结构和快速核验策略确实完整。
    paths = [Path(__file__).resolve(), ROOT / "fastvoicegrad/train.py", ROOT / "fastvoicegrad/objectives.py", ROOT / "fastvoicegrad/config.py", ROOT / "fastvoicegrad/components.py", ROOT / "voicegrad/dataset.py", ROOT / "voicegrad/diffusion.py", ROOT / "voicegrad/model.py", ROOT / "voicegrad/feature_alignment.py", ROOT / "voicegrad/config.py", ROOT / "voicegrad/train.py", ROOT / "hifigan/custom/mrd.py", ROOT / "hifigan/custom/train.py", ROOT / "hifigan/custom/validate_checkpoint.py", ROOT / "hifigan/official/env.py", ROOT / "hifigan/official/models.py", ROOT / "hifigan/official/utils.py", TOOLS / "run_acdd_development.py", DEVELOPMENT, DEVELOPMENT_MANIFEST, DEVELOPMENT.parent / "pairs.csv", DEVELOPMENT.parent / "self_reconstruction.jsonl", DEVELOPMENT.parent / "student_train_manifest.csv", DEVELOPMENT.parent / "dev_excluded_manifest.csv", DATA_RESTORE_RECEIPT, ROOT / "data/metadata/train_manifest.csv", ROOT / "data/metadata/test_manifest.csv", ROOT / "data/metadata/all_manifest.csv", ROOT / "data/metadata/split_protocol.json", ROOT / "data/stats/mel_mean.npy", ROOT / "data/stats/mel_std.npy", ROOT / "data/stats/mel_stats_current.json", ROOT / "exp/logs/bnf_extraction_report.json", ROOT / "exp/logs/spk_emb_extraction_report.json", ROOT / "exp/logs/mel_extraction_report_22050.json", ROOT / "exp/logs/mel_stats_report.json", baseline.TEACHER, baseline.VOCODER_DIR / "generator_step_2500000.pt", baseline.VOCODER_DIR / "config.json", baseline.VOCODER_DIR / "final_checkpoint_attestation_comments_20260906.json"]  # 锁定入口、数学实现、快速数据收据、全量清单、正式来源报告、教师和自训练HiFi-GAN身份。
    fixed_sha256 = {str(ROOT / "data/metadata/all_manifest.csv"): "a4c71d063ec65f03294f6053e3244c80fe14dc6c23afa798fefe670cde8d9e61", str(ROOT / "data/metadata/split_protocol.json"): "1dcc9a91f2be5718d30ed6c7ced98b354128d6e796d2138f26fe981096b9752f", str(ROOT / "data/stats/mel_stats_current.json"): "933f6f607a896cddebcef9fd689a398f10a160ba2e09c9247978472b9a3dd266", str(ROOT / "exp/logs/bnf_extraction_report.json"): "127bdad5d9a50377c6cdeb5d59addab0f0df1877a9c16105a1087991c0e24ebd", str(ROOT / "exp/logs/spk_emb_extraction_report.json"): "8ef508eab880fbadff0d919b8e5f7fec8984b852b33dd11bd12b40ad1c82bf3f", str(ROOT / "exp/logs/mel_extraction_report_22050.json"): "e96b84398a8ec036ac6b37ed433f20f4c0e173c6601a35f5b4d4ef7571debe29", str(ROOT / "exp/logs/mel_stats_report.json"): "25823115dc00431775938a1491fe8462f36a41e2c15b02a04f7e83fc41053810"}  # 固定不能由当前恢复环境重新自报的辅助清单、统计说明和四份正式报告哈希。
    result = {}  # 收集绝对路径到 SHA256 的映射。
    for path in paths:  # 逐个核对并哈希关键源码。
        if not path.is_file() or path.is_symlink():  # 源码必须是普通文件而非可变链接。
            raise FileNotFoundError(f"缺少关键源码或发现软链接：{path}")  # 源码身份不清时停止。
        result[str(path)] = sha256_file(path)  # 保存当前实际字节的哈希。
        if str(path) in fixed_sha256 and result[str(path)] != fixed_sha256[str(path)]:  # 对有服务器封存身份的辅助文件额外做硬编码比对。
            raise RuntimeError(f"固定辅助数据身份不一致：{path}")  # 不允许首个实验合同自行接受漂移后的报告或清单。
    return result  # 返回可写入运行合同的源码身份表。


def build_run_contract(plan: dict, parent_sha: str, device: torch.device) -> dict:  # 构造不会随断线重跑改变的阶段实验合同。
    return {"format_version": FORMAT_VERSION, "kind": "fastvoicegrad_cycle_stage_v2_contract", "semantic_plan": semantic_plan({key: value for key, value in plan.items() if not key.startswith("_")}), "origin_parent_sha256": parent_sha, "effective_optimizer_lrs": {"student": STUDENT_LR, "discriminator": DISCRIMINATOR_LR}, "optimizer_state_policy": "restore_both_without_reset", "rng_resume_policy": {"version": 1, "stage_checkpoint": "exact_all_rng_state", "legacy_parent_non_cuda": "exact_restore_required", "legacy_parent_cuda": "exact_if_format_compatible_else_reseed_from_absolute_next_step", "per_update_seed_formula": f"{PER_UPDATE_SEED_OFFSET}+next_global_step"}, "cycle_random_policy": "completed_cycle_updates_start_at_zero_and_first_seed_index_is_one", "base_random_policy": "student12_absolute_step_seeds", "updates_per_pass": UPDATES_PER_PASS, "loss_keys": list(LOSS_KEYS), "source_sha256": source_hashes(), "runtime_identity": runtime_identity(device), "runtime_policy": "same_software_backend_and_gpu_model_required_for_resume_and_paired_arm"}  # 完整登记比较所需来源、旧CUDA状态兼容规则，并把配对实验运行环境锁入数学合同。


def ensure_paired_identity(plan: dict, contract: dict) -> Path:  # 让同一组Base与Cycle分支共享父断点、源码和运行环境身份。
    path = DRIVE_RUN_ROOT / f"{plan['paired_group']}.paired_identity.json"  # 在两个分支共同父目录保存一份持久配对身份。
    value = {"format_version": FORMAT_VERSION, "kind": "fastvoicegrad_cycle_stage_v2_paired_identity", "paired_group": plan["paired_group"], "origin_parent_sha256": contract["origin_parent_sha256"], "effective_optimizer_lrs": contract["effective_optimizer_lrs"], "rng_resume_policy": contract["rng_resume_policy"], "base_random_policy": contract["base_random_policy"], "source_sha256": contract["source_sha256"], "runtime_identity": contract["runtime_identity"]}  # 只收录两臂必须相同的来源和RNG恢复规则，不收录Cycle开关本身。
    if path.exists():  # 第二个分支或同一分支重启时读取首个分支锁定的身份。
        if read_json(path) != value:  # 父断点、源码、数据、GPU型号或软件任一变化都破坏严格配对。
            raise RuntimeError(f"配对实验身份与首个分支不一致：{path}")  # 要求恢复相同运行环境，而不是静默混合结果。
    else:  # 当前配对组第一次正式启动时发布共享身份。
        try:  # 使用排他创建避免两个Colab同时把不同身份写成首个版本。
            with path.open("x", encoding="utf-8") as handle:  # 文件已存在时由操作系统拒绝本次创建。
                handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")  # 一次写入可人工审计的完整身份。
                handle.flush()  # 立即把Python缓冲区交给Drive文件系统。
        except FileExistsError:  # 另一运行时刚刚抢先创建时重新读取其正式内容。
            if read_json(path) != value:  # 并发首启也必须得到完全相同的共享身份。
                raise RuntimeError(f"并发创建的配对实验身份不同：{path}")  # 拒绝继续而不覆盖对方记录。
    return path  # 返回已确认的持久配对身份路径供尝试日志记录。


def save_full_checkpoint(output: Path, local_output: Path, active_checkpoint_path: object, contract_sha: str, plan: dict, global_step: int, cycle_step: int, config: object, student: torch.nn.Module, discriminator: torch.nn.Module, student_optimizer: torch.optim.Optimizer, discriminator_optimizer: torch.optim.Optimizer, recent_losses: dict) -> dict:  # 先在本地高速盘序列化，再向非活跃槽原子提交完整状态并保留上一代断点。
    validate_model_state(student, "学生")  # 保存前确认本次Adam更新没有产生非有限学生状态。
    validate_model_state(discriminator, "判别器")  # 保存前确认本次Adam更新没有产生非有限判别器状态。
    validate_optimizer_state(student_optimizer, student, STUDENT_LR, global_step, "学生")  # 保存前核对学生Adam全部动量和游标。
    validate_optimizer_state(discriminator_optimizer, discriminator, DISCRIMINATOR_LR, global_step, "判别器")  # 保存前核对判别器Adam全部动量和游标。
    expected_cycle_step = global_step - plan["expected_start_global_step"] if plan["cycle_enabled"] else 0  # 根据阶段进度计算应有 Cycle 更新数。
    if cycle_step != expected_cycle_step:  # 独立 Cycle 游标必须与当前阶段更新数一致。
        raise RuntimeError("拒绝保存 cycle_step 与阶段进度不一致的断点。")  # 防止恢复后随机序列跳号。
    payload = {"kind": CHILD_KIND, "format_version": FORMAT_VERSION, "plan_name": plan["plan_name"], "cycle_enabled": plan["cycle_enabled"], "global_step": global_step, "completed_passes": global_step // UPDATES_PER_PASS, "cycle_step": cycle_step, "partial_batch": False, "resumable": True, "origin_parent_sha256": plan["parent_checkpoint_sha256"], "configuration": config.as_dict(), "effective_optimizer_lrs": {"student": STUDENT_LR, "discriminator": DISCRIMINATOR_LR}, "mc_samples": 1, "score_semantics": "reconstruction_clean_x0_sqrt_alpha_bar", "run_contract_sha256": contract_sha, "student": student.state_dict(), "discriminator": discriminator.state_dict(), "student_optimizer": student_optimizer.state_dict(), "discriminator_optimizer": discriminator_optimizer.state_dict(), "rng_state": training.capture_rng_state(), "recent_loss_means": recent_losses}  # 组装论文分叉要求的全部模型、优化器、随机数和来源字段。
    local_output.mkdir(parents=True, exist_ok=True)  # 确保当前计划的本地高速暂存目录存在。
    local_staging = local_output / "latest_full.staging.pt"  # 固定本地暂存文件，避免直接向Drive做昂贵序列化。
    local_temporary = local_output / "latest_full.staging.pt.tmp"  # 本地序列化也先写临时文件，完整后才替换暂存文件。
    if local_staging.exists():  # 上次上传失败可能留下已经过时的完整暂存文件。
        local_staging.unlink()  # 当前内存状态可能已经前进，因此不猜测复用旧文件并先释放约1.4GB空间。
    if local_temporary.exists():  # 上次会话内中断可能留下精确的本地临时文件。
        local_temporary.unlink()  # 只删除当前实验专用临时文件后重新生成。
    local_free = shutil.disk_usage(local_output).free  # 在清理旧暂存后读取真正可供本次保存使用的本地空间。
    if local_free < MINIMUM_LOCAL_FREE_BYTES:  # 保存时再次检查，防止训练期间其他文件消耗启动时的余量。
        raise RuntimeError(f"完整断点保存前本地空间不足：剩余{local_free}字节。")  # 在序列化之前失败，保留Drive中上一个完整断点。
    with local_temporary.open("wb") as handle:  # 以二进制写模式在本地SSD序列化完整训练态。
        torch.save(dict(payload), handle)  # 保存普通字典，兼容后续完整torch.load恢复。
        handle.flush()  # 把Python缓冲区送到操作系统。
        os.fsync(handle.fileno())  # 要求本地文件数据在提交前真正落盘。
    os.replace(local_temporary, local_staging)  # 完整序列化后原子发布本地暂存文件。
    local_sha = sha256_file(local_staging)  # 计算将要上传到Drive的权威字节身份。
    checkpoint_slots = (output / "latest_full.pt", output / "previous_full.pt")  # 使用两个轮换槽保证新断点失败时上一代完整状态仍然存在。
    if active_checkpoint_path is not None:  # 从阶段断点恢复或本进程已经保存过一次时必须保护当前活跃槽。
        active_checkpoint_path = Path(active_checkpoint_path)  # 将调用方记录的当前活跃断点规范为Path对象。
        if active_checkpoint_path not in checkpoint_slots:  # 活跃路径只能是当前实验两个固定槽之一。
            raise RuntimeError(f"当前活跃断点不属于双槽目录：{active_checkpoint_path}")  # 防止错误路径影响槽位选择。
    checkpoint_path = checkpoint_slots[1] if active_checkpoint_path == checkpoint_slots[0] else checkpoint_slots[0]  # 总是写非活跃槽，首次保存固定使用latest槽。
    drive_partial = checkpoint_path.with_name(checkpoint_path.name + ".partial")  # Drive端使用目标槽专属临时文件，复制中断不会破坏当前活跃槽。
    if drive_partial.exists():  # 上次断线可能留下未正式提交的Drive临时副本。
        drive_partial.unlink()  # 只删除本实验精确临时文件，旧正式断点保持不动。
    shutil.copyfile(local_staging, drive_partial)  # 从本地顺序复制完整字节到Drive临时文件。
    drive_partial_sha = sha256_file(drive_partial)  # 从Drive读回一次，确认上传字节没有截断或损坏。
    if drive_partial_sha != local_sha:  # 上传前后哈希必须逐字节完全一致。
        raise RuntimeError("Drive 完整断点临时文件与本地暂存文件 SHA256 不一致。")  # 保留旧latest并停止提交坏文件。
    os.replace(drive_partial, checkpoint_path)  # 校验成功后原子替换非活跃槽，当前上一代断点此刻仍然完整保留。
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size != local_staging.stat().st_size:  # 改名后核对正式文件存在且字节数保持不变。
        raise RuntimeError("Drive 完整断点提交后的文件大小异常。")  # 不发布指向异常文件的轻量指针。
    local_staging.unlink()  # Drive正式断点提交后删除本地大文件释放SSD空间。
    pointer = {"path": str(checkpoint_path), "global_step": global_step, "cycle_step": cycle_step, "checkpoint_sha256": local_sha, "run_contract_sha256": contract_sha, "updated_at_unix": time.time()}  # 生成便于恢复和人工核对的轻量指针。
    write_json_atomic(output / "latest_full.json", pointer)  # 断点完整提交后再原子更新指针。
    append_json_line(output / "checkpoint_history.jsonl", pointer)  # 保留每次滚动保存的哈希和步数历史。
    return pointer  # 返回已经持久化的断点记录供状态文件引用。


def load_committed_stage_checkpoint(output: Path, contract_sha: str, plan: dict) -> object:  # 优先恢复指针正式提交的槽，损坏或指针丢失时回退另一代完整断点。
    slots = (output / "latest_full.pt", output / "previous_full.pt")  # 固定当前实验两个轮换完整断点槽。
    pointer_path = output / "latest_full.json"  # 使用单个原子指针声明哪一槽是最近正式提交状态。
    pointer = None  # 没有指针时稍后通过双槽完整校验恢复。
    pointer_error = None  # 保存损坏指针的诊断信息但不立即放弃上一代断点。
    if pointer_path.exists():  # 正常运行至少保存一次后应存在轻量正式指针。
        try:  # 指针本身损坏时仍允许扫描两份大型完整状态自救。
            pointer = read_json(pointer_path)  # 读取最近正式提交的槽位和哈希。
        except BaseException as error:  # 捕获Drive短写、编码或JSON解析问题。
            pointer_error = repr(error)  # 保存错误文本供最终失败报告或修复记录。
    slot_by_text = {str(slot): slot for slot in slots}  # 建立只允许两个固定绝对路径的白名单映射。
    if isinstance(pointer, dict) and pointer.get("path") in slot_by_text:  # 指针路径合法时先只验证它声明的最近正式槽。
        pointed_slot = slot_by_text[pointer["path"]]  # 将指针文本恢复为白名单中的Path对象。
        try:  # 最近槽损坏时不能阻止回退上一代槽。
            pointed_sha = sha256_file(pointed_slot)  # 从Drive完整读取最近槽并计算真实字节身份。
            pointed_payload = validate_child_checkpoint(load_checkpoint(pointed_slot), contract_sha, plan)  # 核对模型、优化器、游标和运行合同。
            stable_pointer = {"path": str(pointed_slot), "global_step": pointed_payload["global_step"], "cycle_step": pointed_payload["cycle_step"], "checkpoint_sha256": pointed_sha, "run_contract_sha256": contract_sha}  # 构造最近槽必须与指针一致的稳定字段。
            if not all(pointer.get(key) == value for key, value in stable_pointer.items()):  # 正式指针必须逐项匹配实际完整状态。
                raise RuntimeError("最近断点指针与其声明的完整槽不一致。")  # 不把被替换、回退或混用的槽当作正式最近状态。
            return pointed_payload, pointed_slot, pointed_sha  # 最近正式槽完整有效时直接恢复，避免无谓读取上一代1.4GB文件。
        except BaseException as error:  # 最近正式槽无法使用时保存诊断并进入双槽回退。
            pointer_error = repr(error)  # 最终若两槽都不可用则报告最近槽的真实失败原因。
    elif isinstance(pointer, dict):  # 指针存在但指向两个固定槽之外时不能直接信任。
        pointer_error = f"指针路径不属于固定双槽：{pointer.get('path')}"  # 保存非法路径诊断并只扫描白名单槽。
    valid_candidates = []  # 收集在没有可信指针时通过全部深度校验的候选状态。
    candidate_errors = []  # 收集每个存在但损坏槽的独立诊断信息。
    for slot in slots:  # 依次检查两个固定槽并选择global_step最高的完整状态。
        if not slot.exists():  # 尚未使用过的槽无需读取。
            continue  # 继续检查另一槽。
        try:  # 单个槽损坏时仍继续尝试另一代。
            slot_sha = sha256_file(slot)  # 计算当前候选槽的真实字节身份。
            slot_payload = validate_child_checkpoint(load_checkpoint(slot), contract_sha, plan)  # 深度核对候选的全部可恢复训练状态。
            valid_candidates.append((int(slot_payload["global_step"]), slot_payload, slot, slot_sha))  # 保存步数、载荷、路径和哈希供选择。
        except BaseException as error:  # 保存当前槽失败原因但不阻止另一槽回退。
            candidate_errors.append(f"{slot.name}: {error!r}")  # 形成可读的逐槽诊断文本。
    if not valid_candidates:  # 两个槽都不存在或都无法通过完整校验。
        if pointer is None and pointer_error is None and not candidate_errors and not any(slot.exists() for slot in slots):  # 从未保存过阶段断点时应正常回到共享父状态。
            return None  # 用明确空值通知主流程加载学生12号共享父断点。
        raise RuntimeError(f"没有可恢复的阶段完整断点；指针错误={pointer_error}；槽错误={candidate_errors}")  # 已有持久痕迹却全部损坏时失败关闭。
    valid_candidates.sort(key=lambda item: item[0], reverse=True)  # 按绝对训练步从高到低选择最靠后的完整候选。
    _, selected_payload, selected_slot, selected_sha = valid_candidates[0]  # 取得校验通过且训练进度最高的槽。
    repaired_pointer = {"path": str(selected_slot), "global_step": selected_payload["global_step"], "cycle_step": selected_payload["cycle_step"], "checkpoint_sha256": selected_sha, "run_contract_sha256": contract_sha, "repaired_at_unix": time.time(), "previous_pointer_error": pointer_error, "candidate_errors": candidate_errors}  # 记录本次双槽自救选择及异常证据。
    write_json_atomic(pointer_path, repaired_pointer)  # 只有候选通过完整模型与优化器校验后才修复正式指针。
    return selected_payload, selected_slot, selected_sha  # 返回可用于恢复的最高步完整状态。


def acquire_run_lock(plan_name: str):  # 在项目本地 SSD 上锁定当前计划，防止重复训练进程。
    lock_dir = ROOT / "exp/locks/cycle_stage_v2"  # 使用专用锁目录，不触碰旧实验锁。
    lock_dir.mkdir(parents=True, exist_ok=True)  # 首次运行时创建锁目录。
    lock_path = lock_dir / "all_stage_training.lock"  # 单块Colab GPU上的四个阶段计划共同使用一把独占锁。
    handle = lock_path.open("a+", encoding="utf-8")  # 打开后保持句柄存活到训练结束。
    try:  # 非阻塞申请独占锁以便立即发现重复启动。
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # 同一计划只能有一个活跃进程。
    except BlockingIOError:  # 已有进程持锁时进入清理分支。
        handle.close()  # 关闭本次未取得锁的文件句柄。
        raise RuntimeError(f"计划 {plan_name} 已有训练进程。")  # 明确拒绝第二个进程。
    handle.seek(0)  # 回到文件开头写入当前进程信息。
    handle.truncate()  # 清除上次正常退出留下的旧文字。
    handle.write(json.dumps({"pid": os.getpid(), "plan": plan_name, "started_at_unix": time.time()}, ensure_ascii=False))  # 记录当前持锁进程便于人工排查。
    handle.flush()  # 立即把锁说明刷新到文件系统。
    return handle  # 返回句柄并由主流程在 finally 中关闭释放锁。


def runtime_boot_id() -> str:  # 读取当前Colab虚拟机启动身份以区分同机活进程和旧运行时残留租约。
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")  # Linux为每次系统启动提供一个不会在当前虚拟机内变化的唯一标识。
    if not boot_id_path.is_file():  # 极少数Linux环境可能没有暴露该内核标识。
        return "unavailable"  # 返回明确占位值并让跨运行时判断继续依赖心跳时间。
    return boot_id_path.read_text(encoding="utf-8").strip()  # 返回当前虚拟机稳定启动标识。


def local_process_is_alive(pid: object) -> bool:  # 只在租约来自当前虚拟机时检查对应进程是否仍然存在。
    if type(pid) is not int or pid < 1:  # 非普通正整数不能作为可信进程号。
        return False  # 将非法进程号视为不存在并继续使用心跳保护。
    try:  # 使用零信号只检查进程存在性，不向目标进程发送实际信号。
        os.kill(pid, 0)  # 当前用户能看到同一虚拟机中的活进程时调用会成功。
    except ProcessLookupError:  # 内核明确报告该进程号不存在。
        return False  # 允许后续根据心跳年龄判断旧租约。
    except PermissionError:  # 进程存在但当前用户没有发送信号权限。
        return True  # 保守视为仍在运行，绝不自动接管。
    return True  # 零信号成功表示当前虚拟机中的目标进程仍然存在。


def archive_stale_drive_lease(path: Path, payload: dict) -> Path:  # 将确认超时的旧租约移到审计目录而不是永久删除证据。
    stale_root = DRIVE_RUN_ROOT / "_stale_leases"  # 在所有实验臂之外使用专用目录保存异常断线记录。
    stale_root.mkdir(parents=True, exist_ok=True)  # 首次需要接管时创建持久审计目录。
    token_text = str(payload.get("token", "missing-token"))  # 读取旧租约令牌并兼容早期没有心跳字段的格式。
    token_suffix = hashlib.sha256(token_text.encode("utf-8")).hexdigest()[:12]  # 用短哈希避免把任意旧令牌直接放进文件名。
    created_value = payload.get("created_at_unix", 0.0)  # 读取旧租约首次创建时间供文件名排序。
    created_seconds = int(created_value) if isinstance(created_value, (int, float)) and math.isfinite(float(created_value)) else 0  # 将合法时间规范为整数秒。
    archive_path = stale_root / f"{path.parent.name}_{created_seconds}_{token_suffix}.json"  # 为当前实验和旧运行创建不冲突的审计文件名。
    os.replace(path, archive_path)  # 在Drive同一挂载内移动旧租约，只有一个并发接管者能够成功。
    return archive_path  # 返回旧租约审计位置供新租约记录来源。


def acquire_drive_lease(output: Path, plan_name: str) -> tuple:  # 用带心跳和超时接管的Drive租约阻止两个Colab同时写同一实验。
    path = output / "ACTIVE_RUN.lock"  # 把跨运行时租约放在当前实验的持久目录中。
    now = time.time()  # 使用同一个当前时间计算旧租约年龄并创建新租约。
    reclaimed_from = None  # 默认没有接管任何异常断线留下的旧租约。
    if path.exists():  # 发现旧租约时必须先判断同机进程和持续心跳，不能直接删除。
        try:  # 先尝试按正常JSON租约读取当前所有者信息。
            existing = read_json(path)  # 读取旧租约的人类可审计字段。
        except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:  # 初始排他写入瞬间虚拟机死亡时可能留下残缺或非对象租约。
            modified_at = path.stat().st_mtime  # 使用残缺文件自身修改时间作为保守的最后活跃时间。
            malformed_age = now - modified_at  # 计算残缺租约已经保持不变的时间。
            if malformed_age <= DRIVE_LEASE_STALE_SECONDS:  # 保护期内无法证明原运行时已经死亡。
                raise RuntimeError(f"发现尚在保护期内且无法解析的Drive租约：{path}；{error!r}")  # 不冒险与可能仍活跃的初始写入并发。
            existing = {"token": "malformed-lease", "created_at_unix": modified_at, "heartbeat_at_unix": modified_at, "parse_error": repr(error)}  # 为超过保护期的残缺租约构造可归档审计说明。
        same_boot = existing.get("runtime_boot_id") == runtime_boot_id()  # 判断旧租约是否来自当前仍在使用的同一虚拟机。
        if same_boot and local_process_is_alive(existing.get("pid")):  # 同一虚拟机中的原训练进程仍存在时绝不接管。
            raise RuntimeError(f"发现当前运行时仍存活的训练进程租约：{path}")  # 防止浏览器重复点击启动第二个进程。
        heartbeat_value = existing.get("heartbeat_at_unix", existing.get("created_at_unix"))  # 新旧租约都使用最后心跳或创建时间判断活跃度。
        if not isinstance(heartbeat_value, (int, float)) or not math.isfinite(float(heartbeat_value)):  # 没有合法时间的未知租约不能自动判死。
            raise RuntimeError(f"发现无法判断心跳时间的Drive租约：{path}；请先人工核对。")  # 对无法证明陈旧的状态失败关闭。
        heartbeat_age = now - float(heartbeat_value)  # 计算旧运行时已经多久没有向Drive报告活跃。
        if heartbeat_age <= DRIVE_LEASE_STALE_SECONDS:  # 三十分钟内有心跳的其他运行时仍可能在训练或上传断点。
            raise RuntimeError(f"发现仍在心跳保护期内的Drive租约：{path}；最后心跳距今{heartbeat_age:.0f}秒。")  # 不冒险同时写同一个滚动断点。
        reclaimed_from = str(archive_stale_drive_lease(path, existing))  # 超时后保留旧租约证据并开放安全重连。
    token = f"{plan_name}:{os.getpid()}:{time.time_ns()}"  # 生成当前进程唯一且便于人工辨认的租约令牌。
    payload = {"token": token, "plan_name": plan_name, "pid": os.getpid(), "runtime_boot_id": runtime_boot_id(), "created_at_unix": now, "heartbeat_at_unix": now, "phase": "acquired", "global_step": None, "cycle_step": None, "reclaimed_stale_lease": reclaimed_from, "stale_after_seconds": DRIVE_LEASE_STALE_SECONDS}  # 记录自动续训、安全接管和人工排查所需全部字段。
    try:  # 只有第一个运行时可以排他创建当前正式租约文件。
        with path.open("x", encoding="utf-8") as handle:  # 并发接管者已经先创建时由文件系统拒绝本次创建。
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")  # 写入完整的人类可读租约内容。
            handle.flush()  # 立即把缓冲区交给Drive文件系统。
    except FileExistsError:  # 另一个运行时在检查后抢先取得租约时进入安全停止分支。
        raise RuntimeError(f"另一个运行时刚刚取得Drive租约：{path}")  # 不覆盖对方租约或实验状态。
    return path, token, threading.Lock()  # 返回路径、令牌和进程内互斥锁，让主线程与后台心跳绝不同时改写租约。


def refresh_drive_lease(lease: tuple, phase: object = None, global_step: object = None, cycle_step: object = None) -> None:  # 周期性刷新Drive心跳，让浏览器断网或上传大断点时仍保持所有权。
    path, token, mutex = lease  # 取出当前进程持有的租约路径、唯一令牌和线程互斥锁。
    with mutex:  # 串行化主训练线程和后台心跳线程对同一个Drive租约的全部读写。
        if not path.is_file():  # 活跃训练期间租约不应被人工删除或挂载丢失。
            raise RuntimeError(f"训练中的Drive租约已经丢失：{path}")  # 立即停止继续更新以避免失去跨运行时互斥。
        payload = read_json(path)  # 读取当前Drive租约防止另一个进程替换文件。
        if payload.get("token") != token:  # 令牌不同说明当前进程不再拥有该实验目录。
            raise RuntimeError(f"训练中的Drive租约所有者发生变化：{path}")  # 不允许旧进程继续写滚动断点。
        payload["heartbeat_at_unix"] = time.time()  # 每次调用都推进最后心跳时间。
        if phase is not None:  # 后台纯心跳不应覆盖主线程登记的当前阶段。
            payload["phase"] = str(phase)  # 只有主线程提供阶段时才更新人类可读状态。
        if global_step is not None:  # 后台纯心跳不应把最近训练游标重置为空值。
            payload["global_step"] = int(global_step)  # 保存最近完整返回的绝对训练步。
        if cycle_step is not None:  # Base分支的零游标和Cycle分支正游标都由主线程显式更新。
            payload["cycle_step"] = int(cycle_step)  # 保存最近完整返回的Cycle独立步数。
        token_suffix = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]  # 为当前所有者生成不会与其他运行时共用的安全临时文件后缀。
        temporary = path.with_name(f".{path.name}.{token_suffix}.partial")  # 使用令牌专属临时文件避免不同运行时争用固定partial名称。
        if temporary.exists():  # 同一所有者上一次异常心跳可能留下精确临时文件。
            temporary.unlink()  # 只清理当前令牌对应且从未正式发布的临时租约。
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # 一次写完下一版租约内容。
        current = read_json(path)  # 在正式替换前再次确认所有权没有发生变化。
        if current.get("token") != token:  # 写临时文件期间若所有权改变则绝不覆盖新运行时。
            temporary.unlink()  # 删除只属于当前旧令牌的未提交临时文件。
            raise RuntimeError(f"Drive租约在心跳提交前已经换主：{path}")  # 让旧进程在下一个安全边界停止。
        os.replace(temporary, path)  # 二次所有权检查通过后原子发布最新心跳。
        published = read_json(path)  # 从Drive读回正式租约确认原子替换结果。
        if published.get("token") != token:  # 发布后令牌异常意味着互斥保证已经失效。
            raise RuntimeError(f"Drive租约心跳发布后身份异常：{path}")  # 拒绝继续更新模型或断点。


def start_drive_lease_heartbeat(lease: tuple) -> tuple:  # 启动独立后台心跳，完整断点序列化、复制和回读哈希期间也持续保护租约。
    stop_event = threading.Event()  # 使用线程安全事件请求后台心跳正常退出。
    errors = []  # 保存后台线程遇到的首个异常并由主线程在安全边界重新抛出。
    def heartbeat_worker() -> None:  # 定义只负责刷新租约且不接触模型状态的后台工作函数。
        while not stop_event.wait(DRIVE_LEASE_HEARTBEAT_SECONDS):  # 每六十秒等待一次并允许退出请求立即唤醒。
            try:  # Drive暂时异常时记录错误而不让后台线程静默消失。
                refresh_drive_lease(lease)  # 只更新时间，不覆盖主线程登记的阶段和训练游标。
            except BaseException as error:  # 捕获挂载、解析或所有权异常供主线程处理。
                errors.append(error)  # 保存真实异常对象以保留完整错误类型和信息。
                return  # 心跳失去可信所有权后立即结束后台线程。
    thread = threading.Thread(target=heartbeat_worker, name="cycle-drive-lease-heartbeat", daemon=True)  # 创建不会阻止Python异常退出的专用守护线程。
    thread.start()  # 在任何耗时来源哈希、模型构建或断点上传之前启动持续心跳。
    return stop_event, thread, errors  # 返回停止事件、线程和错误容器供主流程管理。


def raise_if_drive_heartbeat_failed(errors: list) -> None:  # 在每个batch和断点边界把后台心跳异常转回主训练线程。
    if errors:  # 非空表示后台线程已经无法安全维持当前Drive所有权。
        raise RuntimeError(f"Drive后台心跳失败：{errors[0]!r}") from errors[0]  # 停止继续训练并保留原始异常因果链。


def release_drive_lease(lease: tuple) -> None:  # 正常、失败或安全停止时释放当前进程自己的Drive租约。
    path, token, mutex = lease  # 取出租约文件路径、当前进程令牌和线程互斥锁。
    with mutex:  # 等待可能正在收尾的后台心跳完全离开租约读写区。
        if not path.exists():  # 文件被人工删除时无需再次操作。
            return  # 直接结束清理而不掩盖原训练结果。
        payload = read_json(path)  # 读取当前租约以防另一个进程已经替换该文件。
        if payload.get("token") != token:  # 令牌不同说明当前进程不再拥有这个租约。
            raise RuntimeError(f"拒绝删除不属于当前进程的 Drive 租约：{path}")  # 防止清理动作解除其他运行时的保护。
        path.unlink()  # 只删除经过令牌确认的当前租约文件。


def parse_args(argv=None) -> argparse.Namespace:  # 解析阶段训练只接受的两个显式参数。
    parser = argparse.ArgumentParser(description=__doc__)  # 使用文件顶部中文职责作为帮助说明。
    parser.add_argument("--plan", type=Path, required=True)  # 要求明确指定准备 Notebook 生成的计划 JSON。
    parser.add_argument("--run-approved", action="store_true")  # 要求用户主动添加训练授权，防止准备阶段误启动。
    args = parser.parse_args(argv)  # 解析实际命令行或测试传入的参数列表。
    if not args.run_approved:  # 只给计划路径不代表已经授权消耗 GPU。
        raise RuntimeError("缺少 --run-approved；本入口不会在准备阶段自动训练。")  # 用失败关闭保护用户算力。
    args.plan = args.plan.expanduser().resolve()  # 将计划路径规范为绝对路径便于合同记录。
    if not args.plan.is_file() or args.plan.is_symlink():  # 计划必须是真实普通文件。
        raise FileNotFoundError(f"实验计划不存在或是软链接：{args.plan}")  # 避免路径在运行中被替换。
    return args  # 返回已经确认包含人工授权的参数。


def main(argv=None) -> None:  # 执行一个计划的公平分叉、可恢复训练和持久保存。
    args = parse_args(argv)  # 首先确认计划路径和人工训练授权。
    torch.set_num_threads(TORCH_NUM_THREADS)  # 在构建数据与模型前恢复学生12号长期训练使用的两个CPU线程。
    raw_plan = read_json(args.plan)  # 读取 Notebook 生成的计划原文。
    plan = validate_plan(raw_plan)  # 失败关闭地核对父断点、步数和输出目录。
    device = require_runtime()  # 训练前确认当前是显存足够的 Colab CUDA 环境。
    output = plan["_persistent_path"]  # 使用 Google Drive 持久目录保存可恢复状态。
    output.mkdir(parents=True, exist_ok=True)  # 创建当前计划专用目录。
    lock_handle = acquire_run_lock(plan["plan_name"])  # 防止两个单元同时更新同一实验。
    drive_lease = None  # 在取得跨Colab租约前保持空值，便于安全处理获取失败。
    heartbeat_control = None  # 取得租约并成功启动后台心跳后才保存线程控制对象。
    try:  # 租约获取失败时只释放本地锁，绝不改写另一个活跃运行的状态文件。
        drive_lease = acquire_drive_lease(output, plan["plan_name"])  # 阻止其他Colab运行时同时写当前实验目录。
        heartbeat_control = start_drive_lease_heartbeat(drive_lease)  # 在来源哈希和任何大断点上传之前启动六十秒后台心跳。
    except BaseException:  # 已有租约或Drive写入失败时进入只清理本地资源的分支。
        if drive_lease is not None:  # 租约取得后若线程启动失败则尽力释放自己的租约。
            try:  # 清理异常不能掩盖真正的获取或线程启动问题。
                release_drive_lease(drive_lease)  # 只删除令牌仍属于当前进程的租约。
            except BaseException as cleanup_error:  # Drive清理失败只附加到终端供排查。
                print(f"启动失败后的Drive租约清理也发生错误：{cleanup_error!r}", file=sys.stderr, flush=True)  # 保留原始异常作为主错误。
        lock_handle.close()  # 释放本地GPU工作流锁，避免当前失败阻塞后续正确启动。
        raise  # 原样报告跨运行时租约问题，不触碰共享状态。
    heartbeat_stop, heartbeat_thread, heartbeat_errors = heartbeat_control  # 解包后台心跳的停止事件、线程和错误容器。
    stop = {"requested": False}  # 信号处理器只修改这个轻量标记。
    previous_handlers = {}  # 保存原信号处理器以便退出时恢复。
    state = {"status": "preparing", "plan_name": plan["plan_name"], "pid": os.getpid()}  # 建立可在失败时写出的最小状态。
    attempt_dir = None  # 完成合同与恢复来源核验后才创建本次尝试专用日志目录。
    contract_ready = False  # 正式合同确认前的预检失败不能污染持久实验目录。
    progress_bar = None  # 训练数据和模型就绪后才创建逐batch百分比进度条。
    batch_loss_buffer = []  # 暂存最近最多两百个batch的九项损失，并保证在每个完整断点前先写入Drive。
    try:  # 从父身份核验到训练完成的全部步骤都受统一失败记录保护。
        raise_if_drive_heartbeat_failed(heartbeat_errors)  # 在任何正式来源核验前确认后台租约线程仍然健康。
        for signum in (signal.SIGINT, signal.SIGTERM):  # 同时处理手动停止和运行时终止请求。
            previous_handlers[signum] = signal.getsignal(signum)  # 保存进入本函数前的处理器。
            signal.signal(signum, lambda *_: stop.update(requested=True))  # 只请求在完整批次边界停止。
        refresh_drive_lease(drive_lease, "validating_sources")  # 来源核验期间先发布心跳，浏览器断网不会释放实验所有权。
        parent_sha = sha256_file(plan["_parent_path"])  # 分块核验1.4GB父断点的真实文件身份。
        if parent_sha != plan["parent_checkpoint_sha256"]:  # 实际父断点必须逐字节等于计划登记值。
            raise RuntimeError(f"父断点 SHA256 不一致：实际 {parent_sha}，计划 {plan['parent_checkpoint_sha256']}")  # 禁止从同名错误文件分叉。
        proposed_contract = build_run_contract(plan, parent_sha, device)  # 生成本阶段不可静默改变的运行合同。
        contract_path = output / "run_contract.json"  # 设置合同在 Drive 中的固定位置。
        if contract_path.exists():  # 重启已有实验时必须复用完全相同的合同。
            contract = read_json(contract_path)  # 读取首次启动保存的正式合同。
            if contract != proposed_contract:  # 源码、计划、父断点或数学设置变化都会触发差异。
                raise RuntimeError("现有运行合同与本次计划或源码不同；请新建实验名，不能覆盖续训。")  # 防止实验身份混合。
        else:  # 首次启动当前计划时进入合同发布分支。
            contract_partial = output / "run_contract.json.partial"  # 定位上次在正式提交前中断的精确合同临时文件。
            if contract_partial.exists():  # 临时文件从未成为正式合同，可以在当前租约保护下安全丢弃。
                contract_partial.unlink()  # 只清理这一个明确的未提交临时文件，不删除任何正式结果。
            existing = [path for path in output.iterdir() if path.name != "ACTIVE_RUN.lock" and not (path.name.startswith(".ACTIVE_RUN.lock.") and path.name.endswith(".partial"))]  # 忽略正式租约和后台心跳瞬时临时文件，并检查其他无法解释的旧产物。
            if existing:  # 没有合同却有文件意味着目录不是安全的新实验目录。
                raise RuntimeError(f"输出目录已有无合同文件：{existing[:5]}")  # 不自动删除或覆盖用户数据。
            contract = proposed_contract  # 将本次审核过的合同设为正式合同。
            write_json_atomic(contract_path, contract)  # 在任何模型更新前原子保存合同。
        contract_ready = True  # 已有合同通过一致性检查或新合同已完整原子提交。
        contract_sha = canonical_sha256(contract)  # 生成写入每个子断点的稳定合同身份。
        paired_identity_path = ensure_paired_identity(plan, contract)  # 强制同组Base与Cycle共享父断点、源码、数据和运行环境。
        refresh_drive_lease(drive_lease, "building_models")  # 模型加载前刷新心跳，长时间准备不会被新运行时误判成陈旧。
        config, teacher, student, vocoder, discriminator, diffusion, mel_mean, mel_std, student_optimizer, discriminator_optimizer = build_components(device, plan["cycle_enabled"])  # 按原顺序构建并冻结依赖模型。
        committed_checkpoint = load_committed_stage_checkpoint(output, contract_sha, plan)  # 优先读取正式指针并在必要时自动回退上一代完整槽。
        if committed_checkpoint is not None:  # 已经保存过当前阶段时从最近可验证的完整状态续训。
            payload, source_path, source_sha = committed_checkpoint  # 取得完整训练态、双槽路径和真实SHA256。
            active_checkpoint_path = source_path  # 记录当前活跃槽，使下一次保存只覆盖另一槽。
        else:  # 首次运行当前分支时从共享完整父断点建立新实验。
            payload = validate_parent_checkpoint(load_checkpoint(plan["_parent_path"]), plan)  # 严格拒绝权重-only或错误步数父文件。
            source_path = plan["_parent_path"]  # 记录共享父断点路径。
            source_sha = parent_sha  # 记录已经核验的共享父断点身份。
            active_checkpoint_path = None  # 首次保存尚无阶段槽需要保护并固定写入latest槽。
        retained_indices = load_training_indices()  # 读取学生12号固定39335条训练人口。
        planner = GlobalBatchPlan(retained_indices)  # 建立可从任意绝对步恢复的1229批顺序规划器。
        dataset = baseline.VoiceGradDataset(ROOT / "data", "train", segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)  # 复用学生12号128帧裁剪和BNF对齐数据集。
        if len(dataset) != 39387:  # 原训练清单必须仍为39387条。
            raise RuntimeError(f"训练数据条数变化：{len(dataset)}")  # 数据身份变化时不继续。
        global_step, cycle_step, rng_restore_audit = restore_state(payload, student, discriminator, student_optimizer, discriminator_optimizer, device)  # 所有模型、数据对象和审计完成后恢复两模型、两Adam并记录真实RNG恢复方式。
        del payload  # 释放 CPU 中重复持有的大型断点字典以节省内存。
        refresh_drive_lease(drive_lease, "restored_state", global_step, cycle_step)  # 完整恢复模型、优化器和随机状态后立即发布真实续点游标。
        save_every = int(plan.get("save_every_updates", 2000))  # 最多每2000个完整更新写一次1.4GB滚动状态，兼顾稳定与速度。
        save_every_seconds = int(plan.get("save_every_seconds", 1800))  # 即使训练较慢也至少每三十分钟产生一个可跨虚拟机恢复的完整断点。
        status_every = int(plan.get("status_every_updates", 20))  # 默认每20步保存一次轻量状态和窗口损失。
        if save_every < 1 or save_every_seconds < 1 or status_every < 1:  # 步数、时间和状态三个间隔都必须是正整数。
            raise RuntimeError("保存和状态间隔必须为正整数。")  # 拒绝永不保存或非法取模。
        started = time.monotonic()  # 记录本次进程启动后的单调时间。
        resumed_step = global_step  # 保存本次尝试的起始游标用于速度估算。
        last_saved_step = global_step if active_checkpoint_path is not None else plan["expected_start_global_step"]  # 记录最近持久完整断点或共享父状态的步数。
        last_saved_monotonic = time.monotonic()  # 从当前恢复点重新计时，三十分钟后强制写下一份完整状态。
        window = []  # 收集最近若干完整批次的九项损失。
        attempt_number = 1  # 每次重连都使用新的日志目录，避免重做旧断点后的步数与旧日志混在一起。
        while (output / f"attempt_{attempt_number:04d}").exists():  # 顺序查找第一个尚未使用的尝试编号。
            attempt_number += 1  # 已存在的尝试目录永久保留并跳到下一个编号。
        attempt_dir = output / f"attempt_{attempt_number:04d}"  # 为本次进程分配独立且单调递增的日志目录。
        attempt_dir.mkdir()  # 排他创建本次尝试目录，意外重复编号会立即失败。
        attempt_record = {"attempt": attempt_number, "plan_name": plan["plan_name"], "paired_identity": str(paired_identity_path), "resume_checkpoint": str(source_path), "resume_checkpoint_sha256": source_sha, "resume_global_step": global_step, "resume_cycle_step": cycle_step, "rng_restore_audit": rng_restore_audit, "runtime_identity": runtime_identity(device), "started_at_unix": time.time()}  # 记录每次Colab连接的真实恢复来源、RNG恢复证据和已锁运行环境。
        write_json_atomic(attempt_dir / "resume.json", attempt_record)  # 在训练第一批前持久保存本次尝试的恢复证据。
        append_json_line(output / "attempts.jsonl", attempt_record)  # 持久保存本次尝试，不把可变硬件写进数学合同。
        state.update(status="running", global_step=global_step, cycle_step=cycle_step, target_global_step=plan["_target_global_step"], last_saved_global_step=last_saved_step, resume_checkpoint=str(source_path), rng_restore_audit=rng_restore_audit)  # 发布正式循环开始前的状态和随机状态恢复证据。
        write_json_atomic(output / "status.json", state)  # 将当前游标写到 Drive 供断线后人工查看。
        stage_total_updates = plan["_target_global_step"] - plan["expected_start_global_step"]  # 计算当前100→150阶段固定需要完成的61450个batch。
        stage_completed_updates = global_step - plan["expected_start_global_step"]  # 计算从共享父断点起已经永久恢复到的batch数量。
        resume_percent = 100.0 * stage_completed_updates / stage_total_updates  # 计算重连时进度条应从哪个真实百分比开始。
        print(f"从Drive完整断点恢复：阶段已完成 {stage_completed_updates:,}/{stage_total_updates:,} batch（{resume_percent:.2f}%），global_step={global_step}。", flush=True)  # 明确显示断线前已经跑到的位置。
        progress_bar = tqdm(total=stage_total_updates, initial=stage_completed_updates, desc=plan["plan_name"], unit="batch", dynamic_ncols=True, mininterval=0.5, smoothing=0.1)  # 创建每个batch推进一次的百分比、速度和预计剩余时间进度条。
        progress_bar.set_postfix(saved_step=last_saved_step, refresh=False)  # 在第一批开始前显示最近可跨虚拟机恢复的永久步数。
        refresh_drive_lease(drive_lease, "running", global_step, cycle_step)  # 正式循环开始前把恢复游标写入Drive心跳。
        for next_step in range(global_step + 1, plan["_target_global_step"] + 1):  # 从下一个未完成绝对步持续到阶段终点。
            raise_if_drive_heartbeat_failed(heartbeat_errors)  # 每个batch开始前确认完整断点保护线程仍然有效。
            if stop["requested"]:  # 只在上一批已经完整完成后响应停止请求。
                break  # 离开循环后保存当前完整状态。
            indices = planner.for_step(next_step)  # 取得与原学生12号完全相同的32条训练索引。
            batch = baseline.batch_for(dataset, indices, 101234 + next_step)  # 用原绝对步裁剪种子重建同一批128帧片段。
            training.seed_everything(PER_UPDATE_SEED_OFFSET + next_step)  # 用合同登记的原绝对步种子固定基础ACDD噪声和蒸馏时刻。
            if optimizer_uniform_step(student_optimizer) != global_step or optimizer_uniform_step(discriminator_optimizer) != global_step:  # 进入本批前核对两个Adam仍位于同一游标。
                raise RuntimeError("进入训练批前两个 Adam 游标与 global_step 不一致。")  # 防止在隐藏的半批状态上继续。
            relative_pass = (next_step - plan["expected_start_global_step"] - 1) // UPDATES_PER_PASS + 1  # 计算当前是分叉后的第几个一基 pass。
            losses = training.train_batch(batch, teacher, student, vocoder, discriminator, diffusion, student_optimizer, discriminator_optimizer, mel_mean, mel_std, device, config, epoch=relative_pass, global_step=global_step, cycle_step=cycle_step + 1 if plan["cycle_enabled"] else None)  # 基础路径读取进入批次前的绝对步，Cycle路径读取从1开始的独立更新编号。
            global_step = next_step  # 两个优化器都成功返回后才提交全局进度。
            if plan["cycle_enabled"]:  # 只有 Cycle 分支每个完整批次递增独立游标。
                cycle_step += 1  # 下一批 Cycle 将使用下一个配对和两程噪声编号。
            if optimizer_uniform_step(student_optimizer) != global_step or optimizer_uniform_step(discriminator_optimizer) != global_step:  # 返回后确认D和学生各恰好更新一次。
                raise RuntimeError("训练批完成后两个 Adam 游标没有同步前进一次。")  # 失败时保留上一个持久断点。
            if set(losses) != set(LOSS_KEYS) or any(not math.isfinite(float(value)) for value in losses.values()):  # 九项损失必须完整且全部有限。
                raise RuntimeError("训练批返回缺失、额外或非有限损失。")  # 坏数不能写入日志或断点。
            if plan["cycle_enabled"] and (float(losses["cycle_raw"]) <= 0.0 or not math.isclose(float(losses["cycle_weighted"]), config.lambda_cycle * float(losses["cycle_raw"]), rel_tol=1e-6, abs_tol=1e-9)):  # Cycle分支每批都必须真实执行且加权关系正确。
                raise RuntimeError("Cycle 计划没有产生正的 Cycle Loss 或加权关系异常。")  # 防止门控回归后把基线误标为Cycle结果。
            if not plan["cycle_enabled"] and (float(losses["cycle_raw"]) != 0.0 or float(losses["cycle_weighted"]) != 0.0):  # Base分支必须严格跳过全部Cycle计算。
                raise RuntimeError("Base 计划意外产生了非零 Cycle Loss。")  # 防止对照组被额外目标污染。
            loss_values = {key: float(losses[key]) for key in LOSS_KEYS}  # 将本批九项有限损失转换为可持久化的纯Python浮点数。
            window.append(loss_values)  # 保存本批损失用于每二十batch的稳定窗口平均。
            batch_loss_buffer.append({"global_step": global_step, "cycle_step": cycle_step, "relative_pass": relative_pass, "losses": loss_values, "recorded_at_unix": time.time()})  # 保存本批完整损失，断线重跑时可按global_step去重绘图。
            progress_bar.update(1)  # 每完成一个学生和判别器完整更新就推进百分比进度条一个batch。
            progress_bar.set_postfix(total=f"{loss_values['student_total']:.4f}", cycle=f"{loss_values['cycle_raw']:.4f}", d=f"{loss_values['discriminator']:.4f}", saved_step=last_saved_step, refresh=False)  # 在进度条右侧显示最近总损失、Cycle损失、判别器损失和永久断点步数。
            if global_step % status_every == 0 or global_step == plan["_target_global_step"]:  # 到轻量状态间隔或最终步时更新进度。
                means = {key: math.fsum(item[key] for item in window) / len(window) for key in LOSS_KEYS}  # 用稳定求和计算当前窗口九项均值。
                elapsed = time.monotonic() - started  # 计算本次尝试已经运行的秒数。
                completed_here = global_step - resumed_step  # 计算本次进程实际完成的更新数。
                remaining = plan["_target_global_step"] - global_step  # 计算距离阶段终点还剩多少更新。
                state.update(global_step=global_step, cycle_step=cycle_step, recent_loss_means=means, elapsed_seconds=elapsed, remaining_updates=remaining, estimated_remaining_seconds=remaining * elapsed / max(1, completed_here))  # 更新可读进度和粗略剩余时间。
                append_json_line(attempt_dir / "loss_windows.jsonl", {"global_step": global_step, "cycle_step": cycle_step, "count": len(window), "means": means})  # 在本次尝试独立目录记录窗口损失，重做步数不会伪装成唯一轨迹。
                if len(batch_loss_buffer) >= LOSS_CHUNK_UPDATES or global_step == plan["_target_global_step"]:  # 满两百条或到达终点时才提交原子块，避免数千个过小Drive文件。
                    append_json_lines(attempt_dir / "batch_losses.jsonl", batch_loss_buffer)  # 原子持久化当前缓冲区内每个batch的九项完整损失。
                    batch_loss_buffer = []  # 只有原子日志块完整发布后才清空内存缓冲区。
                write_json_atomic(output / "status.json", state)  # 原子刷新轻量状态文件。
                refresh_drive_lease(drive_lease, "running", global_step, cycle_step)  # 与轻量状态同步刷新租约心跳，浏览器断网时训练继续拥有目录。
                print(json.dumps(state, ensure_ascii=False, separators=(",", ":")), flush=True)  # 在Colab输出当前进度供用户查看。
                window = []  # 清空已持久化窗口并开始下一段统计。
            time_checkpoint_due = time.monotonic() - last_saved_monotonic >= save_every_seconds  # 判断距离上次完整断点是否已经达到三十分钟。
            if global_step % save_every == 0 or time_checkpoint_due or global_step == plan["_target_global_step"] or stop["requested"]:  # 步数、时间、终点或停止任一条件满足就保存完整状态。
                append_json_lines(attempt_dir / "batch_losses.jsonl", batch_loss_buffer)  # 断点提交前先保存尚未达到两百条的逐batch损失。
                batch_loss_buffer = []  # 清空已经写入Drive的损失缓冲区。
                recent_means = state.get("recent_loss_means", {key: float(losses[key]) for key in LOSS_KEYS})  # 若尚未到状态窗口则使用当前批损失。
                raise_if_drive_heartbeat_failed(heartbeat_errors)  # 开始昂贵完整断点前确认后台心跳没有失败。
                refresh_drive_lease(drive_lease, "saving_checkpoint", global_step, cycle_step)  # 大文件上传前刷新心跳并明确当前正在保存断点。
                pointer = save_full_checkpoint(output, plan["_local_path"], active_checkpoint_path, contract_sha, plan, global_step, cycle_step, config, student, discriminator, student_optimizer, discriminator_optimizer, recent_means)  # 经本地暂存后向非活跃槽原子提交可恢复1.4GB状态。
                raise_if_drive_heartbeat_failed(heartbeat_errors)  # 完整断点提交后立即检查上传期间的后台心跳状态。
                active_checkpoint_path = Path(pointer["path"])  # 指针正式提交后将刚写入的槽设为下一轮必须保护的活跃槽。
                last_saved_step = global_step  # 只有保存函数完整返回后才推进已持久游标。
                last_saved_monotonic = time.monotonic()  # 从刚完成的Drive断点重新计算三十分钟保存间隔。
                state.update(global_step=global_step, cycle_step=cycle_step, last_saved_global_step=last_saved_step, latest_checkpoint=pointer)  # 同步写入真实断点游标和身份，避免时间触发保存落在非二十倍数时状态滞后。
                write_json_atomic(output / "status.json", state)  # 断点提交后再更新状态指针。
                refresh_drive_lease(drive_lease, "running", global_step, cycle_step)  # 断点校验成功后恢复普通训练心跳阶段。
                progress_bar.set_postfix(total=f"{loss_values['student_total']:.4f}", cycle=f"{loss_values['cycle_raw']:.4f}", d=f"{loss_values['discriminator']:.4f}", saved_step=last_saved_step, refresh=True)  # 立即在进度条显示新的永久可恢复步数。
        if global_step > last_saved_step:  # 用户在非定期保存步停止时需要补写当前完整状态。
            append_json_lines(attempt_dir / "batch_losses.jsonl", batch_loss_buffer)  # 安全停止前保存最后不足两百条的逐batch损失。
            batch_loss_buffer = []  # 清空已经持久化的损失缓冲区。
            recent_means = state.get("recent_loss_means", {})  # 复用最近已记录损失窗口。
            raise_if_drive_heartbeat_failed(heartbeat_errors)  # 最终补写完整断点前确认后台心跳仍然有效。
            refresh_drive_lease(drive_lease, "saving_final_checkpoint", global_step, cycle_step)  # 最终补写断点前刷新租约避免重连误判超时。
            pointer = save_full_checkpoint(output, plan["_local_path"], active_checkpoint_path, contract_sha, plan, global_step, cycle_step, config, student, discriminator, student_optimizer, discriminator_optimizer, recent_means)  # 经本地暂存向非活跃槽持久化最后一个完整批次边界。
            raise_if_drive_heartbeat_failed(heartbeat_errors)  # 最终断点提交后立即检查上传期间的后台心跳状态。
            active_checkpoint_path = Path(pointer["path"])  # 记录最终正式指针当前指向的槽位。
            last_saved_step = global_step  # 更新最终已保存游标。
            state.update(last_saved_global_step=last_saved_step, latest_checkpoint=pointer)  # 记录最终断点身份。
        final_status = "completed" if global_step == plan["_target_global_step"] else "stopped_at_complete_batch"  # 区分达到阶段终点和用户安全停止。
        state.update(status=final_status, global_step=global_step, cycle_step=cycle_step, last_saved_global_step=last_saved_step, finished_at_unix=time.time())  # 完成最终状态记录。
        write_json_atomic(output / "status.json", state)  # 原子发布最终状态。
        write_json_atomic(attempt_dir / "completion.json", state)  # 在本次尝试目录保存完整结束或安全停止收据。
        refresh_drive_lease(drive_lease, final_status, global_step, cycle_step)  # 正常结束前发布最后心跳和最终可恢复游标。
    except BaseException as error:  # 捕获训练、校验和Drive写入错误并保留失败信息。
        state.update(status="failed", error=repr(error), failed_at_unix=time.time(), no_partial_checkpoint_written=True)  # 明确声明不会把半批状态写成断点。
        try:  # 即使Drive本身异常，也尽力发布轻量失败记录。
            if contract_ready and attempt_dir is not None and batch_loss_buffer:  # 已完成但尚未刷新的一小批损失仍可作为本次尝试诊断证据。
                append_json_lines(attempt_dir / "batch_losses.jsonl", batch_loss_buffer)  # 一次性保存失败前已经完整返回的batch损失。
            if contract_ready:  # 只有正式合同已经确认后才允许在持久实验目录发布状态。
                write_json_atomic(output / "status.json", state)  # 覆盖状态为失败并保留最近完整断点不动。
            if contract_ready and attempt_dir is not None:  # 只有正式合同和尝试目录都存在时才能写独立失败收据。
                write_json_atomic(attempt_dir / "failure.json", state)  # 保留本次尝试失败位置且不与其他重连日志混合。
        except BaseException as reporting_error:  # 报告失败不能覆盖原始训练异常。
            print(f"失败状态写入也发生错误：{reporting_error!r}", file=sys.stderr, flush=True)  # 仅在终端附加说明。
        raise  # 原样重新抛出最初错误，让Colab单元显示失败。
    finally:  # 无论成功、停止或失败都恢复信号并释放Drive租约和本地进程锁。
        if progress_bar is not None:  # 只有正式循环已经创建进度条时才需要关闭。
            progress_bar.close()  # 保留最终百分比、速度、预计时间和最近损失显示。
        for signum, handler in previous_handlers.items():  # 逐个恢复进入主流程前的信号处理器。
            signal.signal(signum, handler)  # 避免本入口影响同一Colab中的后续代码。
        heartbeat_stop.set()  # 请求后台心跳线程立即结束等待并停止访问Drive租约。
        heartbeat_thread.join(timeout=10.0)  # 最多等待十秒让正在进行的一次小型心跳安全退出。
        if heartbeat_thread.is_alive():  # 极端Drive挂起时不无限阻塞主进程清理。
            print("Drive后台心跳线程在十秒内没有退出；租约将保留等待超时接管。", file=sys.stderr, flush=True)  # 保留租约比误删仍在访问的文件更安全。
        else:  # 后台线程已经完全离开租约临界区时可以尝试正常释放。
            try:  # 租约清理错误只报告，不覆盖训练成功或原始训练异常。
                release_drive_lease(drive_lease)  # 只删除当前进程自己创建且令牌仍匹配的跨Colab租约。
            except BaseException as cleanup_error:  # 捕获Drive挂载或JSON读取异常。
                print(f"Drive租约清理发生错误并已保留租约：{cleanup_error!r}", file=sys.stderr, flush=True)  # 让三十分钟超时机制稍后安全接管。
        lock_handle.close()  # 无论Drive清理结果如何都释放当前VM中的文件锁。


if __name__ == "__main__":  # 只有直接执行本文件时才进入训练主流程。
    main()  # 导入本模块只定义函数，不会自动消耗 GPU。
