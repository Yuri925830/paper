from __future__ import annotations  # 导入启动器时不提前执行模型计算。
import argparse  # 支持从单单元格或终端明确选择训练任务。
from collections import deque  # 保留安装失败前最后几十行真实输出，方便直接看原因。
import csv  # 读取测试和开发语音清单，只缓存必要参考音频。
import hashlib  # 核对断点和小型来源文件的完整身份。
import importlib.util  # 检查 Colab 是否已有环境管理工具。
import json  # 读写永久指针、租约和缓存完成标记。
import os  # 使用排他创建和原子替换保护输出文件。
import shutil  # 顺序复制和读取真实磁盘可用空间。
import shlex  # 保留Python原有编译和链接参数，只替换缺失的编译器程序。
import socket  # 为当前运行时记录可辨认的主机名称。
import subprocess  # 在兼容的独立 Python 环境中运行唯一训练进程。
import sys  # 定位当前解释器和禁止本地 Mac 启动训练。
import tarfile  # 顺序展开已经永久制作完成的必要数据分卷。
import threading  # 为长时间训练保留后台租约心跳。
import time  # 判断租约过期及记录实际运行时间。
import uuid  # 为临时文件和运行时所有权分配唯一名称。
from pathlib import Path, PurePosixPath  # 安全处理中文路径并拒绝归档路径跳转。
from loss_tracking import atomic_json, plot_losses  # 共享原子JSON和真实损失绘图工具。

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/FsatVoiceGrad_服务器原文件_20260906")  # 固定用户已经恢复材料的永久总目录。
DEFAULT_LOCAL_PROJECT = Path("/root/FsatVoiceGrad")  # 沿用原训练辅助代码要求的高速SSD项目路径。
DEFAULT_RELEASE_NAME = "report_experiments_20261008"  # 新代码和启动文件的永久目录名称。
DEFAULT_RUN_NAME = "report_experiment_runs_20261008"  # 与旧Cycle训练结果分开的新正式记录目录。
FVG_ARMS = ("base_100", "early_cycle_100", "base_post", "cycle_post")  # 固定最新报告中的四种训练分支。
MAIN_SEEDS = (1234, 2234, 3234)  # 每个正式分支按相同三个种子依次运行。
LAMBDA_CANDIDATES = (0.0125, 0.05, 0.1)  # 使用报告中预先指定的三个Cycle权重候选。
SHARD_COUNTS = {"mel": 20, "bnf": 20, "spk_emb": 20, "metadata": 1, "stats": 1}  # 五类必要数据合计62个无压缩分卷。
VQ_TASKS = ("cd_512", "cd_1024", "cd_4096", "cd_8192", "initial_grid", "vq_512", "evq_512", "vqcd_512", "evqcd_512")  # 公共复现入口登记的实际任务名称。

def sha256_file(path):  # 对单个模型或小型文件分块计算身份，不扫描全部训练数组。
    digest = hashlib.sha256()  # 为当前文件创建计算器。
    with Path(path).open("rb") as handle:  # 只读打开明确的单个文件。
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):  # 每次最多读取十六MiB。
            digest.update(block)  # 将当前文件块加入摘要。
    return digest.hexdigest()  # 返回可写入收据的身份字符串。

def _load_json(path, fallback=None):  # 读取可选轻量JSON文件。
    target = Path(path)  # 统一路径类型。
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else fallback  # 文件不存在时使用明确默认值。

def copy_file_resumable(source, destination, description=None, expected_sha256=None):  # 逐字节续传单个必要文件并在完整后发布。
    from tqdm.auto import tqdm  # 在Colab显示实际字节进度、速度和预计时间。
    source, destination = Path(source), Path(destination)  # 统一来源和缓存路径。
    if not source.is_file() or source.is_symlink():  # 永久来源必须是普通文件。
        raise FileNotFoundError(f"缺少必要永久文件：{source}")  # 在训练前直接列出缺项。
    stat = source.stat()  # 读取源文件长度和修改时间。
    signature = {"source": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "expected_sha256": expected_sha256}  # 把续传前缀绑定到同一永久来源。
    marker = destination.with_name("." + destination.name + ".cache_ready.json")  # 为单个缓存文件保存完成收据。
    if destination.is_file() and destination.stat().st_size == stat.st_size and _load_json(marker) == signature:  # 同一运行时完整缓存直接复用。
        return destination  # 不重新读取模型或复制数据。
    destination.parent.mkdir(parents=True, exist_ok=True)  # 创建当前缓存文件所需目录。
    partial = destination.with_name(destination.name + ".cache_partial")  # 保存可在同一运行时断点继续的文件前缀。
    partial_marker = partial.with_name(partial.name + ".json")  # 记录当前前缀来自哪一个永久来源。
    if partial.is_file() and _load_json(partial_marker) != signature:  # 来源改变时不能把不同文件的字节拼接。
        partial.unlink()  # 只删除当前精确识别的本地不可信前缀。
    offset = partial.stat().st_size if partial.is_file() else 0  # 从上次已经缓存的字节位置继续。
    if offset > stat.st_size:  # 超长前缀不可能属于当前来源。
        partial.unlink()  # 只清理当前错误缓存前缀。
        offset = 0  # 改为从当前来源开头复制。
    atomic_json(partial_marker, signature)  # 在追加字节前绑定续传身份。
    with tqdm(total=stat.st_size, initial=offset, desc=description or source.name, unit="B", unit_scale=True, unit_divisor=1024, dynamic_ncols=True, disable=stat.st_size < 8 * 1024 * 1024) as progress:  # 大文件显示续传字节进度，小源码由外层总进度显示。
        with source.open("rb") as source_handle, partial.open("ab") as destination_handle:  # 同时打开Drive来源和本地续传目标。
            source_handle.seek(offset)  # 跳过已经完整缓存的前缀。
            for block in iter(lambda: source_handle.read(16 * 1024 * 1024), b""):  # 顺序传输避免Drive大量随机读取。
                destination_handle.write(block)  # 将当前来源块追加到本地SSD。
                progress.update(len(block))  # 每个块实时推进进度。
            destination_handle.flush()  # 将Python缓冲交给系统。
            os.fsync(destination_handle.fileno())  # 确保本地文件完整写入。
    if partial.stat().st_size != stat.st_size:  # 长度不足表示未完整传输。
        raise IOError(f"缓存文件未复制完整：{partial}")  # 保留可续传前缀并报告异常。
    if expected_sha256 is not None and sha256_file(partial) != expected_sha256:  # 有权威模型摘要时只核对该模型一次。
        raise IOError(f"必要模型缓存内容与正式来源不一致：{source}")  # 拒绝错误权重进入训练。
    os.replace(partial, destination)  # 完整缓存成功后发布正式文件。
    atomic_json(marker, signature)  # 完成收据让同一运行时重跑直接跳过。
    return destination  # 返回真实就位的本地文件路径。

def copy_tree(source, destination, title):  # 只复制明确需要的源码或固定模型目录。
    from tqdm.auto import tqdm  # 显示逐文件真实缓存进度。
    source, destination = Path(source), Path(destination)  # 统一来源与目标根目录。
    if not source.is_dir():  # 必要目录缺失时不使用其他近似目录代替。
        raise FileNotFoundError(f"缺少必要永久目录：{source}")  # 在启动前指出准确缺项。
    files = sorted(path for path in source.rglob("*") if path.is_file() and not set(path.parts) & {".git", "__pycache__", ".ipynb_checkpoints"} and path.name != ".DS_Store" and path.suffix != ".pyc")  # 跳过版本库和生成缓存，只遍历指定必要目录。
    for path in tqdm(files, desc=title, unit="文件", dynamic_ncols=True):  # 每完成一个必要文件就更新进度。
        relative = path.relative_to(source)  # 保留上游模块原目录结构。
        copy_file_resumable(path, destination / relative)  # 当前文件中断后可以重新运行继续。
    return len(files)  # 返回实际缓存文件数量供收据记录。

class RunLease:  # 防止两个运行时同时写入同一条正式训练任务。
    def __init__(self, run_dir, ttl_seconds=1800, heartbeat_seconds=60):  # 每分钟更新一次所有权心跳，半小时无响应才允许接管。
        self.root = Path(run_dir)  # 保存当前任务独立的永久输出目录。
        self.path = self.root / "run_lease.json"  # 固定当前任务的唯一所有权位置。
        self.token = uuid.uuid4().hex  # 分配当前运行时唯一所有者编号。
        self.ttl = int(ttl_seconds)  # 保存租约过期时间。
        self.interval = int(heartbeat_seconds)  # 保存心跳间隔。
        self.stop_event = threading.Event()  # 为退出训练后停止心跳准备同步事件。
        self.error = None  # 记录后台心跳可能出现的存储异常。
        self.thread = None  # 暂不启动后台线程。

    def _value(self):  # 生成当前所有权的真实状态。
        return {"token": self.token, "host": socket.gethostname(), "pid": os.getpid(), "heartbeat_unix": time.time(), "ttl_seconds": self.ttl}  # 保存主机、进程和最后活动时间。

    def assert_owned(self):  # 每次发布断点前核对所有权仍属于当前运行时。
        value = _load_json(self.path, {})  # 读取当前正式所有者。
        if self.error is not None or value.get("token") != self.token:  # 心跳失败或已被其他运行时接管时停止写入。
            raise RuntimeError(f"当前任务租约已经丢失：{self.path}；{self.error or '所有者发生改变'}")  # 防止并行进程相互覆盖结果。

    def _heartbeat(self):  # 训练期间更新轻量存活信号，不发起模型计算。
        while not self.stop_event.wait(self.interval):  # 等待一分钟后再更新，退出信号可以提前结束等待。
            try:  # 将磁盘异常保存供主线程及时检查。
                self.assert_owned()  # 先确认没有被其他运行时接管。
                atomic_json(self.path, self._value())  # 发布当前所有者的新活动时间。
            except Exception as exc:  # 存储失败或所有权丢失都不能被忽略。
                self.error = str(exc)  # 保存主线程可以读取的错误原因。
                return  # 停止心跳，让主线程在检查时安全退出。

    def __enter__(self):  # 在启动唯一训练进程前取得当前任务租约。
        self.root.mkdir(parents=True, exist_ok=True)  # 创建独立输出目录。
        previous = _load_json(self.path)  # 读取可能存在的上一运行时所有权。
        if previous is not None:  # 已有所有者时先检查是否仍然活动。
            age = time.time() - float(previous.get("heartbeat_unix", time.time()))  # 计算上次心跳距离当前时间。
            if age <= int(previous.get("ttl_seconds", self.ttl)):  # 最近半小时内有心跳时不能并行覆盖。
                raise RuntimeError(f"这个任务已有运行时执行，最后心跳在 {age:.0f} 秒前：{self.root}。请让朋友运行另一个任务文件。")  # 明确提示使用不同任务并行。
            stale = self.path.with_name("run_lease.expired." + str(int(time.time())) + ".json")  # 给过期记录设置可审计保留路径。
            os.replace(self.path, stale)  # 保存旧租约后允许新运行时接续。
        handle = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # 排他创建同一任务的当前所有权。
        with os.fdopen(handle, "w", encoding="utf-8") as stream:  # 用受控句柄保存新租约。
            json.dump(self._value(), stream, ensure_ascii=False)  # 写入当前运行时身份。
            stream.flush()  # 将文本缓冲交给挂载文件系统。
            os.fsync(stream.fileno())  # 要求文件系统接收当前租约内容。
        self.assert_owned()  # 写入后回读检查当前所有权。
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)  # 创建低开销后台心跳线程。
        self.thread.start()  # 在唯一训练进程运行期间保护输出目录。
        return self  # 把租约交给调用方用于检查。

    def __exit__(self, exc_type, exc_value, traceback):  # 正常结束或异常退出时释放自身租约。
        self.stop_event.set()  # 通知后台心跳线程停止。
        if self.thread is not None:  # 只有真正启动过线程才等待退出。
            self.thread.join(timeout=5)  # 最多等待五秒，避免影响用户响应。
        value = _load_json(self.path, {})  # 读取退出瞬间的真实所有者。
        if value.get("token") == self.token:  # 只释放自己创建的租约。
            self.path.unlink()  # 删除精确的本任务临时所有权文件，不删除训练数据或结果。
        return False  # 让原训练异常原样传回Notebook。

def checkpoint_store(run_dir, state, global_step, local_temp_root=None):  # 将完整训练态发布到Drive双槽并保留上一完整代。
    import torch  # 只在训练器真正提交断点时导入张量序列化工具。
    root = Path(run_dir)  # 固定当前任务唯一永久结果目录。
    root.mkdir(parents=True, exist_ok=True)  # 创建永久输出目录。
    owner = os.environ.get("VOICE_RESEARCH_LEASE_TOKEN")  # 读取启动器交给唯一训练子进程的所有权编号。
    lease_root = Path(os.environ.get("VOICE_RESEARCH_LEASE_ROOT", str(root))).resolve()  # VQ四个说话人断点可共同借用上级任务租约。
    if owner and lease_root != root.resolve() and lease_root not in root.resolve().parents:  # 上级租约只保护自己的子目录，不能借用其他实验的租约。
        raise RuntimeError("断点目录不属于启动器租约范围。")  # 防止拿一个任务的令牌去覆盖另一个任务。
    if owner and _load_json(lease_root / "run_lease.json", {}).get("token") != owner:  # 外层租约丢失时禁止发布新训练态。
        raise RuntimeError("断点保存时任务租约已丢失，保留此前完整断点并停止。")  # 防止两个运行时交叉覆盖完整状态。
    pointer_path = root / "checkpoint_pointer.json"  # 正式续训只读取这个完整提交指针。
    old = _load_json(pointer_path, {})  # 获取上一代已完成的训练态收据。
    current = old.get("current")  # 当前完整代在写入新代期间保持可恢复。
    step = int(global_step)  # 固定本断点对应的完整优化更新数。
    if current and step < int(current["global_step"]):  # 同一步验证结果允许更新，步数倒退禁止覆盖。
        raise RuntimeError("新断点的训练步数小于当前正式断点。")  # 避免错误回滚覆盖进度。
    slot = "checkpoint_b.pt" if current and Path(current["path"]).name == "checkpoint_a.pt" else "checkpoint_a.pt"  # 始终写入非活跃槽。
    temporary_root = Path(local_temp_root or "/root/voice_research_checkpoint_cache")  # 使用本地SSD序列化大张量。
    temporary_root.mkdir(parents=True, exist_ok=True)  # 创建本地断点暂存目录。
    local = temporary_root / (uuid.uuid4().hex + ".pt")  # 用唯一文件名防止不同任务共用暂存文件。
    payload = dict(state)  # 保留模型、双优化器、随机数和训练/验证游标的完整字典。
    payload["global_step"] = step  # 保存该训练态真实对应的累计更新数。
    with local.open("wb") as handle:  # 先在高速SSD写出完整模型状态。
        torch.save(payload, handle)  # 保存全部训练态而不仅是模型权重。
        handle.flush()  # 刷新序列化缓冲。
        os.fsync(handle.fileno())  # 确保本地暂存文件完整。
    digest = sha256_file(local)  # 为这一代完整训练态计算实际字节身份。
    destination = root / slot  # 定位当前非活跃永久槽。
    partial = root / (slot + ".writing." + uuid.uuid4().hex)  # Drive侧临时文件不作为可恢复断点。
    with local.open("rb") as source, partial.open("wb") as target:  # 顺序把本地完整断点上传到Drive。
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)  # 每次传输十六MiB，避免大内存副本。
        target.flush()  # 将上传缓冲交给Drive挂载层。
        os.fsync(target.fileno())  # 要求挂载文件系统接收所有断点字节。
    if partial.stat().st_size != local.stat().st_size or sha256_file(partial) != digest:  # 只有Drive完整副本通过内容校验才能发布。
        raise IOError(f"Drive断点未完整上传：{partial}")  # 旧完整断点继续可恢复。
    os.replace(partial, destination)  # 新代上传完整后替换非活跃槽。
    receipt = {"path": str(destination), "sha256": digest, "bytes": destination.stat().st_size, "global_step": step, "loss_sequence": int(payload.get("loss_committed_sequence", 0)), "saved_unix": time.time()}  # 绑定训练态与损失日志的共同安全游标。
    atomic_json(pointer_path, {"format_version": 1, "current": receipt, "previous": current})  # 最后一步才让续训指针指向新完整代。
    local.unlink()  # 删除精确的本次本地序列化暂存文件，永久两代断点仍保留。
    return receipt  # 返回真实完整提交收据供训练日志记录。

def checkpoint_load(run_dir, map_location="cpu"):  # 优先恢复最新完整代，损坏时回退上一完整代。
    import torch  # 只在训练器实际恢复状态时导入张量工具。
    root = Path(run_dir)  # 固定当前任务独立永久目录。
    pointer = _load_json(root / "checkpoint_pointer.json")  # 读取已完整提交的双槽指针。
    if pointer is None:  # 新任务尚未完成首次断点时从正确父状态开始。
        if any((root / name).exists() for name in ("checkpoint_a.pt", "checkpoint_b.pt", "loss_manifest.json")):  # 有旧正式产物而缺指针时不能视作全新任务。
            raise RuntimeError("当前实验已有断点或损失但完整指针缺失，请先恢复指针；不会从头覆盖旧结果。")  # 失败关闭并保留既有产物。
        return None, None  # 不把.partial或writing文件误认成断点。
    failures = []  # 收集损坏候选的实际原因。
    for key in ("current", "previous"):  # 先读最新代，再尝试保留的上一代。
        receipt = pointer.get(key)  # 取当前候选的正式收据。
        if not receipt:  # 第一代任务可能尚无上一完整代。
            continue  # 跳过不存在的候选。
        path = root / Path(receipt["path"]).name  # 只允许当前实验目录内的两个正式槽。
        try:  # 一个候选损坏不影响另一完整候选的恢复。
            if path.name not in {"checkpoint_a.pt", "checkpoint_b.pt"} or not path.is_file() or path.stat().st_size != int(receipt["bytes"]) or sha256_file(path) != receipt["sha256"]:  # 核对路径、大小和完整字节身份。
                raise IOError("断点大小、位置或内容摘要不一致")  # 未完整写入的文件不能用于续训。
            payload = torch.load(path, map_location=map_location, weights_only=False)  # 恢复包含优化器和随机数的完整字典。
            if not isinstance(payload, dict) or int(payload.get("global_step", -1)) != int(receipt["global_step"]) or int(payload.get("loss_committed_sequence", 0)) != int(receipt.get("loss_sequence", 0)):  # 张量内容游标必须和正式收据共同匹配。
                raise IOError("完整训练态与指针游标不一致")  # 拒绝状态和日志错配。
            if key == "previous":  # 当前代损坏时将正式指针回退为实际恢复代。
                atomic_json(root / "checkpoint_pointer.json", {"format_version": 1, "current": receipt, "previous": None, "fallback_reason": failures})  # 绘图和后续保存同步使用回退边界。
                print(f"已回退到上一完整断点：step {receipt['global_step']}")  # 让用户看到实际恢复进度。
            else:  # 最新完整代正常时直接报告真实游标。
                print(f"从Drive完整断点续训：step {receipt['global_step']}")  # 不把已回滚批次计入百分比。
            return payload, receipt  # 同时返回训练态和安全日志边界。
        except Exception as exc:  # 保存当前候选实际失败原因。
            failures.append(f"{key}: {exc}")  # 让无法恢复时的诊断完整可读。
    raise RuntimeError("两个正式断点均无法完整恢复：" + "; ".join(failures))  # 两代都损坏时禁止从头覆盖旧目录。

def _shard_plan(drive_root, project_root):  # 从已经永久保存的小清单重建必要分卷计划。
    shard_root = Path(drive_root) / "FastVoiceGrad_必要数据_持久分卷_20261005"  # 沿用朋友已完成的62卷永久数据。
    if not (shard_root / "全部分卷已完成.txt").is_file():  # 永久制作流程没有完整结束时不开始缓存。
        raise FileNotFoundError(f"必要永久分卷尚未完成：{shard_root}")  # 不重新制作归档或重复上传。
    plans = []  # 保存当前训练实际需要恢复的卷。
    for group, expected_count in SHARD_COUNTS.items():  # 仅选择mel、BNF、说话人向量、清单和统计。
        manifest = shard_root / f"{group}_文件清单.txt"  # 读取对应组的已固定路径清单。
        marker = shard_root / f"{group}_分卷已完成.txt"  # 核对永久制作结束标记。
        if not manifest.is_file() or not marker.is_file():  # 当前数据组缺材料时直接停止。
            raise FileNotFoundError(f"必要数据组缺少完整清单：{group}")  # 向用户指出准确的数据组。
        names = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]  # 只读取几MiB以内的路径文本。
        if not names or len(names) != len(set(names)):  # 路径清单必须非空且没有重复。
            raise RuntimeError(f"必要数据清单无效：{manifest}")  # 不用不完整清单猜测数据布局。
        if (len(names) + 1999) // 2000 != expected_count:  # 每卷最多2000文件的分卷规则必须对应正式数量。
            raise RuntimeError(f"必要分卷数量与固定计划不同：{group}")  # 阻止永久材料被误换后继续训练。
        manifest_sha = sha256_file(manifest)  # 只对小型路径清单计算来源身份。
        for index in range(expected_count):  # 按已经固定的卷序号逐一定位。
            path = shard_root / f"{group}_{index:04d}.tar"  # 当前卷是已完成的无压缩TAR。
            if not path.is_file() or path.stat().st_size <= 0:  # 缺卷或空卷不能开始训练。
                raise FileNotFoundError(f"缺少必要数据分卷：{path}")  # 一次指出准确缺失卷。
            members = names[index * 2000:(index + 1) * 2000]  # 当前卷必须对应清单中的固定连续切片。
            identity = {"source": str(path), "bytes": path.stat().st_size, "manifest_sha256": manifest_sha, "members": len(members)}  # 为同一运行时快跳过卷保存小收据。
            done = Path(project_root) / ".report_cache_markers" / (path.name + ".json")  # 保留每卷完成标记以支持安全重跑。
            plans.append((group, path, members, identity, done))  # 保存实际需要的完整恢复计划。
    return plans  # 返回62卷的小型计划，不遍历Drive的大量数组文件。

def _restore_shard(plan, project_root):  # 顺序恢复一卷并在完成后保存可复用标记。
    group, path, expected_members, identity, done = plan  # 拆出当前组、卷来源、路径白名单和收据。
    if _load_json(done) == identity:  # 当前运行时已经完整展开相同永久来源时直接复用。
        return False  # 不再次解包或读取几千个数组。
    with tarfile.open(path, mode="r|", bufsize=16 * 1024 * 1024) as archive:  # 使用十六MiB顺序缓冲读取Drive，减少默认十KiB小请求带来的等待。
        count = 0  # 记录本卷实际已展开的文件数。
        for member in archive:  # 文件读取和安全检查与实际解包合并一次完成。
            if count >= len(expected_members) or member.name != expected_members[count]:  # 内部条目必须逐项对应固定白名单。
                raise RuntimeError(f"分卷内容路径与清单不同：{path.name}:{member.name}")  # 防止不明文件写入项目。
            relative = PurePosixPath(member.name)  # 按归档标准解析当前成员路径。
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts or tuple(relative.parts[:2]) != ("data", group):  # 仅允许本组普通数据文件。
                raise RuntimeError(f"分卷成员路径或类型不安全：{member.name}")  # 拒绝软链接、设备和上级跳转。
            target = Path(project_root).joinpath(*relative.parts)  # 将安全白名单路径映射到本地SSD。
            target.parent.mkdir(parents=True, exist_ok=True)  # 创建当前说话人或统计子目录。
            partial = target.with_name(target.name + ".extract_partial")  # 当前数组先写入临时文件。
            source = archive.extractfile(member)  # 只读取当前普通成员的实际数据。
            if source is None:  # TAR声明普通文件却无法读取时不能跳过。
                raise IOError(f"无法读取分卷成员：{member.name}")  # 保留此前完整卷标记并停止当前卷。
            with source, partial.open("wb") as output:  # 将当前数组顺序写入本地SSD。
                shutil.copyfileobj(source, output, length=1024 * 1024)  # 逐MiB写入以控制内存占用。
            if partial.stat().st_size != member.size:  # 只对当前正在写入的文件核对长度。
                raise IOError(f"当前数据文件展开不完整：{member.name}")  # 当前卷不写完成标记，下次重跑即可继续恢复。
            os.replace(partial, target)  # 只有完整成员才替换当前缓存文件。
            count += 1  # 累计成功恢复的成员数。
        if count != len(expected_members):  # TAR提前结束不能伪称当前卷完成。
            raise IOError(f"必要分卷缺少文件：{path.name}")  # 当前卷重跑不会跳过。
    atomic_json(done, identity)  # 整卷成功关闭后才保存快速复用标记。
    return True  # 返回当前卷确实发生过解包。

def _reference_waves(project_root, drive_project):  # 只缓存开发转换和最终测试真正需要的参考音频。
    root, permanent = Path(project_root), Path(drive_project)  # 固定本地SSD与永久来源根目录。
    table_paths = [root / "data/metadata/test_manifest.csv", root / "exp/diagnostics/student_development_20260906/pairs.csv"]  # 测试100句和开发90对提供必要波形路径。
    paths = set()  # 去除多转换对共用的相同参考语音。
    for table in table_paths:  # 仅读取两份已固定的小型CSV。
        if not table.is_file():  # 表格不存在时交给训练前契约检查报出具体缺项。
            continue  # 此处不猜测其他音频来源。
        with table.open("r", encoding="utf-8", newline="") as handle:  # 按CSV标准处理字段。
            for row in csv.DictReader(handle):  # 逐行查看实际登记的参考来源。
                for field, value in row.items():  # 只筛选明确含波形路径的字段。
                    if value and "wav" in field.lower() and Path(value).suffix.lower() in {".wav", ".flac"}:  # 排除文本、数组和生成音频目录。
                        normalized = value.replace("\\", "/")  # 统一旧Windows与Linux路径分隔符。
                        if "/data/" in normalized:  # 历史绝对路径只使用项目内的数据后缀。
                            paths.add("data/" + normalized.split("/data/", 1)[1])  # 将同一项目文件映射到Drive来源。
                        elif normalized.startswith("data/"):  # 项目相对路径可以直接使用。
                            paths.add(normalized)  # 保留经过来源约束的相对音频路径。
    from tqdm.auto import tqdm  # 显示实际参考音频的逐文件复制进度。
    for relative in tqdm(sorted(paths), desc="缓存必要测试/开发参考音频", unit="语音", dynamic_ncols=True):  # 最多缓存固定协议实际引用的有限语音。
        pure = PurePosixPath(relative)  # 对清单来源路径再做安全解析。
        if pure.is_absolute() or ".." in pure.parts:  # 任何越界路径都不能访问。
            raise ValueError(f"参考音频路径不安全：{relative}")  # 不允许清单跳转到其他账户材料。
        copy_file_resumable(permanent.joinpath(*pure.parts), root.joinpath(*pure.parts))  # 保留原路径让现有评测直接读取。

def cache_fastvoicegrad(drive_root, project_root, include_parent=True):  # 用永久小源码及必要分卷快速建立可重建训练缓存。
    from tqdm.auto import tqdm  # 显示62卷的真实百分比进度。
    drive_root, root = Path(drive_root), Path(project_root)  # 固定永久总目录与本地项目。
    permanent = drive_root / "FsatVoiceGrad"  # 已恢复的模型、源码和实验合同永久项目。
    plans = _shard_plan(drive_root, root)  # 从小清单直接重建卷计划。
    model_paths = ["exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt", "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/generator_step_2500000.pt"]  # 两种任务共同需要教师和自训HiFi-GAN。
    parent_relative = "exp/diagnostics/student_development_20260906/acdd_mc1_longtrain_from3327_v1/latest_full.pt"  # Post两臂使用同一正式100pass完整父状态。
    if include_parent:  # Post和筛选任务需要完整父断点。
        model_paths.append(parent_relative)  # 从永久同一父断点建立两条公平分支。
    for relative in model_paths:  # 在缓存任何大量数据前核对主模型来源。
        if not (permanent / relative).is_file():  # 缺正式模型时不能启动近似实验。
            raise FileNotFoundError(f"缺少必要永久模型：{permanent / relative}")  # 明确告知需要恢复的准确文件。
    remaining_data = sum(identity["bytes"] for group, path, members, identity, done in plans if _load_json(done) != identity)  # 无压缩TAR字节数提供本轮数据空间上界。
    remaining_models = sum((permanent / relative).stat().st_size for relative in model_paths if not (root / relative).is_file())  # 统计尚未缓存的主要模型长度。
    checkpoint_bytes = (permanent / parent_relative).stat().st_size if (permanent / parent_relative).is_file() else 0  # 用现有完整训练态的实际大小预留两槽及上传暂存。
    required = remaining_data + remaining_models + 3 * checkpoint_bytes + 5 * 1024**3  # 根据真实来源大小另留环境、波形和断点暂存空间。
    free = shutil.disk_usage("/root").free  # 读取当前SSD真实可用容量。
    print(f"本地剩余 {free / 1024**3:.1f} GiB，本次必要缓存及暂存预留 {required / 1024**3:.1f} GiB")  # 显示当前实际空间要求。
    if free < required:  # 条件不足时在下载和解包前停止。
        raise RuntimeError("当前Colab本地磁盘不足，请更换更大运行时再运行同一个单元格。")  # 保留所有永久材料，避免下载半途耗尽磁盘。
    root.mkdir(parents=True, exist_ok=True)  # 通过容量检查后建立本地高速项目。
    for name in ("fastvoicegrad", "voicegrad", "hifigan", "preprocess", "evaluation", "experiments/cycle_stage", "external/legacy_compute_bnf"):  # 仅缓存实际模型、原训练、在线BNF和评价源代码。
        copy_tree(permanent / name, root / name, "缓存 " + name)  # 不复制旧推理目录或全部训练波形。
    for name in ("UTMOS-demo", "pretrained_spkrec_ecapa"):  # 正式开发和端点评估需要这两套已经下载的模型。
        copy_tree(permanent / name, root / name, "缓存评测模型 " + name)  # 只复制明确的评价模型目录。
    helper = "exp/diagnostics/student_performance_tools_20260906/run_acdd_development.py"  # 原FastVoiceGrad训练工具是正式参数来源。
    copy_file_resumable(permanent / helper, root / helper)  # 保留原训练裁剪、数据和初始化接口。
    development_relative = "exp/diagnostics/student_development_20260906"  # 固定52句开发划分所在目录。
    for name in ("development_contract.json", "development_manifest.json", "pairs.csv", "self_reconstruction.jsonl", "student_train_manifest.csv", "dev_excluded_manifest.csv"):  # 只复制固定划分材料，不复制该目录里旧训练结果。
        copy_file_resumable(permanent / development_relative / name, root / development_relative / name)  # 将严格开发排除协议原样缓存。
    for name in ("config.json", "final_checkpoint_attestation_comments_20260906.json"):  # 自训HiFi-GAN的权重必须配合原配置和证明来源。
        relative = "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/" + name  # 定位固定声码器配置材料。
        copy_file_resumable(permanent / relative, root / relative)  # 不用其他预训练声码器代替。
    for relative in model_paths:  # 主要模型按大文件断点复制。
        copy_file_resumable(permanent / relative, root / relative)  # 同一运行时重跑复用完整缓存或继续前缀。
    for plan in tqdm(plans, desc="恢复必要数据分卷", unit="卷", dynamic_ncols=True):  # 一卷完成就推进真实百分比。
        _restore_shard(plan, root)  # 不再重新压缩或做全48GB校验遍历。
    _reference_waves(root, permanent)  # 只补充开发与最终测试所用的小批参考音频。
    for name in ("requirements-evaluation-matpool.txt",):  # 将兼容旧评测环境的既有依赖协议原样缓存。
        copy_file_resumable(permanent / name, root / name)  # 给独立环境安装提供固定来源。
    atomic_json(root / ".report_cache_ready.json", {"permanent_project": str(permanent), "data_shards": len(plans), "created_unix": time.time()})  # 保存当前运行时准备成功的轻量收据。
    return root  # 返回训练器实际使用的SSD项目目录。

def runtime_preflight(drive_root):  # 在复制大文件前确认真实Colab GPU和永久挂载条件。
    if sys.platform != "linux" or not Path("/content").is_dir():  # 本地Mac只允许代码编辑及静态检查。
        raise RuntimeError("此启动器只能在Colab GPU运行时启动实验。")  # 防止本地误用CPU训练。
    if not Path(drive_root).is_dir() or not Path("/content/drive/MyDrive").is_dir():  # 当前账号必须真实挂载正确永久来源。
        raise RuntimeError("Google Drive尚未挂载或永久总目录不存在。")  # 明确要求挂载而不退回临时输出。
    command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]  # 直接读取当前分配GPU真实配置。
    result = subprocess.run(command, capture_output=True, text=True, check=True)  # 只查询硬件，不启动任何神经网络。
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]  # 读取可见GPU列表。
    if len(lines) != 1:  # 默认任务固定使用一个独立GPU运行时。
        raise RuntimeError(f"当前运行时应恰好有一张可见GPU，实际：{lines}")  # 不偷偷共享或并行占用多张GPU。
    name, total, free, driver = [part.strip() for part in lines[0].split(",")]  # 拆出实际型号、总显存、空闲显存和驱动。
    print(f"GPU：{name}；总显存 {float(total) / 1024:.1f} GiB；空闲显存 {float(free) / 1024:.1f} GiB；驱动 {driver}")  # 用户可以核对确实获得的GPU。
    print("每个启动文件在本运行时顺序执行任务。多个文件只有连接到独立GPU运行时才真正并行。")  # 说明浏览器多标签不等于多GPU。
    return {"name": name, "memory_total_mib": float(total), "memory_free_mib": float(free), "driver": driver}  # 保存实际环境供永久记录使用。

def cache_metric_tools(drive_root, project_root):  # VQ只缓存通用评分工具，不恢复FastVoiceGrad巨大训练特征。
    permanent, root = Path(drive_root) / "FsatVoiceGrad", Path(project_root)  # 固定永久来源和只用于通用评价的SSD目录。
    root.mkdir(parents=True, exist_ok=True)  # 创建评价工具缓存根目录。
    for relative in ("evaluation", "voicegrad", "UTMOS-demo", "pretrained_spkrec_ecapa"):  # 原评价器需要共享模块及两套已下载评分模型。
        copy_tree(permanent / relative, root / relative, "缓存通用评分工具 " + relative)  # 不复制教师、学生、mel、BNF或完整训练集。
    copy_file_resumable(permanent / "requirements-evaluation-matpool.txt", root / "requirements-evaluation-matpool.txt")  # 使用已有固定通用评分依赖协议。
    return root  # 返回VQ评分时引用的实际工具根目录。

def _compiler_overrides(configuration, gcc, gxx):  # 将独立Python默认缺失的clang替换为现有GCC，不改变原链接参数。
    replacements = {}  # 保存只传给安装子进程的环境变量，不修改系统全局配置。
    for key, executable, fallback in (("CC", gcc, "clang"), ("CXX", gxx, "clang++"), ("LDSHARED", gcc, "clang -shared"), ("BLDSHARED", gcc, "clang -shared")):  # 同时修正C、C++及共享库链接，避免编译成功后链接仍找clang。
        original = shlex.split(configuration.get(key) or fallback)  # 读取实际独立解释器参数并保留原旗标顺序。
        replacements[key] = shlex.join([str(executable)] + original[1:])  # 只换首个编译器程序，保留pthread、shared等原参数。
    return replacements  # 返回当前包安装使用的明确工具链。

def _run_environment_command(command, log_path, title, environment_overrides=None):  # 实时转发安装子进程输出，避免Colab只显示外层退出码。
    command = [str(value) for value in command]  # 把路径转换成参数字符串，不经过Shell拼接执行。
    log_path = Path(log_path)  # 将全部安装日志保存在明确的永久路径。
    log_path.parent.mkdir(parents=True, exist_ok=True)  # 只创建环境准备日志目录，不改训练数据或断点。
    if command[1:3] == ["-m", "pip"]:  # pip普通输出会省略底层wheel错误，因此额外保存它自己的完整调试日志。
        command[3:3] = ["--log", str(log_path.with_suffix(".pip.log"))]  # pip完整编译输出永久保存，不让错误再次只剩See above。
    header = f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {title}\n命令：{command!r}\n"  # 记录当前安装步骤和实际调用参数。
    print(f"环境准备：{title}\n完整安装日志：{log_path}", flush=True)  # 在每个耗时步骤开始前明确显示当前工作。
    recent = deque(maxlen=60)  # 只在内存保留最后六十行，不累积整套安装日志。
    child_environment = dict(os.environ, PYTHONUNBUFFERED="1", PIP_PROGRESS_BAR="off")  # 实时输出安装信息，并避免下载动画刷出大量重复日志。
    child_environment.update(environment_overrides or {})  # 在当前安装进程中使用明确GCC/G++，不更改训练代码或全局环境。
    with log_path.open("a", encoding="utf-8") as log:  # 追加新尝试，保留上次失败输出，不清空历史。
        log.write(header)  # 将真实步骤标题写入永久日志。
        if environment_overrides:  # 使用修正编译器时记录实际参数，方便核对没有换模型版本。
            log.write(f"编译器设置：{environment_overrides!r}\n")  # 明确保存编译和链接都已替换的工具链。
        log.flush()  # 在安装前保存标题，即使运行时随后中断也知道停在哪一步。
        try:  # 安装程序本身无法启动时也保存明确错误。
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=child_environment)  # 同时捕获标准输出和错误输出，修复Notebook漏显示pip报错的问题。
        except Exception as exc:  # 解释器缺失或进程创建失败时不只留下步骤标题。
            log.write(f"\n程序启动失败：{type(exc).__name__}: {exc}\n")  # 保存真实启动错误及类型。
            log.flush()  # 把失败详情写入永久日志。
            raise RuntimeError(f"无法启动环境准备步骤：{title}；完整日志：{log_path}；{exc}") from exc  # 同一异常里指出可读取的日志位置。
        try:  # 无论安装失败还是用户停止，都妥善关闭当前子进程。
            for line in process.stdout:  # 按真实输出逐行更新，而不是等所有安装完成才显示。
                print(line, end="", flush=True)  # 将pip完整错误直接显示在当前唯一单元格中。
                log.write(line)  # 保存同一行真实输出，方便断线后读取。
                log.flush()  # 及时提交小型日志，避免关闭页面后丢失报错细节。
                recent.append(line.rstrip())  # 保存有限错误上下文供异常消息显示。
            returncode = process.wait()  # 获取当前安装命令的真实退出状态。
        except BaseException:  # 用户中断或读取失败时不让隐藏的安装继续占用资源。
            process.terminate()  # 只停止本函数明确创建的安装子进程。
            try:  # 给子进程短时间正常退出。
                process.wait(timeout=5)  # 最多等待五秒，不阻塞用户长时间响应。
            except subprocess.TimeoutExpired:  # 当前安装仍不退出时停止这个精确子进程。
                process.kill()  # 不终止训练任务或其他运行时，只停止当前安装进程。
                process.wait()  # 回收已停止的子进程。
            log.write("\n本次环境准备被中断；缓存和训练断点未删除。\n")  # 永久标明此次没有完成安装。
            log.flush()  # 保存中断原因。
            raise  # 原样报告用户中断或真实读取异常。
        finally:  # 在成功和失败两种情况下都关闭输出管道。
            process.stdout.close()  # 释放当前命令的输出句柄。
        log.write(f"\n步骤退出码：{returncode}\n")  # 将实际完成状态写入永久日志。
        log.flush()  # 确保失败尾部已交给Drive文件系统。
    if returncode:  # 软件包没有安装成功时禁止继续训练。
        tail = "\n".join(recent)  # 汇总真正的pip错误，不只显示CalledProcessError外壳。
        details = str(log_path.with_suffix(".pip.log")) if command[1:3] == ["-m", "pip"] else str(log_path)  # wheel的完整内部错误以pip自身日志为准。
        raise RuntimeError(f"环境准备失败：{title}，退出码{ returncode }。\n完整内部错误日志：{details}\n最后60行输出：\n{tail}")  # 在同一个报错块里指出真正完整的编译日志。
    return subprocess.CompletedProcess(command, returncode)  # 保留标准命令完成结果供测试使用。

def ensure_environment(project_root, release_root, needs_metrics=True, needs_vq=False, environment_log_root=None):  # 使用Python3.10隔离旧语音依赖，不破坏Colab默认Python3.13。
    project_root, release_root = Path(project_root), Path(release_root)  # 固定高速项目与新源码来源。
    kind = "vq" if needs_vq else "fastvoicegrad"  # 两个论文实现使用各自明确记录的依赖环境。
    environment = Path("/root/voice_research_envs") / ("py310_" + kind)  # 当前运行时的独立兼容环境目录。
    python = environment / "bin/python"  # 此任务唯一训练解释器。
    marker = environment / "environment_ready.json"  # 同一运行时成功安装后可以快速复用。
    log_root = Path(environment_log_root) if environment_log_root is not None else release_root.parent / DEFAULT_RUN_NAME / "environment_logs"  # 正式入口传入自己的永久结果目录，兼容原直接调用。
    log_path = log_root / (socket.gethostname() + "_" + kind + ".log")  # 每个真实运行时的安装输出永久保存，三个任务不会覆盖彼此。
    compiler_environment = {}  # 独立Python就位后才根据实际sysconfig准备编译器参数。
    def run_install(command, title):  # 当前函数的安装命令统一使用可见且持久的日志通道。
        return _run_environment_command(command, log_path, title, compiler_environment)  # 保留原软件版本，只修正实际缺失编译器和错误日志。
    signature = {"python": "3.10", "torch": "2.6.0+cu124", "torchaudio": "2.6.0+cu124", "numpy": "1.26.4", "librosa": "0.9.2" if needs_vq else "0.11.0", "kind": kind, "metrics": bool(needs_metrics), "compiler_policy": "gcc_gxx_preserve_flags_v1", "schema": 6}  # 保留全部数值软件版本，登记本次仅构建工具链修复。
    if needs_vq:  # VQ就绪标记必须证明附加依赖的实际接口也已检查。
        signature["vq_probe_revision"] = 1  # 旧的通用导入检查不能冒充本次完整VQ检查。
    if python.is_file() and _load_json(marker) == signature:  # 已准备过同一环境时不再重复下载数GiB软件包。
        print(f"✓ 复用已安装的Python3.10环境：{environment}")  # 显示本次实际使用的解释器环境。
        return python  # 返回已准备好的环境解释器。
    if importlib.util.find_spec("uv") is None:  # 当前Colab没有环境管理工具时先安装轻量uv。
        run_install([sys.executable, "-m", "pip", "install", "uv==0.6.17"], "安装环境管理工具uv")  # 只在用户实际Colab运行时下载安装工具。
    uv = [sys.executable, "-m", "uv"]  # 使用当前Notebook解释器找到刚安装的uv。
    run_install(uv + ["python", "install", "3.10"], "准备独立Python3.10")  # 下载Python3.10供旧UTMOS和fairseq使用。
    if not python.is_file():  # 新运行时才创建兼容环境。
        run_install(uv + ["venv", "--python", "3.10", str(environment)], "创建独立环境")  # 保持默认Notebook解释器不变。
    gcc, gxx = shutil.which("gcc"), shutil.which("g++")  # 检查Colab真实安装的C与C++编译器，不继续调用不存在的clang。
    if gcc is None or gxx is None:  # 某个运行时缺编译器时只补正常的Ubuntu开发工具。
        run_install(["apt-get", "update", "-qq"], "更新官方系统包索引")  # 仅在编译器缺失时访问系统官方包索引。
        run_install(["apt-get", "install", "-y", "--no-install-recommends", "build-essential"], "补齐GCC与G++工具链")  # 不替换Python、CUDA、模型权重或实验设置。
        gcc, gxx = shutil.which("gcc"), shutil.which("g++")  # 补齐后再次确认实际可执行路径。
    if gcc is None or gxx is None:  # 补齐未成功时不冒称编译环境就绪。
        raise RuntimeError(f"未找到可用GCC/G++，请检查安装日志：{log_path}")  # 在开始构建fairseq前明确停止。
    query = "import json, sysconfig; print(json.dumps({key: sysconfig.get_config_var(key) for key in ('CC','CXX','LDSHARED','BLDSHARED')})) # 只查询当前独立Python的实际编译配置。"  # 查询不加载模型，也不修改sysconfig。
    configuration = json.loads(subprocess.run([str(python), "-c", query], capture_output=True, text=True, check=True).stdout)  # 读取实际解释器原有旗标，避免臆造链接参数。
    compiler_environment = _compiler_overrides(configuration, gcc, gxx)  # 同时修正编译与链接使用的程序。
    print(f"构建工具链已明确设置：{compiler_environment}", flush=True)  # 让用户看到当前不再调用缺失的clang。
    run_install([str(python), "-m", "ensurepip", "--upgrade"], "确认独立pip入口")  # 为独立解释器建立自身pip入口。
    run_install([str(python), "-m", "pip", "install", "pip==23.3.2", "setuptools==59.5.0", "wheel==0.41.3"], "安装固定版本构建工具")  # 兼容旧OmegaConf元数据和Lightning安装规定。
    run_install([str(python), "-m", "pip", "install", "torch==2.6.0", "torchaudio==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"], "确认固定PyTorch与CUDA音频环境")  # 同一运行时已安装的版本直接复用，不主动清理下载缓存。
    core = ["numpy==1.26.4", "Cython==0.29.36", "scipy==1.13.1", "soundfile==0.12.1", "librosa==0.9.2" if needs_vq else "librosa==0.11.0", "tqdm==4.67.1", "matplotlib==3.8.4", "PyYAML==6.0.2", "einops==0.3.0", "webrtcvad-wheels==2.0.14", "gdown==6.2.0", "scikit-learn==1.3.2", "pandas==2.2.3", "torch-complex==0.4.4", "humanfriendly==10.0", "typeguard==2.13.3", "resampy==0.4.3", "editdistance==0.6.2", "six==1.16.0"]  # 固定原特征提取协议的Conformer依赖，无需安装整个ESPnet。
    run_install([str(python), "-m", "pip", "install"] + core, "安装训练与音频基础依赖")  # 先准备扩展构建所需的NumPy和Cython。
    if needs_metrics:  # FastVoiceGrad需要在首批训练前准备全部正式评分依赖。
        requirements = project_root / "requirements-evaluation-matpool.txt"  # 使用原项目已审计的旧评估依赖清单。
        if not requirements.is_file():  # 缺既有正式依赖协议时不能临时装任意最新版本。
            raise FileNotFoundError(f"缺少固定评测依赖协议：{requirements}")  # 在训练前报缺项。
        run_install([str(python), "-m", "pip", "install", "--no-build-isolation", "-r", str(requirements)], "安装固定评测依赖")  # 保留原版本，直接显示具体失败包和编译或下载错误。
    if needs_vq:  # 官方VQ的MCD和离散特征处理使用独立附加包。
        vq_packages = ["tgt==1.5", "pymcd==0.2.1", "pyworld==0.3.4", "pysptk==1.0.1", "fastdtw==0.3.4", "umap-learn==0.5.5", "visdom==0.2.4"]  # 保留公开实现使用的主要VQ评价组件版本。
        run_install([str(python), "-m", "pip", "install", "--no-build-isolation"] + vq_packages, "安装作者VQ附加依赖")  # 软件包失败时停止，避免训练几小时后才发现缺评价工具。
        run_install([str(python), "-m", "pip", "install", "librosa==0.9.2"], "确认作者VQ音频API版本")  # 通用评分清单安装后恢复作者音频API，避免旧位置参数在0.11中报错。
    probe = "import torch, torchaudio, numpy, librosa, soundfile, matplotlib, yaml, einops, torch_complex, humanfriendly, typeguard, resampy # 检查真实训练、Conformer前端与说话人编码器依赖。\nassert torch.cuda.is_available(), '独立Python环境无法使用CUDA GPU' # 在首批训练前检查GPU可用。\nprint('兼容环境就绪:', torch.__version__, torchaudio.__version__, librosa.__version__, torch.cuda.get_device_name(0)) # 显示实际软件版本和GPU。\n"  # 定义只导入依赖和查询CUDA的无模型预检。
    if needs_metrics:  # 旧评测的依赖必须在首批训练前可以真实导入。
        probe += "import fairseq, pytorch_lightning, speechbrain, onnxruntime, torchmetrics # 检查正式评测依赖都能真实导入。\n"  # 只查模块兼容性，不评分音频。
    if needs_vq:  # VQ额外检查真实MCD、TextGrid、码本和身份编码器所用接口。
        probe += "from pymcd.mcd import Calculate_MCD # 同时检查pyworld、pysptk、fastdtw与SciPy的实际MCD依赖链。\nfrom tgt.io import read_textgrid # 检查作者TextGrid读取接口。\nfrom umap import UMAP # 检查已安装UMAP及其依赖。\nfrom visdom import Visdom # 只检查接口，不连接可视化服务。\nfrom sklearn.cluster import KMeans # 检查码本准备实际使用的接口。\nimport gdown, webrtcvad # 检查作者资产下载和说话人编码器的音频依赖。\nfrom torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score # 检查P.808评分实际入口，不执行评分。\nprint('VQ附加依赖实际接口检查通过') # 只在全部真实导入完成后报告。\n"  # 不导入带顶层计算的作者训练脚本，也不初始化任何神经网络。
    run_install([str(python), "-c", probe], "实际导入及CUDA预检")  # 提前发现解释器、CUDA或旧包兼容错误。
    atomic_json(marker, signature)  # 只有实际导入成功才标记兼容环境准备完成。
    print(f"✓ 使用独立兼容环境：{python}")  # 显示之后所有训练和评测真正使用的解释器。
    return python  # 返回唯一用于子进程训练的Python3.10入口。

def ensure_vq_reference(release_root, local_root):  # 取得固定公开提交的VQ源码并保留上游目录结构。
    release_root, local_root = Path(release_root), Path(local_root)  # 固定新发布目录和当前运行时SSD位置。
    reference = local_root / "reference/VQ-CycleDiffusion"  # VQworker从该位置加载未经改写的公开模型。
    commit = "096df7587a03927a1b02258b01ae3ade2d959ab5"  # 固定本次审阅及复现的作者源码版本。
    if (reference / ".git").is_dir():  # 同一运行时已经取得公开源码时核对提交后复用。
        result = subprocess.run(["git", "-C", str(reference), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)  # 只读检查实际版本。
        if result.stdout.strip() == commit:  # 只有同一正式版本可以直接继续。
            return reference  # 不重复下载或切换上游版本。
        raise RuntimeError("当前VQ源码缓存提交与正式公开版本不同。")  # 不覆盖可能属于用户的其他版本。
    reference.parent.mkdir(parents=True, exist_ok=True)  # 创建明确的上游源码专用目录。
    subprocess.run(["git", "clone", "--no-checkout", "https://github.com/hpjang/VQ-CycleDiffusion.git", str(reference)], check=True)  # 下载作者公开仓库供Colab运行。
    subprocess.run(["git", "-C", str(reference), "checkout", "--detach", commit], check=True)  # 明确固定到已经审计的提交。
    atomic_json(local_root / "vq_reference_receipt.json", {"url": "https://github.com/hpjang/VQ-CycleDiffusion", "commit": commit})  # 保存上游代码真实来源。
    return reference  # 返回VQworker实际需要的源码根目录。

def _selected_lambda(run_root, manual_lambda=None):  # 只读取明确选择的权重，不把初始候选冒称为最优值。
    if manual_lambda is not None:  # 用户可以在Notebook变量里明确固定正式权重。
        value = float(manual_lambda)  # 将用户提供值转换为普通数值。
        if value not in LAMBDA_CANDIDATES:  # 正式比较只能使用报告预先指定的候选。
            raise ValueError(f"正式Cycle权重应为 {LAMBDA_CANDIDATES} 之一。")  # 避免无记录地扩展实验网格。
        return value  # 使用用户明确指定值并由训练合同记录。
    path = Path(run_root) / "selected_lambda.json"  # 三个筛选任务完成后把最终开发集选择保存在这里。
    value = _load_json(path)  # 读取实验者实际选定结果。
    if not isinstance(value, dict) or value.get("lambda_cycle") not in LAMBDA_CANDIDATES:  # 尚未选择时不自动拿测试表现调权重。
        raise RuntimeError(f"请先完成三个Lambda筛选并在 {path} 写入明确选择，例如 {{\"lambda_cycle\": 0.0125, \"selection_basis\": \"development结果\"}}；也可在本Notebook明确设置 MANUAL_LAMBDA。")  # 说明准确可执行的下一步。
    if not value.get("selection_basis"):  # 最终选择应留下开发结果依据以便报告追溯。
        raise RuntimeError("selected_lambda.json还需要填写 selection_basis，注明开发集选择依据。")  # 不宣称未记录依据的最优权重。
    return float(value["lambda_cycle"])  # 返回已明确选定的正式权重。

def _run_child(command, cwd, lease_root=None):  # 同一GPU仅运行一个训练子进程并实时显示其每batch进度。
    print("启动命令：", " ".join(str(item) for item in command))  # 让用户核对实际启动了哪一个任务。
    if lease_root is None:  # 不需要外层租约的只读准备命令按原方式运行。
        subprocess.run([str(item) for item in command], cwd=str(cwd), check=True)  # 原样转发进度和异常，失败后停止下游任务。
        return  # 当前唯一子进程已经完整结束。
    with RunLease(lease_root) as lease:  # 保护当前实际run目录，而不锁住其他可并行的任务。
        environment = dict(os.environ)  # 保留当前Colab环境但只增添本任务所有权。
        environment["VOICE_RESEARCH_LEASE_TOKEN"] = lease.token  # 让共享断点工具在每次保存前再次核对所有权。
        environment["VOICE_RESEARCH_LEASE_ROOT"] = str(Path(lease_root).resolve())  # 明确VQ任务父目录的租约也保护四个说话人子目录。
        process = subprocess.Popen([str(item) for item in command], cwd=str(cwd), env=environment)  # 输出直接显示在同一个Notebook单元格。
        try:  # Notebook停止或租约异常时给训练器完整批次安全退出机会。
            while process.poll() is None:  # 一个GPU内只等待这个唯一训练子进程。
                lease.assert_owned()  # 每十秒检查后台心跳与永久所有权。
                time.sleep(10)  # 短等待不占用GPU，Notebook输出仍由子进程实时显示。
            if process.returncode != 0:  # 训练失败时禁止继续其他种子或下游实验。
                raise subprocess.CalledProcessError(process.returncode, command)  # 原错误明确显示在Notebook。
        except BaseException:  # 用户中断或存储错误都应先请求训练器安全结束。
            if process.poll() is None:  # 只向当前尚未退出的唯一训练器发送停止信号。
                process.terminate()  # 训练器在完整batch边界提交断点后结束。
                try:  # 给训练器留出向Drive上传完整状态的时间。
                    process.wait(timeout=120)  # 最多等两分钟完成当前断点，运行在用户Colab而非本地工具等待。
                except subprocess.TimeoutExpired:  # 超时仍不退出时保留此前完整断点。
                    process.kill()  # 只结束本次子进程，不删除任何永久数据。
                    process.wait()  # 回收进程状态，避免后台进程继续抢显存。
            raise  # 将真实中断原因传回用户。

def launch(task, drive_root=DEFAULT_DRIVE_ROOT, project_root=DEFAULT_LOCAL_PROJECT, release_root=None, run_root=None, seeds=MAIN_SEEDS, manual_lambda=None, vq_asset_root=None):  # 单个Notebook单元格调用的完整缓存、续训和训练入口。
    drive_root, project_root = Path(drive_root), Path(project_root)  # 固定永久来源与本地SSD位置。
    release_root = Path(release_root) if release_root is not None else drive_root / DEFAULT_RELEASE_NAME  # 使用用户已授权的新代码永久目录。
    run_root = Path(run_root) if run_root is not None else drive_root / DEFAULT_RUN_NAME  # 所有正式结果保存进新永久运行记录根目录。
    if task not in FVG_ARMS + ("lambda_screen", "lambda_00125", "lambda_005", "lambda_01", "all_serial") + VQ_TASKS:  # 只接受已经编写并登记的真实任务。
        raise ValueError(f"未知启动任务：{task}")  # 不允许模糊名称启动不明计算。
    hardware = runtime_preflight(drive_root)  # 先核对实际GPU和永久挂载。
    if not release_root.is_dir():  # 新代码必须真实上传成功后才运行。
        raise FileNotFoundError(f"找不到已上传的实验源码：{release_root}")  # 不从旧压缩包重新解包新代码。
    run_root.mkdir(parents=True, exist_ok=True)  # 创建全部训练成果的永久根目录。
    atomic_json(run_root / ("runtime_" + socket.gethostname() + ".json"), {"hardware": hardware, "task": task, "started_unix": time.time()})  # 保存实际启动机器配置。
    code_root = Path("/root/voice_research_release_20261008")  # 当前运行时的新源码高速副本。
    code_root.mkdir(parents=True, exist_ok=True)  # 创建唯一发布源码缓存目录。
    for name in ("suite_launcher.py", "loss_tracking.py", "fastvoicegrad_formal.py", "fastvoicegrad_evaluate.py", "vq_reproduction.py"):  # 原样缓存所有实际训练和日志入口。
        copy_file_resumable(release_root / name, code_root / name)  # 同一运行时重跑安全复用完整源码。
    is_vq = task in VQ_TASKS  # 判断当前任务是否属于独立的四说话人VQ复现。
    if is_vq:  # VQ任务只准备自己的公开资产，避免恢复无关的FastVoiceGrad巨大特征。
        cache_metric_tools(drive_root, project_root)  # 只恢复通用客观评分模型和小源码，VQ数据仍由作者永久资产提供。
        python = ensure_environment(project_root, release_root, needs_metrics=True, needs_vq=True, environment_log_root=run_root / "environment_logs")  # 训练前准备固定依赖，真实安装错误写到本次永久结果目录。
        assets = Path(vq_asset_root) if vq_asset_root is not None else drive_root / "VQCycleDiffusion_assets_20261008"  # 作者公开数据和权重的唯一永久存放位置。
        output = run_root / "vq" / (task + "_seed37")  # 每个VQ实验使用独立永久目录。
        command = [python, "-u", code_root / "vq_reproduction.py", "--project-root", project_root, "--output-dir", output, "--asset-root", assets, "--experiment", task, "--seed", "37", "--phase", "all", "--checkpoint-every", "25"]  # 由worker依次完成真实准备、训练、转换、评分和绘图。
        _run_child(command, code_root, output)  # 外层心跳和worker断点共同保护这一条VQ实验。
        return str(output)  # 返回该任务的实际永久结果位置。
    if task == "all_serial":  # 一个独立GPU可以按明确依赖顺序完成全部可执行任务。
        ordered = ("base_100", "base_post", "cd_512", "cd_1024", "cd_4096", "cd_8192", "initial_grid", "lambda_screen", "early_cycle_100", "cycle_post")  # 独立Base与VQ先运行，筛选后只有选定权重才能继续正式Cycle。
        for child_task in ordered:  # 同一GPU内依次运行，避免多个进程抢同一显存。
            launch(child_task, drive_root, project_root, release_root, run_root, seeds, manual_lambda, vq_asset_root)  # 重跑会恢复每个任务最近完整断点。
        return str(run_root)  # 全部已完成任务都保存在永久根目录。
    cycle_main = task in {"early_cycle_100", "cycle_post"}  # 正式Cycle两分支需要统一的选定权重。
    chosen = _selected_lambda(run_root, manual_lambda) if cycle_main else 0.0  # 在缓存巨大数据前检查正式Cycle权重已明确。
    cache_fastvoicegrad(drive_root, project_root, include_parent=task.endswith("post") or task.startswith("lambda"))  # FastVoiceGrad仅恢复报告所需的原模型和五类特征。
    python = ensure_environment(project_root, release_root, needs_metrics=True, needs_vq=False, environment_log_root=run_root / "environment_logs")  # 全部正式评分依赖在首批前导入，失败详情在同一单元格和Drive中保留。
    if task.startswith("lambda"):  # 权重筛选使用Cycle-Post的相同父状态和seed1234。
        candidates = LAMBDA_CANDIDATES if task == "lambda_screen" else (float({"lambda_00125": "0.0125", "lambda_005": "0.05", "lambda_01": "0.1"}[task]),)  # 支持三候选串行或独立GPU各跑一个候选。
        for value in candidates:  # 单运行时内顺序执行各候选，保证不同时占用同一GPU。
            output_root = run_root / "lambda_screen" / ("lambda_" + str(value).replace(".", "p"))  # 每个候选使用独立完整断点和loss目录。
            command = [python, "-u", code_root / "fastvoicegrad_formal.py", "--project-root", project_root, "--run-root", output_root, "--arm", "cycle_post", "--seed", "1234", "--lambda-cycle", str(value), "--screen-passes", "50", "--metrics-endpoint", "--run-approved"]  # 完整100→150筛选只做development，worker禁止计算test来选择权重。
            name = "cycle_post_seed_1234_lambda_" + format(value, ".8g").replace(".", "p") + "_screen_50pass"  # 与worker实际run_name函数保持完全一致。
            _run_child(command, code_root, output_root / name)  # worker按环境token借用外层唯一租约，避免重复加锁。
        atomic_json(run_root / "lambda_screen_ready.json", {"candidate_weights": list(candidates), "seed": 1234, "additional_passes": 50, "test_used_for_selection": False, "selection_required": True})  # 明确保存已完成的候选和仍需开发结果选择的状态。
        print(f"三个候选均保留独立development结果。请选择统一权重并保存 {run_root / 'selected_lambda.json'} 后启动正式Cycle两分支。")  # 不编造报告未约定的自动排名规则。
        return str(run_root / "lambda_screen")  # 返回权重筛选的永久结果位置。
    for seed in tuple(int(value) for value in seeds):  # 正式分支按报告三个种子依次执行。
        if seed not in MAIN_SEEDS:  # 未登记种子不能偷偷混入正式比较。
            raise ValueError(f"报告正式种子为 {MAIN_SEEDS}，收到 {seed}")  # 需要改实验时由用户明确修改合同。
        command = [python, "-u", code_root / "fastvoicegrad_formal.py", "--project-root", project_root, "--run-root", run_root / "fastvoicegrad", "--arm", task, "--seed", str(seed), "--lambda-cycle", str(chosen), "--metrics-endpoint", "--run-approved"]  # 正式任务保存全部loss并完成实际端点客观指标。
        name = task + "_seed_" + str(seed) + "_lambda_" + format(chosen, ".8g").replace(".", "p")  # 与worker实际run目录命名一致。
        _run_child(command, code_root, run_root / "fastvoicegrad" / name)  # worker借用外层租约，断点保存逐次核对真实所有权。
    print(f"✓ {task} 的已选种子任务完成，模型、逐批loss和曲线已保存：{run_root / 'fastvoicegrad'}")  # 完成信息只在全部真实子进程成功后出现。
    return str(run_root / "fastvoicegrad")  # 返回永久正式结果位置。

def main():  # 提供与单单元格相同的命令行入口。
    parser = argparse.ArgumentParser(description="Google Drive永久保存的报告实验启动器")  # 用清楚中文说明启动器用途。
    parser.add_argument("--task", required=True)  # 必须明确选择唯一训练任务。
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)  # 默认使用用户现有永久总目录。
    parser.add_argument("--project-root", type=Path, default=DEFAULT_LOCAL_PROJECT)  # 原训练工具固定使用该SSD缓存目录。
    parser.add_argument("--release-root", type=Path)  # 允许明确指定本次发布代码目录。
    parser.add_argument("--run-root", type=Path)  # 允许明确指定新实验永久结果根目录。
    parser.add_argument("--manual-lambda", type=float)  # 仅在用户明确固定正式Cycle权重时提供。
    parser.add_argument("--seeds", type=int, nargs="+", default=list(MAIN_SEEDS))  # 同一GPU按指定报告种子顺序执行。
    args = parser.parse_args()  # 读取用户实际调用时的参数。
    launch(args.task, args.drive_root, args.project_root, args.release_root, args.run_root, args.seeds, args.manual_lambda)  # 交给同一缓存、续训和训练实现。

if __name__ == "__main__":  # 导入共享工具不会意外启动训练。
    main()  # 只有用户运行正式入口才消耗GPU。
