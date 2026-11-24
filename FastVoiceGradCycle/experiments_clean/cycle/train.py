# 训练文件负责模型构建、循环损失、梯度更新及完整断点，实验参数统一读取config。
from __future__ import annotations  # 延后解释类型提示，导入本文件不会加载模型。
import argparse  # 读取本次明确执行的训练命令。
import hashlib  # 核对模型、源码、日志和断点的内容身份。
import importlib.util  # 加载baseline固定的音频特征提取工具。
import json  # 保存可阅读的训练配置和运行记录。
import math  # 检查损失和优化器游标是否有效。
import os  # 完整写入文件后再原子发布正式状态。
import random  # 按baseline规则恢复批次和裁剪顺序。
import shutil  # 检查磁盘空间并复制完整断点。
import signal  # 收到停止请求后等待当前完整更新再保存。
import socket  # 将当前运行主机写入独占运行租约。
import sys  # 设置baseline导入位置并限制训练主机。
import threading  # 在长时间保存期间持续刷新运行租约。
import time  # 记录实际训练时间和保存间隔。
import uuid  # 为日志段和临时文件分配唯一名称。
from dataclasses import replace  # 从baseline模型配置派生本次训练设置。
from pathlib import Path  # 处理数据、源码和断点的明确路径。
from types import SimpleNamespace  # 集中保存延迟加载的baseline组件。
import config as settings  # 所有实验设置都从同目录配置文件读取。
from plot import atomic_json, read_committed_records, _pointer_sequence, plot_losses  # 复用本组画图文件的日志读取与原子收据工具。

def load_runtime(project_root):  # 只在明确执行CUDA任务时导入baseline模型和音频库。
    root = Path(project_root).resolve()  # 使用真正存在的baseline项目路径。
    if sys.platform != "linux" or root != settings.PROJECT_ROOT.resolve() or not root.is_dir():  # 正式计算继续限定在原实验主机的项目目录。
        raise RuntimeError(f"请在Linux实验主机使用 {settings.PROJECT_ROOT}。")  # 本地导入和帮助命令不会启动模型。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 在首次CUDA计算前保留原确定性工作区设置。
    for path in (root, root / "exp/diagnostics/student_performance_tools_20260906"):  # 只加入baseline本身及原始模型构建工具。
        if str(path) not in sys.path:  # 避免重复插入同一个导入位置。
            sys.path.insert(0, str(path))  # 优先加载已核验的baseline实现。
    import torch  # 此处才加载真正的张量计算库。
    import torch.nn.functional as functional  # 复用张量补齐和插值函数。
    import torchaudio  # 对生成波形执行真实重采样。
    from torch.utils.data import default_collate  # 沿用baseline的批量数据组织方式。
    import fastvoicegrad.train as training  # 沿用baseline学生预测、判别器和蒸馏工具。
    import run_acdd_development as baseline  # 沿用学生12号模型、数据与裁剪构建流程。
    from voicegrad.feature_alignment import align_bnf_to_mel  # 保留原BNF与mel的长度对齐方式。
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3:  # 原累计方案仍要求至少10GiB显存的CUDA设备。
        raise RuntimeError("正式计算需要至少10GiB显存的CUDA GPU。")  # 条件不足时拒绝偷偷退回CPU。
    torch.backends.cudnn.benchmark = False  # 不依据首次计时挑选不同的卷积算法。
    torch.backends.cudnn.deterministic = True  # 保留原实验的确定性卷积选择。
    torch.use_deterministic_algorithms(True, warn_only=False)  # 除单独注明的反射填充反向外严格检查确定性。
    torch.set_num_threads(settings.TORCH_NUM_THREADS)  # 保留原CPU线程设置。
    return SimpleNamespace(  # 提供后续训练与推理共同使用的实际组件。
        torch=torch,  # 明确传入张量库版本。
        F=functional,  # 明确传入F参数。
        torchaudio=torchaudio,  # 明确传入torchaudio参数。
        default_collate=default_collate,  # 明确传入default_collate参数。
        training=training,  # 明确传入training参数。
        baseline=baseline,  # 明确传入baseline参数。
        align_bnf_to_mel=align_bnf_to_mel  # 明确传入align_bnf_to_mel参数。
    )  # 提供后续训练与推理共同使用的实际组件。

def arm_spec(arm):  # 返回两组共用的起点终点以及循环损失的启用边界。
    if arm not in ("base_post", "cycle_post"):  # 整理版只保留当前两组已完成的实验定义。
        raise ValueError(f"不支持的实验组：{arm}")  # 不保留未运行的历史Early和权重筛选入口。
    return {  # Cycle从baseline之后的第一批开始启用。
        "start": settings.START_STEP,  # 保存start字段。
        "end": settings.END_STEP,  # 保存end字段。
        "cycle_after": settings.START_STEP if arm == "cycle_post" else None  # 保存cycle_after字段。
    }  # Cycle从baseline之后的第一批开始启用。

def expected_cycle_step(arm, step):  # 根据完整训练步数计算真正完成的循环更新次数。
    after = arm_spec(arm)["cycle_after"]  # 取得本组循环损失的开启边界。
    return 0 if after is None else max(0, int(step) - after)  # Base始终为零，Cycle只统计开启后的更新。

def parse_args(argv=None):  # 读取路径与执行开关，实验数值设置集中由config控制。
    parser = argparse.ArgumentParser(description="从完整baseline继续本组Base或Cycle实验")  # 提供可单独查看的训练帮助。
    parser.add_argument("--project-root", type=Path, default=settings.PROJECT_ROOT)  # 显示并核对baseline原项目的位置。
    parser.add_argument("--run-root", type=Path, default=settings.RUN_ROOT)  # 整理版只能向独立的新运行根目录写入。
    parser.add_argument("--parent", type=Path, default=settings.POST_PARENT_PATH)  # 两组读取同一个完整baseline父断点。
    parser.add_argument("--bnf-checkpoint", type=Path)  # 允许明确指定同一固定SHA的Conformer权重位置。
    parser.add_argument("--metrics-endpoint", action="store_true")  # 显式要求按原计划生成开发及最终测试客观结果。
    parser.add_argument("--session-update-limit", type=int)  # 可在明确测试会话中限制新增完整更新数量。
    parser.add_argument("--run-approved", action="store_true")  # 单纯整理或导入代码不会授权占用GPU。
    args = parser.parse_args(argv)  # 解析用户实际提供的命令。
    args.arm, args.seed, args.lambda_cycle = settings.ARM, settings.SEED, settings.LAMBDA_CYCLE  # 分组与实验参数只读取本组设置。
    args.project_root, args.run_root = args.project_root.resolve(), args.run_root.resolve()  # 固定真实数据位置与新输出位置。
    if not args.run_approved:  # 准备文件和查看设置都不自动启动训练。
        raise RuntimeError("训练必须显式传入 --run-approved。")  # 缺少明确执行开关时提前停止。
    legacy_root = settings.LEGACY_RUN_ROOT.resolve()  # 明确定位历史结果根目录，不能靠修改新配置绕过保护。
    if args.run_root == legacy_root or legacy_root in args.run_root.parents:  # 整理版拒绝向历史目录或其子目录写入。
        raise ValueError("整理版不能写入历史结果目录及其子目录。")  # 防止误改设置后续写或覆盖原实验资料。
    if args.run_root != settings.RUN_ROOT.resolve():  # 防止整理版向旧代码生成的结果目录续写。
        raise ValueError("整理版只能写入config中独立的RUN_ROOT，不能续写历史实验目录。")  # 旧断点和旧日志始终保持原来源。
    if args.session_update_limit is not None and args.session_update_limit < 1:  # 会话上限必须至少容纳一次完整更新。
        raise ValueError("session-update-limit必须是正整数。")  # 零更新不能当作训练验证。
    if (  # 继续核对当前条件，保持原判断顺序。
        not math.isfinite(args.lambda_cycle)  # 两组的标签与真实损失开关必须一致。
        or args.lambda_cycle < 0  # 继续核对当前条件，保持原判断顺序。
        or (args.arm == "cycle_post" and args.lambda_cycle <= 0)  # 继续核对当前条件，保持原判断顺序。
        or (args.arm == "base_post" and args.lambda_cycle != 0)  # 继续核对当前条件，保持原判断顺序。
    ):  # 两组的标签与真实损失开关必须一致。
        raise ValueError("Base的Cycle权重必须为零，Cycle的权重必须为有限正值。")  # 配置写错时不创建实验。
    return args  # 返回已核对的路径、分组和执行参数。


# 以下函数保留baseline数据核验、优化器和随机状态的恢复规则。

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

class GlobalBatchPlan:  # 按绝对步数重建学生12号每一轮的固定随机打乱顺序。
    def __init__(self, indices: list, seed: int = 1234) -> None:  # 保存39335条保留索引和原始打乱种子。
        self.indices = list(indices)  # 复制索引，避免调用方后来修改原列表。
        self.seed = int(seed)  # 规范化原学生12号批次种子。
        self.batches_per_pass = len(self.indices) // settings.BATCH_SIZE  # 用整除丢弃每轮最后7条不足批样本。
        self.pass_index = None  # 新进程尚未重建任何轮次顺序。
        self.order = None  # 当前轮次的打乱索引将在第一次请求时生成。
        if self.batches_per_pass != settings.UPDATES_PER_PASS:  # 39335条保留数据必须恰好形成1229个完整批。
            raise RuntimeError(f"学生12号批次数变化：{self.batches_per_pass}")  # 数据人口不一致时停止。

    def for_step(self, global_step: int) -> list:  # 返回指定一基绝对更新步对应的32条数据索引。
        if type(global_step) is not int or global_step < 1:  # 全局更新步从1开始且必须是普通整数。
            raise ValueError("global_step 必须是一基正整数。")  # 拒绝0、负数或浮点游标。
        current_pass = (global_step - 1) // self.batches_per_pass  # 计算当前步位于第几个零基 pass。
        if self.pass_index != current_pass:  # 跨 pass 或首次调用时需要重建顺序。
            self.order = list(self.indices)  # 每轮都从同一原始索引顺序开始。
            random.Random(self.seed + current_pass).shuffle(self.order)  # 用原规则独立打乱当前 pass。
            self.pass_index = current_pass  # 缓存当前 pass，避免每个 batch 重复打乱。
        offset = ((global_step - 1) % self.batches_per_pass) * settings.BATCH_SIZE  # 计算当前 batch 在打乱列表中的起点。
        return self.order[offset : offset + settings.BATCH_SIZE]  # 返回严格32条且不含尾部余数的索引。

def load_training_indices() -> list:  # 读取学生12号固定52句排除协议中的训练索引。
    if not settings.DEVELOPMENT.is_file():  # 恢复目录必须包含原开发协议。
        raise FileNotFoundError(f"缺少学生12号开发排除协议：{settings.DEVELOPMENT}")  # 缺少协议时不能退回普通全数据训练。
    if sha256_file(settings.DEVELOPMENT) != settings.DEVELOPMENT_SHA256:  # 固定协议必须逐字节等于学生12号训练时归档版本。
        raise RuntimeError("学生12号固定52句排除协议的 SHA256 不一致。")  # 不允许换另一组数量相同的52句。
    if (  # 继续核对当前条件，保持原判断顺序。
        not settings.DEVELOPMENT_MANIFEST.is_file()  # 同时核对完整发布收据的固定身份。
        or sha256_file(settings.DEVELOPMENT_MANIFEST) != settings.DEVELOPMENT_MANIFEST_SHA256  # 继续核对当前条件，保持原判断顺序。
    ):  # 同时核对完整发布收据的固定身份。
        raise RuntimeError("学生12号开发划分发布收据缺失或 SHA256 不一致。")  # 防止同步替换协议和自报哈希。
    receipt = read_json(settings.DEVELOPMENT_MANIFEST)  # 读取已经固定身份的开发划分发布收据。
    expected_artifacts = {  # 明确传入expected_artifacts参数。
        "development_contract.json",  # 固定构成该划分的五份发布文件。
        "pairs.csv",  # 固定构成该划分的五份发布文件。
        "self_reconstruction.jsonl",  # 固定构成该划分的五份发布文件。
        "student_train_manifest.csv",  # 固定构成该划分的五份发布文件。
        "dev_excluded_manifest.csv"  # 固定构成该划分的五份发布文件。
    }  # 固定构成该划分的五份发布文件。
    if receipt.get("status") != "complete" or set(receipt.get("artifacts", {})) != expected_artifacts:  # 收据必须声明完成且文件集合不多不少。
        raise RuntimeError("学生12号开发划分发布收据不完整。")  # 不从缺失配套证据的协议继续训练。
    for name, expected_sha in receipt["artifacts"].items():  # 逐份核对固定收据登记的配套文件。
        artifact = settings.DEVELOPMENT.parent / name  # 将收据内安全的固定文件名定位到同一发布目录。
        if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != expected_sha:  # 文件必须是普通文件且内容身份一致。
            raise RuntimeError(f"学生12号开发划分配套文件损坏：{name}")  # 报告具体损坏文件并停止。
    contract = read_json(settings.DEVELOPMENT)  # 读取已经归档的固定开发划分。
    frozen_sources = contract.get("source_sha256")  # 读取固定协议登记的原训练清单和归一化统计身份。
    if not isinstance(frozen_sources, dict):  # 缺少来源哈希时无法证明数据协议仍与学生12号相同。
        raise RuntimeError("学生12号固定协议缺少 source_sha256。")  # 不根据当前文件自行建立新的有利身份。
    for source in (  # 按原顺序遍历需要处理的项目。
        settings.PROJECT_ROOT / "data/metadata/train_manifest.csv",  # 核对训练索引语义与mel归一化直接依赖的四份资产。
        settings.PROJECT_ROOT / "data/metadata/test_manifest.csv",  # 核对训练索引语义与mel归一化直接依赖的四份资产。
        settings.PROJECT_ROOT / "data/stats/mel_mean.npy",  # 核对训练索引语义与mel归一化直接依赖的四份资产。
        settings.PROJECT_ROOT / "data/stats/mel_std.npy"  # 核对训练索引语义与mel归一化直接依赖的四份资产。
    ):  # 核对训练索引语义与mel归一化直接依赖的四份资产。
        archived_source = Path("/root/FsatVoiceGrad") / source.relative_to(settings.PROJECT_ROOT)  # 用旧发布根拼出合同中保留的历史键，不改变文件内容身份。
        expected_source_sha = frozen_sources.get(str(archived_source))  # 从固定合同的历史绝对路径键读取该资产发布时哈希。
        if (  # 继续核对当前条件，保持原判断顺序。
            not source.is_file()  # 路径、普通文件属性和内容都必须匹配。
            or source.is_symlink()  # 继续核对当前条件，保持原判断顺序。
            or expected_source_sha is None  # 继续核对当前条件，保持原判断顺序。
            or sha256_file(source) != expected_source_sha  # 继续核对当前条件，保持原判断顺序。
        ):  # 路径、普通文件属性和内容都必须匹配。
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

def runtime_identity(device: torch.device) -> dict:  # 记录并锁定配对实验真正使用的软件与GPU数学环境。
    import torch  # 只有真正核验模型或恢复断点时才加载张量库。
    properties = torch.cuda.get_device_properties(device)  # 读取当前可见GPU的型号、显存和计算能力。
    return {  # 用纯JSON字段描述会影响数值轨迹的运行条件。
        "python": sys.version.split()[0],  # 保存Python版本。
        "torch": torch.__version__,  # 保存张量库版本。
        "cuda": torch.version.cuda,  # 保存CUDA版本。
        "cudnn": torch.backends.cudnn.version(),  # 保存卷积后端版本。
        "gpu_name": properties.name,  # 保存gpu_name字段。
        "gpu_total_memory": properties.total_memory,  # 保存gpu_total_memory字段。
        "compute_capability": list(torch.cuda.get_device_capability(device)),  # 保存compute_capability字段。
        "torch_num_threads": torch.get_num_threads(),  # 保存torch_num_threads字段。
        "torch_num_interop_threads": torch.get_num_interop_threads(),  # 保存torch_num_interop_threads字段。
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),  # 保存cuda_matmul_allow_tf32字段。
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),  # 保存cudnn_allow_tf32字段。
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),  # 保存cudnn_benchmark字段。
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),  # 保存cudnn_deterministic字段。
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())  # 保存deterministic_algorithms字段。
    }  # 用纯JSON字段描述会影响数值轨迹的运行条件。

def optimizer_uniform_step(optimizer: torch.optim.Optimizer) -> int:  # 读取一个 Adam 对全部参数一致的真实更新游标。
    import torch  # 只有真正核验模型或恢复断点时才加载张量库。
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
    import torch  # 只有真正核验模型或恢复断点时才加载张量库。
    if len(optimizer.param_groups) != 1:  # 学生12号的学生和判别器各只有一个参数组。
        raise RuntimeError(f"{role} Adam 参数组数量变化。")  # 多组学习率会改变原训练解释。
    group = optimizer.param_groups[0]  # 读取唯一参数组的实际运行设置。
    expected = {"lr": expected_lr, "betas": settings.ADAM_BETAS, "eps": settings.ADAM_EPS, "weight_decay": 0, "amsgrad": False}  # 锁定学生12号真实优化器设置。
    for key, value in expected.items():  # 逐项核对学习率、动量和数值稳定设置。
        if group.get(key) != value:  # 任何一项不同都表示分叉后优化条件改变。
            raise RuntimeError(f"{role} Adam 的 {key}={group.get(key)!r}，期望 {value!r}。")  # 报告具体差异后停止。

def validate_optimizer_state(  # 深入核对恢复后的 Adam 所有权、游标和动量张量。
    optimizer: torch.optim.Optimizer,  # 深入核对恢复后的 Adam 所有权、游标和动量张量。
    model: torch.nn.Module,  # 深入核对恢复后的 Adam 所有权、游标和动量张量。
    expected_lr: float,  # 深入核对恢复后的 Adam 所有权、游标和动量张量。
    expected_step: int,  # 深入核对恢复后的 Adam 所有权、游标和动量张量。
    role: str  # 深入核对恢复后的 Adam 所有权、游标和动量张量。
) -> None:  # 深入核对恢复后的 Adam 所有权、游标和动量张量。
    import torch  # 只有真正核验模型或恢复断点时才加载张量库。
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
            if (  # 继续核对当前条件，保持原判断顺序。
                not isinstance(moment, torch.Tensor)  # 动量的形状、精度和设备必须与参数完全对应。
                or moment.shape != parameter.shape  # 继续核对当前条件，保持原判断顺序。
                or moment.dtype != parameter.dtype  # 继续核对当前条件，保持原判断顺序。
                or moment.device != parameter.device  # 继续核对当前条件，保持原判断顺序。
            ):  # 动量的形状、精度和设备必须与参数完全对应。
                raise RuntimeError(f"{role} Adam 的 {key} 形状、精度或设备不符。")  # 拒绝PyTorch可能广播或隐式转换的异常状态。
            if not bool(torch.isfinite(moment).all()):  # 任何NaN或无穷都会污染后续全部更新。
                raise RuntimeError(f"{role} Adam 的 {key} 含非有限数。")  # 在第一批训练前失败关闭。
        if not bool((state["exp_avg_sq"] >= 0).all()):  # Adam二阶矩按定义不能出现负值。
            raise RuntimeError(f"{role} Adam 的 exp_avg_sq 含负值。")  # 拒绝损坏或格式误读的状态。

def validate_model_state(model: torch.nn.Module, role: str) -> None:  # 核对模型全部参数和缓冲区仍为有限数。
    import torch  # 只有真正核验模型或恢复断点时才加载张量库。
    for name, value in model.state_dict().items():  # 逐项检查权重、偏置和归一化缓冲区。
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):  # 正式模型状态必须是有限张量。
            raise RuntimeError(f"{role} 模型状态含非有限数或非法对象：{name}")  # 在恢复或保存边界阻止坏状态继续传播。

def rng_state_equal(expected: object, actual: object) -> bool:  # 精确比较Python、NumPy、CPU与CUDA四类随机状态。
    import torch  # 只有真正核验模型或恢复断点时才加载张量库。
    import numpy as np  # 对随机状态中的数组执行精确比较。
    if (  # 继续核对当前条件，保持原判断顺序。
        not isinstance(expected, dict)  # 两边必须具有完全相同的固定键集合。
        or not isinstance(actual, dict)  # 继续核对当前条件，保持原判断顺序。
        or set(expected) != {"python", "numpy", "torch", "cuda_all"}  # 继续核对当前条件，保持原判断顺序。
        or set(actual) != set(expected)  # 继续核对当前条件，保持原判断顺序。
    ):  # 两边必须具有完全相同的固定键集合。
        return False  # 格式缺失或多出未知字段时直接判为不一致。
    if expected["python"] != actual["python"]:  # Python random 状态由纯整数元组构成，可直接精确比较。
        return False  # Python随机流位置不同即恢复失败。
    expected_numpy, actual_numpy = expected["numpy"], actual["numpy"]  # 读取NumPy的算法名、状态数组和游标元组。
    if (  # 继续核对当前条件，保持原判断顺序。
        len(expected_numpy) != len(actual_numpy)  # NumPy状态数组和其余标量都必须逐项一致。
        or expected_numpy[0] != actual_numpy[0]  # 继续核对当前条件，保持原判断顺序。
        or not np.array_equal(expected_numpy[1], actual_numpy[1])  # 继续核对当前条件，保持原判断顺序。
        or expected_numpy[2:] != actual_numpy[2:]  # 继续核对当前条件，保持原判断顺序。
    ):  # NumPy状态数组和其余标量都必须逐项一致。
        return False  # NumPy随机流位置不同即恢复失败。
    if not torch.equal(expected["torch"].cpu(), actual["torch"].cpu()):  # CPU生成器状态按字节张量精确比较。
        return False  # PyTorch CPU随机流不同即恢复失败。
    if len(expected["cuda_all"]) != len(actual["cuda_all"]):  # 可见CUDA设备数量必须与保存状态列表一致。
        return False  # 设备状态数量不同无法声称完整恢复。
    return all(torch.equal(left.cpu(), right.cpu()) for left, right in zip(expected["cuda_all"], actual["cuda_all"]))  # 逐张GPU精确比较CUDA生成器状态。

def restore_stage_rng_state(payload: dict, device: torch.device, completed_step: int) -> dict:  # 对阶段断点严格恢复全部随机状态，并只为旧学生父断点兼容CUDA状态长度变化。
    import torch  # 只有真正核验模型或恢复断点时才加载张量库。
    import numpy as np  # 对随机状态中的数组执行精确比较。
    import fastvoicegrad.train as training  # 使用已加载的baseline随机状态工具。
    saved_state = payload["rng_state"]  # 读取父断点或本阶段断点保存的四类随机数状态。
    checkpoint_kind = payload.get("kind")  # 依据断点类型区分必须精确恢复的本阶段状态与旧版学生父状态。
    required_keys = {"python", "numpy", "torch", "cuda_all"}  # 固定任何可接受随机状态必须具有的完整键集合。
    if not isinstance(saved_state, dict) or set(saved_state) != required_keys:  # 旧父断点也不能缺少任一随机状态类别。
        raise RuntimeError(f"完整随机状态字段不符：期望{sorted(required_keys)}。")  # 不允许用兼容逻辑掩盖残缺断点。
    if checkpoint_kind != settings.PARENT_KIND:  # 本阶段自己保存的断点必须在合同锁定的同一后端逐字节精确恢复。
        training.restore_rng_state(saved_state, device)  # 恢复Python、NumPy、PyTorch CPU和当前CUDA生成器状态。
        if not rng_state_equal(saved_state, training.capture_rng_state()):  # 恢复后立即重新捕获并比较四类状态。
            raise RuntimeError("完整随机状态恢复后回读不一致。")  # 本阶段续训不开放任何近似或重建路径。
        exact_sizes = [int(state.numel()) for state in saved_state["cuda_all"]]  # 记录本阶段断点逐字节恢复的CUDA状态长度供续训审计。
        return {  # 记录本阶段断点已经完成严格全量恢复。
            "mode": "exact_checkpoint_rng_state",  # 保存mode字段。
            "checkpoint_kind": checkpoint_kind,  # 保存checkpoint_kind字段。
            "saved_cuda_device_count": len(exact_sizes),  # 保存saved_cuda_device_count字段。
            "runtime_cuda_device_count": len(exact_sizes),  # 保存runtime_cuda_device_count字段。
            "saved_cuda_state_sizes": exact_sizes,  # 保存saved_cuda_state_sizes字段。
            "runtime_cuda_state_sizes": exact_sizes  # 保存runtime_cuda_state_sizes字段。
        }  # 记录本阶段断点已经完成严格全量恢复。
    random.setstate(saved_state["python"])  # 对旧学生父断点仍逐项精确恢复Python随机状态。
    np.random.set_state(saved_state["numpy"])  # 对旧学生父断点仍逐项精确恢复NumPy随机状态。
    torch.set_rng_state(saved_state["torch"])  # 对旧学生父断点仍逐项精确恢复PyTorch CPU随机状态。
    restored_non_cuda = training.capture_rng_state()  # 回读当前状态以核对前三类旧状态没有因兼容路径被重置。
    non_cuda_equal = (  # 逐项核对旧父状态中可以精确恢复的随机序列。
        saved_state["python"] == restored_non_cuda["python"]  # Python随机状态必须完全相同。
        and len(saved_state["numpy"]) == len(restored_non_cuda["numpy"])  # NumPy状态字段数量必须相同。
        and saved_state["numpy"][0] == restored_non_cuda["numpy"][0]  # NumPy生成器类型必须相同。
        and np.array_equal(saved_state["numpy"][1], restored_non_cuda["numpy"][1])  # NumPy内部状态数组必须逐项相同。
        and saved_state["numpy"][2:] == restored_non_cuda["numpy"][2:]  # NumPy其余游标和缓存必须相同。
        and torch.equal(saved_state["torch"].cpu(), restored_non_cuda["torch"].cpu())  # CPU张量生成器状态必须相同。
    )  # 完成非CUDA随机状态的精确回读检查。
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
    exact_cuda_compatible = len(saved_cuda_states) == len(runtime_cuda_states) and all(  # 明确传入exact_cuda_compatible参数。
        saved.numel() == current.numel() for saved,  # 已确认类型与维度合法后，只有设备数和长度完全相同时才逐字节恢复。
        current in zip(saved_cuda_states, runtime_cuda_states)  # 已确认类型与维度合法后，只有设备数和长度完全相同时才逐字节恢复。
    )  # 已确认类型与维度合法后，只有设备数和长度完全相同时才逐字节恢复。
    if exact_cuda_compatible:  # 旧父断点碰巧来自兼容CUDA生成器格式时保留原始状态。
        torch.cuda.set_rng_state_all(saved_cuda_states)  # 在已经确认尺寸兼容后恢复旧CUDA生成器字节。
        if not rng_state_equal(saved_state, training.capture_rng_state()):  # 四类状态都必须与断点逐项完全一致。
            raise RuntimeError("旧父断点的兼容CUDA随机状态恢复后回读不一致。")  # 接口无报错但内容不同仍然停止。
        return {  # 永久记录本次无需重建旧CUDA状态。
            "mode": "legacy_parent_exact_all_rng_state",  # 保存mode字段。
            "checkpoint_kind": checkpoint_kind,  # 保存checkpoint_kind字段。
            "saved_cuda_device_count": len(saved_sizes),  # 保存saved_cuda_device_count字段。
            "runtime_cuda_device_count": len(runtime_sizes),  # 保存runtime_cuda_device_count字段。
            "saved_cuda_state_sizes": saved_sizes,  # 保存saved_cuda_state_sizes字段。
            "runtime_cuda_state_sizes": runtime_sizes  # 保存runtime_cuda_state_sizes字段。
        }  # 永久记录本次无需重建旧CUDA状态。
    next_step = int(completed_step) + 1  # 计算恢复后即将执行的第一个绝对更新步。
    replacement_seed = settings.PER_UPDATE_SEED_OFFSET + next_step  # 使用与正式循环第一批完全相同的绝对步种子公式。
    torch.cuda.manual_seed_all(replacement_seed)  # 只重建无法载入的CUDA生成器，不改变已经精确恢复的Python、NumPy和CPU状态。
    rebuilt_once = [state.clone() for state in torch.cuda.get_rng_state_all()]  # 保存第一次确定性重建后的当前格式字节。
    torch.cuda.manual_seed_all(replacement_seed)  # 用同一整数再重建一次以验证当前后端结果可重复。
    rebuilt_twice = torch.cuda.get_rng_state_all()  # 回读第二次重建结果用于确定性核验。
    if len(rebuilt_once) != len(rebuilt_twice) or not all(  # 继续核对当前条件，保持原判断顺序。
        torch.equal(left, right) for left,  # 同种子两次结果必须逐字节相同。
        right in zip(rebuilt_once, rebuilt_twice)  # 同种子两次结果必须逐字节相同。
    ):  # 同种子两次结果必须逐字节相同。
        raise RuntimeError("旧父断点CUDA随机状态的确定性重建失败。")  # 当前后端不能稳定重建时拒绝进入训练。
    return {  # 记录旧格式与当前格式长度、兼容原因以及唯一替代种子。
        "mode": "legacy_parent_non_cuda_exact_cuda_reseeded",  # 保存mode字段。
        "checkpoint_kind": checkpoint_kind,  # 保存checkpoint_kind字段。
        "fallback_reason": "saved_and_runtime_cuda_state_layout_mismatch",  # 保存fallback_reason字段。
        "saved_cuda_device_count": len(saved_sizes),  # 保存saved_cuda_device_count字段。
        "runtime_cuda_device_count": len(runtime_sizes),  # 保存runtime_cuda_device_count字段。
        "saved_cuda_state_sizes": saved_sizes,  # 保存saved_cuda_state_sizes字段。
        "runtime_cuda_state_sizes": runtime_sizes,  # 保存runtime_cuda_state_sizes字段。
        "cuda_replacement_seed": replacement_seed  # 保存cuda_replacement_seed字段。
    }  # 记录旧格式与当前格式长度、兼容原因以及唯一替代种子。

def validate_core_configuration(configuration: object) -> None:  # 核对旧断点中与学生12号数学直接相关的配置子集。
    if not isinstance(configuration, dict):  # 完整断点必须带有可读配置字典。
        raise RuntimeError("完整断点缺少配置字典。")  # 没有配置无法证明来源一致。
    expected = {  # 明确传入expected参数。
        "score_target": "clean_x0",  # 保存score_target字段。
        "score_weight": "sqrt_alpha_bar",  # 保存score_weight字段。
        "batch_size": settings.BATCH_SIZE,  # 保存batch_size字段。
        "student_start_step": settings.STUDENT_START_STEP,  # 保存student_start_step字段。
        "lambda_fm": settings.LAMBDA_FM,  # 保存lambda_fm字段。
        "lambda_dist": settings.LAMBDA_DIST,  # 保存lambda_dist字段。
        "train_frames": settings.TRAIN_FRAMES,  # 保存train_frames字段。
        "bnf_alignment": settings.BNF_ALIGNMENT  # 保存bnf_alignment字段。
    }  # 固定学生12号核心目标和数据设置。
    for key, value in expected.items():  # 逐项核对旧配置中必须存在的核心字段。
        if configuration.get(key) != value:  # 缺失或取值变化都不允许静默继续。
            raise RuntimeError(f"父断点核心配置不符：{key}={configuration.get(key)!r}，期望 {value!r}。")  # 报告确切字段。

def load_checkpoint(path: Path) -> dict:  # 用兼容新旧 PyTorch 的方式把完整训练状态加载到 CPU。
    import torch  # 只有真正核验模型或恢复断点时才加载张量库。
    try:  # 新版 PyTorch 支持显式关闭 weights_only 限制。
        value = torch.load(path, map_location="cpu", weights_only=False)  # 旧断点包含优化器和 RNG 字典，必须完整读取。
    except TypeError:  # PyTorch 1.13 尚没有 weights_only 参数。
        value = torch.load(path, map_location="cpu")  # 在旧环境中使用等价的完整加载路径。
    if not isinstance(value, dict):  # 上层验证依赖明确字段映射。
        raise RuntimeError(f"断点顶层不是字典：{path}")  # 裸模型权重不能用于分叉。
    return value  # 返回仍在 CPU 的完整状态，稍后严格加载到模型。

def _build_baseline_components(runtime, device):  # 按学生12号原顺序构建教师、学生、声码器和判别器。
    baseline, torch = runtime.baseline, runtime.torch  # 使用已经延迟加载的baseline原组件。
    cycle_enabled = False  # 模型容器先按原baseline构建，随后统一覆盖本组Cycle设置。
    config = replace(  # 明确传入模型实际配置。
        baseline.PAPER_STUDENT_CONFIG,  # 阶段内第一步即按计划开关 Cycle，其他核心参数不变。
        num_workers=0,  # 明确传入num_workers参数。
        cycle_enabled=cycle_enabled,  # 明确传入cycle_enabled参数。
        cycle_start_epoch=1,  # 明确传入cycle_start_epoch参数。
        epochs=50  # 明确传入epochs参数。
    )  # 阶段内第一步即按计划开关 Cycle，其他核心参数不变。
    if config.cycle_enabled is not cycle_enabled or config.cycle_start_epoch != 1 or config.lambda_cycle != 1.0:  # 构建后立即核对Cycle开关、门控轮次和预注册权重。
        raise RuntimeError("阶段配置没有正确锁定 Cycle 开关、起始轮次或权重。")  # 防止计划标签和真实训练目标不一致。
    teacher, student, _ = baseline.load_teacher_and_student(baseline.TEACHER, config, device)  # 用500轮教师结构建立并初始化学生。
    vocoder, vocoder_config, _ = baseline.load_frozen_vocoder(  # 严格加载自训250万步 HiFi-GAN。
        settings.PROJECT_ROOT,  # 严格加载自训250万步 HiFi-GAN。
        baseline.VOCODER_DIR / "generator_step_2500000.pt",  # 严格加载自训250万步 HiFi-GAN。
        baseline.VOCODER_DIR / "config.json",  # 严格加载自训250万步 HiFi-GAN。
        baseline.VOCODER_DIR / "final_checkpoint_attestation_comments_20260906.json",  # 严格加载自训250万步 HiFi-GAN。
        config,  # 严格加载自训250万步 HiFi-GAN。
        device  # 严格加载自训250万步 HiFi-GAN。
    )  # 严格加载自训250万步 HiFi-GAN。
    discriminator = baseline.build_paper_discriminator(settings.PROJECT_ROOT, vocoder_config, config, device)  # 建立与父断点相同的 MPD+MRD 判别器。
    diffusion = baseline.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)  # 建立相同1000步余弦扩散日程。
    mel_mean, mel_std = baseline.load_mel_statistics(settings.PROJECT_ROOT / "data", device)  # 读取原训练集固定 mel 均值和标准差。
    student_optimizer = torch.optim.Adam(  # 明确传入student_optimizer参数。
        student.parameters(),  # 用真实5e-5学生学习率建立待恢复 Adam 容器。
        lr=settings.STUDENT_LR,  # 明确传入lr参数。
        betas=settings.ADAM_BETAS,  # 明确传入betas参数。
        eps=settings.ADAM_EPS  # 明确传入eps参数。
    )  # 用真实5e-5学生学习率建立待恢复 Adam 容器。
    discriminator_optimizer = torch.optim.Adam(  # 明确传入discriminator_optimizer参数。
        discriminator.parameters(),  # 用真实2e-4判别器学习率建立待恢复 Adam 容器。
        lr=settings.DISCRIMINATOR_LR,  # 明确传入lr参数。
        betas=settings.ADAM_BETAS,  # 明确传入betas参数。
        eps=settings.ADAM_EPS  # 明确传入eps参数。
    )  # 用真实2e-4判别器学习率建立待恢复 Adam 容器。
    return config, teacher, student, vocoder, discriminator, diffusion, mel_mean, mel_std, student_optimizer, discriminator_optimizer  # 返回训练一批所需全部对象。


# 以下函数将日志、独占租约和完整双槽断点集中放在训练文件中。

def _load_json(path, fallback=None):  # 读取可选轻量JSON文件。
    target = Path(path)  # 统一路径类型。
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else fallback  # 文件不存在时使用明确默认值。

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
        return {  # 保存主机、进程和最后活动时间。
            "token": self.token,  # 保存token字段。
            "host": socket.gethostname(),  # 保存host字段。
            "pid": os.getpid(),  # 保存pid字段。
            "heartbeat_unix": time.time(),  # 保存heartbeat_unix字段。
            "ttl_seconds": self.ttl  # 保存ttl_seconds字段。
        }  # 保存主机、进程和最后活动时间。

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

def checkpoint_store(run_dir, state, global_step, local_temp_root=None):  # 将完整训练态发布到实验数据盘双槽并保留上一完整代。
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
    partial = root / (slot + ".writing." + uuid.uuid4().hex)  # 实验数据盘侧临时文件不作为可恢复断点。
    with local.open("rb") as source, partial.open("wb") as target:  # 顺序把本地完整断点上传到实验数据盘。
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)  # 每次传输十六MiB，避免大内存副本。
        target.flush()  # 将上传缓冲交给实验数据盘挂载层。
        os.fsync(target.fileno())  # 要求挂载文件系统接收所有断点字节。
    if partial.stat().st_size != local.stat().st_size or sha256_file(partial) != digest:  # 只有实验数据盘完整副本通过内容校验才能发布。
        raise IOError(f"实验数据盘断点未完整上传：{partial}")  # 旧完整断点继续可恢复。
    os.replace(partial, destination)  # 新代上传完整后替换非活跃槽。
    receipt = {  # 明确传入receipt参数。
        "path": str(destination),  # 保存实际文件位置。
        "sha256": digest,  # 保存文件内容摘要。
        "bytes": destination.stat().st_size,  # 保存文件字节数。
        "global_step": step,  # 保存已完成的更新步数。
        "loss_sequence": int(payload.get("loss_committed_sequence", 0)),  # 保存已提交的损失游标。
        "saved_unix": time.time()  # 保存saved_unix字段。
    }  # 绑定训练态与损失日志的共同安全游标。
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
            if (  # 继续核对当前条件，保持原判断顺序。
                path.name not in {"checkpoint_a.pt", "checkpoint_b.pt"}  # 核对路径、大小和完整字节身份。
                or not path.is_file()  # 继续核对当前条件，保持原判断顺序。
                or path.stat().st_size != int(receipt["bytes"])  # 继续核对当前条件，保持原判断顺序。
                or sha256_file(path) != receipt["sha256"]  # 继续核对当前条件，保持原判断顺序。
            ):  # 核对路径、大小和完整字节身份。
                raise IOError("断点大小、位置或内容摘要不一致")  # 未完整写入的文件不能用于续训。
            payload = load_checkpoint(path)  # 恢复包含优化器和随机数的完整字典。
            if (  # 继续核对当前条件，保持原判断顺序。
                not isinstance(payload, dict)  # 张量内容游标必须和正式收据共同匹配。
                or int(payload.get("global_step", -1)) != int(receipt["global_step"])  # 继续核对当前条件，保持原判断顺序。
                or int(payload.get("loss_committed_sequence", 0)) != int(receipt.get("loss_sequence", 0))  # 继续核对当前条件，保持原判断顺序。
            ):  # 张量内容游标必须和正式收据共同匹配。
                raise IOError("完整训练态与指针游标不一致")  # 拒绝状态和日志错配。
            if key == "previous":  # 当前代损坏时将正式指针回退为实际恢复代。
                atomic_json(  # 绘图和后续保存同步使用回退边界。
                    root / "checkpoint_pointer.json",  # 绘图和后续保存同步使用回退边界。
                    {"format_version": 1, "current": receipt, "previous": None, "fallback_reason": failures}  # 绘图和后续保存同步使用回退边界。
                )  # 绘图和后续保存同步使用回退边界。
                print(f"已回退到上一完整断点：step {receipt['global_step']}")  # 让用户看到实际恢复进度。
            else:  # 最新完整代正常时直接报告真实游标。
                print(f"从实验数据盘完整断点续训：step {receipt['global_step']}")  # 不把已回滚批次计入百分比。
            return payload, receipt  # 同时返回训练态和安全日志边界。
        except Exception as exc:  # 保存当前候选实际失败原因。
            failures.append(f"{key}: {exc}")  # 让无法恢复时的诊断完整可读。
    raise RuntimeError("两个正式断点均无法完整恢复：" + "; ".join(failures))  # 两代都损坏时禁止从头覆盖旧目录。

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
        self.segments = [{  # 保留安全段并明确截断旧轨迹尾部，避免重训后编号冲突。
            **item,  # 保留安全段并明确截断旧轨迹尾部，避免重训后编号冲突。
            "last_sequence": min(int(item["last_sequence"]), self.limit)  # 保存last_sequence字段。
        } for item in old.get("segments", []) if int(item["first_sequence"]) <= self.limit]  # 保留安全段并明确截断旧轨迹尾部，避免重训后编号冲突。

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
        record = {  # 明确传入record参数。
            "sequence": self.sequence,  # 保存sequence字段。
            "split": split,  # 保存split字段。
            "global_step": int(step),  # 保存已完成的更新步数。
            "time_unix": time.time(),  # 保存time_unix字段。
            "losses": values,  # 保存losses字段。
            "metadata": metadata  # 保存metadata字段。
        }  # 保存步数、集合、损失及附加统计。
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
        self.segments.append({  # 将当前段纳入轨迹清单。
            "path": relative.as_posix(),  # 保存实际文件位置。
            "first_sequence": first,  # 保存first_sequence字段。
            "last_sequence": last,  # 保存last_sequence字段。
            "bytes": target.stat().st_size,  # 保存文件字节数。
            "sha256": sha256_file(target)  # 保存文件内容摘要。
        })  # 将当前段纳入轨迹清单。
        atomic_json(self.manifest_path, {"format_version": 1, "last_sequence": last, "segments": self.segments})  # 原子发布新日志清单。
        self.records.extend(accepted)  # 更新当前进程的已提交记录。
        self.pending = [record for record in self.pending if record["sequence"] > last]  # 仅保留尚未纳入完整训练态的尾部。
        self.limit = last  # 保存新的正式日志边界。
        return last  # 训练器必须把该值共同保存进完整断点。


# 以下函数保留原正式训练的前向、梯度累计与循环损失计算次序。


def finite_losses(losses):  # 让非法数值不能进入损失日志和断点。
    if not set(settings.LOSS_KEYS) <= set(losses):  # 每次完整更新必须报告全部基础和Cycle组成。
        raise RuntimeError(f"损失字段不完整：{sorted(set(settings.LOSS_KEYS) - set(losses))}")  # 列出缺失字段供排查。
    if any(not math.isfinite(float(value)) for value in losses.values()):  # 检查所有实际记录的数值。
        raise FloatingPointError("损失含 NaN 或无穷大，停止并保留此前完整断点。")  # 坏批次不能伪装成已完成。
    return {name: float(value) for name, value in losses.items()}  # 用普通数值记录，避免日志保存GPU计算图。


def run_known_reflection_backward(torch_module, operation):  # 只在包含CUDA反射填充反向的最小调用范围允许PyTorch没有确定性实现的算子。
    torch_module.use_deterministic_algorithms(False)  # RTX2080Ti上的reflection_pad1d_backward没有可选择的确定性CUDA实现。
    try:  # 无论目标反向成功或失败都必须恢复其余训练的严格确定性检查。
        return operation()  # 执行已知包含冻结判别器或声码器反射填充反向的单个调用。
    finally:  # 异常路径同样恢复严格模式，避免后续算子静默放宽。
        torch_module.use_deterministic_algorithms(True, warn_only=False)  # 除明确包围的反射填充反向外继续拒绝非确定性CUDA算子。


def load_python(path, name):  # 从明确的只读源码路径载入一个Python模块。
    spec = importlib.util.spec_from_file_location(name, Path(path))  # 指定文件来源而不是靠随机同名模块搜索。
    if spec is None or spec.loader is None:  # 文件不存在或无法载入时停止。
        raise ImportError(f"不能载入必要源码：{path}")  # 显示需要缓存的具体文件。
    module = importlib.util.module_from_spec(spec)  # 创建模块对象供加载器执行。
    sys.modules[name] = module  # 让模块内部类型和导入能够找到自身。
    spec.loader.exec_module(module)  # 执行定义，不执行带main保护的训练入口。
    return module  # 返回原实现供精确复用。


class LegacySTFTTorchProxy:  # 为旧Conformer的一个模块补上新版PyTorch要求的明确复数表示参数。
    def __init__(self, real_torch):  # 保留真正的PyTorch对象，其他所有接口继续原样使用。
        self.real_torch = real_torch  # 不修改真实torch模块或任何其他调用者。

    def __getattr__(self, name):  # 张量类型、补齐函数和其余属性直接委托原PyTorch。
        return getattr(self.real_torch, name)  # 只有stft这一处使用明确的兼容默认值。

    def stft(self, *args, **options):  # 新版要求实输入显式指定复数返回格式，旧代码需要末轴实部与虚部。
        options.setdefault("return_complex", False)  # 保留旧[...,2]实虚布局，避免改成复杂张量后破坏原前端。
        return self.real_torch.stft(*args, **options)  # 实际FFT仍由原PyTorch实现，不改变数值运算和其他默认参数。


class FrozenBNFBridge:  # 将第一次生成的mel真正转成波形并重新提取内容特征。
    def __init__(self, runtime, root, device, checkpoint=None):  # 绑定固定Conformer实现、权重及GPU设备。
        self.runtime = runtime  # 保存共享的原实现和张量工具。
        self.device = device  # 保存当前实验主机GPU。
        legacy = Path(root) / "external/legacy_compute_bnf"  # 定位与离线提取相同的公开内容编码器。
        weights = Path(checkpoint) if checkpoint else legacy / "conformer_ppg_model/en_conformer_ctc_att/24epoch.pth"  # 默认只使用已恢复的24epoch权重。
        if not weights.is_file() or sha256_file(weights) != settings.BNF_SHA256:  # 不允许其他权重被同名文件冒充。
            raise RuntimeError(f"缺少或不匹配的固定Conformer权重：{weights}")  # 在第一批训练前报出缺项。
        helper = load_python(Path(root) / "preprocess/05_extract_bnf.py", "formal_bnf_compat")  # 载入原提取程序中已经锁定的ESPnet兼容实现。
        sys.path.insert(0, str(legacy))  # 让内容编码器包从原项目中导入。
        helper.install_pinned_espnet_embedding_compat(legacy)  # 安装经过原项目审核的同源位置编码接口。
        from conformer_ppg_model.build_ppg_model import build_model  # 复用原结构构建函数，避免旧加载器受新版torch.load默认值影响。
        import conformer_ppg_model.stft as legacy_stft  # 只定位上游Conformer独立STFT模块的局部torch引用。
        legacy_stft.torch = LegacySTFTTorchProxy(runtime.torch)  # 仅替换这个模块的引用，原torch.stft与所有资产文件保持不被修改。
        import yaml  # 按原模型加载器相同方法读取固定结构配置。
        config_path = legacy / "conformer_ppg_model/en_conformer_ctc_att/config.yaml"  # Conformer前端和网络设置必须与离线BNF相同。
        if sha256_file(config_path) != settings.BNF_CONFIG_SHA256:  # 文件名相同不能证明实际前端及结构相同。
            raise RuntimeError("Conformer配置与固定公开权重不匹配。")  # 在任何训练前拒绝结构漂移。
        self.model = build_model(argparse.Namespace(**yaml.safe_load(config_path.read_text(encoding="utf-8"))))  # 逐项按同一个上游函数建立前端、归一化及编码器。
        pretrained = load_checkpoint(weights)  # 明确weights_only=False兼容新旧PyTorch，不修改全局torch.load行为。
        encoder_weights = {name: value for name, value in pretrained.items() if "encoder" in name}  # 完全遵循上游只载入Conformer encoder权重的规则。
        complete = self.model.state_dict()  # 音频前端固定buffer由同一结构代码产生。
        if set(encoder_weights) != {name for name in complete if "encoder" in name}:  # 不允许留下随机encoder层冒充冻结预训练编码器。
            raise RuntimeError("Conformer预训练encoder权重字段不完整。")  # 缺层或多层应在正式训练前报告。
        complete.update(encoder_weights)  # 用已验证的预训练encoder替换构造时的临时参数。
        self.model.load_state_dict(complete, strict=True)  # 严格核对全部参数形状和固定buffer。
        self.model.to(device)  # 将完整固定内容编码器放到当前实验主机GPU。
        self.model.eval()  # 固定编码器关闭随机训练行为。
        for parameter in self.model.parameters():  # 遍历全部编码器参数。
            parameter.requires_grad_(False)  # 参数冻结，Cycle只通过返回转换更新FastVoiceGrad。

    def extract(self, converted_mel, lengths, vocoder, mean, std):  # 从生成mel获得重新提取且对齐的144维BNF。
        torch = self.runtime.torch  # 使用当前真实PyTorch版本。
        with torch.no_grad():  # 第一程和内容重提取都不保留梯度图。
            waveform = vocoder(  # 使用冻结声码器把生成mel还原成真实波形。
                self.runtime.training.denormalize_mel(converted_mel.detach(), mean, std)  # 先恢复自然尺度并停止第一程梯度。
            ).squeeze(1)  # 去掉单通道维度，便于逐条裁剪。
            waves = []  # 保存去掉补齐部分并重采样后的每条有效波形。
            for index, valid_frames in enumerate(lengths.tolist()):  # 每条语音按真实帧数处理，避免把padding当作内容。
                samples = min(waveform.shape[-1], int(valid_frames) * settings.HOP_SIZE)  # HiFi-GAN每个mel帧对应256个音频点。
                waves.append(self.runtime.torchaudio.functional.resample(  # 真实重采样到Conformer使用的16kHz。
                    waveform[index, :samples],  # 真实重采样到Conformer使用的16kHz。
                    settings.SAMPLE_RATE,  # 真实重采样到Conformer使用的16kHz。
                    settings.CONTENT_SAMPLE_RATE  # 真实重采样到Conformer使用的16kHz。
                ))  # 真实重采样到Conformer使用的16kHz。
            wav_lengths = torch.tensor([wave.numel() for wave in waves], device=self.device, dtype=torch.long)  # 让前端和归一化知道每条波形的有效长度。
            padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)  # 只为同一小批组装张量，实际长度另外传入。
            features, feature_lengths = self.model._extract_feats(padded, wav_lengths)  # 使用同一Conformer音频前端，保留有效特征长度。
            normalized, feature_lengths = self.model.normalize(features, feature_lengths)  # 按有效帧完成与离线提取一致的归一化。
            encoded, encoded_lengths, _ = self.model.encoder(normalized, feature_lengths)  # 同一编码器输出144维内容并返回实际长度。
            aligned = []  # 保存逐样本对齐到有效mel长度的内容条件。
            for index, valid_frames in enumerate(lengths.tolist()):  # 不让不同长度样本被相同最大长度错误拉伸。
                raw = encoded[index, : int(encoded_lengths[index]), :].transpose(0, 1)  # 先裁掉Conformer内部补帧再转成通道在前。
                if raw.shape[0] != 144 or raw.shape[-1] < 1:  # 原模型接口必须提供有效144维内容。
                    raise RuntimeError(f"重新提取BNF维数错误：{tuple(raw.shape)}")  # 异常内容不能进入第二程。
                value = self.runtime.align_bnf_to_mel(raw, int(valid_frames), mode="legacy_length_linear")  # 保留现有FastVoiceGrad权重的对齐协议。
                aligned.append(self.runtime.F.pad(value, (0, converted_mel.shape[-1] - int(valid_frames))))  # 只有有效部分对齐，右边补齐到训练128帧。
            result = torch.stack(aligned).detach()  # 返回与第二程mel同长的冻结内容条件。
            if not bool(torch.isfinite(result).all()):  # 新BNF必须完全为有限数。
                raise FloatingPointError("重新提取的BNF包含坏数。")  # 不用源BNF静默替代失败的重提取。
            return result  # 交给B到A返回转换使用。


def masked_per_sample_mae(runtime, reference, prediction, lengths):  # 先计算每个样本的80通道有效帧平均绝对误差。
    torch = runtime.torch  # 使用运行时张量库。
    if reference.shape != prediction.shape or reference.shape[1] != settings.N_MELS:  # 公式要求两份mel对应且有80个通道。
        raise ValueError("masked MAE要求同形的80通道mel。")  # 不接受广播产生的假误差。
    lengths = lengths.to(reference.device, dtype=torch.long)  # 把有效长度转换为同设备整数。
    if bool((lengths < 1).any()) or bool((lengths > reference.shape[-1]).any()):  # 有效帧数必须落在真实张量范围内。
        raise ValueError("masked MAE的有效长度越界。")  # 防止除零和漏屏蔽padding。
    mask = torch.arange(reference.shape[-1], device=reference.device)[None, :] < lengths[:, None]  # q_it在有效帧为1，右侧补帧为0。
    return ((reference - prediction).abs() * mask[:, None, :]).sum(dim=(1, 2)) / (settings.N_MELS * lengths)  # 已平均80通道及有效帧，绝不再额外除80。


def cycle_per_sample(  # 为同一小批执行真实生成内容重提取的A到B再到A。
    runtime,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    components,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    mel,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    bnf,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    source_spk,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    target_spk,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    lengths,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    forward_noise,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    backward_noise,  # 为同一小批执行真实生成内容重提取的A到B再到A。
    gradient=False  # 明确传入gradient参数。
):  # 为同一小批执行真实生成内容重提取的A到B再到A。
    training = runtime.training  # 复用K1/SK950的原单步预测公式。
    with runtime.torch.no_grad():  # 第一程stop-gradient只切断这次图，不冻结共享学生参数的未来能力。
        converted, _ = training.make_student_prediction(  # 源A内容与目标B身份产生第一次转换。
            components["student"],  # 源A内容与目标B身份产生第一次转换。
            components["diffusion"],  # 源A内容与目标B身份产生第一次转换。
            mel,  # 源A内容与目标B身份产生第一次转换。
            target_spk,  # 源A内容与目标B身份产生第一次转换。
            bnf,  # 源A内容与目标B身份产生第一次转换。
            settings.STUDENT_START_STEP,  # 源A内容与目标B身份产生第一次转换。
            forward_noise  # 源A内容与目标B身份产生第一次转换。
        )  # 源A内容与目标B身份产生第一次转换。
        generated_bnf = components["bridge"].extract(  # 明确传入generated_bnf参数。
            converted,  # 对真实生成波形重提取BNF，约束转换后实际内容。
            lengths,  # 对真实生成波形重提取BNF，约束转换后实际内容。
            components["vocoder"],  # 对真实生成波形重提取BNF，约束转换后实际内容。
            components["mean"],  # 对真实生成波形重提取BNF，约束转换后实际内容。
            components["std"]  # 对真实生成波形重提取BNF，约束转换后实际内容。
        )  # 对真实生成波形重提取BNF，约束转换后实际内容。
    with runtime.torch.set_grad_enabled(bool(gradient)):  # 训练保留第二程梯度，验证完全不建反向图。
        returned, _ = training.make_student_prediction(  # 使用独立第二程噪声和原A身份返回A。
            components["student"],  # 使用独立第二程噪声和原A身份返回A。
            components["diffusion"],  # 使用独立第二程噪声和原A身份返回A。
            converted.detach(),  # 使用独立第二程噪声和原A身份返回A。
            source_spk,  # 使用独立第二程噪声和原A身份返回A。
            generated_bnf,  # 使用独立第二程噪声和原A身份返回A。
            settings.STUDENT_START_STEP,  # 使用独立第二程噪声和原A身份返回A。
            backward_noise  # 使用独立第二程噪声和原A身份返回A。
        )  # 使用独立第二程噪声和原A身份返回A。
        return masked_per_sample_mae(runtime, mel, returned, lengths)  # 每条语音独立平均，调用者再对选中的样本平均。


def build_components(runtime, args, device):  # 精确复用已复现模型结构及冻结依赖，派生报告中的新增Cycle参数。
    (  # 按原构建顺序接收模型和优化器，避免改变初始化随机序列。
        config, teacher, student, vocoder, discriminator, diffusion,  # 接收模型配置与原有网络组件。
        mean, std, optimizer, d_optimizer,  # 接收归一化统计和两份优化器。
    ) = _build_baseline_components(runtime, device)  # 沿用原baseline的模型构建顺序。
    config = replace(  # 从同一个baseline配置派生当前文件夹的实验设置。
        config,  # 保留未明确覆盖的baseline结构和数据设置。
        epochs=settings.END_STEP // settings.UPDATES_PER_PASS,  # 两组都训练到固定的第150个pass。
        seed=args.seed,  # 保留本组固定随机种子。
        cycle_enabled=args.arm == "cycle_post",  # 只有Cycle组实际加入循环梯度。
        cycle_start_epoch=settings.START_STEP // settings.UPDATES_PER_PASS + 1 if args.arm == "cycle_post" else 1,  # Cycle从baseline之后的第一轮开始。
        lambda_cycle=args.lambda_cycle,  # 外层权重集中在config中设置。
        lambda_fm=settings.LAMBDA_FM,  # 原特征匹配权重明确使用集中设置。
        lambda_dist=settings.LAMBDA_DIST,  # 原教师蒸馏权重明确使用集中设置。
        train_frames=settings.TRAIN_FRAMES,  # 训练片段长度明确使用集中设置。
        batch_size=settings.BATCH_SIZE,  # 有效批大小明确使用集中设置。
        n_mels=settings.N_MELS,  # mel通道数量明确使用集中设置。
        student_start_step=settings.STUDENT_START_STEP,  # 单步学生起始扩散时刻明确使用集中设置。
        bnf_alignment=settings.BNF_ALIGNMENT,  # 内容特征对齐方式明确使用集中设置。
        cycle_examples_per_batch=settings.CYCLE_SAMPLES,  # 当前每个逻辑批使用24条循环样本。
        cycle_content_mode="reextract_generated_waveform_bnf",  # 返回程使用生成波形重新提取的内容特征。
        cycle_loss_reduction="masked_per_example_mae_then_batch_mean",  # 每条有效帧MAE后再平均24条样本。
        cycle_train_pair_seed=settings.CYCLE_PAIR_SEED,  # 异说话人抽样使用独立随机流。
        cycle_train_forward_noise_seed=settings.CYCLE_FORWARD_SEED,  # 第一程噪声独立可恢复。
        cycle_train_backward_noise_seed=settings.CYCLE_BACKWARD_SEED,  # 返回程使用另一条独立噪声流。
        num_workers=0,  # 保留原实验单进程数据读取次序。
        learning_rate=settings.STUDENT_LR,  # 保留baseline学生实际使用的学习率。
    )  # 完成当前分组的配置派生。
    bridge = FrozenBNFBridge(runtime, args.project_root, device, getattr(args, "bnf_checkpoint", None))  # 所有臂均准备同一个冻结内容提取器，用于公平验证Cycle指标。
    return {  # 返回训练和评估共享的实际对象。
        "config": config,  # 保存模型实际配置。
        "teacher": teacher,  # 保存冻结教师。
        "student": student,  # 保存当前学生。
        "vocoder": vocoder,  # 保存冻结声码器。
        "discriminator": discriminator,  # 保存判别器。
        "diffusion": diffusion,  # 保存原扩散日程。
        "mean": mean,  # 保存mel均值。
        "std": std,  # 保存mel标准差。
        "optimizer": optimizer,  # 保存学生优化器。
        "d_optimizer": d_optimizer,  # 保存判别器优化器。
        "bridge": bridge,  # 保存bridge字段。
        "device": device  # 保存device字段。
    }  # 返回训练和评估共享的实际对象。


def independent_noise(runtime, shape, device, seed):  # 使用独立局部生成器，不改变ACDD全局随机序列。
    generator = runtime.torch.Generator(device=device).manual_seed(int(seed))  # 每条噪声流由独立整数种子定义。
    return runtime.torch.randn(shape, device=device, dtype=runtime.torch.float32, generator=generator)  # 返回与学生输入同精度的标准高斯噪声。


def accumulated_base_gradients(runtime, components, batch):  # 用四个物理batch8精确累计原论文逻辑batch32的D及学生梯度。
    torch = runtime.torch  # 复用当前CUDA运行时，不在主机端执行模型计算。
    training = runtime.training  # 复用已审核的学生预测、蒸馏和串行判别器实现。
    config = components["config"]  # 原论文目标权重、扩散步数和输入尺寸保持不变。
    device = components["device"]  # 所有微批都固定在当前进程可见的单张GPU上。
    tensors = training._slice_batch(batch, device, require_cycle=False)  # 先按原契约验证完整逻辑batch32，再只在内部切成四段。
    x_zero = tensors["mel"]  # 完整32条标准化目标mel用于生成一次性的随机输入。
    bnf = tensors["bnf"]  # 完整32条内容条件保持原样本顺序。
    speaker = tensors["spk_emb"]  # 完整32条说话人条件保持原样本顺序。
    source_noise = torch.randn_like(x_zero)  # 一次抽取完整32条学生起始噪声，保持原随机数消费量。
    score_noise = torch.randn_like(x_zero)  # 一次抽取完整32条教师蒸馏噪声，保持与原实现独立。
    score_timesteps = torch.randint(1, config.diffusion_steps + 1, (config.batch_size,), device=device, dtype=torch.long)  # 一次抽取完整32个教师时刻，之后只做确定性切片。
    microbatch_size = settings.BASE_MICROBATCH_SIZE  # GTX1080Ti及RTX2080Ti显存适配采用物理batch8。
    microbatch_count = config.batch_size // microbatch_size  # 论文逻辑batch32恰好由四个等大微批组成。
    if config.batch_size != 32 or microbatch_count != 4:  # 训练配置改变时不能静默套用错误缩放。
        raise RuntimeError("梯度累计协议要求逻辑batch32由四个物理batch8组成。")  # 保护学习率与每pass更新数的既定含义。
    training.set_requires_grad(components["discriminator"], True)  # 判别器阶段允许D参数接收四个微批的累计梯度。
    components["d_optimizer"].zero_grad(set_to_none=True)  # 一个逻辑批开始时只清一次D梯度。
    gradient_hooks = [  # 为每个判别器参数设置相同的微批平均规则。
        parameter.register_hook(lambda gradient: gradient / microbatch_count)  # 每次反向的梯度除以四。
        for parameter in components["discriminator"].parameters()  # 遍历判别器实际参数。
        if parameter.requires_grad  # 只为本次需要学习的参数安装规则。
    ]  # 四次微批累积后得到逻辑batch32的平均梯度。
    discriminator_value = 0.0  # 日志同样累计四个微批损失的算术平均。
    try:  # 无论某个微批是否失败都必须移除临时梯度缩放钩子。
        for start in range(0, config.batch_size, microbatch_size):  # 按原顺序依次处理0到7、8到15、16到23、24到31。
            stop = start + microbatch_size  # 当前物理微批固定包含八条语音。
            with torch.no_grad():  # D阶段不为学生和冻结声码器保存计算图。
                prediction, _ = training.make_student_prediction(  # 使用完整逻辑批预抽取噪声的对应切片。
                    components["student"],  # 使用完整逻辑批预抽取噪声的对应切片。
                    components["diffusion"],  # 使用完整逻辑批预抽取噪声的对应切片。
                    x_zero[start:stop],  # 使用完整逻辑批预抽取噪声的对应切片。
                    speaker[start:stop],  # 使用完整逻辑批预抽取噪声的对应切片。
                    bnf[start:stop],  # 使用完整逻辑批预抽取噪声的对应切片。
                    config.student_start_step,  # 使用完整逻辑批预抽取噪声的对应切片。
                    source_noise[start:stop]  # 使用完整逻辑批预抽取噪声的对应切片。
                )  # 使用完整逻辑批预抽取噪声的对应切片。
                real_waveform = components["vocoder"](training.denormalize_mel(  # 明确传入real_waveform参数。
                    x_zero[start:stop],  # 当前八条真实mel经冻结声码器生成真实波形。
                    components["mean"],  # 当前八条真实mel经冻结声码器生成真实波形。
                    components["std"]  # 当前八条真实mel经冻结声码器生成真实波形。
                ))  # 当前八条真实mel经冻结声码器生成真实波形。
                fake_waveform = components["vocoder"](training.denormalize_mel(  # 明确传入fake_waveform参数。
                    prediction,  # 当前八条学生mel经同一冻结声码器生成假波形。
                    components["mean"],  # 当前八条学生mel经同一冻结声码器生成假波形。
                    components["std"]  # 当前八条学生mel经同一冻结声码器生成假波形。
                ))  # 当前八条学生mel经同一冻结声码器生成假波形。
            training._assert_finite("accumulated D real waveform", real_waveform)  # 坏波形不得进入累计D梯度。
            training._assert_finite("accumulated D fake waveform", fake_waveform)  # 学生输出出现坏数时立即停止当前逻辑更新。
            micro_discriminator_loss = training.serial_discriminator_backward(  # 明确传入micro_discriminator_loss参数。
                components["discriminator"],  # 八个判别器分支对当前八条样本反传。
                real_waveform,  # 八个判别器分支对当前八条样本反传。
                fake_waveform  # 八个判别器分支对当前八条样本反传。
            )  # 八个判别器分支对当前八条样本反传。
            discriminator_value += float(micro_discriminator_loss.detach()) / microbatch_count  # 四个微批均值对应完整32条损失均值。
            del prediction, real_waveform, fake_waveform, micro_discriminator_loss  # 立即释放本微批D阶段张量以控制峰值显存。
    finally:  # 钩子只属于本次D累计阶段。
        for hook in gradient_hooks:  # 遍历所有D参数的临时缩放钩子。
            hook.remove()  # 后续学生阶段和下一逻辑批不再自动缩放梯度。
    discriminator_gradients = [  # 收集四个微批已经累积的判别器梯度。
        parameter.grad  # 保留参数真实收到的梯度。
        for parameter in components["discriminator"].parameters()  # 遍历当前判别器参数。
        if parameter.grad is not None  # 尚未收到梯度的参数不加入数值检查。
    ]  # 后续检查这些实际累积结果是否有限。
    if not discriminator_gradients or any(not bool(torch.isfinite(value).all()) for value in discriminator_gradients):  # D梯度必须存在且全部有限。
        raise FloatingPointError("判别器累计梯度异常，当前逻辑批不会提交。")  # 不执行任何D优化器更新。
    components["d_optimizer"].step()  # 完整32条D梯度到齐后只执行一次Adam更新。
    components["d_optimizer"].zero_grad(set_to_none=True)  # 清除D梯度，为冻结的学生阶段建立明确边界。
    training.set_requires_grad(components["discriminator"], False)  # 学生阶段只求D对波形输入的导数，不更新D参数。
    components["optimizer"].zero_grad(set_to_none=True)  # 一个逻辑批开始时只清一次学生梯度。
    adversarial_value = 0.0  # 累计四个微批未加权对抗损失的平均。
    feature_matching_value = 0.0  # 累计四个微批未加权特征匹配损失的平均。
    distillation_value = 0.0  # 累计四个微批未加权蒸馏损失的平均。
    for start in range(0, config.batch_size, microbatch_size):  # D已经完成一次更新后，再按相同四个微批累计学生梯度。
        stop = start + microbatch_size  # 当前学生物理微批仍固定为八条。
        with torch.no_grad():  # 先构造判别器VJP使用的无梯度波形代理。
            proxy_prediction, _ = training.make_student_prediction(  # 学生参数尚未更新，复用对应噪声切片。
                components["student"],  # 学生参数尚未更新，复用对应噪声切片。
                components["diffusion"],  # 学生参数尚未更新，复用对应噪声切片。
                x_zero[start:stop],  # 学生参数尚未更新，复用对应噪声切片。
                speaker[start:stop],  # 学生参数尚未更新，复用对应噪声切片。
                bnf[start:stop],  # 学生参数尚未更新，复用对应噪声切片。
                config.student_start_step,  # 学生参数尚未更新，复用对应噪声切片。
                source_noise[start:stop]  # 学生参数尚未更新，复用对应噪声切片。
            )  # 学生参数尚未更新，复用对应噪声切片。
            real_proxy = components["vocoder"](training.denormalize_mel(  # 明确传入real_proxy参数。
                x_zero[start:stop],  # 为当前微批生成不需要梯度的真实波形代理。
                components["mean"],  # 为当前微批生成不需要梯度的真实波形代理。
                components["std"]  # 为当前微批生成不需要梯度的真实波形代理。
            ))  # 为当前微批生成不需要梯度的真实波形代理。
            fake_proxy = components["vocoder"](training.denormalize_mel(  # 明确传入fake_proxy参数。
                proxy_prediction,  # 为当前微批生成不需要梯度的学生波形代理。
                components["mean"],  # 为当前微批生成不需要梯度的学生波形代理。
                components["std"]  # 为当前微批生成不需要梯度的学生波形代理。
            ))  # 为当前微批生成不需要梯度的学生波形代理。
        adversarial_loss, feature_matching_loss, waveform_gradient = run_known_reflection_backward(  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
            torch,  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
            lambda: training.serial_student_discriminator_gradient(  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
                components["discriminator"],  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
                real_proxy,  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
                fake_proxy,  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
                config.lambda_fm  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
            )  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
        )  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
        prediction, _ = training.make_student_prediction(  # 重新建立学生计算图并复用完全相同的输入。
            components["student"],  # 重新建立学生计算图并复用完全相同的输入。
            components["diffusion"],  # 重新建立学生计算图并复用完全相同的输入。
            x_zero[start:stop],  # 重新建立学生计算图并复用完全相同的输入。
            speaker[start:stop],  # 重新建立学生计算图并复用完全相同的输入。
            bnf[start:stop],  # 重新建立学生计算图并复用完全相同的输入。
            config.student_start_step,  # 重新建立学生计算图并复用完全相同的输入。
            source_noise[start:stop]  # 重新建立学生计算图并复用完全相同的输入。
        )  # 重新建立学生计算图并复用完全相同的输入。
        fake_waveform = components["vocoder"](training.denormalize_mel(prediction, components["mean"], components["std"]))  # 建立波形到学生的真实反向路径。
        if not torch.equal(fake_waveform.detach(), fake_proxy):  # 确定性后端必须让VJP代理点和真实反向点逐位完全相同。
            maximum_difference = float(torch.max(torch.abs(fake_waveform.detach() - fake_proxy)))  # 报告最大差值以定位随机层或数值漂移。
            raise RuntimeError(f"物理微批波形重算不一致：max_abs_diff={maximum_difference:.9g}")  # 不把错误VJP加入学生梯度。
        distillation_loss, _ = training.score_distillation_loss(  # 当前八条使用完整逻辑批预抽取的教师随机量。
            prediction,  # 当前八条使用完整逻辑批预抽取的教师随机量。
            components["teacher"],  # 当前八条使用完整逻辑批预抽取的教师随机量。
            components["diffusion"],  # 当前八条使用完整逻辑批预抽取的教师随机量。
            score_timesteps[start:stop],  # 当前八条使用完整逻辑批预抽取的教师随机量。
            score_noise[start:stop],  # 当前八条使用完整逻辑批预抽取的教师随机量。
            speaker[start:stop],  # 当前八条使用完整逻辑批预抽取的教师随机量。
            bnf[start:stop],  # 当前八条使用完整逻辑批预抽取的教师随机量。
            target_mode=config.score_target,  # 明确传入target_mode参数。
            weight_mode=config.score_weight  # 明确传入weight_mode参数。
        )  # 当前八条使用完整逻辑批预抽取的教师随机量。
        for name, value in (  # 按原顺序遍历需要处理的项目。
            ("student adversarial loss", adversarial_loss),  # 三项原论文基础损失都必须有限。
            ("raw feature matching loss", feature_matching_loss),  # 三项原论文基础损失都必须有限。
            ("raw score distillation loss", distillation_loss)  # 三项原论文基础损失都必须有限。
        ):  # 三项原论文基础损失都必须有限。
            training._assert_finite(name, value)  # 任一微批坏数都会取消整个逻辑更新。
        run_known_reflection_backward(  # 只让冻结声码器路径中的已知反射填充反向采用CUDA现有实现，并保留共享学生图供蒸馏项继续累计。
            torch,  # 只让冻结声码器路径中的已知反射填充反向采用CUDA现有实现，并保留共享学生图供蒸馏项继续累计。
            lambda: torch.autograd.backward(  # 只让冻结声码器路径中的已知反射填充反向采用CUDA现有实现，并保留共享学生图供蒸馏项继续累计。
                fake_waveform,  # 只让冻结声码器路径中的已知反射填充反向采用CUDA现有实现，并保留共享学生图供蒸馏项继续累计。
                waveform_gradient / microbatch_count,  # 只让冻结声码器路径中的已知反射填充反向采用CUDA现有实现，并保留共享学生图供蒸馏项继续累计。
                retain_graph=True  # 明确传入retain_graph参数。
            )  # 只让冻结声码器路径中的已知反射填充反向采用CUDA现有实现，并保留共享学生图供蒸馏项继续累计。
        )  # 只让冻结声码器路径中的已知反射填充反向采用CUDA现有实现，并保留共享学生图供蒸馏项继续累计。
        distillation_loss.backward(distillation_loss.new_tensor(config.lambda_dist / microbatch_count))  # 不含反射填充的蒸馏路径仍在严格确定性模式下累计四分之一梯度。
        adversarial_value += float(adversarial_loss.detach()) / microbatch_count  # 日志按四个等大微批平均。
        feature_matching_value += float(feature_matching_loss.detach()) / microbatch_count  # 日志缩放与实际梯度目标一致。
        distillation_value += float(distillation_loss.detach()) / microbatch_count  # 蒸馏日志同样恢复完整batch32均值。
        del proxy_prediction, real_proxy, fake_proxy, adversarial_loss, feature_matching_loss, waveform_gradient, prediction, fake_waveform, distillation_loss  # 释放本微批计算图。
    if any(parameter.grad is not None for parameter in components["teacher"].parameters()):  # 冻结教师不能因微批累计收到参数梯度。
        raise RuntimeError("冻结教师意外收到梯度。")  # 保护原论文教师固定不更新的约束。
    if any(parameter.grad is not None for parameter in components["vocoder"].parameters()):  # 声码器只允许梯度穿过输入。
        raise RuntimeError("冻结声码器意外收到参数梯度。")  # 防止把声码器混入学生Adam。
    if any(parameter.grad is not None for parameter in components["discriminator"].parameters()):  # 学生阶段D参数必须保持无梯度。
        raise RuntimeError("冻结判别器在学生阶段意外收到梯度。")  # 保证一次D更新与一次学生更新严格分离。
    return {  # 返回与实际逻辑batch32目标一致的九项日志，学生step由外层在Cycle梯度后执行。
        "student_total": adversarial_value + config.lambda_fm * feature_matching_value + config.lambda_dist * distillation_value,  # 保存实际加权学生总损失。
        "student_adv": adversarial_value,  # 保存学生对抗损失。
        "fm_raw": feature_matching_value,  # 保存未加权特征匹配损失。
        "fm_weighted": config.lambda_fm * feature_matching_value,  # 保存加权特征匹配损失。
        "dist_raw": distillation_value,  # 保存未加权蒸馏损失。
        "dist_weighted": config.lambda_dist * distillation_value,  # 保存加权蒸馏损失。
        "cycle_raw": 0.0,  # 保存未加权循环损失。
        "cycle_weighted": 0.0,  # 保存实际加入梯度的循环损失。
        "discriminator": discriminator_value  # 保存判别器。
    }  # 返回与实际逻辑batch32目标一致的九项日志，学生step由外层在Cycle梯度后执行。


def formal_train_batch(runtime, components, batch, args, next_step, cycle_step):  # 一次完整D更新及S更新，按门控决定是否加入Cycle。
    config = components["config"]  # 读取报告中固定参数。
    device = components["device"]  # 使用当前真实GPU。
    active = arm_spec(args.arm)["cycle_after"] is not None and next_step > arm_spec(args.arm)["cycle_after"]  # 历史Early在第21pass首批开启，Post在父断点后首批开启。
    losses = accumulated_base_gradients(runtime, components, batch)  # 四个物理batch8先累计成原论文逻辑batch32，D和学生都暂不增加额外更新次数。
    cycle_value = 0.0  # Base和历史Early前20pass完全不运行Cycle额外前向。
    if active:  # 只有实际Cycle更新才构建24条往返样本。
        tensors = runtime.training._slice_batch(batch, device, require_cycle=True)  # 原字段检查保证mel80、BNF144、身份256及有效长度正确。
        pair_generator = runtime.torch.Generator(device=device).manual_seed(  # 为异说话人选择创建独立随机流。
            config.cycle_train_pair_seed + (args.seed - settings.SEED) + cycle_step + 1  # 依据循环更新游标续接原随机序列。
        )  # 不消耗原ACDD使用的全局随机流。
        targets = runtime.training.select_cycle_target_indices(  # 明确传入targets参数。
            batch["spk_name"],  # 从batch32候选池为24个源选异说话人，允许重复目标。
            settings.CYCLE_SAMPLES,  # 从batch32候选池为24个源选异说话人，允许重复目标。
            device,  # 从batch32候选池为24个源选异说话人，允许重复目标。
            generator=pair_generator  # 明确传入generator参数。
        )  # 从batch32候选池为24个源选异说话人，允许重复目标。
        forward_noise = independent_noise(  # 明确传入forward_noise参数。
            runtime,  # 第一程独立噪声按已完成Cycle步数续接，1234重复不改原登记种子。
            (settings.CYCLE_SAMPLES, settings.N_MELS, settings.TRAIN_FRAMES),  # 第一程独立噪声按已完成Cycle步数续接，1234重复不改原登记种子。
            device,  # 第一程独立噪声按已完成Cycle步数续接，1234重复不改原登记种子。
            config.cycle_train_forward_noise_seed + (args.seed - settings.SEED) + cycle_step + 1  # 第一程独立噪声按已完成Cycle步数续接，1234重复不改原登记种子。
        )  # 第一程独立噪声按已完成Cycle步数续接，1234重复不改原登记种子。
        backward_noise = independent_noise(  # 明确传入backward_noise参数。
            runtime,  # 第二程噪声与第一程和ACDD都独立。
            (settings.CYCLE_SAMPLES, settings.N_MELS, settings.TRAIN_FRAMES),  # 第二程噪声与第一程和ACDD都独立。
            device,  # 第二程噪声与第一程和ACDD都独立。
            config.cycle_train_backward_noise_seed + (args.seed - settings.SEED) + cycle_step + 1  # 第二程噪声与第一程和ACDD都独立。
        )  # 第二程噪声与第一程和ACDD都独立。
        for start in range(0, settings.CYCLE_SAMPLES, settings.CYCLE_MICROBATCH_SIZE):  # 24条分成8组，每组3条，只控制峰值显存而不改变逻辑batch。
            stop = start + settings.CYCLE_MICROBATCH_SIZE  # 每组刚好3个Cycle源样本。
            errors = cycle_per_sample(  # 明确传入errors参数。
                runtime,  # 第一程无梯度，第二程保留对同一个学生的梯度。
                components,  # 第一程无梯度，第二程保留对同一个学生的梯度。
                tensors["mel"][start:stop],  # 第一程无梯度，第二程保留对同一个学生的梯度。
                tensors["bnf"][start:stop],  # 第一程无梯度，第二程保留对同一个学生的梯度。
                tensors["spk_emb"][start:stop],  # 第一程无梯度，第二程保留对同一个学生的梯度。
                tensors["spk_emb"].index_select(0, targets[start:stop]),  # 第一程无梯度，第二程保留对同一个学生的梯度。
                tensors["mel_len"][start:stop],  # 第一程无梯度，第二程保留对同一个学生的梯度。
                forward_noise[start:stop],  # 第一程无梯度，第二程保留对同一个学生的梯度。
                backward_noise[start:stop],  # 第一程无梯度，第二程保留对同一个学生的梯度。
                gradient=True  # 明确传入gradient参数。
            )  # 第一程无梯度，第二程保留对同一个学生的梯度。
            contribution = errors.sum() / settings.CYCLE_SAMPLES  # 八组相加等于24条逐样本MAE的平均，每组均值等价乘1/8。
            if not bool(runtime.torch.isfinite(contribution)):  # 任一微批坏数都会让当前完整更新失效。
                raise FloatingPointError("Cycle微批误差出现NaN或无穷大。")  # 不执行学生Adam也不保存半批。
            (args.lambda_cycle * contribution).backward()  # 仅累加当前微批的加权梯度，累计原ACDD梯度不清零。
            cycle_value += float(contribution.detach())  # 保存实际24条平均的未加权Cycle损失。
            del contribution, errors  # 立即释放本微批第二程计算图，避免8组同时占显存。
    gradients = [parameter.grad for parameter in components["student"].parameters() if parameter.grad is not None]  # 收集原ACDD及Cycle叠加后的真实梯度。
    if not gradients or any(not bool(runtime.torch.isfinite(value).all()) for value in gradients):  # 合并后的梯度必须存在且全部为有限数。
        raise FloatingPointError("学生合并梯度异常，当前半批不会提交。")  # 留下此前完整断点用于重新运行。
    if not any(bool(runtime.torch.count_nonzero(value)) for value in gradients):  # 全零梯度不能冒充一次有效逻辑更新。
        raise RuntimeError("学生合并梯度全部为零，当前逻辑批不会提交。")  # 避免推进无学习效果的Adam游标。
    components["optimizer"].step()  # 一个逻辑batch只执行一次学生Adam，保持D到S更新比例1比1。
    components["optimizer"].zero_grad(set_to_none=True)  # 完整更新后清空梯度，为下一批准备。
    runtime.training.set_requires_grad(components["discriminator"], True)  # 恢复D参数可训练状态，保持后续验证和下一逻辑批的明确边界。
    losses.update(  # 日志总值与实际反向目标严格对应。
        cycle_raw=cycle_value,  # 明确传入未加权循环损失。
        cycle_weighted=args.lambda_cycle * cycle_value,  # 明确传入实际加入梯度的循环损失。
        student_total=losses["student_total"] + args.lambda_cycle * cycle_value  # 明确传入实际加权学生总损失。
    )  # 日志总值与实际反向目标严格对应。
    return finite_losses(losses)  # 返回全部实际训练项供逐批持久化。


def fixed_sample(runtime, dataset, index, seed):  # 给开发或测试语音固定128帧片段，跨pass和续训不改变输入。
    sample = dict(dataset[int(index)])  # 底层完整语音只读，不使用训练随机裁剪。
    frames = int(sample["mel"].shape[-1])  # 读取真实帧数以裁剪和屏蔽padding。
    identity = f"{seed}\0{sample['spk_name']}\0{sample['utt_id']}".encode("utf-8")  # 固定种子与样本身份共同定义片段。
    choices = max(1, frames - 128 + 1)  # 最后一个合法裁剪起点也包含在候选中。
    start = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % choices  # 用稳定哈希避免Python随机哈希和worker调度影响。
    valid = min(128, frames)  # 短语音只有真实帧计入MAE。
    for key in ("mel", "bnf"):  # mel与BNF必须同时裁剪到同一内容片段。
        value = sample[key][..., start : start + 128]  # 保留每种特征所有通道。
        sample[key] = runtime.F.pad(value, (0, 128 - value.shape[-1])).contiguous()  # 在右侧补零，不增加有效长度。
    sample["mel_len"] = runtime.torch.tensor(valid, dtype=runtime.torch.long)  # 显式保留供masked MAE使用的真实帧数。
    return sample  # 返回固定语音片段及原身份元数据。


def evaluate_losses(runtime, components, dataset, indices, split, args, global_step):  # 在完整52条开发集或100条测试集上计算真实损失，绝不反向或更新。
    torch = runtime.torch  # 使用实际运行时张量库。
    models = {key: components[key] for key in ("teacher", "student", "vocoder", "discriminator")}  # 所有模型模式都要在诊断后恢复。
    modes = {key: model.training for key, model in models.items()}  # 记录训练前的原模式。
    rng = runtime.training.capture_rng_state()  # 验证不能改变正式下一批的训练随机序列。
    device = components["device"]  # 诊断与训练使用同一GPU。
    config = components["config"]  # 诊断保留相同损失定义和系数。
    all_indices = list(indices)  # 固定顺序覆盖整个对应集合。
    speaker_names = [str(dataset.items[index]["speaker"]) for index in all_indices]  # 跨完整清单找异说话人目标，末尾小批也不会退化。
    if len(set(speaker_names)) < 2:  # 所有Cycle诊断必须包含至少两个真实说话人。
        raise RuntimeError("诊断集合没有异说话人可供Cycle转换。")  # 禁止用A到A假装语音转换。
    references = [  # 每个固定诊断样本都使用确定的异说话人目标。
        next(  # 按原顺序寻找下一位不同说话人。
            all_indices[(position + shift) % len(all_indices)]  # 返回实际数据索引。
            for shift in range(1, len(all_indices))  # 从下一条开始逐项查找。
            if speaker_names[(position + shift) % len(all_indices)] != speaker_names[position]  # 目标身份必须不同于源。
        )  # 不随机更换诊断目标。
        for position in range(len(all_indices))  # 为全部实际诊断样本建立固定目标。
    ]  # 完成可跨轮比较的目标配对。
    totals = {key: 0.0 for key in settings.LOSS_KEYS}  # 累积逐批均值乘真实样本数，尾批不被放大。
    totals["reconstruction_masked_mae"] = 0.0  # 额外记录对原mel的有效帧重建误差。
    seen = set()  # 验证每条规定语音都恰好读到一次。
    active = expected_cycle_step(args.arm, global_step) > 0  # 总目标Cycle贡献只在相应正式分支启用后计入。
    try:  # 失败也必须恢复模型模式和训练随机状态。
        for model in models.values():  # 教师、学生、声码器和D全都使用评估模式。
            model.eval()  # 固定诊断不使用训练随机行为。
        with torch.no_grad():  # 完全禁止建立参数反向图。
            for offset in range(0, len(all_indices), 32):  # 不丢尾批，开发52条为32及20，测试100条为32及32及32及4。
                selected = all_indices[offset : offset + 32]  # 当前批真实样本索引。
                batch = runtime.default_collate([fixed_sample(runtime, dataset, index, 10_001_234) for index in selected])  # 各模型各pass比较同一固定片段。
                tensors = {key: batch[key].to(device) for key in ("mel", "bnf", "spk_emb", "mel_len")}  # 只搬运实际计算所需张量。
                mel, bnf, speaker, lengths = (tensors[key] for key in ("mel", "bnf", "spk_emb", "mel_len"))  # 取出同一源语音的声学、内容、身份和有效长度。
                size = len(selected)  # 尾批实际规模参与加权。
                bank_seed = (300_000_000 if split == "development" else 400_000_000) + offset  # 开发和测试噪声银行固定且彼此独立。
                noise = independent_noise(runtime, mel.shape, device, bank_seed + 1)  # 固定学生起始噪声，跨pass可直接比较。
                score_noise = independent_noise(runtime, mel.shape, device, bank_seed + 2)  # 蒸馏噪声与学生起始噪声独立。
                generator = torch.Generator(device=device).manual_seed(bank_seed + 3)  # 教师时刻使用第三条独立随机流。
                timesteps = torch.randint(1, 1001, (size,), device=device, generator=generator)  # 每条样本使用固定的随机教师噪声等级。
                predicted, _ = runtime.training.make_student_prediction(  # 与训练相同的单步学生预测。
                    components["student"],  # 与训练相同的单步学生预测。
                    components["diffusion"],  # 与训练相同的单步学生预测。
                    mel,  # 与训练相同的单步学生预测。
                    speaker,  # 与训练相同的单步学生预测。
                    bnf,  # 与训练相同的单步学生预测。
                    settings.STUDENT_START_STEP,  # 与训练相同的单步学生预测。
                    noise  # 与训练相同的单步学生预测。
                )  # 与训练相同的单步学生预测。
                real_wave = components["vocoder"](runtime.training.denormalize_mel(mel, components["mean"], components["std"]))  # 真实mel经过相同冻结声码器供D诊断。
                fake_wave = components["vocoder"](runtime.training.denormalize_mel(predicted, components["mean"], components["std"]))  # 当前学生生成波形供D诊断。
                d_loss, adv, fm = runtime.training.serial_discriminator_metrics(  # 真实计算8个判别分支的D、对抗和特征匹配损失。
                    components["discriminator"],  # 真实计算8个判别分支的D、对抗和特征匹配损失。
                    real_wave,  # 真实计算8个判别分支的D、对抗和特征匹配损失。
                    fake_wave  # 真实计算8个判别分支的D、对抗和特征匹配损失。
                )  # 真实计算8个判别分支的D、对抗和特征匹配损失。
                dist, _ = runtime.training.score_distillation_loss(  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                    predicted,  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                    components["teacher"],  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                    components["diffusion"],  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                    timesteps,  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                    score_noise,  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                    speaker,  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                    bnf,  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                    target_mode=config.score_target,  # 明确传入target_mode参数。
                    weight_mode=config.score_weight  # 明确传入weight_mode参数。
                )  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                targets = torch.stack([dataset[index]["spk_emb"] for index in references[offset : offset + size]]).to(device)  # 固定跨全清单异说话人目标参考。
                first_noise = independent_noise(runtime, mel.shape, device, bank_seed + 4)  # 固定第一程Cycle噪声。
                second_noise = independent_noise(runtime, mel.shape, device, bank_seed + 5)  # 固定独立第二程Cycle噪声。
                cycle_sum = 0.0  # 验证Cycle覆盖每一条语音，不仅覆盖训练的前24条。
                for start in range(0, size, 3):  # 验证也采用3条微批避免突然增加显存。
                    stop = min(size, start + 3)  # 最后一组可以少于3条，仍保留全部样本。
                    cycle_sum += float(cycle_per_sample(  # 每条先平均80通道有效帧再加入样本总和。
                        runtime,  # 每条先平均80通道有效帧再加入样本总和。
                        components,  # 每条先平均80通道有效帧再加入样本总和。
                        mel[start:stop],  # 每条先平均80通道有效帧再加入样本总和。
                        bnf[start:stop],  # 每条先平均80通道有效帧再加入样本总和。
                        speaker[start:stop],  # 每条先平均80通道有效帧再加入样本总和。
                        targets[start:stop],  # 每条先平均80通道有效帧再加入样本总和。
                        lengths[start:stop],  # 每条先平均80通道有效帧再加入样本总和。
                        first_noise[start:stop],  # 每条先平均80通道有效帧再加入样本总和。
                        second_noise[start:stop],  # 每条先平均80通道有效帧再加入样本总和。
                        gradient=False  # 明确传入gradient参数。
                    ).sum())  # 每条先平均80通道有效帧再加入样本总和。
                cycle = cycle_sum / size  # 当前批逐样本Cycle-MAE均值，Base也测量相同诊断值。
                weighted = args.lambda_cycle * cycle if active else 0.0  # Base及warmup的目标贡献仍为零。
                values = finite_losses({  # 明确传入values参数。
                    "student_total": float(adv + 2 * fm + 45 * dist) + weighted,  # 保存实际加权学生总损失。
                    "student_adv": float(adv),  # 保存学生对抗损失。
                    "fm_raw": float(fm),  # 保存未加权特征匹配损失。
                    "fm_weighted": float(2 * fm),  # 保存加权特征匹配损失。
                    "dist_raw": float(dist),  # 保存未加权蒸馏损失。
                    "dist_weighted": float(45 * dist),  # 保存加权蒸馏损失。
                    "cycle_raw": cycle,  # 保存未加权循环损失。
                    "cycle_weighted": weighted,  # 保存实际加入梯度的循环损失。
                    "discriminator": float(d_loss),  # 保存判别器。
                    "reconstruction_masked_mae": float(masked_per_sample_mae(runtime, mel, predicted, lengths).mean())  # 保存固定重建误差。
                })  # 保留全部实际组成及额外固定重建读数。
                for key, value in values.items():  # 每项按真实样本数加权，避免末尾4条占整个batch权重。
                    totals[key] += value * size  # 累积该项样本均值之和。
                for speaker_name, utterance in zip(batch["spk_name"], batch["utt_id"]):  # 逐条检查固定集合覆盖。
                    identity = (str(speaker_name), str(utterance))  # 样本由说话人和句子共同唯一标识。
                    if identity in seen:  # 重复语音不能被当作额外验证样本。
                        raise RuntimeError(f"诊断样本重复：{identity}")  # 拒绝重复或漏项的曲线。
                    seen.add(identity)  # 登记本条已经完整计算。
    finally:  # 诊断无论成功失败都不能改变之后的训练行为。
        for key, model in models.items():  # 遍历进入诊断前记录的全部模式。
            model.train(modes[key])  # 恢复原训练或冻结模式。
        runtime.training.restore_rng_state(rng, device)  # 精确恢复训练随机数，不让验证占用下一批噪声。
    if len(seen) != len(all_indices):  # 整个固定集合必须完整唯一覆盖。
        raise RuntimeError("开发或测试损失覆盖不完整。")  # 缺项时不写看似完整的验证均值。
    return {key: value / len(all_indices) for key, value in totals.items()}  # 返回按真实总样本数平均的全部损失。


def run_name(args):  # 每个分支、种子和Cycle权重都使用独立永久目录。
    weight = format(args.lambda_cycle, ".8g").replace(".", "p")  # 文件名保留完整候选值而不依赖浮点显示小数位。
    return f"{args.arm}_seed_{args.seed}_lambda_{weight}"  # 避免多个Notebook覆盖其他分支或权重。


def state_fingerprint(runtime, model):  # 用逐张量摘要确认两臂共有同一个模型初始化。
    digest = hashlib.sha256()  # 小内存流式摘要不另存整个模型。
    for name, value in model.state_dict().items():  # 模型结构固定顺序逐项处理。
        digest.update(name.encode("utf-8"))  # 参数名也参与身份，防止同数值不同结构混用。
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())  # 记录真实参数及buffer的字节，不用文件名代替。
    return digest.hexdigest()  # 返回可直接写入合同的共同起点身份。


def restore_complete(runtime, components, payload, args, contract_sha, is_parent=False):  # 恢复两模型、两Adam和全部随机状态，权重-only文件不能续训。
    required = {  # 明确传入required参数。
        "student",  # 定义真正完整训练态所需字段。
        "discriminator",  # 定义真正完整训练态所需字段。
        "student_optimizer",  # 定义真正完整训练态所需字段。
        "discriminator_optimizer",  # 定义真正完整训练态所需字段。
        "rng_state",  # 定义真正完整训练态所需字段。
        "global_step",  # 定义真正完整训练态所需字段。
        "completed_passes",  # 定义真正完整训练态所需字段。
        "partial_batch",  # 定义真正完整训练态所需字段。
        "resumable",  # 定义真正完整训练态所需字段。
        "configuration"  # 定义真正完整训练态所需字段。
    }  # 定义真正完整训练态所需字段。
    if not isinstance(payload, dict) or not required <= set(payload):  # 缺少Adam或随机状态会改变续训数学。
        raise RuntimeError(f"断点不是完整训练状态，缺少：{sorted(required - set(payload or {}))}")  # 明确拒绝伪造或近似的续训。
    step = int(payload["global_step"])  # 读取已完成更新游标。
    if (  # 继续核对当前条件，保持原判断顺序。
        payload["partial_batch"] is not False  # 模型和优化器必须位于同一个完整批次边界。
        or payload["resumable"] is not True  # 继续核对当前条件，保持原判断顺序。
        or payload["completed_passes"] != step // settings.UPDATES_PER_PASS  # 继续核对当前条件，保持原判断顺序。
    ):  # 模型和优化器必须位于同一个完整批次边界。
        raise RuntimeError("断点不是同一完整批次边界的可恢复状态。")  # 半次D或S更新绝不能接入正式对照。
    validate_core_configuration(payload["configuration"])  # 核对原ACDD核心目标、帧数及对齐协议。
    if (  # 继续核对当前条件，保持原判断顺序。
        not is_parent  # 正式续训还必须属于同一合同和Cycle游标。
        and (  # 正式续训必须同时满足原实验身份和循环游标。
            payload.get("kind") != settings.KIND  # 核对新运行格式身份。
            or payload.get("run_contract_sha256") != contract_sha  # 核对当前实验合同。
            or payload.get("arm") != args.arm  # 核对Base或Cycle分组。
            or payload.get("cycle_step") != expected_cycle_step(args.arm, step)  # 核对真正完成的循环更新数。
        )  # 任一身份不同都不能接入当前实验。
    ):  # 正式续训还必须属于同一合同和Cycle游标。
        raise RuntimeError("当前续训断点不属于这个正式实验。")  # 防止不同臂或lambda互相接续。
    if (  # 继续核对当前条件，保持原判断顺序。
        is_parent  # Post必须共享无Cycle的完整100pass基线。
        and (  # 父状态必须是未加入循环约束的完整pass100。
            step != 100 * settings.UPDATES_PER_PASS  # 核对父状态的实际训练步数。
            or payload.get("cycle_enabled", False)  # 已训练Cycle的状态不能冒充baseline。
            or payload.get("arm", "base_100") not in ("base_100", None)  # 核对允许的历史基线标签。
        )  # 完成父状态范围和实验身份检查。
    ):  # Post必须共享无Cycle的完整100pass基线。
        raise RuntimeError("Post父断点必须是无Cycle的完整100pass基线状态。")  # 不以已经训练Cycle的模型作父状态。
    for role, model_key, optimizer_key, lr in (("学生", "student", "optimizer", 5e-5), ("判别器", "discriminator", "d_optimizer", 2e-4)):  # 同时恢复并审核两组实际状态。
        components[model_key].load_state_dict(payload[model_key], strict=True)  # 严格恢复真实权重与全部buffer。
        saved_key = "student_optimizer" if model_key == "student" else "discriminator_optimizer"  # 兼容原完整断点的正式Adam键名。
        components[optimizer_key].load_state_dict(payload[saved_key])  # 恢复动量、二阶矩和Adam step，不重置学习率。
        if step > 0:  # 初始0步Adam尚未创建动量，这是合法初始化状态。
            validate_optimizer_state(components[optimizer_key], components[model_key], lr, step, role)  # 所有参数的Adam游标必须等于真实已完成步数。
        elif components[optimizer_key].state:  # 0步断点不能含已经更新过的Adam动量。
            raise RuntimeError("0步初始化断点含有已训练Adam状态。")  # 禁止把历史学生改标签作为新起点。
        validate_model_state(components[model_key], role)  # 检查加载后的权重和buffer没有坏数。
        components[model_key].train()  # 模式不在state_dict中，因此显式恢复训练状态。
    audit = restore_stage_rng_state(payload, components["device"], step)  # 同后端严格恢复全部随机状态；旧父CUDA格式差异有原入口明确审计。
    return step, audit  # 返回真实恢复游标和随机状态恢复证据。


def checkpoint_payload(runtime, components, args, state, contract_sha, journal):  # 将完整训练态和对应损失边界组装到同一提交。
    step = int(state["global_step"])  # 只允许使用最后完成的D及S更新游标。
    sequence = journal.commit(step)  # 先发布对应已完成更新的真实loss记录，不包含未完成批次。
    return {  # 保存完整动量、RNG、累计loss、验证游标和成本，避免重连后损失曲线断层。
        "kind": settings.KIND,  # 保存kind字段。
        "arm": args.arm,  # 保存实验分组。
        "cycle_enabled": args.arm == "cycle_post",  # 保存cycle_enabled字段。
        "global_step": step,  # 保存已完成的更新步数。
        "completed_passes": step // settings.UPDATES_PER_PASS,  # 保存completed_passes字段。
        "cycle_step": expected_cycle_step(args.arm, step),  # 保存已完成的循环更新数。
        "partial_batch": False,  # 保存partial_batch字段。
        "resumable": True,  # 保存resumable字段。
        "run_contract_sha256": contract_sha,  # 保存run_contract_sha256字段。
        "configuration": components["config"].as_dict(),  # 保存configuration字段。
        "effective_optimizer_lrs": {"student": 5e-5, "discriminator": 2e-4},  # 保存effective_optimizer_lrs字段。
        "mc_samples": 1,  # 保存mc_samples字段。
        "score_semantics": "reconstruction_clean_x0_sqrt_alpha_bar",  # 保存score_semantics字段。
        "student": components["student"].state_dict(),  # 保存当前学生。
        "discriminator": components["discriminator"].state_dict(),  # 保存判别器。
        "student_optimizer": components["optimizer"].state_dict(),  # 保存student_optimizer字段。
        "discriminator_optimizer": components["d_optimizer"].state_dict(),  # 保存discriminator_optimizer字段。
        "rng_state": runtime.training.capture_rng_state(),  # 保存可恢复的随机状态。
        "loss_committed_sequence": sequence,  # 保存loss_committed_sequence字段。
        "formal_state": dict(state)  # 保存formal_state字段。
    }  # 保存完整动量、RNG、累计loss、验证游标和成本，避免重连后损失曲线断层。

def contract_record(runtime, components, args, parent_sha, initial_sha):  # 为整理版记录新代码身份，不把新文件冒充旧结果的生成源码。
    root = args.project_root  # baseline资产继续从原项目读取。
    own = Path(__file__).resolve().parent  # 当前文件夹就是本组全部五个代码文件所在的位置。
    paths = [own / name for name in ("config.py", "train.py", "infer.py", "plot.py", "evaluate.py")]  # 将本组五个文件全部纳入运行身份。
    paths.extend(root / name for name in (  # 原模型及特征处理依赖也保存源码摘要。
        "fastvoicegrad/train.py",  # 原模型及特征处理依赖也保存源码摘要。
        "fastvoicegrad/objectives.py",  # 原模型及特征处理依赖也保存源码摘要。
        "fastvoicegrad/config.py",  # 原模型及特征处理依赖也保存源码摘要。
        "fastvoicegrad/components.py",  # 原模型及特征处理依赖也保存源码摘要。
        "exp/diagnostics/student_performance_tools_20260906/run_acdd_development.py",  # 原模型及特征处理依赖也保存源码摘要。
        "preprocess/05_extract_bnf.py",  # 原模型及特征处理依赖也保存源码摘要。
        "preprocess/03_extract_mel_22050.py",  # 原模型及特征处理依赖也保存源码摘要。
        "preprocess/06_extract_spk_emb.py"  # 原模型及特征处理依赖也保存源码摘要。
    ))  # 原模型及特征处理依赖也保存源码摘要。
    directories = (  # 与生成历史结果的正式入口保留相同的六类源码目录。
        "fastvoicegrad",  # 学生网络、配置和训练公式。
        "voicegrad",  # 教师网络、扩散日程和数据接口。
        "hifigan/custom",  # 自训声码器的模型与判别器实现。
        "hifigan/official",  # 自训声码器复用的原始网络工具。
        "evaluation",  # 计算正式客观指标的评分工具。
        "external/legacy_compute_bnf/conformer_ppg_model",  # 在线重新提取内容特征的Conformer实现。
    )  # 完成与历史正式入口一致的源码覆盖范围。
    for relative in directories:  # 逐类记录真实参与计算的底层实现。
        directory = root / relative  # 将固定相对目录定位到baseline项目。
        if not directory.is_dir():  # 缺少必要目录时不能生成看似完整的来源合同。
            raise FileNotFoundError(f"缺少必要源码目录：{directory}")  # 显示缺失位置并停止初始化。
        paths.extend(sorted(directory.rglob("*.py")))  # 只遍历小型Python源码，不重复扫描音频和特征。
    source = {str(path.relative_to(root)) if root in path.parents else path.name: sha256_file(path) for path in paths}  # 缺少任何必要源码都会直接报错。
    return {  # 用分行字段清楚展示本次运行到底采用什么设置。
        "kind": settings.KIND,  # 整理版与历史运行使用不同身份。
        "arm": args.arm,  # 保存当前组名。
        "seed": args.seed,  # 保存本次固定随机种子。
        "lambda_cycle": args.lambda_cycle,  # 保存实际参与反向传播的Cycle权重。
        "spec": arm_spec(args.arm),  # 保存共用起点、终点与Cycle开启步。
        "configuration": components["config"].as_dict(),  # 保存模型实际使用的完整配置。
        "initial_model_sha256": initial_sha,  # 保存学生和判别器共同起点的内容身份。
        "parent_sha256": parent_sha,  # 保存完整baseline父断点的内容身份。
        "source_sha256": source,  # 保存本组及baseline必要源码的实际字节指纹。
        "shared_settings": settings.shared_settings(),  # 记录两组应当相同的全部实验设置。
        "development_contract_sha256": sha256_file(settings.DEVELOPMENT),  # 固定开发数据排除协议。
        "runtime_identity": runtime_identity(components["device"]),  # 固定真实GPU与数值后端。
        "batching": {"effective": settings.BATCH_SIZE, "physical": settings.BASE_MICROBATCH_SIZE},  # 清楚区分有效批大小与物理微批。
        "cycle": {  # 保存cycle字段。
            "selected_samples": settings.CYCLE_SAMPLES,  # 保存selected_samples字段。
            "microbatch_samples": settings.CYCLE_MICROBATCH_SIZE,  # 保存microbatch_samples字段。
            "first_leg_stop_gradient": True,  # 保存first_leg_stop_gradient字段。
            "generated_bnf_reextracted": True,  # 保存generated_bnf_reextracted字段。
            "extra_division_by_80": False  # 保存extra_division_by_80字段。
        },  # 保存真实循环路径与归一化定义。
        "provenance": "refactored_from_gpu3_formal_v7; historical_results_remain_bound_to_original_sources",  # 声明重构来源且不改变历史结果归属。
    }  # 返回能与完整断点共同保存的实验合同。

def run_training(argv=None, lease=None):  # 在已取得独占数据盘租约后执行一个可以反复续训的正式训练分支。
    args = parse_args(argv)  # 在加载ML库前拒绝没有明确启动开关的调用。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 在首次CUDA数学操作前固定cuBLAS确定性工作区配置。
    runtime = load_runtime(args.project_root)  # 本地Mac在这里直接拒绝计算。
    torch = runtime.torch  # 后续只操作实验主机中的真实CUDA模型。
    torch.backends.cudnn.benchmark = False  # 禁止根据首次输入计时选择可能不同的卷积算法。
    torch.backends.cudnn.deterministic = True  # 强制CuDNN为相同输入选择确定性实现。
    torch.use_deterministic_algorithms(True, warn_only=False)  # 任一无确定性实现的训练算子必须显式报错而不是悄悄漂移。
    assigned_root = Path("/mnt/lab-data/임서/FastVoiceGrad").resolve()  # 所有项目、结果和断点只能位于用户分配的数据盘目录。
    if not assigned_root.is_dir() or (args.run_root != assigned_root and assigned_root not in args.run_root.parents):  # 永久输出不能落到用户分配目录之外。
        raise RuntimeError("结果根目录必须位于 /mnt/lab-data/임서/FastVoiceGrad。")  # 防止训练产物写进系统盘或他人目录。
    args.run_root.mkdir(parents=True, exist_ok=True)  # 先建立本轮指定的永久结果根目录。
    if shutil.disk_usage(args.project_root).free < 4 * 1024**3 or shutil.disk_usage(args.run_root).free < 4 * 1024**3:  # 两份滚动断点都要求足够剩余空间。
        raise RuntimeError("本地或实验数据盘剩余空间不足4GiB，不能安全保存完整状态。")  # 不等存储写满后才报错。
    torch.set_num_threads(2)  # 与已复现长期FastVoiceGrad保持相同CPU线程设置。
    if not torch.cuda.is_available():  # 正式训练绝不允许回退到CPU。
        raise RuntimeError("当前进程没有可用CUDA GPU。")  # 要求用CUDA_VISIBLE_DEVICES固定到一张2080Ti后重试。
    device = torch.device("cuda:0")  # 每个独立tmux进程只看见一张物理GPU并在逻辑cuda0运行。
    properties = torch.cuda.get_device_properties(device)  # 读取当前可见卡的真实型号和显存。
    if properties.total_memory < 10 * 1024**3:  # 新累计协议仍要求至少10GiB单卡显存。
        raise RuntimeError(f"当前GPU显存不足10GiB：{properties.name}。")  # 不在不足设备上冒险启动正式轨迹。
    runtime.training.seed_everything(args.seed)  # 两个历史Early分支共享同一教师初始化和随机判别器起点。
    components = build_components(runtime, args, device)  # 构建全部冻结依赖和可训练模型，缺文件立即停止。
    train_indices = load_training_indices()  # 精确读取已有52条排除协议及固定数据统计身份。
    development = read_json(settings.DEVELOPMENT)  # 读取已有固定开发集合，绝不临时重新抽样。
    dev_indices = sorted(development["excluded_train_indices"])  # 每pass诊断覆盖全部52条，不把开发集丢回梯度更新。
    parent = args.parent or settings.POST_PARENT_PATH  # Post默认继承已经复现的无Cycle完整100pass状态。
    parent_sha = sha256_file(parent) if args.arm.endswith("post") and parent.is_file() else None  # 内容身份用于两臂绑定同一个父状态。
    if args.arm.endswith("post") and parent_sha is None:  # Post不能用教师重新初始化冒充已有100pass。
        raise FileNotFoundError(f"Post缺少完整100pass父断点：{parent}")  # 直接列出必要资产。
    parent_payload = load_checkpoint(parent) if args.arm.endswith("post") else None  # 只读取Post真正需要的完整训练态。
    if parent_payload is not None:  # 父文件必须是已确认的历史基线或本次base_100正式完整断点。
        if parent_sha != settings.LEGACY_PARENT_SHA256:  # 任意其他同名100pass不能自动混入。
            raise RuntimeError("Post父断点既不是已复现基线，也不是本次正式Base-100。")  # 缺少共同起点证据时停止。
        restore_complete(runtime, components, parent_payload, args, None, is_parent=True)  # 确认两Adam、RNG和权重完整并立即严格恢复。
    initial_sha = {key: state_fingerprint(runtime, components[key]) for key in ("student", "discriminator")}  # 明确两个历史Early初始模型或Post父模型实际字节身份。
    contract = contract_record(runtime, components, args, parent_sha, initial_sha)  # 固定新的实验设置，后续重连不允许静默改变。
    contract["compatibility"] = {  # 将新版STFT局部兼容方法和保持旧数值布局的策略永久纳入合同。
        "conformer_stft_policy": settings.BNF_STFT_COMPAT_POLICY,  # 保存conformer_stft_policy字段。
        "conformer_complex_representation": "last_axis_real_imag_pair",  # 保存conformer_complex_representation字段。
        "global_torch_monkeypatch": False,  # 保存global_torch_monkeypatch字段。
        "legacy_asset_source_mutation": False  # 保存legacy_asset_source_mutation字段。
    }  # 将新版STFT局部兼容方法和保持旧数值布局的策略永久纳入合同。
    output = args.run_root / run_name(args)  # 每臂每seed每lambda使用独立永久结果目录。
    output.mkdir(parents=True, exist_ok=True)  # 首次运行只创建自己的结果目录。
    contract_path = output / "run_contract.json"  # 正式合同位置用于评测和续训身份校验。
    if contract_path.is_file() and read_json(contract_path) != contract:  # 已有实验参数、初始状态或后端发生变化时不覆盖。
        raise RuntimeError("现有正式合同与当前代码或GPU环境不同，请使用原设置续训。")  # 拒绝将不同轨迹拼接成一条曲线。
    atomic_json(contract_path, contract)  # 在第一批梯度更新前发布整个实验身份。
    contract_sha = canonical_sha256(contract)  # 将合同内容摘要放入每个完整断点。
    paired_runtime_identity = dict(contract["runtime_identity"])  # 复制完整运行时身份，避免配对归一化修改各分支实际合同记录。
    paired_runtime_identity.pop("gpu_total_memory", None)  # 同型号显卡可能因固件保留区相差少量可见字节，配对仍严格比较型号、算力和全部数学后端设置。
    shared_identity = {  # 明确传入shared_identity参数。
        "kind": settings.KIND,  # 保存kind字段。
        "seed": args.seed,  # 保存随机种子。
        "parent_sha256": parent_sha,  # 保存parent_sha256字段。
        "initial_model_sha256": initial_sha,  # 保存initial_model_sha256字段。
        "source_sha256": {key: value for key, value in contract["source_sha256"].items() if key != "config.py"},  # 保存source_sha256字段。
        "shared_settings": contract["shared_settings"],  # 保存shared_settings字段。
        "development_contract_sha256": contract["development_contract_sha256"],  # 保存development_contract_sha256字段。
        "runtime_identity": paired_runtime_identity  # 保存runtime_identity字段。
    }  # 两臂必须共用相同模型、数据、源码和除驱动保留显存字节外的后端身份。
    pair_root = args.run_root / "paired_origins"  # 短小配对来源收据位于所有臂的共享永久根目录。
    pair_path = pair_root / f"{'post' if args.arm.endswith('post') else 'early'}_seed_{args.seed}.json"  # lambda变化仍必须继承相同Post父状态。
    pair_lease = None  # 两个不同臂同时启动时，来源登记需要短暂串行确认。
    pair_deadline = time.monotonic() + 30  # 来源短临界区争锁至多等待30秒。
    while pair_lease is None:  # 等另一个初始化进程完成几KiB收据的发布。
        candidate_lease = RunLease(pair_root / (pair_path.stem + "_source_lock"), ttl_seconds=120, heartbeat_seconds=30)  # 独占共同来源收据的检查及首次写入。
        try:  # 来源短期争锁失败不代表整个实验失败。
            pair_lease = candidate_lease.__enter__()  # 通过原子排他创建取得跨实验主机保护。
        except (RuntimeError, FileExistsError):  # 另一个臂正在写共享来源时短暂重试。
            if time.monotonic() >= pair_deadline:  # 已等待足够时间仍争锁失败时明确停止。
                raise RuntimeError("共同来源登记仍被另一个运行时占用，请稍后重跑当前单元格。")  # 不能绕过来源比较强行并行。
            time.sleep(1)  # 只在实验主机用户运行代码中等待一秒，不持续请求外部工具。
    try:  # 在同一个短临界区完成严格比较或第一次写入。
        pair_lease.assert_owned()  # 防止来源保护在读写期间意外丢失。
        if pair_path.is_file():  # 已登记共同来源时不覆盖原证据。
            if read_json(pair_path) != shared_identity:  # 两臂必须真正共享模型、数据、源码和同型号GPU。
                raise RuntimeError("配对实验来源不同；续训须使用同型号GPU、同软件和原配置。")  # 禁止不同数学轨迹混为同一正式对照。
        else:  # 第一个臂只发布一次共同来源。
            atomic_json(pair_path, shared_identity)  # 在排他保护下登记真实模型及环境来源。
    finally:  # 来源比较失败也必须释放自己取得的短租约。
        pair_lease.__exit__(None, None, None)  # 只释放当前初始化进程的来源保护。
    payload, receipt = checkpoint_load(output, map_location="cpu")  # 两槽只恢复已经正式提交且完整可读的训练态。
    spec = arm_spec(args.arm)  # 取得报告中的固定阶段范围。
    target_step = spec["end"]  # 筛选只追加指定短训长度，正式臂终点不变化。
    state = {  # 明确传入state参数。
        "global_step": spec["start"],  # 保存已完成的更新步数。
        "cycle_step": 0,  # 保存已完成的循环更新数。
        "train_pass_sums": {key: 0.0 for key in settings.LOSS_KEYS},  # 保存train_pass_sums字段。
        "train_pass_batches": 0,  # 保存train_pass_batches字段。
        "development_done_passes": [],  # 保存development_done_passes字段。
        "conversion_done_passes": [],  # 保存conversion_done_passes字段。
        "test_done_steps": [],  # 保存test_done_steps字段。
        "endpoint_metrics_done": False,  # 保存endpoint_metrics_done字段。
        "training_seconds": 0.0,  # 保存training_seconds字段。
        "evaluation_seconds": 0.0,  # 保存evaluation_seconds字段。
        "allocated_runtime_seconds": 0.0,  # 保存allocated_runtime_seconds字段。
        "peak_vram_bytes": 0,  # 保存peak_vram_bytes字段。
        "status": "preparing"  # 保存当前完成状态。
    }  # 这些累加器与断点一起保存，避免中途续训丢失当前pass均值。
    if payload is not None:  # 已有正式断点时从最近完整更新继续。
        restored_step, rng_audit = restore_complete(runtime, components, payload, args, contract_sha)  # 恢复两模型、两Adam以及精确随机状态。
        state.update(payload["formal_state"])  # 恢复所有损失、验证及成本游标。
        if restored_step != state["global_step"] or not spec["start"] <= restored_step <= target_step:  # 模型游标和附属状态必须一致。
            raise RuntimeError("恢复模型游标与正式阶段范围不一致。")  # 拒绝错误状态提前宣布完成。
    else:  # 首次运行使用已验证的教师初始状态或共享Post父状态。
        rng_audit = {  # 明确传入rng_audit参数。
            "mode": "shared_full_parent" if parent_payload is not None else "teacher_initialized_step0",  # 保存mode字段。
            "global_step": state["global_step"]  # 保存已完成的更新步数。
        }  # 明确没有使用历史3327学生冒充0步。
    journal = LossJournal(output, committed_sequence=payload.get("loss_committed_sequence", 0) if payload else 0)  # 丢弃断点之后未提交的旧loss尾段，恢复真实曲线游标。
    del payload, parent_payload  # 释放CPU中重复的大型模型与优化器字典。
    training_dataset = runtime.baseline.VoiceGradDataset(  # 明确传入training_dataset参数。
        args.project_root / "data",  # 训练沿用原随机128帧裁剪。
        "train",  # 训练沿用原随机128帧裁剪。
        segment_length=128,  # 明确传入segment_length参数。
        expected_speakers=100,  # 明确传入expected_speakers参数。
        bnf_alignment=components["config"].bnf_alignment  # 明确传入bnf_alignment参数。
    )  # 训练沿用原随机128帧裁剪。
    full_train_dataset = runtime.baseline.VoiceGradDataset(  # 明确传入full_train_dataset参数。
        args.project_root / "data",  # 固定开发片段从完整语音确定性裁剪。
        "train",  # 固定开发片段从完整语音确定性裁剪。
        segment_length=None,  # 明确传入segment_length参数。
        expected_speakers=100,  # 明确传入expected_speakers参数。
        bnf_alignment=components["config"].bnf_alignment  # 明确传入bnf_alignment参数。
    )  # 固定开发片段从完整语音确定性裁剪。
    test_dataset = runtime.baseline.VoiceGradDataset(  # 明确传入test_dataset参数。
        args.project_root / "data",  # 正式分支读固定测试，lambda筛选不读取测试语音或特征。
        "test",  # 正式分支读固定测试，lambda筛选不读取测试语音或特征。
        segment_length=None,  # 明确传入segment_length参数。
        expected_speakers=10,  # 明确传入expected_speakers参数。
        bnf_alignment=components["config"].bnf_alignment  # 明确传入bnf_alignment参数。
    )  # 正式分支读固定测试，lambda筛选不读取测试语音或特征。
    if len(training_dataset) != 39387 or len(dev_indices) != 52 or (test_dataset is not None and len(test_dataset) != 100):  # 数量变化意味着不再是报告中的同一数据协议。
        raise RuntimeError("训练39387、开发52、测试100的固定人口不一致。")  # 禁止自动缩减或重复数据。
    planner = GlobalBatchPlan(train_indices, seed=args.seed)  # 按绝对步重建每pass相同的打乱顺序。
    stop = {"requested": False}  # 手动中止只写轻量标记，不在信号处理器里写大断点。
    for signum in (signal.SIGINT, signal.SIGTERM):  # 用户点击停止和实验主机终止都尽量在完整批次保存。
        signal.signal(signum, lambda *_: stop.update(requested=True))  # 当前D及S完整返回后再停。
    torch.cuda.reset_peak_memory_stats(device)  # 记录本次运行真实训练及评测峰值显存。
    local_temp = args.project_root / "exp/refactored_checkpoint_cache" / run_name(args)  # 序列化先用高速本地盘，永久指针仍只在实验数据盘。
    session_started = time.monotonic()  # 统计当前实验主机实际占用时长。
    last_saved = time.monotonic()  # 从恢复时点开始下一次30分钟保存倒计时。
    last_saved_step = int(state["global_step"])  # 进度条显示真实已经提交的可恢复游标。
    atomic_json(  # 启动后立即让用户看到此前已完成的真实进度。
        output / "resume.json",  # 启动后立即让用户看到此前已完成的真实进度。
        {  # 启动后立即让用户看到此前已完成的真实进度。
            "global_step": state["global_step"],  # 保存已完成的更新步数。
            "cycle_step": state["cycle_step"],  # 保存已完成的循环更新数。
            "rng_audit": rng_audit,  # 保存rng_audit字段。
            "time_unix": time.time()  # 保存time_unix字段。
        }  # 启动后立即让用户看到此前已完成的真实进度。
    )  # 启动后立即让用户看到此前已完成的真实进度。
    session_allocated_before = float(state["allocated_runtime_seconds"])  # 之前各个会话的真实GPU占用时长已保存在断点。
    session_target_step = min(  # 明确传入session_target_step参数。
        target_step,  # 预检会话只推进指定数量完整更新，正式目标本身不变。
        int(state["global_step"]) + args.session_update_limit  # 预检会话只推进指定数量完整更新，正式目标本身不变。
    ) if args.session_update_limit is not None else target_step  # 预检会话只推进指定数量完整更新，正式目标本身不变。
    in_batch = False  # 保存只允许发生在完整批次外部。

    def save_boundary():  # 当前模型、两Adam和loss一起提交，断线时至多回滚到这个真正完整点。
        nonlocal last_saved, last_saved_step  # 更新外层用于时间触发及进度显示的已提交游标。
        if in_batch:  # D已经更新而S还没返回时不能保存。
            raise RuntimeError("拒绝保存半次D/S更新。")  # 保护严格公平的续训状态。
        if lease is not None:  # 正式入口持有独占运行时租约。
            lease.assert_owned()  # 写断点前确认没有被另一个朋友或运行时接管。
        state["allocated_runtime_seconds"] = session_allocated_before + time.monotonic() - session_started  # 从各次会话累积真实GPU占用成本。
        state["peak_vram_bytes"] = max(int(state["peak_vram_bytes"]), int(torch.cuda.max_memory_allocated(device)))  # 跨续训会话保留真实峰值显存。
        saved = checkpoint_store(  # 明确传入saved参数。
            output,  # 本地序列化、实验数据盘双槽完整提交及指针发布。
            checkpoint_payload(runtime, components, args, state, contract_sha, journal),  # 本地序列化、实验数据盘双槽完整提交及指针发布。
            state["global_step"],  # 本地序列化、实验数据盘双槽完整提交及指针发布。
            local_temp_root=local_temp  # 明确传入local_temp_root参数。
        )  # 本地序列化、实验数据盘双槽完整提交及指针发布。
        last_saved = time.monotonic()  # 只有实际提交成功才重置保存倒计时。
        last_saved_step = int(saved["global_step"])  # 保存收据中的实际完成步数供用户查看。
        atomic_json(  # 同步轻量进度和真实可恢复断点身份。
            output / "status.json",  # 同步轻量进度和真实可恢复断点身份。
            {**state, "target_global_step": target_step, "last_saved_global_step": last_saved_step, "latest_checkpoint": saved}  # 同步轻量进度和真实可恢复断点身份。
        )  # 同步轻量进度和真实可恢复断点身份。
        return saved  # 最终评估入口需要明确正式断点位置。

    def evaluate_boundary():  # 重连若恰好落在完整pass边界，继续之前未完成的固定验证。
        step = int(state["global_step"])  # 当前必须来自完整训练态。
        if step % settings.UPDATES_PER_PASS:  # pass中途恢复不用重复每pass诊断。
            return  # 等当前pass最后一批完成后再诊断。
        current_pass = step // settings.UPDATES_PER_PASS  # 使用绝对pass编号，不重新从1编号Post阶段。
        if current_pass not in state["development_done_passes"]:  # 原始共同起点及每个完整pass都各诊断一次。
            started = time.monotonic()  # 单独统计评测实际耗时。
            losses = evaluate_losses(runtime, components, full_train_dataset, dev_indices, "development", args, step)  # 无梯度覆盖52条固定开发语音的全部真实损失。
            journal.append(  # 验证曲线就是固定development，绝不虚构第二个validation集。
                "development",  # 验证曲线就是固定development，绝不虚构第二个validation集。
                step,  # 验证曲线就是固定development，绝不虚构第二个validation集。
                losses,  # 验证曲线就是固定development，绝不虚构第二个validation集。
                completed_pass=current_pass,  # 明确传入completed_pass参数。
                examples=52,  # 明确传入examples参数。
                used_for_gradients=False,  # 明确传入used_for_gradients参数。
                cycle_diagnostic_on_base=True  # 明确传入cycle_diagnostic_on_base参数。
            )  # 验证曲线就是固定development，绝不虚构第二个validation集。
            state["development_done_passes"].append(current_pass)  # 只有整个集合成功才登记完成。
            state["evaluation_seconds"] += time.monotonic() - started  # 训练与验证成本分别保存。
            save_boundary()  # 验证结果与相同模型完整状态一同提交，重连不会丢曲线。
            plot_losses(output)  # 每pass自动刷新训练和development各项真实loss图。
        if (  # 继续核对当前条件，保持原判断顺序。
            args.metrics_endpoint  # 预先固定每5pass的90对开发转换评估。
            and current_pass > 0  # 继续核对当前条件，保持原判断顺序。
            and current_pass % 5 == 0  # 继续核对当前条件，保持原判断顺序。
            and current_pass not in state["conversion_done_passes"]  # 继续核对当前条件，保持原判断顺序。
        ):  # 预先固定每5pass的90对开发转换评估。
            from evaluate import evaluate_conversion_in_memory  # 使用根代理实现的真实客观评分适配器。
            started = time.monotonic()  # 记录转换及评分耗时，不混进训练速度。
            result = evaluate_conversion_in_memory(  # 明确传入result参数。
                runtime,  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
                components,  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
                args.project_root,  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
                output,  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
                "development",  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
                args.seed,  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
                step  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
            )  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
            atomic_json(output / f"development_metrics_pass_{current_pass:04d}.json", result)  # 每个固定时点独立保留评分。
            state["conversion_done_passes"].append(current_pass)  # 评分全成功后才把这个时点标为完成。
            state["evaluation_seconds"] += time.monotonic() - started  # 将开发转换的实际成本纳入报告。
            save_boundary()  # 评分完成游标与同一步模型同时持久化。
        if step in (spec["start"], target_step) and step not in state["test_done_steps"]:  # 正式臂只在共同起点及固定终点测100条测试loss。
            started = time.monotonic()  # 记录固定测试诊断成本。
            losses = evaluate_losses(runtime, components, test_dataset, range(100), "test", args, step)  # 覆盖全部100条真实测试语音，不更新参数也不用于选lambda。
            journal.append("test", step, losses, completed_pass=current_pass, examples=100, used_for_selection=False)  # 测试曲线只标注确实算过的起点和终点。
            state["test_done_steps"].append(step)  # 成功覆盖整个测试集后记录真实完成位置。
            state["evaluation_seconds"] += time.monotonic() - started  # 评测和训练时间分别报告。
            save_boundary()  # 测试记录和模型游标一起提交。
            plot_losses(output)  # 输出可直接检查的训练、验证、测试loss图与CSV。

    from tqdm.auto import tqdm  # 用真实已提交批数初始化百分比进度条。
    total = target_step - spec["start"]  # Post只显示追加50pass的工作量，历史Early显示完整100pass。
    completed = state["global_step"] - spec["start"]  # 已完成比例来自实验数据盘完整断点而不是上次浏览器显示。
    print(  # 显示真实续点。
        f"实验数据盘续点：已完成 {completed:,}/{total:,} batch，{100 * completed / total:.2f}%，global_step={state['global_step']}，Cycle更新={state['cycle_step']}。",  # 显示真实续点。
        flush=True  # 明确传入flush参数。
    )  # 显示真实续点。
    progress = tqdm(  # 明确传入progress参数。
        total=total,  # 明确传入total参数。
        initial=completed,  # 明确传入initial参数。
        desc=run_name(args),  # 明确传入desc参数。
        unit="batch",  # 明确传入unit参数。
        dynamic_ncols=True,  # 明确传入dynamic_ncols参数。
        mininterval=0.5  # 明确传入mininterval参数。
    )  # 每batch前进一次并显示百分比、速度和预计时间。
    try:  # 有异常也只保留此前完整点，不提交损坏的半批状态。
        save_boundary()  # 第一批前持久保存教师0步或Post共同100pass状态。
        evaluate_boundary()  # 完成原共同起点诊断，或继续重连前尚未成功的完整pass验证。
        state["status"] = "running"  # 在数据、权重与共同来源都验证后才标记正式训练。
        for next_step in range(state["global_step"] + 1, session_target_step + 1):  # 从完整断点继续到本次会话边界，已经提交的批次不重复。
            if stop["requested"]:  # 手动停止时可以直接在当前完整点退出。
                break  # 不启动下一次D或S更新。
            batch = runtime.baseline.batch_for(  # 明确传入batch参数。
                training_dataset,  # 原绝对步裁剪及批序规则可从任意pass中间恢复。
                planner.for_step(next_step),  # 原绝对步裁剪及批序规则可从任意pass中间恢复。
                args.seed + 100000 + next_step  # 原绝对步裁剪及批序规则可从任意pass中间恢复。
            )  # 原绝对步裁剪及批序规则可从任意pass中间恢复。
            runtime.training.seed_everything(args.seed + 200000 + next_step)  # 原ACDD输入噪声及教师时刻按绝对步恢复。
            in_batch = True  # 从这一处开始直到两Adam完整返回都禁止保存。
            started = time.monotonic()  # 记录实际单批训练成本。
            losses = formal_train_batch(runtime, components, batch, args, next_step, state["cycle_step"])  # 真实原ACDD加可选24条重新提取BNF的Cycle更新。
            cursors = [optimizer_uniform_step(components[key]) for key in ("optimizer", "d_optimizer")]  # 两个Adam都必须恰好完成当前一次更新。
            if cursors != [next_step, next_step]:  # 任一优化器少更新或多更新都会破坏严格配对。
                raise RuntimeError(f"两Adam游标错误：{cursors}，要求{next_step}。")  # 不把不完整批次标记为完成。
            in_batch = False  # 两个实际游标验证后重新允许完整断点提交。
            state["training_seconds"] += time.monotonic() - started  # 只计训练更新耗时，开发指标时间单独保存。
            state["global_step"] = next_step  # 只有真实完整返回后才推进绝对更新游标。
            state["cycle_step"] = expected_cycle_step(args.arm, next_step)  # Cycle游标按实际门控唯一推导。
            journal.append(  # 逐batch记录九项真实训练loss。
                "train",  # 逐batch记录九项真实训练loss。
                next_step,  # 逐batch记录九项真实训练loss。
                losses,  # 逐batch记录九项真实训练loss。
                completed_pass=next_step // settings.UPDATES_PER_PASS,  # 明确传入completed_pass参数。
                cycle_step=state["cycle_step"],  # 明确传入已完成的循环更新数。
                batch_size=settings.BATCH_SIZE,  # 明确传入batch_size参数。
                cycle_samples=settings.CYCLE_SAMPLES if losses["cycle_raw"] else 0  # 明确传入cycle_samples参数。
            )  # 逐batch记录九项真实训练loss。
            state["train_pass_batches"] += 1  # 保存当前pass已完成批数用于中途恢复。
            for key in settings.LOSS_KEYS:  # 累积每个pass的九项完整训练损失。
                state["train_pass_sums"][key] += losses[key]  # 等完整pass后按实际批数平均。
            progress.update(1)  # 每个真正完成的batch推进一次百分比。
            progress.set_postfix(  # 实时展示损失及此前永久断点游标。
                total=f"{losses['student_total']:.4f}",  # 明确传入total参数。
                cycle=f"{losses['cycle_raw']:.4f}",  # 明确传入cycle参数。
                d=f"{losses['discriminator']:.4f}",  # 明确传入d参数。
                saved_step=last_saved_step,  # 明确传入saved_step参数。
                refresh=False  # 明确传入refresh参数。
            )  # 实时展示损失及此前永久断点游标。
            pass_boundary = next_step % settings.UPDATES_PER_PASS == 0  # 完整pass结束才发布训练均值和固定验证。
            if pass_boundary:  # 恢复中途pass的累加器也必须恰好覆盖1229个完整批。
                if state["train_pass_batches"] != settings.UPDATES_PER_PASS:  # 完整训练臂的每pass只能有1229批。
                    raise RuntimeError("当前pass累计batch数不是1229，拒绝发布不完整训练均值。")  # 缺步曲线不能被误标成完整pass。
                means = {key: value / settings.UPDATES_PER_PASS for key, value in state["train_pass_sums"].items()}  # 每batch都为32条，所以完整pass按批均值即可。
                atomic_json(  # 保存便于组会作图的完整pass读数。
                    output / "train_pass_means" / f"pass_{next_step // settings.UPDATES_PER_PASS:04d}.json",  # 保存便于组会作图的完整pass读数。
                    {"global_step": next_step, "batches": settings.UPDATES_PER_PASS, "losses": means}  # 保存便于组会作图的完整pass读数。
                )  # 保存便于组会作图的完整pass读数。
                state["train_pass_sums"] = {key: 0.0 for key in settings.LOSS_KEYS}  # 下一pass从零累积各项真实loss。
                state["train_pass_batches"] = 0  # 清零当前pass计数并与完整断点一起提交。
            if (  # 继续核对当前条件，保持原判断顺序。
                pass_boundary  # 按步、时间、完整pass及停止请求都可安全触发提交。
                or next_step - last_saved_step >= settings.SAVE_EVERY_UPDATES  # 继续核对当前条件，保持原判断顺序。
                or time.monotonic() - last_saved >= settings.SAVE_EVERY_SECONDS  # 继续核对当前条件，保持原判断顺序。
                or stop["requested"]  # 继续核对当前条件，保持原判断顺序。
            ):  # 按步、时间、完整pass及停止请求都可安全触发提交。
                save_boundary()  # 断线后从此真实游标继续，之后未保存的小段可以重新算。
            if pass_boundary:  # 模型已保存到完整pass边界之后再做验证。
                evaluate_boundary()  # 失败只需重试诊断，已经完成的训练pass无需重做。
        state["status"] = (  # 依据真实完成游标记录退出原因。
            "training_complete" if state["global_step"] == target_step  # 到达完整训练终点才标记训练完成。
            else "session_limit_reached"  # 明确区分用户指定的短会话上限。
            if state["global_step"] == session_target_step and args.session_update_limit is not None  # 只有真正到达该上限才采用此状态。
            else "stopped_at_complete_batch"  # 其余安全停止都停在完整批次边界。
        )  # 不将中途退出误记为完成。
        final_receipt = save_boundary()  # 最后一批或停止点完整持久化。
        plot_losses(output)  # 最终损失图和CSV只使用已提交的真实记录。
        if args.metrics_endpoint and state["global_step"] == target_step and not state["endpoint_metrics_done"]:  # 固定终点才计算8100对测试客观指标，筛选从不读测试集。
            from evaluate import evaluate_conversion_in_memory  # 真实评测接口可从已完成的单对WAV和评分继续。
            started = time.monotonic()  # 记录完整8100对端点评估耗时。
            result = evaluate_conversion_in_memory(  # 明确传入result参数。
                runtime,  # 测全部客观指标及真实CycleMAE和推理速度。
                components,  # 测全部客观指标及真实CycleMAE和推理速度。
                args.project_root,  # 测全部客观指标及真实CycleMAE和推理速度。
                output,  # 测全部客观指标及真实CycleMAE和推理速度。
                "test",  # 测全部客观指标及真实CycleMAE和推理速度。
                args.seed,  # 测全部客观指标及真实CycleMAE和推理速度。
                target_step  # 测全部客观指标及真实CycleMAE和推理速度。
            )  # 测全部客观指标及真实CycleMAE和推理速度。
            state["evaluation_seconds"] += time.monotonic() - started  # 固定测试转换成本与训练分开统计。
            state["endpoint_metrics_done"] = True  # 只有转换和全部评分成功才登记完成。
            state["status"] = "complete"  # 正式实验包含固定端点评分，完成训练本身不等于完成实验。
            atomic_json(output / "endpoint_metrics.json", result)  # 永久保存实际测试客观结果。
            final_receipt = save_boundary()  # 把真实评测完成游标保存到同一个完整模型断点。
        atomic_json(  # 如实记录训练、验证、测试及成本最终状态。
            output / "completion.json",  # 如实记录训练、验证、测试及成本最终状态。
            {**state, "target_global_step": target_step, "checkpoint": final_receipt, "screening_uses_test": False}  # 如实记录训练、验证、测试及成本最终状态。
        )  # 如实记录训练、验证、测试及成本最终状态。
        print(f"当前分支状态：{state['status']}；永久结果：{output}", flush=True)  # 用户直接看到实验真正跑到哪一步及保存位置。
    except BaseException as error:  # 网络挂载、显存、模型和评分异常都保留最近完整断点。
        atomic_json(  # 明确当前批是否只完成了一部分，不伪造成功。
            output / "failure.json",  # 明确当前批是否只完成了一部分，不伪造成功。
            {  # 明确当前批是否只完成了一部分，不伪造成功。
                "error": repr(error),  # 保存error字段。
                "last_committed_global_step": last_saved_step,  # 保存last_committed_global_step字段。
                "current_global_step": state["global_step"],  # 保存current_global_step字段。
                "partial_batch_not_saved": in_batch,  # 保存partial_batch_not_saved字段。
                "time_unix": time.time()  # 保存time_unix字段。
            }  # 明确当前批是否只完成了一部分，不伪造成功。
        )  # 明确当前批是否只完成了一部分，不伪造成功。
        raise  # 原样显示完整错误，让再次运行继续已提交状态。
    finally:  # 无论完整结束还是错误都关闭进度条显示。
        progress.close()  # 释放终端进度条资源。


class BorrowedRunLease:  # 使用启动器已经取得的同一任务租约，避免同一个进程链重复争锁。
    def __init__(self, run_dir, token):  # 绑定实际输出目录和启动器传入的随机所有权令牌。
        self.path = Path(run_dir) / "run_lease.json"  # 必须检查当前分支真实目录，不能借用其他任务租约。
        self.token = str(token)  # 只在进程内部保存令牌，不把它打印到用户日志。

    def assert_owned(self):  # 由父启动器刷新心跳，子工作器每次提交前只核对所有权。
        if not self.path.is_file():  # 父租约不存在时不能假定仍有写入资格。
            raise RuntimeError("启动器传入了租约令牌，但当前实验租约文件不存在。")  # 不重新取得另一个租约掩盖父进程故障。
        value = json.loads(self.path.read_text(encoding="utf-8"))  # 回读当前真实持久目录所有者。
        if (  # 继续核对当前条件，保持原判断顺序。
            value.get("token") != self.token  # 令牌及心跳必须同时有效。
            or time.time() - float(value.get("heartbeat_unix", 0)) > int(value.get("ttl_seconds", 1800))  # 继续核对当前条件，保持原判断顺序。
        ):  # 令牌及心跳必须同时有效。
            raise RuntimeError("启动器租约已失效，停止提交当前训练结果。")  # 防止两个不同实验主机进程覆盖同一结果。


def main(argv=None):  # 独占当前实际输出目录后运行正式工作器。
    args = parse_args(argv)  # 先确定独立分支、种子和候选权重的准确目录。
    assigned_root = Path("/mnt/lab-data/임서/FastVoiceGrad").resolve()  # 获取租约前固定用户的永久数据盘边界。
    if (  # 继续核对当前条件，保持原判断顺序。
        sys.platform != "linux"  # Mac和分配目录外路径都不能启动。
        or not assigned_root.is_dir()  # 继续核对当前条件，保持原判断顺序。
        or (args.run_root != assigned_root and assigned_root not in args.run_root.parents)  # 继续核对当前条件，保持原判断顺序。
    ):  # Mac和分配目录外路径都不能启动。
        raise RuntimeError("正式启动必须在Linux主机并把结果写入 /mnt/lab-data/임서/FastVoiceGrad。")  # 文件传输和静态检查本身不会创建训练租约。
    token = os.environ.get("VOICE_RESEARCH_LEASE_TOKEN")  # 启动器可以通过环境明确授予已取得的同任务租约。
    if token:  # Notebook父进程已经保护本分支时，不再次创建同目录租约。
        lease = BorrowedRunLease(args.run_root / run_name(args), token)  # 子进程只借用父进程已经取得的所有权。
        lease.assert_owned()  # 第一批之前核对父令牌确实属于当前实际输出目录。
        run_training(argv, lease=lease)  # 全部保存仍核对所有权，心跳由启动器继续刷新。
        return  # 子进程不释放父进程拥有的租约，由父启动器统一清理。
    with RunLease(args.run_root / run_name(args)) as lease:  # 同一训练目录只能有一个朋友或实验主机运行时写入。
        run_training(argv, lease=lease)  # 训练、验证和断点都在同一独占所有权下进行。

if __name__ == "__main__":  # 导入只定义功能，只有明确运行脚本才进入命令行。
    main()  # 本组训练从同目录config读取设置并要求显式执行开关。
