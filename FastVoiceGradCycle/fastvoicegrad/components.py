# 装配学生蒸馏使用的冻结教师、冻结声码器和可训练波形判别器。
"""Load and validate the frozen teacher/vocoder and trainable discriminator."""

# 本文件负责三类网络组件：冻结的 VoiceGrad 教师、冻结的修改版 HiFi-GAN
# 声码器，以及需要和学生一起训练的波形判别器。加载时会核对文件哈希、训练步数、
# 配置和参数归属，任何一处不符合本次严格复现实验的记录都会立即停止。

# hashlib 用于计算 SHA-256；哈希相同表示文件字节内容与审计记录完全一致。
import hashlib
# json 用于读取、整理运行证明中的 JSON 数据。
import json
# Path 提供不依赖字符串拼接的文件路径操作。
from pathlib import Path
# 这些类型只帮助读者和检查工具理解各函数接收、返回什么，不会改变计算。
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

# NumPy 用来读取预处理阶段保存的 mel 均值和标准差 .npy 文件。
import numpy as np
# PyTorch 负责载入权重、保存 Tensor，以及把网络放到指定设备。
import torch
# nn 是 PyTorch 神经网络接口；这里主要使用 nn.Module 作为统一网络类型。
import torch.nn as nn

# 导入项目中修改版 HiFi-GAN 的多分辨率判别器 MRD。
from hifigan.custom.mrd import MultiResolutionDiscriminator
# 从已经审计的 HiFi-GAN 训练实现中复用架构名称、MRD 配置和官方模块导入方法。
from hifigan.custom.train import (
    # 固定的架构名称，用于阻止误载公开版或其他版本的声码器。
    ARCHITECTURE_NAME,
    # 根据配置生成 MRD 的三个 STFT 分辨率组合。
    build_mrd_resolutions,
    # 从项目锁定位置导入官方 HiFi-GAN 的 Generator 和 MPD 类。
    import_official_hifigan,
    # 读取并验证声码器配置 JSON。
    load_config as load_hifigan_config,
)
# 导入最终声码器检查点验证所需的规范化 JSON 与哈希工具。
from hifigan.custom.validate_checkpoint import (
    # 把 JSON 对象转成键顺序固定的字符串，便于稳定比较。
    canonical_json,
    # 对规范化 JSON 内容计算 SHA-256。
    canonical_sha256,
    # 读取并确认文件顶层是 JSON 对象。
    load_json_object,
    # 重新检查声码器权重、配置、步数和实现指纹。
    validate_checkpoint as validate_custom_vocoder_checkpoint,
)
# PAPER_CONFIG 是教师训练时锁定的论文参数，加载教师时要逐项比对。
from voicegrad.config import PAPER_CONFIG
# build_model 用相同构造方法创建教师和学生，保证二者网络结构一致。
from voicegrad.train import build_model

# 学生训练配置集中保存论文已公开设定与本实验明确登记的补充设定。
from .config import FastVoiceGradStudentConfig


# 逐块读取文件并计算 SHA-256，用于确认大检查点没有被替换或损坏。
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # 先要求路径确实指向普通文件；不存在或是文件夹都不能继续。
    if not path.is_file():
        # 抛出具体路径，便于马上定位缺失的权重或证明文件。
        raise FileNotFoundError(path)
    # 创建一个尚未写入任何内容的 SHA-256 计算器。
    digest = hashlib.sha256()
    # 以二进制只读方式打开；哈希必须基于原始字节，不能按文字解码。
    with path.open("rb") as handle:
        # 大模型文件不能一次全读入内存，所以循环按块处理。
        while True:
            # 默认每次读取 1 MiB；改块大小只影响速度和内存，不改变最终哈希。
            block = handle.read(chunk_size)
            # 读到空字节串表示已经到达文件末尾。
            if not block:
                # 结束循环，所有文件字节都已送入哈希计算器。
                break
            # 把当前数据块加入累计哈希状态。
            digest.update(block)
    # 返回 64 个十六进制字符，供后续与登记值做精确比较。
    return digest.hexdigest()


# 兼容 PyTorch 1.13 至新版本，读取本项目可信的字典格式检查点。
def load_torch_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    # 兼容本项目不同 PyTorch 版本读取受信任的本地检查点。
    """Load a trusted project checkpoint on PyTorch 1.13 through newer releases."""

    # 加载前确认文件存在，避免 torch.load 给出不直观的底层错误。
    if not path.is_file():
        # 错误信息包含完整目标路径，方便核对脚本参数。
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # 新版 PyTorch 支持显式 weights_only；这里的可信检查点还含 epoch 等元数据。
    try:
        # 指定 False 才能读取完整字典；map_location 通常先放 CPU，减少误占 GPU 显存。
        value = torch.load(path, map_location=map_location, weights_only=False)
    # 旧版 PyTorch 不认识 weights_only 参数，会抛 TypeError。
    except TypeError:
        # 在旧版环境退回等价的传统调用，保持跨版本兼容。
        value = torch.load(path, map_location=map_location)
    # 本项目规定检查点顶层必须是字典，不能是裸模型对象或单个 Tensor。
    if not isinstance(value, dict):
        # 类型不符说明文件格式与本实验约定不同，拒绝继续猜测。
        raise TypeError(f"Checkpoint must be a dictionary: {path}")
    # 通过类型检查后返回完整检查点元数据和权重。
    return value


# 一次性开启或关闭一个网络全部参数的梯度计算。
def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    # 遍历网络中所有可学习参数 Tensor。
    for parameter in module.parameters():
        # 原地设置 requires_grad；False 表示冻结，True 表示允许反向传播更新。
        parameter.requires_grad_(enabled)


# 验证指定网络已经完全冻结，防止教师或声码器被优化器意外修改。
def assert_frozen(module: nn.Module, name: str) -> None:
    # 收集仍然允许求梯度的参数名称；正常冻结时列表必须为空。
    trainable = [key for key, value in module.named_parameters() if value.requires_grad]
    # 只要找到任意一个可训练参数，就说明冻结操作没有完整生效。
    if trainable:
        # 报错仅展示前三个名称，既便于定位也避免超长日志。
        raise RuntimeError(f"{name} must be frozen; trainable parameters include {trainable[:3]}")


# 验证指定网络至少有一个可训练参数，防止学生或判别器根本无法学习。
def assert_trainable(module: nn.Module, name: str) -> None:
    # any 在找到第一个 requires_grad=True 的参数时即可结束遍历。
    if not any(parameter.requires_grad for parameter in module.parameters()):
        # 没有可训练参数意味着 optimizer.step 不会改变模型，因此立即停止。
        raise RuntimeError(f"{name} has no trainable parameters")


# 从唯一允许的 VoiceGrad epoch-500 权重同时构建冻结教师和可训练学生。
def load_teacher_and_student(
    # checkpoint_path 指向经过审计且不可变的 epoch_0500.pt。
    checkpoint_path: Path,
    # config 保存预期哈希、epoch、global step 和数据协议等锁定值。
    config: FastVoiceGradStudentConfig,
    # device 是实际运行设备；正式训练应为出租服务器的 CUDA A100。
    device: torch.device,
# 结束 load_teacher_and_student 的参数声明；返回类型为 Tuple[nn.Module, nn.Module, Dict[str, Any]]，具体内容由下方实现给出。
) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    # 用同一个 VCTK 第 500 轮教师检查点分别初始化教师和学生，参数对象保持独立。
    """Strictly initialize both networks from the VCTK epoch-500 teacher."""

    # 先根据文件当前字节重新计算 SHA-256，不能只相信文件名。
    checkpoint_sha256 = sha256_file(checkpoint_path)
    # 与配置中事先登记的教师权重哈希逐字符比较。
    if checkpoint_sha256 != config.expected_teacher_checkpoint_sha256:
        # 不一致说明权重被换过、传输损坏或选错文件，正式复现必须停止。
        raise RuntimeError(
            # 第一部分说明错误性质。
            "Teacher checkpoint byte identity mismatch: "
            # 第二部分记录实际计算得到的哈希。
            f"got {checkpoint_sha256}, expected "
            # 第三部分记录严格期望的哈希，方便人工核对。
            f"{config.expected_teacher_checkpoint_sha256}"
        )
    # 哈希通过后读取检查点字典；默认先放在 CPU。
    checkpoint = load_torch_checkpoint(checkpoint_path)
    # 格式版本 5 是这次审计后的 VoiceGrad 权重格式。
    if checkpoint.get("format_version") != 5:
        # 其他版本可能字段或语义不同，不能按当前逻辑解释。
        raise RuntimeError(
            # repr 形式可清楚显示缺失时的 None 或异常类型。
            f"Teacher checkpoint format mismatch: {checkpoint.get('format_version')!r}"
        )
    # 论文复现实验明确使用训练到 500 epoch 的教师，不接受其他轮数。
    if int(checkpoint.get("epoch", -1)) != config.teacher_epoch:
        # 即使文件名写着 0500，也以文件内部可信字段再次确认。
        raise RuntimeError(
            # 告知当前学生实验要求的教师 epoch。
            f"FastVoiceGrad requires the {config.teacher_epoch}-epoch VoiceGrad teacher; "
            # 同时打印检查点自己报告的 epoch。
            f"checkpoint reports epoch={checkpoint.get('epoch')!r}"
        )
    # 正式初始化必须使用只含固定模型权重的归档版本，而不是可继续训练的临时状态。
    if checkpoint.get("checkpoint_type") != "weights_only":
        # 类型不符时拒绝加载，避免误把优化器状态或未完成快照当最终教师。
        raise RuntimeError(
            # 给出应使用的明确文件名。
            "Use the immutable epoch_0500.pt weights-only teacher checkpoint, "
            # 显示实际类型，帮助定位选错了哪个文件。
            f"not checkpoint_type={checkpoint.get('checkpoint_type')!r}"
        )
    # global_step 是实际执行的优化器更新次数，比只看 epoch 更精确。
    if int(checkpoint.get("global_step", -1)) != config.expected_teacher_global_step:
        # 步数不符说明即便 epoch 相同，批次数量或训练进度仍可能不一致。
        raise RuntimeError(
            # 说明这是优化器步数游标问题。
            "Teacher optimizer-step cursor mismatch: "
            # 打印实际值。
            f"got {checkpoint.get('global_step')!r}, expected "
            # 打印审计记录要求的值，本实验为 615000。
            f"{config.expected_teacher_global_step}"
        )
    # 比对教师检查点内保存的整套论文参数，确保不是其他 VoiceGrad 实验的权重。
    if checkpoint.get("paper_config") != PAPER_CONFIG.as_dict():
        # 任何公开设定差异都会使教师监督与本次复现不再同源。
        raise RuntimeError(
            # 明确指出差异来自 paper_config。
            "Teacher paper_config differs from the audited VoiceGrad-500 configuration"
        )
    # 从检查点中取出真正的模型 state_dict，即“参数名 → Tensor”映射。
    state = checkpoint.get("model")
    # 权重必须是 Mapping 且非空；空字典无法构建有效教师或学生。
    if not isinstance(state, Mapping) or not state:
        # 缺少权重时给出明确字段名，便于检查归档格式。
        raise KeyError("Teacher checkpoint has no non-empty 'model' state dict")
    # 对每个参数逐一检查类型和数值有效性。
    for key, value in state.items():
        # state_dict 中的每个值都应是 Tensor，不允许混入任意对象。
        if not isinstance(value, torch.Tensor):
            # 报出具体参数名，便于确定损坏位置。
            raise TypeError(f"Teacher state {key!r} is not a tensor")
        # 只有浮点或复数 Tensor 才可能含 NaN/Inf；整数缓冲区无需做该检查。
        if (value.is_floating_point() or value.is_complex()) and not bool(
            # torch.isfinite 要求每个元素都既不是 NaN 也不是正负无穷。
            torch.isfinite(value).all()
        ):
            # 非有限权重会污染所有后续输出，因此不允许开始学生训练。
            raise FloatingPointError(f"Teacher state {key!r} contains NaN or Inf")

    # 取出教师训练运行参数；旧或错误检查点缺失时先得到空字典，随后会比较失败。
    run_config = checkpoint.get("run_config", {})
    # 明确列出必须与已审计教师实验一致的运行级字段。
    expected_run_values = {
        # 教师是在出租服务器规范项目根目录中训练与归档的。
        "project_root": "/root/FsatVoiceGrad",
        # 总训练轮数来自论文配置。
        "epochs": PAPER_CONFIG.epochs,
        # 全局 batch size 来自论文配置。
        "batch_size": PAPER_CONFIG.batch_size,
        # 数据读取使用 4 个 worker，这是公开设置缺失后登记的复现选择。
        "num_workers": 4,
        # 固定随机种子保证可重复运行。
        "seed": PAPER_CONFIG.seed,
        # 教师测试损失的记录间隔沿用已审计配置。
        "loss_every": PAPER_CONFIG.paper_epochs,
        # 每 100 epoch 保存一个归档检查点。
        "save_every": 100,
        # 数据拆分必须是 VCTK text-unseen 协议。
        "expected_protocol": config.expected_protocol,
        # BNF 到 mel 的对齐方法必须与学生配置相同。
        "bnf_alignment": config.bnf_alignment,
        # BNF 的时间下采样倍率也必须一致。
        "bnf_downsampling": config.bnf_downsampling,
    }
    # 逐项比较实际教师 run_config 与上面的期望值。
    for key, expected in expected_run_values.items():
        # get 在字段缺失时返回 None，因此缺字段也会被当作不一致。
        if run_config.get(key) != expected:
            # 任一差异都拒绝继续，避免用不明设定的教师初始化学生。
            raise RuntimeError(
                # 指明发生差异的字段名称。
                f"Teacher run_config mismatch for {key}: "
                # 显示检查点里的实际值。
                f"got {run_config.get(key)!r}, expected {expected!r}"
            )

    # 按相同 BNF 下采样配置新建一份教师结构。
    teacher = build_model(config.bnf_downsampling)
    # 再新建一份完全相同的学生结构；学生不是共享教师参数对象。
    student = build_model(config.bnf_downsampling)
    # strict=True 要求教师权重名称和形状全部精确匹配，不能漏层或多层。
    teacher.load_state_dict(state, strict=True)
    # 学生从完全相同的 epoch-500 权重开始，这正是论文的初始化要求。
    student.load_state_dict(state, strict=True)

    # 即使未来构造函数或加载逻辑改变，也要“默认拒绝”任何教师/学生初始化差异。
    for key, teacher_value in teacher.state_dict().items():
        # torch.equal 要求形状、类型以及每个元素都完全相同，不使用近似比较。
        if not torch.equal(teacher_value, student.state_dict()[key]):
            # 报出第一个不一致参数名，阻止错误初始化进入正式训练。
            raise RuntimeError(f"Student initialization differs from teacher at {key}")

    # 把教师移到指定设备并设为 eval；eval 固定归一化/随机层的推理行为。
    teacher.to(device).eval()
    # 把学生移到同一设备并设为 train，允许执行训练模式前向。
    student.to(device).train()
    # 关闭教师全部参数梯度，保证其只生成监督目标。
    set_requires_grad(teacher, False)
    # 开启学生全部参数梯度，允许优化器更新。
    set_requires_grad(student, True)
    # 再检查一次教师确实没有遗漏的可训练参数。
    assert_frozen(teacher, "teacher")
    # 再检查一次学生不是意外全冻结。
    assert_trainable(student, "student")
    # 返回两个网络和原始检查点元数据，供训练与运行合同继续使用。
    return teacher, student, checkpoint


# 对已读入的最终 HiFi-GAN 检查点做内容级验证，防止仅凭文件名误判。
def _validate_vocoder_checkpoint(
    # checkpoint 是权重文件反序列化后的顶层映射。
    checkpoint: Mapping[str, Any],
    # checkpoint_path 只用于在报错中指出实际文件位置。
    checkpoint_path: Path,
    # hifigan_config 是当前选定 JSON 配置的完整内容。
    hifigan_config: Mapping[str, Any],
    # config 保存学生实验锁定的架构名与最终训练步数。
    config: FastVoiceGradStudentConfig,
# 结束 _validate_vocoder_checkpoint 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # 首先确认检查点声明的是论文要求的修改版 HiFi-GAN V1 架构。
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        # 架构名不同可能意味着误用了公开 HiFi-GAN，因此正式运行立即停止。
        raise RuntimeError(
            # 显示文件中实际声明的架构。
            f"Vocoder architecture mismatch: got {checkpoint.get('architecture')!r}, "
            # 显示代码审计后允许的架构名称。
            f"expected {ARCHITECTURE_NAME!r}"
        )
    # 再确认学生配置和声码器实现自身对架构名称的约定一致。
    if ARCHITECTURE_NAME != config.vocoder_architecture:
        # 两处登记不一致说明运行合同内部自相矛盾，不能继续。
        raise RuntimeError("Student config and audited HiFi-GAN architecture disagree")
    # 正式学生训练只允许最终 2,500,000 步声码器，不使用 50 万或 100 万中间版本。
    if int(checkpoint.get("global_step", -1)) != config.expected_vocoder_step:
        # global_step 不符时明确打印期望、实际步数与文件路径。
        raise RuntimeError(
            # 说明学生训练已锁定最终声码器。
            "The formal student run is locked to the final custom vocoder: "
            # 显示配置要求的最终步数。
            f"expected step {config.expected_vocoder_step}, got "
            # 显示文件内部步数和来源路径，方便排查选错检查点。
            f"{checkpoint.get('global_step')!r} from {checkpoint_path}"
        )
    # 格式版本 2 对应本项目修改版声码器的最终归档结构。
    if checkpoint.get("format_version") != 2:
        # 其他格式可能使用不同键名或字段语义，所以拒绝猜测读取。
        raise RuntimeError(
            # 将实际格式版本完整打印出来。
            f"Vocoder checkpoint format mismatch: {checkpoint.get('format_version')!r}"
        )
    # 最终文件必须只保存生成器权重，不能误用含训练中优化器状态的快照。
    if checkpoint.get("checkpoint_type") != "generator_weights_only":
        # 类型异常时把实际值写入错误信息。
        raise RuntimeError(
            # 只有 generator_weights_only 符合冻结声码器加载契约。
            f"Unsupported vocoder checkpoint type {checkpoint.get('checkpoint_type')!r}"
        )
    # 取出声码器训练时随检查点保存的配置副本。
    saved_config = checkpoint.get("config")
    # 要求副本与现在指定的 JSON 每个键值完全一致。
    if saved_config != dict(hifigan_config):
        # 配置不同会改变生成器结构或音频参数，不能加载后继续训练学生。
        raise RuntimeError("Vocoder checkpoint config differs from the selected config JSON")
    # 取出生成器 state_dict；学生训练只通过这个生成器把 mel 变成波形。
    state = checkpoint.get("generator")
    # 权重必须是非空映射，缺失时不允许构造一个随机声码器。
    if not isinstance(state, Mapping) or not state:
        # 明确指出缺少 generator 字段。
        raise KeyError("Vocoder checkpoint has no non-empty 'generator' state dict")


# 将磁盘上的最终证明文件与当前权重、配置和代码重新生成的报告逐项核对。
def validate_final_vocoder_attestation(
    # attestation_path 是 250 万步训练完成时写出的不可变证明 JSON。
    attestation_path: Path,
    # checkpoint_path 是本次准备实际加载的生成器权重。
    checkpoint_path: Path,
    # config_path 是本次准备实际使用的 HiFi-GAN 配置。
    config_path: Path,
    # fresh_report 是刚刚根据当前文件字节和实现重新计算的验证报告。
    fresh_report: Mapping[str, Any],
# 结束 validate_final_vocoder_attestation 的参数声明；返回类型为 Dict[str, Any]，具体内容由下方实现给出。
) -> Dict[str, Any]:
    # 把最终声码器证明文件与当前权重、配置和源码字节逐一核对。
    """Validate the immutable final-attestation payload against live bytes/code."""

    # 从 JSON 文件读取最终证明对象。
    attestation = load_json_object(attestation_path)
    # 取出证明文件自己保存的整体负载哈希。
    stored_digest = attestation.get("attestation_payload_sha256")
    # 复制一份字典，避免删除字段时修改原证明对象。
    unsigned = dict(attestation)
    # 计算自哈希时必须排除“哈希自身”字段，否则会形成无法满足的循环定义。
    unsigned.pop("attestation_payload_sha256", None)
    # 对剩余内容做规范化 SHA-256，并与保存值比较。
    if stored_digest != canonical_sha256(unsigned):
        # 自哈希不一致表示证明内容被改过或文件损坏。
        raise RuntimeError("Final vocoder attestation self-hash mismatch")
    # 证明文件自哈希已经锁定历史绝对路径；迁移服务器后比较末四级角色路径而不是旧根目录。
    attested_checkpoint = Path(str(attestation.get("checkpoint", "")))  # 读取证明中原 `/root/FsatVoiceGrad` 权重路径。
    live_checkpoint = checkpoint_path.expanduser().resolve()  # 规范化当前 NFS 恢复副本中的权重路径。
    if attested_checkpoint.parts[-4:] != live_checkpoint.parts[-4:]:  # 必须同为 exp/checkpoints/实验目录/固定文件名。
        raise RuntimeError("Final vocoder attestation names a different checkpoint role")  # 相对角色不同仍严格拒绝。
    # 配置也只允许项目根整体迁移，末四级目录和文件名必须与原证明完全一致。
    attested_config = Path(str(attestation.get("config", "")))  # 读取证明中原配置绝对路径。
    live_config = config_path.expanduser().resolve()  # 规范化当前实际配置路径。
    if attested_config.parts[-4:] != live_config.parts[-4:]:  # 必须仍是同一声码器目录内的 config.json。
        raise RuntimeError("Final vocoder attestation names a different config role")  # 防止拿正确权重配另一配置。

    # 这些字段在完成训练后必须保持稳定，并且应能从当前文件重新验证。
    stable_keys = (
        # 总体验证状态，例如 OK。
        "status",
        # 检查点精确字节大小。
        "checkpoint_size_bytes",
        # 检查点原始文件 SHA-256。
        "checkpoint_sha256",
        # 配置 JSON 原始文件 SHA-256。
        "config_file_sha256",
        # 配置内容规范化后的 SHA-256，排除无关空格或键顺序影响。
        "config_canonical_sha256",
        # 权重文件结构版本。
        "format_version",
        # 权重归档类型。
        "checkpoint_type",
        # 修改版声码器架构名称。
        "architecture",
        # 完成时的优化器总更新步数。
        "global_step",
        # 完成时所在 epoch。
        "epoch",
        # 完成时在该 epoch 内的 batch 位置。
        "batch_in_epoch",
        # 按步数规则应有的最终文件名。
        "expected_filename",
        # 所有训练数据清单和统计量的指纹集合。
        "data_fingerprints",
        # 数据指纹集合规范化后的总哈希。
        "data_fingerprints_canonical_sha256",
        # 生成器是否以 strict=True 完整加载成功。
        "generator_strict_load",
        # 生成器结构、参数量和数值检查结果。
        "generator",
        # 与声码器训练有关的实现文件整体哈希。
        "implementation_sha256",
    )
    # 逐项把历史证明与现场重新生成报告比较。
    for key in stable_keys:
        # 先规范化成确定性 JSON 再比较，避免字典键顺序造成假差异。
        if canonical_json(attestation.get(key)) != canonical_json(fresh_report.get(key)):
            # 任一核心字段不同都说明当前环境不再等同于最终审计状态。
            raise RuntimeError(f"Final vocoder attestation mismatch for {key}")
    # 所有检查通过后返回证明内容，供运行合同记录。
    return attestation


# 验证并加载最终修改版 HiFi-GAN 生成器 V，然后把它完全冻结。
def load_frozen_vocoder(
    # project_root 是出租服务器上的权威代码根目录 /root/FsatVoiceGrad。
    project_root: Path,
    # checkpoint_path 是训练到 2,500,000 步的生成器权重文件。
    checkpoint_path: Path,
    # config_path 是与该权重配套的修改版 HiFi-GAN 配置 JSON。
    config_path: Path,
    # attestation_path 是最终权重完成后生成的验证证明。
    attestation_path: Path,
    # config 给出本次学生实验期望的声码器步数和架构。
    config: FastVoiceGradStudentConfig,
    # device 指定网络最终放置的位置，正式训练为出租服务器 A100。
    device: torch.device,
# 结束 load_frozen_vocoder 的参数声明；返回类型为 Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]，具体内容由下方实现给出。
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    # 加载指定自训声码器并冻结其参数；从波形到学生梅尔的梯度仍允许通过。
    """Load the selected modified HiFi-GAN V1 Generator and freeze V only."""

    # 根据当前磁盘上的代码、配置和权重重新做一次完整声码器验证。
    fresh_report = validate_custom_vocoder_checkpoint(
        # 验证器从权威项目根目录寻找锁定实现。
        project_root=project_root,
        # 明确指定要验证的最终检查点。
        checkpoint_path=checkpoint_path,
        # 明确指定配套配置文件。
        config_path=config_path,
        # 要求内部 global_step 精确等于最终步数 2,500,000。
        expected_step=config.expected_vocoder_step,
    )
    # 把刚生成的现场报告与训练完成时保存的最终证明逐项比较。
    validate_final_vocoder_attestation(
        # 历史证明文件。
        attestation_path,
        # 当前检查点文件。
        checkpoint_path,
        # 当前配置文件。
        config_path,
        # 当前重新验证报告。
        fresh_report,
    )
    # 在身份检查通过后读取 HiFi-GAN 配置字典。
    hifigan_config = load_hifigan_config(config_path)
    # 读取检查点完整字典；实际权重暂时仍在 CPU。
    checkpoint = load_torch_checkpoint(checkpoint_path)
    # 对检查点内部架构、步数、类型、配置和权重字段再做内容检查。
    _validate_vocoder_checkpoint(
        # 传入刚读取的检查点。
        checkpoint, checkpoint_path, hifigan_config, config
    )
    # 官方导入函数返回六个对象；这里只需要配置包装 AttrDict 和 Generator 类。
    AttrDict, Generator, _, _, _, _ = import_official_hifigan(project_root)
    # 按已验证配置实例化一个新的 HiFi-GAN 生成器结构。
    vocoder = Generator(AttrDict(hifigan_config))
    # strict=True 完整加载 generator 权重，任何缺失或多余参数都会报错。
    vocoder.load_state_dict(checkpoint["generator"], strict=True)
    # 把声码器移到目标设备并切换到推理模式。
    vocoder.to(device).eval()
    # 关闭声码器所有参数梯度；学生梯度仍可穿过它回到输入 mel。
    set_requires_grad(vocoder, False)
    # 显式确认冻结完整，防止 V 被学生优化器误更新。
    assert_frozen(vocoder, "vocoder V")
    # 保留 weight normalization（权重归一化）的训练参数化形式。推理时移除它虽然
    # 理论输出等价，但会改变计算图；正式蒸馏不做论文未报告的图结构转换。
    # 返回冻结声码器、解析后的配置和原始检查点元数据。
    return vocoder, hifigan_config, checkpoint


# 把论文指定的五个多周期判别器 MPD 与三个多分辨率判别器 MRD 组合起来。
class PaperWaveformDiscriminator(nn.Module):
    # 学生波形判别器由多周期 MPD 和多分辨率 MRD 组成。
    """Modified HiFi-GAN V1 discriminator: MPD plus MRD, with MSD absent."""

    # 初始化判别器结构；权重保持随机初始化，随后与学生同步训练。
    def __init__(self, project_root: Path, hifigan_config: Mapping[str, Any]) -> None:
        # 调用 nn.Module 的初始化，让 PyTorch 正确登记子模块和参数。
        super().__init__()
        # 官方导入结果中第三项是 MultiPeriodDiscriminator，其余此处不用。
        _, _, MultiPeriodDiscriminator, _, _, _ = import_official_hifigan(project_root)
        # 建立五个不同周期的子判别器，用于识别波形周期结构和音高相关伪影。
        self.mpd = MultiPeriodDiscriminator()
        # 建立三个不同 STFT 分辨率的子判别器，用于检查时频纹理。
        self.mrd = MultiResolutionDiscriminator(
            # 分辨率组合由锁定 HiFi-GAN 配置生成，例如不同 FFT/hop/window。
            resolutions=build_mrd_resolutions(dict(hifigan_config)),
            # 读取 LeakyReLU 负半轴斜率；数值会影响弱负激活保留程度。
            lrelu_slope=float(hifigan_config["mrd_lrelu_slope"]),
        )

    # 按论文目标的求和顺序，展开并返回八个独立子判别器。
    def loss_branches(self) -> Tuple[nn.Module, ...]:
        # 按先 MPD、后 MRD 的稳定顺序返回八个分支供逐支计算损失。
        """Return MPD then MRD branches in the paper objective's sum order."""

        # 先放五个 MPD 分支，再接三个 MRD 分支；顺序只用于稳定记录和逐支计算。
        branches = tuple(self.mpd.discriminators) + tuple(self.mrd.discriminators)
        # 论文修改版结构应严格为 5+3=8 支，未来依赖变化时不能静默接受。
        if len(branches) != 8:
            # 返回实际数量，帮助判断 MPD 或 MRD 上游实现是否发生改变。
            raise RuntimeError(
                # 说明预期的两组分支数量和当前总数。
                f"Paper discriminator must expose five MPD and three MRD branches, got {len(branches)}"
            )
        # 返回不可变 tuple，调用者可逐支前向/反向以降低峰值显存。
        return branches

    # 同时让真实与生成波形经过完整的 MPD+MRD，返回分数及逐层特征。
    def forward(
        # 两个输入通常都是 [B, 1, samples]，分别表示真实和生成波形。
        self, real_waveform: torch.Tensor, fake_waveform: torch.Tensor
    # 结束 forward 的参数声明；返回类型为 Tuple[，具体内容由下方实现给出。
    ) -> Tuple[
        # 第一项：八个子判别器对真实波形的输出分数。
        List[torch.Tensor],
        # 第二项：八个子判别器对生成波形的输出分数。
        List[torch.Tensor],
        # 第三项：八个子判别器处理真实波形时各层特征。
        List[List[torch.Tensor]],
        # 第四项：八个子判别器处理生成波形时各层特征。
        List[List[torch.Tensor]],
    ]:
        # 先通过五个不同周期的 MPD 分支。
        mpd_real, mpd_fake, mpd_real_features, mpd_fake_features = self.mpd(
            # MPD 同时接收真实和生成波形，以相同网络分别计算。
            real_waveform, fake_waveform
        )
        # 再通过三个不同频谱分辨率的 MRD 分支。
        mrd_real, mrd_fake, mrd_real_features, mrd_fake_features = self.mrd(
            # MRD 也对同一对真实/生成波形进行比较。
            real_waveform, fake_waveform
        )
        # 把 MPD 与 MRD 结果拼成论文损失函数需要的统一列表。
        return (
            # 真实分数：五个 MPD 在前，三个 MRD 在后。
            list(mpd_real) + list(mrd_real),
            # 生成分数采用完全相同顺序，确保逐支配对。
            list(mpd_fake) + list(mrd_fake),
            # 真实中间特征采用相同顺序。
            list(mpd_real_features) + list(mrd_real_features),
            # 生成中间特征采用相同顺序。
            list(mpd_fake_features) + list(mrd_fake_features),
        )


# 按论文公开结构建立本次学生训练专用的可训练波形判别器 D。
def build_paper_discriminator(
    # project_root 用于定位锁定的官方 HiFi-GAN MPD 实现。
    project_root: Path,
    # hifigan_config 提供 MRD 分辨率和激活斜率等参数。
    hifigan_config: Mapping[str, Any],
    # config 明确登记判别器如何初始化。
    config: FastVoiceGradStudentConfig,
    # device 是正式训练所在的出租服务器 A100。
    device: torch.device,
# 结束 build_paper_discriminator 的参数声明；返回类型为 PaperWaveformDiscriminator，具体内容由下方实现给出。
) -> PaperWaveformDiscriminator:
    # 论文没有发布学生阶段的判别器检查点，因此本复现只实现从头随机初始化。
    if config.discriminator_initialization != "scratch":
        # 如果配置要求其他来源却没有可审计实现，就拒绝自行猜测。
        raise RuntimeError(
            # 第一段说明当前唯一实现的选择。
            "Only the declared scratch discriminator initialization is implemented; "
            # 第二段说明这样限制的论文依据。
            "the FastVoiceGrad paper does not publish a discriminator checkpoint."
        )
    # 构造 MPD+MRD 并一次性移到指定 GPU；MSD 按论文修改明确不加入。
    discriminator = PaperWaveformDiscriminator(project_root, hifigan_config).to(device)
    # 判别器需要在学生阶段学习，所以开启其全部参数梯度。
    set_requires_grad(discriminator, True)
    # 立刻确认确有可训练参数，避免空网络或错误冻结。
    assert_trainable(discriminator, "discriminator D")
    # 返回可交给独立 Adam 优化器的判别器。
    return discriminator


# 从预处理数据中加载 80 个 mel 通道的训练集均值与标准差。
def load_mel_statistics(data_root: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # 均值文件固定放在数据根目录的 stats 子目录。
    mean_path = data_root / "stats" / "mel_mean.npy"
    # 标准差文件与均值文件放在同一目录。
    std_path = data_root / "stats" / "mel_std.npy"
    # 禁止 pickle，只读取纯 NumPy 数组，减少不必要的任意对象反序列化风险。
    mean = np.load(mean_path, allow_pickle=False)
    # 以相同安全方式读取标准差数组。
    std = np.load(std_path, allow_pickle=False)
    # 80 维 log-mel 的每个频率通道都应恰好有一个均值和一个标准差。
    if mean.shape != (80,) or std.shape != (80,):
        # 形状错误说明统计量来自不同特征设置，不能用于本实验反标准化。
        raise ValueError(
            # 同时打印实际均值和标准差形状，便于找错文件。
            f"Expected [80] mel statistics, got mean={mean.shape}, std={std.shape}"
        )
    # 所有值必须有限，并且标准差必须严格大于 0 才能表示有效缩放尺度。
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        # NaN、Inf 或非正标准差都会使还原后的 mel 无意义。
        raise ValueError("Mel statistics contain non-finite values or non-positive std")
    # 转成 float32 Tensor，并从 [80] 整理为 [1,80,1] 以广播 batch 和时间维。
    mean_tensor = torch.from_numpy(mean.astype(np.float32)).view(1, 80, 1).to(device)
    # 标准差做完全相同的类型、形状与设备转换。
    std_tensor = torch.from_numpy(std.astype(np.float32)).view(1, 80, 1).to(device)
    # 返回可直接用于 mel * std + mean 的两个常量 Tensor。
    return mean_tensor, std_tensor


# 逆标准化恢复自然对数 mel 的单位；波形峰值归一化造成的幅度分布差异不会由这一步消除。
def denormalize_mel(
    # normalized_mel 形状为 [B,80,T]，若来自学生则仍带有梯度图。
    normalized_mel: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
# 结束 denormalize_mel 的参数声明；返回类型为 torch.Tensor，具体内容由下方实现给出。
) -> torch.Tensor:
    # 把标准化梅尔恢复为自然对数梅尔，同时保留学生输出的梯度。
    """Return raw natural-log mel while preserving gradients to student output."""

    # 强制要求三维且通道数为 80，防止声码器收到错误频谱格式。
    if normalized_mel.ndim != 3 or normalized_mel.shape[1] != 80:
        # 不在这里自动转置，因为自动猜测维度可能把时间和频率含义弄反。
        raise ValueError("normalized mel must have shape [B, 80, T]")
    # 执行 z-score 的逆变换；普通 Tensor 运算会保留从声码器回到学生的梯度。
    return normalized_mel * std + mean


# 确保教师、学生、声码器和判别器没有意外共享同一个 Parameter 对象。
def assert_disjoint_parameter_sets(named_modules: Sequence[Tuple[str, nn.Module]]) -> None:
    # owners 用 Python 对象 id 记录每个参数第一次出现在哪个网络。
    owners: Dict[int, str] = {}
    # 逐个遍历带名称的网络，名称只用于清楚报错。
    for name, module in named_modules:
        # 再遍历该网络登记的每个 Parameter。
        for parameter in module.parameters():
            # id 表示当前 Python 参数对象的唯一身份，而不是比较数值是否相同。
            identity = id(parameter)
            # 若相同 id 已出现，说明两个网络实际引用同一参数，会互相误更新。
            if identity in owners:
                # 报出第一次和第二次发现该参数的网络名称。
                raise RuntimeError(
                    # 使用已记录 owner 与当前 name 说明共享双方。
                    f"Parameter object is shared by {owners[identity]} and {name}"
                )
            # 首次见到时登记其所有者，供后续网络检查。
            owners[identity] = name


# 把优化器所有参数组中的 Parameter 转成对象 id 集合，便于精确核对归属。
def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set:
    # 集合会自动去重；正常情况下每个参数只应登记一次。
    return {
        # 保存参数对象身份，而不是会随训练变化的参数数值。
        id(parameter)
        # 一个优化器可能分成多组设置不同超参数，所以先遍历所有参数组。
        for group in optimizer.param_groups
        # 再遍历每一组的 params 列表。
        for parameter in group["params"]
    }


# 验证一个优化器只拥有指定网络的全部可训练参数，且不碰其他网络。
def assert_optimizer_ownership(
    # optimizer 是需要检查的学生优化器或判别器优化器。
    optimizer: torch.optim.Optimizer,
    # owner 是该优化器唯一应该更新的网络。
    owner: nn.Module,
    # forbidden 列出绝对不允许被该优化器包含的教师、声码器或另一网络。
    forbidden: Iterable[nn.Module],
    # name 用于在报错信息中标识具体优化器。
    name: str,
# 结束 assert_optimizer_ownership 的参数声明；返回类型为 None，具体内容由下方实现给出。
) -> None:
    # 读取优化器当前实际登记的全部参数对象 id。
    actual = optimizer_parameter_ids(optimizer)
    # 计算 owner 中全部 requires_grad=True 参数应有的 id 集合。
    expected = {id(parameter) for parameter in owner.parameters() if parameter.requires_grad}
    # 集合必须完全相等：少一个参数学不到，多一个参数可能误更新别的网络。
    if actual != expected:
        # 不输出庞大 id 列表，只明确指出归属集合不精确。
        raise RuntimeError(f"{name} does not own exactly the intended trainable parameters")
    # 收集所有禁止网络的全部参数 id；即使其已冻结也必须排除。
    forbidden_ids = {
        # 将每个禁止网络中的每个参数转换为对象 id。
        id(parameter) for module in forbidden for parameter in module.parameters()
    }
    # 计算实际优化器参数与禁止集合是否存在交集。
    if actual.intersection(forbidden_ids):
        # 一旦有交集就立即停止，防止下一次 optimizer.step 修改错误模型。
        raise RuntimeError(f"{name} includes frozen or foreign parameters")
