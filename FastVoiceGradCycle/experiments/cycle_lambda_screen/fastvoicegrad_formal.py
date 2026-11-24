"""在固定 development 协议上执行 FastVoiceGrad Cycle-loss lambda 后续筛选。"""  # 本地只允许静态检查，真正计算由Linux实验主机执行。
import argparse  # 读取分支、权重、路径及用户启动开关。
import hashlib  # 绑定数据、源码、父断点和模型初始化的真实内容。
import importlib.util  # 只在 Colab 中载入原训练和 Conformer 辅助入口。
import json  # 保存可阅读的固定实验合同和进度。
import math  # 检查损失及参数是不是有限数。
import os  # 检查 Drive 真实挂载并原子发布文件。
import random  # 重放原训练的样本打乱和裁剪。
import shutil  # 检查本地缓存和 Drive 剩余空间。
import signal  # 在完整批次结束后响应停止请求。
import sys  # 限制正式计算只能在 Linux Colab 中进行。
import time  # 统计真实耗时和断点保存间隔。
from dataclasses import replace  # 派生正式配置而不改原始代码。
from pathlib import Path  # 组织固定资产和持久结果目录。
from types import SimpleNamespace  # 把只在运行时导入的组件集中交给评测入口。

ARMS = ("base_100", "early_cycle_100", "base_post", "cycle_post")  # 四个分支分别对应完整基线、早期加入和两种后期继续训练。
UPDATES_PER_PASS = 1229  # 39335 条训练语音按 batch32 丢弃尾批后每 pass 有1229次更新。
LOSS_KEYS = ("student_total", "student_adv", "fm_raw", "fm_weighted", "dist_raw", "dist_weighted", "cycle_raw", "cycle_weighted", "discriminator")  # 逐批及验证保存全部九项实际目标。
BNF_SHA256 = "94362c99f408cbe891785537e222c2f1f9119edb11b95107b9faa4c0f0d679a5"  # 固定与现有离线BNF相同的24epoch公开Conformer权重。
BNF_CONFIG_SHA256 = "d8d3df407a9b88500cf62e32a8bdef78068b3a09e9f8bc47904843bef1fff6c4"  # 固定与已归档24epoch权重配套的Conformer结构和音频前端配置。
LEGACY_PARENT_SHA256 = "2ca44269eefac2dd3f8ffa7afe0337f1f3684469aec9fadeff623237cf37b830"  # 指定已经正式复现的100pass完整FastVoiceGrad父状态。
KIND = "fastvoicegrad_cycle_lambda_rescreen_v1_5090_20261112"  # 5090后端重新建立全部候选合同，不接续2080Ti中间状态。
SCREEN_LAMBDAS = (0.0, 0.5, 2.5, 10.0)  # 同预算lambda0续训对照加三个正权重候选，避免把普通追加训练收益误判为Cycle收益。
BNF_STFT_COMPAT_POLICY = "module_local_torch_proxy_stft_return_complex_false_preserves_legacy_real_imag_last_axis"  # 明确只兼容旧Conformer局部STFT接口，不改变全局PyTorch或原资产文件。
LAB_PROJECT_ROOT = Path("/root/FastVoiceGrad/project")  # Matpool系统盘有300GB；5GB网盘不足以容纳固定训练特征。

def arm_spec(arm):  # 返回固定分支的起点、终点和Cycle真正启用的绝对步。
    if arm not in ARMS:  # 只允许已经写入报告的四个实验名称。
        raise ValueError(f"未知正式分支：{arm}")  # 名称错误不自动创建其他实验。
    start = 100 * UPDATES_PER_PASS if arm.endswith("post") else 0  # Post从同一个100pass父状态继续，Early两臂从教师初始化开始。
    end = (150 if arm.endswith("post") else 100) * UPDATES_PER_PASS  # Post额外50pass，完整训练两臂均为100pass。
    cycle_after = 20 * UPDATES_PER_PASS if arm == "early_cycle_100" else 100 * UPDATES_PER_PASS if arm == "cycle_post" else None  # Early在完成20pass后从21pass开启，Post立即开启。
    return {"start": start, "end": end, "cycle_after": cycle_after}  # 返回只依赖分支名称的不可变实验身份。

def expected_cycle_step(arm, step):  # 从绝对已完成更新数推导已完成Cycle更新数。
    after = arm_spec(arm)["cycle_after"]  # 取得Cycle门控边界。
    return 0 if after is None else max(0, int(step) - after)  # Base恒为零，Cycle只累计真正启用后的更新。

def finite_losses(losses):  # 让非法数值不能进入损失日志和断点。
    if not set(LOSS_KEYS) <= set(losses):  # 每次完整更新必须报告全部基础和Cycle组成。
        raise RuntimeError(f"损失字段不完整：{sorted(set(LOSS_KEYS) - set(losses))}")  # 列出缺失字段供排查。
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

def load_runtime(project_root):  # 在Linux实验主机中加载已复现的训练组件和兼容接口。
    root = Path(project_root).resolve()  # 确定真正使用的高速缓存目录。
    if sys.platform != "linux" or root != LAB_PROJECT_ROOT.resolve() or not root.is_dir():  # 只接受分配目录中的稳定入口，Mac仍只负责传输与静态检查。
        raise RuntimeError(f"正式计算必须在Linux CUDA主机使用 {LAB_PROJECT_ROOT} 稳定项目入口。")  # 明确失败原因，绝不在Mac退回CPU。
    for path in (root, root / "exp/diagnostics/student_performance_tools_20260906"):  # 加入原模型和原批次辅助入口位置。
        if str(path) not in sys.path:  # 避免重复插入相同路径。
            sys.path.insert(0, str(path))  # 优先使用已经恢复的原项目。
    import torch  # 只在用户实际启动Linux CUDA计算后载入PyTorch。
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3:  # batch8累计方案要求单卡至少10GiB可用总显存。
        raise RuntimeError("需要至少10GiB显存的真实CUDA GPU。")  # 不把非常慢的CPU运行隐藏为正常实验。
    import torch.nn.functional as functional  # 使用补零与插值等张量函数。
    import torchaudio  # 用真实音频重采样把22.05kHz转换为16kHz。
    from torch.utils.data import default_collate  # 保留原数据字段和批量组织方式。
    import fastvoicegrad.train as training  # 复用原ACDD、VJP显存优化及冻结组件。
    import run_acdd_development as baseline  # 复用已验证的原数据裁剪和模型构建。
    from voicegrad.feature_alignment import align_bnf_to_mel  # 在线BNF继续使用原权重的长度线性对齐。
    stage = load_python(root / "experiments/cycle_stage/train.py", "formal_legacy_stage")  # 复用原完整状态检查和固定开发划分审计。
    return SimpleNamespace(torch=torch, F=functional, torchaudio=torchaudio, default_collate=default_collate, training=training, baseline=baseline, stage=stage, align_bnf_to_mel=align_bnf_to_mel)  # 供训练与最终指标入口共享实际实现。

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
        self.device = device  # 保存当前ColabGPU。
        legacy = Path(root) / "external/legacy_compute_bnf"  # 定位与离线提取相同的公开内容编码器。
        weights = Path(checkpoint) if checkpoint else legacy / "conformer_ppg_model/en_conformer_ctc_att/24epoch.pth"  # 默认只使用已恢复的24epoch权重。
        if not weights.is_file() or runtime.stage.sha256_file(weights) != BNF_SHA256:  # 不允许其他权重被同名文件冒充。
            raise RuntimeError(f"缺少或不匹配的固定Conformer权重：{weights}")  # 在第一批训练前报出缺项。
        helper = load_python(Path(root) / "preprocess/05_extract_bnf.py", "formal_bnf_compat")  # 载入原提取程序中已经锁定的ESPnet兼容实现。
        sys.path.insert(0, str(legacy))  # 让内容编码器包从原项目中导入。
        helper.install_pinned_espnet_embedding_compat(legacy)  # 安装经过原项目审核的同源位置编码接口。
        from conformer_ppg_model.build_ppg_model import build_model  # 复用原结构构建函数，避免旧加载器受新版torch.load默认值影响。
        import conformer_ppg_model.stft as legacy_stft  # 只定位上游Conformer独立STFT模块的局部torch引用。
        legacy_stft.torch = LegacySTFTTorchProxy(runtime.torch)  # 仅替换这个模块的引用，原torch.stft与所有资产文件保持不被修改。
        import yaml  # 按原模型加载器相同方法读取固定结构配置。
        config_path = legacy / "conformer_ppg_model/en_conformer_ctc_att/config.yaml"  # Conformer前端和网络设置必须与离线BNF相同。
        if runtime.stage.sha256_file(config_path) != BNF_CONFIG_SHA256:  # 文件名相同不能证明实际前端及结构相同。
            raise RuntimeError("Conformer配置与固定公开权重不匹配。")  # 在任何训练前拒绝结构漂移。
        self.model = build_model(argparse.Namespace(**yaml.safe_load(config_path.read_text(encoding="utf-8"))))  # 逐项按同一个上游函数建立前端、归一化及编码器。
        pretrained = runtime.stage.load_checkpoint(weights)  # 明确weights_only=False兼容新旧PyTorch，不修改全局torch.load行为。
        encoder_weights = {name: value for name, value in pretrained.items() if "encoder" in name}  # 完全遵循上游只载入Conformer encoder权重的规则。
        complete = self.model.state_dict()  # 音频前端固定buffer由同一结构代码产生。
        if set(encoder_weights) != {name for name in complete if "encoder" in name}:  # 不允许留下随机encoder层冒充冻结预训练编码器。
            raise RuntimeError("Conformer预训练encoder权重字段不完整。")  # 缺层或多层应在正式训练前报告。
        complete.update(encoder_weights)  # 用已验证的预训练encoder替换构造时的临时参数。
        self.model.load_state_dict(complete, strict=True)  # 严格核对全部参数形状和固定buffer。
        self.model.to(device)  # 将完整固定内容编码器放到当前ColabGPU。
        self.model.eval()  # 固定编码器关闭随机训练行为。
        for parameter in self.model.parameters():  # 遍历全部编码器参数。
            parameter.requires_grad_(False)  # 参数冻结，Cycle只通过返回转换更新FastVoiceGrad。

    def extract(self, converted_mel, lengths, vocoder, mean, std):  # 从生成mel获得重新提取且对齐的144维BNF。
        torch = self.runtime.torch  # 使用当前真实PyTorch版本。
        with torch.no_grad():  # 第一程和内容重提取都不保留梯度图。
            waveform = vocoder(self.runtime.training.denormalize_mel(converted_mel.detach(), mean, std)).squeeze(1)  # 用冻结自训HiFi-GAN从自然尺度mel合成22.05kHz波形。
            waves = []  # 保存去掉补齐部分并重采样后的每条有效波形。
            for index, valid_frames in enumerate(lengths.tolist()):  # 每条语音按真实帧数处理，避免把padding当作内容。
                samples = min(waveform.shape[-1], int(valid_frames) * 256)  # HiFi-GAN每个mel帧对应256个音频点。
                waves.append(self.runtime.torchaudio.functional.resample(waveform[index, :samples], 22050, 16000))  # 真实重采样到Conformer使用的16kHz。
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
    if reference.shape != prediction.shape or reference.shape[1] != 80:  # 公式要求两份mel对应且有80个通道。
        raise ValueError("masked MAE要求同形的80通道mel。")  # 不接受广播产生的假误差。
    lengths = lengths.to(reference.device, dtype=torch.long)  # 把有效长度转换为同设备整数。
    if bool((lengths < 1).any()) or bool((lengths > reference.shape[-1]).any()):  # 有效帧数必须落在真实张量范围内。
        raise ValueError("masked MAE的有效长度越界。")  # 防止除零和漏屏蔽padding。
    mask = torch.arange(reference.shape[-1], device=reference.device)[None, :] < lengths[:, None]  # q_it在有效帧为1，右侧补帧为0。
    return ((reference - prediction).abs() * mask[:, None, :]).sum(dim=(1, 2)) / (80 * lengths)  # 已平均80通道及有效帧，绝不再额外除80。

def cycle_per_sample(runtime, components, mel, bnf, source_spk, target_spk, lengths, forward_noise, backward_noise, gradient=False):  # 为同一小批执行真实生成内容重提取的A到B再到A。
    training = runtime.training  # 复用K1/SK950的原单步预测公式。
    with runtime.torch.no_grad():  # 第一程stop-gradient只切断这次图，不冻结共享学生参数的未来能力。
        converted, _ = training.make_student_prediction(components["student"], components["diffusion"], mel, target_spk, bnf, 950, forward_noise)  # 源A内容与目标B身份产生第一次转换。
        generated_bnf = components["bridge"].extract(converted, lengths, components["vocoder"], components["mean"], components["std"])  # 对真实生成波形重提取BNF，约束转换后实际内容。
    with runtime.torch.set_grad_enabled(bool(gradient)):  # 训练保留第二程梯度，验证完全不建反向图。
        returned, _ = training.make_student_prediction(components["student"], components["diffusion"], converted.detach(), source_spk, generated_bnf, 950, backward_noise)  # 使用独立第二程噪声和原A身份返回A。
        return masked_per_sample_mae(runtime, mel, returned, lengths)  # 每条语音独立平均，调用者再对选中的样本平均。

def build_components(runtime, args, device):  # 精确复用已复现模型结构及冻结依赖，派生报告中的新增Cycle参数。
    config, teacher, student, vocoder, discriminator, diffusion, mean, std, optimizer, d_optimizer = runtime.stage.build_components(device, False)  # 原ACDD结构、教师500轮与自训HiFi-GAN250万步完全沿用。
    config = replace(config, epochs=150 if args.arm.endswith("post") else 100, seed=args.seed, cycle_enabled=args.arm in ("early_cycle_100", "cycle_post"), cycle_start_epoch=21 if args.arm == "early_cycle_100" else 101 if args.arm == "cycle_post" else 1, lambda_cycle=args.lambda_cycle, cycle_examples_per_batch=24, cycle_content_mode="reextract_generated_waveform_bnf", cycle_loss_reduction="masked_per_example_mae_then_batch_mean", num_workers=0, learning_rate=5e-5)  # 新增设定登记在配置，不改变旧源码。
    bridge = FrozenBNFBridge(runtime, args.project_root, device, getattr(args, "bnf_checkpoint", None))  # 所有臂均准备同一个冻结内容提取器，用于公平验证Cycle指标。
    return {"config": config, "teacher": teacher, "student": student, "vocoder": vocoder, "discriminator": discriminator, "diffusion": diffusion, "mean": mean, "std": std, "optimizer": optimizer, "d_optimizer": d_optimizer, "bridge": bridge, "device": device}  # 返回训练和评估共享的实际对象。

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
    microbatch_size = 8  # GTX1080Ti及RTX2080Ti显存适配采用物理batch8。
    microbatch_count = config.batch_size // microbatch_size  # 论文逻辑batch32恰好由四个等大微批组成。
    if config.batch_size != 32 or microbatch_count != 4:  # 训练配置改变时不能静默套用错误缩放。
        raise RuntimeError("梯度累计协议要求逻辑batch32由四个物理batch8组成。")  # 保护学习率与每pass更新数的既定含义。
    training.set_requires_grad(components["discriminator"], True)  # 判别器阶段允许D参数接收四个微批的累计梯度。
    components["d_optimizer"].zero_grad(set_to_none=True)  # 一个逻辑批开始时只清一次D梯度。
    gradient_hooks = [parameter.register_hook(lambda gradient: gradient / microbatch_count) for parameter in components["discriminator"].parameters() if parameter.requires_grad]  # 每次微批反传乘四分之一，使总梯度等于32条样本均值。
    discriminator_value = 0.0  # 日志同样累计四个微批损失的算术平均。
    try:  # 无论某个微批是否失败都必须移除临时梯度缩放钩子。
        for start in range(0, config.batch_size, microbatch_size):  # 按原顺序依次处理0到7、8到15、16到23、24到31。
            stop = start + microbatch_size  # 当前物理微批固定包含八条语音。
            with torch.no_grad():  # D阶段不为学生和冻结声码器保存计算图。
                prediction, _ = training.make_student_prediction(components["student"], components["diffusion"], x_zero[start:stop], speaker[start:stop], bnf[start:stop], config.student_start_step, source_noise[start:stop])  # 使用完整逻辑批预抽取噪声的对应切片。
                real_waveform = components["vocoder"](training.denormalize_mel(x_zero[start:stop], components["mean"], components["std"]))  # 当前八条真实mel经冻结声码器生成真实波形。
                fake_waveform = components["vocoder"](training.denormalize_mel(prediction, components["mean"], components["std"]))  # 当前八条学生mel经同一冻结声码器生成假波形。
            training._assert_finite("accumulated D real waveform", real_waveform)  # 坏波形不得进入累计D梯度。
            training._assert_finite("accumulated D fake waveform", fake_waveform)  # 学生输出出现坏数时立即停止当前逻辑更新。
            micro_discriminator_loss = training.serial_discriminator_backward(components["discriminator"], real_waveform, fake_waveform)  # 八个判别器分支对当前八条样本反传。
            discriminator_value += float(micro_discriminator_loss.detach()) / microbatch_count  # 四个微批均值对应完整32条损失均值。
            del prediction, real_waveform, fake_waveform, micro_discriminator_loss  # 立即释放本微批D阶段张量以控制峰值显存。
    finally:  # 钩子只属于本次D累计阶段。
        for hook in gradient_hooks:  # 遍历所有D参数的临时缩放钩子。
            hook.remove()  # 后续学生阶段和下一逻辑批不再自动缩放梯度。
    discriminator_gradients = [parameter.grad for parameter in components["discriminator"].parameters() if parameter.grad is not None]  # 收集四个微批累计后的D梯度。
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
            proxy_prediction, _ = training.make_student_prediction(components["student"], components["diffusion"], x_zero[start:stop], speaker[start:stop], bnf[start:stop], config.student_start_step, source_noise[start:stop])  # 学生参数尚未更新，复用对应噪声切片。
            real_proxy = components["vocoder"](training.denormalize_mel(x_zero[start:stop], components["mean"], components["std"]))  # 为当前微批生成不需要梯度的真实波形代理。
            fake_proxy = components["vocoder"](training.denormalize_mel(proxy_prediction, components["mean"], components["std"]))  # 为当前微批生成不需要梯度的学生波形代理。
        adversarial_loss, feature_matching_loss, waveform_gradient = run_known_reflection_backward(torch, lambda: training.serial_student_discriminator_gradient(components["discriminator"], real_proxy, fake_proxy, config.lambda_fm))  # 仅冻结D的已知反射填充CUDA反向在最小范围内采用原论文可用实现。
        prediction, _ = training.make_student_prediction(components["student"], components["diffusion"], x_zero[start:stop], speaker[start:stop], bnf[start:stop], config.student_start_step, source_noise[start:stop])  # 重新建立学生计算图并复用完全相同的输入。
        fake_waveform = components["vocoder"](training.denormalize_mel(prediction, components["mean"], components["std"]))  # 建立波形到学生的真实反向路径。
        if not torch.equal(fake_waveform.detach(), fake_proxy):  # 确定性后端必须让VJP代理点和真实反向点逐位完全相同。
            maximum_difference = float(torch.max(torch.abs(fake_waveform.detach() - fake_proxy)))  # 报告最大差值以定位随机层或数值漂移。
            raise RuntimeError(f"物理微批波形重算不一致：max_abs_diff={maximum_difference:.9g}")  # 不把错误VJP加入学生梯度。
        distillation_loss, _ = training.score_distillation_loss(prediction, components["teacher"], components["diffusion"], score_timesteps[start:stop], score_noise[start:stop], speaker[start:stop], bnf[start:stop], target_mode=config.score_target, weight_mode=config.score_weight)  # 当前八条使用完整逻辑批预抽取的教师随机量。
        for name, value in (("student adversarial loss", adversarial_loss), ("raw feature matching loss", feature_matching_loss), ("raw score distillation loss", distillation_loss)):  # 三项原论文基础损失都必须有限。
            training._assert_finite(name, value)  # 任一微批坏数都会取消整个逻辑更新。
        run_known_reflection_backward(torch, lambda: torch.autograd.backward(fake_waveform, waveform_gradient / microbatch_count, retain_graph=True))  # 只让冻结声码器路径中的已知反射填充反向采用CUDA现有实现，并保留共享学生图供蒸馏项继续累计。
        distillation_loss.backward(distillation_loss.new_tensor(config.lambda_dist / microbatch_count))  # 不含反射填充的蒸馏路径仍在严格确定性模式下累计四分之一梯度。
        adversarial_value += float(adversarial_loss.detach()) / microbatch_count  # 日志按四个等大微批平均。
        feature_matching_value += float(feature_matching_loss.detach()) / microbatch_count  # 日志缩放与实际梯度目标一致。
        distillation_value += float(distillation_loss.detach()) / microbatch_count  # 蒸馏日志同样恢复完整batch32均值。
        del proxy_prediction, real_proxy, fake_proxy, adversarial_loss, feature_matching_loss, waveform_gradient, prediction, fake_waveform, distillation_loss  # 释放当前学生微批全部计算图后再处理下一组。
    if any(parameter.grad is not None for parameter in components["teacher"].parameters()):  # 冻结教师不能因微批累计收到参数梯度。
        raise RuntimeError("冻结教师意外收到梯度。")  # 保护原论文教师固定不更新的约束。
    if any(parameter.grad is not None for parameter in components["vocoder"].parameters()):  # 声码器只允许梯度穿过输入。
        raise RuntimeError("冻结声码器意外收到参数梯度。")  # 防止把声码器混入学生Adam。
    if any(parameter.grad is not None for parameter in components["discriminator"].parameters()):  # 学生阶段D参数必须保持无梯度。
        raise RuntimeError("冻结判别器在学生阶段意外收到梯度。")  # 保证一次D更新与一次学生更新严格分离。
    return {"student_total": adversarial_value + config.lambda_fm * feature_matching_value + config.lambda_dist * distillation_value, "student_adv": adversarial_value, "fm_raw": feature_matching_value, "fm_weighted": config.lambda_fm * feature_matching_value, "dist_raw": distillation_value, "dist_weighted": config.lambda_dist * distillation_value, "cycle_raw": 0.0, "cycle_weighted": 0.0, "discriminator": discriminator_value}  # 返回与实际逻辑batch32目标一致的九项日志，学生step由外层在Cycle梯度后执行。

def formal_train_batch(runtime, components, batch, args, next_step, cycle_step):  # 一次完整D更新及S更新，按门控决定是否加入Cycle。
    config = components["config"]  # 读取报告中固定参数。
    device = components["device"]  # 使用当前真实GPU。
    active = arm_spec(args.arm)["cycle_after"] is not None and next_step > arm_spec(args.arm)["cycle_after"]  # Early在第21pass首批开启，Post在父断点后首批开启。
    losses = accumulated_base_gradients(runtime, components, batch)  # 四个物理batch8先累计成原论文逻辑batch32，D和学生都暂不增加额外更新次数。
    cycle_value = 0.0  # Base和Early前20pass完全不运行Cycle额外前向。
    if active:  # 只有实际Cycle更新才构建24条往返样本。
        tensors = runtime.training._slice_batch(batch, device, require_cycle=True)  # 原字段检查保证mel80、BNF144、身份256及有效长度正确。
        pair_generator = runtime.torch.Generator(device=device).manual_seed(config.cycle_train_pair_seed + (args.seed - 1234) + cycle_step + 1)  # seed1234沿用已登记Cycle种子，另两次重复只增加相应种子偏移。
        targets = runtime.training.select_cycle_target_indices(batch["spk_name"], 24, device, generator=pair_generator)  # 从batch32候选池为24个源选异说话人，允许重复目标。
        forward_noise = independent_noise(runtime, (24, 80, 128), device, config.cycle_train_forward_noise_seed + (args.seed - 1234) + cycle_step + 1)  # 第一程独立噪声按已完成Cycle步数续接，1234重复不改原登记种子。
        backward_noise = independent_noise(runtime, (24, 80, 128), device, config.cycle_train_backward_noise_seed + (args.seed - 1234) + cycle_step + 1)  # 第二程噪声与第一程和ACDD都独立。
        for start in range(0, 24, 3):  # 24条分成8组，每组3条，只控制峰值显存而不改变逻辑batch。
            stop = start + 3  # 每组刚好3个Cycle源样本。
            errors = cycle_per_sample(runtime, components, tensors["mel"][start:stop], tensors["bnf"][start:stop], tensors["spk_emb"][start:stop], tensors["spk_emb"].index_select(0, targets[start:stop]), tensors["mel_len"][start:stop], forward_noise[start:stop], backward_noise[start:stop], gradient=True)  # 第一程无梯度，第二程保留对同一个学生的梯度。
            contribution = errors.sum() / 24  # 八组相加等于24条逐样本MAE的平均，每组均值等价乘1/8。
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
    losses.update(cycle_raw=cycle_value, cycle_weighted=args.lambda_cycle * cycle_value, student_total=losses["student_total"] + args.lambda_cycle * cycle_value)  # 日志总值与实际反向目标严格对应。
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
    references = [next(all_indices[(position + shift) % len(all_indices)] for shift in range(1, len(all_indices)) if speaker_names[(position + shift) % len(all_indices)] != speaker_names[position]) for position in range(len(all_indices))]  # 固定选择循环向后的第一位异说话人参考。
    totals = {key: 0.0 for key in LOSS_KEYS}  # 累积逐批均值乘真实样本数，尾批不被放大。
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
                predicted, _ = runtime.training.make_student_prediction(components["student"], components["diffusion"], mel, speaker, bnf, 950, noise)  # 与训练相同的单步学生预测。
                real_wave = components["vocoder"](runtime.training.denormalize_mel(mel, components["mean"], components["std"]))  # 真实mel经过相同冻结声码器供D诊断。
                fake_wave = components["vocoder"](runtime.training.denormalize_mel(predicted, components["mean"], components["std"]))  # 当前学生生成波形供D诊断。
                d_loss, adv, fm = runtime.training.serial_discriminator_metrics(components["discriminator"], real_wave, fake_wave)  # 真实计算8个判别分支的D、对抗和特征匹配损失。
                dist, _ = runtime.training.score_distillation_loss(predicted, components["teacher"], components["diffusion"], timesteps, score_noise, speaker, bnf, target_mode=config.score_target, weight_mode=config.score_weight)  # 真实冻结教师蒸馏目标，不用简单重建误差冒充。
                targets = torch.stack([dataset[index]["spk_emb"] for index in references[offset : offset + size]]).to(device)  # 固定跨全清单异说话人目标参考。
                first_noise = independent_noise(runtime, mel.shape, device, bank_seed + 4)  # 固定第一程Cycle噪声。
                second_noise = independent_noise(runtime, mel.shape, device, bank_seed + 5)  # 固定独立第二程Cycle噪声。
                cycle_sum = 0.0  # 验证Cycle覆盖每一条语音，不仅覆盖训练的前24条。
                for start in range(0, size, 3):  # 验证也采用3条微批避免突然增加显存。
                    stop = min(size, start + 3)  # 最后一组可以少于3条，仍保留全部样本。
                    cycle_sum += float(cycle_per_sample(runtime, components, mel[start:stop], bnf[start:stop], speaker[start:stop], targets[start:stop], lengths[start:stop], first_noise[start:stop], second_noise[start:stop], gradient=False).sum())  # 每条先平均80通道有效帧再加入样本总和。
                cycle = cycle_sum / size  # 当前批逐样本Cycle-MAE均值，Base也测量相同诊断值。
                weighted = args.lambda_cycle * cycle if active else 0.0  # Base及warmup的目标贡献仍为零。
                values = finite_losses({"student_total": float(adv + 2 * fm + 45 * dist) + weighted, "student_adv": float(adv), "fm_raw": float(fm), "fm_weighted": float(2 * fm), "dist_raw": float(dist), "dist_weighted": float(45 * dist), "cycle_raw": cycle, "cycle_weighted": weighted, "discriminator": float(d_loss), "reconstruction_masked_mae": float(masked_per_sample_mae(runtime, mel, predicted, lengths).mean())})  # 保留全部实际组成及额外固定重建读数。
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

def parse_args(argv=None):  # 读取一个分支的明确运行参数，Notebook负责选择并逐个启动。
    parser = argparse.ArgumentParser(description=__doc__)  # 显示当前正式训练职责。
    parser.add_argument("--project-root", type=Path, required=True)  # 明确原项目高速缓存位置。
    parser.add_argument("--run-root", type=Path, required=True)  # 明确实验主机数据盘上的永久实验记录根目录。
    parser.add_argument("--arm", choices=ARMS, required=True)  # 仅运行报告中四个固定分支之一。
    parser.add_argument("--seed", type=int, choices=(1234, 2234, 3234), default=1234)  # 正式实验预先声明三次重复种子。
    parser.add_argument("--lambda-cycle", type=float, default=0.0125)  # 删除额外除80后以1/80作为起始尺度。
    parser.add_argument("--parent", type=Path)  # Post两臂可显式指定同一个完整100pass基线父断点。
    parser.add_argument("--bnf-checkpoint", type=Path)  # 可指定已知固定Conformer权重的缓存位置。
    parser.add_argument("--screen-passes", type=int)  # 权重筛选只额外训练固定数pass且仅使用development选择。
    parser.add_argument("--metrics-endpoint", action="store_true")  # 启动器传入此开关运行规定的90对开发及8100对测试客观评估。
    parser.add_argument("--session-update-limit", type=int)  # 首次双GPU联合预检可只完成少量真实更新并保存，之后从同一断点继续。
    parser.add_argument("--run-approved", action="store_true")  # 只有用户实际运行启动单元时才授权消耗GPU。
    args = parser.parse_args(argv)  # 解析实际Notebook或静态测试传入参数。
    if not args.run_approved:  # 上传代码或导入函数不代表启动训练。
        raise RuntimeError("正式运行需要 --run-approved，准备文件不会自动训练。")  # 不在本地或上传阶段意外启动。
    if not math.isfinite(args.lambda_cycle) or args.lambda_cycle < 0 or (args.arm in ("early_cycle_100", "cycle_post") and args.lambda_cycle <= 0):  # Cycle必须有限正值，Base允许显式lambda0。
        raise ValueError("Cycle的lambda必须为有限正值，Base可用0。")  # 拒绝坏参数进入计算图。
    expected_screen_arm = "base_post" if args.lambda_cycle == 0 else "cycle_post"  # lambda0必须同预算继续训练，不能拿未续训的共同父模型冒充对照。
    if args.screen_passes is not None and (args.arm != expected_screen_arm or args.seed != 1234 or not 1 <= args.screen_passes <= 50):  # 对照与Cycle候选都固定seed1234及相同追加预算。
        raise ValueError("权重筛选要求lambda0使用base_post、正lambda使用cycle_post、seed1234和1到50个追加pass。")  # 防止对照臂、选参集合或预算混淆。
    if args.screen_passes is not None and args.lambda_cycle not in SCREEN_LAMBDAS:  # 后续筛选使用单独预声明的三个候选权重。
        raise ValueError("后续筛选候选必须是0、0.5、2.5、10.0之一。")  # 候选集合在启动前固定，绝不根据测试结果增删。
    if args.session_update_limit is not None and args.session_update_limit < 1:  # 会话上限必须至少允许一个完整逻辑更新。
        raise ValueError("session-update-limit必须是正整数。")  # 零或负数不能冒充CUDA训练验证。
    args.project_root = args.project_root.resolve()  # 固定真正使用的模型源码与数据路径。
    args.run_root = args.run_root.resolve()  # 固定永久实验记录的实际目录。
    return args  # 返回已经完成参数语义检查的命令行对象。

def run_name(args):  # 每个分支、种子和Cycle权重都使用独立永久目录。
    weight = format(args.lambda_cycle, ".8g").replace(".", "p")  # 文件名保留完整候选值而不依赖浮点显示小数位。
    suffix = f"_screen_{args.screen_passes}pass" if args.screen_passes is not None else ""  # 筛选轨迹与正式实验永久分开。
    return f"{args.arm}_seed_{args.seed}_lambda_{weight}{suffix}"  # 避免多个Notebook覆盖其他分支或权重。

def state_fingerprint(runtime, model):  # 用逐张量摘要确认两臂共有同一个模型初始化。
    digest = hashlib.sha256()  # 小内存流式摘要不另存整个模型。
    for name, value in model.state_dict().items():  # 模型结构固定顺序逐项处理。
        digest.update(name.encode("utf-8"))  # 参数名也参与身份，防止同数值不同结构混用。
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())  # 记录真实参数及buffer的字节，不用文件名代替。
    return digest.hexdigest()  # 返回可直接写入合同的共同起点身份。

def restore_complete(runtime, components, payload, args, contract_sha, is_parent=False):  # 恢复两模型、两Adam和全部随机状态，权重-only文件不能续训。
    required = {"student", "discriminator", "student_optimizer", "discriminator_optimizer", "rng_state", "global_step", "completed_passes", "partial_batch", "resumable", "configuration"}  # 定义真正完整训练态所需字段。
    if not isinstance(payload, dict) or not required <= set(payload):  # 缺少Adam或随机状态会改变续训数学。
        raise RuntimeError(f"断点不是完整训练状态，缺少：{sorted(required - set(payload or {}))}")  # 明确拒绝伪造或近似的续训。
    step = int(payload["global_step"])  # 读取已完成更新游标。
    if payload["partial_batch"] is not False or payload["resumable"] is not True or payload["completed_passes"] != step // UPDATES_PER_PASS:  # 模型和优化器必须位于同一个完整批次边界。
        raise RuntimeError("断点不是同一完整批次边界的可恢复状态。")  # 半次D或S更新绝不能接入正式对照。
    runtime.stage.validate_core_configuration(payload["configuration"])  # 核对原ACDD核心目标、帧数及对齐协议。
    if not is_parent and (payload.get("kind") != KIND or payload.get("run_contract_sha256") != contract_sha or payload.get("arm") != args.arm or payload.get("cycle_step") != expected_cycle_step(args.arm, step)):  # 正式续训还必须属于同一合同和Cycle游标。
        raise RuntimeError("当前续训断点不属于这个正式实验。")  # 防止不同臂或lambda互相接续。
    if is_parent and (step != 100 * UPDATES_PER_PASS or payload.get("cycle_enabled", False) or payload.get("arm", "base_100") not in ("base_100", None)):  # Post必须共享无Cycle的完整100pass基线。
        raise RuntimeError("Post父断点必须是无Cycle的完整100pass基线状态。")  # 不以已经训练Cycle的模型作父状态。
    for role, model_key, optimizer_key, lr in (("学生", "student", "optimizer", 5e-5), ("判别器", "discriminator", "d_optimizer", 2e-4)):  # 同时恢复并审核两组实际状态。
        components[model_key].load_state_dict(payload[model_key], strict=True)  # 严格恢复真实权重与全部buffer。
        saved_key = "student_optimizer" if model_key == "student" else "discriminator_optimizer"  # 兼容原完整断点的正式Adam键名。
        components[optimizer_key].load_state_dict(payload[saved_key])  # 恢复动量、二阶矩和Adam step，不重置学习率。
        if step > 0:  # 初始0步Adam尚未创建动量，这是合法初始化状态。
            runtime.stage.validate_optimizer_state(components[optimizer_key], components[model_key], lr, step, role)  # 所有参数的Adam游标必须等于真实已完成步数。
        elif components[optimizer_key].state:  # 0步断点不能含已经更新过的Adam动量。
            raise RuntimeError("0步初始化断点含有已训练Adam状态。")  # 禁止把历史学生改标签作为新起点。
        runtime.stage.validate_model_state(components[model_key], role)  # 检查加载后的权重和buffer没有坏数。
        components[model_key].train()  # 模式不在state_dict中，因此显式恢复训练状态。
    audit = runtime.stage.restore_stage_rng_state(payload, components["device"], step)  # 同后端严格恢复全部随机状态；旧父CUDA格式差异有原入口明确审计。
    return step, audit  # 返回真实恢复游标和随机状态恢复证据。

def checkpoint_payload(runtime, components, args, state, contract_sha, journal):  # 将完整训练态和对应损失边界组装到同一提交。
    step = int(state["global_step"])  # 只允许使用最后完成的D及S更新游标。
    sequence = journal.commit(step)  # 先发布对应已完成更新的真实loss记录，不包含未完成批次。
    return {"kind": KIND, "arm": args.arm, "cycle_enabled": args.arm in ("early_cycle_100", "cycle_post"), "global_step": step, "completed_passes": step // UPDATES_PER_PASS, "cycle_step": expected_cycle_step(args.arm, step), "partial_batch": False, "resumable": True, "run_contract_sha256": contract_sha, "configuration": components["config"].as_dict(), "effective_optimizer_lrs": {"student": 5e-5, "discriminator": 2e-4}, "mc_samples": 1, "score_semantics": "reconstruction_clean_x0_sqrt_alpha_bar", "student": components["student"].state_dict(), "discriminator": components["discriminator"].state_dict(), "student_optimizer": components["optimizer"].state_dict(), "discriminator_optimizer": components["d_optimizer"].state_dict(), "rng_state": runtime.training.capture_rng_state(), "loss_committed_sequence": sequence, "formal_state": dict(state)}  # 保存完整动量、RNG、累计loss、验证游标和成本，避免重连后损失曲线断层。

def contract_record(runtime, components, args, parent_sha, initial_sha):  # 在任何正式更新前固定实验来源及数学设置。
    root = args.project_root  # 使用唯一缓存副本计算源代码身份。
    paths = [Path(__file__).resolve(), Path(__file__).parent / "loss_tracking.py", Path(__file__).parent / "suite_launcher.py", Path(__file__).parent / "fastvoicegrad_evaluate.py", root / "experiments/cycle_stage/train.py", root / "exp/diagnostics/student_performance_tools_20260906/run_acdd_development.py", root / "preprocess/05_extract_bnf.py", root / "preprocess/03_extract_mel_22050.py", root / "preprocess/06_extract_spk_emb.py"]  # 训练、保存、批次读取和完整评测的核心入口全部绑定。
    for relative in ("fastvoicegrad", "voicegrad", "hifigan/custom", "hifigan/official", "evaluation", "external/legacy_compute_bnf/conformer_ppg_model"):  # 记录影响训练、在线内容与正式指标的真实模块。
        paths.extend(sorted((root / relative).rglob("*.py")))  # 对小型源码计算摘要，不重复扫描数万数据文件。
    source = {str(path.relative_to(root)) if root in path.parents else path.name: runtime.stage.sha256_file(path) for path in paths if path.is_file()}  # 每个被实际引用的源码内容有明确身份。
    return {"kind": KIND, "arm": args.arm, "seed": args.seed, "lambda_cycle": args.lambda_cycle, "screen_passes": args.screen_passes, "spec": arm_spec(args.arm), "configuration": components["config"].as_dict(), "initial_model_sha256": initial_sha, "parent_sha256": parent_sha, "source_sha256": source, "development_contract_sha256": runtime.stage.sha256_file(runtime.stage.DEVELOPMENT), "runtime_identity": runtime.stage.runtime_identity(components["device"]), "optimizer": {"student_lr": 5e-5, "discriminator_lr": 2e-4, "betas": [0.5, 0.9], "eps": 1e-8, "weight_decay": 0, "amsgrad": False, "ratio": "1D:1S", "schedule": "constant", "amp": False, "gradient_clipping": None}, "batching": {"paper_effective_batch_size": 32, "physical_microbatch_size": 8, "base_microbatches_per_update": 4, "discriminator_steps_per_logical_update": 1, "student_steps_per_logical_update": 1, "updates_per_pass": UPDATES_PER_PASS, "loss_scaling": "equal_microbatch_mean_divided_by_4", "vjp_recompute_identity": "bitwise_equal_forward_with_cudnn_deterministic", "cuda_determinism_exception": "reflection_pad1d_backward_scoped_only_because_pytorch_has_no_deterministic_cuda_implementation"}, "cycle": {"selected_samples": 24, "microbatch_samples": 3, "microbatches": 8, "first_leg_stop_gradient": True, "generated_bnf_reextracted": True, "noise_per_leg": "independent_standard_normal", "reduction": "mean_over_80_valid_frames_per_sample_then_mean_over_24", "extra_division_by_80": False, "lambda_rationale": "new development-only rescreen uses 0.5/2.5/10.0 to span the empirically estimated optimizer-update influence from about 0.9% to 18%; all candidates were fixed before execution", "warmup_rationale": "post-stage screening starts from the same complete pass-100 baseline state"}, "initialization": "frozen_teacher_epoch500_weights_for_student; scratch_discriminator; empty_Adam_states_at_global_step0" if not args.arm.endswith("post") else "shared_full_baseline_pass100_including_both_Adam_and_RNG", "historical_model_notice": "new Early study starts from teacher at step0 and is not claimed to replay the historical selected3327 warm-start trajectory", "loss_evaluation": {"development_examples": 52, "development_conversion_pairs": 90 if args.metrics_endpoint else 0, "test_examples": 0 if args.screen_passes is not None else 100, "development_frequency_passes": 1, "development_conversion_frequency_passes": 5 if args.metrics_endpoint else None, "test_use_for_selection": False, "test_frequency": "never_during_screening" if args.screen_passes is not None else "initial_and_fixed_endpoint_only", "cycle_diagnostic_on_base": True}, "cost_policy": "actual training and evaluation seconds plus allocated GPU wall-clock saved separately", "save_every_updates": 2000, "save_every_seconds": 1800}  # 后续筛选完整登记候选依据、development-only选参和不读取测试集的边界。

def run_training(argv=None, lease=None):  # 在已取得独占数据盘租约后执行一个可以反复续训的正式训练分支。
    args = parse_args(argv)  # 在加载ML库前拒绝没有明确启动开关的调用。
    from loss_tracking import LossJournal, atomic_json, plot_losses  # 共享真实loss分段提交与多集合绘图入口。
    from suite_launcher import checkpoint_load, checkpoint_store, RunLease  # 共享两槽完整断点以及共同来源短期排他保护。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 在首次CUDA数学操作前固定cuBLAS确定性工作区配置。
    runtime = load_runtime(args.project_root)  # 本地Mac在这里直接拒绝计算。
    torch = runtime.torch  # 后续只操作实验主机中的真实CUDA模型。
    torch.backends.cudnn.benchmark = False  # 禁止根据首次输入计时选择可能不同的卷积算法。
    torch.backends.cudnn.deterministic = True  # 强制CuDNN为相同输入选择确定性实现。
    torch.use_deterministic_algorithms(True, warn_only=False)  # 任一无确定性实现的训练算子必须显式报错而不是悄悄漂移。
    assigned_root = Path("/root/FastVoiceGrad").resolve()  # 所有5090结果和断点固定在本实例300GB系统盘目录。
    if not assigned_root.is_dir() or (args.run_root != assigned_root and assigned_root not in args.run_root.parents):  # 永久输出不能落到用户分配目录之外。
        raise RuntimeError("结果根目录必须位于 /root/FastVoiceGrad。")  # 防止写入容量仅5GB的/mnt网盘或其他目录。
    args.run_root.mkdir(parents=True, exist_ok=True)  # 先建立本轮指定的永久结果根目录。
    runtime.stage.DRIVE_RUN_ROOT = args.run_root  # 复用旧保护逻辑时把永久根切换为本机分配数据盘。
    if shutil.disk_usage(args.project_root).free < 4 * 1024**3 or shutil.disk_usage(args.run_root).free < 4 * 1024**3:  # 两份滚动断点都要求足够剩余空间。
        raise RuntimeError("本地或Drive剩余空间不足4GiB，不能安全保存完整状态。")  # 不等存储写满后才报错。
    torch.set_num_threads(2)  # 与已复现长期FastVoiceGrad保持相同CPU线程设置。
    if not torch.cuda.is_available():  # 正式训练绝不允许回退到CPU。
        raise RuntimeError("当前进程没有可用CUDA GPU。")  # 要求用CUDA_VISIBLE_DEVICES固定到一张2080Ti后重试。
    device = torch.device("cuda:0")  # 每个独立tmux进程只看见一张物理GPU并在逻辑cuda0运行。
    properties = torch.cuda.get_device_properties(device)  # 读取当前可见卡的真实型号和显存。
    if properties.total_memory < 10 * 1024**3:  # 新累计协议仍要求至少10GiB单卡显存。
        raise RuntimeError(f"当前GPU显存不足10GiB：{properties.name}。")  # 不在不足设备上冒险启动正式轨迹。
    runtime.training.seed_everything(args.seed)  # 两个Early分支共享同一教师初始化和随机判别器起点。
    components = build_components(runtime, args, device)  # 构建全部冻结依赖和可训练模型，缺文件立即停止。
    train_indices = runtime.stage.load_training_indices()  # 精确读取已有52条排除协议及固定数据统计身份。
    development = runtime.stage.read_json(runtime.stage.DEVELOPMENT)  # 读取已有固定开发集合，绝不临时重新抽样。
    dev_indices = sorted(development["excluded_train_indices"])  # 每pass诊断覆盖全部52条，不把开发集丢回梯度更新。
    parent = args.parent or runtime.stage.POST_PARENT_PATH  # Post默认继承已经复现的无Cycle完整100pass状态。
    parent_sha = runtime.stage.sha256_file(parent) if args.arm.endswith("post") and parent.is_file() else None  # 内容身份用于两臂绑定同一个父状态。
    if args.arm.endswith("post") and parent_sha is None:  # Post不能用教师重新初始化冒充已有100pass。
        raise FileNotFoundError(f"Post缺少完整100pass父断点：{parent}")  # 直接列出必要资产。
    parent_payload = runtime.stage.load_checkpoint(parent) if args.arm.endswith("post") else None  # 只读取Post真正需要的完整训练态。
    if parent_payload is not None:  # 父文件必须是已确认的历史基线或本次base_100正式完整断点。
        if parent_sha != LEGACY_PARENT_SHA256 and not (parent_payload.get("kind") == KIND and parent_payload.get("arm") == "base_100"):  # 任意其他同名100pass不能自动混入。
            raise RuntimeError("Post父断点既不是已复现基线，也不是本次正式Base-100。")  # 缺少共同起点证据时停止。
        restore_complete(runtime, components, parent_payload, args, None, is_parent=True)  # 确认两Adam、RNG和权重完整并立即严格恢复。
    initial_sha = {key: state_fingerprint(runtime, components[key]) for key in ("student", "discriminator")}  # 明确两个Early初始模型或Post父模型实际字节身份。
    contract = contract_record(runtime, components, args, parent_sha, initial_sha)  # 固定新的实验设置，后续重连不允许静默改变。
    contract["compatibility"] = {"conformer_stft_policy": BNF_STFT_COMPAT_POLICY, "conformer_complex_representation": "last_axis_real_imag_pair", "global_torch_monkeypatch": False, "legacy_asset_source_mutation": False}  # 将新版STFT局部兼容方法和保持旧数值布局的策略永久纳入合同。
    output = args.run_root / run_name(args)  # 每臂每seed每lambda使用独立永久结果目录。
    output.mkdir(parents=True, exist_ok=True)  # 首次运行只创建自己的结果目录。
    contract_path = output / "run_contract.json"  # 正式合同位置用于评测和续训身份校验。
    if contract_path.is_file() and runtime.stage.read_json(contract_path) != contract:  # 已有实验参数、初始状态或后端发生变化时不覆盖。
        raise RuntimeError("现有正式合同与当前代码或GPU环境不同，请使用原设置续训。")  # 拒绝将不同轨迹拼接成一条曲线。
    atomic_json(contract_path, contract)  # 在第一批梯度更新前发布整个实验身份。
    contract_sha = runtime.stage.canonical_sha256(contract)  # 将合同内容摘要放入每个完整断点。
    paired_runtime_identity = dict(contract["runtime_identity"])  # 复制完整运行时身份，避免配对归一化修改各分支实际合同记录。
    paired_runtime_identity.pop("gpu_total_memory", None)  # 同型号显卡可能因固件保留区相差少量可见字节，配对仍严格比较型号、算力和全部数学后端设置。
    shared_identity = {"kind": KIND, "seed": args.seed, "parent_sha256": parent_sha, "initial_model_sha256": initial_sha, "source_sha256": contract["source_sha256"], "development_contract_sha256": contract["development_contract_sha256"], "runtime_identity": paired_runtime_identity}  # 两臂必须共用相同模型、数据、源码和除驱动保留显存字节外的后端身份。
    pair_root = args.run_root / "paired_origins"  # 短小配对来源收据位于所有臂的共享永久根目录。
    pair_path = pair_root / f"{'post' if args.arm.endswith('post') else 'early'}_seed_{args.seed}.json"  # lambda变化仍必须继承相同Post父状态。
    pair_lease = None  # 两个不同臂同时启动时，来源登记需要短暂串行确认。
    pair_deadline = time.monotonic() + 30  # 来源短临界区争锁至多等待30秒。
    while pair_lease is None:  # 等另一个初始化进程完成几KiB收据的发布。
        candidate_lease = RunLease(pair_root / (pair_path.stem + "_source_lock"), ttl_seconds=120, heartbeat_seconds=30)  # 独占共同来源收据的检查及首次写入。
        try:  # 来源短期争锁失败不代表整个实验失败。
            pair_lease = candidate_lease.__enter__()  # 通过原子排他创建取得跨Colab保护。
        except (RuntimeError, FileExistsError):  # 另一个臂正在写共享来源时短暂重试。
            if time.monotonic() >= pair_deadline:  # 已等待足够时间仍争锁失败时明确停止。
                raise RuntimeError("共同来源登记仍被另一个运行时占用，请稍后重跑当前单元格。")  # 不能绕过来源比较强行并行。
            time.sleep(1)  # 只在Colab用户运行代码中等待一秒，不持续请求外部工具。
    try:  # 在同一个短临界区完成严格比较或第一次写入。
        pair_lease.assert_owned()  # 防止来源保护在读写期间意外丢失。
        if pair_path.is_file():  # 已登记共同来源时不覆盖原证据。
            if runtime.stage.read_json(pair_path) != shared_identity:  # 两臂必须真正共享模型、数据、源码和同型号GPU。
                raise RuntimeError("配对实验来源不同；续训须使用同型号GPU、同软件和原配置。")  # 禁止不同数学轨迹混为同一正式对照。
        else:  # 第一个臂只发布一次共同来源。
            atomic_json(pair_path, shared_identity)  # 在排他保护下登记真实模型及环境来源。
    finally:  # 来源比较失败也必须释放自己取得的短租约。
        pair_lease.__exit__(None, None, None)  # 只释放当前初始化进程的来源保护。
    payload, receipt = checkpoint_load(output, map_location="cpu")  # 两槽只恢复已经正式提交且完整可读的训练态。
    spec = arm_spec(args.arm)  # 取得报告中的固定阶段范围。
    target_step = spec["end"] if args.screen_passes is None else spec["start"] + args.screen_passes * UPDATES_PER_PASS  # 筛选只追加指定短训长度，正式臂终点不变化。
    state = {"global_step": spec["start"], "cycle_step": 0, "train_pass_sums": {key: 0.0 for key in LOSS_KEYS}, "train_pass_batches": 0, "development_done_passes": [], "conversion_done_passes": [], "test_done_steps": [], "endpoint_metrics_done": False, "training_seconds": 0.0, "evaluation_seconds": 0.0, "allocated_runtime_seconds": 0.0, "peak_vram_bytes": 0, "status": "preparing"}  # 这些累加器与断点一起保存，避免中途续训丢失当前pass均值。
    if payload is not None:  # 已有正式断点时从最近完整更新继续。
        restored_step, rng_audit = restore_complete(runtime, components, payload, args, contract_sha)  # 恢复两模型、两Adam以及精确随机状态。
        state.update(payload["formal_state"])  # 恢复所有损失、验证及成本游标。
        if restored_step != state["global_step"] or not spec["start"] <= restored_step <= target_step:  # 模型游标和附属状态必须一致。
            raise RuntimeError("恢复模型游标与正式阶段范围不一致。")  # 拒绝错误状态提前宣布完成。
    else:  # 首次运行使用已验证的教师初始状态或共享Post父状态。
        rng_audit = {"mode": "shared_full_parent" if parent_payload is not None else "teacher_initialized_step0", "global_step": state["global_step"]}  # 明确没有使用历史3327学生冒充0步。
    journal = LossJournal(output, committed_sequence=payload.get("loss_committed_sequence", 0) if payload else 0)  # 丢弃断点之后未提交的旧loss尾段，恢复真实曲线游标。
    del payload, parent_payload  # 释放CPU中重复的大型模型与优化器字典。
    training_dataset = runtime.baseline.VoiceGradDataset(args.project_root / "data", "train", segment_length=128, expected_speakers=100, bnf_alignment=components["config"].bnf_alignment)  # 训练沿用原随机128帧裁剪。
    full_train_dataset = runtime.baseline.VoiceGradDataset(args.project_root / "data", "train", segment_length=None, expected_speakers=100, bnf_alignment=components["config"].bnf_alignment)  # 固定开发片段从完整语音确定性裁剪。
    test_dataset = runtime.baseline.VoiceGradDataset(args.project_root / "data", "test", segment_length=None, expected_speakers=10, bnf_alignment=components["config"].bnf_alignment) if args.screen_passes is None else None  # 正式分支读固定测试，lambda筛选不读取测试语音或特征。
    if len(training_dataset) != 39387 or len(dev_indices) != 52 or (test_dataset is not None and len(test_dataset) != 100):  # 数量变化意味着不再是报告中的同一数据协议。
        raise RuntimeError("训练39387、开发52、测试100的固定人口不一致。")  # 禁止自动缩减或重复数据。
    planner = runtime.stage.GlobalBatchPlan(train_indices, seed=args.seed)  # 按绝对步重建每pass相同的打乱顺序。
    stop = {"requested": False}  # 手动中止只写轻量标记，不在信号处理器里写大断点。
    for signum in (signal.SIGINT, signal.SIGTERM):  # 用户点击停止和Colab终止都尽量在完整批次保存。
        signal.signal(signum, lambda *_: stop.update(requested=True))  # 当前D及S完整返回后再停。
    torch.cuda.reset_peak_memory_stats(device)  # 记录本次运行真实训练及评测峰值显存。
    local_temp = args.project_root / "exp/formal_v3_checkpoint_cache" / run_name(args)  # 序列化先用高速本地盘，永久指针仍只在Drive。
    session_started = time.monotonic()  # 统计当前Colab实际占用时长。
    last_saved = time.monotonic()  # 从恢复时点开始下一次30分钟保存倒计时。
    last_saved_step = int(state["global_step"])  # 进度条显示真实已经提交的可恢复游标。
    atomic_json(output / "resume.json", {"global_step": state["global_step"], "cycle_step": state["cycle_step"], "rng_audit": rng_audit, "time_unix": time.time()})  # 启动后立即让用户看到此前已完成的真实进度。
    session_allocated_before = float(state["allocated_runtime_seconds"])  # 之前各个会话的真实GPU占用时长已保存在断点。
    session_target_step = min(target_step, int(state["global_step"]) + args.session_update_limit) if args.session_update_limit is not None else target_step  # 预检会话只推进指定数量完整更新，正式目标本身不变。
    in_batch = False  # 保存只允许发生在完整批次外部。

    def save_boundary():  # 当前模型、两Adam和loss一起提交，断线时至多回滚到这个真正完整点。
        nonlocal last_saved, last_saved_step  # 更新外层用于时间触发及进度显示的已提交游标。
        if in_batch:  # D已经更新而S还没返回时不能保存。
            raise RuntimeError("拒绝保存半次D/S更新。")  # 保护严格公平的续训状态。
        if lease is not None:  # 正式入口持有独占运行时租约。
            lease.assert_owned()  # 写断点前确认没有被另一个朋友或运行时接管。
        state["allocated_runtime_seconds"] = session_allocated_before + time.monotonic() - session_started  # 从各次会话累积真实GPU占用成本。
        state["peak_vram_bytes"] = max(int(state["peak_vram_bytes"]), int(torch.cuda.max_memory_allocated(device)))  # 跨续训会话保留真实峰值显存。
        saved = checkpoint_store(output, checkpoint_payload(runtime, components, args, state, contract_sha, journal), state["global_step"], local_temp_root=local_temp)  # 本地序列化、Drive双槽完整提交及指针发布。
        last_saved = time.monotonic()  # 只有实际提交成功才重置保存倒计时。
        last_saved_step = int(saved["global_step"])  # 保存收据中的实际完成步数供用户查看。
        atomic_json(output / "status.json", {**state, "target_global_step": target_step, "last_saved_global_step": last_saved_step, "latest_checkpoint": saved})  # 同步轻量进度和真实可恢复断点身份。
        return saved  # 最终评估入口需要明确正式断点位置。

    def evaluate_boundary():  # 重连若恰好落在完整pass边界，继续之前未完成的固定验证。
        step = int(state["global_step"])  # 当前必须来自完整训练态。
        if step % UPDATES_PER_PASS:  # pass中途恢复不用重复每pass诊断。
            return  # 等当前pass最后一批完成后再诊断。
        current_pass = step // UPDATES_PER_PASS  # 使用绝对pass编号，不重新从1编号Post阶段。
        if current_pass not in state["development_done_passes"]:  # 原始共同起点及每个完整pass都各诊断一次。
            started = time.monotonic()  # 单独统计评测实际耗时。
            losses = evaluate_losses(runtime, components, full_train_dataset, dev_indices, "development", args, step)  # 无梯度覆盖52条固定开发语音的全部真实损失。
            journal.append("development", step, losses, completed_pass=current_pass, examples=52, used_for_gradients=False, cycle_diagnostic_on_base=True)  # 验证曲线就是固定development，绝不虚构第二个validation集。
            state["development_done_passes"].append(current_pass)  # 只有整个集合成功才登记完成。
            state["evaluation_seconds"] += time.monotonic() - started  # 训练与验证成本分别保存。
            save_boundary()  # 验证结果与相同模型完整状态一同提交，重连不会丢曲线。
            plot_losses(output)  # 每pass自动刷新训练和development各项真实loss图。
        screen_endpoint = args.screen_passes is not None and step == target_step  # 一天租期的lambda筛选也必须在自己的预定终点生成完整90对开发评分。
        if args.metrics_endpoint and current_pass > 0 and (current_pass % 5 == 0 or screen_endpoint) and current_pass not in state["conversion_done_passes"]:  # 正式长训每5pass评估，短筛选则额外保证预定终点评估。
            from fastvoicegrad_evaluate import evaluate_conversion_in_memory  # 使用根代理实现的真实客观评分适配器。
            started = time.monotonic()  # 记录转换及评分耗时，不混进训练速度。
            result = evaluate_conversion_in_memory(runtime, components, args.project_root, output, "development", args.seed, step)  # 真正转换固定90对并跑UTMOS、P808、CER及ECAPA等评分。
            atomic_json(output / f"development_metrics_pass_{current_pass:04d}.json", result)  # 每个固定时点独立保留评分。
            state["conversion_done_passes"].append(current_pass)  # 评分全成功后才把这个时点标为完成。
            state["evaluation_seconds"] += time.monotonic() - started  # 将开发转换的实际成本纳入报告。
            save_boundary()  # 评分完成游标与同一步模型同时持久化。
        if args.screen_passes is None and step in (spec["start"], target_step) and step not in state["test_done_steps"]:  # 正式臂只在共同起点及固定终点测100条测试loss。
            started = time.monotonic()  # 记录固定测试诊断成本。
            losses = evaluate_losses(runtime, components, test_dataset, range(100), "test", args, step)  # 覆盖全部100条真实测试语音，不更新参数也不用于选lambda。
            journal.append("test", step, losses, completed_pass=current_pass, examples=100, used_for_selection=False)  # 测试曲线只标注确实算过的起点和终点。
            state["test_done_steps"].append(step)  # 成功覆盖整个测试集后记录真实完成位置。
            state["evaluation_seconds"] += time.monotonic() - started  # 评测和训练时间分别报告。
            save_boundary()  # 测试记录和模型游标一起提交。
            plot_losses(output)  # 输出可直接检查的训练、验证、测试loss图与CSV。

    from tqdm.auto import tqdm  # 用真实已提交批数初始化百分比进度条。
    total = target_step - spec["start"]  # Post只显示追加50pass的工作量，Early显示完整100pass。
    completed = state["global_step"] - spec["start"]  # 已完成比例来自Drive完整断点而不是上次浏览器显示。
    print(f"Drive续点：已完成 {completed:,}/{total:,} batch，{100 * completed / total:.2f}%，global_step={state['global_step']}，Cycle更新={state['cycle_step']}。", flush=True)  # 朋友再次运行时立即看见此前真实进度。
    progress = tqdm(total=total, initial=completed, desc=run_name(args), unit="batch", dynamic_ncols=True, mininterval=0.5)  # 每batch前进一次并显示百分比、速度和预计时间。
    try:  # 有异常也只保留此前完整点，不提交损坏的半批状态。
        save_boundary()  # 第一批前持久保存教师0步或Post共同100pass状态。
        evaluate_boundary()  # 完成原共同起点诊断，或继续重连前尚未成功的完整pass验证。
        state["status"] = "running"  # 在数据、权重与共同来源都验证后才标记正式训练。
        for next_step in range(state["global_step"] + 1, session_target_step + 1):  # 从完整断点继续到本次会话边界，已经提交的批次不重复。
            if stop["requested"]:  # 手动停止时可以直接在当前完整点退出。
                break  # 不启动下一次D或S更新。
            batch = runtime.baseline.batch_for(training_dataset, planner.for_step(next_step), args.seed + 100000 + next_step)  # 原绝对步裁剪及批序规则可从任意pass中间恢复。
            runtime.training.seed_everything(args.seed + 200000 + next_step)  # 原ACDD输入噪声及教师时刻按绝对步恢复。
            in_batch = True  # 从这一处开始直到两Adam完整返回都禁止保存。
            started = time.monotonic()  # 记录实际单批训练成本。
            losses = formal_train_batch(runtime, components, batch, args, next_step, state["cycle_step"])  # 真实原ACDD加可选24条重新提取BNF的Cycle更新。
            cursors = [runtime.stage.optimizer_uniform_step(components[key]) for key in ("optimizer", "d_optimizer")]  # 两个Adam都必须恰好完成当前一次更新。
            if cursors != [next_step, next_step]:  # 任一优化器少更新或多更新都会破坏严格配对。
                raise RuntimeError(f"两Adam游标错误：{cursors}，要求{next_step}。")  # 不把不完整批次标记为完成。
            in_batch = False  # 两个实际游标验证后重新允许完整断点提交。
            state["training_seconds"] += time.monotonic() - started  # 只计训练更新耗时，开发指标时间单独保存。
            state["global_step"] = next_step  # 只有真实完整返回后才推进绝对更新游标。
            state["cycle_step"] = expected_cycle_step(args.arm, next_step)  # Cycle游标按实际门控唯一推导。
            journal.append("train", next_step, losses, completed_pass=next_step // UPDATES_PER_PASS, cycle_step=state["cycle_step"], batch_size=32, cycle_samples=24 if losses["cycle_raw"] else 0)  # 逐batch记录九项真实训练loss。
            state["train_pass_batches"] += 1  # 保存当前pass已完成批数用于中途恢复。
            for key in LOSS_KEYS:  # 累积每个pass的九项完整训练损失。
                state["train_pass_sums"][key] += losses[key]  # 等完整pass后按实际批数平均。
            progress.update(1)  # 每个真正完成的batch推进一次百分比。
            progress.set_postfix(total=f"{losses['student_total']:.4f}", cycle=f"{losses['cycle_raw']:.4f}", d=f"{losses['discriminator']:.4f}", saved_step=last_saved_step, refresh=False)  # 实时展示损失及此前永久断点游标。
            pass_boundary = next_step % UPDATES_PER_PASS == 0  # 完整pass结束才发布训练均值和固定验证。
            if pass_boundary:  # 恢复中途pass的累加器也必须恰好覆盖1229个完整批。
                if state["train_pass_batches"] != UPDATES_PER_PASS:  # 完整训练臂的每pass只能有1229批。
                    raise RuntimeError("当前pass累计batch数不是1229，拒绝发布不完整训练均值。")  # 缺步曲线不能被误标成完整pass。
                means = {key: value / UPDATES_PER_PASS for key, value in state["train_pass_sums"].items()}  # 每batch都为32条，所以完整pass按批均值即可。
                atomic_json(output / "train_pass_means" / f"pass_{next_step // UPDATES_PER_PASS:04d}.json", {"global_step": next_step, "batches": UPDATES_PER_PASS, "losses": means})  # 保存便于组会作图的完整pass读数。
                state["train_pass_sums"] = {key: 0.0 for key in LOSS_KEYS}  # 下一pass从零累积各项真实loss。
                state["train_pass_batches"] = 0  # 清零当前pass计数并与完整断点一起提交。
            if pass_boundary or next_step - last_saved_step >= 2000 or time.monotonic() - last_saved >= 1800 or stop["requested"]:  # 按步、时间、完整pass及停止请求都可安全触发提交。
                save_boundary()  # 断线后从此真实游标继续，之后未保存的小段可以重新算。
            if pass_boundary:  # 模型已保存到完整pass边界之后再做验证。
                evaluate_boundary()  # 失败只需重试诊断，已经完成的训练pass无需重做。
        state["status"] = "training_complete" if state["global_step"] == target_step else "session_limit_reached" if state["global_step"] == session_target_step and args.session_update_limit is not None else "stopped_at_complete_batch"  # 完成、预检边界或手动停止都依据真实更新游标。
        final_receipt = save_boundary()  # 最后一批或停止点完整持久化。
        plot_losses(output)  # 最终损失图和CSV只使用已提交的真实记录。
        if args.metrics_endpoint and args.screen_passes is None and state["global_step"] == target_step and not state["endpoint_metrics_done"]:  # 固定终点才计算8100对测试客观指标，筛选从不读测试集。
            from fastvoicegrad_evaluate import evaluate_conversion_in_memory  # 真实评测接口可从已完成的单对WAV和评分继续。
            started = time.monotonic()  # 记录完整8100对端点评估耗时。
            result = evaluate_conversion_in_memory(runtime, components, args.project_root, output, "test", args.seed, target_step)  # 测全部客观指标及真实CycleMAE和推理速度。
            state["evaluation_seconds"] += time.monotonic() - started  # 固定测试转换成本与训练分开统计。
            state["endpoint_metrics_done"] = True  # 只有转换和全部评分成功才登记完成。
            state["status"] = "complete"  # 正式实验包含固定端点评分，完成训练本身不等于完成实验。
            atomic_json(output / "endpoint_metrics.json", result)  # 永久保存实际测试客观结果。
            final_receipt = save_boundary()  # 把真实评测完成游标保存到同一个完整模型断点。
        atomic_json(output / "completion.json", {**state, "target_global_step": target_step, "checkpoint": final_receipt, "screening_uses_test": False})  # 如实记录训练、验证、测试及成本最终状态。
        print(f"当前分支状态：{state['status']}；永久结果：{output}", flush=True)  # 用户直接看到实验真正跑到哪一步及保存位置。
    except BaseException as error:  # 网络挂载、显存、模型和评分异常都保留最近完整断点。
        atomic_json(output / "failure.json", {"error": repr(error), "last_committed_global_step": last_saved_step, "current_global_step": state["global_step"], "partial_batch_not_saved": in_batch, "time_unix": time.time()})  # 明确当前批是否只完成了一部分，不伪造成功。
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
        if value.get("token") != self.token or time.time() - float(value.get("heartbeat_unix", 0)) > int(value.get("ttl_seconds", 1800)):  # 令牌及心跳必须同时有效。
            raise RuntimeError("启动器租约已失效，停止提交当前训练结果。")  # 防止两个不同Colab进程覆盖同一结果。

def main(argv=None):  # 独占当前实际输出目录后运行正式工作器。
    args = parse_args(argv)  # 先确定独立分支、种子和候选权重的准确目录。
    assigned_root = Path("/root/FastVoiceGrad").resolve()  # 获取租约前固定5090实例输出边界。
    if sys.platform != "linux" or not assigned_root.is_dir() or (args.run_root != assigned_root and assigned_root not in args.run_root.parents):  # Mac和分配目录外路径都不能启动。
        raise RuntimeError("正式启动必须在Linux 5090实例并把结果写入 /root/FastVoiceGrad。")  # 文件传输和静态检查本身不会创建训练租约。
    token = os.environ.get("VOICE_RESEARCH_LEASE_TOKEN")  # 启动器可以通过环境明确授予已取得的同任务租约。
    if token:  # Notebook父进程已经保护本分支时，不再次创建同目录租约。
        lease = BorrowedRunLease(args.run_root / run_name(args), token)  # 子进程只借用父进程已经取得的所有权。
        lease.assert_owned()  # 第一批之前核对父令牌确实属于当前实际输出目录。
        run_training(argv, lease=lease)  # 全部保存仍核对所有权，心跳由启动器继续刷新。
        return  # 子进程不释放父进程拥有的租约，由父启动器统一清理。
    from suite_launcher import RunLease  # 使用每分钟心跳和30分钟超时接管的共享租约。
    with RunLease(args.run_root / run_name(args)) as lease:  # 同一训练目录只能有一个朋友或Colab运行时写入。
        run_training(argv, lease=lease)  # 训练、验证和断点都在同一独占所有权下进行。

if __name__ == "__main__":  # 导入模块只定义函数，不在Mac或上传阶段自动执行训练。
    main()  # 仅用户启动Notebook或明确运行命令时执行当前分支。
