# 隔离声码器权重对照：学生最多训练两轮，只把冻结生成器改为公开 VCTK_V1 权重。
"""Two-epoch student diagnostic with an in-place public VCTK_V1 vocoder swap."""
import argparse  # 仅提供不训练的预检查选项，不提供续训或扩大训练轮数选项。
import hashlib  # 记录模型内容和随机状态的独立指纹。
import json  # 读取已锁定的实验契约与公开声码器配置。
import signal  # 收到停止请求后，在完成当前训练批次时退出。
import sys  # 锁定实际 Python 环境，并导入租用服务器现场代码。
from dataclasses import replace  # 从原配置派生同一套 clean_x0/sqrt_alpha_bar 数学设置。
from pathlib import Path  # 所有输入与输出都使用明确的服务器绝对路径。

# 权威运行目录不能由命令行换成其他服务器或本地副本。
ROOT = Path('/root/FsatVoiceGrad')
# 在导入项目或计算库前拒绝本地Windows执行。
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('This diagnostic can run only on the registered Matpool server.')
# 禁止导入期间改动原源码目录的字节码缓存。
sys.dont_write_bytecode = True
# 真正的训练计算只能在已经授权的租用服务器执行。
import torch
# 导入服务器原训练实现，不复制或修改它的数学和优化步骤。
sys.path.insert(0, str(ROOT))
import fastvoicegrad.train as training  # 复用原数据审计、初始化、训练、测试、保存、绘图和监控。
from fastvoicegrad.components import sha256_file  # 按字节核对实际读取的权重和配置。

# 现场源码必须仍是已经审核的干净提交。
EXPECTED_HEAD = '7e39b0cf6892b4a33a3ef2dd7e7c53b5d9367d8b'
# 原修复实验的契约用于证明数据、代码和初始化设置没有悄悄变化。
BASELINE_PATH = ROOT / 'exp/checkpoints/fastvoicegrad_vctk_text_unseen_repair_x0_sqrt_20260906/run_contract.json'
# 原 JSON 文件的字节指纹。
BASELINE_FILE_SHA = 'bc8ef4218ebbdcecd7cf79cc05417c21f47dddf69817188d60d66202287800be'
# 原契约的规范化内容指纹，忽略 JSON 的排版空格。
BASELINE_CANONICAL_SHA = '787c6fe2f6dc8559d5a8b77289b1bc21d8842729376de91d4d7ceab54607694a'
# 公开 VCTK_V1 生成器的本地缓存位置已由主流程确认。
PUBLIC_WEIGHT = ROOT / 'pretrained_hifigan_22k/VCTK_V1/generator_v1'
# 固定官方公开文件的实际 SHA，不能只凭 VCTK_V1 名称加载。
PUBLIC_WEIGHT_SHA = '27cdeb835874516f9404d6c1a9ea229b092fd98c329b8444f5955c24cf7b29a1'
# 与公开生成器一起提供的配置只用于结构和声学接口检查。
PUBLIC_CONFIG = ROOT / 'pretrained_hifigan_22k/VCTK_V1/config.json'
# 公开配置的实际字节指纹。
PUBLIC_CONFIG_SHA = '12107bd79ec9e854cecb2043aa58345e3079ffeb159c9ab9a68ab25fa63deb9b'
# 新实验不会写入原 custom 学生训练目录。
OUTPUT = ROOT / 'exp/diagnostics/student_public_vocoder_train_20260906'
# 预检查报告位于兄弟路径，不创建 OUTPUT，以免正式启动误判为覆盖。
PREFLIGHT_REPORT = ROOT / 'exp/diagnostics/student_public_vocoder_preflight_20260906.json'
# 本次授权只涵盖两轮短期对照，没有开始新的完整一百轮训练。
DIAGNOSTIC_EPOCHS = 2
# 原数据划分每轮应正好产生 1230 个完整 batch32。
BATCHES_PER_EPOCH = 1230


# 统一读取有固定字节指纹的 JSON，失败时不回退到另一份文件。
def verified_json(path, expected_sha):
    # 文件真实内容必须与已审阅证据一致。
    if not path.is_file() or sha256_file(path) != expected_sha:
        # 不匹配意味着实验依赖改变，应停止等待确认。
        raise RuntimeError(f'Locked JSON identity mismatch: {path}')
    # UTF-8 只影响文本读取，不改变实验数值。
    return json.loads(path.read_text(encoding='utf-8'))


# 指纹覆盖模型所有参数和持久缓冲，用来确认公开 G 在训练前后始终不变。
def module_fingerprint(module):
    # 参数字节顺序与名称、形状、dtype 一起进入摘要。
    digest = hashlib.sha256()
    # 排序保证同一 state_dict 不受外部字典显示顺序影响。
    for name, tensor in sorted(module.state_dict().items()):
        # 张量只读复制到 CPU，不建立图、不消耗随机数。
        value = tensor.detach().cpu().contiguous()
        # 名称、dtype 和形状也纳入身份，不能只比较裸字节。
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode('utf-8'))
        # NumPy 这里只提供现有张量字节，不生成随机数。
        digest.update(value.numpy().tobytes())
    # 返回可写进契约的小型字符串。
    return digest.hexdigest()


# 将 Python、NumPy、CPU 和全部 CUDA 随机状态分别按内容哈希。
def rng_fingerprint(state):
    # 不直接哈希 Tensor 的 pickle，因为序列化存储标识不等同随机状态内容。
    digest = hashlib.sha256()
    # Python random 状态是固定结构的整数元组。
    digest.update(repr(state['python']).encode('utf-8'))
    # NumPy 状态包含算法名、整数数组、游标和缓存正态数。
    numpy_state = state['numpy']
    # 非数组字段使用明确文本表示。
    digest.update(repr((numpy_state[0], numpy_state[2:])).encode('utf-8'))
    # NumPy 的整数状态数组按原始字节比较。
    digest.update(numpy_state[1].tobytes())
    # PyTorch CPU 随机流由一个 byte Tensor 表示。
    digest.update(state['torch'].cpu().numpy().tobytes())
    # 依次记录全部可见 GPU 随机流。
    for cuda_state in state['cuda_all']:
        # 每个 GPU 状态只读转成字节。
        digest.update(cuda_state.cpu().numpy().tobytes())
    # 相同内容必须得到相同摘要。
    return digest.hexdigest()


# 对公开生成器做小型输入梯度检查；整个检查前后恢复所有训练随机数。
def check_public_vocoder_gradient(vocoder, train_dataset, mean, std, device):
    # 读取一条训练裁剪也可能消费 Python RNG，因此先保存全部随机流。
    saved_rng = training.capture_rng_state()
    # 保存 G 内容，用来确认只求输入梯度没有改变冻结参数。
    before = module_fingerprint(vocoder)
    # 即使检查报错，也必须恢复随机状态，避免隐藏的后续轨迹变化。
    try:
        # 直接读 dataset 的一条语音，不迭代训练 DataLoader，不启动训练 sampler。
        probe = train_dataset[0]['mel'].unsqueeze(0).to(device).detach().requires_grad_(True)
        # 与真实学生训练相同，先反归一化，再经冻结 G 合成波形。
        waveform = vocoder(training.denormalize_mel(probe, mean, std))
        # 波形能量只用于测试梯度连通，不加入学生训练损失。
        energy = waveform.square().mean()
        # 只对 mel 输入求导，不给 G 或学生执行任何更新。
        gradient, = torch.autograd.grad(energy, probe)
        # 需要有限、非零的输入梯度，才能把波形损失传回学生。
        if not bool(torch.isfinite(gradient).all()) or float(gradient.abs().sum()) <= 0.0:
            # 连通性不成立时不能开始训练。
            raise RuntimeError('Public vocoder has no finite nonzero mel-input gradient')
        # G 的参数必须始终被冻结，也不能积累 .grad。
        if any(parameter.requires_grad or parameter.grad is not None for parameter in vocoder.parameters()):
            # 这说明冻结边界与对照设计不符。
            raise RuntimeError('Public vocoder parameter freezing failed')
        # 内容哈希必须完全不变，而不只是没有调用 optimizer.step。
        if module_fingerprint(vocoder) != before:
            # 检查不得引入隐式参数或缓冲更新。
            raise RuntimeError('Vocoder state changed during input-gradient check')
        # 只返回诊断数值，不保留计算图。
        return {'input_shape': list(probe.shape), 'waveform_shape': list(waveform.shape), 'input_gradient_l2': float(gradient.norm()), 'input_gradient_max_abs': float(gradient.abs().max()), 'vocoder_state_unchanged': True}
    # 无论成功、失败或被中断，训练随机流都退回检查前的位置。
    finally:
        # 恢复包含 CUDA 在内的完整状态。
        training.restore_rng_state(saved_rng, device)


# 仅用于区分用户要求的停止与真正的程序错误。
class RequestedDiagnosticStop(RuntimeError):
    # 该异常在完整训练批次完成后抛出，不伪装成完整 epoch 保存点。
    pass


# 继续使用原 LossMonitor 的窗口和报警，仅补上完整 batch 边界停止功能。
class DiagnosticMonitor(training.LossMonitor):
    # 记录外部停止标志，其他初始化全部交给原监控器。
    def __init__(self, output, window_steps, stop_requested):
        # 原监控器保留相同窗口和上涨阈值。
        super().__init__(output, window_steps=window_steps, resume_step=0)
        # 可变字典由轻量信号处理器设置。
        self.stop_requested = stop_requested

    # 原 train_epoch 每完成一次完整 D/学生更新后会调用此方法。
    def observe(self, global_step, epoch, losses):
        # 先执行原日志、窗口均值及持续上涨报警，不改变任何梯度。
        super().observe(global_step, epoch, losses)
        # 收到停止请求后不继续下一批，也不把半轮标记为完整断点。
        if self.stop_requested['value']:
            # 原 epoch1/epoch2 或初始化文件仍保持完整，当前半轮不另作正式续训点。
            raise RequestedDiagnosticStop('Stopped after a complete diagnostic training batch')


# 此入口必须由主流程审核后在租用服务器明确执行。
def main():
    # 只允许预检查开关；无 --resume，也无超过两轮的选项。
    parser = argparse.ArgumentParser(description='Isolated two-epoch student training with frozen public VCTK_V1.')
    # 预检查运行全部身份门禁和 G 连通探针，但不训练、不创建正式输出目录。
    parser.add_argument('--preflight-only', action='store_true')
    # 解析实际启动请求。
    cli = parser.parse_args()
    # 首先执行项目自带的租用服务器保护。
    training.assert_rented_server(ROOT)
    # 三把运行锁由当前函数持有到退出，阻止和其他训练并发。
    runtime_locks = training.acquire_runtime_locks(ROOT)
    # 新实验目录必须完全不存在，不能覆盖已有或部分完成的对照。
    if OUTPUT.exists():
        # 即使目录尚无 loss，也要求主流程先确认其来源。
        raise RuntimeError(f'Refusing to reuse diagnostic output directory: {OUTPUT}')
    # 预检查证据也不覆盖已有文件。
    if cli.preflight_only and PREFLIGHT_REPORT.exists():
        # 需要重复预检查时先由主流程明确处理旧证据。
        raise RuntimeError(f'Preflight report already exists: {PREFLIGHT_REPORT}')
    # 原训练解释器是 myconda，不能误用评估环境中的另一版 PyTorch。
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or tuple(sys.version_info[:3]) != (3, 8, 15):
        # 环境不同会让单变量对照失去清晰含义。
        raise RuntimeError('Use the original myconda Python 3.8.15 interpreter')
    # 同时锁定 PyTorch 和其 CUDA 运行时版本。
    if torch.__version__ != '1.13.1+cu116' or torch.version.cuda != '11.6':
        # 不为运行方便自动接受另一个依赖版本。
        raise RuntimeError('Expected torch 1.13.1+cu116 and CUDA runtime 11.6')
    # git_head 同时检查工作区干净，不能只检查 HEAD 标签。
    if training.git_head(ROOT) != EXPECTED_HEAD:
        # 现场源码变化后必须重新审核，不能继续复用旧结论。
        raise RuntimeError('Unexpected source revision for isolated vocoder comparison')
    # 读取并双重验证原修复实验契约。
    baseline = verified_json(BASELINE_PATH, BASELINE_FILE_SHA)
    # 规范化指纹还会用于下方完整重建契约的相等检查。
    if training.canonical_json_sha256(baseline) != BASELINE_CANONICAL_SHA:
        # 不接受内容已经改变的基线。
        raise RuntimeError('Baseline canonical contract fingerprint mismatch')
    # 公开配置的字节身份固定，不能由文件名替代验证。
    public_config = verified_json(PUBLIC_CONFIG, PUBLIC_CONFIG_SHA)
    # 公开生成器文件必须与主流程确认的权重完全相同。
    if not PUBLIC_WEIGHT.is_file() or sha256_file(PUBLIC_WEIGHT) != PUBLIC_WEIGHT_SHA:
        # 不自动下载或改用另一个公开模型。
        raise RuntimeError('Locked public VCTK_V1 generator identity mismatch')
    # 使用原基线的数学、数据与优化设置；一百轮字段只作为基线参考，不控制此 driver 的循环。
    config = replace(training.PAPER_STUDENT_CONFIG, score_target='clean_x0', score_weight='sqrt_alpha_bar', loss_monitor_window_steps=50)
    # 原严格加载函数仍先读取自训 G，保持初始化时的随机数消费及原数据验真顺序。
    custom_dir = ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen'
    # build_run_contract 只需要这些已知依赖路径和设备；全部与基线一致。
    original_args = argparse.Namespace(teacher_checkpoint=ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt', vocoder_checkpoint=custom_dir / 'generator_step_2500000.pt', vocoder_config=custom_dir / 'config.json', vocoder_attestation=custom_dir / 'final_checkpoint_attestation_comments_20260906.json', device='cuda')
    # 原依赖必须都存在，不允许跳过 custom 的严格校验步骤。
    for path in (original_args.teacher_checkpoint, original_args.vocoder_checkpoint, original_args.vocoder_config, original_args.vocoder_attestation):
        # 缺失时在产生新训练产物前停止。
        if not path.is_file():
            # 显示具体缺失项。
            raise FileNotFoundError(path)
    # 本次训练与预检查都只在租用 GPU 运行。
    device = torch.device('cuda')
    # CUDA 不可用时不回退到 CPU 训练。
    if not torch.cuda.is_available():
        # 交主流程处理租用 GPU 状态。
        raise RuntimeError('CUDA is unavailable on the rented server')
    # 从这一位置开始完整复刻原 main 的初始化顺序。
    training.seed_everything(config.seed)
    # 原始全量只读数据审计不能省略；其可能的随机消费也与基线顺序一致。
    training.run_full_data_audit(ROOT, config.expected_protocol)
    # 原协议验证返回当前数据和特征的完整指纹。
    data_contract = training.validate_data_protocol(ROOT, config)
    # 先构建教师和学生，并让学生严格复制教师 500 epoch 权重。
    teacher, student, teacher_checkpoint = training.load_teacher_and_student(original_args.teacher_checkpoint, config, device)
    # 再按原路径严格加载 custom G；此时还没有任何训练更新。
    vocoder, hifigan_config, custom_checkpoint = training.load_frozen_vocoder(ROOT, original_args.vocoder_checkpoint, original_args.vocoder_config, original_args.vocoder_attestation, config, device)
    # 判别器继续使用原 custom 配置建立 MPD5+MRD3，不换成公开声码器的训练判别器。
    discriminator = training.build_paper_discriminator(ROOT, hifigan_config, config, device)
    # 扩散日程及反归一化统计的建立顺序也与原 main 相同。
    diffusion = training.VoiceGradDiffusion(config.diffusion_steps, offset=config.cosine_offset).to(device)
    # 读取原训练集 mel 均值和标准差，不重新计算或修改它们。
    mean, std = training.load_mel_statistics(ROOT / 'data', device)
    # 该检查仅证明教师、原 custom G 和当前数据的一致性；不会伪称 public G 有这些指纹。
    training.validate_dependency_data_fingerprints(data_contract, teacher_checkpoint, custom_checkpoint)
    # 四组参数必须彼此独立，避免换 G 时意外触及其他网络。
    training.assert_disjoint_parameter_sets((('teacher', teacher), ('student', student), ('vocoder', vocoder), ('discriminator', discriminator)))
    # 教师冻结状态与原基线一致。
    training.assert_frozen(teacher, 'teacher')
    # custom G 的初始加载状态仍须冻结。
    training.assert_frozen(vocoder, 'vocoder')
    # 学生必须具备可更新参数。
    training.assert_trainable(student, 'student')
    # 原随机初始化判别器必须具备可更新参数。
    training.assert_trainable(discriminator, 'discriminator')
    # 重建原 custom 基线契约，内容必须完整相等，包含源文件、数据、依赖和环境。
    rebuilt_baseline = training.build_run_contract(ROOT, original_args, config, data_contract, teacher_checkpoint, custom_checkpoint)
    # 完整相等比只挑两三个字段更能发现隐含设定变化。
    if training.canonical_json_sha256(rebuilt_baseline) != BASELINE_CANONICAL_SHA:
        # 不在出现额外变量时继续声称只有 G 权重改变。
        raise RuntimeError('Rebuilt baseline contract differs: source/data/dependency/runtime gate failed')
    # 继续按原 main 顺序建立训练加载器，不提前取出第一个批次。
    train_loader = training.get_dataloader(ROOT / 'data', 'train', batch_size=config.batch_size, num_workers=config.num_workers, segment_length=config.train_frames, expected_speakers=config.train_speakers, bnf_alignment=config.bnf_alignment)
    # 固定测试集也使用原函数，不改变 100 句及 32/32/32/4 的布局。
    test_loader = training.build_test_loss_loader(ROOT / 'data', config)
    # 原创建顺序是学生 Adam 在先，判别器 Adam 在后。
    student_optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon, weight_decay=config.weight_decay, amsgrad=config.adam_amsgrad)
    # D 保留相同的独立 Adam 和恒定学习率。
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon, weight_decay=config.weight_decay, amsgrad=config.adam_amsgrad)
    # 两个优化器都不允许包含教师或声码器。
    training.assert_optimizer_ownership(student_optimizer, student, (teacher, vocoder, discriminator), 'student Adam')
    # 判别器优化器只能持有当前随机 D 的参数。
    training.assert_optimizer_ownership(discriminator_optimizer, discriminator, (teacher, vocoder, student), 'discriminator Adam')
    # 确认批量、worker 数及非持久 worker 行为与原基线相同。
    if len(train_loader) != BATCHES_PER_EPOCH or train_loader.batch_size != 32 or train_loader.num_workers != 4 or train_loader.persistent_workers or not train_loader.drop_last:
        # 不用另一种数据加载方式完成这次对照。
        raise RuntimeError('Training DataLoader differs from original batch32/worker4/nonpersistent protocol')
    # 至此原 main 的模型、加载器和优化器初始化顺序已经完整重放。
    startup_rng = training.capture_rng_state()
    # 记录换 G 之前的随机状态，后面必须完全一致。
    startup_rng_sha = rng_fingerprint(startup_rng)
    # 只比较影响生成器结构及实际声学接口的字段，不把公开 G 的训练超参数套到学生上。
    interface_keys = ('resblock', 'upsample_rates', 'upsample_kernel_sizes', 'upsample_initial_channel', 'resblock_kernel_sizes', 'resblock_dilation_sizes', 'num_mels', 'sampling_rate', 'n_fft', 'hop_size', 'win_size', 'fmin', 'fmax')
    # 原位加载要求这些接口相同，不能暗中重建另一种 G。
    for key in interface_keys:
        # 不一致时应先讨论新的实验变量。
        if public_config.get(key) != hifigan_config.get(key):
            # 显示具体冲突的配置字段。
            raise RuntimeError(f'Public/custom vocoder interface differs at {key}')
    # 记录原 G 的内容，仅作为替换前身份，不把它当作当前冻结声码器。
    custom_state_sha = module_fingerprint(vocoder)
    # 公开文件默认读到 CPU，不新建 Generator，因此不会再次随机初始化网络。
    public_checkpoint = training.load_torch_checkpoint(PUBLIC_WEIGHT)
    # 公开权重应包含生成器状态字典，不能接受其他对象类型。
    if not isinstance(public_checkpoint, dict) or not isinstance(public_checkpoint.get('generator'), dict):
        # 缺失时停止，不能尝试随机 G。
        raise RuntimeError('Public VCTK_V1 file lacks its generator state dictionary')
    # 保持原权重归一化参数，严禁 remove_weight_norm 后再换权重。
    if not any(name.endswith('weight_g') for name in vocoder.state_dict()) or not any(name.endswith('weight_v') for name in vocoder.state_dict()):
        # 没有这些字段意味着加载流程已经改变，不能继续原位对照。
        raise RuntimeError('Expected the original weight-normalized generator object')
    # 在现有 G 对象中严格替换所有权重，键名和形状必须完全匹配。
    vocoder.load_state_dict(public_checkpoint['generator'], strict=True)
    # 公开 G 同样保持冻结，不增加任何声码器训练步骤。
    vocoder.requires_grad_(False)
    # 固定推理行为，保持两次波形重算的一致性。
    vocoder.eval()
    # 非有限的公开权重不能进入学生训练。
    if any(not bool(torch.isfinite(value).all()) for value in vocoder.state_dict().values()):
        # 不尝试在坏权重上修复或继续。
        raise RuntimeError('Non-finite public vocoder weights')
    # 当前冻结 G 的真实内容身份与 custom 的身份分别保存。
    public_state_sha = module_fingerprint(vocoder)
    # 只测 mel 输入梯度，检查内部会恢复包括裁剪在内的随机状态。
    probe_report = check_public_vocoder_gradient(vocoder, train_loader.dataset, mean, std, device)
    # 原位加载、检查和哈希都不得改变之后的样本顺序与噪声流。
    training.restore_rng_state(startup_rng, device)
    # 再按内容确认恢复完成，不能只假定恢复函数被调用就足够。
    if rng_fingerprint(training.capture_rng_state()) != startup_rng_sha:
        # 不带着偏移的随机序列开始单变量对照。
        raise RuntimeError('Startup RNG was not preserved across the public vocoder swap')
    # 初始学生和判别器指纹可与后续检查核对；公开 G 在两轮中始终固定。
    initial_models = {name: module_fingerprint(model) for name, model in (('teacher', teacher), ('student', student), ('discriminator', discriminator), ('active_public_vocoder', vocoder))}
    # 新契约明确声明当前 public G，没有为它生成 custom 数据验真证明。
    contract = {
        'schema_version': 1,  # 独立诊断契约，不冒充原正式训练契约。
        'experiment': 'Isolated student training diagnostic with frozen public HiFi-GAN VCTK_V1',  # 本次对照的准确名称。
        'actual_plan': {'epochs': DIAGNOSTIC_EPOCHS, 'batches_per_epoch': BATCHES_PER_EPOCH, 'maximum_student_updates': DIAGNOSTIC_EPOCHS * BATCHES_PER_EPOCH, 'resume_enabled': False, 'formal_100_epoch_training': False},  # 明确两轮上限。
        'single_changed_training_component': 'Frozen vocoder generator parameters only; student, D, data, objective, optimizers, seed and startup RNG order retained.',  # 区分权重变量与额外记录工具。
        'baseline': {'path': str(BASELINE_PATH), 'file_sha256': BASELINE_FILE_SHA, 'canonical_sha256': BASELINE_CANONICAL_SHA, 'full_rebuilt_contract_matches': True},  # 原实验身份不被改写。
        'active_frozen_vocoder': {'label': 'public VCTK_V1', 'checkpoint': str(PUBLIC_WEIGHT), 'checkpoint_sha256': PUBLIC_WEIGHT_SHA, 'config': str(PUBLIC_CONFIG), 'config_sha256': PUBLIC_CONFIG_SHA, 'state_content_sha256': public_state_sha, 'training_step': None, 'training_step_status': 'not provided in the verified public metadata', 'training_data_fingerprints': None, 'overlap_with_our_unseen_split': 'unknown', 'is_verified_exact_FastVoiceGrad_authors_checkpoint': False, 'parameters_trainable': False},  # 公开信息缺失处如实留空。
        'custom_dependency_used_only_for_startup_and_baseline_validation': {'checkpoint': str(original_args.vocoder_checkpoint), 'attestation': str(original_args.vocoder_attestation), 'state_before_swap_sha256': custom_state_sha, 'custom_training_updates_in_this_run': 0, 'attestation_applies_to_public_vocoder': False},  # 不把原验真记录套到公开 G。
        'training_config_reference': config.as_dict(),  # 原数学配置完整保留；epochs100和expected_vocoder_step只是此参考对象的历史字段。
        'training_config_reference_caveat': 'The reference config retains baseline epochs=100 and expected_custom_vocoder_step=2500000 for exact baseline validation. Actual execution is two diagnostic epochs; active public vocoder step is unknown. Those fields do not describe the active public generator or extend this run.',  # 消除复用配置对象造成的解释歧义。
        'data': rebuilt_baseline['data'],  # 保留原数据完整身份；实际更新预算以上方 actual_plan 为准。
        'runtime': {**rebuilt_baseline['runtime'], 'python': sys.version, 'python_executable': str(Path(sys.executable).resolve()), 'driver_path': str(Path(__file__).resolve()), 'driver_sha256': sha256_file(Path(__file__).resolve())},  # 同时登记额外 driver 的来源。
        'initial_model_fingerprints': initial_models,  # 第零轮的共同起点证据。
        'initial_rng_content_sha256': startup_rng_sha,  # 证明额外检查未偏移随机流。
        'historical_replay_limit': 'The historical run did not archive its actual D0 or per-batch feature digests. This run reproduces the audited initialization/RNG order, but cannot claim an after-the-fact tensor-by-tensor or batch-by-batch verification of that historical run.',  # 如实区分初始化重放和旧现场状态的直接证明。
        'public_source_reference': 'https://github.com/jik876/hifi-gan#pretrained-model',  # 官方公开预训练模型入口；VCTK_V1与UNIVERSAL_V1是不同文件。
        'public_vocoder_input_gradient_check': probe_report,  # 只读连通性探针结果。
        'held_out_test_usage': 'Fixed 100-utterance self-reconstruction loss diagnostics only; no gradient updates, checkpoint selection or claimed formal conversion metrics.',  # 测试只观察、不训练。
        'monitoring': {'window_steps': config.loss_monitor_window_steps, 'uses_original_LossMonitor': True, 'rise_action': 'record and alert without changing the objective', 'signal_stop': 'stop after current complete training batch; keep last complete epoch checkpoint'},  # 报警和停止规则。
    }
    # 将基线的完整一百轮数据预算显式改写为本次实际诊断预算，原值仍可在原契约追溯。
    contract['data'] = {**contract['data'], 'total_student_updates': DIAGNOSTIC_EPOCHS * BATCHES_PER_EPOCH}
    # 新检查点只绑定新的诊断契约，不可能和原 custom 的契约哈希混用。
    contract_sha = training.canonical_json_sha256(contract)
    # 预检查到此结束；不创建新训练目录，也不执行 train_epoch。
    if cli.preflight_only:
        # 独立报告说明“检查通过、训练尚未启动”。
        training.atomic_write_json(PREFLIGHT_REPORT, {'status': 'preflight_passed_no_training', 'diagnostic_contract_sha256': contract_sha, 'contract': contract})
        # 主流程可查看预检查报告后再决定是否正式执行这个两轮入口。
        print('PUBLIC_VOCODER_PREFLIGHT_PASSED_NO_TRAINING ' + str(PREFLIGHT_REPORT), flush=True)
        # 退出时锁自动释放，不产生学生更新。
        return
    # 正式两轮诊断的输出目录只允许首次建立。
    OUTPUT.mkdir(parents=True, exist_ok=False)
    # 写清当前 public G 身份，不生成误导性的 custom attestation。
    training.atomic_write_json(OUTPUT / 'diagnostic_contract.json', contract)
    # 第零轮没有训练损失，单独保存初始化，不用原 save_checkpoint 编造零 loss。
    training.atomic_torch_save({'checkpoint_type': 'diagnostic_initial_state', 'epoch': 0, 'global_step': 0, 'run_contract_sha256': contract_sha, 'student': student.state_dict(), 'discriminator': discriminator.state_dict(), 'student_optimizer': student_optimizer.state_dict(), 'discriminator_optimizer': discriminator_optimizer.state_dict(), 'rng_state': training.capture_rng_state(), 'initial_model_fingerprints': initial_models, 'active_public_vocoder_checkpoint_sha256': PUBLIC_WEIGHT_SHA, 'epoch_losses': None}, OUTPUT / 'epoch_0000_initial_state.pt')
    # 初始化记录也留下内容哈希，方便不加载大权重就先核对身份。
    training.atomic_write_json(OUTPUT / 'initial_state_manifest.json', {'epoch': 0, 'global_step': 0, 'model_fingerprints': initial_models, 'rng_content_sha256': startup_rng_sha, 'checkpoint_sha256': sha256_file(OUTPUT / 'epoch_0000_initial_state.pt')})
    # 附加保存和哈希不应影响训练 RNG，启动第一批前恢复已核对的共同起点。
    training.restore_rng_state(startup_rng, device)
    # 轻量信号标记由监控器在完整批次边界处理。
    stop_requested = {'value': False}
    # 信号处理不执行模型操作，防止打断某个梯度计算后伪造完整断点。
    def request_stop(signum, frame):
        # 通知下一次完成批次后的 monitor.observe 停止。
        stop_requested['value'] = True
        # 立即在日志声明不会继续下一批。
        print(f'DIAGNOSTIC_STOP_REQUESTED signal={signum}; stopping after the current complete batch.', flush=True)
    # 用户 Ctrl+C 与系统 SIGTERM 使用同一停止方式。
    signal.signal(signal.SIGINT, request_stop)
    # 默认终止请求不会等整轮结束后才处理。
    signal.signal(signal.SIGTERM, request_stop)
    # 保持原监控窗口和报警定义，输出位于全新目录。
    monitor = DiagnosticMonitor(OUTPUT, config.loss_monitor_window_steps, stop_requested)
    # 新训练始于初始化；此 driver 没有读取任何旧学生优化器状态。
    global_step, completed_epoch = 0, 0
    # 全部模型训练与测试都复用原低层实现。
    try:
        # 硬上限两轮；config 中的基线 epochs=100 不参与这个循环。
        for epoch in range(1, DIAGNOSTIC_EPOCHS + 1):
            # 每轮开始都在日志重复明确当前是最多两轮的隔离诊断。
            print(f'PUBLIC_VOCODER_DIAGNOSTIC epoch={epoch}/{DIAGNOSTIC_EPOCHS} start_step={global_step}', flush=True)
            # 原 train_epoch 内部复用原 train_batch、完整 batch32、独立 D/S Adam 和相同损失。
            train_losses, global_step = training.train_epoch(train_loader, epoch, global_step, teacher, student, vocoder, discriminator, diffusion, student_optimizer, discriminator_optimizer, mean, std, device, config, monitor=monitor)
            # 每轮完整步数必须符合固定预算，不能静默少跑或多跑。
            if global_step != epoch * BATCHES_PER_EPOCH:
                # 步数不符时不记录为正常完整轮次。
                raise RuntimeError('Diagnostic epoch completed with an unexpected step count')
            # 原评估函数关闭梯度并恢复训练 RNG，不用测试集做参数更新。
            test_losses, test_examples, test_batches = training.evaluate_test_epoch(test_loader, teacher, student, vocoder, discriminator, diffusion, mean, std, device, config)
            # 公开 G 经过训练批次和固定测试后必须仍逐字节不变。
            if module_fingerprint(vocoder) != public_state_sha:
                # 冻结模型变化会破坏对照设计，立即停止。
                raise RuntimeError('Frozen public vocoder changed during student diagnostic')
            # 按原 schema 合并七项训练和测试损失及覆盖数量。
            losses = training.make_epoch_loss_record(train_losses, test_losses, test_examples, test_batches)
            # 每轮 CSV 使用原写入函数，方便和 custom 基线按列比较。
            training.append_loss_csv(OUTPUT / 'loss.csv', epoch, global_step, losses)
            # 保存每一轮完整 S/D/两个 Adam/RNG，而非只有学生权重。
            training.save_checkpoint(OUTPUT / f'epoch_{epoch:04d}.pt', student, discriminator, student_optimizer, discriminator_optimizer, epoch, global_step, contract_sha, losses, include_training_state=True)
            # latest 也只位于新目录，不覆盖原修复训练的任何文件。
            training.save_checkpoint(OUTPUT / 'latest.pt', student, discriminator, student_optimizer, discriminator_optimizer, epoch, global_step, contract_sha, losses, include_training_state=True)
            # 画图函数会恢复 RNG，不改变下一轮数据打乱或噪声。
            training.render_loss_plots_without_rng_side_effects(OUTPUT / 'loss.csv', OUTPUT, device)
            # 只有完整损失、检查点和图表都已保存，才更新已完成轮次。
            completed_epoch = epoch
            # 原训练配置中学习率恒定；这里直接记录完整分项供外部监控。
            print('PUBLIC_VOCODER_EPOCH_COMPLETE ' + json.dumps({'epoch': epoch, 'global_step': global_step, **losses}), flush=True)
            # 若停止请求发生在测试或保存期间，不启动下一轮训练。
            if stop_requested['value']:
                # 已有完整轮次文件，按停止状态结束。
                raise RequestedDiagnosticStop('Stopped after saving the current complete diagnostic epoch')
    # 用户停止与数值/代码错误分别记录，避免混淆。
    except RequestedDiagnosticStop:
        # 监控器保存最近完整训练批的实际步数，可能处于尚未保存的半轮。
        stopped_step = getattr(monitor, 'last_global_step', global_step)
        # 结束标记不冒充两轮已经全部完成。
        monitor.finish('stopped', stopped_step, getattr(monitor, 'last_epoch', completed_epoch))
        # 单独说明最后完整可用轮次，不把内存中的半轮状态当作正式断点。
        training.atomic_write_json(OUTPUT / 'completion.json', {'status': 'stopped', 'observed_step': stopped_step, 'last_complete_checkpoint_epoch': completed_epoch, 'diagnostic_contract_sha256': contract_sha})
        # 停止后直接退出，没有自动续训路径。
        return
    # 其他异常需要保留原 traceback，方便主流程定位问题。
    except BaseException:
        # 包括数值异常在内的失败都写入独立监控状态。
        monitor.finish('failed', getattr(monitor, 'last_global_step', global_step), getattr(monitor, 'last_epoch', completed_epoch))
        # 不吞掉错误，也不自动重试训练。
        raise
    # 两轮结束只表示这次短期对照结束，与正式一百轮是否完成无关。
    monitor.finish('diagnostic_completed', global_step, completed_epoch)
    # 保存明确的实际执行预算和完成状态。
    training.atomic_write_json(OUTPUT / 'completion.json', {'status': 'diagnostic_completed', 'completed_epochs': completed_epoch, 'global_step': global_step, 'formal_100_epoch_training_complete': False, 'diagnostic_contract_sha256': contract_sha})
    # 外部监控可据此停止等待，不会继续创建第三轮。
    print('PUBLIC_VOCODER_DIAGNOSTIC_COMPLETED_TWO_EPOCHS', flush=True)


# 导入此文件只定义入口；不会自动开始训练。
if __name__ == '__main__':
    # 必须由主流程在指定 myconda 环境审核后明确执行。
    main()
