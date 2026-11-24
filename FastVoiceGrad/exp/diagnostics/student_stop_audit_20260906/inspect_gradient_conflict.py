# 停训后的梯度诊断：只测量损失对学生输出的方向，不创建优化器，不改变任何模型权重。
"""Read-only output-space gradient audit of the stopped student."""
import argparse  # 只允许选择固定样本数量和判别器测量小批大小。
import gc  # 每测完一个学生状态，释放不再需要的计算图。
import json  # 打印简短的测量进度，完整证据另存为 JSON。
import random  # 重放原 pilot 的同一批随机裁剪。
import subprocess  # 只读记录现场 Git 版本。
import time  # 记录诊断时间，区分不同来源的结果。
from dataclasses import replace  # 从正式配置派生诊断配置，不修改正式默认值。
import torch  # 自动微分只计算输入梯度；所有运算只能在租用服务器执行。
from torch.utils.data import default_collate  # 将原来 32 句按原次序组合成批次。

# 导入同目录前向检查的组件和读取函数；这里不会调用它的 main，也不会启动其诊断。
from inspect_stopped_student import (
    ROOT,  # 权威项目根目录固定为租用服务器的 /root/FsatVoiceGrad。
    assert_rented_server,  # 拒绝在本地电脑或其他服务器运行。
    acquire_runtime_locks,  # 避免与训练或另一项正式作业并发占用 GPU。
    PAPER_STUDENT_CONFIG,  # 复用已登记的教师、学生和声码器契约。
    VoiceGradDataset,  # 复用正式的归一化、BNF 对齐和裁剪读取。
    VoiceGradDiffusion,  # 使用正式的扩散系数和时间步索引。
    load_teacher_and_student,  # 严格读取共同的教师初始化状态。
    load_frozen_vocoder,  # 读取冻结声码器并校验身份。
    build_paper_discriminator,  # 构建与完整第二轮相同的判别器结构。
    load_mel_statistics,  # 使用原训练集的 mel 均值与标准差。
    load_torch_checkpoint,  # 只读加载已停止实验的完整保存点。
    denormalize_mel,  # 给声码器前恢复其接收的 log-mel 数值尺度。
    make_student_prediction,  # 以固定第 950 步噪声进行学生一步预测。
    score_distillation_loss,  # 测量当前 clean_x0、sqrt_alpha_bar 蒸馏目标。
    seed_everything,  # 保持加载组件时的随机初始化顺序一致。
    sha256_file,  # 保存使用的权重文件身份。
    tensor_sha,  # 保存实际裁剪、噪声与 BNF 的字节身份。
    write_json,  # 只写本次独立诊断文件，不改训练记录。
    mel_measure,  # 同时保留固定 GT 下的逐句重建误差。
)


# 检查一个输入梯度，避免把 NaN 或 Inf 的夹角当成可靠证据。
def checked_gradient(loss, leaf, *, retain_graph=False):
    # autograd.grad 返回梯度张量，不把结果累积到任何参数的 .grad 中。
    gradient, = torch.autograd.grad(loss, leaf, retain_graph=retain_graph)
    # 输入方向必须是有限数，否则本次诊断应立即报错。
    if not bool(torch.isfinite(gradient).all()):
        # 终止当前只读诊断；没有优化器，所以不会留下被更新的模型。
        raise RuntimeError('Non-finite input gradient in read-only audit')
    # 不保存高阶导数图，避免额外占用显存。
    return gradient.detach()


# 把梯度夹角转成易读数字；有一边恰好为零时没有定义方向，返回空值。
def cosine(left, right):
    # double 只用于小张量统计，减少很小梯度做点积时的舍入误差。
    a = left.detach().reshape(-1).double()
    # 第二个梯度采用同样的展开顺序，保证元素一一对应。
    b = right.detach().reshape(-1).double()
    # 夹角分母由两个向量长度相乘得到。
    denominator = a.norm() * b.norm()
    # 零梯度不能提供方向信息，不能把它伪装成垂直或一致。
    if float(denominator) == 0.0:
        # JSON 中的 null 表示本次夹角不可定义。
        return None
    # 正值代表两个损失的局部下降方向较一致，负值代表存在局部冲突。
    return float(torch.dot(a, b) / denominator)


# 每个损失同时报告数值大小、整体梯度长度和逐句方向，避免只看夹角漏掉量级差异。
def describe_gradient(gradient, gt_l1_gradient, gt_mse_gradient):
    # batch 第一维对应语句；展开其余频率和时间维后逐句测量。
    rows = gradient.detach().flatten(1).double()
    # 下方所有量只描述标准化 mel 输出空间，不能当成 Adam 参数更新幅度。
    return {
        'l2_norm': float(rows.norm()),  # 整个固定批次的梯度长度。
        'rms': float(rows.square().mean().sqrt()),  # 每个 mel 元素的平均梯度量级。
        'mean_abs': float(rows.abs().mean()),  # 不受正负抵消影响的平均幅度。
        'max_abs': float(rows.abs().max()),  # 检查是否有少量极大梯度元素。
        'cosine_to_gt_l1': cosine(gradient, gt_l1_gradient),  # 与真实 mel 的 L1 下降方向是否一致。
        'cosine_to_gt_mse': cosine(gradient, gt_mse_gradient),  # 与更强调大误差的平方误差方向是否一致。
        'per_sample_l2_norm': rows.norm(dim=1).cpu().tolist(),  # 保留每句的贡献，便于结合其随机 t 检查。
        'per_sample_cosine_to_gt_l1': [cosine(g, r) for g, r in zip(gradient, gt_l1_gradient)],  # 防止平均值掩盖个别句子冲突。
        'per_sample_cosine_to_gt_mse': [cosine(g, r) for g, r in zip(gradient, gt_mse_gradient)],  # 对同一句同时使用平方误差参照。
    }


# 将 8 个判别器分支逐一测量，再通过冻结声码器把波形梯度拉回学生 mel 输出。
def waveform_objective_gradients(y, reference, vocoder, discriminator, mean, std, microbatch):
    # 生成波形只保留“学生输出 y → 冻结声码器 → 波形”的图。
    fake_waveform = vocoder(denormalize_mel(y, mean, std))
    # 真实参照由同一个冻结声码器合成，不需要反向图。
    with torch.no_grad():
        # 这和正式训练中的 V(x0) 参照相同，没有改成原始录音。
        real_waveform = vocoder(denormalize_mel(reference, mean, std))
    # 分别积累对抗项与未乘 2 的特征匹配项在波形空间的梯度。
    adversarial_wave_gradient = torch.zeros_like(fake_waveform)
    # 两项分开保存，后面才能分辨冲突方向和相对量级。
    matching_wave_gradient = torch.zeros_like(fake_waveform)
    # 测量值按完整固定批次平均，不能让小批大小改变损失尺度。
    adversarial_value = 0.0
    # 特征匹配值保持正式训练的“每层平均、各层各分支相加”定义。
    matching_value = 0.0
    # 总句数用于最后一个不足 microbatch 的小批正确加权。
    count = y.shape[0]
    # 只在判别器内拆小批；无跨样本层时与整批平均完全相同。
    for start in range(0, count, microbatch):
        # 当前小批终点不能越过固定样本数量。
        stop = min(start + microbatch, count)
        # 当前小批占完整批次的比例，用于还原原始平均损失和梯度。
        sample_weight = (stop - start) / count
        # 波形代理与声码器图分离；每个分支结束即可释放它自己的特征图。
        fake_proxy = fake_waveform[start:stop].detach().requires_grad_(True)
        # 真实波形只需当前小批，且本来就不含反向图。
        real_chunk = real_waveform[start:stop]
        # MPD 和 MRD 的 8 个分支分别前向，避免同时存储全部中间特征。
        for branch_index, branch in enumerate(discriminator.loss_branches()):
            # 真实分支只提供固定目标特征，不接收梯度。
            with torch.no_grad():
                # 判别器分数在本项未使用，中间特征用于逐层配对。
                real_scores, real_features = branch(real_chunk)
            # 假波形分支保留对 fake_proxy 的梯度，但判别器参数已冻结。
            fake_scores, fake_features = branch(fake_proxy)
            # 层数必须一致，否则 zip 会悄悄漏掉某些层，改变正式目标。
            if len(real_features) != len(fake_features) or not real_features:
                # 此处是诊断一致性检查，不会尝试修改判别器。
                raise RuntimeError(f'Unexpected feature layout in branch {branch_index}')
            # 正式学生对抗损失要求生成波形的各位置分数靠近 1。
            adversarial = (fake_scores - 1.0).square().mean() * sample_weight
            # 与论文公式对应，每层先按元素平均，随后相加；这里尚未乘 lambda_FM=2。
            matching = sum((r - f).abs().mean() for r, f in zip(real_features, fake_features)) * sample_weight
            # 同一假分支特征还要用于特征匹配，因此第一次求导暂时保留这张图。
            adversarial_wave_gradient[start:stop] += checked_gradient(adversarial, fake_proxy, retain_graph=True)
            # 第二次求导后立即释放该分支图，避免累积 8 份特征图。
            matching_wave_gradient[start:stop] += checked_gradient(matching, fake_proxy)
            # 损失仅保留普通数字，防止无意保留前向图。
            adversarial_value += float(adversarial.detach())
            # 该值用于和原固定判别器测量交叉核对。
            matching_value += float(matching.detach())
            # 删除分支特征和临时损失，下一分支复用显存。
            del real_scores, real_features, fake_scores, fake_features, adversarial, matching
        # 当前波形代理无需跨小批保存。
        del fake_proxy, real_chunk
    # 先以对抗项的波形梯度进行一次声码器向量-Jacobian 乘积，得到对 y 的梯度。
    adversarial_y_gradient, = torch.autograd.grad(fake_waveform, y, grad_outputs=adversarial_wave_gradient, retain_graph=True)
    # 再拉回特征匹配的波形梯度；这次结束后释放声码器图。
    matching_y_gradient, = torch.autograd.grad(fake_waveform, y, grad_outputs=matching_wave_gradient)
    # 两次结果都应是有限数，避免无效夹角进入最终报告。
    if not bool(torch.isfinite(adversarial_y_gradient).all() and torch.isfinite(matching_y_gradient).all()):
        # 没有优化器或参数更新，所以失败不会改变训练状态。
        raise RuntimeError('Non-finite waveform-to-mel gradient')
    # 返回原损失数值和分开的输出梯度，正式权重将在外面明确乘上。
    return adversarial_value, matching_value, adversarial_y_gradient.detach(), matching_y_gradient.detach()


# 所有模型运算都放在 main 内，导入本文件只定义辅助函数。
def main():
    # 命令行只能改变只读诊断规模，不提供训练步数或学习率参数。
    parser = argparse.ArgumentParser(description='Read-only output-space gradient conflict audit; never trains.')
    # 默认测原固定 32 句的前 8 句；也可完整测 32 句。
    parser.add_argument('--samples', type=int, choices=(8, 32), default=8)
    # 小批只用于降低判别器特征图显存，不改变总体平均权重。
    parser.add_argument('--wave-microbatch', type=int, default=4)
    # 解析实际审核后传入的选项。
    args = parser.parse_args()
    # 非正小批大小无法枚举样本，应在加载模型前停止。
    if args.wave_microbatch < 1:
        # 明确诊断配置错误，避免启动任何计算。
        raise ValueError('--wave-microbatch must be positive')
    # 只允许运行在当前租用服务器，遵守项目计算边界。
    assert_rented_server(ROOT)
    # 锁对象一直由 main 持有到退出，保证诊断期间没有并行正式训练。
    locks = acquire_runtime_locks(ROOT)
    # 所有结果独立于训练断点和旧训练日志。
    out = ROOT / 'exp/diagnostics/student_stop_audit_20260906'
    # 新结果使用独立名称，不覆盖已有前向 results.json。
    result_path = out / 'gradient_results.json'
    # 不覆盖前一次已完成的梯度证据；若要重测，应先由主流程选择新目录。
    if result_path.exists():
        # 主流程必须明确处理旧产物后才能再次保存同名结果。
        raise RuntimeError('Refusing to overwrite completed gradient diagnostic')
    # 只创建本次诊断结果目录，不接触任何训练产物。
    out.mkdir(parents=True, exist_ok=True)
    # 控制 CPU 线程数量，模型计算仍只在租用 GPU 上运行。
    torch.set_num_threads(4)
    # 当前任务只允许租用机的 GPU，设备不提供外部覆盖选项。
    device = torch.device('cuda')
    # 目标和权重明确与已停止的修复训练保持一致。
    config = replace(PAPER_STUDENT_CONFIG, score_target='clean_x0', score_weight='sqrt_alpha_bar', num_workers=0)
    # 读取组件前重放同一随机初始化顺序，方便核对身份。
    seed_everything(config.seed)
    # 共同起点来自原本 500 epoch 的教师保存点。
    teacher_path = ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt'
    # 学生此刻完整复制教师参数；后面第一次测量直接使用这一状态。
    teacher, student, _ = load_teacher_and_student(teacher_path, config, device)
    # 声码器沿用正式自训 HIFIGAN 的最终保存点。
    vocoder_dir = ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen'
    # 复用原检查的读取和身份验证方法，不绕开最终权重证明。
    vocoder, vocoder_config, _ = load_frozen_vocoder(
        ROOT,  # 声码器实现从当前租用服务器项目读取。
        vocoder_dir / 'generator_step_2500000.pt',  # 固定 250 万步生成器权重。
        vocoder_dir / 'config.json',  # 固定声码器结构和输入配置。
        vocoder_dir / 'final_checkpoint_attestation_comments_20260906.json',  # 注释等价性校验后的身份文件。
        config,  # 保持正式实验的组件契约。
        device,  # 只在租用机 GPU 计算。
    )
    # 判别器结构先正常构建，随后完全载入已停止实验的 D2 参数。
    discriminator = build_paper_discriminator(ROOT, vocoder_config, config, device)
    # 使用原实验的余弦扩散日程，不能在梯度诊断中另改时间步定义。
    diffusion = VoiceGradDiffusion(config.diffusion_steps, offset=config.cosine_offset).to(device)
    # 声码器反归一化使用同一份训练集统计。
    mean, std = load_mel_statistics(ROOT / 'data', device)
    # 固定保存点来自已停止的 clean_x0 + sqrt_alpha_bar 重训。
    run = ROOT / 'exp/checkpoints/fastvoicegrad_vctk_text_unseen_repair_x0_sqrt_20260906'
    # 只读加载完整第二轮保存点，不尝试恢复训练。
    checkpoint = load_torch_checkpoint(run / 'latest.pt')
    # 本诊断明确比较 epoch 2、step 2460，防止文件后来改变却沿用旧标签。
    if checkpoint['epoch'] != 2 or checkpoint['global_step'] != 2460:
        # 状态不符应先人工确认，不猜测新的保存点含义。
        raise RuntimeError('Expected the stopped epoch-2, step-2460 checkpoint')
    # 两个学生状态都由完全相同的 D2 衡量，排除判别器自身变化。
    discriminator.load_state_dict(checkpoint['discriminator'], strict=True)
    # 教师、学生、声码器、判别器都切换为只读推理模式。
    for model in (teacher, student, vocoder, discriminator):
        # 冻结所有参数；只有后面新建的输出叶子 y 允许求梯度。
        model.requires_grad_(False)
        # 关闭训练态行为，避免运行统计等缓冲量被更新。
        model.eval()
    # 判别器微批需要各句独立；如果出现批归一化，先停止并重新核对等价性。
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in discriminator.modules()):
        # 当前正式 MPD/MRD 不应含此类跨样本层。
        raise RuntimeError('Unexpected BatchNorm: microbatch equivalence must be reviewed')
    # 原 120 步 pilot 的清单只作数据来源，不运行其中的任何优化代码。
    pilot_dir = ROOT / 'exp/diagnostics/acdd_repair_20260906'
    # 保留原来 32 个固定语句索引和顺序。
    manifest = json.loads((pilot_dir / 'diagnostic_manifest.json').read_text())
    # 如果清单发生变化，不能仍声称精确重放原来的 32 句。
    if len(manifest['dev_indices']) != 32:
        # 明确停止本次诊断，不擅自截取另一批索引。
        raise RuntimeError('Expected exactly 32 original pilot dev indices')
    # 数据处理与原固定检查完全相同，片段长度为 128 帧。
    dataset = VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    # 原 pilot 的第一次随机裁剪使用此种子。
    random.seed(1234)
    # 先读取全部 32 句，确保前 8 句与完整前向诊断的裁剪一致。
    full_batch = default_collate([dataset[i] for i in manifest['dev_indices']])
    # 原随机数流按完整 32 句的形状消费，不能先改成 8 句再生成噪声。
    full_shape = full_batch['mel'].shape
    # 三个独立流分别控制学生输入、教师再加噪和教师时间步。
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in (55001, 66002, 77003)]
    # 第一份完整噪声固定学生从第 950 步开始的输入。
    full_source_noise = torch.randn(full_shape, device=device, generator=generators[0])
    # 第二份完整噪声独立用于教师目标，保持 q(x_t|y) 的定义。
    full_score_noise = torch.randn(full_shape, device=device, generator=generators[1])
    # 教师时间步按原方案在 1 至 1000 均匀抽取，共 32 个后再截取。
    full_timesteps = torch.randint(1, 1001, (full_shape[0],), device=device, generator=generators[2])
    # 先切数据再搬到 GPU，不改变原来的归一化或条件维度。
    reference, speaker, bnf = [full_batch[key][:args.samples].to(device) for key in ('mel', 'spk_emb', 'bnf')]
    # 学生输入噪声使用完整随机数流中的同一前缀。
    source_noise = full_source_noise[:args.samples]
    # 教师噪声也使用原流的同一前缀。
    score_noise = full_score_noise[:args.samples]
    # 每句保留原随机时间步，两个模型直接复用该张量。
    timesteps = full_timesteps[:args.samples]
    # 记录有限样本诊断范围，不能把结果冒充正式跨说话人性能评价。
    result = {
        'time': time.time(),  # 诊断开始时间。
        'purpose': 'Read-only output-space local gradient audit; no optimizer, no parameter updates.',  # 明确没有训练。
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),  # 保存当前代码身份。
        'checkpoint_epoch': checkpoint['epoch'],  # D2 和第二个学生的完整训练轮次。
        'checkpoint_step': checkpoint['global_step'],  # 完整保存点步数，不冒充中断前未保存步数。
        'checkpoint_sha256': sha256_file(run / 'latest.pt'),  # 防止同名权重混淆。
        'run_contract_sha256': checkpoint['run_contract_sha256'],  # 保留原实验契约身份。
        'samples': args.samples,  # 实際用于梯度测量的固定语句数量。
        'wave_microbatch': args.wave_microbatch,  # 记录为省显存采用的判别器小批大小。
        'critic': 'fixed_epoch2',  # 两个模型共享一个完全冻结的判别器。
        'score_target': 'clean_x0',  # 教师直接估计干净 mel，而非相邻时刻状态。
        'score_weight': 'sqrt_alpha_bar',  # 蒸馏值已包含时间步权重，外部再乘 45。
        'source_noise_seed': 55001,  # 完整 32 句学生输入噪声种子。
        'score_noise_seed': 66002,  # 完整 32 句教师再加噪种子。
        'timestep_seed': 77003,  # 完整 32 句时间步种子。
        'input_identities': {
            'full32_mel_sha256': tensor_sha(full_batch['mel']),  # 和前向 results.json 的整批身份对照。
            'full32_bnf_sha256': tensor_sha(full_batch['bnf']),  # 核对原始条件没有变化。
            'full32_source_noise_sha256': tensor_sha(full_source_noise),  # 核对完整随机流而非只核对前 8 句。
            'full32_score_noise_sha256': tensor_sha(full_score_noise),  # 明确教师使用的独立固定噪声。
            'full32_timesteps': full_timesteps.cpu().tolist(),  # 留下完整随机时间步顺序。
            'speakers': list(full_batch['spk_name'][:args.samples]),  # 本次实际被测说话人。
            'utterances': list(full_batch['utt_id'][:args.samples]),  # 本次实际被测语句编号。
            'selected_indices': manifest['dev_indices'][:args.samples],  # 可追溯到数据集索引。
        },
        'score_timesteps': timesteps.cpu().tolist(),  # 用于查看每句蒸馏梯度为何量级不同。
        'score_weights': diffusion.sqrt_alphas_cumprod[timesteps].cpu().tolist(),  # 对应每句的实际 √ᾱt。
        'limitations': [
            'These are fixed training-source self-reconstruction excerpts, not formal cross-speaker unseen evaluation.',  # 不扩大诊断结论范围。
            'Cosines concern gradients at the same mel output y; student Jacobian and Adam preconditioning are not measured.',  # 不把输出空间方向当成真实参数更新。
            'Positive dot(g_GT, g_total) predicts a local GT decrease along minus g_total only in output space.',  # 说明点积的准确含义。
            'The teacher target is recomputed at each model output although noise, timesteps and teacher weights are fixed.',  # 固定随机数不等于固定目标。
            'dist_before_lambda includes sqrt(alpha_bar_t); it excludes only the outer lambda_dist=45.',  # 避免误读 raw 蒸馏值。
            'Feature-matching layers and discriminator branches are summed exactly as in the stopped training objective.',  # 明确分支归约口径。
        ],
        'models': {},  # 下方只放两个模型状态的测量结果。
    }
    # 第一组保持刚复制的教师初始化，第二组读取完整第二轮学生权重。
    for label in ('teacher_initialized_one_step', 'repair_epoch2'):
        # 初始化组不必重新加载，避免多余保存大份权重副本。
        if label == 'repair_epoch2':
            # 严格载入当前完整保存点的学生参数，不更新任何保存文件。
            student.load_state_dict(checkpoint['student'], strict=True)
        # 学生只做前向，参数完全没有反向图。
        with torch.no_grad():
            # 两组使用同一源 mel、BNF、说话人和第 950 步噪声。
            prediction, _ = make_student_prediction(student, diffusion, reference, speaker, bnf, 950, source_noise)
        # 新建唯一可导叶子 y，把测量明确限制在学生输出空间。
        y = prediction.detach().requires_grad_(True)
        # 固定 GT L1 为诊断参照，不加入任何训练目标。
        gt_l1 = (y - reference).abs().mean()
        # 固定 GT 平方误差为第二参照，对大误差更敏感。
        gt_mse = (y - reference).square().mean()
        # 两个参照只是求导，不会修改 y 或学生参数。
        gt_l1_gradient = checked_gradient(gt_l1, y)
        # MSE 是另一张简单图，不需要保留 L1 的图。
        gt_mse_gradient = checked_gradient(gt_mse, y)
        # 蒸馏必须调用正式 stop-gradient 实现，不能改成对教师噪声误差直接求导。
        distillation, teacher_target = score_distillation_loss(
            y,  # 唯一可导的是与教师目标作差的学生输出。
            teacher,  # 教师参数和其重建路径保持冻结。
            diffusion,  # 继续使用同一扩散日程。
            timesteps,  # 同一句在两组模型中使用同一个 t。
            score_noise,  # 同一句在两组模型中使用同一份教师噪声。
            speaker,  # 源说话人条件，与原重建蒸馏一致。
            bnf,  # 源语句的同一份内容特征。
            target_mode='clean_x0',  # 明确测当前修复后的干净教师目标。
            weight_mode='sqrt_alpha_bar',  # 明确测当前的累计信号幅度权重。
        )
        # 取得真正的蒸馏输出梯度，教师 Jacobian 不参与其中。
        distillation_gradient = checked_gradient(distillation, y)
        # 由解析式独立核对 stop-gradient 的方向、时间步权重与整体平均分母。
        expected_distillation_gradient = diffusion.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1) * (y.detach() - teacher_target).sign() / y.numel()
        # 不一致意味着梯度语义需要重新检查，不能继续解释夹角。
        if not torch.allclose(distillation_gradient, expected_distillation_gradient, rtol=1e-5, atol=1e-10):
            # 此处只报告失败，不修正代码或启动补训。
            raise RuntimeError('Score-distillation gradient differs from the detached-target analytic expression')
        # 对抗和特征匹配分别经过判别器及冻结声码器，再得到对同一个 y 的完整梯度。
        adv_value, fm_value, adversarial_gradient, matching_gradient = waveform_objective_gradients(y, reference, vocoder, discriminator, mean, std, args.wave_microbatch)
        # 各项使用与正式训练相同的权重，不能比较未乘权重的长度来判断支配关系。
        weighted_gradients = {
            'adv': adversarial_gradient,  # 对抗项外部权重为 1。
            '2fm': matching_gradient * config.lambda_fm,  # lambda_FM 固定为 2。
            '45dist': distillation_gradient * config.lambda_dist,  # lambda_dist 固定为 45。
        }
        # 合成梯度等于正式损失对当前 y 的梯度，但不包含学生网络和 Adam 的变换。
        total_gradient = sum(weighted_gradients.values())
        # 加上合成项和两个 GT 参照，便于完整检查跨项方向。
        all_gradients = {**weighted_gradients, 'total': total_gradient, 'gt_l1': gt_l1_gradient, 'gt_mse': gt_mse_gradient}
        # 统一按相同输出坐标计算所有量级和逐句夹角。
        gradient_rows = {name: describe_gradient(g, gt_l1_gradient, gt_mse_gradient) for name, g in all_gradients.items()}
        # 同一对夹角只保留一次，减少重复字段。
        names = list(all_gradients)
        # 两两夹角只反映局部梯度方向；负值本身不能证明训练失败。
        pairwise = {f'{left}__{right}': cosine(all_gradients[left], all_gradients[right]) for i, left in enumerate(names) for right in names[i + 1:]}
        # 点积保留大小信息：正值表示沿负合成梯度的小步会降低该 GT 误差。
        dot_l1 = float((gt_l1_gradient.double() * total_gradient.double()).sum())
        # 平方误差的对应点积同理，能反映大误差是否受益。
        dot_mse = float((gt_mse_gradient.double() * total_gradient.double()).sum())
        # 单位化只用于说明方向，不实际移动 y。
        total_norm = float(total_gradient.double().norm())
        # 保存本模型的固定重建数值、原损失、分项梯度以及解释所需的限制。
        row = {
            'mel_per_utterance': mel_measure(y.detach(), reference),  # 与前向重建检查交叉核对。
            'losses': {
                'gt_l1': float(gt_l1.detach()),  # 不移动的 GT 参照误差。
                'gt_mse': float(gt_mse.detach()),  # 不移动的 GT 平方误差。
                'adv': adv_value,  # 固定 D2 下的正式对抗损失。
                'fm_before_lambda': fm_value,  # 已按各层各分支相加，尚未乘 2。
                'dist_before_lambda': float(distillation.detach()),  # 已乘时间步权重，尚未乘 45。
                'total': adv_value + config.lambda_fm * fm_value + config.lambda_dist * float(distillation.detach()),  # 与当前正式目标同口径。
            },
            'gradient_statistics': gradient_rows,  # 真正加权后的分项量级和方向。
            'pairwise_cosines': pairwise,  # 检查对抗、特征匹配和蒸馏是否互相冲突。
            'dot_gt_l1_with_total': dot_l1,  # 正值表示输出空间负合成方向局部有利于 GT L1。
            'dot_gt_mse_with_total': dot_mse,  # 正值表示输出空间负合成方向局部有利于 GT MSE。
            'gt_l1_directional_derivative_along_negative_unit_total': -dot_l1 / total_norm if total_norm else None,  # 负值才表示沿这个方向误差下降。
            'gt_mse_directional_derivative_along_negative_unit_total': -dot_mse / total_norm if total_norm else None,  # 同样只是假设的输出空间极小步，不真正更新。
            'sum_term_norms': sum(gradient_rows[name]['l2_norm'] for name in weighted_gradients),  # 与合成范数对照，可查看相互抵消程度。
            'analytic_dist_gradient_max_error': float((distillation_gradient - expected_distillation_gradient).abs().max()),  # 独立解析梯度一致性证据。
            'teacher_target_rms': float(teacher_target.square().mean().sqrt()),  # 查看移动教师目标的量级。
            'prediction_sha256': tensor_sha(y),  # 固定输出的身份记录。
        }
        # 确认 autograd.grad 没有向任何模型参数累积梯度。
        if any(parameter.grad is not None for model in (teacher, student, vocoder, discriminator) for parameter in model.parameters()):
            # 失败时不保存为完整成功结果，避免误称完全只读。
            raise RuntimeError('Unexpected parameter gradient in output-only audit')
        # 将当前模型结果加入同一份独立诊断记录。
        result['models'][label] = row
        # 中间进度采用新文件名，与前向检查的 partial_results.json 分开。
        write_json(out / 'gradient_partial_results.json', result)
        # 简明打印各损失与合成梯度对 GT 的方向，方便主流程查看运行状态。
        print('GRADIENT_RESULT ' + json.dumps({'model': label, 'losses': row['losses'], 'total_cosine_gt_l1': gradient_rows['total']['cosine_to_gt_l1'], 'total_cosine_gt_mse': gradient_rows['total']['cosine_to_gt_mse']}), flush=True)
        # 本轮所有计算图已求导结束；删除大张量，避免下一组保留旧状态。
        del prediction, y, gt_l1, gt_mse, teacher_target, distillation, adversarial_gradient, matching_gradient, distillation_gradient, expected_distillation_gradient, weighted_gradients, all_gradients, total_gradient
        # 回收 Python 容器持有的临时对象；不触碰任何模型权重文件。
        gc.collect()
        # 释放未使用的缓存块，降低下一组诊断的显存峰值。
        torch.cuda.empty_cache()
    # 只有两组都完整完成后才写最终文件，不覆盖任何训练保存点。
    write_json(result_path, result)
    # 明确这次只有梯度测量，没有试训或恢复训练。
    print('READ_ONLY_GRADIENT_DIAGNOSTIC_COMPLETE', flush=True)


# 只有从命令行明确执行本文件时才进入诊断流程。
if __name__ == '__main__':
    # 主流程审核并上传后自行决定执行；本地编写阶段不运行此函数。
    main()
