# 只读方向检查：使用已保存的真实第一步差值，不执行优化器或产生后续训练权重。
"""Probe fixed fractions of an archived update; never call optimizer.step.

The path is theta(s)=theta0+s*(theta1-theta0), in the actual weight-normalized
parameter coordinates. Fractions are fixed before observing the results.
The first training batch and the archived development batch are measured
separately. Only lower first-batch training loss supports a smaller step;
development scores do not select the fraction or establish generalization.
"""

import argparse  # 主线程显式指定新报告目录，不覆盖smoke实验。
import json  # 保存输入身份、方向内积和六个固定比例的读数。
import math  # 检查比例与向量范数，并计算方向余弦。
import sys  # 导入计算库前拒绝Windows本地运行。
import time  # 记录这个有限只读检查的实际耗时。
from pathlib import Path  # 解析服务器绝对路径，防止误写已有权重。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一允许进行项目计算的租用服务器路径。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地仅用于阅读、编辑、上传。
    raise RuntimeError('Run this read-only probe on the registered Matpool rental only')
sys.path.insert(0, str(ROOT))  # 模型定义从已审计的现场提交读取。

import torch  # 模型测量使用租用GPU，所有参数插值在CPU暂存副本中完成。
import run_endpoint_development_v2 as endpoint  # 复用实际失败smoke的同一数据、目标和测量函数。
from fastvoicegrad.components import load_torch_checkpoint  # 只读可信的本项目归档文件。

RUN = ROOT / 'exp/diagnostics/student_development_20260906/endpoint_l1_v2_smoke_1step'  # 已停止的单步来源。
SCALES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)  # 事先固定，不能看开发结果后加密搜索。
EXPECTED = {  # 对本次真实输入文件锁定字节身份，不能换成另一次训练。
    'initial_full.pt': '9d70c10632176461934dbf23cecd0f5dd0b69163c45c362e21bc1b84879998c8',  # 第0步完整归档。
    'student_step_000001.pt': '5c68e465a3b77ce1b46d6b021a0bb0001632a0e841d668f64adda9d5b0c36949',  # 唯一已发生的实际更新。
    'fixed_development_targets.pt': 'd0e2eb0b550a380b432b8c4ee5d3c957aeedca495ff4c8b0460c9ae3b37dbb95',  # 原固定32句及其DDIM目标。
    'run_contract.json': 'b373c7f5b950c1329de3b39a1e542f2a91f4bc115fda07d025de981d5e9172c9',  # smoke配置、来源和目标身份。
    'batch_assets.jsonl': '9dc4de419c34419db27c0b978753ef310bd4d8619da8120e8e9573449e9a1cba',  # 第一次训练使用的32句及资产。
    'step_losses.jsonl': '0c528d838b23200996b91cab395b676a81a91cd7c9f4f14636d68ade9acb4d12',  # 原更新前训练loss。
}
V2_SHA = 'e198a87b6515bde8b6cdbf9ec5e41dd6081e801a4c4170e92c5c4d1e6fc4d933'  # 固定真实smoke驱动代码。


def interpolate_state(initial, updated, scale, parameter_names):
    """Interpolate actual parameters on CPU, preserving both archived states."""
    if not math.isfinite(scale) or not 0 <= scale <= 1:  # 仅检查实际更新线段，不外推未知大步长。
        raise ValueError('Interpolation scale must be finite and in [0,1]')
    if initial.keys() != updated.keys():  # 不能在结构不同的模型之间插值。
        raise ValueError('Archived state keys differ')
    result = {}  # 所有返回值都在新存储中，禁止写入读取到的原始权重张量。
    for name, left in initial.items():  # 保留state_dict原有字段顺序。
        right = updated[name]  # 对应实际训练一步后的同名状态。
        if left.device.type != 'cpu' or right.device.type != 'cpu' or left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('Archive tensors must have matching CPU shape/dtype: ' + name)
        if name not in parameter_names:  # 非参数缓冲不属于Adam步长，必须在这一步没有变化。
            if not torch.equal(left, right):  # 变化的BN缓冲等不能伪装成参数更新的一部分。
                raise ValueError('Non-parameter state changed: ' + name)
            result[name] = left.clone()  # 原样保留未更新的缓冲。
        elif not left.is_floating_point():  # 当前模型参数应为浮点数，不能对整数状态做线性插值。
            raise ValueError('Cannot interpolate a nonfloating parameter: ' + name)
        elif scale == 0:  # 避免0倍差值引入额外运算误差，精确保留初始点。
            result[name] = left.clone()
        elif scale == 1:  # 最后一项必须精确复现已保存的真实第一步。
            result[name] = right.clone()
        else:
            result[name] = (left.double() + scale * (right.double() - left.double())).to(left.dtype)  # 先高精度再回原dtype。
    return result  # 调用者只把副本加载进临时测量模型，不写任何新权重文件。


def training_supported_scales(rows, tolerance=1e-7):  # 只按训练目标判断哪些已测比例确实下降。
    baseline = rows[0]['train']['endpoint_l1']  # s=0是同批、同目标、未更新学生。
    return [row['scale'] for row in rows if row['scale'] > 0 and row['train']['endpoint_l1'] < baseline - tolerance]


def require_close(name, actual, expected, tolerance=1e-7):  # 明确报告重放差异，避免把另一份数据当作原批次。
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:  # CPU/GPU浮点最多允许极小绝对差。
        raise RuntimeError(f'{name} does not replay the archive: {actual} vs {expected}')
    return {'actual': actual, 'archived': expected, 'absolute_difference': abs(actual - expected),
            'bitwise_scalar_equal': actual == expected, 'absolute_tolerance': tolerance}  # 不把近似相等写成精确相同。


def main():  # 此函数只前向测量和一次求梯度，不构建优化器。
    parser = argparse.ArgumentParser(description=__doc__)  # 输出路径必须由本次诊断单独指定。
    parser.add_argument('--output-dir', required=True, type=Path)  # 不会写入smoke或任何正式模型目录。
    args = parser.parse_args()  # 没有训练步数或恢复训练选项。
    endpoint.training.assert_rented_server(ROOT)  # 双重检查主机身份，排除本地及实验室禁用主机。
    locks = endpoint.training.acquire_runtime_locks(ROOT)  # 只有一个GPU流程可以进入本诊断。
    if endpoint.training.git_head(ROOT) != endpoint.EXPECTED_HEAD:  # 要求当前主代码与smoke同版本且干净。
        raise RuntimeError('Source HEAD changed since the archived smoke')
    if (endpoint.platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Use the original myconda training runtime')  # 不能在新版评估环境中重放训练。
    if not torch.cuda.is_available():  # 此真实模型检查已由主线程明确授权使用租用GPU。
        raise RuntimeError('Rental GPU unavailable')
    output = args.output_dir.resolve()  # 先解析，再检查新目录范围。
    if ROOT / 'exp/diagnostics' not in output.parents or output.exists() or RUN == output or RUN in output.parents:
        raise RuntimeError('Use a new diagnostic output directory outside the archived smoke')
    hashes = {str(RUN / name): value for name, value in EXPECTED.items()}  # 固定六份smoke输入。
    hashes[str(Path(endpoint.__file__).resolve())] = V2_SHA  # 同时固定真正被导入的v2模块。
    endpoint.verify_hashes(hashes)  # 先核验再反序列化项目检查点。
    contract = json.loads((RUN / 'run_contract.json').read_text(encoding='utf-8'))  # 只读本次训练身份。
    endpoint.verify_hashes(contract['file_sha256'])  # 全部源码、教师文件、开发数据来源保持不变。
    hashes.update(contract['file_sha256'])  # 写入本次probe的完整依赖表。
    hashes[str(Path(__file__).resolve())] = endpoint.sha256_file(Path(__file__))  # 记录诊断自身实现。
    contract_sha = endpoint.training.canonical_json_sha256(contract)  # 内容哈希与检查点绑定字段一致。
    initial_checkpoint = load_torch_checkpoint(RUN / 'initial_full.pt')  # 所有归档首先只载入CPU。
    updated_checkpoint = load_torch_checkpoint(RUN / 'student_step_000001.pt')  # 读取真实第一步而非推算Adam状态。
    for checkpoint, step in ((initial_checkpoint, 0), (updated_checkpoint, 1)):  # 只接受这两个已知端点。
        if checkpoint['global_step'] != step or checkpoint['run_contract_sha256'] != contract_sha:
            raise RuntimeError('Checkpoint step or contract identity differs from smoke')
    initial, updated = initial_checkpoint['student'], updated_checkpoint['student']  # 保留权重原值供CPU插值。
    dev0, dev1 = initial_checkpoint['development'], updated_checkpoint['development']  # 两端开发误差用于重放校验。
    del initial_checkpoint, updated_checkpoint  # 不使用Adam状态，释放其大体积CPU动量张量。
    batches = [json.loads(line) for line in (RUN / 'batch_assets.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    losses = [json.loads(line) for line in (RUN / 'step_losses.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(batches) != 1 or len(losses) != 1 or batches[0]['step'] != 1 or losses[0]['step'] != 1:
        raise RuntimeError('Expected exactly one completed training batch in the source experiment')
    indices = batches[0]['indices']  # 使用实际发生更新的训练32句，绝不重新抽另一批。
    if len(indices) != 32 or len(set(indices)) != 32 or set(indices) & set(contract['excluded_train_indices']):
        raise RuntimeError('Archived first batch overlaps development or has invalid size')
    endpoint.verify_hashes(batches[0]['file_sha256'])  # 数据文件必须与真实更新时逐字节相同。
    torch.set_num_threads(2)  # 与v2训练的CPU配置一致，不改变TF32或cudnn标志。
    endpoint.training.seed_everything(1234)  # 模型构建沿用相同初始调用顺序。
    device, config = torch.device('cuda'), endpoint.PAPER_STUDENT_CONFIG  # 同一设备与原BNF接口。
    flags = {'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
             'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
    if any(value != contract['runtime'][name] for name, value in flags.items()):  # 不暗中更改计算路径来匹配结果。
        raise RuntimeError('Current backend precision flags differ from the archived training run')
    dataset = endpoint.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100,
                                      bnf_alignment=config.bnf_alignment)  # 复用原归一化、BNF对齐与裁剪。
    batch, current_assets = endpoint.batch_for(dataset, indices, 1234 + 100000 + 1, {})  # 固定真实第1步裁剪种子。
    if current_assets != batches[0]['file_sha256']:  # 不能只看索引相同便假定底层特征相同。
        raise RuntimeError('Replayed first-batch assets differ from the saved ledger')
    teacher, student, _ = endpoint.load_teacher_and_student(endpoint.TEACHER, config, device)  # 教师保持冻结，学生只作临时测量。
    diffusion = endpoint.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)  # 同一扩散系数表。
    names = {name for name, _ in student.named_parameters()}  # 包括weight_norm的g/v，按真实Adam参数坐标插值。
    for name, value in student.state_dict().items():  # 证实归档初始学生确实等于教师初始化。
        if not torch.equal(value.detach().cpu(), initial[name]):
            raise RuntimeError('Archived initial student is not the fixed teacher initialization: ' + name)
    endpoint.training.seed_everything(1234 + 200000 + 1)  # 与原train step1第一次CUDA随机抽样完全同种子。
    fixed_train = {name: batch[name].to(device) for name in ('mel', 'bnf', 'spk_emb')}  # 原训练重建条件。
    noise = torch.randn_like(fixed_train['mel'])  # 只抽原来的源噪声，不增加其他隐藏随机变量。
    times = torch.full((32,), 950, dtype=torch.long, device=device)  # q与教师学生统一使用t950。
    fixed_train['x_start'] = diffusion.q_sample(fixed_train['mel'], times, noise)  # 两模型将共用同一实际状态。
    fixed_train['teacher_target'] = endpoint.ddim_teacher.sample_ddim_teacher(teacher=teacher, diffusion=diffusion,
        x_start=fixed_train['x_start'], speaker_embedding=fixed_train['spk_emb'], bnf=fixed_train['bnf'])  # 只生成一次固定训练答案。
    fixed_dev_cpu = load_torch_checkpoint(RUN / 'fixed_development_targets.pt')  # 开发目标直接读取原缓存，不重新生成。
    for name, value in fixed_dev_cpu.items():  # 每个缓存张量还要与原run_contract比对实际字节。
        if endpoint.tensor_sha(value) != contract['fixed_tensor_sha256'][name]:
            raise RuntimeError('Saved fixed development tensor changed: ' + name)
    fixed_dev = {name: value.to(device) for name, value in fixed_dev_cpu.items()}  # 同一32句逐元素复用。
    del fixed_dev_cpu, batch  # 释放不再使用的CPU批次副本。
    started = time.monotonic()  # 记录下面六点及方向梯度检查用时。
    initial_train, _ = endpoint.evaluate_fixed(student, diffusion, fixed_train)  # 先检查最重要的原训练loss重放。
    replay = {'train_endpoint_initial': require_close('First training endpoint loss', initial_train['endpoint_l1'], losses[0]['student_total'])}
    # 此后只求一次初始训练梯度，不调用任何优化器或改变参数。
    student.train()  # 与真实第一步训练前向模式一致；原模型没有BN/dropout状态变化。
    loss, prediction = endpoint.student_endpoint_loss(student=student, diffusion=diffusion,
        x_start=fixed_train['x_start'], speaker_embedding=fixed_train['spk_emb'], bnf=fixed_train['bnf'],
        teacher_target=fixed_train['teacher_target'])  # 相同输入和同一个固定教师端点。
    named = list(student.named_parameters())  # 保持名字与实际参数对象配对，不能靠偶然顺序猜测。
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)  # 不向.grad累积，也不更新。
    dot, gradient_square, delta_square, unused = 0.0, 0.0, 0.0, []  # CPU高精度累计三个向量统计。
    for (name, _), gradient in zip(named, gradients):  # 将每层真实差值与对应梯度逐项相乘。
        delta = updated[name].double() - initial[name].double()  # 在float64中表示两个归档float32值的真实差。
        delta_square += float(delta.square().sum())  # 计算完整实际更新的参数范数。
        if gradient is None:  # 未参与初始损失图的参数，理论上本次Adam不应更新。
            unused.append(name)  # 保留名字，避免把缺失梯度默默丢弃。
            if bool((delta != 0).any()):
                raise RuntimeError('An unused parameter changed in the archived update: ' + name)
            continue
        cpu_gradient = gradient.detach().cpu().double()  # 不保留任何GPU训练图，仅做向量数学。
        dot += float((cpu_gradient * delta).sum())  # dL/ds在s=0处的方向导数，负数表示局部下降方向。
        gradient_square += float(cpu_gradient.square().sum())  # 同一坐标系下梯度范数。
    grad_norm, delta_norm = math.sqrt(gradient_square), math.sqrt(delta_square)  # 两者都针对全部实际学生参数。
    direction = {'initial_training_gradient_dot_actual_delta': dot, 'gradient_norm': grad_norm,
        'actual_delta_norm': delta_norm, 'gradient_delta_cosine': dot / (grad_norm * delta_norm) if grad_norm * delta_norm else None,
        'unused_parameters': unused, 'negative_dot_means': 'infinitesimal descent of this fixed training objective, not guaranteed descent at s=1'}
    del gradients, prediction, loss, named, delta, cpu_gradient  # 释放梯度及反向图后再做六次只读前向。
    rows = []  # 预先固定六个比例，不依据开发结果自适应加点。
    for scale in SCALES:
        state = interpolate_state(initial, updated, scale, names)  # 原归档只读，插值结果属于独立CPU副本。
        student.load_state_dict(state, strict=True)  # 只改临时测量模型，不保存任何插值权重。
        del state  # 加载后释放本次完整CPU状态副本。
        train_scores, _ = endpoint.evaluate_fixed(student, diffusion, fixed_train)  # 所有比例面对同一个训练目标。
        dev_scores, _ = endpoint.evaluate_fixed(student, diffusion, fixed_dev)  # 原固定开发集仅用于观察独立后果。
        row = {'scale': scale, 'first_step_equivalent_lr': 0.0002 * scale, 'train': train_scores, 'dev': dev_scores}
        rows.append(row)  # 保存包括恶化比例在内的全部结果，不只保留最好点。
        print(json.dumps(row, ensure_ascii=False), flush=True)  # 根代理可及时查看每个固定比例的结果。
        if scale in (0, 1):  # 两端必须复现真实归档的开发读数。
            archived = dev0 if scale == 0 else dev1
            for key in ('endpoint_l1', 'mel_l1', 'prediction_rms'):
                replay[f'dev_{int(scale)}_{key}'] = require_close(f'Development s={scale} {key}', dev_scores[key], archived[key])
    hashes.update(current_assets)  # 原训练批次文件也绑定并在诊断结束再检查。
    endpoint.verify_hashes(hashes)  # 证实没有改写归档、教师、模块或特征来源。
    result = {'kind': 'read_only_archived_step_direction_probe', 'source_run': str(RUN), 'scales': list(SCALES),
        'file_sha256': hashes, 'first_train_indices': indices, 'first_train_crop_seed': 101235, 'first_train_noise_seed': 201235,
        'first_train_noise_sha256': endpoint.tensor_sha(noise),
        'fixed_training_tensor_sha256': {name: endpoint.tensor_sha(value) for name, value in fixed_train.items()},
        'replay': replay, 'direction': direction, 'rows': rows,
        'training_supported_tested_scales': training_supported_scales(rows),
        'interpretation_limits': ['Fixed fractions of one archived update, not new training or a learning-rate search on development scores',
            'Only sampled scales with lower first-training-batch objective support a smaller step; unsampled intervals are not established',
            'The first-step equivalent LR uses the same initial Adam moments; this does not predict a multi-step smaller-LR trajectory',
            'Development reconstruction errors do not establish CER or unseen conversion quality; no formal test was read'],
        'elapsed_probe_seconds': time.monotonic() - started}
    output.mkdir()  # 完成全部检查后只创建新的结果目录，不写任何模型权重。
    endpoint.training.atomic_write_json(output / 'results.json', result)  # 仅保存可复查的数值和身份记录。
    print('READ_ONLY_STEP_PROBE_COMPLETE', output / 'results.json', flush=True)  # 提示根代理结果位置。


if __name__ == '__main__':  # CPU测试导入时不会抢锁、加载模型或启GPU。
    main()  # 仅在主线程授权的明确只读命令下执行。
