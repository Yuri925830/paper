# 独立研究对照：学生直接拟合冻结教师的多步结果，不改正式 ACDD 或教师采样器。
"""Matpool-only, bounded clean-mel endpoint distillation; not original ACDD.

The teacher and student receive exactly the same noisy source z950, source
BNF and original speaker condition. The frozen teacher generates a DDIM30
eta=0 endpoint; the student predicts clean mel in one step. The sole training
loss is mean(abs(student_clean - stop_gradient(teacher_endpoint))), weight 1.
No discriminator, vocoder, score-noise re-corruption or formal-test selection
is used. This is reconstruction training, not cross-speaker training.

Version 3 changes only the endpoint Adam learning-rate policy. The base rate
must be declared as 5e-5 or 1e-4; it decreases linearly over a fixed 100000-update
horizon. A 600-update diagnostic does not compress that schedule. This policy
is informed by Progressive Distillation Appendices E/I, not an ACDD setting,
not a warm-up claim, and not the complete progressive distillation algorithm.
"""

import argparse  # 显式指定开发划分、独立输出和有限运行预算。
import hashlib  # 绑定实际文件与固定张量的字节身份。
import json  # 保存契约、每步记录、开发检查和完成状态。
import math  # 判断汇总梯度范数是否为有限数。
import platform  # 记录并核对实际 Python 版本。
import random  # 复用开发基线的打乱与裁剪种子规则。
import signal  # 在完整学生更新边界响应用户的中止请求。
import sys  # 在导入计算库前拒绝本地 Windows 执行。
import time  # 给在线教师目标生成和整个短训设置明确时限。
from pathlib import Path  # 只在租用服务器的实验目录中读写。

ROOT = Path('/root/FsatVoiceGrad')  # 唯一允许执行项目计算的租用服务器目录。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 导入也不能绕开运行地点约束。
    raise RuntimeError('Run this experiment only on the registered Matpool rental.')  # 本地只编辑上传。
sys.path.insert(0, str(ROOT))  # 模型代码必须来自服务器当前已审计的提交。

import torch  # 真实训练只由主线程审核后在租用 GPU 启动。
from torch.utils.data import default_collate  # 保留原 Dataset 批次的全部字段和维度。
import fastvoicegrad.train as training  # 只复用安全锁、随机状态和原子保存等基础能力。
from fastvoicegrad.components import load_teacher_and_student, sha256_file  # 不导入或构建声码器实例。
from fastvoicegrad.config import PAPER_STUDENT_CONFIG  # 保留既有教师、学生架构及 BNF 设置。
from fastvoicegrad.objectives import reverse_mean_to_zero  # 沿用学生的 epsilon→clean 参数化。
from voicegrad.dataset import VoiceGradDataset  # 保留教师训练时的数据读取和归一化接口。
from voicegrad.diffusion import VoiceGradDiffusion  # 沿用原 1000 步余弦日程。
import ddim_teacher  # 独立的 eta=0 目标生成器，不替换正式教师采样函数。

EXPECTED_HEAD = '0a2c646922704e746642f5f33c203869fe696b3f'  # 已审核的主代码版本。
CONTRACT_SHA = '08737551d34e00ba0b80a5e34adbf73fcfc6c70ece6bc8dc4a6c466e9b87afb1'  # 固定 52 句划分。
DDIM_SHA = '0e3100d14e421a6f7eb1a06b1dc79d195091c2f20a35f4c3e4266e77dbd3c6cc'  # 通过数学测试的采样模块。
TEACHER = ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt'  # 教师始终只读。
OBJECTIVE_ID = 'clean_endpoint_l1_ddim30_reconstruction_v1'  # 与 ACDD 损失及正式实验区分。
OPTIMIZER_POLICY_ID = 'endpoint_adam_linear_100000_v3'  # 新策略只改变端点Adam的学习率，不改变目标ID。
LR_DECAY_STEPS = 100000  # 预先固定长期日程；600步仅是本次短程观察上限。
LR_FORMULA = 'base_lr * max(0, (N-step)/(N-1)); N=100000; one-based update step'  # 第1步为base，第N步恰为0。


def learning_rate_for_step(base_lr, step, decay_steps=LR_DECAY_STEPS):
    # 两个候选量级来自相关论文的设置范围，具体选择由独立训练批次探针后显式传入。
    if base_lr not in (0.00005, 0.0001):
        raise ValueError('Endpoint v3 base learning rate must be 0.00005 or 0.0001')
    # step采用从1开始的实际更新编号，不能把初始化step0错算成第一次更新。
    if type(step) is not int or step < 1:
        raise ValueError('Learning-rate update step must be a positive integer')
    # 不允许为了600步短试验把整条退火时间轴也改为600步。
    if type(decay_steps) is not int or decay_steps != LR_DECAY_STEPS:
        raise ValueError('Endpoint v3 fixes the annealing horizon at 100000 updates')
    # 分母使用N−1，严格保证第1步等于基准值、第100000步等于0，之后不为负。
    return float(base_lr) * max(0.0, (decay_steps - step) / float(decay_steps - 1))


def set_optimizer_learning_rate(optimizer, base_lr, step, decay_steps=LR_DECAY_STEPS):
    # 只把本次更新应使用的标量写入Adam参数组，不触碰动量、参数、梯度或其他选项。
    value = learning_rate_for_step(base_lr, step, decay_steps)
    if not optimizer.param_groups:
        raise ValueError('Endpoint Adam must have at least one parameter group')
    for group in optimizer.param_groups:
        group['lr'] = value
    # 返回真实写入优化器的学习率，逐步日志不能继续记模板中的0.0002。
    return float(optimizer.param_groups[0]['lr'])


def optimizer_learning_rate_record(optimizer, base_lr, decay_steps, completed_steps):
    # 保存时同时记录最后一次更新和下一次计划使用值，明确初始化状态还未做更新。
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError('Completed optimizer updates must be a nonnegative integer')
    actual_lrs = [float(group['lr']) for group in optimizer.param_groups]
    expected = learning_rate_for_step(base_lr, max(1, completed_steps), decay_steps)
    # 断点必须对应真实优化器值，出现不一致时停止保存而不生成错误学习率证据。
    if not actual_lrs or any(value != expected for value in actual_lrs):
        raise ValueError('Actual optimizer learning rate differs from the declared update schedule')
    return {
        'policy_id': OPTIMIZER_POLICY_ID, 'base_learning_rate': float(base_lr),
        'decay_steps': decay_steps, 'formula': LR_FORMULA, 'completed_updates': completed_steps,
        'actual_param_group_learning_rates': actual_lrs,
        'last_update_learning_rate': actual_lrs[0] if completed_steps else None,
        'next_update_learning_rate': learning_rate_for_step(base_lr, completed_steps + 1, decay_steps),
        'gradient_clipping': None, 'ema_used': False,
    }


def tensor_sha(value):  # 对实际输入值而非种子名称建立可比较的身份。
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()  # 不修改张量。


def append_json(path, value):  # 每条完整记录占一行，保存批次与模型状态的对应关系。
    with path.open('a', encoding='utf-8') as handle:  # 仅追加到本次新建实验目录。
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')  # 拒绝无效数字。


def verify_hashes(expected):  # 读取字节重新核验，不对任何来源文件进行改写。
    for name, wanted in expected.items():  # 每个源代码、权重或数据文件有独立摘要。
        if sha256_file(Path(name)) != wanted:  # 同名文件被替换也必须失败。
            raise RuntimeError('Read-only dependency changed: ' + name)  # 不静默接受变化。


def validate_development(contract_path):  # 核对已经生成的固定开发集，不重新挑选语句。
    if sha256_file(contract_path) != CONTRACT_SHA:  # 拒绝其他划分或被修改的契约。
        raise RuntimeError('Development contract is not the reviewed fixed split')  # 需主线程重新审核。
    development = json.loads(contract_path.read_text(encoding='utf-8'))  # 只读元数据，不读测试指标。
    receipt_path = contract_path.parent / 'development_manifest.json'  # 完成收据必须与划分同目录。
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))  # 读取发布成功时保存的摘要。
    names = {'development_contract.json', 'pairs.csv', 'self_reconstruction.jsonl',
             'student_train_manifest.csv', 'dev_excluded_manifest.csv'}  # 五份文件共同定义开发协议。
    if receipt.get('status') != 'complete' or set(receipt.get('artifacts', {})) != names:  # 半成品不能启动。
        raise RuntimeError('Development receipt must contain all five completed artifacts')  # 不补造文件。
    expected = {str(contract_path.parent / name): digest for name, digest in receipt['artifacts'].items()}  # 单层文件名已锁定。
    expected[str(receipt_path)] = sha256_file(receipt_path)  # 收据本身也纳入本次依赖身份。
    expected.update(development['source_sha256'])  # 包含原名单、冻结统计量及 52 句预处理资产。
    verify_hashes(expected)  # 所有固定输入必须与发布开发集时保持一致。
    if receipt['development_contract_file_sha256'] != CONTRACT_SHA:  # 文件摘要不能与内容摘要混用。
        raise RuntimeError('Development file digest differs from its receipt')  # 契约身份不一致即停止。
    if training.canonical_json_sha256(development) != receipt['development_contract_sha256']:  # 再核对内容摘要。
        raise RuntimeError('Development canonical digest differs from its receipt')  # 防止错误标注。
    reconstruction = list(development['reconstruction_dev_indices'])  # 保留旧 32 句原始批内顺序。
    cross = set(development['cross_speaker_dev_indices'])  # 另有 10 句源语音和 10 句目标参考。
    excluded = set(development['excluded_train_indices'])  # 排除的是整句，不只是一个训练片段。
    if len(reconstruction) != 32 or len(set(reconstruction)) != 32 or len(cross) != 20:  # 核对两组实际规模。
        raise RuntimeError('Expected 32 reconstruction and 20 cross-speaker source/reference utterances')
    if set(reconstruction) & cross or excluded != set(reconstruction) | cross or len(excluded) != 52:  # 全部必须排除。
        raise RuntimeError('All 52 development utterances must be excluded from every student update')
    if any(type(index) is not int or not 0 <= index < 39387 for index in excluded):  # 原 CSV 的零起始索引。
        raise RuntimeError('Development index is outside the original training manifest')
    retained = [index for index in range(39387) if index not in excluded]  # 原顺序中的精确补集。
    if retained != development['retained_train_indices']:  # 不能临时增加或遗漏任何训练语句。
        raise RuntimeError('Retained training list differs from the exact development complement')
    protocol = development['reconstruction_protocol']  # 旧固定检查批次的输入约定。
    if (protocol['crop_seed'], protocol['source_noise_seed'], protocol['source_noise_device'],
        protocol['source_noise_dtype'], protocol['batch_size'], protocol['segment_frames']) != (1234, 55001, 'cuda', 'float32', 32, 128):
        raise RuntimeError('The archived 32-example reconstruction protocol changed')  # 不以改种子改善读数。
    return development, retained, reconstruction, expected  # 同时返回静态依赖，供后续保存和复查。


def batch_for(dataset, indices, crop_seed, asset_hashes):  # 每次取数都绑定实际读到的三类特征文件。
    current = {}  # 记录这一批使用过的源 mel、BNF 和说话人嵌入字节。
    for index in indices:  # 索引来自已经排除开发语句的名单，或固定只读开发批次。
        for key in ('mel_path', 'bnf_path', 'spk_emb_path'):  # 不加载或重新提取任何特征。
            path = Path(dataset.items[index][key]).resolve(strict=True)  # 解析原 Dataset 使用的真实路径。
            if ROOT / 'data' not in path.parents:  # 防止同名路径指向其他项目数据。
                raise RuntimeError('Training feature is outside the authoritative data directory')
            digest = sha256_file(path)  # 首次见到的训练文件记录现有字节身份。
            if str(path) in asset_hashes and asset_hashes[str(path)] != digest:  # 重复使用必须保持同一文件。
                raise RuntimeError('A previously read training feature changed: ' + str(path))
            current[str(path)] = digest  # 当前批次读数前的文件摘要。
    saved = random.getstate()  # 开发检查不能改变训练用的全局裁剪随机流。
    try:
        random.seed(crop_seed)  # 与 ACDD 开发基线采用同一逐批裁剪规则。
        batch = default_collate([dataset[index] for index in indices])  # 原 Dataset 完成归一化与 BNF 对齐。
    finally:
        random.setstate(saved)  # 即使数据读取报错，也恢复调用者的随机状态。
    verify_hashes(current)  # 读数前后再比对，拒绝读取期间被替换的特征。
    asset_hashes.update(current)  # 已读文件的摘要在后续检查点中永久绑定。
    return batch, current  # 返回批次和本次资产记录，不写回原始数据。


def student_endpoint_loss(*, student, diffusion, x_start, speaker_embedding, bnf, teacher_target, start_step=950):
    """Keep epsilon parameterization; compare clean outputs with a stopped endpoint."""
    if teacher_target.shape != x_start.shape or teacher_target.dtype != x_start.dtype or teacher_target.device != x_start.device:
        raise ValueError('Endpoint target must exactly match the shared input shape, dtype and device')  # 禁止广播错配。
    timesteps = torch.full((x_start.shape[0],), start_step, dtype=torch.long, device=x_start.device)  # 每例都从同一噪声级开始。
    prediction = reverse_mean_to_zero(student, diffusion, x_start, timesteps, speaker_embedding, bnf)  # 学生只前向一次。
    loss = (prediction - teacher_target.detach()).abs().mean()  # clean 空间逐元素平均 L1，系数为 1。
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(prediction).all()):  # 无效值不能进入 Adam。
        raise FloatingPointError('Nonfinite endpoint prediction or L1 loss')  # 清楚区分数值失败。
    return loss, prediction  # 目标已切断梯度，只有学生预测保留训练图。


@torch.no_grad()  # 固定开发集只作测量，不为模型提供参数更新。
def evaluate_fixed(student, diffusion, fixed):
    previous = student.training  # 评估后恢复原模式，不改变下一训练批次行为。
    student.eval()  # 禁用任何未来可能加入的随机训练层。
    try:
        loss, prediction = student_endpoint_loss(student=student, diffusion=diffusion, x_start=fixed['x_start'],
            speaker_embedding=fixed['spk_emb'], bnf=fixed['bnf'], teacher_target=fixed['teacher_target'])  # 固定同一输入和答案。
        gt_error = prediction - fixed['mel']  # 真实录音 mel 仅作为独立的重建检查参照。
        target_error = prediction - fixed['teacher_target']  # 同一固定教师目标，不随学生更新漂移。
        scores = {  # 同时报告拟合程度与真实 mel 差距，避免只看训练目标。
            'endpoint_l1': float(loss), 'endpoint_mse': float(target_error.square().mean()),  # 新目标误差。
            'mel_l1': float(gt_error.abs().mean()), 'mel_mse': float(gt_error.square().mean()),  # 与 ACDD 开发检查可比。
            'per_utterance_mel_l1': gt_error.abs().flatten(1).mean(1).cpu().tolist(),  # 保留 32 句独立结果。
            'per_utterance_endpoint_l1': target_error.abs().flatten(1).mean(1).cpu().tolist(),  # 防止平均值掩盖个别失败。
            'prediction_rms': float(prediction.square().mean().sqrt()),  # 学生频谱整体幅度。
            'reference_rms': float(fixed['mel'].square().mean().sqrt()),  # 真实 mel 的固定幅度。
            'teacher_target_rms': float(fixed['teacher_target'].square().mean().sqrt()),  # 多步教师答案的固定幅度。
            'teacher_target_gt_l1': float((fixed['teacher_target'] - fixed['mel']).abs().mean()),  # 教师端点自身误差。
            'prediction_sha256': tensor_sha(prediction),  # 与另存的预测张量绑定。
        }
        return scores, prediction.cpu()  # 只把这一小批结果搬到 CPU 保存。
    finally:
        student.train(previous)  # 保留学生后续的原训练模式。


def main():  # 明确启动命令后才会构建真实教师/学生，不提供自动续训入口。
    parser = argparse.ArgumentParser(description=__doc__)  # 帮助信息说明这是新研究对照。
    parser.add_argument('--development-contract', type=Path, required=True)  # 只能接受锁定的划分文件。
    parser.add_argument('--output-dir', type=Path, required=True)  # 每次试验必须使用新目录。
    parser.add_argument('--max-steps', type=int, default=600)  # 在线教师目标会增加每步时间，先限制 600 步。
    parser.add_argument('--max-seconds', type=int, default=3600)  # 时限在完整学生更新边界生效。
    parser.add_argument('--check-every', type=int, default=100)  # 固定开发量尺每 100 次更新测量一次。
    # 没有默认基准学习率，主线程必须依据训练批次探针明确选择，不能读取开发分数自动挑值。
    parser.add_argument('--learning-rate', type=float, choices=(0.00005, 0.0001), required=True)
    # 明确公开固定长期退火长度，choices阻止把短期预算当作退火终点。
    parser.add_argument('--lr-decay-steps', type=int, choices=(100000,), default=LR_DECAY_STEPS)
    parser.add_argument('--ddim-baseline-accepted', action='store_true')  # 仅主线程看过候选客观质量后显式放行。
    args = parser.parse_args()  # 不从旧训练目录猜测或恢复任何状态。
    if not args.ddim_baseline_accepted:  # 本文件写好不等于 DDIM30 的质量已经通过验收。
        raise RuntimeError('Review DDIM30 objective development quality before authorizing this run')
    if not 1 <= args.max_steps <= 600 or not 1 <= args.max_seconds <= 3600 or args.check_every != 100:
        raise ValueError('This first endpoint experiment is limited to 600 steps, 3600 seconds and checks every 100 steps')
    training.assert_rented_server(ROOT)  # 禁止在本地或实验室 101–104 上执行。
    locks = training.acquire_runtime_locks(ROOT)  # 训练与其他学生、声码器流程不能同时占用 GPU。
    if training.git_head(ROOT) != EXPECTED_HEAD:  # 同时要求 Git 工作区没有未提交改动。
        raise RuntimeError('The audited project source version changed')  # 不绕过主代码版本门禁。
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Use the original myconda training runtime, not the evaluation environment')
    if not torch.cuda.is_available():  # 实际短训只使用已登记的租用 GPU。
        raise RuntimeError('Rental CUDA GPU unavailable')
    output = args.output_dir.resolve()  # 解析符号链接后再核对目标范围。
    if ROOT / 'exp/diagnostics' not in output.parents or output.exists():  # 不覆盖历史实验或正式权重。
        raise RuntimeError('Choose a new output directory under exp/diagnostics')
    module_path = Path(ddim_teacher.__file__).resolve()  # 防止 Python 意外导入同名其他模块。
    if module_path != Path(__file__).resolve().with_name('ddim_teacher.py') or sha256_file(module_path) != DDIM_SHA:
        raise RuntimeError('The isolated DDIM sampler differs from the reviewed module')
    contract_path = args.development_contract.resolve()  # 固定开发集来自原训练数据，而非正式测试集。
    development, train_indices, dev_indices, dependencies = validate_development(contract_path)  # 52 句完整排除。
    config = PAPER_STUDENT_CONFIG  # 不修改被冻结的数据类或主代码默认值。
    actual = (config.batch_size, config.train_frames, config.student_start_step, config.learning_rate,
              config.adam_beta1, config.adam_beta2, config.bnf_alignment, config.bnf_downsampling)  # 关键接口联合验真。
    if actual != (32, 128, 950, 0.0002, 0.5, 0.9, 'legacy_length_linear', 'legacy_decimate'):
        raise RuntimeError('Source configuration template or original data interface changed')  # 模板lr仍为0.0002但在v3中不用于Adam。
    torch.set_num_threads(2)  # 与 ACDD 开发基线一致，真实网络更新仍在 GPU。
    training.seed_everything(1234)  # 学生从固定教师初始化，保留同一初始化调用顺序的前半段。
    device = torch.device('cuda')  # 使用该租用任务当前可见的 GPU。
    dataset = VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100,
                               bnf_alignment=config.bnf_alignment)  # 仅构造原训练 Dataset，不动任何 CSV。
    if len(dataset) != 39387:  # 子集索引必须对应已经锁定的原始名单。
        raise RuntimeError('Original training dataset length changed')
    teacher, student, _ = load_teacher_and_student(TEACHER, config, device)  # 严格 SHA 加载教师500并独立初始化学生。
    diffusion = VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)  # 同一累计 alpha 日程。
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate, betas=(0.5, 0.9), eps=1e-8,
                                 weight_decay=0, amsgrad=False)  # 只有学生拥有优化器，不创建 D/G。
    asset_hashes = {}  # 逐批绑定实际读取的训练特征，不声称已校验未读取的全库文件。
    dev_batch, _ = batch_for(dataset, dev_indices, 1234, asset_hashes)  # 与旧32句相同的裁剪与条件。
    dev_noise = torch.randn(dev_batch['mel'].shape, device=device,
        generator=torch.Generator(device=device).manual_seed(55001))  # 一次生成整批 CUDA 噪声，不能逐句重设种子。
    archived = development['reconstruction_protocol']['archived_input_identity']  # 比较实际张量而非只比较种子。
    for value, name in ((dev_batch['mel'], 'mel_sha256'), (dev_batch['bnf'], 'bnf_sha256'), (dev_noise, 'source_noise_sha256')):
        if tensor_sha(value) != archived[name]:  # 任一输入不匹配就不能复用旧开发量尺。
            raise RuntimeError('Archived reconstruction replay mismatch: ' + name)
    fixed = {name: dev_batch[name].to(device) for name in ('mel', 'bnf', 'spk_emb')}  # 开发数据不参加梯度更新。
    fixed['x_start'] = diffusion.q_sample(fixed['mel'], torch.full((32,), 950, device=device, dtype=torch.long), dev_noise)
    fixed['teacher_target'] = ddim_teacher.sample_ddim_teacher(teacher=teacher, diffusion=diffusion,
        x_start=fixed['x_start'], speaker_embedding=fixed['spk_emb'], bnf=fixed['bnf'])  # 固定开发答案只生成一次。
    dependencies[str(TEACHER)] = sha256_file(TEACHER)  # 再把教师权重文件绑定到当前实验契约。
    dependencies[str(module_path)] = DDIM_SHA  # 单独采样模块在本轮期间也不得变更。
    dependencies[str(Path(__file__).resolve())] = sha256_file(Path(__file__))  # 驱动脚本自身的实际字节身份。
    dependencies.update({str(ROOT / name): value for name, value in training.collect_source_fingerprints(ROOT).items()})
    teacher_initial = {name: tensor_sha(value) for name, value in teacher.state_dict().items()}  # 证明内存教师全程未变。
    record = {  # 明确记录活跃目标，不把模板中的 ACDD 字段当作本实验损失。
        'kind': 'endpoint_distillation_development_only', 'objective_id': OBJECTIVE_ID,
        'objective': {'formula': 'mean(abs(y_student_clean - stop_gradient(teacher_DDIM30(z950))))',
                      'endpoint_weight': 1.0, 'gan_used': False, 'vocoder_used': False, 'random_score_t_used': False},
        'sampler_id': ddim_teacher.DDIM_SAMPLER_ID, 'teacher_timesteps_descending': list(reversed(diffusion.sampling_sequence(30, 950))),
        'coupling': 'one q(source_mel,950,noise) state shared exactly by teacher and student; no reverse noise',
        'conditioning': 'reconstruction only: original speaker embedding and source BNF',
        'student_parameterization': 'epsilon; y=(z950-sqrt(1-alpha_bar950)*epsilon_student)/sqrt(alpha_bar950)',
        # 模板只保留教师/学生结构与前端来源；其中0.0002不能被当成本轮真实优化器学习率。
        'source_configuration_template': config.as_dict(), 'inactive_template_fields': ['learning_rate', 'all GAN/discriminator/score/test-loss fields'],
        'source_template_learning_rate_status': 'inactive: source_configuration_template.learning_rate=0.0002 is not used by this endpoint optimizer',
        # 其余Adam选项与v2不变，只有显式base_lr和逐步线性退火生效。
        'optimizer': {'name': 'Adam', 'lr': args.learning_rate, 'betas': [0.5, 0.9], 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False,
            'learning_rate_policy': optimizer_learning_rate_record(optimizer, args.learning_rate, args.lr_decay_steps, 0),
            'gradient_clipping': None, 'ema_used': False},
        'learning_rate_selection': 'explicit command-line value chosen from training-batch step-scale diagnostics, not development/test score selection',
        'batch_size': 32, 'segment_frames': 128, 'num_workers': 0, 'seed': 1234,
        'per_step_crop_seed': '1234+100000+step', 'per_step_source_noise_seed': '1234+200000+step; CUDA randn_like first draw',
        'max_steps': args.max_steps, 'max_seconds': args.max_seconds, 'check_every': args.check_every,
        'time_budget_scope': 'student training loop; fixed development target preparation is separately timed by launcher',
        'development_contract_file_sha256': CONTRACT_SHA, 'development_contract_sha256': training.canonical_json_sha256(development),
        'training_indices': train_indices, 'excluded_train_indices': development['excluded_train_indices'],
        'reconstruction_dev_indices': dev_indices, 'formal_test_used_for_selection': False,
        'teacher_exposure': 'frozen teacher and normalization statistics include these original training-source development utterances',
        'fixed_tensor_sha256': {name: tensor_sha(value) for name, value in fixed.items()},
        'fixed_source_noise_sha256': tensor_sha(dev_noise), 'teacher_state_sha256': teacher_initial,
        'file_sha256': dependencies, 'git_head': EXPECTED_HEAD,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'cudnn': torch.backends.cudnn.version(), 'cpu_threads': torch.get_num_threads(),
                    'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32},
        'development_regression_guard': {'GT_L1_ratio_above_initial': 1.10, 'prediction_RMS_ratio_below_initial': 0.80},
        'research_deviations': ['direct 30-to-1 clean endpoint L1 instead of ACDD', 'no D/G/adversarial/FM loss',
            'deterministic DDIM target sampling; official VoiceGrad sampler unchanged', 'reconstruction only, not conversion training',
            'bounded development run with 52 whole utterances excluded; no formal unseen checkpoint selection',
            'endpoint Adam learning rate inspired by PD Appendices E/I; no gradient clipping, EMA or complete progressive-halving procedure'],
        'source_references': ['https://arxiv.org/html/2010.02502v3#S4.SS1', 'https://arxiv.org/html/2202.00512v2#S3',
            'https://arxiv.org/html/2202.00512v2#A5', 'https://arxiv.org/html/2202.00512v2#A9'],
        'learning_rate_source_scope': 'PD Appendix E uses 1e-4 linearly annealed to zero and 100k updates for 2/1 steps; Appendix I includes 5e-5 and 1e-4 candidates. The endpoint-L1 application and step-index formula are declared experimental choices, not a PD warm-up prescription.',
        'method_boundary': 'direct endpoint fitting is inspired by deterministic distillation, not the complete progressive-halving method',
    }
    verify_hashes(dependencies)  # 确认目标准备期间静态来源没有被替换。
    output.mkdir()  # 通过身份检查与固定目标前向后才发布新实验目录。
    training.atomic_write_json(output / 'run_contract.json', record)  # 保存可追踪的新目标身份。
    contract_sha = training.canonical_json_sha256(record)  # 所有检查点绑定同一套不可变实验设定。
    training.atomic_torch_save({name: value.cpu() for name, value in fixed.items()}, output / 'fixed_development_targets.pt')
    initial, initial_prediction = evaluate_fixed(student, diffusion, fixed)  # 先保留更新前的真实误差和目标误差。
    training.atomic_write_json(output / 'initial_development.json', initial)  # 不是虚构的零损失。
    training.atomic_torch_save({'prediction': initial_prediction, 'step': 0}, output / 'development_prediction_step_000000.pt')  # 初始预测可逐元素复核。

    def checkpoint(step, scores):  # 每次保存学生、唯一优化器、RNG 与已使用文件账本。
        training.atomic_write_json(output / 'read_asset_sha256.json', asset_hashes)  # 只覆盖本轮内部的逐渐增长账本。
        payload = {'kind': 'development_checkpoint', 'objective_id': OBJECTIVE_ID, 'student': student.state_dict(),
                   'student_optimizer': optimizer.state_dict(), 'global_step': step, 'epoch': 0,
                   'run_contract_sha256': contract_sha, 'configuration': config.as_dict(), 'development': scores,
                   # configuration保留原模板，真实Adam设置必须从state_dict和本字段读取。
                   'inactive_configuration_fields': ['learning_rate', 'all GAN/discriminator/score/test-loss fields'],
                   'actual_optimizer_learning_rate': optimizer_learning_rate_record(optimizer, args.learning_rate, args.lr_decay_steps, step),
                   'rng_state': training.capture_rng_state(), 'read_asset_sha256': dict(asset_hashes)}  # 没有教师/D/G优化器。
        destination = output / ('initial_full.pt' if step == 0 else f'student_step_{step:06d}.pt')  # 所有整步状态独立保留。
        training.atomic_torch_save(payload, destination)  # 原子发布，防止断电后把半份文件当作检查点。
        training.atomic_write_json(output / 'latest_checkpoint.json', {'step': step, 'path': str(destination),
            'sha256': sha256_file(destination), 'run_contract_sha256': contract_sha})  # 不复制大型文件也能定位最新完整状态。

    checkpoint(0, initial)  # 初始化状态独立保存，后续可用于严格的学生起点比较。
    stop = {'requested': False}  # 信号回调只改标志，不在半个前向里保存模型。
    def stop_at_boundary(signum, frame):
        stop['requested'] = True  # 完成当前更新后才退出。
    signal.signal(signal.SIGTERM, stop_at_boundary)  # 支持主线程或用户发出正常停止请求。
    signal.signal(signal.SIGINT, stop_at_boundary)  # 终端 Ctrl+C 同样走完整边界。
    monitor = training.LossMonitor(output, window_steps=50, resume_step=0)  # 用同一窗口上涨报警规则。
    student.train()  # 只有学生执行反向传播，教师始终 eval 且 requires_grad=False。
    best, last_scores, final_step, reason = initial['mel_l1'], initial, 0, 'step_budget_complete'  # 初始学生也参与开发选择。
    started = time.monotonic()  # 在线目标计算计入每步和整个训练时限。
    per_pass = len(train_indices) // 32  # 丢弃不足32条的尾批，与开发 ACDD 基线一致。
    try:
        for step in range(1, args.max_steps + 1):  # 不可能无上限自动转为100轮训练。
            if step == 1 or (step - 1) % per_pass == 0:  # 每轮重新建立相同的训练语句顺序。
                order = list(train_indices)  # 永远不加入52句开发语音。
                random.Random(1234 + (step - 1) // per_pass).shuffle(order)  # 与基线使用相同逐轮种子。
            offset = ((step - 1) % per_pass) * 32  # 每次恰好选出一个完整批次。
            indices = order[offset:offset + 32]  # 保留实际参与这一步更新的原始训练行号。
            batch, current_assets = batch_for(dataset, indices, 1234 + 100000 + step, asset_hashes)  # 原路径同步裁剪。
            append_json(output / 'batch_assets.jsonl', {'step': step, 'indices': indices, 'file_sha256': current_assets})
            training.seed_everything(1234 + 200000 + step)  # 与开发ACDD这一批的第一份源噪声一致。
            mel, speaker, bnf = (batch[name].to(device) for name in ('mel', 'spk_emb', 'bnf'))  # 不交换条件角色。
            if tuple(mel.shape) != (32, 80, 128) or tuple(bnf.shape) != (32, 144, 128) or tuple(speaker.shape) != (32, 256):
                raise RuntimeError('Endpoint training batch violates the original 32/128 feature contract')
            noise = torch.randn_like(mel)  # 一次生成批内共同起点噪声，不再抽随机score时刻或第二份噪声。
            times = torch.full((32,), 950, dtype=torch.long, device=device)  # 源加噪和学生时间标签严格一致。
            x_start = diffusion.q_sample(mel, times, noise).detach()  # 教师和学生实际共用这一份张量。
            target_started = time.monotonic()  # 单独记录在线教师30步带来的时间成本。
            target = ddim_teacher.sample_ddim_teacher(teacher=teacher, diffusion=diffusion, x_start=x_start,
                speaker_embedding=speaker, bnf=bnf)  # 整条教师轨迹在no_grad下完成，再建学生图。
            teacher_seconds = time.monotonic() - target_started  # 此时同步的有限性检查确保GPU前向完成。
            optimizer.zero_grad(set_to_none=True)  # 不累加上一步的学生梯度。
            loss, prediction = student_endpoint_loss(student=student, diffusion=diffusion, x_start=x_start,
                speaker_embedding=speaker, bnf=bnf, teacher_target=target)  # clean目标权重始终为1。
            loss.backward()  # 仅学生图参与反向传播，teacher_target已停止梯度。
            gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]  # 未参与图的参数保留None。
            if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):  # 拒绝空/非有限梯度。
                raise FloatingPointError('Endpoint student gradient is missing or nonfinite')
            grad_norm = float(torch.stack([gradient.detach().float().norm() for gradient in gradients]).norm())  # 只记录，不裁剪。
            if not math.isfinite(grad_norm):  # 范数溢出同样不允许更新。
                raise FloatingPointError('Endpoint student gradient norm is nonfinite')
            # 梯度与v2完全相同；仅在Adam实际更新前写入该步的退火学习率。
            actual_lr = set_optimizer_learning_rate(optimizer, args.learning_rate, step, args.lr_decay_steps)
            optimizer.step()  # 真实更新只发生在学生优化器中，一批恰好一次。
            final_step = step  # 仅完整优化器更新后才推进实验进度。
            losses = {'student_total': float(loss.detach()), 'dist_raw': float(loss.detach())}  # 两字段指向同一个纯端点L1。
            monitor.observe(step, 0, losses)  # 复用总损失的50步窗口和持续上涨报警。
            append_json(output / 'step_losses.jsonl', {'step': step, 'elapsed_seconds': time.monotonic() - started,
                'teacher_target_seconds': teacher_seconds, 'student_gradient_norm': grad_norm, **losses,
                # 记录本次真实Adam学习率及长期日程，600步不会把它降到0。
                'learning_rate': actual_lr, 'base_learning_rate': args.learning_rate, 'lr_decay_steps': args.lr_decay_steps,
                'lr_formula': LR_FORMULA, 'optimizer_policy_id': OPTIMIZER_POLICY_ID,
                'loss_phase': 'pre-update student prediction; fixed teacher target', 'objective_id': OBJECTIVE_ID})
            del gradients, loss, prediction, target, x_start, noise, times, mel, speaker, bnf, batch  # 及时释放上一批图与特征。
            reached_time = time.monotonic() - started >= args.max_seconds  # 不在半个更新中强制切断训练。
            if step % 100 == 0 or step == args.max_steps or reached_time or stop['requested']:  # 常规或退出边界测量。
                last_scores, prediction_cpu = evaluate_fixed(student, diffusion, fixed)  # 同一组32句与DDIM答案。
                item = {'step': step, 'elapsed_seconds': time.monotonic() - started, **last_scores}  # 使用更新后的学生。
                append_json(output / 'development.jsonl', item)  # 全部检查点读数保留，不只报告最好值。
                training.atomic_torch_save({'prediction': prediction_cpu, 'step': step}, output / f'development_prediction_step_{step:06d}.pt')
                checkpoint(step, last_scores)  # 同步保存该步学生和优化器，方便主线程检查。
                if last_scores['mel_l1'] < best:  # 仅用固定开发GT误差挑候选，不能看正式test决定。
                    best = last_scores['mel_l1']  # 端点L1下降不足以自动证明真实语音更好。
                    training.atomic_write_json(output / 'best_development.json', item)  # 记录哪个已保存检查点更好。
                training.atomic_write_json(output / 'status.json', {'status': 'running', **item})  # 主线程可读取简洁进度。
                print(json.dumps(item, ensure_ascii=False), flush=True)  # 每次检查及时输出状态。
                if last_scores['mel_l1'] > initial['mel_l1'] * 1.10 or last_scores['prediction_rms'] < initial['prediction_rms'] * 0.80:
                    reason = 'development_regression_guard'  # GT误差明显增大或频谱收缩就先停止此候选。
                    break  # 保留当前完整证据，不自动改变权重继续训练。
                if reached_time or stop['requested']:  # 达到预算或收到用户停止信号都结束本轮。
                    reason = 'requested_stop' if stop['requested'] else 'time_budget_complete'
                    break  # 后续是否继续由主线程结合开发客观评估决定。
        if any(parameter.requires_grad or parameter.grad is not None for parameter in teacher.parameters()):  # 教师冻结边界复核。
            raise RuntimeError('Frozen teacher acquired gradients during endpoint training')
        if teacher_initial != {name: tensor_sha(value) for name, value in teacher.state_dict().items()}:  # 检查内存权重及缓冲。
            raise RuntimeError('Frozen teacher state changed during endpoint training')
        verify_hashes(dependencies)  # 结束时复核全部静态源码、教师文件及固定数据来源。
        verify_hashes(asset_hashes)  # 实际用于训练的特征也必须保持原字节，不声称未读全库已验真。
    except Exception as error:  # 异常时保留最近完整检查点，不能把半步坏模型伪装成成功。
        monitor.finish('failed', final_step, 0)  # 清理运行状态，触发外层监控关注失败。
        training.atomic_write_json(output / 'status.json', {'status': 'failed', 'step': final_step,
            'error_type': type(error).__name__, 'error': str(error), 'latest_checkpoint': str(output / 'latest_checkpoint.json')})
        raise  # 让进程以失败退出码交给主线程，不吞掉错误继续训练。
    monitor.finish(reason, final_step, 0)  # 不留下已结束却仍显示running的监控记录。
    summary = {'status': reason, 'global_step': final_step, 'elapsed_seconds': time.monotonic() - started,
               'initial': initial, 'final': last_scores, 'best_development_mel_l1': best,
               'run_contract_sha256': contract_sha, 'objective_id': OBJECTIVE_ID,
               'actual_optimizer_learning_rate': optimizer_learning_rate_record(optimizer, args.learning_rate, args.lr_decay_steps, final_step),
               'read_asset_ledger_sha256': sha256_file(output / 'read_asset_sha256.json')}  # 绑定已读训练文件账本。
    training.atomic_write_json(output / 'completion.json', summary)  # 完成状态独立保存，不能当正式100epoch结果。
    training.atomic_write_json(output / 'status.json', summary)  # 监控只读这个最终摘要也能正确停止。
    print('ENDPOINT_DEVELOPMENT_FINISHED', reason, final_step, flush=True)  # 明确实验结束和实际步数。


if __name__ == '__main__':  # 导入CPU单元测试时绝不进入真实训练。
    main()  # 只有显式启动并通过DDIM质量放行门禁才会创建GPU学生实验。
