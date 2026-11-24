# 独立续训入口：保留已完成的ACDD600，不覆盖旧目录，也不把续训误写成从头训练。
"""Resume the audited ACDD development run at global step 601, explicitly.

This driver retains the original train_batch, teacher, frozen vocoder, BNF,
2/45 coefficients and constant Adam 2e-4. It restores both models, both Adam
states and all RNG states. The first resumed batch reconstructs the current
shuffle pass even when resuming inside it. No formal test selects checkpoints.
"""

import argparse  # 续训来源SHA和结束的全局步数必须在命令行中明确给出。
import copy  # 构造新契约时保留原配置和父记录，不能反写已读取的对象。
import json  # 保存父子关系、完整检查点身份与开发集结果。
import math  # 拒绝非有限状态，并执行原有开发保护阈值。
import platform  # 保证沿用真实训练解释器版本。
import random  # 根据全局步数重建原训练样本打乱顺序。
import signal  # 用户请求停止时在完整批次边界保存。
import sys  # 导入计算库前拒绝本地Windows执行。
import time  # 给本次续训分段设置独立安全时限。
from pathlib import Path  # 所有权重来源只读，输出必须使用新诊断目录。

ROOT = Path('/root/FsatVoiceGrad')  # 项目计算唯一允许的位置。
if not sys.platform.startswith('linux') or not ROOT.is_dir():  # 本地只可编辑或上传该文件。
    raise RuntimeError('Run only on the registered Matpool rental')
sys.path.insert(0, str(ROOT))  # 模型与训练公式固定读取服务器现场代码。

import torch  # 主入口经审核才使用GPU；单元测试只使用CPU假模型。
import run_acdd_development as baseline  # 原开发驱动冻结，只复用其数据及测量方法。
from fastvoicegrad.components import load_torch_checkpoint  # 读取已有完整归档，不猜测latest内容。

PARENT = ROOT / 'exp/diagnostics/student_development_20260906/acdd_clean_600_v1'  # 唯一允许的父实验。
PARENT_CONTRACT_SHA = 'f9472aea9332e8c261e4045ceac6630efe8f9ba012f135d5b4aba45216b32c13'  # 内容级身份。
PARENT_CONTRACT_FILE_SHA = 'fd55d3708f61c0d5d08189e57a904632605a2576b34262af7f4f9cee401512c3'  # 文件字节身份。
PARENT_FULL_SHA = '70dc5ed188e0eaeccc4c6b8fd80ed79595605b3af5569bafe87264645c54cf43'  # S/D/双Adam/RNG完整600步状态。
ANCHOR_SHA = 'ec504a627350c102cab1647befdc74234969b616238d2cd07d5350ef0c322162'  # 第0步完整状态，保护线从这里锚定。
BASELINE_DRIVER_SHA = '5cd72f6199e542ecb64a49ed9c70883a4f9b6ea2d39d0faf4fb2a652ba0e2395'  # 冻结的原开发脚本。
DEVELOPMENT_SHA = '08737551d34e00ba0b80a5e34adbf73fcfc6c70ece6bc8dc4a6c466e9b87afb1'  # 固定52句排除协议。


class GlobalBatchPlan:
    """Recover the current shuffle from the absolute step, including mid-pass resumes."""
    def __init__(self, indices, batch_size=32, seed=1234):
        self.indices = list(indices)  # 保存原39,335条保留索引的固定顺序。
        self.batch_size, self.seed = batch_size, seed  # 使用原batch32和seed1234。
        self.batches_per_pass = len(self.indices) // batch_size  # 实际每轮1229批，余下7句丢弃。
        if self.batches_per_pass < 1:  # 空训练计划无法定义合法全局步数。
            raise ValueError('Not enough examples for one full batch')
        self.pass_index, self.order = None, None  # 新进程没有旧order，首个请求必须重建。

    def for_step(self, global_step):
        if type(global_step) is not int or global_step < 1:  # 全局步数从1开始，不能从本地段落步数0开始。
            raise ValueError('global_step must be a positive integer')
        current_pass = (global_step - 1) // self.batches_per_pass  # 601仍属于第0轮，1230进入第1轮。
        if self.pass_index != current_pass:  # 包括首次从601进入时，不能等下一轮才创建order。
            self.order = list(self.indices)  # 每轮从原名单重建，不继续打乱上轮排列。
            random.Random(self.seed + current_pass).shuffle(self.order)  # 与原驱动逐轮seed完全一致。
            self.pass_index = current_pass  # 缓存当前轮，避免每一批重复打乱。
        offset = ((global_step - 1) % self.batches_per_pass) * self.batch_size  # 原轮次中的真实批次位置。
        return self.order[offset:offset + self.batch_size]  # 始终完整batch，不重播前600步。


def assert_guard(scores, anchor):  # 所有阶段沿用最初S0保护线，不以更差的父状态重新起算。
    for name in ('mel_l1', 'prediction_rms'):  # 两个保护量都必须是有效非负数。
        if not math.isfinite(float(scores[name])) or scores[name] < 0:
            raise RuntimeError('Invalid development protection metric: ' + name)
    if scores['mel_l1'] > anchor['mel_l1'] * 1.10 or scores['prediction_rms'] < anchor['prediction_rms'] * 0.80:
        raise RuntimeError('Original step-zero development regression guard exceeded')  # 不允许放宽阈值续训。


def validate_adam_state(saved, model, expected_step):  # 加载前按当前模型参数顺序核验历史Adam张量。
    groups = saved['param_groups']  # 原学生和D各只有一组固定学习率参数。
    parameters = list(model.parameters())  # 同一已锁定源码决定相同的参数登记顺序。
    if len(groups) != 1 or len(groups[0]['params']) != len(parameters):
        raise RuntimeError('Adam parameter group layout changed')
    group = groups[0]  # 两套优化器都保留论文的恒定2e-4与beta参数。
    expected = {'lr': 0.0002, 'betas': (0.5, 0.9), 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False}
    if any(group.get(key) != value for key, value in expected.items()):  # 不接受重设学习率或丢失动量配置。
        raise RuntimeError('Saved Adam hyperparameters differ from the original ACDD optimizer')
    if any(group.get(key, False) for key in ('maximize', 'capturable', 'differentiable', 'fused')):  # 保持原非融合Adam路径。
        raise RuntimeError('Saved Adam execution mode differs from the audited optimizer')
    identifiers = group['params']  # 这些编号是优化器归档的局部参数ID。
    if len(set(identifiers)) != len(identifiers) or set(saved['state']) != set(identifiers):  # 600步全部参数都有状态。
        raise RuntimeError('Adam state is incomplete or contains extra parameters')
    for identifier, parameter in zip(identifiers, parameters):  # 明确绑定每个moment与其真实参数形状。
        state = saved['state'][identifier]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != expected_step:
            raise RuntimeError('Adam moment fields or optimizer step counter are inconsistent')
        for key in ('exp_avg', 'exp_avg_sq'):  # 一阶与二阶动量都必须保存并完整恢复。
            value = state[key]
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all()):
                raise RuntimeError('Invalid Adam moment tensor: ' + key)
        if bool((state['exp_avg_sq'] < 0).any()):  # 平方梯度累计不可能为负。
            raise RuntimeError('Saved Adam second moment contains negative values')


def restore_training_state(checkpoint, student, discriminator, student_optimizer, discriminator_optimizer,
                           expected_step, expected_contract, device):
    """Restore S/D, both Adam states and RNG without applying any update."""
    required = {'student', 'discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state',
                'global_step', 'run_contract_sha256', 'configuration', 'development'}  # 精简学生权重不能续训。
    if not required <= set(checkpoint):
        raise RuntimeError('Parent checkpoint is not a full two-model/two-optimizer state')
    if checkpoint['global_step'] != expected_step or checkpoint['run_contract_sha256'] != expected_contract:
        raise RuntimeError('Parent global step or contract differs from the declared resume point')
    validate_adam_state(checkpoint['student_optimizer'], student, expected_step)  # 先验证，不边加载边容忍缺项。
    validate_adam_state(checkpoint['discriminator_optimizer'], discriminator, expected_step)  # D动量不能从头初始化。
    student.load_state_dict(checkpoint['student'], strict=True)  # 覆盖新建模型的随机值，恢复真实S600。
    discriminator.load_state_dict(checkpoint['discriminator'], strict=True)  # 完整恢复与S600共同训练的D600。
    student_optimizer.load_state_dict(checkpoint['student_optimizer'])  # 包括600步一阶/二阶动量与step。
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])  # D也必须从601接着更新。
    if {id(p) for group in student_optimizer.param_groups for p in group['params']} != {id(p) for p in student.parameters()}:
        raise RuntimeError('Student Adam does not exclusively own the student parameters')
    if {id(p) for group in discriminator_optimizer.param_groups for p in group['params']} != {id(p) for p in discriminator.parameters()}:
        raise RuntimeError('Discriminator Adam does not exclusively own D parameters')
    if {id(p) for p in student.parameters()} & {id(p) for p in discriminator.parameters()}:  # 两者不能共享待更新参数。
        raise RuntimeError('Student and discriminator share parameter objects')
    student.train()  # 模式不是state_dict的一部分，显式恢复原训练阶段。
    discriminator.train()  # D训练前向同样保持原模式。
    baseline.training.restore_rng_state(checkpoint['rng_state'], device)  # 所有构建/加载完成后恢复Python/NumPy/CPU/CUDA。


def verify_files(hashes):  # 对输入文件只做只读核验，不改写任何旧证据。
    for name, expected in hashes.items():
        if baseline.sha256_file(Path(name)) != expected:
            raise RuntimeError('Frozen dependency changed: ' + name)


def load_development(path):  # 复用固定划分，不重新选择有利于新结果的语句。
    if baseline.sha256_file(path) != DEVELOPMENT_SHA:
        raise RuntimeError('Development split differs from the reviewed 52-utterance exclusion')
    value = json.loads(path.read_text(encoding='utf-8'))  # 仅读取训练来源的开发定义。
    receipt_path = path.parent / 'development_manifest.json'  # 完整发布后才存在该收据。
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    expected_names = {'development_contract.json', 'pairs.csv', 'self_reconstruction.jsonl',
                      'student_train_manifest.csv', 'dev_excluded_manifest.csv'}  # 五份构成完整开发定义。
    if receipt.get('status') != 'complete' or set(receipt['artifacts']) != expected_names:
        raise RuntimeError('Development receipt is incomplete')
    hashes = {str(path.parent / name): digest for name, digest in receipt['artifacts'].items()}
    hashes[str(receipt_path)] = baseline.sha256_file(receipt_path)  # 记录实际收据字节。
    hashes.update(value['source_sha256'])  # 绑定原名单、冻结mean/std及52句素材。
    verify_files(hashes)  # 数据与划分发布时的实际文件必须一致。
    excluded = set(value['excluded_train_indices'])  # 所有源语音和目标参考都不能重新进入训练。
    if len(excluded) != 52 or excluded != set(value['reconstruction_dev_indices']) | set(value['cross_speaker_dev_indices']):
        raise RuntimeError('All 52 development utterances must remain excluded')
    retained = [index for index in range(39387) if index not in excluded]  # 与原CSV保持相同索引含义。
    if retained != value['retained_train_indices'] or len(retained) != 39335:
        raise RuntimeError('Retained training population changed')
    return value, hashes  # 后面还要与父run_contract逐项比对。


def main():  # 主线程决定继续后才调用，不会因脚本准备完成自动启动训练。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-checkpoint', required=True, type=Path)  # 必须显式指定完整latest_full。
    parser.add_argument('--parent-checkpoint-sha256', required=True)  # 明确承认此次续训的父文件内容。
    parser.add_argument('--parent-contract-sha256', required=True)  # 这里使用内容SHA，不是文件SHA。
    parser.add_argument('--stop-global-step', required=True, type=int, choices=(2000, 6000))  # 由root评估后选择分段终点。
    parser.add_argument('--max-seconds', required=True, type=int)  # 本段有独立时限，不把未完成当完成。
    parser.add_argument('--development-contract', required=True, type=Path)  # 使用同一开发集。
    parser.add_argument('--output-dir', required=True, type=Path)  # 新目录，不覆盖ACDD600。
    parser.add_argument('--resume-approved', action='store_true')  # 准备和CPU测试不等于已批准GPU续训。
    args = parser.parse_args()
    if not args.resume_approved or not 1 <= args.max_seconds <= 43200:  # 最多12小时安全分段，不能无期限运行。
        raise RuntimeError('Explicit resume approval and a bounded positive time budget are required')
    baseline.training.assert_rented_server(ROOT)  # 禁止本地及实验室101–104执行。
    locks = baseline.training.acquire_runtime_locks(ROOT)  # 全局互斥，其他GPU工作未结束时安全拒绝。
    if baseline.training.git_head(ROOT) != baseline.EXPECTED_HEAD:  # 主源码必须仍是同一干净提交。
        raise RuntimeError('The original ACDD implementation changed')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Resume requires the original myconda training runtime')
    if not torch.cuda.is_available():
        raise RuntimeError('Rental CUDA GPU unavailable')
    parent_path = args.parent_checkpoint.resolve()  # 不接受精简S600或另一个同名检查点。
    if parent_path != PARENT / 'latest_full.pt' or args.parent_checkpoint_sha256 != PARENT_FULL_SHA:
        raise RuntimeError('This driver only resumes the audited complete ACDD step-600 checkpoint')
    if args.parent_contract_sha256 != PARENT_CONTRACT_SHA:
        raise RuntimeError('Declared parent contract differs from the archived ACDD600 lineage')
    output = args.output_dir.resolve()  # 解析后才检查目录范围。
    if ROOT / 'exp/diagnostics' not in output.parents or output.exists() or PARENT in output.parents:
        raise RuntimeError('Use a new diagnostic directory outside the frozen parent run')
    baseline_path = Path(baseline.__file__).resolve()  # 冻结辅助函数也要核验实际导入的文件。
    dependencies = {str(parent_path): PARENT_FULL_SHA, str(PARENT / 'initial_full.pt'): ANCHOR_SHA,
                    str(PARENT / 'run_contract.json'): PARENT_CONTRACT_FILE_SHA, str(baseline_path): BASELINE_DRIVER_SHA}
    verify_files(dependencies)  # 先验字节，后读取权重。
    parent = json.loads((PARENT / 'run_contract.json').read_text(encoding='utf-8'))  # 原始训练配置与固定输入身份。
    if baseline.training.canonical_json_sha256(parent) != PARENT_CONTRACT_SHA:
        raise RuntimeError('Parent contract canonical hash mismatch')
    development, dev_hashes = load_development(args.development_contract.resolve())  # 保留全部52句排除和原素材。
    dependencies.update(dev_hashes)
    if parent['training_indices'] != development['retained_train_indices'] or parent['reconstruction_dev_indices'] != development['reconstruction_dev_indices']:
        raise RuntimeError('Resume training or development indices differ from the parent')
    config = baseline.replace(baseline.PAPER_STUDENT_CONFIG, num_workers=0)  # 与原开发基线完全相同配置。
    if config.as_dict() != parent['configuration'] or parent['seed'] != 1234:  # 不附带任何超参、BNF或loss改动。
        raise RuntimeError('ACDD configuration or seed differs from the archived parent')
    sources = baseline.training.collect_source_fingerprints(ROOT)  # 逐文件核对真实公式和数据接口。
    if sources != parent['source_sha256']:
        raise RuntimeError('Production source fingerprints no longer match the parent run')
    dependencies.update({str(ROOT / name): digest for name, digest in sources.items()})
    dependencies[str(Path(__file__).resolve())] = baseline.sha256_file(Path(__file__))  # 独立续训入口身份。
    payload = load_torch_checkpoint(parent_path)  # 含两模型、双Adam和600步RNG。
    anchor_payload = load_torch_checkpoint(PARENT / 'initial_full.pt')  # 只用原step0作保护锚，绝不覆盖父S600。
    if anchor_payload['global_step'] != 0 or anchor_payload['run_contract_sha256'] != PARENT_CONTRACT_SHA:
        raise RuntimeError('Original step-zero protection anchor is invalid')
    anchor = copy.deepcopy(anchor_payload['development'])  # 原始阈值的唯一参照。
    if anchor['mel_l1'] != 0.19941353797912598 or anchor['prediction_rms'] != 0.9994737505912781:
        raise RuntimeError('The original S0 development guard anchor changed')
    del anchor_payload  # 不需要把大体积初始D/S/空优化器继续留在内存中。
    assert_guard(payload['development'], anchor)  # 已经超阈值的父状态不能以新起点为由继续。
    if payload['configuration'] != config.as_dict():  # 完整检查点与父契约也要一致。
        raise RuntimeError('Parent checkpoint configuration mismatch')
    torch.set_num_threads(2)  # 沿用原开发驱动的CPU配置，不改TF32或cudnn选项。
    baseline.training.seed_everything(1234)  # 保留原模型构建顺序，之后仍会恢复600步RNG。
    device = torch.device('cuda')
    dataset = baseline.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100,
                                       bnf_alignment=config.bnf_alignment)  # 原归一化与BNF完全不变。
    if len(dataset) != 39387:
        raise RuntimeError('Original dataset length changed')
    teacher, student, _ = baseline.load_teacher_and_student(baseline.TEACHER, config, device)  # 教师仍固定500epoch。
    vocoder, vconfig, _ = baseline.load_frozen_vocoder(ROOT, baseline.VOCODER_DIR / 'generator_step_2500000.pt',
        baseline.VOCODER_DIR / 'config.json', baseline.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = baseline.build_paper_discriminator(ROOT, vconfig, config, device)  # 新结构马上由D600完整覆盖。
    diffusion = baseline.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = baseline.load_mel_statistics(ROOT / 'data', device)  # 不重估训练统计量。
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2))
    restore_training_state(payload, student, discriminator, optimizer, d_optimizer, 600, PARENT_CONTRACT_SHA, device)  # 真正从S/D600接续。
    parent_rng = payload['rng_state']  # 固定开发重放完成后再恢复一次，防止初始化诊断消耗随机流。
    parent_scores = copy.deepcopy(payload['development'])
    del payload  # 优化器已加载到参数设备，不再保留大体积CPU父副本。
    protocol = development['reconstruction_protocol']  # 与父run完全相同的32句/裁剪/噪声协议。
    dev_batch = baseline.batch_for(dataset, development['reconstruction_dev_indices'], protocol['crop_seed'])
    dev_noise = torch.randn(dev_batch['mel'].shape, device=device,
                           generator=torch.Generator(device=device).manual_seed(protocol['source_noise_seed']))  # 整批同一CUDA噪声。
    def tensor_sha(value):
        return baseline.hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()  # 比较实际张量身份。
    if any(tensor_sha(dev_batch[key]) != parent['fixed_development_inputs'][key] for key in ('mel', 'bnf', 'spk_emb')):
        raise RuntimeError('Fixed development mel/BNF/speaker inputs differ from the parent')
    if tensor_sha(dev_noise) != parent['fixed_development_source_noise_sha256']:
        raise RuntimeError('Fixed CUDA source noise differs from the parent')
    resumed_scores = baseline.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)  # 更新前核对S600重放。
    if any(abs(resumed_scores[key] - parent_scores[key]) > 1e-7 for key in ('mel_l1', 'prediction_rms', 'reference_rms', 'cosine_reference')):
        raise RuntimeError('Restored student does not replay the parent development state')
    assert_guard(resumed_scores, anchor)  # 用实际重放值再检查一次原S0保护线。
    plan = GlobalBatchPlan(parent['training_indices'])  # global601会立即建立pass0的正确order。
    if plan.batches_per_pass != 1229:
        raise RuntimeError('Resume requires the unchanged 39335-example, batch32 schedule')
    windows = PARENT / 'loss_windows.jsonl'  # 继承旧窗口，不能把续训开始后的趋势当作全新曲线。
    dependencies[str(windows)] = baseline.sha256_file(windows)  # 窗口历史也是固定父证据。
    dependencies[str(baseline.TEACHER)] = parent['teacher_sha256']
    dependencies[str(baseline.VOCODER_DIR / 'generator_step_2500000.pt')] = parent['vocoder_sha256']
    verify_files(dependencies)  # 模型加载与开发重放期间不能换文件。
    record = {'kind': 'acdd_development_continuation', 'lineage': {'parent_directory': str(PARENT),
        'parent_full_checkpoint': str(parent_path), 'parent_full_checkpoint_sha256': PARENT_FULL_SHA,
        'parent_contract_sha256': PARENT_CONTRACT_SHA, 'parent_contract_file_sha256': PARENT_CONTRACT_FILE_SHA,
        'anchor_step0_checkpoint_sha256': ANCHOR_SHA, 'start_after_global_step': 600},
        'configuration': config.as_dict(), 'training_indices': parent['training_indices'],
        'excluded_train_indices': parent['excluded_train_indices'], 'reconstruction_dev_indices': parent['reconstruction_dev_indices'],
        'development_contract_file_sha256': DEVELOPMENT_SHA, 'parent_resumed_development': resumed_scores,
        'original_step0_guard_anchor': anchor, 'guard_ratios': {'mel_l1_upper': 1.10, 'prediction_rms_lower': 0.80},
        'stop_global_step': args.stop_global_step, 'max_seconds': args.max_seconds, 'check_every': 100,
        'batch_plan': {'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((global_step-1)/1229)',
            'crop_seed': '1234+100000+global_step', 'noise_seed': '1234+200000+global_step', 'first_resumed_global_step': 601},
        'fixed_development_inputs': parent['fixed_development_inputs'],
        'fixed_development_source_noise_sha256': parent['fixed_development_source_noise_sha256'],
        'file_sha256': dependencies, 'formal_test_used_for_selection': False,
        'method': 'unchanged production train_batch; one D step followed by one student step; adv+2FM+45dist',
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32},
        'scope': 'Continuation of training-source development experiment; no claim that longer training already improves CER'}
    output.mkdir()  # 只有全部门禁与父重放通过才创建新续训目录。
    baseline.save_json(output / 'run_contract.json', record)
    contract_sha = baseline.training.canonical_json_sha256(record)  # 子检查点使用新契约，同时显式保存父关系。
    baseline.training.atomic_torch_save({'batch': dev_batch, 'source_noise': dev_noise.cpu()}, output / 'fixed_development_inputs.pt')
    (output / 'loss_windows.jsonl').write_bytes(windows.read_bytes())  # 只复制父窗口到新目录，原文件保持不变。
    monitor = baseline.training.LossMonitor(output, window_steps=50, resume_step=600)  # 从已有600步完整窗口继续报警。
    baseline.training.restore_rng_state(parent_rng, device)  # 在任何新训练批次之前精确恢复父随机状态。
    del parent_rng
    final_step, last_scores, reason = 600, resumed_scores, 'global_step_budget_complete'  # 全局游标绝不清零。
    best = min(anchor['mel_l1'], resumed_scores['mel_l1'])  # 只对开发结果记账，不读正式测试指标。
    stop = {'requested': False}
    def stop_at_boundary(signum, frame):
        stop['requested'] = True  # 信号处理只改标志，不在半步中保存模型。
    signal.signal(signal.SIGTERM, stop_at_boundary)
    signal.signal(signal.SIGINT, stop_at_boundary)

    def checkpoint(step, scores):  # 每100步保存真正可续训的S/D/双Adam/RNG。
        state = {'kind': 'development_checkpoint', 'student': student.state_dict(), 'discriminator': discriminator.state_dict(),
                 'student_optimizer': optimizer.state_dict(), 'discriminator_optimizer': d_optimizer.state_dict(),
                 'rng_state': baseline.training.capture_rng_state(), 'epoch': 0, 'global_step': step,
                 'configuration': config.as_dict(), 'development': scores, 'run_contract_sha256': contract_sha,
                 'lineage': record['lineage'], 'original_step0_guard_anchor': anchor}
        path = output / f'full_step_{step:06d}.pt'  # 永久保留每个100步边界，不覆盖父checkpoint。
        baseline.training.atomic_torch_save(state, path)  # 实际mapping保存，避免裸Tensor接口错误。
        baseline.save_json(output / 'latest_full.json', {'path': str(path), 'global_step': step,
            'checkpoint_sha256': baseline.sha256_file(path), 'run_contract_sha256': contract_sha})  # 明确指向最新完整文件。

    baseline.save_json(output / 'resume_state.json', {'global_step': 600, 'next_global_step': 601, 'development': resumed_scores})
    started = time.monotonic()  # 本段时限包含模型更新和检查点保存，不包含前置验真。
    try:
        for step in range(601, args.stop_global_step + 1):  # 不重复前600步，也不从1开始改种子。
            if stop['requested']:  # 若停止信号在进入本批之前到达，就不再多做一次更新。
                reason = 'requested_stop'
                break  # 已完成的状态仍由父完整检查点或本段最近完整检查点保留。
            indices = plan.for_step(step)  # 即使从一轮中间恢复，也能取得原连续训练将看到的那批语句。
            batch = baseline.batch_for(dataset, indices, 1234 + 100000 + step)  # 原全局裁剪规则。
            baseline.training.seed_everything(1234 + 200000 + step)  # 原source/score噪声和t按同一全局种子产生。
            losses = baseline.training.train_batch(batch, teacher, student, vocoder, discriminator, diffusion,
                optimizer, d_optimizer, mean, std, device, config)  # 完整复用原实现，不重写GAN/VJP数学。
            final_step = step  # 只有D和S均完成更新才推进游标。
            monitor.observe(step, 0, losses)  # 延续完整历史窗口，而不是重新从0观察。
            with (output / 'step_losses.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'step': step, 'indices': indices, 'elapsed_seconds': time.monotonic() - started, **losses}) + '\n')
            reached_time = time.monotonic() - started >= args.max_seconds
            if step % 100 == 0 or step == args.stop_global_step or reached_time or stop['requested']:
                last_scores = baseline.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
                item = {'step': step, 'elapsed_seconds': time.monotonic() - started, **last_scores}
                with (output / 'development.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(item) + '\n')  # 保留所有检查读数，不只保留较好结果。
                checkpoint(step, last_scores)  # 使用本步真实S/D/Adam状态完整落盘。
                if last_scores['mel_l1'] < best:
                    best = last_scores['mel_l1']
                    baseline.save_json(output / 'best_development.json', item)  # 仅开发读数，不代替CER等客观评估。
                baseline.save_json(output / 'status.json', {'status': 'running', **item})
                print(json.dumps(item), flush=True)  # 主线程及时读取续训实际进度。
                try:
                    assert_guard(last_scores, anchor)  # 永远锚定原S0，不以600步结果重新放宽保护线。
                except RuntimeError:
                    reason = 'original_step0_development_regression_guard'
                    break  # 先保留此完整状态，再交由主线程决定后续。
                if reached_time or stop['requested']:
                    reason = 'requested_stop' if stop['requested'] else 'time_budget_complete'
                    break
        for name, model in (('teacher', teacher), ('vocoder', vocoder)):
            if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Frozen model boundary changed: ' + name)  # 教师和G都没有优化器所有权。
        verify_files(dependencies)  # 结束时复核父检查点、原源码及所有冻结来源字节。
    except Exception as error:
        monitor.finish('failed', final_step, 0)  # 不把失败进程留成running，也不伪造未完整保存的checkpoint。
        baseline.save_json(output / 'status.json', {'status': 'failed', 'global_step': final_step,
            'error': str(error), 'latest_full_pointer': str(output / 'latest_full.json'), 'parent_full_checkpoint': str(parent_path)})
        raise  # 异常退出，由主线程检查，不自动重启。
    monitor.finish(reason, final_step, 0)
    summary = {'status': reason, 'global_step': final_step, 'start_after_global_step': 600,
               'completed_new_steps': final_step - 600, 'elapsed_seconds': time.monotonic() - started,
               'initial_parent_development': resumed_scores, 'final_development': last_scores,
               'original_step0_guard_anchor': anchor, 'best_development_mel_l1_in_this_segment': best,
               'run_contract_sha256': contract_sha, 'lineage': record['lineage']}
    baseline.save_json(output / 'completion.json', summary)
    baseline.save_json(output / 'status.json', summary)
    print('ACDD_CONTINUATION_FINISHED', reason, final_step, flush=True)  # 不能把2000/6000开发段写成正式100epoch完成。


if __name__ == '__main__':  # CPU测试导入辅助函数时不会进入GPU主流程。
    main()  # 仅在主线程明确批准续训并提供全部父身份后执行。
