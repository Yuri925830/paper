# 从真实半学习率3227完整状态继续有限验证；保留旧监测历史，绝不改变旧驱动的全局常量。
"""Bounded480 continuation from half-LR3227 with inherited strict trend guards."""

import argparse  # 执行必须提供已审核父SHA和显式GPU许可。
import copy  # 只在独立副本中适配旧Adam检查器，真实学习率和moment不重置。
import hashlib  # 导入前核验冻结辅助代码。
import json  # 新契约、逐步记录、终点收据分别保存。
import math  # 立即拒绝非有限损失和无效监测值。
import os  # 查询空间，预留系统3GiB；不删除任何旧文件。
import platform  # 锁定原训练解释器和库版本。
import signal  # 用户停止在完整D/S批次边界响应。
import sys  # Windows只允许编辑；项目计算只在Matpool。
import time  # 1200秒上限，不因训练结果临时加时。
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not ROOT.is_dir():
    raise RuntimeError('Execute only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
PAIR_FILE = TOOLS / 'run_acdd_paired_lr120_v1.py'
PAIR_SHA = '6bdf773152950226e8db6716ab64a977f30b3b72588322dd5aa54aee2b318bc0'
if hashlib.sha256(PAIR_FILE.read_bytes()).hexdigest() != PAIR_SHA:
    raise RuntimeError('Frozen paired-training helper changed')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch  # main之外不构建GPU模型；测试只使用小CPU网络。
import run_acdd_paired_lr120_v1 as pair

base, training, resume, probe = pair.base, pair.training, pair.resume, pair.probe
DEV = pair.DEV
PAIRED = DEV / 'acdd_lr_paired120_from3107_v1'
PARENT = PAIRED / 'primary_s1e4'
PARENT_SHA = 'f98b0763ec5a2edbd23d5d493e9dac549f85cf40cfac71e79f0b462f6fc6beb0'
PARENT_CONTRACT = '25c6e14693a940d94e18a85f50e8e3c08bbfbf1e15f53580117662a0f86ad909'
WINDOW_SHA = '17042250baec7af6ac701d3a7637ae293d200aa2af8a641d05f52d973631eb18'
MONITOR_SHA = '185a4499b83858ff1073d04a76d9de86fce87718f7ddc8448151e972c0d2859d'
TARGET_SHA = '2aedd82fb269ed07c98b206c41e071c73f64555c57ed4ce3faf1eb8db12144bb'
PROTOCOL = DEV / 'ACDD3227_LR_CONTINUATION480_PROTOCOL_CN.md'
PROTOCOL_SHA = '50651bd3c37dc42307785f14ef06e7daf5bfb66d7db2a6c481de98031d42c152'
START, STOP, ORIGIN = 3227, 3707, 3107  # 本段最多480步；历史横轴继续从原3107起点计算。
ACTUAL_LRS = {'student': .0001, 'discriminator': .0002}


class FinalBudget:
    def __init__(self, directory, full_bytes, student_bytes):
        self.directory = directory
        self.pending = {'full': full_bytes, 'student': student_bytes}  # 只写一个全新终点full和一个S。

    def require(self):
        need = pair.SYSTEM_RESERVE + pair.REPORT_RESERVE + sum(self.pending.values())
        free = pair.available_bytes(self.directory)
        if free < need:
            raise pair.InsufficientDisk(f'available={free}; required={need}; pending={self.pending}')
        return {'available_bytes': free, 'required_bytes': need, 'pending': dict(self.pending)}

    def committed(self, name):
        self.pending.pop(name)  # 只有文件已原子写成才扣除预算。


def cpu_adam_copy(value):
    # CUDA moment逐张量移到CPU副本，避免为了校验额外占用大量GPU显存。
    return {'param_groups': copy.deepcopy(value['param_groups']), 'state': {
        identifier: {key: item.detach().cpu().clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item)
                     for key, item in state.items()} for identifier, state in value['state'].items()}}


def check_owner(optimizer, model):
    owned = [id(p) for group in optimizer.param_groups for p in group['params']]
    actual = [id(p) for p in model.parameters()]
    if len(owned) != len(actual) or len(set(owned)) != len(owned) or set(owned) != set(actual):
        raise RuntimeError('Optimizer does not exclusively and completely own its intended model')


def restore_actual_parent(payload, student, discriminator, optimizer, d_optimizer, config, device):
    required = {'student', 'discriminator', 'student_optimizer', 'discriminator_optimizer', 'rng_state',
        'global_step', 'run_contract_sha256', 'configuration', 'effective_optimizer_lrs', 'development'}
    if not required <= set(payload) or payload['global_step'] != START or payload['run_contract_sha256'] != PARENT_CONTRACT:
        raise RuntimeError('Only the complete declared3227 parent can resume')
    if payload['configuration'] != config.as_dict() or payload['effective_optimizer_lrs'] != ACTUAL_LRS:
        raise RuntimeError('Parent base configuration or actual role-specific LR changed')
    check_owner(optimizer, student)
    check_owner(d_optimizer, discriminator)
    if {id(p) for p in student.parameters()} & {id(p) for p in discriminator.parameters()}:
        raise RuntimeError('Student and D cannot share parameter objects')
    if [g['lr'] for g in optimizer.param_groups] != [.0001] or [g['lr'] for g in d_optimizer.param_groups] != [.0002]:
        raise RuntimeError('Construct both optimizers with their actual inherited learning rates')
    saved_s = cpu_adam_copy(payload['student_optimizer'])
    saved_d = cpu_adam_copy(payload['discriminator_optimizer'])
    if len(saved_s['param_groups']) != 1 or saved_s['param_groups'][0]['lr'] != .0001:
        raise RuntimeError('Saved student Adam is not the half-LR parent')
    inspected = copy.deepcopy(saved_s)
    inspected['param_groups'][0]['lr'] = .0002  # 仅适配旧检查器；这个副本从不加载到真实优化器。
    resume.validate_adam_state(inspected, student, START)
    resume.validate_adam_state(saved_d, discriminator, START)
    before = {'student_adam': probe.tree_sha(saved_s), 'discriminator_adam': probe.tree_sha(saved_d),
              'rng': probe.tree_sha(payload['rng_state'])}
    student.load_state_dict(payload['student'], strict=True)
    discriminator.load_state_dict(payload['discriminator'], strict=True)
    optimizer.load_state_dict(saved_s)  # 原样加载1e-4、3227计数与全部moment，不先加载2e-4再改回。
    d_optimizer.load_state_dict(saved_d)
    check_owner(optimizer, student)
    check_owner(d_optimizer, discriminator)
    if probe.tree_sha(optimizer.state_dict()) != before['student_adam'] or probe.tree_sha(d_optimizer.state_dict()) != before['discriminator_adam']:
        raise RuntimeError('Loading altered the actual Adam state')
    student.train()
    discriminator.train()
    training.set_requires_grad(student, True)
    training.set_requires_grad(discriminator, True)
    training.restore_rng_state(payload['rng_state'], device)
    # CPU测试隐藏CUDA时只检查其可恢复的三种CPU随机流，实际GPU启动检查全部四项。
    actual_rng = training.capture_rng_state()
    wanted_rng = payload['rng_state'] if device.type == 'cuda' else {**payload['rng_state'], 'cuda_all': []}
    if probe.tree_sha(actual_rng) != probe.tree_sha(wanted_rng):
        raise RuntimeError('Complete parent RNG did not restore')
    return before


def strict_three(values):
    if any(not math.isfinite(float(value)) for value in values):
        raise RuntimeError('Nonfinite trend history')
    return len(values) >= 3 and values[-3] < values[-2] < values[-1]


def trend_state(windows, monitors, anchor, baseline3227):
    # 不再沿用2%/1%累计门槛；最近三点严格上涨就记录，判别器分项上涨仅报告不单独停。
    rising = [key for key in pair.LOSS_KEYS if strict_three([row['means'][key] for row in windows[-3:]])]
    scores = monitors[-1]['development']
    reasons = []
    try:
        resume.assert_guard(scores, anchor)
    except RuntimeError:
        reasons.append('original_S0_guard')
    if not math.isfinite(scores['mel_l1']) or not math.isfinite(scores['prediction_rms']):
        raise RuntimeError('Nonfinite development monitor')
    if scores['mel_l1'] > baseline3227['mel_l1'] * 1.01:
        reasons.append('development_above3227_by1percent')
    if strict_three([row['development']['mel_l1'] for row in monitors[-3:]]):
        reasons.append('development_three_checks_rising')
    if 'student_total' in rising:
        reasons.append('moving_total_three_windows_rising')
    return {'stop_reasons': reasons, 'strictly_rising_loss_fields': rising,
            'discriminator_alone_is_not_a_stop_condition': True}


def load_history(parent):
    read = lambda name: [json.loads(line) for line in (PARENT / name).read_text().splitlines()]
    windows, monitors, rows = read('loss_windows.jsonl'), read('monitor.jsonl'), read('step_losses.jsonl')
    if len(rows) != 120 or [r['global_step'] for r in rows] != list(range(3108, 3228)):
        raise RuntimeError('Parent raw loss history is not the complete120-step arm')
    if len(windows) != 6 or [r['global_step'] for r in windows] != list(range(3127, 3228, 20)):
        raise RuntimeError('All six existing20-step windows are required')
    for i, window in enumerate(windows):
        if pair.window_record(rows[20*i:20*(i+1)]) != window:
            raise RuntimeError('Inherited window differs from original raw losses')
    if [r['global_step'] for r in monitors] != list(range(3107, 3228, 20)) or [r['updates'] for r in monitors] != list(range(0,121,20)):
        raise RuntimeError('All seven original fixed monitors must be inherited without duplicating3227')
    if monitors[-1] != parent['last_monitor']:
        raise RuntimeError('Last historical monitor is not the parent endpoint')
    return windows, monitors


def complete_batch(state, step, call):
    state['partial_batch_may_have_updated_D'] = True
    losses = call()  # 只由main传入原生产train_batch；D半步异常不会走到下一行。
    state.update(completed_global_step=step, new_updates=step-START, updates=step-ORIGIN,
                 partial_batch_may_have_updated_D=False)
    pair.finite_losses(losses)
    return losses


def validate_replay(actual, expected):
    probe.validate_development_replay(actual['development'], expected['development'])
    if len(actual['development']['per_utterance_mel_l1']) != 32 or len(expected['development']['per_utterance_mel_l1']) != 32:
        raise RuntimeError('Replay must retain all32 development utterances')
    if any(abs(a-b) > 1e-7 for a,b in zip(actual['development']['per_utterance_mel_l1'], expected['development']['per_utterance_mel_l1'])):
        raise RuntimeError('The original32 per-utterance errors did not replay')
    if any(not math.isfinite(actual['fixed_train'][key]) or abs(actual['fixed_train'][key]-value) > 2e-5
           for key,value in expected['fixed_train'].items()):
        raise RuntimeError('The frozenD3107/targetA and refreshedB endpoint did not replay')


def monitored(call, frozen, fingerprints, device):
    # 同时核对冻结模型的全部参数与buffer（包括非持久Hann窗），恢复全部随机流。
    with pair.preserve_rng(device):
        result = call()
        pair.assert_frozen(frozen, fingerprints)
        return result


def save_final(output, state, common, student, discriminator, optimizer, d_optimizer, budget):
    step = state['completed_global_step']
    if step == START:
        return  # 没有新增完整更新时保留父文件，不再复制一份大模型。
    if state['partial_batch_may_have_updated_D']:
        raise RuntimeError('Never save a partial alternating D/S update as a full checkpoint')
    full_path, s_path = output / 'final_full.pt', output / f'student_step_{step:06d}.pt'
    if any(p.exists() or p.is_symlink() for p in (full_path, s_path)):
        raise RuntimeError('Final files are immutable; never overwrite or retry on existing targets')
    budget.require()
    full = dict(common, student=student.state_dict(), discriminator=discriminator.state_dict(),
        student_optimizer=optimizer.state_dict(), discriminator_optimizer=d_optimizer.state_dict(),
        rng_state=training.capture_rng_state())
    training.atomic_torch_save(full, full_path)
    state['last_full_saved_global_step'] = step  # full原子成功后立即推进，不等待后续SHA/S/JSON成功。
    state['full_snapshot'] = {'path': str(full_path), 'global_step': step, 'sha256': None, 'committed': True}
    budget.committed('full')
    state['full_snapshot']['sha256'] = base.sha256_file(full_path)
    full_path.chmod(0o444)
    state['full_snapshot']['frozen_readonly'] = True
    budget.require()
    training.atomic_torch_save(dict(common, student=student.state_dict()), s_path)
    state['student_snapshot'] = {'path': str(s_path), 'global_step': step, 'sha256': None, 'committed': True}
    budget.committed('student')
    state['student_snapshot']['sha256'] = base.sha256_file(s_path)
    s_path.chmod(0o444)
    state['student_snapshot']['frozen_readonly'] = True
    base.save_json(output / 'checkpoint_receipt.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parent-checkpoint-sha256', required=True)
    parser.add_argument('--max-seconds', type=int, default=1200)
    parser.add_argument('--run-approved', action='store_true')
    args = parser.parse_args()
    if not args.run_approved or args.parent_checkpoint_sha256 != PARENT_SHA or not 1 <= args.max_seconds <= 1200:
        raise RuntimeError('Explicit reviewed parent and at most1200 seconds required')
    training.assert_rented_server(ROOT)
    locks = training.acquire_runtime_locks(ROOT)  # 保留互斥锁引用，其他GPU工作不能与本次并行。
    if training.git_head(ROOT) != base.EXPECTED_HEAD:
        raise RuntimeError('Frozen clean production source changed')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'):
        raise RuntimeError('Original training runtime required')
    if Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or not torch.cuda.is_available():
        raise RuntimeError('Registered myconda CUDA interpreter required')
    output = args.output_dir.resolve()
    if output.parent != DEV or output.exists():
        raise RuntimeError('Use a new direct child of the fixed development directory')
    hashes = {str(PAIR_FILE): PAIR_SHA, str(PROTOCOL): PROTOCOL_SHA, str(PARENT / 'final_full.pt'): PARENT_SHA,
        str(PAIRED / 'run_contract.json'): 'f60a052e712080d48207be35a9595804a579ac152acdbb8c9b450108a8fde303',
        str(PARENT / 'arm_completion.json'): '50eee0fadb5b25de06e06beec23fa2e4b4f6e9aa1ab38f6aee1128d17ce17dcb',
        str(PARENT / 'step_losses.jsonl'): '01cbc2832a7efa32acf9d153a2bb7372054e6c67dd64e5010a7f3684f42c524a',
        str(PARENT / 'loss_windows.jsonl'): WINDOW_SHA, str(PARENT / 'monitor.jsonl'): MONITOR_SHA,
        str(PAIRED / 'fixed_teacher_target.pt'): TARGET_SHA,
        str(PARENT / 'student_step_003227.pt'): '88da1f14234c9dcb164cdd249d47862b9ba105f1b7a1a858b4a18f9fa921b1dc'}
    resume.verify_files(hashes)
    contract = json.loads((PAIRED / 'run_contract.json').read_text())
    parent = json.loads((PARENT / 'arm_completion.json').read_text())
    if training.canonical_json_sha256(contract) != PARENT_CONTRACT or parent['completed_global_step'] != START or parent['full_snapshot']['sha256'] != PARENT_SHA:
        raise RuntimeError('Parent contract or complete endpoint changed')
    hashes.update(contract['file_sha256'])
    development, dev_hashes = resume.load_development(DEV / 'development_contract.json')
    hashes.update(dev_hashes)
    if development['retained_train_indices'] != contract['training_indices'] or development['excluded_train_indices'] != contract['excluded_train_indices']:
        raise RuntimeError('The52-utterance exclusion or retained training indices changed')
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if config.as_dict() != contract['configuration'] or contract['seed'] != 1234:
        raise RuntimeError('Base ACDD/math/BNF configuration changed')
    grand = json.loads((resume.PARENT / 'run_contract.json').read_text())
    if training.collect_source_fingerprints(ROOT) != grand['source_sha256']:
        raise RuntimeError('Production source fingerprints changed')
    hashes[str(Path(__file__).resolve())] = base.sha256_file(Path(__file__))
    resume.verify_files(hashes)
    windows, monitors = load_history(parent)
    anchor = contract['original_step0_guard_anchor']
    baseline3227 = parent['last_monitor']['development']
    if anchor['mel_l1'] != .19941353797912598 or anchor['prediction_rms'] != .9994737505912781:
        raise RuntimeError('OriginalS0 protection cannot be rebased')
    if trend_state(windows, monitors, anchor, baseline3227)['stop_reasons']:
        raise RuntimeError('Inherited endpoint already violates the new declared stop rules')
    budget = FinalBudget(DEV, (PARENT / 'final_full.pt').stat().st_size, (PARENT / 'student_step_003227.pt').stat().st_size)
    capacity = budget.require()
    torch.set_num_threads(2)  # 原开发驱动设置，其他TF32/cudnn开关保持原值。
    training.seed_everything(1234)
    device = torch.device('cuda')
    started = time.monotonic()
    stop = {'requested': False}
    def stop_at_boundary(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, stop_at_boundary)
    signal.signal(signal.SIGINT, stop_at_boundary)
    payload = resume.load_torch_checkpoint(PARENT / 'final_full.pt')
    if payload['development'] != baseline3227:
        raise RuntimeError('Full checkpoint does not contain the actual parent development endpoint')
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387:
        raise RuntimeError('Original training manifest length changed')
    plan = resume.GlobalBatchPlan(contract['training_indices'])
    if plan.batches_per_pass != 1229:
        raise RuntimeError('Absolute batch plan changed')
    raw_inputs = resume.load_torch_checkpoint(pair.INPUT_DIR / 'fixed_inputs.pt')
    if {key: probe.tensor_sha(value) for key,value in raw_inputs.items()} != contract['input_sha256']:
        raise RuntimeError('OriginalTRAIN3108/fixed32 inputs changed')
    inputs = {key: value.to(device) for key,value in raw_inputs.items()}
    del raw_inputs
    target_payload = resume.load_torch_checkpoint(PAIRED / 'fixed_teacher_target.pt')
    target = target_payload['teacher_target'].to(device)
    if target_payload['global_source_step'] != 3107 or probe.tensor_sha(target) != contract['fixed_teacher_target_sha256'] or target_payload['sha256'] != contract['fixed_teacher_target_sha256']:
        raise RuntimeError('Old3107 teacher targetA changed; do not regenerate it from3227')
    del target_payload
    teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
    vocoder, vconfig, _ = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
        base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
    discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
    fixed_d = base.build_paper_discriminator(ROOT, vconfig, config, device)
    original_d = resume.load_torch_checkpoint(pair.PARENT / 'latest_full.pt')  # 固定监测D必须仍是3107，不能偷换成训练D3227。
    if original_d['global_step'] != 3107 or original_d['run_contract_sha256'] != pair.PARENT_CONTRACT:
        raise RuntimeError('Fixed diagnosticD is not the original3107 state')
    fixed_d.load_state_dict(original_d['discriminator'], strict=True)
    del original_d
    fixed_d.eval()
    training.set_requires_grad(fixed_d, False)
    diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
    mean, std = base.load_mel_statistics(ROOT / 'data', device)
    frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
    fingerprints = {name: pair.model_sha(model) for name,model in frozen.items()}
    if fingerprints != contract['fixed_frozen_models_sha256']:
        raise RuntimeError('Frozen teacher/G/D3107 parameters or buffers changed')
    pair.assert_frozen(frozen, fingerprints)
    optimizer = torch.optim.Adam(student.parameters(), lr=.0001, betas=(.5,.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(.5,.9))
    restored = restore_actual_parent(payload, student, discriminator, optimizer, d_optimizer, config, device)
    del payload
    def evaluate(step):
        return monitored(lambda: pair.monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target,
            mean, std, config, device, step), frozen, fingerprints, device)
    replay = evaluate(START)
    validate_replay(replay, monitors[-1])  # 只验真，不把3227重新追加到历史中。
    budget.require()
    lineage = {'parent_checkpoint': str(PARENT / 'final_full.pt'), 'parent_checkpoint_sha256': PARENT_SHA,
        'parent_contract_sha256': PARENT_CONTRACT, 'start_after_global_step': START,
        'original_step0_checkpoint_sha256': resume.ANCHOR_SHA}
    record = {'kind': 'acdd_half_lr_continue480_development', 'lineage': lineage, 'configuration': config.as_dict(),
        'effective_optimizer_lrs': ACTUAL_LRS, 'configuration_interpretation': 'Base template is2e-4; actual inherited student1e-4 and D2e-4 are authoritative.',
        'method_extension': 'Continue validated half student LR; all ACDD targets/coefficients and teacher/G/BNF remain unchanged. Not a strict2024 LR reproduction.',
        'seed': 1234, 'stop_global_step': STOP, 'maximum_new_updates': 480, 'max_seconds': args.max_seconds,
        'training_indices': contract['training_indices'], 'excluded_train_indices': contract['excluded_train_indices'],
        'reconstruction_dev_indices': contract['reconstruction_dev_indices'], 'development_contract_file_sha256': resume.DEVELOPMENT_SHA,
        'formal_test_used_for_selection': False, 'automatic_metric_evaluation': False,
        'batch_plan': {'first_step': 3228, 'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((step-1)/1229)',
            'crop_seed': '1234+100000+step', 'noise_seed': '1234+200000+step'},
        'inherited_history': {'windows_file': str(PARENT / 'loss_windows.jsonl'), 'windows_sha256': WINDOW_SHA, 'windows_count':6,
            'monitor_file': str(PARENT / 'monitor.jsonl'), 'monitor_sha256': MONITOR_SHA, 'monitor_count':7, 'updates_origin_global_step':ORIGIN},
        'baseline3227_development': baseline3227, 'baseline3227_replay': replay,
        'original_step0_guard_anchor': anchor, 'guard_ratios': {'mel_l1_upper':1.10,'prediction_rms_lower':.80},
        'stop_rules': {'total': 'last3 complete20 means strictly increase, no2percent gate',
            'development': 'last3 fixed checks strictly increase, no1percent cumulative gate',
            'single_development': 'mel_l1 > baseline3227*1.01', 'originalS0':'unchanged',
            'other':['nonfinite','OOM/exception','disk','time','user_stop']},
        'fixed_train_inputs_sha256': contract['input_sha256'], 'fixed_teacher_target_sha256': contract['fixed_teacher_target_sha256'],
        'fixed_teacher_target_file': str(PAIRED / 'fixed_teacher_target.pt'), 'fixed_frozen_models_sha256': fingerprints,
        'fixed_development_inputs': contract['fixed_development_inputs'], 'fixed_development_source_noise_sha256': contract['fixed_development_source_noise_sha256'],
        'storage': {'policy':'one new immutable actual-end full and one S; no intermediate model snapshots', 'initial_budget':capacity,
            'system_reserve_bytes':pair.SYSTEM_RESERVE, 'report_reserve_bytes':pair.REPORT_RESERVE},
        'restored_optimizer_rng_sha256': restored, 'file_sha256': hashes,
        'runtime': {'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'executable':str(Path(sys.executable).resolve()),
            'gpu_name':torch.cuda.get_device_name(0),'gpu_capability':list(torch.cuda.get_device_capability(0)),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}}
    output.mkdir()
    base.save_json(output / 'run_contract.json', record)
    contract_sha = training.canonical_json_sha256(record)
    for name in ('loss_windows.jsonl','monitor.jsonl'):
        with (output / name).open('xb') as handle:
            handle.write((PARENT / name).read_bytes())  # 原记录逐字节继承，无重复起点或重基线。
    state = {'status':'running','completed_global_step':START,'new_updates':0,'updates':120,'last_full_saved_global_step':START,
        'full_snapshot':None,'student_snapshot':None,'partial_batch_may_have_updated_D':False,'contract_sha256':contract_sha,
        'parent_checkpoint_sha256':PARENT_SHA,'effective_optimizer_lrs':ACTUAL_LRS,'last_monitor':monitors[-1]}
    base.save_json(output / 'resume_state.json', dict(state, replay_verified=True, next_global_step=3228))
    pending, reason = [], 'maximum480_updates_completed'
    try:
        for step in range(START+1,STOP+1):
            if stop['requested'] or time.monotonic()-started >= args.max_seconds:
                reason = 'requested_stop' if stop['requested'] else 'time_budget'
                break
            try:
                budget.require()
            except pair.InsufficientDisk:
                reason = 'disk_space_guard'
                break
            indices = plan.for_step(step)
            if set(indices) & set(contract['excluded_train_indices']):
                raise RuntimeError('New training batch overlaps52 excluded utterances')
            batch = base.batch_for(dataset, indices, 1234+100000+step)
            training.seed_everything(1234+200000+step)
            losses = complete_batch(state, step, lambda: training.train_batch(batch,teacher,student,vocoder,discriminator,
                diffusion,optimizer,d_optimizer,mean,std,device,config))  # 原生产函数原样执行，A只在监测中使用。
            row = {'global_step':step,'updates':step-ORIGIN,'new_updates':step-START,'indices':indices,
                'actual_optimizer_lrs':{'student':optimizer.param_groups[0]['lr'],'discriminator':d_optimizer.param_groups[0]['lr']},
                'elapsed_seconds':time.monotonic()-started,**losses}
            pair.append_json(output / 'step_losses.jsonl',row)
            pending.append(row)
            if (step-START)%20 == 0:
                window = pair.window_record(pending)
                pending=[]
                windows.append(window)
                pair.append_json(output / 'loss_windows.jsonl',window)
                entry=evaluate(step)
                monitors.append(entry)
                state['last_monitor']=entry
                pair.append_json(output / 'monitor.jsonl',entry)
                trend=trend_state(windows,monitors,anchor,baseline3227)
                state['trend']=trend
                pair.append_json(output / 'trend.jsonl',{'global_step':step,'new_updates':step-START,**trend})
                print(json.dumps({'global_step':step,'new_updates':step-START,'means':window['means'],'monitor':entry,'trend':trend}),flush=True)
                if trend['stop_reasons']:
                    reason=trend['stop_reasons'][0]
                    break
            base.save_json(output / 'status.json',state)
        step=state['completed_global_step']
        if monitors[-1]['global_step'] != step:
            entry=evaluate(step)
            monitors.append(entry)
            state['last_monitor']=entry
            pair.append_json(output / 'monitor.jsonl',entry)
        trend=trend_state(windows,monitors,anchor,baseline3227)
        state['trend']=trend
        if trend['stop_reasons']:
            reason=trend['stop_reasons'][0]
        pair.assert_frozen(frozen,fingerprints)
        if {key:probe.tensor_sha(value) for key,value in inputs.items()} != contract['input_sha256'] or probe.tensor_sha(target) != contract['fixed_teacher_target_sha256']:
            raise RuntimeError('Fixed inputs or old targetA changed')
        pair.validate_effective_adam(optimizer,d_optimizer,student,discriminator,.0001,step)
        state.update(reason=reason,partial_window_steps=len(pending))
        common={'kind':'acdd_half_lr_continuation_checkpoint','global_step':step,'epoch':0,'configuration':config.as_dict(),
            'effective_optimizer_lrs':ACTUAL_LRS,'configuration_interpretation':record['configuration_interpretation'],
            'development':monitors[-1]['development'],'run_contract_sha256':contract_sha,'lineage':lineage,
            'original_step0_guard_anchor':anchor,'score_semantics':'reconstruction_clean_x0_sqrt_alpha_bar','stop_reason':reason}
        save_final(output,state,common,student,discriminator,optimizer,d_optimizer,budget)
        resume.verify_files(hashes)
        state['status']='completed' if reason=='maximum480_updates_completed' else 'stopped'
        state['elapsed_seconds']=time.monotonic()-started
        base.save_json(output / 'completion.json',state)
        print('HALF_LR_CONTINUATION_FINISHED '+json.dumps(state),flush=True)
    except Exception as error:
        pair.report_failure(output,state,error)  # 已保存full的游标不回退；半步失败不尝试伪造完整保存。
        raise


if __name__ == '__main__':
    main()  # 只有root审核并显式启动时才训练；此准备任务仅做CPU测试。
