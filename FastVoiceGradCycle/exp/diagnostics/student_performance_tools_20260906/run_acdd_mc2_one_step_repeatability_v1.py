"""两个全新进程各做一次3928更新；只读观察3929，不保存模型或继续训练。"""

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
DLR_FILE = TOOLS / 'run_acdd_mc2_dlr_paired120_from3927_v1.py'
DLR_SHA = 'e2eb8d53f18999874dab447d1b124b6c34bd0a0d056d0323df59b320e14ff821'
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Run only on the registered Matpool rental')
if hashlib.sha256(DLR_FILE.read_bytes()).hexdigest() != DLR_SHA:
    raise RuntimeError('Frozen restoration/monitor primitives changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_mc2_dlr_paired120_from3927_v1 as d

old, previous, cross, pair, base = d.old, d.previous, d.cross, d.pair, d.base
training, resume, probe = d.training, d.resume, d.probe
DEV = d.DEV
OUTPUT = DEV / 'acdd_mc2_one_step_repeatability_v1'
PROTOCOL = DEV / 'MC2_ONE_STEP_REPEATABILITY_PROTOCOL_CN.md'
HELPER = 'acdd_mc2_train_batch_3928_4047_v1'
CHILDREN = ('child_a', 'child_b')
THREAD_ENV = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
META_LIMIT, FAILURE_RESERVE = 32 * 1024**2, 64 * 1024
SECONDS, GRACE = 600, 90
MONITOR_COUNTS = {'teacher': 21, 'student': 4, 'vocoder': 2, 'active_D_branches': 0, 'fixed_D_branches': 16}
TRAIN_COUNTS = {'teacher': 2, 'student': 2, 'vocoder': 3, 'active_D_branches': 32, 'fixed_D_branches': 0}
PROBE_COUNTS = {'teacher': 2, 'student': 1, 'vocoder': 0, 'active_D_branches': 0, 'fixed_D_branches': 0}
CHILD_LIMITS = {k: MONITOR_COUNTS[k] + TRAIN_COUNTS[k] + PROBE_COUNTS[k] for k in TRAIN_COUNTS}
TOTAL_LIMITS = {k: 2 * v for k, v in CHILD_LIMITS.items()}
LOCK_PATHS = [ROOT / 'exp/locks' / name for name in (
    'text_unseen_custom_hifigan_pipeline.lock', 'text_unseen_custom_hifigan_train.lock',
    'text_unseen_fastvoicegrad_student.lock', 'text_unseen_fastvoicegrad_student_inference.lock')]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    for name in ('driver', 'tests', 'protocol'): parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--child', choices=CHILDREN)
    parser.add_argument('--supervisor-pid', type=int)
    parser.add_argument('--shared-start', type=float)
    parser.add_argument('--contract-file-sha256')
    parser.add_argument('--lock-fds')
    args = parser.parse_args(argv)
    if not args.run_approved: raise RuntimeError('Explicit bounded diagnostic approval required')
    fields = ('driver_sha256', 'tests_sha256', 'protocol_sha256') + (('contract_file_sha256',) if args.child else ())
    for name in fields:
        value = getattr(args, name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Require reviewed full SHA: ' + name)
    if args.child:
        if not args.supervisor_pid or args.shared_start is None or not math.isfinite(args.shared_start) or not args.lock_fds:
            raise RuntimeError('Child must inherit the live supervisor, deadline and four locks')
    elif any(getattr(args, name) is not None for name in ('supervisor_pid', 'shared_start', 'contract_file_sha256', 'lock_fds')):
        raise RuntimeError('Supervisor cannot accept invented child identity')
    return args


class Budget:
    def __init__(self, output): self.output, self.aborting = Path(output), False
    def used(self):
        total = 0
        for path in self.output.rglob('*') if self.output.exists() else ():
            cross.regular(path, directory=path.is_dir())
            if path.is_file():
                if path.name.endswith(('.pt', '.pt.tmp')): raise RuntimeError('Diagnostic may not create model artifacts')
                total += path.stat().st_size
        return total
    def require(self, extra=0):
        used = self.used(); maximum = META_LIMIT if self.aborting else META_LIMIT - FAILURE_RESERVE
        if used + extra > maximum: raise RuntimeError('Complete diagnostic metadata/log budget exceeded')
        need = previous.RESERVE + previous.DIR_RESERVE + (extra if self.aborting else META_LIMIT - used)
        free = pair.available_bytes(self.output.parent)
        if free < need: raise pair.InsufficientDisk('available=%d required=%d' % (free, need))
        return {'available_bytes': free, 'required_bytes': need, 'used_bytes': used}
    def write(self, path, value, append=False, terminal=False):
        path = Path(path)
        if self.output not in path.parents or path.suffix not in ('.json', '.jsonl'):
            raise RuntimeError('Only JSON inside the new diagnostic output')
        if terminal != (path.name == 'completion.json') or (terminal and append):
            raise RuntimeError('Success publication requires the terminal path')
        cross.regular(path.parent, directory=True)
        data = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        self.require(len(data) + 1)
        if append:
            if path.exists(): cross.regular(path)
            with path.open('ab') as handle: handle.write(data + b'\n'); handle.flush(); os.fsync(handle.fileno())
        else:
            temporary = path.with_suffix(path.suffix + '.tmp')
            if any(p.exists() or p.is_symlink() for p in (path, temporary)): raise RuntimeError('Never overwrite diagnostic evidence')
            with temporary.open('xb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.rename(temporary, path)
        if not terminal: self.require()  # 最终completion发布后不再执行可失败的检查。


class Ledger(old.ForwardLedger):
    def __init__(self): self.counts = dict.fromkeys(CHILD_LIMITS, 0); self.handles = []; self.check = lambda: None
    def before(self, name):
        def observe(module, arguments):
            self.check()
            if self.counts[name] >= CHILD_LIMITS[name]: raise RuntimeError('Per-child forward limit: ' + name)
            self.counts[name] += 1
        return observe


def runtime():
    actual = {name: os.environ.get(name) for name in THREAD_ENV}
    # 环境变量不保证Torch实际采用同一线程数；读取真实值，不为通过门槛强改线程。
    threads = torch.get_num_threads()
    if actual != THREAD_ENV or type(threads) is not int or threads < 1:
        raise RuntimeError('Require explicit diagnostic environment and a positive actual thread count')
    if ((platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6')
            or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve()
            or cross.flags() != cross.ORIGINAL_FLAGS):
        raise RuntimeError('Original runtime and precision must remain unchanged')
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'flags': cross.flags(), 'thread_environment': actual,
        'torch_threads': threads, 'torch_interop_threads': torch.get_num_interop_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'old_CONTROL71128_thread_environment': 'not_recorded_cannot_claim_same'}


def tensor_summary(value):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided: raise RuntimeError('Require a dense tensor')
    # 此CPU复制会同步CUDA；只读，不改原tensor、梯度或计算图。
    cpu = value.detach().cpu().contiguous()
    finite = bool(torch.isfinite(cpu).all())
    return {'sha256': hashlib.sha256(cpu.numpy().tobytes()).hexdigest(), 'shape': list(value.shape),
        'dtype': str(value.dtype), 'device': str(value.device), 'stride': list(value.stride()),
        'finite': finite, 'l2_norm': float(cpu.double().norm()) if finite else None}


def tensor_group(values):
    result = {name: tensor_summary(value) for name, value in values}
    finite = all(item['finite'] for item in result.values())
    return {'tensors': result, 'tensor_count': len(result), 'finite': finite,
        'l2_norm': math.sqrt(math.fsum(v['l2_norm'] ** 2 for v in result.values())) if finite else None}


def gradient_summary(model):
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    return dict(tensor_group((name, p.grad) for name, p in model.named_parameters() if p.grad is not None), missing=missing)


def after_step_summary(model, optimizer):
    values = []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not state or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}: raise RuntimeError('Incomplete real Adam state')
        values.extend((name + '.' + key, state[key]) for key in ('step', 'exp_avg', 'exp_avg_sq'))
    return {'parameters': tensor_group(model.named_parameters()), 'buffers': tensor_group(model.named_buffers()),
        'optimizer_tensors': tensor_group(values), 'actual_cursor': old.optimizer_cursor(optimizer),
        'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]}


@contextlib.contextmanager
def step_observers(student, discriminator, optimizer, d_optimizer, events, publish, check=lambda: None):
    installed = []; missing = object(); calls = {'student': 0, 'discriminator': 0}
    try:
        for role, model, opt in (('discriminator', discriminator, d_optimizer), ('student', student, optimizer)):
            original, local = opt.step, opt.__dict__.get('step', missing)
            def observed(*args, _role=role, _model=model, _opt=opt, _original=original, **kwargs):
                if args or kwargs or calls[_role] or (_role == 'student' and calls['discriminator'] != 1):
                    raise RuntimeError('Only one ordinary D.step then S.step is permitted')
                check()
                gradients = gradient_summary(_model); events[_role + '_gradient'] = gradients
                publish({'stage': _role + '_gradient', 'summary': gradients})
                if gradients['missing'] or not gradients['finite']: raise RuntimeError('Missing/nonfinite pre-step gradients')
                # _original包含冻结真实step返回观察；成功返回计数先完成，再执行本诊断摘要。
                check(); result = _original(); calls[_role] += 1
                events[_role + '_actual_step_returned'] = True
                summary = after_step_summary(_model, _opt); events[_role + '_after_step'] = summary
                publish({'stage': _role + '_after_step', 'summary': summary})
                if any(not summary[key]['finite'] for key in ('parameters', 'buffers', 'optimizer_tensors')):
                    raise RuntimeError('Nonfinite post-step model or Adam state')
                return result
            opt.step = observed; installed.append((opt, local))
        yield calls
    finally:
        for opt, local in reversed(installed):
            if local is missing: del opt.step
            else: opt.step = local


def differences(actual, expected, path='$'):
    # 只输出差异，绝不将有限数值不相等作为A/A诊断的失败门。
    if type(actual) is not type(expected): return [{'path': path, 'actual': actual, 'expected': expected, 'type_difference': True}]
    if isinstance(actual, dict):
        result = []
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected: result.append({'path': path + '.' + key, 'missing_side': 'actual' if key not in actual else 'expected'})
            else: result.extend(differences(actual[key], expected[key], path + '.' + key))
        return result
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected): return [{'path': path, 'actual_length': len(actual), 'expected_length': len(expected)}]
        return [row for i, (a, b) in enumerate(zip(actual, expected)) for row in differences(a, b, path + '[%d]' % i)]
    if isinstance(actual, float) and (not math.isfinite(actual) or not math.isfinite(expected)):
        raise RuntimeError('Nonfinite diagnostic result')
    if actual == expected: return []
    entry = {'path': path, 'actual': actual, 'expected': expected}
    if isinstance(actual, float):
        if not math.isfinite(actual) or not math.isfinite(expected): raise RuntimeError('Nonfinite diagnostic result')
        entry.update(actual_minus_expected=actual - expected, within_old_numeric_tolerance=abs(actual - expected) <= 1e-7 + 1e-6 * abs(expected))
    return [entry]


def prepare_context(args, check):
    # 不扫描未来360步，不递归父清单；只登记两批和固定monitor实际要读的文件。
    files = {p: sha for p, sha in d.FIXED_PINS.items() if not p.endswith('/acdd_mc2_fm_cross_3447_3927_v1/results.json')}
    files.update(d.NEW_SOURCE_PINS)
    files.update({str(DLR_FILE): DLR_SHA, str(TOOLS / (HELPER + '.py')): d.HELPER_SHA,
        str(TOOLS / 'acdd_mc2_train_batch_v1.py'): 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16',
        str(Path(__file__).resolve()): args.driver_sha256,
        str(TOOLS / 'test_run_acdd_mc2_one_step_repeatability_v1.py'): args.tests_sha256, str(PROTOCOL): args.protocol_sha256})
    d.check_files(files, check); d.validate_supplemental_source_pins()
    mc = importlib.import_module(HELPER)
    parent = cross.read_json(d.PARENT_RUN / 'run_contract.json')
    receipt = d.validate_parent_metadata(parent, cross.read_json(d.PARENT_RUN / 'completion.json'))
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if parent['configuration'] != config.as_dict(): raise RuntimeError('Parent mathematical configuration changed')
    development, dependencies = resume.load_development(DEV / 'development_contract.json')
    if (parent['training_indices'], parent['excluded_train_indices']) != (development['retained_train_indices'], development['excluded_train_indices']):
        raise RuntimeError('Original52 exclusions changed')
    cross.merge_hashes(files, dependencies); cross.merge_hashes(files, d.imported_source_files(parent['file_sha256'], files))
    for relative, digest in training.collect_source_fingerprints(ROOT).items():
        path = str(ROOT / relative) if not Path(relative).is_absolute() else relative
        if parent['file_sha256'].get(path) != digest: raise RuntimeError('Production source changed')
        files[path] = digest
    for path in (base.TEACHER, base.VOCODER_DIR / 'generator_step_2500000.pt', ROOT / 'data/stats/mel_mean.npy', ROOT / 'data/stats/mel_std.npy'):
        files[str(path)] = parent['file_sha256'][str(path)]
    plan = parent['monitor64_plan']
    if plan != cross.read_json(previous.CROSS_IDENTITY)['plan']: raise RuntimeError('Fixed64 plan changed')
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387: raise RuntimeError('Dataset indexing changed')
    future = d.future_plan(development)[:2]
    actual_features = previous.register_future_files(dataset, future, files, check)
    monitor_features = previous.register_future_files(dataset, [{'indices': plan['selected_indices']}], files, check)
    if len(monitor_features) != 192 or any(parent['file_sha256'].get(p) != sha for p, sha in monitor_features.items()):
        raise RuntimeError('Original monitor features changed')
    d.check_files(files, check)
    return mc, parent, receipt, config, dataset, future, actual_features, monitor_features, files


def contract_for(args, context, started):
    _, parent, _, config, _, future, feature_files, monitor_files, files = context
    return {'kind': 'two_fresh_processes_one_MC2_step_each_then_readonly3929', 'supervisor_pid': os.getpid(),
        'shared_monotonic_start': started, 'configuration': config.as_dict(), 'effective_optimizer_lrs': d.PARENT_LRS,
        'parent_checkpoint': str(d.PARENT), 'parent_checkpoint_sha256': d.PARENT_SHA, 'parent_contract_sha256': d.PARENT_CANONICAL,
        'runtime_files': files, 'actual_two_batches': future, 'two_batch_feature_sha256': feature_files,
        'fixed64_feature_sha256': monitor_files, 'input_identities': parent['input_identities'],
        'child_names': list(CHILDREN), 'child_limits': CHILD_LIMITS, 'total_limits': TOTAL_LIMITS,
        'maximum_complete_updates': 2, 'thread_environment': THREAD_ENV, 'original_flags': cross.ORIGINAL_FLAGS,
        'protocol_sha256': args.protocol_sha256, 'compute_seconds_all_children': SECONDS, 'cleanup_only_seconds': GRACE,
        'metadata_bytes_all_children': META_LIMIT, 'CPU_copy_synchronization_is_an_observation_perturbation': True,
        'finite_AA_difference_is_a_result_not_failure': True, 'no_checkpoint_metric_or_training_continuation': True}


def inherited_locks(args, contract):
    if os.getppid() != args.supervisor_pid or contract['supervisor_pid'] != args.supervisor_pid:
        raise RuntimeError('Child is not attached to the declared live supervisor')
    if contract['shared_monotonic_start'] != args.shared_start: raise RuntimeError('Child may not reset the600-second clock')
    fds = [int(value) for value in args.lock_fds.split(',')]
    if len(fds) != 4 or len(set(fds)) != 4 or any(fd < 3 for fd in fds): raise RuntimeError('Require four distinct inherited lockFDs')
    for fd, path in zip(fds, LOCK_PATHS):
        cross.regular(path)
        actual, expected = os.fstat(fd), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino): raise RuntimeError('Inherited lock path differs')
        if Path('/proc/%d/fd/%d' % (args.supervisor_pid, fd)).resolve() != path: raise RuntimeError('Supervisor lock identity differs')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fds


@torch.no_grad()
def probe3929(mc, student, teacher, diffusion, batch, config, device):
    tensors = {key: batch[key].to(device) for key in ('mel', 'bnf', 'spk_emb')}
    identity = mc.original_input_identity(batch, device, config)
    source_noise, score_noise = torch.randn_like(tensors['mel']), torch.randn_like(tensors['mel'])
    timesteps = torch.randint(1, 1001, (32,), device=device, dtype=torch.long)
    receipt = {}; mc.record_first_draws(source_noise, score_noise, timesteps, receipt)
    for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
        if receipt[key] != identity[key]: raise RuntimeError('Readonly3929 input/noise identity mismatch')
    prediction, q = training.make_student_prediction(student, diffusion, tensors['mel'], tensors['spk_emb'], tensors['bnf'], 950, source_noise)
    loss, targets = mc.mc2_score_distillation_loss(prediction, teacher, diffusion, timesteps, score_noise,
        tensors['spk_emb'], tensors['bnf'], target_mode=config.score_target, weight_mode=config.score_weight,
        global_step=3929, mc_receipt=receipt)
    result = {'global_step': 3929, 'input_identity': identity, 'prediction': tensor_summary(prediction),
        'q950': tensor_summary(q), 'teacher_targets': [tensor_summary(t) for t in targets], 'dist_raw': float(loss), 'mc': receipt,
        'actual_forward_counts': PROBE_COUNTS, 'mc_training_forward_budget_is_only_frozen_template': True}
    if not all(v['finite'] for v in [result['prediction'], result['q950']] + result['teacher_targets']): raise RuntimeError('Nonfinite readonly prediction')
    return result


def child_main(args):
    clock = old.StopClock(SECONDS, args.shared_start); budget = Budget(OUTPUT); ledger = Ledger()
    folder = OUTPUT / args.child; log = None; inherited = []; result = None; state = {}; events = {}; owned = False
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        clock.install(); clock.active(); cross.regular(OUTPUT, directory=True)
        contract_path = OUTPUT / 'run_contract.json'
        if base.sha256_file(contract_path) != args.contract_file_sha256: raise RuntimeError('Supervisor contractFILE SHA changed')
        contract = cross.read_json(contract_path); inherited = inherited_locks(args, contract)
        log = d.StageLog(folder, budget); folder.mkdir(); owned = True; sys.stdout = sys.stderr = log
        def check():
            clock.active(); budget.require()
            if os.getppid() != args.supervisor_pid: raise RuntimeError('Supervisor disappeared')
        ledger.check = check
        environment = runtime(); context = prepare_context(args, check)
        mc, parent, receipt, config, dataset, future, feature_files, monitor_files, files = context
        if contract_for(args, context, args.shared_start) != dict(contract, supervisor_pid=os.getpid()):
            raise RuntimeError('Child actual inputs/config differ from the shared frozen contract')
        gpu = cross.gpu_gate(); device = torch.device('cuda'); check()
        diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
        inputs, batches, target, input_identity = d.prepare_inputs(parent, dataset, diffusion, device)
        payload = resume.load_torch_checkpoint(d.PARENT)
        teacher, student, unused = base.load_teacher_and_student(base.TEACHER, config, device); del unused
        vocoder, vconfig, unused = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
            base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device); del unused
        discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed_d = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed = resume.load_torch_checkpoint(pair.PARENT / 'latest_full.pt')
        if fixed['global_step'] != 3107 or fixed['run_contract_sha256'] != pair.PARENT_CONTRACT: raise RuntimeError('Wrong fixedD3107')
        fixed_d.load_state_dict(fixed['discriminator'], strict=True); del fixed
        fixed_d.eval(); training.set_requires_grad(fixed_d, False)
        mean, std = base.load_mel_statistics(ROOT / 'data', device); d.imported_source_files(files, files)
        frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
        frozen_sha = {name: pair.model_sha(model) for name, model in frozen.items()}
        if frozen_sha != parent['fixed_frozen_models_sha256']: raise RuntimeError('Frozen model identity changed')
        models = dict(frozen, student=student, active_D=discriminator, diffusion=diffusion)
        optimizer = torch.optim.Adam(student.parameters(), lr=5e-5, betas=(.5, .9))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .9))
        restored = d.restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, parent, receipt, device, d.ARMS[0]); del payload
        if restored['parent_restore'] != restored['role_lr_applied']: raise RuntimeError('Single-step diagnosis may not change the restored state')
        state = d.new_state(d.ARMS[0], training.canonical_json_sha256(contract)); state['restore_completed'] = True
        entering = previous.content_state(student, discriminator, optimizer, d_optimizer)
        budget.write(folder / 'entering_state.json', {'state': entering, 'runtime': environment, 'gpu': gpu})
        ledger.install(student, teacher, vocoder, discriminator, fixed_d)
        def unchanged():
            pair.assert_frozen(frozen, frozen_sha)
            if input_identity() != parent['input_identities'] or cross.flags() != cross.ORIGINAL_FLAGS: raise RuntimeError('Fixed inputs or flags changed')
        check(); before = dict(ledger.counts)
        with previous.monitoring_state(models, device):
            initial = pair.monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, 3927)
            initial['updates'] = initial['trial_updates'] = 0
            initial['uniform64'] = previous.uniform64_monitor(student, teacher, diffusion, batches, parent['monitor64_plan'], config, device, check)
        if ledger.delta(before) != MONITOR_COUNTS: raise RuntimeError('Wrong initial-monitor calls')
        initial['actual_forward_counts'] = ledger.delta(before); unchanged()
        # 历史输出微差和A/A差异均报告；输入、模型恢复、计数身份仍严格要求。
        parent_delta = {k: differences(initial[k], receipt['last_monitor'][k]) for k in ('fixed_train', 'development', 'uniform64')}
        budget.write(folder / 'initial_monitor.json', {'actual': initial, 'historical_parent_differences': parent_delta})
        planned = future[0]; batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
        training.seed_everything(planned['noise_seed']); identity = mc.original_input_identity(batch, device, config); mc_receipt = {}
        check(); before = dict(ledger.counts)
        def update_once():
            # 包装安装在真实返回观察之内，摘要失败也不能隐瞒已经返回的Adam.step。
            with step_observers(student, discriminator, optimizer, d_optimizer, events,
                    lambda value: budget.write(folder / 'optimizer_observations.jsonl', value, append=True), check):
                return mc.train_batch_mc2(batch, teacher, student, vocoder, discriminator, diffusion, optimizer, d_optimizer,
                    mean, std, device, config, global_step=3928, mc_receipt=mc_receipt)
        losses = d.complete_update(state, 3928, optimizer, d_optimizer, update_once)
        if ledger.delta(before) != TRAIN_COUNTS or state['completed_global_step'] != 3928: raise RuntimeError('Wrong actual single update')
        d.validate_adams(student, discriminator, optimizer, d_optimizer, 3928, d.PARENT_LRS)
        for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
            if mc_receipt[key] != identity[key]: raise RuntimeError('3928 original inputs/noise changed')
        budget.write(folder / 'step3928.json', {'losses': losses, 'input_identity': identity, 'mc': mc_receipt, 'state': state})
        check(); before = dict(ledger.counts)
        planned = future[1]; batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
        before_probe = previous.content_state(student, discriminator, optimizer, d_optimizer)
        with previous.monitoring_state(models, device), torch.no_grad():
            training.seed_everything(planned['noise_seed']); readout = probe3929(mc, student, teacher, diffusion, batch, config, device)
        if ledger.delta(before) != PROBE_COUNTS or previous.content_state(student, discriminator, optimizer, d_optimizer) != before_probe:
            raise RuntimeError('Readonly3929 changed model/Adam/RNG or call counts')
        if ledger.counts != CHILD_LIMITS: raise RuntimeError('Actual child total differs from25T7S5V32D16fixedD')
        budget.write(folder / 'probe3929.json', readout); unchanged(); d.imported_source_files(files, files); d.check_files(files, check)
        if runtime() != environment: raise RuntimeError('Diagnostic runtime or threading changed during this child')
        if base.sha256_file(contract_path) != args.contract_file_sha256: raise RuntimeError('Supervisor contract changed')
        result = {'status': 'one_update_and_readonly_probe_completed', 'child': args.child, 'pid': os.getpid(),
            'runtime': environment, 'contract_file_sha256': args.contract_file_sha256, 'entering_state': entering,
            'initial_monitor': initial, 'historical_initial_differences': parent_delta, 'step3928_losses': losses,
            'step3928_receipt': mc_receipt, 'step3928_input_identity': identity, 'optimizer_observations': events,
            'probe3929': readout, 'actual_state': state, 'actual_forward_counts': dict(ledger.counts),
            'CPU_copy_synchronization_disclosed': True, 'no_checkpoint_created': True}
        budget.write(folder / 'results.json', result); check()
    except BaseException as error:
        clock.request('technical_failure'); budget.aborting = True
        if owned:
            try: budget.write(folder / 'failure.json', {'error': repr(error), 'state': state, 'actual_forward_counts': ledger.counts,
                'observed_actual_step_returns': {k: v for k, v in events.items() if k.endswith('_actual_step_returned')}})
            except BaseException as failed: saved_stderr.write('Child failure receipt unavailable: %r; original=%r\n' % (failed, error))
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = old.cleanup_resources(ledger, clock, log, [])
        for fd in reversed(inherited):
            try: os.close(fd)  # 只关本子进程的副本，不LOCK_UN父进程持有的锁。
            except BaseException as error: errors.append({'resource': 'inherited_lock_fd', 'error': repr(error)})
        if errors and owned:
            budget.aborting = True
            budget.write(folder / 'cleanup_failure.json', {'errors': errors})
        if errors and original_error is None: raise RuntimeError('Child cleanup failed: ' + repr(errors))
        if original_error is None and not errors and result is not None:
            try:
                clock.active(); budget.write(folder / 'completion.json', {'status': result['status'], 'cleanup_verified': True,
                    'results_sha256': base.sha256_file(folder / 'results.json'), 'actual_forward_counts': ledger.counts,
                    'complete_updates': 1, 'completed_global_step': 3928, 'probe_global_step': 3929}, terminal=True)
            except BaseException as error:
                budget.aborting = True
                budget.write(folder / 'failure.json', {'error': repr(error), 'phase': 'terminal_publication', 'state': state})
                raise


def wait_child(process, clock):
    while True:
        clock.active()
        try: return process.communicate(timeout=min(1., max(.001, clock.started + SECONDS - time.monotonic())))[0]
        except subprocess.TimeoutExpired: pass


def stop_child(process, clock):
    if process is None or process.poll() is not None: return
    process.terminate()
    remaining = max(.001, clock.started + SECONDS + GRACE - time.monotonic() - 5)
    try: process.communicate(timeout=min(85., remaining))
    except subprocess.TimeoutExpired: process.kill(); process.communicate(timeout=5.)
    if process.poll() is None: raise RuntimeError('Child process did not exit')


def supervisor_main(args):
    started = time.monotonic(); clock = old.StopClock(SECONDS, started); budget = Budget(OUTPUT)
    locks = []; log = process = None; result = None; owned = False; outcomes = []
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        clock.install(); previous.watchdog(SECONDS); training.assert_rented_server(ROOT)
        runtime()
        if training.git_head(ROOT) != cross.HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Require clean frozen production source')
        locks = training.acquire_runtime_locks(ROOT)
        if len(locks) != 3: raise RuntimeError('Require three original workflow locks')
        for path in LOCK_PATHS:
            if path.is_symlink(): raise RuntimeError('Lock symlink')
        handle = LOCK_PATHS[-1].open('a+')
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException: handle.close(); raise
        locks.append(handle)
        if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('Diagnostic root must be new; never retry in place')
        disk = budget.require(); OUTPUT.mkdir(); owned = True
        log = d.StageLog(OUTPUT / 'supervisor', budget); sys.stdout = sys.stderr = log
        def check(): clock.active(); budget.require()
        context = prepare_context(args, check); contract = contract_for(args, context, started)
        budget.write(OUTPUT / 'run_contract.json', contract)
        contract_file_sha = base.sha256_file(OUTPUT / 'run_contract.json'); files = context[-1]
        del context
        children = []
        for name in CHILDREN:
            check(); cross.gpu_gate()
            command = [sys.executable, str(Path(__file__).resolve()), '--run-approved', '--driver-sha256', args.driver_sha256,
                '--tests-sha256', args.tests_sha256, '--protocol-sha256', args.protocol_sha256, '--child', name,
                '--supervisor-pid', str(os.getpid()), '--shared-start', repr(started),
                '--contract-file-sha256', contract_file_sha, '--lock-fds', ','.join(str(h.fileno()) for h in locks)]
            environment = dict(os.environ, **THREAD_ENV, PYTHONDONTWRITEBYTECODE='1')
            # 不建立新process group：外部timeout的最终信号也覆盖两个子进程。
            process = subprocess.Popen(command, cwd=ROOT, env=environment, pass_fds=tuple(h.fileno() for h in locks),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            child_output = wait_child(process, clock)
            outcomes.append({'child': name, 'pid': process.pid, 'returncode': process.returncode})
            budget.write(OUTPUT / 'child_exits.jsonl', outcomes[-1], append=True)
            if child_output: log.write(child_output)
            if process.returncode != 0: raise RuntimeError('Child failed; do not start/retry another child: ' + name)
            process = None
            folder = OUTPUT / name; done = cross.read_json(folder / 'completion.json')
            if (done['cleanup_verified'], done['complete_updates'], done['completed_global_step'], done['probe_global_step'], done['actual_forward_counts']) != (True, 1, 3928, 3929, CHILD_LIMITS):
                raise RuntimeError('Child terminal identity/cursor/counts invalid')
            if any((folder / n).exists() for n in ('failure.json', 'cleanup_failure.json')): raise RuntimeError('Child has failure evidence')
            if base.sha256_file(folder / 'results.json') != done['results_sha256']: raise RuntimeError('Child results changed')
            children.append(cross.read_json(folder / 'results.json'))
        keys = ('entering_state', 'initial_monitor', 'step3928_losses', 'step3928_receipt', 'step3928_input_identity',
                'optimizer_observations', 'probe3929', 'runtime')
        comparisons = {key: differences(children[1][key], children[0][key]) for key in keys}
        counts = {key: sum(child['actual_forward_counts'][key] for child in children) for key in TOTAL_LIMITS}
        if counts != TOTAL_LIMITS: raise RuntimeError('Combined calls exceed the two-child plan')
        d.check_files(files, check)
        if base.sha256_file(OUTPUT / 'run_contract.json') != contract_file_sha: raise RuntimeError('Joint diagnostic contract changed')
        result = {'status': 'two_independent_one_step_diagnostics_completed', 'actual_forward_counts': counts,
            'complete_updates': 2, 'comparisons': comparisons, 'all_compared_values_identical': not any(comparisons.values()),
            'finite_mismatch_is_not_a_failure_or_performance_gate': True, 'initial_disk': disk, 'child_exits': outcomes,
            'no_checkpoint_score_or_automatic_training': True, 'elapsed_seconds': time.monotonic() - started}
        budget.write(OUTPUT / 'results.json', result); check()
    except BaseException as error:
        clock.request('technical_failure'); budget.aborting = True
        process_cleanup_error = None
        try: stop_child(process, clock)
        except BaseException as stopped: process_cleanup_error = repr(stopped)
        if owned:
            try: budget.write(OUTPUT / 'failure.json', {'error': repr(error), 'child_exits': outcomes, 'process_cleanup_error': process_cleanup_error})
            except BaseException as failed: saved_stderr.write('Supervisor failure receipt unavailable: %r; original=%r\n' % (failed, error))
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = []
        try: stop_child(process, clock)
        except BaseException as error: errors.append({'resource': 'child_process', 'error': repr(error)})
        errors += old.cleanup_resources(Ledger(), clock, log, locks)
        if errors and owned:
            budget.aborting = True; budget.write(OUTPUT / 'cleanup_failure.json', {'errors': errors})
        if errors and original_error is None: raise RuntimeError('Supervisor cleanup failed: ' + repr(errors))
        if original_error is None and not errors and result is not None:
            try:
                clock.active(); budget.write(OUTPUT / 'completion.json', {'status': result['status'], 'cleanup_verified': True,
                    'results_sha256': base.sha256_file(OUTPUT / 'results.json'), 'actual_forward_counts': result['actual_forward_counts'],
                    'complete_updates': 2, 'all_compared_values_identical': result['all_compared_values_identical']}, terminal=True)
            except BaseException as error:
                budget.aborting = True
                budget.write(OUTPUT / 'failure.json', {'error': repr(error), 'phase': 'terminal_publication', 'child_exits': outcomes})
                raise


def main(argv=None):
    args = arguments(argv)
    if args.child: child_main(args)
    else: supervisor_main(args)


if __name__ == '__main__': main()
