"""从用户选定3327完整断点长期继续原MC1；保存单个rolling full，不做候选评分。"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
OUTPUT = DEV / 'acdd_mc1_longtrain_from3327_v1'
PARENT_RUN = DEV / 'acdd_s5e5_staged120_from3227_v1'
PARENT = PARENT_RUN / 'final_full.pt'
PARENT_SHA = '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556'
PARENT_CANONICAL = '32651a26cdf7074a9e8ca30c3cdd099697541fa4b261df757d2866f9e0622aba'
PARENT_CONTRACT_SHA = '787759339c886b2918bdda62abcc6818adf3cb410f923aca38c80bbd4b99b071'
PARENT_COMPLETION_SHA = '68ee1cf0e680f3bf665be6dcf46e2aa75c3684262c2da3765ae2c7beebf931b2'
RESTORE_FILE = TOOLS / 'inspect_acdd_d_update_3327_v1.py'
RESTORE_SHA = 'c14d0997173bb734ae9d461d95d1637d2b7503725bbcc474febdb04e12c2cd49'
RESUME_FILE = TOOLS / 'resume_acdd_development.py'
RESUME_SHA = '7b81570909cdf39cfa16c563c7c302e7c3886e5551060917ad61de7e8baf0014'
VOCODER_METADATA_PINS = {
    str(ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/config.json'): '4b570fee140ffbda92bb217218792dca069b2d73627bc5e7bf7b1669869dd05a',
    str(ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/final_checkpoint_attestation_comments_20260906.json'): '2112690afbe6fc2f7a5325d106c8dcfdd4503d7cfb09fee7675712d34b46205e',
}  # 原声码器的已审元数据；3327父清单未重复收录这两项，需显式绑定。
HEAD = '0a2c646922704e746642f5f33c203869fe696b3f'
START, STOP, SAVE_EVERY, STATUS_EVERY = 3327, 122900, 1000, 20
LRS = {'student': .00005, 'discriminator': .0002}
LOSS_KEYS = ('student_total', 'student_adv', 'fm_raw', 'fm_weighted', 'dist_raw', 'dist_weighted', 'discriminator')
KIND = 'acdd_mc1_longtrain_from3327_full_v1'
FULL_UPPER, RESERVE, LOG_LIMIT, DIR_RESERVE = 1460000000, 512 * 1024**2, 128 * 1024**2, 1024**2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''): h.update(block)
    return h.hexdigest()


if os.name != 'posix' or not (ROOT / 'AGENTS.md').is_file(): raise RuntimeError('Only the rented Matpool server may execute this entry')
for path, expected in ((RESTORE_FILE, RESTORE_SHA), (RESUME_FILE, RESUME_SHA)):
    if digest(path) != expected: raise RuntimeError('Frozen recovery source changed: ' + str(path))
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import inspect_acdd_d_update_3327_v1 as recovery
import resume_acdd_development as resume
baseline, training, probe = resume.baseline, resume.baseline.training, recovery.probe


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    # 只允许本次输出目录中的状态/收据；JSON原子替换不触碰父模型。
    training.atomic_write_json(Path(path), value)


def append(path, value):
    with Path(path).open('a', encoding='utf8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        handle.flush()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    parser.add_argument('--driver-sha256', required=True); parser.add_argument('--tests-sha256', required=True)
    args = parser.parse_args(argv)
    if not args.run_approved: raise RuntimeError('Require the user-selected3327 long-training authorization')
    for name in ('driver_sha256', 'tests_sha256'):
        value = getattr(args, name)
        if len(value) != 64 or any(x not in '0123456789abcdef' for x in value): raise RuntimeError('Full reviewed source SHA required')
    return args


def regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise RuntimeError('Expected a regular nonsymlink file: ' + str(path))


def verify(files):
    for name, expected in files.items():
        regular(name)
        if digest(name) != expected: raise RuntimeError('File identity changed: ' + name)


def source_files(parent, explicit):
    # 仅绑定实际导入源码，不继承父清单中的旧full、旧固定D或旧评分文件。
    sources = dict(explicit)
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name: continue
        path = Path(name).resolve()
        if path.suffix != '.py' or ROOT not in path.parents: continue
        expected = sources.get(str(path), parent['file_sha256'].get(str(path)))
        if expected is None:
            # 已跟踪源码可直接对照当前冻结HEAD字节；未跟踪TOOLS必须已有明确pin。
            try: committed = subprocess.check_output(['git', 'show', HEAD + ':' + str(path.relative_to(ROOT))], cwd=ROOT, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError: raise RuntimeError('Unpinned imported source: ' + str(path))
            expected = hashlib.sha256(committed).hexdigest()
        sources[str(path)] = expected
    verify(sources); return sources


def metadata_bytes(output):
    return sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file() and not p.name.endswith(('.pt', '.pt.tmp')))


def require_space(output, saving=False):
    output = Path(output); used = metadata_bytes(output) if output.exists() else 0
    if used > LOG_LIMIT: raise RuntimeError('128MiB long-run log allocation exhausted; preserve all logs')
    latest = output / 'latest_full.pt'; temporary = output / 'latest_full.pt.tmp'
    for path in (latest, temporary):
        if path.is_symlink(): raise RuntimeError('Rolling checkpoint path must not be a symlink')
    allocated = max(latest.stat().st_size, latest.stat().st_blocks * 512) if latest.exists() else 0
    # 初始留两份新full；之后旧latest已占用磁盘，只需补足新.tmp和剩余日志/系统余量。
    needed = RESERVE + DIR_RESERVE + LOG_LIMIT - used + FULL_UPPER + max(0, FULL_UPPER - allocated)
    # 上次中断留下的本run临时文件在下次torch.save时会重写；它已占用的块可计入写入预留。
    if temporary.exists(): needed -= min(FULL_UPPER, temporary.stat().st_blocks * 512)
    stats = os.statvfs(output.parent); free = stats.f_bavail * stats.f_frsize
    if free < needed: raise RuntimeError('Insufficient rolling storage: free=%d required=%d' % (free, needed))
    return {'available_bytes': free, 'required_bytes': needed, 'metadata_bytes': used}


def optimizer_cursor(optimizer):
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    if not parameters or len({id(p) for p in parameters}) != len(parameters): raise RuntimeError('Invalid optimizer ownership')
    values = [float(optimizer.state[p]['step']) for p in parameters]
    if any(not math.isfinite(x) or x != int(x) for x in values) or len(set(values)) != 1: raise RuntimeError('Nonuniform actual Adam cursor')
    return {'parameter_count': len(values), 'uniform_step': int(values[0])}


def complete_batch(state, step, optimizer, d_optimizer, call):
    if step != state['completed_global_step'] + 1: raise RuntimeError('Require the next absolute step')
    for opt in (optimizer, d_optimizer):
        if optimizer_cursor(opt)['uniform_step'] != step - 1: raise RuntimeError('Entering Adam cursor differs')
    state.update(attempted_global_step=step, partial_batch=True, adam_returns={'student': 0, 'discriminator': 0})
    installed = []; missing = object()
    try:
        for role, opt in (('discriminator', d_optimizer), ('student', optimizer)):
            original, local = opt.step, opt.__dict__.get('step', missing)
            def observed(*args, _role=role, _original=original, **kwargs):
                if state['adam_returns'][_role] or (_role == 'student' and state['adam_returns']['discriminator'] != 1): raise RuntimeError('Require exactly D then S')
                result = _original(*args, **kwargs); state['adam_returns'][_role] += 1; return result
            opt.step = observed; installed.append((opt, local))
        losses = call()
    finally:
        for opt, local in installed:
            if local is missing: del opt.step
            else: opt.step = local
        try:
            cursors = {'student': optimizer_cursor(optimizer), 'discriminator': optimizer_cursor(d_optimizer)}
            state['optimizer_cursors'] = cursors
            if state['adam_returns'] == {'student': 1, 'discriminator': 1} and all(v['uniform_step'] == step for v in cursors.values()):
                state.update(completed_global_step=step, partial_batch=False)
        except BaseException as error:
            state['cursor_inspection_error'] = repr(error)  # 不用检查游标的错误覆盖原训练异常。
    if state['partial_batch']: raise RuntimeError('Incomplete D/S update')
    if set(losses) != set(LOSS_KEYS) or any(not math.isfinite(float(v)) for v in losses.values()): raise RuntimeError('Invalid original MC1 loss fields')
    return losses


def validate_resume(payload, contract):
    canonical = training.canonical_json_sha256(contract)
    if (payload.get('kind'), payload.get('run_contract_sha256'), payload.get('effective_optimizer_lrs'), payload.get('mc_samples')) != (KIND, canonical, LRS, 1):
        raise RuntimeError('Rolling full identity/configuration differs')
    step = payload['global_step']
    if type(step) is not int or not START < step <= STOP or payload.get('partial_batch') is not False or payload.get('resumable') is not True: raise RuntimeError('Invalid complete rolling cursor')
    if payload['configuration'] != contract['configuration'] or payload['score_semantics'] != 'reconstruction_clean_x0_sqrt_alpha_bar': raise RuntimeError('Original MC1 objective changed')
    if payload['origin_parent_sha256'] != PARENT_SHA: raise RuntimeError('Wrong original3327 parent')
    return step


def choose_checkpoint(output, contract):
    path = Path(output) / 'latest_full.pt'; pointer = Path(output) / 'latest_full.json'
    if not path.exists():
        if pointer.exists(): raise RuntimeError('Committed rolling full is missing; do not silently return to3327')
        regular(PARENT); return PARENT, PARENT_SHA, None
    regular(path); sha = digest(path); payload = resume.load_torch_checkpoint(path); step = validate_resume(payload, contract)
    if pointer.exists():
        record = read(pointer)
        if record['global_step'] > step or (record['global_step'] == step and record['checkpoint_sha256'] != sha): raise RuntimeError('Rolling pointer/full conflict')
    # 原子替换已成功、指针尚未来得及更新的情况：以通过完整恢复验证的full自身为准。
    return path, sha, payload


def restore(payload, student, discriminator, optimizer, d_optimizer, config, contract, parent, receipt, device, is_parent):
    if is_parent: recovery.validate_parent(payload, student, discriminator, config, parent, receipt); step = START
    else: step = validate_resume(payload, contract)
    for role, model, opt in (('student', student, optimizer), ('discriminator', discriminator, d_optimizer)):
        recovery.old.validate_saved_adam(payload[role + '_optimizer'], model, step, LRS[role])
        model.load_state_dict(payload[role], strict=True); recovery.restore_optimizer(opt, payload[role + '_optimizer'])
        model.train(); training.set_requires_grad(model, True)
        for parameter in model.parameters(): parameter.grad = None
        if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()): raise RuntimeError('Nonfinite saved model')
        if {id(p) for g in opt.param_groups for p in g['params']} != {id(p) for p in model.parameters()}: raise RuntimeError('Wrong Adam ownership')
    training.restore_rng_state(payload['rng_state'], device)
    expected_rng = payload['rng_state'] if device.type == 'cuda' else dict(payload['rng_state'], cuda_all=[])
    if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(expected_rng): raise RuntimeError('Full RNG restoration differs')
    return step


def durable_torch_save(payload, path):
    # 新数据先落盘，再替换唯一latest，最后把目录中的改名也落盘。
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(dict(payload), handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def release_runtime(handlers, locks):
    # 一个清理动作失败也继续释放其余锁；完成收据只能在全部清理成功后发布。
    errors = []
    for signum, handler in handlers.items():
        try: signal.signal(signum, handler)
        except BaseException as error: errors.append('signal: ' + repr(error))
    for handle in reversed(locks):
        try: handle.close()
        except BaseException as error: errors.append('lock: ' + repr(error))
    return errors


def save_latest(output, state, student, discriminator, optimizer, d_optimizer, config, contract, scores):
    if state['partial_batch'] or state.get('execution_error'): raise RuntimeError('Never save partial or technically failed state')
    step = state['completed_global_step']
    if not START < step <= STOP: raise RuntimeError('No new complete update to save')
    for role, model, opt in (('student', student, optimizer), ('discriminator', discriminator, d_optimizer)):
        recovery.old.validate_saved_adam(opt.state_dict(), model, step, LRS[role])
        if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()): raise RuntimeError('Nonfinite model')
    payload = {'kind': KIND, 'epoch': 0, 'global_step': step, 'completed_passes': step // 1229,
        'partial_batch': False, 'resumable': True, 'origin_parent_sha256': PARENT_SHA,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS, 'mc_samples': 1,
        'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'run_contract_sha256': training.canonical_json_sha256(contract),
        'student': student.state_dict(), 'discriminator': discriminator.state_dict(), 'student_optimizer': optimizer.state_dict(),
        'discriminator_optimizer': d_optimizer.state_dict(), 'rng_state': training.capture_rng_state(),
        'development': scores, 'last_development_step': state.get('last_development_step'), 'actual_optimizer_cursors': state['optimizer_cursors']}
    path = Path(output) / 'latest_full.pt'; require_space(output, saving=True)
    # 同目录临时文件完整写出后原子替换；旧latest一直有效至替换瞬间。
    durable_torch_save(payload, path)
    state['last_saved_global_step'] = step; state['checkpoint_committed_before_pointer'] = True
    if max(path.stat().st_size, path.stat().st_blocks * 512) > FULL_UPPER: raise RuntimeError('Committed full exceeds declared capacity; preserve it')
    record = {'path': str(path), 'global_step': step, 'checkpoint_sha256': digest(path), 'run_contract_sha256': payload['run_contract_sha256']}
    write(Path(output) / 'latest_full.json', record); append(Path(output) / 'checkpoint_history.jsonl', record)
    state['checkpoint_committed_before_pointer'] = False; return record


def environment():
    flags = {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
             'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic}
    if flags != {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True, 'cudnn_benchmark': False, 'cudnn_deterministic': False}: raise RuntimeError('Original precision flags changed')
    if (platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6'): raise RuntimeError('Wrong original training environment')
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'flags': flags,
        'executable': str(Path(sys.executable).resolve()), 'torch_threads': torch.get_num_threads()}


def prepare(args):
    fixed = {str(PARENT): PARENT_SHA, str(PARENT_RUN / 'run_contract.json'): PARENT_CONTRACT_SHA,
        str(PARENT_RUN / 'completion.json'): PARENT_COMPLETION_SHA, str(RESTORE_FILE): RESTORE_SHA, str(RESUME_FILE): RESUME_SHA,
        str(Path(__file__).resolve()): args.driver_sha256, str(TOOLS / 'test_run_acdd_mc1_longtrain_from3327_v1.py'): args.tests_sha256,
        str(TOOLS / 'inspect_acdd_adam_3227_v1.py'): recovery.HELPER_SHA,
        str(TOOLS / 'inspect_acdd_teacher_target_cross_v1.py'): recovery.old.CROSS_SHA}
    verify(fixed); parent = read(PARENT_RUN / 'run_contract.json'); receipt = read(PARENT_RUN / 'completion.json')
    if training.canonical_json_sha256(parent) != PARENT_CANONICAL: raise RuntimeError('Original3327 contract changed')
    config = replace(baseline.PAPER_STUDENT_CONFIG, num_workers=0)
    if parent['configuration'] != config.as_dict(): raise RuntimeError('Original cleanx0/sqrt/FM2/dist45 configuration changed')
    development, split_files = resume.load_development(DEV / 'development_contract.json'); fixed.update(split_files)
    if (parent['training_indices'], parent['excluded_train_indices']) != (development['retained_train_indices'], development['excluded_train_indices']): raise RuntimeError('52 exclusions changed')
    planner = resume.GlobalBatchPlan(parent['training_indices'])
    if planner.batches_per_pass != 1229 or STOP != 100 * planner.batches_per_pass: raise RuntimeError('Expected100 passes with1229 full batches each')
    for asset in (baseline.TEACHER, baseline.VOCODER_DIR / 'generator_step_2500000.pt', baseline.VOCODER_DIR / 'config.json',
                  baseline.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', ROOT / 'data/stats/mel_mean.npy', ROOT / 'data/stats/mel_std.npy'):
        fixed[str(asset)] = VOCODER_METADATA_PINS[str(asset)] if str(asset) in VOCODER_METADATA_PINS else parent['file_sha256'][str(asset)]
    fixed = source_files(parent, fixed)
    contract = {'kind': 'user_selected3327_MC1_longtrain_v1', 'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS,
        'origin_parent': str(PARENT), 'origin_parent_sha256': PARENT_SHA, 'origin_parent_canonical': PARENT_CANONICAL,
        'start_after_global_step': START, 'stop_global_step': STOP, 'total_passes': 100, 'batches_per_pass': 1229,
        'training_indices': parent['training_indices'], 'excluded_train_indices': parent['excluded_train_indices'],
        'mc_samples': 1, 'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'runtime_files': fixed,
        'batch_plan': {'shuffle_seed': '1234+floor((step-1)/1229)', 'crop_seed': '101234+step', 'noise_seed': '201234+step'},
        'rolling_save_every': SAVE_EVERY, 'status_every': STATUS_EVERY, 'loss_keys': list(LOSS_KEYS),
        'loss_and_DEV_are_observations_not_stop_gates': True, 'no_candidate_metric_gate': True, 'no_MC2_or_auxiliary_objective': True,
        'storage': {'full_upper': FULL_UPPER, 'reserve': RESERVE, 'log_limit': LOG_LIMIT, 'directory_reserve': DIR_RESERVE},
        'authorization': 'User explicitly selected model12/global3327 for direct long training; no renewed candidate gate',
        'original_parent_remains_readonly': True, 'runtime': environment(), 'no_automatic_scoring': True}
    return parent, receipt, config, development, planner, contract


def main(argv=None):
    args = arguments(argv); locks = []; handlers = {}; state = {}; attempt = None
    stop = {'requested': False}; started = time.monotonic()
    try:
        training.assert_rented_server(ROOT)
        torch.set_num_threads(2)  # 恢复原3327实际入口的CPU线程设置，不使用另一支实验的默认1。
        locks = training.acquire_runtime_locks(ROOT)
        import fcntl
        extra_path = ROOT / 'exp/locks/text_unseen_fastvoicegrad_student_inference.lock'
        if extra_path.is_symlink(): raise RuntimeError('Inference lock cannot be a symlink')
        extra = extra_path.open('a+')
        try: fcntl.flock(extra.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException: extra.close(); raise
        locks.append(extra)
        if training.git_head(ROOT) != HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip(): raise RuntimeError('Frozen production source must remain clean')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(): raise RuntimeError('GPU currently used; do not start another workload')
        if OUTPUT.is_symlink(): raise RuntimeError('Output cannot be a symlink')
        parent, receipt, config, development, planner, proposed = prepare(args)
        if OUTPUT.exists():
            contract = read(OUTPUT / 'run_contract.json')
            if contract != proposed: raise RuntimeError('Restart must retain the same source, settings and run contract')
        else:
            require_space(OUTPUT); OUTPUT.mkdir(); contract = proposed; write(OUTPUT / 'run_contract.json', contract)
        source, source_sha, payload = choose_checkpoint(OUTPUT, contract)
        if payload is None: payload = resume.load_torch_checkpoint(source)
        # 每次进程一个日志子目录；重启后重做上次未保存的步不会混入同一条loss曲线。
        number = 1
        while (OUTPUT / ('attempt_%04d' % number)).exists(): number += 1
        attempt = OUTPUT / ('attempt_%04d' % number); attempt.mkdir()
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: stop.update(requested=True))
        device = torch.device('cuda'); dataset = baseline.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
        if len(dataset) != 39387: raise RuntimeError('Dataset indexing changed')
        teacher, student, unused = baseline.load_teacher_and_student(baseline.TEACHER, config, device); del unused
        vocoder, vconfig, unused = baseline.load_frozen_vocoder(ROOT, baseline.VOCODER_DIR / 'generator_step_2500000.pt', baseline.VOCODER_DIR / 'config.json',
            baseline.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device); del unused
        discriminator = baseline.build_paper_discriminator(ROOT, vconfig, config, device)
        diffusion = baseline.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device); mean, std = baseline.load_mel_statistics(ROOT / 'data', device)
        optimizer = torch.optim.Adam(student.parameters(), lr=LRS['student'], betas=(.5, .9)); d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LRS['discriminator'], betas=(.5, .9))
        step = restore(payload, student, discriminator, optimizer, d_optimizer, config, contract, parent, receipt, device, source == PARENT)
        scores = payload.get('development'); del payload
        state = {'status': 'running', 'attempt': number, 'resume_checkpoint': str(source), 'resume_checkpoint_sha256': source_sha,
            'completed_global_step': step, 'last_saved_global_step': step, 'partial_batch': False,
            'optimizer_cursors': {'student': optimizer_cursor(optimizer), 'discriminator': optimizer_cursor(d_optimizer)}, 'target_global_step': STOP}
        if source != PARENT: write(OUTPUT / 'latest_full.json', {'path': str(source), 'global_step': step, 'checkpoint_sha256': source_sha, 'run_contract_sha256': training.canonical_json_sha256(contract)})
        source_files(parent, contract['runtime_files']); write(attempt / 'resume.json', state)
        saved_rng = training.capture_rng_state(); protocol = development['reconstruction_protocol']
        dev_batch = baseline.batch_for(dataset, development['reconstruction_dev_indices'], protocol['crop_seed'])
        dev_noise = torch.randn(dev_batch['mel'].shape, device=device, generator=torch.Generator(device=device).manual_seed(protocol['source_noise_seed']))
        if any(recovery.probe.tensor_sha(dev_batch[k]) != parent['fixed_development_inputs'][k] for k in ('mel', 'bnf', 'spk_emb')) or recovery.probe.tensor_sha(dev_noise) != parent['fixed_development_source_noise_sha256']: raise RuntimeError('Original fixedDEV inputs changed')
        training.restore_rng_state(saved_rng, device)
        pending = []; excluded = set(development['excluded_train_indices']); resumed_step = step; train_started = time.monotonic()
        for step in range(step + 1, STOP + 1):
            if stop['requested']: break
            require_space(OUTPUT); indices = planner.for_step(step)
            if len(indices) != 32 or set(indices) & excluded: raise RuntimeError('Invalid retained full batch')
            batch = baseline.batch_for(dataset, indices, 101234 + step); training.seed_everything(201234 + step)
            losses = complete_batch(state, step, optimizer, d_optimizer, lambda: training.train_batch(batch, teacher, student, vocoder,
                discriminator, diffusion, optimizer, d_optimizer, mean, std, device, config))
            append(attempt / 'step_losses.jsonl', dict(losses, global_step=step, indices=indices, crop_seed=101234 + step, noise_seed=201234 + step))
            pending.append(losses)
            if step % STATUS_EVERY == 0:
                elapsed = time.monotonic() - train_started
                state.update(training_elapsed_seconds=elapsed, elapsed_total_seconds=time.monotonic()-started,
                    recent_loss_means={k: math.fsum(x[k] for x in pending) / len(pending) for k in LOSS_KEYS},
                    remaining_updates=STOP-step, estimated_remaining_seconds=(STOP-step)*elapsed/max(1,step-resumed_step))
                append(attempt / 'loss_windows.jsonl', dict(global_step=step, count=len(pending), means=state['recent_loss_means'])); pending=[]
                write(OUTPUT / 'status.json', state); print(json.dumps(state, separators=(',', ':')), flush=True)
            if step % SAVE_EVERY == 0 or step == STOP or stop['requested']:
                if environment() != contract['runtime']: raise RuntimeError('Training runtime changed')
                saved_rng = training.capture_rng_state()
                try: scores = baseline.evaluate_reconstruction(student, diffusion, dev_batch, dev_noise, config, device)
                finally: training.restore_rng_state(saved_rng, device)
                state['last_development_step'] = step; append(attempt / 'development.jsonl', dict(global_step=step, **scores))
                save_latest(OUTPUT, state, student, discriminator, optimizer, d_optimizer, config, contract, scores)
        if state['completed_global_step'] > state['last_saved_global_step']:
            save_latest(OUTPUT, state, student, discriminator, optimizer, d_optimizer, config, contract, scores)
        verify(contract['runtime_files'])
        state.update(status='completed100passes' if state['completed_global_step'] == STOP else 'stopped_at_user_signal', elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status='failed', execution_error=repr(error), no_partial_checkpoint_written=True)
        if attempt:
            try: write(attempt / 'failure.json', state); write(OUTPUT / 'status.json', state)
            except BaseException as failed: print('Failure receipt unavailable: ' + repr(failed), file=sys.stderr, flush=True)
        raise
    finally:
        cleanup_errors = release_runtime(handlers, locks)
        if cleanup_errors:
            state.update(status='failed', cleanup_errors=cleanup_errors)
            if attempt:
                try: write(attempt / 'failure.json', state); write(OUTPUT / 'status.json', state)
                except BaseException as failed: print('Cleanup receipt unavailable: ' + repr(failed), file=sys.stderr, flush=True)
            if sys.exc_info()[0] is None: raise RuntimeError('Runtime cleanup failed: ' + repr(cleanup_errors))
    # 清理失败或训练异常都会在上面退出，不留下正常完成收据。
    try: write(OUTPUT / 'status.json', state); write(attempt / 'completion.json', state)
    except BaseException as error:
        state.update(status='failed', execution_error=repr(error))
        try: write(attempt / 'failure.json', state)
        except BaseException: pass
        raise


if __name__ == '__main__': main()
