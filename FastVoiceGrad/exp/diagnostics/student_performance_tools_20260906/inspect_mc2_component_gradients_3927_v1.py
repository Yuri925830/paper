"""固定3927状态的一次分项梯度检查；不创建优化器、不更新或保存模型。"""

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Run project diagnostics only on the Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
BOOT = {
    # 旧封装在import时已校验此文件；新执行清单也必须显式携带同一既有身份。
    'inspect_acdd_fm_crossed_v1.py': '1e4cca2b1d8f437458c158c6b838c685693fec2f8925602c5f2b880aeebab9a8',
    'inspect_acdd_mc2_fm_cross_3447_3927_v1.py': '8db03a2545488985e5d7dd032ec9001928f352d1579699927cb589f344fe5873',
    'acdd_mc2_train_batch_3928_4047_v1.py': '6f04abb3838f574419986829c68196eddc72b3fe545d98a3e6a9f4162beffec3',
    'acdd_mc2_train_batch_v1.py': 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16',
}
for name, digest in BOOT.items():
    if hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen primitive changed: ' + name)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
import torch
import inspect_acdd_mc2_fm_cross_3447_3927_v1 as envelope
import acdd_mc2_train_batch_3928_4047_v1 as mc

base, training, resume = envelope.baseline, envelope.training, envelope.resume
DEV = envelope.DEV
OUTPUT = DEV / 'acdd_mc2_component_gradients_3927_v1'
PROTOCOL = DEV / 'MC2_3927_COMPONENT_GRADIENT_PLAN_CN.md'
PROTOCOL_SHA = '248e2db64c27b835f04db88f92a10ea3d61045d61d0caf6b70c6fa73bdb8338a'
PARENT_RUN = DEV / 'acdd_mc2_continuation480_from3447_v1'
PARENT = PARENT_RUN / 'final_full.pt'
PARENT_SHA = '6a4f5beabcd3264e7aa016b3236f98da174af396b2ce7fb0c0b5b083f8aa2630'
CANONICAL = 'd3e6c7a7f98e58d02985ac5328ae86bcb143ec3db7469b73b79b7c6d93377d96'
LIMITS = {'student': 1, 'teacher': 2, 'vocoder': 2, 'D_branch': 16, 'D_top': 0}
INPUT_KEYS = ('x0', 'bnf', 'spk_emb', 'source_noise', 'score_noise', 'score_timesteps', 'q950')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-approved', action='store_true')
    for name in ('driver', 'tests', 'helper', 'helper-tests'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args(argv)
    if not args.probe_approved:
        raise RuntimeError('An explicit bounded read-only probe approval is required')
    for name in ('driver', 'tests', 'helper', 'helper_tests'):
        value = getattr(args, name + '_sha256')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Each reviewed source/test needs its full SHA256')
    return args


def validate_parent(contract, completion, payload=None):
    if training.canonical_json_sha256(contract) != CANONICAL:
        raise RuntimeError('Wrong3927 canonical contract')
    state = completion['state']
    if (completion['status'], completion['cleanup_verified'], completion['contract_sha256']) != ('completed480', True, CANONICAL):
        raise RuntimeError('Require the successful complete3927 state')
    if (state['completed_global_step'], state['last_full_saved_global_step'], state['partial_batch_may_have_updated_D']) != (3927, 3927, False):
        raise RuntimeError('Incomplete or partial parent state')
    if (state['full_snapshot']['path'], state['full_snapshot']['sha256'], state['full_snapshot']['resumable']) != (str(PARENT), PARENT_SHA, True):
        raise RuntimeError('Parent snapshot identity differs')
    if payload is not None:
        actual = tuple(payload[k] for k in ('global_step', 'epoch', 'mc_samples', 'resumable', 'run_contract_sha256', 'score_semantics', 'execution_error'))
        expected = (3927, 0, 2, True, CANONICAL, 'reconstruction_clean_x0_sqrt_alpha_bar', None)
        if actual != expected or payload['configuration'] != contract['configuration']:
            raise RuntimeError('Actual full header or configuration differs')
        if payload['development'] != state['last_monitor']['development']:
            raise RuntimeError('Full state and endpoint receipt disagree')
    return state


def input_check(raw, contract):
    identities = {name: envelope.old.tensor_sha(value) for name, value in raw.items()}
    if identities != contract['input_identities']['old']:
        raise RuntimeError('Original fixedTRAIN3108 archive changed')
    shapes = dict(envelope.INPUT_SHAPES, score_noise=(32, 80, 128), score_timesteps=(32,), q950=(32, 80, 128))
    for name in INPUT_KEYS:
        value = raw[name]
        dtype = torch.long if name == 'score_timesteps' else torch.float32
        if tuple(value.shape) != shapes[name] or value.dtype != dtype or not bool(torch.isfinite(value).all()):
            raise RuntimeError('Input shape, dtype or finite value differs: ' + name)
    if not bool(((raw['score_timesteps'] >= 1) & (raw['score_timesteps'] <= 1000)).all()):
        raise RuntimeError('Teacher times outside1..1000')
    return identities


def snapshot(model):
    # 包括非持久buffer、每个子层模式和requires_grad；学生允许求导，但不允许更新。
    return {'tensors': envelope.old.fingerprint_model(model),
            'modes': {n: m.training for n, m in model.named_modules()},
            'requires_grad': {n: p.requires_grad for n, p in model.named_parameters()}}


def unchanged(model, expected):
    if snapshot(model) != expected or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Model state, mode, flags or parameter.grad changed')


def source_files(contract, files):
    # 只验证当前实际导入的项目源码，不遍历已归档祖先权重或历史音频。
    actual = set()
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and Path(name).suffix == '.py' and ROOT in Path(name).resolve().parents:
            actual.add(str(Path(name).resolve()))
    actual.add(str(TOOLS / 'acdd_mc2_train_batch_v1.py'))  # 私有MC2模块未注册在sys.modules。
    for path in sorted(actual):
        if path in files:
            continue
        expected = contract['file_sha256'].get(path)
        if expected is None:
            # 少数__init__.py从已确认干净的同一Git提交取身份，不信任无来源的新文件。
            if Path(path).name != '__init__.py':
                raise RuntimeError('Imported source lacks a predeclared identity: ' + path)
            relative = Path(path).relative_to(ROOT).as_posix()
            contents = subprocess.check_output(['git', 'show', base.EXPECTED_HEAD + ':' + relative], cwd=ROOT)
            expected = hashlib.sha256(contents).hexdigest()
        files[path] = expected
    return files


def main(argv=None):
    args = arguments(argv)
    # 沿用已审核的300秒时钟、小日志预算及互斥收尾，不调用旧四格诊断main。
    clock, budget = envelope.Clock(), envelope.Budget(OUTPUT)
    locks, models, handles, cleanup_errors = [], {}, [], []
    log = None
    original_rng = original_flags = device = None
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    counts = dict.fromkeys(LIMITS, 0)
    owned, complete, failure = False, None, None
    restored = {'models_unchanged': False, 'rng_restored': False, 'flags_unchanged_and_restored': False}

    def check():
        clock.active()
        budget.require()

    def register(model, name):
        def hook(module, inputs):
            check()
            if counts[name] >= LIMITS[name]:
                raise RuntimeError('Declared forward limit exceeded: ' + name)
            counts[name] += 1
        handles.append(model.register_forward_pre_hook(hook))

    try:
        clock.install()
        parent_process = Path('/proc') / str(os.getppid())
        command = [x.decode() for x in (parent_process / 'cmdline').read_bytes().split(b'\0') if x]
        if (parent_process / 'exe').resolve() != Path('/usr/bin/timeout').resolve() or not all(x in command for x in ('--signal=TERM', '--kill-after=30s', '300s')):
            raise RuntimeError('Require the declared external300+30second timeout')
        training.assert_rented_server(ROOT)
        envelope.regular(DEV, True)
        locks = training.acquire_runtime_locks(ROOT)
        lock_path = ROOT / 'exp/locks/text_unseen_fastvoicegrad_student_inference.lock'
        if lock_path.is_symlink():
            raise RuntimeError('Lock must not be a symlink')
        lock = lock_path.open('a+')
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lock.close()
            raise
        locks.append(lock)
        if len(locks) != 4:
            raise RuntimeError('Four workflow locks required')
        disk = budget.require()
        runtime = envelope.runtime_gate()
        if runtime['initial_GPU_free_MiB'] < 32768:
            raise RuntimeError('Retain32GiB free GPU for intact batch32 parameter VJPs')
        if training.git_head(ROOT) != base.EXPECTED_HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Keep the frozen clean tracked runtime')
        files = {str(TOOLS / n): sha for n, sha in BOOT.items()}
        files.update({str(Path(__file__).resolve()): args.driver_sha256,
            str(TOOLS / 'test_inspect_mc2_component_gradients_3927_v1.py'): args.tests_sha256,
            str(TOOLS / 'mc2_component_gradients_v1.py'): args.helper_sha256,
            str(TOOLS / 'test_mc2_component_gradients_v1.py'): args.helper_tests_sha256,
            str(PROTOCOL): PROTOCOL_SHA, str(PARENT): PARENT_SHA,
            str(PARENT_RUN / 'run_contract.json'): envelope.CONTRACT_SHAS[3927],
            str(PARENT_RUN / 'completion.json'): envelope.COMPLETION_SHAS[3927],
            str(envelope.INPUT): envelope.INPUT_SHA})
        files.update(envelope.NEW_G_ASSET_SHAS)
        envelope.verify_files(files, check)
        helper = importlib.import_module('mc2_component_gradients_v1')
        contract = envelope.read_json(PARENT_RUN / 'run_contract.json')
        receipt = envelope.read_json(PARENT_RUN / 'completion.json')
        validate_parent(contract, receipt)
        config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
        if config.as_dict() != contract['configuration'] or (config.score_target, config.score_weight, config.lambda_fm, config.lambda_dist) != ('clean_x0', 'sqrt_alpha_bar', 2, 45):
            raise RuntimeError('Keep currentMC2 objective and original coefficients')
        for path in (base.TEACHER, base.VOCODER_DIR / 'generator_step_2500000.pt', ROOT / 'data/stats/mel_mean.npy', ROOT / 'data/stats/mel_std.npy'):
            files[str(path)] = contract['file_sha256'][str(path)]
        source_files(contract, files)
        envelope.verify_files(files, check)
        log = envelope.LimitedLog(OUTPUT, budget)
        OUTPUT.mkdir()
        owned = True
        sys.stdout = sys.stderr = log
        torch.set_num_threads(2)
        device = torch.device('cuda')
        original_rng, original_flags = training.capture_rng_state(), envelope.flags()
        torch.cuda.reset_peak_memory_stats(device)
        raw = resume.load_torch_checkpoint(envelope.INPUT)
        archived_ids = input_check(raw, contract)
        inputs = {name: raw[name].to(device) for name in INPUT_KEYS}
        del raw
        input_ids = {n: envelope.old.tensor_identity(v) for n, v in inputs.items()}
        payload = resume.load_torch_checkpoint(PARENT)
        validate_parent(contract, receipt, payload)
        weights = {n: payload[n] for n in ('student', 'discriminator')}
        del payload  # 未使用Adam/RNG立即释放，绝不构造optimizer。
        check()
        teacher, student, _ = base.load_teacher_and_student(base.TEACHER, config, device)
        student.load_state_dict(weights['student'], strict=True)
        vocoder, vconfig, _ = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt', base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
        discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
        discriminator.load_state_dict(weights['discriminator'], strict=True)
        del weights
        diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
        mean, std = base.load_mel_statistics(ROOT / 'data', device)
        student.train().requires_grad_(True)
        discriminator.train().requires_grad_(False)
        teacher.eval().requires_grad_(False)
        vocoder.eval().requires_grad_(False)
        diffusion.eval().requires_grad_(False)
        for name, model in (('student', student), ('teacher', teacher), ('vocoder', vocoder), ('discriminator', discriminator), ('diffusion', diffusion)):
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Freshly loaded model has preexisting parameter.grad')
            models[name] = (model, snapshot(model))
        for model, name in ((student, 'student'), (teacher, 'teacher'), (vocoder, 'vocoder'), (discriminator, 'D_top')):
            register(model, name)
        for branch in discriminator.loss_branches():
            register(branch, 'D_branch')
        source_files(contract, files)
        envelope.verify_files(files, check)
        budget.write('run_contract.json', {'kind': 'MC2_fixed3927_component_gradients', 'file_sha256': files,
            'configuration': config.as_dict(), 'parent_step': 3927, 'input_step': 3108, 'second_draw_global_step': 3928,
            'input_identities': input_ids, 'archive_all_tensor_sha256': archived_ids, 'forward_limits': LIMITS,
            'flags': original_flags, 'runtime': runtime, 'initial_disk': disk, 'optimizer_updates': 0,
            'not_actual_next3928_training_replay': True, 'not_objective_metric_evaluation': True})
        print('MC2_COMPONENTS starting one fixed graph', flush=True)
        prediction, q950 = training.make_student_prediction(student, diffusion, inputs['x0'], inputs['spk_emb'], inputs['bnf'], 950, inputs['source_noise'])
        if not torch.equal(q950, inputs['q950']):
            raise RuntimeError('Originalq950 did not replay')
        readout = {'mel_l1': float((prediction - inputs['x0']).abs().mean().detach()), 'rms': float(prediction.detach().square().mean().sqrt())}
        envelope.close_numbers(readout, envelope.ANCHORS[3927])
        mc_receipt = {}
        dist, targets = mc.mc2_score_distillation_loss(prediction, teacher, diffusion, inputs['score_timesteps'], inputs['score_noise'], inputs['spk_emb'], inputs['bnf'], target_mode=config.score_target, weight_mode=config.score_weight, global_step=3928, mc_receipt=mc_receipt)
        template_budget = mc_receipt.pop('training_forward_budget')
        with torch.no_grad():
            real_wave = vocoder(training.denormalize_mel(inputs['x0'], mean, std))
        fake_wave = vocoder(training.denormalize_mel(prediction, mean, std))
        for wave in (real_wave, fake_wave):
            if tuple(wave.shape) != (32, 1, 32768) or not bool(torch.isfinite(wave).all()):
                raise RuntimeError('Invalid full32 waveform')
        output_ids = {n: envelope.old.tensor_identity(v) for n, v in [('prediction', prediction), ('real_wave', real_wave), ('fake_wave', fake_wave)]}
        report = helper.component_gradients(torch=torch, training=training, prediction=prediction, reference_mel=inputs['x0'], fake_waveform=fake_wave, real_waveform=real_wave, discriminator=discriminator, dist_loss=dist, named_parameters=list(student.named_parameters()), check=check)
        if counts != LIMITS:
            raise RuntimeError('Actual forward counts differ from1S2T2G16D')
        if report['actual_counts'] != {'D_branch_forwards': 16, 'branch_autograd_calls': 24, 'parameter_vjp_calls': 5}:
            raise RuntimeError('Actual derivative call counts differ from the registered scope')
        for model, before in models.values():
            unchanged(model, before)
        if {n: envelope.old.tensor_identity(v) for n, v in inputs.items()} != input_ids:
            raise RuntimeError('Input tensor mutation')
        original_file_set = set(files)
        source_files(contract, files)
        if set(files) != original_file_set:
            raise RuntimeError('Project import closure changed during forward/backward')
        envelope.verify_files(files, check)
        result = {'status': 'results_ready_pending_cleanup', 'parameter_gradient_report': report, 'mc2_receipt': mc_receipt,
            'helper_original_training_template_not_observed_counts': template_budget, 'actual_forward_counts': counts,
            'fixed_train_readout': readout, 'input_and_output_identities': dict(input_ids, **output_ids),
            'file_sha256': files, 'models_unchanged_fingerprints': {n: before for n, (_, before) in models.items()},
            'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'process_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'elapsed_seconds': time.monotonic() - clock.started,
            'limitations': ['One fixedTRAIN3108 batch atS3927/D3927; no D update and not next3928 production replay.',
                'Parameter components are diagnostic splitVJPs; direct joint reference is computed independently.',
                'GTmel alignment does not establish CER/P808 causality or new quality scores. No Adam update or checkpoint.']}
        budget.write('results.json', result)
        complete = {'status': 'completed_readonly_MC2_component_gradients', 'result_sha256': base.sha256_file(OUTPUT / 'results.json'), 'actual_forward_counts': counts}
    except BaseException as error:
        failure = error
        clock.request('execution_error')
        budget.aborting = True
        raise
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        # 清理失败也保留证据；一次失败不会妨碍后续锁、日志和RNG清理。
        for name, (model, before) in models.items():
            try:
                unchanged(model, before)
            except BaseException as error:
                cleanup_errors.append({'part': name, 'error': repr(error)})
        restored['models_unchanged'] = bool(models) and not cleanup_errors
        if original_rng is not None:
            try:
                training.restore_rng_state(original_rng, device)
                restored['rng_restored'] = envelope.rng_fingerprint(training.capture_rng_state()) == envelope.rng_fingerprint(original_rng)
            except BaseException as error:
                cleanup_errors.append({'part': 'rng', 'error': repr(error)})
        if original_flags is not None:
            try:
                unchanged_flags = envelope.flags() == original_flags
                envelope.restore_flags(original_flags)
                restored['flags_unchanged_and_restored'] = unchanged_flags and envelope.flags() == original_flags
            except BaseException as error:
                cleanup_errors.append({'part': 'flags', 'error': repr(error)})
        for handle in reversed(handles):
            try:
                handle.remove()
            except BaseException as error:
                cleanup_errors.append({'part': 'hook', 'error': repr(error)})
        for name, action in [('clock', clock.close)] + ([('log', log.close)] if log else []) + [('lock', lock.close) for lock in reversed(locks)]:
            try:
                action()
            except BaseException as error:
                cleanup_errors.append({'part': name, 'error': repr(error)})
        if owned:
            if failure or cleanup_errors or not all(restored.values()) or complete is None:
                budget.aborting = True
                budget.write('failure.json', {'status': 'failed', 'error': repr(failure), 'cleanup_errors': cleanup_errors, 'cleanup': restored, 'counts': counts, 'automatic_retry_authorized': False})
                if failure is None:
                    raise RuntimeError('Read-only cleanup incomplete')
            else:
                try:
                    check()
                    budget.write('completion.json', dict(complete, cleanup_verified=True, cleanup=restored), terminal=True)
                except BaseException as error:
                    budget.aborting = True
                    budget.write('failure.json', {'status': 'failed', 'error': repr(error), 'cleanup_errors': cleanup_errors,
                        'cleanup': restored, 'counts': counts, 'automatic_retry_authorized': False})
                    raise


if __name__ == '__main__':
    main()
