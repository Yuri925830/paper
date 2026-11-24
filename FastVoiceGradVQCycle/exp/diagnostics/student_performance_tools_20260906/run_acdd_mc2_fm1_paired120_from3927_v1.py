"""FM1候选至多120步；复用已完成FM2控制，不重训、不评估、不删除。"""

import argparse
import fcntl
import hashlib
import importlib
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
DLR_SHA = 'e2eb8d53f18999874dab447d1b124b6c34bd0a0d056d0323df59b320e14ff821'
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Run only on the registered Matpool rental')
if hashlib.sha256((TOOLS / 'run_acdd_mc2_dlr_paired120_from3927_v1.py').read_bytes()).hexdigest() != DLR_SHA:
    raise RuntimeError('Frozen Dlr primitives changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_mc2_dlr_paired120_from3927_v1 as d

old, previous, cross, pair, base = d.old, d.previous, d.cross, d.pair, d.base
training, resume, probe = d.training, d.resume, d.probe
DEV, START, STOP = d.DEV, 3927, 4047
OUTPUT = DEV / 'acdd_mc2_fm1_paired120_from3927_v1'
ARM = 'candidate_fm1'
PROTOCOL = DEV / 'FM1_MC2_PAIRED120_PLAN_CN.md'
HELPER = 'acdd_mc2_fm1_train_batch_3928_4047_v1.py'
HELPER_TESTS = 'test_acdd_mc2_fm1_train_batch_3928_4047_v1.py'
LRS = {'student': .00005, 'discriminator': .0002}
FORWARD_LIMITS = dict(d.FORWARD_LIMITS)
LOSS_KEYS = pair.LOSS_KEYS + ('actual_total', 'paper_equivalent_total')
CONTROL_CONTRACT_SHA = 'c7b403ff9d84257bea2f80a9f8523d90bc94a30edbf1811de3c9f0fcaa5d8eb4'
CONTROL_CANONICAL = '9aba349cdcf14b39cc224abf02fb1a95107fb8ff6a9fad8bb0d7da13137ce551'
CONTROL_ARCHIVE_SHA = '1f3f69658dd2498a2b577295ec2925ae98c05f66d8fbfc2d261a11edc883afc9'
CONTROL_COMPLETION_SHA = '221f9aa42d24c2ff3e94a753dba920042e38034e5b0a0bd21fe43f11b41f376c'
EVALUATION = {'output_root': str(DEV / 'evaluation_mc2_fm1_paired120_endpoint_v1'),
              'label': 'student_mc2_fm1_paired120_endpoint_v1'}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    for name in ('driver', 'tests', 'helper', 'helper-tests', 'protocol'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args(argv)
    if not args.run_approved: raise RuntimeError('Explicit candidate-only bounded approval required')
    for name in ('driver', 'tests', 'helper', 'helper_tests', 'protocol'):
        value = getattr(args, name + '_sha256')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Every new source/test/protocol needs its reviewed full SHA256')
    return args


class Budget(d.Budget):
    def __init__(self, output):
        # 沿用16MiB总metadata记账与原子commit注销；本次只有一个full。
        self.output, self.arm, self.cancelled = Path(output), ARM, {}
        self.pending = {'full': d.FULL_BYTES + previous.MODEL_SLACK, 'later_DEV90': d.EVAL_BYTES}


def candidate_config(parent_config):
    if parent_config.lambda_fm != 2 or parent_config.lambda_dist != 45:
        raise RuntimeError('Restore the true FM2/dist45 parent before applying FM1')
    effective = base.replace(parent_config, lambda_fm=1.0)
    before, after = parent_config.as_dict(), effective.as_dict()
    if {key for key in before if before[key] != after[key]} != {'lambda_fm'} or set(before) != set(after):
        raise RuntimeError('Only lambda_fm may differ from the parent')
    return effective


def restore_parent(payload, student, discriminator, optimizer, d_optimizer, parent_config, contract, receipt, device):
    # 原control角色的D lr仍是2e-4；先完整恢复，不把FM1冒称父配置。
    restored = d.restore_parent(payload, student, discriminator, optimizer, d_optimizer,
                                parent_config, contract, receipt, device, d.ARMS[0])
    if restored['parent_restore'] != restored['role_lr_applied']:
        raise RuntimeError('FM experiment may not change any restored optimizer or model content')
    effective = candidate_config(parent_config)
    return effective, {'parent_restore': restored['parent_restore'],
                       'all_models_optimizers_rng_restored_without_change': True,
                       'configuration_delta': {'lambda_fm': {'parent': 2.0, 'candidate': 1.0}}}


def loss_columns(losses, fm_weight):
    pair.finite_losses(losses)
    if fm_weight not in (1., 2.): raise RuntimeError('Undeclared FM coefficient')
    adv, fm, dist = (float(losses[k]) for k in ('student_adv', 'fm_raw', 'dist_raw'))
    expected = {'fm_weighted': fm_weight * fm, 'dist_weighted': 45. * dist,
                'student_total': math.fsum((adv, fm_weight * fm, 45. * dist))}
    d.same_tree({k: float(losses[k]) for k in expected}, expected)
    result = dict(losses, actual_total=float(losses['student_total']),
                  paper_equivalent_total=math.fsum((adv, 2. * fm, 45. * dist)))
    if any(not math.isfinite(float(result[k])) for k in LOSS_KEYS): raise RuntimeError('Nonfinite derived loss')
    return result


def paired_input_check(row, control):
    # 旧函数核全输入/噪声与首步D、dist，本次同D lr还需核首步adv和FM raw。
    d.paired_input_check(row, control)
    if row['global_step'] == START + 1:
        d.same_tree({k: row[k] for k in ('student_adv', 'fm_raw', 'dist_weighted')},
                    {k: control[k] for k in ('student_adv', 'fm_raw', 'dist_weighted')})
        expected = loss_columns(control, 2.)
        d.same_tree(row['paper_equivalent_total'], expected['paper_equivalent_total'])


def window60(rows):
    # 旧窗口先验证全部绝对步和原7字段，新两种total也覆盖同一完整60步。
    result = d.window60(rows)
    for key in ('actual_total', 'paper_equivalent_total'):
        result['means'][key] = math.fsum(float(r[key]) for r in rows) / 60
        result['first_last_difference'][key] = rows[-1][key] - rows[0][key]
    return result


def run_updates(state, future, step_call, evaluate, publish, check, initial):
    if [r['global_step'] for r in future] != list(range(3928, 4048)) or initial['global_step'] != START:
        raise RuntimeError('Only the original absolute120 plan is allowed')
    if old.guard_reasons([initial], d.cont.BASELINE3327): raise RuntimeError('Initial3927 replay violates original guard')
    rows, windows, monitors, reason = [], [], [initial], 'maximum120_updates_completed'
    for planned in future:
        check(); row = step_call(planned); state['unpublished_step_record'] = row
        publish('step_losses.jsonl', row); state.pop('unpublished_step_record', None); rows.append(row)
        state['complete_loss_records'] = len(rows)
        if len(rows) % 60 == 0:
            window = window60(rows[-60:]); windows.append(window); publish('loss_windows.jsonl', window)
            entry = evaluate(planned['global_step']); monitors.append(entry); state['last_monitor'] = entry
            publish('monitor.jsonl', entry)
            guards = old.guard_reasons(monitors[1:], d.cont.BASELINE3327)
            state['guard_reasons'] = guards
            publish('guard_checks.jsonl', {'global_step': planned['global_step'], 'stop_reasons': guards})
            if guards: reason = guards[0]; break
        publish('status.json', state, False)
    return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': reason}


def control_evidence(parent, future, files, check):
    # 固定旧契约/旧归档，不用新契约去冒认历史控制归属。
    paths = {str(d.OUTPUT / 'run_contract.json'): CONTROL_CONTRACT_SHA,
             str(d.OUTPUT / d.ARCHIVE_NAME): CONTROL_ARCHIVE_SHA,
             str(d.OUTPUT / d.ARMS[0] / 'completion.json'): CONTROL_COMPLETION_SHA}
    d.check_files(paths, check); joint = cross.read_json(d.OUTPUT / 'run_contract.json')
    if (training.canonical_json_sha256(joint), joint['parent_checkpoint_sha256'], joint['configuration'],
        joint['future_batches'], joint['effective_optimizer_lrs_by_arm'][d.ARMS[0]]) != (
        CONTROL_CANONICAL, d.PARENT_SHA, parent['configuration'], future, LRS):
        raise RuntimeError('Historical control is not the same parent/config/lr/absolute plan')
    rows, archive = d.candidate_gate(SimpleNamespace(control_archive_sha256=CONTROL_ARCHIVE_SHA), joint, check)
    done = cross.read_json(d.OUTPUT / d.ARMS[0] / 'completion.json')
    if done['status'] != 'completed120' or done['state']['completed_global_step'] != STOP or len(rows) != 120:
        raise RuntimeError('Require the actual fully completed120 historical control')
    if done['actual_forward_counts'] != FORWARD_LIMITS: raise RuntimeError('Historical control call counts differ')
    # 旧日志此前未被父契约逐项pin；新候选执行前首次显式登记这些保留字节。
    kept = [d.OUTPUT / d.ARMS[0] / name for name in ('step_losses.jsonl', 'loss_windows.jsonl', 'monitor.jsonl')]
    ev = Path(d.EVALUATION[d.ARMS[0]]['output_root']); system = ev / d.EVALUATION[d.ARMS[0]]['label']
    kept += [ev / 'run_identity.json', system / 'inference_complete.json', system / 'pairs.csv',
             system / 'metrics/development_summary_cer_dnsmos_sva_utmos.json']
    kept += [system / 'metrics' / (name + '.csv') for name in ('cer', 'sva', 'dnsmos', 'utmos')]
    for path in kept:
        check(); cross.regular(path); paths[str(path)] = base.sha256_file(path)
    for planned in future:
        row = rows[planned['global_step']]
        for key in ('global_step', 'indices', 'crop_seed', 'noise_seed'):
            if row[key] != planned[key]: raise RuntimeError('Control raw plan mismatch')
        loss_columns(row, 2.)
    cross.merge_hashes(files, paths)
    return rows, {'role': 'post_registered_reused_historical_control', 'original_contract_canonical_sha256': CONTROL_CANONICAL,
                  'files_sha256': paths, 'archive': archive, 'control_retrained_or_rescored': False,
                  'retained_logs_new_preexecution_binding_not_claimed_old_parent_pin': True}


def prepare_context(args, check):
    # 只收当前模型/监测/训练数据与实际import源码，去掉不再实际读取的旧FM诊断结果。
    omit = {str(DEV / 'acdd_mc2_fm_cross_3447_3927_v1/results.json')}
    files = {p: sha for p, sha in d.FIXED_PINS.items() if p not in omit}
    files.update(d.NEW_SOURCE_PINS)
    files.update({str(TOOLS / 'run_acdd_mc2_dlr_paired120_from3927_v1.py'): DLR_SHA,
                  str(TOOLS / 'acdd_mc2_train_batch_3928_4047_v1.py'): d.HELPER_SHA,
                  str(TOOLS / 'acdd_mc2_train_batch_v1.py'): 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16',
                  str(Path(__file__).resolve()): args.driver_sha256,
                  str(TOOLS / 'test_run_acdd_mc2_fm1_paired120_from3927_v1.py'): args.tests_sha256,
                  str(TOOLS / HELPER): args.helper_sha256, str(TOOLS / HELPER_TESTS): args.helper_tests_sha256,
                  str(PROTOCOL): args.protocol_sha256})
    d.check_files(files, check); d.validate_supplemental_source_pins()
    mc = importlib.import_module(HELPER[:-3])
    if (mc.START, mc.STOP, mc.MC2_SEED, mc.FM_WEIGHT) != (START, STOP, 74000000, 1.): raise RuntimeError('Wrong FM1 helper')
    parent = cross.read_json(d.PARENT_RUN / 'run_contract.json')
    receipt = d.validate_parent_metadata(parent, cross.read_json(d.PARENT_RUN / 'completion.json'))
    parent_config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if parent['configuration'] != parent_config.as_dict(): raise RuntimeError('Wrong originalFM2 parent config')
    effective = candidate_config(parent_config)
    development, dependencies = resume.load_development(DEV / 'development_contract.json')
    if parent['training_indices'] != development['retained_train_indices'] or parent['excluded_train_indices'] != development['excluded_train_indices']:
        raise RuntimeError('Original52 exclusions or retained training plan changed')
    cross.merge_hashes(files, dependencies)
    cross.merge_hashes(files, d.imported_source_files(parent['file_sha256'], files))
    for relative, digest in training.collect_source_fingerprints(ROOT).items():
        path = str(ROOT / relative) if not Path(relative).is_absolute() else relative
        if parent['file_sha256'].get(path) != digest: raise RuntimeError('Frozen production source changed')
        files[path] = digest
    for asset in (base.TEACHER, base.VOCODER_DIR / 'generator_step_2500000.pt', ROOT / 'data/stats/mel_mean.npy', ROOT / 'data/stats/mel_std.npy'):
        files[str(asset)] = parent['file_sha256'][str(asset)]
    plan = parent['monitor64_plan']
    if plan != cross.read_json(previous.CROSS_IDENTITY)['plan']: raise RuntimeError('Original64 monitor plan changed')
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=parent_config.bnf_alignment)
    if len(dataset) != 39387: raise RuntimeError('Original dataset length changed')
    future = d.future_plan(development)
    future_files = previous.register_future_files(dataset, future, files, check)
    monitor_files = previous.register_future_files(dataset, [{'indices': plan['selected_indices']}], files, check)
    if len(monitor_files) != 192 or any(parent['file_sha256'].get(p) != sha for p, sha in monitor_files.items()):
        raise RuntimeError('Original192 fixed64 features changed')
    controls, evidence = control_evidence(parent, future, files, check)
    d.check_files(files, check)
    return mc, parent, receipt, parent_config, effective, development, dataset, future, future_files, monitor_files, files, controls, evidence


def make_contract(args, parent, receipt, parent_config, config, development, future, future_files, monitor_files, files, evidence):
    return {'kind': 'MC2_FM1_candidate_against_completed_FM2_control', 'protocol_sha256': args.protocol_sha256,
        'configuration': config.as_dict(), 'parent_configuration': parent_config.as_dict(),
        'paper_reference_coefficients': {'lambda_fm': 2., 'lambda_dist': 45.}, 'candidate_deviation': {'lambda_fm': 1.},
        'parent_checkpoint': str(d.PARENT), 'parent_checkpoint_sha256': d.PARENT_SHA,
        'parent_contract_sha256': d.PARENT_CANONICAL, 'effective_optimizer_lrs': LRS,
        'mc_samples': 2, 'mc_private_seed_base': 74000000, 'start_after_global_step': START,
        'stop_global_step': STOP, 'maximum_updates': 120, 'future_batches': future,
        'future_training_feature_sha256': future_files, 'fixed64_feature_sha256': monitor_files,
        'training_indices': development['retained_train_indices'], 'excluded_train_indices': development['excluded_train_indices'],
        'input_identities': parent['input_identities'], 'monitor64_plan': parent['monitor64_plan'],
        'monitor_configuration': parent_config.as_dict(), 'baseline3927_monitor': receipt['last_monitor'],
        'baseline3327_development': d.cont.BASELINE3327, 'original_step0_guard_anchor': d.cont.S0_ANCHOR,
        'fixed_frozen_models_sha256': parent['fixed_frozen_models_sha256'], 'monitor_updates': [0, 60, 120],
        'loss_windows': [[3928, 3987], [3988, 4047]], 'loss_keys': list(LOSS_KEYS),
        'totals': {'actual_total': 'adv+FM+45dist (student_total is actual returned float)',
                   'paper_equivalent_total': 'adv+2FM+45dist from same raw scalars; no backward'},
        'historical_control': evidence, 'runtime_files': files, 'forward_limits': FORWARD_LIMITS,
        'storage': {'initial_required_bytes': 4832505856, 'full_upper_bytes': d.FULL_BYTES, 'save_slack': previous.MODEL_SLACK,
                    'system_reserve': previous.RESERVE, 'directory_reserve': previous.DIR_RESERVE,
                    'metadata_limit': d.META_LIMIT, 'later_DEV90_bytes': d.EVAL_BYTES},
        'evaluation': EVALUATION, 'metric_identity_sha256': d.METRIC_IDENTITY_SHA,
        'compute_seconds': 900, 'cleanup_only_seconds': 90, 'original_flags': cross.ORIGINAL_FLAGS,
        'no_automatic_evaluation_training_retry_archive_or_deletion': True, 'formal3327_remains_stopped': True}


def save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock):
    if not state['restore_completed'] or state['partial_batch_may_have_updated_D'] or state.get('execution_error'):
        raise RuntimeError('No checkpoint for technical failure or uncertain update')
    step = state['completed_global_step']; last = state['last_monitor']
    if step not in (3987, 4047) or last['global_step'] != step or state['arm'] != ARM or config.lambda_fm != 1.:
        raise RuntimeError('Save only the actual complete monitoredFM1 endpoint, never zero steps')
    d.validate_adams(student, discriminator, optimizer, d_optimizer, step, LRS)
    state['optimizer_cursors'] = old.actual_cursors(optimizer, d_optimizer)
    payload = {'kind': 'MC2_FM1_candidate_full', 'arm': ARM, 'global_step': step, 'epoch': 0, 'resumable': True,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'run_contract_sha256': state['contract_sha256'],
        'development': last['development'], 'development_monitor_global_step': step,
        'paper_reference_coefficients': {'lambda_fm': 2., 'lambda_dist': 45.},
        'lineage': {'parent_checkpoint_sha256': d.PARENT_SHA, 'parent_contract_sha256': d.PARENT_CANONICAL, 'start_after_global_step': START},
        'stop_reason': state['reason'], 'execution_error': None, 'actual_optimizer_cursors': state['optimizer_cursors'],
        'student': student.state_dict(), 'discriminator': discriminator.state_dict(),
        'student_optimizer': optimizer.state_dict(), 'discriminator_optimizer': d_optimizer.state_dict(),
        'rng_state': training.capture_rng_state()}
    path = budget.output / ARM / 'final_full.pt'; clock.active(); budget.require()
    def committed():
        state['full_snapshot'] = {'path': str(path), 'global_step': step, 'sha256': None, 'committed': True, 'resumable': True}
        state['last_full_saved_global_step'] = step; budget.committed('full', path)
    old.atomic_checkpoint_commit(payload, path, committed)
    state['full_snapshot']['sha256'] = base.sha256_file(path); path.chmod(0o444)
    state['full_snapshot']['frozen_readonly'] = True
    clock.active(); budget.write(path.parent / 'checkpoint_receipt.json', state)


def main(argv=None):
    started = time.monotonic(); clock = budget = log = None; locks = []; owned = False
    state, completion, stage = {}, None, OUTPUT / ARM; ledger = d.ForwardLedger()
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        args = arguments(argv); clock = old.StopClock(900, started); clock.install(); previous.watchdog(900)
        training.assert_rented_server(ROOT); cross.regular(DEV, directory=True)
        locks = training.acquire_runtime_locks(ROOT)
        if len(locks) != 3: raise RuntimeError('Require three actual workflow locks')
        path = ROOT / 'exp/locks/text_unseen_fastvoicegrad_student_inference.lock'
        if path.is_symlink(): raise RuntimeError('Inference lock symlink')
        handle = path.open('a+')
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException: handle.close(); raise
        locks.append(handle)
        if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('Candidate root must be entirely new')
        budget = Budget(OUTPUT); disk = budget.require(); OUTPUT.mkdir()
        log = d.StageLog(stage, budget); sys.stdout = sys.stderr = log; stage.mkdir(); owned = True
        def check(): clock.active(); budget.require()
        if training.git_head(ROOT) != cross.HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Keep the frozen clean production source')
        if ((platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6')
                or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve() or cross.flags() != cross.ORIGINAL_FLAGS):
            raise RuntimeError('Keep actual myconda runtime and original precision')
        context = prepare_context(args, check)
        mc, parent, receipt, parent_config, config, development, dataset, future, future_files, monitor_files, files, controls, evidence = context
        contract = make_contract(args, parent, receipt, parent_config, config, development, future, future_files, monitor_files, files, evidence)
        contract_sha = training.canonical_json_sha256(contract)
        state = d.new_state(d.ARMS[0], contract_sha); state.update(arm=ARM, effective_optimizer_lrs=dict(LRS))
        budget.write(OUTPUT / 'run_contract.json', contract)
        gpu = cross.gpu_gate(); device = torch.device('cuda'); check()
        diffusion = base.VoiceGradDiffusion(parent_config.diffusion_steps, parent_config.cosine_offset).to(device)
        inputs, batches, target, input_identity = d.prepare_inputs(parent, dataset, diffusion, device)
        check(); payload = resume.load_torch_checkpoint(d.PARENT)
        teacher, student, unused = base.load_teacher_and_student(base.TEACHER, parent_config, device); del unused
        vocoder, vconfig, unused = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
            base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', parent_config, device); del unused
        discriminator = base.build_paper_discriminator(ROOT, vconfig, parent_config, device)
        fixed_d = base.build_paper_discriminator(ROOT, vconfig, parent_config, device)
        fixed = resume.load_torch_checkpoint(pair.PARENT / 'latest_full.pt')
        if fixed['global_step'] != 3107 or fixed['run_contract_sha256'] != pair.PARENT_CONTRACT: raise RuntimeError('Wrong original fixedD3107')
        fixed_d.load_state_dict(fixed['discriminator'], strict=True); del fixed
        fixed_d.eval(); training.set_requires_grad(fixed_d, False)
        mean, std = base.load_mel_statistics(ROOT / 'data', device)
        d.imported_source_files(files, files)  # 延迟源码必须早已pin，不在此临时接纳未知文件。
        frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
        frozen_sha = {name: pair.model_sha(model) for name, model in frozen.items()}
        if frozen_sha != parent['fixed_frozen_models_sha256']: raise RuntimeError('Original frozen assets changed')
        models = dict(frozen, student=student, active_D=discriminator, diffusion=diffusion)
        optimizer = torch.optim.Adam(student.parameters(), lr=LRS['student'], betas=(.5, .9))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LRS['discriminator'], betas=(.5, .9))
        effective, restored = restore_parent(payload, student, discriminator, optimizer, d_optimizer, parent_config, parent, receipt, device); del payload
        if effective.as_dict() != config.as_dict(): raise RuntimeError('Restored candidate configuration differs from new contract')
        state.update(restore_completed=True, restore_identity=restored, optimizer_cursors=old.actual_cursors(optimizer, d_optimizer),
                     status='running', initial_disk=disk, initial_gpu=gpu)
        ledger.install(student, teacher, vocoder, discriminator, fixed_d)
        def unchanged():
            pair.assert_frozen(frozen, frozen_sha)
            if input_identity() != parent['input_identities'] or cross.flags() != cross.ORIGINAL_FLAGS: raise RuntimeError('Fixed inputs/flags changed')
        def evaluate(step):
            if step - START not in (0, 60, 120): raise RuntimeError('Undeclared monitoring cadence')
            check(); before = dict(ledger.counts)
            with previous.monitoring_state(models, device):
                entry = pair.monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, parent_config, device, step)
                entry['updates'] = entry['trial_updates'] = step - START
                entry['uniform64'] = previous.uniform64_monitor(student, teacher, diffusion, batches, parent['monitor64_plan'], parent_config, device, check)
            if ledger.delta(before) != {'teacher': 21, 'student': 4, 'vocoder': 2, 'active_D_branches': 0, 'fixed_D_branches': 16}: raise RuntimeError('Wrong monitor count')
            entry['actual_forward_counts'] = ledger.delta(before); unchanged(); return entry
        initial = evaluate(START)
        for key in ('fixed_train', 'development', 'uniform64'): d.same_tree(initial[key], receipt['last_monitor'][key])
        state['last_monitor'] = initial; budget.write(stage / 'monitor.jsonl', initial, append=True)
        def step_call(planned):
            step = planned['global_step']; batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
            training.seed_everything(planned['noise_seed']); identity = mc.original_input_identity(batch, device, config); mc_receipt = {}
            check(); before = dict(ledger.counts)
            losses = d.complete_update(state, step, optimizer, d_optimizer, lambda: mc.train_batch_mc2(
                batch, teacher, student, vocoder, discriminator, diffusion, optimizer, d_optimizer, mean, std, device, config,
                global_step=step, mc_receipt=mc_receipt))
            # 若新列的校验失败，仍保留真正已返回的raw值、步号和MC2证据。
            state['unpublished_returned_losses'] = {'global_step': step, 'losses': losses, 'mc': mc_receipt}
            row = dict(loss_columns(losses, 1.), global_step=step, updates=step - START, indices=planned['indices'],
                crop_seed=planned['crop_seed'], noise_seed=planned['noise_seed'], input_identity=identity, mc=mc_receipt,
                actual_optimizer_cursors=state['optimizer_cursors'], actual_forward_counts=ledger.delta(before))
            state['unpublished_step_record'] = row
            state.pop('unpublished_returned_losses', None)
            for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
                if mc_receipt[key] != identity[key]: raise RuntimeError('Original RNG mismatch')
            if row['actual_forward_counts'] != {'teacher': 2, 'student': 2, 'vocoder': 3, 'active_D_branches': 32, 'fixed_D_branches': 0}: raise RuntimeError('Wrong train count')
            previous.adam_counters(optimizer, d_optimizer, step)
            if [optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr']] != list(LRS.values()): raise RuntimeError('Unexpected learning-rate drift')
            paired_input_check(row, controls[step]); return row
        result = run_updates(state, future, step_call, evaluate,
            lambda name, value, append=True: budget.write(stage / name, value, append), check, initial)
        state.update(reason=result['reason'], loss_windows=result['windows'], monitor_global_steps=[r['global_step'] for r in result['monitors']],
                     status='completed120' if result['reason'] == 'maximum120_updates_completed' else 'protective_stop', phase='saving')
        # 旧控制只读派生两种total对照，完全不改其日志/窗口文件。
        comparison = []
        for window in result['windows']:
            control_window = window60([loss_columns(controls[step], 2.) for step in range(window['first_global_step'], window['global_step'] + 1)])
            comparison.append({'first_global_step': window['first_global_step'], 'global_step': window['global_step'],
                'candidate': window, 'historical_control': control_window,
                'mean_candidate_minus_control': {k: window['means'][k] - control_window['means'][k] for k in LOSS_KEYS},
                'coefficient_change_is_not_quality_improvement': True})
        budget.write(stage / 'loss_comparison.json', comparison)
        unchanged(); save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock)
        d.imported_source_files(files, files); d.check_files(files, check); unchanged()
        if training.canonical_json_sha256(cross.read_json(OUTPUT / 'run_contract.json')) != contract_sha: raise RuntimeError('New contract mutated')
        if state['new_updates'] == 120 and ledger.counts != FORWARD_LIMITS: raise RuntimeError('Full run counts differ')
        completion = {'status': state['status'], 'cleanup_verified': True, 'state': dict(state), 'contract_sha256': contract_sha,
            'actual_forward_counts': dict(ledger.counts), 'elapsed_seconds': time.monotonic() - started,
            'actual_common_prefix_steps': state['new_updates'], 'control_retrained_or_rescored': False,
            'no_evaluation_or_archive_executed': True, 'automatic_next_stage_authorized': False}
        budget.write(stage / 'trial_completion.json', dict(completion, cleanup_verified=False, status='pending_cleanup')); check()
    except BaseException as error:
        if clock: clock.request('technical_failure')
        if budget: budget.cancel_future_on_fatal_abort(repr(error))
        state.update(status='technical_failure', execution_error=repr(error))
        if owned:
            try: budget.write(stage / 'failure.json', {'status': 'technical_failure', 'state': state, 'actual_forward_counts': ledger.counts, 'error': repr(error)})
            except BaseException as failed: saved_stderr.write('Failure receipt unavailable: %r; original=%r\n' % (failed, error))
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = old.cleanup_resources(ledger, clock, log, locks)
        if errors and budget: budget.cancel_future_on_fatal_abort(repr(errors))
        if errors and owned:
            try: budget.write(stage / 'cleanup_failure.json', {'status': 'technical_failure', 'errors': errors, 'state': state})
            except BaseException as error: saved_stderr.write('Cleanup receipt unavailable: %r\n' % error)
        if errors and original_error is None: raise RuntimeError('Candidate cleanup failed: ' + repr(errors))
        if original_error is None and not errors and completion is not None:
            try: clock.active(); budget.publish_success(stage / 'completion.json', completion)
            except BaseException as error:
                budget.cancel_future_on_fatal_abort(repr(error))
                try: budget.write(stage / 'failure.json', {'status': 'technical_failure', 'state': state, 'error': repr(error)})
                except BaseException: pass
                raise


if __name__ == '__main__': main()
