"""完整恢复4287后仅240步感知辅助短训；权重必须来自已完成梯度诊断，不自动评估。"""

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
PRIOR_FILE = TOOLS / 'run_acdd_mc2_independent360_from3927_v1.py'
PRIOR_SHA = '043a914ad9215b24bd0fc1a02861dfe640cb71610c9345321313661607db99ae'
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file(): raise RuntimeError('Run only on the registered rental')
if hashlib.sha256(PRIOR_FILE.read_bytes()).hexdigest() != PRIOR_SHA: raise RuntimeError('Frozen independent360 primitives changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_mc2_independent360_from3927_v1 as prior

d, old, previous, cross, pair, base = prior.d, prior.old, prior.previous, prior.cross, prior.pair, prior.base
training, resume, probe, DEV = prior.training, prior.resume, prior.probe, prior.DEV
START, STOP, UPDATES = 4287, 4527, 240
OUTPUT = DEV / 'acdd_mc2_perceptual240_from4287_v1'
ARM = 'candidate_perceptual'
PARENT_ARM = 'candidate_mc2'
PROTOCOL = DEV / 'ACDD4287_MC2_PERCEPTUAL240_PROTOCOL_CN.md'
HELPER = 'acdd_mc2_perceptual_anchor4287_v1'
ANCHOR_HELPER = 'acdd_mc2_conversion_anchor4287_v1'
LOSS_HELPER = 'wavlm_conv_perceptual_loss_v1'
WAVLM = TOOLS / 'assets/wavlm_large_conv_c1423ed_v1.pt'
WAVLM_RECEIPT = TOOLS / 'assets/wavlm_large_conv_c1423ed_v1.receipt.json'
COMPUTE_SECONDS, CLEANUP_SECONDS = 3300, 90
FULL_UPPER, WRITE_OVERHEAD = 1460000000, 64 * 1024**2
META_LIMIT, EVAL_BYTES, SYSTEM_RESERVE, DIR_RESERVE = 32 * 1024**2, 32 * 1024**2, 1024**3, 1024**2
INITIAL_REQUIRED_BYTES = FULL_UPPER + WRITE_OVERHEAD + META_LIMIT + EVAL_BYTES + SYSTEM_RESERVE + DIR_RESERVE
SHUFFLE_SEED_BASE, REVERSE_SEED_BASE = 90000000, 91000000
AUX_KEYS = ('conversion_perceptual_raw', 'conversion_perceptual_weighted', 'conversion_perceptual_weight', 'student_joint_total')
GRADIENT_ROOT = DEV / 'wavlm_anchor_gradients4287_v1'
GRADIENT_FIXED_COUNTS = {'teacher': 62, 'student': 1, 'vocoder': 3}
PARENT_RUN = DEV / 'acdd_mc2_independent360_from3927_v1'
PARENT = PARENT_RUN / PARENT_ARM / 'final_full.pt'
PARENT_SHA = '5f680fccd37855c0ed58d6655df9dfd0a3ea11410fd629a32020e11eb8bddfdb'
PARENT_CANONICAL = '45213c2fc1b760d08780e1d209c8c9d589eb232d0bc62fdf588c7c1bccb3ad6d'
PARENT_CONTRACT_SHA = 'acd379068d66c51a01dcd62b1e9d91ac1e0434f3b28a54e2e169ef90cadc0263'
PARENT_COMPLETION_SHA = '331d6911e6763b4d16b40e16f7009103ea544fb6e7e2c725f5dedd2b6bb35506'
PARENT_RNG_SHA = '81a3673f136ddaf4f615003e823c2696a69d2586c8c4b8a4ee47fcc066957ac8'
LRS = dict(prior.LRS)
deadline_steps = prior.deadline_steps
runtime, failure_record = prior.runtime, prior.failure_record
TRAIN_COUNTS = {'teacher': 32, 'student': 3, 'vocoder': 5, 'active_D_branches': 32, 'fixed_D_branches': 0}
MONITOR_COUNTS = dict(prior.MONITOR_COUNTS)
FORWARD_LIMITS = {k: 240 * TRAIN_COUNTS[k] + 5 * MONITOR_COUNTS[k] for k in TRAIN_COUNTS}
MONITOR_UPDATES = tuple(range(0, UPDATES + 1, 60))
EVALUATION = {'output_root': str(DEV / 'evaluation_mc2_perceptual240_from4287_endpoint_v1'),
    'label': 'student_mc2_perceptual240_from4287_endpoint_v1', 'normal_endpoint_global_step': STOP,
    'metrics_only': ['cer', 'dnsmos', 'utmos'], 'executed_by_this_driver': False,
    'protective_endpoint_requires_separate_review': True}


class Budget(d.Budget):
    # 本轮只新增一份full，父full已在磁盘；明确降低本分配的系统余量，不改任何旧预算文件。
    def __init__(self, output):
        self.output, self.arm, self.cancelled = Path(output), ARM, {}
        self.pending = {'full': FULL_UPPER + WRITE_OVERHEAD, 'later_DEV90': EVAL_BYTES}
    def require(self, extra_metadata=0):
        used = self.metadata_used()
        if used + extra_metadata > META_LIMIT: raise pair.InsufficientDisk('Keep complete240 rows within32MiB metadata')
        need = SYSTEM_RESERVE + DIR_RESERVE + sum(self.pending.values()) + META_LIMIT - used
        free = pair.available_bytes(self.output.parent)
        if free < need: raise pair.InsufficientDisk('available=%d required=%d' % (free, need))
        return {'available_bytes': free, 'required_bytes': need, 'pending_bytes': dict(self.pending), 'metadata_bytes': used}
    def committed(self, name, path):
        super().committed(name, path)
        if name == 'full' and max(path.stat().st_size, path.stat().st_blocks * 512) > FULL_UPPER:
            raise RuntimeError('Actual full exceeded1,460,000,000 bytes; committed evidence remains preserved')


class ForwardLedger(old.ForwardLedger):
    def __init__(self):
        self.counts, self.handles, self.check, self.wavlm_calls = dict.fromkeys(FORWARD_LIMITS, 0), [], lambda: None, 0
    def before(self, name):
        def hook(module, inputs):
            self.check()
            if self.counts[name] >= FORWARD_LIMITS[name]: raise RuntimeError('Prospective240 forward budget exhausted: ' + name)
            self.counts[name] += 1
        return hook
    def install_wavlm(self, encoder):
        def hook(module, inputs):
            self.check()
            if self.wavlm_calls >= 64 * UPDATES: raise RuntimeError('Auxiliary convolution budget exhausted')
            self.wavlm_calls += 1
        self.handles.append(encoder.register_forward_pre_hook(hook))


def watchdog(parent=None):
    # 只检查自己的直接父进程，既不打印完整argv，也不沿用旧1500秒上限。
    directory = Path('/proc') / str(os.getppid()) if parent is None else Path(parent)
    words = [part.decode() for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
    if ((directory / 'exe').resolve() != Path('/usr/bin/timeout').resolve()
            or not all(word in words for word in ('--signal=TERM', '--kill-after=90s', '3300s'))):
        raise RuntimeError('Require own external timeout --signal=TERM --kill-after=90s 3300s')
    return {'compute_seconds': COMPUTE_SECONDS, 'cleanup_only_seconds': CLEANUP_SECONDS}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-approved', action='store_true')
    names = ('driver', 'tests', 'helper', 'helper-tests', 'anchor-helper', 'loss-helper', 'loss-helper-tests', 'protocol',
             'wavlm', 'wavlm-receipt', 'gradient-results', 'gradient-completion')
    for name in names: parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--anchor-weight', type=float, required=True)
    args = parser.parse_args(argv)
    if not args.run_approved: raise RuntimeError('Require reviewed bounded240 approval after real gradient diagnostic')
    if not math.isfinite(args.anchor_weight) or args.anchor_weight <= 0: raise RuntimeError('Require actual positive calibrated weight; no default')
    for name in names:
        value = getattr(args, name.replace('-', '_') + '_sha256')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value): raise RuntimeError('Require reviewed full SHA: ' + name)
    return args


def finite_losses(losses):
    pair.finite_losses(losses)
    if any(not math.isfinite(float(losses[key])) for key in AUX_KEYS): raise FloatingPointError('Nonfinite auxiliary/joint loss')
    if not math.isclose(losses['student_joint_total'], losses['student_total'] + losses['conversion_perceptual_weighted'], rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError('Joint loss must retain the original reconstruction total plus the actual auxiliary')


def gradient_provenance(args, parent, files, check):
    paths = {str(GRADIENT_ROOT / 'results.json'): args.gradient_results_sha256,
             str(GRADIENT_ROOT / 'completion.json'): args.gradient_completion_sha256}
    d.check_files(paths, check)
    if (GRADIENT_ROOT / 'failure.json').exists() or (GRADIENT_ROOT / 'completion_invalidated.json').exists():
        raise RuntimeError('Gradient diagnostic has failed or invalidated evidence')
    result = cross.read_json(GRADIENT_ROOT / 'results.json'); done = cross.read_json(GRADIENT_ROOT / 'completion.json')
    if (done['status'], done['cleanup_verified'], done['results_sha256']) != (
            'completed_readonly_TRAIN8_gradient_diagnostic', True, args.gradient_results_sha256):
        raise RuntimeError('Require actual completed readonly TRAIN8 gradient evidence')
    contract_path = GRADIENT_ROOT / 'run_contract.json'; paths[str(contract_path)] = done['contract_sha256']; d.check_files(paths, check)
    diagnostic = cross.read_json(contract_path)
    lengths = [row['valid_samples'] for row in diagnostic['pairs']]
    if len(lengths) != 8 or any(type(n) is not int or not 0 < n <= 32768 for n in lengths):
        raise RuntimeError('Require the actual eight valid TRAIN lengths')
    expected_counts = dict(GRADIENT_FIXED_COUNTS, wavlm=(2 if len(set(lengths)) == 1 else 16) + (2 if len(set(lengths[:4])) == 1 else 8))
    if diagnostic['forward_limits'] != expected_counts or done['counts'] != expected_counts or result['counts'] != expected_counts:
        raise RuntimeError('Actual diagnostic counts differ from its declared valid-length plan')
    if any(done['cleanup'].get(key) is not True for key in ('rng_unchanged', 'model_snapshots_unchanged', 'flags_unchanged', 'restored_rng_and_flags')):
        raise RuntimeError('Require complete readonly model/RNG/flag preservation evidence')
    if diagnostic['configuration'] != parent['configuration'] or diagnostic['files'][str(PARENT)] != PARENT_SHA:
        raise RuntimeError('Gradient calibration used another student or reconstruction configuration')
    for path, expected in ((WAVLM, args.wavlm_sha256), (TOOLS / (LOSS_HELPER + '.py'), args.loss_helper_sha256)):
        if diagnostic['files'][str(path)] != expected: raise RuntimeError('Calibration asset/loss differs from this training')
    norms = result['gradient_statistics']['norms']; numerator, denominator = norms['45MC2_conversion'], norms['LMOS']
    if any(not math.isfinite(x) or x <= 0 for x in (numerator, denominator)): raise RuntimeError('Invalid recorded gradient norms')
    expected = numerator / denominator; recorded = result['calibration']['anchor_weight']
    if not math.isclose(args.anchor_weight, expected, rel_tol=1e-12, abs_tol=0.) or recorded != expected:
        raise RuntimeError('CLI weight must be the actual predetermined norm ratio, not a score-tuned choice')
    if (result['optimizer_updates'], result['checkpoint_saves']) != (0, 0): raise RuntimeError('Calibration performed forbidden updates')
    cross.merge_hashes(files, paths)
    return {'anchor_weight': args.anchor_weight, 'rule': 'norm(45*localMC2_conversion)/norm(LMOS) on fixedTRAIN8',
        'numerator_norm': numerator, 'denominator_norm': denominator, 'diagnostic_results_sha256': args.gradient_results_sha256,
        'diagnostic_completion_sha256': args.gradient_completion_sha256, 'diagnostic_contract_file_sha256': done['contract_sha256'],
        'not_GradNorm': True, 'no_performance_claim_from_calibration': True}


def validate_parent_metadata(contract, completion):
    if training.canonical_json_sha256(contract) != PARENT_CANONICAL: raise RuntimeError('Wrong4287 parent canonical contract')
    if (completion['status'], completion['cleanup_verified'], completion['contract_sha256']) != ('completed360', True, PARENT_CANONICAL):
        raise RuntimeError('Require normal complete4287 parent')
    state = completion['state']; snapshot = state['full_snapshot']
    if (state['completed_global_step'], state['last_full_saved_global_step'], state['new_updates'], state['actual_updates_this_process'],
        state['partial_batch_may_have_updated_D'], state['restore_completed'], state['complete_loss_records']) != (START, START, 360, 360, False, True, 360):
        raise RuntimeError('Invalid actual4287 complete cursors')
    if (snapshot['path'], snapshot['sha256'], snapshot['global_step'], snapshot['committed'], snapshot['resumable'], snapshot['frozen_readonly']) != (
        str(PARENT), PARENT_SHA, START, True, True, True): raise RuntimeError('Wrong4287 complete snapshot')
    if state.get('execution_error') or state.get('exception_after_both_adam_steps') or state.get('cursor_inspection_error'):
        raise RuntimeError('Parent carries a technical failure')
    if state['monitor_global_steps'] != list(range(3927, 4288, 60)) or state['last_monitor']['global_step'] != START:
        raise RuntimeError('Parent monitoring is incomplete')
    if completion['actual_forward_counts'] != prior.FORWARD_LIMITS or state['effective_optimizer_lrs'] != LRS:
        raise RuntimeError('Parent calls or effective lr changed')
    if (contract['baseline3327_development'], contract['original_step0_guard_anchor']) != (d.cont.BASELINE3327, d.cont.S0_ANCHOR):
        raise RuntimeError('Original protection anchors changed')
    return state


def restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, contract, receipt, device):
    # 明确认4287的新kind与RNG；绝不调用只认3927的旧恢复函数。
    expected = ('MC2_FM2_independent360_full', PARENT_ARM, START, 0, True, PARENT_CANONICAL, LRS, 2, 74000000, 360, False)
    keys = ('kind', 'arm', 'global_step', 'epoch', 'resumable', 'run_contract_sha256', 'effective_optimizer_lrs',
            'mc_samples', 'mc_private_seed_base', 'actual_updates_this_process', 'historical_state_reconstruction_claimed')
    if tuple(payload[k] for k in keys) != expected: raise RuntimeError('Wrong actual full4287 header')
    if payload['configuration'] != config.as_dict() or contract['configuration'] != config.as_dict(): raise RuntimeError('Actual parent config changed')
    if (payload['score_semantics'], payload['stop_reason'], payload['execution_error'], payload['development_monitor_global_step']) != (
        'reconstruction_clean_x0_sqrt_alpha_bar', 'maximum360_updates_completed', None, START): raise RuntimeError('Invalid4287 normal endpoint semantics')
    if payload['development'] != receipt['last_monitor']['development'] or probe.tree_sha(payload['rng_state']) != PARENT_RNG_SHA:
        raise RuntimeError('Actual4287 monitor or RNG differs')
    if payload['lineage'] != {'parent_checkpoint_sha256': contract['parent_checkpoint_sha256'],
        'parent_contract_sha256': contract['parent_contract_sha256'], 'start_after_global_step': 3927}:
        raise RuntimeError('Stored historical lineage changed; no ancestor file read is required')
    for role, model, opt in (('student', student, optimizer), ('discriminator', discriminator, d_optimizer)):
        previous.prior.check_owner(opt, model)
        previous.dstate.old.validate_saved_adam(payload[role + '_optimizer'], model, START, LRS[role])
        model.load_state_dict(payload[role], strict=True)
        previous.dstate.restore_optimizer(opt, payload[role + '_optimizer'])
        model.train(); training.set_requires_grad(model, True)
        for parameter in model.parameters(): parameter.grad = None
    training.restore_rng_state(payload['rng_state'], device)
    expected_rng = payload['rng_state'] if device.type == 'cuda' else dict(payload['rng_state'], cuda_all=[])
    if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(expected_rng): raise RuntimeError('4287 RNG failed to restore')
    d.validate_adams(student, discriminator, optimizer, d_optimizer, START, LRS)
    for role, opt in (('student', optimizer), ('discriminator', d_optimizer)):
        if probe.tree_sha(opt.state_dict()) != probe.tree_sha(payload[role + '_optimizer']): raise RuntimeError('Adam restore changed stored moments or groups')
    return {'parent_checkpoint_sha256': PARENT_SHA, 'restored_global_step': START,
        'parent_restore': previous.content_state(student, discriminator, optimizer, d_optimizer),
        'configuration_changed': False, 'all_models_optimizers_rng_restored_without_change': True}


def new_state(contract_sha):
    # 独立建立本段计数；240次全部是4287之后的新更新，没有历史重放计数。
    return {'arm': ARM, 'status': 'restoring', 'phase': 'restore', 'restore_completed': False,
        'completed_global_step': START, 'new_updates': 0, 'actual_updates_this_process': 0,
        'attempted_global_step': None, 'partial_batch_may_have_updated_D': False, 'optimizer_cursors': None,
        'last_full_saved_global_step': START, 'full_snapshot': None, 'last_monitor': None,
        'contract_sha256': contract_sha, 'effective_optimizer_lrs': dict(LRS), 'parent_checkpoint_sha256': PARENT_SHA}


def complete_update(state, step, optimizer, d_optimizer, call):
    if step != state['completed_global_step'] + 1 or not START < step <= STOP: raise RuntimeError('Require exactly the next4288..4527 step')
    if any(x['uniform_step'] != step - 1 for x in old.actual_cursors(optimizer, d_optimizer).values()): raise RuntimeError('Bad entering Adam counters')
    state.update(attempted_global_step=step, partial_batch_may_have_updated_D=True, phase='D_S_update'); returned = False
    try:
        with old.observe_adam_step_returns(state, optimizer, d_optimizer): losses = call()
        returned = True
    finally:
        try:
            cursors = old.actual_cursors(optimizer, d_optimizer); state['optimizer_cursors'] = cursors
            if all(x['uniform_step'] == step for x in cursors.values()) and state['adam_step_returns'] == {'student': 1, 'discriminator': 1}:
                state.update(completed_global_step=step, new_updates=step - START, actual_updates_this_process=step - START,
                    partial_batch_may_have_updated_D=False, exception_after_both_adam_steps=not returned)
        except BaseException as error: state['cursor_inspection_error'] = repr(error)
    if state['partial_batch_may_have_updated_D']: raise RuntimeError('Missing complete D then S update proof')
    state['unpublished_returned_losses'] = {'global_step': step, 'losses': losses}
    finite_losses(losses); return losses


def future_plan(development):
    planner = resume.GlobalBatchPlan(development['retained_train_indices'])
    if len(development['retained_train_indices']) != 39335 or planner.batches_per_pass != 1229: raise RuntimeError('Original absolute plan changed')
    rows = [{'global_step': step, 'indices': planner.for_step(step), 'crop_seed': 101234 + step, 'noise_seed': 201234 + step,
             'shuffle_seed': SHUFFLE_SEED_BASE + step, 'reverse_seed': REVERSE_SEED_BASE + step}
            for step in range(START + 1, STOP + 1)]
    if any(set(row['indices']) & set(development['excluded_train_indices']) for row in rows): raise RuntimeError('Original52 exclusions leaked into training')
    return rows


def window120(rows):
    if len(rows) != 120 or rows[0]['global_step'] not in (4288, 4408): raise RuntimeError('Only two complete120-step windows')
    first = rows[0]['global_step']
    if [r['global_step'] for r in rows] != list(range(first, first + 120)): raise RuntimeError('Nonconsecutive raw window')
    for row in rows: finite_losses(row)
    return {'first_global_step': first, 'global_step': rows[-1]['global_step'], 'count': 120, 'phase': 'new_updates_after4287',
        'means': {k: math.fsum(r[k] for r in rows) / 120 for k in tuple(pair.LOSS_KEYS) + AUX_KEYS},
        'MC2_dist_means': {k: math.fsum(r['mc'][k] for r in rows) / 120 for k in ('dist1', 'dist2')},
        'first_last_difference': {k: rows[-1][k] - rows[0][k] for k in tuple(pair.LOSS_KEYS) + AUX_KEYS}, 'used_as_stop_gate': False}


def loss_trend(rows):
    if len(rows) != UPDATES: return {'available': False, 'completed_records': len(rows), 'reason': 'requires_all240_steps'}
    windows = [window120(rows[i:i + 120]) for i in (0, 120)]
    return {'available': True, 'both_windows': windows,
        'second_minus_first_window': {k: windows[1]['means'][k] - windows[0]['means'][k] for k in tuple(pair.LOSS_KEYS) + AUX_KEYS},
        'original_reconstruction_total_and_new_joint_total_reported_separately': True,
        'used_as_stop_gate': False, 'all240_updates_are_after_parent4287': True}


def run_updates(state, future, step_call, evaluate, publish, check, initial):
    if [r['global_step'] for r in future] != list(range(START + 1, STOP + 1)) or initial['global_step'] != START: raise RuntimeError('Wrong prospective plan')
    rows, windows, monitors = [], [], [initial]; initial_guards = old.guard_reasons([initial], d.cont.BASELINE3327)
    if initial_guards: return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': initial_guards[0]}
    reason = 'maximum240_updates_completed'
    for planned in future:
        check(); row = step_call(planned); state['unpublished_step_record'] = row
        publish('step_losses.jsonl', row); state.pop('unpublished_step_record', None); rows.append(row); state['complete_loss_records'] = len(rows)
        if len(rows) % 120 == 0:
            window = window120(rows[-120:]); windows.append(window); publish('loss_windows.jsonl', window)
        if len(rows) % 60 == 0:
            entry = evaluate(planned['global_step']); state['last_monitor'] = entry; monitors.append(entry); publish('monitor.jsonl', entry)
            guards = old.guard_reasons(monitors[1:], d.cont.BASELINE3327); state['guard_reasons'] = guards
            publish('guard_checks.jsonl', {'global_step': entry['global_step'], 'stop_reasons': guards,
                'distinct_global_steps_used': [r['global_step'] for r in monitors[1:]], 'historical_points_pooled': False})
            if guards: reason = guards[0]; break
        publish('status.json', state, False)
    return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': reason}


def prepare_context(args, check):
    # 下列是真正还要读取的资产，不沿用旧FIXED_PINS整张表，避免要求已归档3927。
    files = {str(PARENT): PARENT_SHA, str(PARENT_RUN / 'run_contract.json'): PARENT_CONTRACT_SHA,
        str(PARENT_RUN / PARENT_ARM / 'completion.json'): PARENT_COMPLETION_SHA,
        str(PRIOR_FILE): PRIOR_SHA, str(Path(__file__).resolve()): args.driver_sha256,
        str(TOOLS / 'test_run_acdd_mc2_perceptual240_from4287_v1.py'): args.tests_sha256,
        str(TOOLS / (HELPER + '.py')): args.helper_sha256,
        str(TOOLS / ('test_' + HELPER + '.py')): args.helper_tests_sha256, str(PROTOCOL): args.protocol_sha256,
        str(TOOLS / (ANCHOR_HELPER + '.py')): args.anchor_helper_sha256,
        str(TOOLS / (LOSS_HELPER + '.py')): args.loss_helper_sha256,
        str(TOOLS / ('test_' + LOSS_HELPER + '.py')): args.loss_helper_tests_sha256,
        str(WAVLM): args.wavlm_sha256, str(WAVLM_RECEIPT): args.wavlm_receipt_sha256,
        str(TOOLS / 'acdd_mc2_train_batch_v1.py'): 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16'}
    for path in (cross.INPUT_FILE, cross.TARGET_FILE, pair.PARENT / 'latest_full.pt', previous.CROSS_IDENTITY,
                 base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json'):
        files[str(path)] = d.FIXED_PINS[str(path)]
    files.update(d.NEW_SOURCE_PINS); d.check_files(files, check); d.validate_supplemental_source_pins()
    mc = importlib.import_module(HELPER)
    loss_module = importlib.import_module(LOSS_HELPER)
    if (mc.START, mc.STOP, mc._anchor._core.MC2_SEED, mc.ANCHOR_SHA) != (START, STOP, 74000000, args.anchor_helper_sha256):
        raise RuntimeError('Wrong4288..4527 MC2 augmentation or unchanged-core identity')
    parent = cross.read_json(PARENT_RUN / 'run_contract.json')
    receipt = validate_parent_metadata(parent, cross.read_json(PARENT_RUN / PARENT_ARM / 'completion.json'))
    provenance = gradient_provenance(args, parent, files, check)
    asset_receipt = cross.read_json(WAVLM_RECEIPT)
    if (asset_receipt['status'], asset_receipt['output']['path'], asset_receipt['output']['sha256'],
            asset_receipt['tensor_count'], asset_receipt['parameter_count'], asset_receipt['source_sha256']) != (
            'completed_verified_official_conv_extraction', str(WAVLM), args.wavlm_sha256, 21, 4206592,
            'fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d'):
        raise RuntimeError('Require verified official21-tensor WavLM asset provenance')
    # 只绑定这次小资产的实际来源记录，不递归恢复任何旧训练checkpoint或旧音频。
    if asset_receipt['source_config_sha256'] != 'a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d':
        raise RuntimeError('Official WavLM source configuration changed')
    cross.merge_hashes(files, asset_receipt['source_pins']); d.check_files(asset_receipt['source_pins'], check)
    config = base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0)
    if parent['configuration'] != config.as_dict() or (config.lambda_fm, config.lambda_dist) != (2., 45.): raise RuntimeError('UnchangedFM2/dist45 required')
    development, dependencies = resume.load_development(DEV / 'development_contract.json')
    if (parent['training_indices'], parent['excluded_train_indices']) != (development['retained_train_indices'], development['excluded_train_indices']):
        raise RuntimeError('Original training indices and52 exclusions changed')
    cross.merge_hashes(files, dependencies)
    # 只选择实际导入的.py；父runtime_files里的模型/旧特征不因此成为本段依赖。
    cross.merge_hashes(files, d.imported_source_files(parent['runtime_files'], files))
    for relative, digest in training.collect_source_fingerprints(ROOT).items():
        path = str(ROOT / relative) if not Path(relative).is_absolute() else relative
        if parent['runtime_files'].get(path) != digest: raise RuntimeError('Production source changed')
        files[path] = digest
    for asset in (base.TEACHER, base.VOCODER_DIR / 'generator_step_2500000.pt', ROOT / 'data/stats/mel_mean.npy', ROOT / 'data/stats/mel_std.npy'):
        files[str(asset)] = parent['runtime_files'][str(asset)]
    plan = parent['monitor64_plan']
    if plan != cross.read_json(previous.CROSS_IDENTITY)['plan']: raise RuntimeError('Original64 monitor plan changed')
    dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
    if len(dataset) != 39387: raise RuntimeError('Dataset indexing changed')
    future = future_plan(development); future_files = previous.register_future_files(dataset, future, files, check)
    monitor_files = previous.register_future_files(dataset, [{'indices': plan['selected_indices']}], files, check)
    if len(monitor_files) != 192 or monitor_files != parent['fixed64_feature_sha256']: raise RuntimeError('Original fixed64 features changed')
    d.check_files(files, check)
    return mc, loss_module, provenance, parent, receipt, config, development, dataset, future, future_files, monitor_files, files


def auxiliary_configuration(args, provenance):
    return {'method': 'FINALLY_inspired_conv_feature_plus_STFT_engineering_v1', 'anchor_weight': args.anchor_weight,
        'gradient_calibration': provenance, 'teacher_sampler': 'unchanged_ancestral30_same_actual_q950_crossspeaker',
        'waveform_source_rate': 22050, 'feature_rate': 16000, 'feature_layer': 'seventh_conv_LayerNorm_GELU_512D',
        'wave_normalization_feature_branch_only': 'population_variance_plus1e-7', 'feature_mse_weight': 100.,
        'resampling': {'lowpass_filter_width': 6, 'rolloff': .99, 'resampling_method': 'sinc_interpolation', 'beta': None},
        'STFT': {'n_fft': 512, 'win_length': 512, 'hop_length': 128, 'window': 'periodic_hann', 'center': False,
            'normalized': False, 'onesided': True, 'raw_magnitude': True, 'reduction': 'each_utterance_mean_then_batch_mean'},
        'wavlm_asset_sha256': args.wavlm_sha256, 'wavlm_receipt_sha256': args.wavlm_receipt_sha256,
        'loss_module_sha256': args.loss_helper_sha256, 'dynamic_update_of_weight': False,
        'supervised_metrics': [], 'raw_mel_L1_optimized': False, 'conversion_discriminator': False,
        'not_claimed_original_FastVoiceGrad_or_exact_FINALLY': True}


def make_contract(args, provenance, parent, receipt, config, development, future, future_files, monitor_files, files, environment):
    return {'kind': 'MC2_FINALLY_inspired_perceptual240_from4287', 'protocol_sha256': args.protocol_sha256,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS, 'runtime': environment,
        'auxiliary_configuration': auxiliary_configuration(args, provenance),
        'parent_checkpoint': str(PARENT), 'parent_checkpoint_sha256': PARENT_SHA,
        'parent_contract_sha256': PARENT_CANONICAL, 'parent_contract_file_sha256': PARENT_CONTRACT_SHA,
        'parent_completion_file_sha256': PARENT_COMPLETION_SHA, 'parent_rng_sha256': PARENT_RNG_SHA,
        'start_after_global_step': START, 'stop_global_step': STOP, 'maximum_actual_updates': UPDATES,
        'no_old_update_reexecution': True, 'historical3927_is_not_a_runtime_checkpoint_dependency': True,
        'mc_samples': 2, 'mc_private_seed_base': 74000000, 'paper_reference_coefficients': {'lambda_fm': 2., 'lambda_dist': 45.},
        'disclosed_extensions': {'student_optimizer_lr': .00005, 'MC_samples': 2,
            'score_target_interpretation': 'clean_x0_sqrt_alpha_bar', 'not_claimed_exact_paper_reproduction': True},
        'future_batches': future, 'future_training_feature_sha256': future_files, 'fixed64_feature_sha256': monitor_files,
        'training_indices': development['retained_train_indices'], 'excluded_train_indices': development['excluded_train_indices'],
        'input_identities': parent['input_identities'], 'monitor64_plan': parent['monitor64_plan'],
        'baseline4287_monitor': receipt['last_monitor'], 'baseline3327_development': d.cont.BASELINE3327,
        'original_step0_guard_anchor': d.cont.S0_ANCHOR, 'fixed_frozen_models_sha256': parent['fixed_frozen_models_sha256'],
        'monitor_updates': list(MONITOR_UPDATES), 'loss_windows': [[4288, 4407], [4408, 4527]],
        'loss_keys': list(pair.LOSS_KEYS) + list(AUX_KEYS), 'primary_trend': 'second120_minus_first120; original_total_and_joint_separate; no_scalar_gate',
        'historical_computed_monitor_comparison': 'descriptive_only', 'runtime_files': files, 'forward_limits': FORWARD_LIMITS,
        'wavlm_conv_forward_maximum': 64 * UPDATES,
        'storage': {'initial_required_bytes': INITIAL_REQUIRED_BYTES, 'full_upper_bytes': FULL_UPPER,
            'write_overhead': WRITE_OVERHEAD, 'system_reserve': SYSTEM_RESERVE, 'directory_reserve': DIR_RESERVE,
            'metadata_limit': META_LIMIT, 'later_DEV90_bytes': EVAL_BYTES,
            'allocation_specific_reserve_reduced_to1GiB_no_changes_to_old_protocols': True},
        'evaluation': EVALUATION, 'formal_gate_only': {'UTMOS_min': 3.9812097230664008, 'CER_percent_max': 2.109892438933927,
            'DNSMOS_P808_strictly_greater_than': 3.77, 'SVA_gate': False, 'OVRL_gate': False},
        'compute_seconds': COMPUTE_SECONDS, 'cleanup_only_seconds': 90, 'original_flags': cross.ORIGINAL_FLAGS,
        'no_automatic_evaluation_training_retry_archive_or_deletion': True, 'formal3327_remains_stopped': True}


def save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock):
    if (not state['restore_completed'] or state['partial_batch_may_have_updated_D'] or state.get('execution_error')
            or state.get('exception_after_both_adam_steps') or state.get('cursor_inspection_error')): raise RuntimeError('Never save technical or partial update')
    step = state['completed_global_step']; count = step - START; last = state['last_monitor']
    if step not in range(4347, STOP + 1, 60) or last['global_step'] != step or state['arm'] != ARM: raise RuntimeError('Require actual monitored new endpoint, never zero steps')
    if config.as_dict() != base.replace(base.PAPER_STUDENT_CONFIG, num_workers=0).as_dict(): raise RuntimeError('Save real unchangedFM2 configuration')
    if (state['actual_updates_this_process'], state['new_updates'], state['complete_loss_records']) != (count, count, count): raise RuntimeError('Actual continuation counts differ')
    if state['monitor_global_steps'] != list(range(START, step + 1, 60)): raise RuntimeError('Actual monitor sequence is incomplete')
    protective = ('original_S0_extreme_guard', 'two_consecutive_DEV32_above3327_by5percent')
    if state['reason'] not in protective and not (step == STOP and state['reason'] == 'maximum240_updates_completed'): raise RuntimeError('Require normal4527 or actual protection stop')
    d.validate_adams(student, discriminator, optimizer, d_optimizer, step, LRS); state['optimizer_cursors'] = old.actual_cursors(optimizer, d_optimizer)
    payload = {'kind': 'MC2_FINALLY_inspired_perceptual240_from4287_full', 'arm': ARM, 'global_step': step, 'epoch': 0, 'resumable': True,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'auxiliary_configuration': state['auxiliary_configuration'],
        'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'run_contract_sha256': state['contract_sha256'],
        'development': last['development'], 'development_monitor_global_step': step,
        'lineage': {'parent_checkpoint_sha256': PARENT_SHA, 'parent_contract_sha256': PARENT_CANONICAL, 'start_after_global_step': START},
        'actual_updates_this_process': count, 'no_old_update_reexecution': True, 'stop_reason': state['reason'], 'execution_error': None,
        'actual_optimizer_cursors': state['optimizer_cursors'], 'student': student.state_dict(), 'discriminator': discriminator.state_dict(),
        'student_optimizer': optimizer.state_dict(), 'discriminator_optimizer': d_optimizer.state_dict(), 'rng_state': training.capture_rng_state()}
    path = budget.output / ARM / 'final_full.pt'; clock.active(); budget.require()
    def committed():
        state['full_snapshot'] = {'path': str(path), 'global_step': step, 'sha256': None, 'committed': True, 'resumable': True}
        state['last_full_saved_global_step'] = step; budget.committed('full', path)
    old.atomic_checkpoint_commit(payload, path, committed)
    state['full_snapshot']['sha256'] = base.sha256_file(path); path.chmod(0o444); state['full_snapshot']['frozen_readonly'] = True
    clock.active(); budget.write(path.parent / 'checkpoint_receipt.json', state)


def main(argv=None):
    started = time.monotonic(); clock = budget = log = None; locks = []; owned = False
    state, completion, stage = {}, None, OUTPUT / ARM; ledger = ForwardLedger(); saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        args = arguments(argv); clock = old.StopClock(COMPUTE_SECONDS, started); clock.install(); watchdog()
        training.assert_rented_server(ROOT); cross.regular(DEV, directory=True); locks = training.acquire_runtime_locks(ROOT)
        if len(locks) != 3: raise RuntimeError('Require three workflow locks')
        lock_path = ROOT / 'exp/locks/text_unseen_fastvoicegrad_student_inference.lock'
        if lock_path.is_symlink(): raise RuntimeError('Inference lock symlink')
        handle = lock_path.open('a+')
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException: handle.close(); raise
        locks.append(handle)
        if OUTPUT.exists() or OUTPUT.is_symlink(): raise RuntimeError('New continuation output required; never retry in place')
        budget = Budget(OUTPUT); disk = budget.require(); OUTPUT.mkdir(); log = d.StageLog(stage, budget)
        sys.stdout = sys.stderr = log; stage.mkdir(); owned = True
        def check(): clock.active(); budget.require()
        ledger.check = check; environment = runtime()
        if training.git_head(ROOT) != cross.HEAD or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Require unchanged clean frozen production source')
        mc, loss_module, provenance, parent, receipt, config, development, dataset, future, future_files, monitor_files, files = prepare_context(args, check)
        contract = make_contract(args, provenance, parent, receipt, config, development, future, future_files, monitor_files, files, environment)
        contract_sha = training.canonical_json_sha256(contract); state = new_state(contract_sha)
        state['auxiliary_configuration'] = contract['auxiliary_configuration']; budget.write(OUTPUT / 'run_contract.json', contract)
        gpu = cross.gpu_gate(); device = torch.device('cuda'); check(); diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
        inputs, batches, target, input_identity = d.prepare_inputs(parent, dataset, diffusion, device)
        check(); payload = resume.load_torch_checkpoint(PARENT)
        teacher, student, unused = base.load_teacher_and_student(base.TEACHER, config, device); del unused
        vocoder, vconfig, unused = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
            base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device); del unused
        discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device); fixed_d = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed = resume.load_torch_checkpoint(pair.PARENT / 'latest_full.pt')
        if fixed['global_step'] != 3107 or fixed['run_contract_sha256'] != pair.PARENT_CONTRACT: raise RuntimeError('Wrong fixedD3107')
        fixed_d.load_state_dict(fixed['discriminator'], strict=True); del fixed; fixed_d.eval(); training.set_requires_grad(fixed_d, False)
        mean, std = base.load_mel_statistics(ROOT / 'data', device)
        # 仅小型21参数资产；不加载WavLM Transformer，也不加入优化器。
        encoder_state = resume.load_torch_checkpoint(WAVLM)
        encoder = loss_module.FrozenWavLMConvEncoder(encoder_state).to(device); del encoder_state
        perceptual_loss = loss_module.WavLMConvPerceptualLoss(encoder).to(device)
        d.imported_source_files(files, files)
        frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}; frozen_sha = {name: pair.model_sha(model) for name, model in frozen.items()}
        if frozen_sha != parent['fixed_frozen_models_sha256']: raise RuntimeError('Frozen assets changed')
        frozen['wavlm_conv'] = encoder; frozen_sha['wavlm_conv'] = pair.model_sha(encoder)
        models = dict(frozen, student=student, active_D=discriminator, diffusion=diffusion)
        optimizer = torch.optim.Adam(student.parameters(), lr=LRS['student'], betas=(.5, .9)); d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LRS['discriminator'], betas=(.5, .9))
        restored = restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, parent, receipt, device); del payload
        state.update(restore_completed=True, restore_identity=restored, optimizer_cursors=old.actual_cursors(optimizer, d_optimizer), status='running', initial_disk=disk, initial_gpu=gpu)
        ledger.install(student, teacher, vocoder, discriminator, fixed_d); ledger.install_wavlm(encoder)
        retained, excluded = frozenset(development['retained_train_indices']), frozenset(development['excluded_train_indices'])
        def unchanged():
            pair.assert_frozen(frozen, frozen_sha)
            encoder.assert_frozen()
            if perceptual_loss.training: raise RuntimeError('Frozen perceptual wrapper mode changed')
            if input_identity() != parent['input_identities'] or cross.flags() != cross.ORIGINAL_FLAGS: raise RuntimeError('Fixed inputs or flags changed')
        def evaluate(step):
            if step - START not in MONITOR_UPDATES: raise RuntimeError('Undeclared monitoring cadence')
            check(); before = dict(ledger.counts)
            with previous.monitoring_state(models, device):
                entry = pair.monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, step)
                entry['updates'] = entry['trial_updates'] = step - START
                entry['uniform64'] = previous.uniform64_monitor(student, teacher, diffusion, batches, parent['monitor64_plan'], config, device, check)
            if ledger.delta(before) != MONITOR_COUNTS: raise RuntimeError('Wrong unchanged monitor calls')
            entry['actual_forward_counts'] = ledger.delta(before); entry['observation_role'] = 'current_continuation_after4287'
            entry['historical_computed_comparison'] = 'descriptive_only_not_a_gate'; unchanged(); return entry
        initial = evaluate(START); state['last_monitor'] = initial; budget.write(stage / 'monitor.jsonl', initial, append=True)
        def step_call(planned):
            step = planned['global_step']; batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
            training.seed_everything(planned['noise_seed']); identity = mc._anchor._core.original_input_identity(batch, device, config)
            mc_receipt, anchor_receipt = {}, {}; check(); before = dict(ledger.counts); before_wavlm = ledger.wavlm_calls
            def call():
                with deadline_steps(optimizer, d_optimizer, check):
                    return mc.train_batch_mc2_perceptual_anchor(batch, teacher, student, vocoder, discriminator, diffusion,
                        optimizer, d_optimizer, mean, std, device, config, global_step=step, mc_receipt=mc_receipt,
                        anchor_weight=args.anchor_weight, source_indices=planned['indices'], retained_train_indices=retained,
                        excluded_train_indices=excluded, shuffle_seed=planned['shuffle_seed'], reverse_seed=planned['reverse_seed'],
                        anchor_receipt=anchor_receipt, perceptual_loss=perceptual_loss)
            losses = complete_update(state, step, optimizer, d_optimizer, call)
            row = dict(losses, global_step=step, updates=step - START, indices=planned['indices'], crop_seed=planned['crop_seed'], noise_seed=planned['noise_seed'],
                shuffle_seed=planned['shuffle_seed'], reverse_seed=planned['reverse_seed'], auxiliary=anchor_receipt,
                input_identity=identity, mc=mc_receipt, actual_optimizer_cursors=state['optimizer_cursors'], actual_forward_counts=ledger.delta(before),
                wavlm_conv_forward_calls=ledger.wavlm_calls - before_wavlm, observation_role='new_perceptual_update_after4287')
            state['unpublished_step_record'] = row; state.pop('unpublished_returned_losses', None)
            for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
                if mc_receipt[key] != identity[key]: raise RuntimeError('Original RNG preview mismatch')
            if row['actual_forward_counts'] != TRAIN_COUNTS: raise RuntimeError('Wrong complete train calls')
            expected_wavlm = 2 if len(set(anchor_receipt['valid_lengths'])) == 1 else 64
            if row['wavlm_conv_forward_calls'] != expected_wavlm: raise RuntimeError('Unexpected masked perceptual convolution calls')
            if (anchor_receipt['teacher_forward_calls'], anchor_receipt['same_speaker_count'], anchor_receipt['anchor_weight']) != (30, 0, args.anchor_weight):
                raise RuntimeError('Declared conversion target or fixed auxiliary coefficient changed')
            previous.adam_counters(optimizer, d_optimizer, step)
            if [optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr']] != list(LRS.values()): raise RuntimeError('Learning-rate drift')
            return row
        result = run_updates(state, future, step_call, evaluate, lambda name, value, append=True: budget.write(stage / name, value, append), check, initial)
        state.update(reason=result['reason'], loss_windows=result['windows'], monitor_global_steps=[r['global_step'] for r in result['monitors']],
            status='completed240' if result['reason'] == 'maximum240_updates_completed' else 'protective_stop', phase='saving')
        budget.write(stage / 'loss_trend.json', loss_trend(result['rows'])); unchanged()
        if state['new_updates'] > 0: save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock)
        else: state['no_checkpoint_reason'] = 'initial_guard_before_any_new_update'
        d.imported_source_files(files, files); d.check_files(files, check); unchanged()
        if runtime() != environment: raise RuntimeError('Measured runtime changed during this continuation')
        if training.canonical_json_sha256(cross.read_json(OUTPUT / 'run_contract.json')) != contract_sha: raise RuntimeError('New contract changed')
        expected_counts = {k: state['new_updates'] * TRAIN_COUNTS[k] + len(result['monitors']) * MONITOR_COUNTS[k] for k in FORWARD_LIMITS}
        if ledger.counts != expected_counts: raise RuntimeError('Actual complete or protective endpoint calls differ')
        if ledger.wavlm_calls != sum(row['wavlm_conv_forward_calls'] for row in result['rows']): raise RuntimeError('Unaccounted WavLM calls')
        completion = {'status': state['status'], 'cleanup_verified': True, 'state': dict(state), 'contract_sha256': contract_sha,
            'actual_forward_counts': dict(ledger.counts), 'elapsed_seconds': time.monotonic() - started,
            'actual_wavlm_conv_forward_calls': ledger.wavlm_calls,
            'actual_updates_this_process': state['new_updates'], 'no_old_update_reexecution': True,
            'no_evaluation_or_archive_executed': True, 'automatic_next_stage_authorized': False}
        budget.write(stage / 'trial_completion.json', dict(completion, cleanup_verified=False, status='pending_cleanup')); check()
    except BaseException as error:
        if clock: clock.request('technical_failure')
        if budget: budget.cancel_future_on_fatal_abort(repr(error))
        state.update(status='technical_failure', execution_error=repr(error))
        if owned:
            try: budget.write(stage / 'failure.json', failure_record({'status': 'technical_failure', 'state': state, 'actual_forward_counts': ledger.counts, 'error': repr(error)}))
            except BaseException as failed: saved_stderr.write('Failure receipt unavailable: %r; original=%r\n' % (failed, error))
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = old.cleanup_resources(ledger, clock, log, locks)
        if errors and budget: budget.cancel_future_on_fatal_abort(repr(errors))
        if errors and owned:
            try: budget.write(stage / 'cleanup_failure.json', failure_record({'status': 'technical_failure', 'errors': errors, 'state': state}))
            except BaseException as error: saved_stderr.write('Cleanup receipt unavailable: %r\n' % error)
        if errors and original_error is None: raise RuntimeError('Continuation cleanup failed: ' + repr(errors))
        if original_error is None and not errors and completion is not None:
            try: clock.active(); budget.publish_success(stage / 'completion.json', completion)
            except BaseException as error:
                budget.cancel_future_on_fatal_abort(repr(error))
                try: budget.write(stage / 'failure.json', failure_record({'status': 'technical_failure', 'state': state, 'error': repr(error), 'phase': 'terminal_publication'}))
                except BaseException as failed: saved_stderr.write('Terminal failure receipt unavailable: %r\n' % failed)
                raise


if __name__ == '__main__': main()
