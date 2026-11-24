"""从真实MC2完整3447状态继续至多480步；默认拒绝启动，不评分、不自动长训。"""

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Run only on the registered Matpool rental')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
OLD_NAME = 'run_acdd_mc2_paired120_v1.py'
OLD_SHA = '5607ad36dfc3e510021316a66416c43ad8bb8d9f718f1bd8bf1084ec2bf6fd5d'
if hashlib.sha256((TOOLS / OLD_NAME).read_bytes()).hexdigest() != OLD_SHA:
    raise RuntimeError('Frozen paired120 driver changed')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TOOLS))
import torch
import run_acdd_mc2_paired120_v1 as old

previous, cross, pair, base = old.previous, old.cross, old.pair, old.base
training, resume, probe = old.training, old.resume, old.probe
DEV, START, STOP, UPDATES = old.DEV, 3447, 3927, 480
OUTPUT = DEV / 'acdd_mc2_continuation480_from3447_v1'
PARENT_RUN = old.OUTPUT
PARENT = PARENT_RUN / 'candidate_mc2/final_full.pt'
PARENT_SHA = 'd108d1793a2473bb42f6b1e556c67b6339a04da8b9c8b39f0a7905bf5a26d4f6'
PARENT_CANONICAL = '3be5667d5ab01c3def83536bc7e86e48433a08f9557a0f82d010955b4dbe51ad'
PARENT_CONTRACT_SHA = 'd939982c667e3b48228fe91b2216f93f572643f56837908a50382a0ecc069813'
PARENT_COMPLETION_SHA = '9ec17d2604adb63b692b9d03a7754af8c136ddbc1e6d3a08b008f09d334979ca'
PARENT_RNG_SHA = '67da3f601a11573b583e2118af9289fe6942ce92b5a4cbba8f19ed70dbeac5c7'
PROTOCOL = DEV / 'ACDD3447_MC2_CONTINUATION480_PROTOCOL_CN.md'
HELPER_NAME = 'acdd_mc2_train_batch_3448_3927_v1.py'
HELPER_TEST_NAME = 'test_acdd_mc2_train_batch_3448_3927_v1.py'
HELPER_TEST_SHA = 'b839ed5cd9179298077187e16c9af9259a7b3028a44695c9cd0fb1147d058ace'
LRS, S0_ANCHOR = dict(old.LRS), dict(old.S0_ANCHOR)
BASELINE3327 = {'mel_l1': .19254307448863983}
FULL_BYTES, META_LIMIT = 1424633856, 32 * 1024**2
INITIAL_REQUIRED_BYTES = 4815728640
PAIR_UPDATES = tuple(range(0, UPDATES + 1, 30))
UNIFORM64_UPDATES = tuple(range(0, UPDATES + 1, 60))
WINDOW_STARTS = (3448, 3568, 3688, 3808)
FORWARD_LIMITS = {'teacher': 1157, 'student': 1012, 'vocoder': 1474,
                  'active_D_branches': 15360, 'fixed_D_branches': 272}
BoundaryStop, StopClock = old.BoundaryStop, old.StopClock
actual_cursors, assert_saveable = old.actual_cursors, old.assert_saveable
guard_reasons = old.guard_reasons  # 此函数没有3327..3447范围上界，只用原S0和传入3327基线。
cleanup_resources = old.cleanup_resources
publish_completed_after_cleanup = old.publish_completed_after_cleanup


class Budget(old.Budget):
    """只预留一份full；32MiB包含全部480步原始记录，不以丢日志节省容量。"""
    def __init__(self, output):
        self.output = Path(output)
        self.pending = {'full': FULL_BYTES + previous.MODEL_SLACK}
        self.cancelled = {}

    def require(self, extra_metadata=0):
        used = self.metadata_used()
        if used + extra_metadata > META_LIMIT:
            raise pair.InsufficientDisk('Complete metadata/log exceeds32MiB; do not omit observations')
        need = previous.RESERVE + previous.DIR_RESERVE + sum(self.pending.values()) + META_LIMIT - used
        free = pair.available_bytes(self.output.parent)
        if free < need:
            raise pair.InsufficientDisk('available=%s required=%s pending=%s' % (free, need, self.pending))
        return {'available_bytes': free, 'required_bytes': need, 'pending_bytes': dict(self.pending), 'metadata_bytes': used}


class ForwardLedger(old.ForwardLedger):
    def __init__(self):
        self.counts = {name: 0 for name in FORWARD_LIMITS}; self.handles = []

    def before(self, name):
        def hook(module, arguments):
            if self.counts[name] >= FORWARD_LIMITS[name]:
                raise RuntimeError('Predeclared continuation forward budget exhausted: ' + name)
            self.counts[name] += 1
        return hook


def new_state(contract_sha):
    state = old.new_arm_state('candidate_mc2', contract_sha)
    state.update(completed_global_step=START, last_full_saved_global_step=START,
                 parent_checkpoint_sha256=PARENT_SHA, continuation_from_actual3447=True)
    return state


def complete_update(state, step, optimizer, d_optimizer, call):
    """沿用真实Adam正常返回证明；只将新范围绑定到3448..3927。"""
    if step != state['completed_global_step'] + 1 or not START < step <= STOP:
        raise RuntimeError('Require exactly the next declared3448..3927 complete update')
    before = actual_cursors(optimizer, d_optimizer)
    if any(item['uniform_step'] != step - 1 for item in before.values()):
        raise RuntimeError('Both actual Adam cursors must match the previous complete step')
    state.update(attempted_global_step=step, partial_batch_may_have_updated_D=True, phase='D_S_update')
    returned = False
    try:
        with old.observe_adam_step_returns(state, optimizer, d_optimizer):
            losses = call()
        returned = True
    finally:
        try:
            cursor = actual_cursors(optimizer, d_optimizer); state['optimizer_cursors'] = cursor
            if (all(item['uniform_step'] == step for item in cursor.values())
                    and state['adam_step_returns'] == {'student': 1, 'discriminator': 1}):
                state.update(completed_global_step=step, new_updates=step - START,
                             partial_batch_may_have_updated_D=False, exception_after_both_adam_steps=not returned)
        except BaseException as error:
            state['cursor_inspection_error'] = repr(error)
    if state['partial_batch_may_have_updated_D']:
        raise RuntimeError('Helper returned without two complete uniform Adam updates')
    pair.finite_losses(losses)
    return losses


def window120(rows):
    if len(rows) != 120 or rows[0]['global_step'] not in WINDOW_STARTS:
        raise RuntimeError('Only the four preregistered complete120-step windows are allowed')
    if [row['global_step'] for row in rows] != list(range(rows[0]['global_step'], rows[0]['global_step'] + 120)):
        raise RuntimeError('Window must contain120 consecutive absolute steps')
    for row in rows: pair.finite_losses(row)
    return {'first_global_step': rows[0]['global_step'], 'global_step': rows[-1]['global_step'], 'count': 120,
            'means': {key: math.fsum(row[key] for row in rows) / 120 for key in pair.LOSS_KEYS},
            'first_last_difference': {key: rows[-1][key] - rows[0][key] for key in pair.LOSS_KEYS},
            'first': {key: rows[0][key] for key in pair.LOSS_KEYS},
            'last': {key: rows[-1][key] for key in pair.LOSS_KEYS}, 'used_as_stop_gate': False}


def primary_loss_trend(rows):
    # 只有完整480步才能比较前后240步；提前停止时保留原始行，绝不缩短主要窗口。
    if len(rows) != UPDATES:
        return {'available': False, 'completed_loss_records': len(rows), 'reason': 'requires_all480_registered_steps'}
    if [row['global_step'] for row in rows] != list(range(START + 1, STOP + 1)):
        raise RuntimeError('Primary trend requires the unchanged absolute480-step sequence')
    first = {key: math.fsum(row[key] for row in rows[:240]) / 240 for key in pair.LOSS_KEYS}
    last = {key: math.fsum(row[key] for row in rows[240:]) / 240 for key in pair.LOSS_KEYS}
    return {'available': True, 'first_global_steps': [3448, 3687], 'last_global_steps': [3688, 3927],
            'first240_means': first, 'last240_means': last,
            'last_minus_first': {key: last[key] - first[key] for key in pair.LOSS_KEYS},
            'student_total_relative_change': (last['student_total'] / first['student_total'] - 1.) if first['student_total'] else None,
            'used_as_stop_gate': False}


def future_plan(development):
    plan = resume.GlobalBatchPlan(development['retained_train_indices'])
    if len(development['retained_train_indices']) != 39335 or plan.batches_per_pass != 1229:
        raise RuntimeError('The original global batch plan changed')
    rows = [{'global_step': step, 'indices': plan.for_step(step), 'crop_seed': 101234 + step,
             'noise_seed': 201234 + step} for step in range(START + 1, STOP + 1)]
    if any(set(row['indices']) & set(development['excluded_train_indices']) for row in rows):
        raise RuntimeError('Training intersects the52 development exclusions')
    if (rows[0]['crop_seed'], rows[0]['noise_seed']) != (104682, 204682):
        raise RuntimeError('Do not reset absolute crop/noise seeds at continuation')
    return rows


def validate_parent_metadata(contract, completion):
    if training.canonical_json_sha256(contract) != PARENT_CANONICAL:
        raise RuntimeError('Require the original paired120 parent contract')
    if (completion['status'], completion['cleanup_verified'], completion['contract_sha256']) != (
            'both120_completed', True, PARENT_CANONICAL):
        raise RuntimeError('Parent paired120 completion or cleanup is invalid')
    receipt = completion['arms']['candidate_mc2']
    if (receipt['status'], receipt['completed_global_step'], receipt['last_full_saved_global_step'],
            receipt['partial_batch_may_have_updated_D'], receipt['restore_completed']) != ('completed120', START, START, False, True):
        raise RuntimeError('Require a complete normally terminated candidate3447')
    if (receipt['full_snapshot']['path'], receipt['full_snapshot']['sha256'], receipt['full_snapshot']['resumable']) != (
            str(PARENT), PARENT_SHA, True) or receipt['student_snapshot'] is not None:
        raise RuntimeError('The single full parent identity is invalid')
    if receipt['last_monitor']['global_step'] != START or 'uniform64' not in receipt['last_monitor']:
        raise RuntimeError('Require original actual3447 complete fixed monitors')
    if contract['baseline3327_development']['mel_l1'] != BASELINE3327['mel_l1']:
        raise RuntimeError('Never reset the original3327 protection baseline')
    return receipt


def restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, contract, receipt, device):
    """验证3447的完整状态；不调用只允许3327的旧parent validator。"""
    required = (payload['global_step'], payload['run_contract_sha256'], payload['effective_optimizer_lrs'],
                payload['mc_samples'], payload['mc_private_seed_base'], payload['resumable'], payload['epoch'])
    if required != (START, PARENT_CANONICAL, LRS, 2, 74000000, True, 0):
        raise RuntimeError('Invalid actual3447 MC2 full checkpoint header')
    if payload['configuration'] != config.as_dict() or contract['configuration'] != config.as_dict():
        raise RuntimeError('Parent mathematical configuration changed')
    if payload['score_semantics'] != 'reconstruction_clean_x0_sqrt_alpha_bar':
        raise RuntimeError('Parent score semantics changed')
    if (payload['development'] != receipt['last_monitor']['development']
            or payload['development_monitor_global_step'] != START
            or payload['stop_reason'] != 'maximum120_updates_completed' or payload['execution_error'] is not None):
        raise RuntimeError('Full state and normally completed3447 monitor disagree')
    if probe.tree_sha(payload['rng_state']) != PARENT_RNG_SHA:
        raise RuntimeError('Saved complete3447 RNG identity changed')
    if {id(p) for p in student.parameters()} & {id(p) for p in discriminator.parameters()}:
        raise RuntimeError('Student and D cannot share parameter ownership')
    for role, model, opt in (('student', student, optimizer), ('discriminator', discriminator, d_optimizer)):
        saved = payload[role + '_optimizer']
        previous.dstate.old.validate_saved_adam(saved, model, START, LRS[role])
        previous.prior.check_owner(opt, model)
        model.load_state_dict(payload[role], strict=True)
        previous.dstate.restore_optimizer(opt, saved)
        model.train(); training.set_requires_grad(model, True)
        for parameter in model.parameters(): parameter.grad = None
    training.restore_rng_state(payload['rng_state'], device)
    expected_rng = payload['rng_state'] if device.type == 'cuda' else dict(payload['rng_state'], cuda_all=[])
    if probe.tree_sha(training.capture_rng_state()) != probe.tree_sha(expected_rng):
        raise RuntimeError('Complete3447 RNG failed to restore')
    temporary = new_state('parent_validation')
    temporary['restore_completed'] = True
    assert_saveable(temporary, student, discriminator, optimizer, d_optimizer)
    pair.validate_effective_adam(optimizer, d_optimizer, student, discriminator, LRS['student'], START)
    return previous.content_state(student, discriminator, optimizer, d_optimizer)


def validate_replay(actual, expected):
    previous.prior.validate_replay(actual, expected)
    if actual['global_step'] != START or expected['global_step'] != START:
        raise RuntimeError('Initial monitor must replay the actual3447 endpoint')
    if actual['uniform64'] != expected['uniform64']:
        raise RuntimeError('Complete3447 uniform64 values, all utterances or tensor identities did not replay')


def run_updates(state, future, step_call, evaluate, publish, check, initial):
    if [row['global_step'] for row in future] != list(range(START + 1, STOP + 1)):
        raise RuntimeError('The complete prospective480-step plan must be fixed before training')
    if initial['global_step'] != START:
        raise RuntimeError('Do not reuse a duplicate or mislabeled initial monitor')
    rows, pending, windows, monitors = [], [], [], [initial]
    reason = 'maximum480_updates_completed'
    guards = guard_reasons(monitors, BASELINE3327)
    if guards: return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': guards[0]}
    for planned in future:
        check(); row = step_call(planned)
        state['unpublished_step_record'] = row; publish('step_losses.jsonl', row)
        state.pop('unpublished_step_record', None)
        rows.append(row); pending.append(row)
        state.update(complete_loss_records=len(rows), partial_window_steps=len(pending))
        if len(pending) == 120:
            window = window120(pending); windows.append(window); pending = []
            state.update(partial_window_steps=0, loss_windows=windows)
            publish('loss_windows.jsonl', window)
            print('MC2_CONTINUATION_WINDOW ' + json.dumps(window), flush=True)
        step = planned['global_step']
        if step - START in PAIR_UPDATES:
            check(); entry = evaluate(step)
            monitors.append(entry); state['last_monitor'] = entry
            publish('monitor.jsonl', entry)
            guards = guard_reasons(monitors, BASELINE3327); state['guard_reasons'] = guards
            publish('guard_checks.jsonl', {'global_step': step, 'stop_reasons': guards})
            check()
            if guards: reason = guards[0]; break
        publish('status.json', state, append=False)
    return {'rows': rows, 'windows': windows, 'monitors': monitors, 'reason': reason}


def save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock):
    if state['completed_global_step'] == START: return
    assert_saveable(state, student, discriminator, optimizer, d_optimizer)
    step, last = state['completed_global_step'], state['last_monitor']
    path = budget.output / 'final_full.pt'
    payload = {'kind': 'acdd_mc2_continuation480_checkpoint', 'global_step': step, 'epoch': 0,
        'configuration': config.as_dict(), 'effective_optimizer_lrs': dict(LRS),
        'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'mc_samples': 2, 'mc_private_seed_base': 74000000,
        'development': last['development'] if last and last['global_step'] == step else None,
        'development_monitor_global_step': last['global_step'] if last else None,
        'run_contract_sha256': state['contract_sha256'], 'resumable': True,
        'lineage': {'parent_checkpoint_sha256': PARENT_SHA, 'parent_contract_sha256': PARENT_CANONICAL, 'start_after_global_step': START},
        'original_step0_guard_anchor': S0_ANCHOR, 'stop_reason': state['reason'], 'execution_error': state.get('execution_error'),
        'actual_optimizer_cursors': state['optimizer_cursors'], 'student': student.state_dict(),
        'discriminator': discriminator.state_dict(), 'student_optimizer': optimizer.state_dict(),
        'discriminator_optimizer': d_optimizer.state_dict(), 'rng_state': training.capture_rng_state()}
    clock.saving(); budget.require()
    def committed():
        state['full_snapshot'] = {'path': str(path), 'global_step': step, 'sha256': None, 'committed': True, 'resumable': True}
        state['last_full_saved_global_step'] = step
        budget.committed('full', path)
    old.atomic_checkpoint_commit(payload, path, committed)
    state['full_snapshot']['sha256'] = base.sha256_file(path)
    path.chmod(0o444); state['full_snapshot']['frozen_readonly'] = True
    clock.saving(); budget.write(budget.output / 'checkpoint_receipt.json', state)


def launch_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    parser.add_argument('--parent-checkpoint-sha256')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--helper-sha256')
    parser.add_argument('--driver-sha256')
    parser.add_argument('--tests-sha256')
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if not args.run_approved or args.parent_checkpoint_sha256 != PARENT_SHA or args.max_seconds != 1800:
        raise RuntimeError('Require explicit approval, exact3447 full parent and fixed1800-second budget')
    for name in ('protocol_sha256', 'helper_sha256', 'driver_sha256', 'tests_sha256'):
        digest = getattr(args, name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise RuntimeError('Missing final reviewed SHA: ' + name)
    return args


def main(argv=None):
    started = time.monotonic(); locks, clock, budget, log = [], None, None, None
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    state, owned, completion = {}, False, None
    ledger = ForwardLedger()
    try:
        args = launch_arguments(argv)
        bound = {str(PROTOCOL): args.protocol_sha256, str(TOOLS / HELPER_NAME): args.helper_sha256,
                 str(TOOLS / HELPER_TEST_NAME): HELPER_TEST_SHA,
                 str(Path(__file__).resolve()): args.driver_sha256,
                 str(TOOLS / 'test_acdd_mc2_continuation480_v1.py'): args.tests_sha256,
                 str(TOOLS / OLD_NAME): OLD_SHA, str(PARENT): PARENT_SHA,
                 str(PARENT_RUN / 'run_contract.json'): PARENT_CONTRACT_SHA,
                 str(PARENT_RUN / 'completion.json'): PARENT_COMPLETION_SHA}
        cross.check_files(bound)
        mc = importlib.import_module(HELPER_NAME[:-3])
        if (mc.START, mc.STOP, mc.MC2_SEED) != (START, STOP, 74000000):
            raise RuntimeError('Only the reviewed3448..3927 helper is permitted')
        clock = StopClock(1800, started); clock.install(); external = previous.watchdog(1800)
        training.assert_rented_server(ROOT); cross.regular(DEV, directory=True)
        locks = training.acquire_runtime_locks(ROOT)
        if len(locks) != 3: raise RuntimeError('Require all three original workflow locks')
        budget = Budget(OUTPUT); disk = budget.require()
        if disk['required_bytes'] != INITIAL_REQUIRED_BYTES: raise RuntimeError('Initial capacity accounting changed')
        log = old.replication.open_exclusive_log(OUTPUT, budget); sys.stdout = sys.stderr = log
        print('MC2_CONTINUATION_PREFLIGHT_STARTED', flush=True)
        if training.git_head(ROOT) != cross.HEAD or subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Require unchanged clean authoritative production source')
        if ((platform.python_version(), torch.__version__, torch.version.cuda) != ('3.8.15', '1.13.1+cu116', '11.6')
                or Path(sys.executable).resolve() != Path('/root/miniconda3/envs/myconda/bin/python').resolve()
                or cross.flags() != cross.ORIGINAL_FLAGS):
            raise RuntimeError('Original runtime or precision flags changed')
        if max(PARENT.stat().st_size, PARENT.stat().st_blocks * 512) > FULL_BYTES:
            raise RuntimeError('Declared model allocation is smaller than actual3447 full')
        contract = cross.read_json(PARENT_RUN / 'run_contract.json')
        parent = validate_parent_metadata(contract, cross.read_json(PARENT_RUN / 'completion.json'))
        files, contracts, receipts, development, unused, config = cross.load_metadata()
        # 完整继承父file_sha256；从不删掉缺失旧断点的pin以绕过检查。
        cross.merge_hashes(files, contract['file_sha256']); cross.merge_hashes(files, bound)
        if contract['training_indices'] != development['retained_train_indices'] or contract['excluded_train_indices'] != development['excluded_train_indices']:
            raise RuntimeError('Parent retained training population changed')
        if receipts[3327]['status'] != 'stopped': raise RuntimeError('The formal3327 parent must remain stopped')
        if {key: contract['original_step0_guard_anchor'][key] for key in S0_ANCHOR} != S0_ANCHOR:
            raise RuntimeError('Original S0 anchors changed')
        future = future_plan(development); cross.check_files(files); clock.active()
        gpu = cross.gpu_gate(); device = torch.device('cuda')
        unused_old, batches, target, monitor_plan = cross.prepare_inputs(files, development, contracts, device)
        reference = cross.read_json(previous.CROSS_RESULT)
        if monitor_plan != contract['monitor64_plan'] or monitor_plan != cross.read_json(previous.CROSS_IDENTITY)['plan']:
            raise RuntimeError('Do not reselect fixed64 sentences, crops, t or noise')
        raw = resume.load_torch_checkpoint(cross.INPUT_FILE)
        inputs = {key: value.to(device) for key, value in raw.items()}; del raw, unused_old
        diffusion = base.VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset).to(device)
        for batch, expected in zip(batches, reference['readouts']['input_identities_after'][1:]):
            batch['q950'] = diffusion.q_sample(batch['x0'], torch.full((32,), 950, device=device, dtype=torch.long), noise=batch['source_noise'])
            if {key: cross.tensor_id(value) for key, value in batch.items()} != expected:
                raise RuntimeError('Fixed64 cached inputs changed')
        def input_identity():
            return {'old': {key: probe.tensor_sha(value) for key, value in inputs.items()},
                    'new64': [{key: cross.tensor_id(value) for key, value in batch.items()} for batch in batches],
                    'R3107': probe.tensor_sha(target)}
        input_hashes = input_identity()
        if input_hashes != contract['input_identities']: raise RuntimeError('Parent fixed input identities changed')
        dataset = base.VoiceGradDataset(ROOT / 'data', 'train', segment_length=128, expected_speakers=100, bnf_alignment=config.bnf_alignment)
        if len(dataset) != 39387: raise RuntimeError('Original training manifest changed')
        future_files = previous.register_future_files(dataset, future, files, clock.active)
        cross.check_files(files); clock.active(); budget.require()
        record = {'kind': 'acdd_mc2_continuation480_development', 'protocol_sha256': args.protocol_sha256,
            'parent_checkpoint': str(PARENT), 'parent_checkpoint_sha256': PARENT_SHA, 'parent_contract_sha256': PARENT_CANONICAL,
            'configuration': config.as_dict(), 'effective_optimizer_lrs': LRS, 'mc_samples': 2, 'mc_private_seed_base': 74000000,
            'start_after_global_step': START, 'stop_global_step': STOP, 'maximum_new_updates': UPDATES,
            'future_batches': future, 'future_training_feature_sha256': future_files,
            'training_indices': development['retained_train_indices'], 'excluded_train_indices': development['excluded_train_indices'],
            'development_contract_file_sha256': resume.DEVELOPMENT_SHA, 'monitor64_plan': monitor_plan,
            'input_identities': input_hashes, 'file_sha256': dict(files), 'original_step0_guard_anchor': S0_ANCHOR,
            'baseline3327_development': contract['baseline3327_development'], 'baseline3447_monitor': parent['last_monitor'],
            'fixed_frozen_models_sha256': contract['fixed_frozen_models_sha256'],
            'loss_windows': [[start, start + 119] for start in WINDOW_STARTS],
            'pair_monitor_updates': PAIR_UPDATES, 'uniform64_monitor_updates': UNIFORM64_UPDATES, 'forward_limits': FORWARD_LIMITS,
            'stop_rules': ['original_S0_extreme_guard', 'two_consecutive_DEV32_above3327_by5percent', 'execution_safety'],
            'no_old20_winner_gate_or_three_rising_gate': True, 'no_duplicate3447_independent_observation': True,
            'storage': {'one_full_only': True, 'system_reserve_bytes': previous.RESERVE, 'metadata_limit_bytes': META_LIMIT,
                        'directory_reserve_bytes': previous.DIR_RESERVE, 'model_slack_bytes': previous.MODEL_SLACK, 'initial': disk},
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'flags': cross.flags(), 'gpu': gpu, 'external_timeout': external},
            'maximum_compute_seconds': 1800, 'cleanup_only_seconds': 90, 'formal3327_remains_stopped': True,
            'new_endpoint_inference_or_scoring_authorized': False, 'automatic_continuation_authorized': False,
            'interpretation': 'Observe one existing MC2 trajectory; no new paired causal MC1 comparison or quality claim.'}
        OUTPUT.mkdir(); owned = True; budget.write(OUTPUT / 'run_contract.json', record)
        contract_sha = training.canonical_json_sha256(record); state = new_state(contract_sha)
        clock.active(); payload = resume.load_torch_checkpoint(PARENT)
        teacher, student, unused = base.load_teacher_and_student(base.TEACHER, config, device)
        vocoder, vconfig, unused = base.load_frozen_vocoder(ROOT, base.VOCODER_DIR / 'generator_step_2500000.pt',
            base.VOCODER_DIR / 'config.json', base.VOCODER_DIR / 'final_checkpoint_attestation_comments_20260906.json', config, device)
        discriminator = base.build_paper_discriminator(ROOT, vconfig, config, device)
        fixed_d = base.build_paper_discriminator(ROOT, vconfig, config, device)
        old_d = resume.load_torch_checkpoint(pair.PARENT / 'latest_full.pt')
        if old_d['global_step'] != 3107 or old_d['run_contract_sha256'] != pair.PARENT_CONTRACT:
            raise RuntimeError('Original fixedD3107 changed')
        fixed_d.load_state_dict(old_d['discriminator'], strict=True); del old_d
        fixed_d.eval(); training.set_requires_grad(fixed_d, False)
        mean, std = base.load_mel_statistics(ROOT / 'data', device)
        frozen = {'teacher': teacher, 'vocoder': vocoder, 'fixed_D3107': fixed_d}
        frozen_sha = {name: pair.model_sha(model) for name, model in frozen.items()}
        if frozen_sha != contract['fixed_frozen_models_sha256']: raise RuntimeError('Frozen models changed')
        models = dict(frozen, student=student, active_D=discriminator, diffusion=diffusion)
        optimizer = torch.optim.Adam(student.parameters(), lr=LRS['student'], betas=(.5, .9))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LRS['discriminator'], betas=(.5, .9))
        restored = restore_parent(payload, student, discriminator, optimizer, d_optimizer, config, contract, parent, device)
        state.update(restore_completed=True, restore_identity=restored, status='running', phase='initial_monitor',
                     optimizer_cursors=actual_cursors(optimizer, d_optimizer))
        del payload
        ledger.install(student, teacher, vocoder, discriminator, fixed_d)
        def check(): clock.active(); budget.require()
        def unchanged():
            pair.assert_frozen(frozen, frozen_sha)
            if input_identity() != input_hashes or cross.flags() != cross.ORIGINAL_FLAGS:
                raise RuntimeError('Fixed inputs or precision changed')
        def evaluate(step):
            update = step - START
            if update not in PAIR_UPDATES: raise RuntimeError('Unregistered continuation monitor')
            check(); before = dict(ledger.counts)
            with previous.monitoring_state(models, device):
                entry = pair.monitor(student, teacher, vocoder, fixed_d, diffusion, inputs, target, mean, std, config, device, step)
                entry['legacy_monitor_updates_from3107'] = entry['updates']
                entry['updates'] = entry['trial_updates'] = update
                if update in UNIFORM64_UPDATES:
                    entry['uniform64'] = previous.uniform64_monitor(student, teacher, diffusion, batches, monitor_plan, config, device, clock.active)
            expected = {'teacher': 21 if update in UNIFORM64_UPDATES else 1,
                        'student': 4 if update in UNIFORM64_UPDATES else 2, 'vocoder': 2,
                        'active_D_branches': 0, 'fixed_D_branches': 16}
            if ledger.delta(before) != expected: raise RuntimeError('Unregistered monitoring forwards')
            entry['actual_forward_counts'] = ledger.delta(before); unchanged()
            return entry
        try:
            initial = evaluate(START); validate_replay(initial, parent['last_monitor'])
            state['last_monitor'] = initial; budget.write(OUTPUT / 'monitor.jsonl', initial, append=True)
            def step_call(planned):
                step = planned['global_step']; state['phase'] = 'input_checks'
                batch = base.batch_for(dataset, planned['indices'], planned['crop_seed'])
                training.seed_everything(planned['noise_seed'])
                identity = mc.original_input_identity(batch, device, config); receipt = {}
                check(); before = dict(ledger.counts)
                losses = complete_update(state, step, optimizer, d_optimizer, lambda: mc.train_batch_mc2(
                    batch, teacher, student, vocoder, discriminator, diffusion, optimizer, d_optimizer, mean, std,
                    device, config, global_step=step, mc_receipt=receipt))
                row = dict(losses, global_step=step, updates=step - START, indices=planned['indices'],
                           crop_seed=planned['crop_seed'], noise_seed=planned['noise_seed'], input_identity=identity, mc=receipt,
                           actual_optimizer_cursors=state['optimizer_cursors'], actual_forward_counts=ledger.delta(before))
                state['unpublished_step_record'] = row
                for key in ('source_noise', 'score_noise1', 't1', 't1_values'):
                    if receipt[key] != identity[key]: raise RuntimeError('Actual original RNG draw differs: ' + key)
                if row['actual_forward_counts'] != {'teacher': 2, 'student': 2, 'vocoder': 3, 'active_D_branches': 32, 'fixed_D_branches': 0}:
                    raise RuntimeError('Unexpected actual complete-batch forward counts')
                previous.adam_counters(optimizer, d_optimizer, step)
                if [optimizer.param_groups[0]['lr'], d_optimizer.param_groups[0]['lr']] != [LRS['student'], LRS['discriminator']]:
                    raise RuntimeError('Effective role learning rate changed')
                return row
            def publish(name, value, append=True): budget.write(OUTPUT / name, value, append=append)
            result = run_updates(state, future, step_call, evaluate, publish, check, initial)
            state.update(reason=result['reason'], phase='finalizing', loss_windows=result['windows'],
                primary_loss_trend=primary_loss_trend(result['rows']),
                monitor_global_steps=[entry['global_step'] for entry in result['monitors']],
                end_monitor_available=state['last_monitor']['global_step'] == state['completed_global_step'])
            unchanged(); pair.validate_effective_adam(optimizer, d_optimizer, student, discriminator, LRS['student'], state['completed_global_step'])
            save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock); check()
            state['status'] = 'completed480' if result['reason'] == 'maximum480_updates_completed' else 'protective_stop'
            state['phase'] = 'trial_complete'; budget.write(OUTPUT / 'trial_completion.json', state)
        except BaseException as error:
            clock.request('execution_abort')
            state.update(status='execution_aborted', reason=str(error), execution_error=repr(error))
            state['execution_stop_reasons'].append(str(error))
            if state['restore_completed'] and not state['partial_batch_may_have_updated_D'] and not state['full_snapshot']:
                try:
                    unchanged(); save_endpoint(state, config, student, discriminator, optimizer, d_optimizer, budget, clock)
                except BaseException as save_error: state['checkpoint_error'] = repr(save_error)
            raise
        unchanged(); check(); cross.check_files(files)
        if training.canonical_json_sha256(cross.read_json(OUTPUT / 'run_contract.json')) != contract_sha:
            raise RuntimeError('Published continuation contract changed')
        complete = state['new_updates'] == UPDATES
        if complete and ledger.counts != FORWARD_LIMITS: raise RuntimeError('Complete480 forward budget differs')
        completion = {'status': 'finalization_pending', 'cleanup_verified': False,
            'intended_status': 'completed480' if complete else 'protective_stop', 'state': dict(state),
            'contract_sha256': contract_sha, 'actual_forward_counts': dict(ledger.counts),
            'elapsed_seconds': time.monotonic() - started, 'no_new_endpoint_inference_or_scoring_performed': True,
            'formal3327_remains_stopped': True, 'automatic_continuation_authorized': False}
        budget.write(OUTPUT / 'completion.json', completion); check()
        print('MC2_CONTINUATION_COMPUTE_FINISHED_AWAITING_CLEANUP', flush=True)
    except BaseException as error:
        if budget is not None: budget.cancel_future_on_fatal_abort(repr(error))
        try:
            failure = old.failure_record(OUTPUT if owned else None, state, {}, error, budget, ledger)
            print('MC2_CONTINUATION_FAILED ' + json.dumps(failure, ensure_ascii=False), flush=True)
        except BaseException as secondary:
            try: saved_stderr.write('Failure receipt unavailable: %r; original=%r\n' % (secondary, error)); saved_stderr.flush()
            except BaseException: pass
        raise
    finally:
        original_error = sys.exc_info()[1]; sys.stdout, sys.stderr = saved_stdout, saved_stderr
        errors = cleanup_resources(ledger, clock, log, locks)
        if errors:
            state['cleanup_errors'] = errors; cleanup_error = RuntimeError('MC2 continuation cleanup failed: ' + repr(errors))
            try:
                if budget is not None: budget.cancel_future_on_fatal_abort(repr(cleanup_error))
                old.failure_record(OUTPUT if owned else None, state, {}, original_error or cleanup_error, budget, ledger)
            except BaseException as receipt_error:
                try: saved_stderr.write('Cleanup failure receipt unavailable: %r\n' % receipt_error); saved_stderr.flush()
                except BaseException: pass
            if original_error is None: raise cleanup_error
        elif original_error is None and completion is not None:
            try: publish_completed_after_cleanup(OUTPUT, completion, budget)
            except BaseException as final_error:
                try: old.failure_record(OUTPUT if owned else None, state, {}, final_error, budget, ledger)
                except BaseException: pass
                raise


if __name__ == '__main__':
    main()  # 必须完成主线程与独立复审、CPU测试、冻结及即时容量/锁检查后另行批准。
