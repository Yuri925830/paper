"""草稿：保留重建 MC2，另加转换端点 L1；不启动训练、不保存文件、不决定权重。"""
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import sys
import types

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
FROZEN = TOOLS / 'acdd_mc2_train_batch_v1.py'
FROZEN_SHA = 'bbf94b73449f14d7fd2ef662c464028bdc18af60235b9733edd1adf47551bf16'
START, STOP = 4287, 4527
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Only the registered rental may execute this draft helper')
_source = FROZEN.read_bytes()
if hashlib.sha256(_source).hexdigest() != FROZEN_SHA:
    raise RuntimeError('Frozen reconstruction MC2 source changed')
_spec = importlib.util.spec_from_file_location('_conversion_anchor_frozen_mc2', FROZEN)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA:
    raise RuntimeError('Frozen reconstruction MC2 source changed during import')
torch = _core.torch


def speaker_shuffle(speakers, seed):
    """用局部随机顺序寻找一一对应的换说话人排列；没有可行排列就报错。"""
    if len(speakers) != 32 or any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError('Require the actual32 source speaker identifiers')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Require an explicit nonnegative shuffle seed')
    rng = random.Random(seed)
    sources, targets = list(range(32)), list(range(32))
    rng.shuffle(sources); rng.shuffle(targets)
    # 二分图匹配只按说话人ID判断；不读性能、音频或目标质量，不挑分数好的排列。
    owner = [-1] * 32
    def assign(source, seen):
        for target in targets:
            if target in seen or speakers[source] == speakers[target]: continue
            seen.add(target)
            if owner[target] == -1 or assign(owner[target], seen):
                owner[target] = source
                return True
        return False
    if any(not assign(source, set()) for source in sources):
        raise RuntimeError('No valid within-batch cross-speaker shuffle; never use identity fallback')
    permutation = [None] * 32
    for target, source in enumerate(owner): permutation[source] = target
    if sorted(permutation) != list(range(32)) or any(speakers[i] == speakers[j] for i, j in enumerate(permutation)):
        raise RuntimeError('Invalid cross-speaker permutation')
    return permutation


def validate_anchor_arguments(batch, global_step, weight, source_indices, retained, excluded,
                              shuffle_seed, reverse_seed, receipt):
    if type(global_step) is not int or not START < global_step <= STOP:
        raise ValueError('Only proposed complete steps4288..4527 are permitted')
    if type(weight) not in (float, int) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('Require an explicit finite positive anchor weight; no default is adopted')
    if type(reverse_seed) is not int or not 0 <= reverse_seed < 2**63:
        raise ValueError('Require an explicit nonnegative private reverse seed')
    if not isinstance(receipt, dict) or receipt:
        raise ValueError('Require one fresh empty anchor receipt per batch')
    # 完整数据契约和特征身份由未来driver预先核验；这里拒绝训练/参考混入52条开发语音。
    if (not isinstance(retained, frozenset) or not isinstance(excluded, frozenset)
            or len(retained) != 39335 or len(excluded) != 52 or retained & excluded
            or len(source_indices) != 32 or len(set(source_indices)) != 32
            or any(type(i) is not int or i not in retained or i in excluded for i in source_indices)):
        raise ValueError('Require a32-source batch from the fixed39335 training complement, excluding52')
    # 原VoiceGradDataset实际返回spk_name；不能假造speaker字段使toy测试掩盖真实接口错误。
    speakers = list(batch['spk_name'])
    permutation = speaker_shuffle(speakers, shuffle_seed)
    return speakers, permutation


def global_torch_rng(device):
    # 私有反向随机数不能移动原重建 source/score/MC2 的随机流。
    states = [torch.get_rng_state().clone()]
    if device.type == 'cuda': states.extend(x.clone() for x in torch.cuda.get_rng_state_all())
    return states


@torch.no_grad()
def prepare_anchor(batch, x_zero, bnf, speaker, source_noise, diffusion, teacher, device,
                   config, global_step, anchor_weight, source_indices, retained_train_indices,
                   excluded_train_indices, shuffle_seed, reverse_seed, anchor_receipt):
    speakers, permutation = validate_anchor_arguments(batch, global_step, anchor_weight, source_indices,
        retained_train_indices, excluded_train_indices, shuffle_seed, reverse_seed, anchor_receipt)
    if any(m.training for m in teacher.modules()) or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise RuntimeError('Teacher must already be fully frozen/eval with no gradients')
    length = batch['mel_len'].to(device=device)
    if length.shape != (32,) or length.dtype != torch.long or bool(((length < 1) | (length > 128)).any()):
        raise ValueError('Require the original32 valid frame lengths within1..128')
    target_speaker = speaker[permutation].detach()
    times = torch.full((32,), config.student_start_step, device=device, dtype=torch.long)
    x_start = diffusion.q_sample(x_zero, times, noise=source_noise).detach()
    generator = torch.Generator(device=device).manual_seed(reverse_seed)
    reverse_initial = _core.tensor_identity(generator.get_state())
    before_rng = global_torch_rng(device)
    schedule = list(reversed(diffusion.sampling_sequence(30, 950)))
    observed = []
    def observe_teacher(module, arguments):
        index = len(observed)
        if (index >= 30 or not torch.equal(arguments[1], torch.full_like(times, schedule[index]))
                or not torch.equal(arguments[2], target_speaker) or not torch.equal(arguments[3], bnf)
                or (index == 0 and not torch.equal(arguments[0], x_start))):
            raise RuntimeError('Teacher30 call no longer shares declared q950/target speaker/source BNF')
        observed.append(schedule[index])
    handle = teacher.register_forward_pre_hook(observe_teacher)
    try:
        # 原sample的False分支直接clone这个已加噪状态；后续29次噪声只来自私有generator。
        target = diffusion.sample(teacher, x_start, target_speaker, bnf, num_steps=30,
            start_step=950, generator=generator, diffuse_source=False)
    finally:
        handle.remove()
    after_rng = global_torch_rng(device)
    if len(before_rng) != len(after_rng) or any(not torch.equal(a, b) for a, b in zip(before_rng, after_rng)):
        raise RuntimeError('Teacher anchor unexpectedly advanced the original Torch RNG')
    if len(observed) != 30 or observed != schedule:
        raise RuntimeError('Teacher endpoint must use exactly the frozen30-step schedule')
    if (target.shape != x_start.shape or target.dtype != x_start.dtype or target.device != x_start.device
            or target.requires_grad or target.grad_fn is not None or not bool(torch.isfinite(target).all())):
        raise RuntimeError('Independent teacher endpoint is malformed, nonfinite, or attached')
    anchor_receipt.update(kind='supplemental_conversion_ancestral30_endpoint_L1_draft',
        global_step=global_step, source_indices=list(source_indices), source_speakers=speakers,
        target_permutation=permutation, target_indices=[source_indices[j] for j in permutation],
        target_speakers=[speakers[j] for j in permutation], same_speaker_count=0,
        shuffle_seed=shuffle_seed, reverse_seed=reverse_seed, anchor_weight=float(anchor_weight),
        source_noise=_core.tensor_identity(source_noise), shared_q950=_core.tensor_identity(x_start),
        target_speaker_tensor=_core.tensor_identity(target_speaker), source_bnf=_core.tensor_identity(bnf),
        teacher_target=_core.tensor_identity(target), valid_lengths=length.cpu().tolist(),
        teacher_forward_calls=len(observed), teacher_schedule=observed,
        private_reverse_rng_initial=reverse_initial, private_reverse_rng_final=_core.tensor_identity(generator.get_state()),
        original_torch_rng_unchanged=True, teacher_target_depends_on_student=False,
        reconstruction_branch_unchanged=True, conversion_uses_vocoder_or_discriminator=False)
    return dict(x_start=x_start, times=times, speaker=target_speaker, bnf=bnf,
                teacher_target=target.detach(), valid_lengths=length)


def conversion_anchor_backward(student, diffusion, context, weight, receipt, reconstruction_total):
    # 先释放原重建反传图，再建立转换学生图；Adam仍在原函数后面，仅执行一次。
    from fastvoicegrad.objectives import reverse_mean_to_zero
    prediction = reverse_mean_to_zero(student, diffusion, context['x_start'], context['times'],
        context['speaker'], context['bnf'])
    if prediction.shape != context['teacher_target'].shape or not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError('Conversion student prediction is invalid')
    mask = torch.arange(128, device=prediction.device)[None, None, :] < context['valid_lengths'][:, None, None]
    per_l1 = ((prediction - context['teacher_target']).abs() * mask).sum((1, 2)) / (80 * context['valid_lengths'])
    loss = per_l1.mean()
    joint = float(reconstruction_total.detach()) + float(weight * loss.detach())
    if not bool(torch.isfinite(loss)) or not math.isfinite(joint):
        raise FloatingPointError('Conversion anchor or joint student total is nonfinite')
    (weight * loss).backward()
    receipt.update(anchor_raw=float(loss.detach()), anchor_weighted=float(weight * loss.detach()),
        per_utterance_valid_frame_l1=per_l1.detach().cpu().tolist(), student_conversion=_core.tensor_identity(prediction),
        reconstruction_student_total=float(reconstruction_total.detach()), joint_student_total=joint,
        additional_student_forward_calls=1, anchor_backward_completed=True)
    return loss.detach()


def _make_train_function():
    # 新私有namespace内只绑定新范围；原模块的START/STOP、生产函数和模型forward均不改。
    namespace = dict(vars(_core)); namespace.update(START=START, STOP=STOP)
    for name in ('validate_request', 'second_score_draws', 'mc2_score_distillation_loss'):
        original = getattr(_core, name)
        namespace[name] = types.FunctionType(original.__code__, namespace, name,
            original.__defaults__, original.__closure__)
        namespace[name].__kwdefaults__ = original.__kwdefaults__
    namespace.update(prepare_anchor=prepare_anchor, conversion_anchor_backward=conversion_anchor_backward)
    tree = ast.parse(_source)
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'train_batch_mc2']
    if len(selected) != 1: raise RuntimeError('Frozen MC2 entry count changed')
    function = copy.deepcopy(selected[0]); function.name = '_train_with_anchor'
    extras = ('anchor_weight', 'source_indices', 'retained_train_indices', 'excluded_train_indices',
              'shuffle_seed', 'reverse_seed', 'anchor_receipt')
    function.args.kwonlyargs.extend(ast.arg(arg=name, annotation=None, type_comment=None) for name in extras)
    function.args.kw_defaults.extend(None for _ in extras)  # 权重与种子全部必传，没有隐含默认。
    prepared = ast.parse('''anchor_context = prepare_anchor(batch, x_zero, bnf, speaker, source_noise,
diffusion, teacher, device, config, global_step, anchor_weight, source_indices,
retained_train_indices, excluded_train_indices, shuffle_seed, reverse_seed, anchor_receipt)''').body[0]
    backward = ast.parse('''anchor_loss = conversion_anchor_backward(student, diffusion,
anchor_context, anchor_weight, anchor_receipt, total_loss)''').body[0]
    hits = {'prepare': 0, 'backward': 0, 'return': 0}; body = []
    for statement in function.body:
        body.append(statement)
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = ast.dump(statement.value.func)
            if call == ast.dump(ast.parse('record_first_draws', mode='eval').body):
                body.append(prepared); hits['prepare'] += 1
            if call == ast.dump(ast.parse('torch.autograd.backward', mode='eval').body):
                body.append(backward); hits['backward'] += 1
        if isinstance(statement, ast.Return):
            if not isinstance(statement.value, ast.Dict): raise RuntimeError('Frozen loss return changed')
            for name, expression in (
                ('conversion_anchor_raw', 'float(anchor_loss)'),
                ('conversion_anchor_weighted', 'float(anchor_weight * anchor_loss)'),
                ('conversion_anchor_weight', 'float(anchor_weight)'),
                ('student_joint_total', "anchor_receipt['joint_student_total']"),
            ):
                statement.value.keys.append(ast.Constant(value=name))
                statement.value.values.append(ast.parse(expression, mode='eval').body)
            hits['return'] += 1
    if hits != {'prepare': 1, 'backward': 1, 'return': 1}:
        raise RuntimeError('Frozen MC2 insertion points changed')
    function.body = body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(Path(__file__).resolve()) + ':frozen_MC2_plus_anchor', 'exec'), namespace)
    return namespace['_train_with_anchor'], namespace, function, selected[0]


train_batch_mc2_conversion_anchor, _namespace, _augmented_ast, _original_ast = _make_train_function()
