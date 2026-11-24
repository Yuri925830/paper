"""草稿：原重建MC2之外增加教师转换波形的感知监督；本文件不启动训练。"""
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import types

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
ANCHOR_FILE = TOOLS / 'acdd_mc2_conversion_anchor4287_v1.py'
ANCHOR_SHA = '56d366da65492210b2b3754afbd54dcf3d2fef416bae74c2e3f0489e8473c640'
if hashlib.sha256(ANCHOR_FILE.read_bytes()).hexdigest() != ANCHOR_SHA:
    raise RuntimeError('Reviewed teacher endpoint preparation changed')
_spec = importlib.util.spec_from_file_location('_perceptual_anchor_scaffolding', ANCHOR_FILE)
_anchor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_anchor)
torch = _anchor.torch
START, STOP = _anchor.START, _anchor.STOP


def perceptual_backward(student, diffusion, context, weight, receipt, reconstruction_total,
                        vocoder, mean, std, perceptual_loss):
    """沿学生→冻结G→冻结WavLM保留梯度；参考教师只作为固定目标。"""
    from fastvoicegrad.objectives import reverse_mean_to_zero
    if any(m.training for m in vocoder.modules()) or any(p.requires_grad or p.grad is not None for p in vocoder.parameters()):
        raise RuntimeError('The original vocoder must remain frozen/eval')
    prediction = reverse_mean_to_zero(student, diffusion, context['x_start'], context['times'],
        context['speaker'], context['bnf'])
    if prediction.shape != context['teacher_target'].shape or not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError('Invalid student conversion endpoint')
    with torch.no_grad():
        target_wave = vocoder(context['teacher_target'] * std + mean)
    generated_wave = vocoder(prediction * std + mean)
    expected_shape = (32, 1, 32768)
    if tuple(target_wave.shape) != expected_shape or tuple(generated_wave.shape) != expected_shape:
        raise RuntimeError('Require unchanged256-hop full32 vocoder output')
    if not generated_wave.requires_grad or not bool(torch.isfinite(generated_wave).all()) or not bool(torch.isfinite(target_wave).all()):
        raise FloatingPointError('Waveform gradient disconnected or nonfinite')
    # 每条语音只把真实长度以内的样本交给辅助损失，原重建分支不改。
    result = perceptual_loss(generated_wave, target_wave, context['valid_lengths'] * 256)
    loss = result['loss']
    if loss.ndim != 0 or not loss.requires_grad or not bool(torch.isfinite(loss)):
        raise FloatingPointError('Perceptual scalar disconnected or nonfinite')
    joint = float(reconstruction_total.detach()) + float(weight * loss.detach())
    if not math.isfinite(joint): raise FloatingPointError('Nonfinite joint objective')
    (weight * loss).backward()
    receipt.update(kind='FINALLY_inspired_conversion_perceptual_auxiliary',
        auxiliary_formula='100*WavLM_conv_MSE + raw_16k_STFT_magnitude_L1',
        conversion_uses_vocoder_or_discriminator=True, conversion_uses_discriminator=False,
        anchor_raw=float(loss.detach()), anchor_weighted=float(weight * loss.detach()),
        reconstruction_student_total=float(reconstruction_total.detach()), joint_student_total=joint,
        feature_mse=float(result['feature_mse'].detach()), stft_l1=float(result['stft_l1'].detach()),
        per_utterance_loss=result['per_utterance_loss'].detach().cpu().tolist(),
        student_conversion=_anchor._core.tensor_identity(prediction),
        target_waveform=_anchor._core.tensor_identity(target_wave),
        generated_waveform=_anchor._core.tensor_identity(generated_wave),
        additional_student_forward_calls=1, additional_vocoder_forward_calls=2,
        anchor_backward_completed=True, raw_mel_L1_optimized=False,
        metric_model_optimized=False, original_reconstruction_losses_preserved=True)
    return loss.detach()


def _build():
    # 复用已审核的30步教师配对和原重建流程；替换唯一额外L1反传，不改原生产文件。
    namespace = dict(_anchor._namespace)
    # 浅拷贝字典不会改变函数的__globals__，因此把三个有范围/MC2内部调用的函数重新绑定。
    # 仍执行相同code；旧anchor及生产模块的namespace保持原样。
    for name in ('validate_request', 'second_score_draws', 'mc2_score_distillation_loss'):
        original = _anchor._namespace[name]
        namespace[name] = types.FunctionType(original.__code__, namespace, name,
            original.__defaults__, original.__closure__)
        namespace[name].__kwdefaults__ = original.__kwdefaults__
    namespace['perceptual_backward'] = perceptual_backward
    function = copy.deepcopy(_anchor._augmented_ast)
    function.name = '_train_with_perceptual_anchor'
    function.args.kwonlyargs.append(ast.arg(arg='perceptual_loss', annotation=None, type_comment=None))
    function.args.kw_defaults.append(None)
    hits = 0
    for statement in function.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == 'anchor_loss':
            statement.value = ast.parse('perceptual_backward(student, diffusion, anchor_context, anchor_weight, anchor_receipt, total_loss, vocoder, mel_mean, mel_std, perceptual_loss)', mode='eval').body
            hits += 1
        if isinstance(statement, ast.Return):
            for key in statement.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.startswith('conversion_anchor_'):
                    key.value = key.value.replace('conversion_anchor_', 'conversion_perceptual_', 1)
    if hits != 1: raise RuntimeError('Reviewed auxiliary insertion point changed')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(Path(__file__).resolve()) + ':perceptual_auxiliary', 'exec'), namespace)
    return namespace['_train_with_perceptual_anchor'], namespace, function


train_batch_mc2_perceptual_anchor, _namespace, _augmented_ast = _build()
