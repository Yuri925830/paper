"""Read-only gradient components on an already built S -> G / MC2 graph.

Torch and training are supplied by the caller. This module imports no project or
GPU module, constructs no optimizer, changes no mode/RNG, and writes no files.
The caller owns the frozen S/G/D/input identities and the original precision.
"""
import math


def _finite(torch, value, label):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(label + ' must be a floating tensor')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite ' + label)


def _scalar(torch, value, label):
    _finite(torch, value, label)
    if value.ndim != 0:
        raise ValueError(label + ' must be one scalar')
    return float(value.detach())


def blockwise_statistics(*, torch, banks, block_elements=262144, difference_pair=None, check=None):
    """Read CPU banks; convert only a small block at a time to float64 for dots.

    None means this parameter was unused, not a saved all-zero tensor. Both have
    zero contribution to dot products, but their support is reported separately.
    Zero-vector cosines and relative error with a zero reference are undefined.
    """
    if type(block_elements) is not int or block_elements <= 0 or not banks:
        raise ValueError('Nonempty banks and a positive block size are required')
    check = check if check is not None else lambda: None
    labels = list(banks); names = list(banks[labels[0]])
    if not names or any(set(bank) != set(names) for bank in banks.values()):
        raise ValueError('Parameter names differ between gradient banks')
    if difference_pair is not None and (len(difference_pair) != 2 or any(k not in banks for k in difference_pair)):
        raise ValueError('Difference pair must name two existing banks')
    gram = {a: {b: 0.0 for b in labels} for a in labels}
    absent = {label: [] for label in labels}
    support_difference, error_squared, max_error = [], 0.0, 0.0
    for name in names:
        check()
        flat, shape = {}, None
        for label in labels:
            value = banks[label][name]
            if value is None:
                absent[label].append(name); continue
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or not value.is_floating_point() or value.requires_grad):
                raise ValueError('Gradient banks must contain detached CPU floating tensors or None')
            if shape is not None and value.shape != shape:
                raise ValueError('Gradient shapes differ for ' + name)
            shape = value.shape; flat[label] = value.reshape(-1)
        if difference_pair is not None:
            left, right = difference_pair
            if (banks[left][name] is None) != (banks[right][name] is None): support_difference.append(name)
        if not flat: continue
        size = next(iter(flat.values())).numel()
        for start in range(0, size, block_elements):
            check()
            vectors = {label: value[start:start + block_elements].to(dtype=torch.float64) for label, value in flat.items()}
            for label, value in vectors.items(): _finite(torch, value, 'gradient ' + label + '/' + name)
            for i, a in enumerate(labels):
                if a not in vectors: continue
                for b in labels[i:]:
                    if b not in vectors: continue
                    value = float(torch.dot(vectors[a], vectors[b])); gram[a][b] += value
                    if a != b: gram[b][a] += value
            if difference_pair is not None:
                left, right = difference_pair
                if left in vectors and right in vectors: error = vectors[left] - vectors[right]
                elif left in vectors: error = vectors[left]
                elif right in vectors: error = -vectors[right]
                else: continue
                error_squared += float(torch.dot(error, error))
                max_error = max(max_error, float(error.abs().max()))
    if any(not math.isfinite(v) for row in gram.values() for v in row.values()) or not math.isfinite(error_squared):
        raise ValueError('Nonfinite aggregated gradient statistic')
    norms = {label: math.sqrt(max(gram[label][label], 0.0)) for label in labels}
    cosines = {a: {b: gram[a][b] / (norms[a] * norms[b]) if norms[a] and norms[b] else None for b in labels} for a in labels}
    result = dict(norms=norms, gram=gram, cosines=cosines, none_parameter_names=absent,
                  parameter_count=len(names), aggregation_dtype='float64', block_elements=block_elements)
    if difference_pair is not None:
        left, right = difference_pair; absolute = math.sqrt(error_squared)
        result['difference'] = dict(left=left, reference=right, absolute_l2=absolute, max_absolute=max_error,
            relative_l2=absolute / norms[right] if norms[right] else None, reference_l2=norms[right],
            reference_norm_zero=norms[right] == 0.0, cosine=cosines[left][right],
            none_support_equal=not support_difference, none_support_mismatched_parameters=support_difference,
            interpretation='Descriptive only; no numerical equivalence threshold or production bit-exact claim')
    return result


def _sum_components(banks, names):
    # Add CPU copies only. This is never used to construct the direct joint VJP.
    result = {}
    for name in names:
        values = [banks[label][name] for label in ('adv', '2fm', '45dist') if banks[label][name] is not None]
        value = values[0].clone() if values else None
        for other in values[1:]: value.add_(other)
        result[name] = value
    return result


def component_gradients(*, torch, training, prediction, reference_mel, fake_waveform,
                        real_waveform, discriminator, dist_loss, named_parameters,
                        check=None, block_elements=262144):
    """Return small JSON-safe measurements, never the full gradient banks.

    prediction and fake_waveform must share the caller's one S -> G graph.
    dist_loss is the caller's raw mean of TWO independent clean-target weighted
    L1 losses, using that same prediction. We do not regenerate or detach it.
    real_waveform is detached here. D and G are frozen by the caller; S is not.
    GT mel L1 is diagnostic only and never enters the joint objective.

    Eight D branches: 16 forwards, 24 branch autograd calls, 5 parameter VJPs.
    There are no S/G/teacher forwards here. The final VJP releases this graph.
    """
    check = check if check is not None else lambda: None
    named = list(named_parameters)
    if not named or len({name for name, _ in named}) != len(named) or len({id(p) for _, p in named}) != len(named):
        raise ValueError('Require unique named student parameters')
    names, parameters = zip(*named)
    if any(not p.requires_grad or not p.is_floating_point() for p in parameters):
        raise ValueError('Student parameters must require gradients')
    d_parameters = list(discriminator.parameters())
    if any(p.requires_grad for p in d_parameters): raise ValueError('Discriminator parameters must be frozen')
    if any(p.grad is not None for p in list(parameters) + d_parameters):
        raise ValueError('Start with absent parameter.grad; helper never clears an existing gradient')
    for label, value in (('prediction', prediction), ('reference_mel', reference_mel),
                         ('fake_waveform', fake_waveform), ('real_waveform', real_waveform)):
        _finite(torch, value, label)
    if (prediction.shape != reference_mel.shape or prediction.device != reference_mel.device
            or prediction.dtype != reference_mel.dtype or fake_waveform.shape != real_waveform.shape
            or fake_waveform.device != real_waveform.device or fake_waveform.dtype != real_waveform.dtype
            or fake_waveform.device != prediction.device or fake_waveform.dtype != prediction.dtype
            or not prediction.requires_grad or not fake_waveform.requires_grad):
        raise ValueError('Prediction/reference/wave graph structure differs')
    dist_value = _scalar(torch, dist_loss, 'raw MC2 dist_loss')
    weighted_dist_value = _scalar(torch, 45.0 * dist_loss, 'weighted MC2 dist_loss')
    if not dist_loss.requires_grad or dist_loss.device != prediction.device:
        raise ValueError('MC2 dist_loss must retain the original student graph')
    branches = list(discriminator.loss_branches())
    if len(branches) != 8 or len({id(branch) for branch in branches}) != 8:
        raise ValueError('Require the eight distinct MPD/MRD loss branches')
    counts = dict(D_branch_forwards=0, branch_autograd_calls=0, parameter_vjp_calls=0)
    proxy = fake_waveform.detach().requires_grad_(True)
    wave_adv = torch.zeros_like(proxy); wave_fm = torch.zeros_like(proxy); wave_joint = torch.zeros_like(proxy)
    branch_values = []
    for index, branch in enumerate(branches):
        check()
        with torch.no_grad():
            counts['D_branch_forwards'] += 1
            _, real_features = branch(real_waveform.detach())
        counts['D_branch_forwards'] += 1
        fake_score, fake_features = branch(proxy)
        adv = training.least_squares_student_loss([fake_score])
        fm_raw = training.feature_matching_loss([real_features], [fake_features])
        weighted_fm = 2.0 * fm_raw
        direct_joint = adv + weighted_fm
        branch_values.append(dict(branch_index=index, adv=_scalar(torch, adv, 'adv'),
            fm_raw=_scalar(torch, fm_raw, 'raw FM'), direct_adv_plus_2fm=_scalar(torch, direct_joint, 'branch joint')))
        for loss, accumulator, keep in ((adv, wave_adv, True), (weighted_fm, wave_fm, True), (direct_joint, wave_joint, False)):
            check(); counts['branch_autograd_calls'] += 1
            gradient, = torch.autograd.grad(loss, proxy, retain_graph=keep)
            _finite(torch, gradient, 'branch proxy gradient'); accumulator.add_(gradient.detach())
        del real_features, fake_features, fake_score, adv, fm_raw, weighted_fm, direct_joint, gradient
    gt_loss = torch.abs(prediction - reference_mel.detach()).mean()
    gt_value = _scalar(torch, gt_loss, 'diagnostic GT mel L1')
    banks = {}

    def pull(label, outputs, grad_outputs, keep):
        check(); counts['parameter_vjp_calls'] += 1
        values = torch.autograd.grad(outputs, parameters, grad_outputs=grad_outputs,
                                     retain_graph=keep, allow_unused=True)
        bank = {}
        for name, parameter, value in zip(names, parameters, values):
            if value is not None:
                if value.shape != parameter.shape or value.dtype != parameter.dtype: raise ValueError('Parameter gradient structure differs')
                _finite(torch, value, 'parameter gradient ' + label + '/' + name)
            bank[name] = None if value is None else value.detach().cpu()
        banks[label] = bank

    pull('adv', fake_waveform, wave_adv, True)
    pull('2fm', fake_waveform, wave_fm, True)
    pull('45dist', dist_loss, dist_loss.new_tensor(45.0), True)
    pull('gt_mel_l1', gt_loss, None, True)
    # Crucially, use the separately differentiated branch JOINT vector, not
    # wave_adv + wave_fm or the sum of the three parameter gradient banks.
    pull('direct_joint', (fake_waveform, dist_loss), (wave_joint, dist_loss.new_tensor(45.0)), False)
    banks['split'] = _sum_components(banks, names)
    statistics = blockwise_statistics(torch=torch, banks=banks, block_elements=block_elements,
                                     difference_pair=('split', 'direct_joint'), check=check)
    if any(p.grad is not None for p in list(parameters) + d_parameters):
        raise ValueError('autograd.grad unexpectedly populated parameter.grad')
    check()
    adv_value = math.fsum(row['adv'] for row in branch_values)
    fm_value = math.fsum(row['fm_raw'] for row in branch_values)
    return dict(kind='readonly_MC2_student_parameter_component_gradients',
        objective='adv + 2*FM_raw + 45*mean(two clean-target weighted L1 scalars)',
        diagnostic_only_component='gt_mel_l1', direct_joint_excludes_GT=True,
        losses=dict(adv=adv_value, fm_raw=fm_value, fm_weighted=2.0 * fm_value,
                    dist_raw=dist_value, dist_weighted=weighted_dist_value, gt_mel_l1=gt_value),
        branches=branch_values, actual_counts=counts, parameter_gradients=statistics,
        graph_scope='Caller supplied one S->G graph and MC2 loss; fixed-input diagnostic, not a production batch replay',
        limitation='Separate VJPs and direct joint are measured on the same graph; neither is claimed bit-exact production training')
