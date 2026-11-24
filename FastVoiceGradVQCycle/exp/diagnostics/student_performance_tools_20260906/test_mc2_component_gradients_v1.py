"""Prepared rental-CPU toy tests; never execute locally or construct real models."""
import json
import os
from pathlib import Path
import unittest

if (os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
        or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file()):
    raise RuntimeError('Only explicitly approved GPU-hidden rental CPU tests')
import torch
import mc2_component_gradients_v1 as helper


class ToyTraining:
    @staticmethod
    def least_squares_student_loss(scores):
        return sum(((score - 1.0) ** 2).mean() for score in scores)

    @staticmethod
    def feature_matching_loss(real, fake):
        return sum(torch.abs(r.detach() - f).mean() for rs, fs in zip(real, fake) for r, f in zip(rs, fs))


class Branch(torch.nn.Module):
    def __init__(self, scale):
        super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64), requires_grad=False)
        self.calls = []

    def forward(self, x):
        self.calls.append((torch.is_grad_enabled(), x.requires_grad))
        feature = x * self.scale
        return feature.mean(-1, keepdim=True), [feature, feature * .3]


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.branches = torch.nn.ModuleList([Branch(.1 + .015 * i) for i in range(8)])

    def loss_branches(self): return iter(self.branches)


def fixture(gt_shift=0.0, vocoder_scale=None):
    parameter = torch.nn.Parameter(torch.tensor([.2, -.15, .4], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([.9, .8], dtype=torch.float64))
    prediction = parameter.unsqueeze(0) * torch.tensor([[1.0, 1.5, .7], [.8, 1.1, 1.3]], dtype=torch.float64)
    reference = torch.tensor([[.1, -.3, .6], [.15, -.1, .5]], dtype=torch.float64) + gt_shift
    fake = prediction * (1.7 if vocoder_scale is None else vocoder_scale) + .04  # One fixed linear G, same S graph for every VJP.
    real = torch.tensor([[.8, .7, .9], [.6, .9, .8]], dtype=torch.float64, requires_grad=True)
    first = (torch.abs(prediction - .1) * torch.tensor([[.3], [.8]], dtype=torch.float64)).mean()
    second = (torch.abs(prediction + .2) * torch.tensor([[.7], [.4]], dtype=torch.float64)).mean()
    dist = (first + second) / 2.0  # Caller computes the actual two-loss mean.
    return dict(torch=torch, training=ToyTraining, prediction=prediction, reference_mel=reference,
        fake_waveform=fake, real_waveform=real, discriminator=Discriminator(), dist_loss=dist,
        named_parameters=[('used', parameter), ('unused', unused)], block_elements=2)


def direct_expected(args):
    # Independent straightforward graph for small tests, not a production claim.
    adv = args['prediction'].new_zeros(()); fm = adv.clone()
    for branch in args['discriminator'].loss_branches():
        with torch.no_grad(): _, real = branch(args['real_waveform'].detach())
        score, fake = branch(args['fake_waveform'])
        adv = adv + ToyTraining.least_squares_student_loss([score])
        fm = fm + ToyTraining.feature_matching_loss([real], [fake])
    params = tuple(p for _, p in args['named_parameters'])
    losses = dict(adv=adv, **{'2fm': 2.0 * fm, '45dist': 45.0 * args['dist_loss'],
                            'gt_mel_l1': torch.abs(args['prediction'] - args['reference_mel']).mean(),
                            'direct_joint': adv + 2.0 * fm + 45.0 * args['dist_loss']})
    banks = {}
    for name, loss in losses.items():
        values = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        banks[name] = {n: None if g is None else g.detach() for (n, _), g in zip(args['named_parameters'], values)}
    return banks


class ComponentTests(unittest.TestCase):
    def test_linear_model_matches_independent_component_and_joint_gradients(self):
        expected = direct_expected(fixture())
        stats = helper.blockwise_statistics(torch=torch, banks=expected, block_elements=1)
        actual = helper.component_gradients(**fixture())['parameter_gradients']
        for a in expected:
            self.assertAlmostEqual(actual['norms'][a], stats['norms'][a], places=11)
            for b in expected: self.assertAlmostEqual(actual['gram'][a][b], stats['gram'][a][b], places=10)
        self.assertLess(actual['difference']['max_absolute'], 1e-12)
        self.assertLess(actual['difference']['relative_l2'], 1e-12)
        self.assertAlmostEqual(actual['difference']['cosine'], 1.0, places=12)

    def test_exact_forward_and_autograd_budget_and_real_detach(self):
        args = fixture(); real = args['real_waveform']; result = helper.component_gradients(**args)
        self.assertEqual(result['actual_counts'], dict(D_branch_forwards=16, branch_autograd_calls=24, parameter_vjp_calls=5))
        for branch in args['discriminator'].branches: self.assertEqual(branch.calls, [(False, False), (True, True)])
        self.assertIsNone(real.grad)

    def test_none_support_and_zero_vectors_are_reported_not_nan(self):
        banks = {'split': {'unused': None, 'zero': torch.zeros(7)},
                 'direct_joint': {'unused': None, 'zero': torch.zeros(7)}}
        stats = helper.blockwise_statistics(torch=torch, banks=banks, block_elements=2, difference_pair=('split', 'direct_joint'))
        self.assertEqual(stats['norms'], {'split': 0.0, 'direct_joint': 0.0})
        self.assertIsNone(stats['cosines']['split']['direct_joint']); self.assertIsNone(stats['difference']['relative_l2'])
        self.assertTrue(stats['difference']['reference_norm_zero']); self.assertTrue(stats['difference']['none_support_equal'])
        banks['split']['unused'] = torch.zeros(3)
        stats = helper.blockwise_statistics(torch=torch, banks=banks, difference_pair=('split', 'direct_joint'))
        self.assertFalse(stats['difference']['none_support_equal']); self.assertEqual(stats['difference']['absolute_l2'], 0.0)
        json.dumps(stats, allow_nan=False)

    def test_current_mc2_scalar_mean_is_passed_through_without_new_targets(self):
        args = fixture(); expected = float(args['dist_loss'].detach()); result = helper.component_gradients(**args)
        self.assertEqual(result['losses']['dist_raw'], expected); self.assertEqual(result['losses']['dist_weighted'], 45.0 * expected)
        gradients = direct_expected(fixture())['45dist']['used']
        self.assertAlmostEqual(result['parameter_gradients']['norms']['45dist'], float(gradients.norm()), places=12)

    def test_GT_changes_only_diagnostic_gradient_and_not_joint(self):
        a = helper.component_gradients(**fixture()); b = helper.component_gradients(**fixture(gt_shift=10.0))
        for label in ('adv', '2fm', '45dist', 'direct_joint', 'split'):
            self.assertEqual(a['parameter_gradients']['norms'][label], b['parameter_gradients']['norms'][label])
        self.assertNotEqual(a['parameter_gradients']['gram']['gt_mel_l1']['direct_joint'], b['parameter_gradients']['gram']['gt_mel_l1']['direct_joint'])
        self.assertTrue(a['direct_joint_excludes_GT'])

    def test_split_is_built_after_direct_joint_not_used_to_compute_it(self):
        # Source-level ordering check: do not monkeypatch Torch or production.
        import ast
        tree = ast.parse(Path(helper.__file__).read_text())
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'component_gradients')
        pull = [x for x in ast.walk(function) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == 'pull']
        self.assertEqual([x.args[0].value for x in sorted(pull, key=lambda x: x.lineno)], ['adv', '2fm', '45dist', 'gt_mel_l1', 'direct_joint'])
        joint = next(x for x in pull if x.args[0].value == 'direct_joint')
        self.assertEqual(ast.dump(joint.args[2]), ast.dump(ast.parse('(wave_joint, dist_loss.new_tensor(45.0))', mode='eval').body))
        split = next(x for x in ast.walk(function) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == '_sum_components')
        self.assertGreater(split.lineno, joint.lineno)

    def test_difference_above_old_threshold_is_reported_without_rejection(self):
        banks = {'split': {'p': torch.tensor([3.0, 4.0])}, 'direct_joint': {'p': torch.tensor([1.0, 0.0])}}
        result = helper.blockwise_statistics(torch=torch, banks=banks, block_elements=1, difference_pair=('split', 'direct_joint'))
        self.assertEqual(result['difference']['max_absolute'], 4.0)
        self.assertAlmostEqual(result['difference']['relative_l2'], 20.0 ** .5)
        self.assertAlmostEqual(result['difference']['cosine'], .6)

    def test_blockwise_float64_dot_matches_small_direct_reference(self):
        a = torch.tensor([1e6, .1, -3.0, 9.0, -1e6], dtype=torch.float32)
        b = torch.tensor([.1, -2.0, 7.0, .3, 1.0], dtype=torch.float32)
        banks = {'a': {'p': a}, 'b': {'p': b}}
        for block in (1, 2, 3, 99):
            stats = helper.blockwise_statistics(torch=torch, banks=banks, block_elements=block)
            self.assertAlmostEqual(stats['gram']['a']['b'], float(torch.dot(a.double(), b.double())), places=7)

    def test_NaN_loss_input_and_gradient_bank_are_rejected(self):
        args = fixture(); args['dist_loss'] = args['dist_loss'] * float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): helper.component_gradients(**args)
        args = fixture(); args['real_waveform'] = args['real_waveform'] * float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): helper.component_gradients(**args)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            helper.blockwise_statistics(torch=torch, banks={'x': {'p': torch.tensor([float('nan')])}})

    def test_finite_loss_with_nonfinite_backward_is_rejected(self):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x): return x.clone()
            @staticmethod
            def backward(ctx, gradient): return gradient * float('nan')
        args = fixture(); args['dist_loss'] = BadGradient.apply(args['prediction']).mean()
        with self.assertRaisesRegex(ValueError, 'Nonfinite parameter gradient'): helper.component_gradients(**args)
        self.assertTrue(all(p.grad is None for _, p in args['named_parameters']))

    def test_parameters_modes_RNG_and_parameter_grads_remain_unchanged(self):
        vocoder_scale = torch.nn.Parameter(torch.tensor(1.7, dtype=torch.float64), requires_grad=False)
        args = fixture(vocoder_scale=vocoder_scale); student = [p for _, p in args['named_parameters']]; discriminator = args['discriminator']
        params = student + list(discriminator.parameters()) + [vocoder_scale]; before = [p.detach().clone() for p in params]
        modes = [m.training for m in discriminator.modules()]; rng = torch.get_rng_state().clone()
        result = helper.component_gradients(**args)
        for p, old in zip(params, before): self.assertTrue(torch.equal(p, old)); self.assertIsNone(p.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng)); self.assertEqual(modes, [m.training for m in discriminator.modules()])
        for label, absent in result['parameter_gradients']['none_parameter_names'].items(): self.assertEqual(absent, ['unused'])
        json.dumps(result, allow_nan=False)

    def test_structure_and_existing_gradients_refuse_without_clearing_them(self):
        args = fixture(); args['named_parameters'][0][1].grad = torch.ones(3, dtype=torch.float64)
        saved = args['named_parameters'][0][1].grad.clone()
        with self.assertRaisesRegex(ValueError, 'absent parameter.grad'): helper.component_gradients(**args)
        self.assertTrue(torch.equal(args['named_parameters'][0][1].grad, saved))
        args = fixture(); args['discriminator'].branches[0].scale.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'must be frozen'): helper.component_gradients(**args)
        with self.assertRaisesRegex(ValueError, 'names differ'):
            helper.blockwise_statistics(torch=torch, banks={'a': {'x': None}, 'b': {'y': None}})


if __name__ == '__main__': unittest.main(verbosity=2)
