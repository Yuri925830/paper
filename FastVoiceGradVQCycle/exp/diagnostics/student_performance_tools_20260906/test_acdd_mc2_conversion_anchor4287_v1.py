"""待运行的租用 CPU toy 测试；不加载真实权重、语料或GPU，也不写断点。"""
import ast
import copy
import inspect
import math
import os
from pathlib import Path
from types import SimpleNamespace, FunctionType
import unittest
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only GPU-hidden rental CPU tests may run')
import acdd_mc2_conversion_anchor4287_v1 as n
torch = n.torch
from voicegrad.diffusion import VoiceGradDiffusion
from fastvoicegrad.objectives import reverse_mean_to_zero


class ToyEpsilon(torch.nn.Module):
    def __init__(self, diffusion, trainable):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.01), requires_grad=trainable)
        self.speaker_scale = torch.nn.Parameter(torch.tensor(0.002), requires_grad=trainable)
        self.register_buffer('alphas', diffusion.alphas_cumprod.clone())
        self.calls = 0
    def forward(self, x, times, speaker, bnf):
        self.calls += 1
        sigma = torch.sqrt((1 - self.alphas[times]).clamp_min(0)).view(-1, 1, 1)
        return sigma * x + self.scale * x + self.speaker_scale * speaker[:, :1, None]


class ToyVocoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(0.1), requires_grad=False); self.calls = 0
    def forward(self, x):
        self.calls += 1
        return self.scale * x.mean(1, keepdim=True)


class ToyD(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(0.2)); self.calls = 0
    def forward(self, wave):
        self.calls += 1
        return self.scale * wave


class RecordingAdam(torch.optim.Adam):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr=lr, betas=(0.5, 0.9)); self.step_returns = 0; self.saved_gradients = None
    def step(self, *args, **kwargs):
        self.saved_gradients = [p.grad.detach().clone() for group in self.param_groups for p in group['params']]
        value = super().step(*args, **kwargs); self.step_returns += 1; return value


def toy_d_backward(model, real, fake):
    total = real.new_zeros(())
    for _ in range(8):
        loss = ((model(real) - 1).square() + model(fake).square()).mean()
        loss.backward(); total += loss.detach()
    return total


def toy_student_d_gradient(model, real, fake, lambda_fm):
    wave = fake.detach().clone().requires_grad_(True)
    adv = wave.new_zeros(()); fm = wave.new_zeros(())
    for _ in range(8):
        real_feature = model(real.detach()).detach(); fake_feature = model(wave)
        adv = adv + (fake_feature - 1).square().mean()
        fm = fm + (fake_feature - real_feature).abs().mean()
    grad, = torch.autograd.grad(adv + lambda_fm * fm, wave)
    return adv.detach(), fm.detach(), grad.detach()


class ConversionAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def fixture(self):
        diffusion = VoiceGradDiffusion(1000, offset=0.008)
        student = ToyEpsilon(diffusion, True).train()
        teacher = ToyEpsilon(diffusion, False).eval()
        vocoder, discriminator = ToyVocoder().eval(), ToyD().train()
        source_generator = torch.Generator().manual_seed(901)
        batch = {'mel': torch.randn((32, 80, 128), generator=source_generator) * 0.3,
                 'bnf': torch.randn((32, 144, 128), generator=source_generator),
                 'spk_emb': torch.randn((32, 256), generator=source_generator),
                 'mel_len': torch.full((32,), 128, dtype=torch.long),
                 'spk_name': ['speaker_%02d' % i for i in range(32)],
                 'utt_id': ['speaker_%02d_001' % i for i in range(32)]}
        config = SimpleNamespace(batch_size=32, diffusion_steps=1000, student_start_step=950,
            score_target='clean_x0', score_weight='sqrt_alpha_bar', lambda_fm=2.0, lambda_dist=45.0)
        s_opt, d_opt = RecordingAdam(student.parameters(), 5e-5), RecordingAdam(discriminator.parameters(), 2e-4)
        positional = (batch, teacher, student, vocoder, discriminator, diffusion, s_opt, d_opt,
                      torch.zeros((1, 80, 1)), torch.ones((1, 80, 1)), torch.device('cpu'), config)
        extra = dict(global_step=4288, mc_receipt={}, anchor_weight=1.0,
            source_indices=list(range(52, 84)), retained_train_indices=frozenset(range(52, 39387)),
            excluded_train_indices=frozenset(range(52)), shuffle_seed=80701, reverse_seed=80702, anchor_receipt={})
        return positional, extra

    def run_batch(self, positional, extra, baseline=False):
        with mock.patch.dict(n._namespace, serial_discriminator_backward=toy_d_backward,
                             serial_student_discriminator_gradient=toy_student_d_gradient):
            if baseline:
                original = n._core.train_batch_mc2
                function = FunctionType(original.__code__, n._namespace, original.__name__, original.__defaults__, original.__closure__)
                function.__kwdefaults__ = original.__kwdefaults__
                return function(*positional, global_step=extra['global_step'], mc_receipt=extra['mc_receipt'])
            return n.train_batch_mc2_conversion_anchor(*positional, **extra)

    def prepare(self, positional, extra, noise=None):
        batch, teacher, _, _, _, diffusion, _, _, _, _, device, config = positional
        if noise is None: noise = torch.zeros_like(batch['mel'])
        return n.prepare_anchor(batch, batch['mel'], batch['bnf'], batch['spk_emb'], noise,
            diffusion, teacher, device, config, extra['global_step'], extra['anchor_weight'],
            extra['source_indices'], extra['retained_train_indices'], extra['excluded_train_indices'],
            extra['shuffle_seed'], extra['reverse_seed'], extra['anchor_receipt'])

    def test_ast_original_reconstruction_body_unchanged_except_declared_additions(self):
        actual = copy.deepcopy(n._augmented_ast); expected = copy.deepcopy(n._original_ast)
        actual.name = expected.name
        actual.args.kwonlyargs = actual.args.kwonlyargs[:-7]; actual.args.kw_defaults = actual.args.kw_defaults[:-7]
        actual.body = [x for x in actual.body if not (isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name)
                       and x.targets[0].id in ('anchor_context', 'anchor_loss'))]
        returned = [x for x in actual.body if isinstance(x, ast.Return)][0].value
        returned.keys = returned.keys[:-4]; returned.values = returned.values[:-4]
        self.assertEqual(ast.dump(actual), ast.dump(expected))
        self.assertEqual((n._core.START, n._core.STOP), (3327, 3447))
        self.assertEqual((n._namespace['START'], n._namespace['STOP']), (4287, 4527))
        self.assertIs(n._namespace['make_student_prediction'], n._core.make_student_prediction)

    def test_all_anchor_arguments_have_no_defaults_or_adopted_coefficient(self):
        signature = inspect.signature(n.train_batch_mc2_conversion_anchor)
        for name in ('anchor_weight', 'source_indices', 'retained_train_indices', 'excluded_train_indices',
                     'shuffle_seed', 'reverse_seed', 'anchor_receipt'):
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)

    def test_fixture_matches_exact_production_dataset_collated_schema(self):
        positional, extra = self.fixture(); batch = positional[0]
        self.assertEqual(set(batch), {'mel', 'bnf', 'spk_emb', 'mel_len', 'spk_name', 'utt_id'})
        self.assertEqual(tuple(batch['mel'].shape), (32, 80, 128))
        self.assertEqual(tuple(batch['bnf'].shape), (32, 144, 128))
        self.assertEqual(tuple(batch['spk_emb'].shape), (32, 256))
        self.assertEqual(tuple(batch['mel_len'].shape), (32,)); self.assertEqual(batch['mel_len'].dtype, torch.long)
        self.assertEqual(len(batch['spk_name']), 32); self.assertEqual(len(batch['utt_id']), 32)
        tensors = n._core.production._slice_batch(batch, torch.device('cpu'))
        for name in ('mel', 'bnf', 'spk_emb'): self.assertTrue(torch.equal(tensors[name], batch[name]))
        self.prepare(positional, extra)
        self.assertEqual(extra['anchor_receipt']['source_speakers'], batch['spk_name'])

    def test_shuffle_is_deterministic_bijective_cross_speaker_and_does_not_use_global_rng(self):
        speakers = ['a'] * 8 + ['b'] * 8 + ['c'] * 8 + ['d'] * 8
        before = n.random.getstate(); permutation = n.speaker_shuffle(speakers, 44)
        self.assertEqual(before, n.random.getstate()); self.assertEqual(permutation, n.speaker_shuffle(speakers, 44))
        self.assertEqual(sorted(permutation), list(range(32)))
        self.assertTrue(all(speakers[i] != speakers[j] for i, j in enumerate(permutation)))
        for impossible in (['a'] * 32, ['a'] * 17 + ['b'] * 15):
            with self.assertRaisesRegex(RuntimeError, 'No valid'): n.speaker_shuffle(impossible, 44)

    def test_invalid_weight_scope_seed_and_reused_receipt_fail_before_teacher_or_D(self):
        mutations = [('anchor_weight', 0), ('anchor_weight', float('nan')), ('anchor_weight', float('inf')),
            ('anchor_weight', True), ('anchor_weight', None), ('reverse_seed', -1),
            ('source_indices', list(range(32))), ('source_indices', [52] * 32),
            ('excluded_train_indices', frozenset(range(51))), ('anchor_receipt', {'already_used': True})]
        for key, value in mutations:
            positional, extra = self.fixture(); extra[key] = value
            with self.subTest(key=key, value=str(value)), self.assertRaises((ValueError, RuntimeError)):
                self.prepare(positional, extra)
            self.assertEqual(positional[1].calls, 0); self.assertEqual(positional[7].step_returns, 0)

    def test_source_to_teacher_and_student_q950_is_shared_and_teacher_rng_is_private(self):
        positional, extra = self.fixture(); before = torch.get_rng_state().clone()
        context = self.prepare(positional, extra)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        receipt = extra['anchor_receipt']; self.assertEqual(receipt['teacher_forward_calls'], 30)
        self.assertEqual(receipt['shared_q950'], n._core.tensor_identity(context['x_start']))
        self.assertTrue(torch.equal(context['bnf'], positional[0]['bnf']))
        self.assertTrue(torch.equal(context['speaker'], positional[0]['spk_emb'][receipt['target_permutation']]))
        self.assertFalse(context['teacher_target'].requires_grad); self.assertEqual(len(positional[1]._forward_pre_hooks), 0)

    def test_teacher30_reverse_seed_changes_target_but_not_shared_input_or_global_rng(self):
        positional_a, extra_a = self.fixture(); positional_b, extra_b = self.fixture(); extra_b['reverse_seed'] += 1
        a = self.prepare(positional_a, extra_a); b = self.prepare(positional_b, extra_b)
        self.assertTrue(torch.equal(a['x_start'], b['x_start']))
        self.assertFalse(torch.equal(a['teacher_target'], b['teacher_target']))

    def test_wrong_teacher_start_or_conditions_fail_and_hooks_are_removed(self):
        positional, extra = self.fixture(); diffusion = positional[5]; original = diffusion.sample
        def wrong(model, x, speaker, bnf, **kwargs): return original(model, x + 0.1, speaker, bnf, **kwargs)
        with mock.patch.object(diffusion, 'sample', side_effect=wrong), self.assertRaisesRegex(RuntimeError, 'shares declared'):
            self.prepare(positional, extra)
        self.assertEqual(len(positional[1]._forward_pre_hooks), 0); self.assertEqual(positional[7].step_returns, 0)

    def test_accidental_global_rng_use_in_teacher_fails_before_D(self):
        positional, extra = self.fixture(); diffusion = positional[5]; original = diffusion.sample
        def global_draw(*args, **kwargs): torch.rand(1); return original(*args, **kwargs)
        with mock.patch.object(diffusion, 'sample', side_effect=global_draw), self.assertRaisesRegex(RuntimeError, 'advanced'):
            self.prepare(positional, extra)
        self.assertEqual(positional[7].step_returns, 0)

    def test_joint_step_has_one_D_one_S_and_preserves_all_original_rec_losses_and_draws(self):
        baseline, old_extra = self.fixture(); candidate, extra = self.fixture()
        torch.manual_seed(7701); before = torch.get_rng_state().clone()
        old_losses = self.run_batch(baseline, old_extra, baseline=True); old_rng = torch.get_rng_state().clone()
        torch.set_rng_state(before); new_losses = self.run_batch(candidate, extra)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        for name in old_losses: self.assertEqual(old_losses[name], new_losses[name], name)
        self.assertEqual(old_extra['mc_receipt'], extra['mc_receipt'])
        self.assertTrue(torch.equal(baseline[4].scale, candidate[4].scale))
        self.assertEqual((candidate[6].step_returns, candidate[7].step_returns), (1, 1))
        self.assertEqual((candidate[1].calls, candidate[2].calls, candidate[3].calls, candidate[4].calls), (32, 3, 3, 32))
        self.assertEqual((baseline[1].calls, baseline[2].calls, baseline[3].calls, baseline[4].calls), (2, 2, 3, 32))
        self.assertGreater(new_losses['student_joint_total'], new_losses['student_total'])
        self.assertTrue(all(p.grad is None for model in (candidate[1], candidate[3], candidate[4]) for p in model.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(baseline[6].saved_gradients, candidate[6].saved_gradients)))

    def test_weight_scales_only_added_student_gradient_not_D_or_original_rec(self):
        baseline, old_extra = self.fixture(); a, ea = self.fixture(); b, eb = self.fixture(); eb['anchor_weight'] = 2.0
        torch.manual_seed(303); state = torch.get_rng_state().clone()
        self.run_batch(baseline, old_extra, baseline=True)
        torch.set_rng_state(state); la = self.run_batch(a, ea)
        torch.set_rng_state(state); lb = self.run_batch(b, eb)
        for g0, g1, g2 in zip(baseline[6].saved_gradients, a[6].saved_gradients, b[6].saved_gradients):
            self.assertTrue(torch.allclose(g2 - g0, 2 * (g1 - g0), rtol=2e-4, atol=2e-5))
        self.assertEqual(la['student_total'], lb['student_total'])
        self.assertEqual(la['conversion_anchor_raw'], lb['conversion_anchor_raw'])
        self.assertEqual(lb['conversion_anchor_weighted'], 2 * la['conversion_anchor_weighted'])
        self.assertTrue(torch.equal(a[4].scale, b[4].scale))

    def test_valid_frame_mask_excludes_padding_from_new_anchor_only(self):
        positional, extra = self.fixture(); positional[0]['mel_len'][:] = 64
        context = self.prepare(positional, extra)
        prediction = reverse_mean_to_zero(positional[2], positional[5], context['x_start'], context['times'], context['speaker'], context['bnf'])
        expected = (prediction[:, :, :64] - context['teacher_target'][:, :, :64]).abs().flatten(1).mean(1).mean()
        context['teacher_target'][:, :, 64:] += 1000
        loss = n.conversion_anchor_backward(positional[2], positional[5], context, 1.0, extra['anchor_receipt'], torch.tensor(1.0))
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_anchor_backward_accumulates_into_existing_nonzero_parameter_gradients(self):
        positional, extra = self.fixture(); context = self.prepare(positional, extra)
        student, diffusion = positional[2], positional[5]
        prediction = reverse_mean_to_zero(student, diffusion, context['x_start'], context['times'], context['speaker'], context['bnf'])
        loss = (prediction - context['teacher_target']).abs().flatten(1).mean(1).mean()
        parameters = list(student.parameters()); anchor_gradients = torch.autograd.grad(loss, parameters)
        existing = [torch.full_like(p, 3.0 + i) for i, p in enumerate(parameters)]
        for p, value in zip(parameters, existing): p.grad = value.clone()
        n.conversion_anchor_backward(student, diffusion, context, 0.7, extra['anchor_receipt'], torch.tensor(1.0))
        for p, before, additional in zip(parameters, existing, anchor_gradients):
            self.assertTrue(torch.allclose(p.grad, before + 0.7 * additional, rtol=2e-5, atol=2e-6))
        self.assertEqual(positional[6].step_returns, 0)

    def test_teacher_target_failure_occurs_before_any_optimizer_step(self):
        positional, extra = self.fixture()
        with mock.patch.object(positional[5], 'sample', side_effect=RuntimeError('target failure')), self.assertRaisesRegex(RuntimeError, 'target failure'):
            self.run_batch(positional, extra)
        self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 0))
        self.assertEqual(len(positional[1]._forward_pre_hooks), 0)

    def test_added_backward_failure_preserves_real_D_only_cursor_for_future_driver(self):
        positional, extra = self.fixture()
        with mock.patch.dict(n._namespace, conversion_anchor_backward=mock.Mock(side_effect=RuntimeError('anchor failure'))), \
             self.assertRaisesRegex(RuntimeError, 'anchor failure'):
            self.run_batch(positional, extra)
        self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 1))

    def test_teacher_requires_frozen_eval_and_finite_endpoint(self):
        positional, extra = self.fixture(); positional[1].train()
        with self.assertRaisesRegex(RuntimeError, 'fully frozen'): self.prepare(positional, extra)
        positional, extra = self.fixture(); original = positional[5].sample
        def bad(*args, **kwargs): return original(*args, **kwargs) * float('nan')
        with mock.patch.object(positional[5], 'sample', side_effect=bad), self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            self.prepare(positional, extra)

    def test_helper_has_no_optimizer_creation_checkpoint_file_write_or_launcher(self):
        module = ast.parse(Path(n.__file__).read_bytes())
        calls = [ast.dump(x.func) for x in ast.walk(module) if isinstance(x, ast.Call)]
        for forbidden in ('torch.save', 'torch.optim.Adam', 'subprocess.run', 'subprocess.Popen'):
            self.assertNotIn(ast.dump(ast.parse(forbidden, mode='eval').body), calls)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__': unittest.main(verbosity=2)
