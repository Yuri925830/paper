"""租用 CPU toy 测试草稿：检查感知辅助接入，不加载真实WavLM或真实模型，不写断点。"""
import ast
import copy
import inspect
import os
from pathlib import Path
from types import FunctionType
import unittest
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only GPU-hidden rental CPU tests may run')
import numpy as np
import acdd_mc2_perceptual_anchor4287_v1 as n
import test_acdd_mc2_conversion_anchor4287_v1 as fixtures
from fastvoicegrad.objectives import reverse_mean_to_zero
torch = n.torch


class ToyVocoder256(torch.nn.Module):
    def __init__(self, hop=256, disconnect_on=None):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.hop, self.disconnect_on, self.calls, self.trace = hop, disconnect_on, 0, []

    def forward(self, mel):
        self.calls += 1
        wave = (self.scale * mel.mean(1, keepdim=True)).repeat_interleave(self.hop, dim=-1)
        if self.calls == self.disconnect_on: wave = wave.detach()
        self.trace.append(dict(mel=mel.detach().clone(), grad_enabled=torch.is_grad_enabled(),
            input_requires_grad=mel.requires_grad, output_requires_grad=wave.requires_grad))
        return wave


class CheapPerceptualLoss:
    """仅检查接口：句均值差平方模拟feature，波形L1模拟spectrum；不声称等于真实LMOS。"""
    def __init__(self):
        self.calls = 0; self.generated = None; self.target = None; self.lengths = None

    def __call__(self, generated, target, valid_lengths):
        if target.requires_grad or target.grad_fn is not None:
            raise RuntimeError('Toy criterion received an attached teacher waveform')
        self.calls += 1; self.generated = generated; self.target = target
        self.lengths = valid_lengths.detach().clone()
        if generated.requires_grad: generated.retain_grad()
        feature, spectrum = [], []
        for index, length in enumerate(valid_lengths.tolist()):
            left, right = generated[index, :, :length], target[index, :, :length]
            feature.append((left.mean() - right.mean()).square())
            spectrum.append((left - right).abs().mean())
        feature, spectrum = torch.stack(feature), torch.stack(spectrum)
        per_utterance = 100 * feature + spectrum
        return dict(loss=per_utterance.mean(), feature_mse=feature.mean(), stft_l1=spectrum.mean(),
            per_utterance_loss=per_utterance)


class PerceptualAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def fixture(self):
        original, extra = fixtures.ConversionAnchorTests().fixture()
        positional = list(original); positional[3] = ToyVocoder256().eval()
        extra['perceptual_loss'] = CheapPerceptualLoss()
        return positional, extra

    def run_batch(self, positional, extra, baseline=False):
        with mock.patch.dict(n._namespace, serial_discriminator_backward=fixtures.toy_d_backward,
                serial_student_discriminator_gradient=fixtures.toy_student_d_gradient):
            if baseline:
                original = n._anchor._core.train_batch_mc2
                function = FunctionType(original.__code__, n._namespace, original.__name__, original.__defaults__, original.__closure__)
                function.__kwdefaults__ = original.__kwdefaults__
                return function(*positional, global_step=extra['global_step'], mc_receipt=extra['mc_receipt'])
            return n.train_batch_mc2_perceptual_anchor(*positional, **extra)

    def prepare(self, positional, extra):
        batch, teacher, _, _, _, diffusion, _, _, _, _, device, config = positional
        return n._anchor.prepare_anchor(batch, batch['mel'], batch['bnf'], batch['spk_emb'], torch.zeros_like(batch['mel']),
            diffusion, teacher, device, config, extra['global_step'], extra['anchor_weight'], extra['source_indices'],
            extra['retained_train_indices'], extra['excluded_train_indices'], extra['shuffle_seed'], extra['reverse_seed'], extra['anchor_receipt'])

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor): self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, np.ndarray): np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected: self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected)); self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected): self.assert_tree_equal(left, right)
        else: self.assertEqual(actual, expected)

    def test_ast_is_exact_original_rec_body_after_only_declared_additions_removed(self):
        actual, expected = copy.deepcopy(n._augmented_ast), copy.deepcopy(n._anchor._original_ast)
        actual.name = expected.name
        actual.args.kwonlyargs = actual.args.kwonlyargs[:-8]; actual.args.kw_defaults = actual.args.kw_defaults[:-8]
        actual.body = [node for node in actual.body if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ('anchor_context', 'anchor_loss'))]
        returned = [node.value for node in actual.body if isinstance(node, ast.Return)][0]
        returned.keys, returned.values = returned.keys[:-4], returned.values[:-4]
        self.assertEqual(ast.dump(actual), ast.dump(expected))
        signature = inspect.signature(n.train_batch_mc2_perceptual_anchor)
        for name in ('anchor_weight', 'source_indices', 'retained_train_indices', 'excluded_train_indices',
                'shuffle_seed', 'reverse_seed', 'anchor_receipt', 'perceptual_loss'):
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)

    def test_ast_passes_actual_mel_mean_std_and_places_auxiliary_between_rec_backward_and_checks(self):
        body = n._augmented_ast.body
        positions = [i for i, node in enumerate(body) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'anchor_loss']
        self.assertEqual(len(positions), 1); index = positions[0]
        self.assertEqual(ast.dump(body[index-1].value.func), ast.dump(ast.parse('torch.autograd.backward', mode='eval').body))
        self.assertEqual(body[index+1].targets[0].id, 'student_gradients')
        self.assertEqual([arg.id for arg in body[index].value.args], ['student', 'diffusion', 'anchor_context', 'anchor_weight',
            'anchor_receipt', 'total_loss', 'vocoder', 'mel_mean', 'mel_std', 'perceptual_loss'])
        prepared = [i for i, node in enumerate(body) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'anchor_context'][0]
        first_d = [i for i, node in enumerate(body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'set_requires_grad'][0]
        self.assertLess(prepared, first_d)

    def test_private_namespace_rebinds_internal_mc2_functions_without_mutating_ancestor_or_production(self):
        ancestor = dict(n._anchor._namespace); core = dict(vars(n._anchor._core))
        production = dict(vars(n._anchor._core.production))
        function, namespace, _ = n._build()
        self.assertIs(function.__globals__, namespace); self.assertIsNot(namespace, n._anchor._namespace)
        for name in ('validate_request', 'second_score_draws', 'mc2_score_distillation_loss'):
            self.assertIs(namespace[name].__globals__, namespace)
            self.assertIsNot(namespace[name], ancestor[name]); self.assertIs(namespace[name].__code__, ancestor[name].__code__)
        for actual, expected in ((n._anchor._namespace, ancestor), (vars(n._anchor._core), core),
                (vars(n._anchor._core.production), production)):
            self.assertEqual(set(actual), set(expected))
            self.assertTrue(all(actual[key] is value for key, value in expected.items()))
        self.assertEqual((n.START, n.STOP), (4287, 4527))
        self.assertEqual((n._anchor._core.START, n._anchor._core.STOP), (3327, 3447))

    def test_exact_production_schema_and_train_only_cross_speaker_pairing_are_retained(self):
        positional, extra = self.fixture(); batch = positional[0]
        self.assertEqual(set(batch), {'mel', 'bnf', 'spk_emb', 'mel_len', 'spk_name', 'utt_id'})
        self.assertEqual(tuple(batch['mel'].shape), (32, 80, 128)); self.assertEqual(tuple(batch['bnf'].shape), (32, 144, 128))
        self.assertEqual(tuple(batch['spk_emb'].shape), (32, 256)); self.assertEqual(batch['mel_len'].dtype, torch.long)
        context = self.prepare(positional, extra); receipt = extra['anchor_receipt']
        self.assertTrue(all(a != b for a, b in zip(receipt['source_speakers'], receipt['target_speakers'])))
        self.assertTrue(all(i >= 52 for i in receipt['source_indices'] + receipt['target_indices']))
        self.assertEqual(receipt['teacher_forward_calls'], 30); self.assertFalse(context['teacher_target'].requires_grad)
        self.assertTrue(torch.equal(context['bnf'], batch['bnf']))
        self.assertEqual(receipt['shared_q950'], n._anchor._core.tensor_identity(context['x_start']))

    def test_joint_step_preserves_exact_rec_losses_rng_D_update_and_has_one_S_update(self):
        baseline, old_extra = self.fixture(); candidate, extra = self.fixture()
        torch.manual_seed(77201); before = n._anchor._core.production.capture_rng_state()
        old_losses = self.run_batch(baseline, old_extra, baseline=True)
        old_rng = n._anchor._core.production.capture_rng_state()
        n._anchor._core.production.restore_rng_state(before, torch.device('cpu'))
        new_losses = self.run_batch(candidate, extra)
        self.assert_tree_equal(n._anchor._core.production.capture_rng_state(), old_rng)
        for key in old_losses: self.assertEqual(new_losses[key], old_losses[key], key)
        self.assertEqual(extra['mc_receipt'], old_extra['mc_receipt'])
        self.assert_tree_equal(candidate[4].state_dict(), baseline[4].state_dict())
        self.assert_tree_equal(candidate[7].state_dict(), baseline[7].state_dict())
        self.assertEqual((candidate[6].step_returns, candidate[7].step_returns), (1, 1))
        self.assertEqual((baseline[1].calls, baseline[2].calls, baseline[3].calls, baseline[4].calls), (2, 2, 3, 32))
        self.assertEqual((candidate[1].calls, candidate[2].calls, candidate[3].calls, candidate[4].calls), (32, 3, 5, 32))
        self.assertEqual([item['grad_enabled'] for item in candidate[3].trace], [False, False, True, False, True])
        self.assertEqual([item['output_requires_grad'] for item in candidate[3].trace], [False, False, True, False, True])
        self.assertEqual(extra['perceptual_loss'].calls, 1)
        self.assertTrue(all(p.grad is None for model in candidate[1:5] for p in model.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(candidate[6].saved_gradients, baseline[6].saved_gradients)))
        self.assertAlmostEqual(new_losses['student_joint_total'], new_losses['student_total'] + new_losses['conversion_perceptual_weighted'])

    def test_weight_linearly_scales_only_added_gradient_preserving_rec_and_D(self):
        baseline, old_extra = self.fixture(); first, e1 = self.fixture(); second, e2 = self.fixture()
        e1['anchor_weight'], e2['anchor_weight'] = 0.5, 1.0
        torch.manual_seed(81002); before = torch.get_rng_state().clone()
        base_losses = self.run_batch(baseline, old_extra, baseline=True)
        torch.set_rng_state(before); a = self.run_batch(first, e1)
        torch.set_rng_state(before); b = self.run_batch(second, e2)
        for g0, g1, g2 in zip(baseline[6].saved_gradients, first[6].saved_gradients, second[6].saved_gradients):
            torch.testing.assert_close(g2 - g0, 2 * (g1 - g0), rtol=2e-4, atol=2e-5)
        for key, value in base_losses.items(): self.assertEqual(a[key], value); self.assertEqual(b[key], value)
        self.assertEqual(a['conversion_perceptual_raw'], b['conversion_perceptual_raw'])
        self.assertEqual(b['conversion_perceptual_weighted'], 2 * a['conversion_perceptual_weighted'])
        self.assert_tree_equal(first[4].state_dict(), second[4].state_dict())
        self.assertEqual((first[6].step_returns, first[7].step_returns, second[6].step_returns, second[7].step_returns), (1, 1, 1, 1))

    def test_direct_backward_accumulates_into_existing_gradients_and_uses_actual_denormalization(self):
        positional, extra = self.fixture(); context = self.prepare(positional, extra)
        student, vocoder, diffusion = positional[2], positional[3], positional[5]
        mean = torch.linspace(-1.3, 0.7, 80).view(1, 80, 1); std = torch.linspace(0.2, 1.8, 80).view(1, 80, 1)
        prediction = reverse_mean_to_zero(student, diffusion, context['x_start'], context['times'], context['speaker'], context['bnf'])
        with torch.no_grad(): target = vocoder(context['teacher_target'] * std + mean)
        generated = vocoder(prediction * std + mean)
        criterion = CheapPerceptualLoss(); reference = criterion(generated, target, context['valid_lengths'] * 256)
        parameters = list(student.parameters()); auxiliary = torch.autograd.grad(reference['loss'], parameters)
        existing = [torch.full_like(parameter, 3.0 + index) for index, parameter in enumerate(parameters)]
        for parameter, value in zip(parameters, existing): parameter.grad = value.clone()
        target_leaf = context['teacher_target'].detach().clone().requires_grad_(True); context['teacher_target'] = target_leaf
        loss = n.perceptual_backward(student, diffusion, context, 0.7, extra['anchor_receipt'], torch.tensor(2.0),
            vocoder, mean, std, extra['perceptual_loss'])
        torch.testing.assert_close(loss, reference['loss'].detach(), rtol=0, atol=0)
        for parameter, before, gradient in zip(parameters, existing, auxiliary):
            torch.testing.assert_close(parameter.grad, before + 0.7 * gradient, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(vocoder.trace[-2]['mel'], target_leaf.detach() * std + mean, rtol=0, atol=0)
        torch.testing.assert_close(vocoder.trace[-1]['mel'], prediction.detach() * std + mean, rtol=0, atol=0)
        self.assertIsNone(target_leaf.grad); self.assertIsNone(vocoder.scale.grad)
        self.assertGreater(float(extra['perceptual_loss'].generated.grad.abs().sum()), 0)
        self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 0))

    def test_perceptual_valid_lengths_are_original_frames_times256_and_target_stopped(self):
        positional, extra = self.fixture(); positional[0]['mel_len'][:16] = 64
        result = self.run_batch(positional, extra); criterion = extra['perceptual_loss']
        self.assertTrue(torch.equal(criterion.lengths, positional[0]['mel_len'] * 256))
        self.assertEqual(tuple(criterion.target.shape), (32, 1, 32768))
        self.assertFalse(criterion.target.requires_grad); self.assertIsNone(criterion.target.grad_fn)
        self.assertEqual(len(extra['anchor_receipt']['per_utterance_loss']), 32)
        self.assertEqual(result['conversion_perceptual_raw'], extra['anchor_receipt']['anchor_raw'])
        self.assertTrue(extra['anchor_receipt']['anchor_backward_completed'])
        self.assertFalse(extra['anchor_receipt']['conversion_uses_discriminator'])
        self.assertFalse(extra['anchor_receipt']['raw_mel_L1_optimized'])

    def test_bad_train_scope_weight_or_teacher_preparation_fails_before_D_or_S(self):
        for key, value in (('source_indices', list(range(32))), ('anchor_weight', 0.0), ('anchor_weight', float('nan'))):
            positional, extra = self.fixture(); extra[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.run_batch(positional, extra)
            self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 0))
        positional, extra = self.fixture()
        with mock.patch.object(positional[5], 'sample', side_effect=RuntimeError('teacher preparation failed')), self.assertRaisesRegex(RuntimeError, 'teacher preparation'):
            self.run_batch(positional, extra)
        self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 0))
        self.assertEqual(len(positional[1]._forward_pre_hooks), 0)

    def test_criterion_failure_preserves_actual_D_only_halfstep_without_completing_S(self):
        positional, extra = self.fixture(); extra['perceptual_loss'] = mock.Mock(side_effect=RuntimeError('criterion failure'))
        with self.assertRaisesRegex(RuntimeError, 'criterion failure'): self.run_batch(positional, extra)
        self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 1))
        self.assertFalse(positional[6].state); self.assertTrue(positional[7].state)
        self.assertNotIn('anchor_backward_completed', extra['anchor_receipt'])
        self.assertEqual(len(positional[1]._forward_pre_hooks), 0)

    def test_disconnected_auxiliary_waveform_fails_after_D_but_before_S_step(self):
        positional, extra = self.fixture(); positional[3] = ToyVocoder256(disconnect_on=5).eval()
        with self.assertRaisesRegex(FloatingPointError, 'disconnected'): self.run_batch(positional, extra)
        self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 1))
        self.assertEqual(extra['perceptual_loss'].calls, 0)

    def test_wrong_vocoder_hop_is_rejected_without_a_completed_S_step(self):
        positional, extra = self.fixture(); positional[3] = ToyVocoder256(hop=128).eval()
        with self.assertRaisesRegex(RuntimeError, '256-hop'): self.run_batch(positional, extra)
        self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 1))

    def test_frozen_vocoder_requirement_is_checked_before_extra_conversion_forward(self):
        for mode in ('training', 'trainable', 'existing_gradient'):
            positional, extra = self.fixture(); context = self.prepare(positional, extra); vocoder = positional[3]
            if mode == 'training': vocoder.train()
            elif mode == 'trainable': vocoder.scale.requires_grad_(True)
            else: vocoder.scale.grad = torch.ones_like(vocoder.scale)
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'frozen/eval'):
                n.perceptual_backward(positional[2], positional[5], context, 1.0, extra['anchor_receipt'], torch.tensor(2.0),
                    vocoder, positional[8], positional[9], extra['perceptual_loss'])
            self.assertEqual(vocoder.calls, 0); self.assertEqual(positional[2].calls, 0)

    def test_loss_must_be_finite_differentiable_scalar_before_added_backward(self):
        for mode in ('detached', 'nonfinite', 'nonscalar'):
            positional, extra = self.fixture(); good = extra['perceptual_loss']
            def invalid(*args, selected=mode):
                result = good(*args)
                if selected == 'detached': result['loss'] = result['loss'].detach()
                elif selected == 'nonfinite': result['loss'] = result['loss'] * float('nan')
                else: result['loss'] = result['loss'].repeat(2)
                return result
            extra['perceptual_loss'] = invalid
            with self.subTest(mode=mode), self.assertRaisesRegex(FloatingPointError, 'scalar disconnected or nonfinite'):
                self.run_batch(positional, extra)
            self.assertEqual((positional[6].step_returns, positional[7].step_returns), (0, 1))
            self.assertNotIn('anchor_backward_completed', extra['anchor_receipt'])

    def test_no_optimizer_creation_checkpoint_writes_or_production_attribute_assignment(self):
        tree = ast.parse(Path(n.__file__).read_bytes())
        calls = [ast.dump(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for name in ('torch.load', 'torch.save', 'torch.optim.Adam', 'subprocess.run', 'subprocess.Popen'):
            self.assertNotIn(ast.dump(ast.parse(name, mode='eval').body), calls)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        # 原训练环境是Python3.8，没有ast.unparse；直接检查属性链的根名称。
                        root = target
                        while isinstance(root, ast.Attribute): root = root.value
                        self.assertFalse(isinstance(root, ast.Name) and root.id in ('_anchor', 'torch', 'production'))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__': unittest.main(verbosity=2)
