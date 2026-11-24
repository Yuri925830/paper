"""仅租用服务器CPU小模型测试：不读checkpoint，不跑真实特征/模型/评分。"""

import ast
import copy
import inspect
import os
import random
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Run only on Matpool rental CPU with CUDA_VISIBLE_DEVICES empty')

import numpy as np
import torch
from torch import nn
import acdd_mc2_train_batch_v1 as helper
import test_conversion_score_step as fixture


def make_fixture():
    args = fixture.make_fixture(torch.float32)
    args[6].param_groups[0]['lr'] = 5e-5
    return args


class MC2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_ast_clone_only_adds_validation_receipt_and_mc_score_adapter(self):
        original = ast.parse(inspect.getsource(helper.production.train_batch)).body[0]
        clone = ast.parse(inspect.getsource(helper.train_batch_mc2)).body[0]
        self.assertEqual([node.arg for node in clone.args.kwonlyargs], ['global_step', 'mc_receipt'])
        self.assertEqual(clone.args.kw_defaults, [None, None])
        self.assertEqual(clone.body[0].value.func.id, 'validate_request')
        clone.body.pop(0)
        hooks = [i for i, node in enumerate(clone.body) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == 'record_first_draws']
        self.assertEqual(len(hooks), 1)
        self.assertEqual(clone.body[hooks[0] - 1].value.func.attr, 'randint')
        clone.body.pop(hooks[0])
        scores = [node for node in clone.body if isinstance(node, ast.Assign)
                  and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                  and node.value.func.id == 'mc2_score_distillation_loss']
        self.assertEqual(len(scores), 1)
        call = scores[0].value
        self.assertEqual([k.arg for k in call.keywords], ['target_mode', 'weight_mode', 'global_step', 'mc_receipt'])
        call.func.id = 'score_distillation_loss'
        call.keywords = call.keywords[:2]
        clone.name = original.name
        clone.args.kwonlyargs = []
        clone.args.kw_defaults = []
        self.assertEqual(ast.dump(clone, include_attributes=False), ast.dump(original, include_attributes=False))
        wrapper = ast.parse(inspect.getsource(helper.train_batch_mc1))
        actual_calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'production' and node.func.attr == 'train_batch']
        self.assertEqual(len(actual_calls), 1)
        entire = ast.parse(Path(helper.__file__).read_text())
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id in ('eval', 'exec') for n in ast.walk(entire)))

    def test_preview_matches_exact_original_three_draws_and_restores_every_rng(self):
        args = make_fixture()
        fixture.reset_rng(204562)
        before = helper.production.capture_rng_state()
        preview = helper.peek_production_draws(args[0], args[10], args[11])
        self.assert_tree_equal(helper.production.capture_rng_state(), before)
        metadata = helper.original_input_identity(args[0], args[10], args[11])
        self.assert_tree_equal(helper.production.capture_rng_state(), before)
        expected = (torch.randn_like(args[0]['mel']), torch.randn_like(args[0]['mel']),
                    torch.randint(1, 1001, (32,), dtype=torch.long))
        for actual, reference in zip(preview, expected):
            self.assertTrue(torch.equal(actual, reference))
        self.assertEqual(metadata['source_noise'], helper.tensor_identity(expected[0]))
        self.assertEqual(metadata['score_noise1'], helper.tensor_identity(expected[1]))
        self.assertEqual(metadata['t1_values'], expected[2].tolist())
        self.assertEqual(len(metadata['batch_tensors']), 3)

    def test_mc1_is_exact_production_losses_models_two_adam_and_all_rng_for_two_steps(self):
        baseline, observed = make_fixture(), make_fixture()
        production_function = helper.production.train_batch
        for step in (3328, 3329):
            fixture.reset_rng(201234 + step)
            expected = production_function(*baseline)
            expected_models = fixture.model_optimizer_state(baseline)
            expected_rng = helper.production.capture_rng_state()
            fixture.reset_rng(201234 + step)
            receipt = {}
            with mock.patch.object(helper.production, 'train_batch', wraps=production_function) as direct:
                actual = helper.train_batch_mc1(*observed, global_step=step, mc_receipt=receipt)
            self.assertEqual(direct.call_count, 1)
            self.assertEqual(actual, expected)
            self.assert_tree_equal(fixture.model_optimizer_state(observed), expected_models)
            self.assert_tree_equal(helper.production.capture_rng_state(), expected_rng)
            self.assertEqual(receipt['dist1'], expected['dist_raw'])
            self.assertEqual(receipt['dist_mixed'], expected['dist_raw'])
            self.assertIsNone(receipt['second'])
            self.assertEqual(len(receipt['first']['per_utterance_l1']), 32)
            self.assertEqual(len(receipt['first']['per_utterance_weighted_l1']), 32)
            self.assertFalse(observed[1]._forward_hooks)
            self.assertFalse(observed[2]._forward_hooks)
        self.assertEqual(len(observed[1].trace), 2)
        self.assertEqual(len(observed[2].trace), 4)
        self.assertEqual(len(observed[3].trace), 6)

    def test_preview_exception_restores_python_numpy_and_torch_states(self):
        args = make_fixture()
        fixture.reset_rng(872)
        before = helper.production.capture_rng_state()

        def fail_after_draw(*positional):
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError('injected preview failure')

        with mock.patch.object(helper.torch, 'randn_like', side_effect=fail_after_draw):
            with self.assertRaisesRegex(RuntimeError, 'injected preview failure'):
                helper.peek_production_draws(args[0], args[10], args[11])
        self.assert_tree_equal(helper.production.capture_rng_state(), before)

    def test_identical_second_draw_is_exact_original_production_update(self):
        baseline, candidate = make_fixture(), make_fixture()
        fixture.reset_rng(204562)
        expected = helper.production.train_batch(*baseline)
        expected_state = fixture.model_optimizer_state(baseline)
        expected_rng = helper.production.capture_rng_state()
        fixture.reset_rng(204562)
        _, noise, timesteps = helper.peek_production_draws(candidate[0], candidate[10], candidate[11])
        with mock.patch.object(helper, 'second_score_draws', return_value=(noise, timesteps)):
            actual = helper.train_batch_mc2(*candidate, global_step=3328, mc_receipt={})
        self.assertEqual(actual, expected)
        self.assert_tree_equal(fixture.model_optimizer_state(candidate), expected_state)
        self.assert_tree_equal(helper.production.capture_rng_state(), expected_rng)

    def test_invalid_second_target_stops_before_S_step_without_hiding_D_step(self):
        args = make_fixture()
        first_function = helper.score_distillation_loss
        calls = []

        def bad_second(*positional, **keywords):
            calls.append(True)
            result = first_function(*positional, **keywords)
            if len(calls) == 2:
                return result[0] * float('nan'), result[1] * float('nan')
            return result

        with mock.patch.object(helper, 'score_distillation_loss', side_effect=bad_second):
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite MC'):
                helper.train_batch_mc2(*args, global_step=3328, mc_receipt={})
        self.assertEqual(len(calls), 2)
        self.assertFalse(args[6].state)
        self.assertTrue(all(int(state['step']) == 1 for state in args[7].state.values()))

    def test_private_second_draw_order_seed_range_repeatability_and_global_rng_preservation(self):
        args = make_fixture()
        fixture.reset_rng(99331)
        before = helper.production.capture_rng_state()
        for step in (3328, 3387, 3447):
            noise, t = helper.second_score_draws(args[0]['mel'], args[5], step)
            generator = torch.Generator(device='cpu').manual_seed(74000000 + step)
            expected_noise = torch.randn(args[0]['mel'].shape, dtype=torch.float32, generator=generator)
            expected_t = torch.randint(1, 1001, (32,), dtype=torch.long, generator=generator)
            self.assertTrue(torch.equal(noise, expected_noise))
            self.assertTrue(torch.equal(t, expected_t))
            self.assertEqual(t.dtype, torch.long)
            self.assertTrue(bool(((t >= 1) & (t <= 1000)).all()))
            self.assert_tree_equal(helper.second_score_draws(args[0]['mel'], args[5], step), (noise, t))
        self.assert_tree_equal(helper.production.capture_rng_state(), before)
        self.assertFalse(torch.equal(helper.second_score_draws(args[0]['mel'], args[5], 3328)[0],
                                    helper.second_score_draws(args[0]['mel'], args[5], 3329)[0]))

    def test_candidate_first_draws_gan_and_first_dist_match_control_and_only_add_one_T(self):
        control, candidate = make_fixture(), make_fixture()
        receipts, losses, states = [], [], []
        for args, call in ((control, helper.train_batch_mc1), (candidate, helper.train_batch_mc2)):
            fixture.reset_rng(204562)
            receipt = {}
            losses.append(call(*args, global_step=3328, mc_receipt=receipt))
            receipts.append(receipt)
            states.append(helper.production.capture_rng_state())
        for field in ('source_noise', 'score_noise1', 't1', 't1_values', 'first', 'dist1'):
            self.assertEqual(receipts[0][field], receipts[1][field])
        for field in ('student_adv', 'fm_raw', 'fm_weighted', 'discriminator'):
            self.assertEqual(losses[0][field], losses[1][field])
        self.assert_tree_equal(control[4].state_dict(), candidate[4].state_dict())
        self.assert_tree_equal(control[7].state_dict(), candidate[7].state_dict())
        self.assert_tree_equal(states[0], states[1])
        self.assertNotEqual(receipts[1]['score_noise1'], receipts[1]['score_noise2'])
        self.assertNotEqual(receipts[1]['t1_values'], receipts[1]['t2_values'])
        self.assertEqual(receipts[1]['private_seed'], 74003328)
        self.assertEqual(receipts[1]['dist_mixed'], losses[1]['dist_raw'])
        expected_mix = (torch.tensor(receipts[1]['dist1']) + torch.tensor(receipts[1]['dist2'])) / 2
        self.assertEqual(float(expected_mix), receipts[1]['dist_mixed'])
        for args, teachers in ((control, 1), (candidate, 2)):
            self.assertEqual(len(args[1].trace), teachers)
            self.assertEqual(len(args[2].trace), 2)
            self.assertEqual(len(args[3].trace), 3)
            self.assertEqual([row['grad_enabled'] for row in args[2].trace], [False, True])
            self.assertTrue(all(not row['grad_enabled'] and not row['x_requires_grad'] for row in args[1].trace))
            for optimizer in args[6:8]:
                self.assertTrue(all(int(state['step']) == 1 for state in optimizer.state.values()))
            self.assertTrue(all(parameter.grad is None for model in args[1:5] for parameter in model.parameters()))
        self.assertTrue(torch.equal(control[1].trace[0]['x'], candidate[1].trace[0]['x']))

    def test_actual_two_target_gradient_is_mean_of_weighted_L1_gradients(self):
        args = make_fixture()
        batch, teacher, _, _, _, diffusion = args[:6]
        fixture.reset_rng(761)
        prediction = torch.randn_like(batch['mel'], requires_grad=True)
        noise = torch.randn_like(prediction)
        timesteps = torch.randint(1, 1001, (32,))
        before = helper.production.capture_rng_state()
        receipt = {}
        loss, targets = helper.mc2_score_distillation_loss(
            prediction, teacher, diffusion, timesteps, noise, batch['spk_emb'], batch['bnf'],
            target_mode='clean_x0', weight_mode='sqrt_alpha_bar', global_step=3328, mc_receipt=receipt)
        self.assert_tree_equal(helper.production.capture_rng_state(), before)
        self.assertEqual(len(teacher.trace), 2)
        self.assertTrue(all(not target.requires_grad and target.grad_fn is None for target in targets))
        loss.backward()
        t2 = torch.tensor(receipt['t2_values'], dtype=torch.long)
        w1 = diffusion.sqrt_alphas_cumprod[timesteps].to(prediction.dtype).view(32, 1, 1)
        w2 = diffusion.sqrt_alphas_cumprod[t2].to(prediction.dtype).view(32, 1, 1)
        expected = (w1 * torch.sign(prediction.detach() - targets[0])
                    + w2 * torch.sign(prediction.detach() - targets[1])) / (2 * prediction.numel())
        torch.testing.assert_close(prediction.grad, expected, atol=1e-12, rtol=1e-6)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertEqual(len(receipt['second']['per_utterance_weighted_l1']), 32)

    def test_full_eight_tiny_branches_match_declared_network_call_budget(self):
        for call, teacher_calls in ((helper.train_batch_mc1, 1), (helper.train_batch_mc2, 2)):
            args = list(make_fixture())
            args[4].branches.extend([fixture.TinyBranch(0.71 + i * 0.01) for i in range(6)])
            args[7] = torch.optim.Adam(args[4].parameters(), lr=2e-4, betas=(0.5, 0.9))
            counts = {'student': 0, 'teacher': 0, 'vocoder': 0, 'D_branches': 0}
            handles = []

            def count(role):
                def hook(module, inputs, output):
                    counts[role] += 1
                return hook

            try:
                for index, name in ((1, 'teacher'), (2, 'student'), (3, 'vocoder')):
                    handles.append(args[index].register_forward_hook(count(name)))
                for branch in args[4].branches:
                    handles.append(branch.register_forward_hook(count('D_branches')))
                receipt = {}
                call(*args, global_step=3328, mc_receipt=receipt)
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(counts, {'student': 2, 'teacher': teacher_calls, 'vocoder': 3, 'D_branches': 32})
            self.assertEqual(receipt['training_forward_budget'], counts)

    def test_opposite_targets_cannot_be_averaged_before_L1(self):
        args = make_fixture()
        prediction = torch.zeros(32, 1, 1, requires_grad=True)
        timesteps = torch.ones(32, dtype=torch.long)
        noise = torch.zeros_like(prediction)
        targets = [torch.full_like(prediction, -2), torch.full_like(prediction, 2)]

        def fake_score(y, teacher, diffusion, t, n, speaker, bnf, **keywords):
            target = targets.pop(0)
            per = (target - y).abs().flatten(1).mean(1)
            return (diffusion.sqrt_alphas_cumprod[t].to(per.dtype) * per).mean(), target

        with mock.patch.object(helper, 'score_distillation_loss', side_effect=fake_score), \
             mock.patch.object(helper, 'second_score_draws', return_value=(noise, timesteps)):
            loss, actual_targets = helper.mc2_score_distillation_loss(
                prediction, args[1], args[5], timesteps, noise, args[0]['spk_emb'], args[0]['bnf'],
                target_mode='clean_x0', weight_mode='sqrt_alpha_bar', global_step=3328, mc_receipt={})
        wrong = (prediction - (actual_targets[0] + actual_targets[1]) / 2).abs().mean()
        self.assertEqual(float(wrong), 0)
        self.assertGreater(float(loss), 1.9)

    def test_bad_public_requests_stop_before_any_model_or_optimizer_call(self):
        cases = [({'global_step': value}, None) for value in (3327, 3448, True, 3328.0)]
        cases += [({}, replace(fixture.PAPER_STUDENT_CONFIG, **change)) for change in
                  ({'score_target': 'legacy_adjacent'}, {'score_weight': 'alpha_bar'},
                   {'lambda_dist': 44.0}, {'lambda_fm': 1.0})]
        for call in (helper.train_batch_mc1, helper.train_batch_mc2):
            for overrides, config in cases:
                args = list(make_fixture())
                if config is not None:
                    args[-1] = config
                before = fixture.model_optimizer_state(args)
                kwargs = {'global_step': 3328, 'mc_receipt': {}}
                kwargs.update(overrides)
                with self.assertRaises(RuntimeError):
                    call(*args, **kwargs)
                self.assert_tree_equal(fixture.model_optimizer_state(args), before)
                self.assertFalse(args[1].trace)
                self.assertFalse(args[2].trace)

    def test_mc1_observer_hooks_clean_up_when_production_raises(self):
        args = make_fixture()
        with mock.patch.object(helper.production, 'train_batch', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                helper.train_batch_mc1(*args, global_step=3328, mc_receipt={})
        self.assertFalse(args[1]._forward_hooks)
        self.assertFalse(args[2]._forward_hooks)
        self.assertFalse(args[6].state)
        self.assertFalse(args[7].state)

    def test_mc1_checks_actual_inputs_and_removes_hooks_on_mismatch(self):
        args = make_fixture()
        original = helper.peek_production_draws

        def corrupt(*positional):
            source, noise, t = original(*positional)
            return source + 1.0, noise, t

        with mock.patch.object(helper, 'peek_production_draws', side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, 'student call differs'):
                helper.train_batch_mc1(*args, global_step=3328, mc_receipt={})
        self.assertFalse(args[6].state)
        self.assertFalse(args[7].state)
        self.assertFalse(args[1]._forward_hooks)
        self.assertFalse(args[2]._forward_hooks)


if __name__ == '__main__':
    unittest.main(verbosity=2)
