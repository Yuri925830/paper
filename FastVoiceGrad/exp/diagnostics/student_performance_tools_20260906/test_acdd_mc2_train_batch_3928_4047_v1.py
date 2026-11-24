"""租用CPU toy验证新范围与冻结数学；不读真实checkpoint，不运行GPU。"""

import ast
import hashlib
import inspect
import marshal
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only rental CPU tests with CUDA_VISIBLE_DEVICES empty are allowed')

import torch
import acdd_mc2_train_batch_v1 as old
import acdd_mc2_train_batch_3448_3927_v1 as prior
import acdd_mc2_train_batch_3928_4047_v1 as helper
import test_acdd_mc2_train_batch_v1 as previous_tests
import test_conversion_score_step as fixture


class MC2DLRBatchTests(unittest.TestCase):
    assert_tree_equal = previous_tests.MC2Tests.assert_tree_equal

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_thin_layer_is_only_the_reviewed_wrapper_with_new_name_and_bounds(self):
        source = Path(prior.__file__).read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '0414669cb6bce6e5eb09bc4fbc0c57ad620c5a88467ca0d272f381c503b77ece')
        expected = source.decode().replace('START, STOP = 3447, 3927', 'START, STOP = 3927, 4047')
        expected = expected.replace('_mc2_frozen_math_for_3448_3927_v1', '_mc2_frozen_math_for_3928_4047_v1')
        expected = expected.replace('step3448..3927', 'step3928..4047')
        self.assertEqual(Path(helper.__file__).read_text(), expected)
        self.assertEqual((prior.START, prior.STOP, prior._core.START, prior._core.STOP),
                         (3447, 3927, 3447, 3927))
        self.assertIsNot(helper._core, prior._core)
        self.assertNotIn('_mc2_frozen_math_for_3928_4047_v1', helper.sys.modules)

    def test_new_draws_leave_both_existing_module_ranges_and_sources_unchanged(self):
        paths = (Path(old.__file__), Path(prior.__file__), Path(helper.__file__))
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        identities = (old.validate_request, prior.validate_request, old.train_batch_mc2, prior.train_batch_mc2)
        args = previous_tests.make_fixture()
        helper.second_score_draws(args[0]['mel'], args[5], 4047)
        helper.validate_request(4047, args[-1], args[-2], {})
        old.validate_request(3328, args[-1], args[-2], {})
        prior.validate_request(3448, args[-1], args[-2], {})
        for module in (old, prior):
            with self.assertRaises(RuntimeError): module.validate_request(4047, args[-1], args[-2], {})
        self.assertEqual(identities, (old.validate_request, prior.validate_request, old.train_batch_mc2, prior.train_batch_mc2))
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})

    def test_frozen_source_and_every_training_math_function_are_identical(self):
        self.assertEqual(hashlib.sha256(helper.FROZEN_HELPER.read_bytes()).hexdigest(), helper.FROZEN_HELPER_SHA)
        for name in ('train_batch_mc1', 'train_batch_mc2', 'second_score_draws', 'mc2_score_distillation_loss',
                     'original_input_identity', 'peek_production_draws', 'record_first_draws', 'tensor_identity'):
            actual, expected = getattr(helper, name), getattr(old, name)
            self.assertEqual(inspect.getsource(actual), inspect.getsource(expected))
            self.assertEqual(marshal.dumps(actual.__code__), marshal.dumps(expected.__code__))
            self.assertIs(actual.__globals__, helper._core.__dict__)
            self.assertIsNot(actual.__globals__, old.__dict__)
        self.assertIs(helper.production, old.production)
        self.assertIs(helper._core.production.train_batch, old.production.train_batch)
        self.assertEqual((helper.START, helper.STOP, helper.MC2_SEED), (3927, 4047, 74000000))
        self.assertEqual((old.START, old.STOP), (3327, 3447))
        self.assertIsNot(helper.validate_request, old.validate_request)
        self.assertEqual(inspect.getsource(helper._original_validate_request), inspect.getsource(old.validate_request))
        # 复用既有生产AST精确对照，不改变旧测试或其公开helper引用。
        previous_tests.MC2Tests().test_ast_clone_only_adds_validation_receipt_and_mc_score_adapter()

    def test_only_two_original_functions_read_range_constants(self):
        tree = ast.parse(helper.FROZEN_HELPER.read_bytes())
        consumers = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                     and any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                             and item.id in ('START', 'STOP') for item in ast.walk(node))}
        self.assertEqual(consumers, {'validate_request', 'second_score_draws'})
        self.assertEqual(helper.train_batch_mc2.__globals__['START'], 3927)
        self.assertEqual(helper.train_batch_mc2.__globals__['STOP'], 4047)

    def test_source_mismatch_stops_before_importing_helper(self):
        with mock.patch.object(helper.Path, 'read_bytes', return_value=b'changed source'), \
             mock.patch.object(helper.importlib.util, 'spec_from_file_location', side_effect=AssertionError('must not import')):
            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                helper._load_frozen_math()

    def test_actual_new_bounds_do_not_change_the_public_old_bounds(self):
        args = previous_tests.make_fixture()
        for step in (3928, 3987, 4047):
            helper.validate_request(step, args[-1], args[-2], {})
            with self.assertRaises(RuntimeError): old.validate_request(step, args[-1], args[-2], {})
        for step in (3328, 3447):
            old.validate_request(step, args[-1], args[-2], {})
            with self.assertRaisesRegex(RuntimeError, '3928..4047'):
                helper.validate_request(step, args[-1], args[-2], {})
        for step in (3927, 4048, True, 3928.0, None, '3928'):
            with self.assertRaises(RuntimeError): helper.second_score_draws(args[0]['mel'], args[5], step)

    def test_all_other_original_public_gates_reject_before_any_forward_or_update(self):
        changed = {'batch_size': 31, 'diffusion_steps': 999, 'student_start_step': 949,
                   'score_target': 'legacy_adjacent', 'score_weight': 'alpha_bar', 'lambda_fm': 1., 'lambda_dist': 44.}
        cases = [('step', step) for step in (3328, 3927, 4048, True, 3928.0, None)]
        cases += [('config', (key, value)) for key, value in changed.items()]
        cases += [('receipt', value) for value in (None, [], {'not_empty': 1})]
        cases += [('device', torch.device('meta')), ('cpu_hidden', None)]
        for call in (helper.train_batch_mc1, helper.train_batch_mc2):
            for kind, value in cases:
                args = list(previous_tests.make_fixture()); receipt = {}; step = 3928
                if kind == 'step': step = value
                elif kind == 'receipt': receipt = value
                elif kind == 'device': args[10] = value
                elif kind == 'config':
                    values = {name: getattr(args[-1], name) for name in changed}
                    values[value[0]] = value[1]; args[-1] = SimpleNamespace(**values)
                before = fixture.model_optimizer_state(args)
                # 只替换私有模块的os引用，真实进程CUDA_VISIBLE_DEVICES始终为空。
                gate_os = SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': 'not-hidden'}) if kind == 'cpu_hidden' else helper._core.os
                with mock.patch.object(helper._core, 'os', gate_os):
                    with self.assertRaises(RuntimeError): call(*args, global_step=step, mc_receipt=receipt)
                self.assert_tree_equal(fixture.model_optimizer_state(args), before)
                self.assertTrue(all(not args[index].trace for index in (1, 2, 3)))
                self.assertFalse(args[6].state); self.assertFalse(args[7].state)

    def test_private_second_draw_order_and_seed_match_absolute_new_steps_without_global_rng_changes(self):
        args = previous_tests.make_fixture(); fixture.reset_rng(205162)
        before = helper.production.capture_rng_state()
        for step in (3928, 3987, 3988, 4047):
            noise, timesteps = helper.second_score_draws(args[0]['mel'], args[5], step)
            generator = torch.Generator(device='cpu').manual_seed(74000000 + step)
            expected_noise = torch.randn(args[0]['mel'].shape, dtype=torch.float32, generator=generator)
            expected_t = torch.randint(1, 1001, (32,), dtype=torch.long, generator=generator)
            self.assertTrue(torch.equal(noise, expected_noise)); self.assertTrue(torch.equal(timesteps, expected_t))
            self.assert_tree_equal(helper.production.capture_rng_state(), before)
        with self.assertRaisesRegex(RuntimeError, 'distribution'):
            helper.second_score_draws(args[0]['mel'], SimpleNamespace(n_levels=999), 3928)

    def test_mc1_still_calls_exact_production_for_two_new_complete_steps(self):
        baseline, observed = previous_tests.make_fixture(), previous_tests.make_fixture()
        production_function = old.production.train_batch
        for step in (3928, 3929):
            fixture.reset_rng(201234 + step)
            expected = production_function(*baseline)
            expected_state = fixture.model_optimizer_state(baseline)
            expected_rng = helper.production.capture_rng_state()
            fixture.reset_rng(201234 + step); receipt = {}
            with mock.patch.object(helper.production, 'train_batch', wraps=production_function) as direct:
                actual = helper.train_batch_mc1(*observed, global_step=step, mc_receipt=receipt)
            self.assertEqual(direct.call_count, 1); self.assertEqual(actual, expected)
            self.assert_tree_equal(fixture.model_optimizer_state(observed), expected_state)
            self.assert_tree_equal(helper.production.capture_rng_state(), expected_rng)
            self.assertEqual(receipt['global_step'], step); self.assertIsNone(receipt['second'])
            self.assertFalse(observed[1]._forward_hooks); self.assertFalse(observed[2]._forward_hooks)

    def test_mc2_complete_update_equals_old_math_for_the_same_realized_second_draw(self):
        for step in (3928, 4047):
            baseline, candidate = previous_tests.make_fixture(), previous_tests.make_fixture()
            second = helper.second_score_draws(candidate[0]['mel'], candidate[5], step)
            fixture.reset_rng(201234 + step); reference_receipt = {}
            # 旧函数仍用合法旧step；仅注入同一已实现第二draw，从而逐位比较数学而非不同seed。
            with mock.patch.object(old, 'second_score_draws', return_value=second):
                expected = old.train_batch_mc2(*baseline, global_step=3328, mc_receipt=reference_receipt)
            expected_state = fixture.model_optimizer_state(baseline); expected_rng = helper.production.capture_rng_state()
            fixture.reset_rng(201234 + step); actual_receipt = {}
            actual = helper.train_batch_mc2(*candidate, global_step=step, mc_receipt=actual_receipt)
            self.assertEqual(actual, expected)
            self.assert_tree_equal(fixture.model_optimizer_state(candidate), expected_state)
            self.assert_tree_equal(helper.production.capture_rng_state(), expected_rng)
            reference_receipt.update(global_step=step, private_seed=74000000 + step)
            self.assertEqual(actual_receipt, reference_receipt)

    def test_first_original_draws_gan_and_dist1_match_mc1_and_forward_budget_is_unchanged(self):
        results = []
        for call, teachers in ((helper.train_batch_mc1, 1), (helper.train_batch_mc2, 2)):
            args = list(previous_tests.make_fixture())
            args[4].branches.extend([fixture.TinyBranch(.71 + i * .01) for i in range(6)])
            args[7] = torch.optim.Adam(args[4].parameters(), lr=2e-4, betas=(.5, .9))
            counts = {'student': 0, 'teacher': 0, 'vocoder': 0, 'D_branches': 0}; handles = []
            def count(name):
                def observe(*unused): counts[name] += 1
                return observe
            try:
                for index, name in ((1, 'teacher'), (2, 'student'), (3, 'vocoder')):
                    handles.append(args[index].register_forward_hook(count(name)))
                for branch in args[4].branches: handles.append(branch.register_forward_hook(count('D_branches')))
                fixture.reset_rng(205162); before = helper.production.capture_rng_state()
                identity = helper.original_input_identity(args[0], args[10], args[11])
                self.assert_tree_equal(helper.production.capture_rng_state(), before)
                receipt = {}; result = call(*args, global_step=3928, mc_receipt=receipt)
            finally:
                for handle in handles: handle.remove()
            self.assertEqual(counts, {'student': 2, 'teacher': teachers, 'vocoder': 3, 'D_branches': 32})
            self.assertEqual(counts, receipt['training_forward_budget'])
            for key in ('source_noise', 'score_noise1', 't1', 't1_values'): self.assertEqual(receipt[key], identity[key])
            self.assertTrue(all(parameter.grad is None for model in args[1:5] for parameter in model.parameters()))
            results.append((result, receipt, helper.production.capture_rng_state()))
        for field in ('discriminator', 'student_adv', 'fm_raw', 'fm_weighted'):
            self.assertEqual(results[0][0][field], results[1][0][field])
        for field in ('source_noise', 'score_noise1', 't1', 't1_values', 'first', 'dist1'):
            self.assertEqual(results[0][1][field], results[1][1][field])
        self.assert_tree_equal(results[0][2], results[1][2])
        self.assertEqual(results[1][1]['private_seed'], 74003928)

    def test_two_l1_gradients_remain_the_average_not_target_averaging(self):
        args = previous_tests.make_fixture(); diffusion = args[5]
        prediction = torch.zeros(32, 1, 1, requires_grad=True)
        t = torch.ones(32, dtype=torch.long); noise = torch.zeros_like(prediction)
        targets = [torch.full_like(prediction, -2), torch.full_like(prediction, 2)]
        def score(y, teacher, d, timesteps, *unused, **kwargs):
            target = targets.pop(0)
            per = (target - y).abs().flatten(1).mean(1)
            return (d.sqrt_alphas_cumprod[timesteps].to(per.dtype) * per).mean(), target
        with mock.patch.object(helper._core, 'score_distillation_loss', side_effect=score), \
             mock.patch.object(helper._core, 'second_score_draws', return_value=(noise, t)):
            loss, actual_targets = helper.mc2_score_distillation_loss(prediction, args[1], diffusion, t, noise,
                args[0]['spk_emb'], args[0]['bnf'], target_mode='clean_x0', weight_mode='sqrt_alpha_bar',
                global_step=3928, mc_receipt={})
        self.assertGreater(float(loss), 1.9)
        self.assertEqual(float((prediction - sum(actual_targets) / 2).abs().mean()), 0.)
        loss.backward(); self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_nonfinite_second_teacher_target_keeps_D_step_visible_and_prevents_S_step(self):
        args = previous_tests.make_fixture(); original = helper._core.score_distillation_loss; calls = []
        def bad_second(*positional, **keywords):
            calls.append(True); loss, target = original(*positional, **keywords)
            return (loss * float('nan'), target * float('nan')) if len(calls) == 2 else (loss, target)
        with mock.patch.object(helper._core, 'score_distillation_loss', side_effect=bad_second):
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite MC'):
                helper.train_batch_mc2(*args, global_step=3928, mc_receipt={})
        self.assertEqual(len(calls), 2); self.assertFalse(args[6].state)
        self.assertTrue(all(int(state['step']) == 1 for state in args[7].state.values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
