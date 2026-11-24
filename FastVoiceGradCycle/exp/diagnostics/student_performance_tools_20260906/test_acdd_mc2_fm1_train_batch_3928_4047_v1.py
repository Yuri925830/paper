"""仅租用CPU的FM1 helper小测；只用小模型，不读真实checkpoint或改旧模块。"""

import copy
import hashlib
import inspect
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import CodeType, SimpleNamespace
from unittest.mock import patch

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Run tests only on rental CPU with CUDA explicitly hidden')
import torch
import acdd_mc2_train_batch_3928_4047_v1 as control
import acdd_mc2_fm1_train_batch_3928_4047_v1 as candidate
import test_acdd_mc2_train_batch_v1 as oldtests
import test_conversion_score_step as fixture


def toy(fm=1.):
    args = list(oldtests.make_fixture())
    args[-1] = replace(args[-1], lambda_fm=fm)
    return args


class FM1HelperTests(unittest.TestCase):
    assert_tree_equal = oldtests.MC2Tests.assert_tree_equal

    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def assert_code_semantics_equal(self, actual, expected):
        # marshal可因对象引用/字符串intern状态产生不同编码；核真正执行字段和常量。
        self.assertIsInstance(actual, CodeType); self.assertIsInstance(expected, CodeType)
        fields = ('co_code', 'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount',
                  'co_nlocals', 'co_stacksize', 'co_flags', 'co_names', 'co_varnames',
                  'co_freevars', 'co_cellvars', 'co_name')
        for name in fields:
            self.assertEqual(getattr(actual, name), getattr(expected, name), name)
        self.assertEqual(len(actual.co_consts), len(expected.co_consts))
        for left, right in zip(actual.co_consts, expected.co_consts):
            self.assertIs(type(left), type(right))
            if isinstance(left, CodeType):
                self.assert_code_semantics_equal(left, right)
            else:
                self.assertEqual(left, right)

    def test_private_namespace_original_sources_and_training_math_unchanged(self):
        self.assertEqual(hashlib.sha256(candidate.RANGE_FILE.read_bytes()).hexdigest(), candidate.RANGE_SHA)
        self.assertIsNot(candidate._core, control._core)
        self.assertIsNot(candidate._core.validate_request, control._core.validate_request)
        for name in ('train_batch_mc2', 'second_score_draws', 'mc2_score_distillation_loss',
                     'original_input_identity', 'peek_production_draws', 'tensor_identity'):
            actual, original = getattr(candidate, name), getattr(control, name)
            self.assertEqual(inspect.getsource(actual), inspect.getsource(original))
            self.assert_code_semantics_equal(actual.__code__, original.__code__)
            self.assertIs(actual.__globals__, candidate._core.__dict__)
            self.assertIsNot(actual.__globals__, control._core.__dict__)
        self.assertIs(candidate.production, control.production)

    def test_code_identity_still_rejects_changed_constants_operations_and_nested_code(self):
        reference = lambda x: x + 1
        for changed in (lambda x: x + 2, lambda x: x - 1):
            with self.assertRaises(AssertionError):
                self.assert_code_semantics_equal(reference.__code__, changed.__code__)
        nested_reference = lambda: (lambda x: x + 1)
        nested_changed = lambda: (lambda x: x + 2)
        with self.assertRaises(AssertionError):
            self.assert_code_semantics_equal(nested_reference.__code__, nested_changed.__code__)

    def test_candidate_only_FM1_and_control_remains_FM2(self):
        one, two = toy(1.), toy(2.)
        for step in (3928, 3987, 4047):
            candidate.validate_request(step, one[-1], one[-2], {})
            control.validate_request(step, two[-1], two[-2], {})
            with self.assertRaises(RuntimeError): candidate.validate_request(step, two[-1], two[-2], {})
            with self.assertRaises(RuntimeError): control.validate_request(step, one[-1], one[-2], {})
        self.assertEqual((candidate.START, candidate.STOP, candidate.MC2_SEED, candidate.FM_WEIGHT), (3927, 4047, 74000000, 1.))

    def test_all_other_gates_reject_before_forward_or_optimizer(self):
        for key, value in [('batch_size', 31), ('diffusion_steps', 999), ('student_start_step', 949),
                           ('score_target', 'legacy_adjacent'), ('score_weight', 'alpha_bar'),
                           ('lambda_fm', 0.), ('lambda_fm', 2.), ('lambda_dist', 44.)]:
            args = toy(); args[-1] = replace(args[-1], **{key: value})
            before = fixture.model_optimizer_state(args)
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                candidate.train_batch_mc2(*args, global_step=3928, mc_receipt={})
            self.assert_tree_equal(fixture.model_optimizer_state(args), before)
            self.assertTrue(all(not args[index].trace for index in (1, 2, 3)))
        args = toy()
        for step in (3927, 4048, True, 3928.0, None):
            with self.assertRaises(RuntimeError): candidate.validate_request(step, args[-1], args[-2], {})
        for receipt in (None, [], {'used': True}):
            with self.assertRaises(RuntimeError): candidate.validate_request(3928, args[-1], args[-2], receipt)
        with self.assertRaises(RuntimeError): candidate.validate_request(3928, args[-1], torch.device('meta'), {})
        with patch.object(candidate, 'os', SimpleNamespace(environ={})), self.assertRaises(RuntimeError):
            candidate.validate_request(3928, args[-1], torch.device('cpu'), {})

    def test_second_draw_same_absolute_seed_order_and_no_global_rng_consumption(self):
        args = toy(); fixture.reset_rng(205162)
        before = candidate.production.capture_rng_state()
        for step in (3928, 4047):
            actual = candidate.second_score_draws(args[0]['mel'], args[5], step)
            reference = control.second_score_draws(args[0]['mel'], args[5], step)
            for a, b in zip(actual, reference): self.assertTrue(torch.equal(a, b))
            generator = torch.Generator(device='cpu').manual_seed(74000000 + step)
            self.assertTrue(torch.equal(actual[0], torch.randn(args[0]['mel'].shape, generator=generator)))
            self.assertTrue(torch.equal(actual[1], torch.randint(1, 1001, (32,), generator=generator)))
            self.assert_tree_equal(candidate.production.capture_rng_state(), before)

    def test_same_first_raw_losses_D_state_noise_then_true_FM1_optimizer_gradient(self):
        runs = []
        for module, fm in ((control, 2.), (candidate, 1.)):
            args = toy(fm); fixture.reset_rng(205162); receipt = {}
            losses = module.train_batch_mc2(*args, global_step=3928, mc_receipt=receipt)
            runs.append((args, losses, receipt, candidate.production.capture_rng_state()))
        for key in ('discriminator', 'student_adv', 'fm_raw', 'dist_raw', 'dist_weighted'):
            self.assertEqual(runs[0][1][key], runs[1][1][key])
        self.assertEqual(runs[0][2], runs[1][2])
        self.assert_tree_equal(runs[0][0][4].state_dict(), runs[1][0][4].state_dict())
        self.assert_tree_equal(runs[0][0][7].state_dict(), runs[1][0][7].state_dict())
        self.assert_tree_equal(runs[0][3], runs[1][3])
        old, new = runs[0][1], runs[1][1]
        self.assertAlmostEqual(new['fm_weighted'], new['fm_raw'], places=6)
        self.assertAlmostEqual(old['fm_weighted'], 2 * new['fm_raw'], places=6)
        self.assertAlmostEqual(old['student_total'] - new['student_total'], new['fm_raw'], places=5)
        # 首步Adam方向归一化可能使权重接近，但矩必须反映实际不同的FM梯度。
        first, second = runs[0][0][6].state_dict(), runs[1][0][6].state_dict()
        self.assertTrue(any(not torch.equal(first['state'][i]['exp_avg'], second['state'][i]['exp_avg']) for i in first['state']))
        for args, _, _, _ in runs:
            self.assertTrue(all(p.grad is None for model in args[1:5] for p in model.parameters()))

    def test_two_complete_toy_steps_use_same_true_FM1_and_original_forward_budget(self):
        args = toy()
        args[4].branches.extend([fixture.TinyBranch(.71 + i * .01) for i in range(6)])
        args[7] = torch.optim.Adam(args[4].parameters(), lr=2e-4, betas=(.5, .9))
        counts = {'student': 0, 'teacher': 0, 'vocoder': 0, 'D_branches': 0}; handles = []
        def count(name):
            def hook(*unused): counts[name] += 1
            return hook
        try:
            for index, name in ((1, 'teacher'), (2, 'student'), (3, 'vocoder')):
                handles.append(args[index].register_forward_hook(count(name)))
            for branch in args[4].branches: handles.append(branch.register_forward_hook(count('D_branches')))
            for step in (3928, 3929):
                fixture.reset_rng(201234 + step); before = candidate.production.capture_rng_state()
                identity = candidate.original_input_identity(args[0], args[-2], args[-1])
                self.assert_tree_equal(candidate.production.capture_rng_state(), before)
                receipt = {}; losses = candidate.train_batch_mc2(*args, global_step=step, mc_receipt=receipt)
                for key in ('source_noise', 'score_noise1', 't1', 't1_values'): self.assertEqual(identity[key], receipt[key])
                self.assertEqual(receipt['private_seed'], 74000000 + step)
                self.assertAlmostEqual(losses['fm_weighted'], losses['fm_raw'])
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(counts, {'student': 4, 'teacher': 4, 'vocoder': 6, 'D_branches': 64})
        for opt in (args[6], args[7]):
            self.assertTrue(all(float(state['step']) == 2 for state in opt.state.values()))

    def test_second_target_failure_keeps_actual_D_update_but_no_S_update(self):
        args = toy(); original = candidate._core.score_distillation_loss; count = []
        def fail_second(*positional, **keywords):
            count.append(True)
            if len(count) == 2: raise RuntimeError('injected second teacher failure')
            return original(*positional, **keywords)
        with patch.object(candidate._core, 'score_distillation_loss', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'second teacher failure'):
                candidate.train_batch_mc2(*args, global_step=3928, mc_receipt={})
        self.assertFalse(args[6].state)
        self.assertTrue(all(float(state['step']) == 1 for state in args[7].state.values()))


if __name__ == '__main__': unittest.main()
