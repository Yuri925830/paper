"""只在租用CPU运行的新范围小测；不用真实模型或GPU。"""

import ast
import hashlib
import inspect
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import CodeType
from unittest.mock import patch

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Run only on rental CPU with CUDA explicitly hidden')
import torch
import acdd_mc2_train_batch_v1 as original
import acdd_mc2_train_batch_3928_4047_v1 as prior
import acdd_mc2_train_batch_3928_4287_v1 as helper
import test_acdd_mc2_train_batch_v1 as oldtests
import test_conversion_score_step as fixture


class Range360Tests(unittest.TestCase):
    assert_tree_equal = oldtests.MC2Tests.assert_tree_equal

    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def assert_code_equal(self, actual, expected):
        # 不用marshal字节，避免intern/引用编码造成伪差异；实际指令与常量逐项比较。
        self.assertIs(type(actual), CodeType); self.assertIs(type(expected), CodeType)
        for name in ('co_code', 'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
                     'co_stacksize', 'co_flags', 'co_names', 'co_varnames', 'co_freevars', 'co_cellvars', 'co_name'):
            self.assertEqual(getattr(actual, name), getattr(expected, name), name)
        self.assertEqual(len(actual.co_consts), len(expected.co_consts))
        for a, b in zip(actual.co_consts, expected.co_consts):
            self.assertIs(type(a), type(b))
            if isinstance(a, CodeType): self.assert_code_equal(a, b)
            else: self.assertEqual(a, b)

    def test_source_math_private_namespace_and_old_ranges_unchanged(self):
        self.assertEqual(hashlib.sha256(helper.FROZEN_HELPER.read_bytes()).hexdigest(), helper.FROZEN_HELPER_SHA)
        for name in ('train_batch_mc1', 'train_batch_mc2', 'second_score_draws', 'mc2_score_distillation_loss',
                     'original_input_identity', 'peek_production_draws', 'record_first_draws', 'score_record', 'tensor_identity'):
            actual, expected = getattr(helper, name), getattr(original, name)
            self.assertEqual(inspect.getsource(actual), inspect.getsource(expected))
            self.assert_code_equal(actual.__code__, expected.__code__)
            self.assert_code_equal(inspect.unwrap(actual).__code__, inspect.unwrap(expected).__code__)
            self.assertIs(inspect.unwrap(actual).__globals__, helper._core.__dict__)
        self.assertIsNot(helper._core, prior._core)
        self.assertEqual((original.START, original.STOP), (3327, 3447))
        self.assertEqual((prior.START, prior.STOP, prior._core.START, prior._core.STOP), (3927, 4047, 3927, 4047))
        self.assertEqual((helper.START, helper.STOP, helper.MC2_SEED), (3927, 4287, 74000000))
        self.assertIs(helper.production, original.production)
        tree = ast.parse(helper.FROZEN_HELPER.read_bytes())
        users = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and
                 any(isinstance(i, ast.Name) and isinstance(i.ctx, ast.Load) and i.id in ('START', 'STOP') for i in ast.walk(n))}
        self.assertEqual(users, {'validate_request', 'second_score_draws'})

    def test_code_comparison_rejects_real_changes(self):
        for reference, changed in ((lambda x: x + 1, lambda x: x + 2),
                                   (lambda: (lambda x: x + 1), lambda: (lambda x: x - 1))):
            with self.assertRaises(AssertionError): self.assert_code_equal(reference.__code__, changed.__code__)

    def test_range_and_original_FM2_gates_before_any_updates(self):
        args = oldtests.make_fixture()
        for step in (3928, 4047, 4048, 4167, 4287): helper.validate_request(step, args[-1], args[-2], {})
        for step in (3927, 4288, True, 4048., None, '4287'):
            with self.assertRaises(RuntimeError): helper.train_batch_mc2(*args, global_step=step, mc_receipt={})
        for key, value in (('lambda_fm', 1.), ('lambda_dist', 44.), ('batch_size', 31), ('diffusion_steps', 999),
                           ('student_start_step', 949), ('score_target', 'legacy_adjacent'), ('score_weight', 'alpha_bar')):
            changed = list(oldtests.make_fixture()); changed[-1] = replace(changed[-1], **{key: value})
            before = fixture.model_optimizer_state(changed)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                helper.train_batch_mc2(*changed, global_step=4048, mc_receipt={})
            self.assert_tree_equal(before, fixture.model_optimizer_state(changed))
            self.assertTrue(all(not changed[i].trace for i in (1, 2, 3)))
        for receipt in (None, [], {'used': 1}):
            with self.assertRaises(RuntimeError): helper.validate_request(4048, args[-1], args[-2], receipt)
        with self.assertRaises(RuntimeError): prior.validate_request(4048, args[-1], args[-2], {})

    def test_absolute_second_seed_order_no_global_RNG_consumption(self):
        args = oldtests.make_fixture(); fixture.reset_rng(205162); before = helper.production.capture_rng_state()
        for step in (3928, 4047, 4048, 4287):
            noise, t = helper.second_score_draws(args[0]['mel'], args[5], step)
            generator = torch.Generator(device='cpu').manual_seed(74000000 + step)
            self.assertTrue(torch.equal(noise, torch.randn(args[0]['mel'].shape, generator=generator)))
            self.assertTrue(torch.equal(t, torch.randint(1, 1001, (32,), generator=generator)))
            self.assert_tree_equal(helper.production.capture_rng_state(), before)

    def test_first_replay_updates_equal_actual_prior_math_and_all_Adam_RNG(self):
        left, right = oldtests.make_fixture(), oldtests.make_fixture()
        for step in (3928, 3929):
            fixture.reset_rng(201234 + step); old_receipt = {}; expected = prior.train_batch_mc2(*left, global_step=step, mc_receipt=old_receipt)
            expected_state = fixture.model_optimizer_state(left); expected_rng = helper.production.capture_rng_state()
            fixture.reset_rng(201234 + step); receipt = {}; actual = helper.train_batch_mc2(*right, global_step=step, mc_receipt=receipt)
            self.assertEqual(actual, expected); self.assertEqual(receipt, old_receipt)
            self.assert_tree_equal(fixture.model_optimizer_state(right), expected_state)
            self.assert_tree_equal(helper.production.capture_rng_state(), expected_rng)

    def test_two_novel_boundary_updates_have_original_math_and_counts(self):
        args = list(oldtests.make_fixture())
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
            for step in (4048, 4287):
                fixture.reset_rng(201234 + step); receipt = {}
                loss = helper.train_batch_mc2(*args, global_step=step, mc_receipt=receipt)
                self.assertEqual(receipt['private_seed'], 74000000 + step)
                self.assertAlmostEqual(loss['fm_weighted'], 2 * loss['fm_raw'])
                self.assertAlmostEqual(loss['dist_weighted'], 45 * loss['dist_raw'], places=5)
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(counts, {'student': 4, 'teacher': 4, 'vocoder': 6, 'D_branches': 64})
        for opt in (args[6], args[7]): self.assertTrue(all(float(v['step']) == 2 for v in opt.state.values()))

    def test_second_teacher_failure_does_not_hide_D_only_update(self):
        args = oldtests.make_fixture(); original_score = helper._core.score_distillation_loss; calls = []
        def fail_second(*positional, **keywords):
            calls.append(True)
            if len(calls) == 2: raise RuntimeError('injected second teacher failure')
            return original_score(*positional, **keywords)
        with patch.object(helper._core, 'score_distillation_loss', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'second teacher failure'):
                helper.train_batch_mc2(*args, global_step=4048, mc_receipt={})
        self.assertFalse(args[6].state)
        self.assertTrue(all(float(v['step']) == 1 for v in args[7].state.values()))


if __name__ == '__main__': unittest.main()
