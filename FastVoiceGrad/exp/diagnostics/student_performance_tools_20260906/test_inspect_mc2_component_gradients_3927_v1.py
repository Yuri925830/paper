"""新只读梯度入口的小型CPU测试；不读真实权重，不调用主入口或GPU。"""

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# 连导入也只允许在租用服务器进行，且调用方必须明确隐藏CUDA。
ROOT = Path('/root/FsatVoiceGrad')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('These project tests may run only on the Matpool rental')
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES empty is required')

import torch

TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
FILE = TOOLS / 'inspect_mc2_component_gradients_3927_v1.py'
SPEC = importlib.util.spec_from_file_location('component_gradient_entry_cpu_test', FILE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)  # __name__不是__main__，因此不会进入真实实验。
SOURCE = FILE.read_text(encoding='utf8')
TREE = ast.parse(SOURCE)


def isolated_function(function, **globals_to_replace):
    # 只给待测新函数一份独立globals；不修改Torch、生产模块或真正sys.modules。
    namespace = dict(function.__globals__)
    namespace.update(globals_to_replace)
    return types.FunctionType(function.__code__, namespace, function.__name__,
                              function.__defaults__, function.__closure__)


class ToyState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(2, 2)
        self.register_buffer('saved_window', torch.ones(2))
        self.register_buffer('unsaved_window', torch.ones(2), persistent=False)
        self.train()
        self.block.eval()  # 特意混合子层模式，避免只检查顶层training。
        self.block.bias.requires_grad_(False)


class ComponentEntryCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_available():
            raise RuntimeError('These tests must have no visible CUDA device')
        torch.set_num_threads(1)

    def cli(self):
        argv = ['--probe-approved']
        for name in ('driver', 'tests', 'helper', 'helper-tests'):
            argv += ['--' + name + '-sha256', 'a' * 64]
        return argv

    def parent(self):
        # 使用真实字段名，内容是极小的独立JSON；不读取3927 full。
        contract = {'configuration': {'score_target': 'clean_x0',
                    'score_weight': 'sqrt_alpha_bar', 'lambda_fm': 2,
                    'lambda_dist': 45}, 'fixed_toy_contract': True}
        canonical = probe.training.canonical_json_sha256(contract)
        monitor = {'global_step': 3927, 'development': {'mel_l1': .2}}
        state = {'completed_global_step': 3927, 'last_full_saved_global_step': 3927,
                 'partial_batch_may_have_updated_D': False, 'restore_completed': True,
                 'full_snapshot': {'path': str(probe.PARENT), 'sha256': probe.PARENT_SHA,
                                   'resumable': True}, 'last_monitor': monitor}
        completion = {'status': 'completed480', 'cleanup_verified': True,
                      'contract_sha256': canonical, 'state': state}
        payload = {'global_step': 3927, 'epoch': 0, 'mc_samples': 2, 'resumable': True,
                   'run_contract_sha256': canonical,
                   'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar',
                   'execution_error': None, 'configuration': copy.deepcopy(contract['configuration']),
                   'development': copy.deepcopy(monitor['development'])}
        check = isolated_function(probe.validate_parent, CANONICAL=canonical)
        return contract, completion, payload, check

    def inputs(self):
        shapes = dict(probe.envelope.INPUT_SHAPES, score_noise=(32, 80, 128),
                      score_timesteps=(32,), q950=(32, 80, 128))
        values = {name: torch.zeros(shapes[name], dtype=torch.long if name == 'score_timesteps'
                                   else torch.float32) for name in probe.INPUT_KEYS}
        values['score_timesteps'] = torch.arange(1, 33, dtype=torch.long)
        # 原档还有4个DEV键；本次只核它们身份，不拿它们作前向或评分。
        for name in ('dev_mel', 'dev_bnf', 'dev_spk_emb', 'dev_source_noise'):
            values[name] = torch.ones(1)
        return values, self.input_contract(values)

    def input_contract(self, values):
        return {'input_identities': {'old': {name: probe.envelope.old.tensor_sha(value)
                                          for name, value in values.items()}}}

    def test_cli_requires_explicit_approval_and_all_four_valid_hashes(self):
        args = probe.arguments(self.cli())
        self.assertTrue(args.probe_approved)
        self.assertEqual(args.helper_tests_sha256, 'a' * 64)
        with self.assertRaisesRegex(RuntimeError, 'approval'):
            probe.arguments(self.cli()[1:])
        for position in (2, 4, 6, 8):
            for bad in ('a' * 63, 'a' * 65, 'A' * 64, 'g' * 64, ''):
                argv = self.cli(); argv[position] = bad
                with self.subTest(position=position, bad=bad), self.assertRaisesRegex(RuntimeError, 'SHA256'):
                    probe.arguments(argv)

    def test_cli_missing_hash_fails_without_entering_main(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.arguments(['--probe-approved'])

    def test_complete_parent_metadata_and_matching_payload_accepted(self):
        contract, completion, payload, check = self.parent()
        before = copy.deepcopy((contract, completion, payload))
        self.assertIs(check(contract, completion), completion['state'])
        self.assertIs(check(contract, completion, payload), completion['state'])
        self.assertEqual((contract, completion, payload), before)

    def test_parent_failed_cleanup_partial_and_wrong_saved_cursor_rejected(self):
        contract, completion, payload, check = self.parent()
        changes = [('status', 'stopped'), ('cleanup_verified', False),
                   ('contract_sha256', 'b' * 64)]
        for field, value in changes:
            changed = copy.deepcopy(completion); changed[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                check(contract, changed, payload)
        for field, value in [('completed_global_step', 3926),
                             ('last_full_saved_global_step', 3447),
                             ('partial_batch_may_have_updated_D', True)]:
            changed = copy.deepcopy(completion); changed['state'][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                check(contract, changed, payload)

    def test_parent_snapshot_and_contract_identity_rejected_when_changed(self):
        contract, completion, payload, check = self.parent()
        for field, value in [('path', '/wrong/final_full.pt'), ('sha256', 'c' * 64),
                             ('resumable', False)]:
            changed = copy.deepcopy(completion); changed['state']['full_snapshot'][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                check(contract, changed, payload)
        changed = copy.deepcopy(contract); changed['extra'] = True
        with self.assertRaisesRegex(RuntimeError, 'canonical'):
            check(changed, completion, payload)

    def test_actual_payload_configuration_cursor_semantics_and_monitor_rejected(self):
        contract, completion, payload, check = self.parent()
        changes = [('global_step', 4047), ('epoch', 1), ('mc_samples', 1),
                   ('resumable', False), ('run_contract_sha256', 'd' * 64),
                   ('score_semantics', 'legacy_adjacent'), ('execution_error', 'partial'),
                   ('configuration', dict(payload['configuration'], lambda_fm=1)),
                   ('development', {'mel_l1': .201})]
        for field, value in changes:
            changed = copy.deepcopy(payload); changed[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                check(contract, completion, changed)

    def test_input_all_archive_identities_and_seven_used_keys_accepted(self):
        values, contract = self.inputs()
        result = probe.input_check(values, contract)
        self.assertEqual(result, contract['input_identities']['old'])
        self.assertEqual(len(result), 11)
        self.assertEqual(set(probe.INPUT_KEYS),
                         {'x0', 'bnf', 'spk_emb', 'source_noise', 'score_noise', 'score_timesteps', 'q950'})

    def test_input_identity_rejects_mutation_missing_or_extra_archive_key(self):
        values, contract = self.inputs()
        changed = dict(values); changed['dev_mel'] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, 'archive changed'):
            probe.input_check(changed, contract)
        changed = dict(values); del changed['score_noise']
        with self.assertRaisesRegex(RuntimeError, 'archive changed'):
            probe.input_check(changed, contract)
        changed = dict(values, unexpected=torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError, 'archive changed'):
            probe.input_check(changed, contract)

    def test_input_shape_dtype_and_nonfinite_fail_even_with_matching_identity(self):
        values, _ = self.inputs()
        cases = [('bnf', torch.zeros(32, 143, 128)),
                 ('x0', torch.zeros(1, 80, 128)),
                 ('spk_emb', values['spk_emb'].double()),
                 ('score_timesteps', values['score_timesteps'].float()),
                 ('q950', torch.full((32, 80, 128), float('nan'))),
                 ('score_noise', torch.full((32, 80, 128), float('inf')))]
        for key, value in cases:
            changed = dict(values); changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'shape, dtype or finite'):
                probe.input_check(changed, self.input_contract(changed))

    def test_input_time_boundaries_preserved_and_outside_range_rejected(self):
        values, _ = self.inputs()
        values['score_timesteps'][0] = 1; values['score_timesteps'][-1] = 1000
        probe.input_check(values, self.input_contract(values))
        for bad in (0, 1001, -1):
            changed = dict(values); changed['score_timesteps'] = values['score_timesteps'].clone()
            changed['score_timesteps'][0] = bad
            with self.subTest(time=bad), self.assertRaisesRegex(RuntimeError, 'outside1..1000'):
                probe.input_check(changed, self.input_contract(changed))

    def test_snapshot_records_nested_modes_grad_ownership_and_all_buffers(self):
        model = ToyState(); before = probe.snapshot(model)
        self.assertEqual(before['modes'], {'': True, 'block': False})
        self.assertEqual(before['requires_grad'], {'block.weight': True, 'block.bias': False})
        probe.unchanged(model, before)
        with torch.no_grad():
            model.unsaved_window.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            probe.unchanged(model, before)

    def test_unchanged_rejects_weights_persistent_buffer_modes_requires_grad_and_grad(self):
        for kind in ('weight', 'saved_buffer', 'root_mode', 'child_mode', 'requires_grad', 'grad'):
            model = ToyState(); before = probe.snapshot(model)
            with torch.no_grad():
                if kind == 'weight': model.block.weight.add_(.1)
                elif kind == 'saved_buffer': model.saved_window.add_(1)
                elif kind == 'root_mode': model.training = False
                elif kind == 'child_mode': model.block.training = True
                elif kind == 'requires_grad': model.block.weight.requires_grad_(False)
                elif kind == 'grad': model.block.weight.grad = torch.zeros_like(model.block.weight)
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, 'changed'):
                probe.unchanged(model, before)

    def test_source_closure_includes_private_core_without_historical_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); tool_dir = root / 'tools'
            path = root / 'existing.py'; private = tool_dir / 'acdd_mc2_train_batch_v1.py'
            current = types.SimpleNamespace(__file__=str(path))
            fake_sys = types.SimpleNamespace(modules={'visible': current})
            source_check = isolated_function(probe.source_files, ROOT=root, TOOLS=tool_dir, sys=fake_sys)
            stale = str(root / 'archived3447.pt')
            contract = {'file_sha256': {str(path): 'a' * 64, str(private): 'b' * 64, stale: 'c' * 64}}
            files = source_check(contract, {})
            self.assertEqual(files, {str(path): 'a' * 64, str(private): 'b' * 64})
            self.assertNotIn(stale, files)

    def test_source_binding_uses_git_bytes_not_current_unknown_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); tool_dir = root / 'tools'
            missing = root / 'pkg' / '__init__.py'
            private = str(tool_dir / 'acdd_mc2_train_batch_v1.py')
            command = Mock(return_value=b'known committed source\n')
            source_check = isolated_function(probe.source_files, ROOT=root, TOOLS=tool_dir,
                sys=types.SimpleNamespace(modules={'visible': types.SimpleNamespace(__file__=str(missing))}),
                subprocess=types.SimpleNamespace(check_output=command))
            actual = source_check({'file_sha256': {}}, {private: 'b' * 64})
            self.assertEqual(actual[str(missing)], hashlib.sha256(b'known committed source\n').hexdigest())
            command.assert_called_once_with(['git', 'show', probe.base.EXPECTED_HEAD + ':pkg/__init__.py'], cwd=root)

    def test_forward_guard_has_exact_counts_checks_budget_before_model_and_never_D_top(self):
        self.assertEqual(probe.LIMITS,
                         {'student': 1, 'teacher': 2, 'vocoder': 2, 'D_branch': 16, 'D_top': 0})
        main = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        register = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'register')
        # 只执行原入口内的登记函数，不调用main，不模拟或替换Torch。
        counts = dict.fromkeys(probe.LIMITS, 0); handles = []; gate = Mock()
        namespace = {'check': gate, 'counts': counts, 'LIMITS': probe.LIMITS, 'handles': handles}
        module = ast.Module(body=[copy.deepcopy(register)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(FILE), 'exec'), namespace)
        value = torch.ones(1, 2); student = torch.nn.Identity(); top = torch.nn.Identity()
        try:
            namespace['register'](student, 'student')
            namespace['register'](top, 'D_top')
            student(value)
            self.assertEqual(counts['student'], 1)
            with self.assertRaisesRegex(RuntimeError, 'forward limit'):
                student(value)
            with self.assertRaisesRegex(RuntimeError, 'forward limit'):
                top(value)
            self.assertEqual(counts['D_top'], 0)
            gate.side_effect = RuntimeError('capacity deadline')
            with self.assertRaisesRegex(RuntimeError, 'capacity deadline'):
                student(value)
            self.assertEqual(counts['student'], 1)
        finally:
            for handle in handles: handle.remove()

    def test_main_has_no_optimizer_update_checkpoint_save_or_old_training_entry(self):
        banned_names = {'Adam', 'AdamW', 'SGD', 'Optimizer', 'train_batch', 'train_batch_mc1',
                        'train_batch_mc2', 'actual_adam_step', 'd_update', 's_update',
                        'save_checkpoint', 'save_full_checkpoint', 'atomic_torch_save'}
        for node in ast.walk(TREE):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else None)
                self.assertNotIn(name, banned_names)
                self.assertNotEqual(name, 'step')
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    self.assertNotEqual((node.func.value.id, node.func.attr), ('torch', 'save'))
        main = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        mc_calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == 'mc2_score_distillation_loss']
        self.assertEqual(len(mc_calls), 1)
        keywords = {item.arg: item.value for item in mc_calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords['global_step']), 3928)
        self.assertIn("'not_actual_next3928_training_replay': True", SOURCE)
        self.assertIn("mc_receipt.pop('training_forward_budget')", SOURCE)
        self.assertIn("'automatic_retry_authorized': False", SOURCE)


if __name__ == '__main__':
    unittest.main()
