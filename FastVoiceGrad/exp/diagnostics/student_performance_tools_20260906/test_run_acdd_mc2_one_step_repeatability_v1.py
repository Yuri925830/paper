"""只在租用CPU运行的薄层测试；不读真实模型、不启动主入口或GPU。"""

import ast
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Run only on rental CPU with CUDA hidden')
import torch
import run_acdd_mc2_one_step_repeatability_v1 as f
import acdd_mc2_train_batch_3928_4047_v1 as mc
import test_conversion_score_step as fixture


def isolated(function, **replacements):
    """只给测试复制一个函数命名空间，不更改冻结模块或Torch全局函数。"""
    function = inspect.unwrap(function)
    namespace = dict(function.__globals__); namespace.update(replacements)
    result = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def update(model, optimizer):
    for parameter in model.parameters(): parameter.grad = torch.full_like(parameter, .25)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)


def toy():
    torch.manual_seed(713)
    student, discriminator = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr=5e-5, betas=(.5, .9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .9))
    for model, opt in ((student, optimizer), (discriminator, d_optimizer)):
        update(model, opt)
        for item in opt.state.values(): item['step'].fill_(3927)
    state = f.d.new_state(f.d.ARMS[0], 'a' * 64); state['restore_completed'] = True
    return state, student, discriminator, optimizer, d_optimizer


class RepeatabilityTests(unittest.TestCase):
    def cli(self):
        return ['--run-approved'] + [item for name in ('driver', 'tests', 'protocol') for item in ('--' + name + '-sha256', 'a' * 64)]

    def test_fresh_CPU_process_reads_real_threads_without_overriding_them(self):
        # 独立解释器不继承本测试进程的Torch线程状态；不加载模型，不初始化CUDA。
        code = '''import json
import torch
before = torch.get_num_threads()
cuda_before = torch.cuda.is_initialized()
import run_acdd_mc2_one_step_repeatability_v1 as driver
after_import = torch.get_num_threads()
runtime_before = driver.runtime()
runtime_after = driver.runtime()
print(json.dumps({'before': before, 'after_import': after_import,
    'runtime_before': runtime_before, 'runtime_after': runtime_after,
    'cuda_before': cuda_before, 'cuda_after': torch.cuda.is_initialized()}))
'''
        environment = dict(os.environ, **f.THREAD_ENV, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                           PYTHONPATH=str(f.TOOLS) + os.pathsep + str(f.ROOT))
        completed = subprocess.run([f.sys.executable, '-c', code], env=environment, cwd=f.ROOT,
                                   capture_output=True, text=True, timeout=30, check=True)
        result = json.loads(completed.stdout)
        self.assertIs(type(result['before']), int); self.assertGreater(result['before'], 0)
        self.assertEqual(result['before'], result['after_import'])
        self.assertEqual(result['runtime_before']['torch_threads'], result['before'])
        self.assertEqual(result['runtime_before'], result['runtime_after'])
        self.assertEqual(result['runtime_before']['thread_environment'], f.THREAD_ENV)
        self.assertFalse(result['cuda_before']); self.assertFalse(result['cuda_after'])
        self.assertNotIn('set_num_threads', code)

    def test_explicit_approval_full_SHA_and_child_only_fields(self):
        self.assertTrue(f.arguments(self.cli()).run_approved)
        with self.assertRaises(RuntimeError): f.arguments(self.cli()[1:])
        for index in (2, 4, 6):
            for value in ('a' * 63, 'g' * 64, 'A' * 64):
                args = self.cli(); args[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(RuntimeError): f.arguments(args)
        with self.assertRaises(RuntimeError): f.arguments(self.cli() + ['--supervisor-pid', '4'])
        with self.assertRaises(RuntimeError): f.arguments(self.cli() + ['--child', 'child_a'])
        args = self.cli() + ['--child', 'child_b', '--supervisor-pid', '4', '--shared-start', '1.25',
                            '--lock-fds', '3,4,5,6', '--contract-file-sha256', 'b' * 64]
        self.assertEqual(f.arguments(args).shared_start, 1.25)
        args[args.index('1.25')] = 'nan'
        with self.assertRaises(RuntimeError): f.arguments(args)

    def test_tensor_hash_norm_stride_and_readonly_gradient(self):
        tensor = torch.tensor([[3., 0.], [4., 0.]], requires_grad=True)
        tensor.grad = torch.full_like(tensor, .7)
        before = tensor.detach().clone(); gradient = tensor.grad.clone()
        summary = f.tensor_summary(tensor.t())
        self.assertEqual(summary['sha256'], hashlib.sha256(tensor.detach().t().contiguous().numpy().tobytes()).hexdigest())
        self.assertEqual(summary['l2_norm'], 5.); self.assertEqual(summary['stride'], [1, 2])
        self.assertTrue(torch.equal(tensor, before)); self.assertTrue(torch.equal(tensor.grad, gradient))
        self.assertTrue(tensor.requires_grad)
        self.assertFalse(f.tensor_summary(torch.tensor([float('nan')]))['finite'])
        self.assertIsNone(f.tensor_summary(torch.tensor([float('inf')]))['l2_norm'])
        with self.assertRaises(RuntimeError): f.tensor_summary([1.])
        group = f.tensor_group([('a', torch.tensor([3.])), ('b', torch.tensor([4.]))])
        self.assertEqual(group['l2_norm'], 5.); self.assertEqual(group['tensor_count'], 2)

    def test_observers_leave_real_Adam_models_moments_RNG_identical(self):
        _, s, D, a, b = toy(); initial = f.previous.content_state(s, D, a, b)
        events, published = {}, []
        check = Mock()
        with f.step_observers(s, D, a, b, events, published.append, check) as calls:
            update(D, b); update(s, a)
        observed = f.previous.content_state(s, D, a, b)
        _, s2, D2, a2, b2 = toy()
        self.assertEqual(f.previous.content_state(s2, D2, a2, b2), initial)
        update(D2, b2); update(s2, a2)
        self.assertEqual(observed, f.previous.content_state(s2, D2, a2, b2))
        self.assertEqual(calls, {'student': 1, 'discriminator': 1}); self.assertEqual(check.call_count, 4)
        self.assertEqual([r['stage'] for r in published], ['discriminator_gradient', 'discriminator_after_step', 'student_gradient', 'student_after_step'])
        self.assertEqual(events['student_after_step']['actual_cursor']['uniform_step'], 3928)
        self.assertTrue(events['student_actual_step_returned']); self.assertTrue(events['discriminator_actual_step_returned'])
        self.assertNotIn('step', a.__dict__); self.assertNotIn('step', b.__dict__)

    def test_observer_wrong_order_repeat_missing_nonfinite_or_deadline_stops_before_step(self):
        for case in ('wrong_order', 'repeat', 'missing', 'nonfinite', 'deadline'):
            _, s, D, a, b = toy(); events = {}
            checker = Mock(side_effect=RuntimeError('deadline')) if case == 'deadline' else (lambda: None)
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                with f.step_observers(s, D, a, b, events, lambda value: None, checker):
                    if case == 'wrong_order': update(s, a)
                    elif case == 'missing': b.step()
                    elif case == 'nonfinite':
                        for p in D.parameters(): p.grad = torch.full_like(p, float('inf'))
                        b.step()
                    else:
                        update(D, b)
                        if case == 'repeat': update(D, b)
            self.assertEqual(f.old.optimizer_cursor(a)['uniform_step'], 3927)
            self.assertEqual(f.old.optimizer_cursor(b)['uniform_step'], 3928 if case == 'repeat' else 3927)
            self.assertNotIn('step', a.__dict__); self.assertNotIn('step', b.__dict__)

    def test_observer_restores_preexisting_instance_step_on_failure(self):
        _, s, D, a, b = toy(); original = b.step; custom = Mock(side_effect=original); b.step = custom
        with self.assertRaisesRegex(RuntimeError, 'publish'):
            with f.step_observers(s, D, a, b, {}, Mock(side_effect=RuntimeError('publish'))): update(D, b)
        self.assertIs(b.step, custom); self.assertEqual(custom.call_count, 0)
        self.assertNotIn('step', a.__dict__)

    def test_true_return_cursor_survives_post_student_observation_failure(self):
        state, s, D, a, b = toy(); events = {}
        def publish(value):
            if value['stage'] == 'student_after_step': raise RuntimeError('post-S snapshot publication')
        def call():
            with f.step_observers(s, D, a, b, events, publish): update(D, b); update(s, a)
        with self.assertRaisesRegex(RuntimeError, 'post-S'): f.d.complete_update(state, 3928, a, b, call)
        self.assertEqual(state['completed_global_step'], 3928)
        self.assertEqual(state['adam_step_returns'], {'student': 1, 'discriminator': 1})
        self.assertTrue(state['exception_after_both_adam_steps']); self.assertFalse(state['partial_batch_may_have_updated_D'])
        self.assertTrue(events['student_actual_step_returned'])
        self.assertNotIn('step', a.__dict__); self.assertNotIn('step', b.__dict__)

    def test_actual_D_return_retained_when_deadline_prevents_S_step(self):
        state, s, D, a, b = toy(); events = {}
        def check():
            if events.get('discriminator_actual_step_returned'): raise RuntimeError('deadline before S')
        def call():
            with f.step_observers(s, D, a, b, events, lambda value: None, check): update(D, b); update(s, a)
        with self.assertRaisesRegex(RuntimeError, 'deadline'): f.d.complete_update(state, 3928, a, b, call)
        self.assertEqual(state['completed_global_step'], 3927); self.assertTrue(state['partial_batch_may_have_updated_D'])
        self.assertEqual(state['adam_step_returns'], {'student': 0, 'discriminator': 1})
        self.assertEqual(f.old.optimizer_cursor(b)['uniform_step'], 3928)

    def test_readonly3929_real_frozen_formula_calls_and_full_state_RNG_preserved(self):
        batch, T, S, V, D, diffusion, a, b, mean, std, device, config = fixture.make_fixture(torch.float32)
        before = f.previous.content_state(S, D, a, b)
        modes = [m.training for m in (S, T, V, D, diffusion)]
        with f.previous.monitoring_state({'S': S, 'T': T, 'V': V, 'D': D, 'diffusion': diffusion}, device):
            fixture.reset_rng(205163)
            result = f.probe3929(mc, S, T, diffusion, batch, config, device)
        self.assertEqual(f.previous.content_state(S, D, a, b), before)
        self.assertEqual([m.training for m in (S, T, V, D, diffusion)], modes)
        self.assertEqual((len(S.trace), len(T.trace), len(V.trace)), (1, 2, 0))
        self.assertTrue(all(not row['grad_enabled'] for row in S.trace + T.trace))
        self.assertTrue(all(p.grad is None for m in (S, T, V, D) for p in m.parameters()))
        self.assertEqual(result['mc']['private_seed'], 74003929)
        self.assertEqual(result['actual_forward_counts'], f.PROBE_COUNTS)
        self.assertEqual(len(result['mc']['first']['per_utterance_l1']), 32)
        self.assertEqual(len(result['mc']['second']['per_utterance_l1']), 32)
        self.assertAlmostEqual(result['dist_raw'], (result['mc']['dist1'] + result['mc']['dist2']) / 2, places=6)

    def test_finite_AA_and_historical_computed_differences_are_results_not_gate(self):
        a = {'mel': .191, 'target': {'sha': 'a'}, 'rows': [1., 2.], 'same': 4}
        b = {'mel': .189, 'target': {'sha': 'b'}, 'rows': [1., 2.1], 'same': 4}
        result = f.differences(a, b)
        self.assertEqual({r['path'] for r in result}, {'$.mel', '$.target.sha', '$.rows[1]'})
        mel = next(r for r in result if r['path'] == '$.mel')
        self.assertFalse(mel['within_old_numeric_tolerance']); self.assertAlmostEqual(mel['actual_minus_expected'], .002)
        self.assertEqual(f.differences(a, a), [])
        self.assertTrue(f.differences({'v': 1}, {'v': 1.})[0]['type_difference'])
        self.assertEqual(f.differences([], [1])[0]['actual_length'], 0)
        self.assertEqual(f.differences({}, {'x': 1})[0]['missing_side'], 'actual')
        for value in (float('inf'), float('nan'), -float('inf')):
            with self.assertRaises(RuntimeError): f.differences(value, value)

    def test_fixed_forward_limits_check_before_each_call_and_cleanup_all_hooks(self):
        self.assertEqual(f.CHILD_LIMITS, {'teacher': 25, 'student': 7, 'vocoder': 5, 'active_D_branches': 32, 'fixed_D_branches': 16})
        self.assertEqual(f.TOTAL_LIMITS, {k: 2 * v for k, v in f.CHILD_LIMITS.items()})
        ledger = f.Ledger(); calls = []; ledger.check = lambda: calls.append('check')
        hook = ledger.before('student')
        for unused in range(7): hook(None, None)
        with self.assertRaises(RuntimeError): hook(None, None)
        self.assertEqual(ledger.counts['student'], 7); self.assertEqual(len(calls), 8)
        ledger.check = Mock(side_effect=RuntimeError('clock'))
        with self.assertRaisesRegex(RuntimeError, 'clock'): ledger.before('teacher')(None, None)
        self.assertEqual(ledger.counts['teacher'], 0)
        done = []
        def action(name, fail=False):
            def run():
                done.append(name)
                if fail: raise RuntimeError(name)
            return run
        errors = f.old.cleanup_resources(SimpleNamespace(close=action('hooks', True)), SimpleNamespace(close=action('clock')),
            SimpleNamespace(flush=action('flush', True), close=action('log')), [SimpleNamespace(close=action('lock'))])
        self.assertEqual(done, ['hooks', 'clock', 'flush', 'log', 'lock']); self.assertEqual(len(errors), 2)

    def test_budget_all_logs_no_models_shared32MiB_system_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); budget = f.Budget(output)
            (output / 'child_a.log').write_text('x' * 100)
            (output / 'child_a').mkdir(); (output / 'child_a' / 'results.json').write_text('{}')
            self.assertEqual(budget.used(), 102)
            fake_pair = SimpleNamespace(available_bytes=lambda p: 10 * 1024**3, InsufficientDisk=f.pair.InsufficientDisk)
            require = isolated(f.Budget.require, pair=fake_pair)
            self.assertEqual(require(budget)['required_bytes'], f.previous.RESERVE + f.previous.DIR_RESERVE + f.META_LIMIT - 102)
            with self.assertRaises(RuntimeError): require(budget, f.META_LIMIT - f.FAILURE_RESERVE - 101)
            budget.aborting = True; self.assertEqual(require(budget, 7)['required_bytes'], f.previous.RESERVE + f.previous.DIR_RESERVE + 7)
            low = SimpleNamespace(available_bytes=lambda p: 0, InsufficientDisk=f.pair.InsufficientDisk)
            with self.assertRaises(f.pair.InsufficientDisk): isolated(f.Budget.require, pair=low)(budget)
            (output / 'forbidden.pt').write_bytes(b'not a model')
            with self.assertRaises(RuntimeError): budget.used()

    def test_terminal_publication_no_postcheck_no_overwrite_and_pending_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); budget = f.Budget(output); budget.require = Mock(return_value={})
            budget.write(output / 'ordinary.json', {'x': 1}); self.assertEqual(budget.require.call_count, 2)
            budget.require.reset_mock(); budget.write(output / 'completion.json', {'status': 'completed'}, terminal=True)
            self.assertEqual(budget.require.call_count, 1)
            with self.assertRaises(RuntimeError): budget.write(output / 'completion.json', {}, terminal=True)
            with self.assertRaises(RuntimeError): budget.write(output / 'x.json', {}, terminal=True)
            with self.assertRaises(RuntimeError): budget.write(output / 'completion.json', {}, terminal=False)
            with self.assertRaises(RuntimeError): budget.write(output / 'model.pt', {})
            with self.assertRaises(ValueError): budget.write(output / 'nan.json', {'x': float('nan')})
            self.assertFalse((output / 'nan.json').exists())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); budget = f.Budget(output); budget.require = Mock(side_effect=RuntimeError('before rename'))
            with self.assertRaisesRegex(RuntimeError, 'before rename'): budget.write(output / 'completion.json', {}, terminal=True)
            self.assertFalse((output / 'completion.json').exists())

    def test_shared_contract_only_child_pid_rebinding_and_no_reset(self):
        config = SimpleNamespace(as_dict=lambda: {'lambda_fm': 2.})
        parent = {'input_identities': {'fixed': 'a'}}
        context = (None, parent, None, config, None, [{'global_step': 3928}, {'global_step': 3929}], {'feat': 'b'}, {'monitor': 'c'}, {'source': 'd'})
        args = SimpleNamespace(protocol_sha256='e' * 64)
        supervisor = isolated(f.contract_for, os=SimpleNamespace(getpid=lambda: 10))(args, context, 123.)
        child = isolated(f.contract_for, os=SimpleNamespace(getpid=lambda: 11))(args, context, 123.)
        self.assertEqual(child, dict(supervisor, supervisor_pid=11)); self.assertEqual(child['maximum_complete_updates'], 2)
        self.assertEqual(child['runtime_files'], {'source': 'd'}); self.assertNotEqual(f.contract_for(args, context, 124.), child)

    def test_inherited_four_real_FDs_same_owner_and_shared_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / ('lock%d' % n) for n in range(4)]
            handles = [path.open('w+') for path in paths]
            try:
                args = SimpleNamespace(supervisor_pid=os.getpid(), shared_start=12., lock_fds=','.join(str(h.fileno()) for h in handles))
                fake_os = SimpleNamespace(getppid=lambda: os.getpid(), fstat=os.fstat)
                inherited = isolated(f.inherited_locks, os=fake_os, LOCK_PATHS=paths)
                contract = {'supervisor_pid': os.getpid(), 'shared_monotonic_start': 12.}
                self.assertEqual(inherited(args, contract), [h.fileno() for h in handles])
                with self.assertRaises(RuntimeError): inherited(args, dict(contract, shared_monotonic_start=13.))
                with self.assertRaises(RuntimeError): inherited(args, dict(contract, supervisor_pid=1))
                args.lock_fds = ','.join([str(handles[0].fileno())] * 4)
                with self.assertRaises(RuntimeError): inherited(args, contract)
            finally:
                for handle in handles: handle.close()

    def test_wait_timeout_and_cleanup_do_not_start_retry_or_reset_clock(self):
        clock = SimpleNamespace(started=0., active=Mock())
        process = SimpleNamespace(communicate=Mock(side_effect=[subprocess.TimeoutExpired('child', 1), ('done', None)]))
        wait = isolated(f.wait_child, time=SimpleNamespace(monotonic=lambda: 10.))
        self.assertEqual(wait(process, clock), 'done'); self.assertEqual(clock.active.call_count, 2)
        clock.active = Mock(side_effect=RuntimeError('shared deadline'))
        with self.assertRaises(RuntimeError): wait(process, clock)
        stopping = SimpleNamespace(poll=Mock(side_effect=[None, 0]), terminate=Mock(), kill=Mock(),
            communicate=Mock(side_effect=[subprocess.TimeoutExpired('child', 1), ('killed', None)]))
        isolated(f.stop_child, time=SimpleNamespace(monotonic=lambda: 620.))(stopping, clock)
        self.assertEqual(stopping.terminate.call_count, 1); self.assertEqual(stopping.kill.call_count, 1)
        self.assertEqual(stopping.communicate.call_args_list[0].kwargs['timeout'], 65.)
        self.assertEqual(stopping.communicate.call_args_list[1].kwargs['timeout'], 5.)

    def test_actual_source_closure_is_finite_not_recursive_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); current = root / 'current.py'; current.write_text('x=1')
            history = str(root / 'retired.pt'); fake_sys = SimpleNamespace(modules={'current': SimpleNamespace(__file__=str(current))})
            select = isolated(f.d.imported_source_files, ROOT=root, sys=fake_sys)
            self.assertEqual(select({str(current): 'a' * 64, history: 'b' * 64}, {}), {str(current): 'a' * 64})
            with self.assertRaises(RuntimeError): select({history: 'b' * 64}, {})
            # 新私有MC2命名空间没有sys.modules登记，因此真实driver还显式pin原core源码。
            source = Path(f.__file__).read_text()
            self.assertIn("str(TOOLS / 'acdd_mc2_train_batch_v1.py')", source)
            self.assertIn('future = d.future_plan(development)[:2]', source)

    def test_static_exact_one_training_call_no_saved_model_metrics_or_global_Torch_patch(self):
        source = Path(f.__file__).read_text(); tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = [n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, 'attr', None) for n in calls]
        self.assertEqual(names.count('train_batch_mc2'), 1); self.assertEqual(names.count('Popen'), 1)
        for name in ('save', 'unlink', 'rmdir', 'rmtree', 'remove', 'infer_candidate', 'evaluate_metrics', 'set_num_threads',
                     'use_deterministic_algorithms', 'restore_flags', 'save_endpoint', 'backward'):
            self.assertNotIn(name, names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name): self.assertNotEqual(target.value.id, 'torch')
        probe_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'probe3929')
        self.assertEqual(ast.dump(probe_node.decorator_list[0]), ast.dump(ast.parse('torch.no_grad()', mode='eval').body))
        probe_calls = [getattr(n.func, 'attr', '') for n in ast.walk(probe_node) if isinstance(n, ast.Call)]
        self.assertNotIn('step', probe_calls); self.assertNotIn('train_batch_mc2', probe_calls)
        self.assertIn("for name in CHILDREN:", source); self.assertNotIn('start_new_session=True', source)
        self.assertEqual(source.count("'phase': 'terminal_publication'"), 2)
        self.assertEqual(source.count('original_error is None and not errors and result is not None'), 2)
        self.assertIn('self.check()', inspect.getsource(f.Ledger.before))
        self.assertIn('check(); result = _original()', inspect.getsource(f.step_observers))


if __name__ == '__main__': unittest.main()
