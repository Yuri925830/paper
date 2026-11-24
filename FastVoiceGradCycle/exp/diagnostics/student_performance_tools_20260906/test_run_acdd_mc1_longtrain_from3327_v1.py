"""租用CPU小测试；toy模型/临时文件，不执行main、不加载真实训练权重。"""
import ast
import copy
import inspect
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Tests are restricted to rental CPU with CUDA hidden')
import torch
import run_acdd_mc1_longtrain_from3327_v1 as f


def isolated(function, **replace):
    function = inspect.unwrap(function); namespace = dict(function.__globals__); namespace.update(replace)
    return types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


def update(model, optimizer):
    for p in model.parameters(): p.grad = torch.ones_like(p)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)


def raw():
    return {'student_total': 5.5, 'student_adv': 1., 'fm_raw': 1., 'fm_weighted': 2.,
            'dist_raw': 2.5 / 45., 'dist_weighted': 2.5, 'discriminator': 2.}


def toy(step=3327):
    torch.manual_seed(711)
    s, D = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    a = torch.optim.Adam(s.parameters(), lr=5e-5, betas=(.5, .9))
    b = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(.5, .9))
    for model, opt in ((s, a), (D, b)):
        update(model, opt)
        for item in opt.state.values(): item['step'].fill_(step)
    state = {'completed_global_step': step, 'last_saved_global_step': 3327, 'partial_batch': False,
        'optimizer_cursors': {'student': f.optimizer_cursor(a), 'discriminator': f.optimizer_cursor(b)}}
    return state, s, D, a, b


class LongTrainTests(unittest.TestCase):
    def test_explicit_approval_and_source_SHA(self):
        args = ['--run-approved', '--driver-sha256', 'a' * 64, '--tests-sha256', 'b' * 64]
        self.assertTrue(f.arguments(args).run_approved)
        with self.assertRaises(RuntimeError): f.arguments(args[1:])
        for index in (2, 4):
            bad = list(args); bad[index] = 'z' * 64
            with self.assertRaises(RuntimeError): f.arguments(bad)

    def test_global_plan100_passes_first_resume_and_boundaries(self):
        plan = f.resume.GlobalBatchPlan(list(range(39335)))
        self.assertEqual(plan.batches_per_pass, 1229); self.assertEqual(f.STOP, 122900)
        self.assertEqual(f.STOP-f.START, 119573)
        first = plan.for_step(3328); self.assertEqual(len(first), 32); self.assertEqual(plan.pass_index, 2)
        self.assertEqual((101234+3328, 201234+3328), (104562, 204562))
        self.assertEqual(first, plan.order[869*32:870*32])
        self.assertEqual(len(plan.for_step(122900)), 32); self.assertEqual(plan.pass_index, 99)

    def test_complete_D_then_S_and_half_or_poststep_failure_cursors(self):
        for mode in ('normal', 'half', 'afterboth'):
            state, s, D, a, b = toy()
            def call():
                update(D, b)
                if mode == 'half': raise RuntimeError('between D and S')
                update(s, a)
                if mode == 'afterboth': raise RuntimeError('after both')
                return raw()
            if mode == 'normal': self.assertEqual(f.complete_batch(state, 3328, a, b, call), raw())
            else:
                with self.assertRaises(RuntimeError): f.complete_batch(state, 3328, a, b, call)
            self.assertEqual(state['completed_global_step'], 3327 if mode == 'half' else 3328)
            self.assertEqual(state['partial_batch'], mode == 'half')
            self.assertNotIn('step', a.__dict__); self.assertNotIn('step', b.__dict__)

    def test_nonuniform_cursor_does_not_hide_original_update_exception(self):
        state, s, D, a, b = toy()
        def call():
            next(iter(b.state.values()))['step'].add_(1)
            raise ValueError('original optimizer failure')
        with self.assertRaisesRegex(ValueError, 'original optimizer failure'): f.complete_batch(state, 3328, a, b, call)
        self.assertIn('cursor_inspection_error', state); self.assertTrue(state['partial_batch'])

    def parent_fixture(self):
        state, s, D, a, b = toy(); cfg = f.replace(f.baseline.PAPER_STUDENT_CONFIG, num_workers=0)
        parent = {'configuration': cfg.as_dict()}; sha = f.training.canonical_json_sha256(parent)
        scores = {'mel_l1': .19}; receipt = {'completed_global_step': 3327, 'last_full_saved_global_step': 3327,
            'partial_batch_may_have_updated_D': False, 'full_snapshot': {'sha256': f.PARENT_SHA}, 'last_monitor': {'development': scores}}
        payload = {'global_step': 3327, 'run_contract_sha256': sha, 'effective_optimizer_lrs': dict(f.LRS),
            'configuration': cfg.as_dict(), 'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar', 'development': scores,
            'student': copy.deepcopy(s.state_dict()), 'discriminator': copy.deepcopy(D.state_dict()),
            'student_optimizer': copy.deepcopy(a.state_dict()), 'discriminator_optimizer': copy.deepcopy(b.state_dict()),
            'rng_state': f.training.capture_rng_state()}
        validator = isolated(f.recovery.validate_parent, CONTRACT_SHA=sha)
        recovery = SimpleNamespace(**dict(f.recovery.__dict__, validate_parent=validator))
        return state, s, D, a, b, cfg, parent, receipt, payload, isolated(f.restore, recovery=recovery)

    def test_real_parent_S_D_dualAdam_RNG_restore_no_alias(self):
        state, s, D, a, b, cfg, parent, receipt, payload, restore = self.parent_fixture()
        before = {role: f.probe.tree_sha(payload[role]) for role in ('student','discriminator','student_optimizer','discriminator_optimizer')}
        update(s, a); update(D, b); torch.rand(3)
        self.assertEqual(restore(payload, s, D, a, b, cfg, {}, parent, receipt, torch.device('cpu'), True), 3327)
        for role, value in (('student',s.state_dict()),('discriminator',D.state_dict()),('student_optimizer',a.state_dict()),('discriminator_optimizer',b.state_dict())):
            self.assertEqual(f.probe.tree_sha(value), before[role])
        self.assertEqual(f.probe.tree_sha(f.training.capture_rng_state()), f.probe.tree_sha(payload['rng_state']))
        for role, opt in (('student',a),('discriminator',b)):
            for index, values in opt.state_dict()['state'].items():
                self.assertNotEqual(values['exp_avg'].data_ptr(), payload[role+'_optimizer']['state'][index]['exp_avg'].data_ptr())

    def test_parent_wrong_actual_LR_or_moment_refused(self):
        for case in ('lr', 'moment', 'step'):
            state, s, D, a, b, cfg, parent, receipt, payload, restore = self.parent_fixture()
            if case == 'lr': payload['effective_optimizer_lrs']['student'] = 2e-4
            if case == 'moment': next(iter(payload['student_optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
            if case == 'step': payload['global_step'] = 3328
            with self.assertRaises(RuntimeError): restore(payload, s, D, a, b, cfg, {}, parent, receipt, torch.device('cpu'), True)

    def save_fixture(self, directory, step=4000):
        state, s, D, a, b = toy(step); cfg = f.replace(f.baseline.PAPER_STUDENT_CONFIG, num_workers=0)
        contract = {'configuration': cfg.as_dict(), 'kind': 'toy_MC1_longtrain'}
        save = isolated(f.save_latest, require_space=Mock(return_value={}))
        record = save(Path(directory), state, s, D, a, b, cfg, contract, {'mel_l1': .19})
        return state, s, D, a, b, cfg, contract, record, save

    def test_rolling_atomic_replacement_onlyone_full_complete_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            state, s, D, a, b, cfg, contract, record, save = self.save_fixture(directory)
            path = Path(directory) / 'latest_full.pt'; first = f.digest(path)
            payload = f.resume.load_torch_checkpoint(path)
            self.assertEqual(f.validate_resume(payload, contract), 4000); self.assertEqual(payload['mc_samples'], 1)
            for model, opt in ((s,a),(D,b)): update(model,opt)
            state.update(completed_global_step=4001, optimizer_cursors={'student':f.optimizer_cursor(a),'discriminator':f.optimizer_cursor(b)})
            save(Path(directory),state,s,D,a,b,cfg,contract,{'mel_l1':.20})
            self.assertNotEqual(f.digest(path), first); self.assertEqual(len(list(Path(directory).glob('*.pt'))),1)
            self.assertFalse((Path(directory)/'latest_full.pt.tmp').exists())
            self.assertEqual(f.read(Path(directory)/'latest_full.json')['global_step'],4001)
            self.assertEqual(len((Path(directory)/'checkpoint_history.jsonl').read_text().splitlines()),2)

    def test_checkpoint_fsync_precedes_replace_and_directory_fsync(self):
        with tempfile.TemporaryDirectory() as directory:
            events=[]; path=Path(directory)/'latest_full.pt'
            def fsync(fd): events.append('fsync'); os.fsync(fd)
            def replace(source,target): events.append('replace'); os.replace(source,target)
            facade=SimpleNamespace(fsync=fsync,replace=replace,open=os.open,close=os.close,O_RDONLY=os.O_RDONLY,O_DIRECTORY=os.O_DIRECTORY)
            isolated(f.durable_torch_save,os=facade)({'x':torch.ones(1)},path)
            self.assertEqual(events,['fsync','replace','fsync'])
            self.assertTrue(torch.equal(f.resume.load_torch_checkpoint(path)['x'],torch.ones(1)))

    def test_cleanup_attempts_every_lock_after_an_error(self):
        events=[]
        def close_bad(): events.append('bad'); raise OSError('close')
        bad=SimpleNamespace(close=close_bad); good=SimpleNamespace(close=lambda:events.append('good'))
        errors=f.release_runtime({},[good,bad])
        self.assertEqual(events,['bad','good']); self.assertEqual(len(errors),1)

    def test_failed_pointer_after_atomic_commit_is_recoverable_from_newfull(self):
        with tempfile.TemporaryDirectory() as directory:
            state,s,D,a,b,cfg,contract,record,save = self.save_fixture(directory)
            for model,opt in ((s,a),(D,b)): update(model,opt)
            state.update(completed_global_step=4001,optimizer_cursors={'student':f.optimizer_cursor(a),'discriminator':f.optimizer_cursor(b)})
            failed = isolated(f.save_latest,require_space=Mock(return_value={}),write=Mock(side_effect=OSError('pointer write failed')))
            with self.assertRaisesRegex(OSError,'pointer write failed'): failed(Path(directory),state,s,D,a,b,cfg,contract,{'mel_l1':.2})
            self.assertEqual(state['last_saved_global_step'],4001)
            path,sha,payload = f.choose_checkpoint(Path(directory),contract)
            self.assertEqual(payload['global_step'],4001); self.assertEqual(sha,f.digest(path))
            self.assertEqual(f.read(Path(directory)/'latest_full.json')['global_step'],4000)

    def test_restart_full_strict_restore_and_conflicting_pointer_or_partial_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            state,s,D,a,b,cfg,contract,record,save = self.save_fixture(directory)
            path,sha,payload = f.choose_checkpoint(Path(directory),contract)
            update(s,a); update(D,b)
            self.assertEqual(f.restore(payload,s,D,a,b,cfg,contract,{}, {},torch.device('cpu'),False),4000)
            self.assertEqual(f.optimizer_cursor(a)['uniform_step'],4000)
            payload['partial_batch'] = True
            with self.assertRaises(RuntimeError): f.validate_resume(payload,contract)
            f.write(Path(directory)/'latest_full.json',dict(record,global_step=4001))
            with self.assertRaises(RuntimeError): f.choose_checkpoint(Path(directory),contract)

    def test_half_or_technical_failure_never_saves_model(self):
        for key in ('partial_batch','execution_error'):
            state,s,D,a,b = toy(4000); state[key] = True
            cfg = f.replace(f.baseline.PAPER_STUDENT_CONFIG,num_workers=0)
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError): f.save_latest(Path(directory),state,s,D,a,b,cfg,{}, {})
                self.assertEqual(list(Path(directory).iterdir()),[])

    def test_rolling_disk_accounting_two_full_initial_and_one_pending_later(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_os = SimpleNamespace(statvfs=lambda p:SimpleNamespace(f_bavail=10*1024**3,f_frsize=1))
            require = isolated(f.require_space,os=fake_os)
            self.assertEqual(require(root)['required_bytes'],3592137216)
            (root/'latest_full.pt').write_bytes(b'x'*1000)
            allocated = max((root/'latest_full.pt').stat().st_size,(root/'latest_full.pt').stat().st_blocks*512)
            self.assertEqual(require(root)['required_bytes'],3592137216-allocated)
            fake_os.statvfs = lambda p:SimpleNamespace(f_bavail=1,f_frsize=1)
            with self.assertRaises(RuntimeError):require(root)

    def test_current_source_only_no_ancestor_full_or_short_gate_or_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'active.py'; source.write_text('x=1')
            sys_copy=SimpleNamespace(modules={'active':SimpleNamespace(__file__=str(source))})
            select=isolated(f.source_files,ROOT=root,sys=sys_copy)
            self.assertEqual(select({'file_sha256':{str(source):f.digest(source),str(root/'retired.pt'):'a'*64}},{}),{str(source):f.digest(source)})
        source=Path(f.__file__).read_text(); tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        names=[n.func.id if isinstance(n.func,ast.Name) else getattr(n.func,'attr',None) for n in calls]
        self.assertEqual(names.count('train_batch'),1)
        for name in ('train_batch_mc2','stage1_gate','trend_state','infer_candidate','evaluate_metrics','backward'):
            self.assertNotIn(name,names)
        self.assertEqual((f.START,f.STOP,f.SAVE_EVERY,f.STATUS_EVERY),(3327,122900,1000,20))
        self.assertIn('torch.set_num_threads(2)',source)
        self.assertIn('time.monotonic() - train_started',source)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__': unittest.main()
