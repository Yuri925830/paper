"""Rental CPU tiny fixtures only; no real locks, backup, model load or deletion."""
import ast
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.name != 'posix' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Rental GPU-hidden CPU only')
import retire_mc2_dlr_control91_v1 as n
r = n.r


class Control91Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='control91_toy_'); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / 'project'; dev = root / 'exp/diagnostics/student_development_20260906'
        run, ev = dev / n.RUN, dev / n.EVAL; system = ev / n.LABEL
        (run / n.ARM).mkdir(parents=True); (system / 'wavs').mkdir(parents=True); (system / 'metrics').mkdir()
        locks = root / 'exp/locks'; locks.mkdir()
        for name in n.LOCK_NAMES: (locks / name).write_text('')
        self.s = dict(root=root, dev=dev, run=run, evaluation=ev, audit=dev / n.AUDIT, proof=dev / n.PROOF,
            private=n.PRIVATE, pub=Path(self.temp.name) / 'publication', publication_counts=(2, 1),
            protected={}, source_sha=r.sha(Path(n.__file__)), proof_sha='')
        kept = root / 'protected_full.pt'; kept.write_bytes(b'protected3927'); self.s['protected'][str(kept)] = r.sha(kept)
        self.target = run / n.ARM / 'student_step_004047.pt'; self.target.write_bytes(b'tiny-control'); self.target.chmod(0o444)
        self.targets = [self.target]
        for i in range(90):
            path = system / 'wavs' / ('pair_%04d.wav' % i); path.write_bytes(('toy-wav-%d' % i).encode()); self.targets.append(path)
        joint = {'runtime_files': dict(self.s['protected']), 'prospective_design': 'unchanged'}
        self.put(run / 'run_contract.json', joint); canonical = n.canonical(joint)
        snapshot = dict(path=str(self.target), sha256=r.sha(self.target), global_step=4047, committed=True, resumable=False, frozen_readonly=True)
        state = dict(arm=n.ARM, completed_global_step=4047, restore_completed=True, last_student_saved_global_step=4047,
            student_snapshot=snapshot, partial_batch_may_have_updated_D=False, execution_error=None)
        self.put(run / n.ARM / 'completion.json', dict(status='completed120', cleanup_verified=True, contract_sha256=canonical, state=state))
        (run / n.ARM / 'step_losses.jsonl').write_text(''.join(json.dumps({'global_step': i}) + '\n' for i in range(3928, 4048)))
        (run / 'control_d2e4.log').write_text('preserved log\n')
        original = [{'pair_id': str(i), 'source': 'original-%d' % i} for i in range(90)]
        self.csv(dev / 'pairs.csv', original)
        self.csv(system / 'pairs.csv', [dict(x, output_wav=str(system / 'wavs' / ('pair_%04d.wav' % int(x['pair_id'])))) for x in original])
        spec = dict(checkpoint=str(self.target), sha256=r.sha(self.target), state_key='student', epoch=0, global_step=4047, steps=1, sampling='existing_student_t950_to_zero')
        identity = dict(purpose='development_candidate_not_formal_8100', pairs=90, candidate_training_contract=str(run / 'run_contract.json'),
            start_step=950, inference_batch_size=1, systems={n.LABEL: spec},
            inputs={str(self.target): r.sha(self.target), str(run / 'run_contract.json'): r.sha(run / 'run_contract.json'), **self.s['protected']})
        self.put(ev / 'run_identity.json', identity)
        inference = dict(status='development_candidate_only', system=n.LABEL, spec=spec, saved_training_contract_sha256=canonical,
            wav_sha256={str(i): r.sha(self.targets[i + 1]) for i in range(90)}, run_identity_sha256=r.sha(ev / 'run_identity.json'),
            pairs_csv_sha256=r.sha(system / 'pairs.csv'))
        self.put(system / 'inference_complete.json', inference)
        metric_shas = {}
        for name in ('cer', 'sva', 'dnsmos', 'utmos'):
            rows = [{'pair_id': str(i), 'reference_chars': str(49 if i < 72 else 48), 'threshold': '.25'} for i in range(90)]
            self.csv(system / 'metrics' / (name + '.csv'), rows); metric_shas[name] = r.sha(system / 'metrics' / (name + '.csv'))
        self.put(system / 'metrics/development_summary_cer_dnsmos_sva_utmos.json', dict(status='development_only', evaluated_pairs=90,
            metric_identity_sha256=n.METRIC_SHA, system=n.LABEL, inference_complete_sha256=r.sha(system / 'inference_complete.json'),
            run_identity_sha256=r.sha(ev / 'run_identity.json'), metric_csv_sha256=metric_shas))
        self.proof = dict(status='verified_private_backup', scp_exit_code=0, all_local_and_remote_sha256_verified=True,
            local_backups_read_only=True, remote_files_not_yet_removed=True,
            identity_sha256={str(p): r.sha(p) for p in n.paths_for(self.s).values()}, target_files=[])
        for path in self.targets:
            info = path.stat(); local = n.PRIVATE / (path.name if path == self.target else 'wavs/' + path.name)
            self.proof['target_files'].append(dict(remote_path=str(path), local_path=local.as_posix(), sha256=r.sha(path), local_sha256=r.sha(path),
                size_bytes=info.st_size, local_size_bytes=info.st_size, allocated_bytes_remote=info.st_blocks * 512,
                source_mode=oct(info.st_mode & 0o777), source_nlink=1, local_read_only=True, source_identity_verified_after_copy=True,
                source_stat_decimal_strings=[str(v) for v in r.stamp(path)]))
        self.write_proof(); self.publication('/unrelated', 'safe.py')

    def put(self, path, value): path.write_text(json.dumps(value))

    def csv(self, path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def write_proof(self):
        self.put(self.s['proof'], self.proof); self.s['proof_sha'] = r.sha(self.s['proof'])

    def publication(self, source, name):
        folder = self.s['pub'] / 'direct-upload-runtime/lists'; folder.mkdir(parents=True, exist_ok=True)
        raw = folder / 'one.files-from-raw.txt'; metadata = folder / 'one.metadata.jsonl'
        raw.write_text(name + '\n'); metadata.write_text(json.dumps({'source_path': source + '/' + name, 'relative_path': name}) + '\n')
        self.put(self.s['pub'] / 'PLAN.json', dict(source_root=source, file_list_path=str(raw), file_list_sha256=r.sha(raw),
            files_manifest_path=str(metadata), files_manifest_sha256=r.sha(metadata)))

    def call(self, apply=False):
        with mock.patch.object(n.old, 'gpu_guard', return_value={'compute_processes': 0}), \
             mock.patch.object(r, 'process_guard', return_value={'processes_checked': 1, 'references': 0}), \
             contextlib.redirect_stdout(io.StringIO()):
            return n.execute(self.s, apply)

    def failure(self):
        self.assertFalse((self.s['audit'] / 'completion.json').exists())
        self.assertFalse((self.s['run'] / 'control_archive_receipt.json').exists())
        return json.loads((self.s['audit'] / 'failure.json').read_text())

    def test_default_readonly_no_torch_no_actual_locks_no_arbitrary_paths(self):
        with mock.patch.object(n, 'unlink_one', side_effect=AssertionError('readonly')): result = self.call()
        self.assertEqual(result['status'], 'validated_not_deleted'); self.assertFalse(self.s['audit'].exists())
        self.assertTrue(all(p.exists() for p in self.targets)); self.assertNotIn('torch', sys.modules)
        self.assertFalse(n.parse_args(['--source-sha256', 'a', '--backup-receipt-sha256', 'b']).apply)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): n.parse_args(['--target', '/tmp/no'])

    def test_91_exact_targets_removed_all_metadata_preserved_gate_compatible(self):
        before = n.metadata_snapshot(self.s); result = self.call(True)
        self.assertEqual(result['removed_count'], 91); self.assertTrue(all(not p.exists() for p in self.targets))
        after = n.metadata_snapshot(self.s); after.pop(str(self.s['run'] / 'control_archive_receipt.json'))
        self.assertEqual(before, after); self.assertTrue(all(p.parent.is_dir() for p in self.targets))
        journal = [json.loads(x) for x in (self.s['audit'] / 'removed_files.jsonl').read_text().splitlines()]
        self.assertEqual([x['phase'] for x in journal], ['prepared', 'removed'] * 91)
        # Exact frozen candidate_gate/check_files/csv_rows bodies, no driver module import.
        names = {'validate_control_archive', 'candidate_gate', 'check_files', 'csv_rows'}
        nodes = [x for x in ast.parse(n.DRIVER.read_text()).body if isinstance(x, ast.FunctionDef) and x.name in names]
        self.assertEqual({x.name for x in nodes}, names)
        namespace = dict(Path=Path, PureWindowsPath=PureWindowsPath, json=json, csv=csv,
            training=SimpleNamespace(canonical_json_sha256=n.canonical),
            base=SimpleNamespace(sha256_file=r.sha), cross=SimpleNamespace(regular=r.regular, read_json=lambda p: json.loads(Path(p).read_text())),
            OUTPUT=self.s['run'], DEV=self.s['dev'], ARMS=(n.ARM, 'candidate_d1e4'), START=3927,
            ARCHIVE_NAME='control_archive_receipt.json', METRIC_IDENTITY_SHA=n.METRIC_SHA,
            EVALUATION={n.ARM: {'output_root': str(self.s['evaluation']), 'label': n.LABEL}})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(n.DRIVER), 'exec'), namespace)
        args = SimpleNamespace(control_archive_sha256=r.sha(self.s['run'] / 'control_archive_receipt.json'))
        steps, _ = namespace['candidate_gate'](args, json.loads((self.s['run'] / 'run_contract.json').read_text()))
        self.assertEqual(sorted(steps), list(range(3928, 4048))); self.assertNotIn('torch', sys.modules)

    def test_bad_proof_scope_size_private_path_or_stat_blocks(self):
        original = copy.deepcopy(self.proof)
        for field in ('duplicate', 'missing', 'private', 'sha', 'size_bool', 'mode', 'stat_number', 'readonly'):
            with self.subTest(field=field):
                self.proof = copy.deepcopy(original); row = self.proof['target_files'][0]
                if field == 'duplicate': self.proof['target_files'][-1] = row
                if field == 'missing': self.proof['target_files'].pop()
                if field == 'private': row['local_path'] += '.partial'
                if field == 'sha': row['local_sha256'] = '0' * 64
                if field == 'size_bool': row['size_bytes'] = True
                if field == 'mode': row['source_mode'] = '0o644'
                if field == 'stat_number': row['source_stat_decimal_strings'][0] = 1
                if field == 'readonly': row['local_read_only'] = False
                self.write_proof()
                with self.assertRaises(RuntimeError): self.call()
                self.assertTrue(all(p.exists() for p in self.targets))
        self.proof = original; self.write_proof()

    def test_wrong_scope_and_source_sha_block(self):
        original = self.s['run']; self.s['run'] = original.with_name('wrong')
        with self.assertRaisesRegex(RuntimeError, 'scope'): self.call()
        self.s['run'] = original; self.s['source_sha'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'source SHA'): self.call()

    def test_gate_rejects_technical_failure_and_wrong_endpoint(self):
        path = n.paths_for(self.s)['control_completion_file_sha256']; control = json.loads(path.read_text())
        control['state']['execution_error'] = 'failure'; self.put(path, control)
        self.proof['identity_sha256'][str(path)] = r.sha(path); self.write_proof()
        with self.assertRaisesRegex(RuntimeError, 'technically failed'): self.call()

    def test_metric_chain_csv_tampering_rejects(self):
        path = self.s['evaluation'] / n.LABEL / 'metrics/cer.csv'; path.write_text('pair_id,reference_chars\n0,1\n')
        with self.assertRaisesRegex(RuntimeError, 'Metric CSV'): self.call()

    def test_private_proof_sha_failure_is_persistent_no_retry(self):
        self.s['proof_sha'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'proof SHA'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 0); self.write_proof()
        with self.assertRaisesRegex(RuntimeError, 'never repeat'): self.call(True)

    def test_same_inode_symlink_and_target_byte_change_block(self):
        path = self.targets[1]; link = path.with_name('alias.wav'); os.link(path, link)
        with self.assertRaisesRegex(RuntimeError, 'target changed'): self.call()
        link.unlink(); self.proof['target_files'][1]['source_stat_decimal_strings'] = [str(v) for v in r.stamp(path)]; self.write_proof()
        destination = path.with_name('moved.wav'); path.rename(destination); path.symlink_to(destination)
        with self.assertRaisesRegex(RuntimeError, 'archive/removal proof|symlink'): self.call()

    def test_publication_reference_and_live_reader_block(self):
        self.publication(str(self.target.parent), self.target.name)
        with self.assertRaisesRegex(RuntimeError, 'reference'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 0)

    def test_changed_immediately_before_unlink_refuses_and_records_partial_cursor(self):
        original = n.unlink_one; calls = []
        def changed(row, journal, removed):
            if len(calls) == 3: row['path'].write_bytes(b'changed-before-unlink')
            calls.append(row['path']); return original(row, journal, removed)
        with mock.patch.object(n, 'unlink_one', side_effect=changed), self.assertRaisesRegex(RuntimeError, 'target changed'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 3)
        with self.assertRaisesRegex(RuntimeError, 'never repeat'): self.call(True)

    def test_postunlink_failure_reports_actual_removed_cursor(self):
        original = n.old.journal_row
        def journal(handle, value):
            if value['phase'] == 'removed': raise OSError('after unlink')
            return original(handle, value)
        with mock.patch.object(n.old, 'journal_row', side_effect=journal), self.assertRaisesRegex(OSError, 'after unlink'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 1)

    def test_metadata_change_and_protected_model_change_cannot_complete(self):
        original = n.unlink_one
        def changed(*args):
            original(*args); (self.s['run'] / 'control_d2e4.log').write_text('changed log')
        with mock.patch.object(n, 'unlink_one', side_effect=changed), self.assertRaisesRegex(RuntimeError, 'metadata'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 1)

    def test_lock_cleanup_failure_closes_every_handle_no_success(self):
        opening = Path.open; closed = []
        class Handle:
            def __init__(self, actual, name): self.actual, self.name = actual, name
            def fileno(self): return self.actual.fileno()
            def close(self):
                self.actual.close(); closed.append(self.name)
                if self.name == n.LOCK_NAMES[-1]: raise OSError('lock cleanup')
        def wrapped(path, *args, **kwargs):
            value = opening(path, *args, **kwargs)
            return Handle(value, path.name) if path.name in n.LOCK_NAMES and args and args[0] == 'r+' else value
        with mock.patch.object(Path, 'open', wrapped), self.assertRaisesRegex(OSError, 'lock cleanup'): self.call(True)
        self.assertEqual(set(closed), set(n.LOCK_NAMES)); self.assertEqual(self.failure()['removed_count'], 91)

    def test_archive_publication_failure_invalidates_audit_success(self):
        with mock.patch.object(n, 'publish_archive', side_effect=OSError('archive publication')), self.assertRaisesRegex(OSError, 'publication'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 91)
        self.assertTrue((self.s['audit'] / 'completion.invalidated.json').exists())

    def test_runtime_protected_change_after_removal_blocks_completion(self):
        original = n.unlink_one; kept = Path(next(iter(self.s['protected'])))
        def changed(*args):
            original(*args)
            if len(args[-1]) == 91: kept.write_bytes(b'changed protected asset')
        with mock.patch.object(n, 'unlink_one', side_effect=changed), self.assertRaisesRegex(RuntimeError, 'Protected runtime'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 91)

    def test_archive_postrename_fsync_failure_invalidates_both_success_files(self):
        original = os.fsync; archive = self.s['run'] / 'control_archive_receipt.json'
        def sync(fd):
            if archive.exists(): raise OSError('archive directory fsync')
            return original(fd)
        with mock.patch.object(os, 'fsync', side_effect=sync), self.assertRaisesRegex(OSError, 'directory fsync'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 91)
        self.assertTrue((self.s['audit'] / 'completion.invalidated.json').exists())
        self.assertTrue((self.s['audit'] / 'control_archive.invalidated.json').exists())

    def test_real_process_same_inode_guard_and_gpu_failure(self):
        proc = Path(self.temp.name) / 'proc/999999'; (proc / 'fd').mkdir(parents=True)
        (proc / 'cmdline').write_bytes(b'worker'); (proc / 'maps').write_bytes(b'')
        (proc / 'cwd').symlink_to(self.s['root']); (proc / 'exe').symlink_to('/bin/true')
        (proc / 'fd/3').symlink_to(self.target); info = self.target.stat()
        with self.assertRaisesRegex(RuntimeError, 'holds checkpoint inode'):
            r.process_guard(self.targets, {(info.st_dev, info.st_ino)}, proc.parent)
        with mock.patch.object(n.old, 'gpu_guard', side_effect=RuntimeError('GPU active')), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'GPU active'): n.execute(self.s)
        self.assertFalse(self.s['audit'].exists())

    def test_later_live_reader_aborts_partial_retirement(self):
        calls = []
        def guard(*args):
            calls.append(1)
            if len(calls) == 6: raise RuntimeError('new live reader')
            return {'references': 0}
        with mock.patch.object(n.old, 'gpu_guard', return_value={}), mock.patch.object(r, 'process_guard', side_effect=guard), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'new live reader'): n.execute(self.s, True)
        self.assertEqual(self.failure()['removed_count'], 3)


if __name__ == '__main__': unittest.main(verbosity=2)
