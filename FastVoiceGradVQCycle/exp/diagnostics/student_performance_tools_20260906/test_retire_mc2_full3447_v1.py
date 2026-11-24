"""仅租用CPU临时目录的几字节夹具；绝不调用生产apply或读取模型。"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only GPU-hidden rental CPU tests')
import retire_mc2_full3447_v1 as n
r = n.r


class Retirement3447Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='retire3447_toy_')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / 'project'; dev = root / 'exp/diagnostics/student_development_20260906'
        runs = [dev / name for name in n.RUN_NAMES]
        target = runs[0] / 'candidate_mc2/final_full.pt'; target.parent.mkdir(parents=True)
        target.write_bytes(b'full3447'); target.chmod(0o444)
        for run in runs:
            run.mkdir(exist_ok=True)
            (run / 'run_contract.json').write_text(json.dumps({'historical_parent': str(target)}))
            run.with_suffix('.log').write_text('old immutable log\n')
        locks = root / 'exp/locks'; locks.mkdir(parents=True)
        for name in n.LOCK_NAMES: (locks / name).write_text('')
        kept = runs[1] / 'final_full.pt'; kept.write_bytes(b'protected3927')
        self.meta = runs[1] / 'run_contract.json'
        pub = Path(self.temp.name) / 'publication'; lists = pub / 'direct-upload-runtime/lists'; lists.mkdir(parents=True)
        self.raw = lists / 'one.files-from-raw.txt'; self.manifest = lists / 'one.metadata.jsonl'
        self.plan = pub / 'PLAN.json'
        self.set_publication('/unrelated', 'safe.py')
        evaluation = dev / 'evaluation_mc2_paired120_endpoint_v1'; evaluation.mkdir()
        (evaluation / 'run_identity.json').write_text(json.dumps({'checkpoint': str(target)}))
        (evaluation / 'scores.csv').write_text('pair_id,score\n1,0.5\n')
        wavs = evaluation / 'wavs'; wavs.mkdir(); (wavs / 'old.wav').write_bytes(b'not read as metadata')
        queue = dev / 'RUN_QUEUE.md'; queue.write_text('historical ' + str(target))
        self.s = dict(root=root, dev=dev, pub=pub, target=target, target_sha=r.sha(target), logical=target.stat().st_size,
            allocated=target.stat().st_blocks * 512, proof=dev / 'proof.json', proof_sha='', private='C:/private/final_full.pt',
            audit=dev / 'mc2_full3447_retirement_v1', protected={str(kept): r.sha(kept)}, runs=runs, history=[],
            evaluation=evaluation, publication_counts=(2, 1), source_sha=r.sha(Path(n.__file__)))
        self.proof = dict(status='verified_private_backup', scp_exit_code=0,
            all_local_and_remote_sha256_verified=True, local_backups_read_only=True, remote_files_not_yet_removed=True,
            target_files=[dict(remote_path=str(target), local_path=self.s['private'], sha256=self.s['target_sha'],
                local_sha256=self.s['target_sha'], size_bytes=self.s['logical'], local_size_bytes=self.s['logical'],
                allocated_bytes_remote=self.s['allocated'], local_read_only=True, source_nlink=1, source_mode='0o444',
                source_identity_verified_after_copy=True, source_stat_decimal_strings=[str(x) for x in r.stamp(target)])])
        self.write_proof()

    def write_proof(self):
        self.s['proof'].write_text(json.dumps(self.proof)); self.s['proof_sha'] = r.sha(self.s['proof'])

    def set_publication(self, root, relative):
        self.raw.write_text(relative + '\n')
        self.manifest.write_text(json.dumps({'source_path': root + '/' + relative, 'relative_path': relative}) + '\n')
        self.plan.write_text(json.dumps({'source_root': root, 'file_list_path': str(self.raw), 'file_list_sha256': r.sha(self.raw),
            'files_manifest_path': str(self.manifest), 'files_manifest_sha256': r.sha(self.manifest)}))

    def call(self, apply=False):
        with mock.patch.object(n, 'gpu_guard', return_value={'compute_processes': 0}), \
             mock.patch.object(r, 'process_guard', return_value={'processes_checked': 1, 'references': 0}), \
             contextlib.redirect_stdout(io.StringIO()):
            return n.execute(self.s, apply)

    def failure(self):
        self.assertFalse((self.s['audit'] / 'completion.json').exists())
        return json.loads((self.s['audit'] / 'failure.json').read_text())

    def test_default_readonly_has_no_audit_and_no_arbitrary_cli_paths(self):
        before = n.evidence_snapshot(self.s)
        with mock.patch.object(n.os, 'unlink', side_effect=AssertionError('dry run cannot unlink')):
            result = self.call()
        self.assertEqual(result['status'], 'validated_not_deleted'); self.assertFalse(self.s['audit'].exists())
        self.assertEqual(n.evidence_snapshot(self.s), before)
        self.assertFalse(n.parse_args(['--source-sha256', self.s['source_sha']]).apply)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            n.parse_args(['--source-sha256', self.s['source_sha'], '--target', '/tmp/unrelated'])

    def test_single_tiny_file_removed_and_all_other_files_preserved(self):
        before = n.evidence_snapshot(self.s); result = self.call(True)
        self.assertEqual(result['removed_files'], [str(self.s['target'])]); self.assertEqual(result['removed_count'], 1)
        self.assertFalse(self.s['target'].exists()); self.assertTrue(self.s['target'].parent.is_dir())
        self.assertEqual(n.evidence_snapshot(self.s), before)
        rows = [json.loads(row) for row in (self.s['audit'] / 'removed_files.jsonl').read_text().splitlines()]
        self.assertEqual([row['phase'] for row in rows], ['prepared', 'removed'])

    def test_wrong_backup_sha_or_duplicate_scope_or_readonly_cannot_delete(self):
        for kind in ('sha', 'duplicate', 'readonly', 'private_path', 'decimal_stat', 'bool_size'):
            with self.subTest(kind=kind):
                value = json.loads(json.dumps(self.proof)); original_sha = self.s['proof_sha']
                if kind == 'duplicate': value['target_files'].append(value['target_files'][0])
                if kind == 'readonly': value['target_files'][0]['local_read_only'] = False
                if kind == 'private_path': value['target_files'][0]['local_path'] += '.partial'
                if kind == 'decimal_stat': value['target_files'][0]['source_stat_decimal_strings'][1] = 12
                if kind == 'bool_size': value['target_files'][0]['source_nlink'] = True
                self.s['proof'].write_text(json.dumps(value)); self.s['proof_sha'] = r.sha(self.s['proof'])
                if kind == 'sha': self.s['proof_sha'] = '0' * 64
                with self.assertRaises(RuntimeError): self.call()
                self.assertTrue(self.s['target'].exists()); self.assertFalse(self.s['audit'].exists())
                self.write_proof()

    def test_apply_preflight_failure_is_persistent_and_not_retried(self):
        self.s['proof_sha'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'proof SHA'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 0); self.assertTrue(self.s['target'].exists())
        self.write_proof()
        with self.assertRaisesRegex(RuntimeError, 'already exists'): self.call(True)

    def test_exact_target_path_hardlink_symlink_and_byte_change_rejected(self):
        original = self.s['target']; self.s['target'] = original.with_name('another.pt')
        with self.assertRaisesRegex(RuntimeError, 'escaped'): self.call()
        self.s['target'] = original
        link = original.with_name('alias.pt'); os.link(original, link)
        with self.assertRaisesRegex(RuntimeError, 'stat/mode/link'): self.call()
        link.unlink()
        self.proof['target_files'][0]['source_stat_decimal_strings'] = [str(x) for x in r.stamp(original)]; self.write_proof()
        renamed = original.parent.with_name('actual'); original.parent.rename(renamed); original.parent.symlink_to(renamed, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'symlink'): self.call()

    def test_source_or_protected_asset_mutation_blocks(self):
        original = self.s['source_sha']; self.s['source_sha'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'source SHA'): self.call()
        self.s['source_sha'] = original
        Path(next(iter(self.s['protected']))).write_bytes(b'changed3927')
        with self.assertRaisesRegex(RuntimeError, 'Protected asset'): self.call()

    def test_publication_relative_reference_and_missing_plan_scope_block(self):
        self.set_publication(str(self.s['target'].parent), self.s['target'].name)
        with self.assertRaisesRegex(RuntimeError, 'checkpoint reference'): self.call(True)
        self.assertTrue(self.s['target'].exists()); self.assertEqual(self.failure()['removed_count'], 0)

    def test_all_process_and_gpu_failures_block(self):
        for function in ('gpu_guard', 'process_guard'):
            with self.subTest(function=function), contextlib.redirect_stdout(io.StringIO()):
                target = n if function == 'gpu_guard' else r
                with mock.patch.object(n, 'gpu_guard', return_value={}), mock.patch.object(r, 'process_guard', return_value={}):
                    with mock.patch.object(target, function, side_effect=RuntimeError('active reference')):
                        with self.assertRaisesRegex(RuntimeError, 'active reference'): n.execute(self.s)
        self.assertTrue(self.s['target'].exists())

    def test_real_process_guard_blocks_same_inode_and_unreadable_live_process(self):
        proc = Path(self.temp.name) / 'proc/999999'; (proc / 'fd').mkdir(parents=True)
        (proc / 'cmdline').write_bytes(b'worker'); (proc / 'maps').write_bytes(b'')
        (proc / 'cwd').symlink_to(self.s['root']); (proc / 'exe').symlink_to('/bin/true')
        (proc / 'fd/4').symlink_to(self.s['target']); info = self.s['target'].stat()
        with self.assertRaisesRegex(RuntimeError, 'holds checkpoint inode'):
            r.process_guard([self.s['target']], {(info.st_dev, info.st_ino)}, proc.parent)
        with mock.patch.object(Path, 'read_bytes', side_effect=PermissionError('denied')), self.assertRaises(PermissionError):
            r.process_guard([self.s['target']], set(), proc.parent)

    def test_mutation_immediately_before_unlink_blocks(self):
        original = n.unlink_exact
        def altered(s, expected, journal, removed):
            s['target'].chmod(0o644); s['target'].write_bytes(b'changed'); s['target'].chmod(0o444)
            return original(s, expected, journal, removed)
        with mock.patch.object(n, 'unlink_exact', side_effect=altered), self.assertRaisesRegex(RuntimeError, 'changed'):
            self.call(True)
        self.assertEqual(self.failure()['removed_count'], 0); self.assertTrue(self.s['target'].exists())

    def test_failure_after_unlink_preserves_actual_cursor_and_no_retry(self):
        original = n.os.fsync
        def fsync(fd):
            if not self.s['target'].exists(): raise OSError('post unlink fsync failure')
            return original(fd)
        with mock.patch.object(n.os, 'fsync', side_effect=fsync), self.assertRaisesRegex(OSError, 'post unlink'):
            self.call(True)
        self.assertFalse(self.s['target'].exists()); self.assertEqual(self.failure()['removed_count'], 1)
        with self.assertRaisesRegex(RuntimeError, 'already exists'): self.call(True)

    def test_changed_frozen_metadata_after_unlink_rejects_success(self):
        original = n.unlink_exact
        def changed(*args): original(*args); self.meta.write_text('{"changed":true}')
        with mock.patch.object(n, 'unlink_exact', side_effect=changed), self.assertRaisesRegex(RuntimeError, 'Protected evidence'):
            self.call(True)
        self.assertEqual(self.failure()['removed_count'], 1)

    def test_queue_is_inventory_only_and_wavs_not_hashed(self):
        original = n.unlink_exact; old_sha = r.sha
        def changed(*args): original(*args); (self.s['dev'] / 'RUN_QUEUE.md').write_text('new schedule')
        def no_wav(path):
            if Path(path).suffix == '.wav': raise AssertionError('Do not read old WAV')
            return old_sha(path)
        with mock.patch.object(n, 'unlink_exact', side_effect=changed), mock.patch.object(r, 'sha', side_effect=no_wav):
            result = self.call(True)
        queue = str(self.s['dev'] / 'RUN_QUEUE.md')
        self.assertFalse(result['historical_reference_inventory'][queue]['protected_frozen_metadata'])

    def test_cleanup_failure_closes_all_four_and_never_publishes_success(self):
        original = Path.open; closed = []
        class Handle:
            def __init__(self, actual, name): self.actual, self.name = actual, name
            def fileno(self): return self.actual.fileno()
            def close(self):
                self.actual.close(); closed.append(self.name)
                if self.name == n.LOCK_NAMES[-1]: raise OSError('injected lock close')
        def opening(path, *args, **kwargs):
            handle = original(path, *args, **kwargs)
            return Handle(handle, path.name) if path.name in n.LOCK_NAMES and args and args[0] == 'r+' else handle
        with mock.patch.object(Path, 'open', opening), self.assertRaisesRegex(OSError, 'lock close'): self.call(True)
        self.assertEqual(set(closed), set(n.LOCK_NAMES)); self.assertEqual(self.failure()['removed_count'], 1)

    def test_completion_publication_failure_and_repetition_are_truthful(self):
        with mock.patch.object(r, 'write_completion', side_effect=OSError('publish failed')), self.assertRaisesRegex(OSError, 'publish'):
            self.call(True)
        self.assertEqual(self.failure()['removed_count'], 1)
        with self.assertRaisesRegex(RuntimeError, 'already exists'): self.call(True)

    def test_actual_receipt_schema_is_read_without_model_or_apply(self):
        spec = n.production_spec(r.sha(Path(n.__file__)))
        value, saved = n.validate_backup(spec)
        self.assertEqual(len(value['target_files']), 1); self.assertEqual(saved[2], 1424624517)
        self.assertEqual(value['target_files'][0]['source_mode'], '0o444')


if __name__ == '__main__': unittest.main(verbosity=2)
