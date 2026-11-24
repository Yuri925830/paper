"""仅在租用服务器 TemporaryDirectory 中进行CPU安全测试，不触碰真实候选。"""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import retire_mc2_storage_candidates_v1 as m


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mc2-retirement-toy-')
        self.root = Path(self.temp.name) / 'project'
        self.dev = self.root / 'exp/dev'
        self.dev.mkdir(parents=True)
        (self.root / 'exp/locks').mkdir()
        for name in m.LOCKS:
            (self.root / 'exp/locks' / name).write_text('')
        self.runs = tuple(self.dev / ('run%d' % index) for index in range(4))
        self.targets = []
        for index, run in enumerate(self.runs):
            run.mkdir()
            (run / 'metrics.csv').write_text('keep,all,rows\n')
            path = run / ('model.pt' if index < 2 else 'sample.wav')
            path.write_bytes(bytes([index + 1]) * (index + 5))
            self.targets.append(path)
        self.kept_model = self.dev / 'protected.pt'
        self.kept_model.write_bytes(b'protected teacher/current student')
        self.protected = {self.kept_model: m.g.sha(self.kept_model)}
        self.old = self.dev / 'historical_completion.json'
        self.old.write_text(json.dumps({'checkpoint': str(self.targets[0])}))
        self.v1 = self.dev / 'CANDIDATES_V1.json'
        self.v1.write_text(json.dumps({'old_rounded_metadata': True, 'targets': [str(p) for p in self.targets]}))
        self.manifest = self.dev / 'CANDIDATES_V2.json'
        self.proof = self.dev / 'backup_receipt.json'
        self.audit = self.dev / 'retirement'
        self.private = 'C:/private/mc2'
        self.pub = self.root / 'publication'
        lists = self.pub / 'direct-upload-runtime/lists'
        lists.mkdir(parents=True)
        self.raw, self.meta = lists / 'job.files-from-raw.txt', lists / 'job.metadata.jsonl'
        self.plan = self.pub / 'PLAN.json'
        self.publish(self.root / 'unrelated', 'safe.dat')
        rows = []
        for path in self.targets:
            st = path.stat()
            rows.append({'path': str(path), 'sha256': m.g.sha(path), 'stat': m.stamp(path),
                         'size_bytes': st.st_size, 'file_allocation_bytes': st.st_blocks * 512,
                         'type': 'regular_file', 'nlink': 1})
        self.document = {'schema_version': 2, 'stat_encoding': 'decimal_strings_all_five_fields',
                         'candidates': [{'target': row['path'], 'files': [row]} for row in rows],
                         'historical_metadata_references': [{'path': str(p), 'sha256': m.g.sha(p)} for p in (self.old, self.v1)],
                         'previous_manifest': {'path': str(self.v1), 'sha256': m.g.sha(self.v1)},
                         'related_preserved_receipts': []}
        self.save_manifest_and_proof()
        self.process_patch = mock.patch.object(m.g, 'process_guard', return_value={'processes_checked': 1, 'references': 0})
        self.process = self.process_patch.start()

    def tearDown(self):
        self.process_patch.stop()
        self.temp.cleanup()

    def publish(self, source, relative):
        self.raw.write_text(relative + '\n')
        self.meta.write_text(json.dumps({'relative_path': relative, 'source_path': str(source / relative)}) + '\n')
        self.plan.write_text(json.dumps({'jobs': [{'source_root': str(source), 'file_list_path': str(self.raw),
                                                 'files_manifest_path': str(self.meta), 'file_list_sha256': m.g.sha(self.raw),
                                                 'files_manifest_sha256': m.g.sha(self.meta)}]}))

    def save_manifest_and_proof(self):
        self.manifest.write_text(json.dumps(self.document))
        self.manifest_sha = m.g.sha(self.manifest)
        rows = [row for group in self.document['candidates'] for row in group['files']]
        self.backup = {'schema_version': 1, 'status': 'verified_private_backup', 'source_server_alias': 'matpool-rental',
                       'manifest_path': str(self.manifest), 'manifest_sha256': self.manifest_sha,
                       'verified_utc': '2026-09-07T00:00:00Z', 'all_local_and_remote_sha256_verified': True,
                       'local_backups_read_only': True, 'remote_files_not_yet_removed': True, 'scp_exit_codes': [0],
                       'target_files': [{'remote_path': row['path'], 'local_path': self.private + '/' + Path(row['path']).relative_to(self.dev).as_posix(),
                                         'size_bytes': row['size_bytes'], 'sha256': row['sha256'], 'local_sha256': row['sha256']} for row in rows]}
        self.save_proof()

    def save_proof(self):
        self.proof.write_text(json.dumps(self.backup))
        self.proof_sha = m.g.sha(self.proof)

    def execute(self, apply=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return m.execute(self.root, self.dev, self.pub, self.manifest, self.manifest_sha, self.proof,
                             self.proof_sha, self.audit, self.private, self.runs, self.protected, 4,
                             m.g.sha(Path(m.g.__file__).resolve()), apply)

    def locks_free(self):
        for name in m.LOCKS:
            with (self.root / 'exp/locks' / name).open('r+') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_dryrun_no_write(self):
        value = self.execute()
        self.assertEqual(value['status'], 'validated_not_deleted')
        self.assertFalse(self.audit.exists())
        self.assertTrue(all(p.exists() for p in self.targets))
        self.assertIn(str(self.v1), value['historical_reference_hits'])
        self.locks_free()

    def test_apply_exact_files_keep_directories_and_metadata(self):
        before = {p: p.read_bytes() for p in (self.old, self.v1, self.kept_model)}
        value = self.execute(True)
        self.assertEqual(value['removed_count'], 4)
        self.assertTrue(all(not p.exists() for p in self.targets))
        self.assertTrue(all(run.is_dir() and (run / 'metrics.csv').read_text() == 'keep,all,rows\n' for run in self.runs))
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertTrue((self.audit / 'completion.json').exists())
        journal = [json.loads(line) for line in (self.audit / 'removed_files.jsonl').read_text().splitlines()]
        self.assertEqual([r['phase'] for r in journal], ['prepared', 'removed'] * 4)
        self.locks_free()

    def test_repeated_apply_refused(self):
        self.execute(True)
        with self.assertRaisesRegex(RuntimeError, 'Audit exists'):
            self.execute(True)

    def test_arbitrary_cli_path_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            m.parse_args(['--backup-receipt-sha256', 'a' * 64, '--target', '/tmp'])

    def test_numeric_nanosecond_manifest_refused(self):
        self.document['candidates'][0]['files'][0]['stat'] = [int(v) for v in self.document['candidates'][0]['files'][0]['stat']]
        self.save_manifest_and_proof()
        with self.assertRaisesRegex(RuntimeError, 'Unsafe numeric stat'):
            self.execute(True)

    def test_exact_nanoseconds_survive_json_roundtrip(self):
        row = json.loads(self.manifest.read_text())['candidates'][0]['files'][0]
        self.assertTrue(all(isinstance(v, str) for v in row['stat']))
        self.assertGreater(int(row['stat'][3]), 2 ** 53)
        self.assertEqual(row['stat'], m.stamp(self.targets[0]))

    def test_wrong_local_backup_hash_refused(self):
        self.backup['target_files'][0]['local_sha256'] = '0' * 64
        self.save_proof()
        with self.assertRaisesRegex(RuntimeError, 'Backup path/size/hash'):
            self.execute(True)

    def test_wrong_backup_layout_refused(self):
        self.backup['target_files'][0]['local_path'] = self.private + '/../escape.pt'
        self.save_proof()
        with self.assertRaisesRegex(RuntimeError, 'Backup path/size/hash'):
            self.execute(True)

    def test_missing_backup_row_refused(self):
        self.backup['target_files'].pop()
        self.save_proof()
        with self.assertRaisesRegex(RuntimeError, 'Backup target set'):
            self.execute(True)

    def test_unverified_copy_exit_refused(self):
        self.backup['scp_exit_codes'] = [0, 1]
        self.save_proof()
        with self.assertRaisesRegex(RuntimeError, 'transfer did not complete'):
            self.execute(True)

    def test_source_symlink_refused(self):
        self.targets[0].unlink()
        self.targets[0].symlink_to(self.kept_model)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.execute(True)

    def test_source_hardlink_refused(self):
        os.link(self.targets[0], self.dev / 'extra.pt')
        with self.assertRaisesRegex(RuntimeError, 'source identity changed'):
            self.execute(True)

    def test_source_modified_refused(self):
        self.targets[0].write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'source identity changed'):
            self.execute(True)

    def test_protected_overlap_refused(self):
        self.protected[self.targets[0]] = m.g.sha(self.targets[0])
        with self.assertRaisesRegex(RuntimeError, 'protected model'):
            self.execute(True)

    def test_protected_model_change_refused(self):
        self.kept_model.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Protected file changed'):
            self.execute(True)

    def test_publication_relative_source_reference_refused(self):
        self.publish(self.targets[0].parent, self.targets[0].name)
        with self.assertRaisesRegex(RuntimeError, 'External checkpoint reference'):
            self.execute(True)

    def test_unknown_audit_reference_refused(self):
        (self.dev / 'unknown_audit.json').write_text(json.dumps({'input': str(self.targets[1])}))
        with self.assertRaisesRegex(RuntimeError, 'Unknown or changed external'):
            self.execute(True)

    def test_v1_reference_missing_pin_refused(self):
        self.document['historical_metadata_references'] = [r for r in self.document['historical_metadata_references'] if r['path'] != str(self.v1)]
        self.save_manifest_and_proof()
        with self.assertRaisesRegex(RuntimeError, 'V1 precision record'):
            self.execute(True)

    def test_fourth_inference_lock_busy_refused(self):
        with (self.root / 'exp/locks' / m.LOCKS[3]).open('r+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute(True)
        self.locks_free()

    def test_process_permission_error_stops_without_delete(self):
        self.process.side_effect = PermissionError('/proc/toy/maps denied')
        with self.assertRaises(PermissionError):
            self.execute(True)
        self.assertTrue(all(p.exists() for p in self.targets))

    def test_second_unlink_failure_reports_exact_cursor(self):
        unlink, calls = os.unlink, [0]
        def fail_second(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError('second unlink failed')
            return unlink(*args, **kwargs)
        with mock.patch.object(m.os, 'unlink', side_effect=fail_second), self.assertRaisesRegex(OSError, 'second unlink'):
            self.execute(True)
        value = json.loads((self.audit / 'failure.json').read_text())
        self.assertEqual(value['removed_count'], 1)
        self.assertFalse((self.audit / 'completion.json').exists())
        self.locks_free()

    def test_post_unlink_metadata_mutation_stops_with_true_cursor(self):
        unlink = os.unlink
        def mutate(*args, **kwargs):
            result = unlink(*args, **kwargs)
            (self.runs[0] / 'unexpected_metadata.txt').write_text('must preserve')
            return result
        with mock.patch.object(m.os, 'unlink', side_effect=mutate), self.assertRaisesRegex(RuntimeError, 'Preserved run metadata'):
            self.execute(True)
        value = json.loads((self.audit / 'failure.json').read_text())
        self.assertEqual(value['removed_count'], 1)
        self.assertEqual((self.runs[0] / 'unexpected_metadata.txt').read_text(), 'must preserve')

    def test_post_unlink_fsync_failure_reports_true_cursor(self):
        unlink, fsync, deleted = os.unlink, os.fsync, [False]
        def mark(*args, **kwargs):
            result = unlink(*args, **kwargs); deleted[0] = True; return result
        def fail(fd):
            if deleted[0]:
                deleted[0] = False
                raise OSError('post-unlink fsync failed')
            return fsync(fd)
        with mock.patch.object(m.os, 'unlink', side_effect=mark), mock.patch.object(m.os, 'fsync', side_effect=fail), self.assertRaises(OSError):
            self.execute(True)
        self.assertEqual(json.loads((self.audit / 'failure.json').read_text())['removed_count'], 1)

    def test_completion_write_failure_leaves_no_success(self):
        write = m.g.write_new
        def fail(path, value):
            if path.name == 'completion.prepared.json':
                raise OSError('completion write failed')
            return write(path, value)
        with mock.patch.object(m.g, 'write_new', side_effect=fail), self.assertRaises(OSError):
            self.execute(True)
        self.assertFalse((self.audit / 'completion.json').exists())
        self.assertEqual(json.loads((self.audit / 'failure.json').read_text())['removed_count'], 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
