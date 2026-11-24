"""只在租用机临时目录创建几字节假权重，绝不删除真实实验文件。"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import retire_failed3347_checkpoints_v1 as r


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='retire3347_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        self.run = self.root / 'exp/diagnostics/failed_trial'
        self.primary = self.run / 'primary_stratified'
        self.primary.mkdir(parents=True)
        self.pub = Path(self.temp.name) / 'publication'
        self.lists = self.pub / 'direct-upload-runtime/lists'
        self.lists.mkdir(parents=True)
        self.proof = self.root / 'exp/backup_receipt.json'
        self.audit = self.root / 'exp/retirement_audit'
        self.private = 'C:/private/'
        self.specs = []
        for name, content in [('final_full.pt', b'full-adam-3347'), ('student_step_003347.pt', b'student-3347')]:
            path = self.primary / name
            path.write_bytes(content)
            self.specs.append((path, r.sha(path), len(content)))
        locks = self.root / 'exp/locks'
        locks.mkdir()
        for name in r.LOCK_NAMES:
            (locks / name).write_text('')
        self.meta = self.run / 'completion.json'
        self.meta.write_text(json.dumps({'status': 'old_failure', 'final_full': str(self.specs[0][0])}))
        (self.run / 'run_contract.json').write_text('{"old_config": true}')
        (self.primary / 'monitor.jsonl').write_text('{"step":3347,"loss":6.0}\n')
        self.proof_value = {'status': 'verified_private_backup', 'all_local_and_remote_sha256_verified': True,
                            'target_files': [{'remote_path': str(path), 'sha256': digest, 'local_sha256': digest,
                                              'size_bytes': size, 'local_path': self.private + path.name}
                                             for path, digest, size in self.specs]}
        self.write_proof()
        self.raw = self.lists / 'group.files-from-raw.txt'
        self.manifest = self.lists / 'group.metadata.jsonl'
        self.plan = self.pub / 'DIRECT_FILE_UPLOAD_PLAN.json'
        self.set_listing('/unrelated', 'keep.txt')

    def write_proof(self):
        self.proof.write_text(json.dumps(self.proof_value))
        self.proof_sha = r.sha(self.proof)

    def set_listing(self, root, relative):
        self.raw.write_text(relative + '\n')
        self.manifest.write_text(json.dumps({'source_path': root + '/' + relative, 'relative_path': relative}) + '\n')
        self.plan.write_text(json.dumps({'groups': [{'source_root': root, 'file_list_path': str(self.raw),
                                                    'file_list_sha256': r.sha(self.raw),
                                                    'files_manifest_path': str(self.manifest),
                                                    'files_manifest_sha256': r.sha(self.manifest)}]}))

    def execute(self, apply=False):
        with mock.patch.object(r, 'process_guard', return_value={'processes_checked': 0, 'references': 0}), contextlib.redirect_stdout(io.StringIO()):
            return r.execute(self.root, self.run, self.pub, self.proof, self.audit,
                             self.specs, self.private, self.proof_sha, apply)

    def assert_targets_exist(self):
        for path, digest, _ in self.specs:
            self.assertEqual(r.sha(path), digest)

    def test_dry_run_is_readonly_and_counts_all_lists(self):
        before = r.metadata_snapshot(self.run)
        result = self.execute()
        self.assertEqual(result['status'], 'validated_not_deleted')
        self.assertEqual(result['dependency_guard_before']['publication']['documents'], 2)
        self.assertEqual(result['dependency_guard_before']['publication']['lines'], 2)
        self.assertEqual(result['dependency_guard_before']['contracts']['allowed_historical_self_references'], [str(self.meta)])
        self.assertEqual(before, r.metadata_snapshot(self.run))
        self.assertFalse(self.audit.exists())
        self.assert_targets_exist()

    def test_apply_only_removes_two_tiny_fixture_files(self):
        before = r.metadata_snapshot(self.run)
        kept = self.primary / 'other_step.pt'
        kept.write_bytes(b'never selected')
        result = self.execute(True)
        self.assertEqual(result['removed_count'], 2)
        self.assertEqual(before, r.metadata_snapshot(self.run))
        self.assertEqual(kept.read_bytes(), b'never selected')
        self.assertTrue(self.primary.is_dir())
        self.assertTrue((self.audit / 'completion.json').exists())
        journal = [json.loads(line) for line in (self.audit / 'removed_files.jsonl').read_text().splitlines()]
        self.assertEqual([row['phase'] for row in journal], ['prepared', 'removed', 'prepared', 'removed'])

    def test_missing_or_false_backup_cannot_apply(self):
        self.proof_value['all_local_and_remote_sha256_verified'] = False
        self.write_proof()
        with self.assertRaisesRegex(RuntimeError, 'not fully verified'):
            self.execute(True)
        self.assert_targets_exist()
        self.assertFalse(self.audit.exists())
        self.proof.unlink()
        with self.assertRaises(FileNotFoundError):
            self.execute(True)

    def test_backup_partial_name_or_mismatched_sha_rejected(self):
        self.proof_value['target_files'][0]['local_path'] += '.partial'
        self.write_proof()
        with self.assertRaisesRegex(RuntimeError, 'filename mismatch'):
            self.execute(True)
        self.assert_targets_exist()
        self.proof_sha = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'receipt SHA'):
            self.execute(True)

    def test_duplicate_proof_target_rejected(self):
        self.proof_value['target_files'][1] = self.proof_value['target_files'][0]
        self.write_proof()
        with self.assertRaisesRegex(RuntimeError, 'distinct'):
            self.execute(True)

    def test_hardlink_and_symlink_parent_refused(self):
        path = self.specs[0][0]
        os.link(path, self.root / 'second_link')
        with self.assertRaisesRegex(RuntimeError, 'link count'):
            self.execute(True)
        (self.root / 'second_link').unlink()
        renamed = self.run / 'real_primary'
        self.primary.rename(renamed)
        self.primary.symlink_to(renamed, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.execute(True)

    def test_target_byte_mutation_rejected(self):
        self.specs[0][0].write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'bytes/link'):
            self.execute(True)
        self.assertFalse(self.audit.exists())

    def test_external_contract_file_sha_key_blocks(self):
        external = self.root / 'exp/current_run_contract.json'
        external.write_text(json.dumps({'file_sha256': {str(self.specs[1][0]): self.specs[1][1]}}))
        with self.assertRaisesRegex(RuntimeError, 'External experiment dependency'):
            self.execute(True)
        self.assert_targets_exist()

    def test_publication_source_root_relative_file_blocks(self):
        # 绝对路径没有出现在raw行中，也必须通过source_root拼接识别依赖。
        target = self.specs[0][0]
        self.set_listing(str(target.parent), target.name)
        with self.assertRaisesRegex(RuntimeError, 'External checkpoint reference'):
            self.execute(True)
        self.assert_targets_exist()

    def test_every_publication_list_requires_mapping(self):
        (self.lists / 'extra.files-from-raw.txt').write_text('not-in-plan.pt\n')
        with self.assertRaisesRegex(RuntimeError, 'Unmapped'):
            self.execute(True)

    def test_publication_declared_hash_and_escape_rejected(self):
        self.raw.write_text('new-unattested-file.pt\n')
        with self.assertRaisesRegex(RuntimeError, 'Unpinned/changed'):
            self.execute(True)
        self.set_listing('/unrelated', '../escape.pt')
        with self.assertRaisesRegex(RuntimeError, 'Escaping'):
            self.execute(True)

    def test_additional_nested_plan_is_checked(self):
        nested = self.pub / 'direct-upload-runtime/new_plans'
        nested.mkdir()
        (nested / 'NEW_PLAN.json').write_text(json.dumps({'source_path': str(self.specs[0][0])}))
        with self.assertRaisesRegex(RuntimeError, 'External checkpoint reference'):
            self.execute(True)

    def test_proc_open_inode_and_permission_error_block(self):
        procs = Path(self.temp.name) / 'proc'
        proc = procs / '999999'
        (proc / 'fd').mkdir(parents=True)
        (proc / 'cmdline').write_bytes(b'worker\x00')
        (proc / 'maps').write_bytes(b'')
        (proc / 'cwd').symlink_to(self.root)
        (proc / 'exe').symlink_to('/bin/true')
        target = self.specs[0][0]
        (proc / 'fd/3').symlink_to(target)
        info = target.stat()
        with self.assertRaisesRegex(RuntimeError, 'holds checkpoint inode'):
            r.process_guard([target], {(info.st_dev, info.st_ino)}, procs)
        with mock.patch.object(Path, 'read_bytes', side_effect=PermissionError('unreadable')):
            with self.assertRaises(PermissionError):
                r.process_guard([target], set(), procs)

    def test_metadata_fault_after_first_unlink_is_truthful(self):
        original = r.os.unlink
        def alter_metadata(name, *args, **kwargs):
            result = original(name, *args, **kwargs)
            if kwargs.get('dir_fd') is not None:
                self.meta.write_text('{"changed_by_other_actor": true}')
            return result
        with mock.patch.object(r.os, 'unlink', side_effect=alter_metadata):
            with self.assertRaisesRegex(RuntimeError, 'Protected evidence changed'):
                self.execute(True)
        self.assertFalse(self.specs[0][0].exists())
        self.assertTrue(self.specs[1][0].exists())
        failure = json.loads((self.audit / 'failure.json').read_text())
        self.assertEqual(failure['removed_files'], [str(self.specs[0][0])])
        self.assertFalse((self.audit / 'completion.json').exists())

    def test_post_unlink_fsync_failure_records_actual_removed_file(self):
        original_unlink = r.os.unlink
        original_fsync = r.os.fsync
        state = {'unlinked': False}
        def unlink(name, *args, **kwargs):
            result = original_unlink(name, *args, **kwargs)
            if kwargs.get('dir_fd') is not None:
                state['unlinked'] = True
            return result
        def fsync(fd):
            if state['unlinked']:
                state['unlinked'] = False
                raise OSError('injected post-unlink fsync failure')
            return original_fsync(fd)
        with mock.patch.object(r.os, 'unlink', side_effect=unlink), mock.patch.object(r.os, 'fsync', side_effect=fsync):
            with self.assertRaisesRegex(OSError, 'post-unlink'):
                self.execute(True)
        self.assertEqual(json.loads((self.audit / 'failure.json').read_text())['removed_count'], 1)
        self.assertTrue(self.specs[1][0].exists())

    def test_existing_audit_directory_never_overwritten(self):
        self.audit.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            self.execute(True)
        self.assert_targets_exist()

    def test_successful_apply_cannot_be_repeated(self):
        self.execute(True)
        original = (self.audit / 'completion.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            self.execute(True)
        self.assertEqual((self.audit / 'completion.json').read_bytes(), original)

    def test_second_unlink_failure_retains_second_checkpoint(self):
        original = r.os.unlink
        count = [0]
        def unlink(name, *args, **kwargs):
            if kwargs.get('dir_fd') is not None:
                count[0] += 1
                if count[0] == 2:
                    raise OSError('second unlink failed')
            return original(name, *args, **kwargs)
        with mock.patch.object(r.os, 'unlink', side_effect=unlink):
            with self.assertRaisesRegex(OSError, 'second unlink'):
                self.execute(True)
        self.assertFalse(self.specs[0][0].exists())
        self.assertTrue(self.specs[1][0].exists())
        self.assertEqual(json.loads((self.audit / 'failure.json').read_text())['removed_count'], 1)
        self.assertFalse((self.audit / 'completion.json').exists())

    def test_lock_close_failure_releases_every_lock_and_no_complete(self):
        original_open = Path.open
        closed = []
        class CloseFailure:
            def __init__(self, handle, name):
                self.handle, self.name = handle, name
            def fileno(self):
                return self.handle.fileno()
            def close(self):
                self.handle.close()
                closed.append(self.name)
                if self.name == r.LOCK_NAMES[-1]:
                    raise OSError('close failed')
        def open_path(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            if path.name in r.LOCK_NAMES and args and args[0] == 'r+':
                return CloseFailure(handle, path.name)
            return handle
        with mock.patch.object(Path, 'open', open_path):
            with self.assertRaisesRegex(OSError, 'close failed'):
                self.execute(True)
        self.assertEqual(set(closed), set(r.LOCK_NAMES))
        self.assertFalse((self.audit / 'completion.json').exists())
        self.assertEqual(json.loads((self.audit / 'failure.json').read_text())['removed_count'], 2)

    def test_completion_preparation_and_post_publish_faults_never_look_successful(self):
        # 只测试新审计目录的完成发布，不触及真实权重或实验目录。
        for after_publish in (False, True):
            audit = Path(self.temp.name) / ('commit_' + str(after_publish))
            audit.mkdir()
            original = r.os.fsync
            def fsync(fd):
                if (audit / 'completion.json').exists() == after_publish:
                    raise OSError('completion durability failure')
                return original(fd)
            with mock.patch.object(r.os, 'fsync', side_effect=fsync):
                with self.assertRaisesRegex(OSError, 'durability'):
                    r.write_completion(audit, {'status': 'complete'})
            self.assertFalse((audit / 'completion.json').exists())
            expected = 'completion.invalidated.json' if after_publish else 'completion.prepared.json'
            self.assertTrue((audit / expected).exists())

    def test_actual_private_receipt_schema_only_no_model_workload(self):
        # 真收据正例只读取几KB JSON；不加载模型、不运行真实清理入口。
        if not r.PROOF.exists():
            self.skipTest('Only rental has the verified private receipt')
        value = r.validate_backup(r.PROOF, '9bbd6bfac7a48269d8d269cbfbe7fdbd4c2ca398d23e25c100df013b25784061', r.SPECS, r.PRIVATE)
        self.assertEqual(len(value['target_files']), 2)

    def orphan_fixture(self, current_members=None, wrong_current_root=False, wrong_old_source=False):
        # 临时小清单沿用生产精确basename/root/两条路径，仅把字节SHA替换为夹具真实SHA。
        spec = dict(r.AUDITED_ORPHAN)
        relative = list(spec['relative_paths'])
        root = spec['source_root']
        old_raw = self.lists / spec['raw_basename']
        old_meta = self.lists / spec['metadata_basename']
        new_raw = self.lists / spec['current_raw_basename']
        new_meta = self.lists / spec['current_metadata_basename']
        old_raw.write_text(''.join(name + '\n' for name in relative))
        old_meta.write_text(''.join(json.dumps({'relative_path': name,
                            'source_path': ('/wrong' if wrong_old_source else root) + '/' + name}) + '\n' for name in relative))
        current = relative if current_members is None else current_members
        new_raw.write_text(''.join(name + '\n' for name in current))
        new_meta.write_text(''.join(json.dumps({'relative_path': name, 'source_path': root + '/' + name}) + '\n' for name in current))
        for key, path in [('raw_sha256', old_raw), ('metadata_sha256', old_meta),
                          ('current_raw_sha256', new_raw), ('current_metadata_sha256', new_meta)]:
            spec[key] = r.sha(path)
        plan = json.loads(self.plan.read_text())
        plan['groups'].append({'source_root': '/wrong' if wrong_current_root else root,
                               'file_list_path': str(new_raw), 'file_list_sha256': spec['current_raw_sha256'],
                               'files_manifest_path': str(new_meta), 'files_manifest_sha256': spec['current_metadata_sha256']})
        self.plan.write_text(json.dumps(plan))
        return spec, old_raw, old_meta, new_raw, new_meta

    def test_exact_audited_orphan_is_scanned_and_not_claimed_as_old_plan(self):
        spec, old_raw, old_meta, _, _ = self.orphan_fixture()
        with mock.patch.object(r, 'AUDITED_ORPHAN', spec):
            result = self.execute()
        publication = result['dependency_guard_before']['publication']
        self.assertEqual(publication['documents'], 6)
        self.assertIn(str(old_raw), publication['snapshot'])
        self.assertIn(str(old_meta), publication['file_sha256'])
        supplement = publication['audited_orphan_supplemental_mappings']
        self.assertEqual(len(supplement), 1)
        self.assertIs(supplement[0]['old_plan_hash_claimed'], False)
        self.assertEqual(supplement[0]['relative_paths'], list(r.AUDITED_ORPHAN['relative_paths']))
        self.assert_targets_exist()
        self.assertFalse(self.audit.exists())

    def test_audited_orphan_changed_sha_blocks(self):
        spec, old_raw, _, _, _ = self.orphan_fixture()
        old_raw.write_text(old_raw.read_text() + 'extra.py\n')
        with mock.patch.object(r, 'AUDITED_ORPHAN', spec), self.assertRaisesRegex(RuntimeError, 'SHA changed'):
            self.execute(True)
        self.assert_targets_exist()

    def test_audited_orphan_wrong_current_root_blocks(self):
        spec, _, _, _, _ = self.orphan_fixture(wrong_current_root=True)
        with mock.patch.object(r, 'AUDITED_ORPHAN', spec), self.assertRaisesRegex(RuntimeError, 'source_root status changed'):
            self.execute(True)

    def test_audited_orphan_wrong_old_source_blocks_after_valid_sha(self):
        spec, _, _, _, _ = self.orphan_fixture(wrong_old_source=True)
        with mock.patch.object(r, 'AUDITED_ORPHAN', spec), self.assertRaisesRegex(RuntimeError, 'row/source_root mapping'):
            self.execute(True)

    def test_audited_orphan_requires_current_coverage_of_both_members(self):
        spec, _, _, _, _ = self.orphan_fixture(current_members=[r.AUDITED_ORPHAN['relative_paths'][0]])
        with mock.patch.object(r, 'AUDITED_ORPHAN', spec), self.assertRaisesRegex(RuntimeError, 'does not cover both'):
            self.execute(True)

    def test_audited_orphan_missing_current_document_blocks(self):
        spec, _, _, _, new_meta = self.orphan_fixture()
        new_meta.unlink()
        with mock.patch.object(r, 'AUDITED_ORPHAN', spec), self.assertRaisesRegex(RuntimeError, 'Unverifiable|document missing'):
            self.execute(True)

    def test_audited_orphan_does_not_allow_any_other_unmapped_list(self):
        spec, _, _, _, _ = self.orphan_fixture()
        (self.lists / 'p2-unreviewed.files-from-raw.txt').write_text('file.py\n')
        with mock.patch.object(r, 'AUDITED_ORPHAN', spec), self.assertRaisesRegex(RuntimeError, 'Unmapped'):
            self.execute(True)

    def test_cli_default_readonly_has_no_arbitrary_target_argument(self):
        args = r.parse_args(['--backup-receipt-sha256', 'a' * 64])
        self.assertFalse(args.apply)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            r.parse_args(['--backup-receipt-sha256', 'a' * 64, '--target', '/tmp/other'])


if __name__ == '__main__':
    unittest.main()
