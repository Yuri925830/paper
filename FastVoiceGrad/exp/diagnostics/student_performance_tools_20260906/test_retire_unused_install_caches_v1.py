"""租用服务器 CPU toy 测试；全部删除只发生在 TemporaryDirectory 夹具内。"""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import retire_unused_install_caches_v1 as mod


class CacheRetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cache-retirement-toy-')
        self.base = Path(self.temp.name)
        self.root = self.base / 'project'
        locks = self.root / 'exp/locks'
        locks.mkdir(parents=True)
        for name in mod.LOCK_NAMES:
            (locks / name).write_text('')
        self.tree = self.base / 'cache/old-release'
        (self.tree / 'nested').mkdir(parents=True)
        (self.tree / 'a.bin').write_bytes(b'abc')
        (self.tree / 'nested/b.bin').write_bytes(b'defg')
        self.single = self.base / 'installers/package.whl'
        self.single.parent.mkdir()
        self.single.write_bytes(b'wheel')
        self.keep = self.base / 'cache/lru.json'
        self.keep.write_text('["old-release", "current-release"]')
        self.current = self.base / 'cache/current-release/package.json'
        self.current.parent.mkdir()
        self.current.write_text('{"version":"current"}')
        self.audit = self.root / 'exp/diagnostics/audit'
        self.audit.parent.mkdir()
        self.environments = self.base / 'envs'
        self.pub = self.base / 'publication'
        self.lists = self.pub / 'direct-upload-runtime/lists'
        self.lists.mkdir(parents=True)
        self.raw = self.lists / 'job.files-from-raw.txt'
        self.meta = self.lists / 'job.metadata.jsonl'
        self.set_publication(self.base / 'unrelated', 'safe.bin')
        self.process_patch = mock.patch.object(mod.guard, 'process_guard', return_value={'processes_checked': 1, 'references': 0})
        self.process_mock = self.process_patch.start()

    def tearDown(self):
        self.process_patch.stop()
        self.temp.cleanup()

    def set_publication(self, source_root, relative):
        self.raw.write_text(relative + '\n')
        self.meta.write_text(json.dumps({'source_path': str(source_root / relative), 'relative_path': relative}) + '\n')
        plan = {'jobs': [{'source_root': str(source_root), 'file_list_path': str(self.raw),
                         'files_manifest_path': str(self.meta), 'file_list_sha256': mod.digest(self.raw),
                         'files_manifest_sha256': mod.digest(self.meta)}]}
        (self.pub / 'PLAN.json').write_text(json.dumps(plan))

    def run_case(self, apply=False, guard_sha=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return mod.execute(self.root, self.pub, self.audit, self.tree, ((self.single, 5),),
                               (self.keep, self.current), 2, 7,
                               guard_sha or mod.digest(Path(mod.guard.__file__).resolve()), apply, (self.environments,))

    def assert_locks_free(self):
        for name in mod.LOCK_NAMES:
            with (self.root / 'exp/locks' / name).open('r+') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_default_dryrun_does_not_write(self):
        result = self.run_case()
        self.assertEqual(result['status'], 'validated_not_deleted')
        self.assertEqual(result['file_count'], 3)
        self.assertEqual(result['logical_bytes'], 12)
        self.assertFalse(self.audit.exists())
        self.assertTrue(self.single.exists())
        self.assertTrue((self.tree / 'nested/b.bin').exists())
        self.assert_locks_free()

    def test_apply_exact_scope_and_true_journal(self):
        keep = self.keep.read_bytes(), self.current.read_bytes()
        result = self.run_case(True)
        self.assertEqual(result['removed_count'], 3)
        self.assertFalse(self.tree.exists())
        self.assertFalse(self.single.exists())
        self.assertTrue(self.single.parent.is_dir())
        self.assertEqual((self.keep.read_bytes(), self.current.read_bytes()), keep)
        self.assertTrue((self.audit / 'completion.json').exists())
        records = [json.loads(line) for line in (self.audit / 'removed.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['phase'] == 'removed' for row in records), 3)
        self.assertEqual(sum(row['phase'] == 'directory_removed' for row in records), 2)
        self.assert_locks_free()

    def test_repeated_apply_refused(self):
        self.run_case(True)
        with self.assertRaisesRegex(RuntimeError, 'Audit already exists'):
            self.run_case(True)

    def test_no_arbitrary_cli_target(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mod.parse_args(['--guard-sha256', 'a' * 64, '--target', str(self.tree)])

    def test_wrong_guard_sha_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
            self.run_case(True, '0' * 64)
        self.assertTrue(self.single.exists())
        self.assertFalse(self.audit.exists())

    def test_file_symlink_refused(self):
        path = self.tree / 'a.bin'
        path.unlink()
        path.symlink_to(self.single)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.run_case(True)
        self.assertTrue(self.single.exists())

    def test_directory_symlink_refused(self):
        (self.tree / 'elsewhere').symlink_to(self.single.parent, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.run_case(True)

    def test_parent_symlink_refused(self):
        alias = self.base / 'alias'
        alias.symlink_to(self.tree.parent, target_is_directory=True)
        self.tree = alias / self.tree.name
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.run_case(True)

    def test_multiple_hardlinks_refused(self):
        os.link(self.tree / 'a.bin', self.base / 'protected-hardlink')
        with self.assertRaisesRegex(RuntimeError, 'Multiple hard links'):
            self.run_case(True)

    def test_wrong_tree_count_refused(self):
        (self.tree / 'unexpected').write_text('')
        with self.assertRaisesRegex(RuntimeError, 'count or size changed'):
            self.run_case(True)

    def test_wrong_single_size_refused(self):
        self.single.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'installer size changed'):
            self.run_case(True)

    def test_publication_relative_mapping_refused(self):
        self.set_publication(self.tree.parent, 'old-release/a.bin')
        with self.assertRaisesRegex(RuntimeError, 'External checkpoint reference'):
            self.run_case(True)
        self.assertFalse(self.audit.exists())

    def test_unknown_publication_list_refused(self):
        (self.lists / 'unmapped.files-from-raw.txt').write_text('unknown\n')
        with self.assertRaisesRegex(RuntimeError, 'Unmapped'):
            self.run_case(True)

    def test_experiment_contract_reference_refused(self):
        (self.root / 'exp/run_contract.json').write_text(json.dumps({'cache_path': str(self.single)}))
        with self.assertRaisesRegex(RuntimeError, 'code/config/environment reference'):
            self.run_case(True)
        self.assertTrue(self.single.exists())

    def test_escaped_json_reference_refused(self):
        value = json.dumps({'cache_path': str(self.single)}).replace('/', '\\/')
        (self.root / 'exp/run_config.json').write_text(value)
        with self.assertRaisesRegex(RuntimeError, 'code/config/environment reference'):
            self.run_case(True)

    def test_environment_direct_url_refused(self):
        path = self.environments / 'active/lib/python3.8/site-packages/package-1.0.dist-info/direct_url.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'url': 'file://' + str(self.single)}))
        with self.assertRaisesRegex(RuntimeError, 'code/config/environment reference'):
            self.run_case(True)

    def test_normal_terminfo_symlink_is_outside_metadata_scope(self):
        library = self.environments / 'active/lib'
        site = library / 'python3.8/site-packages'
        site.mkdir(parents=True)
        (library / 'terminfo').symlink_to('/usr/share/terminfo', target_is_directory=True)
        result = self.run_case()
        self.assertIn(str(site), result['current_metadata_before']['site_package_roots'])
        self.assertTrue((library / 'terminfo').is_symlink())

    def test_relevant_environment_metadata_symlink_refused(self):
        site = self.environments / 'active/lib/python3.8/site-packages'
        site.mkdir(parents=True)
        (site / 'editable.pth').symlink_to(self.keep)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.run_case(True)

    def test_direct_url_parent_symlink_refused(self):
        site = self.environments / 'active/lib/python3.8/site-packages'
        site.mkdir(parents=True)
        distribution = self.base / 'real-distribution'
        distribution.mkdir()
        (distribution / 'direct_url.json').write_text('{}')
        (site / 'package-1.0.dist-info').symlink_to(distribution, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.run_case(True)

    def test_site_packages_root_symlink_refused(self):
        site = self.environments / 'active/lib/python3.8/site-packages'
        site.parent.mkdir(parents=True)
        site.symlink_to(self.single.parent, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.run_case(True)

    def test_new_empty_environment_root_detected(self):
        site = self.environments / 'new-env/lib/python3.9/site-packages'
        def mutate(*args):
            site.mkdir(parents=True, exist_ok=True)
            return {'processes_checked': 1, 'references': 0}
        self.process_mock.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, 'metadata .*changed'):
            self.run_case(True)
        self.assertTrue(self.single.exists())

    def test_disappeared_empty_environment_root_detected(self):
        site = self.environments / 'active/lib/python3.8/site-packages'
        site.mkdir(parents=True)
        def mutate(*args):
            site.rmdir()
            return {'processes_checked': 1, 'references': 0}
        self.process_mock.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, 'metadata .*changed'):
            self.run_case(True)

    def test_top_level_egg_link_reference_refused(self):
        site = self.environments / 'active/lib/python3.8/site-packages'
        site.mkdir(parents=True)
        (site / 'package.egg-link').write_text(str(self.tree))
        with self.assertRaisesRegex(RuntimeError, 'code/config/environment reference'):
            self.run_case(True)

    def test_metadata_directory_symlink_refused(self):
        (self.root / 'exp/hidden-contracts').symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.run_case(True)

    def test_current_source_config_reference_refused(self):
        path = self.root / 'configs/use.yaml'
        path.parent.mkdir()
        path.write_text('installer: ' + str(self.single))
        with self.assertRaisesRegex(RuntimeError, 'code/config/environment reference'):
            self.run_case(True)

    def test_new_metadata_file_changes_snapshot(self):
        def mutate(*args):
            (self.root / 'exp/new_config.json').write_text('{}')
            return {'processes_checked': 1, 'references': 0}
        self.process_mock.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, 'metadata .*changed'):
            self.run_case(True)
        self.assertTrue(self.single.exists())

    def test_absent_index_creation_refused(self):
        def mutate(*args):
            (self.single.parent / 'index.json').write_text('{}')
            return {'processes_checked': 1, 'references': 0}
        self.process_mock.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, 'Protected index'):
            self.run_case(True)

    def test_existing_index_retained_and_pinned(self):
        index = self.single.parent / 'index.json'
        index.write_text('{"keep":true}')
        result = self.run_case(True)
        self.assertEqual(index.read_text(), '{"keep":true}')
        self.assertEqual(result['optional_indexes_before'][str(index)]['sha256'], mod.digest(index))

    def test_process_reference_refused(self):
        self.process_mock.side_effect = RuntimeError('Process holds cache')
        with self.assertRaisesRegex(RuntimeError, 'Process holds cache'):
            self.run_case(True)
        self.assertFalse(self.audit.exists())

    def test_protected_index_change_refused(self):
        def mutate(*args):
            self.keep.write_text('changed')
            return {'processes_checked': 1, 'references': 0}
        self.process_mock.side_effect = mutate
        with self.assertRaisesRegex(RuntimeError, 'Protected index'):
            self.run_case(True)
        self.assertTrue(self.single.exists())

    def test_busy_lock_refused_and_other_locks_released(self):
        with (self.root / 'exp/locks' / mod.LOCK_NAMES[1]).open('r+') as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_case(True)
        self.assert_locks_free()
        self.assertFalse(self.audit.exists())

    def test_second_unlink_failure_reports_one_removed(self):
        actual_unlink = os.unlink
        count = [0]
        def failing_unlink(*args, **kwargs):
            count[0] += 1
            if count[0] == 2:
                raise OSError('injected second unlink failure')
            return actual_unlink(*args, **kwargs)
        with mock.patch.object(mod.os, 'unlink', side_effect=failing_unlink):
            with self.assertRaisesRegex(OSError, 'second unlink'):
                self.run_case(True)
        report = json.loads((self.audit / 'failure.json').read_text())
        self.assertEqual(report['removed_count'], 1)
        self.assertFalse(Path(report['removed_files'][0]).exists())
        self.assertFalse((self.audit / 'completion.json').exists())
        self.assert_locks_free()

    def test_unlink_followup_fsync_failure_keeps_actual_cursor(self):
        actual_unlink, actual_fsync = os.unlink, os.fsync
        deleted = [False]
        def mark_unlink(*args, **kwargs):
            result = actual_unlink(*args, **kwargs)
            deleted[0] = True
            return result
        def fail_fsync(fd):
            if deleted[0]:
                deleted[0] = False
                raise OSError('post-unlink fsync')
            return actual_fsync(fd)
        with mock.patch.object(mod.os, 'unlink', side_effect=mark_unlink), mock.patch.object(mod.os, 'fsync', side_effect=fail_fsync):
            with self.assertRaisesRegex(OSError, 'post-unlink'):
                self.run_case(True)
        report = json.loads((self.audit / 'failure.json').read_text())
        self.assertEqual(report['removed_count'], 1)
        self.assertFalse((self.audit / 'completion.json').exists())

    def test_unexpected_directory_file_not_deleted(self):
        actual_rmdir = os.rmdir
        surprise = self.tree / 'nested/surprise'
        def introduce_file(*args, **kwargs):
            if not surprise.exists():
                surprise.write_text('unregistered')
            return actual_rmdir(*args, **kwargs)
        with mock.patch.object(mod.os, 'rmdir', side_effect=introduce_file):
            with self.assertRaises(OSError):
                self.run_case(True)
        self.assertEqual(surprise.read_text(), 'unregistered')
        report = json.loads((self.audit / 'failure.json').read_text())
        self.assertEqual(report['removed_count'], 3)
        self.assertEqual(report['removed_directories'], [])

    def test_close_failure_no_completion_all_locks_released(self):
        actual_open = Path.open
        fail_name = mod.LOCK_NAMES[1]
        class CloseFailure:
            def __init__(self, handle):
                self.handle = handle
            def fileno(self):
                return self.handle.fileno()
            def close(self):
                self.handle.close()
                raise OSError('injected lock close')
        def patched_open(path, *args, **kwargs):
            handle = actual_open(path, *args, **kwargs)
            return CloseFailure(handle) if path.name == fail_name and args == ('r+',) else handle
        with mock.patch.object(Path, 'open', patched_open):
            with self.assertRaisesRegex(OSError, 'lock close'):
                self.run_case(True)
        self.assertFalse((self.audit / 'completion.json').exists())
        self.assertEqual(json.loads((self.audit / 'failure.json').read_text())['removed_count'], 3)
        self.assert_locks_free()

    def test_published_completion_fsync_failure_invalidates(self):
        actual_sync = mod.fsync_directory
        def fail_after_publish(path):
            if (self.audit / 'completion.json').exists():
                raise OSError('completion fsync failed')
            return actual_sync(path)
        with mock.patch.object(mod, 'fsync_directory', side_effect=fail_after_publish):
            with self.assertRaisesRegex(OSError, 'completion fsync'):
                self.run_case(True)
        self.assertFalse((self.audit / 'completion.json').exists())
        self.assertTrue((self.audit / 'completion.invalidated.json').exists())
        self.assertEqual(json.loads((self.audit / 'failure.json').read_text())['removed_count'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
