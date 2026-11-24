"""仅租用 CPU 独立临时目录、几字节夹具；不读取真实权重或取得真实工作流锁。"""
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
import retire_mc2_fm1_candidate4047_v1 as n
r = n.r


class Candidate4047RetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='candidate4047_retire_toy_')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / 'project'; dev = root / 'exp/diagnostics/student_development_20260906'
        runs = [dev / name for name in n.RUN_NAMES]
        target = dev / n.RUN / n.ARM / 'final_full.pt'; target.parent.mkdir(parents=True)
        target.write_bytes(b'full4047'); target.chmod(0o444)
        for run in runs:
            run.mkdir(exist_ok=True)
            (run / 'run_contract.json').write_text(json.dumps({'historical_checkpoint': str(target)}))
            run.with_suffix('.log').write_text('immutable history\n')
        self.meta = runs[-1] / 'run_contract.json'
        kept = runs[2] / 'final_full.pt'; kept.write_bytes(b'keep3927')
        locks = root / 'exp/locks'; locks.mkdir(parents=True)
        for name in n.LOCK_NAMES: (locks / name).write_text('')
        pub = Path(self.temp.name) / 'publication'; lists = pub / 'direct-upload-runtime/lists'; lists.mkdir(parents=True)
        self.raw = lists / 'one.files-from-raw.txt'; self.manifest = lists / 'one.metadata.jsonl'; self.plan = pub / 'PLAN.json'
        self.set_publication('/unrelated', 'safe.py')
        evaluation = dev / n.EVALUATION; evaluation.mkdir()
        (evaluation / 'run_identity.json').write_text(json.dumps({'checkpoint': str(target)}))
        (evaluation / 'scores.csv').write_text('pair,score\n0,0.5\n')
        wavs = evaluation / 'wavs'; wavs.mkdir(); (wavs / 'old.wav').write_bytes(b'keep wave')
        (dev / 'RUN_QUEUE.md').write_text('historical ' + str(target))
        interface = dev / n.INTERFACE_NAME; interface.write_text('toy interface')
        extra_logs = [dev / (n.EVALUATION + suffix) for suffix in n.EXTRA_LOG_SUFFIXES]
        for path in extra_logs: path.write_text('preserved original log, including failed summary\n')
        self.s = dict(root=root, dev=dev, pub=pub, target=target, target_sha=r.sha(target),
            logical=target.stat().st_size, allocated=target.stat().st_blocks * 512,
            proof=dev / n.PROOF, proof_sha='', private=n.PRIVATE, audit=dev / n.AUDIT,
            protected={str(kept): r.sha(kept), str(interface): r.sha(interface)}, runs=runs,
            history=[interface] + extra_logs, evaluation=evaluation, publication_counts=(2, 1), source_sha=r.sha(Path(n.__file__)))
        self.proof = dict(status='verified_private_backup', scp_exit_code=0,
            all_local_and_remote_sha256_verified=True, local_backups_read_only=True, remote_files_not_yet_removed=True,
            target_files=[dict(remote_path=str(target), local_path=n.PRIVATE, sha256=self.s['target_sha'],
                local_sha256=self.s['target_sha'], size_bytes=self.s['logical'], local_size_bytes=self.s['logical'],
                allocated_bytes_remote=self.s['allocated'], local_read_only=True, source_nlink=1, source_mode='0o444',
                source_identity_verified_after_copy=True, source_stat_decimal_strings=[str(x) for x in r.stamp(target)])])
        self.write_proof()

    def write_proof(self):
        self.s['proof'].write_text(json.dumps(self.proof)); self.s['proof_sha'] = r.sha(self.s['proof'])

    def set_publication(self, root, relative):
        self.raw.write_text(relative + '\n')
        self.manifest.write_text(json.dumps({'source_path': root + '/' + relative}) + '\n')
        self.plan.write_text(json.dumps({'source_root': root, 'file_list_path': str(self.raw), 'file_list_sha256': r.sha(self.raw),
            'files_manifest_path': str(self.manifest), 'files_manifest_sha256': r.sha(self.manifest)}))

    def call(self, apply=False):
        with mock.patch.object(n.old, 'gpu_guard', return_value={'compute_processes': 0}), \
             mock.patch.object(r, 'process_guard', return_value={'processes_checked': 1, 'references': 0}), \
             contextlib.redirect_stdout(io.StringIO()):
            return n.execute(self.s, apply)

    def failure(self):
        self.assertFalse((self.s['audit'] / 'completion.json').exists())
        return json.loads((self.s['audit'] / 'failure.json').read_text())

    def test_uses_frozen_functions_without_old_main_execute_or_spec(self):
        for name in ('validate_backup', 'target_check', 'unlink_exact', 'guards', 'historical_inventory'):
            self.assertIs(getattr(n, name), getattr(n.old, name))
        with mock.patch.object(n.old, 'main', side_effect=AssertionError), \
             mock.patch.object(n.old, 'execute', side_effect=AssertionError), \
             mock.patch.object(n.old, 'production_spec', side_effect=AssertionError):
            self.call()

    def test_default_readonly_and_sha_only_cli(self):
        before = n.evidence_snapshot(self.s)
        with mock.patch.object(n.old.os, 'unlink', side_effect=AssertionError('No dryrun unlink')):
            result = self.call()
        self.assertEqual(result['status'], 'validated_not_deleted'); self.assertFalse(self.s['audit'].exists())
        self.assertEqual(n.evidence_snapshot(self.s), before)
        args = ['--source-sha256', self.s['source_sha'], '--proof-sha256', self.s['proof_sha']]
        self.assertFalse(n.parse_args(args).apply)
        for extra in (['--target', '/tmp/arbitrary'], ['--proof-sha256', 'bad']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): n.parse_args(args + extra)

    def test_one_tiny_target_removed_metadata_and_directory_kept(self):
        before = n.evidence_snapshot(self.s); result = self.call(True)
        self.assertEqual(result['removed_files'], [str(self.s['target'])]); self.assertEqual(result['removed_count'], 1)
        self.assertFalse(self.s['target'].exists()); self.assertTrue(self.s['target'].parent.is_dir())
        self.assertEqual(n.evidence_snapshot(self.s), before)
        self.assertEqual((self.s['evaluation'] / 'wavs/old.wav').read_bytes(), b'keep wave')
        self.assertEqual((self.s['evaluation'] / 'scores.csv').read_text(), 'pair,score\n0,0.5\n')
        rows = [json.loads(x) for x in (self.s['audit'] / 'removed_files.jsonl').read_text().splitlines()]
        self.assertEqual([x['phase'] for x in rows], ['prepared', 'removed'])

    def test_proof_requires_exact_one_target_bytes_private_readonly_and_decimal_stat(self):
        saved = json.loads(json.dumps(self.proof))
        for kind in ('sha', 'two', 'private', 'readonly', 'stat', 'bool_link'):
            self.proof = json.loads(json.dumps(saved)); row = self.proof['target_files'][0]
            if kind == 'two': self.proof['target_files'].append(dict(row))
            if kind == 'private': row['local_path'] += '.partial'
            if kind == 'readonly': row['local_read_only'] = False
            if kind == 'stat': row['source_stat_decimal_strings'][0] = 1
            if kind == 'bool_link': row['source_nlink'] = True
            self.write_proof()
            if kind == 'sha': self.s['proof_sha'] = '0' * 64
            with self.subTest(kind=kind), self.assertRaises(RuntimeError): self.call()
            self.assertTrue(self.s['target'].exists()); self.assertFalse(self.s['audit'].exists())

    def test_exact_paths_not_overridable(self):
        for key in ('target', 'proof', 'audit', 'private', 'evaluation'):
            saved = self.s[key]; self.s[key] = str(saved) + '.other' if key == 'private' else saved.with_name('other')
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'escaped'): self.call()
            self.s[key] = saved

    def test_hardlink_and_symlink_rejected(self):
        target = self.s['target']; link = target.with_name('alias.pt'); os.link(target, link)
        with self.assertRaisesRegex(RuntimeError, 'stat/mode/link'): self.call()
        link.unlink()
        self.proof['target_files'][0]['source_stat_decimal_strings'] = [str(x) for x in r.stamp(target)]; self.write_proof()
        actual = target.parent.with_name('actual'); target.parent.rename(actual); target.parent.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'symlink'): self.call()

    def test_source_and_protected_bytes_changed_block(self):
        saved = self.s['source_sha']; self.s['source_sha'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'source SHA'): self.call()
        self.s['source_sha'] = saved
        Path(next(iter(self.s['protected']))).write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Protected asset'): self.call()

    def test_production_spec_binds_source_tests_interface_and_fixed_D(self):
        spec = n.production_spec(self.s['source_sha'], self.s['proof_sha'])
        self.assertEqual(spec['protected'][str(n.TOOLS / n.TEST_NAME)], r.sha(n.TOOLS / n.TEST_NAME))
        self.assertEqual(spec['protected'][str(n.DEV / n.INTERFACE_NAME)], r.sha(n.DEV / n.INTERFACE_NAME))
        self.assertIn(str(n.DEV / 'acdd_s2000_to4000_v1/latest_full.pt'), spec['protected'])
        self.assertNotIn(str(spec['target']), spec['protected'])
        self.assertIn(n.DEV / 'acdd_mc2_dlr_paired120_from3927_v1', spec['runs'])
        for suffix in n.EXTRA_LOG_SUFFIXES:
            self.assertIn(n.DEV / (n.EVALUATION + suffix), spec['history'])

    def test_publication_reference_and_scope_change_block(self):
        self.set_publication(str(self.s['target'].parent), self.s['target'].name)
        with self.assertRaisesRegex(RuntimeError, 'checkpoint reference'): self.call()
        self.set_publication('/unrelated', 'safe.py'); self.s['publication_counts'] = (218, 4)
        with self.assertRaisesRegex(RuntimeError, 'enumeration'): self.call()

    def test_gpu_and_process_fail_closed(self):
        for owner, name in ((n.old, 'gpu_guard'), (r, 'process_guard')):
            with mock.patch.object(n.old, 'gpu_guard', return_value={}), mock.patch.object(r, 'process_guard', return_value={}):
                with mock.patch.object(owner, name, side_effect=RuntimeError('active dependency')), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'active dependency'): n.execute(self.s)

    def test_mutation_immediately_before_unlink_rejected(self):
        original = n.unlink_exact
        def changed(s, expected, journal, removed):
            s['target'].chmod(0o644); s['target'].write_bytes(b'changed'); s['target'].chmod(0o444)
            return original(s, expected, journal, removed)
        with mock.patch.object(n, 'unlink_exact', side_effect=changed), self.assertRaisesRegex(RuntimeError, 'changed'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 0); self.assertTrue(self.s['target'].exists())

    def test_apply_preflight_failure_is_persistent(self):
        self.s['proof_sha'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'proof SHA'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 0); self.write_proof()
        with self.assertRaisesRegex(RuntimeError, 'already exists'): self.call(True)

    def test_post_unlink_fsync_failure_records_real_removal_and_no_retry(self):
        original = n.old.os.fsync
        def failed(fd):
            if not self.s['target'].exists(): raise OSError('post unlink fsync')
            return original(fd)
        with mock.patch.object(n.old.os, 'fsync', side_effect=failed), self.assertRaisesRegex(OSError, 'post unlink'): self.call(True)
        self.assertFalse(self.s['target'].exists()); self.assertEqual(self.failure()['removed_count'], 1)
        with self.assertRaisesRegex(RuntimeError, 'already exists'): self.call(True)

    def test_new_gradient_metadata_change_cannot_publish_success(self):
        original = n.unlink_exact
        def changed(*args): original(*args); self.meta.write_text('{"changed":true}')
        with mock.patch.object(n, 'unlink_exact', side_effect=changed), self.assertRaisesRegex(RuntimeError, 'Protected evidence'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 1)

    def test_queue_changes_allowed_and_old_wavs_never_hashed(self):
        original = n.unlink_exact; original_sha = r.sha
        def changed(*args): original(*args); (self.s['dev'] / 'RUN_QUEUE.md').write_text('new schedule')
        def digest(path):
            if Path(path).suffix == '.wav': raise AssertionError('No old WAV read')
            return original_sha(path)
        with mock.patch.object(n, 'unlink_exact', side_effect=changed), mock.patch.object(r, 'sha', side_effect=digest): result = self.call(True)
        self.assertFalse(result['historical_reference_inventory'][str(self.s['dev'] / 'RUN_QUEUE.md')]['protected_frozen_metadata'])

    def test_cleanup_failure_closes_all_locks_without_success(self):
        original = Path.open; closed = []
        class Handle:
            def __init__(self, handle, name): self.handle, self.name = handle, name
            def fileno(self): return self.handle.fileno()
            def close(self):
                self.handle.close(); closed.append(self.name)
                if self.name == n.LOCK_NAMES[-1]: raise OSError('lock close failure')
        def opening(path, *args, **kwargs):
            handle = original(path, *args, **kwargs)
            return Handle(handle, path.name) if path.name in n.LOCK_NAMES and args and args[0] == 'r+' else handle
        with mock.patch.object(Path, 'open', opening), self.assertRaisesRegex(OSError, 'lock close'): self.call(True)
        self.assertEqual(set(closed), set(n.LOCK_NAMES)); self.assertEqual(self.failure()['removed_count'], 1)

    def assert_extra_log_mutation_blocks(self, suffix):
        original = n.guards; calls = []
        path = self.s['dev'] / (n.EVALUATION + suffix)
        self.assertIn(str(path), n.evidence_snapshot(self.s)['metadata_sha256'])
        def changed(s, inodes):
            result = original(s, inodes); calls.append(1)
            if len(calls) == 2: path.write_text('changed historical log')
            return result
        with mock.patch.object(n, 'guards', side_effect=changed), self.assertRaisesRegex(RuntimeError, 'Protected evidence'):
            self.call(True)
        self.assertEqual(self.failure()['removed_count'], 0)
        self.assertTrue(self.s['target'].exists())

    def test_cer_log_mutation_blocks(self):
        self.assert_extra_log_mutation_blocks('.cer.log')

    def test_quality_log_mutation_blocks(self):
        self.assert_extra_log_mutation_blocks('.quality.log')

    def test_failed_summary_log_mutation_blocks(self):
        self.assert_extra_log_mutation_blocks('.summary.log')

    def test_corrected_summary_log_mutation_blocks(self):
        self.assert_extra_log_mutation_blocks('.summary_cuda_identity.log')

    def test_completion_publish_failure_is_not_success(self):
        with mock.patch.object(r, 'write_completion', side_effect=OSError('publish failed')), self.assertRaisesRegex(OSError, 'publish'): self.call(True)
        self.assertEqual(self.failure()['removed_count'], 1)


if __name__ == '__main__': unittest.main(verbosity=2)
