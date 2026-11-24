"""归档182个已审且已备份的开发文件；默认只读，不递归删除任何目录。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

import retire_failed3347_checkpoints_v1 as g

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
PUB = Path('/root/teacher-hifigan-publication-20260906')
MANIFEST = DEV / 'MC2_NEXT_STORAGE_CANDIDATES_V2.json'
MANIFEST_SHA = '391cfb479edfa34a9487c8c1659f8a58d3817af29d899bc380dff20f52a0ac4b'
PROOF = DEV / 'mc2_storage_private_backup_receipt_v1.json'
AUDIT = DEV / 'mc2_storage_retirement_v1'
PRIVATE = 'C:/Users/rich_/AppData/Local/VoiceGradPrivateBackups/20260907/mc2_next_storage_candidates_v1'
GUARD_SHA = '774698257e589c003dbbf846451386d01ad2d1ce36e6552c8b4c91e97bb2b7c9'
LOCKS = ('text_unseen_custom_hifigan_pipeline.lock', 'text_unseen_custom_hifigan_train.lock',
         'text_unseen_fastvoicegrad_student.lock', 'text_unseen_fastvoicegrad_student_inference.lock')
RUNS = tuple(DEV / name for name in ('acdd_mc2_paired120_from3327_v1', 'acdd_half_lr_continuation480_v1',
                                   'evaluation_mc1_paired120_endpoint_v1', 'evaluation_mc2_paired120_endpoint_v1'))
PROTECTED = {
    DEV / 'acdd_mc2_paired120_from3327_v1/candidate_mc2/final_full.pt': 'd108d1793a2473bb42f6b1e556c67b6339a04da8b9c8b39f0a7905bf5a26d4f6',
    DEV / 'acdd_s5e5_staged120_from3227_v1/final_full.pt': '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556',
    DEV / 'acdd_s5e5_staged120_from3227_v1/student_step_003327.pt': '7bb68a0d053335a87a2c3c42684b864e27bf21a9f19d6f8d1b5bc8152955ca60',
    ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt': '1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6',
    ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/generator_step_2500000.pt': 'dacd362c4db46cdf7bd8428ace6f38d7d03976c60a3cd4c55022d38383094fb9',
    ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/config.json': '4b570fee140ffbda92bb217218792dca069b2d73627bc5e7bf7b1669869dd05a',
}


def stamp(path):
    # 每一字段都用十进制字符串，不能经过JS Number或任何浮点表示。
    return [str(value) for value in g.stamp(path)]


def read_pinned(path, expected):
    if g.sha(path) != expected:
        raise RuntimeError('Pinned evidence SHA mismatch: ' + str(path))
    return json.loads(path.read_text())


def load_manifest(path, expected, dev, count):
    value = read_pinned(path, expected)
    if value.get('schema_version') != 2 or value.get('stat_encoding') != 'decimal_strings_all_five_fields':
        raise RuntimeError('Only exact-string-stat V2 manifest is accepted')
    rows = [row for group in value['candidates'] for row in group['files']]
    if len(rows) != count or len({row['path'] for row in rows}) != count:
        raise RuntimeError('Wrong or duplicate fixed target count')
    for row in rows:
        target = Path(row['path'])
        if not target.is_absolute() or dev not in target.parents or '..' in target.parts or target.suffix not in ('.pt', '.wav'):
            raise RuntimeError('Target escaped fixed development scope')
        if len(row['stat']) != 5 or any(not isinstance(v, str) or not v.isdecimal() for v in row['stat']):
            raise RuntimeError('Unsafe numeric stat encoding')
        if int(row['stat'][2]) != row['size_bytes'] or row['nlink'] != 1 or row['type'] != 'regular_file':
            raise RuntimeError('Inconsistent manifest file metadata')
    return value, rows


def validate_backup(path, expected, manifest, manifest_sha, rows, dev, private):
    value = read_pinned(path, expected)
    if (value.get('schema_version'), value.get('status'), value.get('source_server_alias')) != (1, 'verified_private_backup', 'matpool-rental'):
        raise RuntimeError('Backup receipt is not verified private backup')
    if value.get('manifest_path') != str(manifest) or value.get('manifest_sha256') != manifest_sha:
        raise RuntimeError('Backup is bound to another manifest')
    if not value.get('verified_utc') or any(value.get(k) is not True for k in ('all_local_and_remote_sha256_verified', 'local_backups_read_only', 'remote_files_not_yet_removed')):
        raise RuntimeError('Incomplete backup evidence')
    codes = value.get('scp_exit_codes')
    if not isinstance(codes, list) or not codes or any(type(code) is not int or code != 0 for code in codes):
        raise RuntimeError('Backup transfer did not complete successfully')
    backups = value['target_files']
    by_path = {row['remote_path']: row for row in backups}
    if len(backups) != len(rows) or len(by_path) != len(rows) or set(by_path) != {row['path'] for row in rows}:
        raise RuntimeError('Backup target set is not exact')
    for row in rows:
        saved = by_path[row['path']]
        expected_local = private.rstrip('/') + '/' + Path(row['path']).relative_to(dev).as_posix()
        if (saved.get('local_path'), saved.get('size_bytes'), saved.get('sha256'), saved.get('local_sha256')) != (expected_local, row['size_bytes'], row['sha256'], row['sha256']):
            raise RuntimeError('Backup path/size/hash differs: ' + row['path'])
    return value


def file_snapshot(paths, expected=None):
    result = {}
    for path in sorted(paths):
        before = stamp(path)
        digest = g.sha(path)
        if stamp(path) != before or (expected is not None and digest != expected[path]):
            raise RuntimeError('Protected file changed: ' + str(path))
        result[str(path)] = {'sha256': digest, 'stat': before}
    return result


def metadata_paths(runs, targets):
    paths = set()
    for run in runs:
        g.regular(run, True)
        for parent, dirs, files in os.walk(run, followlinks=False):
            for name in dirs:
                g.regular(Path(parent) / name, True)
            for name in files:
                path = Path(parent) / name
                g.regular(path)
                if path not in targets and path.suffix != '.pt':
                    paths.add(path)
        paths.update(run.parent.glob(run.name + '*.log'))
    return sorted(paths)


def external_references(root, audit, exemptions, targets, allowed):
    # 只允许manifest已明确记录并pin住的历史JSON/CSV引用；未知现行依赖一律拒绝。
    found = []
    for parent, dirs, files in os.walk(root / 'exp', followlinks=False):
        dirs[:] = [name for name in dirs if name not in ('wav', 'wavs', 'staging', '.git', '__pycache__', '.pytest_cache') and Path(parent) / name != audit]
        for name in dirs:
            g.regular(Path(parent) / name, True)
        for name in files:
            path = Path(parent) / name
            if path in exemptions or path.suffix not in ('.json', '.csv'):
                continue
            g.regular(path)
            raw = path.read_bytes()
            text = json.dumps(json.loads(raw.decode('utf-8')), ensure_ascii=False) if path.suffix == '.json' else raw.decode('utf-8')
            if any(str(target) in text for target in targets):
                if str(path) not in allowed or g.sha(path) != allowed[str(path)]:
                    raise RuntimeError('Unknown or changed external dependency: ' + str(path))
                found.append(str(path))
    return sorted(found)


def publication(pub, targets):
    value = g.publication_guard(pub, targets)
    value['snapshot'] = {path: [str(v) for v in values] for path, values in value['snapshot'].items()}
    return value


def execute(root, dev, pub, manifest, manifest_sha, proof, proof_sha, audit, private, runs, protected, count, guard_sha, apply=False):
    if g.sha(Path(g.__file__).resolve()) != guard_sha:
        raise RuntimeError('Frozen guard source mismatch')
    if audit.exists() or audit.is_symlink():
        raise RuntimeError('Audit exists; never repeat a partial retirement')
    g.regular(audit.parent, True)
    document, rows = load_manifest(manifest, manifest_sha, dev, count)
    targets = [Path(row['path']) for row in rows]
    if set(targets) & set(protected):
        raise RuntimeError('A target is a protected model')
    handles, removed, failure, result = [], [], None, None
    try:
        for name in LOCKS:
            path = root / 'exp/locks' / name
            g.regular(path)
            handle = path.open('r+')
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        backup = validate_backup(proof, proof_sha, manifest, manifest_sha, rows, dev, private)
        for row in rows:
            path = Path(row['path'])
            if g.regular(path).st_nlink != 1 or stamp(path) != row['stat'] or g.sha(path) != row['sha256'] or stamp(path) != row['stat']:
                raise RuntimeError('Target source identity changed: ' + str(path))
        source_paths = {Path(__file__).resolve(): g.sha(Path(__file__).resolve()), Path(g.__file__).resolve(): guard_sha}
        protected_files = file_snapshot(protected, protected)
        metadata = file_snapshot(metadata_paths(runs, set(targets)))
        allowed = {row['path']: row['sha256'] for row in document['historical_metadata_references']}
        prior = document.get('previous_manifest')
        if prior and allowed.get(prior['path']) != prior['sha256']:
            raise RuntimeError('Historical V1 precision record lacks its exact preservation pin')
        for row in document.get('related_preserved_receipts', []):
            if row['path'] in allowed and allowed[row['path']] != row['sha256']:
                raise RuntimeError('Conflicting historical receipt pin')
            allowed[row['path']] = row['sha256']
        allowed_files = file_snapshot({Path(path) for path in allowed}, {Path(path): value for path, value in allowed.items()})
        exemptions = {manifest, proof}
        refs = external_references(root, audit, exemptions, targets, allowed)
        # publication只需四个固定组路径，避免339528条记录乘182个单文件路径。
        groups = [Path(group['target']) for group in document['candidates']]
        uploaded = publication(pub, groups)
        inodes = {(int(row['stat'][0]), int(row['stat'][1])) for row in rows}
        proc = g.process_guard(groups, inodes)

        def light_check():
            if g.sha(manifest) != manifest_sha or g.sha(proof) != proof_sha or any(g.sha(p) != expected for p, expected in source_paths.items()):
                raise RuntimeError('Manifest, backup receipt or frozen source changed')
            if {str(p): stamp(p) for p in metadata_paths(runs, set(targets))} != {p: row['stat'] for p, row in metadata.items()}:
                raise RuntimeError('Preserved run metadata changed')
            for saved in (protected_files, allowed_files):
                if any(stamp(Path(path)) != row['stat'] for path, row in saved.items()):
                    raise RuntimeError('Protected model or historical evidence changed')

        light_check()
        result = {'status': 'validated_not_deleted', 'source_sha256': g.sha(Path(__file__).resolve()),
                  'guard_sha256': guard_sha, 'manifest_path': str(manifest), 'manifest_sha256': manifest_sha,
                  'backup_receipt_path': str(proof), 'backup_receipt_sha256': proof_sha, 'verified_private_backup': backup,
                  'targets': rows, 'file_count': count, 'logical_bytes': sum(row['size_bytes'] for row in rows),
                  'file_allocation_bytes': sum(row['file_allocation_bytes'] for row in rows),
                  'protected_models_before': protected_files, 'metadata_before': metadata,
                  'allowed_historical_reference_files': allowed_files, 'historical_reference_hits': refs,
                  'publication_before': uploaded, 'processes_before': proc,
                  'restore_required_for_historical_full_validation_inference_or_scoring': True,
                  'all_old_logs_receipts_metrics_and_decisions_preserved': True, 'directories_not_deleted': True}
        if apply:
            if publication(pub, groups) != uploaded or external_references(root, audit, exemptions, targets, allowed) != refs:
                raise RuntimeError('Dependencies changed immediately before apply')
            g.process_guard(groups, inodes)
            light_check()
            audit.mkdir()
            g.write_new(audit / 'preflight.json', result)
            with (audit / 'removed_files.jsonl').open('x', encoding='utf-8') as journal:
                for row in rows:
                    path = Path(row['path'])
                    light_check()
                    g.process_guard(groups, inodes)
                    if stamp(path) != row['stat'] or g.regular(path).st_nlink != 1 or g.sha(path) != row['sha256']:
                        raise RuntimeError('Individual source changed before unlink')
                    parent = g.regular(path.parent, True)
                    fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        held = os.fstat(fd)
                        info = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                        exact = [str(v) for v in (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)]
                        if (held.st_dev, held.st_ino) != (parent.st_dev, parent.st_ino) or exact != row['stat'] or info.st_nlink != 1:
                            raise RuntimeError('Opened-directory target identity mismatch')
                        journal.write(json.dumps({'phase': 'prepared', 'path': str(path), 'sha256': row['sha256']}) + '\n')
                        journal.flush(); os.fsync(journal.fileno())
                        os.unlink(path.name, dir_fd=fd)
                        removed.append(str(path))
                        os.fsync(fd)
                        journal.write(json.dumps({'phase': 'removed', 'path': str(path)}) + '\n')
                        journal.flush(); os.fsync(journal.fileno())
                    finally:
                        os.close(fd)
            light_check()
            after_publication = publication(pub, groups)
            after_refs = external_references(root, audit, exemptions, targets, allowed)
            if after_publication != uploaded or after_refs != refs:
                raise RuntimeError('External references changed during retirement')
            if file_snapshot(protected, protected) != protected_files or file_snapshot(metadata_paths(runs, set(targets))) != metadata or file_snapshot({Path(p) for p in allowed}, {Path(p): h for p, h in allowed.items()}) != allowed_files:
                raise RuntimeError('Protected bytes changed during retirement')
            after_processes = g.process_guard(groups, inodes)
            if any(path.exists() or path.is_symlink() for path in targets):
                raise RuntimeError('A deleted target remains or reappeared')
            result.update(status='mc2_storage_candidates_retired_private_backup_verified', removed_files=removed,
                          removed_count=len(removed), publication_after=after_publication, processes_after=after_processes)
    except BaseException as error:
        failure = error
    finally:
        for handle in reversed(handles):
            try:
                handle.close()
            except BaseException as error:
                if failure is None:
                    failure = error
    if failure is None and apply:
        try:
            g.write_completion(audit, result)
        except BaseException as error:
            failure = error
    if failure is not None:
        report = {'status': 'mc2_storage_retirement_failed', 'error': repr(failure), 'removed_files': removed,
                  'removed_count': len(removed), 'recovery_requires_private_backup': bool(removed),
                  'manifest_sha256': manifest_sha, 'backup_receipt_sha256': proof_sha}
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if apply and audit.exists():
            try:
                g.write_new(audit / 'failure.json', report)
            except BaseException as secondary:
                print('Failure receipt write failed: ' + repr(secondary), flush=True)
        raise failure
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-receipt-sha256', required=True)
    parser.add_argument('--apply', action='store_true')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not sys.platform.startswith('linux') or Path.cwd() != ROOT:
        raise RuntimeError('Only the named rented project root is allowed')
    result = execute(ROOT, DEV, PUB, MANIFEST, MANIFEST_SHA, PROOF, args.backup_receipt_sha256,
                     AUDIT, PRIVATE, RUNS, PROTECTED, 182, GUARD_SHA, args.apply)
    print(json.dumps({key: result[key] for key in ('status', 'file_count', 'logical_bytes', 'file_allocation_bytes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
