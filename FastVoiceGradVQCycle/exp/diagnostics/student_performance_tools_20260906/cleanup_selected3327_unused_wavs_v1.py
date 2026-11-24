"""只删除已明确授权的482个未采用试验WAV；目录、分数、日志均保留。"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
MANIFEST = TOOLS / 'selected3327_cleanup_new_wavs_manifest_v1.json'
MANIFEST_SHA = 'd0c3a40c0eaef894372feca3656eab7fd094f148ff45d6454089017c41c18c90'
GUARD = TOOLS / 'retire_failed3347_checkpoints_v1.py'
GUARD_SHA = '774698257e589c003dbbf846451386d01ad2d1ce36e6552c8b4c91e97bb2b7c9'
PUB = Path('/root/teacher-hifigan-publication-20260906')
AUDIT = DEV / 'selected3327_unused_wavs_cleanup_v1'
GROUPS = {
    'evaluation_mc2_continuation480_endpoint_v1/student_mc2_continuation480_endpoint_v1/wavs': 90,
    'evaluation_mc2_dlr_d1e4_paired120_endpoint_v1/student_mc2_dlr_d1e4_paired120_endpoint_v1/wavs': 90,
    'evaluation_mc2_fm1_paired120_endpoint_v1/student_mc2_fm1_paired120_endpoint_v1/wavs': 90,
    'evaluation_mc2_independent360_endpoint_v1/student_mc2_independent360_endpoint_v1/wavs': 90,
    'evaluation_mc2_perceptual240_from4287_endpoint_v1/student_mc2_perceptual240_from4287_endpoint_v1/wavs': 90,
    'teacher30_anchor_train16_v1/teacher500_formal_k30_train16/wavs': 16,
    'teacher30_anchor_train16_v1/student4287_k1_train16/wavs': 16,
}
PROTECTED = [DEV / 'acdd_s5e5_staged120_from3227_v1/final_full.pt',
    DEV / 'acdd_s5e5_staged120_from3227_v1/student_step_003327.pt',
    ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt',
    ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/generator_step_2500000.pt']


def free_bytes():
    value = os.statvfs(ROOT)
    return value.f_bavail * value.f_frsize


def main():
    if len(sys.argv) != 2 or not sys.platform.startswith('linux') or Path.cwd() != ROOT:
        raise RuntimeError('Exact rental cwd and expected script SHA required')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != sys.argv[1]:
        raise RuntimeError('Executing source SHA changed')
    if hashlib.sha256(GUARD.read_bytes()).hexdigest() != GUARD_SHA:
        raise RuntimeError('Frozen read-only guard changed')
    spec = importlib.util.spec_from_file_location('_selected3327_wav_guard', GUARD)
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    r.regular(MANIFEST)
    if r.sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('Exact manifest SHA mismatch')
    inventory = json.loads(MANIFEST.read_text())
    rows = inventory['files']
    targets = [Path(row['path']) for row in rows]
    allowed = {DEV / key: count for key, count in GROUPS.items()}
    if len(rows) != 482 or len(set(targets)) != 482 or inventory['allocated_bytes_total'] != 79130624:
        raise RuntimeError('Manifest cardinality/allocation mismatch')
    for folder, count in allowed.items():
        r.regular(folder, True)
        chosen = {path for path in targets if path.parent == folder}
        if len(chosen) != count or set(folder.glob('*.wav')) != chosen:
            raise RuntimeError('Exact group WAV set changed')
    for path in targets:
        if not path.is_absolute() or path.parent not in allowed or path.suffix != '.wav' or path.resolve() != path:
            raise RuntimeError('Target escaped the seven exact WAV groups')
    if AUDIT.exists() or AUDIT.is_symlink():
        raise RuntimeError('Audit exists; do not repeat or overwrite a partial cleanup')

    def check(row):
        path = Path(row['path'])
        value = r.regular(path)
        expected = [int(row[key]) for key in ('st_dev_decimal', 'st_ino_decimal', 'st_size', 'st_mtime_ns_decimal', 'st_ctime_ns_decimal')]
        if (r.stamp(path) != expected or value.st_blocks != row['st_blocks']
                or value.st_nlink != 1 or oct(stat.S_IMODE(value.st_mode)) != row['mode']):
            raise RuntimeError('WAV stat or mode changed: ' + str(path))
        return expected

    def publication():
        # 原函数验证全部清单、source_root映射、声明SHA及审计前后不变。
        proof = r.publication_guard(PUB, [])
        names = {str(path) for path in targets}
        listed, plans = r.publication_paths(PUB)
        for path in plans:
            r.reject_references(r.strings(json.loads(path.read_text())), targets, path)
        for path in listed:
            if path.name.endswith('.metadata.jsonl'):
                for line in path.read_text().splitlines():
                    if json.loads(line).get('source_path') in names:
                        raise RuntimeError('WAV is a publication source: ' + str(path))
        return proof

    handles = []
    removed = []
    owned = False
    initial_free = free_bytes()
    locks = r.LOCK_NAMES + ('text_unseen_fastvoicegrad_student_inference.lock',)
    inodes = {(int(row['st_dev_decimal']), int(row['st_ino_decimal'])) for row in rows}
    try:
        for name in locks:
            path = ROOT / 'exp/locks' / name
            r.regular(path)
            handle = path.open('r')
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for row in rows:
            check(row)
        protected = {str(path): r.stamp(path) for path in PROTECTED}
        metadata = {}
        for folder in allowed:
            for parent, dirs, files in os.walk(folder.parent, followlinks=False):
                dirs[:] = [name for name in dirs if name not in ('wav', 'wavs')]
                for name in files:
                    path = Path(parent) / name
                    if path.suffix != '.pt':
                        metadata[str(path)] = r.sha(path)
        before_publication = publication()
        process = r.process_guard(targets, inodes)
        AUDIT.mkdir()
        owned = True
        r.write_new(AUDIT / 'preflight.json', {'manifest_sha256': MANIFEST_SHA,
            'source_sha256': sys.argv[1], 'file_count': 482, 'allocated_bytes': 79130624,
            'free_bytes_before': initial_free, 'protected_stat': protected,
            'metadata_sha256': metadata, 'publication': before_publication, 'process_guard': process})
        with (AUDIT / 'removed_files.jsonl').open('x', encoding='utf-8') as journal:
            for row in rows:
                path = Path(row['path'])
                expected = check(row)
                parent_info = r.regular(path.parent, True)
                fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    opened = os.fstat(fd)
                    current = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                    if ((opened.st_dev, opened.st_ino) != (parent_info.st_dev, parent_info.st_ino)
                            or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
                            or [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns] != expected):
                        raise RuntimeError('Opened-directory WAV identity mismatch')
                    journal.write(json.dumps({'phase': 'prepared', **row}) + '\n')
                    journal.flush()
                    os.fsync(journal.fileno())
                    os.unlink(path.name, dir_fd=fd)
                    removed.append(row['path'])
                    os.fsync(fd)
                    journal.write(json.dumps({'phase': 'removed', 'path': row['path']}) + '\n')
                    journal.flush()
                    os.fsync(journal.fileno())
                finally:
                    os.close(fd)
        if any(path.exists() or path.is_symlink() for path in targets):
            raise RuntimeError('A removed WAV remains or reappeared')
        if any(r.stamp(Path(name)) != value for name, value in protected.items()):
            raise RuntimeError('Protected selected3327/teacher/vocoder stat changed')
        if any(r.sha(Path(name)) != digest for name, digest in metadata.items()):
            raise RuntimeError('Original score/metadata changed')
        if publication() != before_publication:
            raise RuntimeError('Publication list or plan changed')
        process_after = r.process_guard(targets, inodes)
        result = {'status': 'completed_exact482_wav_removal', 'manifest_sha256': MANIFEST_SHA,
            'source_sha256': sys.argv[1], 'removed_count': len(removed), 'removed_files': removed,
            'allocation_bytes_removed': 79130624, 'free_bytes_before': initial_free,
            'free_bytes_after': free_bytes(), 'all_directories_preserved': True,
            'all_original_score_metadata_preserved': True, 'protected_model_stat_unchanged': True,
            'publication_unchanged': True, 'process_guard_after': process_after,
            'recovery_note': inventory['recoverability']}
    except BaseException as error:
        if owned:
            r.write_new(AUDIT / 'failure.json', {'status': 'failed', 'error': repr(error),
                'removed_count': len(removed), 'removed_files': removed})
        raise
    finally:
        for handle in reversed(handles):
            handle.close()
    r.write_completion(AUDIT, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
