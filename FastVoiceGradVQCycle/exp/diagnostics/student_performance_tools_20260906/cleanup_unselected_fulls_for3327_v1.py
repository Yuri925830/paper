"""仅删除用户未选用的三份完整权重，保留逐项记录；只能在租用服务器运行。"""
import fcntl
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
MANIFEST = DEV / 'selected3327_cleanup_three_full_manifest_v1.json'
RECEIPT = DEV / 'selected3327_cleanup_three_full_receipt_v1.json'
EXPECTED = {
    str(DEV / 'acdd_mc2_continuation480_from3447_v1/final_full.pt'),
    str(DEV / 'acdd_mc2_independent360_from3927_v1/candidate_mc2/final_full.pt'),
    str(DEV / 'acdd_mc2_perceptual240_from4287_v1/candidate_perceptual/final_full.pt'),
}
PROTECTED = {
    DEV / 'acdd_s5e5_staged120_from3227_v1/final_full.pt': '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556',
    ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt': '1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6',
    ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/generator_step_2500000.pt': 'dacd362c4db46cdf7bd8428ace6f38d7d03976c60a3cd4c55022d38383094fb9',
}

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save(value):
    temporary = RECEIPT.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, RECEIPT)

def check_file(row):
    path = Path(row['path'])
    if str(path) not in EXPECTED or path.resolve(strict=True) != path or DEV not in path.parents:
        raise RuntimeError('Deletion path or ancestry changed')
    current = path.lstat()
    expected = (int(row['st_dev_decimal']), int(row['st_ino_decimal']), row['st_size'],
                int(row['st_mtime_ns_decimal']), int(row['st_ctime_ns_decimal']))
    actual = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
    if actual != expected or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
        raise RuntimeError('Deletion target stat changed: ' + str(path))
    return path

def main():
    if not sys.platform.startswith('linux') or not ROOT.is_dir() or RECEIPT.exists():
        raise RuntimeError('Require rental and fresh receipt; never repeat cleanup silently')
    manifest = json.loads(MANIFEST.read_text())
    if {row['path'] for row in manifest['files']} != EXPECTED or len(manifest['files']) != 3:
        raise RuntimeError('Only the three reviewed fulls are authorized here')
    locks = []
    for name in ('text_unseen_custom_hifigan_pipeline', 'text_unseen_custom_hifigan_train',
                 'text_unseen_fastvoicegrad_student', 'text_unseen_fastvoicegrad_student_inference'):
        handle = (ROOT / 'exp/locks' / (name + '.lock')).open('a+')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    for path, digest in PROTECTED.items():
        if sha(path) != digest:
            raise RuntimeError('Essential model identity changed: ' + str(path))
    for row in manifest['files']:
        if sha(check_file(row)) != row['sha256']:
            raise RuntimeError('Unselected model identity changed')
    # 文件描述符仍引用目标时，不尝试释放，避免干扰别的工作。
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for descriptor in (process / 'fd').iterdir():
                try:
                    if os.readlink(str(descriptor)) in EXPECTED:
                        raise RuntimeError('A process is still using a deletion target')
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, PermissionError):
            pass
    before = os.statvfs(str(ROOT))
    record = {'status': 'deleting', 'manifest_sha256': sha(MANIFEST), 'deleted': [],
              'free_before_bytes': before.f_bavail * before.f_frsize,
              'protected_model_sha256': {str(p): h for p, h in PROTECTED.items()}}
    save(record)
    for row in manifest['files']:
        path = check_file(row)
        path.unlink()  # 只移除已逐项核验的普通文件，不递归删除任何目录。
        record['deleted'].append(row)
        save(record)
    if any(Path(name).exists() for name in EXPECTED):
        raise RuntimeError('A reviewed target remains')
    for path, digest in PROTECTED.items():
        if sha(path) != digest:
            raise RuntimeError('Post-cleanup essential-model verification failed')
    after = os.statvfs(str(ROOT))
    record.update(status='complete', free_after_bytes=after.f_bavail * after.f_frsize,
                  allocated_bytes_removed=sum(row['allocated_bytes'] for row in record['deleted']))
    save(record)
    print(json.dumps({key: value for key, value in record.items() if key != 'deleted'}))

if __name__ == '__main__':
    main()
