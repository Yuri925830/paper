"""仅移除三个不活跃的旧VSCode服务器缓存与Conda下载归档；保留当前连接和环境。"""

import argparse
import datetime
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
SERVERS = Path('/root/.vscode-server/cli/servers')
OLD = tuple(SERVERS / name for name in (
    'Stable-08d4889f9ec4a1685d257b9b95de036c8e1ce1e5',
    'Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1',
    'Stable-a44adf7f53e00964ab890f9f8758a334f1fc15bc'))
ACTIVE = Path('/root/.vscode-server/bin/125df4672b8a6a34975303c6b0baa124e560a4f7')
PKGS = Path('/root/miniconda3/pkgs')
TEMPFILES = tuple(Path('/tmp') / name for name in ('tmpapgjf4wp', 'tmpb5131qlr', 'tmpqclmhzv5'))
UPLOAD_LISTS = Path('/root/teacher-hifigan-publication-20260906/direct-upload-runtime/lists')
RECEIPT = DEV / 'cleanup_obsolete_runtime_cache_20260907_v1.jsonl'


def file_identity(path):
    value = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise RuntimeError('Refuse symlink/nonregular/shared file: ' + str(path))
    return {'path': str(path), 'size': value.st_size, 'allocated': value.st_blocks * 512,
            'inode': value.st_ino, 'device': value.st_dev, 'mtime_ns': value.st_mtime_ns}


def process_check(archives):
    # /proc中的命令、映射及打开文件都检查；不能只看目录里可能过期的pid.txt。
    matches, inaccessible = [], []
    needles = [str(path) for path in OLD] + [str(path) for path in archives] + [str(path) for path in TEMPFILES]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        bodies = []
        for name in ('cmdline', 'maps'):
            try:
                bodies.append((proc / name).read_text(errors='replace').replace('\x00', ' '))
            except (FileNotFoundError, ProcessLookupError):
                pass
            except PermissionError:
                inaccessible.append(str(proc / name))
        links = [proc / 'cwd', proc / 'exe']
        try:
            links += list((proc / 'fd').iterdir())
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            inaccessible.append(str(proc / 'fd'))
        for link in links:
            try:
                bodies.append(os.readlink(link))
            except (FileNotFoundError, ProcessLookupError, OSError):
                pass
        text = '\n'.join(bodies)
        if any(needle in text for needle in needles) or any(word in text for word in ('conda install', 'conda create', 'conda update')):
            matches.append(proc.name)
    if matches or inaccessible:
        raise RuntimeError('Active or uninspectable process prevents cleanup: ' + repr((matches, inaccessible)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-reviewed', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix' or Path.cwd().resolve() != ROOT or not (ROOT / 'AGENTS.md').is_file():
        raise RuntimeError('Matpool root only')
    if RECEIPT.exists() or not ACTIVE.is_dir() or ACTIVE.resolve() != ACTIVE:
        raise RuntimeError('Refuse duplicate cleanup or missing active VSCode version')
    archives = sorted(PKGS.glob('*.conda'))
    if PKGS.resolve() != PKGS or len(archives) != 48:
        raise RuntimeError('Conda archive inventory changed; review again')
    process_check(archives)
    files, directories = [], []
    for directory in OLD:
        if not directory.is_dir() or directory.is_symlink() or directory.resolve().parent != SERVERS:
            raise RuntimeError('Old server root differs from exact whitelist')
        directories.append(directory)
        for path in directory.rglob('*'):
            if path.is_symlink() or directory not in path.resolve().parents:
                raise RuntimeError('Escaping path or symlink inside old server')
            if path.is_dir():
                directories.append(path)
            else:
                files.append(file_identity(path))
    # 只取顶层下载归档；解包后的pkgs及envs目录从不进入删除列表。
    for path in archives:
        if path.resolve().parent != PKGS:
            raise RuntimeError('Conda archive escaped the package cache')
        files.append(file_identity(path))
    for path in TEMPFILES:
        if path.resolve() != path or path.parent != Path('/tmp'):
            raise RuntimeError('Temporary archive escaped exact reviewed path')
        files.append(file_identity(path))
    needles = [str(path) for path in OLD] + [str(path) for path in archives] + [str(path) for path in TEMPFILES]
    documents = [p for p in (ROOT / 'exp').rglob('*.json') if 'contract' in p.name or 'identity' in p.name]
    documents += [p for p in UPLOAD_LISTS.iterdir() if p.suffix in ('.txt', '.jsonl', '.json')]
    for path in documents:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if any(needle in line for needle in needles):
                    raise RuntimeError('A runtime/upload identity references a target: ' + str(path))
    fs = os.statvfs(DEV)
    before = fs.f_bavail * fs.f_frsize
    plan = {'event': 'reviewed_plan', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'old_server_roots': [str(p) for p in OLD], 'preserved_active_vscode': str(ACTIVE),
        'files': files, 'directories': [str(p) for p in directories], 'conda_archives': len(archives),
        'obsolete_temp_archives': [str(p) for p in TEMPFILES],
        'allocated_file_bytes': sum(r['allocated'] for r in files), 'free_bytes_before': before,
        'documents_checked': len(documents), 'recovery': 'No local backup; obsolete tool versions/packages require reinstall or redownload if needed later.'}
    if not args.execute_reviewed:
        print(json.dumps({k: v for k, v in plan.items() if k not in ('files', 'directories')}, indent=2))
        return
    # 从计划到执行前再查一次使用者；只删除已核对的文件，再rmdir空目录，绝不递归删除泛化路径。
    process_check(archives)
    with RECEIPT.open('x', encoding='utf-8') as receipt:
        def emit(value):
            receipt.write(json.dumps(value, ensure_ascii=False) + '\n')
            receipt.flush()
            os.fsync(receipt.fileno())
        emit(plan)
        for item in files:
            path = Path(item['path'])
            if file_identity(path) != item or not (any(p in path.resolve().parents for p in OLD) or (path.parent == PKGS and path.suffix == '.conda') or (path in TEMPFILES and path.resolve() == path)):
                raise RuntimeError('File identity or deletion boundary changed')
            path.unlink()
            emit({'event': 'removed_file', 'path': str(path), 'size': item['size']})
        for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            if path.is_symlink() or not any(path == p or p in path.resolve().parents for p in OLD):
                raise RuntimeError('Directory escaped the exact old version roots')
            path.rmdir()  # 目录中新出现文件会使操作失败，不会被额外删除。
        if not ACTIVE.is_dir():
            raise RuntimeError('Active VSCode version is unexpectedly absent')
        fs = os.statvfs(DEV)
        after = fs.f_bavail * fs.f_frsize
        result = {'event': 'completed', 'removed_files': len(files), 'removed_directories': len(directories),
                  'free_bytes_after': after, 'observed_free_bytes_increase': after - before,
                  'active_vscode_preserved': str(ACTIVE), 'all_project_models_preserved': True}
        emit(result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
