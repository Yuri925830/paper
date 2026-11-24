"""清理两处明确允许再下载的pip HTTP缓存；保留环境、编译轮子和全部模型。"""

import argparse
import datetime
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
TARGETS = (Path('/root/.cache/pip/http'), Path('/root/.cache/pip/http-v2'))
RECEIPT = DEV / 'cleanup_pip_http_20260907_v1.jsonl'


def snapshot(path):
    value = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise RuntimeError('Refuse symlink, nonregular file or hardlinked cache entry: ' + str(path))
    return {'path': str(path), 'size': value.st_size, 'allocated': value.st_blocks * 512,
            'device': value.st_dev, 'inode': value.st_ino, 'mtime_ns': value.st_mtime_ns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-reviewed', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix' or Path.cwd().resolve() != ROOT or not (ROOT / 'AGENTS.md').is_file():
        raise RuntimeError('Only orchestrate cleanup on the current Matpool project')
    if RECEIPT.exists():
        raise RuntimeError('A cleanup receipt already exists; inspect it, never duplicate this run')
    # 防止与正在安装软件的进程竞争；其他上传工作及模型文件不在目标范围。
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    for line in processes.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != os.getpid():
            if any(token in parts[1] for token in ('pip install', 'pip download', 'pip wheel', 'conda install', 'conda create')):
                raise RuntimeError('An installation/download process is active; postpone cleanup')
    files = []
    for directory in TARGETS:
        if directory.is_symlink() or directory.resolve() != directory or not directory.is_dir():
            raise RuntimeError('Cache root is missing or resolves elsewhere: ' + str(directory))
        for path in sorted(directory.rglob('*')):
            if path.is_symlink() or directory not in path.resolve().parents:
                raise RuntimeError('Refuse an escaping cache path: ' + str(path))
            if path.is_dir():
                continue
            files.append(snapshot(path))
    if not files:
        raise RuntimeError('No cache files remain; do not report a new cleanup')
    # 只读核对项目契约和评分身份，下载缓存不允许是被冻结的运行依赖。
    dependencies_checked = 0
    for path in (ROOT / 'exp').rglob('*.json'):
        if 'contract' in path.name or 'identity' in path.name:
            body = path.read_text(encoding='utf-8')
            if any(str(directory) in body for directory in TARGETS):
                raise RuntimeError('A frozen runtime identity references a cleanup target: ' + str(path))
            dependencies_checked += 1
    fs = os.statvfs(DEV)
    before = fs.f_bavail * fs.f_frsize
    plan = {'event': 'reviewed_plan', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'targets': [str(path) for path in TARGETS], 'files': files,
            'logical_bytes': sum(item['size'] for item in files),
            'allocated_bytes': sum(item['allocated'] for item in files), 'free_bytes_before': before,
            'dependency_documents_checked': dependencies_checked,
            'preserved': ['all .pt/.pth/.onnx and project data', 'both installed conda environments',
                          'pip/wheels including locally compiled fairseq', 'all logs and other upload tasks'],
            'recovery': 'No local backup of disposable HTTP cache; future installation may redownload packages.'}
    if not args.execute_reviewed:
        print(json.dumps({key: value for key, value in plan.items() if key != 'files'}, indent=2))
        return
    # 先持久化完整白名单，再逐个核对未变化并unlink；不递归删除目录，不触及其他缓存。
    with RECEIPT.open('x', encoding='utf-8') as receipt:
        def emit(value):
            receipt.write(json.dumps(value, ensure_ascii=False) + '\n')
            receipt.flush()
            os.fsync(receipt.fileno())
        emit(plan)
        for item in files:
            path = Path(item['path'])
            if snapshot(path) != item:
                emit({'event': 'stopped_cache_changed', 'path': str(path)})
                raise RuntimeError('A cache file changed during cleanup; stop')
            if not any(directory in path.resolve().parents for directory in TARGETS):
                raise RuntimeError('Deletion target escaped approved roots')
            path.unlink()
            emit({'event': 'removed', 'path': str(path), 'size': item['size']})
        fs = os.statvfs(DEV)
        after = fs.f_bavail * fs.f_frsize
        result = {'event': 'completed', 'removed_files': len(files), 'free_bytes_after': after,
                  'observed_free_bytes_increase': after - before,
                  'all_model_files_preserved': True, 'cache_directories_left_in_place': True}
        emit(result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
