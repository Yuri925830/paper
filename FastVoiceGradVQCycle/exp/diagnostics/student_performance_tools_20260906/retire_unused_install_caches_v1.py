"""只清理已审的旧 VS Code 安装缓存、五份 VSIX 与一个孤立安装包；默认只读。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import retire_failed3347_checkpoints_v1 as guard

ROOT = Path('/root/FsatVoiceGrad')
PUB = Path('/root/teacher-hifigan-publication-20260906')
AUDIT = ROOT / 'exp/diagnostics/student_development_20260906/unused_install_cache_retirement_v1'
TREE = Path('/root/.vscode-server/cli/servers/Stable-a44adf7f53e00964ab890f9f8758a334f1fc15bc')
VSIX = Path('/root/.vscode-server/data/CachedExtensionVSIXs')
SINGLES = (
    (VSIX / 'ms-python.debugpy-2026.6.0-linux-x64', 13888374),
    (VSIX / 'ms-python.python-2026.4.0-linux-x64', 17823311),
    (VSIX / 'ms-python.vscode-pylance-2026.3.1', 29646770),
    (VSIX / 'ms-python.vscode-python-envs-1.36.0-linux-x64', 12859354),
    (VSIX / 'ms-ceintl.vscode-language-pack-zh-hans-1.131.2026090407', 686413),
    (Path('/tmp/pip-unpack-p7phbrze/llvmlite-0.41.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'), 34959360),
)
PROTECTED = (
    Path('/root/.vscode-server/cli/servers/lru.json'),
    Path('/root/.vscode-server/bin/125df4672b8a6a34975303c6b0baa124e560a4f7/package.json'),
) + tuple(Path('/root/.vscode-server/extensions') / path.name / 'package.json' for path, _ in SINGLES[:5])
LOCK_NAMES = ('text_unseen_custom_hifigan_pipeline.lock',
              'text_unseen_custom_hifigan_train.lock', 'text_unseen_fastvoicegrad_student.lock')


def digest(path):
    guard.regular(path)
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def enumerate_targets(tree, singles, expected_tree_count, expected_tree_size):
    # 先完整枚举固定树；任何软链、非常规文件或多重硬链接都拒绝。
    guard.regular(tree, True)
    directories, paths = [tree], []
    for parent, dirs, files in os.walk(tree, followlinks=False):
        for name in sorted(dirs):
            path = Path(parent) / name
            guard.regular(path, True)
            directories.append(path)
        for name in sorted(files):
            paths.append(Path(parent) / name)
    if len(paths) != expected_tree_count or sum(guard.regular(p).st_size for p in paths) != expected_tree_size:
        raise RuntimeError('Audited cache tree file count or size changed')
    for path, size in singles:
        if guard.regular(path).st_size != size:
            raise RuntimeError('Fixed installer size changed: ' + str(path))
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise RuntimeError('Overlapping fixed targets')
    records = {}
    for path in sorted(paths):
        info = guard.regular(path)
        if info.st_nlink != 1:
            raise RuntimeError('Multiple hard links: ' + str(path))
        records[str(path)] = {'stamp': guard.stamp(path), 'allocation_bytes': info.st_blocks * 512}
    directory_ids = {str(p): [guard.regular(p, True).st_dev, guard.regular(p, True).st_ino] for p in directories}
    return records, directory_ids


def protected_snapshot(paths):
    return {str(path): {'sha256': digest(path), 'stamp': guard.stamp(path)} for path in paths}


def metadata_paths(root, audit, environment_roots):
    # 固定扫描现行代码/配置、实验契约和安装来源；不会读取模型或音频。
    paths = set()
    for folder in [root / name for name in ('fastvoicegrad', 'voicegrad', 'scripts', 'configs', 'exp')]:
        if not folder.exists():
            continue
        guard.regular(folder, True)
        for parent, dirs, files in os.walk(folder, followlinks=False):
            dirs[:] = [name for name in dirs if name not in ('wav', 'staging', '.git', '__pycache__', '.pytest_cache') and Path(parent) / name != audit]
            for name in dirs:
                guard.regular(Path(parent) / name, True)
            for name in files:
                path = Path(parent) / name
                wanted = (path.suffix == '.json' and any(word in name.lower() for word in ('contract', 'config', 'completion', 'identity', 'manifest', 'summary', 'receipt'))
                          if folder == root / 'exp' else path.suffix in ('.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.txt'))
                if wanted:
                    guard.regular(path)
                    paths.add(path)
    # pip 来源记录只位于这些标准位置；不递归检查 terminfo 等无关环境内容。
    sites = {}
    for environments in environment_roots:
        if not environments.exists() and not environments.is_symlink():
            continue
        guard.regular(environments, True)
        for site in sorted(environments.glob('*/lib/python*/site-packages')):
            info = guard.regular(site, True)
            sites[str(site)] = [info.st_dev, info.st_ino]
            selected = list(site.glob('*.pth')) + list(site.glob('*.egg-link'))
            for distribution in site.glob('*.dist-info'):
                guard.regular(distribution, True)
                direct = distribution / 'direct_url.json'
                if direct.exists() or direct.is_symlink():
                    selected.append(direct)
            for path in selected:
                guard.regular(path)  # 文件及全部父级仍拒绝软链。
                paths.add(path)
    return sorted(paths), sites


def metadata_snapshot(root, audit, targets, environment_roots):
    result = {}
    paths, sites = metadata_paths(root, audit, environment_roots)
    for path in paths:
        before = guard.stamp(path)
        raw = path.read_bytes()
        value = raw.decode('utf-8')
        if path.suffix == '.json':
            value = json.dumps(json.loads(value), ensure_ascii=False)
        if any(str(target) in value for target in targets):
            raise RuntimeError('Current code/config/environment reference: ' + str(path))
        if guard.stamp(path) != before:
            raise RuntimeError('Current metadata changed while read')
        result[str(path)] = {'stamp': before, 'sha256': hashlib.sha256(raw).hexdigest()}
    return {'files': result, 'site_package_roots': sites}


def index_snapshot(paths):
    return {str(path): protected_snapshot((path,))[str(path)] if path.exists() or path.is_symlink() else {'absent': True} for path in paths}


def fsync_directory(path):
    info = guard.regular(path, True)
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        current = os.fstat(fd)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError('Directory inode changed')
        os.fsync(fd)
    finally:
        os.close(fd)


def execute(root, pub, audit, tree, singles, protected, tree_count, tree_size, guard_sha, apply=False, environment_roots=()):
    # 路径注入仅供 CPU 临时夹具；命令行没有任意目标参数。
    source = Path(__file__).resolve()
    dependency = Path(guard.__file__).resolve()
    if len(guard_sha) != 64 or digest(dependency) != guard_sha:
        raise RuntimeError('Dependency source SHA mismatch')
    source_sha = digest(source)
    if audit.exists() or audit.is_symlink():
        raise RuntimeError('Audit already exists; do not repeat a partial cleanup')
    guard.regular(audit.parent, True)
    handles, removed, removed_dirs = [], [], []
    failure = result = None
    try:
        for name in LOCK_NAMES:
            path = root / 'exp/locks' / name
            guard.regular(path)
            handle = path.open('r+')
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        records, directory_ids = enumerate_targets(tree, singles, tree_count, tree_size)
        roots = [tree] + [path for path, _ in singles]
        process_targets = roots + [Path(path) for path in directory_ids]
        inodes = {(row['stamp'][0], row['stamp'][1]) for row in records.values()}
        kept = protected_snapshot(protected)
        optional_indexes = tuple(singles[0][0].parent / name for name in ('index.json', 'extensions.json', '.obsolete'))
        indexes = index_snapshot(optional_indexes)
        metadata = metadata_snapshot(root, audit, roots, environment_roots)
        publication = guard.publication_guard(pub, roots)
        processes = guard.process_guard(process_targets, inodes)

        def unchanged_evidence(full_publication=True):
            if digest(source) != source_sha or digest(dependency) != guard_sha or protected_snapshot(protected) != kept or index_snapshot(optional_indexes) != indexes:
                raise RuntimeError('Protected index/current installations or source changed')
            if full_publication:
                if guard.publication_guard(pub, roots) != publication or metadata_snapshot(root, audit, roots, environment_roots) != metadata:
                    raise RuntimeError('Publication or current metadata references changed')
            else:
                listed, plans = guard.publication_paths(pub)
                if {str(path): guard.stamp(path) for path in listed + plans} != publication['snapshot']:
                    raise RuntimeError('Publication document set/stat changed')
                current_paths, current_sites = metadata_paths(root, audit, environment_roots)
                if current_sites != metadata['site_package_roots'] or {str(path): guard.stamp(path) for path in current_paths} != {name: row['stamp'] for name, row in metadata['files'].items()}:
                    raise RuntimeError('Current metadata set/stat changed')

        unchanged_evidence(full_publication=False)
        if enumerate_targets(tree, singles, tree_count, tree_size) != (records, directory_ids):
            raise RuntimeError('Fixed targets changed during preflight')
        result = {'status': 'validated_not_deleted', 'source_sha256': source_sha,
                  'dependency_sha256': guard_sha, 'fixed_tree': str(tree),
                  'fixed_single_files': [str(path) for path, _ in singles],
                  'file_count': len(records), 'logical_bytes': sum(row['stamp'][2] for row in records.values()),
                  'file_allocation_bytes': sum(row['allocation_bytes'] for row in records.values()),
                  'files_before': records, 'directories_before': directory_ids,
                  'protected_before': kept, 'publication_before': publication, 'processes_before': processes,
                  'current_metadata_before': metadata, 'optional_indexes_before': indexes,
                  'environment_metadata_roots': [str(path) for path in environment_roots],
                  'recovery': 'Install caches can be downloaded again; no project model/data/code is deleted.',
                  'index_left_unchanged': True}
        if apply:
            unchanged_evidence()
            guard.process_guard(process_targets, inodes)
            audit.mkdir()
            guard.write_new(audit / 'preflight.json', result)
            fsync_directory(audit)
            with (audit / 'removed.jsonl').open('x', encoding='utf-8') as journal:
                def record(value):
                    journal.write(json.dumps(value, ensure_ascii=False) + '\n')
                    journal.flush()
                    os.fsync(journal.fileno())

                for index, (name, row) in enumerate(records.items()):
                    path = Path(name)
                    if index % 256 == 0:
                        unchanged_evidence(full_publication=False)
                        guard.process_guard(process_targets, inodes)
                    # 每个文件先查实际占用和 inode；不会递归删除或跟随软链。
                    guard.process_guard([path], {(row['stamp'][0], row['stamp'][1])})
                    if guard.stamp(path) != row['stamp']:
                        raise RuntimeError('Individual installer changed: ' + name)
                    parent = guard.regular(path.parent, True)
                    fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        current_parent = os.fstat(fd)
                        if (parent.st_dev, parent.st_ino) != (current_parent.st_dev, current_parent.st_ino):
                            raise RuntimeError('Parent changed')
                        current = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                        actual = [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns]
                        if actual != row['stamp'] or current.st_nlink != 1:
                            raise RuntimeError('Opened-directory installer changed')
                        record({'phase': 'prepared', 'path': name})
                        os.unlink(path.name, dir_fd=fd)
                        removed.append(name)  # 先记录真实游标；后续 fsync 失败也不能少报。
                        os.fsync(fd)
                        record({'phase': 'removed', 'path': name})
                    finally:
                        os.close(fd)
                unchanged_evidence(full_publication=False)
                for name in sorted(directory_ids, key=lambda value: len(Path(value).parts), reverse=True):
                    path = Path(name)
                    guard.process_guard([path], set())
                    info = guard.regular(path, True)
                    if [info.st_dev, info.st_ino] != directory_ids[name]:
                        raise RuntimeError('Cache directory replaced')
                    parent = guard.regular(path.parent, True)
                    fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        current_parent = os.fstat(fd)
                        current = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                        if (parent.st_dev, parent.st_ino) != (current_parent.st_dev, current_parent.st_ino) or [current.st_dev, current.st_ino] != directory_ids[name]:
                            raise RuntimeError('Opened-directory identity changed')
                        record({'phase': 'directory_prepared', 'path': name})
                        os.rmdir(path.name, dir_fd=fd)  # 只移除已枚举且已空的旧版本目录。
                        removed_dirs.append(name)
                        os.fsync(fd)
                        record({'phase': 'directory_removed', 'path': name})
                    finally:
                        os.close(fd)
            unchanged_evidence()
            after_processes = guard.process_guard(process_targets, inodes)
            if any(Path(path).exists() or Path(path).is_symlink() for path in records) or tree.exists() or tree.is_symlink():
                raise RuntimeError('Deleted cache target remains or reappeared')
            result.update(status='unused_install_caches_removed', removed_files=removed,
                          removed_count=len(removed), removed_directories=removed_dirs,
                          processes_after=after_processes, protected_after=protected_snapshot(protected),
                          current_metadata_after=metadata, optional_indexes_after=index_snapshot(optional_indexes),
                          publication_after=publication)
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
        published = False
        try:
            guard.write_new(audit / 'completion.prepared.json', result)
            os.rename(audit / 'completion.prepared.json', audit / 'completion.json')
            published = True
            fsync_directory(audit)
        except BaseException as error:
            failure = error
            if published:
                try:
                    os.rename(audit / 'completion.json', audit / 'completion.invalidated.json')
                except BaseException as secondary:
                    print('Completion invalidation failed; failure.json takes precedence: ' + repr(secondary), flush=True)
    if failure is not None:
        report = {'status': 'cache_cleanup_failed', 'error': repr(failure), 'removed_files': removed,
                  'removed_count': len(removed), 'removed_directories': removed_dirs,
                  'recovery': 'Removed installation caches require re-download; current installations were protected.'}
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if apply and audit.exists():
            try:
                guard.write_new(audit / 'failure.json', report)
            except BaseException as secondary:
                print('Failure receipt write failed: ' + repr(secondary), flush=True)
        raise failure
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--guard-sha256', required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not sys.platform.startswith('linux') or Path.cwd() != ROOT:
        raise RuntimeError('Only the named rented project root is allowed')
    result = execute(ROOT, PUB, AUDIT, TREE, SINGLES, PROTECTED, 3061, 725793587, args.guard_sha256, args.apply,
                     (Path('/root/miniconda3/envs'),))
    print(json.dumps({key: result[key] for key in ('status', 'file_count', 'logical_bytes', 'file_allocation_bytes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
