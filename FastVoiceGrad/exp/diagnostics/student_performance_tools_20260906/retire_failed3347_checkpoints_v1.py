"""只归档删除两个已有私有备份的失败3347权重；默认只读核查。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
RUN = DEV / 'acdd_timestep_paired60_from3327_v1'
PUB = Path('/root/teacher-hifigan-publication-20260906')
PROOF = DEV / 'failed3347_private_backup_receipt_v1.json'
AUDIT = DEV / 'failed3347_checkpoint_retirement_v1'
PRIVATE = 'C:/Users/rich_/AppData/Local/VoiceGradPrivateBackups/20260907/acdd_timestep_paired60_from3327_v1/'
SPECS = (
    (RUN / 'primary_stratified/final_full.pt',
     'a98832c436e8205d20a6f2b07baef8b0649de673dd8b7ff7c248725539036693', 1424630405),
    (RUN / 'primary_stratified/student_step_003347.pt',
     '237ad8f40a9cf95ad9d9e858356d32de2d2344540fe3fb6ed2c7a4729fe93b35', 309266135),
)
LOCK_NAMES = ('text_unseen_custom_hifigan_pipeline.lock',
              'text_unseen_custom_hifigan_train.lock', 'text_unseen_fastvoicegrad_student.lock')
# 真实只读审计发现的唯一孤立旧清单；这些SHA来自本次核验，绝不冒称旧计划声明。
AUDITED_ORPHAN = {
    'raw_basename': 'p2-preprocess-38347e93fd8f.files-from-raw.txt',
    'metadata_basename': 'p2-preprocess-38347e93fd8f.metadata.jsonl',
    'source_root': '/root/FsatVoiceGrad/preprocess',
    'raw_sha256': 'c0648573b56712f10d2c6b2b7bd748d28a7693c69f727502f06f24388faef671',
    'metadata_sha256': 'd57a44c3a0395d39624271a85bcd5723ec4874aac05e8fe427d139c311d332d9',
    'relative_paths': ('02_build_fastvoicegrad_split.py', '__pycache__/02_build_fastvoicegrad_split.cpython-38.pyc'),
    'current_raw_basename': 'p1-preprocess-cf049bdc244a.files-from-raw.txt',
    'current_metadata_basename': 'p1-preprocess-cf049bdc244a.metadata.jsonl',
    'current_raw_sha256': 'a2316413a5782b684b96f082defe1cf690af1deaf86900822394b0c343bccaad',
    'current_metadata_sha256': '7c86c3aa9026efca84a86ab1d65931c710b801bd23c5b26bb7ac66f606c481e0',
}


def regular(path, directory=False):
    # 逐级拒绝软链，不允许文件或父目录把删除目标重定向。
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path,) + tuple(path.parents)):
        raise RuntimeError('Nonabsolute/symlink path: ' + str(path))
    info = path.stat()
    wanted = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not wanted:
        raise RuntimeError('Wrong file type: ' + str(path))
    return info


def stamp(path):
    info = regular(path)
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def sha(path):
    regular(path)
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(path, parent):
    # Python3.8不提供Path.is_relative_to，使用明确的父目录集合。
    return path == parent or parent in path.parents


def strings(value):
    # file_sha256字典的路径在键中，因此键和值都参与引用检查。
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def reject_references(values, targets, location):
    for value in values:
        if any(str(path) in value for path in targets):
            raise RuntimeError('External checkpoint reference: ' + str(location))


def publication_paths(pub):
    # 枚举全部lists与所有实际上传计划，文件数量不作旧快照式截断。
    lists = pub / 'direct-upload-runtime/lists'
    regular(lists, True)
    listed = []
    for parent, dirs, files in os.walk(lists, followlinks=False):
        for name in dirs:
            regular(Path(parent) / name, True)
        for name in files:
            path = Path(parent) / name
            regular(path)
            listed.append(path)
    plans = set()
    for path in pub.iterdir():
        if 'plan' in path.name.lower() and path.suffix.lower() == '.json':
            regular(path)
            plans.add(path)
    runtime = pub / 'direct-upload-runtime'
    for parent, dirs, files in os.walk(runtime, followlinks=False):
        # 日志和收据不承担计划的角色；仍检查目录链，拒绝软链隐藏计划。
        for name in dirs:
            regular(Path(parent) / name, True)
        for name in files:
            if 'plan' in name.lower() and name.lower().endswith('.json'):
                path = Path(parent) / name
                regular(path)
                plans.add(path)
    if not listed or not plans:
        raise RuntimeError('Missing publication lists/plans')
    return sorted(set(listed)), sorted(plans)


def audited_orphan_mapping(pub, listed, mappings, declared_sha):
    # 旧组仍完整扫描；只在双SHA、逐行映射及现行p1覆盖全部吻合后补上source_root。
    spec = AUDITED_ORPHAN
    folder = pub / 'direct-upload-runtime/lists'
    raw = folder / spec['raw_basename']
    metadata = folder / spec['metadata_basename']
    if raw not in listed and metadata not in listed:
        return []  # CPU临时夹具没有生产旧清单；真实清单枚举不会删掉它来绕过校验。
    current_raw = folder / spec['current_raw_basename']
    current_metadata = folder / spec['current_metadata_basename']
    if any(path not in listed for path in (raw, metadata, current_raw, current_metadata)):
        raise RuntimeError('Audited orphan or current coverage document missing')
    root = PurePosixPath(spec['source_root'])
    if raw in mappings or current_raw not in mappings or mappings[current_raw] != root:
        raise RuntimeError('Audited orphan/current source_root status changed')
    pins = {raw: spec['raw_sha256'], metadata: spec['metadata_sha256'],
            current_raw: spec['current_raw_sha256'], current_metadata: spec['current_metadata_sha256']}
    for path, expected in pins.items():
        if sha(path) != expected:
            raise RuntimeError('Audited orphan/current SHA changed: ' + str(path))
    if (declared_sha.get(current_raw), declared_sha.get(current_metadata)) != (pins[current_raw], pins[current_metadata]):
        raise RuntimeError('Current plan no longer declares the audited p1 coverage hashes')
    relative_paths = list(spec['relative_paths'])
    if raw.read_text().splitlines() != relative_paths:
        raise RuntimeError('Audited orphan exact two relative paths changed')
    old_rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    if len(old_rows) != len(relative_paths):
        raise RuntimeError('Audited orphan metadata row count differs')
    for relative, row in zip(relative_paths, old_rows):
        if row.get('relative_path') != relative or row.get('source_path') != str(root / relative):
            raise RuntimeError('Audited orphan row/source_root mapping differs')
    current_paths = current_raw.read_text().splitlines()
    current_rows = [json.loads(line) for line in current_metadata.read_text().splitlines()]
    if len(set(current_paths)) != len(current_paths) or [row.get('relative_path') for row in current_rows] != current_paths:
        raise RuntimeError('Current p1 raw/metadata ordering or uniqueness differs')
    by_relative = {row['relative_path']: row for row in current_rows}
    for relative in relative_paths:
        if relative not in current_paths or by_relative[relative].get('source_path') != str(root / relative):
            raise RuntimeError('Current plan does not cover both audited orphan sources')
    # 补充pin只作用于已审旧pair；现行计划的mapping/SHA原样保留。
    mappings[raw] = root
    declared_sha[raw] = pins[raw]
    declared_sha[metadata] = pins[metadata]
    return [{'raw_path': str(raw), 'metadata_path': str(metadata), 'source_root': str(root),
             'raw_sha256': pins[raw], 'metadata_sha256': pins[metadata],
             'relative_paths': relative_paths, 'current_plan_raw_path': str(current_raw),
             'current_plan_metadata_path': str(current_metadata),
             'binding_basis': 'explicit_readonly_audit_pins_and_current_plan_same_source_coverage',
             'old_plan_hash_claimed': False}]


def publication_guard(pub, targets):
    listed, plans = publication_paths(pub)
    paths = listed + plans
    before = {str(path): stamp(path) for path in paths}
    mappings = {}
    declared_sha = {}

    def visit(value):
        if isinstance(value, dict):
            if 'file_list_path' in value:
                root = PurePosixPath(value['source_root'])
                listing = Path(value['file_list_path'])
                manifest = Path(value['files_manifest_path'])
                if not root.is_absolute() or '..' in root.parts or listing not in listed or manifest not in listed:
                    raise RuntimeError('Unverifiable publication source_root/list mapping')
                if listing in mappings and mappings[listing] != root:
                    raise RuntimeError('Conflicting publication roots')
                mappings[listing] = root
                for path, key in ((listing, 'file_list_sha256'), (manifest, 'files_manifest_sha256')):
                    expected = value[key]
                    if path in declared_sha and declared_sha[path] != expected:
                        raise RuntimeError('Conflicting declared publication SHA')
                    declared_sha[path] = expected
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in plans:
        value = json.loads(path.read_text())
        reject_references(strings(value), targets, path)
        visit(value)
    supplemental = audited_orphan_mapping(pub, listed, mappings, declared_sha)
    line_count = metadata_count = 0
    hashes = {}
    for path in listed:
        digest = hashlib.sha256()
        is_raw = path.name.endswith('.files-from-raw.txt')
        is_metadata = path.name.endswith('.metadata.jsonl')
        if is_raw and path not in mappings:
            raise RuntimeError('Unmapped files-from list: ' + str(path))
        if not (is_raw or is_metadata):
            raise RuntimeError('Unknown publication list format: ' + str(path))
        with path.open('rb') as handle:
            for raw in handle:
                digest.update(raw)
                line_count += 1
                line = raw.decode('utf-8').rstrip('\r\n')
                if not line:
                    raise RuntimeError('Blank publication list record')
                reject_references((line,), targets, path)
                if is_metadata:
                    row = json.loads(line)
                    source = row['source_path']
                    if not PurePosixPath(source).is_absolute():
                        raise RuntimeError('Nonabsolute structured source_path')
                    reject_references(strings(row), targets, path)
                    metadata_count += 1
                else:
                    relative = PurePosixPath(line)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise RuntimeError('Escaping files-from relative path')
                    resolved = str(mappings[path] / relative)
                    reject_references((resolved,), targets, path)
        hashes[str(path)] = digest.hexdigest()
        if path not in declared_sha or hashes[str(path)] != declared_sha[path]:
            raise RuntimeError('Unpinned/changed publication list: ' + str(path))
    for path in plans:
        hashes[str(path)] = sha(path)
    after_lists, after_plans = publication_paths(pub)
    after = {str(path): stamp(path) for path in after_lists + after_plans}
    if before != after:
        raise RuntimeError('Publication documents changed during audit')
    return {'documents': len(listed), 'plans': len(plans), 'lines': line_count,
            'metadata_records': metadata_count, 'file_sha256': hashes, 'snapshot': before,
            'audited_orphan_supplemental_mappings': supplemental}


def metadata_snapshot(run):
    # 权重以外的所有原实验文件都保留；包含日志，不仅是选中的几个收据。
    result = {}
    regular(run, True)
    for parent, dirs, files in os.walk(run, followlinks=False):
        for name in dirs:
            regular(Path(parent) / name, True)
        for name in files:
            path = Path(parent) / name
            regular(path)
            if path.suffix != '.pt':
                result[str(path)] = sha(path)
    log = run.with_suffix('.log')
    if log.exists() or log.is_symlink():
        result[str(log)] = sha(log)
    if not result:
        raise RuntimeError('Missing original experiment metadata')
    return result


def contract_guard(root, run, proof, audit, targets):
    checked = {}
    allowed = []
    for parent, dirs, files in os.walk(root / 'exp', followlinks=False):
        dirs[:] = [name for name in dirs if name not in ('wav', 'staging', '.git', '__pycache__', '.pytest_cache')]
        for name in dirs:
            regular(Path(parent) / name, True)
        for name in files:
            path = Path(parent) / name
            if inside(path, audit) or path == proof:
                continue
            if path.suffix == '.json' and any(word in name.lower() for word in ('contract', 'config', 'completion', 'identity', 'manifest', 'summary', 'receipt')):
                regular(path)
                data = json.loads(path.read_text())
                checked[str(path)] = sha(path)
                hits = [value for value in strings(data) if any(str(target) in value for target in targets)]
                if hits:
                    if inside(path, run):
                        allowed.append(str(path))
                    else:
                        raise RuntimeError('External experiment dependency: ' + str(path))
    return {'file_sha256': checked, 'allowed_historical_self_references': allowed}


def process_guard(targets, inodes, proc_root=Path('/proc')):
    checked = 0
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        checked += 1
        try:
            for item in ('cmdline', 'maps'):
                value = (proc / item).read_bytes()
                if any(str(path).encode() in value for path in targets):
                    raise RuntimeError('Process text references retired checkpoint: ' + proc.name)
            for item in ('cwd', 'exe'):
                value = os.readlink(proc / item)
                if any(value == str(path) or value == str(path.parent) for path in targets):
                    raise RuntimeError('Process cwd/executable uses target: ' + proc.name)
            for fd in (proc / 'fd').iterdir():
                try:
                    value = os.readlink(fd)
                    info = fd.stat()
                except FileNotFoundError:
                    continue  # 一个已关闭的fd没有继续占用文件。
                if any(str(path) in value for path in targets) or (info.st_dev, info.st_ino) in inodes:
                    raise RuntimeError('Process holds checkpoint inode: ' + str(fd))
        except (FileNotFoundError, ProcessLookupError):
            # 只容许整个PID消失；仍活着但无法完整检查的进程拒绝放行。
            if proc.exists():
                # 内核线程/zombie没有cwd/exe，并且空cmdline、空maps和空fd均可核实。
                if (proc / 'cmdline').read_bytes() or (proc / 'maps').read_bytes() or list((proc / 'fd').iterdir()):
                    raise RuntimeError('Live process could not be completely inspected: ' + proc.name)
        # PermissionError等其余读取错误保持硬失败，不能把未知当作无引用。
    return {'processes_checked': checked, 'references': 0}


def validate_backup(proof, expected_sha, specs, private):
    if len(expected_sha) != 64 or sha(proof) != expected_sha:
        raise RuntimeError('Wrong private backup receipt SHA')
    value = json.loads(proof.read_text())
    if value.get('status') != 'verified_private_backup' or value.get('all_local_and_remote_sha256_verified') is not True:
        raise RuntimeError('Private backup not fully verified')
    rows = value['target_files']
    if len(rows) != 2 or len({row['remote_path'] for row in rows}) != 2:
        raise RuntimeError('Backup must contain exactly two distinct target files')
    by_path = {row['remote_path']: row for row in rows}
    if set(by_path) != {str(path) for path, _, _ in specs}:
        raise RuntimeError('Backup scope differs from fixed targets')
    for path, expected, size in specs:
        row = by_path[str(path)]
        if (row.get('sha256'), row.get('local_sha256'), row.get('size_bytes'), row.get('local_path')) != (expected, expected, size, private + path.name):
            raise RuntimeError('Backup identity/size/final private filename mismatch')
    return value


def write_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def write_completion(audit, value):
    # 完成文件必须完整持久化后再发布；flush/rename后的故障不能留下貌似成功的完成文件。
    temporary = audit / 'completion.prepared.json'
    final = audit / 'completion.json'
    invalidated = audit / 'completion.invalidated.json'
    if any(path.exists() or path.is_symlink() for path in (temporary, final, invalidated)):
        raise RuntimeError('Completion path already exists')
    published = False
    try:
        write_new(temporary, value)
        os.rename(temporary, final)
        published = True
        directory_fd = os.open(str(audit), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if published:
            try:
                os.rename(final, invalidated)
            except BaseException as secondary:
                print('Completion invalidation failed; failure.json takes precedence: ' + repr(secondary), flush=True)
        raise


def dependency_guard(root, run, pub, proof, audit, specs, inodes):
    targets = [path for path, _, _ in specs]
    return {'publication': publication_guard(pub, targets),
            'contracts': contract_guard(root, run, proof, audit, targets),
            'processes': process_guard(targets, inodes)}


def execute(root, run, pub, proof, audit, specs, private, proof_sha, apply=False):
    # 可注入路径只供租用机临时目录CPU测试；CLI绝不接受目标目录或权重参数。
    if audit.exists() or audit.is_symlink():
        raise RuntimeError('Audit directory already exists; never repeat a partial retirement')
    if len(specs) != 2 or len({path for path, _, _ in specs}) != 2:
        raise RuntimeError('Retirement requires exactly two files')
    handles, removed = [], []
    failure = None
    result = None
    try:
        for name in LOCK_NAMES:
            path = root / 'exp/locks' / name
            regular(path)
            handle = path.open('r+')
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        backup = validate_backup(proof, proof_sha, specs, private)
        protected = metadata_snapshot(run)
        snapshots, allocation = {}, 0
        for path, expected, size in specs:
            if path.parent != run / 'primary_stratified' or path.suffix != '.pt':
                raise RuntimeError('Target escaped exact primary checkpoint directory')
            info = regular(path)
            before = stamp(path)
            if info.st_nlink != 1 or info.st_size != size or sha(path) != expected or stamp(path) != before:
                raise RuntimeError('Checkpoint bytes/link count/stat mismatch: ' + str(path))
            snapshots[str(path)] = before
            allocation += info.st_blocks * 512
        inodes = {(value[0], value[1]) for value in snapshots.values()}
        guard = dependency_guard(root, run, pub, proof, audit, specs, inodes)
        if metadata_snapshot(run) != protected or sha(proof) != proof_sha:
            raise RuntimeError('Protected metadata or private proof changed during preflight')
        for path, _, _ in specs:
            if stamp(path) != snapshots[str(path)]:
                raise RuntimeError('Target changed during preflight')
        result = {'status': 'validated_not_deleted', 'source_sha256': sha(Path(__file__).resolve()),
                  'backup_receipt_path': str(proof), 'backup_receipt_sha256': proof_sha,
                  'private_backup': backup, 'exact_target_files': [str(p) for p, _, _ in specs],
                  'logical_bytes': sum(size for _, _, size in specs), 'file_allocation_bytes': allocation,
                  'file_stat_before': snapshots, 'protected_metadata_sha256': protected,
                  'dependency_guard_before': guard, 'restore_required_for_historical_full_validation': True,
                  'old_failure_and_stop_decisions_unchanged': True}
        if apply:
            # apply时先再验依赖；新目录以mkdir独占，不覆盖任何原收据。
            fresh = dependency_guard(root, run, pub, proof, audit, specs, inodes)
            if fresh['publication'] != guard['publication'] or fresh['contracts'] != guard['contracts']:
                raise RuntimeError('Dependencies changed immediately before retirement')
            audit.mkdir()
            write_new(audit / 'preflight.json', result)
            with (audit / 'removed_files.jsonl').open('x', encoding='utf-8') as journal:
                for path, expected, _ in specs:
                    # 每个unlink前重新核进程及自身证据；只使用已验证父目录的文件描述符。
                    process_guard([p for p, _, _ in specs], inodes)
                    if metadata_snapshot(run) != protected or sha(proof) != proof_sha:
                        raise RuntimeError('Protected evidence changed before unlink')
                    if stamp(path) != snapshots[str(path)]:
                        raise RuntimeError('Individual target changed before unlink')
                    parent_info = regular(path.parent, True)
                    parent_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        if (os.fstat(parent_fd).st_dev, os.fstat(parent_fd).st_ino) != (parent_info.st_dev, parent_info.st_ino):
                            raise RuntimeError('Parent directory changed before unlink')
                        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                        if [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns] != snapshots[str(path)] or current.st_nlink != 1:
                            raise RuntimeError('Opened-directory target mismatch')
                        journal.write(json.dumps({'phase': 'prepared', 'path': str(path), 'sha256': expected}) + '\n')
                        journal.flush()
                        os.fsync(journal.fileno())
                        os.unlink(path.name, dir_fd=parent_fd)  # 只删除这一个已核验文件，不删除目录。
                        removed.append(str(path))  # 在后续fsync/收据失败前先记录真实删除游标。
                        os.fsync(parent_fd)
                        journal.write(json.dumps({'phase': 'removed', 'path': str(path)}) + '\n')
                        journal.flush()
                        os.fsync(journal.fileno())
                    finally:
                        os.close(parent_fd)
            after = dependency_guard(root, run, pub, proof, audit, specs, inodes)
            if after['publication'] != guard['publication'] or after['contracts'] != guard['contracts']:
                raise RuntimeError('External references changed during retirement; backup remains recoverable')
            if metadata_snapshot(run) != protected or sha(proof) != proof_sha:
                raise RuntimeError('Protected evidence changed after retirement')
            if any(path.exists() or path.is_symlink() for path, _, _ in specs):
                raise RuntimeError('An exact target unexpectedly remains/reappeared')
            result.update(status='two_failed3347_checkpoints_retired_private_backup_verified',
                          removed_files=removed, removed_count=len(removed), dependency_guard_after=after)
    except BaseException as error:
        failure = error
    finally:
        # 一个close失败也继续释放剩余锁；完成收据在全部close成功后才发布。
        for handle in reversed(handles):
            try:
                handle.close()
            except BaseException as error:
                if failure is None:
                    failure = error
    if failure is None and apply:
        try:
            write_completion(audit, result)
        except BaseException as error:
            failure = error
    if failure is not None:
        report = {'status': 'retirement_failed', 'error': repr(failure), 'removed_files': removed,
                  'removed_count': len(removed), 'backup_receipt_sha256': proof_sha,
                  'recovery_requires_private_backup_for_removed_files': bool(removed)}
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if apply and audit.exists():
            try:
                write_new(audit / 'failure.json', report)
            except BaseException as secondary:
                print('Failure receipt could not be written: ' + repr(secondary), flush=True)
        raise failure
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='在完整备份及全部门禁通过后只删除两个固定文件')
    parser.add_argument('--backup-receipt-sha256', required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not sys.platform.startswith('linux') or Path.cwd() != ROOT:
        raise RuntimeError('Only the named rented project root is allowed')
    result = execute(ROOT, RUN, PUB, PROOF, AUDIT, SPECS, PRIVATE, args.backup_receipt_sha256, args.apply)
    print(json.dumps({key: result[key] for key in ('status', 'logical_bytes', 'file_allocation_bytes', 'exact_target_files')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
