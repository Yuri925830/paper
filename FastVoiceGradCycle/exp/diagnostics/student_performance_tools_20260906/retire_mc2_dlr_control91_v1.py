"""Fixed control S + DEV90 retirement; readonly by default, no backup/model execution."""
import argparse
import ast
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import sys
from types import SimpleNamespace

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
OLD_FILE = TOOLS / 'retire_mc2_full3447_v1.py'
OLD_SHA = 'bd6440c6018f9d8130e6717ced8ee16e88c6b08a3c28765d829edfbe64980a8a'
DRIVER = TOOLS / 'run_acdd_mc2_dlr_paired120_from3927_v1.py'
DRIVER_SHA = 'e2eb8d53f18999874dab447d1b124b6c34bd0a0d056d0323df59b320e14ff821'
RUN = 'acdd_mc2_dlr_paired120_from3927_v1'
ARM = 'control_d2e4'
EVAL = 'evaluation_mc2_dlr_d2e4_paired120_endpoint_v1'
LABEL = 'student_mc2_dlr_d2e4_paired120_endpoint_v1'
AUDIT = 'mc2_dlr_control91_retirement_v1'
PROOF = 'mc2_dlr_control91_private_backup_receipt_v1.json'
PRIVATE = PureWindowsPath('C:/Users/rich_/AppData/Local/VoiceGradPrivateBackups/20260907/' + AUDIT)
METRIC_SHA = 'c0ff0e76a25b5656f1b2c110a2a6664342b03d8cf244971a8590074c75a37d86'
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Only the registered rental may execute')
if hashlib.sha256(OLD_FILE.read_bytes()).hexdigest() != OLD_SHA:
    raise RuntimeError('Frozen retirement primitive changed')
_spec = importlib.util.spec_from_file_location('_control91_old_guards', OLD_FILE)
old = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(old)
r, LOCK_NAMES = old.r, old.LOCK_NAMES


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def frozen_archive_validator(dev):
    # Compile only this reviewed pure function, never driver imports/main/model code.
    if r.sha(DRIVER) != DRIVER_SHA: raise RuntimeError('Frozen candidate gate changed')
    nodes = [n for n in ast.parse(DRIVER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'validate_control_archive']
    if len(nodes) != 1: raise RuntimeError('Require exact single archive validator')
    namespace = dict(Path=Path, PureWindowsPath=PureWindowsPath, training=SimpleNamespace(canonical_json_sha256=canonical),
        OUTPUT=dev / RUN, ARMS=(ARM, 'candidate_d1e4'), METRIC_IDENTITY_SHA=METRIC_SHA,
        EVALUATION={ARM: {'output_root': str(dev / EVAL), 'label': LABEL}})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DRIVER), 'exec'), namespace)
    return namespace['validate_control_archive']


def production_spec(source_sha, proof_sha):
    return dict(root=ROOT, dev=DEV, run=DEV / RUN, evaluation=DEV / EVAL,
        audit=DEV / AUDIT, proof=DEV / PROOF, private=PRIVATE,
        pub=Path('/root/teacher-hifigan-publication-20260906'), publication_counts=(218, 4),
        protected=dict(old.PROTECTED), source_sha=source_sha, proof_sha=proof_sha)


def paths_for(s):
    system = s['evaluation'] / LABEL
    return {'joint_contract_file_sha256': s['run'] / 'run_contract.json',
        'control_completion_file_sha256': s['run'] / ARM / 'completion.json',
        'infer_completion_file_sha256': system / 'inference_complete.json',
        'evaluation_summary_file_sha256': system / 'metrics/development_summary_cer_dnsmos_sva_utmos.json',
        'run_identity_sha256': s['evaluation'] / 'run_identity.json'}


def csv_rows(path):
    with path.open(newline='', encoding='utf8') as handle: return list(csv.DictReader(handle))


def validate_proof(s):
    if r.sha(s['proof']) != s['proof_sha']: raise RuntimeError('Wrong backup proof SHA')
    proof = json.loads(s['proof'].read_text())
    if (proof.get('status'), type(proof.get('scp_exit_code')), proof.get('scp_exit_code')) != ('verified_private_backup', int, 0):
        raise RuntimeError('Private copy did not finish successfully')
    if any(proof.get(k) is not True for k in ('all_local_and_remote_sha256_verified', 'local_backups_read_only', 'remote_files_not_yet_removed')):
        raise RuntimeError('Private bytes/read-only verification incomplete')
    paths = paths_for(s)
    if set(proof['identity_sha256']) != {str(p) for p in paths.values()}: raise RuntimeError('Wrong five endpoint identity files')
    for path in paths.values():
        if r.sha(path) != proof['identity_sha256'][str(path)]: raise RuntimeError('Endpoint identity changed')
    joint, control, inference, summary, identity = [json.loads(p.read_text()) for p in paths.values()]
    step = control['state']['completed_global_step']
    target = s['run'] / ARM / ('student_step_%06d.pt' % step)
    system = s['evaluation'] / LABEL
    expected = {str(target): control['state']['student_snapshot']['sha256']}
    expected.update({str(system / 'wavs' / ('pair_%04d.wav' % i)): inference['wav_sha256'][str(i)] for i in range(90)})
    rows = proof['target_files']
    if len(rows) != 91 or len({row['remote_path'] for row in rows}) != 91 or set(expected) != {row['remote_path'] for row in rows}:
        raise RuntimeError('Backup requires exactly the endpoint S and90 fixed WAVs')
    records, specs = [], []
    for row in sorted(rows, key=lambda x: x['remote_path']):
        path = Path(row['remote_path']); size = row['size_bytes']
        local = s['private'] / (path.name if path == target else 'wavs/' + path.name)
        if (row['local_path'].replace('\\', '/') != local.as_posix() or row['sha256'] != expected[str(path)]
                or row['local_sha256'] != row['sha256'] or type(size) is not int or size <= 0
                or type(row['local_size_bytes']) is not int or row['local_size_bytes'] != size
                or type(row['allocated_bytes_remote']) is not int or row['allocated_bytes_remote'] <= 0
                or row.get('local_read_only') is not True or row.get('source_identity_verified_after_copy') is not True
                or type(row.get('source_nlink')) is not int or row['source_nlink'] != 1
                or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None):
            raise RuntimeError('Private path/bytes/read-only/link proof mismatch')
        saved = row['source_stat_decimal_strings']
        if (not isinstance(saved, list) or len(saved) != 5 or any(not isinstance(v, str) or re.fullmatch('0|[1-9][0-9]*', v) is None for v in saved)
                or not isinstance(row['source_mode'], str) or re.fullmatch('0o[0-7]{3}', row['source_mode']) is None
                or (path == target and row['source_mode'] != '0o444')):
            raise RuntimeError('Require exact decimal stat and original permission fields')
        specs.append(dict(path=path, sha256=row['sha256'], size=size, allocated=row['allocated_bytes_remote'],
                          mode=int(row['source_mode'], 8), stamp=[int(v) for v in saved]))
        records.append(dict(original_remote_path=str(path), sha256=row['sha256'], size_bytes=size,
            private_backup_absolute_path=local.as_posix(), backup_sha256=row['local_sha256'], backup_size_bytes=row['local_size_bytes'], remote_removed=True))
    record = dict(schema_version=1, status='completed_private_archive_and_remote_removal',
        joint_contract_canonical_sha256=canonical(joint), control_global_step=step, records=records,
        **{field: proof['identity_sha256'][str(path)] for field, path in paths.items() if field != 'run_identity_sha256'})
    validator = frozen_archive_validator(s['dev'])
    validator(record, joint, record['joint_contract_file_sha256'], control, inference, summary, exists=lambda p: False)
    if any((s['run'] / ARM / n).exists() for n in ('failure.json', 'cleanup_failure.json')): raise RuntimeError('Control technical failure')
    if (inference['status'], inference['system'], summary['system'], identity['purpose'], identity['pairs'],
            identity['candidate_training_contract'], identity['start_step'], identity['inference_batch_size']) != (
            'development_candidate_only', LABEL, LABEL, 'development_candidate_not_formal_8100', 90, str(s['run'] / 'run_contract.json'), 950, 1):
        raise RuntimeError('Wrong fixed evaluation setting')
    spec = dict(checkpoint=str(target), sha256=expected[str(target)], state_key='student', epoch=0, global_step=step, steps=1, sampling='existing_student_t950_to_zero')
    if inference['spec'] != spec or identity['systems'] != {LABEL: spec}: raise RuntimeError('Wrong student evaluation identity')
    if (summary['inference_complete_sha256'] != record['infer_completion_file_sha256']
            or summary['run_identity_sha256'] != inference['run_identity_sha256']
            or inference['run_identity_sha256'] != proof['identity_sha256'][str(paths['run_identity_sha256'])]):
        raise RuntimeError('Evaluation receipt chain changed')
    if identity['inputs'].get(str(target)) != expected[str(target)] or identity['inputs'].get(str(paths['joint_contract_file_sha256'])) != record['joint_contract_file_sha256']:
        raise RuntimeError('Evaluation lacks actual checkpoint/contract pins')
    original = csv_rows(s['dev'] / 'pairs.csv'); actual = csv_rows(system / 'pairs.csv')
    if (len(original) != 90 or {x['pair_id'] for x in original} != set(map(str, range(90)))
            or actual != [dict(x, output_wav=str(system / 'wavs' / ('pair_%04d.wav' % int(x['pair_id'])))) for x in original]
            or r.sha(system / 'pairs.csv') != inference['pairs_csv_sha256']): raise RuntimeError('Fixed pair CSV changed')
    if set(summary['metric_csv_sha256']) != {'cer', 'sva', 'dnsmos', 'utmos'}: raise RuntimeError('Missing metric table')
    for name, digest in summary['metric_csv_sha256'].items():
        path = system / 'metrics' / (name + '.csv'); table = csv_rows(path)
        if r.sha(path) != digest or len(table) != 90 or {x['pair_id'] for x in table} != set(map(str, range(90))): raise RuntimeError('Metric CSV changed')
        if name == 'cer' and sum(int(x['reference_chars']) for x in table) != 4392: raise RuntimeError('Wrong CER denominator')
        if name == 'sva' and any(float(x['threshold']) != .25 for x in table): raise RuntimeError('Wrong SVA threshold')
    steps = [json.loads(line)['global_step'] for line in (s['run'] / ARM / 'step_losses.jsonl').read_text().splitlines()]
    if steps != list(range(3928, step + 1)): raise RuntimeError('Incomplete original training records')
    pins = dict(s['protected'])
    for source in (joint['runtime_files'], identity['inputs']):
        for name, digest in source.items():
            if name in expected: continue  # 唯一91个移除目标；不继承历史祖先资产。
            if name in pins and pins[name] != digest: raise RuntimeError('Conflicting retained runtime pin')
            pins[name] = digest
    pins.update({str(OLD_FILE): OLD_SHA, str(old.GUARD_FILE): old.GUARD_SHA, str(DRIVER): DRIVER_SHA,
                 str(Path(__file__).resolve()): s['source_sha'], str(s['proof']): s['proof_sha']})
    if set(expected) & set(pins): raise RuntimeError('Deletion overlaps a protected asset')
    return proof, record, specs, pins, (joint, control, inference, summary)


def target_check(row):
    path = row['path']; info = r.regular(path)
    if (r.stamp(path) != row['stamp'] or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != row['mode']
            or info.st_size != row['size'] or info.st_blocks * 512 != row['allocated'] or r.sha(path) != row['sha256']
            or r.stamp(path) != row['stamp']): raise RuntimeError('Individual target changed: ' + str(path))


def metadata_snapshot(s):
    result = r.metadata_snapshot(s['run'])
    for parent, dirs, files in os.walk(s['evaluation'], followlinks=False):
        dirs[:] = [name for name in dirs if name != 'wavs']
        for name in dirs: r.regular(Path(parent) / name, True)
        for name in files:
            path = Path(parent) / name; result[str(path)] = r.sha(path)
    for suffix in ('.infer.log', '.metrics.log'):
        path = s['evaluation'].with_name(s['evaluation'].name + suffix)
        if path.exists() or path.is_symlink(): result[str(path)] = r.sha(path)
    return result


def asset_snapshot(pins):
    result = {}
    for name, digest in pins.items():
        path = Path(name); before = r.stamp(path)
        if r.sha(path) != digest or r.stamp(path) != before: raise RuntimeError('Protected runtime changed: ' + name)
        result[name] = {'sha256': digest, 'stat': before}
    return result


def guards(s, specs):
    targets = [row['path'] for row in specs]; inodes = {(row['stamp'][0], row['stamp'][1]) for row in specs}
    gpu = old.gpu_guard(); publication = r.publication_guard(s['pub'], targets)
    if (publication['documents'], publication['plans']) != s['publication_counts']: raise RuntimeError('Publication scope changed')
    return dict(gpu=gpu, publication=publication, processes=r.process_guard(targets, inodes))


def unlink_one(row, journal, removed):
    target_check(row); path = row['path']; parent = r.regular(path.parent, True)
    fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); info = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        if ((opened.st_dev, opened.st_ino) != (parent.st_dev, parent.st_ino)
                or [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns] != row['stamp']
                or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != row['mode']):
            raise RuntimeError('Opened directory target mismatch')
        old.journal_row(journal, dict(phase='prepared', path=str(path), sha256=row['sha256'], stat=row['stamp']))
        os.unlink(path.name, dir_fd=fd); removed.append(str(path))
        os.fsync(fd); old.journal_row(journal, dict(phase='removed', path=str(path)))
    finally: os.close(fd)


def publish_archive(s, record):
    path = s['run'] / 'control_archive_receipt.json'; prepared = s['audit'] / 'control_archive.prepared.json'
    published = False
    try:
        if path.exists() or path.is_symlink(): raise RuntimeError('Archive receipt already exists')
        r.write_new(prepared, record); os.rename(prepared, path); published = True
        fd = os.open(str(s['run']), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)
    except BaseException:
        if published: os.rename(path, s['audit'] / 'control_archive.invalidated.json')
        raise


def execute(s, apply=False):
    if (s['dev'] != s['root'] / 'exp/diagnostics/student_development_20260906' or s['run'] != s['dev'] / RUN
            or s['evaluation'] != s['dev'] / EVAL or s['audit'] != s['dev'] / AUDIT or s['proof'] != s['dev'] / PROOF):
        raise RuntimeError('Fixed directory scope changed')
    for directory in (s['dev'], s['run'], s['evaluation']): r.regular(directory, True)
    if any(p.exists() or p.is_symlink() for p in (s['audit'], s['run'] / 'control_archive_receipt.json', s['run'] / 'candidate_d1e4')):
        raise RuntimeError('Audit/archive/candidate exists; never repeat or race a retirement')
    if r.sha(Path(__file__).resolve()) != s['source_sha']: raise RuntimeError('Wrong executing source SHA')
    handles, removed, failure, owned, result = [], [], None, False, None
    try:
        if apply: s['audit'].mkdir(); owned = True
        for name in LOCK_NAMES:
            path = s['root'] / 'exp/locks' / name; r.regular(path)
            handle = path.open('r+'); handles.append(handle); fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof, record, specs, pins, payloads = validate_proof(s)
        for row in specs: target_check(row)
        assets = asset_snapshot(pins); metadata = metadata_snapshot(s); before = guards(s, specs)
        if metadata_snapshot(s) != metadata: raise RuntimeError('Metadata changed during preflight')
        result = dict(status='validated_not_deleted', backup_proof_sha256=s['proof_sha'], executing_source_sha256=s['source_sha'],
            exact_targets=[str(x['path']) for x in specs], target_stats={str(x['path']): x['stamp'] for x in specs},
            private_backup_proof=proof, protected_assets=assets, protected_metadata_sha256=metadata,
            dependency_guard_before=before, logical_bytes=sum(x['size'] for x in specs), allocated_bytes=sum(x['allocated'] for x in specs),
            old_metadata_unchanged=True, restore_before_historical_full_revalidation=True)
        if apply:
            r.write_new(s['audit'] / 'preflight.json', result)
            fresh = guards(s, specs)
            if fresh['publication'] != before['publication']: raise RuntimeError('Publication changed before unlink')
            if asset_snapshot(pins) != assets: raise RuntimeError('Protected runtime changed before unlink')
            targets = [x['path'] for x in specs]; inodes = {(x['stamp'][0], x['stamp'][1]) for x in specs}
            with (s['audit'] / 'removed_files.jsonl').open('x', encoding='utf8') as journal:
                for row in specs:
                    r.process_guard(targets, inodes)
                    if r.sha(s['proof']) != s['proof_sha'] or metadata_snapshot(s) != metadata:
                        raise RuntimeError('Protected metadata/proof changed before unlink')
                    unlink_one(row, journal, removed)
            after = guards(s, specs)
            if after['publication'] != before['publication'] or metadata_snapshot(s) != metadata or asset_snapshot(pins) != assets:
                raise RuntimeError('Protected evidence changed after unlink')
            if any(p.exists() or p.is_symlink() for p in targets): raise RuntimeError('Target remains or reappeared')
            frozen_archive_validator(s['dev'])(record, payloads[0], record['joint_contract_file_sha256'], *payloads[1:])
            result.update(status='control91_retired_private_backup_verified', removed_files=list(removed), removed_count=len(removed), dependency_guard_after=after)
            record.update(backup_proof_file_sha256=s['proof_sha'], retirement_source_sha256=s['source_sha'], retirement_audit=str(s['audit']),
                          local_backup_bytes_verified_by='Windows main agent; not directly read by rental', old_metadata_unchanged=True)
    except BaseException as error: failure = error
    finally:
        for handle in reversed(handles):
            try: handle.close()
            except BaseException as error:
                if failure is None: failure = error
    if failure is None and apply:
        try:
            r.write_completion(s['audit'], result)
            publish_archive(s, record)  # Last success publication, only after all cleanup.
        except BaseException as error:
            failure = error
            if (s['audit'] / 'completion.json').exists():
                try: os.rename(s['audit'] / 'completion.json', s['audit'] / 'completion.invalidated.json')
                except BaseException as secondary: print('Completion invalidation failed: ' + repr(secondary), flush=True)
    if failure is not None:
        report = dict(status='retirement_failed', error=repr(failure), removed_files=removed, removed_count=len(removed),
                      recovery_requires_private_backup=bool(removed), candidate_must_not_start=True)
        if owned:
            try: r.write_new(s['audit'] / 'failure.json', report)
            except BaseException as secondary: print('Failure journal failed: ' + repr(secondary), flush=True)
        print(json.dumps(report), flush=True); raise failure
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--backup-receipt-sha256', required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if Path.cwd() != ROOT: raise RuntimeError('Only the registered rental project cwd')
    result = execute(production_spec(args.source_sha256, args.backup_receipt_sha256), args.apply)
    print(json.dumps({k: result[k] for k in ('status', 'logical_bytes', 'allocated_bytes', 'exact_targets')}), flush=True)


if __name__ == '__main__': main()
