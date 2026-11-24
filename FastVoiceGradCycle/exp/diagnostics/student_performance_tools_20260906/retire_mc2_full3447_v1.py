"""只退休已有私有验真备份的唯一3447 full；默认只读，不运行项目模型。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
GUARD_FILE = TOOLS / 'retire_failed3347_checkpoints_v1.py'
GUARD_SHA = '774698257e589c003dbbf846451386d01ad2d1ce36e6552c8b4c91e97bb2b7c9'
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Only the registered rental is allowed')
if hashlib.sha256(GUARD_FILE.read_bytes()).hexdigest() != GUARD_SHA:
    raise RuntimeError('Frozen retirement guards changed')
_spec = importlib.util.spec_from_file_location('_retire3447_frozen_guards', GUARD_FILE)
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)  # 只导入defs；不调用旧main/execute/contract_guard/SPECS。
LOCK_NAMES = r.LOCK_NAMES + ('text_unseen_fastvoicegrad_student_inference.lock',)
RUN_NAMES = ('acdd_mc2_paired120_from3327_v1', 'acdd_mc2_continuation480_from3447_v1',
             'acdd_mc2_fm_cross_3447_3927_v1')
PROTECTED = {
    str(DEV / RUN_NAMES[1] / 'final_full.pt'): '6a4f5beabcd3264e7aa016b3236f98da174af396b2ce7fb0c0b5b083f8aa2630',
    str(DEV / 'acdd_s5e5_staged120_from3227_v1/final_full.pt'): '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556',
    str(DEV / 'acdd_s5e5_staged120_from3227_v1/student_step_003327.pt'): '7bb68a0d053335a87a2c3c42684b864e27bf21a9f19d6f8d1b5bc8152955ca60',
    str(ROOT / 'exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt'): '1a076ac28692c347ac43daa44c932e9f5894588ffbcec7551ed2350f07f1aba6',
    str(ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/generator_step_2500000.pt'): 'dacd362c4db46cdf7bd8428ace6f38d7d03976c60a3cd4c55022d38383094fb9',
    str(ROOT / 'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/config.json'): '4b570fee140ffbda92bb217218792dca069b2d73627bc5e7bf7b1669869dd05a',
}
HISTORY_EXTRA = (
    'ACDD3447_MC2_CONTINUATION480_PROTOCOL_CN.md', 'MC2_CONTINUATION480_REVIEW_V1.json',
    'MC2_FM_CROSSED_3447_3927_DIAGNOSTIC_PLAN_CN.md',
    'mc2_storage_retirement_v1/preflight.json', 'mc2_storage_retirement_v1/completion.json',
)


def production_spec(source_sha):
    return dict(root=ROOT, dev=DEV, pub=Path('/root/teacher-hifigan-publication-20260906'),
        target=DEV / RUN_NAMES[0] / 'candidate_mc2/final_full.pt',
        target_sha='d108d1793a2473bb42f6b1e556c67b6339a04da8b9c8b39f0a7905bf5a26d4f6',
        logical=1424624517, allocated=1424625664,
        proof=DEV / 'mc2_full3447_private_backup_receipt_v1.json',
        proof_sha='25071f674f5991f8e96b21df9d3e16055604a84e7138a5d53aed4f7525c135b8',
        private='C:/Users/rich_/AppData/Local/VoiceGradPrivateBackups/20260907/mc2_full3447_retirement_v1/final_full.pt',
        audit=DEV / 'mc2_full3447_retirement_v1', protected=dict(PROTECTED),
        runs=[DEV / name for name in RUN_NAMES], history=[DEV / name for name in HISTORY_EXTRA],
        evaluation=DEV / 'evaluation_mc2_paired120_endpoint_v1', publication_counts=(218, 4), source_sha=source_sha)


def validate_backup(s):
    if r.sha(s['proof']) != s['proof_sha']:
        raise RuntimeError('Wrong private proof SHA')
    value = json.loads(s['proof'].read_text())
    if (value.get('status') != 'verified_private_backup' or type(value.get('scp_exit_code')) is not int or value['scp_exit_code'] != 0
            or any(value.get(key) is not True for key in ('all_local_and_remote_sha256_verified',
                   'local_backups_read_only', 'remote_files_not_yet_removed'))):
        raise RuntimeError('Private proof is not fully verified/read-only')
    rows = value.get('target_files')
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError('Private proof requires exactly one target')
    row = rows[0]
    expected = dict(remote_path=str(s['target']), local_path=s['private'], size_bytes=s['logical'],
        local_size_bytes=s['logical'], allocated_bytes_remote=s['allocated'], sha256=s['target_sha'],
        local_sha256=s['target_sha'], source_mode='0o444', source_nlink=1,
        local_read_only=True, source_identity_verified_after_copy=True)
    if any(type(row.get(key)) is not type(wanted) or row[key] != wanted for key, wanted in expected.items()):
        raise RuntimeError('Private proof target/path/size/hash/read-only mismatch')
    saved = row.get('source_stat_decimal_strings')
    if (not isinstance(saved, list) or len(saved) != 5
            or any(not isinstance(item, str) or re.fullmatch(r'0|[1-9][0-9]*', item) is None for item in saved)):
        raise RuntimeError('Private proof requires exact decimal stat strings')
    return value, [int(item) for item in saved]


def target_check(s, expected_stat, rehash=True):
    info = r.regular(s['target'])
    if (r.stamp(s['target']) != expected_stat or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444
            or info.st_size != s['logical'] or info.st_blocks * 512 != s['allocated']):
        raise RuntimeError('Target stat/mode/link/allocation changed')
    if rehash and r.sha(s['target']) != s['target_sha']:
        raise RuntimeError('Target SHA changed')
    if r.stamp(s['target']) != expected_stat:
        raise RuntimeError('Target changed while hashing')


def evidence_snapshot(s, rehash=True, previous=None):
    assets = dict(s['protected'])
    assets.update({str(GUARD_FILE): GUARD_SHA, str(Path(__file__).resolve()): s['source_sha'], str(s['proof']): s['proof_sha']})
    checked = {}
    for name, expected in assets.items():
        path = Path(name); before = r.stamp(path)
        digest = r.sha(path) if rehash else previous['assets'][name]['sha256']
        if digest != expected or r.stamp(path) != before:
            raise RuntimeError('Protected asset/source/proof SHA or stat changed: ' + name)
        checked[name] = {'sha256': digest, 'stat': before}
    metadata = {}
    for run in s['runs']: metadata.update(r.metadata_snapshot(run))
    for path in s['history']: metadata[str(path)] = r.sha(path)
    r.regular(s['evaluation'], True)
    for parent, dirs, files in os.walk(s['evaluation'], followlinks=False):
        dirs[:] = [name for name in dirs if name not in ('wav', 'wavs')]
        for name in dirs: r.regular(Path(parent) / name, True)
        for name in files:
            path = Path(parent) / name
            if path.suffix in ('.json', '.jsonl', '.csv', '.log', '.md'): metadata[str(path)] = r.sha(path)
    for suffix in ('.infer.log', '.metrics.log'):
        path = s['evaluation'].with_name(s['evaluation'].name + suffix)
        if path.exists() or path.is_symlink(): metadata[str(path)] = r.sha(path)
    references = [name for name in metadata if str(s['target']).encode() in Path(name).read_bytes()]
    return {'assets': checked, 'metadata_sha256': metadata, 'historical_reference_files': sorted(references)}


def historical_inventory(s, protected):
    # 发现旧18引用及新增说明；调度文档只登记，不硬钉RUN_QUEUE/CURRENT_FINDINGS。
    inventory = {}
    for parent, dirs, files in os.walk(s['dev'], followlinks=False):
        dirs[:] = [name for name in dirs if name not in ('wav', 'wavs', '__pycache__') and Path(parent) / name != s['audit']]
        for name in files:
            path = Path(parent) / name
            if path.suffix not in ('.json', '.md'): continue
            r.regular(path); value = path.read_bytes()
            if str(s['target']).encode() in value or s['target_sha'].encode() in value:
                inventory[str(path)] = {'sha256': hashlib.sha256(value).hexdigest(),
                    'protected_frozen_metadata': str(path) in protected['metadata_sha256'] or str(path) in protected['assets']}
    return inventory


def gpu_guard():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                   text=True, timeout=10).strip()
    if text: raise RuntimeError('GPU is not idle')
    return {'compute_processes': 0}


def guards(s, inodes):
    gpu = gpu_guard(); publication = r.publication_guard(s['pub'], [s['target']])
    if (publication['documents'], publication['plans']) != s['publication_counts']:
        raise RuntimeError('Publication list/plan enumeration differs from reviewed scope')
    return {'gpu': gpu, 'publication': publication, 'processes': r.process_guard([s['target']], inodes)}


def journal_row(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + '\n'); handle.flush(); os.fsync(handle.fileno())


def unlink_exact(s, expected_stat, journal, removed):
    target_check(s, expected_stat)  # 每次唯一unlink前重hash+stat，不沿用早先检查。
    target = s['target']; parent = r.regular(target.parent, True)
    fd = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        current = os.stat(target.name, dir_fd=fd, follow_symlinks=False)
        if ((opened.st_dev, opened.st_ino) != (parent.st_dev, parent.st_ino)
                or [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns] != expected_stat
                or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1 or stat.S_IMODE(current.st_mode) != 0o444):
            raise RuntimeError('Opened-directory target mismatch')
        journal_row(journal, {'phase': 'prepared', 'path': str(target), 'sha256': s['target_sha'], 'stat': expected_stat})
        os.unlink(target.name, dir_fd=fd)  # 唯一文件，不递归，不删除目录。
        removed.append(str(target))  # 后续fsync或journal失败也如实保留实际删除事实。
        os.fsync(fd)
        journal_row(journal, {'phase': 'removed', 'path': str(target)})
    finally:
        os.close(fd)


def execute(s, apply=False):
    # 可注入tiny配置仅供租用CPU tests；CLI不接受目标/目录/私有备份路径覆盖。
    if s['target'] != s['dev'] / RUN_NAMES[0] / 'candidate_mc2/final_full.pt' or s['dev'] != s['root'] / 'exp/diagnostics/student_development_20260906':
        raise RuntimeError('Target escaped the exact registered relative path')
    audit = s['audit']
    if audit != s['dev'] / 'mc2_full3447_retirement_v1': raise RuntimeError('Wrong audit path')
    r.regular(s['dev'], True)
    if audit.exists() or audit.is_symlink(): raise RuntimeError('Audit already exists; never retry a partial retirement')
    if not re.fullmatch('[0-9a-f]{64}', s['source_sha']) or r.sha(Path(__file__).resolve()) != s['source_sha']:
        raise RuntimeError('Wrong executing source SHA')
    owned = False; removed = []; handles = []; failure = None; result = None
    try:
        if apply: audit.mkdir(); owned = True  # apply早期独占占位，任何后续失败不能无痕重试。
        for name in LOCK_NAMES:
            path = s['root'] / 'exp/locks' / name; r.regular(path)
            handle = path.open('r+'); handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        backup, expected_stat = validate_backup(s)
        target_check(s, expected_stat)
        protected = evidence_snapshot(s)
        inodes = {(expected_stat[0], expected_stat[1])}
        before = guards(s, inodes)
        if evidence_snapshot(s, False, protected) != protected: raise RuntimeError('Protected evidence changed during preflight')
        target_check(s, expected_stat, False)
        result = dict(status='validated_not_deleted', exact_target=str(s['target']), backup_receipt_sha256=s['proof_sha'],
            private_backup=backup, executing_source_sha256=s['source_sha'], frozen_guard_sha256=GUARD_SHA,
            logical_bytes=s['logical'], file_allocation_bytes=s['allocated'], file_stat_before=expected_stat,
            protected_before=protected, dependency_guard_before=before,
            historical_reference_inventory=historical_inventory(s, protected),
            historical_references_preserved=True, restore_required_for_full_historical_revalidation=True)
        if apply:
            r.write_new(audit / 'preflight.json', result)
            with (audit / 'removed_files.jsonl').open('x', encoding='utf-8') as journal:
                fresh = guards(s, inodes)
                if fresh['publication'] != before['publication']: raise RuntimeError('Publication changed before unlink')
                if evidence_snapshot(s, False, protected) != protected: raise RuntimeError('Protected evidence changed before unlink')
                unlink_exact(s, expected_stat, journal, removed)
            after = guards(s, inodes)
            if after['publication'] != before['publication']: raise RuntimeError('Publication changed after unlink')
            if evidence_snapshot(s) != protected: raise RuntimeError('Protected evidence changed after unlink')
            if s['target'].exists() or s['target'].is_symlink(): raise RuntimeError('Target remains/reappeared after unlink')
            result.update(status='one_mc2_full3447_retired_private_backup_verified', removed_files=list(removed),
                          removed_count=len(removed), dependency_guard_after=after)
    except BaseException as error:
        failure = error
    finally:
        for handle in reversed(handles):
            try: handle.close()
            except BaseException as error:
                if failure is None: failure = error
    if failure is None and apply:
        try: r.write_completion(audit, result)  # 复用已审持久化/失效机制；全部锁close成功后才发布。
        except BaseException as error: failure = error
    if failure is not None:
        report = dict(status='retirement_failed', error=repr(failure), removed_files=removed, removed_count=len(removed),
                      backup_receipt_sha256=s['proof_sha'], recovery_requires_private_backup=bool(removed), success=False)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if owned:
            try: r.write_new(audit / 'failure.json', report)
            except BaseException as secondary: print('Failure receipt write also failed: ' + repr(secondary), flush=True)
        raise failure
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--source-sha256', required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if Path.cwd() != ROOT: raise RuntimeError('Only the named rented project cwd is allowed')
    result = execute(production_spec(args.source_sha256), args.apply)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
