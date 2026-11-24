"""Only retire the8100 exact, privately backed-up formal3327 WAV files."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
OUTPUT = ROOT / 'exp/evaluation/fastvoicegrad_text_unseen_s003327_custom_hifigan_20260907'
WAV = OUTPUT / 'wav'
LISTS = Path('/root/teacher-hifigan-publication-20260906/direct-upload-runtime/lists')
PROOF = DEV / 'formal3327_wav_private_backup_receipt.json'
AUDIT = DEV / 'formal3327_wav_retirement_v1'
PINS = {
    OUTPUT / 'pairs.csv': '7c03d60e3c3ce310f2b7a0f5f76432bbd445a595bb8ded65ca9c11854bf9eb0b',
    OUTPUT / 'inference_report.json': '064917786c52bc13281efdb290eb9c0e6d02614da77a2c32c9fab411a312b9e9',
    OUTPUT / 'run_identity.json': '31f42c9918af483f0de346c312939b717abc2ad7ecab2d2121f3601c88139a6f',
    OUTPUT / 'metrics/objective_summary_recovered_v1.json': 'cd05d782e72aa1343d0af266d1db61398174b86750645a8e77962e0d34353a69',
}
DEPENDENCIES = [DEV/'acdd_s5e5_staged120_from3227_v1/run_contract.json',
                DEV/'acdd_lr_paired120_from3107_v1/run_contract.json',
                OUTPUT/'inference_report.json', OUTPUT/'metrics/objective_summary_recovered_v1.json',
                DEV/'acdd_timestep_cross_readonly_3327_v1/run_identity.json']


def regular(path, directory=False):
    # 任何父目录重定向都阻止清理；禁止沿软链追到别处。
    if not path.is_absolute() or any(p.is_symlink() for p in [path]+list(path.parents)):
        raise RuntimeError('Redirected/nonabsolute path: '+str(path))
    info = path.stat()
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise RuntimeError('Wrong path type: '+str(path))
    return info


def sha(path):
    regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp(path):
    value = regular(path)
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]


def list_snapshot():
    regular(LISTS, True)
    return {str(p):stamp(p) for p in sorted(LISTS.iterdir()) if p.is_file() or p.is_symlink()}


def dependency_guard(inodes):
    snapshot = list_snapshot()
    needles = (str(WAV), OUTPUT.name)
    lines = records = 0
    for name in snapshot:
        with Path(name).open() as handle:
            for line in handle:
                lines += 1
                if any(needle in line for needle in needles):
                    raise RuntimeError('Publication dependency: '+name)
                if name.endswith('.metadata.jsonl'):
                    row = json.loads(line)
                    records += 1
                    if str(row.get('source_path','')).startswith(str(WAV)+'/'):
                        raise RuntimeError('Structured publication WAV dependency')
    if list_snapshot() != snapshot:
        raise RuntimeError('Publication manifests changed during scan')
    contracts = 0
    for parent, dirs, names in os.walk(ROOT/'exp', followlinks=False):
        dirs[:] = [name for name in dirs if name not in {'wav','.git','__pycache__','.pytest_cache','envs','pkgs'} and not (Path(parent)/name).is_symlink()]
        for name in names:
            if name.endswith('.json') and ('contract' in name or name in {'config.json','configuration.json'}):
                path = Path(parent)/name
                regular(path)
                contracts += 1
                if any(needle in path.read_text() for needle in needles):
                    raise RuntimeError('Contract/configuration dependency: '+str(path))
    counts = {}
    for path in DEPENDENCIES:
        data = json.loads(path.read_text())['file_sha256']
        counts[str(path)] = len(data)
        if any(str(name).startswith(str(WAV)+'/') for name in data):
            raise RuntimeError('Active structured WAV dependency: '+str(path))
    process_count = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        process_count += 1
        try:
            for item in ('cmdline','maps'):
                data = (proc/item).read_bytes()
                if any(needle.encode() in data for needle in needles):
                    raise RuntimeError('Process references this output: '+proc.name)
            for item in ('cwd','exe'):
                target = os.readlink(proc/item)
                if any(needle in target for needle in needles):
                    raise RuntimeError('Process directory/executable dependency: '+proc.name)
            for fd in (proc/'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    info = fd.stat()
                except FileNotFoundError:
                    continue
                if str(WAV) in target or (info.st_dev,info.st_ino) in inodes:
                    raise RuntimeError('Open WAV descriptor: '+str(fd))
        except (FileNotFoundError,ProcessLookupError):
            continue
        # Permission errors deliberately propagate: an unread process is not a clean result.
    return {'publication_snapshot':snapshot,'publication_documents':len(snapshot),
            'publication_lines':lines,'publication_source_records':records,'contracts':contracts,
            'structured_dependency_counts':counts,'processes':process_count,'hits':0}


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--backup-receipt-sha256',required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux') or Path.cwd() != ROOT:
        raise RuntimeError('Only the named rental root is allowed')
    regular(WAV, True)
    if WAV.resolve() != Path('/root/FsatVoiceGrad/exp/evaluation/fastvoicegrad_text_unseen_s003327_custom_hifigan_20260907/wav'):
        raise RuntimeError('Unexpected absolute target')
    if AUDIT.exists() or AUDIT.is_symlink():
        raise RuntimeError('Retirement identity already exists; do not repeat')
    locks = []
    try:
        for name in ('text_unseen_custom_hifigan_pipeline.lock','text_unseen_custom_hifigan_train.lock','text_unseen_fastvoicegrad_student.lock'):
            path = ROOT/'exp/locks'/name
            regular(path)
            handle = path.open('a+')
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            locks.append(handle)
        if sha(PROOF) != args.backup_receipt_sha256:
            raise RuntimeError('Backup receipt changed')
        proof = json.loads(PROOF.read_text())
        if proof['status'] != 'all_8100_backup_wavs_verified' or proof['count'] != 8100 or proof['bytes'] != 1972847664 or proof['collection_sha256'] != '96f88ebd940da574e806d8699055119ac9a81e96c024d117aba14e8410fb64f0':
            raise RuntimeError('Insufficient private backup proof')
        for path, expected in PINS.items():
            if sha(path) != expected:
                raise RuntimeError('Protected evidence changed: '+str(path))
        with (OUTPUT/'pairs.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        report = json.loads((OUTPUT/'inference_report.json').read_text())
        progress_path = OUTPUT/'wav_progress.jsonl'
        if sha(progress_path) != report['wav_progress_sha256']:
            raise RuntimeError('The completed inference journal changed')
        PINS[progress_path] = report['wav_progress_sha256']
        progress_rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
        if len(progress_rows)!=4200 or any(row['type']!=('prepared' if i%2==0 else 'committed') for i,row in enumerate(progress_rows)):
            raise RuntimeError('Inference journal is incomplete or has a pending chunk')
        if len(rows)!=8100 or {row['pair_id'] for row in rows} != {str(i) for i in range(8100)}:
            raise RuntimeError('Wrong exact manifest')
        selected = [Path(row['output_wav']) for row in rows]
        if len(set(selected))!=8100 or set(WAV.iterdir())!=set(selected):
            raise RuntimeError('Unexpected/missing WAV files')
        allocation = total = 0
        snapshots = {}
        for row,path in zip(rows,selected):
            if path.parent != WAV:
                raise RuntimeError('WAV escaped the exact flat directory')
            info = regular(path)
            if info.st_nlink != 1 or sha(path)!=report['generated_wav_sha256'][row['pair_id']]:
                raise RuntimeError('WAV link/hash mismatch: '+str(path))
            snapshots[str(path)] = stamp(path)
            allocation += info.st_blocks*512
            total += info.st_size
        if total!=proof['bytes']:
            raise RuntimeError('WAV byte count changed')
        staging = OUTPUT/'staging'
        regular(staging, True)
        if any(staging.iterdir()):
            raise RuntimeError('Inference staging is not empty')
        inodes = {(value[0],value[1]) for value in snapshots.values()}
        guard = dependency_guard(inodes)
        for path in selected:
            if stamp(path)!=snapshots[str(path)]:
                raise RuntimeError('WAV changed during dependency audit')
        if list_snapshot()!=guard['publication_snapshot']:
            raise RuntimeError('Publication manifests changed before retirement')
        receipt = {'status':'validated_not_deleted','backup_receipt_sha256':args.backup_receipt_sha256,
                   'private_backup':proof,'exact_wave_directory':str(WAV),'count':8100,'bytes':total,
                   'wave_file_allocation_bytes':allocation,'dependency_guard':guard,
                   'protected_metadata_sha256':{str(p):v for p,v in PINS.items()},
                   'restore_required_for_full_output_validator_or_rescoring':True}
        if not args.apply:
            print(json.dumps({k:v for k,v in receipt.items() if k not in {'private_backup','dependency_guard'}},ensure_ascii=False))
            return
        AUDIT.mkdir()
        write_new(AUDIT/'validated_before_removal.json',receipt)
        removed = []
        try:
            with (AUDIT/'removed_ids.jsonl').open('x') as journal:
                for row,path in zip(rows,selected):
                    if stamp(path)!=snapshots[str(path)]:
                        raise RuntimeError('WAV changed before individual unlink')
                    path.unlink()  # 只逐个移除已验证的8100个文件；绝不递归删除目录。
                    removed.append(row['pair_id'])
                    journal.write(json.dumps({'pair_id':row['pair_id']})+'\n')
                    journal.flush()
                os.fsync(journal.fileno())
            if any(WAV.iterdir()):
                raise RuntimeError('Unexpected entries remain in WAV directory')
            for path,expected in PINS.items():
                if sha(path)!=expected:
                    raise RuntimeError('Protected metadata changed after retirement')
            if list_snapshot()!=guard['publication_snapshot']:
                raise RuntimeError('Publication manifests changed during retirement; review the recoverable removal')
            after_guard = dependency_guard(inodes)
            if after_guard['publication_snapshot']!=guard['publication_snapshot']:
                raise RuntimeError('Publication manifests changed during final verification')
            receipt['dependency_guard_after_removal'] = after_guard
            receipt.update(status='8100_wavs_retired_verified_private_backup',removed_count=len(removed),wave_directory_retained=True)
            write_new(AUDIT/'completion.json',receipt)
            print(json.dumps({'status':receipt['status'],'removed_count':len(removed),'released_file_allocation_bytes':allocation,'recoverable':True}))
        except BaseException as error:
            write_new(AUDIT/'failure.json',{'error':str(error),'removed_count':len(removed),'removed_ids':removed,'backup_receipt_sha256':args.backup_receipt_sha256})
            raise
    finally:
        for handle in reversed(locks):
            handle.close()


if __name__=='__main__':
    main()
