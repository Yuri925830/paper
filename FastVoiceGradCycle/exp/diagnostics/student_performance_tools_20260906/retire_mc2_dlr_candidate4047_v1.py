"""仅退休已私有验真备份的未采用 D1e4 full4047；默认只读，无模型执行。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path('/root/FsatVoiceGrad')
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
OLD_FILE = TOOLS / 'retire_mc2_full3447_v1.py'
OLD_SHA = 'bd6440c6018f9d8130e6717ced8ee16e88c6b08a3c28765d829edfbe64980a8a'
TEST_NAME = 'test_retire_mc2_dlr_candidate4047_v1.py'
TEST_SHA = '6a680a9ff6af3846c572a09944cae7c68c06b9ddebbd0b4d0f1db82008a54b4e'
INTERFACE_NAME = 'MC2_DLR_CANDIDATE4047_RETIREMENT_INTERFACE_CN.md'
INTERFACE_SHA = '845a2c5cb442e8f07dcd748f198b2c49d99b1555bf2b1bcd4a9b0d72f2b76b2e'
RUN = 'acdd_mc2_dlr_paired120_from3927_v1'
ARM = 'candidate_d1e4'
EVALUATION = 'evaluation_mc2_dlr_d1e4_paired120_endpoint_v1'
AUDIT = 'mc2_dlr_candidate4047_retirement_v1'
PROOF = 'mc2_dlr_candidate4047_private_backup_receipt_v1.json'
PRIVATE = 'C:/Users/rich_/AppData/Local/VoiceGradPrivateBackups/20260907/' + AUDIT + '/final_full.pt'
RUN_NAMES = (RUN, 'acdd_mc2_continuation480_from3447_v1',
             'acdd_s5e5_staged120_from3227_v1', 'acdd_mc2_component_gradients_3927_v1')
if not sys.platform.startswith('linux') or not (ROOT / 'AGENTS.md').is_file():
    raise RuntimeError('Only the registered rental may execute')
if hashlib.sha256(OLD_FILE.read_bytes()).hexdigest() != OLD_SHA:
    raise RuntimeError('Frozen retirement primitive changed')
_spec = importlib.util.spec_from_file_location('_candidate4047_retirement_guards', OLD_FILE)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)  # 只导入定义，不调用旧 main/execute/production_spec。
r, LOCK_NAMES = old.r, old.LOCK_NAMES
validate_backup, target_check, unlink_exact = old.validate_backup, old.target_check, old.unlink_exact
guards, historical_inventory = old.guards, old.historical_inventory


def sha_arg(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise argparse.ArgumentTypeError('Require the reviewed full lowercase SHA256')
    return value


def production_spec(source_sha, proof_sha):
    protected = dict(old.PROTECTED)
    protected.update({
        str(DEV / 'acdd_s2000_to4000_v1/latest_full.pt'): 'f007dd4375bbf23a78372549e6e79b87c06e94fd38c07b5e09eb6624ae50ce44',
        str(DEV / 'acdd_adam_descent_readonly_3107_v1/fixed_inputs.pt'): '296a7b4a7c289f0d19409947cd2c4dcb54754ae5636a58af43e658bc12b17da8',
        str(TOOLS / TEST_NAME): TEST_SHA,
        str(DEV / INTERFACE_NAME): INTERFACE_SHA,
        str(Path(__file__).resolve()): source_sha,
    })
    return dict(root=ROOT, dev=DEV, pub=Path('/root/teacher-hifigan-publication-20260906'),
        target=DEV / RUN / ARM / 'final_full.pt',
        target_sha='a263577015ebdc4df4ac2cbc8fd8c56d84c1101d492f30bed5ca018bd7bc72c8',
        logical=1424624133, allocated=1424625664, proof=DEV / PROOF, proof_sha=proof_sha,
        private=PRIVATE, audit=DEV / AUDIT, protected=protected,
        runs=[DEV / name for name in RUN_NAMES], history=[DEV / INTERFACE_NAME],
        evaluation=DEV / EVALUATION, publication_counts=(218, 4), source_sha=source_sha)


def evidence_snapshot(s, rehash=True, previous=None):
    # 旧函数检查的是旧文件自身 __file__，必须传其真实 SHA；新三文件另列 protected。
    # 不改旧模块全局变量，也不递归继承旧 contract 中已经归档的祖先权重。
    view = dict(s, source_sha=OLD_SHA)
    view['protected'] = dict(s['protected'])
    view['protected'][str(Path(__file__).resolve())] = s['source_sha']
    return old.evidence_snapshot(view, rehash, previous)


def execute(s, apply=False):
    # 可注入 tiny 路径仅供租用 CPU 测试；CLI 没有任意目标/备份路径选项。
    if (s['dev'] != s['root'] / 'exp/diagnostics/student_development_20260906'
            or s['target'] != s['dev'] / RUN / ARM / 'final_full.pt'
            or s['audit'] != s['dev'] / AUDIT or s['proof'] != s['dev'] / PROOF
            or s['private'] != PRIVATE or s['evaluation'] != s['dev'] / EVALUATION):
        raise RuntimeError('Path escaped the exact registered retirement scope')
    sha_arg(s['source_sha']); sha_arg(s['proof_sha'])
    if r.sha(Path(__file__).resolve()) != s['source_sha']:
        raise RuntimeError('Wrong executing source SHA')
    if str(s['target']) in s['protected']:
        raise RuntimeError('Target overlaps a protected asset')
    r.regular(s['dev'], True)
    audit = s['audit']
    if audit.exists() or audit.is_symlink():
        raise RuntimeError('Audit already exists; never retry a partial retirement')
    owned = False; removed = []; handles = []; failure = None; result = None
    try:
        if apply:
            audit.mkdir(); owned = True  # 此后任何失败都保留本次审计，不覆盖重试。
        for name in LOCK_NAMES:
            path = s['root'] / 'exp/locks' / name; r.regular(path)
            handle = path.open('r+'); handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        backup, expected_stat = validate_backup(s)
        target_check(s, expected_stat)
        protected = evidence_snapshot(s)
        inodes = {(expected_stat[0], expected_stat[1])}
        before = guards(s, inodes)
        if evidence_snapshot(s, False, protected) != protected:
            raise RuntimeError('Protected evidence changed during preflight')
        target_check(s, expected_stat, False)
        result = dict(status='validated_not_deleted', exact_target=str(s['target']),
            backup_receipt_sha256=s['proof_sha'], private_backup=backup,
            executing_source_sha256=s['source_sha'], frozen_retirement_sha256=OLD_SHA,
            frozen_guard_sha256=old.GUARD_SHA, logical_bytes=s['logical'],
            file_allocation_bytes=s['allocated'], file_stat_before=expected_stat,
            protected_before=protected, dependency_guard_before=before,
            historical_reference_inventory=historical_inventory(s, protected),
            historical_references_preserved=True, restore_required_for_full_historical_revalidation=True)
        if apply:
            r.write_new(audit / 'preflight.json', result)
            with (audit / 'removed_files.jsonl').open('x', encoding='utf-8') as journal:
                fresh = guards(s, inodes)
                if fresh['publication'] != before['publication']:
                    raise RuntimeError('Publication changed before unlink')
                if evidence_snapshot(s, False, protected) != protected:
                    raise RuntimeError('Protected evidence changed before unlink')
                unlink_exact(s, expected_stat, journal, removed)
            after = guards(s, inodes)
            if after['publication'] != before['publication']:
                raise RuntimeError('Publication changed after unlink')
            if evidence_snapshot(s) != protected:
                raise RuntimeError('Protected evidence changed after unlink')
            if s['target'].exists() or s['target'].is_symlink():
                raise RuntimeError('Target remains/reappeared after unlink')
            result.update(status='one_mc2_dlr_candidate4047_retired_private_backup_verified',
                removed_files=list(removed), removed_count=len(removed), dependency_guard_after=after)
    except BaseException as error:
        failure = error
    finally:
        for handle in reversed(handles):
            try:
                handle.close()
            except BaseException as error:
                if failure is None: failure = error
    if failure is None and apply:
        try:
            r.write_completion(audit, result)  # 四个锁都正常关闭后才发布成功。
        except BaseException as error:
            failure = error
    if failure is not None:
        report = dict(status='retirement_failed', error=repr(failure), success=False,
            removed_files=list(removed), removed_count=len(removed),
            backup_receipt_sha256=s['proof_sha'], recovery_requires_private_backup=bool(removed))
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if owned:
            try:
                r.write_new(audit / 'failure.json', report)
            except BaseException as secondary:
                print('Failure receipt write also failed: ' + repr(secondary), flush=True)
        raise failure
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--source-sha256', type=sha_arg, required=True)
    parser.add_argument('--proof-sha256', type=sha_arg, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if Path.cwd() != ROOT: raise RuntimeError('Only the named rented project cwd is allowed')
    result = execute(production_spec(args.source_sha256, args.proof_sha256), args.apply)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
