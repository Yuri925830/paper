"""为12号长训终点计算原8100对的UTMOS、CER和DNSMOS，不重算SVA。"""

import argparse
import ctypes
import fcntl
import hashlib
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT/'exp/diagnostics/student_performance_tools_20260906'
OUTPUT = ROOT/'exp/evaluation/fastvoicegrad_text_unseen_s122900_custom_hifigan_20260909'
METRIC_DIR = OUTPUT/'metrics'
BASE = TOOLS/'evaluate_formal_student3327_v1.py'
BASE_SHA = '0842832901c9e6f2fd5d716182b90e988427622768bf267c7116420be2fee5a5'
INFER = TOOLS/'infer_formal_student122900_v1.py'
# 上传前填写经过复核的新推理入口哈希；从头到尾只接受这一份源码。
INFER_SHA = 'ab6f7893e302076382430875eccbd02150496971e322717d61329014406729fa'
FULL_SHA = '2ca44269eefac2dd3f8ffa7afe0337f1f3684469aec9fadeff623237cf37b830'
CONTRACT_SHA = '79d007d0a78796abfe3eeb8c80ff8482cf19dff55bfa33306a31628c2043d1ba'
BASELINE = ROOT/'exp/evaluation/fastvoicegrad_text_unseen_s003327_custom_hifigan_20260907/metrics/objective_summary_recovered_v1.json'
BASELINE_SHA = 'cd05d782e72aa1343d0af266d1db61398174b86750645a8e77962e0d34353a69'
METRICS = ('utmos', 'cer', 'dnsmos')
LOCK_NAMES = ('text_unseen_custom_hifigan_pipeline.lock', 'text_unseen_custom_hifigan_train.lock',
              'text_unseen_fastvoicegrad_student.lock', 'text_unseen_fastvoicegrad_student_inference.lock')
FLAGS = {'cuda_matmul_allow_tf32':False, 'cudnn_allow_tf32':True,
         'cudnn_benchmark':False, 'cudnn_deterministic':False}
PAPER = {'UTMOS':3.96, 'CER_corpus_percent':1.89, 'DNSMOS_P808':3.77}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_import(path, expected):
    # 只复用旧入口的公共校验函数，不修改其全局变量或旧实验身份。
    if path.is_symlink() or sha(path) != expected:
        raise RuntimeError('Source changed: '+str(path))
    for directory in (TOOLS, ROOT):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    module = importlib.import_module(path.stem)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError('Imported module does not match its verified source path')
    return module


def acquire_locks():
    # 四把锁整个评分阶段都持有；子进程继承同一批文件描述符。
    handles = []
    try:
        for name in LOCK_NAMES:
            handle = (ROOT/'exp/locks'/name).open('a+')
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def verify_worker_parent(args):
    fds = [int(value) for value in args.lock_fds.split(',')]
    if os.getppid() != args.parent_pid or len(fds) != 4:
        raise RuntimeError('Worker requires its original parent and four inherited locks')
    for fd, name in zip(fds, LOCK_NAMES):
        if Path(os.readlink('/proc/self/fd/'+str(fd))).resolve() != (ROOT/'exp/locks'/name).resolve():
            raise RuntimeError('Worker inherited another lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != args.parent_pid:
        raise RuntimeError('Worker cannot bind its lifetime to the parent')


def validate_bundle(e, m, bundle):
    # 每项指标使用相同8100条，分母和源文本与教师完全一致。
    report, rows = bundle['report'], bundle['rows']
    expected = {'status':'formal_8100_complete', 'system_id':'fastvoicegrad_student122900',
                'model_role':'student', 'student_global_step':122900, 'student_steps':1,
                'student_start_step':950, 'teacher_epoch':500, 'selected_pair_count':8100,
                'paper_pair_count':8100, 'student_checkpoint_sha256':FULL_SHA,
                'student_run_contract_sha256':CONTRACT_SHA}
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError('Unexpected completed inference field: '+key)
    teacher_rows = m.load_pairs(e.TEACHER/'pairs.csv', 0, None)
    e.require_pairs(rows, teacher_rows)
    if sum(len(m.normalize_cer_text(row['source_text'])) for row in rows) != 566190:
        raise RuntimeError('Formal CER denominator changed')
    if bundle['report_sha256'] != sha(OUTPUT/'inference_report.json'):
        raise RuntimeError('Inference report changed')
    return rows


def schema(m):
    return {'utmos':m.UTMOS_FIELDS, 'cer':m.CER_FIELDS, 'dnsmos':m.DNSMOS_FIELDS}


def read_metrics(m, rows, complete=False):
    data = {name:m.validate_existing_metric_csv(METRIC_DIR/(name+'.csv'), rows, name, fields)
            for name, fields in schema(m).items()}
    if complete and any(len(values) != 8100 for values in data.values()):
        raise RuntimeError('All three selected metrics must contain all8100 pairs')
    for pair, cer in zip(rows, data['cer']):
        if cer['reference_text_normalized'] != m.normalize_cer_text(pair['source_text']):
            raise RuntimeError('CER reference text changed')
    return data


def summarize(data):
    # CER按错误总数除以字符总数，不能平均逐句百分比。
    errors = sum(int(row['char_errors']) for row in data['cer'])
    characters = sum(int(row['reference_chars']) for row in data['cer'])
    if characters != 566190:
        raise RuntimeError('Incomplete CER corpus')
    scores = {'UTMOS':math.fsum(float(row['utmos']) for row in data['utmos'])/8100,
              'CER_corpus_percent':100*errors/characters,
              'DNSMOS_P808':math.fsum(float(row['p808_mos']) for row in data['dnsmos'])/8100}
    if not all(math.isfinite(value) for value in scores.values()):
        raise RuntimeError('Nonfinite final metric')
    return scores, errors, characters


def baseline_comparison(e, scores):
    # 旧模型和教师只读取已完成的正式结果，不重新加载或评分。
    e.verify_files({str(BASELINE):BASELINE_SHA})
    old = e.read_json(BASELINE)['comparison']['systems']
    systems = {'student122900':scores}
    for name in ('student3327', 'teacher500'):
        systems[name] = {key:old[name][key]['estimate'] for key in PAPER}
    systems['paper_fastvoicegrad'] = PAPER
    deltas = {name:{key:scores[key]-reference[key] for key in PAPER}
              for name, reference in systems.items() if name != 'student122900'}
    return {'systems':systems, 'student122900_minus_reference':deltas,
            'DNSMOS_paper_caveat':'Paper Table1 gives DNSMOS3.77 without an explicit implementation/output-field specification; P808 is a provisional comparison, not verified scorer equivalence.'}


def worker(args):
    verify_worker_parent(args)
    e = checked_import(BASE, BASE_SHA)
    e.check_runtime()
    if any(os.environ.get(name) != '1' for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')):
        raise RuntimeError('Scoring must use the existing offline model assets')
    identity = e.read_json(METRIC_DIR/'metrics_run_identity.json')
    if e.canonical(identity) != args.identity_sha:
        raise RuntimeError('Worker scoring identity changed')
    e.verify_files(identity['file_sha256'])
    infer = checked_import(INFER, INFER_SHA)
    infer.capacity(OUTPUT, 0, 0)
    m = e.load_evaluator()
    rows = m.load_pairs(OUTPUT/'pairs.csv', 0, None)
    e.require_pairs(rows, m.load_pairs(e.TEACHER/'pairs.csv', 0, None))
    m.validate_generated_wavs(rows, m.load_generated_wav_hash_manifest(e.read_json(OUTPUT/'inference_report.json'), rows))
    e.auxiliary_sources(e.scoring_assets(m))
    if e.scoring_precision(m) != FLAGS:
        raise RuntimeError('Fresh worker precision differs from original formal UTMOS/CER/DNS')

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Metric received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    destination = METRIC_DIR/(args.worker+'.csv')
    if args.worker == 'utmos':
        m.evaluate_utmos(rows, destination, ROOT/'UTMOS-demo', m.torch.device('cuda'), 4)
    elif args.worker == 'cer':
        m.evaluate_cer(rows, destination, m.torch.device('cuda'), 4)
    else:
        m.evaluate_dnsmos(rows, destination, 4, 3)
    if e.scoring_precision(m) != FLAGS:
        raise RuntimeError('Worker precision changed during scoring')
    e.atomic_json(METRIC_DIR/(args.worker+'_worker_complete.json'),
                  {'metric':args.worker, 'identity_sha256':args.identity_sha,
                   'before_flags':FLAGS, 'after_flags':e.scoring_precision(m),
                   'metric_csv_sha256':sha(destination)})


def metric_group_is_live(pgid):
    # DNSMOS池的直接worker可能先退出；检查整个专属session/group，忽略已死的zombie。
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            value = (entry/'stat').read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        fields = value[value.rfind(')')+2:].split()
        if (len(fields) >= 4 and fields[0] not in ('Z', 'X', 'x')
                and int(fields[2]) == pgid and int(fields[3]) == pgid):
            return True
    return False


def stop_metric_group(process, grace_seconds=30.0, kill_seconds=5.0):
    # Popen(start_new_session=True)给本项评分独占PID=PGID=SID；只处理这个组。
    # 正常退出且无活子进程时立即返回，不因zombie额外等待30秒。
    def send(signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    process.poll()
    if metric_group_is_live(process.pid):
        send(signal.SIGTERM)
        until = time.monotonic()+grace_seconds
        while metric_group_is_live(process.pid) and time.monotonic() < until:
            process.poll()
            time.sleep(0.1)
        if metric_group_is_live(process.pid):
            send(signal.SIGKILL)
            until = time.monotonic()+kill_seconds
            while metric_group_is_live(process.pid) and time.monotonic() < until:
                process.poll()
                time.sleep(0.1)
    if metric_group_is_live(process.pid):
        raise RuntimeError('Owned scoring process group remains alive: '+str(process.pid))
    # 回收本入口的直接子进程；孤儿zombie只能由它们的新父进程回收，不占评分资源。
    process.wait(timeout=kill_seconds)


def run_metric(e, infer, name, identity_sha, locks, deadline):
    # 与旧正式流程一致，每项从干净子进程启动，避免SVA导入改变TF32设置。
    infer.capacity(OUTPUT, 0, 0)
    if time.monotonic() > deadline:
        raise TimeoutError('Formal evaluation deadline passed before starting '+name)
    command = [str(e.PYTHON), '-u', str(Path(__file__).resolve()), '--worker', name,
               '--parent-pid', str(os.getpid()), '--lock-fds', ','.join(str(h.fileno()) for h in locks),
               '--identity-sha', identity_sha]
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    process = subprocess.Popen(command, cwd=ROOT, env=environment, start_new_session=True,
                               pass_fds=tuple(h.fileno() for h in locks))
    try:
        while process.poll() is None:
            infer.capacity(OUTPUT, 0, 0)
            if time.monotonic() > deadline:
                raise TimeoutError('Formal evaluation exceeded three hours')
            time.sleep(2)
        if process.returncode:
            raise RuntimeError('Metric '+name+' exited '+str(process.returncode))
    finally:
        # 直接worker已退出也要处理残留DNSMOS池；绝不触碰训练或其他评分任务。
        stop_metric_group(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=10800)
    parser.add_argument('--worker', choices=METRICS)
    parser.add_argument('--parent-pid', type=int, default=-1)
    parser.add_argument('--lock-fds', default='')
    parser.add_argument('--identity-sha', default='')
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if not args.run_approved or not 1 <= args.max_seconds <= 10800:
        raise RuntimeError('The single authorized evaluation requires a three-hour bound')
    e = checked_import(BASE, BASE_SHA)
    versions = e.check_runtime()
    infer = checked_import(INFER, INFER_SHA)
    m = e.load_evaluator()
    from fastvoicegrad import train as training
    training.assert_rented_server(ROOT)
    locks = acquire_locks()
    started = time.monotonic()
    identity_sha = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Evaluation received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if training.git_head(ROOT) != e.HEAD or subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Frozen production code changed')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('Another GPU process is active')
        infer.capacity(OUTPUT, 0, 0)
        bundle = infer.validate_complete_output(OUTPUT, verify_dependencies=True)
        rows = validate_bundle(e, m, bundle)
        assets = e.scoring_assets(m)
        files = dict(bundle['identity']['file_sha256'])
        e.merge_pins(files, {str(BASE):BASE_SHA, str(INFER):INFER_SHA,
                            str(Path(__file__).resolve()):sha(Path(__file__)), str(BASELINE):BASELINE_SHA,
                            str(e.EVALUATOR):e.EVALUATOR_SHA})
        e.merge_pins(files, {v['path']:v['sha256'] for v in assets.values()})
        e.merge_pins(files, e.auxiliary_sources(assets))
        for name in ('pairs.csv', 'run_identity.json', 'inference_report.json'):
            e.merge_pins(files, {str(OUTPUT/name):sha(OUTPUT/name)})
        e.verify_files(files)
        if e.scoring_precision(m) != FLAGS:
            raise RuntimeError('Parent precision differs from frozen original formal scoring')
        identity = {'schema_version':1, 'system_id':'fastvoicegrad_student122900',
                    'student_global_step':122900, 'completed_passes':100, 'student_checkpoint_sha256':FULL_SHA,
                    'student_run_contract_sha256':CONTRACT_SHA, 'inference_report_sha256':bundle['report_sha256'],
                    'inference_identity_sha256':bundle['identity_sha256'], 'pairs_csv_sha256':sha(OUTPUT/'pairs.csv'),
                    'selected_pair_count':8100, 'batch_size':4, 'dnsmos_workers':3, 'metrics':list(METRICS),
                    'scoring_precision_flags':FLAGS, 'software_versions':versions, 'metric_assets':assets,
                    'file_sha256':files, 'generated_wav_collection_sha256':bundle['report']['generated_wav_collection_sha256'],
                    'SVA_rescored':False, 'DNSMOS_comparison_field':'p808_mos'}
        if METRIC_DIR.is_symlink():
            raise RuntimeError('Redirected metric directory')
        METRIC_DIR.mkdir(exist_ok=True)
        e.bind_identity(METRIC_DIR/'metrics_run_identity.json', identity)
        identity_sha = e.frozen_identity(identity, METRIC_DIR/'metrics_run_identity.json')
        read_metrics(m, rows)
        summary_path = METRIC_DIR/'objective_summary.json'
        if summary_path.exists():
            # 再次检查时只核验已完成结果，不重复调用任何评分模型。
            saved = e.read_json(summary_path)
            scores, errors, characters = summarize(read_metrics(m, rows, complete=True))
            if (saved.get('status') != 'formal_selected3_8100_complete'
                    or saved.get('metrics_run_identity_sha256') != identity_sha
                    or saved.get('headline_metrics') != scores
                    or saved.get('CER_total_character_errors') != errors
                    or saved.get('CER_total_reference_characters') != characters
                    or saved.get('individual_metric_csv_sha256') != {name:sha(METRIC_DIR/(name+'.csv')) for name in METRICS}):
                raise RuntimeError('Existing summary does not match its complete original rows')
            print('FORMAL122900_ALREADY_COMPLETE_VERIFIED '+json.dumps(scores), flush=True)
            return
        for name in METRICS:
            print('FORMAL122900_METRIC_START '+name, flush=True)
            run_metric(e, infer, name, identity_sha, locks, started+args.max_seconds)
            if len(m.validate_existing_metric_csv(METRIC_DIR/(name+'.csv'), rows, name, schema(m)[name])) != 8100:
                raise RuntimeError('Incomplete metric: '+name)
            e.atomic_json(METRIC_DIR/'progress.json', {'status':'partial_metrics_complete',
                          'last_completed_metric':name, 'identity_sha256':identity_sha,
                          'elapsed_seconds':time.monotonic()-started})
        data = read_metrics(m, rows, complete=True)
        scores, errors, characters = summarize(data)
        comparison = baseline_comparison(e, scores)
        final = infer.validate_complete_output(OUTPUT, verify_dependencies=True)
        if (final['report_sha256'], final['identity_sha256']) != (bundle['report_sha256'], bundle['identity_sha256']):
            raise RuntimeError('Inference identity changed during scoring')
        e.verify_files(files)
        e.frozen_identity(identity, METRIC_DIR/'metrics_run_identity.json')
        infer.capacity(OUTPUT, 0, 0)
        if time.monotonic() > started+args.max_seconds:
            raise TimeoutError('Formal evaluation exceeded its time bound before publication')
        result = {'status':'formal_selected3_8100_complete', 'system_id':'fastvoicegrad_student122900',
                  'student_global_step':122900, 'completed_passes':100, 'student_checkpoint_sha256':FULL_SHA,
                  'student_run_contract_sha256':CONTRACT_SHA, 'teacher_epoch':500, 'vocoder_global_step':2500000,
                  'evaluated_pairs':8100, 'metrics':list(METRICS), 'headline_metrics':scores,
                  'CER_total_character_errors':errors, 'CER_total_reference_characters':characters,
                  'individual_metric_csv_sha256':{name:sha(METRIC_DIR/(name+'.csv')) for name in METRICS},
                  'metrics_run_identity_sha256':identity_sha, 'metrics_run_identity_file_sha256':sha(METRIC_DIR/'metrics_run_identity.json'),
                  'inference_report_sha256':bundle['report_sha256'], 'comparison':comparison,
                  'elapsed_seconds':time.monotonic()-started, 'SVA_rescored':False,
                  'limitations':['Student LR5e-5 differs from paper2e-4; the accepted long-run settings and teacher are unchanged.',
                      'The previous52-utterance development set was excluded from39335 training utterances; completed100passes are over that retained set.',
                      'P808 is compared provisionally to the paper DNSMOS; the paper does not identify the exact output implementation.',
                      'Historical teacher backend precision flags were not saved; original available scoring code, weights and input pairs are preserved.',
                      'No subjective test, SVA recomputation, new tuning or checkpoint selection is performed.']}
        e.atomic_json(METRIC_DIR/'objective_summary.json', result)
        print('FORMAL122900_SELECTED3_COMPLETE '+json.dumps(scores), flush=True)
    except BaseException as error:
        failure = {'status':'failed_or_interrupted_no_complete_claim', 'error_type':type(error).__name__,
                   'error':str(error), 'identity_sha256':identity_sha, 'elapsed_seconds':time.monotonic()-started}
        print('FORMAL122900_FAILED '+json.dumps(failure), flush=True)
        if METRIC_DIR.is_dir():
            e.atomic_json(METRIC_DIR/'failure.json', failure)
        raise
    finally:
        for handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
