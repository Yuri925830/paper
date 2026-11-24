"""只补齐固定3327步的DNSMOS；原评分失败记录和三份已有结果保持不动。"""

import argparse  # 必须另行批准本次补齐，不能导入模块就开始评分。
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT/'exp/diagnostics/student_performance_tools_20260906'
DEV = ROOT/'exp/diagnostics/student_development_20260906'
OUTPUT = ROOT/'exp/evaluation/fastvoicegrad_text_unseen_s003327_custom_hifigan_20260907'
METRICS = OUTPUT/'metrics'
BASE = TOOLS/'evaluate_formal_student3327_v1.py'
BASE_SHA = '0842832901c9e6f2fd5d716182b90e988427622768bf267c7116420be2fee5a5'
PROTOCOL = DEV/'FORMAL3327_DNS_COMPLETION_PROTOCOL_CN.md'
PROTOCOL_SHA = '8d5d3eb1c71a8134fc9ff4ef4fb3b81e5d13670a5d3c6a9a672206263de0a28d'
OLD_IDENTITY_SHA = '4e7340ff7e04ff8916e260918ef14ce8504624f5dd737172ff9dde7903288d82'
IDENTITY = METRICS/'dns_completion_identity_v1.json'
SUMMARY = METRICS/'objective_summary_recovered_v1.json'
FAILURE = METRICS/'dns_completion_failure_v1.json'
STATUS = 'formal_primary_8100_complete_sva_qualified'
# 原评分日志已停止写入；新进度只进入metrics/dns_completion_v1.log，由原容量函数计数。
PRESERVED = {
    str(METRICS/'metrics_run_identity.json'): '319a067e48cbdd8c71522e6e4f333422349c756b96d17d4f2a7e982bf2ce24e6',
    str(METRICS/'failure.json'): 'c80ce3d138e604a5a71ed3af88c006673a6d6a142a82745c2775e3d80d71ff5b',
    str(METRICS/'progress.json'): '4dfc5ffe86563ad076ef9b0c86e11e2cf67271d3aeb96be43b67ff9758c6f190',
    str(METRICS/'utmos.csv'): '9ad3661171a00dcf00b7a406d63d0a884ec123be7e73872076f172b4d6966ebd',
    str(METRICS/'cer.csv'): 'b7a23d6e50aafe2e19dc795eec0b0dd5fdffb18e1bb55fbe696f333b803f2ae0',
    str(METRICS/'sva.csv'): '20bc0516cb8417e280f11bc2433d0e0c9504fcf4628c110385b2a18bef2f1f5c',
    str(OUTPUT)+'.metrics.log': '103e74da3f392b1016deded2a6a467bb4a4e3fcc2ee2a0155ed2a17fe411ae88',
    str(OUTPUT)+'.log': '9d959ecd26eca2f486cbfa3b5bce652910b48c572268000706e0af966ec04288',
    str(OUTPUT/'pairs.csv'): '7c03d60e3c3ce310f2b7a0f5f76432bbd445a595bb8ded65ca9c11854bf9eb0b',
    str(OUTPUT/'run_identity.json'): '31f42c9918af483f0de346c312939b717abc2ad7ecab2d2121f3601c88139a6f',
    str(OUTPUT/'inference_report.json'): '064917786c52bc13281efdb290eb9c0e6d02614da77a2c32c9fab411a312b9e9',
}
# 精确记录会触发默认TF32设置的四个已安装SpeechBrain源码文件；不编辑这些文件。
SB_ROOT = Path('/root/miniconda3/envs/voicegrad-eval/lib/python3.9/site-packages/speechbrain')
SB_FILES = {str(SB_ROOT/name): digest for name,digest in (
    ('__init__.py','d0fa13fc7a36cca3d6335e4ca06176803177e81046320f9b9d268cc60705668e'),
    ('core.py','c510aabc8ab49a3098da1c1b90b05d648eb6ad536cb87e6ee89cf62aa37236ab'),
    ('utils/quirks.py','d93afa77b1b2838e99b7b81eb68d8013bcdd69944ed7a7432a8ab673ebe03746'),
    ('inference/speaker.py','5914bcd259f2496cea9747f741399831ae4be3faade46f64be1ba49f1e13e8e5'),
)}
BEFORE = {'cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':True,'cudnn_benchmark':False,'cudnn_deterministic':False}
AFTER = dict(BEFORE,cuda_matmul_allow_tf32=True)
SVA_LIMITATION = (
    'SVA retains all8100 original rows. Its original worker failed the post-scoring precision check; '
    'the four actual end flags were not logged. A later isolated CPU import trace explains the '
    'expected SpeechBrain default transition but is not an observation of that exited worker. '
    'Historical source/version agreement does not prove bytewise historical precision agreement.'
)
# 子进程只导入原SpeakerRecognition类，绝不调用from_hparams或任何网络前向。
TRACE_PROGRAM = '''import json, os, torch
def flags():
    return {'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cudnn_deterministic':torch.backends.cudnn.deterministic}
before=flags()
from speechbrain.inference.speaker import SpeakerRecognition
import speechbrain
from speechbrain.utils import quirks
print('SB_IMPORT_TRACE '+json.dumps({'before':before,'after':flags(),
    'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,
    'speechbrain':speechbrain.__version__,'sb_disable_quirks':os.environ.get('SB_DISABLE_QUIRKS'),
    'applied_quirks':sorted(quirks.applied_quirks),'excluded_quirks':sorted(quirks.excluded_quirks),
    'imported_class':SpeakerRecognition.__module__+'.'+SpeakerRecognition.__name__,
    'network_constructed':False,'model_forward_performed':False},sort_keys=True))
'''


def load_base():
    # 先验原入口字节，再导入它的只读校验和受监控worker；不替换原全局变量。
    if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:
        raise RuntimeError('Frozen original metric driver changed')
    if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
    return importlib.import_module('evaluate_formal_student3327_v1')


def remaining(deadline):
    # CPU预检、DNS和最后汇总共用同一截止时间，不在进入新阶段时重置计时。
    seconds=deadline-time.monotonic()
    if seconds<=0:raise TimeoutError('DNS completion exceeded its declared one-hour budget')
    return seconds


def require_regular(mapping):
    # 原证据必须仍是普通文件，不能用软链接指向另一份结果来通过路径检查。
    for name in mapping:
        p=Path(name)
        if p.is_symlink() or not p.is_file():raise RuntimeError('Missing or redirected frozen evidence: '+name)


def verify_retained(e):
    # 原失败身份必须保留失败结论；不能把原progress改成SVA成功来绕过这次补齐。
    require_regular(PRESERVED);e.verify_files(PRESERVED);e.verify_files(SB_FILES)
    old=e.read_json(METRICS/'metrics_run_identity.json')
    failure=e.read_json(METRICS/'failure.json');progress=e.read_json(METRICS/'progress.json')
    if e.canonical(old)!=OLD_IDENTITY_SHA or old['scoring_precision_flags']!=BEFORE:
        raise RuntimeError('Original metric identity or startup precision differs')
    if failure.get('identity_sha256')!=OLD_IDENTITY_SHA or failure.get('error')!='Metric sva exited 1':
        raise RuntimeError('The preserved failure is not the reviewed SVA worker failure')
    if failure.get('status')!='failed_or_interrupted_no_complete_claim' or progress.get('last_completed_metric')!='cer':
        raise RuntimeError('Original failed/partial status must remain truthful')
    if progress.get('identity_sha256')!=OLD_IDENTITY_SHA or (METRICS/'objective_summary.json').exists():
        raise RuntimeError('Original identity changed or an unapproved standard summary exists')
    return old


def retained_rows(e,m,rows):
    # 原校验器重算CER编辑距离并检查阈值；DNS允许合法前缀，其他三项必须已完整。
    data=e.metric_rows(m,METRICS,rows,complete=False)
    if any(len(data[name])!=8100 for name in ('utmos','cer','sva')):
        raise RuntimeError('Retained UTMOS/CER/SVA must each contain all8100 rows')
    counts=(sum(int(r['char_errors']) for r in data['cer']),sum(int(r['reference_chars']) for r in data['cer']))
    if counts!=(11396,566190):raise RuntimeError('Retained CER integer counts changed')
    return data


def cpu_import_trace(e,deadline):
    # CUDA为空只作用于这个CPU子进程，不改变父进程或随后原DNS worker的环境。
    e.verify_files(SB_FILES)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    done=subprocess.run([str(e.PYTHON),'-c',TRACE_PROGRAM],cwd=str(ROOT),env=env,
        capture_output=True,text=True,timeout=min(60,remaining(deadline)))
    if done.returncode:raise RuntimeError('Isolated SpeechBrain import failed: '+done.stderr[-4000:])
    records=[line[len('SB_IMPORT_TRACE '):] for line in done.stdout.splitlines() if line.startswith('SB_IMPORT_TRACE ')]
    if len(records)!=1:raise RuntimeError('CPU import did not produce exactly one trace')
    trace=json.loads(records[0])
    if trace['before']!=BEFORE or trace['after']!=AFTER or trace['cuda_visible_devices']!='' or trace['cuda_available']:
        raise RuntimeError('CPU import precision transition differs from the reviewed SpeechBrain behavior')
    if trace['torch']!='2.6.0+cu124' or trace['speechbrain']!='1.0.3' or trace['excluded_quirks'] or 'allow_tf32' not in trace['applied_quirks']:
        raise RuntimeError('CPU import uses another runtime or disables the original SpeechBrain quirks')
    trace.update({'evidence_scope':'post_failure_CPU_import_only_not_original_SVA_end_flags',
        'program_sha256':hashlib.sha256(TRACE_PROGRAM.encode()).hexdigest()})
    e.verify_files(SB_FILES)
    return trace


def bind_completion(e,path,identity,dns_path):
    # 手动恢复只接受同一份补齐身份；无新身份的DNS前缀来源不明，不能接着算。
    if path.is_symlink():raise RuntimeError('Redirected DNS completion identity')
    if path.exists():
        if e.read_json(path)!=identity:raise RuntimeError('DNS completion resume identity changed')
    else:
        if dns_path.exists() or dns_path.is_symlink():raise RuntimeError('DNS CSV exists without its completion identity')
        e.atomic_json(path,identity)
    return e.frozen_identity(identity,path)


def run_dns(e,m,rows,locks,deadline):
    # 唯一评分调用是原DNS worker，原三锁、容量巡检、退出/精度检查全部保留。
    remaining(deadline);e.capacity();verify_retained(e)
    e.supervised_metric('dnsmos',OLD_IDENTITY_SHA,locks,deadline)
    remaining(deadline);verify_retained(e)
    result=m.validate_existing_metric_csv(METRICS/'dnsmos.csv',rows,'dnsmos',e.schema(m)['dnsmos'])
    if len(result)!=8100:raise RuntimeError('DNS worker exited without all8100 valid rows')
    return result


def make_summary(e,bundle,identity,identity_sha,comparison,sums,files,elapsed):
    # 新状态明确主要三项成功、SVA有保留条件；原v1失败记录不会被本报告覆盖。
    report=bundle['report'];current=comparison['systems']['student3327']
    return {'status':STATUS,'system_id':'fastvoicegrad_student3327','model_role':'student',
        'student_global_step':3327,'student_epoch':report['student_epoch'],
        'student_epoch_semantics':'Actual development cursor; not100 completed epochs.',
        'student_steps':1,'student_start_step':950,'teacher_epoch':500,
        'student_checkpoint_sha256':e.STUDENT_SHA,'parent_full_sha256':e.FULL_SHA,'student_run_contract_sha256':e.CONTRACT_SHA,
        'vocoder_kind':report['vocoder_kind'],'vocoder_checkpoint_sha256':report['vocoder_checkpoint_sha256'],
        'vocoder_global_step':report['vocoder_global_step'],'evaluated_pairs':8100,'metric_order':list(e.COLUMNS),
        'headline_metrics':{key:current[e.COLUMNS[i]]['estimate'] for i,key in enumerate(e.HEADLINES)},
        'additional_diagnostics':{'CER_total_character_errors':int(sums[:,0].sum()),
            'CER_total_reference_characters':int(sums[:,1].sum()),'SVA_threshold':0.25,'SVA_mean_cosine_score':current['SVA_cosine']['estimate']},
        'individual_metric_csv_sha256':{name:e.sha(METRICS/(name+'.csv')) for name in e.METRICS},
        'metrics_run_identity_sha256':OLD_IDENTITY_SHA,'metrics_run_identity_file_sha256':PRESERVED[str(METRICS/'metrics_run_identity.json')],
        'dns_completion_identity_sha256':identity_sha,'dns_completion_identity_file_sha256':e.sha(IDENTITY),
        'metric_assets':identity['metric_assets'],'software_versions':identity['software_versions'],
        'worker_status':{'utmos':'retained_original_worker_exit0','cer':'retained_original_worker_exit0',
            'sva':'retained_complete_CSV_original_post_scoring_precision_check_failed','dnsmos':'completion_original_DNS_worker_exit0'},
        'precision_evidence':{'original_startup_flags':BEFORE,'original_sva_actual_end_flags':None,
            'later_cpu_import_trace':identity['later_cpu_import_trace'],'dns_original_worker_pre_post_guard_passed':True},
        'metric_qualifications':{'SVA_percent':SVA_LIMITATION,'SVA_cosine':SVA_LIMITATION},
        'comparison':comparison,'file_sha256':dict(files),'elapsed_completion_seconds':elapsed,
        'elapsed_original_failed_seconds':585.5438517779112,
        'limitations':[SVA_LIMITATION,
            'Only the preregistered actual3327 checkpoint is evaluated; no temporary3328 model, new training or checkpoint selection.',
            'Student LR5e-5 and3327 updates differ from paper LR2e-4 and100 epochs; prior52-utterance development-driven selection remains disclosed.',
            'The authors did not identify the exact DNSMOS implementation; current OVRL does not establish equivalence to paper3.77.',
            'All eight columns are retained. Source-cluster intervals condition on the fixed target set, omit some shared-speaker/reference dependencies, and are not multiple-comparison adjusted.',
            'Historical teacher lacks complete per-WAV/noise and actual backend precision receipts; historical CSVs remain frozen.',
            'Performance does not certify sustained or monotonic training-loss decrease.']}


def failure_state(error,identity_sha,elapsed):
    # 只声明本入口没有写旧档；若旧档被外部修改，不能虚称其实际字节仍然保持。
    return {'status':'dns_completion_failed_no_new_complete_claim','error_type':type(error).__name__,'error':str(error),
        'original_metrics_identity_sha256':OLD_IDENTITY_SHA,'dns_completion_identity_sha256':identity_sha,
        'elapsed_seconds':elapsed,'original_artifacts_not_written_by_this_completion':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=3600)
    args=parser.parse_args()
    if not args.run_approved or not 1<=args.max_seconds<=3600:raise RuntimeError('Separate DNS-only approval and at most3600 seconds are required')
    e=load_base();versions=e.check_runtime();m=e.load_evaluator()
    from fastvoicegrad import train as training
    training.assert_rented_server(ROOT)
    locks=training.acquire_runtime_locks(ROOT)
    started=time.monotonic();deadline=started+args.max_seconds;identity_sha=None
    # 信号只中断这个补齐任务；原supervised_metric负责终止自己创建的评分进程组。
    def interrupted(signum,frame):raise KeyboardInterrupt('DNS completion interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        if SUMMARY.exists() or SUMMARY.is_symlink():raise RuntimeError('Recovered summary already exists; no automatic re-evaluation')
        if training.git_head(ROOT)!=e.HEAD or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Frozen project source must remain clean')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Another GPU process is still active')
        # 三锁和GPU门禁之外，再要求原评分/推理PID确实已经不存在。
        if any((Path('/proc')/str(pid)).exists() for pid in (40225,40431)):
            raise RuntimeError('Original inference or failed metric process has not exited')
        old=verify_retained(e);e.capacity();remaining(deadline)
        files=dict(e.PINNED)
        for mapping in (PRESERVED,SB_FILES,old['file_sha256'],{str(BASE):BASE_SHA,str(PROTOCOL):PROTOCOL_SHA,
                str(e.INFER):e.INFER_SHA,str(Path(__file__).resolve()):e.sha(Path(__file__).resolve())}):e.merge_pins(files,mapping)
        require_regular(files);e.verify_files(files)
        print('DNS_COMPLETION_VALIDATE_INFERENCE',flush=True)
        bundle=e.load_inference().validate_complete_output(OUTPUT,verify_dependencies=True)
        rows=e.validate_lineage(bundle);e.merge_pins(files,bundle['identity']['file_sha256'])
        assets=e.scoring_assets(m);e.merge_pins(files,e.auxiliary_sources(assets));retained_rows(e,m,rows)
        # 原历史CSV先绑定来源；后续统计不会暗中扩写已发布的补齐身份字典。
        systems={};clusters=None
        for label,directory in [('teacher500',e.TEACHER),('legacy_student100',e.LEGACY)]:
            found,systems[label],_=e.historical(m,directory,rows,files)
            if clusters is not None and clusters!=found:raise RuntimeError('Historical source clusters differ')
            clusters=found
        if e.scoring_precision(m)!=BEFORE:raise RuntimeError('Parent precision differs before isolated CPU trace')
        trace=cpu_import_trace(e,deadline)
        if e.scoring_precision(m)!=BEFORE:raise RuntimeError('CPU import unexpectedly changed parent precision')
        identity={'schema_version':1,'operation':'DNSMOS_only_completion_of_failed_formal3327_v1',
            'original_metrics_identity_sha256':OLD_IDENTITY_SHA,'new_summary_status':STATUS,
            'inference_report_sha256':bundle['report_sha256'],'inference_identity_sha256':bundle['identity_sha256'],
            'student_checkpoint_sha256':e.STUDENT_SHA,'parent_full_sha256':e.FULL_SHA,'student_run_contract_sha256':e.CONTRACT_SHA,
            'allowed_metric':'dnsmos','batch_size':4,'dnsmos_workers':3,'max_seconds':args.max_seconds,
            'software_versions':versions,'metric_assets':assets,'later_cpu_import_trace':trace,
            'original_sva_actual_end_flags':None,'sva_limitation':SVA_LIMITATION,
            'preserved_file_sha256':dict(PRESERVED),'file_sha256':dict(files),'protocol_sha256':PROTOCOL_SHA}
        e.capacity();remaining(deadline)
        identity_sha=bind_completion(e,IDENTITY,identity,METRICS/'dnsmos.csv')
        print('DNS_COMPLETION_START '+identity_sha,flush=True)
        run_dns(e,m,rows,locks,deadline)
        current=e.metric_rows(m,METRICS,rows);found,sums=e.sums_by_source(rows,current)
        if found!=clusters:raise RuntimeError('Current source clusters differ from historical baselines')
        systems['student3327']=sums;comparison=e.compare_systems(systems,clusters)
        # 完成后再核全部真实WAV和依赖，不仅检查三个保留CSV；任何变化都拒绝发布。
        final=e.load_inference().validate_complete_output(OUTPUT,verify_dependencies=True)
        if (final['report_sha256'],final['identity_sha256'])!=(bundle['report_sha256'],bundle['identity_sha256']):
            raise RuntimeError('Formal inference identity changed during DNS completion')
        verify_retained(e);e.auxiliary_sources(e.scoring_assets(m));e.verify_files(files)
        if e.frozen_identity(identity,IDENTITY)!=identity_sha:raise RuntimeError('Completion identity changed while scoring')
        if e.scoring_precision(m)!=BEFORE:raise RuntimeError('Parent precision changed during completion')
        e.capacity();remaining(deadline)
        # 新摘要只新增DNS的最终指纹，不让可增长来源表反过来改变已发布身份。
        final_files=dict(files);e.merge_pins(final_files,{str(METRICS/'dnsmos.csv'):e.sha(METRICS/'dnsmos.csv'),str(IDENTITY):e.sha(IDENTITY)})
        result=make_summary(e,bundle,identity,identity_sha,comparison,sums,final_files,time.monotonic()-started)
        e.atomic_json(SUMMARY,result)
        print('DNS_COMPLETION_PRIMARY_COMPLETE_SVA_QUALIFIED '+str(SUMMARY),flush=True)
    except BaseException as error:
        state=failure_state(error,identity_sha,time.monotonic()-started)
        print('DNS_COMPLETION_FAILED '+json.dumps(state),flush=True)
        try:e.atomic_json(FAILURE,state)
        except Exception as secondary:print('DNS_COMPLETION_FAILURE_METADATA_UNWRITTEN '+repr(secondary),flush=True)
        raise
    finally:
        # 只关闭本入口持有的锁描述符；不删除锁文件，也不动其他任务的锁。
        for handle in locks:handle.close()


if __name__=='__main__':main()
