"""固定3327步学生的正式客观评估；四项原评分函数不作修改。"""

import argparse  # 主入口需要另行批准；内部单项进程不提供独立实验身份。
import collections  # 100条源语音每条保留81个转换配对。
import csv
import ctypes  # Linux父进程退出时通知本项评分子进程。
import fcntl  # 子进程继承父进程实际持有的三把文件锁。
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT/'exp/diagnostics/student_performance_tools_20260906'
DEV = ROOT/'exp/diagnostics/student_development_20260906'
OUTPUT = ROOT/'exp/evaluation/fastvoicegrad_text_unseen_s003327_custom_hifigan_20260907'
METRIC_DIR = OUTPUT/'metrics'
INFER = TOOLS/'infer_formal_student3327_v1.py'
INFER_SHA = '1699a7961306244a3df73f02ce277276ed9d6f217c8f7fe5c6985e84a6482f32'  # 单向绑定已审核推理入口，避免依赖成环。
EVALUATOR = ROOT/'evaluation/evaluate_metrics.py'
EVALUATOR_SHA = '9797b724c7296517a1e0b31a74dbe05e6f06454669902b351e42caea2cba42ab'
PROTOCOL = DEV/'FORMAL3327_EVALUATION_PROTOCOL_CN.md'
PROTOCOL_SHA = '46588b6a4aa71e401769e4aa7131565091497626c714ecfcdf3e8c89762b043a'
RUN = DEV/'acdd_s5e5_staged120_from3227_v1'
STUDENT_SHA = '7bb68a0d053335a87a2c3c42684b864e27bf21a9f19d6f8d1b5bc8152955ca60'
FULL_SHA = '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556'
CONTRACT_SHA = '32651a26cdf7074a9e8ca30c3cdd099697541fa4b261df757d2866f9e0622aba'
CONTRACT_FILE_SHA = '787759339c886b2918bdda62abcc6818adf3cb410f923aca38c80bbd4b99b071'
HEAD = '0a2c646922704e746642f5f33c203869fe696b3f'
PYTHON = Path('/root/miniconda3/envs/voicegrad-eval/bin/python')
DNSMOS_SOURCE = PYTHON.parent.parent/'lib/python3.9/site-packages/torchmetrics/functional/audio/dnsmos.py'
DNSMOS_SHA = '921a393586987e4344da0ccc264d325f63fc2a211a5c099d922b5b1a5acfc8ae'
TEACHER = ROOT/'exp/evaluation/voicegrad30_text_unseen_epoch0500_custom_hifigan'
LEGACY = ROOT/'exp/evaluation/fastvoicegrad_text_unseen_epoch0100_custom_hifigan'
AUXILIARY_RECEIPT = DEV/'evaluation_candidate_student_acdd_s5e5_s3327/metric_identity.json'
AUXILIARY_RECEIPT_SHA = 'c0ff0e76a25b5656f1b2c110a2a6664342b03d8cf244971a8590074c75a37d86'
PINNED = {
    str(EVALUATOR): EVALUATOR_SHA, str(PROTOCOL): PROTOCOL_SHA, str(DNSMOS_SOURCE): DNSMOS_SHA,
    str(RUN/'student_step_003327.pt'): STUDENT_SHA, str(RUN/'final_full.pt'): FULL_SHA,
    str(RUN/'run_contract.json'): CONTRACT_FILE_SHA,
    str(AUXILIARY_RECEIPT):AUXILIARY_RECEIPT_SHA,
    str(TEACHER/'metrics/objective_summary.json'): '96d5c01990541f2d886a79bdcc8abae3eb7238cacebc20a93ec960fe093ffb9d',
    str(TEACHER/'metrics/metrics_run_identity.json'): '5b6e5219190258673b29e8139d3b57a39fcf7b73d179d3108285eb23a7d25d66',
    str(TEACHER/'inference_report.json'): '16c993c9f1f659d2f9d2dcd2ecb1c7ebda67fc36f1cb06dea2f5461e929f55d0',
    str(TEACHER/'pairs.csv'): '7b0f578d3c6fd83005693d16d5508b41b951a4c6bded72a00a63ec278363ccec',
    str(LEGACY/'metrics/objective_summary.json'): '315be70e4b54c1ffebb04047870f5a55710b58ef063675f4bfb8f49f4a5c8a69',
    str(LEGACY/'metrics/metrics_run_identity.json'): '34ebc8a9e31c968f3d8072131806ac28855c9190162e4ee1da0bf6f30035509c',
    str(LEGACY/'inference_report.json'): 'a8486c59b7dc163a752bc292d62f51004576ed90cf54ef0aaee828a90f8938d9',
    str(LEGACY/'pairs.csv'): '49b51bbeec21b9e18d360774661f1664418ef61f3989f489228dca8a449b24e3',
}
VERSIONS = {'torch':'2.6.0+cu124', 'torchaudio':'2.6.0+cu124', 'speechbrain':'1.0.3',
    'torchmetrics':'1.5.2', 'pytorch-lightning':'1.5.10', 'onnxruntime':'1.19.2', 'librosa':'0.11.0'}
METRICS = ('utmos','cer','sva','dnsmos')  # 与旧正式入口同序，DNSMOS保持3个CPU进程。
COLUMNS = ('CER_corpus_percent','SVA_percent','DNSMOS_OVRL','UTMOS','DNSMOS_SIG','DNSMOS_BAK','DNSMOS_P808','SVA_cosine')
HEADLINES = ('CER_corpus_percent_lower_is_better','SVA_percent_higher_is_better',
    'DNSMOS_mos_ovr_mean_higher_is_better','UTMOS_mean_higher_is_better')
PAPER = {'UTMOS':3.96,'DNSMOS_OVRL':3.77,'CER_corpus_percent':1.89,'SVA_percent':83.0}
RESERVE = 3*1024**3+128*1024**2  # 评分阶段同样保留系统空间和固定缓冲。
METADATA_BUDGET = 32*1024**2  # 与推理共享元数据/两个显式日志的总额度。
DIRECTORY_BUDGET = 1024**2
LOCK_NAMES = ('text_unseen_custom_hifigan_pipeline.lock','text_unseen_custom_hifigan_train.lock','text_unseen_fastvoicegrad_student.lock')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2), b''): value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_files(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected: raise RuntimeError('Input/source SHA changed: '+str(path))


def merge_pins(target, extra):
    # 多份契约不能用后来读到的值覆盖已经钉住的权威SHA。
    for path,value in extra.items():
        if path in target and target[path]!=value:raise RuntimeError('Conflicting pinned SHA: '+path)
        target[path]=value


def atomic_json(path, value):
    # 只覆盖本次可恢复状态；完成摘要已存在时须先核相同身份。
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    if path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise RuntimeError('Uncertified or redirected metadata requires manual review: '+str(path))
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,path)


def load_evaluator():
    # 导入不会启动任何指标模型；仅正式worker调用四个evaluate函数。
    verify_files({str(EVALUATOR):EVALUATOR_SHA})
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    return importlib.import_module('evaluation.evaluate_metrics')


def load_inference():
    verify_files({str(INFER):INFER_SHA})
    if str(TOOLS) not in sys.path: sys.path.insert(0,str(TOOLS))
    return importlib.import_module('infer_formal_student3327_v1')


def check_runtime():
    if not sys.platform.startswith('linux') or not ROOT.is_dir() or Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError('Only the original Matpool voicegrad-eval interpreter is allowed')
    actual = {name:importlib.metadata.version(name) for name in VERSIONS}
    if platform.python_version()!='3.9.23' or actual!=VERSIONS:
        raise RuntimeError('Original scoring software versions changed: '+repr(actual))
    return dict(actual,python=platform.python_version())


def scoring_precision(m):
    # 记录评分环境现场设置，不改动它们，也不捏造旧教师未保存的精度收据。
    return {'cuda_matmul_allow_tf32':m.torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':m.torch.backends.cudnn.allow_tf32,'cudnn_benchmark':m.torch.backends.cudnn.benchmark,
        'cudnn_deterministic':m.torch.backends.cudnn.deterministic}


def metadata_usage():
    # WAV已生成且由推理验真；每秒巡检不再读取其音频内容，也不重复统计其分配量。
    used=directories=0
    for folder,subdirs,names in os.walk(OUTPUT):
        path=Path(folder)
        if path.is_symlink():raise RuntimeError('Redirected metadata directory')
        directories+=path.stat().st_blocks*512
        for name in names:
            item=path/name
            if item.is_symlink():raise RuntimeError('Redirected evaluation artifact')
            if not name.endswith('.wav'):used+=item.stat().st_blocks*512
    for suffix in ('.log','.metrics.log'):
        log=OUTPUT.parent/(OUTPUT.name+suffix)
        if log.exists():
            if log.is_symlink():raise RuntimeError('Redirected evaluation log')
            used+=log.stat().st_blocks*512
    return used,directories


def required_bytes(used, directories):
    if not 0<=used<=METADATA_BUDGET or not 0<=directories<=DIRECTORY_BUDGET:
        raise RuntimeError('Shared32MiB metadata/log or1MiB directory budget exceeded')
    return RESERVE+(METADATA_BUDGET-used)+(DIRECTORY_BUDGET-directories)


def capacity():
    stat = os.statvfs(OUTPUT)
    available = stat.f_bavail*stat.f_frsize
    needed=required_bytes(*metadata_usage())
    if available<needed: raise RuntimeError('Scoring stopped: system 3GiB +128MiB and remaining metadata budget are unavailable')
    return available


def require_pairs(rows, teacher_rows):
    # 输出位置可以改变，其余12列身份、顺序、源文字和参考必须相同。
    if len(rows)!=8100 or [r.get('pair_id') for r in rows]!=[str(i) for i in range(8100)]:
        raise RuntimeError('Require complete ordered8100 pairs')
    if len(teacher_rows)!=8100: raise RuntimeError('The frozen teacher must contain8100 pairs')
    for row, old in zip(rows,teacher_rows):
        if {k:v for k,v in row.items() if k!='output_wav'}!={k:v for k,v in old.items() if k!='output_wav'}:
            raise RuntimeError('Pair identity differs from formal teacher at '+row['pair_id'])
    sources = collections.Counter((r['source_speaker'],r['source_utt']) for r in rows)
    if len(sources)!=100 or set(sources.values())!={81}: raise RuntimeError('Require100 source clusters of81 pairs')


def validate_lineage(bundle):
    report = bundle['report']; rows = bundle['rows']
    expected = {'status':'formal_8100_complete','system_id':'fastvoicegrad_student3327','model_role':'student',
        'student_global_step':3327,'student_steps':1,'student_start_step':950,'teacher_epoch':500,
        'selected_pair_count':8100,'paper_pair_count':8100,'student_checkpoint_sha256':STUDENT_SHA,
        'student_run_contract_sha256':CONTRACT_SHA}
    for key,value in expected.items():
        if report.get(key)!=value: raise RuntimeError('Wrong true3327 inference field: '+key)
    contract=read_json(RUN/'run_contract.json')
    if canonical(contract)!=CONTRACT_SHA or sha(RUN/'run_contract.json')!=CONTRACT_FILE_SHA:
        raise RuntimeError('The actual3327 training contract changed')
    if report.get('parent_full_checkpoint_sha256')!=FULL_SHA:
        raise RuntimeError('The inference must cite the actual3327 complete parent')
    m=load_evaluator(); teacher_rows=m.load_pairs(TEACHER/'pairs.csv',0,None)
    require_pairs(rows,teacher_rows)
    if sum(len(m.normalize_cer_text(r['source_text'])) for r in rows)!=566190:
        raise RuntimeError('Original formal CER denominator must be566190')
    wav_hashes=m.load_generated_wav_hash_manifest(report,rows)
    m.validate_generated_wavs(rows,wav_hashes)
    if bundle['report_sha256']!=sha(OUTPUT/'inference_report.json'):
        raise RuntimeError('Inference report changed while validating')
    return rows


def scoring_assets(m):
    teacher=read_json(TEACHER/'metrics/objective_summary.json')
    actual=m.metric_asset_manifest(ROOT/'UTMOS-demo',ROOT/'pretrained_spkrec_ecapa')
    if set(actual)!=set(teacher['metric_assets']) or len(actual)!=6 or actual!=teacher['metric_assets']:
        raise RuntimeError('The six original teacher scoring weights changed or are unavailable')
    verify_files({v['path']:v['sha256'] for v in actual.values()})
    verify_files({str(DNSMOS_SOURCE):DNSMOS_SHA})
    if any(teacher['software_versions'][k]!=v for k,v in VERSIONS.items()):
        raise RuntimeError('Historical teacher scoring software is incompatible')
    return actual


def auxiliary_sources(assets):
    # 复用已经完成90对评分的附属源码/ECAPA状态身份，不复用它的开发成绩。
    verify_files({str(AUXILIARY_RECEIPT):AUXILIARY_RECEIPT_SHA})
    prior=read_json(AUXILIARY_RECEIPT);aux=prior['auxiliary_source_sha256']
    if prior['assets']!=assets or prior['software_versions']!=VERSIONS:
        raise RuntimeError('Existing auxiliary receipt uses another scorer')
    paths={str(DNSMOS_SOURCE)}
    paths.update(str(ROOT/'pretrained_spkrec_ecapa'/name) for name in ('classifier.ckpt','label_encoder.ckpt','mean_var_norm_emb.ckpt'))
    for directory in (ROOT/'UTMOS-demo',ROOT/'pretrained_spkrec_ecapa'):
        paths.update(str(p) for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json','.txt') and '.git' not in p.parts)
    if set(aux)!=paths or len(aux)!=11:raise RuntimeError('UTMOS/ECAPA auxiliary files were added, removed or replaced')
    verify_files(aux)
    return dict(aux)


def schema(m):
    return {'utmos':m.UTMOS_FIELDS,'dnsmos':m.DNSMOS_FIELDS,'cer':m.CER_FIELDS,'sva':m.SVA_FIELDS}


def metric_rows(m, directory, rows, complete=True):
    # 生产校验器同时重算CER编辑距离、阈值和续算前缀，不能跳过坏尾行。
    result={name:m.validate_existing_metric_csv(directory/(name+'.csv'),rows,name,fields) for name,fields in schema(m).items()}
    if complete and any(len(value)!=8100 for value in result.values()): raise RuntimeError('All four metrics must have8100 rows')
    for pair,cer in zip(rows,result['cer']):
        if cer['reference_text_normalized']!=m.normalize_cer_text(pair['source_text']):
            raise RuntimeError('CER reference differs from the fixed source transcript')
    return result


def sums_by_source(rows, metrics):
    # 每簇统计充分量，CER始终用错误总数/字符总数而非平均逐句百分比。
    import numpy as np
    clusters=sorted({(r['source_speaker'],r['source_utt']) for r in rows})
    index={key:i for i,key in enumerate(clusters)}; result=np.zeros((len(clusters),10),dtype=np.float64)
    for i,pair in enumerate(rows):
        cer,dns,ut,sv=(metrics[k][i] for k in ('cer','dnsmos','utmos','sva'))
        values=[int(cer['char_errors']),int(cer['reference_chars']),int(sv['verified']),float(dns['mos_ovr']),float(ut['utmos']),
            float(dns['mos_sig']),float(dns['mos_bak']),float(dns['p808_mos']),float(sv['cosine_score']),1]
        result[index[(pair['source_speaker'],pair['source_utt'])]]+=values
    if not np.isfinite(result).all(): raise RuntimeError('Nonfinite source totals')
    return clusters,result


def values_from_sums(sums):
    import numpy as np
    a=np.asarray(sums,dtype=np.float64)
    if np.any(a[...,1]<=0) or np.any(a[...,9]<=0): raise RuntimeError('Nonpositive CER or pair denominator')
    return np.stack([100*a[...,0]/a[...,1],100*a[...,2]/a[...,9],a[...,3]/a[...,9],a[...,4]/a[...,9],
        a[...,5]/a[...,9],a[...,6]/a[...,9],a[...,7]/a[...,9],a[...,8]/a[...,9]],axis=-1)


def historical(m, directory, current_rows, files):
    summary=read_json(directory/'metrics/objective_summary.json');identity=read_json(directory/'metrics/metrics_run_identity.json')
    rows=m.load_pairs(directory/'pairs.csv',0,None);require_pairs(current_rows,rows)
    if summary['status']!='formal_8100_complete' or summary['evaluated_pairs']!=8100: raise RuntimeError('Incomplete historical metric baseline')
    if summary['pairs_csv_sha256']!=sha(directory/'pairs.csv') or summary['inference_report_sha256']!=sha(directory/'inference_report.json'):
        raise RuntimeError('Historical summary is not bound to its actual inference')
    if identity['pairs_csv_sha256']!=summary['pairs_csv_sha256'] or identity['inference_report_sha256']!=summary['inference_report_sha256']:
        raise RuntimeError('Historical metric receipt identity differs')
    teacher=read_json(TEACHER/'metrics/objective_summary.json')
    if summary['metric_assets']!=teacher['metric_assets'] or summary['software_versions']!=teacher['software_versions']:
        raise RuntimeError('Historical scoring assets/software differ from the same-condition teacher')
    merge_pins(files,{str(directory/'metrics'/(name+'.csv')):value for name,value in summary['individual_metric_csv_sha256'].items()})
    verify_files(files)
    metrics=metric_rows(m,directory/'metrics',rows)
    clusters,sums=sums_by_source(rows,metrics);point=values_from_sums(sums.sum(0))
    for i,key in enumerate(HEADLINES):
        if not math.isclose(float(point[i]),summary['headline_metrics'][key],rel_tol=1e-12,abs_tol=1e-12):
            raise RuntimeError('Historical headline differs from its8100 actual rows: '+key)
    extra=summary['additional_diagnostics']
    if int(sums[:,0].sum())!=extra['CER_total_character_errors'] or int(sums[:,1].sum())!=566190:
        raise RuntimeError('Historical CER integer counts changed')
    return clusters,sums,summary


def compare_systems(systems, clusters):
    # 一份10000×100整数抽样表留在内存，三系统和八列共用；不生成大型归档。
    import numpy as np
    rng=np.random.Generator(np.random.PCG64(202609073327))
    draws=rng.integers(0,len(clusters),size=(10000,len(clusters)))
    weights=np.stack([(draws==i).sum(1) for i in range(len(clusters))],axis=1)
    points={k:values_from_sums(v.sum(0)) for k,v in systems.items()}
    boots={k:values_from_sums(weights@v) for k,v in systems.items()}
    estimates={}; differences={}; per_source={}
    for label,sums in systems.items():
        if len(clusters)!=100 or sums.shape!=(100,10) or not np.all(sums[:,9]==81) or sums[:,1].sum()!=566190:
            raise RuntimeError('All formal100 source groups and566190 characters are required')
        ci=np.percentile(boots[label],[2.5,97.5],axis=0).T
        estimates[label]={k:{'estimate':float(points[label][i]),'paired_source_cluster_95ci':ci[i].tolist()} for i,k in enumerate(COLUMNS)}
        estimates[label]['CER_total_character_errors']=int(sums[:,0].sum())
        estimates[label]['CER_total_reference_characters']=int(sums[:,1].sum())
        per_source[label]=[{'source_speaker':c[0],'source_utt':c[1],'pairs':81,'char_errors':int(sums[i,0]),
            'reference_chars':int(sums[i,1]),**dict(zip(COLUMNS,values_from_sums(sums[i]).tolist()))} for i,c in enumerate(clusters)]
    for reference in ('teacher500','legacy_student100'):
        ci=np.percentile(boots['student3327']-boots[reference],[2.5,97.5],axis=0).T
        differences['student3327_minus_'+reference]={k:{'delta':float(points['student3327'][i]-points[reference][i]),
            'paired_source_cluster_95ci':ci[i].tolist(),'favorable_sign':'negative' if i==0 else 'positive'} for i,k in enumerate(COLUMNS)}
    candidate=dict(zip(COLUMNS,points['student3327'].tolist())); teacher=dict(zip(COLUMNS,points['teacher500'].tolist()))
    # 数值门槛逐项报告；DNSMOS论文版本不明和训练趋势条件独立保留，不能用CI跨0宣称相等。
    thresholds={reference:{key:{'candidate':candidate[key],'reference':value,'candidate_minus_reference':candidate[key]-value,
        'numerical_threshold_met':candidate[key]<=value if key=='CER_corpus_percent' else candidate[key]>=value}
        for key,value in target.items()} for reference,target in [('same_condition_teacher',{k:teacher[k] for k in ('UTMOS','DNSMOS_OVRL','CER_corpus_percent')}),('paper_fastvoicegrad',PAPER)]}
    return {'systems':estimates,'paired_comparisons':differences,'per_source':per_source,'threshold_checks':thresholds,
        'bootstrap':{'rng':'numpy.PCG64','seed':202609073327,'repetitions':10000,'source_clusters':100,'pairs_per_cluster':81,
            'weight_matrix_sha256':hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest(),'numpy_version':np.__version__,
            'multiple_comparisons_adjusted':False,'targets_conditioned_on_not_resampled':True}}


def bind_identity(path, identity):
    # 无身份的旧CSV不能续算；身份不同的评分必须拒绝，不覆盖已有结果。
    if path.exists():
        if read_json(path)!=identity: raise RuntimeError('Existing metrics belong to another source/model/input identity')
    else:
        if any(path.parent.glob('*.csv')) or (path.parent/'objective_summary.json').exists():
            raise RuntimeError('Metric artifacts exist without their identity')
        atomic_json(path,identity)


def frozen_identity(identity, path):
    # 后续比较可扩展最终来源表，但已绑定CSV的启动身份必须保持原字节语义。
    value=canonical(identity)
    if read_json(path)!=identity:raise RuntimeError('Published metric identity differs from the in-memory identity')
    return value


def stop_child(process):
    # 仅终止本入口刚创建的进程组，DNSMOS子进程同组；不触碰其他任务。
    if process is None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def supervised_metric(name, identity_sha, locks, deadline):
    # 原评分循环不注入callback；父进程独立监视空间/总时限，并保留合法CSV前缀。
    capacity()
    args=[str(PYTHON),'-u',str(Path(__file__).resolve()),'--worker',name,'--parent-pid',str(os.getpid()),
        '--lock-fds',','.join(str(h.fileno()) for h in locks),'--identity-sha',identity_sha]
    environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    process=subprocess.Popen(args,cwd=str(ROOT),start_new_session=True,pass_fds=tuple(h.fileno() for h in locks),env=environment)
    try:
        while process.poll() is None:
            capacity()
            if time.monotonic()>deadline:raise TimeoutError('Formal scoring reached its3-hour budget')
            time.sleep(1)
        if process.returncode:raise RuntimeError('Metric '+name+' exited '+str(process.returncode))
        capacity()
    finally:stop_child(process)


def verify_inherited_locks(fds, parent_pid):
    # 子进程只能从当前主进程及其实际锁描述符启动，不能单独绕过整个验真入口。
    if os.getppid()!=parent_pid or len(fds)!=3: raise RuntimeError('Scoring worker must have the live declared parent and3 inherited locks')
    for fd,name in zip(fds,LOCK_NAMES):
        expected=(ROOT/'exp/locks'/name).resolve()
        if Path(os.readlink('/proc/self/fd/'+str(fd))).resolve()!=expected:raise RuntimeError('Wrong inherited workflow lock')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent_pid:
        raise RuntimeError('Cannot bind scoring worker lifetime to parent')


def worker(args):
    check_runtime();verify_inherited_locks([int(x) for x in args.lock_fds.split(',')],args.parent_pid)
    if any(os.environ.get(k)!='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')):
        raise RuntimeError('Metric workers must use existing offline scoring assets')
    def interrupted(signum,frame):
        # 原函数的finally有机会关闭CSV；若DNSMOS进程池未及时退出，父进程仍会终止整组。
        raise KeyboardInterrupt('Metric worker interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    identity=read_json(METRIC_DIR/'metrics_run_identity.json')
    if canonical(identity)!=args.identity_sha:raise RuntimeError('Worker has a different scoring identity')
    verify_files(identity['file_sha256']);capacity()
    m=load_evaluator();bundle=load_inference().validate_complete_output(OUTPUT,verify_dependencies=True)
    rows=validate_lineage(bundle);auxiliary_sources(scoring_assets(m))
    if scoring_precision(m)!=identity['scoring_precision_flags']:raise RuntimeError('Worker scoring precision differs from its parent')
    if bundle['report_sha256']!=identity['inference_report_sha256']:raise RuntimeError('Worker inference identity changed')
    import torch
    device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('The original CUDA scoring device is unavailable')
    destination=METRIC_DIR/(args.worker+'.csv')
    if args.worker=='utmos':m.evaluate_utmos(rows,destination,ROOT/'UTMOS-demo',device,4)
    elif args.worker=='cer':m.evaluate_cer(rows,destination,device,4)
    elif args.worker=='sva':m.evaluate_sva(rows,destination,ROOT/'pretrained_spkrec_ecapa',device,4)
    elif args.worker=='dnsmos':m.evaluate_dnsmos(rows,destination,4,3)
    else:raise RuntimeError('Unknown internal metric')
    if scoring_precision(m)!=identity['scoring_precision_flags']:raise RuntimeError('Scoring precision changed during the metric')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=10800)
    parser.add_argument('--worker',choices=METRICS,help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid',type=int,default=-1,help=argparse.SUPPRESS)
    parser.add_argument('--lock-fds',default='',help=argparse.SUPPRESS)
    parser.add_argument('--identity-sha',default='',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:return worker(args)
    if not args.run_approved or not 1<=args.max_seconds<=10800:raise RuntimeError('Separate approval and at most3 hours are required')
    versions=check_runtime();m=load_evaluator()
    from fastvoicegrad import train as training
    training.assert_rented_server(ROOT)
    locks=training.acquire_runtime_locks(ROOT)
    if training.git_head(ROOT)!=HEAD or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Frozen production source must remain clean')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU job is still running')
    started=time.monotonic();deadline=started+args.max_seconds
    def interrupted(signum,frame):raise KeyboardInterrupt('Scoring interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    files=dict(PINNED);files[str(INFER)]=INFER_SHA;files[str(Path(__file__).resolve())]=sha(Path(__file__))
    verify_files(files);capacity()
    bundle=load_inference().validate_complete_output(OUTPUT,verify_dependencies=True)
    rows=validate_lineage(bundle);assets=scoring_assets(m)
    merge_pins(files,{v['path']:v['sha256'] for v in assets.values()})
    merge_pins(files,auxiliary_sources(assets))
    for name in ('pairs.csv','inference_report.json','run_identity.json'):files[str(OUTPUT/name)]=sha(OUTPUT/name)
    # 推理端已经核全400特征及8100WAV；此处保留完整依赖索引而非仅口头声明。
    merge_pins(files,bundle['identity']['file_sha256'])
    identity={'schema_version':1,'system_id':'fastvoicegrad_student3327','student_global_step':3327,
        'student_checkpoint_sha256':STUDENT_SHA,'parent_full_sha256':FULL_SHA,'student_run_contract_sha256':CONTRACT_SHA,
        'inference_report_sha256':bundle['report_sha256'],'inference_identity_sha256':bundle['identity_sha256'],
        'pairs_csv_sha256':sha(OUTPUT/'pairs.csv'),'selected_pair_count':8100,'batch_size':4,'dnsmos_workers':3,
        'metric_assets':assets,'software_versions':versions,'scoring_precision_flags':scoring_precision(m),
        'file_sha256':dict(files),'generated_wav_collection_sha256':bundle['report']['generated_wav_collection_sha256'],
        'protocol_sha256':PROTOCOL_SHA,'bootstrap_seed':202609073327,'bootstrap_repetitions':10000,
        'offline_environment':{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}}
    if METRIC_DIR.is_symlink():raise RuntimeError('Metrics directory cannot be a symlink')
    METRIC_DIR.mkdir(exist_ok=True)
    bind_identity(METRIC_DIR/'metrics_run_identity.json',identity)
    identity_sha=frozen_identity(identity,METRIC_DIR/'metrics_run_identity.json')
    metric_rows(m,METRIC_DIR,rows,complete=False)
    try:
        for name in METRICS:
            capacity()
            if time.monotonic()>deadline:raise TimeoutError('Formal scoring reached its3-hour budget')
            print('FORMAL3327_METRIC_START '+name,flush=True)
            supervised_metric(name,identity_sha,locks,deadline)
            completed=m.validate_existing_metric_csv(METRIC_DIR/(name+'.csv'),rows,name,schema(m)[name])
            if len(completed)!=8100:raise RuntimeError('Metric worker did not complete8100 rows: '+name)
            atomic_json(METRIC_DIR/'progress.json',{'status':'partial_metrics_complete','last_completed_metric':name,
                'identity_sha256':identity_sha,'elapsed_seconds':time.monotonic()-started})
        current=metric_rows(m,METRIC_DIR,rows);clusters,sums=sums_by_source(rows,current)
        systems={'student3327':sums};historical_summaries={}
        for label,directory in [('teacher500',TEACHER),('legacy_student100',LEGACY)]:
            old_clusters,systems[label],historical_summaries[label]=historical(m,directory,rows,files)
            if old_clusters!=clusters:raise RuntimeError('Historical source clusters differ')
        comparison=compare_systems(systems,clusters)
        final_bundle=load_inference().validate_complete_output(OUTPUT,verify_dependencies=True)
        if final_bundle['report_sha256']!=bundle['report_sha256'] or final_bundle['identity_sha256']!=bundle['identity_sha256']:
            raise RuntimeError('Inference changed during metrics')
        auxiliary_sources(scoring_assets(m));verify_files(files);verify_files(PINNED);verify_files({str(INFER):INFER_SHA});capacity()
        if frozen_identity(identity,METRIC_DIR/'metrics_run_identity.json')!=identity_sha:
            raise RuntimeError('The metric startup identity changed during comparison')
        if time.monotonic()>deadline:raise TimeoutError('Formal scoring reached its3-hour budget before publication')
        metric_sha={k:sha(METRIC_DIR/(k+'.csv')) for k in METRICS}
        headline={name:comparison['systems']['student3327'][COLUMNS[i]]['estimate'] for i,name in enumerate(HEADLINES)}
        result={'status':'formal_8100_complete','system_id':'fastvoicegrad_student3327','system_label':'FastVoiceGrad学生3327步',
            'model_role':'student','student_global_step':3327,'student_epoch':bundle['report']['student_epoch'],
            'student_epoch_semantics':'Actual development checkpoint cursor; not100 completed epochs.',
            'student_steps':1,'student_start_step':950,'teacher_epoch':500,
            'student_checkpoint_sha256':STUDENT_SHA,'parent_full_sha256':FULL_SHA,'student_run_contract_sha256':CONTRACT_SHA,
            'vocoder_kind':bundle['report']['vocoder_kind'],'vocoder_checkpoint_sha256':bundle['report']['vocoder_checkpoint_sha256'],
            'vocoder_global_step':bundle['report']['vocoder_global_step'],
            'headline_metrics':headline,'additional_diagnostics':{
                'CER_total_character_errors':int(sums[:,0].sum()),'CER_total_reference_characters':int(sums[:,1].sum()),
                'SVA_threshold':0.25,'SVA_mean_cosine_score':comparison['systems']['student3327']['SVA_cosine']['estimate']},
            'evaluated_pairs':8100,'metric_order':list(COLUMNS),'individual_metric_csv_sha256':metric_sha,
            'metrics_run_identity_sha256':identity_sha,'metric_assets':assets,'software_versions':versions,
            'metrics_run_identity_file_sha256':sha(METRIC_DIR/'metrics_run_identity.json'),
            'scoring_precision_flags':identity['scoring_precision_flags'],
            'file_sha256':files,'comparison':comparison,'elapsed_seconds':time.monotonic()-started,
            'limitations':['This is the single preregistered actual3327 checkpoint, not either temporary3328 candidate.',
                'Student LR5e-5 and3327 updates differ from the paper LR2e-4 and100 epochs; prior reproduction limitations remain.',
                'Earlier52-utterance development-driven selection is disclosed; full8100 results do not select another checkpoint.',
                'The authors did not publish the exact DNSMOS version; our OVRL cannot establish equivalence to paper3.77.',
                'Shared100-source paired percentile intervals condition on fixed targets and do not capture every shared-speaker/reference dependency; multiple comparisons are unadjusted.',
                'Historical teacher has no complete saved per-WAV/noise hash receipt; historical metric CSV and inference identities remain frozen.',
                'Historical teacher scoring did not record the four backend precision flags; current scoring flags are recorded without claiming bytewise historical verification.',
                'Objective performance does not certify monotonic or stable long-term training loss; no new training is authorized.']}
        atomic_json(METRIC_DIR/'objective_summary.json',result)
        print('FORMAL3327_ALL4_COMPLETE '+str(METRIC_DIR/'objective_summary.json'),flush=True)
    except BaseException as error:
        state={'status':'failed_or_interrupted_no_complete_claim','error_type':type(error).__name__,'error':str(error),
            'identity_sha256':identity_sha,'elapsed_seconds':time.monotonic()-started,
            'existing_rows_preserved':'Every existing CSV row is retained; torn/malformed rows require manual review and cannot be silently dropped or certified.'}
        print('FORMAL3327_FAILED '+json.dumps(state),flush=True)
        try:atomic_json(METRIC_DIR/'failure.json',state)
        except Exception as secondary:print('FORMAL3327_FAILURE_METADATA_UNWRITTEN '+repr(secondary),flush=True)
        raise


if __name__=='__main__':main()
