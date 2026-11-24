"""为唯一选定的3327步学生生成原8100对语音；不训练，不更改旧正式入口。"""

import argparse  # 不提供任意checkpoint、种子或配对切片选项。
import csv
import gc  # 权重加载后释放原CPU字典，不另存或复制模型文件。
import hashlib
import io
import json
import math
import os
import platform
import signal  # 收到停止请求后保留可核验状态，不自动继续下一批。
import stat
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path('/root/FsatVoiceGrad')
if os.name != 'posix' or not ROOT.is_dir():
    raise RuntimeError('Read and run this entry only on the registered Matpool rental')
TOOLS = ROOT/'exp/diagnostics/student_performance_tools_20260906'
DEV = ROOT/'exp/diagnostics/student_development_20260906'
RUN = DEV/'acdd_s5e5_staged120_from3227_v1'
OUTPUT = ROOT/'exp/evaluation/fastvoicegrad_text_unseen_s003327_custom_hifigan_20260907'
STUDENT = RUN/'student_step_003327.pt'
STUDENT_SHA = '7bb68a0d053335a87a2c3c42684b864e27bf21a9f19d6f8d1b5bc8152955ca60'
FULL = RUN/'final_full.pt'
FULL_SHA = '11a4705b4f7fc3b493cce69377800893f096de1d08292afd95fb38a6d2592556'
CONTRACT = RUN/'run_contract.json'
CONTRACT_SHA = '32651a26cdf7074a9e8ca30c3cdd099697541fa4b261df757d2866f9e0622aba'
CONTRACT_FILE_SHA = '787759339c886b2918bdda62abcc6818adf3cb410f923aca38c80bbd4b99b071'
TEACHER_DIR = ROOT/'exp/evaluation/voicegrad30_text_unseen_epoch0500_custom_hifigan'
TEACHER_PAIRS = TEACHER_DIR/'pairs.csv'
TEACHER_PAIRS_SHA = '7b0f578d3c6fd83005693d16d5508b41b951a4c6bded72a00a63ec278363ccec'
OLD_IDENTITY = ROOT/'exp/evaluation/fastvoicegrad_text_unseen_epoch0100_custom_hifigan/run_identity.json'
VOCODER_DIR = ROOT/'exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen'
VOCODER = VOCODER_DIR/'generator_step_2500000.pt'
VOCODER_SHA = 'dacd362c4db46cdf7bd8428ace6f38d7d03976c60a3cd4c55022d38383094fb9'
ATTESTATION = VOCODER_DIR/'final_checkpoint_attestation_comments_20260906.json'
PROTOCOL = DEV/'FORMAL3327_EVALUATION_PROTOCOL_CN.md'
PROTOCOL_SHA = '46588b6a4aa71e401769e4aa7131565091497626c714ecfcdf3e8c89762b043a'
EXPECTED_HEAD = '0a2c646922704e746642f5f33c203869fe696b3f'
SYSTEM = 'fastvoicegrad_student3327'
SYSTEM_LABEL = 'FastVoiceGrad学生3327步'
PAIR_COUNT, BATCH_SIZE, SEED = 8100, 4, 1234
RESERVE = 3*1024**3 + 128*1024**2
METADATA_BUDGET, DIRECTORY_BUDGET = 32*1024**2, 1024**2
EXPECTED_WAV_BYTES, EXPECTED_WAV_ALLOCATED = 1972847664, 1991651328
EXPECTED_START_FREE = 5382123520
TEMPORARY_BUDGET = 425984  # 实际最长WAV分配421888B；沿用事先多留一块的预算，不下调启动门槛。
FLAGS = {'cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':True,'cudnn_benchmark':False,'cudnn_deterministic':False}
EXPECTED_RUNTIME = {'python':'3.8.15','torch':'1.13.1+cu116','cuda':'11.6',
                    'executable':'/root/miniconda3/envs/myconda/bin/python3.8','precision_flags':FLAGS}
SOURCE_SHA = {
 'fastvoicegrad/infer_paper_eval.py':'2b27245fb062fb7fae8d82492fe32ef868d9c0f4441b7b92945c30b91b8c30dc',
 'voicegrad/infer_voicegrad30_paper_eval.py':'d885f9d4dec7fa35d4c3ae7fb1e102fea0373a4dd57d734ebb099c005518a9af',
 'fastvoicegrad/components.py':'aa15c81b2c0b1e614db997df52ca97c2fa494434a0f10b84d317fe2669583b5f',
 'voicegrad/train.py':'b63dc72ed17ef4196a680f4f58263f88345be43254293b5cdb9c531aec71cbe2',
 'voicegrad/model.py':'5a533381bdfebf1b72f3cc284de24c8ffbd97d7f10d74bbae4b89e7013a6d771',
 'voicegrad/diffusion.py':'c89d7bcee03c5dfbdf967eadd436941da7c28b2c2f9c3db5c482ea9dc3c8bf57',
 'voicegrad/feature_alignment.py':'9bedea5455a118281337ebf72fe50024098b9b3161b32d0cc9efc6ced916109f',
 'voicegrad/config.py':'92ad2e7637156984a0696ace0567427cc8e8d8e4191e193fdc68986bc155658c',
 'fastvoicegrad/objectives.py':'5b1439550205ada6b4f6fca52010963df23f65f28bb185684a912f2c1e9bf58f',
 'fastvoicegrad/config.py':'c4b161f89839b22dcd6711f39b29231a4dbd194b6021990e2dff23967eebc81a',
 'fastvoicegrad/train.py':'d38131843b0b1ba51a10b65bc8bea226c2cecd1ec339d2bb04db5ede2febbaec',
 'hifigan/tools/vocode.py':'e07adb8b269de98d2b0dd6f5c2275b0ce4d0a2e4fd1fd2c140583eee96d50343',
}
PINNED = {str(ROOT/k):v for k,v in SOURCE_SHA.items()}
PINNED.update({str(STUDENT):STUDENT_SHA,str(FULL):FULL_SHA,str(CONTRACT):CONTRACT_FILE_SHA,str(PROTOCOL):PROTOCOL_SHA,
 str(RUN/'completion.json'):'68ee1cf0e680f3bf665be6dcf46e2aa75c3684262c2da3765ae2c7beebf931b2',
 str(TEACHER_PAIRS):TEACHER_PAIRS_SHA,
 str(TEACHER_DIR/'inference_report.json'):'16c993c9f1f659d2f9d2dcd2ecb1c7ebda67fc36f1cb06dea2f5461e929f55d0',
 str(TEACHER_DIR/'run_identity.json'):'25595016c093e74d10ef14b9754f5e16a6427fea70170f8eafc369e1d1704119',
 str(OLD_IDENTITY):'1e25c70bf91721cce406a9a07abc6e6424d9f62ca6c43013400aef73d90800ad',
 str(VOCODER):VOCODER_SHA,str(VOCODER_DIR/'config.json'):'4b570fee140ffbda92bb217218792dca069b2d73627bc5e7bf7b1669869dd05a',
 str(ATTESTATION):'2112690afbe6fc2f7a5325d106c8dcfdd4503d7cfb09fee7675712d34b46205e'})


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def verify_files(records):
    for name,wanted in records.items():
        if not Path(name).is_file() or digest(name)!=wanted:
            raise RuntimeError('Frozen file identity mismatch: '+str(name))


# 导入前核验关键推理源码；没有调用旧的checkpoint身份入口或修改其常量。
verify_files({str(ROOT/k):v for k,v in SOURCE_SHA.items()})
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
import torch
from fastvoicegrad import infer_paper_eval as formal
from fastvoicegrad import train as training
from fastvoicegrad.components import load_frozen_vocoder, load_torch_checkpoint, load_mel_statistics, set_requires_grad, assert_frozen
from fastvoicegrad.config import PAPER_STUDENT_CONFIG
from voicegrad import infer_voicegrad30_paper_eval as teacher_infer
from voicegrad.diffusion import VoiceGradDiffusion
from voicegrad.feature_alignment import align_bnf_to_mel
from voicegrad.train import build_model
from hifigan.tools.vocode import atomic_write_wav


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')


def read_json(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise RuntimeError('Expected a JSON object: '+str(path))
    return value


def csv_rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode('utf-8'),newline='')))


def merge_hashes(destination, source):
    for path,sha in source.items():
        if path in destination and destination[path]!=sha:raise RuntimeError('Conflicting source identity: '+path)
        destination[path]=sha


def validate_payload(payload,contract,completion):
    # epoch0是原开发入口的真实字段；不得为了通过旧总入口改成100。
    if (payload.get('kind'),payload.get('epoch'),payload.get('global_step'),payload.get('run_contract_sha256')) != (
        'acdd_s5e5_staged120_checkpoint',0,3327,CONTRACT_SHA):raise RuntimeError('Require the actual saved3327 student, not epoch100 or a temporary3328 candidate')
    if training.canonical_json_sha256(contract)!=CONTRACT_SHA or payload.get('configuration')!=contract.get('configuration'):
        raise RuntimeError('Student training configuration/contract differs')
    if payload.get('effective_optimizer_lrs')!={'student':.00005,'discriminator':.0002} or payload.get('score_semantics')!='reconstruction_clean_x0_sqrt_alpha_bar':
        raise RuntimeError('Actual3327 optimizer/objective identity differs')
    if completion.get('completed_global_step')!=3327 or completion.get('last_full_saved_global_step')!=3327 or completion.get('partial_batch_may_have_updated_D'):
        raise RuntimeError('Student source did not end at a complete3327 batch')
    if completion['full_snapshot']['sha256']!=FULL_SHA or payload.get('development')!=completion['last_monitor']['development']:
        raise RuntimeError('Student is not bound to the complete3327 source receipt')
    config=replace(PAPER_STUDENT_CONFIG,num_workers=0)
    if config.as_dict()!=payload['configuration'] or (config.student_steps,config.student_start_step,config.bnf_alignment,config.bnf_downsampling)!=(1,950,'legacy_length_linear','legacy_decimate'):
        raise RuntimeError('Inference interface changed from saved3327')
    state=payload.get('student')
    if not isinstance(state,dict) or not state:raise RuntimeError('Student state is missing')
    for name,value in state.items():
        if not isinstance(value,torch.Tensor) or not bool(torch.isfinite(value).all()):raise RuntimeError('Invalid student tensor: '+name)
    return config


def match_teacher_pairs(rows,teacher_rows):
    if len(rows)!=PAIR_COUNT or len(teacher_rows)!=PAIR_COUNT:raise RuntimeError('Require all8100 pairs')
    if [int(r['pair_id']) for r in rows]!=list(range(PAIR_COUNT)):raise RuntimeError('Pair order or uniqueness changed')
    if set(Counter(r['source_utt'] for r in rows).values())!={81} or len({r['source_utt'] for r in rows})!=100:
        raise RuntimeError('Require exactly100 sources with81 conversions each')
    fields=set(teacher_rows[0])
    if fields!=set(rows[0]) or len(fields)!=12:raise RuntimeError('Original pair schema changed')
    for current,old in zip(rows,teacher_rows):
        if any(current[k]!=old[k] for k in fields-{'output_wav'}):raise RuntimeError('Source/reference/text identity changed at pair '+current['pair_id'])


def allocated_size(size,unit):
    if size<0 or unit<=0:raise ValueError('Invalid allocation size')
    return ((int(size)+unit-1)//unit)*unit


def load_plan(verify_dependencies=True):
    # 评分器可以CPU只读调用此函数；实际推理环境检查只在main，不会要求评分Python也为3.8。
    if verify_dependencies:verify_files(PINNED)
    contract,completion=read_json(CONTRACT),read_json(RUN/'completion.json')
    if training.canonical_json_sha256(contract)!=CONTRACT_SHA:raise RuntimeError('Original training contract changed')
    old=read_json(OLD_IDENTITY);teacher_report=read_json(TEACHER_DIR/'inference_report.json')
    sources=teacher_infer.load_test_rows(ROOT/'data/metadata/test_manifest.csv')
    pairs=teacher_infer.build_pairs(sources,OUTPUT)
    pair_bytes=teacher_infer.render_pairs_csv(pairs);rows=csv_rows(pair_bytes)
    match_teacher_pairs(rows,csv_rows(TEACHER_PAIRS.read_bytes()))
    files=dict(PINNED);merge_hashes(files,contract['file_sha256'])
    input_files={}
    for key,sha in old['test_input_file_sha256'].items():
        path=Path(key);path=path if path.is_absolute() else ROOT/path
        if ROOT/'data' not in path.resolve().parents or str(path.resolve()) in input_files:raise RuntimeError('A formal input escaped data or was duplicated')
        input_files[str(path.resolve())]=sha
    if len(input_files)!=400:raise RuntimeError('Require the same400 formal WAV/mel/BNF/speaker inputs')
    merge_hashes(files,input_files)
    for key in ('test_manifest','split_protocol','mel_mean','mel_std'):
        merge_hashes(files,{teacher_report[key]:teacher_report[key+'_sha256']})
    files[str(Path(__file__).resolve())]=digest(Path(__file__))
    if verify_dependencies:
        verify_files(files)
        if formal.collect_test_input_hashes(ROOT,sources)!=old['test_input_file_sha256']:raise RuntimeError('The current400 formal input collection differs')
    lengths={}
    for row in sources:
        path=teacher_infer.feature_path(ROOT/'data','mel',row['speaker'],row['utt_id'])
        shape=np.load(path,mmap_mode='r',allow_pickle=False).shape
        if len(shape)!=2 or 80 not in shape:raise RuntimeError('Invalid formal mel shape')
        lengths[row['utt_id']]=(shape[1] if shape[0]==80 else shape[0])*256
    samples={r['pair_id']:lengths[r['source_utt']] for r in rows}
    unit=os.statvfs(OUTPUT.parent).f_frsize
    wav_sizes={key:44+2*value for key,value in samples.items()}
    allocations={key:allocated_size(value,unit) for key,value in wav_sizes.items()}
    if sum(wav_sizes.values())!=EXPECTED_WAV_BYTES:raise RuntimeError('8100 PCM16 output byte plan changed')
    if unit==4096 and sum(allocations.values())!=EXPECTED_WAV_ALLOCATED:raise RuntimeError('Original WAV allocation plan differs')
    chunks=list(teacher_infer.fixed_chunks(pairs,BATCH_SIZE))
    return {'contract':contract,'completion':completion,'pairs':pairs,'rows':rows,'pair_bytes':pair_bytes,'chunks':chunks,
        'file_sha256':files,'test_input_file_sha256':input_files,'expected_samples':samples,'expected_wav_bytes':wav_sizes,
        'expected_allocations':allocations,'allocation_unit':unit,'actual_max_wav_allocation':max(allocations.values()),
        'max_temporary_wav_allocation':max(TEMPORARY_BUDGET,max(allocations.values()))}


def make_identity(plan):
    return {'identity_format_version':1,'system_id':SYSTEM,'system_label':SYSTEM_LABEL,'model_role':'student','output_dir':str(OUTPUT),
        'student_epoch':0,'student_epoch_semantics':'Actual checkpoint epoch0 is a development cursor; no claim of100 completed epochs.',
        'student_global_step':3327,'teacher_epoch':500,'student_steps':1,'student_start_step':950,'completed_paper_training':False,
        'student_checkpoint':str(STUDENT),'student_checkpoint_sha256':STUDENT_SHA,'parent_full_checkpoint':str(FULL),'parent_full_checkpoint_sha256':FULL_SHA,
        'student_run_contract':str(CONTRACT),'student_run_contract_sha256':CONTRACT_SHA,'student_run_contract_file_sha256':CONTRACT_FILE_SHA,
        'effective_optimizer_lrs':{'student':.00005,'discriminator':.0002},'configuration':plan['contract']['configuration'],
        'configuration_interpretation':'Historical configuration template2e-4 is inactive for S; actual S5e-5 and3327 updates are declared deviations.',
        'vocoder_checkpoint':str(VOCODER),'vocoder_checkpoint_sha256':VOCODER_SHA,'vocoder_kind':'custom','vocoder_global_step':2500000,
        'vocoder_config':str(VOCODER_DIR/'config.json'),'vocoder_attestation':str(ATTESTATION),'protocol':str(PROTOCOL),'protocol_sha256':PROTOCOL_SHA,
        'data_protocol':'text-unseen','paper_pair_count':PAIR_COUNT,'selected_pair_count':PAIR_COUNT,'teacher_pairs_sha256':TEACHER_PAIRS_SHA,
        'pairs_csv_sha256':hashlib.sha256(plan['pair_bytes']).hexdigest(),'file_sha256':plan['file_sha256'],'test_input_file_sha256':plan['test_input_file_sha256'],
        'batch_size':BATCH_SIZE,'base_seed':SEED,'seed_rule':'base_seed + first_pair_id_of_fixed_source-local_chunk',
        'noise_recipe':'one torch.Generator(cuda), original entire source-local chunk shape; no90-pair development audio reuse',
        'bnf_alignment':'legacy_length_linear','bnf_downsampling':'legacy_decimate','waveform_format':{'samplerate':22050,'channels':1,'subtype':'PCM_16','sample_count':'source_mel_T*256'},
        'expected_samples_per_pair':plan['expected_samples'],'expected_logical_wav_bytes':sum(plan['expected_wav_bytes'].values()),
        'expected_allocated_wav_bytes':sum(plan['expected_allocations'].values()),'allocation_unit':plan['allocation_unit'],
        'actual_max_wav_allocation':plan['actual_max_wav_allocation'],'temporary_wav_budget':plan['max_temporary_wav_allocation'],
        'runtime':EXPECTED_RUNTIME,'git_commit':EXPECTED_HEAD,'inference_driver_sha256':digest(Path(__file__)),
        'recovery_protocol':'one shared staging dir; prepare(all chunk WAV SHA/noise/q950), fsync, rename, commit; torn journal or orphan staging stops for review',
        'limitations':['Selected once before8100 evaluation; no checkpoint selection using formal scores.',
            '3327 steps/S learning rate5e-5 do not reproduce the paper100-epoch/2e-4 training schedule.',
            'Training stability remains unresolved; formal metrics do not establish stable long training.',
            'Our fixed DNSMOS OVRL matches our teacher evaluation; equivalence to the paper DNSMOS version is unverified.']}


def ensure_regular(path):
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):raise RuntimeError('Expected a regular non-symlink file: '+str(path))


def publish_exact(path,payload):
    # 已有完整文件只验证不覆盖；未知临时文件不删除、不猜测修复。
    if path.exists():
        ensure_regular(path)
        if path.read_bytes()!=payload:raise RuntimeError('Existing identity/manifest bytes differ: '+str(path))
        return
    temp=path.with_name(path.name+'.tmp')
    if temp.exists():raise RuntimeError('Uncertified partial metadata requires review: '+str(temp))
    with temp.open('xb') as handle:handle.write(payload);handle.flush();os.fsync(handle.fileno())
    temp.rename(path)


def prepare_output(output,identity,pair_bytes):
    if output.is_symlink() or output.resolve()!=Path(identity['output_dir']):raise RuntimeError('Cloned or redirected output identity')
    if not output.exists():output.mkdir()
    identity_path=output/'run_identity.json'
    if not identity_path.exists() and any(output.iterdir()):raise RuntimeError('Partial directory has no authenticated run identity')
    publish_exact(identity_path,json_bytes(identity));publish_exact(output/'pairs.csv',pair_bytes)
    for name in ('wav','staging'):
        folder=output/name
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):raise RuntimeError('Output subdirectory is redirected')
        folder.mkdir(exist_ok=True)


def append_record(path,record):
    # 每条JSON完整换行后fsync；半条尾记录一律拒绝自动续算，保留故障证据。
    payload=json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')+b'\n'
    if path.exists():ensure_regular(path)
    with path.open('ab') as handle:handle.write(payload);handle.flush();os.fsync(handle.fileno())


def validate_wave(path,record):
    ensure_regular(path)
    info=sf.info(str(path))
    if (info.samplerate,info.channels,info.subtype,info.frames)!=(22050,1,'PCM_16',record['num_samples']):raise RuntimeError('Recorded WAV format differs: '+str(path))
    if path.stat().st_size!=44+2*record['num_samples'] or digest(path)!=record['wav_sha256']:raise RuntimeError('Recorded WAV bytes differ: '+str(path))


def check_prepared(record,chunk,rows_by_id,samples,identity_sha):
    if record.get('type')!='prepared' or record.get('run_identity_sha256')!=identity_sha or record.get('chunk_head')!=int(chunk[0].pair_id):raise RuntimeError('Prepared chunk identity differs')
    members=record.get('pairs',[]);wanted=[str(p.pair_id) for p in chunk]
    if [x.get('pair_id') for x in members]!=wanted:raise RuntimeError('Prepared chunk membership/order differs')
    for item in members:
        key=item['pair_id']; row=rows_by_id[key]
        if item.get('output_wav')!=row['output_wav'] or item.get('num_samples')!=samples[key] or item.get('seed')!=SEED+int(chunk[0].pair_id):raise RuntimeError('Prepared WAV/source/seed differs')
        if item.get('chunk_index')!=wanted.index(key) or item.get('chunk_size')!=len(chunk):raise RuntimeError('Prepared noise shape/order changed')
        for name in ('wav_sha256','source_noise_sha256','x950_sha256'):
            value=item.get(name)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise RuntimeError('Invalid prepared SHA: '+name)


def scan_journal(output,chunks,rows,samples,identity_sha,check_files=True):
    path=output/'wav_progress.jsonl';records=[]
    if path.exists():
        ensure_regular(path);payload=path.read_bytes()
        if payload and not payload.endswith(b'\n'):raise RuntimeError('Torn journal tail: stop without deleting or recalculating')
        records=[json.loads(line) for line in payload.splitlines()]
    completed=[];pending=None;position=0;rows_by_id={r['pair_id']:r for r in rows}
    for record in records:
        if pending is None:
            if position>=len(chunks):raise RuntimeError('Journal has extra chunks')
            check_prepared(record,chunks[position],rows_by_id,samples,identity_sha);pending=record
        else:
            expected={'type':'committed','chunk_head':pending['chunk_head'],'run_identity_sha256':identity_sha,
                      'prepared_sha256':hashlib.sha256(json_bytes(pending)).hexdigest()}
            if record!=expected:raise RuntimeError('Journal commit does not match its prepared chunk')
            completed.extend(pending['pairs']);pending=None;position+=1
    if check_files:
        for record in completed:validate_wave(Path(record['output_wav']),record)
        expected_final={Path(r['output_wav']).name for r in completed};expected_staged=set()
        if pending:
            for record in pending['pairs']:
                final=Path(record['output_wav']);staged=output/'staging'/final.name
                if final.exists()==staged.exists():raise RuntimeError('Prepared WAV must exist exactly once in staged/final location')
                validate_wave(final if final.exists() else staged,record)
                (expected_final if final.exists() else expected_staged).add(final.name)
        for folder,allowed in ((output/'wav',expected_final),(output/'staging',expected_staged)):
            if not folder.is_dir() or folder.is_symlink():raise RuntimeError('Expected a real WAV/staging directory')
            if {p.name for p in folder.iterdir()}!=allowed:raise RuntimeError('Orphan/unattested/temporary WAV requires review: '+str(folder))
    return completed,pending,position


def finish_prepared(output,pending):
    # 所有本批WAV在prepare之前已完整暂存；发布中断不需要再次运行学生或G。
    for record in pending['pairs']:
        final=Path(record['output_wav']);staged=output/'staging'/final.name
        if final.exists():validate_wave(final,record)
        else:validate_wave(staged,record);staged.rename(final)
    append_record(output/'wav_progress.jsonl',{'type':'committed','chunk_head':pending['chunk_head'],
        'run_identity_sha256':pending['run_identity_sha256'],'prepared_sha256':hashlib.sha256(json_bytes(pending)).hexdigest()})


def metadata_usage(output):
    # staging WAV属于最终音频那一份分配量；不把它再记成第二份WAV。
    used=directories=0
    if output.exists():
        # 每批无需重新枚举数千份已验真的WAV；只统计少量元数据和三个真实目录。
        paths=[output]
        for child in output.iterdir():
            if child.is_symlink():raise RuntimeError('Output tree contains a symlink')
            paths.append(child)
            if child.is_dir() and child.name not in ('wav','staging'):
                paths.extend(child.rglob('*'))
        for path in paths:
            if path.is_symlink():raise RuntimeError('Output tree contains a symlink')
            if path.is_dir():directories+=path.stat().st_blocks*512
            elif path.is_file() and path.suffix!='.wav':used+=path.stat().st_blocks*512
    log=output.parent/(output.name+'.log')
    if log.exists():ensure_regular(log);used+=log.stat().st_blocks*512
    if used>METADATA_BUDGET or directories>DIRECTORY_BUDGET:raise RuntimeError('Predeclared metadata/log or directory allowance exhausted')
    return used,directories


def required_free(remaining_wave_allocated,metadata_used=0,directory_used=0,temporary=425984):
    if min(remaining_wave_allocated,metadata_used,directory_used,temporary)<0:raise ValueError('Negative disk accounting')
    if metadata_used>METADATA_BUDGET or directory_used>DIRECTORY_BUDGET:raise RuntimeError('Metadata/directory cap exceeded')
    return RESERVE+remaining_wave_allocated+(METADATA_BUDGET-metadata_used)+(DIRECTORY_BUDGET-directory_used)+temporary


def capacity(output,remaining,temporary):
    used,dirs=metadata_usage(output);fs=os.statvfs(output.parent);free=fs.f_bavail*fs.f_frsize
    needed=required_free(remaining,used,dirs,temporary)
    if free<needed:raise RuntimeError('Insufficient disk: free=%d required=%d remaining_WAV=%d'%(free,needed,remaining))
    return {'free_bytes':free,'required_bytes':needed,'remaining_WAV_allocation':remaining,'metadata_allocated':used,'directory_allocated':dirs}


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_fingerprint(model):
    # 推理不保存模型；只把参数与全部buffer的字节逐项加入一个摘要。
    h=hashlib.sha256()
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
        for name,value in items:
            h.update((kind+name+str(value.shape)+str(value.dtype)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def current_runtime():
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'executable':str(Path(sys.executable).resolve()),
        'precision_flags':{'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                           'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic}}


def gpu_idle():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True)
    pids=[int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    if any(pid!=os.getpid() for pid in pids):raise RuntimeError('Another GPU process is active')
    return {'existing_compute_pids':pids}


def readout_report(plan,identity,records,elapsed,generated,skipped):
    wav_hashes={r['pair_id']:r['wav_sha256'] for r in records}
    return {'status':'formal_8100_complete','system_id':SYSTEM,'system_label':SYSTEM_LABEL,'model_role':'student',
        'student_epoch':0,'student_global_step':3327,'teacher_epoch':500,'student_steps':1,'student_start_step':950,
        'paper_pair_count':PAIR_COUNT,'selected_pair_count':len(records),'completed_paper_training':False,'data_protocol':'text-unseen',
        'student_checkpoint':str(STUDENT),'student_checkpoint_sha256':STUDENT_SHA,'parent_full_checkpoint_sha256':FULL_SHA,
        'student_run_contract':str(CONTRACT),'student_run_contract_sha256':CONTRACT_SHA,'student_run_contract_file_sha256':CONTRACT_FILE_SHA,
        'vocoder_checkpoint_sha256':VOCODER_SHA,'vocoder_kind':'custom','vocoder_global_step':2500000,
        'run_identity_sha256':digest(OUTPUT/'run_identity.json'),'pairs_csv_sha256':identity['pairs_csv_sha256'],
        'inference_driver_sha256':identity['inference_driver_sha256'],'file_sha256':identity['file_sha256'],
        'generated_wav_sha256':wav_hashes,
        'generated_wav_collection_sha256':training.canonical_json_sha256(wav_hashes),
        'noise_sha256_per_pair':{r['pair_id']:r['source_noise_sha256'] for r in records},'x950_sha256_per_pair':{r['pair_id']:r['x950_sha256'] for r in records},
        'wav_progress_sha256':digest(OUTPUT/'wav_progress.jsonl'),'expected_samples_per_pair':plan['expected_samples'],
        'generated_this_invocation':generated,'skipped_verified_from_previous_invocations':skipped,'elapsed_seconds':elapsed,
        'runtime':identity['runtime'],'batch_size':BATCH_SIZE,'base_seed':SEED,'seed_rule':identity['seed_rule'],
        'model_checkpoint_writes':0,'original_model_parameters_and_buffers_unchanged':True,'limitations':identity['limitations']}


def validate_complete_output(output_dir=OUTPUT,verify_dependencies=True):
    # 公共只读验收接口：metrics在自己的3.9环境可用，不导入旧的epoch100身份解析器。
    output=Path(output_dir)
    if output.is_symlink() or output.resolve()!=OUTPUT:raise RuntimeError('Require the fixed3327 output, not a cloned directory')
    plan=load_plan(verify_dependencies);identity=make_identity(plan)
    for name in ('run_identity.json','pairs.csv','inference_report.json'):ensure_regular(output/name)
    if (output/'run_identity.json').read_bytes()!=json_bytes(identity) or (output/'pairs.csv').read_bytes()!=plan['pair_bytes']:
        raise RuntimeError('Inference identity or full pair manifest changed')
    identity_sha=digest(output/'run_identity.json')
    records,pending,count=scan_journal(output,plan['chunks'],plan['rows'],plan['expected_samples'],identity_sha)
    if pending is not None or count!=len(plan['chunks']) or len(records)!=PAIR_COUNT:raise RuntimeError('Inference has not completed every formal chunk')
    report=read_json(output/'inference_report.json')
    expected=readout_report(plan,identity,records,report.get('elapsed_seconds'),report.get('generated_this_invocation'),report.get('skipped_verified_from_previous_invocations'))
    if report!=expected:raise RuntimeError('Complete inference receipt differs from actual journal/model identity')
    if not isinstance(report['elapsed_seconds'],(int,float)) or not math.isfinite(report['elapsed_seconds']) or not 0<=report['elapsed_seconds']<=10800:
        raise RuntimeError('Inference duration is invalid')
    if any(type(report[k]) is not int or report[k]<0 for k in ('generated_this_invocation','skipped_verified_from_previous_invocations')) or report['generated_this_invocation']+report['skipped_verified_from_previous_invocations']!=PAIR_COUNT:
        raise RuntimeError('Generated/skipped invocation counts differ')
    return {'rows':plan['rows'],'identity':identity,'report':report,'report_sha256':digest(output/'inference_report.json'),'identity_sha256':identity_sha}


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    parser.add_argument('--max-seconds',type=int,default=10800)
    parser.add_argument('--run-approved',action='store_true')
    args=parser.parse_args(argv)
    if not args.run_approved:parser.error('An explicit one-run GPU approval is required')
    if args.output_dir.is_symlink() or args.output_dir.resolve()!=OUTPUT:parser.error('Only the fixed3327 output directory is allowed')
    if not 1<=args.max_seconds<=10800:parser.error('The inference stage is bounded by3 hours')
    return args


def main():
    args=parse_args();started=time.monotonic();stop={'requested':False}
    signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True));signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True))
    training.assert_rented_server(ROOT);locks=training.acquire_runtime_locks(ROOT)
    if formal.git_head(ROOT)!=EXPECTED_HEAD:raise RuntimeError('Expected clean frozen production HEAD')
    if current_runtime()!=EXPECTED_RUNTIME:raise RuntimeError('Original inference runtime/precision flags differ: '+repr(current_runtime()))
    initial_gpu=gpu_idle()
    def check(remaining=0,temporary=0):
        if stop['requested']:raise RuntimeError('External stop requested; preserved verified/partial artifacts')
        if time.monotonic()-started>args.max_seconds:raise TimeoutError('Formal inference reached its predeclared time bound')
        return capacity(OUTPUT,remaining,temporary)
    plan=load_plan();identity=make_identity(plan)
    identity_sha=hashlib.sha256(json_bytes(identity)).hexdigest()
    # 已有完成结果只读验真；不重复生成，不覆盖旧完成报告。
    if (OUTPUT/'inference_report.json').exists():
        validate_complete_output();print('FORMAL3327_ALREADY_COMPLETE_VERIFIED',flush=True);return
    if OUTPUT.exists():
        # 恢复前只读验身份；不能在容量门禁之前补写缺失清单或修复半初始化目录。
        for name in ('run_identity.json','pairs.csv'):ensure_regular(OUTPUT/name)
        if (OUTPUT/'run_identity.json').read_bytes()!=json_bytes(identity) or (OUTPUT/'pairs.csv').read_bytes()!=plan['pair_bytes']:
            raise RuntimeError('Existing output has a different complete inference identity')
        complete,pending,position=scan_journal(OUTPUT,plan['chunks'],plan['rows'],plan['expected_samples'],identity_sha)
    else:complete,pending,position=[],None,0
    present={r['pair_id'] for r in complete};present.update(r['pair_id'] for r in pending['pairs']) if pending else None
    remaining=sum(v for k,v in plan['expected_allocations'].items() if k not in present)
    initial_disk=check(remaining,plan['max_temporary_wav_allocation'])
    if not OUTPUT.exists():prepare_output(OUTPUT,identity,plan['pair_bytes'])
    if pending:
        finish_prepared(OUTPUT,pending);complete.extend(pending['pairs']);position+=1
    skipped=len(complete)
    print(json.dumps({'stage':'authenticated_plan_before_GPU_models','verified_pairs':skipped,'initial_disk':initial_disk,'runtime':current_runtime(),'initial_gpu':initial_gpu}),flush=True)
    # 只有不完整而且已经核验身份的目录才加载模型；CPU验证恢复不需要任何前向。
    model=vocoder=None;fingerprints={};generated=0
    if position<len(plan['chunks']):
        check(remaining,plan['max_temporary_wav_allocation'])
        payload=load_torch_checkpoint(STUDENT);config=validate_payload(payload,plan['contract'],plan['completion'])
        torch.set_num_threads(4);device=torch.device('cuda')
        model=build_model('legacy_decimate');model.load_state_dict(payload['student'],strict=True);del payload;gc.collect()
        model.to(device).eval();set_requires_grad(model,False);assert_frozen(model,'fixed3327 student')
        vocoder,_,_=load_frozen_vocoder(ROOT,VOCODER,VOCODER_DIR/'config.json',ATTESTATION,config,device)
        mean,std=load_mel_statistics(ROOT/'data',device)
        mean_np=np.load(ROOT/'data/stats/mel_mean.npy',allow_pickle=False).astype(np.float32).reshape(80,1)
        std_np=np.load(ROOT/'data/stats/mel_std.npy',allow_pickle=False).astype(np.float32).reshape(80,1)
        diffusion=VoiceGradDiffusion(1000,offset=.008).to(device)
        fingerprints={'student':parameter_fingerprint(model),'vocoder':parameter_fingerprint(vocoder)}
        for chunk in plan['chunks'][position:]:
            check(remaining,plan['max_temporary_wav_allocation'])
            if current_runtime()!=EXPECTED_RUNTIME:raise RuntimeError('Runtime precision flags changed during inference')
            source=chunk[0];mel=teacher_infer.feature_matrix(teacher_infer.feature_path(ROOT/'data','mel',source.source_speaker,source.source_utt),80)
            bnf=teacher_infer.feature_matrix(teacher_infer.feature_path(ROOT/'data','bnf',source.source_speaker,source.source_utt),144)
            if not np.isfinite(mel).all() or not np.isfinite(bnf).all():raise RuntimeError('Nonfinite source mel/BNF')
            x=torch.from_numpy(((mel-mean_np)/std_np).astype(np.float32)).unsqueeze(0).to(device)
            condition=align_bnf_to_mel(torch.from_numpy(bnf).unsqueeze(0).to(device),x.shape[-1],mode='legacy_length_linear',sample_rate=22050,n_fft=1024,hop_size=256,bnf_frame_shift_seconds=.010)
            x=x.expand(len(chunk),-1,-1).contiguous();condition=condition.expand(len(chunk),-1,-1).contiguous()
            vectors=[]
            for pair in chunk:
                vector=np.load(teacher_infer.feature_path(ROOT/'data','spk_emb',pair.reference_speaker,pair.reference_utt),allow_pickle=False).squeeze()
                if vector.shape!=(256,) or not np.isfinite(vector).all():raise RuntimeError('Invalid target speaker embedding')
                vectors.append(vector.astype(np.float32))
            speaker=torch.from_numpy(np.stack(vectors)).to(device)
            seed=SEED+int(chunk[0].pair_id);generator=torch.Generator(device=device).manual_seed(seed)
            noise=torch.randn(x.shape,device=device,dtype=x.dtype,generator=generator)
            with torch.inference_mode():
                converted=formal.student_one_step(model,diffusion,x,speaker,condition,noise)
                q950=diffusion.q_sample(x,torch.full((len(chunk),),950,device=device,dtype=torch.long),noise=noise)
                audio=vocoder(converted*std+mean).squeeze(1).cpu().numpy()
            if not bool(torch.isfinite(converted).all()) or not np.isfinite(audio).all():raise RuntimeError('Nonfinite generated mel/waveform')
            prepared={'type':'prepared','chunk_head':int(chunk[0].pair_id),'run_identity_sha256':identity_sha,'pairs':[]}
            for index,pair in enumerate(chunk):
                key=str(pair.pair_id);path=OUTPUT/'staging'/Path(pair.output_wav).name
                if path.exists() or path.with_name(path.stem+'.tmp.wav').exists() or Path(pair.output_wav).exists():raise RuntimeError('Unexpected staged/final WAV; never overwrite or recompute verified output')
                if audio[index].ndim!=1 or len(audio[index])!=plan['expected_samples'][key]:raise RuntimeError('Generated sample count differs from source melT*256')
                atomic_write_wav(path,np.clip(audio[index].astype(np.float32),-1.,1.),22050)
                record={'pair_id':key,'output_wav':pair.output_wav,'num_samples':len(audio[index]),'wav_sha256':digest(path),
                    'seed':seed,'chunk_index':index,'chunk_size':len(chunk),'source_noise_sha256':tensor_sha(noise[index]),'x950_sha256':tensor_sha(q950[index])}
                validate_wave(path,record);prepared['pairs'].append(record)
                remaining-=plan['expected_allocations'][key]
                check(remaining,plan['max_temporary_wav_allocation'])
            # 先准备全部本批WAV，再把完整意图写盘；发布中途恢复不再需要模型。
            append_record(OUTPUT/'wav_progress.jsonl',prepared);finish_prepared(OUTPUT,prepared)
            complete.extend(prepared['pairs']);generated+=len(chunk)
            print(json.dumps({'stage':'chunk_committed','chunk_head':prepared['chunk_head'],'verified_pairs':len(complete),'remaining_wave_allocation':remaining}),flush=True)
        if fingerprints!={'student':parameter_fingerprint(model),'vocoder':parameter_fingerprint(vocoder)}:raise RuntimeError('Inference mutated model parameters or buffers')
        assert_frozen(model,'fixed3327 student');assert_frozen(vocoder,'fixed customG')
    check(0,plan['max_temporary_wav_allocation'])
    verify_files(plan['file_sha256'])
    records,pending,count=scan_journal(OUTPUT,plan['chunks'],plan['rows'],plan['expected_samples'],identity_sha)
    if pending or count!=len(plan['chunks']) or len(records)!=PAIR_COUNT:raise RuntimeError('Incomplete formal8100 inference')
    if (OUTPUT/'run_identity.json').read_bytes()!=json_bytes(identity) or (OUTPUT/'pairs.csv').read_bytes()!=plan['pair_bytes']:raise RuntimeError('Inference identity changed during generation')
    report=readout_report(plan,identity,records,time.monotonic()-started,generated,skipped)
    check(0,plan['max_temporary_wav_allocation']);publish_exact(OUTPUT/'inference_report.json',json_bytes(report))
    print('FORMAL3327_8100_COMPLETE '+str(OUTPUT/'inference_report.json'),flush=True)


if __name__=='__main__':
    main()
