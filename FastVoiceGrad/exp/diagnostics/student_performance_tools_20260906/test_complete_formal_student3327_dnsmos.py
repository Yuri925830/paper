"""补齐入口的租用CPU验收；不评分、不推理、不写正式实验结果。"""

import ast  # 检查新入口没有三项重算或修改冻结模块全局的代码。
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Only Matpool CPU tests with CUDA_VISIBLE_DEVICES='' are allowed")
import numpy as np
import complete_formal_student3327_dnsmos_v1 as c


class DNSCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=c.load_base();cls.m=cls.e.load_evaluator()
        cls.rows=cls.m.load_pairs(c.OUTPUT/'pairs.csv',0,None)

    def test_actual_frozen_failure_protocol_and_three_complete_CSVs(self):
        # 只读真实8100行，用原校验器重算文本编辑距离；绝不调用ASR或声纹模型。
        e=self.e;old=c.verify_retained(e)
        self.assertEqual(e.canonical(old),c.OLD_IDENTITY_SHA)
        self.assertEqual(e.sha(c.PROTOCOL),c.PROTOCOL_SHA)
        self.assertEqual(e.sha(c.BASE),c.BASE_SHA)
        data=c.retained_rows(e,self.m,self.rows)
        self.assertEqual({k:len(data[k]) for k in ('utmos','cer','sva')},{'utmos':8100,'cer':8100,'sva':8100})
        self.assertEqual(sum(int(row['char_errors']) for row in data['cer']),11396)
        self.assertEqual(sum(int(row['reference_chars']) for row in data['cer']),566190)
        self.assertEqual(sum(int(row['verified']) for row in data['sva']),7840)
        self.assertEqual(old['batch_size'],4);self.assertEqual(old['dnsmos_workers'],3)
        self.assertEqual(old['software_versions'],e.check_runtime())

    def test_all_original_dependencies_regular_and_pins_do_not_conflict(self):
        # 在真实文件上核身份合并和普通文件条件；这里只散列字节，没有torch.load或网络前向。
        e=self.e;old=e.read_json(c.METRICS/'metrics_run_identity.json');files=dict(e.PINNED)
        for mapping in (c.PRESERVED,c.SB_FILES,old['file_sha256'],{str(c.BASE):c.BASE_SHA,str(c.PROTOCOL):c.PROTOCOL_SHA}):
            e.merge_pins(files,mapping)
        c.require_regular(files);e.verify_files(files)
        self.assertGreater(len(files),700)
        with self.assertRaises(RuntimeError):e.merge_pins(files,{str(c.BASE):'wrong'})

    def test_false_success_or_changed_startup_identity_is_rejected(self):
        # 只替换内存中的读取结果来注入错误，磁盘上的原failure/progress完全不动。
        e=self.e;reader=e.read_json
        for filename,key,value in [('failure.json','status','formal_8100_complete'),
                ('failure.json','error','another failure'),('progress.json','last_completed_metric','sva'),
                ('metrics_run_identity.json','scoring_precision_flags',c.AFTER)]:
            def changed(path):
                value_read=copy.deepcopy(reader(path))
                if Path(path).name==filename:value_read[key]=value
                return value_read
            with mock.patch.object(e,'read_json',side_effect=changed):
                with self.assertRaises(RuntimeError):c.verify_retained(e)

    def test_retained_missing_rows_or_changed_CER_count_is_rejected(self):
        # 简化的假记录只测试门禁本身；真实CSV格式另有上面的完整现场测试。
        data={'utmos':[{}]*8100,'sva':[{}]*8100,'cer':[{'char_errors':'0','reference_chars':'1'}]*8100,'dnsmos':[]}
        with mock.patch.object(self.e,'metric_rows',return_value=data):
            with self.assertRaisesRegex(RuntimeError,'integer counts'):c.retained_rows(self.e,self.m,self.rows)
        data['sva']=data['sva'][:-1]
        with mock.patch.object(self.e,'metric_rows',return_value=data):
            with self.assertRaisesRegex(RuntimeError,'each contain'):c.retained_rows(self.e,self.m,self.rows)

    def test_real_CPU_import_trace_is_posthoc_and_parent_unchanged(self):
        # 真子进程执行精确SpeakerRecognition导入；CUDA为空，也不创建ECAPA实例。
        before=self.e.scoring_precision(self.m);loaded_before='speechbrain' in sys.modules
        trace=c.cpu_import_trace(self.e,time.monotonic()+60)
        self.assertEqual(trace['before'],c.BEFORE);self.assertEqual(trace['after'],c.AFTER)
        self.assertFalse(trace['cuda_available']);self.assertEqual(trace['cuda_visible_devices'],'')
        self.assertFalse(trace['network_constructed']);self.assertFalse(trace['model_forward_performed'])
        self.assertEqual(trace['imported_class'],'speechbrain.inference.speaker.SpeakerRecognition')
        self.assertEqual(trace['evidence_scope'],'post_failure_CPU_import_only_not_original_SVA_end_flags')
        self.assertEqual(self.e.scoring_precision(self.m),before)
        self.assertEqual('speechbrain' in sys.modules,loaded_before)

    def test_CPU_trace_rejects_wrong_flags_or_disabled_quirks(self):
        trace={'before':c.BEFORE,'after':c.AFTER,'cuda_visible_devices':'','cuda_available':False,
            'torch':'2.6.0+cu124','speechbrain':'1.0.3','excluded_quirks':[],'applied_quirks':['allow_tf32']}
        for key,value in [('after',c.BEFORE),('cuda_available',True),('excluded_quirks',['allow_tf32'])]:
            broken=dict(trace);broken[key]=value
            done=SimpleNamespace(returncode=0,stdout='SB_IMPORT_TRACE '+json.dumps(broken),stderr='')
            with mock.patch.object(c.subprocess,'run',return_value=done) as run:
                with self.assertRaises(RuntimeError):c.cpu_import_trace(self.e,time.monotonic()+60)
                self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                self.assertEqual(run.call_args.kwargs['env']['HF_HUB_OFFLINE'],'1')

    def test_only_identical_completion_identity_can_resume(self):
        # 临时目录模拟合法前缀恢复；不生成任何正式DNSMOS文件。
        with tempfile.TemporaryDirectory(prefix='dnsmos-completion-identity-') as temp:
            p=Path(temp)/'identity.json';dns=p.with_name('dnsmos.csv');identity={'old':c.OLD_IDENTITY_SHA,'files':{'a':'b'}}
            wanted=c.bind_completion(self.e,p,identity,dns)
            dns.write_text('pair_id,p808_mos,mos_sig,mos_bak,mos_ovr\n')
            self.assertEqual(c.bind_completion(self.e,p,copy.deepcopy(identity),dns),wanted)
            with self.assertRaisesRegex(RuntimeError,'resume identity'):c.bind_completion(self.e,p,{'old':'changed'},dns)
        with tempfile.TemporaryDirectory(prefix='dnsmos-unattested-') as temp:
            p=Path(temp)/'identity.json';dns=p.with_name('dnsmos.csv');dns.write_text('unattested prefix')
            with self.assertRaisesRegex(RuntimeError,'without'):c.bind_completion(self.e,p,identity,dns)
            self.assertEqual(dns.read_text(),'unattested prefix')

    def test_one_DNS_worker_receives_original_identity_same_locks_and_deadline(self):
        # worker被mock，绝不启动评分；验证只传一次DNS和原身份，原锁对象原样传递。
        deadline=time.monotonic()+60;locks=[object(),object(),object()]
        with mock.patch.object(c,'verify_retained'),mock.patch.object(self.e,'capacity'),\
             mock.patch.object(self.e,'supervised_metric') as worker,\
             mock.patch.object(self.m,'validate_existing_metric_csv',return_value=[{}]*8100):
            result=c.run_dns(self.e,self.m,self.rows,locks,deadline)
            self.assertEqual(len(result),8100)
            worker.assert_called_once_with('dnsmos',c.OLD_IDENTITY_SHA,locks,deadline)

    def test_worker_failure_partial_DNS_disk_and_deadline_do_not_claim_success(self):
        # 故障注入只操作mock；不足行数、磁盘不足和超时都不能返回完整DNS结果。
        with mock.patch.object(c,'verify_retained'),mock.patch.object(self.e,'capacity'),\
             mock.patch.object(self.e,'supervised_metric',side_effect=RuntimeError('worker failed')):
            with self.assertRaisesRegex(RuntimeError,'worker failed'):c.run_dns(self.e,self.m,self.rows,[],time.monotonic()+60)
        with mock.patch.object(c,'verify_retained'),mock.patch.object(self.e,'capacity'),\
             mock.patch.object(self.e,'supervised_metric'),\
             mock.patch.object(self.m,'validate_existing_metric_csv',return_value=[{}]*8099):
            with self.assertRaisesRegex(RuntimeError,'all8100'):c.run_dns(self.e,self.m,self.rows,[],time.monotonic()+60)
        for cause in ('deadline','disk'):
            with mock.patch.object(self.e,'supervised_metric') as worker,\
                 mock.patch.object(self.e,'capacity',side_effect=RuntimeError('low disk') if cause=='disk' else None):
                with self.assertRaises((RuntimeError,TimeoutError)):
                    c.run_dns(self.e,self.m,self.rows,[],time.monotonic()+(-1 if cause=='deadline' else 60))
                worker.assert_not_called()
        self.assertEqual(self.e.RESERVE,3*1024**3+128*1024**2)
        self.assertEqual(self.e.METADATA_BUDGET,32*1024**2)

    def test_standard_summary_is_not_relabelled_and_all_eight_columns_remain(self):
        # 只测试报告schema；虚拟小统计不冒充真实尚未完成的DNSMOS结果。
        e=self.e;current={name:{'estimate':float(i)} for i,name in enumerate(e.COLUMNS)}
        comparison={'systems':{'student3327':current},'paired_comparisons':{'all_eight':list(e.COLUMNS)}}
        bundle={'report':e.read_json(c.OUTPUT/'inference_report.json')}
        sums=np.zeros((100,10));sums[0,0]=11396;sums[0,1]=566190
        identity={'metric_assets':{},'software_versions':e.VERSIONS,'later_cpu_import_trace':{'evidence_scope':'post_failure_CPU_import_only_not_original_SVA_end_flags'}}
        files={'old':'a'}
        with mock.patch.object(e,'sha',return_value='b'*64):
            result=c.make_summary(e,bundle,identity,'c'*64,comparison,sums,files,10)
        self.assertEqual(result['status'],'formal_primary_8100_complete_sva_qualified')
        self.assertNotEqual(result['status'],'formal_8100_complete')
        self.assertEqual(result['metric_order'],list(e.COLUMNS))
        self.assertEqual(result['metrics_run_identity_sha256'],c.OLD_IDENTITY_SHA)
        self.assertIsNone(result['precision_evidence']['original_sva_actual_end_flags'])
        self.assertIn('failed',result['worker_status']['sva'])
        self.assertEqual(result['additional_diagnostics']['CER_total_reference_characters'],566190)
        files['new']='later';self.assertNotIn('new',result['file_sha256'])
        self.assertEqual(c.SUMMARY.name,'objective_summary_recovered_v1.json')

    def test_failure_record_does_not_claim_externally_changed_bytes_preserved(self):
        # 在临时文件注入外部变更，验证真正散列失败仍记录原错误，不伪造旧文件保持完好。
        with tempfile.TemporaryDirectory(prefix='dnsmos-external-change-') as temp:
            p=Path(temp)/'old_failure.json';p.write_text('original failed evidence')
            wanted=self.e.sha(p);p.write_text('externally changed evidence')
            try:self.e.verify_files({str(p):wanted})
            except RuntimeError as error:state=c.failure_state(error,None,1.0)
            else:self.fail('The externally changed evidence must fail its checksum')
            self.assertIn('Input/source SHA changed',state['error'])
            self.assertEqual(state['status'],'dns_completion_failed_no_new_complete_claim')
            self.assertTrue(state['original_artifacts_not_written_by_this_completion'])
            self.assertNotIn('original_failure_preserved',state)
            self.assertEqual(p.read_text(),'externally changed evidence')

    def test_actual_historical_functions_and_metric_sequence_match(self):
        # 原Git字节与历史identity相符后比较AST；只读代码，绝不执行历史评分main。
        e=self.e;names={'evaluate_sva','evaluate_utmos','evaluate_cer','evaluate_dnsmos','load_audio',
            'pad_waveforms','grouped_chunks','open_result_writer','validate_existing_metric_csv'}
        def functions(source):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names}
        current=functions(e.EVALUATOR.read_text())
        for rev,wanted in [('629b96a11c61a3ed99a7ca4a2fbc4c485e3dea43','95b6710e9ee6a6ab3cf7318900e4915cc21bb89a633ed0aede634f606525a97a'),
                ('570c690a096d4d9c484994c28f1319e713a074e9','9fe50b2608b5eabf7e0f61c2f6b18886285d853811620942c40a467b1ce8dd26')]:
            raw=subprocess.check_output(['git','show',rev+':evaluation/evaluate_metrics.py'],cwd=c.ROOT)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),wanted)
            self.assertEqual(functions(raw.decode()),current)
            self.assertIn('["utmos", "cer", "sva", "dnsmos"] if args.metric == "all"',raw.decode())

    def test_import_safe_source_has_no_old_writes_no_global_mutation_or_other_scores(self):
        source=Path(c.__file__).read_text();tree=ast.parse(source);dns_calls=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('evaluate_utmos','evaluate_cer','evaluate_sva','evaluate_dnsmos',
                    'train_batch','step','save_checkpoint','write_summary'))
                if node.func.attr=='supervised_metric':dns_calls.append(node)
                if node.func.attr=='atomic_json':
                    self.assertIsInstance(node.args[0],ast.Name)
                    self.assertIn(node.args[0].id,('path','SUMMARY','FAILURE'))
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
                    and t.value.id in ('e','m','torch','training') for t in targets))
        self.assertEqual(len(dns_calls),1);self.assertEqual(dns_calls[0].args[0].value,'dnsmos')
        self.assertIn("'original_sva_actual_end_flags':None",source)
        self.assertIn('e.compare_systems(systems,clusters)',source)
        self.assertIn('e.historical(m,directory,rows,files)',source)
        self.assertNotIn('from_hparams(',c.TRACE_PROGRAM)
        # 测试完毕再次核保留文件，防止验收代码无意污染原实验。
        self.e.verify_files(c.PRESERVED)


if __name__=='__main__':unittest.main()
