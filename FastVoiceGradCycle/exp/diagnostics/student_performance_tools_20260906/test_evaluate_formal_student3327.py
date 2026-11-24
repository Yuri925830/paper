"""仅在租用服务器CPU验收正式评分入口；不评分、不生成正式结果。"""

import ast  # 验证入口没有训练、旧身份汇总或全局monkeypatch。
import copy
import csv
import hashlib
import json
import os
import fcntl
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Tests are restricted to Matpool CPU with CUDA_VISIBLE_DEVICES=''")
import numpy as np
import evaluate_formal_student3327_v1 as e


class FormalMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=e.load_evaluator()  # 只导入原CSV/数学校验函数，不加载评分网络。
        cls.rows=cls.m.load_pairs(e.TEACHER/'pairs.csv',0,None)

    def test_real_teacher_pairs_8100_100sources_denominator(self):
        e.require_pairs(self.rows,self.rows)
        self.assertEqual(sum(len(self.m.normalize_cer_text(r['source_text'])) for r in self.rows),566190)
        broken=copy.deepcopy(self.rows);broken[81]['source_text']+=' CHANGED'
        with self.assertRaises(RuntimeError):e.require_pairs(broken,self.rows)
        with self.assertRaises(RuntimeError):e.require_pairs(self.rows[:-1],self.rows)
        broken=copy.deepcopy(self.rows);broken[1],broken[2]=broken[2],broken[1]
        with self.assertRaises(RuntimeError):e.require_pairs(broken,self.rows)

    def test_real_legacy_and_teacher_CSV_integer_counts_and_full_precision(self):
        # 全部8100行转写只重算字符距离，绝不重新ASR；历史收据原样只读。
        for directory,errors,cer,utmos in [(e.TEACHER,11946,2.109892438933927,3.9812097230664008),
                (e.LEGACY,27336,4.82806125152334,3.719332473439935)]:
            files={};clusters,sums,summary=e.historical(self.m,directory,self.rows,files)
            self.assertEqual(len(clusters),100);self.assertEqual(int(sums[:,0].sum()),errors)
            self.assertEqual(int(sums[:,1].sum()),566190)
            self.assertAlmostEqual(e.values_from_sums(sums.sum(0))[0],cer,places=12)
            self.assertAlmostEqual(e.values_from_sums(sums.sum(0))[3],utmos,places=12)
            self.assertEqual(set(summary['individual_metric_csv_sha256']),set(e.METRICS))
            self.assertEqual(len(files),4)

    def test_6_assets_DNSMOS_original_runtime_no_models(self):
        self.assertEqual(e.check_runtime()['python'],'3.9.23')
        assets=e.scoring_assets(self.m)
        self.assertEqual(len(assets),6)
        aux=e.auxiliary_sources(assets)
        self.assertEqual(len(aux),11)
        self.assertIn(str(e.ROOT/'pretrained_spkrec_ecapa/mean_var_norm_emb.ckpt'),aux)
        self.assertEqual(e.sha(e.DNSMOS_SOURCE),e.DNSMOS_SHA)
        with mock.patch.object(self.m,'metric_asset_manifest',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'six'):e.scoring_assets(self.m)

    def test_protocol_and_source_pins_real_files(self):
        self.assertEqual(e.sha(e.PROTOCOL),e.PROTOCOL_SHA)
        self.assertEqual(e.sha(e.EVALUATOR),e.EVALUATOR_SHA)
        self.assertEqual(e.sha(e.RUN/'run_contract.json'),e.CONTRACT_FILE_SHA)
        self.assertEqual(e.canonical(e.read_json(e.RUN/'run_contract.json')),e.CONTRACT_SHA)

    def test_wrong_student_or_partial_inference_rejected_before_audio(self):
        report={'status':'formal_8100_complete','system_id':'fastvoicegrad_student3327','model_role':'student',
            'student_global_step':3327,'student_steps':1,'student_start_step':950,'teacher_epoch':500,
            'selected_pair_count':8100,'paper_pair_count':8100,'student_checkpoint_sha256':e.STUDENT_SHA,
            'student_run_contract_sha256':e.CONTRACT_SHA,'parent_full_checkpoint_sha256':e.FULL_SHA}
        for key,value in [('student_global_step',3328),('student_checkpoint_sha256','bad'),('status','partial'),
                ('student_run_contract_sha256','bad'),('parent_full_checkpoint_sha256','bad'),('student_steps',30)]:
            changed=dict(report);changed[key]=value
            with self.assertRaises(RuntimeError):e.validate_lineage({'report':changed,'rows':self.rows})

    @unittest.skipIf(e.INFER_SHA.startswith('PENDING'),'Inference driver has not reached its final freeze yet')
    def test_real_inference_plan_and_report_schema_under_eval_environment(self):
        # 真400特征、3327契约、8100配对均读取现场；只把尚未生成的WAV替换为明确的schema样例。
        # 这个测试不产生新推理文件，不把虚拟WAV摘要当作真实结果。
        inference=e.load_inference();plan=inference.load_plan(verify_dependencies=True)
        identity=inference.make_identity(plan)
        self.assertEqual(len(identity['test_input_file_sha256']),400)
        self.assertEqual(identity['student_global_step'],3327)
        records=[{'pair_id':r['pair_id'],'wav_sha256':'a'*64,'source_noise_sha256':'b'*64,'x950_sha256':'c'*64} for r in plan['rows']]
        original_digest=inference.digest
        with mock.patch.object(inference,'digest',side_effect=lambda path:'d'*64 if Path(path).parent==e.OUTPUT else original_digest(path)):
            report=inference.readout_report(plan,identity,records,1.0,8100,0)
        self.assertEqual(report['generated_wav_collection_sha256'],e.canonical(report['generated_wav_sha256']))
        original_sha=e.sha
        with mock.patch.object(e,'sha',side_effect=lambda path:'e'*64 if Path(path)==e.OUTPUT/'inference_report.json' else original_sha(path)),\
             mock.patch.object(self.m,'validate_generated_wavs') as validate_wav:
            accepted=e.validate_lineage({'report':report,'rows':plan['rows'],'report_sha256':'e'*64})
            self.assertEqual(len(accepted),8100)
            validate_wav.assert_called_once()
            self.assertEqual(len(validate_wav.call_args.args[1]),8100)

    def test_CSV_bad_tail_and_foreign_transcript_cannot_resume(self):
        with tempfile.TemporaryDirectory(prefix='formal3327-csv-') as temp:
            directory=Path(temp);p=directory/'utmos.csv'
            p.write_text('pair_id,utmos\n0,3.4\n1,\n')
            with self.assertRaises(ValueError):e.metric_rows(self.m,directory,self.rows,complete=False)
            p.write_text('pair_id,utmos\n0,3.4\n')
            self.assertEqual(len(e.metric_rows(self.m,directory,self.rows,complete=False)['utmos']),1)
            with self.assertRaises(RuntimeError):e.metric_rows(self.m,directory,self.rows,complete=True)
            cer=directory/'cer.csv'
            with cer.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=self.m.CER_FIELDS);writer.writeheader()
                writer.writerow({'pair_id':0,'reference_text_normalized':'FOREIGN','hypothesis_text_normalized':'FOREIGN',
                    'char_errors':0,'reference_chars':7,'cer_percent':0})
            with self.assertRaisesRegex(RuntimeError,'fixed source'):e.metric_rows(self.m,directory,self.rows,complete=False)

    def test_identity_reuse_is_exact_and_unattested_CSV_rejected(self):
        with tempfile.TemporaryDirectory(prefix='formal3327-identity-') as temp:
            p=Path(temp)/'metrics_run_identity.json';identity={'input':'a','model':3327}
            e.bind_identity(p,identity);e.bind_identity(p,copy.deepcopy(identity))
            with self.assertRaises(RuntimeError):e.bind_identity(p,{'input':'b','model':3327})
        with tempfile.TemporaryDirectory(prefix='formal3327-unattested-') as temp:
            p=Path(temp)/'metrics_run_identity.json';p.with_name('utmos.csv').write_text('pair_id,utmos\n')
            with self.assertRaises(RuntimeError):e.bind_identity(p,identity)

    def test_identity_is_immutable_when_final_provenance_grows(self):
        with tempfile.TemporaryDirectory(prefix='formal3327-frozen-identity-') as temp:
            files={'original':'a'};identity={'file_sha256':dict(files)}
            path=Path(temp)/'metrics_run_identity.json';e.bind_identity(path,identity)
            wanted=e.frozen_identity(identity,path)
            e.merge_pins(files,{'historical_cer_csv':'b'})
            self.assertNotIn('historical_cer_csv',identity['file_sha256'])
            self.assertEqual(e.frozen_identity(identity,path),wanted)
            with self.assertRaises(RuntimeError):e.merge_pins(files,{'original':'wrong'})
            identity['file_sha256']['original']='wrong'
            with self.assertRaises(RuntimeError):e.frozen_identity(identity,path)

    def test_partial_or_redirected_metadata_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='formal3327-metadata-') as temp:
            path=Path(temp)/'progress.json';e.atomic_json(path,{'completed':1})
            partial=path.with_name(path.name+'.tmp');partial.write_text('unfinished evidence')
            with self.assertRaises(RuntimeError):e.atomic_json(path,{'completed':2})
            self.assertEqual(e.read_json(path),{'completed':1})
            self.assertEqual(partial.read_text(),'unfinished evidence')
            link=Path(temp)/'redirected.json';link.symlink_to(path)
            with self.assertRaises(RuntimeError):e.atomic_json(link,{'completed':3})
            self.assertEqual(e.read_json(path),{'completed':1})

    def toy_systems(self):
        clusters=[('speaker%03d'%i,'source%03d'%i) for i in range(100)]
        sums=np.zeros((100,10),dtype=np.float64);sums[:,9]=81
        sums[:,1]=81*70;sums[:10,1]=81*69
        sums[:,0]=np.arange(100)%7;sums[:,2]=80;sums[:,3]=81*3.2;sums[:,4]=81*3.8
        sums[:,5]=81*3.4;sums[:,6]=81*4.0;sums[:,7]=81*3.6;sums[:,8]=81*.4
        better=sums.copy();better[:,3]+=81*.125;better[:,4]+=81*.25
        return clusters,{'student3327':better,'teacher500':sums,'legacy_student100':sums.copy()}

    def test_shared100cluster_bootstrap_8metrics_and_paper_boundaries(self):
        clusters,systems=self.toy_systems();result=e.compare_systems(systems,clusters)
        self.assertEqual(result['bootstrap']['source_clusters'],100)
        self.assertEqual(result['bootstrap']['repetitions'],10000)
        # 直接独立构造相同PCG64抽样表，验证三系统用的是同一随机流。
        rng=np.random.Generator(np.random.PCG64(202609073327));draws=rng.integers(0,100,size=(10000,100))
        weights=np.zeros((10000,100),dtype=np.int64)
        for i in range(100):weights[:,i]=(draws==i).sum(1)
        self.assertEqual(result['bootstrap']['weight_matrix_sha256'],hashlib.sha256(weights.astype('<i8').tobytes()).hexdigest())
        a=result['paired_comparisons']['student3327_minus_teacher500'];b=result['paired_comparisons']['student3327_minus_legacy_student100']
        self.assertEqual(a,b);self.assertEqual(set(a),set(e.COLUMNS))
        self.assertAlmostEqual(a['UTMOS']['delta'],.25,places=12)
        self.assertTrue(np.allclose(a['UTMOS']['paired_source_cluster_95ci'],[.25,.25],rtol=0,atol=1e-12))
        self.assertEqual(a['CER_corpus_percent']['delta'],0)
        self.assertFalse(result['threshold_checks']['paper_fastvoicegrad']['DNSMOS_OVRL']['numerical_threshold_met'])
        self.assertEqual(len(result['per_source']['student3327']),100)
        self.assertEqual(result['systems']['student3327']['CER_total_reference_characters'],566190)

    def test_corpus_CER_not_macro_and_missing_source_rejected(self):
        a=np.array([[1,2,1,3,4,3,4,3,.5,1],[1,8,1,3,4,3,4,3,.5,1]],dtype=float)
        self.assertEqual(e.values_from_sums(a.sum(0))[0],20)
        self.assertNotEqual(e.values_from_sums(a.sum(0))[0],np.mean(e.values_from_sums(a)[:,0]))
        clusters,systems=self.toy_systems();systems['student3327'][0,9]=80
        with self.assertRaises(RuntimeError):e.compare_systems(systems,clusters)

    def test_disk_threshold_is_not_reduced(self):
        needed=e.RESERVE+e.METADATA_BUDGET+e.DIRECTORY_BUDGET
        self.assertEqual(e.required_bytes(0,0),needed)
        self.assertEqual(e.required_bytes(e.METADATA_BUDGET,e.DIRECTORY_BUDGET),e.RESERVE)
        with self.assertRaises(RuntimeError):e.required_bytes(e.METADATA_BUDGET+1,0)
        for available,accepted in [(needed-1,False),(needed,True)]:
            with mock.patch.object(e.os,'statvfs',return_value=SimpleNamespace(f_bavail=available,f_frsize=1)),\
                 mock.patch.object(e,'metadata_usage',return_value=(0,0)):
                if accepted:self.assertEqual(e.capacity(),available)
                else:
                    with self.assertRaisesRegex(RuntimeError,'3GiB'):e.capacity()

    def test_worker_group_is_stopped_on_disk_timeout_nonzero(self):
        # 所有PID/killpg/Popen均为mock，不创建进程、不发送任何真实信号。
        for cause in ('disk','time','exit'):
            process=mock.Mock(pid=987654,returncode=1 if cause=='exit' else None)
            process.poll.side_effect=[0] if cause=='exit' else [None]
            checks=[100,RuntimeError('low disk')] if cause=='disk' else [100,100]
            with mock.patch.object(e,'capacity',side_effect=checks),mock.patch.object(e.subprocess,'Popen',return_value=process) as popen,\
                 mock.patch.object(e.time,'monotonic',return_value=101),mock.patch.object(e,'stop_child') as stop:
                with self.assertRaises((RuntimeError,TimeoutError)):e.supervised_metric('cer','identity',[],100 if cause=='time' else 200)
                stop.assert_called_once_with(process)
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
                self.assertEqual(popen.call_args.kwargs['env']['HF_HUB_OFFLINE'],'1')
                self.assertEqual(popen.call_args.kwargs['env']['TRANSFORMERS_OFFLINE'],'1')

    def test_worker_lock_and_parent_cannot_be_bypassed(self):
        with mock.patch.object(e.os,'getppid',return_value=12):
            with self.assertRaises(RuntimeError):e.verify_inherited_locks([],12)
            with self.assertRaises(RuntimeError):e.verify_inherited_locks([1,2,3],13)
            with mock.patch.object(e.os,'readlink',return_value='/tmp/wrong-lock'):
                with self.assertRaises(RuntimeError):e.verify_inherited_locks([1,2,3],12)

    def test_true_CPU_subprocess_inherits_three_temporary_locks(self):
        # 只在临时目录建立小锁；不触碰项目真实锁，也不进入评分worker或加载网络。
        with tempfile.TemporaryDirectory(prefix='formal3327-locks-') as temp:
            root=Path(temp);folder=root/'exp/locks';folder.mkdir(parents=True)
            handles=[]
            try:
                for name in e.LOCK_NAMES:
                    handle=(folder/name).open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(handle)
                program="from pathlib import Path; from unittest import mock; import evaluate_formal_student3327_v1 as e; import sys; scope=mock.patch.object(e,'ROOT',Path(sys.argv[1])); scope.start(); e.verify_inherited_locks([int(x) for x in sys.argv[2].split(',')],int(sys.argv[3])); scope.stop()"
                done=subprocess.run([sys.executable,'-c',program,str(root),','.join(str(h.fileno()) for h in handles),str(os.getpid())],
                    cwd=str(Path(__file__).resolve().parent),  # unittest discover只修改父sys.path，子进程需明确测试模块目录。
                    pass_fds=tuple(h.fileno() for h in handles),capture_output=True,text=True,timeout=10)
                self.assertEqual(done.returncode,0,done.stderr)
                with (folder/e.LOCK_NAMES[0]).open('a+') as independent:
                    with self.assertRaises(BlockingIOError):fcntl.flock(independent,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                for handle in handles:handle.close()

    def test_source_uses_only_original_metric_functions_no_globals_or_training(self):
        source=Path(e.__file__).read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('write_summary','resolve_inference_system','train_batch','train_epoch','save_checkpoint'))
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('m','torch','training','inference') for t in targets))
        for name in ('evaluate_utmos','evaluate_dnsmos','evaluate_cer','evaluate_sva'):self.assertIn('m.'+name+'(',source)
        self.assertIn("m.evaluate_dnsmos(rows,destination,4,3)",source)
        self.assertIn("'file_sha256':dict(files)",source)
        self.assertIn('malformed rows require manual review',source)


if __name__=='__main__':unittest.main()
