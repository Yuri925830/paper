"""租用CPU小测；不加载真实checkpoint，不运行学生/G/评分模型。"""
import ast
import copy
import os
import unittest
from pathlib import Path

if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Rental CPU only, with CUDA hidden')
import infer_formal_student122900_v1 as f
torch=f.torch


def payload():
    contract=f.read_json(f.CONTRACT);completion=f.read_json(f.COMPLETION)
    p={'kind':'acdd_mc1_longtrain_from3327_full_v1','epoch':0,'global_step':f.STEP,'completed_passes':100,
        'run_contract_sha256':f.CONTRACT_SHA,'mc_samples':1,'effective_optimizer_lrs':dict(f.LRS),'origin_parent_sha256':f.PARENT_SHA,
        'partial_batch':False,'resumable':True,'score_semantics':'reconstruction_clean_x0_sqrt_alpha_bar',
        'configuration':contract['configuration'],'actual_optimizer_cursors':completion['optimizer_cursors'],
        'rng_state':{'python':None,'numpy':None,'torch':None,'cuda_all':None}}
    for role,count in (('student',71),('discriminator',144)):
        p[role]={str(i):torch.ones(1) for i in range(count)}
        p[role+'_optimizer']={'param_groups':[{'params':list(range(count)),'lr':f.LRS[role],'betas':(.5,.9),'eps':1e-8,'weight_decay':0,'amsgrad':False}],
            'state':{i:{'step':torch.tensor(float(f.STEP)),'exp_avg':torch.ones(1),'exp_avg_sq':torch.ones(1)} for i in range(count)}}
    return p,contract,completion


class FormalTests(unittest.TestCase):
    def test_true122900_header_and_both215_Adam_states(self):
        p,c,d=payload();config=f.validate_payload(p,c,d)
        self.assertEqual((config.student_steps,config.student_start_step),(1,950))
        self.assertEqual(sum(len(p[r+'_optimizer']['state']) for r in ('student','discriminator')),215)

    def test_wrong_end_partial_MCs_or_original_parent_are_rejected(self):
        for key,value in [('global_step',122899),('partial_batch',True),('mc_samples',2),('origin_parent_sha256','a'*64)]:
            p,c,d=payload();p[key]=value
            with self.assertRaises(RuntimeError):f.validate_payload(p,c,d)

    def test_bad_Adam_LR_cursor_shape_nonfinite_or_negative_second_moment(self):
        for case in ('lr','cursor','shape','nonfinite','negative'):
            p,c,d=payload();opt=p['discriminator_optimizer'];item=opt['state'][0]
            if case=='lr':opt['param_groups'][0]['lr']=1e-4
            if case=='cursor':item['step'].sub_(1)
            if case=='shape':item['exp_avg']=torch.ones(2)
            if case=='nonfinite':item['exp_avg'].fill_(float('nan'))
            if case=='negative':item['exp_avg_sq'].fill_(-1)
            with self.assertRaises(RuntimeError):f.validate_payload(p,c,d)

    def test_changed_contract_or_completion_cannot_certify_full(self):
        p,c,d=payload();d['status']='running'
        with self.assertRaises(RuntimeError):f.validate_payload(p,c,d)
        p,c,d=payload();c['mc_samples']=2
        with self.assertRaises(RuntimeError):f.validate_payload(p,c,d)

    def test_explicit_new_space_allocation(self):
        self.assertEqual(f.RESERVE,1207959552)
        self.assertEqual(f.required_free(1991651328),3234639872)
        self.assertEqual(f.required_free(0,1024,512,0),f.RESERVE+f.METADATA_BUDGET+f.DIRECTORY_BUDGET-1536)
        with self.assertRaises(RuntimeError):f.required_free(0,f.METADATA_BUDGET+1)
        with self.assertRaises(ValueError):f.required_free(-1)

    def test_frozen_math_and_journal_aliases_no_oldmain_or_training(self):
        self.assertIs(f.formal.student_one_step,f.old.formal.student_one_step)
        self.assertIs(f.scan_journal,f.old.scan_journal);self.assertIs(f.finish_prepared,f.old.finish_prepared)
        source=Path(f.__file__).read_text();tree=ast.parse(source)
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(calls.count('load_torch_checkpoint'),1)
        self.assertEqual(calls.count('student_one_step'),1)
        for name in ('backward','train_batch','step','evaluate_utmos','evaluate_cer','evaluate_dnsmos','save'):
            self.assertNotIn(name,calls)
        self.assertNotIn('merge_hashes(files,contract[',source)
        self.assertIn("output).name+'.metrics.log'",source)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
