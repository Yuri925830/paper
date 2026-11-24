"""Matpool纯CPU合成测试；不读取真实候选、不产生模型分数。"""

import copy  # 错误案例不改动合法测试契约。
import importlib.util  # 导入新比较脚本的纯函数。
import sys  # 主线程显式传入远端脚本路径。
import unittest  # 所有测试仅使用小字典和数组。
from pathlib import Path  # Windows禁止执行项目测试。

if not sys.platform.startswith('linux') or not Path('/root/FsatVoiceGrad').is_dir():
    raise RuntimeError('Run tests only on Matpool.')
spec = importlib.util.spec_from_file_location('_cv_comparison_test', Path(sys.argv.pop(1)).resolve())
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)  # 仍核验机器和冻结helper身份。


class ComparisonTests(unittest.TestCase):
    def contracts(self):  # 合成最小契约，真实数据与checkpoint从不加载。
        parent = {'seed': 1234, 'configuration': {'fixed': True}, 'training_indices': [0, 2], 'excluded_train_indices': [1], 'reconstruction_dev_indices': [1], 'fixed_development_inputs': {'mel': 'fixed'}, 'fixed_development_source_noise_sha256': 'fixed'}
        cv = copy.deepcopy(parent)
        cv.update(kind='conversion_score_matched_development', stop_global_step=1200, method='reconstruction adv +2 reconstruction FM +45 conversion clean_x0 sqrt_alpha_bar score; oneD then oneS update', permutation_seed=991000, lineage={'parent_contract_sha256': m.PARENT_CONTRACT_SHA, 'parent_full_checkpoint_sha256': m.PARENT_FULL_SHA, 'start_after_global_step': 600}, batch_plan={'first_step': 601, 'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((step-1)/1229)', 'crop_seed': '1234+100000+step', 'noise_seed': '1234+200000+step'})
        control = copy.deepcopy(cv)
        control.update(kind='acdd_development_continuation', stop_global_step=2000, batch_plan={'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((global_step-1)/1229)', 'crop_seed': '1234+100000+global_step', 'noise_seed': '1234+200000+global_step', 'first_resumed_global_step': 601})
        return parent, cv, control

    def selection(self, step=900):  # 900用于测试提前停止，不暗示真实900步结果存在。
        candidate = {'label': 'student_cv_s900', 'candidate_root': str(m.DEV / 'synthetic_cv_evaluation'), 'checkpoint': str(m.DEV / 'synthetic_cv/student_step_000900.pt'), 'checkpoint_sha256': 'a' * 64, 'training_contract_sha256': 'b' * 64, 'global_step': step, 'epoch': 0}
        control = {**candidate, 'label': 'student_acdd_s900', 'checkpoint': str(m.DEV / f'acdd_clean_resume_s2000_v1/full_step_{step:06d}.pt'), 'training_contract_sha256': m.CONTROL_CONTRACT_SHA}
        return {'schema_version': 1, 'candidate': candidate, 'matched_control': control, 'output_dir': str(m.DEV / 'synthetic_test_never_written_cv')}

    def test_accepts_actual_early_endpoint(self):
        p, c, a = self.contracts()
        m.validate_lineage(p, c, 900, a)  # stop计划1200可以接受实际保存900。
        self.assertEqual(m.validate_selection(self.selection())[0]['global_step'], 900)

    def test_missing_control_is_explicit(self):
        selection = self.selection()
        selection['matched_control'] = None  # 不补造控制，不启动任何任务。
        self.assertIsNone(m.validate_selection(selection)[1])
        p, c, _ = self.contracts()
        m.validate_lineage(p, c, 900)

    def test_rejects_mismatched_step_and_archive(self):
        for field, value in [('global_step', 1200), ('checkpoint', str(m.DEV / 'other/full_step_000900.pt'))]:
            s = self.selection()
            s['matched_control'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.validate_selection(s)

    def test_rejects_changed_parent_and_data(self):
        for field in ('configuration', 'training_indices', 'fixed_development_source_noise_sha256'):
            p, c, a = self.contracts()
            c[field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.validate_lineage(p, c, 900, a)
        p, c, a = self.contracts()
        c['lineage']['parent_full_checkpoint_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            m.validate_lineage(p, c, 900, a)

    def test_rejects_smoke_as_performance(self):
        p, c, a = self.contracts()
        c.update(kind='conversion_score_one_step_integration_smoke', stop_global_step=601)
        with self.assertRaises(ValueError):
            m.validate_lineage(p, c, 601, a)

    def test_requires_all90_identical_noise_hashes(self):
        noise = {'x950_sha256_per_pair': {str(i): 'a' * 64 for i in range(90)}}
        m.same_noise(noise, copy.deepcopy(noise))
        changed = copy.deepcopy(noise)
        changed['x950_sha256_per_pair']['0'] = 'b' * 64
        with self.assertRaises(ValueError):
            m.same_noise(noise, changed)
        with self.assertRaises(ValueError):
            m.same_noise({}, {})  # 缺哈希不能被当作“两个空字典相同”。

    def test_paired_intervals_and_corpus_cer(self):
        clusters = [(f'p{i}', f'u{i}') for i in range(10)]
        rows = [{'source_speaker': s, 'source_utt': u, 'char_errors': 1, 'reference_chars': 10 if i == 0 else 100, 'verified': 1, 'mos_ovr': 3, 'utmos': 4, 'mos_sig': 3, 'mos_bak': 4, 'p808_mos': 3, 'cosine_score': .8} for i, (s, u) in enumerate(clusters) for _ in range(9)]
        weights = m.base.bootstrap_weights(10, 10000, 2026090617)
        same = m.compare_rows(rows, rows, clusters, weights)
        self.assertAlmostEqual(same['CER_corpus_percent']['reference'], 100 * 90 / 8190)  # 不是逐句CER均值。
        self.assertTrue(all(v['cluster_bootstrap_95_percentile_ci'] == [0, 0] for v in same.values()))
        changed = [{**r, 'char_errors': 0, 'utmos': 4.5} for r in rows]
        result = m.compare_rows(rows, changed, clusters, weights)
        self.assertLess(result['CER_corpus_percent']['delta'], 0)
        self.assertAlmostEqual(result['UTMOS']['delta'], .5)


if __name__ == '__main__':
    unittest.main()  # 测试不创建任何真实候选输出。
