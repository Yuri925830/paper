"""租用CPU验收三组只读指标比较；不读取模型张量、不运行真实评分。"""

import ast  # 确认新脚本没有模型加载、优化器或评分入口。
import copy  # 每个错误案例都从独立副本开始。
import importlib.util  # 从显式新文件导入，不改冻结统计辅助模块。
import json  # 只读取已完成训练契约和收据，不读取未来90对评分。
import os  # 测试必须显式隐藏GPU。
import sys
import unittest
from pathlib import Path
from unittest import mock  # 仅模拟目录已存在，不创建真实比较目录。

if not sys.platform.startswith('linux') or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
spec = importlib.util.spec_from_file_location('_paired_lr90_test', Path(sys.argv.pop(1)).resolve())
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PairedMetricComparisonTests(unittest.TestCase):
    def selection(self):
        result = {'schema_version': 1, 'output_dir': str(module.DEV / 'comparison_lr90_cpu_test_not_created')}
        for role, (label, checkpoint, sha, contract, contract_sha, step) in module.EXPECTED.items():
            result[role] = {'label': label, 'candidate_root': str(module.DEV / ('evaluation_candidate_' + label)),
                'checkpoint': str(checkpoint), 'checkpoint_sha256': sha, 'run_contract': str(contract),
                'training_contract_sha256': contract_sha, 'global_step': step, 'epoch': 0}
        return result

    def fixture_rows(self):
        rows = []
        for source in range(10):
            for target in range(9):
                rows.append({'pair_id': source * 9 + target, 'source_speaker': f'p{source}', 'source_utt': f'u{source}',
                    'reference_speaker': f't{target}', 'char_errors': source + target % 2,
                    'reference_chars': 10 + 10 * source, 'verified': int(target != 0),
                    'mos_ovr': 3 + source / 100, 'utmos': 3.5 + target / 100,
                    'mos_sig': 3.6, 'mos_bak': 4.0, 'p808_mos': 3.7, 'cosine_score': .6})
        return rows

    def test_exact_selection_and_shared_root_contract(self):
        entries, output = module.validate_selection(self.selection())
        self.assertEqual(entries['primary']['run_contract'], entries['matched_control']['run_contract'])
        self.assertNotEqual(entries['parent']['run_contract'], entries['primary']['run_contract'])
        self.assertEqual(output.parent, module.DEV)

    def test_refuses_swapped_roles_wrong_step_full_parent_or_unknown_fields(self):
        for role, field, value in [('primary', 'global_step', 3107), ('primary', 'epoch', False),
                ('parent', 'checkpoint', str(module.PARENT / 'latest_full.pt')),
                ('primary', 'checkpoint_sha256', 'pending'), ('matched_control', 'label', 'student_unknown'),
                ('primary', 'run_contract', str(module.PAIRED / 'primary_s1e4/run_contract.json'))]:
            selection = self.selection()
            selection[role][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_selection(selection)
        selection = self.selection()
        selection['primary'], selection['matched_control'] = selection['matched_control'], selection['primary']
        with self.assertRaises(ValueError): module.validate_selection(selection)
        selection = self.selection()
        selection['bootstrap_seed'] = 1
        with self.assertRaises(ValueError): module.validate_selection(selection)
        with mock.patch.object(Path, 'exists', return_value=True), self.assertRaises(ValueError):
            module.validate_selection(self.selection())

    def test_real_training_lineage_and_completion(self):
        parent = json.loads((module.PARENT / 'run_contract.json').read_text())
        paired = json.loads((module.PAIRED / 'run_contract.json').read_text())
        module.validate_lineage(parent, paired)
        receipt = json.loads((module.PAIRED / 'completion.json').read_text())
        module.validate_completion(receipt, self.selection())
        self.assertEqual(receipt['arms']['primary_s1e4']['updates'], 120)

    def test_lineage_or_effective_d_lr_changes_are_rejected(self):
        parent = json.loads((module.PARENT / 'run_contract.json').read_text())
        original = json.loads((module.PAIRED / 'run_contract.json').read_text())
        for field, value in [('maximum_updates_per_arm', 100), ('configuration', {}), ('lineage', {}),
                ('excluded_train_indices', []), ('effective_optimizer_lrs', {'student': .0001, 'discriminator': .0001})]:
            changed = copy.deepcopy(original)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_lineage(parent, changed)

    def test_missing_failed_or_unequal_endpoint_receipt_rejected(self):
        original = json.loads((module.PAIRED / 'completion.json').read_text())
        for name in ('failed', 'steps', 'partial', 'sha', 'firstloss', 'full', 'common'):
            changed = copy.deepcopy(original)
            primary = changed['arms']['primary_s1e4']
            if name == 'failed': changed['status'] = 'running'
            if name == 'steps': primary['updates'] = 100
            if name == 'partial': primary['partial_batch_may_have_updated_D'] = True
            if name == 'sha': primary['student_snapshot']['sha256'] = 'b' * 64
            if name == 'firstloss': primary['first3108_loss_absolute_differences']['student_total'] = .01
            if name == 'full': primary['full_snapshot'] = None
            if name == 'common': changed['common_monitor_updates'] = [0, 20, 40]
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.validate_completion(changed, self.selection())

    def test_ninety_actual_q950_hashes_all_three_must_match(self):
        values = {str(i): f'{i:064x}' for i in range(90)}
        reports = {role: {'x950_sha256_per_pair': copy.deepcopy(values)} for role in module.ROLES}
        self.assertEqual(module.validate_same_noise(reports), values)
        for change in ('different', 'missing', 'invalid'):
            altered = copy.deepcopy(reports)
            if change == 'different': altered['parent']['x950_sha256_per_pair']['3'] = 'f' * 64
            if change == 'missing': del altered['primary']['x950_sha256_per_pair']['89']
            if change == 'invalid': altered['primary']['x950_sha256_per_pair']['0'] = 'unknown'
            with self.subTest(change=change), self.assertRaises(ValueError): module.validate_same_noise(altered)

    def test_all_pairs_ten_sources_nine_targets_and_original_order(self):
        rows = self.fixture_rows()
        self.assertEqual(len(module.source_clusters(rows)), 10)
        for altered in (rows[:-1], list(reversed(rows)), [dict(r, source_utt='same') for r in rows]):
            # 最后一种仍保留10个不同speaker，另将speaker也统一以构造真正错误的聚类。
            if len(altered) == 90 and altered[0].get('source_utt') == 'same':
                altered = [dict(r, source_speaker='same') for r in altered]
            with self.assertRaises(ValueError): module.source_clusters(altered)

    def test_corpus_cer_not_unweighted_sentence_average(self):
        rows = self.fixture_rows()
        clusters = module.source_clusters(rows)
        values = module.absolute_values(rows, clusters)
        expected = 100 * sum(r['char_errors'] for r in rows) / sum(r['reference_chars'] for r in rows)
        incorrect = sum(100 * r['char_errors'] / r['reference_chars'] for r in rows) / 90
        self.assertAlmostEqual(values['CER_corpus_percent'], expected, places=13)
        self.assertGreater(abs(expected - incorrect), .1)
        self.assertAlmostEqual(values['SVA_percent'], 100 * 8 / 9, places=12)

    def test_paired_identity_zero_and_known_shift_all_metrics_kept(self):
        rows = self.fixture_rows()
        clusters = module.source_clusters(rows)
        weights = module.base.bootstrap_weights(10, 10000, 2026090617)
        unchanged = module.compare_rows(rows, rows, clusters, weights)
        self.assertEqual(set(unchanged), set(module.base.METRICS))
        for item in unchanged.values():
            self.assertEqual(item['delta'], 0)
            self.assertEqual(item['cluster_bootstrap_95_percentile_ci'], [0, 0])
        shifted = [dict(row, utmos=row['utmos'] + .2) for row in rows]
        result = module.compare_rows(rows, shifted, clusters, weights)
        self.assertAlmostEqual(result['UTMOS']['delta'], .2, places=12)
        for bound in result['UTMOS']['cluster_bootstrap_95_percentile_ci']:
            self.assertAlmostEqual(bound, .2, places=12)
        self.assertEqual(result['CER_corpus_percent']['favorable_delta_sign'], 'negative')
        self.assertEqual(result['CER_corpus_percent']['unit'], 'percentage_points')
        self.assertEqual(result['DNSMOS_OVRL']['delta'], 0)

    def test_bootstrap_rng_seed_reproducible_and_shared_no_global_rng_mutation(self):
        before = module.np.random.get_state()
        a = module.base.bootstrap_weights(10, 10000, 2026090617)
        b = module.base.bootstrap_weights(10, 10000, 2026090617)
        after = module.np.random.get_state()
        module.np.testing.assert_array_equal(a, b)
        module.np.testing.assert_array_equal(a.sum(axis=1), module.np.full(10000, 10))
        module.np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[2:], after[2:])

    def test_no_gpu_model_scoring_or_optimizer_calls(self):
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        forbidden = {'torch', 'torchaudio', 'subprocess', 'whisper', 'onnxruntime'}
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            names = [alias.name.split('.')[0] for alias in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]]
            self.assertFalse(set(names) & forbidden)
        self.assertIn("multiple_comparison_adjustment': None", source)
        self.assertIn("for role in ('matched_control', 'parent')", source)
        self.assertIn("base.bootstrap_weights(10, 10000, 2026090617)", source)


if __name__ == '__main__':
    unittest.main()
