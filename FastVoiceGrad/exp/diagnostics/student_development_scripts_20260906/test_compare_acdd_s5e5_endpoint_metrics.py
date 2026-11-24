"""租用CPU核验3327终点比较器；不加载模型、不重新评分、不发布正式比较。"""

import ast  # 静态检查新入口没有模型、GPU或优化器调用。
import copy  # 错误案例使用副本，不修改真实完成收据。
import importlib.util  # 导入显式脚本路径，避免误测其他版本。
import json  # 真实身份测试只读取已经完成的契约和原评价文件。
import os  # 必须显式隐藏GPU。
import sys
import tempfile  # 仅创建少量临时CSV，退出时只清理测试自己的目录。
import unittest
from pathlib import Path
from types import SimpleNamespace

if not sys.platform.startswith('linux') or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
spec = importlib.util.spec_from_file_location('_s5e5_endpoint_test', Path(sys.argv.pop(1)).resolve())
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EndpointMetricTests(unittest.TestCase):
    def read(self, path):
        return json.loads(path.read_text())  # 返回独立Python对象，原JSON不变。

    def training(self):
        return (self.read(module.RUN / 'run_contract.json'), self.read(module.PAIRED / 'run_contract.json'),
            self.read(module.RUN / 'completion.json'), self.read(module.PAIRED / 'primary_s1e4/arm_completion.json'))

    def noise_fixture(self):
        parent_root = module.DEV / ('evaluation_candidate_' + module.STUDENTS['parent'][0])
        parent_report = self.read(parent_root / module.STUDENTS['parent'][0] / 'inference_complete.json')
        parent_identity = self.read(parent_root / 'run_identity.json')
        teacher_report = self.read(module.BASELINES / module.TEACHER / 'inference_complete.json')
        teacher_identity = self.read(module.BASELINES / 'run_identity.json')
        # 未来候选在本测试中仅复制旧元数据，绝不读取尚未完成的新90对评分。
        return copy.deepcopy(parent_report), parent_report, teacher_report, {
            'candidate': copy.deepcopy(parent_identity), 'parent': parent_identity, 'teacher': teacher_identity}

    def rows(self):
        return [{'pair_id': source * 9 + target, 'source_speaker': 'p%d' % source, 'source_utt': 'u%d' % source,
            'reference_speaker': 't%d' % target, 'char_errors': source + target % 2, 'reference_chars': 10 + source * 10,
            'verified': int(target != 0), 'mos_ovr': 3 + source / 100, 'utmos': 3.5 + target / 100,
            'mos_sig': 3.6, 'mos_bak': 4.0, 'p808_mos': 3.7, 'cosine_score': .6}
            for source in range(10) for target in range(9)]  # 各源长度不同，能检出错误的平均句CER。

    def systems(self, rows):
        return {label: copy.deepcopy(rows) for label in (module.STUDENTS['candidate'][0], module.STUDENTS['parent'][0], module.TEACHER)}

    def test_real_training_and_protocol_identities(self):
        module.verify_files(module.FROZEN, {})
        module.validate_training(*self.training())  # 使用真实100步终点而非合成假收据。
        self.assertEqual(module.OUTPUT.name, 'comparison_acdd_s5e5_3327_v1')

    def test_wrong_lineage_step_partial_and_snapshot_rejected(self):
        for change in ('contract', 'parent', 'step', 'updates', 'partial', 'stops', 'unpaired', 'snapshot', 'full', 'gate'):
            candidate, parent, receipt, parent_receipt = self.training()
            if change == 'contract': candidate['effective_optimizer_lrs']['student'] = .0001
            if change == 'parent': parent['excluded_train_indices'] = []
            if change == 'step': receipt['completed_global_step'] = 3347
            if change == 'updates': receipt['new_updates'] = 120
            if change == 'partial': receipt['partial_batch_may_have_updated_D'] = True
            if change == 'stops': receipt['trend']['stop_reasons'] = []
            if change == 'unpaired': receipt['unpaired_updates'] = 0
            if change == 'snapshot': receipt['student_snapshot']['sha256'] = 'b' * 64
            if change == 'full': parent_receipt['full_snapshot']['committed'] = False
            if change == 'gate': receipt['stage1_gate']['passed'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate_training(candidate, parent, receipt, parent_receipt)

    def test_actual_parent_noise_recipe_and_honest_teacher_limitation(self):
        result = module.validate_noise_and_recipe(*self.noise_fixture())
        self.assertTrue(result['candidate_parent_actual_q950_equal'])
        self.assertEqual(len(result['actual_q950_sha256_per_pair']), 90)
        self.assertFalse(result['teacher_actual_q950_hashes_available'])
        self.assertFalse(result['teacher_tensor_equality_proven'])

    def test_missing_changed_or_malformed_noise_rejected(self):
        for change in ('missing', 'mismatch', 'invalid', 'teacher_added'):
            candidate, parent, teacher, identities = self.noise_fixture()
            if change == 'missing': del candidate['x950_sha256_per_pair']['89']
            if change == 'mismatch': parent['x950_sha256_per_pair']['0'] = '0' * 64
            if change == 'invalid': candidate['x950_sha256_per_pair']['0'] = 'unknown'
            if change == 'teacher_added': teacher['x950_sha256_per_pair'] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate_noise_and_recipe(candidate, parent, teacher, identities)

    def test_changed_teacher_seed_code_or_interface_rejected(self):
        for change in ('seed', 'batch', 'timestep', 'source', 'missing_source'):
            candidate, parent, teacher, identities = self.noise_fixture()
            if change == 'seed': identities['teacher']['seed_rule'] = 'new seed'
            if change == 'batch': teacher['inference_batch_size'] = 2
            if change == 'timestep': identities['candidate']['start_step'] = 999
            key = str(module.ROOT / 'voicegrad/diffusion.py')
            if change == 'source': identities['candidate']['inputs'][key] = '0' * 64
            if change == 'missing_source': del identities['teacher']['inputs'][key]
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate_noise_and_recipe(candidate, parent, teacher, identities)

    def test_all90_pairs_ten_clusters_and_expected_roles_required(self):
        rows = self.rows()
        systems = self.systems(rows)
        for change in ('missing_row', 'swap', 'source', 'role'):
            altered = copy.deepcopy(systems)
            candidate = altered[module.STUDENTS['candidate'][0]]
            if change == 'missing_row': candidate.pop()
            if change == 'swap': candidate[0], candidate[1] = candidate[1], candidate[0]
            if change == 'source': candidate[0]['source_utt'] = 'different'
            if change == 'role': altered['unregistered_student'] = altered.pop(module.TEACHER)
            with self.subTest(change=change), self.assertRaises(ValueError): module.compare(altered, rows)
        with self.assertRaises(ValueError): module.compare(systems, rows[:-1])

    def test_corpus_cer_exact_counts_and_all_eight_metrics(self):
        rows = self.rows()
        absolute, comparisons, per_source, source_deltas, clusters, weights = module.compare(self.systems(rows), rows)
        expected = 100 * sum(r['char_errors'] for r in rows) / sum(r['reference_chars'] for r in rows)
        wrong = sum(100 * r['char_errors'] / r['reference_chars'] for r in rows) / 90
        self.assertGreater(abs(expected - wrong), .1)
        for label, values in absolute.items():
            self.assertAlmostEqual(values['CER_corpus_percent']['estimate'], expected, places=13)
            self.assertIsInstance(values['CER_total_errors'], int)
            self.assertEqual(values['CER_total_reference_chars'], sum(r['reference_chars'] for r in rows))
            self.assertEqual(len(per_source[label]), 10)
        for metric_map in comparisons.values():
            self.assertEqual(set(metric_map), set(module.base.METRICS))
            for value in metric_map.values():
                self.assertEqual(value['delta'], 0)
                self.assertEqual(value['cluster_bootstrap_95_percentile_ci'], [0, 0])
        self.assertEqual(weights.shape, (10000, 10))
        self.assertEqual(len(source_deltas), 2)

    def test_same_bootstrap_for_both_contrasts_and_known_shift(self):
        rows, before_rng = self.rows(), module.np.random.get_state()
        systems = self.systems(rows)
        candidate = module.STUDENTS['candidate'][0]
        systems[candidate] = [dict(r, utmos=r['utmos'] + .2, char_errors=r['char_errors'] + int(r['pair_id'] == 0)) for r in rows]
        _, comparisons, _, deltas, clusters, weights = module.compare(systems, rows)
        module.np.testing.assert_array_equal(weights, module.base.bootstrap_weights(10, 10000, 2026090617))
        a, b = module.base.cluster_sums(systems[candidate], clusters), module.base.cluster_sums(rows, clusters)
        expected_ci = module.base.confidence_interval(module.base.values_from_sums(weights @ a) - module.base.values_from_sums(weights @ b))
        self.assertEqual(comparisons[module.TEACHER], comparisons[module.STUDENTS['parent'][0]])
        for metric_map in comparisons.values():
            for i, name in enumerate(module.base.METRICS):
                module.np.testing.assert_array_equal(metric_map[name]['cluster_bootstrap_95_percentile_ci'], expected_ci[i])
            self.assertAlmostEqual(metric_map['UTMOS']['delta'], .2, places=12)
        self.assertEqual(sum(x['CER_error_count_delta'] for x in deltas[module.TEACHER]), 1)
        after_rng = module.np.random.get_state()
        module.np.testing.assert_array_equal(before_rng[1], after_rng[1])
        self.assertEqual(before_rng[2:], after_rng[2:])

    def test_real_frozen_parent_loader_four_metric_receipts(self):
        # 实际读取已完成父模型及四基线的90行；不调用新候选main，也不读取未来3327评分。
        historical, template, development, metric_identity, hashes = module.base.load_results(module.BASELINES)
        label, checkpoint, sha, contract_path, contract_sha, step = module.STUDENTS['parent']
        args = SimpleNamespace(candidate_root=module.DEV / ('evaluation_candidate_' + label), label=label,
            checkpoint_sha256=sha, training_contract_sha256=contract_sha, global_step=step, epoch=0)
        rows, contract, identity, report = module.helper.load_candidate(args, template, development, metric_identity, hashes)
        self.assertEqual(len(rows), 90)
        self.assertEqual(sum(r['char_errors'] for r in rows), 109)
        self.assertEqual(sum(r['reference_chars'] for r in rows), 4392)
        self.assertEqual(identity['candidate_training_contract'], str(contract_path))
        self.assertEqual(len(historical[module.TEACHER]), 90)
        self.assertEqual(module.helper.canonical_sha(contract), module.PARENT_SHA)

    def test_frozen_csv_reader_rejects_missing_duplicate_rows_and_changed_file(self):
        with tempfile.TemporaryDirectory(prefix='s5e5_compare_cpu_') as temporary:
            path = Path(temporary) / 'sample.csv'
            path.write_text('pair_id,value\n' + ''.join('%d,1\n' % i for i in range(90)))
            hashes = {}
            self.assertEqual(len(module.base.read_csv(path, hashes, ('pair_id', 'value'))), 90)
            module.verify_files(hashes.copy(), {})
            for ids in (list(range(89)), list(range(89)) + [88]):
                path.write_text('pair_id,value\n' + ''.join('%d,1\n' % i for i in ids))
                with self.assertRaises(ValueError): module.base.read_csv(path, {}, ('pair_id', 'value'))
            with self.assertRaises(ValueError): module.verify_files(hashes.copy(), {})

    def test_fixed_anchor_cannot_be_replaced_by_a_later_read_hash(self):
        path, expected = next(iter(module.FROZEN.items()))
        with self.assertRaisesRegex(ValueError, 'Conflicting identities'):
            module.verify_files({path: expected}, {str(path): '0' * 64})

    def test_no_gpu_models_optimizer_or_frozen_global_override(self):
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [x.name.split('.')[0] for x in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]]
                self.assertFalse(set(names) & {'torch', 'torchaudio', 'subprocess', 'onnxruntime', 'whisper'})
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                self.assertFalse(any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in ('base', 'helper') for t in targets))
        self.assertIn("multiple_comparison_adjustment': None", source)
        self.assertIn("teacher_tensor_equality_proven': False", source)
        self.assertEqual(module.base.SYSTEMS, ('teacher500_k30', 'old_student_epoch100_k1', 'teacher_initialized_s0_k1', 'repair_custom_step2460_k1'))


if __name__ == '__main__':
    unittest.main()
