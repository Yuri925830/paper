"""租用CPU测试新ACDD终点比较；不加载模型、语音或评分器。"""

import copy  # 每个拒绝案例单独修改副本。
import importlib.util  # 从显式上传路径导入新比较入口。
import json  # 只读冻结的原600/2000训练契约。
import os  # 要求GPU在本测试进程中不可见。
import sys  # 首个参数是待测脚本，后续交给unittest。
import unittest  # 仅运行小型CPU身份和统计测试。
from pathlib import Path  # 所有真实文件读取限于租用服务器。
from unittest import mock  # 模拟输出已存在，不碰任何旧实验目录。

if not sys.platform.startswith('linux') or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on the Matpool CPU with CUDA_VISIBLE_DEVICES=''.")
spec = importlib.util.spec_from_file_location('_continuation_comparison_test', Path(sys.argv.pop(1)).resolve())
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # 冻结helper和租用机身份仍然真实校验，不在测试中绕过。


class ContinuationComparisonTests(unittest.TestCase):
    def selection(self, step=4000):
        return {'schema_version': 1, 'candidate': {
            'label': f'student_acdd_continued_s{step}',
            'candidate_root': str(module.DEV / f'evaluation_candidate_test_s{step}'),
            'checkpoint': str(module.DEV / 'test_continuation_source' / f'student_step_{step:06d}.pt'),
            'checkpoint_sha256': 'b' * 64, 'training_contract_sha256': 'c' * 64,
            'global_step': step, 'epoch': 0},
            'output_dir': str(module.DEV / 'comparison_cpu_test_never_written')}

    def contracts(self):
        early = json.loads((module.DEV / 'acdd_clean_600_v1/run_contract.json').read_text(encoding='utf-8'))
        parent = json.loads((module.PARENT_DIR / 'run_contract.json').read_text(encoding='utf-8'))
        self.assertEqual(module.helper.canonical_sha(early), module.EARLY_CONTRACT_SHA)
        self.assertEqual(module.helper.canonical_sha(parent), module.PARENT_CONTRACT_SHA)
        child = {name: copy.deepcopy(parent[name]) for name in (
            'configuration', 'training_indices', 'excluded_train_indices', 'reconstruction_dev_indices',
            'fixed_development_inputs', 'fixed_development_source_noise_sha256', 'original_step0_guard_anchor')}
        child.update(kind='acdd2000_to4000_development', stop_global_step=4000, seed=1234, max_seconds=3600,
            development_contract_file_sha256=module.DEVELOPMENT_FILE_SHA,
            method='unchanged production train_batch; reconstruction adv+2FM+45clean_x0 sqrt_alpha_bar score',
            lineage={'parent_directory': str(module.PARENT_DIR), 'parent_checkpoint': str(module.PARENT_DIR / 'full_step_002000.pt'),
                     'parent_checkpoint_sha256': module.PARENT_FULL_SHA, 'parent_contract_sha256': module.PARENT_CONTRACT_SHA,
                     'original_step0_checkpoint_sha256': module.ANCHOR_SHA, 'start_after_global_step': 2000},
            batch_plan={'first_step': 2001, 'batches_per_pass': 1229, 'shuffle_seed': '1234+floor((step-1)/1229)',
                        'crop_seed': '1234+100000+step', 'noise_seed': '1234+200000+step'},
            guard_ratios={'mel_l1_upper': 1.10, 'prediction_rms_lower': 0.80},
            file_sha256={str(module.DRIVER): module.DRIVER_SHA})
        return early, parent, child  # 真实父资料加合成新契约，不读取训练权重。

    def completion(self, step=4000, status='global_step_budget_complete'):
        entry = self.selection(step)['candidate']
        directory = Path(entry['checkpoint']).parent
        receipt = {'global_step': step, 'path': str(directory / 'latest_full.pt'), 'checkpoint_sha256': 'a' * 64,
                   'run_contract_sha256': entry['training_contract_sha256'], 'full_committed': True,
                   'student_snapshot': {'path': entry['checkpoint'], 'sha256': entry['checkpoint_sha256'], 'committed': True}}
        completed = {'status': status, 'global_step': step, 'last_saved_global_step': step,
                     'start_after_global_step': 2000, 'completed_new_steps': step - 2000,
                     'unsaved_completed_steps': 0, 'run_contract_sha256': entry['training_contract_sha256'],
                     'stop_global_step': 4000, 'latest_full': receipt}
        return completed, dict(receipt, frozen=True), entry, directory / 'run_contract.json'

    def test_explicit_selection_accepts_real_endpoint_range(self):
        for step in (2001, 2517, 4000):
            value = self.selection(step)
            entry, output = module.validate_selection(value)
            self.assertEqual(entry['global_step'], step)
            self.assertEqual(output.parent, module.DEV)
        for step in (2000, 4001, True):
            with self.assertRaises(ValueError):
                module.validate_selection(self.selection(step))

    def test_selection_rejects_unknown_sha_smuggled_reference_and_existing_output(self):
        for field, bad in (('label', module.PARENT_LABEL), ('label', 'teacher500_k30'),
                           ('checkpoint_sha256', 'pending'), ('training_contract_sha256', 'A' * 64),
                           ('candidate_root', '/tmp/foreign'), ('epoch', 100)):
            value = self.selection()
            value['candidate'][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_selection(value)
        value = self.selection()
        value['primary_reference'] = module.EARLY_LABEL  # 主要对照固定，选择清单不能改成有利参照。
        with self.assertRaises(ValueError):
            module.validate_selection(value)
        with mock.patch.object(Path, 'exists', return_value=True), self.assertRaises(ValueError):
            module.validate_selection(self.selection())  # 只模拟存在性，不创建或覆盖真实目录。

    def test_original2000_lineage_and_configuration_are_accepted(self):
        module.validate_lineage(*self.contracts())

    def test_changed_objective_parent_guard_seed_or_source_are_rejected(self):
        changes = [
            ('kind', 'acdd2001_integration_smoke'), ('kind', 'conversion_score_matched_development'),
            ('method', 'conversion score'), ('seed', 999), ('stop_global_step', 6000),
            ('max_seconds', 3601), ('development_contract_file_sha256', 'canonical-not-file'),
            ('guard_ratios', {'mel_l1_upper': 1.2, 'prediction_rms_lower': 0.8}),
            ('file_sha256', {str(module.DRIVER): 'wrong'}), ('batch_plan', {})]
        for field, bad in changes:
            early, parent, child = self.contracts()
            child[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_lineage(early, parent, child)
        for field in ('parent_checkpoint_sha256', 'parent_contract_sha256', 'original_step0_checkpoint_sha256', 'start_after_global_step'):
            early, parent, child = self.contracts()
            child['lineage'][field] = 'wrong'
            with self.subTest(lineage=field), self.assertRaises(ValueError):
                module.validate_lineage(early, parent, child)

    def test_changed52_exclusion_or_loss_weight_is_rejected(self):
        for field in ('excluded_train_indices', 'training_indices', 'reconstruction_dev_indices'):
            early, parent, child = self.contracts()
            child[field] = child[field][1:]
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_lineage(early, parent, child)
        early, parent, child = self.contracts()
        child['configuration']['lambda_fm'] = 1.0  # 不能把调权重后的候选说成同目标延长训练。
        with self.assertRaises(ValueError):
            module.validate_lineage(early, parent, child)

    def test_final_snapshot_and_full_both_accept_only_exact_end_identity(self):
        values = self.completion()
        full_path, full_sha = module.validate_completion(*values)
        self.assertEqual(full_path.name, 'latest_full.pt')
        self.assertEqual(full_sha, 'a' * 64)
        completed, frozen, entry, path = values
        entry.update(checkpoint=str(full_path), checkpoint_sha256=full_sha)
        module.validate_completion(completed, frozen, entry, path)  # 低盘仅有full时也能如实评分。
        for step in (2001, 2727):
            module.validate_completion(*self.completion(step, 'original_step0_development_regression_guard'))

    def test_cannot_select_intermediate_snapshot_or_claim_smoke_failed_complete(self):
        for field, bad in (('status', 'running'), ('status', 'failed'), ('status', 'one_step_integration_smoke_complete'),
                           ('last_saved_global_step', 3000), ('global_step', 3999),
                           ('completed_new_steps', 1000), ('unsaved_completed_steps', 1)):
            completed, frozen, entry, path = self.completion()
            completed[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_completion(completed, frozen, entry, path)
        completed, frozen, entry, path = self.completion()
        entry.update(global_step=3000, checkpoint=str(path.parent / 'student_step_003000.pt'))
        with self.assertRaises(ValueError):
            module.validate_completion(completed, frozen, entry, path)
        with self.assertRaises(ValueError):
            module.validate_completion(*self.completion(3000))  # 3000不能冒充4000预算完成。

    def test_low_disk_reports_unsaved_steps_without_relabelling_saved_model(self):
        completed, frozen, entry, path = self.completion(3100, 'disk_space_guard_last_full_preserved')
        completed.update(global_step=3107, completed_new_steps=1107, unsaved_completed_steps=7)
        module.validate_completion(completed, frozen, entry, path)
        entry['global_step'] = 3107
        with self.assertRaises(ValueError):
            module.validate_completion(completed, frozen, entry, path)

    def test_frozen_full_receipt_and_snapshot_hash_must_match(self):
        for target in ('frozen', 'snapshot_sha', 'full_flag', 'foreign_full_path'):
            completed, frozen, entry, path = self.completion()
            if target == 'frozen':
                frozen['frozen'] = False
            elif target == 'snapshot_sha':
                entry['checkpoint_sha256'] = 'd' * 64
            elif target == 'full_flag':
                completed['latest_full']['full_committed'] = False
            else:
                completed['latest_full']['path'] = str(module.PARENT_DIR / 'full_step_002000.pt')
                frozen = dict(completed['latest_full'], frozen=True)
            with self.subTest(target=target), self.assertRaises(ValueError):
                module.validate_completion(completed, frozen, entry, path)

    def test_all90_actual_q950_hashes_are_required(self):
        report = {'x950_sha256_per_pair': {str(i): format(i, '064x') for i in range(90)}}
        module.same_noise(report, copy.deepcopy(report))
        for change in ('missing', 'different', 'malformed'):
            other = copy.deepcopy(report)
            if change == 'missing':
                del other['x950_sha256_per_pair']['89']
            else:
                other['x950_sha256_per_pair']['89'] = 'f' * 64 if change == 'different' else 'not-a-sha'
            with self.assertRaises(ValueError):
                module.same_noise(report, other)

    def test_paired_bootstrap_corpus_CER_direction_and_all_metrics(self):
        clusters = [(f'p{i}', f'u{i}') for i in range(10)]
        rows = [{'source_speaker': speaker, 'source_utt': utterance, 'char_errors': i + 1,
                 'reference_chars': 50 + i * 10, 'verified': 1, 'mos_ovr': 3.2, 'utmos': 3.7,
                 'mos_sig': 3.4, 'mos_bak': 4.0, 'p808_mos': 3.8, 'cosine_score': 0.75}
                for i, (speaker, utterance) in enumerate(clusters) for _ in range(9)]
        weights = module.base.bootstrap_weights(10, 10000, 2026090617)
        identical = module.compare_rows(rows, rows, clusters, weights)
        self.assertEqual(set(identical), set(module.base.METRICS))
        for item in identical.values():
            self.assertEqual(item['delta'], 0)
            self.assertEqual(item['cluster_bootstrap_95_percentile_ci'], [0, 0])
        changed = [{**r, 'char_errors': 0, 'utmos': 3.5} for r in rows]
        values = module.compare_rows(rows, changed, clusters, weights)
        corpus = 100 * sum(r['char_errors'] for r in rows) / sum(r['reference_chars'] for r in rows)
        self.assertAlmostEqual(values['CER_corpus_percent']['reference'], corpus)
        self.assertAlmostEqual(values['CER_corpus_percent']['delta'], -corpus)
        self.assertAlmostEqual(values['UTMOS']['delta'], -0.2)
        self.assertLess(values['UTMOS']['cluster_bootstrap_95_percentile_ci'][1], 0)
        self.assertEqual(values['SVA_percent']['candidate'], 100)


if __name__ == '__main__':
    unittest.main()  # 此入口不会运行main、不生成真实比较报告。
