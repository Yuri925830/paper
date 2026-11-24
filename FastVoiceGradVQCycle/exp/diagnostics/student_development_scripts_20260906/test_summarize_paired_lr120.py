"""租用服务器CPU小型测试：核对真实窗口、共同前缀和早停界限。"""

import copy
import json
import unittest
import summarize_paired_lr120 as report


def fixture(count=60):
    steps, windows, monitors = [], [], []
    for update in range(1, count + 1):
        row = {'global_step': 3107 + update, 'updates': update, 'indices': list(range(update, update + 32)),
               'actual_optimizer_lrs': {'student': .0002, 'discriminator': .0002},
               'student_adv': 2., 'fm_raw': .5, 'fm_weighted': 1., 'dist_raw': .05, 'dist_weighted': 2.25,
               'student_total': 5.25, 'discriminator': 4.}
        steps.append(row)
        if update % 20 == 0:
            windows.append({'global_step': 3107 + update, 'updates': update, 'first_global_step': 3107 + update - 19,
                            'count': 20, 'means': {key: row[key] for key in report.LOSS_KEYS}})
    for update in sorted({0, count, *range(20, count + 1, 20)}):
        monitors.append({'global_step': 3107 + update, 'updates': update,
            'development': {'mel_l1': .2, 'per_utterance_mel_l1': [.2] * 32},
            'fixed_train': {'primary_moving_score': .05, 'primary_moving_total': 5.25,
                'diagnostic_frozen_target_score': .05, 'diagnostic_frozen_target_total': 5.25,
                'adv': 2., 'fm_raw': .5, 'train_gt_l1': .19}})
    completion = {'updates': count, 'completed_global_step': 3107 + count, 'status': 'complete', 'reason': 'bounded_endpoint',
                  'effective_optimizer_lrs': {'student': .0002, 'discriminator': .0002}}
    return steps, windows, monitors, completion


class ComparisonTests(unittest.TestCase):
    def test_actual_readonly3107_monitor_schema(self):
        # 使用已经完成的真实诊断JSON，避免测试替身和生产字段一起写错。
        real = json.loads((report.DEV / 'acdd_adam_descent_readonly_3107_v1/results.json').read_text())['models']['3107']
        rows = fixture()
        for row in rows[2]:
            row['development'] = copy.deepcopy(real['before_dev'])
            row['fixed_train'] = copy.deepcopy(real['before_train'])
        validated = report.validate_arm(*rows)
        self.assertEqual(validated['monitors'][0]['development']['mel_l1'], .20179936289787292)

    def test_valid_all_steps_and_twenty_step_means(self):
        arm = report.validate_arm(*fixture())
        self.assertEqual(len(arm['steps']), 60)
        self.assertEqual(list(arm['by_monitor']), [0, 20, 40, 60])

    def test_missing_duplicate_or_changed_step_rejected(self):
        for mode in ('missing', 'duplicate', 'global'):
            rows = list(fixture())
            if mode == 'missing':
                rows[0].pop(3)
            elif mode == 'duplicate':
                rows[0].append(copy.deepcopy(rows[0][-1]))
            else:
                rows[0][3]['global_step'] += 1
            with self.assertRaises(ValueError):
                report.validate_arm(*rows)

    def test_changed_weight_mean_or_nan_rejected(self):
        for mode in ('weight', 'mean', 'nan', 'lr'):
            rows = fixture()
            if mode == 'weight':
                rows[0][0]['fm_weighted'] = .5
            elif mode == 'mean':
                rows[1][0]['means']['student_total'] += .01
            elif mode == 'lr':
                rows[0][0]['actual_optimizer_lrs']['discriminator'] = .0001
            else:
                rows[0][0]['dist_raw'] = float('nan')
            with self.assertRaises(ValueError):
                report.validate_arm(*rows)

    def test_unequal_endpoints_keep_full_arms_and_shared_prefix(self):
        arms = {report.ARMS[0]: report.validate_arm(*fixture(60)), report.ARMS[1]: report.validate_arm(*fixture(120))}
        result = report.compare(arms)
        self.assertEqual(result['latest_common_monitor_updates'], 60)
        self.assertEqual([row['updates'] for row in result['common_window_rows']], [20, 40, 60])
        self.assertEqual(result['arms'][report.ARMS[1]]['completion']['updates'], 120)
        self.assertEqual(len(result['arms'][report.ARMS[1]]['complete_window_total_values']), 6)

    def test_pairing_mismatch_and_first_update_loss_mismatch_rejected(self):
        for mode in ('indices', 'first_loss'):
            left, right = report.validate_arm(*fixture()), report.validate_arm(*fixture())
            if mode == 'indices':
                right['by_step'][5]['indices'][0] = 999
            else:
                right['by_step'][1]['student_total'] += .1
            with self.assertRaises(ValueError):
                report.compare(dict(zip(report.ARMS, (left, right))))

    def test_exact_three_value_stop_definition(self):
        rows = [{'updates': step, 'value': value} for step, value in zip((20, 40, 60), (5., 5.1, 5.2))]
        stop = report.rising_triplets(rows, lambda row: row['value'])
        self.assertEqual(stop[0]['updates'], [20, 40, 60])
        self.assertEqual(report.rising_triplets(rows[:2], lambda row: row['value']), [])
        rows[1]['value'] = 5.
        self.assertEqual(report.rising_triplets(rows, lambda row: row['value']), [])

    def test_gt_stop_uses_arm_initial_anchor_not_triplet_start(self):
        rows = [{'updates': step, 'value': value} for step, value in zip((20, 40, 60), (.201, .2015, .2025))]
        stop = report.rising_triplets(rows, lambda row: row['value'], relative_start=.2)
        self.assertEqual(len(stop), 1)
        self.assertAlmostEqual(stop[0]['relative_change'], .0125)

    def test_stop_threshold_uses_exact_same_ratio_expression_as_driver(self):
        loss = [{'updates': step, 'value': value} for step, value in zip((20, 40, 60), (5., 5.05, 5.1))]
        self.assertEqual(len(report.rising_triplets(loss, lambda row: row['value'])), 1)
        dev = [{'updates': step, 'value': value} for step, value in zip((0, 20, 40), (.2, .201, .202))]
        self.assertEqual(len(report.rising_triplets(dev, lambda row: row['value'], relative_start=.2)), 1)

    def test_running_past_declared_stop_is_rejected(self):
        arms = {label: report.validate_arm(*fixture(120)) for label in report.ARMS}
        for row, value in zip(arms[report.ARMS[0]]['windows'], (5., 5.05, 5.1)):
            row['means']['student_total'] = value
        with self.assertRaisesRegex(ValueError, 'continued after'):
            report.compare(arms)


if __name__ == '__main__':
    unittest.main()
