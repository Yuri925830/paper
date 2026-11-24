# 租用服务器CPU测试，检查参数插值与训练目标判断，不构建真实模型。
import unittest  # 独立的小型数值检查。
import torch  # 全部测试张量留在CPU。
from probe_endpoint_actual_step import interpolate_state, training_supported_scales  # 导入不会进入真实GPU诊断。


class ActualStepProbeTests(unittest.TestCase):
    def test_interpolation_is_in_parameter_coordinates_and_preserves_archives(self):
        left = {'weight_g': torch.tensor([1.0, -2.0]), 'weight_v': torch.tensor([0.25, 0.5]), 'buffer': torch.tensor([2])}
        right = {'weight_g': torch.tensor([1.2, -1.7]), 'weight_v': torch.tensor([0.6, 0.8]), 'buffer': torch.tensor([2])}
        before_left = {key: value.clone() for key, value in left.items()}  # 保留两端未被改写的证据。
        before_right = {key: value.clone() for key, value in right.items()}
        names = {'weight_g', 'weight_v'}  # 模拟weight_norm的实际可训练参数。
        for scale in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):  # 与真实probe事先约定的比例相同。
            actual = interpolate_state(left, right, scale, names)
            for name in names:
                expected = (left[name].double() + scale * (right[name].double() - left[name].double())).float()
                torch.testing.assert_close(actual[name], expected, atol=0, rtol=0)  # 核对中间点及两个端点的原dtype值。
                self.assertNotEqual(actual[name].data_ptr(), left[name].data_ptr())  # 副本不能写回原归档。
            torch.testing.assert_close(actual['buffer'], left['buffer'], atol=0, rtol=0)  # 非参数缓冲保持原样。
        for name in left:
            torch.testing.assert_close(left[name], before_left[name], atol=0, rtol=0)  # 左端归档未变。
            torch.testing.assert_close(right[name], before_right[name], atol=0, rtol=0)  # 右端归档未变。

    def test_changed_buffer_and_invalid_scale_fail(self):
        left = {'weight': torch.tensor([1.0]), 'buffer': torch.tensor([0])}
        right = {'weight': torch.tensor([2.0]), 'buffer': torch.tensor([1])}  # 缓冲变化不能冒充Adam参数差。
        with self.assertRaises(ValueError):
            interpolate_state(left, right, 0.5, {'weight'})
        for scale in (-0.1, 1.1, float('nan')):  # 不允许外推或无效比例。
            with self.assertRaises(ValueError):
                interpolate_state(left, left, scale, {'weight'})

    def test_selection_uses_training_loss_not_development(self):
        rows = [  # 故意让开发最好点的训练目标恶化，确认不会拿它作步长支持证据。
            {'scale': 0.0, 'train': {'endpoint_l1': 1.0}, 'dev': {'endpoint_l1': 1.0}},
            {'scale': 0.25, 'train': {'endpoint_l1': 0.9}, 'dev': {'endpoint_l1': 1.2}},
            {'scale': 0.5, 'train': {'endpoint_l1': 1.1}, 'dev': {'endpoint_l1': 0.1}},
            {'scale': 1.0, 'train': {'endpoint_l1': 1.3}, 'dev': {'endpoint_l1': 0.2}},
        ]
        self.assertEqual(training_supported_scales(rows), [0.25])  # 只返回确有训练目标下降的已测比例。


if __name__ == '__main__':  # 导入时不运行测试或GPU主函数。
    unittest.main()  # 此命令仅在租用服务器CPU执行。
