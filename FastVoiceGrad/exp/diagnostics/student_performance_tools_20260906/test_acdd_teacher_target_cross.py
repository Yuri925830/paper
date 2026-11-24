# CPU只用小网络和真实生产目标公式，不加载实际教师/学生，更不启动训练。
"""CPU acceptance for the fixed-input crossed teacher-target diagnostic."""

import ast  # 静态确认主诊断没有反传、优化器、声码器或权重保存调用。
import copy  # 拒绝案例使用独立元信息副本。
import json
import os  # CPU测试必须明确隐藏CUDA，并在租用服务器执行。
import tempfile  # 仅创建KB级测试夹具，结束后自动清理本次目录。
import unittest
from pathlib import Path
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError("Run only on Matpool CPU with CUDA_VISIBLE_DEVICES=''")
import torch
from torch import nn
import inspect_acdd_teacher_target_cross_v1 as d


class TinyNoise(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)
        self.register_buffer('unchanged_buffer', torch.tensor(3.), persistent=False)
        self.calls = []  # 只记录测试调用次数，不属于模型可训练参数或buffer。

    def forward(self, x, t, speaker, bnf):
        self.calls.append((speaker.detach().clone(), bnf.detach().clone(), torch.is_grad_enabled()))
        # 输出显式依赖两种条件，能识别误传speaker或BNF的实现错误。
        return self.scale * x + speaker.mean(1)[:, None, None] * .001 + bnf.mean(1, keepdim=True) * .002


def example():
    generator = torch.Generator().manual_seed(8921)
    shape = (32, 80, 128)  # 与真实固定批次同形状，覆盖两层mean的实际浮点计算。
    inputs = {name: torch.randn(shape, generator=generator) for name in ('x0', 'source_noise', 'score_noise')}
    inputs['bnf'] = torch.randn((32, 144, 128), generator=generator)
    inputs['spk_emb'] = torch.randn((32, 256), generator=generator)
    inputs['score_timesteps'] = torch.linspace(1, 1000, 32).round().long()
    config = d.base.replace(d.base.PAPER_STUDENT_CONFIG, num_workers=0)
    diffusion = d.base.VoiceGradDiffusion(1000, .008)
    inputs['q950'] = diffusion.q_sample(inputs['x0'], torch.full((32,), 950, dtype=torch.long), inputs['source_noise'])
    return inputs, config, diffusion


class TargetCrossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_production_forward_twice_only_no_grad_unchanged_conditions(self):
        inputs, config, diffusion = example()
        student, teacher = TinyNoise(.03).train(), TinyNoise(.04).eval()
        before = d.probe.tree_sha(inputs)
        teacher_before = d.pair.model_sha(teacher)
        for _ in (3227, 3247):
            prediction, target, hashes = d.endpoint_forward(student, teacher, diffusion, inputs, config)
            self.assertEqual(prediction.shape, (32, 80, 128))
            self.assertFalse(prediction.requires_grad or target.requires_grad)
            self.assertEqual(hashes['teacher'], teacher_before)
        self.assertEqual(len(student.calls), 2)
        self.assertEqual(len(teacher.calls), 2)
        self.assertTrue(student.training)
        self.assertFalse(teacher.training)
        self.assertEqual(d.probe.tree_sha(inputs), before)
        for module in (student, teacher):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
            for speaker, bnf, grad_enabled in module.calls:
                self.assertTrue(torch.equal(speaker, inputs['spk_emb']))
                self.assertTrue(torch.equal(bnf, inputs['bnf']))
                self.assertFalse(grad_enabled)

    def test_l1_matches_actual_production_weighting_and_unweighted(self):
        inputs, config, diffusion = example()
        teacher = TinyNoise(.04).eval()
        prediction = inputs['x0'].clone().requires_grad_(True)
        score, target = d.training.score_distillation_loss(prediction, teacher, diffusion, inputs['score_timesteps'],
            inputs['score_noise'], inputs['spk_emb'], inputs['bnf'], target_mode='clean_x0', weight_mode='sqrt_alpha_bar')
        weights = diffusion.sqrt_alphas_cumprod[inputs['score_timesteps']]
        record = d.l1_readout(prediction.detach(), target, weights)
        self.assertEqual(record['weighted'], float(score.detach()))
        manual = (prediction.detach() - target).abs().flatten(1).mean(1)
        self.assertEqual(record['per_utterance_unweighted'], manual.tolist())
        self.assertEqual(record['per_utterance_weighted'], (weights * manual).tolist())
        self.assertFalse(target.requires_grad)
        # 真生产SG梯度只沿学生输出；这里CPU求一次解析核对，不存在优化器或模型更新。
        gradient, = torch.autograd.grad(score, prediction)
        expected = weights[:, None, None] * (prediction.detach() - target).sign() / prediction.numel()
        self.assertTrue(torch.allclose(gradient, expected, rtol=1e-6, atol=1e-12))

    def test_cross_matrix_order_two_paths_and_all32(self):
        y0 = torch.zeros(32, 2, 3)
        y1 = torch.ones_like(y0)
        r0, r1 = torch.zeros_like(y0), torch.ones_like(y0) * 2
        weights = torch.linspace(.01, 1., 32)
        cells, splits = d.crossed_readouts({3227: y0, 3247: y1}, {3227: r0, 3247: r1}, weights)
        self.assertEqual(cells['y3227_R3227']['unweighted'], 0.)
        self.assertEqual(cells['y3247_R3227']['unweighted'], 1.)
        self.assertEqual(cells['y3227_R3247']['unweighted'], 2.)
        self.assertEqual(cells['y3247_R3247']['unweighted'], 1.)
        for metric, row in splits.items():
            self.assertEqual(len(row['per_utterance']), 32)
            for entry in [row] + row['per_utterance']:
                self.assertAlmostEqual(entry['student_change_at_R3227'] + entry['target_change_at_y3247'], entry['diagonal_change'], places=12)
                self.assertAlmostEqual(entry['target_change_at_y3227'] + entry['student_change_at_R3247'], entry['diagonal_change'], places=12)
                self.assertAlmostEqual(entry['symmetric_student_component'] + entry['symmetric_target_component'], entry['diagonal_change'], places=12)

    def test_known_target_motion_can_offset_student_fixed_target_increase(self):
        row = d.split_scalar(2., 3., 0., 1.)
        self.assertEqual(row['diagonal_change'], -1.)
        self.assertEqual(row['student_change_at_R3227'], 1.)
        self.assertEqual(row['target_change_at_y3247'], -2.)
        self.assertEqual(row['symmetric_student_component'], 1.)
        self.assertEqual(row['symmetric_target_component'], -2.)
        with self.assertRaises(RuntimeError):
            d.split_scalar(0., 1., float('nan'), 2.)

    def test_no_motion_and_symmetric_swap(self):
        self.assertEqual(d.split_scalar(1., 1., 1., 1.)['diagonal_change'], 0.)
        forward = d.split_scalar(.2, .4, .3, .1)
        backward = d.split_scalar(.1, .3, .4, .2)
        for key in ('diagonal_change', 'symmetric_student_component', 'symmetric_target_component'):
            self.assertAlmostEqual(forward[key], -backward[key], places=12)

    def test_bad_shape_nonfinite_weight_and_wrong_q950_rejected(self):
        y = torch.zeros(32, 2, 3)
        for weights in (torch.ones(31), torch.ones(32, 1), torch.full((32,), float('nan')), -torch.ones(32)):
            with self.assertRaises(RuntimeError):
                d.l1_readout(y, y, weights)
        with self.assertRaises(RuntimeError):
            d.l1_readout(y, y[:, :, :1], torch.ones(32))
        inputs, config, diffusion = example()
        inputs['q950'][0, 0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, 'exactq950'):
            d.endpoint_forward(TinyNoise(.03).eval(), TinyNoise(.04).eval(), diffusion, inputs, config)

    def test_frozen_parameters_and_nonpersistent_buffers_are_checked(self):
        inputs, config, diffusion = example()
        student, teacher = TinyNoise(.03).eval(), TinyNoise(.04).eval()
        student.scale.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            d.endpoint_forward(student, teacher, diffusion, inputs, config)
        student.scale.requires_grad_(False)
        hook = teacher.register_forward_hook(lambda module, args, result: module.unchanged_buffer.add_(1) * 0 + result)
        with self.assertRaises(RuntimeError):
            d.endpoint_forward(student, teacher, diffusion, inputs, config)
        hook.remove()

    def test_replay_declared_tolerance_nan_and_real_archived_records(self):
        actual = {'primary_moving_score': .05, 'diagnostic_frozen_target_score': .06}
        archived = dict(actual)
        self.assertEqual(d.verify_replay(actual, archived), {key: 0. for key in actual})
        for value in (.0601, float('nan'), float('inf')):
            with self.assertRaises(RuntimeError):
                d.verify_replay(dict(actual, diagnostic_frozen_target_score=value), archived)
        for path in (d.PAIRED / 'primary_s1e4/arm_completion.json', d.CONTINUED / 'completion.json'):
            row = d.read_json(path)['last_monitor']['fixed_train']
            self.assertTrue(all(delta == 0 for delta in d.verify_replay(row, row).values()))
        self.assertEqual(d.REPLAY_ATOL, 2e-5)

    def test_payload_role_step_configuration_and_score_identity(self):
        config = d.base.replace(d.base.PAPER_STUDENT_CONFIG, num_workers=0)
        row = {'global_step': 3247, 'run_contract_sha256': d.STATES[3247]['contract_sha256'],
               'configuration': config.as_dict(), 'effective_optimizer_lrs': {'student': .0001, 'discriminator': .0002},
               'score_semantics': 'reconstruction_clean_x0_sqrt_alpha_bar'}
        d.validate_payload(row, 3247, config)
        for key, wrong in (('global_step', 3227), ('run_contract_sha256', 'wrong'), ('configuration', {}),
                           ('effective_optimizer_lrs', {'student': .0002, 'discriminator': .0002}), ('score_semantics', 'legacy')):
            changed = copy.deepcopy(row)
            changed[key] = wrong
            with self.assertRaises(RuntimeError):
                d.validate_payload(changed, 3247, config)

    def test_disk_gpu_busy_gates_and_exact_new_protocol_sha(self):
        with mock.patch.object(d.os, 'statvfs', return_value=mock.Mock(f_bavail=1, f_frsize=1024)):
            with self.assertRaisesRegex(RuntimeError, 'reserve'):
                d.capacity(d.DEV)
        with mock.patch.object(d.subprocess, 'run', return_value=mock.Mock(stdout='999999, other-job')):
            with self.assertRaisesRegex(RuntimeError, 'Another GPU'):
                d.gpu_idle()
        with mock.patch.object(d.subprocess, 'run', side_effect=[mock.Mock(stdout=''), mock.Mock(stdout='1000\n')]):
            with self.assertRaisesRegex(RuntimeError, '8GiB'):
                d.gpu_idle()
        self.assertEqual(d.base.sha256_file(d.PROTOCOL), d.PROTOCOL_SHA)

    def test_real_checkpoint_metadata_fixed_shapes_and_old_anchor_identity(self):
        # CPU只读取两份真实S归档字段，不构建实际网络、不计算实际预测。
        config = d.base.replace(d.base.PAPER_STUDENT_CONFIG, num_workers=0)
        parent, child = d.read_json(d.PAIRED / 'run_contract.json'), d.read_json(d.CONTINUED / 'run_contract.json')
        for step, contract in ((3227, parent), (3247, child)):
            info = d.STATES[step]
            path = info['directory'] / info['file']
            self.assertEqual(d.base.sha256_file(path), info['sha256'])
            payload = d.resume.load_torch_checkpoint(path)
            d.validate_payload(payload, step, config)
            self.assertEqual(d.training.canonical_json_sha256(contract), info['contract_sha256'])
            del payload
        inputs = d.resume.load_torch_checkpoint(d.pair.INPUT_DIR / 'fixed_inputs.pt')
        self.assertEqual({key: d.identity(value)['sha256'] for key, value in inputs.items()}, parent['input_sha256'])
        self.assertEqual(parent['input_sha256'], child['fixed_train_inputs_sha256'])
        for key in ('x0', 'source_noise', 'score_noise', 'q950'):
            self.assertEqual(inputs[key].shape, (32, 80, 128))
            self.assertEqual(inputs[key].dtype, torch.float32)
        self.assertEqual(inputs['bnf'].shape, (32, 144, 128))
        self.assertEqual(inputs['spk_emb'].shape, (32, 256))
        old = d.resume.load_torch_checkpoint(d.PAIRED / 'fixed_teacher_target.pt')
        self.assertEqual(old['global_source_step'], 3107)
        self.assertEqual(old['sha256'], d.identity(old['teacher_target'])['sha256'])
        self.assertEqual(old['sha256'], parent['fixed_teacher_target_sha256'])

    def test_main_static_has_no_update_or_extra_model_call_or_weight_output(self):
        tree = ast.parse(Path(d.__file__).read_text())
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)] if hasattr(ast, 'unparse') else []
        # Python3.8无ast.unparse，用属性名同样能检查禁止的调用。
        attributes = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for banned in ('backward', 'grad', 'step', 'Adam', 'atomic_torch_save', 'save', 'load_frozen_vocoder', 'build_paper_discriminator'):
            self.assertNotIn(banned, attributes)
        self.assertEqual(attributes.count('make_student_prediction'), 1)
        self.assertEqual(attributes.count('score_distillation_loss'), 1)
        self.assertIn('acquire_runtime_locks', attributes)
        self.assertIn('assert_rented_server', attributes)
        self.assertIn('collect_source_fingerprints', attributes)
        self.assertIn('preserve_rng', attributes)
        self.assertIn('replace', attributes)  # 小JSON写完之后原子改名，不保存模型。


if __name__ == '__main__':
    unittest.main(verbosity=2)
