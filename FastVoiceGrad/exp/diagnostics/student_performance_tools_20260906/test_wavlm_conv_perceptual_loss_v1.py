"""租用服务器 CPU toy tests：仅合成21张量，不下载/加载真实WavLM，不写文件。"""
import ast
import math
import os
from pathlib import Path
import unittest
from unittest import mock

if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Only GPU-hidden rental CPU tests may run')
import wavlm_conv_perceptual_loss_v1 as n
torch = n.torch


def synthetic_state():
    generator = torch.Generator(device='cpu').manual_seed(70907)
    result = {}
    for key, shape in n.expected_state_shapes().items():
        if key.endswith('conv.weight'):
            result[key] = torch.randn(shape, generator=generator) / math.sqrt(shape[1] * shape[2])
        else:
            result[key] = torch.ones(shape) if key.endswith('.weight') else torch.zeros(shape)
    return result


def waves(length=1103, batch=1):
    time = torch.arange(length, dtype=torch.float32) / 22050
    generated = 0.17 * torch.sin(2 * math.pi * 417 * time) + 0.04 * torch.sin(2 * math.pi * 1801 * time)
    target = 0.13 * torch.sin(2 * math.pi * 489 * time) + 0.025 * torch.cos(2 * math.pi * 1403 * time)
    scales = torch.arange(1, batch + 1, dtype=torch.float32)[:, None]
    return (generated[None] * scales).clone(), (target[None] / scales).clone()


class WavLMConvPerceptualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.state = synthetic_state()
        cls.encoder = n.FrozenWavLMConvEncoder(cls.state)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def test_exact_21_tensor_state_keys_and_4206592_parameters(self):
        self.assertEqual(set(self.encoder.state_dict()), set(n.expected_state_shapes()))
        self.assertEqual(len(self.encoder.state_dict()), 21)
        self.assertEqual(sum(p.numel() for p in self.encoder.parameters()), 4206592)
        self.assertEqual(len(self.encoder.conv_layers), 7)
        for index, layer in enumerate(self.encoder.conv_layers):
            self.assertEqual(layer.conv.kernel_size, (n.KERNELS[index],))
            self.assertEqual(layer.conv.stride, (n.STRIDES[index],))
            self.assertEqual(layer.conv.out_channels, 512); self.assertIsNone(layer.conv.bias)
            self.assertEqual(layer.layer_norm.normalized_shape, (512,))
            self.assertEqual(layer.layer_norm.eps, 1e-5)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.encoder.parameters()))
        self.assertTrue(all(not m.training for m in self.encoder.modules()))

    def test_rejects_extra_missing_wrong_shape_dtype_nonfinite_and_prefixed_state(self):
        key = 'conv_layers.0.conv.weight'
        malformed = [dict(self.state, projection=torch.zeros(1)), {k: v for k, v in self.state.items() if k != key},
            dict(self.state, **{key: torch.zeros(512, 1, 9)}), dict(self.state, **{key: self.state[key].double()}),
            dict(self.state, **{key: torch.full_like(self.state[key], float('nan'))}),
            {'feature_extractor.' + k: v for k, v in self.state.items()}]
        for state in malformed:
            with self.subTest(keys=len(state)), self.assertRaises(ValueError): n.validate_conv_state(state)

    def test_constructor_preserves_global_cpu_rng_and_keeps_no_cuda_context(self):
        before = torch.get_rng_state().clone()
        encoder = n.FrozenWavLMConvEncoder(self.state)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized()); encoder.assert_frozen()

    def test_receptive_field_stride_and_final_feature_shape(self):
        field, jump = 1, 1
        for kernel, stride in zip(n.KERNELS, n.STRIDES):
            field += (kernel - 1) * jump; jump *= stride
        self.assertEqual((field, jump), (400, 320))
        self.assertEqual([n.feature_frames(x) for x in (400, 719, 720, 16000)], [1, 1, 2, 49])
        self.assertEqual(tuple(self.encoder(waves(720)[0]).shape), (1, 512, 2))
        with self.assertRaises(ValueError): self.encoder(torch.zeros(1, 399))

    def test_manual_conv_layernorm_gelu_reference_matches_last_layer_only(self):
        inputs = waves(720)[0]; hidden = inputs[:, None, :]
        for index, (kernel, stride) in enumerate(zip(n.KERNELS, n.STRIDES)):
            prefix = 'conv_layers.%d.' % index
            hidden = n.F.conv1d(hidden, self.state[prefix + 'conv.weight'], stride=stride)
            hidden = n.F.layer_norm(hidden.transpose(1, 2), (512,), self.state[prefix + 'layer_norm.weight'],
                self.state[prefix + 'layer_norm.bias'], 1e-5).transpose(1, 2)
            hidden = n.F.gelu(hidden, approximate='none')
        torch.testing.assert_close(self.encoder(inputs), hidden, rtol=0, atol=0)

    def test_identical_waveforms_have_exact_zero_in_every_component(self):
        generated, _ = waves(batch=2)
        result = n.wavlm_conv_perceptual_loss(self.encoder, generated, generated.clone(), [1103, 1103])
        for key in ('loss', 'feature_mse', 'feature_weighted', 'stft_magnitude_l1', 'per_example_loss'):
            self.assertEqual(float(result[key].abs().max()), 0.0, key)

    def test_student_gradient_is_finite_nonzero_teacher_and_encoder_are_detached(self):
        generated, target = waves(); generated.requires_grad_(True); target.requires_grad_(True)
        before = torch.get_rng_state().clone()
        result = n.wavlm_conv_perceptual_loss(self.encoder, generated, target, [1103])
        feature_gradient, = torch.autograd.grad(result['feature_mse'], generated, retain_graph=True)
        self.assertTrue(bool(torch.isfinite(feature_gradient).all())); self.assertGreater(float(feature_gradient.abs().sum()), 0)
        result['loss'].backward()
        self.assertTrue(bool(torch.isfinite(generated.grad).all())); self.assertGreater(float(generated.grad.abs().sum()), 0)
        self.assertIsNone(target.grad); self.encoder.assert_frozen()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_gradient_passes_through_frozen_vocoder_without_parameter_gradient(self):
        with torch.random.fork_rng(devices=[]): vocoder = torch.nn.Conv1d(1, 1, 1, bias=False)
        with torch.no_grad(): vocoder.weight.fill_(0.7)
        vocoder.requires_grad_(False).eval()
        latent, target = waves(); latent = latent[:, None, :].clone().requires_grad_(True)
        result = n.wavlm_conv_perceptual_loss(self.encoder, vocoder(latent), target[:, None, :], [1103])
        result['loss'].backward()
        self.assertGreater(float(latent.grad.abs().sum()), 0); self.assertTrue(bool(torch.isfinite(latent.grad).all()))
        self.assertIsNone(vocoder.weight.grad); self.encoder.assert_frozen()

    def test_normalization_uses_per_wave_population_variance_plus_1e_minus7_epsilon(self):
        wave = torch.tensor([[1.0, 2.0, 4.0, 8.0], [-3.0, 1.0, 3.0, 7.0]])
        expected = (wave - wave.mean(-1, keepdim=True)) / torch.sqrt(wave.var(-1, unbiased=False, keepdim=True) + 1e-7)
        torch.testing.assert_close(n.normalize_for_features(wave), expected)
        wrong = (wave - wave.mean(-1, keepdim=True)) / torch.sqrt(wave.var(-1, unbiased=True, keepdim=True) + 1e-7)
        self.assertFalse(torch.allclose(expected, wrong))
        torch.testing.assert_close(n.normalize_for_features(torch.ones(2, 900)), torch.zeros(2, 900))

    def test_resampling_parameters_and_raw_magnitude_stft_are_exact(self):
        from torchaudio.functional import resample
        generated, target = waves()
        expected_kwargs = dict(orig_freq=22050, new_freq=16000, lowpass_filter_width=6,
            rolloff=0.99, resampling_method='sinc_interpolation', beta=None)
        self.assertEqual(n.RESAMPLE_KWARGS, expected_kwargs)
        with mock.patch('torchaudio.functional.resample', wraps=resample) as call:
            result = n.wavlm_conv_perceptual_loss(self.encoder, generated, target, [1103])
        self.assertEqual(call.call_count, 2)
        for args in call.call_args_list: self.assertEqual(args.kwargs, expected_kwargs)
        g16, t16 = resample(generated, **expected_kwargs), resample(target, **expected_kwargs)
        options = dict(n_fft=512, hop_length=128, win_length=512, window=torch.hann_window(512, periodic=True),
            center=False, normalized=False, onesided=True, return_complex=True)
        # 与约定一致：先每句的有效元素求均值，再句间平均；不换成一次全局归约。
        manual_l1 = (torch.stft(g16, **options).abs() - torch.stft(t16, **options).abs()).abs().flatten(1).mean(1).mean()
        torch.testing.assert_close(result['stft_magnitude_l1'], manual_l1, rtol=0, atol=0)
        gain = n.wavlm_conv_perceptual_loss(self.encoder, generated * 2, generated, [1103])
        self.assertGreater(float(gain['stft_magnitude_l1']), 0)

    def test_exact_100_feature_weight_and_equal_per_utterance_mean(self):
        generated, target = waves(batch=2)
        result = n.wavlm_conv_perceptual_loss(self.encoder, generated, target, [800, 1103])
        expected = 100 * result['per_example_feature_mse'] + result['per_example_stft_magnitude_l1']
        torch.testing.assert_close(result['per_example_loss'], expected, rtol=0, atol=0)
        torch.testing.assert_close(result['loss'], expected.mean(), rtol=0, atol=0)
        torch.testing.assert_close(result['feature_weighted'], (100 * result['per_example_feature_mse']).mean(), rtol=0, atol=0)

    def test_callable_wrapper_and_compatibility_aliases_preserve_the_same_tensors(self):
        generated, target = waves()
        criterion = n.WavLMConvPerceptualLoss(self.encoder).to('cpu').eval()
        result = criterion(generated, target, [1103])
        self.assertIs(result['stft_l1'], result['stft_magnitude_l1'])
        self.assertIs(result['per_utterance_loss'], result['per_example_loss'])
        direct = n.wavlm_conv_perceptual_loss(self.encoder, generated, target, [1103])
        torch.testing.assert_close(result['loss'], direct['loss'], rtol=0, atol=0)
        self.assertTrue(all(not parameter.requires_grad for parameter in criterion.parameters()))

    def test_variable_length_fallback_matches_independent_examples_and_padding_is_ignored(self):
        generated, target = waves(batch=2); lengths = [800, 1103]
        baseline = n.wavlm_conv_perceptual_loss(self.encoder, generated, target, lengths)
        padded_g, padded_t = generated.clone(), target.clone()
        padded_g[0, 800:] = float('nan'); padded_t[0, 800:] = float('nan')
        padded_g.requires_grad_(True)
        actual = n.wavlm_conv_perceptual_loss(self.encoder, padded_g, padded_t, torch.tensor(lengths, dtype=torch.long))
        self.assertEqual(actual['execution_path'], 'per_example_variable_length')
        torch.testing.assert_close(actual['per_example_loss'], baseline['per_example_loss'], rtol=0, atol=0)
        for index, length in enumerate(lengths):
            single = n.wavlm_conv_perceptual_loss(self.encoder, generated[index:index+1, :length], target[index:index+1, :length], [length])
            torch.testing.assert_close(actual['per_example_loss'][index], single['loss'], rtol=0, atol=0)
        actual['loss'].backward(); self.assertEqual(float(padded_g.grad[0, 800:].abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(padded_g.grad).all()))

    def test_uniform_valid_lengths_use_batch_fast_path_and_crop_before_resampling(self):
        generated, target = waves(batch=2)
        with mock.patch.object(self.encoder, 'forward', wraps=self.encoder.forward) as calls:
            result = n.wavlm_conv_perceptual_loss(self.encoder, generated, target, [800, 800])
        self.assertEqual(calls.call_count, 2); self.assertEqual(result['execution_path'], 'same_valid_length_batch')
        self.assertEqual(result['resampled_lengths'], [581, 581]); self.assertEqual(result['feature_frame_lengths'], [1, 1])
        cropped = n.wavlm_conv_perceptual_loss(self.encoder, generated[:, :800], target[:, :800], [800, 800])
        torch.testing.assert_close(result['per_example_loss'], cropped['per_example_loss'], rtol=0, atol=0)

    def test_short_invalid_or_mismatched_inputs_rejected_without_padding(self):
        generated, target = waves()
        for lengths in ([704], [0], [1104], [True], [1103.0], [], torch.tensor([1103], dtype=torch.int32)):
            with self.subTest(lengths=str(lengths)), self.assertRaises(ValueError):
                n.wavlm_conv_perceptual_loss(self.encoder, generated, target, lengths)
        minimum = n.wavlm_conv_perceptual_loss(self.encoder, generated[:, :705], target[:, :705], [705])
        self.assertEqual(minimum['resampled_lengths'], [512])
        with self.assertRaises(ValueError): n.wavlm_conv_perceptual_loss(self.encoder, generated.double(), target, [1103])
        with self.assertRaises(ValueError): n.wavlm_conv_perceptual_loss(self.encoder, generated, target[:, :-1], [1102])
        with self.assertRaises(ValueError): n.stft_magnitude(torch.zeros(1, 511))
        corrupt = generated.clone(); corrupt[0, 1] = float('inf')
        with self.assertRaises(ValueError): n.wavlm_conv_perceptual_loss(self.encoder, corrupt, target, [1103])

    def test_accidental_training_or_trainable_encoder_fails(self):
        self.encoder.train()
        try:
            with self.assertRaises(RuntimeError): self.encoder.assert_frozen()
        finally: self.encoder.eval()
        parameter = next(self.encoder.parameters()); parameter.requires_grad_(True)
        try:
            with self.assertRaises(RuntimeError): self.encoder.assert_frozen()
        finally: parameter.requires_grad_(False)

    def test_module_has_no_full_model_metric_head_optimizer_file_write_or_download(self):
        tree = ast.parse(Path(n.__file__).read_bytes())
        calls = [ast.dump(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for forbidden in ('torch.load', 'torch.save', 'torch.optim.Adam', 'subprocess.run', 'subprocess.Popen'):
            self.assertNotIn(ast.dump(ast.parse(forbidden, mode='eval').body), calls)
        imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(x.startswith(('transformers', 'huggingface_hub')) for x in imports))
        self.assertFalse(any('transformer' in name or 'projection' in name or 'head' in name
            for name, _ in self.encoder.named_modules()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__': unittest.main(verbosity=2)
