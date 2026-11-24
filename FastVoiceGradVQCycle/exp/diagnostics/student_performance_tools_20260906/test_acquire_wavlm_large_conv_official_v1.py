"""租用CPU小测试：内存流、来源准入、21参数白名单及不可覆盖；无真实下载。"""
import ast
import hashlib
import importlib.util
import inspect
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('wavlm_acquire_under_test', Path(__file__).with_name('acquire_wavlm_large_conv_official_v1.py'))
asset = importlib.util.module_from_spec(spec); spec.loader.exec_module(asset)


class StreamTests(unittest.TestCase):
    def test_explicit_approval_and_full_sha(self):
        args = ['--transport', 'https', '--driver-sha256', 'a' * 64, '--tests-sha256', 'b' * 64, '--protocol-sha256', 'c' * 64]
        with self.assertRaises(RuntimeError): asset.arguments(args)
        self.assertTrue(asset.arguments(['--run-approved'] + args).run_approved)
        with self.assertRaises(RuntimeError): asset.arguments(['--run-approved'] + args[:-1] + ['C' * 64])

    def test_fixed_official_revision_and_redirect_scope(self):
        self.assertIn('/resolve/' + asset.REVISION + '/', asset.MODEL_URL)
        self.assertNotIn('/main/', asset.MODEL_URL)
        for url in ('https://huggingface.co/path', 'https://cas-bridge.xethub.hf.co/path?temporary=redacted'):
            self.assertTrue(asset.require_https(url))
        for url in ('http://huggingface.co/x', 'https://huggingface.co.evil.example/x', 'https://evil.example/x', 'https://user:pass@huggingface.co/x'):
            with self.assertRaises(RuntimeError): asset.require_https(url)

    def test_stdin_config_identity_is_explicit_and_fixed(self):
        args = ['--run-approved', '--transport', 'stdin', '--driver-sha256', 'a' * 64, '--tests-sha256', 'b' * 64, '--protocol-sha256', 'c' * 64]
        with self.assertRaises(RuntimeError): asset.arguments(args)
        with self.assertRaises(RuntimeError): asset.arguments(args + ['--config-sha256', '0' * 64])
        self.assertEqual(asset.arguments(args + ['--config-sha256', asset.CONFIG_SHA]).transport, 'stdin')
        with self.assertRaises(RuntimeError): asset.parse_verified_config(b'{}')

    def test_stdin_and_https_share_identical_byte_verification(self):
        payload = b'exact binary\x00\xff\n\r\x1a'
        expected = hashlib.sha256(payload).hexdigest()
        class SocketLike(io.BytesIO): pass
        direct = asset.stream_to_ram(SocketLike(payload), len(payload), expected, lambda: None, 4)
        stdin = asset.stream_to_ram(io.BytesIO(payload), len(payload), expected, lambda: None, 4)
        self.assertEqual(direct.read(), stdin.read())
        direct.close(); stdin.close()

    def test_exact_stream_is_ram_only_and_rewound(self):
        payload = b'official-toy-bytes'; checks = []
        result = asset.stream_to_ram(io.BytesIO(payload), len(payload), hashlib.sha256(payload).hexdigest(), lambda: checks.append(True), 3)
        self.assertIsInstance(result, io.BytesIO)
        self.assertEqual(result.tell(), 0); self.assertEqual(result.read(), payload)
        self.assertGreater(len(checks), 3); result.close()
        calls = [node.func for node in ast.walk(ast.parse(inspect.getsource(asset.stream_to_ram))) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(call, ast.Name) and call.id == 'open' for call in calls))
        self.assertFalse(any(isinstance(call, ast.Attribute) and call.attr in ('open', 'save', 'load', 'urlretrieve') for call in calls))

    def test_truncated_oversize_and_hashwrong_rejected(self):
        sha = hashlib.sha256(b'abcd').hexdigest()
        for payload, expected_sha in [(b'abc', sha), (b'abcde', sha), (b'abcd', '0' * 64)]:
            with self.assertRaises(RuntimeError): asset.stream_to_ram(io.BytesIO(payload), 4, expected_sha, lambda: None, 2)

    def test_deadline_is_checked_before_any_read(self):
        class NeverRead:
            def read(self, size): raise AssertionError('read happened after deadline')
        def expired(): raise TimeoutError('deadline')
        with self.assertRaises(TimeoutError): asset.stream_to_ram(NeverRead(), 1, 'a' * 64, expired)

    def test_network_failure_is_not_retried(self):
        class FailingRead:
            def __init__(self): self.count = 0
            def read(self, size): self.count += 1; raise OSError('toy disconnected')
        reader = FailingRead()
        with self.assertRaises(OSError): asset.stream_to_ram(reader, 4, 'a' * 64, lambda: None)
        self.assertEqual(reader.count, 1)

    def test_config_matches_large_and_rejects_other_encoder(self):
        cfg = {'model_type': 'wavlm', 'architectures': ['WavLMModel'], 'conv_dim': [512] * 7,
            'conv_kernel': asset.KERNELS, 'conv_stride': asset.STRIDES, 'conv_bias': False,
            'feat_extract_norm': 'layer', 'feat_extract_activation': 'gelu', 'num_feat_extract_layers': 7,
            'hidden_size': 1024, 'num_hidden_layers': 24}
        self.assertEqual(asset.validate_config(cfg), cfg)
        for key, wrong in [('feat_extract_norm', 'group'), ('conv_bias', True), ('hidden_size', 768), ('conv_kernel', [3] * 7)]:
            altered = dict(cfg); altered[key] = wrong
            with self.assertRaises(RuntimeError): asset.validate_config(altered)

    def test_manifest_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            asset.write_json_new(path, {'status': 'toy'})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): asset.write_json_new(path, {'status': 'changed'})
            self.assertEqual(path.read_bytes(), before)

    def test_memory_parser_and_no_model_forward_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'meminfo'
            path.write_text('MemTotal: 1000 kB\nMemAvailable: 42 kB\n')
            self.assertEqual(asset.memory_available(path), 42 * 1024)
        names = []
        for node in ast.walk(ast.parse(inspect.getsource(asset))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute): names.append(node.func.attr)
        self.assertFalse(set(names) & {'forward', 'backward', 'step', 'from_pretrained', 'urlretrieve'})


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not sys.platform.startswith('linux') or Path.cwd().resolve() != asset.ROOT or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('Only execute these tiny tensor tests on the rental, explicitly hiding CUDA')
        import torch
        cls.torch = torch
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.source = {key: torch.zeros(shape, dtype=torch.float32) for key, shape in asset.expected_shapes().items()}
        cls.source['encoder.layers.0.toy_unused.weight'] = torch.ones(1)

    @classmethod
    def tearDownClass(cls):
        cls.source = None; cls.torch.set_num_threads(cls.old_threads)

    def test_exact_21_tensor_shapes_and_independent_storage(self):
        selected, records = asset.select_conv(self.torch, self.source)
        self.assertEqual(len(selected), 21)
        self.assertEqual(sum(v.numel() for v in selected.values()), 4206592)
        self.assertEqual(len(records), 21)
        for item in records:
            self.assertTrue(item['saved_key'].startswith('conv_layers.'))
            self.assertNotEqual(selected[item['saved_key']].data_ptr(), self.source[item['original_key']].data_ptr())
            self.assertEqual(item['sha256'], hashlib.sha256(selected[item['saved_key']].numpy().tobytes()).hexdigest())
        self.assertFalse(any('encoder.layers' in key for key in selected))

    def test_missing_extra_and_wrongshape_rejected(self):
        key = next(iter(asset.expected_shapes()))
        missing = dict(self.source); missing.pop(key)
        with self.assertRaises(RuntimeError): asset.select_conv(self.torch, missing)
        extra = dict(self.source); extra[asset.PREFIX + '7.conv.weight'] = self.torch.zeros(1)
        with self.assertRaises(RuntimeError): asset.select_conv(self.torch, extra)
        wrong = dict(self.source); wrong[key] = self.torch.zeros(1)
        with self.assertRaises(RuntimeError): asset.select_conv(self.torch, wrong)

    def test_dtype_and_nonfinite_rejected(self):
        key = asset.PREFIX + '0.layer_norm.bias'
        for value in (self.torch.zeros(512, dtype=self.torch.float64), self.torch.full((512,), float('nan')), self.torch.full((512,), float('inf'))):
            wrong = dict(self.source); wrong[key] = value
            with self.assertRaises(RuntimeError): asset.select_conv(self.torch, wrong)

    def test_pure_small_state_save_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'conv.pt'; state = {'conv_layers.0.toy': self.torch.ones(2)}
            saved = asset.save_conv_new(self.torch, path, state)
            self.assertEqual(saved['sha256'], asset.sha_file(path))
            restored = self.torch.load(path, map_location='cpu')
            self.assertEqual(set(restored), set(state))
            self.assertTrue(self.torch.equal(restored['conv_layers.0.toy'], state['conv_layers.0.toy']))
            before = path.read_bytes()
            with self.assertRaises(RuntimeError): asset.save_conv_new(self.torch, path, state)
            self.assertEqual(path.read_bytes(), before)

    def test_oversize_partial_asset_preserved_without_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'small.pt'
            with self.assertRaises(RuntimeError): asset.save_conv_new(self.torch, path, {'toy': self.torch.ones(1)}, max_bytes=1)
            self.assertTrue(path.exists())


if __name__ == '__main__': unittest.main()
