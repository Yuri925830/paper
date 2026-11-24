"""在租用服务器内存中校验微软WavLM完整权重，仅保存已验证卷积参数。"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path('/root/FsatVoiceGrad')
TOOLS = ROOT / 'exp/diagnostics/student_performance_tools_20260906'
DEV = ROOT / 'exp/diagnostics/student_development_20260906'
ASSETS = TOOLS / 'assets'
STEM = 'wavlm_large_conv_c1423ed_v1'
OUTPUT = ASSETS / (STEM + '.pt')
RECEIPT = ASSETS / (STEM + '.receipt.json')
STARTED = ASSETS / (STEM + '.started.json')
FAILURE = ASSETS / (STEM + '.failure.json')
SOURCE_CONFIG = ASSETS / (STEM + '.source-config.json')
PROTOCOL = DEV / 'WAVLM_LARGE_CONV_OFFICIAL_ACQUISITION_CN.md'
TESTS = TOOLS / 'test_acquire_wavlm_large_conv_official_v1.py'
REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
MODEL_SIZE = 1261990257
MODEL_SHA = 'fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d'
CONFIG_SHA = 'a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d'
CONFIG_SIZE = 2222
MODEL_URL = 'https://huggingface.co/microsoft/wavlm-large/resolve/' + REVISION + '/pytorch_model.bin'
CONFIG_URL = 'https://huggingface.co/microsoft/wavlm-large/resolve/' + REVISION + '/config.json'
MAX_OUTPUT = 24 * 1024**2
MIN_MEMORY = 8 * 1024**3
DISK_MARGIN = 128 * 1024**2
TOTAL_SECONDS, NETWORK_TIMEOUT = 1200, 30
PREFIX = 'feature_extractor.conv_layers.'
KERNELS = [10, 3, 3, 3, 3, 2, 2]
STRIDES = [5, 2, 2, 2, 2, 2, 2]
PARAMETERS = 4206592


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''): result.update(block)
    return result.hexdigest()


def valid_sha(value):
    return len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def memory_available(path=Path('/proc/meminfo')):
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is unavailable')


def require_https(url):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ''
    if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not (host == 'huggingface.co' or host.endswith('.huggingface.co') or host.endswith('.hf.co'))):
        raise RuntimeError('Refuse non-official or non-HTTPS model redirect')
    return host


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Deadline:
    def __init__(self): self.started, self.reason, self.handlers = time.monotonic(), None, {}
    def install(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            self.handlers[sig] = signal.getsignal(sig); signal.signal(sig, self.stop)
        signal.alarm(TOTAL_SECONDS)
    def stop(self, signum, frame): self.reason = 'signal_%d' % signum
    def check(self):
        if self.reason or time.monotonic() - self.started >= TOTAL_SECONDS: raise TimeoutError(self.reason or 'acquisition deadline')
    def close(self):
        signal.alarm(0)
        for sig, previous in self.handlers.items(): signal.signal(sig, previous)


def stream_to_ram(response, expected_size, expected_sha, check, block_size=1024**2):
    # 只创建内存BytesIO；没有临时权重文件，也不调用网络库的文件缓存功能。
    buffer, digest, count = io.BytesIO(), hashlib.sha256(), 0
    try:
        while True:
            check(); block = response.read(min(block_size, expected_size + 1 - count))
            if not block: break
            count += len(block)
            if count > expected_size: raise RuntimeError('Download exceeds the pinned file size')
            digest.update(block); buffer.write(block)
        check()
        if count != expected_size: raise RuntimeError('Truncated official model download')
        if digest.hexdigest() != expected_sha: raise RuntimeError('Official full-file SHA256 mismatch')
        buffer.seek(0)
        return buffer
    except BaseException:
        buffer.close(); raise


def open_official(opener, url, check):
    require_https(url); check()
    response = opener.open(urllib.request.Request(url, headers={'User-Agent': 'VoiceGrad-WavLM-provenance-v1'}), timeout=NETWORK_TIMEOUT)
    try:
        if response.status != 200: raise RuntimeError('Require one complete HTTP200 response')
        host = require_https(response.geturl())
        return response, host
    except BaseException:
        response.close(); raise


def validate_config(config):
    expected = {'model_type': 'wavlm', 'architectures': ['WavLMModel'], 'conv_dim': [512] * 7,
        'conv_kernel': KERNELS, 'conv_stride': STRIDES, 'conv_bias': False, 'feat_extract_norm': 'layer',
        'feat_extract_activation': 'gelu', 'num_feat_extract_layers': 7, 'hidden_size': 1024, 'num_hidden_layers': 24}
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Official configuration does not match the declared WavLM-large convolutional encoder')
    return expected


def parse_verified_config(raw):
    if len(raw) != CONFIG_SIZE or hashlib.sha256(raw).hexdigest() != CONFIG_SHA:
        raise RuntimeError('Official fixed-revision configuration bytes differ')
    config = json.loads(raw)
    return config, validate_config(config)


def expected_shapes():
    shapes = {}
    for layer, kernel in enumerate(KERNELS):
        base = PREFIX + str(layer) + '.'
        shapes[base + 'conv.weight'] = (512, 1 if layer == 0 else 512, kernel)
        shapes[base + 'layer_norm.weight'] = (512,)
        shapes[base + 'layer_norm.bias'] = (512,)
    return shapes


def select_conv(torch, original):
    # 键白名单包含7个卷积及各层LayerNorm。不要提取Transformer、投影层或指标网络。
    if not isinstance(original, dict): raise RuntimeError('Expected the official plain state dictionary')
    expected = expected_shapes(); found = {key for key in original if key.startswith(PREFIX)}
    if found != set(expected): raise RuntimeError('Missing or extra convolutional keys')
    selected, tensors = {}, []
    for key, shape in expected.items():
        value = original[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != 'cpu' or not bool(torch.isfinite(value).all())):
            raise RuntimeError('Invalid convolutional tensor: ' + key)
        # clone使小文件与完整源对象没有共享storage，避免不小心把完整源storage一起保存。
        clone = value.detach().contiguous().clone()
        selected[key[len('feature_extractor.'):]] = clone
        tensors.append({'original_key': key, 'saved_key': key[len('feature_extractor.'):], 'shape': list(shape),
            'numel': value.numel(), 'dtype': 'float32', 'sha256': hashlib.sha256(clone.numpy().tobytes()).hexdigest()})
    if len(selected) != 21 or sum(v.numel() for v in selected.values()) != PARAMETERS:
        raise RuntimeError('Unexpected convolutional parameter count')
    return selected, tensors


def write_json_new(path, value):
    # 所有文件均新建。中断留下的部分文件也保留，下次拒绝覆盖；只认完整可解析的最终receipt。
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())


def save_conv_new(torch, path, state, max_bytes=MAX_OUTPUT):
    path = Path(path)
    if path.exists() or path.is_symlink(): raise RuntimeError('Never overwrite an asset or partial result')
    with path.open('xb') as handle:
        torch.save(state, handle); handle.flush(); os.fsync(handle.fileno())
    size = path.stat().st_size
    if size > max_bytes: raise RuntimeError('Extracted convolutional asset exceeds budget')
    return {'path': str(path), 'size_bytes': size, 'sha256': sha_file(path), 'state_format': 'plain_conv_only_state_dict'}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-approved', action='store_true')
    parser.add_argument('--transport', choices=('https', 'stdin'), required=True)
    parser.add_argument('--config-sha256')
    for name in ('driver', 'tests', 'protocol'): parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args(argv)
    if not args.run_approved: raise RuntimeError('Reviewed source and explicit bounded acquisition approval required')
    if not all(valid_sha(getattr(args, name + '_sha256')) for name in ('driver', 'tests', 'protocol')):
        raise RuntimeError('Require complete lowercase SHA256 identities')
    if args.transport == 'stdin' and args.config_sha256 != CONFIG_SHA:
        raise RuntimeError('Stdin transport requires the pinned uploaded official config SHA')
    if args.transport == 'https' and args.config_sha256 is not None:
        raise RuntimeError('HTTPS mode reads its fixed official config directly; no implicit transport fallback')
    return args


def main(argv=None):
    args = arguments(argv)
    if not sys.platform.startswith('linux') or Path.cwd().resolve() != ROOT or not (ROOT / 'AGENTS.md').is_file():
        raise RuntimeError('Run only in the registered rented project')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('Explicitly hide CUDA for this CPU-only acquisition')
    pins = {str(Path(__file__).resolve()): args.driver_sha256, str(TESTS): args.tests_sha256, str(PROTOCOL): args.protocol_sha256}
    if args.transport == 'stdin': pins[str(SOURCE_CONFIG)] = CONFIG_SHA
    for name, sha in pins.items():
        path = Path(name)
        if path.is_symlink() or sha_file(path) != sha: raise RuntimeError('Reviewed acquisition source changed')
    if ASSETS.is_symlink() or (ASSETS.exists() and ASSETS.resolve() != ASSETS): raise RuntimeError('Unsafe asset directory')
    ASSETS.mkdir(exist_ok=True)
    paths = (OUTPUT, RECEIPT, STARTED, FAILURE)
    if any(path.exists() or path.is_symlink() for path in paths): raise RuntimeError('No retry or overwrite of this acquisition identity')
    memory = memory_available()
    if memory < MIN_MEMORY: raise RuntimeError('Need at least8GiB available RAM before full-file in-memory validation')
    disk = os.statvfs(ASSETS); free = disk.f_bavail * disk.f_frsize
    if free < MAX_OUTPUT + DISK_MARGIN: raise RuntimeError('Insufficient space for24MiB asset budget plus128MiB operation margin')
    deadline, owned, buffer, original, selected = Deadline(), False, None, None, None
    result = error = None
    try:
        deadline.install()
        # started的独占创建也是本次下载的互斥准入；另一个同名进程会在此立即失败。
        write_json_new(STARTED, {'status': 'started', 'revision': REVISION, 'expected_source_sha256': MODEL_SHA,
            'expected_source_bytes': MODEL_SIZE, 'memory_available_bytes': memory, 'free_bytes': free,
            'transport': args.transport, 'source_pins': pins})
        owned = True
        if args.transport == 'https':
            opener = urllib.request.build_opener(OfficialRedirect())
            response, config_host = open_official(opener, CONFIG_URL, deadline.check)
            with response:
                raw_config = response.read(65537)
                if len(raw_config) > 65536: raise RuntimeError('Unreasonable official configuration length')
            deadline.check(); config, required_config = parse_verified_config(raw_config)
            response, model_host = open_official(opener, MODEL_URL, deadline.check)
            with response:
                advertised = response.headers.get('Content-Length')
                if advertised is not None and int(advertised) != MODEL_SIZE: raise RuntimeError('Official HTTP file size differs')
                buffer = stream_to_ram(response, MODEL_SIZE, MODEL_SHA, deadline.check)
        else:
            # Windows仅搬运官方原件；这里再次完整核对stdin的原始二进制，不相信远端文件名。
            raw_config = SOURCE_CONFIG.read_bytes()
            config, required_config = parse_verified_config(raw_config)
            config_host = model_host = None
            buffer = stream_to_ram(sys.stdin.buffer, MODEL_SIZE, MODEL_SHA, deadline.check)
        deadline.check()
        # pickle只在官方完整字节SHA正确后交给既有PyTorch；不执行模型构建或前向。
        import torch
        if torch.__version__ != '1.13.1+cu116' or torch.cuda.is_initialized(): raise RuntimeError('Require frozen CPU-only myconda Torch')
        original = torch.load(buffer, map_location='cpu')
        deadline.check(); selected, tensors = select_conv(torch, original)
        original = None; buffer.close(); buffer = None
        output = save_conv_new(torch, OUTPUT, selected, MAX_OUTPUT - 1024**2)
        selected = None; deadline.check()
        for name, sha in pins.items():
            if sha_file(name) != sha: raise RuntimeError('Source changed during acquisition')
        used = sum(path.stat().st_size for path in paths + (SOURCE_CONFIG,) if path.exists())
        if used > MAX_OUTPUT - 1024**2: raise RuntimeError('Reserve1MiB for acquisition receipt')
        if torch.cuda.is_initialized(): raise RuntimeError('Unexpected CUDA initialization')
        result = {'status': 'completed_verified_official_conv_extraction', 'output': output, 'source_model_url': MODEL_URL,
            'source_revision': REVISION, 'source_sha256': MODEL_SHA, 'source_size_bytes': MODEL_SIZE,
            'source_config_url': CONFIG_URL, 'source_config_sha256': hashlib.sha256(raw_config).hexdigest(),
            'official_config': config, 'validated_encoder_config': required_config,
            'redirect_hosts_only': {'config': config_host, 'model': model_host}, 'source_pins': pins,
            'transport': args.transport, 'rental_performed_direct_HTTPS_download': args.transport == 'https',
            'uploaded_config_path': str(SOURCE_CONFIG) if args.transport == 'stdin' else None,
            'tensors': tensors, 'tensor_count': 21, 'parameter_count': PARAMETERS,
            'full_model_written_to_disk': False, 'model_initializations': 0, 'forward_calls': 0,
            'optimizer_updates': 0, 'cuda_initialized': False, 'torch_version': torch.__version__,
            'elapsed_seconds': time.monotonic() - deadline.started, 'automatic_training': False}
    except BaseException as caught: error = caught
    finally:
        original = selected = None
        if buffer is not None: buffer.close()
        try: deadline.close()
        except BaseException as caught:
            if error is None: error = caught
        if owned:
            try:
                if error is not None: write_json_new(FAILURE, {'error': repr(error), 'no_retry': True, 'partial_asset_may_exist': OUTPUT.exists()})
                else:
                    deadline.check()
                    write_json_new(RECEIPT, result)
            except BaseException as caught:
                if error is None: error = caught
                if not FAILURE.exists():
                    try: write_json_new(FAILURE, {'error': repr(caught), 'no_retry': True, 'partial_asset_may_exist': OUTPUT.exists()})
                    except BaseException: print('Failure receipt could not be written; no valid completion may be assumed', file=sys.stderr)
    if error is not None: raise error


if __name__ == '__main__': main()
