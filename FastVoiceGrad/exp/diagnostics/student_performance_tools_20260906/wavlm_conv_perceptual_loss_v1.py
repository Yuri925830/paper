"""FINALLY-inspired辅助损失草稿：仅WavLM七层卷积前端和固定STFT，无评分头或训练入口。"""
from collections.abc import Mapping
from pathlib import Path
import sys

if not sys.platform.startswith('linux') or not Path('/root/FsatVoiceGrad/AGENTS.md').is_file():
    raise RuntimeError('Project computations are permitted only on the registered rental')
import torch
from torch import nn
import torch.nn.functional as F

KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)
CHANNELS = 512
RECEPTIVE_FIELD = 400
TOTAL_STRIDE = 320
LAYER_NORM_EPS = 1e-5
WAVE_NORM_EPS = 1e-7
SOURCE_RATE, FEATURE_RATE = 22050, 16000
N_FFT, WIN_LENGTH, HOP_LENGTH = 512, 512, 128
FEATURE_WEIGHT = 100.0
RESAMPLE_KWARGS = dict(orig_freq=SOURCE_RATE, new_freq=FEATURE_RATE,
    lowpass_filter_width=6, rolloff=0.99, resampling_method='sinc_interpolation', beta=None)


def expected_state_shapes():
    """资产接口：原HF key仅去掉feature_extractor.前缀，恰好21个float32张量。"""
    result = {}
    for index, kernel in enumerate(KERNELS):
        prefix = 'conv_layers.%d.' % index
        result[prefix + 'conv.weight'] = (512, 1 if index == 0 else 512, kernel)
        result[prefix + 'layer_norm.weight'] = (512,)
        result[prefix + 'layer_norm.bias'] = (512,)
    return result


def validate_conv_state(state):
    shapes = expected_state_shapes()
    if not isinstance(state, Mapping) or set(state) != set(shapes):
        raise ValueError('Require exactly21 convolution/LayerNorm tensor keys, with no full-model or projection state')
    for name, shape in shapes.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32
                or tuple(value.shape) != shape or value.layout != torch.strided or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid frozen convolution tensor: ' + name)


def feature_frames(length):
    if type(length) is not int or length < RECEPTIVE_FIELD:
        raise ValueError('WavLM convolution input must contain at least400 valid16k samples')
    for kernel, stride in zip(KERNELS, STRIDES): length = (length - kernel) // stride + 1
    return length


class _LayerNormConv(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.conv = nn.Conv1d(1 if index == 0 else 512, 512, KERNELS[index], stride=STRIDES[index],
            bias=False, device='cpu', dtype=torch.float32)
        self.layer_norm = nn.LayerNorm(512, eps=LAYER_NORM_EPS, elementwise_affine=True,
            device='cpu', dtype=torch.float32)

    def forward(self, values):
        values = self.conv(values)
        values = self.layer_norm(values.transpose(1, 2)).transpose(1, 2)
        return F.gelu(values, approximate='none')


class FrozenWavLMConvEncoder(nn.Module):
    """复现HF WavLM-large FeatureEncoder；只返回第七层[batch,512,frames]。"""
    def __init__(self, state):
        super().__init__()
        validate_conv_state(state)
        # 所有模块先在CPU创建，fork_rng只保护CPU随机流，不初始化CUDA或改变全局seed。
        with torch.random.fork_rng(devices=[], enabled=True):
            self.conv_layers = nn.ModuleList([_LayerNormConv(i) for i in range(7)])
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def assert_frozen(self):
        if any(m.training for m in self.modules()) or any(p.requires_grad or p.grad is not None for p in self.parameters()):
            raise RuntimeError('WavLM convolution front end must stay fully frozen/eval without parameter gradients')

    def forward(self, values):
        self.assert_frozen()
        first = self.conv_layers[0].conv.weight
        if (values.ndim != 2 or values.shape[0] < 1 or values.dtype != torch.float32
                or values.device != first.device or not bool(torch.isfinite(values).all())):
            raise ValueError('Require a finite float32[batch,valid16k samples] on the encoder device')
        expected = feature_frames(values.shape[-1])
        hidden = values[:, None, :]
        for layer in self.conv_layers: hidden = layer(hidden)
        if hidden.shape != (values.shape[0], 512, expected) or not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError('Invalid frozen WavLM convolution features')
        return hidden  # 不使用no_grad：学生波形的梯度必须穿过冻结卷积返回G和学生模型。


def normalize_for_features(wave):
    """逐句减均值并除sqrt(总体方差+1e-7)，只用于卷积特征支路。"""
    centered = wave - wave.mean(dim=-1, keepdim=True)
    population_variance = centered.square().mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(population_variance + WAVE_NORM_EPS)


def resample_valid_wave(wave):
    from torchaudio.functional import resample
    return resample(wave, **RESAMPLE_KWARGS)


def stft_magnitude(wave):
    if wave.shape[-1] < N_FFT:
        raise ValueError('STFT requires at least512 valid16k samples; no padding fallback')
    window = torch.hann_window(WIN_LENGTH, periodic=True, dtype=wave.dtype, device=wave.device)
    return torch.stft(wave, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, center=False, normalized=False, onesided=True, return_complex=True).abs()


def _valid_batch_components(encoder, generated, target):
    # 此处输入已经按真实长度裁剪；任何补齐区域都不会参与重采样边界、均值或方差。
    if not bool(torch.isfinite(generated).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError('Nonfinite values inside valid waveform samples')
    generated16 = resample_valid_wave(generated)
    with torch.no_grad(): target16 = resample_valid_wave(target.detach())
    expected = (generated.shape[-1] * FEATURE_RATE + SOURCE_RATE - 1) // SOURCE_RATE
    if generated16.shape != target16.shape or generated16.shape[-1] != expected or expected < max(RECEPTIVE_FIELD, N_FFT):
        raise ValueError('Invalid or too-short matched resampled waveform lengths')
    generated_features = encoder(normalize_for_features(generated16))
    with torch.no_grad(): target_features = encoder(normalize_for_features(target16))
    feature_mse = (generated_features - target_features).square().flatten(1).mean(1)
    # STFT使用未作零均值/单位方差处理的重采样波形，保留幅度差异。
    generated_mag = stft_magnitude(generated16)
    with torch.no_grad(): target_mag = stft_magnitude(target16)
    stft_l1 = (generated_mag - target_mag).abs().flatten(1).mean(1)
    return feature_mse, stft_l1, expected


def _wave_matrix(wave):
    if not isinstance(wave, torch.Tensor): raise TypeError('Waveforms must be Torch tensors')
    if wave.ndim == 3 and wave.shape[1] == 1: wave = wave[:, 0, :]
    if wave.ndim != 2 or wave.shape[0] < 1 or wave.dtype != torch.float32:
        raise ValueError('Require float32[B,T] or float32[B,1,T] waveforms')
    return wave


def wavlm_conv_perceptual_loss(encoder, generated_waveform, target_waveform, valid_lengths):
    """返回可微总loss及逐句分项；valid_lengths以22050Hz波形采样点为单位。"""
    if not isinstance(encoder, FrozenWavLMConvEncoder):
        raise TypeError('Require the frozen21-tensor convolution-only encoder')
    encoder.assert_frozen()
    generated = _wave_matrix(generated_waveform)
    target = _wave_matrix(target_waveform).detach()
    if generated.shape != target.shape or generated.device != target.device:
        raise ValueError('Generated and target waveforms must be matched in batch, device and padded length')
    if isinstance(valid_lengths, torch.Tensor):
        if valid_lengths.dtype != torch.long or valid_lengths.shape != (generated.shape[0],):
            raise ValueError('Require int64[B] original-rate valid lengths')
        lengths = valid_lengths.detach().cpu().tolist()
    else:
        lengths = list(valid_lengths)
    if (len(lengths) != generated.shape[0] or any(type(x) is not int or x < 1 or x > generated.shape[-1] for x in lengths)
            or any((x * FEATURE_RATE + SOURCE_RATE - 1) // SOURCE_RATE < max(RECEPTIVE_FIELD, N_FFT) for x in lengths)):
        raise ValueError('Every valid waveform must fit its tensor and yield at least512 real16k samples')
    features, spectra, resampled_lengths = [], [], []
    if len(set(lengths)) == 1:
        # 相同真实长度时一次批量卷积；即使存在共同padding，也先统一裁掉它。
        feature, spectrum, size = _valid_batch_components(encoder, generated[:, :lengths[0]], target[:, :lengths[0]])
        features.append(feature); spectra.append(spectrum); resampled_lengths = [size] * len(lengths)
        execution_path = 'same_valid_length_batch'
    else:
        # 变长语音逐句计算，不能将零padding当作有效样本纳入卷积或单位方差统计。
        for index, length in enumerate(lengths):
            feature, spectrum, size = _valid_batch_components(encoder,
                generated[index:index + 1, :length], target[index:index + 1, :length])
            features.append(feature); spectra.append(spectrum); resampled_lengths.append(size)
        execution_path = 'per_example_variable_length'
    feature_mse = torch.cat(features); stft_l1 = torch.cat(spectra)
    per_example = FEATURE_WEIGHT * feature_mse + stft_l1
    if not bool(torch.isfinite(per_example).all()):
        raise FloatingPointError('Nonfinite WavLM-convolution/STFT auxiliary loss')
    stft_mean = stft_l1.mean()
    return dict(loss=per_example.mean(), feature_mse=feature_mse.mean(),
        feature_weighted=(FEATURE_WEIGHT * feature_mse).mean(), stft_magnitude_l1=stft_mean, stft_l1=stft_mean,
        per_example_loss=per_example, per_utterance_loss=per_example, per_example_feature_mse=feature_mse,
        per_example_stft_magnitude_l1=stft_l1, resampled_lengths=resampled_lengths,
        feature_frame_lengths=[feature_frames(x) for x in resampled_lengths], execution_path=execution_path,
        method='FINALLY_inspired_conv_feature_plus_STFT_engineering_v1')


class WavLMConvPerceptualLoss(nn.Module):
    """训练附件用的薄封装；内部无权重选择、优化器、文件加载或其他训练副作用。"""
    def __init__(self, encoder):
        super().__init__()
        if not isinstance(encoder, FrozenWavLMConvEncoder):
            raise TypeError('Build FrozenWavLMConvEncoder from the verified21-tensor state first')
        encoder.assert_frozen()
        self.encoder = encoder
        self.eval()

    def forward(self, generated_waveform, target_waveform, valid_lengths):
        if self.training: raise RuntimeError('Perceptual loss must remain in eval mode')
        return wavlm_conv_perceptual_loss(self.encoder, generated_waveform, target_waveform, valid_lengths)
