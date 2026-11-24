# 上游数据实现定义声码器训练的波形裁切、峰值归一化与对数 mel 尺度。
"""HiFi-GAN 官方数据集与 mel 计算（教学注释版）。

这个文件包含两类功能：

1. 把 WAV 变成与声码器一致的 log-mel；
2. ``MelDataset`` 在训练时读取文件列表、随机裁剪固定长度并返回成对数据。

最重要的形状是：波形通常为 ``[batch, time]``，mel 为
``[batch, 80, frames]``。``hop_size=256`` 表示相邻 mel 帧相隔 256 个波形点。
字典 ``mel_basis``/``hann_window`` 是缓存，同一组参数只构造一次滤波器和窗函数。

本项目自训练声码器的数据入口另见 ``vocoder_paper_dataset.py``；那里基于 manifest
做了更严格的说话人/文件检查，但复用了这里相同的声学定义。
"""

# math 做长度计算，os 管路径，random 选择随机裁剪起点。
import math
# 拼接语音清单和预计算mel文件的路径。
import os
# 固定抽样种子，并为长语音选择随机训练片段。
import random
# 完成波形、频谱张量操作和可微STFT。
import torch
# 提供可交给DataLoader逐条读取的Dataset基类。
import torch.utils.data
# 执行NumPy版对数压缩，读取预计算mel数组。
import numpy as np
# 引入整句波形峰值归一化，用于上游从零训练数据路径。
from librosa.util import normalize
# 读取WAV采样率与整数PCM采样序列。
from scipy.io.wavfile import read
# 建立线性频率到mel频带的滤波矩阵。
from librosa.filters import mel as librosa_mel_fn

# 16位PCM以32768作为缩放分母，得到约[-1,1]浮点波形。
MAX_WAV_VALUE = 32768.0


# 读取指定WAV并返回波形及其采样率。
def load_wav(full_path):
    # 返回 WAV 文件中的采样数组与真实采样率，幅度缩放由后续数据步骤完成。
    """读取 WAV，返回“整数采样数组、采样率”二元组。"""

    # 从文件头取得采样率，同时读入采样数组。
    sampling_rate, data = read(full_path)
    # 按数据集调用约定先返回波形，再返回采样率。
    return data, sampling_rate


# 把线性幅度限制为正数后取自然对数，减小数值跨度。
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    # NumPy 输入先下限裁剪再取自然对数，避免 log(0)。
    """NumPy 版对数压缩；clip 防止 log(0) 产生负无穷。"""

    # 最低幅度截到clip_val，乘尺度C后作NumPy自然对数压缩。
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


# 用指数恢复对数压缩前的线性幅度。
def dynamic_range_decompression(x, C=1):
    # 对数值取指数，再除去压缩前的比例因子。
    return np.exp(x) / C


# 对张量执行可微自然对数压缩，保留回到波形的梯度。
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    # 用可求导的 Torch 自然对数压缩幅度，保留生成器训练梯度。
    """与上面数学相同的 PyTorch 版，可参与 GPU 计算。"""

    # 先避免log(0)，再计算张量的自然对数。
    return torch.log(torch.clamp(x, min=clip_val) * C)


# 对张量执行对数压缩的逆运算。
def dynamic_range_decompression_torch(x, C=1):
    # 把对数幅度还原为线性幅度，保留张量设备和梯度关系。
    return torch.exp(x) / C


# 这里的频谱normalize指自然对数压缩，与训练集z-score不同。
def spectral_normalize_torch(magnitudes):
    # 将线性频谱幅度转换成网络使用的对数尺度。
    output = dynamic_range_compression_torch(magnitudes)
    # 把完成压缩或逆压缩的频谱张量交回调用方。
    return output


# 把对数频谱还原为线性幅度。
def spectral_de_normalize_torch(magnitudes):
    # 用指数逆变换恢复未压缩的频谱幅度。
    output = dynamic_range_decompression_torch(magnitudes)
    # 把完成压缩或逆压缩的频谱张量交回调用方。
    return output


# 缓存mel滤波矩阵，减少反复创建相同频带映射的开销。
mel_basis = {}
# 缓存各设备上的Hann分析窗，供STFT复用。
hann_window = {}


# 用配套FFT、帧移、窗口和频率范围将[B,T]波形转换为log-mel。
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # 按 HiFi-GAN 手动补边和 STFT 标度生成自然对数 mel，条件与损失需使用一致定义。
    """波形 -> STFT 幅度 -> mel 滤波 -> log 压缩。"""

    # 输入波形低于标准浮点范围时打印提示，沿用上游非阻断检查。
    if torch.min(y) < -1.:
        # 显示过低的采样值，供排查PCM缩放或输入损坏。
        print('min value is ', torch.min(y))
    # 输入采样值超过1时提示幅度可能未正确归一化。
    if torch.max(y) > 1.:
        # 显示超过正常浮点范围的最大波形幅度。
        print('max value is ', torch.max(y))

    # 复用模块级频带矩阵和分析窗缓存。
    global mel_basis, hann_window
    # 保留上游缓存检查方式；下面的实际缓存键还包含设备字符串。
    if fmax not in mel_basis:
        # 按采样率、FFT和频率上下限建立mel滤波矩阵。
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        # 将mel矩阵转float32并放到当前波形设备，按频率上限和设备缓存。
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        # 建立分析窗并传到波形设备，供当前STFT使用。
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # 左右反射补(n_fft-hop)/2个采样点，配合center=False对齐mel帧。
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    # 去除临时增加的声道轴，恢复STFT要求的[B,T]形状。
    y = y.squeeze(1)

    # 按配置计算短时傅里叶变换；这是保留的上游旧PyTorch接口。
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      # 使用给定中心设置、未标准化FFT和单边频谱，保留上游行为。
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    # 将实部虚部平方相加再开方，得到带微小稳定项的线性幅度。
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    # 用mel滤波矩阵汇总线性频率，产生num_mels个频带。
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    # 对mel幅度取自然对数，得到声码器输入尺度。
    spec = spectral_normalize_torch(spec)

    # 返回[B,mel频带,时间帧]的log-mel张量。
    return spec


# 读取上游训练与验证文件清单，分别拼成WAV路径列表。
def get_dataset_filelist(a):
    # 打开训练文件列表；每行竖线前部分是语音主名。
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        # 把训练条目的文件主名与输入WAV目录拼接。
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          # 逐行读取并排除空行，保留每个有效语音条目。
                          for x in fi.read().split('\n') if len(x) > 0]

    # 读取上游验证清单；项目自训入口另用严格manifest协议。
    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        # 将验证语音主名转换成实际WAV路径。
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            # 逐行读取并排除空行，保留每个有效语音条目。
                            for x in fi.read().split('\n') if len(x) > 0]
    # 分别返回训练和验证列表，避免调用方混用两种数据。
    return training_files, validation_files


# 上游数据集按需读波形、裁剪并产生mel与配对音频。
class MelDataset(torch.utils.data.Dataset):
    # 接收语音列表、波形片段长度、FFT点数及mel频带数。
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 # 配置帧移、分析窗、采样率、频率范围，以及裁剪、打乱和缓存行为。
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 # 设置设备、损失频率上限、微调开关及预计算mel位置。
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        # 保存待读取语音路径列表。
        self.audio_files = training_files
        # 固定上游Python随机起点，影响打乱和裁剪。
        random.seed(1234)
        # 需要打乱时重排语音文件列表。
        if shuffle:
            # 原地随机重排文件，使训练不总按说话人或文件名排序。
            random.shuffle(self.audio_files)
        # 每个训练样本保留的波形采样点数。
        self.segment_size = segment_size
        # 输入WAV必须具有的采样率。
        self.sampling_rate = sampling_rate
        # 是否将整句裁为固定训练片段；验证可保留整句。
        self.split = split
        # 记录STFT的傅里叶变换点数。
        self.n_fft = n_fft
        # 记录log-mel频带数，V1通常为80。
        self.num_mels = num_mels
        # 记录相邻mel帧相隔的波形采样点数。
        self.hop_size = hop_size
        # 记录每一STFT分析窗覆盖的采样点数。
        self.win_size = win_size
        # 记录mel滤波器最低频率。
        self.fmin = fmin
        # 记录用于条件输入mel的最高频率。
        self.fmax = fmax
        # 记录重建损失mel的频率上限，可与条件输入不同。
        self.fmax_loss = fmax_loss
        # 尚未缓存波形，第一次取样需要读盘。
        self.cached_wav = None
        # 规定读取一个波形后可额外复用多少次。
        self.n_cache_reuse = n_cache_reuse
        # 缓存剩余次数初始为零，触发首次读取。
        self._cache_ref_count = 0
        # 保存调用方传入的目标设备信息。
        self.device = device
        # 决定使用现场提取mel或预计算mel的训练路径。
        self.fine_tuning = fine_tuning
        # 保存微调时预计算mel所在目录。
        self.base_mels_path = base_mels_path

    # 读取一个语音条目，返回对齐的条件mel、波形及损失mel。
    def __getitem__(self, index):
        # 按数据加载器给定索引选中当前语音路径。
        filename = self.audio_files[index]
        # 缓存用完时重新从磁盘读取当前波形。
        if self._cache_ref_count == 0:
            # 取得当前WAV的采样序列及实际采样率。
            audio, sampling_rate = load_wav(filename)
            # 把整数PCM缩放到标准浮点波形范围。
            audio = audio / MAX_WAV_VALUE
            # 从零训练路径使用整句峰值归一化并现场提取mel。
            if not self.fine_tuning:
                # 把整句波形最大绝对幅度调整为0.95，再裁片段。
                audio = normalize(audio) * 0.95
            # 缓存读取及归一化后的整句波形，减少重复I/O。
            self.cached_wav = audio
            # 采样率与配置不同则拒绝该数据，避免时长和频谱错配。
            if sampling_rate != self.sampling_rate:
                # 报告实际与目标采样率不一致，阻止继续训练。
                raise ValueError("{} SR doesn't match target {} SR".format(
                    # 错误信息同时保留输入采样率和配置采样率。
                    sampling_rate, self.sampling_rate))
            # 读取完成后重置允许复用缓存的次数。
            self._cache_ref_count = self.n_cache_reuse
        # 此分支：缓存仍有效时直接使用已读入的波形。
        else:
            # 缓存仍有效时直接使用已读入的波形。
            audio = self.cached_wav
            # 每使用一次缓存就减少剩余复用次数。
            self._cache_ref_count -= 1

        # 将波形转换成float32张量，供STFT和训练使用。
        audio = torch.FloatTensor(audio)
        # 加入batch轴，让波形成为[1,T]。
        audio = audio.unsqueeze(0)

        # 从零训练路径使用整句峰值归一化并现场提取mel。
        if not self.fine_tuning:
            # 需要固定训练长度时，裁剪长句或补齐短句。
            if self.split:
                # 波形足够长时从合法区间抽取训练片段。
                if audio.size(1) >= self.segment_size:
                    # 计算仍能容纳完整片段的最大采样点起点。
                    max_audio_start = audio.size(1) - self.segment_size
                    # 在全部合法波形起点间均匀随机选择一个。
                    audio_start = random.randint(0, max_audio_start)
                    # 取连续segment_size个波形采样点作为当前训练片段。
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                # 此分支：短波形在右侧补零，达到统一训练长度。
                else:
                    # 短波形在右侧补零，达到统一训练长度。
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            # 从刚裁剪的同一波形提取输入mel，保持样本对应。
            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  # 使用条件mel的采样率、帧移、窗口和频率范围。
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  # 使用已手动补边的STFT定义，避免重复居中补边。
                                  center=False)
        # 此分支：微调路径读取预先生成的mel，保留上游实现。
        else:
            # 微调路径读取预先生成的mel，保留上游实现。
            mel = np.load(
                # 用当前WAV文件主名寻找同名.npy条件mel。
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            # 将预计算mel数组转换成PyTorch张量。
            mel = torch.from_numpy(mel)

            # 没有batch轴的mel需要先补成三维。
            if len(mel.shape) < 3:
                # 给[mel频带,帧数]补batch轴，成为[1,mel频带,帧数]。
                mel = mel.unsqueeze(0)

            # 需要固定训练长度时，裁剪长句或补齐短句。
            if self.split:
                # 把波形片段长度换算成所需mel帧数，向上取整。
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                # 波形足够长时从合法区间抽取训练片段。
                if audio.size(1) >= self.segment_size:
                    # 按上游范围随机选择预计算mel的片段起点。
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    # 截取微调所需的固定长度mel片段。
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    # 用相同mel起点乘hop裁波形，维持两侧时间对应。
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                # 此分支：预计算mel过短时在时间轴右侧补零。
                else:
                    # 预计算mel过短时在时间轴右侧补零。
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    # 短波形在右侧补零，达到统一训练长度。
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # 为真实波形另算损失mel，供生成音频的重建误差比较。
        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   # 重建mel使用fmax_loss，允许覆盖不同于条件输入的最高频率。
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   # 使用已手动补边的STFT定义，避免重复居中补边。
                                   center=False)

        # 返回条件mel、配对波形、文件名和真实损失mel，由DataLoader堆叠。
        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    # 让DataLoader知道语音文件列表的长度。
    def __len__(self):
        # 返回可按索引读取的语音条目总数。
        return len(self.audio_files)
