# 此上游入口先从输入 WAV 提取 mel，再用 G 还原波形；项目生成 mel 的入口另在 tools/vocode.py。
"""HiFi-GAN 官方“WAV 输入”推理脚本（教学注释版）。

它的路线是 ``输入 WAV -> 重新提取 mel -> Generator -> 输出 WAV``。本项目已经有
VoiceGrad 生成的 mel 时，主要使用根目录 ``vocode_voicegrad_mels.py``；保留本文件
是为了完整保存上游 HiFi-GAN 工具和复核官方行为。

阅读抓手：

* ``h`` 是从 JSON 得到的 AttrDict 配置；
* ``device`` 是 cpu 或 cuda；
* ``torch.no_grad`` 表示推理不保存反向传播中间量；
* 波形乘 ``MAX_WAV_VALUE`` 后转 int16，才是常见 PCM WAV 数值范围。
"""

# future 导入让这份较旧代码在不同 Python 3 小版本保持一致行为。
from __future__ import absolute_import, division, print_function, unicode_literals

# 用通配符查找符合命名规则的已有检查点。
import glob
# 拼接输入输出路径、列出目录内容并建立结果目录。
import os
# 读取命令行传入的输入目录、输出目录及检查点位置。
import argparse
# 解析检查点旁的网络配置JSON。
import json
# 加载声码器、操作张量并控制推理所用设备。
import torch
# 将整数PCM采样数组写成WAV文件。
from scipy.io.wavfile import write
# 使JSON配置能够按h.n_fft等属性形式访问。
from env import AttrDict
# 复用上游mel提取、PCM缩放常数和WAV读取函数。
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
# 加载将80维mel上采样为波形的HiFi-GAN生成器结构。
from models import Generator

# 配置在main读取JSON后填入，供推理函数共享。
h = None
# 实际CPU或GPU设备在main初始化后确定。
device = None


# 读取来源可信的上游权重文件，并映射到指定设备。
def load_checkpoint(filepath, device):
    # 加载已保存状态，并检查其结构和当前运行约定。
    """从磁盘读取 PyTorch 字典，并把其中 Tensor 映射到指定设备。"""

    # 权重路径必须对应实际文件，缺失时停止加载。
    assert os.path.isfile(filepath)
    # 显示真正使用的检查点，便于核对实验身份。
    print("Loading '{}'".format(filepath))
    # 读取检查点字典；仅对已确认来源的权重使用该上游入口。
    checkpoint_dict = torch.load(filepath, map_location=device)
    # 提示检查点读取结束。
    print("Complete.")
    # 将权重字典交给调用方加载生成器。
    return checkpoint_dict


# 按照生成器配套参数从波形提取log-mel。
def get_mel(x):
    # 按照配置的 FFT、窗和帧移从波形提取声码器条件 mel。
    """按配置把一批波形转换为 log-mel。"""

    # 使用相同FFT、频带和帧移设置，防止输入与声码器训练时不一致。
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


# 查找目录中符合前缀的最新文件名，供检查点选择使用。
def scan_checkpoint(cp_dir, prefix):
    # 按固定前缀查找并排序归档文件，无匹配时返回空结果。
    """找指定前缀的最后一个检查点；没有时返回空字符串。"""

    # 构造“指定目录+前缀”的文件匹配规则。
    pattern = os.path.join(cp_dir, prefix + '*')
    # 读取满足检查点前缀规则的全部文件。
    cp_list = glob.glob(pattern)
    # 没有候选检查点时返回空结果。
    if len(cp_list) == 0:
        # 以空字符串表示未找到任何匹配检查点。
        return ''
    # 按文件名排序并取最后一个，沿用上游命名顺序。
    return sorted(cp_list)[-1]


# 加载生成器一次，再逐文件完成mel到波形的合成。
def inference(a):
    # 一次加载生成器后逐个处理输入，输出标准声码器波形文件。
    """建立生成器、遍历输入 WAV、生成并保存波形。"""

    # 按配套配置建立生成器，并把其参数放到当前设备。
    generator = Generator(h).to(device)

    # 读取用户指定的生成器检查点。
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    # 把检查点generator字段中的参数填入网络。
    generator.load_state_dict(state_dict_g['generator'])

    # 列出待还原WAV目录中的文件，保持上游原有遍历行为。
    filelist = os.listdir(a.input_wavs_dir)

    # 建立合成WAV输出目录，目录已存在时继续处理。
    os.makedirs(a.output_dir, exist_ok=True)

    # 切换为推理模式，关闭可能存在的训练态行为。
    generator.eval()
    # 合并权重归一化参数，减少每次合成的额外开销。
    generator.remove_weight_norm()
    # 推理不构建反向传播图，节省显存。
    with torch.no_grad():
        # 逐个处理输入文件，序号仅用于遍历。
        for i, filname in enumerate(filelist):
            # 读取当前WAV的整数采样值及文件采样率。
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            # 将16位PCM采样值缩放到约[-1,1]的浮点范围。
            wav = wav / MAX_WAV_VALUE
            # 把当前波形转换为float32张量并传到生成器所在设备。
            wav = torch.FloatTensor(wav).to(device)
            # 补上batch轴后提取当前波形的log-mel。
            x = get_mel(wav.unsqueeze(0))
            # 把mel一次上采样成合成波形。
            y_g_hat = generator(x)
            # 去除长度为1的batch和声道轴，得到一维采样序列。
            audio = y_g_hat.squeeze()
            # 把浮点采样值放大到16位PCM数值范围。
            audio = audio * MAX_WAV_VALUE
            # 将结果搬回CPU，并转换为写WAV所需的16位整数。
            audio = audio.cpu().numpy().astype('int16')

            # 保留原文件主名，增加_generated后缀区分合成结果。
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            # 按配置采样率写出合成WAV，沿用上游直接写盘行为。
            write(output_file, h.sampling_rate, audio)
            # 输出已写WAV的路径，方便在终端逐条核对。
            print(output_file)


# 读取路径及网络配置、选择设备，然后进入批量合成。
def main():
    # 提示开始初始化声码器推理流程。
    print('Initializing Inference Process..')

    # 建立命令行解析器，自动提供参数帮助。
    parser = argparse.ArgumentParser()
    # 指定待重新提取mel并还原的WAV目录。
    parser.add_argument('--input_wavs_dir', default='test_files')
    # 指定WAV还原结果写入的目录。
    parser.add_argument('--output_dir', default='generated_files')
    # 必须明确提供生成器检查点，避免使用未知权重。
    parser.add_argument('--checkpoint_file', required=True)
    # 将终端输入转换成后续函数可读取的参数对象。
    a = parser.parse_args()

    # 按上游约定，从检查点同目录读取配套config.json。
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    # 打开生成器对应的网络及声学配置。
    with open(config_file) as f:
        # 读入完整JSON文本，供后面解析。
        data = f.read()

    # 更新模块共享的配置对象，使get_mel与inference读取同一设置。
    global h
    # 将JSON文本解析成普通Python字典。
    json_config = json.loads(data)
    # 增加点号访问方式，兼容上游Generator代码。
    h = AttrDict(json_config)

    # 固定PyTorch的CPU随机种子，记录可复现的初始化环境。
    torch.manual_seed(h.seed)
    # 更新模块共享设备，保证模型和输入张量放在同一位置。
    global device
    # 当前环境存在可用CUDA设备时选择GPU。
    if torch.cuda.is_available():
        # 固定当前CUDA设备的随机种子。
        torch.cuda.manual_seed(h.seed)
        # 把GPU设为模型和输入张量的共同设备。
        device = torch.device('cuda')
    # 当前环境没有CUDA时保留上游CPU推理路径。
    else:
        # 选择CPU设备；项目实际任务仍必须遵守出租服务器运行约束。
        device = torch.device('cpu')

    # 使用已解析配置和设备，执行全部输入文件的合成。
    inference(a)


# 只有直接执行本脚本才开始推理，被import时仅提供函数。
if __name__ == '__main__':
    # 进入本文件的命令行入口。
    main()
