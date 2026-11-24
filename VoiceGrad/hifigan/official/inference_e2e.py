# 此上游入口直接读取 mel 文件并调用 G，不执行 VoiceGrad 的去噪转换。
"""HiFi-GAN 官方“mel 输入”推理脚本（教学注释版）。

这里的 e2e 表示端到端从已保存的 mel 数组直接生成波形。与 inference.py 的核心
区别是它不读取原始 WAV，也不重新计算 mel。官方文件对输入数组形状假设较强，
因此本项目正式推理使用检查更严、支持原子写文件的 ``vocode_voicegrad_mels.py``。

``np.load`` 读 NumPy 数组，``torch.FloatTensor`` 转张量，Generator 完成上采样，
``squeeze`` 删除长度为 1 的 batch/channel 轴，最后写成 16-bit WAV。
"""

# 保留上游对导入、除法和文本语义的兼容设置。
from __future__ import absolute_import, division, print_function, unicode_literals

# 用通配符查找符合命名规则的已有检查点。
import glob
# 拼接输入输出路径、列出目录内容并建立结果目录。
import os
# 读取或转换NumPy格式的mel和波形数组。
import numpy as np
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
# 读取上游16位PCM与浮点波形之间的缩放系数。
from meldataset import MAX_WAV_VALUE
# 加载将80维mel上采样为波形的HiFi-GAN生成器结构。
from models import Generator

# 配置在main读取JSON后填入，供推理函数共享。
h = None
# 实际CPU或GPU设备在main初始化后确定。
device = None


# 读取来源可信的上游权重文件，并映射到指定设备。
def load_checkpoint(filepath, device):
    # 加载已保存状态，并检查其结构和当前运行约定。
    """加载含 ``generator`` 键的官方检查点。"""

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


# 查找目录中符合前缀的最新文件名，供检查点选择使用。
def scan_checkpoint(cp_dir, prefix):
    # 按固定前缀查找并排序归档文件，无匹配时返回空结果。
    """按文件名排序寻找某一前缀的最新检查点。"""

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
    """逐个读取 mel 文件并通过生成器输出 WAV。"""

    # 按配套配置建立生成器，并把其参数放到当前设备。
    generator = Generator(h).to(device)

    # 读取用户指定的生成器检查点。
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    # 把检查点generator字段中的参数填入网络。
    generator.load_state_dict(state_dict_g['generator'])

    # 列出待合成mel目录中的文件，输入数组需满足上游形状要求。
    filelist = os.listdir(a.input_mels_dir)

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
            # 读取当前预先保存的mel数组；此脚本不会自动猜测轴顺序。
            x = np.load(os.path.join(a.input_mels_dir, filname))
            # 将mel转成float32张量，并传到生成器设备。
            x = torch.FloatTensor(x).to(device)
            # 把mel一次上采样成合成波形。
            y_g_hat = generator(x)
            # 去除长度为1的batch和声道轴，得到一维采样序列。
            audio = y_g_hat.squeeze()
            # 把浮点采样值放大到16位PCM数值范围。
            audio = audio * MAX_WAV_VALUE
            # 将结果搬回CPU，并转换为写WAV所需的16位整数。
            audio = audio.cpu().numpy().astype('int16')

            # 保留mel文件主名，增加_generated_e2e后缀标识直接合成。
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
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
    # 指定预先保存的mel数组目录。
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    # 指定由mel直接合成的WAV目录。
    parser.add_argument('--output_dir', default='generated_files_from_mel')
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
