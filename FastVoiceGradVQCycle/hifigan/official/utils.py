# 这些辅助函数负责官方频谱图、卷积初始化与权重归档，不改变评价指标口径。
"""HiFi-GAN 官方通用工具（教学注释版）。

这里没有神经网络主结构，只有各文件都会用的小工具：

* 把频谱画成 Matplotlib 图；
* 初始化卷积权重、施加 weight normalization；
* 计算保持一维卷积长度所需的 padding；
* 加载/保存/扫描检查点。

``weight_norm`` 把权重的长度与方向重新参数化，训练更稳定；正式推理时 Generator
会调用 ``remove_weight_norm`` 把它还原成普通权重，减少额外计算。
"""

# glob 按通配符找文件；os 管路径；matplotlib 画图；torch 保存张量字典。
import glob
# os检查模型文件是否存在，并拼接检查点路径。
import os
# 加载绘图库，随后选择无需图形桌面的后端。
import matplotlib
# PyTorch读写模型权重，并原地初始化卷积张量。
import torch
# 引入卷积权重归一化包装，供模型初始化辅助函数使用。
from torch.nn.utils import weight_norm
# 使用无窗口绘图后端，服务器无需显示器即可生成频谱图。
matplotlib.use("Agg")
# plt负责创建图、显示频谱和关闭图资源。
import matplotlib.pylab as plt


# 将二维频谱画成图，供训练日志展示还原效果。
def plot_spectrogram(spectrogram):
    # 把二维频谱绘成图，交给训练日志检查时频结构。
    """把二维频谱矩阵画成可写入 TensorBoard 的 Figure。"""

    # 建立宽10英寸、高2英寸的频谱绘图区。
    fig, ax = plt.subplots(figsize=(10, 2))
    # 将矩阵显示为热力图，低频置于底部，长宽自适应。
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   # 不平滑插值，让图中每个频谱数值按原样显示。
                   interpolation='none')
    # 增加颜色与频谱强度的对应刻度。
    plt.colorbar(im, ax=ax)

    # 执行实际绘制，确保返回的图已经生成图像内容。
    fig.canvas.draw()
    # 从当前绘图会话关闭图，降低批量训练日志的资源占用。
    plt.close()

    # 返回仍可写入TensorBoard的Figure对象。
    return fig


# 对卷积层使用均值0、标准差0.01的正态分布建立初始权重。
def init_weights(m, mean=0.0, std=0.01):
    # 只对卷积类模块使用给定均值与标准差的正态初始化。
    """若模块类名含 Conv，就用指定正态分布初始化它的权重。"""

    # 读取模块类名，用来判断当前对象是否是卷积层。
    classname = m.__class__.__name__
    # 仅处理名称含Conv的卷积模块，跳过激活和容器。
    if classname.find("Conv") != -1:
        # 将卷积权重填成指定正态分布，作为训练起点。
        m.weight.data.normal_(mean, std)


# 为遍历到的卷积添加权重归一化。
def apply_weight_norm(m):
    # 读取模块类名，用来判断当前对象是否是卷积层。
    classname = m.__class__.__name__
    # 仅处理名称含Conv的卷积模块，跳过激活和容器。
    if classname.find("Conv") != -1:
        # 把当前卷积权重重参数化为方向和尺度。
        weight_norm(m)


# 计算奇数卷积核配合膨胀率时所需的单侧补边长度。
def get_padding(kernel_size, dilation=1):
    # 按卷积核大小与膨胀间隔计算对称补边，在指定步幅下保持时间尺寸。
    """返回奇数卷积核在 stride=1 时保持时间长度的左右 padding。"""

    # 有效卷积宽度减1后除2，保证步长1时输出长度不变。
    return int((kernel_size*dilation - dilation)/2)


# 把指定检查点加载到请求设备，返回其中保存的状态字典。
def load_checkpoint(filepath, device):
    # 先确认权重路径是实际文件，避免误把目录交给加载器。
    assert os.path.isfile(filepath)
    # 在日志中显示真正读取的权重路径。
    print("Loading '{}'".format(filepath))
    # 反序列化上游检查点；此入口只应用于来源可信的权重。
    checkpoint_dict = torch.load(filepath, map_location=device)
    # 提示检查点读取已经结束。
    print("Complete.")
    # 返回检查点内容，由调用方分别加载生成器或优化器。
    return checkpoint_dict


# 将当前模型或训练状态保存到指定检查点文件。
def save_checkpoint(filepath, obj):
    # 记录本次写盘的检查点路径。
    print("Saving checkpoint to {}".format(filepath))
    # 序列化传入对象；上游函数直接写目标文件。
    torch.save(obj, filepath)
    # 提示模型状态已经写出。
    print("Complete.")


# 查找指定前缀及8位步数命名的最新上游检查点。
def scan_checkpoint(cp_dir, prefix):
    # 每个问号匹配一个字符，要求前缀后恰好8个位置。
    pattern = os.path.join(cp_dir, prefix + '????????')
    # 列出当前目录所有符合检查点命名规则的文件。
    cp_list = glob.glob(pattern)
    # 没有匹配文件时向调用方报告无可恢复检查点。
    if len(cp_list) == 0:
        # 用None表示尚未保存这类检查点。
        return None
    # 按固定宽度步数文件名排序，取最大步数对应文件。
    return sorted(cp_list)[-1]
