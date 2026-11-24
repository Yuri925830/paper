# 此文件保留官方 G、MPD、MSD 与损失定义，自训 MRD 由 custom/mrd.py 提供。
"""HiFi-GAN 官方网络结构与损失函数（教学注释版）。

把本文件按三层理解即可：

* ``Generator``：把低时间分辨率 mel 逐级上采样为波形；
* ``MultiPeriodDiscriminator``：以 2/3/5/7/11 周期重排波形，检查周期结构真假；
* ``MultiScaleDiscriminator``：在原始及下采样尺度检查真假。

根目录的论文对齐训练没有使用 MSD，而是以 ``vocoder_paper_mrd.py`` 的 MRD 替代；
Generator、MPD、feature/generator/discriminator loss 仍直接来自这里。

PyTorch 模块的共同语法：``__init__`` 建层，``forward`` 写数据怎样流过层，
调用 ``module(x)`` 时框架会自动转到 forward。``ModuleList`` 会登记其中的参数，
普通 Python list 不会被优化器自动发现。
"""

# torch 是张量/自动微分核心；F 放无状态函数；nn 放可训练网络层。
import torch
# F 提供激活、补边等张量运算，参与声码器及判别器的梯度计算。
import torch.nn.functional as F
# nn 提供模块容器，使卷积和残差块的参数都能交给优化器。
import torch.nn as nn
# 分别引入一维卷积、转置上采样卷积、平均池化、周期波形二维卷积。
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
# 训练时规范化权重；推理时可合并权重归一化参数；MSD首分支另用谱归一化。
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
# 复用同目录的卷积初始化和长度保持补边计算。
from utils import init_weights, get_padding

# 负输入仍保留十分之一的斜率，减少激活截断后无法学习的情况。
LRELU_SLOPE = 0.1


# V1残差块用三组不同膨胀率观察短时和较长时的波形结构。
class ResBlock1(torch.nn.Module):
    # 三组双卷积分别学习修正，再加回输入，保持残差结构和时间尺寸。
    """带三组膨胀卷积的残差块；输出 ``x + 学到的修正量``。"""

    # 按通道数、卷积宽度和三个膨胀率建立V1残差块。
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        # 登记为PyTorch模块，后续添加的卷积会成为其可训练参数。
        super(ResBlock1, self).__init__()
        # 保存本块或生成器使用的HiFi-GAN配置，方便其它方法读取。
        self.h = h
        # 登记每组的第一层膨胀卷积，三种采样间隔扩大观察范围。
        self.convs1 = nn.ModuleList([
            # 第一种膨胀率卷积；输入输出通道一致，便于残差相加。
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               # 按第一种膨胀率补齐边界，保证卷积前后时间长度相同。
                               padding=get_padding(kernel_size, dilation[0]))),
            # 第二种膨胀率卷积，用相同卷积宽度观察更疏的时间位置。
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               # 按第二种膨胀率补边，保持残差两侧可逐位置相加。
                               padding=get_padding(kernel_size, dilation[1]))),
            # 第三种膨胀率卷积，用于观察更长时间范围。
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               # 按第三种膨胀率补边，仍保持原时间长度。
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        # 对第一组全部卷积执行上游初始化，建立训练起点。
        self.convs1.apply(init_weights)

        # 每个膨胀卷积后接一个普通卷积，进一步整合相邻时间信息。
        self.convs2 = nn.ModuleList([
            # 普通卷积不跳过采样点，输入和输出通道保持一致。
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               # 用普通卷积所需补边保持当前波形特征长度。
                               padding=get_padding(kernel_size, 1))),
            # 普通卷积不跳过采样点，输入和输出通道保持一致。
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               # 用普通卷积所需补边保持当前波形特征长度。
                               padding=get_padding(kernel_size, 1))),
            # 普通卷积不跳过采样点，输入和输出通道保持一致。
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               # 最后一个普通卷积也保持时间长度，供残差连接相加。
                               padding=get_padding(kernel_size, 1)))
        ])
        # 为每组第二层普通卷积设置初始参数。
        self.convs2.apply(init_weights)

    # 残差块 1 依次通过三组双卷积，保留输入并学习波形细节修正。
    def forward(self, x):
        # 每组两层卷积学习局部修正，三组依次相加后返回与输入同尺寸的隐藏特征。
        """依次经过激活和两层卷积，并在每小组末尾做残差相加。"""

        # 按同一组配对膨胀卷积和普通卷积，依次细化波形特征。
        for c1, c2 in zip(self.convs1, self.convs2):
            # 先激活残差分支，负值保留0.1斜率，再进入当前卷积。
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 用当前膨胀卷积提取更宽时间范围内的特征。
            xt = c1(xt)
            # 第一层输出再次激活，为后续普通卷积提供非线性特征。
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            # 普通卷积整合膨胀层输出，形成待加回主路径的修正量。
            xt = c2(xt)
            # 把学到的修正量加回原特征，保留已有波形信息。
            x = xt + x
        # 返回经过三组残差细化的特征，时间长度和通道数保持不变。
        return x

    # 推理时把归一化参数合并为普通权重，保持同一数学映射。
    def remove_weight_norm(self):
        # 逐一处理第一组膨胀卷积的权重归一化。
        for l in self.convs1:
            # 将当前卷积的方向和尺度参数合并为实际卷积权重。
            remove_weight_norm(l)
        # 逐一处理第二组普通卷积的权重归一化。
        for l in self.convs2:
            # 将当前卷积的方向和尺度参数合并为实际卷积权重。
            remove_weight_norm(l)


# 保留官方第二种轻量残差块；V1通常通过配置选择ResBlock1。
class ResBlock2(torch.nn.Module):
    # 建立两层膨胀卷积的轻量残差块。
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        # 登记轻量残差块及其卷积参数。
        super(ResBlock2, self).__init__()
        # 保存本块或生成器使用的HiFi-GAN配置，方便其它方法读取。
        self.h = h
        # 将当前残差块或判别器的卷积按顺序登记为可训练层。
        self.convs = nn.ModuleList([
            # 第一种膨胀率卷积；输入输出通道一致，便于残差相加。
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               # 按第一种膨胀率补齐边界，保证卷积前后时间长度相同。
                               padding=get_padding(kernel_size, dilation[0]))),
            # 第二种膨胀率卷积，用相同卷积宽度观察更疏的时间位置。
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               # 补齐第二种膨胀卷积的边界，保持残差两侧形状相同。
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        # 对当前轻量残差块的所有卷积执行上游初始化。
        self.convs.apply(init_weights)

    # 残差块 2 逐层进行膨胀卷积，并把每次修正加回原输入。
    def forward(self, x):
        # 逐层进行激活、膨胀卷积和残差相加。
        for c in self.convs:
            # 先激活残差分支，负值保留0.1斜率，再进入当前卷积。
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 当前膨胀卷积从激活特征中计算新的残差修正量。
            xt = c(xt)
            # 把学到的修正量加回原特征，保留已有波形信息。
            x = xt + x
        # 返回较轻量残差块处理后的特征，供生成器继续上采样。
        return x

    # 推理时把归一化参数合并为普通权重，保持同一数学映射。
    def remove_weight_norm(self):
        # 遍历本块的全部卷积，逐层合并归一化权重。
        for l in self.convs:
            # 将当前卷积的方向和尺度参数合并为实际卷积权重。
            remove_weight_norm(l)


# 生成器G把80维log-mel逐级放大为单声道音频波形。
class Generator(torch.nn.Module):
    # 使用配置h确定上采样倍数、初始通道数和残差块类型。
    def __init__(self, h):
        # 登记生成器模块，使全部上采样和残差参数可训练。
        super(Generator, self).__init__()
        # 保存本块或生成器使用的HiFi-GAN配置，方便其它方法读取。
        self.h = h
        # 每次上采样后并行使用几种残差卷积宽度，通常是3种。
        self.num_kernels = len(h.resblock_kernel_sizes)
        # 记录上采样阶段数；V1配置通常为4级。
        self.num_upsamples = len(h.upsample_rates)
        # 将80维mel投影到高维隐特征；宽7卷积和两侧补3保持帧数。
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        # 按配置选择V1残差块或上游另一种轻量结构。
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        # 准备登记逐级上采样卷积，不能用普通列表遗漏参数。
        self.ups = nn.ModuleList()
        # 逐级读取时间放大倍数u和转置卷积宽度k。
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            # 登记当前级转置卷积，并使用权重归一化辅助训练。
            self.ups.append(weight_norm(
                # 每次上采样将通道数减半，用更长时间轴表示音频细节。
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                # 卷积宽度k、步长u及配套补边使时间长度按u倍增加。
                                k, u, padding=(k-u)//2)))

        # 登记各上采样阶段的多种残差块，覆盖不同时间范围。
        self.resblocks = nn.ModuleList()
        # 为每一个上采样阶段分别创建后续残差块。
        for i in range(len(self.ups)):
            # 当前阶段的通道数与对应上采样层输出保持一致。
            ch = h.upsample_initial_channel//(2**(i+1))
            # 按配置配对每种残差卷积宽度与膨胀率组合。
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                # 建立当前尺度的残差块，使不同观察范围共同参与合成。
                self.resblocks.append(resblock(h, ch, k, d))

        # 最后把隐特征压到1个音频通道，同时保持波形采样点数。
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        # 初始化全部上采样卷积，建立训练前的生成器参数。
        self.ups.apply(init_weights)
        # 初始化最后输出单声道波形的卷积。
        self.conv_post.apply(init_weights)

    # 输入[B,80,mel帧数]，输出[B,1,mel帧数×上采样总倍数]。
    def forward(self, x):
        # 先把mel频带转换成可逐级上采样的隐藏特征。
        x = self.conv_pre(x)
        # 依次执行全部上采样阶段，把mel帧展开为波形采样点。
        for i in range(self.num_upsamples):
            # 给当前特征加入非线性，负值仍保留0.1斜率。
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 当前转置卷积将时间轴放大，同时降低通道数。
            x = self.ups[i](x)
            # 准备累加当前阶段不同卷积宽度残差块的输出。
            xs = None
            # 在同一尺度并行评估各个残差块。
            for j in range(self.num_kernels):
                # 第一个残差块先建立累加张量，后续结果再相加。
                if xs is None:
                    # 取当前阶段第一个残差块输出作为多尺度特征初值。
                    xs = self.resblocks[i*self.num_kernels+j](x)
                # 已有残差块结果后，累加另一种卷积宽度的特征。
                else:
                    # 将当前阶段另一种观察范围的特征加到总和中。
                    xs += self.resblocks[i*self.num_kernels+j](x)
            # 对并行残差块取平均，避免分支数直接放大信号。
            x = xs / self.num_kernels
        # 保留上游此处默认负斜率的激活，随后输出最终波形。
        x = F.leaky_relu(x)
        # 最后一层卷积将当前隐藏特征变成1通道输出。
        x = self.conv_post(x)
        # 把生成波形限制在[-1,1]范围内，便于保存和判别。
        x = torch.tanh(x)

        # 返回浮点波形；调用方决定是否转PCM、计算损失或评估。
        return x

    # 推理时把归一化参数合并为普通权重，保持同一数学映射。
    def remove_weight_norm(self):
        # 提示推理优化正在合并生成器的权重归一化。
        print('Removing weight norm...')
        # 逐层合并上采样卷积的归一化参数。
        for l in self.ups:
            # 将当前卷积的方向和尺度参数合并为实际卷积权重。
            remove_weight_norm(l)
        # 每个残差块内部也需要合并其卷积权重。
        for l in self.resblocks:
            # 让当前残差块处理自己的全部卷积层。
            l.remove_weight_norm()
        # 合并mel输入投影层的归一化参数。
        remove_weight_norm(self.conv_pre)
        # 合并最终波形输出层的归一化参数。
        remove_weight_norm(self.conv_post)


# 一个MPD分支按固定周期重排波形，检查有声语音的周期结构。
class DiscriminatorP(torch.nn.Module):
    # period规定周期列数；卷积沿周期组方向提取真假特征。
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        # 登记单个周期判别器的卷积及归一化参数。
        super(DiscriminatorP, self).__init__()
        # 保存该分支的周期长度，前向时据此补边和重排波形。
        self.period = period
        # 默认使用权重归一化；只有显式指定时切换到谱归一化。
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # 将当前残差块或判别器的卷积按顺序登记为可训练层。
        self.convs = nn.ModuleList([
            # 从单通道周期图提取32通道特征，只沿周期组方向下采样。
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            # 将周期特征扩展到128通道，保持每个周期列位置。
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            # 提取512通道更高层周期特征，并压缩周期组数。
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            # 进一步形成1024通道特征，扩大真假判断的时间范围。
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            # 最后一层不再下采样，细化高层周期特征。
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        # 把高层周期特征转为单通道真假分数图，不使用sigmoid。
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    # 输入单声道波形，返回周期真假分数和各层特征。
    def forward(self, x):
        # 保留各层真假特征；生成器训练时用它们计算特征匹配损失。
        fmap = []

        # 1d to 2d
        # 读取批量B、声道C和采样点数T，随后将时间轴拆成周期组。
        b, c, t = x.shape
        # 采样点数无法被周期整除时先补边，保证能够重排成矩阵。
        if t % self.period != 0: # pad first
            # 计算到下一个完整周期还缺多少波形采样点。
            n_pad = self.period - (t % self.period)
            # 用右侧反射补边补成完整周期，避免引入长串零值边界。
            x = F.pad(x, (0, n_pad), "reflect")
            # 记录补边后的实际采样点数，用于后续形状变换。
            t = t + n_pad
        # 把波形变成[B,C,周期组数,周期长度]，同列表示相同周期相位。
        x = x.view(b, c, t // self.period, self.period)

        # 遍历本判别器各卷积层，逐步提取波形的真假判别特征。
        for l in self.convs:
            # 当前卷积处理波形或周期图，生成下一层判别特征。
            x = l(x)
            # 给当前特征加入非线性，负值仍保留0.1斜率。
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 保存该层特征，后续与真实音频对应层作L1比较。
            fmap.append(x)
        # 最后一层卷积将当前隐藏特征变成1通道输出。
        x = self.conv_post(x)
        # 保存该层特征，后续与真实音频对应层作L1比较。
        fmap.append(x)
        # 保留batch轴，把所有局部真假分数展开为每条音频的一行。
        x = torch.flatten(x, 1, -1)

        # 同时返回局部真假分数和各层特征，分别用于对抗及特征匹配。
        return x, fmap


# MPD同时使用五个互质周期检查波形，减少只看单一周期的偏差。
class MultiPeriodDiscriminator(torch.nn.Module):
    # 建立周期2、3、5、7、11的五个独立分支。
    def __init__(self):
        # 登记五个周期判别器，使其参数共同参与训练。
        super(MultiPeriodDiscriminator, self).__init__()
        # 登记并行判别器分支，各分支独立处理自己的周期或尺度。
        self.discriminators = nn.ModuleList([
            # 周期2分支检查每两个采样点重复的局部结构。
            DiscriminatorP(2),
            # 周期3分支从另一种周期重排观察波形。
            DiscriminatorP(3),
            # 周期5分支补充不同相位组织下的真假信息。
            DiscriminatorP(5),
            # 周期7分支观察较长周期结构。
            DiscriminatorP(7),
            # 周期11分支提供五组中最长的重排周期。
            DiscriminatorP(11),
        ])

    # 同时输入真实波形y和生成波形y_hat，分支权重两者共用。
    def forward(self, y, y_hat):
        # 收集各判别器对真实波形的分数。
        y_d_rs = []
        # 收集各判别器对生成波形的分数。
        y_d_gs = []
        # 收集真实波形在各分支、各层的中间特征。
        fmap_rs = []
        # 收集生成波形的对应特征，与真实特征保持相同排列。
        fmap_gs = []
        # 按登记顺序执行每个判别器分支，保持真假结果对应。
        for i, d in enumerate(self.discriminators):
            # 当前判别器评估真实波形，并保留中间特征。
            y_d_r, fmap_r = d(y)
            # 同一判别器评估生成波形，得到可与真实结果对照的特征。
            y_d_g, fmap_g = d(y_hat)
            # 把当前分支的真实分数加入列表。
            y_d_rs.append(y_d_r)
            # 按分支顺序保存真实中间特征。
            fmap_rs.append(fmap_r)
            # 把当前分支的生成分数加入列表。
            y_d_gs.append(y_d_g)
            # 按同样顺序保存生成中间特征。
            fmap_gs.append(fmap_g)

        # 按固定顺序返回真实分数、生成分数、真实特征及生成特征。
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# 保留上游MSD单尺度波形判别器；项目修改版训练以MRD替换MSD。
class DiscriminatorS(torch.nn.Module):
    # 建立一个尺度的判别器，并选择权重或谱归一化。
    def __init__(self, use_spectral_norm=False):
        # 登记单尺度波形判别器的全部卷积。
        super(DiscriminatorS, self).__init__()
        # 默认使用权重归一化；只有显式指定时切换到谱归一化。
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # 将当前残差块或判别器的卷积按顺序登记为可训练层。
        self.convs = nn.ModuleList([
            # 从原始单声道波形提取128通道局部特征。
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            # 宽41分组卷积按2倍压缩时间，并控制参数量。
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            # 扩展到256通道，再按2倍缩短时间轴。
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            # 按4倍下采样并增加通道，扩大波形观察范围。
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            # 继续按4倍下采样，形成1024通道高层特征。
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            # 保持时间分辨率，用分组卷积细化高层特征。
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            # 用较窄卷积整合相邻高层特征，维持当前长度。
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        # 输出每个局部位置的真假分数，供最小二乘损失使用。
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    # 在当前音频尺度上输出局部真假分数及各层特征。
    def forward(self, x):
        # 保留各层真假特征；生成器训练时用它们计算特征匹配损失。
        fmap = []
        # 遍历本判别器各卷积层，逐步提取波形的真假判别特征。
        for l in self.convs:
            # 当前卷积处理波形或周期图，生成下一层判别特征。
            x = l(x)
            # 给当前特征加入非线性，负值仍保留0.1斜率。
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 保存该层特征，后续与真实音频对应层作L1比较。
            fmap.append(x)
        # 最后一层卷积将当前隐藏特征变成1通道输出。
        x = self.conv_post(x)
        # 保存该层特征，后续与真实音频对应层作L1比较。
        fmap.append(x)
        # 保留batch轴，把所有局部真假分数展开为每条音频的一行。
        x = torch.flatten(x, 1, -1)

        # 同时返回局部真假分数和各层特征，分别用于对抗及特征匹配。
        return x, fmap


# 上游MSD从原始及两级池化波形判断真假；本项目自训改用MRD。
class MultiScaleDiscriminator(torch.nn.Module):
    # 建立原始、第一次池化、第二次池化三个音频尺度。
    def __init__(self):
        # 登记三个尺度判别器和两个平均池化层。
        super(MultiScaleDiscriminator, self).__init__()
        # 登记并行判别器分支，各分支独立处理自己的周期或尺度。
        self.discriminators = nn.ModuleList([
            # 原始音频尺度分支使用谱归一化，保持上游设定。
            DiscriminatorS(use_spectral_norm=True),
            # 下采样尺度分支使用默认权重归一化。
            DiscriminatorS(),
            # 下采样尺度分支使用默认权重归一化。
            DiscriminatorS(),
        ])
        # 登记相邻尺度间的平均池化操作。
        self.meanpools = nn.ModuleList([
            # 宽4、步长2的平均池化降低时间分辨率。
            AvgPool1d(4, 2, padding=2),
            # 第二次同样池化，形成更粗的波形尺度。
            AvgPool1d(4, 2, padding=2)
        ])

    # 同时输入真实波形y和生成波形y_hat，分支权重两者共用。
    def forward(self, y, y_hat):
        # 收集各判别器对真实波形的分数。
        y_d_rs = []
        # 收集各判别器对生成波形的分数。
        y_d_gs = []
        # 收集真实波形在各分支、各层的中间特征。
        fmap_rs = []
        # 收集生成波形的对应特征，与真实特征保持相同排列。
        fmap_gs = []
        # 按登记顺序执行每个判别器分支，保持真假结果对应。
        for i, d in enumerate(self.discriminators):
            # 第一个分支直接看原始波形，后续分支先降低分辨率。
            if i != 0:
                # 真实波形按当前尺度进行平均池化。
                y = self.meanpools[i-1](y)
                # 生成波形使用完全相同的池化，保证比较条件一致。
                y_hat = self.meanpools[i-1](y_hat)
            # 当前判别器评估真实波形，并保留中间特征。
            y_d_r, fmap_r = d(y)
            # 同一判别器评估生成波形，得到可与真实结果对照的特征。
            y_d_g, fmap_g = d(y_hat)
            # 把当前分支的真实分数加入列表。
            y_d_rs.append(y_d_r)
            # 按分支顺序保存真实中间特征。
            fmap_rs.append(fmap_r)
            # 把当前分支的生成分数加入列表。
            y_d_gs.append(y_d_g)
            # 按同样顺序保存生成中间特征。
            fmap_gs.append(fmap_g)

        # 按固定顺序返回真实分数、生成分数、真实特征及生成特征。
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# 官方FM损失逐层比较真实与生成特征，函数内部已经乘2。
def feature_loss(fmap_r, fmap_g):
    # 从零累计当前损失函数的各分支贡献。
    loss = 0
    # 先按相同判别器分支配对真实和生成特征。
    for dr, dg in zip(fmap_r, fmap_g):
        # 再按相同网络层配对两侧中间特征。
        for rl, gl in zip(dr, dg):
            # 当前层先求平均绝对差，再加入所有层的总和。
            loss += torch.mean(torch.abs(rl - gl))

    # 返回已乘权重2的官方FM损失；调用者不得无意重复乘2。
    return loss*2


# D希望真实分数接近1、生成分数接近0，按分支累加最小二乘误差。
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    # 从零累计当前损失函数的各分支贡献。
    loss = 0
    # 保留各分支真实样本损失，方便诊断哪一类分支变化。
    r_losses = []
    # 保留各分支生成样本损失，配合真实项查看D表现。
    g_losses = []
    # 按相同判别器分支配对真实与生成分数。
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # 真实波形标签设为1，计算其平均平方误差。
        r_loss = torch.mean((1-dr)**2)
        # 生成波形在D训练阶段标签为0，计算平均平方误差。
        g_loss = torch.mean(dg**2)
        # 把当前分支的真假两项加入总判别器损失。
        loss += (r_loss + g_loss)
        # 将真实分支损失转普通数值用于日志，不保留反向图。
        r_losses.append(r_loss.item())
        # 将生成分支损失转普通数值用于日志。
        g_losses.append(g_loss.item())

    # 总损失保留梯度；两份分项列表仅用于报告。
    return loss, r_losses, g_losses


# G希望各判别器把生成波形判断为真实，因此把其分数推向1。
def generator_loss(disc_outputs):
    # 从零累计当前损失函数的各分支贡献。
    loss = 0
    # 保存各分支对G的对抗损失，方便分项记录。
    gen_losses = []
    # 逐个读取判别器对生成波形的局部分数。
    for dg in disc_outputs:
        # 以真实标签1计算生成波形的平均平方误差。
        l = torch.mean((1-dg)**2)
        # 保留该分支可微损失，供调用方选择记录。
        gen_losses.append(l)
        # 将当前分支贡献加入生成器总对抗损失。
        loss += l

    # 返回可反向传播的对抗总损失及其分支组成。
    return loss, gen_losses
