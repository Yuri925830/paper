# 此入口训练上游 G+MPD+MSD，仅作参考；本项目 MRD 声码器使用 custom/train.py。
"""HiFi-GAN 上游官方训练入口（教学注释版）。

本文件训练原版 ``Generator + MPD + MSD``，用于理解和复核上游实现。FastVoiceGrad
要求把 MSD 换成 UnivNet MRD，所以本项目正式自训练入口是根目录
``hifigan/custom/train.py``，不能把两者的检查点目录混用。

一次迭代分两次更新：

1. 固定生成器输出，判别器比较真实/生成波形并更新 MPD、MSD；
2. 再计算 mel L1、feature matching 和对抗损失，更新 Generator。

``rank`` 是多 GPU 进程编号；一块 GPU 时就是 0。``DistributedDataParallel`` 让
每个 GPU 各处理一部分 batch 并同步梯度。初学者单卡阅读时可先忽略分布式分支。
"""

# 管理上游库的兼容性警告，训练错误仍通过异常报告。
import warnings
# 忽略未来版本弃用提醒，保留上游日志行为。
warnings.simplefilter(action='ignore', category=FutureWarning)
# 把MPD与MSD参数连接起来，交给同一个判别器优化器。
import itertools
# 管理训练输入、检查点及TensorBoard日志路径。
import os
# 记录每批和每轮耗时，方便估计训练速度。
import time
# 读取数据目录、训练轮数和保存间隔等命令行设置。
import argparse
# 解析生成器和训练超参数的JSON配置。
import json
# 提供模型张量、GPU运算、反向传播和优化器。
import torch
# 计算真实与生成mel之间的L1重建误差。
import torch.nn.functional as F
# 记录训练数值、频谱图和诊断音频；音频记录不构成主观评估。
from torch.utils.tensorboard import SummaryWriter
# 分配多GPU数据，并把逐条语音组织成训练批次。
from torch.utils.data import DistributedSampler, DataLoader
# 多卡时为各GPU启动独立训练进程。
import torch.multiprocessing as mp
# 建立多GPU进程通信组，使各卡能够同步梯度。
from torch.distributed import init_process_group
# 包装模型，在多卡训练中同步参数梯度。
from torch.nn.parallel import DistributedDataParallel
# 以属性方式读取配置，并保存配置快照。
from env import AttrDict, build_env
# 读取波形训练样本、计算配套mel并解析文件列表。
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
# 引入上游G、MPD、MSD及三个损失；自训MRD实验使用另一个训练入口。
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss  # 最小二乘真假判别损失；续行处保留反斜杠导入语法。
# 复用上游画图、查找、读取及保存检查点函数。
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

# 允许cuDNN为当前卷积形状选择较快实现，保留上游运行设置。
torch.backends.cudnn.benchmark = True


# rank确定本进程使用哪张GPU，a为路径参数，h为网络和优化设置。
def train(rank, a, h):
    # 每个 GPU 进程建立本地网络和数据批次；先更新 D，再用新 D 的信号更新 G。
    """在编号为 rank 的 GPU 进程中建立模型、恢复检查点并执行训练循环。"""

    # 配置为多卡时建立通信或同步包装，单卡跳过相应分支。
    if h.num_gpus > 1:
        # 按配置的通信后端和地址连接分布式训练进程。
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           # 总进程数由节点数乘每节点GPU数得到，并登记本进程编号。
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    # 固定本GPU随机种子，便于复现实验初始化。
    torch.cuda.manual_seed(h.seed)
    # 为本训练进程选择编号rank对应的GPU。
    device = torch.device('cuda:{:d}'.format(rank))

    # 按配置建立mel到波形的生成器，并传到本GPU。
    generator = Generator(h).to(device)
    # 建立周期2、3、5、7、11的波形判别器。
    mpd = MultiPeriodDiscriminator().to(device)
    # 建立上游原版多尺度判别器；本项目MRD实验不从此入口启动。
    msd = MultiScaleDiscriminator().to(device)

    # 仅主进程写日志、保存模型及评估，避免多卡重复写同一文件。
    if rank == 0:
        # 在启动日志显示生成器实际结构。
        print(generator)
        # 准备检查点保存目录。
        os.makedirs(a.checkpoint_path, exist_ok=True)
        # 显示本次训练写入模型的位置。
        print("checkpoints directory : ", a.checkpoint_path)

    # 目录存在时扫描可恢复的生成器及训练状态文件。
    if os.path.isdir(a.checkpoint_path):
        # 查找最新上游生成器权重。
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        # 查找最新判别器和优化器状态。
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    # 没有恢复记录时，全局更新计数从零开始。
    steps = 0
    # 两类检查点任一缺失时沿用上游从头初始化分支。
    if cp_g is None or cp_do is None:
        # 标记没有可恢复的判别器及优化器状态。
        state_dict_do = None
        # 表示尚未完成任何轮次，供学习率调度器初始化。
        last_epoch = -1
    # 此分支：读取选中的生成器权重文件。
    else:
        # 读取选中的生成器权重文件。
        state_dict_g = load_checkpoint(cp_g, device)
        # 读取配套判别器、优化器及训练游标。
        state_dict_do = load_checkpoint(cp_do, device)
        # 恢复生成器卷积参数。
        generator.load_state_dict(state_dict_g['generator'])
        # 恢复五个周期判别器的参数。
        mpd.load_state_dict(state_dict_do['mpd'])
        # 恢复三个尺度判别器的参数。
        msd.load_state_dict(state_dict_do['msd'])
        # 从检查点记录之后的下一步继续计数。
        steps = state_dict_do['steps'] + 1
        # 取得恢复时的轮次，供训练循环和调度器使用。
        last_epoch = state_dict_do['epoch']

    # 配置为多卡时建立通信或同步包装，单卡跳过相应分支。
    if h.num_gpus > 1:
        # 多卡下包装G，让每卡的生成器梯度同步。
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        # 多卡下同步MPD各分支的参数梯度。
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        # 多卡下同步MSD各尺度的参数梯度。
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    # 建立仅更新G的AdamW，使用声码器配置的学习率与动量。
    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    # 把MSD和MPD全部参数交给同一个D优化器。
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                # D沿用声码器配置的学习率与动量参数。
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    # 有完整训练状态时恢复两个优化器的历史动量。
    if state_dict_do is not None:
        # 恢复生成器优化器状态，避免续训时重置动量。
        optim_g.load_state_dict(state_dict_do['optim_g'])
        # 恢复判别器优化器状态。
        optim_d.load_state_dict(state_dict_do['optim_d'])

    # 建立G的逐轮指数学习率衰减，并接续历史轮次。
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    # D使用相同形式的学习率衰减。
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # 分别读取上游训练和验证WAV路径列表。
    training_filelist, validation_filelist = get_dataset_filelist(a)

    # 为训练语音建立固定波形片段和mel配对数据集。
    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          # 按声学配置提取mel，并关闭WAV缓存复用。
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          # 多卡交给采样器打乱；指定损失mel频率范围和设备。
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          # 由微调开关决定是否使用预先计算的条件mel。
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    # 多卡时给各GPU分配不重复样本；单卡直接使用数据集。
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    # 用多个读取进程提供训练批次，打乱逻辑由数据集或采样器控制。
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              # 在多卡时使用刚建立的分布式样本分配规则。
                              sampler=train_sampler,
                              # 设置本GPU每次更新处理的语音条数。
                              batch_size=h.batch_size,
                              # 使用固定页CPU内存，加速向GPU传输。
                              pin_memory=True,
                              # 舍弃不足一整批的尾部样本，维持训练批量一致。
                              drop_last=True)

    # 仅主进程写日志、保存模型及评估，避免多卡重复写同一文件。
    if rank == 0:
        # 为上游验证列表创建数据集，仅由主进程读取。
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              # 验证不随机裁剪或打乱，并关闭缓存复用。
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              # 验证使用相同损失mel定义及微调模式。
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              # 指定微调所需预计算mel目录。
                              base_mels_path=a.input_mels_dir)
        # 按固定顺序逐条读取验证语音。
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       # 验证由主进程完整遍历，不分配到多张GPU。
                                       sampler=None,
                                       # 验证每次处理一条完整语音，允许句长不同。
                                       batch_size=1,
                                       # 使用固定页CPU内存，加速向GPU传输。
                                       pin_memory=True,
                                       # 舍弃不足一整批的尾部样本，维持训练批量一致。
                                       drop_last=True)

        # 在检查点目录建立TensorBoard日志。
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    # 将G恢复到训练模式，继续参数更新。
    generator.train()
    # 周期判别器进入训练模式。
    mpd.train()
    # 尺度判别器进入训练模式。
    msd.train()
    # 按上游恢复轮次约定继续训练，直到给定总轮数。
    for epoch in range(max(0, last_epoch), a.training_epochs):
        # 仅主进程写日志、保存模型及评估，避免多卡重复写同一文件。
        if rank == 0:
            # 记录本轮开始时间，稍后计算整轮耗时。
            start = time.time()
            # 以从1开始的轮次显示训练进度。
            print("Epoch: {}".format(epoch+1))

        # 配置为多卡时建立通信或同步包装，单卡跳过相应分支。
        if h.num_gpus > 1:
            # 多卡采样器每轮使用新的确定性打乱顺序。
            train_sampler.set_epoch(epoch)

        # 逐批取出条件mel和配对真实波形。
        for i, batch in enumerate(train_loader):
            # 仅主进程写日志、保存模型及评估，避免多卡重复写同一文件。
            if rank == 0:
                # 记录当前批开始时间，用于输出每批耗时。
                start_b = time.time()
            # x为条件mel，y为真实波形，y_mel为重建损失的真实mel。
            x, y, _, y_mel = batch
            # 把条件mel异步传到GPU，保留上游Variable包装。
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            # 把真实波形传到同一GPU。
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            # 把真实损失mel传到GPU，供重建误差比较。
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            # 给真实波形增加声道轴，成为[B,1,T]。
            y = y.unsqueeze(1)

            # 由当前条件mel生成配对音频波形。
            y_g_hat = generator(x)
            # 把生成波形重新提取为可微mel，用于约束声学重建。
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          # 重建比较使用配置的损失频率范围。
                                          h.fmin, h.fmax_for_loss)

            # 先清空上一步D参数梯度，开始本批判别器更新。
            optim_d.zero_grad()

            # MPD
            # MPD比较真实和生成波形；detach阻止D更新传回G。
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            # 计算MPD真假最小二乘总损失及分项。
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            # MSD在多个波形尺度进行相同真假比较，并切断G梯度。
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            # 计算MSD真假损失及分项。
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            # 将两类判别器损失相加，共同训练D。
            loss_disc_all = loss_disc_s + loss_disc_f

            # 计算MPD及MSD参数梯度。
            loss_disc_all.backward()
            # 使用本批梯度更新两个判别器族一次。
            optim_d.step()

            # Generator
            # 开始G更新前清空上一步G梯度。
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            # 真实与生成mel的平均绝对误差乘45，形成声码器重建项。
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            # 使用已更新的MPD重新评分，并保留G方向的梯度和特征。
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            # 使用已更新的MSD获取生成分数及两侧中间特征。
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            # 比较MPD对应层真实与生成特征，官方函数内部已乘2。
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            # 比较MSD对应层特征，保持同样权重定义。
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            # G希望MPD把生成音频分数推向真实标签1。
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            # G同时希望MSD把生成音频判断为真实。
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            # G总损失包括两类对抗项、两类FM项和乘45的mel重建项。
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            # 沿判别器及mel计算反传，取得生成器梯度。
            loss_gen_all.backward()
            # 完成本批生成器参数更新。
            optim_g.step()

            # 仅主进程写日志、保存模型及评估，避免多卡重复写同一文件。
            if rank == 0:
                # STDOUT logging
                # 到达终端日志间隔时记录当前训练数值。
                if steps % a.stdout_interval == 0:
                    # 诊断数值和验证不构建梯度图，避免额外显存占用。
                    with torch.no_grad():
                        # 记录未乘45的mel误差，方便直观看还原差异。
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    # 输出全局步数、G总损失、未加权mel误差和每批耗时。
                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          # 将当前指标填入日志格式，耗时从本批开始计算。
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                # 按配置间隔保存，跳过尚未更新的第0步。
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    # 生成器归档名用8位步数，便于按名称排序。
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    # 将对应模型或训练状态写到刚生成的归档路径。
                    save_checkpoint(checkpoint_path,
                                    # 多卡先取内部G，再保存纯生成器权重。
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    # 另建同一步数的D及优化器状态文件。
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    # 将对应模型或训练状态写到刚生成的归档路径。
                    save_checkpoint(checkpoint_path,
                                    # 准备保存MPD权重，多卡时先解除并行包装。
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         # 单卡直接取得MPD参数字典。
                                                         else mpd).state_dict(),
                                     # 准备保存MSD权重，兼容多卡包装。
                                     'msd': (msd.module if h.num_gpus > 1
                                                         # 单卡直接取得MSD参数字典。
                                                         else msd).state_dict(),
                                     # 同时保存G、D优化器动量和全局更新步数。
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     # 保存当前轮次，供恢复训练与学习率调度使用。
                                     'epoch': epoch})

                # Tensorboard summary logging
                # 到达TensorBoard数值记录间隔时写入曲线点。
                if steps % a.summary_interval == 0:
                    # 记录当前生成器加权总损失。
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    # 记录未加权mel误差，与总损失区分观察。
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                # 按上游验证间隔运行完整验证集，仅提供诊断。
                if steps % a.validation_interval == 0:  # and steps != 0:
                    # 验证期间将生成器切为推理模式。
                    generator.eval()
                    # 释放未使用的CUDA缓存块，为完整验证语音腾出空间。
                    torch.cuda.empty_cache()
                    # 从零累计所有验证语音的mel误差。
                    val_err_tot = 0
                    # 诊断数值和验证不构建梯度图，避免额外显存占用。
                    with torch.no_grad():
                        # 按固定顺序遍历验证语音。
                        for j, batch in enumerate(validation_loader):
                            # x为条件mel，y为真实波形，y_mel为重建损失的真实mel。
                            x, y, _, y_mel = batch
                            # 用当前生成器还原这条验证mel。
                            y_g_hat = generator(x.to(device))
                            # 把真实损失mel传到GPU，供重建误差比较。
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            # 为验证合成波形计算重建mel。
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          # 验证沿用训练的帧移和分析窗长度。
                                                          h.hop_size, h.win_size,
                                                          # 重建比较使用配置的损失频率范围。
                                                          h.fmin, h.fmax_for_loss)
                            # 将当前语音未加权mel误差加入验证总和。
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            # 仅保存前5条验证语音及频谱作为诊断样本。
                            if j <= 4:
                                # 首次验证时保存真实语音和输入频谱参考。
                                if steps == 0:
                                    # 把真实波形写入TensorBoard作为对照，未进行人工打分。
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    # 记录真实输入mel的频谱图。
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                # 记录当前合成音频，方便复查合成是否异常。
                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                # 另提取用于可视化的生成mel。
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             # 可视化频谱使用当前音频采样率和时间分析设置。
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             # 可视化使用条件mel的频率上下限。
                                                             h.fmin, h.fmax)
                                # 把生成频谱作为图像写入日志。
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              # 去除batch轴并转CPU数组，再绘制当前步的生成频谱。
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        # 按验证语音条数平均mel误差。
                        val_err = val_err_tot / (j+1)
                        # 保存验证mel误差随训练步数的变化。
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    # 将G恢复到训练模式，继续参数更新。
                    generator.train()

            # 本批G与D更新完成后，全局步数增加1。
            steps += 1

        # 完整一轮后按配置衰减G学习率。
        scheduler_g.step()
        # 同一轮边界衰减D学习率。
        scheduler_d.step()

        # 仅主进程写日志、保存模型及评估，避免多卡重复写同一文件。
        if rank == 0:
            # 打印本轮总耗时，供估计剩余训练时间。
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


# 解析上游训练参数，加载配置并启动单卡或多卡训练。
def main():
    # 提示进入上游HiFi-GAN训练初始化。
    print('Initializing Training Process..')

    # 建立命令行参数解析器。
    parser = argparse.ArgumentParser()

    # 保留上游分布式分组参数接口。
    parser.add_argument('--group_name', default=None)
    # 指定原始WAV目录，默认值属于上游示例。
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    # 指定微调路径读取的预计算mel目录。
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    # 指定上游训练语音清单。
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    # 指定上游验证语音清单，区别于项目自训manifest协议。
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    # 指定本入口模型及训练状态的保存目录。
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    # 指定网络结构和训练超参数JSON。
    parser.add_argument('--config', default='')
    # 上游默认轮数为3100；这不代表FastVoiceGrad论文给出了声码器轮数。
    parser.add_argument('--training_epochs', default=3100, type=int)
    # 默认每5步打印训练进度。
    parser.add_argument('--stdout_interval', default=5, type=int)
    # 默认每5000步保存检查点。
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    # 默认每100步记录TensorBoard数值。
    parser.add_argument('--summary_interval', default=100, type=int)
    # 默认每1000步运行上游验证流程。
    parser.add_argument('--validation_interval', default=1000, type=int)
    # 保留上游微调开关，控制是否读取预计算条件mel。
    parser.add_argument('--fine_tuning', default=False, type=bool)

    # 将终端参数整理成可供训练函数读取的对象。
    a = parser.parse_args()

    # 打开用户提供的网络及优化配置文件。
    with open(a.config) as f:
        # 读取完整JSON文本。
        data = f.read()

    # 将JSON文本解析为Python字典。
    json_config = json.loads(data)
    # 给配置增加点号访问形式以匹配上游网络代码。
    h = AttrDict(json_config)
    # 在输出目录保存配置快照，使权重可追溯。
    build_env(a.config, 'config.json', a.checkpoint_path)

    # 固定CPU模型初始化随机种子。
    torch.manual_seed(h.seed)
    # CUDA可用时确定本次可见的GPU数量。
    if torch.cuda.is_available():
        # 固定本GPU随机种子，便于复现实验初始化。
        torch.cuda.manual_seed(h.seed)
        # 只统计当前进程可见的GPU设备。
        h.num_gpus = torch.cuda.device_count()
        # 把配置全局batch均分给各GPU，沿用上游整数取整行为。
        h.batch_size = int(h.batch_size / h.num_gpus)
        # 显示每张GPU实际处理的语音条数。
        print('Batch size per GPU :', h.batch_size)
    # 没有CUDA时保持上游空分支，后续训练仍要求GPU。
    else:
        # 保留上游在该分支不执行额外操作的行为。
        pass

    # 配置为多卡时建立通信或同步包装，单卡跳过相应分支。
    if h.num_gpus > 1:
        # 为每张可见GPU启动train进程并传入相同配置。
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    # 没有CUDA时保持上游空分支，后续训练仍要求GPU。
    else:
        # 单卡时直接在GPU0执行上游训练函数。
        train(0, a, h)


# 直接运行脚本时才启动训练，import不会自动开跑。
if __name__ == '__main__':
    # 进入本文件命令行训练入口。
    main()
