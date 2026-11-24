# 验证学生训练的冻结边界、串行省显存计算、测试损失和断点契约。
"""验证学生训练的冻结边界、串行省显存计算、测试损失和断点契约。"""

# ``copy.deepcopy`` 用来复制完全相同的判别器，以公平比较两种反向传播路径。
import copy
# json 把测试构造的声码器证明写成与正式文件相同的 UTF-8 JSON。
import json  # 用标准库生成和读取测试用声码器证明 JSON，不接触真实权重。
# Python 标准单元测试框架，负责发现以 ``test_`` 开头的方法并报告断言结果。
import unittest
# defaultdict 让每个尚未出现的损失累计项自动从 0.0 开始。
from collections import defaultdict
# Path 提供跨平台的临时 CSV 和检查点路径操作。
from pathlib import Path
# TemporaryDirectory 为文件类测试创建自动清理的隔离目录，不污染真实实验产物。
from tempfile import TemporaryDirectory

# PyTorch 提供张量、随机数、自动求导和 Adam 优化器。
import torch
# ``nn`` 提供最小卷积层和测试替身模型的基类。
import torch.nn as nn

# 导入梅尔反规范化与冻结/解冻工具，测试梯度是否只流向正确模型。
from fastvoicegrad.components import (  # 导入声码器证明迁移校验和既有梯度工具。
    canonical_sha256,  # 按正式规则为测试证明负载计算稳定 SHA-256。
    denormalize_mel,  # 把标准化 mel 还原到声码器输入尺度。
    set_requires_grad,  # 切换测试模型参数是否接收梯度。
    validate_final_vocoder_attestation,  # 验证证明可随项目根迁移但不能改变文件角色。
)  # 完成组件工具导入。
# 导入全工程唯一的论文/复现冻结配置，断言测试与正式训练采用相同数值。
from fastvoicegrad.config import PAPER_STUDENT_CONFIG
# 导入三种核心损失，用来建立“完整一次计算”的数学参照。
from fastvoicegrad.objectives import (
    feature_matching_loss,  # 比较真实/生成波形在判别器中间层的差异。
    least_squares_discriminator_loss,  # 判别器的最小二乘 GAN 目标。
    least_squares_student_loss,  # 学生让假样本得分接近真的最小二乘目标。
)
# 导入训练契约相关函数；这些都是本文件需要逐项验证的对外行为。
from fastvoicegrad.train import (
    FixedSegmentTestDataset,  # 把测试语音固定裁成相同片段，保证各轮可比。
    LOSS_COMPONENTS,  # 训练器规定的九个损失分量名称，其中两项属于 Cycle。
    add_sample_weighted_losses,  # 按样本数而不是按批数累计测试损失。
    append_loss_csv,  # 原子追加一轮完整的训练/测试损失记录。
    evaluate_test_epoch,  # 无梯度评估 100 条 text-unseen 测试语音。
    load_resume_checkpoint,  # 恢复学生、判别器、优化器和训练游标。
    make_epoch_loss_record,  # 合并一轮训练与测试分量及覆盖计数。
    reconcile_loss_csv,  # 断点恢复时修复最多一条尚未随检查点保存的 CSV 尾行。
    save_checkpoint,  # 保存模型和可选的完整训练状态。
    serial_discriminator_backward,  # 逐判别器分支反传，降低峰值显存。
    serial_student_discriminator_gradient,  # 逐分支求学生波形梯度，保持数学等价。
    train_batch,  # 执行一次判别器更新和一次学生更新。
)
# 正式 VoiceGrad 扩散实现用于测试学生蒸馏流程，不另造一套近似公式。
from voicegrad.diffusion import VoiceGradDiffusion


# 极小声码器只做 80 通道梅尔到 1 通道波形的线性投影，适合快速检查梯度边界。
class TinyVocoder(nn.Module):
    # 保留真实声码器“梅尔进、波形出”接口的最小可微替身。
    """保留真实声码器“梅尔进、波形出”接口的最小可微替身。"""

    # 创建一个无偏置的 1×1 卷积，参数很少但仍能验证冻结行为。
    def __init__(self):
        # 创建一个无偏置的 1×1 卷积，参数很少但仍能验证冻结行为。
        """创建一个无偏置的 1×1 卷积，参数很少但仍能验证冻结行为。"""

        # 初始化 nn.Module 内部状态，所有子模块都必须在这句之后登记。
        super().__init__()
        # 80 个梅尔通道线性合成为 1 个波形通道，不改变时间长度。
        self.projection = nn.Conv1d(80, 1, kernel_size=1, bias=False)

    # 把输入梅尔张量送进线性投影并返回测试波形。
    def forward(self, mel):
        # 把输入梅尔张量送进线性投影并返回测试波形。
        """把输入梅尔张量送进线性投影并返回测试波形。"""

        # 保留完整计算图，因此即使卷积参数冻结，梯度仍应能传回输入 mel。
        return self.projection(mel)


# 极小 epsilon 网络用一个可选可训练标量缩放 tanh(x)，模拟教师或学生噪声预测器。
class TinyEpsilon(nn.Module):
    # 用单个标量参数模拟 VoiceGrad epsilon 网络的最小替身。
    """用单个标量参数模拟 VoiceGrad epsilon 网络的最小替身。"""

    # ``trainable`` 决定这个标量属于可训练学生还是冻结教师。
    def __init__(self, trainable=True):
        # ``trainable`` 决定这个标量属于可训练学生还是冻结教师。
        """``trainable`` 决定这个标量属于可训练学生还是冻结教师。"""

        # 初始化模块基类，确保参数能被 parameters() 枚举。
        super().__init__()
        # 从 0.01 开始的标量足以产生非零梯度；requires_grad 直接反映冻结状态。
        self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=trainable)

    # 接受正式模型同样的四个参数，只用 x 计算形状一致的 epsilon。
    def forward(self, x, timesteps, speaker_embedding, bnf):
        # 接受正式模型同样的四个参数，只用 x 计算形状一致的 epsilon。
        """接受正式模型同样的四个参数，只用 x 计算形状一致的 epsilon。"""

        # tanh 限制数值范围，再乘可训练标量；其余条件输入保留在签名中以验证接口兼容。
        return self.scale * torch.tanh(x)


# 一个判别器分支产生最终分数和两个中间特征图，以覆盖对抗损失与特征匹配损失。
class TinyDiscriminatorBranch(nn.Module):
    # 用一个标量构造可微的判别器分数及两层特征图。
    """用一个标量构造可微的判别器分数及两层特征图。"""

    # 创建初值为 0.5 的唯一可训练判别器参数。
    def __init__(self):
        # 创建初值为 0.5 的唯一可训练判别器参数。
        """创建初值为 0.5 的唯一可训练判别器参数。"""

        # 初始化模块基类。
        super().__init__()
        # 可训练标量代表整个小判别器，更新后数值变化可直接用于断言。
        self.scale = nn.Parameter(torch.tensor(0.5))

    # 返回每条语音的判别分数，以及用于特征匹配的两层特征。
    def forward(self, waveform):
        # 返回每条语音的判别分数，以及用于特征匹配的两层特征。
        """返回每条语音的判别分数，以及用于特征匹配的两层特征。"""

        # 第一层特征是波形乘判别器标量，保留对波形和参数的梯度。
        first = self.scale * waveform
        # 第二层通过 tanh 加入非线性，使测试不只比较简单线性公式。
        second = torch.tanh(first)
        # 沿时间维求均值得到每个样本/通道的真假分数。
        score = second.mean(dim=-1)
        # 返回最终分数和按层排列的特征列表，接口与真实 MPD/MRD 分支一致。
        return score, [first, second]


# 单分支包装器用于完整 train_batch 和冻结边界测试。
class TinyDiscriminator(nn.Module):
    # 模拟正式判别器的聚合接口，但内部只有一个小分支。
    """模拟正式判别器的聚合接口，但内部只有一个小分支。"""

    # 登记唯一判别器分支。
    def __init__(self):
        # 登记唯一判别器分支。
        """登记唯一判别器分支。"""

        # 初始化模块基类。
        super().__init__()
        # 把小分支作为子模块，使优化器能找到其 scale 参数。
        self.branch = TinyDiscriminatorBranch()

    # 将判别器分支的缩放参数暴露为只读属性入口，方便测试核对。
    @property
    # 直接暴露分支标量，便于测试比较更新前后的数值。
    def scale(self):
        # 直接暴露分支标量，便于测试比较更新前后的数值。
        """直接暴露分支标量，便于测试比较更新前后的数值。"""

        # 返回同一个 Parameter，不复制数据，因此能真实反映优化器更新结果。
        return self.branch.scale

    # 按正式接口返回可逐个执行的判别器分支序列。
    def loss_branches(self):
        # 按正式接口返回可逐个执行的判别器分支序列。
        """按正式接口返回可逐个执行的判别器分支序列。"""

        # 即使只有一个元素也返回元组，串行省显存函数可以统一迭代。
        return (self.branch,)

    # 同时计算真实与生成波形的分数和中间特征，并包装成分支列表。
    def forward(self, real_waveform, fake_waveform):
        # 同时计算真实与生成波形的分数和中间特征，并包装成分支列表。
        """同时计算真实与生成波形的分数和中间特征，并包装成分支列表。"""

        # 用同一套参数处理真实波形，这是判别器真实样本路径。
        real_score, real_features = self.branch(real_waveform)
        # 用同一套参数处理生成波形，这是判别器假样本路径。
        fake_score, fake_features = self.branch(fake_waveform)
        # 外层列表表示“所有分支”，与正式 MPD/MRD 聚合输出结构一致。
        return [real_score], [fake_score], [real_features], [fake_features]


# 双分支包装器专门验证“逐分支计算”与“一次计算所有分支”数值和梯度完全一致。
class TwoBranchTinyDiscriminator(nn.Module):
    # 包含两个参数初值不同的小分支，避免等价性测试因完全对称而过于容易。
    """包含两个参数初值不同的小分支，避免等价性测试因完全对称而过于容易。"""

    # 创建两个独立分支，并把第二个分支设成不同的负缩放。
    def __init__(self):
        # 创建两个独立分支，并把第二个分支设成不同的负缩放。
        """创建两个独立分支，并把第二个分支设成不同的负缩放。"""

        # 初始化模块基类。
        super().__init__()
        # ModuleList 会登记两个分支参数，同时允许像普通列表一样顺序迭代。
        self.branches = nn.ModuleList(
            # 建立两个独立的小判别分支，以便比较串行与整网反传。
            [TinyDiscriminatorBranch(), TinyDiscriminatorBranch()]
        )
        # 改初值只是构造测试夹具，不应被 autograd 记录到任何计算图中。
        with torch.no_grad():
            # 第二分支设为 -0.3，使两个分支的输出、损失和梯度确实不同。
            self.branches[1].scale.fill_(-0.3)

    # 返回不可变元组，让串行函数依次计算两个真实独立分支。
    def loss_branches(self):
        # 返回不可变元组，让串行函数依次计算两个真实独立分支。
        """返回不可变元组，让串行函数依次计算两个真实独立分支。"""

        # tuple 只改变容器形式，不复制其中的模块或参数。
        return tuple(self.branches)

    # 一次性计算全部分支，作为串行实现的数学基准。
    def forward(self, real_waveform, fake_waveform):
        # 一次性计算全部分支，作为串行实现的数学基准。
        """一次性计算全部分支，作为串行实现的数学基准。"""

        # 分别收集每个分支的真实样本最终分数。
        real_scores = []
        # 分别收集每个分支的生成样本最终分数。
        fake_scores = []
        # 分别收集每个分支的真实样本中间特征列表。
        real_features = []
        # 分别收集每个分支的生成样本中间特征列表。
        fake_features = []
        # 两个分支顺序执行，输出顺序必须与 loss_branches() 一致。
        for branch in self.branches:
            # 当前分支处理真实波形。
            real_score, real_maps = branch(real_waveform)
            # 当前分支处理生成波形。
            fake_score, fake_maps = branch(fake_waveform)
            # 保存当前真实分数，稍后统一送入完整损失函数。
            real_scores.append(real_score)
            # 保存当前假分数。
            fake_scores.append(fake_score)
            # 保存当前真实中间特征。
            real_features.append(real_maps)
            # 保存当前假中间特征。
            fake_features.append(fake_maps)
        # 返回与正式判别器完全相同的四列表结构。
        return real_scores, fake_scores, real_features, fake_features


# 逐项锁定学生训练代码不能悄悄改变的外部行为。
class FastVoiceGradStudentContractTests(unittest.TestCase):
    # 逐项锁定学生训练代码不能悄悄改变的外部行为。
    """逐项锁定学生训练代码不能悄悄改变的外部行为。"""

    # 该辅助函数不读取实例状态，所以声明为 staticmethod，测试可直接生成假损失记录。
    @staticmethod
    # 构造九个彼此容易辨认、可按 scale 放大的训练/测试损失。
    def _losses(scale=1.0):
        # 构造九个彼此容易辨认、可按 scale 放大的训练/测试损失。
        """构造九个彼此容易辨认、可按 scale 放大的训练/测试损失。"""

        # 每个字段使用不同基数，若列顺序或映射错误，断言能立刻发现。
        components = {
            "student_total": 1.0 * scale,  # 学生总损失测试值。
            "student_adv": 2.0 * scale,  # 学生对抗项测试值。
            "fm_raw": 3.0 * scale,  # 原始特征匹配项测试值。
            "fm_weighted": 6.0 * scale,  # 计权特征匹配项，保持 2×fm_raw。
            "dist_raw": 4.0 * scale,  # 原始蒸馏项测试值。
            "dist_weighted": 180.0 * scale,  # 计权蒸馏项，保持 45×dist_raw。
            "cycle_raw": 7.0 * scale,  # 原始 masked Cycle 项测试值。
            "cycle_weighted": 7.0 * scale,  # lambda_cycle=1 时计权 Cycle 项与原始值相同。
            "discriminator": 5.0 * scale,  # 判别器诊断项测试值。
        }
        # 测试侧复用同一组分量，并登记完整覆盖 100 个样本、4 个批次。
        return make_epoch_loss_record(components, components, 100, 4)

    # 声码器证明应允许项目根从旧 /root 迁到 NFS，但必须保持末四级文件角色不变。
    def test_vocoder_attestation_allows_only_root_relocation(self):  # 用最小证明负载覆盖迁移成功和角色错误两条路径。
        """项目根可以迁移，声码器实验目录和文件名不能改变。"""  # SHA 与稳定字段仍按正式函数逐项核对。

        stable = {  # 构造正式校验会逐项比较的全部稳定字段。
            "status": "verified",  # 表示历史证明已完成验证。
            "checkpoint_size_bytes": 123,  # 使用容易识别的测试文件大小。
            "checkpoint_sha256": "a" * 64,  # 使用固定测试权重哈希。
            "config_file_sha256": "b" * 64,  # 使用固定配置文件哈希。
            "config_canonical_sha256": "c" * 64,  # 使用固定规范化配置哈希。
            "format_version": 2,  # 沿用正式声码器权重格式版本。
            "checkpoint_type": "generator_weights_only",  # 沿用冻结生成器归档类型。
            "architecture": "test-architecture",  # 架构值只需在历史与现场报告一致。
            "global_step": 2_500_000,  # 使用正式最终训练步数。
            "epoch": 1015,  # 使用证明中的正式轮次示例。
            "batch_in_epoch": 2085,  # 使用证明中的批位置示例。
            "expected_filename": "generator_step_2500000.pt",  # 锁定权重文件角色。
            "data_fingerprints": {"manifest": "d" * 64},  # 用最小映射代表已锁定数据指纹。
            "data_fingerprints_canonical_sha256": "e" * 64,  # 锁定数据指纹集合摘要。
            "generator_strict_load": True,  # 表示生成器权重严格加载成功。
            "generator": {"finite": True},  # 用最小映射代表生成器数值报告。
            "implementation_sha256": {"models.py": "f" * 64},  # 锁定实现文件摘要。
        }  # 完成稳定字段集合。
        attestation = {  # 历史证明仍保留旧服务器绝对路径。
            **stable,  # 复制全部不可变稳定字段。
            "checkpoint": "/root/FsatVoiceGrad/exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/generator_step_2500000.pt",  # 旧权重路径。
            "config": "/root/FsatVoiceGrad/exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/config.json",  # 旧配置路径。
        }  # 完成尚未加入自哈希的证明负载。
        attestation["attestation_payload_sha256"] = canonical_sha256(attestation)  # 按正式规则锁定含旧路径的证明内容。
        with TemporaryDirectory() as directory:  # 临时目录在测试结束后自动清理。
            attestation_path = Path(directory) / "attestation.json"  # 定位临时证明文件。
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")  # 写出正式加载器可读的 JSON。
            relocated_root = Path(directory) / "nfs-copy"  # 模拟恢复到另一项目根目录。
            relocated_checkpoint = relocated_root / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/generator_step_2500000.pt"  # 保持权重角色后缀。
            relocated_config = relocated_root / "exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen/config.json"  # 保持配置角色后缀。
            loaded = validate_final_vocoder_attestation(  # 只变更项目根时校验应成功。
                attestation_path,  # 传入含旧绝对路径且自哈希正确的证明。
                relocated_checkpoint,  # 传入新 NFS 根下同角色权重路径。
                relocated_config,  # 传入新 NFS 根下同角色配置路径。
                stable,  # 现场报告的稳定字段与证明完全一致。
            )  # 完成允许迁移的验证。
            self.assertEqual(loaded["checkpoint_sha256"], "a" * 64)  # 返回证明仍保留原权重哈希。
            wrong_checkpoint = relocated_root / "exp/checkpoints/other_experiment/generator_step_2500000.pt"  # 故意改变实验目录角色。
            with self.assertRaises(RuntimeError):  # 仅文件名相同但目录角色不同必须失败。
                validate_final_vocoder_attestation(  # 调用同一正式校验入口。
                    attestation_path,  # 使用未修改且自哈希正确的证明。
                    wrong_checkpoint,  # 传入错误实验目录中的同名权重。
                    relocated_config,  # 配置路径仍保持合法，以隔离权重角色错误。
                    stable,  # 其他稳定字段继续完全一致。
                )  # 正确实现应在这里抛出 RuntimeError。

    # 声码器参数冻结后，生成波形损失仍必须能把梯度传回学生的假梅尔。
    def test_frozen_vocoder_still_backpropagates_to_fake_mel(self):
        # 声码器参数冻结后，生成波形损失仍必须能把梯度传回学生的假梅尔。
        """声码器参数冻结后，生成波形损失仍必须能把梯度传回学生的假梅尔。"""

        # 创建带可微卷积的最小声码器。
        vocoder = TinyVocoder()
        # 关闭声码器所有参数梯度，模拟正式训练中的冻结声码器。
        set_requires_grad(vocoder, False)
        # 假梅尔设 requires_grad=True，代表学生模型输出端需要收到反向梯度。
        normalized = torch.randn(1, 80, 4, requires_grad=True)
        # 使用全 0 均值，简化反规范化但保留真实函数调用。
        mean = torch.zeros(1, 80, 1)
        # 使用全 1 标准差，反规范化后数值仍等于输入，便于隔离梯度问题。
        std = torch.ones(1, 80, 1)
        # 先反规范化梅尔，再通过冻结声码器得到生成波形。
        fake_wave = vocoder(denormalize_mel(normalized, mean, std))
        # 对波形均方值反传，验证计算图能穿过冻结参数回到输入。
        fake_wave.square().mean().backward()
        # 假梅尔必须收到梯度对象，否则学生无法从波形损失学习。
        self.assertIsNotNone(normalized.grad)
        # 梯度绝对值总和必须大于 0，排除“有 grad 字段但全是零”的假通过。
        self.assertGreater(float(normalized.grad.abs().sum()), 0.0)
        # 声码器每个参数都必须没有梯度，证明冻结边界没有被破坏。
        self.assertTrue(all(parameter.grad is None for parameter in vocoder.parameters()))

    # 逐分支省显存算法必须与一次保留全部分支得到相同损失和假波形梯度。
    def test_serial_student_losses_and_waveform_gradient_match_full_forward(self):
        # 逐分支省显存算法必须与一次保留全部分支得到相同损失和假波形梯度。
        """逐分支省显存算法必须与一次保留全部分支得到相同损失和假波形梯度。"""

        # 固定随机种子，使失败可以在任何机器上用同一输入重现。
        torch.manual_seed(41)
        # 使用两个不同分支，并切到 eval 以排除训练态随机层影响。
        discriminator = TwoBranchTinyDiscriminator().eval()
        # 学生更新阶段判别器只提供梯度路径，其参数自身必须冻结。
        set_requires_grad(discriminator, False)
        # 创建三条长度 11 的真实波形作为特征匹配参照。
        real_waveform = torch.randn(3, 1, 11)
        # 完整算法使用独立假波形，并要求它保留梯度。
        fake_full = torch.randn(3, 1, 11, requires_grad=True)
        # 一次性跑全部分支，得到论文损失原实现所需的四类输出。
        _, fake_scores, real_features, fake_features = discriminator(
            # 真实波形和带梯度的生成波形同时送入完整判别器。
            real_waveform, fake_full
        )
        # 按所有分支之和计算完整学生对抗损失。
        full_adversarial = least_squares_student_loss(fake_scores)
        # 按所有层/分支之和计算完整原始特征匹配损失。
        full_fm = feature_matching_loss(real_features, fake_features)
        # 对“对抗 + 2×特征匹配”一次性求假波形梯度，作为数学金标准。
        (full_gradient,) = torch.autograd.grad(
            # 对计入 2 倍特征匹配的完整学生目标求生成波形梯度。
            full_adversarial + 2.0 * full_fm, fake_full
        )
        # 用省显存串行实现处理相同数值的假波形，并取回两项损失与累计梯度。
        serial_adversarial, serial_fm, serial_gradient = (
            # 调用省显存的逐分支学生梯度实现，与完整图结果比较。
            serial_student_discriminator_gradient(
                discriminator,  # 同一个冻结双分支判别器。
                real_waveform,  # 与完整路径完全相同的真实波形。
                fake_full.detach(),  # 只共享数值，不沿用完整路径已经建立的计算图。
                lambda_fm=2.0,  # 严格使用论文公式（14）的特征匹配权重。
            )
        )
        # 串行累计的对抗损失必须与完整前向逐元素容差内一致。
        torch.testing.assert_close(serial_adversarial, full_adversarial.detach())
        # 串行累计的原始特征匹配损失也必须一致，不能漏掉或重复任何层。
        torch.testing.assert_close(serial_fm, full_fm.detach())
        # 最关键的是对假波形的总梯度必须一致，才能证明省显存没有改变学生更新。
        torch.testing.assert_close(serial_gradient, full_gradient)
        # 判别器处于冻结状态，所以比较过程中它的任何参数都不能积累梯度。
        self.assertTrue(
            # 所有判别器参数都应没有梯度，学生步骤不能顺带训练判别器。
            all(parameter.grad is None for parameter in discriminator.parameters())
        )

    # 判别器逐分支反传必须等于完整分支总损失一次反传。
    def test_serial_discriminator_backward_matches_summed_full_gradient(self):
        # 判别器逐分支反传必须等于完整分支总损失一次反传。
        """判别器逐分支反传必须等于完整分支总损失一次反传。"""

        # 固定输入和初始参数的随机来源，使梯度比较稳定可复现。
        torch.manual_seed(77)
        # ``full`` 用于一次性完整计算，并保持训练模式。
        full = TwoBranchTinyDiscriminator().train()
        # 深复制得到参数完全相同但梯度存储独立的 ``serial`` 判别器。
        serial = copy.deepcopy(full).train()
        # 创建三条真实波形。
        real_waveform = torch.randn(3, 1, 11)
        # 创建三条假波形；判别器更新不需要把梯度传回生成器，所以默认不求输入梯度。
        fake_waveform = torch.randn(3, 1, 11)
        # 一次前向得到所有分支的真实和假分数，中间特征在判别器损失中不用。
        real_scores, fake_scores, _, _ = full(real_waveform, fake_waveform)
        # 计算全部分支相加的完整最小二乘判别器损失。
        full_loss = least_squares_discriminator_loss(real_scores, fake_scores)
        # 一次反传让 full 的各参数获得基准梯度。
        full_loss.backward()
        # 串行函数逐个分支反传并累计参数梯度，同时返回累计损失。
        serial_loss = serial_discriminator_backward(
            # 给串行实现传入相同判别器参数和同一对真假波形。
            serial, real_waveform, fake_waveform
        )
        # 串行损失必须与完整损失数值一致。
        torch.testing.assert_close(serial_loss, full_loss.detach())
        # 按相同参数登记顺序逐个比较两个实现产生的梯度。
        for full_parameter, serial_parameter in zip(
            # 逐个配对完整反传和串行反传的对应参数，比较梯度。
            full.parameters(), serial.parameters()
        ):
            # 任一分支漏算、权重不对或重复反传都会让这里失败。
            torch.testing.assert_close(serial_parameter.grad, full_parameter.grad)

    # 论文未公开的实现选择绝不能被运行契约错误标成论文明确设定。
    def test_unpublished_values_are_not_labelled_paper_explicit(self):
        # 论文未公开的实现选择绝不能被运行契约错误标成论文明确设定。
        """论文未公开的实现选择绝不能被运行契约错误标成论文明确设定。"""

        # 取得只包含论文明确设定的字段映射。
        paper = PAPER_STUDENT_CONFIG.paper_explicit()
        # 取得明确披露的复现选择字段映射。
        choices = PAPER_STUDENT_CONFIG.declared_reproduction_choices()
        # 挑选涵盖数据、随机性、权重身份、判别器和测试协议的代表字段逐个检查。
        for key in (
            "train_frames",  # 训练裁剪长度不是论文公开值。
            "seed",  # 总体随机种子不是论文公开值。
            "expected_teacher_checkpoint_sha256",  # 本次教师文件指纹属于复现实验身份。
            "expected_vocoder_step",  # 250 万声码器步数是本次实现锁定值。
            "discriminator_initialization",  # 判别器从零训练的细节论文未写明。
            "score_t_sampling",  # 蒸馏时刻的精确抽样法论文未写明。
            "test_loss_crop_seed",  # 为用户曲线固定的测试裁剪种子。
            "test_loss_checkpoint_selection",  # 禁止按测试损失选模型的本次协议。
        ):
            # 这些键不得出现在 paper 映射中，否则报告会错误归因给论文。
            self.assertNotIn(key, paper)
            # 同一键必须出现在 choices 映射中，确保我们没有隐瞒自行设定的细节。
            self.assertIn(key, choices)

    # 100 条测试语音的最后 4 条必须按样本加权，不能和 32 条的大批拥有相同权重。
    def test_short_final_test_batch_is_sample_weighted(self):
        # 100 条测试语音的最后 4 条必须按样本加权，不能和 32 条的大批拥有相同权重。
        """100 条测试语音的最后 4 条必须按样本加权，不能和 32 条的大批拥有相同权重。"""

        # 每个未出现的字段从浮点 0 开始累计。
        totals = defaultdict(float)
        # 前三批的每个损失分量均设为 1，代表普通 32 样本批。
        ordinary = {key: 1.0 for key in LOSS_COMPONENTS}
        # 最后一批的每个损失分量设为 9，使错误“按批平均”会非常明显。
        residual = {key: 9.0 for key in LOSS_COMPONENTS}
        # 连续累计三批各 32 条，共 96 条普通样本。
        for _ in range(3):
            # 每个批均值乘 32 后加入总和，保留真实样本权重。
            add_sample_weighted_losses(totals, ordinary, 32)
        # 最后一批只有 4 条，所以高损失 9 只应乘 4，而不是被当成完整 32 条。
        add_sample_weighted_losses(totals, residual, 4)
        # 对九个损失分量分别验证同一加权规则。
        for key in LOSS_COMPONENTS:
            # 正确样本均值为 (96×1 + 4×9)/100 = 1.32。
            self.assertAlmostEqual(totals[key] / 100, 1.32)
            # 明确拒绝错误的四批等权均值 (1+1+1+9)/4 = 3.0。
            self.assertNotAlmostEqual(totals[key] / 100, (1 + 1 + 1 + 9) / 4)

    # 固定测试裁剪对同一语音必须每次选同一起点，并保持 mel/BNF 时间对齐。
    def test_fixed_test_crop_is_identity_deterministic(self):
        # 固定测试裁剪对同一语音必须每次选同一起点，并保持 mel/BNF 时间对齐。
        """固定测试裁剪对同一语音必须每次选同一起点，并保持 mel/BNF 时间对齐。"""

        # 内嵌最小基础数据集，只提供一条长于 128 帧且数值位置可追踪的语音。
        class Base:
            # 模拟原始测试数据集接口的单样本夹具。
            """模拟原始测试数据集接口的单样本夹具。"""

            # 基础数据集只含一条语音。
            def __len__(self):
                # 基础数据集只含一条语音。
                """基础数据集只含一条语音。"""

                # 返回 1 让包装数据集知道唯一合法索引是 0。
                return 1

            # 忽略传入索引并返回一条 200 帧、字段完整的测试样本。
            def __getitem__(self, _index):
                # 忽略传入索引并返回一条 200 帧、字段完整的测试样本。
                """忽略传入索引并返回一条 200 帧、字段完整的测试样本。"""

                # 返回字段名和形状与真实 VoiceGradDataset 一致的字典。
                return {
                    # 使用连续整数填充 80×200 梅尔，便于检测两次裁剪内容是否逐元素相同。
                    "mel": torch.arange(80 * 200, dtype=torch.float32).view(80, 200),
                    # BNF 同样用连续整数填充 144×200，并与梅尔共享时间长度。
                    "bnf": torch.arange(144 * 200, dtype=torch.float32).view(144, 200),
                    # 256 维全零说话人嵌入满足正式数据接口，本测试不关心其数值。
                    "spk_emb": torch.zeros(256),
                    # 原始有效梅尔长度明确登记为 200 帧。
                    "mel_len": torch.tensor(200),
                    # 提供真实风格的说话人名称，使固定裁剪哈希可以包含样本身份。
                    "spk_name": "p225",
                    # 提供真实风格的语句 ID，确保裁剪由身份而非访问顺序决定。
                    "utt_id": "p225_001",
                }

        # 绕过正式 __init__，只注入测试所需三个属性，避免访问服务器数据文件。
        dataset = FixedSegmentTestDataset.__new__(FixedSegmentTestDataset)
        # 底层样本来源设为上面的单样本 Base。
        dataset.base = Base()
        # 固定目标长度 128，与正式学生训练和测试配置相同。
        dataset.segment_length = 128
        # 固定裁剪种子为 1234，使身份哈希结果可重复。
        dataset.crop_seed = 1234
        # 该夹具验证基线固定裁剪，不为样本建立额外的 Cycle 目标说话人。
        dataset.include_cycle_reference = False  # 与正式基线数据集的关闭状态保持一致。
        # 关闭 Cycle 时固定目标索引应为空，避免夹具绕过初始化后缺少新增属性。
        dataset.cycle_reference_indices = ()  # 空元组表示没有任何 A→B 测试配对。
        # 当前读取不需要 Cycle 字段，显式关闭可保持 dataset[0] 的基线返回结构。
        dataset.cycle_reference_active = False  # 防止 __getitem__ 访问未初始化属性。
        # 第一次读取索引 0，得到固定裁剪结果。
        first = dataset[0]
        # 立刻第二次读取同一索引，验证访问次数不会改变裁剪。
        second = dataset[0]
        # 梅尔应保持 80 通道并严格裁成 128 帧。
        self.assertEqual(tuple(first["mel"].shape), (80, 128))
        # BNF 应保持 144 通道并与梅尔严格裁成同样的 128 帧。
        self.assertEqual(tuple(first["bnf"].shape), (144, 128))
        # 两次得到的梅尔每个元素都必须完全相等。
        self.assertTrue(torch.equal(first["mel"], second["mel"]))
        # 两次得到的 BNF 每个元素也必须完全相等。
        self.assertTrue(torch.equal(first["bnf"], second["bnf"]))
        # 记录在样本中的裁剪起点必须相同，便于将来审计具体截取位置。
        self.assertEqual(first["fixed_crop_start"], second["fixed_crop_start"])

    # 测试评估必须覆盖 100 条、可重复、不改变训练随机状态/模式/参数/梯度。
    def test_test_loss_covers_32_32_32_4_and_is_rng_neutral(self):
        # 测试评估必须覆盖 100 条、可重复、不改变训练随机状态/模式/参数/梯度。
        """测试评估必须覆盖 100 条、可重复、不改变训练随机状态/模式/参数/梯度。"""

        # 使用正式冻结配置，避免测试自己的另一套假设。
        config = PAPER_STUDENT_CONFIG
        # 明确构造 100 条在批量 32 下的真实分批形状：32+32+32+4。
        batch_sizes = (32, 32, 32, 4)
        # 用局部 Generator 生成夹具，不消耗全局训练随机序列。
        generator = torch.Generator().manual_seed(771)
        # 用普通列表模拟可迭代 DataLoader，足以验证评估逻辑。
        loader = []
        # offset 给每条语音生成全局唯一且连续的身份编号。
        offset = 0
        # 逐个构造四批输入，最后一批故意只有 4 条。
        for batch_size in batch_sizes:
            # 每一批都包含正式评估所需的五类字段。
            loader.append(
                {
                    # 生成 [批量,80,128] 梅尔，使用局部生成器保证确定性。
                    "mel": torch.randn(
                        # 构造当前批数、80 通道、固定测试帧长的梅尔张量。
                        batch_size, 80, config.test_loss_frames, generator=generator
                    ),
                    # 生成时间长度相同的 [批量,144,128] BNF 条件。
                    "bnf": torch.randn(
                        # 构造相同帧长的 144 通道 BNF，使内容与梅尔对齐。
                        batch_size, 144, config.test_loss_frames, generator=generator
                    ),
                    # 为每条语音生成 256 维说话人嵌入。
                    "spk_emb": torch.randn(batch_size, 256, generator=generator),
                    # 每十条归到一个伪说话人，覆盖列表切片中本批所有全局索引。
                    "spk_name": [f"p{index // 10:03d}" for index in range(offset, offset + batch_size)],
                    # 每条语音取得唯一 utt ID，固定噪声可由身份稳定派生。
                    "utt_id": [f"utt_{index:03d}" for index in range(offset, offset + batch_size)],
                }
            )
            # 为下一批移动全局身份起点，最终 offset 会等于 100。
            offset += batch_size

        # 创建冻结、评估模式的小教师模型。
        teacher = TinyEpsilon(trainable=False).eval()
        # 创建可训练且处于 train 模式的小学生，评估结束后必须恢复这个模式。
        student = TinyEpsilon(trainable=True).train()
        # 创建处于 eval 模式的小声码器，测试评估不能改变其模式。
        vocoder = TinyVocoder().eval()
        # 创建处于 train 模式的小判别器，评估期间可临时切换但结束后必须恢复。
        discriminator = TinyDiscriminator().train()
        # 再显式冻结教师全部参数，锁定正式训练的教师边界。
        set_requires_grad(teacher, False)
        # 显式冻结声码器参数，但保留从波形到梅尔的梯度能力供训练使用。
        set_requires_grad(vocoder, False)
        # 创建与正式配置相同 1000 步、0.008 偏移的扩散过程。
        diffusion = VoiceGradDiffusion(config.diffusion_steps, config.cosine_offset)
        # 保存四个模型的顺序，后面统一检查参数和梯度没有被测试评估修改。
        models = (teacher, student, vocoder, discriminator)
        # 深度复制每个参数的当前数值，作为评估前基准快照。
        before_parameters = [
            # detach 断开计算图，clone 防止后续原地更新同时改变快照。
            [parameter.detach().clone() for parameter in model.parameters()]
            # 对教师、学生、声码器、判别器依次建立参数列表。
            for model in models
        ]
        # 设置一个可识别的全局 RNG 状态，检查评估是否污染后续训练随机序列。
        torch.manual_seed(991)
        # 保存调用评估之前完整的 CPU 随机数状态字节。
        before_rng = torch.get_rng_state().clone()
        # 第一次评估：应使用函数内部固定局部种子，并覆盖全部四批。
        first, examples, batches = evaluate_test_epoch(
            loader,  # 上面构造的 32+32+32+4 测试批列表。
            teacher,  # 冻结教师提供蒸馏目标。
            student,  # 被评估但绝不更新的学生。
            vocoder,  # 冻结声码器把真/假梅尔变成波形。
            discriminator,  # 判别器只计算诊断与学生波形损失。
            diffusion,  # 正式 VoiceGrad 噪声日程和正反向公式。
            torch.zeros(1, 80, 1),  # 测试反规范化均值设 0。
            torch.ones(1, 80, 1),  # 测试反规范化标准差设 1。
            torch.device("cpu"),  # 单元测试只验证逻辑，不调用租用服务器 GPU。
            config,  # 正式学生配置决定权重、种子和输入长度。
        )
        # 第一次评估后立即记录全局 RNG，必须与调用前完全相同。
        after_rng = torch.get_rng_state().clone()
        # 对完全相同输入再评估一次，验证固定噪声、时刻和裁剪产生可重复数值。
        second, second_examples, second_batches = evaluate_test_epoch(
            loader,  # 复用相同四批。
            teacher,  # 复用相同冻结教师。
            student,  # 复用未更新的学生。
            vocoder,  # 复用未更新的冻结声码器。
            discriminator,  # 复用未更新的判别器。
            diffusion,  # 复用相同扩散日程。
            torch.zeros(1, 80, 1),  # 保持反规范化均值相同。
            torch.ones(1, 80, 1),  # 保持反规范化标准差相同。
            torch.device("cpu"),  # 仍在 CPU 上做轻量逻辑测试。
            config,  # 仍用同一个冻结配置。
        )
        # 第一次必须准确报告 100 个样本、4 个批次，没有遗漏最后 4 条。
        self.assertEqual((examples, batches), (100, 4))
        # 第二次覆盖数量也必须完全相同。
        self.assertEqual((second_examples, second_batches), (100, 4))
        # 两次九个测试损失字典必须逐值完全相同，证明评估输入随机量固定。
        self.assertEqual(first, second)
        # 全局 RNG 状态逐字节相等，证明插入每轮测试不会改变后续训练噪声轨迹。
        self.assertTrue(torch.equal(before_rng, after_rng))
        # 学生进入评估前是 train 模式，结束后必须恢复 train 模式。
        self.assertTrue(student.training)
        # 判别器同样必须恢复 train 模式，保证下一轮训练行为不变。
        self.assertTrue(discriminator.training)
        # 教师原本是 eval 模式，评估结束后仍应是 eval。
        self.assertFalse(teacher.training)
        # 声码器原本也是 eval 模式，结束后保持不变。
        self.assertFalse(vocoder.training)
        # 按相同顺序把评估后的每个模型与其评估前参数快照比较。
        for model, saved in zip(models, before_parameters):
            # 模型内各参数按登记顺序与对应快照一一比较。
            for parameter, expected in zip(model.parameters(), saved):
                # 参数数值必须逐元素完全相同，证明测试损失从未执行优化器更新。
                self.assertTrue(torch.equal(parameter.detach(), expected))
                # 参数 grad 必须仍为 None，证明测试评估没有残留反向传播梯度。
                self.assertIsNone(parameter.grad)
        # 测试总损失必须严格满足论文公式（14）的三个分量组合关系。
        self.assertAlmostEqual(
            first["student_total"],  # 评估函数实际报告的学生总损失。
            # 期望值：对抗项 + 2×原始特征匹配项 + 45×原始蒸馏项。
            first["student_adv"]
            # 把原始特征匹配损失乘以配置的权重后加到总量中。
            + config.lambda_fm * first["fm_raw"]
            # 把原始蒸馏损失乘以配置的权重后加到总量中。
            + config.lambda_dist * first["dist_raw"],
            places=5,  # 允许浮点累计产生小数点后第 5 位以外的舍入误差。
        )

    # 断点只允许删除一条已写 CSV、但尚未包含在上一个检查点中的合法尾行。
    def test_resume_log_repairs_only_one_valid_unsaved_tail(self):
        # 断点只允许删除一条已写 CSV、但尚未包含在上一个检查点中的合法尾行。
        """断点只允许删除一条已写 CSV、但尚未包含在上一个检查点中的合法尾行。"""

        # 创建测试结束后自动删除的独立目录，避免接触真实 loss.csv。
        with TemporaryDirectory() as directory:
            # 在临时目录中定义损失表路径。
            path = Path(directory) / "loss.csv"
            # 第一轮使用基准尺度的九个损失值。
            first = self._losses(1.0)
            # 第二轮使用两倍尺度，便于确认被删除的是哪一行。
            second = self._losses(2.0)
            # 写入 checkpoint 已经包含的第 1 轮，global_step=1×1230。
            append_loss_csv(path, 1, 1230, first)
            # 模拟训练先写完第 2 轮 CSV，但还没来得及原子更新 latest.pt 就中断。
            append_loss_csv(path, 2, 2460, second)
            # 用只完成第 1 轮的检查点信息协调 CSV，应安全裁掉唯一未保存尾行。
            reconcile_loss_csv(
                path,  # 要检查和修复的临时损失表。
                completed_epoch=1,  # 检查点声明只完整保存到第 1 轮。
                batches_per_epoch=1230,  # 每轮固定 1230 次更新，用于核对 step 游标。
                checkpoint_losses=first,  # 检查点内保存的第 1 轮损失必须与 CSV 对上。
            )
            # 重新读取修复后的所有文本行，第一行是表头。
            lines = path.read_text(encoding="utf-8").splitlines()
            # 修复后只应剩表头和第 1 轮数据，共两行。
            self.assertEqual(len(lines), 2)
            # 唯一数据行必须确实是 epoch=1、global_step=1230。
            self.assertTrue(lines[1].startswith("1,1230,"))
            # 原子改写使用的临时文件必须已清理，不能残留半写状态。
            self.assertFalse(path.with_suffix(".csv.tmp").exists())

    # 恢复检查点时必须同时锁定下一轮 epoch、global_step 和上一轮完整损失。
    def test_resume_checkpoint_locks_epoch_and_optimizer_step_cursor(self):
        # 恢复检查点时必须同时锁定下一轮 epoch、global_step 和上一轮完整损失。
        """恢复检查点时必须同时锁定下一轮 epoch、global_step 和上一轮完整损失。"""

        # 使用自动清理目录，测试不会覆盖正式 latest.pt。
        with TemporaryDirectory() as directory:
            # 临时检查点文件沿用正式命名，验证路径处理无特殊分支。
            path = Path(directory) / "latest.pt"
            # 创建需要保存和恢复的小学生模型。
            student = TinyEpsilon(trainable=True)
            # 创建需要保存和恢复的小判别器。
            discriminator = TinyDiscriminator()
            # 学生 Adam 只用于验证优化器状态与接口，学习率 0.001 是轻量测试值。
            student_optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
            # 判别器必须有完全独立的 Adam 状态，和正式实现一致。
            discriminator_optimizer = torch.optim.Adam(
                # 优化器只登记判别器参数，测试学习率设为 0.001。
                discriminator.parameters(), lr=0.001
            )
            # 构造一轮完整训练/测试损失字典，随检查点一起保存。
            losses = self._losses()
            # 保存声明已完成第 2 轮、2460 步的完整可恢复检查点。
            save_checkpoint(
                path,  # 临时 latest.pt 输出位置。
                student,  # 要保存的学生参数。
                discriminator,  # 要保存的判别器参数。
                student_optimizer,  # 要保存的学生 Adam 内部状态。
                discriminator_optimizer,  # 要保存的判别器 Adam 内部状态。
                # 当前完整训练轮次。
                epoch=2,  # 检查点完成轮次。
                # 检查点记录的累计优化器更新步数。
                global_step=2460,  # 2×1230，对应轮次边界。
                # 检查点绑定的训练契约规范化指纹。
                run_contract_sha256="fixed-contract",  # 固定字符串模拟代码/配置身份哈希。
                # 该轮损失摘要，随权重归档保存。
                epoch_losses=losses,  # 保存第 2 轮训练和测试损失，供 CSV 协调。
                include_training_state=True,  # 要求写入优化器和 RNG 等完整恢复状态。
            )
            # 从刚保存的检查点恢复所有对象，并取得训练应继续的位置。
            start_epoch, global_step, restored_losses = load_resume_checkpoint(
                path,  # 读取同一个临时检查点。
                student,  # 把参数加载进同结构学生。
                discriminator,  # 把参数加载进同结构判别器。
                student_optimizer,  # 恢复学生 Adam 状态。
                discriminator_optimizer,  # 恢复判别器 Adam 状态。
                torch.device("cpu"),  # 测试环境中的参数存放设备。
                # 预先审核的训练契约指纹。
                expected_run_contract_sha256="fixed-contract",  # 必须与保存值完全一致。
                batches_per_epoch=1230,  # 用于验证 epoch 与 step 数学关系。
                max_epochs=100,  # 正式训练上限，防止恢复到范围外轮次。
            )
            # 已完成 epoch 2 后应从 epoch 3 开始，global_step 仍是 2460。
            self.assertEqual((start_epoch, global_step), (3, 2460))
            # 恢复出的上一轮损失必须与保存前字典逐项一致。
            self.assertEqual(restored_losses, losses)
            # 换蒸馏目标会改变运行契约；旧优化器状态不能伪装成新实验的精确续训。
            # 先设置一个醒目的当前值，确保拒绝发生在任何旧参数写入之前。
            with torch.no_grad():
                student.scale.fill_(0.314)
            # 保存当前参数及优化器状态，检查失败的续训尝试不会污染它们。
            before_student = student.scale.detach().clone()
            before_optimizer = copy.deepcopy(student_optimizer.state_dict())
            # 错误必须明确说明契约变化，而不能先恢复权重再在别处失败。
            with self.assertRaisesRegex(RuntimeError, "Run contract changed"):
                load_resume_checkpoint(
                    path, student, discriminator, student_optimizer,
                    discriminator_optimizer, torch.device("cpu"),
                    expected_run_contract_sha256="changed-distillation-objective",
                    batches_per_epoch=1230, max_epochs=100,
                )
            # 参数和Adam配置/状态逐项保持原样，证明旧实验没有串入新实验。
            torch.testing.assert_close(student.scale, before_student)
            self.assertEqual(student_optimizer.state_dict(), before_optimizer)

    # 完整单批训练应更新学生和判别器，同时绝不更新冻结教师与声码器。
    def test_complete_two_optimizer_step_preserves_gradient_boundaries(self):
        # 完整单批训练应更新学生和判别器，同时绝不更新冻结教师与声码器。
        """完整单批训练应更新学生和判别器，同时绝不更新冻结教师与声码器。"""

        # 固定所有全局随机操作，使参数更新和失败可重复。
        torch.manual_seed(12)
        # 使用正式冻结配置中的批量、输入长度、优化器超参数和损失权重。
        config = PAPER_STUDENT_CONFIG
        # 教师参数不可训练且固定在 eval 模式。
        teacher = TinyEpsilon(trainable=False).eval()
        # 学生参数可训练并处于 train 模式。
        student = TinyEpsilon(trainable=True).train()
        # 声码器处于 eval 模式，避免批归一化等状态改变。
        vocoder = TinyVocoder().eval()
        # 判别器处于 train 模式并应在第一阶段更新。
        discriminator = TinyDiscriminator().train()
        # 显式冻结教师，正式代码也必须保持同一边界。
        set_requires_grad(teacher, False)
        # 显式冻结声码器参数，但它作为可微函数仍向学生输出传梯度。
        set_requires_grad(vocoder, False)
        # 创建学生独立 Adam，严格使用论文学习率和 beta。
        student_optimizer = torch.optim.Adam(
            student.parameters(),  # 只登记学生参数。
            lr=config.learning_rate,  # 论文学习率 0.0002。
            betas=(config.adam_beta1, config.adam_beta2),  # 论文 beta=(0.5,0.9)。
        )
        # 创建判别器自己的 Adam，超参数与学生一致但状态完全分离。
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),  # 只登记判别器参数。
            lr=config.learning_rate,  # 使用同一论文学习率。
            betas=(config.adam_beta1, config.adam_beta2),  # 使用同一论文 beta。
        )
        # 构造形状与正式训练一致的一个完整批次。
        batch = {
            # 32 条、80 通道、128 帧的规范化梅尔输入。
            "mel": torch.randn(config.batch_size, 80, config.train_frames),
            # 同批 144 通道、128 帧的 BNF 内容条件。
            "bnf": torch.randn(config.batch_size, 144, config.train_frames),
            # 每条语音一个 256 维目标说话人嵌入。
            "spk_emb": torch.randn(config.batch_size, 256),
        }
        # 使用正式 1000 步余弦扩散参数，确保 train_batch 走真实数学路径。
        diffusion = VoiceGradDiffusion(
            # 采用配置中的扩散总步数与余弦日程偏移。
            config.diffusion_steps, config.cosine_offset
        )
        # 克隆训练前学生标量，用来证明优化器确实更新了它。
        student_before = student.scale.detach().clone()
        # 克隆训练前判别器标量，用来证明独立优化器也确实更新了它。
        discriminator_before = discriminator.scale.detach().clone()
        # 执行一次完整训练批：先判别器更新，再学生更新并返回九个损失。
        losses = train_batch(
            batch,  # 上面构造的完整 batch32 输入。
            teacher,  # 冻结教师产生分数蒸馏目标。
            student,  # 本批第二阶段要更新的学生。
            vocoder,  # 冻结但可微的梅尔到波形映射。
            discriminator,  # 本批第一阶段要更新、第二阶段要冻结的判别器。
            diffusion,  # 正式噪声日程与扩散公式。
            student_optimizer,  # 只负责学生参数更新。
            discriminator_optimizer,  # 只负责判别器参数更新。
            torch.zeros(1, 80, 1),  # 反规范化梅尔均值测试值 0。
            torch.ones(1, 80, 1),  # 反规范化梅尔标准差测试值 1。
            torch.device("cpu"),  # 轻量单元测试设备，不是项目训练设备。
            config,  # 正式冻结学生配置。
        )
        # 学生标量必须不同于训练前值，证明学生 optimizer.step() 生效。
        self.assertNotEqual(float(student.scale.detach()), float(student_before))
        # 判别器标量也必须改变，证明两个优化阶段都真实执行。
        self.assertNotEqual(
            # 比较判别器当前缩放参数和更新前快照，确认它实际得到更新。
            float(discriminator.scale.detach()), float(discriminator_before)
        )
        # 所有返回损失都必须是有限数，避免 NaN/Inf 被写进 CSV 和曲线图。
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        # 冻结教师的所有参数都必须保持无梯度。
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        # 冻结声码器的所有参数也必须保持无梯度。
        self.assertTrue(all(parameter.grad is None for parameter in vocoder.parameters()))


# 只有直接运行这个文件时才启动 unittest；被测试发现器导入时不会重复执行。
if __name__ == "__main__":
    # 自动收集本类所有 test_ 方法并以标准退出码报告结果。
    unittest.main()
