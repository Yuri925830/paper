# 用可手算的小张量锁定 FastVoiceGrad 学生模型的关键论文公式。
"""用可手算的小张量锁定 FastVoiceGrad 学生模型的关键论文公式。"""

# unittest 负责组织数学测试和给出清楚的断言失败位置。
import unittest

# PyTorch 提供张量、随机噪声、自动求导和数值比较。
import torch
# nn.Module 用来构造输出已知的教师/学生 epsilon 测试替身。
import torch.nn as nn

# 统一读取正式配置，测试论文常数没有被代码修改。
from fastvoicegrad.config import PAPER_STUDENT_CONFIG
# 导入需要逐个验证的学生预测、教师一步均值和三种损失公式。
from fastvoicegrad.objectives import (
    acdd_student_loss,  # 论文公式（14）的最终三项加权和。
    feature_matching_loss,  # 判别器各层真实/生成特征的 L1 汇总。
    make_student_prediction,  # 从扩散起点一步预测干净梅尔。
    reverse_mean_adjacent,  # 教师从 t 到 t-1 的反向均值。
    score_distillation_loss,  # 按显式选择的教师目标和权重约束学生预测。
)
# 使用项目真实 VoiceGradDiffusion，避免单元测试重复实现后与生产公式一起写错。
from voicegrad.diffusion import VoiceGradDiffusion


# 固定输出 epsilon 的模型让期望结果能够直接代入扩散公式手算。
class FixedEpsilon(nn.Module):
    # 无论输入是什么都返回预先给定 epsilon 的只读测试模型。
    """无论输入是什么都返回预先给定 epsilon 的只读测试模型。"""

    # 把给定 epsilon 登记为 buffer，使它随设备移动但不参与优化。
    def __init__(self, epsilon):
        # 把给定 epsilon 登记为 buffer，使它随设备移动但不参与优化。
        """把给定 epsilon 登记为 buffer，使它随设备移动但不参与优化。"""

        # 初始化模块基类，才能正确登记 buffer。
        super().__init__()
        # buffer 不属于可训练参数，但会跟随模型的 state_dict 和 device。
        self.register_buffer("epsilon", epsilon)

    # 把固定 epsilon 扩展成与输入 x 完全相同的批量和形状。
    def forward(self, x, timesteps, speaker_embedding, bnf):
        # 把固定 epsilon 扩展成与输入 x 完全相同的批量和形状。
        """把固定 epsilon 扩展成与输入 x 完全相同的批量和形状。"""

        # expand_as 不复制实际数据，只创建广播视图，并保证输出尺寸与 x 对齐。
        return self.epsilon.expand_as(x)


# 全零 epsilon 模型用于把含噪声预测的公式简化到只剩 alpha 系数。
class ZeroEpsilon(nn.Module):
    # 接受正式模型接口但始终预测零噪声。
    """接受正式模型接口但始终预测零噪声。"""

    # 返回与 x 同形状、同 dtype、同 device 的全零张量。
    def forward(self, x, timesteps, speaker_embedding, bnf):
        # 返回与 x 同形状、同 dtype、同 device 的全零张量。
        """返回与 x 同形状、同 dtype、同 device 的全零张量。"""

        # zeros_like 保留输入元信息，使 CPU/GPU 和不同精度下接口都正确。
        return torch.zeros_like(x)


# 锁定 K=1 学生、教师相邻均值、alpha 权重和论文总损失。
class FastVoiceGradStudentMathTests(unittest.TestCase):
    # 锁定 K=1 学生、教师相邻均值、alpha 权重和论文总损失。
    """锁定 K=1 学生、教师相邻均值、alpha 权重和论文总损失。"""

    # 每个测试前创建相同的小扩散过程和条件张量。
    def setUp(self):
        # 每个测试前创建相同的小扩散过程和条件张量。
        """每个测试前创建相同的小扩散过程和条件张量。"""

        # 用 10 步而非正式 1000 步可快速手算；余弦偏移仍保持正式值 0.008。
        self.diffusion = VoiceGradDiffusion(n_levels=10, offset=0.008)
        # 两条样本、3 维的全零说话人条件；小模型不使用数值但接口必须存在。
        self.speaker = torch.zeros(2, 3)
        # 两条样本、4 通道、8 帧的全零 BNF 条件，时间长度与测试梅尔一致。
        self.bnf = torch.zeros(2, 4, 8)

    # 若学生精确预测生成 x_t 时的噪声，K=1 公式应恢复原始 x_0。
    def test_sparse_one_step_recovers_x0_with_exact_epsilon(self):
        # 若学生精确预测生成 x_t 时的噪声，K=1 公式应恢复原始 x_0。
        """若学生精确预测生成 x_t 时的噪声，K=1 公式应恢复原始 x_0。"""

        # 固定随机种子，使干净梅尔和噪声每次运行一致。
        torch.manual_seed(7)
        # 构造两条、5 通道、8 帧的任意干净梅尔 x_0。
        x_zero = torch.randn(2, 5, 8)
        # 构造与 x_0 同形状的标准正态起始噪声。
        source_noise = torch.randn_like(x_zero)
        # 学生被设成恰好输出真实 source_noise，代表理想噪声预测器。
        student = FixedEpsilon(source_noise)
        # 从第 9 步含噪状态执行一次学生反向预测。
        prediction, _ = make_student_prediction(
            student,  # 输出精确噪声的理想学生。
            self.diffusion,  # 10 级余弦扩散日程。
            x_zero,  # 要恢复的原始干净梅尔。
            self.speaker,  # 接口所需的说话人条件。
            self.bnf,  # 接口所需的内容条件。
            # 单步学生固定的起始时刻950。
            start_step=9,  # 小日程最后一个合法索引，对应正式设定的高噪声起点。
            # 构造x950的独立标准高斯噪声。
            source_noise=source_noise,  # 显式传入已知噪声以便计算期望值。
        )
        # 理想 epsilon 下预测应与 x_0 在 2e-5 相对/绝对误差内一致。
        torch.testing.assert_close(prediction, x_zero, rtol=2e-5, atol=2e-5)

    # K=1 从 x_t 直接估计 x_0 时必须用累计 alpha_bar_t，而不是单步 alpha_t。
    def test_sparse_k1_uses_alpha_bar_at_start_not_nominal_alpha(self):
        # K=1 从 x_t 直接估计 x_0 时必须用累计 alpha_bar_t，而不是单步 alpha_t。
        """K=1 从 x_t 直接估计 x_0 时必须用累计 alpha_bar_t，而不是单步 alpha_t。"""

        # 固定随机输入，保证错误公式与正确公式之间差距稳定。
        torch.manual_seed(91)
        # 创建任意干净梅尔。
        x_zero = torch.randn(2, 5, 8)
        # 让测试模型固定预测 0.25 噪声，避免随机输出干扰公式比较。
        epsilon = torch.full_like(x_zero, 0.25)
        # 构造固定 epsilon 学生。
        model = FixedEpsilon(epsilon)
        # 创建扩散 x_0 到 x_9 所需的源噪声。
        source_noise = torch.randn_like(x_zero)
        # 调用生产 K=1 函数得到实际直接预测。
        direct, _ = make_student_prediction(
            model,  # 固定输出 0.25 的学生替身。
            self.diffusion,  # 测试扩散过程。
            x_zero,  # 原始干净输入。
            self.speaker,  # 说话人条件。
            self.bnf,  # BNF 内容条件。
            # 单步学生固定的起始时刻950。
            start_step=9,  # 从最高测试噪声步直接预测。
            # 构造x950的独立标准高斯噪声。
            source_noise=source_noise,  # 显式固定源噪声。
        )
        # 两条样本都处于扩散索引 9。
        timestep = torch.full((2,), 9, dtype=torch.long)
        # 用真实正向扩散公式计算 x_9，作为两种反推公式共同输入。
        x_nine = self.diffusion.q_sample(x_zero, timestep, source_noise)
        # 正确 K=1 反推必须取从 0 累乘到 9 的 alpha_bar_9。
        alpha_bar = self.diffusion.alphas_cumprod[9]
        # 按 x_0=(x_t-sqrt(1-alpha_bar_t)*epsilon)/sqrt(alpha_bar_t) 手算正确期望。
        expected = (x_nine - torch.sqrt(1.0 - alpha_bar) * epsilon) / torch.sqrt(
            # 分母使用当前时刻累计信号保留率的平方根。
            alpha_bar
        )
        # 单步 alpha_9 只适用于相邻反向均值，这里故意构造错误结果作为反例。
        nominal_alpha = self.diffusion.alphas[9]
        # 把错误 alpha_9 代入同一外形公式，数值应明显偏离生产结果。
        wrong = (x_nine - torch.sqrt(1.0 - nominal_alpha) * epsilon) / torch.sqrt(
            # 此反例故意把累计系数换成单步系数，以检测公式混淆。
            nominal_alpha
        )
        # 实际 K=1 输出必须等于使用累计 alpha_bar 的正确期望。
        torch.testing.assert_close(direct, expected)
        # 同时要求它与错误单步 alpha 版本平均至少相差 1e-3，防止测试没有区分力。
        self.assertGreater(float(torch.mean(torch.abs(direct - wrong))), 1e-3)

    # 通用扩散采样器设 K=1 时必须与学生训练的一步预测公式完全一致。
    def test_generic_k1_sampler_matches_student_training_equation(self):
        # 通用扩散采样器设 K=1 时必须与学生训练的一步预测公式完全一致。
        """通用扩散采样器设 K=1 时必须与学生训练的一步预测公式完全一致。"""

        # 使用全零 x_0，减少无关数值，让两条路径的随机噪声成为唯一变化来源。
        x_zero = torch.zeros(2, 5, 8)
        # 固定模型预测 epsilon=0.25，使两条实现调用的是同一个确定函数。
        epsilon = torch.full_like(x_zero, 0.25)
        # 创建固定输出模型。
        model = FixedEpsilon(epsilon)
        # 第一个局部随机生成器供通用 sample 内部产生源噪声。
        first_generator = torch.Generator().manual_seed(91)
        # 调用通用采样接口，只走一步并要求先从 x_0 正向加噪。
        sampled = self.diffusion.sample(
            model,  # 固定 epsilon 模型。
            x_zero,  # 全零源梅尔。
            self.speaker,  # 说话人条件。
            self.bnf,  # 内容条件。
            num_steps=1,  # 明确要求 K=1。
            # 单步学生固定的起始时刻950。
            start_step=9,  # 从测试日程第 9 步开始。
            # 声码器生成器参数字典。
            generator=first_generator,  # 固定内部源噪声序列。
            diffuse_source=True,  # 先把 x_0 扩散成 x_9，再执行反向一步。
        )
        # 创建同种子第二个局部生成器，以手工复现 sample 内部抽到的第一份噪声。
        second_generator = torch.Generator().manual_seed(91)
        # 用与 sample 相同的形状、dtype 和生成器显式生成源噪声。
        source_noise = torch.randn(
            # 按源梅尔的形状和精度，用独立同种子生成器复现相同噪声。
            x_zero.shape, dtype=x_zero.dtype, generator=second_generator
        )
        # 把这份显式噪声送入学生训练使用的一步预测函数。
        direct, _ = make_student_prediction(
            model,  # 与通用采样相同的固定模型。
            self.diffusion,  # 与通用采样相同的日程。
            x_zero,  # 相同全零源梅尔。
            self.speaker,  # 相同说话人条件。
            self.bnf,  # 相同内容条件。
            # 单步学生固定的起始时刻950。
            start_step=9,  # 相同起始步。
            # 构造x950的独立标准高斯噪声。
            source_noise=source_noise,  # 与 sample 内部逐元素相同的噪声。
        )
        # 两条公开路径必须逐元素相等，否则训练公式和正式推理器已经漂移。
        torch.testing.assert_close(direct, sampled)

    # 教师从 t 到相邻 t-1 的反向均值应使用单步 alpha_t。
    def test_adjacent_teacher_mean_uses_nominal_alpha(self):
        # 教师从 t 到相邻 t-1 的反向均值应使用单步 alpha_t。
        """教师从 t 到相邻 t-1 的反向均值应使用单步 alpha_t。"""

        # 创建两条任意 x_t，覆盖 5 通道、8 帧。
        x_t = torch.randn(2, 5, 8)
        # 两条样本分别选早期 t=2 和晚期 t=9，检查批内不同系数的 gather。
        timesteps = torch.tensor([2, 9], dtype=torch.long)
        # 教师预测零 epsilon 后，公式可简化为 x_t/sqrt(alpha_t)。
        model = ZeroEpsilon()
        # 调用生产教师相邻反向均值函数。
        actual = reverse_mean_adjacent(
            model,  # 始终输出零噪声的教师替身。
            self.diffusion,  # 测试扩散日程。
            x_t,  # 当前含噪状态。
            timesteps,  # 每条样本各自的当前时刻。
            self.speaker,  # 说话人条件。
            self.bnf,  # 内容条件。
        )
        # 从单步 alpha 数组按每条样本时刻取值，并扩成可广播的 [2,1,1]。
        alpha = self.diffusion.alphas.gather(0, timesteps).view(2, 1, 1)
        # 零 epsilon 情况下手算期望相邻均值。
        expected = x_t / torch.sqrt(alpha)
        # 生产函数必须与单步 alpha_t 手算结果逐元素一致。
        torch.testing.assert_close(actual, expected)

    # 只有显式指定历史模式时，才保留旧实验的相邻步目标和单步 alpha_t 权重。
    def test_legacy_nominal_weight_is_preserved(self):
        # 历史公式仍可回放，但不能靠省略参数隐式启用。
        """显式回放旧实验的相邻步目标与单步 alpha_t 权重。"""

        # 学生预测设全零且要求梯度，用来确认蒸馏损失确实能训练学生。
        student_prediction = torch.zeros(2, 5, 8, requires_grad=True)
        # 分数目标加噪统一设为 1，便于稳定比较两种权重。
        score_noise = torch.ones_like(student_prediction)
        # 同批选择 t=2 与 t=9，使 alpha 和 alpha_bar 的差异足够明显。
        timesteps = torch.tensor([2, 9], dtype=torch.long)
        # 调用生产蒸馏损失，教师 epsilon 固定为零。
        loss, target = score_distillation_loss(
            student_prediction,  # 当前学生的一步预测结果。
            ZeroEpsilon(),  # 零 epsilon 教师使目标更容易独立手算。
            self.diffusion,  # 测试扩散过程。
            timesteps,  # 每个样本的蒸馏时刻。
            score_noise,  # 显式固定的目标噪声。
            self.speaker,  # 说话人条件。
            self.bnf,  # BNF 内容条件。
            # 明确要求旧 t→t−1 目标，仅用于核对历史实现。
            target_mode="legacy_adjacent",
            # 同时明确选择旧单步权重，不能依赖生产接口的默认值。
            weight_mode="nominal_alpha",
        )
        # 零学生先加噪成为 sqrt(1−alpha_bar_t)；零教师再除以 sqrt(alpha_t)。
        expected_target = (
            # 正向加噪的系数采用生产日程中保存的累计噪声标准差。
            self.diffusion.sqrt_one_minus_alphas_cumprod[timesteps].reshape(2, 1, 1)
            # 旧相邻反向均值中的分母使用单步信号保留量。
            / torch.sqrt(self.diffusion.alphas[timesteps]).reshape(2, 1, 1)
            # 原始噪声是全1，保留乘法以写清楚这个手算目标的来源。
            * score_noise
        )
        # 单独核对旧目标公式，避免只检验返回目标自身的距离而漏掉目标漂移。
        torch.testing.assert_close(target, expected_target)
        # 按每条样本把目标绝对误差在通道和时间维求平均。
        per_example = target.abs().flatten(1).mean(1)
        # 旧实验值：逐样本乘对应单步 alpha_t，再对 batch 求平均。
        expected = torch.mean(self.diffusion.alphas.gather(0, timesteps) * per_example)
        # 对照另一种权重，确认显式历史回放仍保持原来的加权规则。
        wrong = torch.mean(
            # 逐样本乘累计权重构成对照值，确认显式单步权重没有被改变。
            self.diffusion.alphas_cumprod.gather(0, timesteps) * per_example
        )
        # 实际 loss 必须等于单步 alpha_t 的手算期望。
        torch.testing.assert_close(loss, expected)
        # 实际 loss 与错误 alpha_bar 版本至少差 1e-4，保证测试能抓到混用。
        self.assertGreater(float(torch.abs(loss - wrong)), 1e-4)
        # 对蒸馏 loss 反传，验证其计算图连接学生预测。
        loss.backward()
        # 学生预测必须收到梯度，否则这一项无法训练学生网络。
        self.assertIsNotNone(student_prediction.grad)
        # 教师生成的 target 必须停止梯度，避免反向传播进入冻结教师目标。
        self.assertFalse(target.requires_grad)

    # 原始特征匹配函数只求各层 L1 均值之和，不能内部偷偷再乘论文权重 2。
    def test_feature_matching_has_no_hidden_factor_two(self):
        # 原始特征匹配函数只求各层 L1 均值之和，不能内部偷偷再乘论文权重 2。
        """原始特征匹配函数只求各层 L1 均值之和，不能内部偷偷再乘论文权重 2。"""

        # 两个判别器分支各给一层真实特征，第一层含两个元素、第二层含一个元素。
        real = [[torch.tensor([1.0, 3.0])], [torch.tensor([2.0])]]
        # 构造对应假特征：第一层绝对差为 [1,2]，第二层绝对差为 [3]。
        fake = [[torch.tensor([0.0, 1.0])], [torch.tensor([5.0])]]
        # 第一层 L1 均值=(1+2)/2=1.5；第二层均值=3；原始论文和为 4.5。
        # 若函数内部错误乘了 2，这里会得到 9 并立即失败。
        self.assertAlmostEqual(float(feature_matching_loss(real, fake)), 4.5)

    # 学生总损失必须严格等于论文公式（14）：L_adv + 2L_FM + 45L_dist。
    def test_total_loss_is_exact_equation_14(self):
        # 学生总损失必须严格等于论文公式（14）：L_adv + 2L_FM + 45L_dist。
        """学生总损失必须严格等于论文公式（14）：L_adv + 2L_FM + 45L_dist。"""

        # 传入容易手算的三个标量：对抗=1、特征匹配=2、蒸馏=3。
        total = acdd_student_loss(
            # 三个手算标量依次代表对抗损失、原始特征匹配损失、原始蒸馏损失。
            torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)
        )
        # 正确结果是 1 + 2×2 + 45×3；任一权重或加法顺序错误都会被抓到。
        self.assertEqual(float(total), 1.0 + 2.0 * 2.0 + 45.0 * 3.0)

    # 把论文公开的所有学生训练核心常数锁成明确断言，防止以后无意改动。
    def test_paper_constants_are_locked(self):
        # 把论文公开的所有学生训练核心常数锁成明确断言，防止以后无意改动。
        """把论文公开的所有学生训练核心常数锁成明确断言，防止以后无意改动。"""

        # 从正式训练同一冻结实例读取，不在测试中重复创建另一份配置。
        config = PAPER_STUDENT_CONFIG
        # FastVoiceGrad 学生必须只执行一个反向扩散步 K=1。
        self.assertEqual(config.student_steps, 1)
        # 单步学生必须从论文规定的 S_K=950 开始。
        self.assertEqual(config.student_start_step, 950)
        # 蒸馏教师必须使用训练到 epoch 500 的版本。
        self.assertEqual(config.teacher_epoch, 500)
        # 全局训练 batch size 必须保持论文值 32。
        self.assertEqual(config.batch_size, 32)
        # 学习率必须是论文值 2×10^-4，即 0.0002。
        self.assertEqual(config.learning_rate, 0.0002)
        # Adam 两个动量系数必须同时保持 (0.5, 0.9)。
        self.assertEqual((config.adam_beta1, config.adam_beta2), (0.5, 0.9))
        # 学生训练终点必须固定为 100 个 epoch。
        self.assertEqual(config.epochs, 100)
        # 公式（14）的特征匹配和蒸馏权重必须分别为 2 与 45。
        self.assertEqual((config.lambda_fm, config.lambda_dist), (2.0, 45.0))

    # 单独锁定本次声明的修复默认值，不把有解释空间的公式记号说成论文逐字设定。
    def test_repaired_score_defaults_are_explicit(self):
        # 默认教师目标应从抽到的t直接恢复干净x0，不再留在相邻含噪时刻。
        self.assertEqual(PAPER_STUDENT_CONFIG.score_target, "clean_x0")
        # 修复后的默认权重按累计信号幅度计算，并在运行契约中披露。
        self.assertEqual(PAPER_STUDENT_CONFIG.score_weight, "sqrt_alpha_bar")


# 只有直接执行本文件时才启动测试；被 unittest discover 导入时不会重复运行。
if __name__ == "__main__":
    # 自动运行本类全部 test_ 方法并以退出码表示成功或失败。
    unittest.main()
