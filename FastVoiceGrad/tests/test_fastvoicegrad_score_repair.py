# 独立验证干净目标固定点、停止梯度、解析梯度，并提供可选真实教师诊断。
"""Independent fixed-point/gradient tests and an opt-in teacher target probe.

Run the probe only on Matpool. Its batch file is a trusted tensor dictionary
with student_prediction [B,80,T], speaker_embedding [B,256], bnf [B,144,T].
It never updates model parameters. Example from the project root:

  python -m tests.test_fastvoicegrad_score_repair --diagnose \
    --teacher-checkpoint TEACHER.pt --batch-file BATCH.pt --device cuda

The normal unittest path uses only tiny tensors and no model checkpoints.
"""

# 读取诊断模式、教师检查点、固定批次和运行设备等命令行参数。
import argparse
# 枚举两种教师目标与三种权重的全部六种组合。
import itertools
# 把数值诊断输出为 JSON，方便保存与不同版本逐项对比。
import json
# 区分命令行请求的是数学测试还是实际教师的只读诊断。
import sys
# 使用标准单元测试检查蒸馏公式，所有张量计算在租用服务器执行。
import unittest

# 使用张量和自动求导验证反向扩散目标与梯度。
import torch
# 构建可控的噪声预测器，使数学期望不依赖真实模型性能。
from torch import nn

# 测试直接调用正式蒸馏函数，检查的就是训练所用计算路径。
from fastvoicegrad.objectives import score_distillation_loss
# 使用训练中的同一余弦扩散日程，避免测试另写一套系数。
from voicegrad.diffusion import VoiceGradDiffusion


# 模拟准确知道加噪内容的教师，也可人为加偏差来检查梯度。
class FixedNoise(nn.Module):
    # 保存测试指定的噪声张量，后续每次前向返回相同结果。
    def __init__(self, noise):
        # 初始化 PyTorch 模块，使测试噪声参数能被登记和检查梯度。
        super().__init__()
        # 把噪声存为可求导参数；若 stop-gradient 失效，测试能发现它收到梯度。
        self.noise = nn.Parameter(noise.clone())
        # 统计前向次数，防止一次损失计算偷偷反复调用教师。
        self.calls = 0

    # 保持真实教师的输入接口，但测试替身不利用条件预测噪声。
    def forward(self, x, timesteps, speaker_embedding, bnf):
        # 一次进入 forward 就计数一次，与之后是否反向传播无关。
        self.calls += 1
        # 将预设噪声扩展为输入形状，构造答案已知的理想教师。
        return self.noise.expand_as(x)


# 让教师确实依赖全部输入，避免条件未参与计算时误通过停止梯度检查。
class ConditionedNoise(FixedNoise):
    # 使用与真实教师相同的四个输入，并记录调用时实际看到的值。
    def forward(self, x, timesteps, speaker_embedding, bnf):
        # 只记录本次真实前向，不另调父类而重复累计次数。
        self.calls += 1
        # 保存数值副本供测试核对，避免后续操作改变审计记录。
        self.received = tuple(value.detach().clone() for value in (x, timesteps, speaker_embedding, bnf))
        # 记录教师执行时自动求导是否关闭；这比只检查参数没有更新更直接。
        self.grad_enabled = torch.is_grad_enabled()
        # 教师输出同时依赖自己的参数、带噪特征、说话人向量和内容特征。
        return (
            # 可训练噪声参数用于发现教师权重是否意外收到梯度。
            self.noise.expand_as(x)
            # 保留对带噪输入的依赖，检查学生和再加噪噪声的路径是否被切断。
            + 0.1 * x
            # 每条说话人向量压成一个偏移，广播到本条语音的全部通道与帧。
            + 0.2 * speaker_embedding.mean(1).reshape(-1, 1, 1)
            # 对内容通道平均，仍保留各帧差异并广播到两个测试 mel 通道。
            + 0.3 * bnf.mean(1, keepdim=True)
        )


# 检验修复后的干净目标、历史兼容性和不同噪声权重。
class ScoreDistillationRepairTests(unittest.TestCase):
    # 为每项数学测试重新建立固定的小批次，避免测试相互影响。
    def setUp(self):
        # 建立 T=1000 的实际余弦日程。
        self.diffusion = VoiceGradDiffusion()
        # 固定局部随机种子，确保测试失败可以重现。
        self.generator = torch.Generator().manual_seed(7201)
        # 构造 [3,2,7] 的小型干净特征：3个样本、2个通道、7帧。
        self.x = torch.randn(3, 2, 7, generator=self.generator)
        # 生成与干净特征逐元素对应的标准高斯噪声。
        self.noise = torch.randn(self.x.shape, generator=self.generator)
        # 同批覆盖低、中、高噪声时刻，检查每个样本的系数查表。
        self.timesteps = torch.tensor([100, 500, 950])
        # 构造3个说话人条件占位向量；替身模型不依赖向量值。
        self.speaker = torch.zeros(3, 4)
        # 构造 [3,5,7] 的内容条件，占位BNF时间轴与特征的7帧一致。
        self.bnf = torch.zeros(3, 5, 7)

    # 用固定条件调用损失，测试只需切换教师目标和加权规则。
    def evaluate(self, x, teacher, target_mode, weight_mode):
        # 返回生产函数计算的标量损失和停止梯度的教师目标。
        return score_distillation_loss(
            # 传入待比较特征、测试教师、实际日程、逐样本时刻和已知噪声。
            x, teacher, self.diffusion, self.timesteps, self.noise,
            # 两种条件与特征共享相同样本顺序和时间长度。
            self.speaker, self.bnf,
            # 显式传递本次测试选择的目标空间与权重，防止落回历史默认。
            target_mode=target_mode, weight_mode=weight_mode,
        )

    # 若教师准确预测加噪内容，正确学生输出应保持不变且蒸馏损失近零。
    def test_oracle_clean_reconstruction_is_fixed_point(self):
        # 同批覆盖第一步、中段、学生起点和扩散末端，防止只测温和噪声。
        timesteps = torch.tensor([1, 500, 950, 1000])
        # 末端信号很小，使用双精度输入减少相减抵消误差；生产日程保持原样。
        x = torch.randn(4, 2, 7, generator=self.generator, dtype=torch.float64)
        # 使用非零、已知的噪声，验证教师确实将所加噪声去除。
        noise = torch.randn(x.shape, generator=self.generator, dtype=x.dtype)
        # 条件维度保持小型，但批量必须与四个时刻一一对应。
        speaker, bnf = torch.zeros(4, 4), torch.zeros(4, 5, 7)
        # 逐一测试单步保留量、累计信号功率和累计信号幅度三种权重。
        for weight_mode in ("nominal_alpha", "alpha_bar", "sqrt_alpha_bar"):
            # 失败时单独指出哪一种权重破坏了数学性质。
            with self.subTest(weight_mode=weight_mode):
                # 每种权重重新建立教师计数器，单独检查只调用一次。
                teacher = FixedNoise(noise)
                # 显式调用干净目标路径，保留生产函数的再加噪和反推计算。
                loss, target = score_distillation_loss(
                    # 四个时刻各有自己的学生特征、真实噪声和条件。
                    x, teacher, self.diffusion, timesteps, noise, speaker, bnf,
                    # 权重只能缩放距离，不能改变理想教师恢复出的干净目标。
                    target_mode="clean_x0", weight_mode=weight_mode,
                )
                # 日程缓冲区仍为 float32，给其系数舍入留出小容差，不允许整体收缩。
                torch.testing.assert_close(target, x, atol=1e-5, rtol=1e-5)
                # 理想预测下蒸馏损失必须接近零，不能继续惩罚正确学生。
                self.assertLess(float(loss), 1e-5)
                # 四个时间步组成一个批次，教师仍只需一次前向。
                self.assertEqual(teacher.calls, 1)
                # 教师目标应停止梯度，不能沿教师分支反向影响学生。
                self.assertFalse(target.requires_grad)

    # 保留旧公式的反例，防止今后再次把含噪相邻步当成干净监督。
    def test_legacy_target_fails_same_oracle_fixed_point(self):
        # 取得待检验的蒸馏损失及对应教师重建目标。
        loss, target = self.evaluate(
            # 同一正确输入改用旧 t→t−1 目标和旧权重，直接暴露目标空间差异。
            self.x, FixedNoise(self.noise), "legacy_adjacent", "nominal_alpha"
        )
        # 旧公式即使遇到理想教师仍有明显损失，确认反例有足够区分度。
        self.assertGreater(float(loss), 0.3)
        # 检查 t=950 的高噪声样本仍显著偏离干净特征。
        self.assertGreater(float((target[-1] - self.x[-1]).abs().mean()), 0.3)

    # 把自动求导结果与 L1 的解析梯度比较，同时检查教师参数完全无梯度。
    def test_gradient_is_weighted_l1_with_no_teacher_path(self):
        # 覆盖全部目标/权重组合，避免只测试默认路径。
        for target_mode, weight_mode in itertools.product(
            # 两种目标空间乘三种加权方式，一共检验六条路径。
            ("clean_x0", "legacy_adjacent"), ("nominal_alpha", "alpha_bar", "sqrt_alpha_bar")
        ):
            # 在失败信息中同时保留目标空间和权重名称。
            with self.subTest(target_mode=target_mode, weight_mode=weight_mode):
                # 将学生特征设置为可求导叶子，直接检查损失传回学生输出的梯度。
                x = self.x.clone().requires_grad_(True)
                # A deliberate epsilon error keeps clean-target gradients away
                # from zero, so this checks stop-gradient rather than roundoff.
                # 让教师故意偏差0.2，产生确定的非零梯度，避免舍入噪声干扰。
                teacher = FixedNoise(self.noise + 0.2)
                # 按当前组合计算损失，教师目标在内部已经停止梯度。
                loss, target = self.evaluate(x, teacher, target_mode, weight_mode)
                # 只对这个小型损失反传，检验梯度连接与系数。
                loss.backward()
                # 明确列出三种权重对应的系数数组，避免同名 alpha 混用。
                schedule = {
                    # 旧权重采用当前单步的1−βt。
                    "nominal_alpha": self.diffusion.alphas,
                    # 对照权重采用从0到t的累计信号功率。
                    "alpha_bar": self.diffusion.alphas_cumprod,
                    # 本次声明的修复权重采用累计信号幅度，与 ADD 的幅度参数对应。
                    "sqrt_alpha_bar": self.diffusion.sqrt_alphas_cumprod,
                # 按本次权重名称取得唯一对应日程。
                }[weight_mode]
                # 每个样本取自己的权重，并广播到通道和时间维。
                weight = schedule[self.timesteps].reshape(3, 1, 1)
                # 手算加权平均 L1 的梯度：误差符号乘权重，再除以全部元素数。
                expected = weight * torch.sign(x.detach() - target) / x.numel()
                # 要求自动求导与独立解析公式一致，检查缩放及 stop-gradient。
                torch.testing.assert_close(x.grad, expected)
                # 即便测试教师参数允许求导，蒸馏目标分支也不能给它积累梯度。
                self.assertIsNone(teacher.noise.grad)
                # 教师目标应停止梯度，不能沿教师分支反向影响学生。
                self.assertFalse(target.requires_grad)
                # 一次目标和损失计算不能变成多次教师采样。
                self.assertEqual(teacher.calls, 1)

    # 梯度必须只回到学生预测，教师、再加噪噪声及两类条件均保持不动。
    def test_teacher_noise_and_conditions_are_stopped_without_changing_inputs(self):
        # 干净目标与显式历史回放都必须遵守相同的停止梯度边界。
        for target_mode in ("clean_x0", "legacy_adjacent"):
            # 失败信息保留目标模式，便于找到哪条路径断开方式有误。
            with self.subTest(target_mode=target_mode):
                # 学生预测是唯一应收到损失梯度的叶子张量。
                x = self.x.clone().requires_grad_(True)
                # 故意让噪声也可求导，确认教师分支不会把梯度泄漏给它。
                noise = self.noise.clone().requires_grad_(True)
                # 非零说话人条件确实影响测试教师的输出。
                speaker = (self.speaker + 0.4).requires_grad_(True)
                # 非零内容条件同样参与前向，避免无效的断梯度断言。
                bnf = (self.bnf + 0.6).requires_grad_(True)
                # 保存所有输入，反向后检查没有原位修改任何数值。
                before = tuple(value.detach().clone() for value in (x, noise, speaker, bnf))
                # 测试教师显式使用全部条件和可训练噪声参数。
                teacher = ConditionedNoise(self.noise + 0.2)
                # 仅调用一次正式损失函数，不另写替代生产路径。
                loss, target = score_distillation_loss(
                    # 噪声和条件故意需要梯度，生产函数应主动阻断目标分支。
                    x, teacher, self.diffusion, self.timesteps, noise, speaker, bnf,
                    # 两条目标路径统一使用本次修复的信号幅度权重。
                    target_mode=target_mode, weight_mode="sqrt_alpha_bar",
                )
                # 触发正式 L1 的反向计算，检查所有输入与教师参数的梯度。
                loss.backward()
                # 解析梯度为逐例权重乘误差符号，再除以总元素数。
                expected_gradient = (
                    # 将三条样本的累计信号幅度广播到通道与帧。
                    self.diffusion.sqrt_alphas_cumprod[self.timesteps].reshape(-1, 1, 1)
                    # 教师目标被固定后，只剩 L1 对学生预测这一侧的导数。
                    * torch.sign(x.detach() - target) / x.numel()
                )
                # 自动求导应与独立解析梯度一致，不包含教师的 Jacobian。
                torch.testing.assert_close(x.grad, expected_gradient)
                # 四个被检查对象都不应接收到损失梯度。
                for value in (noise, speaker, bnf, teacher.noise):
                    # grad=None 表示没有通过该对象建立回传路径。
                    self.assertIsNone(value.grad)
                # 教师重建目标不能保留计算图。
                self.assertFalse(target.requires_grad)
                # 教师实际只执行一次，并且在关闭自动求导的上下文中执行。
                self.assertEqual(teacher.calls, 1)
                # 与冻结参数配合，禁止对教师前向构造输入梯度图。
                self.assertFalse(teacher.grad_enabled)
                # 按原噪声和原学生数值独立构造应送给教师的带噪特征。
                expected_noisy = self.diffusion.q_sample(before[0], self.timesteps, before[1])
                # 时刻、条件和带噪值都应原样传入，停止梯度不能改变内容。
                for actual, expected in zip(teacher.received, (expected_noisy, self.timesteps, before[2], before[3])):
                    # 检查精确相等，排除为了断梯度而偷偷改变教师输入。
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                # 反向计算结束后，调用者提供的全部输入也必须数值不变。
                for actual, expected in zip((x, noise, speaker, bnf), before):
                    # 这里只检查数值，学生的 grad 允许存在但数据本身不可被原位改写。
                    torch.testing.assert_close(actual.detach(), expected, atol=0, rtol=0)

    # 更换权重只应改变损失大小，不能改变教师重建出的特征。
    def test_weight_choice_changes_loss_but_not_target(self):
        # 加入固定0.3噪声预测偏差，使不同权重的数值差异可被检测。
        teacher = FixedNoise(self.noise + 0.3)
        # 先计算旧单步权重下的干净目标与损失。
        nominal, nominal_target = self.evaluate(
            # 固定干净目标解释，只使用单步保留量加权。
            self.x, teacher, "clean_x0", "nominal_alpha"
        )
        # 再以完全相同的教师和输入计算累计功率加权结果。
        cumulative, cumulative_target = self.evaluate(
            # 保留相同目标空间，仅切换累计信号功率权重。
            self.x, teacher, "clean_x0", "alpha_bar"
        )
        # 两个教师目标必须逐位相同，确保权重没有混进目标构造。
        torch.testing.assert_close(nominal_target, cumulative_target, atol=0, rtol=0)
        # 逐样本平均通道和时间上的绝对误差，保留3个样本各自的权重位置。
        error = (nominal_target - self.x).abs().flatten(1).mean(1)
        # 将生产结果与独立写出的逐样本加权公式比较。
        torch.testing.assert_close(
            # 旧模式的期望值为每例 L1 乘对应单步 alpha，再对3例平均。
            nominal, (self.diffusion.alphas[self.timesteps] * error).mean()
        )
        # 将生产结果与独立写出的逐样本加权公式比较。
        torch.testing.assert_close(
            # 累计模式的期望值改为每例 L1 乘对应 alpha_bar，再平均。
            cumulative, (self.diffusion.alphas_cumprod[self.timesteps] * error).mean()
        )
        # 要求这个反例中两种权重确有明显差距，防止测试误把等价数值当有效覆盖。
        self.assertGreater(float(nominal), float(cumulative) * 2)

    # 核对本次修复选择的累计信号幅度，并明确每个样本先加权再求平均。
    def test_score_weight_is_cumulative_signal_amplitude(self):
        # 加入固定0.3噪声预测偏差，使不同权重的数值差异可被检测。
        teacher = FixedNoise(self.noise + 0.3)
        # 显式选择修复实验的干净目标和信号幅度权重。
        loss, target = self.evaluate(self.x, teacher, "clean_x0", "sqrt_alpha_bar")
        # 保留每个时刻对应样本的原始 L1，方便独立验证加权。
        error = (target - self.x).abs().flatten(1).mean(1)
        # 从逐样本误差建立信号幅度加权的手算期望。
        expected = (
            # 按t取 sqrt(alpha_bar)，减轻高噪声重建误差的影响。
            self.diffusion.sqrt_alphas_cumprod[self.timesteps] * error
        # 最后对整个小批次取平均，与生产损失的归约方式一致。
        ).mean()
        # 确认实现确实使用幅度系数，数值与手算结果一致。
        torch.testing.assert_close(loss, expected)
        # 另算累计功率权重作为容易混淆的对照。
        power_weight = (
            # 对照项使用 alpha_bar，数值比其平方根更小。
            self.diffusion.alphas_cumprod[self.timesteps] * error
        # 最后对整个小批次取平均，与生产损失的归约方式一致。
        ).mean()
        # 验证两个解释没有被代码混为同一权重。
        self.assertGreater(float(loss), float(power_weight))
        # 若先对误差和权重各自平均再相乘，就会丢掉每例t与其误差的对应关系。
        wrong_order = self.diffusion.sqrt_alphas_cumprod[self.timesteps].mean() * error.mean()
        # 这个非均匀时刻批次应能区分错误归约顺序，不能靠数值巧合通过。
        self.assertGreater(float((loss - wrong_order).abs()), 1e-3)

    # 缺少任意一个模式时必须拒绝调用，避免无意回到旧公式或另一套权重。
    def test_both_modes_are_required_keyword_arguments(self):
        # 分别检查全部省略、只给目标和只给权重这三种不完整调用。
        for keywords in ({}, {"target_mode": "clean_x0"}, {"weight_mode": "sqrt_alpha_bar"}):
            # 显示缺少哪个参数，便于定位 API 默认值被重新加入的问题。
            with self.subTest(keywords=keywords):
                # 记录教师调用，确认参数缺失在任何模型计算前就被拒绝。
                teacher = FixedNoise(self.noise)
                # 必填关键字缺失应由 Python 函数签名报 TypeError。
                with self.assertRaises(TypeError):
                    # 故意只提供部分模式，其他输入完整有效。
                    score_distillation_loss(
                        # 构造一个本来可以正常计算的批次。
                        self.x, teacher, self.diffusion, self.timesteps, self.noise,
                        # 展开不完整关键字，检验真正的函数调用契约。
                        self.speaker, self.bnf, **keywords,
                    )
                # 不能先调用教师再报告配置缺失。
                self.assertEqual(teacher.calls, 0)

    # 两个模式必须写出参数名，防止位置顺序混淆后选错训练公式。
    def test_modes_cannot_be_supplied_as_positional_arguments(self):
        # 建立可计数的测试教师，确认调用约束在模型计算之前生效。
        teacher = FixedNoise(self.noise)
        # 即使提供了两个有效名称，放在位置参数中也必须报TypeError。
        with self.assertRaises(TypeError):
            # 故意把本应是关键字参数的目标和权重接在七个输入后面。
            score_distillation_loss(
                # 前七个位置参数都是正常的训练输入。
                self.x, teacher, self.diffusion, self.timesteps, self.noise, self.speaker, self.bnf,
                # 这两个名称必须显式写target_mode=与weight_mode=，此调用因此应被拒绝。
                "clean_x0", "sqrt_alpha_bar",
            )
        # 参数位置错误不能触发一次无意义的教师前向。
        self.assertEqual(teacher.calls, 0)

    # 模式名称拼错时必须立刻失败，不能悄悄选择另一种训练公式。
    def test_invalid_modes_fail_before_model_call(self):
        # 分别构造目标名错误和权重名错误两类输入。
        for target_mode, weight_mode in (
            # 一组只错目标名，另一组只错权重名，确保两项都有校验。
            ("adjacent_typo", "nominal_alpha"), ("clean_x0", "weight_typo")
        ):
            # 在失败信息中同时保留目标空间和权重名称。
            with self.subTest(target_mode=target_mode, weight_mode=weight_mode):
                # 记录教师调用次数，不能只凭错误类型猜测是否在前向前完成校验。
                teacher = FixedNoise(self.noise)
                # 无效模式应以明确参数错误终止。
                with self.assertRaises(ValueError):
                    # 把故意拼错的配置传入生产函数，检验失败行为。
                    self.evaluate(self.x, teacher, target_mode, weight_mode)
                # 配置拼写错误时教师不应执行任何计算。
                self.assertEqual(teacher.calls, 0)


# 在固定真实模型批次上比较目标失真，只做前向和解析计算，不更新参数。
def diagnose_teacher_targets(
    # 接收冻结教师、学生mel [B,80,T]、说话人条件及对齐BNF。
    teacher, student_prediction, speaker_embedding, bnf, diffusion=None,
    # 覆盖从低噪声到末端的10个时刻，并固定独立诊断噪声种子。
    timesteps=(10, 100, 250, 500, 750, 900, 950, 990, 999, 1000), seed=92601,
):
    # 在同一个固定真实批上，逐噪声级比较两种目标与三种权重的六个组合。
    """Return six target/weight probes per t on one fixed real-model batch.

    The exact score noise is shared by every mode and every t. Gradients are
    analytical dL/dx for the stopped-target L1 objective, not optimizer steps.
    Positive gradient dot x indicates a direction reducing mel magnitude.
    """
    # 切断学生训练图，诊断不能意外改变训练梯度。
    x = student_prediction.detach()
    # 未提供现有日程时，创建同配置的余弦日程。
    if diffusion is None:
        # 将扩散系数放到学生特征所在设备，避免CPU/GPU混用。
        diffusion = VoiceGradDiffusion().to(x.device)
    # 使用局部随机生成器，诊断不消耗正式训练的全局随机序列。
    generator = torch.Generator(device=x.device).manual_seed(seed)
    # 六种组合共用同一份逐元素高斯噪声，数值差异来自公式选择。
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    # 收集每个时刻、每种目标和每种权重的可比较诊断记录。
    records = []
    # 让教师使用推理模式，固定可能存在的随机或归一化层行为。
    teacher.eval()
    # 逐个检查预先选定的噪声时刻。
    for step in timesteps:
        # 本次让整个批次处于相同t，以便观察噪声强度的影响。
        t = torch.full((x.shape[0],), step, dtype=torch.long, device=x.device)
        # Six modes need only two teacher forward calls; weighting cannot
        # change the reconstruction. This keeps the diagnostic inexpensive.
        # 比较保留噪声的历史相邻目标与修复后的干净目标。
        for target_mode in ("legacy_adjacent", "clean_x0"):
            # 只取教师重建目标；下面重新计算三种权重，不重复教师前向。
            _, target = score_distillation_loss(
                # 同一批特征、条件、时间步和噪声用于所有目标解释。
                x, teacher, diffusion, t, noise, speaker_embedding, bnf,
                # 此处权重仅为取得目标所需的占位选择，重建目标本身与权重无关。
                target_mode=target_mode, weight_mode="nominal_alpha",
            )
            # 计算教师重建相对学生特征的偏差。
            error = target - x
            # 逐一测试单步保留量、累计信号功率和累计信号幅度三种权重。
            for weight_mode in ("nominal_alpha", "alpha_bar", "sqrt_alpha_bar"):
                # 明确列出三种权重对应的系数数组，避免同名 alpha 混用。
                schedule = {
                    # 单步权重用于重放原实验。
                    "nominal_alpha": diffusion.alphas,
                    # 累计信号功率权重用于对照。
                    "alpha_bar": diffusion.alphas_cumprod,
                    # 累计信号幅度权重对应修复主解释。
                    "sqrt_alpha_bar": diffusion.sqrt_alphas_cumprod,
                # 按本次权重名称取得唯一对应日程。
                }[weight_mode]
                # 把每例权重扩成 [B,1,1] 并使用与特征相同精度。
                weight = schedule[t].reshape(-1, 1, 1).to(x.dtype)
                # 直接计算停止梯度 L1 对学生输出的解析梯度，避免执行模型反传。
                gradient = weight * torch.sign(x - target) / x.numel()
                # 将教师目标展平成向量，便于计算方向相似度。
                target_flat = target.flatten().float()
                # 将学生特征展平，作为目标和梯度方向的参照。
                x_flat = x.flatten().float()
                # 将损失梯度展平，检查它是否倾向压低整体特征幅度。
                gradient_flat = gradient.flatten().float()
                # 将所加噪声展平，检查教师目标中是否仍大量残留该噪声。
                noise_flat = noise.flatten().float()
                # 余弦相似度只比较向量方向；越接近1表示越同向。
                cosine = lambda a, b: float(torch.nn.functional.cosine_similarity(
                    # 给向量加批次维，并用极小下限避免零向量导致除零。
                    a[None], b[None], dim=1, eps=1e-12
                ))
                # 保存本时刻、本目标、本权重的全部诊断数值。
                records.append({
                    # 记录扩散时刻和教师目标空间，便于定位异常来自哪个阶段。
                    "t": int(step), "target_mode": target_mode,
                    # 记录采用的明确权重名，避免只写含义不清的 alpha。
                    "weight_mode": weight_mode,
                    # 报告相邻扩散步的信号保留量。
                    "alpha_t": float(diffusion.alphas[step]),
                    # 报告从干净状态累计到当前时刻的信号功率。
                    "alpha_bar_t": float(diffusion.alphas_cumprod[step]),
                    # 本批共享t，取第一个元素即可记录实际损失权重。
                    "weight": float(weight.flatten()[0]),
                    # 学生特征的均方根幅度，用作教师目标幅度的参照。
                    "input_rms": float(x.square().mean().sqrt()),
                    # 教师目标的均方根幅度，检查是否收缩或异常放大。
                    "target_rms": float(target.square().mean().sqrt()),
                    # 未加权的平均绝对偏差，显示教师目标离学生输出多远。
                    "target_l1_from_student": float(error.abs().mean()),
                    # 偏差的均方根，更敏感地反映少量较大误差。
                    "target_l2_from_student": float(error.square().mean().sqrt()),
                    # 该组合实际加入蒸馏目标的加权平均L1，还未乘总权重45。
                    "weighted_loss": float((weight * error.abs()).mean()),
                    # 记录传回学生特征的梯度强度，比较不同权重对更新的影响。
                    "weighted_gradient_rms": float(gradient.square().mean().sqrt()),
                    # 梯度与学生特征同向时，梯度下降会压小特征幅度。
                    "gradient_cosine_student": cosine(gradient_flat, x_flat),
                    # 目标越接近学生内容方向，该数值越接近1。
                    "target_cosine_student": cosine(target_flat, x_flat),
                    # 该值很高说明目标仍跟随所加噪声方向，需要检查目标空间。
                    "target_cosine_score_noise": cosine(target_flat, noise_flat),
                    # 检查重建目标是否包含NaN或无穷大。
                    "finite": bool(torch.isfinite(target).all()),
                })
    # 返回六种组合在各个噪声时刻的记录，交给调用方保存或比较。
    return records


# 命令行诊断入口，只加载指定教师和固定批次，不启动训练。
def _diagnostic_main():
    # 建立只读诊断参数说明，并提醒实际计算必须在Matpool执行。
    parser = argparse.ArgumentParser(description="Read-only teacher target probe; run on Matpool")
    # 显式选择真实教师诊断，避免普通测试误加载大型模型。
    parser.add_argument("--diagnose", action="store_true")
    # 要求提供已知教师权重文件，不能猜测或默认挑选检查点。
    parser.add_argument("--teacher-checkpoint", required=True)
    # 要求提供固定输入批次，保证多种目标使用完全相同的数据。
    parser.add_argument("--batch-file", required=True)
    # 默认在租用服务器CPU运行；需要GPU时显式指定cuda。
    parser.add_argument("--device", default="cpu")
    # 解析并验证本次诊断的命令行参数。
    args = parser.parse_args()
    # 复用项目兼容不同PyTorch版本的可信检查点读取函数。
    from fastvoicegrad.components import load_torch_checkpoint
    # 按原教师架构构建模型，保证参数名称和形状可严格匹配。
    from voicegrad.train import build_model
    # 读取指定教师检查点的模型参数与元数据。
    checkpoint = load_torch_checkpoint(__import__("pathlib").Path(args.teacher_checkpoint))
    # 读取已固定的学生特征、说话人向量和BNF张量。
    batch = load_torch_checkpoint(__import__("pathlib").Path(args.batch_file))
    # 使用与原教师一致的BNF下采样配置，在指定设备建立推理模型。
    model = build_model("legacy_decimate").to(args.device).eval()
    # 严格加载全部教师参数，任何缺层、多层或维度差异都会报错。
    model.load_state_dict(checkpoint["model"], strict=True)
    # 冻结教师全部参数，诊断不产生参数更新。
    model.requires_grad_(False)
    # 按固定顺序取出诊断需要的三类张量。
    x, speaker, bnf = (
        # 把每个张量搬到教师所在设备。
        batch[key].to(args.device)
        # 文件字段分别对应学生mel、说话人条件和内容条件。
        for key in ("student_prediction", "speaker_embedding", "bnf")
    )
    # 执行六种公式组合的只读比较，并逐条输出结果。
    for record in diagnose_teacher_targets(model, x, speaker, bnf):
        # 每条记录输出一行JSON并立即刷新，便于实时保存与审查。
        print(json.dumps(record, ensure_ascii=False), flush=True)


# 仅在直接运行模块时进入诊断或单元测试，导入时不执行计算。
if __name__ == "__main__":
    # 根据显式标志决定是否加载真实教师。
    if "--diagnose" in sys.argv:
        # 执行只读真实模型诊断。
        _diagnostic_main()
    # 没有诊断标志时，只运行小张量数学测试。
    else:
        # 运行本文件的测试并用退出码报告通过或失败。
        unittest.main()
