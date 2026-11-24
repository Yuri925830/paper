# 用小型 CPU 张量锁定 FastVoiceGrad Cycle v1 的配置、配对、往返梯度和损失公式。
"""用小型 CPU 张量锁定 FastVoiceGrad Cycle v1 的关键行为。"""  # 本文件只定义测试，不会在导入时启动训练。

from dataclasses import replace  # replace 只复制并改动测试需要的冻结配置字段。
import unittest  # unittest 负责组织测试并提供异常与数值断言。

import torch  # torch 提供测试张量、自动求导和固定随机抽样。
import torch.nn as nn  # nn.Module 用来构造只有一个参数的学生替身。

from fastvoicegrad.config import PAPER_STUDENT_CONFIG  # 直接检查正式训练使用的同一份冻结配置。
from fastvoicegrad.objectives import (  # 导入正式 Cycle 数学函数，避免测试另写一套实现。
    cycle_augmented_student_loss,  # 检查基线损失之外只增加一次加权 Cycle 项。
    cycle_consistency_loss,  # 检查屏蔽补零帧的逐样本 L1 归约。
    make_cycle_round_trip_prediction,  # 检查 A→B→A 两程及梯度边界。
)  # 完成 Cycle 数学函数导入。
from fastvoicegrad.train import (  # 导入训练入口实际使用的轮次门控和目标选择函数。
    cycle_is_active,  # 检查 Cycle 是否只从指定轮次开始生效。
    select_cycle_target_indices,  # 检查 batch 内随机目标一定来自不同说话人。
)  # 完成训练辅助函数导入。
from voicegrad.diffusion import VoiceGradDiffusion  # 使用项目真实扩散日程验证两程输入。


class RecordingCycleStudent(nn.Module):  # 这个小学生记录两次调用，便于直接查看梯度边界和条件。
    """记录每次前向的梯度状态、输入和说话人条件。"""  # 模型只服务于 Cycle 单元测试。

    def __init__(self) -> None:  # 初始化唯一可训练参数和三组调用记录。
        """建立一个足以产生非零第二程梯度的学生替身。"""  # 记录不会进入优化器。

        super().__init__()  # 先初始化模块基类，才能登记可训练参数。
        self.scale = nn.Parameter(torch.tensor(0.05))  # 单个缩放参数让梯度来源容易检查。
        self.grad_modes = []  # 依次保存每次 forward 是否启用自动求导。
        self.seen_inputs = []  # 依次保存两程真正送进学生的带噪 mel。
        self.seen_speakers = []  # 依次保存第一程目标 B 和第二程源 A 的身份条件。

    def forward(  # 保持与正式 VoiceGrad 学生相同的四参数调用接口。
        self,  # self 指向当前记录型学生实例。
        x: torch.Tensor,  # x 是当前一程已经加噪的 mel。
        timesteps: torch.Tensor,  # timesteps 是每条样本对应的扩散起点。
        speaker_embedding: torch.Tensor,  # speaker_embedding 指定本程要生成的说话人。
        bnf: torch.Tensor,  # bnf 是当前 v1 两程共同复用的源内容特征。
    ) -> torch.Tensor:  # 返回与 x 同形的噪声预测。
        """记录调用现场，再用一个参数生成可微噪声预测。"""  # 未使用的条件仍保留正式接口。

        self.grad_modes.append(torch.is_grad_enabled())  # 第一程应记录 False，第二程应记录 True。
        self.seen_inputs.append(x.detach().clone())  # 保存数值副本，避免后续反向修改计算图。
        self.seen_speakers.append(speaker_embedding.detach().clone())  # 保存本程身份条件用于核对顺序。
        return self.scale * torch.tanh(x)  # 非线性输出让第二程对 scale 产生稳定非零梯度。


class CycleConfigurationAndSamplingTests(unittest.TestCase):  # 这一组只检查配置披露、轮次门控和异说话人配对。
    """检查 Cycle 配置和目标说话人选择不会悄悄退化。"""  # 所有测试都只用内存对象。

    def test_cycle_defaults_are_disabled_and_declared(self) -> None:  # 基线默认不得被新增目标改变。
        """Cycle 默认关闭，新增选择只出现在复现声明而不是论文明确参数中。"""  # 区分论文事实与实验选择。

        config = PAPER_STUDENT_CONFIG  # 读取正式入口使用的冻结配置实例。
        self.assertFalse(config.cycle_enabled)  # 不加命令行开关时必须保持原 FastVoiceGrad 基线。
        self.assertEqual(config.cycle_start_epoch, 71)  # CycleDiffusion 式门控从第 71 轮开始。
        self.assertEqual(config.lambda_cycle, 1.0)  # 新增 Cycle 外层权重固定为 1。
        self.assertEqual(config.cycle_examples_per_batch, 3)  # 每批只对前三条源语音建立往返路径。
        self.assertTrue(config.cycle_first_leg_stop_gradient)  # 正式 v1 的 A→B 第一程必须停止梯度。
        self.assertEqual(config.cycle_content_mode, "reuse_source_bnf")  # 当前两程如实复用源 BNF。
        self.assertEqual(config.cycle_loss_reduction, "masked_per_example_l1_then_batch_mean_divide_n_mels")  # 锁定公开代码额外除 80 的尺度。
        self.assertEqual(config.cycle_reverse_steps, 1)  # FastVoiceGrad 每一程仍然只执行一个反向步。
        declared = config.declared_reproduction_choices()  # 取得论文未明确、由本项目公开登记的选择。
        paper = config.paper_explicit()  # 取得只能声称为论文明确给出的参数集合。
        expected_declared_keys = {  # 汇总每个 Cycle 字段，防止运行契约漏掉关键语义。
            "cycle_enabled",  # 记录是否启用新增实验。
            "cycle_start_epoch",  # 记录实际开始轮次。
            "lambda_cycle",  # 记录 Cycle 权重。
            "cycle_examples_per_batch",  # 记录每批 Cycle 子批大小。
            "cycle_first_leg_stop_gradient",  # 记录第一程梯度边界。
            "cycle_content_mode",  # 记录当前内容条件处理方式。
            "cycle_target_sampling",  # 记录异说话人目标抽样规则。
            "cycle_loss_reduction",  # 记录有效帧归约规则。
            "cycle_noise_policy",  # 记录两程噪声彼此独立。
            "cycle_train_pair_seed",  # 记录训练配对独立随机流的基种子。
            "cycle_train_forward_noise_seed",  # 记录训练 A→B 噪声独立随机流的基种子。
            "cycle_train_backward_noise_seed",  # 记录训练 B→A 噪声独立随机流的基种子。
            "cycle_reverse_steps",  # 记录每程反向步数。
            "cycle_test_target_policy",  # 记录测试时固定异说话人配对策略。
        }  # 完成应该公开披露的 Cycle 字段集合。
        self.assertTrue(expected_declared_keys.issubset(declared))  # 所有新增选择都必须进入复现声明。
        self.assertTrue(expected_declared_keys.isdisjoint(paper))  # 不得把新增 Cycle 选择冒充论文原参数。

    def test_cycle_epoch_gate_is_one_based_and_explicit(self) -> None:  # 检查第 70/71 轮边界和非法轮次。
        """只有显式开启并到达第 71 轮时 Cycle 才能激活。"""  # 关闭路径应一直保持基线行为。

        enabled = replace(PAPER_STUDENT_CONFIG, cycle_enabled=True)  # 创建只在本测试中开启 Cycle 的配置副本。
        disabled = replace(PAPER_STUDENT_CONFIG, cycle_enabled=False)  # 创建明确关闭 Cycle 的配置副本。
        self.assertFalse(cycle_is_active(enabled, 70))  # 第 70 轮仍属于原始 FastVoiceGrad 训练。
        self.assertTrue(cycle_is_active(enabled, 71))  # 第 71 轮开始建立 A→B→A 路径。
        self.assertTrue(cycle_is_active(enabled, 100))  # 门控开启后应一直保持激活直到训练结束。
        self.assertFalse(cycle_is_active(disabled, 100))  # 总开关关闭时即使轮次很高也不能激活。
        with self.assertRaises(ValueError):  # 轮次从 1 开始，0 必须被明确拒绝。
            cycle_is_active(enabled, 0)  # 故意传入非法的第 0 轮。

    def test_target_sampling_always_chooses_a_different_speaker(self) -> None:  # 检查每个源样本的目标身份。
        """目标索引可以重复，但绝不能指向与源样本相同的说话人。"""  # 使用多 utterance 同说话人覆盖易错情况。

        torch.manual_seed(2026)  # 固定全局随机状态，用于证明 Cycle 专属抽样不会改变它。
        global_state_before = torch.random.get_rng_state().clone()  # 保存调用前 CPU 全局随机状态。
        cycle_generator = torch.Generator(device="cpu")  # 建立与全局状态隔离的 Cycle 配对生成器。
        cycle_generator.manual_seed(2026)  # 固定专属生成器，使目标索引可复现。
        speakers = ["p001", "p001", "p002", "p003"]  # 前两条属于同一说话人，后两条各不相同。
        selected = select_cycle_target_indices(  # 为前三条源语音从完整 batch 选择目标。
            speakers,  # 传入与 batch 顺序一致的说话人名称。
            source_count=3,  # 只检查前三条源语音，与正式 Cycle 子批语义一致。
            device=torch.device("cpu"),  # 单元测试只在 CPU 构造索引。
            generator=cycle_generator,  # 目标选择只能消费 Cycle 专属随机流。
        )  # 完成异说话人目标抽样。
        torch.testing.assert_close(torch.random.get_rng_state(), global_state_before)  # 全局基线随机流必须逐字节不变。
        self.assertEqual(tuple(selected.shape), (3,))  # 每个源样本必须恰好得到一个目标索引。
        self.assertEqual(selected.dtype, torch.long)  # index_select 需要 long 型索引。
        for source_index, target_index in enumerate(selected.tolist()):  # 逐条核对源 A 与目标 B 身份。
            self.assertNotEqual(speakers[source_index], speakers[target_index])  # 严格禁止 identity cycle。

    def test_target_sampling_fails_closed_for_single_speaker_batch(self) -> None:  # 单说话人 batch 无合法 B。
        """没有异说话人候选时必须报错，不能退回自身身份。"""  # 失败关闭比悄悄训练 identity cycle 更安全。

        with self.assertRaises(RuntimeError):  # 单说话人属于运行期数据组成错误。
            cycle_generator = torch.Generator(device="cpu")  # 即使会提前失败也显式提供专属生成器。
            cycle_generator.manual_seed(2026)  # 固定测试专属生成器状态。
            select_cycle_target_indices(  # 尝试为只有同一身份的 batch 选择目标。
                ["p001", "p001", "p001"],  # 三条 utterance 全部来自同一说话人。
                source_count=3,  # 三条都被要求进入 Cycle 子批。
                device=torch.device("cpu"),  # 在 CPU 上执行索引选择即可。
                generator=cycle_generator,  # 禁止错误路径接触全局随机流。
            )  # 正确实现应在这里抛出 RuntimeError。

    def test_target_sampling_rejects_invalid_source_counts(self) -> None:  # 同时覆盖空子批和元数据不足。
        """Cycle 子批数量必须为正，且不能超过说话人元数据长度。"""  # 防止索引错位。

        with self.assertRaises(ValueError):  # source_count=0 没有数学意义。
            empty_generator = torch.Generator(device="cpu")  # 为非法空子批提供隔离生成器。
            empty_generator.manual_seed(2026)  # 固定生成器但正确实现不应消费它。
            select_cycle_target_indices(  # 故意请求空 Cycle 子批。
                ["p001", "p002"],  # 提供两个不同说话人以隔离数量错误。
                source_count=0,  # 零条源语音应被拒绝。
                device=torch.device("cpu"),  # 设备不是本场景的错误来源。
                generator=empty_generator,  # 显式满足新的随机流接口。
            )  # 正确实现应报告 source_count 非法。
        with self.assertRaises(ValueError):  # 请求数量超过元数据长度也必须失败。
            oversized_generator = torch.Generator(device="cpu")  # 为越界数量建立独立生成器。
            oversized_generator.manual_seed(2026)  # 固定生成器但正确实现同样不应消费它。
            select_cycle_target_indices(  # 故意让源样本数与名称列表错位。
                ["p001", "p002"],  # 实际只有两条说话人元数据。
                source_count=3,  # 却请求给三条源语音配目标。
                device=torch.device("cpu"),  # 仍只在 CPU 检查输入契约。
                generator=oversized_generator,  # 显式满足新的随机流接口。
            )  # 正确实现应在任何随机抽样前报错。


class CycleConsistencyLossTests(unittest.TestCase):  # 这一组手算 masked L1 与公开代码额外 1/80 尺度。
    """检查补零帧不进入 Cycle 数值或梯度，并锁定公开缩放。"""  # 同时区分逐样本平均与全元素混合平均。

    def test_masked_per_example_l1_value_and_padding_gradient(self) -> None:  # 数值和反向梯度一起锁定。
        """两条不同长度语音先各自平均，再等权求 batch 平均。"""  # 补零区域即使误差很大也应完全无贡献。

        original = torch.zeros(2, 80, 4)  # 两条原始 mel 都设为零，便于直接读取绝对误差。
        cycled_values = torch.full((2, 80, 4), 4.0)  # 第二条四帧全部误差为 4。
        cycled_values[0, :, :2] = 2.0  # 第一条两帧有效区域误差为 2。
        cycled_values[0, :, 2:] = 99.0  # 第一条补零区域故意放大，验证掩码确实生效。
        cycled = cycled_values.requires_grad_()  # 把往返结果设为叶张量，以便直接检查每帧梯度。
        lengths = torch.tensor([2, 4], dtype=torch.long)  # 第一条只有两帧有效，第二条四帧都有效。
        loss = cycle_consistency_loss(original, cycled, lengths)  # 计算正式训练使用的 masked L1。
        self.assertAlmostEqual(float(loss.detach()), 3.0 / 80.0)  # 标准 batch 均值为 3，公开实现再除 80。
        loss.backward()  # 对标量损失反传，检查 padding 是否收到梯度。
        self.assertIsNotNone(cycled.grad)  # 往返 mel 必须连接到 Cycle 损失计算图。
        expected_first_valid = torch.full((80, 2), 1.0 / (2 * 80 * 2 * 80))  # 第一条梯度再含公开 1/80 系数。
        expected_second = torch.full((80, 4), 1.0 / (2 * 80 * 4 * 80))  # 第二条梯度也按同一公开尺度缩小。
        torch.testing.assert_close(cycled.grad[0, :, :2], expected_first_valid)  # 第一条有效帧梯度应均匀非零。
        torch.testing.assert_close(cycled.grad[0, :, 2:], torch.zeros(80, 2))  # 第一条 padding 梯度必须严格为零。
        torch.testing.assert_close(cycled.grad[1], expected_second)  # 第二条四帧梯度按该样本长度归一。

    def test_cycle_loss_rejects_invalid_mel_shapes(self) -> None:  # 覆盖广播和错误特征误传风险。
        """两个输入必须是完全同形的三维 80 通道 mel。"""  # 不允许广播悄悄通过。

        valid = torch.zeros(2, 80, 4)  # 建立符合契约的参照 mel。
        lengths = torch.tensor([4, 4], dtype=torch.long)  # 两条样本都声明四帧有效。
        with self.assertRaises(ValueError):  # 时间长度不同不能自动裁剪或广播。
            cycle_consistency_loss(valid, torch.zeros(2, 80, 3), lengths)  # 故意传入少一帧的往返 mel。
        with self.assertRaises(ValueError):  # 二维输入缺少 batch 或通道语义。
            cycle_consistency_loss(valid[0], valid[0], torch.tensor([4], dtype=torch.long))  # 故意去掉 batch 维。
        with self.assertRaises(ValueError):  # 144 通道 BNF 不能误当成 80 通道 mel。
            cycle_consistency_loss(torch.zeros(2, 144, 4), torch.zeros(2, 144, 4), lengths)  # 故意传入错误特征。

    def test_cycle_loss_rejects_invalid_lengths(self) -> None:  # 覆盖长度形状、类型和范围错误。
        """每条样本必须有一个位于 1..T 的 long 型有效长度。"""  # 空语音和越界值均不能生成掩码。

        mel = torch.zeros(2, 80, 4)  # 建立合法 mel，让失败只来自长度参数。
        with self.assertRaises(ValueError):  # 长度数量必须与 batch 大小相等。
            cycle_consistency_loss(mel, mel, torch.tensor([4], dtype=torch.long))  # 故意只给一条长度。
        with self.assertRaises(TypeError):  # 浮点长度容易隐藏上游裁剪错误，必须拒绝。
            cycle_consistency_loss(mel, mel, torch.tensor([4.0, 4.0]))  # 故意使用默认浮点 dtype。
        with self.assertRaises(ValueError):  # 有效长度不能为零。
            cycle_consistency_loss(mel, mel, torch.tensor([0, 4], dtype=torch.long))  # 第一条故意声明空语音。
        with self.assertRaises(ValueError):  # 有效长度不能超过统一时间轴 T=4。
            cycle_consistency_loss(mel, mel, torch.tensor([4, 5], dtype=torch.long))  # 第二条故意越界一帧。


class CycleRoundTripAndObjectiveTests(unittest.TestCase):  # 这一组检查实际两程调用和最终四项损失。
    """检查第一程 no_grad、第二程可微以及 Cycle 权重只加一次。"""  # 使用真实扩散公式和微型学生。

    def test_round_trip_stops_first_leg_and_backpropagates_second_leg(self) -> None:  # 锁定正式梯度边界。
        """A→B 不建图，B→A 使用真实 mel_AB 并把梯度传给共享学生。"""  # 同时核对两程身份条件顺序。

        diffusion = VoiceGradDiffusion(n_levels=6, offset=0.008)  # 用六步小日程降低单元测试计算量。
        student = RecordingCycleStudent()  # 建立记录两次 forward 的可训练学生替身。
        source_mel = torch.full((2, 80, 4), 0.20)  # 全正源 mel 让第二程参数梯度方向一致，避免元素间偶然抵消。
        source_speaker = torch.tensor([[1.0, 0.0], [2.0, 0.0]])  # A 身份条件使用容易识别的首维数值。
        target_speaker = torch.tensor([[9.0, 0.0], [8.0, 0.0]])  # B 身份条件与 A 明确不同。
        source_bnf = torch.zeros(2, 3, 4)  # 内容条件只需与 mel 的 batch 和时间轴对齐。
        forward_noise = torch.full_like(source_mel, 0.25)  # 第一程使用固定正噪声。
        backward_noise = torch.full_like(source_mel, 0.50)  # 第二程使用数值不同的固定正噪声，确保与第一程彼此独立。
        converted, cycled = make_cycle_round_trip_prediction(  # 执行正式 A→B→A 构造函数。
            student,  # 两程共享同一个学生参数。
            diffusion,  # 两程共享同一扩散日程。
            source_mel,  # 第一程从真实 A mel 出发。
            source_speaker,  # 第二程回到源说话人 A。
            target_speaker,  # 第一程转换到目标说话人 B。
            source_bnf,  # 当前 v1 两程都复用源 BNF。
            5,  # 小日程使用最后一个合法起点，对应正式实验的 950。
            forward_noise,  # 传入第一程专属噪声。
            backward_noise,  # 传入第二程专属噪声。
            first_leg_stop_gradient=True,  # 显式启用公开实现采用的第一程 stop-gradient。
        )  # 完成两程往返预测。
        self.assertEqual(student.grad_modes, [False, True])  # 第一程必须 no_grad，第二程必须保留计算图。
        self.assertFalse(converted.requires_grad)  # 中间 mel_AB 不得携带第一程计算图。
        self.assertTrue(cycled.requires_grad)  # 最终 mel_ABA 必须能通过第二程更新学生。
        torch.testing.assert_close(student.seen_speakers[0], target_speaker)  # 第一程条件应为目标说话人 B。
        torch.testing.assert_close(student.seen_speakers[1], source_speaker)  # 第二程条件应切回源说话人 A。
        timesteps = torch.full((2,), 5, dtype=torch.long)  # 建立与函数内部相同的小日程起点向量。
        expected_first_input = diffusion.q_sample(source_mel, timesteps, noise=forward_noise)  # 手算第一程加噪输入。
        expected_second_input = diffusion.q_sample(converted, timesteps, noise=backward_noise)  # 手算从实际 mel_AB 加噪的第二程输入。
        torch.testing.assert_close(student.seen_inputs[0], expected_first_input)  # 第一程必须消费源 mel 和第一份噪声。
        torch.testing.assert_close(student.seen_inputs[1], expected_second_input)  # 第二程必须消费实际 mel_AB 和第二份噪声。
        cycle_consistency_loss(  # 用正式 masked L1 生成反向标量。
            source_mel,  # 往返目标是原始 A mel。
            cycled,  # 预测侧是第二程生成的 mel_ABA。
            torch.tensor([4, 4], dtype=torch.long),  # 两条测试语音的四帧都有效。
        ).backward()  # 立即反传，检查共享学生是否从第二程得到梯度。
        self.assertIsNotNone(student.scale.grad)  # 第二程必须把梯度传到学生唯一参数。
        self.assertGreater(float(torch.abs(student.scale.grad)), 0.0)  # 梯度还必须是实际非零值。

    def test_round_trip_rejects_mismatched_noise_or_speaker_shapes(self) -> None:  # 检查两类高风险广播输入。
        """两程噪声必须逐元素同形，A/B 说话人向量也必须同形。"""  # 错误条件应在模型前向前失败。

        diffusion = VoiceGradDiffusion(n_levels=6, offset=0.008)  # 建立与正常测试相同的小扩散对象。
        student = RecordingCycleStudent()  # 建立不应被错误输入调用的学生替身。
        mel = torch.zeros(2, 80, 4)  # 建立合法源 mel。
        speaker = torch.zeros(2, 2)  # 建立合法源说话人条件。
        bnf = torch.zeros(2, 3, 4)  # 建立与 mel 时间对齐的内容条件。
        noise = torch.zeros_like(mel)  # 建立合法噪声作为对照。
        with self.assertRaises(ValueError):  # 第一程噪声少一帧时必须拒绝。
            make_cycle_round_trip_prediction(  # 调用正式往返函数验证形状门。
                student, diffusion, mel, speaker, speaker, bnf, 5,  # 前七项都保持合法。
                torch.zeros(2, 80, 3), noise,  # 只让第一程噪声形状错误。
                first_leg_stop_gradient=True,  # 梯度开关不是本场景错误来源。
            )  # 正确实现应在任何学生前向之前抛出 ValueError。
        with self.assertRaises(ValueError):  # 目标身份少一条时也必须拒绝。
            make_cycle_round_trip_prediction(  # 再次调用正式往返函数验证身份形状门。
                student, diffusion, mel, speaker, torch.zeros(1, 2), bnf, 5,  # 只让目标身份 batch 错位。
                noise, noise,  # 两程噪声都保持合法同形。
                first_leg_stop_gradient=True,  # 保持正式 stop-gradient 设置。
            )  # 正确实现应拒绝说话人条件广播。
        self.assertEqual(len(student.grad_modes), 0)  # 两次错误都应在进入学生前向之前被挡住。

    def test_augmented_loss_adds_exactly_one_weighted_cycle_term(self) -> None:  # 手算四项总目标与梯度。
        """总损失严格等于 L_adv+2L_FM+45L_dist+lambda_cycle L_cycle。"""  # 同时防止隐藏重复权重。

        adversarial = torch.tensor(1.0, requires_grad=True)  # 对抗项设为 1，便于手算。
        feature_matching = torch.tensor(2.0, requires_grad=True)  # 特征匹配项设为 2。
        score_distillation = torch.tensor(3.0, requires_grad=True)  # 蒸馏项设为 3。
        cycle = torch.tensor(4.0, requires_grad=True)  # Cycle 原始项设为 4。
        total = cycle_augmented_student_loss(  # 调用正式四项学生总损失。
            adversarial,  # 传入未加权对抗项。
            feature_matching,  # 传入未加权特征匹配项。
            score_distillation,  # 传入未加权蒸馏项。
            cycle,  # 传入未加权 Cycle 项。
            lambda_fm=2.0,  # 保持 FastVoiceGrad 论文特征匹配权重。
            lambda_dist=45.0,  # 保持 FastVoiceGrad 论文蒸馏权重。
            lambda_cycle=0.5,  # 用半权重让是否重复乘权容易被发现。
        )  # 完成四项总目标计算。
        self.assertEqual(float(total.detach()), 142.0)  # 正确手算为 1+2×2+45×3+0.5×4=142。
        total.backward()  # 反传后各输入标量梯度应等于自己的外层权重。
        self.assertEqual(float(adversarial.grad), 1.0)  # 对抗项权重为 1。
        self.assertEqual(float(feature_matching.grad), 2.0)  # 特征匹配项权重为 2。
        self.assertEqual(float(score_distillation.grad), 45.0)  # 蒸馏项权重为 45。
        self.assertEqual(float(cycle.grad), 0.5)  # Cycle 项只能乘一次 0.5。

    def test_augmented_loss_rejects_negative_cycle_weight(self) -> None:  # 负权重会奖励误差增大。
        """lambda_cycle 不能小于零。"""  # 非法超参数应在构造总损失时立即失败。

        scalar = torch.tensor(1.0)  # 四个损失位置复用同一个常数即可触发参数检查。
        with self.assertRaises(ValueError):  # 负 Cycle 权重必须由明确异常拒绝。
            cycle_augmented_student_loss(  # 调用正式总损失并故意传入负权重。
                scalar,  # 对抗项数值与本场景无关。
                scalar,  # 特征匹配项数值与本场景无关。
                scalar,  # 蒸馏项数值与本场景无关。
                scalar,  # Cycle 项数值与本场景无关。
                lambda_fm=2.0,  # 保持合法特征匹配权重。
                lambda_dist=45.0,  # 保持合法蒸馏权重。
                lambda_cycle=-0.1,  # 故意给出会反向奖励 Cycle 误差的负权重。
            )  # 正确实现应在返回张量前抛出 ValueError。

    def test_augmented_loss_rejects_nonfinite_cycle_weight(self) -> None:  # NaN 或无穷大都会污染总损失。
        """lambda_cycle 必须是有限数，不能依靠浮点比较漏过坏值。"""  # 覆盖 NaN 和正负无穷大。

        scalar = torch.tensor(1.0)  # 四个损失位置复用同一个合法常数，单独隔离权重错误。
        for bad_weight in (float("nan"), float("inf"), float("-inf")):  # 逐一检查三种非有限浮点值。
            with self.subTest(bad_weight=bad_weight):  # 子测试能明确报告具体漏过了哪一种坏值。
                with self.assertRaises(ValueError):  # 每一种非有限权重都必须在构造总损失前被拒绝。
                    cycle_augmented_student_loss(  # 调用正式总损失验证有限数门禁。
                        scalar,  # 对抗项保持合法。
                        scalar,  # 特征匹配项保持合法。
                        scalar,  # 蒸馏项保持合法。
                        scalar,  # Cycle 项保持合法。
                        lambda_fm=2.0,  # 保持合法特征匹配权重。
                        lambda_dist=45.0,  # 保持合法蒸馏权重。
                        lambda_cycle=bad_weight,  # 只把当前非有限值放到 Cycle 外层权重。
                    )  # 正确实现应在返回总损失张量前抛出 ValueError。


if __name__ == "__main__":  # 只有人工直接执行本文件时才启动 unittest。
    unittest.main()  # 被测试发现器导入时不会在这里重复启动。
