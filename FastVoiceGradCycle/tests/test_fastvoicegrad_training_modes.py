# 检查修复后的目标和权重选择确实传到训练、测试及损失监控。
"""Check repaired distillation modes reach both training and diagnostic evaluation."""

# 复制冻结配置时只替换本测试需要的目标和权重，不改全局默认。
from dataclasses import replace
# 捕获预期的参数报错，避免测试输出被完整命令行帮助刷屏。
from contextlib import redirect_stderr
# 把参数解析器写到stderr的错误信息收集为可检查的内存文本。
import io
# 穷举两个目标和三个权重，确保混合组合也需要显式对照许可。
import itertools
# 临时替换启动参数，直接检查正式parse_args入口而不启动真实训练。
import sys
# 使用标准单元测试检查蒸馏公式，所有张量计算在租用服务器执行。
import unittest
# 用替身监控器记录调用次数，并包装真实损失函数检查传参。
from unittest.mock import Mock, patch

# 使用张量和自动求导验证反向扩散目标与梯度。
import torch

# 控制模型参数是否可求导，验证冻结声码器的训练边界。
from fastvoicegrad.components import set_requires_grad
# 从与正式训练相同的配置建立修复实验。
from fastvoicegrad.config import PAPER_STUDENT_CONFIG
# 测试直接调用正式蒸馏函数，检查的就是训练所用计算路径。
from fastvoicegrad.objectives import score_distillation_loss
# 保留训练模块引用，以检查真实命令行入口及模型加载前的拒绝顺序。
import fastvoicegrad.train as training
# 同时测试训练更新和固定测试损失两条调用路径。
from fastvoicegrad.train import evaluate_test_epoch, train_epoch
# 导入只含少量参数的网络替身，低成本检查完整训练流程。
from tests.test_fastvoicegrad_student_contract import (
    # 分别模拟波形判别器、噪声预测器和可微声码器。
    TinyDiscriminator, TinyEpsilon, TinyVocoder,
)
# 使用训练中的同一余弦扩散日程，避免测试另写一套系数。
from voicegrad.diffusion import VoiceGradDiffusion


# 命令行必须默认采用修复公式；所有历史或混合组合只能作为显式对照进入。
class ScoreModeCliTests(unittest.TestCase):
    # 封装sys.argv替换，不创建模型、优化器、日志目录或运行锁。
    def parse(self, *arguments):
        # 使用与python -m fastvoicegrad.train一致的参数布局。
        with patch.object(sys, "argv", ["fastvoicegrad.train", *arguments]):
            # 调用生产解析器，防止测试另写一套与正式入口不同的规则。
            return training.parse_args()

    # 省略两个模式时，配置实例和命令行必须落到同一个修复组合。
    def test_cli_defaults_match_repaired_config(self):
        # 仅解析默认参数，不进入main，也不检查磁盘模型是否存在。
        args = self.parse()
        # 配置本身应以干净重建和累计信号幅度作为本轮修复默认。
        self.assertEqual(
            # 先核对配置，防止配置和CLI一起退回旧值而蒙混通过。
            (PAPER_STUDENT_CONFIG.score_target, PAPER_STUDENT_CONFIG.score_weight),
            # 两个名称分别固定目标空间和权重解释。
            ("clean_x0", "sqrt_alpha_bar"),
        )
        # 再核对CLI确实引用同一默认配置。
        self.assertEqual(
            # 命令行最终解析出的两个字段。
            (args.score_target, args.score_weight),
            # 与训练契约使用的配置实例保持一致。
            (PAPER_STUDENT_CONFIG.score_target, PAPER_STUDENT_CONFIG.score_weight),
        )
        # 不提供开关时不能自动赋予旧公式或混合公式的使用许可。
        self.assertFalse(args.allow_score_ablation)

    # 显式写出修复模式也属于正常启动，无须额外声明为对照。
    def test_explicit_repaired_combination_needs_no_ablation_flag(self):
        # 模拟正式启动脚本同时写出两个修复参数。
        args = self.parse("--score-target", "clean_x0", "--score-weight", "sqrt_alpha_bar")
        # 参数应被原样保留，供后面生成本次运行契约。
        self.assertEqual((args.score_target, args.score_weight), ("clean_x0", "sqrt_alpha_bar"))
        # 正式修复组合不依赖对照开关。
        self.assertFalse(args.allow_score_ablation)

    # 任何非默认组合都必须主动许可，包括只改权重、只改目标及两者都改。
    def test_all_nondefault_combinations_require_explicit_permission(self):
        # 枚举全部六种组合，而不只覆盖历史的相邻步加单步权重。
        for target, weight in itertools.product(
            # 两种目标空间与三种权重形成完整笛卡尔积。
            ("clean_x0", "legacy_adjacent"), ("sqrt_alpha_bar", "alpha_bar", "nominal_alpha")
        ):
            # 修复默认组合由独立测试确认可直接接受。
            if (target, weight) == ("clean_x0", "sqrt_alpha_bar"):
                # 本项仅检查剩下的五种受保护组合。
                continue
            # 用模式名称标记失败场景，便于确认是否漏拦某一个混合组合。
            with self.subTest(target=target, weight=weight):
                # 收集错误文字，确保报错确实说明缺少对照许可。
                stderr = io.StringIO()
                # argparse应该在解析阶段以标准状态2拒绝，不能等模型加载后才失败。
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    # 故意不传许可，检验普通训练命令无法悄悄进入非默认公式。
                    self.parse("--score-target", target, "--score-weight", weight)
                # 状态2代表参数使用错误，而不是后续训练运行失败。
                self.assertEqual(raised.exception.code, 2)
                # 错误应指出所需开关，方便使用者分清正式训练与有意对照。
                self.assertIn("--allow-score-ablation", stderr.getvalue())
                # 这次只增加许可，不改变目标与权重，用于确认历史回放仍可显式进入。
                accepted = self.parse(
                    # 保留刚刚被拒绝的同一组合。
                    "--score-target", target, "--score-weight", weight,
                    # 显式声明这是另行记录的对照选择。
                    "--allow-score-ablation",
                )
                # 允许进入后也必须保留准确参数，不能替换成默认再假装完成回放。
                self.assertEqual((accepted.score_target, accepted.score_weight), (target, weight))
                # 许可字段供后续记录使用，必须明确为真。
                self.assertTrue(accepted.allow_score_ablation)

    # 单独覆盖只写一个旧参数的复制旧命令场景，另一个参数会由默认值补齐。
    def test_partial_nondefault_override_is_rejected(self):
        # 列出三种可把默认组合变成非默认组合的单字段修改。
        for arguments in (
            # 只把目标改回相邻含噪状态。
            ("--score-target", "legacy_adjacent"),
            # 只把权重改回单步信号保留量。
            ("--score-weight", "nominal_alpha"),
            # 只把幅度权重改成累计功率权重。
            ("--score-weight", "alpha_bar"),
        ):
            # 在失败提示中保留完整单字段命令。
            with self.subTest(arguments=arguments):
                # 抑制预期帮助文本，只核对解析阶段的退出行为。
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    # 没有显式许可时，这些参数都不应进入训练准备阶段。
                    self.parse(*arguments)
                # 单字段覆盖同样要按参数错误退出。
                self.assertEqual(raised.exception.code, 2)

    # 对照许可只允许已知公式，不能使拼错的模式名称绕过choices校验。
    def test_ablation_permission_does_not_allow_unknown_modes(self):
        # 分别检查目标和权重名称的拼写错误。
        for arguments in (
            # 目标名错误，权重将采用有效默认值。
            ("--score-target", "clean_typo"),
            # 权重名错误，目标将采用有效默认值。
            ("--score-weight", "sqrt_typo"),
        ):
            # 保留失败参数，方便区分哪一个choices校验被绕过。
            with self.subTest(arguments=arguments):
                # 即使允许对照，未知名称也仍是参数错误。
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    # 对所有故意拼错的命令都添加许可，证明许可作用范围有限。
                    self.parse(*arguments, "--allow-score-ablation")
                # 标准参数错误状态为2。
                self.assertEqual(raised.exception.code, 2)

    # 真实main必须在环境检查、加锁和加载模型之前拒绝未许可的历史目标。
    def test_main_rejects_implicit_legacy_run_before_runtime_setup(self):
        # 构造复制旧训练参数却遗漏许可开关的启动命令。
        arguments = ["fastvoicegrad.train", "--score-target", "legacy_adjacent", "--score-weight", "nominal_alpha"]
        # 替换启动参数，仅触发main最前面的参数解析。
        with patch.object(sys, "argv", arguments):
            # 监视环境检查；它不应在参数错误时执行。
            with patch.object(training, "assert_rented_server") as runtime_guard:
                # 监视独占锁创建，错误命令不能占住其他任务的运行锁。
                with patch.object(training, "acquire_runtime_locks") as runtime_locks:
                    # 监视教师和学生加载，错误命令不能读取检查点或消耗显存。
                    with patch.object(training, "load_teacher_and_student") as load_models:
                        # 捕获main由argparse发出的预期退出，禁止进入后续训练路径。
                        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                            # 使用正式main入口，证明保护位于真正的启动链最前面。
                            training.main()
                        # 仍应是参数解析的标准错误状态。
                        self.assertEqual(raised.exception.code, 2)
                        # 拒绝命令后不得执行服务器身份检查。
                        runtime_guard.assert_not_called()
                        # 不得创建训练独占锁。
                        runtime_locks.assert_not_called()
                        # 不得加载教师、学生或对应检查点。
                        load_models.assert_not_called()


# 集成测试确认修复参数能从配置一路传到实际损失与监控器。
class TrainingModeIntegrationTests(unittest.TestCase):
    # 检查修复配置是否同时传入训练、测试损失与监控器。
    def test_repaired_objective_reaches_training_evaluation_and_monitor(self):
        # 固定本测试的随机输入和网络初始参数。
        torch.manual_seed(103)
        # 以原配置为基础创建局部配置副本。
        config = replace(
            # 教师目标选择直接还原干净x0，避免旧相邻步噪声残留。
            PAPER_STUDENT_CONFIG, score_target="clean_x0",
            # 损失采用sqrt(alpha_bar)信号幅度加权。
            score_weight="sqrt_alpha_bar",
        )
        # 教师替身关闭训练参数并使用评估模式。
        teacher = TinyEpsilon(trainable=False).eval()
        # 学生替身保持训练模式，允许本批更新。
        student = TinyEpsilon().train()
        # 声码器替身使用评估模式，仍允许梯度通过输入。
        vocoder = TinyVocoder().eval()
        # 冻结声码器参数，检查训练只改变学生。
        set_requires_grad(vocoder, False)
        # 建立需要与学生交替更新的判别器替身。
        discriminator = TinyDiscriminator().train()
        # 使用真实T=1000余弦扩散系数。
        diffusion = VoiceGradDiffusion()
        # 均值0、标准差1使反归一化成为恒等变换，隔离目标传参问题。
        mean, std = torch.zeros(1, 80, 1), torch.ones(1, 80, 1)
        # 小测试在租用服务器CPU执行，无须占GPU。
        device = torch.device("cpu")

        # 生成指定批量和样本编号偏移的合成训练输入。
        def batch(size, offset):
            # 返回与生产DataLoader相同的字段结构。
            return {
                # mel形状为[批量,80,128]，符合正式训练接口。
                "mel": torch.randn(size, 80, 128),
                # BNF形状为[批量,144,128]，时间轴与mel对齐。
                "bnf": torch.randn(size, 144, 128),
                # 每条样本提供256维说话人向量。
                "spk_emb": torch.randn(size, 256),
                # 用可控说话人编号构造100条测试数据的身份分组。
                "spk_name": [f"p{i // 10}" for i in range(offset, offset + size)],
                # 句子编号随偏移变化，避免不同批次重用同一身份。
                "utt_id": [f"u{i}" for i in range(offset, offset + size)],
            }

        # 用Mock监控器检查每次参数更新是否及时上报损失。
        monitor = Mock()
        # 保存学生更新前的参数，稍后确认优化器确实执行。
        before = student.scale.detach().clone()
        # 包装真实蒸馏函数，既运行数学公式又能检查传入的目标/权重。
        with patch("fastvoicegrad.train.score_distillation_loss", wraps=score_distillation_loss) as objective:
            # 仅运行一个完整batch32，核对训练集成行为。
            losses, step = train_epoch(
                # 从epoch1、step0开始，传入同一组教师、学生、声码器和判别器。
                [batch(32, 0)], 1, 0, teacher, student, vocoder, discriminator,
                # 学生使用学习率0.0002的独立Adam。
                diffusion, torch.optim.Adam(student.parameters(), lr=2e-4),
                # 判别器使用自己的Adam，避免两套参数共享更新状态。
                torch.optim.Adam(discriminator.parameters(), lr=2e-4),
                # 传入标准化统计、设备、修复配置及监控器。
                mean, std, device, config, monitor=monitor,
            )
            # 一个批次结束后，全局步数应从0增加到1。
            self.assertEqual(step, 1)
            # 学生参数必须改变，防止损失有数值却没有发生有效学习。
            self.assertFalse(torch.equal(before, student.scale.detach()))
            # 这一个训练批次只应调用一次蒸馏损失。
            self.assertEqual(objective.call_count, 1)
            # 检查训练路径实际收到的两个模式关键字。
            self.assertEqual(objective.call_args.kwargs, {
                # 明确要求干净目标和平方根累计权重，没有退回旧默认。
                "target_mode": "clean_x0", "weight_mode": "sqrt_alpha_bar",
            })
            # 监控器必须恰好收到当前step、epoch和训练损失。
            monitor.observe.assert_called_once_with(1, 1, losses)
            # 教师全部参数梯度应保持None。
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            # 冻结声码器参数梯度同样必须保持None。
            self.assertTrue(all(p.grad is None for p in vocoder.parameters()))

            # 清空训练阶段调用记录，独立检查测试阶段。
            objective.reset_mock()
            # 调用固定测试损失评估，检查其模式选择与训练一致。
            _, examples, batches = evaluate_test_epoch(
                # 100条样本分为32、32、32、4四批，末批不能丢失。
                [batch(32, 0), batch(32, 32), batch(32, 64), batch(4, 96)],
                # 测试使用同一组网络和扩散日程。
                teacher, student, vocoder, discriminator, diffusion,
                # 传入同样的统计量、设备和修复配置。
                mean, std, device, config,
            )
            # 固定测试应覆盖全部100条语音，组成4批。
            self.assertEqual((examples, batches), (100, 4))
            # 测试阶段每批各调用一次蒸馏目标，共4次。
            self.assertEqual(objective.call_count, 4)
            # 逐批核对测试阶段的参数，防止只有首批正确。
            for call in objective.call_args_list:
                # 测试函数的关键字必须继续保留修复设定。
                self.assertEqual(call.kwargs, {
                    # 确认测试也使用clean_x0与sqrt_alpha_bar。
                    "target_mode": "clean_x0", "weight_mode": "sqrt_alpha_bar",
                })

    # 检查实现解释和论文公开设定在运行记录中保持清楚区分。
    def test_distillation_interpretation_is_disclosed(self):
        # 取出公开论文设定列表。
        paper = PAPER_STUDENT_CONFIG.paper_explicit()
        # 取出本项目显式登记的复现选择列表。
        choices = PAPER_STUDENT_CONFIG.declared_reproduction_choices()
        # 目标解释、具体权重实现和报警窗口都是需要披露的复现选择。
        for key in ("score_target", "score_weight", "loss_monitor_window_steps"):
            # 它们不能直接混入2024论文明确设定列表。
            self.assertNotIn(key, paper)
            # 复现选择记录必须覆盖这些字段，保证报告可追溯。
            self.assertIn(key, choices)
        # 论文原文的alpha记号仍应在公开设定中保留。
        self.assertIn("paper_score_weight_notation", paper)


# 仅在直接运行模块时进入诊断或单元测试，导入时不执行计算。
if __name__ == "__main__":
    # 运行本文件的测试并用退出码报告通过或失败。
    unittest.main()
