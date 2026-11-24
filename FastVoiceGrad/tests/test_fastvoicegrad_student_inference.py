# 本文件围绕 fastvoicegrad_student_inference 的生产接口进行小规模回归测试。
"""Unit tests for the formal FastVoiceGrad epoch-100 inference contract."""

# 延后解析类型标注，便于受支持的 Python 版本导入测试。
from __future__ import annotations

# 计算文件或 JSON 的 SHA-256，检测内容被替换。
import hashlib
# 读写实验身份、指纹和报告的 JSON。
import json
# 将故意损坏的测试文件隔离在自动清理目录中。
import tempfile
# 使用标准库测试框架组织断言和失败报告。
import unittest
# 统计每条源语音的参考配对数量。
from collections import Counter
# 组合临时测试文件和仓库路径。
from pathlib import Path

# 创建和读取极小的人工音频、梅尔或特征数组。
import numpy as np
# 写出合法 WAV，测试文件头正确但内容变化的情况。
import soundfile as sf
# 构造小张量和参考公式，不加载正式训练数据。
import torch
# 建立最小模型替身，核对接口和梯度。
import torch.nn as nn

# 导入本组测试实际调用的 fastvoicegrad.infer_paper_eval 接口；测试直接约束生产实现。
from fastvoicegrad.infer_paper_eval import (
    # 正式历史学生评估锁定的输出目录。
    DEFAULT_OUTPUT_DIR,
    # 正式历史学生评估锁定的第 100 轮。
    EXPECTED_STUDENT_EPOCH,
    # 正式历史学生评估锁定的 123000 次更新。
    EXPECTED_STUDENT_GLOBAL_STEP,
    # 一组源语音、目标参考语音和输出WAV的结构化记录。
    EvaluationPair,
    # 建立完成报告必须包含的系统、数据和权重身份字段。
    _report_required_fields,
    # 按100×9×9的固定规则建立8100组客观转换配对。
    build_pairs,
    # 计算规范化JSON的SHA-256，核对结构化实验身份。
    canonical_json_sha256,
    # 检查WAV音频头后，按pair_id记录每个生成文件的内容指纹。
    collect_generated_wav_hashes,
    # 在写WAV前检查已有目录的模型身份和配对血缘，安全恢复中断。
    prepare_output_lineage,
    # 得到发布到磁盘的缩进JSON精确字节，包含末尾换行。
    rendered_json_bytes,
    # 先把源mel扩散到950，再让学生只执行一次950→0重建。
    student_one_step,
    # 核验已有完成报告的身份及生成计数，防止错误跳过任务。
    validate_completed_report,
    # 确认推理声码器与学生训练使用的声码器文件逐字节一致。
    validate_vocoder_dependency_identity,
    # 限定学生专属目录，防止覆盖教师结果或混入诊断输出。
    validate_output_directory,
    # 检查学生权重归档格式、轮次、步数、契约和参数数值。
    validate_student_checkpoint_payload,
    # 核对原学生训练契约的指纹、协议、数据数量和模型设置。
    validate_student_run_contract,
)
# 导入本组测试实际调用的 voicegrad.diffusion 接口；测试直接约束生产实现。
from voicegrad.diffusion import VoiceGradDiffusion


# 构造含数据划分、教师来源、学生步数和声码器身份的最小运行契约。
def minimal_run_contract() -> dict:
    # 按本辅助函数的接口构造最小有效测试字典。
    return {
        # 明确区分FastVoiceGrad text-unseen学生实验；此处为测试使用的明确值。
        "experiment": "FastVoiceGrad VCTK text-unseen ACDD student",
        # 训练和测试划分摘要；此处为测试使用的明确值。
        "data": {
            # 当前数据清单声明的划分协议；此处为测试使用的明确值。
            "protocol": "text-unseen",
            # 测试集语音条数；此处为测试使用的明确值。
            "test_rows": 100,
            # 测试说话人数；此处为测试使用的明确值。
            "test_speakers": 10,
            # 本次学生训练预定总更新次数；此处为测试使用的明确值。
            "total_student_updates": EXPECTED_STUDENT_GLOBAL_STEP,
        },
        # 论文设定与实现解释的分组；此处为测试使用的明确值。
        "paper": {
            # 已在论文中明确给出的设定；此处为测试使用的明确值。
            "paper_explicit": {
                # 学生计划训练100轮；此处为测试使用的明确值。
                "epochs": EXPECTED_STUDENT_EPOCH,
                # 学生反向采样次数 K；此处为测试使用的明确值。
                "student_steps": 1,
                # 一步学生的加噪起始索引；此处为测试使用的明确值。
                "student_start_step": 950,
                # 扩散过程总噪声级数；此处为测试使用的明确值。
                "diffusion_steps": 1000,
                # 作为来源的教师训练轮次；此处为测试使用的明确值。
                "teacher_epoch": 500,
            }
        },
        # 论文未完整公开而由我明确登记的实现选择；此处为测试使用的明确值。
        "paper_undisclosed_declared_choices": {
            # BNF 到梅尔的帧对齐模式；此处为测试使用的明确值。
            "bnf_alignment": "legacy_length_linear",
            # ASR BNF 帧率降采样方式；此处为测试使用的明确值。
            "bnf_downsampling": "legacy_decimate",
            # 固定实验随机种子；此处为测试使用的明确值。
            "seed": 1234,
            # 要求的自训声码器训练步数；此处为测试使用的明确值。
            "expected_vocoder_step": 2_500_000,
        },
        # 冻结教师和声码器等外部依赖的身份；此处为测试使用的明确值。
        "dependencies": {"vocoder_global_step": 2_500_000},
    }


# 构造第 100 轮权重专用学生检查点，用极小张量检验加载规则。
def minimal_checkpoint(contract_sha256: str) -> dict:
    # 按本辅助函数的接口构造最小有效测试字典。
    return {
        # 检查点序列化格式版本；此处为测试使用的明确值。
        "format_version": 3,
        # 区分推理权重档案与训练恢复状态的类型标记；此处为测试使用的明确值。
        "checkpoint_type": "student_weights_only",
        # 当前完整训练轮次；此处为测试使用的明确值。
        "epoch": EXPECTED_STUDENT_EPOCH,
        # 检查点记录的累计优化器更新步数；此处为测试使用的明确值。
        "global_step": EXPECTED_STUDENT_GLOBAL_STEP,
        # 检查点绑定的训练契约规范化指纹；此处为测试使用的明确值。
        "run_contract_sha256": contract_sha256,
        # 该轮损失摘要，随权重归档保存；此处为测试使用的明确值。
        "epoch_losses": {"student_total": 1.25, "test_examples": 100},
        # 学生网络参数字典；此处为测试使用的明确值。
        "student": {"layer.weight": torch.ones(2, 2)},
    }


# 把同一类故障保护与期望行为组织成一组测试：CountingZeroStudent。
class CountingZeroStudent(nn.Module):
    # 初始化当前模型替身的参数和调用记录，不构造正式大网络。
    def __init__(self) -> None:
        # 初始化模块基类，确保参数和缓冲区可正确登记。
        super().__init__()
        # 学生前向调用次数，初始为零。
        self.calls = 0
        # 每次学生前向收到的噪声索引记录。
        self.timesteps = []

    # 提供与正式网络一致的前向接口，输出供本组回归测试核对。
    def forward(self, x, timesteps, speaker_embedding, bnf):
        # 记录一次学生前向调用，检查 K=1 未偷偷执行多步。
        self.calls += 1
        # 保存噪声索引独立副本，验证每条样本都在 t=950。
        self.timesteps.append(timesteps.detach().cpu().clone())
        # 返回同形状零噪声，使一步还原公式可手算。
        return torch.zeros_like(x)


# 构造跨说话人且不同句子的单个转换对，用于输出安全测试。
def one_pair(output_path: Path) -> EvaluationPair:
    # 封装源、参考、句子和输出位置构成的单条转换任务。
    return EvaluationPair(
        # 转换对在全量清单中的固定编号。
        pair_id=0,
        # 提供语音内容的源说话人。
        source_speaker="p001",
        # 源语音 ID。
        source_utt="p001_001",
        # 源语音的公共句子槽位。
        source_sentence_slot=1,
        # 源语音文本。
        source_text="ONE",
        # 源 WAV 路径。
        source_wav="/data/p001_001.wav",
        # 提供目标音色的参考说话人。
        reference_speaker="p002",
        # 参考语音 ID。
        reference_utt="p002_002",
        # 参考语音句子槽位，需与源句不同。
        reference_sentence_slot=2,
        # 参考语音文本。
        reference_text="TWO",
        # 参考 WAV 路径。
        reference_wav="/data/p002_002.wav",
        # 转换后的输出 WAV 路径。
        output_wav=str(output_path),
    )


# 构造不依赖真实模型文件的运行指纹集合。
def identity_fixture() -> dict:
    # 按本辅助函数的接口构造最小有效测试字典。
    return {
        # 系统机器标识，学生固定为fastvoicegrad；此处为测试使用的明确值。
        "system_id": "fastvoicegrad",
        # 报告中显示的学生名称与K=1、epoch100；此处为测试使用的明确值。
        "system_label": "FastVoiceGrad (K=1) epoch 100",
        # 明确标记student，防止与teacher评估混用；此处为测试使用的明确值。
        "model_role": "student",
        # 学生权重文件内容指纹。
        "student_checkpoint_sha256": "student-sha",
        # 学生训练契约指纹。
        "student_run_contract_sha256": "contract-sha",
        # 冻结声码器权重文件指纹。
        "vocoder_checkpoint_sha256": "vocoder-sha",
        # 声码器配置文件指纹。
        "vocoder_config_sha256": "config-sha",
        # 声码器最终归档证明指纹。
        "vocoder_attestation_sha256": "attestation-sha",
        # 测试语音清单指纹。
        "test_manifest_sha256": "manifest-sha",
        # seen/unseen 划分协议文件指纹。
        "split_protocol_sha256": "split-sha",
        # 梅尔均值文件指纹。
        "mel_mean_sha256": "mean-sha",
        # 梅尔标准差文件指纹。
        "mel_std_sha256": "std-sha",
    }


# 把同一类故障保护与期望行为组织成一组测试：StudentCheckpointContractTests。
class StudentCheckpointContractTests(unittest.TestCase):
    # 为每个测试创建新的隔离目录、模型替身和有效基准，避免前一例污染。
    def setUp(self) -> None:
        # 独立创建的学生运行契约。
        self.contract = minimal_run_contract()
        # 运行契约规范化 JSON 的 SHA-256 指纹。
        self.contract_sha = canonical_json_sha256(self.contract)

    # 确认固定的一步学生配置能够通过契约校验。
    def test_run_contract_accepts_the_fixed_one_step_settings(self) -> None:
        # 生产函数实际输出，供与独立参考比较。
        actual = validate_student_run_contract(
            # 使用未修改契约及其真实指纹，检查合法基准。
            self.contract, expected_sha256=self.contract_sha
        )
        # 核对 actual, contract_sha：应完全相等。
        self.assertEqual(actual, self.contract_sha)

    # 确认 30 步采样配置不能作为正式一步学生加载。
    def test_run_contract_rejects_a_thirty_step_student_label(self) -> None:
        # 故意改成 30 步，验证正式学生 K=1 契约。
        self.contract["paper"]["paper_explicit"]["student_steps"] = 30
        # 篡改契约后重新计算的指纹，用于单独检验语义。
        changed_sha = canonical_json_sha256(self.contract)
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "student_steps"):
            # 核对学生契约的设置与 SHA-256。
            validate_student_run_contract(
                # 传入已变更契约的新指纹，确保拒绝来自语义而非指纹失配。
                self.contract, expected_sha256=changed_sha
            )

    # 确认学生契约必须引用第 500 轮教师。
    def test_run_contract_rejects_a_different_teacher_epoch(self) -> None:
        # 故意引用第 499 轮教师，检验来源限制。
        self.contract["paper"]["paper_explicit"]["teacher_epoch"] = 499
        # 篡改契约后重新计算的指纹，用于单独检验语义。
        changed_sha = canonical_json_sha256(self.contract)
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "teacher_epoch"):
            # 核对学生契约的设置与 SHA-256。
            validate_student_run_contract(
                # 传入已变更契约的新指纹，确保拒绝来自语义而非指纹失配。
                self.contract, expected_sha256=changed_sha
            )

    # 确认正式推理只接受第 100 轮学生权重档案。
    def test_checkpoint_accepts_only_weights_only_epoch_100(self) -> None:
        # 通过身份检查后取出的学生权重。
        state = validate_student_checkpoint_payload(
            # 生成与当前契约绑定的第 100 轮权重档案。
            minimal_checkpoint(self.contract_sha),
            # 检查点绑定的训练契约规范化指纹，按下列值传给生产函数。
            run_contract_sha256=self.contract_sha,
        )
        # 核对 set(state), {"layer.weight"}：应完全相等。
        self.assertEqual(set(state), {"layer.weight"})

    # 确认训练恢复用完整状态不能误作正式推理归档。
    def test_checkpoint_rejects_latest_full_training_state(self) -> None:
        # 建立绑定当前契约的第 100 轮学生权重归档测试对象。
        checkpoint = minimal_checkpoint(self.contract_sha)
        # 将权重归档改成训练恢复类型，正式推理应拒绝。
        checkpoint["checkpoint_type"] = "full_training_state"
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "checkpoint_type"):
            # 检查学生归档身份与参数结构，无需加载大模型。
            validate_student_checkpoint_payload(
                # 按当前训练契约身份检查这份学生权重归档。
                checkpoint, run_contract_sha256=self.contract_sha
            )

    # 确认轮次和全局更新步同时满足正式学生身份。
    def test_checkpoint_rejects_wrong_epoch_or_step(self) -> None:
        # 分别测试错误轮次和少一次更新的全局步数。
        for key, value in (("epoch", 90), ("global_step", 122_999)):
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(key=key):
                # 建立绑定当前契约的第 100 轮学生权重归档测试对象。
                checkpoint = minimal_checkpoint(self.contract_sha)
                # 分别改变轮次或步数，检查训练进度身份。
                checkpoint[key] = value
                # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
                with self.assertRaisesRegex(RuntimeError, key):
                    # 检查学生归档身份与参数结构，无需加载大模型。
                    validate_student_checkpoint_payload(
                        # 按当前训练契约身份检查这份学生权重归档。
                        checkpoint, run_contract_sha256=self.contract_sha
                    )

    # 确认其他训练契约下的学生权重无法混入当前评估。
    def test_checkpoint_rejects_another_run_contract(self) -> None:
        # 建立绑定错误训练契约的学生归档，检验身份不一致被拒绝。
        checkpoint = minimal_checkpoint("other-contract")
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "run_contract_sha256"):
            # 检查学生归档身份与参数结构，无需加载大模型。
            validate_student_checkpoint_payload(
                # 按当前训练契约身份检查这份学生权重归档。
                checkpoint, run_contract_sha256=self.contract_sha
            )

    # 确认检查点含 NaN 或 Inf 权重时立即拒绝推理。
    def test_checkpoint_rejects_nonfinite_state(self) -> None:
        # 建立绑定当前契约的第 100 轮学生权重归档测试对象。
        checkpoint = minimal_checkpoint(self.contract_sha)
        # 向权重注入 NaN，测试加载前的有限性保护。
        checkpoint["student"]["layer.weight"][0, 0] = float("nan")
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
            # 检查学生归档身份与参数结构，无需加载大模型。
            validate_student_checkpoint_payload(
                # 按当前训练契约身份检查这份学生权重归档。
                checkpoint, run_contract_sha256=self.contract_sha
            )

    # 确认多出另一套模型字段的检查点结构被拒绝。
    def test_checkpoint_rejects_unexpected_schema(self) -> None:
        # 建立绑定当前契约的第 100 轮学生权重归档测试对象。
        checkpoint = minimal_checkpoint(self.contract_sha)
        # 插入额外模型字段，检查结构不能模糊匹配。
        checkpoint["model"] = checkpoint["student"]
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "top-level schema"):
            # 检查学生归档身份与参数结构，无需加载大模型。
            validate_student_checkpoint_payload(
                # 按当前训练契约身份检查这份学生权重归档。
                checkpoint, run_contract_sha256=self.contract_sha
            )


# 把同一类故障保护与期望行为组织成一组测试：OneStepMathTests。
class OneStepMathTests(unittest.TestCase):
    # 确认学生只在 t=950 调用一次，并按累计系数直接还原干净梅尔。
    def test_model_is_called_exactly_once_at_timestep_950(self) -> None:
        # 使用 1000 级、余弦偏移 0.008 的扩散过程，核对正式一步公式。
        diffusion = VoiceGradDiffusion(1000, 0.008)
        # 创建记录前向次数并预测零噪声的学生网络替身。
        student = CountingZeroStudent()
        # 测试源梅尔，维度为批数×80通道×时间帧。
        source = torch.linspace(-1.0, 1.0, 2 * 80 * 8).reshape(2, 80, 8)
        # 构造两条样本各 256 维的说话人条件，零噪声替身不依赖其数值。
        speaker = torch.zeros(2, 256)
        # 形状固定且数值可辨认的 BNF 测试输入。
        bnf = torch.zeros(2, 144, 8)
        # 与源梅尔同形状且数值固定的加噪张量。
        noise = torch.linspace(1.0, -1.0, source.numel()).reshape_as(source)

        # 生产函数实际输出，供与独立参考比较。
        actual = student_one_step(
            # 传入学生、日程、源梅尔、说话人条件、BNF 和固定源噪声。
            student, diffusion, source, speaker, bnf, noise
        )

        # 核对 student.calls, 1：应完全相等。
        self.assertEqual(student.calls, 1)
        # 核对 len(student.timesteps), 1：应完全相等。
        self.assertEqual(len(student.timesteps), 1)
        # 核对 torch.equal(student.timesteps[0], torch.tensor([950, 950]))：必须成立。
        self.assertTrue(torch.equal(student.timesteps[0], torch.tensor([950, 950])))
        # 批内每个样本的扩散索引。
        timesteps = torch.full((2,), 950, dtype=torch.long)
        # 源梅尔按正向过程加噪到 t=950 的张量。
        x_950 = diffusion.q_sample(source, timesteps, noise=noise)
        # 零噪声预测时，干净梅尔理论值就是 x_950 除以累计信号幅度。
        expected = x_950 / torch.sqrt(diffusion.alphas_cumprod[950])
        # 核对 actual, expected：应在容差内逐元素一致。
        torch.testing.assert_close(actual, expected)

    # 确认正式一步学生起点必须是 950。
    def test_any_other_start_step_is_rejected(self) -> None:
        # 使用 1000 级、余弦偏移 0.008 的扩散过程，核对正式一步公式。
        diffusion = VoiceGradDiffusion(1000, 0.008)
        # 创建记录前向次数并预测零噪声的学生网络替身。
        student = CountingZeroStudent()
        # 测试源梅尔，维度为批数×80通道×时间帧。
        source = torch.zeros(1, 80, 8)
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "must be 950"):
            # 调用正式 K=1 公式，不经过多步采样器。
            student_one_step(
                # 传入能统计调用次数且预测零噪声的学生替身。
                student,
                # 传入具有 1000 级噪声的余弦扩散日程。
                diffusion,
                # 传入测试源梅尔，维度为批数×80通道×时间帧。
                source,
                # 一条样本的 256 维零说话人条件。
                torch.zeros(1, 256),
                # 一条样本、144 内容通道、8 帧的零 BNF。
                torch.zeros(1, 144, 8),
                # 与源梅尔同形状的零源噪声。
                torch.zeros_like(source),
                # 故意传入错误起点 949，正式学生入口应要求 950 并拒绝。
                start_step=949,
            )


# 把同一类故障保护与期望行为组织成一组测试：PairProtocolTests。
class PairProtocolTests(unittest.TestCase):
    # 确认 100 条源语音各配 81 条异人异句参考，总数为 8100。
    def test_pair_grid_is_exactly_8100_and_81_per_source(self) -> None:
        # 从空记录列表开始逐项构造测试输入。
        rows = []
        # 枚举 10 位测试说话人。
        for speaker_index in range(10):
            # 按当前序号构造人工测试说话人 ID，后续为其生成十句语音。
            speaker = f"p{speaker_index:03d}"
            # 为每位说话人创建第 1 到第 10 个公共句子槽位。
            for slot in range(1, 11):
                # 把说话人 ID 与句子槽位组合成的语音 ID。
                utterance = f"{speaker}_{slot:03d}"
                # 加入当前说话人的当前句子，组成完整测试清单。
                rows.append(
                    {
                        # 当前测试原语音所属说话人。
                        "speaker": speaker,
                        # 语音的唯一身份 ID。
                        "utt_id": utterance,
                        # 当前数据属于训练还是测试。
                        "split": "test",
                        # 是否属于训练中未出现的测试说话人。
                        "is_unseen_speaker": "1",
                        # 语音转写文本或文本模式标记，具体取决于当前调用；此处为测试使用的明确值。
                        "text": f"SENTENCE {slot}",
                        # 22050 Hz 源音频路径。
                        "wav22k_path": f"/data/{utterance}.wav",
                    }
                )
        # 让生产配对器从人工 100 句生成异人异句的完整 8100 对清单。
        pairs = build_pairs(rows, Path("/evaluation/student"))
        # 核对 len(pairs), 8100：应完全相等。
        self.assertEqual(len(pairs), 8100)
        # 每条源语音出现的参考配对次数。
        counts = Counter(pair.source_utt for pair in pairs)
        # 核对 set(counts.values()), {81}：应完全相等。
        self.assertEqual(set(counts.values()), {81})
        # 核对下面完整给出的断言条件。
        self.assertTrue(
            # 要求清单中所有项目均满足当前身份约束。
            all(pair.source_speaker != pair.reference_speaker for pair in pairs)
        )
        # 核对下面完整给出的断言条件。
        self.assertTrue(
            # 要求下面的条件对集合中每个转换对都成立。
            all(
                # 源句子槽位必须不同于参考句子，防止同文本配对。
                pair.source_sentence_slot != pair.reference_sentence_slot
                # 遍历所有源参考转换对，逐项检查身份约束。
                for pair in pairs
            )
        )


# 把同一类故障保护与期望行为组织成一组测试：OutputSafetyTests。
class OutputSafetyTests(unittest.TestCase):
    # 确认学生推理绝不覆盖教师评估目录。
    def test_teacher_output_directory_is_always_rejected(self) -> None:
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "teacher directory"):
            # 检查学生输出位置，避免覆盖教师或其他实验。
            validate_output_directory(
                # 传入此处明确指定的测试文件位置，检验目录归属或报告路径约束。
                Path("/root/FsatVoiceGrad/exp/evaluation/voicegrad30_text_unseen_epoch0500_custom_hifigan"),
                # 是否启用正式全量评估的路径约束，按下列值传给生产函数。
                formal=False,
            )

    # 确认正式学生推理使用固定目录，避免实验身份混杂。
    def test_formal_output_directory_is_fixed(self) -> None:
        # 固定正式目录应通过学生输出位置检查。
        validate_output_directory(DEFAULT_OUTPUT_DIR, formal=True)
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "Formal student output"):
            # 检查学生输出位置，避免覆盖教师或其他实验。
            validate_output_directory(
                # 传入此处明确指定的测试文件位置，检验目录归属或报告路径约束。
                Path("/root/FsatVoiceGrad/exp/evaluation/another_student_run"),
                # 是否启用正式全量评估的路径约束，按下列值传给生产函数。
                formal=True,
            )

    # 确认目录已有音频时不能换运行身份后继续写入。
    def test_identity_drift_with_material_wav_fails_closed(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as temporary:
            # 模拟的学生推理输出目录。
            output = Path(temporary) / "student"
            # 学生输出目录中的音频子目录。
            wav_dir = output / "wav"
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            wav_dir.mkdir(parents=True)
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            (wav_dir / "old.wav").write_bytes(b"material")
            # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
            (output / "run_identity.json").write_text(
                # 按所列 JSON 编码参数写测试结构，保证后续指纹计算一致。
                json.dumps({"old": True}) + "\n", encoding="utf-8"
            )
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            (output / "pairs.csv").write_bytes(b"old-pairs")
            # 本测试的单条源参考转换对。
            pair = one_pair(wav_dir / "pair.wav")
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "another run"):
                # 调用输出身份与断点续推安全检查。
                prepare_output_lineage(
                    # 本次学生推理的专属输出目录，按下列值传给生产函数。
                    output_dir=output,
                    # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                    run_identity=identity_fixture(),
                    # 按固定顺序编码的配对CSV字节，按下列值传给生产函数。
                    pairs_payload=b"new-pairs",
                    # 本次需要生成的有序转换组列表，按下列值传给生产函数。
                    selected_pairs=[pair],
                )

    # 确认只有元数据、尚无音频的失败尝试可以安全刷新身份。
    def test_lineage_only_failed_attempt_can_refresh(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as temporary:
            # 模拟的学生推理输出目录。
            output = Path(temporary) / "student"
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            output.mkdir(parents=True)
            # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
            (output / "run_identity.json").write_text(
                # 按所列 JSON 编码参数写测试结构，保证后续指纹计算一致。
                json.dumps({"old": True}) + "\n", encoding="utf-8"
            )
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            (output / "pairs.csv").write_bytes(b"old-pairs")
            # 将模型、数据和预处理绑定的指纹字典。
            identity = identity_fixture()
            # 本测试的单条源参考转换对。
            pair = one_pair(output / "wav/pair.wav")
            # 取得完成标记、恢复标记、身份/配对指纹及强制重生成的 pair_id。
            complete, recovery, identity_sha, pairs_sha, forced = prepare_output_lineage(
                # 本次学生推理的专属输出目录，按下列值传给生产函数。
                output_dir=output,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                run_identity=identity,
                # 按固定顺序编码的配对CSV字节，按下列值传给生产函数。
                pairs_payload=b"new-pairs",
                # 本次需要生成的有序转换组列表，按下列值传给生产函数。
                selected_pairs=[pair],
            )
            # 核对 complete：必须不成立。
            self.assertFalse(complete)
            # 核对 recovery：必须不成立。
            self.assertFalse(recovery)
            # 核对 forced, set()：应完全相等。
            self.assertEqual(forced, set())
            # 核对下面完整给出的断言条件。
            self.assertEqual(
                # 传入运行身份 JSON 的字节指纹。
                identity_sha,
                # 对当前文件或 JSON 字节独立求 SHA-256，作为预期内容身份。
                hashlib.sha256(rendered_json_bytes(identity)).hexdigest(),
            )
            # 核对 pairs_sha, hashlib.sha256(b"new-pairs").hexdigest()：应完全相等。
            self.assertEqual(pairs_sha, hashlib.sha256(b"new-pairs").hexdigest())

    # 确认同一运行续推时保留已生成音频的字节内容。
    def test_partial_run_with_matching_identity_is_resumable(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as temporary:
            # 模拟的学生推理输出目录。
            output = Path(temporary) / "student"
            # 该转换对对应的输出 WAV 路径。
            destination = output / "wav/pair.wav"
            # 将模型、数据和预处理绑定的指纹字典。
            identity = identity_fixture()
            # 用固定字节代表本例单条配对清单，用于测试续推身份。
            pairs_payload = b"one-pair"
            # 本测试的单条源参考转换对。
            pair = one_pair(destination)
            # 调用输出身份与断点续推安全检查。
            prepare_output_lineage(
                # 本次学生推理的专属输出目录，按下列值传给生产函数。
                output_dir=output,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                run_identity=identity,
                # 按固定顺序编码的配对CSV字节，按下列值传给生产函数。
                pairs_payload=pairs_payload,
                # 本次需要生成的有序转换组列表，按下列值传给生产函数。
                selected_pairs=[pair],
            )
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            destination.parent.mkdir(parents=True, exist_ok=True)
            # 写入测试 WAV；即使格式合法，后续也必须通过内容指纹检查。
            sf.write(destination, np.zeros(2205, dtype=np.float32), 22050)
            # 续推前音频文件的 SHA-256，用于确认没有被重写。
            before = hashlib.sha256(destination.read_bytes()).hexdigest()

            # 读取续推状态和需重生成的转换对，忽略本例无需比较的两个指纹。
            complete, recovery, _, _, forced = prepare_output_lineage(
                # 本次学生推理的专属输出目录，按下列值传给生产函数。
                output_dir=output,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                run_identity=identity,
                # 按固定顺序编码的配对CSV字节，按下列值传给生产函数。
                pairs_payload=pairs_payload,
                # 本次需要生成的有序转换组列表，按下列值传给生产函数。
                selected_pairs=[pair],
            )

            # 核对 complete：必须不成立。
            self.assertFalse(complete)
            # 核对 recovery：必须不成立。
            self.assertFalse(recovery)
            # 核对 forced, set()：应完全相等。
            self.assertEqual(forced, set())
            # 核对 before, hashlib.sha256(destination.read_bytes()).hexdigest()：应完全相等。
            self.assertEqual(before, hashlib.sha256(destination.read_bytes()).hexdigest())

    # 确认完整结果可直接复用，合法 WAV 被替换内容后仍能按指纹进入恢复。
    def test_completed_run_is_noop_and_corrupt_wav_enters_recovery(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as temporary:
            # 模拟的学生推理输出目录。
            output = Path(temporary) / "student"
            # 该转换对对应的输出 WAV 路径。
            destination = output / "wav/pair.wav"
            # 将模型、数据和预处理绑定的指纹字典。
            identity = identity_fixture()
            # 用固定字节代表本例单条配对清单，用于测试续推身份。
            pairs_payload = b"one-pair"
            # 本测试的单条源参考转换对。
            pair = one_pair(destination)

            # 初始化同一运行身份，保存两个清单指纹以构造完成报告。
            complete, _, identity_sha, pairs_sha, forced = prepare_output_lineage(
                # 本次学生推理的专属输出目录，按下列值传给生产函数。
                output_dir=output,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                run_identity=identity,
                # 按固定顺序编码的配对CSV字节，按下列值传给生产函数。
                pairs_payload=pairs_payload,
                # 本次需要生成的有序转换组列表，按下列值传给生产函数。
                selected_pairs=[pair],
            )
            # 核对 complete：必须不成立。
            self.assertFalse(complete)
            # 核对 forced, set()：应完全相等。
            self.assertEqual(forced, set())
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            destination.parent.mkdir(parents=True, exist_ok=True)
            # 写入测试 WAV；即使格式合法，后续也必须通过内容指纹检查。
            sf.write(destination, np.zeros(2205, dtype=np.float32), 22050)
            # 构造当前单条转换的完成报告，绑定运行身份并登记生成与跳过数量。
            report = {
                # 复用生产推理器规定的必须字段，避免测试另造结构。
                **_report_required_fields(
                    # 传入将模型、数据和预处理绑定的指纹字典。
                    identity,
                    # 已发布运行身份文件的字节指纹，按下列值传给生产函数。
                    run_identity_sha256=identity_sha,
                    # 8100组转换配对清单的字节指纹，按下列值传给生产函数。
                    pairs_csv_sha256=pairs_sha,
                    # 本次实际选择的转换组数，按下列值传给生产函数。
                    selected_pair_count=1,
                ),
                # 本次实际新生成的WAV数量；此处为测试使用的明确值。
                "generated": 1,
                # 已经完整存在且身份一致、因而跳过的WAV数量；此处为测试使用的明确值。
                "skipped_existing": 0,
            }
            # 所有生成音频按 pair_id 记录的内容指纹。
            wav_hashes = collect_generated_wav_hashes([pair])
            # 在完成报告记录每条生成 WAV 的内容指纹。
            report["generated_wav_sha256"] = wav_hashes
            # 对全部音频指纹再求集合指纹，检测条目变化。
            report["generated_wav_collection_sha256"] = canonical_json_sha256(
                # 传入所有生成音频按 pair_id 记录的内容指纹。
                wav_hashes
            )
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            (output / "inference_report.json").write_bytes(
                # 按生产格式把完成报告编码成固定 JSON 字节。
                rendered_json_bytes(report)
            )

            # 读取续推状态和需重生成的转换对，忽略本例无需比较的两个指纹。
            complete, recovery, _, _, forced = prepare_output_lineage(
                # 本次学生推理的专属输出目录，按下列值传给生产函数。
                output_dir=output,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                run_identity=identity,
                # 按固定顺序编码的配对CSV字节，按下列值传给生产函数。
                pairs_payload=pairs_payload,
                # 本次需要生成的有序转换组列表，按下列值传给生产函数。
                selected_pairs=[pair],
            )
            # 核对 complete：必须成立。
            self.assertTrue(complete)
            # 核对 recovery：必须不成立。
            self.assertFalse(recovery)
            # 核对 forced, set()：应完全相等。
            self.assertEqual(forced, set())

            # Replacing the output with another perfectly valid WAV must still
            # be detected by its content hash, not merely by header checks.
            # 写入测试 WAV；即使格式合法，后续也必须通过内容指纹检查。
            sf.write(destination, np.ones(2205, dtype=np.float32) * 0.25, 22050)
            # 读取续推状态和需重生成的转换对，忽略本例无需比较的两个指纹。
            complete, recovery, _, _, forced = prepare_output_lineage(
                # 本次学生推理的专属输出目录，按下列值传给生产函数。
                output_dir=output,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                run_identity=identity,
                # 按固定顺序编码的配对CSV字节，按下列值传给生产函数。
                pairs_payload=pairs_payload,
                # 本次需要生成的有序转换组列表，按下列值传给生产函数。
                selected_pairs=[pair],
            )
            # 核对 complete：必须不成立。
            self.assertFalse(complete)
            # 核对 recovery：必须成立。
            self.assertTrue(recovery)
            # 核对 forced, {0}：应完全相等。
            self.assertEqual(forced, {0})
            # 核对 (output / "inference_report.json").exists()：必须不成立。
            self.assertFalse((output / "inference_report.json").exists())
            # 核对 (output / "inference_report.recovery.json").is_file()：必须成立。
            self.assertTrue((output / "inference_report.recovery.json").is_file())

            # Simulate a second interruption after the active report was moved
            # to the recovery marker but before the corrupt WAV was repaired.
            # 读取续推状态和需重生成的转换对，忽略本例无需比较的两个指纹。
            complete, recovery, _, _, forced = prepare_output_lineage(
                # 本次学生推理的专属输出目录，按下列值传给生产函数。
                output_dir=output,
                # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                run_identity=identity,
                # 按固定顺序编码的配对CSV字节，按下列值传给生产函数。
                pairs_payload=pairs_payload,
                # 本次需要生成的有序转换组列表，按下列值传给生产函数。
                selected_pairs=[pair],
            )
            # 核对 complete：必须不成立。
            self.assertFalse(complete)
            # 核对 recovery：必须成立。
            self.assertTrue(recovery)
            # 核对 forced, {0}：应完全相等。
            self.assertEqual(forced, {0})
            # 核对 (output / "inference_report.json").exists()：必须不成立。
            self.assertFalse((output / "inference_report.json").exists())
            # 核对 (output / "inference_report.recovery.json").is_file()：必须成立。
            self.assertTrue((output / "inference_report.recovery.json").is_file())

    # 确认生成数量与跳过数量不能用负数凑出完整覆盖。
    def test_negative_completion_counter_is_rejected(self) -> None:
        # 将模型、数据和预处理绑定的指纹字典。
        identity = identity_fixture()
        # 运行身份 JSON 的字节指纹。
        identity_sha = hashlib.sha256(rendered_json_bytes(identity)).hexdigest()
        # 构造当前单条转换的完成报告，绑定运行身份并登记生成与跳过数量。
        report = {
            # 复用生产推理器规定的必须字段，避免测试另造结构。
            **_report_required_fields(
                # 传入将模型、数据和预处理绑定的指纹字典。
                identity,
                # 已发布运行身份文件的字节指纹，按下列值传给生产函数。
                run_identity_sha256=identity_sha,
                # 8100组转换配对清单的字节指纹，按下列值传给生产函数。
                pairs_csv_sha256="pairs-sha",
                # 本次实际选择的转换组数，按下列值传给生产函数。
                selected_pair_count=1,
            ),
            # 本次实际新生成的WAV数量；此处为测试使用的明确值。
            "generated": -1,
            # 已经完整存在且身份一致、因而跳过的WAV数量；此处为测试使用的明确值。
            "skipped_existing": 2,
        }
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "completion counters"):
            # 核对报告身份、完整数量和输出指纹。
            validate_completed_report(
                # 传入读回或构造的可追溯评估报告。
                report,
                # 传入此处明确指定的测试文件位置，检验目录归属或报告路径约束。
                Path("inference_report.json"),
                # 将本次模型、数据、代码和输出绑定在一起的身份记录，按下列值传给生产函数。
                run_identity=identity,
                # 已发布运行身份文件的字节指纹，按下列值传给生产函数。
                run_identity_sha256=identity_sha,
                # 8100组转换配对清单的字节指纹，按下列值传给生产函数。
                pairs_csv_sha256="pairs-sha",
                # 本次实际选择的转换组数，按下列值传给生产函数。
                selected_pair_count=1,
            )


# 把同一类故障保护与期望行为组织成一组测试：VocoderIdentityTests。
class VocoderIdentityTests(unittest.TestCase):
    # 确认推理声码器权重、配置和证明文件均匹配学生训练契约。
    def test_vocoder_must_match_the_student_training_contract(self) -> None:
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as temporary:
            # 测试目录的 Path 对象。
            root = Path(temporary)
            # 人工声码器权重文件名标为 250 万步，供依赖指纹检查。
            checkpoint = root / "generator_step_2500000.pt"
            # 人工声码器配置文件的位置，稍后写入固定测试字节。
            config = root / "config.json"
            # 自训声码器最终证明文件路径。
            attestation = root / "final_checkpoint_attestation.json"
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            checkpoint.write_bytes(b"generator")
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            config.write_bytes(b"config")
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            attestation.write_bytes(b"attestation")
            # 构造学生训练时登记的声码器依赖路径和内容指纹。
            contract = {
                # 冻结教师和声码器等外部依赖的身份；此处为测试使用的明确值。
                "dependencies": {
                    # 本次冻结声码器的最终权重文件；此处为测试使用的明确值。
                    "vocoder_checkpoint": str(checkpoint),
                    # 声码器的结构与音频配置；此处为测试使用的明确值。
                    "vocoder_config": str(config),
                    # 声码器250万步最终检查点的审核记录；此处为测试使用的明确值。
                    "vocoder_attestation": str(attestation),
                    # 冻结声码器权重文件指纹。
                    "vocoder_checkpoint_sha256": hashlib.sha256(b"generator").hexdigest(),
                    # 声码器配置文件指纹。
                    "vocoder_config_sha256": hashlib.sha256(b"config").hexdigest(),
                    # 声码器最终归档证明指纹。
                    "vocoder_attestation_sha256": hashlib.sha256(b"attestation").hexdigest(),
                }
            }
            # 生产函数实际输出，供与独立参考比较。
            actual = validate_vocoder_dependency_identity(
                # 使用该学生的原始声码器依赖契约。
                contract,
                # 学生或声码器权重文件路径，按下列值传给生产函数。
                checkpoint_path=checkpoint,
                # 声码器结构配置文件路径，按下列值传给生产函数。
                config_path=config,
                # 声码器最终检查点审核文件路径，按下列值传给生产函数。
                attestation_path=attestation,
            )
            # 核对下面完整给出的断言条件。
            self.assertEqual(
                # 取实际声码器权重文件计算得到的指纹。
                actual["vocoder_checkpoint_sha256"],
                # 取学生训练时登记的声码器权重指纹，要求完全相同。
                contract["dependencies"]["vocoder_checkpoint_sha256"],
            )

            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            checkpoint.write_bytes(b"changed")
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "differs"):
                # 将声码器三份实际文件与训练登记指纹核对。
                validate_vocoder_dependency_identity(
                    # 使用该学生的原始声码器依赖契约。
                    contract,
                    # 学生或声码器权重文件路径，按下列值传给生产函数。
                    checkpoint_path=checkpoint,
                    # 声码器结构配置文件路径，按下列值传给生产函数。
                    config_path=config,
                    # 声码器最终检查点审核文件路径，按下列值传给生产函数。
                    attestation_path=attestation,
                )


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
