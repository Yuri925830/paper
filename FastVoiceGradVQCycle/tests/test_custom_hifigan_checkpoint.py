# 本文件围绕 custom_hifigan_checkpoint 的生产接口进行小规模回归测试。
"""Unit tests for the fail-closed custom HiFi-GAN archive attestation."""

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
# 组合临时测试文件和仓库路径。
from pathlib import Path
# 导入本组测试实际调用的 typing 接口；测试直接约束生产实现。
from typing import Any, Dict, Optional
# 临时替换依赖或注入故障，测试后自动恢复。
from unittest import mock

# 构造小张量和参考公式，不加载正式训练数据。
import torch

# 导入本组测试实际调用的 hifigan.custom 接口；测试直接约束生产实现。
from hifigan.custom import validate_checkpoint as validator


# 把同一类故障保护与期望行为组织成一组测试：_AttrDict。
class _AttrDict(dict):
    # 初始化当前模型替身的参数和调用记录，不构造正式大网络。
    def __init__(self, *args, **kwargs):
        # 让替身字典保留普通 dict 的构造行为。
        super().__init__(*args, **kwargs)
        # 使配置字典也能按 config.xxx 访问，匹配官方配置接口。
        self.__dict__ = self


# 把同一类故障保护与期望行为组织成一组测试：_TinyGenerator。
class _TinyGenerator(torch.nn.Module):
    # 记录每次 state_dict 加载是否启用 strict。
    strict_arguments = []

    # 初始化当前模型替身的参数和调用记录，不构造正式大网络。
    def __init__(self, _config):
        # 初始化模块基类，确保参数和缓冲区可正确登记。
        super().__init__()
        # 创建仅含两个标量的可训练权重，足够测试严格加载。
        self.weight = torch.nn.Parameter(torch.zeros(2))
        # 登记一个必须随检查点保存、但不参加优化的缩放缓冲量。
        self.register_buffer("scale", torch.ones(1))

    # 记录 strict 标志再使用 PyTorch 严格加载逻辑。
    def load_state_dict(self, state_dict, strict=True):
        # 记录加载调用的 strict 参数，以验证生产代码没有宽松加载。
        type(self).strict_arguments.append(strict)
        # 把参数交给真实 PyTorch 加载器检查缺失键、额外键和形状。
        return super().load_state_dict(state_dict, strict=strict)


# 构造满足锁定结构和数据来源要求的声码器配置。
def _valid_config() -> Dict[str, Any]:
    # 复制验证器要求的声码器结构参数，作为合法配置基准。
    config = dict(validator.REQUIRED_CONFIG_VALUES)
    # 补充实验名、数据清单指纹及未公开选择说明，构成完整配置。
    config.update(
        {
            # 声码器实验名称；此处为测试使用的明确值。
            "experiment_name": "unit_test_custom_hifigan",
            # 训练、测试和划分清单的固定指纹；此处为测试使用的明确值。
            "data_fingerprints": {
                # 训练使用的文本划分协议指纹。
                "split_protocol.json": "a" * 64,
                # 测试语音清单指纹。
                "test_manifest.csv": "b" * 64,
                # 训练语音清单指纹。
                "train_manifest.csv": "c" * 64,
            },
            # 论文未公开实现选择的说明；此处为测试使用的明确值。
            "undisclosed_choice_notes": {"max_steps": "explicit reconstruction choice"},
        }
    )
    # 返回可独立保存并与归档内部配置核对的配置字典。
    return config


# 构造完整声码器权重归档，并附带配置语义指纹。
def _valid_payload(config: Dict[str, Any], step: int) -> Dict[str, Any]:
    # 按本辅助函数的接口构造最小有效测试字典。
    return {
        # 检查点序列化格式版本；此处为测试使用的明确值。
        "format_version": validator.EXPECTED_FORMAT_VERSION,
        # 区分推理权重档案与训练恢复状态的类型标记；此处为测试使用的明确值。
        "checkpoint_type": validator.EXPECTED_CHECKPOINT_TYPE,
        # 声码器生成器和判别器结构标识；此处为测试使用的明确值。
        "architecture": validator.EXPECTED_ARCHITECTURE,
        # 检查点记录的累计优化器更新步数；此处为测试使用的明确值。
        "global_step": step,
        # 当前完整训练轮次；此处为测试使用的明确值。
        "epoch": 1015,
        # 当前轮次内已完成批位置；此处为测试使用的明确值。
        "batch_in_epoch": 2085,
        # 归档嵌入的完整声码器配置；此处为测试使用的明确值。
        "config": config,
        # 忽略 JSON 键顺序后的完整配置指纹；此处为测试使用的明确值。
        "config_semantic_sha256": validator.canonical_sha256(config),
        # 训练、测试和划分清单的固定指纹；此处为测试使用的明确值。
        "data_fingerprints": dict(config["data_fingerprints"]),
        # 声码器生成器参数字典；此处为测试使用的明确值。
        "generator": {
            # 极小生成器的可训练权重；此处为测试使用的明确值。
            "weight": torch.tensor([1.0, -1.0]),
            # 随检查点保存的测试缓冲量；此处为测试使用的明确值。
            "scale": torch.tensor([1.0]),
        },
    }


# 把同一类故障保护与期望行为组织成一组测试：CustomHiFiGANCheckpointTests。
class CustomHiFiGANCheckpointTests(unittest.TestCase):
    # 为每个测试创建新的隔离目录、模型替身和有效基准，避免前一例污染。
    def setUp(self) -> None:
        # 清空上一例的 strict 调用记录，避免测试互相污染。
        _TinyGenerator.strict_arguments.clear()
        # 每个测试独占的自动清理目录。
        self.temporary = tempfile.TemporaryDirectory()
        # 为每个测试保存测试目录的 Path 对象。
        self.root = Path(self.temporary.name)
        # 本轮归档测试要求的 2,500,000 更新步。
        self.step = 2_500_000
        # 为每个测试保存声码器配置文件或配置对象。
        self.config = _valid_config()
        # 声码器独立配置文件位置。
        self.config_path = self.root / "hifigan" / "configs" / "text_unseen.json"
        # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
        self.config_path.parent.mkdir(parents=True)
        # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
        self.config_path.write_text(
            # 按所列 JSON 编码参数写测试结构，保证后续指纹计算一致。
            json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 模拟官方声码器源码的目录。
        official_dir = self.root / "hifigan" / "official"
        # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
        official_dir.mkdir(parents=True)
        # 逐项处理 source_name in ("env.py", "models.py", "utils.py")，覆盖这组反例的全部输入。
        for source_name in ("env.py", "models.py", "utils.py"):
            # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
            (official_dir / source_name).write_text(
                # 为每个锁定源码文件创建可哈希的占位内容，测试不执行这些占位脚本。
                f"# unit-test placeholder for {source_name}\n", encoding="utf-8"
            )
        # 按要求步数命名的声码器归档位置。
        self.checkpoint_path = self.root / validator.expected_checkpoint_filename(self.step)
        # 把当前人工权重和元数据保存为真实 PyTorch 归档，检验读取路径。
        torch.save(_valid_payload(self.config, self.step), self.checkpoint_path)
        # 把大型生成器导入替换成极小模型的临时补丁。
        self.import_patch = mock.patch.object(
            # 被替换或监视的声码器归档验证模块。
            validator,
            # 替换官方声码器导入函数，使测试只构造极小替身。
            "import_official_hifigan",
            # 替身调用返回的测试对象，按下列值传给生产函数。
            return_value=(_AttrDict, _TinyGenerator),
        )
        # 启用极小生成器替身；不实际加载大型声码器网络。
        self.import_patch.start()

    # 恢复临时替换的依赖并清理本例独占的测试目录。
    def tearDown(self) -> None:
        # 恢复生产导入入口，避免替身泄漏到其他测试。
        self.import_patch.stop()
        # 清理本例独占临时目录，保留真实项目文件不动。
        self.temporary.cleanup()

    # 把当前反例写成真实检查点文件，以覆盖生产读盘校验。
    def _save_payload(self, payload: Dict[str, Any], name: Optional[str] = None) -> Path:
        # 显式给文件名时使用反例名称，否则按要求步数生成合法归档名。
        path = self.root / (name or validator.expected_checkpoint_filename(self.step))
        # 把当前人工权重和元数据保存为真实 PyTorch 归档，检验读取路径。
        torch.save(payload, path)
        # 返回已写入文件路径，供生产校验或比较函数读取。
        return path

    # 确认声码器严格加载成功后才原子发布完整证明。
    def test_success_strict_loads_and_atomically_attests(self) -> None:
        # 声码器归档验证成功后，证明写到本例 reports 目录。
        report_path = self.root / "reports" / "custom_hifigan.json"
        # 监视真实 torch.load 的调用参数，同时保持真实加载行为。
        with mock.patch.object(torch, "load", wraps=torch.load) as load_mock:
            # 执行完整声码器归档验证，并取得成功后发布的证明内容。
            report = validator.attest_checkpoint(
                # 测试用项目根目录，按下列值传给生产函数。
                project_root=self.root,
                # 待验证声码器归档路径，按下列值传给生产函数。
                checkpoint_path=self.checkpoint_path,
                # 独立保存的配置路径，按下列值传给生产函数。
                config_path=self.config_path,
                # 本次要求归档达到的训练步数，按下列值传给生产函数。
                expected_step=self.step,
                # 仅验证成功后发布的证明路径，按下列值传给生产函数。
                output_report=report_path,
            )

        # 确认归档恰好读取一次，并使用指定 CPU 和可信归档选项。
        load_mock.assert_called_once_with(
            # 要求解析绝对路径，在 CPU 读取项目受信任的完整归档内容。
            self.checkpoint_path.resolve(), map_location="cpu", weights_only=False
        )
        # 核对 _TinyGenerator.strict_arguments, [True]：应完全相等。
        self.assertEqual(_TinyGenerator.strict_arguments, [True])
        # 核对 report[“是否通过全部参数名和形状的严格加载”]：必须成立。
        self.assertTrue(report["generator_strict_load"])
        # 核对 report[“检查点记录的累计优化器更新步数”], step：应完全相等。
        self.assertEqual(report["global_step"], self.step)
        # 核对 report[“当前完整训练轮次”], 1015：应完全相等。
        self.assertEqual(report["epoch"], 1015)
        # 核对 report[“当前轮次内已完成批位置”], 2085：应完全相等。
        self.assertEqual(report["batch_in_epoch"], 2085)
        # 核对下面完整给出的断言条件。
        self.assertEqual(
            # 取证明文件记录的归档内容指纹，与独立求值比较。
            report["checkpoint_sha256"],
            # 对当前文件或 JSON 字节独立求 SHA-256，作为预期内容身份。
            hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest(),
        )
        # 核对 json.loads(report_path.read_text(encoding="utf-8")), report：应完全相等。
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)
        # 删除自身签名后的证明报告副本。
        unsigned = dict(report)
        # 证明文件中先前登记的内容签名。
        recorded_payload_hash = unsigned.pop("attestation_payload_sha256")
        # 核对 recorded_payload_hash, validator.canonical_sha256(unsigned)：应完全相等。
        self.assertEqual(recorded_payload_hash, validator.canonical_sha256(unsigned))
        # 核对 list(report_path.parent.glob("*.tmp")), []：应完全相等。
        self.assertEqual(list(report_path.parent.glob("*.tmp")), [])

    # 逐项检查格式版本、架构、检查点类型和步数不一致会被拒绝。
    def test_rejects_each_schema_metadata_mismatch(self) -> None:
        # 逐个隔离检查的非法输入组合。
        cases = (
            # 把检查点序列化格式版本改为该行给出的非法值，独立测试拒绝路径。
            ("format_version", 1, "format_version mismatch"),
            # 把检查点序列化格式版本改为该行给出的非法值，独立测试拒绝路径。
            ("format_version", True, "format_version mismatch"),
            # 把区分推理权重档案与训练恢复状态的类型标记改为该行给出的非法值，独立测试拒绝路径。
            ("checkpoint_type", "full_training_state", "checkpoint_type mismatch"),
            # 把声码器生成器和判别器结构标识改为该行给出的非法值，独立测试拒绝路径。
            ("architecture", "HiFi-GAN V1", "architecture mismatch"),
            # 把检查点记录的累计优化器更新步数改为该行给出的非法值，独立测试拒绝路径。
            ("global_step", self.step - 1, "global_step mismatch"),
        )
        # 逐项处理 key, bad_value, message in cases，覆盖这组反例的全部输入。
        for key, bad_value, message in cases:
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(key=key, value=bad_value):
                # 创建与独立配置、数据来源和目标步数一致的声码器归档字典。
                payload = _valid_payload(self.config, self.step)
                # 仅替换当前测试字段，保持其余归档内容完全合法。
                payload[key] = bad_value
                # 把当前反例真正保存到临时归档，覆盖生产文件读取路径。
                self._save_payload(payload)
                # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
                with self.assertRaisesRegex(validator.CheckpointValidationError, message):
                    # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
                    validator.validate_checkpoint(
                        # 传入同一临时项目、归档、独立配置和要求步数。
                        self.root, self.checkpoint_path, self.config_path, self.step
                    )

    # 确认文件名中的训练步数也必须匹配归档内部步数。
    def test_rejects_filename_step_even_when_payload_step_matches(self) -> None:
        # 故意与内部训练步数不匹配的文件名。
        wrong_name = "generator_step_2499999.pt"
        # 文件名被污染但内部内容仍合法的检查点路径。
        wrong_path = self._save_payload(_valid_payload(self.config, self.step), wrong_name)
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(
            # 预期为声码器归档校验异常，消息还须包含此处给出的具体失败字段。
            validator.CheckpointValidationError, "Checkpoint filename mismatch"
        ):
            # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
            validator.validate_checkpoint(
                # 仅替换为错误文件名路径，其他验证条件保持有效。
                self.root, wrong_path, self.config_path, self.step
            )

    # 确认声码器轮次与批游标不能为负数或布尔值。
    def test_rejects_invalid_epoch_and_batch_cursor(self) -> None:
        # 逐个隔离检查的非法输入组合。
        cases = (
            # 把当前完整训练轮次改为该行给出的非法值，独立测试拒绝路径。
            ("epoch", -1, "checkpoint.epoch must be a non-negative integer"),
            # 把当前完整训练轮次改为该行给出的非法值，独立测试拒绝路径。
            ("epoch", True, "checkpoint.epoch must be a non-negative integer"),
            (
                # 当前轮次内已完成批位置。
                "batch_in_epoch",
                # 注入负数游标，生产校验应拒绝不可能的训练进度。
                -1,
                # 负批游标的预期错误信息，必须明确要求非负整数。
                "checkpoint.batch_in_epoch must be a non-negative integer",
            ),
        )
        # 逐项处理 key, value, message in cases，覆盖这组反例的全部输入。
        for key, value, message in cases:
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(key=key, value=value):
                # 创建与独立配置、数据来源和目标步数一致的声码器归档字典。
                payload = _valid_payload(self.config, self.step)
                # 每次只替换一个身份字段，使错误可归因到该项。
                payload[key] = value
                # 把当前反例真正保存到临时归档，覆盖生产文件读取路径。
                self._save_payload(payload)
                # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
                with self.assertRaisesRegex(validator.CheckpointValidationError, message):
                    # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
                    validator.validate_checkpoint(
                        # 传入同一临时项目、归档、独立配置和要求步数。
                        self.root, self.checkpoint_path, self.config_path, self.step
                    )

    # 确认完整配置内容必须匹配，而字典键排列顺序不影响语义身份。
    def test_canonical_config_comparison_is_complete_but_order_independent(self) -> None:
        # 只颠倒键顺序、语义内容不变的配置。
        reordered = dict(reversed(list(self.config.items())))
        # 将仅键顺序变化的配置封装进归档，验证规范化比较。
        self._save_payload(_valid_payload(reordered, self.step))
        # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
        validator.validate_checkpoint(
            # 传入同一临时项目、归档、独立配置和要求步数。
            self.root, self.checkpoint_path, self.config_path, self.step
        )

        # 用于注入实验名称或采样率变化的配置副本。
        altered = dict(reordered)
        # 改变实验名，检查完整配置比较不能只看少数超参数。
        altered["experiment_name"] = "not_the_supplied_experiment"
        # 将配置内容变化的反例写入归档。
        self._save_payload(_valid_payload(altered, self.step))
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(
            # 预期为声码器归档校验异常，消息还须包含此处给出的具体失败字段。
            validator.CheckpointValidationError, "canonical JSON"
        ):
            # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
            validator.validate_checkpoint(
                # 传入同一临时项目、归档、独立配置和要求步数。
                self.root, self.checkpoint_path, self.config_path, self.step
            )

    # 确认采样率等关键设置变化会在模型加载前拦截。
    def test_rejects_critical_config_change_before_model_loading(self) -> None:
        # 用于注入实验名称或采样率变化的配置副本。
        altered = dict(self.config)
        # 故意把采样率从 22050 改为 24000，制造关键配置不一致。
        altered["sampling_rate"] = 24_000
        # 将配置内容变化的反例写入归档。
        self._save_payload(_valid_payload(altered, self.step))
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(
            # 预期为声码器归档校验异常，消息还须包含此处给出的具体失败字段。
            validator.CheckpointValidationError, "Critical config mismatch for sampling_rate"
        ):
            # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
            validator.validate_checkpoint(
                # 传入同一临时项目、归档、独立配置和要求步数。
                self.root, self.checkpoint_path, self.config_path, self.step
            )

    # 确认检查点内部配置指纹不能与实际配置内容矛盾。
    def test_rejects_embedded_canonical_config_digest_mismatch(self) -> None:
        # 创建与独立配置、数据来源和目标步数一致的声码器归档字典。
        payload = _valid_payload(self.config, self.step)
        # 填入格式正确但内容错误的配置指纹。
        payload["config_semantic_sha256"] = "0" * 64
        # 把当前反例真正保存到临时归档，覆盖生产文件读取路径。
        self._save_payload(payload)
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(
            # 预期为声码器归档校验异常，消息还须包含此处给出的具体失败字段。
            validator.CheckpointValidationError, "config_semantic_sha256"
        ):
            # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
            validator.validate_checkpoint(
                # 传入同一临时项目、归档、独立配置和要求步数。
                self.root, self.checkpoint_path, self.config_path, self.step
            )

    # 分别验证缺失、格式错误和内容不匹配的数据指纹都会报错。
    def test_rejects_missing_invalid_and_mismatched_data_fingerprints(self) -> None:
        # 逐个隔离检查的非法输入组合。
        cases = []
        # 将要删除来源指纹字段的检查点副本。
        missing = _valid_payload(self.config, self.step)
        # 删去数据来源证明，检验归档不得缺少清单身份。
        missing.pop("data_fingerprints")
        # 登记缺失来源的反例及其预期报错提示。
        cases.append((missing, "non-empty mapping"))

        # 将要注入非法长度指纹的检查点副本。
        malformed = _valid_payload(self.config, self.step)
        # 复制内部指纹表，保证后续污染仅影响当前反例。
        malformed["data_fingerprints"] = dict(malformed["data_fingerprints"])
        # 用长度和大小写均非法的指纹测试格式保护。
        malformed["data_fingerprints"]["train_manifest.csv"] = "ABC"
        # 登记非法指纹格式的反例及预期说明。
        cases.append((malformed, "lowercase 64-hex"))

        # 将要注入合法格式但错误内容指纹的检查点副本。
        mismatched = _valid_payload(self.config, self.step)
        # 为数据内容不匹配反例准备独立指纹字典。
        mismatched["data_fingerprints"] = dict(mismatched["data_fingerprints"])
        # 填入合法 64 位格式但与配置登记值不符的训练清单指纹。
        mismatched["data_fingerprints"]["train_manifest.csv"] = "d" * 64
        # 登记数据身份不相等反例及其预期报错。
        cases.append((mismatched, "do not exactly match"))

        # 逐项处理 payload, message in cases，覆盖这组反例的全部输入。
        for payload, message in cases:
            # 把当前字段或反例单独标记，失败报告能指出具体出错项。
            with self.subTest(message=message):
                # 把当前反例真正保存到临时归档，覆盖生产文件读取路径。
                self._save_payload(payload)
                # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
                with self.assertRaisesRegex(validator.CheckpointValidationError, message):
                    # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
                    validator.validate_checkpoint(
                        # 传入同一临时项目、归档、独立配置和要求步数。
                        self.root, self.checkpoint_path, self.config_path, self.step
                    )

    # 确认生成器参数结构不匹配时绝不发布成功证明。
    def test_rejects_generator_state_dict_mismatch_and_writes_no_report(self) -> None:
        # 创建与独立配置、数据来源和目标步数一致的声码器归档字典。
        payload = _valid_payload(self.config, self.step)
        # 将生成器参数替换为未知键，检验 strict=True 真正生效。
        payload["generator"] = {"unexpected": torch.ones(1)}
        # 把当前反例真正保存到临时归档，覆盖生产文件读取路径。
        self._save_payload(payload)
        # 故意选择一个原本不存在的证明路径，失败后仍必须不存在。
        report_path = self.root / "must_not_exist.json"
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(
            # 预期为声码器归档校验异常，消息还须包含此处给出的具体失败字段。
            validator.CheckpointValidationError, "strict state-dict load failed"
        ):
            # 调用归档验证加证明发布入口，验证失败不得写成功报告。
            validator.attest_checkpoint(
                # 传入本例临时项目根目录。
                self.root,
                # 传入按要求步数命名的声码器归档位置。
                self.checkpoint_path,
                # 传入声码器独立配置文件位置。
                self.config_path,
                # 传入本轮归档测试要求的 2,500,000 更新步。
                self.step,
                # 传入测试证明或提取报告的输出路径。
                report_path,
            )
        # 核对 report_path.exists()：必须不成立。
        self.assertFalse(report_path.exists())

    # 确认含 NaN 或 Inf 的声码器权重不能通过归档验证。
    def test_rejects_nonfinite_generator_tensor(self) -> None:
        # 创建与独立配置、数据来源和目标步数一致的声码器归档字典。
        payload = _valid_payload(self.config, self.step)
        # 让一个真实权重元素为 NaN，测试有限性保护。
        payload["generator"]["weight"] = torch.tensor([float("nan"), 1.0])
        # 把当前反例真正保存到临时归档，覆盖生产文件读取路径。
        self._save_payload(payload)
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(
            # 预期为声码器归档校验异常，消息还须包含此处给出的具体失败字段。
            validator.CheckpointValidationError, "contains NaN or Inf"
        ):
            # 调用完整声码器归档验证，检查元数据、配置、来源与权重。
            validator.validate_checkpoint(
                # 传入同一临时项目、归档、独立配置和要求步数。
                self.root, self.checkpoint_path, self.config_path, self.step
            )

    # 确认声码器校验命令行正确解析相对路径并调用主机保护。
    def test_cli_resolves_relative_paths_and_publishes_report(self) -> None:
        # 相对于项目根的归档路径。
        relative_checkpoint = self.checkpoint_path.relative_to(self.root)
        # 相对于项目根的配置路径。
        relative_config = self.config_path.relative_to(self.root)
        # 相对于项目根的证明输出路径。
        relative_report = Path("reports") / "cli_attestation.json"
        # 记录命令行是否调用租用主机保护；文件操作仍只发生在临时测试目录。
        with mock.patch.object(validator, "assert_rented_server") as guard:
            # 生产 CLI 返回的成功或失败退出码。
            exit_code = validator.run_cli(
                [
                    # 指定临时人工项目根，避免脚本接触真实实验文件。
                    "--project-root",
                    # 将self.root转换为命令行字符串。
                    str(self.root),
                    # 指定要严格验证的声码器权重归档。
                    "--checkpoint",
                    # 将相对于项目根的归档路径转换为命令行字符串。
                    str(relative_checkpoint),
                    # 指定独立声码器配置，与归档内配置核对。
                    "--config",
                    # 将相对于项目根的配置路径转换为命令行字符串。
                    str(relative_config),
                    # 要求归档达到明确训练步数。
                    "--expected-step",
                    # 将本轮归档测试要求的 2,500,000 更新步转换为命令行字符串。
                    str(self.step),
                    # 指定仅校验成功后写入的证明文件。
                    "--output-report",
                    # 将相对于项目根的证明输出路径转换为命令行字符串。
                    str(relative_report),
                ]
            )
        # 核对 exit_code, 0：应完全相等。
        self.assertEqual(exit_code, 0)
        # 确认命令行对解析后的项目根恰好执行一次主机保护。
        guard.assert_called_once_with(self.root.resolve())
        # 核对 (root / relative_report).is_file()：必须成立。
        self.assertTrue((self.root / relative_report).is_file())

    # 确认重复 JSON 键不能悄悄覆盖实验身份。
    def test_strict_json_rejects_duplicate_keys(self) -> None:
        # 含重复 JSON 键的反例文件。
        duplicate_path = self.root / "duplicate.json"
        # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
        duplicate_path.write_text('{"data_protocol":"text-unseen","data_protocol":"x"}')
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(validator.CheckpointValidationError, "Duplicate JSON key"):
            # 用严格 JSON 读取器检查重复键文件，防止后值静默覆盖前值。
            validator.load_json_object(duplicate_path)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
