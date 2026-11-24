# 读写评估表格和语音清单，核对字段和行顺序。
import csv
# 按完整模块名加载位置编码类，核对兼容入口真实来源。
import importlib
# 按路径导入名称以数字开头的预处理脚本。
import importlib.util
# 读写实验身份、指纹和报告的 JSON。
import json
# 在临时目录复制或恢复文件，模拟统计量发布事务。
import shutil
# 沿用当前服务器 Python 解释器启动测试子进程。
import sys
# 将故意损坏的测试文件隔离在自动清理目录中。
import tempfile
# 创建可控的模块替身，隔离可选依赖或导入污染反例。
import types
# 使用标准库测试框架组织断言和失败报告。
import unittest
# 组合临时测试文件和仓库路径。
from pathlib import Path

# 创建和读取极小的人工音频、梅尔或特征数组。
import numpy as np

# 导入本组测试实际调用的 voicegrad 接口；测试直接约束生产实现。
from voicegrad import validate_setup


# 由测试文件位置定位的仓库根目录。
ROOT = Path(__file__).resolve().parents[1]


# 按文件位置导入数字开头的预处理脚本，不启动其命令行主流程。
def load_script(module_name, relative_path):
    # 按文件路径建立的 Python 模块加载描述。
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    # 尚待执行脚本定义的模块对象。
    module = importlib.util.module_from_spec(spec)
    # 执行模块定义；__main__ 入口不会因此启动。
    spec.loader.exec_module(module)
    # 返回已加载定义的预处理模块，供测试直接调用。
    return module


# 按路径导入 BNF 提取脚本模块，用其来源证明和续作函数。
bnf = load_script("voicegrad_provenance_bnf", "preprocess/05_extract_bnf.py")
# 导入说话人向量提取脚本模块，测试模型来源证明。
speaker = load_script("voicegrad_provenance_speaker", "preprocess/06_extract_spk_emb.py")


# 按正式列名写人工特征清单。
def write_manifest(path, rows):
    # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 按指定编码打开测试文件，作用域结束后自动关闭。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 按给定字段顺序写入 CSV 的记录器。
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        # 先写与正式清单相同的列名。
        writer.writeheader()
        # 按固定字段顺序写人工样本清单。
        writer.writerows(rows)


# 以固定键顺序写测试报告，构造可追溯的文件内容。
def write_json(path, value):
    # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 按指定编码打开测试文件，作用域结束后自动关闭。
    with path.open("w", encoding="utf-8") as handle:
        # 按键排序写入测试 JSON，便于稳定比较指纹。
        json.dump(value, handle, sort_keys=True)
        # 文件末尾加换行，保持逐行报告格式完整。
        handle.write("\n")


# 把同一类故障保护与期望行为组织成一组测试：ReuseAttestationTests。
class ReuseAttestationTests(unittest.TestCase):
    # 确认 ESPnet 兼容入口实际绑定到已审计本地位置编码类。
    def test_espnet_embedding_compat_uses_the_audited_local_class(self):
        # 旧仓库位置编码实现的真实模块名。
        local_names = (
            # 旧 BNF 仓库的顶层真实模块。
            "conformer_ppg_model",
            # 旧 BNF 编码器模块。
            "conformer_ppg_model.encoder",
            # 已审计的位置编码类真实实现模块。
            "conformer_ppg_model.encoder.embedding",
        )
        # 为旧检查点提供的 ESPnet 兼容模块名。
        compat_names = (
            # 旧检查点可能引用的 ESPnet 顶层兼容模块。
            "espnet",
            # 兼容模块层次中的 nets 包。
            "espnet.nets",
            # 兼容模块层次中的 PyTorch 后端包。
            "espnet.nets.pytorch_backend",
            # 兼容模块层次中的 Transformer 包。
            "espnet.nets.pytorch_backend.transformer",
            # 要绑定到本地审计实现的位置编码兼容入口。
            "espnet.nets.pytorch_backend.transformer.embedding",
        )
        # 测试期间需要临时替换和恢复的全部模块名。
        touched_names = local_names + compat_names
        # 测试开始前的模块缓存快照。
        saved_modules = {name: sys.modules.get(name) for name in touched_names}
        # 逐项处理 name in touched_names，覆盖这组反例的全部输入。
        for name in touched_names:
            # 移除本例临时模块缓存，不影响未存在的条目。
            sys.modules.pop(name, None)
        # 执行可能失败的测试操作，并为后面的恢复或依赖回退保留入口。
        try:
            # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
            with tempfile.TemporaryDirectory() as directory:
                # 人工旧特征提取仓库根目录。
                legacy_root = Path(directory)
                # 人工位置编码模块所在目录。
                encoder_dir = legacy_root / "conformer_ppg_model" / "encoder"
                # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
                encoder_dir.mkdir(parents=True)
                # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
                (encoder_dir / "__init__.py").write_text("", encoding="utf-8")
                # 把当前测试的文本或 JSON 写入指定临时文件，供生产读取路径验证。
                (encoder_dir / "embedding.py").write_text(
                    # 仅创建空的可辨认位置编码类，用于核对导入对象身份。
                    "class PositionalEncoding:\n    pass\n",
                    # 文件文本编码，保证中文和摘要可稳定读写，按下列值传给生产函数。
                    encoding="utf-8",
                )
                # 临时优先从人工旧仓库导入，核对兼容类来源。
                sys.path.insert(0, str(legacy_root))
                # 执行可能失败的测试操作，并为后面的恢复或依赖回退保留入口。
                try:
                    # 安装生产兼容入口，要求其绑定到锁定本地实现。
                    bnf.install_pinned_espnet_embedding_compat(legacy_root)
                    # 按真实仓库路径导入的位置编码模块。
                    local_embedding = importlib.import_module(local_names[-1])
                    # ESPnet 兼容入口当前指向的位置编码模块。
                    compat_embedding = sys.modules[compat_names[-1]]
                    # 核对下面完整给出的断言条件。
                    self.assertIs(
                        # 取兼容模块暴露的位置编码类对象。
                        compat_embedding.PositionalEncoding,
                        # 取审计本地模块的真实类，要求对象身份完全相同。
                        local_embedding.PositionalEncoding,
                    )
                    # 核对下面完整给出的断言条件。
                    self.assertEqual(
                        # 读取兼容模块记录的源码绝对路径。
                        Path(compat_embedding.__file__).resolve(),
                        # 要求源码路径就是本例创建的本地实现文件。
                        (encoder_dir / "embedding.py").resolve(),
                    )
                    # 核对下面完整给出的断言条件。
                    self.assertIs(
                        # 检查父兼容模块导出的 embedding 子模块对象。
                        sys.modules[compat_names[-2]].embedding,
                        # 传入ESPnet 兼容入口当前指向的位置编码模块。
                        compat_embedding,
                    )
                # 无论测试成功还是异常，都执行后续导入状态恢复。
                finally:
                    # 移除本例临时导入路径，避免影响其他测试。
                    sys.path.remove(str(legacy_root))
        # 无论测试成功还是异常，都执行后续导入状态恢复。
        finally:
            # 逐项处理 name in touched_names，覆盖这组反例的全部输入。
            for name in touched_names:
                # 测试前对应模块名指向的原对象。
                old_module = saved_modules[name]
                # 测试前该模块不存在时，结束后也恢复为未导入状态。
                if old_module is None:
                    # 移除本例临时模块缓存，不影响未存在的条目。
                    sys.modules.pop(name, None)
                # 处理另一种输入或恢复分支，保持测试前后状态一致。
                else:
                    # 还原测试开始前保存的真实模块对象。
                    sys.modules[name] = old_module

    # 确认进程已载入 ESPnet 时拒绝注入兼容层，避免混用来源。
    def test_espnet_embedding_compat_rejects_preloaded_modules(self):
        # 测试前 ESPnet 顶层模块状态。
        saved_espnet = sys.modules.get("espnet")
        # 故意预载入一个 ESPnet 模块，模拟不干净进程。
        sys.modules["espnet"] = types.ModuleType("espnet")
        # 执行可能失败的测试操作，并为后面的恢复或依赖回退保留入口。
        try:
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "fresh Python process"):
                # 在已预载入模块的进程尝试安装兼容层，应该先报错。
                bnf.install_pinned_espnet_embedding_compat(Path("unused"))
        # 无论测试成功还是异常，都执行后续导入状态恢复。
        finally:
            # 原进程没有 ESPnet 时，清理测试注入的顶层模块。
            if saved_espnet is None:
                # 删除本例人为预载入的 ESPnet 模块。
                sys.modules.pop("espnet", None)
            # 处理另一种输入或恢复分支，保持测试前后状态一致。
            else:
                # 若原进程已有 ESPnet，则原样恢复其对象。
                sys.modules["espnet"] = saved_espnet

    # 确认少量试跑报告不会覆盖已有全量正式提取报告。
    def test_limit_run_does_not_select_a_larger_formal_report_for_overwrite(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 设置已有正式提取报告位置，检查限量试跑是否避开它。
            report_path = Path(directory) / "report.json"
            # 保存本例构造的模型来源或统计量证明 JSON。
            write_json(report_path, {"total_rows": 100})
            # 逐项处理 module in (bnf, speaker)，覆盖这组反例的全部输入。
            for module in (bnf, speaker):
                # 取得限量试跑专用输出路径及正式报告保护标志。
                destination, protected = module.choose_limited_report_output(
                    # 传入测试证明或提取报告的输出路径。
                    report_path,
                    # 本次限量试跑选择的记录数，按下列值传给生产函数。
                    selected_rows=10,
                    # 试跑最大样本数，按下列值传给生产函数。
                    limit=10,
                )
                # 核对 protected：必须成立。
                self.assertTrue(protected)
                # 核对 destination.name, "report.limit-10.json"：应完全相等。
                self.assertEqual(destination.name, "report.limit-10.json")
                # 核对 destination, report_path：应不同。
                self.assertNotEqual(destination, report_path)

    # 确认形状正确的旧 BNF 也必须有模型与输入指纹证明才能复用。
    def test_shape_valid_bnf_without_ledger_is_not_attributed(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 测试音频文件路径。
            wav_path = root / "p001_001.wav"
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            wav_path.write_bytes(b"original wav")
            # 人工清单只有 p001_001 一条，并登记源 WAV 路径以核对输入内容。
            rows = [{"speaker": "p001", "utt_id": "p001_001", "wav22k_path": str(wav_path)}]
            # 预处理测试的特征输出目录。
            out_dir = root / "bnf"
            # 按说话人/语音.npy 规范组织人工特征文件。
            path = out_dir / "p001" / "p001_001.npy"
            # 当前特征提取测试的最终来源证明报告位置。
            report_path = root / "report.json"
            # 当前特征提取模型、配置和代码指纹。
            provenance = {
                # 特征提取旧代码的实际 Git 提交；此处为测试使用的明确值。
                "legacy_git_commit": bnf.EXPECTED_LEGACY_COMMIT,
                # 审计允许的旧代码 Git 提交；此处为测试使用的明确值。
                "legacy_git_expected_commit": bnf.EXPECTED_LEGACY_COMMIT,
                # 特征模型权重或检查点文件的实际指纹；此处为测试使用的明确值。
                "checkpoint_sha256": bnf.EXPECTED_BNF_CHECKPOINT_SHA256,
                # 事先锁定的特征模型权重指纹；此处为测试使用的明确值。
                "checkpoint_expected_sha256": bnf.EXPECTED_BNF_CHECKPOINT_SHA256,
                # 特征提取配置的实际指纹；此处为测试使用的明确值。
                "config_sha256": bnf.EXPECTED_BNF_CONFIG_SHA256,
                # 事先锁定的特征提取配置指纹；此处为测试使用的明确值。
                "config_expected_sha256": bnf.EXPECTED_BNF_CONFIG_SHA256,
                # 本地位置编码兼容源码的实际指纹；此处为测试使用的明确值。
                "compat_embedding_sha256": bnf.EXPECTED_BNF_COMPAT_EMBEDDING_SHA256,
                # 允许使用的位置编码兼容源码指纹；此处为测试使用的明确值。
                "compat_embedding_expected_sha256": bnf.EXPECTED_BNF_COMPAT_EMBEDDING_SHA256,
            }
            # 通过生产 BNF 原子写入接口保存；新内容非法时应保留旧文件。
            bnf.atomic_save_bnf(path, np.ones((144, 3), dtype=np.float32))
            # 可在中断后继续读取的逐特征来源日志路径。
            journal_path = bnf.progress_journal_path(report_path, provenance)

            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "--force"):
                # 检查现有 BNF 的模型来源、源 WAV 和数值内容证明。
                bnf.verify_reused_feature_records(
                    # 一起传入样本清单、特征目录、正式报告、断点日志和要求来源。
                    rows, out_dir, report_path, journal_path, provenance
                )

            # A durable per-file journal makes a hard-interrupted first run resumable
            # even though no final report was written yet.
            # 单个 BNF 的数值、输入 WAV 和操作来源记录。
            journal_record = bnf.make_feature_record(
                # 传入当前测试要读写或校验的文件路径。
                path,
                # 读取实际特征数组用于独立登记内容指纹。
                np.load(path, allow_pickle=False),
                # 生成当前特征的源 WAV 路径，按下列值传给生产函数。
                input_wav_path=wav_path,
                # 该条特征是本轮生成还是已验证复用，按下列值传给生产函数。
                action="generated",
            )
            # 先创建与当前模型来源绑定的持久逐文件日志。
            bnf.ensure_progress_journal(journal_path, provenance)
            # 把已完成特征证明追加到日志，中断时仍可审计复用。
            bnf.append_progress_record(
                # 传入可在中断后继续读取的逐特征来源日志路径。
                journal_path,
                # 使用说话人/语音.npy 组成稳定特征身份键。
                bnf.feature_key(rows[0]),
                # 传入单个 BNF 的数值、输入 WAV 和操作来源记录。
                journal_record,
            )
            # 核对下面完整给出的断言条件。
            self.assertEqual(
                # 只比较可复用特征的身份键集合，不依赖字典顺序。
                set(
                    # 检查现有 BNF 的模型来源、源 WAV 和数值内容证明。
                    bnf.verify_reused_feature_records(
                        # 一起传入样本清单、特征目录、正式报告、断点日志和要求来源。
                        rows, out_dir, report_path, journal_path, provenance
                    )
                ),
                # 当前人工库存恰好只有这一条合法可复用特征。
                {"p001/p001_001.npy"},
            )
            # 删除本例临时日志，随后单独验证正式报告提供的复用证明。
            journal_path.unlink()

            # 按测试夹具所需内容写入 JSON 来源证明。
            write_json(
                # 传入测试证明或提取报告的输出路径。
                report_path,
                {
                    # 特征提取报告结构版本；此处为测试使用的明确值。
                    "report_schema_version": bnf.REPORT_SCHEMA_VERSION,
                    # 特征类型，如 BNF 或说话人向量；此处为测试使用的明确值。
                    "feature_kind": bnf.FEATURE_KIND,
                    # 产生这些特征的模型、配置和代码来源；此处为测试使用的明确值。
                    "output_provenance": provenance,
                    # 逐文件记录的内容与输入指纹；此处为测试使用的明确值。
                    "file_records": {
                        # 为当前 BNF 建立与内容和源音频绑定的逐文件证明。
                        bnf.feature_key(rows[0]): bnf.make_feature_record(
                            # 传入当前测试要读写或校验的文件路径。
                            path,
                            # 读取实际特征数组用于独立登记内容指纹。
                            np.load(path, allow_pickle=False),
                            # 生成当前特征的源 WAV 路径，按下列值传给生产函数。
                            input_wav_path=wav_path,
                            # 该条特征是本轮生成还是已验证复用，按下列值传给生产函数。
                            action="generated",
                        )
                    },
                },
            )
            # 通过来源与内容检查后允许复用的特征字典。
            verified = bnf.verify_reused_feature_records(
                # 一起传入样本清单、特征目录、正式报告、断点日志和要求来源。
                rows, out_dir, report_path, journal_path, provenance
            )
            # 核对 set(verified), {"p001/p001_001.npy"}：应完全相等。
            self.assertEqual(set(verified), {"p001/p001_001.npy"})

            # Same dtype and shape, different content: shape-only resume must not pass.
            # 通过生产 BNF 原子写入接口保存；新内容非法时应保留旧文件。
            bnf.atomic_save_bnf(path, np.zeros((144, 3), dtype=np.float32))
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "attestation"):
                # 检查现有 BNF 的模型来源、源 WAV 和数值内容证明。
                bnf.verify_reused_feature_records(
                    # 一起传入样本清单、特征目录、正式报告、断点日志和要求来源。
                    rows, out_dir, report_path, journal_path, provenance
                )

    # 确认说话人特征复用必须匹配同一模型与配置来源。
    def test_speaker_reuse_requires_matching_model_provenance(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 测试音频文件路径。
            wav_path = root / "p001_001.wav"
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            wav_path.write_bytes(b"speaker wav")
            # 人工清单只有 p001_001 一条，并登记源 WAV 路径以核对输入内容。
            rows = [{"speaker": "p001", "utt_id": "p001_001", "wav22k_path": str(wav_path)}]
            # 预处理测试的特征输出目录。
            out_dir = root / "spk_emb"
            # 按说话人/语音.npy 规范组织人工特征文件。
            path = out_dir / "p001" / "p001_001.npy"
            # 当前特征提取测试的最终来源证明报告位置。
            report_path = root / "report.json"
            # 当前已提交生效的统计代次清单。
            current = {
                # 特征提取旧代码的实际 Git 提交；此处为测试使用的明确值。
                "legacy_git_commit": speaker.EXPECTED_LEGACY_COMMIT,
                # 审计允许的旧代码 Git 提交；此处为测试使用的明确值。
                "legacy_git_expected_commit": speaker.EXPECTED_LEGACY_COMMIT,
                # 特征模型权重或检查点文件的实际指纹；此处为测试使用的明确值。
                "checkpoint_sha256": speaker.EXPECTED_SPK_CHECKPOINT_SHA256,
                # 事先锁定的特征模型权重指纹；此处为测试使用的明确值。
                "checkpoint_expected_sha256": speaker.EXPECTED_SPK_CHECKPOINT_SHA256,
                # 特征提取配置的实际指纹；此处为测试使用的明确值。
                "config_sha256": speaker.EXPECTED_SPK_CONFIG_SHA256,
                # 事先锁定的特征提取配置指纹；此处为测试使用的明确值。
                "config_expected_sha256": speaker.EXPECTED_SPK_CONFIG_SHA256,
            }
            # 通过生产原子写入接口保存 256 维说话人特征。
            speaker.atomic_save_speaker(path, np.ones(256, dtype=np.float32))
            # 可在中断后继续读取的逐特征来源日志路径。
            journal_path = speaker.progress_journal_path(report_path, current)
            # 故意篡改模型配置指纹后的来源字典。
            wrong = dict(current)
            # 故意替换模型配置指纹，虽然向量形状正确也应拒绝复用。
            wrong["config_sha256"] = "c" * 64
            # 按测试夹具所需内容写入 JSON 来源证明。
            write_json(
                # 传入测试证明或提取报告的输出路径。
                report_path,
                {
                    # 特征提取报告结构版本；此处为测试使用的明确值。
                    "report_schema_version": speaker.REPORT_SCHEMA_VERSION,
                    # 特征类型，如 BNF 或说话人向量；此处为测试使用的明确值。
                    "feature_kind": speaker.FEATURE_KIND,
                    # 产生这些特征的模型、配置和代码来源；此处为测试使用的明确值。
                    "output_provenance": wrong,
                    # 逐文件记录的内容与输入指纹；此处为测试使用的明确值。
                    "file_records": {
                        # 为当前说话人向量建立模型来源和输入音频证明。
                        speaker.feature_key(rows[0]): speaker.make_feature_record(
                            # 传入当前测试要读写或校验的文件路径。
                            path,
                            # 读取实际特征数组用于独立登记内容指纹。
                            np.load(path, allow_pickle=False),
                            # 生成当前特征的源 WAV 路径，按下列值传给生产函数。
                            input_wav_path=wav_path,
                            # 该条特征是本轮生成还是已验证复用，按下列值传给生产函数。
                            action="generated",
                        )
                    },
                },
            )
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "trusted record"):
                # 对说话人特征执行与 BNF 同样严格的来源复用检查。
                speaker.verify_reused_feature_records(
                    # 一起传入当前清单与当前说话人模型来源，旧来源不得混入。
                    rows, out_dir, report_path, journal_path, current
                )


# 把同一类故障保护与期望行为组织成一组测试：TrainingProvenanceAuditTests。
class TrainingProvenanceAuditTests(unittest.TestCase):
    # 创建一份带输入 WAV、特征内容和模型来源证明的最小 BNF 集合。
    def _build_bnf_collection(self, root):
        # 测试音频文件路径。
        wav_path = root / "data" / "wav22k" / "p001" / "p001_001.wav"
        # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
        wav_path.parent.mkdir(parents=True)
        # 写入本例规定的文件字节，用于构造或篡改内容指纹。
        wav_path.write_bytes(b"audited input wav")
        # 人工清单只有 p001_001 一条，并登记源 WAV 路径以核对输入内容。
        rows = [{"speaker": "p001", "utt_id": "p001_001", "wav22k_path": str(wav_path)}]
        # 人工训练或全量样本清单路径。
        manifest_path = root / "data" / "metadata" / "all_manifest.csv"
        # 把人工语音清单写盘，使生产审计能独立核对清单指纹。
        write_manifest(manifest_path, rows)
        # 预处理测试的特征输出目录。
        out_dir = root / "data" / "bnf"
        # 待完整审计的 BNF 文件路径。
        feature_path = out_dir / "p001" / "p001_001.npy"
        # 可辨认的 144 通道小 BNF 数组。
        array = np.arange(144 * 3, dtype=np.float32).reshape(144, 3)
        # 通过生产 BNF 原子写入接口保存；新内容非法时应保留旧文件。
        bnf.atomic_save_bnf(feature_path, array)
        # 当前特征提取模型、配置和代码指纹。
        provenance = {
            # 特征提取旧代码的实际 Git 提交；此处为测试使用的明确值。
            "legacy_git_commit": validate_setup.EXPECTED_LEGACY_COMMIT,
            # 审计允许的旧代码 Git 提交；此处为测试使用的明确值。
            "legacy_git_expected_commit": validate_setup.EXPECTED_LEGACY_COMMIT,
            # 特征模型权重或检查点文件的实际指纹；此处为测试使用的明确值。
            "checkpoint_sha256": validate_setup.EXPECTED_FEATURE_CHECKPOINTS["bnf"],
            # 事先锁定的特征模型权重指纹；此处为测试使用的明确值。
            "checkpoint_expected_sha256": validate_setup.EXPECTED_FEATURE_CHECKPOINTS["bnf"],
            # 特征提取配置的实际指纹；此处为测试使用的明确值。
            "config_sha256": validate_setup.EXPECTED_FEATURE_CONFIGS["bnf"],
            # 事先锁定的特征提取配置指纹；此处为测试使用的明确值。
            "config_expected_sha256": validate_setup.EXPECTED_FEATURE_CONFIGS["bnf"],
            # 本地位置编码兼容源码的实际指纹；此处为测试使用的明确值。
            "compat_embedding_sha256": validate_setup.EXPECTED_BNF_COMPAT_EMBEDDING,
            # 允许使用的位置编码兼容源码指纹；此处为测试使用的明确值。
            "compat_embedding_expected_sha256": validate_setup.EXPECTED_BNF_COMPAT_EMBEDDING,
        }
        # 按语音身份键保存逐文件特征来源证明，供生产审计复核。
        records = {
            # 为当前 BNF 建立与内容和源音频绑定的逐文件证明。
            bnf.feature_key(rows[0]): bnf.make_feature_record(
                # 传入待完整审计的 BNF 文件路径。
                feature_path,
                # 传入可辨认的 144 通道小 BNF 数组。
                array,
                # 生成当前特征的源 WAV 路径，按下列值传给生产函数。
                input_wav_path=wav_path,
                # 该条特征是本轮生成还是已验证复用，按下列值传给生产函数。
                action="generated",
            )
        }
        # 用生产接口生成包含 BNF 来源、输入与内容指纹的提取报告。
        report = bnf.build_extraction_report(
            # 人工训练或全量样本清单路径。
            manifest_path=manifest_path,
            # 传入本例仅含一条语音的人工清单，用于构建提取证明。
            rows=rows,
            # BNF 特征输出目录，按下列值传给生产函数。
            bnf_dir=out_dir,
            # 固定旧版特征提取仓库路径，按下列值传给生产函数。
            legacy_bnf_root=root / "external",
            # 特征模型配置文件路径，按下列值传给生产函数。
            train_config=root / "config.yaml",
            # 特征模型权重路径，按下列值传给生产函数。
            model_file=root / "model.pt",
            # 产生这些特征的模型、配置和代码来源，按下列值传给生产函数。
            output_provenance=provenance,
            # 是否明确强制重做已有文件，按下列值传给生产函数。
            force=True,
            # 本次新生成特征条数，按下列值传给生产函数。
            generated_rows=1,
            # 本次复用数量，按下列值传给生产函数。
            reused_rows=0,
            # 提取失败条数，按下列值传给生产函数。
            failed_rows=0,
            # 逐文件来源证明记录，按下列值传给生产函数。
            records=records,
            # 失败样本信息列表，按下列值传给生产函数。
            fail_examples=[],
        )
        # 保存本例构造的模型来源或统计量证明 JSON。
        write_json(root / "exp" / "logs" / "bnf_extraction_report.json", report)
        # 返回清单、特征文件与源音频，供后续分别篡改验证。
        return rows, feature_path, wav_path

    # 确认完整审计能发现同路径同形状的特征或输入音频内容替换。
    def test_full_audit_detects_same_shape_content_change(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 建立完整有效的最小 BNF 集合，再针对不同对象做内容变更。
            rows, feature_path, wav_path = self._build_bnf_collection(root)
            # 对人工特征集合执行训练前来源审计，取得已核验的集合指纹。
            fingerprints = validate_setup.audit_feature_provenance(
                # 启用完整内容重哈希，不能只比较路径和文件大小。
                root, rows, "bnf", check_content=True
            )
            # 核对 "bnf.content_sha256", fingerprints：应包含目标内容。
            self.assertIn("bnf.content_sha256", fingerprints)

            # 形状与类型保持不变但内容改成全零的反例。
            changed = np.zeros((144, 3), dtype=np.float32)
            # 通过生产 BNF 原子写入接口保存；新内容非法时应保留旧文件。
            bnf.atomic_save_bnf(feature_path, changed)
            # The lightweight startup layout is intentionally unchanged.
            # 调用训练启动前特征来源审计。
            validate_setup.audit_feature_provenance(
                # 仅检查来源和布局，不重新哈希全部特征内容。
                root, rows, "bnf", check_content=False
            )
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "Content changed"):
                # 调用训练启动前特征来源审计。
                validate_setup.audit_feature_provenance(
                    # 启用完整内容重哈希，不能只比较路径和文件大小。
                    root, rows, "bnf", check_content=True
                )

            # Restore the feature, then change the input WAV at the same path/size.
            # 可辨认的 144 通道小 BNF 数组。
            array = np.arange(144 * 3, dtype=np.float32).reshape(144, 3)
            # 通过生产 BNF 原子写入接口保存；新内容非法时应保留旧文件。
            bnf.atomic_save_bnf(feature_path, array)
            # 写入本例规定的文件字节，用于构造或篡改内容指纹。
            wav_path.write_bytes(b"changed input wav")
            # 核对 len(b"changed input wav"), len(b"audited input wav")：应完全相等。
            self.assertEqual(len(b"changed input wav"), len(b"audited input wav"))
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "Input WAV content changed"):
                # 调用训练启动前特征来源审计。
                validate_setup.audit_feature_provenance(
                    # 启用完整内容重哈希，不能只比较路径和文件大小。
                    root, rows, "bnf", check_content=True
                )

    # 确认混合模型来源或未审计配置不能供正式训练使用。
    def test_mixed_or_incomplete_report_is_rejected(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 准备有效集合，仅保留清单供报告来源反例使用。
            rows, _, _ = self._build_bnf_collection(root)
            # BNF 来源报告位于生产启动验证器约定的日志路径。
            report_path = root / "exp" / "logs" / "bnf_extraction_report.json"
            # 读回本例特征来源报告，随后分别注入配置来源或混合状态反例。
            report = json.loads(report_path.read_text(encoding="utf-8"))
            # 深复制后可独立篡改的提取报告。
            wrong_config = json.loads(json.dumps(report))
            # 改变报告中的模型配置来源，检查未审计配置被拒绝。
            wrong_config["output_provenance"]["config_sha256"] = "e" * 64
            # 保存本例构造的模型来源或统计量证明 JSON。
            write_json(report_path, wrong_config)
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "Unaudited model provenance"):
                # 调用训练启动前特征来源审计。
                validate_setup.audit_feature_provenance(
                    # 仅检查来源和布局，不重新哈希全部特征内容。
                    root, rows, "bnf", check_content=False
                )

            # 把来源状态标成混合，正式训练不得接受。
            report["provenance_status"] = "mixed"
            # 保存本例构造的模型来源或统计量证明 JSON。
            write_json(report_path, report)
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "mixed provenance"):
                # 调用训练启动前特征来源审计。
                validate_setup.audit_feature_provenance(
                    # 仅检查来源和布局，不重新哈希全部特征内容。
                    root, rows, "bnf", check_content=False
                )

    # 确认归一化统计量来自当前训练集，且有未完成发布标记时阻止训练。
    def test_mel_stats_current_manifest_and_pending_marker(self):
        # 在租用服务器创建自动清理的独立临时目录，故意损坏的文件不碰真实实验。
        with tempfile.TemporaryDirectory() as directory:
            # 测试目录的 Path 对象。
            root = Path(directory)
            # 人工清单只含 p001_001 一条，测试逐文件续作或统计发布保护。
            rows = [{"speaker": "p001", "utt_id": "p001_001"}]
            # 当前训练集身份清单路径。
            train_manifest = root / "data" / "metadata" / "train_manifest.csv"
            # 把人工语音清单写盘，使生产审计能独立核对清单指纹。
            write_manifest(train_manifest, rows)
            # 对应测试音频的梅尔数组路径。
            mel_path = root / "data" / "mel" / "p001" / "p001_001.npy"
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            mel_path.parent.mkdir(parents=True)
            # 人工梅尔为 80 通道、2 帧全零数组，统计值和内容哈希可独立核对。
            mel = np.zeros((80, 2), dtype=np.float32)
            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(mel_path, mel, allow_pickle=False)
            # 梅尔文件在集合中使用的稳定语音键。
            mel_key = "p001/p001_001.npy"
            # 按语音键累计梅尔数值指纹的哈希对象。
            mel_content = __import__("hashlib").sha256()
            # 向按语音键累计梅尔数值指纹的哈希对象追加当前语音身份键。
            mel_content.update(mel_key.encode("utf-8") + b"\0")
            # 向按语音键累计梅尔数值指纹的哈希对象追加当前特征数组内容指纹。
            mel_content.update(validate_setup.sha256_array(mel).encode("ascii") + b"\n")
            # 按语音键累计文件大小的布局哈希对象。
            mel_layout = __import__("hashlib").sha256()
            # 向按语音键累计文件大小的布局哈希对象追加当前语音身份键。
            mel_layout.update(mel_key.encode("utf-8") + b"\0")
            # 向按语音键累计文件大小的布局哈希对象追加当前文件字节长度。
            mel_layout.update(str(mel_path.stat().st_size).encode("ascii") + b"\n")
            # 同一测试代次的梅尔均值和标准差发布目录。
            stats_dir = root / "data" / "stats"
            # 给本次人工均值与标准差发布固定代次 ID。
            generation_id = "test-generation"
            # 保存同一代均值、标准差和证明的不可变目录。
            generation_dir = stats_dir / ".mel_stats_generations" / generation_id
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            generation_dir.mkdir(parents=True)

            # 逐梅尔通道的人工均值数组。
            mean = np.zeros(80, dtype=np.float32)
            # 逐梅尔通道的正标准差数组。
            std = np.ones(80, dtype=np.float32)
            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(generation_dir / "mel_mean.npy", mean, allow_pickle=False)
            # 保存当前人工数组，检验实际磁盘文件的形状、有限性或身份。
            np.save(generation_dir / "mel_std.npy", std, allow_pickle=False)
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            stats_dir.mkdir(parents=True, exist_ok=True)
            # 在本例临时目录复制统计量或证明，为不可变代次和兼容别名建立一致基准。
            shutil.copyfile(generation_dir / "mel_mean.npy", stats_dir / "mel_mean.npy")
            # 在本例临时目录复制统计量或证明，为不可变代次和兼容别名建立一致基准。
            shutil.copyfile(generation_dir / "mel_std.npy", stats_dir / "mel_std.npy")

            # 当前均值文件的 SHA-256。
            mean_hash = validate_setup.sha256_file(generation_dir / "mel_mean.npy")
            # 当前标准差文件的 SHA-256。
            std_hash = validate_setup.sha256_file(generation_dir / "mel_std.npy")
            # 构造归一化统计计算证明，绑定训练清单、梅尔集合和统计文件。
            report = {
                # 归一化统计量发布结构版本；此处为测试使用的明确值。
                "publication_schema_version": 1,
                # 把均值与标准差绑定到同一次计算的代次标识；此处为测试使用的明确值。
                "generation_id": generation_id,
                # 计算统计量所依据训练清单的指纹；此处为测试使用的明确值。
                "manifest_sha256": validate_setup.sha256_file(train_manifest),
                # 实际参与统计的训练语音条数；此处为测试使用的明确值。
                "rows_used": 1,
                # 缺失梅尔文件数；此处为测试使用的明确值。
                "missing_count": 0,
                # 格式错误梅尔文件数；此处为测试使用的明确值。
                "bad_count": 0,
                # 80 维梅尔均值文件指纹；此处为测试使用的明确值。
                "mean_sha256": mean_hash,
                # 80 维梅尔标准差文件指纹；此处为测试使用的明确值。
                "std_sha256": std_hash,
                # 参与统计的全部梅尔内容集合指纹；此处为测试使用的明确值。
                "mel_collection_content_sha256": mel_content.hexdigest(),
                # 参与统计的梅尔路径和大小集合指纹；此处为测试使用的明确值。
                "mel_collection_layout_sha256": mel_layout.hexdigest(),
                # 参与统计的梅尔文件数量；此处为测试使用的明确值。
                "mel_collection_count": 1,
            }
            # 不可变代次内的统计计算报告路径。
            generation_report = generation_dir / "report.json"
            # 保存本例构造的模型来源或统计量证明 JSON。
            write_json(generation_report, report)
            # 供历史读取接口使用的统计报告别名路径。
            report_alias = root / "exp" / "logs" / "mel_stats_report.json"
            # 创建此测试需要的目录层次；仅操作当前临时测试根下的路径。
            report_alias.parent.mkdir(parents=True)
            # 在本例临时目录复制统计量或证明，为不可变代次和兼容别名建立一致基准。
            shutil.copyfile(generation_report, report_alias)
            # 当前已提交生效的统计代次清单。
            current = {
                # 当前 JSON 身份或发布清单结构版本。
                "schema_version": 1,
                # 把均值与标准差绑定到同一次计算的代次标识；此处为测试使用的明确值。
                "generation_id": generation_id,
                # 发布记录的 UTC 时间；此处为测试使用的明确值。
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                # 同一发布代次包含的统计文件；此处为测试使用的明确值。
                "files": {
                    # 当前代次均值文件及其字节身份。
                    "mel_mean.npy": {
                        # 文件实际字节数；此处为测试使用的明确值。
                        "bytes": (generation_dir / "mel_mean.npy").stat().st_size,
                        # 数组数值存储类型；此处为测试使用的明确值。
                        "dtype": "float32",
                        # 数组维度，区分通道与时间；此处为测试使用的明确值。
                        "shape": [80],
                        # 当前内容的 SHA-256 指纹；此处为测试使用的明确值。
                        "sha256": mean_hash,
                    },
                    # 当前代次标准差文件及其字节身份。
                    "mel_std.npy": {
                        # 文件实际字节数；此处为测试使用的明确值。
                        "bytes": (generation_dir / "mel_std.npy").stat().st_size,
                        # 数组数值存储类型；此处为测试使用的明确值。
                        "dtype": "float32",
                        # 数组维度，区分通道与时间；此处为测试使用的明确值。
                        "shape": [80],
                        # 当前内容的 SHA-256 指纹；此处为测试使用的明确值。
                        "sha256": std_hash,
                    },
                },
                # 当前代次统计报告的文件身份；此处为测试使用的明确值。
                "report": {
                    # 文件实际字节数；此处为测试使用的明确值。
                    "bytes": generation_report.stat().st_size,
                    # 当前内容的 SHA-256 指纹；此处为测试使用的明确值。
                    "sha256": validate_setup.sha256_file(generation_report),
                },
            }
            # 不可变代次内的发布清单路径。
            generation_manifest = generation_dir / "manifest.json"
            # 当前有效统计代次指针文件。
            current_path = stats_dir / "mel_stats_current.json"
            # 保存本例构造的模型来源或统计量证明 JSON。
            write_json(generation_manifest, current)
            # 在本例临时目录复制统计量或证明，为不可变代次和兼容别名建立一致基准。
            shutil.copyfile(generation_manifest, current_path)

            # 检查统计量发布与当前训练清单一致，并取得当前均值等文件指纹。
            fingerprints = validate_setup.audit_mel_stats_publication(root, rows)
            # 核对下面完整给出的断言条件。
            self.assertEqual(
                # 取审计确认的均值文件指纹。
                fingerprints["mel_stats.mean_sha256"],
                # 独立读取当前均值字节指纹，核对审计返回身份。
                validate_setup.sha256_file(stats_dir / "mel_mean.npy"),
            )
            # 注入一个未完成发布标记，检验训练启动会被阻止。
            write_json(stats_dir / ".mel_stats_publish_transaction.json", {})
            # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
            with self.assertRaisesRegex(RuntimeError, "Pending"):
                # 检查统计量发布完整性、当前训练清单与未完成事务标记。
                validate_setup.audit_mel_stats_publication(root, rows)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
