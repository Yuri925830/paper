# 计算文件或 JSON 的 SHA-256，检测内容被替换。
import hashlib
# 读写实验身份、指纹和报告的 JSON。
import json
# 使用标准库测试框架组织断言和失败报告。
import unittest

# 导入本组测试实际调用的 fastvoicegrad.infer_paper_eval 接口；测试直接约束生产实现。
from fastvoicegrad.infer_paper_eval import _report_required_fields
# 导入本组测试实际调用的 evaluation.evaluate_metrics 接口；测试直接约束生产实现。
from evaluation.evaluate_metrics import (
    # 锁定正式历史学生训练契约的 SHA-256。
    EXPECTED_FASTVOICEGRAD_RUN_CONTRACT_SHA256,
    # 锁定正式历史学生权重文件的 SHA-256。
    EXPECTED_FASTVOICEGRAD_STUDENT_SHA256,
    # 论文表格中单步 FastVoiceGrad 学生的参考行。
    PAPER_FASTVOICEGRAD_REFERENCE,
    # 论文表格中 30 步 VoiceGrad 教师的参考行。
    PAPER_VOICEGRAD30_REFERENCE,
    # 核对所有输出音频指纹及完整覆盖的生产函数。
    load_generated_wav_hash_manifest,
    # 区分教师与学生，并选择对应论文参考行。
    resolve_inference_system,
)


# 构造正式 8100 对学生推理报告的最小身份字段。
def formal_student_metadata():
    # 按本辅助函数的接口构造最小有效测试字典。
    return {
        # 系统机器标识，学生固定为fastvoicegrad；此处为测试使用的明确值。
        "system_id": "fastvoicegrad",
        # 明确标记student，防止与teacher评估混用；此处为测试使用的明确值。
        "model_role": "student",
        # 本次运行完成状态；仅全量8100组可标记正式完成；此处为测试使用的明确值。
        "status": "formal_8100_complete",
        # 论文客观评估要求的8100组转换；此处为测试使用的明确值。
        "paper_pair_count": 8100,
        # 本次实际选择的转换组数；此处为测试使用的明确值。
        "selected_pair_count": 8100,
        # 作为来源的教师训练轮次；此处为测试使用的明确值。
        "teacher_epoch": 500,
        # 学生已完成的训练轮次；此处为测试使用的明确值。
        "student_epoch": 100,
        # 学生已执行的优化器更新次数；此处为测试使用的明确值。
        "student_global_step": 123000,
        # 学生反向采样次数 K；此处为测试使用的明确值。
        "student_steps": 1,
        # 一步学生的加噪起始索引；此处为测试使用的明确值。
        "student_start_step": 950,
        # 加载的学生epoch100权重路径；此处为测试使用的明确值。
        "student_checkpoint": "/tmp/epoch_0100.pt",
        # 学生权重文件内容指纹。
        "student_checkpoint_sha256": EXPECTED_FASTVOICEGRAD_STUDENT_SHA256,
        # 学生训练契约指纹。
        "student_run_contract_sha256": EXPECTED_FASTVOICEGRAD_RUN_CONTRACT_SHA256,
        # 使用自训修改版HiFi-GAN；此处为测试使用的明确值。
        "vocoder_kind": "custom",
        # 声码器已完成2500000次训练更新；此处为测试使用的明确值。
        "vocoder_global_step": 2_500_000,
    }


# 把同一类故障保护与期望行为组织成一组测试：EvaluationSystemIdentityTests。
class EvaluationSystemIdentityTests(unittest.TestCase):
    # 保证历史教师报告仍标记为 VoiceGrad-30，不能误贴学生标签。
    def test_legacy_teacher_report_remains_supported(self):
        # 生产身份识别器返回的系统标签和论文参考行。
        result = resolve_inference_system({"teacher_epoch": 500})
        # 核对 result[“系统机器标识，学生固定为fastvoicegrad”], "voicegrad30"：应完全相等。
        self.assertEqual(result["system_id"], "voicegrad30")
        # 核对 result["paper_reference"], PAPER_VOICEGRAD30_REFERENCE：应完全相等。
        self.assertEqual(result["paper_reference"], PAPER_VOICEGRAD30_REFERENCE)
        # 核对 result[“报告中显示的学生名称与K=1、epoch100”], "VoiceGrad-30 epoch 500"：应完全相等。
        self.assertEqual(result["system_label"], "VoiceGrad-30 epoch 500")

    # 保证单步学生报告对应论文 FastVoiceGrad 行。
    def test_formal_student_report_is_labeled_as_fastvoicegrad(self):
        # 生产身份识别器返回的系统标签和论文参考行。
        result = resolve_inference_system(formal_student_metadata())
        # 核对 result[“系统机器标识，学生固定为fastvoicegrad”], "fastvoicegrad"：应完全相等。
        self.assertEqual(result["system_id"], "fastvoicegrad")
        # 核对 result["paper_reference"], PAPER_FASTVOICEGRAD_REFERENCE：应完全相等。
        self.assertEqual(result["paper_reference"], PAPER_FASTVOICEGRAD_REFERENCE)
        # 核对 result["paper_reference_row"], "FastVoiceGrad"：应完全相等。
        self.assertEqual(result["paper_reference_row"], "FastVoiceGrad")

    # 确认 30 步采样结果不能冒充 K=1 学生结果。
    def test_student_identity_rejects_wrong_sparse_schedule(self):
        # 构造正式学生推理身份字典，随后只改变当前测试的角色或步数。
        metadata = formal_student_metadata()
        # 故意将 K=1 改成 30 步，检验学生身份约束。
        metadata["student_steps"] = 30
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(ValueError, "K=1"):
            # 让评估器识别元数据，错误角色或步数应在此拒绝。
            resolve_inference_system(metadata)

    # 确认学生评估报告不能标记成教师角色。
    def test_student_identity_rejects_wrong_role(self):
        # 构造正式学生推理身份字典，随后只改变当前测试的角色或步数。
        metadata = formal_student_metadata()
        # 故意污染角色，确认学生不使用教师比较行。
        metadata["model_role"] = "teacher"
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(ValueError, "model_role=student"):
            # 让评估器识别元数据，错误角色或步数应在此拒绝。
            resolve_inference_system(metadata)

    # 确认生产推理器生成的报告字段能被评估器直接识别。
    def test_real_inference_report_contract_is_accepted_by_evaluator(self):
        # 将模型、数据和预处理绑定的指纹字典。
        identity = {
            # 学生权重文件内容指纹。
            "student_checkpoint_sha256": EXPECTED_FASTVOICEGRAD_STUDENT_SHA256,
            # 学生训练契约指纹。
            "student_run_contract_sha256": EXPECTED_FASTVOICEGRAD_RUN_CONTRACT_SHA256,
            # 冻结声码器权重文件指纹。
            "vocoder_checkpoint_sha256": "b" * 64,
            # 声码器配置文件指纹。
            "vocoder_config_sha256": "c" * 64,
            # 声码器最终归档证明指纹。
            "vocoder_attestation_sha256": "d" * 64,
            # 测试语音清单指纹。
            "test_manifest_sha256": "e" * 64,
            # seen/unseen 划分协议文件指纹。
            "split_protocol_sha256": "f" * 64,
            # 梅尔均值文件指纹。
            "mel_mean_sha256": "1" * 64,
            # 梅尔标准差文件指纹。
            "mel_std_sha256": "2" * 64,
        }
        # 用真实推理器字段构造学生身份报告，检验评估器能直接接收。
        report = {
            # 复用生产推理器规定的必须字段，避免测试另造结构。
            **_report_required_fields(
                # 传入将模型、数据和预处理绑定的指纹字典。
                identity,
                # 已发布运行身份文件的字节指纹，按下列值传给生产函数。
                run_identity_sha256="3" * 64,
                # 8100组转换配对清单的字节指纹，按下列值传给生产函数。
                pairs_csv_sha256="4" * 64,
                # 本次实际选择的转换组数，按下列值传给生产函数。
                selected_pair_count=8100,
            ),
            # 本次运行完成状态；仅全量8100组可标记正式完成；此处为测试使用的明确值。
            "status": "formal_8100_complete",
            # 加载的学生epoch100权重路径；此处为测试使用的明确值。
            "student_checkpoint": "/root/FsatVoiceGrad/epoch_0100.pt",
            # 使用自训修改版HiFi-GAN；此处为测试使用的明确值。
            "vocoder_kind": "custom",
            # 声码器已完成2500000次训练更新；此处为测试使用的明确值。
            "vocoder_global_step": 2_500_000,
        }
        # 生产身份识别器返回的系统标签和论文参考行。
        result = resolve_inference_system(report)
        # 核对 result[“系统机器标识，学生固定为fastvoicegrad”], "fastvoicegrad"：应完全相等。
        self.assertEqual(result["system_id"], "fastvoicegrad")
        # 核对 result[“学生已完成的训练轮次”], 100：应完全相等。
        self.assertEqual(result["student_epoch"], 100)

    # 确认输出音频指纹表覆盖全部评估对，缺一条也不能继续。
    def test_waveform_hash_manifest_must_cover_every_pair(self):
        # 只构造两个转换对，检查音频指纹表是否漏掉第二条。
        rows = [{"pair_id": "0"}, {"pair_id": "1"}]
        # 每个 pair_id 对应的音频 SHA-256 指纹。
        hashes = {"0": "a" * 64, "1": "b" * 64}
        # 按稳定键顺序序列化后的指纹表字节。
        encoded = json.dumps(
            # 按键排序、无多余空格序列化音频指纹表，以得到稳定集合签名。
            hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        # 把 JSON 文本编码为 UTF-8 字节，供 SHA-256 计算。
        ).encode("utf-8")
        # 构造每条 WAV 指纹及其集合签名组成的报告。
        report = {
            # 每个pair_id对应WAV文件的SHA-256；此处为测试使用的明确值。
            "generated_wav_sha256": hashes,
            # 按固定键顺序汇总全部输出WAV的整体指纹；此处为测试使用的明确值。
            "generated_wav_collection_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        # 核对 load_generated_wav_hash_manifest(report, rows), hashes：应完全相等。
        self.assertEqual(load_generated_wav_hash_manifest(report, rows), hashes)
        # 删掉第二个转换对的音频指纹，制造覆盖缺项。
        report["generated_wav_sha256"] = {"0": "a" * 64}
        # 该反例应被明确拒绝；同时核对异常类型及消息中的原因。
        with self.assertRaisesRegex(RuntimeError, "all pairs"):
            # 核对每个转换对都有音频指纹，缺项即报错。
            load_generated_wav_hash_manifest(report, rows)


# 仅在租用服务器直接执行该文件时启动测试入口。
if __name__ == "__main__":
    # 运行本文件全部测试，以退出码报告成功或失败。
    unittest.main()
