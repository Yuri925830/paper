# 用完全模拟的 nvidia-smi 输出锁定正式训练首次 CUDA 加载前的二次空闲检查。
"""检查 GPU 二次身份、占用和计算进程守卫。"""  # 本文件只定义 CPU 单元测试，不会接触真实 GPU。

import os  # os.getpid 提供当前测试进程 PID，用来覆盖允许自身 CUDA 探测的分支。
import subprocess  # CompletedProcess 构造不执行外部命令的模拟查询结果。
import unittest  # unittest 负责组织成功与失败关闭场景。
from unittest.mock import patch  # patch 把生产函数中的 subprocess.run 临时替换为可控假对象。

from fastvoicegrad.train import revalidate_gpu_idle_before_cuda_load  # 直接测试正式训练调用的同一二次守卫。


def make_attestation() -> dict:  # 建立已经通过启动器初检的最小正式证明。
    """返回供二次守卫交叉核对的固定 GPU 身份。"""  # 所有值都采用生产映射中的同名字段。

    return {  # 只放二次守卫会读取或更新的字段，便于看清测试契约。
        "selected_gpu_physical_id": "2",  # 假设跨节点盘点后选择物理 GPU 2。
        "selected_gpu_name": "NVIDIA A100-SXM4-80GB",  # 固定型号用于身份交叉核对。
        "selected_gpu_uuid": "GPU-1234-abcd",  # 固定 UUID 防止 index 重排误用其他卡。
        "selected_gpu_total_mib": "81920",  # 固定总显存与启动器初检保持一致。
        "gpu_launch_guard_status": "validated",  # 表示上游初检已经通过。
        "gpu_recheck_status": "pending",  # 首次 CUDA 加载前仍等待本函数二次确认。
        "gpu_recheck_utc": None,  # 二次查询尚未发生，因此没有时间。
        "gpu_recheck_used_mib": None,  # 二次查询尚未发生，因此没有显存值。
        "gpu_recheck_util_percent": None,  # 二次查询尚未发生，因此没有利用率值。
        "gpu_recheck_compute_process_state": None,  # 二次查询尚未发生，因此没有进程结论。
    }  # 完成最小初检证明。


def completed(stdout: str) -> subprocess.CompletedProcess:  # 简化每个测试的模拟 nvidia-smi 成功返回值。
    """构造退出码为零且只含指定标准输出的假子进程结果。"""  # 不执行任何真实命令。

    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=stdout, stderr="")  # 字段满足生产代码读取接口。


class GpuLaunchRecheckTests(unittest.TestCase):  # 覆盖空闲通过、被抢占和身份改变三类关键结果。
    """二次查询必须按 UUID 失败关闭，且不得信任过期启动数值。"""  # 所有外部结果都由 mock 提供。

    def test_recheck_accepts_idle_same_gpu(self) -> None:  # 同一 GPU 仍空闲时更新动态证明。
        """身份不变、低占用且无进程时应返回完整二次证据。"""  # 成功路径仍不接触 CUDA。

        gpu_row = "2, NVIDIA A100-SXM4-80GB, GPU-1234-abcd, 81920, 12, 0\n"  # 模拟低显存且零利用率的同一卡。
        with patch("fastvoicegrad.train.subprocess.run", side_effect=[completed(gpu_row), completed("")]) as mocked_run:  # 第二次查询模拟没有计算 PID。
            result = revalidate_gpu_idle_before_cuda_load(make_attestation())  # 调用生产二次守卫并取得新映射。
        self.assertEqual(mocked_run.call_count, 2)  # 必须分别查询 GPU 数值和计算进程。
        self.assertEqual(result["gpu_launch_guard_status"], "revalidated_before_cuda_load")  # 正式状态应升级为二次已验证。
        self.assertEqual(result["gpu_recheck_status"], "validated")  # 机器可读二次状态必须明确成功。
        self.assertEqual(result["gpu_recheck_used_mib"], "12")  # 记录二次查询而不是沿用启动器旧显存值。
        self.assertEqual(result["gpu_recheck_util_percent"], "0")  # 记录二次查询看到的实时利用率。
        self.assertTrue(result["gpu_recheck_utc"].endswith("Z"))  # 二次完成时间必须保持 UTC Z 格式。

    def test_recheck_allows_only_current_python_pid(self) -> None:  # CUDA 可用性探测有时会让驱动先登记当前进程。
        """当前进程自身可以出现，但任何其他 PID 都必须拒绝。"""  # 避免把自身误报为外部抢占。

        gpu_row = "2, NVIDIA A100-SXM4-80GB, GPU-1234-abcd, 81920, 24, 1\n"  # 模拟仍低于阈值的实时占用。
        current_pid_row = f"{os.getpid()}\n"  # 模拟 nvidia-smi 只返回当前测试进程。
        with patch("fastvoicegrad.train.subprocess.run", side_effect=[completed(gpu_row), completed(current_pid_row)]):  # 两次查询都使用可控结果。
            result = revalidate_gpu_idle_before_cuda_load(make_attestation())  # 当前进程不应导致误拒绝。
        self.assertEqual(result["gpu_recheck_compute_process_state"], "none_or_current_process_only")  # 动态日志明确说明允许范围。

    def test_recheck_rejects_gpu_that_became_busy(self) -> None:  # 初检后被其他任务占用时必须在模型加载前停止。
        """实时显存或利用率越界不能沿用旧空闲证明。"""  # 先覆盖显存超过 256 MiB 的情况。

        busy_gpu_row = "2, NVIDIA A100-SXM4-80GB, GPU-1234-abcd, 81920, 4096, 87\n"  # 模拟另一任务已经大量占卡。
        with patch("fastvoicegrad.train.subprocess.run", return_value=completed(busy_gpu_row)) as mocked_run:  # 第一条 GPU 查询就足以拒绝。
            with self.assertRaisesRegex(RuntimeError, "became busy"):  # 错误必须清楚指出初检后发生抢占。
                revalidate_gpu_idle_before_cuda_load(make_attestation())  # 不应继续到计算进程查询。
        self.assertEqual(mocked_run.call_count, 1)  # 数值已越界时无需再执行第二条外部查询。

    def test_recheck_rejects_foreign_compute_pid(self) -> None:  # 低显存也不能掩盖其他用户计算进程。
        """任意非当前 PID 都表示所选卡不再可用。"""  # 只检查 PID，不读取其他用户命令行。

        gpu_row = "2, NVIDIA A100-SXM4-80GB, GPU-1234-abcd, 81920, 32, 0\n"  # 模拟数值阈值表面仍为空闲。
        with patch("fastvoicegrad.train.subprocess.run", side_effect=[completed(gpu_row), completed("98765\n")]):  # 第二次查询返回外部 PID。
            with self.assertRaisesRegex(RuntimeError, "another compute process"):  # 外部计算进程必须失败关闭。
                revalidate_gpu_idle_before_cuda_load(make_attestation())  # 禁止仅凭低显存继续加载模型。

    def test_recheck_rejects_identity_change(self) -> None:  # GPU index、型号、UUID和总显存都必须保持一致。
        """查询到另一张 GPU 时不能把它误当成最初选择的设备。"""  # 这里用 UUID 变化代表错卡。

        wrong_gpu_row = "2, NVIDIA A100-SXM4-80GB, GPU-other-card, 81920, 0, 0\n"  # 模拟驱动返回不同硬件身份。
        with patch("fastvoicegrad.train.subprocess.run", return_value=completed(wrong_gpu_row)):  # 只提供第一条身份查询结果。
            with self.assertRaisesRegex(RuntimeError, "identity changed"):  # 错卡必须在进程查询之前停止。
                revalidate_gpu_idle_before_cuda_load(make_attestation())  # 不允许通过相同 index 猜测为同一卡。


if __name__ == "__main__":  # 允许开发者日后在合规环境显式单独调用本测试文件。
    unittest.main()  # 本行只有直接执行文件时才运行，正常导入不会自动启动测试。
