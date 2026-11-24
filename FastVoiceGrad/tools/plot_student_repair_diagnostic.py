# 用真实固定样本测量绘制短程修复对照图，并披露它只属于训练来源诊断。
"""Plot the paired, training-only check; never substitute it for unseen eval."""
# 读取命令行指定的诊断目录，不在此重新训练或生成测试集指标。
import argparse
# 哈希用于记录作图时实际源码与汇总文件的身份。
import hashlib
# JSON 保存和读取每次固定样本诊断的数值。
import json
# 通过 Git 的只读命令记录提交及尚未提交的差异。
import subprocess
# 保存本次记录的时间，明确它发生在短程对照结束之后。
import time
# Path 负责组合诊断文件和项目源码路径。
from pathlib import Path

# 绘图库把已经测得的误差与幅度画成曲线，图中没有虚构数据。
import matplotlib
# 租用服务器没有桌面显示，使用可直接输出 PNG 的后端。
matplotlib.use("Agg")
# pyplot 提供画布、曲线、图例和保存功能。
import matplotlib.pyplot as plt

# 即使仅作图，也按项目约束在登记的租用服务器执行。
from voicegrad.runtime_guard import assert_rented_server


# 主流程：读固定样本诊断、作图，再保存“作图时”源码快照。
def main():
    # 接收路径参数，避免把某次实验目录藏在程序内部。
    parser = argparse.ArgumentParser()
    # 项目根目录用于身份检查及读取源码指纹。
    parser.add_argument("--project-root", type=Path, default=Path("/root/FsatVoiceGrad"))
    # 必须明确提供对照实验目录，防止混用旧日志。
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    # 将用户参数转换为可按名称访问的对象。
    args = parser.parse_args()
    # 主机身份不匹配就报错，不允许回退到本地计算。
    assert_rented_server(args.project_root)
    # root 指源码位置；output 指包含 fixed_dev.jsonl 的诊断目录。
    root, output = args.project_root, args.diagnostic_dir
    # 每行是一次固定输入测量；逐行解码，保留原始测量字段。
    rows = [json.loads(line) for line in (output / "fixed_dev.jsonl").read_text().splitlines()]
    # 左图看误差，右图看预测幅度；两项共同帮助识别输出塌缩。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # 旧设置用红色，标签同时列出目标与权重，避免误认成单变量消融。
    styles = [("legacy_adjacent_nominal_alpha", "Original target + weight", "#b74739"),
              # 修复设置用蓝色；两条曲线采用同一组真实参考 mel。
              ("clean_x0_sqrt_alpha_bar", "Clean target + sqrt(alpha_bar)", "#246b91")]
    # 分别整理旧设置和修复设置的固定输入测量。
    for arm, label, color in styles:
        # 同一步重复记录时取最后一条，避免结束测量重复画两次。
        selected = {r["step"]: r for r in rows if r["arm"] == arm}
        # 按更新步数排序，保证曲线由训练开始向结束连接。
        ordered = [selected[s] for s in sorted(selected)]
        # 左侧使用平均绝对误差，右侧使用预测频谱的均方根。
        for axis, metric in zip(axes, ["fixed_dev_mel_l1", "prediction_rms"]):
            # 横轴是真实更新步数，纵轴直接读取相应记录，不进行平滑或重缩放。
            axis.plot([r["step"] for r in ordered], [r[metric] for r in ordered],
                      # 圆点标记实际测量位置，同一实验组使用同一种颜色。
                      marker="o", color=color, label=label)
    # 标明误差越低越好，避免将训练损失和这个固定参考误差混为一谈。
    axes[0].set_title("Fixed-sample mel error (lower is better)")
    # 该误差位于标准化后的 mel 空间，没有声压或分贝单位。
    axes[0].set_ylabel("Mean absolute error")
    # 第二幅图专门观察预测频谱是否被压缩到接近零。
    axes[1].set_title("Predicted mel scale")
    # RMS 是均方根，只表示标准化频谱数值大小，不能直接解释成音频响度。
    axes[1].set_ylabel("RMS in normalized mel space")
    # 真实参考幅度固定不变，画虚线方便判断预测幅度偏离多少。
    axes[1].axhline(rows[0]["reference_rms"], color="gray", linestyle="--", label="Clean reference RMS")
    # 为两张图添加相同的横轴说明和便于读数的网格。
    for axis in axes:
        # step 表示优化器已经更新的次数，不是扩散模型的加噪时间步。
        axis.set_xlabel("Optimizer step")
        # 使用浅网格，防止背景线遮盖真实曲线。
        axis.grid(alpha=.2)
        # 图例区分实验组及固定参考线。
        axis.legend(fontsize=8)
    # 标题提醒两组具有共同起点、批次和噪声，便于公平解释短程变化。
    fig.suptitle("Paired repair diagnostic: same initialization, batches and noise", fontsize=12)
    # 显式标注样本来自训练集；这张图不能替代正式 text-unseen 客观评估。
    fig.text(.5, .01, "32 training-set diagnostic utterances excluded from pilot updates; not formal text-unseen metrics.",
             # 页脚居中、小字号，给两幅图保留足够空间。
             ha="center", fontsize=8)
    # 给顶部标题和底部样本说明留白，避免文字被裁掉。
    fig.tight_layout(rect=(0, .04, 1, .93))
    # 输出可插入报告的 PNG，180 dpi 保证文字与曲线清晰。
    fig.savefig(output / "paired_reconstruction_check.png", dpi=180)
    # 关闭画布，释放作图占用的内存。
    plt.close(fig)
    # 记录影响数值训练的主要代码，以及监控和对照驱动代码。
    paths = ["fastvoicegrad/objectives.py", "fastvoicegrad/config.py", "fastvoicegrad/train.py",
             # 这两份补充文件分别控制报警统计和成对诊断流程。
             "fastvoicegrad/loss_monitor.py", "tools/diagnose_student_repair.py"]
    # 快照只能证明记录时的状态；没有倒推或伪造实验启动时的源码身份。
    record = {
        # 使用 Unix 秒数保存记录时间，避免不同时区的文字时间歧义。
        "recorded_unix_time": time.time(),
        # 明确披露短程实验启动时未提交，以及期间仅监控记录逻辑发生过调整。
        "timing_note": "Post-pilot snapshot; diagnostic launched before commit. Monitor resume/status bookkeeping was refined during the pilot; numerical train_batch/objective code did not change.",
        # 按原始字节逐文件计算 SHA-256；连注释改动也会改变这些指纹。
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
        # 当前 Git 提交与未提交差异必须同时看，单独的提交号不足以绑定脏工作区。
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        # 差异的哈希辅助识别当时未提交的内容，无法代替完整源码备份。
        "git_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=root)).hexdigest(),
        # 同时锁定汇总结果，防止图与另一份汇总文件混用。
        "summary_sha256": hashlib.sha256((output / "paired_summary.json").read_bytes()).hexdigest(),
    }
    # 保存事后快照为独立文件，不覆盖启动清单 diagnostic_manifest.json。
    (output / "post_pilot_source_snapshot.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    # 打印图的实际位置，方便远程下载与人工查看。
    print(output / "paired_reconstruction_check.png")


# 只有作为命令行程序执行时才作图；被其他模块导入时保持只定义函数。
if __name__ == "__main__":
    # 运行上面定义的完整作图和快照流程。
    main()
