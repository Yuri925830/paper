# 将两次学生训练的前两轮损失放在一起；只读模型结果，不启动或延长训练。
import csv  # 读取原脚本生成的逐轮损失表。
import hashlib  # 记录真正用于比较的CSV字节身份。
import json  # 保存未四舍五入的比较数值。
import math  # 拒绝NaN或无穷大，防止曲线静默漏点。
import sys  # 阻止在Windows本地计算项目结果。
from pathlib import Path  # 所有输入和输出都使用出租服务器的明确路径。

ROOT = Path('/root/FsatVoiceGrad')  # 本项目的唯一计算位置。


def main():  # 仅在远端直接执行时整理结果。
    if not sys.platform.startswith('linux') or not ROOT.is_dir():
        raise RuntimeError('Run only on the registered rental server.')  # 不提供本地执行路径。
    sys.path.insert(0, str(ROOT))  # 使用现场的服务器身份检查。
    from voicegrad.runtime_guard import assert_rented_server
    assert_rented_server(ROOT)  # 确认并非禁止使用的实验室服务器。
    import matplotlib  # 曲线完全依据训练日志，不生成示意数据。
    matplotlib.use('Agg')  # 使用无窗口绘图，不需要桌面会话。
    import matplotlib.pyplot as plt  # 输出可供检查的静态曲线图。
    run = ROOT / 'exp/diagnostics/student_public_vocoder_train_20260906'  # 公开声码器学生训练的新目录。
    baseline = ROOT / 'exp/checkpoints/fastvoicegrad_vctk_text_unseen_repair_x0_sqrt_20260906'  # 原自训声码器组保持只读。
    series = {}  # 各组只取前两轮，防止混入旧组未保存的第三轮。
    input_hashes = {}  # 原始损失表只读，汇总保留可追溯指纹。
    for name, directory in (('Custom G', baseline), ('Public VCTK_V1', run)):
        with (directory / 'loss.csv').open(newline='') as handle:
            rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]  # 保留相同字段定义。
        series[name] = [row for row in rows if 1 <= row['epoch'] <= 2]  # 当前组尚未完成第二轮时，只显示已有点。
        series[name].sort(key=lambda row: row['epoch'])  # 不依赖文本行顺序判断轮次。
        if [row['epoch'] for row in series[name]] not in ([1.0], [1.0, 2.0]):  # 不能接受重复轮次或单独第二轮。
            raise RuntimeError(f'Invalid completed epoch sequence: {name}')
        for row in series[name]:  # 每个有效点必须覆盖一整轮训练和完整固定测试集。
            if row['global_step'] != row['epoch'] * 1230 or row['test_examples'] != 100 or row['test_batches'] != 4 or not all(math.isfinite(value) for value in row.values()):
                raise RuntimeError(f'Incomplete or non-finite epoch row: {name}')
        input_hashes[name] = hashlib.sha256((directory / 'loss.csv').read_bytes()).hexdigest()
    result = {'scope': 'diagnostic_student_training_vocoder_swap_only', 'input_csv_sha256': input_hashes, 'series': series, 'epoch1_to_epoch2': {}, 'caution': 'Each G changes both real and fake waveforms; D evolves during training. Compare trends, not absolute loss as a universal quality scale. Two epochs are a short diagnostic, not formal100 training. Confirm saved checkpoints and the diagnostic contract separately before interpreting a completed run.'}
    for name, rows in series.items():
        if len(rows) != 2:  # 未完成两轮时不推算第二轮，也不提前下结论。
            continue
        first, second = sorted(rows, key=lambda row: row['epoch'])  # 保持实际时间顺序。
        result['epoch1_to_epoch2'][name] = {key: {'epoch1': first[key], 'epoch2': second[key], 'delta': second[key] - first[key], 'relative_percent': 100 * (second[key] - first[key]) / abs(first[key]) if first[key] else None} for key in first if key.startswith(('train_', 'test_')) and key not in ('test_examples', 'test_batches')}
    fields = [('student_total', 'Student total'), ('student_adv', 'Adversarial'), ('fm_weighted', 'Feature matching (x2)'), ('dist_weighted', 'Distillation (x45)'), ('discriminator', 'Discriminator')]  # 用已经加权的分项解释总损失变化。
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))  # 五项指标共用一张图，避免只看总损失。
    colors = {'Custom G': '#a34c2f', 'Public VCTK_V1': '#2368a0'}  # 颜色只用于区分声码器组。
    for axis, (field, title) in zip(axes.flat, fields):
        for name, rows in series.items():
            for prefix, style in (('train_', '-'), ('test_', '--')):  # 实线训练、虚线固定测试。
                axis.plot([row['epoch'] for row in rows], [row[prefix + field] for row in rows], style, marker='o', color=colors[name], label=name + ' ' + prefix[:-1])
        axis.set_title(title)  # 标题注明该损失项的实际权重。
        axis.set_xlabel('Completed epoch')  # 横轴只显示完成的轮次。
        axis.set_xticks([1, 2])  # 不暗示尚未训练的长期走势。
        axis.grid(alpha=0.2)  # 轻网格方便比较数值。
    axes.flat[-1].axis('off')  # 最后一个面板放图例和实验边界。
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='upper left')
    axes.flat[-1].text(0, 0.25, 'Design: same teacher start and training recipe.\nIntended change: frozen vocoder weights.\nShort diagnostic: at most 2 epochs.\nAbsolute loss is not an audio quality score.', transform=axes.flat[-1].transAxes, fontsize=10, va='top')
    fig.suptitle('Student training: custom HiFi-GAN vs public VCTK_V1')
    fig.tight_layout()
    fig.savefig(run / 'vocoder_training_comparison.png', dpi=160)  # 仅更新本次诊断的汇总图，不覆盖历史训练图。
    plt.close(fig)  # 释放绘图占用。
    with (run / 'vocoder_training_comparison.json').open('w') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)  # 汇总可随着第二轮完成更新，原始CSV保持不变。
    print(json.dumps(result['epoch1_to_epoch2'], ensure_ascii=False, indent=2))  # 终端同步输出真实的逐轮变化。


if __name__ == '__main__':
    main()
