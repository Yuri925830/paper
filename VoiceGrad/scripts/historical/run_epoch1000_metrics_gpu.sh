#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。
# 历史 epoch-1000 GPU 指标入口：按顺序计算 UTMOS、CER、SVA。
# 三个模型依次加载而非同时常驻，可以降低显存峰值。DNSMOS 是 CPU 指标，由分片脚本
# 另算。运行前可用 CUDA_VISIBLE_DEVICES 选择一块空闲 GPU。
set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 本次要评估的推理结果目录，必须与生成 WAV 的实验版本一致。
INFERENCE_DIR="$PROJECT_ROOT/exp/evaluation/voicegrad30_text_unseen_epoch1000_public_hifigan"
# 逐句客观指标和最终汇总的存放目录。
METRICS_DIR="$INFERENCE_DIR/metrics"

# GPU 由调用者通过 CUDA_VISIBLE_DEVICES 选择，不把某台实验室主机写进公开代码。
cd "$PROJECT_ROOT"

# 等待推理完成报告出现；WAV 可能仍在写入时不启动评价。
while [[ ! -f "$INFERENCE_DIR/inference_report.json" ]]; do
    # 只有批量推理完整结束才会出现该报告；轮询可避免评价半成品。
    sleep 30
done

# 自然度：UTMOS 越高越好。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
"$PYTHON_BIN" -m evaluation.evaluate_metrics \
    --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
    --metric utmos --batch-size 4 --device cuda

# 内容保持：wav2vec2 识别后计算字符错误率，越低越好。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
"$PYTHON_BIN" -m evaluation.evaluate_metrics \
    --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
    --metric cer --batch-size 4 --device cuda

# 目标音色：ECAPA 比较转换语音与 target reference，准确率越高越好。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
"$PYTHON_BIN" -m evaluation.evaluate_metrics \
    --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
    --metric sva --batch-size 81 --device cuda

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_EPOCH1000_GPU_METRICS_8100_PASSED"
