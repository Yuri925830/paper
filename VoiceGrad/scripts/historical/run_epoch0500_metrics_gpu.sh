#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。

# 任意一步报错就停止；未定义变量也报错；管道中任一程序失败都算失败。
set -euo pipefail

# 从脚本位置推导仓库根目录；也允许环境变量覆盖 Python 和根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 本次要评估的推理结果目录，必须与生成 WAV 的实验版本一致。
INFERENCE_DIR="$PROJECT_ROOT/exp/evaluation/voicegrad30_text_unseen_epoch0500_public_hifigan"
# 逐句客观指标和最终汇总的存放目录。
METRICS_DIR="$INFERENCE_DIR/metrics"

# 具体使用哪块 GPU 由调用者设置 CUDA_VISIBLE_DEVICES，本脚本不假定服务器名称。
cd "$PROJECT_ROOT"

# 批量推理最后才会原子写 inference_report.json；看到它就表示 8,100 条已经完整结束。
while [[ ! -f "$INFERENCE_DIR/inference_report.json" ]]; do
    # 等待 30 秒再核对上游产物，避免频繁轮询占用资源。
    sleep 30
done

# 运行本脚本前可由外层命令用 CUDA_VISIBLE_DEVICES 选择一块空闲 GPU。
# 三个神经指标依次加载、依次释放模型，避免 UTMOS、ASR、ECAPA 同时占显存。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
"$PYTHON_BIN" -m evaluation.evaluate_metrics \
    --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
    --metric utmos --batch-size 4 --device cuda

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
"$PYTHON_BIN" -m evaluation.evaluate_metrics \
    --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
    --metric cer --batch-size 4 --device cuda

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
"$PYTHON_BIN" -m evaluation.evaluate_metrics \
    --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
    --metric sva --batch-size 81 --device cuda

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_GPU_METRICS_8100_PASSED"
