#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。
# 历史 epoch-1000 DNSMOS 分片入口。
# 用法：bash run_epoch1000_dnsmos_shard.sh START LIMIT SHARD_NAME
# START 是 pairs.csv 中的起始行编号，LIMIT 是本机处理数量，SHARD_NAME 是输出子目录。
# 三台 CPU 机器可分别处理互不重叠的 0–2672、2673–5345、5346–8099。
# VOICEGRAD_PROJECT_ROOT 和 VOICEGRAD_PYTHON 都可由调用者设置，脚本不依赖主机名。
set -euo pipefail

# 此处按整数判断不等；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ "$#" -ne 3 ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "Usage: $0 START LIMIT SHARD_NAME" >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 2
fi

# 当前指标分片从第几条配对开始，采用零起始索引。
START="$1"
# 当前分片最多处理多少条配对。
LIMIT="$2"
# 当前 DNSMOS 分片的独立标签。
SHARD_NAME="$3"
# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 本次要评估的推理结果目录，必须与生成 WAV 的实验版本一致。
INFERENCE_DIR="$PROJECT_ROOT/exp/evaluation/voicegrad30_text_unseen_epoch1000_public_hifigan"
# 当前分片指标输出目录。
SHARD_DIR="$INFERENCE_DIR/metrics_dnsmos_shards/$SHARD_NAME"

# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "$PROJECT_ROOT"

# 等待推理完成报告出现；WAV 可能仍在写入时不启动评价。
while [[ ! -f "$INFERENCE_DIR/inference_report.json" ]]; do
    # 报告只在 8,100 对推理完成后原子写出；没看到它就不要读可能仍在写的 WAV。
    sleep 30
done

# DNSMOS 只使用 CPU；--start/--limit 确保每台机器负责的 pair_id 不重叠。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--start：当前分片的零起始配对编号；--limit：当前分片处理的配对数量。
# 命令第 4 行：--batch-size：每批样本数，影响内存和更新统计；--dnsmos-workers：DNSMOS 的 CPU 工作进程数；--device：在租用服务器选择 GPU(cuda) 或 CPU。
"$PYTHON_BIN" -m evaluation.evaluate_metrics \
    --inference-dir "$INFERENCE_DIR" --output-dir "$SHARD_DIR" \
    --metric dnsmos --start "$START" --limit "$LIMIT" \
    --batch-size 4 --dnsmos-workers 3 --device cpu

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "EPOCH1000_DNSMOS_SHARD_PASSED name=$SHARD_NAME start=$START limit=$LIMIT"
