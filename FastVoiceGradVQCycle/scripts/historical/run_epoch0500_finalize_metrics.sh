#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。

# 等三项 GPU 指标和三份 DNSMOS 分片全部完成，再合并并生成最终组会表格。
set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 本次要评估的推理结果目录，必须与生成 WAV 的实验版本一致。
INFERENCE_DIR="$PROJECT_ROOT/exp/evaluation/voicegrad30_text_unseen_epoch0500_public_hifigan"
# 逐句客观指标和最终汇总的存放目录。
METRICS_DIR="$INFERENCE_DIR/metrics"
# 所有 DNSMOS 分片的父目录，后续会核对无重复且无缺失。
SHARDS_DIR="$INFERENCE_DIR/metrics_dnsmos_shards"

# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "$PROJECT_ROOT"

# CSV 第一行是列名，因此 8,100 个结果应有 8,101 行；分片同理要加一行表头。
csv_has_exact_lines() {
    # 当前函数要核对的文件路径。
    local path="$1"
    # CSV 应有的精确行数，包含一行表头。
    local expected_lines="$2"
    # 此处要求普通文件存在；按整数判断相等；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    [[ -f "$path" ]] && [[ "$(wc -l < "$path")" -eq "$expected_lines" ]]
}

# 持续等待完整指标文件，只有全部满足行数条件才跳出循环。
while true; do
    # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
    # 命令第 1 行：此处检查该指标 CSV 的行数等于结果数加一行表头；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    # 命令第 2 行：此处检查该指标 CSV 的行数等于结果数加一行表头；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    # 命令第 3 行：此处检查该指标 CSV 的行数等于结果数加一行表头；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    # 命令第 4 行：此处检查该指标 CSV 的行数等于结果数加一行表头；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    # 命令第 5 行：此处检查该指标 CSV 的行数等于结果数加一行表头；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    # 命令第 6 行：此处检查该指标 CSV 的行数等于结果数加一行表头；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    if csv_has_exact_lines "$METRICS_DIR/utmos.csv" 8101 \
        && csv_has_exact_lines "$METRICS_DIR/cer.csv" 8101 \
        && csv_has_exact_lines "$METRICS_DIR/sva.csv" 8101 \
        && csv_has_exact_lines "$SHARDS_DIR/shard_00/dnsmos.csv" 2674 \
        && csv_has_exact_lines "$SHARDS_DIR/shard_01/dnsmos.csv" 2674 \
        && csv_has_exact_lines "$SHARDS_DIR/shard_02/dnsmos.csv" 2755; then
        # 所需结果已满足完成条件，结束等待循环并继续下一阶段。
        break
    fi
    # 等待 30 秒再核对上游产物，避免频繁轮询占用资源。
    sleep 30
done

# 合并程序不只拼文件，还会要求 pair_id 恰好是 0~8099，任何重复或缺失都会报错停止。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：合并 DNSMOS 分片并核对配对编号恰好覆盖完整集合，无重复无缺失；-u 让日志及时写出。
# 命令第 2 行：--shards：后续逐行给出需要合并的 DNSMOS CSV。
# 命令第 3 行：列出这一份 DNSMOS 分片文件；合并器后续还会检查配对编号覆盖和重复情况。
# 命令第 4 行：列出这一份 DNSMOS 分片文件；合并器后续还会检查配对编号覆盖和重复情况。
# 命令第 5 行：列出这一份 DNSMOS 分片文件；合并器后续还会检查配对编号覆盖和重复情况。
# 命令第 6 行：--output：指定合并后的唯一结果文件。
"$PYTHON_BIN" -m evaluation.merge_dnsmos_shards \
    --shards \
        "$SHARDS_DIR/shard_00/dnsmos.csv" \
        "$SHARDS_DIR/shard_01/dnsmos.csv" \
        "$SHARDS_DIR/shard_02/dnsmos.csv" \
    --output "$METRICS_DIR/dnsmos.csv"

# dnsmos.csv 已完整存在，所以评价程序不会重复计算；它会复核四份 8,100 行 CSV，
# 然后生成 objective_summary.json 和方便组会复制的 objective_summary.csv。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
"$PYTHON_BIN" -m evaluation.evaluate_metrics \
    --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
    --metric dnsmos --batch-size 4 --device cpu

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_EPOCH0500_OBJECTIVE_EVALUATION_COMPLETE"
