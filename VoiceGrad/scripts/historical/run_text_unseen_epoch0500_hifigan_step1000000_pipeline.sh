#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。

set -euo pipefail

# 保留当时实验室服务器的历史绝对路径，仅供追溯；当前租用任务不可直接执行这一旧入口。
PROJECT_ROOT="/nfs/speechst2/storage/rich_925/VoiceGrad"
# 保留当时实验室服务器的历史绝对路径，仅供追溯；当前租用任务不可直接执行这一旧入口。
PYTHON_BIN="/nfs/speechst2/storage/rich_925/miniforge3/envs/vg_eval/bin/python"
# 本次实际选用的教师权重路径，必须与评估标签中的 epoch 对应。
TEACHER_CHECKPOINT="$PROJECT_ROOT/exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt"
# 所选声码器权重及配置所在目录。
VOCODER_DIR="$PROJECT_ROOT/exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
# 本次实际声码器生成器文件；历史与当前入口的步数不同，以下赋值原样保留。
VOCODER_CHECKPOINT="$VOCODER_DIR/generator_step_1000000.pt"
# 与生成器配套的结构和音频配置。
VOCODER_CONFIG="$VOCODER_DIR/config.json"
# 本次要评估的推理结果目录，必须与生成 WAV 的实验版本一致。
INFERENCE_DIR="$PROJECT_ROOT/exp/evaluation/voicegrad30_text_unseen_epoch0500_hifigan_step1000000"
# 逐句客观指标和最终汇总的存放目录。
METRICS_DIR="$INFERENCE_DIR/metrics"
# 所有 DNSMOS 分片的父目录，后续会核对无重复且无缺失。
SHARDS_DIR="$INFERENCE_DIR/metrics_dnsmos_shards"
# 本次历史流水线日志文件名前缀。
LOG_PREFIX="$PROJECT_ROOT/exp/logs/text_unseen_epoch0500_hifigan_step1000000"
# 保存当前阶段状态的文本文件。
STATUS_FILE="${LOG_PREFIX}.status"

# 这份历史实验锁定的教师字节指纹，不能当作当前教师身份。
EXPECTED_TEACHER_SHA256="3477b95abe2a8390351c9b64083f58a928139d36d96f61aa844210202427fbc1"
# 历史实验锁定的声码器字节指纹。
EXPECTED_VOCODER_SHA256="e2b4d370bc435d183083563b261a2b192d5a5bab1e23b74002affcf685df8662"
# 历史实验锁定的声码器配置字节指纹。
EXPECTED_CONFIG_SHA256="ba53d3c693dd5b7eb1046e579b0aa3e079d1d02988d48e90779e188d322540c4"

# 保存当前历史流程阶段和实时时间。
write_status() {
    # 将当前时间与阶段写到状态文件，并同时输出到日志，便于追查任务停在哪一步。
    printf '%s %s\n' "$(date '+%F %T %Z')" "$1" | tee "$STATUS_FILE"
}

# 从文件摘要命令中只保留十六进制 SHA-256。
sha256_only() {
    # 按原始字节计算文件 SHA-256，并仅提取小写摘要供后续精确比较。
    sha256sum "$1" | cut -d ' ' -f 1
}

# 逐个核对历史模型/配置的实际字节与预登记摘要。
check_hash() {
    # 当前函数要核对的文件路径。
    local path="$1"
    # 该文件预先登记的正确 SHA-256。
    local expected="$2"
    # actual：从现场文件重新计算的 SHA-256。
    local actual
    # 从现场文件重新计算的 SHA-256。
    actual="$(sha256_only "$path")"
    # 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    if [[ "$actual" != "$expected" ]]; then
        # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
        echo "ERROR: SHA-256 mismatch: $path" >&2
        # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
        echo "expected=$expected" >&2
        # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
        echo "actual=$actual" >&2
        # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
        exit 3
    fi
}

# 进程退出时检查是否失败，失败则更新历史状态文件。
on_exit() {
    # 刚才命令的退出码，保留它以免错误被后续日志覆盖。
    local status="$?"
    # 此处按整数判断不等；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    if [[ "$status" -ne 0 ]]; then
        # 把当前阶段更新为 "FAILED exit_status=$status"，供外部查看历史流程进度。
        write_status "FAILED exit_status=$status"
    fi
}
# 登记 on_exit 作为 EXIT 事件处理函数，及时记录失败或清理本次临时目录。
trap on_exit EXIT

# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "$PROJECT_ROOT"
# 设为 1 后及时刷新 Python 日志，便于远程发现失败。
export PYTHONUNBUFFERED=1

# 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ "$(hostname -s)" != "speech301" ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "ERROR: this evaluated pipeline is assigned to speech301." >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 2
fi

# 此处检查路径是否已存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -e "$INFERENCE_DIR" ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "ERROR: output directory already exists: $INFERENCE_DIR" >&2
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "Refusing to overwrite an existing evaluation." >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 4
fi

# 核对这一份模型或配置与预登记指纹一致，避免误用同名不同内容文件。
check_hash "$TEACHER_CHECKPOINT" "$EXPECTED_TEACHER_SHA256"
# 核对这一份模型或配置与预登记指纹一致，避免误用同名不同内容文件。
check_hash "$VOCODER_CHECKPOINT" "$EXPECTED_VOCODER_SHA256"
# 核对这一份模型或配置与预登记指纹一致，避免误用同名不同内容文件。
check_hash "$VOCODER_CONFIG" "$EXPECTED_CONFIG_SHA256"

# 把当前阶段更新为 "INFERENCE_RUNNING"，供外部查看历史流程进度。
write_status "INFERENCE_RUNNING"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：仅向当前子进程暴露指定 GPU；历史编号保留作记录，当前租用任务不能直接照搬。
# 命令第 2 行：--checkpoint：指定本次真正加载的权重文件。
# 命令第 3 行：--vocoder-checkpoint：指定声码器生成器权重。
# 命令第 4 行：--vocoder-config：指定声码器结构与声学参数。
# 命令第 5 行：--output-dir：指定本阶段的独立输出目录。
# 命令第 6 行：--batch-size：每批样本数，影响内存和更新统计。
# 命令第 7 行：--seed：固定随机噪声以便重复比较。
# 命令第 8 行：--device：在租用服务器选择 GPU(cuda) 或 CPU。
# 命令第 9 行：将本阶段标准输出与标准错误写入指定日志；& 表示后台进程，后续必须检查退出码。
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" infer_voicegrad30_paper_eval.py \
    --checkpoint "$TEACHER_CHECKPOINT" \
    --vocoder-checkpoint "$VOCODER_CHECKPOINT" \
    --vocoder-config "$VOCODER_CONFIG" \
    --output-dir "$INFERENCE_DIR" \
    --batch-size 4 \
    --seed 1234 \
    --device cuda \
    > "${LOG_PREFIX}_inference.log" 2>&1

# 此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
[[ -f "$INFERENCE_DIR/inference_report.json" ]]
# 此处按整数判断相等；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
[[ "$(find "$INFERENCE_DIR/wav" -type f -name '*.wav' | wc -l)" -eq 8100 ]]

# 把当前阶段更新为 "METRICS_RUNNING"，供外部查看历史流程进度。
write_status "METRICS_RUNNING"
# 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
mkdir -p "$METRICS_DIR" "$SHARDS_DIR"

(
    # 仅向当前子进程暴露指定 GPU；历史编号保留作记录，当前租用任务不能直接照搬。
    export CUDA_VISIBLE_DEVICES=1
    # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
    # 命令第 1 行：调用历史版客观指标入口，按下一行选择 UTMOS/DNSMOS/CER/SVA；此旧路径仅保留审计，不作为当前启动入口。
    # 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
    # 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
    "$PYTHON_BIN" evaluate_voicegrad30_paper_metrics.py \
        --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
        --metric utmos --batch-size 4 --device cuda
    # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
    # 命令第 1 行：调用历史版客观指标入口，按下一行选择 UTMOS/DNSMOS/CER/SVA；此旧路径仅保留审计，不作为当前启动入口。
    # 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
    # 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
    "$PYTHON_BIN" evaluate_voicegrad30_paper_metrics.py \
        --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
        --metric cer --batch-size 4 --device cuda
    # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
    # 命令第 1 行：调用历史版客观指标入口，按下一行选择 UTMOS/DNSMOS/CER/SVA；此旧路径仅保留审计，不作为当前启动入口。
    # 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
    # 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
    "$PYTHON_BIN" evaluate_voicegrad30_paper_metrics.py \
        --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
        --metric sva --batch-size 81 --device cuda
# 将本阶段标准输出与标准错误写入指定日志；& 表示后台进程，后续必须检查退出码。
) > "${LOG_PREFIX}_gpu_metrics.log" 2>&1 &
# 刚启动的 GPU 指标子进程编号，用来等待真实结束。
GPU_METRICS_PID="$!"

# 对指定配对区间计算 DNSMOS，各分片独立输出。
run_dnsmos_shard() {
    # 此 DNSMOS 分片的首条配对索引。
    local start="$1"
    # 此分片要处理的配对数量。
    local limit="$2"
    # 当前阶段或分片名称，用于区分日志与输出。
    local name="$3"
    # 当前分片写指标的位置。
    local output_dir="$SHARDS_DIR/$name"
    # 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
    mkdir -p "$output_dir"
    # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
    # 命令第 1 行：调用历史版客观指标入口，按下一行选择 UTMOS/DNSMOS/CER/SVA；此旧路径仅保留审计，不作为当前启动入口。
    # 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
    # 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--start：当前分片的零起始配对编号；--limit：当前分片处理的配对数量。
    # 命令第 4 行：--batch-size：每批样本数，影响内存和更新统计；--dnsmos-workers：DNSMOS 的 CPU 工作进程数；--device：在租用服务器选择 GPU(cuda) 或 CPU。
    # 命令第 5 行：将本阶段标准输出与标准错误写入指定日志；& 表示后台进程，后续必须检查退出码。
    "$PYTHON_BIN" evaluate_voicegrad30_paper_metrics.py \
        --inference-dir "$INFERENCE_DIR" --output-dir "$output_dir" \
        --metric dnsmos --start "$start" --limit "$limit" \
        --batch-size 4 --dnsmos-workers 3 --device cpu \
        > "${LOG_PREFIX}_dnsmos_${name}.log" 2>&1
}

# 启动指定起点和数量的 DNSMOS 分片；末尾 & 表示后台执行，随后必须 wait 检查结果。
run_dnsmos_shard 0 2673 shard_00 &
# 第一个 DNSMOS 分片子进程编号。
DNSMOS_PID_0="$!"
# 启动指定起点和数量的 DNSMOS 分片；末尾 & 表示后台执行，随后必须 wait 检查结果。
run_dnsmos_shard 2673 2673 shard_01 &
# 第二个 DNSMOS 分片子进程编号。
DNSMOS_PID_1="$!"
# 启动指定起点和数量的 DNSMOS 分片；末尾 & 表示后台执行，随后必须 wait 检查结果。
run_dnsmos_shard 5346 2754 shard_02 &
# 第三个 DNSMOS 分片子进程编号。
DNSMOS_PID_2="$!"

# 等待指定子进程真实结束，并传播失败状态；后台启动成功不等于指标已经算完。
wait "$GPU_METRICS_PID"
# 等待指定子进程真实结束，并传播失败状态；后台启动成功不等于指标已经算完。
wait "$DNSMOS_PID_0"
# 等待指定子进程真实结束，并传播失败状态；后台启动成功不等于指标已经算完。
wait "$DNSMOS_PID_1"
# 等待指定子进程真实结束，并传播失败状态；后台启动成功不等于指标已经算完。
wait "$DNSMOS_PID_2"

# 把当前阶段更新为 "FINALIZING"，供外部查看历史流程进度。
write_status "FINALIZING"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：核对并合并历史 DNSMOS 分片，要求所有配对完整且无重复。
# 命令第 2 行：--shards：后续逐行给出需要合并的 DNSMOS CSV。
# 命令第 3 行：列出这一份 DNSMOS 分片文件；合并器后续还会检查配对编号覆盖和重复情况。
# 命令第 4 行：列出这一份 DNSMOS 分片文件；合并器后续还会检查配对编号覆盖和重复情况。
# 命令第 5 行：列出这一份 DNSMOS 分片文件；合并器后续还会检查配对编号覆盖和重复情况。
# 命令第 6 行：--output：指定合并后的唯一结果文件。
# 命令第 7 行：将本阶段标准输出与标准错误写入指定日志；& 表示后台进程，后续必须检查退出码。
"$PYTHON_BIN" merge_dnsmos_shards.py \
    --shards \
        "$SHARDS_DIR/shard_00/dnsmos.csv" \
        "$SHARDS_DIR/shard_01/dnsmos.csv" \
        "$SHARDS_DIR/shard_02/dnsmos.csv" \
    --output "$METRICS_DIR/dnsmos.csv" \
    > "${LOG_PREFIX}_finalize.log" 2>&1

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：调用历史版客观指标入口，按下一行选择 UTMOS/DNSMOS/CER/SVA；此旧路径仅保留审计，不作为当前启动入口。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单；--output-dir：指定本阶段的独立输出目录。
# 命令第 3 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好；--batch-size：每批样本数，影响内存和更新统计；--device：在租用服务器选择 GPU(cuda) 或 CPU。
# 命令第 4 行：将本阶段标准输出与标准错误写入指定日志；& 表示后台进程，后续必须检查退出码。
"$PYTHON_BIN" evaluate_voicegrad30_paper_metrics.py \
    --inference-dir "$INFERENCE_DIR" --output-dir "$METRICS_DIR" \
    --metric dnsmos --batch-size 4 --device cpu \
    >> "${LOG_PREFIX}_finalize.log" 2>&1

# 逐一核对四项客观指标的文件和结果数，缺少任何一项就不能发布最终汇总。
for metric in utmos dnsmos cer sva; do
    # 此处按整数判断相等；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    [[ "$(wc -l < "$METRICS_DIR/$metric.csv")" -eq 8101 ]]
done
# 此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
[[ -f "$METRICS_DIR/objective_summary.csv" ]]
# 此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
[[ -f "$METRICS_DIR/objective_summary.json" ]]

# 把当前阶段更新为 "COMPLETE"，供外部查看历史流程进度。
write_status "COMPLETE"
# 成功结束后取消退出错误处理，避免正常完成被再次登记为失败。
trap - EXIT
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_TEXT_UNSEEN_HIFIGAN_STEP1000000_COMPLETE"
