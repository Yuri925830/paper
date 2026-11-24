#!/usr/bin/env bash
# 该入口固定绑定旧正式学生身份；本轮修复训练完成后需单独登记新模型，不能直接复用旧结果标签。
# Compute the four objective metrics and compare the student with paper Table 1.

set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 独立的客观指标 Python 环境，用于 UTMOS、DNSMOS、CER 和 SVA。
EVAL_PYTHON="${VOICEGRAD_EVAL_PYTHON:-/root/miniconda3/envs/voicegrad-eval/bin/python}"

# shellcheck source=scripts/common/require_rented_server.sh
# 将租用服务器身份检查函数加载到当前 shell；此处尚未启动训练。
source "${SCRIPT_DIR}/require_rented_server.sh"
# 核对登记主机、machine-id 摘要及项目路径，任何不符都拒绝运行。
require_rented_server "${PROJECT_ROOT}"

# 此处要求 Python 可执行；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ ! -x "${EVAL_PYTHON}" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "Objective-evaluation Python is not executable: ${EVAL_PYTHON}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi

# 所选声码器权重及配置所在目录。
VOCODER_DIR="${PROJECT_ROOT}/exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
# 本次要评估的推理结果目录，必须与生成 WAV 的实验版本一致。
INFERENCE_DIR="${PROJECT_ROOT}/exp/evaluation/fastvoicegrad_text_unseen_epoch0100_custom_hifigan"
# 逐句客观指标和最终汇总的存放目录。
METRICS_DIR="${INFERENCE_DIR}/metrics"

# 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
mkdir -p "${PROJECT_ROOT}/exp/locks"
# 用专用文件描述符打开这项任务的锁文件；持有期间可防止并发覆盖同一实验。
exec 9>"${PROJECT_ROOT}/exp/locks/text_unseen_custom_hifigan_pipeline.lock"
# 尝试立即取得对应任务的排他锁；已有任务持锁时失败退出，不并发启动冲突任务。
flock -n 9 || { echo "Custom HiFi-GAN pipeline is active." >&2; exit 3; }
# 用专用文件描述符打开这项任务的锁文件；持有期间可防止并发覆盖同一实验。
exec 8>"${PROJECT_ROOT}/exp/locks/text_unseen_custom_hifigan_train.lock"
# 尝试立即取得对应任务的排他锁；已有任务持锁时失败退出，不并发启动冲突任务。
flock -n 8 || { echo "Custom HiFi-GAN training is active." >&2; exit 3; }
# 用专用文件描述符打开这项任务的锁文件；持有期间可防止并发覆盖同一实验。
exec 7>"${PROJECT_ROOT}/exp/locks/text_unseen_fastvoicegrad_student.lock"
# 尝试立即取得对应任务的排他锁；已有任务持锁时失败退出，不并发启动冲突任务。
flock -n 7 || { echo "FastVoiceGrad student training is active." >&2; exit 3; }
# 用专用文件描述符打开这项任务的锁文件；持有期间可防止并发覆盖同一实验。
exec 6>"${PROJECT_ROOT}/exp/locks/text_unseen_fastvoicegrad_student_inference.lock"
# 尝试立即取得对应任务的排他锁；已有任务持锁时失败退出，不并发启动冲突任务。
flock -n 6 || { echo "FastVoiceGrad student inference/evaluation is active." >&2; exit 3; }

# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${PROJECT_ROOT}"

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：预先检查客观指标环境和模型依赖，避免生成全部语音后才发现评价不能运行；-u 让日志及时写出。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--vocoder-kind：标记自训 custom 或公开 public 来源。
# 命令第 4 行：--vocoder-checkpoint：指定声码器生成器权重。
# 命令第 5 行：--vocoder-config：指定声码器结构与声学参数。
"${EVAL_PYTHON}" -u -m evaluation.preflight_paper_metrics \
  --project-root "${PROJECT_ROOT}" \
  --vocoder-kind custom \
  --vocoder-checkpoint "${VOCODER_DIR}/generator_step_2500000.pt" \
  --vocoder-config "${VOCODER_DIR}/config.json"

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--inference-dir：指定指标对应的推理结果及配对清单。
# 命令第 4 行：--output-dir：指定本阶段的独立输出目录。
# 命令第 5 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好。
# 命令第 6 行：--batch-size：每批样本数，影响内存和更新统计。
# 命令第 7 行：--dnsmos-workers：DNSMOS 的 CPU 工作进程数。
# 命令第 8 行：--device：在租用服务器选择 GPU(cuda) 或 CPU。
"${EVAL_PYTHON}" -u -m evaluation.evaluate_metrics \
  --project-root "${PROJECT_ROOT}" \
  --inference-dir "${INFERENCE_DIR}" \
  --output-dir "${METRICS_DIR}" \
  --metric all \
  --batch-size 4 \
  --dnsmos-workers 3 \
  --device cuda

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：将学生四项客观指标与论文结果比较，不能混用教师系统标签；-u 让日志及时写出。
# 命令第 2 行：--summary：指定完整四指标汇总，避免读取半次评估。
# 命令第 3 行：--output-dir：指定本阶段的独立输出目录。
"${EVAL_PYTHON}" -u -m evaluation.compare_fastvoicegrad_with_paper \
  --summary "${METRICS_DIR}/objective_summary.json" \
  --output-dir "${METRICS_DIR}"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "FASTVOICEGRAD_FORMAL_OBJECTIVE_EVALUATION_COMPLETE output=${METRICS_DIR}"
