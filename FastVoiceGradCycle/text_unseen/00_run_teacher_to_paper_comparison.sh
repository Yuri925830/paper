#!/usr/bin/env bash
# 从 text-unseen 特征一直串行运行到 epoch-500 教师的论文指标比较。
# 任一步失败都会停止，绝不会把上游失败误当成“已完成”而继续评价。

set -euo pipefail

# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 教师训练和推理环境，保持与已训练权重的运行库一致。
TEACHER_PYTHON="${VOICEGRAD_TEACHER_PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
# 独立的客观指标 Python 环境，用于 UTMOS、DNSMOS、CER 和 SVA。
EVAL_PYTHON="${VOICEGRAD_EVAL_PYTHON:-/root/miniconda3/envs/voicegrad-eval/bin/python}"
# 声码器来源标签；public 为公开预训练，custom 为本项目自训版本。
VOCODER_KIND="${VOICEGRAD_VOCODER_KIND:-public}"

# shellcheck source=scripts/common/require_rented_server.sh
# 将租用服务器身份检查函数加载到当前 shell；此处尚未启动训练。
source "${ROOT}/scripts/common/require_rented_server.sh"
# 核对登记主机、machine-id 摘要及项目路径，任何不符都拒绝运行。
require_rented_server "${ROOT}"

# 此处要求 Python 可执行；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ ! -x "${TEACHER_PYTHON}" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "教师 Python 不可执行：${TEACHER_PYTHON}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi
# 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ "${VOCODER_KIND}" != "public" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "这条正式自动链当前只接受已固定哈希的 public VCTK HiFi-GAN。" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi

# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${ROOT}"
# 向子任务传递计算设备选择，当前项目只允许在租用服务器计算。
export VOICEGRAD_DEVICE="${VOICEGRAD_DEVICE:-cuda}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${ROOT}"

# 把阶段名、开始/结束时间和实际命令包装在一起，保留失败的准确位置。
run_stage() {
  # 当前阶段或分片名称，用于区分日志与输出。
  local name="$1"
  # 移除已经读取的阶段名，保留后面的真实命令与参数供本阶段执行。
  shift
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "PIPELINE_STAGE_BEGIN stage=${name} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # 保持参数原有边界执行本阶段命令；前面阶段名已经由 shift 移除。
  "$@"
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "PIPELINE_STAGE_COMPLETE stage=${name} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 prepare_features 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：只为这个子阶段设置 Python 或新建/恢复模式环境变量，不改变其他已运行任务的环境。
# 命令第 3 行：为固定协议准备全部特征；等待该阶段成功后才执行下一条命令。
run_stage prepare_features \
  env VOICEGRAD_PYTHON="${TEACHER_PYTHON}" \
  bash text_unseen/01_prepare_features.sh

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 train_teacher_epoch500 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：只为这个子阶段设置 Python 或新建/恢复模式环境变量，不改变其他已运行任务的环境。
# 命令第 3 行：按给定协议训练固定教师；等待该阶段成功后才执行下一条命令。
run_stage train_teacher_epoch500 \
  env VOICEGRAD_PYTHON="${TEACHER_PYTHON}" \
  bash text_unseen/02_train_teacher.sh

# 本次实际选用的教师权重路径，必须与评估标签中的 epoch 对应。
TEACHER_CHECKPOINT="${ROOT}/exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt"
# 此处要求文件存在且非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ ! -s "${TEACHER_CHECKPOINT}" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "训练命令返回成功，但 epoch-500 checkpoint 不存在：${TEACHER_CHECKPOINT}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 1
fi

# 此处要求 Python 可执行；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ ! -x "${EVAL_PYTHON}" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "评价 Python 不可执行：${EVAL_PYTHON}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 evaluation_preflight 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：预先检查客观指标环境和模型依赖，避免生成全部语音后才发现评价不能运行；-u 让日志及时写出。
# 命令第 3 行：--project-root：明确权威项目根目录；--vocoder-kind：标记自训 custom 或公开 public 来源。
run_stage evaluation_preflight \
  "${EVAL_PYTHON}" -u -m evaluation.preflight_paper_metrics \
  --project-root "${ROOT}" --vocoder-kind "${VOCODER_KIND}"

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 infer_voicegrad30_8100 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：只为这个子阶段设置 Python 或新建/恢复模式环境变量，不改变其他已运行任务的环境。
# 命令第 3 行：生成指定教师轮数与声码器来源的配对波形；等待该阶段成功后才执行下一条命令。
run_stage infer_voicegrad30_8100 \
  env VOICEGRAD_PYTHON="${TEACHER_PYTHON}" \
  bash text_unseen/04_infer_8100.sh 500 "${VOCODER_KIND}"

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 evaluate_objective_metrics 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：只为这个子阶段设置 Python 或新建/恢复模式环境变量，不改变其他已运行任务的环境。
# 命令第 3 行：评价前一步同版本生成的音频；等待该阶段成功后才执行下一条命令。
run_stage evaluate_objective_metrics \
  env VOICEGRAD_PYTHON="${EVAL_PYTHON}" \
  bash text_unseen/05_evaluate.sh 500 "${VOCODER_KIND}"

# 教师结果与论文数值的比较报告位置。
COMPARISON="${ROOT}/exp/evaluation/voicegrad30_text_unseen_epoch0500_public_hifigan/metrics/paper_comparison.md"
# 此处要求文件存在且非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ ! -s "${COMPARISON}" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "评价返回成功，但论文比较报告不存在：${COMPARISON}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 1
fi

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_TEACHER_TO_PAPER_COMPARISON_COMPLETE report=${COMPARISON}"
