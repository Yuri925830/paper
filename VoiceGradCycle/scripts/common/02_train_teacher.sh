#!/usr/bin/env bash
# 训练最新版 VoiceGrad-30 教师。两个协议使用同一模型代码，只改变活动数据清单。
#
# 用法：
#   bash scripts/common/02_train_teacher.sh text-unseen
#   bash scripts/common/02_train_teacher.sh text-seen
# 中断后显式续训（只接受当前协议目录内含完整 Adam/RNG 状态的 latest.pt）：
#   RESUME_CHECKPOINT=exp/checkpoints/voicegrad30_vctk_text_unseen/latest.pt \
#     bash scripts/common/02_train_teacher.sh text-unseen
#
# 固定重建设置：500 epoch、batch=32、DataLoader workers=4。
# 论文未给 worker 数；4 是为本次可复现运行固定的工程选择。

set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 活动数据协议；text-seen 与 text-unseen 必须对应各自的清单和统计量。
PROTOCOL="${1:-text-unseen}"
# 教师训练总轮数，正式入口固定为 500。
EPOCHS=500
# 每次更新的训练样本数，正式教师固定为 32。
BATCH_SIZE=32
# 并行读取/处理进程数；教师固定为 4，库存重采样可由 RESAMPLE_WORKERS 指定。
WORKERS=4

# 此处检查值非空；精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -n "${TEACHER_EPOCHS:-}" && "${TEACHER_EPOCHS}" != "500" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "论文复现入口固定为 500 epoch；拒绝 TEACHER_EPOCHS=${TEACHER_EPOCHS}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi
# 此处检查值非空；精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -n "${TEACHER_BATCH_SIZE:-}" && "${TEACHER_BATCH_SIZE}" != "32" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "论文教师 batch 固定为 32；拒绝 TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi
# 此处检查值非空；精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -n "${TEACHER_WORKERS:-}" && "${TEACHER_WORKERS}" != "4" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "本次可审计运行固定 DataLoader workers=4；拒绝 TEACHER_WORKERS=${TEACHER_WORKERS}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi

# 按 "${PROTOCOL}" 选择唯一分支；未列出的取值由兜底分支报错。
case "${PROTOCOL}" in
  # 只接受列出的合法模式；其他值交给后面的错误分支。
  text-unseen|text-seen) ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *) echo "无效协议：${PROTOCOL}" >&2; exit 2 ;;
esac

# 为了与已有检查点目录统一命名，把 text-unseen 变为 text_unseen；路径本身允许连字符。
PROTOCOL_TAG="${PROTOCOL//-/_}"
# 本次任务的输出目录；名称区分协议、模型和声码器来源。
OUTPUT_DIR="${PROJECT_ROOT}/exp/checkpoints/voicegrad30_vctk_${PROTOCOL_TAG}"
# 断点恢复参数数组，默认空数组表示没有请求续训。
RESUME_ARGS=()

# 此处检查值非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  # 相对路径统一相对项目根解析；不允许从别的协议/实验目录借 checkpoint。
  if [[ "${RESUME_CHECKPOINT}" = /* ]]; then
    # 将用户指定断点解析为候选绝对路径，随后仍需检查归属。
    RESUME_CANDIDATE="${RESUME_CHECKPOINT}"
  # 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
  else
    # 将用户指定断点解析为候选绝对路径，随后仍需检查归属。
    RESUME_CANDIDATE="${PROJECT_ROOT}/${RESUME_CHECKPOINT}"
  fi
  # 此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ ! -f "${RESUME_CANDIDATE}" ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "续训 checkpoint 不存在或不是普通文件：${RESUME_CANDIDATE}" >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 2
  fi

  # realpath -e 同时消除 ../ 和符号链接，防止路径文本看似位于 OUTPUT_DIR，
  # 实际却指向另一实验。严格恢复只允许该协议自己的 latest.pt。
  RESUME_PATH="$(realpath -e -- "${RESUME_CANDIDATE}")"
  # 当前协议输出目录的真实位置。
  RESOLVED_OUTPUT_DIR="$(realpath -e -- "${OUTPUT_DIR}" 2>/dev/null || true)"
  # 仅允许恢复的本实验 latest.pt 路径。
  EXPECTED_LATEST="${RESOLVED_OUTPUT_DIR}/latest.pt"
  # 此处检查值为空；精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ -z "${RESOLVED_OUTPUT_DIR}" || "${RESUME_PATH}" != "${EXPECTED_LATEST}" ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "严格续训只允许 ${OUTPUT_DIR}/latest.pt；收到 ${RESUME_PATH}" >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 2
  fi
  # 断点恢复参数数组，默认空数组表示没有请求续训。
  RESUME_ARGS=(--resume "${RESUME_PATH}")
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "[RESUME] 将从 ${RESUME_PATH} 严格恢复；训练终点仍固定为 epoch 500。"
# 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
else
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "[NEW RUN] 未设置 RESUME_CHECKPOINT，将启动全新 500-epoch 训练。"
fi

# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${PROJECT_ROOT}"

# 训练前再次验证，防止上次命令切换了协议但忘记重算统计量。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：逐项验证活动划分、人数、特征来源和统计量，检查失败就阻止后续任务；-u 让日志及时写出。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--expected-protocol：要求现场活动清单与此协议一致。
# 命令第 4 行：--full：执行全量数据核查。
"${PYTHON_BIN}" -u -m voicegrad.validate_setup \
  --project-root "${PROJECT_ROOT}" \
  --expected-protocol "${PROTOCOL}" \
  --full

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：启动教师训练，下面参数固定训练量、保存频率和断点来源；-u 让日志及时写出。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--output-dir：指定本阶段的独立输出目录。
# 命令第 4 行：--epochs：训练最终轮数；续训也以同一终点结束。
# 命令第 5 行：--batch-size：每批样本数，影响内存和更新统计。
# 命令第 6 行：--num-workers：训练数据加载工作进程数。
# 命令第 7 行：--expected-protocol：要求现场活动清单与此协议一致。
# 命令第 8 行：--save-every：按指定轮数额外归档检查点。
# 命令第 9 行：--loss-every：按指定更新步数打印损失。
# 命令第 10 行：按数组原有参数边界展开已审定的新建/恢复选项；空数组不会额外传入无效参数。
"${PYTHON_BIN}" -u -m voicegrad.train \
  --project-root "${PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${WORKERS}" \
  --expected-protocol "${PROTOCOL}" \
  --save-every 100 \
  --loss-every 500 \
  "${RESUME_ARGS[@]}"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_TEACHER_TRAINING_COMPLETE protocol=${PROTOCOL}"
