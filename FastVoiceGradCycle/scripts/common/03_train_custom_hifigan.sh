#!/usr/bin/env bash
# 训练本项目自己的 modified HiFi-GAN：V1 Generator + MPD + UnivNet MRD。
#
# 用法：
#   bash scripts/common/03_train_custom_hifigan.sh text-unseen
#   bash scripts/common/03_train_custom_hifigan.sh text-seen
#
# FastVoiceGrad 没公开独立声码器训练时长；正式 2,500,000 steps 继承官方
# HiFi-GAN 的 VCTK 实验，1,000,000 steps 只保留为中间归档。

set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 活动数据协议；text-seen 与 text-unseen 必须对应各自的清单和统计量。
PROTOCOL="${1:-text-unseen}"
# 此次启动模式，fresh 为新建训练，resume 为恢复本实验断点。
RUN_MODE="${VOCODER_RUN_MODE:-fresh}"

# shellcheck source=scripts/common/require_rented_server.sh
# 将租用服务器身份检查函数加载到当前 shell；此处尚未启动训练。
source "${SCRIPT_DIR}/require_rented_server.sh"
# 核对登记主机、machine-id 摘要及项目路径，任何不符都拒绝运行。
require_rented_server "${PROJECT_ROOT}"

# 按 "${PROTOCOL}" 选择唯一分支；未列出的取值由兜底分支报错。
case "${PROTOCOL}" in
  # 训练与测试同时隔离说话人及指定文本的协议。
  text-unseen)
    # 与当前 seen/unseen 数据协议对应的声码器训练配置。
    CONFIG="${PROJECT_ROOT}/hifigan/configs/text_unseen.json"
    ;;
  # 测试说话人仍隔离、公共文本在训练说话人中保留的对照协议。
  text-seen)
    # 与当前 seen/unseen 数据协议对应的声码器训练配置。
    CONFIG="${PROJECT_ROOT}/hifigan/configs/text_seen.json"
    ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *)
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "无效协议：${PROTOCOL}" >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 2
    ;;
esac

# 将协议名中的连字符换为下划线，仅用于统一目录命名，两种写法在文件路径中都合法。
PROTOCOL_TAG="${PROTOCOL//-/_}"
# 本协议的独立检查点目录，避免不同实验互相覆盖。
CHECKPOINT_DIR="${PROJECT_ROOT}/exp/checkpoints/hifigan_v1_mrd_vctk_${PROTOCOL_TAG}"
# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${PROJECT_ROOT}"

# 此处检查值非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -n "${VOCODER_MAX_STEPS:-}" || -n "${VOCODER_BATCH_SIZE:-}" || -n "${VOCODER_NUM_WORKERS:-}" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "正式 wrapper 禁止覆盖 step/batch/worker；冒烟必须直接使用 --diagnostic-smoke 和 exp/smoke。" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi

# 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
mkdir -p "${PROJECT_ROOT}/exp/locks"
# 用专用文件描述符打开这项任务的锁文件；持有期间可防止并发覆盖同一实验。
exec 9>"${PROJECT_ROOT}/exp/locks/${PROTOCOL_TAG}_custom_hifigan_train.lock"
# 尝试立即取得对应任务的排他锁；已有任务持锁时失败退出，不并发启动冲突任务。
if ! flock -n 9; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "同一协议的 custom HiFi-GAN 已有训练进程持有锁。" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 3
fi

# 声码器新建/续训参数数组，确保路径不被空格拆开。
MODE_ARGS=()
# 按 "${RUN_MODE}" 选择唯一分支；未列出的取值由兜底分支报错。
case "${RUN_MODE}" in
  # 新建实验分支，必须保护已有目录。
  fresh)
    # 追加明确的新建参数，训练器会拒绝覆盖已有实验。
    MODE_ARGS+=(--fresh)
    ;;
  # 续训分支，要求本实验完整断点。
  resume)
    # 追加当前实验 latest.pt 的恢复参数，继续其完整优化器与随机状态。
    MODE_ARGS+=(--resume "${CHECKPOINT_DIR}/latest.pt")
    ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *)
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "VOCODER_RUN_MODE 只能是 fresh 或 resume，实际得到：${RUN_MODE}" >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 2
    ;;
esac

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：训练本项目修改版 HiFi-GAN，或按 preflight 参数只核对输入；-u 让日志及时写出。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--config：指定与所选模型配套的完整配置。
# 命令第 4 行：--checkpoint-dir：指定本实验检查点目录。
# 命令第 5 行：--full-file-check：逐文件核对特征来源与完整性。
# 命令第 6 行：按数组原有参数边界展开已审定的新建/恢复选项；空数组不会额外传入无效参数。
"${PYTHON_BIN}" -u -m hifigan.custom.train \
  --project-root "${PROJECT_ROOT}" \
  --config "${CONFIG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --full-file-check \
  "${MODE_ARGS[@]}"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "CUSTOM_HIFIGAN_TRAINING_COMPLETE protocol=${PROTOCOL}"
