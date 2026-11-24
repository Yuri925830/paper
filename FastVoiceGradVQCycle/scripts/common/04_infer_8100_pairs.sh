#!/usr/bin/env bash
# 用指定协议和教师 epoch 生成论文公开设定下可审计重建的 8,100 对转换波形。
#
# 用法：
#   bash scripts/common/04_infer_8100_pairs.sh text-unseen 500 public
#   bash scripts/common/04_infer_8100_pairs.sh text-seen 500 public
#
# 当前正式默认 custom；public 只保留为已明确标记的历史对照。

set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 活动数据协议；text-seen 与 text-unseen 必须对应各自的清单和统计量。
PROTOCOL="${1:-text-unseen}"
# 要使用的教师训练轮数，用来定位对应检查点。
EPOCH="${2:-500}"
# 声码器来源标签；public 为公开预训练，custom 为本项目自训版本。
VOCODER_KIND="${3:-${VOICEGRAD_VOCODER_KIND:-custom}}"

# shellcheck source=scripts/common/require_rented_server.sh
# 将租用服务器身份检查函数加载到当前 shell；此处尚未启动训练。
source "${SCRIPT_DIR}/require_rented_server.sh"
# 核对登记主机、machine-id 摘要及项目路径，任何不符都拒绝运行。
require_rented_server "${PROJECT_ROOT}"

# 按 "${PROTOCOL}" 选择唯一分支；未列出的取值由兜底分支报错。
case "${PROTOCOL}" in
  # 只接受列出的合法模式；其他值交给后面的错误分支。
  text-unseen|text-seen) ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *) echo "无效协议：${PROTOCOL}" >&2; exit 2 ;;
esac
# 此处检查是否小于或等于边界；用正则确认 epoch 只含十进制数字；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ ! "${EPOCH}" =~ ^[0-9]+$ ]] || [[ "${EPOCH}" -le 0 ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "epoch 必须是正整数，实际得到：${EPOCH}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi

# 将协议名中的连字符换为下划线，仅用于统一目录命名，两种写法在文件路径中都合法。
PROTOCOL_TAG="${PROTOCOL//-/_}"
# 将教师 epoch 格式化为四位数字写入 EPOCH_PADDED，匹配检查点文件名。
printf -v EPOCH_PADDED "%04d" "${EPOCH}"
# 本次实际选用的教师权重路径，必须与评估标签中的 epoch 对应。
TEACHER_CHECKPOINT="${PROJECT_ROOT}/exp/checkpoints/voicegrad30_vctk_${PROTOCOL_TAG}/epoch_${EPOCH_PADDED}.pt"

# 按 "${VOCODER_KIND}" 选择唯一分支；未列出的取值由兜底分支报错。
case "${VOCODER_KIND}" in
  # 本项目自训声码器分支。
  custom)
    # 所选声码器权重及配置所在目录。
    VOCODER_DIR="${PROJECT_ROOT}/exp/checkpoints/hifigan_v1_mrd_vctk_${PROTOCOL_TAG}"
    # 此处检查值非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    if [[ -n "${VOICEGRAD_VOCODER_CHECKPOINT:-}" || -n "${VOICEGRAD_VOCODER_CONFIG:-}" || -n "${INFERENCE_BATCH_SIZE:-}" ]]; then
      # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
      echo "正式 custom 推理禁止覆盖 checkpoint/config/batch size，必须使用已验证的 2,500,000-step 归档和固定 batch=4。" >&2
      # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
      exit 2
    fi
    # 本次实际声码器生成器文件；历史与当前入口的步数不同，以下赋值原样保留。
    VOCODER_CHECKPOINT="${VOCODER_DIR}/generator_step_2500000.pt"
    # 与生成器配套的结构和音频配置。
    VOCODER_CONFIG="${VOCODER_DIR}/config.json"
    # 本次任务的输出目录；名称区分协议、模型和声码器来源。
    OUTPUT_DIR="${PROJECT_ROOT}/exp/evaluation/voicegrad30_${PROTOCOL_TAG}_epoch${EPOCH_PADDED}_custom_hifigan"
    ;;
  # 公开 VCTK V1 声码器历史对照分支。
  public)
    # 所选声码器权重及配置所在目录。
    VOCODER_DIR="${PROJECT_ROOT}/pretrained_hifigan_22k/VCTK_V1"
    # 本次实际声码器生成器文件；历史与当前入口的步数不同，以下赋值原样保留。
    VOCODER_CHECKPOINT="${VOICEGRAD_VOCODER_CHECKPOINT:-${VOCODER_DIR}/generator_v1}"
    # 与生成器配套的结构和音频配置。
    VOCODER_CONFIG="${VOICEGRAD_VOCODER_CONFIG:-${VOCODER_DIR}/config.json}"
    # 本次任务的输出目录；名称区分协议、模型和声码器来源。
    OUTPUT_DIR="${PROJECT_ROOT}/exp/evaluation/voicegrad30_${PROTOCOL_TAG}_epoch${EPOCH_PADDED}_public_hifigan"
    ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *)
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "声码器参数只能是 public 或 custom，实际得到：${VOCODER_KIND}" >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 2
    ;;
esac

# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${PROJECT_ROOT}"

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：用指定教师和声码器生成受控配对语音，并保存输入/模型来源；-u 让日志及时写出。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--protocol：选择 seen/unseen 数据协议。
# 命令第 4 行：--epoch：明确本次教师轮数标签。
# 命令第 5 行：--checkpoint：指定本次真正加载的权重文件。
# 命令第 6 行：--vocoder-checkpoint：指定声码器生成器权重。
# 命令第 7 行：--vocoder-config：指定声码器结构与声学参数。
# 命令第 8 行：--vocoder-kind：标记自训 custom 或公开 public 来源。
# 命令第 9 行：--output-dir：指定本阶段的独立输出目录。
# 命令第 10 行：--batch-size：每批样本数，影响内存和更新统计。
# 命令第 11 行：--device：在租用服务器选择 GPU(cuda) 或 CPU。
"${PYTHON_BIN}" -u -m voicegrad.infer_voicegrad30_paper_eval \
  --project-root "${PROJECT_ROOT}" \
  --protocol "${PROTOCOL}" \
  --epoch "${EPOCH}" \
  --checkpoint "${TEACHER_CHECKPOINT}" \
  --vocoder-checkpoint "${VOCODER_CHECKPOINT}" \
  --vocoder-config "${VOCODER_CONFIG}" \
  --vocoder-kind "${VOCODER_KIND}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "4" \
  --device "${VOICEGRAD_DEVICE:-cuda}"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_8100_PAIR_INFERENCE_COMPLETE output=${OUTPUT_DIR}"
