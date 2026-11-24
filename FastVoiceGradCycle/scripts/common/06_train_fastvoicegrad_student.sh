#!/usr/bin/env bash
# 上面的 shebang 指定用服务器上的 Bash 解释这个脚本。
# 这是 VCTK text-unseen 学生模型的正式入口；只有定制 HiFi-GAN 已训练到
# 2,500,000 step 并完成证明文件后，下面的检查才允许学生训练继续。

# -e：任一命令失败就停；-u：未定义变量就报错；pipefail：管道任一步失败都报错。
set -euo pipefail

# BASH_SOURCE[0] 是本脚本路径；dirname 取文件夹；cd+pwd 得到不受当前终端位置影响的绝对路径。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 优先采用外部明确传入的项目根目录；没有传入时，从本脚本向上两级得到 /root/FsatVoiceGrad。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 优先使用指定的 Conda Python；没有指定时才退回系统 python3。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 模式默认 fresh；也可明确设为 resume（断点续训）或 preflight（只检查、不训练）。
RUN_MODE="${STUDENT_RUN_MODE:-fresh}"

# 让 ShellCheck 知道下一行会加载哪个公共安全脚本。
# shellcheck source=scripts/common/require_rented_server.sh
# source 在当前 Bash 中加载“只准租用服务器运行”的保护函数。
source "${SCRIPT_DIR}/require_rented_server.sh"
# 立即核对主机和项目路径，防止误在本机或实验室服务器占用算力。
require_rented_server "${PROJECT_ROOT}"

# 固定教师为已经审计过的 VoiceGrad 第 500 epoch 权重。
TEACHER="${PROJECT_ROOT}/exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt"
# 定制声码器目录集中保存权重、配置和最终证明。
VOCODER_DIR="${PROJECT_ROOT}/exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
# 固定使用定制 HiFi-GAN 第 2,500,000 step 的生成器，不能换成公开预训练权重。
VOCODER="${VOCODER_DIR}/generator_step_2500000.pt"
# config.json 记录定制声码器结构和声学参数。
VOCODER_CONFIG="${VOCODER_DIR}/config.json"
# attestation 文件保存权重哈希和数据/代码指纹，用于证明所加载文件身份。
# 继续使用同一个250万步G；这份已有证明对应源码逐行注释后的指纹，无需改权重。
VOCODER_ATTESTATION="${VOCODER_DIR}/final_checkpoint_attestation_comments_20260906.json"
# 蒸馏修复使用独立目录，旧100轮权重、曲线和优化器状态全部保留。
OUTPUT="${PROJECT_ROOT}/exp/checkpoints/fastvoicegrad_vctk_text_unseen_distillation_clean_v1"

# 依次检查教师、声码器权重、声码器配置和证明文件，少一个都不开始训练。
for required in "${TEACHER}" "${VOCODER}" "${VOCODER_CONFIG}" "${VOCODER_ATTESTATION}"; do
  # -f 表示路径必须存在而且必须是普通文件。
  if [[ ! -f "${required}" ]]; then
    # 把具体缺失路径写到标准错误，便于在日志中直接定位。
    echo "严格学生训练缺少固定依赖：${required}" >&2
    # 状态码 2 表示调用条件不满足，而不是继续使用替代文件。
    exit 2
  # 结束当前文件存在性判断。
  fi
# 结束四个固定依赖的循环检查。
done

# 把已经解析好的绝对项目路径传给 Python 及其子模块。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切到项目根目录，保证相对导入、Git 检查和输出位置一致。
cd "${PROJECT_ROOT}"
# 训练锁由 Python 持有，从“声码器流水线”一直锁到“学生训练”；直接调用 Python 模块也受保护。

# COMMON_ARGS 是三种模式共享的命令行参数数组；数组可避免含空格路径被错误拆分。
COMMON_ARGS=(
  # 告诉 Python 权威项目根目录在哪里。
  --project-root "${PROJECT_ROOT}"
  # 指定第 500 epoch 教师权重。
  --teacher-checkpoint "${TEACHER}"
  # 指定第 2,500,000 step 定制声码器权重。
  --vocoder-checkpoint "${VOCODER}"
  # 指定与声码器权重配套的结构配置。
  --vocoder-config "${VOCODER_CONFIG}"
  # 指定声码器完整性证明文件。
  --vocoder-attestation "${VOCODER_ATTESTATION}"
  # 指定所有学生训练产物的输出目录。
  --output-dir "${OUTPUT}"
  # 教师从t直接估计干净mel，避免相邻含噪状态监督干净学生输出。
  --score-target clean_x0
  # 同作者2025回顾公式(6)使用累计信号幅度；外层蒸馏权重45仍保持不变。
  --score-weight sqrt_alpha_bar
# 结束共享参数数组。
)

# 根据 RUN_MODE 选择“全新训练、断点恢复、只做检查”中的唯一一个分支。
case "${RUN_MODE}" in
  # fresh 从教师权重初始化学生并从 epoch 1 开始。
  fresh)
    # exec 用 Python 替换当前 Bash；-u 让日志立即刷新；cuda 强制只在租用 A100 上训练。
    exec "${PYTHON_BIN}" -u -m fastvoicegrad.train "${COMMON_ARGS[@]}" --device cuda
    # 双分号结束 fresh 分支。
    ;;
  # resume 从 OUTPUT/latest.pt 恢复模型、优化器、epoch 和随机状态。
  resume)
    # 下面是一条完整命令：只允许恢复本次正式目录的 latest.pt，并继续使用租用 A100。
    # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
    # 命令第 1 行：启动学生训练，模型身份、冻结状态和数据契约仍由 Python 严格校验；-u 让日志及时写出。
    # 命令第 2 行：--resume：恢复指定权重及优化器/RNG状态；--device：在租用服务器选择 GPU(cuda) 或 CPU。
    exec "${PYTHON_BIN}" -u -m fastvoicegrad.train "${COMMON_ARGS[@]}" \
      --resume "${OUTPUT}/latest.pt" --device cuda
    # 双分号结束 resume 分支。
    ;;
  # preflight 会加载并核对全部依赖、数据和指纹，但不会执行任何训练 step。
  preflight)
    # --preflight-only 让 Python 核对后退出；cpu 避免只做审计时占用 A100 显存。
    # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
    # 命令第 1 行：启动学生训练，模型身份、冻结状态和数据契约仍由 Python 严格校验；-u 让日志及时写出。
    # 命令第 2 行：--preflight-only：只审计不更新模型；--device：在租用服务器选择 GPU(cuda) 或 CPU。
    exec "${PYTHON_BIN}" -u -m fastvoicegrad.train "${COMMON_ARGS[@]}" \
      --preflight-only --device cpu
    # 双分号结束 preflight 分支。
    ;;
  # 星号匹配 fresh/resume/preflight 以外的所有拼写。
  *)
    # 输出实际收到的错误模式，防止用户误以为已经开始训练。
    echo "STUDENT_RUN_MODE 只能是 fresh、resume 或 preflight，实际得到：${RUN_MODE}" >&2
    # 用非零状态码结束。
    exit 2
    # 双分号结束兜底分支。
    ;;
# esac 结束模式选择。
esac
