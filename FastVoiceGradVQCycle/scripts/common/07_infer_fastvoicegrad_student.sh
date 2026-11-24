#!/usr/bin/env bash
# 该入口固定绑定旧正式学生身份；本轮修复训练完成后需单独登记新模型，不能直接复用旧结果标签。
# Generate the formal 8,100 text-unseen conversions with the completed student.

set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-/root/miniconda3/envs/myconda/bin/python}"

# shellcheck source=scripts/common/require_rented_server.sh
# 将租用服务器身份检查函数加载到当前 shell；此处尚未启动训练。
source "${SCRIPT_DIR}/require_rented_server.sh"
# 核对登记主机、machine-id 摘要及项目路径，任何不符都拒绝运行。
require_rented_server "${PROJECT_ROOT}"

# 此处要求 Python 可执行；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ ! -x "${PYTHON_BIN}" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "FastVoiceGrad inference Python is not executable: ${PYTHON_BIN}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi

# 旧学生正式目录；新修复模型的评估需另建明确版本，不能自动套用旧权重身份。
STUDENT_DIR="${PROJECT_ROOT}/exp/checkpoints/fastvoicegrad_vctk_text_unseen"
# 所选声码器权重及配置所在目录。
VOCODER_DIR="${PROJECT_ROOT}/exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
# 本次任务的输出目录；名称区分协议、模型和声码器来源。
OUTPUT_DIR="${PROJECT_ROOT}/exp/evaluation/fastvoicegrad_text_unseen_epoch0100_custom_hifigan"

# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${PROJECT_ROOT}"

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：执行旧正式学生的单步推理；修复后的新模型必须另行登记身份；-u 让日志及时写出。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--checkpoint：指定本次真正加载的权重文件。
# 命令第 4 行：--run-contract：加载学生训练时登记的数据/模型/参数契约。
# 命令第 5 行：--vocoder-checkpoint：指定声码器生成器权重。
# 命令第 6 行：--vocoder-config：指定声码器结构与声学参数。
# 命令第 7 行：--vocoder-attestation：要求与声码器和现场代码匹配的验证证明。
# 命令第 8 行：--output-dir：指定本阶段的独立输出目录。
# 命令第 9 行：--batch-size：每批样本数，影响内存和更新统计。
# 命令第 10 行：--seed：固定随机噪声以便重复比较。
# 命令第 11 行：--device：在租用服务器选择 GPU(cuda) 或 CPU。
exec "${PYTHON_BIN}" -u -m fastvoicegrad.infer_paper_eval \
  --project-root "${PROJECT_ROOT}" \
  --checkpoint "${STUDENT_DIR}/epoch_0100.pt" \
  --run-contract "${STUDENT_DIR}/run_contract.json" \
  --vocoder-checkpoint "${VOCODER_DIR}/generator_step_2500000.pt" \
  --vocoder-config "${VOCODER_DIR}/config.json" \
  --vocoder-attestation "${VOCODER_DIR}/final_checkpoint_attestation.json" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 4 \
  --seed 1234 \
  --device cuda
