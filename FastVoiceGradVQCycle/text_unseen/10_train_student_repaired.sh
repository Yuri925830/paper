#!/usr/bin/env bash
# 使用 Bash；这是本次修复后的独立入口，旧学生训练目录继续保留。
# 命令失败、变量漏设或管道任一步失败，都结束脚本，避免带着错误开始训练。
set -euo pipefail
# 找到脚本自身所在文件夹，调用者在任何目录执行都能定位项目。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# text_unseen 的上一层就是项目根目录。
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# 导入租用服务器身份检查，source 本身不开始训练。
source "${PROJECT_ROOT}/scripts/common/require_rented_server.sh"
# 核对主机名、机器标识和项目路径，禁止误占本机或实验室服务器算力。
require_rented_server "${PROJECT_ROOT}"
# 切到根目录后，Python 才能按模块名找到 fastvoicegrad 包。
cd "${PROJECT_ROOT}"
# 使用已安装并核对过的训练环境；显式设置 VOICEGRAD_PYTHON 可覆盖路径。
PYTHON_BIN="${VOICEGRAD_PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
# 新目录独立保存本轮修复后的权重、损失和契约，不覆盖旧学生实验。
OUTPUT="${PROJECT_ROOT}/exp/checkpoints/fastvoicegrad_vctk_text_unseen_repair_x0_sqrt_20260906"
# Bash 数组将每个命令行参数分别保存，防止路径中的空格被错误拆开。
ARGS=(--project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
      # 教师目标直接还原到干净 mel；蒸馏权重采用累计信号幅度系数。
      --score-target clean_x0 --score-weight sqrt_alpha_bar
      # 每 50 次更新汇总一次损失；正式训练使用租用服务器 GPU。
      --loss-monitor-window 50 --device cuda)
# 未指定时从教师初始化新学生；只有明确选择 resume 才恢复本轮学生。
case "${STUDENT_RUN_MODE:-fresh}" in
  # fresh 不附加恢复参数，Python 仍会拒绝覆盖已有正式产物。
  fresh) ;;
  # resume 恢复新目录的权重、优化器与训练游标，不读取旧退化学生。
  resume) ARGS+=(--resume "${OUTPUT}/latest.pt") ;;
  # preflight 只检查输入和运行契约，不执行参数更新。
  preflight) ARGS+=(--preflight-only) ;;
  # 拼写错误立即报到标准错误并退出，避免悄悄选错实验模式。
  *) printf 'STUDENT_RUN_MODE must be fresh, resume, or preflight\n' >&2; exit 2 ;;
esac
# 限制 CPU 数学库并行线程数，避免小批计算因线程争用反而变慢。
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# 用训练进程替换当前 shell；-u 及时刷新日志，便于发现损失告警。
exec "${PYTHON_BIN}" -u -m fastvoicegrad.train "${ARGS[@]}"
