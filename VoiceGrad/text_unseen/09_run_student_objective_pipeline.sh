#!/usr/bin/env bash
# Resume-safe formal pipeline: one-step inference, four objective metrics, paper comparison.
# 启用失败即停、未定义变量报错和管道错误检查；带 -E 时让函数继承 ERR 处理。
set -Eeuo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "FASTVOICEGRAD_PIPELINE_STAGE_BEGIN stage=inference utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# 执行学生正式配对推理；等待该阶段成功后才执行下一条命令。
bash "${HERE}/07_infer_student_8100.sh"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "FASTVOICEGRAD_PIPELINE_STAGE_COMPLETE stage=inference utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "FASTVOICEGRAD_PIPELINE_STAGE_BEGIN stage=objective_evaluation utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# 执行同批配对的客观评估；等待该阶段成功后才执行下一条命令。
bash "${HERE}/08_evaluate_student.sh"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "FASTVOICEGRAD_PIPELINE_STAGE_COMPLETE stage=objective_evaluation utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "FASTVOICEGRAD_STUDENT_TO_PAPER_PIPELINE_COMPLETE"
