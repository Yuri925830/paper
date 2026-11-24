#!/usr/bin/env bash
# Formal text-unseen inference for the completed FastVoiceGrad epoch-100 student.
# 启用失败即停、未定义变量报错和管道错误检查；带 -E 时让函数继承 ERR 处理。
set -euo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 调用旧正式学生推理入口，读取其固定模型身份；exec 让子任务继承退出码和信号。
exec bash "${ROOT}/scripts/common/07_infer_fastvoicegrad_student.sh"
