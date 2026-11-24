#!/usr/bin/env bash
# 上面的 shebang 指定由 Bash 执行；这是 text-unseen 目录中给用户点击/调用的短入口。
# 真正的严格参数与检查集中在公共脚本中，避免多个实验入口各自复制后悄悄不一致。
# 任一命令失败、变量未定义或管道中间失败时立即停止。
set -euo pipefail
# 取得这个短入口所在的绝对目录，例如 /root/FsatVoiceGrad/text_unseen。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 从 text_unseen 向上一级得到项目根目录 /root/FsatVoiceGrad。
ROOT="$(cd "${HERE}/.." && pwd)"
# exec 用公共训练脚本替换当前进程，退出码和信号都能原样传递给真正训练程序。
exec bash "${ROOT}/scripts/common/06_train_fastvoicegrad_student.sh"
