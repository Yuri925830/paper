#!/usr/bin/env bash
# 对照版第 2 步：只允许使用 text-seen 活动清单训练 VoiceGrad-30 教师。
# 底层还会核对 split_protocol.json、训练人数、提示句保留和所有特征形状。
# exec 让训练程序的成功/失败退出码直接成为本脚本退出码。
set -euo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 按给定协议训练固定教师；exec 让子任务继承退出码和信号。
exec bash "${ROOT}/scripts/common/02_train_teacher.sh" text-seen
