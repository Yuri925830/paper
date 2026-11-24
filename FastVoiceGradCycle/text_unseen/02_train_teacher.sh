#!/usr/bin/env bash
# 可审计重建第 2 步：只允许使用 text-unseen 活动清单训练 VoiceGrad-30 教师到 500 epoch。
# 底层脚本开跑前还会读取 split_protocol.json 再验证一次，防止目录名正确但活动
# 清单仍是另一版。exec 会把退出码和 Ctrl+C 原样传给真正训练程序。
set -euo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 按给定协议训练固定教师；exec 让子任务继承退出码和信号。
exec bash "${ROOT}/scripts/common/02_train_teacher.sh" text-unseen
