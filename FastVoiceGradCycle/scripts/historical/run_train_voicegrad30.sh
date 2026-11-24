#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。
# 历史兼容入口。第一个参数决定 text-unseen 或 text-seen，默认前者。
# ``exec`` 用目标脚本替换当前 shell 进程，使退出码和 Ctrl+C 都能原样传递。
set -euo pipefail
# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 活动数据协议；text-seen 与 text-unseen 必须对应各自的清单和统计量。
PROTOCOL="${1:-text-unseen}"
# 按给定协议训练固定教师；exec 让子任务继承退出码和信号。
exec bash "${PROJECT_ROOT}/scripts/common/02_train_teacher.sh" "${PROTOCOL}"
