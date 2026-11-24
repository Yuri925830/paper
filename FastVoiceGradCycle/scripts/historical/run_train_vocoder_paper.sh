#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。
# 历史兼容入口。可把第一个参数写成 text-seen；省略时保持原来的 text-unseen。
# 详细训练解释见 scripts/common/03_train_custom_hifigan.sh 和 docs/HIFIGAN_CN.md。
set -euo pipefail
# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 活动数据协议；text-seen 与 text-unseen 必须对应各自的清单和统计量。
PROTOCOL="${1:-text-unseen}"
# 按给定协议训练本项目修改版声码器；exec 让子任务继承退出码和信号。
exec bash "${PROJECT_ROOT}/scripts/common/03_train_custom_hifigan.sh" "${PROTOCOL}"
