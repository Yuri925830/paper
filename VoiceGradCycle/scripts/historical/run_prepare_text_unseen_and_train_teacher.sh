#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。
# 历史兼容入口：保留 2026-07-20 任务记录中的文件名。
# 新读者应直接阅读 scripts/01_prepare_protocol_features.sh 与
# scripts/02_train_teacher.sh；这里仅按顺序调用它们，不再写死主机、环境或路径。
set -euo pipefail
# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 按给定协议建立清单和提取特征；等待该阶段成功后才执行下一条命令。
bash "${PROJECT_ROOT}/scripts/common/01_prepare_protocol_features.sh" text-unseen
# 按给定协议训练固定教师；等待该阶段成功后才执行下一条命令。
bash "${PROJECT_ROOT}/scripts/common/02_train_teacher.sh" text-unseen
