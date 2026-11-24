#!/usr/bin/env bash
# 历史入口仅保留以追溯旧实验；当前项目只允许在 matpool-rental 租用服务器计算，禁止 speech101–104。
# 历史兼容入口：用公开 VCTK HiFi-GAN 评估 text-unseen 教师 epoch 1000。
# 真正逻辑集中在通用脚本，避免 500/1000 两份代码以后改得不一致。
set -euo pipefail
# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：仅对紧接的子命令指定 public 声码器，保留历史公开权重对照。
# 命令第 2 行：生成指定教师/声码器版本的 8100 对转换；等待该阶段成功后才执行下一条命令。
VOICEGRAD_VOCODER_KIND=public \
  bash "${PROJECT_ROOT}/scripts/common/04_infer_8100_pairs.sh" text-unseen 1000
