#!/usr/bin/env bash
# 对照版第 3 步：用 split_protocol.json 审计确认的 text-seen 清单训练 modified HiFi-GAN。
# 入口固定选择 hifigan/configs/text_seen.json，并写入带 text_seen 标签的检查点
# 目录；它与 text-unseen 声码器互不覆盖。
# 这是并行长期任务，不阻塞当前 04-public；自训练权重完成后才另跑 custom 结果。
set -euo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 按给定协议训练本项目修改版声码器；exec 让子任务继承退出码和信号。
exec bash "${ROOT}/scripts/common/03_train_custom_hifigan.sh" text-seen
