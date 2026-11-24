#!/usr/bin/env bash
# 可审计重建第 3 步：用 split_protocol.json 审计确认的 text-unseen 清单训练 modified HiFi-GAN。
# 入口会自动选择 hifigan/configs/text_unseen.json，并把检查点放入带
# text_unseen 标签的独立目录，因此不会覆盖对照版声码器。
# 这是耗时很长的并行任务，不是当前 04-public 推理的前置条件；只有训练权重真正
# 完成以后，04/05 才显式使用 custom。
set -euo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 按给定协议训练本项目修改版声码器；exec 让子任务继承退出码和信号。
exec bash "${ROOT}/scripts/common/03_train_custom_hifigan.sh" text-unseen
