#!/usr/bin/env bash
# 对照版第 4 步：生成 8,100 对转换语音。第一个参数是教师 epoch，默认 500；
# 第二个参数是 public/custom，默认 public，与当前“先用公开 HiFi-GAN、等待自训练
# 声码器完成”的真实执行状态一致。测试人、句位和配对规则与 text-unseen 相同。
set -euo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 要使用的教师训练轮数，用来定位对应检查点。
EPOCH="${1:-500}"
# 声码器来源标签；public 为公开预训练，custom 为本项目自训版本。
VOCODER_KIND="${2:-public}"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：生成指定教师/声码器版本的 8100 对转换；exec 让子任务继承退出码和信号。
# 命令第 2 行：依次传入固定数据协议、教师 epoch 和声码器来源标签，生成与评价必须保持这三项一致。
exec bash "${ROOT}/scripts/common/04_infer_8100_pairs.sh" \
  text-seen "${EPOCH}" "${VOCODER_KIND}"
