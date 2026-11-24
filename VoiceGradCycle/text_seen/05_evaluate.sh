#!/usr/bin/env bash
# 对照版第 5 步：评价 8,100 对；参数依次是 epoch 和 custom/public 声码器标签。
# 默认读取当前实际使用的 public 输出；若以后推理时显式用了 custom，这里也必须
# 传 custom。04 与 05 标签不一致时，程序会寻找另一目录，不能得到正确结果。
# 四项 CSV 完整后才写最终汇总，断线重跑会跳过已有 pair_id。
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
# 命令第 1 行：计算四项客观指标并与论文比较；exec 让子任务继承退出码和信号。
# 命令第 2 行：依次传入固定数据协议、教师 epoch 和声码器来源标签，生成与评价必须保持这三项一致。
exec bash "${ROOT}/scripts/common/05_evaluate_8100_pairs.sh" \
  text-seen "${EPOCH}" "${VOCODER_KIND}"
