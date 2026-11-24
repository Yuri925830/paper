#!/usr/bin/env bash
# 对照版第 1 步：生成 text-seen 清单并提取/复用全部三类特征。
# text-seen 在训练人中保留公共句 001–010，但测试说话人仍然完全隔离。
# 末行协议名写死，使用这个目录时不需要自己记忆或拼写 --protocol。
set -euo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 按给定协议建立清单和提取特征；exec 让子任务继承退出码和信号。
exec bash "${ROOT}/scripts/common/01_prepare_protocol_features.sh" text-seen
