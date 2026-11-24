#!/usr/bin/env bash
# 可审计重建第 1 步：生成双轴隔离的 text-unseen 清单并提取 mel、BNF、speaker embedding。
# 协议名在最后一行写死，不能被命令行参数改成 text-seen；这正是独立入口的作用。
# HERE 是本文件夹，ROOT 是仓库根目录，双引号用于保护路径中可能出现的空格。
set -euo pipefail
# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 按给定协议建立清单和提取特征；exec 让子任务继承退出码和信号。
exec bash "${ROOT}/scripts/common/01_prepare_protocol_features.sh" text-unseen
