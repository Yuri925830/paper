#!/usr/bin/env bash
# 从原始 VCTK 开始建立完整音频库存并重采样到 22.05 kHz。
#
# 用法：
#   bash scripts/common/00_prepare_inventory.sh /path/to/VCTK-Corpus-0.92
#
# 环境变量：
#   VOICEGRAD_PROJECT_ROOT  仓库根目录；不设时自动根据本脚本位置计算。
#   VOICEGRAD_PYTHON        Python 命令；默认 python3。
#   RESAMPLE_WORKERS        ffmpeg 并行进程数；默认 6。
#
# 初学者提示：``$1`` 是第一个位置参数；``${名字:-默认值}`` 表示变量没设置时使用
# 冒号后的默认值。双引号能保护路径中的空格，实际脚本中不要随便删掉。

# -e：命令失败立刻退出；-u：使用未定义变量时报错；pipefail：管道任一步失败都算失败。
set -euo pipefail

# BASH_SOURCE[0] 是当前脚本路径；dirname 连用两次得到仓库根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 并行读取/处理进程数；教师固定为 4，库存重采样可由 RESAMPLE_WORKERS 指定。
WORKERS="${RESAMPLE_WORKERS:-6}"

# 原始 VCTK 很大，必须由用户明确给路径；缺少参数时打印用法并返回退出码 2。
if [[ "$#" -ne 1 ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "用法：$0 /path/to/VCTK-Corpus-0.92" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi
# 用户明确给出的原始 VCTK 数据目录。
VCTK_ROOT="$1"

# export 让接下来启动的 Python 子进程也能读取这个环境变量。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${PROJECT_ROOT}"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "[00/01] 扫描原始 VCTK：${VCTK_ROOT}"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：扫描官方原始库存，记录说话人、语音、文字与麦克风来源；-u 及时刷新日志。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--vctk-root：指定原始官方语料目录。
# 命令第 4 行：--audio-layout：只采用去静音后的 wav48_silence_trimmed 库存。
# 命令第 5 行：--microphone：只选 mic1 音频，避免混入不同麦克风。
"${PYTHON_BIN}" -u preprocess/00_build_vctk_manifest.py \
  --project-root "${PROJECT_ROOT}" \
  --vctk-root "${VCTK_ROOT}" \
  --audio-layout wav48_silence_trimmed \
  --microphone mic1

# 论文数据必须覆盖官方 mic1 的全部 110 人和 44,455 条音频。这里在四万条
# 重采样开始前就拒绝社区裁剪版、mic2 或残缺解压目录。
"${PYTHON_BIN}" - "${PROJECT_ROOT}/data/metadata/vctk_all_pairs.csv" <<'PY'
# CSV 按表头读取清单，避免依赖列的次序。
import csv
# sys.argv 读取 shell 传来的清单路径。
import sys

# 取得第一个命令行参数作为待核对的清单文件。
manifest = sys.argv[1]
# 以 UTF-8 打开该清单，CSV 模块自行处理行结束符。
with open(manifest, "r", encoding="utf-8", newline="") as handle:
    # 读取全部清单行，用来精确检查音频数量。
    rows = list(csv.DictReader(handle))
# 用集合去掉重复说话人编号，得到真实人数。
speakers = {row["speaker"] for row in rows}
# 官方完整 mic1 库存应为 44455 条、110 人，任一不同就拒绝继续重采样。
if len(rows) != 44455 or len(speakers) != 110:
    # 库存不完整时立即结束，避免在错误数据上继续消耗算力。
    raise SystemExit(
        # 错误信息先写明预期的语音数和人数。
        f"official VCTK 0.92 mic1 inventory expected 44455 rows/110 speakers; "
        # 再给出实际数量，便于区分下载残缺或筛选错误。
        f"got {len(rows)} rows/{len(speakers)} speakers"
    )
# 检查全部语音都来自官方去静音目录，不混用其他音频布局。
if {row["audio_layout"] for row in rows} != {"wav48_silence_trimmed"}:
    # 遇到混合布局直接报错，保证后续 mel 来源一致。
    raise SystemExit("inventory is not exclusively wav48_silence_trimmed")
# 检查所有清单行都采用第一支麦克风 mic1。
if {row["microphone"] for row in rows} != {"mic1"}:
    # 混入 mic2 或其他标签时拒绝继续处理。
    raise SystemExit("inventory is not exclusively mic1")
# 通过三类库存检查后才打印验证成功，后面再进行重采样。
print("official VCTK 0.92 mic1 inventory verified: 44455 rows, 110 speakers")
PY

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "[01/01] 使用 ffmpeg 重采样到 22.05 kHz"
# 使用 ffmpeg 将库存重采样到 22.05 kHz，给下游 mel 和声码器统一采样率；-u 及时刷新日志。
"${PYTHON_BIN}" -u preprocess/01_resample_to_22050.py --workers "${WORKERS}"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VCTK_INVENTORY_AND_RESAMPLE_COMPLETE"
