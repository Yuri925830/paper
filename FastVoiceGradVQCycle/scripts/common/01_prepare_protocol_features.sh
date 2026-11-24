#!/usr/bin/env bash
# 为 text-unseen 或 text-seen 生成活动清单，并提取教师模型需要的全部特征。
#
# 用法：
#   bash scripts/common/01_prepare_protocol_features.sh text-unseen
#   bash scripts/common/01_prepare_protocol_features.sh text-seen
#
# 运行前需要：
#   1. scripts/common/00_prepare_inventory.sh 已完成；
#   2. VOICEGRAD_LEGACY_BNF_ROOT 指向旧 BNF/GE2E 模型目录；
#   3. 当前 Python 3.8 环境已经安装 requirements-teacher-matpool.txt 中的固定依赖。

set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 活动数据协议；text-seen 与 text-unseen 必须对应各自的清单和统计量。
PROTOCOL="${1:-text-unseen}"
# mel 提取的并行工作进程数，控制 CPU 与磁盘负载。
MEL_WORKERS="${MEL_WORKERS:-4}"

# ``case`` 是 shell 的多分支判断；这里只接受两个精确名字，拼错就立即停止。
case "${PROTOCOL}" in
  # 只接受列出的合法模式；其他值交给后面的错误分支。
  text-unseen|text-seen) ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *)
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "协议只能是 text-unseen 或 text-seen，实际得到：${PROTOCOL}" >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 2
    ;;
esac

# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${PROJECT_ROOT}"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "[02] 建立 ${PROTOCOL} 训练/测试清单"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按指定协议建立活动训练/测试清单，明确文本与说话人的隔离；-u 及时刷新日志。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--protocol：选择 seen/unseen 数据协议。
"${PYTHON_BIN}" -u preprocess/02_build_fastvoicegrad_split.py \
  --project-root "${PROJECT_ROOT}" \
  --protocol "${PROTOCOL}"

# 03、05、06 都支持断点续跑：目标 .npy 已存在时跳过。切换协议不会删除旧特征，
# 因为同一 utt_id 的 mel/BNF/embedding 与它属于哪个 split 无关。
echo "[03] 提取 80 维 log-mel"
# mel 提取器还没有逐文件来源账本；严格入口因此每次完整重建，确保最终报告
# attest `ok=all, skip=0`，不会把来源未知但形状相同的旧数组重新背书。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：提取 80 维 log-mel，参数与教师/声码器接口一致；-u 及时刷新日志。
# 命令第 2 行：--num_workers：控制 mel 提取并行进程数。
# 命令第 3 行：--force：完整重建对应特征，旧未知来源缓存不会直接背书。
"${PYTHON_BIN}" -u preprocess/03_extract_mel_22050.py \
  --num_workers "${MEL_WORKERS}" \
  --force

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "[05] 提取 144 维 BNF"
# 用固定 Conformer 提取 144 维内容 BNF，并保留来源账本；-u 及时刷新日志。
"${PYTHON_BIN}" -u preprocess/05_extract_bnf.py

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "[06] 提取 256 维 speaker embedding"
# 用固定 GE2E 编码器提取 256 维说话人嵌入并保留来源；-u 及时刷新日志。
"${PYTHON_BIN}" -u preprocess/06_extract_spk_emb.py

# mean/std 必须最后重算，因为 text-unseen 与 text-seen 的 train 行数不同。
echo "[04] 只用当前训练集重算 mel mean/std"
# 只用当前训练集重算逐频带均值和标准差，避免测试信息泄漏；-u 及时刷新日志。
"${PYTHON_BIN}" -u preprocess/04_compute_mel_stats.py

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "[检查] 核对人数、文本边界、特征形状和数据泄漏"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：逐项验证活动划分、人数、特征来源和统计量，检查失败就阻止后续任务；-u 让日志及时写出。
# 命令第 2 行：--project-root：明确权威项目根目录。
# 命令第 3 行：--expected-protocol：要求现场活动清单与此协议一致。
# 命令第 4 行：--full：执行全量数据核查。
"${PYTHON_BIN}" -u -m voicegrad.validate_setup \
  --project-root "${PROJECT_ROOT}" \
  --expected-protocol "${PROTOCOL}" \
  --full

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "PROTOCOL_FEATURES_COMPLETE protocol=${PROTOCOL}"
