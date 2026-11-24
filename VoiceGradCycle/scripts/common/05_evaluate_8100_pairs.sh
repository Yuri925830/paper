#!/usr/bin/env bash
# 对 8,100 条波形计算 UTMOS、DNSMOS、CER 和说话人相似度，并汇总 CSV/JSON。
#
# 用法：
#   bash scripts/common/05_evaluate_8100_pairs.sh text-unseen 500 public
# 第三个参数可以是 custom 或 public，必须与 04 推理脚本选用的声码器一致。
# 未填写时默认 custom；public 只用于历史对照目录。

set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 执行本阶段的 Python 环境；训练与客观评估需要使用各自配套依赖。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}"
# 活动数据协议；text-seen 与 text-unseen 必须对应各自的清单和统计量。
PROTOCOL="${1:-text-unseen}"
# 要使用的教师训练轮数，用来定位对应检查点。
EPOCH="${2:-500}"
# 声码器来源标签；public 为公开预训练，custom 为本项目自训版本。
VOCODER_KIND="${3:-custom}"

# shellcheck source=scripts/common/require_rented_server.sh
# 将租用服务器身份检查函数加载到当前 shell；此处尚未启动训练。
source "${SCRIPT_DIR}/require_rented_server.sh"
# 核对登记主机、machine-id 摘要及项目路径，任何不符都拒绝运行。
require_rented_server "${PROJECT_ROOT}"

# 按 "${PROTOCOL}" 选择唯一分支；未列出的取值由兜底分支报错。
case "${PROTOCOL}" in
  # 只接受列出的合法模式；其他值交给后面的错误分支。
  text-unseen|text-seen) ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *) echo "无效协议：${PROTOCOL}" >&2; exit 2 ;;
esac
# 按 "${VOCODER_KIND}" 选择唯一分支；未列出的取值由兜底分支报错。
case "${VOCODER_KIND}" in
  # 只接受列出的合法模式；其他值交给后面的错误分支。
  custom|public) ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *) echo "声码器标签只能是 custom 或 public。" >&2; exit 2 ;;
esac

# 将协议名中的连字符换为下划线，仅用于统一目录命名，两种写法在文件路径中都合法。
PROTOCOL_TAG="${PROTOCOL//-/_}"
# 将教师 epoch 格式化为四位数字写入 EPOCH_PADDED，匹配检查点文件名。
printf -v EPOCH_PADDED "%04d" "${EPOCH}"
# 本次要评估的推理结果目录，必须与生成 WAV 的实验版本一致。
INFERENCE_DIR="${PROJECT_ROOT}/exp/evaluation/voicegrad30_${PROTOCOL_TAG}_epoch${EPOCH_PADDED}_${VOCODER_KIND}_hifigan"
# 逐句客观指标和最终汇总的存放目录。
METRICS_DIR="${INFERENCE_DIR}/metrics"

# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${PROJECT_ROOT}"

# --metric all 会依次计算四项并写 objective_summary.csv/json；已经完整存在的 CSV
# 会被识别并复用，因此中途断线后可以直接重跑本脚本。
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取固定配对的生成语音，计算或复核指定客观指标，不进行主观听评；-u 让日志及时写出。
# 命令第 2 行：--inference-dir：指定指标对应的推理结果及配对清单。
# 命令第 3 行：--output-dir：指定本阶段的独立输出目录。
# 命令第 4 行：--metric：选择四项指标中的一项或 all；CER 越低越好，UTMOS/DNSMOS/SVA 越高越好。
# 命令第 5 行：--batch-size：每批样本数，影响内存和更新统计。
# 命令第 6 行：--dnsmos-workers：DNSMOS 的 CPU 工作进程数。
# 命令第 7 行：--device：在租用服务器选择 GPU(cuda) 或 CPU。
"${PYTHON_BIN}" -u -m evaluation.evaluate_metrics \
  --inference-dir "${INFERENCE_DIR}" \
  --output-dir "${METRICS_DIR}" \
  --metric all \
  --batch-size "${METRICS_BATCH_SIZE:-4}" \
  --dnsmos-workers "${DNSMOS_WORKERS:-3}" \
  --device "${VOICEGRAD_DEVICE:-cuda}"

# 四项客观指标的完整汇总文件，论文比较从这里读取。
SUMMARY_JSON="${METRICS_DIR}/objective_summary.json"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：读取教师指标汇总并与论文表格比较，保留评价实现差异的说明；-u 让日志及时写出。
# 命令第 2 行：--summary：指定完整四指标汇总，避免读取半次评估。
# 命令第 3 行：--output-dir：指定本阶段的独立输出目录。
"${PYTHON_BIN}" -u -m evaluation.compare_with_paper \
  --summary "${SUMMARY_JSON}" \
  --output-dir "${METRICS_DIR}"

# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_OBJECTIVE_EVALUATION_COMPLETE output=${METRICS_DIR}"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VOICEGRAD30_PAPER_COMPARISON_COMPLETE output=${METRICS_DIR}/paper_comparison.md"
