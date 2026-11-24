#!/usr/bin/env bash
# 上面的 shebang 指定使用服务器上的 Bash 解释器执行本脚本。
# 本入口从同一教师和定制声码器重新训练学生，并从第 71 轮起加入 Cycle 一致性目标。

set -euo pipefail # 任一命令失败、变量未定义或管道中途失败时立即停止，避免留下伪完成结果。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # 根据脚本自身位置计算稳定的公共脚本绝对目录。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}" # 优先采用外部指定项目根，否则从脚本位置向上两级推导。
PYTHON_BIN="${VOICEGRAD_PYTHON:-python3}" # 优先采用明确指定的实验 Python，未指定时才使用 python3。
RUN_MODE="${CYCLE_RUN_MODE:-fresh}" # 默认启动全新实验，也允许显式选择 resume 或 preflight。

case "${RUN_MODE}" in # 先验证运行模式，避免非法模式在 GPU 检查后才暴露。
  fresh|resume|preflight) ;; # 只允许全新训练、断点恢复或只读预检查三种模式。
  *) echo "CYCLE_RUN_MODE 只能是 fresh、resume 或 preflight，实际得到：${RUN_MODE}" >&2; exit 2 ;; # 非法模式立即停止。
esac # 完成运行模式的早期校验。

# 下一行加载统一的计算节点与项目路径保护逻辑，实际允许的 speech 主机由公共保护脚本集中管理。
source "${SCRIPT_DIR}/require_rented_server.sh" # 在当前 Bash 进程中载入启动前的服务器安全检查函数。
require_rented_server "${PROJECT_ROOT}" # 在接触模型文件前核对主机与项目根，禁止在登录节点或禁用 GPU 节点运行。

if [[ "${RUN_MODE}" != "preflight" ]]; then # 正式训练或恢复训练都必须显式指定并现场核对一张空闲 GPU。
  GPU_ID="${VOICEGRAD_GPU_ID:-}" # 读取人工根据全实验室资源比较后选定的物理 GPU 编号。
  if [[ ! "${GPU_ID}" =~ ^(0|[1-9][0-9]*)$ ]]; then # GPU 编号必须是无前导零的单个非负整数，禁止空值、列表或模糊设备名。
    echo "正式 Cycle 训练必须先设置单个物理 GPU：VOICEGRAD_GPU_ID=<编号>" >&2 # 提示先完成跨 speech 节点选择。
    exit 2 # 未明确选择设备时失败关闭，绝不默认占用逻辑 GPU 0。
  fi # 完成 GPU 编号格式检查。
  if ! command -v nvidia-smi >/dev/null 2>&1; then # speech GPU 节点必须能提供驱动侧占用信息。
    echo "找不到 nvidia-smi，无法证明所选 GPU 当前空闲" >&2 # 无法核验时不给训练放行。
    exit 2 # 缺少可靠现场检查工具时停止。
  fi # 完成 nvidia-smi 可用性检查。
  GPU_INDEX="$(nvidia-smi -i "${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits)" # 核对编号确实对应本机 GPU。
  GPU_NAME="$(nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader,nounits)" # 记录型号，供与启动前的跨节点资源盘点结果核对。
  GPU_UUID="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader,nounits)" # 记录不可与其他卡混淆的硬件身份。
  GPU_TOTAL_MIB="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.total --format=csv,noheader,nounits)" # 记录显存总量供运行契约审计。
  GPU_USED_MIB="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits)" # 启动前读取当前已用显存。
  GPU_UTIL_PERCENT="$(nvidia-smi -i "${GPU_ID}" --query-gpu=utilization.gpu --format=csv,noheader,nounits)" # 启动前读取当前计算利用率。
  for numeric_value in "${GPU_INDEX}" "${GPU_TOTAL_MIB}" "${GPU_USED_MIB}" "${GPU_UTIL_PERCENT}"; do # 四个数值字段都必须可安全用于整数比较。
    if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]]; then # MIG、驱动异常或意外输出不能被当成空闲状态。
      echo "nvidia-smi 返回了无法解析的 GPU 数值：${numeric_value}" >&2 # 显示具体异常值便于排查。
      exit 2 # 解析不确定时失败关闭。
    fi # 完成当前数值字段校验。
  done # 四个数值字段全部通过后再检查占用。
  if [[ "${GPU_INDEX}" != "${GPU_ID}" ]]; then # 查询返回的实际编号必须与人工选择的物理编号完全一致。
    echo "GPU 身份核对失败：请求 ${GPU_ID}，nvidia-smi 返回 ${GPU_INDEX}" >&2 # 明确报告编号错配，避免误用其他设备。
    exit 2 # 设备身份不一致时失败关闭。
  fi # 完成物理编号一致性检查。
  if [[ ! "${GPU_UUID}" =~ ^GPU-[[:alnum:]-]+$ ]]; then # UUID 必须符合 NVIDIA 单卡标识格式，禁止空值和设备列表。
    echo "nvidia-smi 返回了无法识别的 GPU UUID：${GPU_UUID}" >&2 # 显示异常身份值便于排查。
    exit 2 # 无法建立唯一设备身份时停止。
  fi # 完成硬件 UUID 格式检查。
  if ! GPU_COMPUTE_PIDS="$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then # 必须成功查询计算进程。
    echo "无法查询所选 GPU 的计算进程，拒绝在不确定状态下启动" >&2 # 查询失败不能静默当成无人使用。
    exit 2 # GPU 进程状态未知时停止。
  fi # 完成计算进程查询。
  if [[ -n "${GPU_COMPUTE_PIDS}" ]]; then # 任意已有计算进程都表示该卡并非空闲。
    echo "所选 GPU ${GPU_ID} 已有计算进程，拒绝启动：${GPU_COMPUTE_PIDS}" >&2 # 只报告 PID，不读取其他用户参数。
    exit 2 # 防止与现有实验抢占同一张卡。
  fi # 完成计算进程空闲检查。
  if (( GPU_USED_MIB > 256 || GPU_UTIL_PERCENT > 5 )); then # 无进程时仍要求显存和利用率接近空闲基线。
    echo "所选 GPU ${GPU_ID} 当前不够空闲：used=${GPU_USED_MIB} MiB, util=${GPU_UTIL_PERCENT}%" >&2 # 报告现场占用证据。
    exit 2 # 设备仍繁忙时停止，稍后重新选择。
  fi # 完成显存与利用率阈值检查。
  export CUDA_DEVICE_ORDER="PCI_BUS_ID" # 保留稳定设备排序作为兼容设置，真正的单卡选择由下行 UUID 完成。
  export CUDA_VISIBLE_DEVICES="${GPU_UUID}" # 用已核验的 UUID 暴露唯一目标卡，避免 CUDA 序号与 nvidia-smi 编号错配。
  export VOICEGRAD_SELECTED_GPU_PHYSICAL_ID="${GPU_ID}" # 把物理编号交给 Python 守卫和本次启动审计日志。
  export VOICEGRAD_SELECTED_GPU_NAME="${GPU_NAME}" # 把具体 GPU 型号交给 Python 守卫和本次启动审计日志。
  export VOICEGRAD_SELECTED_GPU_UUID="${GPU_UUID}" # 把硬件 UUID 交给 Python 守卫并核对实际可见设备。
  export VOICEGRAD_SELECTED_GPU_TOTAL_MIB="${GPU_TOTAL_MIB}" # 把总显存交给 Python 守卫和本次启动审计日志。
  export VOICEGRAD_SELECTED_GPU_USED_MIB="${GPU_USED_MIB}" # 把检查时已用显存交给 Python 再次核对并记录。
  export VOICEGRAD_SELECTED_GPU_UTIL_PERCENT="${GPU_UTIL_PERCENT}" # 把检查时利用率交给 Python 再次核对并记录。
  export VOICEGRAD_GPU_IDLE_ATTESTATION="no_compute_process_used_le_256MiB_util_le_5pct" # 标记已经通过本脚本的三重空闲检查。
  export VOICEGRAD_GPU_CHECK_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)" # 记录空闲检查发生的 UTC 时间。
fi # preflight 只用 CPU；正式模式至此才拥有可审计的单卡选择。

TEACHER="${PROJECT_ROOT}/exp/checkpoints/voicegrad30_vctk_text_unseen/epoch_0500.pt" # 固定使用已复现并审计过的第 500 轮 VoiceGrad 教师。
VOCODER_DIR="${PROJECT_ROOT}/exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen" # 定制 HiFi-GAN 的权重、配置和证明统一放在此目录。
VOCODER="${VOCODER_DIR}/generator_step_2500000.pt" # 固定使用第 2,500,000 步定制声码器生成器。
VOCODER_CONFIG="${VOCODER_DIR}/config.json" # 使用与定制声码器权重完全配套的网络和声学配置。
VOCODER_ATTESTATION="${VOCODER_DIR}/final_checkpoint_attestation_comments_20260906.json" # 使用现有证明文件核对声码器权重与代码身份。
OUTPUT="${PROJECT_ROOT}/exp/checkpoints/fastvoicegrad_vctk_text_unseen_cycle_v1" # Cycle 实验使用独立目录，绝不覆盖已复现基线产物。

for required in "${TEACHER}" "${VOCODER}" "${VOCODER_CONFIG}" "${VOCODER_ATTESTATION}"; do # 逐一检查四个固定依赖，缺少任意一个都不能启动。
  if [[ ! -f "${required}" ]]; then # 依赖路径必须存在且必须是普通文件，目录或失效链接都不接受。
    echo "Cycle 训练缺少固定依赖：${required}" >&2 # 把缺失的准确路径写到标准错误，方便直接定位问题。
    exit 2 # 用非零状态结束，防止调度脚本误认为实验已经正常开始。
  fi # 结束当前依赖的存在性判断。
done # 四个依赖全部检查完毕后才继续组装训练参数。

export VOICEGRAD_PROJECT_ROOT="${PROJECT_ROOT}" # 把同一个权威项目根传给训练器及其所有子模块。
cd "${PROJECT_ROOT}" # 切到项目根，保证模块导入、指纹检查和相对路径解释一致。

COMMON_ARGS=( # 用数组保存三种运行模式共享的参数，避免含空格路径被错误拆分。
  --project-root "${PROJECT_ROOT}" # 明确告诉训练器权威项目根目录。
  --teacher-checkpoint "${TEACHER}" # 指定第 500 轮教师检查点作为学生蒸馏来源。
  --vocoder-checkpoint "${VOCODER}" # 指定第 2,500,000 步定制声码器权重。
  --vocoder-config "${VOCODER_CONFIG}" # 指定与声码器权重配套的配置文件。
  --vocoder-attestation "${VOCODER_ATTESTATION}" # 指定声码器完整性证明文件。
  --output-dir "${OUTPUT}" # 把所有 Cycle 检查点、日志和曲线写入独立实验目录。
  --score-target clean_x0 # 沿用已复现基线的干净 mel 分数蒸馏目标。
  --score-weight sqrt_alpha_bar # 沿用已复现基线的累计信号幅度蒸馏权重定义。
  --enable-cycle-loss # 显式开启 A→B→A mel 往返一致性损失，避免依赖隐式默认值。
  --cycle-start-epoch 71 # 前 70 轮保持原基线目标，从第 71 轮开始加入 Cycle 项。
  --lambda-cycle 1.0 # 按教授指定阶段先使用论文默认量级 1.0 计权 Cycle 损失。
  --cycle-examples-per-batch 3 # 每批只取前三个源样本构造往返路径，与参考实现的训练预算一致。
) # 结束三种运行模式共享参数数组。

case "${RUN_MODE}" in # 根据明确的模式只执行全新训练、断点恢复或预检查中的一个分支。
  fresh) # fresh 从同一教师初始化新学生，并写入全新的 Cycle 输出目录。
    exec "${PYTHON_BIN}" -u -m fastvoicegrad.train "${COMMON_ARGS[@]}" --device cuda # 用当前 Bash 进程启动正式 GPU 训练并让日志立即刷新。
    ;; # 结束 fresh 分支，禁止继续落入其他模式。
  resume) # resume 只恢复本 Cycle 实验自己的完整 latest.pt 状态。
    exec "${PYTHON_BIN}" -u -m fastvoicegrad.train "${COMMON_ARGS[@]}" --resume "${OUTPUT}/latest.pt" --device cuda # 恢复模型、优化器、轮次和随机状态后继续 GPU 训练。
    ;; # 结束 resume 分支，禁止继续落入其他模式。
  preflight) # preflight 只核对依赖、数据和实验契约，不执行任何训练更新。
    exec "${PYTHON_BIN}" -u -m fastvoicegrad.train "${COMMON_ARGS[@]}" --preflight-only --device cpu # 在合规 speech 节点用 CPU 完成只读预检查。
    ;; # 结束 preflight 分支，禁止继续落入其他模式。
  *) # 上方已经拒绝非法模式；此分支保留为防御性兜底。
    echo "内部运行模式校验不一致：${RUN_MODE}" >&2 # 理论上不可到达，仍给出明确诊断。
    exit 2 # 防御性停止，绝不猜测运行方式。
    ;; # 结束不可达兜底分支。
esac # 完成唯一运行模式的选择。
