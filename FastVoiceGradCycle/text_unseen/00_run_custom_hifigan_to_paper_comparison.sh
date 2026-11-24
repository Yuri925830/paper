#!/usr/bin/env bash
# Train the paper-architecture-aligned custom vocoder, then run formal evaluation.

# 启用失败即停、未定义变量报错和管道错误检查；带 -E 时让函数继承 ERR 处理。
set -Eeuo pipefail

# 当前入口所在目录，调用时不依赖终端的位置。
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本脚本对应的项目根目录。
ROOT="$(cd "${HERE}/.." && pwd)"
# 公共脚本所在目录，各协议入口复用它们。
COMMON="${ROOT}/scripts/common"
# 训练/模型校验环境，不和较新的客观指标环境混装依赖。
TRAIN_PYTHON="${VOICEGRAD_TEACHER_PYTHON:-/root/miniconda3/envs/myconda/bin/python}"
# 独立的客观指标 Python 环境，用于 UTMOS、DNSMOS、CER 和 SVA。
EVAL_PYTHON="${VOICEGRAD_EVAL_PYTHON:-/root/miniconda3/envs/voicegrad-eval/bin/python}"
# 外部请求的声码器流水线模式；auto 会检查现有产物后决定。
REQUESTED_RUN_MODE="${VOICEGRAD_CUSTOM_PIPELINE_MODE:-auto}"
# 本协议的独立检查点目录，避免不同实验互相覆盖。
CHECKPOINT_DIR="${ROOT}/exp/checkpoints/hifigan_v1_mrd_vctk_text_unseen"
# 声码器训练最终 250 万步生成器文件。
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/generator_step_2500000.pt"
# 最终权重的校验报告路径；历史证明应单独保留，不能靠关闭校验绕过变更。
FINAL_ATTESTATION="${CHECKPOINT_DIR}/final_checkpoint_attestation.json"
# 完整流水线的来源和完成验证报告位置。
COMPLETION_REPORT="${ROOT}/exp/evaluation/voicegrad30_text_unseen_epoch0500_custom_hifigan/custom_pipeline_completion.json"

# shellcheck source=scripts/common/require_rented_server.sh
# 将租用服务器身份检查函数加载到当前 shell；此处尚未启动训练。
source "${COMMON}/require_rented_server.sh"
# 核对登记主机、machine-id 摘要及项目路径，任何不符都拒绝运行。
require_rented_server "${ROOT}"

# 按 "${REQUESTED_RUN_MODE}" 选择唯一分支；未列出的取值由兜底分支报错。
case "${REQUESTED_RUN_MODE}" in
  # 只接受列出的合法模式；其他值交给后面的错误分支。
  auto|fresh|resume) ;;
  # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
  *) echo "VOICEGRAD_CUSTOM_PIPELINE_MODE 只能是 auto、fresh 或 resume。" >&2; exit 2 ;;
esac
# 此处要求 Python 可执行；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ ! -x "${TRAIN_PYTHON}" || ! -x "${EVAL_PYTHON}" ]]; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "训练或评价 Python 不可执行。" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 2
fi

# 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
mkdir -p "${ROOT}/exp/locks" "${ROOT}/exp/logs"
# 用专用文件描述符打开这项任务的锁文件；持有期间可防止并发覆盖同一实验。
exec 8>"${ROOT}/exp/locks/text_unseen_custom_hifigan_pipeline.lock"
# 尝试立即取得对应任务的排他锁；已有任务持锁时失败退出，不并发启动冲突任务。
if ! flock -n 8; then
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "custom HiFi-GAN 全流程已经在运行。" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 3
fi

# 向 Python 子进程传递同一个权威项目根目录。
export VOICEGRAD_PROJECT_ROOT="${ROOT}"
# 向子任务传递计算设备选择，当前项目只允许在租用服务器计算。
export VOICEGRAD_DEVICE="cuda"
# 切换到已确定的项目根目录，让 Python 模块导入和相对产物路径一致。
cd "${ROOT}"

# 当前流水线阶段名，失败时用来准确指出出错位置。
current_stage="startup"
# 失败时记录当前流水线阶段及原始退出码。
on_failure() {
  # 刚才命令的退出码，保留它以免错误被后续日志覆盖。
  local status=$?
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "CUSTOM_PIPELINE_FAILED stage=${current_stage} exit=${status} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit "${status}"
}
# 登记 on_failure 作为 ERR 事件处理函数，及时记录失败或清理本次临时目录。
trap on_failure ERR

# 把阶段名、开始/结束时间和实际命令包装在一起，保留失败的准确位置。
run_stage() {
  # 当前流水线阶段名，失败时用来准确指出出错位置。
  current_stage="$1"
  # 移除已经读取的阶段名，保留后面的真实命令与参数供本阶段执行。
  shift
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "CUSTOM_PIPELINE_STAGE_BEGIN stage=${current_stage} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # 保持参数原有边界执行本阶段命令；前面阶段名已经由 shift 移除。
  "$@"
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "CUSTOM_PIPELINE_STAGE_COMPLETE stage=${current_stage} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# 确认声码器检查点目录为空，保护已有权重免受 fresh 覆盖。
checkpoint_directory_is_empty() {
  # 此处检查路径是否已存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ ! -e "${CHECKPOINT_DIR}" ]]; then
    # 向调用者返回条件已满足，例如检查点目录为空，可继续选择相应模式。
    return 0
  fi
  # 此处检查目录是否存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
    return 1
  fi
  # first_entry：检查点目录中找到的第一项；为空才能安全判断目录尚无产物。
  local first_entry
  # 此处核对下方明确列出的条件；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if ! first_entry="$(find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)"; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "无法安全读取 checkpoint 目录：${CHECKPOINT_DIR}" >&2
    # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
    return 1
  fi
  # 此处检查值为空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ -z "${first_entry}" ]]
}

# 核对现有最终声码器权重和配置，再生成现场验证记录。
validate_existing_final() {
  # 此处要求文件存在且非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ -s "${FINAL_CHECKPOINT}" ]] || return 1
  # 此处要求文件存在且非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ -s "${CHECKPOINT_DIR}/config.json" ]] || return 1
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：严格检查生成器步数、权重、配置和实现指纹，然后输出指定证明；-u 让日志及时写出。
  # 命令第 2 行：--project-root：明确权威项目根目录。
  # 命令第 3 行：--checkpoint：指定本次真正加载的权重文件。
  # 命令第 4 行：--config：指定与所选模型配套的完整配置。
  # 命令第 5 行：--expected-step：要求检查点内部全局步数精确匹配。
  # 命令第 6 行：--output-report：将验证记录写到指定 JSON 文件。
  "${TRAIN_PYTHON}" -u -m hifigan.custom.validate_checkpoint \
    --project-root "${ROOT}" \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --config "${CHECKPOINT_DIR}/config.json" \
    --expected-step 2500000 \
    --output-report "${FINAL_ATTESTATION}"
}

# 根据用户请求与现有权重验证状态选择 fresh、resume 或跳过。
resolve_training_mode() {
  # 检查现有权重后实际采用的模式，不单靠文件名猜训练完成。
  EFFECTIVE_RUN_MODE=""
  # 标记本次流程是否已经验证最终声码器，避免重复验证阶段。
  FINAL_ALREADY_ATTESTED=0

  # 按 "${REQUESTED_RUN_MODE}" 选择唯一分支；未列出的取值由兜底分支报错。
  case "${REQUESTED_RUN_MODE}" in
    # 新建实验分支，必须保护已有目录。
    fresh)
      # 此处确认已有目录没有产物，避免 fresh 覆盖；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
      if ! checkpoint_directory_is_empty; then
        # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
        echo "显式 fresh 只允许空 checkpoint 目录：${CHECKPOINT_DIR}" >&2
        # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
        return 2
      fi
      # 检查现有权重后实际采用的模式，不单靠文件名猜训练完成。
      EFFECTIVE_RUN_MODE="fresh"
      ;;
    # 续训分支，要求本实验完整断点。
    resume)
      # 此处要求文件存在且非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
      if [[ ! -s "${CHECKPOINT_DIR}/latest.pt" ]]; then
        # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
        echo "显式 resume 需要非空 latest.pt：${CHECKPOINT_DIR}/latest.pt" >&2
        # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
        return 2
      fi
      # 检查现有权重后实际采用的模式，不单靠文件名猜训练完成。
      EFFECTIVE_RUN_MODE="resume"
      ;;
    # 自动判断已有最终权重或 latest.pt 的分支。
    auto)
      # 此处要求文件存在且非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
      if [[ -s "${FINAL_CHECKPOINT}" ]]; then
        # 此处执行完整最终权重验证后才允许跳过训练；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
        if validate_existing_final; then
          # 检查现有权重后实际采用的模式，不单靠文件名猜训练完成。
          EFFECTIVE_RUN_MODE="skip"
          # 标记本次流程是否已经验证最终声码器，避免重复验证阶段。
          FINAL_ALREADY_ATTESTED=1
        # 此处要求文件存在且非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
        elif [[ -s "${CHECKPOINT_DIR}/latest.pt" ]]; then
          # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
          echo "现有 final 未通过验证；将从 latest.pt 安全续训并重建 final。" >&2
          # 检查现有权重后实际采用的模式，不单靠文件名猜训练完成。
          EFFECTIVE_RUN_MODE="resume"
        # 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
        else
          # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
          echo "现有 final 验证失败，且没有可续训的 latest.pt。" >&2
          # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
          return 2
        fi
      # 此处要求文件存在且非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
      elif [[ -s "${CHECKPOINT_DIR}/latest.pt" ]]; then
        # 检查现有权重后实际采用的模式，不单靠文件名猜训练完成。
        EFFECTIVE_RUN_MODE="resume"
      # 此处确认已有目录没有产物，避免 fresh 覆盖；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
      elif checkpoint_directory_is_empty; then
        # 检查现有权重后实际采用的模式，不单靠文件名猜训练完成。
        EFFECTIVE_RUN_MODE="fresh"
      # 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
      else
        # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
        echo "checkpoint 目录非空，但无已验证 final 或 latest.pt；拒绝猜测/覆盖：${CHECKPOINT_DIR}" >&2
        # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
        return 2
      fi
      ;;
  esac

  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "CUSTOM_PIPELINE_TRAINING_DECISION requested=${REQUESTED_RUN_MODE} effective=${EFFECTIVE_RUN_MODE}"
}

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 full_data_and_feature_audit 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：逐项验证活动划分、人数、特征来源和统计量，检查失败就阻止后续任务；-u 让日志及时写出。
# 命令第 3 行：--project-root：明确权威项目根目录；--expected-protocol：要求现场活动清单与此协议一致；--full：执行全量数据核查。
run_stage full_data_and_feature_audit \
  "${TRAIN_PYTHON}" -u -m voicegrad.validate_setup \
  --project-root "${ROOT}" --expected-protocol text-unseen --full

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 custom_vocoder_preflight 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：训练本项目修改版 HiFi-GAN，或按 preflight 参数只核对输入；-u 让日志及时写出。
# 命令第 3 行：--project-root：明确权威项目根目录。
# 命令第 4 行：--config：指定与所选模型配套的完整配置。
# 命令第 5 行：--checkpoint-dir：指定本实验检查点目录。
# 命令第 6 行：--full-file-check：逐文件核对特征来源与完整性；--preflight-only：只审计不更新模型。
run_stage custom_vocoder_preflight \
  "${TRAIN_PYTHON}" -u -m hifigan.custom.train \
  --project-root "${ROOT}" \
  --config "${ROOT}/hifigan/configs/text_unseen.json" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --full-file-check --preflight-only

# 按顺序执行 resolve_custom_vocoder_training_state 阶段，记录开始/完成时间；失败立即阻止下游继续。
run_stage resolve_custom_vocoder_training_state resolve_training_mode

# 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ "${EFFECTIVE_RUN_MODE}" == "skip" ]]; then
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "CUSTOM_PIPELINE_STAGE_SKIPPED stage=train_custom_hifigan reason=validated_final_checkpoint"
# 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
else
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：按顺序执行 train_custom_hifigan 阶段，记录开始/完成时间；失败立即阻止下游继续。
  # 命令第 2 行：只为这个子阶段设置 Python 或新建/恢复模式环境变量，不改变其他已运行任务的环境。
  # 命令第 3 行：按给定协议训练本项目修改版声码器；等待该阶段成功后才执行下一条命令。
  run_stage train_custom_hifigan \
    env VOICEGRAD_PYTHON="${TRAIN_PYTHON}" VOCODER_RUN_MODE="${EFFECTIVE_RUN_MODE}" \
    bash "${COMMON}/03_train_custom_hifigan.sh" text-unseen
fi

# 此处按整数判断相等；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ "${FINAL_ALREADY_ATTESTED}" -eq 0 ]]; then
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：按顺序执行 attest_final_custom_checkpoint 阶段，记录开始/完成时间；失败立即阻止下游继续。
  # 命令第 2 行：严格检查生成器步数、权重、配置和实现指纹，然后输出指定证明；-u 让日志及时写出。
  # 命令第 3 行：--project-root：明确权威项目根目录。
  # 命令第 4 行：--checkpoint：指定本次真正加载的权重文件。
  # 命令第 5 行：--config：指定与所选模型配套的完整配置。
  # 命令第 6 行：--expected-step：要求检查点内部全局步数精确匹配。
  # 命令第 7 行：--output-report：将验证记录写到指定 JSON 文件。
  run_stage attest_final_custom_checkpoint \
    "${TRAIN_PYTHON}" -u -m hifigan.custom.validate_checkpoint \
    --project-root "${ROOT}" \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --config "${CHECKPOINT_DIR}/config.json" \
    --expected-step 2500000 \
    --output-report "${FINAL_ATTESTATION}"
# 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
else
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "CUSTOM_PIPELINE_STAGE_SKIPPED stage=attest_final_custom_checkpoint reason=validated_during_auto_detection"
fi

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 evaluation_environment_preflight 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：预先检查客观指标环境和模型依赖，避免生成全部语音后才发现评价不能运行；-u 让日志及时写出。
# 命令第 3 行：--project-root：明确权威项目根目录。
# 命令第 4 行：--vocoder-kind：标记自训 custom 或公开 public 来源。
# 命令第 5 行：--vocoder-checkpoint：指定声码器生成器权重。
# 命令第 6 行：--vocoder-config：指定声码器结构与声学参数。
run_stage evaluation_environment_preflight \
  "${EVAL_PYTHON}" -u -m evaluation.preflight_paper_metrics \
  --project-root "${ROOT}" \
  --vocoder-kind custom \
  --vocoder-checkpoint "${FINAL_CHECKPOINT}" \
  --vocoder-config "${CHECKPOINT_DIR}/config.json"

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 infer_teacher_epoch500_custom_8100 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：只为这个子阶段设置 Python 或新建/恢复模式环境变量，不改变其他已运行任务的环境。
# 命令第 3 行：生成指定教师轮数与声码器来源的配对波形；等待该阶段成功后才执行下一条命令。
run_stage infer_teacher_epoch500_custom_8100 \
  env VOICEGRAD_PYTHON="${TRAIN_PYTHON}" \
  bash "${ROOT}/text_unseen/04_infer_8100.sh" 500 custom

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 evaluate_custom_8100_and_compare 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：只为这个子阶段设置 Python 或新建/恢复模式环境变量，不改变其他已运行任务的环境。
# 命令第 3 行：评价前一步同版本生成的音频；等待该阶段成功后才执行下一条命令。
run_stage evaluate_custom_8100_and_compare \
  env VOICEGRAD_PYTHON="${EVAL_PYTHON}" \
  bash "${ROOT}/text_unseen/05_evaluate.sh" 500 custom

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：按顺序执行 final_lineage_and_completion_audit 阶段，记录开始/完成时间；失败立即阻止下游继续。
# 命令第 2 行：最终复核自训声码器、教师、配对及指标的完整来源链；-u 让日志及时写出。
# 命令第 3 行：--project-root：明确权威项目根目录；--output-report：将验证记录写到指定 JSON 文件。
run_stage final_lineage_and_completion_audit \
  "${EVAL_PYTHON}" -u -m evaluation.verify_custom_pipeline \
  --project-root "${ROOT}" --output-report "${COMPLETION_REPORT}"

# 当前流水线阶段名，失败时用来准确指出出错位置。
current_stage="complete"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "CUSTOM_HIFIGAN_TO_PAPER_COMPARISON_COMPLETE report=${COMPLETION_REPORT}"
