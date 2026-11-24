#!/usr/bin/env bash
# 下载、校验并解压论文使用的官方 VCTK 0.92 归档。

set -euo pipefail

# 本脚本所在的绝对目录，用于定位同仓库的公共入口。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录；有显式环境变量时采用指定值，否则从脚本位置推导。
PROJECT_ROOT="${VOICEGRAD_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# 保存下载归档的目录。
DOWNLOAD_DIR="${VCTK_DOWNLOAD_DIR:-${PROJECT_ROOT}/data/downloads}"
# 本次 VCTK ZIP 文件的准确路径。
ARCHIVE="${DOWNLOAD_DIR}/VCTK-Corpus-0.92.zip"
# 存放解压语料及保留备份的父目录。
EXTRACT_ROOT="${VCTK_EXTRACT_ROOT:-${PROJECT_ROOT}/data/raw}"
# VCTK 0.92 官方压缩包下载地址。
URL="${VCTK_DOWNLOAD_URL:-https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content}"
# 固定的官方归档校验值，用于完整性检查。
EXPECTED_MD5="8a6ba2946b36fcbef0212cad601f4bfa"
# 官方归档预期字节数，下载完成后先检查是否完整。
EXPECTED_BYTES="11747302977"

# 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
mkdir -p "${DOWNLOAD_DIR}" "${EXTRACT_ROOT}" "${PROJECT_ROOT}/exp/logs"

# 复用下载前先检查归档存在、字节大小和 MD5，防止把半份下载当完成。
archive_is_verified() {
  # 此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ -f "${ARCHIVE}" ]] || return 1
  # 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ "$(stat -c '%s' "${ARCHIVE}")" == "${EXPECTED_BYTES}" ]] || return 1
  # 把预登记 MD5 与现场归档进行核对，任何字节不完整都会阻止解压。
  printf '%s  %s\n' "${EXPECTED_MD5}" "${ARCHIVE}" | md5sum -c - >/dev/null
}

# 此处核对下方明确列出的条件；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if archive_is_verified; then
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "verified archive already exists; download skipped"
# 此处核对下方明确列出的条件；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
elif command -v aria2c >/dev/null 2>&1; then
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：以断点续传和受控并发下载固定 VCTK 归档，完成后仍需严格摘要核验。
  # 命令第 2 行：--min-split-size：避免过小片段造成额外网络开销；--file-allocation：关闭下载前预分配以降低启动耗时；--auto-file-renaming：禁止自动改下载文件名，保证后续检查准确路径。
  # 命令第 3 行：--allow-overwrite：允许在固定下载目标续传或更新归档，不涉及模型权重。
  aria2c --continue=true --max-connection-per-server=16 --split=16 \
    --min-split-size=1M --file-allocation=none --auto-file-renaming=false \
    --allow-overwrite=true -d "${DOWNLOAD_DIR}" -o "$(basename "${ARCHIVE}")" "${URL}"
# 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
else
  # 使用断点续传下载固定官方归档，并设置重试次数及连接超时。
  wget -c --tries=20 --timeout=30 -O "${ARCHIVE}" "${URL}"
fi

# 下载完成后现场读取的压缩包字节数。
actual_bytes="$(stat -c '%s' "${ARCHIVE}")"
# 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
[[ "${actual_bytes}" == "${EXPECTED_BYTES}" ]] || {
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "VCTK byte-size mismatch: ${actual_bytes} != ${EXPECTED_BYTES}" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 1
}
# 把预登记 MD5 与现场归档进行核对，任何字节不完整都会阻止解压。
printf '%s  %s\n' "${EXPECTED_MD5}" "${ARCHIVE}" | md5sum -c -
# 完整测试 ZIP 内部文件校验，确认下载内容没有损坏后才解压。
unzip -tq "${ARCHIVE}"

# 已核验语料的正式目录。
CORPUS_ROOT="${EXTRACT_ROOT}/VCTK-Corpus-0.92"
# 记录本次归档 MD5 的完成标记，配合库存检查决定能否复用。
VERIFICATION_FILE="${CORPUS_ROOT}/.voicegrad_archive_md5"

# 核对官方 mic1 音频数和说话人数，避免使用残缺或混合语料。
corpus_counts_are_valid() {
  # 此处检查目录是否存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ -d "$1/wav48_silence_trimmed" ]] || return 1
  # mic1_count：第一支麦克风音频文件数，官方完整版本应为 44455；speaker_count：语料说话人目录数量，此库存校验要求 110。
  local mic1_count speaker_count
  # 第一支麦克风音频文件数，官方完整版本应为 44455。
  mic1_count="$(find "$1/wav48_silence_trimmed" -type f -name '*_mic1.flac' | wc -l)"
  # 语料说话人目录数量，此库存校验要求 110。
  speaker_count="$(find "$1/wav48_silence_trimmed" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  # 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ "${mic1_count}" == "44455" && "${speaker_count}" == "110" ]]
}

# A directory is reusable only when this script completed both archive and
# inventory verification previously.  An old or interrupted directory is kept
# as a timestamped backup, never silently merged with a fresh extraction.
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
# 命令第 2 行：此处检查目录是否存在；精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
# 命令第 3 行：此处检查完整 mic1 文件数和说话人数；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -f "${VERIFICATION_FILE}" ]] \
  && [[ "$(tr -d '\r\n' <"${VERIFICATION_FILE}")" == "${EXPECTED_MD5}" ]] \
  && corpus_counts_are_valid "${CORPUS_ROOT}"; then
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  echo "verified extracted corpus already exists; extraction skipped"
# 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
else
  # 本次独立解压暂存目录，验证通过前不和旧语料混合。
  STAGE_ROOT="$(mktemp -d "${EXTRACT_ROOT}/.vctk-extract.XXXXXX")"
  # 临时区域内按统一目录名摆放的解压语料。
  STAGED_CORPUS="${STAGE_ROOT}/VCTK-Corpus-0.92"
  # The official 0.92 ZIP stores speaker-info.txt/txt/wav48_* directly at
  # archive root.  Create the corpus directory explicitly instead of assuming
  # that the ZIP contains a top-level VCTK-Corpus-0.92 folder.
  # 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
  mkdir "${STAGED_CORPUS}"
  # 将已经通过大小和摘要检查的归档解到新暂存语料目录，不与旧目录混合。
  unzip -q "${ARCHIVE}" -d "${STAGED_CORPUS}"
  # 此处检查完整 mic1 文件数和说话人数；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if ! corpus_counts_are_valid "${STAGED_CORPUS}"; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "staged VCTK extraction failed the 44455/110 inventory check: ${STAGE_ROOT}" >&2
    # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
    exit 1
  fi
  # 写入已经核验的归档指纹作为完成标记，后续仍会检查语料库存数量。
  printf '%s\n' "${EXPECTED_MD5}" >"${STAGED_CORPUS}/.voicegrad_archive_md5"
  # 此处检查路径是否已存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ -e "${CORPUS_ROOT}" ]]; then
    # 为已有未核验语料建立的带时间戳备份目录。
    BACKUP_ROOT="${EXTRACT_ROOT}/unverified-$(date -u +%Y%m%dT%H%M%SZ)"
    # 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
    mkdir -p "${BACKUP_ROOT}"
    # 移动本次已明确的暂存语料或保留备份；已有未核验语料被保留，便于恢复。
    mv "${CORPUS_ROOT}" "${BACKUP_ROOT}/VCTK-Corpus-0.92"
    # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
    echo "previous unverified extraction preserved at ${BACKUP_ROOT}/VCTK-Corpus-0.92"
  fi
  # 移动本次已明确的暂存语料或保留备份；已有未核验语料被保留，便于恢复。
  mv "${STAGED_CORPUS}" "${CORPUS_ROOT}"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：只尝试删除已经搬空的本次暂存目录；非空则保留供人工检查。
  # 命令第 2 行：向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  rmdir "${STAGE_ROOT}" 2>/dev/null || \
    echo "non-corpus archive entries retained for inspection at ${STAGE_ROOT}"
fi

# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：第一支麦克风音频文件数，官方完整版本应为 44455。
# 命令第 2 行：只统计 mic1 的普通 FLAC 文件，不能把 mic2 或目录计入库存。
mic1_count="$(find "${CORPUS_ROOT}/wav48_silence_trimmed" \
  -type f -name '*_mic1.flac' | wc -l)"
# 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
# 命令第 1 行：语料说话人目录数量，此库存校验要求 110。
# 命令第 2 行：只统计这一层的说话人目录，不把嵌套文件夹重复计数。
speaker_count="$(find "${CORPUS_ROOT}/wav48_silence_trimmed" \
  -mindepth 1 -maxdepth 1 -type d | wc -l)"
# 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
[[ "${mic1_count}" == "44455" ]] || {
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "VCTK mic1 count mismatch: ${mic1_count} != 44455" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 1
}
# 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
[[ "${speaker_count}" == "110" ]] || {
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  echo "VCTK speaker count mismatch: ${speaker_count} != 110" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 1
}
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "verified mic1 files=${mic1_count} speakers=${speaker_count}"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
echo "VCTK_092_DOWNLOAD_AND_VERIFY_COMPLETE"
