#!/usr/bin/env bash
# Prepare the exact public ppg-vc snapshot used by the audited teacher rebuild.
#
# Default (network clone):
#   bash scripts/paper/01_prepare_ppg_vc.sh
#
# From the transferred Git-only partial-clone archive:
#   PPG_VC_ARCHIVE=/path/to/ppg-vc-partial-git.tar.gz \
#     bash scripts/paper/01_prepare_ppg_vc.sh
#
# Optional destination override:
#   PPG_VC_TARGET=/absolute/path bash scripts/paper/01_prepare_ppg_vc.sh

# 启用失败即停、未定义变量报错和管道错误检查；带 -E 时让函数继承 ERR 处理。
set -Eeuo pipefail

# 上游 ppg-vc 必须对应的固定提交，避免下载到后来修改过的版本。
readonly EXPECTED_COMMIT="b59cb9862cf4b82a3bdb589950e25cab85fc9b03"
# 本实验 BNF 提取器权重的固定字节指纹。
readonly EXPECTED_BNF_SHA256="94362c99f408cbe891785537e222c2f1f9119edb11b95107b9faa4c0f0d679a5"
# 本实验说话人编码器权重的固定字节指纹。
readonly EXPECTED_SPK_SHA256="bc7ff82ef75becd495aab2ede3a8220da393a717f178ae9534df355a6173bbca"
# 固定的 ppg-vc 官方仓库来源。
readonly UPSTREAM_URL="https://github.com/liusongxiang/ppg-vc.git"
# 上游快照内 Conformer BNF 检查点的相对位置。
readonly BNF_RELATIVE_PATH="conformer_ppg_model/en_conformer_ctc_att/24epoch.pth"
# 上游快照内 GE2E 说话人编码器检查点的位置。
readonly SPK_RELATIVE_PATH="speaker_encoder/ckpt/pretrained_bak_5805000.pt"

# 统一打印错误并以失败状态退出。
die() {
  # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
  printf '[ERROR] %s\n' "$*" >&2
  # 用非零或原始失败退出码结束，防止下游把当前阶段误判为完成。
  exit 1
}

# 确认安装/核验所需系统命令存在。
require_command() {
  # 只检查所需命令是否在 PATH 中可用，不执行该命令的项目工作。
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

# 计算指定上游模型文件的原始字节 SHA-256。
sha256_file() {
  # 按原始字节计算文件 SHA-256，并仅提取小写摘要供后续精确比较。
  sha256sum -- "$1" | awk '{print tolower($1)}'
}

# 严格检查上游 Git 提交、受控源码无改动以及两份模型权重。
verify_checkout() {
  # 待验证的上游 Git 工作区。
  local checkout_root="$1"
  # Git 识别出的真实工作区根目录。
  local resolved_root=""
  # 从当前上游工作区读取的提交号。
  local commit=""
  # 上游受控文件是否存在已暂存或未暂存改动；缓存文件不参与此检查。
  local tracked_status=""
  # 现场 BNF 模型文件的字节摘要。
  local actual_bnf_sha256=""
  # 现场说话人模型文件的字节摘要。
  local actual_spk_sha256=""

  # 此处检查目录是否存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ -d "$checkout_root/.git" ]] || die "Not a ppg-vc Git checkout: $checkout_root"

  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：Git 识别出的真实工作区根目录。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  resolved_root="$(git -C "$checkout_root" rev-parse --show-toplevel 2>/dev/null)" \
    || die "Cannot inspect Git checkout: $checkout_root"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：此处检查路径是否已存在；精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  [[ "$(realpath -e -- "$resolved_root")" == "$(realpath -e -- "$checkout_root")" ]] \
    || die "Git top level does not match destination: $resolved_root"

  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：从当前上游工作区读取的提交号。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  commit="$(git -C "$checkout_root" rev-parse --verify HEAD 2>/dev/null)" \
    || die "Cannot resolve ppg-vc HEAD in $checkout_root"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  [[ "$commit" == "$EXPECTED_COMMIT" ]] \
    || die "Unexpected ppg-vc commit in $checkout_root: expected $EXPECTED_COMMIT, got $commit"

  # Ignore untracked caches, but reject every staged or unstaged tracked-file change.
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：上游受控文件是否存在已暂存或未暂存改动；缓存文件不参与此检查。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  tracked_status="$(git -C "$checkout_root" status --porcelain --untracked-files=no)" \
    || die "Cannot audit tracked files in $checkout_root"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：此处检查值为空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  [[ -z "$tracked_status" ]] \
    || die "Tracked ppg-vc files are not clean in $checkout_root"

  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  [[ -f "$checkout_root/$BNF_RELATIVE_PATH" ]] \
    || die "Missing BNF checkpoint: $checkout_root/$BNF_RELATIVE_PATH"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  [[ -f "$checkout_root/$SPK_RELATIVE_PATH" ]] \
    || die "Missing speaker checkpoint: $checkout_root/$SPK_RELATIVE_PATH"

  # 现场 BNF 模型文件的字节摘要。
  actual_bnf_sha256="$(sha256_file "$checkout_root/$BNF_RELATIVE_PATH")"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  [[ "$actual_bnf_sha256" == "$EXPECTED_BNF_SHA256" ]] \
    || die "BNF checkpoint SHA-256 mismatch: expected $EXPECTED_BNF_SHA256, got $actual_bnf_sha256"

  # 现场说话人模型文件的字节摘要。
  actual_spk_sha256="$(sha256_file "$checkout_root/$SPK_RELATIVE_PATH")"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  [[ "$actual_spk_sha256" == "$EXPECTED_SPK_SHA256" ]] \
    || die "Speaker checkpoint SHA-256 mismatch: expected $EXPECTED_SPK_SHA256, got $actual_spk_sha256"
}

# 只接受包含 .git 元数据的恢复归档，避免错误压缩包写入未知内容。
validate_git_only_archive() {
  # 待验证或恢复的压缩包绝对路径。
  local archive="$1"
  # 逐条读取的归档成员名称。
  local entry=""
  # 去掉开头 ./ 后的归档成员名，便于统一检查是否限于 .git。
  local normalized=""
  # 标记归档是否包含 .git/HEAD，防止把普通文件包当 Git 元数据。
  local saw_head=0

  # The transferred recovery artifact is intentionally Git-only. Restricting the
  # accepted member names keeps extraction from writing outside the new staging
  # directory and prevents an unrelated source archive from being mistaken for it.
  # 逐字读取压缩包成员名，不拆分含空格的路径，检查其是否都限于 .git。
  while IFS= read -r entry; do
    # 去掉开头 ./ 后的归档成员名，便于统一检查是否限于 .git。
    normalized="${entry#./}"
    # 按 "$normalized" 选择唯一分支；未列出的取值由兜底分支报错。
    case "$normalized" in
      # 仅接受 .git 本身及其内部成员；其他归档路径由兜底分支拒绝。
      .git|.git/*) ;;
      # 兜底处理非法取值，明确报错退出，不猜测应该运行哪个实验。
      *) die "PPG_VC_ARCHIVE is not the expected Git-only archive (member: $entry)" ;;
    esac
    # 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
    [[ "$normalized" == ".git/HEAD" ]] && saw_head=1
  # 把 tar 列出的全部归档成员送入上面的逐行检查循环；只列目录不解压。
  done < <(tar -tf "$archive")

  # 此处按整数判断相等；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ "$saw_head" -eq 1 ]] || die "PPG_VC_ARCHIVE does not contain .git/HEAD"
}

# 运行前确认 git 命令可用，缺少时在下载或改动前停止。
require_command git
# 运行前确认 realpath 命令可用，缺少时在下载或改动前停止。
require_command realpath
# 运行前确认 sha256sum 命令可用，缺少时在下载或改动前停止。
require_command sha256sum
# 运行前确认 awk 命令可用，缺少时在下载或改动前停止。
require_command awk
# 运行前确认 tar 命令可用，缺少时在下载或改动前停止。
require_command tar
# 运行前确认 mktemp 命令可用，缺少时在下载或改动前停止。
require_command mktemp
# 运行前确认 mv 命令可用，缺少时在下载或改动前停止。
require_command mv

# 安装脚本的真实所在目录。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# 函数接收或从脚本推导的项目根目录。
project_root="$(cd -- "$script_dir/../.." && pwd -P)"
# 将要发布 ppg-vc 快照的精确目标目录。
target="$(realpath -m -- "${PPG_VC_TARGET:-$project_root/external/legacy_compute_bnf}")"
# 目标目录的父级，用于在同一文件系统建立暂存目录。
target_parent="$(dirname -- "$target")"

# 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
[[ "$target" != "/" ]] || die "Refusing unsafe PPG_VC_TARGET: $target"

# 此处检查路径是否已存在；检查符号链接；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -e "$target" || -L "$target" ]]; then
  # 重新核对这一上游快照的固定提交、源码清洁状态和两份权重摘要。
  verify_checkout "$target"
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  printf '[OK] Existing ppg-vc checkout is already exact and verified: %s\n' "$target"
  # 正常结束当前脚本，保留已经验证的现有产物。
  exit 0
fi

# 只创建下方明确指定的输出、锁或暂存目录；-p 保留已经存在的目录。
mkdir -p -- "$target_parent"
# 本次专用临时工作区，核验通过前不会发布到正式目标。
staging_dir="$(mktemp -d -- "$target_parent/.legacy_compute_bnf.staging.XXXXXXXX")"

# 退出时仅清理本次专用暂存目录，正式上游快照与用户目录保留。
cleanup() {
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：此处检查目录是否存在；检查值非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  # 命令第 2 行：此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ -n "${staging_dir:-}" && -d "$staging_dir" \
        && "$staging_dir" == "$target_parent"/.legacy_compute_bnf.staging.* ]]; then
    # 仅在上面的非空、目录存在且专用前缀检查全部通过后，清理本次暂存目录；绝不用于项目根或用户目录。
    rm -rf -- "$staging_dir"
  fi
}
# 登记 cleanup 作为 EXIT 事件处理函数，及时记录失败或清理本次临时目录。
trap cleanup EXIT

# 此处检查值非空；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -n "${PPG_VC_ARCHIVE:-}" ]]; then
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：待验证或恢复的压缩包绝对路径。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  archive="$(realpath -e -- "$PPG_VC_ARCHIVE")" \
    || die "PPG_VC_ARCHIVE does not exist: $PPG_VC_ARCHIVE"
  # 此处要求普通文件存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  [[ -f "$archive" ]] || die "PPG_VC_ARCHIVE is not a regular file: $archive"
  # 核对恢复归档成员只包含 Git 元数据，再允许解包到新建暂存区。
  validate_git_only_archive "$archive"
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  printf '[INFO] Restoring ppg-vc Git metadata from %s\n' "$archive"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：把已检查成员名称的 Git 元数据归档解到本次暂存目录，不继承归档中的所有者和权限。
  # 命令第 2 行：解包时不继承压缩包里原有用户身份和文件权限，使用当前安全环境的设置。
  tar --extract --file="$archive" --directory="$staging_dir" \
    --no-same-owner --no-same-permissions

  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：先确认固定提交对象存在；缺失时由下一行明确报错或向登记上游取回。
  # 命令第 2 行：前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  git -C "$staging_dir" cat-file -e "$EXPECTED_COMMIT^{commit}" 2>/dev/null \
    || die "Archive does not contain commit $EXPECTED_COMMIT"
  # 为本次恢复的暂存仓库登记已核验的上游地址，便于补取缺失对象。
  git -C "$staging_dir" remote set-url origin "$UPSTREAM_URL"
# 前面的条件未满足时执行这一分支，具体恢复/报错动作见下面命令。
else
  # 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
  printf '[INFO] Cloning %s\n' "$UPSTREAM_URL"
  # 将固定上游仓库克隆到本次新建暂存目录，按需取对象，尚不发布为正式依赖。
  git clone --filter=blob:none --no-checkout -- "$UPSTREAM_URL" "$staging_dir"
  # 下方是一条由反斜杠连接的完整命令。为保持 shell 语义，各续行说明集中放在命令之前：
  # 命令第 1 行：先确认固定提交对象存在；缺失时由下一行明确报错或向登记上游取回。
  # 命令第 2 行：只取登记的固定提交，不切换到不确定的最新上游版本。
  git -C "$staging_dir" cat-file -e "$EXPECTED_COMMIT^{commit}" 2>/dev/null \
    || git -C "$staging_dir" fetch --depth=1 origin "$EXPECTED_COMMIT"
fi

# 在新暂存工作区开启稀疏检出，只取项目真正需要的目录。
git -C "$staging_dir" sparse-checkout init --cone
# 仅展开 BNF 与说话人编码器两个目录，降低上游快照磁盘占用。
git -C "$staging_dir" sparse-checkout set conformer_ppg_model speaker_encoder
# 只在新建且受控的上游暂存工作区检出固定提交；此命令不能移用于用户已有工作区。
git -C "$staging_dir" checkout --force --detach "$EXPECTED_COMMIT"
# 重新核对这一上游快照的固定提交、源码清洁状态和两份权重摘要。
verify_checkout "$staging_dir"

# GNU mv -T keeps the destination exact; -n makes a concurrent destination
# creation non-destructive. Detect that no-clobber case before declaring success.
# 将验证通过的专用暂存目录发布到精确目标；-n 拒绝覆盖并发出现的已有目录。
mv -T -n -- "$staging_dir" "$target"
# 此处检查目录是否存在；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
if [[ -d "$staging_dir" ]]; then
  # 前一项安全/完整性检查失败时给出具体原因并结束，保留已有正式目录。
  die "Destination appeared while publishing; preserved it and did not overwrite: $target"
fi
# 本次专用临时工作区，核验通过前不会发布到正式目标。
staging_dir=""

# 重新核对这一上游快照的固定提交、源码清洁状态和两份权重摘要。
verify_checkout "$target"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
printf '[OK] Prepared exact ppg-vc snapshot: %s\n' "$target"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
printf '[OK] commit=%s\n' "$EXPECTED_COMMIT"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
printf '[OK] BNF_SHA256=%s\n' "$EXPECTED_BNF_SHA256"
# 向日志输出当前阶段、完成标志或已核验路径；是否完成仍由前面的实际检查决定。
printf '[OK] SPK_SHA256=%s\n' "$EXPECTED_SPK_SHA256"
