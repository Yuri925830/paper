#!/usr/bin/env bash
# Fail-closed guard shared by every project compute wrapper.

# 集中检查允许的租用服务器身份，防止项目跑到禁用机器。
require_rented_server() {
  # 函数接收或从脚本推导的项目根目录。
  local project_root="$1"
  # 当前允许使用的租用服务器主机名，实验室机器不能冒用。
  local expected_hostname="Pbrrlv"
  # 登记的租用机器身份摘要，与主机名一起核对。
  local expected_machine_id_sha256="d29dd7f850d46b841a76a7e54e30df1d44a8fd801a342e52fdb97472e944a8f5"
  # 允许执行项目计算的准确目录。
  local expected_project_root="/root/FsatVoiceGrad"
  # actual_hostname：现场读取的短主机名；actual_machine_id_sha256：现场 machine-id 的 SHA-256，不输出原始机器标识；actual_project_root：消除相对路径和符号链接后的实际项目目录。
  local actual_hostname actual_machine_id_sha256 actual_project_root

  # 现场读取的短主机名。
  actual_hostname="$(hostname -s)"
  # 按 "${actual_hostname}" 选择唯一分支；未列出的取值由兜底分支报错。
  case "${actual_hostname}" in
    # 明确禁止 speech101、speech102、speech103、speech104；任意空闲状态也不能绕过。
    speech101|speech102|speech103|speech104)
      # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
      echo "项目规则禁止在实验室主机 ${actual_hostname} 运行任何项目计算。" >&2
      # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
      return 2
      ;;
  esac
  # 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ "${actual_hostname}" != "${expected_hostname}" ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "项目计算只允许在已登记的租用服务器 ${expected_hostname}；当前是 ${actual_hostname}。" >&2
    # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
    return 2
  fi
  # 此处要求身份文件可读；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ ! -r /etc/machine-id ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "无法读取租用服务器身份文件 /etc/machine-id。" >&2
    # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
    return 2
  fi
  # 现场 machine-id 的 SHA-256，不输出原始机器标识。
  actual_machine_id_sha256="$(sha256sum -- /etc/machine-id | awk '{print tolower($1)}')"
  # 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ "${actual_machine_id_sha256}" != "${expected_machine_id_sha256}" ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "租用服务器 machine-id 哈希不匹配，拒绝运行。" >&2
    # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
    return 2
  fi
  # 消除相对路径和符号链接后的实际项目目录。
  actual_project_root="$(realpath -- "${project_root}")"
  # 此处精确比较模式、路径或指纹；! 表示取反，&& 要求前项成功，|| 在前项失败时处理。
  if [[ "${actual_project_root}" != "${expected_project_root}" ]]; then
    # 将具体失败原因输出到标准错误，远程日志可据此定位缺失文件、非法参数或指纹差异。
    echo "项目计算必须使用 ${expected_project_root}；当前是 ${actual_project_root}。" >&2
    # 向调用者返回检查未满足或参数错误，由上层决定停止或采用已明确的恢复分支。
    return 2
  fi
}
