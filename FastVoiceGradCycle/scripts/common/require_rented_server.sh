#!/usr/bin/env bash
# 兼容旧文件名，同时只允许明确登记的 speech GPU 计算节点运行项目。  # 文件用途说明。

# 校验当前主机和项目目录是否符合实验室 speech GPU 运行规则。  # 新函数说明。
require_speech_gpu_server() {  # 调用方把项目根目录作为第一个参数传入。
  local project_root="$1"  # 保存调用方给出的项目根目录。
  local expected_project_root="/nfs/speechst2/storage/rich_925/FastVoiceGrad_cycle_v1_restore_20261001/root/FsatVoiceGrad"  # 权威恢复目录。
  local actual_hostname  # 保存去掉域名后的现场主机名。
  local actual_project_root  # 保存消除符号链接后的现场项目目录。

  actual_hostname="$(hostname -s)"  # 从操作系统读取当前短主机名。
  case "${actual_hostname}" in  # 先处理登录节点与永久禁用节点，再判断允许列表。
    voiceexp-lab)  # 登录/管理节点不能执行训练、推理、评估或项目测试。
      echo "voiceexp-lab 仅用于登录和文件管理，禁止运行项目计算。" >&2  # 把明确原因写入错误日志。
      return 2  # 用非零状态阻止上层脚本继续。
      ;;  # 结束登录节点分支。
    speech101|speech102|speech103|speech104)  # 四台主机永久禁用，即使 GPU 空闲也不例外。
      echo "项目规则永久禁止在 ${actual_hostname} 运行任何项目计算。" >&2  # 报告具体违规主机。
      return 2  # 用非零状态阻止上层脚本继续。
      ;;  # 结束永久禁用分支。
    speech001|speech002|speech003|speech004|speech105|speech106|speech107|speech108|speech109|speech110|speech201|speech202|speech203|speech204|speech205|speech206|speech301)  # 允许候选计算节点的精确清单。
      :  # 主机名已通过，继续核对权威项目目录。
      ;;  # 结束允许节点分支。
    *)  # 任何未登记主机都按失败关闭处理。
      echo "项目计算只允许在已登记的 speech GPU 节点；当前是 ${actual_hostname}。" >&2  # 报告未知主机。
      return 2  # 用非零状态阻止上层脚本继续。
      ;;  # 结束未知主机分支。
  esac  # 完成主机名判断。

  actual_project_root="$(realpath -- "${project_root}")"  # 规范化项目目录后再精确比较。
  if [[ "${actual_project_root}" != "${expected_project_root}" ]]; then  # 旧副本或临时目录都不能执行计算。
    echo "项目计算必须使用 ${expected_project_root}；当前是 ${actual_project_root}。" >&2  # 显示期望与实际路径。
    return 2  # 用非零状态阻止上层脚本继续。
  fi  # 完成项目目录检查。
}  # 完成新的 speech GPU 安全函数。

# 保留旧函数名，让现有入口无需一次性改名也能使用新规则。  # 兼容层说明。
require_rented_server() {  # 旧脚本仍调用这个名称。
  require_speech_gpu_server "$@"  # 原样转交全部参数给新的门禁函数。
}  # 完成兼容函数。
