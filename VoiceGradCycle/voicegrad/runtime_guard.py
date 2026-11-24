# 只允许在用户当前登记的 Matpool 出租服务器上执行项目计算。
"""Fail-closed identity guard for the user's currently rented Matpool server.

Project compute is authorized only on the rented host registered below.  Keeping
this check in Python as well as the shell wrappers prevents an accidental direct
``python -m ...`` invocation on a local machine or a laboratory server.
"""

# 延后解析类型标注，便于在函数定义中引用稍后声明的模型和路径类型。
from __future__ import annotations

# 计算文件或数组的 SHA-256 指纹，识别数据或模型内容变化。
import hashlib
# 读取当前服务器主机名，核对项目计算所在机器。
import socket
# 用路径对象定位数据、权重和报告，减少手工拼接目录的错误。
from pathlib import Path
# 这些类型标注说明参数、返回数组和字段结构，不改变音频或模型的数值。
from typing import Optional


# 事先登记的出租服务器名称。
EXPECTED_HOSTNAME = "Pbrrlv"
# 登记的出租服务器机器指纹。
EXPECTED_MACHINE_ID_SHA256 = (
    # 当前获准 Matpool 出租服务器的机器身份摘要，不能用其他空闲机器替代。
    "d29dd7f850d46b841a76a7e54e30df1d44a8fd801a342e52fdb97472e944a8f5"
)
# 允许执行项目计算的远端目录。
EXPECTED_PROJECT_ROOT = Path("/root/FsatVoiceGrad")
# 用户禁止使用的四台实验室服务器，即使 GPU 空闲也不得运行项目计算。
FORBIDDEN_LAB_HOSTS = {"speech101", "speech102", "speech103", "speech104"}


# 专门表示主机身份校验失败，阻止项目计算进入错误服务器。
class RuntimeIdentityError(RuntimeError):
    # 主机身份不符时用此异常明确停止，避免把计算送到禁用服务器。
    """Raised when a project workload is not running on the rented server."""


# 从机器标识等原始字节计算稳定指纹；下面的参数分别限定输入数据和处理约束。
def _sha256_bytes(payload: bytes) -> str:
    # 返回机器标识字节的 64 位十六进制摘要，避免仅凭可重名的主机名判断。
    return hashlib.sha256(payload).hexdigest()


# 把实际主机、机器指纹与项目目录对照出租服务器登记值；下面的参数分别限定输入数据和处理约束。
def validate_runtime_identity(
    # 从这里起的参数必须写明名称传入，减少路径或审核开关位置混淆。
    *,
    # 实际运行机器的主机名。
    hostname: str,
    # 运行机器标识文件的 SHA-256 指纹。
    machine_id_sha256: str,
    # 保存代码、数据与检查点的项目目录。
    project_root: Path,
    # 事先登记的出租服务器名称。
    expected_hostname: str = EXPECTED_HOSTNAME,
    # 登记的出租服务器机器指纹。
    expected_machine_id_sha256: str = EXPECTED_MACHINE_ID_SHA256,
    # 允许执行项目计算的远端目录。
    expected_project_root: Path = EXPECTED_PROJECT_ROOT,
# 本接口负责把实际主机、机器指纹与项目目录对照出租服务器登记值，处理结果通过对象状态或文件保存，不额外返回数值。
) -> None:
    # 把实际主机、机器指纹与项目目录对照出租服务器登记值。
    """Validate already-resolved identity facts; split out for unit testing."""

    # 去掉域名后可与禁用主机列表比较的短名称。
    short_hostname = hostname.strip().split(".", 1)[0]
    # 按去掉域名后可与禁用主机列表比较的短名称决定是否进入此分支；判断条件为 short_hostname in FORBIDDEN_LAB_HOSTS。
    if short_hostname in FORBIDDEN_LAB_HOSTS:
        # 把实际主机、机器指纹与项目目录对照出租服务器登记值未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeIdentityError(
            # 报错明确当前机器位于禁止使用的实验室主机列表。
            f"Project compute is forbidden on laboratory host {short_hostname!r}"
        )
    # 按去掉域名后可与禁用主机列表比较的短名称决定是否进入此分支；判断条件为 short_hostname != expected_hostname。
    if short_hostname != expected_hostname:
        # 把实际主机、机器指纹与项目目录对照出租服务器登记值未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeIdentityError(
            # 身份拒绝提示说明当前只有登记的出租服务器获准执行项目计算。
            "Project compute is authorized only on the registered rented server: "
            # 列出实际主机名和获准出租服务器主机名。
            f"hostname={short_hostname!r}, expected={expected_hostname!r}"
        )
    # 忽略十六进制大小写比较机器指纹，防止连到同名但不同的实例。
    if machine_id_sha256.lower() != expected_machine_id_sha256.lower():
        # 把实际主机、机器指纹与项目目录对照出租服务器登记值未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeIdentityError(
            # 错误说明机器内容指纹与登记值不同，即使主机名相同也拒绝运行。
            "Rented-server machine identity mismatch: "
            # 列出本次读取到的机器身份摘要，便于识别连接错机器。
            f"sha256={machine_id_sha256!r}"
        )
    # 要求项目运行目录就是登记的远端主目录，路径别名先解析后再比对。
    if project_root.resolve() != expected_project_root.resolve():
        # 把实际主机、机器指纹与项目目录对照出租服务器登记值未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeIdentityError(
            # 错误说明项目执行目录必须是登记的远端权威代码位置。
            "Project compute must use the rented-server checkout: "
            # 同时列出实际项目目录和登记的远端权威目录。
            f"project_root={project_root.resolve()}, expected={expected_project_root.resolve()}"
        )


# 读取真实主机身份并阻止项目计算在未授权机器上执行；下面的参数分别限定输入数据和处理约束。
def assert_rented_server(
    # 保存代码、数据与检查点的项目目录。
    project_root: Path,
    # 从这里起的参数必须写明名称传入，减少路径或审核开关位置混淆。
    *,
    # 实际运行机器的主机名。
    hostname: Optional[str] = None,
    # 系统机器标识文件。
    machine_id_path: Path = Path("/etc/machine-id"),
# 本接口负责读取真实主机身份并阻止项目计算在未授权机器上执行，处理结果通过对象状态或文件保存，不额外返回数值。
) -> None:
    # 读取真实主机身份并阻止项目计算在未授权机器上执行。
    """Read the live host identity and fail unless it is the registered rental."""

    # 检查系统机器标识文件是否缺失；缺失时不能继续信任该输入。
    if not machine_id_path.is_file():
        # 读取真实主机身份并阻止项目计算在未授权机器上执行未通过，抛出下方包含具体路径或实际值的错误。
        raise RuntimeIdentityError(f"Machine identity file is missing: {machine_id_path}")
    # 系统机器标识文件的原始字节。
    machine_id = machine_id_path.read_bytes()
    # 将实际主机名、机器指纹和项目位置一起交给同一身份校验函数。
    validate_runtime_identity(
        # 实际运行机器的主机名。
        hostname=hostname or socket.gethostname(),
        # 运行机器标识文件的 SHA-256 指纹。
        machine_id_sha256=_sha256_bytes(machine_id),
        # 保存代码、数据与检查点的项目目录。
        project_root=project_root,
    )
