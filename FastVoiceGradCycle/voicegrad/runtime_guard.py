# 只允许在用户明确认可的实验室 speech GPU 计算节点上执行项目计算。  # 文件用途说明。
"""对 speech GPU 主机名和权威 NFS 项目目录执行失败关闭校验。"""  # 模块说明。

from __future__ import annotations  # 延后解析类型标注，兼容当前项目的 Python 版本。

import socket  # 读取当前服务器主机名，避免在登录节点误启动计算。
from pathlib import Path  # 规范化并核对权威 NFS 项目目录。
from typing import Collection, Optional  # 描述可注入的主机集合与兼容参数。


ALLOWED_SPEECH_HOSTS = frozenset(  # 只收录已经从实验室主机清单确认的可候选计算节点。
    {  # 使用集合可以直接做精确成员判断，不接受相似名称。
        "speech001",  # 第一组允许候选节点。
        "speech002",  # 第一组允许候选节点。
        "speech003",  # 第一组允许候选节点。
        "speech004",  # 第一组允许候选节点。
        "speech105",  # 避开永久禁用的 101～104 后的允许候选节点。
        "speech106",  # 允许候选节点。
        "speech107",  # 允许候选节点。
        "speech108",  # 允许候选节点。
        "speech109",  # 允许候选节点。
        "speech110",  # 允许候选节点。
        "speech201",  # 第二组允许候选节点。
        "speech202",  # 第二组允许候选节点。
        "speech203",  # 第二组允许候选节点。
        "speech204",  # 第二组允许候选节点。
        "speech205",  # 第二组允许候选节点。
        "speech206",  # 第二组允许候选节点。
        "speech301",  # 第三组允许候选节点。
    }  # 完成允许主机集合。
)  # 冻结集合，防止运行时被意外修改。
FORBIDDEN_SPEECH_HOSTS = frozenset(  # 这些主机即使 GPU 空闲也永久不能使用。
    {"speech101", "speech102", "speech103", "speech104"}  # 教授实验室规则中的四台禁用节点。
)  # 冻结禁用集合，避免代码在运行中放宽规则。
LOGIN_HOSTS = frozenset({"voiceexp-lab"})  # 登录/管理节点只准做文件管理，绝不运行 ML 计算。
EXPECTED_PROJECT_ROOT = Path(  # Cycle v1 完整恢复后的唯一权威项目目录。
    "/nfs/speechst2/storage/rich_925/FastVoiceGrad_cycle_v1_restore_20261001/root/FsatVoiceGrad"  # 精确 NFS 路径。
)  # 完成权威目录常量。


class RuntimeIdentityError(RuntimeError):  # 用独立异常类型区分环境错误和模型错误。
    """当前主机或项目目录不满足 speech GPU 运行规则时抛出。"""  # 异常语义说明。


def validate_runtime_identity(  # 对已经读取好的主机名和项目目录做纯校验，便于静态单元测试。
    *,  # 所有参数必须按名称传入，避免把路径与主机名写反。
    hostname: str,  # 当前服务器主机名，可以带域名后缀。
    project_root: Path,  # 本次命令使用的项目根目录。
    allowed_hostnames: Collection[str] = ALLOWED_SPEECH_HOSTS,  # 默认只允许已登记的 speech 计算节点。
    expected_project_root: Path = EXPECTED_PROJECT_ROOT,  # 默认要求使用完整恢复后的权威 NFS 副本。
    machine_id_sha256: Optional[str] = None,  # 保留旧调用接口；多台 speech 主机不共享一个 machine-id。
    expected_hostname: Optional[str] = None,  # 保留旧测试接口；传入时把允许集合收紧到该单台主机。
    expected_machine_id_sha256: Optional[str] = None,  # 仅兼容旧的单机测试，不用于正式多机门禁。
) -> None:  # 校验成功不返回数值，失败直接抛出异常。
    """只接受明确允许的 speech 主机和唯一的权威 NFS 项目目录。"""  # 函数行为说明。

    short_hostname = hostname.strip().split(".", 1)[0]  # 去掉域名后缀并清理两端空白。
    if short_hostname in LOGIN_HOSTS:  # 登录节点只允许 SSH、解压和文件管理。
        raise RuntimeIdentityError(  # 明确拒绝在登录节点启动任何项目计算。
            f"Project compute is forbidden on login host {short_hostname!r}"  # 错误包含实际主机便于排查。
        )  # 完成登录节点错误。
    if short_hostname in FORBIDDEN_SPEECH_HOSTS:  # 四台禁用节点优先给出最明确的错误。
        raise RuntimeIdentityError(  # 永久规则不能被空闲 GPU 或命令参数绕过。
            f"Project compute is permanently forbidden on host {short_hostname!r}"  # 列出违规主机。
        )  # 完成禁用主机错误。
    effective_allowed = (  # 生产环境使用整套允许列表，旧单机测试可显式收紧。
        frozenset({expected_hostname})  # 兼容旧测试时只接受其指定的单台主机。
        if expected_hostname is not None  # 仅显式传入旧参数时启用兼容路径。
        else frozenset(str(name) for name in allowed_hostnames)  # 正式路径复制为不可变字符串集合。
    )  # 完成实际允许主机集合。
    if short_hostname not in effective_allowed:  # 未登记节点一律失败，不能按 speech 前缀模糊放行。
        raise RuntimeIdentityError(  # 失败关闭可防止未来新增主机未经确认就被使用。
            f"Project compute requires an approved speech GPU host; got {short_hostname!r}"  # 报告实际值。
        )  # 完成未知主机错误。
    if expected_machine_id_sha256 is not None:  # 旧单机测试仍可要求机器指纹完全一致。
        if machine_id_sha256 is None or machine_id_sha256.lower() != expected_machine_id_sha256.lower():  # 忽略十六进制大小写。
            raise RuntimeIdentityError("Legacy single-host machine identity mismatch")  # 旧接口指纹不符时拒绝。
    actual_root = Path(project_root).expanduser().resolve()  # 消除用户目录、相对路径和符号链接。
    required_root = Path(expected_project_root).expanduser().resolve()  # 用同一种规则规范化期望路径。
    if actual_root != required_root:  # 只有完整恢复副本中的项目根目录可以执行计算。
        raise RuntimeIdentityError(  # 防止在旧副本、临时目录或下载目录运行。
            f"Project compute must use {required_root}; got {actual_root}"  # 同时给出期望值与实际值。
        )  # 完成项目路径错误。


def assert_speech_gpu_server(  # 读取现场主机名后调用统一校验逻辑。
    project_root: Path,  # 调用方解析得到的项目根目录。
    *,  # 可选主机名只能按名称传入，主要用于测试。
    hostname: Optional[str] = None,  # 省略时读取当前操作系统主机名。
) -> None:  # 通过时静默返回，失败时抛出 RuntimeIdentityError。
    """确认当前进程位于获准 speech GPU 节点和权威 NFS 项目中。"""  # 函数行为说明。

    validate_runtime_identity(  # 把现场事实交给唯一校验入口。
        hostname=hostname or socket.gethostname(),  # 默认读取当前服务器真实主机名。
        project_root=project_root,  # 核对调用方所用项目根目录。
    )  # 完成现场身份校验。


def assert_rented_server(  # 保留旧函数名，避免大量训练和评估入口同时失效。
    project_root: Path,  # 原调用方传入的项目根目录。
    *,  # 可选主机名继续采用关键字方式。
    hostname: Optional[str] = None,  # 测试可注入主机名，生产环境留空。
    machine_id_path: Optional[Path] = None,  # 兼容旧签名；多 speech 主机不再核对单一 machine-id。
) -> None:  # 行为已改为 speech GPU 门禁，返回约定保持不变。
    """兼容旧入口名称，并转交新的 speech GPU 服务器门禁。"""  # 兼容层说明。

    del machine_id_path  # 明确表明旧单机指纹路径不再参与多节点身份判断。
    assert_speech_gpu_server(project_root, hostname=hostname)  # 统一执行新的失败关闭规则。
