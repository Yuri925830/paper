# 汇总教师参数，并在字段旁区分论文明确数值与本次补足的实现选择。
"""Paper-aligned constants for the FastVoiceGrad VoiceGrad-30 teacher."""

# =============================================================================
# 这个文件是干什么的？
# =============================================================================
# 它是整个工程的“总参数表”，自己不训练模型，也不读取音频。
# train.py、infer_voicegrad30.py、validate_setup.py 都从这里读取同一套数字，
# 避免出现“训练用 1000 步、推理却误写成 20 步”这种很难发现的错误。
#
# 对 Python 零基础读者的语法说明：
# 1. name: int = 1000 中，冒号后的 int 是类型提示，等号右边才是默认值。
# 2. @dataclass 会自动生成 __init__ 等重复代码，可以把配置写得像一张表。
# 3. frozen=True 表示实例创建后不能随手改字段，防止训练中途误改超参数。
# 4. Path 是“路径对象”；Path("a") / "b" 表示拼成 a/b，不是做数学除法。

# --------------------------- 导入包逐项解释 ---------------------------
# os 也是标准库。这里仅用它读取环境变量，不会执行系统命令。
import os
# dataclasses 是 Python 标准库：dataclass 自动生成配置类初始化代码；asdict 把配置
# 对象转换成普通字典。它们不需要 pip install。
from dataclasses import asdict, dataclass
# pathlib 也是标准库。Path 比手写 "/" 拼字符串更安全、可读，并能检查文件存在性。
from pathlib import Path
# typing 只提供类型提示。Dict 表示字典，不会改变程序运行结果。
from typing import Dict


# 公开仓库不能把某一台实验室服务器的私人绝对路径写死。
# 如果设置了 VOICEGRAD_PROJECT_ROOT，就使用该环境变量；否则把“本文件所在目录”
# 当作项目根目录。Path(__file__).resolve() 得到 config.py 的绝对路径，parent 再取
# 它的父目录。这样把仓库复制到任何 Linux/Windows 机器都不需要改源码。
PROJECT_ROOT = Path(
    # 优先读取显式工程根目录；没有环境变量时回到本文件向上两级的仓库目录。
    os.environ.get("VOICEGRAD_PROJECT_ROOT", Path(__file__).resolve().parents[1])
# 展开用户目录并转换为绝对路径，后续训练与推理据此寻找同一套数据。
).expanduser().resolve()

# 旧版文件曾从 config 导入 SERVER_PROJECT_ROOT。保留这个别名可以兼容旧命令，
# 但它现在只是可移植的 PROJECT_ROOT，已经不包含任何私人服务器信息。
SERVER_PROJECT_ROOT = PROJECT_ROOT


# 把下面的配置字段封装为数据对象，并按声明限制意外修改。
@dataclass(frozen=True)
# 保存教师实验的声学、网络、扩散和数据划分参数。
class VoiceGrad30Config:
    # 教师配置表同时保留论文实验数值和已说明来源的补充超参数。
    """Settings stated by FastVoiceGrad for the VCTK teacher."""

    # -------------------------------------------------------------------------
    # 第一组：声学特征参数，对应论文 4.1 节 Experimental settings
    # -------------------------------------------------------------------------
    # sample_rate=22050：每秒声音保存 22050 个采样点，也写作 22.05 kHz。
    sample_rate: int = 22050
    # n_fft=1024：做短时傅里叶变换时，每个分析块进行 1024 点 FFT。
    n_fft: int = 1024
    # hop_size=256：相邻频谱帧之间向前移动 256 个波形采样点。
    hop_size: int = 256
    # win_size=1024：每次分析观察 1024 个采样点，论文明确给出。
    win_size: int = 1024
    # 原始频谱最终压缩为 80 个 mel 频带，因此 mel 形状是 [80, 时间帧数]。
    n_mels: int = 80
    # BNF 是尽量去除说话人身份、保留音素内容的 bottleneck feature，维数 144。
    n_bnf: int = 144
    # GE2E speaker encoder 把一段参考语音压成 256 维“这个人是谁”的向量。
    spk_emb_dim: int = 256

    # -------------------------------------------------------------------------
    # 第二组：VoiceGrad 噪声预测 U-Net 参数
    # -------------------------------------------------------------------------
    # 论文规定 12 层一维卷积的隐藏通道数为 512。
    hidden_channels: int = 512
    # timestep 和 speaker embedding 分别编码成 128 维，之后沿通道拼接。
    # 注意：论文没有公布这个中间维数；128 是为了让未公开细节固定、可复现的工程选择。
    cond_dim: int = 128
    # 每个卷积块把 144 维 BNF 先投影成 32 维，减少计算和显存。
    bnf_channels: int = 32

    # -------------------------------------------------------------------------
    # 第三组：DDPM 扩散和 VoiceGrad-30 推理参数
    # -------------------------------------------------------------------------
    # 完整训练扩散链 T=1000。注意：扩散步数不是 epoch 数，两者完全不同。
    diffusion_steps: int = 1000
    # cosine noise schedule 使用的小偏移量，避免时间起点附近的数值问题。
    cosine_offset: float = 0.008
    # VoiceGrad-30 名字中的“30”：推理只选择 30 个反向去噪时间步。
    teacher_sampling_steps: int = 30
    # 论文要求最高选择步 S_K < T，但没有公布 VoiceGrad-30 的完整整数序列。
    # 950 是论文唯一明确出现的相关起点，所以作为可重复的默认假设；推理命令仍
    # 可以用 --start-step 修改。写实验报告时必须记录实际使用的 start_step。
    # FastVoiceGrad states S_K < T but does not publish the exact VoiceGrad-30
    # subsequence. S_K=950 is the only explicit starting step reported in the
    # paper (for its source-diffusion experiments), so it is the reproducible
    # default here and remains configurable at inference time.
    teacher_start_step: int = 950

    # -------------------------------------------------------------------------
    # 第四组：训练设置，对应论文教师模型的 optimizer 描述
    # -------------------------------------------------------------------------
    # 长语音随机裁成 128 个 mel 帧训练，使 batch 内长度统一并控制显存。
    # 论文没有给出裁剪长度；这里沿用参考 DiffVC 实现的 128 帧，属于本次复现补充的选择。
    train_frames: int = 128
    # 论文教师训练 batch size=32。
    batch_size: int = 32
    # 2e-4 是科学计数法，等于 2×10^-4，也就是 0.0002。
    learning_rate: float = 2e-4
    # Adam 的两个指数移动平均系数，论文明确设为 (0.9, 0.999)。
    adam_beta1: float = 0.9
    # beta2=0.999 负责平滑梯度平方；它和上一行 beta1 一起原样交给 Adam。
    adam_beta2: float = 0.999
    # 论文原实验只训练 500 epoch，必须单独保留，不能冒充 1000。
    paper_epochs: int = 500
    # 主复现实验严格停在论文给出的 500 epoch；不再把 1000-epoch 扩展实验设为默认。
    epochs: int = 500
    # 伪随机种子：固定裁剪位置、噪声抽样等随机过程，尽量让实验可重复。
    seed: int = 1234

    # -------------------------------------------------------------------------
    # 第五组：论文 VCTK 数据划分
    # -------------------------------------------------------------------------
    # 官方 VCTK 0.92 和论文均为 110 人。p315 虽缺 transcript，但其音频必须保留。
    expected_total_speakers: int = 110
    # 其中 100 位说话人的约 390 句用于训练；论文没有额外 validation split。
    expected_train_speakers: int = 100
    # 剩下 10 位只用于测试，训练时绝不能看见，用来检验 any-to-any 未见说话人转换。
    expected_test_speakers: int = 10
    # 每个未见测试说话人取 10 句，10 人总共 100 条 test utterance。
    test_utterances_per_speaker: int = 10

    # def 表示定义成员函数；self 指当前这一个配置对象，箭头后是返回值的类型提示。
    def as_dict(self) -> Dict[str, object]:
        # asdict 是 dataclasses 自带函数，把配置对象变成普通 Python 字典。
        # checkpoint 能直接保存字典，之后就能检查权重到底对应哪套配置。
        # Dict[str, object] 表示：键是字符串，值可以是整数、浮点数等对象。
        return asdict(self)


# 这是全工程真正使用的唯一配置实例。其他文件写
# from voicegrad.config import PAPER_CONFIG 后，就能读取 PAPER_CONFIG.learning_rate。
PAPER_CONFIG = VoiceGrad30Config()
