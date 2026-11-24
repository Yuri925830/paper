# 本文件集中保存本组实验设置，训练、推理、画图和评估都读取这里。
from pathlib import Path  # 用明确路径定位baseline资产与本组独立输出。

ARM = 'cycle_post'  # 当前文件夹只负责这一组实验，避免启动时选错分支。
LAMBDA_CYCLE = 0.0125  # Base不加入循环梯度，Cycle沿用已完成实验的0.0125权重。
PROJECT_ROOT = Path('/mnt/lab-data/임서/FastVoiceGrad/project')  # baseline模型、特征、数据和原评分器所在的项目目录。
RUN_ROOT = Path('/mnt/lab-data/임서/FastVoiceGrad/runs/refactored_20261104')  # 整理版使用独立输出根目录，避免写入已完成的旧实验。
LEGACY_RUN_ROOT = Path('/mnt/lab-data/임서/FastVoiceGrad/runs/fastvoicegrad')  # 旧结果只作为历史来源，训练入口禁止向这里写入。
SEED = 1234  # 本轮只使用原实验的随机种子，不自动增加重复实验。
START_STEP = 122900  # 两组都从baseline已经完成的全局更新步开始。
END_STEP = 184350  # 两组追加相同的61450次更新，到第150个pass结束。
UPDATES_PER_PASS = 1229  # 每轮39335条保留语音按batch32丢弃尾批后的更新次数。
BATCH_SIZE = 32  # 每次优化更新实际对应的样本总数。
BASE_MICROBATCH_SIZE = 8  # 基础训练每次装入显存的样本数，累积四次组成batch32。
CYCLE_SAMPLES = 24  # Cycle分支从每个batch中取24条源语音。
CYCLE_MICROBATCH_SIZE = 3  # 循环重建每次处理3条，共累积八次。
TRAIN_FRAMES = 128  # 训练时每条语音截取的mel帧数。
N_MELS = 80  # 每个时间帧的mel频率通道数。
SAMPLE_RATE = 22050  # 训练及生成语音的采样率，单位为每秒采样点。
HOP_SIZE = 256  # 相邻mel帧在波形上相隔的采样点数。
STUDENT_START_STEP = 950  # 每程单步FastVoiceGrad从扩散时刻950开始。
STUDENT_LR = 5e-5  # baseline断点中学生Adam的真实学习率。
DISCRIMINATOR_LR = 2e-4  # baseline判别器Adam的真实学习率。
ADAM_BETAS = (0.5, 0.9)  # Adam一阶和二阶动量的衰减系数。
ADAM_EPS = 1e-8  # Adam分母中防止除零的稳定常数。
LAMBDA_FM = 2.0  # 原特征匹配损失的权重。
LAMBDA_DIST = 45.0  # 原教师蒸馏损失的权重。
CYCLE_PAIR_SEED = 80001234  # 训练中选择异说话人目标的独立随机流起点。
CYCLE_FORWARD_SEED = 90001234  # 训练中A到B转换噪声的独立随机流起点。
CYCLE_BACKWARD_SEED = 100001234  # 训练中B到A返回噪声的独立随机流起点。
INFERENCE_FORWARD_SEED = 90000000  # 完整句子推理时第一程按配对编号派生的噪声起点。
INFERENCE_BACKWARD_SEED = 100000000  # 完整句子推理时返回程独立噪声的起点。
DEVELOPMENT_EXAMPLES = 52  # 从训练梯度中固定排除的开发语音条数。
TEST_EXAMPLES = 100  # 正式测试包含十位说话人各十条语音。
RETAINED_TRAIN_EXAMPLES = 39335  # 排除开发语音后允许参与训练的样本数。
DEVELOPMENT_PAIRS = 90  # 固定开发转换配对数量，与52条损失诊断语音分开。
TEST_PAIRS = 8100  # 正式测试的全部源语音与目标参考配对数量。
CONVERSION_EVERY_PASSES = 5  # 每五个pass对固定开发配对进行转换与评分。
SAVE_EVERY_UPDATES = 2000  # 达到这个新增更新数时保存完整训练断点。
SAVE_EVERY_SECONDS = 1800  # 距离上次保存满三十分钟时提交完整断点。
TIMING_PAIRS = 32  # 端到端速度固定测量清单中的前32对。
TIMING_WARMUPS = 3  # 正式计时前固定进行三次预热。
METRIC_BATCH_SIZE = 4  # 原UTMOS、ASR和ECAPA评分接口的批大小。
DNSMOS_THREADS = 2  # P808评分器使用的线程数。
SVA_THRESHOLD = 0.25  # ECAPA余弦严格大于该值时才算验证通过。
TORCH_NUM_THREADS = 2  # 保留原实验的PyTorchCPU线程数。
MINIMUM_FREE_BYTES = 4 * 1024**3  # 保存大型完整断点前要求至少四GiB剩余空间。
BNF_SHA256 = '94362c99f408cbe891785537e222c2f1f9119edb11b95107b9faa4c0f0d679a5'  # 与原离线BNF对应的固定Conformer权重内容指纹。
BNF_CONFIG_SHA256 = 'd8d3df407a9b88500cf62e32a8bdef78068b3a09e9f8bc47904843bef1fff6c4'  # 与固定Conformer权重配套的结构和前端配置指纹。
LEGACY_PARENT_SHA256 = '2ca44269eefac2dd3f8ffa7afe0337f1f3684469aec9fadeff623237cf37b830'  # 学生12号完整pass100父断点的固定内容指纹。
DEVELOPMENT_SHA256 = '08737551d34e00ba0b80a5e34adbf73fcfc6c70ece6bc8dc4a6c466e9b87afb1'  # 固定开发排除协议的原始内容指纹。
DEVELOPMENT_MANIFEST_SHA256 = '4b530a15caa2804f817437870e91a0da1fed843f0f9ec0666f8f0bc1e8064cb3'  # 固定开发划分完整发布收据的内容指纹。
PARENT_KIND = 'acdd_mc1_longtrain_from3327_full_v1'  # baseline完整断点的原格式名称，只允许它使用已登记的旧CUDA随机状态兼容。
KIND = 'fastvoicegrad_post_refactor_v1'  # 整理版新运行身份，与生成旧结果的原代码明确区分。
BNF_STFT_COMPAT_POLICY = 'module_local_torch_proxy_stft_return_complex_false_preserves_legacy_real_imag_last_axis'  # 只在旧Conformer局部保留原实部虚部表示，不修改全局PyTorch。

CONTENT_SAMPLE_RATE = 16000  # 内容编码器输入波形的采样率，与原实验一致。
N_FFT = 1024  # mel前端每次傅里叶变换使用的采样点数。
WIN_SIZE = 1024  # mel前端每个分析窗的采样点数。
F_MIN = 0  # mel滤波器覆盖的最低频率，单位为赫兹。
F_MAX = 8000  # mel滤波器覆盖的最高频率，单位为赫兹。
BNF_ALIGNMENT = "legacy_length_linear"  # 保留baseline训练时的BNF长度线性对齐方式。
PER_UPDATE_SEED_OFFSET = 201234  # 只为旧baseline父断点的CUDA状态布局兼容派生替代种子。

TOOLS = PROJECT_ROOT / "exp/diagnostics/student_performance_tools_20260906"  # baseline原模型构建与批处理工具所在的位置。
DEVELOPMENT = PROJECT_ROOT / "exp/diagnostics/student_development_20260906/development_contract.json"  # 固定52条开发排除协议。
DEVELOPMENT_MANIFEST = DEVELOPMENT.parent / "development_manifest.json"  # 固定划分及其配套文件的发布收据。
DATA_RESTORE_RECEIPT = PROJECT_ROOT / ".essential_data_restore_identity.json"  # 原必要数据恢复时保存的来源与完整性收据。
POST_PARENT_PATH = PROJECT_ROOT / "exp/diagnostics/student_development_20260906/acdd_mc1_longtrain_from3327_v1/latest_full.pt"  # 两组共用的完整baseline父断点。
LOSS_KEYS = (  # 按固定顺序记录逐批损失，未计算的部分不伪造数据。
    "student_total",  # 学生本次更新实际优化的加权总损失。
    "student_adv",  # 学生对抗判别器的损失。
    "fm_raw",  # 未乘权重的特征匹配损失。
    "fm_weighted",  # 乘以特征匹配权重后的损失。
    "dist_raw",  # 未乘权重的教师蒸馏损失。
    "dist_weighted",  # 乘以教师蒸馏权重后的损失。
    "cycle_raw",  # 未乘权重的循环重建MAE。
    "cycle_weighted",  # 实际加入学生梯度的加权循环损失。
    "discriminator",  # 判别器本次更新使用的损失。
)  # 完成原正式训练的九项字段定义。
RUN_DIR = RUN_ROOT / f"{ARM}_seed_{SEED}_lambda_{format(LAMBDA_CYCLE, '.8g').replace('.', 'p')}"  # 当前文件夹对应的独立新运行目录。

def shared_settings():  # 提取两组应当完全一致的实验设置，用于运行前核对公平性。
    excluded = {"ARM", "LAMBDA_CYCLE", "RUN_DIR", "LEGACY_RUN_ROOT"}  # 只排除两组本来就不同的标签、Cycle权重与派生路径。
    result = {}  # 为两组共同设置建立一个可直接保存的普通字典。
    for name, value in globals().items():  # 逐项检查本文件中集中定义的设置。
        if name.isupper() and name not in excluded:  # 只保留两组都必须一致的设置名称。
            if isinstance(value, Path):  # JSON没有路径类型，保存前先转换为明确的文本。
                result[name] = str(value)  # 保留路径指向的位置。
            elif isinstance(value, tuple):  # JSON读回数组时使用列表，需要提前统一类型。
                result[name] = list(value)  # 避免断点恢复时因元组与列表不同误判配置改变。
            else:  # 普通数字、字符串等设置已经可以直接保存。
                result[name] = value  # 保留这些设置的原始数值或含义。
    return result  # 交给训练入口核对两组共同起点和实验条件。
