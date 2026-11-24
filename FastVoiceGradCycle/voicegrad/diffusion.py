# 实现教师前向加噪与稀疏反向转换，扩散步编号统一从 1 开始。
"""Forward diffusion and the paper's shortened VoiceGrad reverse process."""

# =============================================================================
# 这个文件在整个系统里的位置
# =============================================================================
# 一句话理解扩散模型：
#   训练时，我们故意给正确 mel 加一份“自己知道答案”的高斯噪声，让 model.py
#   里的网络猜这份噪声；推理时，再反复使用网络预测噪声并把它去掉。
#
# 本文件对应 FastVoiceGrad 论文第 2 节 Preliminary: VoiceGrad：
#   q_sample() 对应公式 (3) 的前向加噪；
#   sample() 对应 Algorithm 1 的反向转换；
#   sampling_sequence() 负责从完整 T=1000 中选择 VoiceGrad-30 的 K=30 步。
#
# 四个最重要的数学量：
#   beta_t：第 t 步新加入多少噪声；
#   alpha_t = 1-beta_t：第 t 步保留多少原信号；
#   alpha_bar_t = alpha_1×...×alpha_t：到第 t 步累计还剩多少原信号；
#   epsilon：标准高斯噪声 N(0,I)，形状和 mel 完全一样。

# --------------------------- 导入包逐项解释 ---------------------------
# math 提供圆周率 pi，用于构造 cosine noise schedule。
import math
# List/Optional 是类型提示：List[int] 是整数列表；Optional[X] 表示可以是 X 或 None。
from typing import List, Optional

# torch 提供 Tensor、cos/sqrt/randn 等张量数学运算。
import torch
# nn.Module 让扩散 schedule 可以 register_buffer，并随模型一起搬到 GPU。
import torch.nn as nn


# 这个类不包含可学习卷积权重；它保存扩散系数表，并实现训练加噪和推理去噪规则。
class VoiceGradDiffusion(nn.Module):
    # 余弦扩散系数表保留第 0 步干净端点，训练采样使用第 1 至 1000 步。
    """Cosine DDPM with explicit paper-style 1-based timesteps."""

    # 构造函数接收总扩散步数 T 和 cosine schedule 偏移量，并一次性算好全部系数。
    def __init__(self, n_levels: int = 1000, offset: float = 0.008) -> None:
        # nn.Module 子类必须先调用父类初始化，PyTorch 才能正确登记 buffer/参数。
        super().__init__()
        # 总步数必须至少为 1；0 或负数既无法建立时间网格，也没有扩散含义。
        if n_levels < 1:
            # ValueError 表示调用者给的参数值不合法，并在真正做除法前就终止。
            raise ValueError("n_levels must be positive.")
        # 保存 T，后面检查 timestep 范围和采样起点时都会用到。
        self.n_levels = n_levels
        # 保存 cosine 曲线的小偏移，便于对象明确记录实际使用的 schedule。
        self.offset = offset

        # arange(n_levels+1) 产生 0,1,...,1000；再除以 1000 得到 [0,1] 网格。
        # 使用 float64 只为更精确地预计算 schedule，最后会转回 float32。
        grid = torch.arange(n_levels + 1, dtype=torch.float64) / n_levels
        # cosine schedule：按 Improved DDPM/VoiceGrad 的余弦曲线定义累计信号量。
        # math.pi 是 π；.square() 就是把 cos 的结果平方。
        cosine = torch.cos(
            # 把归一化时间加小偏移后映射到四分之一余弦周期。
            ((grid + offset) / (1.0 + offset)) * math.pi / 2.0
        # 余弦平方给出信号功率随扩散时间减弱的曲线。
        ).square()
        # 除以第 0 项，强制 alpha_bar_0=1，代表 x_0 是完全干净的 mel。
        nominal_alpha_bar = cosine / cosine[0]

        # 为了让数组索引和论文 timestep 一致，位置 0 专门留给干净状态。
        # 实际扩散 beta 从索引 1 到 1000。
        betas = torch.zeros(n_levels + 1, dtype=torch.float64)
        # 由相邻累计量反推出单步 beta_t = 1-alpha_bar_t/alpha_bar_(t-1)。
        betas[1:] = 1.0 - nominal_alpha_bar[1:] / nominal_alpha_bar[:-1]
        # 最后几步可能非常接近 1，限制为 0.999 避免除以 0 和数值爆炸。
        # clamp_ 末尾下划线表示直接修改原 Tensor，不另外创建副本。
        betas.clamp_(min=0.0, max=0.999)
        # alpha_t 表示这一小步保留下来的信号比例。
        alphas = 1.0 - betas

        # cumprod 是 cumulative product（累计乘积），得到真实 alpha_bar_t。
        alpha_bar = torch.ones(n_levels + 1, dtype=torch.float64)
        # 索引 0 保持 1；索引 1..T 写入 alpha_1×...×alpha_t 的累计乘积。
        alpha_bar[1:] = torch.cumprod(alphas[1:], dim=0)

        # register_buffer 用于“不学习、但要随模块移动到 GPU 和写入 state_dict”的量。
        # 这些 schedule 系数不是神经网络参数，不能交给 Adam 更新。
        # 平方根也提前算好，避免每个 batch 重复计算完全相同的东西。
        self.register_buffer("betas", betas.float())
        # 保存每一步 alpha_t=1-beta_t；float() 把预计算 float64 转成训练常用 float32。
        self.register_buffer("alphas", alphas.float())
        # 保存累计信号比例 alpha_bar_t，反向跳步采样会按 timestep 直接索引它。
        self.register_buffer("alphas_cumprod", alpha_bar.float())
        # 保存 sqrt(alpha_bar_t)，训练 q_sample 时乘在干净 mel x_0 前面。
        self.register_buffer("sqrt_alphas_cumprod", alpha_bar.sqrt().float())
        # 保存 sqrt(1-alpha_bar_t)，训练 q_sample 时乘在标准高斯噪声前面。
        self.register_buffer(
            # 登记每一步噪声幅度 sqrt(1−alpha_bar) 的 float32 缓冲表，q_sample 按时间步查询。
            "sqrt_one_minus_alphas_cumprod", (1.0 - alpha_bar).sqrt().float()
        )

    # 该辅助方法只使用传入的数据，不依赖网络实例内部状态。
    @staticmethod
    # 按 batch 内每条语音自己的 t 取系数，并改形成能与 mel 自动广播的形状。
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape) -> torch.Tensor:
        # 例子：batch 有 3 条语音，timesteps=[5,200,900]，就从 schedule 表中
        # 分别取第 5、200、900 项。gather 的索引必须是 torch.long 整数类型。
        if timesteps.dtype != torch.long:
            # 若调用者传了其他类型，就显式转为 long，以满足 gather 的索引规则。
            timesteps = timesteps.long()
        # 从一维 schedule 表一次取出 batch 对应的 B 个系数，得到 [B]。
        selected = values.gather(0, timesteps)
        # selected 原来是 [B]。reshape 为 [B,1,1] 后，乘 [B,80,T] mel 时，
        # PyTorch broadcasting（广播）会自动把系数应用到全部 mel 通道和时间帧。
        # *((1,) * n) 是 Python 参数解包：生成 n 个 1 并作为多个维度传给 reshape。
        return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    # 统一检查训练用时间步，避免每个调用位置重复写边界判断。
    def _validate_timesteps(self, timesteps: torch.Tensor) -> None:
        # 这里主动检查，是为了尽早发现最常见的错误：把 0-based t 传进 1-based 实现。
        if timesteps.ndim != 1:
            # 每条样本只需一个整数 t，因此合法形状必须是一维 [B]。
            raise ValueError("timesteps must be a one-dimensional tensor.")
        # torch.any 检查 batch 中是否至少有一个小于 1 或大于 T 的非法值。
        if torch.any(timesteps < 1) or torch.any(timesteps > self.n_levels):
            # 越界时停止，避免非法索引 schedule 或把 t=0 错当成训练噪声步。
            raise ValueError(f"timesteps must be in [1, {self.n_levels}].")

    # q_sample 是训练阶段的前向扩散：用闭式公式从干净 x_0 一次跳到任意 x_t。
    def q_sample(
        # 当前数据集或网络对象。
        self,
        # 加噪前的干净标准化 mel。
        x_start: torch.Tensor,
        # 每条样本选中的 1 到 1000 扩散时间步。
        timesteps: torch.Tensor,
        # 与 mel 逐元素对应的标准高斯噪声。
        noise: Optional[torch.Tensor] = None,
    # 返回按累计信号比例给干净 mel 添加已知高斯噪声得到的张量；类型标注帮助调用方理解结果，不改变数值。
    ) -> torch.Tensor:
        # 按累计信号比例给干净 mel 添加已知高斯噪声。
        """Sample x_t = sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon."""
        # 论文公式 (3)：不需要真的循环加噪 t 次，可以从 x_0 一次跳到任意 x_t。
        # x_start/noise/返回值的形状均是 [B,80,T_frames]。
        self._validate_timesteps(timesteps)
        # noise=None 表示调用者没指定噪声，此时函数自己抽一份标准高斯噪声。
        if noise is None:
            # randn_like 生成形状、dtype、CPU/GPU 位置均和 x_start 相同的 N(0,1)。
            noise = torch.randn_like(x_start)
        # 外部传入固定 noise 时，它必须与 mel 每个元素一一对应，形状不能不同。
        if noise.shape != x_start.shape:
            # 主动报清楚的错误，防止错误广播产生看似能运行、其实监督错位的结果。
            raise ValueError("noise and x_start must have identical shapes.")
        # 按每条样本的 t 取得 sqrt(alpha_bar_t)，并改形成 [B,1,1]。
        signal = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        # signal 是干净语音保留比例；noise_scale 是噪声比例。
        noise_scale = self._extract(
            # 按每条样本的 t 取累计噪声幅度，并扩成可与 mel 相乘的形状。
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        # 训练时 noise 就是监督答案，网络要根据生成的 x_t 把它猜出来。
        return signal * x_start + noise_scale * noise

    # 只生成 S_1<...<S_K 的整数稀疏序列；VoiceGrad-30 就是 K=30。
    def sampling_sequence(self, num_steps: int = 30, start_step: int = 950) -> List[int]:
        # 创建实际推理使用的稀疏反向时间步序列。
        """Return the ascending {S_1,...,S_K} integer subsequence."""
        # num_steps=30 对应 VoiceGrad-30。start_step 必须严格小于训练总步数 T。
        if not 1 <= num_steps <= start_step:
            # K 至少是 1 且不能超过可选整数范围，否则不能得到 K 个不同时间步。
            raise ValueError("num_steps must be in [1, start_step].")
        # 论文要求 S_K<T；这里也同时拒绝 0 和负起点。
        if not 1 <= start_step < self.n_levels:
            # 错误信息把本配置允许的最大值 T-1 直接告诉使用者。
            raise ValueError(
                # 错误说明采样起点要低于扩散总步数，并列出合法范围。
                f"The paper requires S_K < T; start_step must be in [1, {self.n_levels - 1}]."
            )
        # K=1 has exactly one selected timestep, and that sole timestep is S_K.
        # torch.linspace(1, start_step, 1) returns 1, which would silently turn
        # FastVoiceGrad/VoiceGrad-1 with S_K=950 into a t=1 denoising run.
        # 单步采样只保留指定起点，单独处理以免等距序列重复或遗漏端点。
        if num_steps == 1:
            # 单步采样只保留起点，后续一步直接到干净端点。
            return [start_step]
        # linspace 在 1 到 950 间等间隔取 30 个数（包含首尾）；round 取最近整数；
        # long 转整数 Tensor；tolist 最后变成普通 Python list，便于循环。
        sequence = torch.linspace(1, start_step, num_steps).round().long().tolist()
        # set 会去重；长度若减少，说明舍入让两个选择步撞成同一个整数。
        if len(set(sequence)) != num_steps:
            # 重复时间步会破坏“K 步”定义，因此不默默接受，而是要求修改参数。
            raise RuntimeError("Rounded sampling timesteps are not unique.")
        # 返回升序列表；sample 中会再反转成从大 timestep 走向 0 的顺序。
        return sequence

    # 关闭这段评估或采样的梯度记录，避免把诊断过程当成训练更新。
    @torch.no_grad()
    # 这个函数实现论文 Algorithm 1 的稀疏反向转换；装饰器已让整段不保存梯度。
    def sample(
        # 当前数据集或网络对象。
        self,
        # 预测 mel 噪声的 VoiceGrad 网络。
        model: nn.Module,
        # 源语音的标准化 mel。
        x_source: torch.Tensor,
        # 256 维目标说话人条件。
        spk_emb: torch.Tensor,
        # 表示语音内容的 144 维 BNF 特征。
        bnf: torch.Tensor,
        # 实际执行的反向采样步数。
        num_steps: int = 30,
        # 开始反向转换的扩散时间步。
        start_step: int = 950,
        # 单独控制采样噪声的随机数生成器。
        generator: Optional[torch.Generator] = None,
        # 是否按论文公式先将源 mel 加噪。
        diffuse_source: bool = False,
    # 返回用目标说话人和源 BNF 条件逐步转换源 mel得到的张量；类型标注帮助调用方理解结果，不改变数值。
    ) -> torch.Tensor:
        # 用目标说话人和源 BNF 条件逐步转换源 mel。
        """Run the sparse VoiceGrad reverse process from clean or diffused source mel.

        The transition from selected step t to the next lower selected step s
        redefines alpha as alpha_bar_t / alpha_bar_s. The final transition uses
        s=0 and zero Gaussian noise, as required by the DDPM posterior.
        """
        # @torch.no_grad() 表示整个函数不建立反向传播图：推理不训练，能节省显存。
        # 三个条件各有明确职责：x_source 保留源语音结构，bnf 提供源音素内容，
        # spk_emb 告诉网络输出应该像哪个目标说话人。
        if x_source.shape[-1] != bnf.shape[-1]:
            # 内容特征若与源 mel 帧数不同，就无法逐帧条件化，必须先修正预处理结果。
            raise ValueError("Source mel and BNF must have identical frame counts.")
        # 建立本次实际使用的 K 个时间步；默认是 30 个且最高为 950。
        sequence = self.sampling_sequence(num_steps, start_step)
        # batch 大小取自 mel 第 0 维，用于每轮建立 [B] timestep 向量。
        batch_size = x_source.shape[0]

        # FastVoiceGrad 第 3.1 节公式 (9) 提出用 x_src_SK 代替干净 x_src_0，
        # 并明确说后续实验采用 x_src_SK、S_K=950。diffuse_source=True 就执行该公式：
        # x_src_SK = sqrt(alpha_bar_SK)*x_src_0 + sqrt(1-alpha_bar_SK)*epsilon。
        if diffuse_source:
            # 每个 batch 样本都从同一个最高时间步 S_K=start_step 开始。
            start_timesteps = torch.full(
                # 为批内每条语音建立相同起点编号，整数张量用于扩散表索引。
                (batch_size,), start_step, device=x_source.device, dtype=torch.long
            )
            # 使用调用方给定的 generator 抽 epsilon，使同一 seed 可以复现实验结果。
            source_noise = torch.randn(
                # 起点噪声与源 mel 的 batch、频带和时间维完全一致。
                x_source.shape,
                # 张量执行计算的设备。
                device=x_source.device,
                # 张量的数值类型。
                dtype=x_source.dtype,
                # 单独控制采样噪声的随机数生成器。
                generator=generator,
            )
            # q_sample 是公式 (3)/(9) 的闭式前向扩散，不需要真的循环加噪 950 次。
            x = self.q_sample(x_source, start_timesteps, noise=source_noise)
        # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
        else:
            # 原始 VoiceGrad Algorithm 1 使用干净 source；保留该分支只用于对照实验。
            # clone 创建副本，防止下面的更新意外改变调用者手里的原 source mel。
            x = x_source.clone()

        # 扩散训练方向是 0 -> 大 t（越来越吵）；去噪方向必须反过来大 t -> 0。
        descending = list(reversed(sequence))
        # enumerate 同时给出循环编号和当前 t，编号用来找到列表中的下一个较小时间步。
        for index, current_step in enumerate(descending):
            # Python 条件表达式：如果还有下一项就取它；已经到末尾则目标是干净状态 0。
            previous_step = descending[index + 1] if index + 1 < len(descending) else 0
            # torch.full 创建 [B] timestep，让 batch 中每条语音本轮都使用 current_step。
            timesteps = torch.full(
                # 把当前反向时间步复制给批内各语音，送入网络位置编码。
                (batch_size,), current_step, device=x.device, dtype=torch.long
            )
            # model.py 输出 epsilon_theta(x_t,t,speaker,BNF)，形状与 x 完全相同。
            predicted_noise = model(x, timesteps, spk_emb, bnf)

            # 只走稀疏选中步时，从 t 跳到 s 的单步 alpha 要重新计算。
            alpha_bar_t = self.alphas_cumprod[current_step].to(dtype=x.dtype)
            # 取得目标较小时间步 s 的累计 alpha；最后 s=0 时该值按定义为 1。
            alpha_bar_s = self.alphas_cumprod[previous_step].to(dtype=x.dtype)
            # 两个累计量相除得到从 s 前进到 t 时保留信号的相对比例。
            alpha_t_given_s = alpha_bar_t / alpha_bar_s
            # 跳步后的噪声比例仍满足 beta=1-alpha。
            beta_t_given_s = 1.0 - alpha_t_given_s

            # 对应论文反向过程的均值：从当前 x 中按系数减掉预测噪声。
            mean = (
                # 从当前带噪 mel 开始构造跨步反向均值。
                x
                # 按该稀疏跳步的新增噪声比例扣除预测噪声。
                - beta_t_given_s
                # 用当前累计噪声标准差换算噪声预测的尺度。
                / torch.sqrt(1.0 - alpha_bar_t)
                # 把网络估计的噪声方向加入反向均值公式。
                * predicted_noise
            # 除以本次有效信号幅度，将状态尺度恢复到下一所选时间步。
            ) / torch.sqrt(alpha_t_given_s)

            # 对应 DDPM posterior variance。浮点误差可能产生 -1e-12 一类小负数，
            # clamp_min(0) 避免后面 sqrt 对负数得到 NaN。
            posterior_variance = (
                # 后验方差的分子是下一目标时间步的累计噪声量。
                (1.0 - alpha_bar_s)
                # 再除以当前时间步累计噪声量，得到条件方差比例。
                / (1.0 - alpha_bar_t)
                # 乘以本次跨步引入的有效噪声量，组成 DDPM 后验方差。
                * beta_t_given_s
            # 消除浮点舍入可能产生的极小负方差，保证后面能取平方根。
            ).clamp_min(0.0)
            # 到 s=0 已是最终干净结果，DDPM 后验方差为零，不应再注入随机量。
            if previous_step == 0:
                # 最终一步必须不再加入随机噪声，否则已经得到的干净 mel 会再次变脏。
                noise = torch.zeros_like(x)
            # 进入前面条件均未满足时的处理分支，具体数据操作在下面逐项说明。
            else:
                # 其他步骤按论文抽 z~N(0,I)。generator 用来固定随机种子。
                noise = torch.randn(
                    # 每个反向步的随机噪声保持当前 mel 的完整形状。
                    x.shape,
                    # 张量执行计算的设备。
                    device=x.device,
                    # 张量的数值类型。
                    dtype=x.dtype,
                    # 单独控制采样噪声的随机数生成器。
                    generator=generator,
                )
            # 完整一次反向更新：后验均值 + 后验标准差×随机噪声。
            x = mean + torch.sqrt(posterior_variance) * noise
        # 30 轮更新完成后，返回目标说话人条件下的转换 mel。
        return x


# 入口保护：只有直接运行本文件才执行下面检查，被其他脚本 import 时不会执行。
if __name__ == "__main__":
    # Python 入口保护：只有运行 python diffusion.py 才执行；被 train.py import 时不执行。
    # assert 条件不成立会立即报错，用于防止有人误改 schedule 长度或 30 步设置。
    diffusion = VoiceGradDiffusion()
    # T=1000 且额外保留索引 0，所以累计系数表长度必须为 1001。
    assert diffusion.alphas_cumprod.shape == (1001,)
    # 默认稀疏序列的最大一步必须是当前明确记录的 S_K=950。
    assert diffusion.sampling_sequence()[-1] == 950
    # 再确认序列确实含 30 个选择步，名副其实是 VoiceGrad-30。
    assert len(diffusion.sampling_sequence()) == 30
    # 三项断言都通过后给出肉眼可见的成功提示。
    print("VoiceGrad diffusion schedule check passed.")
