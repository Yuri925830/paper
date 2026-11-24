# 自训声码器采用 V1 生成器与 MPD+MRD；独立训练方案属于本项目明确记录的选择。
"""FastVoiceGrad 论文对齐的自训练 HiFi-GAN：V1 Generator + MPD + MRD。

``train.py`` 是训练总入口，``dataset.py`` 负责对齐波形与 mel，``mrd.py`` 实现
三分辨率判别器。本文件不自动导入它们，防止仅查看包时就加载 PyTorch/GPU。
"""
