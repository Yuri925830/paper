# 语音转换实验源码

| 实验 | 训练与评估范围 |
| --- | --- |
| [FastVoiceGrad](FastVoiceGrad/) | 学生模型、蒸馏训练和客观评估 |
| [VoiceGrad](VoiceGrad/) | 扩散基线、四说话人闭集与 VCTK 测试 |
| [FastVoiceGradCycle](FastVoiceGradCycle/) | 循环损失、系数筛选和测试集转换 |
| [VoiceGradCycle](VoiceGradCycle/) | 循环续训、闭集配对转换与评估 |
| [FastVoiceGradVQCycle](FastVoiceGradVQCycle/) | 冻结 BNF 码本、VQ-only 对照和 VQCycle 评估 |

各目录的 README 列出数据、训练、推理和评估入口。运行完整实验需要原语料、预训练权重及相应运行环境。
