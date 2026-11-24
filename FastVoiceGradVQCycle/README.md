# FastVoiceGradVQCycle

| 阶段 | 代码入口 |
| --- | --- |
| 数据清单与特征 | `scripts/common/00_prepare_inventory.sh`、`scripts/common/01_prepare_protocol_features.sh` |
| 声码器 | `scripts/common/03_train_custom_hifigan.sh` |
| 训练 | `experiments/vq_small_experiment/vq_train_runner.py` |
| 推理 | `experiments/vq_small_experiment/fastvoicegrad_evaluate.py` |
| 评估 | `experiments/vq_small_experiment/fastvoicegrad_evaluate.py` |

运行需要与该实验匹配的语料、特征、模型权重和评估器。脚本保留原实验环境的路径约定。
