# VoiceGradCycle

| 阶段 | 代码入口 |
| --- | --- |
| 数据清单与特征 | `scripts/common/00_prepare_inventory.sh`、`scripts/common/01_prepare_protocol_features.sh` |
| 声码器 | `scripts/common/03_train_custom_hifigan.sh` |
| 训练 | `voicegrad_continue_cycle.py` |
| 推理 | `voicegrad/infer_voicegrad30_paper_eval.py` |
| 评估 | `scripts/common/05_evaluate_8100_pairs.sh` |

运行需要与该实验匹配的语料、特征、模型权重和评估器。脚本保留原实验环境的路径约定。

四说话人闭集实验在 `closed_set/`。原模型、扩散器、数据读取器和 BNF 提取代码在 `source/`。按 12 个有方向的说话人配对各转换 32 句；`source/closedset_evaluation.ipynb` 提供推理、CER 和声学性能评估，`source/checkpoint_sweep.ipynb` 用于筛选检查点。
