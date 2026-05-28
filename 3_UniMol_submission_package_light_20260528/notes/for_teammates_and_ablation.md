# 给其他组员和消融分析同学的交付说明

## 另外两个模型复现者需要提供

MolFingerprints 和 MPNN 组员也建议按照同样结构提交：

```text
model_name/
├── README_experiment.md
├── scripts/
├── configs/
├── data_splits/
├── predictions/
├── metrics/
├── checkpoints/
├── logs/
└── environment/
```

必须交：

| 类别 | 文件 | 目的 |
|---|---|---|
| 数据 | 四个任务的 train/valid/test CSV 或 fold index | 确认公平比较 |
| 预测 | 每个任务的 `y_true/y_pred` CSV | 统一重算指标 |
| 配置 | 超参数、seed、batch size、lr、epoch | 复现训练设置 |
| 指标 | metric CSV/JSON/日志 | 快速汇总 |
| 权重 | checkpoint | 必要时重新推理 |
| 环境 | requirements、Python/PyTorch/CUDA 版本 | 复现实验环境 |

预测 CSV 建议统一字段：

```csv
model,dataset,split,seed,smiles,label,prediction
```

分类任务建议使用：

```csv
model,dataset,split,seed,smiles,label,pred_prob,pred_label
```

Tox21 多标签任务建议使用长表格式：

```csv
model,dataset,split,seed,smiles,label_name,label,pred_prob
```

## 最后一个组员如何做消融

1. 先检查三个人是否使用相同数据划分。如果数据划分不同，不要直接比较最终分数。
2. 用同一个脚本重新计算指标，不直接抄各自报告里的数字。
3. 主表按照四个任务汇总 MolFingerprints、MPNN、Uni-Mol Tools。
4. 分析模型表征差异：
   - MolFingerprints：固定 2D 子结构特征。
   - MPNN：可学习 2D 图结构特征。
   - Uni-Mol Tools：预训练 3D 分子表征。
5. 分析任务差异：
   - ESOL：溶解度回归，关注 RMSE/MAE/R2/Pearson。
   - BBBP：二分类，关注 AUC/PR-AUC/MCC。
   - Tox21：多标签毒性，关注 AUC/PR-AUC，不要只看 Accuracy。
   - QM9DFT：量子化学属性，关注 MAE/R2。
6. 消融优先级：
   - 最高优先级：三模型主结果对比。
   - 中等优先级：每个模型至少一个内部消融。
   - 时间充足：做 seed 方差、不同超参数、不同输入表征对比。

## 建议的最终结果表

| Dataset | Metric | MolFingerprints | MPNN | Uni-Mol Tools | Best |
|---|---:|---:|---:|---:|---|
| ESOL | RMSE/MAE ↓ |  |  |  |  |
| BBBP | AUC ↑ |  |  |  |  |
| Tox21 | AUC ↑ |  |  |  |  |
| QM9DFT | MAE ↓ |  |  |  |  |

## Uni-Mol Tools 口径说明

本包中的 Uni-Mol 结果来自 `unimol_tools.MolTrain`，配置见 `configs/<task>/config.yaml`。其中 `tox21` 为 12 标签多标签分类，`qm9dft` 为 `homo,lumo,gap` 三目标回归。
