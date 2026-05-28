# Uni-Mol Tools 可提交实验包

本文件夹整理的是本组 Uni-Mol 复现实验中可提交、可复查、可交给消融分析同学使用的材料。模型实现口径为 `unimol_tools.MolTrain`，不是完整 Uni-Mol 原仓库的底层训练框架。

## 实验口径

| 项目 | 内容 |
|---|---|
| 模型 | Uni-Mol Tools / `unimol_tools` |
| 训练接口 | `MolTrain` |
| 模型名 | `unimolv1` |
| 模型规模 | `84m` |
| 数据类型 | `molecule` |
| 数据集 | ESOL, BBBP, Tox21, QM9DFT |
| 训练评估 | `kfold=5`, `split_method=5fold_random` |
| 随机种子 | `seed=42`, `split_seed=42` |
| 输入 | SMILES，训练时由 Uni-Mol Tools 生成/缓存分子构象 |
| GPU | `use_cuda=true`, `use_amp=true` |

## 目录说明

```text
UniMol_submission_package_20260528/
├── README_experiment.md
├── scripts/              # 训练与 LMDB->CSV 转换脚本
├── configs/              # 每个任务的 config.yaml
├── data_splits/          # 四个任务的 train/valid/test CSV
├── predictions/          # 已导出的 test prediction 文件
├── metrics/              # 指标、图表、cv.data、汇总表
├── checkpoints/          # 每个任务的模型权重和训练产物
├── logs/                 # 训练日志
├── output/               # 可直接展示的报告、图表、poster 摘要
├── environment/          # requirements 和项目说明
└── notes/                # 给组员和消融分析同学的说明
```

## 任务与目标列

| 任务 | 类型 | 目标列 | 主指标 |
|---|---|---|---|
| ESOL | regression | `TARGET` | MSE/MAE/Pearson/R2 |
| BBBP | classification | `TARGET` | AUC |
| Tox21 | multilabel classification | 12 个毒性标签 | AUC |
| QM9DFT | multilabel regression | `homo,lumo,gap` | MAE |

## 关键文件

- 训练脚本：`scripts/2_train.py`, `scripts/3_train_all.py`
- 数据划分：`data_splits/<task>/<task>_train.csv`, `valid.csv`, `test.csv`
- 配置文件：`configs/<task>/config.yaml`
- 原始指标：`metrics/<task>/metric.result`
- 可读汇总：`metrics/metrics_summary.csv`
- 模型权重：`checkpoints/<task>/model_*.pth`
- 测试预测：`predictions/<task>/test_predictions/`，四个任务均已包含官方 test split prediction
- 展示材料：`output/UniMol训练结果分析报告.md`, `output/poster_summary_bilingual.md`, `output/*_metrics_chart.png`

## 注意事项

1. `metric.result` 是 Uni-Mol Tools 生成的二进制结果文件，建议保留原文件作为证据。
2. `metrics_summary.csv` 是为了组内汇总人工整理出的可读版本，最终论文/报告中应优先注明指标来源。
3. `tox21` 当前下载版 CSV 已保留 12 个标签列，不再是单标签简化版本。
4. `qm9dft` 当前下载版 CSV 预测 `homo,lumo,gap` 三个目标。
5. 若最后消融分析要求所有模型统一用官方 test 集比较，建议补齐 ESOL 和 BBBP 的 test prediction CSV。
