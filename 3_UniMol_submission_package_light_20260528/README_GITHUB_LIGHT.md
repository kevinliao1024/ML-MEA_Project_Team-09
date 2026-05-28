# Uni-Mol Tools GitHub Light Package

这是适合放入 GitHub 的轻量版实验包。它保留可读说明、脚本、配置、指标汇总、图表和 test metric JSON；不包含大体积模型权重、完整预测 CSV、构象 SDF、LMDB 数据和压缩数据集。

## 包含内容

```text
README_experiment.md
README_GITHUB_LIGHT.md
scripts/
configs/
data_splits/data_inventory.csv
metrics/metrics_summary.csv
metrics/*_metrics_chart.*
predictions/*/test_predictions/test_metric.json
notes/for_teammates_and_ablation.md
output/
environment/
.gitignore
```

## 未包含的大文件

以下内容请放在网盘、Release 附件、实验服务器或 Git LFS，而不是直接提交到普通 Git 仓库：

```text
checkpoints/
*.pth
*.sdf
*.lmdb
test.predict.0.csv
cv.data
molecular_property_prediction.tar.gz
```

完整包位于本机：

```text
G:\Unimol\UniMol_submission_package_20260528
```

## 使用建议

如果只是展示实验方法、结果表、图表和消融说明，上传本 light 包即可。

如果组员需要重新推理或复查每个样本的预测，请同时分享 full 包，或至少分享：

```text
checkpoints/<task>/
predictions/<task>/test_predictions/test.predict.0.csv
data_splits/<task>/*.csv
```
