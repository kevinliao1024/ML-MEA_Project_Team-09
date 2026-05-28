# Uni-Mol 分子属性预测训练指南（unimol_tools）

> 使用 `unimol_tools` Python 包进行分子属性预测/微调  
> 数据集：`molecular_property_prediction.tar.gz`（3.51 GB）

---

## 一、安装依赖

```bash
pip install -r requirements.txt
```

**国内用户建议**设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 二、准备数据集

### 1. 下载数据集
```bash
# 数据集 3.51 GB
wget https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz
```

### 2. 解压到项目 data/ 目录
```bash
tar -xzf molecular_property_prediction.tar.gz -C data/
```

解压后目录结构：
```
data/molecular_property_prediction/
├── qm9dft/
├── esol/
├── bace/
├── tox21/
└── ...（共 15 个子任务）
```

### 3. 转换为 CSV（unimol_tools 推荐格式）
```bash
python scripts/1_convert_lmdb_to_csv.py
```
转换后 CSV 保存在 `csv_data/` 目录。

---

## 三、环境检查

```bash
python scripts/check_env.py
```

确认输出中以下项均为 ✅：
- Python >= 3.8
- PyTorch + CUDA
- RDKit
- unimol_tools
- 数据集目录

---

## 四、训练任务

### 方式 A：训练单个任务

```bash
# 回归任务示例（ESOL - 水溶性）
python scripts/2_train.py --task esol

# 分类任务示例（BACE - BACE 抑制剂）
python scripts/2_train.py --task bace

# 多标签分类示例（Tox21 - 毒性预测）
python scripts/2_train.py --task tox21

# 自定义学习率
python scripts/2_train.py --task esol --lr 5e-4 --epochs 100
```

### 方式 B：批量训练所有任务

```bash
python scripts/3_train_all.py
```

---

## 五、各任务参考数据

| 任务名 | 类型 | 指标 | 默认学习率 | 默认轮数 |
|--------|------|------|------------|------------|
| `esol` | 回归 | RMSE | 1e-3 | 50 |
| `freesolv` | 回归 | RMSE | 1e-3 | 50 |
| `lipo` | 回归 | RMSE | 1e-3 | 50 |
| `qm9dft` | 回归 | MAE | 1e-4 | 40 |
| `bace` | 分类 | AUC | 1e-4 | 50 |
| `bbbp` | 分类 | AUC | 1e-4 | 50 |
| `hiv` | 分类 | AUC | 1e-4 | 50 |
| `tox21` | 多标签分类 | AUC | 1e-4 | 50 |

---

## 六、硬件建议

| GPU 显存 | 推荐 batch_size | 备注 |
|------------|-----------------|------|
| 8 GB | 16 | 单任务可行 |
| 16 GB | 32~64 | 推荐 |
| 24 GB | 64~128 | 高效 |

---

## 七、输出说明

训练完成后：
```
checkpoints/
└── esol/
    ├── best_checkpoint.ckpt
    └── predictions_test.csv
```

---

## 八、常见问题

### Q: `unimol_tools` 安装失败
```bash
# 使用 conda 安装 RDKit（更可靠）
conda install -c conda-forge rdkit==2022.9.3
pip install unimol_tools
```

### Q: HuggingFace 下载预训练模型慢
```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/2_train.py --task esol
```

### Q: CUDA out of memory
```bash
python scripts/2_train.py --task esol --batch_size 8
```
