# Uni-Mol 分子属性预测训练项目（unimol_tools）

> 使用 `unimol_tools` 对分子属性进行预测/微调训练  
> 数据集：molecular_property_prediction.tar.gz（3.51 GB）

---

## 项目结构

```
Uni-Mol-Training-Project/
├── config/
│   └── training_config.py          # Python 训练配置（推荐）
├── data/
│   └── molecular_property_prediction/   # ← 解压后的数据集放这里
│       ├── qm9dft/
│       ├── esol/
│       ├── freesolv/
│       ├── lipo/
│       ├── bbbp/
│       ├── bace/
│       ├── clin_tox/
│       ├── tox21/
│       ├── toxcast/
│       ├── sider/
│       ├── hiv/
│       ├── pcba/
│       └── muv/
├── csv_data/                       # 转换后的 CSV 数据（自动生成）
├── checkpoints/                    # 训练产出的模型
├── logs/                           # 日志
├── scripts/
│   ├── 1_convert_lmdb_to_csv.py  # 将官方 LMDB 数据转为 CSV
│   ├── 2_train.py                 # 训练脚本（单任务）
│   ├── 3_train_all.py             # 批量训练所有任务
│   └── check_env.py               # 环境检查
├── requirements.txt
├── README.md
└── TRAINING_GUIDE.md
```

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据

将下载好的 `molecular_property_prediction.tar.gz` 放到项目根目录，然后：
```bash
tar -xzf molecular_property_prediction.tar.gz -C data/
```

### 3. 转换数据为 CSV（unimol_tools 推荐使用 CSV）
```bash
python scripts/1_convert_lmdb_to_csv.py
```
转换后的 CSV 会保存在 `csv_data/` 目录。

### 4. 训练（以 ESOL 为例）
```bash
python scripts/2_train.py --task esol
```

### 5. 批量训练所有任务
```bash
python scripts/3_train_all.py
```

---

## 各任务说明

| 任务名 | 类型 | 评估指标 | 说明 |
|--------|------|----------|------|
| `qm9dft` | 回归 | MAE | QM9 量子化学性质（13个属性） |
| `qm7dft` | 回归 | MAE | QM7 性质 |
| `qm8dft` | 回归 | MAE | QM8 性质 |
| `esol` | 回归 | RMSE | 水溶性预测 |
| `freesolv` | 回归 | RMSE | 自由能溶解 |
| `lipo` | 回归 | RMSE | 亲脂性 |
| `bbbp` | 分类 | AUC | 血脑屏障穿透 |
| `bace` | 分类 | AUC | BACE 抑制剂 |
| `clin_tox` | 分类 | AUC | 临床毒性 |
| `tox21` | 多标签分类 | AUC | 毒性（12个标签） |
| `toxcast` | 多标签分类 | AUC | 毒性（617个标签） |
| `sider` | 多标签分类 | AUC | 药物副作用（27个标签） |
| `hiv` | 分类 | AUC | HIV 抑制剂 |
| `pcba` | 多标签分类 | AUC | 生化实验（128个标签） |
| `muv` | 多标签分类 | AUC/PRC | 活性预测（17个标签） |

---

## 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|------------|------------|
| GPU | 1 × 8GB 显存 | 1 × 16GB 或 4 × 16GB |
| 内存 | 16GB | 32GB+ |
| 存储 | 10GB 可用空间 | 20GB+ |
| Python | 3.8+ | 3.9 - 3.11 |

---

## 注意事项

- `unimol_tools` 会自动从 HuggingFace 下载预训练权重  
  如需使用镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 分类任务使用 `--maximize-best-checkpoint-metric`  
  回归任务使用 `--minimize-best-checkpoint-metric`
