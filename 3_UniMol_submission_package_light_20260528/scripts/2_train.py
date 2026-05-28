#!/usr/bin/env python3
"""
脚本2: 单任务训练（unimol_tools）
用法:
  python scripts/2_train.py --task esol
  python scripts/2_train.py --task bace
  python scripts/2_train.py --task tox21
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

# ==========================================
# 每个任务的超参数（参考 Uni-Mol 论文 + unimol_tools 文档）
# task_type : unimol_tools 的 task 参数
# metrics   : 评估指标
# lr        : 学习率
# epochs    : 训练轮数
# ==========================================
TASK_CONFIG = {
    # ---- 回归任务 ----
    "qm9dft": {"type": "multilabel_regression",   "metrics": "mae",  "lr": 1e-4, "epochs": 40},
    "qm7dft": {"type": "regression",              "metrics": "mae",  "lr": 1e-4, "epochs": 40},
    "qm8dft": {"type": "regression",              "metrics": "mae",  "lr": 1e-4, "epochs": 40},
    "esol":   {"type": "regression",              "metrics": "mse",  "lr": 1e-3, "epochs": 50},
    "freesolv": {"type": "regression",           "metrics": "mse",  "lr": 1e-3, "epochs": 50},
    "lipo":   {"type": "regression",              "metrics": "mse",  "lr": 1e-3, "epochs": 50},
    # ---- 二分类任务 ----
    "bbbp":    {"type": "classification",          "metrics": "auc", "lr": 1e-4, "epochs": 50},
    "bace":    {"type": "classification",          "metrics": "auc", "lr": 1e-4, "epochs": 50},
    "clintox": {"type": "classification",          "metrics": "auc", "lr": 1e-4, "epochs": 50},
    "hiv":     {"type": "classification",          "metrics": "auc", "lr": 1e-4, "epochs": 50},
    # ---- 多标签分类任务 ----
    "tox21":   {"type": "multilabel_classification", "metrics": "auc", "lr": 1e-4, "epochs": 50},
    "toxcast": {"type": "multilabel_classification", "metrics": "auc", "lr": 1e-4, "epochs": 50},
    "sider":   {"type": "multilabel_classification", "metrics": "auc", "lr": 1e-4, "epochs": 50},
    "pcba":    {"type": "multilabel_classification", "metrics": "auc", "lr": 1e-4, "epochs": 50},
    "muv":     {"type": "multilabel_classification", "metrics": "auc", "lr": 1e-4, "epochs": 50},
}

QM9DFT_TARGET_COLS = ["homo", "lumo", "gap"]
NON_TARGET_COLS = {"SMILES", "atoms", "coordinates"}


def get_csv_path(task_name, split):
    """返回 csv_data/ 下对应任务的 CSV 路径"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "csv_data", f"{task_name}_{split}.csv")


def get_lmdb_path(task_name, split):
    """返回 data/molecular_property_prediction/ 下对应任务的 LMDB 路径"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "molecular_property_prediction", task_name, f"{split}.lmdb")


def get_target_cols(df):
    """Return all target columns in a Uni-Mol CSV."""
    if "SMILES" not in df.columns:
        raise ValueError("CSV 必须包含 SMILES 列")
    target_cols = [col for col in df.columns if col not in NON_TARGET_COLS]
    if not target_cols:
        raise ValueError("CSV 必须至少包含一个目标列")
    return target_cols


def _target_values(target):
    if hasattr(target, "tolist"):
        target = target.tolist()
    if isinstance(target, (tuple, list)):
        return list(target)
    return [target]


def _as_coordinate_matrix(coordinates):
    array = np.asarray(coordinates, dtype=float)
    if array.ndim == 2 and array.shape[1] == 3:
        return array
    return None


def select_aligned_coordinates(coordinates, atoms):
    """Return one coordinate matrix shaped [n_atoms, 3]."""
    atoms_count = len(atoms)
    matrix = _as_coordinate_matrix(coordinates)
    if matrix is not None and matrix.shape[0] == atoms_count:
        return matrix.tolist()

    for conformer in coordinates:
        matrix = _as_coordinate_matrix(conformer)
        if matrix is not None and matrix.shape[0] == atoms_count:
            return matrix.tolist()

    raise ValueError("QM9DFT coordinates shape is not aligned with atoms")


def qm9dft_record_to_row(record):
    """Convert one official QM9DFT LMDB item into a MolTrain-ready row."""
    target_values = _target_values(record.get("target"))
    if len(target_values) != len(QM9DFT_TARGET_COLS):
        raise ValueError(f"QM9DFT 需要 {len(QM9DFT_TARGET_COLS)} 个 target，实际为 {len(target_values)}")

    smiles = record.get("smi", record.get("smiles", record.get("SMILES")))
    if not smiles:
        raise ValueError("QM9DFT LMDB 记录缺少 SMILES")
    if "atoms" not in record or "coordinates" not in record:
        raise ValueError("QM9DFT LMDB 记录缺少 atoms/coordinates")

    row = {
        "SMILES": smiles,
        "atoms": record["atoms"],
        "coordinates": select_aligned_coordinates(record["coordinates"], record["atoms"]),
    }
    row.update(dict(zip(QM9DFT_TARGET_COLS, [float(value) for value in target_values])))
    return row


def load_qm9dft_lmdb_split(split):
    """Load QM9DFT from official LMDB so DFT coordinates are preserved."""
    import lmdb

    lmdb_path = get_lmdb_path("qm9dft", split)
    if not os.path.isfile(lmdb_path):
        raise FileNotFoundError(f"未找到 QM9DFT {split} LMDB: {lmdb_path}")

    rows = []
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(values=False):
                rows.append(qm9dft_record_to_row(pickle.loads(txn.get(key))))
    finally:
        env.close()

    return pd.DataFrame(rows, columns=["SMILES", "atoms", "coordinates"] + QM9DFT_TARGET_COLS)


def load_csv_split(task_name, split):
    path = get_csv_path(task_name, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 {task_name} {split} CSV: {path}")
    return pd.read_csv(path)


def load_task_data(task_name):
    if task_name == "qm9dft":
        return (
            load_qm9dft_lmdb_split("train"),
            load_qm9dft_lmdb_split("valid"),
            load_qm9dft_lmdb_split("test"),
            QM9DFT_TARGET_COLS,
            "官方 LMDB atoms/coordinates",
        )

    train_df = load_csv_split(task_name, "train")
    valid_df = load_csv_split(task_name, "valid")
    test_df = load_csv_split(task_name, "test")
    target_cols = get_target_cols(train_df)
    for split_name, df in [("验证集", valid_df), ("测试集", test_df)]:
        other_cols = get_target_cols(df)
        if other_cols != target_cols:
            raise ValueError(f"{split_name}目标列与训练集不一致: {other_cols} != {target_cols}")
    return train_df, valid_df, test_df, target_cols, "CSV/SMILES"


def evaluate_test_set(save_dir, test_df, metrics):
    """Evaluate the trained checkpoint on the official test split."""
    from unimol_tools import MolPredict

    test_save_dir = os.path.join(save_dir, "test_predictions")
    predictor = MolPredict(load_model=save_dir)
    predictor.predict(data=test_df, save_path=test_save_dir, metrics=metrics)
    return test_save_dir


def train_task(task_name, lr=None, epochs=None, batch_size=32):
    cfg = TASK_CONFIG.get(task_name)
    if cfg is None:
        print(f"❌ 未知任务: {task_name}")
        print(f"支持的任务: {list(TASK_CONFIG.keys())}")
        sys.exit(1)

    lr     = lr     or cfg["lr"]
    epochs = epochs or cfg["epochs"]
    task_type = cfg["type"]
    metrics   = cfg["metrics"]

    print("=" * 60)
    print(f"任务       : {task_name}")
    print(f"类型       : {task_type}")
    print(f"评估指标 : {metrics}")
    print(f"学习率   : {lr}")
    print(f"训练轮数 : {epochs}")
    print(f"Batch Size: {batch_size}")
    print("=" * 60)

    # ---- 读取数据 ----
    try:
        train_df, valid_df, test_df, target_cols, data_source = load_task_data(task_name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("请先确认数据已准备好；CSV 任务可运行: python scripts/1_convert_lmdb_to_csv.py")
        sys.exit(1)

    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(valid_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    print(f"数据来源: {data_source}")
    print(f"目标列: {target_cols}")

    # ---- 用 unimol_tools 训练 ----
    from unimol_tools import MolTrain

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints", task_name
    )
    os.makedirs(save_dir, exist_ok=True)

    # 只传训练集给 fit()，kfold=5 表示 5 折交叉验证
    clf = MolTrain(
        task           = task_type,
        data_type      = "molecule",
        epochs         = epochs,
        batch_size     = batch_size,
        metrics        = metrics,
        learning_rate  = lr,
        kfold          = 5,
        smiles_col     = "SMILES",
        target_cols    = target_cols,
        pretrained_model_path = None,
        save_path      = save_dir,
    )

    print(f"开始训练（训练集 {len(train_df)} 条，5 折交叉验证）...\n")
    clf.fit(train_df)

    # 用训练好的 k-fold checkpoint 在官方测试集 split 上预测并计算指标
    test_save_dir = evaluate_test_set(save_dir, test_df, metrics)

    print(f"\n✅ 训练完成！模型保存在: {save_dir}")
    print(f"   评估指标（5 折交叉验证）见: {save_dir}/metric.result")
    print(f"   官方测试集预测与指标见: {test_save_dir}")
    visualize_results(save_dir, task_name)
    return clf


def visualize_results(save_dir, task_name):
    """读取 metric.result 并生成可视化图表（PNG + HTML）"""
    import pickle
    import numpy as np

    result_path = os.path.join(save_dir, "metric.result")
    if not os.path.isfile(result_path):
        print(f"\n⚠️  未找到指标文件: {result_path}，跳过可视化")
        return

    with open(result_path, "rb") as f:
        metrics = pickle.load(f)

    # 转为普通 dict
    if hasattr(metrics, "item"):
        metrics = {k: float(v) for k, v in metrics.items()}
    else:
        metrics = {k: float(v) for k, v in metrics.items()}

    # 按类型分组
    regression_keys   = ["mae", "mse", "rmse", "r2", "spearmanr", "pearsonr"]
    classification_keys = ["auc", "auroc", "auprc", "acc", "f1_score", "mcc",
                            "precision", "recall", "cohen_kappa", "log_loss"]
    keys = list(metrics.keys())

    # 过滤有效数值型指标
    display_keys = [k for k in keys if k in regression_keys + classification_keys]
    values = [metrics[k] for k in display_keys]

    print("\n" + "=" * 50)
    print("📊 5 折交叉验证结果汇总")
    print("=" * 50)
    for k, v in zip(display_keys, values):
        bar_len = int(v * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {k:<16} {v:.4f}  {bar}")
    print("=" * 50)

    # ---------- PNG 图表（matplotlib） ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 尝试找中文字体
        chinese_fonts = [f.name for f in fm.fontManager.ttflist
                         if any(kw in f.name.lower() for kw in ["noto", "wqy", "heiti", "songti", "simhei", "simsun", "droid"])]
        plt.rcParams["font.family"] = chinese_fonts[0] if chinese_fonts else "DejaVu Sans"

        fig, ax = plt.subplots(figsize=(10, max(5, len(display_keys) * 0.45)))

        colors = [
            "#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#BA7517",
            "#ED93B1", "#639922", "#888780", "#F09595", "#AFA9EC"
        ]
        bars = ax.barh(display_keys, values, color=colors[:len(display_keys)], edgecolor="none", height=0.6)
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f" {val:.4f}", va="center", fontsize=11)

        ax.set_xlim(0, min(1.15, max(values) * 1.2 + 0.1))
        ax.set_xlabel("Score", fontsize=12)
        ax.set_title(f"Uni-Mol — {task_name} — 5-Fold CV Results", fontsize=14, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#f0f0f0", linewidth=0.8)
        ax.tick_params(axis="y", labelsize=11)
        plt.tight_layout()

        # 项目根目录 = checkpoints 的上一级
        project_root = os.path.dirname(os.path.dirname(save_dir))
        png_path = os.path.join(project_root, f"{task_name}_metrics_chart.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 PNG 图表已保存: {png_path}")
    except Exception as e:
        print(f"\n⚠️  matplotlib 图表生成失败: {e}")
        print("  （服务器可能无显示环境，不影响训练结果）")

    # ---------- HTML 图表（Chart.js，无需字体） ----------
    try:
        html_path = os.path.join(project_root, f"{task_name}_metrics_chart.html")
        labels_js = str(display_keys)
        values_js = [f"{v:.4f}" for v in values]
        color_js = [colors[i % len(colors)] for i in range(len(display_keys))]

        html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Uni-Mol {task_name} — Metrics</title>
<style>
 *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:#f5f6fa;padding:2rem;color:#2c2c2c}}
  .card{{background:#fff;border-radius:12px;padding:1.5rem;margin-bottom:1rem;
         border:0.5px solid rgba(0,0,0,0.08);max-width:800px;margin:0 auto 1rem}}
  h1{{font-size:1.1rem;font-weight:600;margin-bottom:0.25rem}}
  .subtitle{{font-size:0.8rem;color:#888}}
  .chart-wrap{{position:relative;height:340px}}
  .warn{{font-size:0.85rem;color:#888;margin-top:0.5rem}}
  .badge{{display:inline-block;padding:0.15rem 0.6rem;border-radius:20px;
          font-size:0.75rem;font-weight:500;margin-right:0.5rem;margin-bottom:0.3rem}}
  .badge-good{{background:#e6f4ea;color:#276221}}
  .badge-ok{{background:#fff3e0;color:#7d3c00}}
  .badge-low{{background:#fce8e6;color:#9c1919}}
</style>
</head>
<body>
<div class="card">
  <h1>Uni-Mol · {task_name}</h1>
  <div class="subtitle">5-Fold Cross-Validation · 5折交叉验证</div>
</div>
<div class="card">
  <div class="chart-wrap">
    <canvas id="chart" role="img" aria-label="Metrics chart"></canvas>
  </div>
</div>
<div class="card">
  <div style="font-size:0.85rem;color:#666;margin-bottom:0.6rem">数值速览</div>
  <div>
"""

        for k, v in zip(display_keys, values):
            if v >= 0.8:
                badge_cls = "badge-good"
            elif v >= 0.6:
                badge_cls = "badge-ok"
            else:
                badge_cls = "badge-low"
            html_content += f'    <span class="badge {badge_cls}">{k}: {v:.4f}</span>\n'

        html_content += f"""  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
new Chart(document.getElementById('chart'), {{
  type: 'bar',
  data: {{
    labels: {labels_js},
    datasets: [{{
      label: 'Score',
      data: {values_js},
      backgroundColor: {str(color_js)},
      borderRadius: 5,
      borderSkipped: false,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    scales: {{
      x: {{ min: 0, max: 1, grid: {{ color: 'rgba(0,0,0,0.05)' }} }},
      y: {{ grid: {{ display: false }} }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});
</script>
</body>
</html>"""

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"🌐 HTML 图表已保存: {html_path}")
        print(f"   （本地浏览器打开，或上传到服务器公网访问）")
    except Exception as e:
        print(f"\n⚠️  HTML 图表生成失败: {e}")



def main():
    parser = argparse.ArgumentParser(description="Uni-Mol 分子属性预测训练（unimol_tools）")
    parser.add_argument("--task",  type=str, required=True,
                        choices=list(TASK_CONFIG.keys()),
                        help="任务名称")
    parser.add_argument("--lr",     type=float, default=None, help="学习率（覆盖默认值）")
    parser.add_argument("--epochs", type=int,   default=None, help="训练轮数（覆盖默认值）")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size（默认 32）")
    args = parser.parse_args()

    # 检查 unimol_tools 是否可用；放在参数解析后，避免 --help 被依赖检查拦截
    try:
        from unimol_tools import MolTrain  # noqa: F401
    except ImportError:
        print("ERROR: 未安装 unimol_tools！")
        print("请运行: pip install unimol_tools")
        sys.exit(1)

    train_task(args.task, args.lr, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
