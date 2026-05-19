import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import median_filter
from sklearn.metrics import f1_score, accuracy_score
import itertools

from TEST_DATASET import TrainDataset
from DeepConvNet import DeepConvNet

# =====================================================
# 1. 基础配置
# =====================================================
DATA_INFO_PATH = r"D:\course project\SLEEP\dataset_info.json"
DATA_NAME = "SLEEP"
INDEX_PATH_VAL = fr"D:\course project\{DATA_NAME}\val.h5"

# 待融合的四个核心模型路径
MODEL_PATHS = {
    "v5": r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v5.pth",
    "v6": r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v6.pth",
    "v8": r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v8.pth",
    "v9": r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v9.pth"
}

# 搜索步长设置
SEARCH_STEP = 0.008

device = torch.device("cpu")


# =====================================================
# 2. 数据加载与预处理
# =====================================================
def z_score_normalize(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-8)


with open(DATA_INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)

CHANNELS = len(info["dataset"]["channels"])
CLASSES = len(info["dataset"]["category_list"])
TIME_POINTS = int(info["processing"]["target_sampling_rate"] * info["processing"]["window_sec"])

val_loader = DataLoader(TrainDataset(INDEX_PATH_VAL), batch_size=32, shuffle=False)

# =====================================================
# 3. 离线概率缓存：每个模型仅跑一次，提取 Softmax 概率
# =====================================================
print("正在执行一轮离线推理，缓存各个模型的特征空间概率...")
model_probs = {}
all_labels = []
labels_loaded = False

for name, path in MODEL_PATHS.items():
    print(f"-> 正在加载并提取模型 {name} 的预测概率...")
    model = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    probs_list = []
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            data = z_score_normalize(data)

            outputs = model(data)
            probs = F.softmax(outputs, dim=1)
            probs_list.append(probs.cpu().numpy())

            if not labels_loaded:
                all_labels.extend(label.numpy())

    model_probs[name] = np.concatenate(probs_list, axis=0)  # 形状: [样本数, 类别数]
    labels_loaded = True

true_labels = np.array(all_labels)
print(f"概率缓存完毕！样本总数: {len(true_labels)}")

# =====================================================
# 4. 生成所有合法的权重组合
# =====================================================
model_names = list(MODEL_PATHS.keys())
grid = np.arange(0.0, 1.01, SEARCH_STEP)

valid_combinations = []
# 使用 itertools.product 穷举组合
for w in itertools.product(grid, repeat=len(model_names)):
    if np.isclose(sum(w), 1.0):
        valid_combinations.append(w)

print(f"生成的合法权重网格组合数: {len(valid_combinations)} 组。开始在内存中爆速搜索...")

# =====================================================
# 5. 高速网格搜索核心逻辑
# =====================================================
best_macro_f1 = 0.0
best_weights = None
best_preds_smoothed = None

for w_comb in valid_combinations:
    # 构建当前组合的权重映射
    weights = {model_names[i]: w_comb[i] for i in range(len(model_names))}

    # 纯 NumPy 矩阵乘法，瞬间完成概率加权
    ensemble_prob = np.zeros_like(model_probs[model_names[0]])
    for name in model_names:
        ensemble_prob += weights[name] * model_probs[name]

    # 决策输出与生理连续性平滑
    preds = np.argmax(ensemble_prob, axis=1)
    preds_smoothed = median_filter(preds, size=3)

    # 计算评估指标（这里以你最关心的 Macro F1 为主优化目标）
    current_f1 = f1_score(true_labels, preds_smoothed, average='macro')

    if current_f1 > best_macro_f1:
        best_macro_f1 = current_f1
        best_weights = weights
        best_preds_smoothed = preds_smoothed

# =====================================================
# 6. 打印最优刷分报告
# =====================================================
best_acc = accuracy_score(true_labels, best_preds_smoothed)

print("\n" + "=" * 50)
print("最优集成权重搜索成功！结果如下：")
print("=" * 50)
for name, weight in best_weights.items():
    print(f" 模型 {name} 最佳权重: {weight:.2f}")
print("-" * 50)
print(f"优化后最高 Macro F1 : {best_macro_f1:.4f} ")
print(f"优化后对应 Accuracy  : {best_acc:.4f}")