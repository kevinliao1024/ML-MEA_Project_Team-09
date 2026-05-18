import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)
from scipy.ndimage import median_filter
from TEST_DATASET import TrainDataset, TestDataset
from DeepConvNet import DeepConvNet

# =====================================================
# 1. 基础配置与路径
# =====================================================
DATA_INFO_PATH = r"D:\course project\SLEEP\dataset_info.json"
DATA_NAME = "SLEEP"
INDEX_PATH_VAL = fr"D:\course project\{DATA_NAME}\val.h5"  # 外部验证集

# 模型的权重文件路径
PATH_V9 = r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v5.pth"
PATH_V10 = r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v6.pth"
PATH_V6 = r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v7.pth"

device = torch.device("cpu")

# =====================================================
# 2. 数据预处理
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

BATCH_SIZE = 32

# 加载验证集
external_val_ds = TrainDataset(INDEX_PATH_VAL)
external_val_loader = DataLoader(external_val_ds, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# 3. 初始化并加载模型
# =====================================================
print("Loading models for Ensemble...")

model_v9 = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)
model_v10 = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)
model_v6 = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)

model_v9.load_state_dict(torch.load(PATH_V9, map_location=device))
model_v10.load_state_dict(torch.load(PATH_V10, map_location=device))
model_v6.load_state_dict(torch.load(PATH_V6, map_location=device))

model_v9.eval()
model_v10.eval()
model_v6.eval()

# =====================================================
# 4. Ensemble 权重分配
# =====================================================
# 根据单模型表现分配权重
WEIGHT_V9 = 0.6
WEIGHT_V10 = 0.25
WEIGHT_V6 = 0.15

# =====================================================
# 5. 加权软投票预测 (Weighted Soft Voting)
# =====================================================
print("Running Inference and Soft Voting...")
all_preds = []
all_labels = []

with torch.no_grad():
    for data, label in external_val_loader:
        data = data.to(device)
        label = label.to(device)
        data = z_score_normalize(data)

        # 获取每个模型的 Raw Output (Logits)
        out_v9 = model_v9(data)
        out_v10 = model_v10(data)
        out_v6 = model_v6(data)

        # 转换为概率分布 (Softmax)
        prob_v9 = F.softmax(out_v9, dim=1)
        prob_v10 = F.softmax(out_v10, dim=1)
        prob_v6 = F.softmax(out_v6, dim=1)

        # 加权融合概率
        ensemble_prob = (WEIGHT_V9 * prob_v9) + (WEIGHT_V10 * prob_v10) + (WEIGHT_V6 * prob_v6)

        # 取最大概率的索引作为最终预测
        pred = torch.argmax(ensemble_prob, dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

# =====================================================
# 6. 生理平滑后处理
# =====================================================
print("Applying Median Smoothing...")
all_preds_smoothed = median_filter(np.array(all_preds), size=3)

# =====================================================
# 7. 评估与打印结果
# =====================================================
final_acc = accuracy_score(all_labels, all_preds_smoothed)
final_f1 = f1_score(all_labels, all_preds_smoothed, average='macro')
final_kappa = cohen_kappa_score(all_labels, all_preds_smoothed)

print("========== ENSEMBLE RESULTS (Smoothed) ==========")
print(f"Accuracy : {final_acc:.4f}")
print(f"Macro F1 : {final_f1:.4f}")
print(f"Kappa    : {final_kappa:.4f}")

print("\n========== Classification Report ==========")
print(classification_report(all_labels, all_preds_smoothed, digits=4))

print("\n========== Confusion Matrix ==========")
cm = confusion_matrix(all_labels, all_preds_smoothed)
print(cm)

# 简单的混淆矩阵可视化
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Ensemble Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(CLASSES)
plt.xticks(tick_marks, ['Wake', 'N1', 'N2', 'N3', 'REM'])
plt.yticks(tick_marks, ['Wake', 'N1', 'N2', 'N3', 'REM'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()