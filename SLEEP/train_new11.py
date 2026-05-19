import json
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
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


# =========================================================
# Focal Loss + Label Smoothing
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(inputs, dim=-1)

        # 交叉熵部分
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        smooth_loss = -log_probs.mean(dim=-1)
        ce_loss = confidence * nll_loss + self.smoothing * smooth_loss

        # Focal 权重
        pt = torch.exp(-nll_loss)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


# --- Mixup 数据增强 ---
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =====================================================
# 基础配置
# =====================================================
seed = 2025 #3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_INFO_PATH = r"D:\course project\SLEEP\dataset_info.json"
DATA_NAME = "SLEEP"
INDEX_PATH_TRAIN = fr"D:\course project\{DATA_NAME}\train.h5"
INDEX_PATH_VAL = fr"D:\course project\{DATA_NAME}\val.h5"
MODEL_SAVE_PATH = "best_sleep_model_v9.pth"  #v8

device = torch.device("cpu")


# =====================================================
# 数据预处理与增强
# =====================================================
def z_score_normalize(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-8)


with open(DATA_INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)

category_list = info["dataset"]["category_list"]
CHANNELS = len(info["dataset"]["channels"])
CLASSES = len(category_list)
TIME_POINTS = int(info["processing"]["target_sampling_rate"] * info["processing"]["window_sec"])

# 超参数
BATCH_SIZE = 32
EPOCHS = 120
LR = 8e-4

# =====================================================
# 数据加载
# =====================================================
full_train_ds = TrainDataset(INDEX_PATH_TRAIN)
train_size = int(0.8 * len(full_train_ds))
val_size = len(full_train_ds) - train_size
train_ds, internal_val_ds = random_split(full_train_ds, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(seed))
external_val_ds = TrainDataset(INDEX_PATH_VAL)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
internal_val_loader = DataLoader(internal_val_ds, batch_size=BATCH_SIZE, shuffle=False)
external_val_loader = DataLoader(external_val_ds, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# 权重方案
# =====================================================
with h5py.File(INDEX_PATH_TRAIN, "r") as f:
    y_train = f["y"][()]

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights[1] = weights[1] * 1.62  # 强化 N1
weights[2] = weights[2] * 1.25  # 维持 N2
weights = torch.tensor(weights, dtype=torch.float32).to(device)

model = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)
criterion = FocalLoss(alpha=weights, gamma=2.0, smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =====================================================
# 训练循环
# =====================================================
best_val_acc = 0.0
patience = 30
trigger = 0
train_losses, val_accuracies, val_f1s, val_kappas = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_num = 0

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        data = z_score_normalize(data)

        # Mixup 逻辑，稍微减弱 alpha 以保护细微特征
        if np.random.random() > 0.4:
            inputs, targets_a, targets_b, lam = mixup_data(data, label, alpha=0.2)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(data)
            loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss_sum += loss.item() * label.size(0)
        train_num += label.size(0)

    scheduler.step()

    # 验证
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for v_data, v_label in internal_val_loader:
            v_data = z_score_normalize(v_data.to(device))
            v_output = model(v_data)
            all_preds.extend(torch.argmax(v_output, dim=1).cpu().numpy())
            all_labels.extend(v_label.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)

    train_losses.append(train_loss_sum / train_num)
    val_accuracies.append(acc)
    val_f1s.append(f1)
    val_kappas.append(kappa)

    if acc > best_val_acc:
        best_val_acc = acc
        trigger = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        msg = "<<< BEST MODEL SAVED >>>"
    else:
        trigger += 1
        msg = ""

    print(
        f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Loss: {train_losses[-1]:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f} {msg}")
    if trigger >= patience: break

# =====================================================
# 最终评估
# =====================================================
print("\n" + "=" * 50)
print("Training Finished")
print(f"Best Internal Validation Accuracy: {best_val_acc:.4f}")

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for ex_data, ex_label in external_val_loader:
        ex_data = z_score_normalize(ex_data.to(device))
        ex_output = model(ex_data)
        all_preds.extend(torch.argmax(ex_output, dim=1).cpu().numpy())
        all_labels.extend(ex_label.cpu().numpy())

# 生理平滑后处理
all_preds = median_filter(np.array(all_preds), size=3)

final_acc = accuracy_score(all_labels, all_preds)
final_f1 = f1_score(all_labels, all_preds, average='macro')
final_kappa = cohen_kappa_score(all_labels, all_preds)

print("\n========== External Validation (Smoothed) ==========")
print(f"Accuracy : {final_acc:.4f}")
print(f"Macro F1 : {final_f1:.4f}")
print(f"Kappa    : {final_kappa:.4f}")

print("\n========== Classification Report ==========")
print(classification_report(all_labels, all_preds, digits=4))

print("\n========== Confusion Matrix ==========")
print(confusion_matrix(all_labels, all_preds))

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Val Accuracy')
plt.plot(val_f1s, label='Val Macro F1')
plt.plot(val_kappas, label='Val Kappa')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Metrics Curve')
plt.legend()
plt.show()