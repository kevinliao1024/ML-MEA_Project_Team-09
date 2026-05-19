import json
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
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
from TEST_DATASET import TrainDataset, TestDataset
#from EEGNet import EEGNet
#from EEGNet_SE import EEGNet_SE
from DeepConvNet import DeepConvNet
#加
import torch.nn.functional as F

# =========================================================
# Focal Loss
# =========================================================
class FocalLoss(nn.Module):

    def __init__(
            self,
            alpha=None,
            gamma=2.0,
            reduction='mean'
    ):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        # Cross Entropy
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            weight=self.alpha
        )

        # pt = exp(-CE)
        pt = torch.exp(-ce_loss)

        # Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()

        elif self.reduction == 'sum':
            return focal_loss.sum()

        else:
            return focal_loss


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =====================================================
# 配置与路径
# =====================================================
DATA_INFO_PATH = r"D:\course project\SLEEP\dataset_info.json"
DATA_NAME = "SLEEP"

INDEX_PATH_TRAIN = fr"D:\course project\{DATA_NAME}\train.h5"
INDEX_PATH_VAL = fr"D:\course project\{DATA_NAME}\val.h5"
INDEX_PATH_TEST = fr"D:\course project\{DATA_NAME}\test_x_only.h5"

MODEL_SAVE_PATH = "best_eegnet_model.pth"

# CPU ONLY
device = torch.device("cpu")
print(f"Using device: {device}")

# =====================================================
# 读取数据配置
# =====================================================
with open(DATA_INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)

category_list = info["dataset"]["category_list"]
channels = info["dataset"]["channels"]
target_sampling_rate = info["processing"]["target_sampling_rate"]
window_sec = info["processing"]["window_sec"]

CHANNELS = len(channels)
CLASSES = len(category_list)
TIME_POINTS = int(target_sampling_rate * window_sec)

BATCH_SIZE = 4
EPOCHS = 80
LR = 5e-4

print("\n========== Dataset Info ==========")
print("Classes:", category_list)
print("Num Channels:", CHANNELS)
print("Time Points:", TIME_POINTS)

# =====================================================
# EEG Z-score 标准化
# =====================================================
def z_score_normalize(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-8)

# =====================================================
# EEG 数据增强
# =====================================================
def eeg_augmentation(data):
    # ---------- Gaussian Noise ----------
    noise = 0.01 * torch.randn_like(data)
    data = data + noise

    # ---------- Random Temporal Shift ----------
    shift = np.random.randint(-20, 20)
    data = torch.roll(data, shifts=shift, dims=-1)

    return data

# =====================================================
# 加载数据
# =====================================================
full_train_ds = TrainDataset(INDEX_PATH_TRAIN)

train_size = int(0.8 * len(full_train_ds))
internal_val_size = len(full_train_ds) - train_size

train_ds, internal_val_ds = random_split(
    full_train_ds,
    [train_size, internal_val_size],
    generator=torch.Generator().manual_seed(seed)
)

external_val_ds = TrainDataset(INDEX_PATH_VAL)
test_ds = TestDataset(INDEX_PATH_TEST)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

internal_val_loader = DataLoader(
    internal_val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

external_val_loader = DataLoader(
    external_val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =====================================================
# 自动计算类别权重
# =====================================================
with h5py.File(INDEX_PATH_TRAIN, "r") as f:
    y_train = f["y"][()]

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

weights = torch.tensor(weights, dtype=torch.float32).to(device)

print("\n========== Class Distribution ==========")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

print("\nClass Weights:")
print(weights)

# =====================================================
# 模型
# =====================================================
'''model = EEGNet(
    chans=CHANNELS,
    nb_classes=CLASSES,
    time_point=TIME_POINTS
).to(device)'''
'''model = EEGNet_SE(
    chans=CHANNELS,
    time_point=TIME_POINTS,
    nb_classes=CLASSES
)'''
model = DeepConvNet(
    chans=CHANNELS,
    time_point=TIME_POINTS,
    nb_classes=CLASSES,
    dropoutRate=0.6
)
# =====================================================
# 损失、优化器（加入 Label Smoothing）
# =====================================================
'''criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=0.1
)'''
criterion = FocalLoss(
    alpha=weights,
    gamma=2.0
)

'''optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)'''
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# =====================================================
# Scheduler
# =====================================================
'''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=10
)'''
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5
)

# Early Stopping
patience = 15
trigger = 0

train_losses = []
val_accuracies = []
val_f1s = []
val_kappas = []

best_val_acc = 0.0

# =====================================================
# 训练循环
# =====================================================
for epoch in range(EPOCHS):
    model.train()

    train_loss_sum = 0.0
    train_num = 0

    for data, label in train_loader:

        data = data.to(device)
        label = label.to(device)

        # ---------- 标准化 ----------
        data = z_score_normalize(data)

        # ---------- 数据增强 ----------
        data = eeg_augmentation(data)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        '''加'''
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )
        # ---------- Gradient Clipping ----------
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss_sum += loss.item() * label.size(0)
        train_num += label.size(0)

    epoch_train_loss = train_loss_sum / train_num

    # INTERNAL VALIDATION
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for v_data, v_label in internal_val_loader:

            v_data = v_data.to(device)
            v_label = v_label.to(device)

            v_data = z_score_normalize(v_data)

            v_output = model(v_data)

            pred = torch.argmax(v_output, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(v_label.cpu().numpy())

    # Metrics
    epoch_val_acc = accuracy_score(all_labels, all_preds)

    epoch_val_f1 = f1_score(
        all_labels,
        all_preds,
        average='macro'
    )

    epoch_val_kappa = cohen_kappa_score(
        all_labels,
        all_preds
    )

    train_losses.append(epoch_train_loss)
    val_accuracies.append(epoch_val_acc)
    val_f1s.append(epoch_val_f1)
    val_kappas.append(epoch_val_kappa)

    # Scheduler Step
    scheduler.step(epoch_val_acc)

    # Save Best Model
    if epoch_val_acc > best_val_acc:

        best_val_acc = epoch_val_acc

        trigger = 0

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_val_acc
        }, MODEL_SAVE_PATH)

        status_msg = "<<< BEST MODEL SAVED >>>"

    else:
        trigger += 1
        status_msg = ""


    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch [{epoch+1:03d}/{EPOCHS}] | "
        f"Loss: {epoch_train_loss:.4f} | "
        f"Acc: {epoch_val_acc:.4f} | "
        f"F1: {epoch_val_f1:.4f} | "
        f"Kappa: {epoch_val_kappa:.4f} | "
        f"LR: {current_lr:.6f} "
        f"{status_msg}"
    )

    # Early Stopping
    if trigger >= patience:
        print("\nEarly stopping triggered!")
        break


print("\n" + "=" * 50)
print("Training Finished")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# =====================================================
# Load Best Model
# =====================================================
checkpoint = torch.load(MODEL_SAVE_PATH)

model.load_state_dict(checkpoint['model'])

model.eval()

# =====================================================
# External Validation（真正泛化能力）
# =====================================================
all_preds = []
all_labels = []

with torch.no_grad():

    for ex_data, ex_label in external_val_loader:

        ex_data = ex_data.to(device)
        ex_label = ex_label.to(device)

        ex_data = z_score_normalize(ex_data)

        ex_output = model(ex_data)

        pred = torch.argmax(ex_output, dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(ex_label.cpu().numpy())

# =====================================================
# Final Metrics
# =====================================================
final_acc = accuracy_score(all_labels, all_preds)

final_f1 = f1_score(
    all_labels,
    all_preds,
    average='macro'
)

final_kappa = cohen_kappa_score(
    all_labels,
    all_preds
)

print("\n========== External Validation ==========")
print(f"Accuracy : {final_acc:.4f}")
print(f"Macro F1 : {final_f1:.4f}")
print(f"Kappa    : {final_kappa:.4f}")

# =====================================================
# Classification Report
# =====================================================
print("\n========== Classification Report ==========")
print(
    classification_report(
        all_labels,
        all_preds,
        digits=4
    )
)

# =====================================================
# Confusion Matrix
# =====================================================
cm = confusion_matrix(all_labels, all_preds)

print("\n========== Confusion Matrix ==========")
print(cm)

# =====================================================
# Plot Curves
# =====================================================
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