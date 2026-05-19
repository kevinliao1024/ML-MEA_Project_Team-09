import json
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

from torch.utils.data import DataLoader

from TEST_DATASET import TrainDataset
from DeepConvNet import DeepConvNet

# =====================================================
# PATH
# =====================================================

DATA_INFO_PATH = r"D:\course project\SLEEP\dataset_info.json"

INDEX_PATH_VAL = r"D:\course project\SLEEP\val.h5"

MODEL_055 = r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_eegnet_model_dropout0.55.pth"
MODEL_060 = r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_eegnet_model_batch8.pth"
MODEL_065 = r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_eegnet_model_dropout0.65.pth"

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cpu")

print(f"Using device: {device}")

# =====================================================
# LOAD DATA INFO
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

# =====================================================
# Z-SCORE
# =====================================================

def z_score_normalize(x):

    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)

    return (x - mean) / (std + 1e-8)

# =====================================================
# DATASET
# =====================================================

external_val_ds = TrainDataset(INDEX_PATH_VAL)

external_val_loader = DataLoader(
    external_val_ds,
    batch_size=8,
    shuffle=False
)

# =====================================================
# LOAD MODELS
# =====================================================

model_055 = DeepConvNet(
    chans=CHANNELS,
    time_point=TIME_POINTS,
    nb_classes=CLASSES,
    dropoutRate=0.55
).to(device)

model_060 = DeepConvNet(
    chans=CHANNELS,
    time_point=TIME_POINTS,
    nb_classes=CLASSES,
    dropoutRate=0.60
).to(device)

model_065 = DeepConvNet(
    chans=CHANNELS,
    time_point=TIME_POINTS,
    nb_classes=CLASSES,
    dropoutRate=0.65
).to(device)

# =====================================================
# LOAD CHECKPOINT
# =====================================================

ckpt_055 = torch.load(MODEL_055, map_location=device)
ckpt_060 = torch.load(MODEL_060, map_location=device)
ckpt_065 = torch.load(MODEL_065, map_location=device)

model_055.load_state_dict(ckpt_055['model'])
model_060.load_state_dict(ckpt_060['model'])
model_065.load_state_dict(ckpt_065['model'])

model_055.eval()
model_060.eval()
model_065.eval()

print("\nModels Loaded Successfully!")

# =====================================================
# ENSEMBLE INFERENCE
# =====================================================

all_preds = []
all_labels = []

with torch.no_grad():

    for data, label in external_val_loader:

        data = data.to(device)
        label = label.to(device)

        # ---------------------------------------------
        # normalization
        # ---------------------------------------------

        data = z_score_normalize(data)

        # ---------------------------------------------
        # forward
        # ---------------------------------------------

        out_055 = model_055(data)
        out_060 = model_060(data)
        out_065 = model_065(data)

        # ---------------------------------------------
        # probabilities
        # ---------------------------------------------

        prob_055 = torch.softmax(out_055, dim=1)
        prob_060 = torch.softmax(out_060, dim=1)
        prob_065 = torch.softmax(out_065, dim=1)

        # ---------------------------------------------
        # weighted ensemble
        # ---------------------------------------------

        prob = (
            0.6 * prob_055 +
            0.25 * prob_060 +
            0.15 * prob_065
        )

        # ---------------------------------------------
        # prediction
        # ---------------------------------------------

        pred = torch.argmax(prob, dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

# =====================================================
# FINAL METRICS
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

print("\n========== Ensemble External Validation ==========")

print(f"Accuracy : {final_acc:.4f}")
print(f"Macro F1 : {final_f1:.4f}")
print(f"Kappa    : {final_kappa:.4f}")

# =====================================================
# CLASSIFICATION REPORT
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
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(all_labels, all_preds)

print("\n========== Confusion Matrix ==========")

print(cm)