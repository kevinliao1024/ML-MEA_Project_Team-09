import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import median_filter
from TEST_DATASET import TestDataset
from DeepConvNet import DeepConvNet

# =====================================================
# 1. 基础配置与路径
# =====================================================
DATA_INFO_PATH = r"D:\course project\SLEEP\dataset_info.json"
DATA_NAME = "SLEEP"

INDEX_PATH_TEST = fr"D:\course project\SLEEP\test_x_only.h5"
OUTPUT_TXT_PATH = "submit_labels.txt"  # 最终生成的 txt 文件名

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

# 加载测试集
test_ds = TestDataset(INDEX_PATH_TEST)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# 3. 初始化并加载三个模型
# =====================================================
print("正在加载 Ensemble 模型权重...")
model_v9 = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)
model_v10 = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)
model_v6 = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)

model_v9.load_state_dict(torch.load(PATH_V9, map_location=device))
model_v10.load_state_dict(torch.load(PATH_V10, map_location=device))
model_v6.load_state_dict(torch.load(PATH_V6, map_location=device))

model_v9.eval()
model_v10.eval()
model_v6.eval()

# 最强权重组合
WEIGHT_V9 = 0.6
WEIGHT_V10 = 0.25
WEIGHT_V6 = 0.15

# =====================================================
# 4. 在测试集上进行预测 (加权软投票)
# =====================================================
print("正在对未知测试集进行预测...")
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        # TestDataset 可能只返回 data，也可能返回 (data, id)。这里做个兼容处理
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch

        data = data.to(device)
        data = z_score_normalize(data)

        # 获取各模型的 Raw Output
        out_v9 = model_v9(data)
        out_v10 = model_v10(data)
        out_v6 = model_v6(data)

        # 转换为 Softmax 概率
        prob_v9 = F.softmax(out_v9, dim=1)
        prob_v10 = F.softmax(out_v10, dim=1)
        prob_v6 = F.softmax(out_v6, dim=1)

        # 加权融合概率
        ensemble_prob = (WEIGHT_V9 * prob_v9) + (WEIGHT_V10 * prob_v10) + (WEIGHT_V6 * prob_v6)

        # 取最大概率的索引作为预测类别
        pred = torch.argmax(ensemble_prob, dim=1)
        all_preds.extend(pred.cpu().numpy())

# =====================================================
# 5. 生理平滑后处理与导出
# =====================================================
print("正在应用中值滤波进行生理平滑...")
# 使用与验证集一致的平滑逻辑
all_preds_smoothed = median_filter(np.array(all_preds), size=3)

print(f"准备将 {len(all_preds_smoothed)} 个预测标签写入文件...")

# 导出为 txt 文件（每行一个标签，整数格式）
np.savetxt(OUTPUT_TXT_PATH, all_preds_smoothed, fmt='%d')

print("\n")
print(f"预测完成！测试集标签已成功保存至当前目录下的: 【{OUTPUT_TXT_PATH}】")