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
OUTPUT_TXT_PATH = "final_submit_labels.txt"

MODEL_PATHS = {
    "v6": r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v6.pth",
    "v8": r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v8.pth",
    "v9": r"C:\Users\lenovo\PycharmProjects\machine_learning_project1\.venv\Scripts\best_sleep_model_v9.pth"
}

# 网格搜索出来的黄金权重
MODEL_WEIGHTS = {
    "v6": 0.15,
    "v8": 0.55,
    "v9": 0.30
}

device = torch.device("cpu")


# =====================================================
# 2. 数据处理与加载
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

# 加载测试集：shuffle=False 严格保证顺序与原数据集完全一致
test_loader = DataLoader(TestDataset(INDEX_PATH_TEST), batch_size=32, shuffle=False)

# =====================================================
# 3. 初始化并加载 3 个有效模型
# =====================================================
models = {}
print("正在加载模型...")
for name, path in MODEL_PATHS.items():
    model = DeepConvNet(chans=CHANNELS, time_point=TIME_POINTS, nb_classes=CLASSES, dropoutRate=0.5).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    models[name] = model
    print(f"模型 {name} 加载成功，将注入 {MODEL_WEIGHTS[name] * 100}% 的决策权重。")

# =====================================================
# 4. 测试集加权软投票推理
# =====================================================
print("\n正在对未知测试集进行集成推理...")
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        # 自动兼容 Dataset 返回 (data, label) 或 单纯 data 的情况
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch

        data = data.to(device)
        data = z_score_normalize(data)

        # 初始化加权概率矩阵 [Batch_Size, Classes]
        ensemble_prob = torch.zeros((data.size(0), CLASSES)).to(device)

        # 累加 Softmax 概率
        for name, model in models.items():
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)
            ensemble_prob += MODEL_WEIGHTS[name] * probs

        pred = torch.argmax(ensemble_prob, dim=1)
        all_preds.extend(pred.cpu().numpy())

# =====================================================
# 5. 生理连续性平滑（必须与验证集完全一致）
# =====================================================
print("正在执行中值滤波生理平滑后处理...")
all_preds_smoothed = median_filter(np.array(all_preds), size=3)

# =====================================================
# 6. 严格依照格式导出为 txt
# =====================================================
print(f"检查输出数据量: 共有 {len(all_preds_smoothed)} 个测试样本预测完毕。")

# 导出：没有表头，没有文件名，纯单列整数
np.savetxt(OUTPUT_TXT_PATH, all_preds_smoothed, fmt='%d')

print("\n" + "=" * 50)
print(f"保存路径：【{OUTPUT_TXT_PATH}】")