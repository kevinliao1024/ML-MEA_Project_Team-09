import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np

from feature_extraction import smiles_to_graph
from model import NeuralFingerprint

class ESOLDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.smiles = df['smiles'].values
        # 目标值：measured log solubility in mols per litre
        self.labels = df['measured log solubility in mols per litre'].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        graph = smiles_to_graph(self.smiles[idx])
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return graph, label

def collate_fn(batch):
    # 这是一个简单的处理方式：保持为列表，在模型 forward 里循环
    # 或者使用 torch_geometric 的 Batch 功能（如果你之后想优化速度）
    return batch

#定义预测网络 (Regression Head)
class MoleculeRegressor(nn.Module):
    def __init__(self, model_fp, fp_size, hidden_dim=100):
        super(MoleculeRegressor, self).__init__()
        self.model_fp = model_fp
        self.predictor = nn.Sequential(
            nn.Linear(fp_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # 1. 提取指纹
        fp = self.model_fp(data['x'], data['edge_index'])
        # 2. 回归预测
        return self.predictor(fp)

#训练循环
def train():
    # 1. 核心超参数对齐 (论文 Table 1 设置)
    fp_size = 512  # 指纹长度
    depth = 2  # 卷积深度 (Radius)
    lr = 1e-3  # 学习率
    target_rmse = 0.52

    # 2. 初始化模型
    fp_model = NeuralFingerprint(atom_dim=49, fp_size=fp_size, depth=depth)
    model = MoleculeRegressor(fp_model, fp_size=fp_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    # 3. 加载数据集 (请确保路径正确)
    dataset = ESOLDataset(r"D:\下载\delaney-processed.csv")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"开始训练... 目标 RMSE: {target_rmse}")
    model.train()

    for epoch in range(137):
        total_loss = 0
        optimizer.zero_grad()

        for i, (graph, label) in enumerate(loader):
            if graph is None: continue

            # 将所有 Tensor 的第 0 维（batch 维）去掉
            # 因为 batch_size=1，squeeze(0) 会把 [1, 2, E] 变成 [2, E]
            input_graph = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                           for k, v in graph.items()}

            optimizer.zero_grad()

            # 使用处理后的 input_graph
            output = model(input_graph)

            # label 同样需要处理，确保形状对齐 [1]
            loss = criterion(output.view(-1), label.view(-1)) # 确保都是一维向量
            loss.backward()

            if (i + 1) % 100 == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        # 计算当前 Epoch 的 RMSE
        epoch_rmse = (total_loss / len(loader)) ** 0.5
        print(f"Epoch {epoch + 1}, Current RMSE: {epoch_rmse:.4f}")

        # 如果 RMSE 达到对齐范围，可以提前停止
        if epoch_rmse <= target_rmse:
            print("实验结果已成功对齐论文指标！")
            break


if __name__ == "__main__":
    train()