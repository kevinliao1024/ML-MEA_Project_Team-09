import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from rdkit import Chem
from sklearn.model_selection import train_test_split


# ==========================================
# 1. 特征提取模块 (Stage 2)
# ==========================================
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom):
    symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
               'Tl', 'Y', 'Sb', 'Sn', 'Ag', 'Pd', 'In', 'Gd', 'Yb', 'Er', 'U', 'Unknown']

    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), symbols) +  # 31 bits
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +  # 6 bits
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +  # 5 bits
        one_of_k_encoding_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5]) +  # 6 bits
        [atom.GetIsAromatic()]  # 1 bit
    ).astype(np.float32)  # 总计: 31+6+5+6+1 = 49 维


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    # 原子特征 [N, 49]
    x = torch.tensor(np.stack([get_atom_features(a) for a in mol.GetAtoms()]), dtype=torch.float32)

    # 建立邻接表
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return {'x': x, 'edge_index': edge_index}


# ==========================================
# 2. 模型架构 (Stage 3)
# ==========================================
class NeuralFPLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.degree_weights = nn.ModuleList([nn.Linear(dim, dim) for _ in range(6)])

    def forward(self, x, edge_index):
        N = x.size(0)
        out = torch.zeros_like(x)
        # 消息聚合
        neighbor_sum = torch.zeros_like(x)
        if edge_index.numel() > 0:
            row, col = edge_index
            neighbor_sum.index_add_(0, row, x[col])

        combined = x + neighbor_sum

        # 度相关权重
        degrees = torch.zeros(N, dtype=torch.long, device=x.device)
        if edge_index.numel() > 0:
            degrees.index_add_(0, row, torch.ones(row.size(0), dtype=torch.long, device=x.device))

        for d in range(6):
            mask = (degrees == d)
            if mask.any():
                out[mask] = self.degree_weights[d](combined[mask])
        return torch.relu(out)


class NeuralFingerprint(nn.Module):
    def __init__(self, atom_dim, fp_size, depth):
        super().__init__()
        self.layers = nn.ModuleList([NeuralFPLayer(atom_dim) for _ in range(depth)])
        self.output_layers = nn.ModuleList([nn.Linear(atom_dim, fp_size) for _ in range(depth + 1)])

    def forward(self, data):
        x, edge_index = data['x'], data['edge_index']
        # 每一层贡献加和
        fp = torch.softmax(self.output_layers[0](x), dim=-1)
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index)
            fp = fp + torch.softmax(self.output_layers[i + 1](x), dim=-1)
        return fp.sum(dim=0)


class MoleculeRegressor(nn.Module):
    def __init__(self, fp_size):
        super().__init__()
        self.fp_net = NeuralFingerprint(atom_dim=49, fp_size=fp_size, depth=3)
        self.predictor = nn.Sequential(
            nn.Linear(fp_size, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        )

    def forward(self, data):
        fp = self.fp_net(data)
        return self.predictor(fp)


# ==========================================
# 3. 训练与验证逻辑 (Stage 4)
# ==========================================
class ESOLDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.smiles = df['smiles'].values
        labels = df['measured log solubility in mols per litre'].values
        # 关键优化：标签标准化
        self.mean = labels.mean()
        self.std = labels.std()
        self.labels = (labels - self.mean) / self.std

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        return smiles_to_graph(self.smiles[idx]), torch.tensor([self.labels[idx]], dtype=torch.float32)


def train_model():
    # 数据加载
    full_dataset = ESOLDataset(r"D:\下载\delaney-processed.csv")  # 请确认路径
    train_idx, test_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=1, shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1)

    # 初始化
    model = MoleculeRegressor(fp_size=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # 增加正则化
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print("开始对齐实验结果 (目标 RMSE: 0.52)...")
    for epoch in range(100):
        model.train()
        train_sq_error = 0
        for graph, label in train_loader:
            if graph is None: continue
            # 修复 batch_size=1 的维度问题
            input_graph = {k: v.squeeze(0) for k, v in graph.items()}

            pred = model(input_graph)
            loss = F.mse_loss(pred, label.squeeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 转换回原始量级计算 MSE
            unnorm_error = (pred.item() * full_dataset.std) - (label.item() * full_dataset.std)
            train_sq_error += unnorm_error ** 2

        # 验证集评估
        model.eval()
        test_sq_error = 0
        with torch.no_grad():
            for graph, label in test_loader:
                if graph is None: continue
                input_graph = {k: v.squeeze(0) for k, v in graph.items()}
                pred = model(input_graph)
                unnorm_error = (pred.item() * full_dataset.std) - (label.item() * full_dataset.std)
                test_sq_error += unnorm_error ** 2

        train_rmse = (train_sq_error / len(train_loader)) ** 0.5
        test_rmse = (test_sq_error / len(test_loader)) ** 0.5
        scheduler.step(test_rmse)

        print(f"Epoch {epoch + 1:02d} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        if test_rmse < 0.55: print(">>> 接近论文水平！")


if __name__ == "__main__":
    train_model()