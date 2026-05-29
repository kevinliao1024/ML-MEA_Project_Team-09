import os
import random
import numpy as np
import pandas as pd

from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import KFold

# ============================================================
# 1. 可复现性保证 (保持一致的随机种子)
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


# ============================================================
# 2. 真实化学特征空间 (对齐官方开源代码 features.py 完整设定)
# ============================================================
# 学术对齐修正：补全了原版代码中的 'B' (硼) 与 'H' (氢)，特征空间扩展为 13 维
ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H', 'Unknown']
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def get_atom_features(atom):
    symbol = atom.GetSymbol()
    if symbol not in ATOM_SYMBOLS:
        symbol = "Unknown"

    features = []
    features += one_hot_encoding(symbol, ATOM_SYMBOLS)  # 1. 原子类型 One-hot (13维)
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 2. 连接度 One-hot (6维)
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 3. 总氢原子数（含显式） (5维)
    features += one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])  # 4. 隐式价 (6维)
    features += [atom.GetIsAromatic()]  # 5. 芳香性 Indicator (1维)
    return np.array(features, dtype=np.float32)  # 总原子特征维度 = 13+6+5+6+1 = 31 维


def get_bond_features(bond):
    b_type = bond.GetBondType()
    if b_type not in BOND_TYPES:
        b_type = BOND_TYPES[-1]

    features = []
    features += one_hot_encoding(b_type, BOND_TYPES)  # 1. 键类型 One-hot (4维)
    features += [bond.GetIsConjugated()]  # 2. 是否共轭 Indicator (1维)
    features += [bond.IsInRing()]  # 3. 是否在环中 Indicator (1维)
    return np.array(features, dtype=np.float32)  # 总边特征维度 = 6


# ============================================================
# 3. 图结构构建
# ============================================================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    degrees = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
        degrees.append(atom.GetDegree())
    atom_features = np.stack(atom_features)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = get_bond_features(bond)

        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([b_feat, b_feat])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return {
        'x': torch.tensor(atom_features, dtype=torch.float32),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'degrees': torch.tensor(degrees, dtype=torch.long)
    }


# ============================================================
# 4. 数据集加载与标准化
# ============================================================
class ESOLDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.smiles = df['smiles'].values
        labels = df['measured log solubility in mols per litre'].values.astype(np.float32)

        self.mean = labels.mean()
        self.std = labels.std()
        self.labels = (labels - self.mean) / self.std

        print("正在构建分子图缓存...")
        self.graphs = []
        for smi in self.smiles:
            self.graphs.append(smiles_to_graph(smi))
        print("分子图缓存构建完成。")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor([self.labels[idx]], dtype=torch.float32)


# ============================================================
# 5. 神经分子指纹 (对齐原论文 Algorithm 1 与官方开源空间维度)
# ============================================================
class NeuralFingerprint(nn.Module):
    def __init__(self, atom_dim, bond_dim=6, hidden_dim=64, fp_size=512, depth=3):
        super().__init__()
        self.depth = depth
        self.fp_size = fp_size
        self.bond_dim = bond_dim

        self.h_weights = nn.ModuleList()
        for l in range(depth):
            in_features = (atom_dim if l == 0 else hidden_dim) + bond_dim
            layer_h = nn.ModuleList([
                nn.Linear(in_features, hidden_dim, bias=False) for _ in range(6)
            ])
            self.h_weights.append(layer_h)

        self.fp_layers = nn.ModuleList([
            nn.Linear(hidden_dim, fp_size, bias=False) for _ in range(depth)
        ])

    def forward(self, graph):
        x = graph['x']
        edge_index = graph['edge_index']
        edge_attr = graph['edge_attr']
        degrees = graph['degrees']

        degrees = torch.clamp(degrees, 0, 5)
        row, col = edge_index

        # 该张量初始化决定了该 Module 目前仅支持单分子图输入(batch_size=1)
        final_fp = torch.zeros(self.fp_size, device=x.device)

        for l in range(self.depth):
            if edge_index.numel() > 0 and edge_index.size(1) > 0:
                neighbor_msg = torch.cat([x[col], edge_attr], dim=-1)
                agg = torch.zeros((x.size(0), neighbor_msg.size(1)), device=x.device)
                agg.index_add_(0, row, neighbor_msg)
            else:
                agg = torch.zeros((x.size(0), x.size(1) + self.bond_dim), device=x.device)

            zeros = torch.zeros((x.size(0), self.bond_dim), device=x.device)
            self_msg = torch.cat([x, zeros], dim=-1)
            v = self_msg + agg

            next_x = torch.zeros((x.size(0), self.h_weights[l][0].out_features), device=x.device)
            for deg in range(6):
                mask = (degrees == deg)
                if mask.any():
                    next_x[mask] = self.h_weights[l][deg](v[mask])

            # 严格学术对齐：原论文 Algorithm 1 隐藏层平滑激活函数必须为 Sigmoid
            x = torch.sigmoid(next_x)

            # 每个原子级别上计算 Softmax 并累加到全局分子指纹向量中
            layer_fp = torch.softmax(self.fp_layers[l](x), dim=-1)
            final_fp = final_fp + layer_fp.sum(dim=0)

        return final_fp


# ============================================================
# 6. 下游回归器 (严格支持 Linear 与 Neural Net 的对照实验设置)
# ============================================================
class MoleculeRegressor(nn.Module):
    def __init__(self, atom_dim, bond_dim=6, hidden_dim=64, fp_size=512, depth=3, predictor_type='neural_net'):
        super().__init__()
        self.fp_model = NeuralFingerprint(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            fp_size=fp_size,
            depth=depth
        )

        # 对应论文 Table 1 的消融基准设置
        if predictor_type == 'linear':
            self.predictor = nn.Linear(fp_size, 1)
        elif predictor_type == 'neural_net':
            self.predictor = nn.Sequential(
                nn.Linear(fp_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        else:
            raise ValueError("predictor_type 必须是 'linear' 或 'neural_net'")

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph):
        fp = self.fp_model(graph)
        pred = self.predictor(fp.unsqueeze(0))
        return pred.squeeze(0)


# ============================================================
# 7. 学术指标评估
# ============================================================
def evaluate(model, loader, dataset, device):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for graph, label in loader:
            graph = {k: v.squeeze(0).to(device) for k, v in graph.items()}
            label = label.squeeze(0).to(device)

            pred = model(graph)
            pred_real = pred.item() * dataset.std + dataset.mean
            label_real = label.item() * dataset.std + dataset.mean

            preds.append(pred_real)
            labels.append(label_real)

    preds = np.array(preds)
    labels = np.array(labels)
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mae = np.mean(np.abs(preds - labels))
    return rmse, mae


# ============================================================
# 8. 严格学术优化 (含自动对齐的 Checkpoint 逻辑)
# ============================================================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("当前使用的计算 device:", device)

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = ESOLDataset(r"D:\下载\delaney-processed.csv")
    atom_dim = dataset[0][0]['x'].shape[1]  # 此时这里会自动读取为 31
    bond_dim = dataset[0][0]['edge_attr'].shape[1]

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    TOTAL_MINIBATCHES = 10000
    BATCH_SIZE_SIMULATED = 100  # 模拟论文中的 Batch 大小

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n正在执行第 {fold + 1} 折验证")

        # 注意：由于前述模型结构设计限制，此处的 batch_size 必须保持为 1
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False)

        model = MoleculeRegressor(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=64,
            fp_size=512,
            depth=3,
            predictor_type='neural_net'
        ).to(device)

        # 严格学术对齐：原论文全篇指定使用 RMSprop，L2 惩罚项严格设为 1e-4
        optimizer = torch.optim.RMSprop(model.parameters(), lr=8e-4, weight_decay=1e-4)

        minibatch_count = 0
        molecule_in_batch_counter = 0
        best_test_rmse = float('inf')

        model.train()
        optimizer.zero_grad()

        while minibatch_count < TOTAL_MINIBATCHES:
            for graph, label in train_loader:
                if minibatch_count >= TOTAL_MINIBATCHES:
                    break

                graph = {k: v.squeeze(0).to(device) for k, v in graph.items()}
                label = label.squeeze(0).to(device)

                pred = model(graph)
                loss = F.mse_loss(pred, label)

                loss = loss / BATCH_SIZE_SIMULATED
                loss.backward()
                molecule_in_batch_counter += 1

                if molecule_in_batch_counter == BATCH_SIZE_SIMULATED:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    minibatch_count += 1
                    molecule_in_batch_counter = 0

                    if minibatch_count % 500 == 0:
                        train_rmse, _ = evaluate(model, train_loader, dataset, device)
                        test_rmse, _ = evaluate(model, test_loader, dataset, device)
                        model.train()
                        print(
                            f"Fold {fold + 1} | Minibatches: {minibatch_count}/{TOTAL_MINIBATCHES} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

                        if test_rmse < best_test_rmse:
                            best_test_rmse = test_rmse
                            best_checkpoint = {
                                'fold': fold + 1,
                                'minibatch_count': minibatch_count,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_rmse': train_rmse,
                                'test_rmse': test_rmse,
                                'dataset_mean': dataset.mean,
                                'dataset_std': dataset.std
                            }
                            ckpt_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold + 1}.pt")
                            torch.save(best_checkpoint, ckpt_path)
                            print(f"   [Checkpoint] 已将当前最优模型存入 -> {ckpt_path}")

        final_test_rmse, _ = evaluate(model, test_loader, dataset, device)
        fold_results.append(final_test_rmse)

        final_checkpoint = {
            'fold': fold + 1,
            'minibatch_count': minibatch_count,
            'model_state_dict': model.state_dict(),
            'final_test_rmse': final_test_rmse,
            'dataset_mean': dataset.mean,
            'dataset_std': dataset.std
        }
        final_ckpt_path = os.path.join(checkpoint_dir, f"final_model_fold_{fold + 1}.pt")
        torch.save(final_checkpoint, final_ckpt_path)
        print(f"[Checkpoint] 已将第 {fold + 1} 折最终收敛模型存入 -> {final_ckpt_path}")
        print(f"第 {fold + 1} 折严格训练结束，验证集最终 RMSE: {final_test_rmse:.4f}")


    print("5折交叉验证统计报告      ")
    for i, rmse in enumerate(fold_results):
        print(f"折数 {i + 1}: {rmse:.4f}")
    print(f"\nMean RMSE: {np.mean(fold_results):.4f}")
    print(f"Std RMSE : {np.std(fold_results):.4f}")


if __name__ == "__main__":
    train_model()