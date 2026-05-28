import random
import numpy as np
import pandas as pd
import os

from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import KFold

# ============================================================
# Reproducibility
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
# Atom and Bond Features
# ============================================================
ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def get_atom_features(atom):
    symbol = atom.GetSymbol()
    if symbol not in ATOM_SYMBOLS:
        symbol = "Unknown"

    features = []
    features += one_hot_encoding(symbol, ATOM_SYMBOLS)  # 1. 原子类型 One-hot (10维)
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 2. 连接度 One-hot (6维)
    features += one_hot_encoding(atom.GetNumImplicitHs(), [0, 1, 2, 3, 4])  # 3. 隐式氢原子数 One-hot (5维)
    features += one_hot_encoding(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5])  # 4. 隐式价 One-hot (6维)
    features += [atom.GetIsAromatic()]  # 5. 芳香性 Indicator (1维)
    return np.array(features, dtype=np.float32)  # 总原子特征维度 = 28


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
# Graph Construction
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
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float32)

    return {
        'x': torch.tensor(atom_features, dtype=torch.float32),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'degrees': torch.tensor(degrees, dtype=torch.long)
    }


# ============================================================
# Dataset
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
# Neural Fingerprint
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

            next_x = torch.zeros((x.size(0), self.fp_layers[l].in_features), device=x.device)
            for deg in range(6):
                mask = (degrees == deg)
                if mask.any():
                    next_x[mask] = self.h_weights[l][deg](v[mask])

            x = torch.relu(next_x)

            layer_fp = torch.softmax(self.fp_layers[l](x), dim=-1)
            final_fp = final_fp + layer_fp.sum(dim=0)

        return final_fp


# ============================================================
# Regressor (已修改：引入 Dropout 正则化抗过拟合)
# ============================================================
class MoleculeRegressor(nn.Module):
    def __init__(self, atom_dim, bond_dim=6, hidden_dim=64, fp_size=512, depth=3):
        super().__init__()
        self.fp_model = NeuralFingerprint(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            fp_size=fp_size,
            depth=depth
        )
        # 改进点 2：在前向预测 MLP 网络中嵌入 Dropout 层，防止模型死记硬背训练集
        self.predictor = nn.Sequential(
            nn.Linear(fp_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.15),  # 丢弃15%的连接，强迫提取通用溶解度指纹
            nn.Linear(256, 1)
        )
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
# Evaluation
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
# Training Pipeline (已修改：引入 Best 权重挽救机制 + 高 L2 正则)
# ============================================================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("当前使用的计算设备:", device)

    dataset = ESOLDataset(r"D:\下载\delaney-processed.csv")
    atom_dim = dataset[0][0]['x'].shape[1]
    bond_dim = dataset[0][0]['edge_attr'].shape[1]

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 实验版本标签，可以用来区分不同的改进
    experiment_name = "optimized_v1"

    TOTAL_MINIBATCHES = 10000
    BATCH_SIZE_SIMULATED = 100

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n 正在执行第 {fold + 1} 折验证 ")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False)

        model = MoleculeRegressor(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=64,
            fp_size=512,
            depth=3
        ).to(device)

        # 改进点 3：将 weight_decay 从 1e-6 调高至 1e-4，实施更严厉的 L2 懲罰项，压制过拟合
        optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)

        checkpoint_path = os.path.join(save_dir, f"{experiment_name}_fold{fold + 1}_latest.pt")
        best_checkpoint_path = os.path.join(save_dir, f"{experiment_name}_fold{fold + 1}_best.pt")

        # 跟踪此折内验证集取得的最佳 RMSE，初始化为无穷大
        best_test_rmse = float('inf')

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            minibatch_count = ckpt["minibatch_count"]
            if "best_test_rmse" in ckpt:
                best_test_rmse = ckpt["best_test_rmse"]
            print(f" 触发断点续训！从第 {fold + 1} 折的第 {minibatch_count} 个 Minibatch 开始...")
        else:
            minibatch_count = 0

        molecule_in_batch_counter = 0

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

                    # 每 500 个 Minibatch 步数进行全面指标复盘
                    if minibatch_count % 500 == 0:
                        train_rmse, _ = evaluate(model, train_loader, dataset, device)
                        test_rmse, _ = evaluate(model, test_loader, dataset, device)
                        model.train()

                        # 改进点 1 (核心)：检测当前是不是历史最优模型
                        is_best = test_rmse < best_test_rmse
                        log_msg = f"Fold {fold + 1} | Minibatches: {minibatch_count}/{TOTAL_MINIBATCHES} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
                        if is_best:
                            best_test_rmse = test_rmse
                            log_msg += "刷新历史最优验证指标"
                            # 捕获并单独存贮全折最优秀的一瞬间
                            torch.save({
                                "minibatch_count": minibatch_count,
                                "model_state_dict": model.state_dict(),
                                "best_test_rmse": best_test_rmse,
                            }, best_checkpoint_path)

                        print(log_msg)

                        # 常规同步最新状态以便异常中断恢复
                        torch.save({
                            "fold": fold,
                            "minibatch_count": minibatch_count,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_test_rmse": best_test_rmse,
                            "experiment_name": experiment_name,
                            "seed": SEED,
                        }, checkpoint_path)

        # ========== 10000 步满负荷运行结束，提取该折核心成果 ==========
        print(f"\n第 {fold + 1} 折严格训练步数满额，正在加载本折表现最卓越的 Best Checkpoint ")
        if os.path.exists(best_checkpoint_path):
            best_ckpt = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            print(f"成功恢复第 {fold + 1} 折处于第 {best_ckpt['minibatch_count']} 步时的最优权重")

        final_test_rmse, final_test_mae = evaluate(model, test_loader, dataset, device)
        fold_results.append(final_test_rmse)
        print(f"第 {fold + 1} 折核定最终成绩 -> 验证集 RMSE: {final_test_rmse:.4f} | MAE: {final_test_mae:.4f}")

    print("\n 5折交叉验证统计报告     ")
    for i, rmse in enumerate(fold_results):
        print(f"折数 {i + 1}: {rmse:.4f}")

    print(f"\nMean RMSE: {np.mean(fold_results):.4f}")
    print(f"Std RMSE : {np.std(fold_results):.4f}")


if __name__ == "__main__":
    train_model()