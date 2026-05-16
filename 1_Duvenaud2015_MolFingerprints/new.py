import random
import numpy as np
import pandas as pd

from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# Feature Utilities
# ============================================================

def one_hot_encoding(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]

    return [x == s for s in allowable_set]

# ============================================================
# Atom Features
# ============================================================

ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F',
    'P', 'Cl', 'Br', 'I',
    'Unknown'
]

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    "OTHER"
]

def get_atom_features(atom):

    symbol = atom.GetSymbol()

    if symbol not in ATOM_SYMBOLS:
        symbol = "Unknown"

    hybridization = atom.GetHybridization()

    if hybridization not in HYBRIDIZATION_TYPES:
        hybridization = "OTHER"

    features = []

    # atom type
    features += one_hot_encoding(
        symbol,
        ATOM_SYMBOLS
    )

    # degree
    features += one_hot_encoding(
        atom.GetDegree(),
        [0,1,2,3,4,5]
    )

    # H count
    features += one_hot_encoding(
        atom.GetTotalNumHs(),
        [0,1,2,3,4]
    )

    # valence
    features += one_hot_encoding(
        atom.GetValence(Chem.ValenceType.IMPLICIT),
        [0, 1, 2, 3, 4, 5]
    )

    # hybridization
    features += one_hot_encoding(
        hybridization,
        HYBRIDIZATION_TYPES
    )

    # aromatic
    features.append(atom.GetIsAromatic())

    # in ring
    features.append(atom.IsInRing())

    # formal charge
    features.append(atom.GetFormalCharge())

    # scaled mass
    features.append(atom.GetMass() * 0.01)

    return np.array(features, dtype=np.float32)

# ============================================================
# Bond Features
# ============================================================

def get_bond_features(bond):

    bt = bond.GetBondType()

    features = [

        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,

        bond.GetIsConjugated(),

        bond.IsInRing()
    ]

    return np.array(features, dtype=np.float32)

# ============================================================
# Graph Construction
# ============================================================

def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atom_features = []

    for atom in mol.GetAtoms():

        atom_features.append(
            get_atom_features(atom)
        )

    atom_features = np.stack(atom_features)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = get_bond_features(bond)

        edge_index.extend([
            [i, j],
            [j, i]
        ])

        edge_attr.extend([
            bf,
            bf
        ])

    # self-loop
    for i in range(len(atom_features)):

        edge_index.append([i, i])

        self_loop = np.zeros(6, dtype=np.float32)
        self_loop[0] = 1

        edge_attr.append(self_loop)

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        np.stack(edge_attr),
        dtype=torch.float32
    )

    return {
        'x': torch.tensor(
            atom_features,
            dtype=torch.float32
        ),
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }

# ============================================================
# Dataset
# ============================================================

class ESOLDataset(Dataset):

    def __init__(self, csv_path):

        df = pd.read_csv(csv_path)

        self.smiles = df['smiles'].values

        labels = df[
            'measured log solubility in mols per litre'
        ].values.astype(np.float32)

        self.mean = labels.mean()
        self.std = labels.std()

        self.labels = (
            labels - self.mean
        ) / self.std

        print("Building graph cache...")

        self.graphs = []

        for smi in self.smiles:

            self.graphs.append(
                smiles_to_graph(smi)
            )

        print("Graph cache completed.")

    def __len__(self):

        return len(self.smiles)

    def __getitem__(self, idx):

        return (
            self.graphs[idx],
            torch.tensor(
                [self.labels[idx]],
                dtype=torch.float32
            )
        )

# ============================================================
# Neural FP Layer
# ============================================================

class NeuralFPLayer(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.edge_mlp = nn.Sequential(

            nn.Linear(6, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

        self.message_mlp = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update_mlp = nn.Sequential(

            nn.Linear(hidden_dim * 2, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.10)

    def forward(self, x, edge_index, edge_attr):

        row, col = edge_index

        edge_emb = self.edge_mlp(edge_attr)

        messages = x[col] + edge_emb

        messages = self.message_mlp(messages)

        agg = torch.zeros_like(x)

        agg.index_add_(0, row, messages)

        combined = torch.cat(
            [x, agg],
            dim=-1
        )

        out = self.update_mlp(combined)

        # residual connection
        out = out + x

        out = self.norm(out)

        out = F.relu(out)

        out = self.dropout(out)

        return out

# ============================================================
# Neural Fingerprint
# ============================================================

class NeuralFingerprint(nn.Module):

    def __init__(
        self,
        atom_dim,
        hidden_dim=128,
        fp_size=1024,
        depth=4
    ):

        super().__init__()

        self.depth = depth

        self.input_proj = nn.Sequential(

            nn.Linear(atom_dim, hidden_dim),

            nn.LayerNorm(hidden_dim),

            nn.ReLU()
        )

        self.layers = nn.ModuleList([

            NeuralFPLayer(hidden_dim)

            for _ in range(depth)
        ])

        self.fp_layers = nn.ModuleList([

            nn.Linear(hidden_dim, fp_size)

            for _ in range(depth + 1)
        ])

    def forward(self, graph):

        x = graph['x']
        edge_index = graph['edge_index']
        edge_attr = graph['edge_attr']

        x = self.input_proj(x)

        fp = torch.softmax(
            self.fp_layers[0](x),
            dim=-1
        )

        for i in range(self.depth):

            x = self.layers[i](
                x,
                edge_index,
                edge_attr
            )

            fp = fp + torch.softmax(
                self.fp_layers[i + 1](x),
                dim=-1
            )

        # global sum pooling
        fp = fp.sum(dim=0)

        return fp

# ============================================================
# Regressor
# ============================================================

class MoleculeRegressor(nn.Module):

    def __init__(
        self,
        atom_dim,
        hidden_dim=128,
        fp_size=1024,
        depth=4
    ):

        super().__init__()

        self.fp_model = NeuralFingerprint(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            fp_size=fp_size,
            depth=depth
        )

        self.predictor = nn.Sequential(

            nn.Linear(fp_size, 256),

            nn.LayerNorm(256),

            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Linear(128, 1)
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

        pred = self.predictor(fp)

        return pred

# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, dataset, device):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for graph, label in loader:

            graph = {
                k: v.squeeze(0).to(device)
                for k, v in graph.items()
            }

            label = label.to(device)

            pred = model(graph)

            pred_real = (
                pred.item() * dataset.std
                + dataset.mean
            )

            label_real = (
                label.item() * dataset.std
                + dataset.mean
            )

            preds.append(pred_real)
            labels.append(label_real)

    preds = np.array(preds)
    labels = np.array(labels)

    rmse = np.sqrt(
        np.mean((preds - labels) ** 2)
    )

    mae = np.mean(
        np.abs(preds - labels)
    )

    return rmse, mae

# ============================================================
# Training
# ============================================================

def train_model():

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print("Using device:", device)

    # ========================================================
    # Dataset
    # ========================================================

    dataset = ESOLDataset(
        r"D:\下载\delaney-processed.csv"
    )

    atom_dim = dataset[0][0]['x'].shape[1]

    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=SEED
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=1,
        shuffle=True
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=1,
        shuffle=False
    )

    # ========================================================
    # Model
    # ========================================================

    model = MoleculeRegressor(
        atom_dim=atom_dim,
        hidden_dim=128,
        fp_size=1024,
        depth=4
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=7e-4,
        weight_decay=1e-6
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=8
    )

    # ========================================================
    # Training
    # ========================================================

    best_rmse = 999

    early_stop_patience = 20
    wait = 0

    print("\nStart Training...")
    print("Target RMSE ≈ 0.52\n")

    for epoch in range(1, 201):

        model.train()

        for graph, label in train_loader:

            graph = {
                k: v.squeeze(0).to(device)
                for k, v in graph.items()
            }

            label = label.squeeze(0).to(device)

            pred = model(graph)

            loss = F.smooth_l1_loss(
                pred,
                label
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

        train_rmse, _ = evaluate(
            model,
            train_loader,
            dataset,
            device
        )

        test_rmse, _ = evaluate(
            model,
            test_loader,
            dataset,
            device
        )

        scheduler.step(test_rmse)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:03d} | "
            f"LR {current_lr:.6f} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Test RMSE: {test_rmse:.4f}"
        )

        if test_rmse < best_rmse:

            best_rmse = test_rmse

            wait = 0

            torch.save(
                model.state_dict(),
                "best_neural_fp.pt"
            )

            print(
                f">>> New Best RMSE: "
                f"{best_rmse:.4f}"
            )

        else:
            wait += 1

        if wait >= early_stop_patience:

            print("\nEarly stopping triggered.")

            break

    print("\n===================================")
    print(f"Best Test RMSE: {best_rmse:.4f}")
    print("===================================")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    train_model()