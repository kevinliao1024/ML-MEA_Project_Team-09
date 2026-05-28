import numpy as np
from rdkit import Chem
import torch

#定义 One-hot 编码工具函数
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """如果不在集合内，归为最后一类"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

#提取原子特征
def get_atom_features(atom):
    # 1. 原子元素符号 (可根据数据集扩充)
    symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
               'Tl', 'Y', 'Sb', 'Sn', 'Ag', 'Pd', 'In', 'Gd', 'Yb', 'Er', 'U', 'Unknown']

    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), symbols) +  # 元素类型
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +  # 度
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +  # 氢原子数
        one_of_k_encoding_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5]) +  # 价态
        [atom.GetIsAromatic()]  # 芳香性 (1 bit)
    ).astype(np.float32)

#提取化学键特征
def get_bond_features(bond):
    bt = bond.GetBondType()
    return np.array([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]).astype(np.float32)

#将分子转换为分子图数据结构
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # 1. 提取所有原子特征 (维度: [原子数, 49])
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    atom_features = np.stack(atom_features)

    # 2. 建立邻接表和键特征
    # 我们需要记录：哪些原子相连，以及它们之间键的特征
    adj_list = []
    edge_index = []  # 存储键的索引 [2, 2*键数]
    edge_features = []  # 存储键的特征

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = get_bond_features(bond)

        # 因为是无向图，我们要添加双向关系
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_features.append(b_feat)
        edge_features.append(b_feat)

    # 如果分子中没有键（如单个原子），处理边界情况
    if len(edge_index) > 0:
        edge_index = np.array(edge_index).T  # 转置为 [2, E]
        edge_features = np.stack(edge_features)
    else:
        edge_index = np.empty((2, 0))
        edge_features = np.empty((0, 6))  # 6是get_bond_features的维度

    return {
        "x": torch.tensor(atom_features, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_features, dtype=torch.float32)
    }


# 快速测试
test_smiles = "CCO"  # 乙醇
graph = smiles_to_graph(test_smiles)
print(f"原子特征矩阵形状: {graph['x'].shape}")  # 应该是 (3, 约44-50维)