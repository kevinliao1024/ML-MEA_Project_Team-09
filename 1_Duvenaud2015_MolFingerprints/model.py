import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralFPLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralFPLayer, self).__init__()
        self.output_dim = output_dim
        # 核心对齐点：为 degree 0 到 5 分别准备权重矩阵 H (Algorithm 2, line 9)
        self.degree_weights = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(6)
        ])

    def forward(self, x, edge_index):
        # x: [N, input_dim], edge_index: [2, E]
        N = x.size(0)

        #增加保护逻辑：如果没有化学键（例如甲烷忽略氢原子后，或单原子离子）
        if edge_index.numel() == 0:
            # 直接根据 Degree 0 处理并返回
            new_x = self.degree_weights[0](x)
            return torch.relu(new_x)

        row, col = edge_index

        # 1. 消息聚合：原子特征 + 邻居特征之和 (Algorithm 2, line 8)
        neighbor_repr = torch.zeros_like(x)
        neighbor_repr.index_add_(0, row, x[col])
        combined = x + neighbor_repr

        # 2. 根据每个原子的 Degree 选择对应的权重矩阵计算
        # 先计算每个节点的度
        degrees = torch.zeros(N, dtype=torch.long, device=x.device)
        degrees.index_add_(0, row, torch.ones(row.size(0), dtype=torch.long, device=x.device))

        new_x = torch.zeros(N, self.output_dim, device=x.device)
        for d in range(6):
            mask = (degrees == d)
            if mask.any():
                new_x[mask] = self.degree_weights[d](combined[mask])

        return torch.relu(new_x)  # 论文推荐使用 ReLU


class NeuralFingerprint(nn.Module):
    def __init__(self, atom_dim, fp_size, depth):
        super(NeuralFingerprint, self).__init__()
        self.depth = depth

        # 每一层的原子更新层 (H 矩阵)
        self.layers = nn.ModuleList([
            NeuralFPLayer(atom_dim, atom_dim) for _ in range(depth)
        ])

        # 每一层映射到指纹位的权重 (W 矩阵) (Algorithm 2, line 11)
        self.output_layers = nn.ModuleList([
            nn.Linear(atom_dim, fp_size) for _ in range(depth + 1)
        ])

    def forward(self, x, edge_index):
        # 初始指纹贡献 (radius 0)
        # 通过 Softmax 模拟哈希索引：让每一行在 fp_size 长度上“激活”某些位
        fp_out = torch.softmax(self.output_layers[0](x), dim=-1)

        # 迭代 radius 1 到 depth
        for i in range(self.depth):
            x = self.layers[i](x, edge_index)
            # 累加每一层的指纹贡献
            fp_out = fp_out + torch.softmax(self.output_layers[i + 1](x), dim=-1)

        # 分子级指纹：所有原子指纹向量的总和 (Algorithm 2, line 13)
        # 形状: [fp_size]
        return fp_out.sum(dim=0)