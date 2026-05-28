一、代码架构与设计核心 (new4.py)
本模块的代码严格对齐原论文的理论公式，采用模块化设计，核心逻辑如下：

1. 精细化分子特征工程 (Atom and Bond Features)
原子特征 (get_atom_features)：提取 28维 节点特征，包含原子类型 One-hot（10维）、连接度 One-hot（6维）、隐式氢原子数 One-hot（5维）、隐式价 One-hot（6维）及芳香性 Indicator（1维）。

化学键特征 (get_bond_features)：提取 6维 边特征，包含键类型 One-hot（4维）、是否共轭（1维）以及是否在环中（1维）。

2. 图结构构建 (smiles_to_graph)
利用 RDKit 解析标准 SMILES 字符串，将其转化为无向图结构，自动输出包含 x（节点特征矩阵）、edge_index（稀疏邻接矩阵）、edge_attr（边特征矩阵）和 degrees（原子度）的分子图字典。

3. 数据集与标签归一化 (ESOLDataset)
采用经典的分子水溶性评估数据集 ESOL (Delaney)。

自动对目标标签进行 Z-Score 标准化（均值/标准差校准），确保回归任务的梯度平稳收敛。

4. Neural Fingerprint 核心架构 (NeuralFingerprint)
按度独立聚合 (Degree-Specific Weights)：严格遵循原论文设计，由于不同连接度的原子局部拓扑结构差异巨大，模型根据原子的度（0-5）分别路由到 6 组完全独立的线性权重矩阵（h_weights）中进行特征变换。

全局指纹平滑更新：层级节点特征通过 ReLU 激活后，经由独立的指纹发射层映射到全局空间，利用 Softmax 实现局部特征的平滑过渡，并在分子全图上执行加和（sum）汇聚，最终生成 512维 的全局分子指纹向量。

5. 训练与评测机制
大批次梯度累加模拟：为精准复现原论文“10,000 个 Minibatch，每个 Minibatch 包含 100 个分子”的设定，同时规避分子图大小不一导致的复杂 Padding 降效，代码采用单分子前向传播 + 梯度累加（累加至 100 个分子时触发 optimizer.step()）的精细控制机制。

5折交叉验证 (5-Fold Cross-Validation)：内置严格的 KFold 划分（固定随机种子 42），杜绝数据泄露与过拟合，并在训练结束后自动打印包含均值（Mean RMSE）与标准差（Std RMSE）的最终学术报告。

二、启动与运行指南
下载 ESOL 数据集文件 delaney-processed.csv。

打开 new4.py 文件，将 train_model() 函数中的数据集路径修改为你的本地路径：

Python


dataset = ESOLDataset(r"你的本地路径/delaney-processed.csv")
在终端中直接运行复现脚本：

Bash


python new4.py
系统将依次执行 5 折验证，并每隔 500 步打印一次当前 Fold 的训练集与测试集 RMSE 指标。