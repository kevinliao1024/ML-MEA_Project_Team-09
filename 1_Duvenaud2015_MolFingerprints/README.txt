Neural-Fingerprint-PyTorch本仓库提供了对 2015 年 NIPS 经典论文 《Convolutional Networks on Graphs for Learning Molecular Fingerprints》 的复现。

本项目在化学特征提取空间（Feature Space）、可变度权重选择（Degree-specific Weights）以及平滑消息传递机制（Message Passing & Pooling）上，均与哈佛大学 HIPS 实验室官方开源代码（基于 Autograd）及原论文 Algorithm 1 实现了对齐。同时，内置了针对 ESOL（Delaney）数据集的 5 折交叉验证（5-Fold Cross-Validation）标准评估管线。

核心对齐特性 (Alignment Checklist)：
严格闭环了原论文的核心设计：真实化学特征空间对齐：完整补全了基准特征中缺失的 B（硼）和 H（氢）原子，原子特征空间精确映射为 31 维，边特征精确映射为 6 维，彻底避免了化学信息截断。
可变度权重（Degree-specific Weights）：严格复现了原论文为不同连接度（0~5）的原子分配独立权重矩阵的设计，用于捕捉不同的局部几何拓扑。
平滑激活与全局无偏累加：隐藏层严格采用原论文指定的 Sigmoid 平滑激活函数。每层指纹更新均通过 Softmax 进行局部特征分布化，随后通过原子级 .sum(dim=0) 无偏累加至全局分子指纹向量中。
精确梯度累加（Batch 仿真）：利用微批次（Mini-batch）梯度累加技术，在单分子图输入架构下，数学等价地实现了原论文要求的 batch_size = 100 联合优化。

说明：
1. 环境依赖 PyTorch 和化学信息学核心库 RDKit。
2. 数据准备下载经典 ESOL 溶解度数据集 delaney-processed.csv。打开 main.py，将 train_model() 函数中的数据集路径修改为你本地的实际存放路径：Pythondataset = ESOLDataset(r"YOUR_PATH_TO/delaney-processed.csv")
3. 运行严格训练管线执行以下命令启动 5 折交叉验证，训练过程中会自动打印每折的 Train/Test RMSE，并保存最优 Checkpoint：Bashpython main.py

核心算法数学映射：
邻域消息聚合 (Neighborhood Aggregation)论文公式：$a_v = x_v + \sum_{w \in N(v)} \text{concat}(x_w, e_{vw})$代码实现：Pythonneighbor_msg = torch.cat([x[col], edge_attr], dim=-1)
agg.index_add_(0, row, neighbor_msg)
v = torch.cat([x, zeros], dim=-1) + agg
(注：通过对中心原子拼接垫零 zeros，在数学结果完全等价的前提下，优雅地解决了 PyTorch 维度对齐的工程痛点)指纹更新与池化 (Fingerprint Update & Pooling)论文公式：$r \leftarrow r + \sum_{v} \text{softmax}(x_v H_l)$代码实现：Pythonlayer_fp = torch.softmax(self.fp_layers[l](x), dim=-1)
final_fp = final_fp + layer_fp.sum(dim=0)

实验配置：predictor_type='linear'：将生成的神经分子指纹直接送入单层线性回归器。predictor_type='neural_net'：后接双层全连接网络（含有 ReLU 激活），对应论文中的标准端到端回归预测器。超参数规范（完全对齐原论文）：优化器：RMSprop基础学习率 (LR)：8e-4L2 正则化 (Weight Decay)：1e-4梯度裁剪 (Grad Norm)：5.0总迭代 Mini-batches：10,000