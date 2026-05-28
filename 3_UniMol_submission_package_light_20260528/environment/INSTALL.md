# Uni-Mol 蛋白-配体结合姿态预测 - 依赖安装

# ==========================================
# 方法A: 使用 Docker (推荐 for Windows)
# ==========================================

# 1. 安装 NVIDIA Docker 支持
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. 拉取官方镜像
docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3

# 3. 运行容器
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -w /workspace \
    dptechnology/unimol:latest-pytorch1.11.0-cuda11.3

# ==========================================
# 方法B: 从源码安装 (Linux/WSL2)
# ==========================================

# 1. 安装基础依赖
pip install torch>=1.11.0
pip install fairseq
pip install rdkit==2022.9.3

# 2. 安装 Uni-Core
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
pip install -e .
cd ..

# 3. 克隆 Uni-Mol
git clone https://github.com/deepmodeling/Uni-Mol.git
cd Uni-Mol

# ==========================================
# 方法C: Conda 环境 (推荐)
# ==========================================

conda create -n unimol python=3.8
conda activate unimol

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install fairseq
pip install rdkit==2022.9.3
pip install lmdb pickle5 numpy

# 安装 Uni-Core
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
pip install -e .
cd ..

# 克隆 Uni-Mol
git clone https://github.com/deepmodeling/Uni-Mol.git
