#!/bin/bash
# ==============================================================================
# 新 autodl 实例初始化脚本
# 用法: 创建实例后，挂载数据盘，然后运行此脚本
#       bash init_new_instance.sh
# ==============================================================================
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DATA_DISK="${DATA_ROOT}/data"
WORKDIR="${DATA_ROOT}"

echo "=== Step 1: 检查数据盘 ==="
if [ ! -d "$DATA_DISK" ]; then
    echo "ERROR: 数据盘未挂载到 $DATA_DISK"
    echo "请先在 autodl 控制台创建/挂载数据盘"
    exit 1
fi

echo "=== Step 2: 配置 Git ==="
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "建议先配置 SSH key: ssh-keygen -t ed25519 && cat ~/.ssh/id_ed25519.pub"
    echo "添加到 GitHub Settings > SSH Keys"
fi

echo "=== Step 3: 安装 Python 依赖 ==="
conda init bash 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

# autodl 镜像通常已有 PyTorch，跳过重装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

pip install -q --no-cache-dir \
    opencv-python-headless openmim scipy tensorboard \
    einops timm scikit-image matplotlib tqdm

# 安装 mmseg（核心依赖）
mim install -q mmengine mmcv mmsegmentation 2>/dev/null || \
    pip install mmengine mmcv mmsegmentation

echo "=== Step 4: 克隆仓库 ==="
cd "$WORKDIR"

if [ ! -d "Drone-SAM-Adapter" ]; then
    git clone git@github.com:jialingchen12346/Drone_SAM.git Drone-SAM-Adapter
fi
cd Drone-SAM-Adapter && git checkout feat/mmsa-baseline-refactor && git pull && cd ..

if [ ! -d "SHIFNet" ]; then
    git clone git@github.com:jialingchen12346/SHIFNet.git
fi

echo "=== Step 5: 链接数据集和权重 ==="
# SAM2.1 checkpoint (两个 repo 共用)
ln -sf "$DATA_DISK/checkpoints/sam2.1_hiera_large.pt" \
    Drone-SAM-Adapter/checkpoints/sam2.1_hiera_large.pt
ln -sf "$DATA_DISK/checkpoints/sam2.1_hiera_large.pt" \
    SHIFNet/checkpoints/sam2.1_hiera_large.pt

# SHIFNet 权重
ln -sf "$DATA_DISK/checkpoints/shifnet_fmb.pth" \
    SHIFNet/SHIFTNetepoch/fmb.pth
ln -sf "$DATA_DISK/checkpoints/fmb_class_embedding.pt" \
    SHIFNet/SHIFTNetepoch/fmb_class_embedding.pt

# FMB 数据集
ln -sf "$DATA_DISK/datasets/FMB" \
    Drone-SAM-Adapter/data/FMB
ln -sf "$DATA_DISK/datasets/FMB" \
    SHIFNet/dataset/FMB

echo "=== 初始化完成 ==="
echo ""
echo "快速验证:"
echo "  cd $WORKDIR/SHIFNet && python eval_checkpoint.py --ckpt SHIFTNetepoch/fmb.pth"
echo "  cd $WORKDIR/Drone-SAM-Adapter && python -c 'from segmentation.models.segmentors.mmsa_baseline_segmentor import MMSABaselineSegmentor; print(\"OK\")'"
