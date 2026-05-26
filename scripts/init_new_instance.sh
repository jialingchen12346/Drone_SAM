#!/bin/bash
# ==============================================================================
# 新 autodl 实例初始化脚本
# 用法: 在 autodl 创建实例，选择 PyTorch 2.x + CUDA 12.x 镜像
#       挂载数据盘后运行: bash init_new_instance.sh
# 耗时: ~5 分钟
# ==============================================================================
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DATA_DISK="${DATA_ROOT}/data"
WORKDIR="${DATA_ROOT}"

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
log()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${RED}[!!]${NC} $*"; }

echo "=== Step 1: 检查数据盘 ==="
if [ ! -d "$DATA_DISK" ]; then
    warn "数据盘未挂载到 $DATA_DISK"
    echo "请在 autodl 控制台创建/挂载数据盘后重新运行"
    exit 1
fi
log "数据盘已挂载"

echo ""
echo "=== Step 2: 检查 Git SSH ==="
if [ ! -f ~/.ssh/id_ed25519 ] && [ ! -f ~/.ssh/id_rsa ]; then
    warn "未检测到 SSH key"
    echo "请手动配置: ssh-keygen -t ed25519 && cat ~/.ssh/id_ed25519.pub"
    echo "将公钥添加到 GitHub Settings > SSH and GPG keys"
    exit 1
fi
# Test GitHub connection
ssh -o StrictHostKeyChecking=no -T git@github.com 2>&1 | grep -q "successfully authenticated" && \
    log "GitHub SSH 连接正常" || \
    warn "GitHub SSH 连接失败，请检查 SSH key 配置"

echo ""
echo "=== Step 3: 检查 PyTorch ==="
source ~/.bashrc 2>/dev/null || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU count: {torch.cuda.device_count()}')"
log "PyTorch 可用"

echo ""
echo "=== Step 4: 克隆仓库 ==="
mkdir -p "$WORKDIR" && cd "$WORKDIR"

if [ ! -d "Drone-SAM-Adapter" ]; then
    git clone git@github.com:jialingchen12346/Drone_SAM.git Drone-SAM-Adapter
fi
cd Drone-SAM-Adapter && git checkout feat/mmsa-baseline-refactor && git pull && cd ..
log "Drone-SAM-Adapter 就绪"

if [ ! -d "SHIFNet" ]; then
    git clone git@github.com:jialingchen12346/SHIFNet.git
fi
log "SHIFNet 就绪"

echo ""
echo "=== Step 5: 安装 Python 依赖 (~3 分钟) ==="
# mmcv 需要特殊处理（mim install 最可靠）
pip install -q openmim
mim install -q mmengine mmcv 2>/dev/null
pip install -q mmsegmentation

# 其余依赖
pip install -q -r Drone-SAM-Adapter/requirements.txt

log "Python 依赖安装完成"

echo ""
echo "=== Step 6: 链接数据集和权重 ==="
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

echo ""
echo "=============================================="
echo "  初始化完成!"
echo "=============================================="
echo ""
echo "快速验证:"
echo "  cd $WORKDIR/SHIFNet && python eval_checkpoint.py --ckpt SHIFTNetepoch/fmb.pth"
echo "  cd $WORKDIR/Drone-SAM-Adapter && python -c 'from segmentation.models.segmentors.mmsa_baseline_segmentor import MMSABaselineSegmentor; print(\"OK\")'"
echo ""
echo "Drone-SAM-Adapter (主仓库):  $WORKDIR/Drone-SAM-Adapter"
echo "SHIFNet (编码器实验):       $WORKDIR/SHIFNet"
echo "数据集:                      $DATA_DISK/datasets/FMB"
echo "权重:                        $DATA_DISK/checkpoints/"
