#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Default: use two local GPUs for single-node DDP.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29523}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DIST_BACKEND="${DIST_BACKEND:-auto}"
NCCL_SAFE_5090="${NCCL_SAFE_5090:-1}"
NCCL_PREFLIGHT="${NCCL_PREFLIGHT:-0}"
NCCL_LIB_DIR="${NCCL_LIB_DIR:-}"
PREFLIGHT_VARIANT="${PREFLIGHT_VARIANT:-rrf_dsd}"
PREFLIGHT_IMG_SIZE="${PREFLIGHT_IMG_SIZE:-256}"
PREFLIGHT_BATCH_SIZE="${PREFLIGHT_BATCH_SIZE:-1}"
PREFLIGHT_SAM2_CKPT="${PREFLIGHT_SAM2_CKPT:-${ROOT_DIR}/checkpoints/sam2.1_hiera_large.pt}"
PREFLIGHT_SAM2_CFG="${PREFLIGHT_SAM2_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"

cd "${ROOT_DIR}"

# Ensure child Python processes flush logs line-by-line for remote supervisors.
export PYTHONUNBUFFERED

if [[ -n "${NCCL_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

if [[ "${DIST_BACKEND}" == "nccl" ]]; then
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  if [[ "${NCCL_SAFE_5090}" == "1" ]]; then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  fi
fi

echo "[ddp-launch] backend=${DIST_BACKEND} nproc=${NPROC_PER_NODE} master_port=${MASTER_PORT} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
if [[ "${DIST_BACKEND}" == "nccl" ]]; then
  echo "[ddp-launch] NCCL_SAFE_5090=${NCCL_SAFE_5090} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-unset} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-unset}"
fi

if [[ "${NCCL_PREFLIGHT}" == "1" ]]; then
  PREFLIGHT_BACKEND="${DIST_BACKEND}"
  if [[ "${PREFLIGHT_BACKEND}" == "auto" ]]; then
    PREFLIGHT_BACKEND="nccl"
  fi
  preflight_cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="$((MASTER_PORT + 1))"
    segmentation/tools/ddp_model_preflight.py
    --backend "${PREFLIGHT_BACKEND}"
    --model-variant "${PREFLIGHT_VARIANT}"
    --sam2-ckpt "${PREFLIGHT_SAM2_CKPT}"
    --sam2-cfg "${PREFLIGHT_SAM2_CFG}"
    --image-size "${PREFLIGHT_IMG_SIZE}"
    --batch-size "${PREFLIGHT_BATCH_SIZE}"
  )
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${preflight_cmd[@]}"
fi

train_cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  segmentation/train_cacaf.py
  --dist-backend "${DIST_BACKEND}"
  "$@"
)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${train_cmd[@]}"
