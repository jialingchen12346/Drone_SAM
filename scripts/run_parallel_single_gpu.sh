#!/usr/bin/env bash
set -euo pipefail

# Parallel single-GPU comparison runner:
#   GPU_RRF  -> rrf_dsd
#   GPU_CACAF -> cacaf
#
# Example:
#   PYTHON_BIN=/home/jl/miniconda3/envs/sam2-unet/bin/python \
#   DATA_ROOT=/root/autodl-tmp/datasets/FMB \
#   SAM2_CKPT=/root/Drone-SAM-Adapter/checkpoints/sam2.1_hiera_large.pt \
#   SEEDS=42,43,44 EPOCHS=5 BATCH_SIZE=4 \
#   bash scripts/run_parallel_single_gpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/FMB}"
SAM2_CKPT="${SAM2_CKPT:-${ROOT_DIR}/checkpoints/sam2.1_hiera_large.pt}"
WORK_BASE="${WORK_BASE:-${ROOT_DIR}/work_dirs/parallel_single_gpu}"

GPU_RRF="${GPU_RRF:-0}"
GPU_CACAF="${GPU_CACAF:-1}"
SEEDS="${SEEDS:-42}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CROP_SIZE="${CROP_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_FREQ="${VAL_FREQ:-1}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${WORK_BASE}"

if [[ ! -f "${SAM2_CKPT}" ]]; then
  echo "[error] SAM2 checkpoint not found: ${SAM2_CKPT}"
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[error] DATA_ROOT not found: ${DATA_ROOT}"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/segmentation/train_cacaf.py" ]]; then
  echo "[error] train script missing: ${ROOT_DIR}/segmentation/train_cacaf.py"
  exit 1
fi

to_array() {
  local input="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${input}"
}

extract_first_epoch_loss() {
  local log_file="$1"
  awk '
    $0 ~ /^Epoch 1\/[0-9]+  avg_loss=/ {
      split($0, a, "avg_loss=")
      split(a[2], b, " ")
      print b[1]
      exit
    }
  ' "${log_file}"
}

extract_first_val_miou() {
  local log_file="$1"
  awk '
    $0 ~ /\[val\] mIoU=/ {
      split($0, a, "mIoU=")
      split(a[2], b, " ")
      print b[1]
      exit
    }
  ' "${log_file}"
}

extract_best_miou() {
  local log_file="$1"
  awk '
    $0 ~ /Training complete\. Best val mIoU =/ {
      split($0, a, "= ")
      print a[2]
      val = a[2]
    }
    END {
      if (val == "") print "N/A"
    }
  ' "${log_file}"
}

run_pair_for_seed() {
  local seed="$1"

  local seed_root="${WORK_BASE}/seed_${seed}"
  local rrf_dir="${seed_root}/rrf_dsd"
  local cacaf_dir="${seed_root}/cacaf"
  local rrf_log="${seed_root}/rrf_dsd.log"
  local cacaf_log="${seed_root}/cacaf.log"

  mkdir -p "${rrf_dir}" "${cacaf_dir}"

  local common_args=(
    --data-root "${DATA_ROOT}"
    --sam2-ckpt "${SAM2_CKPT}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --crop-size "${CROP_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --val-freq "${VAL_FREQ}"
    --save-freq "${SAVE_FREQ}"
    --seed "${seed}"
  )

  echo "[run] seed=${seed}  gpu(rrf_dsd)=${GPU_RRF}  gpu(cacaf)=${GPU_CACAF}"

  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_RRF}" \
      "${PYTHON_BIN}" segmentation/train_cacaf.py \
      --model-variant rrf_dsd \
      --work-dir "${rrf_dir}" \
      "${common_args[@]}" ${EXTRA_ARGS} \
      2>&1 | tee "${rrf_log}"
  ) &
  local pid_rrf=$!

  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_CACAF}" \
      "${PYTHON_BIN}" segmentation/train_cacaf.py \
      --model-variant cacaf \
      --work-dir "${cacaf_dir}" \
      "${common_args[@]}" ${EXTRA_ARGS} \
      2>&1 | tee "${cacaf_log}"
  ) &
  local pid_cacaf=$!

  cleanup() {
    echo "[abort] terminating child processes..."
    kill "${pid_rrf}" "${pid_cacaf}" 2>/dev/null || true
  }
  trap cleanup INT TERM

  local rc_rrf=0
  local rc_cacaf=0
  wait "${pid_rrf}" || rc_rrf=$?
  wait "${pid_cacaf}" || rc_cacaf=$?
  trap - INT TERM

  if [[ "${rc_rrf}" -ne 0 || "${rc_cacaf}" -ne 0 ]]; then
    echo "[error] one or more runs failed: rrf_dsd=${rc_rrf}, cacaf=${rc_cacaf}"
    return 1
  fi

  local rrf_loss rrf_miou rrf_best
  local cacaf_loss cacaf_miou cacaf_best
  rrf_loss="$(extract_first_epoch_loss "${rrf_log}")"
  rrf_miou="$(extract_first_val_miou "${rrf_log}")"
  rrf_best="$(extract_best_miou "${rrf_log}")"

  cacaf_loss="$(extract_first_epoch_loss "${cacaf_log}")"
  cacaf_miou="$(extract_first_val_miou "${cacaf_log}")"
  cacaf_best="$(extract_best_miou "${cacaf_log}")"

  local summary="${seed_root}/summary_seed_${seed}.txt"
  {
    echo "seed=${seed}"
    echo "rrf_dsd:first_epoch_loss=${rrf_loss},first_val_mIoU=${rrf_miou},best_mIoU=${rrf_best}"
    echo "cacaf:first_epoch_loss=${cacaf_loss},first_val_mIoU=${cacaf_miou},best_mIoU=${cacaf_best}"
  } > "${summary}"

  echo "[done] seed=${seed}"
  echo "  rrf_dsd: loss1=${rrf_loss}, val1_mIoU=${rrf_miou}, best_mIoU=${rrf_best}"
  echo "  cacaf  : loss1=${cacaf_loss}, val1_mIoU=${cacaf_miou}, best_mIoU=${cacaf_best}"
}

main() {
  local seeds_arr=()
  to_array "${SEEDS}" seeds_arr

  echo "[config] ROOT_DIR=${ROOT_DIR}"
  echo "[config] WORK_BASE=${WORK_BASE}"
  echo "[config] DATA_ROOT=${DATA_ROOT}"
  echo "[config] SAM2_CKPT=${SAM2_CKPT}"
  echo "[config] SEEDS=${SEEDS}"
  echo "[config] EPOCHS=${EPOCHS}, BATCH_SIZE=${BATCH_SIZE}, CROP_SIZE=${CROP_SIZE}"
  echo "[config] NUM_WORKERS=${NUM_WORKERS}, VAL_FREQ=${VAL_FREQ}"

  for seed in "${seeds_arr[@]}"; do
    run_pair_for_seed "${seed}"
  done

  echo "[all_done] summaries:"
  find "${WORK_BASE}" -maxdepth 3 -type f -name 'summary_seed_*.txt' | sort
}

main "$@"
