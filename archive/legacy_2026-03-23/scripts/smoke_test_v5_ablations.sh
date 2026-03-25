#!/usr/bin/env bash
set -euo pipefail

cd /home/jl/Drone-SAM-Adapter
PY=/home/jl/miniconda3/envs/sam2-unet/bin/python
DATA=/home/jl/dataset/FMB

run_one () {
  local name="$1"
  local workdir="$2"
  shift 2
  local extra=("$@")

  mkdir -p "$workdir"
  echo "\n=== $name ==="
  echo "$PY segmentation/train_cacaf.py --data-root $DATA --batch-size 4 --epochs 1 --save-freq 1 --val-freq 1 --bf16 --eval-resize-mode letterbox --cat-max-ratio 0.75 --blur-prob 0.2 --photo-distort --work-dir $workdir ${extra[*]}"

  set +e
  "$PY" segmentation/train_cacaf.py \
    --data-root "$DATA" \
    --batch-size 4 --epochs 1 \
    --save-freq 1 --val-freq 1 \
    --bf16 \
    --eval-resize-mode letterbox \
    --cat-max-ratio 0.75 \
    --blur-prob 0.2 \
    --photo-distort \
    --work-dir "$workdir" \
    "${extra[@]}" \
    > "$workdir/smoke.log" 2>&1
  local ret=$?
  set -e

  local ok="FAIL"
  if [[ $ret -eq 0 && -f "$workdir/epoch_1.pth" && -f "$workdir/best.pth" ]]; then
    ok="PASS"
  fi
  echo -e "$name\tret=$ret\t$ok\t$workdir/smoke.log"
}

echo "Smoke testing 6 v5 ablation configs (1 epoch each)..."

run_one "V5-Full" "work_dirs/v5_smoke_full" --use-dice --use-ohem
run_one "V5-NoCACAF" "work_dirs/v5_smoke_no_cacaf" --use-dice --use-ohem --no-cacaf
run_one "V5-NoSAGU" "work_dirs/v5_smoke_no_sagu" --use-dice --use-ohem --no-sagu
run_one "V5-NoDice" "work_dirs/v5_smoke_no_dice" --use-ohem
run_one "V5-NoOHEM" "work_dirs/v5_smoke_no_ohem" --use-dice
run_one "V5-NoCACAF-NoSAGU" "work_dirs/v5_smoke_no_cacaf_no_sagu" --use-dice --use-ohem --no-cacaf --no-sagu

echo "\nDone. Check each work_dir/smoke.log for details."
