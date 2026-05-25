#!/bin/bash
cd /root/Drone-SAM-Adapter
export PYTHONPATH=/root/Drone-SAM-Adapter
mkdir -p /root/autodl-tmp/work_dirs/label_eff_r10_iter_refine2_e30_seed3407_softw_agf_dref_prob
/root/miniconda3/envs/torch211/bin/python segmentation/train_label_efficient.py \
  --data-root /root/autodl-tmp/datasets/FMB \
  --protocol-file /root/Drone-SAM-Adapter/research_workspace/artifacts/label_protocol/fmb_label_splits_seed42.json \
  --label-ratio 10 \
  --batch-size-l 2 --batch-size-u 2 \
  --epochs 30 \
  --seed 3407 \
  --val-model teacher \
  --model-variant mmsa_baseline \
  --sam2-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-ckpt checkpoints/sam2.1_hiera_large.pt \
  --enable-modality-heads --modality-head-weight 0.2 \
  --fusion-use-agreement-map --fusion-agreement-mode argmax \
  --enable-disagreement-refine --disagreement-refine-mode prob --disagreement-refine-weight 0.5 \
  --num-refine-rounds 2 --refine-round-weights 1.0,0.5 \
  --pseudo-use-agreement --pseudo-agreement-policy soft_weight --pseudo-agreement-floor 0.5 \
  --point-index /root/Drone-SAM-Adapter/research_workspace/artifacts/weak_labels/fmb_points_seed42_r10_u/index.json \
  --point-weight 0.2 \
  --unsup-weight 1.0 \
  --bf16 \
  --work-dir /root/autodl-tmp/work_dirs/label_eff_r10_iter_refine2_e30_seed3407_softw_agf_dref_prob \
  2>&1 | tee /root/autodl-tmp/logs/label_eff_r10_iter_refine2_e30_seed3407_softw_agf_dref_prob.log
