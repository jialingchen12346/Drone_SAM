#!/usr/bin/env python3
"""
快速推理时消融（Zero-shot Ablation）

无需重新训练，直接加载 v3 best.pth，在 forward 时动态旁路各模块，
在 val 集评估 mIoU，几分钟出全套消融结果。

局限性：模型权重是按完整结构优化的，旁路某模块会破坏激活分布，
所以结果偏保守（实际重训后差距会更大）。可作为"快速方向验证"使用。

消融配置：
  full        完整模型（v3 best，基准）
  no_cacaf    旁路 CACAF → 用 RGB+Thermal 特征简单平均
  no_sagu     旁路 SAGU → 跳过通道注意力门控
  no_cacaf_no_sagu  同时旁路两者

用法:
    cd /home/jl/Drone-SAM-Adapter
    conda activate sam2-unet
    python scripts/quick_ablation_eval.py \
        --checkpoint work_dirs/cacaf_fmb_v3/best.pth \
        --split val

输出: 终端打印消融结果表 + work_dirs/ablation_zeroshot/results.csv
"""

import argparse
import csv
import os
import os.path as osp
import sys
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.datasets.fmb_dataset import CLASSES, NUM_CLASSES, IGNORE_INDEX

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
THM_MEAN = [0.5, 0.5, 0.5]
THM_STD  = [0.5, 0.5, 0.5]
INFER_SIZE = 512


# ---------------------------------------------------------------------------
# 推理时 Monkey-patch：动态替换 forward 行为
# ---------------------------------------------------------------------------

def patch_no_cacaf(model):
    """旁路 CACAF：将各 level 的 RGB 和 Thermal 特征直接平均融合。"""
    original_fusion_forward = model.fusion.forward

    def bypass_fusion(rgb_features, aux_features):
        # 对齐通道后平均（利用 CACAF 内部的 align 层，但跳过注意力和 MQA）
        fused = []
        for i, block in enumerate(model.fusion.blocks):
            r = block.align_rgb(rgb_features[i])
            a = block.align_aux(aux_features[i])
            # 等权平均，不经过 MQA 和 cross-attention
            fused.append((r + a) * 0.5)
        B = rgb_features[0].shape[0]
        dummy = torch.full((B, 2), 0.5, device=rgb_features[0].device)
        return fused, [dummy] * 4

    model.fusion.forward = bypass_fusion
    return original_fusion_forward


def restore_cacaf(model, original_forward):
    model.fusion.forward = original_forward


def patch_no_sagu(model):
    """旁路 SAGU：将 ScaleAwareGatingUnit 替换为 Identity。"""
    original_sagu_forwards = {}
    if not hasattr(model.decoder, 'sagu4'):
        return {}  # use_sagu=False 时 SAGU 不存在，无需 patch

    for name in ['sagu1', 'sagu2', 'sagu3', 'sagu4']:
        sagu = getattr(model.decoder, name)
        original_sagu_forwards[name] = sagu.forward
        sagu.forward = lambda x, _sagu=sagu: x  # identity

    return original_sagu_forwards


def restore_sagu(model, original_forwards):
    for name, fwd in original_forwards.items():
        getattr(model.decoder, name).forward = fwd


# ---------------------------------------------------------------------------
# 数据 & 推理
# ---------------------------------------------------------------------------

def load_sample_list(data_root, split):
    samples = []
    rgb_dir = osp.join(data_root, split, "Visible")
    thm_dir = osp.join(data_root, split, "Infrared")
    lbl_dir = osp.join(data_root, split, "Label")
    for subset in ("easy", "hard"):
        txt = osp.join(data_root, f"{split}_{subset}_files.txt")
        if not osp.exists(txt):
            continue
        with open(txt) as f:
            for line in f:
                fname = line.strip()
                if not fname:
                    continue
                samples.append(dict(
                    rgb=osp.join(rgb_dir, subset, fname),
                    thm=osp.join(thm_dir, fname),
                    lbl=osp.join(lbl_dir, fname),
                ))
    return samples


class SegMetric:
    def __init__(self):
        self.confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, pred, gt):
        mask = gt != IGNORE_INDEX
        p = pred[mask]; g = gt[mask]
        valid = (p >= 0) & (p < NUM_CLASSES)
        np.add.at(self.confusion.ravel(), (g * NUM_CLASSES + p)[valid], 1)

    def compute(self):
        cm = self.confusion
        tp = np.diag(cm)
        union = cm.sum(1) + cm.sum(0) - tp
        iou = np.where(union > 0, tp / union, np.nan)
        acc = np.where(cm.sum(1) > 0, tp / cm.sum(1), np.nan)
        return dict(
            mIoU=float(np.nanmean(iou) * 100),
            mAcc=float(np.nanmean(acc) * 100),
            aAcc=float(tp.sum() / cm.sum() * 100),
            iou_per_class=(iou * 100).tolist(),
        )


@torch.no_grad()
def evaluate(model, samples, device, amp_dtype):
    metric = SegMetric()
    for s in samples:
        rgb_pil = Image.open(s["rgb"]).convert("RGB")
        thm_pil = Image.open(s["thm"]).convert("RGB")
        gt_raw  = np.array(Image.open(s["lbl"]), dtype=np.int64)
        gt_np   = gt_raw - 1
        gt_np[gt_np < 0] = IGNORE_INDEX

        sz = INFER_SIZE
        rgb_t = TF.normalize(TF.to_tensor(
            rgb_pil.resize((sz, sz), Image.BILINEAR)), RGB_MEAN, RGB_STD
        ).unsqueeze(0).to(device)
        thm_t = TF.normalize(TF.to_tensor(
            thm_pil.resize((sz, sz), Image.BILINEAR)), THM_MEAN, THM_STD
        ).unsqueeze(0).to(device)

        with torch.autocast("cuda", dtype=amp_dtype):
            out = model(rgb_t, thm_t)
        pred = out["pred"].squeeze(0).argmax(0).cpu().numpy().astype(np.int32)

        # resize pred back to gt size
        if pred.shape != gt_np.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt_np.shape[1], gt_np.shape[0]), Image.NEAREST
                ), dtype=np.int32
            )
        metric.update(pred, gt_np)

    return metric.compute()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    ("A4_full",           False, False, "完整模型（v3 best）"),
    ("A2_no_sagu",        False, True,  "旁路 SAGU（+CACAF，−HGSOAD注意力）"),
    ("A3_no_cacaf",       True,  False, "旁路 CACAF（−融合注意力，+SAGU）"),
    ("A1_no_cacaf_sagu",  True,  True,  "旁路 CACAF+SAGU（简单平均+标准UNet）"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="work_dirs/cacaf_fmb_v3/best.pth")
    p.add_argument("--data-root",  default="/home/jl/dataset/FMB")
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--out-dir",    default="work_dirs/ablation_zeroshot")
    p.add_argument("--sam2-cfg",   default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt",  default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--bf16",       action="store_true", default=True)
    p.add_argument("--no-bf16",    dest="bf16", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    print("加载模型（完整结构，v3 best）...")
    model = CACafSegmentor(
        sam2_checkpoint=args.sam2_ckpt,
        sam2_config=args.sam2_cfg,
        num_classes=NUM_CLASSES,
        use_dice=True,
        use_cacaf=True,
        use_sagu=True,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}, best_mIoU={ckpt.get('best_miou',0):.2f}")

    samples = load_sample_list(args.data_root, args.split)
    print(f"  {args.split} 集: {len(samples)} 张\n")

    results = []

    for config_id, bypass_cacaf, bypass_sagu, desc in ABLATION_CONFIGS:
        print(f"[{config_id}] {desc}")

        # Apply patches
        orig_cacaf = patch_no_cacaf(model) if bypass_cacaf else None
        orig_sagu  = patch_no_sagu(model)  if bypass_sagu  else {}

        metrics = evaluate(model, samples, device, amp_dtype)

        # Restore
        if orig_cacaf is not None:
            restore_cacaf(model, orig_cacaf)
        if orig_sagu:
            restore_sagu(model, orig_sagu)

        results.append((config_id, desc, metrics))
        print(f"  mIoU={metrics['mIoU']:.2f}  mAcc={metrics['mAcc']:.2f}  "
              f"aAcc={metrics['aAcc']:.2f}\n")

    # Summary table
    print("=" * 70)
    print(f"{'配置':<25} {'mIoU':>7} {'Δ mIoU':>8} {'mAcc':>7} {'aAcc':>7}")
    print("-" * 70)
    base_miou = results[0][2]["mIoU"]
    for config_id, desc, m in results:
        delta = m["mIoU"] - base_miou
        sign  = "+" if delta >= 0 else ""
        print(f"{config_id:<25} {m['mIoU']:>7.2f} {sign+f'{delta:.2f}':>8} "
              f"{m['mAcc']:>7.2f} {m['aAcc']:>7.2f}")
    print("=" * 70)
    print("注：推理时消融，结果偏保守（权重未针对旁路结构重新训练）\n")

    # CSV
    csv_path = osp.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "desc", "mIoU", "delta_mIoU", "mAcc", "aAcc"]
                   + [f"iou_{c}" for c in CLASSES])
        for config_id, desc, m in results:
            delta = m["mIoU"] - base_miou
            w.writerow([config_id, desc,
                        f"{m['mIoU']:.2f}", f"{delta:+.2f}",
                        f"{m['mAcc']:.2f}", f"{m['aAcc']:.2f}"]
                       + [f"{v:.2f}" for v in m["iou_per_class"]])
    print(f"CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
