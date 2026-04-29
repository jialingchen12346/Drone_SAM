#!/usr/bin/env python3
"""
Test-set evaluation for train_label_efficient checkpoints.

Usage:
  python segmentation/eval_label_eff.py \
    --checkpoint /path/to/best.pth \
    --data-root /root/autodl-tmp/datasets/FMB \
    --split test \
    --absent-score 0.0
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.datasets.fmb_dataset import CLASSES, FMBDataset, IGNORE_INDEX, NUM_CLASSES
from segmentation.train_label_efficient import SegMetric, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--checkpoint-state",
        default="model",
        choices=["model", "teacher"],
        help="State dict inside train_label_efficient checkpoint to evaluate.",
    )
    p.add_argument("--data-root", default="/root/autodl-tmp/datasets/FMB")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument(
        "--model-variant",
        default="mmsa_baseline",
        choices=["mmsa_baseline", "rrf_dsd", "cacaf", "dual_branch_ct"],
    )
    p.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--absent-score", type=float, default=0.0)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--aux-encoder-size", default="tiny", choices=["tiny", "small"])
    p.add_argument("--aux-pretrained-path", default=None,
                   help="Optional local ConvNeXt aux checkpoint path; must match training if aux size differs")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-cacaf", action="store_true", default=False)
    p.add_argument("--no-sagu", action="store_true", default=False)
    p.add_argument("--out", default=None, help="Output txt path (default: next to checkpoint)")
    p.add_argument("--use-dice", action="store_true", default=False)
    p.add_argument("--use-ohem", action="store_true", default=False)
    p.add_argument("--ohem-thresh", type=float, default=0.7)
    p.add_argument("--ohem-min-kept", type=int, default=100000)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--detail-aux-weight", type=float, default=0.2)
    p.add_argument("--eval-resize-mode", default="letterbox", choices=["stretch", "letterbox"])
    p.add_argument("--enable-modality-heads", action="store_true", default=False)
    p.add_argument("--modality-head-weight", type=float, default=0.2)
    p.add_argument("--fusion-use-agreement-map", action="store_true", default=False)
    p.add_argument("--fusion-agreement-mode", default="prob", choices=["argmax", "prob"])
    p.add_argument("--mmsa-fusion-mode", default="mmsa", choices=["mmsa", "naive"])
    p.add_argument("--enable-disagreement-refine", action="store_true", default=False)
    p.add_argument("--disagreement-refine-weight", type=float, default=0.5)
    p.add_argument("--disagreement-refine-mode", default="argmax", choices=["argmax", "prob"])
    p.add_argument("--no-disagreement-refine-gate", action="store_true", default=False)
    p.add_argument("--enable-reliability-guided-refine", action="store_true", default=False)
    p.add_argument("--thermal-prior-injection", action="store_true", default=False)
    p.add_argument("--thermal-prior-init", type=float, default=0.1)
    p.add_argument("--dual-ensemble-mode", default="avg", choices=["avg", "confidence"])
    p.add_argument("--pseudo-use-agreement", action="store_true", default=False)
    p.add_argument("--pseudo-agreement-mode", default="argmax", choices=["argmax", "prob"])
    p.add_argument(
        "--pseudo-agreement-policy",
        default="hard_filter",
        choices=["hard_filter", "soft_weight"],
    )
    p.add_argument("--pseudo-agreement-floor", type=float, default=0.5)
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, device, amp_dtype, absent_score):
    model.eval()
    metric = SegMetric(NUM_CLASSES, IGNORE_INDEX)
    for rgb, thm, gt in loader:
        rgb = rgb.to(device, non_blocking=True)
        thm = thm.to(device, non_blocking=True)
        with autocast("cuda", dtype=amp_dtype):
            out = model(rgb, thm, gt=None)
        pred = out["pred"].argmax(dim=1).cpu().numpy()
        for p, g in zip(pred, gt.numpy()):
            metric.update(p, g)
    return metric.compute(absent_score=absent_score)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = build_model(args, device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.checkpoint_state == "teacher":
        if not isinstance(ckpt, dict) or "teacher" not in ckpt:
            raise KeyError(f"Checkpoint has no teacher state: {args.checkpoint}")
        state = ckpt["teacher"]
    else:
        state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"[eval] loaded {args.checkpoint} state={args.checkpoint_state}")

    ds = FMBDataset(
        args.data_root,
        split=args.split,
        crop_size=args.crop_size,
        augment=False,
        eval_resize_mode=args.eval_resize_mode,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    results = run_eval(model, loader, device, amp_dtype, args.absent_score)

    iou = results["iou_per_class"]
    valid_cls = [c for c, v in zip(CLASSES, iou) if not (v != v)]
    absent_cls = [c for c, v in zip(CLASSES, iou) if (v != v)]

    lines = [
        f"model_variant: {args.model_variant}",
        f"checkpoint: {args.checkpoint}",
        f"checkpoint_state: {args.checkpoint_state}",
        f"split: {args.split}",
        f"present_mIoU: {results.get('present_mIoU', float('nan')):.2f}",
        f"present_mAcc: {results.get('present_mAcc', float('nan')):.2f}",
        f"strict_mIoU: {results['mIoU']:.2f}",
        f"strict_mAcc: {results['mAcc']:.2f}",
        f"aAcc: {results['aAcc']:.2f}",
        f"valid_classes: {len(valid_cls)}/{NUM_CLASSES}",
        f"absent_classes: {absent_cls}",
        "",
        "Per-class IoU (strict absent=0):",
    ]
    for c, v in zip(CLASSES, iou):
        lines.append(f"  {c}: {v:.1f}" if v == v else f"  {c}: nan")

    report = "\n".join(lines)
    print(report)

    out_path = args.out or osp.join(osp.dirname(args.checkpoint), "test_results.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"[eval] saved -> {out_path}")


if __name__ == "__main__":
    main()
