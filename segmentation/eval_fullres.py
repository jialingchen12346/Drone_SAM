#!/usr/bin/env python3
"""
Full-resolution evaluation for CACAF model on FMB dataset.

Unlike train_cacaf.py validation (which resizes to 512x512), this script
evaluates at the original image resolution (800x600) to match the baseline
evaluation protocol.

Usage:
    cd /home/jl/Drone-SAM-Adapter
    python segmentation/eval_fullres.py \
        --checkpoint work_dirs/cacaf_fmb_v2/best.pth \
        --split val
"""

import argparse
import os.path as osp
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.amp import autocast

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.datasets.fmb_dataset import (
    CLASSES, NUM_CLASSES, IGNORE_INDEX,
    RGB_MEAN, RGB_STD, THM_MEAN, THM_STD,
)
from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.train_cacaf import SegMetric
import torchvision.transforms.functional as TF


def parse_args():
    p = argparse.ArgumentParser(description="Full-resolution evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--data-root", default="/home/jl/dataset/FMB")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt", default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--no-cacaf", action="store_true", default=False,
                   help="Build model with SimpleFusion instead of CACAF")
    p.add_argument("--no-sagu", action="store_true", default=False,
                   help="Disable SAGU in HGSOAD during model construction")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument(
        "--absent-score",
        type=float,
        default=None,
        help=(
            "Score assigned to absent classes when averaging mIoU/mAcc. "
            "Default None skips absent classes; use 0.0 to match protocols "
            "that count empty classes as zero."
        ),
    )
    return p.parse_args()


def load_sample_list(root, split):
    """Load sample paths for a given split."""
    samples = []
    rgb_dir = osp.join(root, split, "Visible")
    thm_dir = osp.join(root, split, "Infrared")
    lbl_dir = osp.join(root, split, "Label")

    for subset in ("easy", "hard"):
        txt = osp.join(root, f"{split}_{subset}_files.txt")
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


def pad_to_multiple(img_t, multiple=32):
    """Pad tensor [1, C, H, W] so H and W are multiples of `multiple`."""
    _, _, h, w = img_t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img_t, h, w
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
    return img_t, h, w


@torch.no_grad()
def evaluate(model, samples, device, amp_dtype, absent_score=None):
    metric = SegMetric(NUM_CLASSES, IGNORE_INDEX)

    for i, s in enumerate(samples):
        rgb = Image.open(s["rgb"]).convert("RGB")
        thm = Image.open(s["thm"]).convert("RGB")
        lbl = np.array(Image.open(s["lbl"]), dtype=np.int64)

        # Label: 1-indexed -> 0-indexed, 0 -> ignore
        lbl -= 1
        lbl[lbl < 0] = IGNORE_INDEX

        # To tensor + normalize (keep original resolution)
        rgb_t = TF.normalize(TF.to_tensor(rgb), RGB_MEAN, RGB_STD).unsqueeze(0)
        thm_t = TF.normalize(TF.to_tensor(thm), THM_MEAN, THM_STD).unsqueeze(0)

        # Pad to multiple of 32 for encoder compatibility
        rgb_t, orig_h, orig_w = pad_to_multiple(rgb_t)
        thm_t, _, _ = pad_to_multiple(thm_t)

        rgb_t = rgb_t.to(device)
        thm_t = thm_t.to(device)

        with autocast('cuda', dtype=amp_dtype):
            out = model(rgb_t, thm_t, gt=None)

        # Crop back to original size before argmax
        pred = out["pred"][:, :, :orig_h, :orig_w]
        pred = pred.argmax(dim=1).cpu().numpy()[0]

        metric.update(pred, lbl)

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}]")

    return metric.compute(absent_score=absent_score)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Load model
    model = CACafSegmentor(
        sam2_checkpoint=args.sam2_ckpt,
        sam2_config=args.sam2_cfg,
        num_classes=NUM_CLASSES,
        use_cacaf=not args.no_cacaf,
        use_sagu=not args.no_sagu,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # Load samples
    samples = load_sample_list(args.data_root, args.split)
    print(f"Evaluating {len(samples)} images at full resolution ({args.split} split)")

    # Evaluate
    results = evaluate(
        model,
        samples,
        device,
        amp_dtype,
        absent_score=args.absent_score,
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"  mIoU  = {results['mIoU']:.2f}")
    print(f"  mAcc  = {results['mAcc']:.2f}")
    print(f"  aAcc  = {results['aAcc']:.2f}")
    print(f"{'='*50}")
    for cls, iou in zip(CLASSES, results["iou_per_class"]):
        print(f"  {cls:16s}: {iou:.1f}")

    if args.absent_score is None:
        alt_results = evaluate(model, samples, device, amp_dtype, absent_score=0.0)
        print(f"\n[reference] mIoU with absent classes scored as 0.0 = {alt_results['mIoU']:.2f}")


if __name__ == "__main__":
    main()
