#!/usr/bin/env python3
"""
SAM 3.1 zero-shot segmentation test on FMB.

Tests: text-only prompt vs text + point prompt, measuring per-class IoU
against FMB test set ground truth.

Usage:
  python research_workspace/scripts/sam3_zero_shot_test.py \
    --checkpoint /path/to/sam3.1_multiplex.pt \
    --data-root /home/jl/dataset/FMB \
    --split test \
    --max-images 20
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))
sys.path.insert(0, "/home/jl/sam3")

from segmentation.datasets.fmb_dataset import CLASSES, NUM_CLASSES, IGNORE_INDEX

# SAM 3.1 classes → FMB classes mapping
# SAM 3.1 prompts are the FMB class names directly — it's open-vocabulary
FMB_CLASS_PROMPTS = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "traffic light",
    4: "traffic sign",
    5: "vegetation",
    6: "sky",
    7: "person",
    8: "car",
    9: "truck",
    10: "bus",
    11: "motorcycle",
    12: "bicycle",
    13: "pole",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="/home/jl/dataset/FMB")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--max-images", type=int, default=20)
    p.add_argument("--confidence-threshold", type=float, default=0.3)
    p.add_argument("--use-points", action="store_true", default=False,
                   help="Add GT mask centroid as geometric prompt")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Build and load SAM 3.1 image model."""
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        enable_segmentation=True,
        enable_inst_interactivity=False,
        load_from_HF=False,
        checkpoint_path=checkpoint_path,
        device="cpu",
        eval_mode=True,
    )
    model.to(device)
    model.eval()
    return model


def load_fmb_samples(data_root: str, split: str, max_images: int):
    """Load FMB test samples (rgb_path, label_path)."""
    samples = []
    for subset in ("easy", "hard"):
        txt = osp.join(data_root, f"{split}_{subset}_files.txt")
        if not osp.exists(txt):
            continue
        rgb_dir = osp.join(data_root, split, "Visible")
        lbl_dir = osp.join(data_root, split, "Label")
        with open(txt) as f:
            for line in f:
                fname = line.strip()
                if not fname:
                    continue
                rgb_p = osp.join(rgb_dir, subset, fname)
                lbl_p = osp.join(lbl_dir, fname)
                if osp.exists(rgb_p) and osp.exists(lbl_p):
                    samples.append((rgb_p, lbl_p))
                    if len(samples) >= max_images:
                        return samples
    return samples


def get_gt_centroid(label, class_id: int):
    """Get centroid point of GT mask for a given class, normalized to [0,1]."""
    mask = (label == class_id)
    if not mask.any():
        return None
    if isinstance(mask, torch.Tensor):
        idx = mask.nonzero()  # [N, ndim]
        ys, xs = idx[:, 0].float(), idx[:, 1].float()
    else:
        ys, xs = mask.nonzero()
    cy = ys.mean().item() / label.shape[0]
    cx = xs.mean().item() / label.shape[1]
    return (cx, cy)


@torch.inference_mode()
def run_sam3_inference(model, image: Image.Image, text_prompt: str,
                       point: tuple | None = None, confidence_threshold: float = 0.3,
                       resolution: int = 1008):
    """Run SAM 3.1 inference with text prompt and optional point.

    Bypasses Sam3Processor to avoid torch/numpy compat issues.
    """
    device = next(model.parameters()).device
    orig_h, orig_w = image.height, image.width

    # Preprocess image
    arr = np.array(image)
    img_tensor = torch.tensor(arr.tolist(), dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0), size=(resolution, resolution), mode="bilinear"
    )
    img_tensor = (img_tensor - 0.5) / 0.5  # normalize to [-1, 1] (SAM 3 norm)
    img_tensor = img_tensor.to(device=device)

    # Forward backbone on image
    backbone_out = model.backbone.forward_image(img_tensor)

    # Forward text encoder
    text_outputs = model.backbone.forward_text([text_prompt], device=device)
    backbone_out.update(text_outputs)

    # Build geometric prompt (empty or with point)
    from sam3.model.geometry_encoders import Prompt as GeoPrompt
    if point is not None:
        cx, cy = point
        box_w, box_h = 0.02, 0.02
        geometric_prompt = GeoPrompt(
            box_embeddings=torch.tensor([[[cx, cy, box_w, box_h]]], device=device),
            box_mask=torch.zeros(1, 1, device=device, dtype=torch.bool),
            box_labels=torch.ones(1, 1, device=device, dtype=torch.long),
            point_embeddings=torch.zeros(0, 1, 2, device=device),
            point_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, 1, device=device, dtype=torch.long),
        )
    else:
        geometric_prompt = GeoPrompt(
            box_embeddings=torch.zeros(0, 1, 4, device=device),
            box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            point_embeddings=torch.zeros(0, 1, 2, device=device),
            point_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, 1, device=device, dtype=torch.long),
        )

    # Build find input
    from sam3.model.data_misc import FindStage
    find_input = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )

    # Run grounding
    outputs = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )

    out_probs = outputs["pred_logits"].sigmoid().squeeze(-1)
    presence_score = outputs.get("presence_logit_dec", torch.zeros_like(out_probs)).sigmoid()
    # presence_score might have different shape, handle carefully
    if presence_score.dim() == 2:
        presence_score = presence_score.squeeze(-1)
    out_scores = (out_probs * presence_score).squeeze()
    if out_scores.dim() == 0:
        out_scores = out_scores.unsqueeze(0)

    out_masks = outputs["pred_masks"].squeeze(0)  # [N, 1, H_pred, W_pred] -> [N, H_pred, W_pred]
    if out_masks.dim() == 3:
        out_masks = out_masks.unsqueeze(1)

    # Filter by confidence
    keep = out_scores > confidence_threshold
    if not keep.any():
        return None

    out_scores = out_scores[keep]
    out_masks = out_masks[keep]

    # Resize masks to original resolution
    from torch.nn.functional import interpolate
    out_masks_resized = interpolate(
        out_masks.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
    ) > 0.5  # [N, 1, H, W]

    masks_th = out_masks_resized.squeeze(1)  # [N, H, W]
    scores_np = out_scores.cpu().tolist()
    if isinstance(scores_np, float):
        scores_np = [scores_np]

    return masks_th, scores_np


def masks_to_semantic(masks_list, scores_list, h, w, num_classes, confidence_threshold):
    """Convert instance masks to a single semantic segmentation map (torch version)."""
    semantic = torch.full((h, w), IGNORE_INDEX, dtype=torch.long)
    assigned = torch.zeros((h, w), dtype=torch.bool)

    # Collect all (class_id, mask, score) tuples
    all_masks = []
    for class_id, (masks, scores) in enumerate(zip(masks_list, scores_list)):
        if masks is None:
            continue
        for mask, score in zip(masks, scores):
            if score >= confidence_threshold:
                all_masks.append((class_id, mask.cpu().bool(), score))

    if not all_masks:
        return semantic

    # Sort by score descending
    all_masks.sort(key=lambda x: x[2], reverse=True)

    # Assign pixels: first-come (highest confidence) wins
    for class_id, mask, score in all_masks:
        new_pixels = mask & ~assigned
        semantic[new_pixels] = class_id
        assigned |= new_pixels

    return semantic


def compute_iou(pred: torch.Tensor, gt: torch.Tensor, num_classes: int):
    """Compute per-class IoU and mIoU."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)
        intersection = (pred_c & gt_c).sum().item()
        union = (pred_c | gt_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union * 100)
    valid = [v for v in ious if v == v]
    return ious, np.mean(valid) if valid else float("nan")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading SAM 3.1 model...")
    model = load_model(args.checkpoint, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    samples = load_fmb_samples(args.data_root, args.split, args.max_images)
    print(f"Testing on {len(samples)} images from {args.split} split")

    all_ious = []
    for rgb_path, lbl_path in tqdm(samples):
        image = Image.open(rgb_path).convert("RGB")
        gt_arr = np.array(Image.open(lbl_path))
        gt_label = torch.tensor(gt_arr.tolist(), dtype=torch.long)

        # Find classes present in GT
        present_classes = [c for c in range(NUM_CLASSES)
                          if (gt_label == c).any() and c in FMB_CLASS_PROMPTS]

        if not present_classes:
            continue

        h, w = gt_label.shape
        masks_per_class = []
        scores_per_class = []

        for class_id in present_classes:
            text_prompt = FMB_CLASS_PROMPTS[class_id]
            point = None
            if args.use_points:
                point = get_gt_centroid(gt_label, class_id)

            result = run_sam3_inference(model, image, text_prompt, point,
                                         confidence_threshold=args.confidence_threshold)
            if result is None:
                masks_per_class.append(None)
                scores_per_class.append(None)
            else:
                masks, scores = result
                masks_per_class.append(masks)
                scores_per_class.append(scores)

        # Build semantic prediction
        pred = masks_to_semantic(masks_per_class, scores_per_class, h, w,
                                 NUM_CLASSES, args.confidence_threshold)

        ious, miou = compute_iou(pred, gt_label, NUM_CLASSES)
        all_ious.append(ious)

    # Aggregate results
    all_ious = np.array(all_ious)  # [N_images, N_classes]
    per_class_mean = np.nanmean(all_ious, axis=0)
    overall_miou = np.nanmean(all_ious)

    print(f"\n{'='*60}")
    print(f"SAM 3.1 Zero-Shot Results (text-only, {len(samples)} images)")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"{'='*60}")
    print(f"{'Class':<20} {'IoU':>6}")
    print(f"{'-'*26}")
    for c, name in enumerate(CLASSES):
        val = per_class_mean[c]
        print(f"{name:<20} {val:5.1f}" if val == val else f"{name:<20}   nan")
    print(f"{'-'*26}")
    print(f"{'mIoU':<20} {overall_miou:5.1f}")
    print(f"{'='*60}")

    # Save
    out_path = osp.join(osp.dirname(__file__), "..", "artifacts", "sam3_zero_shot_results.txt")
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"SAM 3.1 Zero-Shot on FMB {args.split}\n")
        f.write(f"Images: {len(samples)}, Confidence threshold: {args.confidence_threshold}\n")
        f.write(f"Use points: {args.use_points}\n\n")
        for c, name in enumerate(CLASSES):
            val = per_class_mean[c]
            f.write(f"{name:<20} {val:5.1f}\n" if val == val else f"{name:<20}   nan\n")
        f.write(f"\nmIoU: {overall_miou:.1f}\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
