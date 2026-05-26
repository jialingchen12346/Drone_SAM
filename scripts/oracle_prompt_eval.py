#!/usr/bin/env python3
"""
Oracle Prompt Evaluation v2: Per-class oracle SAM2 prompting with full information.

For each FMB test image, for each present class:
  1. Compute GT bounding box (with margin) as box prompt
  2. Sample positive points from eroded class mask (interior, most representative)
  3. Sample negative points from dilated boundary ring (hardest negatives at decision boundary)
  4. Round 1: box + points → mask_1
  5. Round 2: box + points + mask_1 as mask prompt → refined mask
  6. Stack all 14 class masks, argmax → final prediction

Strict mIoU (absent-score=0.0).
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# FMB constants
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Traffic Light", "Traffic Sign",
    "Vegetation", "Sky", "Person", "Car", "Truck", "Bus",
    "Motorcycle", "Bicycle", "Pole",
]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255


def remap_label(raw_label: np.ndarray) -> np.ndarray:
    out = np.where(raw_label == 0, IGNORE_INDEX, raw_label - 1)
    return out.astype(np.int64)


# ---------------------------------------------------------------------------
# Smart point sampling
# ---------------------------------------------------------------------------
def _bbox_from_mask(mask: np.ndarray, margin: int, h: int, w: int):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h - 1, int(ys.max()) + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w - 1, int(xs.max()) + margin)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# Eight compass directions as (dy, dx) unit vectors
_DIRECTIONS = [
    (-1,  0),  # N
    (-1,  1),  # NE
    ( 0,  1),  # E
    ( 1,  1),  # SE
    ( 1,  0),  # S
    ( 1, -1),  # SW
    ( 0, -1),  # W
    (-1, -1),  # NW
]


def _sample_8boundary_points(class_mask: np.ndarray, gt: np.ndarray, cls_id: int):
    """
    Sample 8 positive + 8 negative points covering the mask boundary.

    Positive: 8 inner-boundary pixels selected deterministically by angle
    from centroid (covering 360° evenly). Uses the mask's eroded boundary
    so points are always ON the mask (unlike ConvexHull for concave shapes).

    Negative: for each positive, steps outward along the ray from centroid
    to find the first non-mask valid pixel.
    """
    h, w = class_mask.shape
    if class_mask.sum() == 0:
        return None, None

    # Inner boundary: mask minus lightly-eroded mask
    from scipy import ndimage
    se = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(class_mask, structure=se, iterations=2)
    inner_boundary = class_mask & (~eroded)
    inner_yx = np.argwhere(inner_boundary)
    if len(inner_yx) == 0:
        inner_yx = np.argwhere(class_mask)
    if len(inner_yx) == 0:
        return None, None

    # Outer boundary for negatives
    dilated = ndimage.binary_dilation(class_mask, structure=se, iterations=2)
    outer_boundary = dilated & (~class_mask) & (gt != IGNORE_INDEX)
    outer_yx = np.argwhere(outer_boundary)

    # Centroid
    ys_all, xs_all = np.where(class_mask)
    cy, cx = ys_all.mean(), xs_all.mean()

    # --- 8 positive points: equally spaced by angle from centroid ---
    angles = np.arctan2(inner_yx[:, 0] - cy, inner_yx[:, 1] - cx)
    n_pts = 8
    pos_xy = []
    for k in range(n_pts):
        target = -np.pi + 2 * np.pi * k / n_pts
        # Find boundary pixel closest to this angle
        d = np.abs(np.arctan2(np.sin(angles - target), np.cos(angles - target)))
        best = np.argmin(d)
        py, px = inner_yx[best]
        pos_xy.append([px, py])  # xy order

    # --- 8 negative points: along the same radial direction, just outside ---
    neg_xy = []
    for px, py in pos_xy:
        dx, dy = px - cx, py - cy
        dist = max(abs(dx), abs(dy), 1)
        dx, dy = dx / dist, dy / dist
        for s in range(1, max(h, w)):
            nx = int(px + dx * s)
            ny = int(py + dy * s)
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                break
            if (not class_mask[ny, nx]) and gt[ny, nx] != IGNORE_INDEX:
                neg_xy.append([float(nx), float(ny)])
                break

    # Deduplicate
    seen = set()
    pos_dedup = []
    for p in pos_xy:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            pos_dedup.append(p)
            seen.add(key)
    neg_dedup = []
    for p in neg_xy:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            neg_dedup.append(p)
            seen.add(key)

    point_coords = np.array(pos_dedup + neg_dedup, dtype=np.float32)
    point_labels = np.array(
        [1] * len(pos_dedup) + [0] * len(neg_dedup), dtype=np.int32
    )
    return point_coords, point_labels


def sample_oracle_prompts(
    gt: np.ndarray,
    cls_id: int,
    box_margin: int,
):
    """
    Sample oracle prompts for a single class using 8-directional boundary coverage.
    Returns (point_coords_xy, point_labels, box_xyxy) or (None, None, None).
    """
    h, w = gt.shape
    class_mask = gt == cls_id

    if class_mask.sum() == 0:
        return None, None, None

    box = _bbox_from_mask(class_mask, margin=box_margin, h=h, w=w)
    coords, labels = _sample_8boundary_points(class_mask, gt, cls_id)

    if coords is None:
        return None, box, None

    return coords, labels, box


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class StrictSegMetric:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        valid = gt != IGNORE_INDEX
        gt_v = gt[valid]
        pd_v = pred[valid]
        keep = pd_v != IGNORE_INDEX
        gt_v, pd_v = gt_v[keep], pd_v[keep]
        np.add.at(self.cm, (gt_v, pd_v), 1)

    def compute(self):
        ious = []
        for c in range(self.num_classes):
            intersection = self.cm[c, c]
            union = self.cm[c, :].sum() + self.cm[:, c].sum() - intersection
            if union == 0:
                ious.append(0.0)
            else:
                ious.append(float(intersection) / float(union))
        return {
            "mIoU": np.mean(ious),
            "per_class_IoU": {CLASS_NAMES[i]: ious[i] for i in range(self.num_classes)},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate_oracle(args):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.sam2_config}")
    print(f"8-directional boundary sampling, box_margin={args.box_margin}")
    print(f"Refinement rounds: {args.refine_rounds}")
    print(f"Seed: {args.seed}")

    # Build SAM2
    print("\n[1/4] Building SAM2 model...")
    sam2 = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    sam2.eval()
    predictor = SAM2ImagePredictor(sam2)
    print("Model built.")

    # Gather test images
    print("\n[2/4] Loading test image list...")
    visible_dir = os.path.join(args.data_root, "test", "Visible")
    label_dir = os.path.join(args.data_root, "test", "Label")

    easy_dir = os.path.join(visible_dir, "easy")
    hard_dir = os.path.join(visible_dir, "hard")

    rgb_files = []
    for d in [easy_dir, hard_dir]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".png"):
                    rgb_files.append(os.path.join(d, f))

    print(f"Found {len(rgb_files)} test images")

    # Evaluate
    print("\n[3/4] Running oracle prompt evaluation...")
    metric = StrictSegMetric(NUM_CLASSES)
    rng = np.random.RandomState(args.seed)

    eval_files = rgb_files[:args.max_images] if args.max_images else rgb_files
    print(f"Evaluating {len(eval_files)} images")

    # Cache total classes present for later absent-class handling
    all_classes_present = set()

    for rgb_path in tqdm(eval_files, desc="Oracle prompt eval"):
        fname = os.path.basename(rgb_path)
        label_path = os.path.join(label_dir, fname)

        img = np.array(Image.open(rgb_path).convert("RGB"))
        raw_label = np.array(Image.open(label_path))
        gt = remap_label(raw_label)
        H, W = gt.shape

        predictor.set_image(img)

        all_logits = []

        for c in range(NUM_CLASSES):
            gt_c = gt == c

            if gt_c.sum() == 0:
                all_logits.append(np.full((H, W), -100.0, dtype=np.float32))
                continue

            all_classes_present.add(c)

            points, labels, box = sample_oracle_prompts(
                gt, c, args.box_margin
            )

            try:
                # Prompt mode:
                # - point: points only
                # - point_box: points + box
                # - point_box_mask: points + box + iterative mask refinement
                use_box = args.prompt_mode in ("point_box", "point_box_mask")
                rounds = 1 if args.prompt_mode != "point_box_mask" else max(1, args.refine_rounds)

                masks, ious, low_res_mask = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box if use_box else None,
                    multimask_output=args.multimask_output,
                    return_logits=True,
                )

                if args.multimask_output:
                    best_idx = int(np.argmax(ious))
                    best_mask_logits = masks[best_idx]
                else:
                    best_idx = 0
                    best_mask_logits = masks[0]

                # Round 2+: mask refinement (keep batch dim for predictor API)
                for _ in range(rounds - 1):
                    best_lr = low_res_mask[best_idx:best_idx+1]  # [1, 256, 256]
                    masks2, ious2, low_res_mask2 = predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        box=box if use_box else None,
                        mask_input=best_lr,
                        multimask_output=args.multimask_output,
                        return_logits=True,
                    )
                    if args.multimask_output:
                        best_idx2 = np.argmax(ious2)
                        best_mask_logits = masks2[best_idx2]
                    else:
                        best_mask_logits = masks2[0]
                    low_res_mask = low_res_mask2  # for next round

                all_logits.append(best_mask_logits.astype(np.float32))

            except Exception as e:
                print(f"\n  WARNING: predict failed for {fname} class {c} ({CLASS_NAMES[c]}): {e}")
                all_logits.append(np.full((H, W), -100.0, dtype=np.float32))

        # Stack and argmax
        all_logits = np.stack(all_logits)  # [14, H, W]
        pred = all_logits.argmax(axis=0).astype(np.int64)

        # Pixels where all classes are very negative → background
        all_bg = all_logits.max(axis=0) < -10.0
        pred[all_bg] = IGNORE_INDEX

        metric.update(pred, gt)
        predictor.reset_predictor()

    # Report
    print(f"\n[4/4] Computing metrics...")
    results = metric.compute()
    present_list = sorted(all_classes_present)
    print(f"Classes present in eval: {[CLASS_NAMES[i] for i in present_list]}")
    print(f"Absent classes (scored 0.0): {[CLASS_NAMES[i] for i in range(NUM_CLASSES) if i not in all_classes_present]}")
    print(f"\n{'='*60}")
    print(f"ORACLE PROMPT RESULTS (seed={args.seed})")
    print(f"  sampling: 8-directional boundary coverage")
    print(f"  prompt_mode={args.prompt_mode}")
    print(f"  box_margin={args.box_margin}, refine_rounds={args.refine_rounds}")
    print(f"  multimask_output={args.multimask_output}")
    print(f"{'='*60}")
    print(f"  Test strict mIoU: {results['mIoU']*100:.2f}")
    print(f"\n  Per-class IoU:")
    for name, iou in results["per_class_IoU"].items():
        print(f"    {name:15s}: {iou*100:5.1f}")

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"oracle_prompt_v3  mIoU={results['mIoU']*100:.2f}  "
                    f"mode={args.prompt_mode}  "
                    f"8-directional  box_margin={args.box_margin}  "
                    f"refine_rounds={args.refine_rounds}  "
                    f"multimask={args.multimask_output}  seed={args.seed}\n\n")
            for name, iou in results["per_class_IoU"].items():
                f.write(f"  {name:15s}: {iou*100:.1f}\n")
        print(f"\nResults saved to {args.out}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Oracle Prompt Evaluation v2")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--data-root", default="/home/jl/dataset/FMB")
    parser.add_argument("--box-margin", type=int, default=32)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument(
        "--prompt-mode",
        default="point_box_mask",
        choices=["point", "point_box", "point_box_mask"],
        help="Prompt composition mode.",
    )
    parser.add_argument("--multimask-output", action="store_true", default=True)
    parser.add_argument("--no-multimask", dest="multimask_output", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not os.path.isabs(args.sam2_checkpoint):
        args.sam2_checkpoint = os.path.join(_PROJECT_ROOT, args.sam2_checkpoint)

    evaluate_oracle(args)


if __name__ == "__main__":
    main()
