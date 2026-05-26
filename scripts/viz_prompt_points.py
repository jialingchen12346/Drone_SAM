#!/usr/bin/env python3
"""Visualize oracle prompt points on FMB test images."""
import os, sys, numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Matplotlib Chinese font fix
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Traffic Light", "Traffic Sign",
    "Vegetation", "Sky", "Person", "Car", "Truck", "Bus",
    "Motorcycle", "Bicycle", "Pole",
]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

# Class colors (distinct)
CLASS_COLORS = [
    '#7B68EE', '#9370DB', '#8B4513', '#FF1493', '#FFD700',
    '#228B22', '#87CEEB', '#FF4500', '#4169E1', '#8B0000',
    '#FF8C00', '#00CED1', '#FF69B4', '#808080',
]

# Use the same sampling implementation as oracle_prompt_eval
from scripts.oracle_prompt_eval import _sample_8boundary_points


def visualize(args):
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

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    selected = rgb_files[args.start : args.start + args.n_images]

    for img_idx, rgb_path in enumerate(selected):
        fname = os.path.basename(rgb_path)
        label_path = os.path.join(label_dir, fname)

        img = np.array(Image.open(rgb_path).convert("RGB"))
        raw_label = np.array(Image.open(label_path))
        gt = np.where(raw_label == 0, IGNORE_INDEX, raw_label - 1).astype(np.int64)
        H, W = gt.shape

        present = sorted(set(np.unique(gt)) - {IGNORE_INDEX})

        # Build figure: one row per present class + summary row
        n_present = len(present)
        n_cols = 4  # RGB, GT mask, Points, Prediction (placeholder)
        fig, axes = plt.subplots(
            max(n_present, 1) + 1, n_cols,
            figsize=(n_cols * 4, (max(n_present, 1) + 1) * 3.5),
            squeeze=False,
        )

        all_pred = np.full((H, W), IGNORE_INDEX, dtype=np.int64)

        for row_i, cls_id in enumerate(present):
            class_mask = gt == cls_id
            result = _sample_8boundary_points(class_mask, gt, cls_id)
            if result is not None and result[0] is not None:
                all_coords, all_labels = result
                pos_mask = all_labels == 1
                neg_mask = all_labels == 0
                pos_xy = all_coords[pos_mask] if pos_mask.any() else None
                neg_xy = all_coords[neg_mask] if neg_mask.any() else None
            else:
                pos_xy, neg_xy = None, None

            # Column 0: RGB
            axes[row_i, 0].imshow(img)
            axes[row_i, 0].set_title(f"RGB ({fname})", fontsize=9)
            axes[row_i, 0].axis('off')

            # Column 1: GT mask
            gt_vis = np.zeros((H, W, 3), dtype=np.uint8)
            gt_vis[class_mask] = [0, 255, 0]
            axes[row_i, 1].imshow(img)
            axes[row_i, 1].imshow(gt_vis, alpha=0.4)
            axes[row_i, 1].set_title(f"GT: {CLASS_NAMES[cls_id]}", fontsize=9)
            axes[row_i, 1].axis('off')

            # Column 2: Points overlay
            axes[row_i, 2].imshow(img)
            if pos_xy is not None and len(pos_xy) > 0:
                axes[row_i, 2].scatter(pos_xy[:, 0], pos_xy[:, 1],
                    c='lime', s=50, edgecolors='black', linewidth=0.5, marker='o', zorder=5, label=f'+{len(pos_xy)}')
            if neg_xy is not None and len(neg_xy) > 0:
                axes[row_i, 2].scatter(neg_xy[:, 0], neg_xy[:, 1],
                    c='red', s=40, edgecolors='black', linewidth=0.5, marker='x', zorder=5, label=f'-{len(neg_xy)}')
            axes[row_i, 2].set_title(f"Prompt: {CLASS_NAMES[cls_id]}", fontsize=9)
            axes[row_i, 2].axis('off')
            if pos_xy is not None:
                axes[row_i, 2].legend(fontsize=7, loc='lower right')

            # Column 3: Class mask filled
            mask_fill = np.zeros((H, W, 4), dtype=np.uint8)
            color_hex = CLASS_COLORS[cls_id]
            rgb_tuple = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            mask_fill[class_mask] = [*rgb_tuple, 180]
            axes[row_i, 3].imshow(img)
            axes[row_i, 3].imshow(mask_fill)
            axes[row_i, 3].set_title(f"Mask: {CLASS_NAMES[cls_id]}", fontsize=9)
            axes[row_i, 3].axis('off')

            # Update composite prediction (naive: fill with class)
            all_pred[class_mask] = cls_id

        # Last row: composite overview
        row_i = max(n_present, 1)
        # RGB
        axes[row_i, 0].imshow(img)
        axes[row_i, 0].set_title("Original", fontsize=9)
        axes[row_i, 0].axis('off')

        # GT all
        gt_colored = np.zeros((H, W, 3), dtype=np.uint8)
        for c in present:
            color_hex = CLASS_COLORS[c]
            rgb_tuple = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            gt_colored[gt == c] = rgb_tuple
        axes[row_i, 1].imshow(gt_colored)
        axes[row_i, 1].set_title(f"GT All ({len(present)} classes)", fontsize=9)
        axes[row_i, 1].axis('off')

        # All points overlay
        axes[row_i, 2].imshow(img)
        for cls_id in present:
            class_mask = gt == cls_id
            result = _sample_8boundary_points(class_mask, gt, cls_id)
            if result is not None and result[0] is not None:
                all_coords, all_labels = result
                pos_mask = all_labels == 1
                neg_mask = all_labels == 0
                pos_xy = all_coords[pos_mask] if pos_mask.any() else None
                neg_xy = all_coords[neg_mask] if neg_mask.any() else None
            else:
                pos_xy, neg_xy = None, None
            if pos_xy is not None and len(pos_xy) > 0:
                axes[row_i, 2].scatter(pos_xy[:, 0], pos_xy[:, 1],
                    c=CLASS_COLORS[cls_id], s=40, edgecolors='white', linewidth=0.3, marker='o', zorder=5)
            if neg_xy is not None and len(neg_xy) > 0:
                axes[row_i, 2].scatter(neg_xy[:, 0], neg_xy[:, 1],
                    c=CLASS_COLORS[cls_id], s=25, edgecolors='white', linewidth=0.3, marker='x', zorder=4)
        axes[row_i, 2].set_title(f"All prompts ({len(present)} classes)", fontsize=9)
        axes[row_i, 2].axis('off')

        # Composite prediction
        pred_colored = np.zeros((H, W, 3), dtype=np.uint8)
        for c in present:
            color_hex = CLASS_COLORS[c]
            rgb_tuple = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            pred_colored[all_pred == c] = rgb_tuple
        axes[row_i, 3].imshow(pred_colored)
        axes[row_i, 3].set_title("GT Composite", fontsize=9)
        axes[row_i, 3].axis('off')

        plt.suptitle(f"Oracle Prompt Points — {fname}  ({len(present)} classes)", fontsize=12, y=1.01)
        plt.tight_layout()

        out_path = os.path.join(args.out_dir, f"prompts_{fname}")
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/jl/dataset/FMB")
    p.add_argument("--out-dir", default="/home/jl/Drone-SAM-Adapter/scripts/prompt_viz")
    p.add_argument("--n-images", type=int, default=2)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
