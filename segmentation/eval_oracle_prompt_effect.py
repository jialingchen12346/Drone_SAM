#!/usr/bin/env python3
"""
Oracle point-prompt effect evaluation (no retraining).

Runs three settings on the same checkpoint:
  1) baseline model prediction
  2) oracle prompt fusion with 1 positive + 1 negative point per class
  3) oracle prompt fusion with 3 positive + 3 negative points per class

Notes:
  - Prompts are sampled from GT masks of the evaluation split (oracle protocol).
  - SAM2 runs on the same resized eval image seen by the segmentation model.
  - This is an upper-bound diagnostic, not a deploy-time setting.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segmentation.datasets.fmb_dataset import (
    CLASSES,
    FMBDataset,
    IGNORE_INDEX,
    NUM_CLASSES,
    RGB_MEAN,
    RGB_STD,
)
try:
    from segmentation.train_label_efficient import SegMetric, apply_single_modality_inputs, build_model
except ImportError:
    from segmentation.train_label_efficient import SegMetric, build_model

    def apply_single_modality_inputs(
        rgb: torch.Tensor, thm: torch.Tensor, mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "rgb":
            return rgb, torch.zeros_like(thm)
        if mode == "thermal":
            return torch.zeros_like(rgb), thm
        return rgb, thm


@dataclass(frozen=True)
class OracleMode:
    name: str
    pos_points: int
    neg_points: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate oracle point prompt effect without retraining")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-state", default="model", choices=["model", "teacher"])
    p.add_argument("--data-root", default="/root/autodl-tmp/datasets/FMB")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--model-variant", default="mmsa_baseline",
                   choices=["mmsa_baseline", "rrf_dsd", "cacaf", "dual_branch_ct"])
    p.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--rgb-backbone", default="sam2", choices=["sam2", "sam3"])
    p.add_argument("--sam3-ckpt", default="/home/jl/sam3/sam3.1_multiplex.pt")
    p.add_argument("--absent-score", type=float, default=0.0)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--aux-encoder-size", default="tiny", choices=["tiny", "small"])
    p.add_argument("--aux-pretrained-path", default=None)
    p.add_argument("--single-modality", default="none", choices=["none", "rgb", "thermal"])
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-cacaf", action="store_true", default=False)
    p.add_argument("--no-sagu", action="store_true", default=False)
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
    p.add_argument("--num-refine-rounds", type=int, default=1)
    p.add_argument("--enable-reliability-guided-refine", action="store_true", default=False)
    p.add_argument("--thermal-prior-injection", action="store_true", default=False)
    p.add_argument("--thermal-prior-init", type=float, default=0.1)
    p.add_argument("--enable-gffm", action="store_true", default=False)
    p.add_argument("--enable-mid-correction", action="store_true", default=False)
    p.add_argument("--dual-ensemble-mode", default="avg", choices=["avg", "confidence"])
    p.add_argument("--pseudo-use-agreement", action="store_true", default=False)
    p.add_argument("--pseudo-agreement-mode", default="argmax", choices=["argmax", "prob"])
    p.add_argument("--pseudo-agreement-policy", default="hard_filter", choices=["hard_filter", "soft_weight"])
    p.add_argument("--pseudo-agreement-floor", type=float, default=0.5)

    # Oracle prompt protocol
    p.add_argument("--oracle-modes", default="1p1n,3p3n",
                   help="Comma-separated modes. Supports presets {1p1n,3p3n} or generic pattern XpYn (e.g. 5p5n,8p8n).")
    p.add_argument("--oracle-max-samples", type=int, default=0,
                   help="0 means full split; >0 limits evaluated samples")
    p.add_argument("--oracle-min-class-pixels", type=int, default=8)
    p.add_argument("--oracle-uncert-thresh", type=float, default=0.60,
                   help="Override only where model max prob < thresh when policy=uncertain")
    p.add_argument("--oracle-override-policy", default="uncertain", choices=["uncertain", "all"])
    p.add_argument("--oracle-box-margin", type=int, default=32,
                   help="Expand class bbox by this many pixels for SAM2 prompt box")
    p.add_argument(
        "--oracle-prompt-type",
        default="point_box",
        choices=["point_box", "point_only", "box_only"],
        help="SAM2 prompt composition used per class.",
    )
    p.add_argument(
        "--oracle-add-gt-mask-prompt",
        action="store_true",
        default=False,
        help="Also feed GT binary mask as SAM2 mask-input prompt (oracle upper-bound diagnostic).",
    )
    p.add_argument(
        "--oracle-only-gt-mask-prompt",
        action="store_true",
        default=False,
        help="Use only GT mask-input prompt (ignore point/box prompts).",
    )
    p.add_argument(
        "--oracle-mask-prompt-logit",
        type=float,
        default=20.0,
        help="Absolute logit magnitude for GT mask prompt (+/-value).",
    )
    p.add_argument("--oracle-multimask", action="store_true", default=False,
                   help="Enable SAM2 multimask_output")
    p.add_argument(
        "--oracle-edge-fine-classes",
        default="Traffic Light,Traffic Sign,Person,Pole,Motorcycle,Bicycle",
        help="Comma-separated class names using edge-focused point sampling.",
    )
    p.add_argument(
        "--oracle-edge-width",
        type=int,
        default=3,
        help="Edge band width (pixels) for fine-class edge-focused sampling.",
    )
    p.add_argument(
        "--oracle-class-point-overrides",
        default="",
        help=(
            "Per-class point override. Format: "
            "'Person:8p3n,Traffic Sign:8p2n'. "
            "Overrides global XpYn for listed classes."
        ),
    )
    p.add_argument(
        "--oracle-exclude-classes",
        default="",
        help="Comma-separated class names to exclude from oracle prompt injection.",
    )
    p.add_argument(
        "--oracle-spread-sampling",
        action="store_true",
        default=True,
        help="Use spatially spread point sampling (farthest-point style) instead of random clustering.",
    )
    p.add_argument(
        "--no-oracle-spread-sampling",
        dest="oracle_spread_sampling",
        action="store_false",
    )
    p.add_argument(
        "--oracle-point-layout",
        default="spread",
        choices=["spread", "random", "component_rays8"],
        help=(
            "Point layout strategy: "
            "spread=random candidates + farthest-point, "
            "random=plain random sampling, "
            "component_rays8=per-connected-component 8-direction ray prompts."
        ),
    )
    p.add_argument(
        "--oracle-instance-union-classes",
        default="Person",
        help="Comma-separated classes that use per-instance prompting + union (e.g. Person).",
    )
    p.add_argument(
        "--oracle-instance-refine-iters",
        type=int,
        default=1,
        help="Extra refinement iterations for per-instance prompting (0 disables iterative extra points).",
    )
    p.add_argument(
        "--oracle-instance-refine-min-miss",
        type=int,
        default=16,
        help="If missed GT pixels in a prompted instance >= this value, add extra points and refine.",
    )
    p.add_argument(
        "--oracle-disagreement-roi",
        action="store_true",
        default=False,
        help="Gate prompt override to high-disagreement regions only.",
    )
    p.add_argument(
        "--oracle-disagreement-thresh",
        type=float,
        default=0.5,
        help="Threshold for disagreement ROI/mask when mode=prob (or binary argmax map).",
    )
    p.add_argument(
        "--oracle-disagreement-use-mask-prompt",
        action="store_true",
        default=False,
        help="Also feed disagreement map as SAM2 mask_input prompt.",
    )
    p.add_argument(
        "--oracle-disagreement-mask-logit",
        type=float,
        default=8.0,
        help="Absolute logit magnitude for disagreement mask prompt (+/-value).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _tensor_rgb_to_uint8(rgb_t: torch.Tensor) -> np.ndarray:
    # rgb_t: [3,H,W], normalized by ImageNet mean/std
    arr = rgb_t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    mean = np.asarray(RGB_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(RGB_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray, margin: int, h: int, w: int) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h - 1, int(ys.max()) + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w - 1, int(xs.max()) + margin)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _sample_points_for_class(
    gt: np.ndarray,
    cls_id: int,
    n_pos: int,
    n_neg: int,
    min_pixels: int,
    rng: np.random.Generator,
    edge_focus: bool = False,
    edge_width: int = 3,
    spread_sampling: bool = True,
    point_layout: str = "spread",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    class_mask = gt == cls_id
    pos = np.argwhere(class_mask)
    if pos.shape[0] < min_pixels:
        return None

    valid = gt != IGNORE_INDEX
    neg_all = np.argwhere(valid & (~class_mask))

    if point_layout == "component_rays8":
        pos_ray, neg_ray = _sample_component_rays8_points(gt, class_mask, cls_id, min_pixels)
        if pos_ray.shape[0] > 0:
            pos = pos_ray
        if n_neg > 0 and neg_ray.shape[0] > 0:
            neg = neg_ray
        elif n_neg > 0:
            neg = neg_all
        else:
            neg = np.empty((0, 2), dtype=np.int32)
    elif edge_focus and edge_width > 0:
        # Positive: inside boundary band. Negative: outside boundary ring.
        inner = _erode(class_mask, edge_width)
        pos_band = class_mask & (~inner)
        pos_edge = np.argwhere(pos_band)
        if pos_edge.shape[0] > 0:
            pos = pos_edge

        outer = _dilate(class_mask, edge_width)
        neg_ring = outer & (~class_mask) & valid
        neg_edge = np.argwhere(neg_ring)
        neg = neg_edge if neg_edge.shape[0] > 0 else neg_all
    else:
        neg = neg_all

    if n_neg > 0 and neg.shape[0] == 0:
        return None

    k_pos = min(n_pos, pos.shape[0])
    k_neg = min(max(n_neg, 0), neg.shape[0])
    if k_pos <= 0:
        return None

    use_spread = spread_sampling and point_layout != "random"
    if use_spread:
        pos_pts = _select_spread_points(pos, k_pos, rng)  # [N,2] as (y,x)
        neg_pts = _select_spread_points(neg, k_neg, rng) if k_neg > 0 else np.empty((0, 2), dtype=pos.dtype)
    else:
        pos_idx = rng.choice(pos.shape[0], size=k_pos, replace=False)
        pos_pts = pos[pos_idx]  # [N,2] as (y,x)
        if k_neg > 0:
            neg_idx = rng.choice(neg.shape[0], size=k_neg, replace=False)
            neg_pts = neg[neg_idx]
        else:
            neg_pts = np.empty((0, 2), dtype=pos.dtype)

    # SAM2 uses (x,y)
    xy_pos = np.stack([pos_pts[:, 1], pos_pts[:, 0]], axis=1).astype(np.float32)
    if neg_pts.shape[0] > 0:
        xy_neg = np.stack([neg_pts[:, 1], neg_pts[:, 0]], axis=1).astype(np.float32)
        coords = np.concatenate([xy_pos, xy_neg], axis=0)
        labels = np.concatenate(
            [
                np.ones((xy_pos.shape[0],), dtype=np.int32),
                np.zeros((xy_neg.shape[0],), dtype=np.int32),
            ],
            axis=0,
        )
    else:
        coords = xy_pos
        labels = np.ones((xy_pos.shape[0],), dtype=np.int32)
    return coords, labels


def _dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(steps))):
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        nbr = (
            p[0:-2, 0:-2] | p[0:-2, 1:-1] | p[0:-2, 2:] |
            p[1:-1, 0:-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
            p[2:,   0:-2] | p[2:,   1:-1] | p[2:,   2:]
        )
        out = nbr
    return out


def _erode(mask: np.ndarray, steps: int) -> np.ndarray:
    # Erode via complement dilation.
    out = mask.astype(bool)
    for _ in range(max(0, int(steps))):
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=True)
        nbr = (
            p[0:-2, 0:-2] & p[0:-2, 1:-1] & p[0:-2, 2:] &
            p[1:-1, 0:-2] & p[1:-1, 1:-1] & p[1:-1, 2:] &
            p[2:,   0:-2] & p[2:,   1:-1] & p[2:,   2:]
        )
        out = nbr
    return out


def _select_spread_points(coords: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point sampling on integer pixel coordinates (y, x)."""
    n = int(coords.shape[0])
    if k >= n:
        return coords.copy()
    if k <= 0:
        return np.empty((0, 2), dtype=coords.dtype)

    pts = coords.astype(np.float32, copy=False)
    sel = np.empty((k,), dtype=np.int64)
    sel[0] = int(rng.integers(0, n))

    # min squared distance to selected set for every candidate
    d2 = np.sum((pts - pts[sel[0]]) ** 2, axis=1)
    d2[sel[0]] = -1.0
    for i in range(1, k):
        nxt = int(np.argmax(d2))
        sel[i] = nxt
        nd2 = np.sum((pts - pts[nxt]) ** 2, axis=1)
        d2 = np.minimum(d2, nd2)
        d2[sel[: i + 1]] = -1.0
    return coords[sel]


def _sample_component_rays8_points(
    gt: np.ndarray,
    class_mask: np.ndarray,
    cls_id: int,
    min_pixels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-connected-component 8-ray prompts.

    Positive candidates: last in-class pixel on each ray from component center.
    Negative candidates: first valid non-class pixel after leaving component on each ray.
    Returns coords in (y, x).
    """
    comps = _connected_components(class_mask)
    if not comps:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)
    h, w = class_mask.shape
    valid = gt != IGNORE_INDEX
    dirs = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1),
    ]
    pos_list: List[Tuple[int, int]] = []
    neg_list: List[Tuple[int, int]] = []
    for comp in comps:
        if comp.shape[0] < min_pixels:
            continue
        cy = int(round(float(comp[:, 0].mean())))
        cx = int(round(float(comp[:, 1].mean())))
        # Snap center to nearest pixel inside component.
        d2 = (comp[:, 0] - cy) ** 2 + (comp[:, 1] - cx) ** 2
        cidx = int(np.argmin(d2))
        cy, cx = int(comp[cidx, 0]), int(comp[cidx, 1])

        comp_mask = np.zeros((h, w), dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        max_step = max(h, w)
        for dy, dx in dirs:
            last_inside: Optional[Tuple[int, int]] = None
            entered = False
            first_out: Optional[Tuple[int, int]] = None
            for t in range(1, max_step + 1):
                y = int(round(cy + dy * t))
                x = int(round(cx + dx * t))
                if y < 0 or y >= h or x < 0 or x >= w:
                    break
                if comp_mask[y, x]:
                    entered = True
                    last_inside = (y, x)
                elif entered:
                    first_out = (y, x)
                    break
            if last_inside is not None:
                pos_list.append(last_inside)
            if first_out is not None:
                oy, ox = first_out
                # Walk forward until valid non-class pixel.
                for t2 in range(0, max_step + 1):
                    yy = int(round(oy + dy * t2))
                    xx = int(round(ox + dx * t2))
                    if yy < 0 or yy >= h or xx < 0 or xx >= w:
                        break
                    if not valid[yy, xx]:
                        continue
                    if gt[yy, xx] != cls_id:
                        neg_list.append((yy, xx))
                        break

    def _dedup(points: List[Tuple[int, int]]) -> np.ndarray:
        if not points:
            return np.empty((0, 2), dtype=np.int32)
        arr = np.asarray(points, dtype=np.int32)
        # np.unique sorts; fine for candidate set.
        arr = np.unique(arr, axis=0)
        return arr

    return _dedup(pos_list), _dedup(neg_list)


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """8-connected components for binary mask; returns list of (N,2) coords (y,x)."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    comps: List[np.ndarray] = []
    neigh = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    ys, xs = np.where(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = 1
        coords: List[Tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = 1
                stack.append((ny, nx))
        comps.append(np.asarray(coords, dtype=np.int32))
    return comps


def _sample_points_for_component(
    gt: np.ndarray,
    comp_mask: np.ndarray,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
    spread_sampling: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pos = np.argwhere(comp_mask)
    if pos.shape[0] == 0:
        return None
    valid = gt != IGNORE_INDEX
    neg = np.argwhere(valid & (~comp_mask))
    if n_neg > 0 and neg.shape[0] == 0:
        return None

    k_pos = min(n_pos, pos.shape[0])
    k_neg = min(max(n_neg, 0), neg.shape[0])
    if k_pos <= 0:
        return None

    if spread_sampling:
        pos_pts = _select_spread_points(pos, k_pos, rng)
        neg_pts = _select_spread_points(neg, k_neg, rng) if k_neg > 0 else np.empty((0, 2), dtype=pos.dtype)
    else:
        pos_idx = rng.choice(pos.shape[0], size=k_pos, replace=False)
        pos_pts = pos[pos_idx]
        if k_neg > 0:
            neg_idx = rng.choice(neg.shape[0], size=k_neg, replace=False)
            neg_pts = neg[neg_idx]
        else:
            neg_pts = np.empty((0, 2), dtype=pos.dtype)

    xy_pos = np.stack([pos_pts[:, 1], pos_pts[:, 0]], axis=1).astype(np.float32)
    if neg_pts.shape[0] > 0:
        xy_neg = np.stack([neg_pts[:, 1], neg_pts[:, 0]], axis=1).astype(np.float32)
        coords = np.concatenate([xy_pos, xy_neg], axis=0)
        labels = np.concatenate(
            [
                np.ones((xy_pos.shape[0],), dtype=np.int32),
                np.zeros((xy_neg.shape[0],), dtype=np.int32),
            ],
            axis=0,
        )
    else:
        coords = xy_pos
        labels = np.ones((xy_pos.shape[0],), dtype=np.int32)
    return coords, labels


def _predict_best_mask(
    predictor: SAM2ImagePredictor,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
    mask_input: Optional[np.ndarray],
    multimask_output: bool,
) -> Tuple[np.ndarray, float]:
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=mask_input,
        multimask_output=multimask_output,
        return_logits=False,
        normalize_coords=True,
    )
    if masks.ndim == 2:
        masks = masks[None, ...]
    if np.ndim(scores) == 0:
        scores = np.asarray([scores], dtype=np.float32)
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), float(scores[best_idx])


def _append_extra_point(
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    candidate_mask: np.ndarray,
    label: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    cand = np.argwhere(candidate_mask)
    if cand.shape[0] == 0:
        return point_coords, point_labels
    pick = cand[int(rng.integers(0, cand.shape[0]))]  # (y, x)
    extra_xy = np.asarray([[float(pick[1]), float(pick[0])]], dtype=np.float32)
    extra_lb = np.asarray([int(label)], dtype=np.int32)
    if point_coords is None or point_labels is None:
        return extra_xy, extra_lb
    return np.concatenate([point_coords, extra_xy], axis=0), np.concatenate([point_labels, extra_lb], axis=0)


def _predict_component_with_refine(
    predictor: SAM2ImagePredictor,
    gt: np.ndarray,
    comp_mask: np.ndarray,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
    mask_input: Optional[np.ndarray],
    prompt_type: str,
    multimask_output: bool,
    refine_iters: int,
    refine_min_miss: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    best_mask, best_score = _predict_best_mask(
        predictor=predictor,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=mask_input,
        multimask_output=multimask_output,
    )
    if refine_iters <= 0 or prompt_type == "box_only" or point_coords is None or point_labels is None:
        return best_mask, best_score

    valid = gt != IGNORE_INDEX
    cur_coords, cur_labels = point_coords, point_labels
    for _ in range(refine_iters):
        missed = comp_mask & (~best_mask)
        if int(missed.sum()) < refine_min_miss:
            break
        cur_coords, cur_labels = _append_extra_point(cur_coords, cur_labels, missed, label=1, rng=rng)
        false_pos = best_mask & (~comp_mask) & valid
        if int(false_pos.sum()) > 0:
            cur_coords, cur_labels = _append_extra_point(cur_coords, cur_labels, false_pos, label=0, rng=rng)
        new_mask, new_score = _predict_best_mask(
            predictor=predictor,
            point_coords=cur_coords,
            point_labels=cur_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )
        best_mask, best_score = new_mask, new_score
    return best_mask, best_score


def _build_mask_prompt_logits(mask: np.ndarray, logit_value: float) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None, ...]
    t = F.interpolate(t, size=(256, 256), mode="nearest")
    t = (t * (2.0 * logit_value)) - logit_value  # 1 -> +L, 0 -> -L
    return t.squeeze(0).numpy().astype(np.float32)  # [1, 256, 256]


def _combine_mask_prompts(primary: Optional[np.ndarray], secondary: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    return np.maximum(primary, secondary)


def _build_prompt_map(
    predictor: SAM2ImagePredictor,
    rgb_uint8: np.ndarray,
    gt: np.ndarray,
    mode: OracleMode,
    min_class_pixels: int,
    box_margin: int,
    multimask_output: bool,
    rng: np.random.Generator,
    edge_focus_class_ids: Sequence[int],
    edge_width: int,
    class_point_overrides: Dict[int, Tuple[int, int]],
    spread_sampling: bool,
    excluded_class_ids: Sequence[int],
    prompt_type: str,
    point_layout: str,
    add_gt_mask_prompt: bool,
    only_gt_mask_prompt: bool,
    mask_prompt_logit: float,
    instance_union_class_ids: Sequence[int],
    instance_refine_iters: int,
    instance_refine_min_miss: int,
    disagreement_mask_logits: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gt.shape
    prompt_cls = np.full((h, w), -1, dtype=np.int16)
    prompt_score = np.full((h, w), -1.0, dtype=np.float32)
    predictor.set_image(rgb_uint8)

    present_classes = np.unique(gt)
    present_classes = present_classes[(present_classes >= 0) & (present_classes < NUM_CLASSES)]
    edge_focus_set = set(edge_focus_class_ids)
    excluded_set = set(excluded_class_ids)
    instance_union_set = set(instance_union_class_ids)
    for cls_id in present_classes.tolist():
        if int(cls_id) in excluded_set:
            continue
        class_mask = gt == cls_id
        n_pos, n_neg = class_point_overrides.get(int(cls_id), (mode.pos_points, mode.neg_points))
        if int(cls_id) in instance_union_set:
            comps = _connected_components(class_mask)
            for comp in comps:
                if comp.shape[0] < min_class_pixels:
                    continue
                comp_mask = np.zeros_like(class_mask, dtype=bool)
                comp_mask[comp[:, 0], comp[:, 1]] = True
                sampled_comp = _sample_points_for_component(
                    gt=gt,
                    comp_mask=comp_mask,
                    n_pos=n_pos,
                    n_neg=n_neg,
                    rng=rng,
                    spread_sampling=spread_sampling and point_layout != "random",
                )
                if sampled_comp is None:
                    continue
                comp_coords, comp_labels = sampled_comp
                comp_box = _bbox_from_mask(comp_mask, margin=box_margin, h=h, w=w)
                if only_gt_mask_prompt:
                    use_point_coords, use_point_labels, use_box = None, None, None
                elif prompt_type == "point_only":
                    use_point_coords, use_point_labels, use_box = comp_coords, comp_labels, None
                elif prompt_type == "box_only":
                    use_point_coords, use_point_labels, use_box = None, None, comp_box
                elif prompt_type == "point_box":
                    use_point_coords, use_point_labels, use_box = comp_coords, comp_labels, comp_box
                else:
                    raise ValueError(f"Unsupported oracle_prompt_type: {prompt_type}")
                use_mask_input = None
                if add_gt_mask_prompt or only_gt_mask_prompt:
                    use_mask_input = _build_mask_prompt_logits(comp_mask, mask_prompt_logit)
                use_mask_input = _combine_mask_prompts(use_mask_input, disagreement_mask_logits)
                best_mask, best_score = _predict_component_with_refine(
                    predictor=predictor,
                    gt=gt,
                    comp_mask=comp_mask,
                    point_coords=use_point_coords,
                    point_labels=use_point_labels,
                    box=use_box,
                    mask_input=use_mask_input,
                    prompt_type=prompt_type,
                    multimask_output=multimask_output,
                    refine_iters=instance_refine_iters,
                    refine_min_miss=instance_refine_min_miss,
                    rng=rng,
                )
                update = best_mask & (best_score > prompt_score)
                prompt_cls[update] = int(cls_id)
                prompt_score[update] = best_score
            continue

        sampled = _sample_points_for_class(
            gt=gt,
            cls_id=int(cls_id),
            n_pos=n_pos,
            n_neg=n_neg,
            min_pixels=min_class_pixels,
            rng=rng,
            edge_focus=int(cls_id) in edge_focus_set,
            edge_width=edge_width,
            spread_sampling=spread_sampling,
            point_layout=point_layout,
        )
        if sampled is None:
            continue
        point_coords, point_labels = sampled
        box = _bbox_from_mask(class_mask, margin=box_margin, h=h, w=w)
        if only_gt_mask_prompt:
            use_point_coords, use_point_labels, use_box = None, None, None
        elif prompt_type == "point_only":
            use_point_coords, use_point_labels, use_box = point_coords, point_labels, None
        elif prompt_type == "box_only":
            use_point_coords, use_point_labels, use_box = None, None, box
        elif prompt_type == "point_box":
            use_point_coords, use_point_labels, use_box = point_coords, point_labels, box
        else:
            raise ValueError(f"Unsupported oracle_prompt_type: {prompt_type}")
        use_mask_input = None
        if add_gt_mask_prompt or only_gt_mask_prompt:
            use_mask_input = _build_mask_prompt_logits(class_mask, mask_prompt_logit)
        use_mask_input = _combine_mask_prompts(use_mask_input, disagreement_mask_logits)
        best_mask, best_score = _predict_best_mask(
            predictor=predictor,
            point_coords=use_point_coords,
            point_labels=use_point_labels,
            box=use_box,
            mask_input=use_mask_input,
            multimask_output=multimask_output,
        )
        update = best_mask & (best_score > prompt_score)
        prompt_cls[update] = int(cls_id)
        prompt_score[update] = best_score

    return prompt_cls, prompt_score


def _apply_prompt_map(
    pred_base: np.ndarray,
    conf_max: np.ndarray,
    prompt_cls: np.ndarray,
    override_policy: str,
    uncert_thresh: float,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    pred = pred_base.copy()
    valid = prompt_cls >= 0
    if override_policy == "uncertain":
        override = valid & (conf_max < uncert_thresh)
    else:
        override = valid
    if roi_mask is not None:
        override = override & roi_mask
    pred[override] = prompt_cls[override].astype(np.int64)
    return pred


def _format_report(
    args: argparse.Namespace,
    baseline: dict,
    mode_results: Dict[str, dict],
) -> str:
    lines: List[str] = []
    lines.append(f"checkpoint: {args.checkpoint}")
    lines.append(f"checkpoint_state: {args.checkpoint_state}")
    lines.append(f"split: {args.split}")
    lines.append(f"model_variant: {args.model_variant}")
    lines.append(
        "oracle_protocol: "
        f"override={args.oracle_override_policy}, "
        f"uncert_thresh={args.oracle_uncert_thresh:.2f}, "
        f"min_class_pixels={args.oracle_min_class_pixels}, "
        f"edge_width={args.oracle_edge_width}, "
        f"edge_fine_classes={args.oracle_edge_fine_classes}, "
        f"class_overrides={args.oracle_class_point_overrides or 'none'}, "
        f"exclude_classes={args.oracle_exclude_classes or 'none'}, "
        f"spread_sampling={args.oracle_spread_sampling}, "
        f"point_layout={args.oracle_point_layout}, "
        f"instance_union_classes={args.oracle_instance_union_classes or 'none'}, "
        f"instance_refine_iters={args.oracle_instance_refine_iters}, "
        f"instance_refine_min_miss={args.oracle_instance_refine_min_miss}, "
        f"disagreement_roi={args.oracle_disagreement_roi}, "
        f"disagreement_thresh={args.oracle_disagreement_thresh:.2f}, "
        f"disagreement_use_mask_prompt={args.oracle_disagreement_use_mask_prompt}, "
        f"disagreement_mask_logit={args.oracle_disagreement_mask_logit:.1f}, "
        f"add_gt_mask_prompt={args.oracle_add_gt_mask_prompt}, "
        f"only_gt_mask_prompt={args.oracle_only_gt_mask_prompt}, "
        f"mask_prompt_logit={args.oracle_mask_prompt_logit:.1f}, "
        f"prompt_type={args.oracle_prompt_type}, "
        f"box_margin={args.oracle_box_margin}, "
        f"multimask={args.oracle_multimask}"
    )
    lines.append("")
    lines.append(f"baseline strict_mIoU: {baseline['mIoU']:.2f}")
    lines.append(f"baseline strict_mAcc: {baseline['mAcc']:.2f}")
    lines.append("")

    for mode_name, res in mode_results.items():
        delta = res["mIoU"] - baseline["mIoU"]
        lines.append(f"{mode_name} strict_mIoU: {res['mIoU']:.2f}  (delta {delta:+.2f})")
        lines.append(f"{mode_name} strict_mAcc: {res['mAcc']:.2f}")
        lines.append("")

    base_iou = np.asarray(baseline["iou_per_class"], dtype=np.float32)
    lines.append("Per-class IoU and delta vs baseline:")
    for i, cls_name in enumerate(CLASSES):
        b = base_iou[i]
        b_str = f"{b:.1f}" if b == b else "nan"
        chunk = [f"{cls_name}: base={b_str}"]
        for mode_name, res in mode_results.items():
            m = float(res["iou_per_class"][i])
            m_str = f"{m:.1f}" if m == m else "nan"
            d = m - b if (m == m and b == b) else float("nan")
            d_str = f"{d:+.1f}" if d == d else "nan"
            chunk.append(f"{mode_name}={m_str} ({d_str})")
        lines.append("  " + " | ".join(chunk))
    return "\n".join(lines)


def _parse_oracle_modes(spec: str) -> List[OracleMode]:
    preset = {
        "1p1n": OracleMode(name="1p1n", pos_points=1, neg_points=1),
        "3p3n": OracleMode(name="3p3n", pos_points=3, neg_points=3),
    }
    out: List[OracleMode] = []
    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    seen = set()
    for name in names:
        if name in preset:
            mode = preset[name]
        else:
            m = re.fullmatch(r"(\d+)p(\d+)n", name)
            if not m:
                raise ValueError(
                    f"Unsupported oracle mode '{name}'. Use preset (1p1n,3p3n) or pattern XpYn (e.g. 5p5n)."
                )
            p_cnt = int(m.group(1))
            n_cnt = int(m.group(2))
            if p_cnt <= 0 or n_cnt <= 0:
                raise ValueError(f"Oracle mode '{name}' must have positive counts.")
            mode = OracleMode(name=f"{p_cnt}p{n_cnt}n", pos_points=p_cnt, neg_points=n_cnt)
        if mode.name in seen:
            continue
        seen.add(mode.name)
        out.append(mode)
    if not out:
        raise ValueError("No oracle modes selected.")
    return out


def _parse_edge_fine_classes(spec: str) -> List[int]:
    name_to_id = {name.lower(): idx for idx, name in enumerate(CLASSES)}
    ids: List[int] = []
    for raw in [s.strip() for s in spec.split(",") if s.strip()]:
        key = raw.lower()
        if key not in name_to_id:
            raise ValueError(
                f"Unknown class '{raw}' in --oracle-edge-fine-classes. Valid: {', '.join(CLASSES)}"
            )
        cid = name_to_id[key]
        if cid not in ids:
            ids.append(cid)
    return ids


def _parse_class_list(spec: str) -> List[int]:
    if not spec.strip():
        return []
    name_to_id = {name.lower(): idx for idx, name in enumerate(CLASSES)}
    out: List[int] = []
    for raw in [s.strip() for s in spec.split(",") if s.strip()]:
        key = raw.lower()
        if key not in name_to_id:
            raise ValueError(f"Unknown class '{raw}'. Valid: {', '.join(CLASSES)}")
        cid = name_to_id[key]
        if cid not in out:
            out.append(cid)
    return out


def _parse_class_point_overrides(spec: str) -> Dict[int, Tuple[int, int]]:
    if not spec.strip():
        return {}
    name_to_id = {name.lower(): idx for idx, name in enumerate(CLASSES)}
    out: Dict[int, Tuple[int, int]] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected '<Class Name>:XpYn' (e.g. Person:8p3n)."
            )
        cls_raw, mode_raw = item.split(":", 1)
        cls_name = cls_raw.strip().lower()
        if cls_name not in name_to_id:
            raise ValueError(
                f"Unknown class '{cls_raw}' in --oracle-class-point-overrides. Valid: {', '.join(CLASSES)}"
            )
        m = re.fullmatch(r"\s*(\d+)p(\d+)n\s*", mode_raw.strip().lower())
        if not m:
            raise ValueError(
                f"Invalid mode '{mode_raw}' for class '{cls_raw}'. Expected XpYn (e.g. 8p3n)."
            )
        p_cnt = int(m.group(1))
        n_cnt = int(m.group(2))
        if p_cnt <= 0 or n_cnt < 0:
            raise ValueError(
                f"Override for '{cls_raw}' must satisfy pos>0 and neg>=0 (e.g. 4p0n, 8p2n)."
            )
        out[name_to_id[cls_name]] = (p_cnt, n_cnt)
    return out


def _extract_disagreement_map(out: Dict[str, torch.Tensor], mode: str) -> Optional[np.ndarray]:
    if "disagreement_map" in out:
        d = out["disagreement_map"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return d
    if "pred_rgb" in out and "pred_thm" in out:
        pred_rgb = out["pred_rgb"]
        pred_thm = out["pred_thm"]
        if mode == "argmax":
            d = (pred_rgb.argmax(dim=1) != pred_thm.argmax(dim=1)).float()
            return d.squeeze(0).detach().cpu().numpy().astype(np.float32)
        prob_rgb = torch.softmax(pred_rgb, dim=1)
        prob_thm = torch.softmax(pred_thm, dim=1)
        overlap = (prob_rgb * prob_thm).sum(dim=1)
        d = 1.0 - overlap
        return d.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return None


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    rng = np.random.default_rng(args.seed)

    model = build_model(args, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.checkpoint_state == "teacher":
        if not isinstance(ckpt, dict) or "teacher" not in ckpt:
            raise KeyError(f"Checkpoint has no teacher state: {args.checkpoint}")
        state = ckpt["teacher"]
    else:
        state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    sam2_model = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    ds = FMBDataset(
        args.data_root,
        split=args.split,
        crop_size=args.crop_size,
        augment=False,
        eval_resize_mode=args.eval_resize_mode,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    oracle_modes = _parse_oracle_modes(args.oracle_modes)
    edge_focus_class_ids = _parse_edge_fine_classes(args.oracle_edge_fine_classes)
    class_point_overrides = _parse_class_point_overrides(args.oracle_class_point_overrides)
    excluded_class_ids = _parse_class_list(args.oracle_exclude_classes)
    instance_union_class_ids = _parse_class_list(args.oracle_instance_union_classes)

    metric_base = SegMetric(NUM_CLASSES, IGNORE_INDEX)
    metric_oracle = {m.name: SegMetric(NUM_CLASSES, IGNORE_INDEX) for m in oracle_modes}

    max_samples = int(args.oracle_max_samples)
    for idx, (rgb, thm, gt) in enumerate(loader):
        if max_samples > 0 and idx >= max_samples:
            break

        rgb = rgb.to(device, non_blocking=True)
        thm = thm.to(device, non_blocking=True)
        rgb, thm = apply_single_modality_inputs(rgb, thm, args.single_modality)
        gt_np = gt.squeeze(0).cpu().numpy().astype(np.int64)

        with autocast("cuda", dtype=amp_dtype):
            out = model(rgb, thm, gt=None)
        logits = out["pred"]
        prob = torch.softmax(logits, dim=1)
        conf_max = prob.max(dim=1).values.squeeze(0).detach().cpu().numpy()
        pred_base = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
        metric_base.update(pred_base, gt_np)
        disagreement_map = _extract_disagreement_map(out, args.disagreement_refine_mode)
        disagreement_roi = None
        disagreement_mask_logits = None
        if disagreement_map is not None:
            disagreement_roi = disagreement_map >= float(args.oracle_disagreement_thresh)
            if args.oracle_disagreement_use_mask_prompt:
                disagreement_mask_logits = _build_mask_prompt_logits(
                    disagreement_roi.astype(np.bool_),
                    float(args.oracle_disagreement_mask_logit),
                )

        rgb_uint8 = _tensor_rgb_to_uint8(rgb.squeeze(0))
        for mode in oracle_modes:
            prompt_cls, _prompt_score = _build_prompt_map(
                predictor=predictor,
                rgb_uint8=rgb_uint8,
                gt=gt_np,
                mode=mode,
                min_class_pixels=args.oracle_min_class_pixels,
                box_margin=args.oracle_box_margin,
                multimask_output=args.oracle_multimask,
                rng=rng,
                edge_focus_class_ids=edge_focus_class_ids,
                edge_width=args.oracle_edge_width,
                class_point_overrides=class_point_overrides,
                spread_sampling=args.oracle_spread_sampling,
                excluded_class_ids=excluded_class_ids,
                prompt_type=args.oracle_prompt_type,
                point_layout=args.oracle_point_layout,
                add_gt_mask_prompt=bool(args.oracle_add_gt_mask_prompt),
                only_gt_mask_prompt=bool(args.oracle_only_gt_mask_prompt),
                mask_prompt_logit=float(args.oracle_mask_prompt_logit),
                instance_union_class_ids=instance_union_class_ids,
                instance_refine_iters=max(0, int(args.oracle_instance_refine_iters)),
                instance_refine_min_miss=max(1, int(args.oracle_instance_refine_min_miss)),
                disagreement_mask_logits=disagreement_mask_logits,
            )
            pred_oracle = _apply_prompt_map(
                pred_base=pred_base,
                conf_max=conf_max,
                prompt_cls=prompt_cls,
                override_policy=args.oracle_override_policy,
                uncert_thresh=args.oracle_uncert_thresh,
                roi_mask=(disagreement_roi if args.oracle_disagreement_roi else None),
            )
            metric_oracle[mode.name].update(pred_oracle, gt_np)

        if (idx + 1) % 50 == 0:
            print(f"[progress] {idx + 1} samples")

    baseline = metric_base.compute(absent_score=args.absent_score)
    mode_results = {m.name: metric_oracle[m.name].compute(absent_score=args.absent_score) for m in oracle_modes}

    report = _format_report(args=args, baseline=baseline, mode_results=mode_results)
    print(report)
    out_path = args.out or osp.join(osp.dirname(args.checkpoint), "oracle_prompt_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[ok] saved -> {out_path}")


if __name__ == "__main__":
    main()
