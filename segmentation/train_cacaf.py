#!/usr/bin/env python3
"""
Independent training script for multimodal segmentors on FMB dataset.

Usage:
    cd /home/jl/Drone-SAM-Adapter
    conda activate sam2-unet
    python segmentation/train_cacaf.py [options]

Key hyper-parameters:
    --epochs        200
    --batch-size    4      (effective; adjust for VRAM)
    --crop-size     512
    --lr-adapter    1e-4   (SAM2 bottleneck adapters)
    --lr-aux        1e-4   (ConvNeXt-Tiny backbone)
    --lr-head       2e-4   (fusion + decoder)
    --warmup-iters  500
"""

import argparse
import copy
import inspect
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

# Make sure segmentation/ is importable
_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.datasets.fmb_dataset import (
    CLASSES,
    FMBDataset,
    NUM_CLASSES,
    IGNORE_INDEX,
)
from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.models.segmentors.mmsa_baseline_segmentor import MMSABaselineSegmentor
from segmentation.models.segmentors.rrf_dsd_segmentor import RRFDSDSegmentor


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multimodal segmentor training on FMB")

    # Paths
    p.add_argument("--data-root", default="/home/jl/dataset/FMB")
    p.add_argument("--sam2-cfg",  default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--work-dir",  default="work_dirs/cacaf_fmb")
    p.add_argument(
        "--model-variant",
        default="rrf_dsd",
        choices=["cacaf", "rrf_dsd", "mmsa_baseline"],
        help="Segmentor variant to train",
    )
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1,
                   help=argparse.SUPPRESS)

    # Training
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size * accum_steps * world_size")
    p.add_argument("--crop-size",  type=int,   default=512)
    p.add_argument("--num-workers",type=int,   default=4)
    p.add_argument("--aux-encoder-size", default="tiny", choices=["tiny", "small"],
                   help="Thermal/auxiliary ConvNeXt encoder size")
    p.add_argument("--aux-pretrained-path", default=None,
                   help="Optional local ConvNeXt aux checkpoint path; avoids network download")

    # Optimizer
    p.add_argument("--lr-adapter", type=float, default=1e-4,
                   help="LR for SAM2 bottleneck adapter parameters")
    p.add_argument("--lr-rgb-backbone", type=float, default=1e-5,
                   help="LR for unfrozen SAM2/Hiera backbone block parameters")
    p.add_argument("--rgb-layer-decay", type=float, default=0.90,
                   help="Layer-wise LR decay for unfrozen SAM2/Hiera blocks")
    p.add_argument("--lr-aux",     type=float, default=1e-4,
                   help="LR for ConvNeXt-Tiny auxiliary encoder")
    p.add_argument("--lr-head",    type=float, default=2e-4,
                   help="LR for fusion + decoder modules")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-iters",  type=int,   default=500)

    # Misc
    p.add_argument("--resume",  default=None, help="Path to checkpoint to resume")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--amp",     action="store_true", default=True,
                   help="Use mixed-precision training (default: on)")
    p.add_argument("--no-amp",  dest="amp", action="store_false")
    p.add_argument("--bf16",    action="store_true", default=False,
                   help="Use BF16 instead of FP16 (wider range, fewer NaN risks)")
    p.add_argument("--val-freq", type=int, default=5,
                   help="Run validation every N epochs")
    p.add_argument("--save-freq", type=int, default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for teacher model")
    p.add_argument("--val-model", default="student", choices=["student", "teacher"],
                   help="Model used for validation and best checkpoint selection")
    p.add_argument("--best-metric", default="miou",
                   choices=["miou", "critical_mean"],
                   help="Validation metric used for best checkpoint selection")
    p.add_argument("--critical-classes", type=str, default="3,9,10,11,12,13",
                   help="Comma-separated 0-based class ids used by critical_mean selection")
    p.add_argument("--use-dice", action="store_true", default=False,
                   help="Add Dice loss alongside CE (improves rare-class IoU)")
    p.add_argument("--dice-weight", type=float, default=1.0,
                   help="Weight for Dice loss component")
    p.add_argument("--detail-aux-weight", type=float, default=0.2,
                   help="RRF-DSD only: weight for stride-4 detail auxiliary head")
    p.add_argument("--use-thin-structure-refiner", action="store_true", default=False,
                   help="RRF-DSD only: enable thin-structure compensation branch in decoder")
    p.add_argument("--thin-refiner-scale", type=float, default=0.15,
                   help="RRF-DSD only: scale factor for thin-structure compensation")
    p.add_argument("--use-boundary-refiner", action="store_true", default=False,
                   help="RRF-DSD only: enable boundary-guided refinement branch in decoder")
    p.add_argument("--boundary-refiner-scale", type=float, default=0.5,
                   help="RRF-DSD only: residual scale for boundary-guided logits")
    p.add_argument("--boundary-aux-loss-weight", type=float, default=0.05,
                   help="RRF-DSD only: BCE boundary auxiliary loss weight")
    p.add_argument("--use-rare-class-residual", action="store_true", default=False,
                   help="RRF-DSD only: enable rare-class residual logits branch")
    p.add_argument("--rare-class-indices", type=str, default="1,3,4,13",
                   help="RRF-DSD only: comma-separated rare class indices (0-based)")
    p.add_argument("--rare-class-scale", type=float, default=1.0,
                   help="RRF-DSD only: residual scale for rare-class logits")
    p.add_argument("--use-ohem", action="store_true", default=False,
                   help="Use OHEM CrossEntropy instead of vanilla CE")
    p.add_argument("--ohem-thresh", type=float, default=0.7,
                   help="OHEM confidence threshold")
    p.add_argument("--ohem-min-kept", type=int, default=100000,
                   help="OHEM minimum number of hard pixels kept")
    p.add_argument("--use-class-balanced-loss", action="store_true", default=False,
                   help="Enable class-balanced CE/OHEM weights estimated from train labels")
    p.add_argument("--class-balance-alpha", type=float, default=0.5,
                   help="Class-balance exponent on pixel frequency (weight=freq^-alpha)")
    p.add_argument("--class-weight-min", type=float, default=0.5,
                   help="Minimum clamp value for class-balanced weights")
    p.add_argument("--class-weight-max", type=float, default=3.0,
                   help="Maximum clamp value for class-balanced weights")
    p.add_argument("--no-cacaf", action="store_true", default=False,
                   help="CACAF variant only: replace CACAF with simple concat fusion")
    p.add_argument("--no-sagu",  action="store_true", default=False,
                   help="CACAF variant only: disable SAGU channel attention in HGSOAD")
    p.add_argument("--enable-modality-heads", action="store_true", default=False,
                   help="Enable RGB-only and Thermal-only auxiliary heads")
    p.add_argument("--modality-head-weight", type=float, default=0.2,
                   help="Auxiliary supervision weight for modality heads")
    p.add_argument("--fusion-use-agreement-map", action="store_true", default=False,
                   help="MMSA baseline only: use modality agreement map in fusion blocks")
    p.add_argument("--fusion-agreement-mode", default="prob", choices=["argmax", "prob"],
                   help="MMSA baseline only: agreement map mode for fusion")
    p.add_argument("--mmsa-fusion-mode", default="mmsa", choices=["mmsa", "naive"],
                   help="MMSA baseline only: fusion block type")
    p.add_argument("--enable-disagreement-refine", action="store_true", default=False,
                   help="MMSA baseline only: enable disagreement refinement branch")
    p.add_argument("--disagreement-refine-weight", type=float, default=0.5,
                   help="Loss weight for refined prediction")
    p.add_argument("--disagreement-refine-mode", default="argmax", choices=["argmax", "prob"],
                   help="Disagreement map mode for refinement")
    p.add_argument("--no-disagreement-refine-gate", action="store_true", default=False,
                   help="Remove spatial disagreement gate from refinement residual")
    p.add_argument("--enable-reliability-guided-refine", action="store_true", default=False,
                   help="RRF-DSD only: refine main prediction with disagreement and multi-scale reliability maps")
    p.add_argument("--thermal-prior-injection", action="store_true", default=False,
                   help="MMSA baseline only: inject thermal features into RGB encoder features before fusion")
    p.add_argument("--thermal-prior-init", type=float, default=0.1,
                   help="MMSA baseline only: initial scale for thermal prior injection")
    p.add_argument("--unfreeze-rgb-last-n-blocks", type=int, default=0,
                   help="MMSA baseline only: unfreeze the last N original SAM2/Hiera blocks")
    p.add_argument("--freeze-encoders", default=None, metavar="CKPT",
                   help="Load encoder (SAM2 adapter + ConvNeXt) weights from CKPT "
                        "and freeze them. Only fusion+decoder will be trained.")
    p.add_argument("--use-trainval", action="store_true", default=False,
                   help="Train on train+val (1220 images) instead of train only (1060). "
                        "No validation during training; use fixed epochs or train loss.")
    p.add_argument("--eval-resize-mode", default="letterbox",
                   choices=["stretch", "letterbox"],
                   help="Validation/Test resize mode: stretch or keep-ratio letterbox")
    p.add_argument("--train-resize-mode", default="legacy",
                   choices=["legacy", "mmseg"],
                   help="Training resize mode: legacy or mmseg-style ratio resize before pad/crop")
    p.add_argument("--cat-max-ratio", type=float, default=0.75,
                   help="RandomCrop max dominant-class ratio (1.0 disables)")
    p.add_argument("--blur-prob", type=float, default=0.2,
                   help="Gaussian blur probability in training augmentation")
    p.add_argument("--photo-distort", dest="photo_distort", action="store_true",
                   help="Enable stronger photometric distortion")
    p.add_argument("--no-photo-distort", dest="photo_distort", action="store_false",
                   help="Disable stronger photometric distortion")
    p.add_argument("--ddp-find-unused-params", dest="ddp_find_unused_params",
                   action="store_true", default=True,
                   help="DDP only: enable find_unused_parameters")
    p.add_argument("--no-ddp-find-unused-params", dest="ddp_find_unused_params",
                   action="store_false",
                   help="DDP only: disable find_unused_parameters")
    p.add_argument("--ddp-bucket-cap-mb", type=int, default=25,
                   help="DDP only: bucket size in MB")
    p.add_argument("--ddp-broadcast-buffers", dest="ddp_broadcast_buffers",
                   action="store_true", default=False,
                   help="DDP only: broadcast buffers from rank0")
    p.add_argument("--no-ddp-broadcast-buffers", dest="ddp_broadcast_buffers",
                   action="store_false",
                   help="DDP only: disable buffer broadcast (default)")
    p.add_argument("--ddp-init-sync", dest="ddp_init_sync",
                   action="store_true", default=True,
                   help="DDP only: sync params/buffers at construction")
    p.add_argument("--no-ddp-init-sync", dest="ddp_init_sync",
                   action="store_false",
                   help="DDP only: skip init sync at DDP construction")
    p.add_argument("--force-nccl-on-5090", action="store_true", default=False,
                   help="Allow auto mode to use nccl on RTX 5090")
    p.add_argument("--dist-backend", default="auto", choices=["auto", "nccl", "gloo"],
                   help="DDP only: distributed backend")
    p.set_defaults(photo_distort=True)

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[warn] Ignoring unknown args: {unknown}")
    if args.accum_steps < 1:
        raise ValueError("--accum-steps must be >= 1")
    return args


def parse_int_list(raw: str | None):
    if raw is None:
        return None
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


# ---------------------------------------------------------------------------
# IoU metric (numpy, no external deps)
# ---------------------------------------------------------------------------

class SegMetric:
    """Accumulates confusion matrix for mIoU / mAcc / aAcc.

    By default, mean metrics skip absent classes (`union == 0`) via `nanmean`,
    which matches the current in-repo CACAF evaluation behavior. For strict
    comparison with external results that count absent classes as zero, set
    `absent_score=0.0` in `compute()`.
    """

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.confusion     = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """pred, gt: (H, W) int arrays."""
        # Robust guard: ignore any unexpected GT ids outside [0, num_classes-1]
        mask = (gt != self.ignore_index) & (gt >= 0) & (gt < self.num_classes)
        pred = pred[mask]
        gt   = gt[mask]
        idx  = gt * self.num_classes + pred
        # Clip invalid predictions to ignore
        valid = (pred >= 0) & (pred < self.num_classes)
        np.add.at(self.confusion.ravel(), idx[valid], 1)

    def compute(self, absent_score=None):
        cm = self.confusion
        tp = np.diag(cm)
        gt_sum   = cm.sum(axis=1)
        pred_sum = cm.sum(axis=0)
        union    = gt_sum + pred_sum - tp

        iou  = np.where(union > 0, tp / union, np.nan)
        acc  = np.where(gt_sum > 0, tp / gt_sum, np.nan)

        if absent_score is None:
            miou = float(np.nanmean(iou) * 100)
            macc = float(np.nanmean(acc) * 100)
        else:
            iou = np.where(np.isnan(iou), absent_score, iou)
            acc = np.where(np.isnan(acc), absent_score, acc)
            miou = float(iou.mean() * 100)
            macc = float(acc.mean() * 100)
        aacc = tp.sum() / cm.sum()

        return dict(
            mIoU  = miou,
            mAcc  = macc,
            aAcc  = float(aacc * 100),
            iou_per_class = (iou * 100).tolist(),
        )

    def reset(self):
        self.confusion[:] = 0


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def cosine_lr_schedule(optimizer, step, total_steps, warmup_steps,
                        base_lrs, min_lr_ratio=1e-3):
    """In-place update of param group LRs."""
    if step < warmup_steps:
        factor = step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (
            1 + np.cos(np.pi * progress)
        )
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state, path):
    os.makedirs(osp.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] saved → {path}")


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


def load_checkpoint(model, teacher, optimizer, scaler, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model"])
    if teacher is not None:
        teacher_state = ckpt.get("teacher", ckpt["model"])
        teacher.load_state_dict(teacher_state)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_miou   = ckpt.get("best_miou", 0.0)
    print(f"  [ckpt] resumed from epoch {ckpt['epoch']}  best_mIoU={best_miou:.2f}")
    return start_epoch, best_miou


def build_ema_teacher(model: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(unwrap_model(model))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    student_core = unwrap_model(student)
    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student_core.named_parameters())
    for name, param_t in teacher_params.items():
        param_s = student_params[name]
        param_t.data.mul_(decay).add_(param_s.data, alpha=1.0 - decay)

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student_core.named_buffers())
    for name, buffer_t in teacher_buffers.items():
        buffer_s = student_buffers[name]
        buffer_t.data.copy_(buffer_s.data)


def summarize_critical_classes(iou_per_class, critical_indices):
    critical_rows = []
    critical_vals = []
    for idx in critical_indices:
        if idx < 0 or idx >= len(iou_per_class):
            continue
        val = float(iou_per_class[idx])
        critical_rows.append((CLASSES[idx], val))
        if not np.isnan(val):
            critical_vals.append(val)
    if critical_vals:
        critical_mean = float(np.mean(critical_vals))
        critical_min = float(np.min(critical_vals))
    else:
        critical_mean = float("nan")
        critical_min = float("nan")
    return critical_rows, critical_mean, critical_min


def select_validation_score(results, best_metric, critical_indices):
    if best_metric == "miou":
        return float(results["mIoU"])
    if best_metric == "critical_mean":
        _, critical_mean, _ = summarize_critical_classes(results["iou_per_class"], critical_indices)
        return critical_mean
    raise ValueError(f"Unsupported best_metric: {best_metric}")


def _iter_label_paths(data_root: str, split: str):
    splits_to_load = ["train", "val"] if split == "trainval" else [split]
    for split_name in splits_to_load:
        lbl_dir = osp.join(data_root, split_name, "Label")
        for subset in ("easy", "hard"):
            txt = osp.join(data_root, f"{split_name}_{subset}_files.txt")
            if not osp.exists(txt):
                continue
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    fname = line.strip()
                    if fname:
                        yield osp.join(lbl_dir, fname)


def estimate_class_weights_from_labels(
    data_root: str,
    split: str,
    num_classes: int,
    alpha: float,
    weight_min: float,
    weight_max: float,
):
    counts = np.zeros(num_classes, dtype=np.int64)
    num_files = 0
    for label_path in _iter_label_paths(data_root, split):
        num_files += 1
        lbl = np.array(Image.open(label_path), dtype=np.int64)
        valid = (lbl >= 1) & (lbl <= num_classes)  # FMB labels: 1..14
        if not np.any(valid):
            continue
        cls = lbl[valid] - 1
        counts += np.bincount(cls, minlength=num_classes)

    if num_files == 0:
        raise RuntimeError(f"No label files found for split '{split}' under: {data_root}")

    present = counts > 0
    if not np.any(present):
        raise RuntimeError("All class counts are zero; cannot build class-balanced weights.")

    freq = counts.astype(np.float64)
    freq = freq / max(freq.sum(), 1.0)

    weights = np.ones(num_classes, dtype=np.float64)
    weights[present] = np.power(np.maximum(freq[present], 1e-12), -alpha)
    weights[present] = weights[present] / max(weights[present].mean(), 1e-12)
    weights[present] = np.clip(weights[present], weight_min, weight_max)
    weights[present] = weights[present] / max(weights[present].mean(), 1e-12)

    return weights.astype(np.float32).tolist(), counts.tolist(), num_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = 0
    local_rank = 0
    dist_backend = None

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA, but CUDA is not available.")
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
        torch.cuda.set_device(local_rank)
        if args.dist_backend == "auto":
            gpu_name = torch.cuda.get_device_name(local_rank).lower()
            # 5090 + current stack has NCCL reducer instability in this project.
            if "5090" in gpu_name and not args.force_nccl_on_5090:
                dist_backend = "gloo"
            else:
                dist_backend = "nccl"
        else:
            dist_backend = args.dist_backend
        init_pg_kwargs = dict(backend=dist_backend, init_method="env://")
        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_pg_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_pg_kwargs)
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rank0_print(*msgs, **kwargs):
        if rank == 0:
            print(*msgs, **kwargs)

    # Reproducibility: keep model init consistent across ranks.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    rank0_print(
        f"[dist] enabled={distributed}  rank={rank}/{world_size}  local_rank={local_rank}"
        + (f"  backend={dist_backend}" if distributed else "")
    )
    if distributed:
        rank0_print(
            f"[ddp] find_unused={args.ddp_find_unused_params}  "
            f"bucket_cap_mb={args.ddp_bucket_cap_mb}  "
            f"broadcast_buffers={args.ddp_broadcast_buffers}  "
            f"init_sync={args.ddp_init_sync}"
        )
    rank0_print(f"[model] variant={args.model_variant}")

    if args.model_variant != "cacaf" and (args.no_cacaf or args.no_sagu):
        rank0_print("  [warn] --no-cacaf/--no-sagu 仅对 cacaf 变体生效，当前将忽略这些参数")
    if args.model_variant != "mmsa_baseline" and (
        args.enable_modality_heads
        or args.fusion_use_agreement_map
        or args.enable_disagreement_refine
        or args.thermal_prior_injection
    ):
        rank0_print("  [warn] MMSA-specific options are ignored because model_variant is not mmsa_baseline")

    train_split = "trainval" if args.use_trainval else "train"
    ce_class_weight = None
    if args.use_class_balanced_loss:
        if distributed:
            if rank == 0:
                w_list, count_list, n_files = estimate_class_weights_from_labels(
                    data_root=args.data_root,
                    split=train_split,
                    num_classes=NUM_CLASSES,
                    alpha=args.class_balance_alpha,
                    weight_min=args.class_weight_min,
                    weight_max=args.class_weight_max,
                )
                weight_tensor = torch.tensor(w_list, dtype=torch.float32, device=device)
                count_tensor = torch.tensor(count_list, dtype=torch.float64, device=device)
                rank0_print(f"[class-balance] label_files={n_files} split={train_split}")
            else:
                weight_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
                count_tensor = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
            dist.broadcast(weight_tensor, src=0)
            dist.broadcast(count_tensor, src=0)
            ce_class_weight = weight_tensor.cpu().tolist()
            class_counts = [int(x) for x in count_tensor.cpu().tolist()]
        else:
            ce_class_weight, class_counts, n_files = estimate_class_weights_from_labels(
                data_root=args.data_root,
                split=train_split,
                num_classes=NUM_CLASSES,
                alpha=args.class_balance_alpha,
                weight_min=args.class_weight_min,
                weight_max=args.class_weight_max,
            )
            rank0_print(f"[class-balance] label_files={n_files} split={train_split}")

        rank0_print(
            f"[class-balance] enabled alpha={args.class_balance_alpha} "
            f"clip=[{args.class_weight_min}, {args.class_weight_max}]"
        )
        for cls_name, cnt, w in zip(CLASSES, class_counts, ce_class_weight):
            rank0_print(f"  {cls_name:16s} pixels={cnt:10d}  weight={w:.3f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    rare_class_indices = parse_int_list(args.rare_class_indices)
    if args.use_rare_class_residual and (not rare_class_indices):
        raise ValueError("--use-rare-class-residual requires non-empty --rare-class-indices")

    if args.model_variant == "cacaf":
        model = CACafSegmentor(
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
            num_classes=NUM_CLASSES,
            use_dice=args.use_dice,
            dice_weight=args.dice_weight,
            use_ohem=args.use_ohem,
            ohem_thresh=args.ohem_thresh,
            ohem_min_kept=args.ohem_min_kept,
            use_cacaf=not args.no_cacaf,
            use_sagu=not args.no_sagu,
            ce_class_weight=ce_class_weight,
        ).to(device)
    elif args.model_variant == "rrf_dsd":
        model = RRFDSDSegmentor(
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
            num_classes=NUM_CLASSES,
            use_dice=args.use_dice,
            dice_weight=args.dice_weight,
            detail_aux_loss_weight=args.detail_aux_weight,
            use_ohem=args.use_ohem,
            ohem_thresh=args.ohem_thresh,
            ohem_min_kept=args.ohem_min_kept,
            use_thin_structure_refiner=args.use_thin_structure_refiner,
            thin_refiner_scale=args.thin_refiner_scale,
            use_boundary_refiner=args.use_boundary_refiner,
            boundary_refiner_scale=args.boundary_refiner_scale,
            boundary_aux_loss_weight=args.boundary_aux_loss_weight,
            use_rare_class_residual=args.use_rare_class_residual,
            rare_class_indices=rare_class_indices,
            rare_class_scale=args.rare_class_scale,
            ce_class_weight=ce_class_weight,
            enable_modality_heads=args.enable_modality_heads,
            modality_head_weight=args.modality_head_weight,
            enable_reliability_guided_refine=args.enable_reliability_guided_refine,
            disagreement_refine_weight=args.disagreement_refine_weight,
            disagreement_refine_mode=args.disagreement_refine_mode,
            disagreement_refine_use_gate=not args.no_disagreement_refine_gate,
        ).to(device)
    else:
        model = MMSABaselineSegmentor(
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
            num_classes=NUM_CLASSES,
            aux_encoder_size=args.aux_encoder_size,
            aux_pretrained_path=args.aux_pretrained_path,
            unfreeze_rgb_last_n_blocks=args.unfreeze_rgb_last_n_blocks,
            use_dice=args.use_dice,
            dice_weight=args.dice_weight,
            use_ohem=args.use_ohem,
            ohem_thresh=args.ohem_thresh,
            ohem_min_kept=args.ohem_min_kept,
            ce_class_weight=ce_class_weight,
            enable_modality_heads=args.enable_modality_heads,
            modality_head_weight=args.modality_head_weight,
            fusion_use_agreement_map=args.fusion_use_agreement_map,
            fusion_agreement_mode=args.fusion_agreement_mode,
            mmsa_fusion_mode=args.mmsa_fusion_mode,
            enable_disagreement_refine=args.enable_disagreement_refine,
            disagreement_refine_weight=args.disagreement_refine_weight,
            disagreement_refine_mode=args.disagreement_refine_mode,
            disagreement_refine_use_gate=not args.no_disagreement_refine_gate,
            thermal_prior_injection=args.thermal_prior_injection,
            thermal_prior_init=args.thermal_prior_init,
        ).to(device)

    # ------------------------------------------------------------------
    # Optional: freeze encoders for fast ablation training
    # ------------------------------------------------------------------
    if args.freeze_encoders:
        rank0_print(f"[freeze-encoders] 加载编码器权重: {args.freeze_encoders}")
        enc_ckpt = torch.load(args.freeze_encoders, map_location=device)
        enc_state = enc_ckpt["model"]
        # Load only rgb_encoder and aux_encoder weights (strict=False, others skipped)
        enc_keys = {k: v for k, v in enc_state.items()
                    if k.startswith("rgb_encoder.") or k.startswith("aux_encoder.")}
        missing, unexpected = model.load_state_dict(enc_keys, strict=False)
        loaded = len(enc_keys)
        rank0_print(f"  加载 {loaded} 个编码器参数，ignored {len(unexpected)} unexpected keys")
        # Freeze
        for name, param in model.named_parameters():
            if "rgb_encoder" in name or "aux_encoder" in name:
                param.requires_grad_(False)
        frozen = sum(p.numel() for n, p in model.named_parameters()
                     if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"  冻结: {frozen/1e6:.1f}M  可训练: {trainable/1e6:.1f}M "
                    f"(仅 fusion + decoder)\n")

    # Parameter groups with separate LRs.  If SAM2/Hiera blocks are partially
    # unfrozen, keep their LR much lower than the adapters and apply layer decay.
    adapter_params = []
    rgb_backbone_by_block = {}
    rgb_backbone_other = []
    aux_params     = []
    head_params    = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "rgb_encoder" in name and ".prompt_learn." in name:
            adapter_params.append(param)
        elif "rgb_encoder" in name:
            block_idx = None
            parts = name.split(".")
            if "blocks" in parts:
                idx_pos = parts.index("blocks") + 1
                if idx_pos < len(parts) and parts[idx_pos].isdigit():
                    block_idx = int(parts[idx_pos])
            if block_idx is None:
                rgb_backbone_other.append(param)
            else:
                rgb_backbone_by_block.setdefault(block_idx, []).append(param)
        elif "aux_encoder" in name:
            aux_params.append(param)
        else:
            head_params.append(param)

    if args.freeze_encoders:
        # Only one param group when encoders are frozen
        param_groups = [{"params": head_params, "lr": args.lr_head, "name": "head"}]
        base_lrs = [args.lr_head]
    else:
        param_groups = []
        base_lrs = []
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": args.lr_adapter, "name": "adapter"})
            base_lrs.append(args.lr_adapter)
        if rgb_backbone_other:
            lr = args.lr_rgb_backbone * (args.rgb_layer_decay ** max(args.unfreeze_rgb_last_n_blocks, 1))
            param_groups.append({"params": rgb_backbone_other, "lr": lr, "name": "rgb_backbone_other"})
            base_lrs.append(lr)
        if rgb_backbone_by_block:
            max_block_idx = max(rgb_backbone_by_block)
            for block_idx in sorted(rgb_backbone_by_block):
                decay_power = max_block_idx - block_idx
                lr = args.lr_rgb_backbone * (args.rgb_layer_decay ** decay_power)
                param_groups.append(
                    {
                        "params": rgb_backbone_by_block[block_idx],
                        "lr": lr,
                        "name": f"rgb_backbone_block{block_idx}",
                    }
                )
                base_lrs.append(lr)
        if aux_params:
            param_groups.append({"params": aux_params, "lr": args.lr_aux, "name": "aux_encoder"})
            base_lrs.append(args.lr_aux)
        if head_params:
            param_groups.append({"params": head_params, "lr": args.lr_head, "name": "head"})
            base_lrs.append(args.lr_head)

    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        rank0_print(f"[optim] group={pg.get('name', '?')} params={n_params/1e6:.2f}M lr={pg['lr']:.2e}")

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay
    )
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = GradScaler('cuda') if (args.amp and not args.bf16) else None

    if distributed:
        ddp_kwargs = dict(
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused_params,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            broadcast_buffers=args.ddp_broadcast_buffers,
        )
        if "init_sync" in inspect.signature(nn.parallel.DistributedDataParallel).parameters:
            ddp_kwargs["init_sync"] = args.ddp_init_sync
        model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    teacher = build_ema_teacher(model)

    # Rank-specific randomness for data pipeline after model construction.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = FMBDataset(args.data_root, split=train_split,
                          crop_size=args.crop_size, augment=True,
                          eval_resize_mode=args.eval_resize_mode,
                          train_resize_mode=args.train_resize_mode,
                          cat_max_ratio=args.cat_max_ratio,
                          blur_prob=args.blur_prob,
                          photo_distort=args.photo_distort)
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )

    # Val dataset: only used if not using trainval
    if args.use_trainval:
        val_ds = None
        val_loader = None
        rank0_print(f"  [trainval mode] Training on {len(train_ds)} images (train+val), no validation")
    else:
        val_ds = FMBDataset(args.data_root, split="val",
                            crop_size=args.crop_size, augment=False,
                            eval_resize_mode=args.eval_resize_mode,
                            cat_max_ratio=1.0,
                            blur_prob=0.0,
                            photo_distort=False)
        if distributed and rank != 0:
            val_loader = None
        else:
            val_loader = DataLoader(
                val_ds, batch_size=1, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            rank0_print(f"  Training on {len(train_ds)} images, validating on {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    updates_per_epoch = max(1, (len(train_loader) + args.accum_steps - 1) // args.accum_steps)
    total_iters = args.epochs * updates_per_epoch
    effective_batch = args.batch_size * args.accum_steps * world_size
    rank0_print(
        f"[protocol] train_resize={args.train_resize_mode}  eval_resize={args.eval_resize_mode}  "
        f"batch_per_gpu={args.batch_size}  accum_steps={args.accum_steps}  "
        f"world_size={world_size}  effective_batch={effective_batch}"
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_miou   = 0.0
    if args.resume:
        start_epoch, best_miou = load_checkpoint(
            model, teacher, optimizer, scaler, args.resume, device
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    metric = SegMetric(NUM_CLASSES, IGNORE_INDEX)
    critical_indices = parse_int_list(args.critical_classes) or []
    global_step = start_epoch * updates_per_epoch

    for epoch in range(start_epoch, args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (rgb, thm, gt) in enumerate(train_loader):
            rgb = rgb.to(device, non_blocking=True)
            thm = thm.to(device, non_blocking=True)
            gt  = gt.to(device,  non_blocking=True)

            # LR schedule
            cosine_lr_schedule(optimizer, global_step, total_iters,
                                args.warmup_iters, base_lrs)

            if args.amp:
                with autocast('cuda', dtype=amp_dtype):
                    out = model(rgb, thm, gt)
                loss = out["loss"]
                finite_flag = torch.isfinite(loss).to(dtype=torch.int32, device=device)
                if distributed:
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                if finite_flag.item() == 0:
                    rank0_print(
                        f"  WARNING: non-finite loss detected at iter={batch_idx}, skipping batch on all ranks"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss_to_backward = loss / args.accum_steps
                if scaler is not None:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()
            else:
                out  = model(rgb, thm, gt)
                loss = out["loss"]
                finite_flag = torch.isfinite(loss).to(dtype=torch.int32, device=device)
                if distributed:
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                if finite_flag.item() == 0:
                    rank0_print(
                        f"  WARNING: non-finite loss detected at iter={batch_idx}, skipping batch on all ranks"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss_to_backward = loss / args.accum_steps
                loss_to_backward.backward()

            should_step = ((batch_idx + 1) % args.accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                ema_update(teacher, model, args.ema_decay)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += loss.item()

            if rank == 0 and batch_idx % 50 == 0:
                lr_now = optimizer.param_groups[-1]["lr"]
                rank0_print(
                    f"  Ep[{epoch+1}/{args.epochs}] "
                    f"iter {batch_idx}/{len(train_loader)}  step={global_step}/{total_iters}  "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}"
                )

        if distributed:
            epoch_stats = torch.tensor(
                [epoch_loss, float(len(train_loader))], dtype=torch.float64, device=device
            )
            dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
            avg_loss = float(epoch_stats[0].item() / max(epoch_stats[1].item(), 1.0))
        else:
            avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        if rank == 0:
            rank0_print(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"avg_loss={avg_loss:.4f}  time={elapsed:.1f}s"
            )

        # Periodic checkpoint
        if rank == 0 and (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                dict(epoch=epoch, model=unwrap_model(model).state_dict(),
                     teacher=teacher.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict() if scaler else None,
                     best_miou=best_miou),
                osp.join(args.work_dir, f"epoch_{epoch+1}.pth"),
            )

        # Validation (skip if using trainval)
        if distributed:
            dist.barrier()
        if val_loader and (epoch + 1) % args.val_freq == 0:
            val_core = teacher if args.val_model == "teacher" else unwrap_model(model)
            results = validate(val_core, val_loader, device, metric, args.amp, amp_dtype)
            miou = results["mIoU"]
            critical_rows, critical_mean, critical_min = summarize_critical_classes(
                results["iou_per_class"], critical_indices
            )
            val_score = select_validation_score(results, args.best_metric, critical_indices)
            rank0_print(
                f"  [val:{args.val_model}] mIoU={miou:.2f}  mAcc={results['mAcc']:.2f}"
                f"  aAcc={results['aAcc']:.2f}"
            )
            _print_per_class(results["iou_per_class"])
            if critical_rows:
                crit_txt = "  ".join(
                    f"{name}={val:.1f}" if not np.isnan(val) else f"{name}=nan"
                    for name, val in critical_rows
                )
                rank0_print(
                    f"  [val:{args.val_model}] critical_mean={critical_mean:.2f}  "
                    f"critical_min={critical_min:.2f}  select({args.best_metric})={val_score:.2f}"
                )
                rank0_print(f"  [critical] {crit_txt}")

            if val_score > best_miou:
                best_miou = val_score
                save_checkpoint(
                    dict(epoch=epoch, model=unwrap_model(model).state_dict(),
                         teacher=teacher.state_dict(),
                         optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict() if scaler else None,
                         best_miou=best_miou),
                    osp.join(args.work_dir, "best.pth"),
                )
                rank0_print(
                    f"  [val] *** new best {args.best_metric}={best_miou:.2f} "
                    f"(mIoU={miou:.2f}, critical_mean={critical_mean:.2f}) ***"
                )
        if distributed:
            dist.barrier()

    if rank == 0:
        rank0_print(f"\nTraining complete. Best val mIoU = {best_miou:.2f}")
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, metric, use_amp, amp_dtype=torch.float16):
    was_training = model.training
    model.eval()
    metric.reset()

    for rgb, thm, gt in loader:
        rgb = rgb.to(device, non_blocking=True)
        thm = thm.to(device, non_blocking=True)

        if use_amp:
            with autocast('cuda', dtype=amp_dtype):
                out = model(rgb, thm, gt=None)
        else:
            out = model(rgb, thm, gt=None)

        pred = out["pred"].argmax(dim=1)   # (B, H, W)
        pred_np = pred.cpu().numpy()
        gt_np   = gt.numpy()

        for p, g in zip(pred_np, gt_np):
            metric.update(p, g)

    model.train(was_training)
    return metric.compute()


def _print_per_class(iou_list):
    from segmentation.datasets.fmb_dataset import CLASSES
    rows = []
    for cls, iou in zip(CLASSES, iou_list):
        rows.append(f"  {cls:16s}: {iou:.1f}")
    print("\n".join(rows))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
