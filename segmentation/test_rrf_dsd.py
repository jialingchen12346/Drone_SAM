#!/usr/bin/env python3
"""
独立的测试脚本，用于在 FMB 测试集上评估训练好的模型。

Usage:
    python segmentation/test_rrf_dsd.py \
        --checkpoint /tmp/smoke_single/best.pth \
        --data-root /root/autodl-tmp/datasets/FMB
"""

import argparse
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.datasets.fmb_dataset import FMBDataset, NUM_CLASSES, IGNORE_INDEX, CLASSES
from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.models.segmentors.mmsa_baseline_segmentor import MMSABaselineSegmentor
from segmentation.models.segmentors.rrf_dsd_segmentor import RRFDSDSegmentor


# IoU metric（复用训练脚本中的实现）
class SegMetric:
    def __init__(self, num_classes, ignore_index=255):
        """
        Args:
            num_classes: 类别数量
            ignore_index: 忽略索引（默认 255）
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        mask = (gt != self.ignore_index) & (gt >= 0) & (gt < self.num_classes)
        pred = pred[mask]
        gt = gt[mask]
        idx = gt * self.num_classes + pred
        valid = (pred >= 0) & (pred < self.num_classes)
        np.add.at(self.confusion.ravel(), idx[valid], 1)

    def compute(self, absent_score=None):
        cm = self.confusion
        tp = np.diag(cm)
        gt_sum = cm.sum(axis=1)
        pred_sum = cm.sum(axis=0)
        union = gt_sum + pred_sum - tp

        # 记录有效类别（在测试集中实际出现的类别）
        valid_classes = np.where(gt_sum > 0)[0]
        absent_classes = np.where(gt_sum == 0)[0]

        # 计算 IoU 和 Acc
        iou = np.where(union > 0, tp / union, np.nan)
        acc = np.where(gt_sum > 0, tp / gt_sum, np.nan)

        # 处理未出现的类别
        if absent_score is not None:
            iou[absent_classes] = absent_score
            acc[absent_classes] = absent_score

        # 计算 mIoU/mAcc
        if absent_score is None:
            # 忽略未出现的类别
            miou = float(np.nanmean(iou) * 100)
            macc = float(np.nanmean(acc) * 100)
        else:
            # 对未出现的类别计入 0 分
            miou = float(iou.mean() * 100)
            macc = float(acc.mean() * 100)

        aacc = tp.sum() / cm.sum() * 100

        return dict(
            mIoU=miou, mAcc=macc, aAcc=float(aacc),
            iou_per_class=(iou * 100).tolist(),
            valid_classes=valid_classes.tolist(),
            absent_classes=absent_classes.tolist(),
            num_valid=len(valid_classes),
        )

    def reset(self):
        self.confusion[:] = 0


def parse_args():
    parser = argparse.ArgumentParser(description="RRF-DSD testing on FMB")
    parser.add_argument(
        "--model-variant",
        default="rrf_dsd",
        choices=["rrf_dsd", "cacaf", "mmsa_baseline"],
        help="Model variant to evaluate",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--data-root", default="/root/autodl-tmp/datasets/FMB")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Evaluation split")
    parser.add_argument("--sam2-ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-resize-mode", default="letterbox", choices=["stretch", "letterbox"])
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use-thin-structure-refiner", action="store_true", default=False,
                        help="Enable thin-structure compensation branch for V3 checkpoints")
    parser.add_argument("--thin-refiner-scale", type=float, default=0.15,
                        help="Thin-structure compensation scale (must match training)")
    parser.add_argument("--use-boundary-refiner", action="store_true", default=False,
                        help="Enable boundary-guided refinement branch")
    parser.add_argument("--boundary-refiner-scale", type=float, default=0.5,
                        help="Boundary-guided residual scale (must match training)")
    parser.add_argument("--use-rare-class-residual", action="store_true", default=False,
                        help="Enable rare-class residual branch")
    parser.add_argument("--rare-class-indices", type=str, default="1,3,4,13",
                        help="Comma-separated rare class indices (0-based)")
    parser.add_argument("--rare-class-scale", type=float, default=1.0,
                        help="Rare-class residual scale (must match training)")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    rare_class_indices = [
        int(x.strip()) for x in args.rare_class_indices.split(",") if x.strip()
    ]
    if args.use_rare_class_residual and not rare_class_indices:
        raise ValueError("--use-rare-class-residual requires non-empty --rare-class-indices")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_cn = "验证" if args.split == "val" else "测试"

    print(f"[config] model_variant: {args.model_variant}")
    print(f"[config] checkpoint: {args.checkpoint}")
    print(f"[config] data_root: {args.data_root}")
    print(f"[config] split: {args.split}")
    print(f"[config] crop_size: {args.crop_size}, batch_size: {args.batch_size}")
    print(f"[config] bf16: {args.bf16}")

    # 加载模型
    print("\n[load] 构建模型...")
    if args.model_variant == "rrf_dsd":
        model = RRFDSDSegmentor(
            num_classes=NUM_CLASSES,
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
            use_thin_structure_refiner=args.use_thin_structure_refiner,
            thin_refiner_scale=args.thin_refiner_scale,
            use_boundary_refiner=args.use_boundary_refiner,
            boundary_refiner_scale=args.boundary_refiner_scale,
            use_rare_class_residual=args.use_rare_class_residual,
            rare_class_indices=rare_class_indices,
            rare_class_scale=args.rare_class_scale,
        ).to(device)
    elif args.model_variant == "cacaf":
        model = CACafSegmentor(
            num_classes=NUM_CLASSES,
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
            use_cacaf=True,
            use_sagu=True,
        ).to(device)
    else:
        model = MMSABaselineSegmentor(
            num_classes=NUM_CLASSES,
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
        ).to(device)

    # 加载 checkpoint
    print(f"[load] 加载权重: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[load] checkpoint epoch={ckpt.get('epoch', 'N/A')}, best_mIoU={ckpt.get('best_miou', 'N/A')}")

    model.eval()

    # 评测数据集
    print(f"\n[data] 加载{split_cn}集...")
    eval_ds = FMBDataset(
        args.data_root, split=args.split,
        crop_size=args.crop_size, augment=False,
        eval_resize_mode=args.eval_resize_mode,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"[data] {split_cn}集样本数: {len(eval_ds)}")

    # 测试循环
    metric = SegMetric(NUM_CLASSES, IGNORE_INDEX)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"\n[eval] 开始评估（{args.split}）...")
    t0 = time.time()

    for idx, (rgb, thm, gt) in enumerate(eval_loader):
        rgb = rgb.to(device, non_blocking=True)
        thm = thm.to(device, non_blocking=True)

        if args.bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(rgb, thm, gt=None)
        else:
            out = model(rgb, thm, gt=None)

        pred = out["pred"].argmax(dim=1)
        pred_np = pred.cpu().numpy()
        gt_np = gt.numpy()

        for p, g in zip(pred_np, gt_np):
            metric.update(p, g)

        if (idx + 1) % 50 == 0:
            print(f"  进度: {idx + 1}/{len(eval_loader)}")

    elapsed = time.time() - t0
    print(f"\n[eval] 完成，耗时: {elapsed:.1f}s")

    # 计算指标
    results_present = metric.compute(absent_score=None)
    results_strict = metric.compute(absent_score=0.0)
    print("\n" + "=" * 60)
    print(f"{split_cn}集 present-only mIoU:  {results_present['mIoU']:.2f}")
    print(f"{split_cn}集 present-only mAcc:  {results_present['mAcc']:.2f}")
    print(f"{split_cn}集 strict(absent=0) mIoU:  {results_strict['mIoU']:.2f}")
    print(f"{split_cn}集 strict(absent=0) mAcc:  {results_strict['mAcc']:.2f}")
    print(f"{split_cn}集 aAcc:  {results_present['aAcc']:.2f}")
    print(f"有效类别数:  {results_present['num_valid']}/{NUM_CLASSES}")
    print("=" * 60)

    print("\n各类别 IoU (present-only):")
    for i, (cls, iou) in enumerate(zip(CLASSES, results_present["iou_per_class"])):
        marker = " [absent]" if i in results_present["absent_classes"] else ""
        print(f"  {cls:16s}: {iou:.1f}{marker}")

    if results_present["absent_classes"]:
        print(f"\n未出现的类别: {[CLASSES[i] for i in results_present['absent_classes']]}")

    # 保存结果（统一输出双口径）
    out_dir = osp.dirname(args.checkpoint)
    result_file = osp.join(out_dir, f"{args.split}_results.txt")
    with open(result_file, "w") as f:
        f.write(f"model_variant: {args.model_variant}\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"split: {args.split}\n")
        f.write(f"present_mIoU: {results_present['mIoU']:.2f}\n")
        f.write(f"present_mAcc: {results_present['mAcc']:.2f}\n")
        f.write(f"strict_mIoU: {results_strict['mIoU']:.2f}\n")
        f.write(f"strict_mAcc: {results_strict['mAcc']:.2f}\n")
        f.write(f"aAcc: {results_present['aAcc']:.2f}\n")
        f.write(f"valid_classes: {results_present['num_valid']}/{NUM_CLASSES}\n")
        f.write(
            f"absent_classes: {[CLASSES[i] for i in results_present['absent_classes']]}\n"
        )
        f.write("\nPer-class IoU (present-only):\n")
        for cls, iou in zip(CLASSES, results_present["iou_per_class"]):
            f.write(f"  {cls}: {iou:.1f}\n")
        f.write("\nPer-class IoU (strict absent=0):\n")
        for cls, iou in zip(CLASSES, results_strict["iou_per_class"]):
            f.write(f"  {cls}: {iou:.1f}\n")
    print(f"\n[result] 结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
