#!/usr/bin/env python3
"""
BN 校准版消融脚本

针对 no_cacaf 系列配置，在推理时消融基础上增加 BN 统计重校准：
  1. 应用旁路 patch（同 quick_ablation_eval.py）
  2. 重置 decoder 中所有 BN 层的 running_mean/var
  3. 用训练集前 N 张图跑 forward（no_grad），让 BN 在新特征分布下重新积累统计
  4. 恢复 eval 模式，正常评估

解决 quick_ablation_eval.py 中 A3/A1 因 BN 分布错配而崩溃（mIoU≈0）的问题。

用法:
    cd /home/jl/Drone-SAM-Adapter && conda activate sam2-unet
    python scripts/ablation_bn_calib.py \
        --checkpoint work_dirs/cacaf_fmb_v3/best.pth \
        --calib-batches 300        # 校准用训练样本数，默认300

输出: work_dirs/ablation_bn_calib/results.csv
"""

import argparse
import csv
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.datasets.fmb_dataset import CLASSES, NUM_CLASSES, IGNORE_INDEX

RGB_MEAN   = [0.485, 0.456, 0.406]
RGB_STD    = [0.229, 0.224, 0.225]
THM_MEAN   = [0.5, 0.5, 0.5]
THM_STD    = [0.5, 0.5, 0.5]
INFER_SIZE = 512


# ---------------------------------------------------------------------------
# Bypass patches（与 quick_ablation_eval.py 相同）
# ---------------------------------------------------------------------------

def patch_no_cacaf(model):
    def bypass_fusion(rgb_features, aux_features):
        fused = []
        for i, block in enumerate(model.fusion.blocks):
            r = block.align_rgb(rgb_features[i])
            a = block.align_aux(aux_features[i])
            fused.append((r + a) * 0.5)
        B = rgb_features[0].shape[0]
        dummy = torch.full((B, 2), 0.5, device=rgb_features[0].device)
        return fused, [dummy] * 4
    orig = model.fusion.forward
    model.fusion.forward = bypass_fusion
    return orig


def restore_cacaf(model, orig):
    model.fusion.forward = orig


def patch_no_sagu(model):
    orig = {}
    if not hasattr(model.decoder, 'sagu4'):
        return orig
    for name in ['sagu1', 'sagu2', 'sagu3', 'sagu4']:
        sagu = getattr(model.decoder, name)
        orig[name] = sagu.forward
        sagu.forward = lambda x, _s=sagu: x
    return orig


def restore_sagu(model, orig):
    for name, fwd in orig.items():
        getattr(model.decoder, name).forward = fwd


# ---------------------------------------------------------------------------
# BN 校准
# ---------------------------------------------------------------------------

def calibrate_bn(model, train_samples, device, amp_dtype, num_batches):
    """
    重置 decoder（+fusion）中所有 BN 层的 running stats，
    然后用 num_batches 张训练图做 forward（no_grad）重新积累统计。
    其他模块（dropout 等）保持 eval 模式不受影响。
    """
    import random

    # 找出所有需要校准的 BN 层（decoder + fusion）
    bn_layers = []
    for module in [model.decoder, model.fusion]:
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                bn_layers.append(m)

    if not bn_layers:
        print("  [BN校准] 未找到 BN 层，跳过校准")
        return

    print(f"  [BN校准] 重置 {len(bn_layers)} 个 BN 层，"
          f"用 {num_batches} 张训练图重新校准...")

    # 重置 running stats 并临时切到 train 模式（仅 BN 层）
    for m in bn_layers:
        m.reset_running_stats()
        m.train()

    cal_samples = random.sample(train_samples, min(num_batches, len(train_samples)))

    with torch.no_grad():
        for i, s in enumerate(cal_samples):
            rgb_pil = Image.open(s["rgb"]).convert("RGB")
            thm_pil = Image.open(s["thm"]).convert("RGB")
            sz = INFER_SIZE
            rgb_t = TF.normalize(
                TF.to_tensor(rgb_pil.resize((sz, sz), Image.BILINEAR)),
                RGB_MEAN, RGB_STD
            ).unsqueeze(0).to(device)
            thm_t = TF.normalize(
                TF.to_tensor(thm_pil.resize((sz, sz), Image.BILINEAR)),
                THM_MEAN, THM_STD
            ).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=amp_dtype):
                model(rgb_t, thm_t)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(cal_samples)}")

    # 恢复 eval 模式
    for m in bn_layers:
        m.eval()

    print("  [BN校准] 完成")


# ---------------------------------------------------------------------------
# 数据 & 评估
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
        rgb_t = TF.normalize(
            TF.to_tensor(rgb_pil.resize((sz, sz), Image.BILINEAR)),
            RGB_MEAN, RGB_STD
        ).unsqueeze(0).to(device)
        thm_t = TF.normalize(
            TF.to_tensor(thm_pil.resize((sz, sz), Image.BILINEAR)),
            THM_MEAN, THM_STD
        ).unsqueeze(0).to(device)

        with torch.autocast("cuda", dtype=amp_dtype):
            out = model(rgb_t, thm_t)
        pred = out["pred"].squeeze(0).argmax(0).cpu().numpy().astype(np.int32)

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

# (bypass_cacaf, bypass_sagu, need_bn_calib)
ABLATION_CONFIGS = [
    ("A4_full",          False, False, False, "完整模型（v3 best，基准）"),
    ("A2_no_sagu",       False, True,  False, "旁路 SAGU（+CACAF，-HGSOAD注意力）"),
    ("A3_no_cacaf",      True,  False, True,  "旁路 CACAF（-融合注意力，+SAGU）+ BN校准"),
    ("A1_no_cacaf_sagu", True,  True,  True,  "旁路 CACAF+SAGU（简单平均+标准UNet）+ BN校准"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="work_dirs/cacaf_fmb_v3/best.pth")
    p.add_argument("--data-root",     default="/home/jl/dataset/FMB")
    p.add_argument("--split",         default="val", choices=["val", "test"])
    p.add_argument("--out-dir",       default="work_dirs/ablation_bn_calib")
    p.add_argument("--sam2-cfg",      default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt",     default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--calib-batches", type=int, default=300,
                   help="BN 校准用的训练样本数（越多越准，越慢）")
    p.add_argument("--bf16",          action="store_true", default=True)
    p.add_argument("--no-bf16",       dest="bf16", action="store_false")
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

    val_samples   = load_sample_list(args.data_root, args.split)
    train_samples = load_sample_list(args.data_root, "train")
    print(f"  val={len(val_samples)} 张，train={len(train_samples)} 张（供 BN 校准）\n")

    results = []

    for config_id, bypass_cacaf, bypass_sagu, need_calib, desc in ABLATION_CONFIGS:
        print(f"[{config_id}] {desc}")

        orig_cacaf = patch_no_cacaf(model) if bypass_cacaf else None
        orig_sagu  = patch_no_sagu(model)  if bypass_sagu  else {}

        if need_calib:
            calibrate_bn(model, train_samples, device, amp_dtype, args.calib_batches)

        metrics = evaluate(model, val_samples, device, amp_dtype)

        if orig_cacaf is not None:
            restore_cacaf(model, orig_cacaf)
            # BN stats 已更新，恢复后需重新用训练集或重载 checkpoint 来还原
            # 这里每个配置评估完后重载权重，确保下一配置不受污染
        if orig_sagu:
            restore_sagu(model, orig_sagu)

        if need_calib:
            # 重载 checkpoint，避免 BN stats 污染后续配置
            ckpt2 = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt2["model"])
            model.eval()

        results.append((config_id, desc, metrics))
        print(f"  mIoU={metrics['mIoU']:.2f}  mAcc={metrics['mAcc']:.2f}  "
              f"aAcc={metrics['aAcc']:.2f}\n")

    # Summary
    print("=" * 72)
    print(f"{'配置':<25} {'mIoU':>7} {'Δ mIoU':>8} {'mAcc':>7} {'aAcc':>7}  说明")
    print("-" * 72)
    base_miou = results[0][2]["mIoU"]
    for config_id, desc, m in results:
        delta = m["mIoU"] - base_miou
        sign  = "+" if delta >= 0 else ""
        print(f"{config_id:<25} {m['mIoU']:>7.2f} {sign+f'{delta:.2f}':>8} "
              f"{m['mAcc']:>7.2f} {m['aAcc']:>7.2f}")
    print("=" * 72)
    note = "(no_cacaf 系列已做 BN 校准，结果比 zero-shot 消融更可靠；" \
           "但仍非重训，需标注)"
    print(f"注：{note}\n")

    csv_path = osp.join(args.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "desc", "mIoU", "delta_mIoU", "mAcc", "aAcc",
                    "bn_calibrated"]
                   + [f"iou_{c}" for c in CLASSES])
        for config_id, desc, m in results:
            delta = m["mIoU"] - base_miou
            calib = "yes" if "BN校准" in desc else "no"
            w.writerow([config_id, desc,
                        f"{m['mIoU']:.2f}", f"{delta:+.2f}",
                        f"{m['mAcc']:.2f}", f"{m['aAcc']:.2f}", calib]
                       + [f"{v:.2f}" for v in m["iou_per_class"]])
    print(f"CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
