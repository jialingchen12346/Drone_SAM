#!/usr/bin/env python3
"""
Bus 类可视化脚本 — 对 test 集中含 Bus 的图像生成 RGB/Thermal/GT/Pred 四列对比图
按 Bus 像素数从多到少排序，取 top-N

用法:
    cd /home/jl/Drone-SAM-Adapter
    conda run -n sam2-unet python scripts/visualize_bus.py \
        --checkpoint work_dirs/cacaf_fmb_v3/epoch_60.pth \
        --out-dir work_dirs/visualizations/bus_debug \
        --num-images 20
"""
import argparse
import os
import os.path as osp
import sys
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.datasets.fmb_dataset import CLASSES, NUM_CLASSES, IGNORE_INDEX

PALETTE = np.array([
    [128,  64, 128],  #  0 Road
    [244,  35, 232],  #  1 Sidewalk
    [ 70,  70,  70],  #  2 Building
    [250, 170,  30],  #  3 Traffic Light
    [220, 220,   0],  #  4 Traffic Sign
    [107, 142,  35],  #  5 Vegetation
    [ 70, 130, 180],  #  6 Sky
    [220,  20,  60],  #  7 Person
    [  0,  60, 100],  #  8 Car
    [  0,  80, 100],  #  9 Truck
    [  0,   0, 142],  # 10 Bus
    [  0,   0, 230],  # 11 Motorcycle
    [119,  11,  32],  # 12 Bicycle
    [153, 153, 153],  # 13 Pole
], dtype=np.uint8)

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
THM_MEAN = [0.485, 0.456, 0.406]
THM_STD  = [0.229, 0.224, 0.225]
INFER_SIZE = 512
BUS_CLASS = 10  # 0-indexed


def label_to_color(lbl):
    h, w = lbl.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(PALETTE)):
        color[lbl == c] = PALETTE[c]
    color[lbl == 255] = [0, 0, 0]
    return color


def find_bus_images(data_root, split="test"):
    lbl_dir = osp.join(data_root, split, "Label")
    rgb_base = osp.join(data_root, split, "Visible")
    thm_dir  = osp.join(data_root, split, "Infrared")

    # build name→rgb path map (easy/hard subdirs)
    rgb_map = {}
    for sub in ["easy", "hard"]:
        for p in glob.glob(osp.join(rgb_base, sub, "*.png")):
            rgb_map[osp.basename(p)] = p

    bus_samples = []
    for lbl_path in sorted(glob.glob(osp.join(lbl_dir, "*.png"))):
        name = osp.basename(lbl_path)
        lbl = np.array(Image.open(lbl_path)).astype(np.int32) - 1
        bus_px = (lbl == BUS_CLASS).sum()
        if bus_px == 0:
            continue
        rgb_path = rgb_map.get(name)
        thm_path = osp.join(thm_dir, name)
        if rgb_path and osp.exists(thm_path):
            bus_samples.append((bus_px, name, rgb_path, thm_path, lbl_path))

    bus_samples.sort(key=lambda x: -x[0])
    return bus_samples


def run_inference(model, rgb_path, thm_path, device, dtype):
    rgb = Image.open(rgb_path).convert("RGB")
    thm = Image.open(thm_path).convert("RGB")
    orig_w, orig_h = rgb.size

    rgb_t = TF.resize(rgb, [INFER_SIZE, INFER_SIZE])
    thm_t = TF.resize(thm, [INFER_SIZE, INFER_SIZE])

    rgb_t = TF.normalize(TF.to_tensor(rgb_t), RGB_MEAN, RGB_STD).unsqueeze(0).to(device, dtype)
    thm_t = TF.normalize(TF.to_tensor(thm_t), THM_MEAN, THM_STD).unsqueeze(0).to(device, dtype)

    with torch.no_grad():
        out = model(rgb_t, thm_t)
        if isinstance(out, dict):
            out = out["pred"] if "pred" in out else list(out.values())[0]
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = F.interpolate(out.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred = pred.argmax(1).squeeze(0).cpu().numpy()
    return pred, np.array(rgb), np.array(thm.convert("L"))


def compute_bus_iou(pred, lbl):
    pred_bus = (pred == BUS_CLASS)
    gt_bus   = (lbl  == BUS_CLASS)
    tp = (pred_bus & gt_bus).sum()
    union = (pred_bus | gt_bus).sum()
    return tp / union if union > 0 else float("nan")


def make_panel(rgb_arr, thm_arr, gt_lbl, pred_lbl, name, bus_iou, bus_px):
    gt_color   = label_to_color(gt_lbl)
    pred_color = label_to_color(pred_lbl)

    thm_rgb = np.stack([thm_arr] * 3, axis=-1)

    h, w = rgb_arr.shape[:2]
    panel_w = w * 4 + 15  # 4 cols + 3 gaps of 5px
    panel_h = h + 30
    panel = Image.new("RGB", (panel_w, panel_h), (30, 30, 30))

    x = 0
    for img_arr, title in [
        (rgb_arr,    "RGB"),
        (thm_rgb,    "Thermal"),
        (gt_color,   "GT"),
        (pred_color, f"Pred  Bus IoU={bus_iou:.1f}%"),
    ]:
        panel.paste(Image.fromarray(img_arr.astype(np.uint8)), (x, 0))
        draw = ImageDraw.Draw(panel)
        draw.text((x + 4, h + 4), title, fill=(255, 255, 255))
        x += w + 5

    # Bus highlight overlay on pred
    bus_mask = (pred_lbl == BUS_CLASS)
    pred_hl = pred_color.copy()
    pred_hl[bus_mask] = [255, 0, 0]   # 红色高亮预测到的 Bus
    gt_mask  = (gt_lbl  == BUS_CLASS)
    gt_hl    = gt_color.copy()
    gt_hl[gt_mask] = [0, 255, 0]      # 绿色高亮 GT Bus

    hl_w = w * 2 + 5
    hl_panel = Image.new("RGB", (hl_w, panel_h), (30, 30, 30))
    hl_panel.paste(Image.fromarray(gt_hl.astype(np.uint8)), (0, 0))
    hl_panel.paste(Image.fromarray(pred_hl.astype(np.uint8)), (w + 5, 0))
    draw = ImageDraw.Draw(hl_panel)
    draw.text((4, h + 4), "GT Bus (green)", fill=(0, 255, 0))
    draw.text((w + 9, h + 4), "Pred Bus (red)", fill=(255, 80, 80))

    full = Image.new("RGB", (panel_w + hl_w + 5, panel_h), (10, 10, 10))
    full.paste(panel, (0, 0))
    full.paste(hl_panel, (panel_w + 5, 0))
    return full


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="work_dirs/cacaf_fmb_v3/epoch_60.pth")
    p.add_argument("--data-root",  default="/home/jl/dataset/FMB")
    p.add_argument("--split",      default="test")
    p.add_argument("--out-dir",    default="work_dirs/visualizations/bus_debug")
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--sam2-cfg",   default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt",  default="checkpoints/sam2_hiera_large.pt")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    print("Loading model...")
    model = CACafSegmentor(
        sam2_checkpoint=osp.join(_ROOT, args.sam2_ckpt),
        sam2_config=args.sam2_cfg,
        num_classes=NUM_CLASSES,
    ).to(device)
    ckpt = torch.load(osp.join(_ROOT, args.checkpoint), map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    bus_samples = find_bus_images(args.data_root, args.split)
    print(f"Found {len(bus_samples)} Bus images in {args.split} set")
    bus_samples = bus_samples[:args.num_images]

    iou_list = []
    for i, (bus_px, name, rgb_path, thm_path, lbl_path) in enumerate(bus_samples):
        gt_lbl = np.array(Image.open(lbl_path)).astype(np.int32) - 1
        gt_lbl[gt_lbl < 0] = 255

        with torch.autocast("cuda", dtype=dtype):
            pred, rgb_arr, thm_arr = run_inference(model, rgb_path, thm_path, device, dtype)

        bus_iou = compute_bus_iou(pred, gt_lbl) * 100
        iou_list.append(bus_iou)

        panel = make_panel(rgb_arr, thm_arr, gt_lbl, pred, name, bus_iou, bus_px)
        out_path = osp.join(args.out_dir, f"{i+1:02d}_{name.replace('.png','')}_busIoU{bus_iou:.0f}.png")
        panel.save(out_path)
        print(f"  [{i+1}/{len(bus_samples)}] {name}  Bus={bus_px}px  IoU={bus_iou:.1f}%  → {osp.basename(out_path)}")

    valid = [x for x in iou_list if not np.isnan(x)]
    print(f"\n平均 Bus IoU (top-{len(bus_samples)}): {np.mean(valid):.1f}%")
    print(f"输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
