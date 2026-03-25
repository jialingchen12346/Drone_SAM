#!/usr/bin/env python3
"""
B2 失败案例分析 — 按 Person / Bicycle 类别 IoU 排序，输出最差/最好的对比图

用法:
    cd /home/jl/Drone-SAM-Adapter
    conda activate sam2-unet
    python scripts/analyze_failure_cases.py \
        --checkpoint work_dirs/cacaf_fmb_v3/best.pth \
        --out-dir work_dirs/visualizations/failure_cases

输出:
    failure_cases/person_worst_N.png    — Person IoU 最差的 N 张
    failure_cases/person_best_N.png     — Person IoU 最好的 N 张（对比用）
    failure_cases/bicycle_all_N.png     — 含 Bicycle GT 像素的全部图
    failure_cases/per_image_iou.csv     — 全量 per-image IoU 数据（可用于进一步分析）
"""

import argparse
import csv
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional as TF

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.datasets.fmb_dataset import CLASSES, NUM_CLASSES, IGNORE_INDEX

# 复用 B1 脚本的常量和工具函数
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
THM_MEAN = [0.5, 0.5, 0.5]
THM_STD  = [0.5, 0.5, 0.5]
INFER_SIZE = 512

PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [ 70, 70, 70], [250, 170,  30],
    [220, 220,   0], [107, 142,  35], [ 70, 130, 180], [220,  20,  60],
    [  0,  60, 100], [  0,  80, 100], [  0,   0, 142], [  0,   0, 230],
    [119,  11,  32], [153, 153, 153],
], dtype=np.uint8)

CLASS_PERSON  = 7
CLASS_BICYCLE = 12


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def label_to_color(label_np):
    h, w = label_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color[label_np == c] = PALETTE[c]
    return color


def denorm_rgb(tensor):
    mean = torch.tensor(RGB_MEAN).view(3, 1, 1)
    std  = torch.tensor(RGB_STD).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    return (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


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
                    name=osp.splitext(fname)[0],
                    subset=subset,
                ))
    return samples


def per_image_class_iou(pred_np, gt_np, cls_id):
    """计算单张图对指定类别的 IoU。若 GT 无该类像素返回 nan。"""
    pred_c = (pred_np == cls_id)
    gt_c   = (gt_np  == cls_id)
    if gt_c.sum() == 0:
        return float("nan"), 0
    tp    = (pred_c & gt_c).sum()
    union = (pred_c | gt_c).sum()
    iou   = tp / union if union > 0 else 0.0
    return float(iou * 100), int(gt_c.sum())


@torch.no_grad()
def run_inference(model, rgb_pil, thm_pil, device, amp_dtype):
    sz = INFER_SIZE
    rgb_r = rgb_pil.resize((sz, sz), Image.BILINEAR)
    thm_r = thm_pil.resize((sz, sz), Image.BILINEAR)
    rgb_t = TF.normalize(TF.to_tensor(rgb_r), RGB_MEAN, RGB_STD).unsqueeze(0).to(device)
    thm_t = TF.normalize(TF.to_tensor(thm_r), THM_MEAN, THM_STD).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=amp_dtype):
        out = model(rgb_t, thm_t)
    pred = out["pred"].squeeze(0).argmax(0).cpu().numpy().astype(np.int32)
    return pred  # (512, 512)


def make_comparison_image(rgb_pil, thm_pil, gt_np, pred_np,
                          title: str, highlight_cls: int = None,
                          target_h: int = 300) -> Image.Image:
    """4 列对比图，可选高亮某个类别的 GT/Pred 差异。"""
    orig_w, orig_h = rgb_pil.size
    scale  = target_h / orig_h
    disp_w = int(orig_w * scale)

    def rs(img, mode=Image.BILINEAR):
        return img.resize((disp_w, target_h), mode)

    gt_color   = Image.fromarray(label_to_color(gt_np))
    pred_color = Image.fromarray(label_to_color(pred_np))

    panels = [
        (rs(rgb_pil),                       "RGB"),
        (rs(thm_pil),                       "Thermal"),
        (rs(gt_color,   Image.NEAREST),     "Ground Truth"),
        (rs(pred_color, Image.NEAREST),     "Prediction"),
    ]

    # 如果指定了 highlight_cls，再加一列差异图
    if highlight_cls is not None:
        diff = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
        tp_  = (gt_np == highlight_cls) & (pred_np == highlight_cls)
        fn_  = (gt_np == highlight_cls) & (pred_np != highlight_cls)
        fp_  = (gt_np != highlight_cls) & (pred_np == highlight_cls)
        diff[tp_] = [0, 200, 0]    # 绿=正确
        diff[fn_] = [255, 60, 60]  # 红=漏检
        diff[fp_] = [255, 200, 0]  # 黄=误检
        panels.append((rs(Image.fromarray(diff), Image.NEAREST),
                       f"{CLASSES[highlight_cls]} diff"))

    total_w = disp_w * len(panels)
    header_h = 18
    canvas = Image.new("RGB", (total_w, header_h + target_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), title, fill=(210, 210, 210))

    for i, (img, label) in enumerate(panels):
        img = img.copy()
        ImageDraw.Draw(img).text((3, 2), label, fill=(255, 255, 255))
        canvas.paste(img, (i * disp_w, header_h))

    return canvas


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="work_dirs/cacaf_fmb_v3/best.pth")
    p.add_argument("--data-root",  default="/home/jl/dataset/FMB")
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--out-dir",    default="work_dirs/visualizations/failure_cases")
    p.add_argument("--top-n",      type=int, default=8,
                   help="输出最差/最好的 top-N 张图")
    p.add_argument("--target-h",   type=int, default=280)
    p.add_argument("--sam2-cfg",   default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt",  default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--bf16",       action="store_true", default=True)
    p.add_argument("--no-bf16",    dest="bf16", action="store_false")
    return p.parse_args()


def stack_images(images, cols=2):
    """把多张等宽等高的 PIL 图片拼成 grid。"""
    if not images:
        return None
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (w * cols, h * rows), (10, 10, 10))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * w, r * h))
    return canvas


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # 加载模型
    print("加载模型...")
    model = CACafSegmentor(
        sam2_checkpoint=args.sam2_ckpt,
        sam2_config=args.sam2_cfg,
        num_classes=NUM_CLASSES,
        use_dice=True,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}, best_mIoU={ckpt.get('best_miou',0):.2f}")

    samples = load_sample_list(args.data_root, args.split)
    print(f"  共 {len(samples)} 张图片")

    # ---------------------------------------------------------------
    # 逐张推理 + 统计 per-image IoU
    # ---------------------------------------------------------------
    records = []  # list of dict

    for idx, s in enumerate(samples):
        rgb_pil = Image.open(s["rgb"]).convert("RGB")
        thm_pil = Image.open(s["thm"]).convert("RGB")
        gt_raw  = np.array(Image.open(s["lbl"]), dtype=np.int64)
        gt_np   = gt_raw - 1
        gt_np[gt_np < 0] = IGNORE_INDEX

        pred_np = run_inference(model, rgb_pil, thm_pil, device, amp_dtype)

        # pred 是 512×512，需要 resize 到 GT 尺寸
        orig_h, orig_w = gt_np.shape
        if pred_np.shape != (orig_h, orig_w):
            pred_pil_tmp = Image.fromarray(pred_np.astype(np.uint8)).resize(
                (orig_w, orig_h), Image.NEAREST)
            pred_np = np.array(pred_pil_tmp).astype(np.int32)

        person_iou,  person_px  = per_image_class_iou(pred_np, gt_np, CLASS_PERSON)
        bicycle_iou, bicycle_px = per_image_class_iou(pred_np, gt_np, CLASS_BICYCLE)

        records.append(dict(
            name=s["name"], subset=s["subset"],
            rgb_path=s["rgb"], thm_path=s["thm"],
            gt_np=gt_np, pred_np=pred_np,
            person_iou=person_iou,   person_px=person_px,
            bicycle_iou=bicycle_iou, bicycle_px=bicycle_px,
        ))

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(samples)}]")

    print("推理完成，生成输出图像...")

    # ---------------------------------------------------------------
    # 保存 per-image CSV
    # ---------------------------------------------------------------
    csv_path = osp.join(args.out_dir, "per_image_iou.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "subset", "person_iou", "person_px",
                    "bicycle_iou", "bicycle_px"])
        for r in records:
            w.writerow([r["name"], r["subset"],
                        f"{r['person_iou']:.2f}" if not np.isnan(r["person_iou"]) else "nan",
                        r["person_px"],
                        f"{r['bicycle_iou']:.2f}" if not np.isnan(r["bicycle_iou"]) else "nan",
                        r["bicycle_px"]])
    print(f"  CSV 已保存: {csv_path}")

    # ---------------------------------------------------------------
    # Person 最差 / 最好各 top-N
    # ---------------------------------------------------------------
    person_records = [r for r in records if not np.isnan(r["person_iou"])]
    person_sorted  = sorted(person_records, key=lambda r: r["person_iou"])

    for tag, subset_recs in [("worst", person_sorted[:args.top_n]),
                              ("best",  person_sorted[-args.top_n:])]:
        imgs = []
        for r in subset_recs:
            rgb_pil = Image.open(r["rgb_path"]).convert("RGB")
            thm_pil = Image.open(r["thm_path"]).convert("RGB")
            title = (f"{r['name']}[{r['subset']}]  "
                     f"Person IoU={r['person_iou']:.1f}%  px={r['person_px']}")
            comp = make_comparison_image(
                rgb_pil, thm_pil, r["gt_np"], r["pred_np"],
                title=title, highlight_cls=CLASS_PERSON,
                target_h=args.target_h,
            )
            imgs.append(comp)
        grid = stack_images(imgs, cols=1)
        if grid:
            path = osp.join(args.out_dir, f"person_{tag}{args.top_n}.png")
            grid.save(path)
            print(f"  保存: {path}")

    # ---------------------------------------------------------------
    # Bicycle — 含 GT 像素的全部图
    # ---------------------------------------------------------------
    bicycle_records = [r for r in records if r["bicycle_px"] > 0]
    print(f"  含 Bicycle GT 像素的图片: {len(bicycle_records)} 张")
    bicycle_sorted = sorted(bicycle_records, key=lambda r: r["bicycle_px"], reverse=True)

    imgs = []
    for r in bicycle_sorted:
        rgb_pil = Image.open(r["rgb_path"]).convert("RGB")
        thm_pil = Image.open(r["thm_path"]).convert("RGB")
        title = (f"{r['name']}[{r['subset']}]  "
                 f"Bicycle IoU={r['bicycle_iou']:.1f}%  px={r['bicycle_px']}")
        comp = make_comparison_image(
            rgb_pil, thm_pil, r["gt_np"], r["pred_np"],
            title=title, highlight_cls=CLASS_BICYCLE,
            target_h=args.target_h,
        )
        imgs.append(comp)

    if imgs:
        grid = stack_images(imgs, cols=1)
        path = osp.join(args.out_dir, f"bicycle_all{len(imgs)}.png")
        grid.save(path)
        print(f"  保存: {path}")
    else:
        print("  val 集无含 Bicycle GT 的图片")

    # ---------------------------------------------------------------
    # 统计摘要
    # ---------------------------------------------------------------
    person_ious  = [r["person_iou"]  for r in person_records]
    print(f"\n--- Person 统计（{len(person_ious)} 张含 GT）---")
    print(f"  mean={np.mean(person_ious):.1f}  "
          f"median={np.median(person_ious):.1f}  "
          f"min={np.min(person_ious):.1f}  "
          f"max={np.max(person_ious):.1f}")
    print(f"  IoU<25: {sum(v<25 for v in person_ious)} 张  "
          f"IoU<50: {sum(v<50 for v in person_ious)} 张")

    print(f"\n完成！结果保存至: {args.out_dir}")


if __name__ == "__main__":
    main()
