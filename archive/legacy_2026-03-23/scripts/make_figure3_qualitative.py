#!/usr/bin/env python3
"""
Generate paper-ready Figure 3 qualitative comparisons for FMB.

Output format:
    one row per sample, four or five columns:
        RGB | Thermal | Ground Truth | Ours
        RGB | Thermal | Ground Truth | MM SAM-Adapter | Ours

The script uses the same full-resolution inference protocol as eval_fullres.py:
    - keep original image resolution
    - pad to multiples of 32
    - crop prediction back to original size

Example:
    conda run --no-capture-output -n sam2-unet \
        python scripts/make_figure3_qualitative.py \
            --checkpoint work_dirs/v5_ablation_no_ohem/epoch_60.pth \
            --data-root /home/jl/dataset/FMB \
            --split test \
            --subset hard \
            --samples 00043,00381,00501,01046 \
            --out-dir work_dirs/paper_figures/figure3
"""

import argparse
import contextlib
import os
import os.path as osp
import sys
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torch.amp import autocast
import torchvision.transforms.functional as TF

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.datasets.fmb_dataset import (  # noqa: E402
    CLASSES,
    NUM_CLASSES,
    IGNORE_INDEX,
    RGB_MEAN,
    RGB_STD,
    THM_MEAN,
    THM_STD,
)
from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor  # noqa: E402


PALETTE = np.array([
    [128, 64, 128],   # Road
    [244, 35, 232],   # Sidewalk
    [70, 70, 70],     # Building
    [250, 170, 30],   # Traffic Light
    [220, 220, 0],    # Traffic Sign
    [107, 142, 35],   # Vegetation
    [70, 130, 180],   # Sky
    [220, 20, 60],    # Person
    [0, 60, 100],     # Car
    [0, 80, 100],     # Truck
    [0, 0, 142],      # Bus
    [0, 0, 230],      # Motorcycle
    [119, 11, 32],    # Bicycle
    [153, 153, 153],  # Pole
], dtype=np.uint8)


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if osp.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 3 qualitative comparison figure")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    p.add_argument("--data-root", default="/home/jl/dataset/FMB")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--subset", default="all", choices=["all", "easy", "hard"])
    p.add_argument("--samples", default="",
                   help="Comma-separated basenames, e.g. 00043,00381,00501")
    p.add_argument("--samples-file", default=None,
                   help="Optional text file with one basename per line")
    p.add_argument("--num-images", type=int, default=4,
                   help="Used only when no explicit samples are provided")
    p.add_argument("--target-width", type=int, default=280,
                   help="Display width of each panel in the paper figure")
    p.add_argument("--panel-gap", type=int, default=8)
    p.add_argument("--row-gap", type=int, default=14)
    p.add_argument("--header-h", type=int, default=12)
    p.add_argument("--label-h", type=int, default=24)
    p.add_argument("--outer-pad", type=int, default=12)
    p.add_argument("--sample-label-w", type=int, default=92)
    p.add_argument("--show-figure-title", action="store_true", default=False)
    p.add_argument("--out-dir", default="work_dirs/paper_figures/figure3")
    p.add_argument("--out-name", default="figure3_qualitative.png")
    p.add_argument(
        "--baseline-vis-dir",
        default="/home/jl/Multimodal-SAM-Adapter/segmentation/work_dirs/FMB_visualization/prediction",
        help="Directory containing baseline qualitative PNGs organized by subset/name.png",
    )
    p.add_argument(
        "--baseline-name",
        default="MM SAM-Adapter",
        help="Column title for the baseline visualization",
    )
    p.add_argument(
        "--render-mode",
        default="overlay",
        choices=["overlay", "mask"],
        help="How to render our prediction panels",
    )
    p.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Segmentation overlay opacity when render-mode=overlay",
    )
    p.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt", default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--no-cacaf", action="store_true", default=False)
    p.add_argument("--no-sagu", action="store_true", default=False)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    return p.parse_args()


def load_sample_list(root: str, split: str):
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
                    name=osp.splitext(fname)[0],
                    subset=subset,
                    rgb=osp.join(rgb_dir, subset, fname),
                    thm=osp.join(thm_dir, fname),
                    lbl=osp.join(lbl_dir, fname),
                ))
    return samples


def parse_requested_names(args) -> List[str]:
    names: List[str] = []
    if args.samples.strip():
        names.extend([x.strip() for x in args.samples.split(",") if x.strip()])
    if args.samples_file:
        with open(args.samples_file) as f:
            names.extend([line.strip() for line in f if line.strip()])
    # Keep order, dedupe.
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered


def filter_samples(samples, subset: str, requested_names: List[str], num_images: int):
    if subset != "all":
        samples = [s for s in samples if s["subset"] == subset]

    if requested_names:
        by_name = {s["name"]: s for s in samples}
        missing = [n for n in requested_names if n not in by_name]
        if missing:
            raise ValueError(f"Requested samples not found in split={subset}: {missing}")
        return [by_name[n] for n in requested_names]

    if num_images > 0:
        return samples[:num_images]
    return samples


def pad_to_multiple(img_t: torch.Tensor, multiple=32):
    _, _, h, w = img_t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img_t, h, w
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
    return img_t, h, w


def label_to_color(label_np: np.ndarray) -> np.ndarray:
    h, w = label_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[label_np == c] = PALETTE[c]
    out[label_np == IGNORE_INDEX] = 0
    return out


def blend_segmentation(rgb_pil: Image.Image, label_np: np.ndarray, opacity: float) -> Image.Image:
    color_np = label_to_color(label_np)
    rgb_np = np.array(rgb_pil.convert("RGB"), dtype=np.float32)
    color_np = color_np.astype(np.float32)
    valid = (label_np != IGNORE_INDEX)[..., None].astype(np.float32)
    blended = rgb_np * (1.0 - opacity * valid) + color_np * (opacity * valid)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def resize_label_nearest(label_np: np.ndarray, size) -> np.ndarray:
    label_img = Image.fromarray(label_np.astype(np.uint8), mode="L")
    return np.array(label_img.resize(size, Image.Resampling.NEAREST), dtype=np.int32)


def render_mask_panel(label_np: np.ndarray, size) -> Image.Image:
    return Image.fromarray(label_to_color(resize_label_nearest(label_np, size)))


def render_overlay_panel(rgb_pil: Image.Image, label_np: np.ndarray, size, opacity: float) -> Image.Image:
    rgb_disp = rgb_pil.resize(size, Image.Resampling.BILINEAR)
    label_disp = resize_label_nearest(label_np, size)
    return blend_segmentation(rgb_disp, label_disp, opacity)


@torch.no_grad()
def run_fullres_inference(model, rgb_pil: Image.Image, thm_pil: Image.Image, device, amp_dtype):
    rgb_t = TF.normalize(TF.to_tensor(rgb_pil), RGB_MEAN, RGB_STD).unsqueeze(0)
    thm_t = TF.normalize(TF.to_tensor(thm_pil), THM_MEAN, THM_STD).unsqueeze(0)

    rgb_t, orig_h, orig_w = pad_to_multiple(rgb_t)
    thm_t, _, _ = pad_to_multiple(thm_t)
    rgb_t = rgb_t.to(device)
    thm_t = thm_t.to(device)

    amp_ctx = (
        autocast("cuda", dtype=amp_dtype)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with amp_ctx:
        out = model(rgb_t, thm_t, gt=None)

    pred = out["pred"][:, :, :orig_h, :orig_w]
    pred = pred.argmax(dim=1).cpu().numpy()[0].astype(np.int32)
    return pred


def make_panel(
    img: Image.Image,
    title: str,
    label_h: int,
    title_font,
    border_color=(210, 214, 220),
):
    width, disp_h = img.size
    canvas = Image.new("RGB", (width, disp_h + label_h), (255, 255, 255))
    canvas.paste(img.convert("RGB"), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width - 1, label_h - 1], fill=(246, 247, 249))
    draw.rectangle([0, 0, width - 1, disp_h + label_h - 1], outline=border_color, width=1)
    bbox = draw.textbbox((0, 0), title, font=title_font)
    text_h = bbox[3] - bbox[1]
    text_y = max(0, (label_h - text_h) // 2 - 1)
    draw.text((8, text_y), title, fill=(32, 36, 42), font=title_font)
    return canvas


def make_row(
    rgb_pil,
    thm_pil,
    gt_np,
    pred_np,
    target_width: int,
    label_h: int,
    panel_gap: int,
    render_mode: str,
    opacity: float,
    baseline_img: Optional[Image.Image] = None,
    baseline_name: str = "MM SAM-Adapter",
    title_font=None,
):
    disp_h = int(rgb_pil.height * target_width / rgb_pil.width)
    disp_size = (target_width, disp_h)

    rgb_img = rgb_pil.resize(disp_size, Image.Resampling.BILINEAR)
    thm_img = thm_pil.resize(disp_size, Image.Resampling.BILINEAR)
    gt_img = render_mask_panel(gt_np, disp_size)
    if render_mode == "overlay":
        pred_img = render_overlay_panel(rgb_pil, pred_np, disp_size, opacity)
    else:
        pred_img = render_mask_panel(pred_np, disp_size)

    if baseline_img is not None:
        baseline_img = baseline_img.resize(disp_size, Image.Resampling.BILINEAR)

    panels = [
        make_panel(rgb_img, "RGB", label_h, title_font),
        make_panel(thm_img, "Thermal", label_h, title_font),
        make_panel(gt_img, "Ground Truth", label_h, title_font),
    ]
    if baseline_img is not None:
        panels.append(make_panel(baseline_img, baseline_name, label_h, title_font))
    panels.append(make_panel(pred_img, "Ours", label_h, title_font))

    row_h = max(p.height for p in panels)
    row_w = sum(p.width for p in panels) + panel_gap * (len(panels) - 1)
    row = Image.new("RGB", (row_w, row_h), (255, 255, 255))

    x = 0
    for p in panels:
        y = (row_h - p.height) // 2
        row.paste(p, (x, y))
        x += p.width + panel_gap
    return row


def compose_figure(
    rows,
    sample_labels: List[str],
    header_h: int,
    row_gap: int,
    outer_pad: int,
    sample_label_w: int,
    show_figure_title: bool,
):
    if not rows:
        raise ValueError("No rows to compose.")

    title_font = load_font(19)
    sample_font = load_font(16)
    top_pad = header_h if show_figure_title else 0
    figure_w = max(r.width for r in rows) + sample_label_w + outer_pad * 2
    figure_h = top_pad + sum(r.height for r in rows) + row_gap * (len(rows) - 1) + outer_pad * 2
    canvas = Image.new("RGB", (figure_w, figure_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if show_figure_title:
        draw.text(
            (outer_pad, max(0, (header_h - 18) // 2)),
            "Figure 3. Qualitative comparisons on representative FMB scenes",
            fill=(18, 18, 18),
            font=title_font,
        )

    y = outer_pad + top_pad
    for label, row in zip(sample_labels, rows):
        bbox = draw.textbbox((0, 0), label, font=sample_font)
        text_h = bbox[3] - bbox[1]
        label_x = outer_pad
        label_y = y + max(0, (row.height - text_h) // 2 - 1)
        draw.text((label_x, label_y), label, fill=(70, 74, 80), font=sample_font)
        canvas.paste(row, (outer_pad + sample_label_w, y))
        y += row.height + row_gap
    return canvas


def resolve_baseline_path(base_dir: str, subset: str, name: str) -> Optional[str]:
    if not base_dir:
        return None
    path = osp.join(base_dir, subset, f"{name}.png")
    return path if osp.exists(path) else None


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    print("Loading model...")
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
    print(f"  checkpoint: {args.checkpoint}")

    samples = load_sample_list(args.data_root, args.split)
    requested = parse_requested_names(args)
    samples = filter_samples(samples, args.subset, requested, args.num_images)
    print(f"Selected {len(samples)} samples")

    title_font = load_font(14)
    rows = []
    labels = []
    for i, s in enumerate(samples, start=1):
        rgb_pil = Image.open(s["rgb"]).convert("RGB")
        thm_pil = Image.open(s["thm"]).convert("RGB")
        gt_np = np.array(Image.open(s["lbl"]), dtype=np.int64)
        gt_np -= 1
        gt_np[gt_np < 0] = IGNORE_INDEX

        pred_np = run_fullres_inference(model, rgb_pil, thm_pil, device, amp_dtype)
        baseline_path = resolve_baseline_path(args.baseline_vis_dir, s["subset"], s["name"])
        baseline_img = Image.open(baseline_path).convert("RGB") if baseline_path else None

        row = make_row(
            rgb_pil, thm_pil, gt_np, pred_np,
            target_width=args.target_width,
            label_h=args.label_h,
            panel_gap=args.panel_gap,
            render_mode=args.render_mode,
            opacity=args.opacity,
            baseline_img=baseline_img,
            baseline_name=args.baseline_name,
            title_font=title_font,
        )
        rows.append(row)
        labels.append(f"({chr(96 + i)}) {s['name']}")

        single_path = osp.join(args.out_dir, f"row_{i:02d}_{s['subset']}_{s['name']}.png")
        row.save(single_path)
        print(f"  saved row: {single_path}")

    figure = compose_figure(
        rows,
        labels,
        args.header_h,
        args.row_gap,
        args.outer_pad,
        args.sample_label_w,
        args.show_figure_title,
    )
    out_path = osp.join(args.out_dir, args.out_name)
    figure.save(out_path)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
