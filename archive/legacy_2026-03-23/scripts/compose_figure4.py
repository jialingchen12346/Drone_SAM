#!/usr/bin/env python3
"""
Compose Figure 4 paper subfigures into a single publication-ready panel.

Default layout:
    (a) easy vs hard thermal-weight summary
    (b) representative-sample weight profiles
"""

import argparse
import os
import os.path as osp

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if osp.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def parse_args():
    p = argparse.ArgumentParser(description="Compose Figure 4 subfigures into one paper figure")
    p.add_argument(
        "--fig4a",
        default="work_dirs/paper_figures/figure4_final/figure4a_easy_hard_thermal.png",
    )
    p.add_argument(
        "--fig4b",
        default="work_dirs/paper_figures/figure4_final/figure4b_sample_profiles.png",
    )
    p.add_argument(
        "--out",
        default="work_dirs/paper_figures/figure4_final/figure4_combined.png",
    )
    p.add_argument("--outer-pad", type=int, default=20)
    p.add_argument("--panel-gap", type=int, default=24)
    p.add_argument("--label-pad-x", type=int, default=10)
    p.add_argument("--label-pad-y", type=int, default=8)
    p.add_argument("--label-font-size", type=int, default=24)
    p.add_argument("--max-width", type=int, default=2200)
    return p.parse_args()


def fit_to_width(img: Image.Image, max_width: int) -> Image.Image:
    if img.width <= max_width:
        return img
    new_h = int(img.height * max_width / img.width)
    return img.resize((max_width, new_h), Image.Resampling.LANCZOS)


def add_panel_label(base: Image.Image, label: str, font) -> Image.Image:
    out = base.copy()
    draw = ImageDraw.Draw(out)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0, y0 = 0, 0
    rect = [x0, y0, x0 + text_w + 20, y0 + text_h + 14]
    draw.rounded_rectangle(rect, radius=8, fill=(255, 255, 255), outline=(205, 210, 216), width=1)
    draw.text((10, 7), label, fill=(25, 25, 25), font=font)
    return out


def main():
    args = parse_args()
    os.makedirs(osp.dirname(args.out), exist_ok=True)

    fig4a = Image.open(args.fig4a).convert("RGB")
    fig4b = Image.open(args.fig4b).convert("RGB")

    target_inner_w = min(max(fig4a.width, fig4b.width), args.max_width)
    fig4a = fit_to_width(fig4a, target_inner_w)
    fig4b = fit_to_width(fig4b, target_inner_w)

    font = load_font(args.label_font_size)
    fig4a = add_panel_label(fig4a, "(a)", font)
    fig4b = add_panel_label(fig4b, "(b)", font)

    canvas_w = max(fig4a.width, fig4b.width) + args.outer_pad * 2
    canvas_h = fig4a.height + fig4b.height + args.panel_gap + args.outer_pad * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    x_a = (canvas_w - fig4a.width) // 2
    x_b = (canvas_w - fig4b.width) // 2
    y_a = args.outer_pad
    y_b = y_a + fig4a.height + args.panel_gap

    canvas.paste(fig4a, (x_a, y_a))
    canvas.paste(fig4b, (x_b, y_b))

    canvas.save(args.out)
    print(f"Saved combined figure: {args.out}")


if __name__ == "__main__":
    main()
