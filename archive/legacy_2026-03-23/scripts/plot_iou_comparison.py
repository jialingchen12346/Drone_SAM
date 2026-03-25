#!/usr/bin/env python3
"""
B4 可视化脚本 — 14 类 IoU 对比柱状图（v3 best vs 基线）

用法:
    cd /home/jl/Drone-SAM-Adapter
    python scripts/plot_iou_comparison.py \
        --out-dir work_dirs/visualizations/plots

输出:
    iou_comparison_bar.png   — 论文主图（双色柱状图 + 差值标注）
    iou_comparison_delta.png — 差值 Δ 单独图（正负颜色区分）
"""

import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# 数据（来自 SESSION_SNAPSHOT，全分辨率 800×600 评估结果）
# ---------------------------------------------------------------------------

CLASSES = [
    "Road", "Sidewalk", "Building", "Traffic\nLight", "Traffic\nSign",
    "Vegetation", "Sky", "Person", "Car", "Truck", "Bus",
    "Motorcycle", "Bicycle", "Pole",
]

BASELINE_IOU = [
    89.9,   # Road
    58.6,   # Sidewalk
    86.0,   # Building
    55.6,   # Traffic Light
    84.2,   # Traffic Sign
    88.0,   # Vegetation
    96.4,   # Sky
    76.8,   # Person
    86.0,   # Car
    38.0,   # Truck
    50.1,   # Bus
    57.2,   # Motorcycle
     0.0,   # Bicycle
    58.6,   # Pole
]

V3_IOU = [
    93.6,   # Road
    68.4,   # Sidewalk
    88.2,   # Building
    61.0,   # Traffic Light
    89.0,   # Traffic Sign
    90.6,   # Vegetation
    96.2,   # Sky
    72.9,   # Person
    87.0,   # Car
    62.1,   # Truck
    78.0,   # Bus
    67.5,   # Motorcycle
     0.0,   # Bicycle
    65.5,   # Pole
]

BASELINE_MIOU = 66.1
V3_MIOU       = 72.86

NUM_CLASSES = 14


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_bar_comparison(out_dir: str):
    """双色柱状图：基线 vs v3，差值标注在柱顶。"""
    x = np.arange(NUM_CLASSES)
    width = 0.38

    baseline = np.array(BASELINE_IOU)
    v3       = np.array(V3_IOU)
    delta    = v3 - baseline

    fig, ax = plt.subplots(figsize=(14, 5.5))

    bars_b = ax.bar(x - width / 2, baseline, width,
                    label=f"Baseline (mIoU={BASELINE_MIOU:.1f})",
                    color="#5c85d6", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_v = ax.bar(x + width / 2, v3, width,
                    label=f"Ours / v3   (mIoU={V3_MIOU:.2f})",
                    color="#e06c3e", alpha=0.85, edgecolor="white", linewidth=0.5)

    # 差值标注（只标注非零 delta）
    for i, (bv, vv, dv) in enumerate(zip(baseline, v3, delta)):
        if abs(dv) < 0.05:
            continue
        sign = "+" if dv >= 0 else ""
        color = "#2a7a2a" if dv >= 0 else "#aa2222"
        top = max(bv, vv) + 1.2
        ax.text(x[i], top, f"{sign}{dv:.1f}",
                ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold")

    # mIoU 参考线
    ax.axhline(BASELINE_MIOU, color="#5c85d6", linestyle="--",
               linewidth=1.0, alpha=0.6, label=f"_nolegend_")
    ax.axhline(V3_MIOU, color="#e06c3e", linestyle="--",
               linewidth=1.0, alpha=0.6, label=f"_nolegend_")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_ylabel("IoU (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title("Per-class IoU Comparison: Baseline vs Ours (FMB val, full-res 800×600)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = osp.join(out_dir, "iou_comparison_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_delta(out_dir: str):
    """Δ IoU 图（v3 − baseline），正负不同颜色。"""
    delta = np.array(V3_IOU) - np.array(BASELINE_IOU)
    colors = ["#2a7a2a" if d >= 0 else "#aa2222" for d in delta]

    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.bar(range(NUM_CLASSES), delta, color=colors,
                  alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    for i, d in enumerate(delta):
        sign = "+" if d >= 0 else ""
        ax.text(i, d + (0.6 if d >= 0 else -1.5),
                f"{sign}{d:.1f}", ha="center", va="bottom",
                fontsize=8, color="black")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_ylabel("ΔIoU (Ours − Baseline, %)", fontsize=11)
    ax.set_title("Per-class ΔIoU: Ours vs Baseline  "
                 f"(ΔmIoU = +{V3_MIOU - BASELINE_MIOU:.2f})",
                 fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    pos_patch = mpatches.Patch(color="#2a7a2a", alpha=0.85, label="Ours > Baseline")
    neg_patch = mpatches.Patch(color="#aa2222", alpha=0.85, label="Ours < Baseline")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10, loc="upper right")

    plt.tight_layout()
    path = osp.join(out_dir, "iou_comparison_delta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="work_dirs/visualizations/plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("生成 B4 IoU 对比图...")
    plot_bar_comparison(args.out_dir)
    plot_delta(args.out_dir)
    print("完成！")


if __name__ == "__main__":
    main()
