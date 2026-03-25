#!/usr/bin/env python3
"""
B3 CACAF 融合权重可视化

fusion_weights: 4 个 [B, 2] 张量（每张图像），[:, 0]=w_rgb, [:, 1]=w_aux
这些是 MQA 输出的逐图像自适应标量权重（非空间热图），通过 softmax 归一化。

输出:
    fusion_weights_dist.png  — 4 个 level 的权重分布（boxplot + 均值线）
    fusion_weights_scatter.png — 每张图 w_rgb vs w_aux 的散点图（按 level）
    fusion_weights_stats.csv — 全量统计数据

用法:
    cd /home/jl/Drone-SAM-Adapter
    conda activate sam2-unet
    python scripts/visualize_fusion_weights.py \
        --checkpoint work_dirs/cacaf_fmb_v3/best.pth \
        --split val \
        --out-dir work_dirs/visualizations/plots
"""

import argparse
import csv
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segmentation.models.segmentors.cacaf_segmentor import CACafSegmentor
from segmentation.datasets.fmb_dataset import NUM_CLASSES

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
THM_MEAN = [0.5, 0.5, 0.5]
THM_STD  = [0.5, 0.5, 0.5]
INFER_SIZE = 512
LEVEL_NAMES = ["f1 (×4, 128²)", "f2 (×8, 64²)", "f3 (×16, 32²)", "f4 (×32, 16²)"]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_sample_list(data_root, split):
    samples = []
    rgb_dir = osp.join(data_root, split, "Visible")
    thm_dir = osp.join(data_root, split, "Infrared")
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
                    name=osp.splitext(fname)[0],
                    subset=subset,
                ))
    return samples


@torch.no_grad()
def get_fusion_weights(model, rgb_pil, thm_pil, device, amp_dtype):
    """返回 list of 4 floats: [w_rgb_f1, w_rgb_f2, w_rgb_f3, w_rgb_f4]
    以及对应的 w_aux。"""
    sz = INFER_SIZE
    rgb_r = rgb_pil.resize((sz, sz), Image.BILINEAR)
    thm_r = thm_pil.resize((sz, sz), Image.BILINEAR)
    rgb_t = TF.normalize(TF.to_tensor(rgb_r), RGB_MEAN, RGB_STD).unsqueeze(0).to(device)
    thm_t = TF.normalize(TF.to_tensor(thm_r), THM_MEAN, THM_STD).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=amp_dtype):
        out = model(rgb_t, thm_t)
    # fusion_weights: list of 4 [1, 2] tensors
    w_rgb = [fw[0, 0].item() for fw in out["fusion_weights"]]
    w_aux = [fw[0, 1].item() for fw in out["fusion_weights"]]
    return w_rgb, w_aux


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_weight_distribution(all_w_rgb, all_w_aux, out_dir):
    """
    all_w_rgb: (N, 4) array — N 张图，4 个 level 的 w_rgb
    all_w_aux: (N, 4) array
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors_rgb = "#5c85d6"
    colors_aux = "#e06c3e"

    for ax, data, color, title in [
        (axes[0], all_w_rgb, colors_rgb, "w_RGB  (SAM2 Hiera)"),
        (axes[1], all_w_aux, colors_aux, "w_Thermal  (ConvNeXt-Tiny)"),
    ]:
        bp = ax.boxplot(
            [data[:, i] for i in range(4)],
            labels=LEVEL_NAMES,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # 均值标注
        for i in range(4):
            mean_val = data[:, i].mean()
            ax.text(i + 1, mean_val + 0.005, f"μ={mean_val:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color="black")

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="0.5 (equal weight)")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Weight", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8.5)

    fig.suptitle("CACAF Adaptive Fusion Weight Distribution (val set, per scale level)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = osp.join(out_dir, "fusion_weights_dist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_weight_scatter(all_w_rgb, out_dir):
    """
    每个 level 一个子图：x 轴=图片序号，y 轴=w_rgb，彩点连线显示每张图的走势。
    """
    N = all_w_rgb.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    axes = axes.flatten()
    colors = ["#5c85d6", "#e06c3e", "#2a7a2a", "#aa2222"]

    for i in range(4):
        ax = axes[i]
        ax.scatter(range(N), all_w_rgb[:, i], s=8, alpha=0.5,
                   color=colors[i], label=f"w_rgb")
        # 滑动平均
        kernel = 10
        if N > kernel:
            smoothed = np.convolve(all_w_rgb[:, i],
                                   np.ones(kernel) / kernel, mode="valid")
            ax.plot(range(kernel - 1, N), smoothed, color="black",
                    linewidth=1.2, alpha=0.7, label=f"MA({kernel})")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_title(f"{LEVEL_NAMES[i]}  μ={all_w_rgb[:, i].mean():.3f}", fontsize=10)
        ax.set_ylabel("w_RGB", fontsize=9)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    axes[2].set_xlabel("Image index (val set)", fontsize=10)
    axes[3].set_xlabel("Image index (val set)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("CACAF w_RGB per image across scale levels  (w_Thermal = 1 − w_RGB)",
                 fontsize=12)
    plt.tight_layout()
    path = osp.join(out_dir, "fusion_weights_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_weight_mean_bar(all_w_rgb, out_dir):
    """均值柱状图（最简洁，适合放入论文）。"""
    means_rgb = all_w_rgb.mean(axis=0)
    means_aux = 1.0 - means_rgb
    stds_rgb  = all_w_rgb.std(axis=0)

    x = np.arange(4)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_r = ax.bar(x - width / 2, means_rgb, width, yerr=stds_rgb,
                    label="w_RGB (SAM2 Hiera)", color="#5c85d6",
                    alpha=0.85, capsize=4, error_kw=dict(linewidth=1.2))
    bars_a = ax.bar(x + width / 2, means_aux, width, yerr=stds_rgb,
                    label="w_Thermal (ConvNeXt)", color="#e06c3e",
                    alpha=0.85, capsize=4, error_kw=dict(linewidth=1.2))

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_NAMES, fontsize=9)
    ax.set_ylabel("Mean Adaptive Weight", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.set_title("CACAF Mean Fusion Weights per Scale Level  (val set, mean ± std)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # 数值标注
    for i in range(4):
        ax.text(x[i] - width / 2, means_rgb[i] + stds_rgb[i] + 0.01,
                f"{means_rgb[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, means_aux[i] + stds_rgb[i] + 0.01,
                f"{means_aux[i]:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = osp.join(out_dir, "fusion_weights_mean_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_easy_hard_thermal_bar(all_w_aux, subsets, out_dir):
    """Paper-ready Figure 4a: mean thermal weight under easy vs hard scenes."""
    subset_arr = np.array(subsets)
    easy_mask = subset_arr == "easy"
    hard_mask = subset_arr == "hard"

    easy_mean = all_w_aux[easy_mask].mean(axis=0)
    hard_mean = all_w_aux[hard_mask].mean(axis=0)
    easy_std = all_w_aux[easy_mask].std(axis=0)
    hard_std = all_w_aux[hard_mask].std(axis=0)

    x = np.arange(4)
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    easy_color = "#8fb6ff"
    hard_color = "#e07a4f"

    ax.bar(
        x - width / 2,
        easy_mean,
        width,
        yerr=easy_std,
        color=easy_color,
        edgecolor="#4c72b0",
        linewidth=1.1,
        capsize=4,
        label="Easy scenes",
        alpha=0.95,
    )
    ax.bar(
        x + width / 2,
        hard_mean,
        width,
        yerr=hard_std,
        color=hard_color,
        edgecolor="#b95d34",
        linewidth=1.1,
        capsize=4,
        label="Hard scenes",
        alpha=0.95,
    )

    for i in range(4):
        ax.text(x[i] - width / 2, easy_mean[i] + easy_std[i] + 0.012,
                f"{easy_mean[i]:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(x[i] + width / 2, hard_mean[i] + hard_std[i] + 0.012,
                f"{hard_mean[i]:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["f1", "f2", "f3", "f4"], fontsize=10)
    ax.set_ylabel("Thermal Adaptive Weight", fontsize=11)
    ax.set_xlabel("Fusion scale", fontsize=11)
    ax.set_ylim(0.0, max(hard_mean.max() + hard_std.max() + 0.09, 0.38))
    ax.set_title("CACAF thermal weight under easy and hard FMB scenes", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    # Brief callout for the strongest effect
    delta_f2 = hard_mean[1] - easy_mean[1]
    ax.text(
        1.5,
        ax.get_ylim()[1] * 0.92,
        f"Hard scenes increase thermal reliance at f2 by {delta_f2:+.3f}",
        fontsize=9.5,
        color="#444444",
        ha="center",
    )

    plt.tight_layout()
    path = osp.join(out_dir, "figure4a_easy_hard_thermal.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="work_dirs/cacaf_fmb_v3/best.pth")
    p.add_argument("--data-root",  default="/home/jl/dataset/FMB")
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--out-dir",    default="work_dirs/visualizations/plots")
    p.add_argument(
        "--highlight-samples",
        default="00043,00381,01046,01212",
        help="Comma-separated sample IDs for paper-style per-sample weight plot",
    )
    p.add_argument("--sam2-cfg",   default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt",  default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--bf16",       action="store_true", default=True)
    p.add_argument("--no-bf16",    dest="bf16", action="store_false")
    return p.parse_args()


def plot_sample_weight_profiles(all_w_rgb, all_w_aux, names, out_dir, highlight_ids):
    """Paper-ready Figure 4b: weight profiles of representative samples."""
    selected = []
    for sid in highlight_ids:
        sid = sid.strip()
        if not sid:
            continue
        matches = [i for i, name in enumerate(names) if name.endswith(f"_{sid}")]
        if matches:
            selected.append((sid, matches[0]))

    if not selected:
        print("  跳过 figure4b_sample_profiles.png：未找到指定样本")
        return

    fig, axes = plt.subplots(1, len(selected), figsize=(3.2 * len(selected), 3.8), sharey=True)
    if len(selected) == 1:
        axes = [axes]

    x = np.arange(4)
    width = 0.34
    rgb_color = "#5c85d6"
    aux_color = "#e07a4f"

    for ax, (sid, idx) in zip(axes, selected):
        rgb_vals = all_w_rgb[idx]
        aux_vals = all_w_aux[idx]
        ax.bar(x - width / 2, rgb_vals, width, color=rgb_color, alpha=0.92, label="RGB")
        ax.bar(x + width / 2, aux_vals, width, color=aux_color, alpha=0.92, label="Thermal")

        for j in range(4):
            ax.text(x[j] - width / 2, rgb_vals[j] + 0.015, f"{rgb_vals[j]:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
            ax.text(x[j] + width / 2, aux_vals[j] + 0.015, f"{aux_vals[j]:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_title(sid, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["f1", "f2", "f3", "f4"], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Adaptive weight", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Scale", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("CACAF weight profiles on representative FMB test samples", fontsize=12)
    plt.tight_layout()
    path = osp.join(out_dir, "figure4b_sample_profiles.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

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

    all_w_rgb = []  # (N, 4)
    all_w_aux = []
    names     = []
    subsets   = []

    for idx, s in enumerate(samples):
        rgb_pil = Image.open(s["rgb"]).convert("RGB")
        thm_pil = Image.open(s["thm"]).convert("RGB")
        w_rgb, w_aux = get_fusion_weights(model, rgb_pil, thm_pil, device, amp_dtype)
        all_w_rgb.append(w_rgb)
        all_w_aux.append(w_aux)
        names.append(f"{s['subset']}_{s['name']}")
        subsets.append(s["subset"])
        if (idx + 1) % 40 == 0:
            print(f"  [{idx+1}/{len(samples)}]")

    all_w_rgb = np.array(all_w_rgb)  # (N, 4)
    all_w_aux = np.array(all_w_aux)

    # CSV
    csv_path = osp.join(args.out_dir, "fusion_weights_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "w_rgb_f1", "w_rgb_f2", "w_rgb_f3", "w_rgb_f4",
                    "w_aux_f1", "w_aux_f2", "w_aux_f3", "w_aux_f4"])
        for i, name in enumerate(names):
            w.writerow([name] +
                       [f"{v:.4f}" for v in all_w_rgb[i]] +
                       [f"{v:.4f}" for v in all_w_aux[i]])
    print(f"  CSV 已保存: {csv_path}")

    # 统计摘要
    print("\n--- CACAF 融合权重均值 ---")
    for i in range(4):
        print(f"  {LEVEL_NAMES[i]:20s}  "
              f"w_rgb={all_w_rgb[:, i].mean():.4f}±{all_w_rgb[:, i].std():.4f}  "
              f"w_aux={all_w_aux[:, i].mean():.4f}±{all_w_aux[:, i].std():.4f}")

    # 绘图
    print("\n生成图表...")
    plot_weight_distribution(all_w_rgb, all_w_aux, args.out_dir)
    plot_weight_mean_bar(all_w_rgb, args.out_dir)
    plot_weight_scatter(all_w_rgb, args.out_dir)
    plot_easy_hard_thermal_bar(all_w_aux, subsets, args.out_dir)
    plot_sample_weight_profiles(
        all_w_rgb,
        all_w_aux,
        names,
        args.out_dir,
        args.highlight_samples.split(","),
    )

    print(f"\n完成！结果保存至: {args.out_dir}")


if __name__ == "__main__":
    main()
