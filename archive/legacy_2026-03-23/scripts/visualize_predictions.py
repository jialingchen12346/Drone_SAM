#!/usr/bin/env python3
"""
B1 可视化脚本 — RGB / Thermal / GT / Pred 四列对比图

用法 (从项目根目录运行):
    conda activate sam2-unet
    python scripts/visualize_predictions.py \
        --checkpoint work_dirs/cacaf_fmb_v3/best.pth \
        --data-root /home/jl/dataset/FMB \
        --split val \
        --out-dir work_dirs/visualizations/v3_val \
        --num-images 20 \
        --full-res

参数说明:
    --checkpoint   模型检查点路径
    --data-root    FMB 数据集根目录
    --split        val / test
    --out-dir      输出目录
    --num-images   生成多少张图（默认 all，-1 表示全部）
    --full-res     使用全分辨率 800x600（不加则 resize 到 512x512）
    --mosaic       额外生成 N 张图的拼接总览图（默认 16 张）
    --mosaic-n     mosaic 中包含的图片数量
"""

import argparse
import os
import os.path as osp
import sys

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

# ---------------------------------------------------------------------------
# 14 类颜色表（BGR 风格，但我们用 RGB）
# 参考 Cityscapes / ADE 风格，尽量区分相邻类
# ---------------------------------------------------------------------------
PALETTE = np.array([
    [128, 64, 128],   # 0  Road          — 紫灰
    [244, 35, 232],   # 1  Sidewalk      — 品红
    [ 70, 70, 70],    # 2  Building      — 深灰
    [250, 170,  30],  # 3  Traffic Light — 橙黄
    [220, 220,   0],  # 4  Traffic Sign  — 黄
    [107, 142,  35],  # 5  Vegetation    — 草绿
    [ 70, 130, 180],  # 6  Sky           — 钢蓝
    [220,  20,  60],  # 7  Person        — 深红
    [  0,  60, 100],  # 8  Car           — 深蓝
    [  0,  80, 100],  # 9  Truck         — 青蓝
    [  0,   0, 142],  # 10 Bus           — 蓝
    [  0,   0, 230],  # 11 Motorcycle    — 亮蓝
    [119,  11,  32],  # 12 Bicycle       — 暗红
    [153, 153, 153],  # 13 Pole          — 中灰
], dtype=np.uint8)

IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)  # black for ignore

# ImageNet normalization
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
THM_MEAN = [0.5, 0.5, 0.5]
THM_STD  = [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def label_to_color(label_np: np.ndarray) -> np.ndarray:
    """(H, W) int → (H, W, 3) uint8 RGB."""
    h, w = label_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        mask = label_np == c
        color[mask] = PALETTE[c]
    color[label_np == IGNORE_INDEX] = IGNORE_COLOR
    return color


def denorm_rgb(tensor: torch.Tensor) -> np.ndarray:
    """[3, H, W] normalized tensor → (H, W, 3) uint8 RGB."""
    mean = torch.tensor(RGB_MEAN).view(3, 1, 1)
    std  = torch.tensor(RGB_STD).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def denorm_thm(tensor: torch.Tensor) -> np.ndarray:
    """[3, H, W] normalized thermal tensor → (H, W, 3) uint8 grayscale-ish."""
    mean = torch.tensor(THM_MEAN).view(3, 1, 1)
    std  = torch.tensor(THM_STD).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def add_text(img_pil: Image.Image, text: str,
             pos=(4, 2), color=(255, 255, 255), bg=(0, 0, 0)) -> Image.Image:
    """在图片左上角叠加文字标签。"""
    draw = ImageDraw.Draw(img_pil)
    # 画黑色背景矩形（增加可读性）
    try:
        bbox = draw.textbbox(pos, text)
        draw.rectangle(bbox, fill=bg)
    except AttributeError:
        pass  # 旧版 Pillow 无 textbbox
    draw.text(pos, text, fill=color)
    return img_pil


def make_legend_strip(width: int, cell_h: int = 20) -> Image.Image:
    """生成 14 类颜色图例条（横排）。"""
    n = NUM_CLASSES
    cell_w = max(width // n, 40)
    total_w = cell_w * n
    legend = Image.new("RGB", (total_w, cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(legend)
    for i, cls_name in enumerate(CLASSES):
        x = i * cell_w
        draw.rectangle([x, 0, x + cell_w - 1, cell_h - 1],
                       fill=tuple(PALETTE[i]))
        # 短名缩写
        short = cls_name[:4]
        draw.text((x + 2, 2), short, fill=(255, 255, 255))
    # 缩放到目标宽度
    if total_w != width:
        legend = legend.resize((width, cell_h), Image.NEAREST)
    return legend


def load_sample_list(data_root: str, split: str):
    """返回 list of dict: {rgb, thm, lbl, name}."""
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


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

# SAM2 Hiera 的位置编码 window_embed 大小为 12×12=144，
# 只有 512×512（feature map 128×128）经过实测可用。
# 800×600 → 150 tokens 高度会触发 RuntimeError（150 vs 144 不匹配）。
# 因此推理始终 resize 到 512×512（与训练完全一致），显示时再 upsample 回原尺寸。
INFER_SIZE = 512


@torch.no_grad()
def run_inference(model, rgb_pil, thm_pil, device):
    """给定 PIL 图像，返回预测的 (512, 512) int 数组。"""
    sz = INFER_SIZE
    rgb_r = rgb_pil.resize((sz, sz), Image.BILINEAR)
    thm_r = thm_pil.resize((sz, sz), Image.BILINEAR)

    rgb_t = TF.normalize(TF.to_tensor(rgb_r), RGB_MEAN, RGB_STD).unsqueeze(0).to(device)
    thm_t = TF.normalize(TF.to_tensor(thm_r), THM_MEAN, THM_STD).unsqueeze(0).to(device)

    out = model(rgb_t, thm_t)
    pred = out["pred"].squeeze(0).argmax(0).cpu().numpy().astype(np.int32)
    return pred  # (512, 512)


# ---------------------------------------------------------------------------
# 生成单张对比图
# ---------------------------------------------------------------------------

def make_comparison_image(
    rgb_pil, thm_pil, gt_pil, pred_np,
    name: str, subset: str, target_h: int = 400
) -> Image.Image:
    """
    生成四列横排对比图:
        RGB | Thermal | Ground Truth | Prediction
    图片统一缩放到 target_h 高度。
    """
    orig_w, orig_h = rgb_pil.size
    scale = target_h / orig_h
    disp_w = int(orig_w * scale)
    disp_h = target_h

    def resize_pil(img, mode=Image.BILINEAR):
        return img.resize((disp_w, disp_h), mode)

    # GT mask → 颜色图
    gt_np = np.array(gt_pil, dtype=np.int64)
    gt_np -= 1
    gt_np[gt_np < 0] = IGNORE_INDEX
    gt_color = Image.fromarray(label_to_color(gt_np))

    pred_color = Image.fromarray(label_to_color(pred_np))

    panels = [
        (resize_pil(rgb_pil),   "RGB"),
        (resize_pil(thm_pil),   "Thermal"),
        (resize_pil(gt_color, Image.NEAREST),   "Ground Truth"),
        (resize_pil(pred_color, Image.NEAREST), "Prediction"),
    ]

    # 标注列标题
    labeled = []
    for img, title in panels:
        img = img.copy()
        add_text(img, title, pos=(4, 2), color=(255, 255, 255), bg=(0, 0, 0))
        labeled.append(img)

    # 图例条
    total_w = disp_w * 4
    legend = make_legend_strip(total_w, cell_h=22)

    # 合并
    header_h = 20
    canvas_h = header_h + disp_h + legend.height
    canvas = Image.new("RGB", (total_w, canvas_h), (20, 20, 20))

    # 顶部文字：文件名 + subset
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), f"{name}  [{subset}]", fill=(200, 200, 200))

    # 拼贴四列
    for i, img in enumerate(labeled):
        canvas.paste(img, (i * disp_w, header_h))

    # 图例
    canvas.paste(legend, (0, header_h + disp_h))

    return canvas


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="B1 Visualization: RGB/Thermal/GT/Pred comparison")
    p.add_argument("--checkpoint", default="work_dirs/cacaf_fmb_v3/best.pth")
    p.add_argument("--data-root",  default="/home/jl/dataset/FMB")
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--out-dir",    default="work_dirs/visualizations/v3_val")
    p.add_argument("--num-images", type=int, default=-1,
                   help="最多生成多少张（-1 表示全部）")
    p.add_argument("--full-res",   action="store_true",
                   help="以原图分辨率显示（推理始终用 512x512，预测 upsample 到原图尺寸展示）")
    p.add_argument("--target-h",   type=int, default=300,
                   help="对比图中每列图片的显示高度（像素）")
    p.add_argument("--mosaic",     action="store_true",
                   help="额外生成 mosaic 总览图")
    p.add_argument("--mosaic-n",   type=int, default=16,
                   help="mosaic 中包含的图片数")
    p.add_argument("--sam2-cfg",   default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt",  default="checkpoints/sam2_hiera_large.pt")
    p.add_argument("--bf16",       action="store_true", default=True)
    p.add_argument("--no-bf16",    dest="bf16", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------
    print("加载模型...")
    model = CACafSegmentor(
        sam2_checkpoint=args.sam2_ckpt,
        sam2_config=args.sam2_cfg,
        num_classes=NUM_CLASSES,
        use_dice=True,  # 与 v3 训练配置一致（loss 结构不影响推理）
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  已加载: {args.checkpoint}  (epoch={ckpt.get('epoch', '?')}, "
          f"best_mIoU={ckpt.get('best_miou', '?'):.2f})")

    # AMP dtype
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # ------------------------------------------------------------------
    # 推理分辨率说明
    # ------------------------------------------------------------------
    # SAM2 只兼容 512×512 输入（位置编码约束），推理固定此尺寸。
    # --full-res 控制对比图中原图/GT 是否以原始分辨率显示（而非 512×512）。
    print(f"  推理分辨率: 512x512 (SAM2 固定)  显示模式: {'原图分辨率' if args.full_res else '512x512'}")

    # ------------------------------------------------------------------
    # 样本列表
    # ------------------------------------------------------------------
    samples = load_sample_list(args.data_root, args.split)
    if args.num_images > 0:
        samples = samples[:args.num_images]
    print(f"  共 {len(samples)} 张图片")

    # ------------------------------------------------------------------
    # 逐张推理 + 保存
    # ------------------------------------------------------------------
    mosaic_frames = []

    for idx, s in enumerate(samples):
        rgb_pil = Image.open(s["rgb"]).convert("RGB")
        thm_pil = Image.open(s["thm"]).convert("RGB")
        gt_pil  = Image.open(s["lbl"])  # mode 'L', values 1-14

        with torch.autocast("cuda", dtype=amp_dtype):
            pred_np = run_inference(model, rgb_pil, thm_pil, device)

        # 把 pred 缩放回原图尺寸（显示用）
        orig_w, orig_h = rgb_pil.size
        if pred_np.shape != (orig_h, orig_w):
            pred_pil_tmp = Image.fromarray(pred_np.astype(np.uint8)).resize(
                (orig_w, orig_h), Image.NEAREST
            )
            pred_np_disp = np.array(pred_pil_tmp).astype(np.int32)
        else:
            pred_np_disp = pred_np

        comp = make_comparison_image(
            rgb_pil, thm_pil, gt_pil, pred_np_disp,
            name=s["name"], subset=s["subset"],
            target_h=args.target_h,
        )

        save_path = osp.join(args.out_dir, f"{s['subset']}_{s['name']}.png")
        comp.save(save_path)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(f"  [{idx+1}/{len(samples)}] 已保存: {save_path}")

        if args.mosaic and len(mosaic_frames) < args.mosaic_n:
            mosaic_frames.append(comp)

    # ------------------------------------------------------------------
    # Mosaic 总览图
    # ------------------------------------------------------------------
    if args.mosaic and mosaic_frames:
        cols = 4
        rows = (len(mosaic_frames) + cols - 1) // cols
        cell_w, cell_h = mosaic_frames[0].size
        mosaic_w = cell_w * cols
        mosaic_h = cell_h * rows
        mosaic_canvas = Image.new("RGB", (mosaic_w, mosaic_h), (10, 10, 10))
        for i, frame in enumerate(mosaic_frames):
            r, c = divmod(i, cols)
            mosaic_canvas.paste(frame, (c * cell_w, r * cell_h))
        mosaic_path = osp.join(args.out_dir, "mosaic_512infer.png")
        mosaic_canvas.save(mosaic_path)
        print(f"\n  Mosaic 已保存: {mosaic_path}")

    print(f"\n完成！全部结果保存至: {args.out_dir}")


if __name__ == "__main__":
    main()
