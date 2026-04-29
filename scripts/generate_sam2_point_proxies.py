#!/usr/bin/env python3
"""Generate SAM2 point-to-mask proxy labels from sparse FMB point prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SAM2-guided point proxy masks")
    p.add_argument("--data-root", default="/root/autodl-tmp/datasets/FMB")
    p.add_argument("--point-index", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--box-radius", type=int, default=128)
    p.add_argument("--min-mask-pixels", type=int, default=16)
    p.add_argument("--max-mask-area-ratio", type=float, default=0.35)
    p.add_argument("--multimask", action="store_true", default=True)
    p.add_argument("--no-multimask", dest="multimask", action="store_false")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _load_points(root: Path, rel_npz: str) -> dict:
    arr = np.load(root / rel_npz)
    return {
        "x": np.array(arr["x"], dtype=np.float32),
        "y": np.array(arr["y"], dtype=np.float32),
        "cls": np.array(arr["cls"], dtype=np.int64),
        "h": int(np.array(arr["h"])[0]),
        "w": int(np.array(arr["w"])[0]),
    }


def _make_box(x: float, y: float, w: int, h: int, radius: int) -> np.ndarray | None:
    if radius <= 0:
        return None
    return np.array(
        [
            max(0.0, x - radius),
            max(0.0, y - radius),
            min(float(w - 1), x + radius),
            min(float(h - 1), y + radius),
        ],
        dtype=np.float32,
    )


def _select_mask(masks: np.ndarray, scores: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = masks.shape[-2:]
    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))
    contains = np.array([bool(m[y, x]) for m in masks], dtype=np.bool_)
    if contains.any():
        candidates = np.where(contains)[0]
        best = candidates[np.argmax(scores[candidates])]
    else:
        best = int(np.argmax(scores))
    return masks[best].astype(np.bool_)


def _generate_one(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    points: dict,
    box_radius: int,
    min_mask_pixels: int,
    max_mask_area_ratio: float,
    multimask: bool,
) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    target = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
    candidates = []

    predictor.set_image(image)
    max_area = max_mask_area_ratio * float(h * w)

    for x_f, y_f, cls_id in zip(points["x"], points["y"], points["cls"]):
        x = int(round(float(x_f)))
        y = int(round(float(y_f)))
        box = _make_box(float(x_f), float(y_f), w, h, box_radius)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x_f, y_f]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=box,
            multimask_output=multimask,
            return_logits=False,
            normalize_coords=True,
        )
        mask = _select_mask(masks, scores, x, y)
        area = int(mask.sum())
        if area < min_mask_pixels:
            continue
        if max_mask_area_ratio > 0 and area > max_area:
            continue
        candidates.append((area, int(cls_id), mask))

    # Fill small masks first so rare objects are not overwritten by stuff regions.
    candidates.sort(key=lambda row: row[0])
    for _, cls_id, mask in candidates:
        fill = mask & (target == IGNORE_INDEX)
        target[fill] = np.uint8(cls_id)

    stats = {
        "points": int(points["cls"].shape[0]),
        "kept_masks": int(len(candidates)),
        "proxy_pixels": int((target != IGNORE_INDEX).sum()),
    }
    return target, stats


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    point_index = Path(args.point_index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with point_index.open("r", encoding="utf-8") as f:
        point_payload = json.load(f)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    sam2 = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)

    items = point_payload.get("items", [])
    if args.limit is not None:
        items = items[: args.limit]

    records = []
    total_proxy_pixels = 0
    total_kept_masks = 0
    for idx, item in enumerate(items):
        split = item.get("split", "train")
        subset = item["subset"]
        filename = item["filename"]
        sample_id = item.get("id", f"{subset}/{filename}")
        rgb_path = data_root / split / "Visible" / subset / filename
        if not rgb_path.exists():
            print(f"[warn] missing image: {rgb_path}")
            continue

        image = np.array(Image.open(rgb_path).convert("RGB"))
        points = _load_points(point_index.parent, item["npz"])
        target, stats = _generate_one(
            predictor,
            image,
            points,
            box_radius=args.box_radius,
            min_mask_pixels=args.min_mask_pixels,
            max_mask_area_ratio=args.max_mask_area_ratio,
            multimask=args.multimask,
        )

        rel_dir = Path("proxies") / subset
        abs_dir = out_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        npz_name = f"{Path(filename).stem}.npz"
        np.savez_compressed(
            abs_dir / npz_name,
            target=target,
            h=np.array([target.shape[0]], dtype=np.int32),
            w=np.array([target.shape[1]], dtype=np.int32),
            ignore_index=np.array([IGNORE_INDEX], dtype=np.int32),
        )

        total_proxy_pixels += stats["proxy_pixels"]
        total_kept_masks += stats["kept_masks"]
        records.append(
            {
                "id": sample_id,
                "subset": subset,
                "filename": filename,
                "split": split,
                "orig_size": [int(target.shape[0]), int(target.shape[1])],
                "npz": str((rel_dir / npz_name).as_posix()),
                **stats,
            }
        )

        if (idx + 1) % 25 == 0:
            print(
                f"[progress] {idx + 1}/{len(items)} "
                f"kept_masks={total_kept_masks} proxy_pixels={total_proxy_pixels}"
            )

    payload = {
        "meta": {
            "format": "fmb_sam2_point_proxy_v1",
            "dataset": "FMB",
            "source_point_index": os.path.relpath(point_index, _ROOT),
            "sam2_cfg": args.sam2_cfg,
            "sam2_ckpt": args.sam2_ckpt,
            "box_radius": args.box_radius,
            "min_mask_pixels": args.min_mask_pixels,
            "max_mask_area_ratio": args.max_mask_area_ratio,
            "multimask": args.multimask,
            "num_samples": len(records),
            "total_kept_masks": total_kept_masks,
            "total_proxy_pixels": total_proxy_pixels,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
        "items": records,
    }
    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {out_dir / 'index.json'}")
    print(
        f"[summary] samples={len(records)} kept_masks={total_kept_masks} "
        f"proxy_pixels={total_proxy_pixels}"
    )


if __name__ == "__main__":
    main()
