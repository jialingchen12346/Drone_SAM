#!/usr/bin/env python3
"""
Generate sparse point labels from FMB dense masks based on split protocol.

Example:
  python scripts/generate_point_labels.py \
    --data-root /home/jl/dataset/FMB \
    --protocol-file research_workspace/artifacts/label_protocol/fmb_label_splits_seed42.json \
    --ratio 10 \
    --pool unlabeled \
    --points-per-class 1 \
    --seed 42 \
    --out-dir research_workspace/artifacts/weak_labels/fmb_points_seed42_r10_u
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate FMB point-label supervision")
    p.add_argument("--data-root", default="/home/jl/dataset/FMB")
    p.add_argument(
        "--protocol-file",
        default="research_workspace/artifacts/label_protocol/fmb_label_splits_seed42.json",
    )
    p.add_argument("--ratio", type=int, default=10)
    p.add_argument("--pool", choices=["labeled", "unlabeled", "all"], default="unlabeled")
    p.add_argument("--points-per-class", type=int, default=1)
    p.add_argument(
        "--class-points",
        default="",
        help=(
            "Optional per-class point counts using raw FMB label ids 1..14, "
            "e.g. '4:3,10:3,11:3,12:3,13:3,14:3'. "
            "Unspecified classes use --points-per-class."
        ),
    )
    p.add_argument("--min-class-pixels", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument(
        "--out-dir",
        default="research_workspace/artifacts/weak_labels/fmb_points_seed42_r10_u",
    )
    p.add_argument(
        "--index-file",
        default=None,
        help="Default: <out-dir>/index.json",
    )
    return p.parse_args()


def _load_protocol(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_items(protocol: dict, ratio: int, pool: str) -> list[dict]:
    ratio_key = str(ratio)
    table = protocol.get("protocol", {})
    if ratio_key not in table:
        raise KeyError(f"ratio '{ratio}' not found in protocol")

    row = table[ratio_key]
    if pool == "all":
        items = list(row.get("labeled", [])) + list(row.get("unlabeled", []))
    else:
        items = list(row.get(pool, []))

    if not items:
        raise RuntimeError(f"No samples found for ratio={ratio}, pool={pool}")
    return items


def _sample_points_for_mask(
    label: np.ndarray,
    points_per_class: int,
    class_points: dict[int, int],
    min_class_pixels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys_all = []
    xs_all = []
    cls_all = []

    for cls_raw in np.unique(label):
        # FMB dense labels are 1..14 (0: background)
        if cls_raw <= 0:
            continue
        coords = np.argwhere(label == cls_raw)
        if coords.shape[0] < min_class_pixels:
            continue

        cls_id = int(cls_raw)
        k = min(class_points.get(cls_id, points_per_class), coords.shape[0])
        if k <= 0:
            continue
        pick = rng.choice(coords.shape[0], size=k, replace=False)
        pts = coords[pick]
        ys = pts[:, 0].astype(np.int32)
        xs = pts[:, 1].astype(np.int32)
        cls = np.full((k,), int(cls_raw - 1), dtype=np.int32)  # -> 0..13

        ys_all.append(ys)
        xs_all.append(xs)
        cls_all.append(cls)

    if not ys_all:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.concatenate(ys_all, axis=0),
        np.concatenate(xs_all, axis=0),
        np.concatenate(cls_all, axis=0),
    )


def _parse_class_points(spec: str) -> dict[int, int]:
    if not spec.strip():
        return {}
    out: dict[int, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --class-points item '{item}', expected '<raw_label_id>:<count>'")
        key, value = item.split(":", 1)
        cls_id = int(key)
        count = int(value)
        if cls_id < 1:
            raise ValueError(f"FMB raw label ids are 1..14, got {cls_id}")
        if count < 0:
            raise ValueError(f"Point count must be non-negative, got {count}")
        out[cls_id] = count
    return out


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    protocol_file = Path(args.protocol_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_file = Path(args.index_file) if args.index_file else out_dir / "index.json"

    protocol = _load_protocol(protocol_file)
    items = _select_items(protocol, ratio=args.ratio, pool=args.pool)
    class_points = _parse_class_points(args.class_points)

    rng = np.random.default_rng(args.seed)

    records = []
    total_points = 0
    missing_labels = 0

    for item in items:
        split = item.get("split", "train")
        subset = item["subset"]
        fname = item["filename"]
        sample_id = item.get("id", f"{subset}/{fname}")

        label_path = data_root / split / "Label" / fname
        if not label_path.exists():
            missing_labels += 1
            continue

        lbl = np.array(Image.open(label_path), dtype=np.int32)
        h, w = lbl.shape[:2]

        ys, xs, cls = _sample_points_for_mask(
            lbl,
            points_per_class=args.points_per_class,
            class_points=class_points,
            min_class_pixels=args.min_class_pixels,
            rng=rng,
        )

        rel_dir = Path("points") / subset
        abs_dir = out_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)

        npz_name = f"{Path(fname).stem}.npz"
        npz_path = abs_dir / npz_name
        np.savez_compressed(
            npz_path,
            y=ys,
            x=xs,
            cls=cls,
            h=np.array([h], dtype=np.int32),
            w=np.array([w], dtype=np.int32),
            ignore_index=np.array([args.ignore_index], dtype=np.int32),
        )

        n_points = int(ys.shape[0])
        total_points += n_points
        records.append(
            {
                "id": sample_id,
                "subset": subset,
                "filename": fname,
                "split": split,
                "orig_size": [h, w],
                "n_points": n_points,
                "npz": str((rel_dir / npz_name).as_posix()),
            }
        )

    payload = {
        "meta": {
            "dataset": "FMB",
            "ratio": args.ratio,
            "pool": args.pool,
            "points_per_class": args.points_per_class,
            "class_points": class_points,
            "min_class_pixels": args.min_class_pixels,
            "seed": args.seed,
            "ignore_index": args.ignore_index,
            "source_protocol": str(protocol_file.as_posix()),
            "num_samples": len(records),
            "num_input_items": len(items),
            "missing_label_files": missing_labels,
            "total_points": total_points,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "format": "fmb_point_labels_v1",
        },
        "items": records,
    }

    with index_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote point label index: {index_file}")
    print(
        "[summary] "
        f"samples={len(records)} / {len(items)}  "
        f"total_points={total_points}  "
        f"avg_points_per_sample={total_points / max(len(records), 1):.2f}"
    )
    if missing_labels > 0:
        print(f"[warn] missing label files: {missing_labels}")


if __name__ == "__main__":
    main()
