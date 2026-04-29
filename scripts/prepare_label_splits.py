#!/usr/bin/env python3
"""
Prepare deterministic labeled/unlabeled split protocol for FMB train set.

Example:
  python scripts/prepare_label_splits.py \
    --data-root /home/jl/dataset/FMB \
    --seed 42 \
    --ratios 1,2,5,10,20,50 \
    --out-file research_workspace/artifacts/label_protocol/fmb_label_splits_seed42.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare FMB label-ratio protocol")
    p.add_argument("--data-root", default="/home/jl/dataset/FMB")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=str, default="1,2,5,10,20,50",
                   help="Comma-separated percentages")
    p.add_argument(
        "--out-file",
        default="research_workspace/artifacts/label_protocol/fmb_label_splits_seed42.json",
    )
    return p.parse_args()


def read_ids(data_root: Path, split: str) -> list[dict]:
    items: list[dict] = []
    for subset in ("easy", "hard"):
        txt = data_root / f"{split}_{subset}_files.txt"
        if not txt.exists():
            continue
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                fname = line.strip()
                if not fname:
                    continue
                items.append(
                    {
                        "id": f"{subset}/{fname}",
                        "subset": subset,
                        "filename": fname,
                        "split": split,
                    }
                )

    if not items:
        raise RuntimeError(f"No samples found from split list under: {data_root}")

    # Stable base ordering before seeded shuffle.
    items.sort(key=lambda x: (x["subset"], x["filename"]))
    return items


def parse_ratios(raw: str) -> list[int]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(tok)
        if val <= 0 or val >= 100:
            raise ValueError(f"ratio must be in (0,100): {val}")
        vals.append(val)
    if not vals:
        raise ValueError("No valid ratio provided")
    return sorted(set(vals))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ratios = parse_ratios(args.ratios)
    all_items = read_ids(data_root, args.split)

    rng = random.Random(args.seed)
    shuffled = all_items.copy()
    rng.shuffle(shuffled)

    protocol: dict[str, dict] = {}
    total = len(shuffled)

    for ratio in ratios:
        n_labeled = max(1, int(round(total * (ratio / 100.0))))
        n_labeled = min(n_labeled, total - 1) if total > 1 else total

        labeled = shuffled[:n_labeled]
        unlabeled = shuffled[n_labeled:]

        protocol[str(ratio)] = {
            "ratio": ratio,
            "n_labeled": len(labeled),
            "n_unlabeled": len(unlabeled),
            "labeled": labeled,
            "unlabeled": unlabeled,
        }

    payload = {
        "meta": {
            "dataset": "FMB",
            "split": args.split,
            "seed": args.seed,
            "ratios": ratios,
            "total_samples": total,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "format": "v1",
        },
        "protocol": protocol,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote split protocol: {out_file}")
    for r in ratios:
        info = protocol[str(r)]
        print(
            f"  ratio={r:>2d}%  labeled={info['n_labeled']:>4d}  "
            f"unlabeled={info['n_unlabeled']:>4d}"
        )


if __name__ == "__main__":
    main()
