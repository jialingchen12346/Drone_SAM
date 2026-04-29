#!/usr/bin/env python3
"""Build external unlabeled RGB/aux manifests for label-efficient training."""

from __future__ import annotations

import argparse
import json
import os.path as osp


def read_ids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_mfnet_rgba(root: str, splits: list[str], include_missing: bool = False) -> list[dict]:
    blacklist_path = osp.join(root, "black_list.txt")
    blacklist = set(read_ids(blacklist_path)) if osp.exists(blacklist_path) else set()

    samples = []
    seen = set()
    for split in splits:
        split_path = osp.join(root, f"{split}.txt")
        if not osp.exists(split_path):
            raise FileNotFoundError(split_path)
        for sample_id in read_ids(split_path):
            if sample_id in seen or sample_id in blacklist:
                continue
            img_rel = f"images/{sample_id}.png"
            img_path = osp.join(root, img_rel)
            if not osp.exists(img_path):
                if include_missing:
                    samples.append({"id": f"mfnet/{sample_id}", "rgba": img_rel, "missing": True})
                continue
            samples.append({"id": f"mfnet/{sample_id}", "rgba": img_rel})
            seen.add(sample_id)
    return samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mfnet_rgba", choices=["mfnet_rgba"])
    p.add_argument("--root", required=True)
    p.add_argument("--splits", default="train,val", help="Comma-separated split names without .txt")
    p.add_argument("--out", required=True)
    p.add_argument("--include-missing", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if args.dataset == "mfnet_rgba":
        samples = build_mfnet_rgba(args.root, splits, include_missing=args.include_missing)
    else:
        raise ValueError(args.dataset)

    payload = {
        "type": args.dataset,
        "root": args.root,
        "splits": splits,
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[external-index] samples={len(samples)} -> {args.out}")


if __name__ == "__main__":
    main()
