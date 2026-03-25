#!/usr/bin/env python3
"""
Collect experiment metrics from eval JSON files and eval logs, then write a flat
summary table for quick comparison across sessions.

Examples:
    python scripts/summarize_experiments.py
    python scripts/summarize_experiments.py --root work_dirs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    REPO_ROOT / "work_dirs",
    REPO_ROOT / "segmentation" / "work_dirs",
]
OUT_DIR = REPO_ROOT / "research_workspace" / "experiments" / "summary"
CSV_PATH = OUT_DIR / "experiment_summary.csv"
MD_PATH = OUT_DIR / "experiment_summary.md"

JSON_METRIC_KEYS = ("mIoU", "mAcc", "aAcc")
LOG_METRIC_RE = re.compile(r"^\s*(mIoU|mAcc|aAcc)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$")
CHECKPOINT_RE = re.compile(r"Loaded checkpoint:\s+(.+?)\s+\(epoch\s+([0-9?]+)\)")
CONFIG_RE = re.compile(r'"config"\s*:\s*"([^"]+)"')


def parse_json_metrics(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metric = payload.get("metric")
    if not isinstance(metric, dict):
        return None
    row = {
        "source_type": "json",
        "run_name": path.parent.name,
        "path": str(path.relative_to(REPO_ROOT)),
        "checkpoint": "",
        "config": str(payload.get("config", "")),
        "epoch": "",
    }
    for key in JSON_METRIC_KEYS:
        value = metric.get(key)
        row[key] = round(float(value) * 100, 2) if isinstance(value, (int, float)) and value <= 1.0 else (
            round(float(value), 2) if isinstance(value, (int, float)) else ""
        )
    return row


def parse_log_metrics(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics = {}
    checkpoint = ""
    epoch = ""
    config = ""
    for line in text.splitlines():
        match = LOG_METRIC_RE.match(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
        ckpt_match = CHECKPOINT_RE.search(line)
        if ckpt_match:
            checkpoint = ckpt_match.group(1).strip()
            epoch = ckpt_match.group(2).strip()
        if not config:
            cfg_match = CONFIG_RE.search(line)
            if cfg_match:
                config = cfg_match.group(1)
    if not metrics:
        return None
    return {
        "source_type": "log",
        "run_name": path.parent.name,
        "path": str(path.relative_to(REPO_ROOT)),
        "checkpoint": checkpoint,
        "config": config,
        "epoch": epoch,
        "mIoU": round(metrics.get("mIoU", 0.0), 2) if "mIoU" in metrics else "",
        "mAcc": round(metrics.get("mAcc", 0.0), 2) if "mAcc" in metrics else "",
        "aAcc": round(metrics.get("aAcc", 0.0), 2) if "aAcc" in metrics else "",
    }


def collect_rows(roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if name.endswith(".json") and "eval" in name:
                row = parse_json_metrics(path)
            elif name.endswith(".log") and "eval" in name:
                row = parse_log_metrics(path)
            else:
                row = None
            if row:
                rows.append(row)
    rows.sort(key=lambda item: (-float(item["mIoU"] or 0.0), item["path"]))
    return rows


def write_csv(rows: list[dict]) -> None:
    fields = ["source_type", "run_name", "mIoU", "mAcc", "aAcc", "epoch", "config", "checkpoint", "path"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict]) -> None:
    lines = [
        "# Experiment Summary",
        "",
        f"Total runs: {len(rows)}",
        "",
        "| Run | Type | mIoU | mAcc | aAcc | Epoch | Path |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['source_type']} | {row['mIoU']} | {row['mAcc']} | {row['aAcc']} | "
            f"{row['epoch']} | `{row['path']}` |"
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Extra root to scan. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = DEFAULT_ROOTS + [Path(item).resolve() for item in args.root]
    rows = collect_rows(roots)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    write_markdown(rows)
    print(f"Wrote {len(rows)} experiment rows")
    print(f"CSV: {CSV_PATH}")
    print(f"MD:  {MD_PATH}")


if __name__ == "__main__":
    main()
