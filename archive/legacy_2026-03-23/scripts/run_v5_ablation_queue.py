#!/usr/bin/env python3
"""Run 6 v5 ablations sequentially and auto-sync final results to docs.

- Waits for V5-Full checkpoint if already running.
- Trains remaining experiments one-by-one.
- Evaluates each with segmentation/eval_fullres.py on test split.
- Auto-updates markdown table between marker tags in docs.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path("/home/jl/Drone-SAM-Adapter")
DATA_ROOT = Path("/home/jl/dataset/FMB")
RESULTS_CSV = REPO / "work_dirs" / "v5_ablation_results.csv"
QUEUE_LOG = REPO / "work_dirs" / "v5_ablation_queue.log"

DOCS = [
    REPO / "docs" / "SESSION_SNAPSHOT.md",
    REPO / "docs" / "EVIDENCE_CHAIN.md",
]
MARKER_BEGIN = "<!-- V5_ABLATION_RESULTS_BEGIN -->"
MARKER_END = "<!-- V5_ABLATION_RESULTS_END -->"

BASE_TRAIN = [
    sys.executable,
    "segmentation/train_cacaf.py",
    "--data-root", str(DATA_ROOT),
    "--batch-size", "4",
    "--epochs", "60",
    "--bf16",
    "--eval-resize-mode", "letterbox",
    "--cat-max-ratio", "0.75",
    "--blur-prob", "0.2",
    "--photo-distort",
]

BASE_EVAL = [
    sys.executable,
    "segmentation/eval_fullres.py",
    "--data-root", str(DATA_ROOT),
    "--split", "test",
    "--bf16",
]


@dataclass
class Exp:
    exp_id: str
    work_dir: str
    train_flags: List[str]
    eval_flags: List[str]
    full_maybe_running: bool = False


EXPS = [
    Exp("V5-Full", "work_dirs/v5_ablation_full", ["--use-dice", "--use-ohem"], [], True),
    Exp("V5-NoCACAF", "work_dirs/v5_ablation_no_cacaf", ["--use-dice", "--use-ohem", "--no-cacaf"], ["--no-cacaf"]),
    Exp("V5-NoSAGU", "work_dirs/v5_ablation_no_sagu", ["--use-dice", "--use-ohem", "--no-sagu"], ["--no-sagu"]),
    Exp("V5-NoDice", "work_dirs/v5_ablation_no_dice", ["--use-ohem"], []),
    Exp("V5-NoOHEM", "work_dirs/v5_ablation_no_ohem", ["--use-dice"], []),
    Exp("V5-NoCACAF-NoSAGU", "work_dirs/v5_ablation_no_cacaf_no_sagu", ["--use-dice", "--use-ohem", "--no-cacaf", "--no-sagu"], ["--no-cacaf", "--no-sagu"]),
]


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    QUEUE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: List[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
        return proc.wait()


def parse_metrics(eval_log: Path) -> Optional[Dict[str, float]]:
    if not eval_log.exists():
        return None
    txt = eval_log.read_text(encoding="utf-8", errors="ignore")
    m1 = re.search(r"mIoU\s*=\s*([0-9]+\.?[0-9]*)", txt)
    m2 = re.search(r"mAcc\s*=\s*([0-9]+\.?[0-9]*)", txt)
    m3 = re.search(r"aAcc\s*=\s*([0-9]+\.?[0-9]*)", txt)
    if not (m1 and m2 and m3):
        return None
    return {
        "mIoU": float(m1.group(1)),
        "mAcc": float(m2.group(1)),
        "aAcc": float(m3.group(1)),
    }


def load_results() -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows[row["exp_id"]] = row
    return rows


def save_results(rows: Dict[str, Dict[str, str]]):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exp_id", "checkpoint", "mIoU", "mAcc", "aAcc", "status"])
        w.writeheader()
        for exp in EXPS:
            row = rows.get(exp.exp_id, {
                "exp_id": exp.exp_id,
                "checkpoint": "待执行",
                "mIoU": "-",
                "mAcc": "-",
                "aAcc": "-",
                "status": "pending",
            })
            w.writerow(row)


def render_table(rows: Dict[str, Dict[str, str]]) -> str:
    lines = [
        "| 实验ID | Checkpoint | mIoU | mAcc | aAcc |",
        "|---|---|---:|---:|---:|",
    ]
    for exp in EXPS:
        row = rows.get(exp.exp_id)
        if not row:
            ckpt, miou, macc, aacc = "待执行", "-", "-", "-"
        else:
            ckpt = row.get("checkpoint", "待执行")
            miou = row.get("mIoU", "-")
            macc = row.get("mAcc", "-")
            aacc = row.get("aAcc", "-")
        lines.append(f"| {exp.exp_id} | {ckpt} | {miou} | {macc} | {aacc} |")
    return "\n".join(lines)


def replace_between(text: str, begin: str, end: str, replacement: str) -> str:
    b = text.find(begin)
    e = text.find(end)
    if b == -1 or e == -1 or e < b:
        return text
    head = text[: b + len(begin)]
    tail = text[e:]
    return head + "\n" + replacement + "\n" + tail


def sync_docs(rows: Dict[str, Dict[str, str]]):
    table = render_table(rows)
    for p in DOCS:
        if not p.exists():
            continue
        old = p.read_text(encoding="utf-8")
        new = replace_between(old, MARKER_BEGIN, MARKER_END, table)
        if new != old:
            p.write_text(new, encoding="utf-8")


def update_status(rows: Dict[str, Dict[str, str]], exp_id: str, status: str, ckpt: str = "", metrics: Optional[Dict[str, float]] = None):
    row = rows.get(exp_id, {"exp_id": exp_id, "checkpoint": "待执行", "mIoU": "-", "mAcc": "-", "aAcc": "-", "status": "pending"})
    row["status"] = status
    if ckpt:
        row["checkpoint"] = ckpt
    if metrics:
        row["mIoU"] = f"{metrics['mIoU']:.2f}"
        row["mAcc"] = f"{metrics['mAcc']:.2f}"
        row["aAcc"] = f"{metrics['aAcc']:.2f}"
    rows[exp_id] = row
    save_results(rows)
    sync_docs(rows)


def wait_for_ckpt(ckpt: Path, timeout_sec: int) -> bool:
    start = time.time()
    while True:
        if ckpt.exists():
            return True
        if time.time() - start > timeout_sec:
            return False
        time.sleep(120)


def is_training_running(work_dir: str) -> bool:
    """Whether a train_cacaf.py process is currently running for this work_dir."""
    pattern = f"segmentation/train_cacaf.py.*--work-dir {work_dir}"
    proc = subprocess.run(
        ["pgrep", "-af", pattern],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def latest_epoch_ckpt(work_dir: Path) -> Optional[Path]:
    """Find latest epoch_*.pth checkpoint in work_dir, if any."""
    cands = []
    for p in work_dir.glob("epoch_*.pth"):
        m = re.match(r"epoch_(\d+)\.pth$", p.name)
        if m:
            cands.append((int(m.group(1)), p))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]


def train_exp(exp: Exp, rows: Dict[str, Dict[str, str]]) -> bool:
    work_dir = REPO / exp.work_dir
    ckpt = work_dir / "epoch_60.pth"
    train_log = work_dir / "train.log"

    if ckpt.exists():
        log(f"{exp.exp_id}: found existing checkpoint {ckpt}, skip training")
        update_status(rows, exp.exp_id, "trained", ckpt="epoch_60.pth")
        return True

    if exp.full_maybe_running:
        if is_training_running(exp.work_dir):
            log(f"{exp.exp_id}: detected active training process, waiting for epoch_60.pth")
            ok = wait_for_ckpt(ckpt, timeout_sec=2 * 3600)
            if ok:
                log(f"{exp.exp_id}: detected checkpoint from running job")
                update_status(rows, exp.exp_id, "trained", ckpt="epoch_60.pth")
                return True
            log(f"{exp.exp_id}: running process did not produce epoch_60 within timeout; will resume/start now")
        else:
            log(f"{exp.exp_id}: no active training process found; start/resume immediately")

    update_status(rows, exp.exp_id, "training", ckpt="训练中")
    cmd = BASE_TRAIN + ["--work-dir", exp.work_dir] + exp.train_flags
    latest = latest_epoch_ckpt(work_dir)
    if latest and not ckpt.exists():
        log(f"{exp.exp_id}: resuming from {latest.name}")
        cmd += ["--resume", str(latest)]
    code = run(cmd, train_log)
    if code != 0:
        log(f"{exp.exp_id}: training failed with exit code {code}")
        update_status(rows, exp.exp_id, "failed", ckpt=f"训练失败(code={code})")
        return False

    if not ckpt.exists():
        log(f"{exp.exp_id}: training ended but epoch_60.pth missing")
        update_status(rows, exp.exp_id, "failed", ckpt="缺少epoch_60.pth")
        return False

    update_status(rows, exp.exp_id, "trained", ckpt="epoch_60.pth")
    return True


def eval_exp(exp: Exp, rows: Dict[str, Dict[str, str]]) -> bool:
    work_dir = REPO / exp.work_dir
    ckpt = work_dir / "epoch_60.pth"
    eval_log = work_dir / "eval_test_epoch60.log"

    if not ckpt.exists():
        update_status(rows, exp.exp_id, "failed", ckpt="评估失败:无epoch_60")
        return False

    update_status(rows, exp.exp_id, "evaluating", ckpt="epoch_60.pth")
    cmd = BASE_EVAL + ["--checkpoint", str(ckpt)] + exp.eval_flags
    code = run(cmd, eval_log)
    if code != 0:
        log(f"{exp.exp_id}: eval failed with exit code {code}")
        update_status(rows, exp.exp_id, "failed", ckpt=f"评估失败(code={code})")
        return False

    metrics = parse_metrics(eval_log)
    if not metrics:
        log(f"{exp.exp_id}: eval log parsed no metrics")
        update_status(rows, exp.exp_id, "failed", ckpt="评估失败:无指标")
        return False

    update_status(rows, exp.exp_id, "done", ckpt="epoch_60.pth", metrics=metrics)
    log(f"{exp.exp_id}: mIoU={metrics['mIoU']:.2f}, mAcc={metrics['mAcc']:.2f}, aAcc={metrics['aAcc']:.2f}")
    return True


def main():
    os.chdir(REPO)
    rows = load_results()
    save_results(rows)
    sync_docs(rows)

    log("=== v5 ablation queue started ===")
    for exp in EXPS:
        log(f"--- {exp.exp_id} start ---")
        if not train_exp(exp, rows):
            log(f"Training failed at {exp.exp_id}, continuing with next experiment")
            continue
        if not eval_exp(exp, rows):
            log(f"Eval failed at {exp.exp_id}, continuing with next experiment")
            continue
        log(f"--- {exp.exp_id} done ---")

    log("=== all 6 ablations completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
