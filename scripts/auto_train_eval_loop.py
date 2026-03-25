#!/usr/bin/env python3
"""自动化训练-评测-反馈循环脚本（RRF-DSD/FMB）。

功能：
1. 从 JSON 配置读取实验 recipe。
2. 自动选择下一轮 recipe（基于 val-test gap 与停滞信号）。
3. 运行训练、评测（best + latest epoch；val+test）。
4. 记录历史（JSONL）与排行榜（CSV/MD）。
5. 给出下一轮建议与技能建议。

示例：
    python scripts/auto_train_eval_loop.py --dry-run
    python scripts/auto_train_eval_loop.py --execute --max-cycles 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "research_workspace/plans/auto_loop_config.json"
METRIC_RE = re.compile(r"(mIoU|mAcc|aAcc)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
EPOCH_CKPT_RE = re.compile(r"epoch_(\d+)\.pth$")


@dataclass
class EvalResult:
    checkpoint: str
    split: str
    mIoU: float
    mAcc: float
    aAcc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动训练-评测-反馈循环")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="JSON 配置路径")
    parser.add_argument("--max-cycles", type=int, default=1, help="最大循环次数")
    parser.add_argument("--execute", action="store_true", help="真实执行训练与评测")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    parser.add_argument("--stop-on-error", action="store_true", default=True,
                        help="命令失败即停止（默认开启）")
    parser.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_and_stream(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None,
    dry_run: bool,
) -> tuple[int, str]:
    cmd_str = " ".join(subprocess.list2cmdline([item]) for item in cmd)
    print(f"[cmd] {cmd_str}")
    if dry_run:
        return 0, ""

    ensure_dir(log_path.parent)
    collected: list[str] = []
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n\n===== {datetime.now().isoformat()} =====\n")
        logf.write(f"$ {cmd_str}\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            collected.append(line)
        ret = proc.wait()
        logf.write(f"[exit_code] {ret}\n")
        logf.flush()

    return ret, "".join(collected)


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        for name, value in METRIC_RE.findall(line):
            metrics[name] = float(value)
    return metrics


def load_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_history(history_path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(history_path.parent)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_next_recipe(
    recipes: list[dict[str, Any]],
    history: list[dict[str, Any]],
    gap_threshold: float,
    stagnation_delta: float,
) -> dict[str, Any] | None:
    completed = {item.get("recipe", "") for item in history if item.get("status") == "done"}
    candidates = [r for r in recipes if r.get("name") not in completed]
    if not candidates:
        return None

    if not history:
        return candidates[0]

    last = history[-1]
    last_gap = to_float(last.get("selected", {}).get("val_test_gap"), 0.0)
    scores: list[tuple[float, dict[str, Any]]] = []

    recent = [to_float(item.get("selected", {}).get("test_mIoU"), 0.0) for item in history[-3:]]
    stagnated = False
    if len(recent) >= 2:
        stagnated = (max(recent) - min(recent)) < stagnation_delta

    for idx, recipe in enumerate(candidates):
        tags = set(recipe.get("tags", []))
        score = 0.0

        if last_gap >= gap_threshold and "generalization" in tags:
            score += 3.0
        if stagnated and "strong_aug" in tags:
            score += 2.0
        if last_gap < gap_threshold and "ablation" in tags:
            score += 1.5

        # 稳定顺序：同分时按配置顺序优先
        score += max(0, 1.0 - idx * 0.01)
        scores.append((score, recipe))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1]


def latest_epoch_checkpoint(work_dir: Path) -> Path | None:
    best_epoch = -1
    best_path: Path | None = None
    for path in work_dir.glob("epoch_*.pth"):
        match = EPOCH_CKPT_RE.search(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = path
    return best_path


def build_skill_recommendations(
    stage: str,
    val_miou: float,
    test_miou: float,
    gap: float,
) -> list[str]:
    base_map = {
        "planning": ["project-planner", "sprint-planner", "filesystem-context"],
        "training": ["computer-vision-expert", "hugging-face-vision-trainer"],
        "evaluation": ["data-analyst", "visualization-expert"],
        "analysis": ["data-analyst", "fact-checker", "visualization-expert"],
        "paper": ["academic-researcher", "citation-management", "deep-research"],
    }
    skills = list(base_map.get(stage, ["project-planner", "data-analyst"]))

    if gap >= 5.0:
        for s in ["data-analyst", "visualization-expert", "fact-checker"]:
            if s not in skills:
                skills.append(s)
    if test_miou >= 67.0:
        for s in ["academic-researcher", "citation-management"]:
            if s not in skills:
                skills.append(s)
    if val_miou - test_miou >= 6.0 and "computer-vision-expert" not in skills:
        skills.append("computer-vision-expert")

    return skills


def write_leaderboard(history: list[dict[str, Any]], artifacts_dir: Path) -> None:
    ensure_dir(artifacts_dir)
    csv_path = artifacts_dir / "leaderboard.csv"
    md_path = artifacts_dir / "leaderboard.md"

    rows = [h for h in history if h.get("status") == "done"]
    rows.sort(key=lambda x: to_float(x.get("selected", {}).get("score"), -1e9), reverse=True)

    fields = [
        "timestamp",
        "recipe",
        "work_dir",
        "selected_checkpoint",
        "val_mIoU",
        "test_mIoU",
        "val_test_gap",
        "score",
        "suggested_next_recipe",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            sel = row.get("selected", {})
            writer.writerow(
                {
                    "timestamp": row.get("timestamp", ""),
                    "recipe": row.get("recipe", ""),
                    "work_dir": row.get("work_dir", ""),
                    "selected_checkpoint": sel.get("checkpoint", ""),
                    "val_mIoU": sel.get("val_mIoU", ""),
                    "test_mIoU": sel.get("test_mIoU", ""),
                    "val_test_gap": sel.get("val_test_gap", ""),
                    "score": sel.get("score", ""),
                    "suggested_next_recipe": row.get("suggested_next_recipe", ""),
                }
            )

    lines = [
        "# Auto Loop Leaderboard",
        "",
        f"Total completed runs: {len(rows)}",
        "",
        "| Rank | Recipe | test mIoU | val mIoU | Gap | Score | Ckpt |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        sel = row.get("selected", {})
        lines.append(
            "| {rank} | {recipe} | {test:.2f} | {val:.2f} | {gap:.2f} | {score:.2f} | `{ckpt}` |".format(
                rank=idx,
                recipe=row.get("recipe", ""),
                test=to_float(sel.get("test_mIoU"), 0.0),
                val=to_float(sel.get("val_mIoU"), 0.0),
                gap=to_float(sel.get("val_test_gap"), 0.0),
                score=to_float(sel.get("score"), 0.0),
                ckpt=sel.get("checkpoint", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dry_run = args.dry_run or not args.execute

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    cfg = load_json(config_path)
    repo_root = Path(cfg.get("repo_root", Path(__file__).resolve().parents[1])).resolve()
    artifacts_dir = (repo_root / cfg.get("artifacts_dir", "research_workspace/artifacts/auto_loop")).resolve()
    history_path = artifacts_dir / "history.jsonl"
    logs_dir = artifacts_dir / "logs"

    launcher = str((repo_root / cfg.get("launcher", "scripts/run_ddp_train.sh")).resolve())
    eval_script = str((repo_root / cfg["eval"]["script"]).resolve())

    base_train_flags = cfg.get("base_train_flags", [])
    recipes = cfg.get("recipes", [])
    if not recipes:
        raise ValueError("配置中 recipes 为空，无法自动搜索")

    feedback = cfg.get("feedback", {})
    gap_threshold = to_float(feedback.get("gap_threshold", 5.0), 5.0)
    stagnation_delta = to_float(feedback.get("stagnation_delta", 0.3), 0.3)
    gap_penalty = to_float(feedback.get("gap_penalty", 0.25), 0.25)

    env = os.environ.copy()
    for key, value in cfg.get("train_env", {}).items():
        env[str(key)] = str(value)
    env.setdefault("PYTHONUNBUFFERED", "1")

    work_dir_root = (repo_root / cfg.get("work_dir_root", "work_dirs/auto_loop")).resolve()
    ensure_dir(work_dir_root)

    history = load_history(history_path)

    for cycle in range(1, args.max_cycles + 1):
        cycle_t0 = time.time()
        recipe = choose_next_recipe(recipes, history, gap_threshold, stagnation_delta)
        if recipe is None:
            print("[auto-loop] 所有 recipe 已完成，无可执行项。")
            break

        run_name = f"{recipe['name']}_{now_ts()}"
        work_dir = work_dir_root / run_name
        ensure_dir(work_dir)

        train_cmd = [launcher] + list(base_train_flags) + list(recipe.get("flags", [])) + ["--work-dir", str(work_dir)]
        train_log = logs_dir / f"{run_name}_train.log"

        print("\n" + "=" * 72)
        print(f"[cycle {cycle}] recipe={recipe['name']}  dry_run={dry_run}")
        print(f"[cycle {cycle}] work_dir={work_dir}")
        print("=" * 72)

        ret, _ = run_and_stream(
            cmd=train_cmd,
            cwd=repo_root,
            log_path=train_log,
            env=env,
            dry_run=dry_run,
        )
        if ret != 0:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "recipe": recipe["name"],
                "work_dir": str(work_dir),
                "error": f"train_exit_code={ret}",
            }
            append_history(history_path, payload)
            history.append(payload)
            print(f"[error] 训练失败: exit_code={ret}")
            if args.stop_on_error:
                break
            continue

        if dry_run:
            print("[dry-run] 仅预览命令，不写入 history/leaderboard。")
            continue

        # 评测 checkpoint 集合：best + latest epoch
        checkpoints: list[Path] = []
        best_ckpt = work_dir / "best.pth"
        if best_ckpt.exists():
            checkpoints.append(best_ckpt)
        latest_ckpt = latest_epoch_checkpoint(work_dir)
        if latest_ckpt and latest_ckpt not in checkpoints:
            checkpoints.append(latest_ckpt)
        if not checkpoints:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "recipe": recipe["name"],
                "work_dir": str(work_dir),
                "error": "no_checkpoint_found",
            }
            append_history(history_path, payload)
            history.append(payload)
            print("[error] 未找到 checkpoint，跳过评测")
            if args.stop_on_error:
                break
            continue

        eval_cfg = cfg["eval"]
        python_bin = str(eval_cfg.get("python_bin", env.get("PYTHON_BIN", "python")))
        splits: list[str] = list(eval_cfg.get("splits", ["val", "test"]))
        eval_base_flags: list[str] = [str(x) for x in eval_cfg.get("extra_flags", [])]
        recipe_eval_flags: list[str] = [str(x) for x in recipe.get("eval_flags", [])]

        eval_rows: list[EvalResult] = []
        for ckpt in checkpoints:
            for split in splits:
                eval_cmd = [
                    python_bin,
                    eval_script,
                    "--checkpoint",
                    str(ckpt),
                    "--data-root",
                    str(eval_cfg["data_root"]),
                    "--split",
                    split,
                    "--eval-resize-mode",
                    str(eval_cfg.get("eval_resize_mode", "letterbox")),
                    "--crop-size",
                    str(eval_cfg.get("crop_size", 512)),
                    "--batch-size",
                    str(eval_cfg.get("batch_size", 1)),
                    "--num-workers",
                    str(eval_cfg.get("num_workers", 4)),
                ] + eval_base_flags + recipe_eval_flags
                eval_log = logs_dir / f"{run_name}_eval_{ckpt.stem}_{split}.log"
                ret_eval, out_eval = run_and_stream(
                    cmd=eval_cmd,
                    cwd=repo_root,
                    log_path=eval_log,
                    env=env,
                    dry_run=False,
                )
                if ret_eval != 0:
                    raise RuntimeError(f"评测失败: {ckpt.name} split={split} exit={ret_eval}")

                parsed = parse_metrics(out_eval)
                if not {"mIoU", "mAcc", "aAcc"}.issubset(parsed.keys()):
                    raise RuntimeError(f"无法从评测输出解析完整指标: {ckpt.name} split={split}")

                eval_rows.append(
                    EvalResult(
                        checkpoint=ckpt.name,
                        split=split,
                        mIoU=parsed["mIoU"],
                        mAcc=parsed["mAcc"],
                        aAcc=parsed["aAcc"],
                    )
                )

        grouped: dict[str, dict[str, EvalResult]] = {}
        for row in eval_rows:
            grouped.setdefault(row.checkpoint, {})[row.split] = row

        selected_ckpt = None
        selected_test = -1.0
        for ckpt_name, split_map in grouped.items():
            test_score = split_map.get("test", split_map.get("val", None))
            if test_score and test_score.mIoU > selected_test:
                selected_test = test_score.mIoU
                selected_ckpt = ckpt_name

        if not selected_ckpt:
            raise RuntimeError("无法选择 checkpoint")

        selected_map = grouped[selected_ckpt]
        val_miou = selected_map.get("val", selected_map.get("test")).mIoU
        test_miou = selected_map.get("test", selected_map.get("val")).mIoU
        gap = val_miou - test_miou
        score = test_miou - gap_penalty * max(0.0, gap)

        next_recipe = choose_next_recipe(recipes, history + [{"status": "done", "recipe": recipe["name"], "selected": {"val_test_gap": gap, "test_mIoU": test_miou}}], gap_threshold, stagnation_delta)
        next_name = next_recipe["name"] if next_recipe else ""

        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": "done",
            "recipe": recipe["name"],
            "tags": recipe.get("tags", []),
            "work_dir": str(work_dir),
            "selected": {
                "checkpoint": selected_ckpt,
                "val_mIoU": round(val_miou, 2),
                "test_mIoU": round(test_miou, 2),
                "val_test_gap": round(gap, 2),
                "score": round(score, 2),
            },
            "eval_rows": [
                {
                    "checkpoint": row.checkpoint,
                    "split": row.split,
                    "mIoU": round(row.mIoU, 2),
                    "mAcc": round(row.mAcc, 2),
                    "aAcc": round(row.aAcc, 2),
                }
                for row in eval_rows
            ],
            "suggested_next_recipe": next_name,
            "suggested_skills": build_skill_recommendations(
                stage="analysis",
                val_miou=val_miou,
                test_miou=test_miou,
                gap=gap,
            ),
            "duration_sec": round(time.time() - cycle_t0, 1),
        }

        append_history(history_path, payload)
        history.append(payload)
        write_leaderboard(history, artifacts_dir)

        print(
            f"[cycle {cycle}] done | recipe={recipe['name']} | "
            f"selected_ckpt={selected_ckpt} | val={val_miou:.2f} | test={test_miou:.2f} | gap={gap:.2f}"
        )
        if next_name:
            print(f"[cycle {cycle}] suggested_next_recipe={next_name}")

    write_leaderboard(history, artifacts_dir)
    print(f"[auto-loop] history: {history_path}")
    print(f"[auto-loop] leaderboard: {artifacts_dir / 'leaderboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
