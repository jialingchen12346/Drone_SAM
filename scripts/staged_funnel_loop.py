#!/usr/bin/env python3
"""分阶段漏斗式实验调度（12/20/30）.

流程：
1) Stage-1: 多候选快速筛选（默认 12 epoch）
2) Stage-2: Top-K 复筛（默认 20 epoch）
3) Stage-3: Top-1 满训确认（默认 30 epoch）

底层复用 scripts/auto_train_eval_loop.py。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "research_workspace/plans/staged_funnel_strategy.json"


@dataclass
class StageResult:
    stage_name: str
    history_path: Path
    leaderboard_path: Path
    done_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分阶段漏斗式实验调度")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="策略配置 JSON")
    parser.add_argument("--execute", action="store_true", help="真实执行")
    parser.add_argument("--dry-run", action="store_true", help="仅预览")
    parser.add_argument("--max-stages", type=int, default=0, help="最多执行前 N 个 stage（0=全部）")
    parser.add_argument("--stop-on-error", action="store_true", default=True)
    parser.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def strip_flag_with_value(flags: list[str], key: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for idx, item in enumerate(flags):
        if skip_next:
            skip_next = False
            continue
        if item == key:
            if idx + 1 < len(flags):
                skip_next = True
            continue
        out.append(item)
    return out


def normalize_recipes(stage_recipe_names: list[str], recipe_catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []
    for name in stage_recipe_names:
        if name not in recipe_catalog:
            raise KeyError(f"recipe_catalog 不存在 recipe: {name}")
        src = recipe_catalog[name]
        recipe = {
            "name": src["name"],
            "tags": list(src.get("tags", [])),
            "flags": list(src.get("flags", [])),
        }
        if src.get("eval_flags"):
            recipe["eval_flags"] = list(src["eval_flags"])
        recipes.append(recipe)
    return recipes


def sort_done_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    done = [r for r in rows if r.get("status") == "done"]

    def score(row: dict[str, Any]) -> float:
        selected = row.get("selected", {})
        try:
            return float(selected.get("score", -1e9))
        except (TypeError, ValueError):
            return -1e9

    done.sort(key=score, reverse=True)
    return done


def get_recipe_name(row: dict[str, Any]) -> str:
    return str(row.get("recipe", "")).strip()


def get_test_miou(row: dict[str, Any]) -> float:
    selected = row.get("selected", {})
    try:
        return float(selected.get("test_mIoU", -1e9))
    except (TypeError, ValueError):
        return -1e9


def run_subprocess(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print("[cmd]", " ".join(subprocess.list2cmdline([x]) for x in cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)
    return proc.wait()


def build_stage_config(
    base_cfg: dict[str, Any],
    stage: dict[str, Any],
    stage_recipes: list[dict[str, Any]],
    stage_artifacts_dir: Path,
    stage_work_dir_root: Path,
) -> dict[str, Any]:
    base_train_flags = [str(x) for x in base_cfg.get("base_train_flags", [])]
    base_train_flags = strip_flag_with_value(base_train_flags, "--epochs")
    base_train_flags += ["--epochs", str(stage["epochs"])]

    out = {
        "repo_root": base_cfg["repo_root"],
        "launcher": base_cfg.get("launcher", "scripts/run_ddp_train.sh"),
        "work_dir_root": str(stage_work_dir_root),
        "artifacts_dir": str(stage_artifacts_dir),
        "train_env": dict(base_cfg.get("train_env", {})),
        "base_train_flags": base_train_flags,
        "eval": dict(base_cfg["eval"]),
        "recipes": stage_recipes,
        "feedback": dict(base_cfg.get("feedback", {})),
    }
    return out


def main() -> int:
    args = parse_args()
    dry_run = args.dry_run or not args.execute

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置不存在: {cfg_path}")

    cfg = load_json(cfg_path)
    repo_root = Path(cfg.get("repo_root", Path(__file__).resolve().parents[1])).resolve()
    artifacts_root = (repo_root / cfg.get("artifacts_root", "research_workspace/artifacts/staged_funnel")).resolve()
    work_root = Path(cfg.get("work_dir_root", "/root/autodl-tmp/work_dirs/staged_funnel")).resolve()

    auto_loop_script = (repo_root / cfg.get("auto_loop_script", "scripts/auto_train_eval_loop.py")).resolve()
    python_bin = str(cfg.get("python_bin", cfg.get("train_env", {}).get("PYTHON_BIN", sys.executable)))

    ensure_dir(artifacts_root)
    ensure_dir(work_root)

    recipe_catalog_list = cfg.get("recipe_catalog", [])
    if not recipe_catalog_list:
        raise ValueError("recipe_catalog 为空")
    recipe_catalog = {item["name"]: item for item in recipe_catalog_list}

    stages = cfg.get("stages", [])
    if not stages:
        raise ValueError("stages 为空")
    if args.max_stages > 0:
        stages = stages[: args.max_stages]

    env = os.environ.copy()
    for key, val in cfg.get("train_env", {}).items():
        env[str(key)] = str(val)
    env.setdefault("PYTHONUNBUFFERED", "1")

    prev_result: StageResult | None = None
    stage_report: list[dict[str, Any]] = []

    for idx, stage in enumerate(stages, start=1):
        stage_name = str(stage["name"])
        top_k_next = int(stage.get("top_k_next", 1))

        if "recipes" in stage:
            stage_recipe_names = [str(x) for x in stage["recipes"]]
        else:
            if prev_result is None:
                raise ValueError(f"stage={stage_name} 未提供 recipes，且不存在上一阶段结果")
            stage_recipe_names = [get_recipe_name(r) for r in prev_result.done_rows[:top_k_next] if get_recipe_name(r)]

        if not stage_recipe_names:
            print(f"[stage {idx}] {stage_name}: 无候选 recipe，停止")
            break

        stage_recipes = normalize_recipes(stage_recipe_names, recipe_catalog)

        stage_artifacts_dir = artifacts_root / stage_name
        stage_work_dir_root = work_root / stage_name
        ensure_dir(stage_artifacts_dir)
        ensure_dir(stage_work_dir_root)

        stage_auto_cfg = build_stage_config(
            base_cfg=cfg,
            stage=stage,
            stage_recipes=stage_recipes,
            stage_artifacts_dir=stage_artifacts_dir,
            stage_work_dir_root=stage_work_dir_root,
        )

        stage_cfg_path = stage_artifacts_dir / "auto_loop_config.generated.json"
        write_json(stage_cfg_path, stage_auto_cfg)

        max_cycles = int(stage.get("max_cycles", len(stage_recipes)))
        if dry_run:
            # auto_train_eval_loop 在 dry-run 时不写 history，连续多 cycle 会重复同一 recipe。
            # 这里固定为 1，仅用于命令预览。
            max_cycles = 1
        cmd = [python_bin, str(auto_loop_script), "--config", str(stage_cfg_path), "--max-cycles", str(max_cycles)]
        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd.append("--execute")

        print("\n" + "=" * 78)
        print(f"[stage {idx}] {stage_name}")
        print(f"[stage {idx}] epochs={stage['epochs']} recipes={','.join(stage_recipe_names)}")
        print(f"[stage {idx}] artifacts={stage_artifacts_dir}")
        print("=" * 78)

        ret = run_subprocess(cmd, cwd=repo_root, env=env)
        if ret != 0:
            print(f"[stage {idx}] 失败: exit={ret}")
            if args.stop_on_error:
                return ret

        history_path = stage_artifacts_dir / "history.jsonl"
        leaderboard_path = stage_artifacts_dir / "leaderboard.md"
        done_rows = sort_done_rows(load_history(history_path))

        if done_rows:
            best_test = get_test_miou(done_rows[0])
            abort_if_best_below = stage.get("abort_if_best_below")
            if abort_if_best_below is not None and best_test < float(abort_if_best_below):
                print(
                    f"[stage {idx}] 触发止损: best_test={best_test:.2f} < {float(abort_if_best_below):.2f}"
                )
                stage_report.append(
                    {
                        "stage": stage_name,
                        "done": len(done_rows),
                        "best_recipe": get_recipe_name(done_rows[0]),
                        "best_test": round(best_test, 2),
                        "stopped_by_threshold": True,
                    }
                )
                break

        prev_result = StageResult(
            stage_name=stage_name,
            history_path=history_path,
            leaderboard_path=leaderboard_path,
            done_rows=done_rows,
        )

        stage_report.append(
            {
                "stage": stage_name,
                "done": len(done_rows),
                "best_recipe": get_recipe_name(done_rows[0]) if done_rows else "",
                "best_test": round(get_test_miou(done_rows[0]), 2) if done_rows else None,
                "history": str(history_path),
                "leaderboard": str(leaderboard_path),
            }
        )

    report_path = artifacts_root / "staged_funnel_report.json"
    write_json(report_path, {"config": str(cfg_path), "dry_run": dry_run, "stages": stage_report})
    print(f"[staged-funnel] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
