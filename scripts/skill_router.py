#!/usr/bin/env python3
"""根据研究阶段与指标，自动推荐 Codex 技能组合。"""

from __future__ import annotations

import argparse
import json
from typing import Any


STAGE_SKILLS = {
    "planning": ["project-planner", "sprint-planner", "filesystem-context"],
    "training": ["computer-vision-expert", "hugging-face-vision-trainer"],
    "evaluation": ["data-analyst", "visualization-expert", "fact-checker"],
    "analysis": ["data-analyst", "visualization-expert", "fact-checker", "deep-research"],
    "ablation": ["computer-vision-expert", "data-analyst", "visualization-expert"],
    "paper": ["academic-researcher", "citation-management", "deep-research"],
    "automation": ["project-planner", "filesystem-context", "search-specialist"]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="技能自动路由")
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(STAGE_SKILLS.keys()),
        help="当前阶段"
    )
    parser.add_argument("--val-miou", type=float, default=None)
    parser.add_argument("--test-miou", type=float, default=None)
    parser.add_argument("--gap", type=float, default=None, help="val-test")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    return parser.parse_args()


def dedup(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def route(stage: str, val_miou: float | None, test_miou: float | None, gap: float | None) -> dict[str, Any]:
    skills = list(STAGE_SKILLS[stage])
    reasons = [f"阶段={stage}"]

    if gap is None and val_miou is not None and test_miou is not None:
        gap = val_miou - test_miou

    if gap is not None and gap >= 5.0:
        skills.extend(["data-analyst", "visualization-expert", "computer-vision-expert"])
        reasons.append(f"val-test gap={gap:.2f} 较大，优先误差分析与泛化诊断")

    if test_miou is not None and test_miou >= 67.0:
        skills.extend(["academic-researcher", "citation-management"])
        reasons.append(f"test mIoU={test_miou:.2f} 进入投稿候选区间，启动论文写作链路")

    if test_miou is not None and test_miou < 65.0:
        skills.extend(["computer-vision-expert", "fact-checker"])
        reasons.append(f"test mIoU={test_miou:.2f} 偏低，优先模型/数据链路排错")

    return {
        "stage": stage,
        "val_miou": val_miou,
        "test_miou": test_miou,
        "gap": gap,
        "skills": dedup(skills),
        "reasons": reasons,
    }


def main() -> int:
    args = parse_args()
    routed = route(args.stage, args.val_miou, args.test_miou, args.gap)

    if args.json:
        print(json.dumps(routed, ensure_ascii=False, indent=2))
        return 0

    print(f"[skill-router] stage={routed['stage']}")
    print(f"[skill-router] skills={', '.join(routed['skills'])}")
    for reason in routed["reasons"]:
        print(f"- {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
