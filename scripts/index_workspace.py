#!/usr/bin/env python3
"""
Build and search a lightweight local index for long-running research work.

Examples:
    python scripts/index_workspace.py build
    python scripts/index_workspace.py search "bus IoU ablation"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "research_workspace" / "artifacts" / "index"
INDEX_PATH = INDEX_DIR / "workspace_index.json"

TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".tex",
    ".bib",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".log",
    ".sh",
}

SEARCH_ROOTS = [
    REPO_ROOT / "docs",
    REPO_ROOT / "scripts",
    REPO_ROOT / "segmentation" / "configs",
    REPO_ROOT / "segmentation" / "work_dirs",
    REPO_ROOT / "work_dirs",
    REPO_ROOT / "research_workspace",
]

SKIP_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}

SKIP_SUFFIXES = {
    ".pth",
    ".pt",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".npy",
    ".npz",
    ".so",
    ".bin",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-\.]{2,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "work",
    "works",
    "current",
    "using",
    "use",
    "when",
}


@dataclass
class Document:
    path: str
    category: str
    size: int
    preview: str
    token_counts: Counter


def tokenize(text: str) -> list[str]:
    tokens = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        parts = token.replace("-", "_").split("_")
        for part in parts:
            if len(part) < 2 or part in STOPWORDS:
                continue
            tokens.append(part)
    return tokens


def should_skip(path: Path) -> bool:
    if any(part in SKIP_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def iter_files() -> Iterable[Path]:
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if should_skip(path):
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            yield path


def categorize(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT)
    top = rel.parts[0]
    if top == "work_dirs":
        return "work_dir"
    if top == "docs":
        return "docs"
    if top == "scripts":
        return "scripts"
    if top == "research_workspace":
        return "workspace"
    return top


def build_index(max_chars: int = 200_000) -> dict:
    documents: list[Document] = []
    doc_freq: Counter = Counter()
    total_files = 0

    for path in iter_files():
        total_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        trimmed = text[:max_chars]
        tokens = tokenize(trimmed)
        if not tokens:
            continue
        counts = Counter(tokens)
        doc_freq.update(counts.keys())
        preview = " ".join(trimmed.split())[:240]
        rel = str(path.relative_to(REPO_ROOT))
        documents.append(
            Document(
                path=rel,
                category=categorize(path),
                size=len(trimmed),
                preview=preview,
                token_counts=counts,
            )
        )

    payload = {
        "repo_root": str(REPO_ROOT),
        "document_count": len(documents),
        "indexed_files_seen": total_files,
        "documents": [
            {
                "path": doc.path,
                "category": doc.category,
                "size": doc.size,
                "preview": doc.preview,
                "token_counts": dict(doc.token_counts),
            }
            for doc in documents
        ],
        "doc_freq": dict(doc_freq),
    }
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    return payload


def load_index() -> dict:
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"Index not found: {INDEX_PATH}\nRun `python scripts/index_workspace.py build` first."
        )
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def score_document(query_tokens: list[str], doc: dict, doc_freq: dict, total_docs: int) -> float:
    counts = doc["token_counts"]
    if not counts:
        return 0.0
    length = sum(counts.values())
    score = 0.0
    path_tokens = set(tokenize(doc["path"]))
    for token in query_tokens:
        tf = counts.get(token, 0)
        df = max(int(doc_freq.get(token, 0)), 1)
        idf = math.log(1 + (total_docs / df))
        if tf:
            score += (tf / length) * idf * 100
        if token in path_tokens:
            score += idf * 2.0
    return score


def search_index(query: str, limit: int) -> list[tuple[float, dict]]:
    payload = load_index()
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    total_docs = max(int(payload["document_count"]), 1)
    scored = []
    for doc in payload["documents"]:
        score = score_document(query_tokens, doc, payload["doc_freq"], total_docs)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], item[1]["path"]))
    return scored[:limit]


def cmd_build(_: argparse.Namespace) -> None:
    payload = build_index()
    print(f"Indexed {payload['document_count']} documents")
    print(f"Index written to {INDEX_PATH}")


def cmd_search(args: argparse.Namespace) -> None:
    results = search_index(args.query, args.limit)
    if not results:
        print("No matches found")
        return
    for rank, (score, doc) in enumerate(results, start=1):
        print(f"{rank}. [{doc['category']}] {doc['path']}  score={score:.3f}")
        print(f"   {doc['preview']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the local workspace index")
    build.set_defaults(func=cmd_build)

    search = subparsers.add_parser("search", help="Search the local workspace index")
    search.add_argument("query", help="Query string")
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(func=cmd_search)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
