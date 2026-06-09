#!/usr/bin/env python3
"""Build a deterministic markdown corpus for prompt-logprob PPL sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_GLOBS = ("docs/*.md", "tasks/*.md", "results/*.md")
DEFAULT_EXCLUDE_SUBSTRINGS = (
    "vllm_qwen_clean_ppl_",
    "_server",
    "_trace",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=1_500_000)
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--exclude-substring", action="append", default=[])
    return parser.parse_args()


def iter_sources(repo_root: Path, globs: list[str], exclude_substrings: tuple[str, ...]):
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(repo_root.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root)
            rel_text = rel.as_posix()
            if rel in seen:
                continue
            if any(fragment in rel_text for fragment in exclude_substrings):
                continue
            seen.add(rel)
            yield rel, path


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    globs = args.glob or list(DEFAULT_GLOBS)
    exclude_substrings = tuple(
        list(DEFAULT_EXCLUDE_SUBSTRINGS) + list(args.exclude_substring)
    )

    parts: list[str] = []
    sources: list[dict[str, object]] = []
    total_chars = 0

    for rel, path in iter_sources(repo_root, globs, exclude_substrings):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        header = f"\n\n<!-- source: {rel.as_posix()} -->\n\n"
        budget = args.max_chars - total_chars - len(header)
        if budget <= 0:
            break
        used_text = text[:budget]
        parts.append(header + used_text)
        total_chars += len(header) + len(used_text)
        sources.append(
            {
                "path": rel.as_posix(),
                "chars_used": len(used_text),
                "chars_total": len(text),
                "truncated": len(used_text) < len(text),
            }
        )
        if total_chars >= args.max_chars:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(parts).lstrip(), encoding="utf-8")
    manifest = {
        "schema": "ppl-corpus-manifest/v1",
        "output": args.output.as_posix(),
        "max_chars": args.max_chars,
        "actual_chars": len(args.output.read_text(encoding="utf-8")),
        "globs": globs,
        "exclude_substrings": list(exclude_substrings),
        "sources": sources,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("output", "actual_chars")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
