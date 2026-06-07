#!/usr/bin/env python3
"""Audit solution-plan coverage across status docs and issue tracking."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SOLUTION_HEADING_RE = re.compile(r"^## (?P<id>\d+[a-z]?)\. (?P<title>.+)$")
TABLE_ROW_RE = re.compile(r"^\|(?P<cells>.+)\|$")
STATUS_ID_RE = re.compile(r"^(?P<id>\d+[a-z]?)\.\s+(?P<title>.+)$")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def table_cells(line: str) -> list[str] | None:
    match = TABLE_ROW_RE.match(line)
    if not match:
        return None
    cells = [cell.strip() for cell in match.group("cells").split("|")]
    if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
        return None
    return cells


def extract_solution_headings(path: Path) -> list[dict[str, Any]]:
    lines = read(path).splitlines()
    headings: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_acceptance = False

    for line in lines:
        heading = SOLUTION_HEADING_RE.match(line)
        if heading:
            current = {
                "id": heading.group("id"),
                "title": heading.group("title"),
                "acceptance_items": 0,
            }
            headings.append(current)
            in_acceptance = False
            continue

        if line.startswith("## "):
            current = None
            in_acceptance = False
            continue

        if current is None:
            continue
        if line.strip() == "Acceptance test:":
            in_acceptance = True
            continue
        if in_acceptance and line.startswith("- "):
            current["acceptance_items"] += 1
        elif in_acceptance and line.strip() and not line.startswith("- "):
            in_acceptance = False

    return headings


def extract_status_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read(path).splitlines():
        cells = table_cells(line)
        if cells is None or cells[0] == "solution area":
            continue
        first = cells[0].replace("`", "")
        match = STATUS_ID_RE.match(first)
        if not match:
            continue
        rows.append(
            {
                "id": match.group("id"),
                "title": match.group("title").strip("`"),
                "status": cells[1] if len(cells) > 1 else "",
            }
        )
    return rows


def normalize_plan_ids(raw: str) -> list[str]:
    text = raw.replace("`", "")
    ids: list[str] = []
    for part in re.split(r"[,/ ]+", text):
        part = part.strip()
        if re.fullmatch(r"\d+[a-z]?|qwen-speed", part):
            ids.append(part)
    return ids


def extract_issue_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read(path).splitlines():
        cells = table_cells(line)
        if cells is None or not cells or cells[0] == "plan id(s)":
            continue
        if len(cells) < 4:
            continue
        rows.append(
            {
                "plan_ids": normalize_plan_ids(cells[0]),
                "area": cells[1].replace("`", ""),
                "status": cells[2],
                "issue": cells[3],
            }
        )
    return rows


def contains_all(path: Path, needles: tuple[str, ...]) -> bool:
    text = read(path)
    return all(needle in text for needle in needles)


def qwen_lane_checks(root: Path) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "readme_fixed_criteria_mentions_qwen_speed_capacity",
            "path": "README.md",
            "ok": contains_all(root / "README.md", ("Qwen speed/capacity rows exist",)),
        },
        {
            "name": "benchmark_protocol_requires_qwen_and_gemma",
            "path": "docs/BENCHMARK_PROTOCOL.md",
            "ok": contains_all(
                root / "docs" / "BENCHMARK_PROTOCOL.md",
                (
                    "Qwen speed/capacity: required",
                    "Do not generalize a Gemma-only row to Qwen",
                ),
            ),
        },
        {
            "name": "qwen_notes_doc_exists",
            "path": "docs/QWEN_ON_DGX_SPARK.md",
            "ok": (root / "docs" / "QWEN_ON_DGX_SPARK.md").exists(),
        },
        {
            "name": "qwen_speed_lane_runner_exists",
            "path": "scripts/qwen_speed_lane.py",
            "ok": (root / "scripts" / "qwen_speed_lane.py").exists(),
        },
        {
            "name": "qwen_speed_lane_sample_exists",
            "path": "tasks/qwen_speed_lane_sample.jsonl",
            "ok": (root / "tasks" / "qwen_speed_lane_sample.jsonl").exists(),
        },
    ]
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/solution_coverage_audit.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    solution_headings = extract_solution_headings(root / "docs" / "DGX_SPARK_SOLUTIONS.md")
    status_rows = extract_status_rows(root / "docs" / "SOLUTIONS_STATUS.md")
    issue_rows = extract_issue_rows(root / "docs" / "ISSUE_TRACKER.md")

    solution_ids = [row["id"] for row in solution_headings]
    status_ids = [row["id"] for row in status_rows]
    issue_ids = sorted({plan_id for row in issue_rows for plan_id in row["plan_ids"]})
    qwen_checks = qwen_lane_checks(root)
    qwen_issue_ok = any("qwen-speed" in row["plan_ids"] for row in issue_rows)

    summary: dict[str, Any] = {
        "schema": "solution-coverage-audit/v1",
        "solution_headings": solution_headings,
        "status_rows": status_rows,
        "issue_rows": issue_rows,
        "missing_status_ids": [plan_id for plan_id in solution_ids if plan_id not in status_ids],
        "extra_status_ids": [status_id for status_id in status_ids if status_id not in solution_ids],
        "missing_issue_plan_ids": [plan_id for plan_id in solution_ids if plan_id not in issue_ids],
        "qwen_speed_lane": {
            "issue_row_ok": qwen_issue_ok,
            "checks": qwen_checks,
        },
    }
    summary["ok"] = (
        not summary["missing_status_ids"]
        and not summary["extra_status_ids"]
        and not summary["missing_issue_plan_ids"]
        and qwen_issue_ok
        and all(check["ok"] for check in qwen_checks)
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
