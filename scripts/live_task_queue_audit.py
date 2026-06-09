#!/usr/bin/env python3
"""Validate the ordered live-task queue used when the GB10 host is reachable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "id",
    "priority",
    "lane",
    "issue",
    "status",
    "needs_live_gb10",
    "why_now",
    "acceptance",
)


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{lineno}: row must be a JSON object")
            row["_line"] = lineno
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no task rows found")
    return rows


def validate_string(row: dict[str, Any], field: str, findings: list[str]) -> None:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        findings.append(f"{field} must be a non-empty string")


def validate_string_list(
    row: dict[str, Any],
    field: str,
    findings: list[str],
    *,
    required: bool = False,
) -> None:
    value = row.get(field)
    if value is None and not required:
        return
    if not isinstance(value, list) or (required and not value):
        findings.append(f"{field} must be a {'non-empty ' if required else ''}list")
        return
    for item in value:
        if not isinstance(item, str) or not item.strip():
            findings.append(f"{field} entries must be non-empty strings")


def task_ref_exists(root: Path, raw_ref: str) -> bool:
    path_text, _, requirement = raw_ref.partition("#")
    path = root / path_text
    if not path.exists():
        return False
    if not requirement:
        return True
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    needle = f'"requirement":"{requirement}"'
    return any(needle in line for line in lines)


def validate_row(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    findings: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            findings.append(f"missing required field {field}")
    for field in ("id", "lane", "issue", "status", "why_now"):
        validate_string(row, field, findings)
    priority = row.get("priority")
    if not isinstance(priority, int) or priority <= 0:
        findings.append("priority must be a positive integer")
    if not isinstance(row.get("needs_live_gb10"), bool):
        findings.append("needs_live_gb10 must be boolean")
    if isinstance(row.get("issue"), str) and not str(row["issue"]).startswith("#"):
        findings.append("issue should be a GitHub issue reference such as #7")
    task_packet = row.get("task_packet")
    task_ref = row.get("task_ref")
    if not task_packet and not task_ref:
        findings.append("one of task_packet or task_ref is required")
    if task_packet:
        if not isinstance(task_packet, str):
            findings.append("task_packet must be a string")
        elif not (root / task_packet).exists():
            findings.append(f"task_packet does not exist: {task_packet}")
    if task_ref:
        if not isinstance(task_ref, str):
            findings.append("task_ref must be a string")
        elif not task_ref_exists(root, task_ref):
            findings.append(f"task_ref does not resolve: {task_ref}")
    validate_string_list(row, "blocked_by", findings)
    validate_string_list(row, "acceptance", findings, required=True)
    validate_string_list(row, "expected_artifacts", findings)
    return {
        "id": row.get("id"),
        "line": row.get("_line"),
        "priority": row.get("priority"),
        "ok": not findings,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="tasks/live_gb10_queue.jsonl")
    parser.add_argument("--output", default="results/live_task_queue_audit_20260609.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    queue_path = Path(args.queue)
    output = Path(args.output)
    rows = load_jsonl(queue_path)
    row_checks = [validate_row(root, row) for row in rows]
    ids = [str(row.get("id")) for row in rows]
    duplicate_ids = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    priorities = [
        row.get("priority") for row in rows if isinstance(row.get("priority"), int)
    ]
    priority_order_ok = priorities == sorted(priorities)
    if not priority_order_ok:
        row_checks.append(
            {
                "id": "__queue_order__",
                "line": None,
                "priority": None,
                "ok": False,
                "findings": ["queue rows must be sorted by ascending priority"],
            }
        )
    if duplicate_ids:
        row_checks.append(
            {
                "id": "__duplicates__",
                "line": None,
                "priority": None,
                "ok": False,
                "findings": [f"duplicate ids: {', '.join(duplicate_ids)}"],
            }
        )

    live_required = [row for row in rows if row.get("needs_live_gb10") is True]
    offline_possible = [row for row in rows if row.get("needs_live_gb10") is False]
    summary = {
        "schema": "live-task-queue-audit/v1",
        "queue": rel(queue_path, root),
        "task_count": len(rows),
        "live_gb10_required_count": len(live_required),
        "offline_or_non_gb10_count": len(offline_possible),
        "duplicate_ids": duplicate_ids,
        "priority_order_ok": priority_order_ok,
        "row_checks": row_checks,
        "next_live_tasks": [
            {
                "id": row["id"],
                "priority": row["priority"],
                "lane": row["lane"],
                "issue": row["issue"],
                "task_packet": row.get("task_packet"),
                "task_ref": row.get("task_ref"),
            }
            for row in sorted(live_required, key=lambda item: item["priority"])[:5]
        ],
        "ok": all(check["ok"] for check in row_checks) and not duplicate_ids,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    output.write_text(text)
    print(text, end="")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
