#!/usr/bin/env python3
"""Validate live task definitions for missing counterpart evidence rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "requirement",
    "issue",
    "runtime",
    "model_family",
    "status",
    "run_id_template",
    "needs_live_gb10",
    "commands",
    "expected_claim_artifacts",
)


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: top-level JSON must be an object")
    return data


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


def validate_command(row: dict[str, Any], command: Any, index: int) -> list[str]:
    findings: list[str] = []
    req = row.get("requirement", f"line {row.get('_line', '?')}")
    if not isinstance(command, dict):
        return [f"{req}: command {index} is not an object"]
    for field in ("name", "shell"):
        value = command.get(field)
        if not isinstance(value, str) or not value.strip():
            findings.append(f"{req}: command {index} missing non-empty {field}")
    artifacts = command.get("expected_artifacts", [])
    if artifacts and not isinstance(artifacts, list):
        findings.append(f"{req}: command {index} expected_artifacts must be a list")
    for artifact in artifacts if isinstance(artifacts, list) else []:
        if not isinstance(artifact, str) or not artifact.startswith("results/"):
            findings.append(f"{req}: command {index} artifact must be results/*: {artifact!r}")
    return findings


def validate_task(row: dict[str, Any]) -> dict[str, Any]:
    findings: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            findings.append(f"missing required field {field}")
    for field in ("requirement", "issue", "runtime", "model_family", "status", "run_id_template"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            findings.append(f"field {field} must be a non-empty string")
    if not isinstance(row.get("needs_live_gb10"), bool):
        findings.append("field needs_live_gb10 must be boolean")
    commands = row.get("commands")
    if not isinstance(commands, list) or not commands:
        findings.append("commands must be a non-empty list")
    else:
        for index, command in enumerate(commands):
            findings.extend(validate_command(row, command, index))
    artifacts = row.get("expected_claim_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        findings.append("expected_claim_artifacts must be a non-empty list")
    else:
        for artifact in artifacts:
            if not isinstance(artifact, str) or not artifact.startswith("results/"):
                findings.append(f"expected claim artifact must be results/*: {artifact!r}")
    return {
        "requirement": row.get("requirement"),
        "line": row.get("_line"),
        "ok": not findings,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="tasks/counterpart_evidence_tasks.jsonl")
    parser.add_argument("--audit", default="results/counterpart_evidence_audit_20260608.json")
    parser.add_argument("--output", default="results/counterpart_task_matrix.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tasks_path = Path(args.tasks)
    audit_path = Path(args.audit)
    output = Path(args.output)
    rows = load_jsonl(tasks_path)
    audit = load_json(audit_path) if audit_path.exists() else {}
    audit_requirements = {
        str(req.get("name"))
        for req in audit.get("requirements", [])
        if isinstance(req, dict) and req.get("claim_ready") is False
    }
    claim_ready_requirements = {
        str(req.get("name"))
        for req in audit.get("requirements", [])
        if isinstance(req, dict) and req.get("claim_ready") is True
    }
    task_requirements = {str(row.get("requirement")) for row in rows}
    duplicate_requirements = sorted(
        requirement
        for requirement in task_requirements
        if sum(1 for row in rows if row.get("requirement") == requirement) > 1
    )
    row_checks = [validate_task(row) for row in rows]
    missing_tasks_for_audit = sorted(audit_requirements - task_requirements)
    task_already_claim_ready = sorted(task_requirements & claim_ready_requirements)
    task_without_audit_requirement = sorted(
        task_requirements - audit_requirements - claim_ready_requirements
    ) if audit_requirements or claim_ready_requirements else []
    summary = {
        "schema": "counterpart-task-matrix/v1",
        "tasks": rel(tasks_path, root),
        "audit": rel(audit_path, root) if audit_path.exists() else None,
        "task_count": len(rows),
        "audit_missing_or_partial_count": len(audit_requirements),
        "missing_tasks_for_audit": missing_tasks_for_audit,
        "task_already_claim_ready": task_already_claim_ready,
        "task_without_audit_requirement": task_without_audit_requirement,
        "duplicate_requirements": duplicate_requirements,
        "row_checks": row_checks,
        "ok": (
            all(check["ok"] for check in row_checks)
            and not missing_tasks_for_audit
            and not duplicate_requirements
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    output.write_text(text)
    print(text, end="")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
