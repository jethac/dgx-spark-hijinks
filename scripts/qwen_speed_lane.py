#!/usr/bin/env python3
"""Run a Qwen speed/capacity lane against OpenAI-compatible servers."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPTIONAL_STRING_FIELDS = [
    "model_revision",
    "runtime_ref",
    "container_image",
    "container_digest",
    "quantization",
    "kv_cache_dtype",
    "attention_backend",
    "cuda_graph_mode",
    "server_log",
    "process_match",
    "llama_bench_command",
]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def sanitize(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    clean = clean.strip("._-")
    return clean.lower() or "row"


def load_rows(path: Path) -> list[dict[str, Any]]:
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
            row["_source_lineno"] = lineno
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows found")
    return rows


def require(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        lineno = row.get("_source_lineno", "?")
        raise SystemExit(f"row at line {lineno}: missing required string field {field!r}")
    return value.strip()


def build_command(
    row: dict[str, Any],
    *,
    repo_root: Path,
    results_dir: Path,
    campaign_id: str,
    dry_run: bool,
) -> tuple[str, list[str]]:
    backend = require(row, "backend")
    phase = row.get("phase", "exploratory")
    if phase not in {"before", "after", "exploratory"}:
        lineno = row.get("_source_lineno", "?")
        raise SystemExit(f"row at line {lineno}: invalid phase {phase!r}")
    name = require(row, "name")
    url = require(row, "url")
    model = require(row, "model")
    run_id = row.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = f"{campaign_id}_{sanitize(backend)}_{sanitize(name)}"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "record_openai_serving_row.py"),
        "--backend",
        backend,
        "--phase",
        phase,
        "--run-id",
        run_id,
        "--url",
        url,
        "--model",
        model,
        "--results-dir",
        str(results_dir),
    ]

    for field in OPTIONAL_STRING_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            cmd.extend([f"--{field.replace('_', '-')}", value.strip()])

    for package in row.get("cuda_so_package", []) or []:
        if not isinstance(package, str) or not package.strip():
            lineno = row.get("_source_lineno", "?")
            raise SystemExit(f"row at line {lineno}: cuda_so_package entries must be strings")
        cmd.extend(["--cuda-so-package", package.strip()])

    if bool(row.get("run_gguf_logprobs_probe")):
        cmd.append("--run-gguf-logprobs-probe")

    if dry_run:
        cmd.append("--dry-run")

    return run_id, cmd


def sanitize_command(cmd: list[str], repo_root: Path) -> list[str]:
    sanitized: list[str] = []
    for item in cmd:
        if item == sys.executable:
            sanitized.append("python")
            continue
        path = Path(item)
        if path.is_absolute():
            sanitized.append(rel(path, repo_root))
        else:
            sanitized.append(item)
    return sanitized


def run_one(cmd: list[str], *, repo_root: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "command": sanitize_command(cmd, repo_root),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL Qwen row definition file")
    parser.add_argument("--campaign-id")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    campaign_id = args.campaign_id or (
        "qwen_speed_lane_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    summary: dict[str, Any] = {
        "schema": "qwen-speed-lane/v1",
        "campaign_id": campaign_id,
        "input": rel(input_path, repo_root),
        "dry_run": args.dry_run,
        "rows": [],
    }

    ok = True
    for row in load_rows(input_path):
        run_id, cmd = build_command(
            row,
            repo_root=repo_root,
            results_dir=results_dir,
            campaign_id=campaign_id,
            dry_run=args.dry_run,
        )
        record = {
            "name": row.get("name"),
            "backend": row.get("backend"),
            "phase": row.get("phase", "exploratory"),
            "model": row.get("model"),
            "run_id": run_id,
            "result": run_one(cmd, repo_root=repo_root),
        }
        summary["rows"].append(record)
        if not record["result"]["ok"]:
            ok = False
            if not args.continue_on_error:
                break

    summary["ok"] = ok
    summary_path = results_dir / f"{campaign_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
