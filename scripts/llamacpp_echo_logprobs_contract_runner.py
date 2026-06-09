#!/usr/bin/env python3
"""Run llama.cpp echo-logprobs probes for every supplied-token smoke row."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            for key in ("id", "context", "continuation"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"{path}:{line_no}: missing non-empty string {key}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no task rows found")
    return rows


def safe_id(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum() or char in {"-", "_"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "case"


def run_command(command: list[str], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"command": command, "returncode": None, "dry_run": True}
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="unused")
    parser.add_argument("--input", default="tasks/llamacpp_loglikelihood_smoke.jsonl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    max_tokens_values = args.max_tokens if args.max_tokens is not None else [0, 1]

    rows = read_jsonl(input_path)
    probe_paths: list[Path] = []
    probe_runs: list[dict[str, Any]] = []
    for row in rows:
        case_name = safe_id(row["id"])
        for max_tokens in max_tokens_values:
            output = results_dir / f"{args.run_id}_{case_name}_max{max_tokens}.json"
            probe_paths.append(output)
            command = [
                sys.executable,
                str(repo_root / "scripts" / "gguf_logprobs_probe.py"),
                "--url",
                args.url,
                "--model",
                args.model,
                "--context",
                row["context"],
                "--continuation",
                row["continuation"],
                "--max-tokens",
                str(max_tokens),
                "--timeout",
                str(args.timeout),
                "--output",
                str(output),
            ]
            probe_runs.append(
                {
                    "case": row["id"],
                    "max_tokens": max_tokens,
                    "output": str(output),
                    **run_command(command, dry_run=args.dry_run),
                }
            )

    contract_artifact = results_dir / f"{args.run_id}_contract_artifact.json"
    contract_audit = results_dir / f"{args.run_id}_contract_audit.json"
    bridge_command = [
        sys.executable,
        str(repo_root / "scripts" / "llamacpp_echo_logprobs_to_contract.py"),
        "--input",
        str(input_path),
    ]
    for probe_path in probe_paths:
        bridge_command.extend(["--probe", str(probe_path)])
    bridge_command.extend(["--output", str(contract_artifact)])

    audit_command = [
        sys.executable,
        str(repo_root / "scripts" / "llamacpp_loglikelihood_contract_audit.py"),
        "--artifact",
        str(contract_artifact),
        "--input",
        str(input_path),
        "--output",
        str(contract_audit),
    ]
    bridge_run = run_command(bridge_command, dry_run=args.dry_run)
    audit_run = run_command(audit_command, dry_run=args.dry_run)

    manifest = {
        "schema": "llamacpp-echo-logprobs-contract-run/v1",
        "run_id": args.run_id,
        "url": args.url,
        "model": args.model,
        "input": str(input_path),
        "max_tokens_values": max_tokens_values,
        "probe_count": len(probe_runs),
        "probe_outputs": [str(path) for path in probe_paths],
        "contract_artifact": str(contract_artifact),
        "contract_audit": str(contract_audit),
        "probes": probe_runs,
        "bridge": bridge_run,
        "audit": audit_run,
        "ok": (audit_run.get("returncode") == 0) if not args.dry_run else None,
        "dry_run": args.dry_run,
    }
    manifest_path = results_dir / f"{args.run_id}_echo_probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.dry_run:
        return 0
    return 0 if audit_run.get("returncode") == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
