#!/usr/bin/env python3
"""Run a tiny GGUF loglikelihood task through llama.cpp native endpoints."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from pathlib import Path
from typing import Any

from llamacpp_native_loglikelihood_probe import score_case


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            for key in ("id", "context", "continuation"):
                if not isinstance(row.get(key), str):
                    raise ValueError(f"{path}:{line_no}: {key!r} must be a string")
            expected = row.get("expected_greedy")
            if expected is not None and not isinstance(expected, bool):
                raise ValueError(
                    f"{path}:{line_no}: 'expected_greedy' must be a bool when present"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no task rows found")
    return rows


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    target_found = sum(1 for case in cases if case.get("target_found"))
    greedy_known = sum(1 for case in cases if case.get("all_tokens_greedy") is not None)
    expected_checked = 0
    expected_mismatches = 0
    for case in cases:
        expected = case.get("expected_greedy")
        if expected is None:
            continue
        expected_checked += 1
        if case.get("all_tokens_greedy") != expected:
            expected_mismatches += 1
    return {
        "total": total,
        "target_found": target_found,
        "target_missing": total - target_found,
        "greedy_known": greedy_known,
        "expected_greedy_checked": expected_checked,
        "expected_greedy_mismatches": expected_mismatches,
        "ok": target_found == total and expected_mismatches == 0,
    }


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(Path(args.input))
    base_url = args.url.rstrip("/")
    report: dict[str, Any] = {
        "schema": "llamacpp-native-loglikelihood-task/v1",
        "input": args.input,
        "tokenize_endpoint": base_url + "/tokenize",
        "completion_endpoint": base_url + "/completion",
        "n_probs": args.n_probs,
        "dry_run": args.dry_run,
        "cases": [],
        "ok": False,
    }

    if args.dry_run:
        report["cases"] = [
            {
                "id": row["id"],
                "context": row["context"],
                "continuation": row["continuation"],
                "expected_greedy": row.get("expected_greedy"),
            }
            for row in rows
        ]
        report["summary"] = {
            "total": len(rows),
            "target_found": 0,
            "target_missing": None,
            "greedy_known": 0,
            "expected_greedy_checked": sum(
                1 for row in rows if row.get("expected_greedy") is not None
            ),
            "expected_greedy_mismatches": None,
            "ok": True,
        }
        report["ok"] = True
        return report

    for row in rows:
        case = score_case(
            row["id"],
            row["context"],
            row["continuation"],
            report["tokenize_endpoint"],
            report["completion_endpoint"],
            args.n_probs,
            args.timeout,
        )
        case["expected_greedy"] = row.get("expected_greedy")
        report["cases"].append(case)

    report["summary"] = summarize(report["cases"])
    report["ok"] = bool(report["summary"]["ok"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--n-probs", type=int, default=512)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.time()
    try:
        report = run_task(args)
    except (ValueError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        report = {
            "schema": "llamacpp-native-loglikelihood-task/v1",
            "input": args.input,
            "ok": False,
            "error": repr(exc),
        }
    report["elapsed_s"] = round(time.time() - started, 3)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
