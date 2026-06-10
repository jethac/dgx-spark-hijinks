#!/usr/bin/env python3
"""Run supplied-token loglikelihood rows against a llama.cpp /loglikelihood endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
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


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    result = json.loads(body)
    if not isinstance(result, dict):
        raise ValueError(f"{url}: expected JSON object response")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(Path(args.input))
    endpoint = args.url.rstrip("/") + args.endpoint
    report: dict[str, Any] = {
        "schema": "llamacpp-supplied-loglikelihood-endpoint/v1",
        "input": args.input,
        "endpoint": endpoint,
        "dry_run": args.dry_run,
        "cases": [],
        "ok": False,
    }

    for row in rows:
        case: dict[str, Any] = {
            "id": row["id"],
            "context": row["context"],
            "continuation": row["continuation"],
            "expected_greedy": row.get("expected_greedy"),
        }
        if args.dry_run:
            case["request"] = {
                "context": row["context"],
                "continuation": row["continuation"],
            }
        else:
            try:
                response = post_json(
                    endpoint,
                    {
                        "context": row["context"],
                        "continuation": row["continuation"],
                    },
                    args.timeout,
                )
                case.update(response)
                case.setdefault("context", row["context"])
                case.setdefault("continuation", row["continuation"])
                case.setdefault("id", row["id"])
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                case["error"] = repr(exc)
        report["cases"].append(case)

    report["ok"] = bool(report["cases"]) and all("error" not in case for case in report["cases"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--endpoint", default="/loglikelihood")
    parser.add_argument("--input", default="tasks/llamacpp_loglikelihood_smoke.jsonl")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.time()
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema": "llamacpp-supplied-loglikelihood-endpoint/v1",
            "input": args.input,
            "endpoint": args.url.rstrip("/") + args.endpoint,
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
