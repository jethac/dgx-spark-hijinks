#!/usr/bin/env python3
"""Audit SGLang FP4-KV dense-cache trace summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact(path_text: str | None, *, summary_path: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = summary_path.parent.parent / path
    if candidate.exists():
        return candidate
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    findings: list[str] = []
    if not summary_path.exists():
        raise SystemExit(f"summary JSON does not exist: {summary_path}")

    summary = load_json(summary_path)
    if summary.get("schema") != "sglang-fp4-kv-dense-cache-trace-summary/v1":
        findings.append(f"unexpected schema: {summary.get('schema')!r}")

    cases = summary.get("cases")
    if not isinstance(cases, dict) or not cases:
        findings.append("summary has no cases")
        cases = {}

    case_reports: dict[str, Any] = {}
    for name, case in sorted(cases.items()):
        if not isinstance(case, dict):
            findings.append(f"{name}: case entry is not an object")
            case_reports[name] = {"ok": False}
            continue

        case_findings: list[str] = []
        if case.get("ok") is not True:
            case_findings.append(f"case ok is not true: {case.get('ok')!r}")
        if case.get("trace_compare_ok") is not True:
            case_findings.append(f"trace_compare_ok is not true: {case.get('trace_compare_ok')!r}")

        request_json = resolve_artifact(case.get("request_json"), summary_path=summary_path)
        server_log = resolve_artifact(case.get("server_log"), summary_path=summary_path)
        trace_compare = resolve_artifact(case.get("trace_compare"), summary_path=summary_path)
        for label, path in (
            ("request_json", request_json),
            ("server_log", server_log),
            ("trace_compare", trace_compare),
        ):
            if path is None:
                case_findings.append(f"{label} path missing from summary")
            elif not path.exists():
                case_findings.append(f"{label} does not exist: {path}")

        compare_obj: dict[str, Any] | None = None
        if trace_compare is not None and trace_compare.exists():
            try:
                compare_obj = load_json(trace_compare)
            except Exception as exc:  # noqa: BLE001 - report artifact parse failures.
                case_findings.append(f"trace_compare parse failed: {exc!r}")
        if compare_obj is not None:
            if compare_obj.get("ok") is not True:
                case_findings.append(f"trace_compare artifact ok is not true: {compare_obj.get('ok')!r}")
            event_counts = compare_obj.get("event_counts")
            if not isinstance(event_counts, dict):
                case_findings.append("trace_compare event_counts missing")
            else:
                for key in ("dense", "cached"):
                    value = event_counts.get(key)
                    if not isinstance(value, int) or value <= 0:
                        case_findings.append(f"trace_compare event_counts.{key} is not positive: {value!r}")
            comparisons = compare_obj.get("comparisons")
            if not isinstance(comparisons, list) or not comparisons:
                case_findings.append("trace_compare has no dense/cached comparisons")

        for finding in case_findings:
            findings.append(f"{name}: {finding}")
        case_reports[name] = {
            "ok": not case_findings,
            "request_json": str(request_json) if request_json else None,
            "server_log": str(server_log) if server_log else None,
            "trace_compare": str(trace_compare) if trace_compare else None,
            "findings": case_findings,
        }

    report = {
        "schema": "sglang-dense-cache-trace-summary-audit/v1",
        "summary_json": str(summary_path),
        "case_count": len(cases),
        "cases": case_reports,
        "ok": not findings,
        "findings": findings,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
