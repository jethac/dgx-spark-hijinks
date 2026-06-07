#!/usr/bin/env python3
"""Audit CUDA build or JIT logs for Spark target evidence.

This complements cuda_so_audit.py. Use this script before or beside binary
inspection when the relevant evidence lives in build logs, JIT logs, CMake
output, or server logs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


PATTERNS = {
    "spark_target": [
        re.compile(r"\bsm_121\b"),
        re.compile(r"\bcompute_121\b"),
        re.compile(r"\b121-real\b"),
        re.compile(r"\bCMAKE_CUDA_ARCHITECTURES\b[^\n\r]*(?:\b121\b|\b12\.1\b)"),
        re.compile(r"\bTORCH_CUDA_ARCH_LIST\b[^\n\r]*(?:\b12\.1\b|\bsm_121\b)"),
        re.compile(r"\bCUDA\s*:.*\bARCHS\s*=\s*1210\b"),
    ],
    "native_fp4_target": [
        re.compile(r"\bsm_121a\b"),
        re.compile(r"\bcompute_121a\b"),
        re.compile(r"\b121a(?:-real)?\b"),
        re.compile(r"\b12\.1a\b"),
    ],
    "family_target": [
        re.compile(r"\bsm_120f\b"),
        re.compile(r"\bcompute_120f\b"),
        re.compile(r"\b120f(?:-real)?\b"),
        re.compile(r"\b12\.0f\b"),
    ],
    "adjacent_or_legacy_target": [
        re.compile(r"\bsm_(?:70|75|80|86|87|89|90|90a|100|100a|103|103a|110|120|120a)\b"),
        re.compile(r"\bcompute_(?:70|75|80|86|87|89|90|90a|100|100a|103|103a|110|120|120a)\b"),
    ],
}


def _match_line(line: str) -> list[dict[str, str]]:
    hits = []
    for kind, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(line):
                hits.append({"kind": kind, "token": match.group(0)})
    return hits


def audit_log(path: Path, allow_family_targets: set[str]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    evidence = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for hit in _match_line(line):
            evidence.append(
                {
                    "line": line[:500],
                    "line_no": line_no,
                    **hit,
                }
            )

    kinds = {item["kind"] for item in evidence}
    family_tokens = {
        item["token"]
        for item in evidence
        if item["kind"] == "family_target"
    }
    allowed_family = sorted(family_tokens & allow_family_targets)
    unallowed_family = sorted(family_tokens - allow_family_targets)
    has_spark = "spark_target" in kinds
    has_native_fp4 = "native_fp4_target" in kinds
    has_allowed_family = bool(allowed_family)
    has_adjacent = "adjacent_or_legacy_target" in kinds

    findings = []
    if has_native_fp4:
        findings.append("OK: log contains native Spark FP4 target evidence.")
    if has_spark:
        findings.append("OK: log contains Spark sm_121 target evidence.")
    if has_allowed_family:
        findings.append(
            "OK: log contains explicitly allowed SM12x family target evidence: "
            + ", ".join(allowed_family)
        )
    if unallowed_family:
        findings.append(
            "WARN: log contains SM12x family targets that were not explicitly allowed: "
            + ", ".join(unallowed_family)
        )
    if has_adjacent and not (has_spark or has_native_fp4 or has_allowed_family):
        findings.append(
            "WARN: log contains adjacent or legacy CUDA targets but no accepted Spark target evidence."
        )
    if not evidence:
        findings.append("WARN: no CUDA architecture target evidence found in log.")

    accepted = has_spark or has_native_fp4 or has_allowed_family
    return {
        "path": str(path),
        "exists": path.exists(),
        "accepted_spark_target_evidence": accepted,
        "has_spark_target": has_spark,
        "has_native_fp4_target": has_native_fp4,
        "allowed_family_targets": allowed_family,
        "unallowed_family_targets": unallowed_family,
        "evidence": evidence,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", default=[], type=Path)
    parser.add_argument(
        "--allow-family-target",
        action="append",
        default=[],
        help="Accept a family-compatible target token such as 120f or compute_120f.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit 1 if any log lacks accepted Spark target evidence.",
    )
    args = parser.parse_args()

    allow_family_targets = set(args.allow_family_target)
    logs = []
    for path in args.log:
        item: dict[str, Any]
        if not path.exists():
            item = {
                "path": str(path),
                "exists": False,
                "accepted_spark_target_evidence": False,
                "findings": ["ERROR: log path does not exist."],
                "evidence": [],
            }
        else:
            item = audit_log(path, allow_family_targets)
        logs.append(item)

    report = {
        "schema": "cuda-build-target-audit/v1",
        "created_unix": time.time(),
        "inputs": {
            "logs": [str(path) for path in args.log],
            "allow_family_targets": sorted(allow_family_targets),
        },
        "logs": logs,
        "summary": {
            "log_count": len(logs),
            "accepted_log_count": sum(
                1 for item in logs if item.get("accepted_spark_target_evidence")
            ),
            "native_fp4_log_count": sum(
                1 for item in logs if item.get("has_native_fp4_target")
            ),
            "missing_or_unaccepted_count": sum(
                1 for item in logs if not item.get("accepted_spark_target_evidence")
            ),
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.fail_on_missing and report["summary"]["missing_or_unaccepted_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
