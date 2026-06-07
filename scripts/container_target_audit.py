#!/usr/bin/env python3
"""Audit container/image artifacts for Spark target and family/PTX evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SPARK_TOKENS = ("sm_121", "SM121", "12.1", "121-real", "compute_121")
NATIVE_FP4_TOKENS = ("sm_121a", "SM121A", "12.1a", "121a", "compute_121a")
FAMILY_OR_PTX_PATTERNS = (
    re.compile(r"12\.0\+PTX"),
    re.compile(r"sm_120\b"),
    re.compile(r"compute_120\b"),
    re.compile(r"120f\b"),
    re.compile(r"compute_120f\b"),
)


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def flatten_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            strings.extend(flatten_strings(key))
            strings.extend(flatten_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(flatten_strings(item))
    return strings


def collect_inspect_evidence(path: Path) -> dict[str, Any]:
    data = load_json(path)
    text_items = flatten_strings(data)
    evidence: list[dict[str, str]] = []
    for item in text_items:
        for token in SPARK_TOKENS:
            if token in item:
                evidence.append({"kind": "spark_reference", "text": item})
                break
        for token in NATIVE_FP4_TOKENS:
            if token in item:
                evidence.append({"kind": "native_fp4_reference", "text": item})
                break
        for pattern in FAMILY_OR_PTX_PATTERNS:
            if pattern.search(item):
                evidence.append({"kind": "family_or_ptx_reference", "text": item})
                break

    architecture = None
    image_tags: list[str] = []
    image_digests: list[str] = []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        first = data[0]
        architecture = first.get("Architecture")
        image_tags = list(first.get("RepoTags") or [])
        image_digests = list(first.get("RepoDigests") or [])

    return {
        "path": str(path),
        "exists": path.exists(),
        "architecture": architecture,
        "image_tags": image_tags,
        "image_digests": image_digests,
        "evidence": evidence,
    }


def collect_versions_evidence(path: Path) -> dict[str, Any]:
    data = load_json(path)
    arch_list = data.get("arch_list") if isinstance(data, dict) else None
    device_capability = data.get("device_capability") if isinstance(data, dict) else None
    device_name = data.get("device_name") if isinstance(data, dict) else None
    packages = data.get("packages") if isinstance(data, dict) else {}
    torch_cuda = data.get("torch_cuda") if isinstance(data, dict) else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "arch_list": arch_list,
        "device_capability": device_capability,
        "device_name": device_name,
        "multi_processor_count": data.get("multi_processor_count") if isinstance(data, dict) else None,
        "packages": packages,
        "torch_cuda": torch_cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-inspect", type=Path)
    parser.add_argument("--container-inspect", type=Path)
    parser.add_argument("--container-versions", type=Path)
    parser.add_argument("--output", default="results/container_target_audit.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inspect_records = []
    for path in (args.image_inspect, args.container_inspect):
        if path is not None:
            inspect_records.append(collect_inspect_evidence(path))
    versions = (
        collect_versions_evidence(args.container_versions)
        if args.container_versions is not None
        else None
    )

    inspect_evidence = [
        item
        for record in inspect_records
        for item in record.get("evidence", [])
    ]
    arch_list = versions.get("arch_list") if isinstance(versions, dict) else []
    has_native_spark = any(
        item.get("kind") in {"native_fp4_reference", "spark_reference"}
        and any(token in item.get("text", "") for token in ("sm_121", "compute_121", "121-real", "sm_121a", "compute_121a", "121a"))
        for item in inspect_evidence
    )
    has_family_or_ptx = any(item.get("kind") == "family_or_ptx_reference" for item in inspect_evidence)
    arch_list_has_sm120 = isinstance(arch_list, list) and "sm_120" in arch_list
    arch_list_has_sm121 = isinstance(arch_list, list) and "sm_121" in arch_list
    device_is_gb10 = (
        isinstance(versions, dict)
        and versions.get("device_name") == "NVIDIA GB10"
        and versions.get("device_capability") == [12, 1]
    )

    findings = []
    if device_is_gb10:
        findings.append("OK: runtime device evidence is NVIDIA GB10 with capability 12.1.")
    if has_native_spark or arch_list_has_sm121:
        findings.append("OK: artifact contains explicit native sm_121 target evidence.")
    if has_family_or_ptx or arch_list_has_sm120:
        findings.append("INFO: artifact contains SM120-family/PTX evidence, not native sm_121 or sm_121a proof.")
    if not (has_native_spark or arch_list_has_sm121):
        findings.append("WARN: no explicit native sm_121/sm_121a build-target evidence found.")

    report: dict[str, Any] = {
        "schema": "container-target-audit/v1",
        "inputs": {
            "image_inspect": rel(args.image_inspect, repo_root) if args.image_inspect else None,
            "container_inspect": rel(args.container_inspect, repo_root) if args.container_inspect else None,
            "container_versions": rel(args.container_versions, repo_root) if args.container_versions else None,
        },
        "inspect_records": inspect_records,
        "versions": versions,
        "summary": {
            "device_is_gb10_sm121": device_is_gb10,
            "explicit_native_sm121": bool(has_native_spark or arch_list_has_sm121),
            "family_or_ptx_evidence": bool(has_family_or_ptx or arch_list_has_sm120),
            "claim": (
                "native-sm121"
                if (has_native_spark or arch_list_has_sm121)
                else "family-or-ptx-only"
                if (has_family_or_ptx or arch_list_has_sm120)
                else "missing-target-evidence"
            ),
            "findings": findings,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
