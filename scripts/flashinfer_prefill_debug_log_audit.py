#!/usr/bin/env python3
"""Audit FlashInfer paged-prefill debug logs for the Gemma NVFP4-KV probe.

The FlashInfer debug patch prints host-side identity lines when
FLASHINFER_PREFILL_DEBUG_ONCE=1 is enabled. This script turns those lines into a
machine-checkable artifact so the next live GB10 run does not rely on manual log
inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


IDENTITY_RE = re.compile(
    r"\[flashinfer\]\[prefill-debug\] path=(?P<path>\w+) "
    r"compiled=\{(?P<compiled>[^}]*)\} runtime=\{(?P<runtime>[^}]*)\}"
)
TENSOR_RE = re.compile(r"\[flashinfer\]\[prefill-debug\] tensors (?P<body>.*)")
PAIR_RE = re.compile(r"(?P<key>\w+)=(?P<value>.*?)(?=,\w+=|$)")


def parse_pairs(raw: str) -> dict[str, str]:
    return {
        match.group("key").strip(): match.group("value").strip()
        for match in PAIR_RE.finditer(raw)
    }


def parse_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def parse_log(path: Path) -> dict[str, list[dict[str, Any]]]:
    identities: list[dict[str, Any]] = []
    tensor_lines: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        identity_match = IDENTITY_RE.search(line)
        if identity_match:
            identities.append(
                {
                    "line": lineno,
                    "path": identity_match.group("path"),
                    "compiled": parse_pairs(identity_match.group("compiled")),
                    "runtime": parse_pairs(identity_match.group("runtime")),
                    "raw": line.strip(),
                }
            )
            continue
        tensor_match = TENSOR_RE.search(line)
        if tensor_match:
            body = tensor_match.group("body")
            tensor_lines.append(
                {
                    "line": lineno,
                    "has_paged_k_cache": "paged_k_cache=" in body,
                    "has_paged_v_cache": "paged_v_cache=" in body,
                    "has_ragged_k": " k=" in f" {body}",
                    "has_ragged_v": " v=" in f" {body}",
                    "has_output": " o=" in f" {body}",
                    "raw": line.strip(),
                }
            )
    return {"identities": identities, "tensor_lines": tensor_lines}


def contains_csv_value(raw: str | None, needle: str) -> bool:
    if raw is None:
        return False
    return needle in {item.strip() for item in raw.split(",")}


def check_paged_identity(
    identity: dict[str, Any],
    *,
    expected_head_dim: int | None,
    expected_kv_heads: int | None,
    expected_q_heads: int | None,
    expected_page_size: int | None,
) -> list[str]:
    findings: list[str] = []
    compiled = identity["compiled"]
    runtime = identity["runtime"]
    if compiled.get("require_fp4_kv") != "1":
        findings.append("paged identity require_fp4_kv is not 1")
    if compiled.get("is_kv_fp4x2") != "1":
        findings.append("paged identity is_kv_fp4x2 is not 1")
    if compiled.get("dtype_kv") not in {"__nv_fp4x2_e2m1", "uint8_t"}:
        findings.append(f"paged identity dtype_kv is unexpected: {compiled.get('dtype_kv')!r}")
    for name in ("maybe_k_cache_sf", "maybe_v_cache_sf"):
        if not contains_csv_value(compiled.get("additional_tensors"), name):
            findings.append(f"paged identity missing additional tensor {name}")
    for key in ("head_dim_qk", "head_dim_vo"):
        actual = parse_int(compiled.get(key))
        if expected_head_dim is not None and actual != expected_head_dim:
            findings.append(f"paged identity {key}={actual}, expected {expected_head_dim}")
    if expected_kv_heads is not None:
        actual = parse_int(runtime.get("num_kv_heads"))
        if actual != expected_kv_heads:
            findings.append(f"paged runtime num_kv_heads={actual}, expected {expected_kv_heads}")
    if expected_q_heads is not None:
        actual = parse_int(runtime.get("num_qo_heads"))
        if actual != expected_q_heads:
            findings.append(f"paged runtime num_qo_heads={actual}, expected {expected_q_heads}")
    if expected_page_size is not None:
        actual = parse_int(runtime.get("page_size"))
        if actual != expected_page_size:
            findings.append(f"paged runtime page_size={actual}, expected {expected_page_size}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-head-dim", type=int, default=128)
    parser.add_argument("--expected-kv-heads", type=int, default=16)
    parser.add_argument("--expected-q-heads", type=int, default=32)
    parser.add_argument("--expected-page-size", type=int, default=16)
    parser.add_argument("--require-paged", action="store_true", default=True)
    parser.add_argument("--require-tensor-dump", action="store_true", default=True)
    args = parser.parse_args()

    log_path = Path(args.server_log)
    findings: list[str] = []
    if not log_path.exists():
        findings.append(f"server log does not exist: {args.server_log}")
        parsed = {"identities": [], "tensor_lines": []}
    else:
        parsed = parse_log(log_path)

    identities = parsed["identities"]
    tensor_lines = parsed["tensor_lines"]
    paged = [identity for identity in identities if identity.get("path") == "paged"]
    ragged = [identity for identity in identities if identity.get("path") == "ragged"]
    paged_tensor_lines = [
        line for line in tensor_lines if line["has_paged_k_cache"] and line["has_paged_v_cache"]
    ]

    if args.require_paged and not paged:
        findings.append("no paged prefill debug identity line found")
    if args.require_tensor_dump and not paged_tensor_lines:
        findings.append("no paged tensor dump line found")
    for identity in paged:
        findings.extend(
            check_paged_identity(
                identity,
                expected_head_dim=args.expected_head_dim,
                expected_kv_heads=args.expected_kv_heads,
                expected_q_heads=args.expected_q_heads,
                expected_page_size=args.expected_page_size,
            )
        )

    report = {
        "schema": "flashinfer-prefill-debug-log-audit/v1",
        "server_log": args.server_log,
        "expectations": {
            "head_dim": args.expected_head_dim,
            "kv_heads": args.expected_kv_heads,
            "q_heads": args.expected_q_heads,
            "page_size": args.expected_page_size,
            "require_paged": args.require_paged,
            "require_tensor_dump": args.require_tensor_dump,
        },
        "counts": {
            "identity_lines": len(identities),
            "paged_identity_lines": len(paged),
            "ragged_identity_lines": len(ragged),
            "tensor_lines": len(tensor_lines),
            "paged_tensor_lines": len(paged_tensor_lines),
        },
        "paged_identities": paged,
        "ragged_identities": ragged,
        "paged_tensor_lines": paged_tensor_lines,
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
