#!/usr/bin/env python3
"""Audit FlashInfer paged-prefill debug logs for the Gemma NVFP4-KV probe.

The FlashInfer debug patch prints host-side identity and tensor lines when
FLASHINFER_PREFILL_DEBUG_ONCE=1 is enabled. This script checks that paged-prefill evidence
is bound by shared call_id/module_uri/module_key/path fields so the next live GB10 run does
not rely on manual log inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


IDENTITY_RE = re.compile(
    r"\[flashinfer\]\[prefill-debug\]\s+"
    r"(?:call_id=(?P<call_id>\d+)\s+)?"
    r"(?:module_uri=(?P<module_uri>.*?)\s+module_key=)?"
    r"(?:(?P<module_key>.*?)\s+path=)?"
    r"(?P<path>\w+)\s+"
    r"compiled=\{(?P<compiled>[^}]*)\}\s+runtime=\{(?P<runtime>[^}]*)\}"
)
TENSOR_RE = re.compile(r"\[flashinfer\]\[prefill-debug\] tensors (?P<body>.*)")
CALL_ID_RE = re.compile(r"(?:^|\s)call_id=(?P<call_id>\d+)(?:\s|$)")
MODULE_URI_RE = re.compile(r"(?:^|\s)module_uri=(?P<module_uri>.*?)\s+module_key=")
MODULE_KEY_RE = re.compile(r"(?:^|\s)module_key=(?P<module_key>.*?)\s+path=")
PATH_RE = re.compile(r"(?:^|\s)path=(?P<path>\w+)(?:\s|$)")
PAIR_RE = re.compile(r"(?P<key>\w+)=(?P<value>.*?)(?=,\w+=|$)")
TENSOR_VIEW_RE = re.compile(
    r"(?P<name>\w+)=\{ptr=(?P<ptr>[^,]+),device=(?P<device>-?\d+),"
    r"ndim=(?P<ndim>\d+),shape=\[(?P<shape>[^\]]*)\],"
    r"stride=\[(?P<stride>[^\]]*)\],_dtype=\{code=(?P<dtype_code>\d+),"
    r"bits=(?P<dtype_bits>\d+),lanes=(?P<dtype_lanes>\d+)\}\s*\}"
)
NULL_TENSOR_RE = re.compile(r"(?P<name>\w+)=null")


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


def parse_int_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_tensor_views(raw: str) -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    for match in TENSOR_VIEW_RE.finditer(raw):
        name = match.group("name")
        tensors[name] = {
            "ptr": match.group("ptr"),
            "device": int(match.group("device")),
            "ndim": int(match.group("ndim")),
            "shape": parse_int_list(match.group("shape")),
            "stride": parse_int_list(match.group("stride")),
            "dtype": {
                "code": int(match.group("dtype_code")),
                "bits": int(match.group("dtype_bits")),
                "lanes": int(match.group("dtype_lanes")),
            },
        }
    for match in NULL_TENSOR_RE.finditer(raw):
        tensors.setdefault(match.group("name"), {"is_null": True})
    return tensors


def parse_log(path: Path) -> dict[str, list[dict[str, Any]]]:
    identities: list[dict[str, Any]] = []
    tensor_lines: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        identity_match = IDENTITY_RE.search(line)
        if identity_match:
            identities.append(
                {
                    "line": lineno,
                    "call_id": parse_int(identity_match.group("call_id")),
                    "module_uri": identity_match.group("module_uri"),
                    "module_key": identity_match.group("module_key"),
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
            call_id_match = CALL_ID_RE.search(body)
            module_uri_match = MODULE_URI_RE.search(body)
            module_key_match = MODULE_KEY_RE.search(body)
            path_match = PATH_RE.search(body)
            tensors = parse_tensor_views(body)
            tensor_lines.append(
                {
                    "line": lineno,
                    "call_id": parse_int(call_id_match.group("call_id")) if call_id_match else None,
                    "module_uri": module_uri_match.group("module_uri") if module_uri_match else None,
                    "module_key": module_key_match.group("module_key") if module_key_match else None,
                    "path": path_match.group("path") if path_match else None,
                    "tensors": tensors,
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


def check_tensor(
    tensor: dict[str, Any] | None,
    *,
    name: str,
    expected_ndim: int | None = None,
    expected_dtype_bits: int | None = None,
    expected_dtype_lanes: int | None = None,
) -> list[str]:
    findings: list[str] = []
    if tensor is None:
        return [f"paged tensor dump missing {name}"]
    if tensor.get("is_null"):
        return [f"paged tensor dump has {name}=null"]
    ptr = str(tensor.get("ptr", ""))
    if ptr in ("", "0", "0x0", "nullptr", "NULL", "null"):
        findings.append(f"{name} ptr is null or empty: {ptr!r}")
    if parse_int(tensor.get("device")) is None:
        findings.append(f"{name} device is not an integer: {tensor.get('device')!r}")
    if expected_ndim is not None and tensor.get("ndim") != expected_ndim:
        findings.append(f"{name} ndim={tensor.get('ndim')}, expected {expected_ndim}")
    dtype = tensor.get("dtype") if isinstance(tensor.get("dtype"), dict) else {}
    if expected_dtype_bits is not None and dtype.get("bits") != expected_dtype_bits:
        findings.append(f"{name} dtype bits={dtype.get('bits')}, expected {expected_dtype_bits}")
    if expected_dtype_lanes is not None and dtype.get("lanes") != expected_dtype_lanes:
        findings.append(f"{name} dtype lanes={dtype.get('lanes')}, expected {expected_dtype_lanes}")
    return findings


def check_paged_identity(
    identity: dict[str, Any],
    *,
    require_bound_call: bool,
    expected_head_dim: int | None,
    expected_kv_heads: int | None,
    expected_q_heads: int | None,
    expected_page_size: int | None,
) -> list[str]:
    findings: list[str] = []
    compiled = identity["compiled"]
    runtime = identity["runtime"]
    if require_bound_call:
        if identity.get("call_id") is None:
            findings.append("paged identity missing call_id")
        if not identity.get("module_uri"):
            findings.append("paged identity missing module_uri")
        if not identity.get("module_key"):
            findings.append("paged identity missing module_key")
    if compiled.get("require_fp4_kv") != "1":
        findings.append("paged identity require_fp4_kv is not 1")
    if compiled.get("is_kv_fp4x2") != "1":
        findings.append("paged identity is_kv_fp4x2 is not 1")
    if compiled.get("dtype_kv") != "__nv_fp4x2_e2m1":
        findings.append(f"paged identity dtype_kv is not __nv_fp4x2_e2m1: {compiled.get('dtype_kv')!r}")
    if parse_int(compiled.get("sizeof_kv")) != 1:
        findings.append(f"paged identity sizeof_kv={compiled.get('sizeof_kv')!r}, expected 1")
    for name in ("maybe_k_cache_sf", "maybe_v_cache_sf"):
        if not contains_csv_value(compiled.get("additional_tensors"), name):
            findings.append(f"paged identity missing additional tensor {name}")
    for dtype in ("float_e4m3_t", "uint8_t"):
        if dtype in str(compiled.get("additional_tensor_dtypes", "")):
            break
    else:
        findings.append(
            "paged identity additional_tensor_dtypes does not show FP8 scale buffer dtype"
        )
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


def check_paged_tensor_line(
    tensor_line: dict[str, Any],
    *,
    identity: dict[str, Any] | None,
    expected_head_dim: int | None,
    expected_kv_heads: int | None,
    expected_q_heads: int | None,
    expected_page_size: int | None,
) -> list[str]:
    findings: list[str] = []
    tensors = tensor_line.get("tensors") if isinstance(tensor_line.get("tensors"), dict) else {}
    required = (
        "float_workspace",
        "int_workspace",
        "q",
        "paged_k_cache",
        "paged_v_cache",
        "maybe_k_cache_sf",
        "maybe_v_cache_sf",
        "qo_indptr",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "o",
    )
    for name in required:
        findings.extend(check_tensor(tensors.get(name), name=name))

    q = tensors.get("q")
    o = tensors.get("o")
    for name, tensor in (("q", q), ("o", o)):
        findings.extend(check_tensor(tensor, name=name, expected_ndim=3))
        if isinstance(tensor, dict) and not tensor.get("is_null"):
            shape = tensor.get("shape") if isinstance(tensor.get("shape"), list) else []
            if expected_q_heads is not None and len(shape) >= 2 and shape[1] != expected_q_heads:
                findings.append(f"{name} shape[1]={shape[1]}, expected q_heads {expected_q_heads}")
            if expected_head_dim is not None and len(shape) >= 3 and shape[2] != expected_head_dim:
                findings.append(f"{name} shape[2]={shape[2]}, expected head_dim {expected_head_dim}")

    layout = parse_int((identity or {}).get("runtime", {}).get("layout"))
    runtime_page_size = parse_int((identity or {}).get("runtime", {}).get("page_size"))
    runtime_kv_heads = parse_int((identity or {}).get("runtime", {}).get("num_kv_heads"))
    page_size = expected_page_size if expected_page_size is not None else runtime_page_size
    kv_heads = expected_kv_heads if expected_kv_heads is not None else runtime_kv_heads

    for name in ("paged_k_cache", "paged_v_cache"):
        tensor = tensors.get(name)
        findings.extend(
            check_tensor(
                tensor,
                name=name,
                expected_dtype_bits=8,
                expected_dtype_lanes=1,
            )
        )
        if not isinstance(tensor, dict) or tensor.get("is_null"):
            continue
        shape = tensor.get("shape") if isinstance(tensor.get("shape"), list) else []
        if len(shape) < 4:
            findings.append(f"{name} shape rank too small for paged KV: {shape}")
            continue
        if expected_head_dim is not None and shape[-1] != expected_head_dim // 2:
            findings.append(
                f"{name} FP4x2 carrier last dim shape[-1]={shape[-1]}, expected head_dim/2={expected_head_dim // 2}"
            )
        if layout == 0:
            if page_size is not None and shape[1] != page_size:
                findings.append(f"{name} NHD page dimension shape[1]={shape[1]}, expected {page_size}")
            if kv_heads is not None and shape[2] != kv_heads:
                findings.append(f"{name} NHD kv-head dimension shape[2]={shape[2]}, expected {kv_heads}")
        elif layout == 1:
            if kv_heads is not None and shape[1] != kv_heads:
                findings.append(f"{name} HND kv-head dimension shape[1]={shape[1]}, expected {kv_heads}")
            if page_size is not None and shape[2] != page_size:
                findings.append(f"{name} HND page dimension shape[2]={shape[2]}, expected {page_size}")
        elif layout is None:
            findings.append(f"{name} cannot validate layout-specific dimensions; runtime layout missing")
        else:
            findings.append(f"{name} runtime layout is unexpected: {layout}")

    for name in ("maybe_k_cache_sf", "maybe_v_cache_sf"):
        findings.extend(
            check_tensor(
                tensors.get(name),
                name=name,
                expected_dtype_bits=8,
                expected_dtype_lanes=1,
            )
        )

    for name in ("qo_indptr", "paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"):
        tensor = tensors.get(name)
        if tensor is not None:
            findings.extend(check_tensor(tensor, name=name, expected_ndim=1))

    device_names = (
        "q",
        "paged_k_cache",
        "paged_v_cache",
        "maybe_k_cache_sf",
        "maybe_v_cache_sf",
        "o",
    )
    devices = {
        name: tensors[name].get("device")
        for name in device_names
        if isinstance(tensors.get(name), dict) and not tensors[name].get("is_null")
    }
    unique_devices = {device for device in devices.values() if isinstance(device, int)}
    if len(unique_devices) > 1:
        findings.append(f"paged tensors are not on one device: {devices}")
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
    parser.add_argument("--require-bound-call", action="store_true", default=True)
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
    paged_groups: list[dict[str, Any]] = []
    for identity in paged:
        matching_tensor_lines = [
            line
            for line in paged_tensor_lines
            if line.get("call_id") == identity.get("call_id")
            and line.get("module_uri") == identity.get("module_uri")
            and line.get("module_key") == identity.get("module_key")
            and line.get("path") == identity.get("path")
        ]
        paged_groups.append(
            {
                "call_id": identity.get("call_id"),
                "module_uri": identity.get("module_uri"),
                "module_key": identity.get("module_key"),
                "path": identity.get("path"),
                "identity_line": identity.get("line"),
                "tensor_lines": [line.get("line") for line in matching_tensor_lines],
            }
        )

    if args.require_paged and not paged:
        findings.append("no paged prefill debug identity line found")
    if args.require_tensor_dump and not paged_tensor_lines:
        findings.append("no paged tensor dump line found")
    if args.require_bound_call and paged:
        if any(
            identity.get("call_id") is None
            or not identity.get("module_uri")
            or not identity.get("module_key")
            or identity.get("path") != "paged"
            for identity in paged
        ):
            findings.append("one or more paged identity lines are missing call_id/module_uri/module_key/path")
        if any(
            line.get("call_id") is None
            or not line.get("module_uri")
            or not line.get("module_key")
            or line.get("path") != "paged"
            for line in paged_tensor_lines
        ):
            findings.append("one or more paged tensor dump lines are missing call_id/module_uri/module_key/path")
        if not any(group["tensor_lines"] for group in paged_groups):
            findings.append(
                "no paged identity line has a tensor dump with matching path/call_id/module_uri/module_key"
            )
    for identity in paged:
        findings.extend(
            check_paged_identity(
                identity,
                require_bound_call=args.require_bound_call,
                expected_head_dim=args.expected_head_dim,
                expected_kv_heads=args.expected_kv_heads,
                expected_q_heads=args.expected_q_heads,
                expected_page_size=args.expected_page_size,
            )
        )
    identities_by_bound_key = {
        (identity.get("path"), identity.get("call_id"), identity.get("module_uri"), identity.get("module_key")): identity
        for identity in paged
        if identity.get("call_id") is not None and identity.get("module_uri") is not None and identity.get("module_key")
    }
    representative_identity = paged[0] if paged else None
    paged_tensor_checks: list[dict[str, Any]] = []
    tensor_lines_to_check = paged_tensor_lines
    if args.require_bound_call and paged:
        bound_keys = {
            (identity.get("path"), identity.get("call_id"), identity.get("module_uri"), identity.get("module_key"))
            for identity in paged
            if identity.get("call_id") is not None and identity.get("module_uri") is not None and identity.get("module_key")
        }
        tensor_lines_to_check = [
            line
            for line in paged_tensor_lines
            if (line.get("path"), line.get("call_id"), line.get("module_uri"), line.get("module_key")) in bound_keys
        ]
    for tensor_line in tensor_lines_to_check:
        matching_identity = identities_by_bound_key.get(
            (tensor_line.get("path"), tensor_line.get("call_id"), tensor_line.get("module_uri"), tensor_line.get("module_key")),
            representative_identity,
        )
        line_findings = check_paged_tensor_line(
            tensor_line,
            identity=matching_identity,
            expected_head_dim=args.expected_head_dim,
            expected_kv_heads=args.expected_kv_heads,
            expected_q_heads=args.expected_q_heads,
            expected_page_size=args.expected_page_size,
        )
        findings.extend(f"tensor line {tensor_line['line']}: {finding}" for finding in line_findings)
        paged_tensor_checks.append(
            {
                "line": tensor_line["line"],
                "ok": not line_findings,
                "findings": line_findings,
                "tensors": tensor_line.get("tensors", {}),
            }
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
            "require_bound_call": args.require_bound_call,
        },
        "counts": {
            "identity_lines": len(identities),
            "paged_identity_lines": len(paged),
            "ragged_identity_lines": len(ragged),
            "tensor_lines": len(tensor_lines),
            "paged_tensor_lines": len(paged_tensor_lines),
        },
        "paged_identities": paged,
        "paged_bound_groups": paged_groups,
        "ragged_identities": ragged,
        "paged_tensor_lines": paged_tensor_lines,
        "paged_tensor_checks": paged_tensor_checks,
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
