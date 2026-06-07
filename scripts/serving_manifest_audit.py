#!/usr/bin/env python3
"""Audit OpenAI-compatible serving row manifests for claim-ready evidence."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_METADATA = (
    "quantization",
    "kv_cache_dtype",
    "attention_backend",
    "cuda_graph_mode",
)

BACKEND_LOG_MARKERS = {
    "vllm": (
        "FlashInfer",
        "TRITON_ATTN",
        "FLASH_ATTN",
        "VLLM_CUTLASS",
        "CUDAGraph",
        "CUDA graph",
        "DFlash",
        "NvFp4",
        "NVFP4",
    ),
    "sglang": (
        "attention backend",
        "flashinfer",
        "cuda graph",
        "KV cache",
        "KV Cache",
        "fp8",
        "fp4",
    ),
    "llamacpp": (
        "CUDA : ARCHS",
        "USE_GRAPHS",
        "ggml_cuda",
        "CUDA0",
        "BLACKWELL_NATIVE_FP4",
    ),
}


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - audit records parse failures.
        return None, repr(exc)
    if not isinstance(data, dict):
        return None, "top-level JSON is not an object"
    return data, None


def artifact_path(root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return root / path


def read_artifact_json(root: Path, raw: str | None) -> tuple[dict[str, Any] | None, str | None]:
    path = artifact_path(root, raw)
    if path is None:
        return None, "artifact missing from manifest"
    if not path.exists():
        return None, f"artifact does not exist: {path}"
    return load_json(path)


def detect_model_family(record: dict[str, Any], path: Path) -> str:
    text = " ".join(
        str(record.get(key, "")) for key in ("run_id", "model", "served_model", "backend")
    )
    text += " " + path.name
    lower = text.lower()
    if "qwen" in lower:
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    return "unknown"


def hardware_ok_from_blob(blob: dict[str, Any] | None) -> bool:
    if not blob:
        return False
    hardware = blob.get("hardware")
    if isinstance(hardware, dict):
        for device in hardware.get("devices") or []:
            if not isinstance(device, dict):
                continue
            capability = device.get("capability")
            name = str(device.get("name", ""))
            comparison_key = str(device.get("comparison_key", ""))
            if capability == [12, 1] and "GB10" in name:
                return True
            if "sm_121" in comparison_key and "GB10" in comparison_key:
                return True
    commands = blob.get("commands")
    if isinstance(commands, dict):
        smi = commands.get("nvidia_smi_query")
        if isinstance(smi, dict):
            stdout = str(smi.get("stdout", ""))
            if "NVIDIA GB10" in stdout and "12.1" in stdout:
                return True
    if str(blob.get("hardware_key", "")).startswith("NVIDIA_GB10:sm_121"):
        return True
    return False


def benchmark_ok(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    data, error = read_artifact_json(root, artifacts.get("openai_benchmark"))
    record: dict[str, Any] = {"exists": data is not None, "ok": False}
    if error:
        record["error"] = error
        return record
    cases = data.get("cases") if isinstance(data, dict) else []
    record.update(
        {
            "ok": bool(data.get("ok")) and bool(cases),
            "case_count": len(cases) if isinstance(cases, list) else 0,
            "hardware_ok": hardware_ok_from_blob(data),
        }
    )
    return record


def runtime_probe_ok(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    data, error = read_artifact_json(root, artifacts.get("runtime_probe"))
    record: dict[str, Any] = {"exists": data is not None, "hardware_ok": False, "process_count": 0}
    if error:
        record["error"] = error
        return record
    processes = data.get("processes") if isinstance(data, dict) else []
    record.update(
        {
            "hardware_ok": hardware_ok_from_blob(data),
            "process_count": len(processes) if isinstance(processes, list) else 0,
            "server_models_ok": bool((data.get("server") or {}).get("models")),
        }
    )
    return record


def build_target_ok(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    data, error = read_artifact_json(root, artifacts.get("build_target_audit"))
    record: dict[str, Any] = {"exists": data is not None, "accepted": False}
    if error:
        record["error"] = error
        return record
    summary = data.get("summary") if isinstance(data, dict) else {}
    record.update(
        {
            "accepted": int(summary.get("accepted_log_count") or 0) > 0,
            "native_fp4_log_count": int(summary.get("native_fp4_log_count") or 0),
            "missing_or_unaccepted_count": int(summary.get("missing_or_unaccepted_count") or 0),
        }
    )
    return record


def server_log_markers(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    path = artifact_path(root, artifacts.get("server_log"))
    backend = str(manifest.get("backend", "")).lower()
    markers = BACKEND_LOG_MARKERS.get(backend, ())
    record: dict[str, Any] = {"exists": False, "matched_markers": [], "path": artifacts.get("server_log")}
    if path is None:
        record["error"] = "server_log artifact missing from manifest"
        return record
    if not path.exists():
        record["error"] = f"server log does not exist: {path}"
        return record
    text = path.read_text(encoding="utf-8", errors="replace")
    matched = [marker for marker in markers if marker in text]
    record.update(
        {
            "exists": True,
            "matched_markers": matched,
            "has_backend_marker": bool(matched),
        }
    )
    return record


def command_ok(manifest: dict[str, Any], command_name: str) -> bool:
    commands = manifest.get("commands")
    if not isinstance(commands, dict):
        return False
    command = commands.get(command_name)
    return isinstance(command, dict) and bool(command.get("ok"))


def audit_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest, error = load_json(path)
    record: dict[str, Any] = {
        "path": rel(path, root),
        "parse_ok": manifest is not None,
        "claim_ready": False,
        "findings": [],
    }
    if error:
        record["error"] = error
        record["findings"].append("manifest is not valid JSON; row cannot be machine-audited")
        return record
    assert manifest is not None

    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    missing_metadata = [
        field for field in REQUIRED_METADATA if not isinstance(metadata.get(field), str) or not metadata.get(field)
    ]
    benchmark = benchmark_ok(root, manifest)
    runtime = runtime_probe_ok(root, manifest)
    build_target = build_target_ok(root, manifest)
    server_log = server_log_markers(root, manifest)

    record.update(
        {
            "schema": manifest.get("schema"),
            "run_id": manifest.get("run_id"),
            "backend": manifest.get("backend"),
            "model": manifest.get("model"),
            "model_family": detect_model_family(manifest, path),
            "phase": manifest.get("phase"),
            "dry_run": bool(manifest.get("dry_run")),
            "manifest_ok": bool(manifest.get("ok")),
            "missing_metadata": missing_metadata,
            "command_checks": {
                "chat_smoke": command_ok(manifest, "chat_smoke"),
                "openai_benchmark": command_ok(manifest, "openai_benchmark"),
                "runtime_probe": command_ok(manifest, "runtime_probe"),
                "build_target_audit": command_ok(manifest, "build_target_audit"),
            },
            "benchmark": benchmark,
            "runtime_probe": runtime,
            "build_target_audit": build_target,
            "server_log": server_log,
        }
    )

    if record["dry_run"]:
        record["findings"].append("dry-run manifest is planning evidence only")
    if missing_metadata:
        record["findings"].append("missing required runtime metadata: " + ", ".join(missing_metadata))
    if not benchmark.get("ok"):
        record["findings"].append("OpenAI benchmark artifact is missing or not fully ok")
    if not (benchmark.get("hardware_ok") or runtime.get("hardware_ok")):
        record["findings"].append("GB10 sm_121 hardware evidence is missing from benchmark/runtime artifacts")
    if not server_log.get("has_backend_marker"):
        record["findings"].append("server log has no backend marker from the audit marker set")
    if not build_target.get("accepted"):
        record["findings"].append("build-target audit lacks accepted Spark target evidence")

    record["claim_ready"] = (
        not record["dry_run"]
        and bool(record["manifest_ok"])
        and not missing_metadata
        and command_ok(manifest, "chat_smoke")
        and command_ok(manifest, "openai_benchmark")
        and bool(benchmark.get("ok"))
        and bool(benchmark.get("hardware_ok") or runtime.get("hardware_ok"))
        and bool(server_log.get("has_backend_marker"))
        and bool(build_target.get("accepted"))
    )
    return record


def expand_inputs(patterns: list[str], root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = glob.glob(str(root / pattern)) if not Path(pattern).is_absolute() else glob.glob(pattern)
        if not expanded:
            expanded = [str(root / pattern)] if not Path(pattern).is_absolute() else [pattern]
        paths.extend(Path(item) for item in expanded)
    unique = sorted({path.resolve() for path in paths})
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Manifest path or glob relative to repo root. Defaults to results/*row_manifest.json.",
    )
    parser.add_argument("--output", default="results/serving_manifest_audit.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any live row is not claim-ready.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    patterns = args.manifest or ["results/*row_manifest.json"]
    rows = [audit_manifest(path, root) for path in expand_inputs(patterns, root)]

    live_rows = [row for row in rows if row.get("parse_ok") and not row.get("dry_run")]
    claim_ready_rows = [row for row in live_rows if row.get("claim_ready")]
    model_families = sorted(
        {
            row.get("model_family")
            for row in claim_ready_rows
            if isinstance(row.get("model_family"), str) and row.get("model_family") != "unknown"
        }
    )
    summary: dict[str, Any] = {
        "schema": "serving-manifest-audit/v1",
        "inputs": [rel(path, root) for path in expand_inputs(patterns, root)],
        "row_count": len(rows),
        "live_row_count": len(live_rows),
        "claim_ready_count": len(claim_ready_rows),
        "claim_ready_model_families": model_families,
        "broad_runtime_claim_ready": {"qwen": "qwen" in model_families, "gemma": "gemma" in model_families},
        "rows": rows,
    }
    summary["ok"] = bool(rows)
    summary["strict_ok"] = bool(live_rows) and all(row.get("claim_ready") for row in live_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict and not summary["strict_ok"]:
        return 1
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
