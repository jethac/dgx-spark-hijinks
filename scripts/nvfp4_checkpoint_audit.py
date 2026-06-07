#!/usr/bin/env python3
"""Audit NVFP4 checkpoint metadata before Spark serving or GGUF conversion."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


COMPRESSED_TENSORS_MARKERS = (
    ".weight_packed",
    ".weight_scale",
    ".weight_global_scale",
    ".input_global_scale",
)

MODELOPT_MARKERS = (
    ".weight_scale_inverse",
    ".input_scale",
)

SENSITIVE_PATTERNS = (
    re.compile(r"(^|\.)router(\.|$)"),
    re.compile(r"router\.proj"),
    re.compile(r"vision_tower"),
    re.compile(r"embed_vision"),
    re.compile(r"visual"),
)

GEMMA_EOS_IDS = {1, 50, 98, 100, 101, 106}


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - audit should report parse failures.
        return None, repr(exc)
    if not isinstance(data, dict):
        return None, "top-level JSON is not an object"
    return data, None


def flatten_ids(value: Any) -> set[int]:
    ids: set[int] = set()
    if isinstance(value, int):
        ids.add(value)
    elif isinstance(value, list):
        for item in value:
            ids.update(flatten_ids(item))
    elif isinstance(value, dict):
        for item in value.values():
            ids.update(flatten_ids(item))
    return ids


def read_index_keys(model_dir: Path) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    errors: list[str] = []
    for path in sorted(model_dir.glob("*.safetensors.index.json")):
        data, error = load_json(path)
        if error:
            errors.append(f"{path.name}: {error}")
            continue
        weight_map = data.get("weight_map")
        if isinstance(weight_map, dict):
            keys.extend(str(key) for key in weight_map)
    return sorted(set(keys)), errors


def read_safetensors_keys(model_dir: Path) -> tuple[list[str], list[str], bool]:
    paths = sorted(model_dir.glob("*.safetensors"))
    if not paths:
        return [], [], False
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as exc:  # noqa: BLE001 - dependency is optional.
        return [], [f"safetensors unavailable: {exc!r}"], False

    keys: list[str] = []
    errors: list[str] = []
    for path in paths:
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys.extend(handle.keys())
        except Exception as exc:  # noqa: BLE001 - record corrupt/incompatible shards.
            errors.append(f"{path.name}: {exc!r}")
    return sorted(set(keys)), errors, True


def has_any_marker(name: str, markers: tuple[str, ...]) -> bool:
    return any(marker in name for marker in markers)


def is_sensitive(name: str) -> bool:
    return any(pattern.search(name) for pattern in SENSITIVE_PATTERNS)


def base_tensor_name(name: str) -> str:
    for marker in COMPRESSED_TENSORS_MARKERS + MODELOPT_MARKERS:
        if marker in name:
            return name.split(marker, 1)[0]
    return name


def infer_format(keys: list[str], config: dict[str, Any] | None) -> dict[str, Any]:
    compressed = [key for key in keys if has_any_marker(key, COMPRESSED_TENSORS_MARKERS)]
    modelopt = [key for key in keys if has_any_marker(key, MODELOPT_MARKERS)]
    config_blob = json.dumps(config or {}, sort_keys=True).lower()
    config_markers = {
        "mentions_nvfp4": "nvfp4" in config_blob,
        "mentions_compressed_tensors": "compressed-tensors" in config_blob
        or "compressed_tensors" in config_blob,
        "mentions_modelopt": "modelopt" in config_blob,
    }

    if compressed and modelopt:
        guess = "mixed"
    elif compressed:
        guess = "compressed_tensors_nvfp4"
    elif modelopt:
        guess = "modelopt_nvfp4"
    elif config_markers["mentions_nvfp4"]:
        guess = "nvfp4_config_only"
    else:
        guess = "unknown"

    return {
        "guess": guess,
        "compressed_tensors_marker_count": len(compressed),
        "modelopt_marker_count": len(modelopt),
        "config_markers": config_markers,
        "sample_compressed_tensors_keys": compressed[:10],
        "sample_modelopt_keys": modelopt[:10],
    }


def audit_sensitive_quantization(keys: list[str]) -> dict[str, Any]:
    sensitive_keys = [key for key in keys if is_sensitive(key)]
    quantized_sensitive = [
        key
        for key in sensitive_keys
        if has_any_marker(key, COMPRESSED_TENSORS_MARKERS + MODELOPT_MARKERS)
    ]
    quantized_base_counts = Counter(base_tensor_name(key) for key in quantized_sensitive)
    return {
        "sensitive_key_count": len(sensitive_keys),
        "quantized_sensitive_key_count": len(quantized_sensitive),
        "quantized_sensitive_bases": sorted(quantized_base_counts),
        "sample_sensitive_keys": sensitive_keys[:20],
        "sample_quantized_sensitive_keys": quantized_sensitive[:20],
        "ok": not quantized_sensitive,
    }


def audit_eos(model_dir: Path, config: dict[str, Any] | None) -> dict[str, Any]:
    generation_config_path = model_dir / "generation_config.json"
    generation_config, generation_error = (
        load_json(generation_config_path) if generation_config_path.exists() else (None, None)
    )
    ids = set()
    for data in (config, generation_config):
        if isinstance(data, dict):
            ids.update(flatten_ids(data.get("eos_token_id")))
            ids.update(flatten_ids(data.get("eos_token_ids")))
    model_type = str((config or {}).get("model_type", "")).lower()
    looks_gemma = "gemma" in model_type or "gemma" in model_dir.name.lower()
    missing = sorted(GEMMA_EOS_IDS - ids) if looks_gemma else []
    return {
        "looks_gemma": looks_gemma,
        "generation_config_exists": generation_config_path.exists(),
        "generation_config_error": generation_error,
        "observed_eos_ids": sorted(ids),
        "expected_gemma_eos_ids": sorted(GEMMA_EOS_IDS) if looks_gemma else [],
        "missing_gemma_eos_ids": missing,
        "ok": not looks_gemma or not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="HF checkpoint directory to inspect.")
    parser.add_argument("--output", help="Write audit JSON to this path.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when audit is unsafe.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = model_dir / "config.json"
    config, config_error = load_json(config_path) if config_path.exists() else (None, None)
    index_keys, index_errors = read_index_keys(model_dir)
    safetensors_keys, safetensors_errors, safetensors_loaded = read_safetensors_keys(model_dir)
    keys = sorted(set(safetensors_keys or index_keys))

    format_guess = infer_format(keys, config)
    sensitive = audit_sensitive_quantization(keys)
    eos = audit_eos(model_dir, config)
    has_quant_markers = format_guess["guess"] in {
        "compressed_tensors_nvfp4",
        "modelopt_nvfp4",
        "mixed",
        "nvfp4_config_only",
    }
    key_source = "safetensors" if safetensors_keys else "index" if index_keys else "none"
    safe_for_gemma_nvfp4_serving = (
        bool(keys)
        and has_quant_markers
        and sensitive["ok"]
        and eos["ok"]
    )

    result: dict[str, Any] = {
        "schema": "nvfp4-checkpoint-audit/v1",
        "model_dir": rel(model_dir, repo_root),
        "config": {
            "exists": config_path.exists(),
            "error": config_error,
            "model_type": (config or {}).get("model_type") if isinstance(config, dict) else None,
            "architectures": (config or {}).get("architectures")
            if isinstance(config, dict)
            else None,
        },
        "key_source": key_source,
        "key_count": len(keys),
        "index_error_count": len(index_errors),
        "index_errors": index_errors,
        "safetensors_loaded": safetensors_loaded,
        "safetensors_error_count": len(safetensors_errors),
        "safetensors_errors": safetensors_errors,
        "format_guess": format_guess,
        "sensitive_quantization": sensitive,
        "eos_check": eos,
        "safe_for_gemma_nvfp4_serving": safe_for_gemma_nvfp4_serving,
        "ok": bool(keys) and sensitive["ok"] and eos["ok"],
        "warnings": [],
    }

    if not keys:
        result["warnings"].append("no checkpoint keys found from safetensors index or shards")
    if not has_quant_markers:
        result["warnings"].append("no NVFP4 quantization markers found")
    if format_guess["guess"] == "mixed":
        result["warnings"].append("both compressed-tensors and ModelOpt markers found")
    if not sensitive["ok"]:
        result["warnings"].append("router/vision/visual tensors appear quantized")
    if not eos["ok"]:
        result["warnings"].append("Gemma EOS/control-token set appears incomplete")

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")
    if args.strict and not result["safe_for_gemma_nvfp4_serving"]:
        return 1
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
