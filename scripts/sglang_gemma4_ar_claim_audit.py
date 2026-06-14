#!/usr/bin/env python3
"""Audit whether SGLang Gemma 4 AR ladder manifests are claim-ready."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


REQUIRED_MODELS = [
    "google/gemma-4-12B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
]
REQUIRED_ROWS = ["bf16", "fp8", "fullnvfp4"]
EXPECTED_ROW_KV_DTYPES = {
    "bf16": "auto",
    "fp8": "fp8_e4m3",
    "fullnvfp4": "fp4_e2m1",
}
REQUIRED_COMPARISONS = [
    "compare_bf16_vs_fullnvfp4",
    "compare_fp8_vs_fullnvfp4",
]
REQUIRED_ROW_ARTIFACTS = [
    "{label}_summary.json",
    "{label}_ppl.json",
    "{label}_chat_1.json",
    "{label}_chat_2.json",
    "{label}_preflight.log",
    "{label}_provenance.log",
    "{label}_server.log",
    "{label}_container_inspect.json",
]
REQUIRED_PROVENANCE_MARKERS = [
    "transformers ",
    "sglang ",
    "flashinfer ",
    "flashinfer_python ",
    "sglang_kernel ",
    "binary_md5 sgl_kernel ",
    "flashinfer_data ",
    "flashinfer_csrc ",
    "flashinfer_include ",
    "source_git_rev /work/third_party/sglang ",
    "source_git_rev /work/third_party/flashinfer ",
]
REQUIRED_SERVER_MARKERS = [
    "server_args=ServerArgs",
    "attention_backend='flashinfer'",
    "SGLANG_GEMMA_KV_GEOMETRY",
    "SGLang FlashInfer VO split enabled",
]
REQUIRED_FULLNVFP4_SERVER_MARKERS = [
    "FP4 KV FlashInfer module trace",
    "deswizzle_macro_active=False",
    "fp4_kv=1",
    "k_sf=",
    "v_sf=",
]
REQUIRED_CAPACITY_FIELDS = [
    "full_tokens",
    "swa_tokens",
    "total_token_slots",
    "full_per_token_bytes",
    "swa_per_token_bytes",
    "cell_size_bytes",
]
REQUIRED_HARDWARE_COMPARISON_KEY_PREFIX = "NVIDIA_GB10:sm_121"
REQUIRED_POSITIVE_INT_FIELDS = [
    "reuse_prefix_len",
    "logprob_start_len",
    "max_new_tokens",
    "context_length",
    "page_size",
]
REQUIRED_CHAT_CONTENT_SUBSTRING = "tokyo"
REQUIRED_DOCKER_MEMORY_BYTES = 100 * 1024 * 1024 * 1024
REQUIRED_CONTAINER_ENV = {
    "TORCH_CUDA_ARCH_LIST": "12.1a",
    "FLASHINFER_PREFILL_DEBUG_ONCE": "1",
    "SGLANG_FLASHINFER_VOSPLIT": "1",
    "SGLANG_GEMMA4_TRACE_GEOMETRY": "1",
    "SGLANG_GEMMA_KV_GEOMETRY": "1",
    "SGLANG_FP4_KV_MIXED_KV": "0",
    "SGLANG_FP4_KV_TRACE_MODULE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
}
REQUIRED_CONTAINER_CUDA_FLAG_MARKERS = [
    "compute_121a",
    "sm_121a",
]
REQUIRED_DEPENDENCY_NAMES = {"flashinfer", "sglang"}
REQUIRED_DEPENDENCY_REF_LENGTH = 40


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_path(path: str | None, manifest_path: pathlib.Path) -> pathlib.Path | None:
    if not path:
        return None
    p = pathlib.Path(path)
    if p.is_absolute() and p.exists():
        return p
    if p.is_absolute():
        copied_path = manifest_path.parent / p.name
        if copied_path.exists():
            return copied_path.resolve()
        return p
    return (manifest_path.parent / p).resolve()


def row_slug(model: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in model).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


def iter_deltas(compare: dict[str, Any] | None) -> list[float]:
    if not isinstance(compare, dict):
        return []
    rows = compare.get("rows")
    if not isinstance(rows, list):
        return []
    deltas = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("delta_nats_per_token")
        if isinstance(value, (int, float)):
            deltas.append(float(value))
    return deltas


def comparison_ctxs(compare: dict[str, Any] | None) -> list[int]:
    if not isinstance(compare, dict):
        return []
    rows = compare.get("rows")
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("ctx"), int):
            result.append(row["ctx"])
    return result


def missing_markers(path: pathlib.Path, markers: list[str]) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return markers
    return [marker for marker in markers if marker not in text]


def parse_key_value_log(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            result[key] = value.strip()
    return result


def capacity_total(payload: dict[str, Any] | None) -> int | None:
    if not isinstance(payload, dict):
        return None
    capacity = payload.get("kv_capacity")
    if not isinstance(capacity, dict):
        return None
    value = capacity.get("total_token_slots")
    return int(value) if isinstance(value, int) and value > 0 else None


def audit_run_preflight(
    manifest_path: pathlib.Path,
    *,
    manifest: dict[str, Any],
    findings: list[str],
) -> None:
    path = manifest_path.parent / "preflight.log"
    try:
        fields = parse_key_value_log(path)
    except OSError:
        findings.append("manifest missing run preflight artifact")
        return
    expected = {
        "run_id": manifest.get("run_id"),
        "image": manifest.get("image"),
        "image_digest": manifest.get("image_digest"),
        "models": " ".join(manifest.get("models") or []),
        "row_labels": " ".join(manifest.get("row_labels") or []),
        "mem_fraction_static": str(manifest.get("mem_fraction_static")),
        "page_size": str(manifest.get("page_size")),
        "context_length": str(manifest.get("context_length")),
        "ctx_list": " ".join(str(value) for value in manifest.get("ctx_list") or []),
    }
    for key, expected_value in expected.items():
        if fields.get(key) != expected_value:
            findings.append(f"run preflight {key} is not {expected_value}")
    for field in ("chat_timeout_s", "request_timeout_s", "ppl_timeout_s"):
        value = fields.get(field)
        if value is None:
            findings.append(f"run preflight missing {field}")
        else:
            try:
                if int(value) <= 0:
                    findings.append(f"run preflight {field} must be positive")
            except ValueError:
                findings.append(f"run preflight {field} must be an integer")


def audit_blocker_audit(payload: dict[str, Any], findings: list[str]) -> None:
    if payload.get("schema") != "sglang-gemma4-ar-ladder-blocker-audit/v1":
        findings.append("blocker_audit schema mismatch")
    if payload.get("can_run_claim_ladder") is not False:
        findings.append("blocker_audit can_run_claim_ladder must be false")

    status = payload.get("ladder_status")
    if status == "blocked-known-red-dependencies":
        findings.append("blocker_audit still records known-blocked dependency refs")
    elif status != "dependency-changed-review-before-rerun":
        findings.append("blocker_audit ladder_status is not dependency-changed-review-before-rerun")

    branches = payload.get("dependency_branches")
    if not isinstance(branches, dict) or set(branches) != REQUIRED_DEPENDENCY_NAMES:
        findings.append("blocker_audit dependency_branches must include flashinfer and sglang")

    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        findings.append("blocker_audit dependencies is not a list")
        return
    by_name = {
        item.get("name"): item
        for item in dependencies
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(by_name) != REQUIRED_DEPENDENCY_NAMES:
        findings.append("blocker_audit dependencies must include flashinfer and sglang")
        return

    changed = []
    for name, item in by_name.items():
        current_ref = item.get("current_ref")
        known_ref = item.get("known_blocked_ref")
        for field, value in (
            ("current_ref", current_ref),
            ("known_blocked_ref", known_ref),
        ):
            if not (
                isinstance(value, str)
                and len(value) == REQUIRED_DEPENDENCY_REF_LENGTH
                and all(ch in "0123456789abcdef" for ch in value.lower())
            ):
                findings.append(f"blocker_audit {name} {field} is not a full git sha")
        dependency_changed = item.get("dependency_changed")
        if dependency_changed != (current_ref != known_ref):
            findings.append(f"blocker_audit {name} dependency_changed mismatch")
        changed.append(bool(dependency_changed))

    if status == "dependency-changed-review-before-rerun" and not any(changed):
        findings.append("blocker_audit records dependency-changed status without changed dependency")
    if payload.get("diagnostic_override_allowed") is not True:
        findings.append("blocker_audit diagnostic_override_allowed must be true")


def audit_row_preflight(
    row_dir_path: pathlib.Path,
    *,
    label: str,
    model: str,
    expected_dtype: str,
    manifest: dict[str, Any],
    findings: list[str],
) -> None:
    path = row_dir_path / f"{label}_preflight.log"
    try:
        fields = parse_key_value_log(path)
    except OSError:
        findings.append(f"{label} preflight is unreadable")
        return
    expected = {
        "run_id": manifest.get("run_id"),
        "model": model,
        "label": label,
        "kv_cache_dtype": expected_dtype,
        "mixed_kv": "0",
        "image": manifest.get("image"),
        "image_digest": manifest.get("image_digest"),
        "mem_fraction_static": str(manifest.get("mem_fraction_static")),
        "page_size": str(manifest.get("page_size")),
        "context_length": str(manifest.get("context_length")),
        "sglang_source_overlay": "0",
    }
    for key, expected_value in expected.items():
        if fields.get(key) != expected_value:
            findings.append(f"{label} preflight {key} is not {expected_value}")
    for key in (
        "sglang_fp4_kv_global_scale_multiplier",
        "sglang_fp4_kv_k_global_scale_multiplier",
        "sglang_fp4_kv_v_global_scale_multiplier",
    ):
        if fields.get(key, ""):
            findings.append(f"{label} preflight {key} must be empty")
    if fields.get("allow_known_blocked_sglang_ar_ladder") == "1":
        reason = fields.get("sglang_ar_ladder_override_reason", "")
        if not reason:
            findings.append(f"{label} preflight override reason is empty")


def env_map(record: dict[str, Any]) -> dict[str, str]:
    config = record.get("Config") if isinstance(record.get("Config"), dict) else {}
    raw_env = config.get("Env")
    result: dict[str, str] = {}
    if not isinstance(raw_env, list):
        return result
    for item in raw_env:
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def container_args(record: dict[str, Any]) -> list[str]:
    raw_args = record.get("Args")
    if isinstance(raw_args, list):
        return [str(item) for item in raw_args]
    config = record.get("Config") if isinstance(record.get("Config"), dict) else {}
    raw_cmd = config.get("Cmd")
    if isinstance(raw_cmd, list):
        return [str(item) for item in raw_cmd]
    return []


def arg_value(args: list[str], key: str) -> str | None:
    for index, item in enumerate(args):
        if item == key and index + 1 < len(args):
            return args[index + 1]
    return None


def load_container_inspect(path: pathlib.Path) -> dict[str, Any] | None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    if isinstance(raw, dict):
        return raw
    return None


def audit_container_inspect(
    row_dir_path: pathlib.Path,
    *,
    label: str,
    model: str,
    expected_dtype: str,
    manifest: dict[str, Any],
    findings: list[str],
) -> None:
    path = row_dir_path / f"{label}_container_inspect.json"
    try:
        record = load_container_inspect(path)
    except (OSError, json.JSONDecodeError):
        findings.append(f"{label} container inspect is unreadable")
        return
    if not isinstance(record, dict):
        findings.append(f"{label} container inspect has no record")
        return

    config = record.get("Config") if isinstance(record.get("Config"), dict) else {}
    host_config = (
        record.get("HostConfig") if isinstance(record.get("HostConfig"), dict) else {}
    )
    state = record.get("State") if isinstance(record.get("State"), dict) else {}
    args = container_args(record)
    env = env_map(record)

    expected_image = manifest.get("image")
    if config.get("Image") != expected_image:
        findings.append(f"{label} container image does not match manifest image")
    if config.get("WorkingDir") != "/hijinks":
        findings.append(f"{label} container working dir is not /hijinks")
    if host_config.get("Memory") != REQUIRED_DOCKER_MEMORY_BYTES:
        findings.append(f"{label} container memory is not 100g")
    if host_config.get("MemorySwap") != REQUIRED_DOCKER_MEMORY_BYTES:
        findings.append(f"{label} container memory-swap is not 100g")
    if host_config.get("NetworkMode") != "host":
        findings.append(f"{label} container network mode is not host")
    if host_config.get("IpcMode") != "host":
        findings.append(f"{label} container ipc mode is not host")
    if state.get("OOMKilled") is not False:
        findings.append(f"{label} container OOMKilled is not false")

    binds = host_config.get("Binds")
    if manifest.get("source_overlay") is False and isinstance(binds, list):
        if any(isinstance(item, str) and ":/work/third_party/sglang" in item for item in binds):
            findings.append(f"{label} container unexpectedly mounts SGLang source overlay")

    for key, expected_value in REQUIRED_CONTAINER_ENV.items():
        if env.get(key) != expected_value:
            findings.append(f"{label} container env {key} is not {expected_value}")
    cuda_flags = env.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    for marker in REQUIRED_CONTAINER_CUDA_FLAG_MARKERS:
        if marker not in cuda_flags:
            findings.append(
                f"{label} container FLASHINFER_EXTRA_CUDAFLAGS missing {marker}"
            )
    cache_dir = env.get("FLASHINFER_CACHE_DIR", "")
    if not cache_dir.startswith("/tmp/flashinfer-cache-"):
        findings.append(f"{label} container FLASHINFER_CACHE_DIR is not per-run /tmp cache")

    expected_args = {
        "--model-path": model,
        "--served-model-name": model,
        "--dtype": "bfloat16",
        "--attention-backend": "flashinfer",
        "--page-size": str(manifest.get("page_size")),
        "--context-length": str(manifest.get("context_length")),
        "--mem-fraction-static": str(manifest.get("mem_fraction_static")),
    }
    for key, expected_value in expected_args.items():
        if arg_value(args, key) != expected_value:
            findings.append(f"{label} container arg {key} is not {expected_value}")
    for flag in ("--disable-cuda-graph", "--disable-piecewise-cuda-graph"):
        if flag not in args:
            findings.append(f"{label} container missing arg {flag}")
    kv_arg = arg_value(args, "--kv-cache-dtype")
    if expected_dtype == "auto":
        if kv_arg is not None:
            findings.append(f"{label} container unexpectedly sets --kv-cache-dtype")
    elif kv_arg != expected_dtype:
        findings.append(f"{label} container --kv-cache-dtype is not {expected_dtype}")


def chat_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def audit_chat_artifacts(
    row_dir_path: pathlib.Path,
    *,
    label: str,
    model: str,
    summary: dict[str, Any],
    findings: list[str],
) -> None:
    contents: list[str] = []
    for index in (1, 2):
        path = row_dir_path / f"{label}_chat_{index}.json"
        try:
            report = load_json(path)
        except (OSError, json.JSONDecodeError):
            findings.append(f"{label} chat_{index} artifact is unreadable")
            continue
        if report.get("object") != "chat.completion":
            findings.append(f"{label} chat_{index} object is not chat.completion")
        if report.get("model") != model:
            findings.append(f"{label} chat_{index} model mismatch")
        content = chat_content(report)
        if not isinstance(content, str) or not content.strip():
            findings.append(f"{label} chat_{index} content is missing")
        else:
            contents.append(content.strip())
        choices = report.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
        if finish_reason != "stop":
            findings.append(f"{label} chat_{index} finish_reason is not stop")
        usage = report.get("usage")
        if not isinstance(usage, dict):
            findings.append(f"{label} chat_{index} usage is missing")
        else:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(field)
                if not isinstance(value, int) or value <= 0:
                    findings.append(f"{label} chat_{index} usage {field} must be positive")

    if len(contents) == 2:
        if contents[0] != contents[1]:
            findings.append(f"{label} chat artifact content mismatch")
        if REQUIRED_CHAT_CONTENT_SUBSTRING not in contents[0].casefold():
            findings.append(
                f"{label} chat content does not contain {REQUIRED_CHAT_CONTENT_SUBSTRING!r}"
            )
        for summary_key, content in (("chat_1", contents[0]), ("chat_2", contents[1])):
            summary_value = summary.get(summary_key)
            if isinstance(summary_value, str) and summary_value.strip() != content:
                findings.append(f"{label} summary {summary_key} does not match chat artifact")


def audit_ppl_report(
    report: dict[str, Any],
    *,
    label: str,
    model: str,
    expected_dtype: str,
    expected_ctxs: set[int],
    manifest: dict[str, Any],
    findings: list[str],
) -> None:
    if report.get("schema") != "sglang-prompt-ppl-sweep/v1":
        findings.append(f"{label} ppl report schema mismatch")
    if report.get("ok") is not True:
        findings.append(f"{label} ppl report ok is not true")
    if report.get("tokenizer") != model:
        findings.append(f"{label} ppl report tokenizer mismatch")
    if report.get("kv_cache_dtype") != expected_dtype:
        findings.append(f"{label} ppl report kv_cache_dtype is not {expected_dtype}")
    if report.get("container_image") != manifest.get("image"):
        findings.append(f"{label} ppl report container_image does not match manifest image")

    contexts = report.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        findings.append(f"{label} ppl report has no contexts")
        return
    ctxs = {
        row.get("ctx")
        for row in contexts
        if isinstance(row, dict) and isinstance(row.get("ctx"), int)
    }
    if expected_ctxs and ctxs != expected_ctxs:
        findings.append(
            f"{label} ppl report ctx coverage {sorted(ctxs)} "
            f"does not match manifest ctx_list {sorted(expected_ctxs)}"
        )

    for row in contexts:
        if not isinstance(row, dict):
            findings.append(f"{label} ppl report contains non-object context row")
            continue
        ctx = row.get("ctx")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        hardware = report.get("hardware") if isinstance(report.get("hardware"), dict) else {}

        if payload.get("prompt_token_count") != ctx:
            findings.append(f"{label} ctx {ctx}: prompt_token_count mismatch")
        for field in ("max_new_tokens", "reuse_prefix_len", "logprob_start_len"):
            if payload.get(field) != manifest.get(field):
                findings.append(f"{label} ctx {ctx}: payload {field} mismatch")
        if payload.get("score_start_index") != manifest.get("logprob_start_len"):
            findings.append(f"{label} ctx {ctx}: score_start_index mismatch")
        if score.get("ok") is not True:
            findings.append(f"{label} ctx {ctx}: score ok is not true")
        if score.get("cached_tokens") != manifest.get("reuse_prefix_len"):
            findings.append(f"{label} ctx {ctx}: cached_tokens mismatch")
        if not isinstance(score.get("num_scored_tokens"), int) or score["num_scored_tokens"] <= 0:
            findings.append(f"{label} ctx {ctx}: num_scored_tokens must be positive")
        if score.get("num_missing_tokens") != 0:
            findings.append(f"{label} ctx {ctx}: num_missing_tokens is not zero")
        if score.get("num_mismatched_tokens") != 0:
            findings.append(f"{label} ctx {ctx}: num_mismatched_tokens is not zero")
        if hardware.get("cuda_available") is not True:
            findings.append(f"{label} ppl report hardware cuda_available is not true")
        devices = hardware.get("devices")
        if not isinstance(devices, list) or not devices:
            findings.append(f"{label} ppl report hardware devices missing")
        else:
            comparison_key = devices[0].get("comparison_key") if isinstance(devices[0], dict) else None
            if not isinstance(comparison_key, str) or not comparison_key.startswith(
                REQUIRED_HARDWARE_COMPARISON_KEY_PREFIX
            ):
                findings.append(
                    f"{label} ppl report hardware comparison_key is not GB10 sm_121"
                )


def audit_manifest(
    manifest_path: pathlib.Path,
    *,
    max_delta_nats: float,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    findings: list[str] = []
    warnings: list[str] = []

    if manifest.get("schema") != "sglang-gemma4-ar-ladder-pair/v1":
        findings.append("manifest schema is not sglang-gemma4-ar-ladder-pair/v1")

    row_labels = manifest.get("row_labels")
    if not isinstance(row_labels, list):
        findings.append("manifest row_labels is missing or not a list")
        row_labels = []
    elif set(row_labels) != set(REQUIRED_ROWS):
        findings.append(
            f"manifest row_labels must be exactly {REQUIRED_ROWS}, got {row_labels}"
        )
    for label in REQUIRED_ROWS:
        if label not in row_labels:
            findings.append(f"manifest row_labels missing required row {label}")

    models = manifest.get("models")
    if not isinstance(models, list):
        findings.append("manifest models is missing or not a list")
    elif set(models) != set(REQUIRED_MODELS):
        findings.append(
            f"manifest models must be exactly {REQUIRED_MODELS}, got {models}"
        )

    image = manifest.get("image")
    image_digest = manifest.get("image_digest")
    if not image or not image_digest:
        findings.append("manifest missing image or image_digest")
    if isinstance(image_digest, str) and "sha256:" not in image_digest:
        findings.append("image_digest does not include sha256")

    for artifact_name in ("corpus", "corpus_manifest"):
        artifact_path = normalize_path(manifest.get(artifact_name), manifest_path)
        if not artifact_path or not artifact_path.exists():
            findings.append(f"manifest missing {artifact_name} artifact")

    ctx_list = manifest.get("ctx_list")
    expected_ctxs: set[int] = set()
    if not isinstance(ctx_list, list) or not ctx_list:
        findings.append("manifest ctx_list is missing or empty")
    elif not all(isinstance(value, int) and value > 0 for value in ctx_list):
        findings.append("manifest ctx_list must contain positive integers")
    else:
        expected_ctxs = set(ctx_list)

    for field in REQUIRED_POSITIVE_INT_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, int) or value <= 0:
            findings.append(f"manifest {field} must be a positive integer")

    if manifest.get("graphs") != "disabled":
        findings.append("manifest graphs must be disabled for the current AR claim gate")
    if manifest.get("source_overlay") is not False:
        findings.append("manifest source_overlay must be false for claim-grade rows")
    if manifest.get("allow_retracted_global_scale_diagnostic") is not False:
        findings.append(
            "manifest allow_retracted_global_scale_diagnostic must be false for claim-grade rows"
        )
    audit_run_preflight(manifest_path, manifest=manifest, findings=findings)

    blocker_audit_path = normalize_path(manifest.get("blocker_audit"), manifest_path)
    if not blocker_audit_path or not blocker_audit_path.exists():
        findings.append("manifest missing blocker_audit artifact")
    else:
        blocker_audit = load_json(blocker_audit_path)
        audit_blocker_audit(blocker_audit, findings)

    rows = manifest.get("rows")
    if not isinstance(rows, list):
        findings.append("manifest rows is missing or not a list")
        rows = []
    rows_by_model = {
        row.get("model"): row for row in rows if isinstance(row, dict) and row.get("model")
    }

    model_results: dict[str, Any] = {}
    for model in REQUIRED_MODELS:
        row = rows_by_model.get(model)
        model_result: dict[str, Any] = {"findings": [], "warnings": []}
        model_results[model] = model_result
        if row is None:
            model_result["findings"].append("missing model row")
            findings.extend(f"{model}: {item}" for item in model_result["findings"])
            continue

        expected_slug = row_slug(model)
        row_dir = row.get("dir")
        row_dir_path = normalize_path(row_dir, manifest_path) if isinstance(row_dir, str) else None
        if isinstance(row_dir, str) and not row_dir.replace("\\", "/").endswith(expected_slug):
            model_result["warnings"].append(
                f"row dir does not end with expected slug {expected_slug}: {row_dir}"
            )
        if not row_dir_path or not row_dir_path.exists() or not row_dir_path.is_dir():
            model_result["findings"].append("row dir artifact is missing")

        capacity_totals: dict[str, int] = {}
        for label in REQUIRED_ROWS:
            payload = row.get(label)
            if not isinstance(payload, dict):
                model_result["findings"].append(f"missing {label} summary")
                continue
            if row_dir_path and row_dir_path.exists():
                for artifact_template in REQUIRED_ROW_ARTIFACTS:
                    artifact = row_dir_path / artifact_template.format(label=label)
                    if not artifact.exists():
                        model_result["findings"].append(
                            f"missing {label} artifact {artifact.name}"
                        )
                provenance_log = row_dir_path / f"{label}_provenance.log"
                for marker in missing_markers(provenance_log, REQUIRED_PROVENANCE_MARKERS):
                    model_result["findings"].append(
                        f"{label} provenance log missing marker {marker!r}"
                    )
                server_log = row_dir_path / f"{label}_server.log"
                server_markers = list(REQUIRED_SERVER_MARKERS)
                if label == "fullnvfp4":
                    server_markers.extend(REQUIRED_FULLNVFP4_SERVER_MARKERS)
                for marker in missing_markers(server_log, server_markers):
                    model_result["findings"].append(
                        f"{label} server log missing marker {marker!r}"
                    )
                ppl_report_path = row_dir_path / f"{label}_ppl.json"
                try:
                    ppl_report = load_json(ppl_report_path)
                except (OSError, json.JSONDecodeError):
                    model_result["findings"].append(f"{label} ppl report is unreadable")
                else:
                    audit_ppl_report(
                        ppl_report,
                        label=label,
                        model=model,
                        expected_dtype=EXPECTED_ROW_KV_DTYPES[label],
                        expected_ctxs=expected_ctxs,
                        manifest=manifest,
                        findings=model_result["findings"],
                    )
                audit_chat_artifacts(
                    row_dir_path,
                    label=label,
                    model=model,
                    summary=payload,
                    findings=model_result["findings"],
                )
                audit_container_inspect(
                    row_dir_path,
                    label=label,
                    model=model,
                    expected_dtype=EXPECTED_ROW_KV_DTYPES[label],
                    manifest=manifest,
                    findings=model_result["findings"],
                )
                audit_row_preflight(
                    row_dir_path,
                    label=label,
                    model=model,
                    expected_dtype=EXPECTED_ROW_KV_DTYPES[label],
                    manifest=manifest,
                    findings=model_result["findings"],
                )
            if payload.get("model") != model:
                model_result["findings"].append(f"{label} summary model mismatch")
            if payload.get("label") != label:
                model_result["findings"].append(f"{label} summary label mismatch")
            expected_dtype = EXPECTED_ROW_KV_DTYPES[label]
            if payload.get("kv_cache_dtype") != expected_dtype:
                model_result["findings"].append(
                    f"{label} kv_cache_dtype is not {expected_dtype}"
                )
            if payload.get("ppl_ok") is not True:
                model_result["findings"].append(f"{label} ppl_ok is not true")
            if payload.get("chat_transport_ok") is not True:
                model_result["findings"].append(f"{label} chat_transport_ok is not true")
            if payload.get("chat_content_equal") is not True:
                model_result["findings"].append(f"{label} chat content is not repeat-stable")
            capacity = payload.get("kv_capacity")
            if not isinstance(capacity, dict):
                model_result["findings"].append(f"{label} missing kv_capacity")
            else:
                for field in REQUIRED_CAPACITY_FIELDS:
                    value = capacity.get(field)
                    if not isinstance(value, (int, float)) or value <= 0:
                        model_result["findings"].append(
                            f"{label} kv_capacity {field} must be positive"
                        )
                total = capacity_total(payload)
                if total is not None:
                    capacity_totals[label] = total

        if all(label in capacity_totals for label in REQUIRED_ROWS):
            bf16_total = capacity_totals["bf16"]
            fp8_total = capacity_totals["fp8"]
            fullnvfp4_total = capacity_totals["fullnvfp4"]
            model_result["capacity_token_slots"] = capacity_totals
            model_result["fp8_vs_bf16_capacity_ratio"] = fp8_total / bf16_total
            model_result["fullnvfp4_vs_bf16_capacity_ratio"] = (
                fullnvfp4_total / bf16_total
            )
            if not (bf16_total < fp8_total < fullnvfp4_total):
                model_result["findings"].append(
                    "capacity token slots must increase bf16 < fp8 < fullnvfp4"
                )

        for comparison_name in REQUIRED_COMPARISONS:
            compare = row.get(comparison_name)
            if not isinstance(compare, dict):
                model_result["findings"].append(f"missing {comparison_name}")
                continue
            if row_dir_path and row_dir_path.exists():
                artifact = row_dir_path / f"{comparison_name}.json"
                if not artifact.exists():
                    model_result["findings"].append(
                        f"missing comparison artifact {artifact.name}"
                    )
            if compare.get("ok") is not True:
                model_result["findings"].append(f"{comparison_name} ok is not true")
            deltas = iter_deltas(compare)
            if not deltas:
                model_result["findings"].append(f"{comparison_name} has no delta rows")
                continue
            ctxs = set(comparison_ctxs(compare))
            if expected_ctxs and ctxs != expected_ctxs:
                model_result["findings"].append(
                    f"{comparison_name} ctx coverage {sorted(ctxs)} "
                    f"does not match manifest ctx_list {sorted(expected_ctxs)}"
                )
            max_abs_delta = max(abs(value) for value in deltas)
            model_result[f"{comparison_name}_max_abs_delta_nats"] = max_abs_delta
            if max_abs_delta > max_delta_nats:
                model_result["findings"].append(
                    f"{comparison_name} max |delta_nats_per_token| "
                    f"{max_abs_delta:.6f} exceeds threshold {max_delta_nats:.6f}"
                )

        findings.extend(f"{model}: {item}" for item in model_result["findings"])
        warnings.extend(f"{model}: {item}" for item in model_result["warnings"])

    ok = not findings
    return {
        "schema": "sglang-gemma4-ar-claim-audit/v1",
        "manifest": str(manifest_path),
        "ok": ok,
        "max_delta_nats": max_delta_nats,
        "required_models": REQUIRED_MODELS,
        "required_rows": REQUIRED_ROWS,
        "expected_row_kv_dtypes": EXPECTED_ROW_KV_DTYPES,
        "required_comparisons": REQUIRED_COMPARISONS,
        "required_row_artifacts": REQUIRED_ROW_ARTIFACTS,
        "required_provenance_markers": REQUIRED_PROVENANCE_MARKERS,
        "required_server_markers": REQUIRED_SERVER_MARKERS,
        "required_fullnvfp4_server_markers": REQUIRED_FULLNVFP4_SERVER_MARKERS,
        "required_capacity_fields": REQUIRED_CAPACITY_FIELDS,
        "required_hardware_comparison_key_prefix": REQUIRED_HARDWARE_COMPARISON_KEY_PREFIX,
        "required_positive_int_fields": REQUIRED_POSITIVE_INT_FIELDS,
        "required_docker_memory_bytes": REQUIRED_DOCKER_MEMORY_BYTES,
        "required_container_env": REQUIRED_CONTAINER_ENV,
        "required_container_cuda_flag_markers": REQUIRED_CONTAINER_CUDA_FLAG_MARKERS,
        "required_chat_content_substring": REQUIRED_CHAT_CONTENT_SUBSTRING,
        "findings": findings,
        "warnings": warnings,
        "model_results": model_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--max-delta-nats", type=float, default=0.25)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    result = audit_manifest(args.manifest, max_delta_nats=args.max_delta_nats)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
