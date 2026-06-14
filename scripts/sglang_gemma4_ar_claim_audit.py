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


def capacity_total(payload: dict[str, Any] | None) -> int | None:
    if not isinstance(payload, dict):
        return None
    capacity = payload.get("kv_capacity")
    if not isinstance(capacity, dict):
        return None
    value = capacity.get("total_token_slots")
    return int(value) if isinstance(value, int) and value > 0 else None


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

    blocker_audit_path = normalize_path(manifest.get("blocker_audit"), manifest_path)
    if not blocker_audit_path or not blocker_audit_path.exists():
        findings.append("manifest missing blocker_audit artifact")
    else:
        blocker_audit = load_json(blocker_audit_path)
        if blocker_audit.get("can_run_claim_ladder") is not False:
            warnings.append("blocker_audit did not explicitly keep can_run_claim_ladder=false")
        if blocker_audit.get("ladder_status") == "blocked-known-red-dependencies":
            findings.append("blocker_audit still records known-blocked dependency refs")

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
