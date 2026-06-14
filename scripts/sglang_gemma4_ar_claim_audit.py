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
REQUIRED_COMPARISONS = [
    "compare_bf16_vs_fullnvfp4",
    "compare_fp8_vs_fullnvfp4",
]
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
    if p.is_absolute():
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
    for label in REQUIRED_ROWS:
        if label not in row_labels:
            findings.append(f"manifest row_labels missing required row {label}")

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
    if not isinstance(ctx_list, list) or not ctx_list:
        findings.append("manifest ctx_list is missing or empty")
    elif not all(isinstance(value, int) and value > 0 for value in ctx_list):
        findings.append("manifest ctx_list must contain positive integers")

    for field in REQUIRED_POSITIVE_INT_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, int) or value <= 0:
            findings.append(f"manifest {field} must be a positive integer")

    if manifest.get("graphs") != "disabled":
        findings.append("manifest graphs must be disabled for the current AR claim gate")

    blocker_audit_path = normalize_path(manifest.get("blocker_audit"), manifest_path)
    if not blocker_audit_path or not blocker_audit_path.exists():
        findings.append("manifest missing blocker_audit artifact")
    else:
        blocker_audit = load_json(blocker_audit_path)
        if blocker_audit.get("can_run_claim_ladder") is not False:
            warnings.append("blocker_audit did not explicitly keep can_run_claim_ladder=false")
        if blocker_audit.get("ladder_status") == "blocked-known-red-dependencies":
            warnings.append("blocker_audit still records known-blocked dependency refs")

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
        if isinstance(row_dir, str) and not row_dir.replace("\\", "/").endswith(expected_slug):
            model_result["warnings"].append(
                f"row dir does not end with expected slug {expected_slug}: {row_dir}"
            )

        for label in REQUIRED_ROWS:
            payload = row.get(label)
            if not isinstance(payload, dict):
                model_result["findings"].append(f"missing {label} summary")
                continue
            if payload.get("model") != model:
                model_result["findings"].append(f"{label} summary model mismatch")
            if payload.get("ppl_ok") is not True:
                model_result["findings"].append(f"{label} ppl_ok is not true")
            if payload.get("chat_transport_ok") is not True:
                model_result["findings"].append(f"{label} chat_transport_ok is not true")
            if payload.get("chat_content_equal") is not True:
                model_result["findings"].append(f"{label} chat content is not repeat-stable")

        for comparison_name in REQUIRED_COMPARISONS:
            compare = row.get(comparison_name)
            if not isinstance(compare, dict):
                model_result["findings"].append(f"missing {comparison_name}")
                continue
            if compare.get("ok") is not True:
                model_result["findings"].append(f"{comparison_name} ok is not true")
            deltas = iter_deltas(compare)
            if not deltas:
                model_result["findings"].append(f"{comparison_name} has no delta rows")
                continue
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
        "required_comparisons": REQUIRED_COMPARISONS,
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
