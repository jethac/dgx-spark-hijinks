#!/usr/bin/env python3
"""Audit whether a Gemma NVFP4-KV row has enough quality evidence to bless.

The existing OpenAI text-quality heuristic can miss Gemma KV corruption because
garbled multilingual output can still look non-empty and non-repetitive. This
gate requires either supplied-token PPL evidence or a first-token/logprob
comparator, matched against an fp8/bf16 KV baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def lower_text(value: Any) -> str:
    return str(value or "").lower()


def manifest_kv_dtype(manifest: dict[str, Any]) -> str:
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    kv_dtype = metadata.get("kv_cache_dtype")
    if isinstance(kv_dtype, str) and kv_dtype:
        return kv_dtype.lower()
    run_id = lower_text(manifest.get("run_id"))
    if "nvfp4" in run_id:
        return "nvfp4"
    if "fp8" in run_id:
        return "fp8"
    if "bf16" in run_id:
        return "bf16"
    return "unknown"


def validate_manifest(
    *,
    label: str,
    manifest: dict[str, Any] | None,
    expected_kind: str,
    findings: list[str],
) -> dict[str, Any]:
    if manifest is None:
        findings.append(f"{label}: manifest is missing")
        return {"present": False}

    schema = manifest.get("schema")
    model = lower_text(manifest.get("model"))
    run_id = manifest.get("run_id")
    kv_dtype = manifest_kv_dtype(manifest)
    ok = manifest.get("ok")
    summary = {
        "present": True,
        "schema": schema,
        "run_id": run_id,
        "model": manifest.get("model"),
        "kv_cache_dtype": kv_dtype,
        "ok": ok,
    }
    if schema != "openai-serving-row-manifest/v1":
        findings.append(f"{label}: expected openai-serving-row-manifest/v1, got {schema!r}")
    if "gemma" not in model and "gemma" not in lower_text(run_id):
        findings.append(f"{label}: row does not look like a Gemma row")
    if ok is not True:
        findings.append(f"{label}: row manifest ok is not true")
    if expected_kind == "candidate" and kv_dtype != "nvfp4":
        findings.append(f"{label}: candidate kv_cache_dtype is {kv_dtype!r}, expected nvfp4")
    if expected_kind == "baseline" and kv_dtype == "nvfp4":
        findings.append(f"{label}: baseline must not use nvfp4 KV")
    if expected_kind == "baseline" and kv_dtype not in {"fp8", "bf16", "auto", "unknown"}:
        findings.append(f"{label}: unexpected baseline kv_cache_dtype {kv_dtype!r}")
    return summary


def validate_first_token(
    report: dict[str, Any] | None,
    *,
    min_overlap_ratio: float,
) -> tuple[dict[str, Any], list[str]]:
    findings: list[str] = []
    if report is None:
        return {"present": False}, findings
    schema = report.get("schema")
    comparisons = report.get("comparisons")
    summary = {
        "present": True,
        "schema": schema,
        "ok": report.get("ok"),
        "comparison_count": len(comparisons) if isinstance(comparisons, list) else 0,
        "min_top_logprob_overlap_ratio": None,
    }
    if schema != "openai-first-token-compare/v1":
        findings.append(f"first-token report: unexpected schema {schema!r}")
        return summary, findings
    if not isinstance(comparisons, list) or not comparisons:
        findings.append("first-token report: comparisons are missing")
        return summary, findings
    ratios: list[float] = []
    for item in comparisons:
        if not isinstance(item, dict):
            findings.append("first-token report: comparison entry is not an object")
            continue
        if item.get("ok") is not True:
            findings.append(
                f"first-token report: case {item.get('case')!r} is not ok"
            )
        ratio = item.get("top_logprob_overlap_ratio")
        if isinstance(ratio, (int, float)) and math.isfinite(float(ratio)):
            ratios.append(float(ratio))
    if ratios:
        summary["min_top_logprob_overlap_ratio"] = min(ratios)
        if min(ratios) < min_overlap_ratio:
            findings.append(
                "first-token report: minimum top-logprob overlap ratio "
                f"{min(ratios):.6g} is below {min_overlap_ratio:.6g}"
            )
    else:
        findings.append("first-token report: no finite top-logprob overlap ratios")
    if report.get("ok") is not True:
        findings.append("first-token report: report ok is not true")
    return summary, findings


def validate_ppl(
    report: dict[str, Any] | None,
    *,
    max_delta_nats: float,
    min_contexts: int,
) -> tuple[dict[str, Any], list[str]]:
    findings: list[str] = []
    if report is None:
        return {"present": False}, findings
    schema = report.get("schema")
    rows = report.get("rows")
    summary = {
        "present": True,
        "schema": schema,
        "ok": report.get("ok"),
        "row_count": len(rows) if isinstance(rows, list) else 0,
        "max_abs_delta_nats_per_token": None,
    }
    if schema != "vllm-prompt-ppl-comparison/v1":
        findings.append(f"PPL report: unexpected schema {schema!r}")
        return summary, findings
    if not isinstance(rows, list) or len(rows) < min_contexts:
        findings.append(
            f"PPL report: expected at least {min_contexts} context row(s), "
            f"got {0 if not isinstance(rows, list) else len(rows)}"
        )
        return summary, findings
    deltas: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            findings.append("PPL report: row is not an object")
            continue
        if row.get("fp8_ok") is not True or row.get("nvfp4_ok") is not True:
            findings.append(f"PPL report: ctx {row.get('ctx')} has a non-ok score")
        delta = row.get("delta_nats_per_token")
        if isinstance(delta, (int, float)) and math.isfinite(float(delta)):
            deltas.append(float(delta))
        else:
            findings.append(f"PPL report: ctx {row.get('ctx')} lacks finite delta nats")
    if deltas:
        max_abs = max(abs(delta) for delta in deltas)
        summary["max_abs_delta_nats_per_token"] = max_abs
        if max_abs > max_delta_nats:
            findings.append(
                "PPL report: max abs delta nats/token "
                f"{max_abs:.6g} exceeds {max_delta_nats:.6g}"
            )
    if report.get("ok") is not True:
        findings.append("PPL report: report ok is not true")
    return summary, findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--first-token-compare")
    parser.add_argument("--ppl-compare")
    parser.add_argument("--min-first-token-overlap-ratio", type=float, default=0.2)
    parser.add_argument("--max-delta-nats-per-token", type=float, default=0.05)
    parser.add_argument("--min-ppl-contexts", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    findings: list[str] = []
    candidate = load_json(args.candidate_manifest)
    baseline = load_json(args.baseline_manifest)
    first_token = load_json(args.first_token_compare)
    ppl = load_json(args.ppl_compare)

    candidate_summary = validate_manifest(
        label="candidate",
        manifest=candidate,
        expected_kind="candidate",
        findings=findings,
    )
    baseline_summary = validate_manifest(
        label="baseline",
        manifest=baseline,
        expected_kind="baseline",
        findings=findings,
    )
    first_summary, first_findings = validate_first_token(
        first_token,
        min_overlap_ratio=args.min_first_token_overlap_ratio,
    )
    ppl_summary, ppl_findings = validate_ppl(
        ppl,
        max_delta_nats=args.max_delta_nats_per_token,
        min_contexts=args.min_ppl_contexts,
    )
    first_ok = first_summary.get("present") and not first_findings
    ppl_ok = ppl_summary.get("present") and not ppl_findings
    if not first_ok and not ppl_ok:
        findings.append(
            "quality evidence: provide a passing first-token/logprob comparison "
            "or a passing supplied-token PPL comparison for the Gemma row"
        )
        findings.extend(first_findings)
        findings.extend(ppl_findings)

    report = {
        "schema": "gemma-nvfp4-kv-quality-gate/v1",
        "candidate_manifest": args.candidate_manifest,
        "baseline_manifest": args.baseline_manifest,
        "first_token_compare": args.first_token_compare,
        "ppl_compare": args.ppl_compare,
        "thresholds": {
            "min_first_token_overlap_ratio": args.min_first_token_overlap_ratio,
            "max_delta_nats_per_token": args.max_delta_nats_per_token,
            "min_ppl_contexts": args.min_ppl_contexts,
        },
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "first_token": first_summary,
        "ppl": ppl_summary,
        "ok": not findings,
        "findings": findings,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
