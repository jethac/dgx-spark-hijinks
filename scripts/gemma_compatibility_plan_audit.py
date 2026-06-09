#!/usr/bin/env python3
"""Audit Gemma compatibility-plan invariants.

This is a lightweight guard for the campaign ladder. It does not prove a model
serves; it makes sure the written plan still encodes the constraints that keep
Gemma NVFP4-KV claims honest.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_PHRASES = {
    "running_model_geometry": "Every rung re-measures geometry from the running model",
    "gemma_specific_quality": "PPL/quality comparator measured on Gemma specifically",
    "do_not_transfer_qwen_ppl": "do not assume Qwen's PPL transfers",
    "capacity_not_speed": "frame rung results as",
    "attention_outlier_risk": "Gemma attention is outlier-sensitive",
    "quality_gate_task": "correct output",
}

EXPECTED_ORDER = [
    "Gemma 3 27B",
    "Gemma 4 31B",
    "Gemma 4 26B-A4B",
    "Gemma 4 12B",
]


def load_json(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def phrase_present(text: str, phrase: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text)).lower()
    normalized_phrase = re.sub(r"\s+", " ", re.sub(r"[*_`]", "", phrase)).lower()
    return normalized_phrase in normalized_text


def find_rung_order(text: str) -> list[str]:
    _, _, ladder_text = text.partition("## Main ladder")
    if ladder_text:
        ladder_text, _, _ = ladder_text.partition("## Side track")
    else:
        ladder_text = text
    order: list[tuple[int, str]] = []
    for model in EXPECTED_ORDER:
        match = re.search(re.escape(model), ladder_text)
        if match:
            order.append((match.start(), model))
    return [model for _, model in sorted(order)]


def audit_plan(plan_text: str, strict_artifact: dict[str, Any] | None) -> dict[str, Any]:
    findings: list[str] = []
    phrase_checks = []
    for name, phrase in REQUIRED_PHRASES.items():
        ok = phrase_present(plan_text, phrase)
        phrase_checks.append({"name": name, "phrase": phrase, "ok": ok})
        if not ok:
            findings.append(f"missing required phrase for {name}: {phrase!r}")

    rung_order = find_rung_order(plan_text)
    if rung_order != EXPECTED_ORDER:
        findings.append(
            "Gemma server rung order mismatch: "
            f"found {rung_order}, expected {EXPECTED_ORDER}"
        )

    strict_order = None
    if strict_artifact is not None:
        strict_order = (
            strict_artifact.get("decision", {})
            .get("recommended_main_ladder_after_audit")
        )
        if strict_order != [
            "rung_0_qwen_standard_attention_done",
            "rung_1_gemma3_27b_swa_uniform_d128_no_d512",
            "rung_2_gemma4_31b_dense_d512_text_only",
            "rung_3_gemma4_26b_a4b_moe_text_only",
            "rung_4_gemma4_12b_encoder_free_multimodal_kv",
        ]:
            findings.append(
                "strict rung-minus-one artifact recommended_ladder is missing or stale"
            )

    return {
        "schema": "gemma-compatibility-plan-audit/v1",
        "required_phrase_checks": phrase_checks,
        "server_rung_order": rung_order,
        "expected_server_rung_order": EXPECTED_ORDER,
        "strict_artifact_recommended_ladder": strict_order,
        "ok": not findings,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="docs/GEMMA_COMPATIBILITY_PLAN.md")
    parser.add_argument(
        "--strict-artifact",
        default="results/gemma_rung_minus1_config_audit_strict_20260608.json",
    )
    parser.add_argument("--output", default="results/gemma_compatibility_plan_audit_20260609.json")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    strict_path = Path(args.strict_artifact) if args.strict_artifact else None
    report = audit_plan(
        plan_path.read_text(encoding="utf-8", errors="replace"),
        load_json(strict_path) if strict_path and strict_path.exists() else None,
    )
    report["plan"] = args.plan
    report["strict_artifact"] = args.strict_artifact
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
