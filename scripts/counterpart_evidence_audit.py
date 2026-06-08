#!/usr/bin/env python3
"""Audit SGLang and llama.cpp counterpart evidence for AEON-derived goals."""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceRequirement:
    name: str
    runtime: str
    model_family: str
    title: str
    acceptance: str
    claim_groups: tuple[tuple[str, ...], ...]
    partial_patterns: tuple[str, ...]
    blocked_patterns: tuple[str, ...] = ()
    forbidden_claim_fragments: tuple[str, ...] = ()


REQUIREMENTS = (
    EvidenceRequirement(
        name="sglang_gemma_nvfp4_ordinary_kv",
        runtime="sglang",
        model_family="gemma",
        title="SGLang Gemma NVFP4-weight serving with ordinary KV",
        acceptance=(
            "SGLang serves a Gemma NVFP4-weight checkpoint with BF16/fp8 ordinary KV, "
            "deterministic output sanity, backend logs, and row manifest."
        ),
        claim_groups=(
            (
                "results/sglang*gemma*nvfp4*row_manifest.json",
                "results/sglang*gemma*nvfp4*openai_benchmark.json",
            ),
            ("results/sglang*gemma*nvfp4*checkpoint_audit.json",),
        ),
        partial_patterns=(
            "results/sglang_gemma4_e2b_w4a16*_server.log",
            "results/sglang_gemma4_e2b_w4a16*_health_status.txt",
        ),
    ),
    EvidenceRequirement(
        name="sglang_clean_fp4_kv_after_row",
        runtime="sglang",
        model_family="qwen",
        title="SGLang FP4 KV capacity and quality after-row",
        acceptance=(
            "Clean fork/container SGLang FP4 KV row serves with a justified graph policy, "
            "passes deterministic quality checks, includes an fp8 comparator, and records "
            "capacity and throughput in row manifests."
        ),
        claim_groups=(
            (
                "results/sglang*fp4kv*row_manifest.json",
                "results/sglang*fp4_e2m1*row_manifest.json",
            ),
            (
                "results/sglang*fp4kv*openai_benchmark.json",
                "results/sglang*fp4_e2m1*openai_benchmark.json",
            ),
        ),
        partial_patterns=(
            "results/flashinfer_sglang_linear_nvfp4_kv_probe*.json",
            "results/sglang_qwen25_1_5b_fp4kv*_startup.log",
            "results/sglang_qwen25_1_5b_fp4kv*_openai_benchmark.json",
            "results/sglang_qwen_fp4kv_clean*.json",
            "results/sglang_qwen_fp4kv_clean*.log",
            "results/sglang_qwen_fp4kv_decode_dtype*.json",
            "results/sglang_qwen_fp4kv_decode_dtype*.log",
            "results/sglang_qwen_fp4kv_eager_only*.json",
            "results/sglang_qwen_fp4kv_eager_only*.log",
            "results/sglang_qwen_fp4kv_autosafe*.json",
            "results/sglang_qwen_fp4kv_autosafe*.log",
            "results/sglang_qwen_fp4kv_autosafe*.md",
            "results/sglang_qwen_fp4kv_d7d931f_logprob_quality*.json",
            "results/sglang_qwen_fp4kv_d7d931f_logprob_quality*.md",
            "results/sglang_qwen_fp4kv_d7d931f_native_divergence*.json",
            "results/sglang_qwen_fp4kv_d7d931f_native_divergence*.md",
            "results/sglang_qwen_fp4kv_prompt_path_reconcile*.json",
            "results/sglang_qwen_fp4kv_prompt_path_reconcile*.md",
            "results/sglang_qwen_fp4kv_prompt_path_reconcile*.log",
            "results/sglang_nvfp4_kv_layout_probe*.json",
            "results/sglang_nvfp4_kv_convention_probe*.json",
            "results/sglang_fp4_pool_bridge_probe*.json",
            "results/sglang_fp4_kv_sm121_*.md",
        ),
        forbidden_claim_fragments=(
            "_patched_",
            "_nograph",
            "_nographs",
            "_startup",
            "_autosafe",
            "_d7d931f_matched_",
        ),
    ),
    EvidenceRequirement(
        name="sglang_dflash_or_eagle_qwen",
        runtime="sglang",
        model_family="qwen",
        title="SGLang DFlash/EAGLE Qwen counterpart",
        acceptance=(
            "SGLang Qwen-class speculative row records target/drafter, accepted-draft "
            "metrics, quality, and speed against a non-speculative comparator."
        ),
        claim_groups=(
            (
                "results/sglang*dflash*row_manifest.json",
                "results/sglang*eagle*row_manifest.json",
            ),
            (
                "results/sglang*nodflash*row_manifest.json",
                "results/sglang*baseline*row_manifest.json",
            ),
        ),
        partial_patterns=(
            "results/sglang*dflash*.md",
            "results/sglang*eagle*.md",
            "results/aeon_prior_art_audit_*.json",
        ),
    ),
    EvidenceRequirement(
        name="vllm_qwen36_nvfp4_dflash",
        runtime="vllm",
        model_family="qwen",
        title="vLLM Qwen3.6 NVFP4+DFlash local serving row",
        acceptance=(
            "AEON Qwen3.6 NVFP4+DFlash or matched jethac/vLLM row starts, serves, "
            "records backend/DFlash evidence, and captures a serving manifest."
        ),
        claim_groups=(
            (
                "results/*qwen36*dflash*row_manifest.json",
                "results/*qwen3.6*dflash*row_manifest.json",
            ),
            (
                "results/*qwen36*dflash*openai_benchmark.json",
                "results/*qwen3.6*dflash*openai_benchmark.json",
            ),
        ),
        partial_patterns=(
            "results/aeon_qwen36_dflash*_summary.md",
            "results/aeon_qwen36_dflash*_stop_point.md",
            "results/aeon_qwen36_dflash_tailnet*_row_manifest.json",
            "results/aeon_qwen36_dflash_tailnet*_server.log",
            "results/aeon_qwen36_dflash_tailnet*_nvfp4_checkpoint_audit.json",
            "results/vllm_aeon_qwen_patch_port_*.md",
        ),
        blocked_patterns=(
            "results/aeon_qwen36_dflash*_stop_point.md",
            "results/aeon_qwen36_dflash*_summary.md",
        ),
    ),
    EvidenceRequirement(
        name="vllm_qwen_nvfp4_kv_capacity",
        runtime="vllm",
        model_family="qwen",
        title="vLLM Qwen fp8-vs-NVFP4 KV capacity row",
        acceptance=(
            "vLLM serves the same Qwen-class model with matched fp8 and NVFP4 KV "
            "settings, logs FlashInfer FA2 NVFP4 KV selection, records KV capacity, "
            "quality, and serving manifests for both rows."
        ),
        claim_groups=(
            (
                "results/vllm*qwen*nvfp4*kv*row_manifest.json",
                "results/*qwen*nvfp4*kv*vllm*row_manifest.json",
            ),
            (
                "results/vllm*qwen*nvfp4*kv*openai_benchmark.json",
                "results/*qwen*nvfp4*kv*vllm*openai_benchmark.json",
            ),
            (
                "results/vllm*qwen*fp8*kv*row_manifest.json",
                "results/vllm*qwen*nvfp4*kv*fp8*row_manifest.json",
                "results/*qwen*fp8*kv*vllm*row_manifest.json",
            ),
        ),
        partial_patterns=(
            "results/vllm_nvfp4_sm12x_routing_probe_*.json",
            "results/flashinfer_nvfp4_kv_probe_*.json",
            "results/vllm_qwen_dflash_sm121a_patch_verify_*.md",
            "results/vllm_aeon_qwen_patch_port_*.md",
        ),
    ),
    EvidenceRequirement(
        name="llamacpp_larger_qwen_gguf",
        runtime="llamacpp",
        model_family="qwen",
        title="llama.cpp larger Qwen3/Qwen3.6 GGUF row",
        acceptance=(
            "llama.cpp serves a Qwen3/Qwen3.6-class GGUF with build-target evidence, "
            "llama-bench, OpenAI smoke, and row manifest."
        ),
        claim_groups=(
            (
                "results/llamacpp*qwen3*row_manifest.json",
                "results/llamacpp*qwen36*row_manifest.json",
            ),
            (
                "results/llamacpp*qwen3*llama_bench.txt",
                "results/llamacpp*qwen36*llama_bench.txt",
            ),
        ),
        partial_patterns=(
            "results/llamacpp_qwen25_1_5b_q4_k_m_*row_manifest.json",
            "results/llamacpp_qwen25_1_5b_q4_k_m_*openai_benchmark.json",
            "results/llamacpp_nvfp4_runtime_gate_*.md",
            "results/llamacpp_nvfp4_runtime_gate_*/*.json",
        ),
    ),
    EvidenceRequirement(
        name="llamacpp_native_fp4_gguf",
        runtime="llamacpp",
        model_family="qwen_or_gemma",
        title="llama.cpp native NVFP4/MXFP4 GGUF tensor-core proof",
        acceptance=(
            "NVFP4/MXFP4 GGUF model uses a native FP4 path with build-target and runtime "
            "dispatch proof, separate from Q4_0/Q4_K practical serving."
        ),
        claim_groups=(
            (
                "results/llamacpp*nvfp4*row_manifest.json",
                "results/llamacpp*mxfp4*row_manifest.json",
                "results/llamacpp*native_fp4*row_manifest.json",
            ),
            (
                "results/llamacpp*nvfp4*build_target_audit.json",
                "results/llamacpp*mxfp4*build_target_audit.json",
                "results/llamacpp*native_fp4*build_target_audit.json",
            ),
        ),
        partial_patterns=(
            "results/llamacpp*_server.log",
            "results/llamacpp*_build_target_audit.json",
            "results/llamacpp_nvfp4_runtime_gate_*.md",
            "results/llamacpp_nvfp4_runtime_gate_*/*.log",
            "results/llamacpp_nvfp4_runtime_gate_*/*.txt",
            "results/llamacpp_nvfp4_runtime_gate_*/*.json",
        ),
    ),
    EvidenceRequirement(
        name="llamacpp_live_loglikelihood",
        runtime="llamacpp",
        model_family="qwen_or_gemma",
        title="llama.cpp live native loglikelihood task proof",
        acceptance=(
            "Native llama.cpp loglikelihood probe and tiny JSONL task run against a live "
            "server and prove arbitrary continuation-token logprobs."
        ),
        claim_groups=(
            (
                "results/llamacpp_native_loglikelihood_probe_*.json",
                "results/llamacpp_native_loglikelihood_*_probe.json",
            ),
            (
                "results/llamacpp_native_loglikelihood_task_*.json",
                "results/llamacpp_native_loglikelihood_*_task.json",
            ),
        ),
        partial_patterns=(
            "results/llamacpp_native_loglikelihood_probe*_selftest*.json",
            "results/llamacpp_native_loglikelihood_task_dryrun*.json",
            "results/llamacpp_native_loglikelihood_*_summary.md",
            "results/*gguf_logprobs_probe*.json",
            "results/llamacpp_gguf_echo_logprobs_probe*.json",
        ),
    ),
)


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def match_patterns(root: Path, patterns: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        for raw in glob.glob(str(root / pattern)):
            matches.append(rel(Path(raw), root))
    return sorted(set(matches))


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def is_dry_run_or_selftest(path: str, root: Path) -> bool:
    lower = path.lower()
    if "dryrun" in lower or "dry-run" in lower or "selftest" in lower:
        return True
    data = load_json(root / path)
    if isinstance(data, dict) and data.get("dry_run"):
        return True
    return False


def is_failed_claim_artifact(path: str, root: Path) -> bool:
    data = load_json(root / path)
    if not isinstance(data, dict):
        return False
    return data.get("ok") is False


def filter_claim_matches(root: Path, matches: list[str]) -> list[str]:
    return [
        path
        for path in matches
        if not is_dry_run_or_selftest(path, root)
        and not is_failed_claim_artifact(path, root)
    ]


def audit_requirement(root: Path, req: EvidenceRequirement) -> dict[str, Any]:
    claim_groups: list[dict[str, Any]] = []
    all_claim_matches: list[str] = []
    all_ignored_matches: list[str] = []
    for group in req.claim_groups:
        raw_group_matches = match_patterns(root, group)
        filtered = filter_claim_matches(root, raw_group_matches)
        forbidden = [
            path
            for path in filtered
            if any(fragment in path for fragment in req.forbidden_claim_fragments)
        ]
        accepted = sorted(set(filtered) - set(forbidden))
        all_claim_matches.extend(accepted)
        all_ignored_matches.extend(sorted(set(raw_group_matches) - set(accepted)))
        claim_groups.append(
            {
                "patterns": list(group),
                "accepted_matches": accepted,
                "ignored_matches": sorted(set(raw_group_matches) - set(accepted)),
                "ok": bool(accepted),
            }
        )
    partial = match_patterns(root, req.partial_patterns)
    blocked = match_patterns(root, req.blocked_patterns)
    claim_ready = all(group["ok"] for group in claim_groups)
    if claim_ready:
        status = "claim-evidence-present"
    elif partial:
        status = "partial-evidence-only"
    elif blocked:
        status = "blocked-or-stop-point"
    else:
        status = "missing"
    return {
        "name": req.name,
        "runtime": req.runtime,
        "model_family": req.model_family,
        "title": req.title,
        "acceptance": req.acceptance,
        "status": status,
        "claim_ready": claim_ready,
        "claim_groups": claim_groups,
        "claim_matches": sorted(set(all_claim_matches)),
        "ignored_claim_matches": sorted(set(all_ignored_matches)),
        "forbidden_claim_fragments": list(req.forbidden_claim_fragments),
        "partial_matches": partial,
        "blocked_matches": blocked,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/counterpart_evidence_audit.json")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any counterpart is missing.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    requirements = [audit_requirement(root, req) for req in REQUIREMENTS]
    missing = [req["name"] for req in requirements if not req["claim_ready"]]
    summary = {
        "schema": "counterpart-evidence-audit/v1",
        "repo_root": rel(root, root),
        "requirement_count": len(requirements),
        "claim_ready_count": len(requirements) - len(missing),
        "missing_or_partial_count": len(missing),
        "all_claim_ready": not missing,
        "missing_or_partial": missing,
        "requirements": requirements,
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    output.write_text(text)
    print(text, end="")
    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
