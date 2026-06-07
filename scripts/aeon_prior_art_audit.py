#!/usr/bin/env python3
"""Audit local submodules for AEON prior-art port markers."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TextCheck:
    name: str
    path: str
    markers: tuple[str, ...]
    reason: str


VLLM_CHECKS = [
    TextCheck(
        name="vllm_lazy_stable_libtorch_import",
        path="third_party/vllm/vllm/platforms/cuda.py",
        markers=("RTLD_LAZY", "_C_stable_libtorch", "import_module"),
        reason="AEON Qwen patch: lazy import around unused SM100-only symbols on SM12x.",
    ),
    TextCheck(
        name="vllm_qwen3_5_text_registry",
        path="third_party/vllm/vllm/model_executor/models/registry.py",
        markers=("Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"),
        reason="AEON Qwen patch: register text-only Qwen3.5/Qwen3.6 model classes.",
    ),
    TextCheck(
        name="vllm_spec_decode_cudagraph_alignment",
        path="third_party/vllm/vllm/config/compilation.py",
        markers=(
            "cudagraph_mode != CUDAGraphMode.NONE",
            "uniform_decode_query_len > 1",
            "adjust_cudagraph_sizes_for_spec_decode",
        ),
        reason="AEON Qwen patch: align capture sizes for speculative decode beyond FULL mode.",
    ),
    TextCheck(
        name="vllm_mamba_block_size_fallback",
        path="third_party/vllm/vllm/model_executor/layers/mamba/abstract.py",
        markers=("mamba_block_size is None", "vllm_config.cache_config.block_size or 16"),
        reason="AEON Qwen patch: keep hybrid/Mamba cache arithmetic from seeing block_size=None.",
    ),
    TextCheck(
        name="vllm_hybrid_kv_block_size_none_safe",
        path="third_party/vllm/vllm/v1/worker/gpu_model_runner.py",
        markers=("if block_size is None", "else cdiv(max_model_len"),
        reason="AEON Qwen patch: guard hybrid linear-attention groups that do not use KV blocks.",
    ),
]


SGLANG_CHECKS = [
    TextCheck(
        name="sglang_dflash_algorithm_hook",
        path="third_party/sglang/python/sglang/srt/arg_groups/speculative_hook.py",
        markers=("speculative_algorithm == \"DFLASH\"", "def _handle_dflash"),
        reason="SGLang counterpart surface for AEON's DFlash speed lane.",
    ),
    TextCheck(
        name="sglang_dflash_model",
        path="third_party/sglang/python/sglang/srt/models/dflash.py",
        markers=("class DFlashDraftModel", "DFlashAttention"),
        reason="SGLang native DFlash draft-model surface exists; serving proof remains separate.",
    ),
    TextCheck(
        name="sglang_qwen_dflash_capture_hooks",
        path="third_party/sglang/python/sglang/srt/models/qwen3_moe.py",
        markers=("set_dflash_layers_to_capture",),
        reason="Qwen MoE model exposes DFlash capture hooks for counterpart testing.",
    ),
    TextCheck(
        name="sglang_speculative_acceptance_metrics",
        path="third_party/sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py",
        markers=("spec_accept_length", "spec_accept_rate"),
        reason="Accepted-draft metrics are needed to judge DFlash/EAGLE counterpart quality.",
    ),
]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def git_ref(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--short"],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"exists": True, "error": str(exc)}
    return {"exists": True, "branch": branch, "commit": commit, "dirty": dirty}


def run_check(check: TextCheck, repo_root: Path) -> dict[str, Any]:
    path = repo_root / check.path
    record: dict[str, Any] = {
        "name": check.name,
        "path": check.path,
        "reason": check.reason,
        "markers": list(check.markers),
        "exists": path.exists(),
        "ok": False,
    }
    if not path.exists():
        record["missing_markers"] = list(check.markers)
        return record
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [marker for marker in check.markers if marker not in text]
    record["missing_markers"] = missing
    record["ok"] = not missing
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/aeon_prior_art_audit.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    vllm_checks = [run_check(check, repo_root) for check in VLLM_CHECKS]
    sglang_checks = [run_check(check, repo_root) for check in SGLANG_CHECKS]
    summary = {
        "schema": "aeon-prior-art-audit/v1",
        "repo_root": rel(repo_root, repo_root),
        "submodules": {
            "vllm": git_ref(repo_root / "third_party" / "vllm"),
            "sglang": git_ref(repo_root / "third_party" / "sglang"),
            "flashinfer": git_ref(repo_root / "third_party" / "flashinfer"),
        },
        "vllm_ports": vllm_checks,
        "sglang_counterpart_surfaces": sglang_checks,
    }
    all_checks = vllm_checks + sglang_checks
    summary["ok"] = all(check["ok"] for check in all_checks)
    summary["failed"] = [check["name"] for check in all_checks if not check["ok"]]

    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
