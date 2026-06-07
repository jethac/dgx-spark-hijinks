#!/usr/bin/env python3
"""Microbenchmark FlashInfer NVFP4 mm_fp4 auto-dispatch on GB10.

This is a kernel-level diagnostic, not an end-to-end serving benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch.nn.functional as F
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
from flashinfer.gemm import gemm_base


CASE_PRESETS = {
    "smoke": ["1x128x128", "16x256x256", "64x512x512"],
    # Decode-like dense projections: small token count, model-sized N/K.
    "dense_decode": [
        "1x4096x4096",
        "4x4096x4096",
        "16x4096x4096",
        "1x8192x4096",
        "4x8192x4096",
        "16x8192x4096",
    ],
    # MoE-ish expert GEMMs: small per-expert token counts and wide FFN dimensions.
    "moe_expert": [
        "1x14336x4096",
        "4x14336x4096",
        "16x14336x4096",
        "1x4096x14336",
        "4x4096x14336",
        "16x4096x14336",
    ],
}


def parse_case(raw: str) -> tuple[int, int, int]:
    parts = raw.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"case must be MxNxK, got {raw!r}")
    try:
        m, n, k = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"case must be MxNxK, got {raw!r}") from exc
    if min(m, n, k) <= 0:
        raise argparse.ArgumentTypeError(f"case dimensions must be positive: {raw!r}")
    return m, n, k


def heuristic_order() -> list[str]:
    tensor = type("MockTensor", (), {"device": torch.device("cuda")})
    return gemm_base._heuristic_func_mm_fp4(
        ["cudnn", "cutlass", "b12x"],
        tensor(),
        tensor(),
        tensor(),
        tensor(),
        use_nvfp4=True,
    )


def bench_case(
    m: int,
    n: int,
    k: int,
    *,
    backend: str,
    iterations: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed + m + n + k)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    global_scale_a = (448 * 6) / a.float().abs().nan_to_num().max()
    global_scale_b = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_scale = nvfp4_quantize(
        a, global_scale_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_scale = nvfp4_quantize(
        b, global_scale_b, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = 1.0 / (global_scale_a * global_scale_b)
    reference = torch.mm(a, b.T)
    output = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)

    for _ in range(warmup):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_scale,
            b_scale.T,
            alpha,
            torch.bfloat16,
            output,
            block_size=16,
            backend=backend,
            use_nvfp4=True,
            skip_check=False,
        )
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(iterations):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_scale,
            b_scale.T,
            alpha,
            torch.bfloat16,
            output,
            block_size=16,
            backend=backend,
            use_nvfp4=True,
            skip_check=False,
        )
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started

    cosine = F.cosine_similarity(
        reference.reshape(-1).float(), output.reshape(-1).float(), dim=0
    ).item()
    return {
        "m": m,
        "n": n,
        "k": k,
        "backend": backend,
        "iterations": iterations,
        "warmup": warmup,
        "elapsed_s": elapsed_s,
        "mean_ms": elapsed_s * 1000 / iterations,
        "finite": bool(torch.isfinite(output).all().item()),
        "cosine_similarity_vs_bf16_torch_mm": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--container", default="unknown")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(CASE_PRESETS),
        help="Append a named case preset. Ignored when --case is provided.",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output")
    args = parser.parse_args()

    raw_cases = args.case
    if not raw_cases:
        preset_names = args.preset or ["smoke"]
        raw_cases = [
            raw_case
            for preset_name in preset_names
            for raw_case in CASE_PRESETS[preset_name]
        ]
    cases = [parse_case(raw) for raw in raw_cases]
    report: dict[str, Any] = {
        "schema": "flashinfer-mm-fp4-microbench/v1",
        "phase": args.phase,
        "run_id": args.run_id,
        "container": args.container,
        "presets": args.preset or (["smoke"] if not args.case else []),
        "raw_cases": raw_cases,
        "flashinfer_file": gemm_base.__file__,
        "flashinfer_version": __import__("flashinfer").__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "heuristic": heuristic_order(),
        "cases": [],
    }

    for case in cases:
        report["cases"].append(
            bench_case(
                *case,
                backend=args.backend,
                iterations=args.iterations,
                warmup=args.warmup,
                seed=args.seed,
            )
        )

    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
