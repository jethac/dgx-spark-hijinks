# SPDX-License-Identifier: Apache-2.0
"""MTP/spec-decode greedy identity runner (zero-bug bar gate).

Runs ONE offline vLLM configuration (spec decode on or off) over a fixed
greedy prompt set and banks outputs + timing + spec metrics as JSON.
A `compare` mode string-compares two banked runs: greedy spec decode must
be OUTPUT-IDENTICAL to non-spec greedy at temp 0, or the row is RED.

Usage (run):
  python mtp_identity_run.py run \
      --target google/gemma-4-E2B-it \
      [--spec-config '{"model": "google/gemma-4-E2B-it-assistant",
                       "num_speculative_tokens": 3}'] \
      [--kv-cache-dtype nvfp4] [--max-model-len 4096] \
      [--gpu-memory-utilization 0.80] [--max-tokens 128] \
      --out results/p520_mtp_<date>/e2b_bf16_specoff.json

Usage (compare):
  python mtp_identity_run.py compare --baseline specoff.json --spec specon.json \
      --out identity_verdict.json

Windows/WSL note: always encoding='utf-8'.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

PROMPTS = [
    "Explain the difference between a process and a thread in one paragraph.",
    "Write a Python function that returns the n-th Fibonacci number "
    "iteratively, with a docstring.",
    "List the planets of the solar system in order from the Sun, "
    "one per line, with one fact each.",
    "Translate to Japanese, then back to English, the sentence: "
    "'The quick brown fox jumps over the lazy dog.' Show both steps.",
    "What is 137 * 24? Show the multiplication steps, then the result.",
    "Summarize the plot of Romeo and Juliet in exactly three sentences.",
    # Long prompt: cross several KV pages before the first generated token.
    "You are reviewing a design document. "
    + "The system ingests telemetry from edge devices, batches it, and "
    "writes it to object storage; a downstream job compacts the batches "
    "hourly and updates a serving index. " * 12
    + "Identify the two most likely failure modes and propose one "
    "mitigation for each.",
    "Repeat the word 'token' exactly five times, separated by commas, "
    "then stop.",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    spec_config = json.loads(args.spec_config) if args.spec_config else None

    llm_kwargs = dict(
        model=args.target,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=args.enforce_eager,
    )
    if args.kv_cache_dtype_skip_layers:
        llm_kwargs["kv_cache_dtype_skip_layers"] = (
            args.kv_cache_dtype_skip_layers.split(",")
        )
    if spec_config is not None:
        llm_kwargs["speculative_config"] = spec_config

    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - t0

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    # Warmup (excluded from timing; outputs discarded).
    llm.generate([PROMPTS[0]], SamplingParams(temperature=0.0, max_tokens=8))

    t1 = time.perf_counter()
    outputs = llm.generate(PROMPTS, sp)
    gen_s = time.perf_counter() - t1

    rows = []
    total_tokens = 0
    for prompt, out in zip(PROMPTS, outputs):
        comp = out.outputs[0]
        total_tokens += len(comp.token_ids)
        rows.append(
            {
                "prompt": prompt,
                "text": comp.text,
                "token_ids": list(comp.token_ids),
                "finish_reason": comp.finish_reason,
            }
        )

    metrics = {}
    try:
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            if "spec" in name or "draft" in name or "accept" in name:
                metrics[name] = repr(m)
    except Exception as exc:  # noqa: BLE001 - bank whatever we can
        metrics["_error"] = repr(exc)

    result = {
        "schema": "spark-mtp-identity-run/v1",
        "created_utc": _now(),
        "target": args.target,
        "spec_config": spec_config,
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_cache_dtype_skip_layers": args.kv_cache_dtype_skip_layers,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "env_knobs": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_NVFP4", "VLLM_FLASHINFER", "VLLM_ATTENTION"))
        },
        "load_seconds": load_s,
        "generate_seconds": gen_s,
        "total_generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / gen_s if gen_s > 0 else None,
        "spec_metrics": metrics,
        "outputs": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"banked {args.out}: {total_tokens} tokens in {gen_s:.1f}s")
    return 0


def compare(args: argparse.Namespace) -> int:
    with open(args.baseline, encoding="utf-8") as f:
        base = json.load(f)
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    mismatches = []
    for i, (b, s) in enumerate(zip(base["outputs"], spec["outputs"])):
        ids_equal = b["token_ids"] == s["token_ids"]
        text_equal = b["text"] == s["text"]
        if not (ids_equal and text_equal):
            div = next(
                (
                    j
                    for j, (x, y) in enumerate(zip(b["token_ids"], s["token_ids"]))
                    if x != y
                ),
                min(len(b["token_ids"]), len(s["token_ids"])),
            )
            mismatches.append(
                {
                    "prompt_index": i,
                    "prompt": b["prompt"][:120],
                    "first_divergence_token_index": div,
                    "baseline_ids_around": b["token_ids"][max(0, div - 3) : div + 4],
                    "spec_ids_around": s["token_ids"][max(0, div - 3) : div + 4],
                    "baseline_text": b["text"],
                    "spec_text": s["text"],
                }
            )

    speedup = None
    if base.get("tokens_per_second") and spec.get("tokens_per_second"):
        speedup = spec["tokens_per_second"] / base["tokens_per_second"]

    verdict = {
        "schema": "spark-mtp-identity-verdict/v1",
        "created_utc": _now(),
        "baseline_file": os.path.abspath(args.baseline),
        "spec_file": os.path.abspath(args.spec),
        "num_prompts": len(base["outputs"]),
        "identical": not mismatches,
        "verdict": "GREEN" if not mismatches else "RED",
        "mismatches": mismatches,
        "baseline_tokens_per_second": base.get("tokens_per_second"),
        "spec_tokens_per_second": spec.get("tokens_per_second"),
        "speedup": speedup,
        "spec_metrics": spec.get("spec_metrics"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(
        f"verdict: {verdict['verdict']} "
        f"({len(mismatches)} mismatching prompts of {verdict['num_prompts']}), "
        f"speedup={speedup}"
    )
    return 0 if not mismatches else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    rp = sub.add_parser("run")
    rp.add_argument("--target", required=True)
    rp.add_argument("--spec-config", default=None)
    rp.add_argument("--kv-cache-dtype", default="auto")
    rp.add_argument("--kv-cache-dtype-skip-layers", default=None)
    rp.add_argument("--max-model-len", type=int, default=4096)
    rp.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    rp.add_argument("--max-tokens", type=int, default=128)
    rp.add_argument("--enforce-eager", action="store_true")
    rp.add_argument("--out", required=True)
    rp.set_defaults(fn=run)

    cp = sub.add_parser("compare")
    cp.add_argument("--baseline", required=True)
    cp.add_argument("--spec", required=True)
    cp.add_argument("--out", required=True)
    cp.set_defaults(fn=compare)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
