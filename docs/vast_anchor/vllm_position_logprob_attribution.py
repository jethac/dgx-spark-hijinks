#!/usr/bin/env python3
"""Emit per-position supplied-token NLL for one vLLM KV-cache row.

This is a narrow attribution helper for the Gemma 4 26B-A4B NVFP4 calibration
work. The existing matched anchor reports only a mean NLL; after the K-refine
run proved non-monotonic, the next useful question is where the low-NLL
collapse lives.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def key_matches_token_id(key: Any, token_id: int) -> bool:
    if key == token_id:
        return True
    if isinstance(key, str):
        if key == str(token_id):
            return True
        if key.startswith("token_id:") and key.removeprefix("token_id:") == str(token_id):
            return True
    return False


def logprob_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    raw = getattr(value, "logprob", None)
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(value, dict):
        raw = value.get("logprob")
        if isinstance(raw, (int, float)):
            return float(raw)
    return None


def supplied_logprob(entry: Any, token_id: int) -> float | None:
    if entry is None or not isinstance(entry, dict):
        return None
    for key, value in entry.items():
        if key_matches_token_id(key, token_id):
            return logprob_value(value)
    return None


def bucket_stats(rows: list[dict[str, Any]], ranges: list[tuple[int, int]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_rel = {row["rel_index"]: row for row in rows if math.isfinite(row["nll"])}
    for lo, hi in ranges:
        vals = [by_rel[i]["nll"] for i in range(lo, hi) if i in by_rel]
        out.append(
            {
                "rel_range": [lo, hi],
                "count": len(vals),
                "mean_nll": statistics.fmean(vals) if vals else math.nan,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("ANCHOR_MODEL", "google/gemma-4-26B-A4B-it"))
    parser.add_argument("--tokenizer", default=os.environ.get("ANCHOR_TOKENIZER", "google/gemma-4-26B-A4B-it"))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kv-cache-dtype", required=True)
    parser.add_argument("--calib-json", default=None)
    parser.add_argument("--ctx", type=int, default=8185)
    parser.add_argument("--prefix-len", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--skip-mm-profiling", action="store_true")
    args = parser.parse_args()

    if args.calib_json:
        os.environ["VLLM_NVFP4_KV_CALIB"] = args.calib_json

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    text = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    token_ids = tokenizer.encode(text, add_special_tokens=False)[: args.ctx]
    if len(token_ids) < args.ctx:
        raise ValueError(f"corpus tokenized to {len(token_ids)} tokens, shorter than ctx={args.ctx}")

    llm = LLM(
        model=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        language_model_only=args.language_model_only,
        skip_mm_profiling=args.skip_mm_profiling,
    )

    warm_elapsed_s = 0.0
    if not args.skip_warmup:
        warm_started = time.perf_counter()
        llm.generate(token_ids[: args.prefix_len], SamplingParams(max_tokens=1, temperature=0.0))
        warm_elapsed_s = time.perf_counter() - warm_started

    score_started = time.perf_counter()
    scored = llm.generate(
        token_ids,
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1),
    )[0]
    score_elapsed_s = time.perf_counter() - score_started

    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    start_index = args.prefix_len + 1 if args.prefix_len > 0 else 1
    for index in range(start_index, len(token_ids)):
        lp = supplied_logprob(scored.prompt_logprobs[index], token_ids[index])
        if lp is None or not math.isfinite(lp):
            missing.append(index)
            nll = math.nan
        else:
            nll = -lp
        rows.append(
            {
                "index": index,
                "rel_index": index - start_index,
                "token_id": token_ids[index],
                "token": tokenizer.decode([token_ids[index]]),
                "logprob": lp,
                "nll": nll,
            }
        )

    vals = [row["nll"] for row in rows if math.isfinite(row["nll"])]
    if not vals:
        raise ValueError("no finite supplied-token logprobs captured")

    ranges = [(0, 256), (256, 1024), (1024, 2048), (2048, 3072), (3072, len(rows))]
    report = {
        "schema": "vllm-26b-position-logprob-attribution/v1",
        "model": args.model,
        "tokenizer": args.tokenizer,
        "kv_cache_dtype": args.kv_cache_dtype,
        "calib_json": args.calib_json,
        "ctx": args.ctx,
        "prefix_len": args.prefix_len,
        "score_start_index": start_index,
        "num_scored_tokens": len(vals),
        "num_missing_tokens": len(missing),
        "missing_positions_preview": missing[:20],
        "mean_nll_nats": statistics.fmean(vals),
        "ppl": math.exp(statistics.fmean(vals)) if statistics.fmean(vals) < 700 else float("inf"),
        "buckets": bucket_stats(rows, ranges),
        "run_mode": {
            "skip_warmup": bool(args.skip_warmup),
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "single_chunk": args.max_num_batched_tokens >= args.ctx,
            "reuse_path": not args.skip_warmup,
        },
        "timing_s": {
            "warm": warm_elapsed_s,
            "score": score_elapsed_s,
            "total": time.perf_counter() - started,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "positions": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "mean_nll_nats": report["mean_nll_nats"],
                "ppl": report["ppl"],
                "num_missing_tokens": len(missing),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
