#!/usr/bin/env python3
"""Capture prompt top-logprobs for one vLLM KV-cache row.

This is a readout-side discriminator for the Gemma 4 26B-A4B NVFP4 work.  The
attention reader and per-position target NLL have already been localized; this
helper records the visible logit distribution proxy that vLLM exposes through
``prompt_logprobs`` so bf16 and NVFP4 rows can be compared at the same supplied
positions without rebuilding the wheel.
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


def decoded_value(value: Any, tokenizer: Any, token_id: int) -> str:
    decoded = getattr(value, "decoded_token", None)
    if isinstance(decoded, str):
        return decoded
    if isinstance(value, dict):
        decoded = value.get("decoded_token") or value.get("token")
        if isinstance(decoded, str):
            return decoded
    return tokenizer.decode([token_id])


def key_to_token_id(key: Any) -> int | None:
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        if key.startswith("token_id:"):
            key = key.removeprefix("token_id:")
        try:
            return int(key)
        except ValueError:
            return None
    return None


def supplied_logprob(entry: Any, token_id: int) -> float | None:
    if entry is None or not isinstance(entry, dict):
        return None
    for key, value in entry.items():
        if key_matches_token_id(key, token_id):
            return logprob_value(value)
    return None


def top_entries(entry: Any, tokenizer: Any, limit: int) -> list[dict[str, Any]]:
    if entry is None or not isinstance(entry, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key, value in entry.items():
        token_id = key_to_token_id(key)
        logprob = logprob_value(value)
        if token_id is None or logprob is None or not math.isfinite(logprob):
            continue
        rows.append(
            {
                "token_id": token_id,
                "token": decoded_value(value, tokenizer, token_id),
                "logprob": logprob,
            }
        )
    rows.sort(key=lambda row: row["logprob"], reverse=True)
    return rows[:limit]


def should_keep(rel_index: int, stride: int, dense_prefix: int) -> bool:
    return rel_index < dense_prefix or (stride > 0 and rel_index % stride == 0)


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
    parser.add_argument("--prompt-logprobs", type=int, default=20)
    parser.add_argument("--keep-topk", type=int, default=20)
    parser.add_argument("--position-stride", type=int, default=16)
    parser.add_argument("--dense-prefix-positions", type=int, default=256)
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
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=args.prompt_logprobs),
    )[0]
    score_elapsed_s = time.perf_counter() - score_started

    start_index = args.prefix_len + 1 if args.prefix_len > 0 else 1
    rows: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    missing: list[int] = []
    for index in range(start_index, len(token_ids)):
        entry = scored.prompt_logprobs[index]
        target_lp = supplied_logprob(entry, token_ids[index])
        if target_lp is None or not math.isfinite(target_lp):
            missing.append(index)
            nll = math.nan
        else:
            nll = -target_lp
        top = top_entries(entry, tokenizer, args.keep_topk)
        top1 = top[0] if top else None
        topk_mass = sum(math.exp(item["logprob"]) for item in top)
        row = {
            "index": index,
            "rel_index": index - start_index,
            "token_id": token_ids[index],
            "token": tokenizer.decode([token_ids[index]]),
            "target_logprob": target_lp,
            "target_nll": nll,
            "top1_token_id": top1["token_id"] if top1 else None,
            "top1_token": top1["token"] if top1 else None,
            "top1_logprob": top1["logprob"] if top1 else None,
            "topk_mass": topk_mass,
        }
        rows.append(row)
        if should_keep(row["rel_index"], args.position_stride, args.dense_prefix_positions):
            kept.append({**row, "top": top})

    vals = [row["target_nll"] for row in rows if math.isfinite(row["target_nll"])]
    if not vals:
        raise ValueError("no finite supplied-token logprobs captured")

    report = {
        "schema": "vllm-26b-toplogprob-attribution/v1",
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
        "prompt_logprobs": args.prompt_logprobs,
        "keep_topk": args.keep_topk,
        "position_stride": args.position_stride,
        "dense_prefix_positions": args.dense_prefix_positions,
        "mean_nll_nats": statistics.fmean(vals),
        "ppl": math.exp(statistics.fmean(vals)) if statistics.fmean(vals) < 700 else float("inf"),
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
        "position_summaries": rows,
        "sampled_toplogprobs": kept,
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
                "sampled_positions": len(kept),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
