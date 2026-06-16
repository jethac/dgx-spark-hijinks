#!/usr/bin/env python3
"""Needle-in-haystack long-context retrieval eval for KV-cache quality.

KV-cache quantization damage shows up on RETRIEVAL (recalling a fact buried in long context),
not next-token perplexity. This plants a unique access code at varying depths in an ~N-token
haystack and measures whether the model can read it back. A KV dtype that holds retrieval as
well as bf16/fp8 is genuinely fine; one that quietly drops it (even at mild PPL) is the
deceptive below-truth collapse. Run once per KV config on the SAME seeded prompt set.

Output JSON: per-depth and overall retrieval accuracy.
"""
import argparse
import json
import os
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_prompts(tokenizer, filler_text, trials, ctx_tokens, num_needles, seed=0):
    """Plant `num_needles` distinct codes at evenly-spread depths in an ~ctx_tokens haystack
    and ask for all of them (multi-needle = the hard RULER variant; num_needles=1 is single)."""
    rng = random.Random(seed)
    filler_ids = tokenizer.encode(filler_text)
    while len(filler_ids) < ctx_tokens + 512:
        filler_ids = filler_ids + filler_ids
    items = []
    for _ in range(trials):
        base = filler_ids[:ctx_tokens]
        lockers, codes, seen = [], [], set()
        while len(lockers) < num_needles:
            L = rng.randint(100, 999)
            if L in seen:
                continue
            seen.add(L)
            lockers.append(L)
            codes.append(rng.randint(100000, 999999))
        # evenly-spread insertion positions (depth (i+1)/(N+1))
        positions = [int(len(base) * (i + 1) / (num_needles + 1)) for i in range(num_needles)]
        text = ""
        prev = 0
        for pos, L, C in zip(positions, lockers, codes):
            text += tokenizer.decode(base[prev:pos]) + f" The access code for locker {L} is {C}. "
            prev = pos
        text += tokenizer.decode(base[prev:])
        lockq = ", ".join(str(L) for L in lockers)
        q = (
            f"\n\nQuestion: What are the access codes for lockers {lockq}? "
            "List each locker and its code.\nAnswer:"
        )
        items.append({"codes": [str(c) for c in codes], "num": num_needles, "prompt": text + q})
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--kv-cache-dtype", required=True)
    p.add_argument("--kv-cache-dtype-skip-layers", nargs="+", default=None)
    p.add_argument("--ctx-tokens", type=int, default=7200)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--num-needles", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    filler = open(args.corpus, encoding="utf-8").read()
    items = build_prompts(tok, filler, args.trials, args.ctx_tokens, args.num_needles)

    kwargs = dict(model=args.model, kv_cache_dtype=args.kv_cache_dtype,
                  max_model_len=args.max_model_len, enforce_eager=True,
                  trust_remote_code=True, enable_prefix_caching=False,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_num_batched_tokens=args.max_model_len)
    if args.kv_cache_dtype_skip_layers:
        kwargs["kv_cache_dtype_skip_layers"] = args.kv_cache_dtype_skip_layers
    llm = LLM(**kwargs)

    outs = llm.generate([it["prompt"] for it in items],
                        SamplingParams(temperature=0.0, max_tokens=24 + 14 * args.num_needles))
    total_found = total_codes = full_hits = 0
    for it, o in zip(items, outs):
        gen = o.outputs[0].text
        found = sum(c in gen for c in it["codes"])
        total_found += found
        total_codes += len(it["codes"])
        full_hits += int(found == len(it["codes"]))
    report = {
        "label": args.label,
        "kv_cache_dtype": args.kv_cache_dtype,
        "ctx_tokens": args.ctx_tokens,
        "num_needles": args.num_needles,
        "n_queries": len(items),
        "needle_recall": round(total_found / total_codes, 3),
        "all_needles_acc": round(full_hits / len(items), 3),
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
