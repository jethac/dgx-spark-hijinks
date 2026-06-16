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


def build_prompts(tokenizer, filler_text, depths, trials, ctx_tokens, seed=0):
    rng = random.Random(seed)
    filler_ids = tokenizer.encode(filler_text)
    # repeat filler to reach ctx if short
    while len(filler_ids) < ctx_tokens + 256:
        filler_ids = filler_ids + filler_ids
    items = []
    for depth in depths:
        for _ in range(trials):
            locker = rng.randint(100, 999)
            code = rng.randint(100000, 999999)
            needle = f" The secret access code for locker {locker} is {code}. "
            base = filler_ids[:ctx_tokens]
            split = int(len(base) * depth)
            pre = tokenizer.decode(base[:split])
            post = tokenizer.decode(base[split:])
            q = (
                f"\n\nQuestion: What is the secret access code for locker {locker}? "
                "Answer with only the 6-digit number.\nAnswer:"
            )
            items.append({"depth": depth, "locker": locker, "code": str(code),
                          "prompt": pre + needle + post + q})
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
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    filler = open(args.corpus, encoding="utf-8").read()
    depths = [0.05, 0.25, 0.5, 0.75, 0.95]
    items = build_prompts(tok, filler, depths, args.trials, args.ctx_tokens)

    kwargs = dict(model=args.model, kv_cache_dtype=args.kv_cache_dtype,
                  max_model_len=args.max_model_len, enforce_eager=True,
                  trust_remote_code=True, enable_prefix_caching=False,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_num_batched_tokens=args.max_model_len)
    if args.kv_cache_dtype_skip_layers:
        kwargs["kv_cache_dtype_skip_layers"] = args.kv_cache_dtype_skip_layers
    llm = LLM(**kwargs)

    outs = llm.generate([it["prompt"] for it in items],
                        SamplingParams(temperature=0.0, max_tokens=16))
    per_depth = {}
    n_ok = 0
    for it, o in zip(items, outs):
        gen = o.outputs[0].text
        ok = it["code"] in gen
        n_ok += ok
        d = per_depth.setdefault(it["depth"], [0, 0])
        d[0] += ok
        d[1] += 1
    report = {
        "label": args.label,
        "kv_cache_dtype": args.kv_cache_dtype,
        "ctx_tokens": args.ctx_tokens,
        "n": len(items),
        "overall_accuracy": n_ok / len(items),
        "per_depth_accuracy": {str(k): round(v[0] / v[1], 3) for k, v in sorted(per_depth.items())},
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
