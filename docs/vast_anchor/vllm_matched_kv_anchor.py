#!/usr/bin/env python3
"""Offline vLLM matched bf16-vs-NVFP4 KV anchor harness.

The script intentionally mirrors the SGLang supplied-token PPL row:

* token IDs come from the same text corpus;
* a prefix request warms the prefix cache;
* the full request scores supplied prompt tokens;
* position `prefix_len` is treated as a boundary and not scored.

It also runs a chat-template smoke because Gemma-4 `-it` raw prompts are invalid
coherence controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_token_ids(tokenizer: Any, text: str, ctx: int) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < ctx:
        raise ValueError(f"corpus tokenized to {len(token_ids)} tokens, shorter than ctx={ctx}")
    return token_ids[:ctx]


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


def score_prompt_logprobs(
    prompt_logprobs: list[Any],
    token_ids: list[int],
    *,
    prefix_len: int,
) -> dict[str, Any]:
    if len(prompt_logprobs) != len(token_ids):
        raise ValueError(
            f"prompt_logprobs length mismatch: {len(prompt_logprobs)} vs {len(token_ids)}"
        )

    scored: list[float] = []
    missing: list[int] = []
    # Match the SGLang row: score starts after the cached-prefix boundary token.
    start_index = prefix_len + 1 if prefix_len > 0 else 1
    for index in range(start_index, len(token_ids)):
        lp = supplied_logprob(prompt_logprobs[index], token_ids[index])
        if lp is None or not math.isfinite(lp):
            missing.append(index)
            continue
        scored.append(lp)
    if not scored:
        raise ValueError("no supplied prompt tokens had recoverable logprobs")
    mean_logprob = sum(scored) / len(scored)
    mean_nll = -mean_logprob
    return {
        "ok": not missing,
        "score_start_index": start_index,
        "num_prompt_tokens": len(token_ids),
        "num_scored_tokens": len(scored),
        "num_missing_tokens": len(missing),
        "missing_positions_preview": missing[:20],
        "mean_logprob": mean_logprob,
        "mean_nll_nats": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
    }


def chat_smoke(llm: LLM, tokenizer: Any, model: str) -> dict[str, Any]:
    prompt = os.environ.get(
        "CHAT_PROMPT", "What is the capital of Japan? Answer in one word."
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    output = llm.generate(
        rendered,
        SamplingParams(max_tokens=8, temperature=0.0),
    )[0]
    text = output.outputs[0].text
    return {
        "model": model,
        "prompt": prompt,
        "generated": text,
        "contains_tokyo_or_paris": ("Tokyo" in text) or ("Paris" in text),
    }


def build_llm(args: argparse.Namespace) -> LLM:
    kwargs = {
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": True,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    return LLM(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("ANCHOR_MODEL", "google/gemma-4-12B-it"))
    parser.add_argument("--tokenizer", default=os.environ.get("ANCHOR_TOKENIZER", "google/gemma-4-12B-it"))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--kv-cache-dtype", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ctx", type=int, default=8185)
    parser.add_argument("--prefix-len", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    # Decisive knob for the +0.40 root-cause: when set, do NOT warm the prefix.
    # The full ctx is scored in one request, so scored positions attend to prefix
    # KV computed in-pass (single-chunk if max_num_batched_tokens>=ctx) instead of
    # the cross-request radix/partial-state-merge path. Scored token set is identical.
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    text = read_text(Path(args.corpus))
    token_ids = load_token_ids(tokenizer, text, args.ctx)
    llm = build_llm(args)

    chat = chat_smoke(llm, tokenizer, args.model)

    warm_started = time.perf_counter()
    if args.skip_warmup:
        warm = None
        warm_elapsed_s = 0.0
    else:
        warm = llm.generate(
            token_ids[: args.prefix_len],
            SamplingParams(max_tokens=1, temperature=0.0),
        )[0]
    warm_elapsed_s = time.perf_counter() - warm_started

    score_started = time.perf_counter()
    scored = llm.generate(
        token_ids,
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1),
    )[0]
    score_elapsed_s = time.perf_counter() - score_started
    score = score_prompt_logprobs(
        scored.prompt_logprobs,
        token_ids,
        prefix_len=args.prefix_len,
    )

    report = {
        "schema": "vllm-matched-kv-anchor/v1",
        "ok": bool(score["ok"] and chat["contains_tokyo_or_paris"]),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "kv_cache_dtype": args.kv_cache_dtype,
        "ctx": args.ctx,
        "prefix_len": args.prefix_len,
        "run_mode": {
            "skip_warmup": bool(args.skip_warmup),
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "single_chunk": args.max_num_batched_tokens >= args.ctx,
            "reuse_path": (not args.skip_warmup),
        },
        "corpus": args.corpus,
        "text_chars": len(text),
        "available_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
        "env": {
            key: os.environ.get(key)
            for key in [
                "VLLM_ATTENTION_BACKEND",
                "VLLM_FLASHINFER_BF16_GEMMA",
                "VLLM_FLASHINFER_VOSPLIT",
                "VLLM_NVFP4_KV_VOSPLIT",
                "VLLM_NVFP4_KV_LINEAR_V_SF",
                "PYTHONPATH",
            ]
            if os.environ.get(key)
        },
        "chat_smoke": chat,
        "warmup": {
            "skipped": bool(args.skip_warmup),
            "elapsed_s": warm_elapsed_s,
            "prompt_tokens": 0 if warm is None else args.prefix_len,
            "generated": None if warm is None else warm.outputs[0].text,
        },
        "score_request": {
            "elapsed_s": score_elapsed_s,
            "prompt_tokens": len(token_ids),
            "generated": scored.outputs[0].text,
        },
        "score": score,
        "elapsed_s": time.perf_counter() - started,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
