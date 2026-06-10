#!/usr/bin/env python3
"""Block E3 A/B benchmark (claude_blockE23) for Gemma 4 E4B serving.

Measurement method copied from dgx-spark-hijinks scripts/openai_serving_benchmark.py
(streamed /v1/chat/completions; TTFT = first content delta; usage from
stream_options.include_usage), extended with the Block E required cases:

  a) short_decode : ~32-token prompt, 256 new tokens, batch 1 -> decode tok/s
  b) long_prefill : ~2048-token prompt, 64 new tokens -> TTFT + prefill tok/s
  c) concurrent4  : 4 simultaneous short_decode requests -> aggregate decode tok/s

3 repetitions per case, temperature 0, seed 0, ignore_eos so token counts are
fixed. Full request payloads are recorded in the output JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from typing import Any

SHORT_PROMPT = (
    "Explain briefly why a desktop AI workstation benefits from fast unified "
    "memory, covering bandwidth, latency, and capacity tradeoffs for local "
    "inference."
)  # ~32 prompt tokens once chat-templated

LONG_SENTENCE = (
    "The benchmark must capture environment metadata, backend selection, "
    "cold-start time, warm decode speed, memory pressure, output quality, "
    "and reproducibility of every serving row. "
)
LONG_PROMPT = (
    "You are benchmarking a local inference server. Summarize the following "
    "requirements into five bullets. " + LONG_SENTENCE * 64
)  # ~2048 prompt tokens (actual prompt_tokens taken from usage)


_NONCE_COUNTER = [0]


def make_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    # unique per-request nonce prefix defeats vLLM's automatic prefix caching,
    # so every repetition pays the full prefill (v2 fix: v1 medians were
    # contaminated by prefix-cache hits on repeated identical prompts)
    _NONCE_COUNTER[0] += 1
    nonce = f"[request nonce {_NONCE_COUNTER[0]:04d}] "
    return {
        "model": model,
        "messages": [{"role": "user", "content": nonce + prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def stream_chat(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    first_token_s = None
    content_parts: list[str] = []
    usage = None
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_text = line[5:].strip()
            if data_text == "[DONE]":
                break
            try:
                event = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - started
                content_parts.append(text)
    total_s = time.perf_counter() - started
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    decode_s = None
    decode_tok_s = None
    prefill_tok_s = None
    if total_s is not None and first_token_s is not None:
        decode_s = max(total_s - first_token_s, 0.0)
    if completion_tokens and decode_s and decode_s > 0:
        # first token is produced AT ttft, so decode window covers the
        # remaining completion_tokens - 1 tokens
        decode_tok_s = (completion_tokens - 1) / decode_s
    if prompt_tokens and first_token_s and first_token_s > 0:
        prefill_tok_s = prompt_tokens / first_token_s
    return {
        "started_offset_s": None,  # filled by caller for concurrent case
        "ttft_s": first_token_s,
        "total_s": total_s,
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "content_preview": "".join(content_parts)[:400],
    }


def run_concurrent(url: str, payloads: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
    n = len(payloads)
    results: list[dict[str, Any] | None] = [None] * n
    errors: list[str | None] = [None] * n
    barrier = threading.Barrier(n + 1)
    ends: list[float | None] = [None] * n

    def worker(i: int) -> None:
        barrier.wait()
        try:
            results[i] = stream_chat(url, payloads[i], timeout)
        except Exception as exc:  # noqa: BLE001
            errors[i] = repr(exc)
        ends[i] = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    barrier.wait()
    batch_start = time.perf_counter()
    for t in threads:
        t.join()
    batch_wall_s = max(e for e in ends if e is not None) - batch_start
    ok = [r for r in results if r]
    total_completion = sum(r["completion_tokens"] or 0 for r in ok)
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"]]
    # aggregate decode tok/s: total completion tokens divided by the decode
    # window of the batch (batch wall minus the earliest TTFT)
    agg_total_tok_s = total_completion / batch_wall_s if batch_wall_s > 0 else None
    decode_window = batch_wall_s - min(ttfts) if ttfts else None
    agg_decode_tok_s = (
        (total_completion - len(ok)) / decode_window
        if decode_window and decode_window > 0
        else None
    )
    return {
        "batch_wall_s": batch_wall_s,
        "total_completion_tokens": total_completion,
        "aggregate_tok_s_incl_prefill": agg_total_tok_s,
        "aggregate_decode_tok_s": agg_decode_tok_s,
        "per_request": results,
        "errors": [e for e in errors if e],
    }


def median_of(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return statistics.median(clean) if clean else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase", required=True)  # before / after
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"

    report: dict[str, Any] = {
        "schema": "blockE23-benchmark/v2",
        "run_id": args.run_id,
        "phase": args.phase,
        "model": args.model,
        "url": args.url,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payloads": {
            "short_decode": make_payload(args.model, SHORT_PROMPT, 256),
            "long_prefill": make_payload(args.model, LONG_PROMPT, 64),
            "concurrent4": {
                "note": "4 simultaneous short_decode payloads, unique nonces"
            },
            "nonce_note": (
                "every real request gets a fresh '[request nonce NNNN] ' "
                "prefix to defeat prefix caching; these recorded payloads "
                "are templates (nonces 0001/0002)"
            ),
        },
        "cases": {},
    }

    # warmup (one short request, not counted)
    try:
        warm = stream_chat(endpoint, make_payload(args.model, SHORT_PROMPT, 16), args.timeout)
        report["warmup"] = {"ok": True, "ttft_s": warm["ttft_s"], "total_s": warm["total_s"]}
    except Exception as exc:  # noqa: BLE001
        report["warmup"] = {"ok": False, "error": repr(exc)}

    # (a) short decode
    reps = []
    for _ in range(args.reps):
        try:
            reps.append(
                stream_chat(endpoint, make_payload(args.model, SHORT_PROMPT, 256), args.timeout)
            )
        except Exception as exc:  # noqa: BLE001
            reps.append({"error": repr(exc)})
    report["cases"]["short_decode"] = {
        "reps": reps,
        "median_decode_tok_s": median_of([r.get("decode_tok_s") for r in reps]),
        "median_ttft_s": median_of([r.get("ttft_s") for r in reps]),
    }

    # (b) long prefill
    reps = []
    for _ in range(args.reps):
        try:
            reps.append(
                stream_chat(endpoint, make_payload(args.model, LONG_PROMPT, 64), args.timeout)
            )
        except Exception as exc:  # noqa: BLE001
            reps.append({"error": repr(exc)})
    report["cases"]["long_prefill"] = {
        "reps": reps,
        "median_ttft_s": median_of([r.get("ttft_s") for r in reps]),
        "median_prefill_tok_s": median_of([r.get("prefill_tok_s") for r in reps]),
    }

    # (c) concurrent 4x short decode
    reps = []
    for _ in range(args.reps):
        try:
            payloads = [make_payload(args.model, SHORT_PROMPT, 256) for _ in range(4)]
            reps.append(run_concurrent(endpoint, payloads, args.timeout))
        except Exception as exc:  # noqa: BLE001
            reps.append({"error": repr(exc)})
    report["cases"]["concurrent4"] = {
        "reps": reps,
        "median_aggregate_decode_tok_s": median_of(
            [r.get("aggregate_decode_tok_s") for r in reps]
        ),
    }

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(report, indent=2, sort_keys=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
