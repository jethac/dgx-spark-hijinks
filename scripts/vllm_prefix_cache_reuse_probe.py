#!/usr/bin/env python3
"""Probe vLLM prefix-cache reuse quality through the OpenAI chat API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any


PROMPTS = {
    "medium_decode": (
        "You are validating local inference cache correctness. "
        "Reply with exactly one concise markdown bullet that starts with two asterisks, "
        "then says why a deterministic prefix-cache hit should preserve the first token. "
        "Keep the wording stable."
    ),
    "long_shared_prefix": (
        "You are validating local inference cache correctness. "
        "The following repeated context is intentionally identical across requests so the "
        "server can reuse prefix KV cache blocks. "
        + (
            "A correct prefix-cache implementation must preserve logits when it reuses "
            "cached K and V blocks instead of recomputing the prompt. "
        )
        * 240
        + "Now reply with exactly one concise markdown bullet that starts with two asterisks."
    ),
}


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def first_choice_token(choice: dict[str, Any]) -> dict[str, Any]:
    message = choice.get("message") or {}
    content = message.get("content") or ""
    logprobs = choice.get("logprobs") or {}
    content_logprobs = logprobs.get("content") or []
    first_logprob = content_logprobs[0] if content_logprobs else None
    token = None
    top_logprobs = None
    if isinstance(first_logprob, dict):
        token = first_logprob.get("token")
        top_logprobs = first_logprob.get("top_logprobs")
    return {
        "content": content,
        "content_preview": content[:200],
        "first_token_from_logprobs": token,
        "first_logprob": first_logprob,
        "top_logprobs": top_logprobs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    prompt = PROMPTS[args.case] + args.prompt_suffix
    rows = []
    for i in range(args.requests):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "logprobs": True,
            "top_logprobs": args.top_logprobs,
        }
        if args.chat_template_kwargs_json:
            payload["chat_template_kwargs"] = json.loads(
                args.chat_template_kwargs_json
            )
        started = time.perf_counter()
        error = None
        obj = None
        try:
            obj = post_json(endpoint, payload, args.timeout)
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
        elapsed = time.perf_counter() - started
        choice = ((obj or {}).get("choices") or [{}])[0]
        usage = (obj or {}).get("usage") or {}
        rows.append(
            {
                "index": i,
                "elapsed_s": elapsed,
                "error": error,
                "usage": usage,
                "choice_summary": first_choice_token(choice),
                "finish_reason": choice.get("finish_reason"),
                "raw_id": (obj or {}).get("id"),
            }
        )
        if args.sleep_s:
            time.sleep(args.sleep_s)

    first_tokens = [
        row["choice_summary"].get("first_token_from_logprobs")
        or row["choice_summary"].get("content_preview", "")[:16]
        for row in rows
        if not row.get("error")
    ]
    cached_tokens = []
    for row in rows:
        usage = row.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens.append(details.get("cached_tokens"))

    return {
        "schema": "vllm-prefix-cache-reuse-probe/v1",
        "run_id": args.run_id,
        "url": args.url,
        "model": args.model,
        "case": args.case,
        "max_tokens": args.max_tokens,
        "request_count": args.requests,
        "prompt_chars": len(prompt),
        "chat_template_kwargs_json": args.chat_template_kwargs_json,
        "rows": rows,
        "first_tokens": first_tokens,
        "all_first_tokens_same": len(set(first_tokens)) <= 1 if first_tokens else False,
        "cached_tokens_from_usage": cached_tokens,
        "any_usage_reports_cache_hit": any(
            isinstance(x, int) and x > 0 for x in cached_tokens
        ),
        "ok": bool(rows)
        and all(row.get("error") is None for row in rows)
        and (len(set(first_tokens)) <= 1 if first_tokens else False),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=sorted(PROMPTS), default="long_shared_prefix")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--chat-template-kwargs-json", default="")
    args = parser.parse_args()
    result = run(args)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
