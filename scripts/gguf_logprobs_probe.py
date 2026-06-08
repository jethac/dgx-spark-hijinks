#!/usr/bin/env python3
"""Probe llama.cpp OpenAI-compatible logprobs for lm-eval suitability."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


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


def extract_token_ids(response: dict[str, Any]) -> list[int]:
    tokens = response.get("tokens")
    if isinstance(tokens, list):
        ids = []
        for item in tokens:
            if isinstance(item, int):
                ids.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), int):
                ids.append(item["id"])
        return ids
    return []


def tokenize(base_url: str, content: str, add_special: bool, timeout: int) -> dict[str, Any]:
    return post_json(
        base_url.rstrip("/") + "/tokenize",
        {"content": content, "add_special": add_special, "with_pieces": True},
        timeout,
    )


def classify(
    response: dict[str, Any],
    *,
    expected_prompt_tokens: int | None = None,
    continuation_tokens: list[int] | None = None,
) -> dict[str, Any]:
    result = {
        "has_choices": False,
        "has_logprobs": False,
        "has_token_logprobs": False,
        "has_tokens": False,
        "covers_prompt_tokens": False,
        "covers_continuation_span": False,
        "looks_lm_eval_compatible": False,
        "notes": [],
    }
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        result["has_choices"] = True
        logprobs = choices[0].get("logprobs")
        if isinstance(logprobs, dict):
            result["has_logprobs"] = True
            token_logprobs = logprobs.get("token_logprobs")
            tokens = logprobs.get("tokens")
            result["has_token_logprobs"] = isinstance(token_logprobs, list)
            result["has_tokens"] = isinstance(tokens, list)
            if result["has_token_logprobs"] and result["has_tokens"]:
                if expected_prompt_tokens is None:
                    result["covers_prompt_tokens"] = True
                else:
                    result["covers_prompt_tokens"] = (
                        len(tokens) >= expected_prompt_tokens
                        and len(token_logprobs) >= expected_prompt_tokens
                    )
                if continuation_tokens is None:
                    result["covers_continuation_span"] = result["covers_prompt_tokens"]
                else:
                    result["covers_continuation_span"] = (
                        result["covers_prompt_tokens"]
                        and len(continuation_tokens) > 0
                        and all(
                            idx < len(token_logprobs)
                            and token_logprobs[idx] is not None
                            for idx in range(
                                expected_prompt_tokens - len(continuation_tokens),
                                expected_prompt_tokens,
                            )
                        )
                    )
                result["looks_lm_eval_compatible"] = bool(
                    result["covers_prompt_tokens"]
                    and result["covers_continuation_span"]
                )
            else:
                result["notes"].append(
                    "logprobs object exists but does not expose both tokens and token_logprobs"
                )
        else:
            result["notes"].append("choice has no dict logprobs object")
    else:
        result["notes"].append("response has no choices")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="unused")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--context", default="The capital of Japan is")
    parser.add_argument("--continuation", default=" zebra")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    endpoint = args.url.rstrip("/") + "/v1/completions"
    prompt = args.context + args.continuation
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "logprobs": 5,
        "echo": True,
    }

    started = time.time()
    report: dict[str, Any] = {
        "schema": "gguf-logprobs-probe/v1",
        "endpoint": endpoint,
        "payload": payload,
        "tokenization": {},
        "ok": False,
        "elapsed_s": None,
    }

    try:
        context_tokens = tokenize(base_url, args.context, True, args.timeout)
        continuation_tokens = tokenize(base_url, args.continuation, False, args.timeout)
        context_token_ids = extract_token_ids(context_tokens)
        continuation_token_ids = extract_token_ids(continuation_tokens)
        report["tokenization"] = {
            "context": context_tokens,
            "continuation": continuation_tokens,
            "context_token_ids": context_token_ids,
            "continuation_token_ids": continuation_token_ids,
            "expected_prompt_tokens": len(context_token_ids) + len(continuation_token_ids),
        }
        response = post_json(endpoint, payload, args.timeout)
        report["response"] = response
        report["classification"] = classify(
            response,
            expected_prompt_tokens=len(context_token_ids) + len(continuation_token_ids),
            continuation_tokens=continuation_token_ids,
        )
        report["ok"] = bool(report["classification"]["looks_lm_eval_compatible"])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        report["error"] = repr(exc)
    finally:
        report["elapsed_s"] = round(time.time() - started, 3)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
