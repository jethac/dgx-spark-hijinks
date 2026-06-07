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


def classify(response: dict[str, Any]) -> dict[str, Any]:
    result = {
        "has_choices": False,
        "has_logprobs": False,
        "has_token_logprobs": False,
        "has_tokens": False,
        "looks_lm_eval_compatible": False,
        "notes": [],
    }
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        result["has_choices"] = True
        logprobs = choices[0].get("logprobs")
        if isinstance(logprobs, dict):
            result["has_logprobs"] = True
            result["has_token_logprobs"] = "token_logprobs" in logprobs
            result["has_tokens"] = "tokens" in logprobs
            if result["has_token_logprobs"] and result["has_tokens"]:
                result["looks_lm_eval_compatible"] = True
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
    parser.add_argument("--output")
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/completions"
    payload = {
        "model": args.model,
        "prompt": "The capital of Japan is",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 5,
        "echo": True,
    }

    started = time.time()
    report: dict[str, Any] = {
        "schema": "gguf-logprobs-probe/v1",
        "endpoint": endpoint,
        "payload": payload,
        "ok": False,
        "elapsed_s": None,
    }

    try:
        response = post_json(endpoint, payload, args.timeout)
        report["response"] = response
        report["classification"] = classify(response)
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

