#!/usr/bin/env python3
"""Small OpenAI-compatible chat smoke test for vLLM/Ollama/llama.cpp servers."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output")
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly this text: spark-ok",
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
    }

    started = time.time()
    report: dict[str, Any] = {
        "schema": "openai-chat-smoke/v1",
        "endpoint": endpoint,
        "model": args.model,
        "ok": False,
        "elapsed_s": None,
    }

    try:
        response = post_json(endpoint, payload, args.timeout)
        report["response"] = response
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        report["content"] = content
        report["ok"] = "spark-ok" in content.lower()
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

