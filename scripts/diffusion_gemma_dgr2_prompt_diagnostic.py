#!/usr/bin/env python3
"""Probe DiffusionGemma DG-R2 prompt sensitivity without changing the gate."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROBES = [
    {
        "id": "capital_only_city",
        "prompt": "Answer with only the city name: What is the capital of Japan?",
        "max_tokens": 32,
    },
    {
        "id": "capital_short_sentence",
        "prompt": "In one short sentence, what is the capital of Japan?",
        "max_tokens": 64,
    },
    {
        "id": "capital_direct",
        "prompt": "What is the capital of Japan?",
        "max_tokens": 64,
    },
    {
        "id": "capital_direct_full_canvas",
        "prompt": "What is the capital of Japan?",
        "max_tokens": 256,
    },
    {
        "id": "capital_tell_me",
        "prompt": "Tell me the capital city of Japan.",
        "max_tokens": 64,
    },
    {
        "id": "arithmetic_only_number",
        "prompt": "Answer with only the number: What is 2 + 2?",
        "max_tokens": 32,
    },
    {
        "id": "arithmetic_short_sentence",
        "prompt": "In one short sentence, what is 2 + 2?",
        "max_tokens": 64,
    },
    {
        "id": "arithmetic_direct",
        "prompt": "What is 2 + 2?",
        "max_tokens": 64,
    },
    {
        "id": "arithmetic_direct_full_canvas",
        "prompt": "What is 2 + 2?",
        "max_tokens": 256,
    },
    {
        "id": "digit_four",
        "prompt": "Write the digit four.",
        "max_tokens": 32,
    },
    {
        "id": "known_good_dgx",
        "prompt": (
            "In one short sentence, say what the NVIDIA DGX Spark desktop AI "
            "computer is useful for."
        ),
        "max_tokens": 64,
    },
]


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {
            "status": resp.status,
            "body": json.loads(resp.read().decode("utf-8")),
        }


def request_capture(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.time()
    try:
        result = request_json(url, payload, timeout)
        return {
            "ok": True,
            "elapsed_s": time.time() - started,
            "status": result["status"],
            "response": result["body"],
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "elapsed_s": time.time() - started,
            "status": exc.code,
            "error": body,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must preserve raw failure.
        return {
            "ok": False,
            "elapsed_s": time.time() - started,
            "error": repr(exc),
        }


def wait_ready(base_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            result = request_json(f"{base_url}/model_info", None, 5)
            if result["status"] == 200:
                return result["body"]
        except Exception as exc:  # noqa: BLE001 - readiness loop records last error.
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout_s}s: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30125")
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ready-timeout-s", type=float, default=1200)
    parser.add_argument("--request-timeout-s", type=float, default=900)
    args = parser.parse_args()

    model_info = wait_ready(args.base_url, args.ready_timeout_s)
    rows = []
    for probe in PROBES:
        chat_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": probe["prompt"]}],
            "max_tokens": probe["max_tokens"],
            "temperature": 0.0,
        }
        generate_payload = {
            "text": probe["prompt"],
            "sampling_params": {
                "max_new_tokens": probe["max_tokens"],
                "temperature": 0.0,
            },
        }
        chat_no_skip_payload = dict(chat_payload)
        chat_no_skip_payload["skip_special_tokens"] = False
        generate_no_skip_payload = {
            "text": probe["prompt"],
            "sampling_params": {
                "max_new_tokens": probe["max_tokens"],
                "temperature": 0.0,
                "skip_special_tokens": False,
            },
        }
        rows.append(
            {
                "probe": probe,
                "chat": request_capture(
                    f"{args.base_url}/v1/chat/completions",
                    chat_payload,
                    args.request_timeout_s,
                ),
                "chat_no_skip_special": request_capture(
                    f"{args.base_url}/v1/chat/completions",
                    chat_no_skip_payload,
                    args.request_timeout_s,
                ),
                "generate": request_capture(
                    f"{args.base_url}/generate",
                    generate_payload,
                    args.request_timeout_s,
                ),
                "generate_no_skip_special": request_capture(
                    f"{args.base_url}/generate",
                    generate_no_skip_payload,
                    args.request_timeout_s,
                ),
            }
        )

    artifact = {
        "schema": "sglang-diffusiongemma-dgr2-prompt-diagnostic/v1",
        "base_url": args.base_url,
        "model": args.model,
        "model_info": model_info,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
