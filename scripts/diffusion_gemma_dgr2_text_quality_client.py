#!/usr/bin/env python3
"""Run the DG-R2 text-only DiffusionGemma smoke-quality prompt set."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPTS = [
    {
        "id": "capital_japan",
        "prompt": "Answer with only the city name: What is the capital of Japan?",
        "max_tokens": 32,
    },
    {
        "id": "arithmetic_2_plus_2",
        "prompt": "Answer with only the number: What is 2 + 2?",
        "max_tokens": 32,
    },
    {
        "id": "dgx_spark_use",
        "prompt": (
            "In one short sentence, say what the NVIDIA DGX Spark desktop AI "
            "computer is useful for."
        ),
        "max_tokens": 64,
    },
]


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/model_info", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout_s}s: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30125")
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ready-timeout-s", type=float, default=1200)
    parser.add_argument("--request-timeout-s", type=float, default=900)
    args = parser.parse_args()

    wait_ready(args.base_url, args.ready_timeout_s)
    rows = []
    for prompt in PROMPTS:
        for repeat in range(args.repeats):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt["prompt"]}],
                "max_tokens": prompt["max_tokens"],
                "temperature": 0.0,
            }
            started = time.time()
            response = post_json(
                f"{args.base_url}/v1/chat/completions",
                payload,
                args.request_timeout_s,
            )
            rows.append(
                {
                    "prompt_id": prompt["id"],
                    "repeat": repeat,
                    "prompt": prompt["prompt"],
                    "request": payload,
                    "response": response,
                    "elapsed_s": time.time() - started,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": "sglang-diffusiongemma-dgr2-text-quality/v1",
                "base_url": args.base_url,
                "model": args.model,
                "repeats": args.repeats,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
