#!/usr/bin/env python3
"""Run cookbook-style DiffusionGemma chat probes."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = [
    {
        "id": "cookbook_tcp_udp",
        "prompt": "What are the key differences between TCP and UDP?",
        "max_tokens": 1024,
        "expect": ["tcp", "udp"],
    },
    {
        "id": "capital_direct",
        "prompt": "What is the capital of Japan?",
        "max_tokens": 256,
        "expect": ["tokyo"],
    },
    {
        "id": "arithmetic_direct",
        "prompt": "What is 2 + 2?",
        "max_tokens": 256,
        "expect": ["4"],
    },
    {
        "id": "digit_four",
        "prompt": "Write the digit four.",
        "max_tokens": 64,
        "expect": ["4"],
    },
    {
        "id": "known_good_dgx",
        "prompt": (
            "In one short sentence, say what the NVIDIA DGX Spark desktop AI "
            "computer is useful for."
        ),
        "max_tokens": 256,
        "expect": ["ai"],
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
        return {"status": resp.status, "body": json.loads(resp.read().decode("utf-8"))}


def wait_ready(base_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            result = request_json(f"{base_url}/model_info", None, 5)
            if result["status"] == 200:
                return result["body"]
        except Exception as exc:  # noqa: BLE001 - preserve last readiness failure.
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout_s}s: {last_error}")


def post_capture(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
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
        return {
            "ok": False,
            "elapsed_s": time.time() - started,
            "status": exc.code,
            "error": exc.read().decode("utf-8", errors="replace"),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must preserve raw failure.
        return {"ok": False, "elapsed_s": time.time() - started, "error": repr(exc)}


def message_text(response: dict[str, Any]) -> str:
    if not response.get("ok"):
        return ""
    choices = response.get("response", {}).get("choices", [])
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30125")
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ready-timeout-s", type=float, default=1200)
    parser.add_argument("--request-timeout-s", type=float, default=1200)
    args = parser.parse_args()

    model_info = wait_ready(args.base_url, args.ready_timeout_s)
    rows = []
    checks = []
    for prompt in PROMPTS:
        texts = []
        for repeat in range(args.repeats):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt["prompt"]}],
                "max_tokens": prompt["max_tokens"],
                "temperature": 0.0,
            }
            response = post_capture(
                f"{args.base_url}/v1/chat/completions",
                payload,
                args.request_timeout_s,
            )
            text = message_text(response)
            texts.append(text)
            rows.append(
                {
                    "prompt_id": prompt["id"],
                    "repeat": repeat,
                    "prompt": prompt["prompt"],
                    "request": payload,
                    "response": response,
                    "text": text,
                }
            )
        lower_texts = [text.lower() for text in texts]
        checks.append(
            {
                "prompt_id": prompt["id"],
                "stable": len(set(texts)) == 1,
                "non_empty": all(bool(text) for text in texts),
                "expectations": prompt["expect"],
                "expectations_met": [
                    all(term in lower for lower in lower_texts)
                    for term in prompt["expect"]
                ],
                "texts": texts,
            }
        )

    summary_ok = True
    for check in checks:
        summary_ok &= check["stable"]
        summary_ok &= check["non_empty"]
        summary_ok &= all(check["expectations_met"])

    artifact = {
        "schema": "sglang-diffusiongemma-cookbook-conformance/v1",
        "base_url": args.base_url,
        "model": args.model,
        "model_info": model_info,
        "repeats": args.repeats,
        "all_ok": summary_ok,
        "checks": checks,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_ok": summary_ok, "checks": checks}, indent=2, sort_keys=True))
    if not summary_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
