#!/usr/bin/env python3
"""Run the revised DiffusionGemma DG-R2 text-only quality gate."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = [
    {
        "id": "capital_japan_direct",
        "prompt": "What is the capital of Japan?",
        "max_tokens": 64,
        "rule": "contains_tokyo",
    },
    {
        "id": "arithmetic_2_plus_2_direct",
        "prompt": "What is 2 + 2?",
        "max_tokens": 64,
        "rule": "contains_standalone_4",
    },
    {
        "id": "dgx_spark_use",
        "prompt": (
            "In one short sentence, say what the NVIDIA DGX Spark desktop AI "
            "computer is useful for."
        ),
        "max_tokens": 64,
        "rule": "mentions_local_or_desktop_ai",
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
        except Exception as exc:  # noqa: BLE001 - readiness loop records last error.
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


def check_rule(rule: str, text: str) -> tuple[bool, str]:
    lower = text.lower()
    if rule == "contains_tokyo":
        return "tokyo" in lower, "contains Tokyo"
    if rule == "contains_standalone_4":
        return re.search(r"(^|[^0-9])4([^0-9]|$)", text) is not None, "contains standalone 4"
    if rule == "mentions_local_or_desktop_ai":
        mentions_ai = "ai" in lower or "machine learning" in lower
        mentions_form = (
            "local" in lower
            or "desktop" in lower
            or "development" in lower
            or "workstation" in lower
        )
        return mentions_ai and mentions_form, "mentions local/desktop/development AI use"
    raise KeyError(rule)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30125")
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ready-timeout-s", type=float, default=1200)
    parser.add_argument("--request-timeout-s", type=float, default=900)
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
        answer_ok, answer_rule = check_rule(prompt["rule"], texts[0])
        checks.append(
            {
                "prompt_id": prompt["id"],
                "stable": len(set(texts)) == 1,
                "non_empty": all(bool(text) for text in texts),
                "answer_ok": answer_ok,
                "answer_rule": answer_rule,
                "texts": texts,
            }
        )

    all_ok = all(
        check["stable"] and check["non_empty"] and check["answer_ok"]
        for check in checks
    )
    artifact = {
        "schema": "sglang-diffusiongemma-dgr2-revised-text-quality/v1",
        "base_url": args.base_url,
        "model": args.model,
        "model_info": model_info,
        "repeats": args.repeats,
        "all_ok": all_ok,
        "checks": checks,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_ok": all_ok, "checks": checks}, indent=2, sort_keys=True))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
