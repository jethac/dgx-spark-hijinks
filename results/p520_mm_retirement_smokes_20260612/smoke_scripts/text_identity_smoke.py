#!/usr/bin/env python3
"""Text-only token-identity smoke for the mm-retire gate (c).

A text-only chat request must produce TOKEN-IDENTICAL output with the mm
knob on vs off (the mm path must never perturb pure-text serving). This
client captures the generated content, the completion token count, and
per-token logprobs (temp 0, seed 0) so knob-on vs knob-off runs can be
compared byte/token-identical by compare_smoke.py.

Banks a JSON with: content, finish_reason, usage, and the chat-completion
token logprobs list (token + logprob), for a deterministic comparison.
"""

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
    parser.add_argument(
        "--prompt",
        default=(
            "Write a single paragraph explaining what a perplexity metric "
            "measures in language modeling."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--label", default="text_identity")
    parser.add_argument("--output")
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "seed": 0,
        "logprobs": True,
        "top_logprobs": 1,
    }

    report: dict[str, Any] = {
        "schema": "text-identity-smoke/v1",
        "label": args.label,
        "endpoint": endpoint,
        "model": args.model,
        "prompt": args.prompt,
        "repeats": args.repeats,
        "contents": [],
        "token_seqs": [],
        "usages": [],
        "ok_deterministic": False,
        "ok": False,
        "elapsed_s": None,
    }

    def extract_tokens(response: dict[str, Any]) -> list[Any]:
        choice = response.get("choices", [{}])[0]
        lp = choice.get("logprobs") or {}
        content = lp.get("content") or []
        return [
            {"token": t.get("token"), "logprob": t.get("logprob")}
            for t in content
        ]

    started = time.time()
    try:
        for _ in range(args.repeats):
            response = post_json(endpoint, payload, args.timeout)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content")
            report["contents"].append(content if isinstance(content, str) else None)
            report["token_seqs"].append(extract_tokens(response))
            report["usages"].append(response.get("usage"))
        first = report["contents"][0]
        report["ok_deterministic"] = (
            all(c == first for c in report["contents"])
            and isinstance(first, str)
            and all(ts == report["token_seqs"][0] for ts in report["token_seqs"])
        )
        report["ok"] = report["ok_deterministic"]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        report["error"] = repr(exc)
        if isinstance(exc, urllib.error.HTTPError):
            try:
                report["error_body"] = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
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
