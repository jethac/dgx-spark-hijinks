#!/usr/bin/env python3
"""Audio multimodal smoke for vLLM OpenAI servers (Amendment 5 cells).

Pattern of openai_chat_smoke.py, extended for audio: sends a local WAV
as a base64 data URL via the OpenAI ``audio_url`` content part, temp 0,
N repeats (default 2) for the repeat-determinism gate, and keyword
grounding gates. Banks verbatim responses.

Gates reported (zero-bug bar):
- ok_grounded: every --expect-keyword (case-insensitive) appears in the
  reply (use a distinctive transcript word for the speech clip; leave
  empty for the tone control, which is adjudicated from the banked
  verbatim reply).
- ok_deterministic: all repeats byte-identical.
- ok = grounded AND deterministic AND no transport error.
"""

from __future__ import annotations

import argparse
import base64
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
    parser.add_argument("--audio", required=True, help="path to a WAV file")
    parser.add_argument("--prompt", default="What is said in this audio?")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--expect-keywords",
        default="",
        help="comma-separated; ALL must appear (case-insensitive)",
    )
    parser.add_argument("--label", default="audio_smoke")
    parser.add_argument("--output")
    args = parser.parse_args()

    with open(args.audio, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    audio_url = f"data:audio/wav;base64,{b64}"

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }

    keywords = [k.strip() for k in args.expect_keywords.split(",") if k.strip()]
    report: dict[str, Any] = {
        "schema": "audio-mm-smoke/v1",
        "label": args.label,
        "endpoint": endpoint,
        "model": args.model,
        "audio_file": args.audio,
        "prompt": args.prompt,
        "expect_keywords": keywords,
        "repeats": args.repeats,
        "contents": [],
        "usages": [],
        "ok_grounded": False,
        "ok_deterministic": False,
        "ok": False,
        "elapsed_s": None,
    }

    started = time.time()
    try:
        for _ in range(args.repeats):
            response = post_json(endpoint, payload, args.timeout)
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content")
            report["contents"].append(content if isinstance(content, str) else None)
            report["usages"].append(response.get("usage"))
        first = report["contents"][0]
        if isinstance(first, str) and first.strip():
            low = first.lower()
            report["ok_grounded"] = all(k.lower() in low for k in keywords)
        report["ok_deterministic"] = all(
            c == first for c in report["contents"]
        ) and isinstance(first, str)
        report["ok"] = report["ok_grounded"] and report["ok_deterministic"]
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
