#!/usr/bin/env python3
"""Small before/after benchmark for OpenAI-compatible local servers."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


PROMPTS = {
    "short_decode": {
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": "In one paragraph, explain why a local AI workstation benefits from fast memory.",
            }
        ],
    },
    "medium_decode": {
        "max_tokens": 192,
        "messages": [
            {
                "role": "user",
                "content": "Write a concise engineering note explaining how to validate that an LLM server is using the intended GPU kernels. Include concrete evidence to collect.",
            }
        ],
    },
    "long_prefill": {
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are benchmarking a local inference server. "
                    "Summarize the following requirements into five bullets. "
                    + "The benchmark must capture environment metadata, backend selection, cold-start time, warm decode speed, memory pressure, output quality, and reproducibility. "
                    * 80
                ),
            }
        ],
    },
}


def request_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


def stream_chat(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_s = None
    content_parts: list[str] = []
    usage = None
    chunk_count = 0
    finish_reason = None
    matched_stop = None
    last_event = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_text = line[5:].strip()
            if data_text == "[DONE]":
                break
            chunk_count += 1
            try:
                event = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            last_event = event
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            if choice.get("matched_stop") is not None:
                matched_stop = choice.get("matched_stop")
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - started
                content_parts.append(text)
    total_s = time.perf_counter() - started
    return {
        "ttft_s": first_token_s,
        "total_s": total_s,
        "chunk_count": chunk_count,
        "content": "".join(content_parts),
        "finish_reason": finish_reason,
        "last_event": last_event,
        "matched_stop": matched_stop,
        "usage": usage,
    }


def choose_model(base_url: str, requested: str | None, timeout: int) -> tuple[str | None, dict[str, Any] | None]:
    if requested:
        return requested, None
    models = request_json(base_url.rstrip("/") + "/v1/models", timeout)
    data = models.get("data") or []
    if not data:
        return None, models
    return data[0].get("id"), models


def run_case(args: argparse.Namespace, model: str, case_name: str, case: dict[str, Any]) -> dict[str, Any]:
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": case["messages"],
        "temperature": 0,
        "max_tokens": case["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    report = {
        "case": case_name,
        "ok": False,
        "payload": {
            "max_tokens": payload["max_tokens"],
            "message_chars": sum(len(m["content"]) for m in payload["messages"]),
        },
    }
    try:
        result = stream_chat(endpoint, payload, args.timeout)
        content = result.get("content") or ""
        usage = result.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        total_s = result.get("total_s")
        ttft_s = result.get("ttft_s")
        decode_s = None
        decode_tok_s = None
        if total_s is not None and ttft_s is not None:
            decode_s = max(total_s - ttft_s, 0.0)
        if completion_tokens and decode_s and decode_s > 0:
            decode_tok_s = completion_tokens / decode_s
        report.update(
            {
                "ok": bool(content.strip()),
                "ttft_s": ttft_s,
                "total_s": total_s,
                "decode_s": decode_s,
                "decode_tok_s": decode_tok_s,
                "usage": usage,
                "content_chars": len(content),
                "content_preview": content[:500],
                "chunk_count": result.get("chunk_count"),
                "finish_reason": result.get("finish_reason"),
                "last_event": result.get("last_event"),
                "matched_stop": result.get("matched_stop"),
            }
        )
    except Exception as exc:
        report["error"] = repr(exc)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--backend", default="unknown")
    parser.add_argument("--phase", choices=["before", "after", "exploratory"], default="exploratory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", action="append", choices=sorted(PROMPTS), default=[])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output")
    args = parser.parse_args()

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report: dict[str, Any] = {
        "schema": "openai-serving-benchmark/v1",
        "run_id": args.run_id,
        "phase": args.phase,
        "backend": args.backend,
        "url": args.url,
        "started_utc": started,
        "cases": [],
    }
    try:
        model, models_response = choose_model(args.url, args.model, args.timeout)
        report["models_response"] = models_response
        report["model"] = model
        if not model:
            raise RuntimeError("No model was specified and /v1/models returned no model id")
        case_names = args.case or ["short_decode", "medium_decode", "long_prefill"]
        for case_name in case_names:
            report["cases"].append(run_case(args, model, case_name, PROMPTS[case_name]))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        report["error"] = repr(exc)

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["ok"] = bool(report.get("cases")) and all(case.get("ok") for case in report["cases"])
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
