#!/usr/bin/env python3
"""Small before/after benchmark for OpenAI-compatible local servers."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any

from spark_hardware import collect_cuda_hardware


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
    "natural_long_prefill": {
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are reviewing notes from a local AI infrastructure test. "
                    "Write six concise bullets that identify the operational risks, "
                    "the measurements still needed, and the next engineering actions.\n\n"
                    "Field notes:\n"
                    "A single workstation is being used to validate local inference "
                    "stacks before broader deployment. The machine has a unified memory "
                    "pool, so CPU and GPU allocations compete for the same physical "
                    "capacity. During earlier tests, two large serving containers were "
                    "started together for a matched comparison. The combined allocation "
                    "left too little room for the operating system, and the host stopped "
                    "responding until it was power-cycled. Future comparisons must run "
                    "sequentially, keep a visible memory margin, and place each container "
                    "inside a cgroup limit so a runaway process is killed without wedging "
                    "the driver.\n\n"
                    "The software stack under test includes several runtimes. One path "
                    "uses a general-purpose server with continuous batching. Another path "
                    "uses a lightweight implementation that is excellent for practical "
                    "quantized serving. A third runtime has strong scheduling machinery "
                    "but still needs custom attention-cache support for the local hardware. "
                    "The same model family should be measured across these runtimes, but "
                    "claims must stay scoped: a small text-only model does not prove a "
                    "large multimodal mixture-of-experts model, and a capacity result does "
                    "not prove output quality.\n\n"
                    "Kernel selection has been a recurring source of false confidence. "
                    "Some builds report a broad architecture family while the actual "
                    "native instructions require an architecture-specific target. Other "
                    "paths compile but silently use a generic fallback, producing correct "
                    "results at disappointing speed. Every run should record the compute "
                    "capability, the number of streaming multiprocessors, the selected "
                    "attention backend, the cache element type, and the generated module "
                    "flags. A useful benchmark artifact should include enough information "
                    "for a second engineer to tell whether the intended kernels actually "
                    "ran.\n\n"
                    "The key optimization under evaluation is a smaller key-value cache. "
                    "A full four-bit cache creates the largest theoretical capacity gain, "
                    "but the key side influences attention logits and can destabilize "
                    "prefix reuse if the runtime feeds the kernel inconsistent scale "
                    "information. A mixed design keeps keys in eight-bit format and stores "
                    "values in packed four-bit format. This gives up some theoretical "
                    "capacity, but it protects the most sensitive part of attention while "
                    "still reducing memory pressure. The mixed design must be tested with "
                    "radix or prefix caching enabled, because disabling cache reuse hides "
                    "the behavior that production serving depends on.\n\n"
                    "Quality measurement is as important as throughput. A server can "
                    "return non-empty text while still producing repetitive or incoherent "
                    "answers. First-token checks are useful for localizing a regression, "
                    "but they do not establish long-form correctness. The next quality "
                    "gate should compare matched prompts under the same graph policy and "
                    "memory fraction, inspect repetition metrics, and eventually compute "
                    "supplied-token log likelihood on a natural document. Any public "
                    "summary should separate capacity, speed, first-token stability, and "
                    "long-context quality as distinct claims.\n\n"
                    "The immediate engineering plan is conservative. Keep the host safe, "
                    "run one server at a time, document every artifact, and avoid broad "
                    "upstream changes until a reproducible before-and-after story exists. "
                    "When a workaround is used, label it clearly. When a hypothesis fails, "
                    "record the negative result instead of retuning the benchmark around "
                    "it. The goal is not to make a single row look impressive; the goal is "
                    "to make the workstation perform as well as the silicon allows while "
                    "leaving a public trail that other owners can reproduce."
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
    reasoning_parts: list[str] = []
    legacy_reasoning_parts: list[str] = []
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
            reasoning = delta.get("reasoning_content")
            legacy_reasoning = delta.get("reasoning")
            if text or reasoning or legacy_reasoning:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - started
            if text:
                content_parts.append(text)
            if reasoning:
                reasoning_parts.append(reasoning)
            if legacy_reasoning:
                legacy_reasoning_parts.append(legacy_reasoning)
    total_s = time.perf_counter() - started
    return {
        "ttft_s": first_token_s,
        "total_s": total_s,
        "chunk_count": chunk_count,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "reasoning": "".join(legacy_reasoning_parts),
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
    messages = [
        {
            **message,
            "content": message["content"] + args.prompt_suffix,
        }
        for message in case["messages"]
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": case["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chat_template_kwargs = None
    if args.chat_template_kwargs_json:
        chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
        payload["chat_template_kwargs"] = chat_template_kwargs
    report = {
        "case": case_name,
        "ok": False,
        "payload": {
            "max_tokens": payload["max_tokens"],
            "message_chars": sum(len(m["content"]) for m in payload["messages"]),
            "prompt_suffix": args.prompt_suffix,
            "chat_template_kwargs": chat_template_kwargs,
        },
    }
    try:
        result = stream_chat(endpoint, payload, args.timeout)
        content = result.get("content") or ""
        reasoning_content = result.get("reasoning_content") or ""
        output_text = content if content.strip() else reasoning_content
        usage = result.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        total_s = result.get("total_s")
        ttft_s = result.get("ttft_s")
        decode_s = None
        decode_tok_s = None
        completion_tok_s_total = None
        if total_s is not None and ttft_s is not None:
            decode_s = max(total_s - ttft_s, 0.0)
        if completion_tokens and decode_s and decode_s > 0:
            decode_tok_s = completion_tokens / decode_s
        if completion_tokens and total_s and total_s > 0:
            completion_tok_s_total = completion_tokens / total_s
        legacy_reasoning = result.get("reasoning") or ""
        report.update(
            {
                "ttft_s": ttft_s,
                "total_s": total_s,
                "decode_s": decode_s,
                "decode_tok_s": decode_tok_s,
                "completion_tok_s_total": completion_tok_s_total,
                "usage": usage,
                "ok": bool(output_text.strip()),
                "content_chars": len(content),
                "content_preview": content[:500],
                "reasoning_content_chars": len(reasoning_content),
                "reasoning_content_preview": reasoning_content[:500],
                "reasoning_chars": len(legacy_reasoning),
                "reasoning_preview": legacy_reasoning[:500],
                "output_chars": len(output_text),
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
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--chat-template-kwargs-json")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output")
    args = parser.parse_args()

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    hardware = collect_cuda_hardware()
    report: dict[str, Any] = {
        "schema": "openai-serving-benchmark/v1",
        "run_id": args.run_id,
        "phase": args.phase,
        "backend": args.backend,
        "hardware": hardware,
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
