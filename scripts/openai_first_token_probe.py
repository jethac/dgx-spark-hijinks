#!/usr/bin/env python3
"""First-token probe for OpenAI-compatible chat servers.

This is a cheap localization tool for "capacity works but text is garbage" rows. It asks
for one deterministic token with top-logprobs and records enough metadata to compare fp8
and NVFP4 KV rows before long decode errors compound.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from openai_serving_benchmark import PROMPTS
from spark_hardware import collect_cuda_hardware


EXTRA_CASES: dict[str, dict[str, Any]] = {
    "exact_spark_ok": {
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly this text: spark-ok",
            }
        ],
    },
    "simple_math": {
        "messages": [
            {
                "role": "user",
                "content": "Answer with only the number: what is 2 + 2?",
            }
        ],
    },
}


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


def text_script_summary(text: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    control = 0
    replacement = 0
    non_ascii = 0
    for char in text:
        if ord(char) > 127:
            non_ascii += 1
        if char == "\ufffd":
            replacement += 1
        category = unicodedata.category(char)
        if category.startswith("C") and char not in "\n\r\t":
            control += 1
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "UNNAMED"
        script = name.split(" ", 1)[0] if name else "UNKNOWN"
        counts[script] = counts.get(script, 0) + 1
    return {
        "chars": len(text),
        "non_ascii_chars": non_ascii,
        "replacement_chars": replacement,
        "control_chars": control,
        "script_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]),
    }


def extract_message(choice: dict[str, Any]) -> str:
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


def extract_first_logprob(choice: dict[str, Any]) -> dict[str, Any]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return {"available": False, "reason": "missing-logprobs"}
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return {"available": False, "reason": "missing-content", "raw_keys": sorted(logprobs)}
    first = content[0] if isinstance(content[0], dict) else {}
    top_logprobs = []
    raw_top = first.get("top_logprobs")
    if isinstance(raw_top, list):
        for item in raw_top:
            if not isinstance(item, dict):
                continue
            token = item.get("token")
            top_logprobs.append(
                {
                    "token": token,
                    "logprob": item.get("logprob"),
                    "bytes": item.get("bytes"),
                    "script_summary": text_script_summary(token) if isinstance(token, str) else None,
                }
            )
    token = first.get("token")
    return {
        "available": True,
        "token": token,
        "logprob": first.get("logprob"),
        "bytes": first.get("bytes"),
        "script_summary": text_script_summary(token) if isinstance(token, str) else None,
        "top_logprobs": top_logprobs,
    }


def get_case(case_name: str) -> dict[str, Any]:
    if case_name in EXTRA_CASES:
        return EXTRA_CASES[case_name]
    case = PROMPTS[case_name]
    return {"messages": case["messages"]}


def run_case(args: argparse.Namespace, endpoint: str, case_name: str) -> dict[str, Any]:
    case = get_case(case_name)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": case["messages"],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
    }
    if args.chat_template_kwargs_json:
        payload["chat_template_kwargs"] = json.loads(args.chat_template_kwargs_json)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "case": case_name,
        "ok": False,
        "payload": {
            "message_chars": sum(len(item.get("content", "")) for item in case["messages"]),
            "max_tokens": args.max_tokens,
            "top_logprobs": args.top_logprobs,
            "chat_template_kwargs_json": args.chat_template_kwargs_json,
        },
    }
    try:
        response = post_json(endpoint, payload, args.timeout)
        elapsed_s = time.perf_counter() - started
        choice = (response.get("choices") or [{}])[0]
        text = extract_message(choice)
        report.update(
            {
                "ok": bool(text),
                "elapsed_s": elapsed_s,
                "text": text,
                "text_script_summary": text_script_summary(text),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "first_logprob": extract_first_logprob(choice),
            }
        )
        if args.include_response:
            report["response"] = response
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        report["error"] = repr(exc)
    return report


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def token_set(case: dict[str, Any]) -> set[str]:
    first = case.get("first_logprob") or {}
    tokens = set()
    token = first.get("token")
    if isinstance(token, str):
        tokens.add(token)
    for item in first.get("top_logprobs") or []:
        if isinstance(item, dict) and isinstance(item.get("token"), str):
            tokens.add(item["token"])
    return tokens


def compare_reports(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_cases = {item.get("case"): item for item in baseline.get("cases", [])}
    comparisons = []
    for candidate_case in candidate.get("cases", []):
        case_name = candidate_case.get("case")
        baseline_case = baseline_cases.get(case_name)
        if not baseline_case:
            comparisons.append({"case": case_name, "ok": False, "error": "missing-baseline"})
            continue
        cand_first = candidate_case.get("first_logprob") or {}
        base_first = baseline_case.get("first_logprob") or {}
        cand_token = cand_first.get("token")
        base_token = base_first.get("token")
        cand_tokens = token_set(candidate_case)
        base_tokens = token_set(baseline_case)
        overlap = sorted(cand_tokens & base_tokens)
        union_count = len(cand_tokens | base_tokens)
        overlap_ratio = len(overlap) / union_count if union_count else 0.0
        comparisons.append(
            {
                "case": case_name,
                "ok": cand_token == base_token,
                "baseline_first_token": base_token,
                "candidate_first_token": cand_token,
                "baseline_text": baseline_case.get("text"),
                "candidate_text": candidate_case.get("text"),
                "top_logprob_overlap_ratio": overlap_ratio,
                "top_logprob_overlap": overlap[:20],
                "baseline_script_summary": baseline_case.get("text_script_summary"),
                "candidate_script_summary": candidate_case.get("text_script_summary"),
            }
        )
    return {
        "schema": "openai-first-token-compare/v1",
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "ok": all(item.get("ok") for item in comparisons),
        "comparisons": comparisons,
    }


def main() -> int:
    all_cases = sorted(set(PROMPTS) | set(EXTRA_CASES))
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--backend", default="unknown")
    parser.add_argument("--phase", choices=["before", "after", "exploratory"], default="exploratory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", action="append", choices=all_cases, default=[])
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--chat-template-kwargs-json")
    parser.add_argument("--compare-to")
    parser.add_argument("--input-report")
    parser.add_argument("--include-response", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.input_report:
        report = load_report(Path(args.input_report))
        report["run_id"] = args.run_id
    else:
        if not args.model:
            parser.error("--model is required without --input-report")
        endpoint = args.url.rstrip("/") + "/v1/chat/completions"
        case_names = args.case or ["exact_spark_ok", "simple_math", "short_decode"]
        report = {
            "schema": "openai-first-token-probe/v1",
            "run_id": args.run_id,
            "phase": args.phase,
            "backend": args.backend,
            "url": args.url,
            "model": args.model,
            "hardware": collect_cuda_hardware(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cases": [run_case(args, endpoint, case_name) for case_name in case_names],
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        report["ok"] = all(item.get("ok") for item in report["cases"])

    output = report
    if args.compare_to:
        output = compare_reports(report, load_report(Path(args.compare_to)))

    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if output.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
