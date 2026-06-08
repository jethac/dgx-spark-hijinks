#!/usr/bin/env python3
"""Compare SGLang native /generate output/logprobs across two servers.

This is a focused divergence-window probe for cases where OpenAI chat output is
bad but backend routing already looks correct. It intentionally avoids row-level
benchmark claims.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from openai_serving_benchmark import PROMPTS
from spark_hardware import collect_cuda_hardware


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


def render_prompt(messages: list[dict[str, str]], model_path: str | None) -> dict[str, Any]:
    if not model_path:
        return {
            "mode": "raw-last-user-content",
            "text": messages[-1]["content"],
            "token_count": None,
        }
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return {
            "mode": "tokenizer-chat-template",
            "model_path": model_path,
            "text": text,
            "token_count": len(token_ids),
            "token_ids_preview": token_ids[:24],
        }
    except Exception as exc:
        return {
            "mode": "raw-last-user-content-after-template-error",
            "model_path": model_path,
            "error": repr(exc),
            "text": messages[-1]["content"],
            "token_count": None,
        }


def normalize_logprob_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return {
            "logprob": item.get("logprob"),
            "token_id": item.get("token_id", item.get("id")),
            "token": item.get("token", item.get("text")),
            "raw": item,
        }
    if isinstance(item, (list, tuple)):
        logprob = item[0] if len(item) > 0 else None
        token_id = item[1] if len(item) > 1 else None
        token = item[2] if len(item) > 2 else None
        return {"logprob": logprob, "token_id": token_id, "token": token, "raw": item}
    return {"logprob": None, "token_id": None, "token": None, "raw": item}


def normalize_top_logprobs(items: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [normalize_logprob_item(item) for item in items[:limit]]


def run_one(
    *,
    label: str,
    base_url: str,
    prompt_text: str,
    max_new_tokens: int,
    top_logprobs_num: int,
    timeout: int,
    include_raw: bool,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/generate"
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "logprob_start_len": -1,
        "top_logprobs_num": top_logprobs_num,
    }
    started = time.perf_counter()
    response = post_json(endpoint, payload, timeout)
    elapsed_s = time.perf_counter() - started
    meta = response.get("meta_info") or {}
    output_logprobs = [
        normalize_logprob_item(item)
        for item in meta.get("output_token_logprobs", [])
    ]
    output_top_logprobs = [
        normalize_top_logprobs(item, top_logprobs_num)
        for item in meta.get("output_top_logprobs", [])
    ]
    output_ids = response.get("output_ids")
    return {
        "label": label,
        "url": base_url,
        "elapsed_s": elapsed_s,
        "text": response.get("text", ""),
        "text_preview": response.get("text", "")[:800],
        "output_ids": output_ids if isinstance(output_ids, list) else None,
        "meta_info_subset": {
            "finish_reason": meta.get("finish_reason"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "output_token_logprobs_length": meta.get("output_token_logprobs_length"),
        },
        "output_token_logprobs": output_logprobs,
        "output_top_logprobs": output_top_logprobs,
        "raw_response": response if include_raw else None,
    }


def first_text_diff(a: str, b: str) -> dict[str, Any] | None:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return {
                "char_index": idx,
                "baseline_context": a[max(0, idx - 32) : idx + 64],
                "candidate_context": b[max(0, idx - 32) : idx + 64],
            }
    if len(a) != len(b):
        return {
            "char_index": limit,
            "baseline_context": a[max(0, limit - 32) : limit + 64],
            "candidate_context": b[max(0, limit - 32) : limit + 64],
        }
    return None


def first_sequence_diff(
    baseline_items: list[Any] | None,
    candidate_items: list[Any] | None,
    key: str | None = None,
) -> dict[str, Any] | None:
    if baseline_items is None or candidate_items is None:
        return None
    limit = min(len(baseline_items), len(candidate_items))
    for idx in range(limit):
        b_val = baseline_items[idx].get(key) if key else baseline_items[idx]
        c_val = candidate_items[idx].get(key) if key else candidate_items[idx]
        if b_val != c_val:
            return {
                "index": idx,
                "baseline": baseline_items[idx],
                "candidate": candidate_items[idx],
            }
    if len(baseline_items) != len(candidate_items):
        return {
            "index": limit,
            "baseline": baseline_items[limit] if limit < len(baseline_items) else None,
            "candidate": candidate_items[limit] if limit < len(candidate_items) else None,
        }
    return None


def window(items: list[Any], index: int | None, radius: int = 2) -> list[Any]:
    if index is None:
        return []
    start = max(0, index - radius)
    end = min(len(items), index + radius + 1)
    return items[start:end]


def compare_runs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_logprobs = baseline.get("output_token_logprobs") or []
    candidate_logprobs = candidate.get("output_token_logprobs") or []
    token_id_diff = first_sequence_diff(
        baseline_logprobs,
        candidate_logprobs,
        key="token_id",
    )
    diff_index = token_id_diff.get("index") if token_id_diff else None
    baseline_top = baseline.get("output_top_logprobs") or []
    candidate_top = candidate.get("output_top_logprobs") or []
    return {
        "text_diff": first_text_diff(baseline.get("text", ""), candidate.get("text", "")),
        "output_id_diff": first_sequence_diff(
            baseline.get("output_ids"),
            candidate.get("output_ids"),
        ),
        "token_id_diff": token_id_diff,
        "token_text_diff": first_sequence_diff(
            baseline_logprobs,
            candidate_logprobs,
            key="token",
        ),
        "divergence_window": {
            "token_index": diff_index,
            "baseline_tokens": window(baseline_logprobs, diff_index),
            "candidate_tokens": window(candidate_logprobs, diff_index),
            "baseline_top_logprobs": window(baseline_top, diff_index),
            "candidate_top_logprobs": window(candidate_top, diff_index),
        },
        "first_baseline_tokens": baseline_logprobs[:12],
        "first_candidate_tokens": candidate_logprobs[:12],
        "first_baseline_top_logprobs": (baseline.get("output_top_logprobs") or [])[:4],
        "first_candidate_top_logprobs": (candidate.get("output_top_logprobs") or [])[:4],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8-url", required=True)
    parser.add_argument("--fp4-url", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--case", default="medium_decode", choices=sorted(PROMPTS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--top-logprobs-num", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()

    case = PROMPTS[args.case]
    max_new_tokens = args.max_new_tokens or int(case["max_tokens"])
    prompt = render_prompt(case["messages"], args.model_path)
    fp8 = run_one(
        label="fp8",
        base_url=args.fp8_url,
        prompt_text=prompt["text"],
        max_new_tokens=max_new_tokens,
        top_logprobs_num=args.top_logprobs_num,
        timeout=args.timeout,
        include_raw=args.include_raw,
    )
    fp4 = run_one(
        label="fp4",
        base_url=args.fp4_url,
        prompt_text=prompt["text"],
        max_new_tokens=max_new_tokens,
        top_logprobs_num=args.top_logprobs_num,
        timeout=args.timeout,
        include_raw=args.include_raw,
    )
    report = {
        "schema": "sglang-native-logprob-compare/v1",
        "run_id": args.run_id,
        "case": args.case,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": collect_cuda_hardware(),
        "prompt": {
            key: value
            for key, value in prompt.items()
            if key != "text" or args.include_raw
        },
        "max_new_tokens": max_new_tokens,
        "top_logprobs_num": args.top_logprobs_num,
        "baseline": fp8,
        "candidate": fp4,
        "comparison": compare_runs(fp8, fp4),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
