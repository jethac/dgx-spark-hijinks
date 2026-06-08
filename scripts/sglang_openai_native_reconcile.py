#!/usr/bin/env python3
"""Reconcile SGLang OpenAI chat serialization with native /generate.

This is a client-side probe for already-running SGLang servers. It does not
launch serving and does not allocate GPU memory by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from openai_serving_benchmark import PROMPTS


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(token_ids: list[int]) -> str:
    data = ",".join(str(x) for x in token_ids)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def render_local_prompt(messages: list[dict[str, str]], model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return {
        "mode": "hf_apply_chat_template",
        "model_path": model_path,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "token_ids_sha256": sha256_ids(token_ids),
        "text_sha256": sha256_text(text),
        "text_preview": text[:500],
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
        return {
            "logprob": item[0] if len(item) > 0 else None,
            "token_id": item[1] if len(item) > 1 else None,
            "token": item[2] if len(item) > 2 else None,
            "raw": item,
        }
    return {"logprob": None, "token_id": None, "token": None, "raw": item}


def normalize_top_logprobs(items: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [normalize_logprob_item(item) for item in items[:limit]]


def summarize_openai_chat(response: dict[str, Any], top_logprobs_num: int) -> dict[str, Any]:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    prompt_token_ids = choice.get("prompt_token_ids")
    logprobs = choice.get("logprobs") or {}
    content_logprobs = logprobs.get("content") if isinstance(logprobs, dict) else None
    first_tokens = []
    if isinstance(content_logprobs, list):
        for item in content_logprobs[:12]:
            if not isinstance(item, dict):
                continue
            top = item.get("top_logprobs")
            first_tokens.append(
                {
                    "token": item.get("token"),
                    "logprob": item.get("logprob"),
                    "top_logprobs": normalize_top_logprobs(top, top_logprobs_num),
                }
            )
    return {
        "text": message.get("content") or message.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "prompt_token_ids": prompt_token_ids if isinstance(prompt_token_ids, list) else None,
        "prompt_token_count": len(prompt_token_ids) if isinstance(prompt_token_ids, list) else None,
        "prompt_token_ids_sha256": sha256_ids(prompt_token_ids)
        if isinstance(prompt_token_ids, list)
        else None,
        "first_tokens": first_tokens,
        "raw_response": response,
    }


def summarize_native(response: dict[str, Any], top_logprobs_num: int) -> dict[str, Any]:
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
        "raw_response": response,
    }


def openai_chat_one(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    top_logprobs_num: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_logprobs_num,
        "return_prompt_token_ids": True,
        "return_meta_info": True,
    }
    started = time.perf_counter()
    response = post_json(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    return {
        "elapsed_s": time.perf_counter() - started,
        "payload": payload,
        "summary": summarize_openai_chat(response, top_logprobs_num),
    }


def native_generate_one(
    *,
    base_url: str,
    prompt_key: str,
    prompt_value: str | list[int],
    max_new_tokens: int,
    top_logprobs_num: int,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        prompt_key: prompt_value,
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
    response = post_json(base_url.rstrip("/") + "/generate", payload, timeout)
    return {
        "elapsed_s": time.perf_counter() - started,
        "payload_prompt_key": prompt_key,
        "payload_prompt_token_count": len(prompt_value)
        if isinstance(prompt_value, list)
        else None,
        "summary": summarize_native(response, top_logprobs_num),
    }


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


def compare_native(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    baseline_logprobs = baseline_summary.get("output_token_logprobs") or []
    candidate_logprobs = candidate_summary.get("output_token_logprobs") or []
    token_diff = first_sequence_diff(baseline_logprobs, candidate_logprobs, "token_id")
    diff_index = token_diff.get("index") if token_diff else None
    return {
        "output_id_diff": first_sequence_diff(
            baseline_summary.get("output_ids"),
            candidate_summary.get("output_ids"),
        ),
        "token_id_diff": token_diff,
        "token_text_diff": first_sequence_diff(baseline_logprobs, candidate_logprobs, "token"),
        "divergence_window": {
            "token_index": diff_index,
            "baseline_tokens": window(baseline_logprobs, diff_index),
            "candidate_tokens": window(candidate_logprobs, diff_index),
            "baseline_top_logprobs": window(
                baseline_summary.get("output_top_logprobs") or [], diff_index
            ),
            "candidate_top_logprobs": window(
                candidate_summary.get("output_top_logprobs") or [], diff_index
            ),
        },
    }


def compare_prompt_ids(openai_ids: list[int] | None, local_ids: list[int]) -> dict[str, Any]:
    return {
        "openai_prompt_token_count": len(openai_ids) if openai_ids is not None else None,
        "local_prompt_token_count": len(local_ids),
        "openai_prompt_token_ids_sha256": sha256_ids(openai_ids)
        if openai_ids is not None
        else None,
        "local_prompt_token_ids_sha256": sha256_ids(local_ids),
        "first_prompt_id_diff": first_sequence_diff(openai_ids, local_ids),
    }


def run_server_pair(
    *,
    label: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    local_prompt: dict[str, Any],
    max_tokens: int,
    top_logprobs_num: int,
    timeout: int,
) -> dict[str, Any]:
    chat = openai_chat_one(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        top_logprobs_num=top_logprobs_num,
        timeout=timeout,
    )
    openai_prompt_ids = chat["summary"].get("prompt_token_ids")
    native_from_openai_ids = None
    if openai_prompt_ids:
        native_from_openai_ids = native_generate_one(
            base_url=base_url,
            prompt_key="input_ids",
            prompt_value=openai_prompt_ids,
            max_new_tokens=max_tokens,
            top_logprobs_num=top_logprobs_num,
            timeout=timeout,
        )
    native_from_local_text = native_generate_one(
        base_url=base_url,
        prompt_key="text",
        prompt_value=local_prompt["text_preview"]
        if "text" not in local_prompt
        else local_prompt["text"],
        max_new_tokens=max_tokens,
        top_logprobs_num=top_logprobs_num,
        timeout=timeout,
    )
    return {
        "label": label,
        "url": base_url,
        "openai_chat": chat,
        "native_from_openai_prompt_ids": native_from_openai_ids,
        "native_from_local_render_text": native_from_local_text,
        "prompt_id_comparison": compare_prompt_ids(
            openai_prompt_ids,
            local_prompt["token_ids"],
        ),
    }


def strip_raw(report: dict[str, Any]) -> dict[str, Any]:
    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: visit(item)
                for key, item in value.items()
                if key not in {"raw_response"}
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8-url", required=True)
    parser.add_argument("--fp4-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--case", default="medium_decode", choices=sorted(PROMPTS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--top-logprobs-num", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()

    case = PROMPTS[args.case]
    max_tokens = args.max_new_tokens or int(case["max_tokens"])
    local_prompt = render_local_prompt(case["messages"], args.model_path)

    # Keep the full rendered text for the native replay but do not duplicate it
    # in the artifact unless --include-raw is requested.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    local_prompt["text"] = tokenizer.apply_chat_template(
        case["messages"],
        tokenize=False,
        add_generation_prompt=True,
    )

    fp8 = run_server_pair(
        label="fp8",
        base_url=args.fp8_url,
        model=args.model,
        messages=case["messages"],
        local_prompt=local_prompt,
        max_tokens=max_tokens,
        top_logprobs_num=args.top_logprobs_num,
        timeout=args.timeout,
    )
    fp4 = run_server_pair(
        label="fp4",
        base_url=args.fp4_url,
        model=args.model,
        messages=case["messages"],
        local_prompt=local_prompt,
        max_tokens=max_tokens,
        top_logprobs_num=args.top_logprobs_num,
        timeout=args.timeout,
    )

    native_key = "native_from_openai_prompt_ids"
    fp8_native = fp8.get(native_key)
    fp4_native = fp4.get(native_key)
    report = {
        "schema": "sglang-openai-native-reconcile/v1",
        "run_id": args.run_id,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case": args.case,
        "max_new_tokens": max_tokens,
        "top_logprobs_num": args.top_logprobs_num,
        "local_prompt": local_prompt,
        "fp8": fp8,
        "fp4": fp4,
        "comparison": {
            "fp8_vs_fp4_native_from_openai_prompt_ids": compare_native(
                fp8_native, fp4_native
            )
            if fp8_native and fp4_native
            else None,
            "fp8_vs_fp4_native_from_local_render_text": compare_native(
                fp8["native_from_local_render_text"],
                fp4["native_from_local_render_text"],
            ),
        },
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not args.include_raw:
        report = strip_raw(report)
        report["local_prompt"].pop("text", None)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
