#!/usr/bin/env python3
"""Probe whether SGLang FP4-KV divergence follows request order / radix reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from openai_serving_benchmark import PROMPTS


def post_json(url: str, payload: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def try_post_json(url: str, payload: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    try:
        return {"ok": True, "response": post_json(url, payload, timeout)}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "body": body}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def sha256_ids(token_ids: list[int] | None) -> str | None:
    if token_ids is None:
        return None
    return hashlib.sha256(",".join(str(x) for x in token_ids).encode("ascii")).hexdigest()


def render_prompt(messages: list[dict[str, str]], model_path: str) -> dict[str, Any]:
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
        "text": text,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "token_ids_sha256": sha256_ids(token_ids),
    }


def normalize_logprob_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return {
            "token": item.get("token") or item.get("text"),
            "token_id": item.get("token_id", item.get("id")),
            "logprob": item.get("logprob"),
            "raw": item,
        }
    if isinstance(item, (list, tuple)):
        return {
            "logprob": item[0] if len(item) > 0 else None,
            "token_id": item[1] if len(item) > 1 else None,
            "token": item[2] if len(item) > 2 else None,
            "raw": item,
        }
    return {"token": None, "token_id": None, "logprob": None, "raw": item}


def normalize_top_logprobs(items: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [normalize_logprob_item(item) for item in items[:limit]]


def summarize_openai(response: dict[str, Any], top_logprobs_num: int) -> dict[str, Any]:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    first = content[0] if isinstance(content, list) and content else {}
    prompt_ids = choice.get("prompt_token_ids")
    meta = choice.get("meta_info") or {}
    return {
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "meta_info": {
            "cached_tokens": meta.get("cached_tokens"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "output_token_logprobs_length": len(
                meta.get("output_token_logprobs") or []
            ),
        },
        "prompt_token_count": len(prompt_ids) if isinstance(prompt_ids, list) else None,
        "prompt_token_ids_sha256": sha256_ids(prompt_ids)
        if isinstance(prompt_ids, list)
        else None,
        "prompt_token_ids": prompt_ids,
        "first_token": normalize_logprob_item(first),
        "first_token_top_logprobs": normalize_top_logprobs(
            first.get("top_logprobs") if isinstance(first, dict) else None,
            top_logprobs_num,
        ),
    }


def summarize_native(response: dict[str, Any], top_logprobs_num: int) -> dict[str, Any]:
    meta = response.get("meta_info") or {}
    output_logprobs = meta.get("output_token_logprobs")
    first = output_logprobs[0] if isinstance(output_logprobs, list) and output_logprobs else None
    top = meta.get("output_top_logprobs")
    first_top = top[0] if isinstance(top, list) and top else None
    return {
        "finish_reason": meta.get("finish_reason"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "output_token_logprobs_length": meta.get("output_token_logprobs_length"),
        "output_ids_head": response.get("output_ids", [])[:8]
        if isinstance(response.get("output_ids"), list)
        else None,
        "first_token": normalize_logprob_item(first),
        "first_token_top_logprobs": normalize_top_logprobs(first_top, top_logprobs_num),
    }


def flush_cache(base_url: str, timeout: int) -> dict[str, Any]:
    return try_post_json(base_url.rstrip("/") + "/flush_cache", {}, timeout)


def openai_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    rid: str,
    max_tokens: int,
    top_logprobs_num: int,
    extra_key: str | None = None,
    cache_salt: str | None = None,
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
        "rid": rid,
    }
    if extra_key is not None:
        payload["extra_key"] = extra_key
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    return payload


def native_payload(
    *,
    input_ids: list[int],
    rid: str,
    max_new_tokens: int,
    top_logprobs_num: int,
    extra_key: str | None = None,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "logprob_start_len": -1,
        "top_logprobs_num": top_logprobs_num,
        "rid": rid,
    }
    if extra_key is not None:
        payload["extra_key"] = extra_key
    return payload


def run_openai(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
    top_logprobs_num: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = post_json(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - started
    summary = summarize_openai(response, top_logprobs_num)
    return {
        "endpoint": "openai_chat",
        "rid": payload.get("rid"),
        "extra_key": payload.get("extra_key"),
        "cache_salt": payload.get("cache_salt"),
        "elapsed_s": elapsed,
        "payload": {
            key: payload.get(key)
            for key in (
                "rid",
                "extra_key",
                "cache_salt",
                "temperature",
                "max_tokens",
                "return_prompt_token_ids",
                "return_meta_info",
            )
            if key in payload
        },
        "summary": summary,
    }


def run_native(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
    top_logprobs_num: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = post_json(base_url.rstrip("/") + "/generate", payload, timeout)
    elapsed = time.perf_counter() - started
    summary = summarize_native(response, top_logprobs_num)
    return {
        "endpoint": "native_generate",
        "rid": payload.get("rid"),
        "extra_key": payload.get("extra_key"),
        "elapsed_s": elapsed,
        "payload": {
            "rid": payload.get("rid"),
            "extra_key": payload.get("extra_key"),
            "prompt_token_count": len(payload.get("input_ids") or []),
            "sampling_params": payload.get("sampling_params"),
        },
        "summary": summary,
    }


def token_line(row: dict[str, Any]) -> dict[str, Any]:
    requests = row.get("requests") or []
    result = {"name": row.get("name")}
    for item in requests:
        summary = item.get("summary") or {}
        first = summary.get("first_token") or {}
        result[item.get("endpoint")] = {
            "rid": item.get("rid"),
            "extra_key": item.get("extra_key"),
            "cache_salt": item.get("cache_salt"),
            "token": first.get("token"),
            "token_id": first.get("token_id"),
            "logprob": first.get("logprob"),
            "cached_tokens": (
                summary.get("cached_tokens")
                if item.get("endpoint") == "native_generate"
                else (summary.get("meta_info") or {}).get("cached_tokens")
            ),
        }
    return result


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    case = PROMPTS[args.case]
    endpoint = args.url.rstrip("/")
    prompt = render_prompt(case["messages"], args.model_path)
    max_tokens = args.max_new_tokens or 1

    def mk_openai(rid: str, extra_key: str | None = None, cache_salt: str | None = None):
        return openai_payload(
            model=args.model,
            messages=case["messages"],
            rid=rid,
            max_tokens=max_tokens,
            top_logprobs_num=args.top_logprobs_num,
            extra_key=extra_key,
            cache_salt=cache_salt,
        )

    def mk_native(rid: str, extra_key: str | None = None):
        return native_payload(
            input_ids=prompt["token_ids"],
            rid=rid,
            max_new_tokens=max_tokens,
            top_logprobs_num=args.top_logprobs_num,
            extra_key=extra_key,
        )

    rows: list[dict[str, Any]] = []

    flush = flush_cache(endpoint, args.timeout)
    baseline_requests = [
        run_openai(
            base_url=endpoint,
            payload=mk_openai("openai-first"),
            timeout=args.timeout,
            top_logprobs_num=args.top_logprobs_num,
        ),
        run_native(
            base_url=endpoint,
            payload=mk_native("native-second"),
            timeout=args.timeout,
            top_logprobs_num=args.top_logprobs_num,
        ),
    ]
    rows.append({"name": "baseline_openai_then_native", "flush_before": flush, "requests": baseline_requests})

    flush = flush_cache(endpoint, args.timeout)
    reverse_requests = [
        run_native(
            base_url=endpoint,
            payload=mk_native("native-first"),
            timeout=args.timeout,
            top_logprobs_num=args.top_logprobs_num,
        ),
        run_openai(
            base_url=endpoint,
            payload=mk_openai("openai-second"),
            timeout=args.timeout,
            top_logprobs_num=args.top_logprobs_num,
        ),
    ]
    rows.append({"name": "reverse_native_then_openai", "flush_before": flush, "requests": reverse_requests})

    flush1 = flush_cache(endpoint, args.timeout)
    openai_only = run_openai(
        base_url=endpoint,
        payload=mk_openai("openai-flush-first"),
        timeout=args.timeout,
        top_logprobs_num=args.top_logprobs_num,
    )
    flush2 = flush_cache(endpoint, args.timeout)
    native_after_flush = run_native(
        base_url=endpoint,
        payload=mk_native("native-after-flush"),
        timeout=args.timeout,
        top_logprobs_num=args.top_logprobs_num,
    )
    rows.append(
        {
            "name": "flush_between_openai_native",
            "flush_before": flush1,
            "flush_between": flush2,
            "requests": [openai_only, native_after_flush],
        }
    )

    namespace_error = None
    flush = flush_cache(endpoint, args.timeout)
    try:
        namespace_requests = [
            run_openai(
                base_url=endpoint,
                payload=mk_openai(
                    "openai-namespace-a",
                    extra_key="fp4-order-openai",
                    cache_salt="fp4-order-openai",
                ),
                timeout=args.timeout,
                top_logprobs_num=args.top_logprobs_num,
            ),
            run_native(
                base_url=endpoint,
                payload=mk_native("native-namespace-b", extra_key="fp4-order-native"),
                timeout=args.timeout,
                top_logprobs_num=args.top_logprobs_num,
            ),
        ]
        rows.append(
            {
                "name": "namespace_isolation_extra_key",
                "flush_before": flush,
                "requests": namespace_requests,
            }
        )
    except Exception as exc:
        namespace_error = repr(exc)
        rows.append(
            {
                "name": "namespace_isolation_extra_key",
                "flush_before": flush,
                "error": namespace_error,
                "requests": [],
            }
        )

    report = {
        "schema": "sglang-fp4-request-order-probe/v1",
        "run_id": args.run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": endpoint,
        "model": args.model,
        "model_path": args.model_path,
        "case": args.case,
        "max_new_tokens": max_tokens,
        "top_logprobs_num": args.top_logprobs_num,
        "prompt": {key: value for key, value in prompt.items() if key != "text"},
        "rows": rows,
        "token_summary": [token_line(row) for row in rows],
        "namespace_error": namespace_error,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--case", default="medium_decode", choices=sorted(PROMPTS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--top-logprobs-num", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    report = run_cases(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["token_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
