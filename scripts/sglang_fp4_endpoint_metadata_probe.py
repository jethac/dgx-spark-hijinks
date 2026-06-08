#!/usr/bin/env python3
"""Compare SGLang FP4 OpenAI chat vs native /generate endpoint metadata.

This is intentionally a narrow quality-localization probe. It can either:

* send one-token, metadata-heavy requests to an already-running FP4 SGLang
  server, or
* analyze a prior ``sglang_openai_native_reconcile.py`` artifact.

It does not launch servers, build images, or run capacity rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from openai_serving_benchmark import PROMPTS


TRACE_RE = re.compile(
    r"NVFP4 KV backend trace label=(?P<label>\S+) layer=(?P<layer>\d+) "
    r".*?metadata=(?P<metadata>\{.*?\}) q=(?P<q>\{.*?\}) "
    r".*?k_scale=(?P<k_scale>[-+0-9.eE]+) v_scale=(?P<v_scale>[-+0-9.eE]+)"
)
HTTP_RE = re.compile(r'"POST (?P<route>/v1/chat/completions|/generate) HTTP/1.1"')


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


def sha256_ids(token_ids: list[int] | None) -> str | None:
    if token_ids is None:
        return None
    data = ",".join(str(x) for x in token_ids)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


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


def summarize_openai_response(response: dict[str, Any], top_logprobs_num: int) -> dict[str, Any]:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    first = content[0] if isinstance(content, list) and content else {}
    prompt_ids = choice.get("prompt_token_ids")
    return {
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "prompt_token_count": len(prompt_ids) if isinstance(prompt_ids, list) else None,
        "prompt_token_ids_sha256": sha256_ids(prompt_ids) if isinstance(prompt_ids, list) else None,
        "first_token": normalize_logprob_item(first),
        "first_token_top_logprobs": normalize_top_logprobs(
            first.get("top_logprobs") if isinstance(first, dict) else None,
            top_logprobs_num,
        ),
        "meta_info": choice.get("meta_info"),
    }


def summarize_native_response(response: dict[str, Any], top_logprobs_num: int) -> dict[str, Any]:
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


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    case = PROMPTS[args.case]
    max_new_tokens = args.max_new_tokens or 1
    local_prompt = render_prompt(case["messages"], args.model_path)
    endpoint = args.url.rstrip("/")

    openai_payload = {
        "model": args.model,
        "messages": case["messages"],
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": False,
        "logprobs": True,
        "top_logprobs": args.top_logprobs_num,
        "return_prompt_token_ids": True,
        "return_meta_info": True,
    }
    native_payload = {
        "input_ids": local_prompt["token_ids"],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "logprob_start_len": -1,
        "top_logprobs_num": args.top_logprobs_num,
    }

    openai_response = post_json(
        endpoint + "/v1/chat/completions", openai_payload, args.timeout
    )
    native_response = post_json(endpoint + "/generate", native_payload, args.timeout)

    return build_report(
        run_id=args.run_id,
        source="live",
        case_name=args.case,
        max_new_tokens=max_new_tokens,
        top_logprobs_num=args.top_logprobs_num,
        local_prompt=local_prompt,
        openai_payload=openai_payload,
        openai_summary=summarize_openai_response(openai_response, args.top_logprobs_num),
        native_payload=native_payload,
        native_summary=summarize_native_response(native_response, args.top_logprobs_num),
        fp4_server_log=args.fp4_server_log,
    )


def from_reconcile(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(Path(args.input_reconcile).read_text(encoding="utf-8"))
    fp4 = data["fp4"]
    openai = fp4["openai_chat"]
    native = fp4.get("native_from_openai_prompt_ids") or fp4["native_from_local_render_text"]
    local_prompt = data.get("local_prompt") or {}
    local_prompt = {
        key: value
        for key, value in local_prompt.items()
        if key in {"mode", "model_path", "token_count", "token_ids_sha256"}
    }
    if "token_count" not in local_prompt:
        local_prompt["token_count"] = fp4.get("prompt_id_comparison", {}).get(
            "local_prompt_token_count"
        )
    if "token_ids_sha256" not in local_prompt:
        local_prompt["token_ids_sha256"] = fp4.get("prompt_id_comparison", {}).get(
            "local_prompt_token_ids_sha256"
        )

    return build_report(
        run_id=args.run_id,
        source=str(args.input_reconcile),
        case_name=data.get("case"),
        max_new_tokens=data.get("max_new_tokens"),
        top_logprobs_num=data.get("top_logprobs_num", args.top_logprobs_num),
        local_prompt=local_prompt,
        openai_payload=openai.get("payload", {}),
        openai_summary=openai["summary"],
        native_payload={
            "prompt_key": native.get("payload_prompt_key"),
            "prompt_token_count": native.get("payload_prompt_token_count"),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": data.get("max_new_tokens"),
            },
            "stream": False,
            "return_logprob": True,
            "return_text_in_logprobs": True,
            "logprob_start_len": -1,
            "top_logprobs_num": data.get("top_logprobs_num", args.top_logprobs_num),
        },
        native_summary=native["summary"],
        fp4_server_log=args.fp4_server_log,
    )


def first_token(summary: dict[str, Any], endpoint: str) -> dict[str, Any]:
    if endpoint == "openai":
        tokens = summary.get("first_tokens") or []
        if tokens:
            return tokens[0]
        return summary.get("first_token") or {}
    tokens = summary.get("output_token_logprobs") or []
    if tokens:
        return tokens[0]
    return summary.get("first_token") or {}


def meta_subset(summary: dict[str, Any], endpoint: str) -> dict[str, Any]:
    if endpoint == "openai":
        return {
            "usage": summary.get("usage"),
            "prompt_token_count": summary.get("prompt_token_count"),
            "prompt_token_ids_sha256": summary.get("prompt_token_ids_sha256"),
            "finish_reason": summary.get("finish_reason"),
            "meta_info": summary.get("meta_info"),
        }
    if "meta_info_subset" in summary:
        return summary["meta_info_subset"]
    return {
        "meta_info_subset": summary.get("meta_info_subset"),
        "prompt_tokens": summary.get("prompt_tokens"),
        "completion_tokens": summary.get("completion_tokens"),
        "cached_tokens": summary.get("cached_tokens"),
        "output_token_logprobs_length": summary.get("output_token_logprobs_length"),
        "finish_reason": summary.get("finish_reason"),
    }


def analyze_trace_log(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    trace_path = Path(path)
    if not trace_path.exists():
        return {"path": str(trace_path), "error": "missing"}

    label_counts: dict[str, int] = {}
    layers_by_label: dict[str, set[int]] = {}
    first_trace_by_label: dict[str, dict[str, Any]] = {}
    http_routes: list[str] = []

    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if route_match := HTTP_RE.search(line):
            http_routes.append(route_match.group("route"))
        trace_match = TRACE_RE.search(line)
        if not trace_match:
            continue
        label = trace_match.group("label")
        layer = int(trace_match.group("layer"))
        label_counts[label] = label_counts.get(label, 0) + 1
        layers_by_label.setdefault(label, set()).add(layer)
        first_trace_by_label.setdefault(
            label,
            {
                "layer": layer,
                "metadata": trace_match.group("metadata"),
                "q": trace_match.group("q"),
                "k_scale": float(trace_match.group("k_scale")),
                "v_scale": float(trace_match.group("v_scale")),
            },
        )

    return {
        "path": str(trace_path),
        "label_counts": label_counts,
        "layers_by_label": {
            label: sorted(layers) for label, layers in layers_by_label.items()
        },
        "first_trace_by_label": first_trace_by_label,
        "http_routes_seen": http_routes,
        "request_tagged": False,
        "limitation": (
            "Existing NVFP4 backend trace lines are not tagged with rid or endpoint, "
            "so they prove which forward labels occurred but cannot bind a given "
            "trace group to OpenAI chat versus native /generate."
        ),
    }


def build_report(
    *,
    run_id: str,
    source: str,
    case_name: str | None,
    max_new_tokens: int | None,
    top_logprobs_num: int,
    local_prompt: dict[str, Any],
    openai_payload: dict[str, Any],
    openai_summary: dict[str, Any],
    native_payload: dict[str, Any],
    native_summary: dict[str, Any],
    fp4_server_log: str | None,
) -> dict[str, Any]:
    openai_first = first_token(openai_summary, "openai")
    native_first = first_token(native_summary, "native")
    openai_prompt_sha = (
        openai_summary.get("prompt_token_ids_sha256")
        or openai_summary.get("prompt_token_ids_sha256".replace("_ids", "_ids"))
    )
    local_prompt_sha = local_prompt.get("token_ids_sha256")
    return {
        "schema": "sglang-fp4-endpoint-metadata-probe/v1",
        "run_id": run_id,
        "source": source,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case": case_name,
        "max_new_tokens": max_new_tokens,
        "local_prompt": {
            key: value for key, value in local_prompt.items() if key != "text"
        },
        "endpoint_request_metadata": {
            "openai_chat": {
                "endpoint": "/v1/chat/completions",
                "prompt_carrier": "messages",
                "sampling_fields": {
                    key: openai_payload.get(key)
                    for key in (
                        "temperature",
                        "max_tokens",
                        "stream",
                        "logprobs",
                        "top_logprobs",
                        "return_prompt_token_ids",
                        "return_meta_info",
                    )
                    if key in openai_payload
                },
            },
            "native_generate": {
                "endpoint": "/generate",
                "prompt_carrier": native_payload.get("prompt_key", "input_ids")
                if "prompt_key" in native_payload
                else ("input_ids" if "input_ids" in native_payload else "text"),
                "prompt_token_count": native_payload.get("prompt_token_count")
                or (
                    len(native_payload.get("input_ids"))
                    if isinstance(native_payload.get("input_ids"), list)
                    else None
                ),
                "sampling_fields": {
                    "sampling_params": native_payload.get("sampling_params"),
                    "stream": native_payload.get("stream"),
                    "return_logprob": native_payload.get("return_logprob"),
                    "return_text_in_logprobs": native_payload.get(
                        "return_text_in_logprobs"
                    ),
                    "logprob_start_len": native_payload.get("logprob_start_len"),
                    "top_logprobs_num": native_payload.get("top_logprobs_num"),
                },
            },
        },
        "response_metadata": {
            "openai_chat": meta_subset(openai_summary, "openai"),
            "native_generate": meta_subset(native_summary, "native"),
        },
        "first_generated_token": {
            "openai_chat": openai_first,
            "native_generate": native_first,
            "same_text": openai_first.get("token") == native_first.get("token"),
            "same_token_id": (
                openai_first.get("token_id") is not None
                and openai_first.get("token_id") == native_first.get("token_id")
            ),
        },
        "prompt_id_reconciliation": {
            "openai_prompt_token_ids_sha256": openai_prompt_sha,
            "local_prompt_token_ids_sha256": local_prompt_sha,
            "openai_matches_local": (
                openai_prompt_sha == local_prompt_sha
                if openai_prompt_sha and local_prompt_sha
                else None
            ),
        },
        "fp4_backend_trace": analyze_trace_log(fp4_server_log),
        "localized_issue": (
            "FP4 OpenAI chat and native /generate can consume the same prompt IDs "
            "but produce different first-token distributions. The smallest missing "
            "signal is request-tagged pre-sampling logits at ModelRunner.sample(), "
            "plus ForwardBatch mode/input_ids/positions/seq_lens/rids for the same "
            "first generated token."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--model-path")
    parser.add_argument("--case", default="medium_decode", choices=sorted(PROMPTS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--top-logprobs-num", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--input-reconcile")
    parser.add_argument("--fp4-server-log")
    args = parser.parse_args()

    if args.input_reconcile:
        report = from_reconcile(args)
    else:
        if not args.url or not args.model or not args.model_path:
            parser.error(
                "live mode requires --url, --model, and --model-path; "
                "offline mode requires --input-reconcile"
            )
        report = run_live(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["first_generated_token"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
