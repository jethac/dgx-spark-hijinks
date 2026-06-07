#!/usr/bin/env python3
"""Probe llama.cpp native endpoints for GGUF loglikelihood suitability.

This checks whether `/tokenize` plus `/completion` can expose the pre-sampling
logprob for arbitrary continuation tokens. A top-N generated-token API is only
useful for lm-eval if every target continuation token can be recovered.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
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


def token_ids(tokenize_response: dict[str, Any]) -> list[int]:
    raw = tokenize_response.get("tokens")
    if not isinstance(raw, list):
        raise ValueError("tokenize response has no tokens list")
    ids: list[int] = []
    for item in raw:
        if isinstance(item, int):
            ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), int):
            ids.append(item["id"])
        else:
            raise ValueError(f"unsupported token item: {item!r}")
    return ids


def normalize_logprob(entry: dict[str, Any]) -> float | None:
    if isinstance(entry.get("logprob"), (int, float)):
        return float(entry["logprob"])
    if isinstance(entry.get("prob"), (int, float)) and entry["prob"] > 0:
        return math.log(float(entry["prob"]))
    return None


def extract_top_entries(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the first generated token's top-prob entries across known shapes."""
    candidates: list[Any] = []
    for key in ("completion_probabilities", "probs"):
        value = response.get(key)
        if isinstance(value, list) and value:
            candidates.append(value[0])

    entries: list[dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, dict):
            for key in ("top_logprobs", "top_probs"):
                value = item.get(key)
                if isinstance(value, list):
                    entries.extend(entry for entry in value if isinstance(entry, dict))
            nested = item.get("probs")
            if isinstance(nested, list) and nested:
                first = nested[0]
                if isinstance(first, dict):
                    for key in ("top_logprobs", "top_probs"):
                        value = first.get(key)
                        if isinstance(value, list):
                            entries.extend(
                                entry for entry in value if isinstance(entry, dict)
                            )
    return entries


def find_token_logprob(entries: list[dict[str, Any]], token_id: int) -> float | None:
    for entry in entries:
        if entry.get("id") == token_id:
            return normalize_logprob(entry)
    return None


def classify(results: list[dict[str, Any]]) -> dict[str, Any]:
    all_scored = bool(results) and all(item.get("target_found") for item in results)
    has_unlikely = any(item.get("case") == "unlikely" for item in results)
    unlikely_scored = any(
        item.get("case") == "unlikely" and item.get("target_found") for item in results
    )
    notes = []
    if not all_scored:
        notes.append(
            "native endpoint returns top-N probabilities; lm-eval needs every continuation token"
        )
    return {
        "all_targets_scored": all_scored,
        "unlikely_target_scored": unlikely_scored,
        "looks_lm_eval_candidate": all_scored and has_unlikely and unlikely_scored,
        "notes": notes,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.url.rstrip("/")
    tokenize_url = base_url + "/tokenize"
    completion_url = base_url + "/completion"
    cases = [
        ("likely", args.context, args.likely_continuation),
        ("unlikely", args.context, args.unlikely_continuation),
    ]
    report: dict[str, Any] = {
        "schema": "llamacpp-native-loglikelihood-probe/v1",
        "tokenize_endpoint": tokenize_url,
        "completion_endpoint": completion_url,
        "n_probs": args.n_probs,
        "ok": False,
        "cases": [],
    }

    for case_name, context, continuation in cases:
        context_response = post_json(
            tokenize_url,
            {
                "content": context,
                "add_special": True,
                "with_pieces": True,
            },
            args.timeout,
        )
        continuation_response = post_json(
            tokenize_url,
            {
                "content": continuation,
                "add_special": False,
                "with_pieces": True,
            },
            args.timeout,
        )
        context_tokens = token_ids(context_response)
        continuation_tokens = token_ids(continuation_response)
        scored_tokens: list[dict[str, Any]] = []
        previous: list[int] = []
        for index, target_id in enumerate(continuation_tokens):
            payload = {
                "prompt": context_tokens + previous,
                "n_predict": 1,
                "n_probs": args.n_probs,
                "return_tokens": True,
                "temperature": -1,
                "cache_prompt": False,
            }
            completion_response = post_json(completion_url, payload, args.timeout)
            entries = extract_top_entries(completion_response)
            logprob = find_token_logprob(entries, target_id)
            scored_tokens.append(
                {
                    "index": index,
                    "target_id": target_id,
                    "target_found": logprob is not None,
                    "target_logprob": logprob,
                    "top_entry_count": len(entries),
                    "generated_tokens": completion_response.get("tokens"),
                }
            )
            previous.append(target_id)
        target_found = all(item["target_found"] for item in scored_tokens)
        report["cases"].append(
            {
                "case": case_name,
                "context": context,
                "continuation": continuation,
                "context_tokens": context_response.get("tokens"),
                "continuation_tokens": continuation_response.get("tokens"),
                "target_found": target_found,
                "scored_tokens": scored_tokens,
            }
        )

    report["classification"] = classify(report["cases"])
    report["ok"] = bool(report["classification"]["looks_lm_eval_candidate"])
    return report


def run_self_test() -> dict[str, Any]:
    good_entries = [
        {"id": 101, "logprob": -0.1, "token": " A"},
        {"id": 202, "logprob": -7.5, "token": " z"},
    ]
    missing_entries = [{"id": 101, "logprob": -0.1, "token": " A"}]
    good_response = {"completion_probabilities": [{"top_logprobs": good_entries}]}
    nested_response = {
        "completion_probabilities": [{"probs": [{"top_logprobs": good_entries}]}]
    }
    missing_response = {"completion_probabilities": [{"top_logprobs": missing_entries}]}
    checks = {
        "direct_extract": find_token_logprob(extract_top_entries(good_response), 202)
        == -7.5,
        "nested_extract": find_token_logprob(extract_top_entries(nested_response), 202)
        == -7.5,
        "missing_extract": find_token_logprob(extract_top_entries(missing_response), 202)
        is None,
        "tokenize_ints": token_ids({"tokens": [1, 2, 3]}) == [1, 2, 3],
        "tokenize_objects": token_ids({"tokens": [{"id": 1}, {"id": 2}]}) == [1, 2],
    }
    return {
        "schema": "llamacpp-native-loglikelihood-probe-self-test/v1",
        "ok": all(checks.values()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--n-probs", type=int, default=256)
    parser.add_argument("--context", default="The capital of Japan is")
    parser.add_argument("--likely-continuation", default=" Tokyo")
    parser.add_argument("--unlikely-continuation", default=" zebra")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    started = time.time()
    try:
        report = run_self_test() if args.self_test else run_probe(args)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        report = {
            "schema": "llamacpp-native-loglikelihood-probe/v1",
            "ok": False,
            "error": repr(exc),
        }
    report["elapsed_s"] = round(time.time() - started, 3)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
