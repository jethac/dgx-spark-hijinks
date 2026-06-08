#!/usr/bin/env python3
"""Quality probe and comparator for OpenAI-compatible serving rows.

This is intentionally heuristic. It does not replace PPL or task accuracy; it makes
"ok=true but garbage text" visible in a machine-readable artifact.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from openai_serving_benchmark import PROMPTS
from spark_hardware import collect_cuda_hardware


WORD_RE = re.compile(r"[A-Za-z0-9_']+")
LEADING_BAD_CHARS = set(",.;:)]}")


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


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    if len(items) < n:
        return []
    return [tuple(items[idx : idx + n]) for idx in range(0, len(items) - n + 1)]


def extract_chat_content(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        return "", {}
    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    text = content if isinstance(content, str) else ""
    if not text and isinstance(reasoning, str):
        text = reasoning
    return text, first


def summarize_logprobs(choice: dict[str, Any]) -> dict[str, Any]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return {"available": False}
    content = logprobs.get("content")
    if not isinstance(content, list):
        return {"available": False, "raw_keys": sorted(logprobs)}
    tokens: list[dict[str, Any]] = []
    for item in content[:8]:
        if not isinstance(item, dict):
            continue
        top = item.get("top_logprobs")
        top_tokens = []
        if isinstance(top, list):
            for candidate in top[:5]:
                if isinstance(candidate, dict):
                    top_tokens.append(
                        {
                            "token": candidate.get("token"),
                            "logprob": candidate.get("logprob"),
                        }
                    )
        tokens.append(
            {
                "token": item.get("token"),
                "logprob": item.get("logprob"),
                "top_logprobs": top_tokens,
            }
        )
    return {
        "available": True,
        "token_count_observed": len(content),
        "first_tokens": tokens,
    }


def quality_metrics(text: str, *, max_tokens: int | None = None) -> dict[str, Any]:
    token_words = words(text)
    word_count = len(token_words)
    counts = Counter(token_words)
    bigrams = ngrams(token_words, 2)
    bigram_counts = Counter(bigrams)
    trigrams = ngrams(token_words, 3)
    trigram_counts = Counter(trigrams)
    unique_word_ratio = (len(counts) / word_count) if word_count else 0.0
    top_word, top_word_count = counts.most_common(1)[0] if counts else ("", 0)
    top_word_fraction = (top_word_count / word_count) if word_count else 0.0
    repeated_bigram_fraction = (
        sum(count - 1 for count in bigram_counts.values() if count > 1) / len(bigrams)
        if bigrams
        else 0.0
    )
    repeated_trigram_fraction = (
        sum(count - 1 for count in trigram_counts.values() if count > 1) / len(trigrams)
        if trigrams
        else 0.0
    )
    stripped = text.lstrip()
    flags: list[str] = []
    if not stripped:
        flags.append("empty")
    if stripped and stripped[0] in LEADING_BAD_CHARS:
        flags.append("starts_with_punctuation")
    if max_tokens and max_tokens >= 32 and len(text) < 40:
        flags.append("too_short_for_requested_decode")
    if word_count >= 40 and unique_word_ratio < 0.35:
        flags.append("low_unique_word_ratio")
    if word_count >= 40 and top_word_fraction > 0.18:
        flags.append("dominant_repeated_word")
    if len(bigrams) >= 20 and repeated_bigram_fraction > 0.18:
        flags.append("repeated_bigrams")
    if len(trigrams) >= 20 and repeated_trigram_fraction > 0.12:
        flags.append("repeated_trigrams")
    return {
        "chars": len(text),
        "word_count": word_count,
        "unique_word_ratio": unique_word_ratio,
        "top_word": top_word,
        "top_word_fraction": top_word_fraction,
        "repeated_bigram_fraction": repeated_bigram_fraction,
        "repeated_trigram_fraction": repeated_trigram_fraction,
        "starts_with": stripped[:40],
        "flags": flags,
        "flagged": bool(flags),
    }


def case_from_benchmark(item: dict[str, Any]) -> dict[str, Any]:
    text = item.get("content_preview") or ""
    payload = item.get("payload") or {}
    max_tokens = payload.get("max_tokens")
    return {
        "case": item.get("case"),
        "source": "benchmark",
        "text": text,
        "finish_reason": item.get("finish_reason"),
        "usage": item.get("usage"),
        "decode_tok_s": item.get("decode_tok_s"),
        "payload": payload,
        "quality": quality_metrics(
            text,
            max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        ),
    }


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("schema")
    if schema == "openai-quality-probe/v1":
        return data
    if schema == "openai-serving-benchmark/v1":
        return {
            "schema": "openai-quality-probe/v1",
            "run_id": data.get("run_id"),
            "backend": data.get("backend"),
            "model": data.get("model"),
            "source_report": path.as_posix(),
            "cases": [case_from_benchmark(item) for item in data.get("cases", [])],
        }
    raise ValueError(f"Unsupported report schema in {path}: {schema!r}")


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    case_names = args.case or sorted(PROMPTS)
    report: dict[str, Any] = {
        "schema": "openai-quality-probe/v1",
        "run_id": args.run_id,
        "backend": args.backend,
        "model": args.model,
        "url": args.url,
        "hardware": collect_cuda_hardware(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": [],
    }
    for case_name in case_names:
        case = PROMPTS[case_name]
        max_tokens = int(case["max_tokens"])
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": case["messages"],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if args.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = args.top_logprobs
        if args.chat_template_kwargs_json:
            payload["chat_template_kwargs"] = json.loads(args.chat_template_kwargs_json)
        case_report: dict[str, Any] = {
            "case": case_name,
            "source": "live",
            "payload": {
                "max_tokens": max_tokens,
                "request_logprobs": args.request_logprobs,
                "top_logprobs": args.top_logprobs if args.request_logprobs else None,
            },
            "ok": False,
        }
        try:
            started = time.perf_counter()
            response = post_json(endpoint, payload, args.timeout)
            elapsed_s = time.perf_counter() - started
            text, choice = extract_chat_content(response)
            case_report.update(
                {
                    "ok": bool(text.strip()),
                    "elapsed_s": elapsed_s,
                    "text": text,
                    "text_preview": text[:800],
                    "finish_reason": choice.get("finish_reason") if choice else None,
                    "usage": response.get("usage"),
                    "quality": quality_metrics(text, max_tokens=max_tokens),
                    "logprobs": summarize_logprobs(choice),
                    "response": response if args.include_response else None,
                }
            )
        except Exception as exc:
            case_report["error"] = repr(exc)
        report["cases"].append(case_report)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return report


def case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item.get("case"): item
        for item in report.get("cases", [])
        if isinstance(item, dict) and item.get("case")
    }


def compare_reports(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_cases = case_map(candidate)
    baseline_cases = case_map(baseline)
    comparisons = []
    for case_name in sorted(set(candidate_cases) | set(baseline_cases)):
        cand = candidate_cases.get(case_name)
        base = baseline_cases.get(case_name)
        if not cand or not base:
            comparisons.append({"case": case_name, "ok": False, "error": "missing case"})
            continue
        cand_text = cand.get("text") or ""
        base_text = base.get("text") or ""
        cand_quality = cand.get("quality") or quality_metrics(cand_text)
        base_quality = base.get("quality") or quality_metrics(base_text)
        comparisons.append(
            {
                "case": case_name,
                "ok": not bool(cand_quality.get("flags")),
                "candidate_flags": cand_quality.get("flags", []),
                "baseline_flags": base_quality.get("flags", []),
                "text_similarity": difflib.SequenceMatcher(
                    None, base_text[:2000], cand_text[:2000]
                ).ratio(),
                "candidate_chars": len(cand_text),
                "baseline_chars": len(base_text),
                "candidate_starts_with": cand_quality.get("starts_with"),
                "baseline_starts_with": base_quality.get("starts_with"),
            }
        )
    return {
        "schema": "openai-quality-compare/v1",
        "candidate_run_id": candidate.get("run_id"),
        "baseline_run_id": baseline.get("run_id"),
        "candidate_source": candidate.get("source_report"),
        "baseline_source": baseline.get("source_report"),
        "ok": all(item.get("ok") for item in comparisons),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--backend", default="unknown")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-report")
    parser.add_argument("--compare-to")
    parser.add_argument("--case", action="append", choices=sorted(PROMPTS), default=[])
    parser.add_argument("--request-logprobs", action="store_true")
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--chat-template-kwargs-json")
    parser.add_argument("--include-response", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.input_report:
        report = load_report(Path(args.input_report))
        report["run_id"] = args.run_id
    else:
        if not args.url or not args.model:
            parser.error("either --input-report or both --url and --model are required")
        report = run_live(args)

    if args.compare_to:
        baseline = load_report(Path(args.compare_to))
        output = compare_reports(report, baseline)
    else:
        output = report

    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if output.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
