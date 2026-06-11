#!/usr/bin/env python3
"""Compare token-level supplied-logprob PPL artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_context(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    contexts = report.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != 1:
        raise ValueError(f"{path} must contain exactly one context")
    return contexts[0]


def load_token_logprobs(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = load_context(path)
    score = context.get("score", {})
    token_logprobs = score.get("token_logprobs")
    if not isinstance(token_logprobs, list):
        raise ValueError(f"{path} has no score.token_logprobs list")
    return context, token_logprobs


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def summarize_windows(rows: list[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    windows = []
    for start in range(0, len(rows), window_size):
        window = rows[start : start + window_size]
        if not window:
            continue
        total_delta = sum(row["delta_nll"] for row in window)
        windows.append(
            {
                "start_position": window[0]["position"],
                "end_position": window[-1]["position"],
                "tokens": len(window),
                "mean_delta_nll": total_delta / len(window),
                "sum_delta_nll": total_delta,
            }
        )
    return windows


def compare(fp8_path: Path, candidate_path: Path, *, window_size: int) -> dict[str, Any]:
    fp8_context, fp8_tokens = load_token_logprobs(fp8_path)
    candidate_context, candidate_tokens = load_token_logprobs(candidate_path)
    if len(fp8_tokens) != len(candidate_tokens):
        raise ValueError(
            f"token count mismatch: fp8={len(fp8_tokens)} candidate={len(candidate_tokens)}"
        )

    rows = []
    for fp8, cand in zip(fp8_tokens, candidate_tokens, strict=True):
        if fp8.get("position") != cand.get("position"):
            raise ValueError(f"position mismatch: {fp8} vs {cand}")
        if fp8.get("token_id") != cand.get("token_id"):
            raise ValueError(f"token_id mismatch: {fp8} vs {cand}")
        delta_nll = float(fp8["logprob"]) - float(cand["logprob"])
        rows.append(
            {
                "position": int(fp8["position"]),
                "token_id": int(fp8["token_id"]),
                "fp8_logprob": float(fp8["logprob"]),
                "candidate_logprob": float(cand["logprob"]),
                "delta_nll": delta_nll,
            }
        )

    deltas = [row["delta_nll"] for row in rows]
    sorted_deltas = sorted(deltas)
    positive = [delta for delta in deltas if delta > 0.0]
    negative = [delta for delta in deltas if delta < 0.0]
    total_delta = sum(deltas)
    windows = summarize_windows(rows, window_size)
    worst_windows = sorted(windows, key=lambda row: row["sum_delta_nll"], reverse=True)[:10]
    best_windows = sorted(windows, key=lambda row: row["sum_delta_nll"])[:10]
    worst_tokens = sorted(rows, key=lambda row: row["delta_nll"], reverse=True)[:25]
    best_tokens = sorted(rows, key=lambda row: row["delta_nll"])[:25]

    return {
        "schema": "sglang-token-logprob-compare/v1",
        "fp8_report": str(fp8_path),
        "candidate_report": str(candidate_path),
        "ctx": fp8_context.get("ctx"),
        "cached_tokens_fp8": fp8_context.get("score", {}).get("cached_tokens"),
        "cached_tokens_candidate": candidate_context.get("score", {}).get("cached_tokens"),
        "num_tokens": len(rows),
        "mean_delta_nll": total_delta / len(rows),
        "sum_delta_nll": total_delta,
        "delta_quantiles": {
            "p00": percentile(sorted_deltas, 0.0),
            "p01": percentile(sorted_deltas, 0.01),
            "p05": percentile(sorted_deltas, 0.05),
            "p25": percentile(sorted_deltas, 0.25),
            "p50": percentile(sorted_deltas, 0.50),
            "p75": percentile(sorted_deltas, 0.75),
            "p95": percentile(sorted_deltas, 0.95),
            "p99": percentile(sorted_deltas, 0.99),
            "p100": percentile(sorted_deltas, 1.0),
        },
        "positive_delta_tokens": len(positive),
        "negative_delta_tokens": len(negative),
        "positive_delta_sum": sum(positive),
        "negative_delta_sum": sum(negative),
        "window_size": window_size,
        "worst_windows": worst_windows,
        "best_windows": best_windows,
        "worst_tokens": worst_tokens,
        "best_tokens": best_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    parser.add_argument("--window-size", type=int, default=256)
    args = parser.parse_args()

    report = compare(
        Path(args.fp8),
        Path(args.candidate),
        window_size=args.window_size,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
