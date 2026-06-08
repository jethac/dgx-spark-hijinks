#!/usr/bin/env python3
"""Compare vLLM Gemma tensor trace summaries from fp8 and NVFP4-KV rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def event_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("event")), str(row.get("layer_name"))


def select_rows(
    rows: list[dict[str, Any]],
    *,
    occurrence: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = event_key(row)
        if occurrence == "first" and key in selected:
            continue
        selected[key] = row
    return selected


def find_summaries(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if isinstance(value, dict):
        if "shape" in value and "dtype" in value and "numel" in value:
            found[prefix or "<root>"] = value
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.update(find_summaries(child, prefix=child_prefix))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            found.update(find_summaries(child, prefix=child_prefix))
    return found


def compare_number(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    if name not in baseline or name not in candidate:
        return None
    try:
        before = float(baseline[name])
        after = float(candidate[name])
    except (TypeError, ValueError):
        return None
    return {
        "baseline": before,
        "candidate": after,
        "delta": after - before,
        "ratio": (after / before) if before else None,
    }


def topk_ids(summary: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for item in summary.get("topk", []) or []:
        try:
            ids.append(int(item["token_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def compare_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "baseline_shape": baseline.get("shape"),
        "candidate_shape": candidate.get("shape"),
        "baseline_dtype": baseline.get("dtype"),
        "candidate_dtype": candidate.get("dtype"),
    }
    stats: dict[str, Any] = {}
    for name in ("finite", "min", "max", "mean", "rms", "max_abs"):
        value = compare_number(baseline, candidate, name)
        if value is not None:
            stats[name] = value
    if stats:
        comparison["stats"] = stats
    baseline_topk = topk_ids(baseline)
    candidate_topk = topk_ids(candidate)
    if baseline_topk or candidate_topk:
        overlap = sorted(set(baseline_topk) & set(candidate_topk))
        comparison["topk"] = {
            "baseline_ids": baseline_topk,
            "candidate_ids": candidate_topk,
            "overlap": overlap,
            "overlap_ratio": (
                len(overlap) / min(len(baseline_topk), len(candidate_topk))
                if baseline_topk and candidate_topk
                else 0.0
            ),
        }
    return comparison


def compare_rows(
    baseline_rows: dict[tuple[str, str], dict[str, Any]],
    candidate_rows: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    common_keys = sorted(set(baseline_rows) & set(candidate_rows))
    only_baseline = sorted(set(baseline_rows) - set(candidate_rows))
    only_candidate = sorted(set(candidate_rows) - set(baseline_rows))
    comparisons: list[dict[str, Any]] = []

    for key in common_keys:
        baseline_summaries = find_summaries(baseline_rows[key])
        candidate_summaries = find_summaries(candidate_rows[key])
        summary_keys = sorted(set(baseline_summaries) & set(candidate_summaries))
        tensor_comparisons = {
            summary_key: compare_summaries(
                baseline_summaries[summary_key],
                candidate_summaries[summary_key],
            )
            for summary_key in summary_keys
        }
        comparisons.append(
            {
                "event": key[0],
                "layer_name": key[1],
                "tensor_summaries": tensor_comparisons,
                "only_baseline_summaries": sorted(
                    set(baseline_summaries) - set(candidate_summaries)
                ),
                "only_candidate_summaries": sorted(
                    set(candidate_summaries) - set(baseline_summaries)
                ),
            }
        )

    return {
        "common_event_layers": len(common_keys),
        "only_baseline_event_layers": [
            {"event": event, "layer_name": layer} for event, layer in only_baseline
        ],
        "only_candidate_event_layers": [
            {"event": event, "layer_name": layer} for event, layer in only_candidate
        ],
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--occurrence",
        choices=("first", "last"),
        default="last",
        help="Which record to compare for each event/layer key.",
    )
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    candidate = load_jsonl(args.candidate)
    baseline_counts = Counter(event_key(row) for row in baseline)
    candidate_counts = Counter(event_key(row) for row in candidate)
    report = {
        "schema": "vllm-gemma-tensor-trace-compare/v1",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "occurrence": args.occurrence,
        "baseline_events": {
            f"{event}|{layer}": count
            for (event, layer), count in sorted(baseline_counts.items())
        },
        "candidate_events": {
            f"{event}|{layer}": count
            for (event, layer), count in sorted(candidate_counts.items())
        },
        **compare_rows(
            select_rows(baseline, occurrence=args.occurrence),
            select_rows(candidate, occurrence=args.occurrence),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
