#!/usr/bin/env python3
"""Compare SGLang FP4-KV dense-vs-cached trace coverage.

This joins request-order probe rows to server-log trace events by rid. The goal is
to make the dense-cache diagnostic packet produce a machine-readable artifact even
before the underlying quality bug is fixed.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TRACE_PATTERNS = {
    "attention": re.compile(r"FP4 KV dense-cache attention trace (?P<payload>\{.*\})"),
    "qwen2": re.compile(r"FP4 KV dense-cache Qwen2 trace (?P<payload>\{.*\})"),
    "logits": re.compile(r"FP4 KV dense-cache logits trace (?P<payload>\{.*\})"),
    "sampler": re.compile(r"FP4 KV dense-cache sampler trace (?P<payload>\{.*\})"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def request_cached_tokens(request: dict[str, Any]) -> int | None:
    summary = request.get("summary") if isinstance(request.get("summary"), dict) else {}
    if request.get("endpoint") == "native_generate":
        value = summary.get("cached_tokens")
    else:
        meta = summary.get("meta_info") if isinstance(summary.get("meta_info"), dict) else {}
        value = meta.get("cached_tokens")
    return int(value) if isinstance(value, int) else None


def request_map(probe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in probe.get("rows", []):
        if not isinstance(row, dict):
            continue
        row_name = row.get("name")
        for request in row.get("requests") or []:
            if not isinstance(request, dict):
                continue
            rid = request.get("rid")
            if not isinstance(rid, str) or not rid:
                continue
            cached = request_cached_tokens(request)
            mapping[rid] = {
                "rid": rid,
                "row": row_name,
                "endpoint": request.get("endpoint"),
                "cached_tokens": cached,
                "cache_class": "cached" if cached and cached > 0 else "dense",
            }
    return mapping


def parse_traces(server_log: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for lineno, line in enumerate(server_log.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        for kind, pattern in TRACE_PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue
            try:
                payload = ast.literal_eval(match.group("payload"))
            except (SyntaxError, ValueError) as exc:
                traces.append(
                    {
                        "kind": kind,
                        "line": lineno,
                        "parse_error": repr(exc),
                        "raw": line.strip(),
                    }
                )
                break
            if not isinstance(payload, dict):
                traces.append(
                    {
                        "kind": kind,
                        "line": lineno,
                        "parse_error": "payload is not a dict",
                        "raw": line.strip(),
                    }
                )
                break
            payload["_kind"] = kind
            payload["_line"] = lineno
            traces.append(payload)
            break
    return traces


def event_rids(event: dict[str, Any]) -> list[str]:
    rids = event.get("rids")
    if isinstance(rids, list):
        return [str(rid) for rid in rids if str(rid)]
    return []


def sample_vector(row: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(row, dict):
        return None
    sample = row.get("sample")
    if not isinstance(sample, list):
        return None
    values: list[float] = []
    for value in sample:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
        else:
            return None
    return values


def first_rows(event: dict[str, Any], field: str) -> list[dict[str, Any]]:
    data = event.get(field)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [row for row in data["rows"] if isinstance(row, dict)]
    return []


def vector_compare(a: list[float] | None, b: list[float] | None) -> dict[str, Any] | None:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    aa = a[:n]
    bb = b[:n]
    dot = sum(x * y for x, y in zip(aa, bb))
    norm_a = math.sqrt(sum(x * x for x in aa))
    norm_b = math.sqrt(sum(y * y for y in bb))
    diff = [x - y for x, y in zip(aa, bb)]
    return {
        "count": n,
        "cosine": dot / (norm_a * norm_b) if norm_a and norm_b else None,
        "max_abs": max(abs(x) for x in diff),
        "rms": math.sqrt(sum(x * x for x in diff) / n),
    }


def topk_overlap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    a_rows = a.get("indices")
    b_rows = b.get("indices")
    if not isinstance(a_rows, list) or not isinstance(b_rows, list) or not a_rows or not b_rows:
        return None
    a_set = {int(value) for value in a_rows[0] if isinstance(value, int)}
    b_set = {int(value) for value in b_rows[0] if isinstance(value, int)}
    if not a_set or not b_set:
        return None
    return {
        "overlap": len(a_set & b_set),
        "a_count": len(a_set),
        "b_count": len(b_set),
        "overlap_ratio": len(a_set & b_set) / min(len(a_set), len(b_set)),
    }


def compare_events(dense: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
    kind = dense.get("_kind")
    if kind == "attention":
        for field in ("merged_rows", "o_rows", "o2_rows", "o1_rows"):
            dense_rows = first_rows(dense, field)
            cached_rows = first_rows(cached, field)
            if dense_rows and cached_rows:
                return {"field": field, "vector": vector_compare(sample_vector(dense_rows[0]), sample_vector(cached_rows[0]))}
    if kind == "qwen2":
        dense_rows = first_rows(dense, "hidden_rows")
        cached_rows = first_rows(cached, "hidden_rows")
        if dense_rows and cached_rows:
            return {"field": "hidden_rows", "vector": vector_compare(sample_vector(dense_rows[0]), sample_vector(cached_rows[0]))}
    if kind in {"logits", "sampler"}:
        return {"field": "topk", "topk": topk_overlap(dense.get("topk"), cached.get("topk"))}
    return {"field": None}


def event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("_kind"),
        event.get("label"),
        event.get("layer"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-cached", action="store_true", default=True)
    parser.add_argument("--require-dense", action="store_true", default=True)
    args = parser.parse_args()

    findings: list[str] = []
    request_path = Path(args.request_json)
    log_path = Path(args.server_log)
    if not request_path.exists():
        raise SystemExit(f"request JSON does not exist: {args.request_json}")
    if not log_path.exists():
        raise SystemExit(f"server log does not exist: {args.server_log}")

    requests = request_map(load_json(request_path))
    traces = parse_traces(log_path)
    parse_errors = [trace for trace in traces if trace.get("parse_error")]
    if parse_errors:
        findings.append(f"{len(parse_errors)} trace line(s) failed to parse")

    events_by_class: dict[str, list[dict[str, Any]]] = {"dense": [], "cached": [], "unknown": []}
    per_rid: dict[str, dict[str, Any]] = {}
    for event in traces:
        if event.get("parse_error"):
            continue
        rids = event_rids(event)
        if not rids:
            events_by_class["unknown"].append(event)
            continue
        for rid in rids:
            req = requests.get(rid)
            if req is None:
                events_by_class["unknown"].append(event)
                continue
            event_copy = dict(event)
            event_copy["_rid"] = rid
            event_copy["_request"] = req
            events_by_class[req["cache_class"]].append(event_copy)
            rid_entry = per_rid.setdefault(
                rid,
                {
                    "request": req,
                    "event_counts": defaultdict(int),
                },
            )
            rid_entry["event_counts"][event_copy["_kind"]] += 1

    if args.require_dense and not events_by_class["dense"]:
        findings.append("no dense/no-cache trace events matched request rids")
    if args.require_cached and not events_by_class["cached"]:
        findings.append("no cached-prefix trace events matched request rids")

    dense_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    cached_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events_by_class["dense"]:
        dense_by_key[event_key(event)].append(event)
    for event in events_by_class["cached"]:
        cached_by_key[event_key(event)].append(event)

    comparisons: list[dict[str, Any]] = []
    for key in sorted(set(dense_by_key) & set(cached_by_key), key=repr):
        dense_event = dense_by_key[key][0]
        cached_event = cached_by_key[key][0]
        comparisons.append(
            {
                "kind": key[0],
                "label": key[1],
                "layer": key[2],
                "dense_rid": dense_event.get("_rid"),
                "cached_rid": cached_event.get("_rid"),
                "dense_count": len(dense_by_key[key]),
                "cached_count": len(cached_by_key[key]),
                "comparison": compare_events(dense_event, cached_event),
            }
        )
    if not comparisons:
        findings.append("no comparable dense/cached trace event keys found")

    report = {
        "schema": "sglang-dense-cache-trace-compare/v1",
        "request_json": args.request_json,
        "server_log": args.server_log,
        "request_count": len(requests),
        "trace_count": len(traces),
        "event_counts": {
            "dense": len(events_by_class["dense"]),
            "cached": len(events_by_class["cached"]),
            "unknown": len(events_by_class["unknown"]),
        },
        "per_rid": {
            rid: {
                "request": entry["request"],
                "event_counts": dict(entry["event_counts"]),
            }
            for rid, entry in sorted(per_rid.items())
        },
        "comparisons": comparisons,
        "parse_errors": parse_errors[:10],
        "ok": not findings,
        "findings": findings,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
