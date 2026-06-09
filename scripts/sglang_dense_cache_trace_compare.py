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
import tempfile
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
        value = row.get("value")
        if isinstance(value, dict):
            sample = value.get("sample")
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


def comparison_has_metric(comparison: dict[str, Any] | None) -> bool:
    if not isinstance(comparison, dict):
        return False
    vector = comparison.get("vector")
    if isinstance(vector, dict) and isinstance(vector.get("count"), int) and vector["count"] > 0:
        return True
    topk = comparison.get("topk")
    if isinstance(topk, dict) and isinstance(topk.get("overlap"), int):
        return True
    return False


def event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("_kind"),
        event.get("label"),
        event.get("layer"),
        event.get("mode"),
    )


def event_sort_key(event: dict[str, Any]) -> tuple[Any, ...]:
    request = event.get("_request") if isinstance(event.get("_request"), dict) else {}
    return (
        event.get("_line", 0),
        event.get("forward_pass_id") if event.get("forward_pass_id") is not None else -1,
        request.get("row") or "",
        event.get("_rid") or "",
    )


def build_report(
    *,
    request_json: str,
    server_log: str,
    require_cached: bool = True,
    require_dense: bool = True,
) -> dict[str, Any]:
    findings: list[str] = []
    request_path = Path(request_json)
    log_path = Path(server_log)
    if not request_path.exists():
        raise SystemExit(f"request JSON does not exist: {request_json}")
    if not log_path.exists():
        raise SystemExit(f"server log does not exist: {server_log}")

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

    if require_dense and not events_by_class["dense"]:
        findings.append("no dense/no-cache trace events matched request rids")
    if require_cached and not events_by_class["cached"]:
        findings.append("no cached-prefix trace events matched request rids")

    dense_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    cached_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events_by_class["dense"]:
        dense_by_key[event_key(event)].append(event)
    for event in events_by_class["cached"]:
        cached_by_key[event_key(event)].append(event)

    comparisons: list[dict[str, Any]] = []
    for key in sorted(set(dense_by_key) & set(cached_by_key), key=repr):
        dense_events = sorted(dense_by_key[key], key=event_sort_key)
        cached_events = sorted(cached_by_key[key], key=event_sort_key)
        for dense_event in dense_events:
            for cached_event in cached_events:
                comparison = compare_events(dense_event, cached_event)
                comparisons.append(
                    {
                        "kind": key[0],
                        "label": key[1],
                        "layer": key[2],
                        "mode": key[3],
                        "dense_rid": dense_event.get("_rid"),
                        "cached_rid": cached_event.get("_rid"),
                        "dense_line": dense_event.get("_line"),
                        "cached_line": cached_event.get("_line"),
                        "dense_forward_pass_id": dense_event.get("forward_pass_id"),
                        "cached_forward_pass_id": cached_event.get("forward_pass_id"),
                        "dense_count": len(dense_by_key[key]),
                        "cached_count": len(cached_by_key[key]),
                        "comparison": comparison,
                        "metric_ok": comparison_has_metric(comparison),
                    }
                )
    if not comparisons:
        findings.append("no comparable dense/cached trace event keys found")
    metric_comparisons = [item for item in comparisons if item.get("metric_ok") is True]
    if comparisons and not metric_comparisons:
        findings.append("dense/cached trace event keys matched, but no vector or top-k metric was comparable")

    first_divergence = None
    for item in metric_comparisons:
        comparison = item.get("comparison") if isinstance(item.get("comparison"), dict) else {}
        vector = comparison.get("vector") if isinstance(comparison.get("vector"), dict) else None
        topk = comparison.get("topk") if isinstance(comparison.get("topk"), dict) else None
        diverged = False
        if vector is not None:
            diverged = (
                (isinstance(vector.get("max_abs"), (int, float)) and float(vector["max_abs"]) > 1e-6)
                or (
                    isinstance(vector.get("cosine"), (int, float))
                    and math.isfinite(float(vector["cosine"]))
                    and float(vector["cosine"]) < 0.999999
                )
            )
        if topk is not None:
            diverged = isinstance(topk.get("overlap_ratio"), (int, float)) and float(topk["overlap_ratio"]) < 1.0
        if diverged:
            first_divergence = {
                "kind": item.get("kind"),
                "label": item.get("label"),
                "layer": item.get("layer"),
                "field": comparison.get("field"),
                "dense_rid": item.get("dense_rid"),
                "cached_rid": item.get("cached_rid"),
                "comparison": comparison,
            }
            break

    return {
        "schema": "sglang-dense-cache-trace-compare/v1",
        "request_json": request_json,
        "server_log": server_log,
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
        "metric_comparison_count": len(metric_comparisons),
        "first_divergence": first_divergence,
        "parse_errors": parse_errors[:10],
        "ok": not findings,
        "findings": findings,
    }


def run_self_test() -> int:
    request = {
        "rows": [
            {
                "name": "synthetic",
                "requests": [
                    {
                        "rid": "dense-rid",
                        "endpoint": "openai_chat",
                        "summary": {"meta_info": {"cached_tokens": 0}},
                    },
                    {
                        "rid": "cached-rid",
                        "endpoint": "openai_chat",
                        "summary": {"meta_info": {"cached_tokens": 55}},
                    },
                ],
            }
        ]
    }
    dense_trace = {
        "label": "extend_merge_paged",
        "rids": ["dense-rid"],
        "layer": 0,
        "merged_rows": {"rows": [{"row": 0, "value": {"sample": [1.0, 2.0, 3.0]}}]},
    }
    cached_trace = {
        "label": "extend_merge_paged",
        "rids": ["cached-rid"],
        "layer": 0,
        "merged_rows": {"rows": [{"row": 0, "value": {"sample": [1.0, 2.5, 3.0]}}]},
    }
    bad_dense = {"label": "bad", "rids": ["dense-rid"], "layer": 1}
    bad_cached = {"label": "bad", "rids": ["cached-rid"], "layer": 1}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        request_path = tmp_path / "request.json"
        log_path = tmp_path / "server.log"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        log_path.write_text(
            "\n".join(
                [
                    f"INFO FP4 KV dense-cache attention trace {dense_trace!r}",
                    f"INFO FP4 KV dense-cache attention trace {cached_trace!r}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report = build_report(request_json=str(request_path), server_log=str(log_path))
        if report.get("ok") is not True:
            raise AssertionError(f"expected synthetic good report to be ok: {report.get('findings')}")
        if report.get("metric_comparison_count") != 1:
            raise AssertionError("expected exactly one metric comparison")
        if not isinstance(report.get("first_divergence"), dict):
            raise AssertionError("expected first_divergence for changed synthetic vector")

        log_path.write_text(
            "\n".join(
                [
                    f"INFO FP4 KV dense-cache attention trace {bad_dense!r}",
                    f"INFO FP4 KV dense-cache attention trace {bad_cached!r}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        red = build_report(request_json=str(request_path), server_log=str(log_path))
        if red.get("ok") is True:
            raise AssertionError("expected metricless dense/cached comparison to be red")
    print("sglang dense-cache trace compare self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json")
    parser.add_argument("--server-log")
    parser.add_argument("--output")
    parser.add_argument("--require-cached", action="store_true", default=True)
    parser.add_argument("--require-dense", action="store_true", default=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.request_json or not args.server_log:
        parser.error("--request-json and --server-log are required unless --self-test is set")

    report = build_report(
        request_json=args.request_json,
        server_log=args.server_log,
        require_cached=args.require_cached,
        require_dense=args.require_dense,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
