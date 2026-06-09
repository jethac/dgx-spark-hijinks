#!/usr/bin/env python3
"""Convert llama.cpp echo-logprobs probes into the supplied-token contract shape."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected JSON object")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            for key in ("id", "context", "continuation"):
                if not isinstance(row.get(key), str):
                    raise ValueError(f"{path}:{line_no}: missing string {key}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows found")
    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def token_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), int):
            out.append(int(item["id"]))
    return out


def extract_logprobs(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    logprobs = choices[0].get("logprobs")
    return logprobs if isinstance(logprobs, dict) else {}


def top_logprob_values(top_entry: Any) -> list[float]:
    values: list[float] = []
    if isinstance(top_entry, dict):
        iterable = top_entry.values()
    elif isinstance(top_entry, list):
        iterable = [item.get("logprob") for item in top_entry if isinstance(item, dict)]
    else:
        iterable = []
    for value in iterable:
        if is_number(value):
            values.append(float(value))
    return values


def direct_case_from_probe(
    *,
    task: dict[str, Any],
    probe: dict[str, Any],
    artifact: str,
) -> dict[str, Any]:
    tokenization = probe.get("tokenization") if isinstance(probe.get("tokenization"), dict) else {}
    context_ids = token_ids(tokenization.get("context_token_ids"))
    continuation_ids = token_ids(tokenization.get("continuation_token_ids"))
    expected_prompt_tokens = tokenization.get("expected_prompt_tokens")
    if not isinstance(expected_prompt_tokens, int):
        expected_prompt_tokens = len(context_ids) + len(continuation_ids)

    case: dict[str, Any] = {
        "id": task["id"],
        "context": task["context"],
        "continuation": task["continuation"],
        "expected_greedy": task.get("expected_greedy"),
        "source_artifact": artifact,
        "context_token_ids": context_ids,
        "continuation_token_ids": continuation_ids,
        "continuation_token_logprobs": [],
        "target_logprob_sum": None,
        "all_tokens_greedy": False,
        "lm_eval_loglikelihood_tuple": [None, False],
        "findings": [],
    }

    logprobs = extract_logprobs(probe.get("response") if isinstance(probe.get("response"), dict) else {})
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    top_logprobs = logprobs.get("top_logprobs")
    if not isinstance(tokens, list) or not isinstance(token_logprobs, list):
        case["findings"].append("probe response has no prompt tokens/token_logprobs arrays")
        return case

    start = expected_prompt_tokens - len(continuation_ids)
    end = expected_prompt_tokens
    if start < 0 or end > len(token_logprobs) or end > len(tokens):
        case["findings"].append(
            "probe token_logprobs do not cover supplied continuation span "
            f"[{start}, {end}) of {len(token_logprobs)}"
        )
        return case

    entries: list[dict[str, Any]] = []
    for offset, idx in enumerate(range(start, end)):
        logprob = token_logprobs[idx]
        token_id = continuation_ids[offset] if offset < len(continuation_ids) else None
        token_text = tokens[idx]
        is_greedy = False
        if is_number(logprob) and isinstance(top_logprobs, list) and idx < len(top_logprobs):
            top_values = top_logprob_values(top_logprobs[idx])
            if top_values:
                is_greedy = float(logprob) >= max(top_values) - 1e-6
        entries.append(
            {
                "token_id": token_id,
                "token": token_text,
                "logprob": float(logprob) if is_number(logprob) else None,
                "is_greedy": bool(is_greedy),
            }
        )

    if len(entries) != len(continuation_ids):
        case["findings"].append("continuation token count and extracted logprob count differ")
    finite_logprobs = [entry["logprob"] for entry in entries if is_number(entry.get("logprob"))]
    if len(finite_logprobs) != len(entries):
        case["findings"].append("one or more continuation logprobs are missing/non-finite")

    target_sum = sum(float(value) for value in finite_logprobs)
    all_greedy = bool(entries) and all(bool(entry["is_greedy"]) for entry in entries)
    case.update(
        {
            "continuation_token_logprobs": entries,
            "target_logprob_sum": target_sum if len(finite_logprobs) == len(entries) else None,
            "all_tokens_greedy": all_greedy,
            "lm_eval_loglikelihood_tuple": [
                target_sum if len(finite_logprobs) == len(entries) else None,
                all_greedy,
            ],
        }
    )
    return case


def task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("context")), str(row.get("continuation"))


def probe_key(probe: dict[str, Any]) -> tuple[str, str]:
    payload = probe.get("payload") if isinstance(probe.get("payload"), dict) else {}
    prompt = str(payload.get("prompt", ""))
    tokenization = probe.get("tokenization") if isinstance(probe.get("tokenization"), dict) else {}
    continuation_ids = token_ids(tokenization.get("continuation_token_ids"))
    continuation = ""
    cont_obj = tokenization.get("continuation")
    if isinstance(cont_obj, dict):
        pieces = [
            str(item.get("piece", ""))
            for item in cont_obj.get("tokens", [])
            if isinstance(item, dict)
        ]
        continuation = "".join(pieces)
    if not continuation:
        return prompt, ""
    return prompt[: -len(continuation)], continuation


def build_report(task_rows: list[dict[str, Any]], probe_paths: list[Path]) -> dict[str, Any]:
    probes = [(path, read_json(path)) for path in probe_paths]
    probes_by_key: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path, probe in probes:
        key = probe_key(probe)
        existing = probes_by_key.get(key)
        if existing is None or probe_contract_score(probe) > probe_contract_score(existing[1]):
            probes_by_key[key] = (path, probe)

    cases: list[dict[str, Any]] = []
    for task in task_rows:
        probe_item = probes_by_key.get(task_key(task))
        if probe_item is None:
            cases.append(
                {
                    "id": task["id"],
                    "context": task["context"],
                    "continuation": task["continuation"],
                    "continuation_token_ids": [],
                    "continuation_token_logprobs": [],
                    "target_logprob_sum": None,
                    "all_tokens_greedy": False,
                    "lm_eval_loglikelihood_tuple": [None, False],
                    "findings": ["no matching probe artifact"],
                }
            )
            continue
        path, probe = probe_item
        cases.append(direct_case_from_probe(task=task, probe=probe, artifact=path.as_posix()))

    ok = all(
        case.get("target_logprob_sum") is not None
        and isinstance(case.get("continuation_token_logprobs"), list)
        and len(case["continuation_token_logprobs"]) == len(case.get("continuation_token_ids") or [])
        and not case.get("findings")
        for case in cases
    )
    return {
        "schema": "llamacpp-echo-logprobs-contract-artifact/v1",
        "probe_artifacts": [path.as_posix() for path, _ in probes],
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "target_found": sum(1 for case in cases if case.get("target_logprob_sum") is not None),
            "ok": ok,
        },
        "ok": ok,
    }


def probe_contract_score(probe: dict[str, Any]) -> tuple[int, int, int, int]:
    classification = probe.get("classification") if isinstance(probe.get("classification"), dict) else {}
    logprobs = extract_logprobs(probe.get("response") if isinstance(probe.get("response"), dict) else {})
    token_logprobs = logprobs.get("token_logprobs")
    tokens = logprobs.get("tokens")
    tokenization = probe.get("tokenization") if isinstance(probe.get("tokenization"), dict) else {}
    expected_prompt_tokens = tokenization.get("expected_prompt_tokens")
    if not isinstance(expected_prompt_tokens, int):
        expected_prompt_tokens = 0
    token_count = len(token_logprobs) if isinstance(token_logprobs, list) else 0
    has_prompt_logprobs = isinstance(tokens, list) and isinstance(token_logprobs, list)
    return (
        1 if has_prompt_logprobs else 0,
        1 if token_count >= expected_prompt_tokens > 0 else 0,
        1 if classification.get("looks_lm_eval_compatible") is True else 0,
        token_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL task rows")
    parser.add_argument("--probe", action="append", required=True, help="gguf_logprobs_probe JSON")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = build_report(read_jsonl(Path(args.input)), [Path(item) for item in args.probe])
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
