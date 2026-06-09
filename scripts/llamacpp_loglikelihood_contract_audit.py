#!/usr/bin/env python3
"""Audit llama.cpp GGUF loglikelihood artifacts against the supplied-token contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return obj


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"{path}:{line_no}: missing string id")
            rows[row_id] = row
    if not rows:
        raise ValueError(f"{path}: no task rows found")
    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def numbers_close(left: Any, right: Any, *, atol: float = 1e-5) -> bool:
    return is_number(left) and is_number(right) and abs(float(left) - float(right)) <= atol


def case_id(case: dict[str, Any]) -> str | None:
    for key in ("id", "case"):
        value = case.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def token_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for item in value:
        if isinstance(item, int):
            ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), int):
            ids.append(int(item["id"]))
        else:
            return []
    return ids


def direct_logprob_entries(case: dict[str, Any]) -> list[dict[str, Any]]:
    entries = case.get("continuation_token_logprobs")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return []


def audit_case(case: dict[str, Any], expected: dict[str, Any] | None) -> dict[str, Any]:
    cid = case_id(case)
    findings: list[str] = []
    continuation_ids = token_ids(
        case.get("continuation_token_ids", case.get("continuation_tokens"))
    )
    expected_sum: float | None = None
    expected_all_greedy: bool | None = None

    direct_entries = direct_logprob_entries(case)
    if direct_entries:
        entry_ids = [
            entry.get("token_id")
            for entry in direct_entries
            if isinstance(entry.get("token_id"), int)
        ]
        if len(direct_entries) != len(continuation_ids):
            findings.append(
                "direct logprob entry count does not match continuation token count "
                f"({len(direct_entries)} != {len(continuation_ids)})"
            )
        if continuation_ids and entry_ids[: len(continuation_ids)] != continuation_ids:
            findings.append(
                f"direct logprob token ids are not in continuation order: {entry_ids}"
            )
        missing_ids = [token_id for token_id in continuation_ids if token_id not in entry_ids]
        if missing_ids:
            findings.append(f"missing direct logprob entries for token ids {missing_ids}")
        direct_logprobs: list[float] = []
        direct_greedy: list[bool] = []
        for entry in direct_entries:
            if not isinstance(entry.get("token_id"), int):
                findings.append("direct logprob entry missing token_id")
            if not is_number(entry.get("logprob")):
                findings.append(
                    f"token {entry.get('token_id', '<unknown>')} has non-finite logprob"
                )
            else:
                direct_logprobs.append(float(entry["logprob"]))
            if not isinstance(entry.get("is_greedy"), bool):
                findings.append(
                    f"token {entry.get('token_id', '<unknown>')} missing boolean is_greedy"
                )
            else:
                direct_greedy.append(bool(entry["is_greedy"]))
        if len(direct_logprobs) == len(direct_entries):
            expected_sum = sum(direct_logprobs)
        if len(direct_greedy) == len(direct_entries) and direct_entries:
            expected_all_greedy = all(direct_greedy)
    else:
        scored = case.get("scored_tokens")
        if not isinstance(scored, list) or not scored:
            findings.append("no continuation_token_logprobs or scored_tokens present")
        else:
            scored_ids: list[int] = []
            scored_logprobs: list[float] = []
            scored_greedy: list[bool] = []
            for item in scored:
                if not isinstance(item, dict):
                    findings.append("scored_tokens contains a non-object entry")
                    continue
                if isinstance(item.get("target_id"), int):
                    scored_ids.append(int(item["target_id"]))
                if item.get("target_found") is not True:
                    findings.append(f"target token {item.get('target_id')} was not scored")
                if not is_number(item.get("target_logprob")):
                    findings.append(
                        f"target token {item.get('target_id')} has non-finite logprob"
                    )
                else:
                    scored_logprobs.append(float(item["target_logprob"]))
                if not isinstance(item.get("target_is_greedy"), bool):
                    findings.append(
                        f"target token {item.get('target_id')} missing boolean target_is_greedy"
                    )
                else:
                    scored_greedy.append(bool(item["target_is_greedy"]))
            if len(scored) != len(continuation_ids):
                findings.append(
                    "scored_tokens count does not match continuation token count "
                    f"({len(scored)} != {len(continuation_ids)})"
                )
            if continuation_ids and scored_ids[: len(continuation_ids)] != continuation_ids:
                findings.append(f"scored token ids are not in continuation order: {scored_ids}")
            missing_ids = [token_id for token_id in continuation_ids if token_id not in scored_ids]
            if missing_ids:
                findings.append(f"missing scored_tokens entries for token ids {missing_ids}")
            if len(scored_logprobs) == len(scored):
                expected_sum = sum(scored_logprobs)
            if len(scored_greedy) == len(scored) and scored:
                expected_all_greedy = all(scored_greedy)

    if not is_number(case.get("target_logprob_sum")):
        findings.append("target_logprob_sum is missing or non-finite")
    elif expected_sum is not None and not numbers_close(case.get("target_logprob_sum"), expected_sum):
        findings.append(
            "target_logprob_sum does not equal sum of continuation token logprobs "
            f"({case.get('target_logprob_sum')} != {expected_sum})"
        )
    if not isinstance(case.get("all_tokens_greedy"), bool):
        findings.append("all_tokens_greedy is missing or not boolean")
    elif expected_all_greedy is not None and case.get("all_tokens_greedy") != expected_all_greedy:
        findings.append(
            "all_tokens_greedy does not equal per-token greedy conjunction "
            f"({case.get('all_tokens_greedy')} != {expected_all_greedy})"
        )

    tuple_value = case.get("lm_eval_loglikelihood_tuple")
    if (
        not isinstance(tuple_value, list)
        or len(tuple_value) != 2
        or not is_number(tuple_value[0])
        or not isinstance(tuple_value[1], bool)
    ):
        findings.append("lm_eval_loglikelihood_tuple is not [finite_number, bool]")
    else:
        if is_number(case.get("target_logprob_sum")) and not numbers_close(tuple_value[0], case["target_logprob_sum"]):
            findings.append("lm_eval tuple logprob does not match target_logprob_sum")
        if isinstance(case.get("all_tokens_greedy"), bool) and tuple_value[1] != case["all_tokens_greedy"]:
            findings.append("lm_eval tuple greedy flag does not match all_tokens_greedy")

    if expected is not None:
        expected_greedy = expected.get("expected_greedy")
        if isinstance(expected_greedy, bool) and case.get("all_tokens_greedy") != expected_greedy:
            findings.append(
                "all_tokens_greedy does not match expected_greedy "
                f"({case.get('all_tokens_greedy')} != {expected_greedy})"
            )
        expected_continuation = expected.get("continuation")
        if (
            isinstance(expected_continuation, str)
            and expected_continuation
            and case.get("continuation") != expected_continuation
        ):
            findings.append("continuation text does not match task row")

    return {
        "id": cid,
        "ok": not findings,
        "findings": findings,
        "continuation_token_ids": continuation_ids,
        "target_logprob_sum": case.get("target_logprob_sum"),
        "all_tokens_greedy": case.get("all_tokens_greedy"),
    }


def audit(report: dict[str, Any], expected_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        return {
            "schema": "llamacpp-loglikelihood-contract-audit/v1",
            "ok": False,
            "error": "artifact has no cases list",
        }

    case_results = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            case_results.append({"id": None, "ok": False, "findings": ["case is not object"]})
            continue
        cid = case_id(case)
        if cid:
            seen.add(cid)
        case_results.append(audit_case(case, expected_rows.get(cid or "")))

    missing_rows = sorted(set(expected_rows) - seen)
    if missing_rows:
        for row_id in missing_rows:
            case_results.append(
                {
                    "id": row_id,
                    "ok": False,
                    "findings": ["expected task row missing from artifact"],
                }
            )

    ok = bool(case_results) and all(item.get("ok") for item in case_results)
    return {
        "schema": "llamacpp-loglikelihood-contract-audit/v1",
        "artifact_schema": report.get("schema"),
        "ok": ok,
        "case_count": len(case_results),
        "cases": case_results,
        "notes": [
            "A green audit requires every supplied continuation token to have a finite logprob.",
            "Top-N artifacts are accepted only when every requested continuation token was scored.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="JSON loglikelihood artifact")
    parser.add_argument(
        "--input",
        default="tasks/llamacpp_loglikelihood_smoke.jsonl",
        help="JSONL task rows used to define required cases",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    report = read_json(Path(args.artifact))
    expected_rows = read_jsonl(Path(args.input))
    result = audit(report, expected_rows)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
