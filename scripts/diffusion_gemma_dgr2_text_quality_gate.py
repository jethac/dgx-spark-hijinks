#!/usr/bin/env python3
"""Validate the DG-R2 DiffusionGemma text-only quality artifact."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def normalized_text(row: dict) -> str:
    return (
        row["response"]["choices"][0]["message"]["content"]
        .strip()
        .replace("\r\n", "\n")
    )


def check_answer(prompt_id: str, text: str) -> tuple[bool, str]:
    lower = text.lower()
    if prompt_id == "capital_japan":
        return ("tokyo" in lower, "contains Tokyo")
    if prompt_id == "arithmetic_2_plus_2":
        return (re.search(r"(^|[^0-9])4([^0-9]|$)", text) is not None, "contains standalone 4")
    if prompt_id == "dgx_spark_use":
        useful_terms = ("ai" in lower or "machine learning" in lower) and (
            "local" in lower or "desktop" in lower or "development" in lower
        )
        return (useful_terms, "mentions local/desktop/development AI use")
    raise KeyError(prompt_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if artifact.get("schema") != "sglang-diffusiongemma-dgr2-text-quality/v1":
        raise SystemExit(f"unexpected schema: {artifact.get('schema')}")

    by_prompt = defaultdict(list)
    for row in artifact["rows"]:
        by_prompt[row["prompt_id"]].append(row)

    checks = []
    all_ok = True
    for prompt_id, rows in sorted(by_prompt.items()):
        texts = [normalized_text(row) for row in sorted(rows, key=lambda r: r["repeat"])]
        stable = len(set(texts)) == 1
        answer_ok, answer_rule = check_answer(prompt_id, texts[0])
        ok = stable and answer_ok
        all_ok &= ok
        checks.append(
            {
                "prompt_id": prompt_id,
                "ok": ok,
                "stable": stable,
                "answer_ok": answer_ok,
                "answer_rule": answer_rule,
                "responses": texts,
            }
        )

    summary = {
        "schema": "sglang-diffusiongemma-dgr2-text-quality-gate/v1",
        "input": str(args.artifact),
        "all_ok": all_ok,
        "checks": checks,
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text, encoding="utf-8")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
