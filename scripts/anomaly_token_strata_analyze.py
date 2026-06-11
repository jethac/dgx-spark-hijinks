#!/usr/bin/env python3
"""Stratify per-token NLL deltas between a baseline and a comparator KV run.

Consumes vllm-prompt-token-logprobs/v1 dumps (written by
vllm_prompt_ppl_sweep.py --dump-token-logprobs) and answers: WHERE does the
aggregate delta concentrate? Strata:

  - baseline-surprisal bands (easy tokens vs hard tokens under bf16)
  - position buckets across the context window
  - sign decomposition (improved / worsened / exactly tied)
  - top-N |delta| tokens with surrounding text

Delta convention: delta = nll_comparator - nll_baseline, so NEGATIVE means
the comparator (quantized cache) is BETTER, matching the corpus-sweep tables.
Pure stdlib; runs offline on the banked dumps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SURPRISAL_BANDS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, math.inf]


def load_dump(path: Path) -> dict[str, Any]:
    dump = json.loads(path.read_text(encoding="utf-8"))
    if dump.get("schema") != "vllm-prompt-token-logprobs/v1":
        raise ValueError(f"{path}: unexpected schema {dump.get('schema')!r}")
    return dump


def render_token(token_strs: list[str] | None, index: int) -> str:
    if not token_strs:
        return "?"
    return token_strs[index].replace("▁", " ").replace("\n", "\\n")


def context_snippet(token_strs: list[str] | None, index: int, radius: int = 6) -> str:
    if not token_strs:
        return ""
    lo = max(0, index - radius)
    hi = min(len(token_strs), index + radius + 1)
    parts = []
    for i in range(lo, hi):
        tok = render_token(token_strs, i)
        parts.append(f"[{tok}]" if i == index else tok)
    return "".join(parts)


def stratify(
    base: dict[str, Any],
    comp: dict[str, Any],
    *,
    position_buckets: int,
    top_n: int,
) -> dict[str, Any]:
    if base["token_ids"] != comp["token_ids"]:
        raise ValueError("token_ids differ between baseline and comparator dumps")
    n = len(base["token_ids"])
    base_lp = base["token_logprobs"]
    comp_lp = comp["token_logprobs"]

    rows = []  # (index, base_nll, comp_nll, delta)
    for i in range(n):
        if base_lp[i] is None or comp_lp[i] is None:
            continue
        b, c = -base_lp[i], -comp_lp[i]
        rows.append((i, b, c, c - b))
    if not rows:
        raise ValueError("no jointly-scored positions")

    total_delta = sum(r[3] for r in rows)
    mean_delta = total_delta / len(rows)

    def band_label(lo: float, hi: float) -> str:
        return f"[{lo:g},{hi:g})" if math.isfinite(hi) else f">={lo:g}"

    bands = []
    for lo, hi in zip(SURPRISAL_BANDS, SURPRISAL_BANDS[1:]):
        members = [r for r in rows if lo <= r[1] < hi]
        if not members:
            bands.append({"band_nats": band_label(lo, hi), "count": 0})
            continue
        band_sum = sum(r[3] for r in members)
        bands.append(
            {
                "band_nats": band_label(lo, hi),
                "count": len(members),
                "mean_base_nll": sum(r[1] for r in members) / len(members),
                "mean_delta": band_sum / len(members),
                "share_of_total_delta": band_sum / total_delta if total_delta else 0.0,
            }
        )

    buckets = []
    width = max(1, n // position_buckets)
    for b in range(position_buckets):
        lo, hi = b * width, (b + 1) * width if b < position_buckets - 1 else n
        members = [r for r in rows if lo <= r[0] < hi]
        if not members:
            buckets.append({"positions": f"{lo}-{hi - 1}", "count": 0})
            continue
        bucket_sum = sum(r[3] for r in members)
        buckets.append(
            {
                "positions": f"{lo}-{hi - 1}",
                "count": len(members),
                "mean_delta": bucket_sum / len(members),
                "share_of_total_delta": bucket_sum / total_delta if total_delta else 0.0,
            }
        )

    improved = sum(1 for r in rows if r[3] < 0)
    worsened = sum(1 for r in rows if r[3] > 0)
    tied = len(rows) - improved - worsened

    token_strs = base.get("token_strs")
    top = sorted(rows, key=lambda r: abs(r[3]), reverse=True)[:top_n]
    top_rows = [
        {
            "position": i,
            "token": render_token(token_strs, i),
            "base_nll": b,
            "comp_nll": c,
            "delta": d,
            "context": context_snippet(token_strs, i),
        }
        for i, b, c, d in top
    ]

    return {
        "num_scored": len(rows),
        "mean_delta_nats": mean_delta,
        "total_delta_nats": total_delta,
        "sign": {
            "improved": improved,
            "worsened": worsened,
            "tied": tied,
            "improved_frac": improved / len(rows),
            "worsened_frac": worsened / len(rows),
            "mean_abs_delta": sum(abs(r[3]) for r in rows) / len(rows),
        },
        "surprisal_bands": bands,
        "position_buckets": buckets,
        "top_abs_delta_tokens": top_rows,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"## {report['label']}",
        "",
        f"baseline `{report['baseline']}` vs comparator `{report['comparator']}` — "
        f"{report['strata']['num_scored']} tokens, mean delta "
        f"**{report['strata']['mean_delta_nats']:+.4f}** nats "
        f"(negative = quantized better)",
        "",
        "| baseline-NLL band | count | mean delta | share of total |",
        "|---|---|---|---|",
    ]
    for band in report["strata"]["surprisal_bands"]:
        if not band["count"]:
            lines.append(f"| {band['band_nats']} | 0 | — | — |")
            continue
        lines.append(
            f"| {band['band_nats']} | {band['count']} | {band['mean_delta']:+.4f} "
            f"| {band['share_of_total_delta']:+.1%} |"
        )
    lines += ["", "| positions | count | mean delta | share of total |", "|---|---|---|---|"]
    for bucket in report["strata"]["position_buckets"]:
        if not bucket["count"]:
            lines.append(f"| {bucket['positions']} | 0 | — | — |")
            continue
        lines.append(
            f"| {bucket['positions']} | {bucket['count']} | {bucket['mean_delta']:+.4f} "
            f"| {bucket['share_of_total_delta']:+.1%} |"
        )
    sign = report["strata"]["sign"]
    lines += [
        "",
        f"sign: {sign['improved_frac']:.1%} improved / {sign['worsened_frac']:.1%} worsened "
        f"/ {sign['tied']} tied; mean |delta| {sign['mean_abs_delta']:.4f} nats",
        "",
        "| pos | token | base NLL | comp NLL | delta | context |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["strata"]["top_abs_delta_tokens"]:
        ctx = row["context"].replace("|", "\\|")
        tok = row["token"].replace("|", "\\|")
        lines.append(
            f"| {row['position']} | `{tok}` | {row['base_nll']:.3f} | {row['comp_nll']:.3f} "
            f"| {row['delta']:+.3f} | `{ctx}` |"
        )
    return "\n".join(lines) + "\n"


def run_self_test() -> dict[str, Any]:
    ids = list(range(40))
    strs = [f"t{i}" for i in ids]
    base = {"token_ids": ids, "token_strs": strs, "token_logprobs": [None] + [-1.0] * 39}
    comp_lp: list[float | None] = [None] + [-1.0] * 39
    comp_lp[5] = -0.5   # improved easy token
    comp_lp[30] = -3.0  # worsened token late in context
    comp = {"token_ids": ids, "token_strs": strs, "token_logprobs": comp_lp}
    strata = stratify(base, comp, position_buckets=4, top_n=3)
    checks = {
        "total": abs(strata["total_delta_nats"] - 1.5) < 1e-12,
        "sign": strata["sign"]["improved"] == 1 and strata["sign"]["worsened"] == 1,
        "top": strata["top_abs_delta_tokens"][0]["position"] == 30,
        "band": next(
            b for b in strata["surprisal_bands"] if b["band_nats"] == "[1,2)"
        )["count"] == 39,
        "bucket_late": next(
            b for b in strata["position_buckets"] if b["positions"] == "30-39"
        )["mean_delta"] > 0,
    }
    return {"schema": "anomaly-token-strata-self-test/v1", "ok": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", help="baseline token-logprob dump (bf16 row)")
    parser.add_argument("--comparator", help="comparator token-logprob dump (fp8/nvfp4 row)")
    parser.add_argument("--label", default="token-strata")
    parser.add_argument("--position-buckets", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        report = run_self_test()
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 2

    base = load_dump(Path(args.baseline))
    comp = load_dump(Path(args.comparator))
    report = {
        "schema": "anomaly-token-strata/v1",
        "label": args.label,
        "baseline": base["run_id"],
        "comparator": comp["run_id"],
        "kv_cache_dtypes": {
            "baseline": base.get("kv_cache_dtype"),
            "comparator": comp.get("kv_cache_dtype"),
        },
        "strata": stratify(
            base, comp, position_buckets=args.position_buckets, top_n=args.top_n
        ),
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    md = to_markdown(report)
    if args.output_md:
        Path(args.output_md).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
