#!/usr/bin/env python3
"""Compare bf16 vs NVFP4 Gemma4DecoderLayer capture directories."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch


PHASES = (
    "input",
    "attention_output",
    "mlp_output",
    "router_logits",
    "moe_output",
    "output",
)


def load_calls(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    calls: dict[tuple[int, int], dict[str, Any]] = {}
    for file in sorted(path.glob("layer_*_call_*.pt")):
        payload = torch.load(file, map_location="cpu", weights_only=False)
        calls[(int(payload["layer_idx"]), int(payload["call_index"]))] = payload
    return calls


def mean(values: list[float]) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else math.nan


def fmt(value: float) -> str:
    return f"{value:.9f}" if math.isfinite(value) else "nan"


def fmt_signed(value: float) -> str:
    return f"{value:+.9f}" if math.isfinite(value) else "nan"


def phase_rows(
    label: str,
    phase: str,
    base: dict[str, Any],
    cur: dict[str, Any],
) -> list[dict[str, Any]]:
    if phase not in base or phase not in cur:
        return []
    b = base[phase]
    c = cur[phase]
    b_rows = [int(x) for x in b["selected_local_rows"].tolist()]
    c_rows = [int(x) for x in c["selected_local_rows"].tolist()]
    b_pos = {row: i for i, row in enumerate(b_rows)}
    c_pos = {row: i for i, row in enumerate(c_rows)}
    rows: list[dict[str, Any]] = []
    for local_row in sorted(set(b_pos) & set(c_pos)):
        bi = b_pos[local_row]
        ci = c_pos[local_row]
        bv = b["selected"][bi].float()
        cv = c["selected"][ci].float()
        denom = torch.linalg.vector_norm(bv).item()
        router_top1_match = math.nan
        router_topk_jaccard = math.nan
        router_margin_delta = math.nan
        if phase == "router_logits":
            topk = min(4, bv.numel(), cv.numel())
            b_vals, b_ids = torch.topk(bv, k=topk)
            c_vals, c_ids = torch.topk(cv, k=topk)
            b_set = {int(x) for x in b_ids.tolist()}
            c_set = {int(x) for x in c_ids.tolist()}
            union = b_set | c_set
            router_top1_match = 1.0 if int(b_ids[0]) == int(c_ids[0]) else 0.0
            router_topk_jaccard = len(b_set & c_set) / len(union) if union else math.nan
            if topk >= 2:
                router_margin_delta = float((c_vals[0] - c_vals[1]) - (b_vals[0] - b_vals[1]))
        rows.append(
            {
                "label": label,
                "layer_idx": int(base.get("layer_idx", -1)),
                "phase": phase,
                "local_row": local_row,
                "cos": torch.nn.functional.cosine_similarity(bv, cv, dim=0).item(),
                "rel_l2": torch.linalg.vector_norm(cv - bv).item() / denom
                if denom
                else math.nan,
                "rms_delta": float(c["row_rms"][local_row] - b["row_rms"][local_row]),
                "mean_abs_delta": float(
                    c["row_mean_abs"][local_row] - b["row_mean_abs"][local_row]
                ),
                "max_abs_delta": float(
                    c["row_max_abs"][local_row] - b["row_max_abs"][local_row]
                ),
                "router_top1_match": router_top1_match,
                "router_topk_jaccard": router_topk_jaccard,
                "router_margin_delta": router_margin_delta,
            }
        )
    return rows


def bucket(local_row: int) -> str:
    for lo, hi in [(0, 64), (64, 256), (256, 512), (512, 768), (768, 1024), (1024, 10**9)]:
        if lo <= local_row < hi:
            return f"{lo}-{hi}"
    return "other"


def compare_one(label: str, base: dict[tuple[int, int], dict[str, Any]], cur: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base) & set(cur)):
        for phase in PHASES:
            rows.extend(phase_rows(label, phase, base[key], cur[key]))
    return rows


def emit(rows: list[dict[str, Any]]) -> None:
    print(
        "label\tlayer\tphase\tbucket\tcount\tmean_cos\tmean_rel_l2\t"
        "mean_rms_delta\tmean_mean_abs_delta\tmean_max_abs_delta\t"
        "mean_router_top1_match\tmean_router_topk_jaccard\t"
        "mean_router_margin_delta"
    )
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["label"], row["layer_idx"], row["phase"], bucket(int(row["local_row"]))),
            [],
        ).append(row)
        grouped.setdefault((row["label"], row["layer_idx"], row["phase"], "all"), []).append(row)
    for (label, layer_idx, phase, bkt), vals in sorted(grouped.items()):
        print(
            f"{label}\t{layer_idx}\t{phase}\t{bkt}\t{len(vals)}\t"
            f"{fmt(mean([v['cos'] for v in vals]))}\t"
            f"{fmt(mean([v['rel_l2'] for v in vals]))}\t"
            f"{fmt_signed(mean([v['rms_delta'] for v in vals]))}\t"
            f"{fmt_signed(mean([v['mean_abs_delta'] for v in vals]))}\t"
            f"{fmt_signed(mean([v['max_abs_delta'] for v in vals]))}\t"
            f"{fmt(mean([v['router_top1_match'] for v in vals]))}\t"
            f"{fmt(mean([v['router_topk_jaccard'] for v in vals]))}\t"
            f"{fmt_signed(mean([v['router_margin_delta'] for v in vals]))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument(
        "--compare",
        action="append",
        required=True,
        help="label=path to an NVFP4 layer-capture directory",
    )
    args = parser.parse_args()

    base = load_calls(args.base)
    if not base:
        raise SystemExit(f"no layer_*_call_*.pt files in base dir {args.base}")
    rows: list[dict[str, Any]] = []
    for item in args.compare:
        if "=" not in item:
            raise SystemExit(f"--compare must be label=path, got {item!r}")
        label, raw_path = item.split("=", 1)
        cur = load_calls(Path(raw_path))
        if not cur:
            raise SystemExit(f"no layer_*_call_*.pt files in compare dir {raw_path}")
        rows.extend(compare_one(label, base, cur))
    emit(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
