#!/usr/bin/env python3
"""Compare bf16 vs NVFP4 readout-capture directories."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch


def load_calls(path: Path) -> dict[int, dict[str, Any]]:
    calls: dict[int, dict[str, Any]] = {}
    for file in sorted(path.glob("readout_call_*.pt")):
        payload = torch.load(file, map_location="cpu", weights_only=False)
        calls[int(payload["call_index"])] = payload
    return calls


def mean(values: list[float]) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else math.nan


def compare_one(label: str, base: dict[int, dict[str, Any]], cur: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call_index in sorted(set(base) & set(cur)):
        b = base[call_index]
        c = cur[call_index]
        b_rows = [int(x) for x in b["selected_local_rows"].tolist()]
        c_rows = [int(x) for x in c["selected_local_rows"].tolist()]
        b_pos = {row: i for i, row in enumerate(b_rows)}
        c_pos = {row: i for i, row in enumerate(c_rows)}
        common_rows = sorted(set(b_pos) & set(c_pos))
        for local_row in common_rows:
            bi = b_pos[local_row]
            ci = c_pos[local_row]
            bh = b["hidden_selected"][bi].float()
            ch = c["hidden_selected"][ci].float()
            hidden_cos = torch.nn.functional.cosine_similarity(bh, ch, dim=0).item()
            denom = torch.linalg.vector_norm(bh).item()
            hidden_rel_l2 = (
                torch.linalg.vector_norm(ch - bh).item() / denom if denom else math.nan
            )
            b_ids = [int(x) for x in b["logits_selected_top_ids"][bi].tolist()]
            c_ids = [int(x) for x in c["logits_selected_top_ids"][ci].tolist()]
            b_vals = [float(x) for x in b["logits_selected_top_values"][bi].tolist()]
            c_vals = [float(x) for x in c["logits_selected_top_values"][ci].tolist()]
            b_set = set(b_ids)
            c_set = set(c_ids)
            union = b_set | c_set
            rows.append(
                {
                    "label": label,
                    "call_index": call_index,
                    "local_row": local_row,
                    "hidden_cos": hidden_cos,
                    "hidden_rel_l2": hidden_rel_l2,
                    "hidden_rms_delta": float(c["hidden_row_rms"][local_row] - b["hidden_row_rms"][local_row]),
                    "hidden_mean_abs_delta": float(
                        c["hidden_row_mean_abs"][local_row] - b["hidden_row_mean_abs"][local_row]
                    ),
                    "logits_top1_match": 1.0 if b_ids and c_ids and b_ids[0] == c_ids[0] else 0.0,
                    "logits_topk_jaccard": len(b_set & c_set) / len(union) if union else math.nan,
                    "logits_top1_delta": (c_vals[0] - b_vals[0]) if b_vals and c_vals else math.nan,
                    "logits_max_delta": float(c["logits_selected_max"][ci] - b["logits_selected_max"][bi]),
                    "logits_lse_delta": float(c["logits_selected_lse"][ci] - b["logits_selected_lse"][bi]),
                }
            )
    return rows


def bucket(local_row: int) -> str:
    for lo, hi in [(0, 64), (64, 256), (256, 512), (512, 768), (768, 1024), (1024, 10**9)]:
        if lo <= local_row < hi:
            return f"{lo}-{hi}"
    return "other"


def emit(rows: list[dict[str, Any]]) -> None:
    print(
        "label\tbucket\tcount\tmean_hidden_cos\tmean_hidden_rel_l2\t"
        "mean_hidden_rms_delta\tmean_logits_top1_match\tmean_logits_topk_jaccard\t"
        "mean_logits_top1_delta\tmean_logits_max_delta\tmean_logits_lse_delta"
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["label"], bucket(int(row["local_row"]))), []).append(row)
        grouped.setdefault((row["label"], "all"), []).append(row)
    for (label, bkt), vals in sorted(grouped.items()):
        print(
            f"{label}\t{bkt}\t{len(vals)}\t"
            f"{mean([v['hidden_cos'] for v in vals]):.9f}\t"
            f"{mean([v['hidden_rel_l2'] for v in vals]):.9f}\t"
            f"{mean([v['hidden_rms_delta'] for v in vals]):+.9f}\t"
            f"{mean([v['logits_top1_match'] for v in vals]):.9f}\t"
            f"{mean([v['logits_topk_jaccard'] for v in vals]):.9f}\t"
            f"{mean([v['logits_top1_delta'] for v in vals]):+.9f}\t"
            f"{mean([v['logits_max_delta'] for v in vals]):+.9f}\t"
            f"{mean([v['logits_lse_delta'] for v in vals]):+.9f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument(
        "--compare",
        action="append",
        required=True,
        help="label=path to an NVFP4 capture directory",
    )
    args = parser.parse_args()

    base = load_calls(args.base)
    if not base:
        raise SystemExit(f"no readout_call_*.pt files in base dir {args.base}")
    rows: list[dict[str, Any]] = []
    for item in args.compare:
        if "=" not in item:
            raise SystemExit(f"--compare must be label=path, got {item!r}")
        label, raw_path = item.split("=", 1)
        cur = load_calls(Path(raw_path))
        if not cur:
            raise SystemExit(f"no readout_call_*.pt files in compare dir {raw_path}")
        rows.extend(compare_one(label, base, cur))
    emit(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
