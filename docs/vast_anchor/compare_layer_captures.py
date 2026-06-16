#!/usr/bin/env python3
"""Compare bf16 vs NVFP4 Gemma4DecoderLayer capture directories."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch


PHASES = ("input", "attention_output", "mlp_output", "moe_output", "output")


def load_calls(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    calls: dict[tuple[int, int], dict[str, Any]] = {}
    for file in sorted(path.glob("layer_*_call_*.pt")):
        payload = torch.load(file, map_location="cpu", weights_only=False)
        calls[(int(payload["layer_idx"]), int(payload["call_index"]))] = payload
    return calls


def mean(values: list[float]) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return sum(vals) / len(vals) if vals else math.nan


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
        "mean_rms_delta\tmean_mean_abs_delta\tmean_max_abs_delta"
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
            f"{mean([v['cos'] for v in vals]):.9f}\t"
            f"{mean([v['rel_l2'] for v in vals]):.9f}\t"
            f"{mean([v['rms_delta'] for v in vals]):+.9f}\t"
            f"{mean([v['mean_abs_delta'] for v in vals]):+.9f}\t"
            f"{mean([v['max_abs_delta'] for v in vals]):+.9f}"
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
