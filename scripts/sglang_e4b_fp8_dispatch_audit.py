#!/usr/bin/env python3
"""Classify the known SGLang Gemma 4 E4B fp8 FlashInfer dispatcher red."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


TRAIT_RE = re.compile(
    r"NUM_MMA_Q=(?P<num_mma_q>\d+)\s+"
    r"NUM_MMA_D_QK=(?P<num_mma_d_qk>\d+)\s+"
    r"NUM_MMA_D_VO=(?P<num_mma_d_vo>\d+)\s+"
    r"NUM_MMA_KV=(?P<num_mma_kv>\d+)\s+"
    r"NUM_WARPS_Q=(?P<num_warps_q>\d+)\s+"
    r"NUM_WARPS_KV=(?P<num_warps_kv>\d+)"
)
GEOMETRY_RE = re.compile(
    r"FlashInferWrapperGeometry\(num_qo_heads=(?P<num_qo_heads>\d+),\s*"
    r"num_kv_heads=(?P<num_kv_heads>\d+),\s*"
    r"head_dim=(?P<head_dim>\d+),\s*head_dim_vo=(?P<head_dim_vo>\d+)\)"
)


def parse_int_dict(match: re.Match[str]) -> dict[str, int]:
    return {key: int(value) for key, value in match.groupdict().items()}


def read_inputs(paths: list[pathlib.Path]) -> tuple[str, list[str]]:
    chunks: list[str] = []
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks), missing


def first_line_containing(text: str, needle: str) -> str | None:
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return None


def classify(text: str, missing: list[str], input_paths: list[pathlib.Path]) -> dict[str, Any]:
    trait_matches = [parse_int_dict(match) for match in TRAIT_RE.finditer(text)]
    geometry_matches = [parse_int_dict(match) for match in GEOMETRY_RE.finditer(text)]

    has_invalid_config = "Invalid configuration" in text
    has_paged_prefill = "BatchPrefillWithPagedKVCacheDispatched" in text
    has_fp8_module = "dtype_kv=__nv_fp8_e4m3" in text
    has_qk512_vo256 = "head_dim_qk=512" in text and "head_dim_vo=256" in text
    has_page_size_1 = "page_size=1" in text
    has_cta_tile_q64 = "cta_tile_q=64" in text

    expected_trait = {
        "num_mma_q": 1,
        "num_mma_d_qk": 32,
        "num_mma_d_vo": 16,
        "num_mma_kv": 1,
        "num_warps_q": 4,
        "num_warps_kv": 1,
    }
    expected_geometry = {
        "num_qo_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 512,
        "head_dim_vo": 256,
    }
    matching_traits = [
        item for item in trait_matches if all(item.get(k) == v for k, v in expected_trait.items())
    ]
    matching_geometries = [
        item
        for item in geometry_matches
        if all(item.get(k) == v for k, v in expected_geometry.items())
    ]

    known_red = (
        has_invalid_config
        and has_paged_prefill
        and bool(matching_traits)
        and has_fp8_module
        and has_qk512_vo256
        and has_page_size_1
        and has_cta_tile_q64
        and bool(matching_geometries)
    )

    if known_red:
        verdict = "known_e4b_fp8_d512_vo256_num_mma_kv1_dispatcher_red"
        recommended_action = (
            "Keep the SGLang E4B fp8 comparator scoped red until the "
            "FlashInfer D512/VO256 1-byte-KV dispatcher fix is in the packaged "
            "dependency, then rerun the smallest E4B fp8 row."
        )
    elif has_invalid_config and has_paged_prefill:
        verdict = "flashinfer_paged_prefill_invalid_config_unclassified"
        recommended_action = (
            "Preserve the log and inspect the FlashInfer module params before "
            "treating this as the known E4B fp8 dispatcher red."
        )
    else:
        verdict = "known_e4b_fp8_dispatcher_red_not_detected"
        recommended_action = (
            "Do not use this audit as evidence for the E4B fp8 dispatcher red; "
            "the required failure signature was not present."
        )

    return {
        "schema": "sglang-e4b-fp8-dispatch-audit/v1",
        "inputs": [str(path) for path in input_paths],
        "missing_inputs": missing,
        "verdict": verdict,
        "known_red": known_red,
        "signals": {
            "invalid_configuration": has_invalid_config,
            "batch_prefill_paged_dispatch": has_paged_prefill,
            "fp8_module": has_fp8_module,
            "head_dim_qk_512_head_dim_vo_256": has_qk512_vo256,
            "page_size_1": has_page_size_1,
            "cta_tile_q_64": has_cta_tile_q64,
            "matching_trait_count": len(matching_traits),
            "matching_geometry_count": len(matching_geometries),
        },
        "expected_trait": expected_trait,
        "matching_traits": matching_traits,
        "expected_geometry": expected_geometry,
        "matching_geometries": matching_geometries,
        "sample_lines": {
            "invalid_configuration": first_line_containing(text, "Invalid configuration"),
            "geometry": first_line_containing(text, "FlashInferWrapperGeometry"),
            "module": first_line_containing(text, "dtype_kv=__nv_fp8_e4m3"),
        },
        "recommended_action": recommended_action,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    text, missing = read_inputs(args.inputs)
    result = classify(text, missing, args.inputs)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if result["known_red"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
