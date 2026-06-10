#!/usr/bin/env python3
"""Diagnostic: with VLLM_NVFP4_KV_LINEAR_V_SF=1 latched at first dispatch,
is the C++ writer's V scale-factor layout actually linear, still swizzled,
or neither? Reuses the roundtrip probe's writer/dequant helpers.

Run inside the rebuiltc image with -e VLLM_NVFP4_KV_LINEAR_V_SF=1.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvfp4_writer_roundtrip_probe as probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    diag_args = parser.parse_args()

    env_at_entry = os.environ.get(probe.LINEAR_V_SF_ENV)
    # Belt and braces, same as the probe's linear mode.
    os.environ[probe.LINEAR_V_SF_ENV] = "1"

    import torch
    import vllm._custom_ops  # noqa: F401  registers _C_cache_ops
    import vllm.utils.torch_utils as tu

    args = argparse.Namespace(
        seed=1234, batch_size=2, kv_len=96, qo_len=16, page_size=16,
        num_kv_heads=4, num_qo_heads=8, head_dim=diag_args.head_dim,
        window_left=-1, dtype="bfloat16", kv_layout=["NHD"],
        v_scale_layout="linear", write_chunk_tokens=32,
        flashinfer_source_root=None, k_global_scale=1.0, v_global_scale=1.0,
        writer_cosine_threshold=0.99, writer_max_abs_threshold=1.0,
        reader_cosine_threshold=0.9999, reader_max_abs_threshold=0.25,
        calibrate=False,
    )
    layout = "NHD"
    case = probe._build_case(torch, tu, args, layout)
    (k_data, v_data), (k_sf, v_sf) = case["views"]

    indptr = case["paging"]["kv_indptr"].cpu()
    last_lens = case["paging"]["last_page_len"].cpu()

    def stream_cosine(sf_interpretation):
        pages = probe._dequant_written_pages(
            torch, v_data, sf_interpretation, args.v_global_scale
        )
        parts = []
        for i in range(args.batch_size):
            start, stop = int(indptr[i]), int(indptr[i + 1])
            parts.append(
                probe._gather_sequence(
                    torch, pages, layout, start, stop, int(last_lens[i])
                )
            )
        stream = torch.cat(parts, dim=0)
        return probe._metrics(torch, stream, case["value"])

    as_linear = stream_cosine(v_sf)
    as_swizzled = stream_cosine(probe._deswizzle_v_sf(torch, v_sf, layout))

    # K control (always linear).
    k_pages = probe._dequant_written_pages(torch, k_data, k_sf, args.k_global_scale)
    k_parts = []
    for i in range(args.batch_size):
        start, stop = int(indptr[i]), int(indptr[i + 1])
        k_parts.append(
            probe._gather_sequence(torch, k_pages, layout, start, stop, int(last_lens[i]))
        )
    k_metrics = probe._metrics(torch, torch.cat(k_parts, dim=0), case["key"])

    report = {
        "schema": "nvfp4-linear-latch-diag/v1",
        "head_dim": args.head_dim,
        "env_at_entry": env_at_entry,
        "env_now": os.environ.get(probe.LINEAR_V_SF_ENV),
        "k_dequant_linear": k_metrics,
        "v_dequant_as_linear": as_linear,
        "v_dequant_as_swizzled": as_swizzled,
        "verdict": (
            "writer wrote LINEAR V-SF"
            if as_linear["cosine"] > as_swizzled["cosine"]
            else "writer wrote SWIZZLED V-SF despite env=1 (latch/dispatch ignored)"
        )
        if max(as_linear["cosine"], as_swizzled["cosine"]) >= 0.99
        else "NEITHER interpretation matches: linear path corrupts V-SF",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    diag_args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
