#!/usr/bin/env python3
"""Run FlashInfer paged prefill directly on dumped vLLM active pages."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vllm_active_page_replay import (  # noqa: E402
    _causal_attention_reference,
    _dequant_nvfp4,
    _deswizzle_v_scale_nhd,
    _gather_pages,
    _metrics,
    _tensor_stats,
)

DESWIZZLE_FLAG = "-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1"


def _ensure_deswizzle_flag(enabled: bool) -> None:
    if not enabled:
        return
    existing = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if "FLASHINFER_PAGED_V_SF_DESWIZZLE" not in existing:
        os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = f"{existing} {DESWIZZLE_FLAG}".strip()


def _configure_flashinfer_source_tree(source_root: Path | None) -> dict[str, str]:
    if source_root is None:
        return {}
    source_root = source_root.resolve()
    from flashinfer.jit import env as jit_env  # type: ignore

    jit_env.FLASHINFER_DATA = source_root
    jit_env.FLASHINFER_CSRC_DIR = source_root / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = source_root / "include"
    jit_env.CUTLASS_INCLUDE_DIRS = [
        source_root / "3rdparty" / "cutlass" / "include",
        source_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]
    jit_env.CCCL_INCLUDE_DIRS = [
        source_root / "3rdparty" / "cccl" / "cub",
        source_root / "3rdparty" / "cccl" / "libcudacxx" / "include",
        source_root / "3rdparty" / "cccl" / "thrust",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = source_root / "3rdparty" / "spdlog" / "include"
    return {"source_root": str(source_root)}


def _remap_indices(obj: dict[str, Any]) -> torch.Tensor:
    active_pages = [int(x) for x in obj["active_pages"].tolist()]
    active_index = {page: i for i, page in enumerate(active_pages)}
    remapped = []
    missing = []
    for page in [int(x) for x in obj["paged_kv_indices"].tolist()]:
        if page not in active_index:
            missing.append(page)
        else:
            remapped.append(active_index[page])
    if missing:
        raise ValueError(
            f"dump does not include active page payloads for referenced pages: {missing[:8]}"
        )
    return torch.tensor(remapped, dtype=torch.int32, device="cuda:0")


def _placed_pages(
    obj: dict[str, Any],
    pages: torch.Tensor,
    *,
    placement: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if placement == "compact":
        return pages, _remap_indices(obj)
    if placement != "sparse":
        raise ValueError(f"unknown page placement: {placement}")

    active_pages = [int(x) for x in obj["active_pages"].tolist()]
    if not active_pages:
        raise ValueError("dump has no active pages")
    max_page = max(active_pages)
    placed = torch.zeros(
        (max_page + 1, *pages.shape[1:]),
        dtype=pages.dtype,
        device=pages.device,
    )
    for src_idx, page in enumerate(active_pages):
        placed[page].copy_(pages[src_idx])
    indices = obj["paged_kv_indices"].to(torch.int32).to(pages.device)
    return placed, indices


def run(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_deswizzle_flag(not args.no_deswizzle_flag)
    import flashinfer  # type: ignore

    source_paths = _configure_flashinfer_source_tree(args.flashinfer_source_root)
    obj = torch.load(args.dump, map_location="cpu")
    if obj.get("schema") != "spark-active-page-prefill-dump/v1":
        raise ValueError(f"{args.dump} has unexpected schema {obj.get('schema')!r}")

    query_cpu = obj["query"].to(torch.bfloat16)
    k_data_cpu, v_data_cpu = obj["kv_data_pages"]
    k_sf_cpu, v_sf_cpu = obj["kv_scale_pages"]
    if args.deswizzle_reference and not args.no_deswizzle_flag:
        v_sf_ref = _deswizzle_v_scale_nhd(torch, v_sf_cpu)
    else:
        v_sf_ref = v_sf_cpu

    key_pages = _dequant_nvfp4(torch, k_data_cpu, k_sf_cpu, obj.get("k_scale", 1.0))
    value_pages = _dequant_nvfp4(torch, v_data_cpu, v_sf_ref, obj.get("v_scale", 1.0))
    key = _gather_pages(torch, key_pages, obj)
    value = _gather_pages(torch, value_pages, obj)
    sm_scale = args.sm_scale if args.sm_scale is not None else query_cpu.shape[-1] ** -0.5
    reference = _causal_attention_reference(
        torch, query_cpu.float(), key, value, sm_scale, args.logits_soft_cap
    )

    query = query_cpu.to("cuda:0")
    k_data_active = k_data_cpu.to("cuda:0")
    v_data_active = v_data_cpu.to("cuda:0")
    k_sf_active = k_sf_cpu.to("cuda:0")
    v_sf_active = v_sf_cpu.to("cuda:0")
    k_data, kv_indices = _placed_pages(obj, k_data_active, placement=args.page_placement)
    v_data, _ = _placed_pages(obj, v_data_active, placement=args.page_placement)
    k_sf, _ = _placed_pages(obj, k_sf_active, placement=args.page_placement)
    v_sf, _ = _placed_pages(obj, v_sf_active, placement=args.page_placement)
    qo_indptr = torch.tensor([0, query.shape[0]], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor(
        [0, int(obj["paged_kv_indices"].numel())], dtype=torch.int32, device="cuda:0"
    )
    last_page_len = obj["paged_kv_last_page_len"].to(torch.int32).to("cuda:0")
    workspace = torch.empty(args.workspace_mb * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    out = torch.empty_like(query)

    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        query.shape[1],
        k_data.shape[2],
        query.shape[2],
        k_data.shape[1],
        causal=True,
        pos_encoding_mode="NONE",
        logits_soft_cap=args.logits_soft_cap,
        sm_scale=sm_scale,
        kv_data_type=torch.uint8,
        q_data_type=query.dtype,
    )
    started = time.perf_counter()
    wrapper.run(
        query,
        (k_data, v_data),
        k_scale=float(obj.get("k_scale", 1.0)),
        v_scale=float(obj.get("v_scale", 1.0)),
        out=out,
        kv_cache_sf=(k_sf, v_sf),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    actual = out.detach().cpu().float()
    out_after = obj["out_after"].float()
    active_v_bytes = _gather_pages(torch, v_data_cpu, obj).float()
    active_v_repeated = active_v_bytes.repeat_interleave(
        query_cpu.shape[1] // value.shape[1], dim=1
    )

    return {
        "schema": "vllm-active-page-flashinfer-replay/v1",
        "dump": str(args.dump),
        "elapsed_sec": elapsed,
        "source_paths": source_paths,
        "flashinfer_extra_cudaflags": os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", ""),
        "layer_name": obj.get("layer_name"),
        "page_placement": args.page_placement,
        "window_left": obj.get("window_left"),
        "active_pages": [int(x) for x in obj["active_pages"].tolist()],
        "paged_kv_indices": [int(x) for x in obj["paged_kv_indices"].tolist()],
        "flashinfer_kv_indices": [int(x) for x in kv_indices.cpu().tolist()],
        "flashinfer_kv_data_shape": list(k_data.shape),
        "flashinfer_kv_scale_shape": list(k_sf.shape),
        "last_page_len": [int(x) for x in obj["paged_kv_last_page_len"].tolist()],
        "logits_soft_cap": args.logits_soft_cap,
        "sm_scale": sm_scale,
        "actual": _tensor_stats(torch, actual, args.head),
        "reference": _tensor_stats(torch, reference, args.head),
        "original_out_after": _tensor_stats(torch, out_after, args.head),
        "active_v_bytes_repeated": _tensor_stats(torch, active_v_repeated, args.head),
        "actual_vs_reference": _metrics(torch, actual, reference),
        "actual_vs_original_out_after": _metrics(torch, actual, out_after),
        "actual_vs_active_v_bytes_repeated": _metrics(
            torch, actual[..., : active_v_repeated.shape[-1]], active_v_repeated
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flashinfer-source-root", type=Path)
    parser.add_argument("--logits-soft-cap", type=float, default=50.0)
    parser.add_argument("--sm-scale", type=float)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--head", type=int, default=16)
    parser.add_argument(
        "--page-placement",
        choices=("compact", "sparse"),
        default="compact",
        help=(
            "compact remaps dumped active pages to 0..N; sparse keeps original "
            "physical page IDs by placing active pages into a zero-padded page array"
        ),
    )
    parser.add_argument("--no-deswizzle-flag", action="store_true")
    parser.add_argument("--deswizzle-reference", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
