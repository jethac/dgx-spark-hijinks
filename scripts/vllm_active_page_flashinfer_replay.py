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


def _safe_repr(value: Any, limit: int = 240) -> str | None:
    if value is None:
        return None
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr_error:{type(exc).__name__}>"
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _tensor_head(tensor: torch.Tensor | None, limit: int = 16) -> list[Any] | None:
    if tensor is None:
        return None
    tensor = tensor.detach()
    if tensor.numel() == 0:
        return []
    flat = tensor.reshape(-1)
    values: list[Any] = []
    for item in flat[: min(limit, flat.numel())].cpu().tolist():
        values.append(item)
    return values


def _tensor_payload(tensor: torch.Tensor | None, limit: int = 16) -> dict[str, Any] | None:
    if tensor is None:
        return None
    payload: dict[str, Any] = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "head": _tensor_head(tensor, limit),
    }
    if tensor.numel() > limit:
        payload["tail"] = _tensor_head(tensor.reshape(-1)[-limit:], limit)
    try:
        payload["data_ptr"] = int(tensor.data_ptr())
        payload["storage_data_ptr"] = int(tensor.untyped_storage().data_ptr())
    except Exception as exc:
        payload["ptr_error"] = type(exc).__name__
    return payload


def _tuple_payload(
    tensors: tuple[torch.Tensor, ...] | torch.Tensor | None,
    limit: int = 16,
) -> list[dict[str, Any] | None] | dict[str, Any] | None:
    if tensors is None:
        return None
    if isinstance(tensors, tuple):
        return [_tensor_payload(tensor, limit) for tensor in tensors]
    return _tensor_payload(tensors, limit)


def _module_payload(module: object) -> dict[str, Any] | None:
    if module is None:
        return None
    return {
        "type": type(module).__name__,
        "name": getattr(module, "__name__", None),
        "file": getattr(module, "__file__", None),
        "repr": _safe_repr(module),
        "has_plan": hasattr(module, "plan"),
        "has_paged_run": hasattr(module, "paged_run"),
        "has_ragged_run": hasattr(module, "ragged_run"),
    }


def _wrapper_payload(wrapper: Any) -> dict[str, Any]:
    return {
        "wrapper_type": type(wrapper).__name__,
        "backend": getattr(wrapper, "_backend", None),
        "kv_layout": getattr(wrapper, "_kv_layout", None),
        "use_cuda_graph": bool(getattr(wrapper, "_use_cuda_graph", False)),
        "causal": bool(getattr(wrapper, "_causal", False)),
        "window_left": int(getattr(wrapper, "_window_left", -9999)),
        "logits_soft_cap": getattr(wrapper, "_logits_soft_cap", None),
        "sm_scale": getattr(wrapper, "_sm_scale", None),
        "batch_size": int(getattr(wrapper, "_batch_size", -1)),
        "num_qo_heads": int(getattr(wrapper, "_num_qo_heads", -1)),
        "num_kv_heads": int(getattr(wrapper, "_num_kv_heads", -1)),
        "qo_indptr_last": int(getattr(wrapper, "_qo_indptr_last", -1)),
        "max_q_len": int(getattr(wrapper, "_max_q_len", -1)),
        "max_kv_len": int(getattr(wrapper, "_max_kv_len", -1)),
        "workspace_size": int(getattr(wrapper, "_workspace_size", -1)),
        "cached_q_data_type": str(getattr(wrapper, "_cached_q_data_type", None)),
        "cached_kv_data_type": str(getattr(wrapper, "_cached_kv_data_type", None)),
        "cached_o_data_type": str(getattr(wrapper, "_cached_o_data_type", None)),
        "plan_info_type": type(getattr(wrapper, "_plan_info", None)).__name__,
        "plan_info_repr": _safe_repr(getattr(wrapper, "_plan_info", None)),
        "cached_module": _module_payload(getattr(wrapper, "_cached_module", None)),
        "jit_module": _module_payload(getattr(wrapper, "_jit_module", None)),
        "qo_indptr": _tensor_payload(getattr(wrapper, "_qo_indptr_buf", None)),
        "paged_kv_indptr": _tensor_payload(
            getattr(wrapper, "_paged_kv_indptr_buf", None)
        ),
        "paged_kv_indices": _tensor_payload(
            getattr(wrapper, "_paged_kv_indices_buf", None)
        ),
        "paged_kv_last_page_len": _tensor_payload(
            getattr(wrapper, "_paged_kv_last_page_len_buf", None)
        ),
    }


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
        head_dim_vo=query.shape[2],
        causal=True,
        pos_encoding_mode="NONE",
        window_left=int(obj.get("window_left", -1)),
        logits_soft_cap=args.logits_soft_cap,
        sm_scale=sm_scale,
        kv_data_type=torch.uint8,
        q_data_type=query.dtype,
        o_data_type=query.dtype,
        fixed_split_size=-1,
        disable_split_kv=False,
    )
    wrapper_plan = _wrapper_payload(wrapper)
    run_signature = {
        "query": _tensor_payload(query, args.head),
        "kv_cache_arg": _tuple_payload((k_data, v_data), args.head),
        "kv_cache_sf": _tuple_payload((k_sf, v_sf), args.head),
        "out_arg": _tensor_payload(out, args.head),
    }
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
        "wrapper_plan": wrapper_plan,
        "run_signature": run_signature,
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
