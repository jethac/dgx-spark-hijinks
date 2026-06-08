#!/usr/bin/env python3
"""Probe FlashInfer FA2 NVFP4 paged KV correctness on GB10-class systems."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


DESWIZZLE_FLAG = "-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1"
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_local_imports() -> None:
    scripts_dir = _repo_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _ensure_deswizzle_flag(enabled: bool) -> None:
    if not enabled:
        return
    existing = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if "FLASHINFER_PAGED_V_SF_DESWIZZLE" in existing:
        return
    os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = f"{existing} {DESWIZZLE_FLAG}".strip()


def _configure_flashinfer_source_tree(flashinfer: Any, source_root: Path | None) -> dict[str, str]:
    """Point FlashInfer JIT at a source checkout instead of packaged data files."""
    if source_root is None:
        return {}
    source_root = source_root.resolve()
    from flashinfer.jit import env as jit_env  # type: ignore

    data_paths = {
        "source_root": str(source_root),
        "csrc": str(source_root / "csrc"),
        "include": str(source_root / "include"),
        "cutlass": str(source_root / "3rdparty" / "cutlass"),
        "cccl": str(source_root / "3rdparty" / "cccl"),
        "spdlog": str(source_root / "3rdparty" / "spdlog"),
    }
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
    return data_paths


def _patch_flashinfer_attention_jit_generators(source_root: Path | None) -> dict[str, str]:
    """Use the source checkout's attention JIT helpers with installed FlashInfer.

    NVIDIA's SGLang container ships a FlashInfer Python package that is compatible
    with the container's CUTLASS bindings. Importing a newer checkout wholesale can
    fail before attention JIT runs, but compiling patched headers still needs the
    matching generated parameter structs.
    """
    if source_root is None:
        return {}
    utils_path = source_root.resolve() / "flashinfer" / "jit" / "attention" / "utils.py"
    if not utils_path.exists():
        return {"attention_jit_utils": "missing"}

    spec = importlib.util.spec_from_file_location(
        "_spark_flashinfer_attention_jit_utils", utils_path
    )
    if spec is None or spec.loader is None:
        return {"attention_jit_utils": "unloadable"}

    local_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_utils)

    import flashinfer.jit.attention.modules as attention_modules  # type: ignore
    import flashinfer.jit.attention.utils as attention_utils  # type: ignore

    attention_utils.generate_additional_params = local_utils.generate_additional_params
    attention_modules.generate_additional_params = local_utils.generate_additional_params
    return {"attention_jit_utils": str(utils_path)}


def _pack_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "batch_size": args.batch_size,
        "kv_len": args.kv_len,
        "qo_len": args.qo_len,
        "page_size": args.page_size,
        "num_kv_heads": args.num_kv_heads,
        "num_qo_heads": args.num_qo_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "layouts": args.layouts,
        "kv_container": args.kv_container,
        "v_scale_layout": args.v_scale_layout,
        "signed_values": args.signed_values,
        "backend": "fa2",
        "deswizzle_flag_requested": args.v_scale_layout == "swizzled"
        and not args.no_deswizzle_flag,
        "flashinfer_source_root": str(args.flashinfer_source_root)
        if args.flashinfer_source_root
        else None,
        "causal": args.causal,
        "k_global_scale": args.k_global_scale,
        "v_global_scale": args.v_global_scale,
        "cosine_threshold": args.cosine_threshold,
        "max_abs_threshold": args.max_abs_threshold,
    }


def _make_nvfp4_kv(
    torch: Any,
    shape: tuple[int, ...],
    device: str,
    global_scale: float,
    signed_values: bool,
):
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    if not signed_values:
        packed &= 0x77
    sf_shape = (*shape[:-1], shape[-1] // 8)
    choices = torch.tensor([56, 48, 40, 32], dtype=torch.uint8, device=device)
    sf = choices[torch.randint(0, 4, sf_shape, device=device)]
    return (
        packed,
        sf.view(torch.float8_e4m3fn),
        torch.tensor(global_scale, device=device, dtype=torch.float32),
    )


def _dequant_nvfp4(torch: Any, packed: Any, sf: Any, global_scale: Any):
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    indices = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    lut = torch.tensor(E2M1_TO_FLOAT32, device=packed.device, dtype=torch.float32)
    values = lut[indices.to(torch.long)]
    sf_expanded = sf.to(torch.float32).repeat_interleave(16, dim=-1)
    return values * sf_expanded * global_scale.to(torch.float32)


def _swizzle_v_sf(torch: Any, v_sf: Any, layout: str):
    if layout == "NHD":
        pages, page_size, heads, scale_dim = v_sf.shape
        if page_size % 4 or scale_dim % 4:
            raise ValueError("NHD V-scale swizzle requires page_size and scale_dim divisible by 4")
        return (
            v_sf.reshape(pages, page_size // 4, 4, heads, 4, scale_dim // 4)
            .permute(0, 1, 4, 3, 5, 2)
            .reshape(pages, page_size, heads, scale_dim)
            .contiguous()
        )
    if layout == "HND":
        pages, heads, page_size, scale_dim = v_sf.shape
        if page_size % 4 or scale_dim % 4:
            raise ValueError("HND V-scale swizzle requires page_size and scale_dim divisible by 4")
        return (
            v_sf.reshape(pages, heads, page_size // 4, 4, 4, scale_dim // 4)
            .permute(0, 1, 2, 4, 5, 3)
            .reshape(pages, heads, page_size, scale_dim)
            .contiguous()
        )
    raise ValueError(f"unsupported KV layout: {layout}")


def _gather_sequence(torch: Any, pages: Any, layout: str, start: int, stop: int, last_len: int):
    full = pages[start : stop - 1]
    last = pages[stop - 1]
    if layout == "NHD":
        last = last[:last_len]
        return torch.cat([full.reshape(-1, pages.shape[-2], pages.shape[-1]), last], dim=0)
    last = last[:, :last_len].permute(1, 0, 2)
    full = full.permute(0, 2, 1, 3)
    return torch.cat([full.reshape(-1, pages.shape[1], pages.shape[-1]), last], dim=0)


def _metrics(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    a = actual.detach().float().reshape(-1)
    b = expected.detach().float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = torch.max(torch.abs(a - b)).item()
    mean_abs = torch.mean(torch.abs(a - b)).item()
    return {"cosine": cosine, "max_abs": max_abs, "mean_abs": mean_abs}


def _tensor_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    stats: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": int(finite.sum().item()),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0 or not bool(finite.any().item()):
        return stats
    finite_values = values[finite]
    min_value = float(finite_values.min().item())
    max_value = float(finite_values.max().item())
    stats.update(
        {
            "min": min_value,
            "max": max_value,
            "mean": float(finite_values.mean().item()),
            "rms": float(torch.sqrt(torch.mean(finite_values * finite_values)).item()),
            "max_abs": float(finite_values.abs().max().item()),
            "byte_like_nonnegative": bool(min_value >= 0.0 and max_value <= 255.0),
        }
    )
    return stats


def _dtype(torch: Any, name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _make_case(torch: Any, args: argparse.Namespace, layout: str, dtype: Any):
    device = "cuda:0"
    num_pages_per_seq = math.ceil(args.kv_len / args.page_size)
    total_pages = num_pages_per_seq * args.batch_size
    kv_indptr = (
        torch.arange(args.batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.full(
        (args.batch_size,),
        (args.kv_len - 1) % args.page_size + 1,
        device=device,
        dtype=torch.int32,
    )
    if layout == "NHD":
        kv_shape = (total_pages, args.page_size, args.num_kv_heads, args.head_dim // 2)
    else:
        kv_shape = (total_pages, args.num_kv_heads, args.page_size, args.head_dim // 2)
    k_packed, k_sf, k_scale = _make_nvfp4_kv(
        torch, kv_shape, device, args.k_global_scale, args.signed_values
    )
    v_packed, v_sf_linear, v_scale = _make_nvfp4_kv(
        torch, kv_shape, device, args.v_global_scale, args.signed_values
    )
    k_dq = _dequant_nvfp4(torch, k_packed, k_sf, k_scale).to(dtype)
    v_dq = _dequant_nvfp4(torch, v_packed, v_sf_linear, v_scale).to(dtype)
    v_sf_for_kernel = (
        _swizzle_v_sf(torch, v_sf_linear, layout)
        if args.v_scale_layout == "swizzled"
        else v_sf_linear
    )
    if args.kv_container == "tuple":
        kv_cache = (k_packed, v_packed)
        kv_cache_sf = (k_sf, v_sf_for_kernel)
    else:
        kv_cache = torch.stack([k_packed, v_packed], dim=1)
        kv_cache_sf = torch.stack([k_sf, v_sf_for_kernel], dim=1)
    return {
        "kv_cache": kv_cache,
        "kv_cache_sf": kv_cache_sf,
        "k_dq": k_dq,
        "v_dq": v_dq,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "last_page_len": last_page_len,
    }


def _run_decode(torch: Any, flashinfer: Any, args: argparse.Namespace, layout: str, dtype: Any):
    case = _make_case(torch, args, layout, dtype)
    q = torch.randn(args.batch_size, args.num_qo_heads, args.head_dim, device="cuda:0", dtype=dtype)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    workspace.fill_(0x7F)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, layout, use_tensor_cores=True, backend="fa2"
    )
    wrapper.plan(
        case["kv_indptr"],
        case["kv_indices"],
        case["last_page_len"],
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        args.page_size,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        q_data_type=dtype,
    )
    actual = wrapper.run(
        q,
        case["kv_cache"],
        k_scale=case["k_scale"].item(),
        v_scale=case["v_scale"].item(),
        kv_cache_sf=case["kv_cache_sf"],
    )
    refs = []
    indptr = case["kv_indptr"].cpu()
    last_lens = case["last_page_len"].cpu()
    for i in range(args.batch_size):
        start = int(indptr[i])
        stop = int(indptr[i + 1])
        last_len = int(last_lens[i])
        ki = _gather_sequence(torch, case["k_dq"], layout, start, stop, last_len)
        vi = _gather_sequence(torch, case["v_dq"], layout, start, stop, last_len)
        refs.append(
            flashinfer.decode.single_decode_with_kv_cache(
                q[i], ki, vi, pos_encoding_mode="NONE"
            )
        )
    expected = torch.stack(refs, dim=0)
    metrics = _metrics(torch, actual, expected)
    metrics.update(
        {
            "actual_stats": _tensor_stats(torch, actual),
            "expected_stats": _tensor_stats(torch, expected),
            "k_scale": float(case["k_scale"].item()),
            "v_scale": float(case["v_scale"].item()),
        }
    )
    return metrics


def _run_prefill(torch: Any, flashinfer: Any, args: argparse.Namespace, layout: str, dtype: Any):
    case = _make_case(torch, args, layout, dtype)
    q = torch.randn(
        args.batch_size * args.qo_len,
        args.num_qo_heads,
        args.head_dim,
        device="cuda:0",
        dtype=dtype,
    )
    qo_indptr = (
        torch.arange(args.batch_size + 1, device="cuda:0", dtype=torch.int32)
        * args.qo_len
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    workspace.fill_(0x7F)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        case["kv_indptr"],
        case["kv_indices"],
        case["last_page_len"],
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        args.page_size,
        causal=args.causal,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        q_data_type=dtype,
    )
    actual = wrapper.run(
        q,
        case["kv_cache"],
        k_scale=case["k_scale"].item(),
        v_scale=case["v_scale"].item(),
        kv_cache_sf=case["kv_cache_sf"],
    )
    refs = []
    indptr = case["kv_indptr"].cpu()
    last_lens = case["last_page_len"].cpu()
    qo_cpu = qo_indptr.cpu()
    for i in range(args.batch_size):
        qi = q[int(qo_cpu[i]) : int(qo_cpu[i + 1])]
        start = int(indptr[i])
        stop = int(indptr[i + 1])
        last_len = int(last_lens[i])
        ki = _gather_sequence(torch, case["k_dq"], layout, start, stop, last_len)
        vi = _gather_sequence(torch, case["v_dq"], layout, start, stop, last_len)
        refs.append(
            flashinfer.prefill.single_prefill_with_kv_cache(
                qi,
                ki,
                vi,
                causal=args.causal,
                pos_encoding_mode="NONE",
                logits_soft_cap=0.0,
            )
        )
    expected = torch.cat(refs, dim=0)
    metrics = _metrics(torch, actual, expected)
    metrics.update(
        {
            "actual_stats": _tensor_stats(torch, actual),
            "expected_stats": _tensor_stats(torch, expected),
            "k_scale": float(case["k_scale"].item()),
            "v_scale": float(case["v_scale"].item()),
        }
    )
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_deswizzle_flag(args.v_scale_layout == "swizzled" and not args.no_deswizzle_flag)
    _add_local_imports()

    import torch  # type: ignore
    import flashinfer  # type: ignore
    from spark_hardware import collect_cuda_hardware

    source_paths = _configure_flashinfer_source_tree(
        flashinfer, args.flashinfer_source_root
    )
    source_paths.update(_patch_flashinfer_attention_jit_generators(args.flashinfer_source_root))
    torch.manual_seed(args.seed)
    dtype = _dtype(torch, args.dtype)
    hardware = collect_cuda_hardware()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FlashInfer NVFP4 KV probe.")

    results = []
    for layout in args.layouts:
        for op_name, op in (("decode", _run_decode), ("prefill", _run_prefill)):
            started = time.perf_counter()
            item = {
                "operation": op_name,
                "layout": layout,
                "passed": False,
            }
            try:
                metrics = op(torch, flashinfer, args, layout, dtype)
            except Exception as exc:  # pragma: no cover - diagnostic path
                item.update(
                    {
                        "elapsed_sec": time.perf_counter() - started,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=20),
                    }
                )
                results.append(item)
                continue
            elapsed = time.perf_counter() - started
            passed = (
                metrics["cosine"] >= args.cosine_threshold
                and metrics["max_abs"] <= args.max_abs_threshold
            )
            item.update({"elapsed_sec": elapsed, "passed": passed, **metrics})
            results.append(item)

    return {
        "schema": "flashinfer-nvfp4-kv-probe/v1",
        "created_unix": time.time(),
        "metadata": _pack_metadata(args),
        "environment": {
            "python": sys.version,
            "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
            "flashinfer_file": getattr(flashinfer, "__file__", None),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "flashinfer_extra_cudaflags": os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", ""),
            "flashinfer_source_paths": source_paths,
        },
        "hardware": hardware,
        "results": results,
        "all_ok": all(item["passed"] for item in results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=64)
    parser.add_argument("--qo-len", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--num-qo-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--layouts", nargs="+", choices=["NHD", "HND"], default=["NHD", "HND"])
    parser.add_argument(
        "--kv-container",
        choices=["stacked", "tuple"],
        default="stacked",
        help="Pass paged KV/SF as one stacked tensor or as separate (k, v) tuples.",
    )
    parser.add_argument(
        "--v-scale-layout",
        choices=["swizzled", "linear"],
        default="swizzled",
        help="Use vLLM-style swizzled V scale factors or SGLang-style linear V scale factors.",
    )
    parser.add_argument(
        "--signed-values",
        action="store_true",
        help="Allow negative E2M1 nibbles in synthetic packed K/V values.",
    )
    parser.add_argument("--flashinfer-source-root", type=Path)
    parser.add_argument(
        "--k-global-scale",
        type=float,
        default=1.0,
        help="Global dequantization scale passed as k_scale.",
    )
    parser.add_argument(
        "--v-global-scale",
        type=float,
        default=1.0,
        help="Global dequantization scale passed as v_scale.",
    )
    parser.add_argument("--cosine-threshold", type=float, default=0.995)
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Run paged prefill with a causal mask instead of full attention.",
    )
    parser.add_argument("--max-abs-threshold", type=float, default=0.25)
    parser.add_argument("--no-deswizzle-flag", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
