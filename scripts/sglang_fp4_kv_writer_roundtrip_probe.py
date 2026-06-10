#!/usr/bin/env python3
"""SGLang FP4 KV writer -> FlashInfer FA2 reader roundtrip probe.

This is a weight-free integration gate for the SGLang Gemma rungs. It writes
synthetic dense K/V through MHATokenToKVPoolFP4.set_kv_buffer(), then reads the
stored packed KV and linear scale-factor buffers through FlashInfer paged
prefill. The reference path dequantizes the exact bytes in the pool and runs
FlashInfer dense prefill on those dequantized tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-len", type=int, default=384)
    parser.add_argument("--qo-len", type=int, default=16)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--window-left", type=int, default=255)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--rtol-cosine", type=float, default=0.999)
    return parser.parse_args()


def compare_tensors(out, ref) -> dict[str, Any]:
    import torch

    out_f = out.float()
    ref_f = ref.float()
    diff = out_f - ref_f
    return {
        "finite": bool(torch.isfinite(out_f).all().item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                out_f.flatten(), ref_f.flatten(), dim=0
            ).item()
        ),
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "ref_norm": float(ref_f.norm().item()),
        "out_norm": float(out_f.norm().item()),
    }


def dequant_pool_tensor(pool, packed, scales, global_scale, kv_len, num_heads, head_dim, dtype):
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    return (
        NVFP4KVQuantizeUtil.dequantize(
            packed.view(__import__("torch").uint8),
            scales.reshape(kv_len, -1),
            global_scale,
            dtype=dtype,
        )
        .reshape(kv_len, num_heads, head_dim)
        .contiguous()
    )


def run_case(
    *,
    name: str,
    q,
    pool,
    loc,
    qo_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    window_left: int,
    dtype,
) -> dict[str, Any]:
    import torch
    import flashinfer
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    q_case = q[:qo_len].contiguous()
    q_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=q.device)
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=q.device)
    kv_indices = loc.to(torch.int32)
    kv_last_page_len = torch.tensor([1], dtype=torch.int32, device=q.device)

    k_cache = pool.get_key_buffer(0).unsqueeze(1)
    v_cache = pool.get_value_buffer(0).unsqueeze(1)
    k_sf, v_sf = pool.get_kv_scale_buffer(0)
    if k_sf is None:
        raise RuntimeError("full NVFP4 writer-roundtrip expected K scale factors")
    k_sf_paged = k_sf.unsqueeze(1)
    v_sf_paged = v_sf.unsqueeze(1)
    k_global, v_global = pool.get_kv_global_scale(0)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    sm_scale = 1.0 / math.sqrt(head_dim)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode="NONE",
        sm_scale=sm_scale,
        window_left=window_left,
        logits_soft_cap=0.0,
        q_data_type=dtype,
        kv_data_type=torch.uint8,
        k_data_type=torch.uint8,
        v_data_type=torch.uint8,
        o_data_type=dtype,
    )
    out, lse = wrapper.run(
        q_case,
        (k_cache, v_cache),
        k_scale=float(k_global),
        v_scale=float(v_global),
        kv_cache_sf=(k_sf_paged, v_sf_paged),
        return_lse=True,
    )

    ref_k = dequant_pool_tensor(
        pool,
        k_cache[loc, 0],
        k_sf_paged[loc, 0],
        pool.k_global[0:1],
        kv_len,
        num_kv_heads,
        head_dim,
        dtype,
    )
    ref_v = dequant_pool_tensor(
        pool,
        v_cache[loc, 0],
        v_sf_paged[loc, 0],
        pool.v_global[0:1],
        kv_len,
        num_kv_heads,
        head_dim,
        dtype,
    )
    ref, ref_lse = flashinfer.prefill.single_prefill_with_kv_cache(
        q_case,
        ref_k,
        ref_v,
        causal=False,
        pos_encoding_mode="NONE",
        sm_scale=sm_scale,
        window_left=window_left,
        logits_soft_cap=0.0,
        backend="fa2",
        return_lse=True,
    )

    result = {
        "case": name,
        "qo_len": qo_len,
        "kv_len": kv_len,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "window_left": window_left,
        "k_global": float(k_global),
        "v_global": float(v_global),
        "k_cache_dtype": str(k_cache.dtype),
        "v_cache_dtype": str(v_cache.dtype),
        "k_sf_dtype": str(k_sf_paged.dtype),
        "v_sf_dtype": str(v_sf_paged.dtype),
        "out": compare_tensors(out, ref),
        "lse": compare_tensors(lse, ref_lse),
    }
    return result


def main() -> None:
    # The SGLang linear-SF path must not inherit vLLM's V-SF deswizzle JIT macro.
    extra_cudaflags = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if "FLASHINFER_PAGED_V_SF_DESWIZZLE" in extra_cudaflags:
        raise RuntimeError(
            "SGLang writer-roundtrip must run with linear scale factors; "
            "remove -DFLASHINFER_PAGED_V_SF_DESWIZZLE from FLASHINFER_EXTRA_CUDAFLAGS"
        )
    os.environ["SGLANG_FP4_KV_MIXED_KV"] = "0"

    args = parse_args()

    import torch

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolFP4

    torch.manual_seed(args.seed)
    device = "cuda:0"
    dtype = torch.bfloat16
    page_size = 1

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    pool = MHATokenToKVPoolFP4(
        size=args.kv_len + 8,
        page_size=page_size,
        dtype=fp4_dtype,
        head_num=args.num_kv_heads,
        head_dim=args.head_dim,
        v_head_dim=args.head_dim,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=1,
        enable_alt_stream=False,
    )

    loc = torch.arange(1, args.kv_len + 1, dtype=torch.int64, device=device)
    dense_k = (
        torch.randn(
            args.kv_len,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        * 0.35
    )
    dense_v = (
        torch.randn(
            args.kv_len,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        * 0.35
    )
    layer = SimpleNamespace(
        layer_id=0,
        k_scale=None,
        v_scale=None,
        k_scale_float=None,
        v_scale_float=None,
    )
    pool.set_kv_buffer(layer, loc, dense_k, dense_v)

    q = (
        torch.randn(
            max(args.qo_len, 1),
            args.num_qo_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        * 0.2
    )

    cases = [
        run_case(
            name="global_qo",
            q=q,
            pool=pool,
            loc=loc,
            qo_len=args.qo_len,
            kv_len=args.kv_len,
            num_qo_heads=args.num_qo_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=page_size,
            window_left=-1,
            dtype=dtype,
        ),
        run_case(
            name="swa_qo",
            q=q,
            pool=pool,
            loc=loc,
            qo_len=args.qo_len,
            kv_len=args.kv_len,
            num_qo_heads=args.num_qo_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=page_size,
            window_left=args.window_left,
            dtype=dtype,
        ),
        run_case(
            name="swa_decode_as_prefill",
            q=q,
            pool=pool,
            loc=loc,
            qo_len=1,
            kv_len=args.kv_len,
            num_qo_heads=args.num_qo_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            page_size=page_size,
            window_left=args.window_left,
            dtype=dtype,
        ),
    ]
    all_ok = all(
        case["out"]["finite"]
        and case["lse"]["finite"]
        and case["out"]["cosine"] >= args.rtol_cosine
        for case in cases
    )
    k_size, v_size = pool.get_kv_size_bytes()
    result = {
        "ok": all_ok,
        "probe": "sglang_fp4_kv_writer_roundtrip",
        "scope": "full NVFP4 K+V, SGLang MHATokenToKVPoolFP4 writer, FlashInfer FA2 paged reader, linear scale factors",
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "flashinfer_extra_cudaflags": extra_cudaflags,
        "mixed_fp8_k_nvfp4_v": bool(pool.mixed_fp8_k_nvfp4_v),
        "k_size_bytes": int(k_size),
        "v_size_bytes": int(v_size),
        "cases": cases,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
