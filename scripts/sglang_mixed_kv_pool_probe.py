#!/usr/bin/env python3
"""Probe SGLang's mixed FP8-K/NVFP4-V KV pool against FlashInfer FA2.

This is a small integration gate for the SGLang radix-cache repair path. It
does not start a server or load a model. It verifies that SGLang's MHA FP4 KV
pool can store K as FP8 and V as packed NVFP4, then feed those buffers to the
patched FlashInfer mixed-KV paged-prefill ABI.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
import flashinfer
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolFP4


def main() -> None:
    os.environ["SGLANG_FP4_KV_MIXED_KV"] = "1"
    torch.manual_seed(20260610)

    device = "cuda:0"
    dtype = torch.bfloat16
    kv_len = 64
    qo_len = 16
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 128
    page_size = 1

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    pool = MHATokenToKVPoolFP4(
        size=kv_len + 8,
        page_size=page_size,
        dtype=fp4_dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=head_dim,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        start_layer=0,
        end_layer=1,
        enable_alt_stream=False,
    )

    loc = torch.arange(kv_len, dtype=torch.int64, device=device)
    dense_k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    dense_v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    layer = SimpleNamespace(
        layer_id=0,
        k_scale=None,
        v_scale=None,
        k_scale_float=None,
        v_scale_float=None,
    )
    pool.set_kv_buffer(layer, loc, dense_k, dense_v)

    k_cache = pool.get_key_buffer(0).unsqueeze(1)
    v_cache = pool.get_value_buffer(0).unsqueeze(1)
    k_sf, v_sf = pool.get_kv_scale_buffer(0)
    assert k_sf is None
    v_sf = v_sf.unsqueeze(1)
    k_global, v_global = pool.get_kv_global_scale(0)
    assert k_global == 1.0

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    kv_indices = loc.to(torch.int32)
    kv_last_page_len = torch.tensor([1], dtype=torch.int32, device=device)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        q_data_type=dtype,
        kv_data_type=torch.uint8,
        k_data_type=torch.float8_e4m3fn,
        v_data_type=torch.uint8,
    )
    out, lse = wrapper.run(
        q,
        (k_cache, v_cache),
        k_scale=1.0,
        v_scale=float(v_global),
        kv_cache_sf=(None, v_sf),
        return_lse=True,
    )

    ref_k = k_cache[loc, 0].to(dtype)
    ref_v = NVFP4KVQuantizeUtil.dequantize(
        v_cache[loc, 0].view(torch.uint8),
        v_sf[loc, 0].reshape(kv_len, -1),
        pool.v_global[0:1],
        dtype=dtype,
    ).reshape(kv_len, num_kv_heads, head_dim)
    ref = flashinfer.prefill.single_prefill_with_kv_cache(
        q,
        ref_k,
        ref_v,
        causal=False,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
    )

    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    max_abs = (out.float() - ref.float()).abs().max().item()

    k_size, v_size = pool.get_kv_size_bytes()
    print(
        json.dumps(
            {
                "ok": bool(torch.isfinite(out).all().item() and torch.isfinite(lse).all().item()),
                "mixed_fp8_k_nvfp4_v": bool(pool.mixed_fp8_k_nvfp4_v),
                "k_buffer_dtype": str(pool.get_key_buffer(0).dtype),
                "v_buffer_dtype": str(pool.get_value_buffer(0).dtype),
                "k_scale_is_none": k_sf is None,
                "k_global": float(k_global),
                "v_global": float(v_global),
                "k_size_bytes": int(k_size),
                "v_size_bytes": int(v_size),
                "cosine_vs_pool_reference": cosine,
                "max_abs_diff_vs_pool_reference": max_abs,
                "out_shape": list(out.shape),
                "lse_shape": list(lse.shape),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
