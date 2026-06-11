#!/usr/bin/env python3
"""Run a tiny FlashInfer paged-prefill probe with FP8 K and NVFP4 V.

This is the standalone numerical gate for the SGLang radix-cache repair path.
It exercises mixed cache storage without starting a model server:

* K cache: FP8 e4m3, preserving QK/LSE precision.
* V cache: packed NVFP4 plus FP8 scale factors, preserving most capacity gain.
"""

from __future__ import annotations

import json
import math
import os

import torch
import flashinfer
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float


def create_nvfp4_kv(shape: tuple[int, ...], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    packed &= 0x77
    sf_shape = (*shape[:-1], shape[-1] // 8)
    sf_choices = torch.tensor([56, 48, 40, 32], dtype=torch.uint8, device=device)
    sf = sf_choices[torch.randint(0, len(sf_choices), sf_shape, device=device)]
    return packed, sf, torch.tensor(1.0, device=device)


def nvfp4_to_float(x: torch.Tensor, sf: torch.Tensor, global_sf: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(-1, x.shape[-1])
    sf_flat = sf.reshape(-1, sf.shape[-1])
    x_dq = e2m1_and_ufp8sf_scale_to_float(
        x_flat,
        sf_flat,
        global_sf,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    return x_dq.reshape(*x.shape[:-1], -1).to(x.device)


def main() -> None:
    torch.manual_seed(1234)
    device = "cuda:0"
    q_dtype = torch.bfloat16
    batch_size = 2
    kv_len = 96
    qo_len = 32
    page_size = int(os.environ.get("FLASHINFER_MIXED_KV_PAGE_SIZE", "16"))
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 128
    causal = False

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device=device,
        dtype=q_dtype,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    num_pages_per_seq = math.ceil(kv_len / page_size)
    total_pages = num_pages_per_seq * batch_size
    kv_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    k_ref = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=q_dtype,
    )
    k_cache = k_ref.to(torch.float8_e4m3fn)
    k_dq = k_cache.to(q_dtype)

    v_shape = (total_pages, page_size, num_kv_heads, head_dim // 2)
    v_packed, v_sf, v_global_scale = create_nvfp4_kv(v_shape, device)
    v_dq = nvfp4_to_float(v_packed, v_sf, v_global_scale).to(q_dtype)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper.plan(
        q_indptr_cpu.to(device),
        kv_indptr_cpu.to(device),
        kv_indices_cpu.to(device),
        kv_last_page_len_cpu.to(device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8,
        k_data_type=torch.float8_e4m3fn,
        v_data_type=torch.uint8,
        q_data_type=q_dtype,
    )
    out, lse = wrapper.run(
        q,
        (k_cache, v_packed),
        k_scale=1.0,
        v_scale=float(v_global_scale.item()),
        kv_cache_sf=(None, v_sf),
        return_lse=True,
    )

    refs = []
    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        start = int(kv_indptr_cpu[i])
        end = int(kv_indptr_cpu[i + 1])
        last_len = int(kv_last_page_len_cpu[i])

        full_k = k_dq[start : end - 1].reshape(-1, num_kv_heads, head_dim)
        last_k = k_dq[end - 1, :last_len]
        ki = torch.cat([full_k, last_k], dim=0)

        full_v = v_dq[start : end - 1].reshape(-1, num_kv_heads, head_dim)
        last_v = v_dq[end - 1, :last_len]
        vi = torch.cat([full_v, last_v], dim=0)

        refs.append(
            flashinfer.prefill.single_prefill_with_kv_cache(
                qi,
                ki,
                vi,
                causal=causal,
                pos_encoding_mode="NONE",
                logits_soft_cap=0.0,
            )
        )

    ref = torch.cat(refs, dim=0)
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    max_abs_diff = (out.float() - ref.float()).abs().max().item()

    print(
        json.dumps(
            {
                "ok": bool(torch.isfinite(out).all().item() and torch.isfinite(lse).all().item()),
                "out_shape": list(out.shape),
                "lse_shape": list(lse.shape),
                "cosine_vs_fp8k_nvfp4v_reference": cosine,
                "max_abs_diff_vs_reference": max_abs_diff,
                "cached_k_dtype": str(wrapper._cached_k_data_type),
                "cached_v_dtype": str(wrapper._cached_v_data_type),
                "page_size": page_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
