#!/usr/bin/env python3
"""Isolate page-size-1 mixed-KV numerical drift in FlashInfer paged prefill."""

from __future__ import annotations

import json
import math
import os
from typing import Any

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


def flatten_pages(
    x: torch.Tensor,
    start: int,
    end: int,
    last_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    full = x[start : end - 1].reshape(-1, num_heads, head_dim)
    last = x[end - 1, :last_len]
    return torch.cat([full, last], dim=0)


def run_case(
    name: str,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_ref_pages: torch.Tensor,
    v_ref_pages: torch.Tensor,
    v_sf: torch.Tensor | None,
    q: torch.Tensor,
    q_indptr_cpu: torch.Tensor,
    kv_indptr_cpu: torch.Tensor,
    kv_indices_cpu: torch.Tensor,
    kv_last_page_len_cpu: torch.Tensor,
    page_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    v_is_fp4 = v_cache.dtype == torch.uint8
    kv_cache_sf = (None, v_sf) if v_is_fp4 else None
    wrapper.plan(
        q_indptr_cpu.to(q.device),
        kv_indptr_cpu.to(q.device),
        kv_indices_cpu.to(q.device),
        kv_last_page_len_cpu.to(q.device),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        kv_data_type=torch.uint8 if v_is_fp4 else dtype,
        k_data_type=k_cache.dtype,
        v_data_type=v_cache.dtype,
        q_data_type=dtype,
    )
    out, lse = wrapper.run(
        q,
        (k_cache, v_cache),
        k_scale=1.0,
        v_scale=1.0,
        kv_cache_sf=kv_cache_sf,
        return_lse=True,
    )

    refs = []
    for i in range(q_indptr_cpu.numel() - 1):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        start = int(kv_indptr_cpu[i])
        end = int(kv_indptr_cpu[i + 1])
        last_len = int(kv_last_page_len_cpu[i])
        ki = flatten_pages(k_ref_pages, start, end, last_len, num_kv_heads, head_dim)
        vi = flatten_pages(v_ref_pages, start, end, last_len, num_kv_heads, head_dim)
        refs.append(
            flashinfer.prefill.single_prefill_with_kv_cache(
                qi,
                ki,
                vi,
                causal=False,
                pos_encoding_mode="NONE",
                logits_soft_cap=0.0,
            )
        )
    ref = torch.cat(refs, dim=0)
    return {
        "case": name,
        "ok": bool(torch.isfinite(out).all().item() and torch.isfinite(lse).all().item()),
        "k_dtype": str(k_cache.dtype),
        "v_dtype": str(v_cache.dtype),
        "cosine": torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        ).item(),
        "max_abs_diff": (out.float() - ref.float()).abs().max().item(),
    }


def main() -> None:
    torch.manual_seed(20260610)
    device = "cuda:0"
    dtype = torch.bfloat16
    batch_size = 2
    kv_len = 96
    qo_len = 32
    page_size = int(os.environ.get("FLASHINFER_MIXED_KV_PAGE_SIZE", "1"))
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 128

    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim, device=device, dtype=dtype)
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    num_pages_per_seq = math.ceil(kv_len / page_size)
    total_pages = num_pages_per_seq * batch_size
    kv_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    k_bf16 = torch.randn(total_pages, page_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_bf16 = torch.randn_like(k_bf16)
    k_fp8 = k_bf16.to(torch.float8_e4m3fn)
    k_fp8_ref = k_fp8.to(dtype)
    v_fp4, v_sf, v_global = create_nvfp4_kv(
        (total_pages, page_size, num_kv_heads, head_dim // 2), device
    )
    assert float(v_global.item()) == 1.0
    v_fp4_ref = nvfp4_to_float(v_fp4, v_sf, v_global).to(dtype)

    cases = [
        run_case(
            "fp8_k_bf16_v",
            k_fp8,
            v_bf16,
            k_fp8_ref,
            v_bf16,
            None,
            q,
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            dtype,
        ),
        run_case(
            "bf16_k_nvfp4_v",
            k_bf16,
            v_fp4,
            k_bf16,
            v_fp4_ref,
            v_sf,
            q,
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            dtype,
        ),
        run_case(
            "fp8_k_nvfp4_v",
            k_fp8,
            v_fp4,
            k_fp8_ref,
            v_fp4_ref,
            v_sf,
            q,
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            dtype,
        ),
    ]
    print(json.dumps({"page_size": page_size, "cases": cases}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
