#!/usr/bin/env python3
"""Compile a FlashInfer FA2 paged-prefill JIT module with mixed K/V dtypes.

This is a lightweight compile gate for the SGLang radix-cache repair path:
FP8 K preserves QK/LSE precision, while NVFP4 V preserves most of the KV
capacity win. It does not load a model or start a server.
"""

from __future__ import annotations

import json

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper


def main() -> None:
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    jit_args = [
        (
            "batch_prefill_mixed_fp8k_fp4v_smoke_"
            "dtype_q_bfloat16_dtype_k_e4m3_dtype_v_fp4x2_e2m1_"
            "dtype_o_bfloat16_dtype_idx_int32_head_dim_qk_128_"
            "head_dim_vo_128_posenc_0_swa_0_logits_cap_0_fp16_qk_0"
        ),
        torch.bfloat16,
        torch.uint8,
        torch.bfloat16,
        torch.int32,
        128,
        128,
        [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
            "maybe_prefix_len_ptr",
            "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr",
            "maybe_k_cache_sf",
            "maybe_v_cache_sf",
        ],
        [
            "uint8_t",
            "int32_t",
            "float",
            "uint32_t",
            "uint16_t",
            "uint16_t",
            "uint8_t",
            "uint8_t",
        ],
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
            "token_pos_in_items_len",
        ],
        ["double", "double", "double", "double", "int64_t"],
        "DefaultAttention<use_custom_mask, false, false, false>",
        "#include<flashinfer/attention/variants.cuh>",
    ]
    jit_kwargs = {
        "pos_encoding_mode": 0,
        "use_sliding_window": False,
        "use_logits_soft_cap": False,
        "use_fp16_qk_reduction": False,
        "fp8_enabled": False,
        "dtype_k": torch.float8_e4m3fn,
        "dtype_v": torch.uint8,
    }
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend="fa2",
        jit_args=jit_args,
        jit_kwargs=jit_kwargs,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "jit_additional_tensor_names": wrapper._jit_additional_tensor_names,
                "jit_module_type": type(wrapper._jit_module).__name__,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
