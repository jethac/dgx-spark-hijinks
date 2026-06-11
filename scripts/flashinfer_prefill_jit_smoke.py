#!/usr/bin/env python3
"""Compile a FlashInfer FA2 NVFP4 paged-prefill JIT module.

This is a lightweight guard for the vLLM Gemma NVFP4-KV prefill path: it builds
the same wrapper-level custom module shape without loading a model.
"""

from __future__ import annotations

import json

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper


def main() -> None:
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    jit_args = [
        (
            "vllm_batch_prefill_nvfp4_kv_smoke_"
            "dtype_q_bfloat16_dtype_kv_fp4x2_e2m1_dtype_o_bfloat16_"
            "dtype_idx_int32_head_dim_qk_128_head_dim_vo_128_posenc_0_"
            "swa_1_logits_cap_0_fp16_qk_0"
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
        "DefaultAttention<use_custom_mask, true, false, false>",
        "#include<flashinfer/attention/variants.cuh>",
    ]
    jit_kwargs = {
        "pos_encoding_mode": 0,
        "use_sliding_window": True,
        "use_logits_soft_cap": False,
        "use_fp16_qk_reduction": False,
        "fp8_enabled": False,
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
                "jit_additional_tensor_names": wrapper._jit_additional_tensor_names,
                "jit_module_type": type(wrapper._jit_module).__name__,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
