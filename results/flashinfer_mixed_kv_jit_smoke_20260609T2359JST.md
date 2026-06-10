# FlashInfer mixed-KV JIT smoke - 2026-06-09 23:59 JST

## Scope

Goal: validate the first FlashInfer ABI step for SGLang FP4-KV recovery: independent K and V cache element dtypes, specifically FP8 K + packed NVFP4 V.

This is a compile/JIT smoke only. It does not prove SGLang radix correctness, capacity, or model quality.

## Stack

- Host: GB10 / sm_121 over tailnet
- Main checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367`
- FlashInfer base commit: `c3dae30f2798e70e03da9e8748efc624b29138cf`
- FlashInfer overlay: local working tree mixed-KV ABI patch
- Container image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- Memory cap: `--memory=16g --memory-swap=16g`

## Command

```bash
ssh jethac@100.113.98.11 \
  "cd /home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367 && \
   bash scripts/run_flashinfer_mixed_kv_jit_smoke_container.sh \
     /home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367 \
     /home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367/third_party/flashinfer"
```

## Result

```json
{
  "jit_additional_tensor_names": [
    "maybe_custom_mask",
    "maybe_mask_indptr",
    "maybe_alibi_slopes",
    "maybe_prefix_len_ptr",
    "maybe_token_pos_in_items_ptr",
    "maybe_max_item_len_ptr",
    "maybe_k_cache_sf",
    "maybe_v_cache_sf"
  ],
  "jit_module_type": "SimpleNamespace",
  "ok": true
}
```

## Finding

The FlashInfer prefill wrapper now accepts split K/V dtype inputs and the mixed-KV JIT path includes FP4 scale-factor tensors in the generated module interface.

This clears the first ABI gate for FP8-K + NVFP4-V. Next gates are:

1. Compile/run a standalone paged-prefill numerical harness with FP8 K and NVFP4 V.
2. Wire SGLang to store/pass FP8 K plus NVFP4 V and V scale buffers.
3. Re-run the default radix-cache row and confirm the attention merge recovers from the prior layer-0 collapse.
4. Measure the honest mixed-capacity claim separately from full NVFP4 K+V.
