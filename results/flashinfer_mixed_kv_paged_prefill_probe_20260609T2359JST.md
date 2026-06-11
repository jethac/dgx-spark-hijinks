# FlashInfer mixed-KV paged-prefill probe - 2026-06-09 23:59 JST

## Scope

Goal: validate that the new FlashInfer mixed-KV ABI can run a standalone paged-prefill attention call with:

- K cache: FP8 e4m3
- V cache: packed NVFP4 plus FP8 V scale factors
- Scale-factor tuple: `(None, v_sf)`

This is still not a SGLang radix-cache result. It proves the standalone FlashInfer prefill path is viable enough to start SGLang pool/backend wiring.

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
     /home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367/third_party/flashinfer \
     jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass \
     scripts/flashinfer_mixed_kv_paged_prefill_probe.py"
```

## Result

```json
{
  "cached_k_dtype": "torch.float8_e4m3fn",
  "cached_v_dtype": "torch.uint8",
  "cosine_vs_fp8k_nvfp4v_reference": 0.9815638661384583,
  "lse_shape": [
    64,
    4
  ],
  "max_abs_diff_vs_reference": 1.90625,
  "ok": true,
  "out_shape": [
    64,
    4,
    128
  ]
}
```

## Finding

The standalone FA2 paged-prefill path accepts and runs mixed KV storage:

- `k_data_type=torch.float8_e4m3fn`
- `v_data_type=torch.uint8`
- `kv_cache_sf=(None, v_sf)`

This supports the SGLang repair direction: protect QK/LSE with FP8 K while keeping V packed NVFP4 for an expected mixed-capacity claim around 1.28x versus fp8 KV, not the full 1.78x NVFP4 K+V claim.

## Next Gate

Wire SGLang to expose a distinct mixed-KV mode/pool:

1. Store K as FP8, not packed NVFP4.
2. Store V as packed NVFP4 plus V scale buffers.
3. Pass `k_data_type=torch.float8_e4m3fn`, `v_data_type=torch.uint8`, and `kv_cache_sf=(None, v_sf)` into FlashInfer prefill/decode.
4. Re-run the default radix-cache row; green requires cache ON, fp8 first-token match, and recovery from the prior layer-0 attention collapse.
