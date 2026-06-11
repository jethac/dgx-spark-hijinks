# SGLang mixed-KV pool probe, 2026-06-10 00:36 JST

## Scope

This is a small GB10/Spark-class probe, not a model-serving row. It validates the proposed
SGLang radix-cache repair storage mode:

- K cache: FP8 e4m3
- V cache: packed NVFP4 with FP8 scale factors
- page size: 1, matching the SGLang FP4-KV serving configuration
- FlashInfer paged prefill: split K/V dtype ABI

The serving gate remains unrun at this stop point.

## Important macro finding

SGLang must use FlashInfer **without** vLLM's V-scale-factor deswizzle macro.

The first probe runner incorrectly compiled with:

```text
-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1
```

That is a vLLM-specific layout fix. It is wrong for SGLang's current linear V-scale
layout and makes page-size-1 NVFP4-V numerically bad.

## Red diagnostic with wrong macro

Standalone FlashInfer page-size-1 mixed-KV result with vLLM deswizzle enabled:

```json
{
  "cached_k_dtype": "torch.float8_e4m3fn",
  "cached_v_dtype": "torch.uint8",
  "cosine_vs_fp8k_nvfp4v_reference": 0.8967183828353882,
  "max_abs_diff_vs_reference": 11.14453125,
  "ok": true,
  "page_size": 1
}
```

Three-case isolation with the wrong macro:

```json
{
  "cases": [
    {"case": "fp8_k_bf16_v", "cosine": 0.9999990463256836, "max_abs_diff": 0.00390625, "ok": true},
    {"case": "bf16_k_nvfp4_v", "cosine": 0.8800402283668518, "max_abs_diff": 12.21875, "ok": true},
    {"case": "fp8_k_nvfp4_v", "cosine": 0.8793418407440186, "max_abs_diff": 12.5546875, "ok": true}
  ],
  "page_size": 1
}
```

Interpretation: FP8 K was clean. The error followed NVFP4 V because the runner used the
wrong V-scale layout policy.

## Green diagnostic with correct SGLang macro policy

Runner policy changed to default:

```text
FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a"
```

Three-case page-size-1 isolation then passed:

```json
{
  "cases": [
    {"case": "fp8_k_bf16_v", "cosine": 0.9999990463256836, "max_abs_diff": 0.00390625, "ok": true},
    {"case": "bf16_k_nvfp4_v", "cosine": 1.0, "max_abs_diff": 0.0, "ok": true},
    {"case": "fp8_k_nvfp4_v", "cosine": 1.0000001192092896, "max_abs_diff": 0.0, "ok": true}
  ],
  "page_size": 1
}
```

SGLang pool integration probe then passed:

```json
{
  "cosine_vs_pool_reference": 0.999995231628418,
  "k_buffer_dtype": "torch.float8_e4m3fn",
  "k_global": 1.0,
  "k_scale_is_none": true,
  "k_size_bytes": 18688,
  "lse_shape": [16, 4],
  "max_abs_diff_vs_pool_reference": 0.0078125,
  "mixed_fp8_k_nvfp4_v": true,
  "ok": true,
  "out_shape": [16, 4, 128],
  "v_buffer_dtype": "torch.uint8",
  "v_global": 0.0016159784281626344,
  "v_size_bytes": 10516
}
```

## Code touched

- `third_party/flashinfer`: split K/V dtype support in paged prefill/decode surfaces.
- `third_party/sglang`: experimental `SGLANG_FP4_KV_MIXED_KV=1` mode in the FP4 KV pool and FlashInfer backend.
- `scripts/run_sglang_fp4_dense_cache_trace.sh`: `MIXED_KV=1` opt-in for the live default-radix runner.
- `scripts/run_sglang_mixed_kv_pool_probe_container.sh`: small 16 GiB cgroup runner for pool and FlashInfer probes.
- `scripts/flashinfer_mixed_kv_page1_isolation_probe.py`: page-size-1 isolation harness.
- `scripts/sglang_mixed_kv_pool_probe.py`: SGLang pool-to-FlashInfer integration harness.

## Current stop point

Green: FlashInfer split-K/V ABI and SGLang pool plumbing for FP8-K plus NVFP4-V at page size 1.

Unproven: SGLang default radix serving quality with `MIXED_KV=1`.

Next command shape:

```bash
MIXED_KV=1 \
PREPARE_RUST_IMAGE=0 \
INSTALL_SOURCE_STACK_PER_CASE=0 \
RUNTIME_IMAGE=sglang-source-stack-c3dae30f-e631a13fd \
CASES=default \
RUN_ID=sglang_qwen_mixedkv_default_YYYYMMDDTHHMMSS \
bash scripts/run_sglang_fp4_dense_cache_trace.sh
```

Use the GB10 memory rules: single server, conservative memory fraction, Docker `--memory=100g --memory-swap=100g`, and no concurrent fp8/fp4 comparators.
