# vLLM Gemma 3 Prefill Call-Site No-Force Probe, 2026-06-09 16:22 JST

Purpose: test Claude's proposed fix shape without externally forcing FlashInfer's
batch-prefill generator. The diagnostic worker-bind run proved the FP4-aware prefill
module exists, but it failed warmup because the generated module expected 29 arguments
while vLLM supplied the old 27-argument raw-KV call shape. This run leaves generator
forcing off and instead tests the vLLM/FlashInfer scale-argument plumbing.

## Setup

- Host: GB10 over tailnet, single server only.
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- Model: `google/gemma-3-27b-it`
- KV dtype: `nvfp4`
- Memory guardrails: `--gpu-memory-utilization 0.72`, Docker `100g` cgroup cap,
  `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=1024`.
- vLLM branch: `jethac/vllm@spark/hijinks-021-gemma3-tensor-trace`
- FlashInfer branch: `jethac/flashinfer@spark/hijinks-021-prefill-debug`

Import probe:

```json
{
  "flashinfer": "0.6.9rc1",
  "device": "NVIDIA GB10",
  "capability": [12, 1],
  "spark_flashinfer_force_prefill_module": "0",
  "spark_flashinfer_patch_prefill_run_scale_args": "1",
  "flashinfer_prefill_run_marker": true,
  "flashinfer_dtype_map_kv_uint8": "__nv_fp4x2_e2m1",
  "flashinfer_filename_safe_dtype_map_kv_uint8": "fp4x2_e2m1"
}
```

## Patch Under Test

- vLLM prefill call-site metadata now appends `maybe_k_cache_sf` and
  `maybe_v_cache_sf` to the wrapper's JIT tensor-name list for SM12x NVFP4 KV.
- FlashInfer `BatchPrefillWithPagedKVCacheWrapper.run(...)` now includes
  `key_block_scales` and `value_block_scales` in the JIT optional-tensor dictionary.
- The source-overlay `sitecustomize.py` keeps the dtype-table patch and the runtime
  prefill-run scale-argument patch, but leaves external generator/module forcing off:
  `SPARK_FLASHINFER_FORCE_PREFILL_MODULE=0`.

## Result

The server reached readiness and completed the first-token probe. The earlier
`Expected 29 but got 27 arguments` warmup mismatch did not occur.

Quality is still red:

| Case | First token |
| --- | --- |
| `exact_spark_ok` | ` Reigns` |
| `simple_math` | non-Latin Gujarati token |
| `short_decode` | `ioane` |

The FlashInfer prefill debug lines still show the natural generated module is the
raw-byte variant for both SWA and global prefill calls:

- SWA: `window_left=1023`, `dtype_kv=uint8_t`, `fp4_kv=0`,
  `require_fp4_kv=0`, `is_kv_fp4x2=0`, `additional_tensors=`.
- Global: `window_left=-1`, `dtype_kv=uint8_t`, `fp4_kv=0`,
  `require_fp4_kv=0`, `is_kv_fp4x2=0`, `additional_tensors=`.
- No live `maybe_k_cache_sf` or `maybe_v_cache_sf` tensor views appear in the paged
  prefill debug lines.

## Interpretation

This falsifies the narrow version of "pass the scale tensors and FlashInfer will
naturally select the FP4 prefill module" for this installed FlashInfer/NVIDIA image stack.
The vLLM call-site and wrapper scale-argument plumbing are still useful and required once
the FP4 module is selected, but they are not sufficient by themselves.

The remaining vLLM Gemma 3 NVFP4-KV bug is now in FlashInfer prefill module
selection/metadata/generator plumbing: without external forcing, the live paged prefill
wrapper still compiles and runs a raw `uint8_t` module against packed NVFP4 KV bytes. The
next fix should make FlashInfer select the FP4-KV paged-prefill module naturally and pass
the scale tensors through the real call path, then rerun `scripts/gemma_nvfp4_kv_quality_gate.py`.

## Artifacts

- `results/vllm_gemma3_prefill_callsite_noforce_20260609T1622JST_import_probe.txt`
- `results/vllm_gemma3_prefill_callsite_noforce_20260609T1622JST_server.log`
- `results/vllm_gemma3_prefill_callsite_noforce_20260609T1622JST_first_token.json`
- `results/vllm_gemma3_prefill_callsite_noforce_20260609T1622JST_tensor_trace.jsonl`
- `results/vllm_gemma3_prefill_callsite_noforce_20260609T1622JST_flashinfer_prefill_debug_audit.json`
