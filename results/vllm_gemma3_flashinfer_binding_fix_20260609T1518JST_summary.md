# vLLM Gemma 3 FlashInfer Binding Fix Probe - 2026-06-09T1518JST

## Run

- Host: `thinkstationpgx-00b4` over tailnet `100.113.98.11`.
- Temp checkout: `/home/jethac/spark_tmp/vllm_gemma3_prefill_debug_20260609T143439JST`.
- vLLM submodule: `jethac/vllm@1fabc6649`.
- FlashInfer source overlay: `jethac/flashinfer@96be2fa8`.
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`.
- Model: `google/gemma-3-27b-it`.
- Args: `--kv-cache-dtype nvfp4`, `--attention-backend flashinfer`, `--enforce-eager`,
  `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=1024`,
  `--gpu-memory-utilization 0.72`.
- Memory guardrails: single server only, Docker `--memory 100g --memory-swap 100g`.

## Artifacts

- Import probe: `results/vllm_gemma3_flashinfer_binding_fix_20260609T1518JST_import_probe.txt`
- Server log: `results/vllm_gemma3_flashinfer_binding_fix_20260609T1518JST_server.log`
- First-token probe: `results/vllm_gemma3_flashinfer_binding_fix_20260609T1518JST_first_token.json`
- Tensor trace: `results/vllm_gemma3_flashinfer_binding_fix_20260609T1518JST_tensor_trace.jsonl`
- FlashInfer audit: `results/vllm_gemma3_flashinfer_binding_fix_20260609T1518JST_flashinfer_prefill_debug_audit.json`

## Result

The source-overlay import probe is now green:

- `flashinfer_dtype_map_kv_uint8 = "__nv_fp4x2_e2m1"`
- `flashinfer_attention_dtype_map_kv_uint8 = "__nv_fp4x2_e2m1"`
- `flashinfer_filename_safe_dtype_map_kv_uint8 = "fp4x2_e2m1"`

The live serving result is still red. First-token corruption is unchanged from the previous
Gemma 3 row:

- `exact_spark_ok` -> `" Reigns"`
- `simple_math` -> Gujarati token `"\u0abe\u0ab0\u0ac0"`
- `short_decode` -> `"ioane"`

The runtime FlashInfer paged-prefill audit is also still red. The worker process continues
to execute raw-byte paged-prefill modules:

- `dtype_kv=uint8_t`
- `fp4_kv=0`
- `require_fp4_kv=0`
- no `maybe_k_cache_sf` / `maybe_v_cache_sf`
- `module_uri` empty

## Interpretation

The pre-serve Python import path can be patched to see packed `torch.uint8` KV as FP4, but
that patch is not reaching the actual EngineCore paged-prefill binding path used during
serving, or the worker is resolving a different cached/native module path after import.

Do not move this lane to FA2 math or SWA cache-reuse debugging yet. The immediate vLLM
problem remains worker/runtime FlashInfer binding: prove the EngineCore process uses the
same `dtype_map_kv`/generator as the import probe, then rerun until the server log shows
`require_fp4_kv=1`, `is_kv_fp4x2=1`, and scale tensors bound.

## Host Safety

The run stayed within the new GB10 memory rules. It used one server, cgroup caps, and the
host returned to about `115 GiB` available memory after container teardown.
