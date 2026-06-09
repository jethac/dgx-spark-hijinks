# vLLM Gemma 3 FlashInfer Worker Binding Probe - 2026-06-09T1552JST

## Scope

Follow-up to `results/vllm_gemma3_flashinfer_binding_fix_20260609T1518JST_summary.md`.
This run tested whether the EngineCore worker-side FlashInfer paged-prefill module binding
could be forced onto the FP4-KV specialization rather than the raw `uint8_t` KV variant.

## Environment

- Host: `thinkstationpgx-00b4` via Tailnet `100.113.98.11`
- Run checkout: `/home/jethac/spark_tmp/vllm_gemma3_prefill_debug_20260609T143439JST`
- vLLM fork: `jethac/vllm@1fabc6649`
- FlashInfer fork: `jethac/flashinfer@96be2fa8`
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- Run id: `vllm_gemma3_flashinfer_worker_bind_20260609T1552JST`
- Model: `google/gemma-3-27b-it`
- Reduced safety envelope: `MAX_MODEL_LEN=4096`, `MAX_NUM_BATCHED_TOKENS=1024`,
  `--gpu-memory-utilization 0.72`, Docker memory/swap cap `100g`.

## Patch Under Test

The source-overlay `sitecustomize.py` now rebinds FlashInfer's standard batch-prefill
generation path, not only the customized path:

- `flashinfer.jit.gen_batch_prefill_module`
- `flashinfer.prefill.gen_batch_prefill_module`
- `flashinfer.jit.gen_customize_batch_prefill_module`
- `flashinfer.prefill.gen_customize_batch_prefill_module`

The runner also clears all FlashInfer generated/cached batch-prefill modules across installed
FlashInfer versions and CUDA arch cache directories.

## Result

The pre-serve import probe was green:

- `flashinfer_dtype_map_kv_uint8="__nv_fp4x2_e2m1"`
- `flashinfer_attention_dtype_map_kv_uint8="__nv_fp4x2_e2m1"`
- `flashinfer_filename_safe_dtype_map_kv_uint8="fp4x2_e2m1"`
- `flashinfer_attention_gen_batch_prefill_module="flashinfer.jit.attention.modules"`
- `flashinfer_jit_gen_batch_prefill_module="flashinfer.jit.attention.modules"`

The live server did not reach readiness. It failed during FlashInfer attention warmup with:

```text
TypeError: Mismatched number of arguments when calling: `paged_run(...) -> void`.
Expected 29 but got 27 arguments
```

This is progress. The generated FlashInfer paged-prefill module now has the FP4-KV-aware
signature that expects the scale-factor tensor arguments, while vLLM still calls it with the
old 27-argument raw-KV call shape. The previous failure was silent use of a raw-byte prefill
module; this run changes the failure mode to an explicit missing-argument crash before any
quality sample is produced.

## Interpretation

The current vLLM Gemma 3 blocker is no longer "does FlashInfer generate an FP4-KV paged
prefill module?" The next blocker is vLLM prefill argument plumbing:

- pass `maybe_k_cache_sf` and `maybe_v_cache_sf` into the paged-prefill call;
- keep the scale tensors matched to the split NVFP4 KV page views from
  `nvfp4_kv_cache_split_views`;
- re-run the same reduced probe and require both startup and the Gemma quality gate.

Do not return to byte/page-pairing or low-level FA2 math until the vLLM call site supplies
the FP4 scale tensors and the server reaches first-token generation.

## Artifacts

- `results/vllm_gemma3_flashinfer_worker_bind_20260609T1552JST_import_probe.txt`
- `results/vllm_gemma3_flashinfer_worker_bind_20260609T1552JST_server.log`
- `results/vllm_gemma3_flashinfer_worker_bind_20260609T1552JST_flashinfer_prefill_debug_audit.json`

Host state after teardown: no matching container remained and `free -h` reported about
`115 GiB` available.
