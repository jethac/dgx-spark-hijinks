# SGLang Gemma 3 27B mixed-KV CUDA graph gate retry 3

Date: 2026-06-11 JST

Scope: SGLang lane, Gemma 3 27B text-only, hybrid SWA pool, mixed FP8-K + NVFP4-V graph re-enable gate. This retry used the FlashInfer source-path `sitecustomize` shim so generated modules resolve templates and headers from the source overlay rather than the installed wheel.

## Run

Run id: `sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry3_sitecustomize_20260611T094007JST`

Launch settings:

- Model: `google/gemma-3-27b-it`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- SGLang source overlay: `jethac/sglang@d048bfedb`
- FlashInfer source overlay: `jethac/flashinfer@fb7d62ea`
- `CTX_LIST=8192`
- `REUSE_PREFIX_LEN=4096`
- `LOGPROB_START_LEN=4096`
- `MEM_FRACTION_STATIC=0.60`
- `PAGE_SIZE=1`
- `DISABLE_GRAPHS=0`
- `ENABLE_FP4_CUDA_GRAPH=1`
- `EXTRA_SERVER_ENVS=SGLANG_GEMMA3_ENABLE_HYBRID_SWA=1 SPARK_FLASHINFER_SOURCE_ROOT=/work/third_party/flashinfer SPARK_FLASHINFER_SITECUSTOMIZE_DEBUG=1 FLASHINFER_PREFILL_DEBUG_ONCE=1`
- Docker memory cgroup: `100g`

Artifacts:

- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry3_sitecustomize_20260611T094007JST_launch_env.txt`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry3_sitecustomize_20260611T094007JST_fp8_install.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry3_sitecustomize_20260611T094007JST_fp8_server.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry3_sitecustomize_20260611T094007JST_fp8_ppl.json`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry3_sitecustomize_20260611T094007JST_mixed_install.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry3_sitecustomize_20260611T094007JST_mixed_server.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_retry3_sitecustomize_20260611T094007JST_runner.log`

## What Passed

The source-path shim worked in both fp8 and mixed containers:

- `flashinfer` resolved to `/work/third_party/flashinfer/flashinfer/__init__.py`.
- FlashInfer JIT paths resolved to the source overlay.
- The previous missing Jinja/header packaging failures did not recur.

The fp8 comparator completed with CUDA graphs enabled:

- `ok: true`
- `ctx: 8192`
- `reuse_prefix_len: 4096`
- `cached_tokens: 4096`
- `mean_nll_nats: 1.430433723561978`
- `ppl: 4.180511985318388`

## Failure

The mixed FP8-K + NVFP4-V server loaded the model and allocated the hybrid-SWA mixed pool, then failed during CUDA graph capture:

```text
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
NVFP4 KV cache calibration warmup failed; first real prefill will remain the fallback for auto-calibration.
Error: BatchPrefillWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'k_data_type'
Capture cuda graph begin.
TypeError: BatchDecodeWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'k_data_type'
```

The traceback lands in the sliding-window graph path:

```text
flashinfer_backend.py:update_sliding_window
  -> call_begin_forward
  -> wrapper.begin_forward
  -> BatchDecodeWithPagedKVCacheWrapper.plan(k_data_type=...)
```

## Status

Red. The retry proves the packaging/source-overlay issue is closed and narrows the remaining graph-gate defect to SGLang's mixed-KV sliding-window decode graph metadata path. The `d048bfedb` split-dtype guard was sufficient for the fp8 comparator but not for the mixed CUDA graph capture path.

Next SGLang code action: gate `k_data_type`/`v_data_type` out of the FlashInfer decode wrapper plan when the installed FlashInfer decode ABI only accepts `kv_data_type`, including the sliding-window graph metadata path. Keep this scoped to the FP4-KV FA2/sm_12x branch.
