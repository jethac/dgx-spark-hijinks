# SGLang Gemma 3 27B mixed-KV CUDA graph gate

Date: 2026-06-11 JST

Scope: SGLang lane, Gemma 3 27B text-only, hybrid SWA pool, mixed FP8-K + NVFP4-V graph re-enable gate. This row is red before the mixed comparator because the fp8 comparator failed during CUDA graph capture.

## Run

Run id: `sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_20260611T071237JST`

Launch settings:

- Model: `google/gemma-3-27b-it`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- SGLang source overlay at launch: `cf7414f80`
- FlashInfer source overlay: `fb7d62ea`
- `CTX_LIST=8192`
- `REUSE_PREFIX_LEN=4096`
- `LOGPROB_START_LEN=4096`
- `MEM_FRACTION_STATIC=0.60`
- `PAGE_SIZE=1`
- `DISABLE_GRAPHS=0`
- `ENABLE_FP4_CUDA_GRAPH=1`
- `EXTRA_SERVER_ENVS=SGLANG_GEMMA3_ENABLE_HYBRID_SWA=1`
- Docker memory cgroup: `100g`

Artifacts:

- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_20260611T071237JST_launch_env.txt`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_20260611T071237JST_fp8_install.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_20260611T071237JST_fp8_server.log`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_20260611T071237JST_fp8_container_inspect.json`
- `results/sglang_gemma3_27b_mixedkv_graph_gate_ctx8192_prefix4096_20260611T071237JST_runner.log`

## What Passed

The fp8 comparator reached the graph-capture gate:

- Gemma 3 loaded from local HF cache.
- Weight load completed in `359.55 s`.
- Hybrid SWA memory pool was selected.
- fp8 KV pools allocated:
  - SWA pool: `58118` tokens, K `5.76 GB`, V `5.76 GB`
  - Full pool: `72648` tokens, K `1.39 GB`, V `1.39 GB`
- CUDA graph capture began with batch sizes `[1, 2, 4, ..., 256]`.

## Failure

The scheduler crashed at the first graph-capture batch:

```text
TypeError: BatchDecodeWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'k_data_type'
```

Root cause: SGLang's graph metadata path passed split K/V dtype kwargs (`k_data_type`, `v_data_type`) unconditionally into FlashInfer decode `begin_forward()`. The FlashInfer branch in this stack exposes the single-KV API (`kv_data_type`) for decode planning, so even the fp8 comparator crashed before readiness.

## Follow-Up Patch

SGLang fork patch pushed after this red row:

- `jethac/sglang@d048bfedb` (`spark/hijinks-018-fp4-e2m1-kv-sm121-serving`)
- Change: gate split K/V dtype kwargs so fp8 and same-dtype modes use the upstream-compatible single `kv_data_type` path.
- Local verification: `python -m py_compile python/sglang/srt/layers/attention/flashinfer_backend.py`

The retry was not started because `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` appeared before launch; the box was left idle with no containers running.

## Status

Red, but narrowed. The previous red was a non-mixed graph-planning compatibility bug, not a Gemma quality failure. The next GPU step is to rerun the same gate on `jethac/sglang@d048bfedb`. Expected outcomes:

- fp8 graph capture should pass the previous `k_data_type` crash.
- mixed FP8-K + NVFP4-V may still require true split-dtype FlashInfer decode/prefill ABI if it reaches a mixed graph-capture or quality failure.
