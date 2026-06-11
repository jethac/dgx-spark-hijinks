# SGLang overnight ladder diagnosis

Run: `sglang_overnight_ladder_20260612T015035JST`
Date: 2026-06-12 JST
Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`

## Summary

Rows attempted: 12
Green rows: 2
Red rows: 10

The only green rows are:

- `e2b_bf16`: C1 mean NLL `3.923662672115819`, deterministic C1 double-run.
- `e2b_nvfp4`: C1 mean NLL `3.9244879447009984`, deterministic C1 double-run.

Delta for E2B NVFP4 vs bf16 on C1 is about `+0.0008252725851794` nats/token, well inside the `0.5` nats/token red threshold.

## Harness caveat for this run

The `smoke_tokyo.json` transcripts for the two green rows are coherent and contain Tokyo, but `tokyo_smoke_rc=2` because `scripts/openai_chat_smoke.py` hard-coded `spark-ok` as its only success predicate. The ladder manifest intentionally used the separate `tokyo_contains_tokyo` predicate for this run.

This harness bug is fixed after the run by adding `--expect-substring` to `scripts/openai_chat_smoke.py` and using `--expect-substring Tokyo` from `scripts/run_sglang_overnight_ladder.sh`; future green rows must have `tokyo_smoke_rc == 0`.

## Red row classes

### `e2b_fp8`

The server became ready, but the first smoke crashed the scheduler with a FlashInfer paged-prefill invalid configuration:

```text
RuntimeError: Error in function 'BatchPrefillWithPagedKVCacheDispatched' at /flashinfer-src/include/flashinfer/attention/prefill.cuh:3215: FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

Artifact: `rows/e2b_fp8/server.log`.

### `12b_*`

All 12B rows fail before readiness because the image's Transformers stack does not recognize `model_type=gemma4_unified`:

```text
ValueError: The checkpoint you are trying to load has model type `gemma4_unified` but Transformers does not recognize this architecture.
```

Artifact: `rows/12b_bf16/server.log` and matching `12b_nvfp4` / `12b_fp8` logs.

### `26b_a4b_*`

All 26B-A4B rows load weights and measure live geometry, but pool sizing computes negative token counts and raises:

```text
SGLANG_GEMMA_KV_POOL_CONFIG full_layers=5 swa_layers=25 full_per_token_bytes=4096 swa_per_token_bytes=8192 swa_full_tokens_ratio=0.8 cell_size_bytes=184320.0 mixed_kv=False
Use sliding window memory pool. full_layer_tokens=-40934, swa_layer_tokens=-32747
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

Artifact: `rows/26b_a4b_bf16/server.log`; fp8/NVFP4 hit the same readiness class.

### `31b_*`

All 31B rows load weights and measure live geometry, but pool sizing similarly computes negative token counts and raises:

```text
SGLANG_GEMMA_KV_POOL_CONFIG full_layers=10 swa_layers=50 full_per_token_bytes=8192 swa_per_token_bytes=16384 swa_full_tokens_ratio=0.8 cell_size_bytes=737280.0 mixed_kv=False
Use sliding window memory pool. full_layer_tokens=-29537, swa_layer_tokens=-23629
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

Artifact: `rows/31b_bf16/server.log`; fp8/NVFP4 hit the same readiness class.

## Invalid earlier attempts preserved

Two earlier run directories are preserved as invalid attempts:

- `sglang_overnight_ladder_20260612T002050JST`: corpus preflight was RED because C1 came from a stale local copy. The runner now defaults to `/home/jethac/spark_tmp/claude_overnight_ladder_20260612/docs` and hard-stops before model rows on any corpus MD5 mismatch.
- `sglang_overnight_ladder_20260612T014220JST`: runner manifest generation crashed with Python `NameError: name 'true' is not defined` because shell booleans were inserted into Python. The runner now maps shell booleans to Python `True` / `False`.

Neither invalid attempt is claim-grade.
