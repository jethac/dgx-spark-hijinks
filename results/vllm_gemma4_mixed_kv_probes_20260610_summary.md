# vLLM Gemma 4 Mixed-KV Probes #1/#2 — GB10 Results

Date: 2026-06-10 JST (04:37–04:39)
Lane: `spark/hijinks-022-gemma4-mixed-kv` (Claude worktree lane; Codex untouched)
Host: `thinkstationpgx-00b4`, `NVIDIA GB10`, capability `[12, 1]` — confirmed idle before
runs (no containers, GPU 0%, 115 GiB free). Probes ran in `--rm` containers with
`--memory 24g` caps on image `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
(torch 2.12.0.dev+cu130, flashinfer 0.6.9rc1). Both are weight-free.

## Probe #1 — selector (open question #1)
Artifact: `results/vllm_gemma4_mixed_kv_selector_probe_20260610T0437JST.json`

| case | head_size | kv_cache_dtype | resolved backend |
|---|---:|---|---|
| local_sliding_nvfp4_d256 | 256 | nvfp4 | `FlashInferBackend` |
| global_full_auto_d512 | 512 | auto | `FlashInferBackend` |

Both shapes resolve (`ok: true`), but **both to FlashInfer** (`distinct_backends: false`).

## Probe #2 — FA2 bf16 D=512 (open question #2)
Artifact: `results/flashinfer_fa2_bf16_d512_probe_20260610T0438JST.json`

`BatchPrefillWithPagedKVCacheWrapper` (fa2, NHD) at `head_dim=512`,
`kv_data_type=bf16`: **fails at run time** with the identical trait guard the FP4 path
hits:

```
prefill.cuh:2615: FlashInfer Internal Error: Invalid configuration :
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

## Conclusions (decisive for VLLM_GEMMA_RUNGS.md §2-M6 / §8)
1. **The D=512 FA2 wall is dtype-independent — confirmed on hardware.** `NUM_MMA_D_VO`
   is head-dim-driven; bf16 trips the same fragment-budget guard as FP4. Gemma 4 global
   layers cannot use FlashInfer FA2 for any KV dtype.
2. **Automatic per-layer fallback does NOT happen.** FlashInfer's
   `validate_configuration` accepts head-512 at selection time, so with the model-wide
   Triton force relaxed, global layers would select FlashInfer and crash at warmup —
   probe #1's premise ("falls back to a head-size-capable backend on its own") is
   falsified. The fix landed in `jethac/vllm@cbfe86bd5`: Gemma 4 explicitly pins
   TRITON_ATTN for global D>256 layers via the Attention `attn_backend` override,
   scoped to mixed-KV runs.
3. **Two-backend machinery is real** (separately verified by code inspection: the worker
   builds AttentionGroups keyed by per-layer backend, with per-group metadata builders;
   the TurboQuant skip-layers path is prior art). The live two-builder coexistence check
   now happens at the Rung 2/M1 bring-up rather than a separate probe.

Upstream note: the selector-vs-kernel disagreement (validate_configuration accepting a
head size the FA2 kernel rejects) is an upstream-worthy FlashInfer/vLLM bug independent
of this campaign.
