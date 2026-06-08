# vLLM Gemma 3 Wrapper-Boundary Trace Packet, 2026-06-09

Purpose: localize the Gemma 3 27B NVFP4-KV first-token failure at the actual
vLLM/FlashInfer wrapper boundary after the standalone Gemma-shaped FA2 probe passed.

## Patch

vLLM fork branch: `spark/hijinks-021-gemma3-tensor-trace`

Commit:

```text
0e07e130d94eddfed209f846ce6c9959c636da02 Trace Gemma FlashInfer wrapper boundary
13da71884640567682cd3ddd4650d2ba3ecb5543 Dump Gemma active FlashInfer prefill pages
```

The patch adds inactive-by-default wrapper events in
`vllm/v1/attention/backends/flashinfer.py`:

- `flashinfer_wrapper_prefill_pre`
- `flashinfer_wrapper_prefill_post`
- `flashinfer_wrapper_decode_pre`
- `flashinfer_wrapper_decode_post`

The events are gated by the existing `VLLM_SPARK_GEMMA_TENSOR_TRACE` environment and
record wrapper type, local/global window state, token counts, scalar K/V scales, compact
query/output summaries, and compact KV argument metadata.

The follow-up commit adds an inactive-by-default active-page dump:

```text
VLLM_SPARK_ACTIVE_PAGE_DUMP=1
VLLM_SPARK_ACTIVE_PAGE_DUMP_DIR=/results/active_page_dump
VLLM_SPARK_ACTIVE_PAGE_DUMP_LIMIT=1
VLLM_SPARK_ACTIVE_PAGE_DUMP_PAGES=4
```

It writes small `torch.save` payloads containing `query`, `out_before`, `out_after`,
the wrapper's active `paged_kv_indptr` / `paged_kv_indices` /
`paged_kv_last_page_len`, and the selected active K/V data and FP8 scale pages. This is
for replay/debug only and must not be enabled in benchmark rows.

## Run

Use the no-downgrade source-overlay pattern:

```text
model=google/gemma-3-27b-it
image=jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass
vllm_commit=0e07e130d94eddfed209f846ce6c9959c636da02
flashinfer_source_overlay=enabled
TORCH_CUDA_ARCH_LIST=12.1a
FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1
VLLM_SPARK_KV_TRACE=1
VLLM_SPARK_GEMMA_TENSOR_TRACE=1
VLLM_SPARK_GEMMA_TENSOR_TRACE_LAYERS=layers.0.self_attn.attn,layers.5.self_attn.attn,lm_head
VLLM_SPARK_ACTIVE_PAGE_DUMP=1
VLLM_SPARK_ACTIVE_PAGE_DUMP_DIR=/results/active_page_dump
VLLM_SPARK_ACTIVE_PAGE_DUMP_LIMIT=1
VLLM_SPARK_ACTIVE_PAGE_DUMP_PAGES=4
```

Serve with:

```text
--attention-backend flashinfer
--kv-cache-dtype nvfp4
--dtype bfloat16
--max-model-len 131072
--gpu-memory-utilization 0.85
--max-num-batched-tokens 4096
--enforce-eager
```

`--enforce-eager` is diagnostic-only. Do not use this row as a speed benchmark.

## Result

Artifact summary:

```text
results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_summary.md
results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_summary.md
results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_summary.md
results/vllm_gemma3_27b_flashinfer_paged_prefill_audit_20260609.md
```

Key result: the real FlashInfer FA2 paged prefill wrapper is the failing boundary. For a
24-token full/global layer call, sane signed BF16 `query` and `out_before` become
byte-like BF16 `out_after` immediately after `BatchPrefillWithPagedKVCacheWrapper.run`.
The corrupted output head aligns with packed active-page `v_data_head`, making the next
task an active-page replay/dequantized-reference comparison.

The active-page dump row confirms this against the exact selected pages. The two useful
request payloads have byte-like `out_after` tensors with max `255.0`, means around
`128..129`, and first 16 `out_after` BF16 values exactly matching the first 16 active
packed V data bytes. This moves the next task from "dump active pages" to "replay the
dumped paged-prefill call against a dequantized reference."

The replay row completes that comparison. Dequantized CPU causal attention over the exact
active pages produces sane signed output (mean near zero, RMS around `1.9..2.0`), while
the real wrapper output remains byte-like and has near-zero cosine against the reference
under all tested variants.

## Next Task

Audit and fix FlashInfer's paged-prefill NVFP4 specialization. The current evidence says
vLLM is passing paired active pages and scales, but FlashInfer returns packed V carrier
bytes as BF16-like output. Prime checks:

- JIT/AOT cache-key collision: generated C++ maps `torch.uint8` to `__nv_fp4x2_e2m1`, but
  the batch-prefill URI still names the module `dtype_kv_u8`; force a fresh NVFP4-specific
  JIT namespace and rerun this packet before editing kernel math;
- V element type / container type binding in generated paged prefill parameters;
- V data pointer versus V scale pointer ordering;
- template path selected when `kv_cache` is a tuple of packed `uint8` K/V tensors and
  `kv_cache_sf` is a tuple of FP8 scale tensors;
- divergence between decode, standalone prefill probe, and real paged prefill wrapper.
