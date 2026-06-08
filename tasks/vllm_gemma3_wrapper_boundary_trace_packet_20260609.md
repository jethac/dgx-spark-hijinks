# vLLM Gemma 3 Wrapper-Boundary Trace Packet, 2026-06-09

Purpose: localize the Gemma 3 27B NVFP4-KV first-token failure at the actual
vLLM/FlashInfer wrapper boundary after the standalone Gemma-shaped FA2 probe passed.

## Patch

vLLM fork branch: `spark/hijinks-021-gemma3-tensor-trace`

Commit:

```text
0e07e130d94eddfed209f846ce6c9959c636da02 Trace Gemma FlashInfer wrapper boundary
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
```

Key result: the real FlashInfer FA2 paged prefill wrapper is the failing boundary. For a
24-token full/global layer call, sane signed BF16 `query` and `out_before` become
byte-like BF16 `out_after` immediately after `BatchPrefillWithPagedKVCacheWrapper.run`.
The corrupted output head aligns with packed active-page `v_data_head`, making the next
task an active-page replay/dequantized-reference comparison.

## Next Task

Dump or reconstruct the exact active block-table paged prefill call for the failing
layer and compare FlashInfer wrapper output with a dequantized reference. The goal is to
prove whether the paged prefill specialization is reading packed uint8 V data as values,
using the wrong V view, or receiving wrong wrapper metadata from vLLM.
