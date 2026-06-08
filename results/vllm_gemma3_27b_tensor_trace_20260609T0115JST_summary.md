# vLLM Gemma 3 27B Tensor Trace, 2026-06-09

Status: diagnostic row complete; NVFP4-KV quality still fails. This row is not
throughput evidence because it used `--enforce-eager` to collect Python tensor
summaries.

## Source Pins

- Parent repo: `0a58a126d3777f46212229070fc9b276f9e302ce` at remote checkout
  creation.
- vLLM fork: `jethac/vllm@5b67b0ea213a5067e7e8e9fb5705b005f6c495f5`
  (`spark/hijinks-021-gemma3-tensor-trace`).
- FlashInfer fork: `jethac/flashinfer@e41016fcd121986aea923d5c7e68fc9f152d2a07`.
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`.
- Model: `google/gemma-3-27b-it`, served as `gemma3-27b-it`.

## Compile-Mode Caveat

The first normal-compile attempt on the earlier tensor-trace commit failed during
vLLM model profiling because TorchDynamo captured the Python tensor-summary code:

```text
torch._dynamo.exc.UserError: Could not guard on data-dependent expression Eq(u1, 1)
```

`jethac/vllm@5b67b0ea2` adds a compile guard so the trace hook is inactive while
TorchDynamo is compiling. The live tensor-summary rows intentionally used
`--enforce-eager`, so they are useful for localization only.

## First-Token Comparator

Baseline: `vllm_gemma3_27b_tensor_trace_20260609T0115JST_fp8_flashinfer_eager`.

Candidate:
`vllm_gemma3_27b_tensor_trace_20260609T0115JST_nvfp4_kv_flashinfer_eager`.

| case | fp8 first token | NVFP4-KV first token | top-logprob overlap |
|---|---:|---:|---:|
| `exact_spark_ok` | `spark` | ` Reigns` | `0.0` |
| `simple_math` | `4` | Gujarati text | `0.0` |
| `short_decode` | `A` | `ioane` | `0.0` |

Result: `0 / 3` first-token probes match. The corruption is present before
sampling and before any long decode compounding.

## Tensor Compare

Comparison artifact:
`results/vllm_gemma3_27b_tensor_trace_20260609T0115JST_compare.json`.

The compare script matched `561` event/layer keys with no event/layer present only
on one side. It uses the last record per event/layer; no benchmark traffic ran after
the first-token probes.

The strongest localization signal is `flashinfer_attn_output`. The NVFP4-KV
candidate returns BF16-shaped attention outputs, but many layer outputs become
almost entirely nonnegative, with means around `124..126` and max values exactly
`255.0`.

| event/layer/tensor | fp8 RMS | NVFP4 RMS | RMS ratio | fp8 min | NVFP4 min | fp8 max abs | NVFP4 max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flashinfer_attn_output` layer 0 `output_last` | `40.3003` | `143.8351` | `3.5691` | `-340.0` | `3.8e-33` | `346.0` | `255.0` |
| `flashinfer_attn_output` layer 1 `output_last` | `5.9186` | `145.7251` | `24.6217` | `-81.5` | `0.0` | `81.5` | `255.0` |
| `flashinfer_attn_output` layer 10 `output_last` | `1.9769` | `146.1581` | `73.9338` | `-12.875` | `1.2e-17` | `23.875` | `255.0` |
| `flashinfer_attn_output` layer 11 `output_last` | `1.0601` | `147.2847` | `138.9304` | `-6.375` | `6.7e-22` | `12.9375` | `255.0` |
| `gemma3_logits_input` `hidden_last` | `5.4301` | `5.4294` | `0.9999` | `-90.0` | `-20.625` | `90.0` | `36.0` |
| `gemma3_logits` `logits_last` | `7.9901` | `5.5064` | `0.6892` | `-33.5` | `-25.625` | `75.5` | `25.625` |

The final hidden-state RMS alone is not a sufficient quality signal: it is nearly
identical at `gemma3_logits_input`, while the logits top-20 sets are disjoint
(`0.0` overlap). The earlier attention-output saturation remains the actionable
localization signal.

## Conclusion

This row clears the page/scale byte-pairing layer as the primary sampled failure
surface and points the next vLLM work at the FlashInfer FA2 NVFP4 attention output
path: output scaling, dequantization, V-scale deswizzle, or output buffer
interpretation. The pattern is too structured for ordinary quantization drift.

Next target: a focused FlashInfer/vLLM probe that compares FA2 NVFP4 attention
output against a dequantized reference for the Gemma 3 `D=128` local/global shapes,
including output dtype, scale selection, and the V-scale deswizzle path.
