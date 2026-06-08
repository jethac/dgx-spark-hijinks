# vLLM Gemma 3 Tensor Trace Packet, 2026-06-09

Status: diagnostic packet complete. The first normal-compile attempt exposed a
TorchDynamo capture guard and produced no serving row; the accepted rows used
`--enforce-eager` and localized the NVFP4-KV failure to FlashInfer attention output.

Purpose: localize the Gemma 3 27B NVFP4-KV first-token quality failure above the
page/scale byte-pairing layer. The prior trace proved sampled read-side packed K/V and
FP8 scale bytes match write-side bytes (`195 / 195`) and the failing short prompts do not
skip SWA blocks. This packet asks whether the first visible divergence is in FlashInfer
attention output, Gemma layer hidden state, final hidden state, or logits.

## Source Pins

- `jethac/vllm@spark/hijinks-021-gemma3-tensor-trace`
- vLLM commit: `5b67b0ea213a5067e7e8e9fb5705b005f6c495f5`
- FlashInfer commit: keep the clean source-overlay pin already used for rung 1,
  `e41016fcd121986aea923d5c7e68fc9f152d2a07`.
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`

## Trace Env

Use this for both fp8 and NVFP4-KV rows:

```bash
VLLM_SPARK_GEMMA_TENSOR_TRACE=1
VLLM_SPARK_GEMMA_TENSOR_TRACE_LIMIT=512
VLLM_SPARK_GEMMA_TENSOR_TRACE_VALUES=8
VLLM_SPARK_GEMMA_TENSOR_TRACE_LAYERS=layers.0,layers.5,layers.6,model,lm_head
```

Layer intent:

- `layers.0`: local/SWA layer.
- `layers.5`: global/full layer.
- `layers.6`: next local/SWA layer after a global layer.
- `model`: final normalized hidden state.
- `lm_head`: logits input and top-20 logits.

## Server Rows

Reuse the source-overlay pattern from
`tasks/vllm_gemma3_nvfp4_trace_packet_20260608.md`, but mount the vLLM source at
`5b67b0ea213a5067e7e8e9fb5705b005f6c495f5` and add:

```bash
-e VLLM_SPARK_GEMMA_TENSOR_TRACE=1 \
-e VLLM_SPARK_GEMMA_TENSOR_TRACE_FILE=/results/${RUN}_tensor_trace.jsonl \
-e VLLM_SPARK_GEMMA_TENSOR_TRACE_LIMIT=512 \
-e VLLM_SPARK_GEMMA_TENSOR_TRACE_VALUES=8 \
-e VLLM_SPARK_GEMMA_TENSOR_TRACE_LAYERS=layers.0,layers.5,layers.6,model,lm_head \
```

Run one row with:

```bash
KV_DTYPE=fp8
RUN=vllm_gemma3_27b_tensor_trace_20260609TTRACEJST_fp8_flashinfer
```

Run the matched row with:

```bash
KV_DTYPE=nvfp4
RUN=vllm_gemma3_27b_tensor_trace_20260609TTRACEJST_nvfp4_kv_flashinfer
```

Use `--enforce-eager` for this diagnostic-only tensor trace. A normal-compile attempt
on `bfa123e1f` failed during model profiling because TorchDynamo captured the Python
tensor-summary code in the Gemma 3 compiled graph. `5b67b0ea2` compiles the hook out when
Dynamo is active, but eager remains the intended mode for collecting per-layer Python
summaries. Do not use these rows as throughput evidence.

Run `scripts/openai_first_token_probe.py` before any benchmark traffic for each row, then
compare the first-token reports as before.

## Compare

After both tensor traces exist:

```bash
python3 scripts/vllm_gemma_tensor_trace_compare.py \
  --baseline results/vllm_gemma3_27b_tensor_trace_20260609TTRACEJST_fp8_flashinfer_tensor_trace.jsonl \
  --candidate results/vllm_gemma3_27b_tensor_trace_20260609TTRACEJST_nvfp4_kv_flashinfer_tensor_trace.jsonl \
  --output results/vllm_gemma3_27b_tensor_trace_20260609TTRACEJST_compare.json
```

The compare script uses the last record for each event/layer by default, which is the
intended first-token-probe record if no benchmark traffic runs after the probe.

## Pass / Fail

Pass for this diagnostic packet means it produces a localization artifact, not that
NVFP4-KV quality is fixed.

Useful outcomes:

- First large fp8/NVFP4 divergence at `flashinfer_attn_output` for `layers.0`, `5`, or
  `6`: attention/NVFP4 consumption is the next target.
- FlashInfer attention output summaries match but `gemma3_layer_*` diverges: Gemma layer
  normalization/residual path is implicated.
- Layer summaries match but `gemma3_final_hidden` diverges: accumulation across later
  layers needs a wider layer sweep.
- Final hidden matches but `gemma3_logits` top-20 overlap fails: LM head/logits path is
  implicated.

Keep the raw JSONL local or force-add only when it is small enough. Commit the summary
and compact compare JSON first.

## Live Result

Artifact: `results/vllm_gemma3_27b_tensor_trace_20260609T0115JST_summary.md`.

Rows:

- fp8 baseline:
  `vllm_gemma3_27b_tensor_trace_20260609T0115JST_fp8_flashinfer_eager`.
- NVFP4-KV candidate:
  `vllm_gemma3_27b_tensor_trace_20260609T0115JST_nvfp4_kv_flashinfer_eager`.

First-token comparator result: fp8 returned `spark`, `4`, and `A`; NVFP4-KV returned
` Reigns`, Gujarati text, and `ioane`, with `0.0` top-logprob overlap in all three
cases.

Tensor compare result: all `561` event/layer keys matched across rows, so the
instrumentation did not lose a layer on either side. The strongest localization signal
is `flashinfer_attn_output`: NVFP4-KV attention outputs are BF16-shaped but become
almost entirely nonnegative with means around `124..126` and max values exactly
`255.0` on many layers. The final hidden-state RMS later looks nearly identical, but
the logits top-20 sets are disjoint. Next target is FlashInfer FA2 NVFP4 attention
output scaling/dequantization/V-scale deswizzle or output-buffer interpretation.
