# vLLM Gemma 3 Tensor Trace Packet, 2026-06-09

Status: next live diagnostic packet.

Purpose: localize the Gemma 3 27B NVFP4-KV first-token quality failure above the
page/scale byte-pairing layer. The prior trace proved sampled read-side packed K/V and
FP8 scale bytes match write-side bytes (`195 / 195`) and the failing short prompts do not
skip SWA blocks. This packet asks whether the first visible divergence is in FlashInfer
attention output, Gemma layer hidden state, final hidden state, or logits.

## Source Pins

- `jethac/vllm@spark/hijinks-021-gemma3-tensor-trace`
- vLLM commit: `bfa123e1fc06b55d8fc420ef93127bcdca23c8b1`
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
`bfa123e1fc06b55d8fc420ef93127bcdca23c8b1` and add:

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
