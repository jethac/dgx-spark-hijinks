# SGLang Qwen FP4-KV Dense-vs-Cached Trace Probe, 2026-06-09

Status: next diagnostic packet; do not run Gemma until this Qwen issue is green.

Purpose: compare the no-prefix/full-prefill FP4 path against the 55-token cached-prefix
FP4 path at the tensor boundary where the first-token distribution becomes disjoint.
The current request-order artifact proves cached-prefix rows have `0 / 20` top-logprob
overlap with full-prefill rows, while no-reuse controls have `20 / 20`.

Driver:

- Use `scripts/sglang_fp4_request_order_probe.py`.
- It already produces the required pairs:
  - dense reference: `openai-first` / `native-first`, `cached_tokens=0`
  - cached failure: `native-second` / `openai-second`, `cached_tokens=55`
  - controls: flush-between and namespace-isolation, both `cached_tokens=0`

Instrumentation gate:

```bash
SGLANG_FP4_KV_TRACE_DENSE_CACHE=1
SGLANG_FP4_KV_TRACE_LAYERS=0,1,7,13,20,27
SGLANG_FP4_KV_TRACE_VALUES=full
```

Initial insertion points:

- `third_party/sglang/python/sglang/srt/model_executor/forward_batch_info.py`
  - Extend the existing ForwardBatch/radix trace to emit `rid`, `forward_mode`,
    `extend_prefix_lens_cpu`, `extend_seq_lens_cpu`, `positions`, `seq_lens`,
    `out_cache_loc`, and `sample_row = cumsum(extend_seq_lens) - 1`.
  - This maps the full dense row's final prompt token to the cached-prefix suffix row.
- `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
  - In `forward_extend_ragged_no_prefix`, after ragged attention returns, dump the
    last-token attention output for the dense reference row.
  - In `forward_extend_merge_paged`, after `o2/s2` and `_safe_merge_state`, dump `o1`,
    `s1`, `o2`, `s2`, and merged `o` for the cached-prefix row.
- `third_party/sglang/python/sglang/srt/models/qwen2.py`
  - In `Qwen2DecoderLayer.forward`, dump selected-layer last-token vectors after
    self-attention, after post-attention norm, after MLP, and at the residual boundary.
  - Start with layers `0,1,7,13,20,27`; expand to all 28 layers only if needed.
  - Also dump the final normalized last-token hidden state before logits.
- `third_party/sglang/python/sglang/srt/layers/logits_processor.py`
  - Dump `pruned_states`, `sample_indices`, raw logits, and sampled logits before sampler
    preprocessing. Mirror the same capture for the `extend_return_logprob` path.
- `third_party/sglang/python/sglang/srt/model_executor/model_runner.py`
  - Dump `next_token_logits` immediately before and after `_preprocess_logits`.
  - The existing patch target in `scripts/sglang_fp4_first_token_dump_patch.yaml` proves
    this location is sufficient for the sampler boundary.

Capture key:

```text
rid, forward_pass_id, forward_mode, layer_id, phase,
extend_prefix_len, extend_seq_len, sample_row
```

Capture values:

- last-token hidden vectors as CPU float32 tensors
- full first-token logits before and after preprocessing
- top-20 token ids/values for logits
- stats: cosine, max_abs, RMS, relative RMS, finite count

Decision rule:

- First divergence at layer-0 attention output:
  FP4 cached-prefix attention quality differs from dense full prefill despite internally
  consistent paged read/LSE/merge traces.
- Attention and hidden vectors match, but raw logits diverge:
  LM head/logits path.
- Raw logits match, but post-preprocess logits diverge:
  sampler preprocessing, logits bias, or grammar state.
- All internal tensors match, but API top-logprobs diverge:
  response/logprob packaging path.

Acceptance:

- The probe must produce one artifact comparing dense no-cache rows against cached-prefix
  rows by `rid` and phase.
- Do not bless `--disable-radix-cache` or selective no-reuse as the FP4-KV capacity path.
  It remains a diagnostic/emergency workaround only.
