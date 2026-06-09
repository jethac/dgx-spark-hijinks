# Trace Audit Hardening, 2026-06-09

Status: offline hardening completed; live GB10 trace rows still queued behind host access.

## SGLang Dense-vs-Cached FP4-KV Trace

- Fork branch: `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving`
- Previous trace head: `43f3123f9`
- New trace head: `e631a13fd`
- Change: logits and sampler trace payloads now include `sample_rows` in addition to
  `kind`, `forward_pass_id`, request ids, mode, and prefix/extend lengths.

Campaign comparator hardening:

- `scripts/sglang_dense_cache_trace_compare.py`
  - validates declared trace `kind` against the log prefix;
  - rejects missing required event metadata;
  - rejects null `forward_pass_id`;
  - rejects missing or malformed request ids;
  - rejects missing `sample_rows`;
  - treats unknown rid/request binding as red;
  - reports `metricless_comparison_count` and fails if any matched dense/cached comparison
    row has no usable vector or top-k metric.
- `scripts/sglang_dense_cache_trace_summary_audit.py`
  - rejects non-zero `metricless_comparison_count`;
  - rejects non-empty `event_schema_issues`.

Validation:

- `python -m py_compile python/sglang/srt/layers/logits_processor.py python/sglang/srt/model_executor/model_runner.py`
- `python -m py_compile scripts/sglang_dense_cache_trace_compare.py scripts/sglang_dense_cache_trace_summary_audit.py`
- `python scripts/sglang_dense_cache_trace_compare.py --self-test`

Outcome: the queued SGLang live packet can no longer pass merely because structural dense
and cached trace keys matched. It must bind metric-bearing trace evidence to known dense
and cached request rows.

## vLLM FlashInfer Paged-Prefill Debug Audit

Subagent review found the staged audit could combine a correct-looking paged identity line
with a tensor dump from a different paged call because the current FlashInfer debug log does
not emit a shared call/module key.

Immediate hardening added to `scripts/flashinfer_prefill_debug_log_audit.py`:

- require runtime `maybe_k_cache_sf` and `maybe_v_cache_sf` tensor views;
- reject null or empty tensor pointers;
- require one device across Q, K, V, scale tensors, and output;
- require packed FP4x2 paged K/V carrier width `shape[-1] == head_dim / 2`.

The packet now documents the remaining limitation: a future FlashInfer debug patch should
emit `debug_call_id`, generated-module URI/source path, object/cache path, compile flags,
and a `paged_run` identity stamp on both identity and tensor lines before deeper
fragment-level instrumentation is treated as fully bound to one C++ call.

Validation:

- `python -m py_compile scripts/flashinfer_prefill_debug_log_audit.py`

Generated campaign audits:

- `python scripts/live_task_queue_audit.py --queue tasks/live_gb10_queue.jsonl --output results/live_task_queue_audit_20260609.json`
- `python scripts/solution_coverage_audit.py --output results/solution_coverage_audit_20260609.json`

Both audits passed.
