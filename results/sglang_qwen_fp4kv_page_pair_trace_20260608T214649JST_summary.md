# SGLang Qwen FP4-KV Page-Pair Trace, 2026-06-08 21:46 JST

Status: red serving path, coarse page-list mismatch not observed.

Purpose: follow the radix trace with a lower-level check of the failing cached-prefix
`extend_merge_paged` path. This row uses `jethac/sglang@839cb7457`, which adds
`SGLANG_FP4_KV_TRACE_PAGE_PAIR=1` instrumentation to record the FlashInfer paged plan and
the FP4 K/V plus scale view geometry used by the native FA2 reader.

Common runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@839cb7457`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- page size: `1`
- trace env:
  `SGLANG_FP4_KV_TRACE_BACKEND=1`,
  `SGLANG_FP4_KV_TRACE_RADIX=1`,
  `SGLANG_FP4_KV_TRACE_PAGE_PAIR=1`
- probe: `medium_decode`, `max_new_tokens=1`, `temperature=0`

Rows:

| row | OpenAI first token | native first token | match | native cached tokens | page-pair trace |
|---|---|---|---|---:|---|
| default FP4 | `**` | `ark` / `838` | fail | 55 | 28 layer traces |
| radix-off FP4 | `**` | `**` / `334` | pass | 0 | none; no paged prefix |

Default cached-prefix signals:

- radix trace: `prefix_indices_len=55`, with head `4113..4120` and tail `4160..4167`.
- FlashInfer paged plan:
  - `paged_kernel_lens=[55]`
  - `prefix_lens=[55]`
  - `seq_lens=[56]`
  - `kv_indptr=[0, 55]`
  - `kv_indices_used` head `4113..4120`, tail `4160..4167`
- FlashInfer path: `forward_extend_merge_paged`.
- page-pair traces: all 28 layers report:
  - `first_dims=[5528886, 5528886, 5528886, 5528886]`
  - `first_dim_match=True`
  - K data, V data, K scale, and V scale views all have `storage_offset=0`
  - K/V data strides are `(128, 64, 1)`
  - K/V scale strides are `(16, 16, 8, 1)`

Radix-off comparator:

- native response reports `cached_tokens=0`.
- the second 56-token request stays on `forward_extend_ragged_no_prefix`.
- there are no page-pair traces because no paged prefix is consumed.

Artifacts:

- default endpoint JSON:
  `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_default.json`
- default server log:
  `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_default_fp4_server.log`
- default container inspect:
  `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_default_fp4_container_inspect.json`
- radix-off endpoint JSON:
  `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_radixoff.json`
- radix-off server log:
  `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_radixoff_fp4_server.log`
- radix-off container inspect:
  `results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_radixoff_fp4_container_inspect.json`

Interpretation:

- The failing native request passes the coarse page-list check: the paged plan consumes the
  same 55 logical token/page indices reported by the radix prefix, and the K data, V data,
  K scale, and V scale tensors expose matching first-dimension extents to FlashInfer.
- This makes a gross cached-prefix page-ID mismatch less likely.
- The row does **not** prove the FP4 data and scale bytes at those page IDs are correct,
  nor does it prove that `merge_state(o1, s1, o2, s2)` combines the ragged suffix state
  with the paged FP4 prefix state correctly.
- SGLang FP4 KV remains red.

Next hook:

Instrument the `extend_merge_paged` branch around the merge itself:

- sample FP4 data bytes and FP8 scale bytes at the reused page IDs before the paged call,
  at least for layer 0 and the first/tail prefix pages;
- log bounded summaries of `o1`, `s1`, `o2`, and `s2` before `_safe_merge_state`;
- compare the paged-prefix result against a no-prefix dense/ragged reference for the same
  prompt. If the page bytes are sane but `o2/s2` or merge output diverges, the next fix is
  in the FlashInfer FA2 paged-prefix consumer or merge-state integration, not radix metadata.
