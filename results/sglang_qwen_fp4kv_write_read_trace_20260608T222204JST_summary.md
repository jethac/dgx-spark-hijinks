# SGLang Qwen FP4-KV Write/Read Trace, 2026-06-08 22:22 JST

Status: red serving path; sampled cache write/read pairing is clean.

Purpose: follow the merge-state trace with direct write/read correlation for the cached
FP4 KV pages. This row uses `jethac/sglang@f76f80484`, which adds
`SGLANG_FP4_KV_TRACE_WRITE_READ=1` instrumentation in `MHATokenToKVPoolFP4.set_kv_buffer()`
and attaches matching write records to the `extend_merge_paged` read-side trace.

Common runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@f76f80484`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- page size: `1`
- trace env:
  `SGLANG_FP4_KV_TRACE_BACKEND=1`,
  `SGLANG_FP4_KV_TRACE_RADIX=1`,
  `SGLANG_FP4_KV_TRACE_PAGE_PAIR=1`,
  `SGLANG_FP4_KV_TRACE_MERGE_STATE=1`,
  `SGLANG_FP4_KV_TRACE_WRITE_READ=1`,
  `SGLANG_FP4_KV_TRACE_LAYERS=0`,
  `SGLANG_FP4_KV_TRACE_VALUES=16`,
  `SGLANG_FP4_KV_TRACE_LOC_LIMIT=128`
- probe: `medium_decode`, `max_new_tokens=1`, `temperature=0`

Rows:

| row | OpenAI first token | native first token | match | native cached tokens | write traces | merge traces |
|---|---|---|---|---:|---:|---:|
| default FP4 | `**` | `ark` / `838` | fail | 55 | 16 | 1 |
| radix-off FP4 | `**` | `**` / `334` | pass | 0 | 16 | 0 |

Default cached-prefix result:

- path: `extend_merge_paged`
- sampled layer: `0`
- cached prefix length: `55`
- sampled cached page IDs in merge trace: `4113, 4114, 4115, 4116`
- global scales: `k_scale=0.1197916716337204`, `v_scale=0.0016276042442768812`
- byte match result:
  - page `4113`: K data, V data, K scale, V scale all match write input = stored bytes = read bytes
  - page `4114`: K data, V data, K scale, V scale all match write input = stored bytes = read bytes
  - page `4115`: K data, V data, K scale, V scale all match write input = stored bytes = read bytes
  - page `4116`: K data, V data, K scale, V scale all match write input = stored bytes = read bytes

Layer-0 merge tensor summary:

| tensor | finite | min | max | mean |
|---|---|---:|---:|---:|
| `s1_ragged` | true | 55.470489501953125 | 19480.431640625 | 3666.537109375 |
| `s2_paged` | true | 63.34410858154297 | 20003.060546875 | 3760.69921875 |
| `merged` | true | -1.875 | 3.1875 | 0.010337634943425655 |

Interpretation:

- The failure still reproduces only when radix/prefix cache reuses a 55-token FP4 KV
  prefix. Disabling radix cache still recomputes the prompt and returns the expected first
  token.
- The sampled cached pages prove the write-side packed K/V bytes and FP8 scale bytes are
  the same bytes that are stored in the FP4 pool and then read by the FlashInfer FA2
  cached-prefix path.
- This makes the simple "scale buffer was not copied / got stale / points at the wrong
  page" hypothesis unlikely for the sampled pages.
- The next useful proof is numerical, not structural: compare the cached FP4 paged-prefix
  contribution against a same-prompt no-prefix/ragged reference for the 55-token prefix,
  or trace the FlashInfer FA2 paged-prefix reader's dequantized values / logits before
  merge.

Artifacts:

- default endpoint JSON:
  `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_default.json`
- default server log:
  `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_default_fp4_server.log`
- default container inspect:
  `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_default_fp4_container_inspect.json`
- radix-off endpoint JSON:
  `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_radixoff.json`
- radix-off server log:
  `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_radixoff_fp4_server.log`
- radix-off container inspect:
  `results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_radixoff_fp4_container_inspect.json`

Next hook:

Build a same-prompt cached-vs-recomputed numerical comparator:

- capture the 55-token prefix contribution through the cached FP4 paged path;
- capture an equivalent no-prefix/full-ragged FP4 path for the same prefix and query;
- compare attention output / LSE / logits before sampling.

If cached paged output differs from the recomputed FP4 reference, the remaining bug is in
the FlashInfer FA2 paged-prefix integration or its dequant/scale convention under prefix
reuse. If they match, continue upward to sampling/logit integration.
