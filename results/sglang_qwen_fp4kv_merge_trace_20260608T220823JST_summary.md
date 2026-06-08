# SGLang Qwen FP4-KV Merge-State Trace, 2026-06-08 22:08 JST

Status: red serving path; cached-prefix merge is now instrumented.

Purpose: follow the page-pair trace with a direct trace inside `extend_merge_paged` on
`jethac/sglang@991ac1e63`. This row samples packed FP4 K/V bytes, FP8 K/V scale bytes,
and the ragged/paged/merged tensors around `_safe_merge_state(o1, s1, o2, s2)`.

Common runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@991ac1e63`
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
  `SGLANG_FP4_KV_TRACE_LAYERS=0`,
  `SGLANG_FP4_KV_TRACE_VALUES=16`
- probe: `medium_decode`, `max_new_tokens=1`, `temperature=0`

Rows:

| row | OpenAI first token | native first token | match | native cached tokens | merge trace |
|---|---|---|---|---:|---|
| default FP4 | `**` | `ark` / `838` | fail | 55 | 1 layer-0 trace |
| radix-off FP4 | `**` | `**` / `334` | pass | 0 | none |

Default cached-prefix signals:

- path: `extend_merge_paged`
- sampled layer: `0`
- `swa_window_left=-1`
- `paged_kernel_lens=[55]`
- `prefix_lens=[55]`
- `seq_lens=[56]`
- `kv_indptr=[0,55]`
- `kv_indices_used`: head `4113..4120`, tail `4160..4167`
- sampled page IDs: `4113, 4114, 4115, 4116`
- global scales: `k_scale=0.1197916716337204`, `v_scale=0.0016276042442768812`
- sampled K/V packed data and K/V FP8 scale bytes are nonzero and readable at those pages.

Layer-0 merge tensor summary:

| tensor | finite | min | max | mean | first sample |
|---|---|---:|---:|---:|---:|
| `o1_ragged` | true | -2.09375 | 1.78125 | -0.0031201243 | -0.1025390625 |
| `s1_ragged` | true | 55.4704895 | 19480.4316406 | 3666.5371094 | 792.6002197 |
| `o2_paged` | true | -1.875 | 3.1875 | 0.0109496191 | 0.0247802734 |
| `s2_paged` | true | 63.3441086 | 20003.0605469 | 3760.6992188 | 814.2984619 |
| `merged` | true | -1.875 | 3.1875 | 0.0103376349 | 0.0247802734 |

Interpretation:

- The failing default request still uses the same 55 cached prefix pages identified by the
  radix and page-pair traces.
- The trace now proves the paged-prefix branch sees readable packed K/V data and readable
  FP8 scale buffers for those pages; it also proves the ragged suffix, paged prefix, and
  merged output are finite at layer 0.
- The merged output's first sampled values match the paged-prefix output, consistent with
  the paged prefix dominating the one-token suffix at this layer. That is plausible for a
  long cached prefix and not itself proof of a merge bug.
- The remaining bug is therefore narrower: compare the cached FP4 paged-prefix numerical
  result against a recomputed full/ragged reference for the same 56-token prompt, or trace
  write-time bytes/scales for the same pages and verify they match the bytes/scales read
  during cached-prefix reuse.

Artifacts:

- default endpoint JSON:
  `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_default.json`
- default server log:
  `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_default_fp4_server.log`
- default container inspect:
  `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_default_fp4_container_inspect.json`
- radix-off endpoint JSON:
  `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_radixoff.json`
- radix-off server log:
  `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_radixoff_fp4_server.log`
- radix-off container inspect:
  `results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_radixoff_fp4_container_inspect.json`

Next hook:

Add a write/read pairing trace for `MHATokenToKVPoolFP4.set_kv_buffer()` and the
`extend_merge_paged` reader so the same physical page IDs can be compared at write time and
read time. If bytes and scales match, build a same-prompt no-prefix reference for the paged
prefix contribution and inspect FlashInfer FA2 paged-prefix numerics / merge weighting.
