# SGLang Qwen FP4-KV Radix Trace, 2026-06-08 21:30 JST

Status: red serving path, cached-prefix merge path isolated.

Purpose: compare SGLang FP4-KV default radix behavior against `--disable-radix-cache`
with request/radix/ForwardBatch/FlashInfer metadata tracing enabled. This follows the
earlier radix-isolation row, where disabling radix cache fixed the FP4 OpenAI/native
first-token split.

Common runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@ce1b6d15e76985240e91592a0f44c0f282fc65af`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- page size: `1`
- probe: `medium_decode`, `max_new_tokens=1`, `temperature=0`
- trace env: `SGLANG_FP4_KV_TRACE_BACKEND=1`, `SGLANG_FP4_KV_TRACE_RADIX=1`
- prompt hash:
  `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`

Rows:

| row | server flag delta | OpenAI first token | native first token | match | native cached tokens | native FlashInfer path |
|---|---|---|---|---|---:|---|
| default FP4 | none | `**` | `ark` / `838` | fail | 55 | `extend_merge_paged` |
| radix-off FP4 | `--disable-radix-cache` | `**` | `**` / `334` | pass | 0 | `forward_extend_ragged_no_prefix` |

Key trace delta:

| signal | default FP4 native request | radix-off FP4 native request |
|---|---|---|
| `tree_disable` | `False` | `True` |
| `prefix_indices_len` | `55` | `0` |
| `prefix_indices` | cache locations `4113..4167` | empty |
| `extend_prefix_lens_cpu` | `[55]` | `[0]` |
| `extend_seq_lens_cpu` | `[1]` | `[56]` |
| `out_cache_loc` | one new token at `4170` | full prompt at `4170..4225` |
| `use_ragged` | `True` | `True` |
| `extend_no_prefix` | `False` | `True` |
| FlashInfer trace label | `forward_extend_merge_paged` | `forward_extend_ragged_no_prefix` |
| server accounting | `#new-token: 1`, `#cached-token: 55` | `#new-token: 56`, `#cached-token: 0` |

Artifacts:

- default endpoint JSON:
  `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_default.json`
- default server log:
  `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_default_fp4_server.log`
- default container inspect:
  `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_default_fp4_container_inspect.json`
- radix-off endpoint JSON:
  `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_radixoff.json`
- radix-off server log:
  `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_radixoff_fp4_server.log`
- radix-off container inspect:
  `results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_radixoff_fp4_container_inspect.json`

Interpretation:

- The FP4 failure is not just "radix enabled" in the abstract. The failing native request
  reuses 55 cached prompt tokens and runs FlashInfer's paged-prefix merge path with one new
  token.
- The same prompt, endpoint sequence, model, KV dtype, and no-graph policy passes when the
  radix cache is disabled. In that row SGLang recomputes the full 56-token prompt and stays
  on the no-prefix ragged extend path.
- This trace does not yet prove the exact low-level defect. It proves the defect is behind
  FP4 cached-prefix reuse / `extend_merge_paged`, where packed FP4 K/V bytes and FP8 scale
  buffers must be read for cached pages and one newly written page.
- `--disable-radix-cache` remains a correctness workaround, not a blessed serving fix.

Next hook:

Instrument the cached-prefix merge path at page granularity: for the 55 reused pages and
the one newly written page, record the physical page IDs handed to FlashInfer and verify
that K data, K scale, V data, and V scale are all derived from the same page indices.
The next useful comparison is default FP4 `extend_merge_paged` versus radix-off
`forward_extend_ragged_no_prefix`, not another raw quantizer or standalone-kernel probe.
