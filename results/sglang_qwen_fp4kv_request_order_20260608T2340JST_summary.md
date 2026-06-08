# SGLang Qwen FP4-KV Request-Order Probe, 2026-06-08 23:40 JST

Status: red serving path; root cause is endpoint-independent FP4 cached-prefix reuse.

Purpose: run one SGLang FP4-KV server with request-tagged radix/ForwardBatch traces and
test whether the known OpenAI/native split follows endpoint identity or the second
request's radix-cache hit.

Runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@d4fe78078633e70fde968c58032a675c72f13bc1`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- page size: `1`
- probe: `medium_decode`, `max_new_tokens=1`, `temperature=0`

Rows:

| row | first request | first cached tokens | first token | second request | second cached tokens | second token | verdict |
|---|---|---:|---|---|---:|---|---|
| baseline | OpenAI | 0 | `**` | native | 55 | `ark` / `838` | second request fails |
| reverse | native | 0 | `**` / `334` | OpenAI | 55 | `ark` | second request fails |
| flush-between | OpenAI | 0 | `**` | native after flush | 0 | `**` / `334` | both pass |
| namespace isolation | OpenAI `extra_key/cache_salt` | 0 | `**` | native different `extra_key` | 0 | `**` / `334` | both pass |

Trace evidence:

- `native-second` and `openai-second` both hit `prefix_indices_len=55` and
  `extend_prefix_lens_cpu=[55]`, consume one new token at position `55`, and emit `ark`.
- `native-first`, `openai-first`, `openai-flush-first`, `native-after-flush`,
  `openai-namespace-a`, and `native-namespace-b` all have `prefix_indices_len=0` and emit
  `**`.
- Namespace isolation is meaningful: OpenAI's effective namespace is
  `fp4-order-openaifp4-order-openai` (`cache_salt + extra_key`), while native uses
  `fp4-order-native`; both miss radix and pass.
- Calibration happened once before the probe requests:
  `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`.

Interpretation:

- The split is not OpenAI serialization, native `/generate`, request IDs, or prompt IDs.
  The endpoint that runs second and reuses a 55-token FP4 cached prefix fails.
- `extra_key`/`cache_salt` preventing the cache hit also prevents the first-token split.
- This reframes the remaining SGLang blocker as FP4 cached-prefix quality/reuse, not
  endpoint mismatch. Prior traces already showed the sampled packed bytes, scale bytes,
  paged-prefix read, LSE convention, and merge math are internally consistent; therefore
  the next fix/probe should compare the quality impact of using FP4 cached-prefix attention
  versus full dense prefill and test whether better scales or a selective no-reuse policy
  is required.

Artifacts:

- full JSON: `results/sglang_qwen_fp4kv_request_order_20260608T2340JST.json`
- parsed analysis: `results/sglang_qwen_fp4kv_request_order_20260608T2340JST_analysis.json`
- radix trace lines: `results/sglang_qwen_fp4kv_request_order_20260608T2340JST_radix_lines.txt`
- ForwardBatch trace lines:
  `results/sglang_qwen_fp4kv_request_order_20260608T2340JST_forward_batch_lines.txt`
- calibration lines:
  `results/sglang_qwen_fp4kv_request_order_20260608T2340JST_calibration_lines.txt`
- server log:
  `results/sglang_qwen_fp4kv_request_order_20260608T2340JST_fp4_server.log`
- container inspect:
  `results/sglang_qwen_fp4kv_request_order_20260608T2340JST_fp4_container_inspect.json`
