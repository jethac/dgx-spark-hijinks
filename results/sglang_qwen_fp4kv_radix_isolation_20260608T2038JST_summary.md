# SGLang Qwen FP4-KV Radix-Cache Isolation, 2026-06-08 20:38 JST

Status: red serving path, strong root-cause localization.

Purpose: isolate why SGLang FP4-KV OpenAI/native first-token pairs diverge while the fp8
control agrees. The previous paired rows showed:

- fp8 KV: OpenAI and native both choose `**`, and both prefill groups argmax `334`.
- FP4 KV default: OpenAI chooses `**`, native chooses `ark` / token id `838`, and native
  prefill argmax is already `838`.

This artifact tests whether the FP4-native divergence is caused by public warmup/cache
seeding or by radix/prefix-cache reuse.

Common runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- piecewise CUDA graphs: disabled
- probe case: `medium_decode`, `max_new_tokens=1`, `temperature=0`
- prompt hash:
  `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`

Rows:

| row | server flags | OpenAI token | native token | endpoint match | endpoint prefill argmaxes |
|---|---|---|---|---|---|
| default FP4 pair | baseline no-graph FP4 | `**` | `ark` / `838` | fail | `334`, `838` |
| no-radix/no-warmup | `--disable-radix-cache --skip-server-warmup` | `**` | `**` / `334` | pass | `334`, `334` |
| skip-warmup only | `--skip-server-warmup` | `**` | `ark` / `838` | fail | `334`, `838` |
| radix-off only | `--disable-radix-cache` | `**` | `**` / `334` | pass | `334`, `334` |

Artifacts:

- default FP4 pair:
  `results/sglang_qwen_fp4kv_first_token_pair_20260608T2021JST_summary.md`
- fp8 control:
  `results/sglang_qwen_fp8_first_token_pair_20260608T2027JST_summary.md`
- no-radix/no-warmup:
  - endpoint probe: `results/sglang_qwen_fp4kv_first_token_noradix_20260608T2033JST.json`
  - dump summary:
    `results/sglang_qwen_fp4kv_first_token_noradix_20260608T2033JST_dump_summary.md`
  - server log:
    `results/sglang_qwen_fp4kv_first_token_noradix_20260608T2033JST_fp4_server.log`
- skip-warmup only:
  - endpoint probe:
    `results/sglang_qwen_fp4kv_first_token_skipwarmup_20260608T2036JST.json`
  - dump summary:
    `results/sglang_qwen_fp4kv_first_token_skipwarmup_20260608T2036JST_dump_summary.md`
  - server log:
    `results/sglang_qwen_fp4kv_first_token_skipwarmup_20260608T2036JST_fp4_server.log`
- radix-off only:
  - endpoint probe:
    `results/sglang_qwen_fp4kv_first_token_radixoff_20260608T2038JST.json`
  - dump summary:
    `results/sglang_qwen_fp4kv_first_token_radixoff_20260608T2038JST_dump_summary.md`
  - server log:
    `results/sglang_qwen_fp4kv_first_token_radixoff_20260608T2038JST_fp4_server.log`

Interpretation:

- The FP4 first-token split is not caused by OpenAI/native prompt serialization, logits
  preprocessing, or endpoint sequencing alone. fp8 proves the endpoint sequence can agree.
- `--skip-server-warmup` alone does not fix the FP4 failure, so public warmup is not the
  root cause.
- `--disable-radix-cache` fixes the FP4 first-token split even when normal server warmup is
  enabled, so the failure is localized to radix/prefix-cache reuse or FP4 cached-prefix
  read/merge behavior.
- This does not bless SGLang FP4 KV yet. Disabling radix cache is a diagnostic workaround
  and likely gives up an important serving optimization. The next code-level diagnostic
  should tag `extend_prefix_lens_cpu`, `prefix_indices`, and `use_ragged`/`extend_no_prefix`
  around prefix matching and FlashInfer prefill metadata, then compare the default FP4
  native request against the radix-off FP4 native request.

Next hook candidates:

- `sglang.srt.managers.schedule_batch.Req.init_next_round_input`: record
  `tree_cache.match_prefix(...)`, `prefix_indices`, `host_hit_length`, and
  `cache_protected_len`.
- `sglang.srt.layers.attention.flashinfer_backend.FlashInferAttnBackend.init_forward_metadata`:
  record `forward_batch.extend_prefix_lens_cpu`, `use_ragged`, and `extend_no_prefix`.
- `sglang.srt.layers.attention.flashinfer_backend.FlashInferAttnBackend.forward_extend`:
  distinguish `extend_paged` from `extend_merge_paged` for the native request.
