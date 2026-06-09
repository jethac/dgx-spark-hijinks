# vLLM Qwen Clean-Path NVFP4-KV PPL Row

Date: 2026-06-09 JST

Scope: clean full-attention Qwen3.6 path only. Prefix caching was disabled with
`--no-enable-prefix-caching`; Gemma/SWA and SGLang radix/prefix-cache reuse paths are
excluded because they are separately known-broken.

Runtime:

- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- vLLM: `jethac/vllm@a919d635d`
- FlashInfer: `jethac/flashinfer@e152cf4d`
- FlashAttention: `jethac/flash-attention@7d53245`
- MoE backend control: `VLLM_TEST_FORCE_FP8_MARLIN=1` on both fp8 and NVFP4 rows
- Hardware key: `NVIDIA_GB10:sm_121:sms_48`

FlashInfer scope:

- `e152cf4d` includes `6b7513ee` (`Enable SM121 FP4 dispatch and 121a cache builds`).
- `e152cf4d` includes `a42c8f07` (`Add SM121 FP4 heuristic regression test`).
- `e152cf4d` adds the FA2 NVFP4 KV explicit scale-stride plumbing used by this row.
- `e152cf4d` does not include the later `3db181f4` JIT module-name disambiguation or
  `spark/hijinks-020` JIT URI follow-ups.

Acceptance evidence:

- fp8 row: `enable_prefix_caching=False`, `kv_cache_dtype=fp8`, `MARLIN` NvFp4 MoE backend.
- NVFP4 row: `enable_prefix_caching=False`, `kv_cache_dtype=nvfp4`, `MARLIN` NvFp4 MoE backend.
- NVFP4 row selected: `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled.`
- fp8 KV pool: `6,675,527` tokens.
- NVFP4 KV pool: `11,456,401` tokens.
- Capacity ratio in this no-prefix-cache launch: `1.716x`.

PPL result over `docs/CAMPAIGN_LOG.md` with supplied prompt-token logprobs:

| ctx | PPL fp8 | PPL NVFP4 | delta PPL | nats/token fp8 | nats/token NVFP4 | delta nats/token |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 7.089731808 | 7.125866375 | +0.036134567 | 1.958647513 | 1.963731315 | +0.005083802 |

Artifacts:

- `results/vllm_qwen_clean_ppl_20260609T0850JST_fp8_ctx8192_ppl.json`
- `results/vllm_qwen_clean_ppl_20260609T0850JST_nvfp4_ctx8192_ppl.json`
- `results/vllm_qwen_clean_ppl_20260609T0850JST_compare.json`
- `results/vllm_qwen_clean_ppl_20260609T0850JST_fp8_server_before_ppl.log`
- `results/vllm_qwen_clean_ppl_20260609T0850JST_nvfp4_server_before_ppl.log`

Interpretation: this is the first clean-path quality number behind the Qwen NVFP4-KV
capacity claim. At 8k context, the measured cost is small (`+0.0051` nats/token), but this
does not clear the broken reuse paths and does not yet answer whether the delta compounds
at 32k or 128k.
