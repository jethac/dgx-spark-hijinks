# vLLM Qwen Clean-Path PPL Smoke Attempt

Date: 2026-06-09 JST

Status: no PPL number accepted.

## Goal

Validate the staged clean full-attention Qwen3.6 fp8-KV versus NVFP4-KV PPL harness on GB10 before running the full 8k / 32k / 128k sweep.

## Harness

- script: `scripts/vllm_prompt_ppl_sweep.py`
- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- model path: `/home/jethac/models/aeon/qwen36-nvfp4`
- served model: `qwen36-fast`
- source text: `docs/CAMPAIGN_LOG.md`
- intended context: `8192`
- prefix caching: disabled with `--no-enable-prefix-caching`

The harness uses raw `/v1/completions` with token-ID prompt input, `prompt_logprobs=1`, `max_tokens=1`, and `return_token_ids=true`. It rejects rows where the prompt-logprob entry does not contain the supplied token ID for that position.

## Attempt 1

Artifact: `results/vllm_qwen_ppl_smoke_20260609Tlocal_fp8_server.log`

Result: stopped before a PPL request ran.

Findings:

- Server args prove `enable_prefix_caching=False` and `kv_cache_dtype='fp8'`.
- The row was already non-isolated: the server selected `VLLM_CUTLASS` NvFp4 MoE, while the accepted Qwen NVFP4-KV capacity row used `MARLIN`.
- The server reached `GPU KV cache size: 6,076,050 tokens`.
- Startup was slow: `init engine (profile, create kv cache, warmup model) took 466.81 s`.
- The readiness wait missed the healthy window; no prompt-logprob response was captured.

## Attempt 2

Artifact: `results/vllm_qwen_ppl_smoke_20260609Tlocal_fp8_retry_server.log`

Result: aborted deliberately before PPL.

Finding: the retry also selected `VLLM_CUTLASS` NvFp4 MoE. That confirms the staged command did not reproduce the accepted capacity row's non-KV backend and is unsuitable for isolating the KV-cache dtype effect.

## Stop Condition

No vLLM PPL containers were left running, and GPU utilization was back to idle.

## Backend Isolation Probe

Artifact: `results/vllm_qwen_ppl_moe_backend_probe_20260609Tmarlin_server.log`

Result: `VLLM_TEST_FORCE_FP8_MARLIN=1` fixes the non-KV backend selection for this launch skeleton.

Evidence:

- Server args still prove `enable_prefix_caching=False` and `kv_cache_dtype='fp8'`.
- The log records `Using 'MARLIN' NvFp4 MoE backend out of potential backends: ['VLLM_CUTLASS', 'MARLIN', 'EMULATION']`.
- The probe was stopped immediately after backend selection, before full model load, and left the GPU idle.

Next action: rerun the packet in `tasks/vllm_qwen_nvfp4_kv_clean_ppl_sweep_20260609.md` with `VLLM_TEST_FORCE_FP8_MARLIN=1` on both fp8 and NVFP4 launches before collecting fp8/NVFP4 PPL numbers.
