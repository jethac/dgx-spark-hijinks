# vLLM Qwen Clean-Path 32k PPL Stop Point

Date: 2026-06-09 JST

Status: no accepted 32k PPL row.

This run attempted to extend the accepted 8k clean full-attention Qwen NVFP4-KV PPL
sweep to `ctx=32768` using the same image and model:

- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- Runtime ref: `jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d`
- Model: `/home/jethac/models/aeon/qwen36-nvfp4`
- Served model: `qwen36-fast`
- Prefix caching: disabled
- MoE backend control: `VLLM_TEST_FORCE_FP8_MARLIN=1`
- Context: `32768`

The first attempt exposed a runner bug: the readiness probe used `docker exec` without
`-i`, so the probe script received no stdin and exited successfully before `/v1/models`
was actually serving. `scripts/run_vllm_qwen_clean_ppl_pair.sh` now uses
`docker exec -i` for the health probe.

After the readiness fix, the live fp8 server progressed through model load, Torch compile,
FlashInfer JIT, warmup, and graph profiling:

- `Using 'MARLIN' NvFp4 MoE backend`
- `Using AttentionBackendEnum.FLASHINFER backend`
- FlashInfer JIT built `fp4_gemm_cutlass_sm120` with `-gencode=arch=compute_121a,code=sm_121a`
- FlashInfer JIT built a fp8 batch-prefill module with `head_dim_qk_256/head_dim_vo_256`
- `Initial profiling/warmup run took 196.62 s`
- `Profiling CUDA graph memory: PIECEWISE=35 (largest=256), FULL=19 (largest=128)`

The server eventually became ready and the `ctx=32768` fp8 supplied-token logprob request
was launched. That request did not return in the available wait window, and subsequent SSH
connections to `100.113.98.11:22` timed out. Local hung SSH clients were terminated, but
the remote host could not be reached to retrieve final logs or verify container cleanup in
this turn.

Interpretation:

- This is not a quality result and must not be quoted as fp8-vs-NVFP4 PPL.
- The generated long corpus is valid for longer rows: it tokenized to `316301` Qwen tokens,
  enough for 32k and 128k.
- The next long-PPL attempt should either prewarm/prebuild the FlashInfer cache and use a
  stricter request timeout, or run a smaller intermediate context before retrying 32k.
- The operational issue itself is useful campaign evidence: clean-path long-context
  prompt-logprob PPL is substantially heavier than the 8k row, and the current runner needs
  remote watchdog/cleanup handling before 32k/128k sweeps are unattended.
