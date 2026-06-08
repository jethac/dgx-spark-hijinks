# SGLang Qwen FP4 KV Native Logprob Divergence Probe

Date: 2026-06-08 JST

Purpose: narrow the FP4-KV quality failure from "bad benchmark text" to an exact
generated-token divergence using SGLang's native `/generate` endpoint.

## Setup

- Runtime: `nvcr.io/nvidia/sglang:26.05-py3` with `jethac/sglang@d7d931f` source
  overlay.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Prompt: `medium_decode` from `scripts/openai_serving_benchmark.py`, rendered with the
  Qwen chat template via `transformers.AutoTokenizer.apply_chat_template`.
- Comparator: fp8 KV on port `30012`.
- Candidate: FP4 KV on port `30013`.
- Shared settings: FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph
  disabled, piecewise CUDA graph disabled.
- Probe: `scripts/sglang_native_logprob_compare.py`, native `/generate`,
  `return_logprob=true`, `return_text_in_logprobs=true`, `top_logprobs_num=20`.

## Artifacts

- `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_native_logprob_compare.json`
- `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_fp8_server.log`
- `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_fp4_server.log`
- `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_fp4_trace_excerpt.txt`
- `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_run_id.txt`

## Result

- Both runs used the same rendered prompt: 56 prompt tokens.
- Both completed 192 output tokens.
- fp8 and FP4 generated the same first four tokens:
  `**`, `Engineering`, ` Note`, `:`.
- First token divergence is output token index `4`:
  - fp8 selected token id `7818`, text ` Valid`, logprob `-0.6169`.
  - FP4 selected token id `23282`, text ` Validate`, logprob `-1.3321`.
- The alternate token is visible in each top-k distribution:
  - fp8 ranks ` Validate` fifth at logprob `-2.8669`.
  - FP4 ranks ` Valid` second at logprob `-1.7071`.
- FP4 output is degraded but not the same catastrophic `import import` collapse seen in
  the earlier OpenAI Chat Completions quality probe. It begins:
  `**Engineering Note: Validate that an LLM Server is Using the Int GPU Kernels**`.

## Interpretation

This localizes the native `/generate` failure to an early decode distribution shift,
not a first-token disaster and not a total backend/scale-layout collapse. The FP4 and
fp8 distributions are close enough that the same plausible alternatives appear in both
top-k lists, but FP4 reverses their rank at token 4 and then the text quality compounds.

That changes the next debugging target: compare OpenAI Chat Completions versus native
`/generate` prompt/path handling, then inspect the logits/hidden-state perturbation at
the token-4 decode step. Do not repeat capacity rows.
