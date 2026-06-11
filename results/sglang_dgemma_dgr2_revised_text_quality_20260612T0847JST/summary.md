# SGLang DiffusionGemma DG-R2 Revised Text-Only Quality Gate

Date: 2026-06-12 JST

Status: GREEN for the scoped revised text-only gate.

This row reruns DG-R2 on the stock SGLang DiffusionGemma runtime, but replaces
the earlier terse "answer only" prompts with direct chat prompts that the
prompt diagnostic showed are appropriate for this dLLM serving path. The older
terse-prompt baseline remains RED and is not overwritten by this result.

## Runtime

- Host: `thinkstationpgx-00b4`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd`
- SGLang commit in image: `651d55cd2e`
- FlashInfer commit in image: `f99323bd`
- Model: `google/diffusiongemma-26B-A4B-it`
- KV dtype: `auto` / BF16
- Attention policy: DiffusionGemma forced Triton attention
- Page size: 256
- Context length: 8192
- `mem_fraction_static`: 0.55
- Docker cap: `--memory=100g --memory-swap=100g`
- dLLM config: `Gemma4Renoise`, `max_denoising_steps=48`, `seed=1234`

## Gate

The client used OpenAI chat completions with `temperature=0`, two repeats per
prompt, and parseable semantic checks:

| Prompt id | Check | Result |
| --- | --- | --- |
| `capital_japan_direct` | non-empty, byte-stable, contains Tokyo | GREEN |
| `arithmetic_2_plus_2_direct` | non-empty, byte-stable, contains standalone `4` | GREEN |
| `dgx_spark_use` | non-empty, byte-stable, mentions local/desktop/development AI use | GREEN |

Observed outputs:

- `The capital of Japan is **Tokyo**.`
- `2 + 2 = 4`
- `The NVIDIA DGX Spark desktop is designed for high-performance AI development, data science, and machine learning prototyping in a compact, desktop-class form factor.`

The row produced HTTP 200 responses for all requests and `all_ok=true` in
`revised_text_quality.json`.

## Scope

This is a text-only quality gate for SGLang's stock DiffusionGemma cookbook-style
runtime on GB10. It does not claim image-path correctness, FlashInfer attention
performance, NVFP4 KV correctness, or dLLM endpoint correctness through native
`/generate`. Native `/generate` remains a diagnostic-only path from the prior
prompt diagnostic.

The prior red result is still important: overly terse "answer only" prompts can
denoise to repeated `<eos>` tokens and become empty after special-token
stripping. This green row establishes that the runtime can pass a scoped,
reproducible, direct-chat text gate once that prompt pathology is excluded.

## Stop State

The server was torn down after the run. `docker ps` was empty and the Claude
window marker was absent after cleanup.
