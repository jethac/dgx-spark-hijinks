# SGLang Qwen mixed-KV natural long-prefill gate, 2026-06-10

## Scope

This follow-up resolves an ambiguity in
`results/sglang_qwen_mixedkv_quality3_20260610TmanualJST.md`.

The older `long_prefill` benchmark prompt is built by repeating the same sentence many
times. A fresh-server control showed that prompt makes fp8 and mixed-KV both generate
repetitive text, so it is not a clean mixed-KV quality discriminator.

This row adds and uses `natural_long_prefill`, a non-repetitive long-context prompt in
`scripts/openai_serving_benchmark.py`.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- Page size: `1`
- `--mem-fraction-static`: `0.40`
- Docker cap: `--memory=100g --memory-swap=100g`
- CUDA graph: disabled
- Runs were sequential, not concurrent.

Mixed-KV means K is FP8 e4m3 and V is packed NVFP4 with FP8 scale factors.

## Artifacts

Repeated-prompt control:

- fp8 repeated prompt: `results/sglang_qwen_fp8_longprefill_freshonly_20260610TmanualJST_openai_benchmark.json`
- mixed repeated prompt: `results/sglang_qwen_mixedkv_longprefill_freshonly_20260610TmanualJST_openai_benchmark.json`
- repeated-prompt compare: `results/sglang_qwen_mixedkv_vs_fp8_longprefill_freshonly_20260610TmanualJST_quality_compare.json`

Natural-prompt gate:

- fp8 benchmark: `results/sglang_qwen_fp8_natural_longprefill_20260610TmanualJST_openai_benchmark.json`
- fp8 quality: `results/sglang_qwen_fp8_natural_longprefill_20260610TmanualJST_quality.json`
- mixed benchmark: `results/sglang_qwen_mixedkv_natural_longprefill_20260610TmanualJST_openai_benchmark.json`
- mixed quality: `results/sglang_qwen_mixedkv_natural_longprefill_20260610TmanualJST_quality.json`
- quality compare: `results/sglang_qwen_mixedkv_vs_fp8_natural_longprefill_20260610TmanualJST_quality_compare.json`

## Repeated-prompt control

Fresh-server `long_prefill` rows:

| KV mode | heuristic flags | repeated bigram fraction | repeated trigram fraction |
|---|---|---:|---:|
| fp8 K + fp8 V | `repeated_bigrams`, `repeated_trigrams` | 0.3243 | 0.2778 |
| fp8 K + NVFP4 V | `repeated_bigrams`, `repeated_trigrams` | 0.3333 | 0.2632 |

This proves the original repeated-sentence `long_prefill` prompt is a poor quality gate
for this small Qwen2.5 row. The red result in
`results/sglang_qwen_mixedkv_quality3_20260610TmanualJST.md` should be read as
"prompt gate invalid / inconclusive," not as a clean mixed-KV-specific failure.

## Natural long-prefill result

| KV mode | allocatable tokens | decode tok/s | heuristic flags |
|---|---:|---:|---|
| fp8 K + fp8 V | 3,115,826 | 57.156 | none |
| fp8 K + NVFP4 V | 5,541,156 | 57.019 | none |

Observed allocator-token ratio:

```text
5,541,156 / 3,115,826 = 1.779x
```

Throughput ratio:

```text
57.019 / 57.156 = 0.998x
```

The quality comparator reports:

```json
{
  "ok": true,
  "case": "natural_long_prefill",
  "candidate_flags": [],
  "baseline_flags": []
}
```

## Interpretation

This is the strongest SGLang mixed-KV serving row so far:

- default radix first-token gate: green;
- fresh fp8 capacity comparator: green;
- short/medium decode heuristic quality: green;
- natural long-prefill heuristic quality: green;
- decode speed: fp8 parity;
- graph safety: tested separately and still red/partial because graph-enabled
  `natural_long_prefill` stops after four completion tokens;
- PPL / supplied-token logprob quality: still missing.

Do not use the older repeated-sentence `long_prefill` prompt as the primary SGLang
quality gate. Keep it only as a regression trap for repetition-sensitive behavior.
