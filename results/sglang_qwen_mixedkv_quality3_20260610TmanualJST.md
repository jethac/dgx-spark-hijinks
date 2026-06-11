# SGLang Qwen mixed-KV three-case quality gate, 2026-06-10

## Scope

This is the next quality gate after:

- `results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.md`
- `results/sglang_qwen_fp8_vs_mixedkv_fresh_comparator_20260610TmanualJST.md`

It compares fp8 KV against the current practical SGLang mixed-KV route:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- Page size: `1`
- `--mem-fraction-static`: `0.40`
- Docker cap: `--memory=100g --memory-swap=100g`
- CUDA graph: disabled
- Runs were sequential, not concurrent.

The mixed row uses `KV_CACHE_DTYPE=fp4_e2m1` plus `SGLANG_FP4_KV_MIXED_KV=1`, so K is
FP8 and V is packed NVFP4.

## Artifacts

- fp8 benchmark: `results/sglang_qwen_fp8_quality3_20260610TmanualJST_openai_benchmark.json`
- fp8 manifest: `results/sglang_qwen_fp8_quality3_20260610TmanualJST_row_manifest.json`
- fp8 heuristic quality: `results/sglang_qwen_fp8_quality3_20260610TmanualJST_quality.json`
- mixed benchmark: `results/sglang_qwen_mixedkv_quality3_20260610TmanualJST_openai_benchmark.json`
- mixed manifest: `results/sglang_qwen_mixedkv_quality3_20260610TmanualJST_row_manifest.json`
- mixed heuristic quality: `results/sglang_qwen_mixedkv_quality3_20260610TmanualJST_quality.json`
- quality comparison: `results/sglang_qwen_mixedkv_vs_fp8_quality3_20260610TmanualJST_quality_compare.json`

## Result

Capacity and short decode remain good:

| KV mode | allocatable tokens | K size | V size |
|---|---:|---:|---:|
| fp8 K + fp8 V | 3,117,095 | 20.81 GB | 20.81 GB |
| fp8 K + NVFP4 V | 5,537,880 | 36.97 GB | 20.80 GB |

Observed allocator-token ratio:

```text
5,537,880 / 3,117,095 = 1.777x
```

Throughput:

| case | fp8 tok/s | mixed-KV tok/s | ratio |
|---|---:|---:|---:|
| short_decode | 57.682 | 57.789 | 1.002x |
| medium_decode | 57.521 | 56.660 | 0.985x |
| long_prefill | 57.492 | 56.672 | 0.986x |

Quality gate:

| case | fp8 flags | mixed-KV flags | comparator |
|---|---|---|---|
| short_decode | none | none | pass |
| medium_decode | none | none | pass |
| long_prefill | `dominant_repeated_word` | `low_unique_word_ratio`, `repeated_bigrams`, `repeated_trigrams` | fail |

The mixed long-prefill output is visibly repetitive:

```text
1. Environment, output quality, and reproducibility. The the benchmark must capture environment metadata, backend, output quality, and reproduc. The benchmark must capture environment metadata, backend, output quality, and reproduc. The pressure must, output quality, and reproduc...
```

The fp8 baseline is also imperfect on this heuristic prompt, but the mixed row is worse:

- fp8 long-prefill repeated bigram fraction: `0.0`
- mixed long-prefill repeated bigram fraction: `0.4878048780487805`
- fp8 long-prefill repeated trigram fraction: `0.0`
- mixed long-prefill repeated trigram fraction: `0.4`
- text similarity for long-prefill: `0.1263823064770932`

## Interpretation

Mixed-KV is no longer just a first-token/capacity proof: it handles short and medium
benchmark generations at fp8-like speed without triggering the heuristic quality flags.

It is still **not a blessed SGLang result**. The long-prefill quality gate is red. The
current status is:

- first-token radix-cache regression: green;
- capacity: green;
- short/medium decode quality smoke: green;
- long-prefill quality: red;
- graph safety: untested.

Next work should target the long-prefill degradation before Gemma or graph capture. The
most likely surface is still cached/reused attention state under mixed-KV on longer
prefill, but this artifact does not localize it; it only proves the practical quality gate
is not closed.
