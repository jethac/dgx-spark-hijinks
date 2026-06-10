# SGLang Gemma 3 27B mixed-KV deep-prefix PPL - 2026-06-11

## Verdict

GREEN. At `ctx=8192` with a `4096`-token reused prefix, SGLang Gemma 3 27B text-only mixed KV (`FP8-K + NVFP4-V`) matches the fp8 comparator with graphs disabled.

## Run Scope

- Host: `thinkstationpgx-00b4`
- Model: `google/gemma-3-27b-it`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- SGLang source overlay: `/home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367/third_party/sglang`
- FlashInfer source overlay: `/home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367/third_party/flashinfer`
- KV modes compared sequentially: `fp8_e4m3` vs `fp4_e2m1` with `SGLANG_FP4_KV_MIXED_KV=1`
- Hybrid SWA: `SGLANG_GEMMA3_ENABLE_HYBRID_SWA=1`
- Context: `8192`
- Reused prefix: `4096`
- Logprob start: `4096`
- Page size: `1`
- `--mem-fraction-static`: `0.60`
- Docker memory cap: `100g`
- CUDA graphs: disabled for both servers

## Quality

| ctx | reused prefix | PPL fp8 | PPL mixed | delta PPL | delta nats/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4096 | 4.1805119853 | 4.1769609268 | -0.0035510585 | -0.0008497925 |

Both rows passed the PPL harness gates.

## Capacity

The SGLang hybrid-SWA pool logs allocate separate SWA/local and full/global pools.

| pool | fp8 tokens | mixed tokens | ratio |
| --- | ---: | ---: | ---: |
| SWA/local | 60,355 | 76,870 | 1.2736x |
| full/global | 75,444 | 96,088 | 1.2736x |

This is consistent with the corrected mixed-KV denominator and should be quoted as about `1.27x` for this Gemma 3 launch, not full-NVFP4 `~1.78x`.

## Runtime Evidence

Server logs show the scored requests used real radix prefix reuse:

- fp8 warm prefix: `#new-token: 4096`, `#cached-token: 0`, `cuda graph: False`
- fp8 scored suffix: `#new-token: 4096`, `#cached-token: 4096`, `cuda graph: False`
- mixed warm prefix: `#new-token: 4096`, `#cached-token: 0`, `cuda graph: False`
- mixed scored suffix: `#new-token: 4096`, `#cached-token: 4096`, `cuda graph: False`

## Interpretation

This upgrades the SGLang Gemma 3 Rung 1 mixed-KV evidence from the prior short `ctx=512` checkpoint to a deep-prefix `ctx=8192` quality row. It does not prove CUDA graph re-enable safety; the next gate is a matched graph-enabled run with the prefix-cache graph-write guard active.

## Artifacts

- `sglang_gemma3_27b_mixedkv_ppl_ctx8192_prefix4096_20260611TmanualJST_manifest.json`
- `sglang_gemma3_27b_mixedkv_ppl_ctx8192_prefix4096_20260611TmanualJST_compare.json`
- `sglang_gemma3_27b_mixedkv_ppl_ctx8192_prefix4096_20260611TmanualJST_fp8_ppl.json`
- `sglang_gemma3_27b_mixedkv_ppl_ctx8192_prefix4096_20260611TmanualJST_mixed_ppl.json`
- `sglang_gemma3_27b_mixedkv_ppl_ctx8192_prefix4096_20260611TmanualJST_fp8_server.log`
- `sglang_gemma3_27b_mixedkv_ppl_ctx8192_prefix4096_20260611TmanualJST_mixed_server.log`
