# SGLang Gemma 4 12B full-NVFP4 scale diagnostic

Date: 2026-06-14 JST

Scope: diagnostic only, not claim-grade. This replays Claude mail 0153's calibrated-global-scale hypothesis on the SGLang packaged Spark image, with radix prefix reuse enabled and CUDA graphs disabled.

## Runtime

- Host: DGX Spark / GB10 `thinkstationpgx-00b4`
- Repo: `dgx-spark-hijinks@1df89b9cbba8c0a4d9577ff3eafdd161eebb9de1`
- Image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
- Model: `google/gemma-4-12B-it`
- Attention: FlashInfer VO-split, page size 1
- KV row: full NVFP4 K+V, `--kv-cache-dtype fp4_e2m1`
- Context: `8185`, reused prefix `4096`, scored continuation tokens `4088`
- Memory guardrail: single server, Docker `--memory 100g`, `--mem-fraction-static 0.72`

## Results

| row | scale knob | NLL | PPL | delta vs bf16 |
| --- | ---: | ---: | ---: | ---: |
| bf16 | n/a | 4.571989822602 | 96.736406679507 | n/a |
| full-NVFP4 | `SGLANG_FP4_KV_GLOBAL_SCALE_MULTIPLIER=0.5` | 6.552172492103 | 700.764927576178 | +1.980182669501 |
| full-NVFP4 | `SGLANG_FP4_KV_GLOBAL_SCALE_MULTIPLIER=2.0` | 4.915603363181 | 136.401584673245 | +0.343613540579 |

The `0.5x` row is worse than the known default/full-NVFP4 red row. The `2.0x` row reproduces the earlier source-overlay plateau at about `+0.3436` nats/token.

## Scale convention

The SGLang knob is a multiplier on its existing per-layer autocalibrated decode-global scale, not a vLLM-style literal `_k_scale`.

Layer-0 trace examples:

- `0.5x`: `k_amax=1.602`, `v_amax=15.88`, `k_gs=0.0002979`, `v_gs=0.002953`
- `2.0x`: `k_amax=1.602`, `v_amax=15.88`, `k_gs=0.001192`, `v_gs=0.01181`

So Claude's vLLM observation that a literal `_k_scale ~= 0.1` is good does not directly transfer to this SGLang multiplier. In SGLang's current convention, reducing the relative global scale is catastrophic, while increasing it only recovers to the known `+0.34` plateau.

## Artifacts

- `results/sglang_gemma4_12b_fullnvfp4_scale05_ctx8185_prefix4096_20260614T222753JST/`
- `results/sglang_gemma4_12b_fullnvfp4_scale2_ctx8185_prefix4096_20260614T223800JST/`

## Interpretation

This does not support "SGLang shares vLLM's uncalibrated literal-1.0 scale bug" as a direct fix path. The next useful SGLang test is either:

1. add an explicit fixed-literal global-scale override in SGLang's FP4 KV pool, separate from the existing multiplier, and sweep that in SGLang's own convention; or
2. compare SGLang's calibrated per-layer scale tensors against vLLM's fixed-scale writer convention at the quantize/dequant boundary before spending more Spark time.

