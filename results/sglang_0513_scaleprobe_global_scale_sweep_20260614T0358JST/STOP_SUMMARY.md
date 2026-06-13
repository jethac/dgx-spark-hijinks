# SGLang 0.5.13 12B Full-NVFP4 Global-Scale Diagnostic

Date: 2026-06-14 JST

Scope: diagnostic source-overlay rows only. These are not packaged-image claim rows because the packaged image digest was run with a mounted SGLang source overlay carrying `jethac/sglang@36c1771eb6`.

## Setup

- Host: DGX Spark / GB10 (`sm_121`)
- Image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-0513-42ce5dad-arm64`
- Image digest: `sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Parent repo: `jethac/dgx-spark-hijinks@5fcf9a2`
- SGLang overlay: `jethac/sglang@36c1771eb6`
- Model: `google/gemma-4-12B-it`
- Row: full NVFP4 K+V (`--kv-cache-dtype fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=0`)
- Context: `ctx=8185`, reused prefix `4096`, scored tokens `4088`
- Graphs: CUDA graphs and piecewise CUDA graphs disabled
- Page size: `1`
- Corpus: reused from the prior 12B matched row to keep the PPL target fixed

## Result

| row | global-scale multiplier | mean NLL | PPL | delta vs bf16 |
| --- | ---: | ---: | ---: | ---: |
| bf16 baseline | n/a | `4.571989822602` | `96.736407` | baseline |
| full-NVFP4 prior | `1x` | `4.974959038640` | `144.742896` | `+0.402969216038` |
| full-NVFP4 source-overlay | `2x` | `4.915603363181` | `136.401585` | `+0.343613540579` |
| full-NVFP4 source-overlay | `4x` | `4.915603363181` | `136.401585` | `+0.343613540579` |

Both diagnostic rows served coherently (`Tokyo` / `Tokyo`) and returned valid prompt-logprob PPL with `cached_tokens=4096`.

## Interpretation

The global-scale multiplier is active and captured in the server logs for all 48 layers. It moves the 12B quality row in the right direction, but it plateaus by `2x` and does not close the `+0.34` nats/token gap. This falsifies "the 12B red is fully fixed by a simple global-scale multiplier" while preserving Claude's saturation finding as a partial contributor.

Next useful checks:

1. Inspect per-block scale saturation before and after `2x` to confirm whether saturation is actually eliminated.
2. If saturation remains, calibrate with a larger explicit headroom formula rather than a fixed multiplier.
3. If saturation is gone at `2x`, chase the remaining error in the SGLang FP4 read/write path, not the scalar global-scale magnitude.

## Artifacts

- `results/sglang_0513_scaleprobe_gemma4_12b_fullnvfp4_x2_ctx8185_prefix4096_20260614T034521JST/`
- `results/sglang_0513_scaleprobe_gemma4_12b_fullnvfp4_x4_ctx8185_prefix4096_20260614T035213JST/`
- Prior bf16 baseline: `results/sglang_gemma4_12b_ar_matched_bf16_fullnvfp4_ctx8185_prefix4096_20260613T153712JST/google-gemma-4-12b-it/bf16_ppl.json`
- Prior full-NVFP4 red: `results/sglang_0513_fix_gemma4_12b_fullnvfp4_only_ctx8185_prefix4096_20260614T024920JST/google-gemma-4-12b-it/fullnvfp4_ppl.json`

## Stop State

- Spark marker absent after run.
- `docker ps` empty after run.
- `free -h` after run: about `114 GiB` available.
