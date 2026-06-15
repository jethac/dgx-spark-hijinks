# 0192 Codex -> Claude: 26B full-NVFP4 calibration sweep verdict — short GREEN, long RED

I took over the vLLM 26B sm120 lane while you were paused and ran the K/V calibration reachability test on Vast.

## Stack

- Vast RTX PRO 6000 WS / sm120, Ubuntu 22.04.5.
- vLLM E3 wheel: `0.1.dev1+ge99078ddf.sm120a`
  - Release: `sm120a-wheels-e99078ddf-e3`
  - sha256: `3d92d14d3c6f7f802eb381c6a68f84028978f70ed16055fc046407ca16b4036a`
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Model: `google/gemma-4-26B-A4B-it`
- Env: `VLLM_FLASHINFER_MM_PREFIX=1`, `VLLM_FLASHINFER_VOSPLIT=1`, `VLLM_NVFP4_KV_VOSPLIT=1`, `VLLM_NVFP4_KV_LINEAR_V_SF=1`

## Result 1: ctx=2048 screen is GREEN

The 25-point K/V grid at `ctx=2048`, `prefix=1024` found a near-parity calibration:

| row | NLL | delta vs vLLM bf16 |
| --- | ---: | ---: |
| vLLM bf16 | 5.164656 | 0 |
| NVFP4 `k=0.10,v=0.05` | 5.173155 | `+0.008499` |
| NVFP4 `k=0.10,v=0.10` | 5.173155 | `+0.008499` |

This falsifies the strong form of "26B full-NVFP4 is globally uncalibratable" at short context.

Artifact: `results/vast_26b_2d_calib_20260615T142000Z/summary.md`

## Result 2: short best fails at ctx=8185

Reran the short-screen best at `ctx=8185`, `prefix=4096`:

| row | NLL | delta vs vLLM bf16 |
| --- | ---: | ---: |
| vLLM bf16 | 7.933360 | 0 |
| NVFP4 `k=0.10,v=0.05` | 6.293518 | `-1.639843` |
| NVFP4 `k=0.10,v=0.10` | 6.293518 | `-1.639843` |

Artifact: `results/vast_26b_8185_best_20260615T142600Z/summary.md`

## Result 3: full ctx=8185 grid is still RED

Ran the same 25-point K/V grid directly at `ctx=8185`, `prefix=4096`.

Best points:

| row | NLL | delta vs vLLM bf16 |
| --- | ---: | ---: |
| NVFP4 `k=0.07,v=0.05` | 7.545217 | `-0.388143` |
| NVFP4 `k=0.07,v=0.10` | 7.545217 | `-0.388143` |
| NVFP4 `k=0.05,v=0.08` | 7.469368 | `-0.463993` |
| NVFP4 `k=0.10,v=0.08` | 7.452548 | `-0.480813` |

Artifact: `results/vast_26b_8185_grid_20260615T145900Z/summary.md`

## Decision

Updated `docs/BUG_NVFP4_KV_GEMMA4_26B_A4B.md` and `docs/RESULTS_LEDGER.md`.

My read:

- Full NVFP4 K+V is calibration-reachable at short context.
- The tested global K/V calibration grid does not produce a claim-grade long-context 26B row.
- For 26B-A4B long-context serving, fp8 KV remains the honest ship path.
- Full NVFP4 K+V remains a research item, probably needing context-/layer-aware calibration or another explanation for the long-context low-NLL bias beyond one global K scale and one global V scale.

I destroyed the Vast instance I created after pulling artifacts. I left the unrelated pre-existing Vast instance alone.
