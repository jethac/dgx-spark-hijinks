# Gemma-4-26B-A4B NVFP4 KV 2D Calibration Sweep (sm120)

Status: ctx=2048 screen complete. This falsifies the prior "full NVFP4 26B is uncalibratable" hypothesis at short context: k=0.10 with v=0.05 or v=0.10 is within +0.008499 nats/token of the vLLM bf16 baseline.

## Stack

- Host: Vast.ai RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, Ubuntu 22.04.5.
- Model: `google/gemma-4-26B-A4B-it`.
- vLLM: `0.1.dev1+ge99078ddf.sm120a`, release `sm120a-wheels-e99078ddf-e3`.
- vLLM wheel sha256: `3d92d14d3c6f7f802eb381c6a68f84028978f70ed16055fc046407ca16b4036a`.
- FlashInfer source overlay: `jethac/flashinfer@1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`.
- Runtime env: `VLLM_FLASHINFER_MM_PREFIX=1`, `VLLM_FLASHINFER_VOSPLIT=1`, `VLLM_NVFP4_KV_VOSPLIT=1`, `VLLM_NVFP4_KV_LINEAR_V_SF=1`.

## Method

- Corpus: `wikitext_8k.txt`, same campaign harness corpus.
- Screen context: `ctx=2048`, `prefix=1024`, scored tokens `n=1023`.
- Baselines: HF eager bf16 truth, then vLLM bf16 (`kv_cache_dtype=auto`).
- NVFP4 sweep: vLLM `--kv-cache-dtype nvfp4` with `VLLM_NVFP4_KV_CALIB` wildcard JSON per row.

## Result

| row | mean NLL | delta vs vLLM bf16 | PPL | chat smoke |
| --- | ---: | ---: | ---: | --- |
| HF eager bf16 truth | 4.979100 | - | 145.337 | - |
| vLLM bf16 baseline | 5.164656 | 0 | - | - |
| NVFP4 k=0.10, v=0.05 | 5.173155 | +0.008499 | 176.471 | Tokyo |
| NVFP4 k=0.10, v=0.10 | 5.173155 | +0.008499 | 176.471 | Tokyo |
| NVFP4 k=0.05, v=0.05 | 5.287458 | +0.122802 | 197.840 | Tokyo |
| NVFP4 k=0.20, v=0.05 | 4.930381 | -0.234275 | 138.432 | Tokyo |

Full row table: `dx26_2d/summary.tsv`.

## Interpretation

- Full NVFP4 K+V on 26B-A4B is calibration-reachable at ctx=2048 on sm120; the prior tied-scale failures were not enough to prove a structural 26B limitation.
- The best short-screen candidate is `k_scale=0.10`, `v_scale=0.05` (tied `v=0.10` is numerically identical in this screen).
- This is not yet a claim-grade serving row. The next gate is long-context validation at `ctx=8185`, `prefix=4096`, followed by prefix/cache-reuse evidence if the long row stays green.
