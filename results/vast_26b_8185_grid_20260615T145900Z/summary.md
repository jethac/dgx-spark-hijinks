# Gemma-4-26B-A4B NVFP4 KV Long-Context 2D Calibration Grid (sm120)

Status: RED for full-NVFP4 long-context serving. No point in the tested 25-point K/V grid reached claim-grade parity at ctx=8185.

## Stack

- Host: Vast.ai RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, Ubuntu 22.04.5.
- Model: `google/gemma-4-26B-A4B-it`.
- vLLM: `0.1.dev1+ge99078ddf.sm120a`, release `sm120a-wheels-e99078ddf-e3`.
- vLLM wheel sha256: `3d92d14d3c6f7f802eb381c6a68f84028978f70ed16055fc046407ca16b4036a`.
- FlashInfer source overlay: `jethac/flashinfer@1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`.
- Runtime env: `VLLM_FLASHINFER_MM_PREFIX=1`, `VLLM_FLASHINFER_VOSPLIT=1`, `VLLM_NVFP4_KV_VOSPLIT=1`, `VLLM_NVFP4_KV_LINEAR_V_SF=1`.

## Method

- Corpus: `wikitext_8k.txt`.
- Context: `ctx=8185`, `prefix=4096`, scored tokens `n=4088`.
- Baselines: HF eager bf16 truth, then vLLM bf16 (`kv_cache_dtype=auto`).
- NVFP4 sweep: same 25-point K/V grid as the ctx=2048 screen, via `VLLM_NVFP4_KV_CALIB` wildcard JSON per row.

## Result

| row | mean NLL | delta vs vLLM bf16 | PPL | chat smoke |
| --- | ---: | ---: | ---: | --- |
| HF eager bf16 truth | 7.992300 | - | 2957.947 | - |
| vLLM bf16 baseline | 7.933360 | 0 | - | - |
| best NVFP4 k=0.07, v=0.05 | 7.545217 | -0.388143 | 1891.674 | Tokyo |
| tied best NVFP4 k=0.07, v=0.10 | 7.545217 | -0.388143 | 1891.674 | Tokyo |
| next NVFP4 k=0.05, v=0.08 | 7.469368 | -0.463993 | 1753.498 | Tokyo |
| next NVFP4 k=0.10, v=0.08 | 7.452548 | -0.480813 | 1724.251 | Tokyo |

Full row table: `dx26_8185_grid/summary.tsv`.

## Interpretation

- The short-context calibration sweet spot does not survive long context.
- The long-context surface improves over the worst tied-scale rows, but the best tested point is still `-0.388143` nats/token vs bf16, which is not claim-grade for full NVFP4 K+V.
- Current honest ship path for 26B-A4B long context remains fp8 KV. Full NVFP4 K+V should stay open as a research item, likely needing a context-/layer-aware calibration or a deeper source of the low-NLL bias beyond single global K/V constants.
