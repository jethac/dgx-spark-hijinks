# Colab G4 (RTX PRO 6000 Blackwell, sm_120) - Gemma 4 NVFP4 KV serving

Run by Jetha, 2026-06-12, native-Linux Colab G4 runtime. Wheel `sm120a-wheels-512cca4e9`
(vLLM e2-vllm, glibc-gated 22.04 build); FlashInfer source-JIT `spark/hijinks-022-fa2-d512`.
Notebook NARWHAL (live-patched to URIAL-equivalent). C1 quick-PPL ctx-2048, --language-model-only.
FIRST consumer-card (non-Spark, non-P520) serving evidence in the campaign.

## The ladder - all 5 Gemma 4 models GREEN, nvfp4 coherent

| model | bf16 KV tok | nvfp4 KV tok | capacity | bf16 PPL | nvfp4 PPL | dPPL |
|---|---:|---:|---:|---:|---:|---:|
| E2B-it | 5,616,320 | 19,969,138 | 3.556x | 5.9730 | 5.9335 | -0.0396 |
| E4B-it | 1,740,455 | 6,188,289 | 3.556x | 4.4416 | 4.4156 | -0.0259 |
| 12B-it | 286,216 | 1,017,670 | 3.556x | 4.6351 | 5.0270 | +0.3919 |
| 26B-A4B-it | 134,344 | 477,676 | 3.556x | 3.9103 | 4.2331 | +0.3228 |
| 31B-it | 27,238 | 96,849 | 3.556x | 5.3439 | 4.9536 | -0.3903 |

**Capacity = 32/9 = 3.556x on every model** (format-exact, geometry-independent). nvfp4 quality
model-dependent: BETTER on E2B/E4B/31B, WORSE on 12B/26B-A4B (the sensitive pair).

## 1B bisect - sm_120 FlashInfer is CLEAN (the P520 was the outlier)

- FLASH_ATTN truth: 2.358032
- FlashInfer (forced via VLLM_FLASHINFER_BF16_GEMMA=1): 2.357274
- **delta = -0.000757 nats = CLEAN** (P520 was +0.221 / wedge at the identical d256/SWA-512 geometry).
- The `bisect_flashinfer_bf16` row (VLLM_ATTENTION_BACKEND=FLASHINFER) was a FALSE GREEN: vLLM
  silently ran FLASH_ATTN. Only the VLLM_FLASHINFER_BF16_GEMMA=1 row actually engaged FlashInfer.

## Provenance
- 15 green rows, 16 errored (the cold-start gauntlet:
  nvcc / glibc / transformers / torchvision / ninja / flashinfer-data-symlinks / venv, before env fix).
- Raw 31-row record: SUMMARY_NARWHAL.json (this dir). Drive: MyDrive/gemma4_nvfp4_colab_results.