# Gemma-4-26B-A4B NVFP4 KV Layer-Type Calibration Sweep (sm120)

Status: RED for claim-grade full-NVFP4 long-context serving. Layer-type calibration improves the best
long-context point, but does not reach parity.

## Stack

- Host: Vast.ai RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, Ubuntu 22.04 container.
- Model: `google/gemma-4-26B-A4B-it`.
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`.
- vLLM release: `sm120a-wheels-1c9686c61-layercalib`.
- vLLM wheel sha256: `256f4df758c463d3c9004b4778d72e078e777451090e1282fa1980490b091975`.
- FlashInfer source overlay: `jethac/flashinfer@1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`.
- Runtime env: `VLLM_FLASHINFER_MM_PREFIX=1`, `VLLM_FLASHINFER_VOSPLIT=1`,
  `VLLM_NVFP4_KV_VOSPLIT=1`, `VLLM_NVFP4_KV_LINEAR_V_SF=1`.

## Method

- Corpus: `wikitext_8k.txt`.
- Context: `ctx=8185`, `prefix=4096`, scored tokens `n=4088`.
- HF eager bf16 truth: NLL `7.992300`, PPL `2957.9474`.
- vLLM bf16 baseline: NLL `7.933360410453398`, PPL `2788.7825`.
- NVFP4 rows: `VLLM_NVFP4_KV_CALIB` JSON with separate:
  - `layer_type_scales.sliding_attention`;
  - `layer_type_scales.full_attention`.
- Vast instance created for this run: `41069113`; destroyed after artifact pull. The unrelated pre-existing
  Vast instance was left untouched.

## Result

| row | sliding K/V | full K/V | mean NLL | delta vs vLLM bf16 | PPL | chat |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vLLM bf16 | n/a | n/a | `7.933360410` | `0` | `2788.783` | `Tokyo` |
| best NVFP4 | `0.10 / 0.08` | `0.07 / 0.05` | `7.785053609` | `-0.148306801` | `2404.395` | `Tokyo` |
| second | `0.05 / 0.08` | `0.07 / 0.05` | `7.661938158` | `-0.271422252` | `2125.874` | `Tokyo` |
| third | `0.07 / 0.05` | `0.05 / 0.05` | `7.599137049` | `-0.334223361` | `1996.472` | `Tokyo` |
| prior global best reproduced | `0.07 / 0.05` | `0.07 / 0.05` | `7.545217218` | `-0.388143193` | `1891.674` | `Tokyo` |

Full tables:

- `dx26_layer_type_20260615T1545Z/summary.tsv`
- `dx26_layer_type_20260615T1545Z/best.tsv`

## Interpretation

- The new layer-type calibration hook is live and useful: the best tested long-context delta improves from
  the prior global-grid best `-0.388143` to `-0.148307` nats/token.
- The row is still not claim-grade. It remains a low-NLL bias relative to the vLLM bf16 baseline.
- The result supports continuing toward finer calibration, likely layer-index or layer-band calibration
  around the promising region, rather than declaring 26B full-NVFP4 inherently impossible.
- Current honest long-context ship path for 26B-A4B remains fp8 KV. Full NVFP4 K+V stays research.
