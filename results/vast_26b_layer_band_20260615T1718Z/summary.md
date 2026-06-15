# Gemma-4-26B-A4B NVFP4 KV Layer-Band Calibration Sweep (sm120)

Status: RED for claim-grade full-NVFP4 long-context serving. Layer-band calibration improves the best
long-context point again, but does not reach parity.

## Stack

- Host: Vast.ai RTX PRO 6000 Blackwell Max-Q Workstation Edition, compute capability 12.0, Ubuntu 22.04
  container.
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
- vLLM bf16 baseline: NLL `7.933360410453398`.
- Base full-attention scale: `0.07 / 0.05` for all full layers.
- Sliding bands:
  - early: layers `0 1 2 3 4 6 7 8 9 10`;
  - mid: layers `12 13 14 15 16 18 19 20 21 22`;
  - late: layers `24 25 26 27 28`.
- Vast instance created for this run: `41072927`; destroyed after artifact pull. The unrelated pre-existing
  Vast instance was left untouched.

## Result

| row | base sliding K/V | early | mid | late | mean NLL | delta vs vLLM bf16 | PPL | chat |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| vLLM bf16 | n/a | n/a | n/a | n/a | `7.933360410` | `0` | n/a | n/a |
| best band | `0.07 / 0.05` | `0.10 / 0.08` | `0.10 / 0.08` | `NA` | `7.815396153` | `-0.117964257` | `2478.469` | `Tokyo` |
| layer-type replay | `0.10 / 0.08` | `NA` | `NA` | `NA` | `7.785053609` | `-0.148306801` | `2404.395` | `Tokyo` |
| early+late | `0.07 / 0.05` | `0.10 / 0.08` | `NA` | `0.10 / 0.08` | `7.711891613` | `-0.221468798` | `2234.766` | `Tokyo` |
| early only | `0.07 / 0.05` | `0.10 / 0.08` | `NA` | `NA` | `7.708026323` | `-0.225334087` | `2226.144` | `Tokyo` |

Full tables:

- `dx26_layer_band_20260615T1718Z/summary.tsv`
- `dx26_layer_band_20260615T1718Z/best.tsv`

## Interpretation

- The layer-aware calibration hook is live at individual-layer override granularity: the `all_high` row
  exactly reproduces the prior layer-type best.
- The best tested row improves from the prior layer-type best `-0.148306801` to `-0.117964257` nats/token
  by applying the higher `0.10/0.08` sliding scale only to early+mid sliding bands, leaving late sliding at
  the prior `0.07/0.05`.
- This is still not claim-grade. It remains a low-NLL bias relative to the vLLM bf16 baseline.
- The bias is at least partly depth-localized: mid or late alone are worse, early+mid is best, and
  early-only with K high but V low collapses badly (`-0.949317347`). The next useful path is a narrow
  per-layer or sub-band search inside early+mid sliding layers, especially around whether V high is needed
  uniformly across those bands.
- Current honest long-context ship path for 26B-A4B remains fp8 KV. Full NVFP4 K+V stays research.
