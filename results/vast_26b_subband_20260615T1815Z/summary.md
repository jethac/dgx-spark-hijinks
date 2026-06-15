# Gemma-4-26B-A4B NVFP4 KV Sub-Band Calibration Sweep (sm120)

Status: RED for claim-grade full-NVFP4 long-context serving, but the useful calibration region is now
bracketed.

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
- Base full and late-sliding scale: `0.07 / 0.05`.
- Sub-bands under test:
  - `e0`: layers `0 1 2 3 4`;
  - `e1`: layers `6 7 8 9 10`;
  - `m0`: layers `12 13 14 15 16`;
  - `m1`: layers `18 19 20 21 22`.
- Vast instance created for this run: `41077335`; destroyed after artifact pull. The unrelated pre-existing
  Vast instance was left untouched.

## Result

| row | e0 | e1 | m0 | m1 | mean NLL | delta vs vLLM bf16 | PPL | chat | status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| vLLM bf16 | n/a | n/a | n/a | n/a | `7.933360410` | `0` | n/a | n/a | ok |
| previous best replay | `0.10/0.08` | `0.10/0.08` | `0.10/0.08` | `0.10/0.08` | `7.815396153` | `-0.117964257` | `2478.469` | `Tokyo` | ok |
| drop m1 | `0.10/0.08` | `0.10/0.08` | `0.10/0.08` | base | `7.786669883` | `-0.146690527` | `2408.284` | `Tokyo` | ok |
| all four `0.11/0.08` | `0.11/0.08` | `0.11/0.08` | `0.11/0.08` | `0.11/0.08` | `8.095619315` | `+0.162258904` | `3280.068` | `Tokyo` | ok |
| drop m0 | `0.10/0.08` | `0.10/0.08` | base | `0.10/0.08` | `7.749327748` | `-0.184032662` | `2320.012` | `Tokyo` | ok |
| all four `0.10/0.07` | `0.10/0.07` | `0.10/0.07` | `0.10/0.07` | `0.10/0.07` | n/a | n/a | n/a | n/a | failed: HF Hub processor timeout |
| all four `0.10/0.09` | `0.10/0.09` | `0.10/0.09` | `0.10/0.09` | `0.10/0.09` | `6.063768142` | `-1.869592268` | `429.993` | `Tokyo` | ok |
| all four `0.09/0.08` | `0.09/0.08` | `0.09/0.08` | `0.09/0.08` | `0.09/0.08` | `6.058890154` | `-1.874470257` | `427.900` | `Tokyo` | ok |

Full tables:

- `dx26_subband_20260615T1815Z/summary.tsv`
- `dx26_subband_20260615T1815Z/best.tsv`

## Interpretation

- No block subset beat the previous early+mid all-four-block row. Single blocks, pairs, and drop-one triples
  all remained worse.
- The useful finding is scalar: all four early+mid blocks at `k=0.10,v=0.08` are still low-NLL by
  `-0.117964257`, while all four at `k=0.11,v=0.08` overshoot to `+0.162258904`.
- A simple interpolation puts the parity-crossing near `k=0.104` at `v=0.08` for the early+mid sub-bands,
  with full and late sliding left at `0.07/0.05`.
- `v=0.09` and `k=0.09` are bad in this local surface, so the next bounded discriminator should refine K
  around `0.103-0.106` at fixed `v=0.08`.
- Current honest long-context ship path for 26B-A4B remains fp8 KV until that scalar refinement is green.
