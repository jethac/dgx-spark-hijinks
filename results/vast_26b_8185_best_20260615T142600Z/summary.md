# Gemma-4-26B-A4B NVFP4 KV Long-Context Check: Short-Screen Best

Status: RED for long-context claim-grade use. The short-screen best calibration does not generalize to ctx=8185.

## Stack

- Host: Vast.ai RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, Ubuntu 22.04.5.
- Model: `google/gemma-4-26B-A4B-it`.
- vLLM: `0.1.dev1+ge99078ddf.sm120a`, release `sm120a-wheels-e99078ddf-e3`.
- vLLM wheel sha256: `3d92d14d3c6f7f802eb381c6a68f84028978f70ed16055fc046407ca16b4036a`.
- FlashInfer source overlay: `jethac/flashinfer@1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`.

## Method

- Corpus: `wikitext_8k.txt`.
- Context: `ctx=8185`, `prefix=4096`, scored tokens `n=4088`.
- Candidates: the two best ctx=2048 calibrations, `k=0.10/v=0.05` and `k=0.10/v=0.10`.

## Result

| row | mean NLL | delta vs vLLM bf16 | PPL | chat smoke |
| --- | ---: | ---: | ---: | --- |
| HF eager bf16 truth | 7.992300 | - | 2957.947 | - |
| vLLM bf16 baseline | 7.933360 | 0 | - | - |
| NVFP4 k=0.10, v=0.05 | 6.293518 | -1.639843 | 541.053 | Tokyo |
| NVFP4 k=0.10, v=0.10 | 6.293518 | -1.639843 | 541.053 | Tokyo |

## Interpretation

The ctx=2048 near-parity row is not enough for a 26B full-NVFP4 claim. At ctx=8185 the same calibration produces the same low-NLL collapse as earlier tied-scale rows. Full long-context calibration requires a direct long-context grid.
