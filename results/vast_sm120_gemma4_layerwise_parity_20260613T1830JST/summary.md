# Vast sm120 Gemma 4 Layerwise fp32-vs-bf16 Parity

Date: 2026-06-13 JST

Scope: pure Hugging Face eager control on a fresh Vast RTX PRO 6000 Workstation instance, intended to distinguish a bf16-only failure from a shared Gemma 4 / Torch 2.12 / CUDA 13 / sm_120 failure mode.

## Environment

- Instance: Vast RTX PRO 6000 Workstation, compute capability 12.0
- Image: `nvidia/cuda:13.0.2-devel-ubuntu22.04`
- Model: `google/gemma-4-12B-it`
- Prompt: `The capital of France is`
- Torch: `2.12.0+cu130`
- CUDA: `13.0`
- Transformers: `5.10.0.dev0` from `effde20942e3f82a1b97449f60b3a48c5ff96145`
- Diagnostic script: `docs/vast_anchor/gemma4_layerwise_parity.py`
- Hook window: first 4 layers, 120 selected modules max

The Vast instance used for this run was destroyed after artifacts were copied back.

## Result

Both fp32 and bf16 are red in the same way:

| dtype | generated text | top next token |
| --- | --- | --- |
| fp32 | `111.111.` | `1` (logit 19.5697) |
| bf16 | `111.1...` | `1` (logit 19.6250) |

The layerwise fp32-vs-bf16 comparison did not find an early divergence:

- `first_bad`: `null`
- 58 comparisons recorded
- lowest non-logit cosine in the first 4 layers: 0.99996966
- final logits cosine: 0.99959373
- final logits max_abs: 1.36977935
- final logits mean_abs: 0.22601420
- logits contained no NaNs or infs in either dtype

## Interpretation

This falsifies a bf16-only explanation for the observed Gemma 4 sm_120 eager failure. Since fp32 and bf16 closely track each other through early layers and still both prefer the same broken numeric-token distribution, the failure is more likely shared by the Gemma 4 / Transformers eager CUDA path on sm_120 than caused by bf16 arithmetic alone.

Claude's separate model-family sweep remains the right discriminator for whether this is Gemma 4-specific, Gemma-family-specific, or a broader Torch/cuBLAS sm_120 issue.

## Artifacts

- `layerwise_parity.log`: raw run log
- `layerwise_parity.json`: metadata, generations, top tokens, activation summaries, fp32-vs-bf16 comparisons
