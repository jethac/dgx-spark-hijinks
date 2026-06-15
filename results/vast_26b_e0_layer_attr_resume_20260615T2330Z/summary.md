# Vast 26B-A4B e0 layer attribution resume

Status: completed missing `l1-l4` rows and combined with the prior stopped baseline.

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Runtime: vLLM `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch/CUDA: `2.12.0+cu130` / CUDA `13.0`
- Device: RTX PRO 6000 Blackwell Workstation Edition, `sm_120`
- Resume instance: Vast `41113137`, Ubuntu 22.04, destroyed after artifact collection.
- Workload: 8k supplied-token logprob, prefix 4096, `max_num_batched_tokens=4096`, prefix cache enabled.
- Calibration shape: all non-hot early/mid sliding layers at `K=0.100,V=0.080`; late sliding and full layers at `K=0.070,V=0.050`; one layer from `0-4` hot at `K=0.103,V=0.080`.

Baseline rows `bf16`, `base_k100`, `e0_all`, and `l0` come from the stopped artifact:

- `results/vast_26b_e0_layer_attr_20260615T2210Z_stopped/summary.md`

Resume rows `l1-l4` come from this artifact:

- `results/vast_26b_e0_layer_attr_resume_20260615T2330Z/dx26_e0_layer_attr_resume_20260615T2330Z.tgz`

## Combined Rows

| row | hot layer(s) | mean NLL | delta vs bf16 | PPL | missing |
|---|---|---:|---:|---:|---:|
| bf16 | none | 7.933360410 | +0.000000000 | 2788.782530 | 0 |
| base_k100 | none | 7.815396153 | -0.117964257 | 2478.468612 | 0 |
| e0_all | 0-4 | 6.125525339 | -1.807835071 | 457.384932 | 0 |
| l0 | 0 | 6.853853891 | -1.079506519 | 947.525540 | 0 |
| l1 | 1 | 6.806950847 | -1.126409564 | 904.109830 | 0 |
| l2 | 2 | 7.131197863 | -0.802162547 | 1250.373847 | 0 |
| l3 | 3 | 7.444459589 | -0.488900821 | 1710.360745 | 0 |
| l4 | 4 | 7.384541935 | -0.548818476 | 1610.889734 | 0 |

Generated files:

- `combined_summary.tsv`
- `combined_delta_report.tsv`

## Interpretation

The collapse is not a single bad layer. Layers `0` and `1` are the largest individual movers, but every layer in the first sliding block moves the score materially. The all-hot `0-4` row is worse than any single layer, which points to an early-stack interaction or cumulative quantization perturbation rather than a one-layer scalar calibration mistake.

Bucket deltas remain negative across the whole scored suffix for each single-layer hot row, so the effect is still distributed across positions, not one token/page.

This keeps full NVFP4 K+V for 26B-A4B in research/red status. The next useful branch is activation/logit attribution around layers `0-4` or a different calibration model for early sliding layers; another broad scalar K sweep is not justified by this result.

