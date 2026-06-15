# 26B-A4B Short Position Attribution

Status: **RED discriminator complete**. Full NVFP4 K+V remains research-only for 26B-A4B long-context; fp8 KV
remains the honest ship path.

## Stack

- Vast instance: `41096054` (`codex-26b-posattr-short-jit4`), destroyed after artifact pull.
- Image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`.
- Device: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, capability `(12, 0)`.
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`.
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`.
- Torch/CUDA: `2.12.0+cu130` / CUDA 13.0.
- Model: `google/gemma-4-26B-A4B-it`.
- Corpus: `wikitext_8k.txt`.
- Context/prefix: `ctx=8185`, `prefix=4096`; scored tokens `4088`.
- Runtime env: `VLLM_FLASHINFER_MM_PREFIX=1`, `VLLM_FLASHINFER_VOSPLIT=1`,
  `VLLM_NVFP4_KV_VOSPLIT=1`, `VLLM_NVFP4_KV_LINEAR_V_SF=1`.
- Build guard: `MAX_JOBS=4` for FlashInfer JIT. The cold bf16 row otherwise OOMs low-RAM hosts during
  `fused_moe_120` compilation.

## Rows

Calibration shape:

- full-attention layers fixed at `k=0.07`, `v=0.05`;
- late sliding layers `[24-28]` fixed at `k=0.07`, `v=0.05`;
- early+mid sliding layers `[0-4]`, `[6-10]`, `[12-16]`, `[18-22]` fixed at `v=0.08`;
- early+mid K tested at `0.100` and `0.103`.

| row | early+mid K | early+mid V | mean NLL | delta vs bf16 | PPL | missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16` | n/a | n/a | `7.933360410` | `0` | `2788.7825` | `0` |
| `k100` | `0.100` | `0.080` | `7.815396153` | `-0.117964257` | `2478.4686` | `0` |
| `k103` | `0.103` | `0.080` | `6.246659276` | `-1.686701135` | `516.2852` | `0` |

Capacity/log proof at the same memory budget:

| row | KV cache tokens | max concurrency @8192 |
| --- | ---: | ---: |
| `bf16` | `183,151` | `22.36x` |
| `nvfp4` | `651,217` | `79.49x` |

This is a raw cache-token ratio of `3.5556x` for full NVFP4 K+V versus bf16/auto at this launch shape. It is
not a quality-green capacity claim.

## Position Buckets

Delta is `row_nll - bf16_nll`; negative means the NVFP4 row assigns higher probability than bf16.

| row | 0-256 | 256-1024 | 1024-2048 | 2048-3072 | 3072-end | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `k100` | `-0.248220337` | `-0.138702851` | `-0.070406208` | `-0.152492993` | `-0.082599317` | `-0.117964257` |
| `k103` | `-2.578748161` | `-2.250232057` | `-1.909711777` | `-1.311380233` | `-1.189466803` | `-1.686701135` |

## Interpretation

The short attribution row validates and sharpens the interrupted K-refinement result:

- `k100` exactly reproduces the prior best row (`-0.117964257`).
- `k103` exactly reproduces the collapse row (`-1.686701135`).
- The collapse is not a single bad token, page, endpoint, or isolated suffix band. It is present across every
  scored bucket, strongest near the start of the scored suffix and decaying but still large through the end.

This falsifies the useful-scalar-crossing path more strongly than the partial K-refinement run alone. The next
full-NVFP4 research step, if pursued, should be layer/activation attribution of the systematic low-NLL shift
or a different calibration model, not another narrow scalar K sweep.

## Files

- `dx26_position_attr_short_20260615T1952Z/RUN_INFO.txt`
- `dx26_position_attr_short_20260615T1952Z/delta_report.tsv`
- `dx26_position_attr_short_20260615T1952Z/rows/{bf16,k100,k103}.json`
- `dx26_position_attr_short_20260615T1952Z/rows/{bf16,k100,k103}.log`
- `dx26_position_attr_short_20260615T1952Z/{k100,k103}_top_deltas.json`
- `docs/vast_anchor/run_26b_position_attribution_short.sh`
