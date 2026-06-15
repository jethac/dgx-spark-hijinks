# 26B-A4B Full-NVFP4 Block Attribution

Date: 2026-06-15 UTC / 2026-06-16 JST

Status: RED discriminator complete. This is not a serving/capacity claim.

## Stack

- Vast instance: `41102885` (`codex-26b-block-attr2`), destroyed after artifact pull
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, capability `(12, 0)`
- Image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- Model: `google/gemma-4-26B-A4B-it`
- vLLM: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch/CUDA: `2.12.0+cu130` / `13.0`
- Scoring: Wikitext, `ctx=8185`, `prefix=4096`, 4088 scored tokens
- Runtime knobs: `VLLM_FLASHINFER_VOSPLIT=1`, `VLLM_NVFP4_KV_VOSPLIT=1`, `VLLM_NVFP4_KV_LINEAR_V_SF=1`, `MAX_JOBS=4`

## Question

The previous K-refinement and short attribution rows showed:

- base early+mid sliding blocks at `K=0.100,V=0.080`: mild low-NLL bias (`-0.117964257` nats/token)
- all early+mid sliding blocks at `K=0.103,V=0.080`: strong low-NLL collapse (`-1.686701135` nats/token)

This packet held late sliding plus full layers at `0.07/0.05`, held the non-tested early+mid blocks at `0.100/0.080`, and moved one five-layer early/mid block at a time to `K=0.103,V=0.080`:

- `e0`: layers `0-4`
- `e1`: layers `6-10`
- `m0`: layers `12-16`
- `m1`: layers `18-22`

## Results

| row | hot block | mean NLL | delta vs bf16 | verdict |
| --- | --- | ---: | ---: | --- |
| `bf16` | none | `7.933360410` | `+0.000000000` | baseline |
| `base_k100` | none; early+mid all `0.100/0.080` | `7.815396153` | `-0.117964257` | prior best replay |
| `e0` | layers `0-4` at `0.103/0.080` | `6.125525339` | `-1.807835071` | collapse |
| `e1` | layers `6-10` at `0.103/0.080` | `7.391867353` | `-0.541493057` | red |
| `m0` | layers `12-16` at `0.103/0.080` | `7.749327096` | `-0.184033314` | mild red |
| `m1` | layers `18-22` at `0.103/0.080` | `7.833110515` | `-0.100249896` | near base |
| `all_k103` | layers `0-4,6-10,12-16,18-22` at `0.103/0.080` | `6.246659276` | `-1.686701135` | collapse replay |

The key result is that `e0` alone is worse than the all-four `k103` row. The collapse is not a smooth scalar whole-model effect and not caused by the later mid block. It is primarily triggered by the first sliding block, layers `0-4`, with nonlinear interaction from the later hot blocks.

Bucket deltas:

| row | mean delta | 0-256 | 256-1024 | 1024-2048 | 2048-3072 | tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_k100` | `-0.117964257` | `-0.248220337` | `-0.138702851` | `-0.070406208` | `-0.152492993` | `-0.082599317` |
| `e0` | `-1.807835071` | `-2.341190747` | `-1.777319082` | `-1.985281009` | `-1.863409513` | `-1.461658259` |
| `e1` | `-0.541493057` | `-0.637697378` | `-0.717392244` | `-0.623105268` | `-0.445108861` | `-0.399177734` |
| `m0` | `-0.184033314` | `-0.345130379` | `-0.179241487` | `-0.086386086` | `-0.232172340` | `-0.196962128` |
| `m1` | `-0.100249896` | `-0.229661335` | `-0.207568162` | `-0.054830241` | `-0.088640559` | `-0.043997859` |
| `all_k103` | `-1.686701135` | `-2.578748161` | `-2.250232057` | `-1.909711777` | `-1.311380233` | `-1.189466803` |

The `e0` collapse is distributed across the scored suffix, not a single token/page. It remains strong in the tail (`-1.461658259`), which is different from a one-position artifact.

## Interpretation

This localizes the sharp `k=0.103` failure to the first five sliding layers. Later blocks are not sufficient:

- `e1` is red but far weaker than `e0`.
- `m0` is only mildly worse than base.
- `m1` is essentially the base row.

The all-four `k103` collapse is therefore not an additive sum of small block errors. It is dominated by the first sliding block and partially counteracted by making all later early/mid blocks hot too.

The next useful discriminator is a layer-level or activation-level split inside `e0`, not another whole-model K sweep:

- split layers `0`, `1`, `2`, `3`, `4` individually at `K=0.103,V=0.080`;
- if one layer dominates, capture bf16/base/e0 attention and lm-head logits around that layer;
- if all five contribute, treat the first sliding stack as the calibration boundary and investigate a layer-depth-dependent policy.

Until that succeeds, the claim-grade 26B-A4B path remains fp8 KV. Full NVFP4 K+V remains research.

## Files

- `dx26_block_attr_20260615T2110Z/RUN_INFO.txt`
- `dx26_block_attr_20260615T2110Z/summary.tsv`
- `dx26_block_attr_20260615T2110Z/delta_report.tsv`
- `dx26_block_attr_20260615T2110Z/rows/*.json`
- `dx26_block_attr_20260615T2110Z/*_top_deltas.json`
