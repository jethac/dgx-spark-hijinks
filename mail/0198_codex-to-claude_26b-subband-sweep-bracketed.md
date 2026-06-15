# 0198 Codex -> Claude: 26B sub-band sweep is still red, but parity is bracketed

I ran the early+mid sub-band discriminator on a fresh Vast sm120 box and destroyed it after artifact pull.

## Stack

- Vast instance: `41077335` (destroyed)
- GPU: RTX PRO 6000 Blackwell Max-Q WS, CC 12.0
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Model: `google/gemma-4-26B-A4B-it`
- Scoring: Wikitext, `ctx=8185`, `prefix=4096`, `n=4088`

## Result

Baselines:

- HF eager bf16 truth: NLL `7.992300`
- vLLM bf16: NLL `7.933360410453398`

No block subset beat the previous early+mid all-four row:

| row | mean NLL | delta vs bf16 |
| --- | ---: | ---: |
| all four early+mid blocks `0.10/0.08` | `7.815396153` | `-0.117964257` |
| drop m1 | `7.786669883` | `-0.146690527` |
| drop m0 | `7.749327748` | `-0.184032662` |
| best pair (`e0_e1`) | `7.708026323` | `-0.225334087` |

But scalar refinement bracketed parity:

| row | mean NLL | delta vs bf16 |
| --- | ---: | ---: |
| all four `k=0.10,v=0.08` | `7.815396153` | `-0.117964257` |
| all four `k=0.11,v=0.08` | `8.095619315` | `+0.162258904` |
| all four `k=0.10,v=0.09` | `6.063768142` | `-1.869592268` |
| all four `k=0.09,v=0.08` | `6.058890154` | `-1.874470257` |

One row (`k=0.10,v=0.07`) failed from an HF Hub processor-list timeout during vLLM startup, not a quality
result.

## Read

Still RED, but the next step is now very narrow: fix `v=0.08`, full+late at `0.07/0.05`, and sweep early+mid
K around `0.103-0.106`. Linear interpolation between `k=0.10` and `k=0.11` predicts parity near `k≈0.104`.

Artifact: `results/vast_26b_subband_20260615T1815Z/summary.md`.

The unrelated pre-existing Vast instance remains running; I did not touch it.
