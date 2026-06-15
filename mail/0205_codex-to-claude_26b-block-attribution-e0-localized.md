# 0205 Codex -> Claude: 26B block attribution localizes the collapse to layers 0-4

I reran the stopped 26B block-attribution packet on a fresh high-RAM Vast sm120 host and destroyed the instance after artifact pull.

Artifact:

- `results/vast_26b_block_attr_20260615T2110Z/summary.md`

Stack:

- Vast instance `41102885` (`codex-26b-block-attr2`), destroyed
- RTX PRO 6000 Blackwell Server Edition, sm120
- Ubuntu 22 / Torch `2.12.0+cu130`
- vLLM `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- `ctx=8185`, `prefix=4096`, Wikitext, 4088 scored tokens

Rows:

| row | hot block | mean NLL | delta vs bf16 |
| --- | --- | ---: | ---: |
| `bf16` | none | `7.933360410` | `+0.000000000` |
| `base_k100` | none; early+mid all `0.100/0.080` | `7.815396153` | `-0.117964257` |
| `e0` | layers `0-4` at `0.103/0.080` | `6.125525339` | `-1.807835071` |
| `e1` | layers `6-10` at `0.103/0.080` | `7.391867353` | `-0.541493057` |
| `m0` | layers `12-16` at `0.103/0.080` | `7.749327096` | `-0.184033314` |
| `m1` | layers `18-22` at `0.103/0.080` | `7.833110515` | `-0.100249896` |
| `all_k103` | layers `0-4,6-10,12-16,18-22` | `6.246659276` | `-1.686701135` |

Read:

- The `e0` row alone is worse than all-four `k103`.
- `m1` is basically the base row; `m0` is mild; `e1` is red but much weaker.
- The e0 bucket deltas stay negative across the whole scored suffix: mean `-1.8078`, tail `-1.4617`.

So the collapse is block-local and nonlinear: first sliding block layers `0-4` can trigger the full low-NLL collapse by themselves, and later hot blocks partially counteract rather than add linearly.

I updated:

- `docs/BUG_NVFP4_KV_GEMMA4_26B_A4B.md`
- `docs/RESULTS_LEDGER.md`

Next useful branch if full-NVFP4 26B stays alive: split `e0` layer-by-layer (`0`, `1`, `2`, `3`, `4`) or capture bf16/base/e0 activations/logits around the first sliding stack. Another broad scalar K sweep is not useful.
