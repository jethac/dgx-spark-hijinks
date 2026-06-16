TL;DR: completed the 26B top-logprob packet for bf16 vs NVFP4 `base_k100`; no support claim, but it shows broad distribution drift and justifies the hidden/readout capture next.

Codex update, 2026-06-16 JST.

Artifact:

- `results/vast_26b_toplogprob_attr_20260616T021036Z/summary.md`
- Raw run tree: `results/vast_26b_toplogprob_attr_20260616T021036Z/dx26_toplogprob_attr_20260616T021036Z/`

Run scope:

- Vast RTX PRO 6000 Blackwell Server Edition (`sm_120`)
- `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- vLLM wheel `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source ref `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- `ctx=8185`, `prefix=4096`, `prompt_logprobs=20`
- rows: bf16 and `base_k100`

Both rows completed:

| row | mean NLL | delta vs bf16 | sampled positions |
| --- | ---: | ---: | ---: |
| bf16 | 7.933360410 | +0.000000000 | 496 |
| `base_k100` | 7.815396153 | -0.117964257 | 496 |

Capacity proof on the same settings:

| row | KV cache tokens | 8192-token concurrency |
| --- | ---: | ---: |
| bf16 / auto KV | 183,468 | 22.40x |
| NVFP4 `base_k100` | 652,335 | 79.63x |

Distribution-level result:

- target NLL delta is negative in every bucket (`-0.248`, `-0.139`, `-0.070`, `-0.152`, `-0.083`)
- top-k Jaccard is only about `0.50-0.55`
- top-1 match ranges about `0.55-0.72`
- top-k mass delta is small, so this looks like broad rank/logit rearrangement rather than simple mass loss

Interpretation: this is not a green quality row. It confirms the earlier NLL bias at the visible distribution level and supports the hidden/readout/layer capture packet as the next useful discriminator. We need to find whether the drift is already in final hidden states, in lm_head projection/readout, or earlier around the first sliding MoE stack/router.

Runtime note: the cold bf16 row spent `1706.09 s` in engine init because it compiled FlashInfer fused-MoE. The NVFP4 row reused that cache but compiled the 512/256 VO-split `vllm_batch_prefill_nvfp4_kv` module, with engine init `262.42 s`.

Cleanup: pulled artifact and destroyed the used Vast instance `41128504`. Vast still shows a separate idle instance `41128851` with different image/port and `mp_*` artifacts; I inspected it read-only and did not destroy it because it does not look like this run.
