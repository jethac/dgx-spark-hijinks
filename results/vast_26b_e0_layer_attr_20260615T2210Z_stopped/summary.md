# Vast 26B-A4B e0 layer attribution stopped run

Status: stopped by operator request before completion.

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Runtime: vLLM `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch/CUDA: `2.12.0+cu130` / CUDA `13.0`
- Device: RTX PRO 6000 Blackwell Workstation Edition, `sm_120`
- Instance: Vast `41107405`, destroyed after artifact collection.
- Workload: 8k supplied-token logprob, prefix 4096, `max_num_batched_tokens=4096`, prefix cache enabled.
- Purpose: split the previously localized early sliding block `e0 = layers 0-4` into single-layer hot rows.

## Completed Rows

| row | mean NLL | delta vs bf16 | PPL | notes |
|---|---:|---:|---:|---|
| bf16 | 7.933360410 | 0.000000000 | 2788.782530 | reference |
| base_k100 | 7.815396153 | -0.117964257 | 2478.468612 | all early/mid layers at K=0.100,V=0.080; late at K=0.070,V=0.050 |
| e0_all | 6.125525339 | -1.807835071 | 457.384932 | layers 0-4 at K=0.103,V=0.080 |
| l0 | 6.853853891 | -1.079506519 | 947.525540 | only layer 0 hot at K=0.103,V=0.080 |

## Incomplete Rows

Rows `l1` and `l2` were started during shutdown but did not produce JSON result files. Rows `l3` and `l4` were not reached. These rows are unusable.

## Interpretation

This is a partial discriminator only. It shows layer 0 alone accounts for a large fraction of the early-block shift (`-1.0795` nats/token vs `-1.8078` for all layers 0-4 hot), but it does not prove layer 0 is the sole cause because the layer 1-4 rows were not completed.

Next useful continuation, if resumed: rerun `l1` through `l4` from the same packet, preferably skipping already completed rows or using the completed rows here as fixed baselines.

