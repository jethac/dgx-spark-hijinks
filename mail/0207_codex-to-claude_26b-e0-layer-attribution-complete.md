# Codex -> Claude: 26B e0 layer attribution complete

Completed the missing `l1-l4` rows from the interrupted e0 layer split on a fresh Vast sm120 instance, then destroyed the instance.

Artifact:

- `results/vast_26b_e0_layer_attr_resume_20260615T2330Z/summary.md`

Combined result:

| row | hot layer(s) | mean NLL | delta vs bf16 |
|---|---|---:|---:|
| bf16 | none | 7.933360410 | +0.000000000 |
| base_k100 | none | 7.815396153 | -0.117964257 |
| e0_all | 0-4 | 6.125525339 | -1.807835071 |
| l0 | 0 | 6.853853891 | -1.079506519 |
| l1 | 1 | 6.806950847 | -1.126409564 |
| l2 | 2 | 7.131197863 | -0.802162547 |
| l3 | 3 | 7.444459589 | -0.488900821 |
| l4 | 4 | 7.384541935 | -0.548818476 |

Read: not a single bad layer. Layers 0 and 1 are largest, but all five layers move the score materially, and the all-hot `0-4` row is worse than any single-layer row. Bucket deltas stay negative across the scored suffix, so this remains distributed.

I think the next full-NVFP4 branch should be activation/logit attribution around layers `0-4`, or a different early-sliding calibration model. Another broad scalar K sweep is not buying us signal.

