# Codex -> Claude: 26B e0 layer attribution stopped after l0

Operator requested `codex-autoresearch stop`, so I interrupted the active Vast run, pulled partial artifacts, and destroyed the instance.

Artifact:

- `results/vast_26b_e0_layer_attr_20260615T2210Z_stopped/summary.md`

Completed rows:

| row | mean NLL | delta vs bf16 | notes |
|---|---:|---:|---|
| bf16 | 7.933360410 | 0.000000000 | reference |
| base_k100 | 7.815396153 | -0.117964257 | baseline calibration |
| e0_all | 6.125525339 | -1.807835071 | layers 0-4 hot |
| l0 | 6.853853891 | -1.079506519 | layer 0 hot only |

Rows `l1`/`l2` only have startup fragments from the interrupted shutdown; `l3`/`l4` were not reached.

Partial read: layer 0 alone is a major contributor, but not the whole e0 effect. The e0 split still needs `l1`-`l4` before we can call the layer attribution closed.

