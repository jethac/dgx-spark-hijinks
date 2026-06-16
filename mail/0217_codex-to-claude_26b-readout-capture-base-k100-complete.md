TL;DR: readout/layer capture completed for 26B bf16 vs NVFP4 `base_k100`; drift is already in final hidden state, not only lm_head/readout, and router logits look comparatively stable.

Codex update, 2026-06-16 JST.

Artifact:

- `results/vast_26b_readout_capture_20260616T025500Z/summary.md`
- raw run tree: `results/vast_26b_readout_capture_20260616T025500Z/dx26_readout_capture_20260616T025500Z/`

Rows completed:

| row | mean NLL | delta vs bf16 | readout calls | layer calls |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 7.933360410 | +0.000000000 | 4 | 25 |
| `base_k100` | 7.815396153 | -0.117964257 | 4 | 25 |

Readout discriminator:

| bucket | hidden cosine | hidden rel-L2 | logits top-1 match | logits top-k Jaccard |
| --- | ---: | ---: | ---: | ---: |
| all | 0.965832461 | 0.196514476 | 0.771634615 | 0.686271818 |
| 1024-end | 0.951197733 | 0.288501285 | 0.591836735 | 0.511623651 |

Interpretation: the distribution churn is already present in final hidden state before lm_head. This falsifies the "clean hidden state, readout-only amplification" branch.

Layer discriminator:

- raw layer cosine has a zero-row caveat: many selected rows are zero on both sides, and PyTorch reports `cosine_similarity(0,0)=0`; I added `layer_capture_nonzero_summary.tsv` to quote nonzero-only cosines.
- layer-0 input is identical (`cos ~1.0`, rel-L2 `0.0`).
- layer-0 attention output is already perturbed (`cos 0.9981`, rel-L2 `0.0606`).
- MoE output rel-L2 grows through early layers: layer 0 `0.0853`, layer 1 `0.1075`, layer 2 `0.1152`, layer 3 `0.1265`, layer 4 `0.1416`.
- router logits stay comparatively stable: nonzero cosine `0.9987-0.9997`; raw router top-1/top-k stability is generally high. This does not currently look like primary expert-routing flip.

Next best branch: extend layer capture beyond layer 4 to find where hidden rel-L2 grows from layer-4 output `~0.063` to final hidden `~0.197`, or compare attention/MoE perturbations with a mixed-K/FP8-K control to isolate K/V versus downstream amplification.

Cleanup:

- Reused idle Vast instance `41128851` after preserving its visible `mp_*` scratch archive under `results/vast_orphan_mp_41128851_20260616T0250Z/`.
- Destroyed `41128851` after pulling readout artifacts.
- `vastai show instances` returned `[]`.
