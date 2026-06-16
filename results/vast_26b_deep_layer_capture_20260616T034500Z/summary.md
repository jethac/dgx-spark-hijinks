# 26B-A4B Deep Layer Capture: `base_k100`

Date: 2026-06-16 UTC

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Hardware: Vast RTX PRO 6000 Blackwell Max-Q Workstation Edition (`sm_120`)
- Container image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source ref: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch/CUDA: `2.12.0+cu130` / CUDA 13.0
- Context/scoring: `ctx=8185`, `prefix=4096`, `max_model_len=8192`, `prompt_logprobs=20`
- Rows: bf16 baseline plus NVFP4 `base_k100`
- Requested capture layers: `0,4,8,12,16,20,24,28,32,36,40`
- Captured layers: `0,4,8,12,16,20,24,28`

This is attribution evidence, not a support claim. Full 26B-A4B NVFP4 K+V remains RED.

## Row Status

```text
label	dtype	status
bf16	auto	ok
base_k100	nvfp4	ok
```

## Mean NLL

| row | mean NLL | delta vs bf16 | PPL | missing | readout calls | layer calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 7.933360410 | +0.000000000 | 2788.782530 | 0 | 4 | 40 |
| NVFP4 `base_k100` | 7.815396153 | -0.117964257 | 2478.468612 | 0 | 4 | 40 |

The NLL numbers reproduce the prior top-logprob/readout rows.

## Readout

The final hidden/readout report reproduces the previous result: the drift is already present before `lm_head`.

| bucket | hidden cosine | hidden rel-L2 | logits top-1 match | logits top-k Jaccard |
| --- | ---: | ---: | ---: | ---: |
| all | 0.965832461 | 0.196514476 | 0.771634615 | 0.686271818 |
| 1024-end | 0.951197733 | 0.288501285 | 0.591836735 | 0.511623651 |

## Deep Layer Trend

The raw `layer_capture_report.tsv` includes zero/zero rows; PyTorch reports `cosine_similarity(0,0)=0`, so
quote `layer_capture_nonzero_summary.tsv` for cosine/rel-L2. Nonzero `all` bucket:

| layer | input rel-L2 | attention rel-L2 | MoE rel-L2 | output rel-L2 | router rel-L2 | router top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000000000 | 0.060645558 | 0.085261208 | 0.037239009 | 0.023324556 | 0.960937500 |
| 4 | 0.053168669 | 0.111771240 | 0.141605817 | 0.062629446 | 0.026655196 | 0.902343750 |
| 8 | 0.097610840 | 0.191567439 | 0.208625158 | 0.120631172 | 0.051521078 | 0.886718750 |
| 12 | 0.154590047 | 0.316604443 | 0.422641794 | 0.162862977 | 0.084291879 | 0.808593750 |
| 16 | 0.208417865 | 0.313635364 | 0.468606157 | 0.204482948 | 0.118675822 | 0.687500000 |
| 20 | 0.323810218 | 0.270760557 | 0.385326693 | 0.297251057 | 0.119151612 | 0.671875000 |
| 24 | 0.368327853 | 0.255562018 | 0.409718721 | 0.385122949 | 0.119406001 | 0.710937500 |
| 28 | 0.400962559 | 0.270923220 | 0.371406624 | 0.380914924 | 0.192313564 | 0.792968750 |

Key readout:

- Layer-0 input is identical, so the first perturbation is still the layer-0 attention read.
- The larger failure is accumulated through the middle stack: layer output rel-L2 grows from `0.037` at layer 0 to `~0.38` by layers 24/28.
- Router logits are not the initial trigger, but by the middle stack they are no longer clean: router top-1 match falls to `0.67-0.71` around layers 16-24 and router rel-L2 reaches `0.19` by layer 28.
- Final hidden rel-L2 (`0.1965`) is lower than the peak mid-stack layer-output rel-L2, so final normalization/readout dampens some accumulated hidden drift rather than creating it from a clean state.

## Interpretation

This run closes the "where does layer-4 rel-L2 become final-hidden rel-L2?" question: the drift is already
large by the middle/deep decoder stack. It is not a readout-only artifact and not a primary layer-0 router
flip. The remaining useful branch is to compare the same deep capture against an FP8-K or mixed-K control to
separate K-driven attention drift from V/downstream MoE amplification.

## Artifacts

- Tarball parts:
  - `dx26_deep_layer_capture_20260616T034500Z.tgz.part00`
  - `dx26_deep_layer_capture_20260616T034500Z.tgz.part01`
  - `dx26_deep_layer_capture_20260616T034500Z.tgz.part02`
- Reconstruct with:
  `python -c "from pathlib import Path; root=Path('.'); out=root/'dx26_deep_layer_capture_20260616T034500Z.tgz'; out.write_bytes(b''.join((root/f'dx26_deep_layer_capture_20260616T034500Z.tgz.part{i:02d}').read_bytes() for i in range(3)))"`
- Raw run tree: inside reconstructed `dx26_deep_layer_capture_20260616T034500Z.tgz`
- Key reports:
  - `readout_capture_report.tsv`
  - `layer_capture_report.tsv`
  - `layer_capture_nonzero_summary.tsv`

The Vast instance (`41134294`) was destroyed after pulling the artifact.
