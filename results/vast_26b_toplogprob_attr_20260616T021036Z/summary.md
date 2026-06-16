# 26B-A4B Top-Logprob Attribution: `base_k100`

Date: 2026-06-16 UTC

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Hardware: Vast RTX PRO 6000 Blackwell Server Edition (`sm_120`)
- Host image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source ref: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch/CUDA: `2.12.0+cu130` / CUDA 13.0
- Packet: `docs/vast_anchor/launch_26b_toplogprob_live.sh`
- Context/scoring: `ctx=8185`, `prefix=4096`, `max_model_len=8192`, `prompt_logprobs=20`
- Rows: bf16 baseline plus NVFP4 `base_k100`

This is an attribution result, not a support claim.

## Row Status

```text
label	dtype	status
bf16	auto	ok
base_k100	nvfp4	ok
```

## Capacity Proof Lines

| row | KV cache tokens | 8192-token concurrency |
| --- | ---: | ---: |
| bf16 / auto KV | 183,468 | 22.40x |
| NVFP4 `base_k100` | 652,335 | 79.63x |

This is a bf16-vs-NVFP4 capacity comparison on the same run settings, not a final fp8-vs-NVFP4 serving ratio.

## Mean NLL

| row | mean NLL | delta vs bf16 | PPL | sampled positions |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 7.933360410 | +0.000000000 | 2788.782530 | 496 |
| NVFP4 `base_k100` | 7.815396153 | -0.117964257 | 2478.468612 | 496 |

The NVFP4 row lowers target-token NLL relative to bf16 on this document. That is not evidence of correctness; it is the same broad calibration bias seen in the earlier layer-band runs.

## Top-Logprob Distribution Drift

| bucket | count | target delta | top-k Jaccard | top-1 match | top-1 logprob delta | top-k mass delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-256 | 256 | -0.248220337 | 0.541004748 | 0.691406250 | +0.038936022 | +0.003816054 |
| 256-1024 | 768 | -0.138702851 | 0.507458563 | 0.708333333 | -0.084075203 | +0.003224819 |
| 1024-2048 | 1024 | -0.070406208 | 0.507187984 | 0.546875000 | -0.082826898 | -0.004469945 |
| 2048-3072 | 1024 | -0.152492993 | 0.546192486 | 0.718750000 | +0.010573704 | -0.004888641 |
| 3072-end | 1016 | -0.082599317 | 0.503665190 | 0.671875000 | -0.070071619 | -0.001355171 |

The shift is broad across the scored suffix: target-token NLL deltas are negative in every bucket, top-k overlap is only about 0.50-0.55, and top-1 match is materially below 1.0 throughout. This argues against a single bad token/page and supports the existing next branch: capture hidden/readout/layer phases to find where the distribution drift first appears.

## Runtime Notes

- First bf16 cold start spent most of its time compiling FlashInfer fused-MoE kernels: engine init took `1706.09 s`.
- NVFP4 reused the fused-MoE cache but compiled the `vllm_batch_prefill_nvfp4_kv` 512/256 VO-split module: engine init took `262.42 s`.
- NVFP4 proof lines show linear V scale factors and the FA2 VO split:
  - `VLLM_NVFP4_KV_LINEAR_V_SF=1`
  - `V-scale-factor mode: linear, in-kernel deswizzle disabled`
  - `head_size 512 runs as 2 passes of head_dim_vo=256`

## Artifacts

- Raw run tree: `dx26_toplogprob_attr_20260616T021036Z/`
- Tarball: `dx26_toplogprob_attr_20260616T021036Z.tgz`
- Console log: `dx26_toplogprob_attr_20260616T021036Z.log`
- Key tables:
  - `dx26_toplogprob_attr_20260616T021036Z/summary.tsv`
  - `dx26_toplogprob_attr_20260616T021036Z/toplogprob_delta_report.tsv`

The used Vast instance (`41128504`) was destroyed after artifact pull. A separate unrelated idle Vast instance (`41128851`) was observed but not destroyed because it contained different `mp_*` artifacts and did not match this run's image, port, or artifact root.
