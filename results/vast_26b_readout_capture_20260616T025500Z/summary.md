# 26B-A4B Readout/Layer Capture: `base_k100`

Date: 2026-06-16 UTC

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Hardware: Vast RTX PRO 6000 Blackwell Max-Q Workstation Edition (`sm_120`)
- Host image: reused idle `nvidia/cuda:13.0.1-devel-ubuntu24.04` instance
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source ref: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch/CUDA: `2.12.0+cu130` / CUDA 13.0
- Packet: `docs/vast_anchor/launch_26b_readout_capture_live.sh`
- Context/scoring: `ctx=8185`, `prefix=4096`, `max_model_len=8192`, `prompt_logprobs=20`
- Rows: bf16 baseline plus NVFP4 `base_k100`
- Capture:
  - readout/final hidden: 4 calls per row
  - layer phases: layers 0-4, 25 captured layer-call payloads per row

This is an attribution result, not a support claim.

## Row Status

```text
label	dtype	status
bf16	auto	ok
base_k100	nvfp4	ok
```

## Mean NLL

| row | mean NLL | delta vs bf16 | PPL | missing | readout calls | layer calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 7.933360410 | +0.000000000 | 2788.782530 | 0 | 4 | 25 |
| NVFP4 `base_k100` | 7.815396153 | -0.117964257 | 2478.468612 | 0 | 4 | 25 |

The NLL numbers reproduce the completed top-logprob packet.

## Readout Result

Readout capture compares final hidden rows and raw logits top-k at `compute_logits()`.

| bucket | count | hidden cosine | hidden rel-L2 | logits top-1 match | logits top-k Jaccard | logits LSE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-64 | 256 | 0.967310618 | 0.178761201 | 0.800781250 | 0.727609054 | -0.340375823 |
| 64-256 | 12 | 0.979761556 | 0.138386759 | 0.916666667 | 0.775657427 | +0.137085915 |
| 256-512 | 16 | 0.982776418 | 0.126535415 | 0.937500000 | 0.811665774 | +0.176347911 |
| 512-768 | 16 | 0.984817572 | 0.117799043 | 0.937500000 | 0.782018723 | +0.253583312 |
| 768-1024 | 18 | 0.983264728 | 0.119113134 | 0.944444444 | 0.793066632 | -0.116552989 |
| 1024-end | 98 | 0.951197733 | 0.288501285 | 0.591836735 | 0.511623651 | +0.744791930 |
| all | 416 | 0.965832461 | 0.196514476 | 0.771634615 | 0.686271818 | -0.018559195 |

Verdict: the drift is already present in the final hidden state before lm_head, so the visible top-logprob churn is not only a readout projection artifact. The tail bucket is worst: hidden cosine `0.9512`, rel-L2 `0.2885`, logits top-1 match `0.5918`, and logits top-k Jaccard `0.5116`.

## Layer-Capture Result

The raw `layer_capture_report.tsv` includes many zero/zero rows. For those rows, PyTorch cosine returns `0.0`, which depresses the raw mean cosine even when tensors match exactly. The derived `layer_capture_nonzero_summary.tsv` filters those both-zero rows for cosine/rel-L2 interpretation.

Nonzero summary, `all` bucket:

| layer | phase | nonzero rows | cosine | rel-L2 |
| ---: | --- | ---: | ---: | ---: |
| 0 | input | 256 | 1.000000461 | 0.000000000 |
| 0 | attention_output | 256 | 0.998060661 | 0.060645558 |
| 0 | moe_output | 256 | 0.992550026 | 0.085261208 |
| 0 | output | 256 | 0.999079949 | 0.037239009 |
| 0 | router_logits | 256 | 0.999725295 | 0.023324556 |
| 1 | attention_output | 256 | 0.994595091 | 0.101807841 |
| 1 | moe_output | 256 | 0.993538929 | 0.107493976 |
| 1 | output | 256 | 0.998372013 | 0.053175282 |
| 1 | router_logits | 256 | 0.998709000 | 0.048534179 |
| 2 | attention_output | 256 | 0.992789651 | 0.115578366 |
| 2 | moe_output | 256 | 0.992186355 | 0.115205889 |
| 2 | output | 256 | 0.998798500 | 0.042704466 |
| 2 | router_logits | 256 | 0.999310002 | 0.034927596 |
| 3 | attention_output | 256 | 0.991944402 | 0.121874186 |
| 3 | moe_output | 256 | 0.990279242 | 0.126473372 |
| 3 | output | 256 | 0.998004269 | 0.053168669 |
| 3 | router_logits | 256 | 0.999210441 | 0.035291989 |
| 4 | attention_output | 256 | 0.993190454 | 0.111771240 |
| 4 | moe_output | 256 | 0.986166848 | 0.141605817 |
| 4 | output | 256 | 0.997235950 | 0.062629446 |
| 4 | router_logits | 256 | 0.999563826 | 0.026655196 |

Layer-0 input is identical, as expected. The earliest perturbation appears in layer-0 attention output, then MoE outputs show larger rel-L2 perturbations through layers 0-4. Router logits remain comparatively stable; router top-1/top-k stability in the raw report is generally high, so this does not currently look like a primary expert-routing flip. The drift is present before final readout and appears to accumulate through hidden-state updates.

## Next Branch

The next useful discriminator is no longer "is lm_head/readout solely amplifying a clean hidden state?" It is not. The next branch should either:

- extend layer capture beyond layer 4 to find where final hidden rel-L2 grows from `~0.06` at layer-4 output to `~0.20` at final hidden, or
- compare attention/MoE component perturbations with a per-layer FP8-K or mixed-K reference to isolate whether the accumulated drift is predominantly attention-K, V, or MoE-input amplification.

## Artifacts

- Raw run tree: `dx26_readout_capture_20260616T025500Z/`
- Tarball: `dx26_readout_capture_20260616T025500Z.tgz`
- Console log: `dx26_readout_capture_20260616T025500Z.log`
- Key reports:
  - `dx26_readout_capture_20260616T025500Z/readout_capture_report.tsv`
  - `dx26_readout_capture_20260616T025500Z/layer_capture_report.tsv`
  - `dx26_readout_capture_20260616T025500Z/layer_capture_nonzero_summary.tsv`

The reused Vast instance (`41128851`) was destroyed after pulling this artifact.
