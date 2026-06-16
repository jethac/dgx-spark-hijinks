# 26B-A4B Active-KV Mix Attribution: Sliding + Global Result

Date: 2026-06-16 UTC

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Hardware: Vast RTX PRO 6000 Blackwell Workstation Edition (`sm_120`)
- Container image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- vLLM wheel: `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source ref: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Context/scoring: `ctx=8185`, `prefix=4096`, `max_model_len=8192`, `max_num_batched_tokens=4096`
- Capture: FlashInfer prefill calls with `qo_len=4096`, first 8 calls per row
- Rows: bf16/auto KV and NVFP4 `base_k100`

This is attribution evidence, not a support claim. Full 26B-A4B NVFP4 K+V remains RED/open.

## Row Status

```text
label	dtype	status
bf16	auto	ok
base_k100	nvfp4	ok
```

## Mean NLL

| row | mean NLL | delta vs bf16 | PPL | active-KV calls |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 7.933360410 | +0.000000000 | 2788.782530 | 8 |
| NVFP4 `base_k100` | 7.815396153 | -0.117964257 | 2478.468612 | 8 |

This reproduces the prior top-logprob/readout/deep-layer NLL.

## Capacity Proof Lines

| row | KV cache tokens | max concurrency @ 8192 |
| --- | ---: | ---: |
| bf16 / auto KV | 183,556 | 22.41x |
| NVFP4 `base_k100` | 652,652 | 79.67x |

NVFP4 log proof:

- `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x`
- `V-scale-factor mode: linear, in-kernel deswizzle disabled`
- `FA2 VO split (nvfp4 KV): head_size 512 runs as 2 passes of head_dim_vo=256`

## Mixed K/V Reference Result

The comparator reconstructs FlashInfer attention from active pages and compares:

- bf16 K + bf16 V baseline
- NVFP4 K + NVFP4 V
- bf16 K + NVFP4 V
- NVFP4 K + bf16 V

All eight captured prefill calls produced comparator rows.

| call | q shape | KV tokens | window | NVFP4 K+V rel-L2 | bf16 K + NVFP4 V rel-L2 | NVFP4 K + bf16 V rel-L2 | dominant |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `(4096, 16, 256)` | 4096 | 1023 | 0.099016505 | 0.072167768 | 0.067827347 | mixed |
| 1 | `(4096, 16, 256)` | 4096 | 1023 | 0.106627960 | 0.076131275 | 0.075108312 | mixed |
| 2 | `(4096, 16, 256)` | 4096 | 1023 | 0.143606733 | 0.091123269 | 0.111562698 | mixed |
| 3 | `(4096, 16, 256)` | 4096 | 1023 | 0.128243660 | 0.083340976 | 0.100394375 | mixed |
| 4 | `(4096, 16, 256)` | 4096 | 1023 | 0.141596498 | 0.098104365 | 0.123338749 | K |
| 5 | `(4096, 16, 512)` | 4096 | -1 | 0.117013252 | 0.102539944 | 0.072383692 | V |
| 6 | `(4096, 16, 512)` | 4096 | -1 | 0.113974661 | 0.098553602 | 0.070764886 | V |
| 7 | `(4096, 16, 256)` | 4096 | 1023 | 0.159155129 | 0.403632046 | 0.259386900 | V |

Interpretation:

- The first sliding-window calls are not a clean K-only story. Calls 0-3 split relatively evenly; call 4 is K-dominant.
- The global D512 calls remain V-dominant, reproducing the previous partial run.
- The late sliding call 7 is strongly V-dominant, and the one-sided substitutions have larger rel-L2 than the all-NVFP4 row. That points to nonlinear interaction between K/V substitutions and the softmax output, not a simple additive error budget.

This result argues against "FP8-K alone is obviously the only lever" as a complete explanation. It also does not prove V is the final root cause: this is local attention-output attribution at eight prefill calls, while the model-level failure is a distributed low-NLL distribution drift.

## Caveats

- `cosine` values in the TSV can exceed 1 by small amounts because the comparator reports torch bf16/float accumulation residuals around nearly identical vectors; use rel-L2 for dominance.
- The raw active-page tensor tree was about 1.9 GB and was intentionally not retained in git. The committed artifact keeps derived reports, row JSON, logs, and rerunnable scripts.
- The Vast instance (`41143506`) was destroyed after pulling the derived reports; `vastai show instances` reported no active instances.

## Artifacts

- `RUN_INFO.txt`
- `row_status.tsv`
- `summary.tsv`
- `active_kv_mix_report.tsv`
- `rows/bf16.json`
- `rows/base_k100.json`
- `rows/bf16.log`
- `rows/base_k100.log`
- `calib/base_k100.json`
