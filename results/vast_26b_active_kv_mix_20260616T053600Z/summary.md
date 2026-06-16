# 26B-A4B Active-KV Mix Attribution: Partial Global-Layer Result

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

This is attribution evidence, not a support claim. Full 26B-A4B NVFP4 K+V remains RED.

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
| bf16 / auto KV | 183,455 | 22.39x |
| NVFP4 `base_k100` | 652,291 | 79.63x |

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

Only calls 5 and 6 produced usable comparator rows. They are global `D=512` full-attention calls:

| call | q shape | KV tokens | NVFP4 K+V rel-L2 | bf16 K + NVFP4 V rel-L2 | NVFP4 K + bf16 V rel-L2 | dominant |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 5 | `(4096, 16, 512)` | 4096 | 0.117013252 | 0.102539937 | 0.072383685 | V |
| 6 | `(4096, 16, 512)` | 4096 | 0.113974661 | 0.098553602 | 0.070764879 | V |

Interpretation for these global calls: replacing V with NVFP4 while keeping bf16 K explains more local
attention-output drift than replacing K with NVFP4 while keeping bf16 V. This is not enough to explain the
whole 26B failure, because calls 0-4 and 7 were skipped by the first comparator version.

## Caveats

- Sliding/SWA calls 0-4 and 7 were captured but skipped with `could not locate bf16 active K/V tensors`.
  The first comparator only handled the global bf16 active-page layout. The scripts have since been hardened
  to search packed bf16 layouts more generically; rerun this packet to get the sliding-layer answer.
- The raw active-page tensor tarball was about 2 GB and was intentionally not retained in git. The committed
  artifact keeps derived reports, row JSON, logs, and rerunnable scripts.
- Two earlier same-instance attempts were discarded:
  - `CAPTURE_QO=512` captured zero calls.
  - `CAPTURE_QO=4096` captured calls but skipped the large `q` tensor; the final run raised the cap and
    produced the global-layer comparator rows above.

## Artifacts

- `RUN_INFO.txt`
- `row_status.tsv`
- `summary.tsv`
- `active_kv_mix_report.tsv`
- `rows/bf16.json`
- `rows/base_k100.json`
- `rows/bf16.log`
- `rows/base_k100.log`

The Vast instance (`41139265`) was destroyed after pulling the derived reports; `vastai show instances`
reported no active instances.
