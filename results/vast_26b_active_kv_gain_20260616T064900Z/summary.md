# 26B-A4B Active-KV Gain Probe: Output Gain Is Not the Fix

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

## Why This Probe Exists

The previous active-KV mix run showed non-additive K/V behavior, especially call 7 where one-sided K/V
substitutions were worse than the all-NVFP4 row. This run adds two diagnostics to the local attention-output
comparison:

- best scalar output gain, fitting `alpha * out_nvfp4 ~= out_bf16`;
- best per-query-head output gain.

If either collapsed local rel-L2, the next implementation path would be a full-NVFP4 read/output gain
correction rather than a mixed-precision fallback.

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

## Gain Result

| call | q shape | window | NVFP4 rel-L2 | scalar-gain rel-L2 | per-head-gain rel-L2 | scalar alpha | head alpha mean | head alpha std | dominant |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `(4096, 16, 256)` | 1023 | 0.099016505 | 0.098901258 | 0.098809141 | 0.995235630 | 0.993790388 | 0.005282900 | mixed |
| 1 | `(4096, 16, 256)` | 1023 | 0.106627960 | 0.106556877 | 0.106423980 | 0.996077763 | 0.994708300 | 0.006296660 | mixed |
| 2 | `(4096, 16, 256)` | 1023 | 0.143606733 | 0.143543869 | 0.143293198 | 0.995726836 | 0.994494438 | 0.008071366 | mixed |
| 3 | `(4096, 16, 256)` | 1023 | 0.128243660 | 0.128093007 | 0.127926136 | 0.993778952 | 0.994014800 | 0.007235472 | mixed |
| 4 | `(4096, 16, 256)` | 1023 | 0.141596498 | 0.141566997 | 0.140974324 | 0.997069158 | 0.996063054 | 0.013864185 | K |
| 5 | `(4096, 16, 512)` | -1 | 0.117013252 | 0.116943320 | 0.116828023 | 1.004112546 | 1.002448916 | 0.005608715 | V |
| 6 | `(4096, 16, 512)` | -1 | 0.113974661 | 0.113761235 | 0.113608030 | 1.007080891 | 1.006154180 | 0.006224681 | V |
| 7 | `(4096, 16, 256)` | 1023 | 0.159155129 | 0.158993903 | 0.158653305 | 0.992788567 | 0.990892291 | 0.011122944 | V |

Verdict: simple scalar or per-head output gain is not the missing full-NVFP4 correction. The gain fit barely
moves local rel-L2; the residual is directional/structural, not a uniform output amplitude error. This pushes
the next practical serving path toward Claude's whole-layer fp8 ladder, while full NVFP4 remains a separate
research item needing something richer than gain correction.

## Artifact Handling

`TAR_ARTIFACT=0` was used so the remote script wrote derived reports and skipped compressing the raw active
tensor tree. The raw active-page tensors were intentionally not retained in git. The committed artifact keeps
derived reports, row JSON, logs, and rerunnable scripts.

The Vast instance used for this run (`41146434`) was destroyed after pulling derived artifacts. `vastai show
instances` still listed a different unlabeled Ubuntu 24.04 instance (`41147023`); it was not created by this
run and was left untouched.

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
