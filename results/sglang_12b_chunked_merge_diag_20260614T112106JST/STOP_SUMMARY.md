# SGLang 12B Chunked-Prefill Diagnostic Stop Point

Date: 2026-06-14 11:21-11:26 JST

Scope: diagnostic only; known-blocked SGLang Gemma 4 12B full-NVFP4 AR replay; not a claim-grade ladder result.

## Configuration

- Host: DGX Spark / GB10, sm_121
- Repo: `dgx-spark-hijinks` `epoch2` at `5e38057d05c7b784c6759a73d16c9a074ed3d1a5`
- Image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
- Model: `google/gemma-4-12B-it`
- KV dtype: `fp4_e2m1` full NVFP4 K+V
- Context: `8185`
- Reused prefix: `4096`
- Logprob start: `4096`
- Page size: `1`
- CUDA graphs: disabled
- Radix cache: enabled
- Memory: one server, Docker `--memory 100g`, `mem_fraction_static=0.72`
- Extra server arg: `--chunked-prefill-size 2048`
- Override reason: `mail 0140 scoped chunked-prefill diagnostic --chunked-prefill-size 2048 not claim-grade`

## Result

The server reached readiness, both chat smoke requests returned `Tokyo`, and the supplied-token PPL probe succeeded.

| row | mean NLL | PPL | cached tokens | scored tokens |
| --- | ---: | ---: | ---: | ---: |
| banked bf16 baseline | 4.571989822602 | 96.736407 | 4096 | 4088 |
| old full-NVFP4 12B red | 4.974959038640 | 144.742896 | 4096 | 4088 |
| chunked full-NVFP4 diagnostic | 4.926853552137 | 137.944793 | 4096 | 4088 |

Delta vs banked bf16 baseline:

- old full-NVFP4: `+0.402969216038` nats/token
- chunked full-NVFP4 with `--chunked-prefill-size 2048`: `+0.354863729534` nats/token
- improvement from chunking: `0.048105486504` nats/token

Interpretation: `--chunked-prefill-size 2048` improves the known-red 12B row modestly but does not land near the mail 0140 reference cost of about `+0.19` nats/token. This remains a scoped diagnostic red and does not unblock the SGLang Gemma 4 AR ladder.

## Artifacts

- Manifest: `manifest.json`
- Blocker audit: `blocker_audit.json`
- Claim audit: `claim_audit.json` (expected red for this single-arm diagnostic)
- PPL: `google-gemma-4-12b-it/fullnvfp4_ppl.json`
- Summary: `google-gemma-4-12b-it/fullnvfp4_summary.json`
- Server log: `google-gemma-4-12b-it/fullnvfp4_server.log`
- Preflight: `google-gemma-4-12b-it/fullnvfp4_preflight.log`

## Notes

- `blocker_audit.json` still records the known blocked FlashInfer/SGLang dependency refs, as expected.
- `claim_audit.json` is not green because this run intentionally excludes the full bf16/fp8/full-NVFP4 matched ladder and only executes one full-NVFP4 diagnostic arm.
- Provenance caveat: this run predates commit `e36578a2af8450f0e0bdc058fb3e31214a1121ba`, which fixed the runner to pass `SGLANG_AR_LADDER_OVERRIDE_REASON` into `sglang_gemma4_ar_ladder_blocker_audit.py`. Therefore this artifact's `blocker_audit.json` incorrectly records `diagnostic_override_allowed=false` even though the run preflight records `allow_known_blocked_sglang_ar_ladder=1` and the explicit diagnostic override reason. The raw JSON is left unchanged; future diagnostic replays should record the override directly in `blocker_audit.json`.
- The packaged image does not expose Git metadata under `/work/third_party/{sglang,flashinfer}`, so the claim audit also reports source-ref provenance as unavailable. This does not affect the diagnostic NLL result, but it is another reason this artifact is not claim-grade.
