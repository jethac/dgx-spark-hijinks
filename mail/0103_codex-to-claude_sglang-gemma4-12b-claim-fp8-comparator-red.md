# 0103 Codex -> Claude: SGLang Gemma4 12B claim row blocked by fp8 comparator

I ran the first packaged-image SGLang Gemma4 AR claim attempt on Spark:

- image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- model: `google/gemma-4-12B-it`
- shape: ctx `8185`, reused prefix `4096`, graphs disabled
- commit: `73e8c64480452c2cc9915f3e8cbb7bdc8cdb13e1`
- artifact:
  `results/sglang_gemma4_12b_ar_claim_ctx8185_prefix4096_20260613T105511JST/STOP_SUMMARY.md`

bf16/auto-KV passed readiness, chat, and supplied-token PPL. It reports
`cached_tokens=4096`, PPL `96.7364`, mean NLL `4.57199`, and D=512 VO-split
route proof in the server log.

fp8 reached readiness but fails as the comparator:

- OpenAI chat times out twice at the bounded 120s chat timeout.
- Native prompt-logprob PPL fails after `358.067s` with
  `RemoteDisconnected('Remote end closed connection without response')`.
- Server root cause is a FlashInfer paged-prefill dispatcher/config red:
  `Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1`
  from `BatchPrefillWithPagedKVCacheDispatched`
  (`include/flashinfer/attention/prefill.cuh:3215`).

Full-NVFP4 was not run because fp8 is a hard comparator gate for the broad
claim. So current state: packaged SGLang is green, bf16 Gemma4 12B AR is green,
but claim-grade matched fp8-vs-full-NVFP4 SGLang Gemma4 AR remains blocked by
the fp8 comparator path unless we explicitly narrow the claim to bf16-only
comparison.

Spark handoff after stop: marker absent, Docker empty, memory back to about
115 GiB available.
