# SGLang Gemma 4 12B AR Claim Attempt

Status: RED - fp8 comparator fails before a matched full-NVFP4 claim row.

## Scope

- Runtime: SGLang on DGX Spark, packaged Ubuntu 22.04 / arm64 / torch 2.11 image
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- Model: `google/gemma-4-12B-it`
- Commit: `73e8c64480452c2cc9915f3e8cbb7bdc8cdb13e1`
- Shape: ctx `8185`, reused prefix `4096`, page size `1`, graphs disabled
- Intended sequence: bf16/auto -> fp8 -> full NVFP4 K+V

## What Passed

The bf16/auto-KV comparator row passed:

- readiness: pass
- chat smoke: `Tokyo` / `Tokyo`
- supplied-token PPL: pass
- prefix reuse proof: score response reports `cached_tokens=4096`
- PPL: `96.7364066795068`
- mean NLL: `4.571989822602299` nats/token
- scored tokens: `4088`
- D=512 VO-split routing proof is present in `bf16_server.log`.

## What Failed

The fp8 comparator row reached readiness but fails before it can serve as a
matched comparator:

- OpenAI chat smoke: both attempts timed out after the bounded 120s chat timeout.
- Prompt-logprob PPL: failed after `358.067s` with
  `RemoteDisconnected('Remote end closed connection without response')`.
- Server log root cause:
  `FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1`
  from `BatchPrefillWithPagedKVCacheDispatched` in
  `include/flashinfer/attention/prefill.cuh:3215`.

The full-NVFP4 row was not run because the matched fp8 comparator gate failed.

## Interpretation

This is not an NVFP4 quality failure. It is the Gemma4 AR fp8 comparator red,
reproduced on the new packaged SGLang image. The broad claim remains blocked:
we cannot honestly publish a matched fp8-vs-full-NVFP4 SGLang Gemma4 AR row
until the fp8 paged-prefill dispatcher/configuration issue is fixed or the
claim is explicitly narrowed to a bf16-only comparator.

The effective prompt length for this launch is also now pinned: `ctx=8192`,
`8191`, and `8186` hit SGLang request-length guards; `ctx=8185` is the valid
8k-class request shape with `max_new_tokens=1`.

## Key Artifacts

- `bf16_ppl.json`
- `bf16_summary.json`
- `bf16_server.log`
- `fp8_ppl.json`
- `fp8_chat_status.txt`
- `fp8_server.log`
- `../sglang_gemma4_12b_ar_claim_ctx8185_prefix4096_20260613T105511JST_runner.log`
