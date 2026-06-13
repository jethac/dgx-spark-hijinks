# SGLang Gemma 4 12B Full-NVFP4 Diagnostic

Status: GREEN as a dispatcher diagnostic; NOT a claim-grade matched quality row.

## Scope

- Runtime: SGLang on DGX Spark, packaged Ubuntu 22.04 / arm64 / torch 2.11 image
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- Model: `google/gemma-4-12B-it`
- Commit: `660f1c38e21ca3137469bfeff641f759a0fe894f`
- Shape: ctx `8185`, reused prefix `4096`, page size `1`, graphs disabled
- Row labels: `fullnvfp4` only
- KV dtype: `fp4_e2m1`, full NVFP4 K+V (`mixed_kv=False`)

This run was launched specifically in response to Claude mail `0104`, to test
whether the fp8 comparator's FlashInfer paged-prefill dispatcher failure also
affects full-NVFP4 K+V at the same Gemma 4 12B shape.

## Result

Full-NVFP4 did not reproduce the fp8 dispatcher failure.

- readiness: pass
- chat smoke: `Tokyo` / `Tokyo`
- supplied-token PPL: pass
- prefix reuse proof: score response reports `cached_tokens=4096`
- PPL: `144.74289616094566`
- mean NLL: `4.974959038640488` nats/token
- scored tokens: `4088`
- prompt-logprob elapsed: `3.7259861910133623` seconds
- `fullnvfp4_server.log` contains no `Invalid configuration` / `NUM_MMA_KV=1`
  failure.

The server log proves the expected packed FP4 path:

- `SGLANG_GEMMA_KV_POOL_CONFIG ... mixed_kv=False`
- FP4 paged-prefill traces have `_cached_kv_data_type='torch.uint8'`
- K/V scale tensors are present as `torch.float8_e4m3fn`
- D=512 global layers use `extend_paged_vosplit0/1`
- the long request logs `#new-token: 4089, #cached-token: 4096`

## Interpretation

This falsifies the broad version of the fp8 hypothesis: the failing fp8
comparator row is not simply "all 1-byte KV dtypes fail at this Gemma 4 12B
VO-split shape." Full-NVFP4 K+V can get through the same ctx `8185` / prefix
`4096` radix-reuse prompt on the packaged SGLang image.

The claim-grade SGLang 12B row is still not green:

- This run intentionally skipped bf16/fp8 comparators, so it is diagnostic only.
- The previous packaged-image bf16 artifact used the same corpus and shape and
  reported mean NLL `4.571989822602299`, so the cross-artifact bf16-vs-NVFP4
  delta is `+0.4029692160381897` nats/token (`96.7364` -> `144.7429` PPL).
- That delta must be treated as a quality blocker or at least rerun as a
  matched bf16-vs-full-NVFP4 pair before any narrowed claim is published.
- The fp8 comparator remains blocked by the FlashInfer dispatcher red recorded
  in `results/sglang_gemma4_12b_ar_claim_ctx8185_prefix4096_20260613T105511JST/STOP_SUMMARY.md`.

## Host State

At stop point:

- marker: absent
- Docker: no running containers
- memory: ~115 GiB available

## Key Artifacts

- `google-gemma-4-12b-it/fullnvfp4_ppl.json`
- `google-gemma-4-12b-it/fullnvfp4_summary.json`
- `google-gemma-4-12b-it/fullnvfp4_server.log`
- `manifest.json`
- `../sglang_gemma4_12b_ar_nvfp4_only_ctx8185_prefix4096_20260613T113248JST_runner.log`
