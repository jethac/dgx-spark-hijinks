# SGLang 0.5.13 Spark Live Validation Stop Point

Date: 2026-06-14 JST  
Host: `thinkstationpgx-00b4` / GB10 / `sm_121`  
Image under test:
`ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:561b2a82b4169905625584ea1837feca5f7e0502b8f5f1bbd2a665c234ec8bb3`

Image provenance:

- build run: `27466068365`
- image tag: `epoch2-sglang-0513-74e0e4bb-arm64`
- base: `nvidia/cuda:13.0.2-devel-ubuntu22.04`
- torch: `2.11.0+cu130`
- SGLang: `74e0e4bb5f058b0e4acac10e769268bb2f9a0c85`
- FlashInfer: `f99323bd7d1cc88d9445202c12934070be754e2d`
- transformers: `5.11.0`

## Verdict

Live validation is not claim-green on this image. The run produced useful
evidence, but full-NVFP4 Gemma 4 12B cannot complete on the packaged
`74e0e4bb` image because the SGLang FP4 module-trace helper is missing:

```text
NameError: name '_fp4_kv_module_trace_enabled' is not defined.
```

Corrective patch landed in SGLang:

```text
jethac/sglang@spark/hijinks-025-sglang-0.5.13-rebase
42ce5dad84 Fix FP4 KV module trace gate
```

A rebuilt packaged image is required before rerunning the full-NVFP4 quality
row. No Spark-local patching or loose-wheel injection was used.

## Block A: SF Layout Probe

Artifact:

- `sglang_0513_nvfp4_sf_stride_probe_20260614T011248JST/layout_probe_gemma4_swa_h256.json`

Shape:

- tokens: `64`
- query heads: `32`
- KV heads: `16`
- head dim: `256`
- page size: `1`
- packed KV shape: `[64, 16, 128]`
- flat scale shape: `[64, 16, 16]`

Result:

- rank-4 SF `[64, 1, 16, 16]`: cosine vs dequant reference `0.9999961853`,
  relative error `0.00478327`, finite, passed.
- rank-3 SF `[64, 16, 16]`: same cosine and relative error, also passed.

Interpretation: the old probe expectation that rank-3 must fail is stale for
FlashInfer `0.6.13` at page size 1. This is not a layout corruption finding;
rank-4, the SGLang serving shape, is numerically green against the dequant
reference.

## 12B BF16 Baseline

Artifact:

- `sglang_0513_gemma4_12b_fullnvfp4_ctx8185_prefix4096_20260614T012657JST/google-gemma-4-12b-it/bf16_ppl.json`

Config:

- model: `google/gemma-4-12B-it`
- KV dtype: `auto` / BF16
- ctx: `8185`
- reused prefix: `4096`
- scored tokens: `4088`
- graphs disabled
- radix/cache reuse enabled

Result:

- chat: `Tokyo` / `Tokyo`
- cached tokens: `4096`
- mean NLL: `4.571989822602299`
- PPL: `96.7364066795068`
- BF16 capacity log: `swa size=174371`, `full size=217964`,
  `SWAKVPool mem usage=56.54 GB`

The earlier ctx `8192` launch failed only because `max_new_tokens=1` made the
request exceed SGLang's context limit. The corrected ctx `8185` row passed.

## FP8 Comparator Blocker

Artifact:

- `sglang_0513_gemma4_12b_fullnvfp4_ctx8185_prefix4096_20260614T012657JST/google-gemma-4-12b-it/fp8_server.log`

The fp8 comparator loaded and allocated cache:

- `swa size=348932`
- `full size=436166`
- `SWAKVPool mem usage=56.57 GB`

It then failed in FlashInfer paged prefill:

```text
Invalid configuration :
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16
NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

Therefore the matched `bf16 -> fp8 -> fullnvfp4` claim row remains blocked by
the fp8 comparator on this image.

## Full-NVFP4 Blocker On 74e0e4bb Image

Artifacts:

- `sglang_0513_gemma4_12b_fullnvfp4_only_ctx8185_prefix4096_20260614T013747JST/`
- `sglang_0513_gemma4_12b_fullnvfp4_only_ctx8185_prefix4096_20260614T014646JST/`

Full-NVFP4 loaded and allocated cache:

- `full_per_token_bytes=576`
- `swa_per_token_bytes=2304`
- `swa size=620197..621111`
- `full size=775247..776389`
- `SWAKVPool mem usage=56.56..56.64 GB`

The diagnostic trace row printed the live FP4 SF layout at layer 0 before
crashing:

```text
k_sf shape=(620198, 1, 8, 16), stride=(128, 128, 16, 1)
v_sf shape=(620198, 1, 8, 16), stride=(128, 128, 16, 1)
k_scale=0.0005958193796686828
v_scale=0.005905878264456987
dtype_kv=__nv_fp4x2_e2m1
fp4_kv=1
```

That is the intended linear per-16 SF shape for SGLang page-size-1 serving. The
quality row still crashed because `_run_paged_native` references the undefined
`_fp4_kv_module_trace_enabled()` helper whenever `is_nvfp4_native` is true.

## Follow-Up

1. Build a new Spark-packaged SGLang image from
   `jethac/sglang@42ce5dad84`.
2. Rerun:
   - full-NVFP4-only 12B ctx `8185`, prefix `4096`;
   - if it serves, compare against the BF16 baseline above;
   - then revisit the fp8 comparator blocker separately.
3. Do not quote a SGLang full-NVFP4 Gemma 4 12B quality or capacity claim from
   this image.
