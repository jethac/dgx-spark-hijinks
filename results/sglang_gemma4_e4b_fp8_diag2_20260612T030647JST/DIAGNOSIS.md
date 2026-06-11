# SGLang Gemma 4 E4B fp8 Comparator Diagnosis

Run: `sglang_gemma4_e4b_fp8_diag2_20260612T030647JST`
Date: 2026-06-12 JST

Status: RED, root-caused to FlashInfer paged-prefill dispatch/configuration for the fp8 global D=512 VO-split path.

## Stack

- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- Parent repo: `403f85a44594e553b7764e80f015fa75685854b0`
- SGLang: `651d55cd2e6a3d90de0eb65af643d0aa4ee7fca2`
- FlashInfer: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Model: `google/gemma-4-E4B-it`
- KV dtype: `fp8_e4m3`
- Graphs: disabled
- Memory cap: Docker `100g`, `--mem-fraction-static 0.40`

## Result

The server reached readiness and allocated the fp8 hybrid-SWA KV pool:

- SWA pool: `579712` tokens
- Full/global pool: `724641` tokens
- `max_total_num_tokens=724641`

The request returned no response body; curl timed out after 120 seconds:

- `request_status.txt`: `28`
- `chat_openai.json`: empty

This is not a quality-comparator row. It dies before producing a parseable response.

## Root Cause

The first request reaches SWA D=256 fp8 paged prefill, then crashes on the first global D=512 VO-split prefill call.

Crash site:

```text
RuntimeError: Error in function 'BatchPrefillWithPagedKVCacheDispatched' at /flashinfer-src/include/flashinfer/attention/prefill.cuh:3215: FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

The failing module parameters are:

```text
dtype_q=nv_bfloat16
dtype_kv=__nv_fp8_e4m3
dtype_o=nv_bfloat16
head_dim_qk=512
head_dim_vo=256
require_fp4_kv=0
runtime: window_left=-1, batch_size=1, num_qo_heads=8, num_kv_heads=2, page_size=1, split_kv=1, cta_tile_q=64
```

The wrapper geometry immediately before the crash is:

```text
label=extend_paged_vosplit0 layer=5
planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256)
layer_head_dim=512 sliding_window=-1
kv_cache dtype=torch.float8_e4m3fn
```

The run also logs:

```text
Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0.
```

That scale warning is not the observed root cause for this row: the request fails inside FlashInfer dispatch before any valid quality comparison can run.

## Interpretation

The old E4B fp8 comparator red is no longer ambiguous. It is not a missing VO-split route, not the earlier `max_mma_kv: 0` wall, and not a decode-side proof gap. The row reaches the intended global D=512 VO-split prefill route and FlashInfer rejects the generated fp8 paged-prefill configuration.

The D=512 decode proof is still missing because the request never survives prefill.

## Next Fix Surface

One of these has to change before fp8 can be used as the E4B comparator:

- FlashInfer accepts a valid paged-prefill config for fp8 KV with `(head_dim_qk=512, head_dim_vo=256, num_qo_heads=8, num_kv_heads=2, page_size=1)`, or
- SGLang's selector/fallback avoids this invalid FlashInfer fp8 paged-prefill configuration for global Gemma 4 layers.

Until then, E4B quality comparison should remain against the working bf16/auto row, and the fp8 comparator remains RED.
