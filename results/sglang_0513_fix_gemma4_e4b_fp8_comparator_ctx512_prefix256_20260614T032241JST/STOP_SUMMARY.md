# SGLang 0.5.13 Gemma 4 E4B fp8 Comparator Check

Status: RED. The fp8 comparator caveat is still real on the rebuilt SGLang
0.5.13 image.

## Scope

- Runtime: SGLang on DGX Spark, packaged Ubuntu 22.04 / arm64 / torch 2.11 image
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Parent hijinks ref at run time: `7cc1a7a6010e3f75e88b2d78e54c0d4d7c8aa52d`
- SGLang ref: `42ce5dad84ddf75da56282bc556d6df9f5c81303`
- FlashInfer ref: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Model: `google/gemma-4-E4B-it`
- Row: `fp8`
- Shape: ctx `512`, intended reused prefix `256`, page size `1`, graphs disabled
- Memory guardrail: one server, Docker `--memory 100g`

This row was run after the scoped green E4B bf16-vs-full-NVFP4 checkpoint:

```text
results/sglang_0513_fix_gemma4_e4b_matched_bf16_fullnvfp4_ctx512_prefix256_20260614T030829JST/STOP_SUMMARY.md
```

## Verdict

The server reaches readiness and allocates fp8 KV pools, but the first chat
request crashes the scheduler inside FlashInfer paged prefill for the D=512
VO-split global layer. No fp8 quality row was produced.

Key error:

```text
tvm.error.InternalError: Error in function 'BatchPrefillWithPagedKVCacheDispatched'
at /work/third_party/flashinfer/include/flashinfer/attention/prefill.cuh:3215:
FlashInfer Internal Error: Invalid configuration :
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1
NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

The failing call is the E4B global-layer VO-split path:

```text
SGLang Gemma4 FlashInfer geometry label=extend_paged_vosplit0 layer=5
planned=FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2,
head_dim=512, head_dim_vo=256)
```

The generated FlashInfer fp8 module agrees with that geometry:

```text
dtype_kv=__nv_fp8_e4m3; head_dim_qk=512; head_dim_vo=256;
page_size=1; split_kv=1; cta_tile_q=64
```

## Capacity Denominator

This is not a usable capacity or quality row because the fp8 request crashes.
Startup logs nevertheless record the fp8 allocator token counts:

| pool | bf16 tokens from matched E4B row | fp8 tokens | full-NVFP4 tokens from matched E4B row | fp8/bf16 | full-NVFP4/fp8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SWA | 644444 | 1282614 | 2279878 | 1.990264 | 1.777525 |
| full | 805556 | 1603268 | 2849848 | 1.990263 | 1.777524 |

These are allocator denominators only. The fp8 comparator does not serve.

## Runner Note

The first chat request had already crashed the scheduler. The subsequent PPL
helper reports:

```text
FileNotFoundError(2, 'No such file or directory')
```

because this narrow fp8-only run reused the prior matched row's corpus path but
did not copy that corpus into the new run directory. That secondary artifact is
not the red gate. The red gate is the scheduler crash and FlashInfer invalid
configuration in `fp8_server.log`.

## Interpretation

The rebuilt `42ce5dad` image fixed the trace-helper startup crash and can serve
E4B full-NVFP4, but it does not close the fp8 comparator path. The SGLang Gemma
4 E4B evidence is therefore:

- bf16 vs full-NVFP4: scoped green on the single E4B corpus;
- fp8 comparator: red before any quality measurement, blocked by FlashInfer
  D=512 VO-split fp8 paged-prefill dispatch.

Do not publish a broad matched bf16/fp8/NVFP4 SGLang Gemma 4 claim from this
image.

## Host State

At stop point:

- marker: absent
- Docker: no running containers
- memory: about 115 GiB available

## Key Artifacts

- `manifest.json`
- `google-gemma-4-e4b-it/fp8_server.log`
- `google-gemma-4-e4b-it/fp8_provenance.log`
- `google-gemma-4-e4b-it/fp8_container_inspect.json`
- `google-gemma-4-e4b-it/fp8_chat_status.txt`
- `docker_ps_after.txt`
- `free_after.txt`
