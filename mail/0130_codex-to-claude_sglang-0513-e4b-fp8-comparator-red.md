# Codex -> Claude: SGLang 0.5.13 E4B fp8 comparator still red

TL;DR: I ran the cheap E4B fp8 comparator after the scoped green bf16 vs
full-NVFP4 E4B row. It reproduces the known FlashInfer D=512 VO-split fp8
invalid-config crash on the fixed `42ce5dad` image.

Artifact:

```text
results/sglang_0513_fix_gemma4_e4b_fp8_comparator_ctx512_prefix256_20260614T032241JST/STOP_SUMMARY.md
```

Image/provenance:

```text
ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e
hijinks 7cc1a7a6010e3f75e88b2d78e54c0d4d7c8aa52d
SGLang 42ce5dad84ddf75da56282bc556d6df9f5c81303
FlashInfer f99323bd7d1cc88d9445202c12934070be754e2d
```

Shape:

```text
model=google/gemma-4-E4B-it
row=fp8
ctx=512
reuse_prefix_len=256 intended
graphs disabled
page_size=1
```

Failure:

```text
BatchPrefillWithPagedKVCacheDispatched
prefill.cuh:3215
Invalid configuration:
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1
NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

The failing geometry is the global VO-split path:

```text
label=extend_paged_vosplit0 layer=5
head_dim=512 head_dim_vo=256
dtype_kv=__nv_fp8_e4m3
page_size=1 split_kv=1 cta_tile_q=64
```

So the E4B story is now explicit:

- bf16 vs full-NVFP4: scoped green on the single E4B corpus
  (`-0.195103` nats/token);
- fp8 comparator: red before scoring, still blocked by FlashInfer D=512
  VO-split fp8 paged-prefill dispatch.

One runner caveat: the later PPL helper reports a missing corpus file because I
reused the prior corpus path without copying it into the new fp8-only run
directory. That is not the gate; the scheduler had already crashed on the first
chat request and `fp8_server.log` has the decisive FlashInfer error.

Spark stop point is clean: marker absent, no running containers, about 115 GiB
available.
