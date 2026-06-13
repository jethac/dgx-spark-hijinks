# FlashInfer prefill dispatcher `NUM_MMA_KV=1` floor — stock-upstream repro verdict

**Date:** 2026-06-13 · **Goal:** build a repro of the FA2 prefill `NUM_MMA_KV`
dispatch red on **stock upstream main**, and *if it reproduces* write the atomic
fix as a standalone PR fileable ahead of the campaign work.

## Verdict: does NOT reproduce on stock upstream → no standalone PR.

The red is **narrow: fp8 KV + `head_dim_qk=512` (asymmetric QK≠VO, our VO-split)**.
Stock upstream cannot reach it on two independent axes, so a maintainer could not
reproduce a standalone PR. The `NUM_MMA_KV` floor fix stays **campaign-local**,
guarding only the fp8 *comparator* path. Our headline full-NVFP4 path is unaffected.

## Repro environment (artifact-grade)

- Stock upstream clone: `flashinfer-ai/flashinfer @ c15ac84` (== `upstream/main`),
  submodules incl. `3rdparty/cccl`, `3rdparty/cutlass` checked out.
- Host: P520 WSL2, sm_120 (RTX 5060 Ti), CUDA 13.0, torch 2.11, JIT for `sm_120f`.
- Driver scripts: `~/fi_sweep.py` (head_dim × qo_len sweep), `~/fi_probe192.py`.
- API: `BatchPrefillWithPagedKVCacheWrapper("NHD")`, GQA 8/1, page 16, causal,
  `q=bf16`, `kv=float8_e4m3fn`, kv_len 8192.

## Evidence

### 1. Head-dim axis — head 128/256 fp8 all PASS (no red at any qo_len)

`FA2DetermineCtaTileQ` (upstream `utils.cuh:384`) is 2-arg / symmetric. The red
condition is `IsInvalid()`'s `sizeof(DTypeKV)==1 && NUM_MMA_KV*2 % NUM_WARPS_Q != 0`
(`prefill.cuh:182`), reachable only when `min(max_num_mma_kv_smem, max_num_mma_kv_reg)`
lands on 1 with `NUM_WARPS_Q=4` (`cta_tile_q ∈ {64,128}`, i.e. `qo_len>16`).

| head_dim | qo_len ∈ {17..512} | cta_tile_q | NUM_WARPS_Q | result |
|---|---|---|---|---|
| 256 | all | 64 | 4 | **OK** (smem/reg fit NUM_MMA_KV ≥ 2) |
| 128 | all | 64/128 | 4 | **OK** |
| 192 | all >16 | 64/128 | 4 | ERROR — *different* invariant (below) |

At the upstream-supported head dims (128/256), smem/reg always admit
`NUM_MMA_KV ≥ 2`. Squeezing the dispatched value to 1 requires `head_dim_qk=512`,
which doubles the QK smem term — and upstream's symmetric `plan(head_dim)` API
**cannot express QK≠VO**. So the targeted red is unreachable on stock upstream.

### 2. head 192 throw is a *different* family (and broadly unsupported)

head 192 + fp8/bf16, `qo_len>16` → `Invalid configuration ... NUM_MMA_D_VO=12
NUM_MMA_KV=4 NUM_WARPS_Q=4` — fails the `NUM_MMA_D_VO % (2·NUM_WARPS_Q)` invariant
(`prefill.cuh:180`), **not** the 1-byte `NUM_MMA_KV` floor (KV=4 here, not 1).
The 1-warp path (`qo_len≤16`) *also* errors for head 192 (both dtypes), so head 192
is broadly unsupported, not a clean selector bug. Not a viable PR vehicle.

### 3. Dtype axis — even at d512, only fp8 reds (Codex 0105)

Independent SGLang confirmation on the *same* VO-split shape (ctx 8185, prefix
4096, head_dim_qk=512, page 1, graphs off), image
`sglang-gemma4-source-stack@sha256:0d5e160c…`, commit `660f1c38`:
**full-NVFP4 K+V is GREEN** (no `Invalid configuration` / `NUM_MMA_KV=1`), chat
smoke Tokyo/Tokyo, cached_tokens 4096. Only the **fp8** comparator reds (0103).

This **corrects mail 0104's over-claim** ("affects fp8 AND nvfp4"): the red is
**fp8-specific** at d512. nvfp4's packed-uint8 smem accounting lands on a valid
`NUM_MMA_KV`; fp8's does not.

## Consequences

- **No standalone upstream PR.** The "file ahead of everything" premise (clean
  standalone upstream bug) is false — the trigger is unreachable on upstream.
- The `NUM_MMA_KV`/`max_mma_kv=0` floor + smem-aware 3-arg `FA2DetermineCtaTileQ`
  stay **campaign-local** (Task #17, branch `spark/hijinks-022-fa2-d512`), guarding
  the fp8 *comparator* path at head_dim_qk=512. Headline full-NVFP4 is unaffected.
- Optional future: a *defensive* upstream PR making `FA2DetermineCtaTileQ`
  constraint-aware (fall back to 1-warp when the 4-warp layout is invalid for the
  head dim) would also fix the head-192 throw — but that is a "harden the
  dispatcher" improvement, not a reproducible bug-fix, and is a maintainer/Jetha
  call (outward-facing). Not filed.
