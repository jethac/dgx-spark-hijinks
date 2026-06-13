# FlashInfer prefill dispatcher `NUM_MMA_KV=1` — stock-upstream repro verdict

**Date:** 2026-06-13 · **Goal:** build a repro of the FA2 prefill `NUM_MMA_KV`
dispatch red on **stock upstream main**, and *if it reproduces* write the atomic
fix as a standalone PR fileable ahead of the campaign work.

> **REVISED (supersedes the earlier "does NOT reproduce" verdict).** The first
> pass concluded the bug couldn't be expressed on upstream because the *paged*
> `plan()` I used takes a single `head_dim`. That was wrong: `head_dim_vo` is a
> **documented public parameter** on both prefill wrappers, and the **ragged**
> wrapper accepts asymmetric `QK≠VO` directly. With it, **the bug reproduces on
> stock upstream `c15ac84`** — so the atomic fix IS warranted and is written.

## Verdict: REPRODUCES on stock upstream → atomic PR written.

- **Repro:** `BatchPrefillWithRaggedKVCacheWrapper.plan(..., head_dim_qk=512,
  head_dim_vo=256, kv_data_type=fp8_e4m3)` on sm_120 (~100 KB/SM) throws
  `Invalid configuration : … NUM_MMA_KV=1 NUM_WARPS_Q=4 … please create an issue`
  at every `qo_len` (8…512). Pure stock upstream, public API, no campaign code.
- **Fix:** floor the *dispatched* `NUM_MMA_KV` against `kMinValidMmaKV` (which
  upstream already computes but only applies to the occupancy budget); when no
  valid tile fits, raise a clear shared-memory diagnostic instead of the
  "please file an issue" path. 3 sites, +57/-3. See `docs/flashinfer_pr/`.
- **Scope (honest):** diagnostic correctness, not functional enablement. The
  config genuinely exceeds sm_120 shared memory; the fix replaces a misleading
  internal-error with the real reason + workaround. Making it *run* needs the
  V/O-split (the larger, separate campaign PR).

## Evidence

### Why the first pass missed it
The paged wrapper requires K/V stride match (symmetric head dims), so passing a
single positional `head_dim` to `plan()` only ever tested `qk==vo`. Head 128/256
fp8 all pass there. But `plan()` also takes `head_dim_vo: Optional[int]` (docs:
"head_dim_vo != head_dim_qk"), and the **ragged** path takes separate k/v tensors
→ asymmetric `qk=512/vo=256` is expressible and is exactly the VO-split shape.

### Root cause
`c15ac84` defines `kMinValidMmaKV` (smallest `NUM_MMA_KV` satisfying the 1-byte
alignment `NUM_MMA_KV*2 % NUM_WARPS_Q == 0`) and uses it for `num_ctas_per_sm`,
but the **dispatched** value `min(max_num_mma_kv_smem, max_num_mma_kv_reg)` is NOT
floored. `DISPATCH_NUM_MMA_KV` snaps to a power of two `{8,4,2,1}`; the only one
that fails the alignment is `1`, dispatched exactly when smem admits one KV tile
but the minimum valid tile needs two (`reg` floors at 2, so smem always binds).

### Verification (sm_120, RTX 5060 Ti, CUDA 13.0)
- `qk=512 vo=256` fp8 → clean "insufficient shared memory … needs NUM_MMA_KV>=2
  … use a smaller head_dim or 2-byte KV" (was "please create an issue").
- head 256 / head 128 fp8, qo_len 17–512 → OK (no regression).
- `qk=512 vo=256` **bf16** → OK (2-byte path unaffected; `kMinValidMmaKV=1`).

### Dtype axis (Codex 0105, still valid)
On SGLang at the same VO-split shape, full-NVFP4 K+V is GREEN; only the fp8
comparator reds. Consistent: nvfp4's packed-uint8 accounting lands on a valid
`NUM_MMA_KV`; fp8 does not. The upstream fix covers any 1-byte KV (fp8 and fp4).

## Disposition
- **Atomic PR materials written** (`docs/flashinfer_pr/`: patch, PR_DESCRIPTION,
  repro scripts). **Not filed** — opening an upstream PR is outward-facing and is
  Jetha's call.
- Independent of the campaign's VO-split FlashInfer PR (which is held behind the
  SGLang AR ladder). This diagnostic fix can land ahead of it.

### Companion finding (out of scope)
head 192 + fp8/bf16 `qo_len>16` throws a *different* invariant
(`NUM_MMA_D_VO % (2·NUM_WARPS_Q)`, `NUM_MMA_KV=4` not 1) and head 192 is broadly
unsupported (1-warp path errors too). Not addressed by this PR.
