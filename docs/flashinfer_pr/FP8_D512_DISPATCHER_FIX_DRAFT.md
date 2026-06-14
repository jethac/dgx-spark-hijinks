# DRAFT — E4B fp8 D512/VO256 dispatcher fix (mail 0136 / 0140)

> Status: **GB10/Spark-specific; authored + scope-corrected on local sm_120, NOT yet verified on
> GB10.** Local sm_120 evidence below corrected two things in the original draft. The GB10
> verification (the only arch where it fails) belongs to the Spark lane (Codex) — see mail 0144.

## GB10 confirmation (Spark, sm_121, cc 12.1, flashinfer 0.6.13; 2026-06-14) — the fix premise is PROVEN
Ran `repro_d512_ragged.py` in the SGLang container on the GB10: fp8 D512/VO256 **rejects for ALL qo
{8,17,32,64,128,256}** with `NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4`
(`prefill.cuh:3073` "Invalid configuration"). Key facts:
- GB10 smem/SM = **102400 B (100 KB)** — SAME as the 5060 Ti, yet GB10 rejects and the 5060 Ti
  (upstream 0.6.12) ran. So it is the **FA2DetermineCtaTileQ smem-check (sizeof=2) returning 64**
  on the campaign-fork flashinfer, not a raw smem-budget difference.
- The reject shows `NUM_WARPS_Q=4` even for qo=8 → `FA2DetermineCtaTileQ` returned **64 for every qo**,
  because the smem check uses `sizeof=2`: `q_tile 16*512*2 + kv_step (512+256)*16*4*2 = 16384+98304
  = 114688 > 102400`.
- With the **real 1-byte size**: `16384 + (512+256)*16*4*1 = 16384+49152 = 65536 ≤ 102400` → returns
  16 → `NUM_WARPS_Q=2` → `NUM_MMA_KV=1` valid (`2*1 % 2 == 0`). **The fix premise is confirmed on the
  failing arch.**

So the fix is: make the smem check use the actual KV element size (1 for fp8/nvfp4). This is a
one-file change to `FA2DetermineCtaTileQ` + threading `kv_elem_size` from `PrefillPlan` / the JIT
plan binding (the KV dtype is known there). Codex owns the SGLang flashinfer build and the GB10 →
it applies the one-file change + reruns the E4B fp8 comparator (mail 0148).

## Local sm_120 evidence (RTX 5060 Ti, cc 12.0, 102400 B smem/SM; 2026-06-14)
Ran `docs/flashinfer_pr/repro_d512_ragged.py` (synthetic ragged D512/VO256 fp8, qo sweep) on the
local sm_120 box with upstream flashinfer 0.6.12:
- **fp8 D512/VO256 dispatches OK for ALL qo {8,17,32,64,128,256}** — including qo>16 → cta_tile_q=64.
  So **the reject is NOT a general FA2-nvfp4/fp8 bug; it is GB10/Spark (sm_121, tighter usable smem)
  specific.** sm_120 consumer Blackwell (100 KB/SM) runs cta_tile_q=64 fp8 D512 fine.
- **cta_tile_q=16 also dispatches OK for fp8 D512** (qo=8 → OK) → the fix direction (force 16 on the
  tight-smem arch) is sound.

## CORRECTION to the original draft fix (a regression it would have caused)
The first draft returned `cta_tile_q=16` for 1-byte D512 *whenever 16 fits smem* — but 16 always
fits, so it would have **forced 16 even on sm_120 where 64 works and is faster** (a perf regression).
The fix must be **smem-conditional**: keep `cta_tile_q=64` when it yields a VALID 1-byte
`NUM_MMA_KV` (≥2 for NUM_WARPS_Q=4) given the SM's smem; only fall to 16 when 64 is invalid (GB10).
Concretely, replicate the dispatch's `max_num_mma_kv_smem` test for cta=64 and return 16 iff it is
`< kMinValidMmaKV`. This needs the dispatch's `kKVSmemPerMmaKV` accounting, which must be confirmed
against the GB10 build — do NOT hard-code a guessed step size.

> Status: **authored from source reading, NOT built or tested.** Must be built + verified on
> GB10/Spark (the E4B fp8 comparator at ctx 512 / prefix 256 must run, not reject; sm_120 must NOT
> regress — cta_tile_q stays 64) before it is claimed or filed.

## The bug (Codex mail 0136, confirmed against source)
For the E4B fp8 comparator, the FA2 D512/VO256 (`head_dim_qk=512, head_dim_vo=256`) 1-byte-KV
paged-prefill picks `cta_tile_q=64` → `NUM_WARPS_Q=4` → the only `NUM_MMA_KV` that fits smem is 1,
which is invalid for 1-byte KV (`NUM_MMA_KV*2 % NUM_WARPS_Q = 2 % 4 ≠ 0`). My existing dispatcher
patch (`0001-prefill-smem-diagnostic-1byte-kv.patch`) then *rejects* it with a clean error rather
than running it. The comparator needs it to RUN.

`cta_tile_q` comes from `FA2DetermineCtaTileQ(qo_len, head_dim_vo, head_dim_qk)` in
`include/flashinfer/utils.cuh`. Two reasons it returns 64 (invalid) instead of 16 (valid):
1. the `avg_packed_qo_len > 16 → return 64` branch (the comparator's suffix qo is ~256 > 16);
2. the `<= 16` branch's smem-fit check hard-codes `sizeof(dtype)=2` (bf16), so even there a 1-byte
   step is mis-sized.

At `cta_tile_q=16` → `NUM_WARPS_Q=2` → `NUM_MMA_KV=1` is valid (`1*2 % 2 = 0`), and the 1-byte 1x4
KV step fits: `(512+256)*16*4*1 + 16*512*2 = 49152 + 16384 = 65536 B < ~100 KB/SM`. So forcing
`cta_tile_q=16` for 1-byte-KV D512 on tight-smem CC 12.x makes the shape run (perf cost accepted —
correctness over throughput for a comparator baseline; it's run-or-reject).

## The fix (two parts)
### Part A — `FA2DetermineCtaTileQ` becomes KV-elem-size aware (`include/flashinfer/utils.cuh`)
Add a `kv_elem_size` param (default 2 = bf16, preserves all existing behavior) and, for CC≥8,
when `head_dim_qk > 256` (the VO-split signature) AND `kv_elem_size == 1`, prefer `cta_tile_q=16`
if the **1-byte** 1x4 KV step fits smem — BEFORE the `qo_len>16→64` branch. Also use `kv_elem_size`
(not a literal 2) in the existing `<=16`-branch smem check.

```cpp
inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len, uint32_t head_dim,
                                     uint32_t head_dim_qk = 0, uint32_t kv_elem_size = 2) {
  const uint32_t qk = head_dim_qk ? head_dim_qk : head_dim;
  if (avg_packed_qo_len > 64 && head_dim < 256) return 128;
  auto cc = GetCudaComputeCapability();
  if (cc.first < 8) return 64;  // Turing: no 1x4 layout
  // 1-byte KV (fp8/nvfp4) at the D>256 VO split: cta_tile_q=64 forces NUM_WARPS_Q=4, under which
  // the only smem-fitting NUM_MMA_KV=1 is invalid for 1-byte KV. cta_tile_q=16 (NUM_WARPS_Q=2)
  // makes NUM_MMA_KV=1 valid and the 1-byte step fits; prefer it so the shape runs at all.
  if (kv_elem_size == 1 && qk > 256) {
    int dev_id = 0, max_smem = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id);
    const uint32_t q_tile_smem = 16 * qk * 2;                       // Q stays 16-bit
    const uint32_t kv_step_1x4 = (qk + head_dim) * 16 * 4 * 1;      // 1-byte KV step
    if (q_tile_smem + kv_step_1x4 <= (uint32_t)max_smem) return 16;
  }
  if (avg_packed_qo_len > 16) return 64;
  // <=16 branch: use the real kv_elem_size in the smem check (was hard-coded 2)
  int dev_id = 0, max_smem = 0;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id);
  const uint32_t q_tile_smem = 16 * qk * 2;
  const uint32_t kv_step_1x4 = (qk + head_dim) * 16 * 4 * kv_elem_size;
  return (q_tile_smem + kv_step_1x4 > (uint32_t)max_smem) ? 64 : 16;
}
```

### Part B — thread `kv_elem_size` to the call sites
- `include/flashinfer/attention/prefill.cuh:1986` (`SinglePrefillWithKVCacheDispatched`, templated on
  `DTypeKV`): pass `sizeof(DTypeKV)`.
- `include/flashinfer/attention/scheduler.cuh:537,550` (`PrefillPlan`): `PrefillPlan` is NOT
  templated on the KV dtype — add a `uint32_t sizeof_dtype_kv` param (next to `sizeof_dtype_o`) and
  pass it to both `FA2DetermineCtaTileQ` calls. Thread it from the Python `plan()` binding
  (`flashinfer/prefill.py` → the JIT plan signature) where the KV dtype is known.
- Keep my existing reject as the genuine fallback: if even `cta_tile_q=16` doesn't fit (smaller
  smem arch), the `0001` diagnostic still fires with a clean message instead of the cryptic crash.

## Verification gate (must pass before claiming)
1. Build the edited FlashInfer (JIT) on a CC 12.x box.
2. E4B fp8 comparator at ctx 512 / prefix 256 **runs** (no reject, no `NUM_MMA_KV` crash); record NLL.
3. Sanity: bf16 and existing nvfp4 paths unchanged (kv_elem_size defaults preserve behavior).
4. Then mail Codex to rerun the SGLang fp8 comparator and update `0001` PR if needed.
