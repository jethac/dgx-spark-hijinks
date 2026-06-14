# DRAFT (UNVERIFIED) — E4B fp8 D512/VO256 dispatcher fix (mail 0136 / 0140)

> Status: **authored from source reading, NOT built or tested.** Must be built + verified on a
> CC 12.x box (the E4B fp8 comparator at ctx 512 / prefix 256 must run, not reject) before it is
> claimed or filed. Do not promote to a real PR until green. This exists to make the next box
> session fast.

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
