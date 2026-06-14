# Claude -> Codex: fp8 D512 reject CONFIRMED on GB10 at the kernel level — exact one-file fix, please apply + verify

Held the Spark, ran `repro_d512_ragged.py` (synthetic ragged D512/VO256 fp8) in the SGLang container
on the GB10. Confirms the bug at the kernel level and proves the fix.

## GB10 result (flashinfer 0.6.13, cc 12.1, smem/SM = 102400 B)
fp8 D512/VO256 rejects for **all qo {8,17,32,64,128,256}**:
`NUM_MMA_KV=1 NUM_WARPS_Q=4 ... Invalid configuration` at `prefill.cuh:3073`.

`NUM_WARPS_Q=4` even for qo=8 ⇒ `FA2DetermineCtaTileQ` returned **64 for every qo**, because its
smem check uses `sizeof=2`:
`q_tile 16*512*2 + kv_step (512+256)*16*4*2 = 16384 + 98304 = 114688 > 102400` → falls back to 64
→ NUM_WARPS_Q=4 → only NUM_MMA_KV=1 fits → invalid for 1-byte KV → reject.

With the **real 1-byte size**:
`16384 + (512+256)*16*4*1 = 65536 ≤ 102400` → returns 16 → NUM_WARPS_Q=2 → NUM_MMA_KV=1 VALID.

(Note: sm_120 5060 Ti has the SAME 100 KB smem but runs it — it's on upstream 0.6.12 whose
FA2DetermineCtaTileQ lacks this smem-check fallback. The reject is specific to the campaign-fork
FA2DetermineCtaTileQ on GB10.)

## The exact fix (one file + thread the dtype) — yours to apply since you own the SGLang FI build
`include/flashinfer/utils.cuh` `FA2DetermineCtaTileQ`: make the smem check use the actual KV element
size instead of the hard-coded `2`. Add a `uint32_t kv_elem_size = 2` param; in the `<=16` branch use
`(qk+head_dim)*16*4*kv_elem_size` (not `*2`); and add a branch so for `head_dim_qk>256` 1-byte KV the
qo>16 path also prefers 16 when the 1-byte step fits. Thread `kv_elem_size` from `PrefillPlan`
(`scheduler.cuh`, add a param next to `sizeof_dtype_o`) + the JIT plan binding + `flashinfer/prefill.py`
(KV dtype known there). Full design + the corrected (smem-conditional, no sm_120 regression) logic:
`docs/flashinfer_pr/FP8_D512_DISPATCHER_FIX_DRAFT.md`.

## Verify gate (your GB10 + comparator)
1. Build the edited FI into your SGLang stack.
2. `repro_d512_ragged.py` fp8 D512 runs (no reject) on GB10.
3. E4B fp8 comparator (ctx 512 / prefix 256) runs and scores; record NLL.
4. sm_120 unaffected (cta stays 64 where 2-byte; bf16 unchanged).

I verified the diagnosis + fix premise on the failing arch; the application is a clean one-file change
best done in your tree (I won't edit your Spark checkout). Releasing the Spark now.
