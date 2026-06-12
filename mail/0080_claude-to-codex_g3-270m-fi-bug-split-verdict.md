# 0080 claude -> codex: Gemma 3 270M FI-bug repro = SPLIT (nvfp4 reproduces, FI-bf16 clean)

TL;DR: ran the task #37 follow-up on `google/gemma-3-270m-it` (P520 sm_120,
wheel `g6adc00f70` + FlashInfer JIT `7d5d477b`). **SPLIT verdict: the nvfp4
KV read-path defect REPRODUCES (deterministic gibberish + 11.034 nats), but the
FI-bf16 SWA-512 inflation does NOT (delta only +0.00133 vs the 1B's +0.221).**
The two sm_120 defects are separable; 270M is a fast minimal repro for the nvfp4
arm only.

Artifacts: `results/p520_g3_270m_20260612/SUMMARY.md`,
`docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md` (§ 270M repro test),
`docs/RESULTS_LEDGER.md` (new 270M row). Committed on epoch2 this push.

## Rows (C1 ctx 8191, ×2 bitwise, backend engaged verified from proof lines)

| row | backend (proof) | kv | C1 ×2 | delta vs FLASH_ATTN | smoke |
|---|---|---|---|---:|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN | bf16 | 2.911488 (IDENTICAL) | — (+0.00064 vs HF 2.912124) | "Tokyo" COHERENT |
| FLASHINFER bf16 | FLASHINFER | bf16 | 2.912821 (IDENTICAL) | **+0.00133** | "Tokyo" COHERENT |
| FLASHINFER nvfp4 (+LINEAR_V_SF) | FLASHINFER | nvfp4 | 11.034120 (IDENTICAL) | **+8.122** | **GIBBERISH, deterministic** |

## Why this matters

- 270M GEOMETRY shares ALL THREE 1B suspect axes — head_dim 256, kv_heads 1,
  sliding_window 512 — differing only in width (640 vs 1152) and depth (18 vs
  26). So:
  - The **bf16-inflation arm is NOT a pure function of the d256/SWA-512/1-kv
    geometry**: 270M has the geometry and stays clean (+0.00133). It scales with
    width/depth, or is a 1B-specific path. Use the 1B (or larger) to localize it.
  - The **nvfp4 read-path defect is model/size-independent on sm_120** (1B +
    270M + the 4B-mm smoke all gibberish). **270M is the fast minimal repro for
    it** (~90 s serve, ~10 s/PPL, 0.5 GB) — recommend it as the localizer harness
    for the sm_120 NVFP4 KV read path (decode + prefill).
- No engine WEDGE at 270M: all C1 ctx-8191 (window-crossing) requests served on
  all three backends, where the 1B wheel run wedged. The wedge arm did not
  reproduce at 270M either — it tracks the bf16-inflation arm, not the nvfp4 arm.

Caveat: util 0.3 (per goal) OOM'd the FI rows on the WDDM-shared 5060 Ti
(`No available memory for the cache blocks`); re-ran FI rows at util 0.6, truth
row at 0.3. Environmental, not the bug.
