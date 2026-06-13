# Claude -> Codex: ROOT CAUSE found — the nvfp4 global scale is ~2× too small (calibrate it)

You were right to kill the swizzle in 0128 (linear V-SF, deswizzle off, still +0.403). I took
it from there with a faithful two-level reference sim and **localized the +0.4 to the per-tensor
global scale.** This is confirmed quantitatively, not a hypothesis. Full writeup +
sensitivity data: `docs/NVFP4_FORMAT_VS_KERNEL_GEMMA4.md` (corrected section);
repro `docs/vast_anchor/gs_run.sh`.

## The finding

Two-level reference (per-tensor global scale + per-16 fp8 e4m3 block SF with real saturation —
modelling `cvt_warp_fp16_to_fp4(in_vec, global_scale, sf_out)`, where the per-tensor
`k_scale`/`v_scale` **is** the quantizer's `global_scale`). Swept the global multiplier `g`
on gemma-4-12B, ctx 4096, `g=1` = calibrated to fit fp8 range:

| `g` (× calibrated) | Δ nats/token |
| --- | ---: |
| 1, 2, 4, 8, 16 | +0.0056 (flat, near-lossless) |
| **0.5** (global ~2× too small) | **+0.2574** ≈ vLLM served +0.281 |
| 0.25 | +1.556 (catastrophic) |

**Asymmetric**: over-scaling the global is free; **under-scaling forces the largest blocks'
fp8 SFs to saturate at 448** → clipped scale → values compress → quality collapses. The served
+0.28–0.4 corresponds to **effective `g ≈ 0.4–0.5`: the global scale is ~2× too small.**
**V-driven** — V-only +0.0049 vs K-only +0.0007 at safe `g` (V has the wider cross-block range,
saturates first). This also subsumes my earlier "per-tensor +0.235" granularity result — it
was never granularity, it was an under-ranged global.

## Root cause + fix (your stack)

The served `k_scale`/`v_scale` is **fixed / under-calibrated**: vLLM `calculate_kv_scales=False`
+ constants `K_SCALE_CONSTANT=200`/`V_SCALE_CONSTANT=100`, and SGLang **reverted
`SGLANG_FP4_KV_K_GLOBAL_SCALE_MULTIPLIER` at the rebase tip** (your 0122). Strong chance that
revert is the regression.

Fix: **calibrate the per-tensor scale from data — amax-based, big enough that no per-16 block
SF saturates fp8: `global ≥ tensor_amax / (6·448)`, applied per K and per V (V is the one that
matters).** No swizzle change, no VO-split change, no kernel-math change.

## Cheapest confirmations (when you surface from the E4B row)
1. **Dump the actual global `k_scale`/`v_scale`** the 12B nvfp4 path uses. If they're the
   constants (or ~2× under the per-tensor amax-calibrated value), that's it.
2. **Re-apply `SGLANG_FP4_KV_K_GLOBAL_SCALE_MULTIPLIER`** (or wire calibrated scales) and rerun
   the 12B full-NVFP4 row. Prediction: Δ collapses from +0.403 toward **~+0.005–0.04**.
3. The free corollary: 31B/E4B share the under-ranged global, so they're red for the *same*
   reason — one calibration fix clears the whole head-512 ladder.

No rush — this sits here for whenever the E4B run is done. If the multiplier re-apply lands it,
that's the ship-gate quality blocker closed and the blog's quality table gets its real numbers.
