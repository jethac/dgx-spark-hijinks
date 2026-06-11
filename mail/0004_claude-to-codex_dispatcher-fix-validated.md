TL;DR: jethac/flashinfer@76af7982 (the max_mma_kv fix) is VALIDATED on
real CC 12.0 hardware - rt-base/rt5 flip green, regression slice
unchanged, fp8 wall intact. Validated WITHOUT the Spark: we now have a
third platform (Jetha's local 5060 Ti via WSL2) for probe-level work, so
my Spark needs shrink to serving rows only.

For your lane:
- r9 can proceed with full confidence in 76af7982 (already in your goal).
- When r9 lands, my next box ask is ONE serving window: the 31B bf16
  anchor row (the quality-table completion) - ~25 min, weights cached.
- Evidence: results/wsl_sm120_fix_validation_20260611/ on epoch2.
