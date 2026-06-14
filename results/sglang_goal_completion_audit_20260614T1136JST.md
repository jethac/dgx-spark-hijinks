# SGLang Goal Completion Audit

Date: 2026-06-14 11:36 JST

Scope: current evidence audit for the active Codex SGLang/infra objective:
clear SGLang reds, finish the matched 12B / 26B-A4B / 31B AR ladder, keep
Claude coordination in `mail/`, and avoid known-red reruns at unchanged
dependencies.

## Current State

- Branch: `epoch2`
- Head at audit start: `e36578a2af8450f0e0bdc058fb3e31214a1121ba`
- Worktree at audit start: clean except unrelated untracked `docs/vast_anchor/trace_diff.sh`
- Latest incoming Claude mail: `mail/0140_claude-to-codex_PLUS040-is-a-kernel-artifact-true-cost-is-0.19.md`
- Latest Codex mail: `mail/0142_codex-to-claude_sglang-12b-chunked-prefill-2048-diagnostic-red.md`
- Lane poll status: `blocked-known-red-dependencies`
- Dependency refs unchanged from known blockers:
  - FlashInfer: `3fa0775cafaf88da5e0fc3b987afa6bd75d9510c`
  - SGLang: `f920e2d88af68031b745494f5435efb71ac93562`

## Requirement Audit

| Requirement | Current Status | Authoritative Evidence | Next Valid Action |
| --- | --- | --- | --- |
| Clear the 12B long-context full-NVFP4 red | **Blocked / not complete** | Matched SGLang row remains red at `+0.402969216038` nats/token in `results/sglang_gemma4_12b_ar_matched_bf16_fullnvfp4_ctx8185_prefix4096_20260613T153712JST/STOP_SUMMARY.md`. Mail 0140 reclassifies this as a shared FlashInfer single-/large-prefill accumulation artifact with exact/chunked reference cost near `+0.19`, not an SGLang radix/global-scale bug. | Wait for Claude's FlashInfer large-prefill accumulation fix, then rerun the matched 12B row at ctx `8185`, prefix `4096`; or run only a new explicitly scoped diagnostic if requested. |
| Test scoped chunking as a possible current-route workaround | **Diagnostic red / not unblocker** | `results/sglang_12b_chunked_merge_diag_20260614T112106JST/STOP_SUMMARY.md` ran full-NVFP4 only with `--chunked-prefill-size 2048`: readiness green, `Tokyo` smoke green, `cached_tokens=4096`, NLL `4.926853552137`, delta `+0.354863729534` vs banked bf16. This improves the old red by `0.048105486504` nats but does not approach the `~+0.19` reference. | Do not promote the diagnostic into a claim row. Further chunk diagnostics need a specific new hypothesis; otherwise wait for the FlashInfer fix. |
| Fix radix / partial-state merge after Claude's +0.40 verdict | **Retracted as the wrong target for this red** | `docs/GOAL_CODEX_SGLANG_LANE.md`, `docs/SGLANG_GEMMA4_AR_LADDER_PACKET_20260612.md`, and `docs/RESULTS_LEDGER.md` record that mail 0140 exonerates SGLang radix/merge and global-scale calibration for the `+0.40` class. | Do not rewrite radix/merge for this blocker unless a new SGLang-specific red appears after the FlashInfer fix or a scoped diagnostic contradicts mail 0140. |
| Finish matched 12B / 26B-A4B / 31B AR ladder | **Not complete** | `docs/SHIP_GATE_SGLANG_GEMMA4_LADDER_PLAN.md` lists 12B red, 26B-A4B not claim-grade, and 31B without a banked SGLang serving row. `scripts/run_sglang_gemma4_ar_ladder_pair.sh` is guarded against known-red rows at unchanged refs. | After the 12B quality and fp8 dispatcher blockers clear, run the packaged-image ladder with matched bf16 / fp8 / full-NVFP4 rows and claim audit artifacts. |
| Clear E4B fp8 comparator red | **Blocked / not complete** | `results/sglang_e4b_fp8_dispatch_analysis_20260614T054129JST/STOP_SUMMARY.md` classifies the red as FlashInfer D512/VO256 fp8 VO-split paged-prefill dispatcher selecting invalid `NUM_MMA_KV=1`; `scripts/sglang_e4b_fp8_dispatch_audit.py` recognizes the artifact. | Wait for the FlashInfer D512/VO256 fp8 dispatcher fix, then rerun E4B fp8 comparator and claim audit. |
| Wire FlashInfer mm-prefix masking into SGLang image path | **Scoped complete for E4B baked image; not broad ladder completion** | `results/sglang_gemma4_source_stack_image_27479559994/summary.md` records the packaged Ubuntu22/arm64 image carrying SGLang `f920e2d88a` and FlashInfer `f99323bd`. `results/sglang_gemma4_e4b_fullnvfp4_mm_prefix_baked_20260614T072000JST/STOP_SUMMARY.md` is green: mm-prefix proof log present, old Triton-only/bidirectional fallback warning absent, text/image/audio outputs stable. | Reuse the packaged image path for remaining Gemma 4 rows. Keep scope labels until matched AR quality/capacity gates pass. |
| Do not chase global-scale calibration | **Satisfied for current guard state** | Goal docs mark the global-scale root cause retracted; `bash scripts/test_sglang_gemma4_ar_ladder_guard.sh` includes `retracted_global_scale_refused` and `retracted_global_scale_diag_requires_reason`. | Keep refusing global-scale diagnostic env vars unless explicitly labeled with override reason. |
| Mail Claude on retractions / cross-lane blockers | **Satisfied for latest diagnostic** | `mail/0142_codex-to-claude_sglang-12b-chunked-prefill-2048-diagnostic-red.md` was committed and pushed with the chunked diagnostic result. | Send new mail only after a fix, rung, FlashInfer/vLLM-pointing red, or claim retraction. |
| Poll at session start and after banking | **Satisfied in this stop point** | `scripts/sglang_lane_state_poll.py` was run at session start and after banking; it reports no new remote mail, unchanged dependencies, and `blocked-known-red-dependencies`. | Continue running the poll before live rows and after banking cross-lane artifacts. |
| Preserve diagnostic provenance in artifacts | **Satisfied after tooling fix** | Commit `e36578a2af8450f0e0bdc058fb3e31214a1121ba` passes `SGLANG_AR_LADDER_OVERRIDE_REASON` into `sglang_gemma4_ar_ladder_blocker_audit.py`, so future known-blocked diagnostic artifacts record `diagnostic_override_allowed=true` and the explicit reason. | Keep using explicit override reasons for any scoped diagnostic replay. |

## Conclusion

The active objective is **not complete**. The lane is correctly guarded and the
latest chunked diagnostic is banked, but the broad SGLang Gemma 4 AR claim
remains blocked on unchanged external dependencies:

1. FlashInfer large-prefill accumulation fix for the 12B long-context quality red.
2. FlashInfer D512/VO256 fp8 dispatcher fix for the E4B comparator.
3. Subsequent matched 12B / 26B-A4B / 31B packaged-image ladder reruns.

Until either dependency changes or a new explicitly scoped diagnostic is
requested, rerunning the known-red rows would regenerate non-claim-grade
artifacts and should be avoided.
