# SGLang Goal Completion Audit

Date: 2026-06-14 10:57 JST

Scope: current evidence audit for the active Codex SGLang/infra objective:
clear SGLang reds, finish the matched 12B / 26B-A4B / 31B AR ladder, keep
Claude coordination in `mail/`, and do not rerun known-red rows at unchanged
dependencies.

## Current State

- Branch: `epoch2`
- Head at audit start: `eab7ed669cd913d37048536c918abc0e8d2602eb`
- Worktree at audit start: clean
- Latest incoming Claude mail: `mail/0140_claude-to-codex_PLUS040-is-a-kernel-artifact-true-cost-is-0.19.md`
- Latest Codex reply: `mail/0141_codex-to-claude_sglang-0140-verdict-banked.md`
- Lane poll status: `blocked-known-red-dependencies`
- Dependency refs unchanged from known blockers:
  - FlashInfer: `3fa0775cafaf88da5e0fc3b987afa6bd75d9510c`
  - SGLang: `f920e2d88af68031b745494f5435efb71ac93562`

## Requirement Audit

| Requirement | Current Status | Authoritative Evidence | Next Valid Action |
| --- | --- | --- | --- |
| Clear the 12B long-context full-NVFP4 red | **Blocked / not complete** | Matched SGLang row remains red at `+0.4029692160381897` nats/token in `results/sglang_gemma4_12b_ar_matched_bf16_fullnvfp4_ctx8185_prefix4096_20260613T153712JST/STOP_SUMMARY.md`. Mail 0140 reclassifies this as a shared FlashInfer single-/large-prefill accumulation artifact with exact/chunked reference cost near `+0.19`, not an SGLang radix/global-scale bug. | Wait for Claude's FlashInfer large-prefill accumulation fix, then rerun the matched 12B row at ctx `8185`, prefix `4096`; or run only an explicitly scoped chunked/merge diagnostic if requested. |
| Fix radix / partial-state merge after Claude's +0.40 verdict | **Retracted as the wrong target for this red** | `docs/GOAL_CODEX_SGLANG_LANE.md`, `docs/SGLANG_GEMMA4_AR_LADDER_PACKET_20260612.md`, and `docs/RESULTS_LEDGER.md` now record that mail 0140 exonerates SGLang radix/merge and global-scale calibration for the `+0.40` class. | Do not rewrite radix/merge for this blocker unless a new SGLang-specific red appears after the FlashInfer fix or a scoped diagnostic contradicts mail 0140. |
| Finish matched 12B / 26B-A4B / 31B AR ladder | **Not complete** | `docs/SHIP_GATE_SGLANG_GEMMA4_LADDER_PLAN.md` lists 12B red, 26B-A4B not claim-grade, and 31B without a banked SGLang serving row. `scripts/run_sglang_gemma4_ar_ladder_pair.sh` is guarded against known-red rows at unchanged refs. | After the 12B quality and fp8 dispatcher blockers clear, run the packaged-image ladder with matched bf16 / fp8 / full-NVFP4 rows and claim audit artifacts. |
| Clear E4B fp8 comparator red | **Blocked / not complete** | `results/sglang_e4b_fp8_dispatch_analysis_20260614T054129JST/STOP_SUMMARY.md` classifies the red as FlashInfer D512/VO256 fp8 VO-split paged-prefill dispatcher selecting invalid `NUM_MMA_KV=1`; `scripts/sglang_e4b_fp8_dispatch_audit.py` recognizes the artifact. | Wait for the FlashInfer D512/VO256 fp8 dispatcher fix, then rerun E4B fp8 comparator and claim audit. |
| Wire FlashInfer mm-prefix masking into SGLang image path | **Scoped complete for E4B baked image; not a broad ladder completion** | `results/sglang_gemma4_source_stack_image_27479559994/summary.md` records the packaged Ubuntu22/arm64 image carrying SGLang `f920e2d88a` and FlashInfer `f99323bd`. `results/sglang_gemma4_e4b_fullnvfp4_mm_prefix_baked_20260614T072000JST/STOP_SUMMARY.md` is green: mm-prefix proof log present, old Triton-only/bidirectional fallback warning absent, text/image/audio outputs stable. | Reuse the packaged image path for remaining Gemma 4 rows. Keep scope labels until matched AR quality/capacity gates pass. |
| Do not chase global-scale calibration | **Satisfied for current guard state** | Goal docs mark the global-scale root cause retracted; `bash scripts/test_sglang_gemma4_ar_ladder_guard.sh` includes `retracted_global_scale_refused` and `retracted_global_scale_diag_requires_reason`. | Keep refusing global-scale diagnostic env vars unless explicitly labeled with override reason. |
| Mail Claude on retractions / cross-lane blockers | **Satisfied for mail 0140 reclassification** | `mail/0141_codex-to-claude_sglang-0140-verdict-banked.md` was committed and pushed, acknowledging the SGLang docs/guards update and current hold state. | Send new mail only after a fix, rung, FlashInfer/vLLM-pointing red, or claim retraction. |
| Poll at session start and after banking | **Satisfied in this stop point** | `scripts/sglang_lane_state_poll.py` was run at session start and after mail banking; both reported clean tree, no new remote mail, unchanged dependencies, and `blocked-known-red-dependencies`. | Continue running the poll before live rows and after banking cross-lane artifacts. |

## Conclusion

The active objective is **not complete**. The SGLang lane is cleanly documented
and guarded, with one scoped green multimodal image-path row, but the broad
SGLang NVFP4 claim remains blocked on external FlashInfer work:

1. 12B long-context large-prefill accumulation fix.
2. E4B D512/VO256 fp8 dispatcher fix.
3. Subsequent matched 12B / 26B-A4B / 31B packaged-image ladder reruns.

Until either dependency changes or an explicitly scoped chunked/merge diagnostic
is requested, rerunning the known-red rows would only regenerate the same red
artifacts and should be avoided.
