# Codex /goal, epoch 2 (rev 2, 2026-06-11)

Status update, 2026-06-14: this rev-2 goal is superseded for the active
SGLang lane by `docs/GOAL_CODEX_SGLANG_LANE.md`,
`docs/SGLANG_GEMMA4_AR_LADDER_PACKET_20260612.md`, and
`docs/SHIP_GATE_SGLANG_GEMMA4_LADDER_PLAN.md`. Do not use the older queue below
to justify rerunning known-red rows. Current blockers are:

- 12B long-context full-NVFP4 is a shared FlashInfer/numerics red, not an
  SGLang radix/partial-state-merge red (`mail/0138`).
- E4B fp8 is a FlashInfer D512/VO256 1-byte-KV paged-prefill dispatcher red,
  documented in
  `results/sglang_e4b_fp8_dispatch_analysis_20260614T054129JST/STOP_SUMMARY.md`.
- The SGLang baked mm-prefix image row is scoped green at
  `results/sglang_gemma4_e4b_fullnvfp4_mm_prefix_baked_20260614T072000JST/STOP_SUMMARY.md`.

Branch: `epoch2`. Plan: docs/CAMPAIGN_PLAN_EPOCH2.md. Mail: mail/ per
mail/README.md (note new numbering rule) - check at session start, every
stop point, and around box windows; coordinate by mail, never via Jetha.
Marker protocol unchanged. Evidence/provenance/guardrails as always.

Status: your lane items 1-3 are DONE (Gemma 4 E4B rungs 0+1 green incl.
the allocator fix; r9 built; DiffusionGemma feasibility study drafted).

Queue, in order:
1. ANSWER mail/0009_claude-to-codex_split-dtype-scope-question.md first -
   one paragraph or a code pointer on how your eager mixed-KV path reads
   K-fp8/V-fp4 at module level. It gates task 22's scope and your own
   graph-gate unpark.
2. Begin DG-S0 -> DG-S2 of your own feasibility ladder
   (docs/SGLANG_DIFFUSIONGEMMA_FEASIBILITY.md): model shell + config
   recognition + BF16 weight remap + geometry/weight-load manifests.
   Offline-heavy; take Spark windows only for the manifests that need a
   live load. NO NVFP4 or serving claims until BF16 matches the official
   vLLM image (your own rule - it is the right one).
3. The E4B fp8 comparator red: superseded diagnosis. It is now formally
   documented as a FlashInfer D512/VO256 1-byte-KV paged-prefill dispatcher
   red, not merely a missing-scale/timeout symptom. Hold the comparator until
   the FlashInfer dispatcher scheduling fix lands.
4. Opportunistic: keep RESULTS_LEDGER.md current; read Claude's DG-0/DG-1
   results mail when it arrives (the cache analysis feeds your DG-S3/S5).

Parked (unchanged): mixed-KV graph gate (until item 1 resolves task 22);
prefill.cuh kernel math (Claude's lane).
