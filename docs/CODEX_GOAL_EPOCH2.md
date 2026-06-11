# Codex /goal, epoch 2 (rev 2, 2026-06-11)

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
3. The E4B fp8 comparator red (no scaling factors -> 600s timeout):
   diagnose or formally document as a known-red with a tracked issue -
   it blocks fp8 quality baselines for the whole E4B family.
4. Opportunistic: keep RESULTS_LEDGER.md current; read Claude's DG-0/DG-1
   results mail when it arrives (the cache analysis feeds your DG-S3/S5).

Parked (unchanged): mixed-KV graph gate (until item 1 resolves task 22);
prefill.cuh kernel math (Claude's lane).
