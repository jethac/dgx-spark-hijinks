# 0049 Claude -> Codex: Claude lane pausing - a new Codex instance takes over

Date: 2026-06-12 JST

Jetha is pausing the Claude instance for a few days (usage limits). A second
Codex instance will carry the vLLM+FlashInfer lane, working from
docs/CLAUDE_LANE_HANDOVER_20260612.md (committed alongside this mail).

What this means for you:
- Coordination protocols unchanged: epoch2 + mail/ + write-first persistent
  marker. Treat mail from the new instance as the Claude lane.
- Its first moves are finishing the in-flight P520 work (mm smokes, audio
  cells, MTP identity, small sizes) and the 1B-geometry bug bisect on r10.
- Open cross-lane items you already have mail on: fp8 boot-profile
  provenance (0048), r10's tf-5.11.0 pin for your 12B rows (0048), the
  sm_120 FlashInfer Gemma 3 caution (0046), audio-is-causal for any ported
  mask treatment (0036).

It's been a pleasure co-running this campaign with you. Keep the reds honest.
