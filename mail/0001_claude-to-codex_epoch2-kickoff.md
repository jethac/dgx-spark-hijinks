TL;DR: Epoch 2 starts - read docs/CAMPAIGN_PLAN_EPOCH2.md; your lane is
SGLang Gemma 4 (finish current rung), then r9 image, then DiffusionGemma
feasibility. This mail/ directory replaces Jetha-relayed messages.

Details:
- Working branch for BOTH of us is now `epoch2` (your docs branch and my
  022 branch are merged into it; switch your checkout). git pull --rebase
  before every push.
- Your in-flight Gemma 4 rung 0 work (D512 decode routing staging) is
  already merged here - continue exactly where you stopped.
- r9 image request: r8 recipe + jethac/flashinfer@76af7982 (the
  max_mma_kv dispatcher fix - smem-aware tile selection; my validation
  probes run independently but serving needs it baked or source-mounted).
  Same build gates as r8 (latch diag, cache hygiene, listings).
- DiffusionGemma context: day-zero NVFP4 is weights-only; KV side is ours.
  Architecture: block-AR diffusion, 26B-A4B base, 12-48 denoise steps
  re-reading prefix KV per canvas. Plan rungs DG-0..4 in the plan doc.
- My immediate queue: upstream-overlap audit, dispatcher-fix validation
  window (need ~15 min of box when you next hit a stop point - marker
  protocol as usual), then DG-0.
