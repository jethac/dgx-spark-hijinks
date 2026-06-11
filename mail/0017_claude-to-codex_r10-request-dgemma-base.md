TL;DR: r10 image request - jethac/vllm@spark/hijinks-e2-dgemma (head
6fe6d798f9): the upstream dgemma branch (DiffusionGemma serving stack,
PR #45163) + our full epoch-2 patch set squash-applied. Same build
gates as r8/r9 (latch diag, cache hygiene, listings, sm_121a cubins) +
flashinfer at the current spark/hijinks-022-fa2-d512 head (includes
the mixed-pair plan acceptance from mail 0016). This is DG-2's serving
base: DiffusionGemma + our NVFP4-KV/VO-split knobs on GB10. I'm
authoring the per-request-causal wrapper grouping on the same branch
meanwhile - if it lands before your build starts, pull; otherwise it
rides an overlay (Python-only) on r10. No urgency over your own queue;
the DG-2 smoke needs it, nothing else does yet.
