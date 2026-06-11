# Codex /goal, epoch 2

Branch: `epoch2` (switch your checkout; `git pull --rebase` before every
push). Read first: docs/CAMPAIGN_PLAN_EPOCH2.md, then your mailbox -
mail/0001_claude-to-codex_epoch2-kickoff.md. Check mail/ at session start,
every stop point, and around box windows; reply by mail file, never via
Jetha.

Your lane, in order:
1. FINISH SGLang Gemma 4 rung 0 -> rung 1 (your in-flight work - D512
   decode routing staging - is already merged into epoch2; gates and
   parked items per docs/CODEX_GOAL_SGLANG_GEMMA4_RUNGS.md, which remains
   valid for this item).
2. r9 image: r8 recipe + jethac/flashinfer@76af7982 (smem-aware tile
   selection - fixes the max_mma_kv=0 dispatch crash). Same verification
   gates as r8: latch diag, module-cache hygiene, listings. Announce the
   tag by mail.
3. SGLang DiffusionGemma feasibility study: day-zero support landed in
   vLLM only. Scope what SGLang serving would take (block-AR diffusion
   scheduling, canvas attention, prefix-cache semantics). Code-anchored
   doc, no implementation yet - your Gemma 4 prep doc is the model.
4. When Claude lands split-dtype module keying (mail will announce):
   unpark your mixed-KV CUDA-graph gate.

Protocols unchanged: marker = Claude holds the Spark; evidence discipline,
memory guardrails, binary provenance (md5 + latch), stop-on-repeated-error.
