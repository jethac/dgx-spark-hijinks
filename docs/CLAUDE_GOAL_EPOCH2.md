# Claude /goal, epoch 2

Branch: `epoch2` (single working branch for both agents; pull --rebase
before push). Plan: docs/CAMPAIGN_PLAN_EPOCH2.md. Mail: mail/ protocol per
mail/README.md - check at session start, stop points, box windows.

Claude lane, in order:
1. Upstream-overlap audit (task 23): vLLM PR #40082 + 0.19 diffs near our
   patched files -> collision report + rebase plan. DG-0 wants a
   current-vLLM base.
2. Dispatcher-fix validation window (task 17, ~15 min box): rt-base/rt5
   flip green on jethac/flashinfer@76af7982; A1/A2+geometry regression
   slice unchanged. Then, behind it: bf16 anchor row -> three-way quality
   table -> Triton-retirement benchmarks (epoch-1 ladder debt, still owed).
3. DG-0 baseline + DG-1 cache analysis (tasks 24+): day-zero DiffusionGemma
   on Spark vLLM; measured dispatch geometry; canvas/prefix cache
   implementation notes; where our NVFP4 KV writer/reader engages.
4. FlashInfer custom-mask enablement (canvas masking, ex-mm-prefix) and
   split-dtype module keying (task 22 - announce to Codex by mail when
   landed).
Standing: Colab lane support (wheel built; F3/F4 = the vllm#40677 demo);
fp8 1-byte guard term (low); M4 (low).

Protocols: marker for the Spark; agent mail for coordination; evidence/
provenance/memory guardrails per epoch 1; blog gated on full ladder.
