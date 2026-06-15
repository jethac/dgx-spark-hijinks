# /goal — Claude (vLLM + FlashInfer lane), epoch-3

**North star (the ship bar, Jetha's words):** *we cannot call this campaign successful until Gemma 3 +
Gemma 4 (including 26B-A4B) + DiffusionGemma all ship with a working NVFP4 KV cache* — on the epoch-3 stack
(vLLM `v0.23.0` + FlashInfer `main`). The one thing blocking that is the **26B-A4B nvfp4 kernel bug**.
Because DiffusionGemma *is* the 26B-A4B base, fixing 26B clears both.

State at handoff: the epoch-3 stack is THREE rebases — **vLLM→v0.23.0 (done, mine)**, **FlashInfer→main
(done, mine)**, **SGLang→latest (Codex's, NOT yet started; `docs/GOAL_CODEX_E3_SGLANG.md`)**. My two are
DONE + validated (12B/31B serve green, bitwise-identical to pre-rebase; `docs/REBASE_E3_STATUS.md`). e3 wheel
rebuilding on `hijinks-build-x64` (148.251.32.85). The FlashInfer e3 ref `spark/hijinks-e3-flashinfer` is the
ONE shared kernel both stacks pin — any 26B fix I land there feeds Codex's SGLang e3 stack too. All my-lane
red rows were already resolved; 26B nvfp4 is the residual ship-gate blocker.

## Priority 1 — resolve 26B-A4B nvfp4 on the e3 stack (THE ship gate)
1. **Re-confirm 26B nvfp4 on e3 first (cheap, do this before deep work).** Rent one vast PRO-6000, install
   the e3 wheel + FlashInfer-e3 (JIT), run the matched anchor: bf16 vs nvfp4 (FIXSCALE/calib) at ctx 8185.
   - If FlashInfer-`main` happens to fix it (nvfp4 ≈ bf16, coherent) → **ship gate substantially clears**;
     re-run the full 12B/31B/26B ladder, bank green, update SOLUTIONS_STATUS + the blog 26B chapter.
   - If still broken (nvfp4 below truth, degenerate at some scales) → go to step 2.
2. **Resume the read-capture (harness banked: `docs/vast_anchor/sitecustomize.py`).** Capture the layer-0
   FlashInfer nvfp4 attention inputs+output (26B + 12B control), dequant the pages with
   `scripts/nvfp4_writer_roundtrip_probe.py` layout, compare to a dequant+SDPA reference. Verdict:
   - 26B FlashInfer output ≠ reference while 12B matches → **reader math bug** (localize the kernel site in
     FlashInfer-e3, write the fix, rebuild, re-verify → full-nvfp4 26B on both stacks).
   - both match reference → the bias is outside reader math (fusion/feed/post-attention) → trace that.
   Coordinate with Codex (he asked for this capture in 0182; he does the SGLang-side reader instrumentation).
3. When 26B nvfp4 is fixed in FlashInfer-e3, re-verify 12B/31B unaffected, re-run DiffusionGemma nvfp4 KV
   (truth-gated, not coherence-only), and update the ship-gate rows.

## Priority 2 — migrate the campaign to e3 (so the rebase "ships")
- Confirm the e3 wheel rebuild finished on `hijinks-build-x64`; publish it as a `jethac/vllm` release
  (the box has no `gh` — install it or use the API with a token, or have the CI `build-sm120a-wheel.yml`
  workflow_dispatch on `spark/hijinks-e3-vllm` produce the release).
- Rebuild the Spark image off e3 (`build-spark-image.yml`); point the Colab G4 notebook + `e2e_setup_wget.sh`
  at the e3 wheel + `spark/hijinks-e3-flashinfer`.
- Re-run the full matched claim ladder (12B/31B/26B) on the e3 wheel and re-bank the headline numbers.

## Priority 3 — blog truth-pass (after 26B resolves)
- `B:\jethac.github.io\collections\_posts\2026-06-XX-nvfp4-kv-cache-support.md`: the "+0.28" section credits
  only the V-SF swizzle fix — calibration of the global scale was the dominant other half; reconcile.
  DiffusionGemma "green" vs 26B-A4B-it broken — both are the 26B base; truth-gate DiffusionGemma before both
  claims sit in the post. The 26B chapter is added; update its verdict once 26B nvfp4 is fixed-or-confirmed.

## Constraints / hygiene (do not drift)
- vast `VAST_API_KEY` + `HF_TOKEN` env-only, never to a file. Print $/hr before renting; **destroy every box
  on bank** ($0 burn between tasks). Re-rent if a box's SSH proxy is flaky (rent+test, destroy, retry).
- Spark: key-based only; the password (Ivana123) never goes in any file; respect the `CLAUDE_WINDOW_OPEN`
  marker; never touch Spark while Codex holds it.
- Commit only relevant changes; never commit `jethac.github.io`. End commits with the Co-Authored-By line.
  Push e3 branch fixes to `jethac/vllm@spark/hijinks-e3-vllm` and `jethac/flashinfer@spark/hijinks-e3-flashinfer`.
- Coordinate with Codex via `mail/` (odd=claude / even=codex). The 26B kernel is the shared blocker.

## Done =
26B-A4B nvfp4 serves green on the e3 vLLM stack (matched anchor near-lossless, coherent) **or** is a
precisely-localized FlashInfer kernel fix landed in `spark/hijinks-e3-flashinfer` and verified; DiffusionGemma
nvfp4 truth-gated; campaign migrated to the e3 wheel; ship-gate rows updated.
