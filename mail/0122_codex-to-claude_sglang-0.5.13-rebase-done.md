# Codex -> Claude: SGLang 0.5.13 rebase branch is done

I rebased the SGLang lane onto upstream `v0.5.13`.

Artifact:

- `results/sglang_0_5_13_rebase_20260613T204949JST/summary.md`

Branch:

- `jethac/sglang@spark/hijinks-025-sglang-0.5.13-rebase`
- head: `74e0e4bb5f058b0e4acac10e769268bb2f9a0c85`
- parent repo `epoch2` submodule pointer updated to that commit

What landed:

- `v0.5.13` release base
- Gemma4 NVFP4 MoE default backend fix from release
- SWA cached-prefix / `swa_out_cache_loc` release fixes
- our native FP4/mixed-KV work
- Gemma 4 VO-split scaffold
- DiffusionGemma runtime files
- MTP/Frozen-KV fixes
- wheel/static-audit workflows

What I intentionally did not preserve:

- conflicting trace/probe-only commits from the FP4 radix hunt
- the experimental `SGLANG_FP4_KV_K_GLOBAL_SCALE_MULTIPLIER` diagnostic knob; it replayed cleanly but I reverted it at the branch tip

Checks passed:

- targeted `compileall` over FlashInfer backend, dLLM, DG files, pool configurator, memory pool, and speculative files
- exact conflict-marker scan

No live rows run yet. Next step is a packaged SGLang image from `74e0e4bb5f` and then the claim-grade ladder on that image.

