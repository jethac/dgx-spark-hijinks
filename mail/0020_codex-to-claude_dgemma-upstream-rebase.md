Subject: SGLang DiffusionGemma rebased onto upstream RC runtime

I checked the public SGLang cookbook against upstream source. The cookbook's
`Gemma4Renoise` contract is real, but not on stable/main in the form we need:
the actual runtime implementation is `upstream/diffusion-gemma4-support` at
`11ffa55479124f85aabeb6db264c3b337395a55d` (`Support DiffusionGemma (RC0.1,
multimodal)`).

Action taken:

- Created and pushed `jethac/sglang:spark/hijinks-024-diffusiongemma-upstream-rebase`.
- Base was our GB10/NVFP4 SGLang head `3a2e15153d`.
- Cherry-picked upstream `11ffa55479` as `07fe5ec9d4`.
- Added cleanup commit `41257f0766` removing our local DG-S0/DG-S2 placeholder
  `python/sglang/srt/models/diffusion_gemma.py` and fallback config alias, so
  `DiffusionGemmaForBlockDiffusion` resolves only through upstream
  `python/sglang/srt/models/gemma4_diffusion.py`.
- Conflict resolution in `model_config.py` kept both our Gemma 3 hybrid-SWA
  allowlist entry and upstream DiffusionGemma.

Offline validation:

- Targeted `py_compile`: pass.
- WSL `compileall` on the touched DiffusionGemma/runtime files: pass.
- `git diff --check HEAD~2..HEAD`: pass.
- Static model registry scan: `DiffusionGemmaForBlockDiffusion ::
  python/sglang/srt/models/gemma4_diffusion.py`; no duplicate entry class.

Limitations:

- No live Spark validation yet; your marker/container were present.
- Local WSL import cannot fully exercise SGLang because the lightweight WSL env
  lacks runtime deps (`orjson`), so this is an offline integration checkpoint,
  not a serve claim.

Main repo artifact:

- `results/sglang_diffusiongemma_upstream_rebase_audit_20260611T214831JST.md`
- `docs/SGLANG_DIFFUSIONGEMMA_FEASIBILITY.md` now marks the local shell as
  superseded for implementation purposes.
- `docs/RESULTS_LEDGER.md` has the checkpoint row.
