Subject: SGLang DiffusionGemma uniform-config fix + re-scoped ladder

I took your env pointer and re-ran validation in `~/sglang_env` with the
B-backed SGLang source on `PYTHONPATH`.

Finding: upstream RC had a config bug in `DllmConfig.from_server_args()`:
the `DiffusionGemmaForBlockDiffusion` branch set `block_size` and `mask_id` but
never assigned `is_uniform` before constructing `DllmConfig`.

Fix:

- `jethac/sglang:spark/hijinks-024-diffusiongemma-upstream-rebase`
- new head: `0705924c1d`
- patch: set `params = DLLM_PARAMS[arch]` and `is_uniform = params["is_uniform"]`
  in the DiffusionGemma branch.

Validation:

- Windows `py_compile` on `python/sglang/srt/dllm/config.py`: pass.
- WSL `~/sglang_env` monkey-patched `ModelConfig.from_server_args()` to return
  a minimal `DiffusionGemmaForBlockDiffusion` HF config.
- Observed:
  - `algorithm Gemma4Renoise`
  - `block_size 256`
  - `mask_id 0`
  - `max_running_requests 1`
  - `is_uniform True`

Docs/artifacts added:

- `docs/SGLANG_DIFFUSIONGEMMA_RUNTIME_LADDER_EPOCH2.md`
- `results/sglang_diffusiongemma_uniform_config_fix_20260611T2207JST.md`
- updated the upstream rebase audit, feasibility doc, and results ledger.

Note: the full model-registry import probe remains unreliable in my local WSL
session because an earlier `git status` against the B mount is stuck in
uninterruptible I/O. I did not use that as evidence. The previous static AST
registry scan still proves a single `DiffusionGemmaForBlockDiffusion` EntryClass.

Spark was still marked busy when I checked, so no live serve/load claim here.
