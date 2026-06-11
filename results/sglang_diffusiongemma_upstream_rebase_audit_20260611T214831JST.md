# SGLang DiffusionGemma Upstream Rebase Audit

Date: 2026-06-11 21:48 JST

## Question

Does SGLang already have real DiffusionGemma support we should rebase on, rather
than continuing from the local DG-S0/DG-S2 foundation shell?

## Cookbook Contract

The public SGLang cookbook page for DiffusionGemma documents:

- `sglang serve --model-path google/diffusiongemma-26B-A4B-it`
- `--dllm-algorithm Gemma4Renoise`
- `--trust-remote-code`
- automatic runtime selection for `Gemma4Renoise`: Triton attention backend,
  eager mode, and unchunked prefill because full-attention head_dim is 512 and
  the denoise canvas uses bidirectional attention.

Source: https://docs.sglang.io/cookbook/autoregressive/Google/DiffusionGemma

## Upstream State

Fetched upstream SGLang branches from `https://github.com/sgl-project/sglang`.

Observed:

- `upstream/main` has the cookbook/documentation page, but the runtime path
  inspected earlier did not contain the complete `Gemma4Renoise` implementation.
- `upstream/diffusion-gemma4-support` contains the actual runtime code.

Relevant upstream commit:

- `11ffa55479124f85aabeb6db264c3b337395a55d`
- subject: `Support DiffusionGemma (RC0.1, multimodal)`

Runtime surfaces in that commit:

- `python/sglang/srt/dllm/algorithm/gemma4_renoise.py`
- `python/sglang/srt/models/gemma4_diffusion.py`
- `python/sglang/srt/multimodal/processors/diffusion_gemma.py`
- `DiffusionGemmaForBlockDiffusion` model/config registration
- scheduler request guards for unsupported `Gemma4Renoise` request features

Interpretation: the docs are ahead of a stable release. The support exists as an
upstream RC/integration branch, not as something proven available from the last
visible stable release.

## Local Integration

Created SGLang branch:

- `spark/hijinks-024-diffusiongemma-upstream-rebase`
- rebase head: `41257f07664a3983dfa9ea0625e3842a48e775e5`
- current head after config fix: `0705924c1d`

Base before integration:

- `origin/spark/hijinks-023-gemma4-fullnvfp4-denominator`
- `3a2e15153d87a0117b0685bb85545bf796b798ee`

Commits added:

- `07fe5ec9d4` cherry-picks upstream `11ffa55479`
- `41257f0766` removes the local foundation-shell duplicate and fallback config
- `0705924c1d` sets the missing `is_uniform=True` on the DiffusionGemma
  `DllmConfig.from_server_args()` branch

Conflict resolution:

- `python/sglang/srt/configs/model_config.py` had one hybrid-SWA registration
  conflict. Kept both local `Gemma3ForConditionalGeneration` and upstream
  `DiffusionGemmaForBlockDiffusion`.

Cleanup:

- Removed local shell `python/sglang/srt/models/diffusion_gemma.py`.
- Removed local fallback config `python/sglang/srt/configs/diffusion_gemma.py`
  and registry imports.
- Kept upstream runtime `python/sglang/srt/models/gemma4_diffusion.py`.
- Kept upstream `diffusion_gemma` model-type allowance in HF config loading,
  because the cookbook contract requires `--trust-remote-code`.

## Offline Validation

Commands/results:

- `python -m py_compile` on the upstream DiffusionGemma runtime files and
  touched scheduler/config files: pass.
- WSL `python3 -m compileall -q` on `python/sglang/srt/dllm`,
  `gemma4_diffusion.py`, `multimodal/processors/diffusion_gemma.py`,
  `model_config.py`, `scheduler.py`, and `forward_batch_info.py`: pass.
- `git diff --check HEAD~2..HEAD`: pass.
- Anchored conflict-marker scan: pass.
- Static AST registry scan over `python/sglang/srt/models/*.py` with
  `utf-8-sig` handling:
  - `DiffusionGemmaForBlockDiffusion :: python/sglang/srt/models/gemma4_diffusion.py`
  - no duplicate `EntryClass`
  - no syntax errors in scanned model files

Runtime import limitation:

- Windows import fails on SGLang's POSIX-only `resource` dependency.
- WSL import reaches package loading but local WSL Python lacks `orjson`.
- Therefore this audit does not claim a full runtime import, BF16 allocation,
  forward parity, or live serving result.

## Verdict

SGLang does have a real DiffusionGemma implementation, but it is on
`upstream/diffusion-gemma4-support`, not proven in the latest stable release.
Our lane should rebase on that implementation rather than maintain the local
foundation shell.

Next gate remains live Spark validation from this branch once the box is free:
load/serve with the cookbook contract, under GB10 memory guardrails.

Postscript: the first offline validation pass found and fixed one upstream RC
config bug: `DiffusionGemmaForBlockDiffusion` did not assign `is_uniform` before
constructing `DllmConfig`. See
`results/sglang_diffusiongemma_uniform_config_fix_20260611T2207JST.md`.
