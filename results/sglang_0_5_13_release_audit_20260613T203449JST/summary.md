# SGLang 0.5.13 Release Audit

Date: 2026-06-13 20:34 JST

Scope: audit whether the SGLang lane should rebase after upstream released `v0.5.13`.

## Inputs

- Local SGLang lane: `third_party/sglang` branch `spark/hijinks-024-diffusiongemma-upstream-rebase`
- Local SGLang commit: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- Upstream tag: `v0.5.13`
- Tag commit: `28b095c01005d4a3a2a5b637b7d028b07fba31b2`
- Merge base: `02be2e71899491b7aaf2849dce6431f61fc190b6`
- Ahead/behind: current branch is `64` commits ahead and `80` commits behind `v0.5.13`

Commands:

```text
git fetch upstream --tags
git rev-list --left-right --count HEAD...v0.5.13
git merge-base HEAD v0.5.13
git diff --name-status HEAD..v0.5.13 -- python/sglang/srt
```

## Finding

We should rebase, but not by replacing our lane with the release tag.

`v0.5.13` contains release fixes directly relevant to our Gemma/SGLang work:

- `4ac66f30f0` - `Fix Gemma4 NVFP4 MoE default attention backend`
- `12e28bdf0c` - `Fix FlashInfer SWA EXTEND-with-prefix correctness in merge_state path`
- `70a39d8f10` - `[SWA] Cache full->SWA out_cache_loc per forward across attention backends`

Our branch already contains `12e28bdf0c` by ancestry, but does not contain `4ac66f30f0` or `70a39d8f10`.

The release tag does **not** contain the DiffusionGemma runtime files currently needed by this lane:

```text
git ls-tree -r --name-only v0.5.13 python/sglang/srt | Select-String "diffusion_gemma|gemma4_diffusion|gemma4_renoise"
# no matches
```

The upstream DiffusionGemma support branch does contain them:

```text
python/sglang/srt/dllm/algorithm/gemma4_renoise.py
python/sglang/srt/models/gemma4_diffusion.py
python/sglang/srt/multimodal/processors/diffusion_gemma.py
```

So a direct checkout/rebase to `v0.5.13` would drop active DiffusionGemma work:

```text
D python/sglang/srt/configs/diffusion_gemma.py
D python/sglang/srt/dllm/algorithm/gemma4_renoise.py
D python/sglang/srt/models/gemma4_diffusion.py
```

## High-Risk Overlap

The release touches the exact files where our lane has local changes and validation history:

```text
python/sglang/srt/layers/attention/flashinfer_backend.py
python/sglang/srt/mem_cache/memory_pool.py
python/sglang/srt/mem_cache/swa_memory_pool.py
python/sglang/srt/model_executor/pool_configurator.py
python/sglang/srt/server_args.py
python/sglang/srt/models/gemma4_vision.py
```

The `HEAD..v0.5.13` diff for the highest-risk files includes:

```text
flashinfer_backend.py     | 2725 ++------------------
swa_memory_pool.py        |   27 +-
pool_configurator.py      |  132 +-
server_args.py            |  163 +-
gemma4_vision.py          |   15 +-
```

This is not safe to absorb as an unreviewed submodule bump.

## Recommendation

Create a new SGLang rebase branch rather than mutating the current validated branch in-place:

```text
spark/hijinks-025-sglang-0.5.13-rebase
```

Recommended order:

1. Start from upstream `v0.5.13`.
2. Replay or rebase the upstream DiffusionGemma support branch content that is not in the tag.
3. Replay our lane-specific patches:
   - Spark-packaged NVFP4/Mixed-KV work.
   - Gemma 4 / DiffusionGemma config and loader fixes.
   - MTP/Frozen-KV patches.
   - pool configurator and graph-gate fixes.
4. Explicitly reconcile release commits `4ac66f30f0` and `70a39d8f10` against our local Gemma/SWA code.
5. Rebuild a packaged SGLang image; do not use loose-wheel injection.
6. Rerun the claim-grade gates on the packaged image:
   - static import/config audit,
   - E2B / 12B / 26B-A4B / 31B ladder rows as applicable,
   - DiffusionGemma DG-S0->DG-S2,
   - fp8 comparator red diagnosis,
   - CUDA-graph gate,
   - MTP identity/no-leak gates.

## Day-0 Support Note

The public cookbook can be true for an upstream branch or docs-main workflow while still not being true for the `v0.5.13` tag. Based on the tag contents audited here, `v0.5.13` alone does not ship the DiffusionGemma runtime files our lane needs.

Any claim should say:

> SGLang `v0.5.13` has relevant Gemma/SWA fixes, but DiffusionGemma support is still carried by upstream support-branch content plus our local integration branch until a release tag contains those runtime files.

