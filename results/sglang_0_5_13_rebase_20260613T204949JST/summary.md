# SGLang 0.5.13 Rebase Stop Point

Date: 2026-06-13 20:49 JST

## Branches

- Parent repo branch: `epoch2`
- SGLang source branch: `spark/hijinks-025-sglang-0.5.13-rebase`
- SGLang source commit: `74e0e4bb5f058b0e4acac10e769268bb2f9a0c85`
- Base release tag: `v0.5.13`
- Base release commit: `28b095c01005d4a3a2a5b637b7d028b07fba31b2`

## Outcome

Rebase completed and pushed to `jethac/sglang`.

The new branch is based on upstream `v0.5.13` and carries the SGLang lane work forward:

- native FP4 KV / mixed-KV plumbing,
- Gemma 4 FlashInfer VO-split scaffold,
- DiffusionGemma runtime support,
- DiffusionGemma static audits and wheel workflows,
- Frozen-KV MTP static/live fixes.

The parent repo now points `third_party/sglang` at the rebased branch head.

## Release Fixes Absorbed

The rebased branch contains the relevant `v0.5.13` release fixes:

- `4ac66f30f0` - Gemma4 NVFP4 MoE default attention backend fix.
- `12e28bdf0c` - FlashInfer SWA EXTEND-with-prefix correctness in merge_state path.
- `70a39d8f10` - per-forward SWA `out_cache_loc` plumbing across attention backends.

## Conflict Policy

Runtime patches were preserved. Stale trace/probe commits that conflicted with the release refactor were dropped rather than forced through:

- `a8ad6a3ac9` - dense quant attention trace,
- `a47111fcaf` - dense FP4 attention soft-cap trace,
- `8b95253af1` - K-only/V-only attention trace,
- `dfd4264424` - K global scale policy trace,
- `d6fa9d1047` - per-head K scale policy trace,
- `5b71bef3c0` - reuse partials vs dense reference trace.

The experimental K-global-scale multiplier commit replayed cleanly but was then explicitly reverted:

- replayed: `96e23f7c45`
- reverted by: `74e0e4bb5f`

Reason: that scale knob was a diagnostic dead end and should not survive into a claim-grade release branch.

## Manual Resolutions

Important manual conflict resolutions:

- `flashinfer_backend.py`: kept release SWA `swa_out_cache_loc` behavior and local native FP4/mixed-KV/VO-split/MTP helper logic.
- `dllm/mixin/req.py`: merged release `full_untruncated_fill_ids` semantics with DiffusionGemma uniform-canvas dLLM staging.
- `memory_pool.py`: kept mixed-KV helpers and removed the experimental global K-scale multiplier.

## Checks

Passed:

```text
python -m compileall -q \
  python/sglang/srt/layers/attention/flashinfer_backend.py \
  python/sglang/srt/dllm \
  python/sglang/srt/models/gemma4_diffusion.py \
  python/sglang/srt/models/gemma4_vision.py \
  python/sglang/srt/multimodal/processors/diffusion_gemma.py \
  python/sglang/srt/model_executor/pool_configurator.py \
  python/sglang/srt/speculative \
  python/sglang/srt/mem_cache/memory_pool.py

rg -n "^(<<<<<<< .+|=======$|>>>>>>> .+)" python docs .github
```

No live serving rows were run at this stop point.

## Next Gates

Before any claim uses this branch:

1. Build a packaged SGLang image from `74e0e4bb5f`; no loose-wheel injection.
2. Rerun static import/config audits.
3. Rerun the SGLang claim-grade ladder on the packaged image:
   - E2B / 12B / 26B-A4B / 31B where applicable,
   - DiffusionGemma DG-S0 through DG-S2,
   - fp8 comparator red diagnosis,
   - CUDA-graph gate,
   - MTP identity/no-leak gates.

