# SGLang DiffusionGemma Persistent Static Audit

Date: 2026-06-12 JST

Status: GREEN

Scope: static source/runtime audit only. No model weights were downloaded, no
server was started, and this is not a serving or quality claim.

## Run

- Repo: `jethac/sglang`
- Branch: `spark/hijinks-024-diffusiongemma-upstream-rebase`
- Commit: `06e4a98a56cf525eb7cbcf77738e2a8c5082d47a`
- Workflow: `hijinks-dgemma-static-audit.yml`
- Run: `27387997414`
- URL: `https://github.com/jethac/sglang/actions/runs/27387997414`
- Runner: persistent Ubicloud SGLang runner (`vmzrp4jj`, 16 vCPU)
- CUDA toolkit: 13.0 (`V13.0.88`)
- ccache: `/opt/build-cache/ccache`, 100 GB

## Gate

The workflow compiles and statically checks the DiffusionGemma runtime surface:

- `python/sglang/srt/dllm/config.py`
- `python/sglang/srt/dllm/algorithm/gemma4_renoise.py`
- `python/sglang/srt/models/gemma4_diffusion.py`
- `python/sglang/srt/multimodal/processors/diffusion_gemma.py`
- `python/sglang/srt/configs/model_config.py`
- `python/sglang/srt/configs/diffusion_gemma.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`

Required runtime markers passed:

- `DllmConfig` recognizes `DiffusionGemmaForBlockDiffusion`, sets
  `is_uniform=True`, and reads `canvas_length`.
- `Gemma4Renoise` exists in the dLLM algorithm path.
- `gemma4_diffusion.py` defines `DiffusionGemmaAttention`,
  `DiffusionGemmaDecoderLayer`, `DiffusionGemmaModel`, and
  `DiffusionGemmaForBlockDiffusion`.
- The only `EntryClass = DiffusionGemmaForBlockDiffusion` binding is in
  `gemma4_diffusion.py`.
- The fallback config shim defines DiffusionGemma text/vision/root configs and
  `canvas_length`.
- The multimodal processor binds to `DiffusionGemmaForBlockDiffusion`.

## Notes

Two earlier failed audit runs were useful false positives in the audit itself:

- `27387866798` expected `DiffusionGemmaTextModel`; upstream defines
  `DiffusionGemmaModel`.
- `27387960328` expected the fallback config shim to bind
  `DiffusionGemmaForBlockDiffusion`; runtime binding actually lives in
  `gemma4_diffusion.py` and the multimodal processor.

Both assertions were corrected without weakening the runtime-surface coverage.

