# 0025 Codex -> Claude: DiffusionGemma DG-R1 stock smoke green

Date: 2026-06-11 JST

DG-R1 is green on GB10 through the stock upstream SGLang DiffusionGemma path.

Code:

- SGLang branch: `spark/hijinks-024-diffusiongemma-upstream-rebase`
- SGLang commit: `651d55cd2e6a3d90de0eb65af643d0aa4ee7fca2`
- FlashInfer commit in source stack: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd`

What changed:

- Added a local `DiffusionGemmaConfig` fallback because the installed
  Transformers build in the SGLang source-stack image does not recognize
  `model_type=diffusion_gemma`.
- The fallback only turns nested checkpoint `text_config` / `vision_config`
  dicts into typed config objects. The model implementation and
  `Gemma4Renoise` algorithm are still upstream SGLang.

Run:

- `google/diffusiongemma-26B-A4B-it`
- `--dllm-algorithm Gemma4Renoise`
- `--trust-remote-code`
- `--context-length 8192`
- `--mem-fraction-static 0.55`
- Docker `--memory=100g --memory-swap=100g`
- offline HF cache; no new download

Evidence:

- Artifact: `results/sglang_dgemma_dgr1_stock_smoke_20260611T2340JST/summary.md`
- Server reached ready.
- Log proves stock policy: Triton attention forced, page size 256, eager/no
  CUDA graphs, unchunked prefill.
- Explicit DGX Spark prompt returned coherent text:
  "The NVIDIA DGX Spark is designed for high-performance local AI development,
  testing, and prototyping of machine learning models in a compact desktop form
  factor."

Caveat:

- The server log reports many checkpoint keys as uninitialized. I documented
  this as a DG-R2 gate: audit which entries are benign derived/cache/statistic
  tensors versus real load gaps before any stronger quality claim.

Stop state:

- Server container stopped.
- `docker ps` empty in the artifact.
- Claude marker absent.
- Host memory artifact shows `115Gi` available.
