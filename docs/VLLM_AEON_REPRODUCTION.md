# vLLM AEON Reproduction

Status: ready for the first local reproduction run; not yet a banked local performance result.

AEON-7's public Gemma and Qwen recipes are currently the highest-leverage vLLM prior art for this campaign because they target GB10 / `sm_121a`, NVFP4 weights, and DFlash speculative decoding. Treat them as external evidence until our local artifacts exist.

Sources:

- `https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- `https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash`

## Local Runner

Use:

```bash
DOWNLOAD=1 DOCKER_PULL=1 RECORD=1 \
scripts/run_aeon_vllm_reproduction.sh gemma26-dflash aeon_gemma26_dflash_YYYYMMDDTHHMMJST
```

or:

```bash
DOWNLOAD=1 DOCKER_PULL=1 RECORD=1 \
scripts/run_aeon_vllm_reproduction.sh qwen36-dflash aeon_qwen36_dflash_YYYYMMDDTHHMMJST
```

Defaults:

- model root: `/opt/spark-models/aeon`
- results directory: `results`
- port: `8000`
- downloads and docker pulls are opt-in
- `RECORD=1` runs OpenAI smoke, compact benchmark, runtime probe, build-target audit, and row manifest capture
- `HF_CLI=/path/to/hf` can be set if `hf` is available only inside a venv

Preflight artifact: `results/aeon_vllm_reproduction_preflight_20260608T0430JST.md`.

## Preflight Result

Both GHCR images resolve:

- `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2`

All four required HF repos are public and non-gated from the GB10 host:

- `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- `z-lab/gemma-4-26B-A4B-it-DFlash`
- `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- `z-lab/Qwen3.6-35B-A3B-DFlash`

## Patch Triage

The `jethac/vllm` branch `spark/hijinks-020-aeon-qwen-dflash-sm121a` already carries the two high-confidence general AEON fixes:

- guarded lazy fallback import for `_C_stable_libtorch` in `vllm/platforms/cuda.py`
- speculative-decode CUDA graph capture-size alignment for every non-`NONE` graph mode in `vllm/config/compilation.py`

Remaining AEON Qwen patch candidates are narrower and should be gated by a failing local reproduction before porting:

- Qwen hybrid/Mamba `block_size=None` safety in `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/engine/core.py`, `vllm/v1/worker/gpu_model_runner.py`, and `vllm/model_executor/layers/mamba/abstract.py`
- text-only Qwen3.5/3.6 registry entries in `vllm/model_executor/models/registry.py`
- text-only M-RoPE fallback in `vllm/v1/worker/gpu_model_runner.py`

The old Qwen `language_model.` prefix stripping is a model-conversion helper, not a vLLM source patch for the current AEON v2 multimodal weights.

For Gemma, the GitHub repo has no patch directory. Its referenced `gemma4_patched.py` is effectively model-loader prior art. The current `jethac/vllm` fork already has the important Gemma compressed-tensors NVFP4 loader concepts in `vllm/model_executor/models/gemma4.py`: expert-name remapping, text config handling, patched load paths, and `_weight_iterator`-style handling.

## Interpretation

This lane is different from the FlashInfer `b12x` and FA2 NVFP4-KV work:

- Gemma 4's immediate vLLM win is expected from NVFP4 weights, correct compressed-tensors loading, CUTLASS/Marlin backend choice, Triton target attention, and DFlash.
- FA2 NVFP4 KV remains a Qwen/standard-attention capacity lane first; it is not the current Gemma 4 DFlash recipe.
- The next useful proof is a local serving row with model download, image digest, server log, build-target audit, runtime probe, OpenAI smoke, and compact benchmark.
