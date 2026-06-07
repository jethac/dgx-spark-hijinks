# AEON vLLM Reproduction Preflight

Timestamp: 2026-06-08T04:30 JST

Purpose: check whether the AEON Gemma/Qwen NVFP4+DFlash reproduction lane is blocked before downloading large weights or occupying the GB10 GPU.

## Image Checks

Command shape:

```bash
docker manifest inspect ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2
docker manifest inspect ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2
```

Result:

- `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`: manifest resolves
- `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2`: manifest resolves

## Hugging Face Metadata Checks

Command ran from the existing benchmark venv on the GB10 host with `huggingface_hub`.

Result:

| repo | sha | private | gated |
|---|---|---:|---:|
| `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4` | `1d7e8dbc4456fedeacd12e5308c7d43fd489dfa6` | `False` | `False` |
| `z-lab/gemma-4-26B-A4B-it-DFlash` | `77d4202772dfe50b2396ec7bac9cfffc7b9e7057` | `False` | `False` |
| `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` | `3d05c6063e82cdae8dd1b9362116ea71be8d1439` | `False` | `False` |
| `z-lab/Qwen3.6-35B-A3B-DFlash` | `42d3b34d588423cdae7ba8f53a8cf7789346a719` | `False` | `False` |

## Local Harness

Added `scripts/run_aeon_vllm_reproduction.sh` with targets:

- `gemma26-dflash`
- `qwen36-dflash`

The script supports opt-in model download (`DOWNLOAD=1`), opt-in docker pull (`DOCKER_PULL=1`), and opt-in benchmark recording (`RECORD=1`) through `scripts/record_openai_serving_row.py`.

The benchmark venv on the host has a working `hf` CLI. If the default shell PATH does not expose it, set `HF_CLI=/home/jethac/gemma4-evals/.venv/bin/hf` when launching the runner.

## Current Stop Point

No large model download, image pull, or GPU serving run has been started by this preflight. The next step is a real `DOWNLOAD=1 DOCKER_PULL=1 RECORD=1` reproduction row for Gemma or Qwen.
