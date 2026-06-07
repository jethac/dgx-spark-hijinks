# vLLM AEON Reproduction

Status: Gemma 4 26B NVFP4+DFlash is locally reproduced; Qwen3.6 NVFP4+DFlash is blocked before serving by image-pull/registration and host-reachability failure.

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
- `RECORD_PYTHON=/path/to/python` should point at a Python with `torch` installed if host hardware metadata is required

Preflight artifact: `results/aeon_vllm_reproduction_preflight_20260608T0430JST.md`.

Gemma reproduction artifact: `results/aeon_gemma26_dflash_20260608T0436JST_summary.md`.

Qwen reproduction attempt artifacts:

- `results/aeon_qwen36_dflash_20260608T0501JST_summary.md`
- `results/aeon_qwen36_dflash_v2_20260608T0555JST_stop_point.md`

## Preflight Result

Both GHCR images resolve:

- `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2`
- `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`

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

- Gemma 4's immediate vLLM win comes from NVFP4 weights, correct compressed-tensors loading, FlashInfer CUTLASS NVFP4 linear kernels, vLLM CUTLASS NvFp4 MoE, Triton target attention, CUDA graphs, and DFlash.
- FA2 NVFP4 KV remains a Qwen/standard-attention capacity lane first; it is not the current Gemma 4 DFlash recipe.
- The next useful proof is unblocking the Qwen3.6 image/container path, then recording a matched serving row.

## 2026-06-08 Gemma 26B Result

Target:

- image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- model: `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- drafter: `z-lab/gemma-4-26B-A4B-it-DFlash`
- hardware key: `NVIDIA_GB10:sm_121:sms_48`

Measured warmed compact row:

| case | prompt tokens | generated tokens | TTFT seconds | decode tok/s |
|---|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.098 | 47.91 |
| `medium_decode` | 40 | 192 | 0.087 | 53.60 |
| `long_prefill` | 2270 | 64 | 0.118 | 98.38 |

Backend evidence from `results/aeon_gemma26_dflash_20260608T0436JST_key_log_lines.txt`:

- `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM
- `VLLM_CUTLASS` NvFp4 MoE backend
- target attention forced to `TRITON_ATTN`
- drafter attention uses `FLASH_ATTN`
- KV cache size `519,920` tokens at `--max-model-len 262144`
- maximum concurrency `1.98x`

Caveats:

- This is a real local vLLM serving win, but it is AEON's container/checkpoint, not proof that a `jethac` fork changed throughput.
- The container reports runtime device capability `[12, 1]`, but `torch.cuda.get_arch_list()` lists `sm_120` and not explicit `sm_121`.
- The server log does not contain explicit build-target strings accepted by `cuda_build_target_audit.py`.
- The server warns about differing NVFP4 global scales across fused parallel layers; accuracy still needs a separate check.

## 2026-06-08 Qwen3.6 Attempt

Target:

- image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2`
- model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`

Result:

- target model downloaded to `/home/jethac/models/aeon/qwen36-nvfp4`, size about `22G`
- drafter downloaded to `/home/jethac/models/aeon/qwen36-dflash`, size about `905M`
- first Docker pull reached late layer extraction but never registered the image
- second bounded pull retry used `timeout 900`; it again reached late `Pull complete` lines but timed out without registering the image
- no vLLM server or model load started, so this is not a Qwen model/runtime/kernel failure yet

Artifacts:

- summary: `results/aeon_qwen36_dflash_20260608T0501JST_summary.md`
- pull retry log: `results/aeon_qwen36_dflash_20260608T0501JST_docker_pull_retry.log`
- post-attempt GPU/process state: `results/aeon_qwen36_dflash_20260608T0501JST_nvidia_smi_after.txt`

Next step: use a more reliable image acquisition path before spending GPU time, for example a longer pull with Docker daemon debug evidence, `skopeo`/`crane` copy into the local daemon, or a source/container build path from the AEON/vLLM recipe.

## 2026-06-08 Qwen3.6 v2 Stop Point

AEON's current documented Qwen image is tracked as `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`, and the local runner now defaults to that tag for `qwen36-dflash`.

After the `v1.2` registration failures, a longer `v1.2` pull and then a `v2` pull were attempted. The `v2` pull had progressed through multiple completed layers, but follow-up inspection failed because the GB10 host stopped answering SSH and ping:

```text
ssh: connect to host 192.168.68.112 port 22: Connection timed out
```

This remains an acquisition/reachability blocker. It is not a Qwen model-load, DFlash, vLLM, FlashInfer, or `sm_121` runtime result.
