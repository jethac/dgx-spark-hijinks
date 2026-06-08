# vLLM AEON Reproduction

Status: Gemma 4 26B NVFP4+DFlash is locally reproduced; Qwen3.6 NVFP4+DFlash is locally reproduced through AEON's `v2` image when Qwen thinking is disabled with OpenAI `chat_template_kwargs`.

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
- Docker pulls use `DOCKER_PLATFORM=linux/arm64` by default
- `RECORD=1` runs OpenAI smoke, compact benchmark, runtime probe, build-target audit, and row manifest capture
- `HF_CLI=/path/to/hf` can be set if `hf` is available only inside a venv
- `RECORD_PYTHON=/path/to/python` should point at a Python with `torch` installed if host hardware metadata is required

Preflight artifact: `results/aeon_vllm_reproduction_preflight_20260608T0430JST.md`.

Gemma reproduction artifact: `results/aeon_gemma26_dflash_20260608T0436JST_summary.md`.

Qwen reproduction attempt artifacts:

- `results/aeon_qwen36_dflash_20260608T0501JST_summary.md`
- `results/aeon_qwen36_dflash_v2_20260608T0555JST_stop_point.md`
- `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_row_manifest.json`
- `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_server.log`
- `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_nvfp4_checkpoint_audit.json`
- `results/qwen_content_probe_20260608T0900JST_direct_chat_probes.json`
- `results/aeon_qwen36_dflash_nothink_20260608T0834JST_row_manifest.json`

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

The `jethac/vllm` branch `spark/hijinks-020-aeon-qwen-dflash-sm121a` carries the AEON Qwen source fixes that apply cleanly to the current fork:

- guarded lazy fallback import for `_C_stable_libtorch` in `vllm/platforms/cuda.py`
- speculative-decode CUDA graph capture-size alignment for every non-`NONE` graph mode in `vllm/config/compilation.py`
- Qwen3.5/3.6 text-only registry entries in `vllm/model_executor/models/registry.py`
- hybrid KV cache `block_size=None` safety in `vllm/v1/engine/core.py` and `vllm/v1/worker/gpu_model_runner.py`
- Mamba block-size fallback in `vllm/model_executor/layers/mamba/abstract.py`
- text-only M-RoPE fallback in `vllm/v1/worker/gpu_model_runner.py`

Current fork evidence:

- first patch artifact: `results/vllm_qwen_dflash_sm121a_patch_verify_20260608T0330JST.md`
- full AEON Qwen source-port artifact: `results/vllm_aeon_qwen_patch_port_20260608T0619JST.md`
- fork commit: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
- limitation: the derived `jethac` fork image now builds and imports, but Qwen serving exits before health until the `compressed-tensors` dependency is aligned with the newer fork

The old Qwen `language_model.` prefix stripping is a model-conversion helper, not a vLLM source patch for the current AEON v2 multimodal weights.

For Gemma, the GitHub repo has no patch directory. Its referenced `gemma4_patched.py` is effectively model-loader prior art. The current `jethac/vllm` fork already has the important Gemma compressed-tensors NVFP4 loader concepts in `vllm/model_executor/models/gemma4.py`: expert-name remapping, text config handling, patched load paths, and `_weight_iterator`-style handling.

## Interpretation

This lane is different from the FlashInfer `b12x` and FA2 NVFP4-KV work:

- Gemma 4's immediate vLLM win comes from NVFP4 weights, correct compressed-tensors loading, FlashInfer CUTLASS NVFP4 linear kernels, vLLM CUTLASS NvFp4 MoE, Triton target attention, CUDA graphs, and DFlash.
- FA2 NVFP4 KV remains a Qwen/standard-attention capacity lane first; it is not the current Gemma 4 DFlash recipe.
- The next useful proof is a matched `jethac/vllm` fork row with the same Qwen thinking control, backend logs, and quality checks.

## Matched Fork Plan

For the first `jethac/vllm` Qwen parity row, derive from AEON's working Qwen image instead of starting with a full source build. The branch changes in `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3` are Python-side compatibility and stability fixes, while AEON's image already has the known-good CUDA, PyTorch, FlashInfer, model, and DFlash runtime shape.

Recommended build shape:

```bash
docker build --platform linux/arm64 \
  -t jethac-vllm-aeon-q36:6804e1b81 \
  -f - third_party/vllm <<'EOF'
FROM ghcr.io/aeon-7/vllm-spark-omni-q36:v2
ARG VLLM_REF=6804e1b81fcc0abcfb4e876425495fa8f7650bcb
LABEL org.opencontainers.image.source="https://github.com/jethac/vllm" \
      org.opencontainers.image.revision="${VLLM_REF}" \
      aeon.base="ghcr.io/aeon-7/vllm-spark-omni-q36:v2"
RUN python3 -m pip install --no-cache-dir \
      'cmake>=3.26.1' ninja 'packaging>=24.2' \
      'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0' \
      'setuptools-rust>=1.9.0' wheel jinja2
COPY . /opt/jethac-vllm
WORKDIR /opt/jethac-vllm
ENV VLLM_USE_PRECOMPILED=1 \
    VLLM_MAIN_CUDA_VERSION=13.0 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa \
    VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1
RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e . -v
EOF
```

Run shape:

```bash
CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking": false}' \
CHAT_SMOKE_MAX_TOKENS=64 \
CUDA_SO_PACKAGE=vllm,flashinfer \
RUNTIME_REF='ghcr.io/aeon-7/vllm-spark-omni-q36:v2 + jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3' \
IMAGE=jethac-vllm-aeon-q36:6804e1b81 \
MODELS_ROOT=/home/jethac/models/aeon \
RECORD=1 DOWNLOAD=0 DOCKER_PULL=0 WAIT_TIMEOUT=1200 \
scripts/run_aeon_vllm_reproduction.sh qwen36-dflash jethac_qwen36_dflash_nothink_YYYYMMDDTHHMMJST
```

Risks: the AEON image uses vLLM `0.20.1.dev0+g101584af0.d20260424`, while the fork branch is newer; installing with `--no-deps` preserves the AEON environment but may expose API drift. This is a fork parity experiment, not a native `sm_121a` proof unless the build/runtime audits find accepted target evidence.

## 2026-06-08 Matched Fork Stop Point

Artifact: `results/jethac_qwen36_dflash_depstop_20260608T0850JST_summary.md`.

The derived image path builds:

- image: `jethac-vllm-aeon-q36:6804e1b81`
- base: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- fork: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
- precompiled wheel source: vLLM aarch64 cu130 wheel at `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`
- imported `vllm 0.1.dev1+g6804e1b81` from `/opt/jethac-vllm`

The server then exited before health:

```text
ModuleNotFoundError: No module named 'compressed_tensors.compressors.pack_quantized'
```

Interpretation: this stop point was dependency/API drift between the newer `jethac/vllm` branch and AEON's older base environment. It was not an `sm_121` kernel failure and not a Qwen model-load failure. The follow-up row below clears this blocker.

## 2026-06-08 Matched Fork Passing Row

Artifact: `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_summary.md`.

The derived fork row passes after three narrow image layers over the first `jethac/vllm` build:

- `compressed-tensors==0.17.0 --no-deps`, matching the fork's requirements.
- `humming-kernels[cu13]==0.1.4 --no-deps` plus `pyelftools`.
- AEON's original `_vllm_fa2_C.abi3.so`, restored after the precompiled fork FA2 extension hit a PyTorch ABI mismatch.

Target:

- image: `jethac-vllm-aeon-q36:6804e1b81-ct017-humming-aeonfa2`
- base: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- fork: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
- Qwen chat setting: `chat_template_kwargs={"enable_thinking": false}`

Backend evidence from `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_server.log`:

- `vllm 0.1.dev1+g6804e1b81`
- `Qwen3_5MoeForConditionalGeneration`
- `DFlashDraftModel`
- `FlashInferCutlassNvFp4LinearKernel`
- `'MARLIN' NvFp4 MoE backend`
- FlashAttention 2
- CUDA graph capture
- `GPU KV cache size: 1,251,446 tokens`
- maximum concurrency `4.77x` at `262,144` tokens/request

Compact benchmark:

| case | prompt tokens | generated tokens | TTFT seconds | decode tok/s |
|---|---:|---:|---:|---:|
| `short_decode` | 27 | 64 | 0.124 | 47.22 |
| `medium_decode` | 39 | 192 | 0.093 | 58.88 |
| `long_prefill` | 2271 | 64 | 0.448 | 61.62 |

Caveats:

- This proves a `jethac/vllm` fork can serve the AEON Qwen3.6 NVFP4+DFlash row on GB10, but it is still an AEON-derived container recipe.
- The passing image depends on AEON's FA2 binary, so it is not a clean fork wheel/container proof.
- The server still warns that the selected FP4 path uses Marlin weight-only FP4 rather than native FP4 compute.
- The host-side `.so` audit could not import container-local packages, and the build-target audit found no accepted native `sm_121` or `sm_121a` target strings. The follow-up in-container binary audit below closes this as functional compatibility evidence, not native Spark target proof.

## 2026-06-08 In-Container Target Audit

Artifact: `results/jethac_qwen36_dflash_aeonfa2_incontainer_target_audit_20260608.md`.

The passing derived Qwen image was inspected from inside the container rather than from host Python:

- image: `jethac-vllm-aeon-q36:6804e1b81-ct017-humming-aeonfa2`
- runtime device: `NVIDIA GB10`, capability `[12, 1]`, `48` SMs
- PyTorch arch list: `sm_80`, `sm_90`, `sm_100`, `sm_110`, `sm_120`, `compute_120`
- vLLM: `0.1.dev1+g6804e1b81` from `/opt/jethac-vllm`
- FlashInfer: `0.6.9rc1`
- `cuobjdump` inspected `14` vLLM/FlashInfer package objects
- objects with `sm_120`: `3`
- objects with `sm_121`: `0`
- no inspected object reports `sm_121a`
- restored AEON FA2 binary reports only `sm_80`

Conclusion: the passing fork-derived Qwen row remains a real serving compatibility result, but it is not native `sm_121` or `sm_121a` proof. The next vLLM proof is a clean fork CUDA/FA2 image without AEON FA2 binary restoration, followed by the same no-think Qwen row and in-container target audit.

Reusable audit command:

```bash
scripts/run_vllm_incontainer_target_audit.sh \
  jethac-vllm-aeon-q36:6804e1b81-ct017-humming-aeonfa2 \
  jethac_qwen36_dflash_aeonfa2_YYYYMMDDTHHMMJST
```

## 2026-06-08 Clean FA2 Packaging Hook

Fork commit: `jethac/vllm@db4b210c1`.

The vLLM fork now adds `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1` for editable installs that use `VLLM_USE_PRECOMPILED=1`. When set, setup still extracts the normal precompiled vLLM extension set but skips bundled `_vllm_fa2_C.abi3.so` and `_vllm_fa3_C.abi3.so`.

Purpose: build the next derived Qwen image without restoring AEON's FA2 binary or accidentally extracting a precompiled FA2 extension with the wrong PyTorch/CUDA ABI. The image still needs an ABI-matched FA2 build or supplied extension before it can be a clean serving proof.

Next image shape:

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1 \
VLLM_MAIN_CUDA_VERSION=13.0 \
VLLM_PRECOMPILED_WHEEL_COMMIT=4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa \
python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e . -v
```

Then install or build the FA2 extension against the container's actual Torch/CUDA ABI, rerun the no-think Qwen row, and run `scripts/run_vllm_incontainer_target_audit.sh` against the resulting image after a warmed request.

Follow-up build attempt:

- script: `scripts/build_vllm_aeon_qwen_cleanfa2_image.sh`
- attempted image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2`
- artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tfixversion_summary.md`
- raw log: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tfixversion.log`

Result: packaging/versioning is fixed, but native FA2 is not. `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1` skipped bundled FA2/FA3 extraction and `VLLM_VERSION_OVERRIDE=0.1.dev1+ga919d635d` avoided the prior `setuptools-scm` failure. Top-level vLLM CMake accepted `12.1a` and printed `arch=compute_121a,code=sm_121a`.

The nested pinned `vllm-project/flash-attention@dd62dac706b1cf7895bd99b18c6cb7e7e117ee25` configure then collapsed to `CUDA supported target architectures: 12.0`, selected `FA2_ARCHS: 8.0+PTX`, and launched `_vllm_fa2_C` `nvcc` commands with only `compute_80`/`sm_80`. The build was stopped rather than producing another non-native FA2 binary. The builder now fails fast when FA2 configure does not select native SM121/SM121a.

Follow-up patch:

- fork: `jethac/flash-attention`
- branch: `spark/hijinks-021-fa2-sm121a`
- commit: `7d53245`
- submodule: `third_party/vllm-flash-attention`

This is now wired into `scripts/build_vllm_aeon_qwen_cleanfa2_image.sh` through `VLLM_FLASH_ATTN_SRC_DIR=/opt/jethac-vllm-flash-attn`. The builder copies both the vLLM fork and the patched vLLM FlashAttention source into a minimal Docker build context, then fails fast unless nested FA2 configure selects native SM121/SM121a.

Next vLLM clean-packaging step: rerun this image build with the patched FlashAttention source, then require successful `_vllm_fa2_C` import, `cuobjdump` evidence for `sm_121a`, and the same no-think Qwen row.

Patched FlashAttention rerun:

- attempted image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2`
- artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_summary.md`
- raw log: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2.log`

Result: the native FA2 target-selection blocker is fixed. The patched nested source selected `CUDA supported target architectures: 12.1a`, `FA2_ARCHS: 12.1a`, and launched `_vllm_fa2_C` compilation with `-gencode arch=compute_121a,code=sm_121a`.

The build then failed because the copied FlashAttention source lacked its nested `csrc/cutlass` submodule, causing missing `cute/tensor.hpp` and `cutlass/numeric_types.h`. This is a packaging/context issue, not an SM121a CMake-selection issue. The builder now initializes `csrc/cutlass` before creating the Docker context.

Patched FlashAttention plus CUTLASS rerun:

- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- build artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_cutlass_summary.md`
- raw log: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_cutlass.log`
- audit artifact: `results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_target_audit.md`

Result: the clean FA2 image now builds. `_vllm_fa2_C.abi3.so` linked, installed, imported from `/opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`, and `cuobjdump` found `sm_121a` cubins in that extension.

Runtime audit context: `NVIDIA GB10`, compute capability `[12, 1]`, `48` SMs, Torch `2.12.0.dev20260408+cu130`, FlashInfer `0.6.9rc1`, vLLM `0.1.dev1+ga919d635d`.

Remaining caveat: this proves native `sm_121a` FA2 for the patched vLLM FlashAttention extension only. Other vLLM extension objects still contain the existing mixed prebuilt architecture surface, including `sm_120`, `sm_100`, and `sm_90a`. The next gate is the no-think Qwen3.6+DFlash serving row on this clean FA2 image.

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
- `results/aeon_gemma26_dflash_20260608T0436JST_container_target_audit.json` classifies the container as SM120-family/PTX evidence only: runtime device capability is `[12, 1]`, image env contains `TORCH_CUDA_ARCH_LIST=... 12.0+PTX`, and `torch.cuda.get_arch_list()` lists `sm_120` but not explicit `sm_121`.
- The server log does not contain explicit build-target strings accepted by `cuda_build_target_audit.py`, so this row is not native `sm_121` or `sm_121a` proof.
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

Next step: use a more reliable image acquisition path before spending GPU time. The scripted path is:

```bash
RESULTS_DIR=results \
PLATFORM=linux/arm64 \
PULL_TIMEOUT=0 \
USE_SKOPEO=1 \
scripts/pull_container_with_evidence.sh \
  ghcr.io/aeon-7/vllm-spark-omni-q36:v2 \
  aeon_qwen36_dflash_v2_image_pull_YYYYMMDDTHHMMJST
```

This records Docker manifest evidence, disk state, Docker daemon logs, `docker pull --platform linux/arm64`, final `docker image inspect`, and an optional `skopeo` OCI copy/import fallback. If that still fails, the remaining options are a Docker daemon/storage investigation or a source/container build path from the AEON/vLLM recipe.

## 2026-06-08 Qwen3.6 v2 Stop Point

AEON's current documented Qwen image is tracked as `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`, and the local runner now defaults to that tag for `qwen36-dflash`.

After the `v1.2` registration failures, a longer `v1.2` pull and then a `v2` pull were attempted. The `v2` pull had progressed through multiple completed layers, but follow-up inspection failed because the GB10 host stopped answering SSH and ping:

```text
ssh: connect to host 192.168.68.112 port 22: Connection timed out
```

This remains an acquisition/reachability blocker. It is not a Qwen model-load, DFlash, vLLM, FlashInfer, or `sm_121` runtime result.

If the image did not register, use `scripts/pull_container_with_evidence.sh` before trying another hand-run pull.

## 2026-06-08 Qwen3.6 Tailnet Retry

After reconnecting through Tailscale, `ghcr.io/aeon-7/vllm-spark-omni-q36:v2` was present locally and the AEON Qwen3.6 target and drafter weights were present under `/home/jethac/models/aeon`.

Target:

- image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- target model: `/home/jethac/models/aeon/qwen36-nvfp4`
- drafter: `/home/jethac/models/aeon/qwen36-dflash`
- run id: `aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST`

Checkpoint audit:

- artifact: `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_nvfp4_checkpoint_audit.json`
- result: `ok=true`
- format guess: `compressed_tensors_nvfp4`
- loaded keys: `124306` from safetensors
- compressed-tensors markers: `123520`
- quantized sensitive key count: `0`

Server/backend evidence from `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_server.log`:

- resolved target architecture: `Qwen3_5MoeForConditionalGeneration`
- resolved drafter architecture: `DFlashDraftModel`
- selected `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM
- selected `MARLIN` NvFp4 MoE backend
- used FlashAttention version 2
- logged `GPU KV cache size: 585,168 tokens`
- logged maximum concurrency `4.73x` at `262,144` tokens/request
- captured CUDA graphs

Failure:

- row manifest: `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_row_manifest.json`, `ok=false`
- chat smoke: `content=null`; vLLM returned text under `message.reasoning`, not normal content, and did not produce `spark-ok`
- compact benchmark: completion token counts were present, but streamed content and `reasoning_content` were empty, so the row has no valid decode-quality output
- build-target audit: no CUDA architecture target evidence found in the server log
- server warning: the GPU was reported as lacking native FP4 computation for this path, so weight-only FP4 compression used Marlin and may degrade compute-heavy workloads

Interpretation: image acquisition and model startup are no longer the blocker for AEON Qwen3.6. The blocker moved to output validation and native-target/kernel proof. This is a useful partial vLLM Qwen row, but it must not be counted as a passing serving benchmark or a fork speedup.

## 2026-06-08 Qwen3.6 No-Think Row

The prior failed row was a Qwen thinking-mode artifact. With `--reasoning-parser qwen3`, normal responses are routed through `message.reasoning` until the model emits `</think>`. The 8-token smoke and compact benchmark never reached normal `message.content`.

Direct probe artifact: `results/qwen_content_probe_20260608T0900JST_direct_chat_probes.json`.

- baseline `qwen36-fast`, `max_tokens=64`: `content=null`, `finish_reason=length`, text in `message.reasoning`
- prompt-level `/no_think`: still `content=null`, text in `message.reasoning`
- API-level `chat_template_kwargs={"enable_thinking": false}`: `content="spark-ok"` for both `qwen36-fast` and `qwen36-deep`

Passing serving row:

- row manifest: `results/aeon_qwen36_dflash_nothink_20260608T0834JST_row_manifest.json`, `ok=true`
- chat smoke: `results/aeon_qwen36_dflash_nothink_20260608T0834JST_chat_smoke.json`, `content="spark-ok"`
- compact benchmark: `results/aeon_qwen36_dflash_nothink_20260608T0834JST_openai_benchmark.json`
- short decode: `50.37 tok/s`
- medium decode: `55.84 tok/s`
- long-prefill decode: `53.75 tok/s`

Caveats:

- This is AEON's container/checkpoint, not a `jethac/vllm` fork speedup.
- The build-target audit still finds no explicit native `sm_121` or `sm_121a` target evidence in the server log.
- The server still warns that the selected FP4 path uses Marlin weight-only FP4 because the GPU is not treated as native FP4 for that path.
