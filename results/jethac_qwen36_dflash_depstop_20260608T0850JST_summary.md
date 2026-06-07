# jethac/vLLM Qwen3.6 DFlash Stop Point

Date: 2026-06-08 JST

Goal: run the AEON Qwen3.6 NVFP4+DFlash row through `jethac/vllm@spark/hijinks-020-aeon-qwen-dflash-sm121a`.

## Build

Derived image:

- image: `jethac-vllm-aeon-q36:6804e1b81`
- base: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- fork: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
- build artifact: `results/jethac_vllm_aeon_q36_6804e1b81_image_build_20260608T0845JST.log`

The first build attempt failed because the AEON base image lacked `setuptools_rust`. A second build installed the Python build helpers, used the precompiled aarch64 cu130 vLLM wheel at commit `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`, and succeeded.

Build verification imported:

- `vllm 0.1.dev1+g6804e1b81` from `/opt/jethac-vllm/vllm/__init__.py`
- `torch 2.12.0.dev20260408+cu130`
- `flashinfer 0.6.9rc1`

## Serve Attempt

Run id: `jethac_qwen36_dflash_depstop_20260608T0850JST`

The container started with the same Qwen target/drafter mounts and serving flags as the passing AEON row, using:

- target: `/home/jethac/models/aeon/qwen36-nvfp4`
- drafter: `/home/jethac/models/aeon/qwen36-dflash`
- quantization: `compressed-tensors`
- attention backend: `flash_attn`
- DFlash: `num_speculative_tokens=15`

It exited before health and before model load completed.

Artifact:

- `results/jethac_qwen36_dflash_depstop_20260608T0850JST_server.log`

Key lines:

```text
version 0.1.dev1+g6804e1b81
Resolved architecture: Qwen3_5MoeForConditionalGeneration
ModuleNotFoundError: No module named 'compressed_tensors.compressors.pack_quantized'
```

## Interpretation

This is a real fork parity blocker, not an `sm_121` kernel failure and not a Qwen checkpoint failure. The derived image proves the fork can be installed over AEON's runtime using precompiled aarch64 artifacts, but the newer fork expects a newer `compressed_tensors` package API than the AEON image provides.

Next step: build a second derived image that updates only the required `compressed-tensors` dependency, then rerun this same row. If that passes import/config, the next risk is broader dependency/API drift between AEON's vLLM `0.20.1.dev0+g101584af0.d20260424` environment and the newer `jethac/vllm` branch.
