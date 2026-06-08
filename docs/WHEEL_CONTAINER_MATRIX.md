# Wheel And Container Matrix

Date: 2026-06-08 JST

Purpose: track which install/container paths have actually worked on the Spark-class GB10 host, and which ones are only candidates. This is the acceptance-evidence companion for `docs/DGX_SPARK_SOLUTIONS.md` solution area 2.

Hardware scope: single GB10 / `sm_121` / Linux `aarch64` system.

## Decision Rules

- `proven`: starts and serves or runs the named task on local GB10 with artifacts.
- `partial`: useful row exists, but the path is too model-specific, source-modified, or incomplete to bless broadly.
- `blocked`: acquisition, install, startup, or runtime prevents the next proof.
- `not blessed`: explicitly not recommended for the named scope yet.

Clean Spark support requires more than runtime device capability. A row should eventually record:

- image tag or source commit
- Linux architecture
- CUDA version
- Python package versions
- model and quantization
- hardware key including compute capability and SM count
- build/JIT target evidence, or an explicit documented compatibility path such as `12.0+PTX`
- selected runtime backends
- smoke and benchmark artifacts

## Current Matrix

| stack path | status | package/build facts | proven local scope | missing before blessing |
|---|---|---|---|---|
| `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2` | `proven` for AEON Gemma 26B serving; `partial` as general vLLM path | Linux `arm64`; CUDA `13.0.2`; vLLM `0.20.1`; PyTorch `2.11.0+cu130`; FlashInfer `0.6.8.post1`; `TORCH_CUDA_ARCH_LIST=... 12.0+PTX`; image digest `sha256:0b938294...` | Gemma 4 26B A4B NVFP4+DFlash serves locally; warmed compact row `47.91`, `53.60`, `98.38 tok/s`; backend logs show FlashInfer CUTLASS NVFP4 linear, vLLM CUTLASS NvFp4 MoE, `TRITON_ATTN`, DFlash, CUDA graphs; container target audit classifies this as SM120-family/PTX evidence only | accuracy check, fork parity row, explicit binary/JIT native `sm_121` or `sm_121a` proof, and a non-AEON/general vLLM recipe |
| `ghcr.io/aeon-7/vllm-spark-omni-q36:v2` | `proven` for AEON Qwen3.6 serving; `partial` as general vLLM path | Linux `arm64` AEON Qwen3.6 NVFP4+DFlash image; local run reports vLLM `0.20.1.dev0+g101584af0.d20260424`; target checkpoint audit passes compressed-tensors NVFP4 checks | target and drafter load; server log shows `Qwen3_5MoeForConditionalGeneration`, `DFlashDraftModel`, `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, CUDA graphs, `585168` KV tokens; with `chat_template_kwargs={"enable_thinking": false}`, compact serving passes at `50-56 tok/s` decode | no accepted native `sm_121`/`sm_121a` build-target evidence, and server warns the selected FP4 path lacks native FP4 compute support |
| `vllm/vllm-openai:latest-cu130` | `partial` | CUDA 13 vLLM image; observed vLLM `0.20.0` on Gemma 26B row; PyTorch `2.11.0+cu130`; FlashInfer `0.6.8.post1` | Gemma 4 26B A4B BF16/unquantized serves at about `24 tok/s`; Gemma 4 QAT-unquantized snapshot loads and serves as BF16/unquantized | not an NVFP4 row; not a Gemma 12B unified clean path; needs explicit build-target evidence and backend/fallback warnings |
| vLLM source/precompiled overlay at `da1daf40` | `partial` | editable source install with `VLLM_USE_PRECOMPILED=1`, `VLLM_MAIN_CUDA_VERSION=13.0`; Transformers main `effde209...`; stale FlashInfer JIT-cache cleanup required | Gemma 4 12B unified starts and serves at about `7.7 tok/s` | clean release/nightly container without source surgery, accuracy task, and performance-worthy quantized/spec-decode path |
| `jethac/vllm@spark/hijinks-020-aeon-qwen-dflash-sm121a` | `partial` derived-image serving proof | fork commit `6804e1b`; inherits SM12x NVFP4 KV routing/deswizzle; ports AEON Qwen source fixes; passing image `jethac-vllm-aeon-q36:6804e1b81-ct017-humming-aeonfa2` derives from AEON Qwen v2, aligns `compressed-tensors==0.17.0`, adds `humming-kernels==0.1.4`, and restores AEON's FA2 binary | serves Qwen3.6 NVFP4+DFlash with `chat_template_kwargs={"enable_thinking": false}`; row `jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST` passes at `47.22`, `58.88`, and `61.62 tok/s`; logs show `Qwen3_5MoeForConditionalGeneration`, `DFlashDraftModel`, FlashInfer NVFP4 linear, Marlin NvFp4 MoE, FlashAttention 2, CUDA graphs, and `1,251,446` KV tokens; in-container audit of this image finds runtime GB10 `[12, 1]`, `sm_120` objects, zero `sm_121` objects, and no `sm_121a` evidence | clean fork wheel/container without AEON FA2 binary, accepted native `sm_121`/`sm_121a` target evidence, then matched before/after throughput/capacity rows |
| `nvcr.io/nvidia/sglang:26.05-py3` | `proven` for small Qwen BF16/fp8; `partial` as SGLang path | Linux image with CUDA `13.2.1`; SGLang `0.5.11+05fb6a52`; `NVIDIA_SGLANG_VERSION=26.05`; `TORCH_CUDA_ARCH_LIST=... 12.0+PTX`; image digest `sha256:80cf6f3...` | Qwen2.5 1.5B BF16/auto and fp8 serve at about `58-59 tok/s`; fp8 roughly doubles KV pool over BF16/auto; CUDA graphs enabled | Gemma path fails, FP4 KV not usable, no explicit `sm_121` SASS proof, logs call path `SM120 (Blackwell)` |
| `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving` | `debug-only` source fork | fork branch carries FP4 KV gate/alias fixes, SGLang-style packed K/V + scale-buffer plumbing, pre-capture global-scale calibration, and SM120-family writer fallback | clean NVIDIA 26.05 source overlay reaches graph-enabled `fp4_e2m1` serving on Qwen after calibrating 28 layers from 4096 eager prefill tokens; BF16 control passes under the same source/graph setup | FP4 KV output quality is still wrong on Qwen, so no throughput/capacity row is blessed; needs end-to-end writer/readback correctness before benchmarking |
| `jethac/flashinfer@spark/hijinks-004-sm121-flashinfer` | `debug-only` source fork | fork commit `a42c8f07`; SM121 `mm_fp4` dispatch includes `b12x`; adds `12.1a` JIT-cache build targets | source/JIT path selects `b12x` on real GB10 and produces finite outputs | clean wheel/container proof; model-shaped proxy was mixed-to-slower; no serving speedup claim |
| `jethac/flashinfer@spark/hijinks-007-fa2-nvfp4-kv-sm121` | `debug-only` source fork | fork commit `e152cf4d`; inherits SM121 `mm_fp4` work; adds FA2 NVFP4 KV explicit scale-factor stride/page and V-SF de-swizzle support | standalone FlashInfer FA2 NVFP4 KV probes pass small and Gemma sliding/local shapes | Gemma global `D=512` path fails; no end-to-end vLLM/SGLang serving proof; no clean wheel/container proof |
| llama.cpp `b9536` source build | `proven` for practical serving | source build `308f61c31 (9536)`; server logs `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, `BLACKWELL_NATIVE_FP4 = 1` | Gemma 4 26B Q4_0 serves around `76 tok/s`; Qwen2.5 1.5B Q4_K_M serves around `167-175 tok/s`; `llama-bench` rows captured | paper-comparable GGUF lm-eval accuracy, native NVFP4/MXFP4 GGUF tensor-core proof, larger Qwen3/Qwen3.6 GGUF rows |
| LiteRT-LM `0.13.1` venv | `side-runtime` | Linux `aarch64` venv; `litert-lm==0.13.1`; `litert-lm-api==0.13.1` | Gemma 4 E2B CPU chat returns `spark-ok`; GPU benchmark runs and improves prefill | GPU chat exits `-11`; OpenCL load warning; not a main throughput path |
| HF fallback in benchmark venv | `partial` | local Python env with PyTorch `2.11.0+cu130` | some fallback rows complete; telemetry/failure annotation exists | several rows die `returncode=-9`; stronger OOM/resource evidence required before comparisons |

## Artifact Index

| stack path | key artifacts |
|---|---|
| AEON Gemma vLLM | `results/aeon_gemma26_dflash_20260608T0436JST_summary.md`, `results/aeon_gemma26_dflash_20260608T0436JST_image_inspect.json`, `results/aeon_gemma26_dflash_20260608T0436JST_container_versions.json`, `results/aeon_gemma26_dflash_20260608T0436JST_container_target_audit.json`, `results/aeon_gemma26_dflash_20260608T0436JST_server.log` |
| AEON Qwen vLLM | `results/aeon_vllm_reproduction_preflight_20260608T0430JST.md`, `results/aeon_qwen36_dflash_20260608T0501JST_summary.md`, `results/aeon_qwen36_dflash_v2_20260608T0555JST_stop_point.md`, `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_row_manifest.json`, `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_server.log`, `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_nvfp4_checkpoint_audit.json`, `results/qwen_content_probe_20260608T0900JST_direct_chat_probes.json`, `results/aeon_qwen36_dflash_nothink_20260608T0834JST_row_manifest.json`, `results/aeon_qwen36_dflash_nothink_20260608T0834JST_openai_benchmark.json`, `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_summary.md`, `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_row_manifest.json` |
| NVIDIA SGLang | `results/sglang_qwen25_1_5b_fp8_vs_fp4kv_20260608T0332JST_summary.md`, `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_image_inspect.json`, `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_server.log` |
| vLLM Gemma 26B BF16 | `results/vllm_gemma4_26b_a4b_bf16_compact_20260607T131917Z.json`, `results/vllm_gemma4_26b_a4b_bf16_20260607T131917Z_server.log` |
| vLLM Gemma 12B unified | `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_server.log`, `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_compact_benchmark.json` |
| fork source verification | `results/vllm_aeon_qwen_patch_port_20260608T0619JST.md`, `results/vllm_qwen_dflash_sm121a_patch_verify_20260608T0330JST.md`, `results/sglang_fp4_kv_sm121_pytest_20260608T0320JST.md`, `results/vllm_nvfp4_sm12x_routing_probe_20260607T171227Z.json` |
| FlashInfer probes | `results/flashinfer_sm121_source_jit_20260607T1250Z.json`, `results/flashinfer_nvfp4_kv_probe_20260608T023901JST.json`, `results/flashinfer_nvfp4_kv_probe_gemma4_26b_sliding_1024_20260608T0340JST.json`, `results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_20260608T0335JST.json` |
| llama.cpp | `results/llamacpp_gemma4_26b_q4_0_20260607T135911Z_server.log`, `results/llamacpp_gemma4_26b_q4_0_compact_20260607T135911Z.json`, `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_server.log`, `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_openai_benchmark.json` |
| LiteRT-LM | `results/litert_lm_20260607T140617Z_import_probe.json`, `results/litert_lm_cpu_e2b_smoke_no_max_telemetry.json`, `results/litert_lm_gpu_e2b_bench_256p64d_telemetry.json`, `results/litert_lm_gpu_e2b_smoke_after_groups_telemetry.json` |

## Current Gaps

1. No clean, official vLLM Spark container is blessed across Gemma and Qwen.
2. The fastest local vLLM row is AEON's Gemma image, not a `jethac` fork or upstream official Spark release.
3. Qwen3.6 NVFP4+DFlash now has passing AEON and derived `jethac/vllm` serving rows, but only with API-level thinking disabled and without native-target proof.
4. SGLang has a proven small-Qwen fp8/BF16 path but no Gemma path and no usable FP4 KV path; the latest fork reaches graph-enabled FP4 startup, but output quality still fails.
5. FlashInfer fork evidence is strong at source/JIT and standalone-kernel level, but not yet packaged as a clean runtime dependency with serving wins.
6. llama.cpp is practical and fast, but its GGUF accuracy path and native FP4/MXFP4 path remain separate workstreams.

## Next Matrix Updates

Update this matrix when:

- a new image is pulled or built with manifest/image-inspect evidence
- a source fork becomes a clean wheel/container
- a runtime row moves from startup/smoke to benchmark or accuracy
- build-target evidence changes from runtime capability only to binary/JIT proof
- dependency-aligned `jethac/vllm` Qwen3.6 fork row gains clean fork packaging or native target evidence
