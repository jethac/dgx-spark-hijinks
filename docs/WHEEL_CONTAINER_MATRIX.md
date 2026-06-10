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
| `jethac/vllm@spark/hijinks-020-aeon-qwen-dflash-sm121a` | `partial` clean Qwen serving proof | fork commit `6804e1b` produced the first derived row but still restored AEON's FA2 binary; later fork commit `a919d635d` adds clean packaging/versioning and `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1`; `jethac/flash-attention@7d53245` patches pinned FA2 CMake for `12.1a`; clean image `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass` builds with nested CUTLASS initialized | serves Qwen3.6 NVFP4+DFlash with `chat_template_kwargs={"enable_thinking": false}`; AEON-FA2-derived row `jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST` passes at `47.22`, `58.88`, and `61.62 tok/s`; clean FA2 row `jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST` passes at `61.07`, `56.97`, and `60.10 tok/s`; clean image audit proves `_vllm_fa2_C.abi3.so` contains `sm_121a` cubins; serving log shows Qwen3.5 MoE, DFlash, FlashInfer NVFP4 linear, Marlin NvFp4 MoE, FlashAttention 2, CUDA graphs, FlashInfer FP4 GEMM autotune, and `1,241,920` KV tokens | still partial because the server warns that the weight/MoE path uses Marlin weight-only FP4 rather than native FP4 compute, host-side row `.so` audit cannot import container packages, and no speedup/capacity win over the AEON-derived row is proven |
| `jethac-vllm-aeon-gemma4:ad2337814-rebuiltc-fb7d62ea-sm121a` | `build/import-proven` rebuilt `_C` image | vLLM `jethac/vllm@ad233781492ca1d4eaa8c1dd0d80777933163d54`, FlashInfer `jethac/flashinfer@fb7d62ea45f19cb61f19057a93519c17b6e257f3`, base `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`; image id `sha256:750ce7b9c59befe4426b484be24a1f58f585a5e13b7ebe13179a16f4feb4d051` | rebuilds vLLM `_C`, `_C_stable_libtorch`, `_moe_C`, FA2, and FA3 inside the image; import probe passes for `vllm._C`, `vllm._moe_C`, and `vllm.vllm_flash_attn._vllm_fa2_C`; cuobjdump on `_C.abi3.so` shows `sm_121a` cubins; Claude Window 2 Block C linear-V-SF writer regression passed on this image (`results/claude_window2_20260611/`, 022 branch head `e08a6f3ae`) | not a serving result yet; this image exists to unblock Claude Blocks C/D for the rebuilt CUDA writer path; build log still contains nonfatal CUDA 13 vector-type warnings and nested flash-attn coverage is separate from the rebuilt vLLM `_C` target; r7 import probe missed `humming`, which vLLM's quantization registry lazy-imports, so the next r8 image should install/verify `humming`; image `WORKDIR=/opt/jethac-vllm` puts cwd ahead of `PYTHONPATH`, so source-overlay runs must use `docker run -w /work` or equivalent to avoid silently importing the baked tree instead of the overlay |
| `nvcr.io/nvidia/sglang:26.05-py3` | `proven` for small Qwen BF16/fp8; `partial` as SGLang path | Linux image with CUDA `13.2.1`; SGLang `0.5.11+05fb6a52`; `NVIDIA_SGLANG_VERSION=26.05`; `TORCH_CUDA_ARCH_LIST=... 12.0+PTX`; image digest `sha256:80cf6f3...` | Qwen2.5 1.5B BF16/auto and fp8 serve at about `58-59 tok/s`; fp8 roughly doubles KV pool over BF16/auto; CUDA graphs enabled | Gemma path fails, stock FP4 KV is not usable, no explicit `sm_121` SASS proof, logs call path `SM120 (Blackwell)` |
| `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving` | `partial` scoped Qwen mixed-KV path | fork branch carries FP4 KV gate/alias fixes, SGLang-style packed K/V + scale-buffer plumbing, pre-capture global-scale calibration, SM120-family writer fallback, prefix-cache graph-write guard, mixed FP8-K/NVFP4-V mode, and radix/page/merge/write-read/prefix-reference traces for native FP4 KV | clean NVIDIA 26.05 source overlay reaches Qwen mixed-KV radix serving; `results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.md` fixes the default-radix first-token failure (`ark` -> `**`); graph-safe fixed-8k PPL sweeps are green through reused prefix `7680`; `results/sglang_qwen_mixedkv_capacity_denominator_audit_20260610TmanualJST.md` corrects current equal-budget capacity to about `1.28x` versus fp8 KV | still not broadly blessed: full NVFP4 K+V remains quality-red under radix reuse, Gemma rows are unrun, and final throughput rows need recording; historical `~1.78x` mixed-KV token counts are allocator-overbudget artifacts, not the current capacity claim |
| `jethac/flashinfer@spark/hijinks-004-sm121-flashinfer` | `debug-only` source fork | fork commit `a42c8f07`; SM121 `mm_fp4` dispatch includes `b12x`; adds `12.1a` JIT-cache build targets | source/JIT path selects `b12x` on real GB10 and produces finite outputs | clean wheel/container proof; model-shaped proxy was mixed-to-slower; no serving speedup claim |
| `jethac/flashinfer@spark/hijinks-007-fa2-nvfp4-kv-sm121` | `debug-only` source fork | fork commit `e152cf4d`; inherits SM121 `mm_fp4` work; adds FA2 NVFP4 KV explicit scale-factor stride/page and V-SF de-swizzle support | standalone FlashInfer FA2 NVFP4 KV probes pass small and Gemma sliding/local shapes | Gemma global `D=512` path fails; no end-to-end vLLM/SGLang serving proof; no clean wheel/container proof |
| llama.cpp `b9536` source build | `proven` for practical serving | source build `308f61c31 (9536)`; server logs `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, `BLACKWELL_NATIVE_FP4 = 1` | Gemma 4 26B Q4_0 serves around `76 tok/s`; Qwen2.5 1.5B Q4_K_M serves around `167-175 tok/s`; `llama-bench` rows captured | paper-comparable GGUF lm-eval accuracy, native NVFP4/MXFP4 GGUF tensor-core proof, larger Qwen3/Qwen3.6 GGUF rows |
| LiteRT-LM `0.13.1` venv | `side-runtime` | Linux `aarch64` venv; `litert-lm==0.13.1`; `litert-lm-api==0.13.1` | Gemma 4 E2B CPU chat returns `spark-ok`; GPU benchmark runs and improves prefill | GPU chat exits `-11`; OpenCL load warning; not a main throughput path |
| HF fallback in benchmark venv | `partial` | local Python env with PyTorch `2.11.0+cu130` | some fallback rows complete; telemetry/failure annotation exists | several rows die `returncode=-9`; stronger OOM/resource evidence required before comparisons |

## Artifact Index

| stack path | key artifacts |
|---|---|
| AEON Gemma vLLM | `results/aeon_gemma26_dflash_20260608T0436JST_summary.md`, `results/aeon_gemma26_dflash_20260608T0436JST_image_inspect.json`, `results/aeon_gemma26_dflash_20260608T0436JST_container_versions.json`, `results/aeon_gemma26_dflash_20260608T0436JST_container_target_audit.json`, `results/aeon_gemma26_dflash_20260608T0436JST_server.log` |
| AEON Qwen vLLM | `results/aeon_vllm_reproduction_preflight_20260608T0430JST.md`, `results/aeon_qwen36_dflash_20260608T0501JST_summary.md`, `results/aeon_qwen36_dflash_v2_20260608T0555JST_stop_point.md`, `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_row_manifest.json`, `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_server.log`, `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_nvfp4_checkpoint_audit.json`, `results/qwen_content_probe_20260608T0900JST_direct_chat_probes.json`, `results/aeon_qwen36_dflash_nothink_20260608T0834JST_row_manifest.json`, `results/aeon_qwen36_dflash_nothink_20260608T0834JST_openai_benchmark.json`, `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_summary.md`, `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_row_manifest.json`, `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_cutlass_summary.md`, `results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_target_audit.md`, `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_summary.md`, `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_row_manifest.json` |
| jethac vLLM Gemma 4 rebuilt-C image | `results/vllm_gemma4_rebuiltc_image_build_20260610T1440JST_r7_summary.md`, `results/vllm_gemma4_rebuiltc_image_build_20260610T1440JST_r7.log`, `results/vllm_gemma4_rebuiltc_image_import_probe_20260610T1545JST.md` |
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
3. Qwen3.6 NVFP4+DFlash now has passing AEON, AEON-FA2-derived `jethac/vllm`, and clean-FA2 `jethac/vllm` serving rows, but only with API-level thinking disabled; native-target proof exists for FA2 only, not for the FP4 weight/MoE path.
4. SGLang has a proven small-Qwen fp8/BF16 path but no Gemma path and no claim-ready FP4 KV benchmark; the latest fork records `1.7769x` fp8 KV capacity only by disabling graph capture for native FP4 KV, and the FP4 output quality still fails standardized checks with logprob evidence of first-token and late-drift divergence.
5. FlashInfer fork evidence is strong at source/JIT and standalone-kernel level, but not yet packaged as a clean runtime dependency with serving wins.
6. llama.cpp is practical and fast, but its GGUF accuracy path and native FP4/MXFP4 path remain separate workstreams.

## Next Matrix Updates

Update this matrix when:

- a new image is pulled or built with manifest/image-inspect evidence
- a source fork becomes a clean wheel/container
- a runtime row moves from startup/smoke to benchmark or accuracy
- build-target evidence changes from runtime capability only to binary/JIT proof
- dependency-aligned `jethac/vllm` Qwen3.6 fork row gains native FP4 weight/MoE evidence, NVFP4 KV capacity evidence, or a matched speedup/capacity win
