# Baseline Results

Status: first compact before row.

This file indexes before/after benchmark artifacts. Raw JSON remains the source of truth.

## 2026-06-07: vLLM Gemma 4 E4B W4A16 Before Row

Server observed on `thinkstationpgx-00b4`:

```text
/usr/local/bin/vllm serve google/gemma-4-E4B-it-qat-w4a16-ct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --host 0.0.0.0 \
  --port 8000
```

Artifacts:

- environment: `results/spark_doctor_before_vllm_gemma4_e4b_w4a16_20260607T1126Z.md`
- runtime probe: `results/runtime_probe_vllm_gemma4_e4b_w4a16_root_20260607T1136Z.json`
- benchmark: `results/vllm_gemma4_e4b_w4a16_before_compact_20260607T1126Z.json`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 24 | 64 | 0.032 | 1.271 | 51.65 |
| `medium_decode` | 36 | 192 | 0.032 | 3.779 | 51.25 |
| `long_prefill` | 2266 | 64 | 0.470 | 1.753 | 49.89 |

Interpretation:

- This is a valid first before row for an already-running vLLM server.
- It is not a cold-start benchmark.
- Runtime probe evidence shows the server was running as root from `/vllm-workspace` with `VLLM_USAGE_SOURCE=production-docker-image`.
- Runtime probe evidence shows `TORCH_CUDA_ARCH_LIST=8.7 8.9 9.0 10.0+PTX 12.0 12.1`.
- Runtime probe evidence found loaded vLLM extension paths including `_C.abi3.so`, `_C_stable_libtorch.abi3.so`, `_moe_C.abi3.so`, `_vllm_fa2_C.abi3.so`, and `_vllm_fa3_C.abi3.so`.
- It is not a full blessed-stack result because exact per-request kernel selection is still under investigation.
- It is useful for future before/after comparisons against the same model, prompts, settings, and server API.

## 2026-06-07: SGLang 26.05 Container Exploratory Row

Server tested on `thinkstationpgx-00b4`:

```text
nvcr.io/nvidia/sglang:26.05-py3
python3 -m sglang.launch_server Qwen/Qwen2.5-1.5B-Instruct
```

Key settings:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- dtype: `bfloat16`
- KV cache dtype: `torch.bfloat16`
- attention backend: `flashinfer`
- CUDA graphs: enabled
- port: `30000`
- vLLM on port `8000` was left running

Artifacts:

- smoke: `results/sglang_20260607T115213Z_chat_smoke.json`
- versions: `results/sglang_20260607T115213Z_python_versions.txt`
- CUDA object audit: `results/sglang_20260607T115213Z_cuda_so_audit_sglang.json`
- benchmark, 0.20 memory fraction: `results/sglang_bench_20260607T120315Z_openai_benchmark.json`
- long-prefill retry, 0.40 memory fraction: `results/sglang_bench_longprefill_20260607T120614Z_openai_benchmark.json`

Result summary:

| case | memory fraction | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `short_decode` | 0.20 | 44 | 64 | 0.647 | 1.705 | 60.50 | pass |
| `medium_decode` | 0.20 | 56 | 192 | 0.036 | 3.218 | 60.34 | pass |
| `long_prefill` | 0.20 | 1 | 1 | n/a | 0.035 | n/a | fail; insufficient/incorrect token budget |
| `long_prefill` | 0.40 | 2369 | 64 | 0.683 | 1.763 | 59.23 | pass |

Interpretation:

- NVIDIA's SGLang 26.05 ARM64 container can serve an OpenAI-compatible request on GB10.
- The container is a better first SGLang path than bare-metal pip.
- The first long-prefill failure is a real tuning/evidence point: `mem_fraction_static=0.20` can produce a too-small effective token budget for the benchmark while coexisting with the live vLLM service.
- The passing long-prefill retry used `mem_fraction_static=0.40`.
- This row is exploratory because it uses Qwen rather than Gemma and because sm121-specific kernel dispatch remains unresolved.
- The CUDA object audit found no explicit `sm_121` SASS in audited SGLang/FlashInfer objects.
- The SGLang log labeled the GB10 path as `SM120 (Blackwell)`, so this still needs upstream dispatch/packaging scrutiny.

## 2026-06-07: SGLang Gemma 4 E2B Blocker

Target:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- model: `google/gemma-4-E2B-it-qat-w4a16-ct`
- dtype: `bfloat16`
- memory fraction: `0.40`

Artifacts:

- default launch: `results/sglang_gemma4_e2b_w4a16_20260607T121536Z_server.log`
- language-only retry: `results/sglang_gemma4_e2b_w4a16_language_only_20260607T121751Z_server.log`

Result:

- Default launch exited before health while constructing the Gemma4 audio tower.
- The concrete exception was `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'`.
- A retry with `--language-only` exited before health during server argument validation.
- The concrete retry exception was `ValueError: requires at least one encoder urls to be set via --encoder-urls`.

Interpretation:

- This is a SGLang Gemma4 model-path blocker, not a proof of a GB10 kernel failure.
- It keeps SGLang marked as functional for some supported models but not yet a Gemma-class blessed path.

## 2026-06-07: FlashInfer SM121 Source/JIT Validation

Target:

- fork: `jethac/flashinfer`
- branch: `spark/hijinks-004-sm121-flashinfer`
- commit: `a42c8f07`
- host: `thinkstationpgx-00b4`

Artifact:

- `results/flashinfer_sm121_source_jit_20260607T1250Z.json`

Result:

- Installed vLLM container baseline: FlashInfer `0.6.8.post1`, CUDA `13.0`, real SM121 NVFP4 `mm_fp4` heuristic returned `["cudnn", "cutlass"]`.
- Installed SGLang container baseline: FlashInfer `0.6.10+cf494fca.nv26.5.cu132.50619265`, CUDA `13.2`, real SM121 NVFP4 `mm_fp4` heuristic returned `["cudnn", "cutlass"]`.
- Patched source: real SM121 NVFP4 `mm_fp4` heuristic returned `["b12x", "cutlass", "cudnn"]`.
- Source/JIT path built FP4 quantization under `/root/.cache/flashinfer/0.6.13/121a/cached_ops/fp4_quantization_120f`.
- Tiny forced-`b12x` NVFP4 GEMM produced finite BF16 output with cosine similarity `0.9882067441940308` against BF16 `torch.mm`.

Interpretation:

- This proves the FlashInfer fork makes the high-impact SM121 dispatch behavior better and that the `b12x` path can execute on GB10 when the source/JIT package set is consistent.
- This does not yet prove a serving-speed improvement. The deployable before/after needs a clean vLLM or SGLang image/wheel set with matching FlashInfer Python, JIT-cache/cubin, CUTLASS DSL, and CUDA targets.

## 2026-06-07: FlashInfer NVFP4 `mm_fp4` Auto Microbenchmark

Artifact:

- `results/flashinfer_mm_fp4_auto_microbench_20260607T1300Z.json`

Script:

- `scripts/flashinfer_mm_fp4_microbench.py`

Result summary:

| case | installed auto heuristic | installed mean ms | patched auto heuristic | patched mean ms | patched latency change |
|---|---|---:|---|---:|---:|
| `1x128x128` | `cudnn`, `cutlass` | 0.0727 | `b12x`, `cutlass`, `cudnn` | 0.0769 | +5.9% |
| `16x256x256` | `cudnn`, `cutlass` | 0.0654 | `b12x`, `cutlass`, `cudnn` | 0.0661 | +1.0% |
| `64x512x512` | `cudnn`, `cutlass` | 0.0651 | `b12x`, `cutlass`, `cudnn` | 0.0757 | +16.3% |

Interpretation:

- The patch did what it was supposed to do at dispatch level: SM121 auto-dispatch now includes `b12x`.
- On these three small dense NVFP4 `mm_fp4` cases, `b12x` was not faster than the installed container path.
- This does not rule out wins for model-shaped GEMMs, MoE `b12x` kernels, underfilled decode paths, or full serving stacks.
- Do not claim an end-to-end throughput improvement from this FlashInfer patch until a clean image/wheel set and model-level before/after rows prove it.

## 2026-06-07: FlashInfer NVFP4 `mm_fp4` Model-Shaped Proxy Microbenchmarks

Artifacts:

- installed dense-decode proxy: `results/flashinfer_mm_fp4_sglang_installed_dense_decode_20260607T161500Z.json`
- installed MoE proxy: `results/flashinfer_mm_fp4_sglang_installed_moe_expert_20260607T161500Z.json`
- patched dense-decode proxy: `results/flashinfer_mm_fp4_sglang_patched_modelshape_20260607T162000Z_dense_decode.json`
- patched MoE proxy: `results/flashinfer_mm_fp4_sglang_patched_modelshape_20260607T162000Z_moe_expert.json`

Target:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- installed FlashInfer: `0.6.10+cf494fca.nv26.05.cu132.50619265`
- patched source: `jethac/flashinfer@a42c8f07`
- patched source/JIT FlashInfer version: `0.6.13`
- GPU: `NVIDIA GB10`, compute capability `12.1`

Result summary, dense-decode proxy:

| case | installed heuristic | installed mean ms | patched heuristic | patched mean ms | patched latency change |
|---|---|---:|---|---:|---:|
| `1x4096x4096` | `cudnn`, `cutlass` | 0.0738 | `b12x`, `cutlass`, `cudnn` | 0.0893 | +21.1% |
| `4x4096x4096` | `cudnn`, `cutlass` | 0.0704 | `b12x`, `cutlass`, `cudnn` | 0.0677 | -3.9% |
| `16x4096x4096` | `cudnn`, `cutlass` | 0.0692 | `b12x`, `cutlass`, `cudnn` | 0.0620 | -10.4% |
| `1x8192x4096` | `cudnn`, `cutlass` | 0.0707 | `b12x`, `cutlass`, `cudnn` | 0.0857 | +21.3% |
| `4x8192x4096` | `cudnn`, `cutlass` | 0.0700 | `b12x`, `cutlass`, `cudnn` | 0.0786 | +12.4% |
| `16x8192x4096` | `cudnn`, `cutlass` | 0.0709 | `b12x`, `cutlass`, `cudnn` | 0.0741 | +4.5% |

Result summary, MoE-shaped proxy:

| case | installed heuristic | installed mean ms | patched heuristic | patched mean ms | patched latency change |
|---|---|---:|---|---:|---:|
| `1x14336x4096` | `cudnn`, `cutlass` | 0.1443 | `b12x`, `cutlass`, `cudnn` | 0.1543 | +6.9% |
| `4x14336x4096` | `cudnn`, `cutlass` | 0.1382 | `b12x`, `cutlass`, `cudnn` | 0.1510 | +9.3% |
| `16x14336x4096` | `cudnn`, `cutlass` | 0.1413 | `b12x`, `cutlass`, `cudnn` | 0.1535 | +8.6% |
| `1x4096x14336` | `cudnn`, `cutlass` | 0.1401 | `b12x`, `cutlass`, `cudnn` | 0.1688 | +20.5% |
| `4x4096x14336` | `cudnn`, `cutlass` | 0.1397 | `b12x`, `cutlass`, `cudnn` | 0.1551 | +11.0% |
| `16x4096x14336` | `cudnn`, `cutlass` | 0.1390 | `b12x`, `cutlass`, `cudnn` | 0.1546 | +11.2% |

Interpretation:

- The patched source/JIT path selected `b12x` on real GB10 and produced finite outputs with cosine similarity around `0.991` against BF16 `torch.mm`.
- The patched source/JIT container compiled FlashInfer FP4 GEMM under an SM121a-targeted path during this run.
- The model-shaped proxy result is not a speedup. Dense-decode proxies were mixed, and all MoE-shaped proxy cases were slower than the installed SGLang container path.
- This makes the FlashInfer predicate patch a correctness/enablement fix. The remaining performance question must be answered in fused serving paths, NVFP4 KV, model-specific quantization plumbing, CUDA graph behavior, or clean package builds.

## 2026-06-07: vLLM SM12x NVFP4 KV Routing And Deswizzle Probe

Artifact:

- `results/vllm_nvfp4_sm12x_routing_probe_20260607T171227Z.json`

Environment:

- host: `thinkstationpgx-00b4`
- GPU: `NVIDIA GB10`
- Torch CUDA capability: `[12, 1]`
- vLLM platform capability: `[12, 1]`
- vLLM capability-family check: `is_capability_family_120: true`
- Torch: `2.11.0+cu130`
- CUDA: `13.0`
- installed vLLM dependency context: `0.22.1`
- fork source revision: `jethac/vllm@8916796bc50926fd61e606718b194a71e2e31a24`

Result:

- SM12x NVFP4 KV prefill wrapper backend: `fa2`
- SM12x NVFP4 KV decode wrapper backend: `fa2`
- SM100-style NVFP4 fallback case remains `trtllm-gen`
- non-NVFP4 case remains `auto`
- vLLM FlashInfer JIT flag helper enables `-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
- probe result: `all_ok: true`

Interpretation:

- This proves the vLLM fork's routing predicate and vLLM-specific V-scale-factor deswizzle flag helper behave as intended on real GB10/SM121.
- `family 120` is intentional here: FA2 NVFP4 KV is the SM12x consumer-Blackwell family path. This is different from native FP4 MMA work, where `sm_121a` remains required for Spark.
- This does not prove the full vLLM fork installs cleanly, that FlashInfer FA2 NVFP4 KV kernels build/run, or that serving correctness/capacity/performance improves.
- Remaining proof must use the layout/correctness harness, including NHD and HND cosine checks, plus an end-to-end serve.

## 2026-06-08: FlashInfer FA2 NVFP4 KV Runtime Correctness Probe

Artifact:

- `results/flashinfer_nvfp4_kv_probe_20260608T023901JST.json`

Environment:

- source: `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
- import path: `/root/spark-validation/flashinfer-fa2-nvfp4-kv-sm121/flashinfer/__init__.py`
- source root supplied to JIT: `/root/spark-validation/flashinfer-fa2-nvfp4-kv-sm121`
- env: `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
- hardware key: `NVIDIA_GB10:sm_121:sms_48`
- Torch: `2.11.0+cu130`
- CUDA: `13.0`

Shape:

- `batch_size=2`
- `kv_len=64`
- `qo_len=16`
- `page_size=16`
- `num_kv_heads=2`
- `num_qo_heads=4`
- `head_dim=128`
- `dtype=bfloat16`
- backend: FlashInfer FA2 paged KV

Result:

| operation | layout | cosine | max abs error | passed |
|---|---:|---:|---:|---|
| decode | NHD | 0.9999995232 | 0.0078125 | true |
| prefill | NHD | 0.9999998808 | 0.0 | true |
| decode | HND | 0.9999994636 | 0.0078125 | true |
| prefill | HND | 0.9999998808 | 0.0 | true |

Interpretation:

- This is the first GB10 runtime proof that the patched FlashInfer FA2 NVFP4 paged-KV path builds, runs, and reads vLLM-style swizzled V scale factors correctly when the in-kernel de-swizzle macro is enabled.
- It is stronger than the vLLM routing probe because it executes FlashInfer kernels and compares against dequantized reference attention for both NHD and HND layouts.
- It is still not an end-to-end vLLM serving proof. It does not prove clean wheel packaging, vLLM metadata integration, fp8-vs-NVFP4 KV capacity, output quality, CUDA graph replay, or serving throughput.

## 2026-06-08: Gemma 4 26B-Shaped FlashInfer NVFP4 KV Probe

Artifacts:

- `results/flashinfer_nvfp4_kv_probe_gemma4_26b_sliding_1024_20260608T0340JST.json`
- `results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_20260608T0335JST.json`

Config source:

- cached `google/gemma-4-26B-A4B-it` config under the benchmark host's Hugging Face cache
- text attention heads: `num_attention_heads=16`
- sliding/local KV heads: `num_key_value_heads=8`
- global/full KV heads: `num_global_key_value_heads=2`
- sliding/local `head_dim=256`
- global/full `global_head_dim=512`
- page size tested: `16`

Sliding/local result:

- shape: `batch_size=2`, `kv_len=1024`, `qo_len=128`, `num_qo_heads=16`, `num_kv_heads=8`, `head_dim=256`
- outcome: NHD decode, NHD prefill, HND decode, and HND prefill all passed.
- minimum cosine: `0.9999961853`
- maximum absolute error: `0.015625`

Global/full result:

- shape: `batch_size=2`, `kv_len=128`, `qo_len=16`, `num_qo_heads=16`, `num_kv_heads=2`, `head_dim=512`
- outcome: all NHD/HND decode/prefill operations failed before numerical comparison.
- failure class: FlashInfer FA2 paged KV invalid configuration from `include/flashinfer/attention/prefill.cuh:3215`
- representative message: `Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 ...`

Interpretation:

- This narrows the vLLM NVFP4-KV blocker. The patched FlashInfer FA2 path is correct for Gemma 4 26B sliding/local attention geometry, including vLLM-style V-scale-factor de-swizzle.
- Gemma 4 26B also has global/full attention layers with `global_head_dim=512`; that geometry currently fails in the standalone FlashInfer probe.
- Do not start or bless a Gemma 4 26B vLLM `--kv-cache-dtype nvfp4` serving row until the `D=512` global path is fixed, routed to a proven fallback, or shown irrelevant for the specific model path.

## 2026-06-07: vLLM Gemma 4 26B A4B Compact MoE Serving Check

Target:

- image: `vllm/vllm-openai:latest-cu130`
- vLLM: `0.20.0`
- PyTorch: `2.11.0+cu130`
- FlashInfer: `0.6.8.post1`
- model: `google/gemma-4-26B-A4B-it`
- served model: `gemma4-26b-a4b-it`
- settings: `--max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096`

Artifacts:

- benchmark: `results/vllm_gemma4_26b_a4b_bf16_compact_20260607T131917Z.json`
- server log: `results/vllm_gemma4_26b_a4b_bf16_20260607T131917Z_server.log`
- run info: `results/vllm_gemma4_26b_a4b_bf16_20260607T131917Z_run_info.txt`
- default-setting failure: `results/vllm_gemma4_26b_a4b_bf16_default_fail_20260607T131837Z_server.log`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 1.228 | 3.832 | 24.58 |
| `medium_decode` | 40 | 192 | 0.137 | 8.039 | 24.30 |
| `long_prefill` | 2270 | 64 | 0.551 | 3.223 | 23.95 |

Startup observations:

- The default launch failed before readiness because Gemma 4 disabled chunked multimodal input while vLLM's default `max_num_batched_tokens=2048` was below Gemma's `max_tokens_per_mm_item=2496`.
- Retrying with `--max-num-batched-tokens 4096` reached readiness.
- BF16 checkpoint loading took about 6m11s for 48.07 GiB of safetensors; CUDA graph capture took another 11s.
- The server log selected `TRITON_ATTN` for attention and `TRITON Unquantized MoE` from `['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']`.

Interpretation:

- This is a useful compact MoE serving row: Gemma 4 26B A4B serves successfully on GB10 through vLLM and sustains about 24 tok/s decode on this three-case OpenAI-compatible harness.
- This row does not exercise the FlashInfer NVFP4 `mm_fp4` dispatch fix. The vLLM path is BF16/unquantized MoE and explicitly chose Triton.
- The `--max-num-batched-tokens 4096` requirement should be part of future Gemma 4 vLLM recipes for this image/model combination.

## 2026-06-07: vLLM Gemma 4 12B Unified Source/Precompiled Probe

Target:

- base image: `vllm/vllm-openai:latest-cu130`
- vLLM source commit: `da1daf40bf18e5eaae04f26a80a537c8168a8bc2`
- install mode: editable source install using `VLLM_USE_PRECOMPILED=1`, `VLLM_MAIN_CUDA_VERSION=13.0`, and matching precompiled wheel metadata
- Transformers: main snapshot installed from `git+https://github.com/huggingface/transformers.git@effde20942e3f82a1b97449f60b3a48c5ff96145`
- model: `google/gemma-4-12B-it`
- served model: `gemma4-12b-it`
- settings: `--max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096`

Artifacts:

- launcher: `scripts/run_vllm_gemma4_12b_unified_probe.sh`
- import probe: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_import_probe.txt`
- chat smoke: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_openai_chat_smoke.json`
- compact benchmark: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_compact_benchmark.json`
- runtime probe: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_runtime_probe.json`
- server log: `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_server.log`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.268 | 8.448 | 7.82 |
| `medium_decode` | 40 | 192 | 0.261 | 25.318 | 7.66 |
| `long_prefill` | 2270 | 64 | 0.976 | 9.252 | 7.73 |

Startup observations:

- The import probe recorded GB10 compute capability `[12, 1]`, PyTorch `2.11.0+cu130`, vLLM `0.1.dev1+gda1daf40b`, Transformers `5.10.0.dev0`, and `has_gemma4_unified: true`.
- Released/current local images checked before this run did not expose `Gemma4UnifiedForConditionalGeneration` through the registry: `vllm/vllm-openai:latest-cu130`, `vllm/vllm-openai:cu130-nightly-aarch64`, and the earlier `gemma4-vllm:v0.22.1-pip` image.
- Upstream vLLM main contains the architecture, and wheel metadata existed for `da1daf40`; the current main commit tested during this audit did not have matching precompiled wheel metadata yet.
- Stale `flashinfer-jit-cache` from the base image had to be removed. Without cleanup, FlashInfer reported a JIT-cache package version mismatch; after `pip uninstall` plus deleting the remaining package directory and dist-info, the server reached health.
- The server resolved `Gemma4UnifiedForConditionalGeneration`, loaded 22.28 GiB of checkpoints, and created a 336,566-token GPU KV cache for 8,192-token requests.
- vLLM forced `TRITON_ATTN` because Gemma 4 has heterogeneous head dimensions. The log also reported Triton JIT compilation during inference for `_compute_slot_mapping_kernel` and `kernel_unified_attention`.

Interpretation:

- This overturns the older local result that 12B was simply not usable in vLLM on Spark: the architecture path can run on GB10 when vLLM and Transformers are new enough and the FlashInfer JIT-cache package set is consistent.
- This is not a blessed clean stack. It required source overlay, a specific precompiled-wheel commit, Transformers main, and manual stale-package cleanup inside the container.
- The measured decode speed is much slower than the 26B A4B vLLM row and the llama.cpp 26B Q4_0 row. Treat it as a compatibility proof and a packaging target, not a performance win.
- The next proof is a clean release/nightly container that starts the same model without source surgery, then a quantized/MTP row that explains whether the SM120 reference results transfer to GB10 `sm_121`.

## 2026-06-07: vLLM Gemma 4 26B A4B QAT-Unquantized Probe

Target:

- image: `vllm/vllm-openai:latest-cu130`
- model: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`
- served model: `gemma4-26b-a4b-it-qat-q4_0-unquantized`
- settings: `--max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096`

Artifacts:

- short benchmark: `results/vllm_gemma4_26b_a4b_qat_unquantized_short_20260607T133040Z.json`
- server log: `results/vllm_gemma4_26b_a4b_qat_unquantized_20260607T133040Z_server.log`
- run info: `results/vllm_gemma4_26b_a4b_qat_unquantized_20260607T133040Z_run_info.txt`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 1.248 | 3.857 | 24.53 |

Interpretation:

- The QAT-unquantized snapshot loads and serves in vLLM with the same corrected batching setting.
- It is not a direct quantized/NVFP4 serving row in this image. The engine config reported `quantization=None`, `dtype=torch.bfloat16`, and the same `TRITON Unquantized MoE` backend.
- If the campaign needs to prove NVFP4 end-to-end impact, this snapshot name is not enough. The run must prove the actual quantization/backend path from logs or profiler evidence.

## 2026-06-07: llama.cpp Gemma 4 26B Q4_0 Practical Serving Row

Target:

- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`
- build: `308f61c31 (9536)`
- model: `/home/jethac/gemma4-vllm/models/gemma-4-26B_q4_0-it.gguf`
- alias: `gemma4-26b-q4_0-gguf`
- settings: `--ctx-size 8192 --gpu-layers all --reasoning off`

Artifacts:

- smoke: `results/llamacpp_gemma4_26b_q4_0_chat_smoke_20260607T135911Z.json`
- serving benchmark: `results/llamacpp_gemma4_26b_q4_0_compact_20260607T135911Z.json`
- `llama-bench`: `results/llamacpp_gemma4_26b_q4_0_bench_20260607T135911Z.txt`
- server log: `results/llamacpp_gemma4_26b_q4_0_20260607T135911Z_server.log`
- `spark_doctor`: `results/spark_doctor_llamacpp_gemma4_26b_q4_0_20260607T135911Z.md`
- logprobs probe: `results/gguf_logprobs_probe_llamacpp_b9536_reasoning_off_20260607T135911Z.json`

Result summary:

| case | prompt tokens | generated tokens | TTFT seconds | total seconds | decode tok/s |
|---|---:|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.107 | 0.939 | 76.94 |
| `medium_decode` | 40 | 192 | 0.106 | 2.633 | 75.97 |

`llama-bench`:

| test | throughput |
|---|---:|
| `pp512` | 3021.76 +/- 34.41 tok/s |
| `tg128` | 77.35 +/- 0.13 tok/s |

Interpretation:

- llama.cpp is now blessed as a practical single-Spark serving path for this GGUF model.
- `--reasoning off` is required for normal OpenAI chat `message.content` output on this Gemma 4 server path.
- Server logs confirm CUDA on `NVIDIA GB10`, `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, and `BLACKWELL_NATIVE_FP4 = 1`.
- This row is Q4_0 GGUF, not NVFP4/MXFP4 GGUF. The measured win is practical 4-bit bandwidth reduction plus mature CUDA graph/quantized-serving kernels on `sm_121`; it does not prove native `sm_121a` FP4 tensor-core MMA dispatch.
- GGUF lm-eval accuracy remains blocked. The same server still exposes logprobs under `choices[0].logprobs.content`, not the `tokens` and `token_logprobs` shape expected by the existing lm-eval adapter.

## 2026-06-07: LiteRT-LM Gemma 4 E2B CPU/GPU Smoke

Target:

- venv: `/home/jethac/spark-validation/litert-lm-20260607T140617Z/venv`
- package: `litert-lm==0.13.1`
- API package: `litert-lm-api==0.13.1`
- model: `litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm`
- host: Linux `aarch64`, Python 3.12.3

Artifacts:

- import probe: `results/litert_lm_20260607T140617Z_import_probe.json`
- run info: `results/litert_lm_20260607T140617Z_run_info.txt`
- CPU chat smoke: `results/litert_lm_cpu_e2b_smoke_no_max_telemetry.json`
- CPU bad-KV smoke: `results/litert_lm_cpu_e2b_smoke4_telemetry.json`
- GPU chat smoke after group fix: `results/litert_lm_gpu_e2b_smoke_after_groups_telemetry.json`
- CPU benchmark: `results/litert_lm_cpu_e2b_bench_256p64d_telemetry.json`
- GPU benchmark: `results/litert_lm_gpu_e2b_bench_256p64d_telemetry.json`

Result summary:

| path | prefill tokens | decode tokens | result |
|---|---:|---:|---|
| CPU `run` chat | n/a | n/a | returned `spark-ok`, exit 0 |
| CPU `benchmark` | 256 | 64 | prefill `125.77` tok/s, decode `43.57` tok/s, init `0.3235` s, TTFT `2.0584` s |
| GPU `benchmark` | 256 | 64 | prefill `1773.07` tok/s, decode `43.70` tok/s, init `2.5860` s, TTFT `0.1673` s |
| GPU `run` chat | n/a | n/a | printed `spark-ok`, then exited `returncode=-11` |
| CPU `run --max-num-tokens 512` | n/a | n/a | failed with `DYNAMIC_UPDATE_SLICE` / `Failed to allocate tensors`; CLI still returned 0 |

Operational observations:

- `litert-lm run` reads all non-TTY stdin before loading the model. The telemetry wrapper originally left stdin open, which made LiteRT-LM appear to hang in `anon_pipe_read`.
- The telemetry wrapper now passes `stdin=subprocess.DEVNULL` for benchmark commands.
- Hugging Face cache subdirectories under `/home/jethac/.cache/huggingface` were root-owned from earlier work and caused permission-denied download errors. Ownership was restored to `jethac`.
- `jethac` was added to `render` and `video` so LiteRT GPU can open `/dev/dri` nodes. This removed the device permission errors but did not fix the GPU chat `SIGSEGV`.
- GPU logs still show `Failed to load OpenCL library with dlopen: libOpenCL.so`; LiteRT falls back to the ICD loader.

Interpretation:

- LiteRT-LM is viable on Spark as a lightweight Gemma E2B CPU path and as a benchmarkable GPU path.
- It is not yet blessed for GPU chat/serving because the `run` command can generate text and still crash at process exit.
- The GPU benchmark has a clear prefill advantage over CPU on this small row, but decode is effectively tied. This should be treated as a complement to llama.cpp/vLLM, not as the main path for extracting GB10 tensor throughput.
