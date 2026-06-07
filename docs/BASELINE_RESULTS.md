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
- GGUF lm-eval accuracy remains blocked. The same server still exposes logprobs under `choices[0].logprobs.content`, not the `tokens` and `token_logprobs` shape expected by the existing lm-eval adapter.
