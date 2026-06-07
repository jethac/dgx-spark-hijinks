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
