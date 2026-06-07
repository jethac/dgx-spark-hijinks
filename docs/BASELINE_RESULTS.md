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
