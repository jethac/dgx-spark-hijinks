# DGX Spark Hijinks

Public record for making DGX Spark-class GB10 systems useful and boring for local AI.

The short version: GB10 is `sm_121`. A lot of the AI stack still assumes x86_64, CUDA 12, datacenter Blackwell, RTX Blackwell, or older CUDA architectures. This repository tracks the work to turn those rough edges into reproducible fixes.

We currently have one Spark-class GB10 machine available. Multi-Spark work is tracked, but not executable yet.

Mission: this machine costs roughly 900k JPY. It needs to be as performant as the silicon is capable of, not merely "technically running."

Latest compact signal: Gemma 4 26B A4B serves through `vllm/vllm-openai:latest-cu130` on GB10 at about 24 tok/s decode, after setting `--max-num-batched-tokens 4096`. Gemma 4 12B also serves now, but only through a source/precompiled vLLM probe at upstream commit `da1daf40` plus Transformers main and stale FlashInfer JIT-cache cleanup; that row is about 7.7 tok/s and forces Triton attention. Qwen is now tracked as a first-class speed/capacity lane: SGLang Qwen2.5 1.5B BF16/auto and fp8 both serve at about 58-59 tok/s, fp8 roughly doubles KV pool tokens, and patched FP4 KV exposes the expected larger pool but is not usable yet because it only serves with graph paths disabled at about 0.28 tok/s. The current FlashInfer SM121 `b12x` patch is dispatch enablement, not a proven speedup: model-shaped SGLang proxy microbenchmarks were mixed-to-slower.

## Start Here

- Diagnosis: [docs/DGX_SPARK_DIAGNOSIS.md](docs/DGX_SPARK_DIAGNOSIS.md)
- Full solution plan: [docs/DGX_SPARK_SOLUTIONS.md](docs/DGX_SPARK_SOLUTIONS.md)
- Initial benchmark report: [docs/BENCHMARKING_REPORT.md](docs/BENCHMARKING_REPORT.md)
- Problems encountered: [docs/GEMMA4_ON_DGX_SPARK.md](docs/GEMMA4_ON_DGX_SPARK.md)
- GGUF/llama.cpp status: [docs/GGUF_LLAMA_CPP_STATUS.md](docs/GGUF_LLAMA_CPP_STATUS.md)
- Issue tracker map: [docs/ISSUE_TRACKER.md](docs/ISSUE_TRACKER.md)
- Fork/worktree policy: [docs/FORKS_AND_WORKTREES.md](docs/FORKS_AND_WORKTREES.md)
- SGLang notes: [docs/SGLANG_ON_DGX_SPARK.md](docs/SGLANG_ON_DGX_SPARK.md)
- Qwen notes: [docs/QWEN_ON_DGX_SPARK.md](docs/QWEN_ON_DGX_SPARK.md)
- Before/after benchmark protocol: [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md)
- Baseline results: [docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md)
- Remediation matrix: [docs/REMEDIATION_MATRIX.md](docs/REMEDIATION_MATRIX.md)
- Spark smoke suite: [docs/SPARK_SMOKE_SUITE.md](docs/SPARK_SMOKE_SUITE.md)
- NVFP4 dependency map: [docs/NVFP4_DEPENDENCY_MAP.md](docs/NVFP4_DEPENDENCY_MAP.md)
- FlashInfer performance hypotheses: [docs/FLASHINFER_PERFORMANCE_HYPOTHESES.md](docs/FLASHINFER_PERFORMANCE_HYPOTHESES.md)
- Upstream latest release audit: [docs/UPSTREAM_LATEST_RELEASE_AUDIT.md](docs/UPSTREAM_LATEST_RELEASE_AUDIT.md)
- Runtime availability: [docs/RUNTIME_AVAILABILITY.md](docs/RUNTIME_AVAILABILITY.md)
- PyTorch sm121 support: [docs/PYTORCH_SM121_SUPPORT.md](docs/PYTORCH_SM121_SUPPORT.md)
- Failure annotations: [docs/FAILURE_ANNOTATIONS.md](docs/FAILURE_ANNOTATIONS.md)
- HF fallback telemetry: [docs/HF_FALLBACK_TELEMETRY.md](docs/HF_FALLBACK_TELEMETRY.md)

## First Commands

Run these on the Spark-class GB10 host:

```bash
python3 scripts/spark_doctor.py --json > results/spark_doctor.json
python3 scripts/spark_doctor.py > results/spark_doctor.md
```

If testing a llama.cpp OpenAI-compatible server for GGUF accuracy:

```bash
python3 scripts/gguf_logprobs_probe.py --url http://127.0.0.1:8080
```

For a compact OpenAI-compatible smoke test against vLLM, SGLang, llama.cpp, or another compatible server:

```bash
python3 scripts/openai_chat_smoke.py --url http://127.0.0.1:8000 --model MODEL_NAME
```

For before/after measurement against an OpenAI-compatible server:

```bash
python3 scripts/openai_serving_benchmark.py \
  --url http://127.0.0.1:8000 \
  --backend vllm \
  --phase before \
  --run-id RUN_ID \
  --output results/RUN_ID.json
```

To inspect installed CUDA extension binaries:

```bash
python3 scripts/cuda_so_audit.py --package vllm --package flashinfer \
  --output results/cuda_so_audit_vllm_flashinfer.json
```

For a FlashInfer NVFP4 `mm_fp4` kernel-level dispatch check:

```bash
python3 scripts/flashinfer_mm_fp4_microbench.py \
  --phase before \
  --run-id flashinfer-mm-fp4-before \
  --container CONTAINER_TAG \
  --output results/flashinfer_mm_fp4_before.json
```

For a FlashInfer FA2 NVFP4 paged-KV correctness check against a source checkout:

```bash
PYTHONPATH=/path/to/flashinfer-src:./scripts \
python3 scripts/flashinfer_nvfp4_kv_probe.py \
  --flashinfer-source-root /path/to/flashinfer-src \
  --output results/flashinfer_nvfp4_kv_probe.json
```

To capture runtime process evidence for a local server:

```bash
python3 scripts/runtime_process_probe.py \
  --url http://127.0.0.1:8000 \
  --match vllm \
  --output results/runtime_probe_vllm.json
```

To record a complete OpenAI-compatible serving row manifest:

```bash
python3 scripts/record_openai_serving_row.py \
  --backend sglang \
  --phase before \
  --run-id RUN_ID \
  --url http://127.0.0.1:30000 \
  --model MODEL_NAME \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --server-log results/RUN_ID_server.log
```

To annotate captured benchmark and server failures:

```bash
python3 scripts/failure_annotator.py \
  --results-dir results \
  --output-json results/failure_annotations.json \
  --output-md docs/FAILURE_ANNOTATIONS.md
```

To run a fragile fallback command with live memory/process telemetry:

```bash
python3 scripts/run_with_telemetry.py \
  --run-id RUN_ID \
  --backend hf \
  --model MODEL_ID \
  --timeout-s 2400 \
  --interval-s 5 \
  --output results/RUN_ID_telemetry.json \
  -- python3 YOUR_COMMAND.py
```

To capture the runtime availability matrix:

```bash
python3 scripts/runtime_availability_matrix.py \
  --include-docker-images \
  --output results/runtime_availability.json
```

To run the compact Spark smoke suite:

```bash
python3 scripts/spark_smoke_suite.py \
  --run-id spark-smoke-before \
  --vllm-url http://127.0.0.1:8000 \
  --vllm-model MODEL \
  --llamacpp-url http://127.0.0.1:18082 \
  --llamacpp-model MODEL \
  --hf-command "python3 path/to/tiny_hf_probe.py" \
  --mtp-command "python3 path/to/tiny_mtp_probe.py"
```

Runtime tracks:

- vLLM: [recipes/single_spark_vllm.md](recipes/single_spark_vllm.md)
- SGLang: [recipes/single_spark_sglang.md](recipes/single_spark_sglang.md)
- llama.cpp serving: [recipes/llamacpp_serving.md](recipes/llamacpp_serving.md)
- llama.cpp GGUF accuracy: [recipes/gguf_llamacpp_accuracy.md](recipes/gguf_llamacpp_accuracy.md)
- LiteRT-LM: [recipes/litert_lm_spark.md](recipes/litert_lm_spark.md)
- NVFP4: [recipes/nvfp4_spark.md](recipes/nvfp4_spark.md)

## What Counts As Fixed

The campaign is not complete when one model runs once.

It is complete when the repo contains a tested, reproducible stack where:

- the machine is identified as GB10 / `sm_121`
- the installed stack is ARM64 + CUDA 13 compatible
- vLLM can serve a Gemma-class model without dependency surgery
- SGLang and LiteRT-LM have explicit go/no-go decisions
- llama.cpp GGUF throughput and lm-eval accuracy paths are separated and validated
- HF fallback failures include memory/process telemetry
- NVFP4 is either validated on Spark or clearly marked unsafe
- benchmarks record backend, selected kernels, CUDA architecture, memory state, and exact commands

## Current State

This repo starts from an initial personal Gemma 4 benchmark run on the machine. That run showed useful vLLM safetensors results for several model rows, but also exposed ecosystem problems around `sm_121` packaging, vLLM model support, HF fallback reliability, GGUF lm-eval logprobs, long benchmark design, and missing observability. A first NVIDIA SGLang 26.05 container smoke now passes on GB10, but it still needs Gemma and NVFP4 validation. LiteRT-LM now has a proven E2B CPU path and benchmarkable GPU path, but GPU chat exits with `returncode=-11` after producing output.

The GitHub Issues are the source of truth for active work. If we need source changes to upstream libraries, the change goes through a `jethac` fork, a submodule under `third_party/`, and an issue-named worktree.
