# DGX Spark Hijinks

Public record for making DGX Spark / ThinkStation PGX useful and boring for local AI.

The short version: GB10 is `sm_121`. A lot of the AI stack still assumes x86_64, CUDA 12, datacenter Blackwell, RTX Blackwell, or older CUDA architectures. This repository tracks the work to turn those rough edges into reproducible fixes.

We currently have one Spark-class machine available: the ThinkStation PGX workstation observed as `NVIDIA GB10`. Multi-Spark work is tracked, but not executable yet.

Mission: this machine costs roughly 900k JPY. It needs to be as performant as the silicon is capable of, not merely "technically running."

Latest compact signal: Gemma 4 26B A4B serves through `vllm/vllm-openai:latest-cu130` on GB10 at about 24 tok/s decode, after setting `--max-num-batched-tokens 4096`. That row proves a useful MoE serving baseline, but vLLM selected BF16/unquantized Triton MoE, so it does not yet prove the FlashInfer NVFP4 path.

## Start Here

- Diagnosis: [docs/DGX_SPARK_DIAGNOSIS.md](docs/DGX_SPARK_DIAGNOSIS.md)
- Full solution plan: [docs/DGX_SPARK_SOLUTIONS.md](docs/DGX_SPARK_SOLUTIONS.md)
- Initial benchmark report: [docs/BENCHMARKING_REPORT.md](docs/BENCHMARKING_REPORT.md)
- Problems encountered: [docs/GEMMA4_ON_DGX_SPARK.md](docs/GEMMA4_ON_DGX_SPARK.md)
- GGUF/llama.cpp status: [docs/GGUF_LLAMA_CPP_STATUS.md](docs/GGUF_LLAMA_CPP_STATUS.md)
- Issue tracker map: [docs/ISSUE_TRACKER.md](docs/ISSUE_TRACKER.md)
- Fork/worktree policy: [docs/FORKS_AND_WORKTREES.md](docs/FORKS_AND_WORKTREES.md)
- SGLang notes: [docs/SGLANG_ON_DGX_SPARK.md](docs/SGLANG_ON_DGX_SPARK.md)
- Before/after benchmark protocol: [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md)
- Baseline results: [docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md)
- NVFP4 dependency map: [docs/NVFP4_DEPENDENCY_MAP.md](docs/NVFP4_DEPENDENCY_MAP.md)
- FlashInfer performance hypotheses: [docs/FLASHINFER_PERFORMANCE_HYPOTHESES.md](docs/FLASHINFER_PERFORMANCE_HYPOTHESES.md)
- Upstream latest release audit: [docs/UPSTREAM_LATEST_RELEASE_AUDIT.md](docs/UPSTREAM_LATEST_RELEASE_AUDIT.md)
- Runtime availability: [docs/RUNTIME_AVAILABILITY.md](docs/RUNTIME_AVAILABILITY.md)
- PyTorch sm121 support: [docs/PYTORCH_SM121_SUPPORT.md](docs/PYTORCH_SM121_SUPPORT.md)
- Failure annotations: [docs/FAILURE_ANNOTATIONS.md](docs/FAILURE_ANNOTATIONS.md)

## First Commands

Run these on the Spark / PGX host:

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

To capture runtime process evidence for a local server:

```bash
python3 scripts/runtime_process_probe.py \
  --url http://127.0.0.1:8000 \
  --match vllm \
  --output results/runtime_probe_vllm.json
```

To annotate captured benchmark and server failures:

```bash
python3 scripts/failure_annotator.py \
  --results-dir results \
  --output-json results/failure_annotations.json \
  --output-md docs/FAILURE_ANNOTATIONS.md
```

To capture the runtime availability matrix:

```bash
python3 scripts/runtime_availability_matrix.py \
  --include-docker-images \
  --output results/runtime_availability.json
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

This repo starts from the first Gemma 4 benchmark campaign. That run showed useful vLLM safetensors results for several model rows, but also exposed ecosystem problems around `sm_121` packaging, vLLM model support, HF fallback reliability, GGUF lm-eval logprobs, long benchmark design, and missing observability. A first NVIDIA SGLang 26.05 container smoke now passes on GB10, but it still needs Gemma and NVFP4 validation.

The GitHub Issues are the source of truth for active work. If we need source changes to upstream libraries, the change goes through a `jethac` fork, a submodule under `third_party/`, and an issue-named worktree.
