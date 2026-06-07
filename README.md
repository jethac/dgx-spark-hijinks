# DGX Spark Hijinks

Public record for making DGX Spark / ThinkStation PGX useful and boring for local AI.

The short version: GB10 is `sm_121`. A lot of the AI stack still assumes x86_64, CUDA 12, datacenter Blackwell, RTX Blackwell, or older CUDA architectures. This repository tracks the work to turn those rough edges into reproducible fixes.

We currently have one Spark-class machine available: the ThinkStation PGX workstation observed as `NVIDIA GB10`. Multi-Spark work is tracked, but not executable yet.

## Start Here

- Diagnosis: [docs/DGX_SPARK_DIAGNOSIS.md](docs/DGX_SPARK_DIAGNOSIS.md)
- Full solution plan: [docs/DGX_SPARK_SOLUTIONS.md](docs/DGX_SPARK_SOLUTIONS.md)
- Initial benchmark report: [docs/BENCHMARKING_REPORT.md](docs/BENCHMARKING_REPORT.md)
- Problems encountered: [docs/GEMMA4_ON_DGX_SPARK.md](docs/GEMMA4_ON_DGX_SPARK.md)
- Issue tracker map: [docs/ISSUE_TRACKER.md](docs/ISSUE_TRACKER.md)

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

For a compact vLLM OpenAI-compatible smoke test:

```bash
python3 scripts/openai_chat_smoke.py --url http://127.0.0.1:8000 --model MODEL_NAME
```

## What Counts As Fixed

The campaign is not complete when one model runs once.

It is complete when the repo contains a tested, reproducible stack where:

- the machine is identified as GB10 / `sm_121`
- the installed stack is ARM64 + CUDA 13 compatible
- vLLM can serve a Gemma-class model without dependency surgery
- llama.cpp GGUF throughput and lm-eval accuracy paths are separated and validated
- HF fallback failures include memory/process telemetry
- NVFP4 is either validated on Spark or clearly marked unsafe
- benchmarks record backend, selected kernels, CUDA architecture, memory state, and exact commands

## Current State

This repo starts from the first Gemma 4 benchmark campaign. That run showed useful vLLM safetensors results for several model rows, but also exposed ecosystem problems around `sm_121` packaging, vLLM model support, HF fallback reliability, GGUF lm-eval logprobs, long benchmark design, and missing observability.

The GitHub Issues are the source of truth for active work.

