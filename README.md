# DGX Spark Hijinks

Public record for making DGX Spark-class GB10 systems useful and boring for local AI.

The short version: GB10 is `sm_121`. A lot of the AI stack still assumes x86_64, CUDA 12, datacenter Blackwell, RTX Blackwell, or older CUDA architectures. This repository tracks the work to turn those rough edges into reproducible fixes.

We currently have one Spark-class GB10 machine available. Multi-Spark work is tracked, but not executable yet.

Mission: this machine costs roughly 900k JPY. It needs to be as performant as the silicon is capable of, not merely "technically running."

Latest compact signal: AEON's Gemma 4 26B A4B NVFP4+DFlash vLLM image now serves locally on GB10 with a warmed compact row at about 48 tok/s short decode, 54 tok/s medium decode, and 98 tok/s on the long-prefill shape. That is the first local vLLM Gemma 26B row materially above the earlier BF16/unquantized vLLM row at about 24 tok/s. Gemma 4 12B also serves now, but only through a source/precompiled vLLM probe at upstream commit `da1daf40` plus Transformers main and stale FlashInfer JIT-cache cleanup; that row is about 7.7 tok/s and forces Triton attention. Qwen is now tracked as a first-class speed/capacity lane: SGLang Qwen2.5 1.5B BF16/auto and fp8 both serve at about 58-59 tok/s, fp8 roughly doubles KV pool tokens, patched FP4 KV exposes the expected larger pool but is not usable yet, and llama.cpp Qwen2.5 1.5B Q4_K_M serves at about 167-175 tok/s. The current FlashInfer SM121 `b12x` patch is dispatch enablement, not a proven speedup: model-shaped SGLang proxy microbenchmarks were mixed-to-slower.

Next vLLM proof lane: AEON Gemma and AEON Qwen3.6 NVFP4+DFlash are now locally reproduced; Qwen requires OpenAI `chat_template_kwargs={"enable_thinking": false}` to return normal `message.content` in the compact benchmark. The `jethac/vllm` Qwen branch has a passing derived AEON row after aligning `compressed-tensors`, adding `humming-kernels`, and restoring AEON's FA2 binary. That proves fork runtime parity, not a clean upstream install or native `sm_121a` target. The fork head now adds `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1` as the clean-image knob for replacing the AEON FA2 binary.

## Start Here

- Diagnosis: [docs/DGX_SPARK_DIAGNOSIS.md](docs/DGX_SPARK_DIAGNOSIS.md)
- Full solution plan: [docs/DGX_SPARK_SOLUTIONS.md](docs/DGX_SPARK_SOLUTIONS.md)
- Solution status: [docs/SOLUTIONS_STATUS.md](docs/SOLUTIONS_STATUS.md)
- Compatibility board: [docs/COMPATIBILITY_BOARD.md](docs/COMPATIBILITY_BOARD.md)
- Live GB10 runbook: [docs/LIVE_GB10_RUNBOOK.md](docs/LIVE_GB10_RUNBOOK.md)
- Incident 2026-06-09 (vLLM OOM deadlock; memory mitigations): [docs/INCIDENT_20260609_OOM_DEADLOCK.md](docs/INCIDENT_20260609_OOM_DEADLOCK.md)
- Gemma compatibility plan (3 → 3n → 4 ladder): [docs/GEMMA_COMPATIBILITY_PLAN.md](docs/GEMMA_COMPATIBILITY_PLAN.md)
- vLLM Gemma 4 rungs modification plan: [docs/VLLM_GEMMA_RUNGS.md](docs/VLLM_GEMMA_RUNGS.md)
- Wheel/container matrix: [docs/WHEEL_CONTAINER_MATRIX.md](docs/WHEEL_CONTAINER_MATRIX.md)
- Initial benchmark report: [docs/BENCHMARKING_REPORT.md](docs/BENCHMARKING_REPORT.md)
- Problems encountered: [docs/GEMMA4_ON_DGX_SPARK.md](docs/GEMMA4_ON_DGX_SPARK.md)
- GGUF/llama.cpp status: [docs/GGUF_LLAMA_CPP_STATUS.md](docs/GGUF_LLAMA_CPP_STATUS.md)
- Issue tracker map: [docs/ISSUE_TRACKER.md](docs/ISSUE_TRACKER.md)
- Fork/worktree policy: [docs/FORKS_AND_WORKTREES.md](docs/FORKS_AND_WORKTREES.md)
- Solution coverage audit: `scripts/solution_coverage_audit.py`
- AEON prior-art port map: [docs/AEON_PRIOR_ART_PORT_MAP.md](docs/AEON_PRIOR_ART_PORT_MAP.md)
- SGLang notes: [docs/SGLANG_ON_DGX_SPARK.md](docs/SGLANG_ON_DGX_SPARK.md)
- Qwen notes: [docs/QWEN_ON_DGX_SPARK.md](docs/QWEN_ON_DGX_SPARK.md)
- vLLM AEON reproduction: [docs/VLLM_AEON_REPRODUCTION.md](docs/VLLM_AEON_REPRODUCTION.md)
- Before/after benchmark protocol: [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md)
- Baseline results: [docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md)
- Remediation matrix: [docs/REMEDIATION_MATRIX.md](docs/REMEDIATION_MATRIX.md)
- Spark smoke suite: [docs/SPARK_SMOKE_SUITE.md](docs/SPARK_SMOKE_SUITE.md)
- NVFP4 dependency map: [docs/NVFP4_DEPENDENCY_MAP.md](docs/NVFP4_DEPENDENCY_MAP.md)
- vLLM Gemma + NVFP4 KV direction (for Codex): [docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md](docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md)
- SGLang NVFP4 KV direction (for Codex): [docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md](docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md)
- llama.cpp direction (for Codex): [docs/CODEX_DIRECTION_LLAMACPP.md](docs/CODEX_DIRECTION_LLAMACPP.md)
- FlashInfer performance hypotheses: [docs/FLASHINFER_PERFORMANCE_HYPOTHESES.md](docs/FLASHINFER_PERFORMANCE_HYPOTHESES.md)
- Upstream latest release audit: [docs/UPSTREAM_LATEST_RELEASE_AUDIT.md](docs/UPSTREAM_LATEST_RELEASE_AUDIT.md)
- Runtime availability: [docs/RUNTIME_AVAILABILITY.md](docs/RUNTIME_AVAILABILITY.md)
- PyTorch sm121 support: [docs/PYTORCH_SM121_SUPPORT.md](docs/PYTORCH_SM121_SUPPORT.md)
- Failure annotations: [docs/FAILURE_ANNOTATIONS.md](docs/FAILURE_ANNOTATIONS.md)
- HF fallback telemetry: [docs/HF_FALLBACK_TELEMETRY.md](docs/HF_FALLBACK_TELEMETRY.md)

## First Commands

Before starting live work from this Windows workspace, check that the GB10 host is actually
usable over Tailscale and SSH:

```powershell
python scripts\gb10_host_access_probe.py `
  --host 100.113.98.11 `
  --ssh-user jethac `
  --output-json results\gb10_host_access_probe_RUN_ID.json `
  --output-md results\gb10_host_access_probe_RUN_ID.md
```

Proceed with live tasks only when `usable_for_live_work` is `true`.

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

To audit image/container target evidence without treating family/PTX evidence as native `sm_121a` proof:

```bash
python3 scripts/container_target_audit.py \
  --image-inspect results/RUN_ID_image_inspect.json \
  --container-versions results/RUN_ID_container_versions.json \
  --output results/RUN_ID_container_target_audit.json
```

To audit an NVFP4 checkpoint before Qwen/Gemma serving or GGUF conversion:

```bash
python3 scripts/nvfp4_checkpoint_audit.py \
  --model-dir /path/to/model \
  --output results/RUN_ID_nvfp4_checkpoint_audit.json \
  --strict
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

To audit serving row manifests before using them as evidence for a runtime claim:

```bash
python3 scripts/serving_manifest_audit.py \
  --output results/serving_manifest_audit_RUN_ID.json
```

To record the required Qwen speed lane against already-running servers:

```bash
python3 scripts/qwen_speed_lane.py \
  --input tasks/qwen_speed_lane_sample.jsonl \
  --campaign-id qwen_speed_lane_RUN_ID \
  --continue-on-error
```

To verify that the solution status, issue tracker, and required Qwen lane still cover the solution plan:

```bash
python3 scripts/solution_coverage_audit.py \
  --output results/solution_coverage_audit_RUN_ID.json
```

To audit whether AEON-derived SGLang/llama.cpp counterpart evidence has actual live artifacts yet:

```bash
python3 scripts/counterpart_evidence_audit.py \
  --output results/counterpart_evidence_audit_RUN_ID.json
```

To verify that every missing counterpart row has a concrete live task definition:

```bash
python3 scripts/counterpart_task_matrix.py \
  --tasks tasks/counterpart_evidence_tasks.jsonl \
  --audit results/counterpart_evidence_audit_RUN_ID.json \
  --output results/counterpart_task_matrix_RUN_ID.json
```

To verify the ordered live queue before using the GB10 host:

```bash
python3 scripts/live_task_queue_audit.py \
  --queue tasks/live_gb10_queue.jsonl \
  --output results/live_task_queue_audit_RUN_ID.json
```

To reproduce AEON-7's vLLM NVFP4+DFlash rows:

```bash
DOWNLOAD=1 DOCKER_PULL=1 RECORD=1 \
scripts/run_aeon_vllm_reproduction.sh gemma26-dflash aeon_gemma26_dflash_RUN_ID
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
- Qwen speed/capacity rows exist for the runtime paths being claimed
- SGLang and LiteRT-LM have explicit go/no-go decisions
- llama.cpp GGUF throughput and lm-eval accuracy paths are separated and validated
- HF fallback failures include memory/process telemetry
- NVFP4 is either validated on Spark or clearly marked unsafe
- benchmarks record backend, selected kernels, CUDA architecture, memory state, and exact commands

## Current State

This repo starts from an initial personal Gemma 4 benchmark run on the machine. That run showed useful vLLM safetensors results for several model rows, but also exposed ecosystem problems around `sm_121` packaging, vLLM model support, HF fallback reliability, GGUF lm-eval logprobs, long benchmark design, and missing observability. A first NVIDIA SGLang 26.05 container smoke now passes on GB10, but it still needs Gemma and NVFP4 validation. LiteRT-LM now has a proven E2B CPU path and benchmarkable GPU path, but GPU chat exits with `returncode=-11` after producing output.

The GitHub Issues are the source of truth for active work. If we need source changes to upstream libraries, the change goes through a `jethac` fork, a submodule under `third_party/`, and an issue-named worktree.
