# Campaign Log

## 2026-06-07

- Created this public record repository.
- Imported the first Gemma 4 benchmark reports.
- Added the initial diagnosis and solution plan.
- Added first-pass tools:
  - `scripts/spark_doctor.py`
  - `scripts/gguf_logprobs_probe.py`
  - `scripts/openai_chat_smoke.py`
- Added SGLang as a first-class runtime workstream after reviewing `hikarioyama/sglang-nvfp4-kv-sm120`.
- Added practical llama.cpp serving as a first-class runtime workstream and LiteRT-LM as optional side-runtime coverage.
- Added the upstream fork/submodule/worktree policy for patched libraries.
- Added before/after GB10 benchmark protocol for measuring campaign impact.
- Added NVFP4 dependency map from the FlashInfer/vLLM/SGLang subagent investigation.
- Added a remediation matrix clarifying that the FlashInfer SM121 `mm_fp4` patch is only one workstream; remaining work includes packaging, vLLM/SGLang integration, Gemma 4 12B, NVFP4 KV, llama.cpp/lm-eval, optional LiteRT-LM, HF fallback, observability, benchmark proof, and upstream coordination.
- Added a compact OpenAI-compatible serving benchmark harness and captured the first vLLM before row.
- Added root-readable runtime process probe evidence for the live vLLM server.
- Added `scripts/spark_smoke_suite.py` as the compact before/after orchestrator. Core tracks are vLLM, SGLang, llama.cpp, HF fallback telemetry, MTP/spec decode, and NVFP4; LiteRT-LM is opt-in.
- Reproduced the llama.cpp GGUF logprobs incompatibility with a temporary `llama-server` on port `18081`; server was stopped after the probe.
- Captured runtime availability matrix showing vLLM/FlashInfer/PyTorch present, SGLang and LiteRT-LM absent, Docker available, and llama.cpp built but not on `PATH`.
- Started tracking work through GitHub Issues.
- Ran `spark_doctor` on `thinkstationpgx-00b4` using `/home/jethac/gemma4-evals/.venv/bin/python`.
  - GPU: `NVIDIA GB10`
  - compute capability: `12.1` / `sm_121`
  - host: `aarch64`
  - driver: `580.159.03`
  - CUDA runtime reported by `nvidia-smi`: `13.0`
  - `nvcc`: CUDA `13.0`, found through `/usr/local/cuda`
  - `cuobjdump`: CUDA `13.0`, found through `/usr/local/cuda`
  - PyTorch: `2.11.0+cu130`
  - vLLM: `0.22.1`
  - FlashInfer: `0.6.11.post2`
  - PyTorch arch list: `sm_80`, `sm_90`, `sm_100`, `sm_110`, `sm_120`; no explicit `sm_121`
  - snapshot: `results/spark_doctor_20260607T110833Z.md`
- Ran `cuda_so_audit` against vLLM and FlashInfer package roots in the benchmark venv.
  - inspected objects: 14
  - objects with explicit `sm_121`: 0
  - objects with `sm_120`: 3
  - vLLM FA2 extension: `sm_80`
  - vLLM FA3 extension: `sm_90a`
  - vLLM FlashMLA extensions: `sm_100`, `sm_90a`
  - snapshot: `results/cuda_so_audit_vllm_flashinfer_20260607T111023Z.json`
- Proved llama.cpp `b9536` as a practical Gemma 4 26B Q4_0 serving path with `--reasoning off`.
- Evaluated LiteRT-LM `0.13.1` on Linux `aarch64`: CPU chat works for Gemma 4 E2B, CPU/GPU benchmark commands work, GPU chat still exits `returncode=-11` after producing `spark-ok`.
- Ran SGLang 26.05 model-shaped FlashInfer `mm_fp4` proxy microbenchmarks against installed and patched source/JIT FlashInfer.
  - patched source selected `b12x`, `cutlass`, `cudnn` and built an SM121a-targeted FP4 GEMM path.
  - patched dense-decode proxies were mixed; patched MoE-shaped proxies were slower on all tested shapes.
  - conclusion: the FlashInfer SM121 predicate patch remains dispatch enablement, not a proven speedup.

## First Benchmark Campaign Summary

The initial personal Gemma 4 benchmark run was run on `thinkstationpgx-00b4` in `/home/jethac/gemma4-evals`.

At the last local sync:

- smoke rows: 152 complete
- smoke `ok`: 21
- smoke `eval_failed`: 11
- smoke `loader_failed`: 120
- full eval records: 70
- full eval `ok`: 65
- full eval failed: 5
- throughput rows observed: 2
- MTP rows observed: 2

That personal benchmark run was still in full accuracy when monitoring stopped. It was not killed.
