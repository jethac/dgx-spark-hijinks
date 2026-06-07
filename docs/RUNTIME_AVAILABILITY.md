# Runtime Availability

Status: current-state matrix for candidate runtimes.

Tracked by:

- vLLM: https://github.com/jethac/dgx-spark-hijinks/issues/6
- SGLang: https://github.com/jethac/dgx-spark-hijinks/issues/14
- LiteRT-LM: https://github.com/jethac/dgx-spark-hijinks/issues/16
- llama.cpp: https://github.com/jethac/dgx-spark-hijinks/issues/17

## Current Snapshot

Artifact:

- `results/runtime_availability_20260607T1155Z.json`

Summary:

| runtime | current state |
|---|---|
| vLLM | installed in benchmark venv; live server running |
| FlashInfer | installed in benchmark venv |
| PyTorch | installed in benchmark venv as `2.11.0+cu130` |
| SGLang | not installed in benchmark venv; no command found |
| LiteRT-LM | not installed in benchmark venv; no command found |
| TensorFlow/LiteRT Python modules | not installed in benchmark venv |
| llama.cpp | built under `/home/jethac/src`, but not on shell `PATH` |
| Ollama | no command found |
| Docker | available |

Docker images currently present include:

- `gemma4-vllm:v0.22.1-pip`
- `gemma4-vllm:tf-main`
- `vllm/vllm-openai:latest-cu130`
- `vllm/vllm-openai:cu130-nightly-aarch64`
- `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04`

## Next Runtime Actions

1. SGLang: run the container smoke with `nvcr.io/nvidia/sglang:26.05-py3`; fall back to `lmsysorg/sglang:latest-cu130-runtime` if CUDA 13.2 / driver compatibility blocks it.
2. LiteRT-LM: install `litert-lm==0.13.1` in a clean venv, prove CPU E2B generation, then test whether `--backend=gpu` uses a useful GB10 path.
3. llama.cpp: pin the build/commit as a practical serving path; keep GGUF lm-eval accuracy blocked until logprobs compatibility is fixed.
4. vLLM: use the current baseline as the before row for future patched/container comparisons.
