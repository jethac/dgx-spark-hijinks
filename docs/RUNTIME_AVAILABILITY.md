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
- `results/spark_doctor_tailnet_reconnect_20260608T074035JST.json`
- `results/gb10_host_access_probe_20260609.json`

Summary:

| runtime | current state |
|---|---|
| vLLM | installed in benchmark venv; AEON Gemma and Qwen containers have local evidence; no live server is assumed |
| FlashInfer | installed in benchmark venv |
| PyTorch | installed in benchmark venv as `2.11.0+cu130` |
| SGLang | not installed in benchmark venv; NVIDIA `nvcr.io/nvidia/sglang:26.05-py3` container smoke passed |
| LiteRT-LM | clean venv install works with `litert-lm==0.13.1`; CPU E2B generation works; GPU benchmark works; GPU chat exits `-11` after output |
| TensorFlow/LiteRT Python modules | not installed in benchmark venv |
| llama.cpp | built under `/home/jethac/src`, but not on shell `PATH` |
| Ollama | no command found |
| Docker | available |

2026-06-08 reconnect note:

- reachable hostname: `thinkstationpgx-00b4.tail740c8d.ts.net`
- reachable Tailscale IP: `100.113.98.11`
- stale/unreachable LAN IP from this client: `192.168.68.112`
- live doctor result: Linux `aarch64`, `NVIDIA GB10`, compute capability `12.1`, driver `580.159.03`, CUDA `13.0`
- caveat: the reconnect doctor used system Python, where Torch is unavailable, so that artifact does not record SM count

2026-06-09 control-path note:

- the same Tailscale IP is visible in the control plane, but live work is not currently usable from this workspace
- `results/gb10_host_access_probe_20260609.json` records `usable_for_live_work=false`
- Tailscale ping, TCP/22, and SSH time out; the peer is active via relay `tok` with `rx_bytes=0`

Docker images currently present include:

- `gemma4-vllm:v0.22.1-pip`
- `gemma4-vllm:tf-main`
- `vllm/vllm-openai:latest-cu130`
- `vllm/vllm-openai:cu130-nightly-aarch64`
- `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04`

## Next Runtime Actions

1. SGLang: fix or escalate the Gemma 4 model-path blocker, then test fp8 vs NVFP4 quality/performance.
2. LiteRT-LM: keep CPU as a complement path, investigate GPU chat `SIGSEGV`, and decide whether the GPU benchmark path is useful enough for local-agent microtasks.
3. llama.cpp: pin the build/commit as a practical serving path; keep GGUF lm-eval accuracy blocked until logprobs compatibility is fixed.
4. vLLM: use the current baseline as the before row for future patched/container comparisons.
