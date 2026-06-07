# LiteRT-LM On Spark

Status: research track, not blessed.

LiteRT-LM is tracked because it may be relevant for Gemma and local-agent prototyping, especially if its Gemma/MTP path is strong on desktop Linux.

Source:

- https://github.com/google-ai-edge/LiteRT-LM

Current assessment, 2026-06-07:

- `litert-lm` and `litert-lm-api` 0.13.1 are the first package targets.
- `litert-lm-api` publishes a Linux `aarch64` wheel, which makes install viability plausible on DGX Spark.
- CPU is the first backend to prove.
- GPU is unproven for GB10 and appears more likely to be LiteRT GPU via WebGPU/Vulkan than CUDA-native tensor-core code.
- `.litertlm` is the required model container format; raw safetensors and GGUF are not first-class inputs.
- Treat E2B/E4B preconverted Gemma repos as first smoke targets. Do not start with 12B.

## Questions

- Does LiteRT-LM build cleanly on Linux `aarch64` / DGX OS?
- Which backend does it use on GB10: CPU, CUDA GPU, LiteRT GPU, or something else?
- Which model formats are required?
- Does it support the Gemma 4 models we care about?
- Does it expose a CLI/server path suitable for local-agent workflows?
- Does it outperform llama.cpp or vLLM for small Gemma text generation or MTP?

## Preflight

```bash
python3 scripts/spark_doctor.py --json > results/spark_doctor_before_litert_lm.json
python3 scripts/spark_doctor.py > results/spark_doctor_before_litert_lm.md
```

## Clean Venv Install Smoke

Use a separate venv, not the benchmark venv:

```bash
python3 -m venv ~/venvs/litert-lm-smoke
source ~/venvs/litert-lm-smoke/bin/activate
python -m pip install -U pip
python -m pip install --only-binary=:all: "litert-lm==0.13.1"

python - <<'PY'
import platform
import importlib.metadata as md
import litert_lm

print(platform.platform(), platform.machine())
print("litert-lm", md.version("litert-lm"))
print("litert-lm-api", md.version("litert-lm-api"))
PY
```

## CPU Smoke

```bash
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --backend=cpu \
  --prompt="What is the capital of France?"
```

## GPU Smoke

Run this only after CPU works:

```bash
vulkaninfo --summary || true
nvidia-smi

litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --backend=gpu \
  --enable-speculative-decoding=true \
  --prompt="Write one sentence about DGX Spark."
```

## Server Smoke

```bash
litert-lm import \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  gemma4-e2b

litert-lm serve --host 127.0.0.1 --port 9379 --verbose

curl http://localhost:9379/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e2b,gpu","messages":[{"role":"user","content":"Hello"}]}'
```

## Acceptance Test

- documented build/install command
- model conversion or download command
- one generation smoke result
- backend evidence
- throughput and latency
- comparison against at least one existing path, usually llama.cpp or vLLM
- logs or library evidence showing whether the GPU path is Vulkan/WebGPU, CUDA, or another backend

Do not treat a successful Python import as sufficient. A go decision requires CPU generation, GPU generation, backend evidence, and CPU-vs-GPU timing.

## Fork Rule

If LiteRT-LM needs source changes, fork `google-ai-edge/LiteRT-LM` to `jethac/LiteRT-LM`, add it as `third_party/LiteRT-LM`, and do the patch in a worktree named for the GitHub Issue.
