# LiteRT-LM On Spark

Status: usable CPU path, benchmark-only GPU path, not blessed for GPU chat.

LiteRT-LM is tracked because it may be relevant for Gemma and local-agent prototyping, especially if its Gemma/MTP path is strong on desktop Linux.

Source:

- https://github.com/google-ai-edge/LiteRT-LM

Current assessment, 2026-06-07:

- `litert-lm==0.13.1` and `litert-lm-api==0.13.1` install cleanly in a Python 3.12 Linux `aarch64` venv on Spark.
- CPU generation is proven with `litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm`; the prompt smoke returned `spark-ok`.
- The built-in CPU and GPU benchmark commands run for the same E2B model.
- GPU chat is not clean yet. It prints `spark-ok` and then exits with `returncode=-11`.
- GPU appears to use LiteRT GPU/Vulkan/OpenCL plumbing, not a CUDA-native tensor-core serving path. The logs include `Failed to load OpenCL library with dlopen: libOpenCL.so` followed by ICD-loader fallback.
- `.litertlm` is the required model container format; raw safetensors and GGUF are not first-class inputs.
- Treat E2B/E4B preconverted Gemma repos as first smoke targets. Do not start with 12B.
- Do not set `--max-num-tokens 512` for this E2B chat smoke. That setting failed with `DYNAMIC_UPDATE_SLICE`, `Failed to allocate tensors`, and `Failed to invoke the compiled model`, while the CLI still returned exit code 0.
- Noninteractive wrappers must close stdin or provide piped input. `litert-lm run` reads all non-TTY stdin before loading the model; leaving an open pipe causes an apparent hang.

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
  --backend=cpu \
  --temperature=0 \
  --prompt="Reply with exactly this text: spark-ok" \
  gemma-4-E2B-it.litertlm
```

Observed 2026-06-07 on `thinkstationpgx-00b4`: `spark-ok`.

Do not add `--max-num-tokens 512` to this smoke. LiteRT-LM selected `Max number of tokens: 4096` in the successful benchmark path.

## GPU Smoke

Run this only after CPU works:

```bash
nvidia-smi

litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  --backend=gpu \
  --temperature=0 \
  --prompt="Reply with exactly this text: spark-ok" \
  gemma-4-E2B-it.litertlm
```

Observed 2026-06-07 after adding `jethac` to `render` and `video`: the command printed `spark-ok` but exited with `returncode=-11`.

The GPU benchmark command is healthier:

```bash
litert-lm benchmark \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  --backend=gpu \
  --prefill-tokens=256 \
  --decode-tokens=64 \
  gemma-4-E2B-it.litertlm
```

Observed GPU benchmark: `1773.07` prefill tok/s, `43.70` decode tok/s, init `2.5860` s, TTFT `0.1673` s.

Observed CPU benchmark with the same token counts: `125.77` prefill tok/s, `43.57` decode tok/s, init `0.3235` s, TTFT `2.0584` s.

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

Do not treat a successful Python import as sufficient. A go decision requires CPU generation, GPU generation, backend evidence, CPU-vs-GPU timing, and a clean exit from the serving/chat path.

## Fork Rule

If LiteRT-LM needs source changes, fork `google-ai-edge/LiteRT-LM` to `jethac/LiteRT-LM`, add it as `third_party/LiteRT-LM`, and do the patch in a worktree named for the GitHub Issue.
