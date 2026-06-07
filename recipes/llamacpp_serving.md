# llama.cpp Serving On Spark

Status: partially observed, not fully blessed.

The first benchmark campaign showed llama.cpp throughput working for at least one GGUF row. It also showed that GGUF lm-eval accuracy was blocked by a logprobs/API mismatch.

Those are separate problems.

## Serving Track

Use llama.cpp/Ollama as a practical local serving path while vLLM and SGLang mature.

Acceptance requires:

- exact llama.cpp commit
- build flags and CUDA architecture settings
- `spark_doctor` snapshot
- `llama-bench` throughput
- OpenAI-compatible smoke result if server mode is used
- model, quantization, context length, prompt length, and generated token count

## Accuracy Track

Do not claim paper-comparable GGUF accuracy until:

```bash
python3 scripts/gguf_logprobs_probe.py --url http://127.0.0.1:8080
```

passes against the target llama.cpp server.

## Fork Rule

If llama.cpp needs source changes, fork `ggml-org/llama.cpp` to `jethac/llama.cpp`, add it as `third_party/llama.cpp`, and do the patch in a worktree named for the GitHub Issue.

