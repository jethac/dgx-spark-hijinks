# LiteRT-LM On Spark

Status: research track, not blessed.

LiteRT-LM is tracked because it may be relevant for Gemma and local-agent prototyping, especially if its Gemma/MTP path is strong on desktop Linux.

Source:

- https://github.com/google-ai-edge/LiteRT-LM

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

## Acceptance Test

- documented build/install command
- model conversion or download command
- one generation smoke result
- backend evidence
- throughput and latency
- comparison against at least one existing path, usually llama.cpp or vLLM

## Fork Rule

If LiteRT-LM needs source changes, fork `google-ai-edge/LiteRT-LM` to `jethac/LiteRT-LM`, add it as `third_party/LiteRT-LM`, and do the patch in a worktree named for the GitHub Issue.

