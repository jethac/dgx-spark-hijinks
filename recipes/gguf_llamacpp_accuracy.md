# GGUF llama.cpp Accuracy Recipe

Status: blocked pending logprobs compatibility.

The initial personal benchmark run showed that llama.cpp throughput can work while lm-eval accuracy fails. Do not treat llama-bench throughput as proof that paper-comparable accuracy is available.

## Preflight

Start a llama.cpp OpenAI-compatible server for the target GGUF.

Then run:

```bash
python3 scripts/gguf_logprobs_probe.py \
  --url http://127.0.0.1:8080 \
  --output results/gguf_logprobs_probe.json
```

## Pass Criteria

The response must include token logprobs in a shape that can score an echoed prompt/continuation pair.

If this probe fails, do not run full lm-eval GGUF accuracy. Fix the adapter or pin a compatible llama.cpp server first.

## Current Result

The probe failed against stock llama.cpp `b9536` on 2026-06-07.

Artifact:

- `results/gguf_logprobs_probe_llamacpp_b9536_20260607T1145Z.json`

The server returned generated-token logprobs under `choices[0].logprobs.content`, but did not return `tokens` or `token_logprobs`. That is insufficient for the current lm-eval GGUF loglikelihood path.
