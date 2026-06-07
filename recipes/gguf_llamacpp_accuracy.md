# GGUF llama.cpp Accuracy Recipe

Status: blocked pending logprobs compatibility.

The first campaign showed that llama.cpp throughput can work while lm-eval accuracy fails. Do not treat llama-bench throughput as proof that paper-comparable accuracy is available.

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

