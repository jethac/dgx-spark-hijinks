# Single-Spark SGLang Recipe

Status: draft, not blessed.

SGLang is tracked as a first-class Spark runtime alongside vLLM, llama.cpp, and LiteRT-LM.

## Why It Matters

The `hikarioyama/sglang-nvfp4-kv-sm120` repo shows an SGLang NVFP4 KV path for RTX Blackwell-class systems:

- `--kv-cache-dtype fp4_e2m1`
- FlashInfer FA2 kernel patches
- native FP4 KV memory pool
- hybrid-SWA wiring
- per-layer global-scale auto-calibration before CUDA graph capture

That is relevant to Spark because it attacks the same missing NVFP4 plumbing problem we saw on the vLLM side. It still needs `sm_121` validation on our GB10 unit.

## Preflight

```bash
python3 scripts/spark_doctor.py --json > results/spark_doctor_before_sglang.json
python3 scripts/spark_doctor.py > results/spark_doctor_before_sglang.md
```

## Smoke

Start an SGLang OpenAI-compatible server using the candidate stack, then run:

```bash
python3 scripts/openai_chat_smoke.py \
  --url http://127.0.0.1:30000 \
  --model MODEL_NAME \
  --output results/sglang_chat_smoke.json
```

## Result Requirements

A blessed SGLang result must record:

- SGLang version or container
- model id and revision
- quantization mode
- KV cache dtype
- attention backend
- CUDA graph enabled/disabled
- Spark doctor snapshot
- output quality check against fp8 KV or another reference path
- prompt/generation throughput and memory state

## Current Rule

Do not bless SGLang NVFP4 KV on Spark until it passes a single-Spark smoke test and a quality check. For small models, prefer fp8 KV unless NVFP4 quality is proven on that model.

## Fork Rule

If SGLang needs source changes, fork `sgl-project/sglang` to `jethac/sglang`, add it as `third_party/sglang`, and do the patch in a worktree named for the GitHub Issue.
