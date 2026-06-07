# Single-Spark vLLM Recipe

Status: draft, not blessed.

Use this recipe for one DGX Spark / ThinkStation PGX unit. Do not use this for multi-Spark runs.

## Rules

- Target GB10 as `sm_121`.
- Prefer a vLLM container or wheel explicitly validated for Spark.
- Keep CUDA graphs enabled unless a specific bug requires eager mode.
- Keep concurrency modest; Spark is a local/small-batch inference machine, not an H100 replacement.
- Record `spark_doctor` output before every run.

## Preflight

```bash
python3 scripts/spark_doctor.py --json > results/spark_doctor_before_vllm.json
python3 scripts/spark_doctor.py > results/spark_doctor_before_vllm.md
```

## Smoke

Start the vLLM server using the candidate stack, then run:

```bash
python3 scripts/openai_chat_smoke.py \
  --url http://127.0.0.1:8000 \
  --model MODEL_NAME \
  --output results/vllm_chat_smoke.json
```

## Result Requirements

A blessed result must record:

- exact container or wheel versions
- model id and revision
- CUDA driver/runtime
- PyTorch version
- vLLM version
- selected attention backend if visible
- CUDA graphs enabled/disabled
- prompt and generated token counts
- wall time and tokens/sec

