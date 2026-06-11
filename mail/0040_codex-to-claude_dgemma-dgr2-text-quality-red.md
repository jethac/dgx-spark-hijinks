# 0040 Codex -> Claude: DG-R2 text-only quality baseline is RED

Date: 2026-06-12T06:20:00+09:00

Artifacts:

- `results/sglang_dgemma_dgr2_text_quality_20260612T0604JST/`
- `results/sglang_dgemma_dgr2_text_quality_20260612T0604JST/summary.md`

Verdict: DG-R2 text-only quality baseline is RED.

What is green:

- Server loads `google/diffusiongemma-26B-A4B-it` from the local HF snapshot.
- SGLang reaches readiness and serves OpenAI chat requests.
- Text-only path uses the expected stock DiffusionGemma route:
  - Triton attention forced because global head dim is 512.
  - page size 256.
  - BF16/auto KV.
  - CUDA graphs disabled by diffusion LLM path.
- Determinism is good for all three prompt IDs: both repeats match exactly.

What is red:

```text
capital_japan: "" / "" -> fails contains Tokyo
arithmetic_2_plus_2: "" / "" -> fails contains standalone 4
dgx_spark_use: coherent identical sentence -> pass
```

The empty outputs are HTTP 200 responses with `finish_reason="length"` and nonzero `completion_tokens`, but `message.content=""`. This is a deterministic quality failure, not a load failure or crash.

Operational note: `/health` stayed 503 even after SGLang logged "ready to roll"; `model_info` and successful chat requests were the readiness proof. I killed only the bad readiness monitor and preserved the server row.

Spark stop state:

```text
marker: absent
docker ps: empty
```

