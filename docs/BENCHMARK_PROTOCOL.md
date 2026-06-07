# Before/After GB10 Benchmark Protocol

Status: draft.

Goal: prove that this campaign makes the GB10 perform closer to what the silicon is capable of.

Compatibility is not enough. A 900k JPY machine should not merely start a model. It should run the best available stack for the workload, with measured speed, memory behavior, and output quality.

Tracked by:

- https://github.com/jethac/dgx-spark-hijinks/issues/19

## Rules

- Every improvement claim needs a before row and an after row.
- Before and after rows must use the same machine, model, prompt set, quantization, context length, batch/concurrency, and output length.
- Capture `spark_doctor` before each run.
- Capture CUDA shared-object/JIT evidence when testing kernel changes.
- Capture runtime process evidence with `scripts/runtime_process_probe.py` for serving baselines.
- Separate cold start, first-token, warm decode, and long-context behavior.
- Keep backend families separate: vLLM, SGLang, llama.cpp/Ollama, LiteRT-LM, and HF fallback.
- Do not compare paper accuracy across backends unless the scoring path is validated.

## Required Columns

| field | why |
|---|---|
| run id | joins logs/results |
| phase | before or after |
| backend | vLLM, SGLang, llama.cpp, LiteRT-LM, HF |
| repo commit/container | reproducibility |
| model id/revision | model control |
| quantization | performance/quality control |
| KV cache dtype | fp8 vs NVFP4 impact |
| attention backend | kernel path |
| CUDA graph mode | performance control |
| prompt tokens | throughput denominator |
| generated tokens | throughput denominator |
| TTFT | interactive feel |
| decode tok/s | steady-state generation |
| prefill tok/s | prompt-processing speed |
| memory state | unified-memory pressure |
| quality check | avoids fast garbage |
| `spark_doctor` path | environment evidence |
| `.so`/JIT audit path | kernel evidence |

## Baseline Suite

Keep this short enough to run repeatedly on one unit.

1. Environment
   - `spark_doctor`
   - CUDA `.so` audit for the active backend

2. Serving smoke
   - one short deterministic OpenAI-compatible chat request
   - one medium generation request
   - use `scripts/openai_serving_benchmark.py` when the backend exposes an OpenAI-compatible API

3. Throughput
   - short prompt / short output
   - long prompt / short output
   - short prompt / long output

Example:

```bash
python3 scripts/openai_serving_benchmark.py \
  --url http://127.0.0.1:8000 \
  --backend vllm \
  --phase before \
  --run-id vllm-gemma4-e4b-w4a16-before-001 \
  --output results/vllm_gemma4_e4b_w4a16_before_001.json
```

4. Quality sanity
   - deterministic prompt expected to produce stable text
   - fp8/bf16 reference for any NVFP4 test
   - no NaN/inf/empty/zero-output path

5. Kernel microbenchmarks
   - use `scripts/flashinfer_mm_fp4_microbench.py` for FlashInfer NVFP4 `mm_fp4` dispatch checks
   - treat kernel microbenchmarks as diagnostic evidence, not serving throughput
   - follow with model-shaped and serving before/after rows before claiming user-visible speedups

6. Optional long checks
   - HellaSwag and other long lm-eval tasks run as separate campaigns
   - RULER/needle-style checks for long-context KV changes

## Initial Before State

The first imported benchmark campaign is a partial before-state artifact, not a complete protocol run.

Known before-state evidence:

- vLLM `0.22.1`
- PyTorch `2.11.0+cu130`
- FlashInfer `0.6.11.post2`
- GB10 reports `sm_121`
- inspected vLLM/FlashInfer objects had no explicit `sm_121` SASS
- vLLM safetensors rows worked for several E2B/E4B/26B-A4B rows
- GGUF lm-eval accuracy path was blocked by logprobs/API mismatch
- HF fallback had `returncode=-9` failures
- HellaSwag dominated wall time

The next step is to turn this into a compact repeatable before/after suite.

First compact before row:

- `docs/BASELINE_RESULTS.md`
- `results/vllm_gemma4_e4b_w4a16_before_compact_20260607T1126Z.json`
- `results/spark_doctor_before_vllm_gemma4_e4b_w4a16_20260607T1126Z.md`
- `results/runtime_probe_vllm_gemma4_e4b_w4a16_root_20260607T1136Z.json`
