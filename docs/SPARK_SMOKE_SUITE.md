# Spark Smoke Suite

Status: orchestrator added; Spark harness validation passed with explicit skips; full core run pending.

The smoke suite is the short, repeatable check that should run before and after any stack change. It does not launch heavyweight servers. It tests what is already running or explicitly supplied, records skipped prerequisites, and fails when required campaign evidence is missing.

## Why LiteRT-LM Is Opt-In

LiteRT-LM is useful for small Gemma/local-agent side work, but it is not the normal Spark performance path people reach for. The default suite therefore treats LiteRT-LM as optional.

The default core tracks are:

- `spark_doctor` environment evidence
- vLLM OpenAI-compatible smoke
- SGLang OpenAI-compatible smoke
- llama.cpp OpenAI-compatible smoke
- HF fallback command under telemetry
- MTP/spec-decode command
- NVFP4/FlashInfer probe

If a core track is not configured, the suite records that as a required missing-evidence failure unless the runner explicitly passes the corresponding `--skip-*` flag.

## Command Shape

Run on the Spark from the repo checkout:

```bash
python3 scripts/spark_smoke_suite.py \
  --run-id spark-smoke-before-YYYYMMDDTHHMMSSZ \
  --vllm-url http://127.0.0.1:8000 \
  --vllm-model gemma4-26b-a4b-it \
  --llamacpp-url http://127.0.0.1:18082 \
  --llamacpp-model gemma4-26b-q4_0-gguf \
  --sglang-url http://127.0.0.1:30000 \
  --sglang-model Qwen/Qwen2.5-1.5B-Instruct \
  --hf-command "python3 path/to/tiny_hf_fallback_probe.py" \
  --mtp-command "python3 path/to/tiny_mtp_probe.py" \
  --mtp-model gemma4-12b-mtp \
  --nvfp4-preset smoke \
  --output results/spark_smoke_before.json
```

For a deliberately partial local run, skip missing tracks explicitly:

```bash
python3 scripts/spark_smoke_suite.py \
  --run-id spark-smoke-partial \
  --skip-vllm \
  --skip-sglang \
  --skip-llamacpp \
  --skip-hf \
  --skip-mtp \
  --skip-nvfp4 \
  --output results/spark_smoke_partial.json
```

LiteRT-LM is included only when requested:

```bash
python3 scripts/spark_smoke_suite.py \
  --include-litert-lm \
  --skip-vllm --skip-sglang --skip-llamacpp --skip-hf --skip-mtp --skip-nvfp4
```

## NVFP4 Defaults

If `--nvfp4-command` is not supplied and `--skip-nvfp4` is not set, the suite runs:

```bash
python3 scripts/flashinfer_mm_fp4_microbench.py \
  --phase exploratory \
  --preset smoke \
  --iterations 10
```

That is a kernel-level dispatch/correctness probe, not a serving benchmark. Use `--nvfp4-command` for a stronger fp8-vs-NVFP4 serving probe once the vLLM/SGLang path exists.

## Output

The suite writes a `spark-smoke-suite/v1` JSON report with:

- command, return code, timeout state, and artifact path for every configured step
- required missing-evidence rows for core tracks that were not configured
- skipped rows only when explicitly skipped or for opt-in LiteRT-LM
- a `spark_doctor` artifact for environment evidence
- telemetry JSON for HF fallback when `--hf-command` is supplied
- telemetry JSON for MTP/spec-decode when `--mtp-command` is supplied

The suite is considered green only when every configured step succeeds and no required core track is missing.

## Tiny HF And MTP Commands

Use a tiny HF fallback smoke when validating the suite itself:

```bash
python3 scripts/spark_smoke_suite.py \
  --run-id spark-smoke-hf-tiny-$(date -u +%Y%m%dT%H%M%SZ) \
  --timeout-s 120 \
  --skip-vllm --skip-sglang --skip-llamacpp --skip-mtp --skip-nvfp4 \
  --hf-command "env CUDA_VISIBLE_DEVICES= lm_eval --model hf --model_args pretrained=sshleifer/tiny-gpt2,device=cpu,dtype=float32 --tasks boolq --limit 1 --batch_size 1 --num_fewshot 0"
```

Use a capped MTP smoke against the known llama.cpp MTP eval checkout:

```bash
python3 scripts/spark_smoke_suite.py \
  --run-id spark-smoke-mtp-tiny-$(date -u +%Y%m%dT%H%M%SZ) \
  --timeout-s 180 \
  --skip-vllm --skip-sglang --skip-llamacpp --skip-hf --skip-nvfp4 \
  --mtp-model gemma4-12b-ud-iq2_m \
  --mtp-command "env GEMMA4_EVAL_ROOT=/home/jethac/gemma4-evals /home/jethac/gemma4-evals/.venv/bin/python /home/jethac/gemma4-evals/run_mtp_speed.py --only-row-id mtp-12b-ud-iq2_m-q8_0-gemma-4-12b-it-ud-iq2_m.gguf --max-tokens 8 --timeout 180 --large-timeout 180"
```

## Current Evidence

Harness validation on `thinkstationpgx-00b4`:

- suite artifact: `results/spark_smoke_suite_harness_20260607T153000Z.json`
- `spark_doctor` artifact: `results/spark-smoke-suite-harness-20260607T153000Z_spark_doctor.json`
- result: `ok=true`
- scope: harness validation only; vLLM, SGLang, llama.cpp, HF, MTP, and NVFP4 were explicitly skipped

The full core suite still needs running with actual configured services and commands.
