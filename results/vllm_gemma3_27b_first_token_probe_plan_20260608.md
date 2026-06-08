# vLLM Gemma 3 27B NVFP4 First-Token Probe Plan, 2026-06-08

Purpose: localize the Gemma 3 27B NVFP4-KV corruption observed in
`results/vllm_gemma3_27b_rung1_nvfp4_20260608T1924JST.md`.

Why this exists:

- The NVFP4 row routes correctly through FlashInfer FA2 and reports the expected `1.777x`
  KV capacity gain.
- Output quality is red: strict `spark-ok` smoke and benchmark generations are nonsensical.
- The old packet stopped after the failed manifest because `record_openai_serving_row.py`
  returned nonzero under `set -e`, so diagnostic probes after the manifest were skipped.

New tool:

- `scripts/openai_first_token_probe.py`

Probe behavior:

- Sends deterministic non-streaming Chat Completions requests with:
  - `temperature=0`
  - `max_tokens=1`
  - `logprobs=true`
  - `top_logprobs=20`
- Default cases:
  - `exact_spark_ok`
  - `simple_math`
  - `short_decode`
- Records:
  - generated first token/text
  - first-token logprob
  - top-logprob candidates
  - script/non-ASCII summary for generated text and candidates
- Offline compare mode reports:
  - fp8 first token
  - NVFP4 first token
  - top-logprob overlap ratio

Packet change:

`scripts/prep_vllm_gemma3_27b_rung1.sh` now:

- keeps using `--no-deps` for vLLM editable installs
- copies the ABI-matched FA2 extension from `/opt/jethac-vllm`
- supports `RUN_FP8` and `RUN_NVFP4`
- runs `openai_first_token_probe.py` for both fp8 and NVFP4 rows
- runs the first-token compare after the NVFP4 row
- continues after red smoke/manifest commands so diagnostic artifacts are still written

Fresh live command shape:

```bash
STAMP=20260608TNNNNJST RUN_FP8=1 RUN_NVFP4=1 \
  scripts/prep_vllm_gemma3_27b_rung1.sh \
  /home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/vllm \
  /home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/flashinfer \
  /home/jethac/.cache/huggingface \
  results \
  > docs/results/vllm_gemma3_27b_rung1_20260608TNNNNJST_command_packet.sh
```

Then on the GB10 host, with `HF_TOKEN` loaded from the existing token file:

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608
export HF_TOKEN="$(tr -d "\r" < /home/jethac/.cache/huggingface/token)"
bash docs/results/vllm_gemma3_27b_rung1_20260608TNNNNJST_command_packet.sh \
  > results/vllm_gemma3_27b_rung1_20260608TNNNNJST_driver.log 2>&1
```

Expected new artifacts:

- `results/vllm_gemma3_27b_rung1_20260608TNNNNJST_fp8_flashinfer_first_token.json`
- `results/vllm_gemma3_27b_rung1_20260608TNNNNJST_nvfp4_kv_flashinfer_first_token.json`
- `results/vllm_gemma3_27b_rung1_20260608TNNNNJST_first_token_compare.json`

Interpretation gates:

- If the first generated token already diverges on `exact_spark_ok` or `simple_math`, the
  bug is in prefill/KV attention or logits before long decode compounding.
- If first token matches but later text corrupts, focus on decode KV append/read behavior,
  CUDA graph replay, or sampling state across steps.
- If top-logprob candidate sets overlap heavily but greedy token rank flips, inspect scale
  precision/calibration bias. If candidate sets are disjoint, inspect KV layout/stride or
  attention output corruption.
